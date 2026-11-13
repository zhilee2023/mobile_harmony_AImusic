"""Matched FFN ONNX layout probes, using explicitly captured real parameters.

This exports an isolated FFN, not a complete DiT or a deployable model bundle.
Decoded MNN W8 parameters are not represented as an original FP32 checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def to_btc(values, shape, *, batch, frames, channels):
    """Decode only known logical host layouts; element count alone is insufficient."""
    import numpy as np
    shape = tuple(shape)
    data = np.asarray(values, dtype=np.float32)
    if data.size != batch * frames * channels:
        raise ValueError('Captured activation element count mismatch')
    if shape in ((batch * frames, channels, 1, 1), (batch * frames, channels), (batch, frames, channels)):
        return np.ascontiguousarray(data.reshape(batch, frames, channels))
    if shape in ((batch, channels, 1, frames), (batch, channels, frames, 1)):
        return np.ascontiguousarray(data.reshape(batch, channels, frames).transpose(0, 2, 1))
    raise ValueError(f'Unsupported captured tensor order: {shape}')


def gelu(values):
    import numpy as np
    x = np.asarray(values, dtype=np.float32)
    return np.float32(.5) * x * (np.float32(1) + np.tanh(np.float32((2 / np.pi) ** .5) * (x + np.float32(.044715) * x * x * x)))


def relative_l2(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Reference shape/non-finite mismatch')
    return float(np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)).ravel()) / max(np.linalg.norm(b.astype(np.float64).ravel()), 1e-30))


def export_ffn_layouts(output: Path, x, w1, b1, w2, b2):
    """Export four layouts and three equivalent GELU expressions, sharing weights."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Preserve existing experiment: {output}')
    x, w1, b1, w2, b2 = [np.ascontiguousarray(a, dtype=np.float32) for a in (x, w1, b1, w2, b2)]
    if x.ndim != 3 or w1.ndim != 2 or w2.ndim != 2:
        raise ValueError('Expected BTC input and OI projection weights')
    batch, frames, hidden = x.shape
    middle = w1.shape[0]
    if w1.shape != (middle, hidden) or w2.shape != (hidden, middle) or b1.shape != (middle,) or b2.shape != (hidden,):
        raise ValueError('FFN projection dimensions do not match')
    if not all(np.isfinite(a).all() for a in (x, w1, b1, w2, b2)):
        raise ValueError('Non-finite source parameter')
    reference = np.ascontiguousarray(gelu(x @ w1.T + b1) @ w2.T + b2)
    output.mkdir(parents=True)
    ranges = {}
    with (output / 'weights.f32').open('wb') as stream:
        for name, a in [('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2)]:
            ranges[name] = (stream.tell(), a.nbytes)
            stream.write(a.tobytes())

    def parameter(name, shape):
        t = TensorProto(name=name, data_type=TensorProto.FLOAT, dims=shape, data_location=TensorProto.EXTERNAL)
        offset, length = ranges[name]
        for key, value in [('location', 'weights.f32'), ('offset', str(offset)), ('length', str(length))]:
            entry = t.external_data.add(); entry.key = key; entry.value = value
        return t

    report = {'scope': 'Isolated real FFN ONNX mathematical/layout validation on PC CPU; no NPU speed claim',
              'batch': batch, 'frames': frames, 'hidden': hidden, 'intermediate': middle,
              'opset': 13, 'graphIrVersion': 8, 'weightsSha256': hashlib.sha256((output / 'weights.f32').read_bytes()).hexdigest(), 'cases': []}
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    for layout in ('gemm-flat', 'conv-flat', 'conv-spatial', 'conv-bridge'):
        for activation in ('pow', 'mul', 'fuser'):
            name = layout + '-' + activation
            conv = layout != 'gemm-flat'
            shape = {'gemm-flat': [batch * frames, hidden], 'conv-flat': [batch * frames, hidden, 1, 1],
                     'conv-spatial': [batch, hidden, 1, frames], 'conv-bridge': [batch, frames, hidden]}[layout]
            initializers = [parameter('w1', [middle, hidden, 1, 1] if conv else [middle, hidden]), parameter('b1', [middle]),
                            parameter('w2', [hidden, middle, 1, 1] if conv else [hidden, middle]), parameter('b2', [hidden])]
            nodes = []
            def node(op, ins, out, label, **attrs):
                nodes.append(helper.make_node(op, ins, [out], name=label, **attrs))
            source = 'inputs'
            if layout == 'conv-bridge':
                initializers.append(numpy_helper.from_array(np.array([2], dtype=np.int64), 'axis2'))
                node('Transpose', ['inputs'], 'bct', 'boundary_to_bct', perm=[0, 2, 1])
                node('Unsqueeze', ['bct', 'axis2'], 'nchw', 'boundary_to_nchw'); source = 'nchw'
            node('Conv' if conv else 'Gemm', [source, 'w1', 'b1'], 'up', 'ffn_up', **({'kernel_shape': [1, 1]} if conv else {'transB': 1}))
            constants = [('cubic', .044715), ('sqrt2pi', (2 / np.pi) ** .5), ('one', 1.), ('half', .5)]
            if activation != 'mul': constants.append(('three', 3.))
            for key, value in constants:
                initializers.append(numpy_helper.from_array(np.array(value, dtype=np.float32), key))
            if activation != 'mul': node('Pow', ['up', 'three'], 'cube', 'gelu_cube_pow')
            else:
                node('Mul', ['up', 'up'], 'square', 'gelu_square_mul'); node('Mul', ['square', 'up'], 'cube', 'gelu_cube_mul')
            for op, ins, out in [('Mul', ['cube', 'cubic'], 'scaled_cube'), ('Add', ['up', 'scaled_cube'], 'polynomial'),
                                 ('Mul', ['polynomial', 'sqrt2pi'], 'tanh_in'), ('Tanh', ['tanh_in'], 'tanh_out'),
                                 ('Add', ['tanh_out', 'one'], 'plus_one')]:
                node(op, ins, out, 'gelu_' + out)
            # The pinned MNN FuseGeLu matcher expects x * ((1 + tanh(...)) * .5).
            # Keep both associations; mathematical equivalence is checked by ORT.
            if activation == 'fuser':
                node('Mul', ['plus_one', 'half'], 'half_plus_one', 'gelu_half_plus_one')
                node('Mul', ['up', 'half_plus_one'], 'activated', 'gelu_activated')
            else:
                node('Mul', ['up', 'half'], 'half_up', 'gelu_half_up')
                node('Mul', ['half_up', 'plus_one'], 'activated', 'gelu_activated')
            destination = 'down' if layout == 'conv-bridge' else 'outputs'
            node('Conv' if conv else 'Gemm', ['activated', 'w2', 'b2'], destination, 'ffn_down', **({'kernel_shape': [1, 1]} if conv else {'transB': 1}))
            if layout == 'conv-bridge':
                node('Squeeze', ['down', 'axis2'], 'down_bct', 'boundary_squeeze')
                node('Transpose', ['down_bct'], 'outputs', 'boundary_to_btc', perm=[0, 2, 1])
            graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info('inputs', TensorProto.FLOAT, shape)],
                                      [helper.make_tensor_value_info('outputs', TensorProto.FLOAT, shape)], initializers)
            model = helper.make_model(graph, producer_name='LeVo current FFN layout research', opset_imports=[helper.make_opsetid('', 13)])
            model.ir_version = 8
            path = output / (name + '.onnx'); onnx.save(model, path); onnx.checker.check_model(str(path))
            if layout == 'conv-spatial':
                data = np.ascontiguousarray(x.transpose(0, 2, 1)[:, :, None, :]); expected = np.ascontiguousarray(reference.transpose(0, 2, 1)[:, :, None, :])
            else:
                data, expected = x.reshape(shape), reference.reshape(shape)
            session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
            actual = session.run(None, {'inputs': data})[0]
            error = relative_l2(actual, expected)
            if error > 1e-5:
                raise ValueError(f'{name} exceeds fixed mathematical parity budget: {error}')
            data.tofile(output / (name + '-input.f32')); expected.tofile(output / (name + '-reference.f32'))
            report['cases'].append({'id': name, 'model': path.name, 'inputShape': shape, 'outputShape': shape,
                                    'relativeL2': error, 'numericalBudget': 1e-5, 'transposeNodes': sum(n.op_type == 'Transpose' for n in nodes),
                                    'internalFfnTransposes': 0, 'nodeCount': len(nodes), 'onnxSha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'weights': 'weights.f32', 'input': name + '-input.f32', 'reference': name + '-reference.f32'})
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report, reference


def prepare_capture(capture_dir: Path, output: Path):
    import numpy as np
    capture_dir, output = Path(capture_dir), Path(output)
    audit = json.loads((capture_dir / 'layout.json').read_text(encoding='utf-8'))
    if not audit.get('success') or (audit.get('batch'), audit.get('frames'), audit.get('hidden'), audit.get('ffn')) != (2, 256, 1200, 4800):
        raise ValueError('Capture does not match the current verified DiT contract')
    def values(filename, count):
        p = capture_dir / filename
        if p.stat().st_size != count * 4:
            raise ValueError('Captured file size mismatch: ' + filename)
        a = np.fromfile(p, dtype='<f4')
        if not np.isfinite(a).all(): raise ValueError('Non-finite capture')
        return a
    def activation(key, channels):
        entry = audit[key]
        return to_btc(values(entry['file'], 2 * 256 * channels), entry['hostShape'], batch=2, frames=256, channels=channels)
    x, up, down_in, actual = activation('upInput', 1200), activation('upOutput', 4800), activation('downInput', 4800), activation('downOutput', 1200)
    activation_error = relative_l2(gelu(up), down_in)
    if activation_error > 1e-4: raise ValueError('Captured activation is not verified tanh GELU')
    w1, w2 = values(audit['up']['weightFile'], 4800 * 1200).reshape(4800, 1200), values(audit['down']['weightFile'], 1200 * 4800).reshape(1200, 4800)
    b1, b2 = values(audit['up']['biasFile'], 4800), values(audit['down']['biasFile'], 1200)
    report, reference = export_ffn_layouts(output, x, w1, b1, w2, b2)
    report['provenance'] = {'parameters': 'Original MNN W8/g64 decoded with pinned ConvolutionCommon::load; not unquantized training weights',
                            'inputs': 'First real sampled-code DiT window, first timestep, first FFN, batch2',
                            'cfgBranches': audit.get('cfgBranches', 'conditional+conditional (historical capture)'),
                            'tanhGeluVsCapturedActivationL2': activation_error, 'fp32FfnVsCapturedMnnW8L2': relative_l2(reference, actual),
                            'runtimeCaptureUsedRandomWeights': False}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report
