"""Export one actually captured music DiT block with NCHW-resident projections.

Not a full DiT, original training checkpoint, or low-memory-MNN parity claim.
The modulation boundary already includes time conditioning and layer table.
"""
from pathlib import Path
import hashlib
import json


def export_block(capture_dir, output):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper, TensorProto
    from .dit_layout import relative_l2, to_btc

    source, output = Path(capture_dir), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous block export')
    audit = json.loads((source / 'layout.json').read_text(encoding='utf-8'))
    if not audit.get('success') or audit['backend'] != 'MNN CPU High / Memory Normal / 6threads':
        raise ValueError('Use the independently verified floating Memory Normal capture')
    if (audit['batch'], audit['frames'], audit['hidden'], audit['ffn']) != (2, 256, 1200, 4800):
        raise ValueError('Unsupported current block architecture')
    def read(name, shape):
        data = np.fromfile(source / name, dtype='<f4')
        if data.size != int(np.prod(shape)) or not np.isfinite(data).all():
            raise ValueError('Capture file shape/finite mismatch: ' + name)
        return data.reshape(shape)
    x = to_btc(read(audit['blockInput']['file'], (2, 256, 1200)), audit['blockInput']['hostShape'], batch=2, frames=256, channels=1200)
    modulation = read(audit['blockModulation']['file'], (2, 6, 1200))
    expected = read(audit['blockOutput']['file'], (2, 256, 1200)).transpose(0, 2, 1)[:, :, None, :].copy()
    payload = np.concatenate([x, modulation, np.zeros((2, 2, 1200), dtype=np.float32)], axis=1).transpose(0, 2, 1)[:, :, None, :].copy()
    nodes, params = [], []
    def const(name, value):
        a = np.asarray(value)
        # ascontiguousarray promotes a scalar to[1], defeating OMG GELU fusion.
        params.append(numpy_helper.from_array(np.ascontiguousarray(a) if a.ndim else a, name)); return name
    def node(op, inputs, output, **kw):
        nodes.append(helper.make_node(op, inputs, [output], name=output, **kw)); return output
    def shape(name, value): return const(name, np.array(value, dtype=np.int64))
    def slice_axis(value, start, stop, axis, name):
        return node('Slice', [value, shape(name+'_start', [start]), shape(name+'_stop', [stop]), shape(name+'_axis', [axis])], name)
    def binary(op, a, b, name): return node(op, [a, b], name)
    const('one', np.array(1, dtype='f4')); const('half', np.array(.5, dtype='f4'))
    const('gelu_cubic', np.array(.044715, dtype='f4')); const('sqrt2pi', np.array((2/np.pi)**.5, dtype='f4'))
    const('attention_scale', np.array(40**-.5, dtype='f4'))
    def norm(value, key):
        eps = audit[key]['epsilon']; const(key+'_eps', np.array(eps, dtype='f4'))
        const(key+'_gamma', read(key+'-gamma.f32', (1, 1200, 1, 1)))
        const(key+'_beta', read(key+'-beta.f32', (1, 1200, 1, 1)))
        mean = node('ReduceMean', [value], key+'_mean', axes=[1], keepdims=1)
        centered = binary('Sub', value, mean, key+'_centered')
        square = binary('Mul', centered, centered, key+'_square')
        variance = node('ReduceMean', [square], key+'_variance', axes=[1], keepdims=1)
        std = node('Sqrt', [binary('Add', variance, key+'_eps', key+'_variance_eps')], key+'_std')
        normalized = binary('Div', centered, std, key+'_normalized')
        return binary('Add', binary('Mul', normalized, key+'_gamma', key+'_scaled'), key+'_beta', key+'_output')
    def conv(value, key, middle, hidden, label):
        const(label+'_weight', read(audit[key]['weightFile'], (middle, hidden, 1, 1)))
        const(label+'_bias', read(audit[key]['biasFile'], (middle,)))
        return node('Conv', [value, label+'_weight', label+'_bias'], label, kernel_shape=[1, 1])
    hidden = slice_axis('inputs', 0, 256, 3, 'hidden')
    mods = [slice_axis('inputs', 256+i, 257+i, 3, 'mod'+str(i)) for i in range(6)]
    def modulate(value, shift, scale, label):
        scale = binary('Add', scale, 'one', label+'_scale')
        return binary('Add', binary('Mul', value, scale, label+'_multiply'), shift, label)
    normalized = modulate(norm(hidden, 'ln1'), mods[0], mods[1], 'attention_input')
    qkv = conv(normalized, 'qkv', 3600, 1200, 'attention_qkv')
    cos = read(audit['ropeCos']['file'], (1, 1, 256, 20)); sin = read(audit['ropeSin']['file'], (1, 1, 256, 20))
    const('cos', cos); const('sin', sin)
    def heads(value, name):
        return node('Transpose', [node('Reshape', [value, shape(name+'_shape', [2, 30, 40, 256])], name+'_reshape')], name, perm=[0, 1, 3, 2])
    def rotary(value, name):
        pairs = node('Reshape', [value, shape(name+'_shape', [2, 30, 256, 20, 2])], name+'_pairs')
        even = node('Squeeze', [slice_axis(pairs, 0, 1, 4, name+'_even'), shape(name+'_squeeze_axis', [4])], name+'_a')
        odd = node('Squeeze', [slice_axis(pairs, 1, 2, 4, name+'_odd'), name+'_squeeze_axis'], name+'_b')
        real = binary('Sub', binary('Mul', even, 'cos', name+'_ac'), binary('Mul', odd, 'sin', name+'_bs'), name+'_real')
        imag = binary('Add', binary('Mul', even, 'sin', name+'_as'), binary('Mul', odd, 'cos', name+'_bc'), name+'_imag')
        a = node('Unsqueeze', [real, name+'_squeeze_axis'], name+'_real_pair')
        b = node('Unsqueeze', [imag, name+'_squeeze_axis'], name+'_imag_pair')
        stacked = node('Concat', [a, b], name+'_stack', axis=4)
        return node('Reshape', [stacked, shape(name+'_output_shape', [2, 30, 256, 40])], name)
    q = rotary(heads(slice_axis(qkv, 0, 1200, 1, 'query'), 'query_heads'), 'query_rotated')
    k = rotary(heads(slice_axis(qkv, 1200, 2400, 1, 'key'), 'key_heads'), 'key_rotated')
    v = heads(slice_axis(qkv, 2400, 3600, 1, 'value'), 'value_heads')
    kt = node('Transpose', [k], 'key_transposed', perm=[0, 1, 3, 2])
    scores = binary('Mul', node('MatMul', [q, kt], 'qk'), 'attention_scale', 'scores')
    probability = node('Softmax', [scores], 'probability', axis=-1)
    attended = node('Transpose', [node('MatMul', [probability, v], 'pv')], 'attention_bhdt', perm=[0, 1, 3, 2])
    attended = node('Reshape', [attended, shape('attention_nchw_shape', [2, 1200, 1, 256])], 'attention_nchw')
    projected = conv(attended, 'attnProj', 1200, 1200, 'attention_projected')
    residual = binary('Add', hidden, binary('Mul', projected, mods[2], 'attention_gated'), 'attention_residual')
    normalized = modulate(norm(residual, 'ln2'), mods[3], mods[4], 'ffn_input')
    up = conv(normalized, 'up', 4800, 1200, 'ffn_up')
    const('three', np.array(3, dtype='f4'))
    cubic = binary('Pow', up, 'three', 'cube')
    polynomial = binary('Add', up, binary('Mul', cubic, 'gelu_cubic', 'scaled_cube'), 'polynomial')
    tanh = node('Tanh', [binary('Mul', polynomial, 'sqrt2pi', 'tanh_in')], 'tanh_out')
    activated = binary('Mul', up, binary('Mul', binary('Add', tanh, 'one', 'gelu_one'), 'half', 'gelu_half'), 'gelu')
    down = conv(activated, 'down', 1200, 4800, 'ffn_down')
    binary('Add', residual, binary('Mul', down, mods[5], 'ffn_gated'), 'outputs')
    model = helper.make_model(helper.make_graph(nodes, 'actual_music_block0_spatial',
        [helper.make_tensor_value_info('inputs', TensorProto.FLOAT, [2, 1200, 1, 264])],
        [helper.make_tensor_value_info('outputs', TensorProto.FLOAT, [2, 1200, 1, 256])], params), opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    output.mkdir(parents=True)
    path = output / 'block_spatial.onnx'
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location='weights.f32', size_threshold=1024, convert_attribute=False)
    onnx.checker.check_model(str(path))
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    actual = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider']).run(None, {'inputs': payload})[0]
    error = relative_l2(actual, expected)
    payload.tofile(output/'input.f32'); actual.tofile(output/'reference.f32')
    report = {'scope': 'One complete real music DiT block incl Attention/RoPE/norm/residual/FFN; first all-valid window only; modulation precomputed outside boundary; not full22-layer DiT',
              'relativeL2VsMnnMemoryNormalBlock': error, 'mathBudget': 1e-4, 'success': error <= 1e-4,
              'source': 'Decoded current W8 parameters, real CFG branch inputs; no random weights',
              'inputShape': [2,1200,1,264], 'outputShape': [2,1200,1,256], 'nodes': len(nodes),
              'onnxSha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'weightsSha256': hashlib.sha256((output/'weights.f32').read_bytes()).hexdigest()}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not report['success']: raise ValueError('Block reference did not pass original floating-path budget: '+str(error))
    return report
