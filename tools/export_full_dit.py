"""Export all 39 real Mixed blocks, keeping >2GB weights outside protobuf.

Simplify the fixed-shape graph before attaching external initializers. This
avoids both the protobuf 2GB ceiling and known unsimplified MNN shape crashes.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file
from restore_mixed_dit import RestoredMixedCfm, upstream_classes
from export_real_dit_block import real_freqs, real_rope

ROOT = Path(__file__).resolve().parents[1]


class FullDit(nn.Module):
    def __init__(self, cfm):
        super().__init__()
        self.cfm = cfm

    def forward(self, inputs, timestep, attention_mask):
        return self.cfm(inputs, timestep, attention_mask)[..., -64:]


def l2(actual, expected):
    return float(np.linalg.norm((actual - expected).astype(np.float64)) /
                 max(np.linalg.norm(expected.astype(np.float64)), 1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnn-python-path', required=True)
    parser.add_argument('--frames', type=int, choices=(64, 256), default=256)
    args = parser.parse_args()
    sys.path.append(args.mnn_python_path)
    import onnx
    import onnxsim
    torch.manual_seed(2307)
    torch.set_num_threads(4)
    out = ROOT / f'models/full-dit-{args.frames}'
    out.mkdir(exist_ok=True)
    if (out / 'full-w8.mnn').exists():
        raise ValueError('Preserve existing converted model; use a new output directory')
    source = ROOT / 'models/levo-mixed-codec/mixed-inference.safetensors'
    manifest = json.loads((source.parent / 'download-manifest.json').read_text())
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != manifest['subset_sha256']:
        raise ValueError('Mixed weights changed')
    state = load_file(source)
    cfm = RestoredMixedCfm(ROOT / 'reference/LeVo').eval()
    weights = {k.removeprefix('cfm_wrapper.'): v for k, v in state.items() if k.startswith('cfm_wrapper.')}
    cfm.load_state_dict(weights, strict=True, assign=True)
    strict_tensor_count = len(weights)
    model = FullDit(cfm).eval()
    frames = args.frames
    inputs = torch.randn(2, frames, 1200) * .15
    timestep = torch.tensor([.35, .35])
    attention = torch.ones(2, 1, frames, frames)
    attention[..., -8:] = 0
    with torch.inference_mode():
        expected = model(inputs, timestep, attention).numpy().copy()
    _, _, upstream = upstream_classes(ROOT / 'reference/LeVo')
    # Precompute the fixed RoPE outside ONNX so there is no constant Einsum.
    frequencies = real_freqs(40, frames)
    upstream.precompute_freqs_cis = lambda dim, end, constant=10000.: frequencies
    upstream.apply_rotary_emb = real_rope
    with torch.inference_mode():
        rewritten = model(inputs, timestep, attention).numpy().copy()
    rewrite_l2 = l2(rewritten, expected)
    if rewrite_l2 > 1e-4:
        raise ValueError('Full-network real RoPE rewrite changed numerics')
    graph_path = out / 'full-shapes.onnx'
    print('Strict complete CFM loaded; exporting graph without embedded weights', flush=True)
    torch.onnx.export(model, (inputs, timestep, attention), str(graph_path),
                      input_names=['inputs', 'timestep', 'attention_mask'], output_names=['velocity'],
                      opset_version=17, dynamo=False, export_params=False, do_constant_folding=False,
                      keep_initializers_as_inputs=True)
    graph = onnx.load(str(graph_path))
    graph, valid = onnxsim.simplify(graph, check_n=0)
    if not valid:
        raise ValueError('Fixed graph simplification failed')
    named = dict(model.named_parameters())
    named.update(dict(model.named_buffers()))
    public = {'inputs', 'timestep', 'attention_mask'}
    retained = []
    weight_path = out / 'full-weights.bin'
    attached = 0
    with weight_path.open('wb') as stream:
        for item in graph.graph.input:
            if item.name in public:
                retained.append(item)
                continue
            if item.name not in named:
                raise ValueError('Unmapped weight input: ' + item.name)
            value = named[item.name].detach().cpu().numpy()
            offset = stream.tell()
            data = np.ascontiguousarray(value).tobytes()
            stream.write(data)
            tensor = onnx.TensorProto()
            tensor.name = item.name
            tensor.data_type = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
            tensor.dims.extend(value.shape)
            tensor.data_location = onnx.TensorProto.EXTERNAL
            for key, val in [('location', weight_path.name), ('offset', str(offset)), ('length', str(len(data)))]:
                record = tensor.external_data.add()
                record.key = key
                record.value = val
            graph.graph.initializer.append(tensor)
            attached += 1
    del graph.graph.input[:]
    graph.graph.input.extend(retained)
    target_onnx = out / 'full-external.onnx'
    onnx.save_model(graph, str(target_onnx))
    print(f'External ONNX ready: {attached} unique parameter inputs, {weight_path.stat().st_size} bytes', flush=True)
    values = {'inputs': inputs.numpy(), 'timestep': timestep.numpy(), 'attention_mask': attention.numpy()}
    np.savez(out / 'fixture.npz', **values, reference=expected)
    for name, value in {**values, 'reference': expected}.items():
        np.ascontiguousarray(value, dtype='<f4').tofile(out / (name + '.f32'))
    del graph, model, cfm, weights, state, named
    gc.collect()
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(target_onnx), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(None, values)[0]
    ort_l2 = l2(actual, expected)
    if not np.isfinite(actual).all() or ort_l2 > 1e-4:
        raise ValueError(f'Full ONNX mismatch: {ort_l2}')
    print(f'Complete ONNX parity L2={ort_l2}', flush=True)
    del session
    gc.collect()
    target = out / 'full-w8.mnn'
    subprocess.run([sys.executable, str(ROOT / 'tools/mnn_convert_once.py'), '--mnn-python-path',
                    args.mnn_python_path, '--', '-f', 'ONNX', '--modelFile', str(target_onnx),
                    '--MNNModel', str(target), '--bizCode', 'levo-full-39-layer-diagnostic',
                    '--saveExternalData', '--weightQuantBits', '8', '--weightQuantBlock', '64'], check=True)
    import MNN
    net = MNN.Interpreter(str(target))
    begin = time.perf_counter()
    mnn_session = net.createSession({'backend': 'CPU', 'numThread': 4, 'precision': 1, 'memory': 2})
    setup = time.perf_counter() - begin
    for name, value in values.items():
        host = MNN.Tensor(tuple(value.shape), MNN.Halide_Type_Float, np.ascontiguousarray(value), MNN.Tensor_DimensionType_Caffe)
        if not net.getSessionInput(mnn_session, name).copyFrom(host):
            raise ValueError('Input copy failed: ' + name)
    begin = time.perf_counter()
    net.runSession(mnn_session)
    output = np.asarray(net.getSessionOutput(mnn_session, 'velocity').getNumpyData()).copy()
    elapsed = time.perf_counter() - begin
    if output.shape != expected.shape or not np.isfinite(output).all():
        raise ValueError('Full MNN produced invalid output')
    np.ascontiguousarray(output, dtype='<f4').tofile(out / 'w8-reference.f32')
    report = {'all_39_layers': True, 'strict_cfm_tensors': strict_tensor_count,
              'source_subset_sha256': digest.hexdigest(), 'batch': 2, 'frames': frames,
              'real_rope_relative_l2': rewrite_l2, 'onnx_relative_l2': ort_l2,
              'mnn_w8_relative_l2': l2(output, expected), 'mnn_setup_seconds': setup,
              'mnn_forward_seconds': elapsed, 'finite': True,
              'scope': 'Complete real Mixed CFM one velocity evaluation, synthetic input; no trained song LM or music-quality proof'}
    report['model_bytes'] = target.stat().st_size
    report['weight_bytes'] = Path(str(target) + '.weight').stat().st_size
    (ROOT / f'reports/full-dit-mnn-{frames}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
