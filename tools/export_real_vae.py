"""Export the strictly verified real decoder and prepare bounded-stream fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file
from restore_vae_decoder import create_decoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnn-python-path', required=True)
    args = parser.parse_args()
    sys.path.append(args.mnn_python_path)
    torch.manual_seed(2307)
    torch.set_num_threads(4)
    source = ROOT / 'models/levo-vae'
    prior = json.loads((ROOT / 'reports/pretrained-vae-cpu.json').read_text())
    if not prior['strict_load'] or not prior['every_source_storage_crc32_verified']:
        raise ValueError('Run the real decoder integrity/CPU validation first')
    with torch.device('meta'):
        model = create_decoder()
    model.load_state_dict(load_file(source / 'decoder.safetensors'), strict=True, assign=True)
    model.eval()
    for child in model.modules():
        if hasattr(child, 'weight_g'):
            torch.nn.utils.remove_weight_norm(child)
    fixture = np.load(source / 'fixture-256.npz')
    x = torch.from_numpy(fixture['latent'])
    with torch.inference_mode():
        output = model(x)
    folded = output.numpy()
    fold_max_abs = float(np.max(np.abs(folded-fixture['output'])))
    fold_l2 = float(np.linalg.norm((folded-fixture['output']).astype(np.float64)) /
                    np.linalg.norm(fixture['output'].astype(np.float64)))
    # Near-zero waveform samples make elementwise relative tolerances misleading.
    # Keep both a global relative-energy and absolute FP32 error bound.
    if not np.isfinite(folded).all() or fold_l2 > 1e-4 or fold_max_abs > 1e-4:
        raise ValueError(f'Weight-norm folding changed decoder: L2={fold_l2}, max={fold_max_abs}')
    out = ROOT / 'models/vae-stream-256'
    out.mkdir(exist_ok=True)
    onnx = out / 'vae-256.onnx'
    torch.onnx.export(model, (x,), str(onnx), input_names=['latent'], output_names=['pcm'],
                      opset_version=17, dynamo=False)
    import onnx as ox
    import onnxsim
    simplified, valid = onnxsim.simplify(ox.load(str(onnx)))
    if not valid:
        raise ValueError('Decoder ONNX simplification check failed')
    onnx = out / 'vae-256-simplified.onnx'
    ox.save(simplified, str(onnx))
    del simplified
    import onnxruntime as ort
    ort_output = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider']).run(None, {'latent': x.numpy()})[0]
    relative = lambda a, b: float(np.linalg.norm((a-b).astype(np.float64)) / max(np.linalg.norm(b.astype(np.float64)), 1e-30))
    ort_l2 = relative(ort_output, fixture['output'])
    if not np.isfinite(ort_output).all() or ort_l2 > 1e-3:
        raise ValueError(f'Exported decoder is not aligned: {ort_l2}')
    target = out / 'vae-256-fp32.mnn'
    subprocess.run([sys.executable, str(ROOT / 'tools/mnn_convert_once.py'),
                    '--mnn-python-path', args.mnn_python_path, '--',
                    '-f', 'ONNX', '--modelFile', str(onnx), '--MNNModel', str(target),
                    '--bizCode', 'levo-real-vae-cpu-test', '--saveExternalData'], check=True)
    import MNN
    net = MNN.Interpreter(str(target))
    session = net.createSession({'backend': 'CPU', 'numThread': 4, 'precision': 1, 'memory': 2})
    value = MNN.Tensor(tuple(x.shape), MNN.Halide_Type_Float, x.numpy(), MNN.Tensor_DimensionType_Caffe)
    if not net.getSessionInput(session, 'latent').copyFrom(value):
        raise ValueError('VAE input copy failed')
    begin = time.perf_counter()
    code = net.runSession(session)
    actual = np.asarray(net.getSessionOutput(session, 'pcm').getNumpyData()).copy()
    mnn_output_shape = list(actual.shape)
    # MNN promotes Conv1D to a 2D convolution with a trailing singleton axis.
    if actual.shape == fixture['output'].shape + (1,):
        actual = actual[..., 0]
    mnn_seconds = time.perf_counter() - begin
    mnn_l2 = relative(actual, fixture['output'])
    if code not in (None, 0) or not np.isfinite(actual).all() or mnn_l2 > 1e-3:
        raise ValueError(f'MNN real decoder is not aligned: {mnn_l2}')
    # Three windows: [0:256], [192:448], [384:640]. They represent
    # 25.60 seconds after two 2.56-second overlaps, not 30.72 seconds.
    files = {}
    def save(name, values):
        values = np.ascontiguousarray(values, dtype='<f4')
        path = out / name
        path.write_bytes(values.tobytes())
        files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    full_latent = torch.randn(1, 64, 640) * .2
    decoded = []
    with torch.inference_mode():
        whole = model(full_latent)[0].numpy()
        for index, start in enumerate((0, 192, 384)):
            latent = full_latent[:, :, start:start+256]
            pcm = model(latent)[0].numpy()
            save(f'latent-{index}.f32', latent.numpy())
            save(f'pcm-{index}.f32', pcm)
            decoded.append(pcm)
    tail = None
    emitted = []
    overlap = 64 * 1920
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)[None, :]
    for index, pcm in enumerate(decoded):
        pcm = pcm.copy()
        if tail is not None:
            pcm[:, :overlap] = tail * (1-ramp) + pcm[:, :overlap] * ramp
        if index < 2:
            tail = pcm[:, -overlap:].copy()
            piece = pcm[:, :-overlap]
        else:
            piece = pcm
        save(f'emit-{index}.f32', piece)
        emitted.append(piece)
    stitched = np.concatenate(emitted, axis=1)
    if stitched.shape != whole.shape:
        raise ValueError('PCM overlap length mismatch')
    stream = {'scope': 'Real decoder, synthetic latent stream; no music LM or DiT in this fixture',
              'frames': 256, 'overlap_frames': 64, 'hop_frames': 192,
              'chunks': 3, 'channels': 2, 'sample_rate': 48000, 'samples_per_frame': 1920,
              'valid_output_frames': 640, 'output_seconds': 25.6,
              'chunked_vs_monolithic_relative_l2': relative(stitched, whole), 'files': files}
    (out / 'fixture.json').write_text(json.dumps(stream, indent=2), encoding='utf8')
    report = {'onnx_vs_pytorch_relative_l2': ort_l2, 'mnn_vs_pytorch_relative_l2': mnn_l2,
              'mnn_first_seconds': mnn_seconds, 'mnn_output_shape': mnn_output_shape,
              'weight_norm_folded': True,
              'weight_norm_fold_relative_l2': fold_l2, 'weight_norm_fold_max_abs': fold_max_abs,
              'full_decoder_weights_loaded': True, 'stream_fixture': stream,
              'real_song_generated': False, 'full_codec_validated': False}
    (ROOT / 'reports/pretrained-vae-mnn.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='stream_fixture'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
