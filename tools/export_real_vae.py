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
    parser.add_argument('--frames', type=int, choices=(64, 256), default=256)
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
    frames = args.frames
    fixture = np.load(source / 'fixture-256.npz')
    x = torch.from_numpy(np.ascontiguousarray(fixture['latent'][..., :frames]))
    with torch.inference_mode():
        reference = fixture['output'] if frames == 256 else model(x).numpy().copy()
    for child in model.modules():
        if hasattr(child, 'weight_g'):
            torch.nn.utils.remove_weight_norm(child)
    with torch.inference_mode():
        output = model(x)
    folded = output.numpy()
    fold_max_abs = float(np.max(np.abs(folded-reference)))
    fold_l2 = float(np.linalg.norm((folded-reference).astype(np.float64)) /
                    np.linalg.norm(reference.astype(np.float64)))
    # Near-zero waveform samples make elementwise relative tolerances misleading.
    # Keep both a global relative-energy and absolute FP32 error bound.
    if not np.isfinite(folded).all() or fold_l2 > 1e-4 or fold_max_abs > 1e-4:
        raise ValueError(f'Weight-norm folding changed decoder: L2={fold_l2}, max={fold_max_abs}')
    out = ROOT / f'models/vae-stream-{frames}'
    out.mkdir(exist_ok=True)
    onnx = out / f'vae-{frames}.onnx'
    torch.onnx.export(model, (x,), str(onnx), input_names=['latent'], output_names=['pcm'],
                      opset_version=17, dynamo=False)
    import onnx as ox
    import onnxsim
    simplified, valid = onnxsim.simplify(ox.load(str(onnx)))
    if not valid:
        raise ValueError('Decoder ONNX simplification check failed')
    onnx = out / f'vae-{frames}-simplified.onnx'
    ox.save(simplified, str(onnx))
    del simplified
    import onnxruntime as ort
    ort_output = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider']).run(None, {'latent': x.numpy()})[0]
    relative = lambda a, b: float(np.linalg.norm((a-b).astype(np.float64)) / max(np.linalg.norm(b.astype(np.float64)), 1e-30))
    ort_l2 = relative(ort_output, reference)
    if not np.isfinite(ort_output).all() or ort_l2 > 1e-3:
        raise ValueError(f'Exported decoder is not aligned: {ort_l2}')
    target = out / f'vae-{frames}-fp32.mnn'
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
    if actual.shape == reference.shape + (1,):
        actual = actual[..., 0]
    mnn_seconds = time.perf_counter() - begin
    mnn_l2 = relative(actual, reference)
    if code not in (None, 0) or not np.isfinite(actual).all() or mnn_l2 > 1e-3:
        raise ValueError(f'MNN real decoder is not aligned: {mnn_l2}')
    # Always the same 640-frame latent timeline and 25.60-second denominator.
    # The64-frame case uses13 windows; the256-frame baseline uses3 windows.
    files = {}
    def save(name, values):
        values = np.ascontiguousarray(values, dtype='<f4')
        path = out / name
        path.write_bytes(values.tobytes())
        files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    baseline = ROOT / 'models/vae-stream-256'
    if all((baseline / f'latent-{i}.f32').exists() for i in range(3)):
        parts = [np.fromfile(baseline / f'latent-{i}.f32', dtype='<f4').reshape(1,64,256) for i in range(3)]
        for index in (0,1):
            np.testing.assert_array_equal(parts[index][..., -64:], parts[index+1][..., :64])
        full_latent = torch.from_numpy(np.concatenate((parts[0][...,:192],parts[1][...,:192],parts[2]),axis=-1))
    elif frames == 256:
        full_latent = torch.randn(1, 64, 640) * .2
    else:
        raise ValueError('Prepare the256-frame baseline before comparing smaller windows')
    overlap_frames = frames // 4
    hop_frames = frames - overlap_frames
    starts = list(range(0, 640-frames+1, hop_frames))
    if starts[-1]+frames != 640:
        raise ValueError('Experiment timeline must fit complete windows')
    decoded = []
    with torch.inference_mode():
        whole = model(full_latent)[0].numpy()
        for index, start in enumerate(starts):
            latent = full_latent[:, :, start:start+frames]
            pcm = model(latent)[0].numpy()
            save(f'latent-{index}.f32', latent.numpy())
            save(f'pcm-{index}.f32', pcm)
            decoded.append(pcm)
            print(f'VAE fixture{frames}: {index+1}/{len(starts)}', flush=True)
    tail = None
    emitted = []
    overlap = overlap_frames * 1920
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)[None, :]
    for index, pcm in enumerate(decoded):
        pcm = pcm.copy()
        if tail is not None:
            pcm[:, :overlap] = tail * (1-ramp) + pcm[:, :overlap] * ramp
        if index < len(decoded)-1:
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
              'frames': frames, 'overlap_frames': overlap_frames, 'hop_frames': hop_frames,
              'chunks': len(starts), 'channels': 2, 'sample_rate': 48000, 'samples_per_frame': 1920,
              'valid_output_frames': 640, 'output_seconds': 25.6,
              'source_latent_sha256': hashlib.sha256(full_latent.numpy().tobytes()).hexdigest(),
              'chunked_vs_monolithic_relative_l2': relative(stitched, whole), 'files': files}
    (out / 'fixture.json').write_text(json.dumps(stream, indent=2), encoding='utf8')
    report = {'onnx_vs_pytorch_relative_l2': ort_l2, 'mnn_vs_pytorch_relative_l2': mnn_l2,
              'mnn_first_seconds': mnn_seconds, 'mnn_output_shape': mnn_output_shape,
              'weight_norm_folded': True,
              'weight_norm_fold_relative_l2': fold_l2, 'weight_norm_fold_max_abs': fold_max_abs,
              'full_decoder_weights_loaded': True, 'stream_fixture': stream,
              'real_song_generated': False, 'full_codec_validated': False}
    report_suffix = '' if frames == 256 else f'-{frames}'
    (ROOT / f'reports/pretrained-vae-mnn{report_suffix}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='stream_fixture'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
