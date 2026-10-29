"""Strictly load real VAE decoder weights and check fixed latent lengths.

Synthetic latent fixtures test numerical/shape behavior, not music quality.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import load_file
from restore_vae_decoder import create_decoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.manual_seed(2307)
    torch.set_num_threads(4)
    folder = ROOT / 'models/levo-vae'
    manifest = json.loads((folder / 'decoder-manifest.json').read_text())
    digest = hashlib.sha256()
    with (folder / 'decoder.safetensors').open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if not manifest['every_storage_crc32_verified'] or digest.hexdigest() != manifest['subset_sha256']:
        raise ValueError('Real VAE decoder integrity check failed')
    begin = time.perf_counter()
    with torch.device('meta'):
        model = create_decoder()
    state = load_file(folder / 'decoder.safetensors')
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    load_seconds = time.perf_counter() - begin
    rows = []
    with torch.inference_mode():
        for frames in (8, 32, 256):
            x = torch.randn(1, 64, frames) * .2
            begin = time.perf_counter()
            output = model(x)
            first = time.perf_counter() - begin
            if list(output.shape) != [1, 2, frames * 1920] or not torch.isfinite(output).all():
                raise ValueError(f'Bad real decoder output for {frames} frames')
            begin = time.perf_counter()
            repeated = model(x)
            warm = time.perf_counter() - begin
            torch.testing.assert_close(output, repeated, rtol=0, atol=0)
            np.savez(folder / f'fixture-{frames}.npz', latent=x.numpy(), output=output.numpy())
            rows.append({'frames': frames, 'seconds_of_pcm': frames / 25,
                         'shape': list(output.shape), 'all_finite': True,
                         'first_seconds': first, 'warm_seconds': warm,
                         'peak': float(output.abs().max()),
                         'rms': float(output.square().mean().sqrt())})
            print(json.dumps(rows[-1]), flush=True)
    report = {'full_decoder_weights_loaded': True, 'strict_load': True,
              'matched_tensors': len(state), 'parameters': sum(p.numel() for p in model.parameters()),
              'every_source_storage_crc32_verified': True, 'subset_sha256': digest.hexdigest(),
              'full_source_sha256_verified': False, 'load_seconds': load_seconds, 'rows': rows,
              'scope': 'Windows CPU FP32, 4 threads, real pretrained decoder and synthetic latent fixtures; not a generated song or full codec RTF',
              'real_song_generated': False}
    (ROOT / 'reports/pretrained-vae-cpu.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
