"""Decode real generated Mixed IDs with the matching pretrained MNN codec.

Separate process from LM to release LM weights/KV. CFG, Euler direction,
context re-noising, latent normalization and overlapping PCM follow the
previously parity-validated Mixed 1RVQ sampler. No random music-head tokens.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import wave
import numpy as np

BASE = Path(__file__).resolve().parent / 'artifacts'
PROJECT = Path('C:/levo-harmony')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--codes', type=Path, default=BASE / 'generated-mixed-codes.json')
    parser.add_argument('--max-frames', type=int, default=750)
    args = parser.parse_args()
    if not 2 <= args.steps <= 50 or not 1 <= args.max_frames <= 1500:
        raise ValueError('Bounded music validation only')
    import MNN
    meta = json.loads(args.codes.read_text())
    if meta['source'] != 'trained public LeVo v2-large primary Mixed LM':
        raise ValueError('Need genuinely pretrained LM tokens')
    codes = np.asarray(meta['codes'][:args.max_frames], dtype=np.int64)
    if codes.ndim != 1 or not len(codes) or codes.min() < 0 or codes.max() >= 16384:
        raise ValueError('Invalid generated music IDs')
    output = BASE / f'public-mixed-{len(codes)/25:.2f}s-dit{args.steps}-w4-w8.wav'
    if output.exists():
        raise FileExistsError('Preserve previous listening experiment')
    aux = PROJECT / 'models/three-module-bundle/aux'
    def load(name, shape):
        result = np.fromfile(aux / (name + '.f32'), dtype='<f4').reshape(shape)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite codec auxiliary tensor')
        return result
    book, weight, bias = load('codebook', (16384, 32)), load('weight', (1024, 32)), load('bias', (1024,))
    mask_embed = load('mask', (3, 48))
    mean, std = load('mean', (64,)), load('std', (64,))
    started = time.perf_counter()
    def session(path):
        tick = time.perf_counter()
        net = MNN.Interpreter(str(path))
        obj = net.createSession({'backend': 'CPU', 'numThread': 4, 'precision': 1, 'memory': 2})
        return net, obj, time.perf_counter() - tick
    def run(net, obj, values, name, shape):
        for key, data in values.items():
            data = np.ascontiguousarray(data, dtype=np.float32)
            host = MNN.Tensor(data.shape, MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe)
            if not net.getSessionInput(obj, key).copyFrom(host):
                raise RuntimeError('Codec input copy failed: ' + key)
        status = net.runSession(obj)
        if status not in (None, 0):
            raise RuntimeError('Codec forward failed: ' + str(status))
        data = np.asarray(net.getSessionOutput(obj, name).getNumpyData()).copy().reshape(shape)
        if not np.isfinite(data).all():
            raise RuntimeError('Codec nonfinite output')
        return data
    dit, dit_session, dit_setup = session(PROJECT / 'models/full-dit-256/full-w8.mnn')
    vae, vae_session, vae_setup = session(PROJECT / 'models/vae-stream-256/vae-256-w8.mnn')
    print(f'Codec ready, DiT setup={dit_setup:.3f}s, VAE setup={vae_setup:.3f}s', flush=True)
    rng = np.random.default_rng(2307)
    last_latent = None
    tail = None
    windows = []
    total_energy, total_samples, clipped_samples = 0.0, 0, 0
    peak = 0.0
    start = 0
    samples_per_frame = 1920
    overlap = 64 * samples_per_frame
    with wave.open(str(output.with_suffix('.partial.wav')), 'wb') as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(48000)
        def emit(pcm):
            nonlocal total_energy, total_samples, clipped_samples, peak
            if not np.isfinite(pcm).all():
                raise RuntimeError('Nonfinite PCM')
            total_energy += float(np.sum(pcm.astype(np.float64)**2))
            total_samples += pcm.size
            clipped_samples += int(np.count_nonzero(np.abs(pcm) > 1))
            peak = max(peak, float(np.max(np.abs(pcm))))
            encoded = (np.clip(pcm.T, -1, 1) * 32767).astype('<i2')
            writer.writeframes(encoded.tobytes())
        while True:
            index = len(windows)
            valid = min(256, len(codes) - start)
            last = start + valid >= len(codes)
            current_codes = np.zeros(256, dtype=np.int64)
            current_codes[:valid] = codes[start:start+valid]
            mu = (book[current_codes] @ weight.T + bias).astype(np.float32)
            ids = np.zeros(256, dtype=np.int64)
            ids[:valid] = 2
            context = np.zeros((256, 64), dtype=np.float32)
            if last_latent is not None:
                context[:64] = last_latent[-64:]
                ids[:64] = 1
            context_mask = (ids == 1)[:, None]
            context_features = context * context_mask
            masks = mask_embed[ids]
            active = ids > 0
            attention = np.broadcast_to((active[:, None] & active[None, :]).astype(np.float32), (2, 256, 256)).copy()
            noise = rng.standard_normal((256, 64), dtype=np.float32)
            x = noise.copy()
            tick = time.perf_counter()
            for step in range(args.steps):
                t = step / args.steps
                mixed_context = (1 - (1 - 1e-4) * t) * noise + t * context_features
                x = np.where(context_mask, mixed_context, x)
                uncond = np.concatenate((masks, context_features, np.zeros_like(mu), x), axis=-1)
                cond = np.concatenate((masks, context_features, mu, x), axis=-1)
                velocity = run(dit, dit_session, {'inputs': np.stack((uncond, cond)),
                               'timestep': np.asarray([t, t], dtype=np.float32), 'attention_mask': attention},
                               'velocity', (2, 256, 64))
                x += (velocity[0] + 1.5 * (velocity[1] - velocity[0])) / args.steps
                if not np.isfinite(x).all():
                    raise RuntimeError('Euler latent nonfinite')
                if step == 0 or (step + 1) % 5 == 0 or step + 1 == args.steps:
                    print(f'Audio chunk {index+1}: DiT {step+1}/{args.steps}', flush=True)
            x = np.where(context_mask, context_features, x)
            dit_seconds = time.perf_counter() - tick
            last_latent = x.copy()
            latent = ((x * std + mean).T)[None].astype(np.float32)
            tick = time.perf_counter()
            pcm = run(vae, vae_session, {'latent': latent}, 'pcm', (2, 256 * samples_per_frame))
            vae_seconds = time.perf_counter() - tick
            pcm = pcm[:, :valid * samples_per_frame]
            if tail is not None:
                ramp = np.linspace(0, 1, overlap, dtype=np.float32)[None]
                pcm[:, :overlap] = tail * (1 - ramp) + pcm[:, :overlap] * ramp
            if last:
                emit(pcm)
            else:
                tail = pcm[:, -overlap:].copy()
                emit(pcm[:, :-overlap])
            row = {'chunk': index + 1, 'startFrame': start, 'validFrames': valid,
                   'contextFrames': 0 if index == 0 else 64, 'ditSeconds': dit_seconds,
                   'vaeSeconds': vae_seconds, 'latentPeak': float(np.max(np.abs(x))),
                   'pcmPeak': float(np.max(np.abs(pcm)))}
            windows.append(row)
            print(json.dumps(row), flush=True)
            if last:
                break
            start += 192
    if total_samples != 2 * len(codes) * samples_per_frame:
        raise RuntimeError('Overlapping WAV duration mismatch')
    output.with_suffix('.partial.wav').rename(output)
    report = {'sourceCheckpoint': 'official SongGeneration-v2-large', 'mixedOnly': True,
              'trainedLmTokens': True, 'randomMusicHead': False, 'lmGuidance': meta['lmGuidance'],
              'ditSteps': args.steps, 'ditGuidance': 1.5, 'lmBodyBits': 4, 'lmHeadBits': 8,
              'ditWeightBits': 8, 'vaeWeightBits': 8, 'phoneTest': False, 'device': 'Windows CPU',
              'wave': str(output), 'sampleRate': 48000, 'channels': 2, 'durationSeconds': len(codes) / 25,
              'pcmRmsBeforeClip': (total_energy / total_samples)**.5, 'pcmPeakBeforeClip': peak,
              'clippedSampleFraction': clipped_samples / total_samples, 'chunks': windows,
              'codecSeconds': time.perf_counter() - started, 'lmSeconds': meta['totalLmSeconds'],
              'totalComputeRtf': (time.perf_counter() - started + meta['totalLmSeconds']) / (len(codes)/25),
              'audioFileGenerated': True, 'listeningQualityAccepted': False,
              'scope': 'Actual public pretrained Mixed LM->matching pretrained codec; listening validation pending'}
    (BASE / f'music-result-dit{args.steps}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    del dit_session, vae_session, dit, vae
    gc.collect()


if __name__ == '__main__':
    main()
