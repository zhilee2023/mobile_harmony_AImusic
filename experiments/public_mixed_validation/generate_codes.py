"""PC MNN generation with trained Mixed LM and explicit source-style CFG.

Prefix rows are appended to a private embedding lookup file. This is exactly
an embedding gather, never feeding text tokenizer IDs into the music table.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np

BASE = Path(__file__).resolve().parent / 'artifacts'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=750)
    parser.add_argument('--guidance', type=float, default=1.5)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--deadline-seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.frames <= 1500 or not 1 <= args.guidance <= 3:
        raise ValueError('Bounded short music experiment')
    import MNN.llm as mllm
    model_dir = BASE / 'mixed-mnn-w4'
    if not (BASE / 'mixed-mnn-export.json').is_file():
        raise ValueError('Need completed verified export')
    embedding = BASE / 'generation-embeddings-bf16.bin'
    cond = np.fromfile(BASE / 'prefix-conditioned.f32', '<f4').reshape(953, 2048)
    null = np.fromfile(BASE / 'prefix-unconditional.f32', '<f4').reshape(953, 2048)
    fixture = np.concatenate((cond[:31], cond[-1:]), axis=0)
    if not embedding.exists():
        shutil.copyfile(BASE / 'music-embeddings-bf16.bin', embedding)
        with embedding.open('ab') as stream:
            for data in (cond, null, fixture):
                # BF16 round-to-nearest-even, preserving trained embedding sums.
                bits = data.astype('<f4').view('<u4')
                rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
                (rounded >> 16).astype('<u2').tofile(stream)
    if embedding.stat().st_size != (16386 + 953 * 2 + 32) * 2048 * 2:
        raise ValueError('Prefix embedding size mismatch')
    config = json.loads((model_dir / 'config.json').read_text())
    config.update(base_dir=BASE.as_posix() + '/', llm_config='mixed-mnn-w4/llm_config.json',
                  llm_model='mixed-mnn-w4/llm.mnn', llm_weight='mixed-mnn-w4/llm.mnn.weight',
                  tokenizer_file='mixed-mnn-w4/tokenizer.mtok',
                  embedding_file=embedding.name, thread_num=4, precision='high', memory='low',
                  backend_type='cpu', reuse_kv=False, attention_option=8)
    config_path = BASE / 'generation-config.json'
    config_path.write_text(json.dumps(config, indent=2))
    started = time.perf_counter()
    def load():
        model = mllm.create(str(config_path))
        if not model._c_obj.load():
            raise RuntimeError('MNN LLM load failed')
        model.reset()
        model.generate_init()
        return model
    def forward(model, ids):
        result = model.forward(ids)
        array = np.asarray(result.read(), dtype=np.float32).reshape(-1, 16385)[-1].copy()
        if not np.isfinite(array).all():
            raise RuntimeError('Nonfinite music logits')
        return array
    conditioned = load()
    print(f'Mixed W4 loaded in {time.perf_counter() - started:.3f}s', flush=True)
    if args.probe:
        rows = []
        expected_token = None
        for index in range(2):
            logits = forward(conditioned, list(range(16386 + 1906, 16386 + 1906 + 32)) if index == 0 else [expected_token])
            reference = np.fromfile(BASE / f'mapping-{index}-reference.f32', '<f4')
            error = float(np.linalg.norm(logits - reference) / max(np.linalg.norm(reference), 1e-12))
            expected_token = int(reference.argmax())
            row = {'phase': 'prefill' if index == 0 else 'decode', 'sameTeacherToken': expected_token,
                   'top1': int(logits.argmax()), 'logitsL2VsOriginal': error,
                   'top10Overlap': len(set(np.argsort(logits)[-10:]) & set(np.argsort(reference)[-10:]))}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if error > .2:
                raise RuntimeError('W4 output differs too much; do not generate blind')
        (BASE / 'mnn-runtime-probe.json').write_text(json.dumps({'rows': rows, 'success': True}, indent=2))
        return
    unconditional = load() if args.guidance > 1 else None
    codes, timings = [], []
    rng = np.random.default_rng(2307)
    eos = False
    prefill_seconds = 0
    for index in range(args.frames):
        if time.perf_counter() - started > args.deadline_seconds:
            raise TimeoutError('Generation deadline, retained partial tokens')
        tick = time.perf_counter()
        left = forward(conditioned, list(range(16386, 16386 + 953)) if index == 0 else [codes[-1]])
        if unconditional is not None:
            right = forward(unconditional, list(range(16386 + 953, 16386 + 1906)) if index == 0 else [codes[-1]])
            logits = right + args.guidance * (left - right)
        else:
            logits = left
        if codes:
            # Match source record_window=50: divide unique recent token logits.
            logits[np.unique(codes[-50:])] /= 1.1
        top = np.argpartition(logits, -50)[-50:]
        values = logits[top].astype(np.float64) / .9
        values -= values.max()
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        token = int(rng.choice(top, p=probabilities))
        elapsed = time.perf_counter() - tick
        timings.append(elapsed)
        if index == 0:
            prefill_seconds = elapsed
        if token == 16384:
            eos = True
            break
        if not 0 <= token < 16384:
            raise ValueError('Invalid codec ID')
        codes.append(token)
        if index == 0 or len(codes) % 25 == 0:
            report = {'codes': codes, 'frames': len(codes), 'audioSeconds': len(codes) / 25,
                      'source': 'trained public LeVo v2-large primary Mixed LM', 'mixedOnly': True,
                      'eos': eos, 'lmGuidance': args.guidance, 'temperature': .9, 'topK': 50,
                      'recentTokenWindow': 50, 'seed': 2307, 'bodyBits': 4, 'headBits': 8,
                      'prefixPositions': 953, 'prefillSeconds': prefill_seconds,
                      'totalLmSeconds': time.perf_counter() - started, 'tokenSeconds': timings,
                      'musicDecoded': False, 'maxFramesRequested': args.frames}
            (BASE / 'generated-mixed-codes.partial.json').write_text(json.dumps(report, indent=2))
            print(f'Mixed codes {len(codes)}/{args.frames}, audio {len(codes)/25:.2f}s, wall {time.perf_counter()-started:.1f}s', flush=True)
    if not codes:
        raise RuntimeError('LM ended before generating audio')
    report.update(codes=codes, frames=len(codes), audioSeconds=len(codes) / 25, eos=eos,
                  totalLmSeconds=time.perf_counter() - started, tokenSeconds=timings,
                  decodeTokensPerSecond=max(0, len(codes)-1) / max(sum(timings[1:]), 1e-12))
    (BASE / 'generated-mixed-codes.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('codes', 'tokenSeconds')}), flush=True)
    del conditioned, unconditional
    gc.collect()


if __name__ == '__main__':
    main()
