"""Original Qwen language-head reference; no random music adapters."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    out = ROOT / 'models/qwen-language-fixture'
    if out.exists():
        raise ValueError('Preserve existing language reference')
    out.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-0.6b', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(ROOT / 'models/qwen3-0.6b', dtype=torch.float32,
        local_files_only=True, attn_implementation='sdpa').eval()
    background = '河边有一座安静的小城。每天清晨，居民走过石桥，去市场买花和面包。孩子们在操场读书，老人沿着河岸散步。'
    intro = tokenizer.encode('<|im_start|>user\n下面是一段旅行故事的背景：', add_special_tokens=False)
    tail = tokenizer.encode('\n请接着写一小段自然流畅的中文故事，不要解释写作方法。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
    fill = tokenizer.encode(background * 40, add_special_tokens=False)
    prefix = intro + fill[:512 - len(intro) - len(tail)] + tail
    assert len(prefix) == 512
    steps = 128
    snapshots = (0, 1, 15, 31, 63, 95, 127)
    tokens, durations = [], []
    cache = None
    started = time.perf_counter()
    with torch.inference_mode():
        for i in range(steps):
            ids = prefix if i == 0 else [tokens[-1]]
            tick = time.perf_counter()
            result = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            logits = result.logits[0, -1].float()
            assert logits.numel() == 151936 and torch.isfinite(logits).all()
            durations.append(time.perf_counter() - tick)
            if i in snapshots:
                logits.numpy().astype('<f4').tofile(out / f'logits-{i}.f32')
            tokens.append(int(logits.argmax()))
    fixture = {'source': 'Original unpruned pretrained Qwen3-0.6B, full language head; FP32 PyTorch SDPA reference',
               'prefix': prefix, 'teacherTokens': tokens, 'snapshotSteps': snapshots,
               'prefixText': tokenizer.decode(prefix), 'referenceGeneratedText': tokenizer.decode(tokens),
               'vocabulary': 151936, 'layers': 28, 'hidden': 1024, 'referenceSeconds': time.perf_counter() - started,
               'firstPrefillSeconds': durations[0], 'decodeSeconds': sum(durations[1:]),
               'fixedLengthIgnoresEosForBenchmark': True, 'qualityValidated': False}
    fixture['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()}
                        for p in out.glob('*.f32')}
    (out / 'fixture.json').write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in fixture.items() if k not in ('prefix', 'teacherTokens', 'files', 'prefixText')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
