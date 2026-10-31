"""Reuse exact smoothed W4 weights; remove only static activation descriptors."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'models/qwen-language-smooth-a8'
out = ROOT / 'models/qwen-language-smooth-noact'
if out.exists():
    raise ValueError('Preserve previous ablation')
out.mkdir()
for name in ('llm.mnn.weight', 'embeddings_bf16.bin', 'tokenizer.mtok'):
    os.link(src / name, out / name)
for name in ('config.json', 'llm_config.json'):
    shutil.copyfile(src / name, out / name)
shutil.copyfile(src / 'llm-noact.mnn', out / 'llm.mnn')
meta = json.loads((src / 'language-probe.json').read_text(encoding='utf8'))
meta.update(variant='smooth-noact', activationRangeBitsRequested=None, quantizedTensorDescriptions=0,
            quantInfoExamples=[], ablation='Exact same smoothed W4/g64 weights, static activation metadata removed')
for name, info in meta['files'].items():
    p = out / name
    info.update(bytes=p.stat().st_size, sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest())
(out / 'language-probe.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
print('Prepared exact-weight static-metadata ablation')
