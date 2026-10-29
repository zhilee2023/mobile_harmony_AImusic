"""Small, pinned Mixed-codec inference buffers for CPU parity checks."""
import hashlib
import json
from pathlib import Path
import struct
from download_mixed_codec import get_range

root = Path(__file__).resolve().parents[1]
source = json.loads((root / 'models/levo-mixed-codec/source.json').read_text())
info = json.loads((root / 'models/levo-mixed-codec/source-header.json').read_text())
header = info['tensors']; base = 8 + info['header_bytes']
selected = {k: v for k, v in header.items() if k.startswith(('mask_emb.', 'normfeat.', 'rvq_bestrq_emb.', 'zero_cond_embedding1'))}
lo = min(v['data_offsets'][0] for v in selected.values())
hi = max(v['data_offsets'][1] for v in selected.values())
url = f'https://huggingface.co/{source["repo"]}/resolve/{source["revision"]}/{source["file"]}'
data = get_range(url, base + lo, base + hi - 1, source['bytes'])
out_header = {}; payload = []
pos = 0
for k, v in sorted(selected.items(), key=lambda x: x[1]['data_offsets'][0]):
    a, b = v['data_offsets']; x = data[a-lo:b-lo]
    out_header[k] = {'dtype': v['dtype'], 'shape': v['shape'], 'data_offsets': [pos, pos+len(x)]}
    payload.append(x); pos += len(x)
encoded = json.dumps(out_header, separators=(',', ':')).encode(); encoded += b' ' * (-len(encoded) % 8)
out = root / 'models/levo-mixed-codec/mixed-aux.safetensors'
with out.open('wb') as f:
    f.write(struct.pack('<Q', len(encoded))); f.write(encoded)
    for x in payload: f.write(x)
print(json.dumps({'bytes': out.stat().st_size, 'tensors': len(selected),
                  'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'full_source_hash_verified': False}, indent=2))
