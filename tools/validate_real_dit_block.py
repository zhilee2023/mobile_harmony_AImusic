"""Load real block-0 weights from completed range parts and compare CPU kernels.

This is a real pretrained block test, not a full pretrained DiT/song test.
"""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import types
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from restore_mixed_dit import upstream_classes


def tensor_from_parts(key, info, parts):
    v = info['tensors'][key]; pos, end = [x + 8 + info['header_bytes'] for x in v['data_offsets']]
    files = sorted((int(p.name), p) for p in parts.iterdir() if p.name.isdigit())
    chunks = []
    while pos < end:
        candidate = next(((a, p) for a, p in files if a <= pos < a + p.stat().st_size), None)
        if candidate is None: raise FileNotFoundError(f'Pending download for {key} at source byte {pos}')
        a, p = candidate
        with p.open('rb') as f:
            f.seek(pos - a); data = f.read(min(end-pos, a+p.stat().st_size-pos))
        chunks.append(data); pos += len(data)
    raw = b''.join(chunks)
    dtype = {'F32': torch.float32, 'I64': torch.int64}[v['dtype']]
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(v['shape']).clone()


def sdpa_attention(self, q, k, v, attention_mask=None, head_mask=None):
    source = self._parity_source
    qr = q.transpose(1, 2)
    freqs = source.precompute_freqs_cis(dim=qr.shape[-1], end=qr.shape[1]).to(qr.device)
    q = source.apply_rotary_emb(qr, freqs).transpose(1, 2)
    k = source.apply_rotary_emb(k.transpose(1, 2), freqs).transpose(1, 2)
    if head_mask is not None: raise ValueError('Head masks not implemented in this experiment')
    scale = (v.shape[-1] ** -0.5) if self.scale_attn_weights else 1.0
    if self.scale_attn_by_inverse_layer_idx: scale /= self.layer_idx + 1
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                          dropout_p=0.0, is_causal=False, scale=scale), None


def main():
    torch.manual_seed(2307); torch.set_num_threads(4)
    root = ROOT / 'models/levo-mixed-codec'
    info = json.loads((root / 'source-header.json').read_text())
    Config, _, source = upstream_classes(ROOT / 'reference/LeVo')
    cfg = Config(n_embd=1200, n_layer=39, n_head=30, n_positions=1000,
                 use_cache=False, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
    cfg._attn_implementation = 'eager'
    prefix = 'cfm_wrapper.estimator.h.0.'
    state = {k.removeprefix(prefix): tensor_from_parts(k, info, root / 'parts')
             for k in info['tensors'] if k.startswith(prefix)}
    block = source.GPT2Block(cfg, layer_idx=0).eval()
    block.load_state_dict(state, strict=True)
    original = block.attn._attn
    block.attn._parity_source = source
    rows = []
    with torch.inference_mode():
        for frames in (8, 32, 128, 256):
            x = torch.randn(1, frames, 1200) * .2
            t = torch.randn(1, 7200) * .1
            mask = torch.zeros(1, 1, frames, frames)
            mask[..., -2:] = torch.finfo(torch.float32).min
            block.attn._attn = original
            ref = block(x, attention_mask=mask, time_step=t, use_cache=False)[0]
            block.attn._attn = types.MethodType(sdpa_attention, block.attn)
            out = block(x, attention_mask=mask, time_step=t, use_cache=False)[0]
            torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-4)
            timings = {}
            for name, method in (('eager', original), ('sdpa', types.MethodType(sdpa_attention, block.attn))):
                block.attn._attn = method; values = []
                for i in range(5):
                    begin = time.perf_counter(); block(x, attention_mask=mask, time_step=t, use_cache=False)
                    elapsed = time.perf_counter() - begin
                    if i: values.append(elapsed)
                timings[name + '_median_seconds'] = statistics.median(values)
            rows.append({'frames': frames, 'relative_l2': float(torch.linalg.vector_norm(out-ref) / torch.linalg.vector_norm(ref)),
                         'finite': bool(torch.isfinite(out).all()), **timings})
    report = {'real_pretrained_block': 0, 'parameters_loaded': sum(v.numel() for v in state.values()),
              'strict_load': True, 'rows': rows, 'full_dit_validated': False, 'real_song_generated': False,
              'source_revision': json.loads((root / 'source.json').read_text())['revision']}
    print(json.dumps(report, indent=2))
    (ROOT / 'reports/real-dit-block-cpu.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__': main()
