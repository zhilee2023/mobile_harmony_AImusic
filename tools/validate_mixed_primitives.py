"""Compare new CPU inference primitives against pinned official implementations.

Uses real VQ/mask/normalization buffers but a tiny random DiT for sampler parity.
No trained song LM or real song generation is claimed.
"""
import ast
from abc import ABC
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from restore_mixed_dit import RestoredMixedCfm
from levo_harmony.mixed_sampler import MixedCodeConditioner, MixedFlowSampler, denormalize_latents


def l2(a, b): return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp(min=1e-12))


def main():
    torch.manual_seed(2307); torch.set_num_threads(4)
    source = ROOT / 'reference/LeVo/codeclm/tokenizer/Flow1dVAE'
    spec = importlib.util.spec_from_file_location('official_rvq_reference', source / 'libs/rvq/descript_quantize3.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    aux = load_file(ROOT / 'models/levo-mixed-codec/mixed-aux.safetensors')
    rvq = module.ResidualVectorQuantize(input_dim=1024, n_codebooks=1, codebook_size=16384, codebook_dim=32)
    rvq.load_state_dict({k.removeprefix('rvq_bestrq_emb.'): v for k, v in aux.items() if k.startswith('rvq_bestrq_emb.')}, strict=True)
    rvq.eval(); ids = torch.tensor([[0, 1, 3, 1024, 8192, 16383]])
    with torch.inference_mode():
        expected = rvq.from_codes(ids[:, None])[0].transpose(1, 2)
        actual = MixedCodeConditioner(aux)(ids)
    # Extract only reference classes; no heavy MusicFM, Hubert or audio imports.
    tree = ast.parse((source / 'model_1rvq.py').read_text(encoding='utf8'))
    chosen = {'SampleProcessor', 'Feature1DProcessor', 'BASECFM'}
    subset = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in chosen], type_ignores=[])
    scope = {'torch': torch, 'nn': nn, 'ABC': ABC, 'F': F, 'tqdm': lambda x: x}
    exec(compile(ast.fix_missing_locations(subset), 'official_reference_classes', 'exec'), scope)
    norm = scope['Feature1DProcessor'](dim=64)
    norm.load_state_dict({k.removeprefix('normfeat.'): v for k, v in aux.items() if k.startswith('normfeat.')}, strict=True)
    x = torch.randn(1, 6, 64)
    denorm_l2 = l2(denormalize_latents(x, aux), norm.return_sample(x.transpose(1, 2)).transpose(1, 2))

    # Exact 1200-wide conditioning contract, reduced layer count and FFN only.
    cfm = RestoredMixedCfm(ROOT / 'reference/LeVo', layers=1, heads=30, inner=64).eval()
    # Dropout zero; tiny random weights are intentionally NOT public DiT weights.
    official = scope['BASECFM'](cfm.estimator, cfm.mlp, 6).eval()
    sampler = MixedFlowSampler(cfm, aux)
    noise = torch.randn(1, 6, 64); context = torch.randn_like(noise)
    rows = []
    for context_frames in (0, 2):
        masks = torch.full((1, 6), 2, dtype=torch.long); masks[:, :context_frames] = 1
        inc = context * (masks == 1).unsqueeze(-1)
        mask_features = F.embedding(masks, aux['mask_emb.weight'])
        attn = torch.ones(1, 1, 6, 6)
        for guidance in (1.0, 1.5):
            with torch.inference_mode():
                ref = official.solve_euler(noise.clone(), mask_features, inc, context_frames,
                                           torch.linspace(0, 1, 4), actual, attn, guidance)
                ref[:, :context_frames] = inc[:, :context_frames]
                out = sampler(actual, noise, inc, masks, steps=3, guidance=guidance)
            relative = l2(out, ref)
            torch.testing.assert_close(out, ref, atol=3e-5, rtol=3e-5)
            rows.append({'context_frames': context_frames, 'guidance': guidance, 'relative_l2': relative,
                         'finite': bool(torch.isfinite(out).all())})
    report = {'rvq_real_weights_relative_l2': l2(actual, expected),
              'normalization_real_stats_relative_l2': denorm_l2,
              'sampler_parity': rows, 'dit_weights': 'tiny random reference, not full pretrained DiT',
              'real_song_generated': False}
    assert report['rvq_real_weights_relative_l2'] < 1e-6 and denorm_l2 < 1e-6
    (ROOT / 'reports/mixed-primitives-parity.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
