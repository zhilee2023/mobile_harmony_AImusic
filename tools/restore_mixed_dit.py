"""Restore only the missing Mixed DiT container, using pinned upstream classes.

Run with isolated legacy codec dependencies before .deps on PYTHONPATH.
This script does NOT deserialize pickle checkpoints or silently ignore keys.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REV = "653cbcf4716101834900c75b7d5da43b07e15d5b"
MODEL_MODULE = "models_gpt.models.gpt2_rope2_time_new_correct_mask_noncasual_reflow"


def upstream_classes(checkout):
    path = Path(checkout).resolve() / "codeclm/tokenizer/Flow1dVAE"
    if not (path / "models_gpt/models/gpt2_config.py").is_file():
        raise FileNotFoundError("Expected pinned LeVo source checkout")
    sys.path.insert(0, str(path))
    config = importlib.import_module("models_gpt.models.gpt2_config")
    source = importlib.import_module(MODEL_MODULE)
    return config.GPT2Config, source.GPT2Model, source


class RestoredMixedCfm(nn.Module):
    def __init__(self, checkout, hidden=1200, layers=39, heads=30, inner=None, positions=1000):
        super().__init__()
        Config, Model, _ = upstream_classes(checkout)
        cfg = Config(n_positions=positions, n_layer=layers, n_head=heads, n_embd=hidden,
                     n_inner=inner, use_cache=False, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        cfg._attn_implementation = "eager"
        self.estimator = Model(cfg)
        self.mlp = nn.Sequential(nn.Linear(hidden, 1024), nn.SiLU(),
                                 nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, 768))

    def forward(self, inputs, timestep, attention_mask):
        return self.estimator(inputs_embeds=inputs, time_step=timestep,
                              attention_mask=attention_mask, use_cache=False).last_hidden_state


def compare_header(model, header):
    expected = {k.removeprefix("cfm_wrapper."): v['shape'] for k, v in header.items()
                if k.startswith("cfm_wrapper.")}
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    return {"missing": sorted(expected.keys() - actual.keys()),
            "unexpected": sorted(actual.keys() - expected.keys()),
            "shape_mismatches": {k: {"weight": expected[k], "module": actual[k]}
                                 for k in expected.keys() & actual.keys() if expected[k] != actual[k]},
            "matched_tensor_count": sum(actual.get(k) == v for k, v in expected.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkout", type=Path, default=ROOT / "reference/LeVo")
    p.add_argument("--header", type=Path, default=ROOT / "reference/mixed-codec-header.json")
    args = p.parse_args()
    with torch.device("meta"):
        model = RestoredMixedCfm(args.checkout)
    compare = compare_header(model, json.loads(args.header.read_text(encoding="utf8")))
    report = {"source_revision": REV, "restored_missing_constructor": True,
              "meta_only": True, "real_weights_loaded": False,
              "parameters": sum(p.numel() for p in model.parameters()), **compare}
    report["exact_header_shape_match"] = not any(compare[k] for k in ('missing', 'unexpected', 'shape_mismatches'))
    if not report['exact_header_shape_match']:
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    (ROOT / 'reports/mixed-dit-restoration.json').write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__': main()
