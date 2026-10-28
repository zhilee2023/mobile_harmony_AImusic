"""Reference-only bridge to a separately obtained official LeVo source checkout.

This delegates sampling/normalization/chunk overlap to upstream instead of
pretending the generic export contract is already checkpoint-compatible.
Upstream dependencies and checkpoints must be installed separately.
"""
from pathlib import Path
import sys
import torch


def load_official_mixed_codec(checkout, checkpoint, vae_config, vae_checkpoint, device="cpu"):
    root = Path(checkout).resolve()
    if not (root / "codeclm/tokenizer/Flow1dVAE/generate_1rvq.py").is_file():
        raise ValueError("Expected official LeVo checkout with Mixed 1RVQ codec")
    for p in (checkpoint, vae_config, vae_checkpoint):
        if not Path(p).is_file(): raise FileNotFoundError(p)
    # The official code uses relative internal imports and config paths. Caller
    # must launch from its Flow1dVAE directory; no global cwd mutation here.
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "codeclm/tokenizer/Flow1dVAE"))
    from codeclm.tokenizer.Flow1dVAE.generate_1rvq import Tango
    return Tango(model_path=str(Path(checkpoint).resolve()),
                 vae_config=str(Path(vae_config).resolve()),
                 vae_model=str(Path(vae_checkpoint).resolve()), device=device)


@torch.inference_mode()
def decode_official_mixed(codec, ids, steps=20, chunk_seconds=10):
    if ids.ndim != 3 or ids.shape[:2] != (1, 1) or not ids.shape[-1]:
        raise ValueError("Official Mixed codec expects [batch=1,codebooks=1,time]")
    if not bool(((ids >= 0) & (ids < 16384)).all()) or steps < 1 or chunk_seconds < 1:
        raise ValueError("Invalid codec IDs or sampling settings")
    return codec.code2sound(ids, duration=chunk_seconds, guidance_scale=1.5, num_steps=steps)
