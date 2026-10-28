"""Tiny synthetic-ID overfit test. Produces no audio and no usable music weights."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from levo_harmony.model import MixedSongStudent, MusicConfig

torch.set_num_threads(2)
torch.manual_seed(2307)
cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                 vocab_size=64, tie_word_embeddings=True)
model = MixedSongStudent(Qwen3ForCausalLM(cfg), MusicConfig(music_codebook_size=16))
model.backbone.requires_grad_(False)
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.02)
lyric = torch.tensor([[1, 2, 3], [5, 6, 7]])
style = torch.tensor([0, 1])
codes = torch.tensor([[4, 5, 6], [8, 9, 10]])
targets = torch.cat((codes, torch.full((2, 1), 16)), 1)
losses = []
for _ in range(60):
    logits = model.teacher_logits(lyric, style, codes)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
assert losses[-1] < losses[0] * 0.5
report = {"synthetic_only": True, "steps": 60, "loss_first": losses[0], "loss_last": losses[-1],
          "optimizer_path_passed": True, "public_0_6b_weights_loaded": False,
          "music_training_or_quality_validated": False, "audio_generated": False}
path = Path(__file__).resolve().parents[1] / 'reports/synthetic-training-smoke.json'
path.write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report, indent=2))
