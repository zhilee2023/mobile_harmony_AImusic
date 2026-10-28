"""No large allocation: inspect exact full-size student parameters on meta device."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from levo_harmony.model import MixedSongStudent
from levo_harmony.contracts import kv_cache_bytes
from levo_harmony.pruning import prune_backbone_depth

cfg = Qwen3Config(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                  num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                  vocab_size=151936, tie_word_embeddings=True)
with torch.device("meta"):
    model = MixedSongStudent(Qwen3ForCausalLM(cfg))
n = sum(p.numel() for p in model.parameters())
report = {"actual_parameters": n, "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
          "music_adapter_trained": False, "shape_only_not_weight_load": True,
          "lyrics_max_tokens": 512, "style_prefix_tokens": 1, "music_codebooks": 1,
          "kv_fp16_bytes": {str(s): kv_cache_bytes(28, 8, 128, 512 + 2 + s * 25) for s in (30, 60, 180, 300)},
          "note": "KV includes max lyric prefix; no CFG duplication, activation or packing overhead"}
report['optional_26_layer_pruning_experiment'] = prune_backbone_depth(model)
print(json.dumps(report, indent=2))
