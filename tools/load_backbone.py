"""Explicit opt-in public checkpoint download/load; this does not create music weights."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from levo_harmony.model import MixedSongStudent

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True, help="HF ID or local pretrained directory")
p.add_argument("--adapter", help="Trained compatible adapter directory")
args = p.parse_args()
model = MixedSongStudent.from_pretrained(args.checkpoint)
if args.adapter:
    model.load_music_adapter(args.adapter)
print({"parameters": sum(p.numel() for p in model.parameters()), "backbone_loaded": True,
       "music_adapter_trained": model.music_trained, "can_generate_song": model.music_trained})
