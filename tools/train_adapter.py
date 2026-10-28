"""Teacher-forced supervised/distillation training on real mixed-code sequences.

JSONL fields: lyric_ids (<=512), style_id, music_ids, codec_id.
Optional teacher_logits_file: torch-saved [T+1,K+1] tensor, same exact codec.
No licensed music is generated or downloaded by this script.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save_file
from levo_harmony.model import MixedSongStudent, MusicConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--codec-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train-backbone", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.steps < 1 or args.codec_id.startswith("UNVERIFIED"):
        p.error("Positive steps and a verified codec identity are required")
    rows = [json.loads(x) for x in args.dataset.read_text(encoding="utf8").splitlines() if x.strip()]
    if not rows or any(x["codec_id"] != args.codec_id for x in rows):
        p.error("Empty dataset or mismatched codec")
    model = MixedSongStudent.from_pretrained(args.backbone, MusicConfig(codec_id=args.codec_id)).to(args.device)
    model.backbone.requires_grad_(args.train_backbone)
    model.train()
    opt = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=args.lr)
    losses = []
    for step in range(args.steps):
        row = rows[step % len(rows)]
        lyric = torch.tensor([row["lyric_ids"]], device=args.device)
        style = torch.tensor([row["style_id"]], device=args.device)
        music = torch.tensor([row["music_ids"]], device=args.device, dtype=torch.long)
        logits = model.teacher_logits(lyric, style, music)
        targets = torch.cat((music, torch.full((1, 1), model.eos, device=args.device)), 1)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if "teacher_logits_file" in row:
            teacher = torch.load(args.dataset.parent / row["teacher_logits_file"], map_location=args.device, weights_only=True)
            if teacher.shape != logits.shape[1:] or not torch.isfinite(teacher).all():
                raise ValueError("Teacher shape/values mismatch")
            kd = torch.nn.functional.kl_div(logits[0].log_softmax(-1), teacher.softmax(-1), reduction="batchmean")
            loss = 0.5 * loss + 0.5 * kd
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    save_file({n + ".weight": getattr(model, n).weight.detach().cpu().contiguous()
               for n in ("style", "music", "music_head")}, args.output / "adapter.safetensors")
    if args.train_backbone:
        model.backbone.save_pretrained(args.output / "backbone")
    meta = {"codec_id": args.codec_id, "music_codebook_size": model.config.music_codebook_size,
            "music_adapter_trained": True, "quality_validated": False, "steps": args.steps,
            "backbone_source": args.backbone, "backbone_finetuned": args.train_backbone,
            "loss_first": losses[0], "loss_last": losses[-1]}
    (args.output / "adapter.json").write_text(json.dumps(meta, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
