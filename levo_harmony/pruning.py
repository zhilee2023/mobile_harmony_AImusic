"""Explicit structural experiment, requiring recovery training before music use."""
import torch


def prune_backbone_depth(student, remove_layers=(9, 18)):
    layers = student.backbone.model.layers
    remove = set(remove_layers)
    if not remove or min(remove) < 0 or max(remove) >= len(layers) or len(remove) >= len(layers):
        raise ValueError("Invalid layer removal set")
    kept = [layer for i, layer in enumerate(layers) if i not in remove]
    student.backbone.model.layers = torch.nn.ModuleList(kept)
    for i, layer in enumerate(kept):
        layer.self_attn.layer_idx = i  # KV-cache indices must be contiguous.
    student.backbone.config.num_hidden_layers = len(kept)
    student.backbone.model.config.num_hidden_layers = len(kept)
    student.music_trained = False
    return {"removed_layers": sorted(remove), "remaining_layers": len(kept),
            "parameters": sum(p.numel() for p in student.parameters()),
            "recovery_training_required": True, "music_quality_validated": False}
