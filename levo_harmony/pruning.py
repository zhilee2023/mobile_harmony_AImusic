"""Explicit structural experiment, requiring recovery training before music use."""
import torch


def prune_backbone_depth(student, remove_layers=(9, 18)):
    layers = student.backbone.model.layers
    remove = set(remove_layers)
    if not remove or min(remove) < 0 or max(remove) >= len(layers) or len(remove) >= len(layers):
        raise ValueError("Invalid layer removal set")
    indices = [i for i in range(len(layers)) if i not in remove]
    kept = [layers[i] for i in indices]
    configs = []
    for cfg in (student.backbone.config, student.backbone.model.config):
        if not any(id(cfg) == id(previous[0]) for previous in configs):
            types = getattr(cfg, 'layer_types', None)
            if types is not None and len(types) != len(layers):
                raise ValueError('Layer metadata does not match the original backbone')
            configs.append((cfg, None if types is None else [types[i] for i in indices]))
    student.backbone.model.layers = torch.nn.ModuleList(kept)
    for i, layer in enumerate(kept):
        layer.self_attn.layer_idx = i  # KV-cache indices must be contiguous.
    for cfg, types in configs:
        cfg.num_hidden_layers = len(kept)
        if types is not None:
            cfg.layer_types = types
        if getattr(cfg, 'max_window_layers', None) is not None:
            cfg.max_window_layers = min(cfg.max_window_layers, len(kept))
    student.music_trained = False
    return {"removed_layers": sorted(remove), "remaining_layers": len(kept),
            "parameters": sum(p.numel() for p in student.parameters()),
            "layer_metadata_reindexed": True,
            "recovery_training_required": True, "music_quality_validated": False}
