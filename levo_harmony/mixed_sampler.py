"""Inference primitives matching the inspected official Mixed 1RVQ equations.

Numerical parity tests are required before using these as a codec replacement.
No audio encoder is needed when mixed IDs are already available.
"""
import torch
from torch import nn
import torch.nn.functional as F


class MixedCodeConditioner(nn.Module):
    def __init__(self, state):
        super().__init__()
        prefix = 'rvq_bestrq_emb.quantizers.0.'
        self.register_buffer('codebook', state[prefix + 'codebook.weight'])
        v = state[prefix + 'out_proj.weight_v']
        g = state[prefix + 'out_proj.weight_g']
        # weight_norm(dim=0) normalizes all dimensions except output channels.
        weight = v * (g / torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True))
        self.register_buffer('weight', weight)
        self.register_buffer('bias', state[prefix + 'out_proj.bias'])

    def forward(self, ids):
        if ids.ndim != 2 or not bool(((ids >= 0) & (ids < self.codebook.shape[0])).all()):
            raise ValueError('Expected valid Mixed IDs [B,T]')
        embedded = F.embedding(ids, self.codebook).transpose(1, 2)
        return F.conv1d(embedded, self.weight, self.bias).transpose(1, 2)


class MixedFlowSampler(nn.Module):
    def __init__(self, cfm, aux):
        super().__init__()
        self.cfm = cfm
        self.register_buffer('mask_embedding', aux['mask_emb.weight'])

    @torch.inference_mode()
    def forward(self, mu, noise, incontext, mask_ids, steps=20, guidance=1.5):
        if mu.shape[:2] != noise.shape[:2] or incontext.shape != noise.shape:
            raise ValueError('Condition, noise and context must have equal time axes')
        if mask_ids.shape != noise.shape[:2] or steps < 1:
            raise ValueError('Invalid mask or step count')
        if mu.shape[-1] != 1024 or noise.shape[-1] != 64:
            raise ValueError('Official Mixed dimensions are mu=1024, latent=64')
        # IDs: 0=padding, 1=tail/context, 2=new audio. Context is a prefix.
        if not bool(((mask_ids >= 0) & (mask_ids <= 2)).all()):
            raise ValueError('Invalid context mask')
        context_mask = (mask_ids == 1).unsqueeze(-1)
        mask_features = F.embedding(mask_ids, self.mask_embedding)
        valid = mask_ids > 0
        attn = (valid[:, None, :, None] & valid[:, None, None, :]).to(mu.dtype)
        x = noise.clone(); context = incontext * context_mask
        batch = x.shape[0]
        for i in range(steps):
            # Official CFM runs t=0 -> 1 and ADDs dt*velocity.
            t = torch.full((batch,), i / steps, dtype=x.dtype, device=x.device)
            mixed_context = (1 - (1 - 1e-4) * t[:, None, None]) * noise + t[:, None, None] * context
            x = torch.where(context_mask, mixed_context, x)
            if guidance > 1:
                inputs = torch.cat((mask_features.repeat(2, 1, 1), context.repeat(2, 1, 1),
                                    torch.cat((torch.zeros_like(mu), mu)), x.repeat(2, 1, 1)), -1)
                pred = self.cfm(inputs, t.repeat(2), attn.repeat(2, 1, 1, 1))[..., -64:]
                uncond, cond = pred.chunk(2)
                velocity = uncond + guidance * (cond - uncond)
            else:
                inputs = torch.cat((mask_features, context, mu, x), -1)
                velocity = self.cfm(inputs, t, attn)[..., -64:]
            if not torch.isfinite(velocity).all():
                raise RuntimeError('Non-finite Mixed DiT velocity')
            x = x + velocity / steps
        return torch.where(context_mask, context, x)


def denormalize_latents(x, aux):
    counts = aux['normfeat.counts']
    if counts.item() < 10:
        mean = torch.zeros_like(aux['normfeat.sum_x']); std = torch.ones_like(mean)
    else:
        mean = aux['normfeat.sum_x'] / counts
        std = (aux['normfeat.sum_x2'] / counts - mean.square()).clamp(min=0).sqrt()
    return x * std.view(1, 1, -1) + mean.view(1, 1, -1)
