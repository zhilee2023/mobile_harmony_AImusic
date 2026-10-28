"""Adapters for validated exported DiT and VAE, not random replacement codecs."""
import torch
from .contracts import CodecContract


class ExportedMixedCodec:
    def __init__(self, dit_path, vae_path, contract: CodecContract):
        contract.validate()
        self.contract = contract
        # Export contract: DiT accepts codec IDs, noise latent, timestep and prior
        # tail; VAE accepts [B,64,T]. Actual upstream export remains unvalidated.
        self.dit = torch.jit.load(str(dit_path), map_location="cpu").eval()
        self.vae = torch.jit.load(str(vae_path), map_location="cpu").eval()

    @torch.inference_mode()
    def decode_chunk(self, codes, noise, tail, steps):
        if codes.ndim != 2 or codes.shape[0] != 1 or steps < 1:
            raise ValueError("Expected mixed codes [1,T] and positive diffusion steps")
        if not bool(((codes >= 0) & (codes < self.contract.codebook_size)).all()):
            raise ValueError("Code out of range")
        if noise.shape != (1, self.contract.latent_channels, codes.shape[1]):
            raise ValueError("Latent/code frame mismatch")
        x = noise.clone()
        for i in range(steps):
            t = torch.tensor([1.0 - i / steps], dtype=x.dtype)
            velocity = self.dit(codes, x, t, tail)
            if velocity.shape != x.shape or not torch.isfinite(velocity).all():
                raise RuntimeError("Invalid DiT output")
            x = x - velocity / steps
        pcm = self.vae(x)
        expected = (1, self.contract.channels, codes.shape[1] * 1920)
        if pcm.shape != expected or not torch.isfinite(pcm).all():
            raise RuntimeError("Invalid VAE waveform output")
        return pcm, x
