"""Shape and identity checks shared by conversion and runtime staging."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CodecContract:
    codec_id: str
    codebook_size: int = 16384
    codebooks: int = 1
    frame_rate: int = 25
    latent_channels: int = 64
    sample_rate: int = 48000
    channels: int = 2

    def validate(self):
        if self.codec_id.startswith("UNVERIFIED") or not self.codec_id:
            raise ValueError("Codec identity must be verified against real weights")
        if (self.codebooks, self.frame_rate, self.latent_channels, self.sample_rate, self.channels) != (1, 25, 64, 48000, 2):
            raise ValueError("This route requires mixed 1RVQ/25Hz/64D/48k stereo")

    def frames(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Duration must be finite and positive")
        return math.ceil(seconds * self.frame_rate)


def kv_cache_bytes(layers, kv_heads, head_dim, positions, bytes_per_element=2, batch=1):
    return 2 * layers * kv_heads * head_dim * positions * bytes_per_element * batch
