from dataclasses import dataclass
from pathlib import Path
import json
import torch
from torch import nn
from transformers import AutoModelForCausalLM


@dataclass
class MusicConfig:
    max_lyric_tokens: int = 512
    style_slots: int = 32
    music_codebook_size: int = 16384
    codec_id: str = "UNVERIFIED_LEVO_MIXED_1RVQ"


class MixedSongStudent(nn.Module):
    """A pretrained text backbone plus new, explicitly untrained music/style adapters.

    Prefix is [one style embedding, <=512 real lyric embeddings, audio BOS].
    Audio BOS/EOS live outside the codec's 0..K-1 code range. No dual-track LM.
    """
    def __init__(self, backbone, config=MusicConfig()):
        super().__init__()
        self.backbone = backbone
        self.config = config
        width = backbone.config.hidden_size
        self.style = nn.Embedding(config.style_slots, width)
        self.music = nn.Embedding(config.music_codebook_size + 2, width)
        self.music_head = nn.Linear(width, config.music_codebook_size + 1, bias=False)
        self.bos = config.music_codebook_size
        self.eos = config.music_codebook_size
        self.music_trained = False

    @classmethod
    def from_pretrained(cls, path="Qwen/Qwen3-0.6B", config=MusicConfig(), **kwargs):
        # HF enforces backbone tensor shapes; no ignore_mismatched_sizes shortcut.
        return cls(AutoModelForCausalLM.from_pretrained(path, **kwargs), config)

    def prefix(self, lyric_ids, style_ids, mask=None):
        if lyric_ids.ndim != 2 or not 1 <= lyric_ids.shape[1] <= self.config.max_lyric_tokens:
            raise ValueError("Lyrics require 1..512 tokenizer IDs; do not silently truncate")
        if style_ids.shape != (lyric_ids.shape[0],):
            raise ValueError("Exactly one style ID per song is required")
        x = self.backbone.get_input_embeddings()(lyric_ids)
        style = self.style(style_ids).unsqueeze(1).to(x.dtype)
        if mask is None:
            mask = torch.ones_like(lyric_ids)
        if mask.shape != lyric_ids.shape or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("Invalid lyric attention mask")
        return torch.cat((style, x), 1), torch.cat((torch.ones_like(mask[:, :1]), mask), 1)

    def teacher_logits(self, lyric_ids, style_ids, music_ids, mask=None):
        """Teacher forcing: predict first code from BOS, then next code/EOS."""
        if music_ids.ndim != 2 or music_ids.shape[0] != lyric_ids.shape[0]:
            raise ValueError("Music IDs must be [batch,time]")
        if music_ids.numel() and not bool(((music_ids >= 0) & (music_ids < self.config.music_codebook_size)).all()):
            raise ValueError("Only mixed codec IDs may be used as training inputs")
        prefix, attention = self.prefix(lyric_ids, style_ids, mask)
        bos = torch.full_like(lyric_ids[:, :1], self.bos)
        audio = self.music(torch.cat((bos, music_ids), 1)).to(prefix.dtype)
        x = torch.cat((prefix, audio), 1)
        attention = torch.cat((attention, torch.ones_like(bos), torch.ones_like(music_ids)), 1)
        out = self.backbone.model(inputs_embeds=x, attention_mask=attention, use_cache=False)
        return self.music_head(out.last_hidden_state[:, prefix.shape[1]:].to(self.music_head.weight.dtype))

    @torch.inference_mode()
    def generate_music(self, lyric_ids, style_ids, max_new_tokens=750, mask=None):
        stream = self.stream_music(lyric_ids, style_ids, max_new_tokens, mask)
        tokens = list(stream)
        return {"tokens": tokens, "eos": stream.ended_by_eos}

    def stream_music(self, lyric_ids, style_ids, max_new_tokens=750, mask=None):
        """Create a pausable decoder retaining one prefix and its growing KV cache.

        Consume IDs incrementally with next(stream), decode a codec window,
        then resume the same iterator. Destroying/closing it discards its KV.
        """
        if not self.music_trained:
            raise RuntimeError("Music adapter is untrained: text backbone weights alone cannot generate songs")
        if lyric_ids.shape[0] != 1 or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("Prototype decode supports one song and a positive token limit")
        return MusicTokenStream(self, lyric_ids, style_ids, max_new_tokens, mask)

    def load_music_adapter(self, folder):
        from safetensors.torch import load_file
        folder = Path(folder)
        meta = json.loads((folder / "adapter.json").read_text(encoding="utf8"))
        if meta.get("codec_id") != self.config.codec_id or meta.get("music_codebook_size") != self.config.music_codebook_size:
            raise ValueError("Codec/vocabulary mismatch; code IDs are not interchangeable")
        if meta.get("music_adapter_trained") is not True:
            raise ValueError("Checkpoint is not marked as trained")
        state = load_file(folder / "adapter.safetensors")
        for name, module in (("style", self.style), ("music", self.music), ("music_head", self.music_head)):
            module.load_state_dict({"weight": state[name + ".weight"]}, strict=True)
        self.music_trained = True


class MusicTokenStream:
    """Single-song decode lifecycle. Does not claim a trained music checkpoint."""
    @torch.inference_mode()
    def __init__(self, model, lyric_ids, style_ids, limit, mask):
        self.model = model
        prefix, attention = model.prefix(lyric_ids, style_ids, mask)
        self.one = torch.ones_like(lyric_ids[:, :1])
        bos = torch.full_like(lyric_ids[:, :1], model.bos)
        self.x = torch.cat((prefix, model.music(bos).to(prefix.dtype)), 1)
        self.attention = torch.cat((attention, self.one), 1)
        self.cache = None
        self.limit, self.tokens_generated = limit, 0
        self.ended_by_eos, self.closed = False, False

    def __iter__(self):
        return self

    @torch.inference_mode()
    def __next__(self):
        if self.closed:
            raise StopIteration
        if self.tokens_generated >= self.limit:
            self.close()
            raise StopIteration
        try:
            out = self.model.backbone.model(inputs_embeds=self.x, attention_mask=self.attention,
                past_key_values=self.cache, use_cache=True)
            self.cache = out.past_key_values
            scores = self.model.music_head(out.last_hidden_state[:, -1].to(self.model.music_head.weight.dtype))
            if not torch.isfinite(scores).all():
                raise RuntimeError('Non-finite music logits')
            token = scores.argmax(-1)
            if token.item() == self.model.eos:
                self.ended_by_eos = True
                self.close()
                raise StopIteration
            self.x = self.model.music(token[:, None]).to(self.x.dtype)
            self.attention = torch.cat((self.attention, self.one), 1)
            self.tokens_generated += 1
            return token.item()
        except StopIteration:
            raise
        except Exception:
            self.close()
            raise

    def close(self):
        self.cache = None
        self.x = self.attention = self.one = None
        self.model = None
        self.closed = True
