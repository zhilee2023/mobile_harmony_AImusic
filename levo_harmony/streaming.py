"""Bounded token windows and PCM overlap for the proposed 256-frame codec.

This is scheduling infrastructure, not proof of pretrained codec audio quality.
The caller keeps the LM KV cache; this module never restarts lyric prefill.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MusicChunk:
    start_frame: int
    tokens: tuple
    valid_frames: int
    context_frames: int
    final: bool


class TokenWindows:
    def __init__(self, window=256, overlap=64, codebook_size=16384):
        if not 0 < overlap < window or codebook_size < 2:
            raise ValueError('Invalid window, overlap or codebook')
        self.window, self.overlap = window, overlap
        self.hop = window - overlap
        self.codebook_size = codebook_size
        self.buffer, self.start, self.produced = [], 0, 0
        self.closed = False
        self.peak_buffer_frames = 0

    def push(self, tokens):
        if self.closed:
            raise ValueError('Cannot push tokens after EOS')
        for token in tokens:
            if not isinstance(token, int) or not 0 <= token < self.codebook_size:
                raise ValueError('Only codec IDs may enter a chunk; filter EOS first')
            self.buffer.append(token)
            self.peak_buffer_frames = max(self.peak_buffer_frames, len(self.buffer))
            if len(self.buffer) == self.window:
                chunk = MusicChunk(self.start, tuple(self.buffer), self.window,
                                   self.overlap if self.produced else 0, False)
                self.buffer = self.buffer[self.hop:]
                self.start += self.hop
                self.produced += 1
                yield chunk

    def finish(self):
        if self.closed:
            raise ValueError('EOS already processed')
        self.closed = True
        context = self.overlap if self.produced else 0
        if len(self.buffer) <= context:
            # Exact full-window EOS: the existing pending PCM tail is sufficient.
            self.buffer.clear()
            return None
        valid = len(self.buffer)
        # Fixed graph requires 256 positions. Repeat only at final EOS and crop
        # decoded PCM to valid_frames; never invent future tokens while waiting.
        padded = (self.buffer * ((self.window + valid - 1) // valid))[:self.window]
        chunk = MusicChunk(self.start, tuple(padded), valid, context, True)
        self.buffer.clear()
        return chunk


class PCMOverlap:
    def __init__(self, window=256, overlap=64, samples_per_frame=1920, channels=2):
        if not 0 < overlap < window or samples_per_frame < 1 or channels < 1:
            raise ValueError('Invalid PCM contract')
        self.window, self.overlap = window, overlap
        self.samples_per_frame, self.channels = samples_per_frame, channels
        self.tail = np.empty((channels, 0), dtype=np.float32)
        self.next_start, self.closed = 0, False

    def accept(self, chunk, pcm):
        if self.closed or chunk.start_frame != self.next_start:
            raise ValueError('Out-of-order or closed PCM stream')
        if len(chunk.tokens) != self.window or not 0 < chunk.valid_frames <= self.window:
            raise ValueError('Chunk shape does not match fixed graph')
        context = self.overlap if chunk.start_frame else 0
        if chunk.context_frames != context or chunk.valid_frames < context:
            raise ValueError('Invalid latent context contract')
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.shape != (self.channels, self.window * self.samples_per_frame) or not np.isfinite(pcm).all():
            raise ValueError('Bad decoder PCM shape or non-finite samples')
        pcm = pcm[:, :chunk.valid_frames * self.samples_per_frame].copy()
        if context:
            count = context * self.samples_per_frame
            if self.tail.shape[1] != count:
                raise ValueError('Missing previous overlap PCM')
            ramp = np.linspace(0, 1, count, dtype=np.float32)[None, :]
            pcm[:, :count] = self.tail * (1-ramp) + pcm[:, :count] * ramp
        if chunk.final:
            self.closed = True
            self.tail = np.empty((self.channels, 0), dtype=np.float32)
            return pcm
        count = self.overlap * self.samples_per_frame
        self.tail = pcm[:, -count:].copy()
        self.next_start += self.window - self.overlap
        return pcm[:, :-count]

    def finish(self):
        if self.closed:
            return np.empty((self.channels, 0), dtype=np.float32)
        self.closed = True
        tail = self.tail
        self.tail = np.empty((self.channels, 0), dtype=np.float32)
        return tail
