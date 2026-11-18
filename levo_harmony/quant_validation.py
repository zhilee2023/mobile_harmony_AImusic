"""Quantization diagnostics for the actual cropped music output vocabulary."""
import numpy as np


def valid_music_logits(values, *, eos=0, audio_base=65, audio_classes=16384):
    x = np.asarray(values)
    if x.ndim != 2 or not x.shape[0]:
        raise ValueError('Expected a nonempty token-by-class logit array')
    if audio_base < 1 or audio_classes < 1 or not 0 <= eos < audio_base or audio_base + audio_classes > x.shape[1]:
        raise ValueError('Invalid cropped music vocabulary contract')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite logits cannot pass a quantization assessment')
    valid = np.r_[eos, np.arange(audio_base, audio_base + audio_classes)]
    return x[:, valid].astype(np.float64)


def music_logit_error(reference, candidate, *, eos=0, audio_base=65, audio_classes=16384,
                      candidate_eos=None, candidate_audio_base=None):
    ref = valid_music_logits(reference, eos=eos, audio_base=audio_base, audio_classes=audio_classes)
    got = valid_music_logits(candidate, eos=eos if candidate_eos is None else candidate_eos,
                             audio_base=audio_base if candidate_audio_base is None else candidate_audio_base,
                             audio_classes=audio_classes)
    if ref.shape != got.shape:
        raise ValueError('Expected matched autoregressive token positions')
    def log_probability(x):
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    log_ref, log_got = log_probability(ref), log_probability(got)
    kl = np.sum(np.exp(log_ref) * (log_ref - log_got), axis=1)
    return {'tokenPositions': ref.shape[0], 'validOutputClasses': ref.shape[1],
            'relativeL2': float(np.linalg.norm(got-ref) / max(np.linalg.norm(ref), 1e-30)),
            'argmaxAgreement': float(np.mean(ref.argmax(1) == got.argmax(1))),
            'eosMaxAbs': float(np.max(np.abs(ref[:, 0]-got[:, 0]))),
            'maxAbs': float(np.max(np.abs(ref-got))),
            'meanKlReferenceToCandidate': float(kl.mean()),
            'maxKlReferenceToCandidate': float(kl.max()),
            'scope': 'Same fixed autoregressive history, valid music classes plus EOS; not listening quality or original prequantization checkpoint parity'}
