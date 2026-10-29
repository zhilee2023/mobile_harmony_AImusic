"""Validate actual local Qwen weights and cache; never generate untrained music."""
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.model import MixedSongStudent


def main():
    torch.manual_seed(2307)
    torch.set_num_threads(4)
    folder = ROOT / 'models/qwen3-0.6b'
    manifest = json.loads((folder / 'download-manifest.json').read_text())
    digest = hashlib.sha256()
    with (folder / 'model.safetensors').open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if not manifest['sha256_verified'] or digest.hexdigest() != manifest['sha256']:
        raise ValueError('Actual Qwen weight SHA256 mismatch')
    begin = time.perf_counter()
    backbone, loading = AutoModelForCausalLM.from_pretrained(
        folder, local_files_only=True, dtype=torch.float32,
        attn_implementation='sdpa', output_loading_info=True)
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if loading.get(key):
            raise ValueError(f'Backbone load is not exact: {key}: {loading[key]}')
    model = MixedSongStudent(backbone).eval()
    load_seconds = time.perf_counter() - begin
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    ids = tokenizer.encode('晚风轻轻吹过窗台，星光落在你的眼海。' * 100,
                           add_special_tokens=False)[:512]
    if len(ids) != 512:
        raise ValueError('Expected full 512-token benchmark prefix')
    lyric = torch.tensor([ids])
    style = torch.tensor([0])
    with torch.inference_mode():
        prefix, attention = model.prefix(lyric, style)
        bos = torch.tensor([[model.bos]])
        initial = torch.cat((prefix, model.music(bos).to(prefix.dtype)), 1)
        attention = torch.cat((attention, torch.ones_like(bos)), 1)
        begin = time.perf_counter()
        out = model.backbone.model(inputs_embeds=initial, attention_mask=attention, use_cache=True)
        prefill_seconds = time.perf_counter() - begin
        cache = out.past_key_values
        times = []
        codes = torch.tensor([[11, 25, 100, 356, 1024, 2048, 8192, 16383]])
        for index in range(codes.shape[1]):
            embedding = model.music(codes[:, index:index+1]).to(prefix.dtype)
            attention = torch.cat((attention, torch.ones_like(bos)), 1)
            begin = time.perf_counter()
            out = model.backbone.model(inputs_embeds=embedding, attention_mask=attention,
                                       past_key_values=cache, use_cache=True)
            times.append(time.perf_counter() - begin)
            cache = out.past_key_values
            if not torch.isfinite(out.last_hidden_state).all():
                raise ValueError('Non-finite cached backbone hidden state')
        cached_last = out.last_hidden_state[:, -1].clone()
        full = torch.cat((initial, model.music(codes).to(prefix.dtype)), 1)
        reference = model.backbone.model(inputs_embeds=full, attention_mask=attention,
                                         use_cache=False).last_hidden_state[:, -1]
        torch.testing.assert_close(cached_last, reference, atol=2e-4, rtol=2e-4)
        cache_l2 = float(torch.linalg.vector_norm(cached_last-reference) /
                         torch.linalg.vector_norm(reference))
        if int(cache.get_seq_length()) != 522:
            raise ValueError('Cache sequence length mismatch')
    try:
        model.generate_music(lyric, style, max_new_tokens=1)
    except RuntimeError as error:
        untrained_guard = 'untrained' in str(error)
    else:
        untrained_guard = False
    if not untrained_guard:
        raise ValueError('Untrained music generation guard failed')
    report = {
        'real_pretrained_weights_loaded': True, 'weight_sha256_verified': True,
        'revision': manifest['revision'], 'backbone_parameters': sum(p.numel() for p in backbone.parameters()),
        'student_parameters': sum(p.numel() for p in model.parameters()),
        'lyrics_tokens': 512, 'style_prefix_tokens': 1, 'audio_bos_tokens': 1,
        'prefill_positions': 514, 'cache_positions_after_fixed_8_codes': 522,
        'load_seconds': load_seconds, 'prefill_seconds': prefill_seconds,
        'teacher_forced_incremental_seconds': times, 'cache_vs_full_relative_l2': cache_l2,
        'untrained_music_guard_passed': untrained_guard, 'real_song_generated': False,
        'scope': 'Windows CPU FP32 4 threads; real text backbone, random untrained music/style adapters; 8 fixed input codes are not generated music',
    }
    (ROOT / 'reports/pretrained-backbone-cpu.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
