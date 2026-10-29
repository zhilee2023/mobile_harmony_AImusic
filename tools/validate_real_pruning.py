"""Prune actual public Qwen weights, verify cache, optionally save a local backbone.

This is an uncalibrated text-backbone pruning experiment, not a music checkpoint.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.model import MixedSongStudent
from levo_harmony.pruning import prune_backbone_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-backbone', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(2307)
    torch.set_num_threads(4)
    source = ROOT / 'models/qwen3-0.6b'
    dest = ROOT / 'models/qwen3-0.6b-26layers-unrecovered'
    if args.save_backbone and dest.exists() and any(dest.iterdir()):
        raise ValueError('Do not overwrite an existing pruned checkpoint')
    manifest = json.loads((source / 'download-manifest.json').read_text())
    digest = hashlib.sha256()
    with (source / 'model.safetensors').open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != manifest['sha256']:
        raise ValueError('Public source weight hash mismatch')
    backbone, info = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa', output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Public backbone did not load exactly')
    student = MixedSongStudent(backbone).eval()
    removed = (9,18)
    kept = [index for index in range(28) if index not in removed]
    addresses = [tuple((p.data_ptr(), p.numel(), p.dtype) for p in backbone.model.layers[index].parameters())
                 for index in kept]
    result = prune_backbone_depth(student, removed)
    for index, layer in enumerate(backbone.model.layers):
        if tuple((p.data_ptr(),p.numel(),p.dtype) for p in layer.parameters()) != addresses[index]:
            raise ValueError('Kept pretrained weights were unexpectedly replaced')
        if layer.self_attn.layer_idx != index:
            raise ValueError('Pruned cache indices must be contiguous')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    lyric = torch.tensor([tokenizer.encode('晚风轻轻吹过窗台，星光落在你的眼海。'*100, add_special_tokens=False)[:512]])
    with torch.inference_mode():
        prefix, mask = student.prefix(lyric, torch.tensor([0]))
        bos = torch.tensor([[student.bos]])
        initial = torch.cat((prefix,student.music(bos)),1)
        mask = torch.cat((mask,torch.ones_like(bos)),1)
        output = backbone.model(inputs_embeds=initial, attention_mask=mask, use_cache=True)
        cache = output.past_key_values
        codes = torch.tensor([[11,25,100,356,1024,2048,8192,16383]])
        for index in range(8):
            mask = torch.cat((mask,torch.ones_like(bos)),1)
            output = backbone.model(inputs_embeds=student.music(codes[:,index:index+1]),
                attention_mask=mask, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
        cached = output.last_hidden_state[:, -1].clone()
        full = backbone.model(inputs_embeds=torch.cat((initial,student.music(codes)),1),
            attention_mask=mask,use_cache=False).last_hidden_state[:, -1]
        if not torch.isfinite(full).all():
            raise ValueError('Pruned backbone contains non-finite outputs')
        torch.testing.assert_close(cached,full,rtol=2e-4,atol=2e-4)
        l2 = float(torch.linalg.vector_norm(cached-full)/torch.linalg.vector_norm(full))
        last_embedding = initial[:, -1:].clone()
    result.update({'real_public_weights_loaded': True, 'source_revision': manifest['revision'],
        'source_sha256_verified': True, 'kept_original_layer_indices': kept,
        'kept_parameter_object_identity_checked': True, 'cache_vs_full_relative_l2': l2,
        'prefill_positions': 514, 'cache_positions_checked': int(cache.get_seq_length()),
        'backbone_parameters': sum(p.numel() for p in backbone.parameters()),
        'actual_student_parameters': sum(p.numel() for p in student.parameters()),
        'music_adapter_trained': student.music_trained, 'real_song_generated': False,
        'saved_backbone': False, 'music_quality_validated': False,
        'scope': 'Actual pretrained26-layer text backbone with random music/style adapters; recovery training still required'})
    if args.save_backbone:
        backbone.to(torch.bfloat16)
        # Compare the saved dtype against itself, not against an FP32 execution.
        with torch.inference_mode():
            bf16_reference = backbone.model(inputs_embeds=last_embedding.to(torch.bfloat16),
                                             use_cache=False).last_hidden_state.float().clone()
        backbone.save_pretrained(dest, safe_serialization=True, max_shard_size='2GB')
        tokenizer.save_pretrained(dest)
        del student,backbone,output,cache
        gc.collect()
        reloaded, load_info = AutoModelForCausalLM.from_pretrained(dest, local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        if any(load_info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Saved pruned checkpoint did not reload exactly')
        with torch.inference_mode():
            restored = reloaded.model(inputs_embeds=last_embedding.to(torch.bfloat16),
                                      use_cache=False).last_hidden_state.float()
        torch.testing.assert_close(restored,bf16_reference,rtol=0,atol=0)
        weights = sorted(dest.glob('*.safetensors'))
        hashes = {}
        for path in weights:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            hashes[path.name] = digest.hexdigest()
        result.update({'saved_backbone': True,'saved_backbone_dtype':'bfloat16',
            'saved_backbone_relative_path': str(dest.relative_to(ROOT)),
            'saved_reload_exact': True,'saved_weight_bytes':sum(p.stat().st_size for p in weights),
            'saved_weight_sha256': hashes})
        (dest/'pruning-manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (ROOT/'reports/real-pruned-backbone.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
