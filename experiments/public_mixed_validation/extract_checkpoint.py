"""Stream the verified official ZIP into an HF Mixed LM and condition tables.

Never execute model.pt pickle. Tensor metadata was independently restricted-
unpickled by the existing audit; ZIP entries, offsets, strides and sizes are
checked again here. No track-specific LM or unused text-model embedding.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import zipfile

EXPECTED_SHA = 'dc763aa9a76a22a87597c2faf9a51c24d13349ac754699b37e9068b483639def'
EXPECTED_BYTES = 12899965446


def records(checkpoint, metadata):
    with zipfile.ZipFile(checkpoint) as archive:
        entries = {e.filename.rsplit('/', 1)[-1]: e for e in archive.infolist() if '/data/' in e.filename}
    result = {}
    with checkpoint.open('rb') as source:
        for name, value in metadata.items():
            if value['dtype'] != 'BF16':
                continue
            item = entries[value['storageKey']]
            if item.compress_type != zipfile.ZIP_STORED or item.file_size != value['storageElements'] * 2:
                raise ValueError('Unexpected ZIP storage: ' + name)
            stride = 1
            for width, actual_stride in reversed(list(zip(value['shape'], value['stride']))):
                if width != 1 and actual_stride != stride:
                    raise ValueError('Non-contiguous view: ' + name)
                stride *= width
            count = math.prod(value['shape'])
            if value['storageOffset'] < 0 or value['storageOffset'] + count > value['storageElements']:
                raise ValueError('View outside storage: ' + name)
            source.seek(item.header_offset)
            header = source.read(30)
            if header[:4] != b'PK\x03\x04':
                raise ValueError('Bad ZIP local header')
            namesize, extrasize = struct.unpack_from('<HH', header, 26)
            if source.read(namesize).decode() != item.filename:
                raise ValueError('Local ZIP filename mismatch')
            offset = item.header_offset + 30 + namesize + extrasize + value['storageOffset'] * 2
            result[name] = (offset, count * 2, value['shape'])
    return result


def write_safetensors(checkpoint, destination, tensors):
    header = {'__metadata__': {'format': 'pt', 'source_sha256': EXPECTED_SHA, 'scope': 'trained Mixed only'}}
    cursor = 0
    for name, (_, size, shape) in tensors.items():
        header[name] = {'dtype': 'BF16', 'shape': shape, 'data_offsets': [cursor, cursor + size]}
        cursor += size
    payload = json.dumps(header, separators=(',', ':')).encode()
    payload += b' ' * (-len(payload) % 8)
    partial = destination.with_suffix('.partial')
    if destination.exists():
        raise FileExistsError(destination)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as source, partial.open('wb') as output:
        def write(data):
            output.write(data)
            digest.update(data)
        write(struct.pack('<Q', len(payload)))
        write(payload)
        for index, (name, (offset, size, _)) in enumerate(tensors.items()):
            source.seek(offset)
            remaining = size
            while remaining:
                data = source.read(min(remaining, 4 * 1024**2))
                if not data:
                    raise ValueError('Truncated tensor payload')
                write(data)
                remaining -= len(data)
            if index % 50 == 0:
                print(f'{destination.name}: {index + 1}/{len(tensors)} {name}', flush=True)
    os.replace(partial, destination)
    return {'bytes': destination.stat().st_size, 'sha256': digest.hexdigest(), 'tensors': len(tensors)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('C:/levo-harmony'))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'artifacts')
    args = parser.parse_args()
    if args.checkpoint.stat().st_size != EXPECTED_BYTES:
        raise ValueError('Source size mismatch')
    with args.checkpoint.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError('Source SHA256 mismatch')
    print('Official checkpoint SHA256 verified', flush=True)
    metadata = json.loads((args.project / 'models/public-levo-v2-large/tensor-metadata.json').read_text())['tensors']
    data = records(args.checkpoint, metadata)
    body, conditions = {}, {}
    for name, record in data.items():
        if name.startswith('audiolm.transformer.model.layers.') or name in (
                'audiolm.transformer.model.norm.weight', 'audiolm.transformer.lm_head.weight'):
            body[name.removeprefix('audiolm.transformer.')] = record
        elif name.startswith('audiolm.condition_provider.conditioners.description.') or name.startswith(
                'audiolm.condition_provider.conditioners.type_info.') or name in (
                'audiolm.condition_provider.conditioners.prompt_audio.EOT_emb',
                'audiolm.condition_provider.conditioners.prompt_audio.emb.0.weight', 'audiolm.emb.0.weight'):
            conditions[name] = record
    music = data['audiolm.emb.0.weight']
    if music[2] != [16386, 2048] or len(body) != 326:
        raise ValueError(f'Unexpected Mixed architecture {music[2]} / {len(body)}')
    # HF needs matching vocab dimensions. Preserve the final BOS row separately
    # in conditions.safetensors; the exporter restores all 16386 embedding rows.
    body['model.embed_tokens.weight'] = (music[0], 16385 * 2048 * 2, [16385, 2048])
    args.out.mkdir(parents=True, exist_ok=True)
    hf = args.out / 'hf-mixed'
    hf.mkdir(exist_ok=True)
    report = {'sourceSha256': digest, 'sourceBytes': EXPECTED_BYTES, 'trackLmIncluded': False,
              'trainedMixedLm': True, 'musicGenerated': False}
    report['hfWeights'] = write_safetensors(args.checkpoint, hf / 'model.safetensors', body)
    report['conditions'] = write_safetensors(args.checkpoint, args.out / 'conditions.safetensors', conditions)
    config = {'architectures': ['LlamaForCausalLM'], 'model_type': 'llama', 'hidden_size': 2048,
              'intermediate_size': 11008, 'num_hidden_layers': 36, 'num_attention_heads': 16,
              'num_key_value_heads': 16, 'head_dim': 128, 'hidden_act': 'silu', 'rms_norm_eps': 1e-5,
              'rope_theta': 500000.0, 'max_position_embeddings': 10000, 'vocab_size': 16385,
              'tie_word_embeddings': False, 'attention_bias': False, 'mlp_bias': False,
              'torch_dtype': 'bfloat16', 'dtype': 'bfloat16', 'bos_token_id': None,
              'eos_token_id': 16384, 'pad_token_id': None, 'use_cache': True}
    (hf / 'config.json').write_text(json.dumps(config, indent=2))
    (args.out / 'checkpoint-verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
