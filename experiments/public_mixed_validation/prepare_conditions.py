"""Prepare real trained LeVo conditions without loading the 1.3GB tables."""
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parent / 'artifacts'
REV = 'cc258cc694a63114c61684cc26d0583b8ad777d0'
ROOT_URL = f'https://huggingface.co/lglg666/SongGeneration-Runtime/resolve/{REV}/third_party/Qwen2-7B/'
SECTIONS = ['[verse]', '[chorus]', '[bridge]', '[intro-short]', '[intro-medium]', '[intro-long]',
            '[outro-short]', '[outro-medium]', '[outro-long]', '[inst-short]', '[inst-medium]', '[inst-long]', '[silence]']
LYRICS = ('[intro-short]\n[verse]\n晚风轻轻吹过窗台\n星光落在你的肩膀\n'
          '沿着灯火走向远方\n把心愿唱给月亮\n[chorus]\n让我们迎着风歌唱\n'
          '让明天开出新的光\n就算路途还有漫长\n你在身旁就是方向\n[outro-short]')
STYLE = 'female, warm clear vocal, Mandarin pop, acoustic guitar, piano, soft drums, hopeful, melodic, 90 BPM'


def main():
    directory = BASE / 'qwen2-tokenizer'
    directory.mkdir(exist_ok=True)
    sources = {}
    for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'tokenizer.json', 'LICENSE'):
        path = directory / name
        if not path.exists():
            with urllib.request.urlopen(ROOT_URL + name, timeout=60) as stream:
                data = stream.read(12 * 1024**2)
            if len(data) >= 12 * 1024**2:
                raise ValueError('Unexpected tokenizer size')
            path.write_bytes(data)
        sources[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print('Tokenizer ready: ' + name, flush=True)
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import Qwen2Tokenizer
    torch.set_num_threads(4)
    style_tokenizer = Qwen2Tokenizer.from_pretrained(directory, local_files_only=True)
    lyric_tokenizer = Qwen2Tokenizer.from_pretrained(directory, local_files_only=True)
    lyric_tokenizer.add_tokens(SECTIONS, special_tokens=True)
    if style_tokenizer.convert_tokens_to_ids('<|im_start|>') != 151644:
        raise ValueError('Wrong tokenizer family')
    ids = lyric_tokenizer.encode('<|im_start|>' + LYRICS, add_special_tokens=False)
    style_ids = style_tokenizer.encode('<|im_start|>' + STYLE, add_special_tokens=False)
    section_ids = {lyric_tokenizer.convert_tokens_to_ids(s) for s in SECTIONS}

    with safe_open(BASE / 'conditions.safetensors', framework='pt', device='cpu') as state:
        def rows(name, indices):
            table = state.get_slice('audiolm.condition_provider.conditioners.' + name)
            size = table.get_shape()[0]
            if any(i < 0 or i >= size for i in indices):
                raise ValueError('Token outside trained table: ' + name)
            return torch.cat([table[i:i+1].float() for i in indices], dim=0)

        def text(ids_in, length, lyric=False):
            cover = [0] * len(ids_in)
            if lyric:
                current = 0
                for i, token in enumerate(ids_in):
                    if token in section_ids:
                        current = token - 151645
                    cover[i] = current
            tokens = (ids_in + [151643] * length)[:length]
            covered = (cover + [0] * length)[:length]
            embedding = rows(('description' if lyric else 'type_info') + '.output_proj.weight', tokens)
            if lyric:
                embedding += rows('description.structure_emb.weight', covered)
            return embedding

        eot = state.get_tensor('audiolm.condition_provider.conditioners.prompt_audio.EOT_emb').float()
        # Official conditioner prepends EOT and pads no-reference codes with 16385.
        audio = torch.cat((eot, rows('prompt_audio.emb.0.weight', [16385] * 251)), dim=0)
        music = state.get_slice('audiolm.emb.0.weight')
        bos = music[16385:16386].float()
        cond = torch.cat((text(ids, 600, True), audio, text(style_ids, 100), bos), dim=0)
        null = torch.cat((text([151644], 600, True), audio, text([151644], 100), bos), dim=0)
        if list(cond.shape) != [953, 2048] or not torch.isfinite(cond).all():
            raise ValueError('Invalid condition prefix')
        cond.numpy().astype('<f4').tofile(BASE / 'prefix-conditioned.f32')
        null.numpy().astype('<f4').tofile(BASE / 'prefix-unconditional.f32')
        # Actual 16386-row input embedding: includes BOS not predicted by LM head.
        music[:].contiguous().view(torch.uint16).numpy().astype('<u2').tofile(BASE / 'music-embeddings-bf16.bin')
    report = {'lyrics': LYRICS, 'style': STYLE, 'lyricTokenCount': len(ids), 'styleTokenCount': len(style_ids),
              'prefixShape': list(cond.shape), 'sectionTokenIds': {s: lyric_tokenizer.convert_tokens_to_ids(s) for s in SECTIONS},
              'promptReferenceAudio': False, 'tokenizerRevision': REV, 'tokenizerFiles': sources,
              'allConditionEmbeddingsTrained': True, 'inputEmbeddingRows': 16386, 'outputVocabulary': 16385,
              'scope': 'Mixed-only, source conditioner order description/prompt_audio/type_info/BOS'}
    (BASE / 'conditions.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
