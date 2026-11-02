"""Independent pinned HF tokenizer fixtures for the App's native conditioner."""
import json
from pathlib import Path
from transformers import Qwen2Tokenizer

ART=Path('C:/Users/lizhi/Documents/Codex/2026-10-13/z/levo-public-validation/artifacts')
BASE=json.loads((ART/'conditions.json').read_text(encoding='utf-8'))
sections=list(BASE['sectionTokenIds'])
lyric=Qwen2Tokenizer.from_pretrained(ART/'qwen2-tokenizer',local_files_only=True)
lyric.add_tokens(sections,special_tokens=True)
style=Qwen2Tokenizer.from_pretrained(ART/'qwen2-tokenizer',local_files_only=True)
examples=[
    ('[verse]\n清晨的阳光，照在窗上。\n[chorus]\n我想为你唱一首歌！','温暖女声，钢琴、吉他，中文流行，90 BPM'),
    ('[verse]\nHello, sunshine! 2026\n[bridge]\n我们一起向前走。','soft male vocal, jazz / acoustic, 6/8, 95 BPM'),
    ('[intro-short]\n[verse]\n月亮🌙照着远方\n[chorus]\n风 轻轻地吹\n[outro-short]','dream pop, airy female vocals, subtle synths'),
    (BASE['lyrics'],BASE['style']),
]
fixtures=[{'lyrics':a,'style':b,'lyricIds':lyric.encode('<|im_start|>'+a,add_special_tokens=False),'styleIds':style.encode('<|im_start|>'+b,add_special_tokens=False)} for a,b in examples]
(ART/'music-condition-fixtures.json').write_text(json.dumps(fixtures,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared 4 independent tokenizer fixtures; canonical sample is last for full prefix comparison')
