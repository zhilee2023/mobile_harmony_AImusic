"""Prepare the thin App's inventory and exact baseline conditioning fixture."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = Path('C:/Users/lizhi/Documents/Codex/2026-10-13/z/levo-public-validation/artifacts')
FILES = {
    'llm.mnn': ('lm/llm.mnn', ART/'mixed-mnn-w4/llm.mnn'),
    'llm.mnn.weight': ('lm/llm.mnn.weight', ART/'mixed-mnn-w4/llm.mnn.weight'),
    'llm_config.json': ('lm/llm_config.json', ART/'mixed-mnn-w4/llm_config.json'),
    'tokenizer.mtok': ('lm/tokenizer.mtok', ART/'mixed-mnn-w4/tokenizer.mtok'),
    'conditions.safetensors': ('conditions.safetensors', ART/'conditions.safetensors'),
    'music-embeddings-bf16.bin': ('music-embeddings-bf16.bin', ART/'music-embeddings-bf16.bin'),
    'mapping-prefix.bf16': ('mapping-prefix.bf16', ART/'mapping-prefix.bf16'),
}
for name in ('mapping-0-reference.f32','mapping-1-reference.f32'):
    FILES[name] = ('lm/'+name, ART/name)
for name in ('full-w8.mnn','full-w8.mnn.weight'):
    FILES[name] = ('dit/'+name, ROOT/'models/full-dit-256'/name)
for name in ('vae-256-w8.mnn','vae-256-w8.mnn.weight'):
    FILES[name] = ('vae/'+name, ROOT/'models/vae-stream-256'/name)
for name in ('codebook.f32','weight.f32','bias.f32','mask.f32','mean.f32','std.f32'):
    FILES[name] = ('aux/'+name, ROOT/'models/three-module-bundle/aux'/name)


def main():
    with (ART/'generation-embeddings-bf16.bin').open('rb') as source:
        source.seek((16386+1906)*4096)
        fixture=source.read(32*4096)
    assert len(fixture)==32*4096
    (ART/'mapping-prefix.bf16').write_bytes(fixture)
    rows=[]
    for name,(relative,path) in FILES.items():
        assert path.is_file()
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b''):digest.update(chunk)
        rows.append({'name':name,'relative':relative,'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'source':str(path)})
    safe=[{k:v for k,v in row.items() if k!='source'} for row in rows]
    inventory=json.dumps(safe,ensure_ascii=False)
    header='#pragma once\nstatic const Json& musicAssetInventory(){static const Json value=Json::parse(R"ASSETS('+inventory+')ASSETS");return value;}\n'
    (ROOT/'music-app/entry/src/main/cpp/MusicAssets.hpp').write_text(header,encoding='utf-8')
    (ART/'music-app-assets-local.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (ROOT/'music-app/model-assets.json').write_text(json.dumps(safe,indent=2),encoding='utf-8')
    print(f'{len(rows)} assets; {sum(r["bytes"] for r in rows)/1024**3:.3f} GiB; no weights included in App')


if __name__=='__main__':main()
