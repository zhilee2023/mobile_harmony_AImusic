"""Index existing validated artifacts; never download or regenerate weights."""
from pathlib import Path
from .contracts import ROOT, load_profile
from .bundles import AUX_ELEMENTS


def baseline_job(profile_id, identity, public_artifacts=None):
    profile = load_profile(profile_id); files = {}
    if profile['family'] == 'small':
        folder = ROOT / 'models/student-diagnostic-mnn'
        for name in ('config.json', 'llm_config.json', 'llm.mnn', 'llm.mnn.weight', 'tokenizer.mtok', 'embeddings_bf16.bin', 'diagnostic-student.json', 'prefix.json'):
            files['student/' + name] = str(folder / name)
        dit = ROOT / 'models/small-music-dit-22-1024'
        training = {'llm': 'untrained', 'dit': 'untrained', 'vae': 'pretrained'}
        title = '0.6B结构研究模型'
    else:
        if not public_artifacts:
            raise ValueError('Provide --public-artifacts with the existing Mixed LM/condition artifact folder')
        artifacts = Path(public_artifacts).resolve(); folder = artifacts / 'mixed-mnn-w4'
        for name in ('llm.mnn', 'llm.mnn.weight', 'llm_config.json', 'tokenizer.mtok'):
            files['lm/' + name] = str(folder / name)
        for name in ('conditions.safetensors', 'music-embeddings-bf16.bin', 'mapping-prefix.bf16'):
            files[name] = str(artifacts / name)
        for step in (0, 1):
            files[f'lm/mapping-{step}-reference.f32'] = str(artifacts / f'mapping-{step}-reference.f32')
        files['input.json'] = str(ROOT / 'configs/deployment/smoke-input.json')
        dit = ROOT / 'models/full-dit-256'; training = {role: 'pretrained' for role in ('llm', 'dit', 'vae')}; title = '标准歌曲模型'
    for name in ('full-w8.mnn', 'full-w8.mnn.weight'):
        files['dit/' + name] = str(dit / name)
    for name in ('vae-256-w8.mnn', 'vae-256-w8.mnn.weight'):
        files['vae/' + name] = str(ROOT / 'models/vae-stream-256' / name)
    for name in AUX_ELEMENTS:
        files[f'aux/{name}.f32'] = str(ROOT / f'models/three-module-bundle/aux/{name}.f32')
    return {'profile': profile_id, 'modelId': identity, 'displayName': title, 'files': files, 'training': training, 'legacyMappingProbe': profile['family'] == 'public'}
