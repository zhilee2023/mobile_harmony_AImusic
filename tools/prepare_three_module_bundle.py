"""Build a local-only, hash-manifested three-module memory diagnostic pack."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from levo_harmony.mixed_sampler import MixedCodeConditioner

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
    import MNN
    out = ROOT / 'models/three-module-bundle'
    out.mkdir(exist_ok=True)
    aux = out / 'aux'
    aux.mkdir(exist_ok=True)
    state = load_file(ROOT / 'models/levo-mixed-codec/mixed-aux.safetensors')
    conditioner = MixedCodeConditioner(state)
    count = state['normfeat.counts']
    mean = state['normfeat.sum_x'] / count
    std = (state['normfeat.sum_x2'] / count - mean.square()).clamp(min=0).sqrt()
    for name, value in {'codebook': conditioner.codebook, 'weight': conditioner.weight,
                        'bias': conditioner.bias, 'mask': state['mask_emb.weight'], 'mean': mean, 'std': std}.items():
        np.ascontiguousarray(value.detach().numpy(), dtype='<f4').tofile(aux / (name + '.f32'))
    source = ROOT / 'models/full-dit-256'
    fixture = np.load(source / 'fixture.npz')
    net = MNN.Interpreter(str(source / 'full-w8.mnn'))
    session = net.createSession({'backend': 'CPU', 'numThread': 4, 'precision': 1, 'memory': 2})
    for name in ('inputs', 'timestep', 'attention_mask'):
        value = np.ascontiguousarray(fixture[name])
        tensor = MNN.Tensor(tuple(value.shape), MNN.Halide_Type_Float, value, MNN.Tensor_DimensionType_Caffe)
        if not net.getSessionInput(session, name).copyFrom(tensor):
            raise ValueError('MNN fixture copy failed')
    net.runSession(session)
    output = np.asarray(net.getSessionOutput(session, 'velocity').getNumpyData()).copy()
    if not np.isfinite(output).all() or output.shape != (2, 256, 64):
        raise ValueError('Invalid PC full-DiT reference')
    output.astype('<f4').tofile(source / 'w8-reference.f32')
    # Correct the first export's manifest-count lookup. The model was strictly
    # loaded; the download manifest has no `tensors` dictionary. Verify the
    # count against safetensors header before recording this metadata repair.
    result_path = ROOT / 'reports/full-dit-mnn-256.json'
    report = json.loads(result_path.read_text())
    with safe_open(ROOT / 'models/levo-mixed-codec/mixed-inference.safetensors', framework='pt') as weights:
        strict_count = len([key for key in weights.keys() if key.startswith('cfm_wrapper.')])
    if strict_count != 526:
        raise ValueError('Unexpected complete CFM tensor count')
    if report['strict_cfm_tensors'] == 0:
        report['initial_metadata_count_lookup_corrected'] = True
        report['strict_cfm_tensors'] = strict_count
        result_path.write_text(json.dumps(report, indent=2), encoding='utf8')
    groups = {
        'student': (ROOT / 'models/student-diagnostic-mnn', ['config.json', 'llm_config.json', 'llm.mnn',
                    'llm.mnn.weight', 'tokenizer.mtok', 'embeddings_bf16.bin', 'diagnostic-student.json', 'prefix.json']),
        'dit': (source, ['full-w8.mnn', 'full-w8.mnn.weight', 'inputs.f32', 'timestep.f32',
                         'attention_mask.f32', 'reference.f32', 'w8-reference.f32']),
        'vae': (ROOT / 'models/vae-stream-256', ['vae-256-fp32.mnn', 'vae-256-fp32.mnn.weight'])}
    for folder, (original, names) in groups.items():
        (out / folder).mkdir(exist_ok=True)
        for name in names:
            target = out / folder / name
            if not target.exists():
                os.link(original / name, target)
            if sha(target) != sha(original / name):
                raise ValueError('Existing bundle file differs; preserve it: ' + name)
    files = {str(p.relative_to(out)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in out.rglob('*') if p.is_file() and p.name != 'bundle.json'}
    manifest = {'music_trained': False, 'real_song_generated': False, 'dit_steps_for_diagnostic': 2,
                'frames': 256, 'duration_seconds': 10.24, 'files': files,
                'scope': 'Real26-layer Qwen + random music adapters -> complete39-layer W8 DiT -> real VAE; memory/flow only'}
    (out / 'bundle.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps({'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()), 'strict_cfm_tensors': strict_count}))


if __name__ == '__main__':
    main()
