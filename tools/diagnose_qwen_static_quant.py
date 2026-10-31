"""Isolate static activation metadata from identical SmoothQuant W4 weights.

CPU numerical ablation only; leaves all original checkpoint files untouched.
"""
import json
from pathlib import Path
import sys

import numpy as np
import _tools
import MNN.llm as mllm

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'models/qwen-language-smooth-a8'
    output = ROOT / 'reports/qwen-static-metadata-cpu-sequence.json'
    if output.exists():
        raise ValueError('Preserve previous numerical ablation')
    graph = json.loads((folder / 'llm.mnn.json').read_text(encoding='utf8'))
    original_count = len(graph['extraTensorDescribe'])
    graph['extraTensorDescribe'] = []
    graph_path = folder / 'llm-noact.json'
    graph_path.write_text(json.dumps(graph), encoding='utf8')
    sys.argv = ['MNNConvert', '-f', 'JSON', '--modelFile', str(graph_path), '--MNNModel', str(folder / 'llm-noact.mnn')]
    _tools.mnnconvert(sys.argv)
    fixture = json.loads((ROOT / 'models/qwen-language-fixture/fixture.json').read_text(encoding='utf8'))
    config = json.loads((folder / 'config.json').read_text(encoding='utf8'))
    config.update(base_dir=str(folder) + '/', llm_model='llm-noact.mnn', llm_config='llm_config.json',
                  embedding_file='embeddings_bf16.bin', thread_num=4, precision='high', attention_mode=8)
    config_path = folder / 'config-noact.json'
    config_path.write_text(json.dumps(config), encoding='utf8')
    model = mllm.create(str(config_path))
    model.load()
    model.generate_init()
    value = model.forward(fixture['prefix'])
    logits = np.asarray(value.read(), dtype=np.float32).reshape(-1)[-151936:]
    reference = np.fromfile(ROOT / 'models/qwen-language-fixture/logits-0.f32', dtype='<f4')
    result = {'scope': 'PC numeric ablation only, not ARM speed; same smoothed W4 weights with activation metadata removed',
              'removedTensorDescriptions': original_count, 'finite': bool(np.isfinite(logits).all()),
              'relativeL2VsFp32': float(np.linalg.norm(logits - reference) / np.linalg.norm(reference)),
              'top1': int(logits.argmax()), 'referenceTop1': int(reference.argmax()),
              'packedActivation4KernelClaimed': False}
    agreed = 0
    nll = 0.
    finite = True
    for i, token in enumerate(fixture['teacherTokens']):
        if i:
            value = model.forward([fixture['teacherTokens'][i - 1]])
            logits = np.asarray(value.read(), dtype=np.float32).reshape(-1)[-151936:]
        finite = finite and bool(np.isfinite(logits).all())
        agreed += int(logits.argmax() == token)
        maximum = float(logits.max())
        nll += float(np.log(np.exp(logits.astype(np.float64) - maximum).sum()) + maximum - logits[token])
    result.update(teacherGreedyAgreement=agreed / len(fixture['teacherTokens']),
                  referenceContinuationMeanNll=nll / len(fixture['teacherTokens']), finite=finite)
    output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
