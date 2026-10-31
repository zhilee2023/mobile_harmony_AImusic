"""Publish public architecture evidence without claiming weight load or music."""
import json
from pathlib import Path

import yaml

from levo_harmony.pipeline_spec import PipelineSpec

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = ROOT / 'models/public-levo-v2-large'
    audit = json.loads((ROOT / 'reports/public-levo-v2-audit.json').read_text(encoding='utf8'))
    metadata = json.loads((model / 'tensor-metadata.json').read_text(encoding='utf8'))
    config = yaml.safe_load((model / 'config.yaml').read_text(encoding='utf8'))
    full = PipelineSpec.from_config('SongGeneration-v2-large', config, 'hierarchical', 'levo-public-separate-2stream')
    mixed = PipelineSpec.from_config('SongGeneration-v2-large-mixed-only-ablation', config, 'mixed-only', 'levo-public-mixed-1rvq')
    shapes = {k: v['shape'] for k, v in metadata['tensors'].items()}
    full.validate_main_weight_shapes(shapes)
    storage = {}
    for item in metadata['tensors'].values():
        identity = item['storageKey']
        storage.setdefault(identity, (item['storageElements'], item['dtype']))
        if storage[identity] != (item['storageElements'], item['dtype']):
            raise ValueError('Inconsistent shared storage descriptor')
    dtype_bytes = {'F32': 4, 'BF16': 2, 'F16': 2, 'I64': 8, 'I32': 4, 'BOOL': 1}
    audit.update(uniqueStorageCount=len(storage), uniqueStorageElements=sum(x[0] for x in storage.values()),
                 storagePayloadBytes=sum(count * dtype_bytes[dtype] for count, dtype in storage.values()),
                 fullSourceSha256Verified=False, architectureShapesValidated=True)
    budgets = []
    for duration in (30, 180, 270):
        # A stated scenario, not a hard-coded official prefix limit.
        positions = 512 + duration * 25 + 250
        budgets.append({'audioSeconds': duration, 'assumedPrefixPositions': 512, 'delayAllowancePositions': 250,
                        'positionsPerLm': positions, 'cfgBranches': 1,
                        'kv16IdealBytes': full.ideal_kv_bytes(positions, 16),
                        'kv4IdealBytes': full.ideal_kv_bytes(positions, 4),
                        'includesQuantizationScalesAllocatorOrActivation': False})
    section = {'audit': audit, 'fullProfile': full.__dict__, 'mixedOnlyProfile': mixed.__dict__,
               'output': 'Both routes ultimately produce one stereo mixed song; hierarchical route internally predicts mixed/vocal/accompaniment streams.',
               'publicRendererArchitectures': {'mixed': {'layers': 39, 'hidden': 1200, 'heads': 30, 'ffn': 4800},
                                                'separateConditioned': {'layers': 16, 'hidden': 2200, 'heads': 20, 'ffn': 4400}},
               'vae': {'paperApproxParameters': 150000000, 'latentChannels': 64, 'frameRate': 25,
                       'sampleRate': 48000, 'outputChannels': 2},
               'idealAllStoredParametersW4BytesWithoutScales': (audit['uniqueStorageElements'] * 4 + 7) // 8,
               'kvBudgets': budgets,
               'w4a4kv4Target': {'requested': True, 'nativePackedA4Validated': False,
                                'public4bCheckpointQuantizationValidated': False, 'generationReady': False},
               'generationStatus': {'trainedPublicWeightPayloadLoaded': False, 'fullCheckpointDownloaded': False,
                                    'musicGeneratedOnPc': False, 'musicGeneratedOnPhone': False},
               'boundaries': [
                   'This is a 90KB metadata and configuration audit, not a 12.9GB model download or inference test.',
                   'Approximate 4B in the paper refers to LeLM core; checkpoint also stores large conditioning embeddings and other tensors. Do not estimate payload only from the label.',
                   'A4 reduces activation precision/storage, not weight parameter count. Packed W4, embedding storage, KV, CFG branches, runtime packing and activation peaks must be budgeted separately.',
                   'TQ4 executed in prior Qwen tests; full public LeVo KV4 has not been validated. Current static A4/A8 path failed Qwen language regressions.',
                   'Size-independent contracts validate condition widths, pretrained tensor shapes, stream identities and codec IDs. They are not a completed MNN runtime adapter or automatic weight conversion.',
                   'Mixed-only drops the trained track refinement LM and changes the decoder checkpoint; it is a separate quality ablation, not the full 4B chain.',
               ],
               'sources': ['https://arxiv.org/html/2606.30642v1',
                           'https://huggingface.co/lglg666/SongGeneration-v2-large/blob/115805364ad74479fb3764fe65970c92faeb1a5a/config.yaml',
                           'https://github.com/levo-demo/LeVo/blob/653cbcf4716101834900c75b7d5da43b07e15d5b/codeclm/models/lm_levo.py',
                           'https://github.com/levo-demo/LeVo/blob/653cbcf4716101834900c75b7d5da43b07e15d5b/codeclm/models/codeclm.py']}
    path = ROOT / 'reports/levo-test-report.json'
    report = json.loads(path.read_text(encoding='utf8'))
    report['public_levo_architecture_2026_10_31'] = section
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'metadataBytesRead': audit['httpBytesRead'], 'validatedMainLayers': full.main_layers,
                      'weightPayloadLoaded': False, 'generationReady': False}))


if __name__ == '__main__':
    main()
