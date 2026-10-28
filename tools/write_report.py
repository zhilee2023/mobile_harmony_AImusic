"""Aggregate verified local results without copying weights or private machine data."""
import json
import math
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
header = json.loads((root / 'reference/mixed-codec-header.json').read_text(encoding='utf8'))
cfm = {k: v for k, v in header.items() if k.startswith('cfm_wrapper.')}
layers = {int(m.group(1)) for k in cfm if (m := re.search(r'estimator\.h\.(\d+)\.', k))}
architecture = json.loads((root / 'reports/student-architecture.json').read_text(encoding='utf-8-sig'))
tests = (root / 'reports/reference-tests.log').read_text(encoding='utf-8-sig')
build = (root / 'build.log').read_text(encoding='utf-8-sig')
report = {
    'version': '0.1.0',
    'student': architecture,
    'reference_tests': {'passed': 'Ran 8 tests' in tests and '\nOK' in tests, 'count': 8,
                        'uses_tiny_random_fixtures': True, 'trained_music_quality_tested': False},
    'harmony': {'build_success': 'BUILD SUCCESSFUL' in build, 'signed': False,
                'mnn_adapter_arm64_compile_success': (root / 'reports/mnn_compile_probe.o').is_file(),
                'music_runtime_linked_and_run': False},
    'public_mixed_codec_header': {
        'url': 'https://huggingface.co/lglg666/SongGeneration-Runtime/resolve/main/ckpt/model_1rvq/model_2_fixed.safetensors',
        'read_header_only': True, 'dit_layers': len(layers),
        'dit_width': cfm['cfm_wrapper.estimator.h.0.ln_1.weight']['shape'][0],
        'dit_ffn_width': cfm['cfm_wrapper.estimator.h.0.mlp.c_fc.bias']['shape'][0],
        'cfm_parameters_from_header': sum(math.prod(v['shape']) for v in cfm.values()),
        'full_weights_loaded': False,
        'source_constructor_missing_cfm_wrapper': True,
        'source_revision': '653cbcf4716101834900c75b7d5da43b07e15d5b'},
    'generation': {'real_song_generated': False, 'rtf_measured': False,
                   'blocking_work': ['music adapter training', 'complete official Mixed DiT construction',
                                     'DiT/VAE export and PC alignment', 'Harmony music runtime and signing']},
}
(root / 'reports/levo-test-report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report, indent=2))
