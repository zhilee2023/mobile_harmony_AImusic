"""Merge bounded CLI experiments into the single public test report.

No model weights, signing settings, device serials, power dumps or cloud keys.
Preserve failures and distinguish numerical flow from trained music quality.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stable(record):
    if 'displayStable' in record:
        return record['displayStable']
    if 'displayStableAtEndpoints' in record:
        return record['displayStableAtEndpoints']
    before, after = record['screenBefore'], record['screenAfter']
    a, b = re.search(r'Last Screen Off:\s*(\d+)', before), re.search(r'Last Screen Off:\s*(\d+)', after)
    return bool(a and b and a[1] == b[1] and 'Current State: AWAKE' in before and 'Current State: AWAKE' in after)


def receipt(path):
    value = load(path)
    result = value['result']
    row = {k: v for k, v in value.items() if k not in ('screenBefore', 'screenAfter', 'result')}
    row.update(sourceReportSha256=sha(path), displayStable=stable(value),
               timingEligible=bool(result.get('success') and stable(value)), result=result)
    return row


def main():
    path = REPORTS / 'levo-test-report.json'
    report = load(path)
    chain_names = ('small-eager10-r1', 'small-overlap10-r1', 'small-overlap10-r2', 'small-compute10-r1')
    chains = [receipt(REPORTS / f'scheduling-{name}.json') for name in chain_names]
    assert not chains[1]['displayStable'], 'Preserve invalid display-transition timing'
    for row in (chains[0], chains[2], chains[3]):
        r = row['result']
        assert row['timingEligible'] and r['requestedChunks'] == 10 and r['generatedCodecIds'] == 1984
        assert abs(r['validPcmSeconds'] - 79.36) < 1e-5 and r['finalKvPositions'] == 2497
        assert r['music_trained'] is False and r['real_song_generated'] is False
    assert len({r['result']['generatedCodeHashFnv64'] for r in chains}) == 1
    assert len({r['result']['pcmRms'] for r in chains}) == 1
    small = load(ROOT / 'models/small-music-dit-22-1024/metadata.json')
    # Initial exporter included reference file hashing in this PC timing field.
    small['pcValidationAndArtifactAccountingSeconds'] = small.pop('pcMnnSeconds')
    small['pcTimingIsIsolatedForward'] = False
    kv = [receipt(p) for p in sorted(REPORTS.glob('scheduling-kv-*.json'))]
    report['scheduling_and_lowbit_2026_10_31'] = {
        'execution': 'Standalone CLI; installed signed v0.1.7 App remains unchanged',
        'smallDit': small, 'chains': chains, 'llmOnlyTeacherForcedKvExperiments': kv,
        'interpretation': [
            'Small DiT uses random untrained weights at CosyVoice-sized dimensions, retaining LeVo scaffold. Not an exact CosyVoice checkpoint.',
            'Ten chunks use one persistent LLM, one DiT and one VAE session; 256-frame windows, 64-frame overlap, 1984 new IDs, 79.36 valid PCM seconds.',
            'Compute overlap bounds pending token batches to one. Per-model CPU contention increases individual times; compare wall time, not summed stage times.',
            'Loading-overlap r1 had a screen-off transition and is excluded from timing decisions. KV persistent r1 did not have that transition.',
            'Small DiT phone/PC L2 is 1.1303%, above the 1% strict threshold; finite flow and memory success do not mean strict parity or music quality.',
            'Teacher-forced KV tests use the untrained music head and fixed artificial IDs; their accuracy metrics are not language-model quality.',
            'KV4 is TQ4; A4 was not applied in these earlier music-head KV experiments.',
        ]}
    report['quarot_projection_2026_10_31'] = load(REPORTS / 'quarot-projection-cpu-20261031.json')
    language = []
    for p in sorted(REPORTS.glob('qwen-language-*.json')):
        raw = load(p)
        if 'result' in raw:
            language.append(receipt(p))
    assert len(language) >= 12, 'Wait for all planned language probes and exact-weight ablation'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-0.6b', local_files_only=True)
    for row in language:
        r = row['result']
        if r.get('success'):
            assert r['modelManifest']['layers'] == 28 and r['modelManifest']['vocabulary'] == 151936
            assert len(r['rounds']) == 2 and all(x['decodeTokens'] == 127 for x in r['rounds'])
            r['greedyOutputText'] = tokenizer.decode(r.get('greedyOutputIds', []))
        r.setdefault('precision', 'low')
    audit_path = REPORTS / 'qwen-language-activation-audit-r3.log'
    audit_line = next(x for x in audit_path.read_text(encoding='utf-8-sig').splitlines() if x.startswith('LEVO_A4_AUDIT '))
    audit = json.loads(audit_line[len('LEVO_A4_AUDIT '):])
    assert audit['runtimeFourBitRangeObserved'] and audit['packedActivation4KernelClaimed'] is False
    fixture = load(ROOT / 'models/qwen-language-fixture/fixture.json')
    report['qwen_language_lowbit_2026_10_31'] = {
        'model': 'Original pretrained Qwen3-0.6B, 28 layers, full trained 151936-token language head; no music adapters',
        'sourceRevision': 'c1899de289a04d12100db370d81485cdf75e47ca',
        'mnnSourceCommit': 'd407447ed56c4121a11ccbd266dc184ca1ead0c2',
        'fixture': fixture, 'fixtureJsonSha256': sha(ROOT / 'models/qwen-language-fixture/fixture.json'),
        'runs': language, 'activationExecutionAudit': audit,
        'staticMetadataCpuAblation': load(REPORTS / 'qwen-static-metadata-cpu-sequence.json'),
        'format': {
            'weightBits': 4, 'weightGroupSize': 64, 'languageHeadWeightBits': 4, 'embedding': 'BF16',
            'kv4': 'MNN attention_mode=14: FlashAttention plus TQ4 K/V',
            'a4Attempt': 'Official SmoothQuant implementation called with experimental act_bit=4, 20 local calibration texts; 308 static descriptors with min=-7/max=7/type=DT_INT8',
            'packedA4NativeKernel': False, 'headActivationStaticQuantization': False,
            'runtimeAudit': 'Callback sees four-bit range metadata and FloatToInt8/Convolution DT_INT8 wrappers; logical tensor type remains float32. This does not prove packed INT4 arithmetic.',
        },
        'parallelism': {
            'prefill': 'One multi-token forward, GEMM tiling and token/output parallelism; causal attention can process query blocks in parallel.',
            'decode': 'One new token per dependent step, GEMV or M=1 GEMM plus head/context parallelism inside a step; future autoregressive steps cannot run independently.',
            'mnnGraphSelection': 'Llm::forwardRaw selects prefill module key 100 and decode sequence-length key 1; both share RuntimeManager and configured thread_num.',
            'independentThreadConfigPubliclyExposedInPinnedLlm': False,
            'setConfigAfterLoadWarning': 'Changing thread_num JSON after load does not reconstruct the loaded runtime; not tested as a valid stage thread switch.',
            'availableHints': ['thread_num', 'dynamic_option', 'attention_mode', 'cpu_sme2_neon_division_ratio'],
            'sme2': 'No SME2 feature reported by this phone; do not attribute results to SME2-specific mixed-core hints.',
        },
        'conclusions': [
            'Standard W4/g64 route works; 6 threads observed about 25-28 decode tokens/s, 512-token prefill about 1.40-1.58s across the paired baseline runs.',
            'KV4 reduces retained KV storage but was slower than KV16 at this context and degraded reference-continuation agreement; peak RSS includes loading and was not lower in every short test.',
            'A4-range plus KV4 is not a deployable result: Low produced finite but unusable language logits, High produced non-finite logits. A8 static control also failed quality.',
            'Exact smoothed W4 weights with activation metadata removed recover language agreement, localizing a failure to the current static activation graph/execution combination rather than proving A4 inherently impossible.',
            'One original synthetic 512-token context and 128 fixed continuation predictions, ignoring EOS for fixed-length timing. Agreement/NLL are regression indicators, not general language quality or perplexity benchmarks.',
            'No App replacement, no packed W4A4KV4 speed claim, no QuaRot whole-model checkpoint deployment. QuaRot-style projection tests are separate.',
        ],
        'primarySources': [
            'https://github.com/alibaba/MNN/blob/master/docs/transformers/llm.md',
            'https://github.com/alibaba/MNN/blob/master/transformers/llm/export/llmexport.py',
            'https://github.com/alibaba/MNN/blob/master/source/backend/cpu/compute/ConvInt8TiledExecutor.cpp',
            'https://github.com/spcl/QuaRot'],
    }
    npu = []
    for name in ('residual-d1-h4', 'pureconv-k7-d1-h4'):
        p = REPORTS / f'mate90-npu-{name}-20261031.json'
        npu.append(receipt(p))
    report['new_soc_omc_compatibility_2026_10_31'] = {
        'tests': npu,
        'result': 'All four old Kirin9030-target OMC variants were rejected at Build code1 on reported Kirin9050; no NPU inference latency measured.',
        'toolchain': {'omgVersion': 'DDK-tools-next-6.1.1.0', 'targetPlugin': 'kirin9030',
                      'omgSha256': '46c23dadcea1d2c42690cdd34356c9d5eb5bf8d933247de793071dbd15672cf8',
                      'pluginArchiveSha256': '5df8110e9b494ba87216b0c195c83d7f6c1407af89613de850ed7c569e57e593'},
        'matching9050ToolchainVerified': False, 'wholeVaeOnNewNpuTested': False,
        'ascendC': 'Official tools extracted and kirin9030 operator skeleton generated. No custom VAE kernel compiled or validated on Kirin9050 yet.',
        'officialDownloadPageObservation': 'Public preparations page version hmos-503 lists 9020/X90/9030 plugins. This is not proof that no Kirin9050 package exists elsewhere.',
        'next': 'Select exact OMG binary and matching SoC plugin, retain versions/hashes and partition evidence; do not rename a 9030 plugin to 9050 or equate Build failure with NPU slowness.'}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'mergedChainRuns': len(chains), 'mergedMusicKvRuns': len(kv), 'mergedLanguageRuns': len(language), 'reportBytes': path.stat().st_size}))


if __name__ == '__main__':
    main()
