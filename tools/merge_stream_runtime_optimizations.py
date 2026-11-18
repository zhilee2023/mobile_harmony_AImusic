"""Merge sanitized, actually observed stream-App experiments into one report.

Run from the repository root. Local weights, PCM, logs and phone paths stay local.
Historical report sections are preserved, including unrelated working changes.
"""
from pathlib import Path
import json
import statistics
import sys
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.quant_validation import music_logit_error


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def perf(row):
    return {'loadSeconds': row['loadSeconds'],
            'rounds': [{k: x[k] for k in ('prefillPositions', 'prefillSeconds', 'decodeForwards',
                                        'decodeSeconds', 'decodeForwardsPerSecond', 'kvPositions')}
                       for x in row['rounds']], 'rssPeakMiB': row['rssPeakMiB']}


def fusion_pair(baseline, candidate, baseline_pcm, candidate_pcm):
    b, c = read(baseline), read(candidate)
    assert b['success'] and c['success']
    x = np.fromfile(ROOT / baseline_pcm, np.float32)
    y = np.fromfile(ROOT / candidate_pcm, np.float32)
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    before = statistics.median(r['totalSeconds'] for r in b['runs'][1:])
    after = statistics.median(r['totalSeconds'] for r in c['runs'][1:])
    return {'baselineWarmMedianSeconds': before, 'fusedWarmMedianSeconds': after,
            'latencyReduction': 1-after/before, 'pcmExact': bool(np.array_equal(x, y)),
            'pcmRelativeL2': float(np.linalg.norm(y-x)/max(np.linalg.norm(x), 1e-30)),
            'baselineRssPeakMiB': b['rssPeakMiB'], 'fusedRssPeakMiB': c['rssPeakMiB'],
            'baselineSessionMiB': b['setup']['mnnSessionMemoryMiB'],
            'fusedSessionMiB': c['setup']['mnnSessionMemoryMiB'],
            'fusedCallsPerWindow': [r['customCalls'] for r in c['runs']]}


def duration_case(folder):
    r = read(folder / 'run.json')
    g = r.get('generation', {})
    result = {k: r[k] for k in ('runTag', 'status', 'requestedDurationSeconds', 'dynamicOption',
                               'coolingSecondsRequested', 'coolingSecondsObserved', 'effectiveAudioSeconds',
                               'wholeRtf', 'hapSha256', 'completionSource', 'lastScreenOffUnchanged',
                               'screenAwakeAtEnd', 'foregroundGapObserved', 'controlledTimingAccepted',
                               'observedWholeRtf') if k in r}
    if r.get('foregroundGapObserved') or r.get('lastScreenOffUnchanged') is False:
        result['controlledTimingAccepted'] = False
        if 'wholeRtf' in result: result['observedWholeRtf'] = result.pop('wholeRtf')
        if result.get('status') == 'completed': result['status'] = 'completed-with-foreground-gap'
    result['vaeFusion'] = bool(g.get('vaeFusion', r.get('vaeFusion', False)))
    if g:
        result['generation'] = {k: g[k] for k in ('conditionFrames', 'generatedSamples', 'elapsedSeconds',
                'lmSeconds', 'ditSeconds', 'vaeSeconds', 'ditSetupSeconds', 'vaeSetupSeconds',
                'firstChunkSeconds', 'rssPeakMiB', 'minimumAvailableMiB', 'kvPositions',
                'suppressEos', 'endedWithEos', 'ditSteps', 'cfgScale', 'ditBatch', 'threads',
                'musicVocabulary', 'vaeFusedCalls', 'resident', 'tokenPrefetch') if k in g}
        result['chunks'] = len(g.get('chunks', []))
        if 'vaeFusionRewrite' in g:
            result['vaeRewrite'] = {k: g['vaeFusionRewrite'][k] for k in
                    ('operatorsBefore', 'operatorsAfter', 'pointwiseNodesRemoved', 'sourceWeightsUnmodified')}
        events = []
        ep = ROOT / folder / 'events.jsonl'
        if ep.exists():
            for line in ep.read_text(encoding='utf-8').splitlines():
                try:
                    item = json.loads(line)
                    if item.get('runId') == g['runId']: events.append(item)
                except ValueError: pass
        result['playback'] = {'maxPlayedMs': max((x.get('playedMs', 0) for x in events), default=0),
                              'errorCount': sum('error' in x.get('kind', '') for x in events),
                              'scope': 'Renderer timestamp/callback observation, not acoustic listening approval'}
    temperatures = [s['batteryC'] for s in r.get('samples', []) if s.get('batteryC') is not None]
    if temperatures:
        result['batteryTemperatureC'] = {'firstObserved': temperatures[0], 'lastObserved': temperatures[-1],
                                         'peakObserved': max(temperatures)}
    return result


def main():
    report_path = ROOT / 'reports/levo-test-report.json'
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    model = ROOT / 'models/music-a8-20261118-r1'
    floating = np.fromfile(ROOT/'models/music-a8-float-reference-20261118-r1/logits.f32', np.float32).reshape(128,16449)
    base = np.fromfile(model/'default/logits.f32', np.float32).reshape(128,16449)
    block = np.fromfile(model/'block/logits.f32', np.float32).reshape(128,16449)
    report['stream_lab_music_a8_2026_11_18'] = {
        'scope': 'Actual28-layer LarkStudio music Qwen; one real513-position prefix and128 teacher-history positions',
        'recipe': {'body': 'W4/group64', 'head': 'W8', 'inputEmbedding': 'BF16 complete input vocabulary',
                   'activation': 'Dynamic INT8 temporaries inside linear kernels; nonlinear/residual graph tensors remain floating',
                   'kvChanged': False, 'attentionMode': 8, 'cpuThreads': 4},
        'baselineAlreadyUsedA8': True,
        'baselineAudit': read('models/music-a8-20261118-r1/default-audit/result.json')['activationQuantKernelAudit'],
        'candidateAudit': read('models/music-a8-20261118-r1/block-audit/result.json')['activationQuantKernelAudit'],
        'referenceMeaning': 'The same already-W4/W8 parameters expanded to float; not the original prequantized checkpoint',
        'baselineError': music_logit_error(floating, base), 'perBlockA8Error': music_logit_error(floating, block),
        'defaultA8': perf(read('models/music-a8-20261118-r1/default/result.json')),
        'perBlockA8': perf(read('models/music-a8-20261118-r1/block/result.json')),
        'reverseOrderDefault': perf(read('reports/local/music-a8-default-repeat-result.json')),
        'reverseOrderPerBlock': perf(read('reports/local/music-a8-block-repeat-result.json')),
        'conclusion': 'Per-block A8 chosen for lower additional numerical error; no stable speed gain established; not AWQ or offline calibrated static A8'}
    report['stream_lab_music_a8_2026_11_18']['attentionModeMeaningFromPinnedSource'] = 'FlashAttention requested; quantMode=8%8=0, no KV quantization requested'
    parity = read('reports/local/music-head-audio-parity-20261118.json')
    report['stream_lab_audio_vocabulary_2026_11_18'] = {
        'originalOutputClasses': 168214, 'previousCropClasses': 16449, 'audioOnlyClasses': 16385,
        'sourceRows': [[151765,1],[151830,16384]], 'eosOutput': 0, 'audioOutputBase': 1,
        'inputAudioBase': 151830, 'inputEmbeddingPreserved': True,
        'originalWeightBytes': 412069026, 'previousCropWeightBytes': 251805186, 'audioOnlyWeightBytes': 251737602,
        'graphSha256': '7a7c1541d4a2503e9c4ff432140a13a03951b208b6a32467b2320b14b7da3e3a',
        'weightSha256': '7368ba40ba062dff320b83394d900b3b8787030a544ad9e0891210619f117417',
        'retainedLogitParity': {k:v for k,v in parity.items() if k != 'newProbe'},
        'sharedSamplerContract': read('reports/local/music-eos-sampling-contract-20261118.json'),
        'appCropParity': read('reports/local/music-audio-head-app-parity-20261118.json'),
        'eosPolicy': 'Normal generation samples EOS and keeps valid output; fixed-duration benchmarks explicitly suppress EOS',
        'scope': 'Exact W8 row/alpha copy. Removing the remaining64 unused rows is only a small extra saving; large crop was already deployed'}
    report['stream_lab_vae_graph_fusion_2026_11_18'] = {
        'kernel': '36 Snake chains: alpha multiply, NEON sine, square, inverse-beta multiply, residual add in one FP16 kernel',
        'operatorsBefore': 380, 'operatorsAfter': 236, 'pointwiseNodesRemoved': 144,
        'weightsChanged': False, 'precision': 'CPU Low FP16,6 threads, original W8/group64 convolutions',
        'scope': 'Two real latent fixtures, first/warm forwards separated, reversed order confirmation. Module RSS excludes other models.',
        'firstFixture': fusion_pair('models/vae-fusion-20261118-r1/baseline-r3/result.json',
                  'models/vae-fusion-20261118-r1/fused-r3/result.json',
                  'models/vae-fusion-20261118-r1/baseline-r3/pcm.f32','models/vae-fusion-20261118-r1/fused-r3/pcm.f32'),
        'tailFixtureReverseOrder': fusion_pair('models/vae-fusion-confirm-20261118-r1/baseline.json',
                  'models/vae-fusion-confirm-20261118-r1/fused.json',
                  'models/vae-fusion-confirm-20261118-r1/baseline-pcm.f32','models/vae-fusion-confirm-20261118-r1/fused-pcm.f32'),
        'trafficEstimate': 'Five pointwise stages11N FP16 logical reads/writes to fused2N,81.8% less for these stages; DRAM traffic not measured',
        'deployment': 'Independent stream-Appv0.1.9 explicit vaeFusion option; old graph remains available; end-to-end results below'}
    ab_path = ROOT/'reports/local/music-audio30-fusion-ab-parity-20261118.json'
    if ab_path.exists():
        report['stream_lab_vae_graph_fusion_2026_11_18']['app30PairedValidation'] = json.loads(ab_path.read_text(encoding='utf-8'))
    qpath = ROOT/'reports/local/vae-activation-result-20261118.json'
    if qpath.exists():
        q = json.loads(qpath.read_text(encoding='utf-8-sig'))
        report['stream_lab_vae_activation_rotation_2026_11_18'] = {k:q[k] for k in
            ('success','scope','maximumForwards','l2Budget','maxAbsBudget','deploymentEnabled',
             'hadamardInversePassed','baseline','trials','heldOutTail','candidatePassedBothFixtures',
             'conclusion','rssPeakMiB','minimumAvailableMiB') if k in q}
        report['stream_lab_vae_activation_rotation_2026_11_18']['observedForwards'] = 1+len(q.get('trials',[]))+(2 if 'heldOutTail' in q else 0)
    folders = sorted((ROOT/'reports/local').glob('music-audio*-20261118-r*/run.json'))
    folders += [ROOT/'reports/local/music-a8-app30-20261118-r1/run.json']
    report['stream_lab_duration_matrix_2026_11_18'] = {
        'requestedDurationsSeconds': [30,60,90,120,300],
        'recipe': '0.6B music W4 body/W8 head/per-block dynamic A8; DiT CPU High6,2 Euler steps,CFG1.5,batch2; CPU Low6 VAE, fusion explicit per case',
        'memoryScope': '20ms sampled process RSS, not dedicated VRAM; guard6GiB RSS and650MiB available; one LM/KV+one DiT+one VAE resident per request',
        'timingScope': 'One full App request, actual nonduplicated48kHz stereo output, first load included;3min cooling excluded; no stage-sum reconstruction',
        'tests': [duration_case(p.parent.relative_to(ROOT)) for p in folders],
        'validation': {'pythonUnitTests': 59, 'pythonUnitTestsPassed': True, 'appBuild': 'v0.1.9 signed BUILD SUCCESSFUL',
                       'mnnArchiveSha256': 'fbaa87b8cae9cd451560803ee3837886fe7e44d48976fc12654654412226159c'},
        'thermalLimitations': 'Battery temperature sampled; coarse thermal_zone0 node is not a validated skin/junction reading; no device power saving claim'}
    block = read('models/dit-block-spatial-20261118-r2/validation.json')
    conversion = read('models/dit-block-compiled-20261118-r2/conversion-result.json')
    npu = read('reports/local/dit-block-npu-20261118-r1.json')
    report['dit_complete_block_npu_2026_11_18'] = {
        'status': 'Paused at user request; not deployed', 'mathValidation': block,
        'revisedScalarOmConversion': conversion,
        'initialPhoneCases': npu['cases'], 'bridgeSelfTests': npu['bridgeSelfTests'],
        'hardwareKernelTraceCollected': False,
        'conclusion': 'Initial complete block portable OM failed phone build with code1; revised scalar OM compiled but was not run. Prior first-FFN control passed. Not proof that every DiT/NPU configuration is unsupported.'}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Updated measured experiment sections; historical sections preserved')


if __name__ == '__main__':
    main()
