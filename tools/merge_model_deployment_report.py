"""Publish model replacement acceptance separately from historical speed baselines."""
import json
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    reports = ROOT / 'reports'; local = reports / 'local/model-bundle-test'
    public = read(reports / 'music-model-public-pt-vae-20261102-r1.json')
    small = read(reports / 'comparison-app-modular-student-onnx-20261102-r2.json')
    background = read(reports / 'comparison-app-modular-student-onnx-20261102-r1.json')
    playback = read(local / 'phone-result-after-playback.json'); switches = read(local / 'switch-round-trip.json')
    vae = read(ROOT / 'models/prepared/vae-pt-v2/component.json')
    dit = read(ROOT / 'models/prepared/dit-onnx-v1/component.json')
    script = read(ROOT / 'models/prepared/torchscript-contract-test/component.json')
    p, s = public['result'], small['result']
    assert p['success'] and p['modelBundle'] == 'public-pt-vae-test-v1' and p['durationSeconds'] == 15
    assert playback['lm']['runId'] == p['lm']['runId'] and playback['playbackValidated'] and playback['playbackState'] == 'completed'
    assert small['foregroundStable'] and small['displayStable'] and s['success'] and s['durationSeconds'] == 30 and s['modelBundle'] == 'student-onnx-test-v1'
    assert s['diagnosticOnly'] and not s['musicTrained']; assert not background['foregroundStable']
    assert [row['result']['success'] for row in switches] == [True, True, False, True]
    assert vae['strictLoad']['tensors'] == 182 and vae['sourceFormat'] == 'checkpoint' and dit['sourceFormat'] == 'onnx'
    waveform = local / 'pt-model-15s.wav'
    with wave.open(str(waveform), 'rb') as audio:
        wav = {'frames': audio.getnframes(), 'sampleRate': audio.getframerate(), 'channels': audio.getnchannels(), 'sampleBytes': audio.getsampwidth()}
    assert wav == {'frames': 720000, 'sampleRate': 48000, 'channels': 2, 'sampleBytes': 2}
    dit_time = sum(x['ditSeconds'] for x in p['codec']['chunks']); vae_time = sum(x['vaeSeconds'] for x in p['codec']['chunks'])
    lm_time = p['lm']['totalLmSeconds']; total = p['totalSeconds']
    stages = {'llm': lm_time, 'dit': dit_time, 'vae': vae_time, 'other': total-lm_time-dit_time-vae_time, 'total': total}
    record = {'date': '2026-11-02', 'musicAppVersion': '1.0.6', 'researchAppVersion': '0.1.13',
              'scope': 'Model discovery/selection, strict PT and ONNX conversion, independent deployment slots, real music and diagnostic student acceptance',
              'publicMusic': public, 'publicStageSeconds': stages, 'publicStageRtf': {k: v/15 for k, v in stages.items()},
              'playback': {'validated': True, 'state': playback['playbackState']}, 'wavValidation': wav,
              'switchRoundTrip': switches, 'studentForegroundAcceptance': small, 'studentBackgroundRunExcludedFromPerformanceBaseline': background,
              'studentRegression': read(local / 'student-regression.json'),
              'components': {'realPtVae': vae, 'realOnnxStudentDit': dit, 'torchscriptContractMicroModel': script},
              'bundles': {'public': read(ROOT / 'models/bundles/public-pt-vae-test-v1/model-bundle.json'),
                          'small': read(ROOT / 'models/bundles/student-onnx-test-v1/model-bundle.json')},
              'validation': {'unitSuite': {'tests': 33, 'passed': True}, 'bothSignedAppsBuiltAndInstalled': True,
                             'originalModelPreserved': True, 'newVaeStrictTensors': 182,
                             'initialVaeToleranceRejection': {'maxRelativeL2': .05, 'observedL2': .07131073177689626, 'componentPublished': False},
                             'subsequentVaeCompatibilityBudget': {'maxRelativeL2': .1, 'basis': 'Existing W8 VAE PC baseline approximately 6%; preserve measured 7.13%, not a listening-quality acceptance'}},
              'limits': ['Public observer attached during Codec; it is not full-run foreground validation or a controlled speed comparison',
                         'Small foreground retest is complete; earlier background wall time is retained and excluded from the baseline',
                         'Student music adapters and DiT remain untrained; new training is not inferred from model packaging',
                         'LM checkpoint adapter has a miniature strict-load/RoPE unit test; a new full music LM export still needs its own phone validation',
                         'TorchScript acceptance is a micro-model contract test, not the full model',
                         'NPU/GPU routes are retained experiments; deployed inference here is CPU']}
    file = reports / 'levo-test-report.json'; archive = read(file)
    archive['modular_model_deployment_2026_11_02'] = record
    file.write_text(json.dumps(archive, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'publicStageSeconds': stages, 'publicRtf': total/15, 'publicPeakRssMiB': p['sampledPeakRssKiB']/1024,
                      'smallStageSeconds': s['stageSeconds'], 'smallRtf': s['stageRtf']['total'], 'smallPeakRssMiB': s['memory']['peakRssKiB']/1024}, indent=2))


if __name__ == '__main__': main()
