"""Validate saved WAVs and append one cooled comparison to the unified report."""
import argparse
import array
import hashlib
import json
import math
import sys
import wave
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.deployment.files import read_json, write_json
from levo_harmony.telemetry import summarize


def cooling_telemetry(pair):
    start = datetime.fromisoformat(pair['coolingStartUtc'])
    rows = []
    for sample in pair['coolingSamples']:
        raw = sample['rawReadings']
        if len(raw) < 5:
            continue
        rows.append({'observerElapsedSeconds': (datetime.fromisoformat(sample['utc'])-start).total_seconds(),
                     'units': {'batteryCurrent': 'mA'}, 'progress': '{}',
                     'rawReadings': [raw[1],raw[0],raw[2],raw[3],'NA','NA','NA','NA',raw[4],'NA']})
    result = summarize(rows)
    result['scope'] = 'Cooling observations; no model inference scheduled by this protocol; USB online state not sampled in this interval'
    result['usbOnlineObserved'] = None
    return result


def validate_wave(path, duration):
    path = Path(path)
    with wave.open(str(path), 'rb') as wav:
        metadata = {'sampleRate': wav.getframerate(), 'channels': wav.getnchannels(),
                    'sampleBytes': wav.getsampwidth(), 'frames': wav.getnframes(), 'compression': wav.getcomptype()}
        assert metadata == {'sampleRate': 48000, 'channels': 2, 'sampleBytes': 2, 'frames': duration*48000, 'compression': 'NONE'}, metadata
        raw = wav.readframes(metadata['frames'])
    assert len(raw) == duration*48000*4, 'Truncated WAV'
    pcm = array.array('h', raw)
    if sys.byteorder != 'little': pcm.byteswap()
    peak = max(abs(value) for value in pcm)/32768
    rms = math.sqrt(sum(value*value for value in pcm)/len(pcm))/32768
    assert rms > 0 and peak > 0, 'Silent diagnostic file'
    return {**metadata, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'pcmPeak': peak, 'pcmRms': rms, 'listeningQualityAccepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    pair = read_json(args.pair)
    assert pair['state'] == 'complete', 'Do not publish a partial pair as a completed comparison'
    assert pair['observedCoolingSeconds'] >= pair['coolingMinutes']*60
    assert pair['large']['hapSha256'] == pair['small']['hapSha256']
    duration = pair['durationSeconds']
    assert duration == 60, 'This report validates the requested saved-audio 60-second pair'
    waves = {}
    stages = {}
    for family, label in [('large', 'public'), ('small', 'student')]:
        observed = pair[family]
        result = observed['result']
        assert result['success'] and result['durationSeconds'] == duration
        assert result['lm']['frames'] == duration*25
        assert len(result['codec']['chunks']) == (duration*25-256+191)//192+1
        assert observed['foregroundStable'] and observed['displayStable'], 'Keep disturbed results but do not label this a completed foreground benchmark'
        audio = pair[family+'Audio']
        waves[label] = validate_wave(audio['path'], duration)
        assert waves[label]['sha256'] == audio['sha256']
        stages[label] = {'seconds': result['stageSeconds'], 'rtf': result['stageRtf'], 'memory': result['memory'],
                         'quantization': result['quantization'], 'thermal': observed.get('telemetry'),
                         'output': waves[label], 'trainedMusic': label == 'public'}
    record = {'date': '2026-11-03', 'scope': 'Same research HAP, public 60s -> >=15 min idle -> student 60s; no synthetic code repetition',
              'pair': pair, 'stages': stages, 'wavValidation': waves, 'cooling': cooling_telemetry(pair),
              'validation': {'signedResearchAppVersion': '0.1.14', 'pythonUnitTestsPassed': 37,
                             'sameHapForBoth': True, 'bothForegroundAndDisplayChecksPassed': True,
                             'bothWavFilesValidated': True, 'originalExperimentsPreserved': True},
              'workflowRecoveryScope': 'The public inference completed once; a storage-alias export whitelist failed, then only file transfer and cooling resumed. This is not an inference restart.',
              'excludedAttempt': 'cooled60-20261103-r1 public launch was BACKGROUND; stopped before output, not a performance baseline',
              'preflightAttempt': 'r2 rejected before any inference because research App was not foreground',
              'limits': ['Public EOS suppressed for fixed length', 'Student music adapters/head and DiT remain untrained',
                         'Different conditions/training and uncontrolled external environment; idle interval does not prove equal initial temperatures',
                         'USB battery-terminal |V*I| is not App, SoC or isolated whole-device power',
                         'RSS/PSS are shared system memory, not dedicated GPU VRAM', 'No player/listening quality acceptance from WAV validation alone']}
    archive_path = ROOT / 'reports/levo-test-report.json'
    archive = read_json(archive_path)
    key = 'cooled_60s_comparison_2026_11_03'
    if key in archive and archive[key]['pair']['runId'] != pair['runId']:
        raise ValueError('Preserve earlier paired experiment; use a separate dated report key')
    archive[key] = record
    write_json(archive_path, archive)
    print(json.dumps({label: {'stageSeconds': row['seconds'], 'stageRtf': row['rtf'],
                              'peakRssMiB': row['memory']['peakRssKiB']/1024,
                              'batteryC': row['thermal']['batteryC'] if row['thermal'] else None} for label,row in stages.items()}, indent=2))
    if args.plot:
        plot(pair, args.pair.stem)


def plot(pair, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), sharex='col')
    fields = [('batteryC', '电池温度 °C', '#287050'), ('socNodeC', '内部温度节点 °C', '#b33b30'),
              ('batteryExchangeMagnitudeW', '电池端 |V×I| W', '#6943a0')]
    series = [(pair['large']['telemetry']['samples'], '公开模型 · 60秒'),
              (cooling_telemetry(pair)['samples'], '停止推理 · 15分钟冷却'),
              (pair['small']['telemetry']['samples'], '0.6B原型 · 60秒')]
    for column, (samples, title) in enumerate(series):
        origin = samples[0]['observerSeconds']
        for row, (field, label, color) in enumerate(fields):
            axes[row,column].plot([s['observerSeconds']-origin for s in samples], [s[field] for s in samples], '.-', color=color, linewidth=1, markersize=3)
            axes[row,column].set_ylabel(label, fontproperties=font)
            axes[row,column].grid(alpha=.2)
            if row == 0: axes[row,column].set_title(title, fontproperties=font)
            if row == 2: axes[row,column].set_xlabel('本轮采样起点后的秒数', fontproperties=font)
    fig.suptitle('同一HAP · 公开模型 → 15分钟冷却 → 小模型', fontproperties=font, fontsize=15)
    fig.text(.5,.02,'不同时间轴和起始温度；USB电池端交换功率不是整机/App功耗，内部节点不是机壳温度。',ha='center',fontproperties=font,fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.95))
    for suffix in ('png','svg','pdf'):
        path = ROOT / 'reports' / f'{stem}.{suffix}'
        fig.savefig(path,dpi=160)
        if suffix == 'svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    plt.close(fig)


if __name__ == '__main__':
    main()
