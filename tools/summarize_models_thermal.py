"""Append a scoped model/thermal summary, keeping historical report sections intact.

Raw device traces and model bundles stay local. --plot additionally needs matplotlib.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.telemetry import summarize


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def trace(name):
    path = ROOT / 'reports' / name
    return path, [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    report_path = ROOT / 'reports/levo-test-report.json'
    archive = read(report_path)
    old_sections = dict(archive)
    model_rows = {}
    counts = {'public': [3106035712, 751534224, 84414082], 'small': [598178304, 195416128, 84414082]}
    for family, bundle_id in [('public', 'public-pt-vae-test-v1'), ('small', 'student-onnx-test-v1')]:
        bundle = read(ROOT / 'models/bundles' / bundle_id / 'model-bundle.json')
        assets = {row['relative']: row for row in bundle['assets']}
        modules = {}
        for name, count in zip(('llm', 'dit', 'vae'), counts[family]):
            module = bundle['modules'][name]
            asset = assets[module['externalWeight']]
            modules[name] = {'sourceParameters': count, 'externalWeightBytes': asset['bytes'],
                             'externalWeightSha256': asset['sha256'], 'recipe': module['recipe'],
                             'runtime': module['runtime'], 'training': module['training']}
        model_rows[family] = {'modelId': bundle_id, 'modules': modules, 'runtime': bundle['runtime'],
                              'listedSourceParameterTotalExcludingSeparateConditionTables': sum(counts[family]),
                              'assetCount': len(assets), 'assetBytes': sum(row['bytes'] for row in assets.values()),
                              'musicTrained': bundle['musicTrained'], 'qualityValidated': bundle['qualityValidated']}
    small_path, small_rows = trace('power-thermal-small90-retest-20261102-r3.jsonl')
    public_path, public_rows = trace('power-thermal-large-90-foreground-20261102-r1.jsonl')
    codec_path, codec_rows = trace('power-thermal-large-codec90-foreground-20261102-r1.jsonl')
    lm_rows = [row for row in public_rows if (lambda stage: 'prefill' in stage or stage.startswith('real-phone-music-codes-'))(json.loads(row['progress']).get('stage', ''))]
    sources = [('small90Retest2', small_path, small_rows), ('public90LmMatchedOnly', public_path, lm_rows),
               ('public90IndependentCodec', codec_path, codec_rows)]
    thermal = {}
    for key, path, rows in sources:
        thermal[key] = {'source': path.name, 'sourceSha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'selection': 'LM stage names only; later Codec stages excluded due to unresolved writer identity' if key == 'public90LmMatchedOnly' else 'All recorded rows; sampling coverage is limited to the observed interval',
                        **summarize(rows)}
    integration = read(ROOT / 'reports/local/sensor-integration/music-idle-report.json')
    assert integration['telemetry']['sampleCount'] == 3
    assert 4 < integration['telemetry']['observedIntervalSeconds']['median'] < 6
    first_codec = next(row for row in public_rows if json.loads(row['progress']).get('stage', '').startswith('chunk-'))
    record = {'date': '2026-11-02', 'scope': 'Model parameters, quantized asset sizes, historical sensor audit, and host-observer periodic sampling',
              'models': model_rows, 'thermal': thermal,
              'public90Audit': {
                  'supersedesExitStageAndThermalAttributionIn': ['serial_chain_comparison_90s_2026_11_02', 'docs/CHAIN_COMPARISON_90S.md'],
                  'parentObserver': 'compare-large90-foreground-observer.log: code 2250 observed, then process-ended exception; no final whole-chain success',
                  'sensorTrace': 'Same-named historical trace contains later chunk 1 through chunk 12 progress events; these are not idle cleanup',
                  'firstCodecEventObserverSeconds': first_codec['observerElapsedSeconds'],
                  'firstCodecProgress': json.loads(first_codec['progress']),
                  'lastCodecProgressInUnattributedTrace': json.loads(public_rows[-1]['progress']),
                  'lastLmSensorEvent': json.loads(lm_rows[-1]['progress']),
                  'excludedLaterRows': len(public_rows)-len(lm_rows),
                  'attribution': 'Unresolved: historical rows have no embedded writer PID/run ID; parent observer and path-read progress cannot prove one uninterrupted process',
                  'exitPhaseConfirmed': False, 'singleRunE2eSuccessArchived': False,
                  'recoveredPhaseSumRtfStillNotE2e': 6.225054406,
                  'earlierPlot': 'levo-90s-power-thermal.* public middle column includes Codec events; do not use its full trace as LM-only or label its tail idle cleanup'},
              'periodicSampling': {'requestedSeconds': 5, 'defaultEnabledIn': ['tools/run_comparison_app.ps1', 'tools/test_music_model.ps1'],
                                   'scope': 'During PC-observed phone tests; not scheduled unattended test reruns or standalone App telemetry',
                                   'sameTestReportField': 'telemetry', 'intervalArgument': 'SensorIntervalSeconds (1..60)',
                                   'disableArgument': '-RecordSensors:$false',
                                   'missingReadings': 'null/NA, never fake zero',
                                   'idleIntegration': integration,
                                   'validation': {'pythonUnitTestsPassed': 37, 'powershellParsePassed': True,
                                                  'phoneIdleSamplerPassed': True, 'newInferenceRunThisTurn': False}},
              'limits': ['Current deployed route is CPU; process RSS is not dedicated GPU VRAM',
                         'Parameter totals exclude separately listed conditioning tables; external weight bytes are not total ROM or RAM',
                         'Public LM is the primary Mixed LM, not the entire official dual-LM pipeline',
                         'Small music adapters/head and DiT remain untrained',
                         'Battery-terminal |V*I| under USB charging is not device/App power; sample means are not whole-run energy averages',
                         'Historical thermal records are not a thermally controlled model-size comparison',
                         'Current public replacement VAE has a different quantized artifact from the old shared W8 baseline']}
    archive['model_thermal_summary_2026_11_02'] = record
    assert all(archive[key] == value for key, value in old_sections.items() if key != 'model_thermal_summary_2026_11_02')
    report_path.write_text(json.dumps(archive, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: {name: value[name] for name in ('sampleCount', 'batteryC', 'socNodeC', 'batteryExchangeMagnitudeW')} for key, value in thermal.items()}, indent=2))
    if args.plot:
        plot(sources, thermal)


def plot(sources, thermal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    titles = ['0.6B原型 · 90秒重测2', '公开模型 · 仅匹配LM阶段', '公开模型 · 独立Codec恢复']
    fields = [('socNodeC', '内部温度节点 °C', '#b33b30'), ('batteryC', '电池温度 °C', '#287050'),
              ('batteryExchangeMagnitudeW', '电池端 |V×I| W', '#6943a0'), ('cpuPolicy3GHz', 'CPU policy3 GHz', '#2869a2')]
    fig, axes = plt.subplots(4, 3, figsize=(13, 9), sharex='col')
    for column, ((key, _, _), title) in enumerate(zip(sources, titles)):
        samples = thermal[key]['samples']
        origin = samples[0]['observerSeconds']
        times = [s['observerSeconds']-origin for s in samples]
        for row, (field, label, color) in enumerate(fields):
            ax = axes[row, column]
            ax.plot(times, [s[field] for s in samples], '.-', color=color, linewidth=1, markersize=3)
            ax.grid(alpha=.2)
            if column == 0: ax.set_ylabel(label, fontproperties=font)
            if row == 0: ax.set_title(title, fontproperties=font, fontsize=11)
            if row == 3: ax.set_xlabel('本列采样起点后的秒数', fontproperties=font)
    fig.suptitle('LeVo 真机采样：各轮独立时间轴，非受控模型对比', fontproperties=font, fontsize=15)
    fig.text(.5, .025, 'USB供电/充电；电池端交换功率不是整机或App功耗。内部温度不是机壳温度。\n公开LM列最后采样到2225码；身份未确认的后续Codec事件已排除。', ha='center', fontproperties=font, fontsize=10)
    fig.tight_layout(rect=(0,.08,1,.95))
    for suffix in ('png', 'svg', 'pdf'):
        path = ROOT / 'reports' / f'levo-model-thermal-summary.{suffix}'
        fig.savefig(path, dpi=160)
        if suffix == 'svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    main()
