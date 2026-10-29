"""Archive matched awake VAE repeats without upgrading them to whole-codec results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdc', required=True)
    args = parser.parse_args()
    path = ROOT / 'reports/levo-test-report.json'
    report = json.loads(path.read_text(encoding='utf8'))
    binary_sha = sha(ROOT / 'native-build/levo_vae_bench')
    expected_sha = report['lifecycle_experiments_2026_10_29']['phone_binary_sha256']['levo_vae_bench-affinity']
    if binary_sha != expected_sha:
        raise ValueError('Current binary differs from historical comparison binary')
    receipts = {}
    for frames in (64, 256):
        folder = ROOT / f'models/vae-stream-{frames}'
        files = sorted(p for p in folder.iterdir() if p.suffix == '.f32' or p.name in (
            'fixture.json', f'vae-{frames}-fp32.mnn', f'vae-{frames}-fp32.mnn.weight'))
        remote = f'/data/local/tmp/levo-vae-stream-{frames}'
        command = f'cd {remote} && sha256sum ' + ' '.join(p.name for p in files)
        output = subprocess.run([args.hdc, 'shell', command], capture_output=True, text=True, check=True).stdout
        observed = {line.split()[1]: line.split()[0] for line in output.splitlines()
                    if len(line.split()) == 2 and len(line.split()[0]) == 64}
        if set(observed) != {p.name for p in files}:
            raise ValueError('Incomplete fixture checksum receipt')
        for item in files:
            if observed[item.name] != sha(item):
                raise ValueError('Phone fixture mismatch: ' + str(item))
        receipts[str(frames)] = observed
    runs = []
    for frames, repeat in ((64, 1), (256, 1), (256, 2), (64, 2)):
        name = f'phone-vae-awake-{frames}-low-6-high-r{repeat}'
        logfile = ROOT / 'reports' / (name + '.log')
        lines = [s.removeprefix('LEVO_VAE_REPORT ') for s in logfile.read_text(encoding='utf-8-sig').splitlines()
                 if s.startswith('LEVO_VAE_REPORT ')]
        if len(lines) != 1:
            raise ValueError('Missing unique completed run: ' + name)
        row = json.loads(lines[0])
        if (not row['success'] or row['windowFrames'] != frames or row['threadsRequested'] != 6
                or row['precision'] != 'low' or row['powerHint'] != 'High'
                or row['validPcmSeconds'] != 25.6 or row['mainPipelineBackend'] != 13
                or len(row['rows']) != (13 if frames == 64 else 3)):
            raise ValueError('Configuration or completion mismatch: ' + name)
        if not math.isclose(sum(s['runAndReadSeconds'] for s in row['rows']) / 25.6,
                            row['decoderOnlyComputeRtf'], rel_tol=1e-10):
            raise ValueError('RTF does not match measured calls')
        for sample in row['rows']:
            if not all(math.isfinite(sample[key]) for key in
                       ('runAndReadSeconds', 'relativeL2VsPytorch', 'emittedRelativeL2')):
                raise ValueError('Non-finite numerical result')
        states = {}
        for moment in ('before', 'after'):
            text = (ROOT / 'reports' / f'{name}-{moment}.log').read_text(encoding='utf-8-sig')
            match = re.search(r'Current State:\s*(\w+)', text)
            states[moment] = match.group(1) if match else 'unavailable'
        if set(states.values()) != {'AWAKE'}:
            raise ValueError('Awake status not confirmed at both endpoints')
        identity = (ROOT / 'reports' / f'{name}-identity.log').read_text(encoding='utf-8-sig')
        if binary_sha not in identity or receipts[str(frames)]['fixture.json'] not in identity:
            raise ValueError('Run identity receipt mismatch')
        row.update(source_log=name + '.log', source_log_sha256=sha(logfile), powerStateEndpoints=states,
                   repeat=repeat, binary_sha256=binary_sha)
        runs.append(row)
    summary = {}
    for frames in (64, 256):
        group = [r for r in runs if r['windowFrames'] == frames]
        historic = next(r for r in report['lifecycle_experiments_2026_10_29']['vae_phone']
                        if r['source_log'] == f'phone-vae-{frames}-low-6-high-power.log')
        for run in group:
            for observed, previous in zip(run['rows'], historic['rows'], strict=True):
                for key in ('relativeL2VsPytorch', 'emittedRelativeL2'):
                    if not math.isclose(observed[key], previous[key], abs_tol=1e-8, rel_tol=1e-8):
                        raise ValueError('Numerical result changed from historical fixture')
        summary[str(frames)] = {
            'awake_compute_rtf': [r['decoderOnlyComputeRtf'] for r in group],
            'awake_peak_rss_mib': [r['sampledPeakRssKiB'] / 1024 for r in group],
            'awake_first_numerical_packet_seconds': [r['firstNumericalPacketSeconds'] for r in group],
            'historical_sleep_start_compute_rtf': historic['decoderOnlyComputeRtf'],
            'numerical_l2_metrics_match_historical': True,
            'interpretation': 'Two awake runs versus one historical sleep-start run; not a causal screen-state experiment'}
    report['awake_vae_validation_2026_10_29'] = {
        'as_of_utc': datetime.now(timezone.utc).isoformat(), 'run_order': ['64-r1', '256-r1', '256-r2', '64-r2'],
        'runs': runs, 'summary': summary, 'phone_fixture_sha256_verified': receipts,
        'limits': ['VAE-only synthetic latent timeline, no LLM/DiT or speaker playback',
                   'CLI shell identity; not a foreground signed App benchmark',
                   'AWAKE checked at endpoints; no continuous display/thermal-state control',
                   'Module compute RTF excludes setup, reference-file IO and full song pipeline']}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
