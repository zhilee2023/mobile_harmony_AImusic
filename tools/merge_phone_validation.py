"""Verify HDC file hashes and merge completed phone results into one report.

Does not run inference or downloads. Requires the connected authorized phone.
Raw machine-local logs and weights are excluded from Git.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_row(path):
    lines = [s for s in path.read_text(encoding='utf-8-sig').splitlines()
             if s.startswith('LEVO_REPORT ')]
    if len(lines) != 1:
        raise ValueError(f'Expected exactly one report in {path.name}')
    row = json.loads(lines[0].removeprefix('LEVO_REPORT '))
    if not row.get('success') or len(row.get('runsSeconds', [])) != 11:
        raise ValueError(f'Incomplete phone result: {path.name}')
    if not all(math.isfinite(v) and v > 0 for v in row['runsSeconds']):
        raise ValueError(f'Invalid phone timing: {path.name}')
    if not math.isfinite(row['relativeL2VsPytorch']):
        raise ValueError(f'Invalid numerical result: {path.name}')
    row.setdefault('frames', 32)
    row['source_log'] = path.name
    return row


def verify_files(hdc, frames):
    suffix = '' if frames == 32 else f'-{frames}'
    fixture = ROOT / f'models/phone-block{suffix}'
    model = ROOT / f'models/dit-block{suffix}'
    expected = {p.name: p for p in fixture.iterdir() if p.is_file()}
    for bits in (32, 8, 4):
        for extension in ('.mnn', '.mnn.weight'):
            path = model / f'real-block0-w{bits}{extension}'
            expected[path.name] = path
    if frames == 32:
        for extension in ('.mnn', '.mnn.weight'):
            path = model / f'w4-g16{extension}'
            expected[path.name] = path
    expected['levo_block_bench'] = ROOT / 'native-build/levo_block_bench'
    remote = f'/data/local/tmp/levo-block-bench{suffix}'
    # All shell tokens below are fixed project filenames, not user input.
    output = subprocess.run([hdc, 'shell', f'cd {remote} && sha256sum ' +
                             ' '.join(sorted(expected))], check=True,
                            text=True, capture_output=True).stdout
    observed = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            observed[parts[1]] = parts[0]
    if set(observed) != set(expected):
        raise ValueError(f'Missing or unexpected phone files for {frames} frames')
    receipt = {}
    for name, path in expected.items():
        digest = sha256(path)
        if observed[name] != digest:
            raise ValueError(f'Phone hash mismatch: {frames}/{name}')
        receipt[name] = {'bytes': path.stat().st_size, 'sha256': digest}
    manifest = json.loads((fixture / 'fixture.json').read_text(encoding='utf8'))
    for name, value in manifest.items():
        if receipt[name + '.f32']['sha256'] != value['sha256']:
            raise ValueError(f'Fixture manifest mismatch: {name}')
    return {'local_remote_sha256_match': True, 'fixture': manifest, 'files': receipt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdc', required=True)
    args = parser.parse_args()
    receipts = {str(frames): verify_files(args.hdc, frames) for frames in (32, 256)}
    cache = (ROOT / 'native-build/CMakeCache.txt').read_text(encoding='utf8')
    build_options = {}
    for line in cache.splitlines():
        if line.startswith(('MNN_ROOT:', 'MNN_STATIC_LIBRARY:')):
            key, value = line.split('=', 1)
            build_options[key.split(':')[0]] = value
    archive = Path(build_options['MNN_STATIC_LIBRARY'])
    revision = subprocess.run(['git', '-C', build_options['MNN_ROOT'], 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
    names = [f'phone-block-{profile}-verified.log' for profile in
             ('w32-high', 'w8-high', 'w4-high', 'w4-g16-high', 'w32-low')]
    names += [f'phone-block-{profile}-256.log' for profile in
              ('w32-high', 'w8-high', 'w4-high', 'w32-low')]
    rows = [read_row(ROOT / 'reports' / name) for name in names]
    historical = []
    for profile in ('w32-high', 'w32-high-repeat', 'w8-high', 'w4-high', 'w4-g16-high', 'w32-low'):
        previous = ROOT / 'reports' / f'phone-block-{profile}.log'
        if previous.exists():
            historical.append(read_row(previous))
    path = ROOT / 'reports/levo-test-report.json'
    report = json.loads(path.read_text(encoding='utf8'))
    section = {
        'device': 'Huawei Mate 80 Pro 16 GB / HarmonyOS / aarch64',
        'scope': 'Actual pretrained Mixed DiT block0 only; synthetic fixed input; HDC native CLI, not application or full song inference',
        'real_pretrained_block_parameters': 17302800,
        'linked_mnn_build': {'source_revision': revision, 'archive_bytes': archive.stat().st_size,
                             'archive_sha256': sha256(archive),
                             'clean_upstream_build_attested': False,
                             'note': 'Read-only reuse of the existing Harmony MNN archive; no source or library modifications made by this experiment.'},
        'configuration': {'mnn': '3.6.1', 'backend': 'CPU', 'memory': 'Low',
                          'threads_requested': 4, 'kleidiai_enabled': False,
                          'input_copied_each_iteration': True,
                          'iterations': 11, 'warm_iterations': 10,
                          'warm_statistic': 'median including input/output copy',
                          'process_per_profile': True},
        'limits': [
            'Weights are real; fixed x/modulation fixtures are synthetic, not generated music latents.',
            'Precision Low is a backend compute policy, not W4 weight quantization.',
            '20ms process RSS sampling plus stage samples can miss shorter transient peaks; no dedicated GPU memory was measured.',
            'System MemAvailable is not an application allocation entitlement or an OOM-free guarantee.',
            'Transfer preceded timing; first process inference is not a device cold-storage benchmark.',
            'Short 32-frame FP32 timings varied substantially across runs; no controlled thermal/frequency benchmark.',
            'Single block memory cannot establish whole-DiT or Qwen/DiT/VAE co-residency.',
            'Relative L2 is single-block numerical error, not an audio perceptual quality score.',
        ],
        'full_chain_validated': False, 'real_song_generated': False,
        'file_verification': receipts, 'rows': rows, 'earlier_32_frame_runs': historical,
        'pc_256_frame_alignment': json.loads((ROOT / 'reports/real-dit-block-mnn-256.json').read_text(encoding='utf8')),
        'next_candidate': 'W8 CPU High: 256-frame warm median 53.4ms vs W4 58.4ms in this run; verify complete pretrained model before audio or RTF claims',
    }
    report['phone_block_validation_2026_10_29'] = section
    report['version'] = '0.1.2-phone-block-validation'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    print(json.dumps({'verified_frame_lengths': list(receipts), 'profiles': len(rows),
                      'all_success': all(row['success'] for row in rows),
                      'report': str(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
