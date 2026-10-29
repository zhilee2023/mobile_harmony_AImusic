"""Merge completed real-backbone/VAE and phone evidence; no new inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdc', required=True)
    args = parser.parse_args()
    path = ROOT / 'reports/levo-test-report.json'
    report = json.loads(path.read_text(encoding='utf8'))
    completed = {}
    for key, name in [('backbone_cpu', 'pretrained-backbone-cpu.json'),
                      ('vae_cpu', 'pretrained-vae-cpu.json'), ('vae_mnn', 'pretrained-vae-mnn.json')]:
        completed[key] = json.loads((ROOT / 'reports' / name).read_text(encoding='utf8'))
    phone = []
    for precision in ('high', 'low'):
        log = ROOT / 'reports' / f'phone-vae-stream-{precision}.log'
        if log.exists():
            lines = [line for line in log.read_text(encoding='utf-8-sig').splitlines()
                     if line.startswith('LEVO_VAE_REPORT ')]
            if len(lines) == 1:
                phone.append(json.loads(lines[0].removeprefix('LEVO_VAE_REPORT ')))
    completed['vae_phone_rows'] = phone
    folder = ROOT / 'models/vae-stream-256'
    names = ['fixture.json', 'vae-256-fp32.mnn', 'vae-256-fp32.mnn.weight']
    names += [f'{prefix}-{index}.f32' for prefix in ('latent', 'pcm', 'emit') for index in range(3)]
    expected = {name: folder / name for name in names}
    expected['levo_vae_bench'] = ROOT / 'native-build/levo_vae_bench'
    output = subprocess.run([args.hdc, 'shell', 'cd /data/local/tmp/levo-vae-stream-256 && sha256sum '+
                             ' '.join(sorted(expected))], check=True, text=True, capture_output=True).stdout
    observed = {line.split()[1]: line.split()[0] for line in output.splitlines()
                if len(line.split()) == 2 and len(line.split()[0]) == 64}
    if set(observed) != set(expected):
        raise ValueError('Phone VAE hash result is incomplete')
    receipts = {}
    for name, local in expected.items():
        value = digest(local)
        if observed[name] != value:
            raise ValueError('Phone VAE file mismatch: '+name)
        receipts[name] = {'bytes': local.stat().st_size, 'sha256': value}
    completed['phone_file_sha256_verified'] = receipts
    hap = ROOT / 'dist/LeVo-Harmony-v0.1.3-unsigned.hap'
    completed['diagnostic_app'] = {'version': '0.1.3', 'bundle': 'com.zhilee.levoharmony',
        'unsigned_build_succeeded': True, 'unsigned_hap_sha256': digest(hap),
        'unsigned_hap_bytes': hap.stat().st_size, 'signed_or_installed_by_agent': False,
        'signing_installation': 'User will operate DevEco manually; no computer-use actions performed',
        'speaker_playback_validated': False}
    tests = (ROOT / 'reports/unit-tests-20261029.log').read_text(encoding='utf-8-sig')
    completed['unit_tests'] = {'passed': 'Ran 11 tests' in tests and tests.rstrip().endswith('OK'),
        'count': 11, 'scope': '8 tiny prototype tests plus3 scheduler tests; actual weights validated separately'}
    completed['streaming_schedule'] = {'window_frames': 256, 'hop_frames': 192, 'overlap_frames': 64,
        'first_stable_pcm_seconds': 7.68, '4000_token_windows': 21,
        'synthetic_latents': True, 'full_dit_in_stream_test': False,
        'lm_prefill_restarted_per_chunk': False,
        'note': 'Scheduler retains256 codec IDs and64 PCM frames; real LM KV still grows with sequence length.'}
    mixed = ROOT / 'models/levo-mixed-codec'
    completed['mixed_dit_download'] = {'completed_range_bytes': sum(p.stat().st_size for p in (mixed/'parts').iterdir() if p.name.isdigit()),
        'expected_range_bytes': 3008576172, 'assembled': (mixed/'mixed-inference.safetensors').exists(),
        'full_real_dit_forward_validated': False}
    completed['as_of_utc'] = datetime.now(timezone.utc).isoformat()
    completed['full_song_generated'] = False
    report['ready_modules_2026_10_29'] = completed
    report['student']['shape_only_not_weight_load'] = False
    report['student']['pretrained_text_backbone_loaded'] = True
    report['generation']['blocking_work'] = ['complete real Mixed DiT download and multi-step validation',
        'train0.6B music/style adapters with legitimate paired data',
        'validate full phone codec and bounded playback queue', 'user signs and installs independent diagnostic App']
    report['version'] = '0.1.3-ready-modules'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    print(json.dumps({'phone_profiles': len(phone), 'file_hashes_verified': len(receipts),
                      'unit_tests': completed['unit_tests'], 'report': str(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
