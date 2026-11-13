"""Bounded official OMG conversion for isolated ONNX FFN layout experiments.

Run on the already authorized Linux conversion host. No SSH credentials or
private key material are accepted by this program.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--toolkit-status', type=Path, default=Path.home() / 'cosyvoice-omc/toolkit-status.json')
    parser.add_argument('--target', choices=['om', 'omc'], default='om')
    parser.add_argument('--platform')
    args = parser.parse_args()
    if args.target == 'omc' and not args.platform:
        raise ValueError('Hardware-specific OMC requires the exact supported platform')
    work = args.work.resolve()
    toolkit = json.loads(args.toolkit_status.read_text())
    manifest = json.loads((work / 'conversion-inputs.json').read_text())
    output = work / 'compiled'
    if output.exists():
        raise FileExistsError('Preserve existing conversion; choose a new work directory')
    output.mkdir()
    environment = os.environ.copy()
    environment.update(LD_LIBRARY_PATH=toolkit['libraryPath'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    os.nice(15)
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (1500 * 2**20, 1500 * 2**20))
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    report = {'scope': 'Official OMG conversion of current real FFN layout candidates; no device execution or full DiT claim',
              'target': args.target, 'platform': args.platform, 'toolkitSha256': toolkit.get('toolkitSha256'), 'cases': []}
    for item in manifest['files']:
        path = (work / item['file']).resolve()
        if not path.is_relative_to(work) or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Source path/hash mismatch')
    for item in manifest['cases']:
        identifier = item['id']
        if not identifier.replace('_', '').isalnum(): raise ValueError('Unsafe identifier')
        model = (work / item['model']).resolve()
        if not model.is_relative_to(work): raise ValueError('ONNX path escapes experiment')
        command = [toolkit['omg'], '--framework=5', '--target=' + args.target, '--model=' + str(model),
                   '--output=' + str(output / identifier), '--weight_data_type=FP16', '--input_type=inputs:FP32', '--output_type=outputs:FP32']
        if args.platform: command.append('--platform=' + args.platform)
        row = {'id': identifier, 'success': False, 'weightPrecisionRequested': 'FP16'}
        start = time.monotonic()
        with (output / (identifier + '.log')).open('w') as log:
            try:
                result = subprocess.run(command, cwd=Path(toolkit['omg']).parent, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=150, preexec_fn=limits)
                row['returnCode'] = result.returncode
            except subprocess.TimeoutExpired:
                row['error'] = '150 second bounded conversion timeout'
        row['seconds'] = time.monotonic() - start
        artifact = output / (identifier + '.' + args.target)
        row['success'] = row.get('returnCode') == 0 and artifact.is_file()
        if artifact.is_file(): row.update(bytes=artifact.stat().st_size, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        report['cases'].append(row)
        (work / 'conversion-result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
