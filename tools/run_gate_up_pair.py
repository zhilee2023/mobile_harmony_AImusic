"""Build an isolated W4 gate/up graph on Harmony and compare with stock."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.operator_lab.device import Hdc
from levo_harmony.operator_lab.runner import DeviceLock, ensure_no_cli, run_device

SOURCE_GRAPH = '7a7c1541d4a2503e9c4ff432140a13a03951b208b6a32467b2320b14b7da3e3a'
REPACKED = {'lm/llm.mnn', 'lm/llm.mnn.weight'}
CASES = '/cases/small'


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_source(hdc, root):
    manifest = hdc.json(root + CASES + '/manifest.json')
    files = {entry['relative']: entry['sha256'] for entry in manifest['files']}
    if files['lm/llm.mnn'] != SOURCE_GRAPH:
        raise ValueError('The repacker accepts only the verified pinned 0.6B graph')
    for entry in manifest['files']:
        rel = entry['relative']
        if not re.fullmatch(r'[A-Za-z0-9_/.-]+', rel) or '..' in rel.split('/'):
            raise ValueError('Unsafe asset path')
        if hdc.sha256(root + CASES + '/' + rel) != entry['sha256']:
            raise ValueError('Immutable source bundle changed')
    return manifest


def prepare(hdc, source, target, folder, exporter, direct):
    with DeviceLock(ROOT/'reports/local'/('operator-device-' + hdc.device + '.lock')):
        ensure_no_cli(hdc)
        stock = check_source(hdc, source)
        if hdc.shell('test -e ' + shlex.quote(target) + ' && echo exists || true').strip():
            raise ValueError('Candidate directory already exists. Choose a unique tag.')
        destination = target + CASES
        hdc.shell(f'mkdir -p {destination}/fixtures')
        binary = target + '/gate-up-repack.bin'
        hdc.send(exporter, binary)
        if hdc.sha256(binary) != sha_file(exporter):
            raise ValueError('Repacker binary changed on transfer')
        flag = ' --direct-gate' if direct else ''
        raw = hdc.shell(f'chmod 755 {binary} && {binary} {source}{CASES}/lm/llm.mnn {destination}/lm{flag}')
        result = json.loads(raw)
        if not result.get('success') or result['pairedSites'] != 28:
            raise RuntimeError(f'Gate/up repacker did not pair all layers: {raw}')
        (folder/'repack.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        candidate = dict(stock)
        files = []
        for entry in stock['files']:
            asset = entry['relative']
            original, transformed = source + CASES + '/' + asset, destination + '/' + asset
            if asset not in REPACKED:
                hdc.shell(f'ln {original} {transformed} && test -f {transformed}')
            digest = hdc.sha256(transformed)
            if (digest != entry['sha256']) != (asset in REPACKED):
                raise ValueError(f'Unexpected candidate file difference: {asset}')
            size = int(hdc.shell(f'stat -c %s {transformed}').strip())
            files.append(dict(relative=asset, bytes=size, sha256=digest))
        candidate['files'] = files
        candidate['origin'] = 'Exact W4 gate/up output-row repack; derived immutable experiment'
        candidate['derivedFromCaseSignature'] = stock['caseSignature']
        candidate['repackerSha256'] = sha_file(exporter)
        del candidate['caseSignature']
        candidate['caseSignature'] = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
        manifest = folder/'candidate-manifest.json'
        manifest.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
        hdc.send(manifest, destination + '/manifest.json')
    return result


def compare_logits(baseline, candidate, decode_steps):
    reference, result = [np.fromfile(d/'reference-logits.f32', dtype='<f4')
                         for d in (baseline, candidate)]
    shape = (decode_steps + 1, 16385)
    a, b = [x.reshape(shape).astype(np.float64) for x in (reference, result)]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite gate/up trial logits')
    delta = b - a
    per_forward = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-30)
    return dict(relativeL2=float(np.linalg.norm(delta) / max(np.linalg.norm(a), 1e-30)),
                maxForwardRelativeL2=float(per_forward.max()),
                maxAbsolute=float(np.abs(delta).max()),
                argmaxMatches=int(np.count_nonzero(a.argmax(axis=1) == b.argmax(axis=1))),
                comparedForwards=shape[0])


def verify_pair(baseline, candidate):
    reference = baseline['modelManifest']
    result = candidate['modelManifest']
    if reference['profile'] != result['profile'] or reference['fixture'] != result['fixture']:
        raise ValueError('Model recipe or teacher fixture changed')
    original = {entry['relative']: entry['sha256'] for entry in reference['files']}
    rewritten = {entry['relative']: entry['sha256'] for entry in result['files']}
    expected = set() if baseline['caseSignature'] == candidate['caseSignature'] else REPACKED
    if (original.keys() != rewritten.keys() or
            {asset for asset in original if original[asset] != rewritten[asset]} != expected or
            expected and result.get('derivedFromCaseSignature') != baseline['caseSignature']):
        raise ValueError('Unexpected model or fixture change during candidate comparison')
    for key in ('precision', 'threads', 'fixture', 'requestedDecodeSteps', 'mnnArchiveSha256'):
        if baseline['native'][key] != candidate['native'][key]:
            raise ValueError(f'Operator comparison setting changed: {key}')


def get_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def diagnostic(result, paired, direct=False):
    nodes = result['native']['operators']['nodes']
    summary = {}
    for phase in ('prefill', 'decode'):
        rows = [row for row in nodes if row['phase'] == phase]
        family = [row for row in rows if row['name'].endswith('/GateUpPairW4') and row['type'] == 'Convolution']
        up_gate = [row for row in rows if row['name'].endswith(('/up_proj/Linear', '/gate_proj/Linear'))]
        gate = [row for row in rows if row['name'].endswith('/GateUpPairW4/DirectGate')] if direct else [
            row for row in rows if re.fullmatch(r'/blocks\.\d+/mlp/Mul_output_0', row['name'])]
        required = 28 if paired else 56
        if len(gate) != 28 or (len(family), len(up_gate)) != ((required, 0) if paired else (0, required)):
            raise RuntimeError(f'{phase} did not execute the expected graph')
        extra = [row for row in rows if row['type'] == 'Raster' and
                 re.fullmatch(r'/blocks\.\d+/mlp/Mul_output_0_raster_[01]', row['name'])]
        if len(extra) != (56 if paired and not direct else 0):
            raise RuntimeError(f'{phase} gate materialization differed from the expected graph')
        summary[phase] = dict(projectionCalls=sum(r['calls'] for r in family + up_gate),
                              gateCalls=sum(r['calls'] for r in gate),
                              projectionCallbackMs=1000*sum(r['seconds'] for r in family + up_gate),
                              gateCallbackMs=1000*sum(r['seconds'] for r in gate),
                              rasterCalls=sum(r['calls'] for r in extra),
                              splitRasterCallbackMs=1000*sum(r['seconds'] for r in extra))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    parser.add_argument('--stock-root', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--diagnostic-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reuse-candidate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--direct-gate', action='store_true')
    parser.add_argument('--hdc', default=r'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', args.tag):
        parser.error('Use an alphanumeric tag of at most 48 characters')
    if not re.fullmatch(r'/data/local/tmp/[A-Za-z0-9_-]+', args.stock_root):
        parser.error('Stock root must name a CLI case under /data/local/tmp')
    if args.prepare_only and (args.diagnostic_only or args.reuse_candidate) or args.resume and not args.reuse_candidate:
        parser.error('Prepare-only is for a new candidate. Resume requires --reuse-candidate.')

    hdc = Hdc(args.hdc, args.device)
    stock_root = args.stock_root
    trial_root = '/data/local/tmp/' + args.tag + '-gate-up'
    study = ROOT/'reports/local'/args.tag
    build_folder = study/'prepare'
    summary_path = study/'performance.json'
    if summary_path.exists() and not args.resume:
        raise ValueError('Keep previous trial results')
    if args.reuse_candidate:
        repack = get_json(build_folder/'repack.json')
        reference_manifest = get_json(build_folder/'candidate-manifest.json')
        if hdc.json(trial_root + CASES + '/manifest.json') != reference_manifest:
            raise ValueError('Prepared candidate manifest differs from the stored build')
    else:
        build_folder.mkdir(parents=True, exist_ok=False)
        repack = prepare(hdc, stock_root, trial_root, build_folder,
                         ROOT/'native-build/llm_operator_gate_up_pair', args.direct_gate)
    if bool(repack.get('directGate')) != args.direct_gate:
        raise ValueError('Requested gate implementation differs from the prepared graph')
    if args.prepare_only:
        print(json.dumps(repack), flush=True)
        return
    campaign = {'repack': repack, 'scope': '513-position teacher workload on phone CLI. No music generation.',
                'diagnostics': {}, 'trials': []}
    if args.resume and summary_path.exists():
        campaign = get_json(summary_path)

    def save():
        summary_path.write_text(json.dumps(campaign, indent=2), encoding='utf-8')

    def invoke(mode, variant, suffix, steps, rounds):
        location = study / suffix
        retry = 1
        while location.exists():
            location = study / f'{suffix}-retry{retry}'
            retry += 1
        direct = variant == 'pair' and args.direct_gate
        executable = 'llm_operator_gate_up_direct' if direct else 'llm_operator_baseline'
        result = run_device(hdc, ROOT, stock_root if variant == 'stock' else trial_root, 0,
            dict(mode=mode, decodeSteps=steps, rounds=rounds, threads=4, precision='high',
                 runTag=f'{args.tag}-{location.name}', maxRssMiB=6144, minAvailableMiB=650,
                 requireCandidateCalls=direct),
            location, transport='cli', binary=ROOT/'native-build'/executable)
        if result['status'] != 'completed' or not result['screenAccepted']:
            raise RuntimeError(f'Incomplete or uncontrolled run {suffix}')
        if direct and result['native']['candidateCalls'] != 28 * (1 + steps) * rounds:
            raise RuntimeError(f'Direct gate did not run at all 28 sites in {suffix}')
        return location, result

    diagnostics_path = study/'campaign.json'
    if diagnostics_path.exists():
        base, proposed = study/'ops-stock', study/'ops-pair'
        previous = get_json(diagnostics_path).get('diagnosticDirectories')
        if previous is not None:
            base, proposed = [study/folder for folder in previous]
        stock, pair = [get_json(folder/'llm-operator-report.json') for folder in (base, proposed)]
    else:
        base, stock = invoke('operators', 'stock', 'ops-stock', 2, 1)
        proposed, pair = invoke('operators', 'pair', 'ops-pair', 2, 1)
    verify_pair(stock, pair)
    campaign['diagnostics'] = {'stock': diagnostic(stock, False), 'pair': diagnostic(pair, True, args.direct_gate),
                               'numerics': compare_logits(base, proposed, 2)}
    if not diagnostics_path.exists():
        record = dict(campaign, diagnosticDirectories=[base.name, proposed.name])
        diagnostics_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    if campaign['diagnostics']['numerics']['maxForwardRelativeL2'] > .001:
        raise RuntimeError('Candidate failed per-forward logits tolerance')
    if args.diagnostic_only:
        print(json.dumps(campaign['diagnostics']), flush=True)
        return

    order = ('stock', 'pair', 'pair', 'stock')
    reference, original = None, None
    for index, trial in enumerate(campaign['trials']):
        if trial['variant'] != order[index] or trial['numerics']['maxForwardRelativeL2'] > .001:
            raise ValueError('Prepared timing run did not pass')
        if index == 0:
            reference = study/trial.get('directory', 'time-0-stock')
            original = get_json(reference/'llm-operator-report.json')
    for index in range(len(campaign['trials']), len(order)):
        variant = order[index]
        folder, data = invoke('baseline', variant, f'time-{index}-{variant}', 64, 3)
        if reference is None:
            reference, original = folder, data
        verify_pair(original, data)
        warm = data['native']['rounds'][1:]
        row = dict(variant=variant, directory=folder.name,
                   prefillMs=statistics.mean(r['prefillMs'] for r in warm),
                   decodeMsPerToken=statistics.mean(r['decodeMsPerToken'] for r in warm),
                   totalMs=statistics.mean(r['prefillMs']+r['decodeMs'] for r in warm),
                   loadMs=data['native']['loadMs'], rssPeakMiB=data['native']['rssPeakMiB'],
                   directGateCalls=data['native']['candidateCalls'],
                   batteryStartC=data['controls'][0]['batteryC'], batteryEndC=data['controls'][-1]['batteryC'],
                   numerics=compare_logits(reference, folder, 64))
        campaign['trials'].append(row)
        save()
        print(json.dumps(row), flush=True)
        if row['numerics']['maxForwardRelativeL2'] > .001:
            raise RuntimeError('Candidate failed whole-workload logits tolerance')


if __name__ == '__main__':
    main()
