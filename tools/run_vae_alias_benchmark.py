"""Run the VAE ranking's copy-elision variants against adjacent phone controls."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from levo_harmony.operator_lab.device import Hdc
from levo_harmony.operator_lab.runner import DeviceLock, ensure_idle, ensure_no_cli, snapshot


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plans(ranking, profile):
    commands = profile['commands']
    by_rank = {}
    for rank, item in enumerate(ranking['methods']['exhaustive']['top10'], 1):
        sites = item.get('selectedPositions')
        if sites is None and item['selectedSites'] == len(item['positions']):
            sites = item['positions']  # Older reports retained every placement for these templates.
        if sites is None or len(sites) != item['selectedSites']:
            raise ValueError('Selected placement IDs required')
        rasters = [commands[node] for site in sites for node in site if commands[node]['type'] == 'Raster']
        names = sorted({op['name'] for op in rasters})
        if not names or len(names) != len(rasters):
            raise ValueError('Candidate needs unique Raster commands')
        by_rank[rank] = {
            'rasterNames': names,
            'priorFullRegionProxyMs': item['optimisticProxyMsPerWindow'],
            'copyOnlyProxyMs': 1000 * math.fsum(op['seconds'] for op in rasters),
        }
    return by_rank


def compare_predictions(candidates, plans_by_rank):
    by_action = defaultdict(list)
    for row in candidates:
        if row['pcmBitIdentical'] and 'measuredLocalCopyDeltaMs' in row:
            by_action[tuple(plans_by_rank[row['rank']]['rasterNames'])].append(row)
    comparisons = []
    for pair in by_action.values():
        measured = statistics.median(row['measuredLocalCopyDeltaMs'] for row in pair)
        copy_error = abs(pair[0]['copyOnlyProxyMs'] - measured)
        region_error = statistics.mean(abs(row['priorFullRegionProxyMs'] - measured) for row in pair)
        comparisons.append((copy_error, region_error))
    if not comparisons:
        return None
    return dict(
        distinctValidatedPlans=len(comparisons),
        copyOnlyMaeMs=statistics.mean(copy for copy, _ in comparisons),
        priorFullRegionProxyMaeMs=statistics.mean(region for _, region in comparisons),
        comparisonTarget='Measured Raster callback savings only',
    )


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('profile', 'ranking', 'latent', 'binary', 'output'):
        parser.add_argument(field, type=Path)
    parser.add_argument('--hdc', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--model', required=True, help='Existing phone bundle root')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--ranks', nargs='+', type=int, default=list(range(1, 11)))
    parser.add_argument('--rounds', type=int, choices=range(2, 7), default=3)
    parser.add_argument('--block-size', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.tag) or not re.fullmatch(r'/[a-zA-Z0-9_./-]+', args.model):
        parser.error('Invalid remote tag or model location')
    if '..' in args.model.split('/') or len(set(args.ranks)) != len(args.ranks):
        parser.error('Invalid model path or duplicate ranks')
    return args


def checked_send(phone, local, remote):
    phone.send(local, remote)
    if digest(local) != phone.sha256(remote):
        raise RuntimeError('Phone input hash mismatch')


def run(args):
    workspace = Path(__file__).resolve().parents[1]
    phone = Hdc(args.hdc, args.device)
    remote = '/data/local/tmp/' + args.tag
    plans_by_rank = plans(read(args.ranking), read(args.profile))
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = {'status': 'running', 'audioSeconds': 10.24, 'runs': [], 'candidates': []}

    def save():
        (args.output / 'campaign.json').write_text(json.dumps(campaign, indent=2), encoding='utf-8')

    def trial(tag, rank):
        out = args.output / tag
        out.mkdir()
        selected = plans_by_rank[rank] if rank else {'rasterNames': []}
        names = selected['rasterNames']
        plan = out / 'plan.json'
        plan.write_text(json.dumps(dict(rasterNames=names, rounds=args.rounds,
                                       captureProfile=not args.no_profile)), encoding='utf-8')
        checked_send(phone, plan, f'{remote}/{tag}.json')
        before = snapshot(phone)
        if not before['awake']:
            phone.shell('power-shell wakeup')
            before = snapshot(phone)
        if not before['awake']:
            raise RuntimeError('Phone is asleep')
        result_dir = f'{remote}/{tag}'
        process = phone.shell(f'timeout 100 {remote}/probe {args.model} {remote}/latent.f32 {result_dir} '
                              f'fused --alias-plan {remote}/{tag}.json', timeout=115)
        (out / 'native.log').write_text(process, encoding='utf-8')
        after = snapshot(phone)
        report = phone.json(result_dir + '/result.json')
        (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if not report['success'] or sorted(report['installedAliases']) != names:
            raise RuntimeError(f'{tag} failed: {report.get("error")} / {report.get("aliasSetupError")}')
        if any(row['aliasCalls'] != len(names) or row['sameSessionRepeatL2'] != 0 for row in report['runs']):
            raise RuntimeError(f'{tag} alias counts or PCM repeat check failed')
        if not after['awake'] or after['lastScreenOff'] != before['lastScreenOff']:
            raise RuntimeError(f'{tag} invalidated by screen-off')
        times = [row['totalSeconds'] * 1000 for row in report['runs'][1:]]
        row = dict(tag=tag, rank=rank, pcmSha256=phone.sha256(result_dir + '/pcm.f32'),
                   warmTrialMs=times, warmMedianMs=statistics.median(times), rssPeakMiB=report['rssPeakMiB'],
                   minAvailableMiB=report['minimumAvailableMiB'], aliases=len(names), batteryC=[before['batteryC'], after['batteryC']])
        if not args.no_profile:
            phone.receive(result_dir + '/profile.json', out / 'profile.json')
            profiled = read(out / 'profile.json')
            if not profiled['pcmMatchesWarmRun']:
                raise RuntimeError(f'{tag} profiled PCM mismatch')
            row['callbackMs'] = {r['name']: r['seconds'] * 1000 for r in profiled['commands'] if r['type'] == 'Raster'}
        campaign['runs'].append(row)
        save()
        print(tag, 'copies', len(names), 'warm ms', round(row['warmMedianMs'], 3), 'RSS MiB', round(row['rssPeakMiB'], 2), flush=True)
        return row

    with DeviceLock(workspace / 'reports/local' / ('operator-device-' + phone.device + '.lock')):
        ensure_no_cli(phone)
        for package in ('com.zhilee.levostreamlab', 'com.local.levostreamdemo'):
            ensure_idle(phone, package, f'/data/app/el2/100/base/{package}/haps/entry/files')
        if phone.shell(f'test ! -e {remote} && mkdir -p {remote} && echo CREATED').strip() != 'CREATED':
            raise RuntimeError('Remote trial already exists')
        checked_send(phone, args.latent, remote + '/latent.f32')
        checked_send(phone, args.binary, remote + '/probe')
        phone.shell('chmod 755 ' + remote + '/probe')
        campaign['deviceModel'] = phone.shell('param get const.product.model').strip()
        campaign['fingerprints'] = dict(graphSha256=phone.sha256(args.model+'/vae/vae.mnn'),
                                        weightSha256=phone.sha256(args.model+'/vae/vae.mnn.weight'),
                                        binarySha256=digest(args.binary), latentSha256=digest(args.latent))
        save()
        control = trial('control-0', 0)
        for offset in range(0, len(args.ranks), args.block_size):
            ranks = args.ranks[offset:offset + args.block_size]
            rows = [trial(f'rank-{rank:02}', rank) for rank in ranks]
            next_control = trial(f'control-{offset // args.block_size + 1}', 0)
            if control['pcmSha256'] != next_control['pcmSha256']:
                raise RuntimeError('Unstable control PCM')
            count = len(rows) + 1
            for position, row in enumerate(rows, 1):
                choice = plans_by_rank[row['rank']]
                fraction = position / count
                adjacent = (1-fraction) * control['warmMedianMs'] + fraction * next_control['warmMedianMs']
                record = {k: v for k, v in row.items() if k != 'callbackMs'}
                record.update({k: v for k, v in choice.items() if k != 'rasterNames'})
                record.update(interpolatedControlMs=adjacent, observedWholeVaeDeltaMs=adjacent-row['warmMedianMs'],
                              controlDriftMs=next_control['warmMedianMs']-control['warmMedianMs'],
                              pcmBitIdentical=row['pcmSha256']==control['pcmSha256'])
                if not args.no_profile:
                    record['measuredLocalCopyDeltaMs'] = math.fsum(
                        ((1-fraction) * control['callbackMs'][name] + fraction * next_control['callbackMs'][name]
                         - row['callbackMs'][name]) for name in choice['rasterNames'])
                campaign['candidates'].append(record)
            save()
            control = next_control
        campaign['status'] = 'complete'
        campaign['pricingComparison'] = compare_predictions(campaign['candidates'], plans_by_rank)
        save()


if __name__ == '__main__':
    run(parse())
