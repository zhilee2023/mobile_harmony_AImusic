"""Single entrypoint for local model preparation, packaging and phone delivery."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.deployment.contracts import load_profile, PROFILE_IDS
from levo_harmony.deployment.files import read_json


def load_job(path):
    path = Path(path).resolve(); text = path.read_text(encoding='utf-8-sig').replace('${ROOT}', ROOT.as_posix())
    job = json.loads(text)
    # Paths in a job are relative to that file, not the shell's current folder.
    def resolve(value): return str((path.parent / value).resolve()) if value and not Path(value).is_absolute() else value
    if 'source' in job:
        for key in ('path', 'configDirectory', 'tokenizerDirectory', 'checkout'):
            if key in job['source']: job['source'][key] = resolve(job['source'][key])
    if 'baseDirectory' in job: job['baseDirectory'] = resolve(job['baseDirectory'])
    for group in ('files', 'components', 'companions'):
        if group in job: job[group] = {name: resolve(value) for name, value in job[group].items()}
    if job.get('profile') not in PROFILE_IDS: job['profile'] = resolve(job['profile'])
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('profiles'); p.add_argument('name', nargs='?')
    p = commands.add_parser('inspect'); p.add_argument('source'); p.add_argument('--format', default='auto'); p.add_argument('--state-key'); p.add_argument('--strip-prefix')
    p = commands.add_parser('prepare'); p.add_argument('--job', required=True); p.add_argument('--out', required=True); p.add_argument('--mnn-packages'); p.add_argument('--exporter-root')
    p = commands.add_parser('pack'); p.add_argument('--job', required=True); p.add_argument('--out', required=True)
    p = commands.add_parser('baseline'); p.add_argument('--profile', choices=PROFILE_IDS, required=True); p.add_argument('--id', required=True); p.add_argument('--out', required=True); p.add_argument('--public-artifacts')
    p = commands.add_parser('verify'); p.add_argument('bundle')
    p = commands.add_parser('deploy'); p.add_argument('bundle'); p.add_argument('--hdc', required=True); p.add_argument('--device', required=True); p.add_argument('--app', choices=('research', 'music'), default='research'); p.add_argument('--activate', action='store_true')
    p = commands.add_parser('experiments'); p.add_argument('--family', choices=('public', 'small'))
    args = parser.parse_args()
    if args.command == 'profiles': result = load_profile(args.name) if args.name else [load_profile(x) for x in PROFILE_IDS]
    elif args.command == 'inspect':
        from levo_harmony.deployment.checkpoints import inspect_source
        result = inspect_source(args.source, args.format, args.state_key, args.strip_prefix)
    elif args.command == 'prepare':
        from levo_harmony.deployment.components import prepare_component
        result = prepare_component(load_job(args.job), args.out, args.mnn_packages, args.exporter_root)
    elif args.command == 'pack':
        from levo_harmony.deployment.bundles import pack_bundle
        result = pack_bundle(load_job(args.job), args.out)
    elif args.command == 'baseline':
        from levo_harmony.deployment.baselines import baseline_job
        from levo_harmony.deployment.bundles import pack_bundle
        result = pack_bundle(baseline_job(args.profile, args.id, args.public_artifacts), args.out)
    elif args.command == 'verify':
        from levo_harmony.deployment.bundles import verify_bundle
        result = verify_bundle(args.bundle)
    elif args.command == 'deploy':
        from levo_harmony.deployment.device import deploy_bundle
        result = deploy_bundle(args.bundle, args.hdc, args.device, args.app, args.activate)
    else:
        rows = read_json(ROOT / 'experiments/catalog.json')['experiments']
        result = [row for row in rows if not args.family or args.family in row['families']]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from None
