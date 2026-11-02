"""HDC transport with immutable slots, remote hash checks and explicit activation."""
import json
import re
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from .bundles import verify_bundle, model_id
from .files import read_json

APPS = {'research': 'com.zhilee.levoharmony', 'music': 'com.zhilee.levomusic'}


def deploy_bundle(bundle, hdc, device, app='research', activate=False):
    if app not in APPS or not re.fullmatch('[a-zA-Z0-9]+', device):
        raise ValueError('Select research/music App and an HDC target ID')
    if activate and app != 'music': raise ValueError('Research bundles are selected per experiment, not a global activation')
    root = Path(bundle).resolve(); result = verify_bundle(root); manifest = read_json(root / 'model-bundle.json')
    identity = model_id(manifest['modelId'])
    if app == 'music' and (manifest['family'] != 'public' or not manifest['musicTrained']):
        raise ValueError('Music App currently requires a trained public-family bundle; small bundles use the research App')
    prefix = f'/data/app/el2/100/base/{APPS[app]}/haps/entry/files'
    final = prefix + '/model-bundles/' + identity; staging = final + '.incoming-' + uuid.uuid4().hex[:10]
    def command(*args):
        return subprocess.run([str(hdc), '-t', device, *args], check=True, capture_output=True, text=True, encoding='utf-8', errors='replace').stdout.strip()
    def shell(command_text): return command('shell', command_text)
    owner = shell('stat -c %u:%g ' + shlex.quote(prefix)).splitlines()[-1].strip()
    if not re.fullmatch(r'\d+:\d+', owner): raise ValueError('Cannot determine sandbox owner; launch/install the App first')
    shell('test ! -e ' + shlex.quote(final)); shell('mkdir -p ' + shlex.quote(staging))
    for item in manifest['assets'] + [{'relative': 'model-bundle.json'}]:
        relative = item['relative']; target = staging + '/' + relative
        shell('mkdir -p ' + shlex.quote(str(Path(target).parent).replace('\\', '/')))
        command('file', 'send', str(root / relative), target)
        if 'sha256' in item:
            remote_hash = shell('sha256sum ' + shlex.quote(target)).split()[0]
            if remote_hash != item['sha256']: raise ValueError(f'Remote hash mismatch; unactivated staging preserved: {relative}')
    shell('chown -R ' + owner + ' ' + shlex.quote(staging))
    shell('mv ' + shlex.quote(staging) + ' ' + shlex.quote(final))
    if activate:
        if app != 'music': raise ValueError('Research bundles are selected per experiment, not a global activation')
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'active-model.json'; marker.write_text(json.dumps({'id': identity}), encoding='utf-8')
            remote = prefix + '/active-model.json.next'
            command('file', 'send', str(marker), remote); shell('chown ' + owner + ' ' + shlex.quote(remote))
            shell('mv ' + shlex.quote(remote) + ' ' + shlex.quote(prefix + '/active-model.json'))
    return {**result, 'app': app, 'remoteSlot': 'model-bundles/' + identity, 'activated': activate,
            'researchWantParameter': {'levoModelBundle': identity}, 'previousModelsPreserved': True}
