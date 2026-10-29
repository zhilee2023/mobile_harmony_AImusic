"""Freeze completed lifecycle/short-window evidence in the unified report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            value.update(data)
    return value.hexdigest()


def log(name, prefix):
    lines = [line[len(prefix):] for line in (ROOT/'reports'/name).read_text(encoding='utf-8-sig').splitlines()
             if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError('No unique completed record: '+name)
    result = json.loads(lines[0])
    if not result['success']:
        raise ValueError('Failed experiment: '+name)
    result['source_log'] = name
    return result


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--hdc',required=True);args=parser.parse_args()
    path = ROOT/'reports/levo-test-report.json';report=json.loads(path.read_text(encoding='utf8'))
    rows = [log(name,'LEVO_VAE_REPORT ') for name in (
        'phone-vae-64-low.log','phone-vae-256-low-repeat.log',
        'phone-vae-64-low-4-high-power.log','phone-vae-64-low-6-high-power.log',
        'phone-vae-256-low-6-high-power.log')]
    for row in rows:
        if row['validPcmSeconds'] != 25.6 or len(row['rows']) != row['chunks']:
            raise ValueError('Window comparison has a different output duration')
        state=ROOT/'reports'/row['source_log'].replace('.log','-state.log')
        if state.exists():
            match=re.search(r'Current State:\s*(\w+)',state.read_text(encoding='utf-8-sig'))
            row['powerStateObservedBefore']=match.group(1) if match else 'unavailable'
        else:
            row['powerStateObservedBefore']='not captured; SLEEP observed nearby, not proof of state throughout'
    pc=json.loads((ROOT/'reports/pretrained-vae-mnn-64.json').read_text())
    baseline=ROOT/'models/vae-stream-256';parts=[np.fromfile(baseline/f'latent-{i}.f32',dtype='<f4').reshape(1,64,256) for i in range(3)]
    timeline=np.concatenate((parts[0][...,:192],parts[1][...,:192],parts[2]),axis=-1)
    if hashlib.sha256(timeline.tobytes()).hexdigest()!=pc['stream_fixture']['source_latent_sha256']:
        raise ValueError('64-frame experiment did not use the same latent timeline')
    folder=ROOT/'models/vae-stream-64'
    names=list(pc['stream_fixture']['files'])+['fixture.json','vae-64-fp32.mnn','vae-64-fp32.mnn.weight']
    binaries=['levo_vae_bench','levo_vae_bench-affinity','levo_vae_profile']
    output=subprocess.run([args.hdc,'shell','cd /data/local/tmp/levo-vae-stream-64 && sha256sum '+
        ' '.join(sorted(names+binaries))],check=True,capture_output=True,text=True).stdout
    observed={line.split()[1]:line.split()[0] for line in output.splitlines() if len(line.split())==2 and len(line.split()[0])==64}
    if set(observed)!=set(names+binaries):raise ValueError('Missing phone SHA256 evidence')
    receipts={}
    for name in names:
        digest=sha(folder/name)
        if digest!=observed[name]:raise ValueError('Phone file mismatch: '+name)
        receipts[name]={'bytes':(folder/name).stat().st_size,'sha256':digest}
    for remote,local in [('levo_vae_bench-affinity','levo_vae_bench'),('levo_vae_profile','levo_vae_profile')]:
        if observed[remote]!=sha(ROOT/'native-build'/local):raise ValueError('Current phone binary mismatch')
    pruned=json.loads((ROOT/'reports/real-pruned-backbone.json').read_text())
    if not pruned['saved_reload_exact'] or pruned['actual_student_parameters']!=598178304:
        raise ValueError('Real pruned checkpoint was not validated')
    weights=sorted((ROOT/pruned['saved_backbone_relative_path']).glob('*.safetensors'))
    if sum(p.stat().st_size for p in weights)!=pruned['saved_weight_bytes']:raise ValueError('Saved pruned weight size changed')
    pruned['saved_weight_sha256']={p.name:sha(p) for p in weights}
    (ROOT/pruned['saved_backbone_relative_path']/'pruning-manifest.json').write_text(json.dumps(pruned,indent=2),encoding='utf8')
    test_text=(ROOT/'reports/unit-tests-stream-lifecycle.log').read_text(encoding='utf-8-sig')
    match=re.search(r'Ran (\d+) tests',test_text)
    if not match or not test_text.rstrip().endswith('OK'):raise ValueError('Latest regression tests did not pass')
    experiments={'as_of_utc':datetime.now(timezone.utc).isoformat(),
        'real_pruning':pruned,'vae_same_timeline_pc':pc,'vae_phone':rows,
        'vae_hotspots':log('phone-vae-64-low-hotspots.log','LEVO_VAE_PROFILE '),
        'phone_fixture_sha256_verified':receipts,'phone_binary_sha256':{name:observed[name] for name in binaries},
        'historical_binary_note':'levo_vae_bench predates affinity instrumentation; other two match current local binaries',
        'source_sha256':{name:sha(ROOT/name) for name in ('levo_harmony/model.py','levo_harmony/pruning.py',
            'native-bench/VaeBench.cpp','native-bench/VaeBench.hpp','native-bench/VaeProfile.cpp')},
        'unit_tests':{'passed':True,'count':int(match.group(1)),'scope':'Tiny prototype/lifecycle/scheduler fixtures; real pretrained tests are recorded separately'},
        'stateful_decode':{'implemented':True,'pause_keeps_same_kv':True,'close_releases_owned_kv_reference':True,
                          'trained_music_checkpoint_available':False},
        'limits':['Synthetic latent stream, not generated music or full codec RTF',
                  'No complete LLM/DiT/VAE co-residency validated',
                  'Screen state and CPU frequency varied; do not treat these runs as controlled foreground benchmarks',
                  'Smaller VAE windows change boundary error; no listening-quality claim'],
        'mixed_download_range_bytes':sum(p.stat().st_size for p in (ROOT/'models/levo-mixed-codec/parts').iterdir() if p.name.isdigit())}
    report['lifecycle_experiments_2026_10_29']=experiments
    report['student']['optional_26_layer_pruning_experiment'].update({'real_weights_saved_and_reloaded':True,
        'backbone_parameters':pruned['backbone_parameters'],'parameters':pruned['actual_student_parameters'],
        'music_quality_validated':False,'recovery_training_required':True})
    report['version']='0.1.3-lifecycle-validation'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(json.dumps({'phone_profiles':len(rows),'same_latent_verified':True,'saved_student_parameters':pruned['actual_student_parameters'],
                      'regression_tests':int(match.group(1))},ensure_ascii=False))


if __name__=='__main__':main()
