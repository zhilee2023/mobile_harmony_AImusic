"""Publish complete signed-App evidence, retaining failed strict parity checks."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def load(name):return json.loads((ROOT/'reports'/name).read_text(encoding='utf-8-sig'))


def main():
    report_path=ROOT/'reports/levo-test-report.json'
    report=json.loads(report_path.read_text(encoding='utf8'))
    bundle=json.loads((ROOT/'models/three-module-bundle/bundle.json').read_text())
    receipts={line.split()[1]:line.split()[0] for line in
              (ROOT/'reports/phone-chain-file-hashes.log').read_text(encoding='utf-8-sig').splitlines()
              if len(line.split())==2 and re.fullmatch('[0-9a-f]{64}',line.split()[0])}
    if set(receipts)!=set(bundle['files']):raise ValueError('Missing phone file hash evidence')
    for name,metadata in bundle['files'].items():
        path=ROOT/'models/three-module-bundle'/name
        if receipts[name]!=metadata['sha256'] or sha(path)!=metadata['sha256']:
            raise ValueError('Phone/local/bundle checksum mismatch: '+name)
    cases=[('chain-serial-20261029-r2','chainSerial','serial'),
           ('chain-keep-llm-20261029-r1','chainKeepLlm','keep-llm'),
           ('chain-all-20261029-r1','chainAll','all'),
           ('chain-all-w4-20261029-r1','chainAllW4','all-w4'),
           ('chain-all-w4-20261029-r2','chainAllW4','all-w4'),
           ('chain-all-20261029-r2','chainAll','all')]
    rows=[]
    for run,key,policy in cases:
        archive=load('signed-'+run+'.json')
        if archive['lastRunId']!=run:raise ValueError('Stale result file')
        row=archive[key]
        if not row['success'] or not row['flowAndMemoryPassed'] or row['requestedPolicy']!=policy:
            raise ValueError('Incomplete memory/flow case: '+run)
        if row['prefillTokens']!=514 or row['kvPositionsAfterDecode']!=769 or len(row['generatedCodecIds'])!=256:
            raise ValueError('LLM chain capacity not exercised')
        if any(not 0<=code<16384 for code in row['generatedCodecIds']):raise ValueError('Invalid codec ID')
        if row['frames']!=256 or row['ditSteps']!=2 or row['validPcmSeconds']!=10.24:
            raise ValueError('Different codec window/solver scope')
        if not math.isclose(row['pipelineRtfThroughPcm'],row['pipelineSecondsThroughPcm']/10.24,rel_tol=1e-10):
            raise ValueError('RTF denominator differs from generated PCM duration')
        if row['music_trained'] or row['real_song_generated']:raise ValueError('Untrained diagnostic scope changed')
        if row['llmKvRetainedThroughCodec']!=(policy!='serial'):
            raise ValueError('LLM lifetime invariant changed')
        if row['allThreeSessionsRetainedThroughForward']!=(policy.startswith('all')):
            raise ValueError('Full co-residency invariant changed')
        names={stage['stage'] for stage in row['stages']}
        required={'llm-256-codes-complete','dit-step-1','dit-step-2','pcm-finite-10.24s','all-models-destroyed',
                  'release-plus-1-second','post-timing-full-dit-parity-measured'}
        if not required<=names:raise ValueError('Missing executed stage evidence')
        if policy.startswith('all') and 'all-three-sessions-loaded' not in names:
            raise ValueError('All-session residency unproven')
        if not all(math.isfinite(row[k]) for k in ('pcmPeak','pcmRms','pipelineSecondsThroughPcm','phoneVsPcSameQuantVelocityL2')):
            raise ValueError('Non-finite report')
        error=row['phoneVsPcSameQuantVelocityL2']
        if row['strictNumericalParityPassed']!=(error<row['strictNumericalParityThreshold']):
            raise ValueError('Numerical error was mislabeled')
        row.update(run_id=run,source_archive_sha256=sha(ROOT/'reports'/('signed-'+run+'.json')))
        row['app_memory_before']=archive['memoryBeforeRun']
        row['app_memory_after']=archive['memory']
        rows.append(row)
    llm=[]
    for kind,prefix in [('short',138),('long',514)]:
        run=f'llm-{kind}-20261029-r1'
        archive=load('signed-'+run+'.json');row=archive['llm'+kind.title()]
        if archive['lastRunId']!=run or not row['success'] or not row['llmOnly']:
            raise ValueError('Incomplete standalone LLM probe')
        if row['prefillTokens']!=prefix or row['kvPositionsAfterDecode']!=prefix+255:
            raise ValueError('Wrong standalone cache length')
        llm.append(row)
    initial=load('signed-chain-serial-r1.json')['chainSerial']
    if initial['success'] or initial['error']!='Phone full-DiT W8 parity failed':
        raise ValueError('Preserve the original strict parity failure')
    vae_archive=load('signed-app-high.json')
    if not vae_archive['low']['success'] or not vae_archive['high']['success']:
        raise ValueError('Signed App VAE baseline missing')
    full=load('full-dit-mnn-256.json');w4=load('full-dit-w4.json')
    if full['strict_cfm_tensors']!=526 or not full['all_39_layers'] or w4['complete_layers']!=39:
        raise ValueError('Full-model coverage unproven')
    # Verify exported ONNX covers each real transformer block, not block0 repeats.
    import onnx
    graph=onnx.load(str(ROOT/'models/full-dit-256/full-external.onnx'),load_external_data=False)
    layers={int(match.group(1)) for tensor in graph.graph.initializer
            if (match:=re.search(r'cfm\.estimator\.h\.(\d+)\.',tensor.name))}
    if layers!=set(range(39)):raise ValueError('Incomplete DiT layer initializers')
    files=('native-bench/ChainBench.cpp','native-bench/ChainBench.hpp','app/entry/src/main/cpp/napi_init.cpp',
           'app/entry/src/main/ets/pages/Index.ets','app/entry/src/main/ets/entryability/EntryAbility.ets',
           'tools/export_full_dit.py','tools/export_diagnostic_student.py')
    artifact=ROOT/'app/entry/build/default/outputs/default/entry-default-signed.hap'
    profile_binary_sha=sha(ROOT/'native-build/levo_dit_profile')
    if profile_binary_sha!=(ROOT/'reports/dit-profile-binary-hash.log').read_text(encoding='utf-8-sig').split()[0]:
        raise ValueError('Phone operator profiler binary differs')
    tests=(ROOT/'reports/unit-tests-v014.log').read_text(encoding='utf-8-sig')
    if not re.search(r'Ran 15 tests',tests) or not tests.rstrip().endswith('OK'):
        raise ValueError('Prototype/KV/untrained-guard regression evidence missing')
    profiles=[]
    for variant in ('w8','w4'):
        lines=[line.removeprefix('LEVO_DIT_PROFILE ') for line in
               (ROOT/f'reports/full-dit-{variant}-phone-profile.log').read_text(encoding='utf-8-sig').splitlines()
               if line.startswith('LEVO_DIT_PROFILE ')]
        if len(lines)!=1:raise ValueError('Missing complete operator profile')
        profile=json.loads(lines[0])
        if not profile['success'] or profile['variant']!=variant:raise ValueError('Invalid operator profile')
        for category in ('FFN up projection','FFN down projection','QKV projection','Attention output projection'):
            entry=next(row for row in profile['byCategory'] if row['name']==category)
            if entry['calls']!=39:raise ValueError('Profile did not execute all39 projections')
        profiles.append(profile)
    previous=[]
    for name in ('kleidi-ffn-bench.json','kleidi-ffn-bench-repeat.json'):
        oldpath=Path(r'C:\cosyvoice-harmony\reports')/name
        data=json.loads(oldpath.read_text(encoding='utf-8-sig'))
        for row in data['rows']:
            if row['threads']==6:
                times=[item['seconds'] for item in row['runs'][1:]]
                previous.append({'source_report':name,'source_sha256':sha(oldpath),'m':row['m'],'k':row['k'],
                                 'n':row['n'],'threads':6,'warm_median_seconds':statistics.median(times),
                                 'warm_best_seconds':min(times),'relative_l2_vs_captured_mnn':row['relativeL2VsCapturedMnn']})
    report['signed_three_module_validation_2026_10_29']={
        'as_of_utc':datetime.now(timezone.utc).isoformat(),'scope':'Signed Harmony App process, fixed diagnostic inputs, no UI automation',
        'package':{'bundle_name':'com.zhilee.levoharmony','version':'0.1.4','local_signed_hap_sha256':sha(artifact),
                   'local_signed_hap_bytes':artifact.stat().st_size,'signing_material_published':False},
        'student':json.loads((ROOT/'models/student-diagnostic-mnn/diagnostic-student.json').read_text()),
        'complete_dit_pc_w8':full,'complete_dit_pc_w4':w4,'vae_signed_app':{'high':vae_archive['high'],'low':vae_archive['low']},
        'complete_dit_operator_profiles':profiles,'previous_cosyvoice_kleidi_kernel_evidence':previous,
        'operator_profile_source_sha256':sha(ROOT/'native-bench/DitProfile.cpp'),
        'operator_profile_binary_sha256':profile_binary_sha,
        'regression_tests':{'count':15,'passed':True,'scope':'Tiny student, untrained guard, KV and streaming protocols; not music-quality proof'},
        'three_module_cases':rows,'standalone_llm_prefix_probes':llm,'initial_strict_gate_failure_preserved':initial,
        'verified_phone_files':bundle['files'],'source_sha256':{name:sha(ROOT/name) for name in files},
        'limits':['Untrained music/style adapters and unrecovered26-layer pruning; no generated-song quality proof',
                  'Two DiT steps are a requested diagnostic assumption, not official20-step quality validation',
                  'One256-frame10.24s window and769 KV positions; not4000-token or minutes-long streaming stability',
                  'Full-model W8/W4 strict1% cross-platform numerical parity failed; finite flow/memory success is separate',
                  'Stage trace and20ms RSS sampling can miss brief peaks; no OOM guarantee',
                  'CPU scheduling/thermal state varied; do not attribute end-to-end speed differences solely to residency',
                  'Pipeline RTF includes initialization through PCM, excludes subsequent parity diagnostic and release wait',
                  'HiDebug returned oversized limit sentinels; not a tested App allocation ceiling']}
    report['version']='0.1.4-signed-three-module-memory'
    report['generation']['blocking_work']=['train music/style adapters and recover pruned backbone on legitimate data',
        'resolve complete cross-platform numerical differences and listening-quality validation',
        'validate multiple bounded codec windows while resuming the same phone LLM KV and a bounded player queue']
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    for row in rows:
        print(json.dumps({key:row[key] for key in ('run_id','requestedPolicy','pipelineRtfThroughPcm','pipelinePeakRssKiB',
            'ditTwoStepSeconds','vaeSeconds','decodeTokensPerSecond','strictNumericalParityPassed')}))


if __name__=='__main__':main()
