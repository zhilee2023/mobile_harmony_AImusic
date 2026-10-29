"""Append completed five-window evidence; do not count observer timeout as completion."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def main():
    path=ROOT/'reports/levo-test-report.json'
    report=json.loads(path.read_text(encoding='utf8'))
    run='five-chunks-20261029-r1'
    source=ROOT/'reports'/f'signed-{run}.json'
    archive=json.loads(source.read_text(encoding='utf-8-sig'))
    if archive['lastRunId']!=run:raise ValueError('Stale five-chunk report')
    result=archive['chunks5']
    if not result['success'] or not result['flowAndMemoryPassed']:raise ValueError('Five-window flow incomplete')
    if len(result['chunks'])!=5 or result['generatedCodecIds']!=1024 or result['finalKvPositions']!=1537:
        raise ValueError('KV/token capacity was not exercised')
    if result['ditStepsPerChunk']!=2 or result['windowFrames']!=256 or result['overlapFrames']!=64:
        raise ValueError('Wrong solver/window profile')
    for key in ('oneLlmSession','onePrefill','oneDitSession','oneVaeSession'):
        if result[key] is not True:raise ValueError('Session was not retained: '+key)
    if result['pcmAccumulatedAcrossWholeSong'] or result['maxCodeWindowIds']!=256:
        raise ValueError('Bounded working-set invariant changed')
    for index,row in enumerate(result['chunks']):
        expected=(index+1)*192 if index<4 else 1024
        if (row['chunk']!=index+1 or row['newCodecIds']!=(256 if index==0 else 192)
                or row['contextFrames']!=(0 if index==0 else 64)
                or row['kvPositions']!=769+192*index
                or not math.isclose(row['cumulativePcmSeconds'],expected/25,abs_tol=1e-9)):
            raise ValueError('Chunk join duplicated/dropped frames or reset KV')
    if not math.isclose(result['validPcmSeconds'],40.96,abs_tol=1e-9):raise ValueError('Invalid duration')
    if not math.isclose(result['pipelineRtfThroughPcm'],result['pipelineSecondsThroughPcm']/40.96,rel_tol=1e-10):
        raise ValueError('Incorrect RTF denominator')
    receipt=json.loads((ROOT/'reports'/f'signed-{run}-receipt.json').read_text(encoding='utf-8-sig'))
    baseline_hap=ROOT/'dist/LeVo-v0.1.5-five-chunk-baseline-signed-local.hap'
    if sha(baseline_hap)!=receipt['hapSha256']:raise ValueError('Baseline package not preserved')
    low_lines=[line.removeprefix('LEVO_DIT_PROFILE ') for line in
        (ROOT/'reports/full-dit-w8-low-phone-probe.log').read_text(encoding='utf-8-sig').splitlines()
        if line.startswith('LEVO_DIT_PROFILE ')]
    if len(low_lines)!=1:raise ValueError('Missing low precision probe')
    low=json.loads(low_lines[0])
    if not low['success'] or low.get('precision')!='Low':raise ValueError('No finite low precision probe')
    low_binary_sha=sha(ROOT/'native-build/levo_dit_low_profile')
    if low_binary_sha!=(ROOT/'reports/dit-low-profile-binary-hash.log').read_text(encoding='utf-8-sig').split()[0]:
        raise ValueError('Low operator probe phone binary mismatch')
    low['verified_phone_binary_sha256']=low_binary_sha
    low['source_sha256']=sha(ROOT/'native-bench/DitProfile.cpp')
    fusion=json.loads((ROOT/'reports/full-dit-fusion-isolation.json').read_text())
    if fusion['conversion_success'] or fusion['model_exists']:raise ValueError('Fusion route changed; validate it separately')
    section={'as_of_utc':datetime.now(timezone.utc).isoformat(),'baseline':result,
        'baseline_package_and_source_receipt':receipt,'baseline_archive_sha256':sha(source),
        'full_dit_low_operator_probe':low,'fusion_candidate_failure':fusion,
        'limits':['Random music/style adapters; no trained singing or listening proof',
                  'One continuous40.96s numerical stream, not unlimited-duration stability',
                  'Sequential CPU stages produce PCM blocks; no concurrent computation or speaker playback',
                  'Two steps are a requested assumption; not official20-step quality validation',
                  '20ms RSS sampling and one run; screen/thermal state not continuously controlled',
                  'TargetRTF1.5 not achieved; original strict numerical parity still failed']}
    tuned_path=ROOT/'reports/signed-five-chunks-tuned-20261029-r1.json'
    if tuned_path.exists():
        tuned_archive=json.loads(tuned_path.read_text(encoding='utf-8-sig'))
        if tuned_archive['lastRunId']=='five-chunks-tuned-20261029-r1':
            tuned=tuned_archive['chunks5Tuned']
            section['tuned_observed_result']=tuned
            section['tuned_performance_comparable']=False
            section['tuned_caveat']='Observer timed out and SLEEP was confirmed while same App PID remained live; no uninterrupted foreground performance claim'
    else:
        observation=json.loads((ROOT/'reports/five-tuned-process-observation.json').read_text(encoding='utf-8-sig'))
        section['tuned_observed_state']={'completion_verified':False,'timestamped_process_observation':observation}
    repeat_path=ROOT/'reports/signed-five-chunks-tuned-20261029-r2.json'
    if repeat_path.exists():
        repeated_archive=json.loads(repeat_path.read_text(encoding='utf-8-sig'))
        if repeated_archive['lastRunId']!='five-chunks-tuned-20261029-r2':raise ValueError('Stale tuned repeat')
        repeated=repeated_archive['chunks5Tuned']
        if (not repeated['success'] or repeated['generatedCodecIds']!=1024 or repeated['finalKvPositions']!=1537
                or len(repeated['chunks'])!=5 or not math.isclose(repeated['validPcmSeconds'],40.96)):
            raise ValueError('Tuned repeat did not exercise the same full stream')
        section['tuned_foreground_repeat']=repeated
        section['tuned_repeat_package_receipt']=json.loads((ROOT/'reports/signed-five-chunks-tuned-20261029-r2-receipt.json').read_text(encoding='utf-8-sig'))
        section['tuned_repeat_power_observed_at_end']=(ROOT/'reports/five-tuned-r2-power-end.log').read_text(encoding='utf-8-sig').strip()
        section['tuned_variant_promoted']=False
    pair=[]
    for switch in ('off','on'):
        log=ROOT/f'reports/llm-kleidi-{switch}-cli.log'
        if not log.exists():continue
        lines=[line.removeprefix('LEVO_CHAIN_REPORT ') for line in log.read_text(encoding='utf-8-sig').splitlines()
               if line.startswith('LEVO_CHAIN_REPORT ')]
        if len(lines)!=1:raise ValueError('No unique CLI switch result')
        row=json.loads(lines[0])
        if not row['success'] or row['prefillTokens']!=514 or row['kvPositionsAfterDecode']!=769:
            raise ValueError('Invalid same-shape CLI switch probe')
        row['source_log_sha256']=sha(log);pair.append(row)
    if pair:
        binary_sha=sha(ROOT/'native-build/levo_chain_bench')
        if binary_sha!=(ROOT/'reports/llm-kleidi-probe-binary-hash.log').read_text(encoding='utf-8-sig').split()[0]:
            raise ValueError('CLI probe phone binary differs')
        section['llm_kleidi_switch_cli']={'cases':pair,'verified_phone_binary_sha256':binary_sha,
            'limits':'Same-binary CLI comparison, not App/whole-chain RTF; enabling flag does not prove each kernel selected KleidiAI; no default promotion'}
    tests=(ROOT/'reports/unit-tests-v015.log').read_text(encoding='utf-8-sig')
    if 'Ran 15 tests' not in tests or not tests.rstrip().endswith('OK'):raise ValueError('Regression test evidence missing')
    section['regression_tests']={'passed':True,'count':15,'scope':'Tiny KV/scheduler/student guards; not music-quality proof'}
    report['five_chunk_validation_2026_10_29']=section
    report['version']='0.1.5-five-chunk-diagnostic'
    report['generation']['blocking_work']=['train music/style adapters and recover pruned backbone on legitimate data',
        'resolve cross-platform numerical differences and evaluate listening quality',
        'validate audible playback and longer-than-five-window stability under controlled foreground conditions',
        'implement and validate real FFN/VAE kernel optimizations; targetRTF1.5 remains unachieved']
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(json.dumps({'five_chunks_complete':True,'seconds':result['pipelineSecondsThroughPcm'],
        'duration':result['validPcmSeconds'],'rtf':result['pipelineRtfThroughPcm'],
        'peak_mib':result['pipelinePeakRssKiB']/1024,'final_kv':result['finalKvPositions']}))


if __name__=='__main__':main()
