"""Record pure-Conv ablation separately from output-equivalent VAE tests."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import statistics

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';MODELS=ROOT/'models/vae-npu-pureconv'
RUNS=['vae-pureconv-default-r1','vae-pureconv-high-r1','vae-residual-fresh-default-r1',
      'vae-residual-fresh-high-r1','vae-pureconv-default-r2','vae-pureconv-high-r2']

def text(p):return p.read_text(encoding='utf-8-sig')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    manifest=json.loads(text(MODELS/'manifest.json'))
    assert manifest['ablation'].startswith('Real Conv only')
    assert [c['operationCount'] for c in manifest['candidateCases']]==[6,1,1,1]
    conversion=json.loads(text(R/'vae-pureconv-conversion-result-20261030.json'))
    assert conversion['complete'] and len(conversion['cases'])==4
    identities=text(R/'vae-pureconv-postrun-artifacts-shas.log')
    before_binary=ROOT/'dist/levo-pureconv-before-counter-fix-local.bin'
    assert sha(before_binary) in identities
    for c in conversion['cases']:
        assert c['success'] and c['returnCode']==0
        assert sha(MODELS/(c['id']+'-fp16.omc'))==c['sha256'] and c['sha256'] in identities
    assets={}
    for p in MODELS.iterdir():
        if p.suffix in ('.mnn','.weight','.f32','.omc') or p.name=='manifest.json':
            digest=sha(p);assert digest in identities;assets[p.name]=digest
    records=[];summary=[]
    for run in RUNS:
        raw=text(R/(run+'.log'))
        assert re.findall(r'^LEVO_PROBE_EXIT (\d+)\s*$',raw,re.M)==['0']
        results=[json.loads(line.split(' ',1)[1]) for line in raw.splitlines() if line.startswith('LEVO_OFFLINE_REPORT ')]
        assert len(results)==1;result=results[0]
        assert result['probeCompleted'] and result['repeatCount']==8
        expected='high' if '-high-' in run else 'default';assert result['performanceHint']==expected
        assert all('Current State: AWAKE' in text(R/(run+suffix+'.log')) for suffix in ('-before','-after'))
        pure=run.startswith('vae-pureconv-');assert len(result['cases'])==(4 if pure else 1)
        for case in result['cases']:
            npu=case['npu'];assert npu['success'] and npu['npuOnlyBuildAndRunSucceeded']
            assert npu['buildCode']==0 and npu['npuOnlyOptionCode']==0 and npu['disableFallbackCode']==0
            assert not npu['cpuFallbackAllowed']
            if expected=='high':assert npu['performanceOptionCode']==0
            row={'run':run,'id':case['id'],'performanceHint':expected,'ablation':pure}
            for backend in ('npu','cpuFp32LowNeon','cpuW8LowNeon'):
                item=case[backend];assert item['success'] and item['numeric']['finite']
                times=item['runIncludingCopiesSeconds'];assert len(times)==8
                if pure and backend!='npu':assert item['vectorSinKernelCalls']==0
                if not pure and backend=='cpuFp32LowNeon':assert item['vectorSinKernelCalls']>0
                row[backend]={'firstMs':times[0]*1000,'warmMedianMs':statistics.median(times[1:])*1000,
                    'warmMinMs':min(times[1:])*1000,'warmMaxMs':max(times[1:])*1000,
                    'relativeL2':item['numeric']['relativeL2']}
            row['npuSpeedupVsCpuW8']=row['cpuW8LowNeon']['warmMedianMs']/row['npu']['warmMedianMs']
            row['npuRunSyncWarmMedianMs']=statistics.median(npu['runSyncSeconds'][1:])*1000
            summary.append(row)
        records.append({'run':run,'result':result,'sourceLogSha256':sha(R/(run+'.log'))})
    counter_checks=[]
    for mode in ('default','high'):
        run='vae-residual-countercheck-'+mode;raw=text(R/(run+'.log'))
        assert re.findall(r'^LEVO_PROBE_EXIT (\d+)\s*$',raw,re.M)==['0']
        result=[json.loads(line.split(' ',1)[1]) for line in raw.splitlines() if line.startswith('LEVO_OFFLINE_REPORT ')][0]
        assert result['probeCompleted'] and result['repeatCount']==8
        assert all('Current State: AWAKE' in text(R/(run+s+'.log')) for s in ('-before','-after'))
        case=result['cases'][0];assert case['npu']['success'] and not case['npu']['cpuFallbackAllowed']
        for backend in ('cpuFp32LowNeon','cpuW8LowNeon'):
            assert case[backend]['success'] and case[backend]['vectorSinKernelCalls']==288
        counter_checks.append({'run':run,'result':result,'sourceLogSha256':sha(R/(run+'.log'))})
    assert sha(ROOT/'native-build/levo_vae_npu_offline') in text(R/'vae-countercheck-probe-sha.log')
    path=R/'levo-test-report.json';report=json.loads(path.read_text(encoding='utf8'))
    report['vae_pureconv_ablation_2026_10_30']={
        'asOfUtc':datetime.now(timezone.utc).isoformat(),'status':'pure-convolution-npu-speedup-observed-not-end-to-end-vae',
        'humanAuthorization':'Continue pure-Conv comparison after ordinary quota window reset',
        'authorizedWindowResetEpoch':1793359789,'startingUsedPercent':0,'stopThresholdUsedPercent':95,
        'manifest':manifest,'conversion':conversion,'summary':summary,'records':records,
        'verifiedPhoneAndHostArtifactsAfterRuns':assets,
        'verifiedPureConvProbeBinarySha256':sha(before_binary),
        'pureConvProbeSourceAtRunSha256':sha(R/'VaeNpuOffline-pureconv-before-counter-fix.cpp.log'),
        'counterChecks':counter_checks,
        'counterFix':{'description':'Selector Scope resets the global counter per session; subtracting its previous value misreported the second CPU session as zero. Record the post-reset absolute count instead.',
            'changesArithmeticOrTimingRegion':False,'freshResidualCpuCallCounts':[288,288],
            'verifiedBinarySha256':sha(ROOT/'native-build/levo_vae_npu_offline'),
            'sourceSha256':sha(ROOT/'native-bench/VaeNpuOffline.cpp')},
        'preparationSourceSha256':sha(ROOT/'tools/prepare_vae_pureconv.py'),
        'appChanged':False,'recommendedAppPolicy':'chunks-5-w8-vsin','newFullVaeRtf':None,
        'interpretation':'Pure Conv can benefit from NPU, while matched full residual remains slower; this motivates diagnosing non-convolution lowering and boundaries rather than blaming the Conv hardware',
        'limits':['Performance ablation changes the function: removes Snake and residuals, so it is not a replacement decoder or a music-quality test',
            'Real same weights/shapes/dilations and identical synthetic external input; intermediate activations differ after removing nonlinearities/residuals',
            'Compiler may fuse or change schedules for a linear Conv chain; single-Conv cases avoid attributing the entire chain benefit to a specific unfused kernel',
            'CPU W8 and NPU FP16 have different numerical/quantization recipes; compare error as well as latency',
            'One first run plus seven warmed runs; two pure-Conv passes per hint, one matched full-residual pass; fixed order, no controlled thermal/frequency measurements',
            'HIGH API accepted; actual physical frequency and per-node AI-core/AICPU placement not profiled',
            'RunSync includes possible internal conversion/synchronization; process RSS is not the NPU private workspace peak',
            'Cannot subtract ablation times to assign all saved time to Sin; graph fusion and data distributions also changed',
            'No complete VAE/whole-chain NPU integration, no new App RTF or threefold decoder claim']}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps([s for s in summary if s['id'] in ('pureconv6-h4','residual-stack3-h4')]))

if __name__=='__main__':main()
