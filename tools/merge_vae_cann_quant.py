"""Validate official calibration, compiler placement and actual phone artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';MODELS=ROOT/'models/vae-cann-quant'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def partition(path):
    text=path.read_text(errors='replace')
    found=re.findall(r'partition type NPU:(\d+), CPU:(\d+), GPU:(\d+), ISP:(\d+)',text)
    assert len(found)==1,(path,found)
    return dict(zip(('NPU','CPU','GPU','ISP'),map(int,found[0])))

def main():
    conversions=[]
    for identifier,folder in [('residual-d1-h1',MODELS),('residual-d1-h4',MODELS/'cases/residual-d1-h4'),
                              ('pureconv-k7-d1-h4',MODELS/'cases/pureconv-k7-d1-h4')]:
        result=load(folder/'compiled/conversion-result.json')
        quant=folder/result['quantDirectory'];log=(quant/'dopt.log').read_text(errors='replace')
        assert '20/20' in log and 'Quantize model success!' in log
        quant_result=load(quant/'result.json');assert quant_result['returnCode']==0
        for artifact in quant_result['artifacts']:
            assert sha(quant/artifact['file'])==artifact['sha256']
        for case in result['cases']:
            assert case['success'] and case['returnCode']==0
            assert sha(folder/'compiled'/(identifier+'-'+case['variant']+'.omc'))==case['sha256']
            case['compiledPartitions']=partition(folder/'compiled'/(case['variant']+'.log'))
        conversions.append({'id':identifier,'cases':result['cases'],'quantArtifacts':quant_result['artifacts'],
            'calibrationConsumedConfirmedFromProgress':True,'calibrationSamples':20,'calibrationLogSha256':sha(quant/'dopt.log'),
            'recipe':'Official DOPT Quant_INT8-8; weight AbsMax, input EmaMinMax; default non-aggressive',
            'quantizedConvolutions':2 if identifier.startswith('residual') else 1})
    records=[];summary=[]
    for path in sorted(R.glob('vae-cann-heldout-*.log'))+sorted(R.glob('vae-cann-*-round[12].log')):
        text=path.read_text(encoding='utf-8-sig',errors='replace')
        lines=[line for line in text.splitlines() if line.startswith('LEVO_OFFLINE_REPORT ')]
        assert len(lines)==1,path
        report=json.loads(lines[0].split(' ',1)[1]);assert report['probeCompleted']
        if 'round' in path.name:assert 'LEVO_PROBE_EXIT 0' in text
        for case in report['cases']:
            variant=case['npu']['offlineVariant']
            placement=next(c for c in conversions if c['id']==case['id'])['cases']
            case['compiledPartitions']=next(c['compiledPartitions'] for c in placement if c['variant']==variant)
            for backend in ('npu','cpuFp32LowNeon','cpuW8LowNeon'):
                row=case[backend];assert row['success'] and row['numeric']['finite']
                assert len(row['runIncludingCopiesSeconds'])==8
                row['warmMedianMs']=statistics.median(row['runIncludingCopiesSeconds'][1:])*1000
            assert case['npu']['npuOnlyOptionCode']==case['npu']['disableFallbackCode']==case['npu']['buildCode']==0
            summary.append({'log':path.name,'id':case['id'],'variant':variant,
                'npuWarmMs':case['npu']['warmMedianMs'],'cpuW8WarmMs':case['cpuW8LowNeon']['warmMedianMs'],
                'npuL2':case['npu']['numeric']['relativeL2'],'cpuW8L2':case['cpuW8LowNeon']['numeric']['relativeL2'],
                'compiledPartitions':case['compiledPartitions']})
        records.append({'log':path.name,'sha256':sha(path),'report':report})
    assert len(records)==8 and len(summary)==16
    assets=[]
    for line in (R/'vae-cann-phone-assets.log').read_text(encoding='utf-8-sig').splitlines():
        if not re.match(r'^[0-9a-f]{64}\s',line):continue
        digest,remote=line.split(maxsplit=1)
        if remote.endswith('/levo_vae_cann_quant_20261030'):local=ROOT/'native-build/levo_vae_npu_offline'
        else:
            assert remote.startswith('/data/local/tmp/vae-cann-')
            local=ROOT/'models'/remote.removeprefix('/data/local/tmp/')
        assert sha(local)==digest,local
        assets.append({'localFile':local.relative_to(ROOT).as_posix(),'sha256':digest})
    assert len(assets)==25,len(assets)
    history=(R/'vae-cann-historical-partitions.log').read_text(encoding='utf-8-sig')
    historical=[]
    for line in history.splitlines():
        match=re.search(r'/([^/]+)\.log:.*partition type NPU:(\d+), CPU:(\d+), GPU:(\d+), ISP:(\d+)',line)
        if match:historical.append({'id':match[1],'NPU':int(match[2]),'CPU':int(match[3]),'GPU':int(match[4]),'ISP':int(match[5])})
    assert len(historical)==8
    correction={'finding':'Runtime NPU-only order and disabled fallback do not remove CPU subgraphs already embedded in offline OMC',
        'singleResidualPartitions':{'NPU':3,'CPU':2},'tripleResidualPartitions':{'NPU':7,'CPU':6},
        'pureConvPartitions':{'NPU':1,'CPU':0},'historicalCompilerEvidence':historical,
        'evidenceSha256':sha(R/'vae-cann-historical-partitions.log'),
        'meaning':'Prior successful NPU-onlyBuildAndRun fields record option acceptance, not exclusive physical NPU execution'}
    report_path=R/'levo-test-report.json';report=load(report_path)
    report['vae_cann_quantization_2026_10_30']={
        'asOfUtc':datetime.now(timezone.utc).isoformat(),'status':'official-w8a8-calibrated-compiled-phone-tested-no-app-speedup-claimed',
        'officialTool':'DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_onnx_py3/dopt_so.py',
        'runtime':'Isolated CPython3.10.22; numpy1.23.5, ONNX1.14.1, ORT1.16.3, protobuf3.20.3',
        'calibration':load(ROOT/'models/vae-cann-calibration/manifest.json'),
        'h1PhoneFixture':load(ROOT/'models/vae-cann-phone/manifest.json'),
        'h4PhoneFixture':load(ROOT/'models/vae-cann-h4-phone/manifest.json'),
        'conversions':conversions,'summary':summary,'records':records,'verifiedPhoneAssets':assets,
        'probeSourceSha256':sha(ROOT/'native-bench/VaeNpuOffline.cpp'),'placementCorrection':correction,
        'appPathChanged':False,'fullVaeOrChainRtfMeasured':False,'recommendedAppPolicy':'chunks-5-w8-vsin',
        'limits':['20 calibration windows and four held-out windows come from the same synthetic latent stream, not real songs',
            'Actual pretrained VAE prefix generates inputs, but 16-frame resets and overlapping tiles limit deployment coverage',
            'H4 packs four independent H1 rows: 20 distinct calibration sources, not80 independent examples',
            'Residual OMC is statically mixed CPU/NPU; pure Conv compiler reports NPU1/CPU0, no hardware per-kernel trace',
            'FP32 public IO; W8A8 only on selected convolutions, not whole VAE or all operators',
            'Eight repeats separate first from seven warmed; H4 order reversed on second pass; no fixed-frequency or controlled thermal experiment',
            'Process RSS20ms does not measure NPU private workspace; no music-quality or end-to-end speed acceptance'],
        'setupIssuesResolved':['DDK archive contained DOPT but previous extraction had only OMG; separately extracted ONNX tool',
            'Python3.12 ABI incompatible; used isolated3.10 and installed missing tqdm',
            'Binary calibration needs official magic510 plus uint32 NCHW header; directory/raw concatenation rejected before calibration'],
        'officialDocumentation':'https://developer.huawei.com/consumer/en/doc/hiai-guides/overview-0000001053005655'}
    # Keep historical numbers, but explicitly correct their placement interpretation.
    for key in ('vae_npu_offline_performance_2026_10_30','vae_pureconv_ablation_2026_10_30'):
        report[key]['executionPlacementCorrection']=correction
    report['vae_cann_quantization_2026_10_30']['calibration']['quantizationExecuted']=True
    report['vae_cann_quantization_2026_10_30']['calibration']['npuExecutionValidated']=False
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
