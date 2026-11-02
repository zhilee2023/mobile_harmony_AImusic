"""Validate matching signed-App pipelines and merge into the single report."""
import argparse
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOC='https://github.com/openharmony/docs/blob/master/en/application-dev/reference/apis-performance-analysis-kit/js-apis-hidebug.md#nativememinfo12'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--large',type=Path,required=True)
    parser.add_argument('--small',type=Path,required=True)
    args=parser.parse_args()
    cases={name:json.loads(path.read_text(encoding='utf-8-sig')) for name,path in [('large',args.large),('small',args.small)]}
    package_hashes=set()
    for name,case in cases.items():
        result=case['result'];assert result['success'] and case['displayStable']
        assert result['ditSteps']==2 and result['durationSeconds']==30
        assert result['quantization']['lmGuidance']==1 and result['quantization']['ditGuidance']==1.5
        assert result['codec']['ditSteps']==2 and len(result['codec']['chunks'])==4
        assert result['codec']['vaeNeonSinCalls']>0
        assert result['memory']['inferenceBackend']=='CPU'
        seconds=result['stageSeconds'];rtf=result['stageRtf']
        for part in ('llm','dit','vae','other','total'):
            assert math.isclose(seconds[part]/30,rtf[part],abs_tol=1e-6)
        assert math.isclose(sum(seconds[part] for part in ('llm','dit','vae','other')),seconds['total'],abs_tol=1e-5)
        result['memory']['pssMetric']='HiDebug NativeMemInfo.pss: Pss + SwapPss; not resident-only PSS'
        result['memory']['pssReference']=DOC
        result['memory']['queryOverheadInterpretation']='Cumulative background query duration, includes waiting; already present during the benchmark, do not subtract from wall time'
        package_hashes.add(case['hapSha256'])
    assert len(package_hashes)==1
    assert cases['small']['result']['diagnosticOnly'] and not cases['small']['result']['musicTrained']
    assert not cases['small']['result']['codec']['audioFileGenerated']
    assert cases['large']['result']['codec']['trainedLmTokens']
    record={
        'date':'2026-11-02','execution':'Signed LeVo Harmony v0.1.8 App, separate process per case',
        'sharedProtocol':{'effectiveSeconds':30,'generatedCodes':750,'frames':256,'overlapFrames':64,'hopFrames':192,'chunks':4,
                          'ditSteps':2,'lmGuidance':1,'ditGuidance':1.5,'lmThreads':4,'codecThreads':6,
                          'lifetime':'Finish and destroy LM, then reuse one DiT and one VAE session for four chunks'},
        'models':{
            'large':{'lm':'Public trained primary Mixed LM, 36 layers / hidden2048; track LM omitted','dit':'39 layers / hidden1200 / FFN4800, trained W8','vae':'same pretrained Mixed VAE W8'},
            'small':{'lm':'26-layer Qwen3-0.6B backbone with untrained music adapters/head','dit':'22 layers / hidden1024 / FFN2048, 195416128 parameters, random CosyVoice-sized music prototype; not a CosyVoice speech checkpoint','vae':'same pretrained Mixed VAE W8'},
        },
        'cases':cases,
        'limitations':[
            'Small model results establish numerical flow and deployment cost, not music quality.',
            'Large prefix has953 positions; student prefix514. CPU precision High for public LM and Low for student: current validated recipes, not identical arithmetic precision.',
            'LM stage includes setup/prefill/decode/destruction; DiT and VAE columns cover forward/sync, other covers remaining setup, conditioning and output.',
            'RSS20ms, HiDebug Pss+SwapPss nominal500ms, graphics nominal1000ms; sampling queries run on a background thread and can perturb timings.',
            'Both graphics API measurements returned0 with success; CPU inference did not allocate model GPU buffers. This does not mean whole-device graphics usage is zero.',
            'Single runs; screen stability checked, temperature and OS file caches not controlled.',
        ],
    }
    prior_path=ROOT/'reports/music-app-first-song.json'
    if prior_path.is_file():
        prior=json.loads(prior_path.read_text(encoding='utf-8'))
        same_codes=cases['large']['result']['lm']['codes']==prior['lm']['codes']
        record['priorMusicApp4StepReference']={
            'ditSteps':4,'effectiveSeconds':prior['durationSeconds'],'totalSeconds':prior['totalSeconds'],'rtf':prior['wholePhoneRtf'],
            'sameGeneratedCodes':same_codes,'sameLyricsAndStyle':cases['large']['result']['condition']['lyrics']==prior['condition']['lyrics'] and cases['large']['result']['condition']['style']==prior['condition']['style'],
            'controlledTimingComparison':False,
        }
    path=ROOT/'reports/levo-test-report.json'
    unified=json.loads(path.read_text(encoding='utf-8'));unified['serial_chain_comparison_2026_11_02']=record
    path.write_text(json.dumps(unified,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for name,case in cases.items():
        result=case['result'];memory=result['memory']
        rows.append({'variant':name,'seconds':result['stageSeconds'],'rtf':result['stageRtf'],
                     'peakRssMiB':memory['peakRssKiB']/1024,'peakHiDebugPssMiB':memory['sampledPeakPssKiB']/1024,
                     'graphicsApiPeakKiB':memory['sampledPeakGraphicsKiB']})
    print(json.dumps({'summary':rows,'reference':record.get('priorMusicApp4StepReference')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
