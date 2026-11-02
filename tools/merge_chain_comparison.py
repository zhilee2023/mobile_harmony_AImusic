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
    parser.add_argument('--duration',type=int,choices=[30,90,120],default=30)
    parser.add_argument('--clocks',type=Path)
    parser.add_argument('--failed-attempt',type=Path)
    parser.add_argument('--failed-low-attempt',type=Path)
    parser.add_argument('--allow-instrumentation-build-change',action='store_true')
    args=parser.parse_args()
    duration=args.duration
    frames=duration*25
    chunks=1+math.ceil(max(0,frames-256)/192)
    cases={name:json.loads(path.read_text(encoding='utf-8-sig')) for name,path in [('large',args.large),('small',args.small)]}
    package_hashes=set()
    for name,case in cases.items():
        result=case['result'];assert result['success'] and case['displayStable']
        assert result['ditSteps']==2 and result['durationSeconds']==duration
        assert result['quantization']['lmGuidance']==1 and result['quantization']['ditGuidance']==1.5
        assert result['codec']['ditSteps']==2 and len(result['codec']['chunks'])==chunks
        assert result['codec']['vaeNeonSinCalls']>0
        assert result['memory']['inferenceBackend']=='CPU'
        seconds=result['stageSeconds'];rtf=result['stageRtf']
        for part in ('llm','dit','vae','other','total'):
            assert math.isclose(seconds[part]/duration,rtf[part],abs_tol=1e-6)
        assert math.isclose(sum(seconds[part] for part in ('llm','dit','vae','other')),seconds['total'],abs_tol=1e-5)
        result['memory']['pssMetric']='HiDebug NativeMemInfo.pss: Pss + SwapPss; not resident-only PSS'
        result['memory']['pssReference']=DOC
        result['memory']['queryOverheadInterpretation']='Cumulative background query duration, includes waiting; already present during the benchmark, do not subtract from wall time'
        package_hashes.add(case['hapSha256'])
        if duration>30:
            assert case['foregroundStable'], 'A long-duration baseline requires foreground monitoring, not just screen-awake checks'
            assert result['requestedMusicCodes']==frames and result['fixedLengthBenchmark']
            assert result['lm']['frames']==frames
            assert result['memory']['independentRssSampler']
            assert result['lm']['eosSuppressedForFixedLengthDiagnostic']
            assert result['lm'].get('conditionedKvPositions',result['lm'].get('finalKvPositions'))==result['lm']['prefixPositions']+frames-1
    assert len(package_hashes)==1 or args.allow_instrumentation_build_change
    assert cases['small']['result']['diagnosticOnly'] and not cases['small']['result']['musicTrained']
    assert not cases['small']['result']['codec']['audioFileGenerated']
    assert cases['large']['result']['codec']['trainedLmTokens']
    record={
        'date':'2026-11-02','execution':'Signed LeVo Harmony App, separate process per case',
        'sameSignedPackage':len(package_hashes)==1,
        'mnnStaticLibrarySha256':'fbaa87b8cae9cd451560803ee3837886fe7e44d48976fc12654654412226159c',
        'sharedProtocol':{'effectiveSeconds':duration,'generatedCodes':frames,'frames':256,'overlapFrames':64,'hopFrames':192,'chunks':chunks,
                          'ditSteps':2,'lmGuidance':1,'ditGuidance':1.5,'lmThreads':4,'codecThreads':6,
                          'fixedLengthBenchmark':duration==120,
                          'lifetime':f'Finish and destroy LM, then reuse one DiT and one VAE session for{chunks}chunks'},
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
    if duration>30:
        if args.clocks:
            observations=[json.loads(line) for line in args.clocks.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            record['publicCpuClockObservations']={'samples':observations,'method':'Read-only HDC sysfs snapshots every20seconds, starting part-way through public LM; not sampled maxima or a controlled thermal ablation'}
        record['recipe']={
            'llm':{'weightBody':'W4 / group64','outputHead':'W8','inputEmbedding':'BF16','threads':4,
                   'largePrecision':cases['large']['result']['quantization']['lmPrecision'],'smallPrecision':'Low','guidance':1,'attentionMode':8,
                   'cache':f'Incremental KV retained within{frames}-code sequence; no requested KV4; exact native storage type not independently audited',
                   'sampling':{'seed':2307,'topK':50,'temperature':.9,'recentWindow':50,'repetitionPenalty':1.1,'eosSuppressed':True},
                   'staticA4':False,'awqOrQdit':False},
            'dit':{'weights':'W8 / group64','precision':'High','threads':6,'stepsPerWindow':2,'guidance':1.5,
                   'large':'39layers / hidden1200 / FFN4800, trained',
                   'small':'22layers / hidden1024 / FFN2048,195416128parameters, untrained',
                   'windows':chunks,'framesPerWindow':256,'overlapFrames':64,'hopFrames':192},
            'vae':{'weights':'W8','precision':'Low','threads':6,'neonSinBuildHook':True,
                   'sharedPretrainedMixedVae':True,'sampleRate':48000,'channels':2,'framesPerCall':256},
            'memoryAndLifetime':{'memoryMode':'Low','powerMode':'High','serialDestroyLlmBeforeCodec':True,
                                 'codecSessionsReused':True,'llmMaxProcessRssGiB':7,'llmMinimumSystemAvailableMiB':1200,
                                 'gpuOrNpuInference':False},
            'weightPayloadBytesOnly':{'largeLlm':1650155246,'smallLlm':248296682,
                                      'largeDit':722214236,'smallDit':221684024,'sharedVae':84569334,
                                      'meaning':'External .weight files from device stat; excludes graph, embeddings, conditions, runtime packing, activations and KV'},
        }
        record['limitations'] += [
            f'Both LM samplers suppress EOS to run exactly{frames}codes. This is a{duration}-second latency/memory workload, not proof of natural song ending or quality.',
            'Independent20ms RSS sampler; HiDebug query thread waits500ms after each query. Cross-stage queries excluded from per-stage attribution but retained in global peaks.',
            'Prior30-second measurements use an earlier profiler. Speed/amortization ratios are descriptive single-run comparisons, not a controlled duration-only ablation.',
        ]
        if len(package_hashes)!=1:
            record['limitations'].append('Small run usedv0.1.9; public retry usedv0.1.10with lifecycle logging/checkpoints and a longer diagnostic deadline. Same pinned MNN library and Small routine, but not the identical signed HAP. Actual precision remains in each recipe. Hashes retained; not a controlled model-size-only comparison.')
        if args.failed_attempt:
            record['priorFailedPublicHighAttempt']=json.loads(args.failed_attempt.read_text(encoding='utf-8-sig'))
        if args.failed_low_attempt:
            record['priorFailedPublicLowAttempt']=json.loads(args.failed_low_attempt.read_text(encoding='utf-8-sig'))
    prior_path=ROOT/'reports/music-app-first-song.json'
    if duration==30 and prior_path.is_file():
        prior=json.loads(prior_path.read_text(encoding='utf-8'))
        same_codes=cases['large']['result']['lm']['codes']==prior['lm']['codes']
        record['priorMusicApp4StepReference']={
            'ditSteps':4,'effectiveSeconds':prior['durationSeconds'],'totalSeconds':prior['totalSeconds'],'rtf':prior['wholePhoneRtf'],
            'sameGeneratedCodes':same_codes,'sameLyricsAndStyle':cases['large']['result']['condition']['lyrics']==prior['condition']['lyrics'] and cases['large']['result']['condition']['style']==prior['condition']['style'],
            'controlledTimingComparison':False,
        }
    path=ROOT/'reports/levo-test-report.json'
    unified=json.loads(path.read_text(encoding='utf-8'))
    if duration>30 and 'serial_chain_comparison_2026_11_02' in unified:
        previous=unified['serial_chain_comparison_2026_11_02']['cases']
        comparison={}
        for name,case in cases.items():
            now=case['result'];old=previous[name]['result']
            comparison[name]={'prior30sRtf':old['stageRtf']['total'],f'current{duration}sRtf':now['stageRtf']['total'],
                              'prior30sDecodeTokensPerSecond':old['lm']['decodeTokensPerSecond'],f'current{duration}sDecodeTokensPerSecond':now['lm']['decodeTokensPerSecond'],
                              'controlledDurationOnlyAblation':False}
            if name=='large':comparison[name]['first750CodesMatch30s']=now['lm']['codes'][:750]==old['lm']['codes']
        record['descriptiveVsPrior30s']=comparison
    key=f'serial_chain_comparison_{duration}s_2026_11_02' if duration>30 else 'serial_chain_comparison_2026_11_02'
    unified[key]=record
    path.write_text(json.dumps(unified,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for name,case in cases.items():
        result=case['result'];memory=result['memory']
        rows.append({'variant':name,'seconds':result['stageSeconds'],'rtf':result['stageRtf'],
                     'peakRssMiB':memory['peakRssKiB']/1024,'peakHiDebugPssMiB':memory['sampledPeakPssKiB']/1024,
                     'graphicsApiPeakKiB':memory['sampledPeakGraphicsKiB']})
    print(json.dumps({'summary':rows,'reference':record.get('priorMusicApp4StepReference')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
