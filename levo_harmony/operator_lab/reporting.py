"""Keep graph callback proxies, standalone probes and LLM timing distinct."""
from collections import defaultdict
import math
import statistics


def baseline_summary(report):
    n=report['native']
    if n['status']!='completed' or n['instrumented'] or not report['controlledAppTimingAccepted']:
        raise ValueError('Need completed, uninstrumented, controlled App baseline')
    warm=n['rounds'][1:]
    if not warm:raise ValueError('At least one separate warm round is required')
    ms=statistics.mean(r['decodeMsPerToken'] for r in warm)
    return dict(profile=n['profile'],precision=n['precision'],threads=n['threads'],
        firstPrefillMs=n['rounds'][0]['prefillMs'],firstDecodeMsPerToken=n['rounds'][0]['decodeMsPerToken'],
        warmPrefillMs=statistics.mean(r['prefillMs'] for r in warm),warmDecodeMsPerToken=ms,warmDecodeTokensPerSecond=1000/ms,
        requestedDecodeSteps=n['requestedDecodeSteps'],kvPositions=[r['kvPositions'] for r in n['rounds']],
        loadMs=n['loadMs'],rssPeakMiB=n['rssPeakMiB'],
        modelRomMiB=sum(f['bytes'] for f in report['modelManifest']['files'] if f['relative'].startswith('lm/'))/1024**2,
        logitRepeatL2=max(r['logitL2VsFirst'] for r in warm),hiddenRepeatL2=max(r['hiddenL2VsFirst'] for r in warm),
        startBatteryC=report['controls'][0]['batteryC'],endBatteryC=report['controls'][-1]['batteryC'],
        hapSha256=report['hapSha256'],mnnArchiveSha256=n['mnnArchiveSha256'],
        scope=f"Fixed513-position real music prefix plus{n['requestedDecodeSteps']} teacher-code forwards; LLM-only, not music RTF")


def operator_tables(report):
    n=report['native']
    if n['status']!='completed' or not n['instrumented'] or not report['controlledAppTimingAccepted']:
        raise ValueError('Need completed separate instrumented App diagnostic')
    groups=defaultdict(lambda:dict(calls=0,totalMs=0.0))
    nodes=[]
    forwards={'prefill':len(n['rounds']),'decode':sum(r['decodeForwards'] for r in n['rounds'])}
    for x in n['operators']['nodes']:
        calls=x['calls'];ms=x['seconds']*1000
        if not isinstance(calls,int) or calls<=0 or not math.isfinite(ms) or ms<0:raise ValueError('Invalid callback observation')
        row=dict(phase=x['phase'],name=x['name'],operator=x['type'],calls=calls,totalMs=ms,meanCallMs=ms/calls,
                 inputShapes=x['inputShapes'],outputShapes=x['outputShapes'],logicalInputBytes=x['logicalInputBytes'],
                 logicalOutputBytes=x['logicalOutputBytes'],backendObservations=x['backendObservations'])
        nodes.append(row);g=groups[(x['phase'],x['type'])];g['calls']+=calls;g['totalMs']+=ms
    types=[dict(phase=p,operator=t,**v,meanCallMs=v['totalMs']/v['calls'],
                sumPerForwardMs=v['totalMs']/forwards[p]) for (p,t),v in groups.items()]
    return dict(types=sorted(types,key=lambda x:(x['phase'],-x['totalMs'])),
                nodes=sorted(nodes,key=lambda x:(x['phase'],-x['totalMs'])),phaseForwards=forwards,
                scope='Graph callback wall times; possibly nested. Not standalone kernel latency. Logical tensor bytes exclude weights/internal KV and are not measured DRAM.')


def roofline_calibration(report):
    n=report['native']
    if not n['success'] or not report['screenAccepted']:raise ValueError('Calibration did not pass or phone slept')
    mm=[r for r in n['rows'] if r['operator']=='MatMul']
    if not mm:raise ValueError('Missing compute calibration')
    compute=max(r['attainedFlopsPerSecond'] for r in mm)
    bandwidth=n['bandwidth']['nominalBytesPerSecond']
    if any(not math.isfinite(v) or v<=0 for v in [compute,bandwidth]):raise ValueError('Invalid calibration rates')
    return dict(flops_per_second=compute,bytes_per_second=bandwidth,threads=n['threads'],precision=n['precision'],
                scope='Attained FP32 MNN MatMul and nominal copy bandwidth; diagnostic reference, not chip peaks or hardware DRAM counters. Not directly calibrated for W4/W8 mixed-precision kernels.',
                strictLowerBoundClaimed=False,memoryBoundClassificationClaimed=False,
                points=[dict(model=r['modelShape'],phase=r['phaseShape'],arithmeticIntensity=r['flops']/r['logicalBytes'],
                             attainedGflops=r['attainedFlopsPerSecond']/1e9,medianMs=r['warmMedianMs']) for r in mm])
