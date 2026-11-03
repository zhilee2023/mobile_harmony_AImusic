"""Merge hidden-interface phone probes into one publishable numerical summary.

Raw hidden/logit tensors remain local. Existing unified-report sections are
preserved; this tool refuses to replace an existing experiment key or output.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.deployment.files import read_json, write_json, sha256


def summarize(run_paths, projection_paths):
    import numpy as np
    first = Path(run_paths[0]).parent
    reference_logits = np.fromfile(first/'1-logits-t4/logits.f32', dtype=np.float32)
    reference_hidden = np.fromfile(first/'2-buffer-t4/hidden.f32', dtype=np.float32)
    if reference_logits.size != 256*16385 or reference_hidden.size != 256*1024:
        raise ValueError('Expected the fixed 256-frame reference suite')
    experiments = []
    for run_path in run_paths:
        run = read_json(run_path)
        if not run['complete'] or run['frames'] != 256:
            raise ValueError('Incomplete/other-size suites need a separate analysis')
        cases = []
        for case in run['cases']:
            if not case['displayStable'] or not case['result']['success']:
                raise ValueError('Do not rank failed or asleep cases as controlled results')
            folder = Path(run_path).parent/case['name']
            logits = np.fromfile(folder/'logits.f32', dtype=np.float32)
            equal = bool(np.array_equal(logits, reference_logits))
            hidden_equal = None
            hidden_path = folder/'hidden.f32'
            if case['result']['mode'] != 'logits' and not hidden_path.is_file():
                raise ValueError('A hidden-mode case is missing its numerical snapshot')
            if hidden_path.exists():
                hidden_equal = bool(np.array_equal(np.fromfile(hidden_path, dtype=np.float32), reference_hidden))
            if not equal or hidden_equal is False:
                raise ValueError('A speed candidate changed fixed-token numerical outputs')
            cases.append({**case, 'numericalValidation': {'logitsBitwiseEqualToReference': equal,
                          'hiddenBitwiseEqualToReference': hidden_equal}})
        experiments.append({'runId': run['runId'], 'suite': run.get('suite','outputs'),
                            'binarySha256': run['binarySha256'], 'cases': cases})

    projections = []
    for path in projection_paths:
        raw = read_json(path)
        if not raw['success']:
            raise ValueError('Failed projection probe')
        rows = []
        for row in raw['rows']:
            a = statistics.median(p['perFrameSeconds'] for p in row['pairs'])
            b = statistics.median(p['batchSeconds'] for p in row['pairs'])
            rows.append({**row, 'medianPerFrameSeconds': a, 'medianBatchSeconds': b,
                         'medianTimeRatio': a/b})
        projections.append({**raw, 'rows': rows})
    confirm = [x for x in experiments if x['suite']=='threads-confirm']
    comparison = None
    if confirm:
        cases = confirm[-1]['cases']
        def totals(threads):
            return [c['result']['rounds'][1]['prefillSeconds']+c['result']['rounds'][1]['decodeSeconds']
                    for c in cases if c['result']['requestedThreads']==threads]
        three, four = totals(3), totals(4)
        if len(three)!=2 or len(four)!=2:
            raise ValueError('Expected interleaved 3/4/3/4 fresh processes')
        comparison = {'scope':'Median warm prefill+255 decode forward time, excludes sampling and all downstream modules',
                      'threeThreadSeconds':three, 'fourThreadSeconds':four,
                      'medianReductionFraction':1-statistics.median(three)/statistics.median(four)}
    return {'scope':'Phone CLI interface/operator probes, not private-model music or App end-to-end RTF',
            'studentTrainingStatus':'Music adapters/head untrained; real pretrained/pruned 0.6B backbone',
            'runs':experiments, 'projectionFixtures':projections, 'threadComparison':comparison,
            'privateHiddenModelDeployed':False, 'wholeChainRtfMeasured':False,
            'limits':['No clock/affinity control; temperatures drift, use repeated adjacent cases',
                      'Hidden-enabled versus logits-only LM overhead is not isolated from observed timing drift',
                      'Projection weights are synthetic FP32 affine fixtures, not a trained private conditioner',
                      'CLI RSS includes numerical-reference buffers and is not App memory or dedicated VRAM',
                      'Fresh-process thread optimum may change with shared thread-pool initialization and other modules']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--projection',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--append-unified-key')
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve prior summaries')
    result=summarize(args.run,args.projection)
    result['sources']={'runJsonSha256':[sha256(x) for x in args.run],
                       'projectionJsonSha256':[sha256(x) for x in args.projection]}
    if args.append_unified_key:
        path=ROOT/'reports/levo-test-report.json'
        report=read_json(path)
        if args.append_unified_key in report:
            raise ValueError('Preserve prior unified-report sections')
        report[args.append_unified_key]=result
        write_json(path,report)
    write_json(args.out,result)
    print(json.dumps({'suites':len(result['runs']), 'projectionProbes':len(result['projectionFixtures']),
                      'threadComparison':result['threadComparison']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
