"""Export actual isolated candidates to the finite pool, keeping E2E validation separate."""
import argparse,hashlib,json,statistics,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'.operator-deps')]
from levo_harmony.operator_lab.pool import Pool,Spec,Outcome
from levo_harmony.operator_lab.solver import solve

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=json.loads(Path(a.campaign).read_text(encoding='utf-8'));stats={}
    if not r['trials'] or not all(t['screenAccepted'] for t in r['trials']):raise ValueError('Need awake, completed process trials')
    count=None
    for variant in r['variants']:
        trials=[t for t in r['trials'] if t['variant']==variant]
        if len(trials)!=3 or any(t['native']['status']!='completed' for t in trials):raise ValueError('Need3 independent processes per variant')
        stats[variant]={}
        for phase in ['prefill','decode']:
            rows=[next(x for x in t['native']['rows'] if x['phase']==phase) for t in trials]
            if any(x['relativeL2']>1e-5 for x in rows):raise ValueError('Numerical gate failed')
            medians=[x['warmMedianMs'] for x in rows]
            stats[variant][phase]=dict(medianMs=statistics.median(medians),processMediansMs=medians,processStdevMs=statistics.stdev(medians),relativeL2=max(x['relativeL2'] for x in rows))
            if phase=='prefill':
                if count is not None and count!=rows[0]['elements']:raise ValueError('Mixed input shapes')
                count=rows[0]['elements']
    names=['stock','tile256','tile1024','tile4096','tile8192']
    # Same logical loads/stores, different cache locality; do not invent DRAM savings.
    specs=[Spec(name,0,1,traffic_bytes=20*count,scratch_bytes=0,baseline=name=='stock') for name in names]
    outcomes={name:Outcome('passed',stats[name]['prefill']['medianMs'],stats[name]['prefill']['relativeL2'],'device_standalone') for name in names}
    signature=hashlib.sha256(json.dumps(dict(capture=r['captureSignature'],binaries=r['variants'],precision='high',threads=4,phase='prefill'),sort_keys=True).encode()).hexdigest()
    pool=Pool(1,specs,outcomes,signature,'device_standalone');oracle=solve(pool,outcomes,integer=True)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(schemaVersion=1,nodes=1,signature=signature,scope='device_standalone',candidates=[asdict(x) for x in specs],outcomes={k:asdict(v) for k,v in outcomes.items()},
        statistics=stats,oracle=dict(selected=oracle.selected,objectiveMs=oracle.objective_ms,optimal=oracle.optimal),
        limitations=['One existing fused operator; not a multi-operator fusion plan or evidence for CG advantage','Cost is median of3 process medians; statistical uncertainty remains',
                    'traffic_bytes estimates boundary/intermediate logical reads/writes; not measured DRAM','scratch_bytes is extra candidate allocation only, not total MNN working set',
                    'Tile0 is an identity recompilation control and excluded from optimized candidates','Decode path for large tiles falls back; apparent differences may be noise',
                    'Whole-LLM validation must gate deployment; isolated winner is not automatically promoted'])
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result['oracle']))
if __name__=='__main__':main()
