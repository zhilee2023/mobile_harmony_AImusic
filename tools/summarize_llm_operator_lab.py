import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from levo_harmony.operator_lab.reporting import baseline_summary,operator_tables,roofline_calibration

def main():
    p=argparse.ArgumentParser();p.add_argument('--baselines',nargs='*',default=[]);p.add_argument('--profiles',nargs='*',default=[])
    p.add_argument('--calibration');p.add_argument('--output',required=True);a=p.parse_args()
    result=dict(schemaVersion=1,baselines=[],operators=[])
    for path in a.baselines:
        r=json.loads(Path(path).read_text(encoding='utf-8'));result['baselines'].append(dict(model=r['native']['modelId'],**baseline_summary(r)))
    for path in a.profiles:
        r=json.loads(Path(path).read_text(encoding='utf-8'));result['operators'].append(dict(model=r['native']['modelId'],**operator_tables(r)))
    if a.calibration:
        r=json.loads(Path(a.calibration).read_text(encoding='utf-8'));result['roofline']=roofline_calibration(r);result['standaloneProbes']=r['native']
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('Unified baseline, per-node callbacks and independent probe report written')
if __name__=='__main__':main()
