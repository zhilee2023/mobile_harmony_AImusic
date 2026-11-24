"""ABBA whole-LLM validation after isolated selection; never promotes a candidate."""
import argparse,json,re,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from levo_harmony.operator_lab.device import Hdc
from levo_harmony.operator_lab.runner import run_device

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--root',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--tag',required=True);p.add_argument('--candidate',choices=['tile256','tile1024','tile4096','tile8192'],required=True)
    p.add_argument('--hdc',default=r'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe');a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',a.tag):p.error('Unsafe tag')
    work=ROOT/'reports/local'/a.tag;work.mkdir(exist_ok=False);hdc=Hdc(a.hdc,a.device)
    binaries={'stock':ROOT/'native-build/llm_operator_baseline',a.candidate:Path(a.campaign).resolve()/a.candidate/'build/llm_operator_baseline'}
    order=['stock',a.candidate,a.candidate,'stock'];trials=[]
    for index,name in enumerate(order):
        options=dict(mode='baseline',decodeSteps=64,rounds=3,threads=4,precision='high',runTag=f'{a.tag}-{index}-{name}',maxRssMiB=6144,minAvailableMiB=650,requireCandidateCalls=name!='stock')
        r=run_device(hdc,ROOT,a.root,0,options,work/f'{index}-{name}',transport='cli',binary=binaries[name],reference=Path(a.reference),epsilon=.001)
        if r['status']!='completed' or not r['regressionPassed'] or not r['screenAccepted']:raise RuntimeError('Numerical, execution or screen control failure')
        n=r['native'];warm=n['rounds'][1:]
        row=dict(index=index,variant=name,prefillMs=statistics.mean(x['prefillMs'] for x in warm),decodeMsPerToken=statistics.mean(x['decodeMsPerToken'] for x in warm),
            workloadMs=statistics.mean(x['prefillMs']+x['decodeMs'] for x in warm),rssPeakMiB=n['rssPeakMiB'],calls=n['candidateCalls'],regression=r['regression'],
            startBatteryC=r['controls'][0]['batteryC'],endBatteryC=r['controls'][-1]['batteryC'])
        trials.append(row);(work/'llm-operator-report.json').write_text(json.dumps(dict(scope='ABBA CLI fixed whole-LLM workload; not App or music speedup',order=order,trials=trials),indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
if __name__=='__main__':main()
