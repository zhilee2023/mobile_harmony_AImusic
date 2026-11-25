import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from levo_harmony.operator_lab.device import Hdc
from levo_harmony.operator_lab.runner import run_device

def main():
    p=argparse.ArgumentParser(description='Independent baseline driver; fetches one unified report')
    p.add_argument('--device',required=True);p.add_argument('--tag',required=True);p.add_argument('--model',choices=['small','large'],default='small')
    p.add_argument('--transport',choices=['app','cli'],default='app');p.add_argument('--root');p.add_argument('--mode',choices=['baseline','operators'],default='baseline')
    p.add_argument('--steps',type=int,default=64);p.add_argument('--rounds',type=int,default=3);p.add_argument('--threads',type=int,default=4);p.add_argument('--precision',choices=['high','low'],default='high')
    p.add_argument('--memory',choices=['low','normal'],default='low',help='Normal expands the same small-model weights on CLI')
    p.add_argument('--cooling-seconds',type=int,default=0);p.add_argument('--max-start-battery-c',type=float);p.add_argument('--reference');p.add_argument('--epsilon',type=float,default=1e-3)
    p.add_argument('--capture-mlp',action='store_true')
    p.add_argument('--hdc',default=r'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe');args=p.parse_args()
    if args.cooling_seconds<0 or args.epsilon<0:p.error('Negative cooling/error budget')
    if args.capture_mlp and (args.mode!='operators' or args.transport!='cli'):p.error('Private MLP capture requires CLI operator diagnostic')
    if args.memory=='normal' and (args.transport!='cli' or args.model!='small' or args.precision!='high'):
        p.error('Normal memory reference requires the small CLI case and CPU High')
    root=args.root or '/data/app/el2/100/base/com.zhilee.llmoperatorlab/haps/entry/files'
    options={'mode':args.mode,'decodeSteps':2 if args.mode=='operators' else args.steps,'rounds':1 if args.mode=='operators' else args.rounds,'threads':args.threads,'precision':args.precision,'runTag':args.tag,'maxRssMiB':6144,'minAvailableMiB':650}
    if args.memory=='normal':options.update(memory='normal',maxRssMiB=3650,minAvailableMiB=820)
    if args.capture_mlp:options['captureMlp']=True
    report=run_device(Hdc(args.hdc,args.device),ROOT,root,int(args.model=='large'),options,ROOT/'reports/local'/args.tag,transport=args.transport,binary=ROOT/'native-build/llm_operator_baseline',cooling_seconds=args.cooling_seconds,max_start_battery_c=args.max_start_battery_c,reference=args.reference,epsilon=args.epsilon)
    print(json.dumps({'status':report['status'],'controlledAppTimingAccepted':report['controlledAppTimingAccepted'],'rssPeakMiB':report['native']['rssPeakMiB'],'rounds':[{k:v for k,v in r.items() if k!='decodeTokenMs'} for r in report['native']['rounds']]},indent=2))
    return 0 if report['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
