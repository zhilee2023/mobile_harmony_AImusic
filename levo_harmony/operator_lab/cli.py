import argparse
import json
from pathlib import Path
from .pool import Pool
from .study import study
from .pipeline import Budget,execute_candidate

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as out:json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False)

def main(argv=None):
    parser=argparse.ArgumentParser(description='LLM operator finite-pool study and candidate regression chain')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pool-study');p.add_argument('--pool',required=True);p.add_argument('--output',required=True);p.add_argument('--budget',type=int,default=100)
    p.add_argument('--seeds',default='42,43,44');p.add_argument('--policies',default='random,greedy,frozen_dual,dual')
    p.add_argument('--traffic-budget',type=int);p.add_argument('--scratch-budget',type=int);p.add_argument('--calibration');p.add_argument('--allow-synthetic',action='store_true')
    p=sub.add_parser('candidate');p.add_argument('--task',required=True);p.add_argument('--commands',required=True);p.add_argument('--workspace',required=True);p.add_argument('--epsilon',type=float,default=1e-3)
    p.add_argument('--max-compiles',type=int,default=100);p.add_argument('--max-device-profiles',type=int,default=100);p.add_argument('--max-seconds',type=float,default=3600);p.add_argument('--max-agent-tokens',type=int)
    args=parser.parse_args(argv)
    if args.command=='pool-study':
        pool=Pool.read(args.pool)
        if pool.scope=='synthetic_test' and not args.allow_synthetic:parser.error('Synthetic fixtures need --allow-synthetic; they are never device performance results')
        calibration=json.loads(Path(args.calibration).read_text()) if args.calibration else None
        result=study(pool,args.policies.split(','),args.budget,[int(s) for s in args.seeds.split(',')],traffic_budget=args.traffic_budget,scratch_budget=args.scratch_budget,calibration=calibration)
        write_new(args.output,result);print(json.dumps(result['summary'],indent=2))
    else:
        budget=Budget(args.max_compiles,args.max_device_profiles,args.max_seconds,args.max_agent_tokens)
        result=execute_candidate(json.loads(Path(args.task).read_text()),json.loads(Path(args.commands).read_text()),args.workspace,budget,epsilon=args.epsilon)
        print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1
    return 0

if __name__=='__main__':raise SystemExit(main())
