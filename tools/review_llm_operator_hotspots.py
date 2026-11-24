"""Send only explicit operator aggregates to the locally configured Claude CLI."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from levo_harmony.operator_lab.reporting import operator_tables
from levo_harmony.operator_lab.claude import review

def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['haiku','sonnet','opus'],default='haiku');a=p.parse_args()
    context=[]
    for path in a.profiles:
        value=json.loads(Path(path).read_text(encoding='utf-8'));t=operator_tables(value)
        context.append(dict(model=value['native']['modelId'],phaseForwards=t['phaseForwards'],operatorTypes=t['types'],scope=t['scope']))
    out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    r=review(context,out,model=a.model);print(json.dumps(dict(requestedAlias=r['requestedAlias'],usage=r['cliResponse'].get('usage'),models=list(r['cliResponse'].get('modelUsage',{})))))
if __name__=='__main__':main()
