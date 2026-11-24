import argparse,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from levo_harmony.operator_lab.device import Hdc,latest_fixture,prepare_case

def main():
    parser=argparse.ArgumentParser(description='Prepare independent exact LM cases; preserve Stream App and bundles')
    parser.add_argument('--device',required=True);parser.add_argument('--root',required=True)
    parser.add_argument('--hdc',default=r'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe')
    parser.add_argument('--mnn',default=r'C:\cosyvoice-harmony\app\entry\.cxx\default\default\debug\arm64-v8a\mnn\libMNN.a')
    parser.add_argument('--model',choices=['small','large','both'],default='both');args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];transport=Hdc(args.hdc,args.device);digest=hashlib.sha256(Path(args.mnn).read_bytes()).hexdigest()
    models=[0,1] if args.model=='both' else [int(args.model=='large')]
    for model in models:
        manifest=prepare_case(transport,repo,args.root,model,latest_fixture(repo,model),mnn_sha256=digest)
        print(json.dumps({'case':manifest['modelId'],'signature':manifest['caseSignature'],'availableDecodeSteps':manifest['fixture']['availableDecodeSteps']},indent=2))

if __name__=='__main__':main()
