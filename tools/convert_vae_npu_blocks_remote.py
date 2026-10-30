"""Run on the existing Linux conversion host, in an isolated VAE work folder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--deadline",type=int,required=True);args=parser.parse_args()
    work=Path(__file__).resolve().parent;toolkit=json.loads((Path.home()/"cosyvoice-omc/toolkit-status.json").read_text())
    manifest=json.loads((work/"manifest.json").read_text());output=work/"outputs";output.mkdir(exist_ok=True)
    status={"scope":"Real VAE residual offline OMC conversion only; not NPU execution","cases":[]}
    env=os.environ.copy();env.update(LD_LIBRARY_PATH=toolkit["libraryPath"],OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    def save():(work/"conversion-result.json").write_text(json.dumps(status,indent=2))
    def limit():
        resource.setrlimit(resource.RLIMIT_AS,(2200*2**20,2200*2**20));resource.setrlimit(resource.RLIMIT_CPU,(65,65))
    for case in manifest["candidateCases"]:
        if time.time()+80>=args.deadline:status["stopReason"]="Authorized window boundary";save();return
        name=case["id"];assert name in ("residual-d1-h1","residual-d1-h4","residual-stack3-h4","residual-stack3-compact")
        source=work/(name+"-nchw.onnx");assert hashlib.sha256(source.read_bytes()).hexdigest()==case["modelSha256"]
        row={"id":name,"sourceSha256":case["modelSha256"],"weightType":"FP16","success":False};status["cases"].append(row);save()
        command=[toolkit["omg"],"--framework=5","--model="+str(source),"--output="+str(output/(name+"-fp16")),
            "--target=omc","--platform=kirin9030","--weight_data_type=FP16","--input_type=x:FP32","--output_type=y:FP32"]
        start=time.monotonic()
        with (output/(name+".log")).open("w") as log:
            try:
                completed=subprocess.run(command,cwd=Path(toolkit["omg"]).parent,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=75,preexec_fn=limit)
                row["returnCode"]=completed.returncode
            except subprocess.TimeoutExpired:row["error"]="75 second wall timeout"
        row["seconds"]=time.monotonic()-start;model=output/(name+"-fp16.omc")
        row["success"]=row.get("returnCode")==0 and model.is_file()
        if model.is_file():row.update(bytes=model.stat().st_size,sha256=hashlib.sha256(model.read_bytes()).hexdigest())
        save();print(json.dumps(row),flush=True)
    status["complete"]=True;save()

if __name__=="__main__":main()
