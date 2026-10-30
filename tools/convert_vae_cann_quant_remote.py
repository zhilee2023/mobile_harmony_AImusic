"""Compile official DOPT artifacts and matched FP16 baseline on Kirin9030."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',required=True,type=int)
    parser.add_argument('--case',choices=('residual-d1-h4','pureconv-k7-d1-h4'));args=parser.parse_args()
    base=Path(__file__).resolve().parent;work=base/'cases'/args.case if args.case else base
    identifier=args.case or 'residual-d1-h1'
    folders=sorted(work.glob('quant-official-*'))
    valid=[p for p in folders if (p/'result.json').is_file() and json.loads((p/'result.json').read_text())['quantizationSucceeded']]
    if not valid:raise RuntimeError('No successful official DOPT calibration')
    quant=valid[-1];log=(quant/'dopt.log').read_text()
    assert '20/20' in log and 'Quantize model success!' in log
    toolkit=json.loads((Path.home()/'cosyvoice-omc/toolkit-status.json').read_text())
    output=work/'compiled';output.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(LD_LIBRARY_PATH=toolkit['libraryPath'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    rows=[]
    for variant,model in [('fp16',work/(identifier+'-nchw.onnx')),('w8a8',quant/'quant.onnx')]:
        if time.time()+80>=args.deadline:raise RuntimeError('Authorized window boundary')
        command=[toolkit['omg'],'--framework=5','--model='+str(model),
            '--output='+str(output/(identifier+'-'+variant)),'--target=omc','--platform=kirin9030',
            '--weight_data_type=FP16','--input_type=x:FP32','--output_type=y:FP32']
        if variant=='w8a8':command.append('--compress_conf='+str(quant/'compress'))
        def limit():
            resource.setrlimit(resource.RLIMIT_AS,(2200*2**20,2200*2**20));resource.setrlimit(resource.RLIMIT_CPU,(65,65))
        start=time.monotonic();row={'variant':variant,'command':command,'returnCode':None}
        with (output/(variant+'.log')).open('w') as stream:
            try:
                result=subprocess.run(command,cwd=Path(toolkit['omg']).parent,env=env,stdout=stream,
                    stderr=subprocess.STDOUT,timeout=75,preexec_fn=limit);row['returnCode']=result.returncode
            except subprocess.TimeoutExpired:row['error']='75 second timeout'
        path=output/(identifier+'-'+variant+'.omc');row['seconds']=time.monotonic()-start
        row['success']=row['returnCode']==0 and path.is_file()
        if path.is_file():row.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row);print(json.dumps(row),flush=True)
        print((output/(variant+'.log')).read_text(errors='replace')[-2800:],flush=True)
    result={'quantDirectory':quant.name,'calibrationConsumedConfirmed':True,'calibrationSamples':20,
        'quantResult':json.loads((quant/'result.json').read_text()),'cases':rows,'npuTested':False}
    (output/'conversion-result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
