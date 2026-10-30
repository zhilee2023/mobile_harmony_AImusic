"""Official DOPT Quant_INT8-8 runner, isolated Linux workspace only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import struct
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',required=True,type=int)
    parser.add_argument('--case',choices=('residual-d1-h4','pureconv-k7-d1-h4'))
    parser.add_argument('--input-mode',choices=('directory','concatenated','list','official'),default='official')
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    work=base/'cases'/args.case if args.case else base
    manifest=json.loads((work/'manifest.json').read_text())
    tool=base/'DDK-tools-next-6.1.1.0/tools/tools_dopt/dopt_onnx_py3'
    model=work/manifest['modelFile']
    assert hashlib.sha256(model.read_bytes()).hexdigest()==manifest['modelSha256']
    for row in manifest['rows']:
        assert hashlib.sha256((work/row['file']).read_bytes()).hexdigest()==row['sha256']
    assert manifest['calibrationSamples']>=20 and manifest['heldOutSamples']>=2
    if time.time()+155>=args.deadline:
        raise RuntimeError('Not enough time in authorized ordinary quota window')
    paths=[work/r['file'] for r in manifest['rows'] if r['split']=='calibration']
    calibration=work/'calibration'
    if args.input_mode in ('concatenated','official'):
        calibration=work/('calibration-official.bin' if args.input_mode=='official' else 'calibration-input.bin')
        with calibration.open('wb') as stream:
            if args.input_mode=='official':
                # Official DDK bin_data_preprocessing.py: magic 510, NCHW uint32.
                stream.write(struct.pack('<5I',510,len(paths),*manifest['shape'][1:]))
            for path in paths:stream.write(path.read_bytes())
    elif args.input_mode=='list':
        calibration=work/'calibration-input.txt'
        calibration.write_text(''.join(str(p)+'\n' for p in paths))
    output=work/('quant-'+args.input_mode+'-'+str(time.time_ns()));output.mkdir()
    config=output/'config.prototxt'
    config.write_text("strategy: 'Quant_INT8-8'\ndevice: USE_CPU\npreprocess_parameter: {\n input_type: BINARY\n input_file_path: '"+str(calibration)+"'\n}\n")
    command=[str(base/'env/bin/python'),str(tool/'dopt_so.py'),'--framework','5','--mode','0',
        '--model',str(model),'--cal_conf',str(config),'--output',str(output/'quant.onnx'),
        '--input_shape','x:'+','.join(map(str,manifest['shape'])),'--out_nodes','y',
        '--compress_conf',str(output/'compress')]
    env=os.environ.copy();env.update(PYTHONPATH=str(tool),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    def limit():
        resource.setrlimit(resource.RLIMIT_AS,(2500*2**20,2500*2**20))
        resource.setrlimit(resource.RLIMIT_CPU,(125,125))
    start=time.monotonic();row={'command':command,'inputMode':args.input_mode,'returnCode':None,
        'calibrationSamplesPrepared':len(paths),'calibrationConsumedConfirmed':False,
        'sourceModelSha256':manifest['modelSha256'],'quantizationSucceeded':False,'npuTested':False}
    with (output/'dopt.log').open('w') as log:
        try:
            result=subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=145,preexec_fn=limit);row['returnCode']=result.returncode
        except subprocess.TimeoutExpired:row['error']='145 second timeout'
    row['seconds']=time.monotonic()-start
    log_text=(output/'dopt.log').read_text(errors='replace')
    row['calibrationConsumedConfirmed']=row['returnCode']==0 and f'{len(paths)}/{len(paths)}' in log_text
    row['artifacts']=[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in output.iterdir() if p.is_file() and p.name not in ('dopt.log','result.json')]
    row['quantizationSucceeded']=row['returnCode']==0 and (output/'quant.onnx').is_file() and any(p.name.startswith('compress') for p in output.iterdir())
    (output/'result.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    print(log_text[-7000:])

if __name__=='__main__':main()
