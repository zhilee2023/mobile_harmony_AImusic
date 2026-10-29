"""Existing MNN post-training weight quantization on the validated real block.

No calibration/audio quality claims. CPU timings are single-block PC timings.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--mnn-python-path',required=True);args=p.parse_args()
    sys.path.append(args.mnn_python_path)
    import MNN,_tools,numpy as np
    root=ROOT/'models/dit-block';data=np.load(root/'fixture.npz');ref=data['reference']
    profiles=[('w8-g16',8,16,False,False),('w8-g48',8,48,False,False),
              ('w4-g16',4,16,False,False),('w4-g48',4,48,False,False),
              ('w4-affine-g16',4,16,True,False),('w4-hqq-g16',4,16,True,True)]
    rows=[]
    for label,bits,group,affine,hqq in profiles:
        target=root/(label+'.mnn');tick=time.perf_counter()
        command=['MNNConvert','-f','ONNX','--modelFile',str(root/'real-block0-simplified.onnx'),
                 '--MNNModel',str(target),'--saveExternalData','--weightQuantBits',str(bits),'--weightQuantBlock',str(group)]
        if affine:command+=['--weightQuantAsymmetric=true']
        if hqq:command+=['--hqq']
        result=subprocess.run([sys.executable,str(ROOT/'tools/mnn_convert_once.py'),'--mnn-python-path',args.mnn_python_path,'--']+command[1:],
                              capture_output=True,text=True,encoding='utf8',errors='replace')
        print(result.stdout,flush=True)
        if result.returncode!=0:raise RuntimeError('Native MNN converter failed: '+str(result.returncode))
        if hqq and 'Disable Hqq' in result.stdout:raise RuntimeError('HQQ was requested but converter disabled it')
        if not target.exists():raise RuntimeError('Converter output missing')
        convert_seconds=time.perf_counter()-tick
        net=MNN.Interpreter(str(target));session=net.createSession({'backend':'CPU','numThread':4,'precision':1,'memory':2})
        timings=[]
        for i in range(6):
            begin=time.perf_counter()
            for name in ('x','modulation','mask'):
                val=np.ascontiguousarray(data[name],dtype=np.float32)
                if not net.getSessionInput(session,name).copyFrom(MNN.Tensor(tuple(val.shape),MNN.Halide_Type_Float,val,MNN.Tensor_DimensionType_Caffe)):
                    raise RuntimeError('Input copy failed')
            code=net.runSession(session)
            if code not in (0,None):raise RuntimeError('MNN forward failed')
            out=np.asarray(net.getSessionOutput(session,'output').getNumpyData()).copy()
            timings.append(time.perf_counter()-begin)
        if not np.isfinite(out).all():raise RuntimeError('Non-finite output')
        row={'profile':label,'bits_requested':bits,'group_requested':group,'asymmetric':affine,'hqq':hqq,
             'hqq_disabled_by_converter':False,'conversion_isolated_process':True,
             'weight_bytes':Path(str(target)+'.weight').stat().st_size,'graph_bytes':target.stat().st_size,
             'relative_l2':float(np.linalg.norm((out-ref).astype(np.float64))/np.linalg.norm(ref.astype(np.float64))),
             'warm_median_seconds':statistics.median(timings[1:]),'convert_seconds':convert_seconds,'finite':True}
        rows.append(row);print(json.dumps(row),flush=True)
        report={'scope':'actual pretrained block0, fixed32frames, PC CPU; not whole-DiT or phone RTF',
                'algorithm':'MNN existing weight-only quantization; no activation calibration',
                'source_width':1200,'note':'64 does not divide1200;16/48 do, avoiding fallback for these projections',
                'rows':rows,'real_song_generated':False}
        (ROOT/'reports/dit-block-quant-recipes.json').write_text(json.dumps(report,indent=2),encoding='utf8')


if __name__=='__main__':main()
