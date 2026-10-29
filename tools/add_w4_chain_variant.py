"""Validate and add the complete W4/g48 DiT to the diagnostic bundle."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from prepare_three_module_bundle import sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
    import MNN
    source=ROOT/'models/full-dit-256'
    fixture=np.load(source/'fixture.npz')
    net=MNN.Interpreter(str(source/'full-w4.mnn'))
    begin=time.perf_counter()
    session=net.createSession({'backend':'CPU','numThread':4,'precision':1,'memory':2})
    setup=time.perf_counter()-begin
    for name in ('inputs','timestep','attention_mask'):
        value=np.ascontiguousarray(fixture[name])
        host=MNN.Tensor(tuple(value.shape),MNN.Halide_Type_Float,value,MNN.Tensor_DimensionType_Caffe)
        if not net.getSessionInput(session,name).copyFrom(host):raise ValueError('Input copy failed')
    begin=time.perf_counter();net.runSession(session)
    output=np.asarray(net.getSessionOutput(session,'velocity').getNumpyData()).copy()
    elapsed=time.perf_counter()-begin
    if output.shape!=(2,256,64) or not np.isfinite(output).all():raise ValueError('Invalid complete W4 output')
    error=float(np.linalg.norm((output-fixture['reference']).astype(np.float64))/np.linalg.norm(fixture['reference'].astype(np.float64)))
    output.astype('<f4').tofile(source/'w4-reference.f32')
    report={'complete_layers':39,'weight_bits':4,'requested_group_size':48,'finite':True,
            'pc_relative_l2_vs_pytorch':error,'setup_seconds':setup,'forward_seconds':elapsed,
            'weight_bytes':(source/'full-w4.mnn.weight').stat().st_size,
            'scope':'One complete CFM velocity evaluation, no listening or trained music-LM validation'}
    (ROOT/'reports/full-dit-w4.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    bundle=ROOT/'models/three-module-bundle'
    manifest=json.loads((bundle/'bundle.json').read_text())
    for name in ('full-w4.mnn','full-w4.mnn.weight','w4-reference.f32'):
        target=bundle/'dit'/name
        if not target.exists():os.link(source/name,target)
        if sha(target)!=sha(source/name):raise ValueError('Existing W4 variant differs')
        manifest['files']['dit/'+name]={'bytes':target.stat().st_size,'sha256':sha(target)}
    manifest['dit_variants']={'w8':{'requested_group_size':64},'w4':{'requested_group_size':48}}
    (bundle/'bundle.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
