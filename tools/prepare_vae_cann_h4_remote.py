"""Pack independent calibrated rows into H4; kernel height=1 prevents mixing.

Twenty distinct latent-prefix inputs remain twenty, not eighty independent samples.
Pure Conv input includes its actual preceding Snake activation.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

base=Path(__file__).resolve().parent
manifest=json.loads((base/'manifest.json').read_text())
model=onnx.load(base/manifest['modelFile'])
conv=next(n for n in model.graph.node if n.op_type=='Conv')
before=[]
for node in model.graph.node:
    if node.name==conv.name:break
    before.append(copy.deepcopy(node))
def graph(nodes,shape,entry,exit_name):
    nodes=copy.deepcopy(nodes)
    for node in nodes:
        for i,name in enumerate(node.input):
            if name==entry:node.input[i]='x'
        for i,name in enumerate(node.output):
            if name==exit_name:node.output[i]='y'
    names={v for n in nodes for v in n.input}
    weights=[copy.deepcopy(t) for t in model.graph.initializer if t.name in names]
    g=helper.make_graph(nodes,'vae-cann-quant-h4',[helper.make_tensor_value_info('x',TensorProto.FLOAT,shape)],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,shape)],weights)
    m=helper.make_model(g,opset_imports=list(model.opset_import));m.ir_version=model.ir_version
    onnx.checker.check_model(m);return m
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
activation=ort.InferenceSession(graph(before,[1,128,1,8320],'x',conv.input[0]).SerializeToString(),opts,providers=['CPUExecutionProvider'])
cases=[('residual-d1-h4',graph(list(model.graph.node),[1,128,4,8320],'x','y'),False),
       ('pureconv-k7-d1-h4',graph([conv],[1,128,4,8320],conv.input[0],conv.output[0]),True)]
for identifier,candidate,pure in cases:
    folder=base/'cases'/identifier;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(identifier+'-nchw.onnx');onnx.save(candidate,path)
    records=[]
    for split in ('calibration','held-out'):
        original=[r for r in manifest['rows'] if r['split']==split]
        values=[np.fromfile(base/r['file'],dtype='<f4').reshape(1,128,1,8320) for r in original]
        if pure:values=[activation.run(None,{'x':x})[0] for x in values]
        for i,row in enumerate(original):
            x=np.concatenate([values[(i+k)%len(values)] for k in range(4)],axis=2)
            assert np.isfinite(x).all() and x.shape==(1,128,4,8320)
            dest=folder/row['file'];dest.parent.mkdir(exist_ok=True);x.astype('<f4').tofile(dest)
            records.append({**row,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size,
                'sourceRows':[original[(i+k)%len(values)]['file'] for k in range(4)],'precedingSnakeIncluded':pure})
    result={**manifest,'shape':[1,128,4,8320],'rows':records,'modelFile':path.name,
        'modelSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'ablation':'Single real K7 Conv only, preceded by actual Snake input; not full VAE' if pure else 'Original residual unchanged',
        'rowPacking':'4 independent H1 windows as H4; Conv height1, no cross-row convolution; 20 distinct calibration windows total'}
    (folder/'manifest.json').write_text(json.dumps(result,indent=2));print(identifier,result['modelSha256'],flush=True)
