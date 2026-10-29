"""Real terminal VAE convolution weights; synthetic bounded activation tiles.

These isolated cases test mobile CANN compatibility, not a full NPU decoder.
"""
from pathlib import Path
import json
import sys
import subprocess
import numpy as np
sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

root=Path(__file__).resolve().parents[1]
out=root/'models/vae-npu-probe';out.mkdir(exist_ok=True)
source=onnx.load(root/'models/vae-stream-256/vae-256-simplified.onnx')
constants={x.name:numpy_helper.to_array(x) for x in source.graph.initializer}
cases=[]
for operation in ('Conv','ConvTranspose'):
    nodes=[node for node in source.graph.node if node.op_type==operation and '/layers/layers.5/' in node.name]
    if operation=='Conv':nodes=[n for n in nodes if len(constants[n.input[1]].shape)==3 and constants[n.input[1]].shape[2]==7]
    else:nodes=nodes[:1]
    for n,node in enumerate(nodes):
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        weight=constants[node.input[1]].astype('float16').astype('float32')
        bias=constants[node.input[2]].astype('float16').astype('float32')
        for length in (8192,32768):
            key=f'{operation.lower()}-{n}-{length}'
            value=(np.sin(np.arange(128*length,dtype=np.float32)*.013).reshape(1,128,length)*.2).astype('float16').astype('float32')
            graph=helper.make_graph([helper.make_node(operation,['x','w','b'],['y'],**attrs)],key,
                [helper.make_tensor_value_info('x',onnx.TensorProto.FLOAT,list(value.shape))],
                [helper.make_tensor_value_info('y',onnx.TensorProto.FLOAT,None)],
                [numpy_helper.from_array(weight,'w'),numpy_helper.from_array(bias,'b')])
            model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=8)
            onnx.save(model,out/f'{key}.onnx')
            subprocess.run([sys.executable,str(root/'tools/mnn_convert_once.py'),'--mnn-python-path',r'C:\ace-step-mnn-ios\.venv\Lib\site-packages','--','-f','ONNX','--modelFile',str(out/f'{key}.onnx'),'--MNNModel',str(out/f'{key}.mnn'),'--bizCode','vae-npu-matched-cpu'],check=True)
            session=ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider'])
            result=session.run(None,{'x':value})[0]
            for name,array in [('input',value),('weight',weight),('bias',bias),('reference',result)]:
                array.astype('<f4').tofile(out/f'{key}-{name}.f32')
            cases.append({'id':key,'sourceNode':node.name,'operation':operation,'inputShape':[1,128,1,length],
                'outputShape':[1,128,1,result.shape[-1]],'weightShape':[128,128,1,weight.shape[-1]],
                'kernel':weight.shape[-1],'dilation':attrs.get('dilations',[1])[0],
                'stride':attrs.get('strides',[1])[0],'padding':attrs.get('pads',[0])[0],
                'realWeights':True,'syntheticActivation':True})
manifest={'scope':'Real VAE terminal convolution weights, synthetic activation; isolated NPU compatibility and copy-inclusive latency only','cases':cases}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps(manifest,indent=2))
