"""Six real Conv layers only: performance ablation, NOT equivalent VAE output."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
import onnx,onnxruntime as ort
from onnx import helper,TensorProto
root=Path(__file__).resolve().parents[1];src=root/'models/vae-npu-stack';out=root/'models/vae-npu-pureconv';out.mkdir(exist_ok=True)
source=src/'residual-stack3-h4-nchw.onnx';m=onnx.load(source)
original=[copy.deepcopy(n) for n in m.graph.node if n.op_type=='Conv'];assert len(original)==6
shape=[1,128,4,8320]
x=np.fromfile(src/'residual-stack3-h4-input.f32',dtype=np.float32).reshape(shape)
opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
selections=[('pureconv6-h4',original)]+[(f'pureconv-k7-d{d}-h4',[original[i]]) for d,i in ((1,0),(3,2),(9,4))]
cases=[]
for name,selected in selections:
    nodes=copy.deepcopy(selected);last='x'
    for i,n in enumerate(nodes):
        n.input[0]=last;n.output[0]='y' if i==len(nodes)-1 else f'conv_only_{i}';last=n.output[0]
    used={v for n in nodes for v in n.input[1:]};weights=[copy.deepcopy(t) for t in m.graph.initializer if t.name in used]
    g=helper.make_graph(nodes,'pureconv_ablation',[helper.make_tensor_value_info('x',TensorProto.FLOAT,shape)],[helper.make_tensor_value_info('y',TensorProto.FLOAT,shape)],weights)
    model=helper.make_model(g,opset_imports=list(m.opset_import));model.ir_version=m.ir_version;onnx.checker.check_model(model)
    p=out/(name+'-nchw.onnx');onnx.save(model,p)
    y=ort.InferenceSession(model.SerializeToString(),opts,providers=['CPUExecutionProvider']).run(None,{'x':x})[0];assert np.isfinite(y).all()
    x.tofile(out/(name+'-input.f32'));y.tofile(out/(name+'-reference.f32'))
    attributes=[{a.name:helper.get_attribute_value(a) for a in n.attribute} for n in nodes]
    cases.append({'id':name,'shape':shape,'operationCount':len(nodes),'operatorTypes':{'Conv':len(nodes)},'convAttributes':attributes,'modelSha256':hashlib.sha256(p.read_bytes()).hexdigest()})
record={'ablation':'Real Conv only with bias; Snake and residual removed; not output-equivalent to trained VAE',
        'sourceResidualGraphSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'candidateCases':cases}
(out/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record))
