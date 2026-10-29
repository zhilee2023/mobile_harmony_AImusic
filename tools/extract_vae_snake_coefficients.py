"""Extract small immutable Snake constants before runtime broadcast expansion."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
import onnx
from onnx import numpy_helper

root=Path(__file__).resolve().parents[1]
source=root/'models/vae-stream-256/vae-256-simplified.onnx'
model=onnx.load(source)
constants={value.name:numpy_helper.to_array(value) for value in model.graph.initializer}
for node in model.graph.node:
    if node.op_type=='Constant':
        attribute=next(a for a in node.attribute if a.name=='value')
        constants[node.output[0]]=numpy_helper.to_array(attribute.t)
by_output={output:node for node in model.graph.node for output in node.output}
groups={}
for node in model.graph.node:
    if node.op_type!='Sin':continue
    prefix=node.name.rsplit('/',1)[0]
    multiply=by_output[node.input[0]]
    assert multiply.op_type=='Mul'
    alpha=next(constants[x] for x in multiply.input if x in constants)
    reciprocal=constants[prefix+'/Reciprocal_output_0']
    assert 1<=alpha.size<=2048 and reciprocal.size==alpha.size
    assert np.isfinite(alpha).all() and np.isfinite(reciprocal).all()
    groups[prefix]={'alpha':alpha.astype(np.float32).reshape(-1).tolist(),
                    'beta':reciprocal.astype(np.float32).reshape(-1).tolist()}
assert len(groups)==36
report={'sourceOnnxSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope':'Immutable original graph constants only; no runtime broadcast tensor copies','groups':groups}
target=source.parent/'snake-coefficients.json'
target.write_text(json.dumps(report,separators=(',',':')),encoding='utf8')
print(json.dumps({'groups':len(groups),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
