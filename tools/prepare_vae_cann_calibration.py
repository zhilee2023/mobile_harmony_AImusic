"""Capture real VAE prefix activations for official DOPT calibration.

The latent stream is synthetic diagnostic data, not a music-quality corpus.
No quantized model or NPU acceleration is claimed by this preparation step.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--held-out', type=int, default=4)
    args = parser.parse_args()
    if args.samples < 20 or args.samples + args.held_out > 32 or args.held_out < 2:
        raise ValueError('Use >=20 calibration and >=2 held-out samples, <=32 total')
    folder = ROOT/'models/vae-cann-calibration'
    folder.mkdir(exist_ok=True)
    baseline = ROOT/'models/vae-stream-256'
    fixture = json.loads((baseline/'fixture.json').read_text())
    parts = []
    for i in range(3):
        path = baseline/f'latent-{i}.f32'
        assert digest(path) == fixture['files'][path.name]['sha256']
        parts.append(np.fromfile(path, dtype='<f4').reshape(1,64,256))
    for i in (0,1):
        np.testing.assert_array_equal(parts[i][...,-64:], parts[i+1][...,:64])
    stream = np.concatenate([parts[0][...,:192],parts[1][...,:192],parts[2]], axis=-1)
    source = baseline/'vae-256-simplified.onnx'
    model = onnx.load(source)
    prefix = '/layers/layers.5/layers/layers.2/'
    first = next(n for n in model.graph.node if n.name.startswith(prefix))
    entry = first.input[0]
    # Recursively retain the original prefix, including its actual trained weights.
    required = {entry}; nodes = []
    for node in reversed(model.graph.node):
        if required.intersection(node.output):
            nodes.append(copy.deepcopy(node)); required.update(node.input)
    nodes.reverse()
    weights = [copy.deepcopy(t) for t in model.graph.initializer if t.name in required]
    inp = copy.deepcopy(model.graph.input[0]); inp.type.tensor_type.shape.dim[2].dim_value=16
    graph = helper.make_graph(nodes,'original_vae_prefix',[inp],
        [helper.make_tensor_value_info(entry,TensorProto.FLOAT,[1,128,30720])],weights)
    extracted=helper.make_model(graph,opset_imports=list(model.opset_import));extracted.ir_version=model.ir_version
    onnx.checker.check_model(extracted)
    path=folder/'prefix-fp32.onnx';onnx.save(extracted,path)
    del model, graph, weights, nodes
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    rows=[];start=time.perf_counter()
    for i in range(args.samples+args.held_out):
        # Non-overlapping latent windows. Prefix reset edges are excluded from tiles.
        x=np.ascontiguousarray(stream[...,i*16:(i+1)*16])
        tick=time.perf_counter();activation=session.run(None,{inp.name:x})[0]
        assert activation.shape==(1,128,30720) and np.isfinite(activation).all()
        positions=(2048,8192,14336,20480)
        tile=np.stack([activation[:,:,p:p+8320] for p in positions],axis=2)
        assert tile.shape==(1,128,4,8320)
        split='calibration' if i<args.samples else 'held-out'
        destination=folder/split;destination.mkdir(exist_ok=True)
        target=destination/f'activation-{i:02d}.f32';tile.astype('<f4').tofile(target)
        rows.append({'file':str(target.relative_to(folder)),'sha256':digest(target),'bytes':target.stat().st_size,
            'latentStart':i*16,'latentFrames':16,'split':split,'peak':float(np.abs(tile).max()),
            'rms':float(np.sqrt(np.mean(tile.astype(np.float64)**2))),'prefixSeconds':time.perf_counter()-tick})
        print(json.dumps(rows[-1]),flush=True)
    result={'scope':'Original FP32 VAE prefix activations from existing synthetic latent stream; no real music corpus',
        'calibrationSamples':args.samples,'heldOutSamples':args.held_out,'shape':[1,128,4,8320],
        'sourceOnnxSha256':digest(source),'prefixOnnxSha256':digest(path),'entry':entry,'rows':rows,
        'seconds':time.perf_counter()-start,'quantizationExecuted':False,'npuExecutionValidated':False,
        'limits':['Synthetic latent source, no trained 0.6B music head or real-song coverage',
                  '16-frame prefix resets and four overlapping interior feature tiles per sample',
                  'Held-out windows are disjoint latent windows from the same synthetic stream']}
    (folder/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
