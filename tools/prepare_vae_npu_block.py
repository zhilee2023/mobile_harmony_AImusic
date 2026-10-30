"""Prepare a real VAE residual block for NPU-only compilation, with CPU proofs.

Exports original weights into ignored local models/. No NPU speed claim is made.
Default: one d=1 block. --stack3 adds d=3/9; --compact changes tile geometry.
Independent-tile/interior proofs are not whole-decoder boundary validation.
"""
from collections import Counter
import copy
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.append(r"C:\ace-step-mnn-ios\.venv\Lib\site-packages")
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"models/vae-stream-256/vae-256-simplified.onnx"
OUT=ROOT/"models/vae-npu-block"
PREFIX="/layers/layers.5/layers/layers.2/"
CORE,HALO,C=8192,64,128

def digest(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()

def session(model):
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    return ort.InferenceSession(model.SerializeToString(),options,providers=["CPUExecutionProvider"])

def relative(actual,expected):
    assert np.isfinite(actual).all() and actual.shape==expected.shape
    return float(np.linalg.norm((actual-expected).astype(np.float64))/max(np.linalg.norm(expected.astype(np.float64)),1e-30))

def main():
    global OUT,CORE
    parser=argparse.ArgumentParser();parser.add_argument('--stack3',action='store_true');parser.add_argument('--compact',action='store_true');args=parser.parse_args()
    assert not args.compact or args.stack3
    if args.compact:CORE=512
    if args.stack3:OUT=ROOT/('models/vae-npu-stack-compact' if args.compact else 'models/vae-npu-stack')
    OUT.mkdir(parents=True,exist_ok=True)
    source=onnx.load(SOURCE)
    constants={tensor.name:tensor for tensor in source.graph.initializer}
    for node in source.graph.node:
        if node.op_type=="Constant":
            value=next(a.t for a in node.attribute if a.name=="value")
            tensor=copy.deepcopy(value);tensor.name=node.output[0];constants[tensor.name]=tensor
    prefixes=[f'/layers/layers.5/layers/layers.{i}/' for i in (2,3,4)] if args.stack3 else [PREFIX]
    nodes=[copy.deepcopy(node) for node in source.graph.node if any(node.name.startswith(p) for p in prefixes)]
    count=3 if args.stack3 else 1
    assert len(nodes)==13*count and Counter(n.op_type for n in nodes)==Counter({"Mul":4*count,"Sin":2*count,"Pow":2*count,"Add":3*count,"Conv":2*count})
    produced={name for node in nodes for name in node.output}
    external={name for node in nodes for name in node.input if name not in produced}
    runtime=external-constants.keys();assert len(runtime)==1
    entry=next(iter(runtime));exit_name=prefixes[-1]+"Add_output_0";assert exit_name in produced
    used={name:constants[name] for name in sorted(external&constants.keys())}

    def make(rows,width,rank4):
        copies=copy.deepcopy(nodes);initializers=[]
        for name,value in used.items():
            tensor=copy.deepcopy(value)
            if rank4 and len(tensor.dims)==3:
                array=numpy_helper.to_array(tensor)
                # Convolution weights [O,I,K] -> [O,I,1,K]; Snake [1,C,1] -> [1,C,1,1].
                array=np.expand_dims(array,2)
                tensor=numpy_helper.from_array(np.ascontiguousarray(array),name)
            initializers.append(tensor)
        for node in copies:
            for i,name in enumerate(node.input):
                if name==entry:node.input[i]="x"
            for i,name in enumerate(node.output):
                if name==exit_name:node.output[i]="y"
            if rank4 and node.op_type=="Conv":
                attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
                assert attrs["group"]==1 and attrs["strides"]==[1] and attrs["dilations"][0] in (1,3,9)
                attrs["kernel_shape"]=[1,*attrs["kernel_shape"]]
                attrs["strides"]=[1,*attrs["strides"]];attrs["dilations"]=[1,*attrs["dilations"]]
                attrs["pads"]=[0,attrs["pads"][0],0,attrs["pads"][1]]
                del node.attribute[:]
                node.attribute.extend(helper.make_attribute(key,val) for key,val in attrs.items())
        shape=[1,C,rows,width] if rank4 else [rows,C,width]
        graph=helper.make_graph(copies,"real_vae_residual_d1",[helper.make_tensor_value_info("x",TensorProto.FLOAT,shape)],
            [helper.make_tensor_value_info("y",TensorProto.FLOAT,shape)],initializers)
        result=helper.make_model(graph,opset_imports=list(source.opset_import),producer_name="levo-harmony-real-vae-npu-probe")
        result.ir_version=source.ir_version;onnx.checker.check_model(result)
        return result

    width=CORE+2*HALO;cases=[];rng=np.random.default_rng(2307)
    for rows in ((64,) if args.compact else ((4,) if args.stack3 else (1,4))):
        one=make(rows,width,False);two=make(rows,width,True)
        values=rng.normal(0,.5,(1,C,rows,width)).astype(np.float32)
        original_input=np.ascontiguousarray(values[0].transpose(1,0,2))
        expected=session(one).run(None,{"x":original_input})[0].transpose(1,0,2)[None]
        actual=session(two).run(None,{"x":values})[0]
        error=relative(actual,expected);assert error<1e-5
        name='residual-stack3-compact' if args.compact else f"residual-{'stack3' if args.stack3 else 'd1'}-h{rows}"
        path=OUT/(name+'-nchw.onnx');onnx.save(two,path)
        values.tofile(OUT/(name+'-input.f32'))
        actual.tofile(OUT/(name+'-reference.f32'))
        cases.append({"id":name,"shape":[1,C,rows,width],"modelBytes":path.stat().st_size,
            "modelSha256":digest(path),"rank4VsOriginalRank3Fp32L2":error,
            "fp16InputPlusOutputBytes":values.size*4,"fp32InputPlusOutputBytes":values.size*8,
            "coreValuesPerRow":CORE,"haloPerSide":HALO,"wholeDecoder":False,
            "npuCompiled":False,"npuOnlyExecutionValidated":False})

    # Separate boundary proof: each tile includes real neighbors, zero only at
    # the actual sequence ends. Conv(k7,d1)+pointwise Conv needs radius three.
    length=4*CORE;continuous=rng.normal(0,.5,(1,C,length)).astype(np.float32)
    original=session(make(1,length,False)).run(None,{"x":continuous})[0]
    padded=np.pad(continuous,((0,0),(0,0),(HALO,HALO)))
    tiles=np.stack([padded[:,:,index*CORE:index*CORE+width] for index in range(4)],axis=2).reshape(1,C,4,width)
    tiled=session(make(4,width,True)).run(None,{"x":tiles})[0]
    joined=np.concatenate([tiled[:,:,index,HALO:HALO+CORE] for index in range(4)],axis=2)
    seam_error=relative(joined,original)
    interior_error=relative(joined[:,:,HALO:-HALO],original[:,:,HALO:-HALO])
    assert interior_error<1e-5
    if not args.stack3:assert seam_error<1e-5
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(source.graph.value_info)+list(source.graph.input)+list(source.graph.output)}
    dense_macs=0;stages=Counter()
    for node in source.graph.node:
        if node.op_type not in ("Conv","ConvTranspose"):continue
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        weight=constants[node.input[1]];inp=shapes[node.input[0]];out=shapes[node.output[0]]
        macs=(int(np.prod(out))*int(weight.dims[1])*attrs["kernel_shape"][0] if node.op_type=="Conv"
              else int(np.prod(inp))*int(weight.dims[1])*attrs["kernel_shape"][0])
        dense_macs+=macs;stages[node.name.split("/")[2]]+=macs
    record={"asOfUtc":datetime.now(timezone.utc).isoformat(),"status":"real-residual-candidates-prepared-cpu-only",
        "sourceOnnxSha256":digest(SOURCE),"wholeDecoderOps":dict(Counter(n.op_type for n in source.graph.node)),
        "sourcePrefixes":prefixes,"blockOps":dict(Counter(n.op_type for n in nodes)),"candidateCases":cases,
        "continuousVsTiledFp32L2":seam_error,"continuousProofLength":length,"continuousProofTileRows":4,"interiorExcludingGlobalEdgesL2":interior_error,
        "globalSequenceEdgePaddingValidated":not args.stack3,"denseConvAndDeconvMacEstimate":dense_macs,
        "denseMacEstimateByStage":dict(stages),"macConvention":"One multiply-accumulate is one MAC; deconv ignores padded-edge elimination, not measured hardware FLOPs",
        "npuPerformanceMeasured":False,"appPathChangedToNpu":False,
        "limits":["Real original FP32 weights, deterministic synthetic activations; not calibrated INT8 or music-quality proof",
            "Rank lifting preserves kernel [1,K], dilation [1,d], padding [0,p,0,p] and broadcasts; no approximate Sin substitution",
            "Halo/tiling proof applies to this one d=1 block only; multiple residual blocks and ConvTranspose require separate boundary verification",
            "NPU-only compilation must disable fallback; checking public API availability is not evidence of actual NPU placement",
            "Compare combined host copy, precision conversion, device run and synchronization against the matched CPU path; do not extrapolate single-op speed"]}
    (OUT/"manifest.json").write_text(json.dumps(record,indent=2),encoding="utf8")
    report_path=ROOT/"reports/levo-test-report.json";report=json.loads(report_path.read_text(encoding="utf8"))
    key='vae_npu_compact_research_2026_10_30' if args.compact else ('vae_npu_stack_research_2026_10_30' if args.stack3 else 'vae_npu_research_2026_10_30')
    previous=report.get(key,{})
    if previous.get("sourceOnnxSha256")==record["sourceOnnxSha256"] and "onlineSupportProbe" in previous:
        record["onlineSupportProbe"]=previous["onlineSupportProbe"]
    report[key]=record
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"cases":cases,"tileSeamL2":seam_error,"npuMeasured":False}))

if __name__=="__main__":main()
