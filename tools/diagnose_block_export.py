"""Localize any PyTorch->ONNX drift with explicit intermediate graph outputs."""
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from restore_mixed_dit import upstream_classes
from validate_real_dit_block import tensor_from_parts
from export_real_dit_block import real_freqs,real_rope


class ObservedBlock(nn.Module):
    def __init__(self,block):
        super().__init__();self.block=block;self.values={}
        self.names=['ln1','qkv','attn_projection','ln2','ffn']
        targets=[block.ln_1,block.attn.c_attn,block.attn.c_proj,block.ln_2,block.mlp]
        for name,module in zip(self.names,targets):
            def hook(module,inputs,out,name=name): self.values[name]=out
            module.register_forward_hook(hook)
    def forward(self,x,modulation,mask):
        out=self.block(x,time_step=modulation,attention_mask=mask,use_cache=False)[0]
        return (out,)+tuple(self.values[n] for n in self.names)


def main():
    import numpy as np
    import onnxruntime as ort
    torch.set_num_threads(4)
    root=ROOT/'models/levo-mixed-codec';info=json.loads((root/'source-header.json').read_text())
    Config,_,source=upstream_classes(ROOT/'reference/LeVo')
    cfg=Config(n_embd=1200,n_layer=39,n_head=30,n_positions=1000,use_cache=False,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)
    cfg._attn_implementation='eager';block=source.GPT2Block(cfg,layer_idx=0).eval()
    prefix='cfm_wrapper.estimator.h.0.'
    block.load_state_dict({k.removeprefix(prefix):tensor_from_parts(k,info,root/'parts') for k in info['tensors'] if k.startswith(prefix)},strict=True)
    source.precompute_freqs_cis=real_freqs;source.apply_rotary_emb=real_rope
    model=ObservedBlock(block).eval()
    data=np.load(ROOT/'models/dit-block/fixture.npz');names=['x','modulation','mask']
    inputs=tuple(torch.from_numpy(data[n]) for n in names)
    with torch.inference_mode():refs=model(*inputs)
    q,k,v=model.values['qkv'].split(1200,dim=2)
    q=block.attn._split_heads(q,30,40);k=block.attn._split_heads(k,30,40);v=block.attn._split_heads(v,30,40)
    freq=real_freqs(40,32)
    qrot=real_rope(q.transpose(1,2),freq).transpose(1,2)
    krot=real_rope(k.transpose(1,2),freq).transpose(1,2)
    scores=(qrot@krot.transpose(-1,-2))/(40**.5)
    prob=(scores+inputs[2]).softmax(-1)
    expected_attention={'q':q,'k':k,'v':v,'qrot':qrot,'krot':krot,'scores':scores,'masked_scores':scores+inputs[2],
                        'prob':prob,'context':prob@v}
    jit=torch.jit.trace(model,inputs,check_trace=False)
    with torch.inference_mode(): traced=jit(*inputs)
    names_out=['output']+model.names
    path=ROOT/'models/dit-block/real-block0-observed.onnx'
    torch.onnx.export(model,inputs,str(path),input_names=names,output_names=names_out,opset_version=17,dynamo=False)
    out=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']).run(None,{n:data[n] for n in names})
    rows=[]
    for name,a,b,c in zip(names_out,refs,out,traced):
        ref=a.detach().numpy()
        rows.append({'name':name,'onnx_l2':float(np.linalg.norm(b-ref)/np.linalg.norm(ref)),
                     'traced_l2':float(torch.linalg.vector_norm(c-a)/torch.linalg.vector_norm(a))})
    print(json.dumps(rows,indent=2))
    (ROOT/'reports/dit-block-export-diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    import onnx
    inferred=onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    # Observe floating-point intermediates; types are inferred, never guessed.
    existing={v.name for v in inferred.graph.output}
    for value in inferred.graph.value_info:
        if value.name not in existing and value.type.tensor_type.elem_type==onnx.TensorProto.FLOAT:
            inferred.graph.output.append(value);existing.add(value.name)
    debug_path=path.with_name('real-block0-all-intermediates.onnx');onnx.save(inferred,str(debug_path))
    sess=ort.InferenceSession(str(debug_path),providers=['CPUExecutionProvider'])
    arrays=sess.run(None,{n:data[n] for n in names})
    outputs={o.name:a for o,a in zip(sess.get_outputs(),arrays)}
    closest=[]
    for name,tensor in expected_attention.items():
        ref=tensor.detach().numpy(); candidates=[]
        for key,value in outputs.items():
            if value.shape==ref.shape:
                candidates.append((float(np.linalg.norm(value-ref)/np.linalg.norm(ref)),key))
        closest.append({'reference':name,'closest':sorted(candidates)[:3]})
    print(json.dumps(closest,indent=2))
    (ROOT/'reports/dit-attention-export-diagnostics.json').write_text(json.dumps(closest,indent=2),encoding='utf8')


if __name__=='__main__':main()
