"""Numerical full-weight Mixed codec validation, never a trained song-LM demo.

Requires the completed pinned inference subset. Synthetic valid codec IDs are
explicitly used because no trained0.6B music checkpoint/token teacher is present.
"""
import argparse
import ast
from abc import ABC
import hashlib
import json
from pathlib import Path
import sys
import time
import types
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from restore_mixed_dit import RestoredMixedCfm
from restore_vae_decoder import create_decoder
from validate_real_dit_block import sdpa_attention
from levo_harmony.mixed_sampler import MixedCodeConditioner, MixedFlowSampler, denormalize_latents


def l2(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp(min=1e-12))


def official_solver(cfm,frames):
    source=ROOT/'reference/LeVo/codeclm/tokenizer/Flow1dVAE/model_1rvq.py'
    tree=ast.parse(source.read_text(encoding='utf8'))
    body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BASECFM']
    scope={'torch':torch,'nn':nn,'ABC':ABC,'F':F,'tqdm':lambda values:values}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),
                 'official_cfm_reference','exec'),scope)
    return scope['BASECFM'](cfm.estimator,cfm.mlp,frames).eval()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=20);args=parser.parse_args()
    if not 1<=args.steps<=50:raise ValueError('Explicit step count must be1..50')
    torch.manual_seed(2307);torch.set_num_threads(4)
    folder=ROOT/'models/levo-mixed-codec'
    manifest=json.loads((folder/'download-manifest.json').read_text())
    digest=hashlib.sha256()
    with (folder/'mixed-inference.safetensors').open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=manifest['subset_sha256']:raise ValueError('Mixed subset integrity mismatch')
    state=load_file(folder/'mixed-inference.safetensors')
    begin=time.perf_counter()
    # Construct on CPU so non-persistent upstream attention buffers are real,
    # too. assign=True alone does not materialize buffers absent from state_dict.
    cfm=RestoredMixedCfm(ROOT/'reference/LeVo')
    weights={name.removeprefix('cfm_wrapper.'):value for name,value in state.items() if name.startswith('cfm_wrapper.')}
    cfm.load_state_dict(weights,strict=True,assign=True);cfm.eval()
    load_seconds=time.perf_counter()-begin
    print(f'Loaded complete real CFM: {len(weights)} tensors in {load_seconds:.3f}s',flush=True)
    aux={name:value for name,value in state.items() if not name.startswith('cfm_wrapper.')}
    conditioner=MixedCodeConditioner(aux).eval();sampler=MixedFlowSampler(cfm,aux).eval()
    # Full-network attention parity before using the less memory-intensive CPU SDPA path.
    probe=torch.randn(1,32,1200)*.1;t=torch.tensor([.5]);attention=torch.ones(1,1,32,32)
    with torch.inference_mode():eager=cfm(probe,t,attention)
    for layer in cfm.estimator.h:
        layer.attn._parity_source=sys.modules[layer.attn.__class__.__module__]
        layer.attn._attn=types.MethodType(sdpa_attention,layer.attn)
    with torch.inference_mode():sdpa=cfm(probe,t,attention)
    attention_l2=l2(sdpa,eager)
    if not torch.isfinite(sdpa).all() or attention_l2>1e-4:raise ValueError('Full-network SDPA parity failed')
    # Compare actual pretrained CFM against the pinned official Euler solver.
    frames=32;mu=conditioner(torch.arange(frames)[None]);noise=torch.randn(1,frames,64)
    context=torch.randn_like(noise);masks=torch.full((1,frames),2,dtype=torch.long);masks[:,:8]=1
    inc=context*(masks==1).unsqueeze(-1);mask_features=F.embedding(masks,aux['mask_emb.weight'])
    reference=official_solver(cfm,frames)
    with torch.inference_mode():
        expected=reference.solve_euler(noise.clone(),mask_features,inc,8,torch.linspace(0,1,4),mu,torch.ones(1,1,frames,frames),1.5)
        expected[:,:8]=inc[:,:8]
        actual=sampler(mu,noise,inc,masks,steps=3,guidance=1.5)
    solver_l2=l2(actual,expected)
    if not torch.isfinite(actual).all() or solver_l2>1e-4:raise ValueError('Pretrained official-solver parity failed')
    print(f'Full CFM parity: SDPA L2={attention_l2:.4g}; Euler L2={solver_l2:.4g}',flush=True)
    vae_manifest=json.loads((ROOT/'models/levo-vae/decoder-manifest.json').read_text())
    if not vae_manifest['every_storage_crc32_verified']:raise ValueError('VAE storage integrity missing')
    with torch.device('meta'):vae=create_decoder()
    vae.load_state_dict(load_file(ROOT/'models/levo-vae/decoder.safetensors'),strict=True,assign=True);vae.eval()
    codes=(torch.arange(448)*37%16384)[None];chunks=[];last=None;pcm_tail=None;emitted=[]
    for index,start in enumerate((0,192)):
        mu=conditioner(codes[:,start:start+256]);noise=torch.randn(1,256,64)
        context=torch.zeros_like(noise);masks=torch.full((1,256),2,dtype=torch.long)
        if last is not None:context[:,:64]=last[:,-64:];masks[:,:64]=1
        calls=[0]
        def progress(_module,_inputs,_output):
            calls[0]+=1
            if calls[0]%5==0 or calls[0]==args.steps:print(f'Codec chunk{index+1}: DiT {calls[0]}/{args.steps}',flush=True)
        hook=cfm.register_forward_hook(progress);begin=time.perf_counter()
        try:
            with torch.inference_mode():latent=sampler(mu,noise,context,masks,steps=args.steps,guidance=1.5)
        finally:hook.remove()
        dit_seconds=time.perf_counter()-begin
        if last is not None and not torch.equal(latent[:,:64],last[:,-64:]):raise ValueError('Latent context was not preserved')
        last=latent;begin=time.perf_counter()
        with torch.inference_mode():pcm=vae(denormalize_latents(latent,aux).transpose(1,2))[0]
        vae_seconds=time.perf_counter()-begin
        if list(pcm.shape)!=[2,256*1920] or not torch.isfinite(pcm).all():raise ValueError('Real codec PCM failed')
        count=64*1920
        if pcm_tail is not None:
            ramp=torch.linspace(0,1,count)[None]
            overlap=pcm_tail*(1-ramp)+pcm[:,:count]*ramp
            pcm=torch.cat((overlap,pcm[:,count:]),dim=-1)
        if index==0:pcm_tail=pcm[:,-count:].clone();emitted.append(pcm[:,:-count].clone())
        else:emitted.append(pcm.clone())
        chunks.append({'chunk':index+1,'frames':256,'context_frames':0 if index==0 else 64,
            'dit_seconds':dit_seconds,'vae_seconds':vae_seconds,'latent_peak':float(latent.abs().max()),
            'pcm_peak':float(pcm.abs().max()),'pcm_finite':True})
        print(json.dumps(chunks[-1]),flush=True)
    pcm=torch.cat(emitted,dim=-1)
    if pcm.shape[-1]!=448*1920:raise ValueError('Full codec overlap length mismatch')
    report={'full_pretrained_cfm_loaded':True,'cfm_tensors':len(weights),
        'cfm_parameters':sum(p.numel() for p in cfm.parameters()),'load_seconds':load_seconds,
        'full_source_sha256_verified':False,'subset_sha256_verified':True,'subset_sha256':digest.hexdigest(),
        'full_network_sdpa_vs_eager_l2':attention_l2,'official_solver_relative_l2':solver_l2,
        'full_pretrained_vae_loaded':True,'steps_per_chunk':args.steps,'guidance':1.5,
        'chunks':chunks,'output_shape':list(pcm.shape),'effective_pcm_seconds':17.92,
        'synthetic_codec_ids':True,'student_lm_connected':False,'real_song_generated':False,
        'scope':'Windows CPU real full codec numerical chain; not phone RTF, trained music-LM generation or listening-quality validation'}
    (ROOT/'reports/full-real-codec-cpu.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'success':True,'frames':448,'seconds':17.92,'steps':args.steps}),flush=True)


if __name__=='__main__':main()
