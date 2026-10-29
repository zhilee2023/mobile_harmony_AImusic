"""Instantiate pinned official Oobleck decoder classes without unrelated audio deps.

The AST-selected definitions execute from the separately downloaded upstream
files; no upstream source is silently copied into this project's MIT license.
"""
import ast
import json
import math
from pathlib import Path
from typing import Literal
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def decoder_class():
    scope = {'torch':torch,'nn':nn,'math':math,'Literal':Literal}
    scope['WNConv1d'] = lambda *a,**kw: nn.utils.weight_norm(nn.Conv1d(*a,**kw))
    scope['WNConvTranspose1d'] = lambda *a,**kw: nn.utils.weight_norm(nn.ConvTranspose1d(*a,**kw))
    selected = {'snake_beta','SnakeBeta'}
    tree = ast.parse((ROOT/'reference/vae-source/blocks.py').read_text(encoding='utf8'))
    body = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in selected]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'official_snake_definitions','exec'),scope)
    selected = {'get_activation','ResidualUnit','DecoderBlock','OobleckDecoder'}
    tree = ast.parse((ROOT/'reference/vae-source/autoencoders.py').read_text(encoding='utf8'))
    body = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in selected]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'official_oobleck_decoder','exec'),scope)
    return scope['OobleckDecoder']


def create_decoder():
    cfg=json.loads((ROOT/'reference/stable_audio_1920_vae.json').read_text())['model']['decoder']['config']
    return decoder_class()(**cfg)


def main():
    with torch.device('meta'): model=create_decoder()
    metadata=json.loads((ROOT/'models/levo-vae/checkpoint-metadata.json').read_text())['tensors']
    expected={k.removeprefix('decoder.'):v['shape'] for k,v in metadata.items() if k.startswith('decoder.')}
    actual={k:list(v.shape) for k,v in model.state_dict().items()}
    missing=sorted(expected.keys()-actual.keys());extra=sorted(actual.keys()-expected.keys())
    mismatch={k:{'checkpoint':expected[k],'module':actual[k]} for k in expected.keys()&actual.keys() if expected[k]!=actual[k]}
    report={'meta_only':True,'full_decoder_weights_loaded':False,'parameters':sum(p.numel() for p in model.parameters()),
            'matched_tensors':sum(actual.get(k)==v for k,v in expected.items()),'missing':missing,'unexpected':extra,
            'shape_mismatches':mismatch,'sample_rate':48000,'latent_channels':64,'output_channels':2,'stride_product':1920}
    if missing or extra or mismatch: raise RuntimeError(json.dumps(report,indent=2))
    (ROOT/'reports/vae-decoder-restoration.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
