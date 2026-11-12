"""Export one text style using the official MuQ-MuLan text branch and weights.

Avoids downloading/initializing an unused audio backbone. Every text parameter
and the final projection is loaded strictly from the pinned joint checkpoint.
The official source checkout is a local dependency, never copied into this repo.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = '28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a'
MODEL_REVISION = '2e01c796b71dca71b45251384c04cd7b237c9020'
TEXT_REVISION = 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
CHECKPOINT_SHA256 = 'd42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444'

def save_style(vector, text, output, provenance):
    import numpy as np
    value = np.asarray(vector, dtype='<f4')
    if value.shape != (512,) or not np.isfinite(value).all():
        raise ValueError('Expected exactly512 finite text features')
    if abs(float(np.linalg.norm(value))-1) > 1e-4:
        raise ValueError('Official text output should already be L2-normalized')
    output = Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Preserve an existing style; select a new output filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    metadata = dict(provenance, text=text, modality='text', encoder='MuQ-MuLan',
        shape=[512], dtype='float32', l2Norm=float(np.linalg.norm(value)),
        npySha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    output.with_suffix('.txt').write_text(text+'\n', encoding='utf-8')
    return metadata

def extract(text, source, checkpoint, threads):
    import torch
    import torch.nn.functional as functional
    from transformers import AutoConfig, XLMRobertaModel
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError('Use the pinned official MuQ source revision')
    compact_path=checkpoint/'text-model.safetensors'
    compact=compact_path.is_file()
    compact_info=None
    if compact:
        compact_info=json.loads((checkpoint/'provenance.json').read_text(encoding='utf-8'))
        if compact_info['revision']!=MODEL_REVISION or compact_info['textSha256']!=hashlib.sha256(text.encode()).hexdigest():
            raise ValueError('Prompt-specific checkpoint does not match this exact text')
        digest=hashlib.sha256(compact_path.read_bytes()).hexdigest()
        if digest!=compact_info['textCheckpointSha256']:
            raise ValueError('Trained text checkpoint SHA256 mismatch')
    else:
        weight = checkpoint / 'pytorch_model.bin'
        with weight.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError('Official joint checkpoint SHA256 mismatch')
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if config['text_model']['name'] != 'xlm-roberta-base' or config['mulan']['dim_latent'] != 512:
        raise ValueError('Unsupported MuQ-MuLan text architecture')
    # Import only the official text implementation, without audio dependency
    # initialization performed by the library's top-level __init__.
    for name, folder in [('_muq_style_ref',source/'src/muq'),
                         ('_muq_style_ref.muq_mulan',source/'src/muq/muq_mulan')]:
        package=types.ModuleType(name);package.__path__=[str(folder)];sys.modules[name]=package
    if 'x_clip' not in sys.modules:
        spec=importlib.util.find_spec('x_clip')
        if spec is None or spec.origin is None:
            raise ImportError('Install x-clip, the official source tokenizer dependency')
        # Text-only use needs its real tokenizer module, not torchvision/CLIP.
        package=types.ModuleType('x_clip');package.__path__=[str(Path(spec.origin).parent)];sys.modules['x_clip']=package
    module=importlib.import_module('_muq_style_ref.muq_mulan.models.text')
    cls=module.TextTransformerPretrained
    roberta_config=AutoConfig.from_pretrained('FacebookAI/xlm-roberta-base', revision=TEXT_REVISION, trust_remote_code=False)
    if compact:
        if compact_info['embeddingOriginalIds'][:4]!=[0,1,2,3]:
            raise ValueError('Special/padding IDs must be preserved in the compact input table')
        roberta_config.vocab_size=len(compact_info['embeddingOriginalIds'])
    original=cls._init_pretrained_model
    cls._init_pretrained_model=lambda self: XLMRobertaModel(roberta_config)
    try:
        model=cls(model_name='xlm-roberta-base', model_dim=None,
                  frozen_pretrained=False, **config['text_transformer'])
    finally:
        cls._init_pretrained_model=original
    if compact:
        from safetensors.torch import load_file
        state=load_file(compact_path,device='cpu')
    else:
        state=torch.load(weight, map_location='cpu', mmap=True, weights_only=True)
    prefix='mulan.text.'
    text_state={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
    if not text_state:
        raise ValueError('No trained text branch in checkpoint')
    model.load_state_dict(text_state, strict=True)
    projection=torch.nn.Linear(768,512)
    projection.load_state_dict({name:state['mulan.text_to_latents.'+name] for name in ('weight','bias')}, strict=True)
    model.eval();projection.eval();torch.set_num_threads(threads)
    tokenizer=module.AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base', revision=TEXT_REVISION, use_fast=False, trust_remote_code=False)
    ids=tokenizer(text, add_special_tokens=True)['input_ids']
    if compact:
        lookup=torch.full((250002,),-1,dtype=torch.long)
        lookup[torch.tensor(compact_info['embeddingOriginalIds'])]=torch.arange(len(compact_info['embeddingOriginalIds']))
        class PromptTokenizer:
            def __call__(self,*args,**kwargs):
                encoded=tokenizer(*args,**kwargs)
                values=encoded['input_ids']
                if not isinstance(values,torch.Tensor):raise ValueError('Expected batched token tensors')
                mapped=lookup[values]
                if bool((mapped<0).any()):raise ValueError('Unknown token; prepare this new prompt first')
                encoded['input_ids']=mapped
                return encoded
        model._tokenizer=PromptTokenizer()
    else:
        model._tokenizer=tokenizer
    if len(ids)>512:
        raise ValueError('Style exceeds the pretrained text backbone position limit')
    with torch.inference_mode():
        embedding=functional.normalize(projection(model(raw_texts=[text])), p=2, dim=-1)[0]
    return embedding.numpy(), dict(model='OpenMuQ/MuQ-MuLan-large', modelRevision=MODEL_REVISION,
        sourceRevision=revision, tokenizerRevision=TEXT_REVISION, checkpointSha256=digest, textTokens=len(ids),
        inference='official text module + trained projection, strict load, FP32 CPU',
        fullAudioBranchLoaded=False, promptSpecificInputTable=compact,
        wholeJointSha256Verified=not compact,
        sourceJointSha256=CHECKPOINT_SHA256,
        textCheckpointProvenance=compact_info)

def main():
    parser=argparse.ArgumentParser()
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text');inputs.add_argument('--text-file',type=Path)
    parser.add_argument('--source',type=Path,default=ROOT/'reference/MuQ')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/muq-mulan-official')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    text=(args.text if args.text is not None else args.text_file.read_text(encoding='utf-8-sig')).strip()
    if not text or len(text)>4000 or args.threads not in range(1,17):
        parser.error('Supply1–4000 style characters and1–16 CPU threads')
    vector,provenance=extract(text,args.source,args.checkpoint,args.threads)
    print(json.dumps(save_style(vector,text,args.out,provenance),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
