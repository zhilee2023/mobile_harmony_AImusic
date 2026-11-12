"""Fetch trained text tensors from the pinned public joint checkpoint.

A restricted metadata unpickler reads shapes without fetching audio tensors.
This is a transfer optimization, not quantization or replacement parameters.
"""
import io
import json
from pathlib import Path
import requests
import pickle
import collections
import zipfile
import concurrent.futures
import struct
import zlib
import hashlib
import time
import argparse

ROOT=Path(__file__).resolve().parents[1]
REVISION='2e01c796b71dca71b45251384c04cd7b237c9020'
SIZE=2653954401
URL=f'https://huggingface.co/OpenMuQ/MuQ-MuLan-large/resolve/{REVISION}/pytorch_model.bin?download=true'

class RangeFile(io.RawIOBase):
    def __init__(self):
        self.pos=0;self.blocks={};self.step=65536
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def readinto(self,b):
        value=self.read(len(b));b[:len(value)]=value;return len(value)
    def read(self,n=-1):
        if n<0 or n>2*1024*1024:raise ValueError(f'Unexpected metadata read size {n}')
        end=min(SIZE,self.pos+n);result=bytearray()
        while self.pos<end:
            index=self.pos//self.step;offset=index*self.step;limit=min(SIZE,offset+self.step)
            if index not in self.blocks:
                part=ROOT/'models/muq-mulan-official/.ranges'/str(offset//(16*1024*1024))
                if part.is_file():
                    with part.open('rb') as file:file.seek(offset%(16*1024*1024));data=file.read(limit-offset)
                else:
                    r=requests.get(URL+f'&metadata={index}',headers={'Range':f'bytes={offset}-{limit-1}'},timeout=(15,45))
                    r.raise_for_status();data=r.content
                    if r.status_code!=206 or r.headers.get('content-range')!=f'bytes {offset}-{limit-1}/{SIZE}':raise ValueError('Metadata range mismatch')
                if len(data)!=limit-offset:raise ValueError('Short metadata block')
                self.blocks[index]=data
            length=min(end-self.pos,limit-self.pos);result.extend(self.blocks[index][self.pos-offset:self.pos-offset+length]);self.pos+=length
        return bytes(result)

def metadata():
    output=ROOT/'models/muq-text-only'
    output.mkdir(parents=True,exist_ok=True)
    path=output/'tensor-index.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    reader=RangeFile()
    with zipfile.ZipFile(reader) as archive:
        pickle_name=next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        base=pickle_name[:-len('data.pkl')]
        infos={name:archive.getinfo(name) for name in archive.namelist()}
        class MetadataOnly(pickle.Unpickler):
            def find_class(self,module,name):
                if module=='collections' and name=='OrderedDict':return collections.OrderedDict
                if module=='torch' and name in ('FloatStorage','LongStorage','HalfStorage','BFloat16Storage'):
                    return {'FloatStorage':'float32','LongStorage':'int64','HalfStorage':'float16','BFloat16Storage':'bfloat16'}[name]
                if module=='torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'):
                    def tensor(storage,offset,shape,stride,*unused):
                        return dict(storage=storage,storageOffset=offset,shape=list(shape),stride=list(stride))
                    return tensor
                raise pickle.UnpicklingError(f'Unsupported metadata class {module}.{name}')
            def persistent_load(self,value):
                if value[0]!='storage':raise pickle.UnpicklingError('Unexpected persistent metadata')
                return dict(dtype=value[1],key=value[2],elements=value[4])
        state=MetadataOnly(io.BytesIO(archive.read(pickle_name))).load()
        tensors={}
        for name,value in state.items():
            storage=value['storage'];info=infos[base+'data/'+storage['key']]
            if info.compress_type!=zipfile.ZIP_STORED:raise ValueError('Compressed tensor storage unsupported')
            tensors[name]=dict(value,dtype=storage['dtype'],recordOffset=info.header_offset,recordBytes=info.compress_size,crc32=info.CRC,recordName=info.filename)
    path.write_text(json.dumps(tensors,indent=2),encoding='utf-8')
    return tensors

def prepare(text):
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    folder=ROOT/'models/muq-text-only';index=metadata()
    tokenizer=AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base',revision='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',use_fast=False,trust_remote_code=False)
    ids=tokenizer(text,add_special_tokens=True)['input_ids']
    if len(ids)>512:raise ValueError('Text exceeds512 tokens')
    allowed=sorted(set(ids+[0,1,2,3]));(folder/'prompt.txt').write_text(text,encoding='utf-8')
    names=[name for name in index if name.startswith('mulan.text.') or name.startswith('mulan.text_to_latents.')]
    word='mulan.text.model.embeddings.word_embeddings.weight'
    records=folder/'records';records.mkdir(exist_ok=True)
    def request(start,end):
        for attempt in range(4):
            try:
                r=requests.get(URL+f'&textpart={start}',headers={'Range':f'bytes={start}-{end}'},timeout=(15,60))
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('content-range')!=f'bytes {start}-{end}/{SIZE}' or len(r.content)!=end-start+1:raise ValueError('Text range mismatch')
                return r.content
            except (requests.RequestException,ValueError):
                if attempt==3:raise
                time.sleep(attempt+1)
    def header(info):
        value=request(info['recordOffset'],info['recordOffset']+29)
        if value[:4]!=b'PK\x03\x04':raise ValueError('Storage ZIP header mismatch')
        name_length,extra_length=struct.unpack_from('<HH',value,26)
        return info['recordOffset']+30+name_length+extra_length
    # Fetch each tensor independently; verify its original ZIP storage CRC.
    def tensor(name):
        info=index[name];key=info['storage']['key'];path=records/(key+'.bin')
        if not key.isdigit():raise ValueError('Unexpected storage key')
        if path.exists() and path.stat().st_size==info['recordBytes'] and zlib.crc32(path.read_bytes())==info['crc32']:
            return name,path
        offset=header(info);size=info['recordBytes'];parts=[]
        # Cache4MiB pieces so retry/resume never repeats a large tensor.
        for start in range(0,size,4*1024*1024):
            n=min(4*1024*1024,size-start);part=records/(key+f'-{start}.part')
            if not part.exists() or part.stat().st_size!=n:
                value=request(offset+start,offset+start+n-1);pending=part.with_suffix('.pending');pending.write_bytes(value);pending.replace(part)
            parts.append(part)
        value=b''.join(part.read_bytes() for part in parts)
        if zlib.crc32(value)!=info['crc32']:raise ValueError('Original storage CRC mismatch')
        pending=path.with_suffix('.pending');pending.write_bytes(value);pending.replace(path)
        return name,path
    tensors={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        jobs=[pool.submit(tensor,name) for name in names if name!=word]
        for count,job in enumerate(concurrent.futures.as_completed(jobs),1):
            name,path=job.result();info=index[name]
            dtype={'float32':torch.float32,'int64':torch.int64,'float16':torch.float16,'bfloat16':torch.bfloat16}[info['dtype']]
            storage=torch.frombuffer(bytearray(path.read_bytes()),dtype=dtype)
            tensors[name]=torch.as_strided(storage,info['shape'],info['stride'],info['storageOffset']).clone()
            if count%20==0:print(f'Verified text tensors {count}/{len(jobs)}',flush=True)
        info=index[word]
        if info['shape']!=[250002,768] or info['dtype']!='float32' or info['storageOffset']!=0:raise ValueError('Unexpected source word embedding')
        offset=header(info);row_bytes=768*4
        def row(token):
            path=records/f'word-{token}.bin'
            if not path.exists():path.write_bytes(request(offset+token*row_bytes,offset+(token+1)*row_bytes-1))
            value=path.read_bytes()
            if len(value)!=row_bytes:raise ValueError('Short word embedding row')
            return torch.frombuffer(bytearray(value),dtype=torch.float32).clone()
        tensors[word]=torch.stack(list(pool.map(row,allowed)))
    save_file(tensors,str(folder/'text-model.safetensors'))
    provenance=dict(model='OpenMuQ/MuQ-MuLan-large',revision=REVISION,officialJointSha256='d42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444',
        wholeJointSha256Verified=False,fullTextStorageCrc32Verified=True,embeddingOriginalIds=allowed,textSha256=hashlib.sha256(text.encode()).hexdigest(),
        scope='Exact trained text branch with prompt-specific input embedding rows; other token IDs must be rejected',
        textCheckpointSha256=hashlib.sha256((folder/'text-model.safetensors').read_bytes()).hexdigest())
    (folder/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    r=requests.get(f'https://huggingface.co/OpenMuQ/MuQ-MuLan-large/raw/{REVISION}/config.json',timeout=30);r.raise_for_status();(folder/'config.json').write_text(r.text,encoding='utf-8')
    print('Trained text checkpoint ready',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--text-file',type=Path);args=parser.parse_args()
    if args.text_file:prepare(args.text_file.read_text(encoding='utf-8-sig').strip())
    else:
        index=metadata();names=[name for name in index if name.startswith('mulan.text.') or name.startswith('mulan.text_to_latents.')]
        print(json.dumps(dict(tensors=len(index),textTensors=len(names),textBytes=sum(index[name]['recordBytes'] for name in names),examples=names[:3]),indent=2),flush=True)
