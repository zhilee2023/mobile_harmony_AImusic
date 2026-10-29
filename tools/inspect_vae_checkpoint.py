"""Inspect a public torch ZIP checkpoint without loading tensor payloads.

Uses PyTorch's restricted weights-only unpickler on data.pkl and meta storages.
No custom globals are allowlisted. No arbitrary pickle deserialization.
"""
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import torch
from torch import _weights_only_unpickler

ROOT = Path(__file__).resolve().parents[1]
REPO = 'lglg666/SongGeneration-Runtime'
REV = 'cc258cc694a63114c61684cc26d0583b8ad777d0'
FILE = 'ckpt/vae/autoencoder_music_1320k.ckpt'
SIZE = 674920616


class HttpFile(io.RawIOBase):
    def __init__(self, url, size): self.url=url; self.size=size; self.pos=0; self.bytes=0; self.requests=0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if not 0 <= self.pos <= self.size: raise ValueError('Invalid remote seek')
        return self.pos
    def read(self, size=-1):
        if size < 0: size = self.size - self.pos
        size = min(size, self.size-self.pos)
        if not size: return b''
        if size > 4_000_000: raise RuntimeError('Refuse unexpectedly large metadata read')
        start=self.pos; end=start+size-1
        req=urllib.request.Request(self.url+f'?zipmeta={start}',headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=45) as r:
            if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise RuntimeError('Invalid remote ZIP range response')
            data=r.read()
        if len(data)!=size: raise RuntimeError('Truncated metadata')
        self.pos+=size;self.bytes+=size;self.requests+=1
        return data


def main():
    url=f'https://huggingface.co/{REPO}/resolve/{REV}/{FILE}'
    remote=HttpFile(url,SIZE)
    with zipfile.ZipFile(remote) as archive:
        records=archive.infolist()
        record=next(x for x in records if x.filename.endswith('/data.pkl'))
        payload=archive.read(record)
        prefix=record.filename.removesuffix('data.pkl')
        names={x.filename:x for x in records}
        storages={}
        def persistent_load(saved):
            kind, storage_type, key, location, count = saved
            if kind!='storage': raise RuntimeError('Unknown persistent object')
            if key not in storages:
                dtype=storage_type.dtype
                storage=torch.UntypedStorage(count*torch._utils._element_size(dtype),device='meta')
                storage._remote_storage_key=key
                storages[key]=torch.storage.TypedStorage(wrap_storage=storage,dtype=dtype,_internal=True)
            return storages[key]
        loader=_weights_only_unpickler.Unpickler(io.BytesIO(payload),encoding='utf-8')
        loader.persistent_load=persistent_load
        data=loader.load()
        state=data['state_dict'] if 'state_dict' in data else data
        tensors={}
        for k,t in state.items():
            if not isinstance(t,torch.Tensor): continue
            key=t.untyped_storage()._remote_storage_key
            rec=names[prefix+'data/'+key]
            tensors[k]={'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),
                        'storage_offset_elements':t.storage_offset(),'storage_key':key,
                        'record':rec.filename,'local_header_offset':rec.header_offset,
                        'storage_bytes':rec.file_size,'zip_crc32':rec.CRC,'compression':rec.compress_type}
    report={'repo':REPO,'revision':REV,'file':FILE,'source_bytes':SIZE,
            'metadata_http_bytes':remote.bytes,'metadata_http_requests':remote.requests,
            'restricted_weights_only':True,'tensor_payloads_loaded':False,'tensors':tensors}
    out=ROOT/'models/levo-vae';out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoint-metadata.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='tensors'},indent=2))
    print('tensors',len(tensors),'decoder keys',sum('decoder.' in k for k in tensors))


if __name__=='__main__': main()
