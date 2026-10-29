"""Extract decoder-only storage ranges, checking every storage against ZIP CRC32.

Pinned official checkpoint; no encoder payload or unsafe pickle execution.
Full source SHA256 is not claimed; subset hash plus each source storage CRC is recorded.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request
import zlib
from download_mixed_codec import get_range

ROOT=Path(__file__).resolve().parents[1]/'models/levo-vae'


def main():
    meta=json.loads((ROOT/'checkpoint-metadata.json').read_text())
    url=f'https://huggingface.co/{meta["repo"]}/resolve/{meta["revision"]}/{meta["file"]}'
    tensors={k:v for k,v in meta['tensors'].items() if k.startswith('decoder.')}
    records={v['record']:v for v in tensors.values()}
    # PyTorch storage payloads are 64-byte aligned. Validate the rule using actual
    # local headers from both ends before deriving offsets for the other records.
    ordered=sorted(records.values(),key=lambda x:x['local_header_offset'])
    for v in (ordered[0],ordered[-1]):
        a=v['local_header_offset'];header=get_range(url,a,a+29,meta['source_bytes'])
        if header[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid ZIP storage header')
        filename_len,extra_len=struct.unpack_from('<HH',header,26)
        actual=a+30+filename_len+extra_len
        predicted=((a+30+len(v['record'].encode())+4+63)//64)*64
        if actual!=predicted:raise RuntimeError('Unknown ZIP alignment; do not guess offsets')
    for v in tensors.values():
        if v['compression']!=0:raise RuntimeError('Compressed tensor storage is unsupported')
        v['payload_offset']=((v['local_header_offset']+30+len(v['record'].encode())+4+63)//64)*64
    lo=min(v['payload_offset'] for v in tensors.values())
    hi=max(v['payload_offset']+v['storage_bytes'] for v in tensors.values())
    parts=ROOT/'decoder-parts';parts.mkdir(exist_ok=True);chunk=8*1024*1024
    jobs=[(a,min(a+chunk,hi)-1) for a in range(lo,hi,chunk)]
    def fetch(bounds):
        a,b=bounds;dest=parts/str(a)
        if dest.exists() and dest.stat().st_size==b-a+1:return b-a+1
        for attempt in range(4):
            try:
                data=get_range(url,a,b,meta['source_bytes'])
                temp=dest.with_suffix('.partial');temp.write_bytes(data);os.replace(temp,dest)
                return len(data)
            except Exception as e:
                print(f'retry={attempt+1} offset={a}: {type(e).__name__}',flush=True)
                if attempt==3:raise
                time.sleep(2+attempt)
    done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for f in concurrent.futures.as_completed([pool.submit(fetch,j) for j in jobs]):
            done+=f.result();print(f'decoder bytes {done}/{hi-lo}',flush=True)
    def read(a,n):
        values=[]
        while n:
            start=lo+(a-lo)//chunk*chunk;part=parts/str(start)
            with part.open('rb') as f:f.seek(a-start);data=f.read(min(n,start+part.stat().st_size-a))
            if not data:raise RuntimeError('Missing downloaded bytes')
            values.append(data);a+=len(data);n-=len(data)
        return b''.join(values)
    out_header={};payloads=[];offset=0
    for k,v in tensors.items():
        data=read(v['payload_offset'],v['storage_bytes'])
        if zlib.crc32(data)!=v['zip_crc32']:raise RuntimeError('ZIP CRC32 mismatch: '+k)
        if v['storage_offset_elements']!=0:raise RuntimeError('Nonzero tensor storage offset')
        out_header[k.removeprefix('decoder.')]={'dtype':'F32','shape':v['shape'],'data_offsets':[offset,offset+len(data)]}
        payloads.append(data);offset+=len(data)
    encoded=json.dumps(out_header,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%8)
    dest=ROOT/'decoder.safetensors';h=hashlib.sha256()
    with dest.with_suffix('.partial').open('wb') as out:
        for data in [struct.pack('<Q',len(encoded)),encoded]+payloads:out.write(data);h.update(data)
    os.replace(dest.with_suffix('.partial'),dest)
    report={'source_revision':meta['revision'],'decoder_tensor_count':len(tensors),'every_storage_crc32_verified':True,
            'subset_sha256':h.hexdigest(),'subset_bytes':dest.stat().st_size,'full_source_sha256_verified':False}
    (ROOT/'decoder-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
