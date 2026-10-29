"""Materialize the exact existing PC reference fixture for HDC native benchmark."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--frames',type=int,choices=(32,128,256),default=32)
args=parser.parse_args()
suffix='' if args.frames==32 else f'-{args.frames}'
data=np.load(root/f'models/dit-block{suffix}/fixture.npz')
out=root/f'models/phone-block{suffix}';out.mkdir(exist_ok=True)
manifest={}
for name in ('x','modulation','mask','reference'):
    val=np.ascontiguousarray(data[name],dtype='<f4');path=out/(name+'.f32');path.write_bytes(val.tobytes())
    manifest[name]={'shape':list(val.shape),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(out/'fixture.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps(manifest,indent=2))
