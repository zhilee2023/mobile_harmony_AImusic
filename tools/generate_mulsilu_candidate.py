import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from levo_harmony.operator_lab.candidates import mulsilu_tile_source

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='C:/cosyvoice-harmony/third_party/MNN/source/backend/cpu/CPUBinary.cpp');p.add_argument('--output',required=True);p.add_argument('--tile',type=int,required=True);a=p.parse_args()
    src=Path(a.source).read_text(encoding='utf-8');out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:f.write(mulsilu_tile_source(src,a.tile))
    print(json.dumps(dict(tile=a.tile,originalFileSha256=hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),normalizedSourceSha256=hashlib.sha256(src.encode()).hexdigest(),generatedSha256=hashlib.sha256(out.read_bytes()).hexdigest(),scope='Existing MUL_SILU cache tiling, tile0 is identity recompile, no new graph fusion')))
if __name__=='__main__':main()
