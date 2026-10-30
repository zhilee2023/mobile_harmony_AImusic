"""Four held-out real-prefix inputs, same FP32 reference for all backends."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
import onnxruntime as ort

root=Path(__file__).resolve().parents[1]
source=root/'models/vae-cann-quant';baseline=root/'models/vae-npu-block'
out=root/'models/vae-cann-phone';out.mkdir(exist_ok=True)
manifest=json.loads((source/'manifest.json').read_text())
conversion=json.loads((source/'compiled/conversion-result.json').read_text())
assert conversion['calibrationConsumedConfirmed'] and conversion['calibrationSamples']==20
for row in conversion['cases']:
    path=source/'compiled'/('residual-d1-h1-'+row['variant']+'.omc')
    assert row['success'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
session=ort.InferenceSession(str(source/manifest['modelFile']),opts,providers=['CPUExecutionProvider'])
records=[]
for i,row in enumerate(r for r in manifest['rows'] if r['split']=='held-out'):
    name='heldout-'+str(i);dest=out/name;dest.mkdir(exist_ok=True)
    x=np.fromfile(source/row['file'],dtype='<f4').reshape(manifest['shape'])
    assert hashlib.sha256((source/row['file']).read_bytes()).hexdigest()==row['sha256']
    y=session.run(None,{'x':x})[0];assert y.shape==x.shape and np.isfinite(y).all()
    identifier='residual-d1-h1'
    x.tofile(dest/(identifier+'-input.f32'));y.tofile(dest/(identifier+'-reference.f32'))
    for variant in ('fp16','w8a8'):
        shutil.copy2(source/'compiled'/(identifier+'-'+variant+'.omc'),dest)
    for variant in ('fp32','w8'):
        for suffix in ('.mnn','.mnn.weight'):
            shutil.copy2(baseline/(identifier+'-cpu-'+variant+suffix),dest)
    variants=('fp16','w8a8') if i%2==0 else ('w8a8','fp16')
    candidate={'ablation':'Original residual with Snake and skip retained; held-out real-prefix activation from synthetic latent',
        'heldOutIndex':i,'calibrationSamples':20,'candidateCases':[
            {'id':identifier,'shape':manifest['shape'],'offlineVariant':variant} for variant in variants]}
    (dest/'manifest.json').write_text(json.dumps(candidate,indent=2),encoding='utf8')
    records.append({'folder':name,'inputSource':row,'files':{p.name:{'bytes':p.stat().st_size,
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dest.iterdir() if p.is_file()}})
(out/'manifest.json').write_text(json.dumps({'cases':records},indent=2),encoding='utf8')
print(json.dumps({'heldOut':len(records),'modelVariants':['FP16','official DOPT W8A8'],'fallbackAllowed':False}))
