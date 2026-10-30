"""Matched H4 holdout comparisons; preserve official quantization evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np
sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
import onnxruntime as ort

root=Path(__file__).resolve().parents[1];source=root/'models/vae-cann-quant'
with tarfile.open(source/'h4-results.tar.gz') as archive:
    archive.extractall(source,filter='data')
out=root/'models/vae-cann-h4-phone';out.mkdir(exist_ok=True)
opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
records=[]
for identifier,baseline in [('residual-d1-h4',root/'models/vae-npu-block'),('pureconv-k7-d1-h4',root/'models/vae-npu-pureconv')]:
    data=source/'cases'/identifier;dest=out/identifier;dest.mkdir(exist_ok=True)
    manifest=json.loads((data/'manifest.json').read_text())
    conv=json.loads((data/'compiled/conversion-result.json').read_text())
    assert conv['calibrationConsumedConfirmed'] and conv['calibrationSamples']==20
    row=next(r for r in manifest['rows'] if r['file']=='held-out/activation-20.f32')
    path=data/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    x=np.fromfile(path,dtype='<f4').reshape(manifest['shape'])
    model=data/manifest['modelFile'];assert hashlib.sha256(model.read_bytes()).hexdigest()==manifest['modelSha256']
    y=ort.InferenceSession(str(model),opts,providers=['CPUExecutionProvider']).run(None,{'x':x})[0]
    previous=ort.InferenceSession(str(baseline/(identifier+'-nchw.onnx')),opts,providers=['CPUExecutionProvider']).run(None,{'x':x})[0]
    error=float(np.linalg.norm((y-previous).astype(np.float64))/np.linalg.norm(previous.astype(np.float64)))
    assert error<1e-6 and np.isfinite(y).all()
    x.tofile(dest/(identifier+'-input.f32'));y.tofile(dest/(identifier+'-reference.f32'))
    for variant in conv['cases']:
        path=data/'compiled'/(identifier+'-'+variant['variant']+'.omc')
        assert variant['success'] and hashlib.sha256(path.read_bytes()).hexdigest()==variant['sha256']
        shutil.copy2(path,dest)
    for variant in ('fp32','w8'):
        for suffix in ('.mnn','.mnn.weight'):shutil.copy2(baseline/(identifier+'-cpu-'+variant+suffix),dest)
    candidate={'ablation':manifest['ablation'],'heldOutSourceRows':row['sourceRows'],
        'candidateCases':[{'id':identifier,'shape':manifest['shape'],'offlineVariant':v} for v in ('fp16','w8a8')]}
    (dest/'manifest.json').write_text(json.dumps(candidate,indent=2),encoding='utf8')
    records.append({'id':identifier,'fp32GraphParityWithCpuSourceL2':error,'heldOutRow':row,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dest.iterdir() if p.is_file()}})
(out/'manifest.json').write_text(json.dumps({'cases':records},indent=2),encoding='utf8')
print(json.dumps([{'id':r['id'],'cpuSourceL2':r['fp32GraphParityWithCpuSourceL2']} for r in records]))
