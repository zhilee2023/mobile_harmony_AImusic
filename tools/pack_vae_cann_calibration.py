"""Small H1 official-DOPT calibration package; full H4 data stays local."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np

root=Path(__file__).resolve().parents[1]
source=root/'models/vae-cann-calibration'
out=root/'models/vae-cann-quant'
out.mkdir(exist_ok=True)
manifest=json.loads((source/'manifest.json').read_text())
records=[]
for record in manifest['rows']:
    path=source/Path(record['file'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    value=np.fromfile(path,dtype='<f4').reshape(manifest['shape'])[:,:,:1,:].copy()
    folder=out/record['split'];folder.mkdir(exist_ok=True)
    dest=folder/Path(record['file']).name
    value.tofile(dest)
    records.append({**record,'file':dest.relative_to(out).as_posix(),'bytes':dest.stat().st_size,
        'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
model=root/'models/vae-npu-block/residual-d1-h1-nchw.onnx'
shutil.copy2(model,out/model.name)
result={**manifest,'shape':[1,128,1,8320],'rows':records,'sourceH4CalibrationManifestSha256':
    hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
    'modelFile':model.name,'modelSha256':hashlib.sha256(model.read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
with tarfile.open(out/'calibration-package.tar.gz','w:gz',compresslevel=1) as archive:
    for path in [out/'manifest.json',out/model.name,*sorted((out/'calibration').glob('*.f32')),
                 *sorted((out/'held-out').glob('*.f32'))]:
        archive.add(path,arcname=path.relative_to(out).as_posix())
print(json.dumps({'samples':result['calibrationSamples'],'heldOut':result['heldOutSamples'],
                  'packageBytes':(out/'calibration-package.tar.gz').stat().st_size}))
