"""Same real VAE and fixed latent; weight-only W8 numeric probe, not listening QA."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
import MNN

folder = ROOT / 'models/vae-stream-256'
rows = []
for variant in ('fp32', 'w8'):
    model = folder / f'vae-256-{variant}.mnn'
    tick = time.perf_counter()
    net = MNN.Interpreter(str(model))
    session = net.createSession({'backend': 'CPU', 'numThread': 4, 'precision': 1, 'memory': 2})
    setup = time.perf_counter() - tick
    errors = []
    for index in range(3):
        latent = np.fromfile(folder / f'latent-{index}.f32', dtype='<f4').reshape(1, 64, 256)
        host = MNN.Tensor(latent.shape, MNN.Halide_Type_Float, latent, MNN.Tensor_DimensionType_Caffe)
        if not net.getSessionInput(session, 'latent').copyFrom(host):
            raise ValueError('Input copy failed')
        tick = time.perf_counter()
        status = net.runSession(session)
        output = np.asarray(net.getSessionOutput(session, 'pcm').getNumpyData()).copy().reshape(-1)
        seconds = time.perf_counter() - tick
        reference = np.fromfile(folder / f'pcm-{index}.f32', dtype='<f4')
        if status not in (None, 0) or output.shape != reference.shape or not np.isfinite(output).all():
            raise ValueError('Invalid VAE output')
        error = float(np.linalg.norm((output-reference).astype(np.float64))/np.linalg.norm(reference.astype(np.float64)))
        errors.append({'chunk': index+1, 'seconds': seconds, 'l2VsPytorch': error})
    files = {}
    for file in (model, Path(str(model)+'.weight')):
        files[file.name] = {'bytes': file.stat().st_size, 'sha256': hashlib.file_digest(file.open('rb'), 'sha256').hexdigest()}
    rows.append({'variant': variant, 'setupSeconds': setup, 'rows': errors, 'files': files})
    print(json.dumps(rows[-1]), flush=True)
    del session, net
report = {'scope': 'PC CPU High numeric validation; no trained music or listening acceptance', 'rows': rows}
(ROOT/'reports/vae-w8-pc-validation.json').write_text(json.dumps(report, indent=2), encoding='utf8')
