"""Create matched MNN CPU Low benchmark graphs for a prepared NPU fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.append(r'C:\ace-step-mnn-ios\.venv\Lib\site-packages')
import _tools

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    root=args.directory.resolve();manifest=json.loads((root/'manifest.json').read_text())
    for case in manifest['candidateCases']:
        name=case['id'];assert name in ('residual-d1-h1','residual-d1-h4','residual-stack3-h4','residual-stack3-compact','pureconv6-h4','pureconv-k7-d1-h4','pureconv-k7-d3-h4','pureconv-k7-d9-h4')
        source=root/(name+'-nchw.onnx')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==case['modelSha256']
        for bits in (0,8):
            target=root/(name+('-cpu-w8.mnn' if bits else '-cpu-fp32.mnn'))
            command=['MNNConvert','-f','ONNX','--modelFile',str(source),'--MNNModel',str(target),'--saveExternalData']
            if bits:command+=['--weightQuantBits','8','--weightQuantBlock','0']
            _tools.mnnconvert(command)
            assert target.is_file() and Path(str(target)+'.weight').is_file()

if __name__=='__main__':main()
