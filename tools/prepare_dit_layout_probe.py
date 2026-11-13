"""Prepare isolated current-DiT ONNX layout candidates; never deploys a bundle."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from levo_harmony.deployment.dit_layout import prepare_capture

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare_capture(args.capture_dir, args.out)
    print(json.dumps({'cases': [{k: r[k] for k in ('id', 'relativeL2', 'transposeNodes')} for r in result['cases']],
                      'provenance': result['provenance']}, ensure_ascii=False, indent=2))
