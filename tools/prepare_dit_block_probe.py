"""Prepare a verified real current DiT block for isolated CPU/CANN testing."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from levo_harmony.deployment.dit_block_layout import export_block

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_block(args.capture, args.out), ensure_ascii=False))
