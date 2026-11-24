"""Convenience entry; optional CPLEX is isolated from other model dependencies."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if (ROOT/'.operator-deps').is_dir():sys.path.insert(0,str(ROOT/'.operator-deps'))
from levo_harmony.operator_lab.cli import main
if __name__=='__main__':raise SystemExit(main())
