"""Isolate native converter failures and per-invocation option state."""
import argparse
import sys

p=argparse.ArgumentParser()
p.add_argument('--mnn-python-path',required=True)
p.add_argument('converter_args',nargs=argparse.REMAINDER)
args=p.parse_args()
sys.path.insert(0,args.mnn_python_path)
import _tools
command=args.converter_args
if command and command[0]=='--':command=command[1:]
_tools.mnnconvert(['MNNConvert']+command)
