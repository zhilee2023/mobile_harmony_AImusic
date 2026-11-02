"""Attach sensor rows and statistics to the same test JSON; no extra upload needed."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.telemetry import summarize
from levo_harmony.deployment.files import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--log', type=Path, required=True); parser.add_argument('--attach-report', type=Path)
    args = parser.parse_args(); rows=[]; dropped=0
    if args.log.is_file():
        for line in args.log.read_text(encoding='utf-8-sig').splitlines():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    dropped += 1
                    continue
                rows.append(row)
            except ValueError: dropped+=1
    result = summarize(rows); result['rawRows']=rows; result['unreadableRows']=dropped
    if args.attach_report:
        report=read_json(args.attach_report); report['telemetry']=result; write_json(args.attach_report, report)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rawRows','samples')},ensure_ascii=False))


if __name__=='__main__': main()
