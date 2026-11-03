"""Validate a hidden-sequence research interface without loading private weights."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from levo_harmony.deployment.files import read_json, write_json, sha256
from levo_harmony.deployment.hidden_conditioning import validate_contract, memory_budget


def inspect(contract_path, seconds, lm_config=None, mnn_json=None):
    contract = validate_contract(read_json(contract_path))
    result = memory_budget(contract, seconds)
    result['contract'] = contract
    if lm_config:
        config = read_json(lm_config)
        if config.get('hidden_size') != contract['hiddenSize']:
            raise ValueError('LM hidden_size differs from the declared hidden sequence')
        result['lmConfig'] = {'hiddenSize': config['hidden_size'], 'sha256': sha256(lm_config)}
    if mnn_json:
        graph = read_json(mnn_json)
        outputs = graph.get('outputName', [])
        if not {'logits', contract['hiddenOutputName']}.issubset(outputs):
            raise ValueError('Graph needs both logits and the declared hidden output; a tensor name alone is insufficient')
        result['mnnGraph'] = {'outputs': outputs, 'metadataJsonSha256': sha256(mnn_json),
                              'semanticsVerified': False, 'note': 'Output names do not prove normalization, layout, alignment or quantized hidden parity'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--lm-config', type=Path)
    parser.add_argument('--mnn-json', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = inspect(args.contract, args.seconds, args.lm_config, args.mnn_json)
    if args.out:
        if args.out.exists():
            raise ValueError('Keep prior research results; choose a new output path')
        write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
