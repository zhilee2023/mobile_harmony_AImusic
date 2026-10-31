"""Export original 28-layer Qwen3-0.6B language checkpoints for low-bit probes.

The experimental A4 route uses the official SmoothQuant metadata writer with
act_bit=4: four-bit numerical range in INT8 tensors, NOT a packed INT4 GEMM.
The LM head is W4 but excluded from static activation quantization by MNN.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('w4', 'smooth-a4', 'smooth-a8'), required=True)
    parser.add_argument('--exporter-root', type=Path, default=Path('C:/cosyvoice-harmony/third_party/MNN/transformers/llm/export'))
    parser.add_argument('--mnn-python-path', default='C:/ace-step-mnn-ios/.venv/Lib/site-packages')
    parser.add_argument('--resume-incomplete', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.exporter_root))
    sys.path.append(args.mnn_python_path)
    from llmexport import LlmExporter, build_args
    from probe_quarot_cpu import LINES, STYLES
    torch.set_num_threads(4)
    torch.manual_seed(2307)
    out = ROOT / ('models/qwen-language-' + args.variant)
    if out.exists() and (not args.resume_incomplete or (out / 'llm.mnn').exists()):
        raise ValueError('Preserve previous export: ' + str(out))
    out.mkdir(exist_ok=True)
    calibration = [f'请用中文说明{STYLES[i % len(STYLES)]}的特点，再续写两句歌词：{LINES[i]}' for i in range(20)]
    calib_path = out / 'calibration.txt'
    calib_path.write_text('\n'.join(calibration), encoding='utf8')
    cli = argparse.ArgumentParser()
    build_args(cli)
    argv = ['--path', str(ROOT / 'models/qwen3-0.6b'), '--dst_path', str(out),
            '--export', 'mnn', '--quant_bit', '4', '--lm_quant_bit', '4',
            '--quant_block', '64', '--seperate_embed', '--sym']
    if args.variant.startswith('smooth'):
        argv += ['--smooth', '--act_bit', args.variant[-1], '--act_sym', '--calib_data', str(calib_path)]
        # Official loader imports HuggingFace datasets even for a local text file.
        # Keep its local-file tokenization, without the unused network loader.
        from utils.smooth_quantizer import SmoothQuantizer
        def local_calibration(data, tokenizer=None, n_samples=128, max_seq_len=512, split='train'):
            lines = Path(data).read_text(encoding='utf8').splitlines()
            samples = []
            for line in lines[:n_samples]:
                prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': ''}, {'role': 'user', 'content': line}])
                samples.append(tokenizer(prompt, return_tensors='pt', max_length=max_seq_len, truncation=True).input_ids)
            return samples
        SmoothQuantizer.get_calib_dataset = staticmethod(local_calibration)
    started = time.perf_counter()
    exporter = LlmExporter(cli.parse_args(argv))
    if len(exporter.model.blocks) != 28 or exporter.model.lm.lm.out_features != 151936:
        raise ValueError('Need unpruned original language model and complete trained head')
    exporter.export('mnn')
    files = {}
    for name in ('llm.mnn', 'llm.mnn.weight', 'embeddings_bf16.bin', 'tokenizer.mtok', 'llm_config.json', 'config.json'):
        path = out / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Incomplete export: ' + name)
        files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
    graph = json.loads((out / 'llm.mnn.json').read_text(encoding='utf8')) if (out / 'llm.mnn.json').exists() else {}
    quant = graph.get('extraTensorDescribe', [])
    metadata = {'source': 'Original pretrained Qwen3-0.6B, 28 layers, full language vocabulary',
                'layers': 28, 'hidden': 1024, 'vocabulary': 151936, 'variant': args.variant,
                'bodyWeightBits': 4, 'headWeightBits': 4, 'embeddingBits': 16,
                'activationRangeBitsRequested': int(args.variant[-1]) if args.variant.startswith('smooth') else None,
                'packedActivation4KernelClaimed': False, 'headActivationStaticQuantized': False,
                'calibrationCount': 20 if args.variant.startswith('smooth') else 0,
                'calibration': calibration, 'quantizedTensorDescriptions': len(quant),
                'quantInfoExamples': quant[:4], 'files': files, 'exportSeconds': time.perf_counter() - started}
    (out / 'language-probe.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
