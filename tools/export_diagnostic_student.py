"""Export the real pruned Qwen body with explicitly RANDOM music adapters.

This is a memory/flow diagnostic artifact, not a music checkpoint. It does not
enable the production student's music_trained guard or claim learned singing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exporter-root', type=Path, required=True)
    parser.add_argument('--mnn-python-path', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.exporter_root))
    sys.path.append(args.mnn_python_path)
    from llmexport import LlmExporter, build_args
    torch.set_num_threads(4)
    torch.manual_seed(2307)
    out = ROOT / 'models/student-diagnostic-mnn'
    out.mkdir(exist_ok=True)
    if (out / 'llm.mnn').exists():
        raise ValueError('Preserve existing export')
    cli = argparse.ArgumentParser()
    build_args(cli)
    options = cli.parse_args(['--path', str(ROOT / 'models/qwen3-0.6b-26layers-unrecovered'),
                             '--tokenizer_path', str(ROOT / 'models/qwen3-0.6b'),
                             '--dst_path', str(out), '--export', 'mnn', '--quant_bit', '4',
                             '--lm_quant_bit', '8', '--quant_block', '64', '--seperate_embed'])
    exporter = LlmExporter(options)
    embedding = exporter.model.embed.embed
    text_rows, width = embedding.weight.shape
    if width != 1024 or len(exporter.model.blocks) != 26:
        raise ValueError('Unexpected Qwen diagnostic body')
    style = torch.nn.Embedding(32, width)
    music = torch.nn.Embedding(16386, width)
    head = torch.nn.Linear(width, 16385, bias=False)
    combined = torch.cat((embedding.weight.detach().float(), style.weight.detach(), music.weight.detach()), dim=0)
    expanded = torch.nn.Embedding(combined.shape[0], width, dtype=embedding.weight.dtype)
    expanded.weight = torch.nn.Parameter(combined.to(embedding.weight.dtype), requires_grad=False)
    exporter.model.embed.embed = expanded
    exporter.model.lm.lm = head
    exporter.args.tie_word_embeddings = False
    exporter.llm_config['vocab_size'] = 16385
    exporter.llm_config['diagnostic_untrained_music_adapters'] = True
    manifest = {'music_trained': False, 'purpose': 'memory and module flow only',
                'backbone': '26-layer pruned real Qwen3-0.6B; no recovery training',
                'layers': 26, 'hidden': width, 'lyric_max_tokens': 512, 'style_prefix_tokens': 1,
                'style_offset': int(text_rows), 'music_offset': int(text_rows + 32),
                'bos_input_id': int(text_rows + 32 + 16384), 'eos_output_id': 16384,
                'codebook_size': 16384, 'embedding_rows': int(combined.shape[0]),
                'body_quant_bits': 4, 'music_head_quant_bits': 8, 'embedding_bits': 16,
                'random_adapter_seed': 2307, 'music_quality_validated': False}
    # Match the prototype's 512 lyric + style + BOS maximum prefix; these are
    # fixed legal tokenizer IDs, not a claim that an actual lyric was sung.
    prefix = [manifest['style_offset']] + [100 + (i * 17) % 10000 for i in range(512)] + [manifest['bos_input_id']]
    (out / 'prefix.json').write_text(json.dumps(prefix), encoding='utf8')
    exporter.export('mnn')
    for filename in ('llm.mnn', 'llm.mnn.weight', 'embeddings_bf16.bin'):
        file = out / filename
        if not file.exists() or file.stat().st_size == 0:
            raise ValueError('Incomplete exported student: ' + filename)
    (out / 'diagnostic-student.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
