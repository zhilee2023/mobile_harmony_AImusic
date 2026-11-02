"""Align the trained Mixed Llama, then export MNN W4 body / W8 head.

Uses existing Llama support; does not alter the MNN checkout. Small FP32
alignment forwards upcast one BF16 linear at a time to bound temporary RAM.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

BASE = Path(__file__).resolve().parent / 'artifacts'
EXPORTER = Path('C:/cosyvoice-harmony/third_party/MNN/transformers/llm/export')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--align-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    torch.manual_seed(2307)
    sys.path.insert(0, str(EXPORTER))
    from llmexport import LlmExporter, build_args
    for filename in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
        target = BASE / 'hf-mixed' / filename
        if not target.exists():
            shutil.copyfile(BASE / 'qwen2-tokenizer' / filename, target)
    destination = BASE / 'mixed-mnn-w4'
    if (destination / 'llm.mnn.weight').exists() and not args.align_only:
        raise FileExistsError('Keep existing export')
    cli = argparse.ArgumentParser()
    build_args(cli)
    options = cli.parse_args(['--path', str(BASE / 'hf-mixed'), '--dst_path', str(destination),
                             '--tokenizer_path', str(BASE / 'qwen2-tokenizer'), '--export', 'mnn',
                             '--quant_bit', '4', '--lm_quant_bit', '8', '--quant_block', '64',
                             '--seperate_embed', '--sym'])
    reference = []
    loader = AutoModelForCausalLM.from_pretrained
    def capture(*values, **keywords):
        keywords.update(local_files_only=True, attn_implementation='sdpa')
        original = loader(*values, **keywords)
        reference.append(original)
        return original
    AutoModelForCausalLM.from_pretrained = capture
    started = time.perf_counter()
    try:
        exporter = LlmExporter(options)
    finally:
        AutoModelForCausalLM.from_pretrained = loader
    origin = reference.pop()
    origin.eval()
    exporter.model.eval()
    if len(exporter.model.blocks) != 36 or exporter.config.hidden_size != 2048:
        raise ValueError('Wrong model architecture')
    with safe_open(BASE / 'conditions.safetensors', framework='pt', device='cpu') as state:
        music = state.get_tensor('audiolm.emb.0.weight').float()
    embed = torch.nn.Embedding(16386, 2048, device='meta')
    embed.weight = torch.nn.Parameter(music, requires_grad=False)
    exporter.model.embed.embed = embed
    # Exported embedding lookup can also address real conditioned prefix rows.
    # The output head stays exactly 16385 classes, independent of lookup rows.
    prefix = torch.from_numpy(np.fromfile(BASE / 'prefix-conditioned.f32', dtype='<f4').reshape(953, 2048).copy())
    fixture = torch.cat((prefix[:31], prefix[-1:]), dim=0).unsqueeze(0)
    baseline_linear = functional.linear
    def fp32_linear(value, weight, bias=None):
        return baseline_linear(value.float(), weight.float(), None if bias is None else bias.float())
    functional.linear = fp32_linear
    checkpoints, handles = {}, []
    active = ['']
    def hook(name, mapped=False):
        def save(_module, _inputs, output):
            if not name.startswith(active[0] + '-'):
                return
            output = output[0] if isinstance(output, tuple) else output
            checkpoints[name] = output.detach().float().clone()
        return save
    for name, module in [('origin-layer0', origin.model.layers[0]), ('origin-last', origin.model.layers[-1]),
                         ('origin-norm', origin.model.norm), ('mapped-layer0', exporter.model.blocks[0]),
                         ('mapped-last', exporter.model.blocks[-1]), ('mapped-norm', exporter.model.final_layernorm)]:
        handles.append(module.register_forward_hook(hook(name)))
    rows = []
    past = None
    try:
        with torch.inference_mode():
            for step in range(2):
                if step == 0:
                    current = fixture
                else:
                    current = embed(torch.tensor([[token]]))
                tick = time.perf_counter()
                active[0] = 'origin'
                result = origin(inputs_embeds=current, past_key_values=past, use_cache=True)
                past = result.past_key_values
                logits_origin = result.logits[:, -1:, :].float()
                sequence = 32 + step
                active[0] = 'mapped'
                mapped, _, _ = exporter.model.forward(
                    current, exporter.model.get_attention_mask(sequence, step),
                    exporter.model.get_position_ids(sequence, step), torch.tensor([-1], dtype=torch.int32))
                logits_mapped = mapped[:, -1:, :].float()
                parity = {}
                for name in ('layer0', 'last', 'norm'):
                    left, right = checkpoints['origin-' + name], checkpoints['mapped-' + name]
                    # Exporter applies final norm only at the requested logits row.
                    if name == 'norm':
                        left = left[:, -1:, :]
                    value = float(torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(left).clamp(min=1e-12))
                    parity[name + 'L2'] = value
                parity['logitsL2'] = float(torch.linalg.vector_norm(logits_origin - logits_mapped) /
                                           torch.linalg.vector_norm(logits_origin).clamp(min=1e-12))
                token = int(logits_origin[0, -1].argmax())
                parity.update(phase='prefill' if step == 0 else 'decode', top1=token,
                              mappedTop1=int(logits_mapped[0, -1].argmax()), seconds=time.perf_counter() - tick)
                if not torch.isfinite(logits_mapped).all() or max(v for k, v in parity.items() if k.endswith('L2')) > 1e-4:
                    raise ValueError('Mapping parity failed: ' + json.dumps(parity))
                logits_origin.numpy().tofile(BASE / f'mapping-{step}-reference.f32')
                rows.append(parity)
                print('ALIGN ' + json.dumps(parity), flush=True)
    finally:
        functional.linear = baseline_linear
        for handle in handles:
            handle.remove()
    report = {'trainedMixedLm': True, 'trackLmIncluded': False, 'sourceSha256':
              'dc763aa9a76a22a87597c2faf9a51c24d13349ac754699b37e9068b483639def',
              'mappingFixturePositions': 32, 'checkpoints': rows, 'fullConditionPrefixPositions': 953,
              'scope': 'FP32 per-linear alignment using original BF16 checkpoint values, not whole music quality'}
    (BASE / 'mapping-validation.json').write_text(json.dumps(report, indent=2))
    if args.align_only:
        return
    del origin, past, result, checkpoints
    for block in exporter.model.blocks:
        block.self_attn.past_key_value = None
    gc.collect()
    exporter.llm_config['vocab_size'] = 16385
    exporter.llm_config['levo_trained_mixed_only'] = True
    exporter.llm_config['levo_input_embedding_rows'] = 16386
    exporter.export('mnn')
    shutil.copyfile(BASE / 'music-embeddings-bf16.bin', destination / 'embeddings_bf16.bin')
    files = {}
    for name in ('llm.mnn', 'llm.mnn.weight', 'llm_config.json', 'embeddings_bf16.bin', 'tokenizer.mtok', 'config.json'):
        path = destination / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Incomplete MNN export: ' + name)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files[name] = {'bytes': path.stat().st_size, 'sha256': digest}
    report.update(weightBits=4, headBits=8, activationBits=32, kvBits=16,
                  packedActivation4Claimed=False, files=files, exportSeconds=time.perf_counter() - started)
    (BASE / 'mixed-mnn-export.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
