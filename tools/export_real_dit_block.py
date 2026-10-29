"""Export a real LeVo Mixed block to ONNX/MNN and validate W8/W4 on PC CPU.

Measures one block at a selected fixed shape. Not whole-DiT or phone RTF.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from restore_mixed_dit import upstream_classes
from validate_real_dit_block import tensor_from_parts


def real_freqs(dim, end, constant=10000.0):
    inverse = 1.0 / (constant ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    # Legacy ONNX outer/Einsum aligns to the FIRST input dtype rather than
    # reproducing PyTorch promotion. An int64 arange would truncate all RoPE
    # inverse frequencies below 1 to zero, rotating only the first pair.
    angles = torch.outer(torch.arange(end, device=inverse.device, dtype=inverse.dtype), inverse).float()
    return torch.stack((angles.cos(), angles.sin()), -1)


def real_rope(x, frequencies):
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cs = frequencies.view(1, x.shape[1], 1, x.shape[-1]//2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a*cs[..., 0] - b*cs[..., 1], a*cs[..., 1] + b*cs[..., 0]), -1).flatten(3).type_as(x)


class Block(nn.Module):
    def __init__(self, block): super().__init__(); self.block = block
    def forward(self, x, modulation, mask):
        return self.block(x, time_step=modulation, attention_mask=mask, use_cache=False)[0]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mnn-python-path', required=True)
    p.add_argument('--frames', type=int, choices=(32, 128, 256), default=32)
    args = p.parse_args(); sys.path.append(args.mnn_python_path)
    import numpy as np
    import MNN
    import _tools
    torch.set_num_threads(4); torch.manual_seed(2307)
    root = ROOT / 'models/levo-mixed-codec'
    info = json.loads((root / 'source-header.json').read_text())
    Config, _, source = upstream_classes(ROOT / 'reference/LeVo')
    cfg = Config(n_embd=1200, n_layer=39, n_head=30, n_positions=1000,
                 use_cache=False, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
    cfg._attn_implementation = 'eager'
    block = source.GPT2Block(cfg, layer_idx=0).eval()
    prefix = 'cfm_wrapper.estimator.h.0.'
    block.load_state_dict({k.removeprefix(prefix): tensor_from_parts(k, info, root/'parts')
                           for k in info['tensors'] if k.startswith(prefix)}, strict=True)
    model = Block(block).eval()
    frames = args.frames
    x = torch.randn(1, frames, 1200) * .2; modulation = torch.randn(1, 7200) * .1
    mask = torch.zeros(1, 1, frames, frames); mask[..., -2:] = torch.finfo(torch.float32).min
    with torch.inference_mode(): reference = model(x, modulation, mask)
    # Replace complex arithmetic with algebraically equivalent real pairs.
    source.precompute_freqs_cis = real_freqs; source.apply_rotary_emb = real_rope
    with torch.inference_mode(): rewritten = model(x, modulation, mask)
    torch.testing.assert_close(rewritten, reference, atol=1e-6, rtol=1e-5)
    suffix = '' if frames == 32 else f'-{frames}'
    outdir = ROOT / f'models/dit-block{suffix}'; outdir.mkdir(exist_ok=True)
    np.savez(outdir / 'fixture.npz', x=x.numpy(), modulation=modulation.numpy(), mask=mask.numpy(),
             reference=reference.detach().numpy())
    onnx = outdir / 'real-block0.onnx'
    torch.onnx.export(model, (x, modulation, mask), str(onnx), input_names=['x','modulation','mask'],
                      output_names=['output'], opset_version=17, dynamo=False)
    # Fixed-shape simplification is required by the Windows MNN converter:
    # the unsimplified Shape/Einsum path triggered access violation 0xC0000005.
    import onnx as onnxlib
    import onnxsim
    simplified, valid = onnxsim.simplify(onnxlib.load(str(onnx)))
    if not valid: raise RuntimeError('ONNX simplification validation failed')
    onnx = outdir / 'real-block0-simplified.onnx'
    onnxlib.save(simplified, str(onnx))
    import onnxruntime as ort
    ort_output = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider']).run(None,
                  {'x':x.numpy(),'modulation':modulation.numpy(),'mask':mask.numpy()})[0]
    ort_l2 = float(np.linalg.norm((ort_output-reference.detach().numpy()).astype(np.float64))/np.linalg.norm(reference.detach().numpy().astype(np.float64)))
    rows = []
    values = {'x': x.numpy(), 'modulation': modulation.numpy(), 'mask': mask.numpy()}
    expected = reference.detach().numpy()
    for bits in (0, 8, 4):
        target = outdir / f'real-block0-w{bits or 32}.mnn'
        command = ['MNNConvert', '-f', 'ONNX', '--modelFile', str(onnx), '--MNNModel', str(target),
                   '--bizCode', 'levo-harmony-cpu-block-test', '--saveExternalData']
        if bits: command += ['--weightQuantBits', str(bits), '--weightQuantBlock', '64']
        _tools.mnnconvert(command)
        if not target.is_file(): raise RuntimeError('MNN converter did not produce model')
        net = MNN.Interpreter(str(target))
        session = net.createSession({'backend':'CPU','numThread':4,'precision':1,'memory':2})
        times = []
        for repeat in range(6):
            begin = time.perf_counter()
            for name, val in values.items():
                tensor = MNN.Tensor(tuple(val.shape), MNN.Halide_Type_Float,
                                    np.ascontiguousarray(val), MNN.Tensor_DimensionType_Caffe)
                if not net.getSessionInput(session, name).copyFrom(tensor): raise RuntimeError('Input copy failed')
            code = net.runSession(session)
            if code not in (None, 0): raise RuntimeError('MNN run failed')
            output = np.asarray(net.getSessionOutput(session, 'output').getNumpyData()).copy()
            times.append(time.perf_counter()-begin)
        if not np.isfinite(output).all(): raise RuntimeError('Non-finite MNN output')
        weight = Path(str(target)+'.weight')
        rows.append({'requested_weight_bits': bits or 32, 'model_bytes': target.stat().st_size,
                     'external_bytes': weight.stat().st_size if weight.exists() else 0,
                     'relative_l2': float(np.linalg.norm((output-expected).astype(np.float64))/np.linalg.norm(expected.astype(np.float64))),
                     'first_seconds':times[0], 'warm_median_seconds':statistics.median(times[1:]),'finite':True})
        print(json.dumps(rows[-1]), flush=True)
    report = {'scope':f'Windows CPU, 4 requested threads, actual pretrained block0, fixed{frames}frames; not phone RTF',
              'frames':frames,
              'onnx_fixed_shape_simplification_required':True,
              'onnxruntime_relative_l2':ort_l2,
              'real_rope_rewrite_max_abs': float((reference-rewritten).abs().max()),'rows':rows,
              'full_dit_validated':False,'real_song_generated':False}
    (ROOT/f'reports/real-dit-block-mnn{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf8')


if __name__ == '__main__': main()
