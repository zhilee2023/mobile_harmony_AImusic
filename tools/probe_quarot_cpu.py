"""CPU-only QuaRot-style projection experiment, not a full rotated checkpoint.

Uses actual 26-layer Qwen weights and activations, fixed randomized orthogonal
Hadamard transforms, symmetric W4/g64 RTN and per-token activation quantization.
No CUDA dependency, no model mutation and no claim of native ARM INT4 kernels.
"""
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("温暖钢琴流行", "轻快民谣", "安静电子氛围", "明亮摇滚", "柔和爵士", "轻盈舞曲", "抒情弦乐")
LINES = (
    "晚风掠过窗台，我把明天轻轻展开。", "雨停在街角，你的笑留在云外。",
    "沿着河岸慢慢走，灯火把归途照亮。", "远处传来汽笛，心事随晚霞飘扬。",
    "把旧日装进背包，朝着晴空奔跑。", "星光落在肩膀，陪我穿过漫长街道。",
    "听见清晨的鸟鸣，新的故事正在苏醒。", "一杯热茶的温度，让思念慢慢停驻。",
    "山间吹来一阵风，带着自由的梦。", "海浪亲吻沙滩，留下我们的答案。",
    "月亮经过屋檐，照着未寄出的信。", "列车驶过田野，远方和此刻相连。",
    "在人群中回头，熟悉的声音仍温柔。", "雪花落进掌心，愿望悄悄有了回音。",
    "走过小小巷口，春天已经等了很久。", "你说别怕告别，每次出发都有新世界。",
    "云朵借给我翅膀，飞向更明亮的地方。", "钟声绕过广场，唤醒沉睡的梦想。",
    "把疲惫交给夜色，把勇气留给日出。", "窗外树影摇晃，时光缓缓流淌。",
    "操场上的晚霞，收藏着青春的回答。", "蓝色雨伞下面，我们分享同一片天。",
    "记得那年的夏夜，蝉鸣和笑声重叠。", "跨过这座小桥，让所有烦恼退潮。",
    "书页间夹着花香，故事还没有散场。", "风筝飞过山坡，我唱起自由的歌。",
    "清晨第一束微光，落在旅人的行囊。", "晚安轻轻说出口，梦里再一起远游。",
)


def fwht(x):
    width = x.shape[-1]
    assert width and width & (width - 1) == 0
    y = x.contiguous()
    size = 1
    while size < width:
        z = y.reshape(*y.shape[:-1], -1, 2, size)
        a, b = z[..., 0, :], z[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).flatten(-3)
        size *= 2
    return y / width ** .5


def paley12():
    q = torch.empty(11, 11)
    residues = {i * i % 11 for i in range(1, 11)}
    for i in range(11):
        for j in range(11):
            d = (i - j) % 11
            q[i, j] = 0 if d == 0 else 1 if d in residues else -1
    h = torch.ones(12, 12)
    h[1:, 1:] = q - torch.eye(11)
    h /= 12 ** .5
    torch.testing.assert_close(h @ h.T, torch.eye(12), atol=1e-6, rtol=1e-6)
    return h


def rotate(x, signs, h12):
    y = x * signs
    if y.shape[-1] == 1024:
        return fwht(y)
    assert y.shape[-1] == 3072
    shape = y.shape
    y = fwht(y.reshape(-1, 12, 256))
    return torch.einsum("bjt,jk->bkt", y, h12).reshape(shape)


def quantize(x, bits, clip=1., group=None):
    if bits == 16:
        return x
    shape = x.shape
    v = x.reshape(x.shape[0], -1, group) if group else x
    limit = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * clip
    qmax = 2 ** (bits - 1) - 1
    scale = limit / qmax
    return (torch.round(v / scale).clamp(-qmax - 1, qmax) * scale).reshape(shape)


def relative(a, b):
    return float((a - b).double().norm() / b.double().norm().clamp_min(1e-30))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2307)
    target = ROOT / "reports/quarot-projection-cpu-20261031.json"
    if target.exists():
        raise ValueError("Preserve previous projection experiment")
    started = time.perf_counter()
    source = ROOT / "models/qwen3-0.6b-26layers-unrecovered"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/qwen3-0.6b", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32,
                                                local_files_only=True, attn_implementation="eager").eval()
    names = [f"model.layers.{layer}.{part}" for layer in (0, 12, 25)
             for part in ("self_attn.q_proj", "mlp.down_proj")]
    modules = dict(model.named_modules())
    buffers = {name: {"calibration": [], "heldOut": []} for name in names}
    split = "calibration"
    handles = []
    for name in names:
        def hook(module, args, key=name):
            value = args[0].detach().float().reshape(-1, args[0].shape[-1])
            buffers[key][split].append(value[-8:].clone())
        handles.append(modules[name].register_forward_pre_hook(hook))
    prompts = []
    with torch.inference_mode():
        for i, line in enumerate(LINES):
            split = "calibration" if i < 20 else "heldOut"
            text = f"风格：{STYLES[i % len(STYLES)]}。中文歌曲，四拍，清晰人声。歌词：{line}"
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            out = model(**encoded, use_cache=True)
            cache = out.past_key_values
            for step in range(2):
                next_id = out.logits[:, -1:].argmax(dim=-1)
                out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            prompts.append({"id": i, "split": split, "text": text,
                            "prefillTokens": encoded["input_ids"].numel(), "decodeSteps": 2})
            print(f"Captured {i + 1}/28 real activation cases", flush=True)
    weights = {name: modules[name].weight.detach().float().clone() for name in names}
    for handle in handles:
        handle.remove()
    del cache, out, model, modules
    gc.collect()
    h12 = paley12()
    rows = []
    with torch.inference_mode():
        for index, name in enumerate(names):
            w = weights[name]
            calibration = torch.cat(buffers[name]["calibration"])
            held_out = torch.cat(buffers[name]["heldOut"])
            gen = torch.Generator().manual_seed(2307 + index)
            signs = torch.randint(0, 2, (w.shape[1],), generator=gen).float() * 2 - 1
            wr = rotate(w, signs, h12)
            cr = rotate(calibration, signs, h12)
            hr = rotate(held_out, signs, h12)
            reference = held_out @ w.T
            invariant_error = relative(hr @ wr.T, reference)
            assert invariant_error < 1e-5
            variants = []
            for rotated in (False, True):
                weight = wr if rotated else w
                cx, hx = (cr, hr) if rotated else (calibration, held_out)
                qw = quantize(weight, 4, group=64)
                for a_bits in (16, 8, 4):
                    best_clip = 1.
                    if a_bits < 16:
                        expected = calibration @ w.T
                        scores = [(relative(quantize(cx, a_bits, clip) @ qw.T, expected), clip)
                                  for clip in (.7, .8, .9, .95, 1.)]
                        _, best_clip = min(scores)
                    output = quantize(hx, a_bits, best_clip) @ qw.T
                    variants.append({"rotated": rotated, "weightBits": 4, "activationBits": a_bits,
                                     "groupSize": 64, "activationClip": best_clip,
                                     "heldOutOutputRelativeL2": relative(output, reference),
                                     "finite": bool(torch.isfinite(output).all())})
            rows.append({"module": name, "weightShape": list(w.shape),
                         "calibrationRows": len(calibration), "heldOutRows": len(held_out),
                         "rotationFloatInvarianceL2": invariant_error, "variants": variants})
            print(name, json.dumps(variants[-1]), flush=True)
    report = {"scope": "CPU numerical projection experiment, not full QuaRot checkpoint, native INT4 activation kernel, whole-model quality or phone speed",
              "algorithm": "Fixed randomized normalized Hadamard; full H1024 or PaleyH12 x H256, W4/g64 symmetric RTN; per-token activation clipping calibrated on 20 original prompts",
              "calibrationCases": 20, "heldOutCases": 8, "source": str(source),
              "musicTrained": False, "musicHeadEvaluated": False,
              "includesPrefillAndTwoDecodeActivationsPerCase": True,
              "fullResidualAndAttentionRotationsApplied": False,
              "prompts": prompts, "rows": rows, "totalSeconds": time.perf_counter() - started}
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("QuaRot-style six-projection CPU experiment complete", flush=True)


if __name__ == "__main__":
    main()
