"""Untrained CosyVoice-sized music-interface DiT for CPU structure benchmarks.

Retains LeVo attention/AdaLN/RoPE scaffolding, not CosyVoice speech weights.
This artifact has no claim of trained music or two-step generation quality.
"""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from restore_mixed_dit import upstream_classes
from export_real_dit_block import real_freqs, real_rope

ROOT = Path(__file__).resolve().parents[1]
MNN_PACKAGES = r"C:\ace-step-mnn-ios\.venv\Lib\site-packages"


class SmallMusicDit(nn.Module):
    def __init__(self):
        super().__init__()
        config_cls, model_cls, _ = upstream_classes(ROOT / "reference/LeVo")
        config = config_cls(vocab_size=1, n_positions=256, n_layer=22, n_head=16,
                            n_embd=1024, n_inner=2048, use_cache=False,
                            resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
        config._attn_implementation = "eager"
        self.input_projection = nn.Linear(1200, 1024)
        self.estimator = model_cls(config)
        self.output_projection = nn.Linear(1024, 64)

    def forward(self, inputs, timestep, attention_mask):
        hidden = self.estimator(inputs_embeds=self.input_projection(inputs),
                                time_step=timestep, attention_mask=attention_mask,
                                use_cache=False).last_hidden_state
        return self.output_projection(hidden)


def l2(a, b):
    return float(np.linalg.norm((a - b).astype(np.float64)) /
                 max(np.linalg.norm(b.astype(np.float64)), 1e-30))


def main():
    import onnx
    import onnxsim
    torch.set_num_threads(4)
    torch.manual_seed(2307)
    out = ROOT / "models/small-music-dit-22-1024"
    out.mkdir(exist_ok=True)
    if (out / "full-w8.mnn").exists():
        raise ValueError("Preserve existing structure-probe model")
    started = time.perf_counter()
    model = SmallMusicDit().eval()
    fixture = np.load(ROOT / "models/full-dit-256/fixture.npz")
    values = {name: np.ascontiguousarray(fixture[name])
              for name in ("inputs", "timestep", "attention_mask")}
    inputs = tuple(torch.from_numpy(value) for value in values.values())
    with torch.inference_mode():
        expected = model(*inputs).numpy().copy()
    _, _, upstream = upstream_classes(ROOT / "reference/LeVo")
    frequencies = real_freqs(64, 256)
    upstream.precompute_freqs_cis = lambda dim, end, constant=10000.: frequencies
    upstream.apply_rotary_emb = real_rope
    with torch.inference_mode():
        rewritten = model(*inputs).numpy().copy()
    assert l2(rewritten, expected) < 1e-4
    print("Small music-interface 22x1024/2048 structure initialized; exporting", flush=True)
    shapes = out / "shapes.onnx"
    torch.onnx.export(model, inputs, str(shapes), input_names=list(values),
                      output_names=["velocity"], opset_version=17, dynamo=False,
                      export_params=False, do_constant_folding=False,
                      keep_initializers_as_inputs=True)
    graph, valid = onnxsim.simplify(onnx.load(str(shapes)), check_n=0)
    assert valid
    named = dict(model.named_parameters())
    named.update(dict(model.named_buffers()))
    retained = []
    attached_parameters = 0
    weights_path = out / "weights.bin"
    with weights_path.open("wb") as stream:
        for item in graph.graph.input:
            if item.name in values:
                retained.append(item)
                continue
            value = named[item.name].detach().cpu().numpy()
            data = np.ascontiguousarray(value).tobytes()
            offset = stream.tell()
            stream.write(data)
            attached_parameters += value.size
            tensor = onnx.TensorProto()
            tensor.name = item.name
            tensor.data_type = onnx.helper.np_dtype_to_tensor_dtype(value.dtype)
            tensor.dims.extend(value.shape)
            tensor.data_location = onnx.TensorProto.EXTERNAL
            for key, val in (("location", weights_path.name), ("offset", str(offset)),
                             ("length", str(len(data)))):
                entry = tensor.external_data.add()
                entry.key, entry.value = key, val
            graph.graph.initializer.append(tensor)
    del graph.graph.input[:]
    graph.graph.input.extend(retained)
    external = out / "small-external.onnx"
    onnx.save_model(graph, str(external))
    for name, value in {**values, "reference": expected}.items():
        np.ascontiguousarray(value, dtype="<f4").tofile(out / f"{name}.f32")
    del graph, named, model, inputs
    gc.collect()
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(external), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, values)[0]
    ort_error = l2(actual, expected)
    assert np.isfinite(actual).all() and ort_error < 1e-4
    del session
    gc.collect()
    target = out / "full-w8.mnn"
    subprocess.run([sys.executable, str(ROOT / "tools/mnn_convert_once.py"),
                    "--mnn-python-path", MNN_PACKAGES, "--", "-f", "ONNX",
                    "--modelFile", str(external), "--MNNModel", str(target),
                    "--bizCode", "untrained-small-music-structure-probe",
                    "--saveExternalData", "--weightQuantBits", "8",
                    "--weightQuantBlock", "64"], check=True)
    sys.path.insert(0, MNN_PACKAGES)
    import MNN
    net = MNN.Interpreter(str(target))
    session = net.createSession({"backend": "CPU", "numThread": 4,
                                  "precision": 1, "memory": 2})
    for name, value in values.items():
        tensor = MNN.Tensor(tuple(value.shape), MNN.Halide_Type_Float, value,
                            MNN.Tensor_DimensionType_Caffe)
        assert net.getSessionInput(session, name).copyFrom(tensor)
    tick = time.perf_counter()
    net.runSession(session)
    actual = np.asarray(net.getSessionOutput(session, "velocity").getNumpyData()).copy()
    mnn_forward_seconds = time.perf_counter() - tick
    assert actual.shape == (2, 256, 64) and np.isfinite(actual).all()
    actual.astype("<f4").tofile(out / "w8-reference.f32")
    files = {}
    for path in out.iterdir():
        if path.is_file() and path.name not in ("metadata.json",):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    report = {"architectureOnly": True, "music_trained": False,
              "weights_trained": False, "initialization": "random seed2307",
              "architecture": "LeVo attention/AdaLN/RoPE scaffold at CosyVoice-sized dimensions; not an exact CosyVoice architecture or speech checkpoint",
              "layers": 22, "hidden": 1024, "ffn": 2048, "heads": 16,
              "inputProjection": [1200, 1024], "outputProjection": [1024, 64],
              "exportedParameters": int(attached_parameters), "weightBits": 8,
              "ortVsTorchL2": ort_error, "mnnVsTorchL2": l2(actual, expected),
              "pcMnnSeconds": mnn_forward_seconds,
              "totalExportSeconds": time.perf_counter() - started, "files": files}
    (out / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
