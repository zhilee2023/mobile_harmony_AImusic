"""One component conversion per invocation, with reproducible provenance."""
import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .contracts import ROOT, load_profile, io_contract, validate_onnx
from .files import read_json, write_json, sha256
from .checkpoints import format_of


def fixture(role, profile):
    import numpy as np
    shapes, _ = io_contract(role, profile); rng = np.random.default_rng(2307)
    if role == 'llm':
        raise ValueError('LM uses the MNN exporter KV interface, not a generic stateless fixture')
    values = {name: rng.normal(0, .1, shape).astype('float32') for name, shape in shapes.items()}
    if role == 'dit':
        values['timestep'].fill(.5); values['attention_mask'].fill(1)
    return values


def export_codec_onnx(model, role, profile, destination):
    import numpy as np
    import torch
    import onnx
    import onnxsim
    values = fixture(role, profile); inputs = tuple(torch.from_numpy(x) for x in values.values())
    with torch.inference_mode():
        reference = model(*inputs).detach().float().numpy().copy()
    _, output_shapes = io_contract(role, profile)
    if list(reference.shape) != list(output_shapes.values())[0] or not np.isfinite(reference).all():
        raise ValueError('Torch model violates component output contract')
    shapes = destination / 'shapes.onnx'
    torch.onnx.export(model, inputs, str(shapes), input_names=list(values), output_names=list(output_shapes),
                      opset_version=17, dynamo=False, export_params=False, do_constant_folding=False, keep_initializers_as_inputs=True)
    graph, valid = onnxsim.simplify(onnx.load(str(shapes)), check_n=0)
    if not valid:
        raise ValueError('Fixed-shape ONNX simplification failed')
    named = dict(model.named_parameters()); named.update(dict(model.named_buffers())); retained = []
    weights = destination / 'onnx-weights.bin'
    with weights.open('wb') as out:
        for item in graph.graph.input:
            if item.name in values:
                retained.append(item); continue
            if item.name not in named:
                raise ValueError(f'Unmapped ONNX parameter input: {item.name}')
            value = np.ascontiguousarray(named[item.name].detach().cpu().float().numpy()); data = value.tobytes(); offset = out.tell(); out.write(data)
            tensor = onnx.TensorProto(name=item.name, data_type=onnx.TensorProto.FLOAT, dims=value.shape, data_location=onnx.TensorProto.EXTERNAL)
            for key, val in (('location', weights.name), ('offset', str(offset)), ('length', str(len(data)))):
                entry = tensor.external_data.add(); entry.key = key; entry.value = val
            graph.graph.initializer.append(tensor)
    del graph.graph.input[:]; graph.graph.input.extend(retained)
    result = destination / 'model.onnx'; onnx.save_model(graph, str(result))
    np.savez(destination / 'torch-reference.npz', **values, reference=reference)
    return result


def convert_onnx(source, target, role, recipe, mnn_packages):
    args = ['-f', 'ONNX', '--modelFile', str(source), '--MNNModel', str(target), '--bizCode', 'levo-model-bundle', '--saveExternalData']
    if role == 'llm':
        # Compatible exported LM graphs may contain exporter-defined operators.
        args += ['--transformerFuse', '--allowCustomOp']
    args += (['--fp16'] if recipe['weightBits'] == 16 else ['--weightQuantBits', str(recipe['weightBits']), '--weightQuantBlock', str(recipe['groupSize'])])
    subprocess.run([sys.executable, str(ROOT / 'tools/mnn_convert_once.py'), '--mnn-python-path', str(mnn_packages), '--', *args], check=True)
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError('Converter returned without a nonempty MNN graph')
    return args


def validate_codec(graph, onnx_path, role, profile, mnn_packages, tolerance):
    import numpy as np
    import onnxruntime as ort
    values = fixture(role, profile); output_name = next(iter(io_contract(role, profile)[1]))
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=['CPUExecutionProvider'])
    reference = session.run([output_name], values)[0]; del session; gc.collect()
    sys.path.insert(0, str(mnn_packages)); import MNN
    net = MNN.Interpreter(str(graph)); runtime = net.createSession({'backend': 'CPU', 'numThread': 4, 'precision': 1, 'memory': 2})
    for name, value in values.items():
        host = MNN.Tensor(tuple(value.shape), MNN.Halide_Type_Float, value, MNN.Tensor_DimensionType_Caffe)
        if not net.getSessionInput(runtime, name).copyFrom(host):
            raise ValueError(f'MNN input copy failed: {name}')
    error = net.runSession(runtime)
    actual = np.asarray(net.getSessionOutput(runtime, output_name).getNumpyData()).copy()
    if actual.shape == reference.shape + (1,):
        actual = actual[..., 0]
    if error not in (None, 0) or actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Converted model failed finite-output/shape validation')
    l2 = float(np.linalg.norm((actual.astype('float64') - reference)) / max(np.linalg.norm(reference.astype('float64')), 1e-30))
    if l2 > tolerance:
        raise ValueError(f'MNN conversion L2 {l2:.6g} exceeds configured {tolerance}; output not accepted')
    torch_reference = graph.parent / 'torch-reference.npz'
    report = {'finite': True, 'mnnVsOnnxL2': l2, 'tolerance': tolerance, 'outputShape': list(actual.shape), 'fixture': 'seed2307 synthetic; no listening/quality validation'}
    if torch_reference.is_file():
        original = np.load(torch_reference)['reference']
        export_l2 = float(np.linalg.norm(reference.astype('float64') - original) / max(np.linalg.norm(original.astype('float64')), 1e-30))
        if not np.isfinite(original).all() or export_l2 > .001:
            raise ValueError(f'ONNX differs from strict-loaded Torch model: {export_l2}')
        report['onnxVsTorchL2'] = export_l2
    return report


def export_lm(source, destination, profile, exporter_root, mnn_packages):
    from .torch_models import language_model
    model, config_dir, strict = language_model(profile, source)
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(exporter_root)); sys.path.insert(0, str(mnn_packages))
    from llmexport import LlmExporter, build_args
    parser = argparse.ArgumentParser(); build_args(parser)
    recipe = profile['modules']['llm']
    options = parser.parse_args(['--path', str(config_dir), '--tokenizer_path', str(source.get('tokenizerDirectory', config_dir)),
                                '--dst_path', str(destination), '--export', 'mnn', '--quant_bit', str(recipe['weightBits']),
                                '--lm_quant_bit', str(recipe['headBits']), '--quant_block', str(recipe['groupSize']), '--seperate_embed'])
    loader = AutoModelForCausalLM.from_pretrained
    AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: model
    try:
        exporter = LlmExporter(options)
    finally:
        AutoModelForCausalLM.from_pretrained = loader
    exporter.llm_config['vocab_size'] = 16385
    exporter.export('mnn')
    return strict


def prepare_component(job, output, mnn_packages=None, exporter_root=None):
    profile = load_profile(job['profile']); role = job['module']
    if role not in profile['modules']:
        raise ValueError('module must be llm, dit or vae')
    source = dict(job['source']); path = Path(source['path']).resolve(); source['path'] = str(path)
    if not path.exists():
        raise FileNotFoundError(path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous component; use a new output directory')
    # Leave failed work in a marked directory; never publish component.json on failure.
    output.mkdir(parents=True); write_json(output / 'preparing.json', job)
    kind = format_of(path, source.get('format', 'auto')); target = output / ('llm.mnn' if role == 'llm' else 'model.mnn')
    metadata = {'schemaVersion': 1, 'profile': profile['id'], 'contract': profile['contract']['id'], 'module': role,
                'sourceFormat': kind, 'sourceSha256': sha256(path) if path.is_file() else None,
                'recipe': profile['modules'][role], 'training': job.get('training', 'unknown'), 'validation': {'status': 'not-run'}}
    allowed_recipe = {'weightBits', 'headBits', 'groupSize', 'threads', 'precision'}
    if set(job.get('recipe', {})) - allowed_recipe:
        raise ValueError('Unsupported recipe option; activation/KV quantization needs a separate validated adapter')
    metadata['recipe'].update(job.get('recipe', {}))
    if metadata['recipe']['weightBits'] not in (4, 8, 16) or not 1 <= metadata['recipe']['threads'] <= 16 or metadata['recipe']['precision'] not in ('high', 'low'):
        raise ValueError('Supported recipe: W4/W8/FP16, CPU high/low, requested threads 1..16')
    if role == 'llm' and kind == 'onnx':
        # The generic converter quantizes the whole graph uniformly. Mixed
        # W4-body/W8-head needs the official HF/PT exporter weight-op rebuild.
        bits = job.get('recipe', {}).get('weightBits', 8)
        metadata['recipe']['weightBits'] = metadata['recipe']['headBits'] = bits
        metadata['quantizationScope'] = 'Uniform ONNX conversion; mixed head precision requires HF/PT export'
    if kind == 'mnn':
        shutil.copy2(path, target)
        external = Path(str(path) + '.weight')
        if external.is_file(): shutil.copy2(external, Path(str(target) + '.weight'))
        metadata['validation'] = {'status': 'imported-runtime-model', 'numericParityVerified': False}
        metadata['quantizationScope'] = 'Declared by importer; converted kernels/bitwidth not inferred from filename'
    elif kind in ('hf', 'checkpoint') and role == 'llm':
        if not exporter_root or not mnn_packages:
            raise ValueError('LM export needs --exporter-root and --mnn-packages')
        metadata['strictLoad'] = export_lm(source, output, profile, exporter_root, mnn_packages)
        metadata['validation'] = {'status': 'strict-checkpoint-exported', 'numericParityVerified': False, 'next': 'phone LM prefill+two decode probe; full quality not inferred'}
    else:
        if not mnn_packages:
            raise ValueError('Conversion needs --mnn-packages')
        onnx_path = path
        if kind in ('checkpoint', 'torchscript'):
            import torch
            torch.set_num_threads(4)
            if role == 'llm':
                raise ValueError('TorchScript LM needs an MNN-exporter-compatible ONNX and its llm_config/embedding/tokenizer companions')
            if kind == 'checkpoint':
                from .torch_models import codec_model
                model, metadata['strictLoad'] = codec_model(role, profile, source)
            else:
                model = torch.jit.load(str(path), map_location='cpu').float().eval()
            onnx_path = export_codec_onnx(model, role, profile, output); del model; gc.collect()
        elif kind != 'onnx':
            raise ValueError('Expected checkpoint, torchscript, ONNX, HF or MNN')
        metadata['onnxContract'] = validate_onnx(onnx_path, role, profile)
        metadata['converterArgs'] = convert_onnx(onnx_path, target, role, profile['modules'][role], mnn_packages)
        if role != 'llm':
            metadata['validation'] = {'status': 'passed', **validate_codec(target, onnx_path, role, profile, mnn_packages, job.get('maxRelativeL2', .05 if role == 'vae' else .1))}
        else:
            metadata['validation'] = {'status': 'exporter-onnx-imported', 'numericParityVerified': False}
    companions = job.get('companions', {})
    for name, original in companions.items():
        if Path(name).name != name or name in (target.name, target.name + '.weight', 'component.json'):
            raise ValueError('Companions must be unique local filenames; graph/weight cannot be overridden')
        shutil.copy2(original, output / name)
    if role == 'llm':
        needed = ('llm_config.json', 'tokenizer.mtok', 'embeddings_bf16.bin')
        missing = [name for name in needed if not (output / name).is_file()]
        if missing:
            raise ValueError(f'LM runtime companions required: {missing}; full input embedding must include music/condition/style rows for this profile')
        cfg = read_json(output / 'llm_config.json'); recipe = profile['modules']['llm']
        if cfg.get('hidden_size') != recipe['hidden'] or cfg.get('layer_nums') != recipe['layers'] or cfg.get('vocab_size') != 16385:
            raise ValueError('llm_config.json differs from profile/output vocabulary')
    files = [target] + ([Path(str(target) + '.weight')] if Path(str(target) + '.weight').exists() else [])
    if role == 'llm':
        files += [output / name for name in ('llm_config.json', 'tokenizer.mtok', 'embeddings_bf16.bin')]
    metadata['graph'] = target.name
    metadata['files'] = [{'name': f.name, 'bytes': f.stat().st_size, 'sha256': sha256(f)} for f in files]
    write_json(output / 'component.json', metadata)
    return metadata
