"""Tensor-only checkpoints; no unsafe pickle fallback or silent partial load."""
from pathlib import Path
import pickle
import zipfile


def format_of(path, requested='auto'):
    if requested != 'auto':
        return requested
    path = Path(path)
    if path.is_dir():
        return 'hf'
    return {'.onnx': 'onnx', '.mnn': 'mnn', '.safetensors': 'checkpoint',
            '.pt': 'checkpoint', '.pth': 'checkpoint', '.torch': 'checkpoint'}.get(path.suffix.lower(), 'unknown')


def load_state(path, state_key=None, strip_prefix=None):
    import torch
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        try:
            # Modern torch.save archives can be mapped: selecting one module
            # from a multi-GB checkpoint need not materialize all other weights.
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=zipfile.is_zipfile(path))
        except pickle.UnpicklingError as error:
            raise ValueError('Not a tensor-only checkpoint. Export state_dict, or explicitly select format=torchscript; pickle execution is not enabled') from error
    if state_key:
        for key in state_key.split('.'):
            state = state[key]
    elif isinstance(state, dict) and set(state) == {'state_dict'}:
        state = state['state_dict']
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected a tensor state_dict; use stateKey for a training checkpoint, or format=torchscript for TorchScript')
    if strip_prefix:
        selected = {k[len(strip_prefix):]: v for k, v in state.items() if k.startswith(strip_prefix)}
        if not selected:
            raise ValueError(f'No checkpoint keys match explicit prefix {strip_prefix!r}')
        state = selected
    return state


def strict_load(model, state):
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys()); extra = sorted(state.keys() - expected.keys())
    shapes = {key: {'expected': list(expected[key].shape), 'actual': list(state[key].shape)}
              for key in expected.keys() & state.keys() if expected[key].shape != state[key].shape}
    if missing or extra or shapes:
        raise ValueError(f'Checkpoint mismatch: missing={missing[:12]}, unexpected={extra[:12]}, shapes={str(shapes)[:1600]}')
    model.load_state_dict(state, strict=True, assign=True)
    return {'strict': True, 'tensors': len(state), 'parameters': sum(p.numel() for p in model.parameters())}


def inspect_source(path, requested='auto', state_key=None, strip_prefix=None):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    kind = format_of(path, requested)
    result = {'path': str(path), 'format': kind}
    if kind == 'checkpoint':
        state = load_state(path, state_key, strip_prefix)
        result.update(tensors=len(state), parameters=sum(t.numel() for t in state.values()),
                      shapes={k: list(v.shape) for k, v in list(state.items())[:20]})
    elif kind == 'hf':
        from .files import read_json
        result['config'] = read_json(path / 'config.json')
    elif kind == 'onnx':
        import onnx
        model = onnx.load(str(path), load_external_data=False)
        result.update(inputs=[x.name for x in model.graph.input], outputs=[x.name for x in model.graph.output], nodes=len(model.graph.node))
    elif kind == 'torchscript':
        import torch
        model = torch.jit.load(str(path), map_location='cpu')
        result['schema'] = str(model.forward.schema)
    elif kind != 'mnn':
        raise ValueError('Unsupported source; select checkpoint, hf, torchscript, onnx or mnn')
    return result
