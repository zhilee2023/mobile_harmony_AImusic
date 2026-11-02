"""Explicit architecture and I/O contracts shared by converters and packaging."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = ('llm', 'dit', 'vae')
PROFILE_IDS = ('public-mixed-v2', 'student-mixed-v1')


def load_profile(value):
    path = Path(value)
    if not path.is_file():
        path = ROOT / 'configs' / 'profiles' / f'{value}.json'
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    if result.get('schemaVersion') != 1 or result.get('family') not in ('public', 'small'):
        raise ValueError('Unsupported architecture profile schema/family')
    expected = json.loads((ROOT / 'configs/profiles/public-mixed-v2.json').read_text(encoding='utf-8'))['contract']
    if result.get('contract') != expected:
        raise ValueError('Runtime codec contract differs: supply a new runtime adapter, not just new weights')
    if set(result.get('modules', {})) != set(MODULES):
        raise ValueError('A profile must define llm, dit and vae')
    for role, module in result['modules'].items():
        if module['precision'] not in ('high', 'low') or not 1 <= module['threads'] <= 16:
            raise ValueError(f'Invalid runtime policy for {role}')
    return result


def io_contract(role, profile):
    c = profile['contract']; f = c['windowFrames']
    if role == 'dit':
        return {'inputs': [2, f, c['ditInputWidth']], 'timestep': [2], 'attention_mask': [2, 1, f, f]}, {'velocity': [2, f, c['latentChannels']]}
    if role == 'vae':
        return {'latent': [1, c['latentChannels'], f]}, {'pcm': [1, c['audioChannels'], f * c['samplesPerLatent']]}
    if role == 'llm':
        # This is the MNN exporter graph contract, not arbitrary HF KV ONNX.
        return {name: None for name in ('input_ids', 'attention_mask', 'position_ids', 'logits_index')}, {'logits': None}
    raise ValueError(f'Unknown module: {role}')


def canonical_graph(role, profile):
    return {'llm': ('student' if profile['family'] == 'small' else 'lm') + '/llm.mnn',
            'dit': 'dit/full-w8.mnn', 'vae': 'vae/vae-256-w8.mnn'}[role]


def validate_onnx(path, role, profile):
    import onnx
    graph = onnx.load(str(path), load_external_data=False).graph
    initializers = {x.name for x in graph.initializer}
    def shape(value):
        return [d.dim_value if d.HasField('dim_value') else None for d in value.type.tensor_type.shape.dim]
    actual_inputs = {x.name: shape(x) for x in graph.input if x.name not in initializers}
    actual_outputs = {x.name: shape(x) for x in graph.output}
    inputs, outputs = io_contract(role, profile)
    if set(actual_inputs) != set(inputs) or not set(outputs).issubset(actual_outputs):
        raise ValueError(f'{role} ONNX names mismatch: {actual_inputs} -> {actual_outputs}; expected {inputs} -> {outputs}')
    if role != 'llm':
        for expected, actual in ((inputs, actual_inputs), (outputs, actual_outputs)):
            for name, dims in expected.items():
                if actual[name] != dims:
                    raise ValueError(f'{role}/{name}: expected fixed shape {dims}, got {actual[name]}')
    else:
        embedding = actual_inputs['input_ids']; logits = actual_outputs['logits']
        info = next(x for x in graph.input if x.name == 'input_ids')
        if info.type.tensor_type.elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16) or len(embedding) != 3 or embedding[-1] != profile['modules']['llm']['hidden']:
            raise ValueError('MNN LM input_ids must be floating embeddings [sequence,1,hidden], not text IDs')
        if not logits or (logits[-1] is not None and logits[-1] != profile['modules']['llm']['outputClasses']):
            raise ValueError('LM ONNX output head must have 16385 music classes')
    return {'inputs': actual_inputs, 'outputs': actual_outputs}
