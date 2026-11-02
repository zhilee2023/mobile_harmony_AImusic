"""Build immutable, self-describing packages. Baselines are never overwritten."""
import re
import shutil
import struct
import json
from pathlib import Path
from .contracts import MODULES, load_profile, canonical_graph
from .files import inside, inventory, read_json, write_json, sha256

ID_PATTERN = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}\Z')
AUX_ELEMENTS = {'codebook': 16384 * 32, 'weight': 1024 * 32, 'bias': 1024, 'mask': 3 * 48, 'mean': 64, 'std': 64}


def model_id(value):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value) or '..' in value or value == 'builtin' or '.incoming-' in value:
        raise ValueError('modelId must be 1..80 letters/digits/dots/dashes/underscores, without .. or reserved builtin/.incoming- names')
    return value


def sidecars(profile):
    if profile['family'] == 'small':
        files = ['student/' + x for x in ('config.json', 'llm_config.json', 'tokenizer.mtok', 'embeddings_bf16.bin', 'diagnostic-student.json', 'prefix.json')]
    else:
        files = ['lm/llm_config.json', 'lm/tokenizer.mtok', 'conditions.safetensors', 'music-embeddings-bf16.bin', 'mapping-prefix.bf16']
    return files + [f'aux/{name}.f32' for name in AUX_ELEMENTS]


def legacy_files(profile):
    return sidecars(profile) + [name for role in MODULES for name in (canonical_graph(role, profile), canonical_graph(role, profile) + '.weight')]


def verify_bundle(root, hashes=True):
    root = Path(root).resolve(); manifest = read_json(root / 'model-bundle.json')
    if manifest.get('schemaVersion') != 1:
        raise ValueError('Unsupported bundle schema')
    model_id(manifest['modelId']); profile = load_profile(manifest['profile'])
    if manifest.get('contract') != profile['contract'] or manifest.get('family') != profile['family']:
        raise ValueError('Bundle architecture/codec contract mismatch')
    if set(manifest.get('modules', {})) != set(MODULES):
        raise ValueError('Bundle must declare all three modules')
    assets = {}; total = 0
    for asset in manifest['assets']:
        relative = asset['relative']; file = inside(root, relative)
        if relative in assets or not file.is_file() or file.stat().st_size != asset['bytes'] or asset['bytes'] <= 0:
            raise ValueError(f'Duplicate/missing/incorrect asset: {relative}')
        if not re.fullmatch('[a-f0-9]{64}', asset['sha256']) or (hashes and sha256(file) != asset['sha256']):
            raise ValueError(f'Asset hash mismatch: {relative}')
        assets[relative] = asset; total += asset['bytes']
    required = set(sidecars(profile))
    for role, module in manifest['modules'].items():
        if role == 'llm' and module['path'] != canonical_graph(role, profile):
            raise ValueError('LLM graph/companion directory must match family adapter')
        inside(root, module['path']); required.add(module['path'])
        # A converted graph can embed its weights; only declared external files
        # are required. Current baseline MNN files all declare external weights.
        if module.get('externalWeight'):
            if module['externalWeight'] != module['path'] + '.weight':
                raise ValueError('Runtime expects an external weight adjacent to its graph')
            required.add(module['externalWeight'])
        policy = module['runtime']
        if policy['precision'] not in ('high', 'low') or not 1 <= policy['threads'] <= 16:
            raise ValueError('Unsupported module runtime policy')
    if not required.issubset(assets):
        raise ValueError(f'Missing runtime companions: {sorted(required - assets.keys())}')
    for name, count in AUX_ELEMENTS.items():
        if assets[f'aux/{name}.f32']['bytes'] != count * 4:
            raise ValueError(f'Wrong {name} tensor size for this codec')
    lm_dir = 'student' if profile['family'] == 'small' else 'lm'
    cfg = read_json(root / lm_dir / 'llm_config.json'); expected = profile['modules']['llm']
    if cfg.get('hidden_size') != expected['hidden'] or cfg.get('layer_nums') != expected['layers'] or cfg.get('vocab_size') != expected['outputClasses']:
        raise ValueError('LLM dimensions/vocabulary do not match profile')
    if profile['family'] == 'small':
        metadata = read_json(root / 'student/diagnostic-student.json'); prefix = read_json(root / 'student/prefix.json')
        if len(prefix) != 514 or metadata.get('hidden') != 1024 or metadata.get('layers') != 26:
            raise ValueError('Student prefix/embedding metadata is incompatible')
        rows = metadata['embedding_rows']; size = assets['student/embeddings_bf16.bin']['bytes']
        if size != rows * 1024 * 2 or any(type(x) != int or not 0 <= x < rows for x in prefix) or not 0 <= metadata['music_offset'] <= rows-16386 or not 0 <= metadata['bos_input_id'] < rows:
            raise ValueError('Student prefix ID or input embedding dimensions are invalid')
    else:
        if assets['music-embeddings-bf16.bin']['bytes'] != 16386 * 2048 * 2 or assets['mapping-prefix.bf16']['bytes'] != 32 * 2048 * 2:
            raise ValueError('Public music/mapping embedding dimensions differ from the conditioner')
        with (root / 'conditions.safetensors').open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8: raise ValueError('Truncated condition tensor header')
            length = struct.unpack('<Q', raw)[0]
            if length > 1024**2: raise ValueError('Invalid condition tensor header')
            header = json.loads(stream.read(length))
        prefix = 'audiolm.condition_provider.conditioners.'
        minimum_rows = {'description.output_proj.weight': 151659, 'type_info.output_proj.weight': 151652,
                        'description.structure_emb.weight': 14, 'prompt_audio.EOT_emb': 1, 'prompt_audio.emb.0.weight': 16386}
        for name, rows in minimum_rows.items():
            tensor = header[prefix + name]; shape = tensor['shape']; begin, end = tensor['data_offsets']
            if tensor['dtype'] != 'BF16' or len(shape) != 2 or shape[0] < rows or shape[1] != 2048 or end-begin != shape[0]*shape[1]*2 or begin < 0 or end+length+8 > assets['conditions.safetensors']['bytes']:
                raise ValueError(f'Invalid public condition tensor: {name}')
    return {'integrityPassed': True, 'modelId': manifest['modelId'], 'profile': manifest['profile'], 'assets': len(assets),
            'bytes': total, 'musicTrainedDeclared': manifest['musicTrained'], 'qualityValidated': manifest.get('qualityValidated', False)}


def pack_bundle(job, output):
    profile = load_profile(job['profile']); identity = model_id(job['modelId']); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new versioned modelId/output; existing bundles are preserved')
    base = Path(job['baseDirectory']).resolve() if job.get('baseDirectory') else None
    base_manifest = read_json(base / 'model-bundle.json') if base and (base / 'model-bundle.json').is_file() else None
    if base_manifest:
        verify_bundle(base)
        if base_manifest['profile'] != profile['id']:
            raise ValueError('Base bundle belongs to a different architecture')
    sources = {relative: inside(base, relative) for relative in
               ([x['relative'] for x in base_manifest['assets']] if base_manifest else legacy_files(profile))} if base else {}
    for relative, path in job.get('files', {}).items():
        inside(output, relative); sources[relative] = Path(path).resolve()
    modules = {}
    training = job.get('training', {})
    for role in MODULES:
        recipe = dict(profile['modules'][role]); graph = canonical_graph(role, profile)
        component_file = job.get('components', {}).get(role)
        validation = {'status': 'baseline-declared', 'numericParityVerified': False}
        if base_manifest:
            old = base_manifest['modules'][role]; graph = old['path']; recipe.update(old.get('recipe', {})); validation = old.get('validation', validation)
        if component_file:
            component_file = Path(component_file).resolve(); component = read_json(component_file)
            if component['profile'] != profile['id'] or component['module'] != role or component['contract'] != profile['contract']['id']:
                raise ValueError(f'Component contract/profile mismatch: {role}')
            for relative in (graph, graph + '.weight'):
                sources.pop(relative, None)
            folder = 'student' if role == 'llm' and profile['family'] == 'small' else ('lm' if role == 'llm' else role)
            graph_name = 'llm.mnn' if role == 'llm' else role + '.mnn'
            graph = folder + '/' + graph_name
            for item in component['files']:
                original = inside(component_file.parent, item['name'])
                if original.stat().st_size != item['bytes'] or sha256(original) != item['sha256']:
                    raise ValueError(f'Prepared component changed: {original}')
                filename = graph_name if item['name'] == component['graph'] else (graph_name + '.weight' if item['name'] == component['graph'] + '.weight' else item['name'])
                sources[folder + '/' + filename] = original
            recipe.update(component['recipe']); validation = component['validation']
            training.setdefault(role, component.get('training', 'unknown'))
        status = training.get(role, base_manifest['modules'][role].get('training', 'unknown') if base_manifest else 'unknown')
        if status not in ('pretrained', 'trained', 'untrained', 'unknown'):
            raise ValueError('Training status must be pretrained/trained/untrained/unknown')
        modules[role] = {'path': graph, 'externalWeight': graph + '.weight' if graph + '.weight' in sources else None,
                         'runtime': {'precision': recipe['precision'], 'threads': recipe['threads']}, 'recipe': recipe, 'training': status, 'validation': validation}
    if not sources:
        raise ValueError('Supply baseDirectory and/or explicit files')
    # Validate all source paths before creating the destination.
    for relative, original in sources.items():
        inside(output, relative)
        if not original.is_file(): raise FileNotFoundError(original)
    output.mkdir(parents=True)
    for relative, original in sources.items():
        target = inside(output, relative); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(original, target)
    title = job.get('displayName', identity)
    if not isinstance(title, str) or not 1 <= len(title) <= 100:
        raise ValueError('displayName must be 1..100 characters')
    manifest = {'schemaVersion': 1, 'modelId': identity, 'displayName': title, 'profile': profile['id'], 'family': profile['family'],
                'contract': profile['contract'], 'runtime': profile['runtime'], 'modules': modules,
                'musicTrained': all(x['training'] in ('trained', 'pretrained') for x in modules.values()),
                'qualityValidated': False, 'legacyMappingProbe': bool(job.get('legacyMappingProbe', False)),
                'assets': [inventory(output, relative) for relative in sorted(sources)],
                'provenance': {'baseModelId': base_manifest['modelId'] if base_manifest else None,
                               'scope': 'Training status is supplied by the model owner; integrity and numerical checks do not prove music quality'}}
    write_json(output / 'model-bundle.json', manifest)
    result = verify_bundle(output)
    write_json(output / 'bundle-validation.json', result)
    return result
