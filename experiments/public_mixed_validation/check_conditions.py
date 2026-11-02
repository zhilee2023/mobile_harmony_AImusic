"""Compare prepared lyrics/style against the pinned upstream conditioner code."""
import ast
import json
from pathlib import Path
import typing
import warnings
import numpy as np
import torch
from safetensors import safe_open
from prepare_conditions import BASE, LYRICS, STYLE, SECTIONS

source = Path('C:/levo-harmony/reference/LeVo/codeclm/modules/conditioners.py')
tree = ast.parse(source.read_text(encoding='utf8'))
names = {'BaseConditioner', 'TextConditioner', 'QwTokenizerConditioner', 'QwTextConditioner'}
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
if len(nodes) != 4:
    raise ValueError('Upstream classes changed')
scope = {'torch': torch, 'nn': torch.nn, 'tp': typing, 'warnings': warnings,
         'ConditionType': typing.Tuple[torch.Tensor, torch.Tensor]}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), scope)
torch.set_num_threads(4)
prefix = torch.from_numpy(np.fromfile(BASE / 'prefix-conditioned.f32', '<f4').reshape(953, 2048))
null_prefix = torch.from_numpy(np.fromfile(BASE / 'prefix-unconditional.f32', '<f4').reshape(953, 2048))
rows = []
with safe_open(BASE / 'conditions.safetensors', framework='pt', device='cpu') as state:
    for name, cls, tokens, expected_slice in [('description', 'QwTokenizerConditioner', LYRICS, slice(0,600)),
                                              ('type_info', 'QwTextConditioner', STYLE, slice(852,952))]:
        with torch.device('meta'):
            args = {'output_dim': 2048, 'token_path': str(BASE / 'qwen2-tokenizer'),
                    'max_len': 600 if name == 'description' else 100}
            if name == 'description':
                args['add_token_list'] = SECTIONS
            module = scope[cls](**args)
        weights = {key.removeprefix(f'audiolm.condition_provider.conditioners.{name}.'): state.get_tensor(key)
                   for key in state.keys() if key.startswith(f'audiolm.condition_provider.conditioners.{name}.')}
        # Published type_info reserves six more rows than the runtime tokenizer.
        # Keep every trained row; current tokenizer IDs are an unchanged prefix.
        if module.output_proj.weight.shape != weights['output_proj.weight'].shape:
            shape = weights['output_proj.weight'].shape
            if name != 'type_info' or shape != (151652, 2048):
                raise ValueError('Unexpected condition vocabulary mismatch')
            module.output_proj = torch.nn.Embedding(*shape, padding_idx=151643, device='meta')
        module.load_state_dict(weights, strict=True, assign=True)
        for phase, text, expected in [('conditioned', tokens, prefix), ('unconditional', None, null_prefix)]:
            with torch.inference_mode(), torch.device('cpu'):
                actual = module(module.tokenize([text]))[0][0].float()
            # Upstream tables store BF16. Their elementwise addition rounds to
            # BF16, as does the private MNN embedding-lookup prefix.
            target = expected[expected_slice].to(torch.bfloat16).float()
            error = float(torch.linalg.vector_norm(actual - target))
            if error != 0:
                raise ValueError(f'Condition mismatch {name}/{phase}: {error}')
            rows.append({'condition': name, 'phase': phase, 'shape': list(actual.shape), 'absoluteL2': error})
report = {'success': True, 'exactBf16ParityWithPinnedUpstream': True, 'rows': rows,
          'typeInfoCheckpointRows': 151652, 'runtimeTokenizerRows': 151646,
          'unusedReservedRowsRetained': 6}
(BASE / 'condition-parity.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
