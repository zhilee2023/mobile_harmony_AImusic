"""Architecture adapters for tensor checkpoints; no random missing-key fill."""
import sys
from pathlib import Path
from .contracts import ROOT
from .checkpoints import load_state, strict_load


def codec_model(role, profile, source):
    import torch
    from torch import nn
    sys.path.insert(0, str(ROOT / 'tools'))
    if role == 'dit':
        from restore_mixed_dit import RestoredMixedCfm, upstream_classes
        from export_real_dit_block import real_freqs, real_rope
        cfg = profile['modules']['dit']; checkout = Path(source.get('checkout', ROOT / 'reference/LeVo'))
        if profile['family'] == 'public':
            class Velocity(nn.Module):
                def __init__(self):
                    super().__init__(); self.cfm = RestoredMixedCfm(checkout, hidden=cfg['hidden'], layers=cfg['layers'], heads=cfg['heads'], inner=cfg['intermediate'])
                def forward(self, inputs, timestep, attention_mask):
                    return self.cfm(inputs, timestep, attention_mask)[..., -64:]
            # Upstream GPT registers nonpersistent causal-mask buffers. Build
            # those on CPU rather than inventing values for meta-only buffers.
            model = Velocity()
            target = model.cfm
        else:
            Config, Model, _ = upstream_classes(checkout)
            class SmallVelocity(nn.Module):
                def __init__(self):
                    super().__init__()
                    config = Config(vocab_size=1, n_positions=256, n_layer=cfg['layers'], n_head=cfg['heads'], n_embd=cfg['hidden'], n_inner=cfg['intermediate'], use_cache=False, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
                    config._attn_implementation = 'eager'
                    self.input_projection = nn.Linear(1200, cfg['hidden']); self.estimator = Model(config); self.output_projection = nn.Linear(cfg['hidden'], 64)
                def forward(self, inputs, timestep, attention_mask):
                    x = self.estimator(inputs_embeds=self.input_projection(inputs), time_step=timestep, attention_mask=attention_mask, use_cache=False).last_hidden_state
                    return self.output_projection(x)
            model = SmallVelocity()
            target = model
        state = load_state(source['path'], source.get('stateKey'), source.get('stripPrefix'))
        loaded = strict_load(target, state)
        _, _, upstream = upstream_classes(checkout)
        freqs = real_freqs(cfg['hidden'] // cfg['heads'], 256)
        upstream.precompute_freqs_cis = lambda dim, end, constant=10000.: freqs
        upstream.apply_rotary_emb = real_rope
    elif role == 'vae':
        from restore_vae_decoder import create_decoder
        with torch.device('meta'):
            model = create_decoder()
        loaded = strict_load(model, load_state(source['path'], source.get('stateKey'), source.get('stripPrefix')))
        # Preserve checkpoint weight-normalization until all keys are verified.
        for child in model.modules():
            if hasattr(child, 'weight_g'):
                nn.utils.remove_weight_norm(child)
    else:
        raise ValueError('Codec adapter supports dit/vae')
    return model.float().eval(), loaded


def hf_state(folder):
    from .files import read_json
    path = Path(folder)
    for index in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if (path / index).is_file():
            files = sorted(set(read_json(path / index)['weight_map'].values()))
            state = {}
            for name in files:
                part = load_state(path / name)
                if state.keys() & part.keys():
                    raise ValueError('Repeated keys across HF shards')
                state.update(part)
            return state
    for name in ('model.safetensors', 'pytorch_model.bin'):
        if (path / name).is_file():
            return load_state(path / name)
    raise ValueError('HF folder needs local safetensors or PyTorch tensor weights')


def language_model(profile, source):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config_dir = Path(source.get('configDirectory', source['path'] if Path(source['path']).is_dir() else ''))
    if not (config_dir / 'config.json').is_file():
        raise ValueError('A raw LM checkpoint needs configDirectory containing config.json')
    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True, trust_remote_code=False)
    expected = profile['modules']['llm']
    if config.model_type != expected['modelType'] or config.hidden_size != expected['hidden'] or config.num_hidden_layers != expected['layers']:
        raise ValueError('LM config differs from selected architecture profile')
    state = hf_state(source['path']) if Path(source['path']).is_dir() else load_state(source['path'], source.get('stateKey'), source.get('stripPrefix'))
    layout = source.get('checkpointLayout', 'hf')
    if layout == 'student-training':
        if profile['family'] != 'small':
            raise ValueError('student-training layout requires small profile')
        from levo_harmony.model import MixedSongStudent
        with torch.device('meta'):
            student = MixedSongStudent(AutoModelForCausalLM.from_config(config, trust_remote_code=False))
        strict_load(student, state)
        merged = {k.removeprefix('backbone.'): v for k, v in state.items() if k.startswith('backbone.model.')}
        merged['model.embed_tokens.weight'] = torch.cat((state['backbone.model.embed_tokens.weight'], state['style.weight'], state['music.weight']))
        merged['lm_head.weight'] = state['music_head.weight']; state = merged
    elif layout != 'hf':
        raise ValueError('checkpointLayout must be hf or student-training')
    embedding = state.get('model.embed_tokens.weight'); head = state.get('lm_head.weight')
    if embedding is None or head is None or list(head.shape) != [16385, expected['hidden']] or embedding.shape[1] != expected['hidden']:
        raise ValueError('Music LM requires canonical HF input embedding and a 16385-row music output head; a text-only checkpoint is insufficient')
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        model.set_input_embeddings(torch.nn.Embedding(embedding.shape[0], expected['hidden']))
        model.set_output_embeddings(torch.nn.Linear(expected['hidden'], 16385, bias=False))
    loaded = strict_load(model, state); model.config.vocab_size = 16385
    # assign=True fills persistent weights, not HF's nonpersistent RoPE cache.
    # Reconstruct only parameter-free rotary modules, not a second large LM.
    for name, module in list(model.named_modules()):
        if 'RotaryEmbedding' in type(module).__name__ and any(x.is_meta for x in module.buffers()):
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            with torch.device('cpu'):
                replacement = type(module)(config=model.config)
            setattr(parent, child, replacement)
    pending = [name for name, value in model.named_buffers() if value.is_meta]
    if pending:
        raise ValueError(f'Architecture has uninitialized nonpersistent buffers: {pending}; add an explicit buffer adapter')
    return model.eval(), config_dir, loaded
