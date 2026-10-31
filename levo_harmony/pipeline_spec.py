"""Size-independent interface checks for official LeVo pipelines.

These contracts do not load weights or implement a new generation backend.
Mixed-only and hierarchical dual-track decoders have different codec identities.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    main_layers: int
    hidden: int
    ffn: int
    heads: int
    track_layers: int
    codebook_size: int
    frame_rate: int
    sample_rate: int
    codec_id: str
    route: str

    def __post_init__(self):
        if self.route not in ('hierarchical', 'mixed-only'):
            raise ValueError('Unknown token route')
        if min(self.main_layers, self.hidden, self.ffn, self.heads, self.codebook_size) <= 0 or self.hidden % self.heads:
            raise ValueError('Invalid architecture dimensions')
        if self.route == 'hierarchical' and self.track_layers <= 0:
            raise ValueError('Hierarchical route requires a trained track-specific LM')
        if self.route == 'mixed-only' and self.track_layers != 0:
            raise ValueError('Mixed-only profile must explicitly disable the second LM')
        if not self.codec_id or self.codec_id.startswith('UNVERIFIED'):
            raise ValueError('Need explicit codec identity, not only matching tensor width')
        if self.frame_rate != 25 or self.sample_rate != 48000:
            raise ValueError('Current public LeVo contract is 25Hz to 48kHz')

    @classmethod
    def from_config(cls, name, config, route, codec_id):
        lm = config['lm']
        return cls(name, lm['num_layers'], lm['dim'], lm['intermediate_size'], lm['num_heads'],
                   lm['num_layers_sub'] if route == 'hierarchical' else 0, lm['code_size'],
                   config['audio_tokenizer_frame_rate'], config['sample_rate'], codec_id, route)

    @property
    def generated_streams(self):
        return ('mixed', 'vocal', 'accompaniment') if self.route == 'hierarchical' else ('mixed',)

    @property
    def renderer_streams(self):
        return ('vocal', 'accompaniment') if self.route == 'hierarchical' else ('mixed',)

    def validate_condition_width(self, hidden):
        if hidden != self.hidden:
            raise ValueError('Condition embedding width differs from this trained LM; resizing is not an adapter')

    def validate_tokens(self, streams, codec_id):
        if codec_id != self.codec_id:
            raise ValueError('Codec checkpoint identity mismatch')
        if set(streams) != set(self.renderer_streams):
            raise ValueError('Wrong token streams for this decoder route')
        sizes = set()
        for ids in streams.values():
            if not ids or any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < self.codebook_size for x in ids):
                raise ValueError('Invalid audio token; EOS/padding must be stripped before rendering')
            sizes.add(len(ids))
        if len(sizes) != 1:
            raise ValueError('Track token lengths must be aligned before rendering')

    def validate_main_weight_shapes(self, shapes):
        prefix = 'audiolm.transformer.model.layers.'
        layers = {int(k[len(prefix):].split('.')[0]) for k in shapes if k.startswith(prefix)}
        if layers != set(range(self.main_layers)):
            raise ValueError('Checkpoint layer indices differ from the profile')
        for i in range(self.main_layers):
            expected = {f'{prefix}{i}.self_attn.q_proj.weight': [self.hidden, self.hidden],
                        f'{prefix}{i}.mlp.gate_proj.weight': [self.ffn, self.hidden],
                        f'{prefix}{i}.mlp.down_proj.weight': [self.hidden, self.ffn]}
            if any(shapes.get(k) != v for k, v in expected.items()):
                raise ValueError('Trained tensor dimensions do not match this model profile')

    def ideal_kv_bytes(self, positions, bits=16, cfg_branches=1):
        if positions < 1 or bits not in (4, 8, 16, 32) or cfg_branches not in (1, 2):
            raise ValueError('Invalid KV estimate')
        layers = self.main_layers + self.track_layers
        # Official LM uses num_key_value_heads=num_heads, unlike Qwen GQA.
        return (2 * layers * self.hidden * positions * bits * cfg_branches + 7) // 8


def require_verified_quantization(weight_bits, activation_bits, kv_bits, capabilities):
    if (weight_bits, activation_bits, kv_bits) != (4, 4, 4):
        raise ValueError('This check is specifically for requested W4A4KV4')
    for key in ('packed_w4', 'packed_a4_execution', 'kv4_execution', 'checkpoint_numerics_passed'):
        if capabilities.get(key) is not True:
            raise RuntimeError('W4A4KV4 deployment not verified: ' + key)
