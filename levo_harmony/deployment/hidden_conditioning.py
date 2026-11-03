"""Research contract checks; deliberately separate from deployable token bundles."""
import math
from numbers import Real


def _positive_int(data, name):
    value = data.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def validate_contract(data):
    if data.get('schemaVersion') != 1 or data.get('conditioningType') != 'llm_last_layer_sequence':
        raise ValueError('Expected a last-layer hidden sequence research contract')
    if data.get('contractId') != 'levo-hidden-sequence-research-v1':
        raise ValueError('Do not reuse the current token/codebook contract ID')
    if data.get('layout') != 'BTH' or data.get('batchSize') != 1:
        raise ValueError('Reference checks currently support canonical [1,T,H] only')
    if data.get('hiddenOutputName') != 'hidden_states':
        raise ValueError('The pinned MNN API loads hidden_states; custom names need a runtime adapter')
    if data.get('hiddenPoint') not in ('pre_final_norm', 'post_final_norm'):
        raise ValueError('Declare the exact hidden point; never infer normalization from width')
    if data.get('tokenAlignment') not in ('predictor', 'consumed_token'):
        raise ValueError('Declare predictor versus consumed_token alignment explicitly')
    if data.get('eosPolicy') != 'exclude':
        raise ValueError('This reference contract excludes EOS from audio condition frames')
    for key in ('hiddenSize', 'conditionWidth', 'prefixPositions', 'windowFrames'):
        _positive_int(data, key)
    overlap = data.get('contextFrames')
    if type(overlap) is not int or not 0 <= overlap < data['windowFrames']:
        raise ValueError('contextFrames must be between zero and windowFrames-1')
    rates = [data.get(key) for key in ('sourceTokenRateHz', 'conditionFrameRateHz')]
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0 for x in rates):
        raise ValueError('Declare positive finite source and condition frame rates')
    if data.get('temporalAdapter') != 'one_to_one' or rates[0] != rates[1]:
        raise ValueError('Unequal rates need a trained temporal/cross-attention adapter; do not silently repeat frames')
    kind = data.get('conditionerKind')
    if kind not in ('identity', 'trained_projection'):
        raise ValueError('Declare identity or trained_projection conditioner')
    if kind == 'identity' and data['hiddenSize'] != data['conditionWidth']:
        raise ValueError('Identity cannot change the hidden width')
    for name in ('conditionerTrainingStatus', 'diffusionTrainingStatus'):
        if data.get(name) not in ('unknown', 'trained', 'untrained'):
            raise ValueError(f'{name} must be explicit')
    for name in ('lmCheckpointSha256', 'conditionerCheckpointSha256', 'diffusionCheckpointSha256'):
        value = data.get(name)
        if value is not None and (not isinstance(value, str) or len(value) != 64 or any(x not in '0123456789abcdef' for x in value)):
            raise ValueError(f'{name} must be a lowercase SHA256 or null')
    return dict(data)


def window_plan(total_frames, window_frames=256, context_frames=64):
    """Fixed windows covering each sequence; no overlap-only final window."""
    if type(total_frames) is not int or total_frames < 0:
        raise ValueError('total_frames must be a nonnegative integer')
    if type(window_frames) is not int or window_frames < 1 or type(context_frames) is not int or not 0 <= context_frames < window_frames:
        raise ValueError('Invalid window/context')
    windows = []
    start = 0
    while start < total_frames:
        valid = min(window_frames, total_frames-start)
        final = start+valid == total_frames
        windows.append({'start': start, 'validFrames': valid, 'paddingFrames': window_frames-valid, 'final': final})
        if final:
            break
        start += window_frames-context_frames
    return windows


def hidden_positions(prefix_positions, generated_tokens, alignment):
    """Absolute causal positions; fixed teacher-forced token sequence, EOS excluded."""
    if type(prefix_positions) is not int or prefix_positions < 1 or type(generated_tokens) is not int or generated_tokens < 0:
        raise ValueError('Invalid prefix or generated token count')
    if alignment not in ('predictor', 'consumed_token'):
        raise ValueError('Unknown token alignment')
    offset = prefix_positions-1 if alignment == 'predictor' else prefix_positions
    return range(offset, offset+generated_tokens)


def memory_budget(contract, seconds):
    c = validate_contract(contract)
    if isinstance(seconds, bool) or not isinstance(seconds, Real) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be positive and finite')
    frames_float = seconds*c['sourceTokenRateHz']
    frames = round(frames_float)
    if abs(frames_float-frames) > 1e-6:
        raise ValueError('Duration must map to a whole number of condition frames')
    positions = hidden_positions(c['prefixPositions'], frames, c['tokenAlignment'])
    widths = {'hidden': c['hiddenSize'], 'condition': c['conditionWidth']}
    sizes = {name: {'fullSequenceFp32Bytes': frames*width*4, 'windowFp32Bytes': c['windowFrames']*width*4,
                    'fullSequenceFp16Bytes': frames*width*2, 'windowFp16Bytes': c['windowFrames']*width*2} for name,width in widths.items()}
    known = bool(c.get('lmCheckpointSha256') and c.get('diffusionCheckpointSha256')) and c['diffusionTrainingStatus'] == 'trained'
    if c['conditionerKind'] != 'identity':
        known = known and bool(c.get('conditionerCheckpointSha256')) and c['conditionerTrainingStatus'] == 'trained'
    return {'scope': 'Research shape/alignment/memory check only; no execution or deployable model bundle',
            'contractValid': True, 'currentAppCompatible': False, 'requiresNewRuntimeAdapter': True,
            'trainingProvenanceDeclaredComplete': known, 'trainingQualityVerified': False,
            'frames': frames, 'alignmentFirstPosition': positions.start if frames else None,
            'alignmentLastPosition': positions.stop-1 if frames else None,
            'windows': window_plan(frames, c['windowFrames'], c['contextFrames']), 'tensorStorage': sizes,
            'excludes': ['LLM weights and packing', 'KV cache and its expansion', 'DiT/VAE weights and activations',
                         'allocator overhead and model co-residency'],
            'limits': ['Equal shapes do not establish semantic compatibility', 'A bounded hidden queue does not bound KV memory',
                       'FP16 figures are storage arithmetic, not measured runtime or proven numerical compatibility']}
