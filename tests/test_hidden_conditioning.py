"""Hidden semantics and off-by-one checks; no trained music weights required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from levo_harmony.deployment.contracts import ROOT
from levo_harmony.deployment.hidden_conditioning import validate_contract, window_plan, hidden_positions, memory_budget
from tools.inspect_hidden_conditioning import inspect


class HiddenConditioningTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / 'configs/conditioning/hidden-sequence.example.json'
        self.contract = json.loads(self.path.read_text(encoding='utf8'))

    def test_example_is_research_only_not_a_deployable_model(self):
        result = memory_budget(self.contract, 60)
        self.assertTrue(result['contractValid'])
        self.assertFalse(result['currentAppCompatible'])
        self.assertFalse(result['trainingProvenanceDeclaredComplete'])
        self.assertFalse(result['trainingQualityVerified'])

    def test_hidden_point_and_alignment_cannot_be_guessed(self):
        for field in ('hiddenPoint', 'tokenAlignment'):
            broken = copy.deepcopy(self.contract)
            broken.pop(field)
            with self.assertRaises(ValueError): validate_contract(broken)

    def test_temporal_rate_mismatch_needs_a_real_adapter(self):
        self.contract['conditionFrameRateHz'] = 50
        with self.assertRaisesRegex(ValueError, 'trained temporal'):
            validate_contract(self.contract)

    def test_identity_does_not_silently_reshape_channels(self):
        self.contract['conditionerKind'] = 'identity'
        with self.assertRaises(ValueError): validate_contract(self.contract)
        self.contract['conditionWidth'] = self.contract['hiddenSize']
        self.assertEqual(validate_contract(self.contract)['conditionWidth'], 2560)

    def test_window_boundaries_do_not_add_overlap_only_tail(self):
        self.assertEqual(window_plan(0), [])
        self.assertEqual(len(window_plan(256)), 1)
        self.assertEqual([(x['start'], x['validFrames']) for x in window_plan(257)], [(0,256),(192,65)])
        self.assertEqual(len(window_plan(448)), 2)
        self.assertEqual([(x['start'], x['validFrames']) for x in window_plan(449)], [(0,256),(192,256),(384,65)])
        self.assertEqual(len(window_plan(1500)), 8)

    def test_predictor_and_consumed_states_have_one_position_offset(self):
        self.assertEqual(list(hidden_positions(5, 3, 'predictor')), [4,5,6])
        self.assertEqual(list(hidden_positions(5, 3, 'consumed_token')), [5,6,7])

    def test_memory_arithmetic_excludes_kv_and_real_model_allocations(self):
        result = memory_budget(self.contract, 60)
        self.assertEqual(result['frames'], 1500)
        self.assertEqual(result['tensorStorage']['hidden']['fullSequenceFp32Bytes'], 15360000)
        self.assertEqual(result['tensorStorage']['hidden']['windowFp32Bytes'], 2621440)
        self.assertIn('KV cache and its expansion', result['excludes'])
        with self.assertRaises(ValueError): memory_budget(self.contract, .001)
        with self.assertRaises(ValueError): memory_budget(self.contract, float('nan'))

    def test_graph_output_and_hidden_width_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            graph, config = folder/'graph.json', folder/'llm_config.json'
            graph.write_text(json.dumps({'outputName':['logits'],'tensorName':['logits','hidden_states']}))
            with self.assertRaisesRegex(ValueError, 'both logits'): inspect(self.path, 60, mnn_json=graph)
            graph.write_text(json.dumps({'outputName':['logits','hidden_states']}))
            config.write_text(json.dumps({'hidden_size': 1024}))
            with self.assertRaisesRegex(ValueError, 'hidden_size'): inspect(self.path, 60, config, graph)
            config.write_text(json.dumps({'hidden_size': 2560}))
            result = inspect(self.path, 60, config, graph)
            self.assertFalse(result['mnnGraph']['semanticsVerified'])

    def test_qwen_cached_hidden_matches_full_teacher_forcing_for_both_alignments(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(2307)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0.)
        config._attn_implementation = 'eager'
        model = Qwen3ForCausalLM(config).eval()
        prefix = torch.tensor([[1,5,6]])
        tokens = torch.tensor([[7,11,3,2]])
        with torch.inference_mode():
            full = model.model(torch.cat((prefix,tokens),dim=1), use_cache=False).last_hidden_state
            out = model.model(prefix, use_cache=True)
            predictors, consumed = [], []
            for index in range(tokens.shape[1]):
                predictors.append(out.last_hidden_state[:,-1:].clone())
                out = model.model(tokens[:,index:index+1], past_key_values=out.past_key_values, use_cache=True)
                consumed.append(out.last_hidden_state[:,-1:].clone())
            predictor = torch.cat(predictors,dim=1)
            after_token = torch.cat(consumed,dim=1)
            start = prefix.shape[1]
            self.assertTrue(torch.allclose(predictor,full[:,start-1:start-1+tokens.shape[1]],rtol=1e-5,atol=1e-5))
            self.assertTrue(torch.allclose(after_token,full[:,start:start+tokens.shape[1]],rtol=1e-5,atol=1e-5))
            self.assertFalse(torch.allclose(predictor,after_token,rtol=1e-5,atol=1e-5))


if __name__ == '__main__':
    unittest.main()
