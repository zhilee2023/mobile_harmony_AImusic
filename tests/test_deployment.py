"""Integrity boundaries and real input-format handling, independent of phones."""
import json
import tempfile
import unittest
from pathlib import Path

from levo_harmony.deployment.contracts import load_profile, io_contract
from levo_harmony.deployment.files import inside, write_json, read_json, sha256
from levo_harmony.deployment.checkpoints import load_state, strict_load, format_of
from levo_harmony.deployment.bundles import pack_bundle, verify_bundle, model_id, AUX_ELEMENTS


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.base = cls.root / 'source'; cls.base.mkdir()
        for folder in ('student', 'dit', 'vae', 'aux'):
            (cls.base / folder).mkdir()
        for relative in ('student/llm.mnn', 'student/llm.mnn.weight', 'student/tokenizer.mtok', 'dit/full-w8.mnn', 'dit/full-w8.mnn.weight', 'vae/vae-256-w8.mnn', 'vae/vae-256-w8.mnn.weight'):
            (cls.base / relative).write_bytes(b'unit integrity fixture, not an executable model')
        write_json(cls.base / 'student/config.json', {'backend_type': 'cpu'})
        write_json(cls.base / 'student/llm_config.json', {'hidden_size': 1024, 'layer_nums': 26, 'vocab_size': 16385})
        write_json(cls.base / 'student/diagnostic-student.json', {'music_trained': False, 'hidden': 1024, 'layers': 26, 'embedding_rows': 16418, 'music_offset': 32, 'bos_input_id': 16416})
        write_json(cls.base / 'student/prefix.json', [1] * 514)
        with (cls.base / 'student/embeddings_bf16.bin').open('wb') as out:
            out.truncate(16418 * 1024 * 2)
        for name, count in AUX_ELEMENTS.items():
            with (cls.base / f'aux/{name}.f32').open('wb') as out: out.truncate(count * 4)
        cls.job = {'profile': 'student-mixed-v1', 'modelId': 'test-base', 'baseDirectory': str(cls.base),
                   'training': {'llm': 'untrained', 'dit': 'untrained', 'vae': 'pretrained'}}
        cls.bundle = cls.root / 'packed'; pack_bundle(cls.job, cls.bundle)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_profiles_share_codec_but_not_network_dimensions(self):
        large, small = map(load_profile, ('public-mixed-v2', 'student-mixed-v1'))
        self.assertEqual(large['contract'], small['contract'])
        self.assertEqual(large['modules']['dit']['hidden'], 1200)
        self.assertEqual(small['modules']['dit']['hidden'], 1024)
        self.assertEqual(io_contract('vae', small)[1]['pcm'], [1, 2, 491520])

    def test_path_traversal_absolute_and_alias_paths_rejected(self):
        for value in ('../secret', '/secret', 'C:/secret', 'a\\b', 'a/../b', 'a//b', './a', ''):
            with self.subTest(value=value), self.assertRaises(ValueError): inside(self.root, value)

    def test_model_identity_cannot_be_shell_or_path_syntax(self):
        for value in ('../x', 'a b', 'a;touch', 'a$(id)', '/x', '', '-a', 'builtin', 'x.incoming-v1'):
            with self.subTest(value=value), self.assertRaises(ValueError): model_id(value)

    def test_verified_bundle_is_diagnostic_not_trained(self):
        report = verify_bundle(self.bundle)
        self.assertTrue(report['integrityPassed']); self.assertFalse(report['musicTrainedDeclared']); self.assertFalse(report['qualityValidated'])

    def test_existing_bundle_cannot_be_overwritten(self):
        before = sha256(self.bundle / 'model-bundle.json')
        with self.assertRaises(FileExistsError): pack_bundle(self.job, self.bundle)
        self.assertEqual(before, sha256(self.bundle / 'model-bundle.json'))

    def test_same_size_tampering_detected_by_hash_not_just_size(self):
        path = self.bundle / 'dit/full-w8.mnn'; original = path.read_bytes()
        try:
            path.write_bytes(b'x' * len(original))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): verify_bundle(self.bundle)
        finally: path.write_bytes(original)

    def test_component_from_other_architecture_is_rejected_before_copy(self):
        source = self.root / 'wrong-component.json'
        write_json(source, {'profile': 'public-mixed-v2', 'module': 'dit', 'contract': load_profile('public-mixed-v2')['contract']['id']})
        job = {**self.job, 'components': {'dit': str(source)}}
        out = self.root / 'wrong-target'
        with self.assertRaisesRegex(ValueError, 'contract/profile'): pack_bundle(job, out)
        self.assertFalse(out.exists())

    def test_component_replacement_removes_old_weight_and_keeps_base(self):
        prepared = self.root / 'component'; prepared.mkdir(exist_ok=True)
        (prepared / 'model.mnn').write_bytes(b'new graph'); (prepared / 'model.mnn.weight').write_bytes(b'new weights')
        profile = load_profile('student-mixed-v1')
        component = {'profile': profile['id'], 'module': 'dit', 'contract': profile['contract']['id'], 'graph': 'model.mnn',
                     'recipe': {**profile['modules']['dit'], 'weightBits': 4}, 'training': 'untrained', 'validation': {'status': 'unit-integrity-only'},
                     'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in prepared.iterdir()]}
        write_json(prepared / 'component.json', component)
        out = self.root / 'replaced'; pack_bundle({**self.job, 'modelId': 'new-recipe', 'components': {'dit': str(prepared / 'component.json')}}, out)
        self.assertFalse((out / 'dit/full-w8.mnn').exists()); self.assertTrue((out / 'dit/dit.mnn.weight').is_file())
        self.assertEqual(read_json(out / 'model-bundle.json')['modules']['dit']['recipe']['weightBits'], 4)
        self.assertTrue((self.base / 'dit/full-w8.mnn').is_file()); verify_bundle(out)

    def test_raw_checkpoint_nested_prefix_and_strict_shape_validation(self):
        import torch
        model = torch.nn.Linear(3, 2)
        path = self.root / 'weights.torch'
        torch.save({'state_dict': {'module.' + k: v for k, v in model.state_dict().items()}, 'step': 15}, path)
        with self.assertRaises(ValueError): load_state(path)
        state = load_state(path, 'state_dict', 'module.')
        strict_load(torch.nn.Linear(3, 2), state)
        with self.assertRaisesRegex(ValueError, 'Checkpoint mismatch'): strict_load(torch.nn.Linear(4, 2), state)
        with self.assertRaises(ValueError): load_state(path, 'state_dict', 'absent.')
        self.assertEqual(format_of(path), 'checkpoint')

    def test_serialized_python_module_does_not_enable_pickle_fallback(self):
        import torch
        path = self.root / 'object.pt'; torch.save(torch.nn.Linear(2, 2), path)
        with self.assertRaisesRegex(ValueError, 'tensor-only'): load_state(path)

    def test_raw_hf_lm_checkpoint_restores_rotary_cache_and_music_head(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from levo_harmony.deployment.torch_models import language_model
        config = LlamaConfig(vocab_size=16385, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
        folder = self.root / 'tiny-hf'; folder.mkdir(exist_ok=True); config.save_pretrained(folder)
        original = LlamaForCausalLM(config).eval(); checkpoint = folder / 'raw.pt'; torch.save(original.state_dict(), checkpoint)
        profile = load_profile('public-mixed-v2'); profile['modules']['llm'].update(hidden=64, layers=1, intermediate=128)
        restored, _, report = language_model(profile, {'path': str(checkpoint), 'configDirectory': str(folder)})
        self.assertTrue(report['strict']); self.assertFalse(any(x.is_meta for x in restored.buffers()))
        with torch.inference_mode():
            ids = torch.tensor([[1, 2, 3]])
            self.assertTrue(torch.equal(original(ids).logits, restored(ids).logits))

    def test_onnx_wrong_audio_shape_is_rejected(self):
        import onnx
        from onnx import helper, TensorProto
        from levo_harmony.deployment.contracts import validate_onnx
        path = self.root / 'bad-vae.onnx'
        graph = helper.make_graph([helper.make_node('Identity', ['latent'], ['pcm'])], 'bad',
                                 [helper.make_tensor_value_info('latent', TensorProto.FLOAT, [1, 64, 256])],
                                 [helper.make_tensor_value_info('pcm', TensorProto.FLOAT, [1, 64, 256])])
        onnx.save(helper.make_model(graph), path)
        with self.assertRaisesRegex(ValueError, 'expected fixed shape'): validate_onnx(path, 'vae', load_profile('student-mixed-v1'))

    def test_normal_hf_kv_onnx_is_not_mislabelled_as_mnn_llm(self):
        import onnx
        from onnx import helper, TensorProto
        from levo_harmony.deployment.contracts import validate_onnx
        path = self.root / 'hf-kv.onnx'
        graph = helper.make_graph([helper.make_node('Identity', ['past_key_values'], ['logits'])], 'wrong-lm',
                                 [helper.make_tensor_value_info('past_key_values', TensorProto.FLOAT, [1, 2])],
                                 [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 2])])
        onnx.save(helper.make_model(graph), path)
        with self.assertRaisesRegex(ValueError, 'names mismatch'): validate_onnx(path, 'llm', load_profile('public-mixed-v2'))

    def test_exporter_named_lm_with_text_head_is_rejected(self):
        import onnx
        from onnx import helper, TensorProto
        from levo_harmony.deployment.contracts import validate_onnx
        path = self.root / 'text-head.onnx'
        inputs = [helper.make_tensor_value_info('input_ids', TensorProto.FLOAT, [None, 1, 2048]),
                  helper.make_tensor_value_info('attention_mask', TensorProto.FLOAT, [1, 1, None, None]),
                  helper.make_tensor_value_info('position_ids', TensorProto.INT64, [1, None]),
                  helper.make_tensor_value_info('logits_index', TensorProto.INT32, [1])]
        graph = helper.make_graph([helper.make_node('Identity', ['input_ids'], ['logits'])], 'wrong-vocabulary', inputs,
                                 [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [None, 1, 151936])])
        onnx.save(helper.make_model(graph), path)
        with self.assertRaisesRegex(ValueError, '16385 music classes'): validate_onnx(path, 'llm', load_profile('public-mixed-v2'))


if __name__ == '__main__': unittest.main()
