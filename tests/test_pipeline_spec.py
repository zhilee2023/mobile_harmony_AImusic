import unittest

from levo_harmony.pipeline_spec import PipelineSpec, require_verified_quantization


def profile(route='hierarchical', hidden=2048, layers=36, heads=16, ffn=11008):
    return PipelineSpec('test', layers, hidden, ffn, heads, 12 if route == 'hierarchical' else 0,
                        16384, 25, 48000, 'official-separate' if route == 'hierarchical' else 'official-mixed', route)


class PipelineSpecTests(unittest.TestCase):
    def test_single_output_does_not_erase_internal_tracks(self):
        p = profile()
        self.assertEqual(p.generated_streams, ('mixed', 'vocal', 'accompaniment'))
        p.validate_tokens({'vocal': [2, 3], 'accompaniment': [4, 5]}, 'official-separate')
        with self.assertRaises(ValueError):
            p.validate_tokens({'mixed': [2, 3]}, 'official-separate')

    def test_wrong_codec_eos_or_unaligned_tracks_rejected(self):
        p = profile()
        for streams, identity in (({'vocal': [2], 'accompaniment': [3]}, 'official-mixed'),
                                  ({'vocal': [16384], 'accompaniment': [3]}, 'official-separate'),
                                  ({'vocal': [2, 4], 'accompaniment': [3]}, 'official-separate')):
            with self.assertRaises(ValueError):
                p.validate_tokens(streams, identity)

    def test_model_size_and_condition_identity(self):
        medium = profile(hidden=1536, layers=28, heads=12, ffn=8960)
        medium.validate_condition_width(1536)
        with self.assertRaises(ValueError):
            medium.validate_condition_width(1024)
        with self.assertRaises(ValueError):
            medium.validate_main_weight_shapes({})
        self.assertGreater(profile().ideal_kv_bytes(1000), medium.ideal_kv_bytes(1000))

    def test_logical_four_bit_range_cannot_claim_packed_execution(self):
        with self.assertRaises(RuntimeError):
            require_verified_quantization(4, 4, 4, {'packed_w4': True, 'kv4_execution': True,
                                                  'logical_a4_range': True, 'checkpoint_numerics_passed': False})


if __name__ == '__main__':
    unittest.main()
