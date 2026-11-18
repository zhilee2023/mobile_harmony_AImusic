import unittest
import numpy as np
from levo_harmony.quant_validation import music_logit_error


class MusicQuantValidationTests(unittest.TestCase):
    def test_ignores_reserved_classes_that_sampler_cannot_choose(self):
        ref = np.array([[0., 100., 99., 3., 1.]])
        got = np.array([[0., -100., -99., 3., 1.]])
        result = music_logit_error(ref, got, audio_base=3, audio_classes=2)
        self.assertEqual(result['relativeL2'], 0.)
        self.assertEqual(result['argmaxAgreement'], 1.)
        self.assertEqual(result['validOutputClasses'], 3)

    def test_distribution_metric_is_invariant_to_logit_offset(self):
        ref = np.array([[0., 2., -1.], [1., 0., -2.]])
        got = ref + 10
        result = music_logit_error(ref, got, audio_base=1, audio_classes=2)
        self.assertGreater(result['relativeL2'], 1.)
        self.assertAlmostEqual(result['meanKlReferenceToCandidate'], 0., places=14)
        self.assertEqual(result['argmaxAgreement'], 1.)

    def test_rejects_incomplete_or_nonfinite_results(self):
        with self.assertRaises(ValueError):
            music_logit_error(np.zeros((2, 3)), np.zeros((1, 3)), audio_base=1, audio_classes=2)
        with self.assertRaises(ValueError):
            music_logit_error(np.zeros((1, 3)), np.array([[0., np.nan, 0.]]), audio_base=1, audio_classes=2)

    def test_disjoint_row_crop_preserves_eos_and_audio_identity(self):
        original = np.array([[99., 88., 5., 77., 3., 1.], [55., 44., 0., 66., 4., 1.]])
        cropped = np.array([[5., 3., 1.], [0., 4., 1.]])
        result = music_logit_error(original, cropped, eos=2, audio_base=4,
                                   audio_classes=2, candidate_eos=0, candidate_audio_base=1)
        self.assertEqual(result['relativeL2'], 0.)
        self.assertEqual(result['eosMaxAbs'], 0.)
        self.assertEqual(result['argmaxAgreement'], 1.)
        wrong_eos = cropped.copy(); wrong_eos[0, 0] = -10.
        changed = music_logit_error(original, wrong_eos, eos=2, audio_base=4,
                                    audio_classes=2, candidate_eos=0, candidate_audio_base=1)
        self.assertEqual(changed['eosMaxAbs'], 15.)
        self.assertEqual(changed['argmaxAgreement'], .5)


if __name__ == '__main__':
    unittest.main()
