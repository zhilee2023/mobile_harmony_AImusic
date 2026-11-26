import unittest

from tools.summarize_vae_pairs import summarize_factorial


class VaeFactorialTest(unittest.TestCase):
    def test_effects_use_within_block_counterfactuals(self):
        times = {'00': 100, '10': 93, '01': 97, '11': 88}
        rows = [dict(mode=cell, windowMs=value, peakRssMiB=30, screenAwake=True, pcmBitIdentical=True)
                for cell, value in times.items()]
        summary = summarize_factorial(dict(runs=rows, pairs=[dict(windowMs=times)]))
        self.assertEqual(summary['blocks'], [dict(aliasWithoutEpilogueMs=7, aliasWithEpilogueMs=9,
                                                 epilogueWithoutAliasMs=3, epilogueWithAliasMs=5,
                                                 interactionSavingMs=2)])
