import unittest

from tools.price_vae_fusions import price_layout_actions


class VaeLayoutPricingTests(unittest.TestCase):
    def test_charges_only_implemented_copies_and_merges_equivalent_plans(self):
        profile = {'commands': [
            {'name': 'before', 'type': 'Raster', 'seconds': .004},
            {'name': 'conv', 'type': 'Convolution', 'seconds': .200},
            {'name': 'after', 'type': 'Raster', 'seconds': .006},
        ]}
        ranked = [
            {'selectedPositions': [[0, 1]]},
            {'selectedPositions': [[0]]},
            {'selectedPositions': [[1, 2]]},
            {'selectedPositions': [[0, 1, 2]]},
        ]
        plans = price_layout_actions(ranked, profile, {0, 2})
        self.assertEqual([p['priorRanks'] for p in plans], [[4], [3], [1, 2]])
        self.assertEqual([p['copyOnlyProxyMs'] for p in plans], [10.0, 6.0, 4.0])
        self.assertEqual(price_layout_actions(ranked, profile, {0}), [plans[-1]])
