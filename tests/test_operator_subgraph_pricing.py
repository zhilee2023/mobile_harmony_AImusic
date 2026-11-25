import hashlib
import importlib.util
import unittest

from levo_harmony.operator_lab.subgraph_pricing import Site, price_subgraphs, solve_packing
from levo_harmony.operator_lab.subgraphs import TensorGraph, _json
from test_operator_subgraphs import fixture, node


@unittest.skipUnless(importlib.util.find_spec('cplex'), 'Optional CPLEX solver')
class PricingTests(unittest.TestCase):
    def test_disjoint_convex_regions_may_still_need_a_contraction_cut(self):
        graph = TensorGraph('toy', [node('a', 'UnaryOp', [], [0]), node('c', 'UnaryOp', [], [1]),
            node('b', 'BinaryOp', [0, 1], [2]), node('d', 'BinaryOp', [0, 1], [3])])
        sites = [Site('ab', frozenset({0, 2}), 3), Site('cd', frozenset({1, 3}), 4)]
        self.assertTrue(all(graph.convex(s.region) for s in sites))
        result, duals = solve_packing(graph, sites)
        self.assertTrue(result['optimal'])
        self.assertEqual(result['selected'], ['cd'])
        self.assertEqual(result['cycleCuts'], 1)
        self.assertAlmostEqual(sum(duals.values()), 7)  # LP before the cycle cut.

    def test_dual_feedback_changes_the_beam_without_promoting_predictions_to_master(self):
        nodes = [node('a', 'UnaryOp', [0], [1]), node('b', 'UnaryOp', [1], [2]),
                 node('c', 'UnaryOp', [3], [4], opType='RELU'), node('d', 'UnaryOp', [4], [5])]
        graph, report = fixture(nodes)
        for index, row in enumerate(report['native']['operators']['nodes']):
            row['logicalOutputBytes'] = 2000 if index == 0 else 1000
        verified = {
            'scope': 'paired_device_measurements', 'modelSha256': 'test-model',
            'profileSha256': hashlib.sha256(_json(report).encode()).hexdigest(),
            'placements': [{'id': 'ab', 'phase': 'prefill', 'nodes': ['a', 'b'],
                'nativeMsPerCall': 2, 'fusedMsPerCall': 0, 'relativeL2': 0,
                'correctnessPassed': True, 'devicePassed': True}],
        }
        common = dict(model_sha256='test-model', bandwidth_gbps=.002, width=1, maximum=2)
        cold = price_subgraphs(graph, report, **common)['phases']['prefill']
        plain = price_subgraphs(graph, report, verified=verified, policy='frequency', **common)['phases']['prefill']
        guided = price_subgraphs(graph, report, verified=verified, **common)['phases']['prefill']
        self.assertEqual(cold['nonzeroDualRows'], 0)
        self.assertEqual(cold['proposals'][0]['exampleNodes'], ['a', 'b'])
        self.assertEqual(plain['proposals'][0]['exampleNodes'], ['a', 'b'])
        self.assertEqual(plain['dualRowsUsed'], 0)
        self.assertEqual(guided['proposals'][0]['exampleNodes'], ['c', 'd'])
        self.assertGreater(guided['dualRowsUsed'], 0)
        self.assertEqual(guided['verifiedMaster']['selected'], ['ab'])
        self.assertFalse(guided['proposals'][0]['fusionReady'])
        verified['placements'][0]['devicePassed'] = False
        with self.assertRaisesRegex(ValueError, 'gates'):
            price_subgraphs(graph, report, verified=verified, **common)

    def test_four_node_fork_join_and_external_fanout_use_real_tensor_edges(self):
        nodes = []
        for i in range(2):
            t = i * 10
            nodes += [node('root' + str(i), 'UnaryOp', [t], [t + 1]),
                      node('up' + str(i), 'Convolution', [t + 1], [t + 2]),
                      node('gate' + str(i), 'Convolution', [t + 1], [t + 3]),
                      node('join' + str(i), 'BinaryOp', [t + 2, t + 3], [t + 4], opType='MUL_SILU'),
                      node('outside' + str(i), 'Attention', [t + 2], [t + 5])]
        graph, report = fixture(nodes)
        for row in report['native']['operators']['nodes']:
            row['logicalOutputBytes'] = 1000
        result = price_subgraphs(graph, report, model_sha256='test-model',
                                 bandwidth_gbps=.002, width=16, maximum=4)
        proposal = next(p for p in result['phases']['prefill']['proposals'] if p['size'] == 4)
        self.assertEqual(proposal['compatibleHints'], 2)
        self.assertEqual(proposal['selectedCalls'], 2)
        self.assertEqual(proposal['locations'][0]['logicalRemovableBytesPerCall'], 4000)
        self.assertFalse(result['pricingComplete'])
