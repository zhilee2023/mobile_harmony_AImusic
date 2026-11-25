import hashlib
import importlib.util
import unittest

from levo_harmony.operator_lab.subgraph_pricing import Site, _marginal_ranking, price_subgraphs, rank_development_pairs, solve_packing
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

    def test_extension_bound_keeps_a_low_gain_prefix_for_larger_children(self):
        nodes = [node('a', 'UnaryOp', [0], [1]), node('b', 'UnaryOp', [1], [2]),
                 node('c', 'UnaryOp', [2], [3]), node('d', 'UnaryOp', [4], [5], opType='RELU'),
                 node('e', 'UnaryOp', [5], [6])]
        graph, report = fixture(nodes)
        for row, seconds, logical in zip(report['native']['operators']['nodes'],
                                         [.001, .001, .1, .01, .01], [10, 10, 10, 20000, 10]):
            row.update(seconds=seconds, logicalOutputBytes=logical)
        args = dict(model_sha256='test-model', bandwidth_gbps=.002, width=1, maximum=3)
        current = price_subgraphs(graph, report, **args, expansion='current')
        bound = price_subgraphs(graph, report, **args, expansion='bound')
        self.assertEqual([p['exampleNodes'] for p in current['phases']['prefill']['proposals']], [['d', 'e']])
        self.assertIn(['a', 'b', 'c'], [p['exampleNodes'] for p in bound['phases']['prefill']['proposals']])
        exhaustive = price_subgraphs(graph, report, **args, expansion='exhaustive')
        found = exhaustive['phases']['prefill']['proposals']
        self.assertTrue(exhaustive['structuralEnumerationComplete'])
        self.assertTrue({('a', 'b', 'c'), ('d', 'e')} <= {tuple(p['exampleNodes']) for p in found})

    def test_pair_shortlist_keeps_candidates_with_zero_individual_marginal(self):
        graph = TensorGraph('toy', [node(str(i), 'UnaryOp', [i], [i+1]) for i in range(4)])
        existing = [Site('bc', frozenset({1, 2}), 10)]
        new = {'ab': [Site('ab', frozenset({0, 1}), 6)], 'cd': [Site('cd', frozenset({2, 3}), 6)]}
        master, prices = solve_packing(graph, existing)
        _, single = _marginal_ranking(graph, master, existing, [{'id': i} for i in new],
                                      new, 2, None, prices=prices, weight=.5)
        self.assertIsNone(single['recommendedDevelopmentId'])
        result = rank_development_pairs(graph, master, existing, new, 2, {'remainingBudget': 2}, .5)
        self.assertEqual(result['best']['ids'], ['ab', 'cd'])
        self.assertEqual(result['best']['predictedNetMs'], 1)
        capped = rank_development_pairs(graph, master, existing, new, 2, {'remainingBudget': 1}, .5)
        self.assertEqual(capped['evaluatedPairs'], 0)

    def test_integer_marginal_gain_accounts_for_existing_fusion_and_remaining_budget(self):
        graph = TensorGraph('toy', [node(str(i), 'UnaryOp', [i], [i+1]) for i in range(4)])
        existing = [Site('ab', frozenset({0, 1}), 100)]
        candidates = {
            'bc': [Site('bc', frozenset({1, 2}), 120)],
            'cd': [Site('cd', frozenset({2, 3}), 60)],
            'abcd': [Site('abcd', frozenset(range(4)), 190)],
        }
        shortlist = [{'id': candidate, 'priorityScoreMs': gain}
                     for candidate, gain in [('bc', 120), ('cd', 60), ('abcd', 190)]]
        master, _ = solve_packing(graph, existing)
        costs = {'costByPattern': {'abcd': 2}, 'remainingBudget': 2}
        rank, status = _marginal_ranking(graph, master, existing, shortlist, candidates, 3, costs)
        self.assertEqual([(item['id'], item['predictedMarginalMs']) for item in rank],
                         [('cd', 60), ('abcd', 90), ('bc', 20)])
        self.assertTrue(status['bestInDiscoveredPoolCertified'])
        costs.update(remainingBudget=1, blockedPatternIds=['cd'])
        rank, _ = _marginal_ranking(graph, master, existing, shortlist, candidates, 3, costs)
        self.assertEqual([item['id'] for item in rank], ['bc'])

    def test_negative_reduced_profit_can_have_the_largest_integer_marginal_gain(self):
        graph = TensorGraph('toy', [node('a', 'UnaryOp', [], [0]), node('b', 'UnaryOp', [], [1]),
            node('c', 'UnaryOp', [], [2]), node('ab', 'BinaryOp', [0, 1], [3]),
            node('bc', 'BinaryOp', [1, 2], [4]), node('ca', 'BinaryOp', [2, 0], [5]),
            node('d', 'UnaryOp', [], [6]), node('e', 'UnaryOp', [6], [7])])
        existing = [Site('ab', frozenset({0, 1, 3}), 100), Site('bc', frozenset({1, 2, 4}), 100),
                    Site('ca', frozenset({2, 0, 5}), 100)]
        wide = Site('wide', frozenset(range(6)), 140)
        side = Site('side', frozenset({6, 7}), 10)
        candidates = {'wide': [wide], 'side': [side]}
        self.assertTrue(all(graph.convex(s.region) for s in [*existing, wide, side]))
        master, prices = solve_packing(graph, existing)
        self.assertAlmostEqual(master['gainMs'], 100)
        self.assertAlmostEqual(sum(prices.values()), 150)
        reduced = wide.gain_ms - sum(prices.get(n, 0) for n in wide.region)
        self.assertAlmostEqual(reduced, -10)
        # The old top-1 rank used only positive reduced profit and chose 'side'.
        proposals = [{'id': 'wide', 'priorityScoreMs': 0}, {'id': 'side', 'priorityScoreMs': 10}]
        rank, status = _marginal_ranking(graph, master, existing, proposals, candidates, 1, None, prices=prices)
        self.assertEqual(rank[0]['id'], 'wide')
        self.assertEqual(rank[0]['predictedMarginalMs'], 40)
        self.assertEqual(status['unresolvedUpperMsPerCost'], 10)
        self.assertTrue(status['bestInDiscoveredPoolCertified'])

    def test_budget_exhausted_rerank_does_not_certify_a_partial_choice(self):
        graph = TensorGraph('toy', [node(str(i), 'UnaryOp', [i], [i+1]) for i in range(5)])
        existing = [Site('ab', frozenset({0, 1}), 100)]
        pool = {'bc': [Site('bc', frozenset({1, 2}), 101)],
                'de': [Site('de', frozenset({3, 4}), 50)]}
        master, _ = solve_packing(graph, existing)
        proposals = [{'id': identifier} for identifier in pool]
        rank, status = _marginal_ranking(graph, master, existing, proposals, pool, 1, None)
        self.assertEqual(rank[0]['predictedMarginalMs'], 1)
        self.assertEqual(status['unresolvedUpperMsPerCost'], 50)
        self.assertFalse(status['bestInDiscoveredPoolCertified'])

    def test_weighted_next_choice_charges_a_reusable_implementation_once(self):
        graph = TensorGraph('toy', [node(str(i), 'UnaryOp', [i], [i+1]) for i in range(7)])
        existing = [Site('bc0', frozenset({1, 2}), 3), Site('bc1', frozenset({5, 6}), 3)]
        sites = {'eb': [Site('eb', frozenset({4, 5}), 3.5)]}
        master, prices = solve_packing(graph, existing)
        proposals = [{'id': 'eb'}]
        _, expensive = _marginal_ranking(graph, master, existing, proposals, sites,
                                         1, None, prices=prices, weight=1.)
        self.assertIsNone(expensive['recommendedDevelopmentId'])
        self.assertTrue(expensive['bestInDiscoveredPoolCertified'])
        rows, cheap = _marginal_ranking(graph, master, existing, proposals, sites,
                                        1, None, prices=prices, weight=.25)
        self.assertEqual(cheap['recommendedDevelopmentId'], 'eb')
        self.assertAlmostEqual(rows[0]['predictedMarginalMs'], .5)
        self.assertAlmostEqual(rows[0]['predictedNetMarginalMs'], .25)
        empty, prices = solve_packing(graph, [])
        reused, _ = _marginal_ranking(graph, empty, [], [{'id': 'bc'}], {'bc': existing},
                                     1, None, prices=prices, weight=1.)
        self.assertAlmostEqual(reused[0]['predictedNetMarginalMs'], 5)

    def test_stabilization_uses_same_graph_and_profile_without_changing_integer_master(self):
        graph, report = fixture([node('a', 'UnaryOp', [0], [1]), node('b', 'UnaryOp', [1], [2])])
        verified = {
            'scope': 'paired_device_measurements', 'modelSha256': 'test-model',
            'profileSha256': hashlib.sha256(_json(report).encode()).hexdigest(),
            'placements': [{'id': 'ab', 'phase': 'prefill', 'nodes': ['a', 'b'],
                'nativeMsPerCall': 2, 'fusedMsPerCall': .5, 'relativeL2': 0,
                'correctnessPassed': True, 'devicePassed': True}],
        }
        args = dict(model_sha256='test-model', bandwidth_gbps=.002, maximum=2)
        measured = price_subgraphs(graph, report, **args, verified=verified)
        smoothed = price_subgraphs(graph, report, **args, prior=measured, alpha=.5)
        phase = smoothed['phases']['prefill']
        self.assertEqual(phase['verifiedMaster']['gainMs'], 0.0)
        self.assertEqual(phase['nonzeroDualRows'], 0)
        self.assertAlmostEqual(sum(phase['searchDuals'].values()), .75)
        with self.assertRaisesRegex(ValueError, 'different graph or profile'):
            price_subgraphs(graph, report, **args, prior=dict(measured, graphSha256='other'), alpha=.5)
