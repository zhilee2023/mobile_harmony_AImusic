import copy
import unittest

from levo_harmony.operator_lab.subgraphs import TensorGraph, mine_connected_subgraphs


def node(name, kind, inputs, outputs, **main):
    return dict(name=name, type=kind, inputIndexes=inputs, outputIndexes=outputs, main=main)


def fixture(nodes):
    graph = dict(oplists=nodes)
    measured = [dict(id='prefill/' + str(i), phase='prefill', name=n['name'], type=n['type'],
        calls=1, seconds=.001, inputShapes=[[1, 4] for _ in n['inputIndexes']],
        outputShapes=[[1, 4] for _ in n['outputIndexes']], backendObservations={'cpu': 1})
        for i, n in enumerate(nodes) if n['type'] not in {'Input', 'Const'}]
    report = dict(controlledAppTimingAccepted=True, modelManifest={'files': [dict(relative='lm/llm.mnn', sha256='test-model')]},
        native=dict(status='completed', instrumented=True, modelId='fixture-only',
        rounds=[dict(decodeForwards=0)], operators=dict(nodes=measured)))
    return graph, report


def run(nodes, **kwargs):
    graph, report = fixture(nodes)
    return mine_connected_subgraphs(graph, report, model_sha256='test-model', **kwargs)


def chains():
    return [node('a' + str(i), 'UnaryOp', [i * 10], [i * 10 + 1], opType='SILU')
        for i in range(2)] + [node('b' + str(i), 'Convolution', [i * 10 + 1], [i * 10 + 2])
        for i in range(2)] + [node('c' + str(i), 'BinaryOp', [i * 10 + 2, i * 10 + 3], [i * 10 + 4], opType='ADD')
        for i in range(2)]


class SubgraphTests(unittest.TestCase):
    def test_actual_tensor_edges_find_repeated_xy_xyz_despite_nonadjacent_rows(self):
        hints = run(chains())['patternHints']
        self.assertEqual(sorted((h['size'], h['staticOccurrences']) for h in hints), [(2, 2), (2, 2), (3, 2)])
        self.assertTrue(all(h['convexRegionVerified'] and not h['fusionReady'] and not h['speedupProven'] for h in hints))

    def test_adjacent_but_disconnected_profile_rows_do_not_make_pairs(self):
        nodes = [node(str(i), 'UnaryOp', [i * 2], [i * 2 + 1]) for i in range(6)]
        self.assertEqual(run(nodes)['patternHints'], [])

    def test_parallel_branches_and_join_are_found(self):
        nodes = []
        for i in range(2):
            t = i * 10
            nodes += [node('up' + str(i), 'Convolution', [t], [t + 1]),
                      node('gate' + str(i), 'Convolution', [t], [t + 2]),
                      node('join' + str(i), 'BinaryOp', [t + 1, t + 2], [t + 3], opType='MUL_SILU')]
        hints = run(nodes)['patternHints']
        join = [h for h in hints if h['size'] == 3]
        self.assertEqual(len(join), 1)
        self.assertEqual(join[0]['topology'], 'join')
        self.assertEqual(join[0]['staticOccurrences'], 2)

    def test_shared_external_input_alias_is_part_of_contract(self):
        nodes = []
        for i in range(2):
            t = i * 10
            nodes += [node('a' + str(i), 'BinaryOp', [t, t if i == 0 else t + 3], [t + 1]),
                      node('b' + str(i), 'UnaryOp', [t + 1], [t + 2])]
        self.assertEqual(run(nodes)['patternHints'], [])

    def test_port_order_is_not_silently_treated_as_commutative(self):
        nodes = [node('a0', 'UnaryOp', [0], [1]), node('b0', 'BinaryOp', [1, 3], [2], opType='SUB'),
                 node('a1', 'UnaryOp', [10], [11]), node('b1', 'BinaryOp', [13, 11], [12], opType='SUB')]
        self.assertEqual(run(nodes)['patternHints'], [])

    def test_different_subtypes_layouts_quantization_are_not_merged(self):
        for variant in ['subtype', 'layout', 'quant']:
            nodes = chains()[:4]
            if variant == 'subtype':
                nodes[1]['main']['opType'] = 'RELU'
            elif variant == 'layout':
                nodes[1]['defaultDimentionFormat'] = 'NHWC'
            else:
                nodes[3]['main']['quanParameter'] = {'aMaxOrBits': 4}
            self.assertEqual(run(nodes)['patternHints'], [])

    def test_external_fanout_intermediate_is_retained_as_required_output(self):
        nodes = chains()[:4]
        nodes += [node('outside' + str(i), 'Attention', [i * 10 + 1], [i * 10 + 5]) for i in range(2)]
        hint = run(nodes)['patternHints'][0]
        self.assertTrue(any(o['externalConsumers'] == 1 for n in hint['contract'] for o in n['outputs']))

    def test_nonconvex_escape_and_reentry_is_rejected(self):
        nodes = [node('a', 'UnaryOp', [0], [1]), node('b', 'UnaryOp', [1], [2]),
                 node('c', 'BinaryOp', [1, 2], [3])]
        graph = TensorGraph('test', nodes)
        self.assertFalse(graph.convex(frozenset({0, 2})))
        self.assertTrue(graph.convex(frozenset({0, 1, 2})))

    def test_stateful_attention_and_control_flow_are_not_eligible(self):
        for kind in ['Attention', 'While', 'If', 'Extra']:
            nodes = [node('a' + str(i), kind, [i * 10], [i * 10 + 1]) for i in range(2)]
            nodes += [node('b' + str(i), 'UnaryOp', [i * 10 + 1], [i * 10 + 2]) for i in range(2)]
            self.assertEqual(run(nodes)['patternHints'], [])

    def test_cycles_duplicate_producers_and_source_hash_mismatch_fail(self):
        with self.assertRaises(ValueError):
            TensorGraph('test', [node('a', 'UnaryOp', [1], [2]), node('b', 'UnaryOp', [2], [1])])
        with self.assertRaises(ValueError):
            TensorGraph('test', [node('a', 'UnaryOp', [0], [2]), node('b', 'UnaryOp', [0], [2])])
        graph, report = fixture(chains())
        with self.assertRaisesRegex(ValueError, 'SHA'):
            mine_connected_subgraphs(graph, report, model_sha256='wrong')

    def test_missing_ambiguous_profile_and_different_call_counts_not_invented(self):
        for variant in ['missing', 'ambiguous', 'calls']:
            graph, report = fixture(chains()[:4])
            rows = report['native']['operators']['nodes']
            if variant == 'missing':
                rows.pop()
            elif variant == 'ambiguous':
                rows.append(copy.deepcopy(rows[-1]))
            else:
                rows[-1]['calls'] = 2
            result = mine_connected_subgraphs(graph, report, model_sha256='test-model')
            self.assertEqual(result['patternHints'], [])

    def test_constants_compared_by_digest_without_leaking_values(self):
        nodes = chains()[:4]
        nodes[0]['inputIndexes'] = [50]
        nodes[1]['inputIndexes'] = [51]
        nodes += [node('c0', 'Const', [], [50], int32s=[1234567]), node('c1', 'Const', [], [51], int32s=[1234567])]
        hints = run(nodes)['patternHints']
        self.assertEqual(len(hints), 1)
        self.assertNotIn('1234567', str(hints))
        nodes[-1]['main']['int32s'] = [7654321]
        self.assertEqual(run(nodes)['patternHints'], [])

    def test_learned_weights_parametrized_but_not_replaced_or_exposed(self):
        nodes = chains()[:4]
        for i in range(2):
            nodes[i + 2]['main'].update(weight=[9876543 + i], external=[i * 100, 4, 4])
        hints = run(nodes)['patternHints']
        self.assertEqual(len(hints), 1)
        self.assertNotIn('987654', str(hints))
        self.assertFalse(hints[0]['fusionReady'])

    def test_external_constants_with_unknown_values_do_not_merge_by_size(self):
        nodes = chains()[:4]
        nodes[0]['inputIndexes'] = [50]
        nodes[1]['inputIndexes'] = [51]
        nodes += [node('c0', 'Const', [], [50], external=[100, 4]),
                  node('c1', 'Const', [], [51], external=[200, 4])]
        self.assertEqual(run(nodes)['patternHints'], [])

    def test_overlapping_occurrences_keep_conflicts_and_unique_cost(self):
        nodes = [node('a', 'UnaryOp', [0], [1]), node('b', 'UnaryOp', [1], [2]),
                 node('c', 'UnaryOp', [2], [3]), node('d', 'UnaryOp', [3], [4])]
        hints = run(nodes, maximum=2)['patternHints']
        # Boundary fanout/graph-output flags make the last pair a separate hint.
        hint = next(h for h in hints if h['overlappingOccurrences'])
        self.assertGreater(hint['baselineCallbackTotalMs'], hint['uniqueCoveredCallbackMs'])
        self.assertEqual(hint['uniqueCoveredCallbackMs'], 3)

    def test_decode_frequency_is_observed_not_assumed_64_steps(self):
        graph, report = fixture(chains()[:4])
        report['native']['rounds'][0]['decodeForwards'] = 2
        for row in report['native']['operators']['nodes']:
            row.update(phase='decode', calls=2, seconds=.002)
        hint = mine_connected_subgraphs(graph, report, model_sha256='test-model')['patternHints'][0]
        self.assertEqual(hint['observedInvocations'], 4)
        self.assertEqual(hint['callsPerForward'], 2)

    def test_rejected_controls_cannot_rank_performance(self):
        graph, report = fixture(chains())
        report['controlledAppTimingAccepted'] = False
        with self.assertRaisesRegex(ValueError, 'controls'):
            mine_connected_subgraphs(graph, report, model_sha256='test-model')


if __name__ == '__main__':
    unittest.main()
