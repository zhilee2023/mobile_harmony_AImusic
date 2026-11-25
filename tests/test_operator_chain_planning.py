import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import random
import unittest

from levo_harmony.operator_lab.chain_planning import chain_dp, chain_marginal
from levo_harmony.operator_lab.patterns import Catalog, Kernel, Placement, solve_master


def example(name):
    path = Path(__file__).resolve().parents[1] / 'configs/operator_lab' / name
    data = json.loads(path.read_text(encoding='utf-8'))
    return Catalog(data['nodes'], [Kernel(**k) for k in data['kernels']],
                   [Placement(**p) for p in data['placements']])


class ChainTests(unittest.TestCase):
    def test_batch_can_improve_when_each_singleton_is_worse_after_development_cost(self):
        catalog = example('batch_development_example.json')
        single = chain_marginal(catalog, 2, development_weight=.5)
        paired = chain_marginal(catalog, 2, development_weight=.5, max_batch=2)
        self.assertEqual(single['screened']['best']['kernels'], [])
        best = paired['screened']['best']
        self.assertEqual(best, paired['exhaustive']['best'])
        self.assertEqual(best['kernels'], ['AB', 'CD'])
        self.assertEqual(best['marginalMs'], 2)
        self.assertEqual(best['predictedNetMs'], 1)
        blocked = chain_marginal(catalog, 1, development_weight=.5, max_batch=2)
        self.assertEqual(blocked['screened']['best']['kernels'], [])

    def test_chain_dp_pays_for_repeated_kernel_once_and_keeps_budget(self):
        catalog = example('fixed_development_example.json')
        plan = chain_dp(catalog, 1, development_weight=.75)
        self.assertEqual(plan['selected'].count('bc_first') + plan['selected'].count('bc_last'), 2)
        self.assertEqual(plan['developmentCost'], 1)
        self.assertEqual(plan['objectiveMs'], 13.75)
        self.assertEqual(chain_dp(catalog, 1, spent=1, development_weight=.75)['objectiveMs'], 19)

    def test_missing_prefix_potential_disables_pruning(self):
        catalog = Catalog(2, [Kernel('native', 0, 'x', native=True), Kernel('k', 1, 'x')],
                          [Placement('old', 'native', (0, 1), 1, 10, 'x'),
                           Placement('new_a', 'k', (0,), 1, 2, 'x'),
                           Placement('new_b', 'k', (1,), 1, 2, 'x')])
        result = chain_marginal(catalog, 1, development_weight=1)
        self.assertFalse(result['exactPotentialAvailable'])
        self.assertEqual(result['screened']['best']['predictedNetMs'], 5)

    def test_conditional_second_attempt_has_value_even_when_commitment_loses(self):
        catalog = example('batch_development_example.json')
        prior = .65
        cost_per_attempt = .5
        outcomes = {}
        for accepted in [set(), {'AB'}, {'CD'}, {'AB', 'CD'}]:
            kernels = [replace(k, developed=k.developed or k.id in accepted)
                       for k in catalog.kernels.values()]
            deployed = chain_dp(Catalog(catalog.nodes, kernels, list(catalog.placements.values())), 0)
            outcomes[frozenset(accepted)] = 10 - deployed['runtimeMs']
        self.assertEqual(outcomes, {frozenset(): 0, frozenset({'AB'}): 0,
                                   frozenset({'CD'}): 0, frozenset({'AB', 'CD'}): 2})
        success_gain = outcomes[frozenset({'AB', 'CD'})]
        committed = prior**2 * success_gain - 2*cost_per_attempt
        conditional = -cost_per_attempt + prior * max(0., prior*success_gain - cost_per_attempt)
        self.assertAlmostEqual(committed, -.155)
        self.assertAlmostEqual(conditional, .02)

    @unittest.skipUnless(importlib.util.find_spec('cplex'), 'Optional CPLEX solver')
    def test_random_chain_oracles_and_screened_choices_agree(self):
        rng = random.Random(42)
        for case in range(24):
            kernels = [Kernel('native', 0, 'x', native=True)]
            latencies = [rng.randint(2, 8) for _ in range(7)]
            placements = [Placement(f'n{i}', 'native', (i,), 3, base, 'x') for i, base in enumerate(latencies)]
            for index in range(5):
                name = f'k{index}'
                kernels.append(Kernel(name, rng.randint(1, 3), 'x', developed=index == 0))
                for site in range(3):
                    start = rng.randrange(5)
                    stop = start + rng.choice([2, 3])
                    time = max(1, sum(latencies[start:stop]) - rng.randint(-3, 7))
                    placements.append(Placement(f'{name}s{site}', name, tuple(range(start, stop)),
                                                3, time, 'x'))
            catalog = Catalog(7, kernels, placements)
            weight = (case % 4) * 1.25
            oracle = solve_master(catalog, 4, integer=True, development_weight=weight)
            dp = chain_dp(catalog, 4, development_weight=weight)
            self.assertAlmostEqual(dp['objectiveMs'], oracle['objectiveMs'])
            study = chain_marginal(catalog, 4, development_weight=weight, max_batch=2)
            self.assertEqual(study['screened']['best'], study['exhaustive']['best'])
