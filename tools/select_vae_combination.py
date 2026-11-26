"""Select nonconflicting, measured VAE rewrites with CPLEX."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

from run_vae_alias_benchmark import plans, read
from vae_storage_graph import storage_peak


def solve(columns, accept=None, max_actions=None):
    import cplex

    names = list(columns)
    with cplex.Cplex() as solver:
        for stream in ('log', 'results', 'warning', 'error'):
            getattr(solver, 'set_' + stream + '_stream')(None)
        solver.objective.set_sense(solver.objective.sense.maximize)
        solver.variables.add(names=names, lb=[0.] * len(names), obj=[columns[name][1] for name in names])
        covering = defaultdict(list)
        for name, (operations, _) in columns.items():
            for operation in operations:
                covering[operation].append(name)
        solver.linear_constraints.add(lin_expr=[cplex.SparsePair(choices, [1.] * len(choices))
                                               for choices in covering.values()],
                                      senses='L' * len(covering), rhs=[1.] * len(covering))
        if max_actions is not None:
            solver.linear_constraints.add(lin_expr=[cplex.SparsePair(names, [1.] * len(names))],
                                          senses='L', rhs=[float(max_actions)])
        solver.solve()
        if solver.solution.get_status() != solver.solution.status.optimal:
            raise RuntimeError('LP did not reach an optimum')
        relaxed = solver.solution.get_objective_value()
        solver.variables.set_types([(name, 'B') for name in names])
        rejected = 0
        while True:
            solver.solve()
            status = solver.solution.get_status()
            if status == solver.solution.status.MIP_infeasible:
                return relaxed, None, [], rejected
            if status != solver.solution.status.MIP_optimal:
                raise RuntimeError('MILP did not reach an optimum')
            chosen = [name for name in names if solver.solution.get_values(name) > .5]
            if accept is None or accept(chosen):
                return relaxed, solver.solution.get_objective_value(), chosen, rejected
            # Exclude this exact assignment. Shared storage makes the memory
            # bound depend on the combination, so retain its subsets and supersets.
            weights = [1. if name in chosen else -1. for name in names]
            solver.linear_constraints.add(lin_expr=[cplex.SparsePair(names, weights)], senses='L',
                                          rhs=[len(chosen) - 1.])
            rejected += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ranking', 'profile', 'layout_trials', 'epilogue_trials', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--primitive-graph', type=Path)
    parser.add_argument('--max-modeled-peak-mib', type=float)
    parser.add_argument('--objective', choices=('measured', 'frequency', 'traffic'), default='measured')
    parser.add_argument('--max-actions', type=int)
    args = parser.parse_args()
    cap = args.max_modeled_peak_mib
    if cap is not None and (not args.primitive_graph or not math.isfinite(cap) or cap <= 0):
        parser.error('A positive finite memory cap requires --primitive-graph')
    if args.objective == 'traffic' and not args.primitive_graph:
        parser.error('The traffic objective requires --primitive-graph')
    if args.max_actions is not None and args.max_actions < 0:
        parser.error('The action limit must be nonnegative')
    source = plans(read(args.ranking), read(args.profile))
    layout = read(args.layout_trials)
    epilogue = read(args.epilogue_trials)
    if layout['status'] != 'complete' or epilogue['status'] != 'complete':
        raise ValueError('Complete measurements are required')
    if any(layout['fingerprints'][key] != epilogue['fingerprints'][key]
           for key in ('graphSha256', 'weightSha256', 'latentSha256')):
        raise ValueError('Measurements use different VAE files or inputs')
    samples = defaultdict(list)
    for row in layout['candidates']:
        if row['pcmBitIdentical']:
            rasters = tuple(source[row['rank']]['rasterNames'])
            samples[rasters].append(row['measuredLocalCopyDeltaMs'])
    if {run['mode'] for run in epilogue['runs']} != {'baseline', 'fused'} or not all(
            run['pcmBitIdentical'] for run in epilogue['runs']):
        raise ValueError('Expected an exact PCM epilogue comparison')
    columns = {f'layout-{index}': (names, median(times)) for index, (names, times) in enumerate(samples.items(), 1)}
    epilogue_name = 'deconvolution-epilogue'
    columns[epilogue_name] = (tuple(f'deconvolution-{index}' for index in range(5)),
                               median(pair['epilogueSavedMs'] for pair in epilogue['pairs']))
    def aliases(chosen):
        return sorted(op for name in chosen if name != epilogue_name for op in columns[name][0])

    graph = read(args.primitive_graph) if args.primitive_graph else None

    def fits(chosen):
        return storage_peak(graph, aliases(chosen))['peak']['modeledBytes'] <= cap * 2**20

    costs = columns
    units = 'Ms'
    if args.objective == 'frequency':
        # Each eligible rewrite site fires once per fixed VAE window.
        costs = {name: (operations, len(operations)) for name, (operations, _) in columns.items()}
        units = 'Calls'
    elif args.objective == 'traffic':
        sizes = {tensor['tensor']: tensor['storageBytes'] for tensor in graph['tensors']}
        traffic = {site['name']: 2 * sizes[site['destinationTensor']] for site in graph['copyAliases']}
        deconv = [op['local']['executorMeasurement'] for op in graph['operators'] if op['type'] == 'Deconvolution']
        if len(deconv) != 5 or any(op['workers'] != 4 for op in deconv):
            raise ValueError('Expected five four-worker deconvolutions for the measured epilogue')
        # A four-worker fused epilogue eliminates six accesses per output element.
        traffic.update({f'deconvolution-{index}': 6 * op['outputBytes'] for index, op in enumerate(deconv)})
        costs = {name: (operations, sum(traffic[op] for op in operations) / 2**20)
                 for name, (operations, _) in columns.items()}
        units = 'LogicalTrafficMiB'
    relaxed, integer, chosen, rejected = solve(costs, fits if cap is not None else None, args.max_actions)
    result = dict(columns=len(columns), chosen=chosen,
                  rasterNames=aliases(chosen), epilogue=epilogue_name in chosen)
    result.update({f'lp{units}': relaxed, f'mip{units}': integer})
    if args.objective != 'measured':
        result['priorLocalSavingsMs'] = sum(columns[name][1] for name in chosen) if integer is not None else None
    if graph is not None:
        result['modeledStorage'] = dict(baseline=storage_peak(graph, [])['peak'],
                                        selected=storage_peak(graph, aliases(chosen))['peak'] if integer is not None else None,
                                        capMiB=cap, rejectedCombinations=rejected, feasible=integer is not None,
                                        lpIncludesMemoryCap=False)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
