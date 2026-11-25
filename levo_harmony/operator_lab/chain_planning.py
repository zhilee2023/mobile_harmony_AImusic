"""Exact small-chain planning and shortest-path sensitivity on known costs."""
import math
import time
from collections import defaultdict
from itertools import combinations

from .patterns import amortized_development_weight, eligible


def _chain_edges(catalog, *, allow_predictions, workspace_budget):
    outgoing = [[] for _ in range(catalog.nodes)]
    for placement in eligible(catalog, allow_predictions=allow_predictions, workspace_budget=workspace_budget):
        region = sorted(placement.covers)
        begin, end = region[0], region[-1] + 1
        if len(region) != end - begin:
            raise ValueError('Chain planning requires consecutive node intervals')
        outgoing[begin].append((end, placement))
    return outgoing


def _remaining_budget(budget, spent):
    if any(not math.isfinite(v) or v < 0 for v in (budget, spent)) or spent > budget:
        raise ValueError('Invalid development budget ledger')
    return budget - spent


def chain_dp(catalog, budget, *, spent=0, development_weight=0., expected_runs=1,
             allow_predictions=False, workspace_budget=None):
    """Optimize runtime plus fixed development cost in the given finite chain catalog."""
    remaining = _remaining_budget(budget, spent)
    weight = amortized_development_weight(development_weight, expected_runs)
    edges = _chain_edges(catalog, allow_predictions=allow_predictions, workspace_budget=workspace_budget)
    charged = [k for k in catalog.kernels.values() if k.incremental_cost > 0]
    bits = {k.id: 1 << i for i, k in enumerate(charged)}
    costs = {0: 0.0}
    # Minimum runtime for a fixed development set. The penalty and budget are
    # constant across paths with that set, so only one predecessor is needed.
    states = [dict() for _ in range(catalog.nodes + 1)]
    states[0][0] = (0.0, None)
    for begin, labels in enumerate(states[:-1]):
        for mask, (runtime, _) in labels.items():
            for end, p in edges[begin]:
                next_mask = mask | bits.get(p.kernel, 0)
                if next_mask not in costs:
                    costs[next_mask] = math.fsum(k.incremental_cost for i, k in enumerate(charged)
                                                if next_mask & (1 << i))
                if costs[next_mask] > remaining:
                    continue
                next_time = runtime + p.runtime_cost
                prior = states[end].get(next_mask)
                if prior is None or next_time < prior[0]:
                    states[end][next_mask] = (next_time, (begin, mask, p.id))
    tail = states[-1]
    counts = sum(map(len, states))
    if not tail:
        return {'feasible': False, 'optimal': False, 'objectiveMs': None, 'stateCount': counts, 'selected': []}
    final = min(tail, key=lambda mask: (tail[mask][0] + weight * costs[mask], costs[mask], mask))
    path = []
    position, mask = catalog.nodes, final
    while position:
        position, mask, placement = states[position][mask][1]
        path.append(placement)
    runtime = tail[final][0]
    development = costs[final]
    return {'feasible': True, 'optimal': True, 'objectiveMs': runtime + weight * development,
            'runtimeMs': runtime, 'developmentCost': development, 'stateCount': counts,
            'developmentPenaltyMs': weight * development, 'effectiveDevelopmentWeight': weight,
            'selected': list(reversed(path)), 'performanceProven': False,
            'scope': 'Exact DP over finite serial intervals and implementation sets'}


def _distances(outgoing, library):
    distance = [0.] + [math.inf] * len(outgoing)
    for begin, edges in enumerate(outgoing):
        for end, placement in edges:
            if placement.kernel in library:
                distance[end] = min(distance[end], distance[begin] + placement.runtime_cost)
    return distance


def chain_marginal(catalog, budget, *, spent=0, development_weight=0., expected_runs=1,
                   allow_predictions=False, workspace_budget=None, max_batch=1):
    """Compare exhaustive next-choice scoring against certified single-item screening."""
    if max_batch not in {1, 2}:
        raise ValueError('Chain experiment supports one or two developments per decision')
    remaining = _remaining_budget(budget, spent)
    weight = amortized_development_weight(development_weight, expected_runs)
    outgoing = _chain_edges(catalog, allow_predictions=allow_predictions, workspace_budget=workspace_budget)
    existing = {k.id for k in catalog.kernels.values() if k.native or k.developed}
    initial = _distances(outgoing, existing)
    base = initial[-1]
    if not math.isfinite(base):
        raise ValueError('Chain marginal study needs a deployable existing library')
    by_kernel = defaultdict(list)
    for begin, edges in enumerate(outgoing):
        for end, placement in edges:
            if placement.kernel not in existing:
                by_kernel[placement.kernel].append((begin, end, placement))
    candidates = [k for k in catalog.kernels.values()
                  if k.id in by_kernel and k.incremental_cost <= remaining]
    exact_potential = all(math.isfinite(value) for value in initial)
    entries = []
    for k in candidates:
        upper = (math.fsum(max(0., initial[end] - initial[begin] - p.runtime_cost)
                          for begin, end, p in by_kernel[k.id]) - weight*k.incremental_cost
                 if exact_potential else math.inf)
        entries.append((k, upper))

    def compare(screen):
        tick = time.perf_counter()
        best = {'kernels': [], 'runtimeMs': base, 'marginalMs': 0., 'predictedNetMs': 0.}
        evaluated = pruned = 0
        # For fixed intervals, new edges are the only ones with negative
        # reduced length. Telescoping proves the bound even if they conflict.
        for count in range(1, max_batch + 1):
            for group in combinations(entries, count):
                kernels = [k for k, _ in group]
                cost = math.fsum(k.incremental_cost for k in kernels)
                if cost > remaining:
                    continue
                upper = math.fsum(value for _, value in group)
                if screen and upper < best['predictedNetMs'] - 1e-9:
                    pruned += 1
                    continue
                identifiers = sorted(k.id for k in kernels)
                runtime = _distances(outgoing, existing | set(identifiers))[-1]
                evaluated += 1
                gain = base - runtime
                net = gain - weight * cost
                if net > best['predictedNetMs'] + 1e-9:
                    best = {'kernels': identifiers, 'runtimeMs': runtime, 'marginalMs': gain,
                            'predictedNetMs': net, 'newDevelopmentCost': cost}
        return {'best': best, 'evaluated': evaluated, 'pruned': pruned, 'seconds': time.perf_counter()-tick}

    return {'baselineRuntimeMs': base, 'exactPotentialAvailable': exact_potential,
            'candidateCount': len(candidates), 'maxBatch': max_batch,
            'exhaustive': compare(False), 'screened': compare(True), 'performanceProven': False,
            'scope': 'Bounds are valid for supplied finite edge costs. Predictions remain model inputs.'}
