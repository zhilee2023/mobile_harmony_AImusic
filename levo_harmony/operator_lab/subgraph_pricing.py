"""Bounded search for reusable MNN fusion proposals, with a measured master."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
import time

from .subgraphs import PURE_TYPES, _graphs, _json, matched_profile_rows, profile_context
from .patterns import amortized_development_weight


@dataclass(frozen=True)
class Site:
    identifier: str
    region: frozenset[int]
    gain_ms: float


def _cycle(graph, sites):
    """Return a cycle of selected placement IDs in the contracted graph."""
    owner = {}
    selected = {}
    for group, site in enumerate(sites, start=len(graph.nodes)):
        owner.update(dict.fromkeys(site.region, group))
        selected[group] = site.identifier
    edges = defaultdict(set)
    for source, targets in graph.successors.items():
        u = owner.get(source, source)
        for target in targets:
            v = owner.get(target, target)
            if u != v:
                edges[u].add(v)
    vertices = sorted(set(selected) | (set(range(len(graph.nodes))) - owner.keys()))
    indegree = dict.fromkeys(vertices, 0)
    for neighbors in edges.values():
        for v in neighbors:
            indegree[v] += 1
    ready = deque(v for v in vertices if not indegree[v])
    while ready:
        u = ready.popleft()
        for v in edges[u]:
            indegree[v] -= 1
            if not indegree[v]:
                ready.append(v)
    remaining = {v for v, count in indegree.items() if count}
    if not remaining:
        return ()
    # Every residual vertex has a residual predecessor. Following them gives a cycle.
    parent = {}
    for u in sorted(remaining):
        for v in edges[u] & remaining:
            parent[v] = u
    path = {}
    v = min(remaining)
    while v not in path:
        path[v] = len(path)
        v = parent[v]
    cycle = tuple(sorted(selected[group] for group, position in path.items()
                         if position >= path[v] and group in selected))
    if not cycle:
        raise ValueError('Original tensor graph must be acyclic')
    return cycle


def solve_packing(graph, sites, *, seconds=10, use_cplex=False):
    """Maximum gain, exact on a finite site set when CPLEX finishes optimally.

    Cycle cuts only involve selected fused regions. Their distinct contractions
    remain distinct in any extension of this selection, so the cuts are valid.
    """
    if len({site.identifier for site in sites}) != len(sites):
        raise ValueError('Duplicate site ID')
    profitable = [site for site in sites if site.gain_ms > 1e-9]
    if not profitable:
        return {'gainMs': 0.0, 'selected': [], 'optimal': True, 'cycleCuts': 0}, {}
    ids = {site.identifier: i for i, site in enumerate(profitable)}
    covered = defaultdict(list)
    for i, site in enumerate(profitable):
        for vertex in site.region:
            covered[vertex].append(i)
    if not use_cplex and all(len(columns) == 1 for columns in covered.values()) and not _cycle(graph, profitable):
        # In the disjoint relaxation choose the nonzero price on one node per site.
        prices = {min(site.region): site.gain_ms for site in profitable}
        return {'gainMs': sum(s.gain_ms for s in profitable), 'selected': list(ids),
                'optimal': True, 'cycleCuts': 0}, prices
    import cplex

    model = cplex.Cplex()
    try:
        for stream in ('log', 'results', 'warning', 'error'):
            getattr(model, 'set_' + stream + '_stream')(None)
        model.parameters.threads.set(1)
        model.parameters.timelimit.set(seconds)
        model.objective.set_sense(model.objective.sense.maximize)
        model.variables.add(obj=[site.gain_ms for site in profitable], lb=[0.0] * len(ids))
        rows = sorted(covered)
        model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(covered[v], [1.0] * len(covered[v])) for v in rows],
            rhs=[1.0] * len(rows), senses='L' * len(rows))
        model.solve()
        if model.solution.get_status() != model.solution.status.optimal:
            raise RuntimeError('Restricted pricing LP did not return optimal duals')
        prices = dict(zip(rows, model.solution.get_dual_values()))
        model.variables.set_types([(i, model.variables.type.binary) for i in range(len(ids))])
        model.parameters.mip.tolerances.mipgap.set(0)
        model.parameters.mip.tolerances.absmipgap.set(0)
        deadline = time.monotonic() + seconds
        cuts = 0
        while time.monotonic() < deadline:
            model.parameters.timelimit.set(max(.001, deadline - time.monotonic()))
            model.solve()
            if not model.solution.is_primal_feasible():
                break
            chosen = [site for site, x in zip(profitable, model.solution.get_values()) if x > .5]
            cycle = _cycle(graph, chosen)
            if not cycle:
                return {'gainMs': sum(site.gain_ms for site in chosen),
                        'selected': [site.identifier for site in chosen], 'cycleCuts': cuts,
                        'optimal': model.solution.get_status() == model.solution.status.MIP_optimal}, prices
            model.linear_constraints.add(lin_expr=[cplex.SparsePair(
                [ids[p] for p in cycle], [1.0] * len(cycle))], rhs=[len(cycle) - 1.0], senses='L')
            cuts += 1
        return {'gainMs': 0.0, 'selected': [], 'optimal': False, 'cycleCuts': cuts}, prices
    finally:
        model.end()


class PhaseSearch:
    def __init__(self, graph, rows, bandwidth_gbps, prices, *, eligible_types=None):
        self.graph = graph
        self.rows = rows
        self.prices = prices
        self.bandwidth = bandwidth_gbps * 1e6  # GB/s to bytes/ms
        allowed = PURE_TYPES if eligible_types is None else eligible_types
        self.eligible = {i for i in rows if graph.nodes[i]['type'] in allowed}
        self.labels = {}
        self.nodes_by_label = defaultdict(set)
        self.scores = {}
        for i in self.eligible:
            label = self.graph.contract((i,), rows)
            self.labels[i] = label
            self.nodes_by_label[label].add(i)

    def calls(self, region):
        counts = {self.rows[i]['calls'] for i in region}
        return next(iter(counts)) if len(counts) == 1 else 0

    def score(self, region):
        if region in self.scores:
            return self.scores[region]
        logical_bytes = 0
        # Only single-output callbacks have an unambiguous per-tensor byte count.
        # A graph output or an outside consumer still needs the materialized tensor.
        for i in region:
            outputs = self.graph.nodes[i].get('outputIndexes', [])
            if len(outputs) != 1:
                continue
            tensor = outputs[0]
            consumers = self.graph.consumers[tensor]
            if consumers and consumers <= region and tensor not in self.graph.outputs:
                logical_bytes += 2 * self.rows[i].get('logicalOutputBytes', 0)
        baseline = sum(self.rows[i]['seconds'] * 1000 for i in region)
        predicted = min(baseline, logical_bytes * self.calls(region) / self.bandwidth)
        dual = sum(self.prices.get(i, 0.0) for i in region)
        score = (predicted - dual, predicted, logical_bytes)
        self.scores[region] = score
        return score

    def extension_bound(self, region, maximum):
        """Optimistic single-site bound for the additive callback cost model."""
        remaining = maximum - len(region)
        calls = self.calls(region)
        seen, frontier = set(region), set(region)
        for _ in range(remaining):
            neighbors = set().union(*(self.graph.successors[i] | self.graph.predecessors[i]
                                      for i in frontier))
            frontier = {i for i in neighbors & self.eligible - seen if self.rows[i]['calls'] == calls}
            seen.update(frontier)
            if not frontier:
                break

        def weight(i):
            return self.rows[i]['seconds'] * 1000 - self.prices.get(i, 0.0)

        extras = sorted((max(0.0, weight(i)) for i in seen - region), reverse=True)
        return sum(map(weight, region)) + sum(extras[:remaining])

    @staticmethod
    def _beam_priority(placements, values):
        # An inexpensive, disjoint estimate for beam ordering. CPLEX handles
        # overlaps and contraction legality when pricing the retained templates.
        used = set()
        score = 0.0
        for region in placements:
            value = values[region]
            if value <= 0:
                break
            if used.isdisjoint(region):
                used.update(region)
                score += value
        return score if score else values[placements[0]]

    def beam(self, width, maximum, expansion):
        seeds = {frozenset((i,)) for i in self.eligible}
        stats = []
        discovered = []
        for size in range(2, maximum + 1):
            candidates = {child for seed in seeds
                          for child in self.graph.extensions(seed, self.eligible)
                          if self.calls(child)}
            visited = set()
            groups = []
            for region in sorted(candidates, key=lambda r: tuple(sorted(r))):
                if region in visited:
                    continue
                placements = self.matches(region)
                visited.update(placements)
                values = {r: (self.extension_bound(r, maximum) if expansion == 'bound' and size < maximum
                              else self.score(r)[0]) for r in placements}
                order = sorted(placements, key=lambda r: (-values[r], tuple(sorted(r))))
                groups.append((self._beam_priority(order, values), order))
            groups.sort(key=lambda g: (-g[0], tuple(sorted(g[1][0]))))
            # A nonconvex partial state can expand into a legal larger region.
            retained = groups[:width]
            seeds = {region for _, placements in retained for region in placements[:2]}
            legal = []
            for _, placements in retained:
                convex = [region for region in placements if self.graph.convex(region)]
                if convex:
                    legal.append(convex)
            discovered.extend(legal)
            stats.append({'size': size, 'scored': len(candidates),
                          'matchedTemplates': len(groups), 'retainedTemplates': len(retained),
                          'expansionRepresentatives': len(seeds), 'retainedConvexTemplates': len(legal)})
            if not seeds:
                break
        return discovered, stats

    def exhaustive(self, maximum):
        seen = set()
        groups = []
        scanned, eligible = Counter(), Counter()
        for region in self.graph.connected_sets(self.eligible, 2, maximum):
            size = len(region)
            scanned[size] += 1
            if not self.calls(region) or not self.graph.convex(region):
                continue
            eligible[size] += 1
            if region in seen:
                continue
            matches = {r for r in self.matches(region) if self.graph.convex(r)}
            seen.update(matches)
            groups.append(matches)
        templates = Counter(len(next(iter(group))) for group in groups)
        stats = [{'size': size, 'scored': scanned[size], 'legalPlacements': eligible[size],
                  'retainedConvexTemplates': templates[size]} for size in range(2, maximum + 1)]
        return groups, stats

    def matches(self, region):
        """Find known template placements via dependency-guided matching."""
        graph = self.graph
        template = tuple(sorted(region))
        contract = graph.contract(template, self.rows)
        mapping = {}
        matches = set()

        def visit():
            if len(mapping) == len(template):
                mapped_order = tuple(mapping[v] for v in template)
                mapped = frozenset(mapped_order)
                if self.calls(mapped) and graph.contract(mapped_order, self.rows) == contract:
                    matches.add(mapped)
                return
            remaining = [v for v in template if v not in mapping]
            v = min(remaining, key=lambda i: (
                -sum(n in mapping for n in graph.predecessors[i] | graph.successors[i]),
                len(self.nodes_by_label[self.labels[i]]), i))
            domain = self.nodes_by_label[self.labels[v]].difference(mapping.values())
            for before in graph.predecessors[v] & mapping.keys():
                domain.intersection_update(graph.successors[mapping[before]])
            for after in graph.successors[v] & mapping.keys():
                domain.intersection_update(graph.predecessors[mapping[after]])
            for node in sorted(domain):
                mapping[v] = node
                visit()
                del mapping[v]

        visit()
        return matches


def _verified_sites(graph, rows, measurements, phase, epsilon):
    names = {graph.nodes[i]['name']: i for i in rows}
    eligible = {i for i in rows if graph.nodes[i]['type'] in PURE_TYPES}
    sites = []
    for row in measurements:
        if row['phase'] != phase:
            continue
        if not isinstance(row['id'], str) or not row['id']:
            raise ValueError('Measured placement needs a stable ID')
        try:
            region = frozenset(names[name] for name in row['nodes'])
        except KeyError as error:
            raise ValueError('Measured placement must match profiled operator names') from error
        calls = {rows[i]['calls'] for i in region}
        reachable = {min(region)} if region else set()
        for _ in region:
            for i in tuple(reachable):
                reachable.update((graph.successors[i] | graph.predecessors[i]) & region)
        if (not 2 <= len(region) <= 6 or len(region) != len(row['nodes']) or reachable != region
                or not region <= eligible or len(calls) != 1 or not graph.convex(region)):
            raise ValueError('Measured placement must have a valid connected region and uniform calls')
        before, after, error = (row[key] for key in ('nativeMsPerCall', 'fusedMsPerCall', 'relativeL2'))
        if (any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in (before, after, error))
                or error > epsilon or row.get('devicePassed') is not True or row.get('correctnessPassed') is not True):
            raise ValueError('Measured placement must pass numeric and device gates')
        sites.append(Site(row['id'], region, next(iter(calls)) * (before - after)))
    return sites


def _development_settings(plan):
    plan = plan or {}
    costs = plan.get('costByPattern', {})
    default_cost = plan.get('defaultCost', 1)
    remaining = plan.get('remainingBudget')
    blocked = set(plan.get('blockedPatternIds', []))
    if any(not math.isfinite(v) or v <= 0 for v in (*costs.values(), default_cost)):
        raise ValueError('Development costs must be finite and positive')
    if remaining is not None and (not math.isfinite(remaining) or remaining < 0):
        raise ValueError('Development budget must be finite and nonnegative')
    return costs, default_cost, remaining, blocked


def _cover_price_bound(prices, existing):
    """Repair floating-point LP prices into a feasible coverage dual."""
    cover = {node: max(0.0, value) for node, value in prices.items()}
    for site in existing:
        covered = math.fsum(cover.get(node, 0.0) for node in site.region)
        if covered < site.gain_ms:
            node = min(site.region)
            cover[node] = math.nextafter(cover.get(node, 0.0) + (site.gain_ms - covered), math.inf)
    return cover


def _marginal_ranking(graph, master, existing, proposals, new_sites, limit, plan, *, prices=None, weight=None):
    if not master['optimal']:
        return [], {'status': 'base_integer_optimum_unavailable', 'bestInDiscoveredPoolCertified': False}
    costs, default_cost, remaining, blocked = _development_settings(plan)
    weighted = weight is not None

    def score(gain, cost):
        return gain - weight * cost if weighted else gain / cost

    shortlist = []
    cover = _cover_price_bound(prices, existing) if prices is not None else None
    gap = max(0.0, math.fsum(cover.values()) - master['gainMs']) if cover is not None else None
    for proposal in proposals:
        identifier = proposal['id']
        cost = costs.get(identifier, default_cost)
        if identifier in blocked or remaining is not None and cost > remaining:
            continue
        candidate_sites = new_sites[identifier]
        # The existing part of any joint feasible deployment cannot outperform
        # the existing integer optimum. The proposed part gains at most this sum.
        upper = math.fsum(max(0, site.gain_ms) for site in candidate_sites)
        if cover is not None:
            positive_reduced = math.fsum(max(0, site.gain_ms -
                math.fsum(cover.get(node, 0.0) for node in site.region)) for site in candidate_sites)
            upper = min(upper, gap + positive_reduced)
        shortlist.append((score(upper, cost), identifier, cost, upper))
    shortlist.sort(key=lambda item: (-item[0], item[1]))
    if weighted:
        limit = min(limit, sum(bound > 1e-9 for bound, *_ in shortlist))
    ranking = []
    unresolved = [value for value, *_ in shortlist[limit:]]
    for bound, identifier, cost, gross_upper in shortlist[:limit]:
        trial, _ = solve_packing(graph, existing + new_sites[identifier])
        gain = max(0.0, trial['gainMs'] - master['gainMs']) if trial['optimal'] else None
        if gain is None:
            unresolved.append(bound)
        ranking.append({'id': identifier, 'predictedMarginalMs': gain, 'estimatedCost': cost,
                        'predictedMarginalMsPerCost': gain / cost if gain is not None else None,
                        'marginalUpperMsPerCost': gross_upper / cost,
                        'developmentPenaltyMs': weight * cost if weighted else None,
                        'predictedNetMarginalMs': score(gain, cost) if weighted and gain is not None else None,
                        'selectionScore': score(gain, cost) if gain is not None else None,
                        'deploymentCostModelSolved': trial['optimal']})
    ranking.sort(key=lambda r: (r['selectionScore'] is None, -(r['selectionScore'] or 0), r['id']))
    best = ranking[0]['selectionScore'] if ranking else None
    unresolved_upper = max(unresolved, default=None)
    # Keeping the existing plan is available with zero incremental value.
    incumbent = max(0.0, best or 0.0)
    certify = (weighted or best is not None) and (
        unresolved_upper is None or unresolved_upper <= incumbent + 1e-9)
    summary = {'status': 'evaluated' if ranking else 'none_evaluated',
               'eligibleCandidates': len(shortlist), 'evaluatedCandidates': len(ranking),
               'coverageLpGapUpperMs': gap,
               'selectionMetric': 'net_marginal_ms' if weighted else 'marginal_ms_per_cost',
               'unresolvedUpperMsPerCost': unresolved_upper if not weighted else None,
               'unresolvedUpperNetMs': unresolved_upper if weighted else None,
               'recommendedDevelopmentId': ranking[0]['id'] if incumbent > 1e-9 else None,
               'bestInDiscoveredPoolCertified': certify,
               'certificateScope': 'One next development choice, discovered candidates, additive input cost model'}
    return ranking, summary


def rank_development_pairs(graph, master, existing, new_sites, shortlist_size, plan, weight):
    """Try committed pairs, including members with zero singleton marginal."""
    if not master['optimal']:
        return {'status': 'base_integer_optimum_unavailable', 'evaluatedPairs': 0, 'best': None}
    costs, default_cost, remaining, blocked = _development_settings(plan)
    candidates = []
    for identifier, sites in new_sites.items():
        cost = costs.get(identifier, default_cost)
        if identifier in blocked or remaining is not None and cost > remaining:
            continue
        gross = math.fsum(max(0.0, site.gain_ms) for site in sites)
        # Single-candidate marginal may vanish when both can jointly replace
        # an incumbent, so build the shortlist without it.
        candidates.append((gross - weight * cost, identifier, cost))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    selected = candidates[:shortlist_size]
    best = None
    evaluated = unresolved = 0
    for (upper_a, id_a, cost_a), (upper_b, id_b, cost_b) in combinations(selected, 2):
        cost = cost_a + cost_b
        if remaining is not None and cost > remaining:
            continue
        if upper_a + upper_b <= (best['predictedNetMs'] if best else 0.0) + 1e-9:
            continue
        trial, _ = solve_packing(graph, existing + new_sites[id_a] + new_sites[id_b])
        evaluated += 1
        if not trial['optimal']:
            unresolved += 1
            continue
        gain = max(0., trial['gainMs'] - master['gainMs'])
        net = gain - weight * cost
        if net > (best['predictedNetMs'] if best else 0.0) + 1e-9:
            best = {'ids': [id_a, id_b], 'predictedMarginalMs': gain, 'predictedNetMs': net,
                    'developmentCost': cost, 'developmentPenaltyMs': weight * cost}
    return {'status': 'evaluated', 'shortlistedCandidates': len(selected),
            'eligibleCandidates': len(candidates), 'evaluatedPairs': evaluated,
            'unresolvedSolves': unresolved, 'best': best,
            'scope': 'Two committed developments in gross-gain shortlist; additive modeled costs'}


def _smoothed_prices(prices, rows, previous, alpha):
    ids = {row['id']: i for i, row in rows.items()}
    if any(node not in ids or not math.isfinite(value) or value < 0 for node, value in previous.items()):
        raise ValueError('Prior dual prices must match the same profiled nodes')
    if not previous:
        return prices
    return {i: alpha * max(0.0, prices.get(i, 0)) + (1 - alpha) * previous.get(row['id'], 0)
            for i, row in rows.items()}


def price_subgraphs(export, report, *, model_sha256, bandwidth_gbps,
                    width=32, maximum=6, policy='dual', verified=None, epsilon=.001,
                    expansion='bound', top_k=10, development=None, prior=None, alpha=1.0,
                    development_weight=None, expected_runs=1, batch_shortlist=0):
    if not 2 <= maximum <= 6 or not 1 <= width <= 256:
        raise ValueError('Use 2–6 nodes and a beam width of 1–256')
    if not math.isfinite(bandwidth_gbps) or bandwidth_gbps <= 0 or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('A finite positive bandwidth and nonnegative error bound are required')
    if policy not in {'dual', 'frequency'}:
        raise ValueError('Unknown search policy')
    if (expansion not in {'bound', 'current', 'exhaustive'} or not 0 <= top_k <= 30
            or not 0 < alpha <= 1 or not 0 <= batch_shortlist <= 16):
        raise ValueError('Invalid expansion, shortlist size or dual smoothing factor')
    weight = amortized_development_weight(0. if development_weight is None else development_weight, expected_runs)
    _development_settings(development)
    native, forwards, context, lookup = profile_context(report, model_sha256)
    profile_sha = hashlib.sha256(_json(report).encode()).hexdigest()
    graph_sha = hashlib.sha256(_json(export).encode()).hexdigest()
    if prior is not None and (prior.get('profileSha256') != profile_sha or
                              prior.get('modelSha256') != model_sha256 or prior.get('graphSha256') != graph_sha):
        raise ValueError('Prior dual prices belong to a different graph or profile')
    measurements = []
    if verified is not None:
        if (verified.get('profileSha256') != profile_sha or verified.get('modelSha256') != model_sha256
                or verified.get('scope') != 'paired_device_measurements'):
            raise ValueError('Measured placement catalog must match model and profile')
        measurements = verified['placements']
        if any(m['phase'] not in forwards or not forwards[m['phase']] for m in measurements):
            raise ValueError('Measured placement has an unobserved phase')
    graph = next(_graphs(export))
    phases = {}
    for phase, count in forwards.items():
        if not count:
            continue
        rows = matched_profile_rows(graph, phase, lookup)
        verified_sites = _verified_sites(graph, rows, measurements, phase, epsilon)
        master, prices = solve_packing(graph, verified_sites, use_cplex=True)
        previous = prior['phases'].get(phase, {}).get('searchDuals', {}) if prior else {}
        active_prices = _smoothed_prices(prices, rows, previous, alpha) if policy == 'dual' else {}
        search = PhaseSearch(graph, rows, bandwidth_gbps, active_prices)
        if expansion == 'exhaustive':
            candidates, stats = search.exhaustive(maximum)
        else:
            candidates, stats = search.beam(width, maximum, expansion)
        proposals = []
        new_sites = {}
        measured_ids = {site.identifier for site in verified_sites}
        for group in candidates:
            groups = sorted(group, key=lambda r: tuple(sorted(r)))
            region = groups[0]
            contract, _ = graph.canonical(region, rows)
            signature = hashlib.sha256(_json([context, phase, contract]).encode()).hexdigest()
            sites = []
            prospective = []
            locations = {}
            for index, placement in enumerate(groups):
                identifier = str(index)
                _, predicted, size = search.score(placement)
                reduced = predicted - sum(prices.get(i, 0.0) for i in placement) if policy == 'dual' else predicted
                sites.append(Site(identifier, placement, reduced))
                proposed_id = signature + ':' + identifier
                while proposed_id in measured_ids:
                    proposed_id = 'proposal:' + proposed_id
                prospective.append(Site(proposed_id, placement, predicted))
                locations[identifier] = {'nodes': [graph.nodes[i]['name'] for i in sorted(placement)],
                    'covers': [rows[i]['id'] for i in sorted(placement)], 'calls': search.calls(placement),
                    'logicalRemovableBytesPerCall': size, 'proxyGainMs': predicted, 'priorityScoreMs': reduced}
            bundle, _ = solve_packing(graph, sites)
            new_sites[signature] = prospective
            proposals.append({'id': signature, 'size': len(region), 'compatibleHints': len(groups),
                'exampleNodes': locations['0']['nodes'], 'locations': list(locations.values()),
                'selectedLocationIndices': [int(i) for i in bundle['selected']],
                'selectedCalls': sum(locations[i]['calls'] for i in bundle['selected']),
                'priorityScoreMs': bundle['gainMs'], 'bundleOptimal': bundle['optimal'],
                'evidence': 'predicted_logical_traffic', 'fusionReady': False})
        ranked, ranking_status = _marginal_ranking(graph, master, verified_sites, proposals,
            new_sites, top_k, development, prices=prices if policy == 'dual' else None,
            weight=weight if development_weight is not None else None)
        phases[phase] = {'observedForwards': count, 'verifiedSites': len(verified_sites),
            'verifiedMaster': master, 'nonzeroDualRows': sum(v > 1e-9 for v in prices.values()),
            'dualRowsUsed': sum(v > 1e-9 for v in search.prices.values()),
            'searchDuals': {rows[i]['id']: v for i, v in active_prices.items() if v > 0},
            'levels': stats, 'proposals': sorted(proposals, key=lambda r: (-r['priorityScoreMs'], r['id'])),
            'developmentRanking': ranked, 'developmentRankingStatus': ranking_status}
        if batch_shortlist:
            phases[phase]['pairDevelopment'] = rank_development_pairs(
                graph, master, verified_sites, new_sites, batch_shortlist, development, weight)
    return {'schemaVersion': 1, 'modelId': native['modelId'], 'modelSha256': model_sha256,
        'profileSha256': profile_sha, 'graphSha256': graph_sha, 'policy': policy, 'beamWidth': width,
        'maxNodes': maximum, 'expansion': expansion, 'topK': top_k, 'dualAlpha': alpha,
        'batchShortlist': batch_shortlist,
        'developmentWeight': development_weight, 'expectedRuns': expected_runs,
        'effectiveDevelopmentWeight': weight if development_weight is not None else None,
        'proxyBandwidthGBs': bandwidth_gbps, 'expansionSitesPerTemplate': 2, 'phases': phases,
        'pricingComplete': False, 'structuralEnumerationComplete': expansion == 'exhaustive',
        'dualRelaxation': 'Primitive nonoverlap LP, before contraction cycle cuts',
        'scope': 'Discrete 2–6 node structures under phase and eligibility filters; logical traffic is a proxy. Only paired-device measurements enter the master.'}
