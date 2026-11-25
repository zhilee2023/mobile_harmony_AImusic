"""Bounded search for reusable MNN fusion proposals, with a measured master."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import time

from .subgraphs import PURE_TYPES, _graphs, _json, matched_profile_rows, profile_context


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
    import cplex

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
    def __init__(self, graph, rows, bandwidth_gbps, prices):
        self.graph = graph
        self.rows = rows
        self.prices = prices
        self.bandwidth = bandwidth_gbps * 1e6  # GB/s to bytes/ms
        self.eligible = {i for i in rows if graph.nodes[i]['type'] in PURE_TYPES}
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

    def _beam_priority(self, placements):
        # An inexpensive, disjoint estimate for beam ordering. CPLEX handles
        # overlaps and contraction legality when pricing the retained templates.
        used = set()
        score = 0.0
        for region in placements:
            value = self.score(region)[0]
            if value <= 0:
                break
            if used.isdisjoint(region):
                used.update(region)
                score += value
        return score if score else self.score(placements[0])[0]

    def beam(self, width, maximum):
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
                order = sorted(placements, key=lambda r: (-self.score(r)[0], tuple(sorted(r))))
                groups.append((self._beam_priority(order), order))
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


def price_subgraphs(export, report, *, model_sha256, bandwidth_gbps,
                    width=32, maximum=6, policy='dual', verified=None, epsilon=.001):
    if not 2 <= maximum <= 6 or not 1 <= width <= 256:
        raise ValueError('Use 2–6 nodes and a beam width of 1–256')
    if not math.isfinite(bandwidth_gbps) or bandwidth_gbps <= 0 or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('A finite positive bandwidth and nonnegative error bound are required')
    if policy not in {'dual', 'frequency'}:
        raise ValueError('Unknown search policy')
    native, forwards, context, lookup = profile_context(report, model_sha256)
    profile_sha = hashlib.sha256(_json(report).encode()).hexdigest()
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
        search = PhaseSearch(graph, rows, bandwidth_gbps, prices if policy == 'dual' else {})
        candidates, stats = search.beam(width, maximum)
        proposals = []
        for group in candidates:
            groups = sorted(group, key=lambda r: tuple(sorted(r)))
            region = groups[0]
            contract, _ = graph.canonical(region, rows)
            signature = hashlib.sha256(_json([context, phase, contract]).encode()).hexdigest()
            sites = []
            locations = {}
            for index, placement in enumerate(groups):
                identifier = str(index)
                reduced, predicted, size = search.score(placement)
                sites.append(Site(identifier, placement, reduced))
                locations[identifier] = {'nodes': [graph.nodes[i]['name'] for i in sorted(placement)],
                    'covers': [rows[i]['id'] for i in sorted(placement)], 'calls': search.calls(placement),
                    'logicalRemovableBytesPerCall': size, 'proxyGainMs': predicted, 'priorityScoreMs': reduced}
            bundle, _ = solve_packing(graph, sites)
            proposals.append({'id': signature, 'size': len(region), 'compatibleHints': len(groups),
                'exampleNodes': locations['0']['nodes'], 'locations': list(locations.values()),
                'selectedLocationIndices': [int(i) for i in bundle['selected']],
                'selectedCalls': sum(locations[i]['calls'] for i in bundle['selected']),
                'priorityScoreMs': bundle['gainMs'], 'bundleOptimal': bundle['optimal'],
                'evidence': 'predicted_logical_traffic', 'fusionReady': False})
        phases[phase] = {'observedForwards': count, 'verifiedSites': len(verified_sites),
            'verifiedMaster': master, 'nonzeroDualRows': sum(v > 1e-9 for v in prices.values()),
            'dualRowsUsed': sum(v > 1e-9 for v in search.prices.values()),
            'levels': stats, 'proposals': sorted(proposals, key=lambda r: (-r['priorityScoreMs'], r['id']))}
    return {'schemaVersion': 1, 'modelId': native['modelId'], 'modelSha256': model_sha256,
        'profileSha256': profile_sha, 'policy': policy, 'beamWidth': width, 'maxNodes': maximum,
        'proxyBandwidthGBs': bandwidth_gbps, 'expansionSitesPerTemplate': 2, 'phases': phases,
        'pricingComplete': False, 'dualRelaxation': 'Primitive nonoverlap LP, before contraction cycle cuts',
        'scope': 'Finite beam and logical tensor traffic proxy; only supplied paired-device measurements enter the master. No end-to-end gain estimate.'}
