"""Connected XY/XYZ hints from tensor edges, not callback order.

No generated kernel, automatic rewrite, or speedup is implied by a hint. Raw
MiniReflect exports stay private; parameter values are never emitted here.
"""
from collections import Counter, defaultdict, deque
import hashlib
import itertools
import json
import math


PURE_TYPES = frozenset({'Convolution', 'ConvolutionDepthwise', 'MatMul',
    'BinaryOp', 'UnaryOp', 'LayerNorm', 'ReLU', 'ReLU6', 'PReLU', 'Softmax',
    'RoPE', 'GELU', 'Reshape', 'Transpose', 'ConvertTensor'})
PAYLOADS = frozenset({'weight', 'bias', 'alpha', 'beta', 'gamma', 'buffer', 'slope'})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _attrs(value, *, constants=False):
    """Retain non-payload semantics; parameter slots must be runtime bound.

    external[0] is an offset, not a kernel-compatibility field. Retain sizes.
    Const scalars/vectors participate in equality by digest, never raw values.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'external' and isinstance(item, list):
                result[key] = {'storage': 'external', 'sizes': item[1:]}
                if constants:
                    # Unknown external constant contents must not compare equal
                    # solely by size. Conservative identity until values read.
                    result[key]['storageIdentitySha256'] = hashlib.sha256(_json(item).encode()).hexdigest()
            elif isinstance(item, list) and (key in PAYLOADS or constants and key.endswith('s')):
                result[key] = {'parameterElements': len(item)}
                if constants:
                    result[key]['valueSha256'] = hashlib.sha256(_json(item).encode()).hexdigest()
            else:
                result[key] = _attrs(item, constants=constants)
        return result
    if isinstance(value, list):
        return [_attrs(item, constants=constants) for item in value]
    return value


class TensorGraph:
    def __init__(self, scope, nodes, outputs=()):
        self.scope = scope
        self.nodes = nodes
        self.producer = {}
        self.consumers = defaultdict(set)
        self.successors = defaultdict(set)
        self.predecessors = defaultdict(set)
        self.outputs = set(outputs)
        for i, node in enumerate(nodes):
            for port, tensor in enumerate(node.get('outputIndexes', [])):
                if tensor < 0 or tensor in self.producer:
                    raise ValueError('Graph needs unique nonnegative tensor producers')
                self.producer[tensor] = (i, port)
        for i, node in enumerate(nodes):
            for tensor in node.get('inputIndexes', []):
                if tensor < 0:
                    continue  # MNN absent optional input, not a real tensor.
                self.consumers[tensor].add(i)
                if tensor in self.producer:
                    source = self.producer[tensor][0]
                    self.successors[source].add(i)
                    self.predecessors[i].add(source)
        indegree = {i: len(self.predecessors[i]) for i in range(len(nodes))}
        ready = deque(i for i in indegree if not indegree[i])
        order = []
        while ready:
            i = ready.popleft()
            order.append(i)
            for target in self.successors[i]:
                indegree[target] -= 1
                if not indegree[target]:
                    ready.append(target)
        if len(order) != len(nodes):
            raise ValueError('Cyclic/control-flow graph is not a supported fusion region')
        self.rank = {n: i for i, n in enumerate(order)}
        self.descendants = {}
        for i in reversed(order):
            reached = set(self.successors[i])
            for j in self.successors[i]:
                reached.update(self.descendants[j])
            self.descendants[i] = reached

    def connected_sets(self, eligible, minimum, maximum):
        frontier = {frozenset([i]) for i in eligible}
        for size in range(2, maximum + 1):
            larger = set()
            for region in frontier:
                larger.update(self.extensions(region, eligible))
            frontier = larger
            if size >= minimum:
                yield from sorted(frontier, key=lambda s: tuple(sorted(s)))

    def extensions(self, region, eligible):
        neighbors = set().union(*(self.successors[i] | self.predecessors[i] for i in region))
        return (region | {node} for node in (neighbors & eligible) - region)

    def convex(self, region):
        # Reject paths that leave the region and later reenter it.
        return not any(self.descendants[j] & region for i in region
                       for j in self.successors[i] - region)

    def contract(self, order, rows):
        region = frozenset(order)
        slots = {node: slot for slot, node in enumerate(order)}
        externals = {}
        encoded = []
        for i in order:
            node, row = self.nodes[i], rows[i]
            ins = []
            for tensor in node.get('inputIndexes', []):
                if tensor < 0:
                    ins.append(['absent'])
                elif tensor in self.producer and self.producer[tensor][0] in region:
                    producer, port = self.producer[tensor]
                    ins.append(['internal', slots[producer], port])
                else:
                    if tensor not in externals:
                        externals[tensor] = len(externals)
                    constant = None
                    if tensor in self.producer:
                        source = self.nodes[self.producer[tensor][0]]
                        if source['type'] == 'Const':
                            constant = _attrs(source.get('main', {}), constants=True)
                    ins.append(['external', externals[tensor], constant])
            outs = []
            for tensor in node.get('outputIndexes', []):
                outs.append({'externalConsumers': len(self.consumers[tensor] - region),
                             'graphOutput': tensor in self.outputs})
            encoded.append({'type': node['type'], 'mainType': node.get('main_type'),
                'attrs': _attrs(node.get('main', {})),
                'declaredLayout': node.get('defaultDimentionFormat', 'schema_default'),
                'inputs': ins, 'outputs': outs, 'inputShapes': row['inputShapes'],
                'outputShapes': row['outputShapes'],
                'backendKeys': sorted(row['backendObservations'])})
        return _json(encoded)

    def canonical(self, region, rows):
        best = None
        best_order = None
        for order in itertools.permutations(sorted(region)):
            value = self.contract(order, rows)
            if best is None or value < best:
                best, best_order = value, order
        return best, best_order


def _graphs(export):
    names = export.get('tensorName', [])
    outputs = {i for i, name in enumerate(names) if name in export.get('outputName', [])}
    yield TensorGraph('main', export.get('oplists', []), outputs)
    for number, graph in enumerate(export.get('subgraphs', [])):
        yield TensorGraph('subgraph:' + str(number), graph.get('nodes', []), graph.get('outputs', []))


def profile_context(report, model_sha256):
    native = report['native']
    if native['status'] != 'completed' or not native['instrumented']:
        raise ValueError('Need completed operator diagnostic')
    if not report.get('controlledAppTimingAccepted', False):
        raise ValueError('Timing controls were not accepted; do not rank by these costs')
    expected = next(f['sha256'] for f in report['modelManifest']['files'] if f['relative'] == 'lm/llm.mnn')
    if model_sha256 != expected:
        raise ValueError('Export source model SHA differs from profiled model')
    forwards = {'prefill': len(native['rounds']),
                'decode': sum(r['decodeForwards'] for r in native['rounds'])}
    context = {'modelSha256': expected, 'mnnArchiveSha256': native.get('mnnArchiveSha256'),
               'profile': report['modelManifest'].get('profile'),
               'precision': native.get('precision'), 'threads': native.get('threads')}
    lookup = defaultdict(list)
    for row in native['operators']['nodes']:
        lookup[row['phase'], row['name'], row['type']].append(row)
    return native, forwards, context, lookup


def matched_profile_rows(graph, phase, lookup):
    if graph.scope != 'main':
        return {}
    # Profiles currently lack subgraph scope. Do not match ambiguous names.
    name_count = Counter(n.get('name', '') for n in graph.nodes)
    rows = {}
    for i, node in enumerate(graph.nodes):
        candidates = lookup.get((phase, node.get('name', ''), node['type']), [])
        if name_count[node.get('name', '')] != 1 or len(candidates) != 1:
            continue
        row = candidates[0]
        if len(row['inputShapes']) != len(node.get('inputIndexes', [])) or len(row['outputShapes']) != len(node.get('outputIndexes', [])):
            continue
        if type(row['calls']) is not int or row['calls'] <= 0 or not math.isfinite(row['seconds']) or row['seconds'] < 0:
            raise ValueError('Invalid measured calls or timing')
        rows[i] = row
    return rows


def mine_connected_subgraphs(export, report, *, model_sha256, minimum=2, maximum=3):
    if not 2 <= minimum <= maximum <= 3:
        raise ValueError('Bounded exact canonicalization supports 2/3-node motifs')
    native, forwards, context, lookup = profile_context(report, model_sha256)
    expected = context['modelSha256']
    groups = {}
    statistics = Counter()
    for graph in _graphs(export):
        statistics['staticNodes'] += len(graph.nodes)
        for phase, count in forwards.items():
            if not count:
                continue
            rows = matched_profile_rows(graph, phase, lookup)
            eligible = {i for i in rows if graph.nodes[i]['type'] in PURE_TYPES}
            statistics['profileMatchedNodes'] += len(rows)
            for region in graph.connected_sets(eligible, minimum, maximum):
                statistics['enumeratedConnectedRegions'] += 1
                if not graph.convex(region):
                    statistics['rejectedNonconvex'] += 1
                    continue
                calls = {rows[i]['calls'] for i in region}
                if len(calls) != 1:
                    statistics['rejectedUnequalExecutionCount'] += 1
                    continue
                key, order = graph.canonical(region, rows)
                digest = hashlib.sha256(_json([context, phase, key]).encode()).hexdigest()
                if digest not in groups:
                    types = [graph.nodes[i]['type'] for i in sorted(region, key=graph.rank.get)]
                    edges = sum(len(graph.successors[i] & region) for i in region)
                    fork = any(len(graph.predecessors[i] & region) > 1 for i in region)
                    split = any(len(graph.successors[i] & region) > 1 for i in region)
                    groups[digest] = {'id': digest, 'phase': phase, 'size': len(region),
                        'topology': 'fork_join' if fork and split else 'join' if fork else 'fork' if split else 'chain',
                        'operatorTypes': types, 'internalEdges': edges,
                        'contract': json.loads(key), 'occurrences': [],
                        'observedInvocations': 0, 'baselineCallbackTotalMs': 0.0,
                        'observedForwards': count}
                group = groups[digest]
                occurrence = {'scope': graph.scope, 'nodes': [graph.nodes[i]['name'] for i in order],
                    'covers': [rows[i]['id'] for i in order], 'calls': next(iter(calls)),
                    'callbackTotalMs': sum(rows[i]['seconds'] * 1000 for i in region)}
                group['occurrences'].append(occurrence)
                group['observedInvocations'] += occurrence['calls']
                group['baselineCallbackTotalMs'] += occurrence['callbackTotalMs']
    repeated = []
    for group in groups.values():
        if len(group['occurrences']) < 2:
            continue
        group['staticOccurrences'] = len(group['occurrences'])
        group['callsPerForward'] = group['observedInvocations'] / group['observedForwards']
        primitive_counts = Counter(p for o in group['occurrences'] for p in o['covers'])
        group['overlappingOccurrences'] = any(n > 1 for n in primitive_counts.values())
        group['uniqueCoveredCallbackMs'] = sum(
            row['seconds'] * 1000 for row in native['operators']['nodes'] if row['id'] in primitive_counts)
        group['tensorDependenciesVerified'] = True
        group['convexRegionVerified'] = True
        group['fusionReady'] = False
        group['speedupProven'] = False
        group['parameterBinding'] = 'Learned payloads omitted from matching; bind each site parameters separately'
        group['unverifiedFields'] = ['runtime dtype/layout/quantization packing ABI',
            'kernel support and workspace', 'alias/in-place and state semantics',
            'generated kernel numerical correctness', 'standalone and whole-graph benefit']
        repeated.append(group)
    return {'schemaVersion': 1, 'modelId': native['modelId'], 'modelSha256': expected,
        'compatibilityContext': context,
        'exportSourceIdentity': 'Caller-declared source SHA checked against profile manifest; retain private export provenance',
        'graphExportSha256': hashlib.sha256(_json(export).encode()).hexdigest(),
        'scope': 'Tensor-connected 2/3-node hints. Callback sums are diagnostic costs, not fused latency or speedup. Overlaps require exact-cover selection.',
        'statistics': dict(statistics), 'patternHints': sorted(repeated,
            key=lambda g: g['uniqueCoveredCallbackMs'], reverse=True)}
