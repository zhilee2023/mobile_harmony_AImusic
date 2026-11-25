"""Rank new fusion sites in the executing, already-fused VAE graph."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from levo_harmony.operator_lab.subgraph_pricing import PhaseSearch, Site, solve_packing
from levo_harmony.operator_lab.subgraphs import TensorGraph, _graphs


SUPPORTED = {'Convolution', 'Deconvolution', 'Raster', 'SnakeFP16', 'BinaryOp', 'UnaryOp'}


def read(path):
    return json.loads(path.read_bytes())


def layout(tensor):
    return [tensor['shape'], tensor['format'], tensor['storageBytes'], tensor['backend']]


def runtime_graph(profile, source, snake_names):
    tensors = profile['tensors']
    source_nodes = {node['name']: node for node in source.nodes}
    rows = {}
    nodes = []
    for index, command in enumerate(profile['commands']):
        kind = command['type']
        name = command['name']
        inputs = list(command['inputIndexes'])
        outputs = command['outputIndexes']
        attributes = {}
        eligible = True
        if kind == 'Extra' and name in snake_names:
            kind = 'SnakeFP16'
        elif kind == 'Raster':
            # Raster is a generated runtime command. Keep both explicit inputs
            # and the observed region origins, including its layout/stride contract.
            regions = command['rasterRegions']
            origins = list(dict.fromkeys(region['source'] for region in regions))
            inputs.extend(origin for origin in origins if origin not in inputs)
            attributes = {'regions': [dict(region, source=inputs.index(region['source']))
                                      for region in regions]}
            eligible = bool(regions)
        else:
            schema = source_nodes.get(name)
            eligible = schema is not None and schema['type'] == kind
            if eligible:
                attributes = schema.get('main') or {}
        nodes.append(dict(name=name, type=kind, main=attributes,
                          inputIndexes=inputs, outputIndexes=outputs))
        if eligible and kind in SUPPORTED:
            rows[index] = dict(seconds=command['seconds'], calls=1,
                               inputShapes=[layout(tensors[i]) for i in inputs],
                               outputShapes=[layout(tensors[i]) for i in outputs],
                               logicalOutputBytes=sum(tensors[i]['storageBytes'] for i in outputs),
                               backendObservations={'CPU callback': 1})
    return TensorGraph('VAE runtime', nodes, profile['graphOutputs']), rows


def rank(graph, rows, bandwidth, max_nodes, beam_width):
    search = PhaseSearch(graph, rows, bandwidth, {}, eligible_types=SUPPORTED)
    methods = {'exhaustive': search.exhaustive(max_nodes),
               'beam-gain': search.beam(beam_width, max_nodes, 'current'),
               'beam-bound': search.beam(beam_width, max_nodes, 'bound')}
    result = {}
    exhaustive_order = set()
    for method, (groups, levels) in methods.items():
        candidates = []
        seen = set()
        for group in groups:
            regions = tuple(sorted({tuple(sorted(region)) for region in group}))
            if regions in seen:
                continue
            seen.add(regions)
            sites = [Site(str(i), frozenset(region), search.score(frozenset(region))[1])
                     for i, region in enumerate(regions)]
            if not any(site.gain_ms > 1e-9 for site in sites):
                continue
            best, _ = solve_packing(graph, sites)
            if not best['optimal']:
                raise RuntimeError('Placement solver did not finish optimally')
            selected = [sites[int(i)].region for i in best['selected']]
            sample = sorted(selected[0])
            candidates.append({
                'positions': [list(region) for region in regions],
                'kinds': [graph.nodes[i]['type'] for i in sample],
                'exampleOperators': [graph.nodes[i]['name'] for i in sample],
                'exampleOutputs': [rows[i]['outputShapes'] for i in sample],
                'matchingSites': len(regions),
                'selectedSites': len(selected),
                'optimisticProxyMsPerWindow': best['gainMs'],
                'potentiallyElidedBytesPerWindow': sum(search.score(region)[2] for region in selected),
                'baselineCallbackMsPerWindow': 1000 * math.fsum(rows[i]['seconds']
                                                               for region in selected for i in region),
            })
        candidates.sort(key=lambda row: (-round(row['optimisticProxyMsPerWindow'], 9), row['positions']))
        top = candidates[:10]
        top_keys = {tuple(map(tuple, row['positions'])) for row in top}
        if method == 'exhaustive':
            exhaustive_order = top_keys
        result[method] = dict(levels=levels, positiveTemplates=len(candidates), top10=top,
                              sameTop10AsExhaustive=len(top_keys & exhaustive_order))
    return result


def full_copy_boundaries(graph, profile):
    """Price current copy callbacks alone, without claiming storage can alias."""
    tensors = profile['tensors']
    boundaries = {}
    for node, command in zip(graph.nodes, profile['commands'], strict=True):
        if node['type'] != 'Raster' or len(command['rasterRegions']) != 1 or len(node['outputIndexes']) != 1:
            continue
        region = command['rasterRegions'][0]
        dst, = node['outputIndexes']
        src = region['source']
        if src not in graph.producer or dst in graph.outputs or not graph.consumers[dst]:
            continue
        source, target = tensors[src], tensors[dst]
        if (region['sourceOffset'] != 0 or region['outputOffset'] != 0
                or region['sourceStride'] != region['outputStride'] or region['sourceStride'][-1] != 1
                or region['size'] != [1, 1, math.prod(target['shape'])]
                or math.prod(source['shape']) != math.prod(target['shape'])
                or layout(source)[1:] != layout(target)[1:] or not target['storageBytes']):
            continue
        producer = graph.nodes[graph.producer[src][0]]['type']
        consumers = tuple(sorted(graph.nodes[i]['type'] for i in graph.consumers[dst]))
        entry = boundaries.setdefault((producer, consumers), dict(
            producer=producer, consumers=consumers, copies=0, callbackMs=0.0, outputStorageBytes=0))
        entry['copies'] += 1
        entry['callbackMs'] += command['seconds'] * 1000
        entry['outputStorageBytes'] += target['storageBytes']
    return sorted(boundaries.values(), key=lambda item: -item['callbackMs'])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('profile', type=Path, help='profile.json emitted by levo_stream_vae_fusion --profile')
    cli.add_argument('result', type=Path, help='same run result.json')
    cli.add_argument('source_graph', type=Path, help='original MNN MiniReflect JSON with operator attributes')
    cli.add_argument('output', type=Path)
    cli.add_argument('--bandwidth-gbps', type=float, required=True)
    cli.add_argument('--max-nodes', type=int, choices=range(2, 7), default=6)
    cli.add_argument('--beam-width', type=int, default=32)
    args = cli.parse_args()
    if not math.isfinite(args.bandwidth_gbps) or args.bandwidth_gbps <= 0 or args.beam_width <= 0:
        cli.error('Bandwidth and beam width must be positive')
    result, profile = read(args.result), read(args.profile)
    if not result['success'] or not profile['pcmMatchesWarmRun']:
        cli.error('The input must be a successful, output-verified device profile')
    snake_names = {group['name'] for group in result['rewrite']['groups']} if result['mode'] == 'fused' else set()
    source = next(_graphs(read(args.source_graph)))
    graph, rows = runtime_graph(profile, source, snake_names)
    if snake_names != {node['name'] for node in graph.nodes if node['type'] == 'SnakeFP16'}:
        cli.error('The profile missed an expected deployed Snake kernel')
    hotspots = defaultdict(float)
    for node, command in zip(graph.nodes, profile['commands'], strict=True):
        hotspots[node['type']] += command['seconds'] * 1000
    data = dict(mode=result['mode'], referenceBandwidthGBps=args.bandwidth_gbps,
                inspectedCommands=len(graph.nodes), knownTensorEdges=sum(len(x) for x in graph.successors.values()),
                eligibleCommands=len(rows), deployedSnakeSites=len(snake_names),
                profiledForwardMs=profile['profiledForwardSeconds'] * 1000,
                callbackMsByType=dict(sorted(hotspots.items(), key=lambda pair: -pair[1])),
                sameLayoutFullCopyBoundaries=full_copy_boundaries(graph, profile),
                methods=rank(graph, rows, args.bandwidth_gbps, args.max_nodes, args.beam_width))
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(data, output, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
