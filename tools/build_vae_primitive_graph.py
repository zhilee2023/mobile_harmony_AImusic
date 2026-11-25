"""Join the recorded VAE operator DAG with evidence-scoped local primitive DAGs."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.price_vae_fusions import full_copy_sites, read, runtime_graph
from levo_harmony.operator_lab.subgraphs import _graphs


DECONV_STAGES = ('Clear', 'PackA', 'GEMM', 'Col2Im', 'Reduce', 'BiasClamp')
CONV_STAGES = ('Quant', 'PackA', 'GemmEpilogue', 'InputSum', 'OutputQuant', 'WeightReorder')


def parse_deconvolution(text, count):
    lines = [line.split('LEVO_VAE_DECONV ', 1)[1] for line in text.splitlines()
             if 'LEVO_VAE_DECONV ' in line]
    if count <= 0 or not lines or len(lines) % count:
        raise ValueError('Expected complete, equally sized deconvolution tracing rounds')
    result = []
    for line in lines[-count:]:
        (ic, oc, in_elements, out_elements, workers, element_bytes, pack_width, tile,
         kernel, stride, dilation, out_bytes, pack_bytes, gemm_bytes, first_ns, second_ns,
         *worker_times) = map(int, line.split())
        if len(worker_times) != workers * len(DECONV_STAGES) or not workers:
            raise ValueError('Malformed deconvolution worker profile')
        spans = list(zip(*(iter(worker_times),) * len(DECONV_STAGES), strict=True))
        result.append(dict(inputChannels=ic, outputChannels=oc, inputElements=in_elements,
                           outputElements=out_elements, workers=workers, elementBytes=element_bytes,
                           packWidth=pack_width, inputTile=tile, kernel=kernel, stride=stride,
                           dilation=dilation, outputBytes=out_bytes, packBytesPerWorker=pack_bytes,
                           gemmBytesPerWorker=gemm_bytes,
                           scratchBytes=(workers-1)*out_bytes+workers*(pack_bytes+gemm_bytes),
                           privateScatterBytes=(workers-1)*out_bytes,
                           stageWallMs=[first_ns / 1e6, second_ns / 1e6],
                           workerMs=[dict(zip(DECONV_STAGES, (n/1e6 for n in row), strict=True))
                                     for row in spans]))
    return result


def event(name, kind, **attrs):
    return dict(name=name, kind=kind, **attrs)


def parse_convolution(text, count):
    lines = [line.split('LEVO_VAE_DENSE_CONV ', 1)[1] for line in text.splitlines()
             if 'LEVO_VAE_DENSE_CONV ' in line]
    if not lines:
        return [None] * count
    if count <= 0 or len(lines) % count:
        raise ValueError('Expected complete quantized convolution tracing rounds')
    result = []
    for line in lines[-count:]:
        (plane, ic, oc, bits, quant, option, split, quant_first, e, l, h,
         input_bytes, output_bytes, kernel, stride, wall, *spans) = map(int, line.split())
        if len(spans) != len(CONV_STAGES)*3:
            raise ValueError('Malformed quantized convolution profile')
        timings = {kind: dict(calls=calls, serialMs=serial/1e6, slowestWorkerMs=worker/1e6)
                   for kind, (calls, serial, worker) in zip(CONV_STAGES, zip(*(iter(spans),)*3, strict=True), strict=True)}
        result.append(dict(positions=plane, inputChannels=ic, outputChannels=oc, weightBits=bits,
                           dynamicQuant=bool(quant), quantOption=option, splitByOutputChannel=bool(split),
                           quantBeforePacking=bool(quant_first), tile=[e, l, h],
                           inputElementBytes=input_bytes, outputElementBytes=output_bytes,
                           kernel=kernel, stride=stride, wallMs=wall/1e6, stages=timings))
    return result


def describe(name, operator, tensors, trace):
    inputs = [tensors[i] for i in operator['inputIndexes']]
    outputs = [tensors[i] for i in operator['outputIndexes']]
    in_bytes = sum(t['storageBytes'] for t in inputs)
    out_bytes = sum(t['storageBytes'] for t in outputs)
    kind = operator['type']
    status = 'executor-source-only'
    if kind == 'Raster':
        nodes = [event('Source', 'Read'), event('Raster', 'Layout', regions=operator['main']['regions']),
                 event('Materialize', 'Write', outputAllocationBytes=out_bytes)]
        status = 'observed-runtime-mapping'
    elif kind == 'SnakeFP16':
        nodes = [event('Activation', 'Read', logicalBytes=in_bytes),
                 event('Vector Snake', 'Compute', elements=math.prod(outputs[0]['shape'])),
                 event('Output', 'Write', logicalBytes=out_bytes)]
        status = 'deployed-kernel-source-and-callback'
    elif kind == 'Convolution' and trace:
        nodes = [event('Activation', 'Read', storageBytes=in_bytes)]
        stage_order = [('PackA', 'Pack A / Im2Col', 'Packing'), ('Quant', 'Dynamic A8', 'Compute')]
        if trace['quantBeforePacking']:
            stage_order.reverse()
        stage_order += [('InputSum', 'Input sum and asymmetric correction preparation', 'Compute'),
                        ('GemmEpilogue', 'GEMM and epilogue', 'Compute'),
                        ('OutputQuant', 'Output quantization', 'Compute')]
        nodes.extend(event(label, node_kind, timing=trace['stages'][stage])
                     for stage, label, node_kind in stage_order if trace['stages'][stage]['calls'])
        nodes.append(event('Output', 'Write', storageBytes=out_bytes))
        if trace['stages']['WeightReorder']['calls']:
            nodes.append(event('Online weight reorder', 'Packing', timing=trace['stages']['WeightReorder'],
                               phaseOrder='executor branch dependent'))
        status = 'observed-DenseConvInt8TiledExecutor'
    elif kind == 'Convolution':
        config = operator['main'].get('common', {})
        return dict(op=name, inputStorageBytes=in_bytes, outputStorageBytes=out_bytes,
                    weightQuantMetadata=bool(operator['main'].get('quanParameter')),
                    kernel=[config.get('kernelX'), config.get('kernelY')],
                    dilation=[config.get('dilateX'), config.get('dilateY')],
                    unresolved=['executor dispatch', 'Pack A / Im2Col', 'activation quantization',
                                'GEMM epilogue', 'weight storage and working set'])
    elif kind == 'Deconvolution' and trace:
        count = trace['workers']
        nodes = [event('Clear', 'Write', logicalBytes=trace['outputBytes']*count),
                 event('PackA', 'Packing', scratchBytes=trace['packBytesPerWorker']*count),
                 event('GEMM', 'Compute', tileOutputBytes=trace['gemmBytesPerWorker']*count),
                 event('Col2Im', 'Compute', privateScatterBytes=trace['privateScatterBytes']),
                 event('Parallel scatter complete', 'Synchronization'),
                 event('Reduce', 'Compute', logicalReadBytes=trace['privateScatterBytes']*2,
                       logicalWriteBytes=trace['privateScatterBytes']),
                 event('BiasClamp', 'Compute', logicalReadBytes=trace['outputBytes'],
                       logicalWriteBytes=trace['outputBytes'])]
        status = 'observed-CPUDeconvolutionOrigin'
    elif kind == 'Deconvolution':
        return dict(op=name, inputStorageBytes=in_bytes, outputStorageBytes=out_bytes,
                    unresolved=['executor dispatch', 'per-worker phases and dynamic scratch'])
    else:
        nodes = [event('Inputs', 'Read', inputStorageBytes=in_bytes),
                 event(kind, 'Compute'), event('Output', 'Write', outputStorageBytes=out_bytes)]
    edges = [[i, i+1] for i in range(len(nodes)-1)]
    if kind == 'Convolution' and trace and trace['stages']['WeightReorder']['calls']:
        gemm = next(i for i, node in enumerate(nodes) if node['name'] == 'GEMM and epilogue')
        edges[-1] = [len(nodes)-1, gemm]
    if kind == 'Deconvolution' and trace:
        # These are the phase dependencies of a repeated per-worker tile
        # template. Pack reads only the input; it does not depend on Clear.
        edges = [[0, 3], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    return dict(op=name, status=status, primitives=nodes, dependencies=edges)


def fuse_graph(profile, source, result, executor_log=None):
    snake = {g['name'] for g in result['rewrite']['groups']}
    runtime, _ = runtime_graph(profile, source, snake)
    outputs = profile['graphOutputs']
    tensors = [dict(t, tensor=i, producer=runtime.producer.get(i),
                    consumers=sorted(runtime.consumers[i]),
                    firstObserved=runtime.producer.get(i, (None, 0))[0],
                    lastConsumer=max(runtime.consumers[i], default=None),
                    isGraphOutput=i in outputs)
               for i, t in enumerate(profile['tensors'])]
    expected = sum(n['type'] == 'Deconvolution' for n in runtime.nodes)
    conv_count = sum(n['type'] == 'Convolution' for n in runtime.nodes)
    traced = iter(parse_deconvolution(executor_log, expected) if executor_log else [None]*expected)
    convs = iter(parse_convolution(executor_log or '', conv_count))
    regions = []
    for i, node in enumerate(runtime.nodes):
        trace = None
        if node['type'] == 'Deconvolution':
            trace = next(traced)
        elif node['type'] == 'Convolution':
            trace = next(convs)
        if trace:
            xi, = node['inputIndexes']
            yi, = node['outputIndexes']
            x, y = tensors[xi]['shape'], tensors[yi]['shape']
            if (x[1], y[1]) != (trace['inputChannels'], trace['outputChannels']):
                raise ValueError('Executor trace did not match runtime command channel counts or order')
            if node['type'] == 'Convolution' and math.prod(y) != trace['positions'] * y[1]:
                raise ValueError('Convolution executor trace has different spatial dimensions')
            if node['type'] == 'Deconvolution' and (math.prod(x), math.prod(y)) != (trace['inputElements'], trace['outputElements']):
                raise ValueError('Deconvolution executor trace has different spatial dimensions')
        primitive = describe(node['name'], node, tensors, trace)
        if trace:
            primitive['executorMeasurement'] = trace
        row = dict(id=i, type=node['type'], inputs=node['inputIndexes'], outputs=node['outputIndexes'],
                   callsPerWindow=1, callbackMs=profile['commands'][i]['seconds']*1000, local=primitive)
        if node['type'] in {'Convolution', 'Deconvolution'}:
            common = node['main'].get('common', {})
            row['operatorParameters'] = {name: common.get(name) for name in
                ('kernelX', 'kernelY', 'strideX', 'strideY', 'dilateX', 'dilateY', 'group')}
            row['operatorParameters']['weightQuantMetadata'] = bool(node['main'].get('quanParameter'))
        regions.append(row)
    action_sites = [index for index, _, _, _ in full_copy_sites(runtime, profile)]
    return dict(windowSeconds=10.24, graphOutputs=outputs, operators=regions, tensors=tensors,
                copyElisionSites=action_sites,
                evidence='A device callback graph and traced executor phases when present. Storage sizes and logical access counts are not physical cache/DRAM traffic.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for p in ('profile', 'result', 'source_graph', 'output'):
        parser.add_argument(p, type=Path)
    parser.add_argument('--executor-log', type=Path)
    args = parser.parse_args()
    profile, result = read(args.profile), read(args.result)
    if result['mode'] != 'fused' or not result['success'] or not profile['pcmMatchesWarmRun']:
        parser.error('Only a successful profile of the deployed Snake graph is supported')
    source = next(_graphs(read(args.source_graph)))
    log = args.executor_log.read_text(encoding='utf-8') if args.executor_log else None
    graph = fuse_graph(profile, source, result, log)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(graph, out, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
