"""Replay observed tensor lifetimes with candidate Raster storage aliases."""
from collections import defaultdict


def storage_peak(graph, alias_names):
    ops = graph['operators']
    tensors = {tensor['tensor']: tensor for tensor in graph['tensors']}
    aliases = {site['name']: site for site in graph['copyAliases']}
    parents = {index: index for index in tensors}

    def root(index):
        while index != parents[index]:
            index = parents[index]
        return index

    names = set(alias_names)
    if unknown := names - aliases.keys():
        raise ValueError(f'Unknown storage aliases: {sorted(unknown)}')
    for name in sorted(names):
        edge = aliases[name]
        source = root(edge['sourceTensor'])
        destination = root(edge['destinationTensor'])
        if tensors[source]['storageBytes'] != tensors[destination]['storageBytes']:
            raise ValueError(f'Incompatible alias storage for {name}')
        parents[destination] = source

    groups = defaultdict(list)
    for index, tensor in tensors.items():
        groups[root(index)].append(tensor)
    events = [0] * (len(ops) + 1)
    for group in groups.values():
        births = [tensor['firstObserved'] for tensor in group if tensor['firstObserved'] is not None]
        first = min(births) if len(births) == len(group) else 0
        last = max([first, *(op for tensor in group for op in tensor['consumers'])])
        if any(tensor['isGraphOutput'] for tensor in group):
            last = len(ops) - 1
        size = group[0]['storageBytes']
        events[first] += size
        events[last + 1] -= size

    live = 0
    peak = None
    timeline = []
    for op in ops:
        live += events[op['id']]
        measurement = op['local'].get('executorMeasurement', {})
        scratch = measurement.get('scratchBytes', 0)
        row = dict(command=op['id'], name=op['local']['op'], liveTensorBytes=live,
                   tracedScratchBytes=scratch, modeledBytes=live + scratch)
        timeline.append(row)
        if peak is None or row['modeledBytes'] > peak['modeledBytes']:
            peak = row
    return dict(peak=peak, timeline=timeline, storageGroups=len(groups), aliases=len(names))
