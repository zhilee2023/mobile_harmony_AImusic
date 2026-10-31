"""Inspect pinned public LeVo configs and tensor metadata, never execute pickle.

The .pt metadata reader accepts only symbolic tensor reconstruction functions.
It performs bounded HTTP ranges; no storage payload is deserialized or allocated.
"""
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models/public-levo-v2-large'


def request_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


class RangeFile(io.RawIOBase):
    def __init__(self, url, length):
        self.url, self.length, self.position = url, length, 0
        self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, position, whence=0):
        self.position = position if whence == 0 else self.position + position if whence == 1 else self.length + position
        if not 0 <= self.position <= self.length:
            raise ValueError('Out of checkpoint range')
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        if not size:
            return b''
        if size > 4 * 1024**2 or self.transferred + size > 20 * 1024**2:
            raise ValueError('Metadata inspection must not download tensor storages')
        start, end = self.position, self.position + size - 1
        req = urllib.request.Request(self.url + f'?metadata-range={start}-{end}', headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=90) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.length}':
                raise ValueError('Exact source byte range not honored')
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError('Truncated metadata range')
        self.position += size
        self.transferred += size
        return data


class StorageSpec:
    def __init__(self, dtype):
        self.dtype = dtype


def tensor_spec(storage, offset, size, stride, *unused):
    shape = list(size)
    if len(shape) > 10 or any(not isinstance(d, int) or d < 0 for d in shape):
        raise ValueError('Invalid tensor metadata')
    return {'shape': shape, 'stride': list(stride), 'storageOffset': offset, **storage}


class MetadataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
            return tensor_spec
        types = {'FloatStorage': 'F32', 'HalfStorage': 'F16', 'BFloat16Storage': 'BF16',
                 'LongStorage': 'I64', 'IntStorage': 'I32', 'BoolStorage': 'BOOL'}
        if module == 'torch' and name in types:
            return StorageSpec(types[name])
        raise pickle.UnpicklingError(f'Rejected executable or unsupported global: {module}.{name}')

    def persistent_load(self, value):
        if not isinstance(value, tuple) or len(value) < 5 or value[0] != 'storage' or not isinstance(value[1], StorageSpec):
            raise pickle.UnpicklingError('Unexpected storage descriptor')
        return {'dtype': value[1].dtype, 'storageKey': str(value[2]), 'storageElements': value[4]}


def flatten(value, prefix=''):
    if isinstance(value, dict) and 'shape' in value and 'storageKey' in value:
        return {prefix: value}
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                result.update(flatten(item, prefix + '.' + key if prefix else key))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    pin = OUT / 'source.json'
    if pin.exists():
        source = json.loads(pin.read_text(encoding='utf8'))
    else:
        repo = 'lglg666/SongGeneration-v2-large'
        info = request_json('https://huggingface.co/api/models/' + repo)
        revision = info['sha']
        files = request_json(f'https://huggingface.co/api/models/{repo}/tree/{revision}')
        weight = next(x for x in files if x['path'] == 'model.pt')
        source = {'repo': repo, 'revision': revision, 'file': 'model.pt', 'bytes': weight['size'], 'sha256': weight['lfs']['oid']}
        pin.write_text(json.dumps(source, indent=2), encoding='utf8')
    base = f'https://huggingface.co/{source["repo"]}/resolve/{source["revision"]}/'
    config_path = OUT / 'config.yaml'
    if not config_path.exists():
        with urllib.request.urlopen(base + 'config.yaml', timeout=45) as response:
            config_path.write_bytes(response.read(100_000))
    metadata_path = OUT / 'tensor-metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf8'))
    else:
        file = RangeFile(base + source['file'], source['bytes'])
        with zipfile.ZipFile(file) as archive:
            entry = next(x for x in archive.infolist() if x.filename.endswith('/data.pkl'))
            if entry.file_size > 4 * 1024**2:
                raise ValueError('Oversized metadata pickle')
            payload = archive.read(entry)
        tensors = flatten(MetadataOnlyUnpickler(io.BytesIO(payload)).load())
        if not tensors:
            raise ValueError('No tensor descriptors found')
        metadata = {'source': source, 'metadataOnly': True, 'storagePayloadDownloaded': False,
                    'metadataPickleSha256': hashlib.sha256(payload).hexdigest(), 'httpBytesRead': file.transferred,
                    'tensors': tensors}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf8')
    groups = collections.defaultdict(lambda: {'tensorCount': 0, 'parameters': 0, 'storageDtypes': set()})
    examples = {}
    for name, tensor in metadata['tensors'].items():
        if '.transformer2.' in name:
            group = 'trackSpecificLm'
        elif '.transformer.' in name:
            group = 'mixedSemanticLm'
        elif 'condition' in name:
            group = 'conditioners'
        else:
            group = 'embeddingsHeadsAndOther'
        groups[group]['tensorCount'] += 1
        groups[group]['parameters'] += math.prod(tensor['shape'])
        groups[group]['storageDtypes'].add(tensor['dtype'])
        examples.setdefault(group, {})
        if len(examples[group]) < 4:
            examples[group][name] = tensor['shape']
    for item in groups.values():
        item['storageDtypes'] = sorted(item['storageDtypes'])
    report = {'source': source, 'metadataOnly': True, 'trainedWeightPayloadLoaded': False,
              'httpBytesRead': metadata['httpBytesRead'], 'tensorCount': len(metadata['tensors']),
              'parameterEntries': sum(x['parameters'] for x in groups.values()), 'groups': dict(groups),
              'examples': examples, 'generationPerformed': False, 'wavGenerated': False,
              'note': 'Tensor entries may include tied/shared storage; group counts need semantic state-dict mapping before claiming unique trainable parameters.'}
    (ROOT / 'reports/public-levo-v2-audit.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
