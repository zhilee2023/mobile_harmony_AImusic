"""Bundle paths are relative, hash-verified and independent of this workstation."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Bundle paths must use relative POSIX paths')
    path = PurePosixPath(relative)
    if path.is_absolute() or not relative or relative != path.as_posix() or any(x in ('..', '.') for x in relative.split('/')):
        raise ValueError(f'Invalid bundle path: {relative}')
    root = Path(root).resolve(); target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Bundle path escapes root: {relative}')
    return target


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def inventory(root, relative):
    path = inside(root, relative)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f'Missing or empty model asset: {relative}')
    return {'name': Path(relative).name, 'relative': relative, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
