"""Download inference-only tensor ranges from the pinned public Mixed codec.

Does not deserialize pickle. Omits MusicFM/Hubert encoders unused for token->audio.
The extracted subset has its own SHA256; full upstream file hash is NOT verified.
"""
import concurrent.futures
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'models/levo-mixed-codec'
REPO = 'lglg666/SongGeneration-Runtime'
FILE = 'ckpt/model_1rvq/model_2_fixed.safetensors'


def get_range(url, start, end, total=None):
    req = urllib.request.Request(url + f'?inference-part={start}', headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=90) as r:
        expected = f'bytes {start}-{end}/'
        content_range = r.headers.get('Content-Range', '')
        if r.status != 206 or not content_range.startswith(expected):
            raise RuntimeError('Server did not honor exact range: ' + content_range)
        if total is not None and content_range != expected + str(total):
            raise RuntimeError('Source size changed')
        data = r.read()
    if len(data) != end - start + 1:
        raise RuntimeError('Truncated byte range')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--deadline-epoch', type=float,
                        help='Hard stop at the authorized window boundary; parts remain resumable')
    args = parser.parse_args()
    if args.deadline_epoch is not None:
        remaining = args.deadline_epoch - time.time()
        if remaining <= 0:
            raise SystemExit('Authorized window already ended; no download started')
        def stop_at_deadline():
            print('Authorized window ended; stopping this downloader without a new-window retry', flush=True)
            os._exit(75)
        timer = threading.Timer(remaining, stop_at_deadline)
        timer.daemon = True
        timer.start()
    ROOT.mkdir(parents=True, exist_ok=True)
    pin = ROOT / 'source.json'
    if pin.exists():
        source = json.loads(pin.read_text())
    else:
        info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}', timeout=30))
        rev = info['sha']
        listing = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/tree/{rev}/ckpt/model_1rvq', timeout=30))
        item = next(x for x in listing if x['path'] == FILE)
        source = {'repo': REPO, 'revision': rev, 'file': FILE, 'bytes': item['size'], 'lfs_sha256': item['lfs']['oid']}
        pin.write_text(json.dumps(source, indent=2), encoding='utf8')
    url = f'https://huggingface.co/{REPO}/resolve/{source["revision"]}/{FILE}'
    header_path = ROOT / 'source-header.json'
    if header_path.exists():
        cached = json.loads(header_path.read_text())
        header_len, header = cached['header_bytes'], cached['tensors']
    else:
        header_len = struct.unpack('<Q', get_range(url, 0, 7, source['bytes']))[0]
        if header_len > 4_000_000: raise RuntimeError('Unexpected header size')
        header = json.loads(get_range(url, 8, 7 + header_len, source['bytes']))
        header_path.write_text(json.dumps({'header_bytes': header_len, 'tensors': header}), encoding='utf8')
    prefix = ('cfm_wrapper.', 'mask_emb.', 'normfeat.', 'rvq_bestrq_emb.', 'zero_cond_embedding1')
    selected = {k: v for k, v in header.items() if k.startswith(prefix)}
    # The two inference intervals include only tiny intervening resample buffers.
    cfm = [v['data_offsets'] for k, v in selected.items() if k.startswith('cfm_wrapper.')]
    rest = [v['data_offsets'] for k, v in selected.items() if not k.startswith('cfm_wrapper.')]
    intervals = [(min(x[0] for x in a), max(x[1] for x in a)) for a in (cfm, rest)]
    tasks = []
    chunk = 8 * 1024 * 1024
    base = header_len + 8
    for lo, hi in intervals:
        for start in range(lo, hi, chunk):
            tasks.append((base + start, base + min(start + chunk, hi) - 1))
    parts = ROOT / 'parts'; parts.mkdir(exist_ok=True)
    def fetch(bounds):
        start, end = bounds; dest = parts / str(start)
        if dest.exists() and dest.stat().st_size == end - start + 1: return len(dest.read_bytes())
        for attempt in range(4):
            try:
                data = get_range(url, start, end, source['bytes'])
                temp = dest.with_suffix('.partial'); temp.write_bytes(data); os.replace(temp, dest)
                return len(data)
            except Exception as e:
                print(f'retry range={start} attempt={attempt+1}: {type(e).__name__}', flush=True)
                if attempt == 3: raise
                time.sleep(2 + attempt)
    done = 0; total = sum(b - a + 1 for a, b in tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, x) for x in tasks]
        for f in concurrent.futures.as_completed(futures):
            done += f.result(); print(f'inference weights {done}/{total} bytes', flush=True)
    ordered = sorted(selected.items(), key=lambda x: x[1]['data_offsets'][0])
    out_header = {'__metadata__': {'source_repo': REPO, 'source_revision': source['revision'], 'subset': 'inference-only'}}
    offset = 0
    for k, v in ordered:
        length = v['data_offsets'][1] - v['data_offsets'][0]
        out_header[k] = {'dtype': v['dtype'], 'shape': v['shape'], 'data_offsets': [offset, offset + length]}
        offset += length
    encoded = json.dumps(out_header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    h = hashlib.sha256()
    dest = ROOT / 'mixed-inference.safetensors'
    with dest.with_suffix('.partial').open('wb') as out:
        def write(data): out.write(data); h.update(data)
        write(struct.pack('<Q', len(encoded))); write(encoded)
        for _, v in ordered:
            lo, hi = v['data_offsets']; pos = base + lo; end = base + hi
            while pos < end:
                a, b = next((a, b) for a, b in tasks if a <= pos <= b)
                with (parts / str(a)).open('rb') as part:
                    part.seek(pos - a); data = part.read(min(end - pos, b + 1 - pos))
                write(data); pos += len(data)
    os.replace(dest.with_suffix('.partial'), dest)
    report = {**source, 'subset_tensor_count': len(selected), 'subset_sha256': h.hexdigest(),
              'full_source_sha256_verified': False, 'scope': 'Exact ranged HTTP responses from pinned source; inference subset'}
    (ROOT / 'download-manifest.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
