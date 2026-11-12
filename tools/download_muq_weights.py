"""Resumable ranged download of the pinned official MuQ-MuLan checkpoint."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests

REPO = 'OpenMuQ/MuQ-MuLan-large'
REVISION = '2e01c796b71dca71b45251384c04cd7b237c9020'
SIZE = 2653954401
SHA256 = 'd42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444'

def download(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'pytorch_model.bin'
    parts = folder / '.ranges'
    parts.mkdir(exist_ok=True)
    chunk = 16 * 1024 * 1024
    def fetch(index):
        start, end = index * chunk, min(SIZE, (index + 1) * chunk) - 1
        path = parts / str(index)
        if path.is_file() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(4):
            try:
                url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/pytorch_model.bin?download=true&range={index}'
                with requests.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('content-range') != f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Server did not honor the requested byte range')
                    pending = path.with_suffix('.pending')
                    with pending.open('wb') as stream:
                        for block in response.iter_content(64*1024):
                            stream.write(block)
                    if pending.stat().st_size != end-start+1:
                        raise ValueError('Incomplete range')
                    pending.replace(path)
                return path
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(1+attempt)
    if not output.is_file():
        count = (SIZE + chunk - 1) // chunk
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            jobs = [pool.submit(fetch, i) for i in range(count)]
            for done, job in enumerate(concurrent.futures.as_completed(jobs), 1):
                job.result()
                if done % 2 == 0 or done == count:
                    print(f'Checkpoint ranges {done}/{count}', flush=True)
        pending = output.with_suffix('.pending')
        with pending.open('wb') as stream:
            for i in range(count):
                with (parts / str(i)).open('rb') as part:
                    while block := part.read(1024*1024):
                        stream.write(block)
        pending.replace(output)
    digest = hashlib.file_digest(output.open('rb'), 'sha256').hexdigest()
    if output.stat().st_size != SIZE or digest != SHA256:
        raise ValueError('Checkpoint SHA256 mismatch; preserve file for investigation')
    response = requests.get(f'https://huggingface.co/{REPO}/raw/{REVISION}/config.json', timeout=30)
    response.raise_for_status()
    (folder / 'config.json').write_text(json.dumps(response.json(), indent=2), encoding='utf-8')
    print(f'Official checkpoint SHA256 verified: {digest}', flush=True)

if __name__ == '__main__':
    download(Path(__file__).resolve().parents[1] / 'models/muq-mulan-official')
