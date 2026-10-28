"""Resumable, revision-pinned public weight download with verified range responses.

Uses official HF HTTPS endpoints, never reads local credentials. Range files are
kept inside ignored models/; final SHA256 must match HF's LFS SHA256 metadata.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import time

ROOT = Path(__file__).resolve().parents[1] / "models/qwen3-0.6b"
REPO = "Qwen/Qwen3-0.6B"
REV = "c1899de289a04d12100db370d81485cdf75e47ca"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    listing = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REV}", timeout=30))
    item = next(x for x in listing if x["path"] == "model.safetensors")
    total = item["size"]
    expected = item["lfs"]["oid"]
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE"):
        dest = ROOT / name
        if not dest.exists():
            dest.write_bytes(urllib.request.urlopen(BASE + name, timeout=60).read())
    parts = ROOT / "parts"; parts.mkdir(exist_ok=True)
    chunk = 8 * 1024 * 1024
    def fetch(start):
        end = min(total, start + chunk) - 1
        dest = parts / str(start)
        if dest.exists() and dest.stat().st_size == end - start + 1:
            return dest.stat().st_size
        for attempt in range(4):
            try:
                req = urllib.request.Request(BASE + f"model.safetensors?part={start}", headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError("Server did not honor exact byte range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError("Truncated response")
                dest.write_bytes(data)
                return len(data)
            except Exception:
                if attempt == 3: raise
                time.sleep(1 + attempt)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for size in pool.map(fetch, range(0, total, chunk)):
            done += size
            print(f"downloaded {done}/{total} bytes", flush=True)
    dest = ROOT / "model.safetensors"
    h = hashlib.sha256()
    with (ROOT / "model.safetensors.partial").open("wb") as out:
        for start in range(0, total, chunk):
            data = (parts / str(start)).read_bytes(); h.update(data); out.write(data)
    if h.hexdigest() != expected:
        raise RuntimeError("Weight SHA256 mismatch")
    os.replace(ROOT / "model.safetensors.partial", dest)
    report = {"repo": REPO, "revision": REV, "bytes": total, "sha256": expected, "sha256_verified": True,
              "text_backbone_only": True}
    (ROOT / "download-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
