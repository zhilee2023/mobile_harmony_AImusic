"""Download public Qwen sources for the diagnostic streaming lab; never overwrite old models."""
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    from huggingface_hub import snapshot_download
    target=ROOT/'models/stream-lab/sources/qwen3-4b'
    target.mkdir(parents=True,exist_ok=True)
    info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen3-4B',timeout=30))
    (target/'source.json').write_text(json.dumps({'repository':'Qwen/Qwen3-4B','revision':info['sha'],'musicCheckpoint':False},indent=2))
    snapshot_download('Qwen/Qwen3-4B',revision=info['sha'],local_dir=str(target),
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','LICENSE'],max_workers=2)
    print('PUBLIC_QWEN4B_DOWNLOAD_COMPLETE',flush=True)

if __name__=='__main__':main()
