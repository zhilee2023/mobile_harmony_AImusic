"""Merge only a completed full-weight CPU codec validation, retaining scope."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    result=json.loads((ROOT/'reports/full-real-codec-cpu.json').read_text(encoding='utf8'))
    manifest=json.loads((ROOT/'models/levo-mixed-codec/download-manifest.json').read_text())
    required=('full_pretrained_cfm_loaded','subset_sha256_verified','full_pretrained_vae_loaded')
    if not all(result[key] for key in required):raise ValueError('Full codec is not verified')
    if result['subset_sha256']!=manifest['subset_sha256']:raise ValueError('Mixed codec identity mismatch')
    if len(result['chunks'])!=2 or result['output_shape']!=[2,448*1920] or result['steps_per_chunk']!=20:
        raise ValueError('Expected the complete two-window20-step validation')
    if not all(row['pcm_finite'] for row in result['chunks']):raise ValueError('Invalid PCM chunk')
    if result['real_song_generated'] or result['student_lm_connected']:raise ValueError('Numerical fixture scope changed')
    result['as_of_utc']=datetime.now(timezone.utc).isoformat()
    result['source_code_sha256']=hashlib.sha256((ROOT/'tools/validate_real_codec.py').read_bytes()).hexdigest()
    result['download_manifest']=manifest
    result['full_inference_subset_bytes']=(ROOT/'models/levo-mixed-codec/mixed-inference.safetensors').stat().st_size
    path=ROOT/'reports/levo-test-report.json';report=json.loads(path.read_text(encoding='utf8'))
    report['full_codec_cpu_2026_10_29']=result
    report['generation']['blocking_work']=['complete full codec MNN export and phone numerical validation',
        'train0.6B music/style adapters with legitimate paired data',
        'validate full LLM/DiT/VAE co-residency and bounded player queue',
        'user signs and installs independent diagnostic App']
    report['version']='0.1.3-full-codec-cpu'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(json.dumps({'complete_cpu_codec':True,'cfm_tensors':result['cfm_tensors'],
        'effective_pcm_seconds':result['effective_pcm_seconds'],'real_song_generated':False},ensure_ascii=False))


if __name__=='__main__':main()
