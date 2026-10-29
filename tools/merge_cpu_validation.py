"""Merge existing completed CPU results only; never starts tests or downloads."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
path=root/'reports/levo-test-report.json'
report=json.loads(path.read_text(encoding='utf8'))
sections={
    'mixed_dit_constructor':'mixed-dit-restoration.json',
    'mixed_sampler_and_rvq_parity':'mixed-primitives-parity.json',
    'pretrained_block_cpu':'real-dit-block-cpu.json',
    'pretrained_block_mnn':'real-dit-block-mnn.json',
    'weight_quantization_recipes':'dit-block-quant-recipes.json',
    'vae_decoder_constructor':'vae-decoder-restoration.json',
}
cpu={}
for key,name in sections.items():
    candidate=root/'reports'/name
    if candidate.exists():cpu[key]=json.loads(candidate.read_text(encoding='utf8'))
cpu['hqq']={'validated':False,'reason':'Installed converter disabled HQQ in the attempted run; follow-up flag probe was not executed after approval-review timeout.'}
downloads={}
for name,folder,total in [('qwen_backbone','models/qwen3-0.6b/parts',1503300328),
                          ('vae_decoder','models/levo-vae/decoder-parts',338287232)]:
    files=list((root/folder).glob('*'))
    downloads[name]={'completed_range_bytes':sum(p.stat().st_size for p in files if p.is_file() and p.name.isdigit()),
                     'expected_download_bytes':total,'complete':False,
                     'stop_reason':'DNS getaddrinfo failed; chunks retained; no automatic new-window retry'}
cpu['download_state']=downloads
cpu['scope_note']='Real block0 and codec buffers validated; full pretrained DiT, Qwen backbone and VAE decoder have not been loaded end-to-end.'
report['cpu_validation_2026_10_28']=cpu
report['generation']['blocking_work']=['finish Qwen download and actual full backbone load',
    'finish and strictly load restored full Mixed DiT/VAE weights',
    'music-adapter training with legitimate data', 'complete MNN graph export and phone validation']
report['version']='0.1.1-cpu-validation'
path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps({'sections_merged':list(cpu),'downloads':downloads},indent=2))
