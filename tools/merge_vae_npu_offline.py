"""Preserve all four offline NPU cases in the single public test report."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import statistics

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
GROUPS=[('vae-npu-offline-20261030-r1','vae-npu-omc-conversion-result-20261030.json','vae-npu-block'),
        ('vae-npu-stack-20261030-r1','vae-npu-stack-conversion-result-20261030.json','vae-npu-stack'),
        ('vae-npu-compact-20261030-r1','vae-npu-compact-conversion-result-20261030.json','vae-npu-stack-compact')]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def text(p):return p.read_text(encoding='utf-8-sig')
def main():
    cases=[];records=[]
    for run,conversion,folder in GROUPS:
        raw=text(R/(run+'.log'));assert re.findall(r'^LEVO_PROBE_EXIT (\d+)\s*$',raw,re.M)==['0']
        rows=[json.loads(line.removeprefix('LEVO_OFFLINE_REPORT ')) for line in raw.splitlines() if line.startswith('LEVO_OFFLINE_REPORT ')]
        assert len(rows)==1 and rows[0]['probeCompleted']
        assert all('Current State: AWAKE' in text(R/(run+suffix+'.log')) for suffix in ('-before','-after'))
        compiled=json.loads(text(R/conversion));assert compiled['complete']
        identities=text(R/(run+'-shas.log'))
        for item in compiled['cases']:
            assert item['success'] and item['returnCode']==0
            assert sha(ROOT/'models'/folder/(item['id']+'-fp16.omc'))==item['sha256'] and item['sha256'] in identities
        records.append({'run':run,'result':rows[0],'conversion':compiled,'sourceLogSha256':sha(R/(run+'.log')),
                        'phoneIdentityLogSha256':sha(R/(run+'-shas.log'))})
        for row in rows[0]['cases']:
            npu=row['npu'];assert npu['success'] and npu['npuOnlyBuildAndRunSucceeded']
            assert not npu['cpuFallbackAllowed'] and npu['npuOnlyOptionCode']==0 and npu['disableFallbackCode']==0 and npu['buildCode']==0
            summary={'id':row['id'],'shape':row['shape']}
            for name in ('npu','cpuFp32LowNeon','cpuW8LowNeon'):
                item=row[name];assert item['success'] and item['numeric']['finite']
                times=item['runIncludingCopiesSeconds'];assert len(times)==4
                summary[name]={'firstMs':times[0]*1000,'warmMedianMs':statistics.median(times[1:])*1000,
                    'warmMinMs':min(times[1:])*1000,'warmMaxMs':max(times[1:])*1000,
                    'relativeL2VsOriginalFp32':item['numeric']['relativeL2'],'setupMs':item['setupSeconds']*1000}
            summary['npuSpeedupVsCpuW8']=summary['cpuW8LowNeon']['warmMedianMs']/summary['npu']['warmMedianMs']
            if 'runSyncSeconds' in npu:
                summary['npuRunSyncWarmMedianMs']=statistics.median(npu['runSyncSeconds'][1:])*1000
                summary['npuHostCopyWarmMedianMs']=statistics.median([a+b for a,b in zip(npu['hostInputCopySeconds'][1:],npu['hostOutputCopySeconds'][1:])])*1000
            cases.append(summary)
    assert len(cases)==4 and all(row['npuSpeedupVsCpuW8']<1 for row in cases)
    assert sha(ROOT/'native-build/levo_vae_npu_offline') in text(R/'vae-npu-compact-20261030-r1-shas.log')
    path=R/'levo-test-report.json';report=json.loads(path.read_text(encoding='utf8'))
    report['vae_npu_offline_performance_2026_10_30']={'asOfUtc':datetime.now(timezone.utc).isoformat(),
        'status':'npu-only-build-and-run-passed-all-four-cases-slower-than-cpu','cases':cases,'records':records,
        'finalProbeSourceSha256':sha(ROOT/'native-bench/VaeNpuOffline.cpp'),'appPathChanged':False,
        'recommendedAppPolicy':'chunks-5-w8-vsin','firstRunBinaryDiffersFromLaterTimingBreakdownProbe':True,
        'limits':['Real original weights with synthetic activations; no trained song or whole-VAE RTF result',
            'NPU weights FP16 and FP32 host IO; CPU Low FP32-disk and W8-disk both use NEON Sin, six threads; errors differ across precision/quantization recipes',
            'Exactly one first run and three warmed runs per case, fixed NPU then CPU order, no controlled temperature/frequency experiment',
            'NPU-only/no-fallback options accepted and offline build/run successful; no per-operator AI-core/AICPU hardware trace',
            'RunSync includes device compute and potentially internal layout/precision conversion, scheduling and waits; not pure kernel time or DRAM bandwidth',
            'Process RSS sampled every20ms; NPU private workspace peak not measured',
            'Stacked blocks validate independent tile numerics/interior; global sequence edge padding is not production-validated',
            'Compact and wide stack process32768 valid feature positions, but compact halo work is25% versus1.5625% for wide; not equal total padded FLOPs'],
        'nextAction':'Inspect compiled node/device kernels and internal precision/layout costs; try FP16 IO before any whole-decoder offload, retain CPU fallback-free comparisons'}
    # Correct the old field label from preparation; compact proof used2048 points.
    for key,length in [('vae_npu_stack_research_2026_10_30',32768),('vae_npu_compact_research_2026_10_30',2048)]:
        section=report[key]
        if 'continuous32768VsTiled4x8192Fp32L2' in section:
            section['continuousVsTiledFp32L2']=section.pop('continuous32768VsTiled4x8192Fp32L2')
        section['continuousProofLength']=length;section['continuousProofTileRows']=4
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(cases))
if __name__=='__main__':main()
