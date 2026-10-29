"""Preserve VAE optimization evidence in the existing single public report."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
def record(file, prefix):
    lines=[line[len(prefix):] for line in (REPORTS/file).read_text(encoding='utf-8-sig').splitlines() if line.startswith(prefix)]
    if len(lines)!=1:raise ValueError(f'Expected one completed record: {file}')
    return json.loads(lines[0])
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    modules={key:record(file,'LEVO_VAE_REPORT ') for key,file in {
        'fp32Low6':'vae-fp32-low6-v016-r1.log','w8Low6First':'vae-w8-low6-v016-r1.log',
        'w8Low6Repeat':'vae-w8-low6-v016-r2.log','w8VectorSin6':'vae-w8-vsin-low6-v016-r3.log',
        **{f'w8VectorSin{t}':f'vae-w8-vsin-t{t}-v016.log' for t in (2,4,8)}}.items()}
    for case in modules.values():
        assert case['success'] and case['validPcmSeconds']==25.6 and case['chunks']==3
        assert abs(case['decoderComputeSeconds']/25.6-case['decoderOnlyComputeRtf'])<1e-9
    baseline=record('five-vae-baseline-cli-awake-v016.log','LEVO_CHAIN_REPORT ')
    optimized=record('five-vae-optimized-cli-awake-v016.log','LEVO_CHAIN_REPORT ')
    app=json.loads((REPORTS/'signed-five-vae-w8-vsin-v016-r1.json').read_text(encoding='utf-8-sig'))['chunks5W8Vsin']
    for case in (baseline,optimized,app):
        assert case['success'] and case['generatedCodecIds']==1024 and case['finalKvPositions']==1537 and case['validPcmSeconds']==40.96
    assert optimized['vaeVectorSinKernelCalls']==app['vaeVectorSinKernelCalls']==1080
    for name in ('five-vae-awake-before-v016.log','five-vae-awake-mid-v016.log','five-vae-awake-after-v016.log','vae-thread-before-v016.log','vae-thread-after-v016.log'):
        assert 'Current State: AWAKE' in (REPORTS/name).read_text(encoding='utf-8-sig')
    tests=(REPORTS/'unit-tests-v016.log').read_text(encoding='utf-8-sig')
    assert 'Ran 15 tests' in tests and tests.rstrip().endswith('OK')
    receipt={}
    lines=(REPORTS/'vae-v016-phone-binary-shas.log').read_text(encoding='utf-8-sig').splitlines()
    for local,remote in [('levo_vae_kernel','levo_vae_vsin_scoped'),('levo_chain_vae_optimized','levo_chain_vae_scoped'),('levo_vae_npu_probe','levo_vae_npu_matched_v016')]:
        value=sha(ROOT/'native-build'/local);assert any(line.startswith(value) and line.endswith('/'+remote) for line in lines);receipt[local]=value
    npu=record('vae-npu-matched-cases-v016.log','LEVO_NPU_REPORT ')
    assert len(npu['cases'])==8 and all(c['matchedCpu']['success'] and c['success'] for c in npu['cases'])
    fusion_path=REPORTS/'vae-snake-fusion-v016-r2.log'
    fusion_lines=[line for line in fusion_path.read_text(encoding='utf-8-sig').splitlines() if line.startswith('LEVO_SNAKE_FUSION ')]
    fusion=json.loads(fusion_lines[0].removeprefix('LEVO_SNAKE_FUSION ')) if len(fusion_lines)==1 else {'success':False,'reason':'No completed structured result; bounded 45-second prototype, not promoted'}
    source_files=['native-bench/VaeBench.cpp','native-bench/VaeVectorSin.hpp','native-bench/VaeProfile.cpp','native-bench/VaeNpuProbe.cpp','native-bench/VaeSnakeFusion.cpp','native-bench/ChainBench.cpp','app/entry/src/main/cpp/napi_init.cpp']
    section={'asOfUtc':datetime.now(timezone.utc).isoformat(),'scope':'Real VAE optimization; synthetic latents/untrained music adapters, no listening acceptance',
        'modelValidationPc':json.loads((REPORTS/'vae-w8-pc-validation.json').read_text()),'vaeModuleCases':modules,
        'kernelSelfTest':record('vae-sin-exhaustive-v016.log','LEVO_SIN_SELFTEST '),
        'operatorProfiles':{v:record(f'vae-256-{v}-low-profile-v016.log','LEVO_VAE_PROFILE ') for v in ('fp32','w8')},
        'fiveChunkCliAwake':{'baseline':baseline,'optimized':optimized},'fiveChunkSignedApp':app,
        'package':{'version':'0.1.6','signedHapSha256':sha(ROOT/'app/entry/build/default/outputs/default/entry-default-signed.hap'),'signingMaterialPublished':False},
        'cannMatchedCases':npu,'cannPlacementLimit':'Single-op precheck COMMON=1, not OPTIMIZED=0. No proof of full decoder NPU placement; cannot infer forced-NPU graph speed.',
        'snakeFusionPrototype':fusion,'snakeFusionFirstAttempt':json.loads((REPORTS/'vae-snake-fusion-v016-r1-observation.json').read_text(encoding='utf-8-sig')),
        'sampledKernels':{'fp32CpuLow':'MNNPackedMatMulFP16; scalar sinf plus FP16 conversion',
            'w8CpuLow':'MNNGemmInt8AddBiasScale_ARMV86_Unit_FP16 at sampled address 0x2cbd90; FP16 GEMM remains',
            'source':'hiperf user-space cycle samples, 0 sample losses; assembly addresses resolved with the same unstripped binary',
            'dramBandwidthMeasured':False,'memoryBoundEstablished':False},
        'verifiedPhoneBinarySha256':receipt,'sourceSha256':{name:sha(ROOT/name) for name in source_files},
        'existingRegressionTestsPassed':15,'vaeThreeTimesSpeedupAchieved':False,
        'limits':['Module 1.77x improvement is not a whole-chain 1.77x gain; signed App and CLI differ',
            'W8 waveform L2 vs PyTorch about 6%; no music listening acceptance',
            'Earlier five-chunk pair ended SLEEP and is excluded from ranking',
            'Original graph allocations retained in callback fusion prototype; no memory saving claim',
            'Endpoint AWAKE checks do not replace continuous temperature/scheduling controls']}
    target=REPORTS/'levo-test-report.json';report=json.loads(target.read_text());report['version']='0.1.6-vae-optimization';report['vae_optimization_2026_10_29']=section;target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'cliRtfBefore':baseline['pipelineRtfThroughPcm'],'cliRtfAfter':optimized['pipelineRtfThroughPcm'],'signedAppRtf':app['pipelineRtfThroughPcm'],'fusion':fusion},ensure_ascii=False))
if __name__=='__main__':main()
