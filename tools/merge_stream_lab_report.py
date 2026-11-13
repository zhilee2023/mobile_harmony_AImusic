"""Append independently measured stream-lab results; preserve earlier history."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'reports/local'

def read(path):
    data=Path(path).read_bytes()
    return data.decode('utf-16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')

def case(name,scope):
    path=LOCAL/name
    if path.suffix=='.json':raw=json.loads(read(path))
    else:
        line=next(line for line in read(path).splitlines() if line.startswith('STREAM_LAB_REPORT '))
        raw=json.loads(line[len('STREAM_LAB_REPORT '):])
    keys=('status','error','actualFrames','requestedFrames','generatedSamples','elapsedSeconds','firstChunkSeconds','lmSeconds','muSeconds','ditSeconds','vaeSeconds',
          'ditSetupSeconds','vaeSetupSeconds','rssPeakMiB','minimumAvailableMiB','kvPositions','resident','tokenPrefetch','threads','cfgEnabled','ditBatch','ditSteps','chunkTimings','transportOnly','diagnosticOnly')
    result={key:raw[key] for key in keys if key in raw}
    duration=raw.get('generatedSamples',0)/48000
    result.update(scope=scope,durationSeconds=duration,chunks=len(raw.get('chunks',[])),inputNpy=Path(raw.get('npy','')).name)
    if duration and raw['status']=='completed':result['wholeRtf']=raw['elapsedSeconds']/duration
    return result

def main():
    path=ROOT/'reports/levo-test-report.json';report=json.loads(path.read_text(encoding='utf-8-sig'))
    definitions=[
        ('small10_initial_real','stream-lab-real-small-r1.log','CLI, real lyrics/NPY, original7-step DiT, per-window sessions'),
        ('large10_real','stream-lab-app-user-4b.json','App, real lyrics/NPY,4B consume-hidden projector,2-step DiT'),
        ('large60_partial_stop','stream-lab-app-user-4b-60s.json','User run stopped after53.76 seconds published; not a completed60-second RTF'),
        ('resident2_cli','stream-lab-resident-2step-r3.log','CLI15s,2steps, no CFG mixing but batch2, full LM head, sessions reused'),
        ('resident2_app15','stream-lab-resident-app-r1.json','App15s, same configuration, full LM head'),
        ('resident2_app60','stream-lab-resident-app-60s-r1.json','App60s, same configuration, full LM head'),
        ('head_crop_cli','stream-lab-crop-2step-r1.log','CLI15s, cropped music head'),
        ('head_crop_app_link_failed','stream-lab-crop-app-r1.json','Candidate App timing only; input embedding symlink caused token divergence, superseded'),
        ('head_crop_app_materialized','stream-lab-crop-app-materialized-r1.json','App15s, regular input embedding file;375codes and2WAVs identical to full-head baseline'),
        ('threads346','stream-lab-threads-346-r1.log','CLI15s,3/4/6threads; no verified speed gain'),
        ('prefetch344','stream-lab-prefetch-344-r1.log','CLI15s, next-window token work overlaps DiT/VAE; stage sums may exceed wall time; slower, disabled'),
        ('muq_default60','stream-lab-muq-default-60s.json','Final App60s, MuQ-MuLan text NPY default,2steps, no CFG mixing, batch2, cropped head,3resident models'),
    ]
    cases={key:case(name,scope) for key,name,scope in definitions}
    latest=json.loads(read(LOCAL/'stream-lab-muq-default-60s.json'))
    assert latest['status']=='completed' and latest['actualFrames']==1500 and len(latest['chunks'])==8
    assert latest['rssPeakMiB']<6*1024 and all(row['ditSetupSeconds']==0 and row['vaeSetupSeconds']==0 for row in latest['chunkTimings'][1:])
    events=[json.loads(line) for line in read(LOCAL/'stream-lab-final-events.jsonl').splitlines() if line.strip()]
    events=[row for row in events if row.get('runId')==latest['runId']]
    prefix=np.fromfile(LOCAL/'stream-lab-muq-prefix.f32',dtype='<f4').reshape(513,1024)
    old=np.fromfile(LOCAL/'stream-lab-real-prefix.f32',dtype='<f4').reshape(513,1024)
    assert np.array_equal(prefix[:512],old[:512]) and np.any(prefix[512]!=old[512])
    style=json.loads((ROOT/'models/styles/muq-text-warm-pop.json').read_text(encoding='utf-8'))
    hap=ROOT/'stream-app/entry/build/default/outputs/default/entry-default-signed.hap'
    section=dict(date='2026-11-12',app=dict(package='com.zhilee.levostreamlab',name='LeVo流式实验',version='0.1.6',hapSha256=hashlib.sha256(hap.read_bytes()).hexdigest()),
        device=dict(modelSelfReported='HUAWEI Mate90ProMax',backend='MNN CPU, no GPU/NPU',memory='20ms sampled process RSS; not dedicated VRAM',coreBinding='None; MNN thread requests only'),
        runtime=dict(mnn='Pinned3.6.1archive with project patches',mnnArchiveSha256='fbaa87b8cae9cd451560803ee3837886fe7e44d48976fc12654654412226159c',muqPcTorch='2.14.1+cpu',muqPcTransformers='5.21.0'),
        preservation='Old Apps/projects and original immutable bundles retained. No weights/audio/signing files published.',
        models=dict(small=dict(lmLayers=28,lmHidden=1024,lmFfn=3072,lmParametersOriginal=784969728,lmParametersAfterHeadCrop=629562368,ditLayers=22,ditHidden=1200,ditParameters=394012768,
            vaeParameters=84414082,condition='cond-token lookup1024, no4B mu projector',recipe='LM bodyW4/g64,headW8,BF16input;DiT W8/g64 High6;VAE W8 Low6 +NEON;LM High4; no requested A4/KV4'),
            large=dict(lmLayers=36,lmHidden=2560,lmFfn=9728,lmParameters=3675457536,muParameters=10039024,ditLayers=39,ditHidden=1200,ditParameters=688160368,vaeParameters=84414082,condition='consume-aligned hidden2560 through trained mu projector1024')),
        currentConfiguration=dict(ditSteps=2,cfgMixing=False,actualDitBatch=2,unconditionalComputeRemoved=False,residentSmallModules=3,crossRequestCache=False,threads=dict(lm=4,dit=6,vae=6),rssBoundaryGiB=6,systemAvailableBoundaryMiB=650,tokenPrefetch=False),
        cases=cases,
        headCrop=dict(sourceClasses=168214,retainedRows=[151765,168213],classes=16449,audioOutputBase=65,eosOutput=0,inputAudioBase=151830,weightBytesBefore=412069026,weightBytesAfter=251805186,inputEmbeddingBytes=344502272,
            graphSha256='25f67110669e9358b2dd224023795a2a29011ba348df3ba76672cab939c2085c',weightSha256='0a828f2cd9c8734c389b3d32f875f197f6da901c2b6eea7339d39ced59533f09',
            prefillPlus8TeacherLogitsL2=0,whole15SecondCodeSha256='dcd21c29c0d79144c1bfac5aa952de954534641bd4a769d47b1f86ec8b1d2931',
            matchingWavSha256=['10f3ecd7e9f898451711e0789c9b94f4c7b57769002d049b7cd24cb2aa7bf11b','2c9ee79453a0284391ade70099d8bf7d5c163448d190c86e68c2bae1a6988bdb'],
            appLesson='Materialize mobile input embeddings. CLI symlink success did not imply sandbox success; corrected App375codes/PCM match exactly.'),
        style=style,
        conditioningValidation=dict(oldPrefixRows513L2=0,newStyleKeepsFirst512Rows=True,newStyleChangedElements=int(np.count_nonzero(prefix[512]!=old[512])),newNpyReadConfirmed=latest['npy'].endswith('/muq-text-warm-pop.npy')),
        playback=dict(durationMs=60000,maximumHardwarePositionMs=max(row.get('playedMs',0) for row in events),maximumSubmittedQueueMs=max(row.get('queueMs',0) for row in events),errors=sum(row['kind']=='playback-error' for row in events),rendererRecoveries=[row['detail'] for row in events if 'rendererRecoveries' in row.get('detail','')],scope='Hardware position and submitted queue, not a speaker recording'),
        listening=dict(userConfirmedSmallAndLargeRealMusic=True,newMuqDefaultSubjectiveQuality='User confirmed normal music',userConfirmedNewMuqDefault=True,confirmationDate='2026-11-13',userFeedback='正常，有音乐',objectiveQualityEvaluated=False,styleMatchEvaluated=False),
        experimentsNotEnabled=dict(singleBranch=dict(scope='Static-batch shape rewrite, unchanged W8 weights',relativeL2=[.11572169642199259,.12594451849999524,.11826457718728889],budget=1e-5,passed=False),prefetch='31.389s/15s,slower;codes identical,not enabled',threads='3/4/6did not beat4/6/6'),
        validation=dict(pythonTests=51,passed=True,signedAppBuild=True,defaultStyleActual60Seconds=True,maximumDurationTestedSeconds=60,longerDurationGuaranteed=False),
        limitations=['RTF remains above1, playback can underrun while generation continues.','Different styles/thermal states are not a controlled causal speed comparison.','Batch1 rewrite and concurrent token prefetch are not the App defaults.','MuQ runs offline on PC; only512float32 features are deployed.','Early fixed-code tests were synthetic numeric fixtures, not model-generated music.'])
    report['independent_stream_lab_2026_11_12']=section
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(rtf=cases['muq_default60']['wholeRtf'],rssPeakGiB=latest['rssPeakMiB']/1024,hapSha256=section['app']['hapSha256']),indent=2))

if __name__=='__main__':main()
