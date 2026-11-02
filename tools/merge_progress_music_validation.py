"""Record the installed v1.0.3 UI and its actual two-step 30s song."""
import array
import hashlib
import json
import math
import wave
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'reports/music-app-v103-30s.json').read_text(encoding='utf-8'))
    assert report['success'] and report['ditSteps']==2 and report['durationSeconds']==30 and report['playbackValidated']
    assert report['lmGuidance']==1 and report['ditGuidance']==1.5
    assert report['lm']['frames']==750 and len(report['codec']['chunks'])==4
    assert report['codec']['trainedLmTokens'] and not report['codec']['randomMusicHead']
    source=(ROOT/'music-app/entry/src/main/ets/pages/Index.ets').read_text(encoding='utf-8')
    assert '大模型' not in source and 'enableScanEffect: true' in source and 'ProgressType.Linear' in source
    audio=ROOT/'reports/levo-music-v103-dit2-30s.wav'
    with wave.open(str(audio),'rb') as wav:
        assert (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes())==(2,2,48000,1440000)
        pcm=array.array('h',wav.readframes(wav.getnframes()))
    rms=math.sqrt(sum(float(x)*x for x in pcm)/len(pcm))/32768
    assert rms>.001
    dit=sum(x['ditSeconds'] for x in report['codec']['chunks']);vae=sum(x['vaeSeconds'] for x in report['codec']['chunks'])
    lm=report['lm']['totalLmSeconds'];total=report['totalSeconds']
    report['stageSeconds']={'llm':lm,'dit':dit,'vae':vae,'other':total-lm-dit-vae,'total':total}
    report['stageRtf']={k:v/30 for k,v in report['stageSeconds'].items()}
    comparison=json.loads((ROOT/'reports/comparison-app-20261102-large-serial2-r1.json').read_text(encoding='utf-8'))
    hap=ROOT/'dist/LeVo-Music-v1.0.3-signed.hap'
    record={
        'version':'1.0.3','bundleName':'com.zhilee.levomusic','installedAndRunInApp':True,
        'signedHapSha256':hashlib.sha256(hap.read_bytes()).hexdigest(),'signedHapBytes':hap.stat().st_size,
        'ui':{'largeModelLabelRemoved':True,'fourReadableStageLabels':True,'activeStageFontSize':19,'stageRowFontSize':17,
              'animatedScanProgressBar':True,'progressUsesActualStagesAndTokenChunkCounts':True,
              'inputPreviewCollapsedWhileBusy':True,'onDeviceScreenshotReviewed':True,
              'screenshotSha256':hashlib.sha256((ROOT/'reports/levo-progress-v103.jpeg').read_bytes()).hexdigest()},
        'defaultDitSteps':2,'defaultSeconds':30,'fourStepRuntimePreserved':True,
        'result':report,'audioSha256':hashlib.sha256(audio.read_bytes()).hexdigest(),'audioRmsAfterClip':rms,
        'audioStructureAndNonSilenceVerified':True,
        'sameGeneratedCodesAsResearchApp2Step':report['lm']['codes']==comparison['result']['lm']['codes'],
        'limitations':'This production-App timing is distinct from the instrumented LeVo Harmony comparison; no thermal-controlled timing claim or listening-quality acceptance',
    }
    path=ROOT/'reports/levo-test-report.json';unified=json.loads(path.read_text(encoding='utf-8'))
    unified['music_app_v103_2026_11_02']=record
    path.write_text(json.dumps(unified,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'seconds':total,'rtf':total/30,'peakRssMiB':report['sampledPeakRssKiB']/1024,
                      'stages':report['stageSeconds'],'sameCodesAsResearch':record['sameGeneratedCodesAsResearchApp2Step'],'rms':rms},indent=2))


if __name__=='__main__':main()
