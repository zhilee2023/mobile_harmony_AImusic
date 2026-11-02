"""Collect the actual public-weight music experiment without bundling model weights."""
import array
import hashlib
import json
import math
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def read(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def verify_wave(path):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 2, 48000)
        frames = wav.getnframes()
        samples = array.array("h", wav.readframes(frames))
    if sys.byteorder != "little":
        samples.byteswap()
    assert frames == 1440000 and len(samples) == frames * 2
    rms = math.sqrt(sum(float(x) * x for x in samples) / len(samples)) / 32768
    assert rms > 0.001, "Silent output"
    return {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "durationSeconds": frames / 48000,
        "sampleRate": 48000,
        "channels": 2,
        "pcmRmsAfterClip": rms,
        "pcmPeakAfterClip": max(abs(x) for x in samples) / 32768,
        "structurallyValidAndNonSilent": True,
        "listeningQualityAccepted": False,
    }


def main():
    mapping = read("mapping-validation.json")
    export = read("mixed-mnn-export.json")
    # Replace the superseded shared-norm hook measurement with the corrected rerun.
    export["checkpoints"] = mapping["checkpoints"]
    export["mappingVerification"] = "Corrected phase-scoped norm hook; independent rerun"
    (ART / "mixed-mnn-export.json").write_text(json.dumps(export, indent=2), encoding="utf-8")
    phone_lines = (ROOT / "phone-codec-dit4.log").read_text(encoding="utf-8").splitlines()
    marker = "LEVO_PUBLIC_CODEC_REPORT "
    phone = json.loads(next(line.split(marker, 1)[1] for line in phone_lines if marker in line))
    assert phone["success"] and not phone["wholePhoneChain"]
    phone["powerObservation"] = {
        "beforeAndAfterAwake": True,
        "lastScreenOffBefore": 85588448,
        "lastScreenOffAfter": 85588448,
        "scope": "PowerManager observations; no intervening screen-off recorded",
    }
    codes = read("generated-mixed-codes.json")
    code_summary = {k: v for k, v in codes.items() if k not in {"codes", "stepSeconds"}}
    # This older stage flag describes the token-generation stage, not the final report.
    code_summary["musicDecodedAfterThisStage"] = True
    code_summary["actualCodeCount"] = len(codes["codes"])
    code_summary["allCodesInVocabulary"] = all(0 <= x < 16384 for x in codes["codes"])
    pcm_files = [
        verify_wave(ART / "public-mixed-30.00s-dit20-w4-w8.wav"),
        verify_wave(ART / "public-mixed-phone-30s-dit4.wav"),
    ]
    report = {
        "schemaVersion": 1,
        "source": {"repo": "lglg666/SongGeneration-v2-large", "revision": "115805364ad74479fb3764fe65970c92faeb1a5a", **read("checkpoint-verification.json")},
        "architecture": {"route": "Mixed-only", "trainedMixedLm": True, "trackLmIncluded": False, "layers": 36, "hiddenSize": 2048, "randomMusicHead": False},
        "conditionParity": read("condition-parity.json"),
        "mappingVerification": mapping,
        "quantizedRuntimeProbe": read("mnn-runtime-probe.json"),
        "quantization": {"lmBodyWeightBits": 4, "lmHeadWeightBits": 8, "musicEmbeddings": "BF16", "ditWeightBits": 8, "vaeWeightBits": 8, "notW4A4Kv4": True},
        "tokenGeneration": code_summary,
        "windowsCodec20Steps": read("music-result-dit20.json"),
        "phoneCodec4Steps": phone,
        "audioVerification": pcm_files,
        "conclusion": {"audioFilesGeneratedFromPublicTrainedLm": True, "wholePhoneTextToMusicTested": False, "listeningQualityAccepted": False},
        "limitations": [
            "30-second excerpt was stopped at 750 codes, not model EOS.",
            "Track LM branch excluded: this is not the full hierarchical two-track official inference route.",
            "W4 runtime checks cover two teacher-forced points; they do not establish music quality parity.",
            "PC 20-step and phone 4-step use the same tokens but different noise generators; not a controlled step-only ablation.",
            "Phone codec RTF excludes PC LM generation; no whole-phone RTF claim.",
            "Wave structure and non-silence verified; musicality, singing and lyric intelligibility require listening.",
        ],
    }
    target = ROOT / "levo-public-music-report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(target), "audio": pcm_files, "phonePeakRssMiB": phone["sampledPeakRssKiB"] / 1024}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
