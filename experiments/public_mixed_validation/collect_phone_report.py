"""Validate and merge the complete phone LM/Codec experiment into one report."""
import hashlib
import json
import re
import wave
from pathlib import Path

from collect_report import verify_wave

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def main():
    report = json.loads((ART / "whole-phone-report.json").read_text(encoding="utf-8"))
    assert report["success"] and report["wholePhoneChain"]
    assert report["ditSteps"] == 4 and report["lmDevice"] == "Harmony phone CPU"
    codes = json.loads((ART / "phone-generated-codes.json").read_text(encoding="utf-8"))
    assert codes["source"] == "trained public LeVo v2-large primary Mixed LM"
    assert codes["lmDevice"] == "Harmony phone CPU" and not codes["randomMusicHead"]
    assert 0 < len(codes["codes"]) <= 750 and all(0 <= i < 16384 for i in codes["codes"])
    duration = len(codes["codes"]) / 25
    assert abs(duration - report["durationSeconds"]) < 1e-6
    wav = ART / "whole-phone-dit4.wav"
    with wave.open(str(wav), "rb") as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (2, 2, 48000)
        assert stream.getnframes() == len(codes["codes"]) * 1920
    if duration == 30:
        audio = verify_wave(wav)
    else:
        audio = {"file": str(wav), "durationSeconds": duration, "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                 "structureVerified": True, "listeningQualityAccepted": False}
    assert abs(report["secondsThroughWav"] / duration - report["wholePhoneRtf"]) < .01
    report["audioVerification"] = audio
    report["listeningQualityAccepted"] = False
    observations = {}
    for phase in ("before", "after"):
        power = ROOT / f"phone-power-{phase}-whole.txt"
        if power.is_file():
            raw = power.read_text(encoding="utf-8-sig", errors="replace")
            state = re.search(r"Current State:\s+(\w+)", raw)
            off = re.search(r"Last Screen Off:\s+(\d+)", raw)
            observations[phase] = {"state": state.group(1) if state else None,
                                   "lastScreenOff": int(off.group(1)) if off else None}
    if len(observations) == 2:
        observations["noInterveningScreenOffRecorded"] = (
            observations["before"]["lastScreenOff"] is not None
            and observations["before"]["lastScreenOff"] == observations["after"]["lastScreenOff"]
        )
    report["powerObservations"] = observations
    report["quantization"] = {"lmBodyWeightBits": 4, "lmHeadWeightBits": 8, "ditWeightBits": 8, "vaeWeightBits": 8,
                              "notW4A4Kv4": True}
    path = ROOT / "levo-public-music-report.json"
    combined = json.loads(path.read_text(encoding="utf-8"))
    combined["wholePhone4Steps"] = report
    attempts = []
    for name in ("whole-phone-report-5gib-stop.json", "whole-phone-report-high-memory-stop.json"):
        prior = ART / name
        if prior.is_file():
            attempts.append(json.loads(prior.read_text(encoding="utf-8")))
    low_abort = ROOT / "whole-phone-dit4-low-load-abort.log"
    if low_abort.is_file():
        attempts.append({"success": False, "precision": "low", "lmGuidance": 1.5,
                         "error": "Native MNN CPUBackend allocBuffer assertion during LM load; no fresh completion report",
                         "rawLogSha256": hashlib.sha256(low_abort.read_bytes()).hexdigest()})
    combined["wholePhonePriorAttempts"] = attempts
    combined["conclusion"]["wholePhoneTextToMusicTested"] = True
    combined["limitations"] = [line for line in combined["limitations"] if not line.startswith("Phone codec RTF")]
    combined["limitations"].append("Earlier codec-only tests exclude the PC LM; the new wholePhone4Steps section measures all stages on the phone.")
    path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"success": True, "audioSeconds": duration, "wholePhoneRtf": report["wholePhoneRtf"],
               "totalSecondsThroughWav": report["secondsThroughWav"], "lmSeconds": codes["totalLmSeconds"],
               "prefillSeconds": codes["prefillSeconds"], "decodeSeconds": codes["decodeSeconds"],
               "decodeTokensPerSecond": codes["decodeTokensPerSecond"], "codecSeconds": report["codec"]["codecSeconds"],
               "lmGuidance": codes["lmGuidance"], "ditGuidance": report["codec"]["codecGuidance"],
               "lmInstances": codes["lmInstances"], "lmPrecision": codes["precision"],
               "vaeNeonSinCalls": report["codec"]["vaeNeonSinCalls"],
               "sumDitSeconds": sum(c["ditSeconds"] for c in report["codec"]["chunks"]),
               "sumVaeSeconds": sum(c["vaeSeconds"] for c in report["codec"]["chunks"]),
               "peakProcessRssMiB": report["sampledPeakRssKiB"] / 1024,
               "minimumSystemAvailableMiB": report["minimumSystemAvailableKiB"] / 1024,
               "audio": str(wav), "report": str(path)}
    (ROOT / "whole-phone-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
