"""Publish measured 90s phases while keeping recovered computation distinct from single-run E2E."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exit", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    small, lm, exit_info, recovery = map(read, (args.small, args.checkpoint, args.exit, args.codec))
    s, c = small["result"], recovery["result"]
    assert small["foregroundStable"] and small["displayStable"] and s["success"]
    assert s["durationSeconds"] == 90 and s["lm"]["frames"] == 2250 and s["lm"]["finalKvPositions"] == 2763
    assert s["diagnosticOnly"] and not s["musicTrained"]
    assert len(lm["codes"]) == lm["frames"] == 2250
    assert recovery["foregroundStable"] and recovery["displayStable"] and c["success"]
    assert c["originalRealCodes"] == 2250 and c["repeatedFixtureCodes"] == 0
    assert c["codec"]["audioFileGenerated"] and not c["codec"]["performanceFixtureOnly"]
    for result in (s, c):
        assert result["codec"]["durationSeconds"] == 90 and len(result["codec"]["chunks"]) == 12
        assert result["codec"]["ditSteps"] == 2 and result["codec"]["vaeNeonSinCalls"] > 0
        result["memory"]["peakRssKiB"] = max(result["memory"]["peakRssKiB"], result["sampledPeakRssKiB"])
    llm_seconds = lm["totalLmSeconds"]
    codec_seconds = c["stageSeconds"]["totalCodecOnly"]
    record = {
        "date": "2026-11-02", "app": "LeVo Harmony v0.1.11",
        "effectiveAudioSeconds": 90, "codes": 2250, "windows": 12,
        "smallComplete": small, "publicLmCheckpoint": lm, "publicExitRecord": exit_info, "publicCodecRecovery": recovery,
        "publicSingleRunWholeChainCompleted": False, "publicSingleRunWholeChainRtf": None,
        "publicMeasuredPhaseSum": {"llmThroughLastCodeSeconds": llm_seconds, "codecSeconds": codec_seconds,
                                   "seconds": llm_seconds + codec_seconds, "computeRtf": (llm_seconds + codec_seconds) / 90,
                                   "llmRtfThroughLastCode": llm_seconds / 90,
                                   "meaning": "Sum of measured LM token-production and recovery Codec phases, NOT one uninterrupted end-to-end run",
                                   "excluded": ["Condition preparation before LM timer", "Failed LM cleanup", "Exit and relaunch/transfer/build gap"]},
        "publicCapturedPeakRssLowerBoundKiB": max(lm["checkpointSampledPeakRssKiB"], c["memory"]["peakRssKiB"]),
        "publicCompletePeakKnown": False,
        "recipe": {
            "publicLlm": {"model": "trained primary Mixed LM only; track LM omitted", "layers": 36, "hidden": 2048, "exportedHfStateParameters": 3106035712,
                          "body": "W4/group64", "head": "W8", "embedding": "BF16", "precision": "High", "threads": 4},
            "studentLlm": {"model": "26-layer pruned Qwen3-0.6B backbone; untrained music head/adapters", "layers": 26, "hidden": 1024,"sourceParameters":598178304,
                           "body": "W4/group64", "head": "W8", "embedding": "BF16", "precision": "Low", "threads": 4},
            "publicDit": {"sourceParameters": 751534224, "layers": 39, "hidden": 1200, "ffn": 4800, "weights": "W8/group64"},
            "studentDit": {"sourceParameters": 195416128, "layers": 22, "hidden": 1024, "ffn": 2048, "weights": "W8/group64", "musicTrained": False},
            "vae": {"sourceParameters": 84414082, "weights": "W8", "precision": "Low", "threads": 6, "neonSin": True, "sharedWeights": True},
            "common": {"ditPrecision": "High", "ditThreads": 6, "ditStepsPerWindow": 2, "ditCfg": 1.5, "lmCfg": 1,
                       "windowFrames": 256, "overlapFrames": 64, "hopFrames": 192, "memoryMode": "Low", "powerMode": "High",
                       "codecKleidiAiHint": 0, "requestedStaticA4": False, "requestedKv4": False, "eosSuppressedForBenchmark": True},
            "externalWeightBytesOnly": {"publicLlm": 1650155246, "studentLlm": 248296682, "publicDit": 722214236,
                                       "studentDit": 221684024, "sharedVae": 84569334},
            "mnnStaticLibrarySha256": "fbaa87b8cae9cd451560803ee3837886fe7e44d48976fc12654654412226159c",
        },
        "limitations": [
            "Public LM produced all2250realcodes before process exit. Recovery uses exactly those codes without repetition or padding and writes90seconds of real waveform.",
            "Public phase-sum RTF excludes failed cleanup and recovery gap; never label it single-run E2E RTF.",
            "Student is an untrained music prototype. Finite PCM and performance do not establish song quality.",
            "Foreground was polled about every3seconds; screen-awake checks alone are insufficient. Public exit lacks a completed foreground archive, but no foreground violation was logged before exit.",
            "RSS20ms peaks are sampled; public checkpoint peak is a lower bound before termination, not guaranteed full-run peak. HiDebug PSS includes Pss+SwapPss.",
            "All inference uses CPU. Graphics API values are App-attributed samples, not model dedicated VRAM or whole-device GPU usage.",
            "USB charging was active. Power curves show battery-terminal |VI| magnitude, not App/SoC/whole-device power. Vendor sys_power_now units unverified.",
        ],
    }
    record["powerThermalRawSamples"] = {path.name: [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()] for path in args.sensors}
    target = ROOT / "reports/levo-test-report.json"
    unified = read(target)
    unified["serial_chain_comparison_90s_2026_11_02"] = record
    target.write_text(json.dumps(unified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"studentSeconds": s["stageSeconds"], "studentRtf": s["stageRtf"], "publicPhaseSum": record["publicMeasuredPhaseSum"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
