"""Archive a completed student run and Codec-only run without inventing a full public RTF."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--failed", type=Path, nargs="+", required=True)
    parser.add_argument("--clocks", type=Path)
    args = parser.parse_args()
    small, codec = load(args.small), load(args.codec)
    s, c = small["result"], codec["result"]
    assert s["success"] and small["displayStable"] and s["durationSeconds"] == 120
    assert s["lm"]["frames"] == 3000 and s["lm"]["finalKvPositions"] == 3513
    assert s["diagnosticOnly"] and not s["musicTrained"]
    assert c["success"] and codec["displayStable"] and c["codecOnly"]
    assert not c["wholeChainCompleted"] and c["wholeChainRtf"] is None and not c["llmExecuted"]
    assert c["originalRealCodes"] + c["repeatedFixtureCodes"] == 3000
    for result in (s, c):
        assert result["codec"]["durationSeconds"] == 120 and len(result["codec"]["chunks"]) == 16
        assert result["codec"]["ditSteps"] == 2 and result["codec"]["vaeNeonSinCalls"] > 0
        assert not result["codec"]["audioFileGenerated"]
        result["memory"]["peakRssKiB"] = max(result["memory"]["peakRssKiB"], result["sampledPeakRssKiB"])
        result["memory"]["pssMetric"] = "HiDebug NativeMemInfo.pss = Pss + SwapPss; not resident-only PSS"
        result["memory"]["queryOverheadInterpretation"] = "Background cumulative query duration includes waiting; do not subtract from wall time"
    failures = [load(path) for path in args.failed]
    assert all(not row["success"] and row["wholeChainRtf"] is None for row in failures)
    target = ROOT / "reports/levo-test-report.json"
    unified = load(target)
    previous = unified["serial_chain_comparison_2026_11_02"]
    record = {
        "date": "2026-11-02", "status": "student-complete-public-full-chain-not-complete",
        "public120WholeChainRtf": None, "student120WholeChainRtf": s["stageRtf"]["total"],
        "studentForegroundVerification": small.get("foregroundStable", None),
        "studentBaselineValidity": "Historical completed run without continuous foreground checks; reference only, not a strict foreground baseline",
        "smallComplete": small, "publicCodecOnly": codec, "publicFailedAttempts": failures,
        "models": previous["models"],
        "recipe": {
            "llm": {"body": "W4/group64", "head": "W8", "inputEmbedding": "BF16", "threads": 4,
                    "publicPrecision": "High", "studentPrecision": "Low", "lmCfg": 1,
                    "prefixPositionsPublic": 953, "prefixPositionsStudent": 514,
                    "incrementalKv": True, "requestedStaticA4": False, "requestedKv4": False,
                    "kvStorageTypeIndependentlyAudited": False,
                    "sampling": {"seed": 2307, "topK": 50, "temperature": .9, "repetitionWindow": 50, "repetitionPenalty": 1.1},
                    "eosSuppressedForFixedLengthBenchmark": True},
            "dit": {"weights": "W8/group64", "precision": "High", "threads": 6, "steps": 2, "cfg": 1.5,
                    "windowFrames": 256, "overlapFrames": 64, "hopFrames": 192, "windows": 16},
            "vae": {"weights": "W8", "precision": "Low", "threads": 6, "neonSinHookDuringBuild": True,
                    "sampleRateHz": 48000, "channels": 2, "sharedPretrainedWeights": True},
            "lifetime": "Complete and destroy LLM, then reuse a single DiT and VAE session for all windows",
            "backend": "CPU only; graphics API values are not dedicated model VRAM",
            "memoryMode": "Low", "powerMode": "High",
            "mnnStaticLibrarySha256": "fbaa87b8cae9cd451560803ee3837886fe7e44d48976fc12654654412226159c",
            "externalWeightBytesOnly": {"publicLlm": 1650155246, "studentLlm": 248296682,
                                       "publicDit": 722214236, "studentDit": 221684024, "vae": 84569334},
        },
        "interpretation": [
            "Student generated3000newcodes and120seconds of finite PCM with untrained music components, not a song-quality test.",
            "Student120-second run predates foreground polling; screen-awake alone does not validate foreground execution.",
            "Public CPU High full-chain attempts exited at2725and2450checkpoint codes. No complete120-second public whole-chain RTF exists.",
            "Public CPU Low failed during Llm.load: Arm82 zero-size allocation assertion, SIGABRT. Not a usable deployed recipe.",
            "Codec-only fixture uses2725real partial codes plus275repeated fixture codes. Its total is Codec-only, never add it to an interrupted LM time and call it whole-chain RTF.",
            "Studentv0.1.9 and public diagnosticv0.1.10 are different signed HAPs with the same pinned MNN library. No strict thermal/cache/model-size-only ablation claim.",
            "RSS20ms uses an independent sampler. PSS/graphics queries can perturb execution and miss short peaks; graphics0does not imply whole-device GPU usage is zero.",
            "No confirmed OOM diagnosis. The second High run did record a transient minimum below1GiB before unclassified termination.",
        ],
        "prior30sStudentRtf": previous["cases"]["small"]["result"]["stageRtf"]["total"],
    }
    if args.clocks:
        record["publicHighInterruptedClockObservations"] = [json.loads(line) for line in args.clocks.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    unified["serial_chain_comparison_120s_2026_11_02"] = record
    target.write_text(json.dumps(unified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"small": s["stageSeconds"], "publicCodecOnly": c["stageSeconds"], "publicWholeChainRtf": None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
