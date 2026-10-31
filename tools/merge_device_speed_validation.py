"""Merge a signed-App, same-configuration phone comparison into one report."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--with-dit-comparison", action="store_true")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in args.run_id):
        parser.error("run_id must contain only letters, digits and hyphens")
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    current = load(reports / f"signed-{args.run_id}.json")["chunks5W8Vsin"]
    receipt = load(reports / f"signed-{args.run_id}-receipt.json")
    device = load(reports / f"{args.run_id}-device.json")
    destination = reports / "levo-test-report.json"
    archive = load(destination)
    baseline = archive["vae_snake_app_validation_2026_10_30"]["samePackageOriginalW8Neon"]
    assert current["success"] and current["flowAndMemoryPassed"]
    assert len(current["chunks"]) == 5 and current["generatedCodecIds"] == 1024
    assert current["ditStepsPerChunk"] == 2 and current["ditWeightBits"] == 8
    assert current["vaeVariant"] == "w8-vsin" and current["prefillTokens"] == 514
    assert abs(current["validPcmSeconds"] - 40.96) < 1e-6
    assert device["installedHapSha256"] == receipt["hapSha256"]
    assert device["runId"] == args.run_id
    assert len(device["verifiedModelFiles"]) == 22
    metrics = ("initializationSeconds", "prefillSeconds", "decodeSeconds", "sumDitSeconds",
               "sumVaeSeconds", "pipelineSecondsThroughPcm", "pipelineRtfThroughPcm",
               "pipelinePeakRssKiB", "decodeTokensPerSecond", "firstNumericalPacketSeconds")
    comparison = {}
    for name in metrics:
        old, new = baseline[name], current[name]
        comparison[name] = {"previousPhone": old, "newPhone": new,
                            "newOverPrevious": new / old}
    archive["device_cpu_speed_2026_10_31"] = {
        "scope": "Same signed v0.1.7 App and selected weights, CPU five-window numerical pipeline; single run per device, not controlled thermal/scheduler benchmark or trained music quality validation",
        "runId": args.run_id,
        "device": device,
        "receipt": receipt,
        "baselinePath": "vae_snake_app_validation_2026_10_30.samePackageOriginalW8Neon",
        "configuration": {"llm": "Qwen3-0.6B diagnostic student, W4/g64 body, W8 untrained music head, CPU Low/4 threads",
                          "dit": "Complete 39-layer W8, CPU High/6 threads, Memory Low, Power High, 2 Euler steps/window; custom CosyVoice Kleidi FFN not migrated",
                          "vae": "Original W8, CPU Low/6 threads, NEON Sin",
                          "stream": "One prefill and persistent KV; five 256-frame windows with 64-frame overlap; 40.96 seconds effective PCM"},
        "comparison": comparison,
        "pipelineSpeedup": baseline["pipelineSecondsThroughPcm"] / current["pipelineSecondsThroughPcm"],
        "pipelineTimeReductionFraction": 1 - current["pipelineSecondsThroughPcm"] / baseline["pipelineSecondsThroughPcm"],
        "result": current,
    }
    if args.with_dit_comparison:
        runs = []
        for case in ("w8-r1", "w4-r1", "w4-r2", "w8-r2"):
            lines = (reports / f"mate90-dit-{case}-20261031.log").read_text(encoding="utf-8-sig").splitlines()
            matches = [line.removeprefix("LEVO_DIT_PROFILE ") for line in lines if line.startswith("LEVO_DIT_PROFILE ")]
            assert len(matches) == 1
            row = json.loads(matches[0])
            assert row["success"] and row["variant"] == case.split("-")[0]
            assert row["threads"] == 6 and row["kleidiaiEnabled"] is False
            runs.append({"case": case, "result": row})
        medians = {variant: statistics.median(item["result"]["profiledSeconds"] for item in runs
                                              if item["result"]["variant"] == variant)
                   for variant in ("w8", "w4")}
        models = {}
        for variant in ("w8", "w4"):
            path = root / "models" / "full-dit-256" / f"full-{variant}.mnn.weight"
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            models[variant] = {"bytes": path.stat().st_size, "sha256": digest}
        archive["device_cpu_speed_2026_10_31"]["ditW4W8Comparison"] = {
            "scope": "Two reversed-order pairs, same fixed input, full 39-layer DiT, CLI CPU High/6 threads; one warmup and one callback/debug forward per process. Not App five-chunk W4 RTF or music quality validation; RSS not sampled in this module probe.",
            "order": [item["case"] for item in runs], "runs": runs,
            "profiledSecondsMedian": medians,
            "w4OverW8TimeRatio": medians["w4"] / medians["w8"],
            "weightFiles": models,
            "decision": "Keep W8 for the current CPU speed route; W4 saves stored weights but is slower in this paired module experiment.",
        }
    destination.write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run": args.run_id, "rtf": current["pipelineRtfThroughPcm"],
                      "seconds": current["pipelineSecondsThroughPcm"],
                      "speedup": archive["device_cpu_speed_2026_10_31"]["pipelineSpeedup"],
                      "strictParity": current["strictNumericalParityPassed"]}, indent=2))


if __name__ == "__main__":
    main()
