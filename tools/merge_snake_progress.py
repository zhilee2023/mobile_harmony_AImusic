"""Merge the successful bounded Snake probe without replacing failed history.

Requires the exact local probe artifact used by this run, not a later rebuild.
Only the unified report is published; per-run logs and binaries stay local.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RUN = "snake-paired-neon-all-20261030-r1"


def read(name):
    return (REPORTS / name).read_text(encoding="utf-8-sig")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    raw = read(RUN + ".log")
    records = [json.loads(line.removeprefix("LEVO_SNAKE_FUSION "))
               for line in raw.splitlines() if line.startswith("LEVO_SNAKE_FUSION ")]
    assert len(records) == 1
    case = records[0]
    assert re.findall(r"^LEVO_PROBE_EXIT (\d+)\s*$", raw, re.M) == ["0"]
    assert case["success"] and case["windows"] == 3
    assert case["capturedGroups"] == 36 and case["fusedCalls"] == 108
    assert case["fusedCallsCompleted"] == 108 and case["baselineSin"] == "neon"
    assert case["vectorSinKernelCalls"] == 864
    assert len(case["rows"]) == 3 and all(row["l2VsOriginalPhoneW8"] == 0 for row in case["rows"])
    assert case["warmOriginalL2VsPhoneReference"] == 0
    assert math.isclose(sum(row["seconds"] for row in case["rows"]), case["computeSeconds"])
    assert math.isclose(sum(row["baselineSeconds"] for row in case["rows"]), case["baselineComputeSeconds"])
    assert case["validPcmSeconds"] == 25.6
    assert math.isclose(case["vaeOnlyRtf"], case["computeSeconds"] / 25.6)
    assert math.isclose(case["speedupVsPairedBaseline"], case["baselineComputeSeconds"] / case["computeSeconds"])
    assert case["producerSnapshotPeakBytes"] == 120 * 1024**2
    assert all("Current State: AWAKE" in read(RUN + suffix + ".log") for suffix in ("-before", "-after"))
    observation = json.loads(read(RUN + "-observation.json"))
    assert observation["durationCapSeconds"] == 90 and observation["wallSeconds"] < 90
    hashes = read(RUN + "-shas.log")
    binary = ROOT / "native-build/levo_vae_snake_fusion"
    coefficient = ROOT / "models/vae-stream-256/snake-coefficients.json"
    assert sha(binary) in hashes and sha(coefficient) in hashes

    historical = []
    for stem, status in (
        ("snake-direct-first-20261030-r1", "timeout-before-safe-callback-stop"),
        ("snake-safe-layout-20261030-r1", "rejected-unexpected-layout"),
        ("snake-c8-abi-direct-first-20261030-r1", "failed-numerics-reused-buffer"),
        ("snake-baseline-gate-20261030-r1", "original-passed-fusion-failed-reused-buffer"),
        ("snake-producer-direct-first-20261030-r1", "rejected-nonexecuted-reshape-producer"),
        ("snake-producer-trace-direct-first-20261030-r1", "one-group-pcm-equivalence-passed"),
        ("snake-producer-pool-all-20261030-r1", "all-groups-one-window-passed-scalar-baseline"),
    ):
        historical.append({"run": stem, "status": status,
                           "sourceLogSha256": sha(REPORTS / (stem + ".log"))})

    section = {
        "asOfUtc": datetime.now(timezone.utc).isoformat(),
        "humanAuthorizationScope": "Later human request to continue testing; bounded VAE callback fusion localization and paired validation",
        "status": "three-window-callback-fusion-feasibility-passed",
        "programExitCode": 0,
        "case": case,
        "observation": observation,
        "bothEndpointPowerStates": "AWAKE",
        "relativeTimeReduction": 1 - case["computeSeconds"] / case["baselineComputeSeconds"],
        "numericEquivalenceValidated": True,
        "sameRunPairedPerformanceValidated": True,
        "newBestAbsolutePerformanceEstablished": False,
        "productionAppChanged": False,
        "sourceSha256": {name: sha(ROOT / name) for name in (
            "native-bench/VaeSnakeFusion.cpp", "native-bench/VaeVectorSin.hpp", "native-bench/CMakeLists.txt")},
        "verifiedProbeBinarySha256": sha(binary),
        "verifiedImmutableCoefficientSha256": sha(coefficient),
        "sourceLogSha256": sha(REPORTS / (RUN + ".log")),
        "localizationHistory": historical,
        "findings": [
            "Exceptions escaping MNN callbacks skipped runtime exit/cleanup; catch inside callbacks, return CALL_BACK_STOP safely, then report outside the engine",
            "ARM82 FP16 uses C8 physical storage under the NC4HW4 format tag; complete virtual reshape views must resolve to actual storage",
            "Probe must match archive MNN_SUPPORT_QUANT_EXTEND and MNN_USE_THREAD_POOL for CoreFunctions ABI",
            "Skipping original operators invalidates their original buffer reuse; reading residual input late read overwritten coefficient data",
            "Capture the actual executed producer before reuse, tracing through nonexecuted Squeeze/Reshape metadata; one bounded reusable FP16 snapshot restored equivalence",
        ],
        "limits": [
            "Original graph allocations remain; this probe adds at most one 120 MiB input snapshot, included in sampled RSS and fused timing",
            "Compute timing includes run, synchronization/output copy, fused snapshot copy and callback diagnostics, excludes input fixture read/copy and setup/warmup for both paired arms",
            "25.6 valid PCM seconds excludes overlap; 3 windows are not 30.72 seconds of distinct audio",
            "Same-process alternating order gives a local 1.162x speedup; large per-window drift remains, with no controlled temperature/frequency or DRAM-byte measurement",
            "Absolute fused RTF 0.834 is slower than historical standalone W8+NEON best 0.674; not a new best App or whole-chain result",
            "PCM L2 zero is against same-phone W8, not unquantized quality; no listening acceptance or trained music head",
        ],
        "nextAction": "Evaluate explicit fused CPU Execution/Plugin plus graph rewrite to remove snapshot and unused workspaces, then repeat numeric/memory/paired and whole-chain validation",
    }
    path = REPORTS / "levo-test-report.json"
    report = json.loads(path.read_text(encoding="utf8"))
    report["vae_snake_progress_2026_10_30"] = section
    # Preserve the earlier failure record and its original artifact identities.
    assert report["vae_snake_validation_2026_10_30"]["programExitCode"] == 124
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"status": section["status"], "speedup": case["speedupVsPairedBaseline"],
                      "vaeOnlyRtf": case["vaeOnlyRtf"], "wholeChainValidated": False}))


if __name__ == "__main__":
    main()
