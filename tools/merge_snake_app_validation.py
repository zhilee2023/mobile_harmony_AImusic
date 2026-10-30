"""Publish App integration and retain the recommended W8+NEON entry."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"reports"

def load(name):return json.loads((R/name).read_text(encoding="utf-8-sig"))
def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()

def main():
    fusion=load("signed-five-snake-v017-r2.json")["chunks5W8Snake"]
    baseline=load("signed-five-vsin-v017-r1.json")["chunks5W8Vsin"]
    receipts={key:load("signed-"+run+"-receipt.json") for key,run in {
        "fusion":"five-snake-v017-r2","baseline":"five-vsin-v017-r1"}.items()}
    for case in (fusion,baseline):
        assert case["success"] and case["flowAndMemoryPassed"]
        assert case["generatedCodecIds"]==1024 and case["finalKvPositions"]==1537
        assert len(case["chunks"])==5 and case["validPcmSeconds"]==40.96
        assert math.isclose(case["pipelineRtfThroughPcm"],case["pipelineSecondsThroughPcm"]/40.96)
        assert case["music_trained"] is False and case["real_song_generated"] is False
    stats=fusion["vaeSnakeFusion"]
    assert stats["fusedCalls"]==180 and stats["completedForwards"]==5
    assert stats["firstRealInputParityPassed"] and stats["firstRealInputPcmL2"]==0
    assert stats["producerSnapshotPeakBytes"]==120*1024**2
    assert baseline["vaeVectorSinKernelCalls"]==1080
    assert receipts["fusion"]["hapSha256"]==receipts["baseline"]["hapSha256"]
    assert receipts["fusion"]["chainSourceSha256"]==sha(ROOT/"native-bench/ChainBench.cpp")
    for run in ("five-snake-v017-r2","five-vsin-v017-r1"):
        for suffix in ("-before.log","-after.log"):
            assert "Current State: AWAKE" in (R/("signed-"+run+suffix)).read_text(encoding="utf-8-sig")
    failed=load("signed-five-snake-v017-r1.json")["chunks5W8Snake"]
    assert not failed["success"] and "parity failed" in failed["error"]
    packages=load("app-v017-package-identities.json")
    tested=packages["LeVo-v0.1.7-snake-experimental-signed-r2-local.hap"]
    installed=packages["LeVo-v0.1.7-original-w8-neon-signed-local.hap"]
    assert tested["hapSha256"]==receipts["fusion"]["hapSha256"]
    assert tested["nativePayloadSha256"]==installed["nativePayloadSha256"]
    assert installed["hapSha256"]==sha(ROOT/"dist/LeVo-v0.1.7-original-w8-neon-signed-local.hap")
    assert "install bundle successfully" in (R/"app-v017-original-ui-install.log").read_text(encoding="utf-8-sig")
    tests=(R/"unit-tests-v017-20261030.log").read_text(encoding="utf-8-sig")
    assert "Ran 15 tests" in tests and tests.rstrip().endswith("OK")
    path=R/"levo-test-report.json";report=json.loads(path.read_text(encoding="utf8"))
    report["version"]="0.1.7-original-entry-retained"
    report["vae_snake_app_validation_2026_10_30"]={
        "asOfUtc":datetime.now(timezone.utc).isoformat(),"status":"flow-validated-no-whole-chain-speedup",
        "fusion":fusion,"samePackageOriginalW8Neon":baseline,"receipts":receipts,
        "bothRunsEndpointPowerStates":"AWAKE","experimentOrder":"fusion then original W8+NEON; one sequential comparison, no thermal/frequency control",
        "firstRealInputReferenceForwardCostIncluded":True,
        "failedFirstAttempt":{"error":failed["error"],"totalSeconds":failed["totalSeconds"],
            "sourceReportSha256":sha(R/"signed-five-snake-v017-r1.json"),
            "receipt":load("signed-five-snake-v017-r1-receipt.json"),
            "resolution":"Reupload all VAE inputs between the reference and fused forward in the same session; followup first real-input PCM L2 zero"},
        "decision":{"recommendedPolicy":"chunks-5-w8-vsin","experimentalFusionUiVisible":False,
            "humanSteering":"Keep the original entry","fusionBackendRetainedForReproduction":True,
            "reason":"Measured fused App path slower and uses an extra producer snapshot; no promotion to recommended generation"},
        "installedUiOnlyPackage":installed,"testedNativePayloadUnchangedAfterUiCleanup":True,
        "sharedRuntimeSourceSha256":sha(ROOT/"native-bench/VaeSnakeRuntime.hpp"),
        "regressionTestsPassed":15,
        "limits":["First real App PCM window was paired against original W8; remaining four checked finite, not independently reference-paired",
            "Reference validation cost 6.378 seconds included in fused total; do not subtract it and advertise a new measured App RTF",
            "Original graph workspace retained, extra 120 MiB snapshot; no graph-level memory saving proven",
            "DiT remains original W8 CPU High/two steps; strict same-quant 1% parity still false and reported separately",
            "Random/untrained music adapters; finite diagnostic PCM is not a song or listening quality acceptance"]}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"fusionRtf":fusion["pipelineRtfThroughPcm"],"recommendedRtf":baseline["pipelineRtfThroughPcm"],
                      "recommended":"chunks-5-w8-vsin","fusionUiRemoved":True}))

if __name__=="__main__":main()
