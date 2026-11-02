"""Merge the bounded phone A-B-A test into the one public report."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = json.loads(args.archive.read_text(encoding="utf-8-sig"))
    result = archive["result"]
    rows = result["rows"]
    assert result["success"] and [row["phase"] for row in rows] == [1, 2, 3]
    assert [row["animationEnabled"] for row in rows] == [True, False, True]
    for row in rows:
        assert row["historyAfterPrefill"] == 953 and row["historyAfter"] == 1081
        assert len(row["tokenSeconds"]) == 128 and row["maxLogitsL2VsA1"] < 1e-4
    on_seconds = (rows[0]["decodeSeconds"] + rows[2]["decodeSeconds"]) / 2
    off_seconds = rows[1]["decodeSeconds"]
    root = Path(__file__).resolve().parents[1]
    target = root / "reports/levo-test-report.json"
    report = json.loads(target.read_text(encoding="utf-8-sig"))
    report["music_app_ui_aba_2026_11_02"] = {
        **archive,
        "analysis": {
            "decodeTimeReductionVsMeanOnPercent": 100 * (1 - off_seconds / on_seconds),
            "decodeThroughputGainVsMeanOnPercent": 100 * (on_seconds / off_seconds - 1),
            "peakRssMiB": result["sampledPeakRssKiB"] / 1024,
            "minimumSystemAvailableMiB": result["minimumSystemAvailableKiB"] / 1024,
            "conclusion": "One A-B-A sequence suggests a modest effect; it does not establish animation as the principal cause of the previous whole-song slowdown.",
            "limitations": [
                "128 teacher-forced decode calls per phase, not a 750-token song or whole-chain RTF.",
                "One loaded LLM and identical inputs; full 953-position plus eight-decode warmup before A1.",
                "MNN removes the old KV on the next forward; each measured prefill ends at953 and each phase ends at1081.",
                "CPU frequency samples readable; thermal sysfs unreadable in the App, no measured CPU temperature.",
                "Phase polling and 20ms memory sampling remain active in B. No claim of eliminating every UI/system cost.",
                "Sequential order cannot eliminate thermal/scheduler drift. A2 differs from A1.",
                "RSS is process memory, not dedicated GPU VRAM. This benchmark uses CPU inference.",
            ],
        },
    }
    report["music_app_v105_features_2026_11_02"] = {
        "version": "1.0.5",
        "signedBuildPassed": True,
        "installedOnPhone": True,
        "screenshotReviewed": True,
        "originalLyricSamples": 20,
        "randomButtonExcludesCurrentLyrics": True,
        "randomButtonPreservesStyle": True,
        "durationOptionsSeconds": [15, 30, 60, 90, 120],
        "maxMusicTokens": 3000,
        "maxCodecChunks": 16,
        "codecWindowFrames": 256,
        "codecHopFrames": 192,
        "longDurationStatus": "Options and bounded native path added; 60/90/120-second full phone generation NOT yet validated. EOS can stop early.",
        "defaultDitSteps": 2,
        "ditCfg": 1.5,
        "lmCfg": 1,
        "previousExperimentsPreserved": True,
    }
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["music_app_ui_aba_2026_11_02"]["analysis"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
