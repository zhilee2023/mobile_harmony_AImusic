"""Plot measured internal temperatures and battery-terminal |VI|; never label it App power."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def read_samples(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        raw = row["rawReadings"]
        if len(raw) != 10:
            continue
        try:
            values = [float(v) for v in raw[:8]]
            stage = json.loads(row["progress"])["stage"]
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        rows.append({"time": row["observerElapsedSeconds"], "socC": values[0] / 1000,
                     "batteryC": values[1] / 10, "batteryVoltageV": values[2] / 1e6,
                     "batteryCurrentMa": values[3], "batteryExchangeMagnitudeW": abs(values[2] * values[3]) / 1e9,
                     "vendorSysPowerRaw": values[4], "vendorChargeLossRaw": values[5],
                     "policy2GHz": values[6] / 1e6, "policy3GHz": values[7] / 1e6,
                     "batteryStatus": raw[8], "usbOnline": raw[9], "stage": stage})
    assert len(rows) >= 2, "Insufficient valid sensor samples"
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    cases = [("0.6B 原型 · 90 秒负载", args.small)]
    if args.large:
        cases.append(("公开 Mixed · 90 秒 LLM 段（随后退出）", args.large))
    if args.recovery:
        cases.append(("公开 Mixed · 90 秒 Codec 恢复段", args.recovery))
    fig, axes = plt.subplots(3, len(cases), figsize=(7 * len(cases), 9), squeeze=False, layout="constrained")
    summary = {}
    for column, (name, path) in enumerate(cases):
        rows = read_samples(path)
        time = np.array([r["time"] for r in rows])
        for ax in axes[:, column]:
            ax.grid(alpha=.18)
            ax.set_xlim(time.min(), time.max())
        soc = np.array([r["socC"] for r in rows])
        battery = np.array([r["batteryC"] for r in rows])
        power = np.array([r["batteryExchangeMagnitudeW"] for r in rows])
        axes[0, column].plot(time, soc, label="SoC 节点读数", color="#ba5834", linewidth=1.8)
        axes[0, column].plot(time, battery, label="电池温度", color="#167c83", linewidth=1.8)
        axes[0, column].set(title=name, ylabel="传感器温度（°C）")
        axes[0, column].legend(loc="best")
        first_progress = json.loads(json.loads(path.read_text(encoding="utf-8-sig").splitlines()[0])["progress"])
        if np.ptp(soc) == 0:
            axes[0, column].text(.02, .68, f"SoC 节点全程返回 {soc[0]:g}°C\n不代表精确结温或机壳温度", transform=axes[0, column].transAxes,
                                 fontsize=9, color="#795b4b")
        axes[1, column].plot(time, power, color="#5f58a6", label="电池端 |V×I|", linewidth=1.7)
        axes[1, column].set(ylabel="电池端电功率幅值（W）")
        axes[1, column].legend(loc="best")
        axes[2, column].plot(time, [r["policy2GHz"] for r in rows], color="#237c69", label="CPU policy2", linewidth=1.5)
        axes[2, column].plot(time, [r["policy3GHz"] for r in rows], color="#b18422", label="CPU policy3", linewidth=1.5)
        axes[2, column].set(ylabel="CPU 频率（GHz）", xlabel="传感器采样开始后的时间（s）")
        axes[2, column].legend(loc="best")
        completion = next((r for r in rows if r["stage"] == "real-phone-music-codes-2250"), None)
        if "LLM" in name and completion:
            boundary = completion["time"]
            for ax in axes[:, column]:
                ax.axvspan(boundary, time.max(), color="#999999", alpha=.14)
                ax.axvline(boundary, color="#666666", linestyle="--", linewidth=1)
            axes[1, column].text(.62, .74, "灰区：2250 码已齐后\n释放／等待，随后退出\n未计入 6.225 的计算段合计", transform=axes[1, column].transAxes, fontsize=8, color="#555555")
        summary[name] = {"sampleCount": len(rows), "sampleSpanSeconds": float(time.max() - time.min()),
                         "socTemperatureRangeC": [float(soc.min()), float(soc.max())],
                         "batteryTemperatureRangeC": [float(battery.min()), float(battery.max())],
                         "batteryExchangeMagnitudeRangeW": [float(power.min()), float(power.max())],
                         "batteryExchangeMagnitudeMeanW": float(power.mean()), "usbChargingObserved": any(r["usbOnline"] == "1" for r in rows),
                         "firstProgress": first_progress,"first2250CodeConfirmationSeconds":completion["time"] if completion else None,"samples": rows}
    fig.suptitle("LeVo Harmony 真机测量：内部温度、电池端功率与频率", fontsize=16, fontweight="bold")
    fig.supxlabel("USB 充电；电池端 |VI| 不是 App、SoC 或整机功耗。各列独立录制，约 5 秒采样；未连接恢复间隔。\n采样起点晚于模型加载，尤其小模型仅覆盖 LLM 末尾及 Codec；可能漏采峰值。温度不是机壳表面温度。", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(args.out.with_suffix("." + ext), dpi=180)
        if ext == "svg":
            svg = args.out.with_suffix(".svg")
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    summary["measurementLimits"] = {"powerKind": "Battery-terminal |voltage x current| magnitude during USB charging, not App/SoC/whole-device power",
                                    "vendorSysPowerUnit": "Unverified; raw values retained and never converted to watts",
                                    "temperatureKind": "Internal thermal zone and battery sensors, not case surface temperature",
                                    "batteryUnitReference": "https://github.com/openharmony/docs/blob/master/en/application-dev/reference/apis-basic-services-kit/js-apis-battery-info.md",
                                    "thermalReference": "https://docs.kernel.org/driver-api/thermal/sysfs-api.html"}
    args.out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {a:b for a,b in v.items() if a != "samples"} for k,v in summary.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
