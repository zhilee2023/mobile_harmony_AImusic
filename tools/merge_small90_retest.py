"""Append a matched-input small-chain/VAE replay experiment without replacing history."""
import json
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', default='small90-retest-20261102-r3')
    parser.add_argument('--replay', default='small90-vae-replay-20261102-r2')
    args = parser.parse_args()
    reports = ROOT / "reports"
    chain = read(reports / f"comparison-app-{args.chain}.json")
    replay = read(reports / f"comparison-app-{args.replay}.json")
    old = read(reports / "comparison-app-small-90-foreground-20261102-r1.json")
    a, b = chain["result"], replay["result"]
    for run in (chain, replay):
        assert run["foregroundStable"] and run["displayStable"] and run["result"]["success"]
    assert chain["hapSha256"] == replay["hapSha256"]
    assert a["durationSeconds"] == b["durationSeconds"] == 90
    assert a["lm"]["frames"] == 2250 and a["lm"]["finalKvPositions"] == 2763
    assert a["diagnosticOnly"] and not a["musicTrained"]
    assert len(a["codec"]["chunks"]) == len(b["windows"]) == 12
    assert a["codec"]["vaeReplayManifest"].endswith("/vae-replay/latest.json")
    assert b["allRawPcmFingerprintsMatch"] and b["firstWindowRelativeL2"] == 0
    assert not b["llmExecuted"] and not b["ditExecuted"] and not b["wholeChainCompleted"]
    assert abs(b["chainVaeSeconds"] - a["stageSeconds"]["vae"]) < 1e-7
    sensors = {}
    for run in (chain, replay):
        file = reports / f'power-thermal-{run["runId"]}.jsonl'
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        sensors[run["runId"]] = rows
    comparison = {
        "oldWholeChainRtf": old["result"]["stageRtf"]["total"],
        "retestWholeChainRtf": a["stageRtf"]["total"],
        "oldStageSeconds": old["result"]["stageSeconds"],
        "retestStageSeconds": a["stageSeconds"],
        "matchedChainVaeSeconds": a["stageSeconds"]["vae"],
        "matchedFreshVaeSeconds": b["vaeSeconds"],
        "chainToFreshVaeTimeRatio": a["stageSeconds"]["vae"] / b["vaeSeconds"],
        "allRawOutputFingerprintsMatch": b["allRawPcmFingerprintsMatch"],
        "firstRawOutputRelativeL2": b["firstWindowRelativeL2"],
    }
    record = {
        "date": "2026-11-02", "appVersion": chain["appVersion"],
        "scope": "Untrained small music model; full 90s numerical chain and fresh-process VAE replay using exactly its captured inputs",
        "chain": chain, "freshVaeReplay": replay, "comparison": comparison,
        "earlierCompletedRetest": read(reports / "comparison-app-small90-retest-20261102-r2.json"),
        "earlierReplaySetupFailure": {"error": "Missing ARM82 unary selector", "forwardExecuted": False,
                                      "fix": "Initialize deferred CPU backend before installing the VAE sine hook; rerun the chain and replay using the same rebuilt HAP"},
        "sensors": sensors,
        "limitations": [
            "One sequential pair; thermal/cache/scheduling conditions are observed, not held identical",
            "Fresh VAE process has no DiT resident, unlike the chain; differences are not attributable solely to earlier LLM execution",
            "Only the first raw PCM window has a saved full reference and direct L2 comparison; all 12 windows have bit-pattern fingerprints (not cryptographic hashes)",
            "Captured tensor writes/fingerprints are included in full-chain other time, excluded from VAE run/read time",
            "SoC thermal sensor is not case temperature; USB-connected battery-terminal V*I is not isolated App or whole-device power",
            "RSS/PSS are shared physical process memory, not dedicated VRAM; sampled peaks may miss transients",
        ],
    }
    local = reports / 'local' / 'small90-retest'
    if (local / 'vae-model-sha256.txt').exists():
        hashes = [line.split()[0] for line in (local / 'vae-model-sha256.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        assert len(hashes) == 3 and hashes[1] == hashes[2]
        record['modelSha256'] = {'vaeGraph': hashes[0], 'smallVaeWeight': hashes[1], 'publicVaeWeight': hashes[2], 'weightsIdentical': True}
        fixture = local / 'fixture'
        tensors = list(fixture.glob('latent-*.f32'))
        assert len(tensors) == 12 and all(p.stat().st_size == 64 * 256 * 4 for p in tensors)
        record['capturedTensorSha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tensors)}
    file = reports / "levo-test-report.json"
    archive = read(file)
    archive["small90_matched_vae_retest_2026_11_02"] = record
    file.write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    earlier = record['earlierCompletedRetest']['result']
    rows = '\n'.join(
        f"| {label} | {old['result']['stageSeconds'][key]:.3f} | {earlier['stageSeconds'][key]:.3f} | {a['stageSeconds'][key]:.3f} | {a['stageRtf'][key]:.3f} |"
        for label, key in [('LLM（含加载/释放）', 'llm'), ('DiT', 'dit'), ('VAE', 'vae'), ('其他', 'other'), ('总计', 'total')]
    )
    windows = '\n'.join(f"| {w['chunk']} | {w['chainVaeSeconds']:.3f} | {w['vaeSeconds']:.3f} | 一致 |" for w in b['windows'])
    text = f'''# 小模型90秒重测与同输入VAE对照（2026-11-02）

两次重测整链RTF为 **{earlier['stageRtf']['total']:.3f}、{a['stageRtf']['total']:.3f}**，此前2.163未复现。小模型音乐头和小DiT仍未训练，只验数值链路、耗时和内存，不验歌曲质量。

## 配置与口径

- 签名LeVo Harmony v{chain['appVersion']}，同一手机；90秒等效PCM、2250个新code、12个256帧窗口，hop192/上下文64。EOS为定长诊断关闭，前缀514，最终KV2763。
- LLM：598.178M学生，主体W4/group64、头W8、输入表BF16，CPU Low/4线程；DiT：195.416M，W8/group64、CPU High/请求6线程、两步、CFG1.5；VAE：84.414M，W8、CPU Low/请求6线程、NEON正弦，KleidiAI关闭。
- 内存Low、Power High；没有请求静态A4或KV4。三个阶段串行；LM销毁后，唯一DiT和VAE会话共同保留至全部窗口完成。90秒PCM数值已生成，但写入/dev/null。
- 两次完整重测及VAE重放均通过全程前台/未息屏检查。配对的第二次整链和VAE重放HAP SHA256均为`{chain['hapSha256']}`。传感器从启动阶段开始采样。

## 整链实测（秒）

| 阶段 | 旧90秒 | 重测1 | 重测2／配对基线 | 重测2阶段RTF |
|---|---:|---:|---:|---:|
{rows}

重测2 prefill {a['lm']['prefillSeconds']:.3f}秒，decode {a['lm']['decodeSeconds']:.3f}秒，decode吞吐{a['lm']['decodeTokensPerSecond']:.2f} token/s。RSS峰值{a['memory']['peakRssKiB']/1024:.1f}MiB、PSS采样峰值{a['memory']['sampledPeakPssKiB']/1024:.1f}MiB；图形内存API返回0KiB。这是CPU进程共享物理内存，不是独立显存，也不代表整机GPU没有占用。

## 同一批latent的VAE重放

整链保存每窗实际VAE输入，随后冷启动App，只加载同一VAE，按相同顺序重放12窗。权重没有改变，小/公开两路VAE权重SHA256一致。首窗保存完整原始PCM参考，全部窗口保存交叠混音前的位模式指纹。

| 指标 | 整链中VAE | 新进程独立VAE |
|---|---:|---:|
| 12窗前向＋读回合计 | {a['stageSeconds']['vae']:.3f}秒 | {b['vaeSeconds']:.3f}秒 |
| 按90秒有效音频折算计算RTF | {a['stageRtf']['vae']:.3f} | {b['vaeOnlyComputeRtf']:.3f} |
| 每窗平均 | {a['stageSeconds']['vae']/12:.3f}秒 | {b['vaeSeconds']/12:.3f}秒 |
| 该进程RSS采样峰值 | {a['memory']['peakRssKiB']/1024:.1f}MiB（整链峰值） | {b['memory']['peakRssKiB']/1024:.1f}MiB |

独立VAE耗时少{(1-b['vaeSeconds']/a['stageSeconds']['vae'])*100:.1f}%，不是此前不同输入/进程下约1.7倍的差距。全部12窗输出指纹一致，首窗直接相对L2=0。指纹只是非加密位模式校验，不替代所有窗口的逐元素L2。独立初始化另需{b['setupSeconds']:.3f}秒，测试总墙钟{b['totalSeconds']:.3f}秒；该成绩不能标成整链RTF。

| 窗口 | 整链VAE秒 | 独立VAE秒 | 原始PCM指纹 |
|---|---:|---:|---|
{windows}

## 解释与下一步

1. 相同配方下完整链路两次140.98–144.71秒，旧194.71秒没有复现；不能把旧慢速归因成“小模型VAE更大”或量化本身。
2. 固定输入后，独立VAE仍略快；它没有DiT共驻，温度/频率/缓存也没有锁定，因此目前不能将全部差异只归因于此前运行过LLM。
3. 当前MNN的`ThreadPool::init`按CPU mask复用既有线程池，既有线程池小于新请求时返回旧上限；`CPURuntime::_resetThreadPool`会据此收缩实际线程数。先LLM请求4、后Codec请求6，不保证后者实际有6线程。同轮进程线程快照看到整链只有3个高CPU辅助线程，而独立VAE有5个；工作线程允许CPU列表为2–13，并非锁定最快核。下一轮优先验证先初始化6线程池、LM仍使用4线程的整链对照；本轮没有改线程池、绑核或kernel，不能宣称已验证该优化。
4. 12份latent约768KiB、首窗参考PCM3.75MiB已保留在本机`reports/local/small90-retest`和手机测试目录，不上传权重或音频。捕获/指纹成本{a['codec']['vaeCaptureSeconds']:.3f}秒包含在整链“其他”中，不计入单窗VAE前向时间。
5. 独立VAE首次启动曾在创建会话前安装正弦钩子失败，未执行前向。已调整为后端创建后、延迟resize前安装，并使用同一重建HAP重新跑整链及重放；失败记录保留，不计性能。

统一数据：`reports/levo-test-report.json` → `small90_matched_vae_retest_2026_11_02`。此前40个记录区块原样保留，含原始90秒、公开恢复和120秒未完成记录。USB供电下电池端电压×电流不等于App或整机功耗，内部温度也不是机壳温度；本轮没有据此归因热降频。
'''
    (ROOT / 'docs' / 'SMALL_90S_RETEST.md').write_text(text, encoding='utf-8')
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
