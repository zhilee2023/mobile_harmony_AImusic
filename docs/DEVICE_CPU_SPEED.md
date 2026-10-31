# 新手机五块 CPU 速度对照（2026-10-31）

新手机系统自报 HUAWEI Mate 90 Pro Max、CMM-AL00、Kirin9050，14 个逻辑 CPU；OpenHarmony-7.0.0.125，`MemTotal=11670328 kB`（约 11.13 GiB）。型号和硬件字符串来自手机，不是根据公开规格推断。旧机对照为 2026-10-30 同包 App 基线。

用户要求安装 App 后，实际通过 HDC 安装了原签名 v0.1.7，并在 App 内运行；没有用命令行成绩替代 App 成绩。HAP SHA256 为 `b60e3382a055d10addc5aea669002c89f42cb281003ae0ff858ff7c6de9bdff3`。模型/夹具 22 个文件均在 App 私有目录重新核对 SHA256。启动、运行中及结束时亮屏；系统息屏时长由用户改为 600 秒。

## 配置和计时口径

- Qwen3-0.6B：26 层预训练骨干，主体 W4/g64，音乐头 W8；音乐适配器和音乐头仍未训练。CPU Low、4 线程。
- DiT：完整 39 层 W8，CPU High、6 线程、Memory Low、Power High，每块两步 Euler。基础 CPU 设置与 CosyVoice 路线一致；专用 KleidiAI FFN 尚未迁移，不能称为已经使用旧项目的手写内核。
- VAE：原 W8 权重，CPU Low、6 线程、NEON Sin；三个模块全程共驻，依次计算。
- 一次 514 位置 prefill，保留 KV，五个 256 帧窗口、64 帧重叠，生成 1024 个新码；最终 KV 1537 位置。
- 有效 PCM 为 40.96 秒。总耗时包含加载初始化、prefill、decode、DiT、VAE、衔接；销毁后的参考误差测试不计入 RTF。没有扬声器播放、WAV 保存或歌曲音质验收。

## 实测

| 项目 | 旧机同包基线 | 新机本轮 |
|---|---:|---:|
| 初始化 | 2.088 秒 | 2.442 秒 |
| Qwen prefill | 2.640 秒 | 1.730 秒 |
| Qwen decode | 21.900 秒 | 20.780 秒 |
| decode 吞吐 | 46.71 token/s | 49.23 token/s |
| DiT 五块、共十步 | 36.495 秒 | 26.245 秒 |
| VAE 五块 | 30.189 秒 | 21.673 秒 |
| 总耗时 | 93.799 秒 | 73.356 秒 |
| 整链 RTF | 2.290 | 1.791 |
| 首块数值 PCM 就绪 | 21.367 秒 | 17.361 秒 |
| 20 ms 采样峰值 RSS | 2246.5 MiB | 2203.3 MiB |

新机总耗时减少 21.79%，约 1.279 倍吞吐；DiT、VAE 时间各减少约 28%。初始化没有加速，decode 仅减少约 5%，不能概括为所有阶段均等提速。两台手机各是一轮记录，未控制温度、后台任务和调度，不能把全部差异归因于芯片。

五块均完成、输出有限，最低系统可用内存 3061 MiB。DiT 对 PC 同量化参考的 velocity L2 为 0.02645815（2.646%），与旧机记录相同，仍不满足 1% 严格数值门槛；流程/内存通过与数值验收分开。

统一数据位于 `reports/levo-test-report.json` 的 `device_cpu_speed_2026_10_31`，含逐块耗时、设备事实、文件哈希、安装包凭据和完整结果。本地原始单次报告保留，不单独上传。

复现：安装相同签名包、将同一模型放到 App 的 `files/three-module` 并保持亮屏，然后运行：

```powershell
.\tools\run_signed_chain.ps1 -Policy chunks-5-w8-vsin -RunId unique-device-five-run-id
```

`tools/merge_device_speed_validation.py` 在验证五块、码数、配置、包哈希和设备 run ID 后归并报告；输入设备证据应使用对应 run ID 的本地 `*-device.json`。

## 追加 W4/W8 DiT 对照

用户随后询问 DiT 四比特。已在同一亮屏手机运行 W8、W4、W4、W8 两轮反序对照；完整 39 层、相同固定输入、CPU High/6线程、Memory Low/Power High，均先预热一次，再对一次带回调的前向计时。这里是 CLI Debug/callback 模块探针，不能代替 App 五块 W4 整链 RTF。

| 轮次 | W8 预热后前向 | W4 预热后前向 |
|---|---:|---:|
| 第一轮 | 2.812 秒 | 4.378 秒 |
| 逆序第二轮 | 3.147 秒 | 4.097 秒 |
| 中位数 | 2.980 秒 | 4.237 秒 |

W4 中位耗时约为 W8 的 1.422 倍，当前保留 W8 速度路线。W8 外置权重 722214236 字节，W4 为 462169752 字节，减少 36.0%；由于存在混合保留部分、缩放与对齐，不是整个文件恰好减半。本项未采样进程 RSS，不能把文件差额直接当成实测运行内存节省。

W8/W4 对各自 PC 同量化参考的 velocity L2 为 2.646%/2.909%，不是 W4 相对 W8 的量化音质误差；两者输出有限。四次原始结果和分类热点已归入同一报告的 `ditW4W8Comparison`，无需再上传分日志。

新手机 `/proc/cpuinfo` 包含 `i8mm`。它是 CPU 的 8 位矩阵乘加指令扩展；Arm 官方 `vmmlaq_s32` 映射 `SMMLA`，输入 INT8、累加 INT32。参考 [Arm Neon Intrinsics Reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#matrix-multiply)。支持该特性不等于本探针已经逐算子确认选中相应内核；W4 的解包/缩放及 MNN 选核仍需独立定位，不能把本轮慢 42% 全归因于其中某一步。
