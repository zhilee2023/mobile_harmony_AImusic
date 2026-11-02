# 小模型优化保留清单

2026-11-02：用户明确要求记住并保留小模型已有优化。公开预训练大 Mixed LM 的音乐验证应独立进行，不覆盖 0.6B 路线的模型、App 入口、配置、源码和测试报告。清理已经得出结论的实验按钮，不代表删除可复现实现和结果。

同日新增当前串行基线：LeVo Harmony v0.1.8中，0.6B/W4主体＋W8头、22层1024小DiT/W8两步/CFG1.5、同一VAE/W8＋NEON，LM生成750码后销毁再运行Codec，30秒等效PCM耗时44.155秒、RTF1.472，RSS峰值1177.8MiB。仍是未训练音乐适配器/小DiT的数值测试。此基线与旧共驻/十块计算重叠方案分别保留，见CHAIN_COMPARISON.md。

## 可复用基线

| 部分 | 已核实配置与实现 | 使用边界 |
| --- | --- | --- |
| 0.6B 音乐学生原型 | 26 层预训练文本骨干；主体 W4/g64，音乐输出头 W8；CPU Low / 4 线程 | 音乐 embedding、音乐头和风格适配器尚未训练，不能称为已经会生成歌曲 |
| 原始 Qwen 语言模型探针 | 完整 28 层、完整已训练语言头；W4/g64＋KV16，6 线程约 25–28 token/s；512 token prefill 约 1.40–1.58 秒 | 这是独立语言模型测试，不能替代 26 层学生音乐链路数据 |
| 连续分块 | 一次 514 位置 prefill；保留 KV 跨 Codec 暂停恢复；首块 256 新码，后续每块 192 新码 | 五块共 1024 码、40.96 秒有效 PCM，终态 KV 1537；不逐块重新 prefill 全部历史 |
| DiT | 完整 39 层 W8；CPU High / 6 线程；Memory Low、Power High；学生流程两步 Euler | W4 节省权重文件但现有手机前向更慢；两步是流程测速设置，不是生成音质已验收 |
| VAE | W8、CPU Low / 6 线程，ARM82 NEON Sin；只在 VAE 建图时挂接 selector，再立即恢复 | 必须记录实际 kernel 调用次数；不能把全局开关或编译宏视为已经命中优化，也不能改到 LM 的 Sin |
| 会话生命周期 | 小模型五块 App 基线三个模块共驻、依次计算；保持同一 DiT / VAE 会话 | 共驻可行性不能直接外推到公开约 3B 主 Mixed LM＋两个 CFG 上下文 |
| 边界衔接 | 256 帧窗口、64 帧 latent context、192 帧 hop；PCM 有界交叉渐变 | 不累积整曲中间张量；按实际去重后的有效音频秒数计算 RTF |

新手机同包 v0.1.7 App 五块记录：73.356 秒 / 40.96 秒，RTF 1.791，20 ms 采样峰值 RSS 2203.3 MiB。Qwen prefill 1.730 秒，decode 20.780 秒 / 49.23 token/s，DiT 共十步 26.245 秒，VAE 五块 21.673 秒。该记录为随机音乐适配器的数值流程测试，不是训练后歌曲生成或听感验收。

## 保留但不默认启用的实验

- LLM `cosyvoice_kleidiai` 开关：配对测试没有证实 decode 提速；不要默认打开，更不能将开关等同于旧 CosyVoice 的手写 FFN 已迁移。
- CosyVoice 手写 KleidiAI W8A8 i8mm FFN：旧尺寸 1024→2048→1024、22 层；当前 LeVo DiT 为 1200→4800→1200、39 层。保留打包、线程划分、真实 GELU 和输出注入经验，迁移必须重做尺寸、输出误差和原权重重复常驻记账。
- 静态 A4 / A8 和 KV4 探针：已有质量失败结果。A4 的 INT8 存储范围描述不等于原生 packed INT4 kernel；不能将 W4A4KV4 当作当前可用默认配置。
- Snake 回调融合：同轮探针约 1.16 倍、PCM L2=0；仍有输入快照和回调开销，未证明优于历史最好整链，不能直接替换默认 App 路线。
- 小 DiT 调度探针：计算重叠数值 RTF 1.382、峰值约 1.75 GiB；随机未训练、未接 App，严格 L2 未过 1%，不能当成真实音乐性能。
- CANN/NPU：旧芯片 OMC 不能当作新芯片可用模型；单个 Conv 的收益不能替代 VAE 子图或整链测量。保留 CPU 可复现入口。

## 迁移到新模型时必须核对

1. 区分已训练音乐模型、文本骨干和随机音乐适配器；明确参数规模、词表、特殊 token、hidden 和 Codec 合同。
2. 词表裁剪不是删除输入歌词词表。只对已验证的音乐输出头裁剪，并保留输入 embedding 和所有特殊 token 映射。
3. 先确认原图 Linear 是否已经降为 MNN 的常量量化 Convolution。重复 MatMul→Conv 不产生额外收益；需要实际热点和选核证据。
4. 检查 NEON Sin 挂接是否发生在 VAE create/resize 之前，记录调用数、同 token PCM 误差和阶段耗时。
5. 保留 KV、模型预加载、计算重叠分别验证。多线程同时运行可能竞争 CPU；不能仅凭调度重叠推算加速。
6. 统一记录冷启动、prefill、decode、DiT 步数、VAE、输出有效秒数、完整 RTF、峰值 RSS、最低系统可用内存。不同模型和设备不能混用成绩。

## 实现和证据

- `native-bench/ChainBench.cpp`、`VaeVectorSin.hpp`、`VaeSnakeRuntime.hpp`：小模型连续链路、生命周期和 VAE 优化。
- `tools/export_diagnostic_student.py`、`export_qwen_language_probe.py`：学生与完整语言模型分别导出。
- `docs/DEVICE_CPU_SPEED.md`、`VAE_OPTIMIZATION.md`、`KERNEL_REUSE_REVIEW.md`：可复用基线与未迁移部分。
- `docs/QWEN_LOW_BIT_LANGUAGE_TEST.md`、`SMALL_DIT_SCHEDULING.md`：量化和调度实验的实际边界。
- `reports/levo-test-report.json`：保留所有历史结论的统一报告，不用单层微基准替代整链结果。
