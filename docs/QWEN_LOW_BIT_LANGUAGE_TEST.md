# Qwen 语言模型 W4 / A4 / KV4 与两阶段并行测试

日期：2026-10-31。真机自报 Kirin9050，MNN 源码 d407447ed56c4121a11ccbd266dc184ca1ead0c2（3.6.1 加既有工程补丁）。使用独立 CLI，没有替换签名 App。

## 结论

标准 W4/g64、KV16 可用；6 线程在这次测试中较合适。实验性的 A4 数值范围＋KV4 已执行，但数值质量失败，不能作为可部署的 W4A4KV4 或提速结果。MNN 有 prefill/decode 不同内核与并行路径；当前 LLM 公共配置中的 `thread_num` 仍由两阶段共用。

本轮使用原始 **28 层 Qwen3-0.6B 与完整、已训练的 151936 语言词表头**，不是先前 26 层骨干加随机音乐头。正文和语言头都导出为 W4/g64，embedding 为 BF16。固定 512-token 合成故事背景，随后对同一份 FP32 参考生成的 128 个 token 做 teacher forcing：一次 prefill 预测第一个 token，127 次增量 decode。额外生成至多 48 个贪心 token 用于查看文本，不计入下面两阶段耗时。

## 真机数据

每个用例先冷启动模型，再在同一会话内 reset/KV 清空后重复一次；下表为第二轮。读取模型另计，20ms 采样 RSS。所有列出的语言测试均核对前后亮屏状态和 Last Screen Off 无变化。线程/温度/系统调度仍有波动，不能把一次运行的微小差异当稳定收益。

| 路线 | 线程 | 512-token prefill | decode token/s | 参考 token 一致率 | 采样峰值 RSS |
|---|---:|---:|---:|---:|---:|
| 标准 W4＋KV16 | 4 | 1.795s | 24.35 | 91.41% | 547.6MiB |
| 标准 W4＋KV16，第1次 | 6 | 1.402s | 28.10 | 91.41% | 534.3MiB |
| 标准 W4＋KV16，回测 | 6 | 1.542s | 24.70 | 91.41% | 547.3MiB |
| 标准 W4＋KV4 | 4 | 5.544s | 20.91 | 49.22% | 550.5MiB |
| 标准 W4＋KV4 | 6 | 4.321s | 21.90 | 49.22% | 549.7MiB |
| 标准 W4＋KV16，dynamic_option=8 | 6 | 1.818s | 19.09 | 91.41% | 547.2MiB |
| SmoothQuant W4／A4范围＋KV4 | 6 | 4.481s | 22.27 | **0%** | 562.9MiB |
| SmoothQuant W4／A4范围＋KV16 | 6 | 1.660s | 20.28 | **0%** | 573.9MiB |
| SmoothQuant W4／A8＋KV16 | 6 | 2.388s | 22.82 | **0%** | 564.1MiB |
| 同一 SmoothQuant W4 权重，移除静态激活元数据 | 6 | 1.468s | 25.42 | **94.53%** | 550.3MiB |

CPU High 下的 A8＋KV16、A4范围＋KV4 两组都在首次前向检测到非有限 logit，安全停止，统一报告保留失败。它们没有有效速度或质量结果。

一致率指同一参考历史下逐位置 argmax 是否相同，不是整句生成相同率，也不是一般语言能力准确率。为了固定运行长度，teacher forcing 保留 EOS 后的参考 token，不能将这段数据当语言困惑度测试集。峰值包括加载/首轮；KV4 的常驻 KV 更小，不保证整个短测试的采样峰值一定降低。

## A4 到底执行了什么

1. 使用 MNN 官方 SmoothQuant 实现与 20 条本地原创校准文本，显式传入实验性的 `act_bit=4`。官方命令帮助只公开 A8/A16；这不是官方保证支持的 A4 部署配方。
2. 产物包含 308 个静态激活描述：`min=-7,max=7,type=DT_INT8`。A4/A8 两份 **W4 权重 SHA 完全相同**，区别是激活量化描述。
3. 单独调试前向观察到 616 次四位数值范围输入描述，以及 `FloatToInt8 [DT_INT8]`、`Convolution [DT_INT8]`、`Int8ToFloat` 包装。回调逻辑 tensor 仍显示 float32，不能据此声称物理激活已经以两值一字节打包。
4. **不是原生 packed INT4×INT4 GEMM**；语言头的激活也被官方静态量化实现排除。不能将其命名为全网络原生 W4A4KV4 的加速完成版。
5. 同样的 smoothed W4 权重，移除 308 个静态激活描述后，PC 一致率恢复 93.75%、手机恢复 94.53%。因此当前问题集中在这套静态激活图/执行组合，不能仅归因于 A4 位宽。尚未定位到具体首个失真算子，也没有证明其他导出方式或内核无法支持 A4。

下一步应先做逐层首个失真定位，以相同权重对照静态量化开关，再修复导出/内核或使用逐投影 QDQ 数值参考。通过语言回归后才评价速度；不将失败方案接入 App。

## Prefill / decode 的并行区别与 MNN 接口

- Prefill：一次处理多个新 token，线性层主要是 GEMM；按 token tile、输出通道和 attention query/head 分工，权重可跨 token 复用。
- Decode：每步通常只有一个新 token，线性层是 GEMV 或 M=1 GEMM；可按输出通道、head、历史 KV 分工。同一步可以多线程，依赖前一步结果的未来 token 仍要顺序生成。
- `Llm::forwardRaw` 的 module pool 以 prefill key=100、decode seq_len=1 分开选择；`ConvInt8TiledExecutor` 根据 `planeSize==1`、权重格式和硬件选择路径，所以并不是全部走同一个矩阵计算核。
- 该版本 LLM 的 prefill/decode 共享 `RuntimeManager` 与 `thread_num`。`set_config` 只更新 JSON 等状态，load 后改线程字段不会自动重建已有 runtime；本轮没有伪造“prefill 6线程、decode 4线程已切换”。
- `dynamic_option`、`attention_mode` 可调；`cpu_sme2_neon_division_ratio` 能区分 SME/NEON 的两阶段负载比例，但手机没有报告 SME2，不能把此选项当通用 NEON 加速开关。`dynamic_option=8` 本轮无提速证据，保持关闭。
- `attention_mode=8` 为默认 FlashAttention＋未量化 KV；`14` 为 FlashAttention＋TQ4 KV。KV4 与 A4 是不同设置，不能通过设置一个字段同时开启。

## QuaRot 研究的边界

此前已在真实 Qwen 权重的 6 个投影上，用 20 条校准＋8 条留出输入做 QuaRot 风格正交旋转对照。W4A4 投影误差有改善，例如第0层 FFN down 从 46.3% 到 13.8%，第12层 Q 从 34.3% 到 12.6%。这是 26 层骨干的投影数值模拟，未实施完整 residual/QK 旋转，也不是这里整张语言图的部署或 ARM INT4 提速。

## 复现与报告

- `tools/export_qwen_language_probe.py`：独立导出原始语言 W4、实验 Smooth A4/A8，20 条校准；保留完成的模型，不覆盖。
- `tools/prepare_qwen_language_fixture.py`：生成固定 FP32 文本参考。
- `native-bench/QwenLanguageBench.cpp`、`tools/run_qwen_language_probe.ps1`：CLI 两轮前向、逐组峰值、前后亮屏与模型/夹具哈希检查。`-Precision high` 用于失败定位。
- `native-bench/QwenActivationAudit.cpp`：单独审计精度描述，调试开销不计入性能表。
- `tools/diagnose_qwen_static_quant.py`、`tools/prepare_qwen_noact_ablation.py`：相同权重移除激活元数据的 PC/手机消融。
- `tools/merge_scheduling_language_validation.py`：仅合并到 `reports/levo-test-report.json`，不提交本地权重和多份原始日志。

主要依据：[MNN LLM 文档](https://github.com/alibaba/MNN/blob/master/docs/transformers/llm.md)、[官方导出器](https://github.com/alibaba/MNN/blob/master/transformers/llm/export/llmexport.py)、[CPU 低比特卷积执行器](https://github.com/alibaba/MNN/blob/master/source/backend/cpu/compute/ConvInt8TiledExecutor.cpp)、[QuaRot](https://github.com/spcl/QuaRot)。部署结论以本地固定源码和真机结果为准，不能用 master 的新功能冒充当前二进制能力。
