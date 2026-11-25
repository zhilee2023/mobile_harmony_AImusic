# ARM 执行原语定价与 DiT GELU 验证

设备为 CMM-AL00。DiT 使用 22 层 W8/group64 原始权重、CPU High 六线程、CFG 1.5。每个进程使用同一次实际生成的前 256 个音乐 token、固定噪声和两步去噪。两组比较分别以原版、候选、候选、原版的顺序运行，每个进程三轮，取后两轮两步耗时的均值。耗时包括输入提交和 velocity 同步，模型初始化另计。

上一轮 2–6 节点连通搜索找到了 GELU 尾部。按执行原语识别后，发现每层完整八节点的 Tanh 近似 GELU 中还有一个通用 Pow。原 profile 的 22 个 Pow 回调合计 240.44 ms/步。这是原版热点，运行收益以两步对照为准。

| 只改变 DiT 的候选 | 原版两步 | 候选两步 | 耗时下降 | 两步 velocity 相对 L2 |
| --- | ---: | ---: | ---: | ---: |
| 22 层完整 GELU 换原生 MNN GELU | 3,276.20 ms | 2,864.41 ms | 12.57% | 4.21%、9.33% |
| 22 层只将 Pow(x,3) 换成两次乘法 | 3,303.35 ms | 3,013.07 ms | 8.79% | 2.85%、6.95% |

两种候选均未通过相对 L2 0.1% 的数值筛选。两组的原版重复输出逐元素一致。保持原 DiT 路径。分时数据来自 27–30°C 的亮屏 CLI。两种候选使用各自同组的原版基线。

独立实验入口为 `levo_dit_gelu_pricing BUNDLE CODES_JSON NEW_OUTPUT_DIR stock|gelu|cube [--trace]`。实验在内存中重写，保持原图和外部权重文件的内容。可单独构建 `levo_dit_gelu_pricing`。输出 velocity 是本地配对输入，结果 JSON 可用来验收。`--trace` 采集第一去噪步的中间结果，独立于性能计时。

### 数值差异追踪

复用同一首窗输入。第一层 Up 输出三种写法逐元素一致。原生 ARM GELU 使用有理式近似 Tanh，原图的独立 Tanh 走指数计算。单改 Pow 时，`pow(x,3)` 与两次 FP32 乘法的舍入也不同。测试对相同首个 FFN 的 Tanh GELU 输出和随后 Down 投影的 A8 输入进行了独立采样。

| 首步检查点 | 原生 GELU | 两次乘法 |
| --- | ---: | ---: |
| 第一层 GELU 输出相对 L2 | 0.002565% | 7.94e-7% |
| 第一层动态 A8 变化 | 8 个值，分布在 8 行 | 0 |
| 第一层 Down 输出相对 L2 | 0.10901% | 0% |
| 第 11 层 Down 输出相对 L2 | 4.86546% | 2.29161% |
| 第一处 Down 的动态 A8 变化 | 第 1 层 | 第 5 层 |

GELU 版首层有 8 个 A8 码改变，恰好对应 Down 输出上发生变化的同 8 行和 9,600 个元素。W8 Dense 的输入在 Packing 前动态量化。独立的逐层 A8 摘要显示，到第二去噪步两种候选的 22 层 Down 均与原版不同。第二步的 latent 来自第一步 velocity 更新，会继承第一步差异。插桩前后 velocity 对每个变体均逐元素一致。

## 实际 W4 执行路径

保持原 0.6B W4 主体、W8 输出头和 CPU High 四线程。对固定 MNN 归档对应的 `DenseConvInt8TiledExecutor` 做诊断插桩，执行 2 轮 × (513 位置 Prefill＋32 次固定历史 Decode)。33 个不同前向位置的 logits 与原样归档逐元素一致。只统计 W4 主体，每次前向共 196 个投影。该路径先动态量化再做 Pack A，Prefill 按行分工，Decode 按输出通道分工。

| W4 Dense 范围 | 一次 Prefill | 一次 Decode |
| --- | ---: | ---: |
| 执行器累计墙钟 | 1,375.815 ms | 10.378 ms |
| 动态量化，串行 | 81.048 ms | 0.218 ms |
| Pack A | 38.369 ms | 0.287 ms |
| GEMM 微内核含输出处理 | 1,260.398 ms | 8.989 ms |

Prefill 的 Pack A 和两类 GEMM 数值采用每次执行各阶段最慢 worker 的时间再累加，其余列按标注为串行或墙钟。多线程区间可能交叠，分项不能按墙钟直接加总。W4 Dense 中未观察到独立的输出再量化和在线权重重排调用。GEMM 内的反量化、Bias 和输出写回保留在 GEMM 项，执行器外的布局转换需要另测。

22 层 DiT 均已使用 `1200→3600` 的合并 QKV。0.6B LLM 的 28 层均有独立 Q/K/V 共用一个输入张量，另有 28 处 Up/Gate 共用输入。候选可先根据共享输入和执行器的量化、线程、tile 信息筛查，再通过量化组和 packed ABI 实测确认复用。

以形状组的动态量化、Pack A 为依据，假设 Q 的一次输入准备可复用于 K、V，理想代理收益为约 22.69 ms/Prefill 或 0.102 ms/Decode；Up/Gate 减少一次输入准备的对应估计约 11.92 ms 和 0.054 ms。两项仍需测复用后的 GEMM、输出布局和缓冲生命周期。假设代理不能用于填充实测候选库。

## 启用与关闭动态 A8 的同权重对照

独立 CLI 保持图、外部权重、输入和 CPU High 不变，以 `Memory_Normal` 展开同一批已量化权重为 FP32，作为这轮的参考。原版 DiT 的 GELU 保持原公式。独立插桩确认 DiT 的 92 个 W8 卷积位置在浮点组展开权重，LLM 的 196 个 W4 和 1 个 W8 输出头也是如此。插桩复算输出与对应的无插桩执行逐元素一致。此比较包含动态激活量化和两种矩阵内核的数值差异。未取得这两套音乐模型量化前的匹配 checkpoint。

| 模块 | 动态 A8 | 同权重浮点执行 |
| --- | ---: | ---: |
| DiT，256 帧，两步 | 3.400 秒 | 9.487 秒 |
| DiT 进程最高采样 RSS | 474.65 MiB | 1,596.26 MiB |
| 0.6B LLM Prefill，513 位置 | 2,256.48 ms | 5,114.03 ms |
| 0.6B LLM Decode，32 次 | 18.961 ms/token | 64.870 ms/token |
| 0.6B LLM 进程最高采样 RSS | 497.08 MiB | 1,960.90 MiB |

以浮点组作为相对 L2 的分母。DiT 两步的原始 velocity 为 6.290% 和 11.972%，CFG 1.5 混合后为 6.926% 和 13.275%。第二步沿各自第一步更新后的 latent 继续。DiT 顺序为 Low、Normal、Low，三轮取后两轮。两次 Low 输出逐元素一致。

LLM 使用固定 513 位置 Prefill 和 32 次 teacher-forced Decode。Prefill logits 相对 L2 为 1.351%，Decode 汇总为 1.191%，最差单次前向为 2.145%，33 次前向的 argmax 均一致。LLM 性能为每种路径一个独立进程，各取第二轮，电池约 32–33°C。两项对照均为模块级，没运行歌曲生成。

进一步核对 W4 Dense 的输入准备。对 513 位置 Prefill 和首个 Decode token，在实际执行中记录 INT8 输入和 scale/zero-point 缓冲的摘要。Q/K/V 三路和 Up/Gate 两路，各自在两阶段的 28 层均一致，对应的 Pack A 参数也全部一致，且未进入依赖各投影权重的 bias 更新分支。证据支持优先验证 Q/K/V 复用。跨算子缓存的生命周期及合并后的时延仍需验证。

`levo_dit_gelu_pricing` 的浮点参考使用 `stock --float-activation`。LLM 参考使用 `tools/run_llm_operator_lab.py --transport cli --memory normal`，同一模型和相同 teacher token 数的默认 Low 组用于比较。
