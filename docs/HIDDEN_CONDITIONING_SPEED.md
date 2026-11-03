# hidden条件链路：首轮真机速度优化

2026-11-03，HUAWEI Mate 90 Pro Max / CMM-AL00，同一MNN 3.6.1静态库。CLI独立进程，手机各轮亮屏；没有重装／重启App、改默认权重或启动歌曲生成。测试的是已有0.6B骨干，音乐适配头未训练；不是私有Qwen 3B或公开LeVo 4B的性能结论。

## 实测与选择

同一514位置prefix、同一255个输入音乐码，得到256个predictor hidden，第一项来自prefill。每个独立进程两轮，第二轮记预热；所有轮次的logits、各hidden模式的输出均对固定参考逐值一致。

| 项目 | 原方式／对照 | 候选 | 结果范围 |
|---|---:|---:|---|
| 256项hidden收集 | 每token分配后再拼接：2.284ms | 预分配连续缓冲：0.694–0.720ms | 少约1.6ms；不是LLM主瓶颈 |
| decode线程确认，3→4→3→4 | 4线程：57.58–61.96 forward/s | 3线程：65.48–72.47 forward/s | 输出逐值一致；有温度／调度漂移 |
| 同轮prefill＋255 decode | 4线程：5.916–6.601s | 3线程：5.431–5.822s | 相邻比较少约8.2%／11.8%，中位数少10.1% |
| 256×1024→1024 affine，4线程 | 逐帧：26.64–26.67ms | 整块：4.31–4.86ms | 各进程预热中位数约5.49–6.19倍 |
| 64×1024→1024 affine，4线程 | 逐帧：7.66–8.89ms | 整块：1.53–1.80ms | 约4.93–5.00倍 |

affine使用**真实学生hidden＋合成FP32投影权重**，仅验证按帧独立线性变换的算术和调度，不能当训练过的conditioner或整链提速。批量与逐帧相对L2为`7.1–7.2e-7`，预设门槛`1e-5`。6线程也有约5–6倍的批量收益，但绝对耗时没有优于4线程。

2线程decode约54.70/s、prefill约2.49s；6线程一轮decode约61.16/s、prefill约1.96s，均无优于3线程的本轮证据。确认组3线程prefill约1.91–1.93s；4线程约1.80–2.17s，所以不能把decode的提升比例直接套到整个LM阶段。

确认组采样RSS峰值约464.9–477.9MiB，含16MiB logits参考缓冲、hidden及运行时。它不是App共驻峰值或显存。各测试端点电池温度约28–34°C；确认组33–34°C，未锁频或固定绑核。logits-only首尾吞吐从68.63降至61.81/s，因此**不能把中间hidden输出模式的耗时差全归于hidden读取**。

## 已实现的接口优化

- `HiddenStateTap.hpp`新增`copySingleStepInto()`：把当前forward的hidden直接写进调用方连续缓冲，省去每token独立hidden向量和最后拼接。MNN的`getOutputs()`仍可能分配内部小vector，不能称全引擎零分配。
- 增加显式`SingleStepLayout::Channels1x1`。实测旧转换图输出为`[1,1024,1,1]` FP32；最初只接受`[1,1,H]`的helper主动拒绝，失败记录保留。本轮只适配单位置的channel/1×1形式，不从多位置／带padding输出猜最后一项。
- `FramewiseAffine.hpp`提供可复用FP32批量投影primitive：严格检查weight/bias和输入输出大小，复用MNN图及常量packing。实际使用必须传匹配训练权重；不能替代时序卷积、跨帧归一化、attention或带状态conditioner。
- 首次prefill的最后hidden存为h₀；后续复用KV，绝不为了hidden再forward一次。末层归一化取点仍要与私有训练代码核对。

示意接口（不表示App已完成接入）：

```cpp
std::vector<float> window(256 * H), condition(256 * C);
LevoHiddenResearch::FramewiseAffine projector(256, H, C, trainedWeight, trainedBias);
// 每次模型forward后，先复制再允许下一次forward复用MNN输出。
LevoHiddenResearch::copySingleStepInto(model, window.data() + frame * H, H,
    LevoHiddenResearch::SingleStepLayout::Channels1x1); // 仅适用于已核对的native布局
// 收齐一窗后投影一次；mask、尾窗padding和CFG遵守训练接口。
projector.runInto(window.data(), window.size(), condition.data(), condition.size());
```

## 复现入口

先配置本机鸿蒙工具链及MNN路径，构建两个独立CLI目标：

```powershell
cmake --build native-build --target levo_hidden_condition_bench levo_hidden_projection_bench
pwsh -File tools/run_hidden_condition_bench.ps1 -Device YOUR_DEVICE -RunId YOUR_UNIQUE_RUN -Suite outputs
pwsh -File tools/run_hidden_condition_bench.ps1 -Device YOUR_DEVICE -RunId YOUR_OTHER_RUN -Suite threads-confirm
```

脚本复用研究App已有`chain-comparison/small`模型和固定码，生成独立临时配置，不改原config。每组是独立进程，避免先建4线程池再声称测到了完整6线程。`threads`另保留2／3／4线程探索；`outputs`保留只读logits／逐项分配／直接缓冲和6线程对照。失败或息屏停止，不自动重跑。

投影CLI参数为`hidden256x1024.f32 threads report.json`，权重固定合成seed2307。它是可复现算子测试，不能用该权重生成音乐。源码直接使用新`FramewiseAffine`，最终4→6→4线程三个进程均通过数值门槛。

`tools/summarize_hidden_speed.py`合并`--run`与`--projection`结果，逐值核对本地hidden/logits，再用`--append-unified-key`追加唯一`reports/levo-test-report.json`，拒绝覆盖历史。原始张量和控制台日志只留本地；48项Python回归通过，新增用例会拒绝丢失／改变prefill第一项hidden和未完成的实验。

## 下一步部署边界

优先候选是**LM 3线程＋连续hidden缓冲＋批量投影**。这轮未改App默认值：MNN共享线程池的首次初始化会影响后续DiT/VAE并发能力，必须测试同进程最大线程池初始化和整链RTF，不能把CLI三线程结论直接套到App。

private hidden→DiT仍缺配套conditioner／DiT权重与训练定义，因此没有新的私有歌曲或整链RTF。当前优化仅省毫秒级hidden处理和约10%的本轮LM forward时间；下游DiT/VAE可能占更大比例。接入训练权重后先单窗，再60秒，同时测首块延迟、阶段时间、RSS/PSS与温度，原token/codebook App和模型保持原样。

## 可复用经验：后续模型与Agent按此执行

| 经验 | 具体做法 | 本轮依据／适用边界 |
|---|---|---|
| 从同一次forward取双输出 | prefill最后hidden为h₀；采样z₀后，输入z₀得到h₁。下一次forward前复制当前hidden，持续复用KV | predictor对齐；不能额外forward取hidden，也不能漏掉h₀。consumed-token训练定义需要另行对齐 |
| 先查真实输出布局，再写读取器 | 同时检查shape、元素数、dtype及取点。转换图的`[1,H,1,1]`用显式适配；存储合同仍为`[B,T,H]` | 首轮canonical-only检查拒绝了真实图。多位置、带padding输出不能凭元素总数盲目reshape |
| 连续缓冲优先于逐token容器 | 预分配一窗，直接写入目标行，有限值检查和复制在同一循环；复用缓冲 | 256项约2.28→0.7ms。仍有MNN内部API开销，不能称全路径零分配 |
| 独立线性投影按块做GEMM | 复用固定形状图、常量weight packing和输入输出空间；比较64／256帧。尾窗mask按训练合同处理 | 合成affine约5–6倍，仅省约20ms/256帧。时序卷积、attention或跨帧归一化不能直接套用 |
| 线程数必须实测 | 新进程分别测2／3／4／6，再交错复测候选；分别记录prefill、decode | 本0.6B候选为3线程，不代表私有3B或公开4B也应改成3。load后改JSON不证明runtime线程已变化 |
| 单模块选优后回到整链 | 检查同进程线程池初始化、实际线程规模、LLM与DiT/VAE共驻，再测首块延迟和整链RTF | CLI三线程可能影响后续模块可用线程；不能将单模块收益直接设为App默认 |
| 数值对照使用同一token轨迹 | 固定prefix、teacher token和KV长度，包含prefill第一项；比较hidden与logits，再比较condition、latent和PCM | 本轮不同模式／线程逐值一致。仅token相同或音频有限值不足以证明hidden条件兼容 |
| 先看绝对省时，再看倍数 | 同时列毫秒、阶段占比和整链时间。若阶段占比p、阶段时间减少r，其他不变时整链时间仅减少p×r | hidden收集缩短数倍仍只是毫秒；批量投影5–6倍不等于音乐生成5–6倍 |
| 热漂移与失败也保留 | 首次／预热分列，交错顺序，记录亮屏和温度；失败不自动重跑。所有摘要合并一个JSON | 本轮logits-only首尾68.63→61.81/s，不能把中间差异都归因于hidden输出 |
| 产物不可覆盖 | 每轮用新RunId；CLI在打开输出文件前检查已存在报告。HDC本地文件用Windows绝对路径并检查传输结果 | 已验证两个CLI拒绝覆写sentinel；一次正斜杠Windows路径导致传输失败，修正后才执行模型 |

接入新权重的顺序：**训练合同与参考hidden → 严格加载 → 同轨迹数值对照 → 单模块速度／RSS → 同进程线程池与共驻 → 单窗 → 60秒整链 → 决定默认配置**。匹配conditioner权重未提供时，保留算子实验接口，不用合成投影权重替代真实音乐适配器。

经验对应的代码入口为`HiddenStateTap.hpp`、`FramewiseAffine.hpp`、两个`Hidden*Bench.cpp`及`run_hidden_condition_bench.ps1`。数据以统一报告的`hidden_conditioning_speed_2026_11_03`为准；模型替换语义见[适配指南](HIDDEN_STATE_CONDITIONING.md)。
