# 0.6B音乐模型：A8、音频词表与VAE融合

2026-11-18。对象是流式实验App内的28层LarkStudio音乐Qwen，不是历史26层学生或通用语言Qwen。
实测数据统一写入[测试报告](../reports/levo-test-report.json)的`stream_lab_*_2026_11_18`区块。
原App、模型包、完整头及已完成实验保留。

## 当前配方与本次改动

| 部分 | 之前 | 本次 |
|---|---|---|
| LM主体／输出头 | W4/group64／W8 | 权重不变 |
| LM激活 | 线性内核内已有动态INT8 | 对照MNN `dynamic_option=2`，按权重分组量化输入，降低额外A8误差 |
| 音乐输出头 | 已从168,214裁到16,449类 | 只保留16,384音频类＋EOS，共16,385类 |
| 输入embedding | 完整BF16表 | 保留；歌词输入与音乐输出词表分开 |
| DiT | CPU High，6线程，两步Euler，CFG1.5，batch2 | 固定，暂停OM路线 |
| VAE | CPU Low，6线程，W8/g64卷积＋NEON正弦 | 36处Snake链融合为单算子，可显式关闭 |
| 生命周期 | 请求内唯一LM/KV、DiT、VAE共驻 | 保留；生成后释放，不叠加第二套会话 |

“W4A8”指线性层内部的低位权重和动态INT8临时缓冲；不能称整个计算图激活都为INT8。
本轮不改变KV配置。核对固定MNN源码，`attention_mode=8`选择FlashAttention且`8%8=0`不请求KV量化；
不能将它解释为KV8或KV4，本轮没有部署新的KV量化。

## A8误差与速度

同一真实歌词／MuQ风格前缀513位置，固定128个真实音乐历史位置。
对照将**同一份已W4/W8权重**展开为浮点，隔离额外激活／内核误差；没有原始量化前音乐checkpoint对照。

| 指标 | 原动态A8 | 分组动态A8 |
|---|---:|---:|
| 有效音乐类＋EOS logits相对L2 | 1.487% | 0.695% |
| argmax一致率 | 126/128 | 127/128 |
| 首轮两次decode前向吞吐 | 54.28／54.88 token/s | 48.41／49.58 token/s |
| 反序复测两次吞吐 | 51.00／52.53 token/s | 52.54／52.71 token/s |

选择分组A8是因为误差较低，尚未建立稳定速度收益。这是一个前缀的教师历史测试，吞吐不包含完整采样和音频链路；
不是20条离线校准集、AWQ或静态SmoothQuant音质验收。
实际函数指针审计确认：原版prefill＋一次decode已有196次动态INT8目标缓冲调用及198次FloatToInt8调用；
候选改为394次分组非对称INT8调用。审计不进入测速轮。

入口：[StreamLabMusicA8.cpp](../native-bench/StreamLabMusicA8.cpp)，
数值比较：[quant_validation.py](../levo_harmony/quant_validation.py)。

## 音频输出词表与EOS

新输出映射：EOS＝0；输出1～16,384对应音乐code0～16,383。
实际送回LM的音频输入ID仍为`151830+code`。原EOS行151765和音频行151830～168213逐行复制；
W8字节及每行alpha一起保留，不重新量化，不删除歌词输入embedding。

权重文件：原始412,069,026字节→已有裁剪251,805,186字节→音频专用251,737,602字节。
本轮去掉剩余64个保留类只节省67,584字节，主要裁剪收益此前已经获得。
128位置音频及EOS logits逐元素相同，额外裁剪相对L2＝0。
共享采样函数另用确定性logits检查原全头、两种裁剪头及4B映射：EOS正常停止，定长模式可屏蔽EOS，最高音频code仍映射正确。
这是停止／索引逻辑测试，不是合成音频的音质测试。

普通生成采样EOS并结束，已发布片段保留。**定长测速显式屏蔽EOS**，此策略写入结果，不能据此声称自然EOS输出到300秒。

入口：[StreamLabHeadCrop.cpp](../native-bench/StreamLabHeadCrop.cpp)新增`audio-only`参数。
它只接受验证过的本包W8布局，不是通用裁剪器。输出到独立`music-head-audio-r2`，旧`music-head-r1`保留。
本实验使用`Prior.next`的显式采样／EOS处理；不能直接接到未更新输出ID映射的MNN通用`generate`采样器。

## VAE图级融合

将`alpha×x → sin → square → inverseBeta× → +x`五个逐元素节点合并，FP16逐级舍入与NEON正弦保留。
每窗36次自定义算子；图从380节点减少到236，删除144个中间节点。卷积权重不变。
与早期callback实验不同，本次图规划不再分配被删中间节点，也不做producer快照。

| 同输入对照 | 原版预热中位数 | 融合预热中位数 | 耗时降低 | PCM |
|---|---:|---:|---:|---|
| 第一份真实latent | 5.187秒 | 4.367秒 | 15.8% | 完全相同 |
| 尾窗，反序确认 | 5.547秒 | 4.616秒 | 16.8% | 完全相同 |

MNN会话内存729.02→719.67MiB。模块采样RSS也下降约26～30MiB；这不是整链三个模型的峰值。
五阶段逻辑读写约11N→单核2N，只是这些逐元素阶段的读写估算，未测DRAM带宽；不能声称整个VAE读写下降81.8%。
大型卷积仍占主要时间，当前没有三倍加速结果。

共享代码：[SnakeFusionGraph.hpp](../runtime/include/levo/kernels/SnakeFusionGraph.hpp)、
[SnakeFusedOp.hpp](../runtime/include/levo/kernels/SnakeFusedOp.hpp)。
显式链接的[Lowp分发包装](../runtime/SnakeFusedLowp.cpp)只识别本实验的Extra类型，其他MNN算子沿用原精度判断；
注册表已占用或通道／布局不匹配时停止，不能静默覆盖未知算子。
App v0.1.9包含该能力，`vaeFusion`控制原版／融合版。真实整链成绩以报告内已完成项目为准。

同一v0.1.9 HAP、相同音频专用词表与分组A8的30秒实际对照：

| 项目 | 原VAE | 融合VAE |
|---|---:|---:|
| 总耗时／RTF | 59.979秒／1.999 | 55.691秒／1.856 |
| LM | 23.089秒 | 22.895秒 |
| DiT | 15.029秒 | 14.743秒 |
| VAE，含首次创建 | 21.351秒 | 17.526秒 |
| 20ms采样RSS峰值 | 2250.67MiB | 2241.14MiB |

本对照整链耗时减少7.15%，VAE减少17.92%；这是单次配对观察，不是多轮恒温统计。
750个code、4份WAV字节均完全相同。融合版实际调用144次Snake融合核（每窗36次）。

## A8／A4旋转敏感性：本轮止于误差门槛

新增[受限探针](../native-bench/StreamLabVaeActivation.cpp)与
[Snake＋量化／旋转融合研究核](../runtime/include/levo/kernels/VaeActivationResearch.hpp)。
在Snake输出寄存器中做32通道分组对称量化，选配固定随机符号＋Hadamard及逆变换；
保护早期层的hot方案只处理时间面积≥122880的14处输出。没有跨Snake传播旋转。

**这是完整VAE的激活QDQ误差研究**：输出仍存FP16，原W8权重不变；
没有实现打包A4存储或INT4卷积，也未包含将旋转折入权重后再次W8量化的额外误差，不能据此声称真实低比特提速。
同一真实latent，融合FP16参考一次＋六个候选，共7次前向；上限9次，全部超界后结束，未继续扫参。

| 量化输出 | 无旋转PCM相对L2 | Hadamard32 PCM相对L2 |
|---|---:|---:|
| A8，全部36处 | 4.392% | 4.109% |
| A8，热点14处 | 1.477% | 1.272% |
| A4，热点14处 | 21.162% | 11.413% |

准入线是PCM相对L2≤1%、最大绝对误差≤0.01且无越界PCM。六个候选都没有通过，
保留失败结果，未运行候选尾窗确认、未进入App。当前有效部署仍为数值一致的FP16融合。

### DCT＋SmoothQuant是否适合

DCT可以替换Hadamard作为局部正交变换，但不是必需步骤，也尚未完成本项目真机验证。
应沿卷积输入通道做变换，不能把对最终PCM做DCT编码当作模型激活量化。
对普通group1卷积，列向量记法的等价关系为
`y=W*x=(W*R^T*S)*(S^-1*R*x)`，其中R是通道正交变换、S是正对角缩放；对每个卷积核位置同时改权重输入轴。
分组／深度卷积及反卷积需要单独布局适配，不能直接套这条导出实现。

[QuaRot](https://github.com/spcl/QuaRot)和[SpinQuant](https://arxiv.org/abs/2405.16406)主要针对LLM；
[SmoothQuant官方实现](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py)用激活及权重通道统计计算缩放。
这里将局部变换用于音乐VAE是后续研究方案，不是这些项目已验证的VAE部署结果。

下一轮若继续，应先用真实窗口校准S，比较不旋转＋SmoothQuant、Hadamard＋SmoothQuant及DCT＋SmoothQuant；
先A8，验证浮点等价、W8重新量化和完整PCM，再考虑A4。缩放／旋转与Snake及量化融合，计入在线变换开销。
单独QDQ变好不足以准入；实际MNN整数卷积、权重变化及端到端性能仍须独立验收。本轮按预算停止增加量化实验。

## 手机整链对照与长时测试

已完成的v0.1.8原VAE、旧16,449类头30秒：总59.100秒／RTF1.970，LM22.765秒，
DiT14.710秒，VAE21.074秒，RSS峰值2267.95MiB；播放器时间戳到30秒，无播放错误。
该轮App状态文件漏写busy结束转换，已从模型全部释放后写出的原生`result.json`恢复；没有重新推理或拼阶段计时。
v0.1.9修复状态记录、补齐NAPI长时上限，并保留原版VAE对照。

后续按30秒同HAP原版／融合对照，再逐级60／90／120／300秒。
不把单VAE模块加速率当作整链RTF收益，也不将未完成目标当作已验收。
60秒首轮在发布46.08秒后手机息屏、App进入后台，进程仍在；该轮带有前台中断标记，
即使随后生成完成也隔离其计时，不纳入受控RTF成绩。90／120／300秒尚待连续前台测试。
相同歌词、MuQ风格、温度0.95／topK250／seed2307；每次请求包含首次加载，前台自动播放，间隔至少180秒冷却。
计时不包括冷却；RSS安全上限6GiB，系统余量下限650MiB；20ms采样RSS不是专用显存。

```powershell
pwsh -File tools/run_music_duration.ps1 -Device YOUR_DEVICE_ID -RunId my-audio30-base -DurationSeconds 30 -VaeFusion 0
pwsh -File tools/run_music_duration.ps1 -Device YOUR_DEVICE_ID -RunId my-audio30-fused -DurationSeconds 30 -VaeFusion 1
```

同一入口支持60、90、120、300秒。若冷却后息屏，记录为`awaiting-unlock`，没有开始推理；
手动解锁后可用相同参数加`-ResumePrepared`继续该准备记录。该参数拒绝重启已经开始或完成的请求。
系统终止、后台／息屏与最后有效输出须如实记录；不能仅凭退出就认定OOM。

汇总：`python tools/merge_stream_runtime_optimizations.py`。
只发布源代码、配置、验证摘要与统一报告；权重、音频、原始日志及签名留在本地。
