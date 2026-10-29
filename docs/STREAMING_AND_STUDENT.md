# 0.6B学生与10.24秒分块接口

2026-10-29。用户确认：暂时没有4B服务或Mixed token文件，先验证分块数值与内存。本文区分已实现调度、真实模块验证和待完成音乐链路。

## 维度不要混用

| 部分 | 当前配置 | 是否需要随0.6B学生改变 |
|---|---|---|
| Qwen3-0.6B骨干 | hidden1024、28层、FFN3072、Q16/KV8、head_dim128 | 保留，以直接加载公开Qwen权重 |
| 歌词输入 | 最多512个Qwen tokenizer ID | 与音频时长独立 |
| 风格前缀 | 一个1024维可训练向量，目前32个离散风格ID | 必须训练；任意风格文本不能自动压成有效前缀 |
| Mixed音乐头 | 1024→16385个logit，含EOS；码本有效ID为0..16383 | 新增并训练 |
| 音乐输入embedding | codec ID→1024维，另有BOS约定 | 新增并训练 |
| 官方RVQ反量化 | 16384×32码本，out_proj输出1024维mu | 保留完全相同的码本权重与ID语义 |
| Mixed DiT | hidden1200、39层、30头、FFN4800 | 不因LM缩小而修改 |
| VAE decoder | 输入64维latent，25Hz；输出48kHz双声道 | 保留，DiT与VAE统计量必须匹配 |

DiT输入的1200维为`mask48＋context64＋mu1024＋noise64`。它不是Qwen hidden的直接延伸；双方用离散Mixed token ID解耦。若教师与学生输出完全相同的codec ID，可替换LM而保留DiT/VAE。不要把4B教师的hidden向量直接当0.6B输入，也不要静默改DiT hidden到1024后忽略权重不匹配。

公开Qwen配置：[固定revision的config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)。Mixed codec形状已与实际checkpoint头逐项核对。

当前28层学生共629,640,192参数，约0.630B；Qwen骨干为596,049,920。若硬性要求总参数小于0.6B，现有26层剪枝原型为598,178,304，但必须恢复训练，未经音乐质量验收。直接改变hidden或FFN会破坏公开权重的严格兼容。

## 4B教师的使用方式

可以先由官方教师离线输出Mixed token，在手机测试codec；4B无需放进手机。学生训练样本应包含歌词、风格标注和**同一码本**的Mixed token序列，采用teacher forcing预测下一音乐ID与EOS。先验证CE训练路径，再视数据与教师权限补充logit蒸馏；不同hidden宽度不影响同词表的token/概率蒸馏。

当前没有训练数据、GPU或4B token源，不把随机学生输出称为音乐。手机测试采用明确标记的数值夹具，音乐头的拒绝未训练生成保护继续保留。

## 分块调度

官方Mixed代码默认40秒/1000token窗口、30秒/750token推进、10秒/250token重叠，逐块使用前一块末尾latent作上下文。原公开函数先收集latent，再统一VAE解码返回，不是开箱即用的边生成边播放接口。[官方代码](https://github.com/levo-demo/LeVo/blob/653cbcf4716101834900c75b7d5da43b07e15d5b/codeclm/tokenizer/Flow1dVAE/generate_1rvq.py#L132)

手机候选窗口：256token=10.24秒，推进192token=7.68秒，重叠64token=2.56秒。它沿用官方分块原理，但短窗音质仍需实测，不能声称等效官方40秒窗。

1. 首次积累256有效音乐token，运行完整DiT采样并VAE解码。
2. 输出前192帧PCM；保留64帧latent上下文和PCM尾部。
3. 再取得192个新token，组装“旧64＋新192”；完成下一块后交叉淡化重叠区，再输出稳定PCM。
4. EOS时不足256的位置只在最终块填充，输出按有效长度裁剪；恰好满块时直接释放已保留PCM尾部，不重复推理。
5. LLM权重和KV延续，不重新prefill歌词。DiT/VAE只保留当前块、中间激活与尾部缓存。播放器必须另做有界队列和欠载测试。

4000个有效token为160秒；该短窗规划会产生21个固定大小解码窗口，最终有效160帧。重复上下文和尾部填充使解码工作量大于最终音频长度，RTF分母必须用**有效输出时长**，不能简单把21×10.24秒全部计为新音频。

`levo_harmony/streaming.py`已实现token窗规划和PCM交叉淡化；单元测试覆盖4000token、整窗EOS、短尾、非法码、非有限PCM、顺序错误及无漏帧/重复帧。调度缓冲上限256个ID，PCM尾部64帧。该测试不等于完整codec已流式验收。

LLM KV不是恒定内存。28层、8KV头、head_dim128、FP16时每位置114688字节；4000音乐token加满514位置前缀约493.7MiB，仅为KV，不含权重、CFG或中间激活。

## 当前验证边界

- 真DiT单层32/256帧的手机测试已完成并上传；不是39层完整多步采样。
- 真VAE decoder全部182参数已下载、CRC32校验并严格加载；Windows CPU的8/32/256帧输出形状及有限值检查通过。
- 真实VAE已导出ONNX/MNN并完成手机High/Low三窗夹具测试；同一会话完成有效25.6秒PCM。High模块计算RTF约3.44、峰值RSS1553MiB；Low约1.21、813MiB。Low块输出相对PyTorch L2约3.4%–4.6%，未作为音乐音质分数。它不包含尚在续传的完整DiT，也不测扬声器出声时间。
- 独立鸿蒙App目前仍是未签名诊断构建；CLI数值结果不能当作App播放器验收。
- 完整DiT权重、真实音乐token源与实际歌曲质量尚未闭环；最终只在证据覆盖范围报告。

## 量化与模型生命周期

| 部分 | 目前已验证 | 手机目标/待验收 |
|---|---|---|
| Qwen学生骨干 | 真实高精度PC加载；512歌词+style+BOS prefill，8个固定音乐ID的KV增量与全序列对齐 | 主体W4；音乐输出头先W8/高精度，再用训练后校准数据比较。尚未完成该学生的手机W4导出 |
| Mixed DiT | 单层W8/W4真机；长块本轮W8较快且误差更小 | 优先完整W8；完整图和多步采样仍待下载/导出验收 |
| VAE | 磁盘FP32权重；手机CPU High/Low对照 | Low候选，继续研究较小VAE子块的速度、误差和内存 |

MNN的`Precision_Low`在启用ARMv8.2、CPU支持FP16算术时选择Arm82 CPU扩展后端。它是计算/内部数据精度选项，不改磁盘权重为4bit；`Precision_Low_BF16`是另一独立选项。`Memory_Low`是内存策略，`Power_Low`是功耗/核心调度策略，不能混为一谈。各kernel的累加精度需要逐一核对，不能统称全图FP16存储、FP32累加。[MNN实现](https://github.com/alibaba/MNN/blob/3.6.1/source/backend/cpu/CPUBackend.cpp)

完整DiT基线应先按官方配置对齐；Mixed底层`code2sound`默认20步，而其`AudioTokenizer.decode`封装请求50步。当前单层手机测试没有运行完整去噪循环，不能把CosyVoice蒸馏模型的两步直接套用到LeVo权重。

用户明确希望自主探索加载、生成与释放策略。先比较独立常驻和阶段复用，再叠加模型；现有单层DiT与VAE数字不能证明三模型共驻。优先保留LLM权重和KV，避免每块销毁引擎造成上下文重建；DiT/VAE工作区是否能回收或复用必须实测，不能假定“权重常驻”就只占压缩文件大小。

App的内存按钮同时显示系统可用内存、当前进程RSS/PSS及HiDebug报告的进程RSS/VSS限制。系统可用量与报告的进程限制均不是无OOM保证；CLI身份与真正App身份的测试须分别记录。
