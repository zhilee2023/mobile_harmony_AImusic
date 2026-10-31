# 公开 LeVo 2：4B＋700M 与最终单条歌曲

截至2026-10-31，对照官方论文、v2-large配置与公开模型元数据。**架构核对和接口合同已做，公开4B模型尚未在本项目生成音乐，W4A4KV4尚未通过端侧验证。**

```text
歌词＋风格描述＋可选10秒音频提示
  ↓ Qwen2 BPE分词＋训练过的条件embedding
Mixed Semantic LM：36层 / hidden2048 / FFN11008 / 16头
  ↓ mixed token＋连续hidden特征
Track-Specific LM：12层 / hidden2048
  ↓ 人声token＋伴奏token（并行输出，两路均25Hz）
双轨条件Music Codec DiT：约700M量级
  ↓ 同一份混合音频latent，64维 / 25Hz
VAE decoder：论文约150M
  ↓ 一条48kHz双声道混合歌曲
```

## “单轨”必须区分输出和内部表示

默认可以直接得到包含人声与伴奏的一首歌曲，用户不需要提供两路音轨。源码`generate_audio`将人声和伴奏token共同送入同一个解码器，生成混合latent，再由VAE输出同一条双声道波形；内部双轨token不等于必须交付两个WAV。

若要求内部也只有一路：只保留Mixed Semantic LM，用Mixed 1RVQ codec解码。它仍可输出混合歌曲，但属于去掉Track-Specific LM的单路消融，不能继续称作完整4B原链路，也不能直接复用双轨codec权重。

| 路线 | LM生成流 | codec消费流 | 输出 |
|---|---|---|---|
| 官方层次式完整路线 | Mixed＋人声＋伴奏 | 人声＋伴奏 | 一条混合歌曲 |
| Mixed-only单路路线 | Mixed | Mixed | 一条混合歌曲 |

两套公开渲染器分别有不同权重与接口：已检查的Mixed DiT为39层、hidden1200、FFN4800、30头；双轨条件DiT为16层、hidden2200、FFN4400、20头。约700M是论文中的扩散模型量级，不表示任意一个700M权重都能互换。

## 四个常见误解

- 约4B是Mixed LM与Track-Specific LM的核心参数合计，不是一个现成通用Qwen4B。Qwen2提供BPE分词；不需要额外常驻Qwen2-7B文本模型。
- 700M不包含VAE；参考音频的约300M MuEncoder也另算。纯文字生成无需常驻音频编码器。
- 官方歌词/描述条件不是本项目学生原型的“512歌词＋1个任意风格token”契约；公开配置有描述600、type_info100等上限，使用训练过的条件embedding和段落标记。
- 内部双轨有250帧（25Hz下10秒）的delay pattern。手机端分块不能简单丢掉该时序关系或把两路不齐的token直接送入codec。

## 权重与内存核对

已固定`lglg666/SongGeneration-v2-large@115805364ad74479fb3764fe65970c92faeb1a5a`，官方`model.pt`为12,899,965,446字节。通过约90KB精确HTTP范围读取ZIP和安全的符号化张量元数据，没有执行模型pickle函数、没有加载权重payload。

共有458个张量描述；两LM分组约3.106B与1.080B张量元素，另外有较大的文字条件embedding和其他张量。全文件描述元素约5.122B，不能只用“4B×2字节”估算原包大小。部分FP32张量打包在一个平坦storage的不同offset里，不能将共用storage误判为相同权重。

用户目标是W4A4KV4。全参数理想W4权重下界约2.56GB，尚未算scale、保留高精度的embedding/norm、packing、两套KV、CFG与激活；**A4不会进一步缩小W4权重本身**。官方两LM是MHA而不是Qwen GQA：合计48层、宽2048，FP16 KV每位置约384KiB；理想KV4约96KiB，另有量化元数据。统一报告提供30/180/270秒的明确假设预算，不把估算当真机峰值。

MNN的W4和TQ4 KV可继续研究；当前A4静态路径在原始Qwen语言测试中尚未通过，不能据此声称这份LeVo已能以W4A4KV4部署或产生可听歌曲。下一步先修复/验证A4执行，再做公开4B权重严格加载、短token对齐及对应codec音频生成。

## 不同尺寸接口

新增`levo_harmony/pipeline_spec.py`，由配置读取层数、hidden、FFN、head与码本大小，并检查：

- 实际预训练张量形状和层索引；
- 条件embedding宽度；
- Mixed与双轨的codec身份、输入流及对齐长度；
- EOS/非法token拒绝；
- W4A4KV4执行与数值验证必须全部就绪。

这允许为large（36层/2048宽）或medium（28层/1536宽）建立尺寸合同，不能把不同hidden的权重直接塞进同一个固定MNN图，更不等于已经实现所有尺寸的运行时和转换器。原0.6B学生入口保持独立。

依据：[官方论文](https://arxiv.org/html/2606.30642v1)、[固定版本配置](https://huggingface.co/lglg666/SongGeneration-v2-large/blob/115805364ad74479fb3764fe65970c92faeb1a5a/config.yaml)、[官方LM源码](https://github.com/levo-demo/LeVo/blob/653cbcf4716101834900c75b7d5da43b07e15d5b/codeclm/models/lm_levo.py)、[音频生成分支](https://github.com/levo-demo/LeVo/blob/653cbcf4716101834900c75b7d5da43b07e15d5b/codeclm/models/codeclm.py)。统一报告新增`public_levo_architecture_2026_10_31`。
