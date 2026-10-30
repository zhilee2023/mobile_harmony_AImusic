# MNN 推测解码与本项目接入边界

2026-10-30核对本机固定MNN提交 `d407447ed56c4121a11ccbd266dc184ca1ead0c2`，没有把新版文档中的选项套到旧二进制上。

## 已有路径

官方 `GenerationStrategyFactory` 实际选择 `lookahead`、`mtp`、`eagle`、`dflash`，否则普通自回归。配置键为 `speculative_type`、`draft_predict_length`。模块还须通过 `canSpecDecode` 检查；设置配置本身不保证启用。

| 路线 | 额外要求 | 本音乐原型的可行性 |
|---|---|---|
| Lookahead | 使用历史/查找表中的n-gram候选，不必另外训练神经草稿模型 | 可先做工程探针，但音乐码命中率和CPU验证成本尚未测量 |
| MTP | 匹配主模型的多token预测模块/训练权重 | 当前普通Qwen骨干＋随机音乐头不具备该模块 |
| EAGLE | 匹配目标模型、词表和特征的训练草稿模块 | 没有对应音乐草稿权重，不能直接用文本草稿权重替代 |
| DFlash | 匹配目标特征/词表的训练草稿模块及其导出配置 | 没有对应音乐草稿权重，暂无性能结论 |

本机归档的构建记录包含上述实现对象。此固定源码中没有 `MNN_LLM_SPECULATE` CMake选项，不应照其他版本教程添加无效开关。配置注释里提到 `draftmodel` 也不等于此工厂实际有该分支。

## 本项目尚未启用

`native-bench/ChainBench.cpp` 和共享手机链路采用手动 `Llm::forward` 循环，单步取音乐码并喂回embedding，不经过上述生成策略。输出音乐码0–16383，另有EOS；音乐输入embedding需要加 `music_offset=151968`。主模型验证、草稿输入、被拒绝token后的KV回退和停止条件必须统一这个映射。

可暂停生成还须保留已接受码、正确维护KV位置，并在codec暂停期间维持生命周期约束。只增加 `speculative_type` 不会把现有循环变成正确的音乐推测解码。

最先可验证Lookahead适配的greedy一致性、接受长度、KV回退和峰值内存，再测真实训练音乐模型的token/s与整链RTF。当前音乐头未训练，随机输出上的接受率不能代表真实歌曲。0.6B主模型较小，多token验证的额外CPU成本也可能抵消收益，不能预先承诺加速。

推测解码只减少自回归LLM生成的成本，不直接加速DiT两步采样或VAE声波解码。MTP、EAGLE、DFlash的附加参数和KV还需要单独记账，不能沿用原共驻峰值。

官方源码：[策略工厂](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/transformers/llm/engine/src/speculative_decoding/generate.cpp)、[配置](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/transformers/llm/engine/src/llmconfig.hpp)、[启用检查](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/transformers/llm/engine/src/llm.cpp)。这是源码核对与接入设计，尚未执行音乐推测解码实验。
