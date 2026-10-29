# 旧CosyVoice内核优化对LeVo的可迁移性

依据本机旧项目代码/历史真机报告及本轮完整LeVo DiT真机回调统计，不套用不同模型的微基准成绩。旧工程只读。

## 之前实际做过什么

- MNN把常量Linear/MatMul降低为1×1 Convolution，进入CPU低内存量化矩阵路径；bias/缩放等已有融合，不能重复计收益。
- `C:\cosyvoice-harmony\app\entry\src\main\cpp\KleidiFfnRuntime.inc`手动接入Arm KleidiAI 1.16.0的`kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm`。输入动态量化/打包W8A8，权重按输出通道重新量化、一次RHS打包；6线程分块，调用前可暂停MNN线程池防止竞争，两个矩阵之间调用真实GELU。
- 旧路径为22层FFN1024→2048→1024。早期6线程两矩阵预热中位数约4.96/4.15ms，复测约8.71/8.37ms，波动明显。不是LeVo已测4ms，更不是整个DiT4ms。
- 老代码明确保留`originalFfnWeightsStillResident=true`。直接复制会同时保留原图FFN权重和Kleidi打包副本；生产迁移应去掉被替换原算子/权重，或至少单独记账。

## 本轮完整39层DiT热点

CPU High、Memory Low、Power High、6线程，固定batch2×256帧；一轮预热后回调一轮。CLI身份，Debug/callback有开销，可嵌套计时；比例是callback累计时间占比，不是硬件计数器或整链RTF。

| 类别 | W8占比 | W4/g48占比 |
|---|---:|---:|
| FFN up | 16.49% | 21.62% |
| FFN down | 25.30% | 22.75% |
| QKV投影 | 12.35% | 16.55% |
| Attention输出投影 | 4.45% | 5.56% |
| Raster布局 | 13.69% | 10.94% |
| BinaryOp | 14.07% | 10.92% |
| While | 9.32% | 8.07% |
| Softmax | 2.75% | 2.15% |
| UnaryOp | 1.16% | 1.07% |
| LayerNorm | 0.28% | 0.21% |

W8完整单次velocity墙钟约3.34秒；W4约4.38秒。两种图均有每类投影39次，覆盖所有层。W8最慢单节点集中在`/cfm/estimator/h.N/mlp/c_proj/Gemm_output_0__matmul_converted`，FFN down先做最合理。

本轮没有接入手写Kleidi FFN，也没有实现新的算子融合。MNN会话明确`CPU_ENABLE_KLEIDIAI=0`，结果属于原生MNN量化CPU路径。

## 迁移顺序

1. **真实FFN down，再连完整FFN**：LeVo为1200→4800→1200，每token每层矩阵乘加约是旧1024→2048→1024的2.75倍，且39层对22层；还要计batch与行数。复用Kleidi Projection的打包/分线程方法，改维度、原权重Conv1D转置约定、真实GELU和输出注入节点；先对同一层实际输入做耗时/误差/打包内存，再推39层。
2. **减少FFN内部Raster与中间张量往返**：图中Linear已经变成Convolution，单纯再转一遍不会产生收益。保留子图内部布局，权重只打包一次，边界统一转换。
3. **融合AdaLN/逐元素/RoPE子图**：BinaryOp＋Raster约28%，需要按真实拓扑归因，避免把所有BinaryOp都当GELU。仅融合Unary激活最多覆盖很小部分。
4. **无KV的Attention融合**：保持30头×40维、非因果mask、真实RoPE与CFG语义。当前While＋Softmax约12%，不能把FFN时间算进FlashAttention收益；原图没有Attention算子，运行时开关不能自动创建融合图。

MNN当前`ConvolutionFloatFactory.cpp`中，低内存量化路径默认走`DenseConvInt8TiledExecutor`；Kleidi工厂存在额外分支并检查`canUseInt4`等条件。不能把“启用Kleidi开关”当作旧自定义W8A8核已经接入，也不能凭枚举中有W8就声称该工厂已支持所有W8组合。实际选核必须确认。源码位置：`C:\cosyvoice-harmony\third_party\MNN\source\backend\cpu\compute\ConvolutionFloatFactory.cpp`及`kleidiai/mnn_kleidiai.cpp`。

即使FFN快一倍，按本轮热点粗估也只是DiT约21%的时间减少；DiT约占最快整链的29%，对应整链约6%，不是已实测收益。当前整链还需并行关注LLM和VAE，不能承诺单个FFN kernel把RTF从2.4降到实时。
