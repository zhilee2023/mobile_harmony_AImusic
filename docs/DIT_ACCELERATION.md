# 当前DiT怎样继续加速

2026-10-30核对：本项目完整39层Mixed DiT，hidden1200、FFN4800、30头×40维，256帧、CFG双batch、每块两步。当前App仍用原W8、CPU High、Memory Low、6线程。新增的Snake只改变VAE，不是DiT优化。

## GEMM改Conv已经做过

真机回调中，39个FFN up、39个FFN down、39个QKV和39个输出投影均为`Convolution`。最慢节点名例如`/cfm/estimator/h.16/mlp/c_proj/Gemm_output_0__matmul_converted`，尽管名字保留Gemm，运行时已是转换后的Conv。1×1 Conv与相应Linear执行同样的矩阵乘法；降低成Conv可能让框架选择更合适的核，但不能再降低一次就重复计算收益。

固定版本MNN低内存量化分支使用`DenseConvInt8TiledExecutor`，存在动态激活量化和INT8 GEMM选择。不能简单地将原路径描述为“只有W8存储、完全没有INT8计算”，也不能只看算子类型就断言每个节点用了最佳微内核。需要记录实际选核和打包/量化开销。

官方普通Kleidi量化工厂的这个分支还检查`canUseInt4`、block与CPU能力，不等于任意W8图打开开关就会进入旧CosyVoice的手工W8A8核。当前配置明确关闭该工厂；历史LLM开关实验也没有decode收益。

## 优先级与验收

已有CPU High/6线程完整W8回调：FFN down25.30%、FFN up16.49%、QKV12.35%、Raster13.69%、BinaryOp14.07%、While9.32%、Softmax2.75%。这是单次预热后的CLI统计，不是本轮App新热点或硬件计数器。

| 优先级 | 实验 | 必须计入/验证 |
|---|---|---|
| 1 | 真实FFN down 4800→1200，对比原MNN与手工Kleidi/i8mm矩阵核，再连up 1200→4800与真实GELU | 真实输入、CFG双batch共512行；输入动态量化/打包、线程切换、输出布局桥接全部计时；与原W8输出及最终velocity比较 |
| 2 | 消除FFN内部多余Raster，复用输入/输出打包；融合能保持数值的AdaLN/残差逐元素子图 | 依照真实拓扑，不将所有BinaryOp当作激活；显式图调度管理buffer生命周期，避免VAE回调探针曾遇到的复用覆盖 |
| 3 | 非因果Attention融合/分块，保持RoPE、mask、30×40头维与CFG | 当前While＋Softmax约12%，只是可优化范围；转换器曾失败，不能声称FlashAttention已启用 |

先做单层matched对照，再扩大39层、两步和五块整链。原图权重＋新Kleidi打包权重同时保留时必须标注重复常驻；正式接入应删除被替代原算子及权重，否则微基准快也可能增加内存和初始化时间。旧CosyVoice是22层1024/2048，不可搬用其约4ms作为本模型成绩。

## 已有负面结果和速度上限

- 历史完整W4/g48单次velocity约4.38秒，W8约3.34秒；W4更省存储但更慢，不能默认继续降bit就更快。
- CPU Low独立有限值不等于整链更快。Low DiT＋VAE8线程组合五块RTF3.443，比保留的配置慢，未启用。
- MNN通用Transformer/C4融合和隔离转换失败，没有可用于App的融合模型。
- 仅将FFN耗时减半，按41.8%占比估算DiT约1.26倍；若其他阶段完全不变，整链减少的时间仅为当轮DiT总耗时的约20.9%。这是假设估计，不能当成已测收益，也不够独自保证RTF1.5。

源码核对：[MNN固定版本卷积选择工厂](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/source/backend/cpu/compute/ConvolutionFloatFactory.cpp)、[低内存量化执行器](https://github.com/alibaba/MNN/blob/d407447ed56c4121a11ccbd266dc184ca1ead0c2/source/backend/cpu/compute/ConvInt8TiledExecutor.cpp)。本轮没有修改这两份上游源码，也没有接入新的DiT微内核。
