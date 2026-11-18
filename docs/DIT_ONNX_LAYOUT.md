# 当前 DiT：MNN 布局、ONNX 导出与 CANN 子图验证

2026-11-13，独立布局研究。对象为实验App中28层Qwen配套的22层DiT，H1200、
FFN4800、30头×40维，W8/g64。不是历史1024宽的CosyVoice FFN，也不是已完成整DiT的NPU迁移。
App v0.1.7仍走MNN CPU；按用户要求恢复0.6B的两步CFG1.5。

## 当前真实图

从手机当前模型读取元数据，并用真实音乐code重建首窗输入、运行算子回调：

| 项目 | 结果 |
|---|---|
| 主输入 | `[2,256,1200]` |
| mask | `[2,1,256,256]`，时间输入`[2]` |
| FFN up输入 | `[512,1200,1,1]`，NC4HW4 |
| FFN up输出 | `[512,4800,1,1]`，NC4HW4 |
| GELU内部 | `[512,4800]`，普通布局，Pow/Tanh/乘加链 |
| 静态算子 | 92 Conv、44 MatMul、88 Permute、184 ConvertTensor、427 Reshape |
| 实际回调 | 1506个节点，其中565个Raster |

MNN已把恒定权重Linear转换成1×1 Conv；再次简单地“MatMul转Conv”不是新的优化。
565个Raster也不代表565次等量DRAM拷贝，其中包括几何切片、视图及布局操作。
While节点还用于广播/矩阵计算的几何降低，不能全部解释成Attention或Python循环。

## ONNX候选

本轮从原MNN W8/g64参数解码出首个FFN的OI权重与真实bias，并保留原始捕获。
这是量化后权重的浮点展开，不是恢复了未量化训练checkpoint。

| 候选 | 子图I/O | 用途 |
|---|---|---|
| Gemm flat | `[512,1200]` | 两个Gemm，常量OI权重，`transB=1` |
| Conv flat | `[512,1200,1,1]` | 复现CPU将时间位置摊到batch的布局 |
| Conv spatial | `[2,1200,1,256]` | 连续时间轴作为W，FFN内部不做transpose |
| Conv bridge | `[2,256,1200]` | 输入/输出各一次transpose，内部保持Conv spatial |

每种布局分别导出Pow、两个Mul计算三次方、MNN匹配写法三种GELU，共12个候选。
同一批CFG无条件/条件真实输入下，ORT CPU相对统一FP32数学参考L2为
`8.71e-7`至`9.19e-7`，均通过预设`1e-5`门槛；使用同一份46,104,000字节参数文件。
全部保留tanh近似GELU，没有替换为ReLU。时间/调制分支中的SiLU也不应随意替换。

### GELU表达式会影响融合

本机MNN转换器的`FuseGeLu.cpp`识别：

```text
x * ((1 + tanh(sqrt(2/pi) * (x + 0.044715 * Pow(x,3)))) * 0.5)
```

普通写法`(x*0.5)*(1+tanh(...))`在这次转换中未被同一规则融合。
只改变乘法结合顺序后，实际重新转换的MNN图由9个BinaryOp降至2个，
Unary从TANH变为GELU；两个额外BinaryOp属于形状处理。
两种图静态ConvertTensor数量仍为4，不能据此宣称布局转换已全部消除。
`--convertMatmulToConv`在本机转换器默认1；Transformer融合开关也不能替代数值及图检查。

## CANN/OM实测

使用既有官方DDK-tools-next-6.1.1.0，OMG生成硬件无关OM，浮点权重请求FP16，I/O为FP32。
三个OM均成功转换；日志均包含`GeluFusionMulPass success`。
没有把旧Kirin9030专用OMC改名用于当前手机，也没有声称已取得匹配9050的OMC插件。
官方区分[OM与芯片专用OMC](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-overall-parameter/)。

手机通过NNRT的HIAI_F构建执行，测试整图NPU顺序和仅两个投影节点的NPU顺序，
均请求禁回退、High性能模式。每种配置一次首次、三次预热运行。

最新桥接确认轮，整图NPU顺序请求，各列为逐次总耗时的预热中位数：

| 布局 | 含输入输出处理 | 相对同参数FP32参考L2 |
|---|---:|---:|
| Conv flat，512个batch | 320.245 ms | 0.1924% |
| Gemm flat | 6.062 ms | 0.1490% |
| Conv spatial，含CPU BTC↔NCHW | **3.709 ms** | 0.1924% |

不含BTC转换的首次布局轮，Conv spatial约2.881ms。桥接轮输入转换中位数0.554ms，
RunSync 2.797ms，输出转换0.311ms；分项各自中位数不必相加等于总计中位数。
NEON 4×4转置对256×1200和1200×256均逐元素通过独立标量索引参考。
这里未计入将子图接进MNN后的回调/任务调度成本，不是App整链RTF。

独立进程20ms加阶段检查的采样RSS峰值约165.51MiB；NPU私有工作区峰值未测，
不能将它当作三模型共驻峰值。本轮未获取逐kernel硬件轨迹或可信的编译分区明细：
HMS Profiling选项和Build返回0，但executor创建失败；临时输出目录可写重试仍失败。
失败记录保留，没有用profiling失败推翻普通执行结果，也没有把设备顺序选项当作逐算子硬件证明。

## 与现有量化执行的差异

上述ONNX布局之间正确，不等于与当前MNN低内存执行逐样本一致：
同一低内存捕获输入下，浮点FFN相对原量化执行L2约26.74%。历史双条件捕获约18.28%，两者均保留。
独立MNN CPU High/Memory Normal探针中，使用其自身捕获输入，
解码参数的FP32数学参考与FFN输出L2约`8.53e-6`，up投影约`7.65e-7`。
这验证了浮点展开与该浮点执行路径的对齐，仍不能替代原始训练权重或完整velocity/音频验收。
当前没有将候选FFN替换进App；保留原W8 CPU图，未放宽原模型替换门槛。

## 后续整图导出原则

1. 最终生产转换优先从原Torch/PT或干净ONNX导出；不把NC4HW4、Raster、MNN内部量化编码直接写成通用ONNX。
2. 固定batch2、T256，折叠可确定的Shape/Rank/Gather/Unsqueeze链；保留CFG的两条真实分支。
   无条件分支仅清空mu条件，不能清空latent、context或mask。
3. FFN子图内部维持`[B,C,1,T]`，常量权重提前按OIHW排列；先整FFN，再测连续多层，减少往返边界。
4. Attention保留`[B,30,T,40]`语义与非因果mask，QK和PV保留动态MatMul，不能当作恒定权重Conv。
   保持原RoPE配对次序与缩放，不凭形状相似改成另一种模型的Attention。
5. 原LayerNorm沿每个token的hidden轴计算。在NCHW中直接使用`LayerNormalization(axis=1)`
   会把后续空间轴也纳入，不能当作等价变换；需回到BTC或显式只沿C归约并验证。
6. 先以本轮空间Conv+GELU方案做完整速度场与音频回归，再决定FP16或单独DOPT W8A8校准。
   MNN的W8元数据不自动成为CANN W8A8配方。

4B的DiT同宽但39层，权重和条件接口不同；可复用形状策略，不能直接复用本轮0.6B的FFN权重。

## 可复现入口

* `native-bench/StreamLabDitLayout.cpp`：真实图/布局/首FFN捕获；默认CFG双分支，保留`--no-cfg`与`--memory-normal`独立研究参数。
* `tools/prepare_dit_layout_probe.py` → `levo_harmony/deployment/dit_layout.py`：12种ONNX、共享参数、CPU数学校验；拒绝未知张量顺序和覆盖旧目录。
* `tools/convert_dit_layout_remote.py`：独立Linux目录、输入哈希、资源限制、官方OMG，默认portable OM。
* `native-bench/StreamLabNpuProbe.cpp`：构建/执行/布局桥接/20ms RSS，profiling与未桥接结果分别保留。
* `native-bench/StreamLabGraphAudit.cpp --summary`：检查重新转换后真实算子种类。
* `tests/test_dit_layout.py`；全套53项CPU回归通过。

统一报告：`reports/levo-test-report.json`的`dit_onnx_layout_cann_2026_11_13`；
App恢复CFG记录单列`stream_lab_cfg_restored_2026_11_13`。模型、激活、音频、登录信息和原始日志不上传。

## 2026-11-18完整首block探针（已暂停）

`StreamLabDitLayout.cpp --block --memory-normal`新增真实首block的LayerNorm、调制、QKV、RoPE、Attention、残差及FFN捕获。
`tools/prepare_dit_block_probe.py`导出93节点ONNX，投影保持NCHW，输入`[2,1200,1,264]`携带256位置＋6组已计算调制＋2补齐，
输出`[2,1200,1,256]`。这里只验证首个全有效窗口，未覆盖尾窗mask、调制生成或22层完整DiT。
展开当前W8参数的浮点数学参考相对MNN Memory Normal捕获L2约`2.20e-6`，通过预设`1e-4`。

初版portable OM手机Build返回1，未执行前向；同一探针的旧首FFN控制仍成功。
GELU常量保留0维scalar后，第二版OMG转换成功（34,805,365字节），但按用户要求暂停OM路线，未做第二版手机执行。
保留失败及转换结果，不把它宣称为整DiT加速或硬件不支持的定论。
统一报告新区块`dit_complete_block_npu_2026_11_18`；当前主线为LM、CPU VAE融合与整链长时验证。
