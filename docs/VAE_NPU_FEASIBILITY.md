# VAE在Mate手机NPU上的可行性与已完成准备

2026-10-30。当前App仍推荐原W8＋NEON CPU入口。后续已通过离线OMC实际跑通四组连续子图；最新编译分区证据确认这些完整残差图含CPU子图，不能称为纯NPU。原始耗时保留，没有将App改成NPU，也没有整VAE的NPU RTF。官方W8A8校准与更正详见[CANN量化实测](CANN_VAE_QUANTIZATION.md)。

## 为什么值得研究完整子图

真实VAE只有约84.4M参数，但256帧窗口输出491520采样点/声道。完整图有32个Conv、5个ConvTranspose、36组Sin/Pow激活；单个末级[1,128,491520]的FP16张量120MiB。按原始形状估计卷积/转置卷积约623.89G MAC，若一乘加算2 FLOPs则约1.248T FLOPs；转置卷积忽略padding边缘剔除，并非硬件计数。这与MNN历史624G估算有口径差别，不能混用。

因此参数小不代表计算少。只把一个Conv移到设备侧再把120MiB张量取回，可能被布局/同步和边界数据访问抵消。CPU和NPU共用物理内存也不自动意味着跨运行时零拷贝。目标应是使连续Conv＋Snake＋残差的中间激活留在同一设备图内，最后再扩大到上采样阶段或完整decoder。

此前8组单算子接口PreCheck为COMMON=1，匹配CPU更快；它既不是全图固定NPU实验，也不能证明整个NPU路线无效。

## 已核对的本机SDK能力

本机CANN Kit `hiai_options.h` 提供：

- `HMS_HiAIOptions_SetModelDeviceOrder`：整模型设备顺序。
- `HMS_HiAIOptions_SetOperatorDeviceOrder`：按算子设备顺序；先分别测试整图NPU-only与明确的混合划分，遵守各配置组合限制。
- `HMS_HiAIOptions_SetFallbackMode(..., HIAI_FALLBACK_DISABLED)`：禁止回退。
- `HMS_HiAIOptions_SetDeviceMemoryReusePlan`：内存复用。LOW标注性能较优、内存较大，HIGH相反，须在已编译图上配对验证。
- `HMS_HiAIOptions_SetBandMode`：带宽档位，会增加功耗，不能代替减少数据访问。

NNRT头中有Conv2D、Conv2DTranspose、Sin、Pow等枚举，**这只是API可表达，不代表该型号NPU支持这组形状和精度**。静态图转换成功、NPU-only构建、真实输入推理和性能都要各自验收。

## 已实际导出、通过PC数值校验的候选

`tools/prepare_vae_npu_block.py`从原始完整VAE提取末级第一个残差块，保留真实FP32权重和全部13个节点：2×Conv、2×Sin、2×Pow、4×Mul、3×Add，不替换Snake函数。

| 候选 | NCHW形状 | 图文件 | FP16输入＋输出合计 |
|---|---|---:|---:|
| H1 | [1,128,1,8320] | 531,204字节 | 4.0625MiB |
| H4 | [1,128,4,8320] | 531,204字节 | 16.25MiB |

1D权重[O,I,K]改成[O,I,1,K]，广播、padding、dilation保持对应。每行8192有效点＋两侧各64点上下文。与原1D块的FP32相对L2约1.94×10⁻⁷；32768点连续输入与四块拼接的L2约1.95×10⁻⁷，PC ONNX Runtime已验证。

候选在忽略的`models/vae-npu-block/`，模型及测试数据不上传GitHub。使用真实权重、合成激活，不是INT8校准集或听感验收。halo证明仅适用于这一个dilation1块，不是整个VAE；多个残差块、dilation3/9和ConvTranspose都需另验证，尤其真实序列两端padding不能一概等同于一次补零。

## 在线图探针得到的负面结果

单独CLI检查Sin FP16/FP32在线图，本轮在第一个`OH_NNModel_GetAvailableOperations`调用内部SIGBUS/BUS_ADRALN退出135。自己的崩溃栈位于`NNBackend::GetSupportedOperation`/`InnerModel::GetSupportedOperations`，还没有返回有效支持列表，也没有执行推理。它是这个旧在线查询链路的失败，**不能据此宣布Sin不支持或NPU比CPU慢**。

探针不在App路径中，必须显式`--query-support`才会调用该接口；不自动重试已知异常。下一步优先使用旧CosyVoice已经采用过的离线OMC装载路线，避开这条legacy LiteGraph支持查询。旧项目只读复用，不改其包或模型。

## 后续实验顺序

1. 用匹配Kirin9030和当前系统的CANN/DDK插件，将H1/H4图离线编译OMC；先FP16，不把MNN W8元数据直接当作CANN量化模型。编译日志检查Sin/Pow/Conv实际支持和子图划分。
2. 使用公开NN core离线模型接口，选择HIAI_F并设置NPU-only、禁止回退。仅提供者名字叫HIAI_F不构成NPU证明。构建失败要保留，不改成自动CPU回退后报成功。
3. 对同一真实块和输入比较CPU与NPU：加载/编译单列；预热前后分开；输入转精度/布局、拷贝/共享缓冲、同步、输出转换全计入。H1与H4比较能否摊薄边界成本，再测设备内存复用和带宽档位。
4. 比较局部输出、拼接边界、完整decoder PCM数值及最终音质。若Sin/Pow导致切回CPU，优先判断能否用等价的设备端融合实现；不能擅自把Snake换成SiLU后声称原模型一致。
5. 扩大到连续残差堆栈/上采样阶段，最后整decoder，记账NPU工作区、CPU原模型和常驻权重副本。NPU FP16权重预计约169MB量级，不会自动保持MNN W8的85MB体积；INT8必须另做代表性激活校准及质量验证。

当前统一报告`vae_npu_research_2026_10_30`包含PC证明、候选哈希、在线查询真实退出和限制；没有NPU RTF。原CPU入口继续可用。官方能力入口：[CANN Kit](https://developer.huawei.com/consumer/cn/sdk/cann-kit/)、[图编译与执行](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-graph-compilation-and-execution)。具体API以上述本机SDK头和实际构建结果为准，不套用服务器昇腾算子支持表。

## 连续子图真机性能结果（2026-10-30）

已复用现有Linux转换工具，在独立目录将真实权重子图编译为Kirin9030 FP16 OMC。手机使用NN core离线接口，选择HIAI_F、设置NPU-only并禁用fallback；四组build及RunSync均返回成功，未调用此前异常的在线支持查询。尚未获取逐节点AI-core/AICPU硬件轨迹，不能将设备内部所有时间都称为矩阵核计算。

每组先运行一次，再运行三次，以下为后三次含输入/输出主机拷贝及同步的中位数，CPU为6线程CPU Low＋NEON Sin：

| 连续图/形状 | NPU FP16权重 | CPU FP32文件/Low计算 | CPU W8/Low计算 | NPU相对FP32参考L2 |
|---|---:|---:|---:|---:|
| 1块，[1,128,1,8320] | 19.91ms | 13.79ms | 9.41ms | 0.094% |
| 1块，[1,128,4,8320] | 59.69ms | 58.88ms | 44.06ms | 0.094% |
| 3块，[1,128,4,8320] | 220.61ms | 163.71ms | 112.41ms | 0.162% |
| 3块紧凑，[1,128,64,640] | 257.97ms | 200.13ms | 172.59ms | 0.161% |

NPU首次分别25.66/47.85/224.44/264.54ms，加载构建约35.9–213.5ms，已与前向分列。3块宽图RunSync中位数217.84ms、主机输入输出拷贝合计约2.67ms；紧凑图分别255.46/2.55ms。说明所测显式memcpy不是主要耗时，但RunSync内部仍可能包含布局/精度转换、调度/等待和设备计算，尚不能证明纯计算或DRAM带宽瓶颈。

三块包括dilation1/3/9的6个Conv、6个Sin，合计39节点，OMC约1.34MB；紧凑图约1.32MB。宽图和紧凑图均处理32768个有效特征位置，但halo额外计算分别约1.56%和25%，不能当作完全相同的填充FLOPs。合成输入的统计分布相同，数值不是训练好的歌曲。CPU FP32文件Low误差约0.31%–0.55%，CPU W8约1.05%–1.79%；不同量化/精度方案不是逐位相同的对照。

三个独立进程20ms采样RSS峰值约184/185/213MiB；这是部分模型的进程RSS，未测NPU私有工作区峰值，不能当完整共驻峰值。三块只验证了独立tile和去掉真实序列端点后的内部拼接。宽图真实端点相对误差约0.195%，紧凑短序列约0.828%，需要正确处理每层边界，尚未成为生产流式拼接实现。

更正后的结论：连续子图的离线构建和执行通过，NPU设备顺序/禁回退选项被接受，但官方编译日志显示单残差NPU3/CPU2、三残差NPU7/CPU6；已编译CPU子图不会被运行时禁回退选项移除。四组均未快于原W8＋NEON CPU。两轮端点为AWAKE；固定顺序、仅三次预热样本、未控温，不对所有NPU模型作泛化结论。

复现工具：`prepare_vae_npu_block.py`（默认/`--stack3`/`--stack3 --compact`）、`prepare_vae_npu_cpu.py 夹具目录`、远端`convert_vae_npu_blocks_remote.py --deadline 当前授权截止epoch`、`build-native-bench.ps1`及`levo_vae_npu_offline 夹具目录`。取回并校验OMC，再上传本地绝对路径；统一合并`merge_vae_npu_offline.py`。模型/原始日志不上传，统一报告新增`vae_npu_offline_performance_2026_10_30`，保留全部先前失败。

## 纯Conv消融：卷积可加速，尚非VAE端到端收益

用户在新普通额度窗口继续授权后，完成纯Conv隔离：相同真实权重、[1,128,4,8320]形状，保留bias以及kernel/dilation/padding，只删除Snake、逐元素运算和残差。导出一条六Conv串联，以及kernel7、dilation1/3/9的三个独立单Conv图。外部输入与完整残差夹具相同；中间激活和模型函数已改变，不能替代真实decoder。

默认档和显式High档各测两轮，每轮一次首次运行＋七次预热；纯Conv编译分区为NPU1/CPU0，运行时设置NPU顺序/禁回退，FP32主机接口/FP16权重。完整残差对照含预编译CPU子图，须另行区分。High选项返回成功，不等于已测到固定硬件频率。下表是两轮各自预热中位数的范围，含输入输出拷贝和同步，CPU为6线程CPU Low、W8权重：

| 纯Conv图 | NPU High | CPU W8 | 同轮加速范围 |
|---|---:|---:|---:|
| 六Conv串联 | 25.68–31.64ms | 77.18–78.91ms | 2.44–3.07倍 |
| kernel7/dilation1 | 8.62–9.00ms | 25.08–25.15ms | 2.80–2.91倍 |
| kernel7/dilation3 | 10.50–16.71ms | 25.24–25.51ms | 1.53–2.40倍 |
| kernel7/dilation9 | 11.21–13.31ms | 24.66–24.68ms | 1.85–2.20倍 |

六Conv默认档NPU33.20–42.58ms，CPU77.56–77.94ms，同轮约1.83–2.34倍。NPU纯六Conv相对该消融FP32参考L2约0.181%，单Conv约0.114%–0.116%；CPU W8约1.16%–2.33%。不同精度/量化的误差和耗时分开报告，没有歌曲音质结论。纯Conv图没有Sin，CPU正弦hook调用数为0，不把正弦优化的收益算到这些数据里。

同轮完整三残差块复测，NPU默认约215.52ms、High191.73ms，CPU W8约112.23/110.80ms；计数器补验High为192.43ms，对CPU111.59ms。故本轮不能说整个VAE被NPU加速：卷积消融明显获益，但完整块仍慢。不能简单用两者差额当作Sin耗时，删除非线性/残差还会改变中间数据、编译融合和执行计划。六Conv是否被编译器进一步融合未取得逐节点证据，三个独立单Conv结果提供了另外的卷积路径证据。

报告校验另发现统计问题：正弦selector Scope每个会话重置计数，原先减去上个会话的计数会使第二个CPU结果显示0。已改为记录重置后的绝对值，并补验CPU FP32文件与W8两路各288次正弦调用（6个Sin×6线程×8次）。此更改不改变算术或计时区间；修复前探针/源码哈希和所有原始结果保留，不把新二进制冒充旧产物。

下一步应验证卷积-only NPU＋CPU正弦/残差的真实混合数据路径，并计入每个边界的转换、同步和拷贝；或继续定位完整NPU图的非卷积算子/内部转换。转置卷积、大通道早期层、完整VAE、实际歌曲与全局padding均不能据此宣称已通过。App仍保留原CPU W8＋NEON入口，原五块VAE30.19秒/整链RTF2.290没有因本轮改变。

纯Conv复现：`prepare_vae_pureconv.py`→`prepare_vae_npu_cpu.py models/vae-npu-pureconv`→离线转换→`levo_vae_npu_offline 夹具目录 default 8`或`high 8`；`merge_vae_pureconv.py`写入唯一公开报告的`vae_pureconv_ablation_2026_10_30`。第一轮因旧额度94%未启动的情况由本次新窗口实测推进，未绕过旧窗口停止线。
