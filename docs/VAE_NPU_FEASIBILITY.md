# VAE在Mate手机NPU上的可行性与已完成准备

2026-10-30。当前App仍推荐原W8＋NEON CPU入口，本轮没有将App改为NPU；没有新的NPU VAE速度成绩。

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
