# CANN官方VAE W8A8校准与手机实测

2026-10-30。官方DOPT量化、Kirin9030 OMC编译与手机执行均已通过。尚无VAE端到端或整链加速验收，App仍使用原CPU W8＋NEON入口。统一结果在`reports/levo-test-report.json`的`vae_cann_quantization_2026_10_30`，原始模型、校准数据、OMC和签名均不上传GitHub。

## 实际量化流程

- 工具来自已经校验的官方`DDK-tools-next-6.1.1.0.zip`，SHA256为`87d7e3f186ad5c527a9385cea555559ea53c63b87dc483820523bcf7bf6f87e5`。此前只解压OMG，不能据已解压目录断言DDK没有DOPT。
- 使用`tools/tools_dopt/dopt_onnx_py3/dopt_so.py`，独立CPython3.10.22。Python3.12曾报`_PyUnicode_Ready`，换匹配运行时后解决。
- 官方`Quant_INT8-8`默认非aggressive策略：卷积权重`AbsMax`，输入激活`EmaMinMax`。单残差两层Conv量化，纯卷积对照一层Conv量化；Snake、残差与整个VAE并未全部INT8化。
- 校准二进制采用官方`bin_data_preprocessing.py`格式：little-endian uint32魔数510、四个NCHW维度、连续FP32数据。目录和无头原始二进制均被工具拒绝；补头后日志确认20/20。
- OMG输入DOPT输出ONNX＋`--compress_conf`生成的配置。`--weight_data_type=FP16`只是其余浮点权重设置，单独改这个参数不能得到INT8。

官方流程：[Model Size Reduction](https://developer.huawei.com/consumer/en/doc/hiai-guides/overview-0000001053005655)、[ONNX量化示例](https://developer.huawei.com/consumer/en/doc/harmonyos-guides-V13/hiaifoundation-examples-V13)。实际命令、配置哈希与编译日志摘要保存在统一报告。

## 校准覆盖与留出验证

用原始FP32 VAE真实权重前缀，从既有合成latent流抽取20个不重叠的16帧窗口，取末级残差入口激活。另留4个不参与校准的窗口。H1形状`[1,128,1,8320]`；H4将四个独立H1窗口按行合并，卷积kernel高度1、行间无相互卷积。H4仍只有20个不同校准来源，不能计成80条独立样本。纯Conv输入先执行它真实的前置Snake。

这是合成latent上的数值与性能试验，16帧前缀重置、同源留出和有限幅度分布限制了代表性。没有真实歌曲质量结论，也不替代上线音频校准。两种H4参考图与既有CPU模型源ONNX输出L2均为0。

## 真机结果

每组8次，首次单列，下面是后7次的中位数。H4两轮交换FP16/W8A8顺序，含主机拷贝与同步。CPU为MNN CPU Low、6线程、W8权重、NEON正弦。手机端点AWAKE，未控温/锁频。

| 子图 | CANN W8A8 | 同轮CPU W8 | W8A8相对FP32参考L2 |
|---|---:|---:|---:|
| H1真实残差，4条留出 | 11.06–12.28ms | 7.59–7.71ms | 0.389%–0.406% |
| H4真实残差，两轮 | 46.25–47.01ms | 39.45–41.85ms | 0.396% |
| H4单K7纯卷积，两轮 | 12.26–12.52ms | 24.70–24.78ms | 0.980% |

纯卷积W8A8比同轮CPU约1.98–2.01倍。FP16纯卷积两轮7.16/20.24ms，残差77.95/45.86ms，波动明显，不能据此宣称W8A8稳定优于FP16。W8A8全数有限并不等于音质验收。单残差OMC约467596字节FP16、470572字节W8A8；文件没有缩半，不能用文件大小判断内部算术精度。

## 重要更正：静态CPU/NPU分区

官方编译日志明确：

| 子图 | NPU子图数 | CPU子图数 |
|---|---:|---:|
| 单残差，FP16或W8A8 | 3 | 2 |
| 旧三残差，宽/紧凑 | 7 | 6 |
| 单卷积/六卷积 | 1 | 0 |

两个`Sin`节点被NPU算子库拒绝，随后完整残差被分区。运行时指定NPU设备顺序及关闭fallback均返回成功，**这并不会消除OMC里预编译的CPU子图**。此前“NPU-only运行”的表述应改为“接受NPU请求/禁回退选项的离线混合图”。原始数据保留，历史两节报告新增`executionPlacementCorrection`。纯Conv有NPU1/CPU0编译证据，仍未取得硬件逐kernel轨迹。

## 时间具体花在哪里

H4 W8A8真实残差两轮的分项预热中位数：

| 区间 | 耗时 |
|---|---:|
| 主机输入memcpy | 0.81–0.89ms |
| `OH_NNExecutor_RunSync` | 44.68–45.42ms |
| 主机输出memcpy | 0.81ms |
| 整次执行 | 46.25–47.01ms |

各列独立取中位数，不能严格逐列相加。显式拷贝约1.6–1.7ms，绝大部分时间在RunSync内部。它包含实际Conv、CPU正弦、子图依赖同步、量化/反量化及可能的布局转换；目前没有逐节点计时，不能把这44–45ms全部称为调度开销，也不能用纯Conv与残差耗时差直接估算正弦成本。

当前FP32接口每份H4张量16.25MiB，FP16为8.125MiB。统一内存仍有缓存一致性和同步成本，但未测到内部传输字节数或带宽，不能据此认定memory-bound。下一步应优先取得子图级时间线，再比较减少CPU正弦边界或保持设备张量的方案。

## 复现入口

本地：`prepare_vae_cann_calibration.py`→`pack_vae_cann_calibration.py`。远端隔离环境解压官方ONNX DOPT，用`run_vae_cann_quant_remote.py --deadline 当前授权截止epoch --input-mode official`；`convert_vae_cann_quant_remote.py`使用同一截止参数。H4使用`prepare_vae_cann_h4_remote.py`及上述两脚本的`--case residual-d1-h4`或`--case pureconv-k7-d1-h4`。

取回带哈希的产物后用`prepare_vae_cann_phone.py`、`prepare_vae_cann_h4_phone.py`构建夹具，`levo_vae_npu_offline 夹具目录 high 8`执行。`merge_vae_cann_quant.py`核验20/20校准、FP32图一致性、手机文件哈希、有限值、真实退出和编译分区，再写唯一公开报告。原App五块RTF2.290、VAE30.19秒仍是原CPU路径成绩。
