# VAE v0.1.6 优化与真机结果

此处是完整真实 VAE decoder、固定 synthetic latent/随机音乐头的数值与性能测试，不是训练好的歌曲或听感验收。当前使用 CPU；CANN 单算子实验没有成为生产路径。

## 已实现并安装

1. 官方 MNN 转换器 `--weightQuantBits 8 --weightQuantBlock 0 --saveExternalData`，保持原图、48kHz 双声道及256帧窗口。权重337,468,424→84,569,334字节。模型名 `vae-256-w8.mnn(.weight)`，原FP32基线保留。
2. ARM82 正弦选择器的局部替换：8个FP16值一起转FP32，NEON做范围缩减和正弦/余弦多项式，再转回FP16；大参数保留libm回退。只在VAE建图时挂接，随后立即恢复选择器，避免改变LLM后续decode。固定形状会话中原MNN Unary保存新函数指针。
3. 穷举63,488个有限FP16输入，与原标量libm转FP16结果完全相同；三个完整VAE窗口相对同手机原W8 kernel的PCM L2为0。它不消除W8权重量化本身的误差。
4. v0.1.6独立按钮“连续5块 · VAE W8＋NEON正弦”；原基线保留，已有负面结论的Low/8线程组合按钮移除，历史数据保留。

## 测量范围分开

| 范围 | 基线 | 优化后 |
|---|---:|---:|
| VAE三窗，25.60秒有效PCM | 30.534秒 / RTF1.193 | 17.259秒 / RTF0.674 |
| 五块CLI，40.96秒有效PCM | 105.315秒 / RTF2.571 | 93.043秒 / RTF2.272 |
| 同组CLI五块VAE合计 | 43.202秒 | 31.419秒 |
| 签名App优化候选 | 无同轮App配对基线 | 99.579秒 / RTF2.431 |
| 签名App五块VAE合计 | — | 31.651秒 |

CLI配对三个电源检查点为AWAKE。App真实执行1024码、KV1537、5块，向量正弦1080次；峰值RSS2,297,632KiB（约2.19GiB）。不是独立GPU显存。独立模块最佳1.77倍不等于整链收益；CLI整链约快12%，App的DiT耗时比CLI高。

线程对照2/4/8线程模块RTF为1.140/0.811/0.751，均未超过6线程候选0.674；保持6线程。端点亮屏不代表严格控温、随机顺序交替或持续频率控制。

## 实际算子与kernel

256帧原图：Convolution63.3%、Deconvolution10.2%、Unary12.2%、Raster8.6%、Binary5.8%。末级张量[1,128,491520]有62,914,560元素，FP16单份120MiB、FP32单份240MiB；完整窗口MNN估算约624GFLOP，不是硬件性能计数。

32个卷积和5个转置卷积在W8文件中都有8bit权重量化元数据，但不意味着全部执行INT8。`hiperf`采样和同一binary的`llvm-nm`确认普通量化卷积命中`MNNGemmInt8AddBiasScale_ARMV86_Unit_FP16`；FP16 GEMM仍存在。转置卷积源码会取出浮点权重并转换为当前低精度格式。原图ARM82 Sin通过FP16↔FP32转换及标量`sinf`，原图采样sinf约6.85%。

仅有周期采样不能证明memory-bound，没有测得DRAM读写字节或带宽饱和。大量时间落在GEMM；激活/布局又有独立成本，应视为待进一步归因的混合瓶颈。

## CANN/NPU可行性

用真实末级卷积权重、合成激活，匹配输入/输出形状与FP16舍入权重，测试dilation1/3/9以及转置卷积，长度8192/32768。8组均有限、CANN相对FP32参考L2约0.12%–0.46%；CPU对照也通过。

全部预检`COMMON=1`，不是`OPTIMIZED=0`。例如32768点普通卷积预热后：MNN CPU约40–44ms，CANN单算子约60–62ms；转置卷积约40ms对100ms，均含输入输出复制。两套API的host接口分别FP32/FP16，不是逐字节相同的拷贝量。不能因为接口能运行就说完整VAE在NPU上更快，更不能用该结果否定尚未测量的指定NPU粗粒度图。

官方能力参考：[CANN Kit](https://developer.huawei.com/consumer/cn/sdk/cann-kit/)、[图编译与执行](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-graph-compilation-and-execution)。具体支持状态以本机SDK `CANNKit/hiai_single_op.h`和真机预检为准，不套用服务器昇腾算子表。

## 继续减少读写

优先融合Snake的乘alpha、Sin、平方、乘倒数beta、加残差，尽量在寄存器内完成；再保持卷积/激活内部布局，减少Raster。独立回调融合探针与失败观察均保留在统一报告，不将探针当成已接入的生产融合；它仍保留原图分配，不能声称降低工作区。真正生产实现应进行图重写和调度验证。

3倍目标未达到。按独立模块基线，目标约10.18秒；只优化Sin不能做到。还需更快卷积内核/正确融合的NPU粗粒度子图，并计入边界复制、格式转换和可能的CPU回退。

## 复现与交接

`tools/validate_vae_w8.py`验证PC三窗；`build-native-bench.ps1`编译专用kernel、热点、CANN匹配对照和融合探针。独立kernel参数示例：`fixtureDir low 6 high w8-vsin low`；`--sin-selftest`穷举FP16输入。

统一合并：`tools/merge_vae_optimization.py`，输出仍为`reports/levo-test-report.json`，不用发送散日志。模型与签名材料不上传。手机App的W8两文件在`haps/entry/files/three-module/vae/`；本轮已传入，新设备需另复制。单算子探针是独立CLI，不在App常规生成路径。

## 后续常量提取修正

`tools/extract_vae_snake_coefficients.py`从简化ONNX提取36组原始alpha/倒数beta，623,160字节，避免读取运行时广播展开后的大张量。融合探针改为读取snake-coefficients.json并已重新编译。首次准备真机验证时检测到SLEEP，因此未启动；该改动尚无真机等价性或提速证据，不在v0.1.6 App路径中。执行前必须保持亮屏，使用timeout 45边界，并检查原W8 PCM的L2门槛。

## 2026-10-30 亮屏补验：未通过

用户授权继续本项有界验证后，程序/系数/模型哈希核对一致。原版没有输出结构化结果；HDC传输退出0不能当作程序成功，未发现该探针对应的新cppcrash记录。

随后将融合计算从另建std::thread改为复用MNN CPU线程池，并加入阶段日志和真实退出码。前后电源检查均AWAKE：预热捕获完成、36组不可变系数加载完成，进入首个融合窗口后未完成，45.18秒后外部timeout返回124。尚未完成任何窗口的PCM等价性校验，因此没有新RTF，也不启用完整Snake融合。该结果不能用于估计一个正确融合kernel的性能。

最后阶段标记位于输入复制之前，目前无法仅据此区分输入复制、回调调度与融合内核。下一步应进一步细分这些阶段，并先在独立微基准验证kernel及布局，再做显式CPU Execution/图重写；不能把回调实现卡住解释成融合算法本身慢。原图工作区仍存在，没有内存降低证据。v0.1.6已安装App不受本CLI探针影响，已验证RTF保持不变。

统一报告条目为`vae_snake_validation_2026_10_30`，复核合并脚本`tools/merge_snake_validation.py`。
