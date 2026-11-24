# 音乐LLM的FP16／FP32运行时对照

2026-11-24，CMM-AL00，当前`larkstudio-base-v2`的36层音乐Qwen；不是通用语言模型或0.6B。

## 为什么原模型FP16，端侧仍走FP32

原始checkpoint精度、部署权重位宽、运行时浮点精度是三个不同设置。
当前部署仍为主体W4/group64、头W8，不需要恢复整份FP16权重才能测试FP16计算路径。
`runtime/include/levo/experimental/StreamLab.hpp`的`Prior`沿用`precision=high`基线。
本地固定MNN的CPUBackend只有在`Precision_Low`且硬件支持FP16算术时选择ARM82后端；
High默认CPU浮点核心为4字节。模型原始FP16不会自动覆盖这一运行时配置。

这里改变的是整个LLM的CPU精度配置，包括主要线性层、Norm、Attention及未量化KV，
不是只改KV或只改反量化。W4／W8权重、输入BF16 embedding、动态A8选项和词表均不变。
量化矩阵乘法可以直接融合缩放／输出转换，不代表把整个W4模型常驻展开为FP32。

## 实机执行证据

单独审计进程，513位置prefill＋2次decode；回调和计数钩子不混入性能数据。

| 审计项目 | CPU High | CPU Low |
|---|---|---|
| Attention、Convolution、LayerNorm的实际后端 | CPU=0，coreBytes=4 | CPU Extension=13，coreBytes=2 |
| W4矩阵乘内核实际调用 | `Int8GemmKernel_W4`，15120次 | `MNNGemmInt8AddBiasScale_w4_Unit_FP16`，15120次 |
| W8头对应内核调用 | `Int8GemmKernel`，12次 | `MNNGemmInt8AddBiasScale_Unit_FP16`，12次 |
| RoPE、Cast | CPU／4字节 | 仍为CPU／4字节 |

计数是内部tile调用，不能解释为模型层数。Low确实命中FP16量化矩阵乘路径，并非只改配置名称。
FP16表示这些路径的浮点存储／输出；不能由此声称所有中间累加都是FP16。

`attention_mode=8`在此源码中表示Flash Attention开关＋无KV量化，不是KV8。
CPUKVCacheManager的未量化存储字节数取自Attention CPU核心，因此上述路径对应FP32／FP16 KV。
这是实际Attention后端观察结合源码的结论；没有另行钩住KV缓冲区逐字节审计。
相同数量的未量化KV，FP16原始数据量为FP32一半；不能据此说整条W4链路的总DRAM流量减半。

## 同历史性能与数值

使用完整120秒App测试保存的真实`513×2560`前缀和前64个音乐code。
每个code输入为`1+code`，随后读取对应最后层hidden，即consume对齐。
CPU4线程、Memory Low、动态选项0、attention_mode8、hidden输出开启、KleidiAI关闭。
每种精度独立进程测试两次：High→Low，审计进程后Low→High；两次结果平均如下。

| 指标 | High／FP32主体浮点路径 | Low／FP16主体浮点路径 |
|---|---:|---:|
| Prefill，513位置 | 13480.73ms | 13722.73ms |
| 纯decode，64次均值 | 103.29ms/token | 98.36ms/token |
| 两轮decode范围 | 101.60–104.99ms/token | 93.79–102.93ms/token |
| 两轮最大进程RSS | 2519.77MiB | 2396.71MiB |
| 最终KV位置 | 577 | 577 |

观测decode耗时下降4.77%，prefill反而增加1.80%；RAM峰值少123.06MiB。
两组速度范围重叠，只有两次独立测量，不能宣称稳定获得同等加速，更不是120秒整链RTF。
测试期间观察手机保持AWAKE且末次息屏记录不变；电池观测32–36°C，没有同温随机交叉控制。
不把CLI进程的调度／内存数据替代App前台长链结果。

同一精度两次运行的logits和hidden分别逐字节一致。
Low相对High的logits L2为5.97%，65行argmax一致率90.77%；
64个consume-hidden的L2为6.74%，最大逐行L2为12.78%，平均余弦相似度0.997689。
两种精度全部有限，但它们均使用已经W4／W8量化后的权重；High不是原始checkpoint的准确率金标准。
这个差异不能单独证明哪条更接近原FP16模型，也不能代替音乐试听／下游DiT验证。

## 复现与当前决定

新增独立目标`levo_stream_lm_precision`，链接与App相同的固定MNN静态库，未更改App的默认精度。

```text
levo_stream_lm_precision MODEL_ROOT PREFIX_F32 CODES_JSON NEW_OUTPUT_DIR high|low perf|audit|long DECODE_COUNT
```

长上下文模式为`long`，最多3000次decode；KV持续增长，不按窗口清空。
每250次输出一段计时和RAM记录，只稀疏保留logits／hidden快照，避免保存全部输出显著放大RAM。
它仍使用相同的真实code历史，未运行下游DiT／VAE，也不生成120秒音乐。

`MODEL_ROOT`须为相同36层音乐bundle，前缀是该bundle实际准备的513×2560 float32输入，
codes是实际音频code历史。输出目录必须不存在；perf最多256次decode，audit最多4次。
性能测试禁止audit，logits与hidden在同步计时外保存；hidden复制另行计时。
只公开探针、文档与统一报告，不提交权重、前缀、code文件或音频。

High生产基线保留；v0.1.16已另加可选的Low整链实验，不改默认配方。
该可选配方已经完成[120秒App实测](STREAM_ALL_LOW_120_20261124.md)：RTF6.851、RSS峰值3.62GiB；
它不是上文64-token短测的外推结果。
FP16已证实可运行并省RAM，原FP16权重参考或同输入条件到音频仍需验证，
再决定默认精度或只对部分算子使用FP16；不把配置切换直接当成完整精度与音质验收。
统一记录：`reports/levo-test-report.json`中的`stream_lab_large_lm_precision_2026_11_24`。
