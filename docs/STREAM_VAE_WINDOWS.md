# 当前流式实验的 VAE 窗口验证

2026-11-13，使用手机现有 LarkStudio 0.6B 包的共享 VAE，独立 CPU CLI 运行。
本轮没有修改或安装 App，没有改变原模型，也没有实现新的 Snake 融合内核。
当前 App 的默认窗口与生命周期保持不变。

## LLM 生命周期与两种上下文

当前一轮只有一个 LLM、KV、DiT、VAE。LLM 一次 prefill 513 行，首窗得到256个音乐码，
后续每轮增加192个。LLM权重与KV跨窗保留；DiT/VAE计算期间LLM暂停decode，
不是三个模型同时计算。60秒1500码的历史实测最终KV为2012位置。
此前下一窗token预取没有加速，因此默认关闭；本轮不重做该并发实验。

DiT当前256帧窗口、192帧步长，重叠64帧/2.56秒。它负责生成连续的latent上下文。
VAE的卷积边界上下文是另一个问题：本轮验证16帧/0.64秒可保护指定裁剪区域，
不意味着可以把DiT重叠、PCM交叉淡化、VAE上下文全部统一改成16帧。
单份非因果裁剪输入的上下文测试，也不是有状态连续解码器的完整验收。

## 输入与对照

从手机已完成的60秒生成保存的1500个真实采样code出发，保持原22层DiT、2步、
无CFG混合、batch2、seed2307和原归一化，重建第一窗与第八窗latent。
重建第一窗经完整VAE解码后的PCM16，与App实际首段WAV的368640个双声道采样点逐字节一致。
本诊断不运行LLM，不用随机code或随机权重代替这份输入。

实际尾窗从第1344帧开始，有效156帧，完整图仍输出256帧。
CPU Low、6线程、Memory Low、原W8/g64、原NEON正弦保持不变。
每组独立会话一次预热、两次计时；完整基线在组首/组尾重复，以下不含模型setup。

| 实际尾窗方案 | 两次平均秒 | 比较区域 | 相对L2 |
|---|---:|---|---:|
| 完整256帧，组首 | 5.735 | 全窗 | 0 |
| 有效156＋右侧16帧，共172帧 | 4.151 | 原窗0–156帧 | 0 |
| 从48帧开始，共208帧，含左侧16帧上下文 | 4.791 | 原窗64–256帧 | 0 |
| 完整256帧，组尾 | 6.055 | 全窗 | 0 |

最后一行显示约5.6%的轮内速度漂移，不能把顺序测试当作严格恒温对照。
确认轮前后均AWAKE，Last Screen Off记录未变化；整轮64.44秒，20ms采样峰值RSS750.26MiB，
最低系统可用4327MiB。这是独立VAE进程，不是三模型共驻的整链峰值。
整批探针前电池27°C，确认轮后34°C；电池温度不等于CPU结温，也没有功率实测。

首窗的补充验证：直接截到156帧，PCM L2约5.46%，失败；保留右侧8帧后约0.0442%，
通过预先设定的0.1%门槛；保留16帧则指定区域L2为0。
因此暂以16帧作为候选，不把8帧结果当作普遍无损保证。

172帧只优化最后一块；208帧只验证指定新增区域。
完整流式接入仍须处理前64帧PCM复用、边界交叉淡化和逐窗缓存，不能据此宣称整链已提速。

## 同一会话 resize

直接沿用App当前的`releaseModel()`策略后再resize，输出形状没有更新，探针安全失败。
本机MNN源码`Interpreter::resizeSession`明确在model buffer已释放时返回，不能忽略这个限制。
独立保留190720字节的VAE图结构、外部权重不变后，同一会话256→172重规划3.210毫秒，
恢复256帧2.525毫秒；没有创建第二个VAE。

真实尾窗三次172帧前向总计3.572/3.714/3.748秒，平均3.678秒，指定有效区域PCM L2均0；
本轮完整窗首次5.297秒，恢复后5.857秒且全窗L2为0。20ms采样峰值746.68MiB，
运行结束点检查最高750.49MiB；采样峰值不等于每个瞬间的绝对峰值。
小图保留是这个候选接入所需的生命周期调整，不是重复加载完整权重。
实际只缩短最后一窗，预计整链只省约1–2秒，还不是新的App整链RTF实测。

## 当前模型热点

同一份第一窗输入，单独debug会话预热后采集一次MNN算子回调：

| 算子 | 回调时间占比 |
|---|---:|
| Convolution | 51.99% |
| Deconvolution | 16.26% |
| Raster | 12.43% |
| UnaryOp | 11.35% |
| BinaryOp | 7.97% |

前向约5.596秒，输入约0.036毫秒，输出读取与复制约4.36毫秒。
这说明外部输入输出复制不是主要成本；Raster及卷积内的访存仍存在。
回调不是DRAM带宽计数，不能据此宣布整个VAE是memory-bound或compute-bound。
最慢三个普通卷积均输出`[1,128,491520,1]`，各约0.321/0.292/0.260秒。

后续优先级：真正消除Snake中间张量的图级融合、末级卷积kernel及布局优化、
独立VAE流式缓存。旧回调融合曾在App中变慢，不能直接重新启用或宣称本轮已完成融合。

## 工具与范围

`native-bench/StreamLabVaeWindow.cpp` 为独立入口，不被App包含。
构建目标 `levo_stream_vae_window`，链接与App相同的MNN归档和ARM82 ABI配置。

```text
levo_stream_vae_window BUNDLE REAL_CODES_JSON OUT prepare
levo_stream_vae_window BUNDLE REAL_CODES_JSON OUT windows
levo_stream_vae_window BUNDLE REAL_CODES_JSON OUT profile
levo_stream_vae_window BUNDLE REAL_CODES_JSON TAIL_OUT prepare-tail
levo_stream_vae_window BUNDLE REAL_CODES_JSON TAIL_OUT confirm
levo_stream_vae_window BUNDLE REAL_CODES_JSON TAIL_OUT reuse
levo_stream_vae_window BUNDLE REAL_CODES_JSON TAIL_OUT reuse-kept
```

工具仅用于当前已核对的256帧、64通道、48k双声道VAE和256–1500个code；
`confirm`中的156帧裁剪目标对应本轮60秒尾窗，不是任意长度的通用部署器。
`reuse`保留releaseModel后的resize失败，用于验证限制；`reuse-kept`测试保留小图buffer的路线。
单一汇总位于`reports/levo-test-report.json`的`stream_lab_vae_windows_2026_11_13`。
原始日志、code、latent、PCM及模型保留本地，不上传。
