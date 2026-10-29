# 签名App三模块内存与整链计时（2026-10-29）

本轮按用户要求允许未训练权重，验证真实模块加载、前向、连接与生命周期。包名`com.zhilee.levoharmony`，v0.1.4已由用户签名，使用HDC安装/启动，没有电脑界面自动操作。旧ACE和CosyVoice工程未改动。

## 实际链路

- 26层真实Qwen3裁剪骨干，加随机初始化的音乐embedding、32类风格embedding和16,385维音乐头；合计约0.598B参数，未恢复/音乐训练。
- 固定合法歌词ID512个＋一个风格向量＋音频BOS，共514位置prefill。新音乐ID通过独立embedding反馈，保留KV，自回归得到256个ID；末次KV为769位置。本测试固定长度并抑制EOS，不代表歌词内容已被唱出。
- 真实Mixed RVQ16384×32和out projection1024构造条件；完整39层、hidden1200、FFN4800的DiT，CFG1.5、batch2、256帧。用户假设两步Euler，本轮不按官方20步评价音质。
- 真实VAE CPU Low把256帧latent变成48kHz双声道491,520采样/声道，即10.24秒PCM。PCM仅做有限值/RMS检查，没有播放或保存可听歌曲。

LLM：线性层W4/g64、音乐头W8、embedding BF16；压缩线性权重248,296,682字节，独立embedding344,788,992字节。DiT W8权重722,214,236字节；W4/g48为462,169,752字节。48整除主要1200/4800维度，时间嵌入等不整除维度仍可回退到逐通道。VAE磁盘FP32，CPU Low内部精度；不是VAE W4。

## 真机数据

从后台任务开始到PCM完成计时，包括模型初始化/输入准备/prefill/decode/DiT/VAE；排除随后独立的数值参考前向和1秒释放观察。进程均从全新App启动，操作系统文件缓存未清除；不是存储冷缓存测量。

| 策略 | 整链RTF | 20ms采样流程峰值RSS | 说明 |
|---|---:|---:|---|
| W8，阶段完成就释放 | 2.489 | 909MiB | 514前缀、256码、DiT两步、VAE |
| W8，保留LLM与KV，DiT完成再释放 | 2.841 | 1,310MiB | LLM本次decode较慢；不能据此认定保留本身导致变慢 |
| W8，三模型会话全部共驻 | 2.572 / 2.395 | 2,074 / 2,070MiB | LLM、完整DiT、VAE均完成前向 |
| W4，三模型会话全部共驻 | 3.026 / 2.605 | 1,815 / 1,810MiB | 权重更小，本轮完整DiT两步反而更慢 |

W8共驻两轮DiT为7.871/6.993秒，W4为9.456/9.409秒。W4本轮降低约255–265MiB进程峰值，没有测得提速。W8最快共驻一轮：prefill2.435秒、decode3.926秒、DiT6.993秒、VAE8.826秒；总24.528秒/10.24=2.395。每阶段余量门槛为系统可用量≥1GiB，部分加载门槛1.5/2GiB；另设置5GiB RSS诊断停止线，这不是测得的系统限制。六轮最低系统余量约5.4GiB以上。

短前缀138位置单独测试：prefill0.621秒、decode48.85 token/s；514位置为prefill2.344秒、decode64.97 token/s。输入短明显降低本轮prefill耗时，但decode没有随前缀单调变化。同配置长前缀多轮decode约33–67 token/s，存在调度/运行条件波动，不能将此前CosyVoice约85 token/s与一次LeVo约42.6直接解释为固定硬件差距。

签名App独立VAE基线：High模块RTF2.476，峰值约1,638MiB；Low为1.224，约907MiB，均为3窗有效25.6秒。它与上述整链单窗/6线程/Power High的测试范围不同，不混算。

## 数值与验收边界

完整DiT已严格加载526个CFM张量，ONNX逐个覆盖真实0–38层；不是重复block0。整网实数RoPE改写L2约4.68e-7，ONNX对PyTorch约1.11e-6。外部权重ONNX避免protobuf2GB限制，先简化无权重固定形状图，再附真实外部initializer。

PC单次velocity相对PyTorch：MNN W8约3.93%，W4约16.70%。手机对PC同量化参考约2.65%（W8），W4亦未通过1%严格门槛；具体值保存在统一报告，不能当高保真或音乐音质验证。第一轮因该门槛返回失败的原始结果保留。后续按用户明确的“未训练也可，只看流程/内存”要求，分别记录`flowAndMemoryPassed`与`strictNumericalParityPassed`；前者通过不改变后者失败的事实。

真实App身份下，HiDebug返回的RSS/VSS limit是远大于物理RAM的哨兵值；`reportedLimitAvailable=false`，不能据此宣称无限可分配内存。20ms采样也可能漏过瞬时峰值。本轮证明在当时条件下一个10.24秒窗口可完成，并未验证4000码KV、多分钟常驻、跨窗继续decode、后台运行或播放器有界队列。

## 复现与报告

1. 使用独立依赖环境运行`tools/export_diagnostic_student.py`；官方MNN导出器源码从旧工程只读复用，新增yaspin/termcolor仅装在本工程`.deps`。不要设置生产学生的`music_trained=True`。
2. `tools/export_full_dit.py`导出完整W8；W4通过MNNConvert `--weightQuantBits 4 --weightQuantBlock 48 --saveExternalData`转换同一外部ONNX。
3. `tools/prepare_three_module_bundle.py`和`tools/add_w4_chain_variant.py`生成本机`models/three-module-bundle`，含26个资产及bundle.json；模型不提交GitHub。不要覆盖已完成的模型输出。
4. 用户在DevEco配置签名后，确认default产品关联本地default signingConfig。`build.ps1 -WithMnn`构建签名App。签名配置/私钥留本机，禁止`git add .`。
5. 本次设备的`UIAbilityContext.filesDir`对应`/data/app/el2/100/base/com.zhilee.levoharmony/haps/entry/files`，不是应用级`base/.../files`。诊断包放在该目录的`three-module/`，原VAE12文件放在`vae-stream-256/`，所有者必须与App匹配。
6. 运行`tools/run_signed_chain.ps1 -Policy serial|keep-llm|all|all-w4 -RunId 唯一ID`，通过明确Want参数触发后台任务，不模拟点击。附加`llm-short|llm-long`只测LLM。超时/退出不自动重启，检查原PID与`last-chain-progress.jsonl`。
7. `tools/merge_signed_chain_validation.py`验证模型层覆盖、文件SHA256、KV长度、阶段顺序、RTF和严格误差标记，合并到唯一公开的`reports/levo-test-report.json`。使用本机已有ONNX依赖路径；不上传碎日志、权重或签名。
