# LeVo Harmony：给接手Agent的模型适配指南

> 2026-10-30后续“继续测试”已完成Snake定位与三窗口配对：安全回调退出、C8实际布局/归档ABI匹配、真实producer快照修复后，36组×3窗全部PCM L2=0。同轮W8＋NEON24.803秒→融合21.348秒（25.6秒有效PCM，模块RTF0.969→0.834，1.162倍）；峰值约919MiB，含最多一份120MiB快照。未接入App，原工作区保留；本轮绝对值慢于历史NEON0.674，不是新最好整链RTF。旧timeout已定位为回调异常绕过引擎清理；失败历史保留。下一步考虑显式Execution/图重写去快照，但固定归档PLUGIN=OFF，尚未实现。详见VAE_OPTIMIZATION.md及统一报告vae_snake_progress_2026_10_30。MNN另有Lookahead/MTP/EAGLE/DFlash，当前音乐循环未启用，见SPECULATIVE_DECODING.md；不要把未训练音乐头的接受率当歌曲指标。

> 2026-10-30限定补验：用户只授权继续亮屏后的本项有界Snake验证，不推断为重新授权整窗自动试验。原不可变常量探针无结构化结果；改用共享MNN线程池并加阶段/实际退出码后，AWAKE前后检查下45.18秒返回124。预热与36组系数加载已完成，首个融合窗口未完成，尚未区分输入复制/回调/内核，不能报新RTF或宣称融合无收益。统一报告vae_snake_validation_2026_10_30与VAE_OPTIMIZATION.md记录完整证据。App源码/已安装v0.1.6保持原状；原版/共享池进程均确认终止。下一步应做独立kernel等价性与更细阶段检查，再考虑显式Execution/图重写。

> 最新v0.1.6：先看[VAE优化结果](VAE_OPTIMIZATION.md)。手机新入口连续5块VAE W8＋NEON，量化卷积采样命中ARMV86 INT8→FP16；转置卷积仍有FP16计算。仅在VAE建图期间挂接正弦selector，立即恢复，5块命中1080次，不能把LLM的Sin也改掉。63,488个有限FP16正弦输入逐一对齐原kernel，三窗PCM对同手机W8 L2=0；W8本身对PyTorch约6%，未听测。CLI整链2.272，签名App2.431，生产music_trained仍false。CANN单算子8组COMMON=1，匹配CPU更快；并非全图指定NPU性能结论。Snake回调融合首轮捕获未完成已停止独立PID；修正后45秒有界重测在系数捕获门槛停止，未得到融合性能，不接入App。未来应从原图提取有界常量并做真实图重写，不能从广播后的巨大callback Tensor盲目复制系数。

更新时间：2026-10-30。本文按更新顺序保留历史，下面早期“最新/未完成”须结合顶部新记录；不代表未完成项已验收。

> 最新v0.1.5五块连续测试已完成：同一LLM KV跨codec暂停/恢复，256新码＋4×192新码=1024，KV终态1537；同一DiT/VAE会话，64帧latent context和有界PCM交叉渐变。有效40.96秒，113.107秒，RTF2.761，峰值2324MiB。详见FIVE_CHUNK_VALIDATION.md和统一报告`five_chunk_validation_2026_10_29`。目标1.5尚未实现，不能假定只优化DiT就够：把DiT38.015秒全部去掉，其他阶段不变仍RTF约1.83。

> 新Low/8线程候选：独立DiT CPU Low有限、约3.0秒/单步，不能据此承诺整链更快。首轮观察超时、PID40681/SLEEP，用户亮屏后同一任务已完成；1038秒包含长暂停，不能用于性能排名。确认该任务终止后再独立前台复测R2，141.041秒/RTF3.443、2266MiB，比基线更慢，组合不启用，不要再等待旧PID。MNN Transformer融合+C4转换失败；C4关闭隔离也退出0xC0000409且无模型，保留失败路线。

> LLM Kleidi开关配对也完成：本机库MNN_KLEIDIAI=ON、DEFAULT_ON=OFF，llm.cpp已有`cosyvoice_kleidiai`补丁，静态库包含此字符串。同一CLI/514前缀/256输出，关闭55.40 token/s，开启52.90，无decode提速，未确认逐层实际选核，不启用默认路径。`LevoRunChainBench(...,"llm-kleidi")`仅增加CLI探针；App仍保持现有测试入口。CLI会覆盖通用进度文件，之后判断任务须结合App lastRunId与PID。

> 最新v0.1.4已实际签名安装：26层真实Qwen体W4/g64＋随机音乐头W8/embedding BF16→完整39层DiT W8或W4/g48两步→VAE CPU Low，App后台NAPI跑通单窗10.24秒。阶段释放峰值909MiB/RTF2.49；保留LLM为1310MiB/2.84；全部共驻W8为2070–2074MiB/2.40–2.57，W4为1810–1815MiB/2.60–3.03。KV验证769位置。PC完整DiT W8/W4相对FP32约3.93%/16.70%，手机同量化对照未过1%严格门槛；`flowAndMemoryPassed`不等于`strictNumericalParityPassed`。首轮严格失败保留。详见THREE_MODULE_VALIDATION.md、KERNEL_REUSE_REVIEW.md及统一报告`signed_three_module_validation_2026_10_29`。不许把随机头输出当歌，不许打开生产模型music_trained开关。

> 签名与路径：用户生成default signingConfig后未关联default产品，本轮只在本地`app/build-profile.json5`补关联；该文件现在含本地签名配置，绝不可提交。IDE生成文件原样保留。真实模块filesDir为`.../haps/entry/files`，早先应用级files路径错误导致的失败已查清。诊断包26资产＋bundle.json已在手机three-module/，模型未上传。测试入口Want参数由run_signed_chain.ps1设置；不使用computer-use。HiDebug返回超大限额哨兵，已标不可用，不是无限内存。

> 最新完整DiT热点：W8 FFN41.8%、Raster13.7%、Binary14.1%、QKV12.4%。旧CosyVoice手动Kleidi W8A8 i8mm FFN尚未迁入LeVo；旧1024/2048/22层不同于本1200/4800/39层。复制旧实现会重复保留FFN权重，迁移先做真实FFN down核和内存记账。两个模型所有新测试运行CPU，不是NPU/GPU。

> 亮屏补测完成：同一VAE Low/6线程/Power High按64→256→256→64测试，四轮端点均AWAKE。64帧RTF1.220/1.211、峰值约348MiB、首次数值包2.77/2.64秒；256帧1.119/1.166、约814MiB、9.55/10.79秒。历史SLEEP启动同配置分别1.250/1.145，不能据此声称亮屏大幅提速。全部文件校验、逐块误差与历史一致，统一报告`awake_vae_validation_2026_10_29`。仍是CLI VAE模块，不是整链/App/扬声器成绩。用户产生的app/.idea、oh_modules和lock文件留在原处，未作为本次测试提交。

> 最新完整codec：Mixed推理权重子集已下载完成并核对自身SHA256（未下载整个原文件，不能声称原文件全量SHA已验证）。526个CFM张量严格加载，完整751,534,224参数；真实VAE严格加载。PC两个256帧窗口、64帧上下文、每窗20步通过，PCM有效17.92秒且全部有限。首窗DiT47.141秒/VAE6.414秒，次窗45.587秒/5.641秒；不是手机性能。统一报告`full_codec_cpu_2026_10_29`保存结果和来源。`tools/validate_real_codec.py`复现，`tools/merge_full_codec_validation.py`合并。注意普通CPU构造保留非持久attention buffer；meta构造后仅assign权重不足。PCM拼接采用非原地操作，避免inference tensor限制。仍缺全DiT的MNN导出、手机全链与共驻验证、训练好的0.6B音乐头和有界播放器；不要将测试ID输出称为歌曲。下方下载中描述均属历史状态。

> 后续生命周期实验：真实26层Qwen骨干已保存/重载，缓存检查通过，总学生598,178,304参数；`layer_types`同步裁剪的保存问题已修复。新增可暂停音乐token迭代器，保留KV后可做codec分块再继续，未训练生成仍受保护，15项小型回归通过。64帧VAE候选峰值约348MiB；6线程Power High时模块RTF约1.25，256帧同配置约1.15/814MiB。启动时观测SLEEP，不能当受控前台成绩或整链实时性。统一报告新增`lifecycle_experiments_2026_10_29`，详见分块说明。

> 最新v0.1.3：Qwen完整权重已核对官方SHA256并真实加载；512歌词＋style＋BOS共514位置prefill及固定8码增量KV与全序列对齐。真实VAE全部182参数已CRC校验、严格加载、ONNX/MNN及手机三窗对齐。手机VAE Low模块RTF约1.21、峰值RSS813MiB，尚非实时整链。新增[分块/量化/生命周期说明](STREAMING_AND_STUDENT.md)与[手动安装说明](INSTALL_MANUAL.md)。App已构建但未由Agent签名安装，用户明确不要computer-use。以下下载未完成描述为历史记录；完整Mixed DiT仍在续传，带当前授权窗口硬截止，不自动跨窗口。

> 2026-10-29收尾更新：见[CPU验证记录](CPU_VALIDATION.md)和统一JSON。Mixed DiT构造已恢复并通过526参数形状检查；真实单层、ONNX/MNN FP32已数值对齐。以下早期状态中“尚未修复构造”已被这项结构验证推进，但完整权重及整曲生成仍未完成。下载分块约Qwen1,040MB、VAE159MB；DNS中断后保留，按原窗口规则没有自动重启实验。

> 同日真机更新：用户恢复授权后完成Mate 80 Pro CPU单层32/256帧测试，最终9组均通过。256帧W8预热53.4ms、采样峰值RSS56.1MiB、单层L2约0.962%；W4为58.4ms/47.7MiB/7.91%。输入/图/权重/程序的PC与手机SHA256一致。这是HDC单层CLI，不是App整链或歌曲RTF。优先W8继续完整权重适配，尚不能宣称全模型共驻或音乐头已训练。复现见CPU验证记录；统一报告版本为0.1.2-phone-block-validation。

## 1. 用户目标和工程边界

- 本地工程：`C:\levo-harmony`；公开仓库：`https://github.com/zhilee2023/mobile_harmony_AImusic`。
- 目标设备：Mate 80 Pro 16GB，用户系统版本HarmonyOS 7.0.0.32 SP22；构建SDK26，ARM64。
- 保留 `C:\cosyvoice-harmony`、`C:\cosyvoice-ios`、`C:\ace-step-mnn-ios`，不要重命名、覆盖或移动。
- 输出是**人声＋伴奏的混合歌曲**，取消独立双轨LM，不是只唱人声。
- 输入：最多512个歌词分词token＋恰好一个风格前缀向量。音乐输出长度独立控制。
- 优先MNN CPU。先PC权重与数值对齐，再手机短音频；不先承诺实时或RTF 0.85。

## 2. 已验证和未验证

已验证：

- 原创学生模型原型、teacher forcing、增量KV解码与全序列一致性。
- 已真实加载完整Qwen骨干、完成实际512-token缓存验证；其音乐头和风格参数仍未训练。
- 8项微型模型单元测试通过；60步合成ID训练loss从约3.014降到0.164。它只验证优化器路径，不代表学会音乐。
- 完整meta模型参数：28层学生629,640,192；其中Qwen骨干596,049,920。
- 可选移除层9、18并重排KV索引，26层学生598,178,304参数。选层未经重要度校准，必须恢复训练。
- 独立鸿蒙诊断App构建成功，包名`com.zhilee.levoharmony`；HAP未签名。
- `MnnGraph.h`通过鸿蒙ARM64编译检查；v0.1.3已链接MNN并提供后台VAE数值测试，完整音乐链路仍未完成。
- 已读官方Mixed codec真实safetensors头：DiT 39层、hidden1200、FFN4800，`cfm_wrapper`共751,534,224参数。

尚未验证：

- 训练好的0.6B音乐模型：没有。Qwen文本骨干不等于音乐LM。
- 官方Mixed DiT完整真实权重加载和采样结果；构造形状已验证。
- 完整DiT及完整歌曲的手机音频和RTF；真实DiT单层和完整VAE的手机数值/内存验证已通过，后者是合成latent夹具，不是歌曲。
- 当前统一结果：`reports/levo-test-report.json`。

## 3. 公共权重与学生结构

### Qwen骨干

固定来源：`Qwen/Qwen3-0.6B`，已查询revision：
`c1899de289a04d12100db370d81485cdf75e47ca`。

结构：28层、hidden1024、FFN3072、16个Q头、8个KV头、head_dim128、RMSNorm、RoPE、SwiGLU。

参考入口：`MixedSongStudent.from_pretrained()`。禁止使用`ignore_mismatched_sizes`把LeVo大模型强塞进Qwen。新增音乐embedding、音乐输出头和风格向量均为随机初始化，必须训练。

音乐词表：16384个Mixed code；模型另有BOS/EOS约定，EOS不能送入codec。风格目前是32类ID中的一个可训练向量，不支持未经训练将任意风格长文本压成一个有音乐语义的token。

生成前缀：`[style, lyric_0 ... lyric_n, audio_BOS]`，最多514位置。新歌曲重置KV，歌曲内连续decode保留KV。

### 两条训练路线

1. 先严格加载28层公共Qwen，再训练音乐模块，随后微调骨干。保留28层对照。
2. 调用`prune_backbone_depth(student, (9,18))`成为约0.598B；继承剩余Qwen层权重、重编号`self_attn.layer_idx`，再恢复训练和音乐蒸馏。

两条都不是“直接加载LeVo即可唱歌”。LeVo Base公开配置hidden1536、FFN8960，与Qwen不同。仅删除Track-Specific LM也不会自动得到0.6B。

研究依据：Sheared LLaMA `https://arxiv.org/abs/2310.06694`；Minitron `https://arxiv.org/abs/2407.14679`。这是剪枝/蒸馏方法依据，不能当作音乐质量证据。

## 4. 首先恢复下载和真实加载

Python：`C:\Users\lizhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`。
本项目隔离依赖：`C:\levo-harmony\.deps`。不要改系统Python或其它工程依赖。

```powershell
Set-Location C:\levo-harmony
$env:PYTHONPATH='C:\levo-harmony\.deps;C:\levo-harmony'
$levoPython='C:\Users\lizhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $levoPython -X utf8 tools\download_qwen.py
& $levoPython -X utf8 tools\load_backbone.py --checkpoint models\qwen3-0.6b
& $levoPython -X utf8 -m unittest discover -s tests -v
& $levoPython -X utf8 tools\check_architecture.py
```

下载器按固定revision分块，保留在`models/qwen3-0.6b/parts`，最后对照HF LFS SHA256。不要删分块重下，不要只依据文件名宣称校验成功。公共权重文件同时存有embedding/lm_head，文件中张量数量不等于tie后独立参数数量。

加载后额外记录：实际missing/unexpected keys、模型dtype、前向有限值、KV维度、峰值RSS；文本骨干验证不得标成音乐生成通过。

## 5. 官方Mixed DiT适配：当前关键阻碍

来源：`levo-demo/LeVo`，已检查源码树revision：
`653cbcf4716101834900c75b7d5da43b07e15d5b`。

参考文件位于忽略目录`reference/`，不可当作已集成依赖发布：

- `model_1rvq.py`、`generate_1rvq.py`、`audio_tokenizer.py`。
- `mixed-codec-header.json`：真实权重头。
- `stable_audio_1920_vae.json`、`get_1dvae_large.py`。

Mixed权重：
`https://huggingface.co/lglg666/SongGeneration-Runtime/resolve/main/ckpt/model_1rvq/model_2_fixed.safetensors`。

检查的`PromptCondAudioDiffusion`构造中，`cfm_wrapper`初始化被注释，但`inference_codes()`仍调用它。`Tango`又使用`strict=False`，可能忽略大量DiT权重后打印加载成功。**不能接受这种“成功”。** 新桥接器已检查是否存在`cfm_wrapper`，但属性存在也不等于权重完整匹配。

下一Agent按以下顺序处理：

1. 查找匹配权重的官方实现/历史revision，或根据官方结构和权重头恢复构造。不要仅凭注释启用39层配置后就宣布修复。
2. 核对每个张量名、shape、QKV布局、位置编码、时间注入、scale_shift_table、输入投影和输出MLP。
3. 生成关键DiT参数的覆盖率报告；关键层missing/unexpected必须为0。不能依赖整体`strict=False`。
4. 精确复用RVQ恢复embedding、latent归一化统计、mask语义、CFG和Euler时间方向。
5. 先固定真实Mixed code和噪声，官方PyTorch单步→全部扩散步对齐，再单独导出VAE。

`codec.py`是新的导出合同；其中噪声布局、时间方向、步长及tail输入**尚未对齐官方采样器**。不可把它直接作为官方权重的正确推理实现。`upstream_codec.py`仅是参考桥接器，也未验收运行。

官方Mixed/双轨code不可互换；不要把两轨ID相加或任意丢掉其中一轨来构造Mixed标签。

## 6. VAE架构与导出

官方Runtime配置路径：`ckpt/vae/stable_audio_1920_vae.json`。

- Oobleck encoder/decoder，48kHz、2声道。
- base channels128，multipliers `[1,2,4,8,16]`，Snake激活。
- encoder stride `[2,4,4,6,10]`，VAE瓶颈前128通道，采样latent64维。
- decoder输入64通道，逆序上采样10/6/4/4/2，输出2声道，`final_tanh=false`。
- 总压缩率1920，latent25Hz。30秒对应750帧。

只生成时可只部署decoder。分块要测卷积感受野、padding、上下文和重叠裁切；不能直接独立解码后硬拼。检查Snake、卷积/反卷积及归一化的导出支持。优先保持VAE高精度，再凭真实音频敏感性分析量化。

## 7. 音乐训练及蒸馏

`tools/train_adapter.py`接受JSONL：`lyric_ids`、`style_id`、`music_ids`、`codec_id`；可选`teacher_logits_file`做同词表logit蒸馏。

必须先确定真实codec身份、码本次序、tokenizer和style标签映射。数据需要合法来源的歌词/音频对齐，或者许可允许的教师生成样本；20条量化校准数据不等于足以训练音乐LM的数据集。

先做少量真实数据过拟合以验证映射，然后扩展训练。评估EOS/时长、歌词覆盖、音准节奏、旋律、伴奏协调、连续性和听感。`music_adapter_trained=true`仅表示运行过训练，不代表音质验收；必须同时保留`quality_validated=false`，直到真实验收完成。

## 8. MNN与鸿蒙接入顺序

1. 训练后导出音乐LM的prefill/decode：自定义music embedding/head和style输入必须参与图，不能仅调用普通文本LLM接口后把文本token当音频code。
2. 明确KV输入输出、position/cache索引和mask，完成PyTorch→MNN逐步logit对齐。
3. Mixed DiT导出固定短块、动态/多档长度后再优化。`MnnGraph.h`当前只提供浮点输入工具，音乐整数ID/VQ查表需要独立适配。
4. 所有重复前向重新传入实际需要的全部输入；参考ACE历史上低内存模式输入复用失效的经验。
5. CPU先用Precision High+Memory Low确保数值正确，之后分别测试W8/W4与低精度。不要默认所有算子都变成INT8/INT4。
6. 先30秒端到端，再分块、预加载、共驻、GPU/NPU。设备上显式记录实际backend，禁止静默fallback后声称NPU运行。

构建命令`./build.ps1`。当前为未签名HAP；需为独立bundle在DevEco配置自动签名，不能复制CosyVoice证书配置或覆盖其App。

## 9. 内存与报告

现有只读真机快照：系统可用约6.3GiB；检查到的rlimit及cgroup max/high为unlimited/max。这不是安全分配保证。不能把Swap当作高速可用内存。

28层FP16 KV每位置114688字节；加最大歌词前缀后，30秒约138MiB、3分钟548MiB、5分钟876MiB。CFG、多会话、激活、packing另算。26层按实际结构重算。

首轮优先分阶段：LM→保存Mixed IDs→释放LM/KV→DiT分块→VAE逐块写出，限制播放队列；不要先全组件共驻。

统一写入`reports/levo-test-report.json`，保留模型revision/SHA、参数量、精度、初始化和热运行耗时、prefill/decode、DiT步数、VAE、音频秒数、端到端RTF、峰值RSS/PSS、有限值和听感结果。没有实测的字段标未验证，不能拿微内核时间代替整链RTF。

## 10. 发布和用量边界

- 原创工程MIT；Qwen和LeVo依赖保留各自许可。LeVo当前检查的许可证限制学术/研究/教育用途，不应默认可商业生产。
- 不提交模型、`.deps`、下载分块、证书、密码、设备个人数据或本地签名配置。
- 当前用户约定：本5小时窗口剩余≤5%即停新实验，不自动使用credits或下一窗口。写此交接时已到停止线；后续Agent先查询用量并确认恢复授权，不擅自继续计算。
- 未完成的优先级：真实Qwen加载 → 修复并严格加载Mixed DiT → 真实音乐数据训练 → VAE/DiT导出对齐 → 手机短音频。先做可复核结果，不承诺现有权重可无训练变成音乐0.6B。
