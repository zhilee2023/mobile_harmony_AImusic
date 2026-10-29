# LeVo Harmony：给接手Agent的模型适配指南

更新时间：2026-10-28。本文是交接文档，不代表下列未完成项已验收。

> 2026-10-29收尾更新：见[CPU验证记录](CPU_VALIDATION.md)和统一JSON。Mixed DiT构造已恢复并通过526参数形状检查；真实单层、ONNX/MNN FP32已数值对齐。以下早期状态中“尚未修复构造”已被这项结构验证推进，但完整权重及整曲生成仍未完成。下载分块约Qwen1,040MB、VAE159MB；DNS中断后保留，按原窗口规则没有自动重启实验。

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
- 8项微型模型单元测试通过；60步合成ID训练loss从约3.014降到0.164。它只验证优化器路径，不代表学会音乐。
- 完整meta模型参数：28层学生629,640,192；其中Qwen骨干596,049,920。
- 可选移除层9、18并重排KV索引，26层学生598,178,304参数。选层未经重要度校准，必须恢复训练。
- 独立鸿蒙诊断App构建成功，包名`com.zhilee.levoharmony`；HAP未签名。
- `MnnGraph.h`使用现有MNN 3.6.1头文件通过鸿蒙ARM64编译检查。未完成音乐运行时链接及推理。
- 已读官方Mixed codec真实safetensors头：DiT 39层、hidden1200、FFN4800，`cfm_wrapper`共751,534,224参数。

尚未验证：

- 完整公共Qwen权重加载：下载未完成，最后日志约226MB/1,503,300,328字节；磁盘分块为准。
- 训练好的0.6B音乐模型：没有。Qwen文本骨干不等于音乐LM。
- 官方Mixed DiT完整构造、真实权重加载和采样结果。
- 真DiT/VAE的ONNX→MNN导出、PC对齐、手机音频和RTF。
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
