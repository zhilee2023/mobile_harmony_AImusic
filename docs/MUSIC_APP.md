# LeVo 音乐：独立的大模型歌曲 App

2026-11-02，当前 v1.0.5，包名 `com.zhilee.levomusic`，显示名 **LeVo 音乐**，绿色音符图标。与原 `com.zhilee.levoharmony` 小模型研究 App 并存。源码在 `music-app/`，不替换 0.6B 模型、App、历史实验或优化清单。

## 已实现

- 简洁单页：歌词、风格、15 / 30 / 60 / 90 / 120 秒、生成、状态与计时；完成后出现播放、进度条和保存 WAV。长于30秒的选项标为实验，尚未完成整曲真机验收；模型输出EOS时可能提前结束。
- 内置20份原创中文歌词，歌词栏“随机换歌词”按钮排除当前文本，保留当前风格；生成和导入期间禁用。仍可手工编辑。
- v1.0.3去掉界面所有“大模型”字样；生成时收起长输入预览，四个步骤字号17、当前阶段标题19，使用带扫描动画与平滑过渡的进度条。进度依据真实token/片段阶段，不按时间伪造完成率。
- 使用公开检查点的真实预训练主 Mixed LM。输入由手机上的 Qwen2 tokenizer 和原始训练 embedding 表编码，编辑歌词与风格会改变输入；没有用预计算歌曲 token 代替用户输入。
- 原生异步任务，UI 线程保持响应；生成期间保持屏幕亮起，可请求停止，模型在前向边界响应取消。
- LM 主体 W4 / 头 W8，CPU High / 4 线程，LM CFG=1；DiT W8、CPU High / 6 线程、默认两步、CFG=1.5；VAE W8、CPU Low / 6 线程、建图期间挂接 NEON Sin。四步运行时保留，常用生成入口已按用户要求改为两步。
- LM 生成完后释放，再执行有界的 DiT/VAE 分块，写 48 kHz 双声道 WAV；完成后播放。目前是生成后试听，不是边生成边播。
- 模型导入在原生异步任务中复制文件，避免主线程同步复制 GB 级模型；只导出一份生成报告。

属于公开主 Mixed LM 的 Mixed-only 路线，省略 track LM 分支；不等于官方完整双轨推理，也不等于已训练好的 0.6B 学生模型。15 / 30 秒按最大音乐 token 数截断，模型可能先 EOS，不保证长歌词完整唱完。

## 已验证与当前状态

- Harmony SDK26、ARM64 原生库与 ArkTS 编译通过，已用用户的新调试签名构建并通过 HDC 安装。
- 共 19 个模型文件，3.641 GiB。已在新 App 的独立私有目录部署，并逐文件核对 SHA256，全部通过。HAP 本身约 7.3 MB，不包含模型权重。
- 手机端四组中文、英文、标点、emoji、章节标签输入，对固定版本 HF tokenizer 的 token ID 全部一致。
- 原始样例完整音乐输入 embedding 文件与 PC 参考逐字节一致，SHA256 都是 `8813a0a155cbcdc256a08c7d6d68428afdef11d67e2b78a72b77c821a8955018`。
- 已查看手机截图：新版页面、当前歌词、风格、生成状态和模型就绪信息正常显示，生成时页面保持响应。
- 最新v1.0.3已取回实际App中的两步30秒歌曲并完成播放，耗时 **195.026秒**，完整RTF **6.501**，峰值RSS **4077.6MiB**。LM阶段134.720秒、DiT32.365秒、VAE23.447秒、其他4.494秒。其750个生成code与LeVo Harmony两步对照完全相同，不能把两次不同App的时延混用。
- 历史v1.0.1四步App样例仍保留：223.593秒、RTF7.453、峰值RSS3869.8MiB。它与两步对照的歌词/风格和生成code相同，方便试听，但不是热状态受控的性能消融。
- WAV 5,760,044 字节、48 kHz、双声道，时长和非静音检查通过；播放器回调确认实际进入 playing/completed 状态，`playbackValidated=true`。当前记录最终播放器状态 released，保留已经成功播放的标记。
- USB 一度断开，恢复后已取回音频和完整报告；结果写入统一报告 `music_app_2026_11_02`。这是实际 App 数据，之前 CLI 的 RTF8.252 单独保留。两轮未控制热状态和调度，不把差异归因于特定优化。
- 听感和歌词清晰度仍待用户试听验收，生成/播放通过不代表达到官方完整分轨路线的音质。
- 同日LeVo Harmony v0.1.8还完成公开模型/小模型的统一串行测试：总RTF5.675 / 1.472，RSS峰值3887.3 / 1177.8MiB。见[完整数值、量化和采样口径](CHAIN_COMPARISON.md)。小模型未训练，不代表能生成同等歌曲。

## 直接使用

1. 手机打开绿色音符的 **LeVo 音乐**。当前设备模型已预装，正常显示“已就绪 · 本地生成”。
2. 编辑或随机更换歌词、修改风格，选时长后点击“生成音乐”，保持 App 前台。30秒已有历史实际生成与播放记录，60/90/120秒为实验选项。
3. 完成后点击播放，或选择“保存音频”。

初次使用建议短歌词，章节标签用小写 `[verse]`、`[chorus]`，可参照内置样例。

## 构建与安装

`tools/build_music_app.ps1` 在首次构建时从 `music-app/build-profile.template.json5` 创建本机配置。实际 `build-profile.json5` 被 Git 忽略，防止提交证书路径和密码；不提交模型、HAP、原始日志或签名文件。

打开 `C:\levo-harmony\music-app`，在 DevEco 的 Signing Configs 为新包名生成调试签名，并关联 default 产品。旧研究 App 的 Profile 绑定旧包名，不能用于新包名。参考[华为调试 Profile 文档](https://developer.huawei.com/consumer/cn/doc/doccenter-getting-started/agc-help-debug-profile-0000002248181278)。

本机产物：`C:\levo-harmony\dist\LeVo-Music-v1.0.5-signed.hap`。原研究 App 保持安装，已更新到v0.1.8以增加串行对照入口。

## 同进程界面对照：v1.0.5

为调查两步整曲变慢，在同一个真实App进程复用唯一W4主体/W8头LM、CPU High4线程。先完整953位置prefill加8次decode预热，再做动画开启A1→暂停B→开启A2，每轮reset/generate_init、同一953位置前缀和128个固定真实音乐code；每次读取全16385维logits并检查有限值。暂停轮关闭扫描/平滑动画、spinner和常规状态刷新，800ms阶段轮询和20ms内存采样三轮一致。阶段切换先得到UI确认再计算。测试只由调试intent触发，正常生成仍保留动画。

|阶段|Prefill 秒|128次 Decode 秒|Decode token/s|相对A1最大logits L2|
|---|---:|---:|---:|---:|
|A1 开启|21.014|12.371|10.347|0|
|B 暂停|22.244|11.457|11.172|0|
|A2 开启|22.231|11.775|10.871|0|

B的decode耗时比两次A均值低5.10%，吞吐高5.37%；prefill没有加快。只做了一组顺序对照，不能排除调度/温度漂移，不能解释此前完整750-token测试从10.26降到7.29token/s的全部差异，也不是新整链RTF。CPU频率可读取，App不能读取thermal节点，未获得CPU温度。采样峰值RSS3222.7MiB、最低系统可用2858MiB；RSS不是显存。每轮prefill后历史长度953、decode后1081，输出完全一致。最初测试在错误的“reset立即清零计数”检查处停止；MNN实际在下一次forward执行清除，修正检查后本轮通过。

统一报告字段为`music_app_ui_aba_2026_11_02`与`music_app_v105_features_2026_11_02`。复现使用`tools/run_music_ui_benchmark.ps1`，通过`musicAutoRun=llm-ui-aba-20261102`启动；合并工具为`tools/merge_music_ui_benchmark.py`。此测试不覆盖上次歌曲、音乐code或播放状态。

120秒请求上限3000个音乐code，Codec最多16个固定256帧窗口、步进192帧，PCM逐块写盘，避免累积整曲PCM；LM仍须保存随上下文增长的KV，不能把固定Codec窗口当作120秒内存已验收。保留系统可用内存和RSS门槛、取消与LM660秒时限。生成模式没有改成边生成边播。

## 模型准备

`tools/prepare_music_app_assets.py` 使用已经验证的公开模型文件生成 `music-app/model-assets.json` 和原生 inventory，不重新下载权重。本机 `dist/LeVo-Music-models/` 提供 19 个平铺的导入文件；复制到新手机后，在 App 中“导入模型”多选这 19 个文件。

`conditions.safetensors` 包含完整训练过的歌词、结构、风格和参考音频 embedding 表；采用只读 mmap，按当前 token 读取，避免一次性加载 1.3 GB 条件表。运行时生成的约 75 MB embedding 文件包含真实音乐 embedding、当前条件前缀和固定数值校验夹具。

App 导入时检查名称和精确文件大小，桌面部署另做 SHA256。不要把“文件数量就绪”当作音质验收。

## 复现入口

- `entry/src/main/cpp/MusicConditioner.hpp`：自定义文本分词、章节覆盖、训练 embedding、BF16 前缀。
- `MusicRuntime.hpp` / `napi_init.cpp`：模型清单、异步生成、取消、进度与结果。
- `ConditionProbe.cpp`：独立 HF token 夹具对照，不运行音乐 LM。
- `AppRunProbe.cpp`：可复用同一 App 核心的 CLI 探针，结果需明确与 App 实际运行区分。
- `experiments/public_mixed_validation/native/`：复用已验证的预训练 Mixed 采样器和四步 Codec。
- `tools/prepare_music_condition_tests.py`：生成独立参考夹具。

旧小模型与 kernel 经验继续见 [SMALL_MODEL_OPTIMIZATION_BASELINE.md](SMALL_MODEL_OPTIMIZATION_BASELINE.md)。
