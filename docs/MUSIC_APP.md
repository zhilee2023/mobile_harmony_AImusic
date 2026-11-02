# LeVo 音乐：独立的大模型歌曲 App

2026-11-02，新包名 `com.zhilee.levomusic`，显示名 **LeVo 音乐**，绿色音符图标。与原 `com.zhilee.levoharmony` 小模型研究 App 并存。源码在 `music-app/`，不替换 0.6B 模型、App、历史实验或优化清单。

## 已实现

- 简洁单页：歌词、风格、15 / 30 秒、生成、状态与计时；完成后出现播放、进度条和保存 WAV。
- 使用公开检查点的真实预训练主 Mixed LM。输入由手机上的 Qwen2 tokenizer 和原始训练 embedding 表编码，编辑歌词与风格会改变输入；没有用预计算歌曲 token 代替用户输入。
- 原生异步任务，UI 线程保持响应；生成期间保持屏幕亮起，可请求停止，模型在前向边界响应取消。
- LM 主体 W4 / 头 W8，CPU High / 4 线程，LM CFG=1；DiT W8、CPU High / 6 线程、四步、CFG=1.5；VAE W8、CPU Low / 6 线程、建图期间挂接 NEON Sin。
- LM 生成完后释放，再执行有界的 DiT/VAE 分块，写 48 kHz 双声道 WAV；完成后播放。目前是生成后试听，不是边生成边播。
- 模型导入在原生异步任务中复制文件，避免主线程同步复制 GB 级模型；只导出一份生成报告。

属于公开主 Mixed LM 的 Mixed-only 路线，省略 track LM 分支；不等于官方完整双轨推理，也不等于已训练好的 0.6B 学生模型。15 / 30 秒按最大音乐 token 数截断，模型可能先 EOS，不保证长歌词完整唱完。

## 已验证与当前状态

- Harmony SDK26、ARM64 原生库与 ArkTS 编译通过，已用用户的新调试签名构建并通过 HDC 安装。
- 共 19 个模型文件，3.641 GiB。已在新 App 的独立私有目录部署，并逐文件核对 SHA256，全部通过。HAP 本身约 7.3 MB，不包含模型权重。
- 手机端四组中文、英文、标点、emoji、章节标签输入，对固定版本 HF tokenizer 的 token ID 全部一致。
- 原始样例完整音乐输入 embedding 文件与 PC 参考逐字节一致，SHA256 都是 `8813a0a155cbcdc256a08c7d6d68428afdef11d67e2b78a72b77c821a8955018`。
- 已查看手机截图：新版页面、当前歌词、风格、生成状态和模型就绪信息正常显示，生成时页面保持响应。
- 已启动 App 内 15 秒自定义歌词的真实生成。观察到数值探针通过、完整前缀开始计算，且 App 私有目录产生结果文件；随后 USB 调试连接断开。结果内容、最终 WAV 和播放器状态尚待重新连接后取回核实，不能声称这些验收已通过。
- 之前 CLI 整链 30 秒 / RTF8.252 是该公开模型路线的先行验证；不能冒充本次 App 内最终 RTF。

## 直接使用

1. 手机打开绿色音符的 **LeVo 音乐**。当前设备模型已预装，正常显示“大模型已就绪”。
2. 修改歌词和风格，选 15 或 30 秒，点击“生成音乐”，保持 App 前台。
3. 完成后点击播放，或选择“保存音频”。

初次使用建议短歌词，章节标签用小写 `[verse]`、`[chorus]`，可参照内置样例。

## 构建与安装

`tools/build_music_app.ps1` 在首次构建时从 `music-app/build-profile.template.json5` 创建本机配置。实际 `build-profile.json5` 被 Git 忽略，防止提交证书路径和密码；不提交模型、HAP、原始日志或签名文件。

打开 `C:\levo-harmony\music-app`，在 DevEco 的 Signing Configs 为新包名生成调试签名，并关联 default 产品。旧研究 App 的 Profile 绑定旧包名，不能用于新包名。参考[华为调试 Profile 文档](https://developer.huawei.com/consumer/cn/doc/doccenter-getting-started/agc-help-debug-profile-0000002248181278)。

本机产物：`C:\levo-harmony\dist\LeVo-Music-v1.0.1-signed.hap`。原研究 App 保持安装。

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
