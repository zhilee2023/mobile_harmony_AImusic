# mobile_harmony_AImusic · levo-harmony

> v0.1.7：保留推荐W8＋NEON原入口，同包五块RTF2.290、峰值约2.19GiB。Snake融合接入App后RTF2.697（含首块额外校验），未提速；用户选择原入口，较慢实验按钮已移出常规UI，测试与后端保留。见[VAE记录](docs/VAE_OPTIMIZATION.md)、[DiT后续方向](docs/DIT_ACCELERATION.md)与统一报告。仍是随机音乐头的数值链路，不是训练好的歌曲。

> [VAE NPU研究](docs/VAE_NPU_FEASIBILITY.md)：四组连续真实子图已成功离线OMC NPU-only运行、禁止CPU回退；预热19.9–258ms，本轮均慢于匹配W8＋NEON CPU，App保留原入口。尚非完整VAE RTF，旧在线查询异常和数值/内存限制均保留。

> 后续纯Conv隔离已测：六Conv High两轮比CPU W8快2.44–3.07倍，但完整残差块NPU仍较慢，**尚无新的VAE端到端加速**。App继续原CPU入口，纯Conv数据单列在统一报告，不能拿消融输出替代歌曲。

> 2026-10-30 Snake融合CLI探针三窗通过：对同轮W8＋NEON约1.16倍加速，PCM L2=0；仍有120MiB输入快照、未接入App，绝对RTF未超过历史最好。详见[VAE优化记录](docs/VAE_OPTIMIZATION.md)。[MNN推测解码接入说明](docs/SPECULATIVE_DECODING.md)已核对，音乐链路尚未启用。

> v0.1.6 VAE优化已安装实测：W8权重＋ARM82 NEON正弦，独立VAE三窗RTF 1.193→0.674；亮屏五块CLI整链2.571→2.272，签名App候选2.431/峰值约2.19GiB。不是训练好的音乐或音质验收；3倍VAE和RTF1.5均未达到。见[详细结果](docs/VAE_OPTIMIZATION.md)，统一数据仍在reports/levo-test-report.json。

面向16GB Mate 80 Pro的独立混合歌曲研究工程。**v0.1.5是已签名真机测试的多块流程/内存诊断App，不是已经训练好的0.6B LeVo。** CosyVoice与ACE项目保留。

目标：最多512个歌词token＋1个风格前缀 → 约0.6B自回归骨干 → 单路Mixed音乐token → DiT → 48kHz双声道VAE。取消双轨分支，但保留人声＋伴奏的混合歌曲。

## 已实现

- 已实际加载完整Qwen3-0.6B公开权重并核对SHA256；独立音乐embedding、音乐头、风格embedding仍未训练。
- 单路音乐teacher forcing、KV增量decode、EOS、最大长度与codec身份检查；未训练音乐头禁止生成。
- 监督/教师logit蒸馏训练入口，参数计数与KV预算工具。
- 完整39层真实DiT已导出MNN W8/W4，与26层骨干及真实VAE接入签名App完成单窗流程；严格数值误差与内存流程分开报告，未作音乐质量验收。
- 独立包名鸿蒙ArkTS＋C++诊断App，用户本地自动签名并通过HDC安装测试；签名材料不上传。
- GitHub CPU参考测试；没有上传模型权重。

最新结果见[统一报告](reports/levo-test-report.json)。10.24秒窗口和0.6B接口见[分块与学生说明](docs/STREAMING_AND_STUDENT.md)。App由用户[手动签名安装](docs/INSTALL_MANUAL.md)，无电脑界面自动操作。

已保存可重载的真实26层文本骨干（含音乐模块总参数约0.598B，未恢复训练），提供保留KV的可暂停token流。VAE64帧候选约348MiB。最新单窗三模块W8共驻峰值约2.03GiB、整链RTF2.40–2.57；W4约1.77GiB、RTF2.60–3.03。音乐头为随机初始化，跨窗/长文稳定性及数值/音质验收尚未完成。详见[真机三模块记录](docs/THREE_MODULE_VALIDATION.md)和[内核复用评估](docs/KERNEL_REUSE_REVIEW.md)。

2026-10-29完整codec验证：严格加载526个CFM参数张量，20步Euler解码两个256帧窗口、64帧重叠，输出17.92秒有限值PCM。整网SDPA对eager相对L2约1.01e-6，采样器对官方实现约3.26e-7。使用测试codec ID，未接训练好的音乐LM；这是PC数值验证，不是手机RTF或音质验收。

同日亮屏VAE复测：Low/6线程下，64帧模块RTF1.21–1.22、峰值约348MiB；256帧1.12–1.17、约814MiB。与历史息屏结果相比没有显著提速证据，不能作为整链实时性结论；详见统一报告和分块说明。

最新[连续5块真机记录](docs/FIVE_CHUNK_VALIDATION.md)：一次prefill、同一KV1537位置，5个256帧窗口/64帧重叠，得到40.96秒有限PCM；总113.11秒，RTF2.761，峰值约2.27GiB。未播放音乐；目标1.5尚未达到。Low/8线程组合在前台复测RTF3.443，更慢，不启用；息屏暂停首轮、Kleidi开关未提速和MNN通用融合失败也保留在报告中。

## PC验证

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/check_architecture.py
python tools/audit_public_configs.py
# 以下会下载完整公共文本骨干；不产生已训练音乐能力
python tools/load_backbone.py --checkpoint Qwen/Qwen3-0.6B
```

训练数据JSONL：`lyric_ids`、`style_id`、`music_ids`、`codec_id`。音乐ID必须来自已验证的同一个Mixed codec。`tools/train_adapter.py --help`说明训练入口。小数据过拟合只能验证代码，不能充当音质验收。

## 鸿蒙构建

DevEco Studio SDK26，Windows运行`./build.ps1`。默认无签名，产物为`app/entry/build/default/outputs/default/entry-default-unsigned.hap`。需要在DevEco为**com.zhilee.levoharmony**设置自己的自动签名后才能装机；不要上传证书、密码或签名配置。

架构、尚未完成的部分和验收阶段见[架构说明](docs/architecture.md)。公开权重缩小的研究与许可证见[研究记录](docs/research.md)。不会将Qwen文本权重直接称为音乐模型，也不承诺0.85 RTF。
