# mobile_harmony_AImusic · levo-harmony

> [一张表查看大小模型：手机型号、CPU核心／线程、ROM、参数、量化及60秒RTF](docs/CONSOLIDATED_MODEL_COMPARISON.md)。对应基线模型资产ROM：LeVo 4B约3.641GiB，0.6B原型约0.844GiB；CPU推理，请求线程数不等于固定绑定核心数。

> 汇总名称：**LeVo 4B（Mixed-only）**，实测部署为3.106B LM＋751.534M DiT＋84.414M VAE，约3.942B，不含官方track LM。LM主体W4/g64、音乐头W8、输入／条件表BF16；DiT W8/g64；VAE W8；Codec辅助表FP32。未启用静态A4或KV4。详见[各部分量化和60秒测试](docs/COOLED_60S_COMPARISON.md#配方)。

> **2026-11-03：两边60秒输出已完成，间隔冷却900.304秒。** LeVo Harmony v0.1.14同包：公开主Mixed 351.818秒／RTF5.864／RSS4.645GiB；0.6B原型95.060秒／RTF1.584／RSS1.157GiB。两份60秒WAV已取回并核验，小模型音乐适配层仍未训练。电池温度公开28–37°C、小模型31–35°C；USB电池端交换功率不是整机功耗。见[阶段表、冷却协议与温度曲线](docs/COOLED_60S_COMPARISON.md)。原模型、产品LeVo音乐和旧实验保留。

> **最终模型／温度总结**：小模型0.598B LM＋195.416M DiT＋84.414M VAE，当前包约0.844GiB；公开主Mixed 3.106B LM＋751.534M DiT＋84.414M VAE，当前包约3.642GiB。小模型90秒重测RTF1.566–1.608、RSS约1.10GiB，音乐适配层未训练。电脑观察两款App测试时现默认约5秒采样，并附到同一测试JSON；37项测试通过。电池端交换功率不是整机功耗。见[完整表格／温度图／公开90秒归属更正](docs/MODEL_THERMAL_SUMMARY.md)。

> **模型替换已接入LeVo音乐v1.0.6**：顶部“歌曲模型”菜单支持独立模型包发现/切换，默认模型保留。真实PT VAE→ONNX→MNN后完成15秒歌曲生成/播放（89.944秒，RTF5.996）；ONNX小DiT新包30秒数值链路44.596秒、RTF1.487，750码与重构前一致。当前33项测试通过。见[完整总结](docs/PROJECT_SUMMARY.md)、[替换与部署命令](docs/MODEL_DEPLOYMENT.md)及[Agent skill](skills/levo-model-adaptation/SKILL.md)。这是替换/部署验收，未把不同轮次时延当成受控优化收益。

> **最新小模型重测：v0.1.12，90秒数值链路连续两轮RTF 1.566 / 1.608**，峰值RSS约1.10GiB。相同12份latent的VAE整链59.763秒、独立52.406秒（少12.3%），全部输出指纹一致、首窗L2=0；此前RTF2.163的慢速未复现。量化和线程请求未改，小模型仍未训练。见[同输入重测与线程池线索](docs/SMALL_90S_RETEST.md)，旧实验与公开模型记录均保留。

> **2026-11-02：LeVo Harmony v0.1.11，90秒真机记录**：0.598B原型单次端到端RTF **2.163**、RSS峰值 **1.19GiB**；公开主Mixed LM生成齐2250个code后退出，再以原码恢复90秒WAV，已测计算段合计RTF **6.225**（**不是单次端到端**），退出前已捕获RSS至少 **5.43GiB**。Mixed DiT精确源参数 **751.534M**，小DiT **195.416M**，共用VAE **84.414M**。详见[参数、ROM、量化、阶段RTF与温度/功率图](docs/CHAIN_COMPARISON_90S.md)；[120秒失败与模块记录](docs/CHAIN_COMPARISON_120S.md)保留。功率曲线为USB充电条件下电池端交换功率幅值，不是App/SoC功耗。

> 独立 **LeVo 音乐 v1.0.5** App：20份原创歌词＋随机更换、歌词/风格编辑、试听与保存，已签名安装。时长选项15/30/60/90/120秒；30秒已有实际生成与播放记录，60–120秒为尚未完成整曲验收的实验选项，EOS可提前结束。模型19个文件共约3.64GiB，保持原小模型App并存。见[使用与构建说明及界面对照](docs/MUSIC_APP.md)。

> 同进程LM预热后界面A-B-A对照：动画开/停/开为10.35/11.17/10.87token/s，128次固定输入decode、输出L2=0。暂停轮比两次开启动画均值少约5.1%decode时间，prefill未加快；不足以归因之前整曲速度差异。该结果不是新整链RTF，数据仍合并至唯一统一报告。

> 当前 **LeVo 音乐 v1.0.3** 已去掉界面“大模型”字样、放大四步骤并加入动画进度条，生成入口改为DiT两步/CFG1.5，实际App完成30秒并播放：195.026秒、RTF6.501、峰值RSS4077.6MiB。另在 **LeVo Harmony v0.1.8** 做统一串行两步对照：公开模型RTF5.675 / 小模型数值RTF1.472，各部RTF、量化配置与RSS/PSS/图形内存见[对照表](docs/CHAIN_COMPARISON.md)。不同App结果和历史四步结果分别保留。

> 2026-11-02 公开主 Mixed LM 已完成[真实音乐生成验证](experiments/public_mixed_validation/README.md)：主 LM W4 / 头 W8 → DiT W8 四步 / CFG1.5 → VAE W8＋NEON。手机 LM 关闭 CFG，单分支；完整 30 秒 WAV 用时 247.545 秒，RTF8.252，峰值 RSS3956.8MiB，尚未听感验收或接入 App。公开约4B双LM架构见[架构说明](docs/PUBLIC_LEVO_ARCHITECTURE.md)；本次省略 track LM，是 Mixed-only 路线。0.6B 方案继续保留，见[小模型优化保留清单](docs/SMALL_MODEL_OPTIMIZATION_BASELINE.md)，W4A4KV4 仍待验证。

> 2026-10-31 新增[原始 Qwen 语言模型低比特实测](docs/QWEN_LOW_BIT_LANGUAGE_TEST.md)：28层完整词表，W4＋KV16/6线程约25–28token/s，512-token prefill约1.40–1.58秒。实验A4范围＋KV4已测但数值质量失败，A8静态对照也失败；相同权重移除静态激活描述后恢复。没有可用的原生packed W4A4KV4提速结果。另有[随机小DiT十块调度验证](docs/SMALL_DIT_SCHEDULING.md)：计算重叠数值RTF1.382、峰值约1.75GiB，未接入App、未训练音乐、严格L2未过1%。所有数据只发布到同一份统一报告。

> 2026-10-31 新手机同包 App 五块对照：系统自报 Mate 90 Pro Max / Kirin9050；同样 Qwen W4 → CPU DiT W8 两步 → VAE W8＋NEON，40.96 秒数值 PCM 用时 73.356 秒，RTF **1.791**，采样峰值 2203.3 MiB。旧机同包为 93.799 秒 / 2.290；单轮耗时减少约 21.8%。仍是未训练音乐头的流程测试，DiT 对 PC 参考 L2 2.646%，未通过 1% 严格门槛。见[新机对照](docs/DEVICE_CPU_SPEED.md)和统一报告。

> v0.1.7：保留推荐W8＋NEON原入口，同包五块RTF2.290、峰值约2.19GiB。Snake融合接入App后RTF2.697（含首块额外校验），未提速；用户选择原入口，较慢实验按钮已移出常规UI，测试与后端保留。见[VAE记录](docs/VAE_OPTIMIZATION.md)、[DiT后续方向](docs/DIT_ACCELERATION.md)与统一报告。仍是随机音乐头的数值链路，不是训练好的歌曲。

> [CANN官方W8A8量化实测](docs/CANN_VAE_QUANTIZATION.md)：20条校准＋4条留出输入已跑通；四行完整残差46–47ms，仍慢于CPU W8 39–42ms；纯卷积W8A8约12.3–12.5ms，对同轮CPU约2倍。关键更正：残差OMC预编译为NPU3＋CPU2，运行时禁回退不代表纯NPU；旧三残差为NPU7＋CPU6。App未切换，尚无新整链RTF。

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
