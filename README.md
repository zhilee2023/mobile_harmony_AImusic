# mobile_harmony_AImusic · levo-harmony

面向16GB Mate 80 Pro的独立混合歌曲研究工程。**v0.1.3是可测试的学生模型原型和可构建的鸿蒙诊断App，不是已经训练好的0.6B LeVo。** CosyVoice与ACE项目保留。

目标：最多512个歌词token＋1个风格前缀 → 约0.6B自回归骨干 → 单路Mixed音乐token → DiT → 48kHz双声道VAE。取消双轨分支，但保留人声＋伴奏的混合歌曲。

## 已实现

- 已实际加载完整Qwen3-0.6B公开权重并核对SHA256；独立音乐embedding、音乐头、风格embedding仍未训练。
- 单路音乐teacher forcing、KV增量decode、EOS、最大长度与codec身份检查；未训练音乐头禁止生成。
- 监督/教师logit蒸馏训练入口，参数计数与KV预算工具。
- 真实DiT单层手机32/256帧对齐；完整真实VAE导出、手机三窗数值/内存测试。完整DiT和歌曲codec仍未闭环。
- 独立包名鸿蒙ArkTS＋C++内存诊断App；未签名HAP构建，不复制本地证书。
- GitHub CPU参考测试；没有上传模型权重。

最新结果见[统一报告](reports/levo-test-report.json)。10.24秒窗口和0.6B接口见[分块与学生说明](docs/STREAMING_AND_STUDENT.md)。App由用户[手动签名安装](docs/INSTALL_MANUAL.md)，无电脑界面自动操作。

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
