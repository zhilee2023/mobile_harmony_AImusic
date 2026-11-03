# 模型替换、部署与实验接口

统一入口为`tools/modelctl.py`。电脑端负责Torch/PT/ONNX转换与校验，手机只加载准备好的MNN模型包；不在手机上运行Python转换器。

若新模型使用每token最后层hidden作为Diffusion条件，先看[hidden条件适配指南](HIDDEN_STATE_CONDITIONING.md)。当前部署合同仍为token/codebook条件；研究检查工具不会把不兼容的hidden接口直接打包进App。

## 代码分层

| 位置 | 职责 |
|---|---|
| `configs/profiles/` | 当前大小模型的结构、Codec合同和默认配方 |
| `configs/deployment/` | 权重导入与组件替换任务示例 |
| `levo_harmony/deployment/checkpoints.py` | 张量checkpoint、安全读取、strict键/形状检查 |
| `torch_models.py` / `components.py` | 结构适配、Torch/ONNX转MNN、数值校验 |
| `bundles.py` / `device.py` | 版本化打包、完整性检查、HDC部署 |
| `baselines.py` | 复用已有下载/导出产物，不重新下载 |
| `runtime/include/levo/` | 公共会话、LM、条件编码、DiT/VAE、模型清单与选择 |
| `runtime/include/levo/kernels/` | NEON正弦和保留的Snake实验实现 |
| `music-app/` / `app/` | 产品与研究界面、异步任务、播放 |
| `experiments/catalog.json` | 保留实验入口及结果使用范围 |
| `skills/levo-model-adaptation/` | 后续Agent换模型skill |

旧CLI与App头文件保留为兼容层；产品运行时不再通过包含实验`.cpp`获得推理实现。历史导出/实验工具仍可复现，默认部署流程使用`modelctl`。

## 来源支持与限制

| 来源 | 支持方式 | 必要信息 |
|---|---|---|
| PT/PTH/Torch/safetensors | strict恢复结构，再导出 | 模块/profile，必要时stateKey、stripPrefix |
| TorchScript | 显式format=torchscript | DiT/VAE的forward合同 |
| ONNX DiT/VAE | 检查I/O，转换并做ONNX/MNN对照 | 保留外部tensor文件 |
| HF或PT音乐LM | 官方MNN LLM导出器 | config、tokenizer、已训练音乐头和配套输入表 |
| ONNX LM | 已满足MNN导出器接口的图 | llm_config、tokenizer、BF16输入表；普通显式KV的HF图需先适配 |
| MNN | 导入运行图与外部weight | LM配套文件，导入方声明recipe |

只有权重文件不能恢复未知网络。当前支持两套LM/条件接口；DiT/VAE可换内部实现，但I/O与Codec必须兼容。检查ONNX I/O不代表识别了内部层数。Python对象pickle不能冒充张量checkpoint或TorchScript。

音乐LM输出头为16385类。普通Qwen文字头不能直接当成音乐头。学生训练文件可以用`checkpointLayout=student-training`，严格加载`MixedSongStudent`全部键，再组合文本/风格/音乐输入embedding和音乐头；不会随机补缺少的适配器。

默认配方：公开LM High4、学生LM Low4，主体W4/头W8/BF16输入表；DiT High请求6、W8、两步CFG1.5；VAE Low请求6、W8＋NEON。没有默认A4/KV4、GPU或NPU。现成LM ONNX的通用转换默认统一W8；指定统一W4时头也为W4。混合W4主体/W8头请走HF/PT官方导出器。

## 基础命令

从仓库根目录运行：

```powershell
python tools/modelctl.py profiles
python tools/modelctl.py inspect C:\weights\decoder.pt --state-key state_dict --strip-prefix decoder.
python tools/modelctl.py experiments --family small
```

复制`configs/deployment/`示例后填写新路径。路径相对任务JSON所在目录，`${ROOT}`代表仓库根目录。每个输出使用新目录，不覆盖旧文件。

```powershell
python tools/modelctl.py prepare --job configs/deployment/vae-pt.example.json --out models/prepared/my-vae-v1 --mnn-packages C:\path\to\site-packages
python tools/modelctl.py prepare --job configs/deployment/dit-onnx.example.json --out models/prepared/my-dit-v1 --mnn-packages C:\path\to\site-packages
```

LM另加`--exporter-root <MNN源码>/transformers/llm/export`。LM使用现代Transformers，DiT恢复使用`requirements-codec.txt`的旧版上游依赖，保持隔离。转换器的`_tools`须匹配Python版本和位数。`profiles/pack/verify/deploy`不会加载Torch。

`component.json`仅成功后写出，记录来源SHA、strict键数、配方、I/O、误差和产物SHA。失败目录保留中间文件，不是可部署组件。数值门槛必须在任务中声明：本次PT VAE首轮5%失败；按既有W8约6%的误差范围，另建10%兼容任务，实际7.13%。两轮都保留，不代表听感已验收。

## 整理基础模型与替换

```powershell
python tools/modelctl.py baseline --profile student-mixed-v1 --id student-base-v1 --out models/bundles/student-base-v1
python tools/modelctl.py baseline --profile public-mixed-v2 --id public-base-v1 --out models/bundles/public-base-v1 --public-artifacts C:\path\to\existing\artifacts
python tools/modelctl.py pack --job configs/deployment/replace-components.example.json --out models/bundles/my-song-model-v2
python tools/modelctl.py verify models/bundles/my-song-model-v2
```

`baseline`复用已有文件；public-artifacts目录含`mixed-mnn-w4`和条件表。替换任务填写`baseDirectory`、新`modelId`、`displayName`及`components`。旧图和weight不会重复放进新包，其他模块从基础包复制。

包中`model-bundle.json`保存全量SHA、实际模块路径、训练状态声明和请求运行参数。检查覆盖词表、embedding行数、前缀索引、Codec配套张量与条件表范围。语义不同的codebook不能只改文件名冒充同一Codec；训练状态声明和文件校验都不证明音质。

## 手机部署和切换

```powershell
python tools/modelctl.py deploy models/bundles/my-song-model-v2 --hdc "C:\path\to\hdc.exe" --device YOUR_DEVICE_ID --app music
```

先上传独立incoming目录，逐文件检查远端SHA，再发布为独立版本。原模型不覆盖。产品启动按清单检查文件尺寸；全量SHA在部署工具执行。`--activate`显式更新选择标记；也可在手机 **歌曲模型 → 刷新 → 选择**。

LeVo音乐v1.0.6的菜单和原生`models(filesDir)`/`selectModel(filesDir,id)`使用同一接口。生成/导入时不能切换。无效ID被拒绝，当前模型保持不变；已有歌曲和默认模型保留。构建工具通过本地CMake配置注入MNN路径，不将工作站路径写死在产品CMake中。

产品只列当前public条件接口下声明已训练的模型包。未训练学生用LeVo Harmony验证；未来训练好的学生若要直接处理自由文本歌词/风格，需要学生条件适配器，不能只将musicTrained改为true。

```powershell
python tools/modelctl.py deploy models/bundles/student-base-v1 --hdc "C:\path\to\hdc.exe" --device YOUR_DEVICE_ID --app research
powershell -File tools/run_comparison_app.ps1 -Device YOUR_DEVICE_ID -Variant small-2 -RunId my-small-test -ModelBundle student-base-v1 -RecordSensors
```

产品短生成验证用`tools/test_music_model.ps1`，默认启动15秒测试，`-Attach`仅观察现有任务并明确记录观察起点。研究入口保留30/90/120秒、同输入VAE重放；实验不全部加入产品菜单。

这两个电脑测试入口默认约每5秒采样一次温度、电池电压／电流与CPU频率，结束后附加到**同一个测试JSON**的`telemetry`字段。可用`-SensorIntervalSeconds 5`调整1–60秒间隔，或`-RecordSensors:$false`关闭；必要时用`-Python <python.exe>`指定解释器。任务完成、观察器退出或手机进程变化会停止采样。缺失节点为null，不补0。USB供电下电池端交换功率不是App／整机功耗；换手机要重新核对单位。这是电脑观察期间采样，不是独立App后台监控或定时重跑。图表、验证及历史日志归属更正见[模型与温度总结](MODEL_THERMAL_SUMMARY.md)。

## 本轮验证边界

- 真实VAE原始PT：182个张量，84.414M参数；ONNX对Torch L2约0.000296%，MNN W8对ONNX约7.13%，输出形状与有限值通过。
- 完整小DiT ONNX导入、W8转换：L2约1.64%；随后手机整体小模型链路验证。
- TorchScript微型DiT格式合同转换通过，不冒充完整22层或音乐质量验证。
- 原始HF格式LM小规模单元测试覆盖strict、音乐输出头、非持久RoPE恢复。新完整LM仍要逐个做导出/手机prefill与decode验收，微型测试不能代替。
- 产品菜单发现/选择新包、切回默认、拒绝不存在ID、恢复新包、15秒真实生成及播放已验证；具体结果见统一报告和[完整总结](PROJECT_SUMMARY.md)。
