# 4B音乐模型：CPU Low整链实验

## 配方与启动条件

实验App v0.1.16新增显式CPU精度开关。默认仍是原配方：LLM、DiT、投影High，VAE Low。
实验配方把前三者也设为Low；原W4/group64 LLM、W8头、W8 DiT与W8 VAE权重均不变。
Low是MNN运行时策略，并不意味着所有算子／累加器都是FP16；必要的FP32路径仍可保留。

目标为`larkstudio-base-v2`，36层Qwen音乐LLM＋39层DiT，120秒／3000个code。
DiT两步Euler＋CFG1.5，256帧窗口、64帧重叠；VAE保留36组Snake融合。
LLM常驻请求，DiT、投影、VAE按窗创建和销毁；线程仍为4／6／6，投影4线程。
没有同时调整步数、CFG、词表、动态A8选项或模型驻留方式。

最初按用户要求准备冷却至少600秒且起始电池温度≤30°C的实验。
准备期间用户明确改为“现在测”，因此该准备任务在推理开始前取消，另建33°C立即开始的120秒记录。
本次成绩不标为30°C测试；下面保留可复用的30°C受控启动命令。
电池温度不是芯片结温，SoC可读节点单独记录。启用温度门槛时，未达到或未手动解锁就不启动推理。

```powershell
./tools/run_music_duration.ps1 -Device DEVICE -RunId UNIQUE_ID `
  -DurationSeconds 120 -ModelChoice 1 -DynamicOption 0 -VaeFusion 1 `
  -FloatingPrecision low -Seed 2307 -CoolingSeconds 600 -MaximumStartBatteryC 30
```

原High基线用`-FloatingPrecision high`。手动界面有“全链CPU Low／FP16（实验）”开关，生成中禁用。
原生输出`precisions`逐模块记录请求精度，脚本核对全Low请求是否实际传入原生最终状态；
这不代替逐算子的FP16／FP32内核审计。

## 已完成的前置检查

已完成同一App、同一大模型的15秒Low整链，375个code、两窗、72次融合调用，全部有限并生成WAV。
原生确认LLM、DiT、投影、VAE均收到Low配置。结束前后检测到屏幕关闭／唤醒记录变化，
因此这轮只用于功能检查，不采纳为受控性能成绩，也不冒充正式120秒结果。
App构建通过，现有59项单元测试通过；没有上传权重、音频或签名文件。

## 已完成的完整120秒

16份WAV合计5760000帧／48kHz双声道，3000个有效code、最终KV3513。
本次实际运行576次Snake融合，150次前台检查全部通过，未息屏、未提前EOS；
播放器累计播放到120秒，无错误。起始电池33°C，结束／最高40°C。

“High路线”指LLM、DiT和投影High，VAE原本就是Low；两轮VAE均已融合，全部权重不变。

| 指标 | 原High路线 | 全Low路线 |
|---|---:|---:|
| LLM | 633.920秒 | 514.529秒 |
| DiT，两步＋CFG1.5 | 180.086秒 | 157.146秒 |
| 融合VAE | 139.828秒 | 138.790秒 |
| 条件投影 | 13.610秒 | 8.770秒 |
| 其他 | 3.328秒 | 2.914秒 |
| **总耗时／RTF** | **970.772秒／8.090** | **822.149秒／6.851** |
| **进程RSS峰值** | **4401.99MiB／4.30GiB** | **3703.47MiB／3.62GiB** |
| LLM阶段内进程RSS峰值 | 4401.99MiB | 3420.82MiB |
| DiT阶段内进程RSS峰值 | 4219.89MiB | 3703.47MiB |
| VAE阶段内进程RSS峰值 | 4133.55MiB | 3645.70MiB |
| Prefill，513位置 | 13522.66ms | 14408.21ms |
| 纯decode，3000次 | 204.24ms/token | 163.35ms/token |
| Decode吞吐 | 4.90token/s | 6.12token/s |
| 首块发布 | 59.615秒 | 60.184秒 |
| 电池起始→结束 | 32→40°C | 33→40°C |
| 全包ROM | 3.536GiB | 不变 |

Low分项RTF为LLM4.288、DiT1.310、VAE1.157、投影0.073、其他0.024。
DiT累计24.761秒加载／创建和VAE累计24.299秒加载／创建已计入各自阶段。
纯decode共490056.59ms，排除采样、hidden复制和DiT／VAE等待。
采样共2909.39ms，hidden复制共18.55ms；最低系统可用内存2174MiB。

整链观测耗时下降15.31%，LLM下降18.83%，DiT下降12.74%；VAE仅下降0.74%，未增加新的VAE优化。
进程峰值少698.52MiB。主要收益来自LLM和DiT，prefill与首块等待没有改善。
两轮前缀float32数据逐字节相同，但低精度改变后续logits与随机采样结果，不宣称音频逐字节一致或音质等价。
两轮HAP版本和起温不同，每条路线只有一轮长测，因此这些差值是观测值，不能代替严格同温随机交叉对照。

Low本轮HAP SHA256：`4ed7a0c886ef3bb28a15b616d30a9783a0e6279addbdcb6ace13fcc096b6dd11`。
统一报告：`stream_lab_large120_all_low_2026_11_24`，保留此前High基线与短测隔离标记。

## 采样配方没有改变

LLM为temperature0.95、top-k250，未启用top-p和重复惩罚，独立随机种子2307。
本轮固定时长测速屏蔽EOS，生成满3000个音频code。DiT为两步Euler、CFG1.5，正态初始噪声另用种子2307。
VAE确定性解码，不做随机采样。仅切换CPU精度策略，没有同时调采样配方。

本表记录的是v0.1.16、seed2307。随后v0.1.17按用户要求将默认seed改成42，
LLM与DiT噪声均接收该参数，并在原生结果里分别记录`seed`、`noiseSeed`。
`tools/run_music_duration.ps1`新增`-Seed`，默认42；复现旧性能配方时需显式传2307。
改默认seed不修改本表历史数据，也不能把旧120秒成绩标为seed42测试。

## 另一次LLM独立长解码

此前还完成原生CLI的Low／3000次真实code历史解码：prefill12077.31ms，纯decode497918.51ms，
平均165.97ms/token／6.03token/s，末250次215.49ms/token，RSS峰值3292.63MiB。
该轮98次亮屏检查通过，电池33→39°C；没有运行DiT／VAE，也没有生成120秒音频。
它不是本表App整链的替代数据；用户改测全Low整链后，独立High长解码对照未继续执行。
