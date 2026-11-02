# LeVo Harmony 串行链路与两步 DiT 对照

后续90秒测试、精确参数/ROM和功率温度图见[90秒更新](CHAIN_COMPARISON_90S.md)，120秒结果见[120秒记录](CHAIN_COMPARISON_120S.md)。下表保留原30秒历史成绩，不能直接当作新版本同温/同前台状态消融。

2026-11-02，在同一台手机的签名 **LeVo Harmony v0.1.8** 内分开冷启动 App 测试。每次各运行一个模型组合，均为 750 个新 code / 30 秒有效长度、LM 单分支、保留 KV 增量解码、DiT 两步 / CFG1.5、同一 VAE。LM 全部生成并销毁后，再复用唯一 DiT 与 VAE 会话处理四个窗。没有并行运行两个模型。

## 当前模型和量化

| 项目 | 公开模型路径 | 小模型路径 |
| --- | --- | --- |
| LM | 已训练的主 Mixed LM，36层 / hidden2048；省略 track LM | 26层 Qwen3-0.6B 骨干＋未训练音乐适配器/头 |
| LM 权重 | 主体 W4 / 输出头 W8 / embedding BF16 | 主体 W4 / 输出头 W8 / embedding BF16 |
| LM 计算策略 | CPU High，4线程 | CPU Low，4线程 |
| 前缀 | 953位置，真实歌词/风格条件 | 514位置，原型固定输入 |
| DiT | 已训练39层 / hidden1200 / FFN4800，W8 | 22层 / hidden1024 / FFN2048、约1.954亿参数，W8，随机结构原型 |
| DiT 策略 | CPU High / 6线程，两步，CFG1.5 | 相同 |
| VAE | 同一预训练 Mixed VAE，W8 / CPU Low / 6线程 / NEON Sin | 相同 |
| 输出 | 真实歌曲 WAV | 有限值数值 PCM，丢弃输出，不冒充歌曲 |

小 DiT 参考 CosyVoice 的尺寸，保留音乐 Codec 的输入/输出合同；它不是直接加载 CosyVoice 语音 Flow 权重，也未训练音乐。不同 LM 的前缀长度与计算精度是各自当前已验证配置，不能把此表当作严格同精度的模型尺寸消融。

## 各阶段耗时与 RTF

阶段 RTF＝阶段秒数 / 30。LM 包含初始化、prefill、decode及释放；DiT/VAE列为前向及同步，其他包括条件准备、Codec加载、拼接和输出等。

| 阶段 | 公开模型耗时 | 公开模型 RTF | 小模型耗时 | 小模型 RTF |
| --- | ---: | ---: | ---: | ---: |
| LM 全阶段 | 99.912s | 3.330 | 15.315s | 0.510 |
| 其中 prefill | 21.191s | 0.706 | 1.877s | 0.063 |
| 其中 decode | 73.010s | 2.434 | 12.662s | 0.422 |
| DiT 四窗，共八步 | 38.541s | 1.285 | 8.672s | 0.289 |
| VAE 四窗 | 27.850s | 0.928 | 19.017s | 0.634 |
| 其他 | 3.952s | 0.132 | 1.151s | 0.038 |
| **整链** | **170.255s** | **5.675** | **44.155s** | **1.472** |

## 峰值内存

| 指标 | 公开模型 | 小模型 |
| --- | ---: | ---: |
| 进程 RSS 峰值 | 3887.3MiB | 1177.8MiB |
| HiDebug PSS 采样峰值 | 4035.8MiB | 1122.8MiB |
| HiDebug 图形内存 API 峰值 | 0KiB | 0KiB |
| 最低系统可用内存 | 2457MiB | 4564MiB |

推理后端均为 CPU。手机是统一内存系统，RSS/PSS不是独立显存；图形API两轮均返回成功和0KiB，不等于整台手机的GPU使用量为零。

HiDebug 的 NativeMemInfo.pss 按 **Pss＋SwapPss** 统计，所以不能与纯驻留 RSS 直接比较或相加；它可大于 RSS。见[官方 NativeMemInfo 说明](https://github.com/openharmony/docs/blob/master/en/application-dev/reference/apis-performance-analysis-kit/js-apis-hidebug.md#nativememinfo12)。

RSS由独立20ms采样与阶段采样共同保留最大值，HiDebug PSS名义500ms、图形内存名义1000ms采样，查询在线程中执行。后台查询累计耗时为公开模型46.340s / 小模型7.666s，其中包含等待，不能从墙钟耗时中直接扣除。采样会扰动性能，也可能漏掉更短的峰值。两次均检查前后亮屏且无中途息屏；未控制温度、调度和文件缓存。

## 与此前四步歌曲的关系

公开模型本轮的歌词、风格和750个生成 code 与此前 LeVo 音乐四步样例完全相同，保留了可试听的两步输出。此前四步 App 样例223.593s / RTF7.453；由于 App 上下文、运行时调度及采样不同，不能把总耗时差完全归因于减步数。

两步路径已另外接入 **LeVo 音乐 v1.0.3** 作为常用生成入口，默认30秒；该 App 内实际生成结果另列在 MUSIC_APP.md，不与本表的 LeVo Harmony 对照成绩混写。四步运行时和历史实验仍保留。

全部原始结果、各阶段采样峰值和签名包哈希归入统一 `reports/levo-test-report.json` 的 `serial_chain_comparison_2026_11_02`，不需要上传多份分日志。
