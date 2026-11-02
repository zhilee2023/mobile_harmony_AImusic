# 公开 LeVo Mixed 权重实际音乐生成验证

已从用户下载的 `C:\Users\lizhi\model.pt` 完成真实预训练 Mixed LM → 匹配的 Mixed DiT → VAE → 30 秒双声道 WAV。音频文件结构和非静音检查通过；听感尚待用户验收。

这不是随机音乐头的 0.6B 原型，也不是完整官方分轨推理：使用公开检查点的 36 层、hidden=2048 主 Mixed LM，省略 track LM 分支。

## 本次量化与测试边界

| 部分 | 本次设置 |
| --- | --- |
| 主 Mixed LM | W4，输出头 W8，音乐输入 embedding BF16 |
| 条件 | 真实歌词和风格 embedding；953 位置前缀，无参考音频 |
| PC 采样 | CFG 1.5，top-k=50，temperature=0.9，750 个音乐 token |
| PC Codec | 匹配的 DiT W8，20 步；VAE W8 |
| 手机 Codec | 同一份真实音乐 token；DiT W8，4 步；VAE W8，CPU |

最初手机测试仅执行 Codec；下面追加了整条手机链路。当前未实现 W4A4KV4。

## 2026-11-02：整条真机链路完成

手机原生 CLI 独立运行：歌词/风格条件前缀 → 真实公开预训练主 Mixed LM → 四步 DiT → VAE → 30 秒 WAV。条件 embedding 从原始训练权重精确准备；手机执行完整 LM prefill 和 750 个音乐 token 的自回归采样，不再使用 PC 生成的 token。尚未接入 App 的公开大模型入口。

按用户要求，**LLM CFG=1（单条件分支），DiT CFG=1.5（保留双分支）**。LM 主体 W4、头 W8、CPU High / 4 线程；DiT W8 / CPU High / 6 线程；VAE W8 / CPU Low / 6 线程，复用原小模型的 NEON Sin。已修正独立探针中 hook 晚于 VAE 建图的问题，本轮实际命中 576 次；没有以挂接开关代替调用证据。

| 项目 | 真机实测 |
| --- | ---: |
| 有效音频 | 30.00 秒 |
| LLM 总阶段（含加载、两点数值探针、释放） | 132.787 秒 |
| 条件前缀 prefill | 20.046 秒 |
| 增量 decode | 108.420 秒 / 6.91 token/s |
| 四个 DiT 窗口，共 16 步 | 79.655 秒 |
| 四个 VAE 窗口 | 32.418 秒 |
| Codec 总阶段 | 114.618 秒 |
| 通过 WAV 保存的整链时间 | 247.545 秒 |
| 完整 RTF | **8.252** |
| 20 ms 采样进程 RSS 峰值 | 3956.8 MiB |
| 最低系统可用内存 | 2313.0 MiB |

输出：`artifacts/whole-phone-dit4.wav`。48 kHz 双声道、30 秒、5,760,044 字节，结构与非静音检查通过；音乐质量仍待试听。整轮记录前后 AWAKE，Last Screen Off 相同。后半段 Codec 明显变慢，未控制温度或后台负载，不能直接与较早的独立 Codec 基线做因果提速比较。

三次未完成尝试也保留：双 CFG 上下文 High 分别触发 5 GiB RSS 诊断上限和可用内存门槛；公开大 LM 的 CPU Low 加载在 MNN allocBuffer 断言处终止。最终可用组合是 CPU High 的单 LM 分支，并非大模型 CPU Low 已验收。

原 0.6B 小模型、App 与优化方案保留，详见主工程的 `docs/SMALL_MODEL_OPTIMIZATION_BASELINE.md`。上述公开大模型 RTF 不能替代小模型历史速度。

## 输出

- `artifacts/public-mixed-30.00s-dit20-w4-w8.wav`：PC 20 步，48 kHz 双声道，30 秒。
- `artifacts/public-mixed-phone-30s-dit4.wav`：手机 4 步，48 kHz 双声道，30 秒。
- `artifacts/whole-phone-dit4.wav`：手机自己生成 token 并完成四步 DiT/VAE 的整链样例。
- `levo-public-music-report.json`：统一报告，含来源校验、映射误差、量化配置、耗时、手机内存及 WAV 检查。

早期 PC token → 手机 Codec 实测 73.48 秒，Codec-only RTF 2.449；20 ms 采样进程 RSS 峰值约 1511 MiB，最低系统可用约 3843 MiB。该 RSS 是进程内存，不是独立显存，也不包含 PC LM。此早期版本尚未真正命中 NEON Sin，不能把整链新结果与它混写。

PC LM 生成 750 token 约 251.13 秒；PC 20 步 Codec 约 293.12 秒。当前是能输出音频的验证版，速度和音质仍需后续优化。

前两份音频使用同一组 PC token，但初始噪声生成器不同，不能把听感差异全部归因于 DiT 步数。新整链音频改由手机采样，且 LM CFG 已关闭，token 与前两份不同。30 秒由最大 token 数截断，未遇到 EOS，不保证完整唱完歌词。Mixed-only 路线与量化的质量影响也仍需试听。

## 校验

原始检查点 12,899,965,446 字节，SHA256：

`dc763aa9a76a22a87597c2faf9a51c24d13349ac754699b37e9068b483639def`

固定来源 `lglg666/SongGeneration-v2-large@115805364ad74479fb3764fe65970c92faeb1a5a`。

不执行检查点 pickle；按已审计的张量元数据抽取必要权重。条件 embedding 对官方固定源码的 BF16 输出逐元素一致；HF 到 MNN Python 映射在两个检查位置的 logits 相对 L2 小于 5e-7。量化后两处检查 top-1 保持一致，logits 相对 L2 约 0.150 / 0.122，不能据此宣称整曲无损。

## 保留与复现

旧 App、0.6B 原型、CosyVoice、ACE 项目和原有实验入口均未替换。新增脚本与输出隔离在本目录。

执行顺序为 `extract_checkpoint.py`、`prepare_conditions.py`、`check_conditions.py`、`export_mixed_mnn.py`、`generate_codes.py`、`decode_music.py --steps 20`。`native/PublicMixedCodec.cpp` 是独立手机解码探针，复用原有 Codec 权重，不替换 App。`collect_report.py` 汇总结果并检查两份 WAV。

依赖和已有权重路径见各脚本顶部；模型、大型中间文件与私人设备日志不应提交到 Git。
