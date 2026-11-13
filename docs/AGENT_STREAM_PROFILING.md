# Agent：定位工程、登录转换服务器与整链 Profiling

2026-11-13。本文用于当前独立 **LeVo 流式实验 v0.1.7**，先测0.6B完整链路，再分析热点。
本机工程是 `C:\levo-harmony`，GitHub为 `zhilee2023/mobile_harmony_AImusic`。
先读根目录 `AGENTS.md`、[当前链路](STREAM_LAB.md)、[本轮DiT布局结论](DIT_ONNX_LAYOUT.md)。

## 从哪里开始

| 用途 | 当前目录或入口 |
|---|---|
| 独立实验App | `C:\levo-harmony\stream-app`，包名 `com.zhilee.levostreamlab` |
| 当前推理链 | `runtime/include/levo/experimental/StreamLab.hpp` |
| 原生模块测试 | `native-bench/`；现有通用入口 `StreamLabBench.cpp` |
| 本轮整链热点入口 | `native-bench/StreamLabChainProfile.cpp` |
| 本轮真机执行脚本 | `tools/run_stream_chain_profile.ps1`，先阅读参数声明或执行 `Get-Help` |
| 原生构建输出 | `native-build/`，本机已配置的HarmonyOS arm64工具链 |
| PC模型与捕获 | `models/`，均为本地忽略资产，不上传 |
| 原始日志和本机访问小抄 | `reports/local/`，不上传 |
| 唯一公开测试报告 | `reports/levo-test-report.json`，新增独立节，保留全部历史节 |
| 签名后的HAP | `stream-app/entry/build/default/outputs/default/entry-default-signed.hap` |

不要修改旧 `music-app/`、原App数据或历史模型来让测试通过。
当前实验0.6B是28层Qwen，不是仓库早期26层随机学生；4B实验包是36层Qwen，
也不能与历史公开LeVo的Llama LM混称。没有匹配权重时停止该分支，不能随机补权重生成“成功结果”。

## 本机工具

```powershell
Set-Location C:\levo-harmony
$Hdc = 'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$CMake = 'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native\build-tools\cmake\bin\cmake.exe'
$Python = 'C:\Users\lizhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $Hdc list targets
# 明确选择此次连接设备，不把旧序列号写死进脚本。
$Device = '<hdc-target>'
& $Hdc -t $Device shell 'pidof com.zhilee.levostreamlab'
```

本机MNN依赖目前来自 `C:\cosyvoice-harmony\third_party\MNN`，静态库位于
`C:\cosyvoice-harmony\app\entry\.cxx\default\default\debug\arm64-v8a\mnn\libMNN.a`。
迁移机器先核对本机CMake cache与工具链；不要凭文件名假定库的编译选项或硬件指令支持。

手机App文件根目录：

```text
/data/app/el2/100/base/com.zhilee.levostreamlab/haps/entry/files
  model-bundles/larkstudio-small-0.6b-v1
  model-bundles/larkstudio-base-v2
  experimental/music-head-r1
  clap-styles/muq-text-warm-pop.npy
  stream-runs/<run-id>/
```

App沙箱内部的同一根目录可表示为 `/data/storage/el2/base/haps/entry/files`；
CLI不要未经核对直接使用App沙箱别名。调试访问依赖已授权HDC，不表示普通App能跨沙箱读取权重。
首次部署才使用 `tools/deploy_stream_app.ps1`；模型已存在时先核对manifest/哈希，避免重复复制GB级资产。

## 服务器登录与职责

该服务器用于官方OMG转换与CPU数学验证，**手机的ARM/NPU速度必须在手机测**。
本机真实访问参数保存在忽略文件 `reports/local/AGENT_SERVER_ACCESS.md`；不要把IP、私钥路径、
用户名、SSH配置或登录日志追加进公开报告。2026-11-13已使用现有known_hosts完成一次只读登录验证。

公开复现命令只写参数：

```powershell
$ServerHost = '<authorized-host>'
$ServerUser = '<authorized-user>'
$ServerKey = '<local-private-key-path>'
& ssh -i $ServerKey -o IdentitiesOnly=yes -o BatchMode=yes `
  -o ConnectTimeout=8 -o StrictHostKeyChecking=yes "${ServerUser}@${ServerHost}" `
  'test -d /home/ubuntu/levo_dit_layout_20261113_r1 && echo experiment_directory=present; test -f /home/ubuntu/cosyvoice-omc/toolkit-status.json && echo toolkit_status=present'
```

不要输出私钥内容、关闭SSH主机校验或因旧别名失效就删除全局known_hosts。
如果再次报告主机密钥变化，核验新指纹后再处理，不能自动接受。

已确认的远端目录：

* `/home/ubuntu/levo_dit_layout_20261113_r1`：上轮首FFN ONNX、共享参数、OM、转换日志。
* `/home/ubuntu/cosyvoice-omc/toolkit-status.json`：既有官方OMG路径与库目录配置。

新实验使用新的目录与输出标识，保留原件；转换入口是 `tools/convert_dit_layout_remote.py`。
该脚本默认portable OM，明确支持的平台才生成芯片专用OMC。不要将旧芯片OMC改名冒充新芯片产物。
服务器上其他已有任务（包括Oracle资源申请服务）不属于本轮测试，不能停止、重启或改配置。

## 本轮基准必须保持的配置

| 阶段 | 0.6B基准 |
|---|---|
| LM | Qwen3 28层、H1024；主体W4/g64、输出头W8、输入BF16；CPU High、4线程 |
| 前缀 | start＋511歌词位置＋1风格位置，共513位置；默认MuQ文本512维NPY |
| 条件 | 生成code查 `cond-token.f16[16384,1024]`；不套用4B mu投影 |
| DiT | 22层、H1200、FFN4800；W8/g64，CPU High、6线程；**2步Euler＋CFG1.5，batch2** |
| DiT激活 | FFN为tanh近似GELU；时间/调制分支SiLU，不能替换ReLU当作等价优化 |
| VAE | 共享Oobleck decoder，W8/g64，CPU Low、6线程，NEON正弦 |
| 窗口 | 256帧/10.24秒，hop192帧/7.68秒，重叠64帧/2.56秒 |
| 生命周期 | 同一请求三网复用，共驻但计算串行；结束释放；不跨请求保留隐藏会话池 |

CFG无条件分支只清空mu条件，保留latent/context/mask；公式为
`uncond + 1.5 * (cond - uncond)`。不要为了更快擅自关CFG、改步数或改为batch1。
LM生成时保留KV，不重新prefill每个chunk。`prefetch`默认关闭，原CPU并行候选曾更慢。
线程数只是请求值，不表示绑核或CPU物理核心类型。

## 如何测试整条链

执行脚本默认0.6B、375帧/15秒，分别启动不带逐算子回调的baseline进程和operators进程；
二者均为完整LM→DiT→VAE链、相同歌词/NPY/seed、两窗，使用4/6/6线程、两步CFG1.5。
专用profiling构建不改变App默认行为。

```powershell
Set-Location C:\levo-harmony
$RunId = 'stream-chain-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$SeedResult = 'C:\levo-harmony\reports\local\profile-seed-app-result.json'
# SeedResult为必填；迁移机器请先取回一份真实App result.json，不能用随机歌词替代。
& .\tools\run_stream_chain_profile.ps1 -Device $Device -RunId $RunId -Frames 375 -Model 0 -SeedResult $SeedResult
```

单轮原始产物放在 `reports/local/<RunId>/`。原生底层接口为：

```text
levo_stream_chain_profile appRoot choice frames request.json outputDir baseline|operators
```

`-Model 1`是另一个4B架构，不能把它当作同模型线程对照。
先完成0.6B短轮，再决定是否扩大到60秒或4B；不要并发启动两个整链任务。

执行脚本在`/data/local/tmp/levo-chain-profile-<RunId>/`创建独立输出及只读资产链接，
不把CLI生成结果写入原App任务目录。每个模式最多900秒，旧输出目录会被拒绝覆盖。
执行期间的心跳原始输出保存在`baseline.log`和`operators.log`；失败时保留已写结果，不反复重启。

### 已有短测，先分析再重跑

本机`reports/local/chain-profile-20261113-r1/`已完成一次15秒双进程验证：
baseline 30.641秒，operators 32.476秒；375个code及两份WAV均逐字节一致。
逐算子数据已覆盖`llm.prefill`、`llm.decode`、`dit.forward`和`vae.forward`。
外部Agent应先读这些产物；要复测时使用新的RunId。
这两次均为CLI生成，无播放验收，不是新的App端到端成绩或NPU结果。

每个模式的`profile.json`有`generation`和`profile`两部分；
`profile.spans`为阶段墙钟，`profile.operators`为按名称累计的回调时间、调用数与首个输出shape。
prefill与decode的计时包含强制读取输出同步；sampling和条件查表另列。
嵌套span不可相加，算子回调也不保证互斥；不要把这些累加值称为硬件kernel周期。
`pcm.write`含PCM量化/交错与WAV写入，Graph的`output`含复制和有限值检查。
resize目前计入Graph setup，未进一步拆开；不能伪造一个单独resize耗时。

1. 查当前任务状态，已有任务运行时继续观察，不因日志暂时不更新就杀进程或重启任务。
2. 用户手动解锁并保持实验App前台；只亮锁屏不算前台。记录屏幕与前后台状态，
   若中途息屏则保留结果但标注，不与受控亮屏成绩混作优化收益。
3. 第一轮先做15秒真实歌词＋同一NPY的整链基准。固定seed、temperature、topK、
   模型哈希、CFG、线程、窗口和有效输出长度。不能用公式code或220Hz播放自检声评价音乐生成RTF。
4. **正常计时轮不启用逐算子回调。** 从准备LM到发布最后一段的连续墙钟计算总RTF；
   单列LM加载/prefill/decode、DiT加载/resize/两步、VAE加载/resize/前向、PCM衔接/WAV发布。
   首块与后续稳态chunk分开；不要从不同轮次拼接阶段时间冒充一次端到端成绩。
5. 用同一输入和配置另跑热点轮。逐算子回调的耗时只用于定位热点，
   必须标注debug/callback开销，不能把它当作正常RTF或与上一轮作无条件速度比。
6. LM热点轮区分prefill和单token decode，统计实际生成token数与真实decode计时口径。
   任何重新prefill、teacher-forcing或KV重置都须单列，不冒充连续自回归。
7. DiT与VAE至少报告按类型汇总、最慢节点、输入/输出shape、调用数及前向累计耗时。
   采样回调不代表硬件kernel级时间；Raster也不全是等量DRAM搬运。
8. 记录20ms RSS采样峰值、系统最低可用内存、温度和电量；RSS/PSS不是专用显存。
   当前App有6GiB RSS/650MiB系统余量边界检查，不能据此保证任意长度不会OOM。
9. 核对输出有限、实际样本数、有效音频时长、完整/部分状态；App轮再验播放硬件位置和错误回调。
   CLI生成WAV不等于App已试听；正常退出不等于客观音质已通过。
10. 原始日志只留 `reports/local/`。把参数、哈希、连续整链结果、热点摘要、失败与限制
    合并到 `reports/levo-test-report.json` 的新日期节；保留旧节，最终给用户一份统一报告。

若修改Python行为，运行完整 `python -m unittest discover -s tests -v`；
若修改App运行时，构建并在实际选定模型上做生成与播放验收。纯CLI诊断不能冒充新App已安装。
构建App使用 `tools/build_stream_app.ps1`，签名只用现有本地配置，不提交证书和profile。

## NPU结果如何使用

当前产品链仍是MNN CPU。已有CANN结果只来自真实首FFN，空间Conv含CPU布局桥接约3.7ms，
不是22层DiT耗时，也没有改变App整链RTF。与其同参数FP32展开的数值对齐，
不等于与当前MNN低内存量化算术完全一致；完整velocity和音频验收尚未完成。

`StreamLabNpuProbe.cpp --profile`曾出现设置选项/Build成功但executor创建失败，记录为失败。
普通运行通过不等于已取得逐kernel NPU硬件轨迹；设备顺序请求也不证明每个节点物理上只在NPU执行。
保留正常执行与profiling失败记录，先做CPU整链热点，再有证据地决定连续NPU子图边界。

## 给下一个Agent的最短指令

> 到 `C:\levo-harmony` 读AGENTS.md和本指南；从忽略的本机访问小抄读取服务器参数，
> 用严格SSH校验只读确认转换目录。不要动旧App、模型、签名或其他服务器服务。
> 保持0.6B两步CFG1.5，先做15秒正常整链计时，再另跑LM/DiT/VAE热点轮。
> 报告真实时长、每阶段/每chunk/总RTF、LM token/s、RSS峰值、屏幕状态、模型与HAP哈希。
> 回调热点不是正常RTF，CPU App不是整DiT NPU部署。只上传代码、说明和统一测试报告。
