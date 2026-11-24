# LLM 算子实验：独立App与自动化baseline

## 交付范围

独立工程`llm-operator-app/`，包名`com.zhilee.llmoperatorlab`，不覆盖LeVo音乐、流式实验或原模型包。
手机执行LLM固定工作量的baseline／算子诊断，电脑执行候选库查询、CPLEX／DP与编译回归链。
本版不生成音乐、不调用DiT／VAE，也不把其计时冒充整首音乐RTF。

当前B0链接现有项目固定的MNN静态库，原图与权重不变，未插入候选kernel。
它是**可复现的现有MNN fork基线**，有编译时archive SHA256；没有声称已核验为未改动的上游MNN。
论文若要求纯上游B0，必须另建干净MNN版本、记录补丁和ABI兼容性，再跑同一协议。

## 目录

| 路径 | 职责 |
|---|---|
| `llm-operator-app/` | ArkTS界面、异步NAPI入口、独立签名／安装 |
| `runtime/include/levo/operator_lab/Baseline.hpp` | 唯一的App／CLI共用LLM执行逻辑 |
| `native-bench/LlmOperatorBaseline.cpp` | 无签名依赖的同源OHOS CLI入口 |
| `levo_harmony/operator_lab/` | 候选库、CPLEX、DP、隐藏查询、预算、HDC与回归 |
| `configs/operator_lab/baselines.json` | 采用最新B0–B8编号，逐项标注实际实现范围 |
| `configs/operator_lab/mnn_stage_commands.example.json` | 实际MNN编译／回归阶段命令模板 |
| `tools/build_llm_operator_app.ps1` | 构建；保留本地签名配置 |
| `tools/prepare_llm_operator_lab.py` | 复制LM与真实prefix／code，逐文件核对SHA256 |
| `tools/run_llm_operator_lab.py` | 单设备锁、温度、亮屏／前台、统一取回报告 |
| `tools/operator_lab.py` | 候选库研究与外部Agent／编译链命令入口 |

## 两套基线契约

这是当前的音乐Qwen派生模型，不是未经修改的通用语言Qwen评测。

| 项目 | 0.6B档 | 4B档 |
|---|---|---|
| 模型 | `larkstudio-small-0.6b-v1` | `larkstudio-base-v2` |
| 层数／hidden | 28／1024 | 36／2560 |
| 权重 | W4/group64、头W8 | 同左 |
| 输出类别 | 音频＋EOS共16385 | 同左 |
| 输入音频code偏移 | 151830 | 1 |
| 动态A8选项 | 2 | 0 |
| baseline浮点策略 | CPU High／4线程 | 同左 |
| Prefix | 513×1024真实TXT风格条件 | 513×2560真实探海NPY条件 |
| Hidden输出 | 不开启，与原token条件路线一致 | 保留consume对齐2560维输出 |

每轮一次真实513位置prefill，随后按同一历史逐次调用decode；不是重新prefill每个chunk。
KV从513递增到`513+decodeSteps`，每个重复轮次才reset。`attention_mode=8`是Flash Attention＋无KV量化，不是KV8。
默认64次decode、3轮：第一次单列，后两轮为权重已加载的预热重复。可以选择128／256次；
最长3000次需输入真实code历史足够长，不能用随机token补齐。初始真实短样例只有375个code。
CPU Low是另一精度实验条件，不能拿来替代“同精度比较搜索算法”的控制条件。

正常计时不启用算子回调：计量`forward＋readMap同步`，有限值检查、资源门槛、快照保存和hidden复制在计时外。
算子诊断单独运行：1轮prefill＋最多4次decode，记录名字、调用、形状、逻辑IO字节、MNN FLOPs字段和实际CPU后端字节数。
回调计时含调度等开销，可能包含嵌套区间；不能直接加总作为整模型延迟。
逻辑IO未包含卷积内置权重、Attention内部KV等全部流量，也不是实测DRAM或SRAM。

## 构建、签名与文件部署

```powershell
./tools/build_llm_operator_app.ps1
```

在DevEco打开`C:\levo-harmony\llm-operator-app`，进入Project Structure → Signing Configs，
为`com.zhilee.llmoperatorlab`单独勾选Automatically generate signature并Apply。
新包名不能沿用旧App的调试Profile；签名、SDK本地配置、权重不提交Git。

```powershell
./tools/build_llm_operator_app.ps1
./tools/deploy_llm_operator_app.ps1 -Device DEVICE
python ./tools/prepare_llm_operator_lab.py --device DEVICE `
  --root /data/app/el2/100/base/com.zhilee.llmoperatorlab/haps/entry/files
```

准备工具只复制当前Stream App的LM文件到新包；不重命名或改写旧权重。
它选取最近已完成、模型和风格匹配的真实prefix与code样例，检查形状和token范围，逐文件核对SHA256。
已存在的case只能核验复用，不能静默覆盖。App加载时校验架构、文件尺寸与编译MNN指纹；SHA256由主机准备／执行工具核验。
当前手机有充足空间，但新App会有独立模型副本；没有跨UID直接读取旧App私有目录。

## 真机baseline与统一报告

```powershell
python ./tools/run_llm_operator_lab.py --device DEVICE --tag llm-b0-small-r1 `
  --model small --precision high --steps 64 --rounds 3 --threads 4
python ./tools/run_llm_operator_lab.py --device DEVICE --tag llm-b0-large-r1 `
  --model large --precision high --steps 64 --rounds 3 --threads 4 `
  --cooling-seconds 600 --max-start-battery-c 30
python ./tools/run_llm_operator_lab.py --device DEVICE --tag llm-ops-small-r1 `
  --model small --mode operators
```

手机需解锁并保持前台。工具不唤醒锁屏、不杀已有推理；单设备文件锁避免两个自动任务重叠。
开始前和冷却后均检查旧Stream任务及新App任务是否空闲。计时记录亮屏、前台、起温／过程温度和HAP哈希。
没有满足前台／亮屏控制时保留数据，但不标为受控App性能结果。

手机只导出`llm-operator-report.json`，主机同名报告合并原生结果与控制记录。
私有浮点快照、code和模型仅在本地；用于后续数值回归，不要求用户上传多份分日志。
仓库公开摘要继续汇总到`reports/levo-test-report.json`，不能提交原始私有数据。
RAM是20ms采样的进程RSS；不是GPU显存、CPU cache容量或软件可控SRAM。

### 不依赖App签名的同源CLI

```powershell
cmake --build ./native-build --target llm_operator_baseline -j 4
python ./tools/prepare_llm_operator_lab.py --device DEVICE `
  --root /data/local/tmp/llm-operator-baseline-CASE
python ./tools/run_llm_operator_lab.py --device DEVICE --tag cli-small-r1 `
  --transport cli --root /data/local/tmp/llm-operator-baseline-CASE --model small
```

CLI与App共享执行逻辑，但没有App前台调度身份，因此CLI成绩单列，不能直接替代App延迟。

## Phase1：有限候选库的数学对照

本版实现**固定串行顺序＋连续区间候选**；不声称实现任意DAG的Korch算子拆解和融合调度。
每个primitive须有已验证的native singleton作为初始列。候选metadata与私有实测outcome分开，
选择策略看不到未查询的真实成本或成功状态。失败查询也消耗相同的query预算。

CPLEX可安装在项目隔离目录，不改其他模型环境：

```powershell
python -m pip install --target .operator-deps --no-deps cplex
python ./tools/operator_lab.py pool-study --pool measured-pool.json --output study-new.json `
  --budget 100 --seeds 42,43,44 --policies random,greedy,frozen_dual,dual
```

候选文件schema1含`nodes`、`signature`、`scope`、`candidates`、`outcomes`。
候选字段：`id,start,end,traffic_bytes,scratch_bytes,flops,predicted_ms,baseline`；
outcome字段：`status,latency_ms,relative_l2,source`。状态为passed／compile_failed／correctness_failed／device_failed。
失败项不能有有效latency，不能进入执行方案；不同设备、线程、精度、输入的measurement signature不能混库。
本版尚未建立100–500个真实融合kernel库。单元测试的合成小库明确标为synthetic_test，CLI必须显式`--allow-synthetic`，不算端侧性能。

| 实现 | 当前语义 |
|---|---|
| Random／Greedy／FLOPs／Memory | 同隐藏候选库的预算发现策略 |
| Frozen Dual | 第一轮LP对偶固定，后续不更新 |
| Dual | 每轮已知列重解LP，使用预测成本pricing；无全局保证 |
| Roofline／Dual+Roofline | 要求显式实测校准；启发式预测，不假装芯片理论峰值或可靠全局下界 |
| Exact known-pool CG | 实际成本已全部知道的独立实验；精确有限池pricing可给LP下界，不把这些成本算作免费在线发现 |
| Full-pool Oracle | 对提供的全部合格列解整数规划；只有求到最优才报告Gap基准 |
| DP／Shortest Path | 串行区间且无累计traffic约束时的精确对照 |

受限主问题有Phase-I人工列保证LP基；含人工列的解不报告成真实可行执行方案。
逻辑traffic预算可加和；scratch仅作单kernel可行性／max约束，不错当成可加和的物理SRAM峰值。
最终真实RAM和整模型延迟必须由实际部署再验证，不能用候选成本求和替代。
CPLEX默认不接受MIP相对gap作为“精确Oracle”；超时可行解不冒充最优。

## Agent／编译／数值／整图自动链路

```powershell
python ./tools/operator_lab.py candidate --task task.json `
  --commands ./configs/operator_lab/mnn_stage_commands.example.json `
  --workspace ./reports/local/new-candidate --max-compiles 100 `
  --max-device-profiles 100 --max-seconds 3600 --epsilon 0.001
```

task至少包含：`id,signature,baselineReport,device,remoteRoot,model`。
`signature`来自真实B0主机报告的measurementSignature；`baselineReport`是含参考logits／hidden的目录。
`candidateSource`可指定workspace内的C++文件；省略只允许做identity链路自检，不能当成优化结果。
MNN／SDK位置可用`mnnRoot,mnnLibrary,sdk`覆盖。先构建参考B0，再使用同输入、同权重和同精度运行此链。

默认阶段命令用实际CMake交叉构建新CLI，不覆盖B0二进制；正确性和device／graph阶段均走HDC真机。
候选必须实现可选setup及实际调用计数ABI：

```cpp
extern "C" void LlmOperatorCandidateSetup();
extern "C" uint64_t LlmOperatorCandidateCalls();
```

setup在LLM加载前调用；候选负责正确初始化并注册所需MNN后端功能。
有candidateSource时，真实前向未观察到调用就失败，不能静默回退后宣称候选加速。
计数排除模型load期间调用。编译器、ABI、权重、输入、precision／threads／token历史均有记录。
正确性比较固定历史的logits，4B额外比较consume-hidden；默认L2门槛0.001，可显式配置。
不能用音频峰值有限代替LLM数值等价。

只有单kernel通过不会成为已部署的整图成果：graph阶段要重新跑完整固定LLM工作量并测得真实延迟。
本MNN adapter的latency单位是513位置prefill＋指定次数decode的预热工作量，不是整首音乐或任意单算子耗时。
需要独立单kernel测量时必须提供相应stage adapter；当前没有伪造500个候选或自动替代原MoKA评测。

可在commands前加入`agent`的显式argv数组，传入task／workspace／result路径；没有shell拼接。
Agent stage输出签名和实际totalTokens；缺失计费数据记unknown，配置token预算时停止后续开发。
外部provider仍须自行强制单次API额度；本地账本不保证第三方计费硬配额。
默认流程不调用任何Agent或付费API。失败编译和设备尝试计入预算，所有搜索方法必须使用同一套开发工具和预算。

## 参考与未完成部分

- [Korch官方仓库](https://github.com/humuyan/Korch)：原始实现面向NVIDIA profiler／TVM，本文串行区间参考不是它的完整Harmony移植。
- [MobileKernelBench／MoKA官方代码](https://github.com/zeezou-isee/MobileKernelBench_git)：有MNN／ncnn管线和Agent，现有设备管线以ADB／Android NDK为目标，需要独立HDC／OHOS适配。
- [Ansor论文](https://www.usenix.org/conference/osdi20/presentation/zheng)：参考调度搜索与成本模型，不把本版简单预测器称作Ansor复现。

当前真实实现范围以`configs/operator_lab/baselines.json`为准。ncnn、TVM MetaSchedule、原版MoKA与Korch的端侧移植没有完成或实测。
下一阶段应采集并验证真实独立kernel／融合候选，再跑同预算查询研究，最后接入Agent并验证实际模型部署。

## 本次验证状态（2026-11-24）

- Python完整测试76项通过，含CPLEX实际求解、DP、动态／固定对偶、失败预算与设备空闲门槛。
- 独立App的unsigned HAP和同源OHOS CLI均已交叉构建成功。
- 命令行以100个**合成测试候选**完成8种策略×3个seed的24次查询研究；CPLEX整数Oracle与DP在数值容差内一致。该自检不是实机kernel性能。
- 两套真实音乐LM及固定输入已复制到手机独立CLI测试目录并核对SHA256；没有改写旧App模型。
- 新App独立签名配置尚未完成，手机息屏；本次未执行新的真机推理，不报告token/s、加速比或实机候选库成果。完成手动签名、解锁后按上面命令运行。
