# 执行频率、共享开发与两阶段列生成

本版将“开发哪个实现”与“实现在哪里运行”分开。代码在`operator_lab/workload.py`、`patterns.py`、`mining.py`及`subgraphs.py`。
所有物理性能结论仍需完整模型复测；下面的LP／MIP最优性针对提供的有限成本表，不等于实机全局最优。

## 1. 运行次数不等于开发次数

`Kernel k`是一个兼容语义／Shape范围／Layout／dtype／量化ABI／ISA的实现；`Placement p`是一个合法使用位置，覆盖primitive集合S_p。
开发变量y_k只出现一次，位置变量x_p可以出现多次：

\[
\min\sum_p C_p x_p,\quad
\sum_{p:i\in S_p}x_p=1,\quad x_p\le y_{k(p)},\quad
\sum_k D_k y_k\le B_{remaining},\quad x_p,y_k\in\{0,1\}.
\]

其中`C_p = f_p * c_p + extra_total_ms`。已经加权的耗时必须使用`cost_unit=workload_total`并记录`counted_frequency`，不会再乘f_p。
若同一位置同时经历Prefill和Decode等不同上下文，先分别求`Σ_context n_context * latency_context`，再以总量输入。
不同位置可以有不同单次延迟，不能仅按算子类型假设相同速度。

原生／已经开发的实现新增D_k为0；已失败的开发尝试进入`spent`，不能退还。
当前master支持一项带单位的开发预算；真实Agent账本仍单独限制编译次数、设备次数、时间和Token。预计成本不能代替实际硬限额。

### 一次性开发投入

对于新实现，把同类 Kernel 从提出候选到通过编译、数值和真机门槛所需的投入记为 \(d_k\)。研究初期可以用预计开发尝试次数作为单位，有足够记录后可换成工程分钟，或将相关资源按一致的成本单位折算。实际失败尝试照扣额度。已经发生的成本为沉没成本，后续部署已验收实现的新增成本为零。

`--development-weight` 设置一单位开发成本的累计时延等价 \(\lambda\)。如果 \(N\) 次运行会重复同一个已定义的工作量，`--expected-runs N` 使每次工作量的目标成为 \(T_{\rm runtime} + (\lambda/N)\sum_k d_k y_k\)。一次工作量内各位置的运行频次仍只计一次。输出分别列出 `runtimeMs`、`developmentCost`、`developmentPenaltyMs` 和组合值 `objectiveMs`。硬预算以原单位检查，生命周期摊销只改变目标权重。

合成链 `A–B–C–D–E–B–C` 的三种有效选择是原版 19 ms、复用 BC 13 ms、额外开发 EB 后 12.5 ms。两种实现各需一单位，硬预算允许两个实现时，实际比较的阈值是 \(\lambda/N=0.5\) 与 \(6\)。在阈值处相邻方案并列。额外 EB 的净收益是全局重排后节省的 0.5 ms 减去它的新增成本惩罚。BC 的两次复用只付一次。

```powershell
python tools/operator_lab.py pattern-plan --catalog configs/operator_lab/fixed_development_example.json `
  --allow-synthetic --budget 2 --development-weight 1 --expected-runs 1 --mode mip --output bc-eb.json
```

覆盖冲突由等式阻止。候选合法性、数值误差和兼容性仍须独立验证，`compatibility`字符串是显式合同，不能仅凭Shape相同自动认定合法。
同一覆盖位置的候选必须对应相同执行频率；只覆盖部分分支／KV区间时，要先拆分调用类，不能拿一次融合覆盖全部重复实例。
工作区上限只是单次额外workspace过滤，逻辑traffic是整工作量的估计，均不声称测得真实DRAM、CPU SRAM或整图RAM峰值。

## 2. 两种记账方式不能混用

- **逐实例图**：每个真实调用已经是图上的节点／位置，不再额外乘重复次数。
- **模板／静态位置图**：位置保留f_p，单次耗时只乘一次。

旧区间池增加`costDomain=single_invocation|template_workload`。Outcome和Pricing predictor必须同域，否则拒绝加载。
模板模式里，如果一列覆盖某模板的全部N_i次合法调用，可以将覆盖行`N_i*A_ik*x_k=N_i`除以N_i，继续用0／1覆盖系数；对偶也随行缩放。不能把缩放后的π再乘N_i。
部分调用覆盖、不同分支或不同数据布局不满足该归一化假设，须拆图／扩展系数。

## 3. 主问题与Pricing

LP放松后，覆盖对偶π_i自由；链接行`x_p-y_k<=0`的α_p非正；开发预算对偶λ非正；逻辑traffic预算对偶θ非正。
位置列的reduced cost为：

\[
\bar c_p=C_p-\sum_{i\in S_p}\pi_i-\alpha_p-\theta T_p.
\]

λ通过共享y与α传递开发约束。不能在每个使用位置再加一次完整开发成本，或漏掉链接对偶。
y 变量的 LP 约化成本还包含它在目标层的系数 \((\lambda_{dev}/N)D_k\)。若用 \(\lambda_{budget}\le 0\) 表示这个最小化模型中的硬预算对偶，约化成本为 \((\lambda_{dev}/N)D_k+\sum_{p:k(p)=k}\alpha_p-\lambda_{budget}D_k\)，这里省略上界对偶。位置列的公式保持原样，开发罚项只由共享的 y 承担。
若改为最大化净收益的组合列模型，一列代表同一实现的一组可行部署位置，且有“每个实现至多一个组合”的约束，则定价应扣 \((\lambda_{dev}/N+\mu)D_k+\sigma_k\)。其中 \(\mu\geq0\) 是硬预算对偶，\(\sigma_k\geq0\) 是实现组合约束的对偶。精确定价仍要求这些行和可行列空间已定义。
可行性阶段先最小化人工质量；此时真实运行列的目标系数为0。人工列固定0以后，再优化真实延迟。

### 已实现的精确有限库模式

`exact_catalog_cg`预先建立**所有已知实现的y变量与全部已知位置的链接行**，仅逐步加入x列。
未加入位置的链接行暂为`-y_k<=0`，其对偶已存在，因此上面的Pricing具有明确含义。
候选metadata及实际成本已全部已知，不把这些成本当成免费Agent发现。
完整已知目录pricing闭合可给数值容差下的LP成本界；受限列上的整数解不保证完整目录整数最优，另算fullInteger作Oracle。

### 未来在线未知Pattern模式

新Pattern可能同时新增y变量与链接行，不能直接沿用旧LP对偶当成标准列生成证书。
可用启发式共享收益／开发成本筛选，或使用实现＋非冲突位置集合的bundle定价，再重解master。
bundle内必须处理位置冲突和共享启动费用；成功概率／Roofline都是预测项，不自动构成下界。
本版没有声称完成无限新Pattern的精确定价、分支定价或原版Korch／MoKA移植。

## 4. 重复模式与频率采集

`mine-repetitions`当前识别**重复单节点提示**：保留阶段、归一化层角色、实际输入输出Shape和后端信息，输出调用位置和次数。
它没有依赖边、完整算子属性、Layout、权重量化ABI或ISA证明，因此输出`reuseProven=false`、`dependencyLegalityProven=false`。
不把同类型Conv都当成同一kernel，也不把不同层的非相邻位置直接融合。
Prefill、Decode、DiT步骤和VAE窗口应分成独立上下文；Attention还必须按KV长度分桶。
短诊断向长Decode外推须显式允许，并标为估计，不能对5000个token套用头两个token的Attention延迟。

### XY／XYZ连接子图挖掘（已实现）

`mine-subgraphs`补充真实MNN张量边上的2／3节点枚举，支持链、分叉、汇合，不依赖日志排列顺序。
它使用Op的`inputIndexes/outputIndexes`；按输入端口、共享外部输入、必要外部输出、算子子类型／属性、静态量化元数据、实际阶段Shape及后端记录匹配。
不同层的权重不相等，但可以作为同一实现的不同参数：匹配保留参数长度和外部存储大小，去除权重值／文件偏移；不能共享错误的权重指针。
常量输入的内容以摘要参与匹配，不把`ADD常数1`和`ADD常数2`合并；未知内容的外置常量保留存储身份摘要，不能仅凭大小合并。图和完整私有导出均不上传。

只接受依赖连通且凸的区域：一条依赖路径若离开区域再进入，不能直接做成一个融合节点。
区域内中间结果若仍有外部消费者或是图输出，必须作为融合输出保留，不能为了省读写删除。
Attention／KV／控制流／Raster虚拟区域／未知Extra算子不自动纳入；当前回调缺乏子图scope，所以不匹配子图内的名字。
未知运行时Layout／dtype／packing、alias、状态、workspace、目标Kernel支持仍须验证。
因此`tensorDependenciesVerified=true`只指读取的声明张量边，**所有提示仍为`fusionReady=false`、`speedupProven=false`**。

2026-11-25，对手机实际0.6B图（1116节点）与已有受控Prefill＋2次Decode诊断关联，找到46组重复连接模式（两个阶段分开）。
例如：

| 结构 | 每次前向的静态位置数 | 单次Prefill的回调成本合计 |
| --- | ---: | ---: |
| up_proj、gate_proj → MUL_SILU（三节点汇合） | 28 | 592.87 ms |
| MUL_SILU → down_proj（两节点链） | 28 | 333.06 ms |

这些数字是原算子在诊断中的累计成本，**不是可节省的时间，也不是已融合速度**。MNN原先已经融合SiLU与Mul，本轮发现的是扩大融合边界的候选。
同一个primitive可能被多个模式覆盖，不能把两行成本相加当收益；组内也记录重叠和唯一覆盖成本。

对接master时，每个位置保留自己的`covers`；同模板各位置的共享实现对应一个`y_k`，不把28个位置当成28次开发。
若按静态位置建模，Prefill位置频率为1，Decode位置频率为目标Decode次数；**不再额外乘28**。
融合实现实测前，候选延迟未知，不能把回调成本直接当融合成本送入精确定价。独立开发／数值验收后，填入各位置或可信调用类的实际延迟，再解重叠覆盖master并做整图验收。
当前完成的是自动发现及计数，不是任意新结构的自动Kernel生成／部署；新的未知实现仍属于上述在线row／column问题。

### 有限开发预算下的动态子图定价

`Dual-guided Reusable Kernel Generation` 将已有实现的全局部署、新子图的发现、以及候选实现的开发分开。串行且固定音频长度时，最小化 RTF 等价于最大化累计延迟收益。存在流水重叠时，最终以完整生成的墙钟 RTF 验收。

`price-subgraphs`实现了在线定价的首版。先在已验收位置上求集合打包 LP，再沿实际 MNN 张量边扩展至 2–6 个节点。每层保留指定数量的结构模板，每个模板最多选两个代表位置继续扩展。非凸中间状态可以扩展，提交的融合区域必须凸。模板按算子语义、张量端口、输入别名、输出边界、阶段 Shape 和后端信息匹配其余出现位置。

对于位置 \(j\)，LP 使用 \(g_j=f_j(b_j-t_j)\) 最大化累计收益，每个原始位置至多被一次融合覆盖，未覆盖的算子运行 MNN 原生实现。节点约束的对偶 \(\pi_i\ge 0\)，搜索优先级采用 \(\widehat q_j=\widehat g_j-\sum_{i\in S_j}\pi_i\)。当前 \(\widehat g_j\) 用内部且可省物化的单输出张量逻辑字节估算，参考带宽由参数传入，并以原算子回调成本封顶。Prefill 和 Decode 独立使用已观察的次数。通用算子实现的开发费用仍在外层预算中按实现记账。

同一模板的位置有冲突时，对保留的模板解小规模打包问题。最终选择已测位置时也检查收缩后的 DAG。两个单独凸且互不重叠的区域仍可能相互依赖，如数据边 \(a\to b,c\to d,a\to d,c\to b\) 上同时收缩 \(\{a,b\},\{c,d\}\)。发现环后添加约束，排除同时选择该环上的全部融合区域。当前对偶来自添加环约束前的覆盖 LP 松弛。

配对实验结果可通过 `--verified` 加入下一轮，目录字段为 `scope=paired_device_measurements`、`modelSha256`、`profileSha256` 和 `placements`。每个位置包含 `id`、`phase`、`nodes`、`nativeMsPerCall`、`fusedMsPerCall`、`relativeL2`、`correctnessPassed`、`devicePassed`。采样输入、精度和设备条件应由已有候选验证链保证。CLI 校验目录标识、节点和误差，并记录提供的数据来源。候选评分与已测位置分开；冷启动对偶为零，`--policy frequency` 可关闭对偶作为同预算对照。

```powershell
python tools/operator_lab.py price-subgraphs --graph PRIVATE_GRAPH_JSON --profile PROFILE_JSON `
  --model-sha256 SOURCE_MNN_SHA256 --bandwidth-gbps 36.4042 --beam-width 32 --max-nodes 6 `
  --policy dual --output proposals-new.json
```

在 `--expansion bound` 模式下，用较大的潜力留住当前分数偏低的部分结构。节点权重是累计基线算子成本减对偶价格，每个位置取至多 \(L-|S|\) 个可达节点的最高正权重，再与区域现有权重相加。忽略非负的融合运行成本后，这给出同一加性成本口径下的单位置上界。模板用多位置的贪心聚合值决定 Beam 顺序。有限代表点与模板宽度仍会丢弃搜索分支，且不同扩展位置集合可以变化。当前输入的基线成本来自回调时间，逻辑张量得分来自经验带宽。两者均用于候选排序。对照模式 `--expansion current` 用当前子图的流量代理收益排序。

`--top-k` 按边际收益上界安排整数重求的顺序。每次将一个未开发模板的预测收益与现有已测位置组合，使用覆盖与无环约束算出 \(\widehat{\Delta Z}_p=Z_{IP}(K\cup\{p\})-Z_{IP}(K)\)。预测候选不进入已测主问题。开发次序使用 \(\widehat{\Delta Z}_p/\widehat d_p\)，每个实现的预计成本只计一次。`--development-plan` 接收 `costByPattern`、`defaultCost`、`remainingBudget`、`blockedPatternIds`。缺省成本为一次预计开发尝试。实际编译和设备次数由候选验证链记账。`--prior` 可读取同图同诊断的上轮私有 JSON，`--dual-alpha` 控制 Beam 用的对偶平滑系数。重求 MILP 使用预测的原始收益。

#### LP 定价与整数边际收益的区别

正约化收益说明当前的 LP 对偶解被新列违反。因最优对偶可能有多个，新列加入后 LP 值仍可能一样。负约化收益也不能排除整数部署收益。三个两两冲突、各收益 100 ms 的合法融合位置有 \(Z_{LP}=150\)、\(Z_{IP}=100\)。一个覆盖三者、收益 140 ms 的新位置约化收益是 −10 ms，整数收益却增加 40 ms。旧版若先按约化收益截取 Top-1，可能把它让给一个仅能多省 10 ms 的独立位置。

设 \(D=\sum_i\pi_i\) 为现有覆盖 LP 的可行对偶值，候选收益 \(g_j\) 使用同一可加成本口径。代码改用上界

\[
U_p = \min\left(
\sum_j (g_j)_+,
D-Z_{IP}(K)+\sum_j\left(g_j-\sum_{i\in S_j}\pi_i\right)_+
\right).
\]

第一项成立是因为任一联合可行方案中的旧位置仍是可行的旧部署。第二项来自覆盖约束，每个节点的对偶价值最多计一次，再加上新位置的正约化收益。实现先将数值对偶修成满足旧位置约束的可行对偶。按 \(U_p/\widehat d_p\) 选择要重求的候选，未求或未证最优的候选仍保留上界。只有它们的上界都不超过已求候选的最佳值，JSON 才给出 `bestInDiscoveredPoolCertified=true`。

这个判断限定在该阶段、Beam 已发现的候选、给定成本模型及一次开发选择。同一个模板内的位置全部无重叠且收缩后无环时，才可以直接相加正约化收益。单位置的扩展上界不能用来保证重复模板找全。按边际收益与成本之比逐次开发，也不能证明跨多次开发的全预算最优。整曲 RTF 和峰值内存仍以整条实际链路的测量为准。

为 `price-subgraphs` 传入 `--development-weight` 时，重求后按 \(\widehat{\Delta Z}_p-(\lambda/N)\widehat d_p\) 选择下一次开发。筛选的上界也扣除同一项。保留当前方案的增量价值为零。若还可能有更好的未求候选，报告中的证优字段单独反映。省略权重会使用既有的每单位开发成本收益排序。现有已验收的部署位置仍以零新增开发成本参与，这个排序针对当前阶段的工作量，每次只比较一个新候选。

0.6B 真图宽度 32 的本地试跑中，当前收益扩展为 Prefill、Decode 各 58 个模板，按扩展潜力排序分别为 52、54 个。两条路径均找到了 28 处四算子 FFN 模式。36.4042 GB/s 来自此前复制测试，仍作为代理排序输入。此轮缺少可纳入主问题的逐位置配对耗时，真实图仍从零对偶启动。需比较相同开发预算下的整数边际收益和完整模型实测，才能判断哪种扩展策略更有效。

## 链式精确对照与批量开发

指令集扩展中的凸子图枚举和全局合法性检查，直接对应本项目的融合位置问题。串行固定顺序则允许把每个连续覆盖区间表示为前缀位置 DAG 上的一条边。带颜色最短路径文献提供了追踪已用标签的状态设计。本项目追踪的是已支付开发费的实现，且费用由硬预算和目标权重共同约束。[指令集扩展原文](https://www.epfl.ch/labs/lap/wp-content/uploads/2018/05/AtasuDec03_AutomaticApplicationSpecificInstructionSetExtensionsUnderMicroarchitecturalConstraints_IJPP.pdf) · [带颜色最短路径原文](https://link.springer.com/article/10.1007/s11590-020-01659-z)

- `pattern-plan --mode chain-dp` 精确求解给定链式目录。状态为执行前缀与已付费实现集合，同一实现后续再用时延照计、开发费归零。24 个固定种子合成实例与 CPLEX 目标逐一一致。实现集合数最坏可指数增长，该模式供小实例对照。
- `pattern-plan --mode chain-marginal --max-batch 2` 比较同一预算下的逐组重求和最短路径上界筛选。沿任一新路径将边长重标为 \(t_e+D(u)-D(v)\)，首尾势差抵消为原路径基线。已有边的标记非负，所以累计改善至多是所有新边的负标记绝对值之和。已知边长下可安全剔除这个上界低于当前最佳净值的候选。某个前缀不可达时禁用筛选。经验延迟作为输入时，结论只适用于给定的代理模型。
- `price-subgraphs --expansion exhaustive` 枚举当前允许算子和阶段范围内的所有 2–6 节点结构，按依赖、凸性和同一调用次数过滤，再去重。与现有 Beam 使用相同的候选评分。`--batch-shortlist 8` 额外比较短名单内的候选对，候选即使单独无法改善也能入围。两个实现的新增成本一起扣除。重求部署时包含重叠约束与收缩环检查。

已开发 BC 时，AB 或 CD 单独加入都无法缩短目前的 10 ms。两者同时开发可到 8 ms，按每次开发折合 0.5 ms 计，净收益为 1 ms。单步贪心停在原方案，而链式 DP、CPLEX 和双候选重求一致。在加入 12 个刻意较弱的候选后，筛选将重求次数从 105 减到 14，方案和净值相同。该下降率是受控算例结果。

在 0.6B 的已有诊断上，每个阶段遍历 36,071 个连通区域，1,876 个位置通过结构筛选并聚成 72 个模板。宽度 32 的 Beam 分别发现 52、54 个模板。在相同的流量代理、时延等价开发成本 0.01 ms／实现下，本次两种搜索得到相同的最佳单候选净值，8 个候选内的最佳组合也没有超过单候选。结构发现的差异与真实部署性能是两套证据。

随机算例设 AB、CD 独立以 65% 概率成功，失败照付 0.5 ms 等价成本。事先承诺两次的期望净收益是 −0.155 ms，先试一个且成功后才继续为 +0.020 ms。数值仅用于说明可观察失败如何改变选择。要在本项目拟合概率，需记录编译、正确性、实机性能和资源消耗的各阶段结果。MoKA 表 5 区分编译通过、`fast@1.0` 和 `fast@1.5`，这些统计依赖其基准与阈值。昂贵信息文献的近似界依赖可行集合等假设，而融合覆盖冲突一般不满足拟阵交换性质。[MoKA 原文](https://openreview.net/pdf?id=bTsMfGz85R) · [昂贵信息组合选择原文](https://arxiv.org/abs/2412.03860)

```powershell
python tools/operator_lab.py pattern-plan --catalog configs/operator_lab/batch_development_example.json `
  --allow-synthetic --budget 2 --development-weight 0.5 --mode chain-marginal --max-batch 2 --output batch.json
```

## 5. Frequency Greedy与实现质量

实现`frequency_greedy`是次数加权的**边际收益Greedy＋当前已开发库的全局重新部署**，比简单按频次排序更强。
每次只为一个新实现付款；已付开发成本不可退款。它使用的CPLEX时间也必须纳入搜索时间。
测试中的重叠例子：Greedy先选B，成本320；全局A+C为290。这是明确标记的合成算例，不是手机加速数据。
还测试了：开发一次复用3处、失败支出、已开发成本归零、LP／整数gap、兼容性拒绝、预测数据不能冒充验收，以及Phase-I恢复。

候选质量验收至少分为：数值／极值／尾部／alias适配，真实向量化与调用开销，同运行上下文下的缓存和packing，实际调度与完整模型收益。
本轮tile实现只是现有两个优化函数的分块串联，中间读写仍存在。不能将局部几毫秒的收益忽略占比后宣称RTF明显下降。

## 6. 本轮实际例子

[MUL_SILU真机实验](LLM_MULSILU_PILOT.md)中，一次513位置Prefill有28处门控，64次Decode有1792次。
次数加权的回调代理总耗时约33.03ms，只占原固定工作量约1%。
单次prefill成本表选tile256；按阶段加权并排除未改变Decode路径的虚假收益后，估计模型选tile4096。
额外整图开销未知；实机完整LLM没有稳定提速，所以最终保留stock。
这正是“局部cost最小→频率感知规划→整图验收”三个不同结论，不能混成一个。
已编译的5个版本再使用时新增开发费为0；回顾性budget1只用于说明规划接口，不伪装成只花一次开发就找到结果。

## 7. 命令

```powershell
python tools/operator_lab.py mine-repetitions --profile PROFILE_JSON --output hints-new.json
python tools/operator_lab.py mine-subgraphs --graph PRIVATE_GRAPH_JSON --profile PROFILE_JSON `
  --model-sha256 SOURCE_MNN_SHA256 --output connected-hints-new.json
python tools/operator_lab.py pattern-plan --catalog configs/operator_lab/frequency_example.json `
  --budget 2 --mode mip --allow-synthetic --allow-predictions --output plan-new.json
python tools/operator_lab.py pattern-plan --catalog configs/operator_lab/frequency_example.json `
  --budget 2 --mode cg --allow-synthetic --allow-predictions --output cg-new.json
python tools/operator_lab.py pattern-plan --catalog configs/operator_lab/frequency_example.json `
  --budget 2 --mode frequency-greedy --allow-synthetic --allow-predictions --output greedy-new.json
```

示例明确是合成预测；真实部署必须用已测成本、误差门槛、实际开发账本和完整链路验收。
原CG若已经展开全部调用，本来就隐含运行次数；不要未经能力审计就写“CG／MoKA不考虑频率”。本研究应控制同候选库／工具／预算，再隔离频率、复用、定价与全局组合的贡献。
串行假设下累计延迟可作为目标；有异步流水线时还要建模关键路径和资源竞争，不能把所有模块耗时直接相加当作墙钟RTF。

图导出可用新目标`llm_operator_graph_export INPUT.mnn OUTPUT.private.json`。它只验证和读取FlatBuffer，不创建MNN会话，不推理。
保持导出器／原图／导出JSON的SHA与私有来源记录；CLI核对声明的原图SHA与诊断manifest，但不提供远端可信执行证明。
121项Python测试通过，含XY／XYZ、汇合、端口顺序、共享输入、常量区别、非凸路径、外部fan-out、重复覆盖、控制流及证据门槛。
