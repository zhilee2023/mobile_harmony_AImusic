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
y变量的LP reduced cost（不计上界对偶）是`Σ_{p:k(p)=k} α_p - λ*D_k`。
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
