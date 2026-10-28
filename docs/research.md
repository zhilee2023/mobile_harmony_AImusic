# 公开权重能否变成0.6B？

## 结论

有方法研究，但未找到可直接下载、满足本项目“0.6B+单路Mixed+指定codec”的已训练LeVo检查点。W4表示每参数4bit，不会将4B参数变成0.6B。

1. **保留官方LM，剪掉双轨分支**：最能复用音乐权重，也存在Mixed 1RVQ codec。但仅删除Track-Specific LM并不会自动得到0.6B，必须精确数参数；质量可能下降。
2. **同架构结构化剪枝+恢复训练**：剪层、FFN通道、注意力头。保持剩余张量可继承，但删除大量结构后需继续训练/蒸馏。LeVo的Llama式1536宽Base与Qwen3的1024宽不能直接严格加载。
3. **Qwen3-0.6B学生+音乐教师蒸馏**：公共文本骨干可直接加载；音乐输入embedding、音乐输出头、风格向量新建。必须学习歌词→Mixed code映射。训练时优先更新适配器，随后LoRA或全量微调骨干。无数据无训练不能凭空获得音乐能力。

优先落地路线3的工程；路线1用于获取教师输出和基准。当前不复制/发布上游权重，不把随机音乐头的输出当可听音乐。

## 约0.6B的实际结构实验

未剪枝Qwen骨干加音乐适配器共629,640,192参数。新增可选实验：先加载28层公共Qwen权重，再移除第9、18层，重编号KV层索引，保留其余层原权重，得到约0.598B参数。选层目前是固定实验方案，不是校准后得到的最优方案；需要恢复训练，也不保证保留原文本能力，更不能直接获得音乐能力。默认结构仍保留28层，以便进行公平对照。

## 原始研究

- [LeVo](https://arxiv.org/html/2506.07520v2)：双LM、Mixed/双轨token、DiT+VAE，以及尾部特征提示的分块级联。
- [LeVo 2](https://arxiv.org/html/2606.30642v1)：全局Mixed规划与Track-Specific细化。大模型参数不能套在Base上。
- [Sheared LLaMA](https://arxiv.org/abs/2310.06694)：将7B结构化剪枝到1.3B/2.7B，再继续训练。是文本模型证据，不是LeVo音乐效果保证。
- [Minitron](https://arxiv.org/abs/2407.14679)：深度/宽度/注意力/MLP剪枝配合知识蒸馏与恢复训练。
- [Qwen3-0.6B配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)：本学生骨干的直接权重加载来源。
- [LeVo Base配置](https://huggingface.co/lglg666/SongGeneration-base/blob/main/config.yaml)：主LM dim1536、28层、FFN8960、子LM12层，以及Mixed 1RVQ设置。

## 权重及许可

Qwen3权重按其Apache-2.0许可使用。LeVo仓库的SongGeneration许可限定学术、研究、教育用途；当前不能把其代码/权重视作无条件商业发布许可。参考[上游LICENSE](https://github.com/levo-demo/LeVo/blob/main/LICENSE)。本仓库只发布原创工程与接口，模型目录被gitignore。

新架构即使蒸馏，也不能假设自动摆脱教师权重、数据及生成样本的使用条件。训练前记录各数据和组件的来源。
