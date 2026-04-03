# 技术演进路线七：基础模型 / Foundation Model

> **核心问题**: 能否构建通用时序基础模型, 一次训练覆盖多种任务?
>
> **起点**: TS2Vec (2022) | 引用数: 608
>
> **涉及论文**: TS2Vec, TimesNet, Pangu-Weather, Lag-Llama, Chronos, Moirai, Timer, Time-LLM, PromptCast 等
>
> **交叉路线**: TimesNet → [路线2: 序列分解]; Pangu-Weather → [路线1: 3D Transformer]; Time-LLM → [路线4: Patching]

---

## 1. 问题背景

传统范式: **为每个任务训练专用模型**（训练一个预测模型、一个分类模型、一个异常检测模型...）。

新范式: **预训练通用基础模型 → 下游微调/零样本推理**，类似 NLP 中的 BERT/GPT。

核心挑战:
1. 时间序列是连续实数, 与 NLP 的离散 token 不同
2. 不同任务的输出格式差异巨大
3. 缺乏大规模预训练数据

---

## 2. TS2Vec: 通用时序表征 (2022)

**论文**: *TS2Vec: Towards Universal Representation of Time Series*
**会议**: AAAI 2022 | **引用数**: 608
**涉及路线**: [路线7: FM]

### 2.1 对比学习框架

**实例级对比损失**:
$$\mathcal{L}_{\text{inst}} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j \neq i} \exp(\text{sim}(z_i, z_j) / \tau) + \sum_{j \neq i} \exp(\text{sim}(z_i, z_j^-) / \tau)}$$

- $z_i$: 时间序列 $i$ 的表征
- $z_i^+$: 同一时间序列在不同增强下的表征（正样本）
- $z_j^-$: 不同时间序列的表征（负样本）

**时间级对比损失**:
$$\mathcal{L}_{\text{temp}} = -\log \frac{\exp(\text{sim}(z_{i,t}, z_{i,t}^+) / \tau)}{\sum_{t'} \exp(\text{sim}(z_{i,t}, z_{i,t'}) / \tau)}$$

**总损失**:
$$\mathcal{L} = \mathcal{L}_{\text{inst}} + \mathcal{L}_{\text{temp}}$$

### 2.2 时间增强
通过随机裁剪 (random cropping) 创建同一序列的不同视图:
$$\tilde{X}_i = X_i[t_a:t_b], \quad t_a, t_b \sim \text{Uniform}(0, T)$$

### 2.3 层级聚合
通过最大池化实现任意子序列表征:
$$r_{\text{sub}} = \text{MaxPool}(r_{\text{timestamp}})$$

---

## 3. TimesNet: 统一多任务框架 (2022)

**论文**: *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*
**会议**: ICLR 2023 | **引用数**: 420
**涉及路线**: [路线7: FM] [路线2: 序列分解]

### 3.1 核心思路
单一骨干网络处理多种任务, 只需更换最后的预测头。

### 3.2 统一架构
$$\hat{Y}_{\text{task}} = \text{Head}_{\text{task}}(\text{TimesBlock}(X))$$

- **预测**: $\text{Head}_{\text{forecast}}(\cdot) = \text{Linear}(\cdot)$ → 预测序列
- **分类**: $\text{Head}_{\text{classify}}(\cdot) = \text{Softmax}(\text{Linear}(\cdot))$ → 类别概率
- **异常检测**: $\text{Head}_{\text{anomaly}}(\cdot) = \|\text{Reconstruct}(X) - X\|$ → 重构误差
- **插补**: $\text{Head}_{\text{imputation}}(\cdot) = \text{Linear}(\cdot)$ → 填充值

---

## 4. Pangu-Weather: 3D 天气 FM (2023)

**论文**: *Accurate medium-range global weather forecasting with 3D neural networks*
**期刊**: Nature 2023 | **引用数**: 1264
**涉及路线**: [路线7: FM]

### 4.1 3D Swin Transformer
处理经度 × 纬度 × 气压层的三维数据:
$$\text{3D-SwinAttn}(Q, K, V) = \text{WindowAttn}_{\text{3D}}(Q, K, V)$$
$$Q, K, V \in \mathbb{R}^{H \times W \times Z \times C}$$
其中 $H$ = 纬度格点, $W$ = 经度格点, $Z$ = 气压层, $C$ = 通道。

### 4.2 Earth-Specific 偏置
- **球面坐标**: 使用等经纬度网格而非平面网格
- **层次化时间聚合**: 不同时间步的模型专门化
  $$\hat{Y}_{1h} = f_{1h}(X), \quad \hat{Y}_{24h} = f_{24h}(X), \quad \hat{Y}_{6h} = f_{6h}(X)$$

### 4.3 预训练
在 ERA5 再分析数据 (1979-2021, 约 40 年) 上预训练:
$$\mathcal{L} = \frac{1}{N} \sum_i \| \hat{Y}_i - Y_i \|^2$$

### 4.4 里程碑意义
**首次 AI 模型在确定性天气预报上超越数值天气预报 (NWP) 方法**。推理速度比传统 NWP 快 10000+ 倍。

---

## 5. Lag-Llama: LLM 架构时序预测 (2023)
**论文**: *Lag-Llama: Towards Foundation Models for Time Series Forecasting*
**涉及路线**: [路线7: FM]

### 5.1 Lag 特征 + LLaMA 架构
将时间序列转换为滞后特征序列:
$$\mathbf{x}_t = [x_{t-1}, x_{t-2}, \ldots, x_{t-k}]$$
使用 LLaMA 架构（RMSNorm, RoPE, SwiGLU）处理。

### 5.2 概率预测
使用 Student's t 分布作为输出分布:
$$p(y_t | x_{\leq t}) = \text{StudentT}(y_t; \mu_\theta(x_{\leq t}), \sigma_\theta(x_{\leq t}), \nu_\theta(x_{\leq t}))$$
$$\mathcal{L} = -\sum_t \log p(y_t | x_{\leq t})$$

---

## 6. Chronos: 时序 Tokenization (2024)

**论文**: *Chronos: Learning the Language of Time Series*
**机构**: Amazon | **涉及路线**: [路线7: FM]

### 6.1 核心思想
将连续时间序列值 **量化为离散 token**, 然后用语言模型方法训练。

### 6.2 量化 (Tokenization)
**第一步**: 按均值缩放
$$\mathbf{x}' = \frac{\mathbf{x}}{\frac{1}{T}\sum_{t=1}^{T} |x_t| + \epsilon}$$

**第二步**: 映射到最近的量化中心
$$\text{token}(x_t) = \arg\min_{k \in \{1,\ldots,V\}} |x_t - c_k|$$

量化中心 $\{c_1, \ldots, c_V\}$ 通过 k-means 在大规模数据上学习得到。

### 6.3 语言模型训练
将 tokenized 序列当作"文本", 训练标准语言模型:
$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(\text{token}(x_t) | \text{token}(x_1), \ldots, \text{token}(x_{t-1}))$$

使用 T5 架构（Encoder-Decoder Transformer）。

### 6.4 预测
自回归生成 token, 然后反量化为数值:
$$\hat{y}_t = c_{\text{argmax} P_\theta(\cdot | \text{context})}$$

### 6.5 零样本能力
在未见过的数据集上直接预测, 无需微调——这是 Foundation Model 的核心价值。

---

## 7. Moirai: Salesforce 时序 FM (2024)

**论文**: *Unified Training of Universal Time Series Forecasting Transformers*
**机构**: Salesforce | **涉及路线**: [路线7: FM]

### 7.1 Any-Variate Attention
处理任意数量变量的注意力机制:
$$\mathbf{H} = [\text{Embed}(X_1); \text{Embed}(X_2); \ldots; \text{Embed}(X_D)]$$
变量数量 $D$ 不固定, 动态调整。

### 7.2 Multi-Patch Size Projection
不同变量可能需要不同 patch 大小:
$$\mathbf{Z}_d = \text{Patch}_{p_d}(\text{Embed}(X_d))$$

### 7.3 预训练数据: LOTSA
LOTSA 数据集包含 27B 观测值, 覆盖 9 个领域:
- 交通、能源、天气、金融、销售、服务器、物联网、自然、Web

### 7.4 概率预测
输出参数化分布的参数:
$$\hat{Y} \sim \text{ParametricDist}(\mu_\theta(X), \sigma_\theta(X), \ldots)$$

---

## 8. Timer: 生成式时序预训练 (2024)

**论文**: *Timer: Generative Pre-trained Transformers for Time Series Analysis*
**涉及路线**: [路线7: FM]

### 8.1 Next-Token 预测
将时间序列分段后做自回归预训练:
$$\mathcal{L} = -\sum_{t} \log P_\theta(x_t | x_{<t})$$

### 8.2 统一任务接口
所有任务统一为序列生成:
- **预测**: 生成未来序列
- **插补**: 生成缺失部分
- **异常检测**: 生成正常模式, 对比差异

---

## 9. Time-LLM: 重编程 LLM (2023)

**论文**: *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models*
**引用数**: 127
**涉及路线**: [路线7: FM] [路线4: Patching]

### 9.1 时序 → 文本原型对齐
$$\mathbf{Z} = \text{Patch}(X) \cdot W_{\text{reprogram}}$$
其中 $W_{\text{reprogram}}$ 将时序 patch 嵌入对齐到 LLM 的词嵌入空间。

### 9.2 文本原型
使用一组可学习的文本原型:
$$\mathbf{P} = [\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_K] \in \mathbb{R}^{K \times d}$$
$$\mathbf{Z}_{\text{aligned}} = \text{softmax}(\mathbf{Z} \mathbf{P}^\top / \tau) \cdot \mathbf{P}$$

### 9.3 Prompt-as-Prefix
在输入前添加文本提示:
$$\text{Input} = [\text{TaskPrompt}; \text{DatasetInfo}; \mathbf{Z}_{\text{aligned}}]$$

### 9.4 冻结 LLM
LLM 参数冻结, 只训练重编程层:
$$\hat{Y} = \text{Linear}(\text{LLM}_{\text{frozen}}(\text{Input}))$$

---

## 10. PromptCast: 提示预测 (2023)

**论文**: *PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting*
**引用数**: 210
**涉及路线**: [路线7: FM]

### 10.1 时序 → 文本转换
将时间序列离散化为文本:
$$\text{Prompt}_t = \text{``The value at time } t \text{ is ''} + \text{Round}(x_t, 2)$$

### 10.2 文本到文本预测
$$\hat{Y}_{\text{text}} = \text{LLM}(\text{``Predict the next } T \text{ values: ''} + \text{Prompt}_{1:L})$$

---

## 11. 第二代 FM (2025)

### 11.1 Chronos-2
更大的模型, 更多预训练数据, 改进的量化策略。

### 11.2 Moirai-2
更强的零样本和少样本性能, 更多样化的预训练数据。

---

## 12. 演进对比总结

| 模型 | 年份 | 核心方法 | 预训练数据 | 零样本 | 概率输出 | 引用 |
|------|------|---------|-----------|--------|---------|------|
| TS2Vec | 2022 | 对比学习 | 有监督 | 否 | 否 | 608 |
| TimesNet | 2022 | 2D 多周期 | 无预训练 | 否 | 否 | 420 |
| **Pangu-Weather** | **2023** | **3D Swin** | **ERA5 40年** | **是** | **否** | **1264** |
| Lag-Llama | 2023 | LLaMA + t分布 | 有 | 是 | 是 | — |
| **Chronos** | **2024** | **量化 + LM** | **大规模** | **是** | **是** | — |
| Moirai | 2024 | Any-Variate Attn | LOTSA 27B | 是 | 是 | — |
| Timer | 2024 | Next-Token | 大规模 | 是 | 否 | — |
| **Time-LLM** | **2023** | **重编程 LLM** | **冻结 LLM** | **是** | **否** | **127** |
| Chronos-2 | 2025 | 改进量化 | 更大 | 是 | 是 | — |

---

## 13. 演进逻辑图

```
专用模型 (每任务训练)
  │
  ├─→ TS2Vec (2022, 对比学习表征) ← 通用表征起点
  │     └─→ 时间级 + 实例级对比学习
  │
  ├─→ TimesNet (2022, 统一多任务骨干)
  │     └─→ 同一模型, 不同预测头
  │
  ├─→ Pangu-Weather (2023, 3D FM)
  │     └─→ 首次超越 NWP
  │
  ├─→ LLM 路线
  │     ├─→ Lag-Llama (2023, LLaMA 架构)
  │     ├─→ Time-LLM (2023, 重编程 LLM) [→ 路线4]
  │     ├─→ PromptCast (2023, 文本提示)
  │     └─→ Timer (2024, 生成式预训练)
  │
  ├─→ 纯时序 FM 路线
  │     ├─→ Chronos (2024, 量化 + 语言模型)
  │     ├─→ Moirai (2024, Salesforce FM)
  │     └─→ Chronos-2 / Moirai-2 (2025, 第二代)
  │
  └─→ [专用模型仍在特定 benchmark 上更强]
```

**核心争论**: 专用模型 vs Foundation Model (见主文档"三大论战")

---

*本文档基于 2669 篇论文的 AI 深度分析.*
*数据来源: OpenAlex · Semantic Scholar · arXiv | AI 分析引擎: Claude*
