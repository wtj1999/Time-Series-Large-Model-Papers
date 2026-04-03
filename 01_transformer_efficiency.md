# 技术演进路线一：Transformer 效率优化

> **核心问题**：如何让 Transformer 在长序列时间序列预测中更高效？
>
> **起点**：Informer (2021) | 引用数：5565
>
> **涉及论文**：Informer, Autoformer, FEDformer, Pyraformer, Crossformer, iTransformer, Pathformer, LogTrans, Reformer 等
>
> **交叉路线**：Autoformer → [路线2: 序列分解]；FEDformer → [路线6: 频域方法]；Crossformer/iTransformer → [路线7: 通道建模]

---

## 1. 问题背景

标准 Transformer 的自注意力机制具有 $O(L^2)$ 的时间和空间复杂度，其中 $L$ 为序列长度。在长序列时间序列预测（Long-sequence Time-series Forecasting, LSTF）中，预测长度可达 96、192、336 甚至 720 步，这导致标准 Transformer 在计算和内存上完全不可行。

标准自注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q, K, V \in \mathbb{R}^{L \times d_k}$，计算复杂度为 $O(L^2 d_k)$。

**关键问题**：如何在不显著损失精度的前提下，将注意力复杂度从 $O(L^2)$ 降低？

---

## 2. Informer: ProbSparse 自注意力 (2021)

**论文**: *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*
**会议**: AAAI 2021 | **引用数**: 5565
**涉及路线**: [路线1: Transformer效率] [路线2: 序列分解(间接)]

### 2.1 核心洞察

在自注意力矩阵中，大部分注意力分数呈均匀分布（即"懒惰"查询），只有少数查询的注意力分布是尖锐的（即"活跃"查询）。通过筛选出活跃查询，可以大幅降低计算量。

### 2.2 ProbSparse 自注意力

**第一步：查询稀疏度度量（KL 散度近似）**

$$\bar{M}(q_i, K) = \max_j \frac{q_i k_j^\top}{\sqrt{d_k}} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^\top}{\sqrt{d_k}}$$

- 第一项 $\max_j$：查询 $q_i$ 与所有键的**最大**注意力分数
- 第二项 $\frac{1}{L_K}\sum_j$：查询 $q_i$ 与所有键的**平均**注意力分数
- 差值越大 $\Rightarrow$ $q_i$ 的注意力越集中（越"活跃"）

**实际采样近似**（避免 $O(L^2)$ 全计算）：

$$\bar{M}(q_i, K) = \max_j \left\{\frac{q_i k_j^\top}{\sqrt{d_k}}\right\}_{j \in \mathcal{S}} - \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} \frac{q_i k_j^\top}{\sqrt{d_k}}$$

其中 $\mathcal{S} \subset \{1,\ldots,L_K\}$ 为随机采样的 $c \cdot \ln L_Q$ 个键的索引。

**第二步：Top-$u$ 查询选择**

$$u = c \cdot \ln L_Q$$

只保留 $\bar{M}$ 最高的 $u$ 个查询，其余查询用均值向量 $\bar{q} = \frac{1}{L_Q}\sum_i q_i$ 替代。

**第三步：ProbSparse 注意力计算**

$$\text{ProbAttention}(Q, K, V) = \text{softmax}\left(\frac{\bar{Q}K^\top}{\sqrt{d_k}}\right)V$$

其中 $\bar{Q} \in \mathbb{R}^{u \times d_k}$ 为筛选后的稀疏查询矩阵，复杂度降至 $O(L \log L)$。

### 2.3 自注意力蒸馏（Distilling）

在编码器层间逐步提取主导注意力特征，减少序列长度：

$$\mathbf{X}_{j+1}^{t} = \text{MaxPool}\left(\text{ELU}(\text{Conv1d}([\mathbf{X}_j^{t}]_{\text{AB}}))\right)$$

每经过一层蒸馏，序列长度减半：$L_j \to \lfloor L_j / 2 \rfloor$，进一步降低后续层的计算量。

### 2.4 生成式解码器（Generative Decoder）

一步生成完整预测序列，而非自回归逐步生成：

$$\mathbf{X}_{\text{token}} = [\mathbf{X}_{\text{start}}, \mathbf{0}_{L_{\text{pred}} \times d}]$$

$$\hat{\mathbf{Y}} = \text{Linear}(\text{Decoder}(\mathbf{X}_{\text{token}}, \mathbf{X}_{\text{enc}}^5))$$

其中 $\mathbf{X}_{\text{start}}$ 为输入序列最后 $L_{\text{start}}$ 步的嵌入，$\mathbf{0}$ 为目标长度的零占位符。

### 2.5 整体复杂度

$$O(L \log L) \quad \text{vs 标准 Transformer 的} \quad O(L^2)$$

在 720 步预测场景下，Informer 的内存占用约为标准 Transformer 的 1/50。

---

## 3. Autoformer: 自相关机制 (2021)

**论文**: *Autoformer: Decomposition Transformers with Auto-Correlation for Long Sequence Forecasting*
**会议**: NeurIPS 2021 | **引用数**: 1311
**涉及路线**: [路线1: Transformer效率] [路线2: 序列分解]

### 3.1 核心洞察

时序预测应关注**序列的周期性和趋势**，而非逐点的注意力匹配。Autoformer 用自相关（Auto-Correlation）替代自注意力，直接建模序列的周期依赖；同时首创将分解内置于 Transformer。

### 3.2 自相关机制

**自相关函数**（基于 Wiener-Khinchin 定理通过 FFT 高效计算）：

$$R_{XX}(\tau) = \frac{1}{L} \sum_{t=1}^{L} X_t \cdot X_{t+\tau} = \text{IFFT}\left(|\text{FFT}(X)|^2\right)(\tau)$$

复杂度 $O(L \log L)$。

**延迟选择**：选取 top-$k$ 个最强自相关延迟：

$$\mathcal{T} = \arg\text{Top-}k_\tau R_{QK}(\tau)$$

**自相关注意力聚合**：

$$\text{AutoCorr}(Q, K, V) = \sum_{\tau \in \mathcal{T}} \text{softmax}\left(R_{QK}(\tau)\right) \cdot \text{Roll}(V, \tau)$$

其中 $\text{Roll}(V, \tau)$ 将 $V$ 沿时间轴循环移位 $\tau$ 步，实现跨周期信息聚合。

### 3.3 序列分解模块（详见 [路线2]）

$$\mathcal{F}_{\text{decomp}}(X) = \underbrace{\text{AvgPool}(X)}_{X_{\text{trend}}} + \underbrace{(X - \text{AvgPool}(X))}_{X_{\text{seasonal}}}$$

### 3.4 整体架构

每个 Autoformer 层同时处理趋势和季节：

$$X_{\text{trend}}^{(l)}, X_{\text{seasonal}}^{(l)} = \text{SeriesDecomp}\left(X^{(l-1)} + \text{AutoCorr}(X^{(l-1)}, X^{(l-1)}, X^{(l-1)})\right)$$

最终预测：

$$\hat{Y} = \sum_{l=1}^{L} X_{\text{trend}}^{(l)} + \sum_{l=1}^{L} X_{\text{seasonal}}^{(l)}$$

---

## 4. FEDformer: 频域注意力 (2022)

**论文**: *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*
**会议**: ICML 2022 | **引用数**: 535
**涉及路线**: [路线1: Transformer效率] [路线2: 序列分解] [路线6: 频域方法]

### 4.1 核心洞察

在频域上做注意力计算，利用傅里叶基或小波基作为投影矩阵，用随机采样的少量频率分量替代全连接投影，大幅减少参数量的同时保留频率信息。

### 4.2 傅里叶增强模块（FEB: Fourier Enhanced Block）

**第一步**：FFT 投影到频域

$$\mathbf{Z} = \text{FFT}(\mathbf{X}) \in \mathbb{R}^{L \times d}$$

**第二步**：选择 $M$ 个频率分量（$M \ll L$）

$$\hat{\mathbf{Z}} = \text{Select}_M(\mathbf{Z}) \in \mathbb{R}^{M \times d}$$

**第三步**：频域线性变换

$$\hat{\mathbf{Z}}' = \hat{\mathbf{Z}} \cdot \mathbf{W} + \mathbf{b}$$

**第四步**：IFFT 回到时域

$$\mathbf{X}' = \text{IFFT}(\text{Pad}(\hat{\mathbf{Z}}'))$$

**复杂度**：$O(M \cdot d)$，其中 $M$ 为选择的频率数，通常 $M = O(\log L)$。

### 4.3 小波增强模块（FEW: Wavelet Enhanced Block）

使用离散小波变换（DWT）替代 FFT，同时捕获频率和时域位置信息：

$$\mathbf{A}^{(j)}, \mathbf{D}^{(j)} = \text{DWT}(\mathbf{X})$$

其中 $\mathbf{A}^{(j)}$ 为第 $j$ 层近似系数（低频），$\mathbf{D}^{(j)}$ 为细节系数（高频）。

### 4.4 频域分解（详见 [路线2] 和 [路线6]）

$$X_{\text{trend}} = \text{IFFT}\left(\text{LowPass}_M(\text{FFT}(X))\right), \quad X_{\text{seasonal}} = X - X_{\text{trend}}$$

---

## 5. Pyraformer: 金字塔注意力 (2022)

**论文**: *Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting*
**会议**: ICLR 2022
**涉及路线**: [路线1: Transformer效率]

### 5.1 核心洞察

构建金字塔结构的多分辨率注意力：底层处理细粒度时间步，上层处理粗粒度聚合，通过层级连接实现 $O(L)$ 复杂度的全局信息传递。

### 5.2 金字塔注意力模块（PAM）

**层间压缩**：

$$\mathbf{C}^{(s)} = \text{Condense}\left(\mathbf{C}^{(s-1)}\right) = \frac{1}{C}\sum_{c=1}^{C} \mathbf{C}^{(s-1)}_{(i-1)C+c}$$

其中 $s$ 为层级，$C$ 为压缩率。

**跨层级注意力**（同一层内 + 相邻层间）：

$$\mathbf{A}^{(s)}_i = \sum_{j \in \mathcal{N}(i)} \text{softmax}\left(\frac{q_i k_j^\top}{\sqrt{d}}\right) v_j$$

$\mathcal{N}(i)$ 包含同一层和相邻层的节点，实现多尺度信息融合。

### 5.3 CSC 模块（Cascaded Squeeze-and-Excitation）

用于在不同分辨率之间传递信息：

$$\mathbf{C}^{(s-1)}_i = \text{Concat}\left[\sigma(\mathbf{W}_s \mathbf{C}^{(s)}_j) \odot \mathbf{C}^{(s-1)}_i\right]_{j \in \text{children}(i)}$$

### 5.4 复杂度

| 模型 | 时间复杂度 | 全局信息 |
|------|-----------|---------|
| Vanilla Transformer | $O(L^2)$ | ✓ |
| Informer | $O(L \log L)$ | ✓ |
| **Pyraformer** | **$O(L)$** | **✓** |

---

## 6. Crossformer: 跨维度注意力 (2023)

**论文**: *Crossformer: Transformer Effectively Crosses Time and Dimension for Time Series Forecasting*
**会议**: ICLR 2023
**涉及路线**: [路线1: Transformer效率] [路线4: Patching] [路线7: 通道建模]

### 6.1 核心洞察

多变量时间序列中，变量之间存在交互关系。Crossformer 显式地在变量维度（Dimension）和时间维度（Time）上分别计算注意力，引入 HDS（Hierarchical Dimension-Session）路由降低计算量。

### 6.2 DSEG 嵌入（Dimension-Segment Embedding）

将 $D$ 维时间序列分割为长度 $P$ 的 patch：

$$\mathbf{Z}_{d,p} = \text{Embed}([x_{d,(p-1)P+1}, \ldots, x_{d,pP}]) \in \mathbb{R}^{d_{\text{model}}}$$

得到 $D \times N$ 个 patch token（$N = \lceil L/P \rceil$）。

### 6.3 TSA: 时间-空间交替注意力

**时间注意力（Time Attention, TA）**——沿时间轴，同一变量内：

$$\mathbf{Z}'_{d,p} = \sum_{p'=1}^{N} \alpha_{p,p'} \mathbf{Z}_{d,p'}$$

**维度注意力（Dimension Attention, DA）**——沿变量轴，同一时间步内：

$$\mathbf{Z}''_{d,p} = \sum_{d'=1}^{D} \beta_{d,d'} \mathbf{Z}'_{d',p}$$

两者交替堆叠。

### 6.4 路由机制

为降低变量维度计算量，将 $D$ 个变量分为 $G$ 组，每组用路由 token 聚合：

$$\mathbf{R}_g = \frac{1}{|\mathcal{G}_g|} \sum_{d \in \mathcal{G}_g} \mathbf{Z}_{d,p}$$

---

## 7. iTransformer: 反转维度 (2023)

**论文**: *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*
**会议**: ICLR 2024 | **引用数**: 354
**涉及路线**: [路线1: Transformer效率] [路线7: 通道建模]

### 7.1 核心洞察

传统方法将**时间步**作为 token，在**变量**维度上做注意力。iTransformer **反转**这一设计：将每个**变量**的完整时间序列映射为一个 token，注意力在**变量**维度上计算。

### 7.2 变量 Token 嵌入

$$\mathbf{h}_i = \text{Embed}(X_{i,:}) \in \mathbb{R}^{d_{\text{model}}}, \quad X_{i,:} = [x_{i,1}, \ldots, x_{i,T}] \in \mathbb{R}^T$$

Token 数量 = 变量数量 $D$（而非时间步数 $T$），通常 $D \ll T$。

### 7.3 反转自注意力

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{H}\mathbf{W}_Q (\mathbf{H}\mathbf{W}_K)^\top}{\sqrt{d_k}}\right), \quad \mathbf{H} \in \mathbb{R}^{D \times d_{\text{model}}}$$

注意力矩阵 $A \in \mathbb{R}^{D \times D}$ 直接捕获**变量间依赖**。

### 7.4 FFN 处理时间模式

每个 FFN 层对每个变量的表征进行非线性变换，实质上是在**学习每个变量的时间模式**：

$$\mathbf{h}_i' = \text{FFN}(\mathbf{h}_i) = \max(0, \mathbf{h}_i W_1 + b_1) W_2 + b_2$$

### 7.5 输出投影

$$\hat{X}_{i,:} = \text{Linear}(\mathbf{h}_i^{(L)}) \in \mathbb{R}^{T_{\text{pred}}}$$

### 7.6 核心优势

- 多变量时 $D$ 远小于 $T$，注意力矩阵 $D \times D \ll T \times T$
- 天然支持变量缺失
- 兼容 RevIN 等归一化方法
- **实验证明**：相同的 Transformer 架构，反转后性能大幅提升

---

## 8. Pathformer: 自适应多尺度 (2024)

**论文**: *Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting*
**会议**: AAAI 2024 | **引用数**: 35
**涉及路线**: [路线1: Transformer效率] [路线4: Patching]

### 8.1 核心洞察

时间序列存在多尺度模式（短期波动、中期趋势、长期周期）。Pathformer 自适应地选择不同尺度的注意力路径。

### 8.2 多尺度 Patch

将序列按不同大小 $P = \{p_1, p_2, \ldots, p_S\}$ 分割：

$$\text{Patch}_s(X) = [X_{1:p_s}, X_{p_s+1:2p_s}, \ldots]$$

### 8.3 自适应路径选择

$$g_s = \text{Linear}(\text{GAP}(\text{Attn}_s(X)))$$

$$\alpha_s = \frac{\exp(g_s)}{\sum_{s'=1}^{S} \exp(g_{s'})}$$

$$\hat{Y} = \sum_{s=1}^{S} \alpha_s \cdot \text{Linear}(\text{Attn}_s(X))$$

---

## 9. SFDformer: 稀疏频域分解 (2025)

**论文**: *SFDformer: a frequency-based sparse decomposition transformer for time series forecasting*
**引用数**: 8
**涉及路线**: [路线1: Transformer效率] [路线6: 频域方法]

### 9.1 核心改进

在 FEDformer 基础上引入稀疏约束，用可学习的稀疏掩码替代固定频率选择：

$$\hat{\mathbf{Z}} = \text{SparseMask}(\mathbf{Z}) \odot \mathbf{Z}$$

$$\text{SparseMask}(\mathbf{Z}) = \sigma(\mathbf{W}_m \cdot \text{FFT}(X) + b_m)$$

稀疏化后的频域注意力进一步减少冗余频率分量。

---

## 10. 复杂度对比总结

| 模型 | 年份 | 注意力机制 | 时间复杂度 | 核心创新 |
|------|------|-----------|-----------|---------|
| Transformer | 2017 | 全注意力 | $O(L^2)$ | 基准 |
| LogTrans | 2019 | 对数稀疏 | $O(L \log L)$ | 稀疏掩码 |
| Reformer | 2020 | LSH | $O(L \log L)$ | 哈希分桶 |
| **Informer** | **2021** | **ProbSparse** | **$O(L \log L)$** | **KL散度查询筛选** |
| **Autoformer** | **2021** | **Auto-Correlation** | **$O(L \log L)$** | **FFT自相关** |
| Pyraformer | 2022 | 金字塔 | $O(L)$ | 多分辨率层级 |
| **FEDformer** | **2022** | **频域注意力** | **$O(M)$** | **傅里叶/小波基** |
| Crossformer | 2023 | 跨维度 | $O(TD + DT)$ | 时间×维度 |
| **iTransformer** | **2023** | **反转注意力** | **$O(D^2)$** | **变量为token** |
| Pathformer | 2024 | 自适应路径 | $O(\sum_s L^2/p_s)$ | 多尺度选择 |
| SFDformer | 2025 | 稀疏频域 | $O(M')$ | 可学习稀疏掩码 |

---

## 11. 演进逻辑图

```
Transformer (O(L²), 2017)
  │
  ├─→ LogTrans (对数稀疏掩码, 2019)
  │     └─→ Reformer (LSH哈希, 2020)
  │
  ├─→ Informer (ProbSparse, O(L log L), 2021) ← 长序列预测起点
  │     │
  │     ├─→ Autoformer (自相关 + 内置分解, 2021)
  │     │     ├─→ FEDformer (频域注意力, 2022)
  │     │     │     └─→ SFDformer (稀疏频域, 2025)
  │     │     └─→ [→ 路线2: 序列分解]
  │     │
  │     ├─→ Pyraformer (金字塔 O(L), 2022)
  │     │
  │     ├─→ DLinear (质疑路线 → 路线3: MLP革命)
  │     │
  │     └─→ iTransformer (反转维度, 2023)
  │           └─→ [→ 路线7: 通道建模]
  │
  ├─→ Crossformer (跨维度注意力, 2023)
  │     └─→ [→ 路线4: Patching]
  │
  └─→ PatchTST → Pathformer (自适应多尺度, 2024)
        └─→ [→ 路线4: Patching]
```

---

*本文档基于 2669 篇论文的 AI 深度分析，覆盖 Transformer 效率优化方向的关键进展。*
*数据来源：OpenAlex · Semantic Scholar · arXiv | AI 分析引擎：Claude*
