# 技术演进路线四：Patching 机制

> **核心问题**：如何让模型捕获时间序列的局部语义信息？
>
> **起点**：PatchTST (2022) | 引用数：534
>
> **涉及论文**：PatchTST, Pathformer, TimeMixer, Patcher, Crossformer, HDMixer, Patch ModernTCN, IPatch 等
>
> **交叉路线**：PatchTST → [路线1: Transformer效率]；Crossformer → [路线1] [路线7: 通道建模]；HDMixer → [路线3: MLP革命]

---

## 1. 问题背景

传统方法将时间序列的每个时间步（time step）作为一个 token 输入 Transformer。这有两个问题：

1. **注意力复杂度高**：序列长度 $L$ 直接决定 token 数量，注意力复杂度 $O(L^2)$
2. **局部语义丢失**：单个时间点不携带局部上下文信息，类似于在图像处理中逐像素处理

借鉴 Vision Transformer (ViT) 的成功经验——将图像分割为 patch 再处理——PatchTST 首次将 patching 思路引入时序预测。

---

## 2. PatchTST: 时序 Patching 开创者 (2022)

**论文**: *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*
**会议**: ICLR 2023 | **引用数**: 534
**涉及路线**: [路线4: Patching] [路线1: Transformer效率]

### 2.1 Patching 操作

将长度为 $L$ 的时间序列分割为长度 $P$、步长 $S$ 的 patch：

$$\mathbf{Z} = [\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_N]$$

其中每个 patch：

$$\mathbf{z}_i = W_p \cdot [x_{(i-1)S+1}, x_{(i-1)S+2}, \ldots, x_{(i-1)S+P}] + b_p$$

- $N = \lfloor (L - P) / S \rfloor + 1$ 为 patch 数量
- $W_p \in \mathbb{R}^{d_{\text{model}} \times P}$ 为 patch 嵌入矩阵

### 2.2 Token 数量对比

| 方式 | Token 数量 | 注意力复杂度 |
|------|-----------|-------------|
| 逐时间步 | $L$ | $O(L^2)$ |
| **Patching** ($P=16, S=8$) | **$N \approx L/8$** | **$O((L/8)^2)$** |

在 $L=336, P=16, S=8$ 时：$N = 41$ vs $L = 336$，注意力计算减少约 **67 倍**。

### 2.3 通道独立（Channel Independence, CI）

PatchTST 的另一个关键创新：每个变量独立处理。

$$\hat{Y}_d = \text{Transformer}(\mathbf{Z}_d), \quad d = 1, 2, \ldots, D$$

不进行任何跨变量信息交互。每个变量的 patch 序列独立通过共享权重的 Transformer。

### 2.4 位置编码

为 patch 序列添加可学习位置编码：

$$\mathbf{Z}' = \mathbf{Z} + \mathbf{PE}$$

$$\mathbf{PE} \in \mathbb{R}^{N \times d_{\text{model}}}$$

### 2.5 完整流程

$$X \xrightarrow{\text{Patch}(P,S)} \mathbf{Z} \xrightarrow{+\text{PE}} \mathbf{Z}' \xrightarrow{\text{TransformerEncoder}} \mathbf{H} \xrightarrow{\text{Linear}} \hat{Y}$$

### 2.6 自监督预训练（可选）

通过随机 mask patch 进行自监督学习（类似 MAE）：

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| \hat{\mathbf{z}}_i - \mathbf{z}_i \|^2$$

其中 $\mathcal{M}$ 为被 mask 的 patch 集合。

### 2.7 核心贡献

- **Patching 大幅降低注意力复杂度**：$O(L^2) \to O((L/P)^2)$
- **局部语义提取**：每个 patch 携带局部窗口内的上下文信息
- **通道独立更稳定**：避免变量间虚假相关
- **证明 Transformer 仍有效**：对 DLinear 的质疑做出了有力回应

---

## 3. Pathformer: 自适应多尺度 Patch (2024)

**论文**: *Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting*
**会议**: AAAI 2024 | **引用数**: 35
**涉及路线**: [路线4: Patching] [路线1: Transformer效率]

### 3.1 多尺度 Patch

同时使用多个 patch 大小 $P = \{p_1, p_2, \ldots, p_S\}$：

$$\mathbf{Z}^{(s)} = \text{Patch}_{p_s}(X), \quad N_s = \lfloor (L - p_s) / p_s \rfloor + 1$$

每个尺度产生不同数量的 patch token。

### 3.2 双重注意力

对每个尺度分别计算：
- **时间注意力**：patch 间的时间依赖
- **变量注意力**：patch 间的跨变量依赖

### 3.3 自适应路径选择

通过门控机制自适应选择尺度权重：

$$g_s = \text{Linear}(\text{GAP}(\text{Attn}_s(\mathbf{Z}^{(s)})))$$

$$\alpha_s = \frac{\exp(g_s)}{\sum_{s'=1}^{S} \exp(g_{s'})}$$

最终输出：

$$\hat{Y} = \sum_{s=1}^{S} \alpha_s \cdot \text{Linear}(\text{Attn}_s(\mathbf{Z}^{(s)}))$$

---

## 4. TimeMixer: 多尺度 Patch 混合 (2024)

**论文**: *TimeMixer: Decomposable Time-Series Forecasting with Multi-Resolution Mixing*
**涉及路线**: [路线4: Patching] [路线2: 序列分解]

### 4.1 多尺度下采样

通过平均池化在多个尺度上采样：

$$X^{(s)} = \text{AvgPool}_{2^s}(X), \quad s = 0, 1, \ldots, S-1$$

尺度 $s=0$ 为原始序列，$s=1$ 下采样 2 倍，以此类推。

### 4.2 过去分解混合（PDM）

从细粒度到粗粒度逐级混合：

$$\mathbf{h}^{(s)} = \text{Mix}^{(s)}(\mathbf{h}^{(s-1)}, \mathbf{h}^{(s)})$$

$$\mathbf{h}^{(s)} = \mathbf{h}^{(s)} + W_{\text{up}} \cdot \text{UpSample}(\mathbf{h}^{(s-1)})$$

### 4.3 未来预测混合（FMM）

从粗粒度到细粒度逐级细化：

$$\hat{Y}^{(s)} = \text{Predict}^{(s)}(\mathbf{h}^{(s)}) + W_{\text{down}} \cdot \text{DownSample}(\hat{Y}^{(s+1)})$$

### 4.4 最终聚合

$$\hat{Y} = \sum_{s=0}^{S-1} \hat{Y}^{(s)}$$

---

## 5. Patcher: 可学习分割 (2023)

**论文**: *Learning to Patch: Patch-based Representation Learning for Time Series*
**涉及路线**: [路线4: Patching]

### 5.1 可学习分割点

替代固定大小 patch，学习最优分割位置：

$$\mathbf{b} = \sigma(\text{Linear}(X)) \in (0, 1)^L$$

$$\text{split points} = \{t : \mathbf{b}_t > 0.5\}$$

当 $\mathbf{b}_t > 0.5$ 时在该位置切分，形成不等长的 patch。

### 5.2 优势

- 数据自适应：在不同数据上学习不同的分割策略
- 不等长 patch 更好地捕获变化点

---

## 6. Crossformer: 跨维度 Patch (2023)

**论文**: *Crossformer: Transformer Effectively Crosses Time and Dimension*
**会议**: ICLR 2023
**涉及路线**: [路线4: Patching] [路线1: Transformer效率] [路线7: 通道建模]

### 6.1 DSEG 嵌入

将 $D$ 维序列分割为 $P$ 长度的 patch，产生 $D \times N$ 个 2D patch token：

$$\mathbf{Z}_{d,n} = \text{Embed}([x_{d,(n-1)P+1}, \ldots, x_{d,nP}])$$

### 6.2 双重注意力

**时间注意力（同一变量内跨 patch）**：

$$\mathbf{Z}'_{d,n} = \sum_{n'=1}^{N} \alpha_{n,n'} \mathbf{Z}_{d,n'}$$

**维度注意力（同一时间步跨变量）**：

$$\mathbf{Z}''_{d,n} = \sum_{d'=1}^{D} \beta_{d,d'} \mathbf{Z}'_{d',n}$$

---

## 7. HDMixer: 层次依赖 Patch (2024)

**论文**: *HDMixer: Hierarchical Dependency Mixers for Time Series Forecasting*
**引用数**: 51
**涉及路线**: [路线4: Patching] [路线3: MLP革命]

### 7.1 长度可扩展 Patch (LEP)

$$\text{LEP}(X) = [X_{a_1:b_1}, X_{a_2:b_2}, \ldots, X_{a_K:b_K}]$$

patch 边界 $a_k, b_k$ 可重叠，不要求等长。

### 7.2 三层 MLP 混合

1. **Patch 内**：$\text{MLP}_{\text{intra}}(\mathbf{z}_k)$——捕获短期依赖
2. **时间维度**：$\text{MLP}_{\text{temporal}}([h_1, \ldots, h_N])$——捕获长期依赖
3. **跨变量**：$\text{MLP}_{\text{cross}}([h_{1,D}, \ldots, h_{N,D}])$——捕获变量交互

---

## 8. Patch ModernTCN (2024)

**论文**: *Patch + ModernTCN*
**引用数**: 12
**涉及路线**: [路线4: Patching] [路线3: MLP革命]

### 8.1 Patch + 大核卷积

先 patch，再在 patch 上应用大核卷积：

$$\mathbf{Z} = \text{Patch}(X)$$

$$\hat{Y} = \text{DWConv}_{k=\text{large}}(\mathbf{Z})$$

patch 降低了序列长度，大核卷积在 patch 级别建模长程依赖。

---

## 9. IPatch: 逆 Patch (2024)

**论文**: *IPatch: A Multi-Resolution Transformer Architecture for Time Series*
**涉及路线**: [路线4: Patching]

### 9.1 逆 Patching 分布

将 patch 级别的预测分布回每个时间步：

$$\hat{y}_t = \sum_{k: t \in \text{patch}_k} \frac{\hat{z}_k}{|\{t' : t' \in \text{patch}_k\}|}$$

---

## 10. 演进对比总结

| 模型 | 年份 | Patch 策略 | 建模方式 | 通道策略 | 引用 |
|------|------|-----------|---------|---------|------|
| **PatchTST** | **2022** | **固定大小** | **Transformer** | **独立 (CI)** | **534** |
| Crossformer | 2023 | 固定 + 2D | 双重注意力 | 混合 (CD) | ~200 |
| Patcher | 2023 | 可学习 | Transformer | 独立 | — |
| **Pathformer** | **2024** | **多尺度** | **自适应注意力** | **混合** | **35** |
| TimeMixer | 2024 | 多尺度下采样 | MLP 混合 | 独立 | — |
| HDMixer | 2024 | 可扩展 | MLP 三层 | 混合 | 51 |
| Patch ModernTCN | 2024 | 固定 | 大核卷积 | 混合 | 12 |
| IPatch | 2024 | 固定 + 逆 | Transformer | 独立 | — |

---

## 11. 演进逻辑图

```
ViT Patching (2020, 计算机视觉)
  │
  └─→ PatchTST (2022, 固定 Patch + 通道独立)
        │
        ├─→ Crossformer (2023, 2D Patch + 跨维度注意力)
        │     └─→ [→ 路线7: 通道建模]
        │
        ├─→ Patcher (2023, 可学习分割)
        │
        ├─→ Pathformer (2024, 多尺度 Patch + 自适应路径)
        │     └─→ [→ 路线1: Transformer效率]
        │
        ├─→ TimeMixer (2024, 多尺度下采样 + MLP混合)
        │     └─→ [→ 路线2: 序列分解]
        │
        ├─→ HDMixer (2024, 可扩展Patch + MLP三层混合)
        │     └─→ [→ 路线3: MLP革命]
        │
        ├─→ Patch ModernTCN (2024, Patch + 大核卷积)
        │     └─→ [→ 路线3: MLP革命]
        │
        └─→ IPatch (2024, 逆Patching分布)
```

---

*本文档基于 2669 篇论文的 AI 深度分析。*
*数据来源：OpenAlex · Semantic Scholar · arXiv | AI 分析引擎：Claude*
