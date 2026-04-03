# 技术演进路线二：序列分解方法

> **核心问题**：如何将时间序列分解为趋势、季节等成分，并在模型中有效利用？
>
> **起点**：Autoformer (2021) 首创将分解内置于 Transformer 架构
>
> **涉及论文**：Autoformer, FEDformer, DLinear, TimesNet, ModernTCN, WaveForM, DeMa, DecMamba 等
>
> **交叉路线**：Autoformer → [路线1: Transformer效率]；FEDformer → [路线6: 频域]；DLinear → [路线3: MLP革命]；TimesNet → [路线7: FM]；DeMa/DecMamba → [路线8: Mamba]

---

## 1. 问题背景

时间序列通常可以分解为多个成分的叠加：

$$X_t = T_t + S_t + R_t$$

其中 $T_t$ 为趋势（Trend）、$S_t$ 为季节（Seasonal）、$R_t$ 为残差（Residual）。

传统方法（如 STL、X-13ARIMA-SEATS）将分解作为**预处理步骤**，分解后分别建模。Autoformer 的突破在于：将分解**内置于模型架构**，在每一层都进行自适应分解，使模型能够学习最优分解方式。

---

## 2. Autoformer: 内置序列分解 (2021)

**论文**: *Autoformer: Decomposition Transformers with Auto-Correlation for Long Sequence Forecasting*
**会议**: NeurIPS 2021 | **引用数**: 1311
**涉及路线**: [路线1: Transformer效率] [路线2: 序列分解]

### 2.1 序列分解模块（Series Decomposition Block）

Autoformer 定义了一个可学习的分解算子：

$$\mathcal{F}_{\text{decomp}}(X) = \left(T(X), S(X)\right)$$

其中趋势通过移动平均提取：

$$T(X) = \text{AvgPool}\left(\text{Padding}(X)\right)$$

$$S(X) = X - T(X)$$

这里 AvgPool 使用核大小为 $k$ 的滑动平均（$k$ 为设定的周期长度），Padding 保持序列长度不变。

### 2.2 渐进式分解架构

每个 Transformer 层都包含分解操作，将上一层的输出分解为趋势和季节分别处理：

**编码器**：

$$X_{\text{trend}}^{(l)}, X_{\text{seasonal}}^{(l)} = \mathcal{F}_{\text{decomp}}\left(X^{(l-1)} + \text{AutoCorr}(X^{(l-1)})\right)$$

**解码器**：同时接收编码器的趋势和季节成分：

$$S^{(l)}_{\text{de},1}, T^{(l)}_{\text{de},1} = \mathcal{F}_{\text{decomp}}\left(S^{(l-1)}_{\text{de}} + \text{AutoCorr}(S^{(l-1)}_{\text{de}}, S^{(l-1)}_{\text{en}})\right)$$

$$S^{(l)}_{\text{de},2}, T^{(l)}_{\text{de},2} = \mathcal{F}_{\text{decomp}}\left(S^{(l)}_{\text{de},1} + \text{AutoCorr}(S^{(l)}_{\text{de},1})\right)$$

### 2.3 最终预测

将所有层的趋势和季节分别求和：

$$\hat{Y} = \sum_{l=1}^{L} T^{(l)}_{\text{de}} + \sum_{l=1}^{L} S^{(l)}_{\text{de}}$$

**核心创新**：分解不是固定的预处理，而是模型学习过程中的动态操作——每一层都重新分解，逐步精炼趋势和季节的表征。

---

## 3. FEDformer: 频域分解 (2022)

**论文**: *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*
**会议**: ICML 2022 | **引用数**: 535
**涉及路线**: [路线1: Transformer效率] [路线2: 序列分解] [路线6: 频域方法]

### 3.1 频域分解模块

不同于 Autoformer 的时域移动平均，FEDformer 在**频域**中实现分解：

$$\mathcal{F}_{\text{FFT}} = \text{FFT}(X) \in \mathbb{C}^{L}$$

**趋势提取**：保留低频分量

$$X_{\text{trend}} = \text{IFFT}\left(\sum_{k=1}^{M} \mathcal{F}_{\text{FFT}}[k] \cdot e^{i2\pi kt/L}\right)$$

其中 $M$ 为低频截止阈值。

**季节提取**：保留高频分量

$$X_{\text{seasonal}} = X - X_{\text{trend}} = \text{IFFT}\left(\sum_{k=M+1}^{L/2} \mathcal{F}_{\text{FFT}}[k] \cdot e^{i2\pi kt/L}\right)$$

### 3.2 混合专家分解

FEDformer 使用傅里叶基和小波基的混合：

$$\text{FEB}(X) = \text{IFFT}(\text{Select}_M(\text{FFT}(X)) \cdot W)$$

$$\text{FEW}(X) = \text{IDWT}(\text{Select}_{M'}(\text{DWT}(X)) \cdot W')$$

通过随机选择 FEB 或 FEW 实现频率分量的多样化建模。

---

## 4. DLinear: 极简分解+线性 (2022/2023)

**论文**: *Are Transformers Effective for Time Series Forecasting?*
**引用数**: 2314
**涉及路线**: [路线2: 序列分解] [路线3: MLP革命]

### 4.1 核心思想

DLinear 证明：仅使用 Autoformer 首创的分解方法 + 最简单的线性映射，就能超越所有复杂的 Transformer 变体。

### 4.2 模型结构

**第一步：分解**（与 Autoformer 相同的移动平均）

$$X_{\text{trend}} = \text{AvgPool}(X), \quad X_{\text{seasonal}} = X - X_{\text{trend}}$$

**第二步：独立线性预测**

$$\hat{Y}_{\text{trend}} = W_t \cdot X_{\text{trend}} + b_t$$

$$\hat{Y}_{\text{seasonal}} = W_s \cdot X_{\text{seasonal}} + b_s$$

其中 $W_t \in \mathbb{R}^{T_{\text{pred}} \times T_{\text{input}}}$ 和 $W_s \in \mathbb{R}^{T_{\text{pred}} \times T_{\text{input}}}$ 都是**单层线性映射**。

**第三步：合并**

$$\hat{Y} = \hat{Y}_{\text{trend}} + \hat{Y}_{\text{seasonal}}$$

### 4.3 NLinear 变体

NLinear 额外处理分布漂移：

$$\hat{Y} = W \cdot (X - x_{\text{last}}) + x_{\text{last}}$$

先减去最后一个值（去除均值偏移），线性映射后加回。

### 4.4 影响

DLinear 引用量高达 2314，是最具影响力的"质疑论文"。它引发的讨论改变了整个领域的研究方向——从盲目追求复杂架构，转向重新审视基本设计。

---

## 5. TimesNet: 多周期 2D 建模 (2022)

**论文**: *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*
**会议**: ICLR 2023 | **引用数**: 420
**涉及路线**: [路线2: 序列分解] [路线7: FM]

### 5.1 核心洞察

时间序列的变化可以按周期分解为多个成分，每个周期内的变化模式可以用 2D 张量表示（周期内 × 周期间）。

### 5.2 周期发现

通过 FFT 自动发现主要周期：

$$\mathcal{F}_p = \text{FFT}(X) \in \mathbb{C}^{L}$$

$$\text{Amp}(f_k) = |\mathcal{F}_p[k]|$$

选择振幅最大的 $k$ 个频率，对应的周期为：

$$p_1, p_2, \ldots, p_k = \arg\text{Top-}k \text{Amp}(\mathcal{F}_p)$$

### 5.3 1D → 2D 重塑

对每个周期 $p_i$，将 1D 序列重塑为 2D 张量：

$$\mathbf{X}_{2D}^{p_i} = \text{Reshape}_{p_i \times f_i}(X), \quad f_i = \lceil L / p_i \rceil$$

其中：
- 行维度 = 周期内变化（intra-period）
- 列维度 = 周期间变化（inter-period）

### 5.4 2D Inception 卷积

$$\mathbf{X}_{2D}^{p_i \prime} = \text{Inception2D}(\mathbf{X}_{2D}^{p_i})$$

Inception2D 使用多尺度卷积核（$1 \times 1, 3 \times 3, 5 \times 5, 7 \times 7$）同时捕获不同尺度的 2D 变化模式。

### 5.5 自适应聚合

$$\hat{Y} = \sum_{i=1}^{k} \text{softmax}(A_i) \cdot \text{Reshape1D}(\mathbf{X}_{2D}^{p_i \prime})$$

其中 $A_i$ 为可学习的注意力权重。

### 5.6 统一多任务

TimesNet 的同一骨干网络支持多种任务，只需更换最后的预测头：
- **预测**：输出预测序列
- **分类**：输出类别概率
- **异常检测**：输出重构误差
- **插补**：输出填充值

---

## 6. ModernTCN: 现代时序卷积 (2024)

**论文**: *ModernTCN: Modern Temporal Convolutional Network for Time Series Analysis*
**涉及路线**: [路线2: 序列分解] [路线3: MLP革命]

### 6.1 大核卷积 + 分解

结合现代卷积设计（大核、深度可分离）和分解策略：

$$X_t, X_s = \text{Decomp}(X)$$

$$X_t' = \text{DWConv}_{L_k}(X_t) + \text{FFN}(X_t)$$

$$X_s' = \text{DWConv}_{L_k}(X_s) + \text{FFN}(X_s)$$

其中 DWConv 为深度可分离卷积，$L_k$ 为大卷积核大小（如 51, 71），感受野远大于传统 TCN。

---

## 7. WaveForM: 小波图分解 (2023)

**论文**: *WaveForM: Graph Enhanced Wavelet Learning for Long Sequence Forecasting*
**引用数**: 25
**涉及路线**: [路线2: 序列分解] [路线6: 频域]

### 7.1 小波分解 + 图神经网络

$$\mathbf{A}^{(j)}, \mathbf{D}^{(j)} = \text{DWT}(X)$$

$$\hat{Y}_t = \text{GraphConv}(\mathbf{A}^{(j)}), \quad \hat{Y}_s = \text{GraphConv}(\mathbf{D}^{(j)})$$

$$\hat{Y} = \text{IDWT}(\hat{Y}_t, \hat{Y}_s)$$

---

## 8. DeMa: 双路径分解 Mamba (2026)

**论文**: *DeMa: Dual-Path Delay-Aware Mamba for Efficient Multivariate Time Series Analysis*
**涉及路线**: [路线2: 序列分解] [路线8: Mamba]

### 8.1 分解 + Mamba 双路径

$$X_t, X_s = \text{Decomp}(X)$$

$$\hat{Y}_t = \text{Mamba}_{\text{trend}}(X_t), \quad \hat{Y}_s = \text{Mamba}_{\text{seasonal}}(X_s)$$

$$\hat{Y} = \hat{Y}_t + \hat{Y}_s$$

---

## 9. DecMamba: 分解增强 Mamba (2024)

**论文**: *DecMamba: Mamba Utilizing Series Decomposition for Multivariate Time Series Forecasting*
**涉及路线**: [路线2: 序列分解] [路线8: Mamba]

### 9.1 方法

与 DeMa 类似的思路，但使用不同的 Mamba 变体：

$$X_t, X_s = \text{MovingAvg}(X)$$

$$\hat{Y} = \text{SSM}(X_t) + \text{SSM}(X_s)$$

---

## 10. BHT-ARIMA: 张量分解 (2020)

**论文**: *BHT-ARIMA: Bayesian Hierarchical Tensor ARIMA*
**引用数**: 92
**涉及路线**: [路线2: 序列分解]

### 10.1 多路延迟嵌入 + 张量分解

将多变量时间序列构建为 Block Hankel 张量，通过 Tucker 分解降维：

$$\mathcal{X} \approx \mathcal{G} \times_1 U_1 \times_2 U_2 \times_3 U_3$$

然后在核心张量 $\mathcal{G}$ 上运行 ARIMA：

$$\hat{\mathcal{G}}_t = \sum_{p=1}^{P} \phi_p \mathcal{G}_{t-p} + \epsilon_t$$

---

## 11. 演进对比总结

| 模型 | 年份 | 分解方式 | 分解位置 | 建模方式 | 引用 |
|------|------|---------|---------|---------|------|
| STL | 1990 | 移动平均 | 预处理 | 独立建模 | 经典 |
| **Autoformer** | **2021** | **移动平均** | **每层内置** | **自相关注意力** | **1311** |
| **FEDformer** | **2022** | **频域低通** | **模块内置** | **频域注意力** | **535** |
| **DLinear** | **2022** | **移动平均** | **预处理** | **单层线性** | **2314** |
| **TimesNet** | **2022** | **多周期FFT** | **每层内置** | **2D卷积** | **420** |
| ModernTCN | 2024 | 移动平均 | 模块内置 | 大核卷积 | — |
| WaveForM | 2023 | 小波DWT | 模块内置 | 图卷积 | 25 |
| DecMamba | 2024 | 移动平均 | 预处理 | SSM | — |
| DeMa | 2026 | 移动平均 | 双路径 | Mamba | — |

---

## 12. 演进逻辑图

```
STL 分解 (1990s, 预处理)
  │
  └─→ Autoformer (2021, 内置分解) ← 开创性转变
        │
        ├─→ FEDformer (2022, 频域分解) → [路线6: 频域]
        │
        ├─→ DLinear (2022, 分解+线性) → [路线3: MLP革命]
        │     │
        │     └─→ 证明"分解比模型架构更重要"
        │
        ├─→ TimesNet (2022, 多周期分解+2D建模) → [路线7: FM]
        │
        ├─→ ModernTCN (2024, 分解+大核卷积) → [路线3: MLP]
        │
        └─→ DecMamba/DeMa (2024-2026, 分解+Mamba) → [路线8: Mamba]
```

**核心结论**：DLinear 的实验表明，**分解策略本身可能比模型架构更重要**——即使是单层线性映射，配合适当的分解也能超越复杂的 Transformer。这一发现深刻影响了后续研究。

---

*本文档基于 2669 篇论文的 AI 深度分析。*
*数据来源：OpenAlex · Semantic Scholar · arXiv | AI 分析引擎：Claude*
