# 技术演进路线八：Mamba / SSM 架构

> **核心问题**: 能否用状态空间模型 (SSM) 替代 Transformer, 实现真正的线性复杂度?
>
> **起点**: S4 (2021) → Mamba (2023) → MambaTS (2024) | 引用数: 145
>
> **涉及论文**: S4, Mamba, MambaTS, TimeMachine, CMMamba, SpectroMamba, STM³, ASGMamba, DecMamba 等
>
> **交叉路线**: SpectroMamba → [路线6: 频域]; DecMamba/DeMa → [路线2: 序列分解]; ModernTCN → [路线3: MLP]

---

## 1. 问题背景

Transformer 的注意力机制虽然灵活, 但二次复杂度 $O(L^2)$ 始终是瓶颈。Informer 等方法通过稀疏化降至 $O(L \log L)$, 但本质未变。

状态空间模型 (State Space Model, SSM) 提供了**真正的线性复杂度**替代方案: $O(L)$, 同时具有长程记忆能力。

---

## 2. S4: 结构化状态空间 (2021)

**论文**: *Efficiently Modeling Long Sequences with Structured State Spaces*
**会议**: ICLR 2022 | **引用数**: 1000+
**涉及路线**: [路线8: SSM]

### 2.1 连续 SSM 基础

$$\frac{dh(t)}{dt} = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

其中:
- $h(t) \in \mathbb{R}^N$ 为隐状态
- $A \in \mathbb{R}^{N \times N}$ 为状态转移矩阵
- $B \in \mathbb{R}^{N \times 1}$ 为输入矩阵
- $C \in \mathbb{R}^{1 \times N}$ 为输出矩阵
- $D \in \mathbb{R}$ 为直连项

### 2.2 离散化 (Zero-Order Hold)

$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

离散递推:
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t + Dx_t$$

### 2.3 两种计算模式

**递归模式** (推理):
$$y_t = C\bar{A}h_{t-1} + C\bar{B}x_t + Dx_t$$
每步 $O(N)$, 总 $O(NL)$。

**卷积模式** (训练并行化):
$$y = K * x$$
$$K = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots, C\bar{A}^{L-1}\bar{B})$$
通过 FFT 加速: $O((N+L)\log L)$。

### 2.4 HiPPO 初始化

S4 的关键创新: 用 HiPPO (High-order Polynomial Projection Operators) 初始化 $A$ 矩阵:

$$A_{\text{HiPPO}} = -\frac{1}{2}I - \frac{1}{2}S$$

其中 $S$ 为特殊的斜对称矩阵。HiPPO 理论证明这种初始化能最优地压缩历史信息到隐状态。

### 2.5 对角结构化

S4 将 $A$ 约束为对角 + 低秩结构, 使计算可并行化:

$$A = \text{diag}(\lambda_1, \ldots, \lambda_N) + PQ^\top$$

---

## 3. Mamba / S6: 选择性 SSM (2023)

**论文**: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
**作者**: Gu & Dao | **引用数**: 5000+ (跨领域)
**涉及路线**: [路线8: SSM]

### 3.1 核心创新: 选择性机制

S4 的参数 $B, C, \Delta$ 是**固定的** (与输入无关)。Mamba 使其变为**输入依赖的**:

$$B_t = \text{Linear}_B(x_t) \in \mathbb{R}^{N}$$
$$C_t = \text{Linear}_C(x_t) \in \mathbb{R}^{N}$$
$$\Delta_t = \text{softplus}(\text{Linear}_\Delta(x_t)) \in \mathbb{R}$$

### 3.2 输入依赖离散化

$$\bar{A}_t = \exp(\Delta_t A)$$
$$\bar{B}_t = \Delta_t B_t$$

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$
$$y_t = C_t h_t$$

### 3.3 选择性扫描算法

由于 $B_t, C_t, \Delta_t$ 依赖于输入, 无法使用卷积模式。Mamba 设计了**硬件感知的并行扫描算法**:

$$h_1 = \bar{A}_1 h_0 + \bar{B}_1 x_1$$
$$h_2 = \bar{A}_2 h_1 + \bar{B}_2 x_2 = \bar{A}_2\bar{A}_1 h_0 + \bar{A}_2\bar{B}_1 x_1 + \bar{B}_2 x_2$$

通过并行扫描 (parallel scan) 在 $O(L \log L)$ 内完成, 实际 GPU 利用率接近 $O(L)$。

### 3.4 Mamba 块

完整的 Mamba 块:
$$z_t = \sigma(\text{Conv1d}(\text{Linear}(x_t)))$$
$$y_t = \text{SSM}_\theta(z_t) \odot \text{Linear}(x_t)$$

包含:
1. 线性投影
2. 因果卷积 (Conv1d)
3. SiLU 激活
4. 选择性 SSM
5. 门控机制 (与输入相乘)

### 3.5 复杂度对比

| 模型 | 训练 | 推理 (per token) | 长程记忆 |
|------|------|-----------------|---------|
| Transformer | $O(L^2)$ | $O(L)$ (需 KV cache) | 有界 |
| **Mamba** | **$O(L)$** | **$O(1)$** (无 cache) | **无限** |

---

## 4. MambaTS: Mamba 用于时序预测 (2024)

**论文**: *Is Mamba effective for time series forecasting?*
**引用数**: 145
**涉及路线**: [路线8: SSM] [路线1: Transformer效率]

### 4.1 双向 Mamba 编码

标准 Mamba 是因果的 (只看过去)。MambaTS 引入双向编码:

$$\overrightarrow{h}_t = \text{Mamba}(x_{\leq t}), \quad \overleftarrow{h}_t = \text{Mamba}(x_{\geq t})$$
$$h_t = \overrightarrow{h}_t + \overleftarrow{h}_t$$

预测时能看到完整输入序列。

### 4.2 Patch + Mamba

结合 Patching (借鉴 PatchTST):
$$\mathbf{Z} = \text{Patch}(X)$$
$$\hat{Y} = \text{BiMamba}(\mathbf{Z})$$

Patch 降低序列长度, Mamba 高效处理。

---

## 5. TimeMachine: 四 Mamba 架构 (2024)

**论文**: *TimeMachine: A Time Series is Worth 4 Mambas for Long Sequence Forecasting*
**引用数**: 80
**涉及路线**: [路线8: SSM]

### 5.1 四 Mamba 模块

$$Z_1 = M_1(X)$$  (时间维度, 通道独立)
$$Z_2 = M_2(Z_1)$$  (时间维度, 深层)
$$Z_3 = M_3(Z_2^\top)^\top$$  (通道维度)
$$Z_4 = M_4(Z_3)$$  (最终聚合)

交替在时间维度和变量维度上运行 Mamba。

---

## 6. CMMamba: 通道混合 Mamba (2024)

**论文**: *CMMamba: Channel Mixing Mamba for Time Series Forecasting*
**引用数**: 17
**涉及路线**: [路线8: SSM]

### 6.1 通道混合策略

交替进行时间 Mamba 和通道 Mamba:
$$X' = \text{Mamba}_{\text{temporal}}(X)$$
$$X'' = \text{Mamba}_{\text{channel}}(X'^\top)^\top$$

沿通道轴运行 Mamba 建模变量间交互。

---

## 7. SpectroMamba: 频域 Mamba (2025)

**论文**: *SpectroMamba*
**引用数**: 10
**涉及路线**: [路线8: SSM] [路线6: 频域]

### 7.1 复数 SSM

在频域中运行 Mamba, 使用复数参数:
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad h_t \in \mathbb{C}^N$$
$$y_t = C h_t, \quad C \in \mathbb{C}^{1 \times N}$$

### 7.2 频谱门控

$$X_f' = \sigma(W_g \cdot \text{FFT}(X)) \odot \text{FFT}(X)$$
$$\hat{Y} = \text{IFFT}(\text{Mamba}_{\text{complex}}(X_f'))$$

---

## 8. STM³: 多尺度 Mamba (2025)

**论文**: *STM³: Mixture of Multiscale Mamba for Long-Term Spatio-Temporal Time-Series Prediction*
**涉及路线**: [路线8: SSM]

### 8.1 多尺度 Mamba

$$h^{(s)}_t = \text{Mamba}^{(s)}(\text{Patch}_{p_s}(X))$$

多个 Mamba 模块分别处理不同尺度的 patch。

### 8.2 尺度混合

$$\hat{Y} = \sum_{s=1}^{S} \alpha_s \cdot \text{Proj}_s(h^{(s)})$$
$$\alpha_s = \text{softmax}(g_s), \quad g_s = \text{Linear}(\text{GAP}(h^{(s)}))$$

---

## 9. ASGMamba: 自适应频谱门控 Mamba (2026)

**论文**: *ASGMamba: Adaptive Spectral Gating Mamba for Multivariate Time Series Forecasting*
**涉及路线**: [路线8: SSM] [路线6: 频域]

### 9.1 频谱门控 + Mamba

$$X_f = \text{FFT}(X)$$
$$X_f' = \text{AdaptiveGate}(X_f) = \sigma(W_g X_f + b_g) \odot X_f$$
$$X' = \text{IFFT}(X_f')$$
$$\hat{Y} = \text{BiMamba}(X')$$

---

## 10. DecMamba: 分解增强 Mamba (2024)

**论文**: *DecMamba: Mamba Utilizing Series Decomposition for Multivariate Time Series*
**涉及路线**: [路线8: SSM] [路线2: 序列分解]

### 10.1 分解 + 双路径 Mamba

$$X_t = \text{AvgPool}(X), \quad X_s = X - X_t$$
$$\hat{Y}_t = \text{Mamba}(X_t), \quad \hat{Y}_s = \text{Mamba}(X_s)$$
$$\hat{Y} = \hat{Y}_t + \hat{Y}_s$$

---

## 11. 演进对比总结

| 模型 | 年份 | SSM 类型 | 方向性 | 特殊设计 | 复杂度 | 引用 |
|------|------|---------|--------|---------|--------|------|
| S4 | 2021 | 结构化 | 单向 | HiPPO | $O(L)$ | 1000+ |
| **Mamba** | **2023** | **选择性** | **单向** | **输入依赖参数** | **$O(L)$** | **5000+** |
| MambaTS | 2024 | 选择性 | 双向 | Patch + BiMamba | $O(L)$ | 145 |
| TimeMachine | 2024 | 选择性 | 双向 | 4 Mamba | $O(L)$ | 80 |
| CMMamba | 2024 | 选择性 | 双向 | 通道混合 | $O(L)$ | 17 |
| SpectroMamba | 2025 | 复数 SSM | 双向 | 频域 | $O(L)$ | 10 |
| STM³ | 2025 | 选择性 | 双向 | 多尺度混合 | $O(L)$ | — |
| ASGMamba | 2026 | 选择性 | 双向 | 频谱门控 | $O(L)$ | — |
| DecMamba | 2024 | 选择性 | 双向 | 分解双路径 | $O(L)$ | — |

---

## 12. 演进逻辑图

```
S4 (2021, 结构化 SSM, HiPPO)
  │
  └─→ Mamba / S6 (2023, 选择性 SSM) ← 线性复杂度新架构
        │
        ├─→ MambaTS (2024, 双向 + Patch)
        │     └─→ [→ 路线1: Transformer效率]
        │
        ├─→ TimeMachine (2024, 4 Mamba)
        │
        ├─→ CMMamba (2024, 通道混合)
        │
        ├─→ SpectroMamba (2025, 复数频域 SSM)
        │     └─→ [→ 路线6: 频域]
        │
        ├─→ STM³ (2025, 多尺度 Mamba 混合)
        │
        ├─→ ASGMamba (2026, 频谱门控)
        │
        └─→ DecMamba / DeMa (2024-2026, 分解 + Mamba)
              └─→ [→ 路线2: 序列分解]
```

**核心定位**: Mamba/SSM 是 Transformer 的**线性复杂度替代**, 目前处于快速发展期, 各类变体涌现。关键问题: 在什么场景下 SSM 真正优于 Transformer?

---

*本文档基于 2669 篇论文的 AI 深度分析.*
*数据来源: OpenAlex · Semantic Scholar · arXiv | AI 分析引擎: Claude*
