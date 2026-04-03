# 技术演进路线五：非平稳处理 / 归一化

> **核心问题**：如何处理时间序列中的分布漂移和非平稳性？
>
> **起点**：RevIN (2022) | 引用数：608+
>
> **涉及论文**：RevIN, SAN, Dish-TS, Non-stationary Transformer, ST-Norm, AdaRNN, FlatFormer, TimeAPN, U-Mixer 等
>
> **交叉路线**：Non-stationary Transformer → [路线1: Transformer效率]；ST-Norm → [路线12: 时空建模]

---

## 1. 问题背景

现实世界的时间序列通常具有**非平稳性（non-stationarity）**——统计特性（均值、方差）随时间变化。这导致：

1. **训练-测试分布偏移**：训练数据的分布与测试数据不同
2. **模型性能退化**：在训练集上学到的模式在测试集上失效

数学表达：

$$P_{\text{train}}(X_t) \neq P_{\text{test}}(X_t)$$

具体而言，时间序列的统计量漂移：

$$\mu_t = \mathbb{E}[X_t] \neq \mu_{t+\Delta}, \quad \sigma_t^2 = \text{Var}(X_t) \neq \sigma_{t+\Delta}^2$$

---

## 2. RevIN: 可逆实例归一化 (2022)

**论文**: *RevIN: Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*
**会议**: ICLR 2022 | **引用数**: 608+
**涉及路线**: [路线5: 归一化]

### 2.1 核心洞察

在输入模型前归一化（去除统计量），预测后反归一化（恢复统计量），使模型专注于学习**去除了分布偏移的纯模式**。

### 2.2 前向归一化

$$\mu = \frac{1}{T}\sum_{t=1}^{T} x_t, \quad \sigma = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(x_t - \mu)^2 + \epsilon}$$

$$x'_t = \frac{x_t - \mu}{\sigma}$$

### 2.3 模型预测

$$\hat{y}' = f_\theta(x')$$

### 2.4 逆归一化

$$\hat{y}_t = \hat{y}'_t \cdot \sigma + \mu$$

### 2.5 完整流程

$$\hat{Y} = \sigma \cdot f_\theta\left(\frac{X - \mu}{\sigma}\right) + \mu$$

### 2.6 实验效果

- 在 ETT、Electricity、Traffic 等数据集上，MSE 降低 **20%-60%**
- 可作为**即插即用模块**应用于任何模型
- 与 PatchTST、iTransformer 等后续方法兼容

---

## 3. SAN: 自适应归一化 (2022)

**论文**: *SAN: Self-Adaptive Normalization for Time Series Forecasting*
**涉及路线**: [路线5: 归一化]

### 3.1 可学习归一化参数

在 RevIN 的基础上，将归一化参数设为可学习：

$$x'_t = \gamma \cdot \frac{x_t - \mu}{\sigma + \epsilon} + \beta$$

其中 $\gamma$ 和 $\beta$ 为可学习参数，通过反向传播自适应调整归一化强度。

### 3.2 实例感知

$$\gamma = f_\gamma(\mu, \sigma), \quad \beta = f_\beta(\mu, \sigma)$$

从输入的统计量中动态预测归一化参数。

---

## 4. Dish-TS: 分布偏移处理 (2022)

**论文**: *Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting*
**引用数**: 84
**涉及路线**: [路线5: 归一化]

### 4.1 双分布映射

Dish-TS 分别建模输入空间和输出空间的分布：

**编码器侧映射（input shift）**：

$$x'_t = \phi_e(x_t) = \frac{x_t - \mu_e}{\sigma_e + \epsilon}$$

**解码器侧映射（output shift）**：

$$\hat{y}_t = \phi_d(\hat{y}'_t) = \hat{y}'_t \cdot \sigma_d + \mu_d$$

### 4.2 可学习分布参数

$$\mu_e, \sigma_e = \text{CoNet}_e(X), \quad \mu_d, \sigma_d = \text{CoNet}_d(X)$$

$\text{CoNet}$ 为可学习的系数网络，从输入序列中预测分布参数。

### 4.3 关键区别

| 方法 | 输入处理 | 输出处理 | 分布来源 |
|------|---------|---------|---------|
| RevIN | $x' = (x - \mu) / \sigma$ | $\hat{y} = \hat{y}' \sigma + \mu$ | 同一组 $\mu, \sigma$ |
| **Dish-TS** | $x' = (x - \mu_e) / \sigma_e$ | $\hat{y} = \hat{y}' \sigma_d + \mu_d$ | **分离的** $\mu_e, \sigma_e, \mu_d, \sigma_d$ |

Dish-TS 允许输入和输出使用不同的分布参数，更好地处理 train-test shift。

---

## 5. Non-stationary Transformer (2022)

**论文**: *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting*
**会议**: NeurIPS 2022
**涉及路线**: [路线5: 归一化] [路线1: Transformer效率]

### 5.1 核心洞察

直接归一化（如 RevIN）会**过度平稳化**数据，丢失非平稳信息。Non-stationary Transformer 的解决方案：归一化输入，但将原始统计量**嵌入到注意力机制**中。

### 5.2 去平稳注意力（De-stationary Attention）

$$\text{DeStationaryAttn}(Q', K', V') = \text{softmax}\left(\frac{Q'K'^\top}{\sqrt{d_k}} + \tau \cdot \Delta\right)V'$$

其中：
- $Q', K', V'$ 为归一化后的查询、键、值
- $\tau$ 为从原始方差 $\sigma$ 学习的缩放因子
- $\Delta$ 为从原始统计量学习的偏置项

### 5.3 统计量嵌入

$$\tau = f_\tau(\sigma) = \text{softplus}(W_\tau \cdot \log(\sigma) + b_\tau)$$

$$\Delta = f_\Delta(\mu, \sigma) = W_\Delta \cdot [\mu; \sigma; \mu/\sigma]$$

### 5.4 层级统计量传播

在每个 Transformer 层都注入统计量：

$$Q' = W_Q \cdot x' + \Delta_Q, \quad K' = W_K \cdot x' + \Delta_K$$

---

## 6. ST-Norm: 时空归一化 (2021)

**论文**: *ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting*
**引用数**: 154
**涉及路线**: [路线5: 归一化]

### 6.1 时间归一化

去除低频趋势，保留高频波动：

$$x'_t = \frac{x_t - \bar{x}_t}{\sqrt{\text{Var}(x_t - \bar{x}_t)}}$$

其中 $\bar{x}_t$ 为低频分量（通过滑动平均提取）。

### 6.2 空间归一化

去除全局均值，保留局部差异：

$$x'_d = \frac{x_d - \bar{x}_{\text{global}}}{\sqrt{\text{Var}(x_d - \bar{x}_{\text{global}})}}$$

### 6.3 可插拔设计

两个归一化模块可独立或组合插入任何骨干网络（WaveNet, Transformer 等）。

---

## 7. AdaRNN: 自适应分布匹配 (2021)

**论文**: *AdaRNN: Adaptive Deep Learning for Time Series Forecasting via Distribution Matching*
**引用数**: 221
**涉及路线**: [路线5: 归一化]

### 7.1 时间分布匹配

最小化训练和测试数据的分布差异：

$$\mathcal{L}_{\text{TDM}} = \text{MMD}(P_{\text{train}}, P_{\text{test}})$$

$$\text{MMD}(P, Q) = \left\| \frac{1}{n}\sum_i \phi(x_i) - \frac{1}{m}\sum_j \phi(y_j) \right\|^2$$

其中 $\phi$ 为核函数映射。

### 7.2 分布表征

$$h_t = \text{GRU}(x_t, h_{t-1}) + \text{AdaWeight}_t \cdot h_{t-1}$$

自适应权重 $\text{AdaWeight}_t$ 从时间特征中学习，动态调整对历史信息的依赖。

---

## 8. FlatFormer: 分块归一化 (2024)

**论文**: *FlatFormer: Flattened Attention for Time Series Forecasting*
**涉及路线**: [路线5: 归一化]

### 8.1 自适应分块归一化

将序列分为多个块，每块独立归一化：

$$x'_{[c]} = \frac{x_{[c]} - \mu_c}{\sigma_c + \epsilon}$$

其中块边界通过变化点检测确定：

$$\text{boundary}_c = \{t : |x_t - x_{t-1}| > \theta\}$$

### 8.2 优势

- 避免对平稳段过度归一化
- 对非平稳段精确归一化

---

## 9. TimeAPN: 幅度-相位归一化 (2025)

**论文**: *TimeAPN: Adaptive Amplitude-Phase Non-Stationarity Normalization for Time Series*
**涉及路线**: [路线5: 归一化]

### 9.1 频域分离幅度和相位

$$\mathbf{A}, \mathbf{\Phi} = |\text{FFT}(X)|, \arg(\text{FFT}(X))$$

### 9.2 幅度归一化

$$\mathbf{A}' = \frac{\mathbf{A} - \bar{A}}{\sigma_A + \epsilon}$$

### 9.3 相位自适应偏移

$$\mathbf{\Phi}' = \mathbf{\Phi} + \Delta\Phi, \quad \Delta\Phi = f_\Phi(\mathbf{\Phi})$$

### 9.4 逆变换

$$X' = \text{IFFT}(\mathbf{A}' \cdot e^{i\mathbf{\Phi}'})$$

---

## 10. U-Mixer: 平稳性校正 (2024)

**论文**: *U-Mixer: Unet-Mixer for Time Series Forecasting with Stationarity Correction*
**引用数**: 31
**涉及路线**: [路线5: 归一化]

### 10.1 U-Net + Mixer 架构

结合 U-Net 的跳跃连接和 MLP-Mixer 的混合机制。

### 10.2 平稳性校正损失

$$\mathcal{L}_{\text{stationarity}} = \lambda \left\| \text{ADF}(f_\theta(X)) - \text{ADF}(X) \right\|^2$$

约束模型输出的平稳性与输入一致。

---

## 11. 演进对比总结

| 方法 | 年份 | 核心策略 | 分布参数 | 即插即用 | 引用 |
|------|------|---------|---------|---------|------|
| **RevIN** | **2022** | **实例归一化** | **输入统计量** | **是** | **608** |
| SAN | 2022 | 可学习归一化 | 自适应 | 是 | — |
| **Dish-TS** | **2022** | **双分布映射** | **输入/输出分离** | **是** | **84** |
| **Non-stationary Trans.** | **2022** | **注意力嵌入统计量** | **输入+学习** | 否 | — |
| ST-Norm | 2021 | 时空分离归一化 | 自适应 | 是 | 154 |
| AdaRNN | 2021 | 分布匹配 | MMD | 否 | 221 |
| FlatFormer | 2024 | 分块归一化 | 块级 | 是 | — |
| TimeAPN | 2025 | 频域幅度-相位 | 频率级 | 是 | — |

---

## 12. 演进逻辑图

```
BatchNorm / LayerNorm (通用归一化)
  │
  ├─→ RevIN (2022, 实例归一化 + 逆变换)
  │     │
  │     ├─→ SAN (可学习 γ, β)
  │     │
  │     ├─→ Dish-TS (双分布映射)
  │     │
  │     ├─→ Non-stationary Transformer (统计量嵌入注意力)
  │     │     └─→ [→ 路线1: Transformer]
  │     │
  │     ├─→ FlatFormer (分块归一化)
  │     │
  │     └─→ TimeAPN (频域幅度-相位归一化)
  │           └─→ [→ 路线6: 频域]
  │
  ├─→ ST-Norm (时空分离归一化)
  │
  └─→ AdaRNN (分布匹配 + MMD)
```

**核心结论**：RevIN 的"先归一化、后反归一化"范式成为最广泛使用的策略，可即插即用于几乎所有时序模型。Non-stationary Transformer 进一步指出需要保留非平稳信息而非完全去除。

---

*本文档基于 2669 篇论文的 AI 深度分析。*
*数据来源：OpenAlex · Semantic Scholar · arXiv | AI 分析引擎：Claude*
