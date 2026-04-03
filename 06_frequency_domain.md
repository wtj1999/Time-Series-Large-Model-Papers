# 技术演进路线六：频域方法

> **核心问题**: 频域能否提供比时域更高效的特征表示?
>
> **起点**: FEDformer (2022) | 引用数: 535
>
> **涉及论文**: FEDformer, FreTS, FITS, SpectroMamba, SFDformer, Dualformer, MSGNet, WaveForM 等
>
> **交叉路线**: FEDformer → [路线1: Transformer效率] [路线2: 序列分解]; SpectroMamba → [路线8: Mamba]; WaveForM → [路线2: 分解]

---

## 1. 问题背景

传统时域方法在时间轴上直接处理时间序列, 但时间序列的许多重要特征（周期性、趋势、多尺度模式）在**频域**中更加明显和易于提取。

离散傅里叶变换 (DFT):
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i2\pi kn/N}, \quad k = 0, 1, \ldots, N-1$$

频谱:
$$|X[k]| = \sqrt{\text{Re}(X[k])^2 + \text{Im}(X[k])^2}$$

频率分量的振幅直接揭示了序列中包含哪些周期成分及其强度。

---

## 2. FEDformer: 频域注意力 (2022)
**论文**: *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*
**会议**: ICML 2022 | **引用数**: 535
**涉及路线**: [路线1: Transformer效率] [路线2: 序列分解] [路线6: 频域]

### 2.1 傅里叶增强模块 (FEB)
$$\mathbf{Z} = \text{FFT}(\mathbf{X}) \in \mathbb{C}^{L \times d}$$
选择 $M$ 个频率分量:
$$\hat{\mathbf{Z}} = \text{Select}_M(\mathbf{Z}) \in \mathbb{C}^{M \times d}$$
频域线性变换:
$$\hat{\mathbf{Z}}' = \hat{\mathbf{Z}} \cdot W + b$$
IFFT 回到时域:
$$\mathbf{X}' = \text{IFFT}(\text{Pad}(\hat{\mathbf{Z}}'))$$
**复杂度**: $O(Md)$, 其中 $M \ll L$。

### 2.2 小波增强模块 (FEW)
使用离散小波变换 (DWT):
$$\mathbf{A}^{(j)} = \sum_n h[n] \cdot \mathbf{X}[2n+j], \quad \mathbf{D}^{(j)} = \sum_n g[n] \cdot \mathbf{X}[2n+j]$$
其中 $h[n]$ 为低通滤波器, $g[n]$ 为高通滤波器。

### 2.3 频域分解
$$X_{\text{trend}} = \text{IFFT}\left(\sum_{k=1}^{M} X_f[k] \cdot e^{i2\pi kt/L}\right)$$
$$X_{\text{seasonal}} = X - X_{\text{trend}}$$

---

## 3. FreTS: 频域 MLP (2023)
**论文**: *Frequency-domain MLPs are More Effective Learners in Time Series Forecasting*
**引用数**: 91
**涉及路线**: [路线6: 频域] [路线3: MLP革命]

### 3.1 核心洞察
频域上的简单 MLP 就能超越时域的复杂模型。

### 3.2 频域 MLP
**第一步**: FFT 投影
$$\mathbf{X}_f = \text{FFT}(\mathbf{X}) \in \mathbb{C}^{L \times d}$$

**第二步**: 频域线性变换（分别处理实部和虚部）
$$\text{Re}(\mathbf{Y}_f) = \text{Re}(\mathbf{X}_f) \odot W_r - \text{Im}(\mathbf{X}_f) \odot W_i$$
$$\text{Im}(\mathbf{Y}_f) = \text{Re}(\mathbf{X}_f) \odot W_i + \text{Im}(\mathbf{X}_f) \odot W_r$$
等价于复数乘法: $\mathbf{Y}_f = \mathbf{X}_f \cdot (W_r + iW_i)$

**第三步**: IFFT 回时域
$$\hat{Y} = \text{IFFT}(\mathbf{Y}_f)$$

### 3.3 为什么频域 MLP 更好？
- **全局感受野**: 频域中每个分量看到整个序列
- **解耦**: 不同频率分量独立处理
- **效率**: 低频分量数量远小于时间步数

---

## 4. FITS: 频域插值 (2024)
**论文**: *FITS: Frequency Interpolation Time Series Forecasting*
**涉及路线**: [路线6: 频域]

### 4.1 频域上采样
将输入序列通过频域零填充上采样到预测长度:

$$\mathbf{X}_f = \text{FFT}(\mathbf{X}) \in \mathbb{C}^{L_{\text{in}}}$$
$$\hat{\mathbf{X}}_f = \text{ZeroPad}(\mathbf{X}_f, L_{\text{out}}) \in \mathbb{C}^{L_{\text{out}}}$$
$$\hat{Y} = \text{IFFT}(\hat{\mathbf{X}}_f)$$

### 4.2 理论基础
频域零填充等价于时域 sinc 插值（香农插值定理）:
$$\hat{y}_t = \sum_{n=0}^{N-1} x[n] \cdot \text{sinc}\left(\frac{t - n \cdot L_{\text{out}}/L_{\text{in}}}{L_{\text{out}}/L_{\text{in}}}\right)$$
这是带限信号的最优插值。

---

## 5. SpectroMamba: 频域 Mamba (2025)
**论文**: *SpectroMamba: Complex-Valued Mamba for Spectral Time Series Modeling*
**引用数**: 10
**涉及路线**: [路线6: 频域] [路线8: Mamba]

### 5.1 复数 SSM
在频域中运行 Mamba SSM, 所有参数为复数:
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$
其中 $h_t, x_t, y_t, \bar{A}, \bar{B}, C \in \mathbb{C}$。

### 5.2 频谱门控
$$\mathbf{X}_f' = \sigma(W_g \cdot \text{FFT}(X) + b_g) \odot \text{FFT}(X)$$
可学习的频谱门控选择重要频率分量。

---

## 6. SFDformer: 稀疏频域 (2025)
**论文**: *SFDformer: a frequency-based sparse decomposition transformer*
**引用数**: 8
**涉及路线**: [路线6: 频域] [路线1: Transformer效率]

### 6.1 可学习稀疏掩码
$$\text{SparseMask}(\mathbf{Z}) = \sigma(W_m \cdot \mathbf{Z} + b_m)$$
$$\hat{\mathbf{Z}} = \text{SparseMask}(\mathbf{Z}) \odot \mathbf{Z}$$
用 sigmoid 激活的掩码选择性地保留重要频率分量, 比固定阈值更灵活。

---

## 7. MSGNet: 多尺度频域图网络 (2024)
**论文**: *MSGNet: Learning Multi-scale Sequence Inter-correlations for Time Series Forecasting*
**引用数**: 171
**涉及路线**: [路线6: 频域]

### 7.1 频域多尺度分解
$$\text{DFT}(X) = \sum_k X_f[k] \cdot e^{i2\pi kt/L}$$
按频率振幅选择多尺度窗口。

### 7.2 自适应图卷积
每个尺度上学习变量间的图结构:
$$A_s = \text{softmax}(\text{ReLU}(E_1 \cdot E_2^\top))$$
$$\hat{Y}_s = \text{GraphConv}(X_s, A_s)$$

---

## 8. Dualformer: 时频双域学习 (2025)
**论文**: *Dualformer: Time-Frequency Dual Domain Learning for Long-term Time Series Forecasting*
**涉及路线**: [路线6: 频域]

### 8.1 双路径
$$\hat{Y}_{\text{time}} = f_{\text{time}}(X), \quad \hat{Y}_{\text{freq}} = \text{IFFT}(f_{\text{freq}}(\text{FFT}(X)))$$
$$\hat{Y} = \alpha \cdot \hat{Y}_{\text{time}} + (1 - \alpha) \cdot \hat{Y}_{\text{freq}}$$
其中 $\alpha$ 为可学习的混合权重。

---

## 9. 演进对比总结
| 模型 | 年份 | 频域方法 | 时域方法 | 复杂度 | 引用 |
|------|------|---------|---------|--------|------|
| **FEDformer** | **2022** | **FFT/DWT + 选择** | **分解** | **$O(M)$** | **535** |
| **FreTS** | **2023** | **复数 MLP** | **无** | **$O(L \log L)$** | **91** |
| FITS | 2024 | FFT 零填充插值 | 无 | $O(L \log L)$ | — |
| **SpectroMamba** | **2025** | **复数 SSM** | **无** | **$O(L)$** | **10** |
| SFDformer | 2025 | 可学习稀疏掩码 | 分解 | $O(M')$ | 8 |
| MSGNet | 2024 | DFT 多尺度 | 图卷积 | $O(L \log L)$ | 171 |
| Dualformer | 2025 | 双路径 | Transformer | $O(L^2)$ | — |

---

## 10. 演进逻辑图
```
传统时域方法 (RNN, Transformer, MLP)
  │
  └─→ FEDformer (2022, FFT/DWT 注意力) ← 频域起点
        │
        ├─→ FreTS (2023, 频域 MLP)
        │     └─→ 证明: 频域 MLP > 时域 MLP
        │
        ├─→ FITS (2024, 频域插值)
        │     └─→ 最简: 零填充即可预测
        │
        ├─→ SFDformer (2025, 稀疏频域)
        │     └─→ 可学习频率选择
        │
        ├─→ SpectroMamba (2025, 复数 Mamba)
        │     └─→ [→ 路线8: Mamba]
        │
        ├─→ MSGNet (2024, 多尺度频域图)
        │     └─→ 多变量频域交互
        │
        └─→ Dualformer (2025, 时频双域)
              └─→ 时域+频域互补
```

**核心结论**: FEDformer 首次证明频域注意力的有效性, FreTS 进一步证明频域上简单 MLP 就够用。频域方法天然适合捕获全局周期模式, 但对非周期信号可能不如时域灵活。

---

*本文档基于 2669 篇论文的 AI 深度分析.*
*数据来源: OpenAlex · Semantic Scholar · arXiv | AI 分析引擎: Claude*
