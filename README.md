# Time Series Papers: Foundation Models, LLMs and Beyond

- 2026
    - LLM-based Model
        - [TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](https://iclr.cc/virtual/2026/poster/10007794) — **ICLR 2026**；提出时序推理任务套件，覆盖感知、外推与决策能力。[代码](https://github.com/AntonGuan/TimeOmni-1)
        - [TsLLM: Augmenting LLMs for General Time Series Understanding and Prediction](https://icml.cc/virtual/2026/poster/61098) — **ICML 2026**；使用分块编码器与解码器增强 LLM 的时序感知，统一理解、解释与预测。
        - [MemCast: Memory-Driven Time Series Forecasting with Experience-Conditioned Reasoning](https://icml.cc/virtual/2026/poster/64890) — **ICML 2026**；将历史模式、推理经验与时序规律组织为分层记忆，辅助预测推理。[代码](https://github.com/Xiaoyu-Tao/MemCast-TS)
        - [TiMi: Empower Time Series Transformers with Multimodal Mixture of Experts](https://icml.cc/virtual/2026/poster/64391) — **ICML 2026**；利用语言模型推断与多模态混合专家，将文本上下文融入数值预测。
        - [Rethinking Post-Training Recipes for Multimodal Time-Series Forecasting](https://neurips.cc/virtual/2026/poster/153768) — **NeurIPS 2026**；PostTime 结合监督微调与可验证奖励强化学习，让 LLM 按多模态上下文修正基础模型预测。
        - [AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection](https://icml.cc/virtual/2026/poster/64969) — **ICML 2026**；通过时序细节约束与强化学习，联合进行异常分类、定位和解释。
    - Foundation Model
        - [TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time-Series Analysis](https://iclr.cc/virtual/2026/poster/10010081) — **ICLR 2026**；轻量预训练模型，解耦时间、频率与语义表征，面向时序诊断与迁移。
        - [Baguan-TS: dual in-context learning model for time series forecasting with covariates](https://icml.cc/virtual/2026/poster/60703) — **ICML 2026**；在时间、变量和上下文三个轴上建模，支持带协变量的上下文学习预测。
        - [TempoPFN: Synthetic Pre-training of Linear RNNs for Zero-shot Time Series Forecasting](https://neurips.cc/virtual/2026/poster/155999) — **NeurIPS 2026**；仅用合成数据预训练线性 RNN，研究高效零样本时序预测。[代码](https://github.com/automl/TempoPFN)
        - [Timer-S1: A Billion-Scale Time Series Foundation Model with Serial Scaling](https://arxiv.org/abs/2603.04791) — **arXiv 2026-03（预印本）**；研究串行扩展的十亿参数级时序基础模型。
        - [TS-ICL: A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning](https://arxiv.org/abs/2606.05878) — **arXiv 2026-06（预印本）**；将预测与插补统一为时间戳对齐回归，支持协变量和部分观测。
        - [TimEE: End-to-end Time Series Classification via In-Context Learning](https://arxiv.org/abs/2607.07500) — **arXiv 2026-07（预印本）**；基于合成分类任务预训练，通过带标签支持集直接完成上下文分类。
    - Tricks for TSLM
        - [CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data](https://iclr.cc/virtual/2026/poster/10006662) — **ICLR 2026**；结合高斯过程核与结构因果模型生成合成数据，用于分类基础模型预训练。[代码](https://github.com/ShifengXIE/CauKer)
        - [CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter](https://iclr.cc/virtual/2026/poster/10010200) — **ICLR 2026**；通过相关性感知适配器，补充时序基础模型对跨变量依赖的建模。
        - [UniCA: Unified Covariate Adaptation for Time Series Foundation Model](https://iclr.cc/virtual/2026/poster/10010335) — **ICLR 2026**；统一数值、类别、图像和文本等异构协变量的表征与融合。[代码](https://github.com/hanlu-nju/UniCA)
        - [See More, Forecast Better and Faster: Enhancing Time Series Foundation Models via Inference-Time Plug-and-Play Downsampling](https://icml.cc/virtual/2026/poster/64120) — **ICML 2026**；SPRINT 通过趋势与季节项分解及下采样，在推理阶段扩展历史窗口并降低开销。[代码](https://github.com/NetManAIOps/SPRINT)
        - [OATS: Online Data Augmentation for Time Series Foundation Models](https://neurips.cc/virtual/2026/poster/153919) — **NeurIPS 2026**；按训练阶段动态生成合成时序数据，研究在线数据增强对基础模型的作用。
        - [STAR: Boosting Time Series Foundation Models for Anomaly Detection Through State-Aware Adapter](https://neurips.cc/virtual/2026/poster/153598) — **NeurIPS 2026**；以状态感知适配器区分离散状态与数值变量，增强基础模型异常检测。
        - [VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection](https://neurips.cc/virtual/2026/poster/149823) — **NeurIPS 2026**；对齐时序与视觉表征，融合局部定位和全局上下文以进行零样本异常检测。
    - Other Time Series Model
        - [MambaSL: Exploring Single-Layer Mamba for Time Series Classification](https://iclr.cc/virtual/2026/poster/10008907) — **ICLR 2026**；研究单层 Mamba 在多变量时序分类中的能力，并统一评测基线。[代码](https://github.com/yoom618/MambaSL)
        - [Latent Laplace Diffusion for Irregular Multivariate Time Series](https://icml.cc/virtual/2026/poster/61165) — **ICML 2026**；在拉普拉斯域描述潜在轨迹，面向不规则多变量时序预测与插补。[代码](https://github.com/pixelhero98/LLapDiffusion)
        - [KITE: Knowledge-Guided Probabilistic Modeling for Time Series Forecasting with Exogenous Variables](https://icml.cc/virtual/2026/poster/65231) — **ICML 2026**；借助历史条件流形与知识引导条件模块，进行带外生变量的概率预测。
        - [COMET: Codebook-based Online-adaptive Multi-scale Embedding for Time-series Anomaly Detection](https://neurips.cc/virtual/2026/poster/150690) — **NeurIPS 2026**；结合多尺度编码、正常模式码本与在线适配，处理异常检测中的分布漂移。[代码](https://github.com/JINU6497/COMET)
    - Benchmark
        - [TimeSeriesExamAgent: Creating Time Series Reasoning Benchmarks at Scale](https://iclr.cc/virtual/2026/poster/10010760) — **ICLR 2026**；以模板和 LLM Agent 扩展时序推理基准，覆盖模式、噪声、相似性、异常与因果。[代码](https://github.com/magwiazda/TimeSeriesExamAgent)
        - [TimeRecipe: A Time-Series Forecasting Recipe via Benchmarking Module Level Effectiveness](https://iclr.cc/virtual/2026/poster/10010822) — **ICLR 2026**；在模块层面评估分解、归一化等设计，分析预测模型组件的有效性。[代码](https://github.com/AdityaLab/TimeRecipe)

- 2025
    - LLM-based Model
        - [DualSG: A Dual-Stream Explicit Semantic-Guided Multivariate Time Series Forecasting Framework](https://arxiv.org/abs/2507.21830) — **ACM MM 2025**；以双流架构结合数值预测和显式语义引导。[代码](https://github.com/BenchCouncil/DualSG)[中文解读](https://zhuanlan.zhihu.com/p/1954949007026676229)
        - [TableTime: Reformulating Time Series Classification as Training-Free Table Understanding with Large Language Models](https://arxiv.org/abs/2411.15737) — **CIKM 2025**；将多变量时序转为文本表格，结合上下文与推理提示实现免训练分类。[代码](https://github.com/realwangjiahao/TableTime)
        - [Unleashing The Power of Pre-Trained Language Models for Irregularly Sampled Time Series](https://arxiv.org/abs/2408.08328) — **KDD 2025**；提出 ISTS-PLM，通过时间感知与变量感知语言模型统一处理不规则采样时序任务。[代码](https://github.com/usail-hkust/ISTS-PLM)
        - [STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions](https://arxiv.org/abs/2512.04871v1) — **arXiv 2025（预印本）**；将趋势、季节性与残差转为分层语义锚点，为 LLM 预测提供全局与样本级上下文。
        - [Context-Alignment：Activating and Enhancing LLM Capabilities in Time Series](https://arxiv.org/abs/2501.03747) — **ICLR 2025**；通过双尺度图神经网络对齐时序与语言的结构和逻辑，增强 LLM 的上下文理解。[代码](https://github.com/tokaka22/ICLR25-FSCA)
    - Foundation Model
        - [Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts](https://proceedings.mlr.press/v267/liu25an.html) — **ICML 2025**；以稀疏混合专家实现 token 级时序模式专门化。
        - [This Time is Different: An Observability Perspective on Time Series Foundation Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/48ea942b60e6c52b50ca90e4a40d8ac2-Abstract-Conference.html) — **NeurIPS 2025**；提出 Toto 基础模型与面向可观测性数据的 BOOM 基准。
        - [Sundial: A Family of Highly Capable Time Series Foundation Models](https://arxiv.org/abs/2502.00816) — **ICML 2025**；以基于流匹配的 TimeFlow 损失预训练连续值时序模型，支持概率预测。[代码](https://github.com/thuml/Sundial)
        - [TimeFound: A Foundation Model for Time Series Forecasting](https://arxiv.org/abs/2503.04118) — **arXiv 2025（预印本）**；以多分辨率分块和编码器—解码器 Transformer 学习跨领域零样本预测能力。
        - [TimeHF: Billion-Scale Time Series Models Guided by Human Feedback](https://arxiv.org/abs/2501.15942) — **arXiv 2025（预印本）**；结合分块卷积嵌入与时序策略优化，将人类反馈引入十亿参数级时序模型。
        - [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://arxiv.org/abs/2409.16040) — **ICLR 2025**；通过稀疏混合专家按需激活部分网络，扩展时序模型容量并控制推理成本。[代码](https://github.com/Time-MoE/Time-MoE)
        - [ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data](https://arxiv.org/abs/2412.11376) — **AAAI 2025**；将时序视为一种外语，统一数值与文本建模，支持零样本预测和双模态交互。[代码](https://github.com/ForestsKing/ChatTime)
        - [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821) — **arXiv 2025（预印本）**；通过组注意力共享相关序列与协变量信息，统一单变量、多变量及协变量预测。[代码](https://github.com/amazon-science/chronos-forecasting)
        - [TimeDiT: General-purpose Diffusion Transformers for Time Series Foundation Model](https://arxiv.org/abs/2409.02322) — **arXiv 2024（预印本）**；结合扩散采样与 Transformer，以统一掩码机制支持多任务建模和概率生成。
        - [Aurora: Towards Universal Generative Multimodal Time Series Forecasting](https://arxiv.org/abs/2509.22295) — **ICLR 2026**；利用跨领域多模态时序预训练，融合文本领域知识与数值信息进行零样本预测。

    - Tricks for TSLM
        - [Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](https://arxiv.org/abs/2506.14087) — **NeurIPS 2025**；在编码器基础模型微调中显式建模多尺度结构，减少过拟合并增强下游预测。[代码](https://github.com/zqiao11/MSFT)
        - [Improving Time Series Forecasting via Instance-aware Post-hoc Revision](https://arxiv.org/abs/2505.23583) — **NeurIPS 2025**；先识别预测偏差较大的实例，再利用局部协变量和全局历史信息进行事后修正。[代码](https://github.com/icantnamemyself/PIR)
        - [TimeSeriesScientist：A General-Purpose AI Agent for Time Series Analysis](https://arxiv.org/abs/2510.01538) — **arXiv 2025（预印本）**；以数据整理、规划、预测和报告四类 Agent 协作，自动完成时序预测流程。[代码](https://github.com/Y-Research-SBU/TimeSeriesScientist)
        - [DeepAnalyze: Agentic Large Language Models for Autonomous Data Science](https://arxiv.org/abs/2510.16872) — **ICML 2026**；通过课程式智能体训练与数据驱动任务合成，自动完成从原始数据到研究报告的分析流程。[代码](https://github.com/ruc-datalab/DeepAnalyze)
        - [MoiraiAgent: An Agentic Framework for Context-Aware Time-Series Forecasting](https://www.salesforce.com/blog/moiraiagent/) — **Salesforce 技术博客 2026**；结合 LLM、数值预测工具与外部上下文，自动规划预处理、预测和结果修正。[代码](https://github.com/SalesforceAIResearch/uni2ts/tree/main/project/moirai-agent)
        - [Domain-Oriented Time Series Inference Agents for Reasoning and Automated Analysis](https://arxiv.org/abs/2410.04047) — **TMLR 2026**；提出 TS-Reasoner，以领域工具、程序化执行和错误反馈完成多步骤时序推理。


- 2024
    - LLM-based Model
        - [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728) — **ICLR 2024**；通过文本原型重编程与前缀提示，将时序分块接入冻结的 LLM 进行预测。[代码](https://github.com/KimMeen/Time-LLM)
        - [S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting](https://arxiv.org/pdf/2403.05798) — **ICML 2024**；将分解后的时序分块与语言语义锚点对齐，从联合空间检索提示以辅助预测。[代码](https://github.com/panzijie825/S2IP-LLM)
        - [LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting](https://arxiv.org/abs/2410.11674) — **TRL Workshop 2025**；结合多尺度下采样、时序分解与冻结 LLM，同时捕获短期变化和长期趋势。[代码](https://github.com/Kowsher/LLMMixer)
        - [TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment](https://arxiv.org/abs/2406.01638) — **AAAI 2025**；以数值与语言双分支编码及跨模态对齐，融合时序动态和 LLM 表征进行预测。[代码](https://github.com/ChenxiLiu-HNU/TimeCMA)
        - [CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning](https://arxiv.org/abs/2403.07300) — **AAAI 2025**；通过输入匹配、中间特征正则和输出一致性约束，缩小时序与文本的模态差异。[代码](https://github.com/Hank0626/CALF)
    - Foundation Model
        - [TimesFM: A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688) — **ICML 2024**；在大规模时序数据上预训练分块解码器，适配不同历史长度、预测长度与时间粒度。[代码](https://github.com/google-research/timesfm)
        - [Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://arxiv.org/abs/2402.02368) — **ICML 2024**；以统一序列格式和下一 token 预测预训练 GPT 式模型，支持预测、插补与异常检测。[代码](https://github.com/thuml/Large-Time-Series-Model)
        - [Timer-XL: Long-Context Transformers for Unified Time Series Forecasting](https://arxiv.org/abs/2410.04803) — **ICLR 2025**；将多变量预测统一为长上下文下一 token 预测，用 TimeAttention 建模序列内外依赖。[代码](https://github.com/thuml/Timer-XL)
        - [Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series](https://arxiv.org/abs/2401.03955) — **NeurIPS 2024**；以轻量 TSMixer、适应性分块和多层级建模，实现高效零样本与少样本预测。[代码](https://github.com/ibm-granite/granite-tsfm)
        - [MOIRAI: Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592) — **ICML 2024**；在 LOTSA 上统一预训练，处理跨频率、任意变量数与不同分布的通用预测。[代码](https://github.com/SalesforceAIResearch/uni2ts)
        - [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885) — **ICML 2024**；基于 Time Series Pile 进行多数据集预训练，面向预测、分类、异常检测和插补。[代码](https://github.com/moment-timeseries-foundation-model/moment)
        - [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815) — **TMLR 2024**；将缩放和量化后的数值转为离散 token，用语言模型式预训练实现概率预测。[代码](https://github.com/amazon-science/chronos-forecasting)

    - Benchmark
        - [FoundTS: Comprehensive and Unified Benchmarking of Foundation Models for Time Series Forecasting](https://arxiv.org/abs/2410.11802) — **arXiv 2024（预印本）**；统一比较零样本、少样本与全样本预测设置。[中文解读](https://www.xinfinite.net/t/topic/7659)

- 2023
    - LLM-based Model
        - [PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting](https://arxiv.org/abs/2210.08964) — **TKDE 2024（2023 在线发表）**；将数值输入输出转为自然语言提示，把时序预测表述为句子到句子的生成任务。[代码](https://github.com/kuazhangxiaoai/promptcast)
        - [One Fits All:Power General Time Series Analysis by Pretrained LM](https://arxiv.org/abs/2302.11939) — **NeurIPS 2023**；复用预训练语言模型的注意力与前馈层，以少量参数适配多种时序分析任务。[代码](https://github.com/DAMO-DI-ML/One_Fits_All)
        - [TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series](https://arxiv.org/abs/2308.08241) — **ICLR 2024**；通过实例、特征与文本原型对比学习对齐时序嵌入，并以软提示激活冻结 LLM。[代码](https://github.com/SCXsunchenxi/TEST)
        - [LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters](https://arxiv.org/abs/2308.08469) — **ACM TIST 2025**；以时序对齐和预测适配两阶段微调，增强预训练 LLM 的数据效率与多尺度建模。[代码](https://github.com/blacksnail789521/LLM4TS)
        - [Large Language Models Are Zero-Shot Time Series Forecasters](https://arxiv.org/abs/2310.07820) — **NeurIPS 2023**；将时序编码为数字字符串，利用 LLM 的下一 token 分布构建零样本概率预测。[代码](https://github.com/ngruver/llmtime)
        - [TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting](https://arxiv.org/abs/2310.04948) — **ICLR 2024**；将趋势、季节性和残差分解与提示设计结合，学习跨领域可迁移的时序表征。[代码](https://github.com/DC-research/TEMPO)
        - [UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting](https://arxiv.org/abs/2310.09751) — **WWW 2024**；结合领域语言指令、Language-TS Transformer 与掩码训练，统一跨领域时序预测。[代码](https://github.com/liuxu77/UniTime)
    - Foundation Model
        - [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278) — **arXiv 2023（预印本）**；以滞后值为协变量预训练解码器 Transformer，进行单变量零样本概率预测。[代码](https://github.com/time-series-foundation-models/lag-llama)

## 仓库内技术路线

- [Transformer 效率](01_transformer_efficiency.md)
- [序列分解](02_decomposition.md)
- [MLP](03_mlp_revolution.md)
- [Patching](04_patching.md)
- [归一化](05_normalization.md)
- [频域方法](06_frequency_domain.md)
- [基础模型](07_foundation_model.md)
- [Mamba / SSM](08_mamba_ssm.md)
