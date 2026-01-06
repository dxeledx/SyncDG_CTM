# SyncDG-CTM v1.0 设计终稿（用于 BCI Competition IV-2a 跨被试 MI，Strict DG + Early-Exit）

> **已写死的两项选择（按你最后确认）**
> 1) **对齐空间**：仅在 **\(x_\tau=\log(\bar h_\tau+\epsilon)\)** 上做（CDAN/Proto/CORAL/SupCon 等），**不经过 LN**；LN 仅进入分类头。  
> 2) **衰减参数**：\(\gamma\)（能量衰减）与 \(\beta_h\)（若保留同义符号）在第一批实验 **固定不学习**，主线稳定后再做“可学习衰减”增强版。

---

## 0. 任务边界与实验协议（必须写死，避免 reviewer 攻击）

- **数据集**：BCI Competition IV-2a（9 subjects，4 类 MI：L/R/Feet/Tongue，22 通道，250 Hz）。
- **设定**：**Strict Domain Generalization（DG）**  
  - 训练：只用源域（8 个被试）有标签数据。  
  - 测试：目标被试 **任何数据都不可见**（包括统计量、对齐、EA 等）。  
- **评估**：**LOSO（Leave-One-Subject-Out）** 9 折。  
- **调参/早停**：每折在 8 个源域里再做 **leave-one-source-out** 得到 source-val，被试级早停与超参选择 **严禁**使用 target subject。  

---

## 1. 总体模型：CTM 主干 + 尺度不变同步投影 + 多源 DG + 可靠性早退

### 1.1 输入与符号
- 单 trial EEG：\(X\in\mathbb{R}^{C\times T_s}\)，\(C=22\)，\(T_s = f_s\cdot T_{sec}\)。
- Tokenizer 输出：tokens \(E\in\mathbb{R}^{L\times d}\)（\(L\) 个 token，每个维度 \(d\)）。
- CTM internal ticks：\(\tau=1,\dots,T_{\text{thought}}\)，隐藏状态 \(z_\tau\in\mathbb{R}^{D}\)。

### 1.2 结构图（Mermaid）
```mermaid
flowchart LR
  X[EEG Trial X: CxTs] --> Tok[Tokenizer\n(Multi-scale DWConv + Spatial Conv)]
  Tok --> E[tokens E: Lxd]

  subgraph CTM[CTM Core (T_thought ticks)]
    E -->|Cross-Attn Read| Attn[Read: q_tau -> o_tau]
    Z[z_tau] --> Attn
    Attn --> Syn[Synapse Update]
    Syn --> NLM[Neuron-Level Update]
    NLM --> Z
  end

  Z --> Norm[Online Welford\nmean/var + shrinkage]
  Norm --> Proj[Low-rank Projection r_tau = P^T z~_tau]
  Proj --> Acc[h_tau = gamma*h_{tau-1} + r_tau^2]
  Acc --> LenNorm[hbar_tau = h_tau / s_tau]
  LenNorm --> Log[x_tau = log(hbar_tau + eps)]

  Log -->|DG losses| DG[CDAN(one-hot)\n+ Proto EMA\n+ CORAL (cov)\n+ SupCon margin]
  Log -->|LN only for clf| LN[LN(x_tau)]
  LN --> Cls[Classifier Head -> logits p_tau]

  subgraph Exit[Early-Exit]
    Cls --> Stat[u_tau stats\n(entropy, margin, KL, ...)]
    Stat --> g[reliability predictor g]
    g --> wpred[w_pred over ticks]
    wpred --> Agg[p_agg]
    g --> Rule[exit rule]
  end

  Agg --> yhat[Prediction]
```

---

## 2. Tokenizer（EEG→tokens）：轻量、可复现、兼顾时空解耦

### 2.1 预处理（建议与 MOABB/MNE 一致）
- 参考窗口：**2–6 s**（常用 MI 有效区间）。  
- 滤波：\(4\sim 38\) Hz（或 8–30 Hz 作为消融）。  
- 重参考：CAR。  
- 标准化：**train-set** 统计（channel-wise z-score），禁止用 target 统计。  

### 2.2 Tokenizer 架构（推荐 v1 默认）
目标：把“真实时间序列”编码成 \(L\) 个片段 token，后续 CTM 在 token 上做“反复读取/证据累积”。

- **Temporal DWConv（多尺度）**：
  - 分支 k∈{(k1,k2,k3)}：1D depthwise conv（沿时间），kernel 对应约 0.25s / 0.5s / 1s
  - 输出拼接后 pointwise 1×1 conv 混合。
- **Spatial Conv**：
  - depthwise + pointwise，或 1×C 卷积模拟空间滤波（类似 CSP/EEGNet 空间卷积）。
- **Tokenization**：
  - 时间维平均池化/步长卷积得到 \(L\) 个 token（默认 L=16）。
- 输出维度：\(d=64\)。

> **消融**：  
> - 去掉 multi-scale（只留一个 kernel）  
> - 去掉 spatial conv（只做 temporal）  
> - 改 L∈{8,16,32}

---

## 3. CTM 主干（EEG 版适配）：保留“internal thinking”的机制，不照抄昂贵实现

CTM 核心思想：每个 tick 用“同步表征”产生 query 去读取外部 tokens，再更新内部状态，循环多步形成“思考轨迹”。（CTM 里同步矩阵 \(S^t=Z^t(Z^t)^\top\) 是关键，但 full \(D^2\) 很贵，所以我们后面用低秩同步投影替代。）

### 3.1 Cross-Attention Read
- query：来自同步 head 的 action 分支（见 4.4）或简化版来自 \(x_\tau\)。
- key/value：来自 tokens \(E\) 的线性投影。
- 输出：\(o_\tau\in\mathbb{R}^{d}\)。

### 3.2 Synapse Update（轻量）
- 输入：\([z_\tau; o_\tau]\)。
- 输出：pre-activation \(a_\tau \in\mathbb{R}^{D}\) 或增量 \(\Delta z\)。
- 推荐：2 层 MLP + gating（GLU/SiLU）：
\[
\Delta z_\tau = W_2\,\sigma(W_1[z_\tau;o_\tau]),\quad z'_\tau = z_\tau + \Delta z_\tau
\]

### 3.3 Neuron-Level Update（参数受控）
CTM 原论文有“每个 neuron 私有 NLM”，参数很大。EEG 小数据下建议：
- **共享 NLM + neuron embedding**（FiLM）：  
  - 每个 neuron i 有 embedding \(e_i\)；
  - NLM 共享权重：\(z_{\tau+1,i}=g_\theta(\text{hist}(a_{\tau,i}), e_i)\)。
- hist 长度 \(M\)（默认 8）。

---

## 4. 核心创新：尺度不变 + 低秩同步投影能量（可训练、可 early-exit）

> 目标：把 CTM 的同步思想落地为 **O(D·D_o)** 的可微二阶表征，并对“跨被试幅值/偏置”做严格抑制。

### 4.1 Online Welford：trial 内逐 tick 归一（支持 early-exit）
对每个 sample、每个维度 i（neuron）在线更新：
- 初始化：\(m_{0,i}=0,\; M2_{0,i}=0\)
- tick \(\tau\)：
\[
\delta_{\tau,i}=z_{\tau,i}-m_{\tau-1,i},\quad 
m_{\tau,i}=m_{\tau-1,i}+\delta_{\tau,i}/\tau
\]
\[
\delta'_{\tau,i}=z_{\tau,i}-m_{\tau,i},\quad
M2_{\tau,i}=M2_{\tau-1,i}+\delta_{\tau,i}\delta'_{\tau,i}
\]
\[
v_{\tau,i}=\frac{M2_{\tau,i}}{\max(\tau-1,1)}
\]

**Shrinkage（小 T 稳定方差）**：
\[
\bar v_{\tau}=\frac1D\sum_i v_{\tau,i},\quad
\tilde v_{\tau,i}=(1-\alpha)v_{\tau,i}+\alpha \bar v_\tau
\]
默认：\(\alpha=0.1\)（可扫 0.05/0.1/0.2）。

得到尺度不变标准化：
\[
\tilde z_{\tau,i}=\frac{z_{\tau,i}-m_{\tau,i}}{\sqrt{\tilde v_{\tau,i}+\epsilon}}
\]

> **必须消融**：online vs offline（用全 T 统计）  
> 预期：online 允许 early-exit 且对齐稳定；offline 会与 early-exit 冲突。

### 4.2 低秩同步投影能量（避免显式 \(D^2\)）
定义投影矩阵 \(P\in\mathbb{R}^{D\times D_o}\)（\(D_o\ll D\)，默认 32）：
\[
r_\tau = P^\top \tilde z_\tau \in \mathbb{R}^{D_o}
\]
能量累积（固定衰减 \(\gamma\)）：
\[
h_\tau = \gamma h_{\tau-1} + r_\tau\odot r_\tau,\quad h_0=0
\]

### 4.3 有效长度归一化（解决“tick 越多越大”与 early-exit 分布漂移）
\[
s_\tau=\sum_{k=0}^{\tau-1}\gamma^k=\frac{1-\gamma^\tau}{1-\gamma},\quad
\bar h_\tau = \frac{h_\tau}{s_\tau}
\]
这样 \(\bar h_\tau\) 在不同 \(\tau\) 下尺度可比（关键用于早退与对齐）。

### 4.4 对齐空间（写死）：log 空间，不经 LN
\[
x_\tau = \log(\bar h_\tau + \epsilon_h)
\]
- **所有 DG 对齐损失**在 \(x_{\tau^*}\) 上（随机截断 tick，见 6.2）。
- **分类头**使用 \( \text{LN}(x_\tau) \)（LN 不进入对齐）。

### 4.5 复杂度（实现级别）
- 投影：\(P^\top \tilde z_\tau\) FLOPs \(\approx 2DD_o\)  
- elementwise square + 累积：\(\approx 3D_o\)  
- 总同步 head：\(\mathcal{O}(T_{\text{thought}}DD_o)\)  
默认：\(D=128,D_o=32,T=8\Rightarrow\) head 部分 ~ 65k 乘加/样本级（极轻）。

---

## 5. 反塌缩（anti-collapse）：tick 间 Gram 去相关（可微、非零梯度）

目标：避免“所有 tick 快速收敛到同一吸引子”导致 internal thinking 失效。

对 batch 中每个样本，将 ticks 表征堆叠：
\[
U = [\text{LN}(z_1),\dots,\text{LN}(z_T)]\in\mathbb{R}^{T\times D}
\]
计算 Gram：
\[
G=\frac{1}{D}UU^\top\in\mathbb{R}^{T\times T}
\]
去相关损失（Barlow Twins 风格）：
\[
\mathcal{L}_{div}=\sum_{\tau\neq\tau'} G_{\tau,\tau'}^2
\]
> 诊断图：tick-wise cosine similarity 分布；正确/错误样本的 \(G\) 收敛差异。

---

## 6. DG 组件：在 log-space 同步表征上做“类可分 + 域不可辨”

### 6.1 对齐发生在哪个 tick？
为匹配 early-exit 的“随机停时”，训练时每个样本采样：
\[
\tau^*\sim \text{Unif}\{1,\dots,T_{\text{thought}}\}
\]
并只在 \(x_{\tau^*}\) 上计算 DG losses（adv/proto/coral/supcon）。  
> 必做对比：固定 \(\tau^*=T\) vs 随机 \(\tau^*\)（收敛、seed 方差、worst-subject）。

### 6.2 CDAN(one-hot)（条件对抗）
域判别器 \(D_\phi\) 输出 subject-id（训练域的 8 类）。
- 条件特征（source 有真标签）：
\[
g = x_{\tau^*}\otimes y_{onehot}
\]
对抗损失（多类交叉熵）：
\[
\mathcal{L}_{adv} = \mathbb{E}\big[\text{CE}(D_\phi(g), d)\big]
\]
特征提取器通过 GRL 最大化该损失实现去域。

**关键：batch 采样必须 per-domain per-class 均衡**，避免域判别器走“类别先验捷径”。

### 6.3 原型对齐（EMA memory，按域/按类）
每个训练域 d、每个类别 c 维护原型 \(\mu^{(d)}_c\)：
\[
\mu^{(d)}_c \leftarrow (1-\eta)\mu^{(d)}_c + \eta\cdot \text{mean}\{x_{\tau^*}^{(i)}: y_i=c, d_i=d\}
\]
全局原型 \(\mu_c = \frac{1}{|D_s|}\sum_d \mu^{(d)}_c\)。

原型对齐：
\[
\mathcal{L}_{proto}=\sum_{d}\sum_{c}\|\mu^{(d)}_c-\mu_c\|_2^2
\]

### 6.4 SupCon / margin（抵消对齐导致的类塌缩）
在 \(x_{\tau^*}\) 上做 supervised contrastive：
\[
\mathcal{L}_{supcon}=-\sum_i \frac{1}{|P(i)|}\sum_{p\in P(i)} 
\log \frac{\exp(\text{sim}(x_i,x_p)/\tau)}
{\sum_{a\neq i}\exp(\text{sim}(x_i,x_a)/\tau)}
\]
> 小 batch EEG 风险：建议 **projection head**（2 层 MLP）仅用于 supcon；可选 stop-grad 到主干做消融。

### 6.5 CORAL（对齐二阶矩，log-space，且不经 LN）
对每个域 d，在 batch 或 EMA 中估计协方差 \(C^{(d)}\)（带 shrinkage）：
\[
C^{(d)}\leftarrow (1-\zeta)C^{(d)} + \zeta\cdot \widehat{\text{Cov}}(x_{\tau^*}|d) + \lambda I
\]
\[
\mathcal{L}_{coral}= \sum_d \|C^{(d)}-\bar C\|_F^2,\quad \bar C=\frac{1}{|D_s|}\sum_d C^{(d)}
\]
默认：\(\lambda=10^{-3}\)，\(\zeta=0.01\)。

---

## 7. 分类头 + 可靠性教师蒸馏 + early-exit

### 7.1 分类头（LN 只进分类，不进对齐）
\[
\ell_\tau = W\,\text{LN}(x_\tau)+b,\quad p_\tau=\text{softmax}(\ell_\tau)
\]

### 7.2 可靠性预测器 g（只读无标签统计，低容量）
输入 \(u_\tau\)（全部无需标签）建议：
- Entropy：\(H(p_\tau)\)
- Margin：\(m_\tau = p_\tau^{(1)}-p_\tau^{(2)}\)
- KL 演化：\(\text{KL}(p_\tau\|p_{\tau-1})\)
- \(\|x_\tau\|_1\)、\(\Delta H = H(p_{\tau-1})-H(p_\tau)\)

g 输出标量 \(q_\tau\in(0,1)\)，用 softmax 形成权重：
\[
w_\tau^{pred} = \frac{\exp(q_\tau/\tau_g)}{\sum_{k=1}^T \exp(q_k/\tau_g)}
\]

### 7.3 教师权重（label-dependent，仅训练 g，不回传 backbone）
教师用每 tick 的 CE：
\[
L_\tau = \text{CE}(p_\tau,y)
\]
教师权重：
\[
w_\tau^{teach} = \text{softmax}\big(-\text{sg}(L_\tau)/\tau_L\big)
\]
蒸馏损失（只更新 g）：
\[
\mathcal{L}_{teach}=\text{KL}(w^{teach}\,\|\, w^{pred})
\]
**写死：\(\mathcal{L}_{teach}\) 不回传到主干**（detach \(x_\tau,p_\tau\) 进入 g 的分支，避免“为可预测权重而牺牲分类”捷径）。

### 7.4 训练分类损失（与推理一致：用 w_pred）
聚合预测：
\[
p_{agg}=\sum_{\tau=1}^T w_\tau^{pred}p_\tau
\]
分类主损失：
\[
\mathcal{L}_{cls}=\text{CE}(p_{agg},y)
\]
> 这样 train/test 机制一致；教师只用于训练 g。

### 7.5 Early-exit（可发表的最小化期望代价形式）
定义退出 tick：
\[
\tau_{exit}=\min\{\tau:\ q_\tau>\delta\ \land\ m_\tau>\delta_m\ \land\ \text{KL}(p_\tau\|p_{\tau-1})<\epsilon_{kl}\}
\]
若未满足，\(\tau_{exit}=T\)。

并把它写成目标：
\[
\min\ \mathbb{E}[\ell(y,\hat y_{\tau_{exit}})] + \lambda_{cost}\mathbb{E}[\tau_{exit}]
\]
其中 \(\lambda_{cost}\) 可由“延迟预算/ITR 约束”在 source-val 上选取。

---

## 8. 总损失与权重调度（避免负迁移与类塌缩）

总目标：
\[
\mathcal{L}=
\mathcal{L}_{cls}
+\lambda_{teach}\mathcal{L}_{teach}
+\lambda_{div}\mathcal{L}_{div}
+\lambda_{orth}\mathcal{L}_{orth}
+\lambda_{adv}\mathcal{L}_{adv}
+\lambda_{proto}\mathcal{L}_{proto}
+\lambda_{coral}\mathcal{L}_{coral}
+\lambda_{supcon}\mathcal{L}_{supcon}
\]

- 正交正则：
\[
\mathcal{L}_{orth}=\|P^\top P-I\|_F^2
\]

### 8.1 两阶段 + 平滑 ramp（推荐）
- **Warmup（前 E_w epochs）**：仅 \(\mathcal{L}_{cls}+\lambda_{div}\mathcal{L}_{div}+\lambda_{orth}\mathcal{L}_{orth}\)  
- **DG ramp（E_w 之后）**：\(\lambda_{adv},\lambda_{proto},\lambda_{coral}\) 按 DANN 经典 schedule 从 0→max。

推荐超参（起跑值）：
- \(T_{\text{thought}}=8\)，\(D=128\)，\(D_o=32\)，\(\gamma=0.9\)
- \(\lambda_{div}=0.01\)，\(\lambda_{orth}=0.1\)
- \(\lambda_{adv}=0.5\)，\(\lambda_{proto}=0.1\)，\(\lambda_{coral}=0.05\)，\(\lambda_{supcon}=0.1\)
- \(\lambda_{teach}=1.0\)

> **梯度平衡（可选增强）**：GradNorm/uncertainty-weighting；v1 可先手调并记录各项 gradient norm 作为诊断。

---

## 9. Batch 采样（必须严格 per-domain per-class 平衡）

每个 batch 选 P 个训练被试（域），每个域每类抽 K 个 trial：
- batch size \(B = P\cdot 4K\)
- 推荐：\(P=6, K=2 \Rightarrow B=48\)（稳定 proto/coral 统计且显存友好）
- 备选：\(P\in\{4,6,8\}, K\in\{1,2,3\}\)

---

## 10. 训练细节（可复现）

- Optimizer：AdamW  
  - lr = 1e-3（tokenizer/CTM），head/g 可 2e-3  
  - weight decay = 1e-2  
- Scheduler：cosine + warmup 10 epochs  
- Gradient clip：1.0  
- Epoch：200（早停 patience 30）  
- Seed：至少 5 个；同时报告 **subject 维度方差** 与 seed 方差。

---

## 11. 评估指标与统计（一区必备）

### 11.1 主指标
- Mean Acc（LOSO 平均）
- Cohen’s \(\kappa\)
- Macro-F1
- worst-subject Acc（9 被试最差）
- per-subject 方差（鲁棒性核心）
- 校准：ECE（按 tick 也要画）、NLL/Brier（至少一个）

### 11.2 early-exit 指标
- 平均 \(\tau_{exit}\)、95% 分位 \(\tau_{exit}\)
- Acc–Latency 曲线（按 \(\lambda_{cost}\) 或阈值扫）
- ITR：明确假设（若只减少 compute，不减少 EEG 采集窗，则 ITR 主要受推理延迟影响）

### 11.3 显著性检验
- 9 折是小样本：Wilcoxon signed-rank 或 permutation test（被试维度配对），多重比较 Holm-Bonferroni。

---

## 12. Baseline（分组，避免 apples-to-oranges）

### 12.1 Strict DG（同设定）
- EEGNet、ShallowConvNet/DeepConvNet
- EEGConformer / CTNet / TCFormer（同预处理、同窗口、同调参流程复现）
- Riemannian：Tangent Space + LR、MDM、Shrinkage TS + LR（**不使用 Euclidean Alignment**）

### 12.2 External pretraining（单独组）
- EEGPT linear-probe / finetune（标注“外部预训练”）

### 12.3 UDA/DA（单独组，不与 DG 主结论混写）
- MI-CAT 等（允许看 target 无标签/有标签）

---

## 13. 必做消融矩阵（回答“CTM vs 对齐正则”的贡献）

至少 6 大部件（按你清单）：
1) Tokenizer：多尺度 vs 单尺度；有/无 spatial conv  
2) CTM ticks：T=1 vs 4 vs 8/16  
3) 同步 head：SIP（本方法） vs 普通 pooled embedding（mean/max/attn pooling）  
4) 衰减与有效长度归一：有/无 \(\gamma\)、有/无 \(s_\tau\)  
5) 置信度聚合与 early-exit：无 g（固定权重）vs g（蒸馏）  
6) DG losses：adv/proto/coral/supcon 分别加/去  
+ 额外：anti-collapse（\(\mathcal L_{div}\)）有/无；log 变换有/无；对齐层（x vs LN(x)）

---

## 14. 机制与诊断实验（必须把“拍脑袋”变可验证）

### 14.1 信息保真（对应你 Q19）
- 取每个 trial 的 \(m_T,v_T\) 向量训练线性分类器（LOSO）→ 若可判别，证明它是“域伪特征”还是“被 x 重编码”。

### 14.2 域不可辨识性（domain probe）
- 在 token-level、tick-level \(x_\tau\)、聚合 \(x_{\tau^*}\) 上分别训练外部域探测器  
- sanity：在无对抗版本上 probe 必须显著 > 随机；有对抗后应下降。

### 14.3 ticks 证据累积
- KL(p_\tau||p_{\tau-1})、entropy、margin 随 \(\tau\) 曲线：正确样本应更快收敛  
- domain-probe AUC 随 \(\tau\) 下降：若成立，说明“越思考越去域”。

### 14.4 合成域偏移（对应你 Q24/18）
对测试 EEG 施加扰动，验证 log-space 对齐与尺度不变假设：
- 乘性缩放：\(X' = aX\)（a∈[0.5,2]）  
- 加性偏置：\(X' = X + b\)（b 为通道常量）  
- 通道混合：\(X' = RX\)（R 为随机近正交矩阵）  
报告：Acc/κ、domain probe、\(x\) 分布距离（MMD/FD）。

### 14.5 鲁棒性压力测试（对应你 D3）
- 通道 dropout（丢 10–30%）  
- 噪声注入（1/f、EMG 伪迹）  
预期：尺度不变同步 + log-space 对齐更稳。

---

## 15. 预期性能目标（现实可兑现、可写进论文）
在 strict LOSO（IV-2a）同 pipeline 下，目标写成“相对提升”更可信：
- 相对最强非预训练 baseline（如 EEGConformer/CTNet/TCFormer）**绝对提升 ≥ 2–4% Acc**  
- worst-subject Acc 提升 ≥ 3%  
- 在相近 Acc 下，平均 \(\tau_{exit}\) 降低 20–40%（更好延迟/能耗）

> 如果 EEGPT finetune 更强：主战场转向 **strict DG + 鲁棒性 + Acc–Latency 曲线**。

---

## 16. 最终算法伪码（可直接照着实现）

**Train (one LOSO fold)**  
1. 构造 source_train（8 subjects）与 source_val（从 source 中留 1 subject）。  
2. 迭代 epoch：  
   - 采样 batch：P domains × 4 classes × K samples（均衡）  
   - 前向：
     - tokens \(E\leftarrow\) Tokenizer(X)
     - for τ=1..T:
       - CTM read: \(o_\tau\leftarrow\) Attn(q_\tau(E,z_\tau), E)
       - CTM update: \(z_{\tau+1}\leftarrow\) Synapse+NLM(z_\tau,o_\tau)
       - online Welford: 更新 \(m_\tau,v_\tau\)；得到 \(\tilde z_\tau\)
       - \(r_\tau=P^\top \tilde z_\tau\)
       - \(h_\tau=\gamma h_{\tau-1}+r_\tau^2\)；\(\bar h_\tau=h_\tau/s_\tau\)
       - \(x_\tau=\log(\bar h_\tau+\epsilon)\)
       - logits \(p_\tau=\text{softmax}(W\,\text{LN}(x_\tau)+b)\)
       - 计算 \(u_\tau\)（entropy/margin/KL/…）
     - g 分支（detach backbone 输出）：得到 \(w^{pred}\)
     - 聚合 \(p_{agg}=\sum_\tau w_\tau^{pred}p_\tau\)
   - 计算损失：
     - \(L_{cls}=\text{CE}(p_{agg},y)\)
     - 采样 \(\tau^*\)，在 \(x_{\tau^*}\) 上计算 \(L_{adv},L_{proto},L_{coral},L_{supcon}\)
     - \(L_{div}\)（ticks Gram 去相关），\(L_{orth}\)
     - 教师 \(w^{teach}\)（用 \(L_\tau\)）→ 只更新 g：\(L_{teach}\)
   - 反向更新：
     - 主干更新：除 \(L_{teach}\) 外全部
     - g 更新：只用 \(L_{teach}\)

**Inference**  
for τ=1..T:
- 计算 \(p_\tau,u_\tau,q_\tau\)
- 若满足 exit rule → 输出 \(p_\tau\) 或累计 \(p_{agg}\) 的当前估计。

---

## 17. v1→v2 增强路线（等主线站住再上）
- 学习 \(\gamma\)（加上 \(\gamma\le 1-\delta\) 约束/正则）
- 更强的“相位增强同步”（Hilbert/简化 cross-spectrum）
- 轻量 test-time unlabeled 统计（作为 **非 strict DG** 扩展设定单独报告）

