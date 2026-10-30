# CALIBRATING ATTENTION-BASED TOKEN PRUNING FOR EFFICIENT MULTI-IMAGE SYNTHESIS IN UMMS

Zimeng Wu<sup>1,2,§</sup> Qijian Tian<sup>3</sup> Penghui Du<sup>2</sup> Changqian Yu<sup>2,‡</sup> Jiaxin Chen<sup>1,†</sup> Yunhong Wang<sup>1,†</sup> <sup>1</sup>Beihang University <sup>2</sup>KlingAI Research <sup>3</sup>Shanghai Jiao Tong University

## ABSTRACT

Unified Multimodal Models (UMMs) enable powerful multi-image synthesis but incur substantial computation from long visual contexts. While token pruning offers a direct route to accelerating inference, existing methods are primarily adapted from understanding tasks, suffering from insufficient reference image exploration, confused conditional guidance, and pooling-induced ambiguity that degrades generation quality. To address these issues, we propose CATP, the first training-free context-compression method tailored to UMM generation, which Calibrates the discrepancies of general Attention-based Token Pruning for better quality-efficiency trade-off. Concretely, building on independent layer-wise pruning during denoising, we introduce Guidance-Consistent Pruning (GCP) to synchronize pruning decisions across conditioning branches, and a Taylor-Calibrated Attention Metric (TCAM) that derives a theoretically grounded second-order correction for pooling-induced bias. In addition, a Fast Scoring Operator (FSO) fuses the calibration into tiled online Softmax to reduce pruning cost. Extensive experiments across representative UMMs and benchmarks demonstrate that CATP consistently outperforms state-of-the-art token-reduction methods, achieving up to 2.09× attention-core speedup. Code will be released upon paper acceptance.

## 1 INTRODUCTION

Unified Multimodal Models (UMMs), which integrate understanding and generation within a shared backbone (Deng et al., 2025; Diao et al., 2026), have emerged as a prevalent paradigm for enhancing both capabilities (Zhao et al., 2025). Leveraging their strength in handling complex inputs, multiimage synthesis, which generates a target image from multiple references (Chen et al., 2026; Wei et al., 2026), has recently attracted growing interest in broad applications including fashion design (Choi et al., 2024), personalized content production (Wang et al., 2025), etc. However, this typically requires concatenating visual tokens from multiple high-resolution images, yielding long sequences that incur substantial attention computation during denoising (Chen et al., 2026; He et al., 2025).

In general, such long-context designs are originally intended to support diverse capabilities of foun dation models (Deng et al., 2025), whereas a specific task may exploit only a subset of the available information (Zhang et al., 2025b). Therefore, reducing the effective sequence length is a direct route to improving inference efficiency (Bolya et al., 2023; Chen et al., 2024b). Among strategies, attention-based token pruning offers an effective and general baseline, leveraging native attention score to discard less informative tokens (Yao et al., 2026; Wu et al., 2026b). Nevertheless, existing schemes are primarily developed for understanding-oriented models (Chen et al., 2024b; Wu et al., 2026a), while the few direct adaptations to UMMs overlook generation-specific characteristics (Ke et al., 2026; Chen et al., 2026), potentially discarding visual regions critical to faithful synthesis.

Specifically, we identify three key limitations: 1) Static pruning across timesteps and layers. Understanding tasks usually produce short responses (Liu et al., 2024), with most information aggregated into a few tokens at early layers (Wu et al., 2026a), making one-shot pruning throughout inference feasible (Chen et al., 2024b). In contrast, visual generation attends to complementary regions across denoising timesteps and layers, as in Fig. 1(a). Especially under aggressive token budgets, a fixed pruning fails to cover the full reference content, yielding insufficient exploration of visual cues. 2) Inconsistent compression across conditioning branches. Classifier-Free Guidance (CFG) (Ho & Salimans, 2021), natively instantiated in UMMs (Deng et al., 2025), performs multiple forward passes with different input sequences, naturally producing branch-specific attention scores and motivating independent pruning. However, as shown in Fig. 1(b), this yields mismatched token selections whose symmetric difference carries useless information yet is amplified in the CFG guidance residual, perturbing guided velocity prediction and compromising generation fidelity. 3) Ambiguous token selection under pooled query representations. Adapting pruning to varying context exploration requires frequent importance estimation, while exact attention-based metric with the still-lengthy denoising queries incurs substantial overhead, undermining the goal of acceleration. Average pooling on queries is commonly adopted to reduce this cost (Zhang et al., 2025a). However, as observed in Fig. 1(c), it smooths fine-grained attention responses to critical regions, obscuring their distinction from less relevant content and thus yielding less reliable context retention.

![](images/1f4d8f7fccb08f857d5eec6638412a01bcffe36afd9dd729088a8e9f96d878af.jpg)  
Figure 1: Empirical observations on Bagel using the general attention metric. (a) Top-10% tokens at timesteps 1/31 across layers 9/18/27, with their union in the last column. (b) Top-10% tokens from the (Left) full-condition and (Middle) image-only branches, with (Right) their symmetric difference; red boxes indicate branch-specific tokens. (c) (Left) Reference images with manually selected synthesis-critical regions highlighted in red boxes. (Middle/Right) Heatmaps of the maximum attention from generated-image locations to these regions under dense and pooled attention, respectively.

To address these issues, we propose CATP, a novel training-free framework for efficient multiimage synthesis in UMMs. Starting from the general attention-based token pruning, CATP identifies generation-specific discrepancies and calibrates them at both conditioning-branch and importanceestimation levels, thereby improving generation fidelity. Specifically, CATP prunes reference-image tokens in the KV cache across all layers at predefined timesteps, adapting compression to varying reference explorations. Building on this, we introduce Guidance-Consistent Pruning (GCP), which formulates the cross-branch mismatch of independent compression and synchronizes pruning decisions with the full-condition branch to preserve clean guidance. A second-order Taylor expansion then characterizes the bias of attention metric with pooled-query, leading to a Taylor-Calibrated Attention Metric (TCAM) for more reliable identification of critical reference tokens. Additionally, a Fast Scoring Operator (FSO) is developed with a custom Triton kernel. It integrates the calibration into tiled online Softmax computation, substantially reducing the scoring overhead.

In summary, our main contributions are three-fold:

1) We analyze discrepancies in general attention-based token pruning for UMM multi-image synthesis and propose a training-free acceleration framework, dubbed CATP. To the best of our knowledge, CATP makes the first investigation of token pruning specifically tailored to UMM generation.

2) We develop Guidance-Consistent Pruning (GCP), Taylor-Calibrated Attention Metric (TCAM) and Fast Scoring Operator (FSO), which address the rigidity, inconsistency, and ambiguity of general schemes, jointly enabling a flexible, accurate and efficient online context compression.

3) We conduct extensive experiments on representative UMMs and benchmarks, showing that CATP delivers superior generation quality, with up to 2.09× attention-core acceleration.

## 2 RELATED WORK

## 2.1 UNIFIED MULTIMODAL MODELS AND MULTI-IMAGE SYNTHESIS

Unified Multimodal Models (UMMs) enable understanding and generation within a single architecture (Zhao et al., 2025). Recent prevalent paradigm adopts a Large Language Model (LLM) as the backbone (Zhao et al., 2025), where text generation remains autoregressive process, while images are synthesized via iterative flow matching in latent (Deng et al., 2025) or pixel space (Diao et al., 2026). For image generation, conditioning inputs are encoded and concatenated into a token sequence, cached as fixed KV states, and repeatedly attended to by generated visual tokens throughout denoising (Deng et al., 2025; Cao et al., 2025). Multi-image synthesis leverages UMMs’ strengths in complex instruction understanding and high-quality generation (Wei et al., 2026; Oshima et al., 2026). However, multiple reference images substantially lengthen the input sequence, underscoring the practical necessity of efficient inference, especially in attention computation (Chen et al., 2026).

## 2.2 EFFICIENT INFERENCE VIA TOKEN REDUCTION

Token Pruning. Token pruning has presented clear efficiency gains for long-context tasks (Yao et al., 2026), exploiting the task-irrelevant redundancy inherent in input sequences (Wu et al., 2026a). Initially explored in understanding tasks, methods develop various criteria for token selection (Yang et al., 2025; Alvar et al., 2025). Among them, attention-based pruning offers a strong baseline across LLMs (Wu et al., 2026a), Multimodal Large Language Models (MLLMs) (Chen et al., 2024b), and the understanding branch of UMM (Ke et al., 2026), accelerating either prefilling through early token removal (Chen et al., 2024b) or decoding via KV cache pruning (Wan et al., 2024).

Token Reduction for Visual Generation. Efficiency in visual generation has been studied from multiple perspectives, such as timestep reduction (Fan et al., 2026) and cross-step cache reusing (Liu et al., 2025; Zhao et al., 2026), while token reduction is orthogonal to them. Most methods originally target Diffusion Transformers (DiTs) (Peebles & Xie, 2023), typically reducing the computed sequence by merging and subsequently unmerging denoising tokens (Bolya & Hoffman, 2023; Fang et al., 2025). ToPi (Lin et al., 2026) more closely follows the aforementioned token pruning paradigm, but shares pruning decisions across layers, overlooking layer-wise differences in information demand. The few attempts on UMMs largely transfer understanding-oriented techniques (Chen et al., 2026), neglecting generation-specific characteristics. In this work, we systematically analyze these distinctive properties and develop token pruning specifically tailored to UMM generation.

## 3 METHODOLOGY

## 3.1 FRAMEWORK OVERVIEW

In context of mainstream UMMs, multiple reference images and an instruction are encoded into visual and text token sequences I and ${ \mathbf { } } ^ { T } ;$ , respectively, and concatenated to form a context $C = [ I ; T ]$ of length $N$ . Prefilling $\bar { C }$ through the L-layer, θ-parameterized backbone $f _ { \theta } ( \cdot )$ produces layer-wise representations $\{ C ^ { l } \} _ { l = 1 } ^ { \breve { L } }$ , cached as fixed KV states. Starting from noisy visual tokens $Z _ { T }$ of length $\bar { M }$ , generation proceeds for T iterative flow-based denoising steps as Eq. (1), where the intermediate visual states $\boldsymbol { Z } _ { t } ^ { \dot { l } }$ attend to $C ^ { l }$ within the attention blocks at each layer, and $Z _ { 0 }$ is ultimately decoded into the output image.

$$
Z _ { t - 1 } = f _ { \theta } ( Z _ { t } ; t , \{ C ^ { l } \} _ { l = 1 } ^ { L } ) , t = T , \dots , 1 .\tag{1}
$$

Token pruning shortens $C ^ { l }$ at a predefined timestep $t ^ { \mathrm { p } }$ , reducing attention cost in the current and subsequent denoising steps. Motivated by the dynamic reference exploration discussed in Sec. 1, we adopt the common independent layer-wise pruning (Wan et al., 2024) as the baseline framework. Specifically, each token $C _ { j } ^ { l }$ is assigned an importance score $s _ { j } ^ { l }$ , and the top-B candidates are retained accordingly as:

$$
\hat { C } ^ { t \leq t ^ { \mathrm { p } } , l } = \{ C _ { j } ^ { l } | j \in \mathrm { T o p K } ( \{ s _ { j } ^ { l } \} _ { j = 1 } ^ { N } , B ) \} .\tag{2}
$$

As established in prior work (Lin et al., 2026), the attention from denoising visual tokens to reference tokens effectively reflects the native information flow during image synthesis, providing a principled criterion for token pruning. Formally, let $\pmb { Q } = \{ \pmb q _ { m } \} _ { m = 1 } ^ { M }$ denote the queries projected from $ { \boldsymbol { Z } } _ { t ^ { \mathrm { p } } } ^ { l }$ , and ${ \pmb K } = \{ k _ { j } \} _ { j = 1 } ^ { N }$ the keys of $C ^ { l }$ , with the superscript l omitted for clarity. Accordingly, the baseline importance score $s _ { j } ^ { l }$ is defined as the average attention received by token j over all queries:

$$
s _ { j } ^ { l } = \frac { 1 } { M } \sum _ { \pmb { q } \in \pmb { Q } } \mathrm { A t t n } ( \pmb { q } ) _ { j } , \quad \mathrm { A t t n } ( \pmb { q } ) = \mathrm { S o f t m a x } \left( \frac { \pmb { q } K ^ { \top } } { \sqrt { d } } \right) \in \mathbb { R } ^ { N } ,\tag{3}
$$

where d denotes head dimension, and we omit the head-wise averaging from the notation for brevity.

Building upon this general framework, CATP focuses on accelerating the computation-dominant denoising process, as illustrated in Fig. 2. By further investigating the aforementioned limitations of vanilla attention-based pruning, we develop GCP to determine $\hat { C } ^ { t \leq t ^ { \mathrm { p } } , l }$ <sup>l</sup> across condition branches and TCAM with FSO to refine $s _ { j } ^ { l ^ { \prime } }$ , thereby enabling more accurate and efficient context compression.

![](images/9c6844298778dc5d499a17a72586725bc513a69d484abbb6a195d18972fa6fcb.jpg)  
Figure 2: Framework overview. (Top) UMM inference pipeline with CATP. It accelerates denoising by pruning visual KV cache tokens across layers at predefined timesteps. (Bottom) Illustration of the main modules. (a) GCP synchronizes pruning decisions from the full-condition to imageonly branch, avoiding condition confusion from independent pruning. (b) TCAM estimates token importance using pooled-query attention and calibrates it with a theoretically grounded second-order term. (c) FSO efficiently realizes calibrated scoring via a two-pass tiled online-Softmax kernel.

## 3.2 GUIDANCE-CONSISTENT PRUNING

The native CFG mechanism (Ho & Salimans, 2021) instantiates Eq. (1) into three condition branches, each performing a complete forward pass under full, image-only or text-only conditioning:

$$
V _ { t } ^ { \mathrm { f u l l } } = v _ { \theta } ( Z _ { t } , t ; I , T ) , \quad V _ { t } ^ { \mathrm { i m g } } = v _ { \theta } ( Z _ { t } , t ; I , \mathcal { O } ) , \quad V _ { t } ^ { \mathrm { t e x t } } = v _ { \theta } ( Z _ { t } , t ; \mathcal { O } , T ) ,\tag{4}
$$

where V denotes the velocity field, and $v _ { \theta } ( \cdot )$ is the velocity predictor of $f _ { \theta } ( \cdot )$ . This yields modalityspecific guidance residuals $G _ { t } ^ { \mathrm { i m g } } { = } V _ { t } ^ { \mathrm { f u l l } } { - } V _ { t } ^ { \mathrm { t e x t } }$ and $G _ { t } ^ { \mathrm { t e x t } } { = } V _ { t } ^ { \mathrm { f u l l } } { - } V _ { t } ^ { \mathrm { i m g } }$ , capturing the incremental contribution of each modality. Accordingly, the guided velocity can be equivalently formulated as $V _ { t } ^ { \mathrm { c f g } }$ , controlled by model-inherent weights $\lambda ^ { \mathrm { t e x t } }$ and $\lambda ^ { \mathrm { i m g } }$ , and then used for denoising update:

$$
V _ { t } ^ { \mathrm { c f g } } = V _ { t } ^ { \mathrm { f u l l } } + \lambda ^ { \mathrm { i m g } } G _ { t } ^ { \mathrm { i m g } } + \lambda ^ { \mathrm { t e x t } } G _ { t } ^ { \mathrm { t e x t } } , \quad Z _ { t - 1 } = Z _ { t } - \Delta t V _ { t } ^ { \mathrm { c f g } } .\tag{5}
$$

Notably, reference image tokens, which are the primary pruning targets, are shared by both the fullcondition and image-only branches. However, as shown in Fig. 1(b), the straightforward strategy of pruning each branch using its native online attention scores yields mismatched token selections. Formally, let $I ^ { \mathrm { f u l l } }$ and $I ^ { \mathrm { i m g } }$ denote the visual tokens retained in the corresponding branches, with the layer index omitted for simplicity. The resulting text-guidance can be decomposed as:

$$
\tilde { G } _ { t } ^ { \mathrm { t e x t } } = \underbrace { v _ { \theta } ( Z _ { t } , t ; I ^ { \mathrm { f u l l } } , T ) - v _ { \theta } ( Z _ { t } , t ; I ^ { \mathrm { f u l l } } , \mathcal { Q } ) } _ { \mathrm { c l e a n t e x t g u i d a n c e } } + \underbrace { v _ { \theta } ( Z _ { t } , t ; I ^ { \mathrm { f u l l } } , \mathcal { Q } ) - v _ { \theta } ( Z _ { t } , t ; I ^ { \mathrm { i m g } } , \mathcal { Q } ) } _ { \delta _ { t } ^ { \mathrm { c o u f } } } .\tag{6}
$$

Here, the first term remains clean and valid text guidance, whereas an unexpected condition confusion arises from the second term $\delta _ { t } ^ { \mathrm { c o n f } }$ , as it introduces a spurious guidance signal from the mismatched visual token sets, further amplified by the non-negligible scale $\lambda ^ { \mathrm { t e x t } }$

To address this issue, we adopt a simple yet effective strategy that synchronizes pruning across branches sharing the same visual context. Concretely, token importance is estimated from the fullcondition branch, which has access to complete context, and the resulting pruning decision is reused by the image-only branch, yielding:

$$
\hat { I } ^ { \mathrm { i m g , } t , l } \gets \hat { I } ^ { \mathrm { f u l l , } t , l } , \forall t , l .\tag{7}
$$

This eliminates the confusion term as $\delta _ { t } ^ { \mathrm { c o n f } } = 0$ , ensuring the clean textual guidance as intended.

## 3.3 TAYLOR-CALIBRATED ATTENTION METRIC

Pooling-Induced Bias. Without loss of generality, we focus on a single layer, writing $s _ { j } ^ { l }$ in Eq. (3) as $s _ { j }$ for brevity. Although faithful, the sequence length of Q incurs non-negligible computation, un-

$$
\{ \breve { \mathcal { W } } _ { g } \} _ { g = 1 } ^ { G }
$$

$$
Q
$$

$$
\dot { G } = \dot { M } / w ^ { 2 }
$$

$$
Q
$$

$$
Q
$$

$$
w \times w
$$

$$
\begin{array} { r } { \bar { \pmb q } _ { g } = \frac { 1 } { w ^ { 2 } } \sum _ { \pmb q \in \mathcal { W } _ { q } } \pmb q } \end{array}
$$

$$
\begin{array} { r } { \bar { s } _ { j } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \mathrm { A t t n } ( \bar { \pmb q } _ { g } ) _ { j } } \end{array}
$$

However, interchanging averaging with the nonlinear Softmax operation introduces a window-wise numerical bias between $s _ { j }$ and $\bar { s } _ { j }$ . Considering a single window ${ \mathcal { W } } _ { g }$ , the exact discrepancy is:

$$
\Delta _ { g , j } = \frac { 1 } { w ^ { 2 } } \sum _ { { \pmb q } \in { \mathscr W } _ { g } } \mathrm { A t t n } ( \pmb q ) _ { j } - \mathrm { A t t n } ( \bar { \pmb q } _ { g } ) _ { j } .\tag{8}
$$

Taylor-Based Metric Calibration. Given Eq. (8), our goal is to derive a lightweight estimate of $\Delta _ { g , j }$ that compensates $\bar { s } _ { j }$ toward $s _ { j }$ , thereby improving the accuracy of token selection.

First, let $\mathbf { \epsilon } _ { \mathbf { \mathcal { G } } , i } = \mathbf { q } _ { g , i } - \bar { \mathbf { q } } _ { g }$ denote the deviation of each query $\mathbf { \Delta } \mathbf { q } _ { g , i } \in \mathcal { W } _ { g }$ from $\bar { \pmb q } _ { g }$ . Due to the spatial locality of visual features (Liu et al., $2 0 2 1 ) , \epsilon _ { g , i }$ is typically small, motivating a Taylor approximation of $\mathrm { A t t n } ( \cdot ) _ { j }$ around $\bar { \pmb q } _ { g }$ . Defining $p _ { g , j } = \mathrm { A t t } \mathbf { \bar { n } } ( \bar { \pmb q } _ { g } ) _ { j }$ , and the second-order expansion gives:

$$
\mathrm { A t t n } ( \pmb { q } _ { g , i } ) _ { j } \approx p _ { g , j } + \nabla p _ { g , j } ^ { \top } \pmb { \epsilon } _ { g , i } + \frac { 1 } { 2 } \pmb { \epsilon } _ { g , i } ^ { \top } \pmb { H } _ { g , j } \pmb { \epsilon } _ { g , i } ,\tag{9}
$$

where $\nabla p _ { g , j }$ and $H _ { g , j }$ are the gradient vector and Hessian matrix of $\mathrm { A t t n } ( \cdot ) _ { j }$ at $\bar { \pmb q } _ { g }$ , respectively. Subsequently, averaging Eq. (9) over ${ \mathcal { W } } _ { g }$ yields an approximation of Eq. (8) as:

$$
\Delta _ { g , j } \approx \frac { 1 } { 2 } \operatorname { T r } ( H _ { g , j } \Sigma _ { g } ) , \quad \Sigma _ { g } = \frac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } \epsilon _ { g , i } \epsilon _ { g , i } ^ { \top } ,\tag{10}
$$

where $\operatorname { T r } ( \cdot )$ denotes the trace of a matrix, and $\Sigma _ { g }$ is the covariance matrix of query deviations.

Furthermore, define the attention-weighted key mean as $\begin{array} { r } { \bar { \pmb { k } } _ { g } = \sum _ { j = 1 } ^ { N } p _ { g , j } \pmb { k } _ { j } } \end{array}$ and denote the diagonal entries of $\Sigma _ { g } \mathrm { \scriptsize ~ b y ~ } ( \sigma _ { g , 1 } ^ { 2 } , \dots , \sigma _ { g , d } ^ { 2 } )$ . As detailed in Sec. B.1, $H _ { g , j }$ admits a closed-form expression in Eq. (11), while the weak cross-channel correlations enable $\Sigma _ { g } \approx \sigma _ { g } ^ { 2 } I ,$ where $\begin{array} { r } { \sigma _ { g } ^ { 2 } = \frac { 1 } { d } \sum _ { c = 1 } ^ { d } \sigma _ { g , c } ^ { 2 } . } \end{array}$

$$
H _ { g , j } = \frac { p _ { g , j } } { d } [ ( k _ { j } - \bar { k } _ { g } ) ( { k _ { j } - \bar { k } _ { g } } ) ^ { \top } - \sum _ { n = 1 } ^ { N } p _ { g , n } ( k _ { n } - \bar { k } _ { g } ) ( k _ { n } - \bar { k } _ { g } ) ^ { \top } ] .\tag{11}
$$

Substituting these results into Eq. (10) yields an efficient computable approximation:

$$
\Delta _ { g , j } \approx \frac { p _ { g , j } } { 2 d } ( r _ { g , j } - \bar { r } _ { g } ) , \quad r _ { g , j } = \sigma _ { g } ^ { 2 } \| k _ { j } - \bar { k } _ { g } \| _ { 2 } ^ { 2 } , \quad \bar { r } _ { g } = \sum _ { n = 1 } ^ { N } p _ { g , n } r _ { g , n } .\tag{12}
$$

Finally, aggregating the window-wise corrections formulates the calibrated importance score $\tilde { s } _ { j }$ as Eq. (13), where we introduce the hyperparameter $\lambda$ to control the strength of correction.

$$
\tilde { s } _ { j } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \tilde { p } _ { g , j } , \quad \tilde { p } _ { g , j } = p _ { g , j } + \lambda \frac { p _ { g , j } } { 2 d } ( r _ { g , j } - \bar { r } _ { g } ) .\tag{13}
$$

Remark. This derivation indicates that pooling-induced bias is jointly governed by local query variation and key dispersion within the corresponding attention view. Larger query variance implies greater information loss from pooling, while larger centered key distance makes attention response more sensitive to such variation, both requiring stronger correction. This also echoes prior heuristic use of variance for regional importance (Fang et al., 2025; Wu et al., 2025b), while our derivation further reveals its theoretical role in modulating importance rather than directly estimating it.

## 3.4 FAST SCORING OPERATOR

While parallelizable across windows and keys, the calibration still involves a sequential dependency $p _ { g , j } \to k _ { g } \to r _ { g , j } \to \bar { r } _ { g } \to \tilde { p } _ { g , j }$ , where the repeated $p _ { g , j ^ { - } } \mathrm { w e i g h t e d }$ aggregations require multiple accesses to large-scale matrices, incurring non-negligible memory traffic. Therefore, we develop a tiled window-wise scoring operator that fuses the correction into the native computation of $p _ { g , j }$

First, as detailed in the Sec. ${ \bf B } . 2 , \tilde { p } _ { g , j }$ can be equivalently reformulated as:

$$
\tilde { p } _ { g , j } = p _ { g , j } [ 1 + \frac { \lambda \sigma _ { g } ^ { 2 } } { 2 d } ( \| \pmb { k } _ { j } \| _ { 2 } ^ { 2 } - 2 \pmb { k } _ { j } ^ { \top } \bar { \pmb { k } } _ { g } + 2 \| \bar { \pmb { k } } _ { g } \| _ { 2 } ^ { 2 } - \sum _ { n = 1 } ^ { N } p _ { g , n } \| \pmb { k } _ { n } \| _ { 2 } ^ { 2 } ) ] .\tag{14}
$$

This avoids the explicit dependence on $\bar { r } _ { g } ,$ , allowing the required statistics to be accumulated alongside online Softmax. Concretely, as summarized in Algo. 1, we adopt a tiled two-pass procedure

```latex
Algorithm 1 Inner Loop of the Fast Scoring Operator (FSO) (Native attention; Taylor correction).
Input: Pooled query $\bar { q } _ { g } ,$ variance $\sigma _ { g } ^ { 2 } ,$ key matrix $\overline { { K \in \mathbb { R } ^ { N \times d } } }$ in HBM; K is partitioned into $T _ { K } = \lceil N / B _ { K } \rceil$
blocks $\left\{ K _ { b } \right\} _ { b = 1 } ^ { T _ { K } }$ , each of size $B _ { K } \times d ;$ hyperparameter λ; $\scriptstyle { \frac { 1 } { \sqrt { d } } }$ is absorbed into K and omitted for brevity.
Output: Calibrated importance scores $\tilde { \pmb { p } } _ { g } \in \mathbb { R } ^ { N }$ in HBM.
1: Load $\bar { q } _ { g } , \sigma _ { g } ^ { 2 }$ from HBM to on-chip SRAM.
2: On chip, initialize $( m _ { 0 } , \ell _ { 0 } ) = ( { \dot { - } } \infty , 0 )$ and $( \pmb { v } , \pmb { u } ) = ( \mathbf { 0 } , 0 )$ in SRAM.
3: Pass 1: online Softmax with key-moment accumulation.
4: for $b = 1 , \dots , \overset { \bar { T _ { K } } } { }$ do
5: Load $\pmb { K } _ { b }$ from HBM to on-chip SRAM.
6: On chip, compute $z _ { b } \gets \bar { q } _ { g } K _ { b } ^ { \uparrow } , m _ { b } \gets \operatorname* { m a x } ( m _ { b - 1 } , \operatorname* { m a x } ( z _ { b } ) ) .$
7: On chip, compute $e _ { b } \gets \exp ( z _ { b } - m _ { b } ) , \alpha \gets \exp ( m _ { b - 1 } - m _ { b } ) , \ell _ { b } \gets \underline { { \alpha } } \ell _ { b - 1 } + \mathrm { s u m } ( e _ { b } ) .$
8: On chip, compute $v  \alpha v + e _ { b } K _ { b } , u  \alpha u + e _ { b }$ RowSqNorm(K<sub>b</sub>)<sup>⊤</sup>.
9: end for
10: On chip, compute $\bar { k } _ { g }  { \pmb v } / \ell _ { T _ { K } } , u  u / \ell _ { T _ { K } } .$
11: Pass 2: probability recomputation with Taylor calibration.
12: for $b = \overset { \cdot } { 1 } , \dots , T _ { K }$ do
13: Reload $\pmb { K } _ { b }$ from HBM to on-chip SRAM.
14: On chip, compute $z _ { b } \gets \bar { q } _ { g } K _ { b } ^ { \top } , p _ { g , b } \gets \exp ( z _ { b } \underset { - } { - } m _ { \underline { { T } } _ { K } } ) / \ell _ { \underline { { T } } _ { K } } .$
15: On chip, compute $\begin{array} { r } { A _ { b } \gets \mathrm { R o w S q N o r m } ( K _ { b } ) - 2 \bar { k } _ { g } K _ { b } ^ { \top } + 2 \| \boldsymbol { \vec { k } } _ { g } \| _ { 2 } ^ { 2 } - u , \tilde { p } _ { g , b } \gets p _ { g , b } \odot ( 1 + \frac { \lambda } { 2 } \sigma _ { g } ^ { 2 } A _ { b } ) } \end{array}$
16: Accumulate $\tilde { p } _ { g , b }$ into the b-th tile of $\tilde { p } _ { g }$ in HBM.
17: end for
18: return $\tilde { p } _ { g }$
inspired by FlashAttention (Dao, 2024) and Sec. B.2. Notably, without materializing intermediate
logits, two passes are minimally required to explicitly recover the normalized probabilities $p _ { g , j }$
During the first pass, the native logit maximum $m _ { b }$ and normalizer $\ell _ { b }$ are updated online in Static
Random-Access Memory (SRAM), and the attention-weighted key mean v and squared-norm statis
tic u are simultaneously accumulated under the same rescaling scheme, as Lines $6 { \sim } 8 .$ . After travers
ing all key blocks, normalization by $\ell _ { T _ { K } }$ yields the exact $\bar { k } _ { g }$ and $\begin{array} { r } { \sum _ { n = 1 } ^ { N } p _ { g , n } \| \pmb { k } _ { n } \| _ { 2 } ^ { 2 } = u , } \end{array}$ , as Line 10.
The second pass reloads each key block and recomputes the exact $p _ { g , j }$ utilizing the finalized statistics
$m _ { T _ { K } }$ and $\ell _ { T _ { K } }$ , as shown in Line 14. With all window-level statistics available, the correction then
depends only on the current key-specific terms $\| \pmb { k } _ { j } \| _ { 2 } ^ { 2 }$ and $\pmb { k } _ { i } ^ { \top } \bar { \pmb { k } } _ { g } ,$ enabling $\tilde { p } _ { g , j }$ to be computed
immediately as Line 15 and accumulated back into High Bandwidth Memory (HBM) score tile.
Overall, only calibrated scores are materialized, avoiding separate attention and correction matrices.
```

## 4 EXPERIMENTAL RESULTS AND ANALYSIS

## 4.1 EXPERIMENTAL SETTINGS

Models. Following recent advances in UMMs, we evaluate methods on two representative models: Bagel (Deng et al., 2025) and SenseNova-U1 (Diao et al., 2026), with generation performed in latent and pixel spaces, respectively. For Bagel, we use the multi-image fine-tuned checkpoint from (Chen et al., 2026), while SenseNova-U1 adopts its official checkpoint with native multi-image support.

Datasets and Evaluation Metrics. We conduct experiments on three widely used multi-image synthesis benchmarks, following their official splits and metric aggregation rules. Concretely, we report the geometric mean of Image Consistency and Prompt Following scores on the MacroBench (Chen et al., 2026) Customization subset across four reference-number groups (1–3, 4–5, 6–7, and 8–10), the Weighted-Ref-VIEScore for Object-Centric, Human-Centric, Human-Object Interaction (HOI), and Decomposition-and-Recomposition (De&Re) tasks on MICo-Bench (Wei et al., 2026), and the geometric mean of Prompt Following and Subject Consistency scores for the Single/Multiple/Scene categories of OmniContext (Wu et al., 2025a), each further divided into Character (Char.)/Object (Obj.)/Char.+Obj. subcategories. We also include the average score (Avg.) for MacroBench and OmniContext, and the Overall score for MICo-Bench. For metrics involving MLLM-based evaluation, we utilize the open-source Qwen3-VL-32B-Instruct (Bai et al., 2025a) as the evaluator.

Implementation Details. We implement all methods on top of the TorchUMM library (Luo et al., 2026), with FlashAttention (Dao, 2024) the acceleration baseline. Since visual tokens dominate the multimodal context, pruning is conducted exclusively on them, including both ViT and VAE tokens in Bagel and patch-level visual tokens in SenseNova-U1. By following (Lin et al., 2026), pruning is performed over the full sequence at denoising timesteps $t ^ { \mathrm { p } } \in \{ 4 9 , 3 9 , 2 9 , 1 9 , 9 \}$ . Identical retention ratio is applied to each reference image. In all experiments, we fix λ= 0.5 and w = 4 for TCAM and $B _ { K } = 6 4$ for FSO. All other inference settings, e.g., sampling and guidance parameters, follow the original model configurations. All experiments are conducted on one high-performance GPU.

Table 1: Comparison of multi-image synthesis performance scores by various methods for Bagel, evaluated on the MacroBench and MICo-Bench datasets. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="5">MacroBench</td><td colspan="5">MICo-Bench</td></tr><tr><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td><td>Object</td><td>Human</td><td>HOI</td><td>De&amp;Re</td><td>Overall↑</td></tr><tr><td>Full Model</td><td>8.89</td><td>8.51</td><td>7.97</td><td>7.20</td><td>8.14</td><td>47.52</td><td>40.68</td><td>42.47</td><td>53.69</td><td>46.66</td></tr><tr><td colspan="9">retain 25% tokens</td></tr><tr><td>Text-Aligned</td><td>8.21</td><td>8.22</td><td>8.07</td><td>7.44</td><td>7.98</td><td>52.11</td><td>35.77</td><td>46.95</td><td>52.92</td><td>47.64</td></tr><tr><td>VisionZip</td><td>8.18</td><td>8.07</td><td>8.00</td><td>6.97</td><td>7.81</td><td>49.65</td><td>22.99</td><td>38.79</td><td>56.40</td><td>43.38</td></tr><tr><td>SDTM</td><td>8.13</td><td>7.99</td><td>7.68</td><td>7.50</td><td>7.83</td><td>38.96</td><td>24.79</td><td>25.86</td><td>41.52</td><td>32.90</td></tr><tr><td>ToPi</td><td>8.61</td><td>8.39</td><td>8.12</td><td>7.49</td><td>8.15</td><td>53.48</td><td>40.17</td><td>46.00</td><td>63.69</td><td>51.97</td></tr><tr><td>CATP (Ours)</td><td>8.76</td><td>8.69</td><td>8.51</td><td>7.87</td><td>8.46</td><td>54.81</td><td>39.87</td><td>49.49</td><td>63.61</td><td>53.22</td></tr><tr><td colspan="9"></td></tr><tr><td>Text-Aligned</td><td>7.09</td><td>7.30</td><td>7.53</td><td>6.95</td><td>retain 10% tokens 7.22</td><td>49.42</td><td>23.60</td><td>39.72</td><td>55.18</td><td></td></tr><tr><td>VisionZip</td><td>5.44</td><td>6.02</td><td>6.25</td><td>5.70</td><td>5.85</td><td>37.43</td><td>13.92</td><td>27.58</td><td>46.95</td><td>43.35 33.00</td></tr><tr><td>SDTM</td><td>7.21</td><td>7.08</td><td>6.81</td><td>6.47</td><td>6.89</td><td>31.45</td><td>15.18</td><td>18.82</td><td>28.81</td><td>23.41</td></tr><tr><td>ToPi</td><td>7.32</td><td>7.18</td><td>7.28</td><td>6.86</td><td>7.16</td><td>46.89</td><td>29.29</td><td>38.81</td><td>56.60</td><td>44.20</td></tr><tr><td>CATP (Ours)</td><td>8.40</td><td>8.33</td><td>8.31</td><td>7.54</td><td>8.15</td><td>52.01</td><td>29.51</td><td>42.88</td><td>58.04</td><td>46.84</td></tr></table>

Table 2: Comparison of performance scores by various methods for Bagel/SenseNova-U1 on the OmniContext dataset with 25% tokens retained. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">Single</td><td colspan="3">Multiple</td><td colspan="3">Scene</td><td rowspan="2">Avg.↑</td></tr><tr><td>Char.</td><td>Obj.</td><td>Char.</td><td>Obj.</td><td>Char.+Obj.</td><td>Char.</td><td>Obj.</td><td>Char.+Obj.</td></tr><tr><td colspan="10">Bagel (Latent Space)</td></tr><tr><td>Full Model</td><td>8.50</td><td>9.28</td><td>7.97</td><td>9.22</td><td>8.60</td><td>7.49</td><td>8.21</td><td>8.15</td><td>8.43</td></tr><tr><td>Text-Aligned</td><td>8.16</td><td>8.77</td><td>7.56</td><td>9.10</td><td>8.25</td><td>7.46</td><td>8.23</td><td>7.64</td><td>8.14</td></tr><tr><td>VisionZip</td><td>7.06</td><td>7.90</td><td>6.78</td><td>8.64</td><td>7.97</td><td>7.40</td><td>8.50</td><td>7.58</td><td>7.73</td></tr><tr><td>SDTM</td><td>7.16</td><td>8.32</td><td>6.44</td><td>7.10</td><td>7.56</td><td>6.32</td><td>7.45</td><td>6.97</td><td>7.16</td></tr><tr><td>ToPi</td><td>6.98</td><td>8.61</td><td>6.82</td><td>8.89</td><td>8.40</td><td>7.08</td><td>8.42</td><td>8.02</td><td>7.90</td></tr><tr><td>CATP (Ours)</td><td>8.45</td><td>9.10</td><td>7.55</td><td>8.98</td><td>8.59</td><td>7.57</td><td>8.72</td><td>8.26</td><td>8.40</td></tr><tr><td colspan="10">SenseNova-U1 (Pixel Space)</td></tr><tr><td>Full Model</td><td>7.75</td><td>7.78</td><td>7.71</td><td>8.85</td><td>8.10</td><td>7.12</td><td>7.06</td><td>7.06</td><td>7.68</td></tr><tr><td>Text-Aligned</td><td>7.11</td><td>6.96</td><td>7.69</td><td>8.60</td><td>7.43</td><td>6.97</td><td>7.80</td><td>7.32</td><td>7.48</td></tr><tr><td>VisionZip</td><td>7.34</td><td>7.51</td><td>7.31</td><td>8.66</td><td>7.92</td><td>7.06</td><td>7.21</td><td>6.89</td><td>7.49</td></tr><tr><td>SDTM</td><td>7.02</td><td>7.73</td><td>6.84</td><td>8.01</td><td>7.46</td><td>6.89</td><td>6.69</td><td>6.22</td><td>7.11</td></tr><tr><td>ToPi</td><td>6.09</td><td>6.68</td><td>6.58</td><td>7.40</td><td>7.18</td><td>6.28</td><td>7.94</td><td>6.81</td><td>6.87</td></tr><tr><td>CATP (Ours)</td><td>7.42</td><td>8.06</td><td>7.76</td><td>8.76</td><td>8.40</td><td>7.52</td><td>7.69</td><td>7.63</td><td>7.91</td></tr></table>

## 4.2 MAIN RESULTS

Since token pruning for UMM generation remains rarely explored, we adapt representative baselines from closely related paradigms underlying UMMs, i.e., MLLMs and DiTs. Specifically, Text-Aligned Pruning is previously applied to UMM-based multi-image synthesis by (Chen et al., 2026). Both it and VisionZip (Yang et al., 2025) are originally developed for MLLM understanding tasks. SDTM (Fang et al., 2025) and ToPi (Lin et al., 2026) are DiT-oriented methods. For fair compari son, we restrict all methods to attention acceleration and adjust each to an equivalent computation reduction as CATP under predefined token retention ratios.

Quantitative Results. As shown in Tab. 1, CATP consistently outperforms compared methods across varying numbers of reference images and diverse generation tasks. Specifically, it surpasses the second-best methods by 0.31 and 1.25 points on MacroBench and MICo-Bench, respectively at a 25% retention ratio, and further increases the advantages to 0.93 and 2.64 points respectively at 10% retention ratio, demonstrating more pronounced robustness under aggressive context compression.

![](images/d076628869e11e5add4ac3d0a294f03f6bc369ec41bdaf8b5448221a3fd0eb54.jpg)  
Figure 3: Qualitative comparison of various methods on Bagel at a 25% token retention ratio. The failure/well-preserved regions are marked with red/green boxes.

Notably, CATP even outperforms the original full model on MacroBench Avg. at a 25% retention ratio, with gains primarily arising in multi-image settings (≥ 6 images) and reaching up to 0.67 points. This stems from the effective redundancy removal. As discussed in prior work, the intrinsic normalization property of attention can dilute individual weights in excessively long sequences (Wu et al., 2026a), making critical reference regions harder to distinguish and thereby yielding confused spatial layouts or degraded fine-grained details. Accordingly, pruning redundant context mitigates this effect by reducing unnecessary attention competition (Chen et al., 2026), while CATP further amplifies this benefit by more accurately retaining generation-critical regions.

Generalization Across Denoising Spaces. Given recent advances in pixel-space UMMs, we further adapt methods to the representative SenseNova-U1 and evaluate on OmniContext. As shown in Tab. 2, CATP maintains notable advantages of 0.26 and 0.42 Avg. points on Bagel and SenseNova-U1 at 25% token retention, respectively, with the margins further widening to 0.5 and 1.2 at 10% retention in Tab. A. These results demonstrate its impressive generalizability across denoising spaces.

Qualitative Results. As shown in Fig. 3, CATP delivers more favorable overall generation quality, particularly in spatial composition, human identity preservation, and critical background details. This stems from the joint effects of our designs, where GCP improves generation fidelity through coherent guidance, and TCAM better preserves fine-grained identity cues by assigning stronger importance corrections to regions with more details, making them more likely to survive pruning.

Further Remarks. Text-Aligned relies on attention signals from text or within reference images, which primarily reflect multimodal understanding rather than direct generation relevance. VisionZip merges similar visual regions, favoring coarse structural preservation but potentially losing identitycritical details. SDTM compresses only denoising tokens, making it better suited to simple generation scenarios but prone to substantial degradation when targeting higher acceleration. While ToPi explicitly targets context compression for generation, it reuses pruning decisions across lay ers to reduce scoring overhead, overlooking layer-wise differences in context exploration, and the fully computed anchor timesteps further limit the compression scale. In contrast, CATP develops a more accurate and efficient scoring process, enabling layer-wise token evaluation and selection. Under aggressive compression, different layers can exploit complementary critical regions rather than maintaining complete target content throughout, thereby maintaining overall generation quality.

Efficiency. We further assess the acceleration effect of CATP, with FlashAttention as the baseline. For each image count, we fix 1024-resolution inputs, sample 10 examples and report the average latency of the accelerated attention cores across all denoising calls. As shown in Fig. 4(a), atten-

![](images/a7d2eca47bde15b865890f61fd1b225387592f2d759100d050ce55aea5e32d4a.jpg)  
(a)

![](images/faf67945a83c01ee7104dbf28059d91e5c53423259a7a795c72de78d09cd6f59.jpg)  
(b)  
Figure 4: (a) Attention core latency of the full model and CATP, along with the relative speedup of CATP, across varying numbers of reference images. (b) Visualization of (Left) window-wise query variance and (Right) the resulting aggregated calibration term over reference images in TCAM.

Table 3: Ablation of GCP and TCAM modules for Bagel on MacroBench at two retention ratios.
<table><tr><td rowspan="2">GCP</td><td rowspan="2">TCAM</td><td colspan="5">Retain 25% Tokens</td><td colspan="5">Retain 10% Tokens</td></tr><tr><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td></tr><tr><td></td><td></td><td>8.32</td><td>8.19</td><td>8.14</td><td>7.62</td><td>8.07</td><td>6.65</td><td>6.87</td><td>7.25</td><td>6.85</td><td>6.91</td></tr><tr><td>V</td><td></td><td>8.75</td><td>8.65</td><td>8.36</td><td>7.45</td><td>8.30</td><td>8.23</td><td>8.28</td><td>8.39</td><td>7.38</td><td>8.07</td></tr><tr><td>√</td><td>√</td><td>8.76</td><td>8.69</td><td>8.51</td><td>7.87</td><td>8.46</td><td>8.40</td><td>8.33</td><td>8.31</td><td>7.54</td><td>8.15</td></tr></table>

tion computation becomes increasingly dominant with more reference images, translating context compression into larger practical gains, reaching 2.09× with 10 reference images at 25% retention.

## 4.3 ABLATION STUDY

On GCP. As shown in Tab. 3, GCP effectively improves conditional-guidance consistency, yielding average gains of 0.23 and 1.16 points at 25% and 10% retention ratios, respectively. Intuitively, more aggressive pruning makes the symmetric difference between conditioning branches increasingly dominated by less important tokens, introducing stronger irrelevant signals into the guidance residual. Therefore, GCP becomes more beneficial at lower retention ratios.

On TCAM. Tab. 3 also indicates that TCAM further improves gains under both settings. As visualized in Fig. 4(b), high-variance denoising regions concentrate around object boundaries and identitycritical details, consistent with the widely recognized properties in prior studies (Fang et al., 2025). TCAM exploits this signal via second-order calibration to strengthen corresponding reference regions, thus promoting the retention of identity-critical tokens for finer-grained synthesis.

On FSO. We further assess the effect of FSO in Tab. 4, by measuring scoring latency with ten 1024-resolution reference images. Concretely, dense attention achieves the best generation quality but incurs prohibitive pruning cost, while query pooling substantially reduces latency at the cost of degraded performance. Given TCAM’s additional calibration cost, FSO finally mitigates this by algebraically reformulating it into a fused operator, reducing latency to 5.39 ms without sacrificing quality. Notably, FSO is even faster than the pooling baseline, which owes to fine-grained HBM-level optimization enabling BF16 multiplication with FP32 accumulation.

Table 4: Effect of TCAM and FSO modules on scoring latency (ms) and MacroBench average score (Avg.).
<table><tr><td></td><td>Pooling TCAM FSO | Latency↓ | Avg.↑</td><td></td><td></td></tr><tr><td>√</td><td></td><td>102.97 7.24</td><td>8.48 8.30</td></tr><tr><td>√</td><td>√</td><td>30.12</td><td>8.46</td></tr><tr><td></td><td>√ 5</td><td>5.39</td><td>8.46</td></tr></table>

## 5 CONCLUSION

In this work, we conduct a pioneering investigation of visual token pruning for efficient UMM multiimage synthesis, and develop a novel training-free method, namely CATP. By performing layer-wise pruning on KV cache tokens, CATP enables adaptive utilization of different reference information. Furthermore, we explicitly preserve clean conditional guidance through Guidance-Consistent Pruning (GCP) and improve the discriminability of critical regions with a theoretically derived Taylor-Calibrated Attention Metric (TCAM), together with a Fast Scoring Operator (FSO) that reduces the scoring cost. Experiments on prevalent latent- and pixel-space UMMs across three multi-image benchmarks demonstrate the effectiveness of CATP, consistently outperforming competitive tokenreduction approaches across retention ratios while achieving up to 2.09× attention-core speedup.

## AI USE STATEMENT

In this work, we used generative AI tools for assist in the writing of proofs, assist with translation. We have not used generative AI tools for help develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, propose or refine hypotheses, design or provide feedback on research methodology or experiments, implement methods, interpret results, and generate synthetic data sets, clean and reformat dataset, support qualitative and thematic data analysis are not applicable to this work. Additionally, we used generative AI tools for sourcing/searching for information, identify relevant literature.

We have reviewed all AI-assisted work. Specifically, we independently verified all AI-polished mathematical formulations and proof-related content step by step to ensure their correctness and consistency with the underlying mathematical definitions. AI-assisted translations were manually reviewed by 2 authors to preserve the intended meaning. Information obtained through AI-assisted search was cross-checked against reliable sources, and all AI-identified literature was verified by consulting the corresponding original publications before being cited or discussed in this manuscript.

We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REFERENCES

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversitybased visual token pruning for large multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9392–9401, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Daniel Bolya and Judy Hoffman. Token merging for fast stable diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 4599–4603, 2023.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In Proceedings of the International Conference on Learning Representations, 2023.

Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Proceedings of the International Conference on Machine Learning, volume 235, pp. 6562–6595, 2024a.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In Proceedings of the European Conference on Computer Vision, pp. 19–35. Springer, 2024b.

Zhekai Chen, Yuqing Wang, Manyuan Zhang, and Xihui Liu. Macro: Advancing multi-reference image generation with structured long-context data. arXiv preprint arXiv:2603.25319, 2026.

Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for authentic virtual try-on in the wild. In Proceedings of the European Conference on Computer Vision, pp. 206–235. Springer, 2024.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In Proceedings ofthe International Conference on Learning Representations, 2024.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Haiwen Diao, Penghao Wu, Hanming Deng, Jiahao Wang, Shihao Bai, Silei Wu, Weichen Fan, Wenjie Ye, Wenwen Tong, Xiangyu Fan, et al. Sensenova-u1: Unifying multimodal understanding and generation with neo-unify architecture. arXiv preprint arXiv:2605.12500, 2026.

Xiangyu Fan, Zesong Qiu, Zhuguanyu Wu, Fanzhou Wang, Zhiqian Lin, Tianxiang Ren, Dahua Lin, Ruihao Gong, and Lei Yang. Phased dmd: Few-step distribution matching distillation via score matching within subintervals. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 41667–41676, 2026.

Haipeng Fang, Sheng Tang, Juan Cao, Enshuo Zhang, Fan Tang, and Tong-Yee Lee. Attend to not attended: Structure-then-detail token merging for post-training dit acceleration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18083–18092, 2025.

Google DeepMind. Gemini 3 flash: Model card. Google DeepMind Model Card, 2025. URL https://storage.googleapis.com/deepmind-media/Model-Cards/ Gemini-3-Flash-Model-Card.pdf.

Xin He, Longhui Wei, Jianbo Ouyang, Minghui Liao, Lingxi Xie, and Qi Tian. Emma: Efficient multimodal understanding, generation, and editing with a unified architecture. arXiv preprint arXiv:2512.04810, 2025.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshop on Deep Generative Models and Downstream Applications, 2021.

Junlong Ke, Zichen Wen, Boxue Yang, Yantai Yang, Xuyang Liu, Chenfei Liao, Zhaorun Chen, Shaobo Wang, and Linfeng Zhang. Flash-unified: A training-free and task-aware acceleration framework for native unified models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Findings, pp. 9131–9142, 2026.

Junqing Lin, Xingyu Zheng, Pei Cheng, Bin Fu, Jingwei Sun, and Guangzhong Sun. Token pruning for in-context generation in diffusion transformers. arXiv preprint arXiv:2602.01609, 2026.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26286–26296, 2024.

Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Junjie Chen, and Linfeng Zhang. From reusing to forecasting: Accelerating diffusion models with taylorseers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15853–15863, 2025.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF international conference on computer vision (ICCV), pp. 9992–10002. Ieee, 2021.

Yinyi Luo, Wenwen Wang, Hayes Bai, Hongyu Zhu, Hao Chen, Pan He, Marios Savvides, Sharon Li, and Jindong Wang. Torchumm: A unified multimodal model codebase for evaluation, analysis, and post-training. arXiv preprint arXiv:2604.10784, 2026.

Yuta Oshima, Daiki Miyake, Kohsei Matsutani, Yusuke Iwasawa, Masahiro Suzuki, Yutaka Matsuo, and Hiroki Furuta. Multibanana: A challenging benchmark for multi-reference text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 448–460, 2026.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Zhongwei Wan, Ziang Wu, Che Liu, Jinfa Huang, Zhihong Zhu, Peng Jin, Longyue Wang, and Li Yuan. Look-m: Look-once optimization in kv cache for efficient multimodal long-context inference. In Findings ofthe Associationfor Computational Linguistics, pp. 4065–4078, 2024.

Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. In Proceedings of the International Conference on Learning Representations, volume 2025, pp. 95118–95146, 2025.

Xinyu Wei, Kangrui Cen, Hongyang Wei, Zhen Guo, Bairui Li, Zeqing Wang, Jinrui Zhang, and Lei Zhang. Mico-150k: A comprehensive dataset advancing multi-image composition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 29695– 29706, 2026.

Chenyuan Wu, Jiahao Wang, Pengfei Zheng, et al. Omnigen2: Towards instruction-aligned multimodal generation. arXiv preprint arXiv:2506.18871, 2025a.

Zimeng Wu, Jiaxin Chen, and Yunhong Wang. Unified knowledge maintenance pruning and progressive recovery with weight recalling for large vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 8550–8558, 2025b.

Zimeng Wu, Donghao Wang, Chaozhe Jin, Jiaxin Chen, and Yunhong Wang. Probe and skip: Self-predictive token skipping for efficient long-context llm inference. arXiv preprint arXiv:2601.13155, 2026a.

Zimeng Wu, Yunhong Wang, Donghao Wang, and Jiaxin Chen. Collaborative multi-mode pruning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 39561–39571, 2026b.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19792–19802, 2025.

Linli Yao, Long Xing, Yang Shi, Sida Li, Yuanxin Liu, Yuhao Dong, Yi-Fan Zhang, Lei Li, Qingxiu Dong, Xiaoyi Dong, et al. Towards efficient multimodal large language models: A survey on token compression. 2026.

Jintao Zhang, Chendong Xiang, Haofeng Huang, Haocheng Xi, Jun Zhu, Jianfei Chen, et al. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. In Proceedings of the International Conference on Machine Learning, 2025a.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In Proceedings of the International Conference on Machine Learning, 2025b.

Qicheng Zhao, Qi Sun, and Zheyu Yan. Seeing the end at step zero: Accelerating diffusion mllms via mlp sparsity-aware truncation. arXiv preprint arXiv:2607.14557, 2026.

Shanshan Zhao, Xinjie Zhang, Jintao Guo, Jiakui Hu, Lunhao Duan, Minghao Fu, Yong Xien Chng, Guo-Hua Wang, Qing-Guo Chen, Zhao Xu, et al. Unified multimodal understanding and generation models: Advances, challenges, and opportunities. arXiv preprint arXiv:2505.02567, 2025.

## A MORE EXPERIMENTAL RESULTS AND ANALYSIS

## A.1 ADDITIONAL QUANTITATIVE RESULTS ON OMNICONTEXT

To complement the 25% retention results reported in Tab. 2, we further evaluate all methods on OmniContext (Wu et al., 2025a) under a more aggressive 10% token retention ratio. As shown in Tab. A, CATP continues to achieve superior performance in most scenarios on both Bagel and SenseNova-U1, further widening its margin over the second-best methods to 0.5 and 1.2 points, respectively. These results provide additional evidence for the robustness of CATP under aggressive acceleration and its generalization across various UMM generation spaces.

Table A: Comparison of performance scores by various methods for Bagel/SenseNova-U1 on the OmniContext dataset with 10% tokens retained. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">Single</td><td colspan="3">Multiple</td><td colspan="3">Scene</td><td rowspan="2">Avg.↑</td></tr><tr><td>Char.</td><td>Obj.</td><td>Char.</td><td>Obj.</td><td>Char.+Obj.</td><td>Char.</td><td>Obj.</td><td>Char.+Obj.</td></tr><tr><td colspan="10">Bagel (Latent Space)</td></tr><tr><td>Full Model</td><td>8.50</td><td>9.28</td><td>7.97</td><td>9.22</td><td>8.60</td><td>7.49</td><td>8.21</td><td>8.15</td><td>8.43</td></tr><tr><td>Text-Aligned</td><td>7.06</td><td>7.36</td><td>5.17</td><td>8.33</td><td>7.33</td><td>6.13</td><td>7.95</td><td>7.26</td><td>7.07</td></tr><tr><td>VisionZip</td><td>4.38</td><td>4.62</td><td>3.67</td><td>7.50</td><td>4.01</td><td>4.27</td><td>7.11</td><td>6.06</td><td>5.20</td></tr><tr><td>SDTM</td><td>6.57</td><td>6.23</td><td>5.89</td><td>5.77</td><td>6.68</td><td>5.31</td><td>6.31</td><td>5.75</td><td>6.06</td></tr><tr><td>ToPi</td><td>5.33</td><td>6.65</td><td>4.16</td><td>7.92</td><td>6.90</td><td>5.15</td><td>8.20</td><td>6.89</td><td>6.38</td></tr><tr><td>CATP (Ours)</td><td>6.74</td><td>8.19</td><td>6.78</td><td>7.94</td><td>8.25</td><td>6.89</td><td>8.14</td><td>7.61</td><td>7.57</td></tr><tr><td colspan="10">SenseNova-U1 (Pixel Space)</td></tr><tr><td>Full Model</td><td>7.75</td><td>7.78</td><td>7.71</td><td>8.85</td><td>8.10</td><td>7.12</td><td>7.06</td><td>7.06</td><td>7.68</td></tr><tr><td>Text-Aligned</td><td>5.72</td><td>4.78</td><td>6.21</td><td>7.88</td><td>6.56</td><td>6.40</td><td>7.94</td><td>6.58</td><td>6.51</td></tr><tr><td>VisionZip</td><td>6.35</td><td>6.19</td><td>5.54</td><td>7.90</td><td>6.51</td><td>5.69</td><td>7.39</td><td>6.22</td><td>6.47</td></tr><tr><td>SDTM</td><td>6.36</td><td>7.27</td><td>6.46</td><td>6.90</td><td>6.63</td><td>6.01</td><td>5.76</td><td>5.41</td><td>6.35</td></tr><tr><td>ToPi</td><td>4.70</td><td>5.64</td><td>4.45</td><td>6.96</td><td>5.38</td><td>5.13</td><td>7.54</td><td>5.87</td><td>5.71</td></tr><tr><td>CATP (Ours)</td><td>7.31</td><td>7.45</td><td>7.84</td><td>8.55</td><td>8.18</td><td>7.04</td><td>7.86</td><td>7.42</td><td>7.71</td></tr></table>

## A.2 FURTHER ABLATIONS

On Calibration Strength λ. Since TCAM adopts approximations to avoid computing higherorder terms, its second-order estimate captures the correction trend rather than an exact discrepancy. Therefore, we heuristically introduce λ in Eq. (13) to control the calibration strength, whose effect is ablated in Tab. B. As λ decreases, TCAM approaches the vanilla pooling baseline and provides insufficient emphasis on fine-grained reference regions, whereas an excessively large value biases selection toward high-variance regions and may suppress smoother yet relevant content. Given the overall performance peak at λ = 0.5, we empirically fix this value for all experiments.

Table B: Effect of the correction strength λ on MacroBench score for Bagel at a 25% token retention ratio. The default setting is underlined.
<table><tr><td>λ</td><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td></tr><tr><td>0</td><td>8.75</td><td>8.65</td><td>8.36</td><td>7.45</td><td>8.30</td></tr><tr><td>0.25</td><td>8.73</td><td>8.70</td><td>8.34</td><td>7.73</td><td>8.38</td></tr><tr><td>0.50</td><td>8.76</td><td>8.69</td><td>8.51</td><td>7.87</td><td>8.46</td></tr><tr><td>0.75</td><td>8.79</td><td>8.76</td><td>8.49</td><td>7.77</td><td>8.45</td></tr><tr><td>1.00</td><td>8.81</td><td>8.67</td><td>8.37</td><td>7.74</td><td>8.40</td></tr><tr><td>1.25</td><td>8.83</td><td>8.65</td><td>8.35</td><td>7.68</td><td>8.38</td></tr></table>

On Window Size w. We assess the effect of the pooling window size w in TCAM. As reported in Tab. C, smaller windows incur less loss of fine-grained cues and therefore yield better generation quality, but at the cost of a longer query sequence with higher scoring latency. Larger windows exhibit the opposite trade-off, reducing scoring overhead while sacrificing more local details. Accordingly, we empirically choose $w = 4$ as a balanced configuration and fix it across all experiments.

Table C: Effect of the window size w on MacroBench score and scoring latency (ms) for Bagel at a 25% token retention ratio. The default setting is underlined.
<table><tr><td>w</td><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑ </td><td>Latency↓</td></tr><tr><td>2</td><td>8.75</td><td>8.73</td><td>8.49</td><td>7.98</td><td>8.49</td><td>17.87</td></tr><tr><td>4-8</td><td>8.76</td><td>8.69</td><td>8.51</td><td>7.87</td><td>8.46</td><td>5.39</td></tr><tr><td></td><td>8.68</td><td>8.57</td><td>8.27</td><td>7.68</td><td>8.30</td><td>2.31</td></tr></table>

Qualitative Ablation on Main Modules. Beyond the quantitative ablations in Tab. 3, we further visualize the effects of GCP and TCAM in Fig. A. With the general pruning scheme, independent token selection across conditioning branches amplifies irrelevant visual cues through CFG guidance, leading to implausible rendering or distorted content. GCP alleviates this issue by enforcing consistent pruning across branches. TCAM further improves generation by emphasizing information-rich reference regions, yielding local patterns and details that better match the references.

Reference Images  
Baseline  
+ GCP  
+ GCP + TCAM  
![](images/a28cda3d00fae82dc82b6a0163650b46863a839dc0b06f4006cd853edfee78fa.jpg)  
A … lighting fixture from image 1 hangs above a dining table, illuminating a … bowl from image 2 placed at the center. Nearby, a travel guide app for digital devices from image 3 is open on a smartphone, while a … pill from image 4 rests on the table, hinting at a mindful gathering.  
Figure A: Qualitative ablation of GCP and TCAM modules on Bagel at a 25% token retention ratio. The failure/well-preserved regions are marked with red/green boxes.

On Alternatives for Pooling-Induced Bias. For the pooling-induced bias discussed in Sec. 3.3, we further compare our solution (i.e. TCAM + FSO) with several alternative designs in Tab. D. intuitively, the inferior performance of the baseline Average Pooling stems from indiscriminate within window averaging that smooths fine-grained cues. Therefore, a direct alternative is Max Pooling, however, channel-wise maxima may originate from different spatial tokens, disrupting the semantic coherence of the pooled representation and yielding even worse performance. Inspired by our derivation in Eq. (13), which suggests strengthening the contribution of high-variance denoising regions, we further consider an intuitive alternative that leaves queries from the highest-variance windows unpooled, denoted as Mixed Query. Its improved generation quality in Tab. D provides additional empirical support for our motivation, but the enlarged query sequence incurs increased scoring latency. In contrast, TCAM together with FSO achieves gains in both generation quality and scoring efficiency, further validating the effectiveness of our designs.

## A.3 ROBUSTNESS ACROSS MLLM EVALUATORS

Noting that MLLM-based evaluation can exhibit evaluator-specific biases and judgment inconsistencies (Chen et al., 2024a), we therefore re-evaluate all methods on MacroBench utilizing two other representative MLLMs: the open-source Qwen2.5-VL-72B-Instruct (Bai et al., 2025b) and the closed-source Gemini-3-Flash (Google DeepMind, 2025). As shown in Tab. E, CATP still consistently achieves the best performance across both evaluators and distinct token retention ratios, despite noticeable differences in their scoring distributions. This further suggests that CATP more reliably preserves generation quality rather than benefiting from a particular evaluator.

Table D: Effect of various design choices for the pooling-induced bias on MacroBench for Bagel at a 25% retention ratio. The best results are highlighted in bold and the default setting is underlined.
<table><tr><td>Method</td><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td><td>Latency↓</td></tr><tr><td>Average Pooling</td><td>8.75</td><td>8.65</td><td>8.36</td><td>7.45</td><td>8.30</td><td>7.24</td></tr><tr><td>Max Pooling</td><td>8.59</td><td>8.56</td><td>8.29</td><td>7.50</td><td>8.24</td><td>7.23</td></tr><tr><td>Mixed Query</td><td>8.80</td><td>8.74</td><td>8.48</td><td>7.66</td><td>8.42</td><td>31.47</td></tr><tr><td>TCAM + FSO</td><td>8.76</td><td>8.69</td><td>8.51</td><td>7.87</td><td>8.46</td><td>5.39</td></tr></table>

Additionally, it is noteworthy that, under the more capable evaluator, performance with larger numbers of reference images remains far from the maximum score (i.e. 10.0). While context compression provides partial recovery, we consider that further gains in such challenging scenarios should primarily stem from stronger native multi-reference generation capabilities of the foundation model, motivating continued research along this direction.

Table E: Comparison of multi-image synthesis performance scores by various methods for Bagel on MacroBench, evaluated by Qwen2.5-VL-72B-Instruct and Gemini-3-Flash. The best results are highlighted in bold.
<table><tr><td rowspan="3">Method</td><td colspan="5">Qwen2.5-VL-72B-Instruct</td><td colspan="5">Gemini-3-Flash</td></tr><tr><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td></tr><tr><td>Full Model</td><td>8.43</td><td>7.65</td><td>6.45</td><td>5.42</td><td>6.99</td><td>9.01</td><td>8.48</td><td>6.93</td><td>4.45</td><td>7.21</td></tr><tr><td colspan="9">retain 25% tokens</td><td></td></tr><tr><td>Text-Aligned</td><td>8.07</td><td>7.52</td><td>6.26</td><td>5.34</td><td>6.80</td><td>8.05</td><td>8.34</td><td>6.76</td><td>4.82</td><td>6.99</td></tr><tr><td>VisionZip</td><td>7.81</td><td>7.36</td><td>5.88</td><td>5.02</td><td>6.52</td><td>7.43</td><td>7.71</td><td>6.36</td><td>3.72</td><td>6.31</td></tr><tr><td>SDTM</td><td>7.68</td><td>6.75</td><td>5.28</td><td>3.88</td><td>5.90</td><td>7.84</td><td>7.69</td><td>6.42</td><td>4.59</td><td>6.63</td></tr><tr><td>ToPi</td><td>8.10</td><td>7.45</td><td>6.31</td><td>5.18</td><td>6.76</td><td>8.47</td><td>8.33</td><td>6.80</td><td>4.66</td><td>7.06</td></tr><tr><td>CATP (Ours)</td><td>8.44</td><td>7.84</td><td>6.26</td><td>5.62</td><td>7.04</td><td>8.87</td><td>8.70</td><td>7.54</td><td>5.25</td><td>7.59</td></tr><tr><td colspan="9">retain 10% tokens</td><td></td></tr><tr><td>Text-Aligned</td><td>7.34</td><td>6.46</td><td>5.75</td><td>4.63</td><td>6.05</td><td>6.13</td><td>6.45</td><td>5.41</td><td>3.89</td><td>5.47</td></tr><tr><td>VisionZip</td><td>5.57</td><td>5.05</td><td>4.57</td><td>4.18</td><td>4.84</td><td>4.43</td><td>4.07</td><td>3.17</td><td>2.11</td><td>3.45</td></tr><tr><td>SDTM</td><td>6.71</td><td>5.40</td><td>4.08</td><td>3.05</td><td>4.81</td><td>6.74</td><td>6.37</td><td>5.42</td><td>3.79</td><td>5.58</td></tr><tr><td>ToPi</td><td>7.13</td><td>6.39</td><td>5.05</td><td>4.64</td><td>5.80</td><td>6.94</td><td>6.34</td><td>4.71</td><td>3.20</td><td>5.30</td></tr><tr><td>CATP (Ours)</td><td>8.08</td><td>7.55</td><td>6.03</td><td>5.15</td><td>6.70</td><td>8.10</td><td>8.20</td><td>7.01</td><td>4.99</td><td>7.07</td></tr></table>

## A.4 STATISTICAL STABILITY

We further assess the statistical stability of methods on Bagel at a 25% token retention ratio. Specifically, all methods are evaluated on MacroBench over five random seeds for image generation, with the mean and standard deviation reported. As shown in Tab. F, CATP consistently maintains superior performance, improving the Avg. over the second-best method by 0.306 points while exhibiting the lowest Avg. standard deviation of 0.02. These results further demonstrate both the effectiveness and stability of our method.

## A.5 COMPATIBILITY WITH ORTHOGONAL ACCELERATION METHODS

CATP accelerates inference from the perspective of context compression, reducing attention computation by removing redundant conditional tokens. As discussed in Sec. 2, several methods optimize efficiency along orthogonal dimensions, motivating an exploration of their joint effect with CATP. Specifically, we combine CATP at a 25% token retention ratio with two representative approaches,

Table F: Comparison of MacroBench scores (mean ± std) by various methods on Bagel at a 25% token retention ratio. The best results are highlighted in bold.
<table><tr><td>Method</td><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td></tr><tr><td>Full Model</td><td> $8 . 8 1 0 \pm 0 . 0 4 5$ </td><td> $8 . 4 1 2 \pm 0 . 0 8 3$ </td><td> $8 . 0 4 8 \pm 0 . 0 7 8$ </td><td> $7 . 2 5 6 \pm 0 . 0 9 9$ </td><td> $8 . 1 3 2 \pm 0 . 0 4 7$ </td></tr><tr><td>Text-Aligned</td><td> $8 . 2 1 0 \pm 0 . 0 4 1$ </td><td> $8 . 1 6 4 \pm 0 . 0 6 0$ </td><td> $8 . 0 6 4 \pm 0 . 0 4 8$ </td><td> $7 . 4 0 6 \pm 0 . 0 6 4$ </td><td> $7 . 9 6 0 \pm 0 . 0 2 7$ </td></tr><tr><td>VisionZip</td><td> $8 . 1 2 8 \pm 0 . 0 6 5$ </td><td> $8 . 0 1 0 \pm 0 . 0 9 1$ </td><td> $7 . 9 6 6 \pm 0 . 1 6 8$ </td><td> $7 . 0 1 2 \pm 0 . 1 5 3$ </td><td> $7 . 7 8 0 \pm 0 . 0 6 5$ </td></tr><tr><td>SDTM</td><td> $8 . 1 1 0 \pm 0 . 2 2 2$ </td><td> $8 . 0 4 8 \pm 0 . 1 2 0$ </td><td> $7 . 8 7 6 \pm 0 . 1 5 7$ </td><td> $7 . 2 5 2 \pm 0 . 1 6 5$ </td><td> $7 . 8 2 2 \pm 0 . 0 6 6$ </td></tr><tr><td>ToPi</td><td> $8 . 5 7 0 \pm 0 . 0 5 6$ </td><td> $8 . 2 7 8 \pm 0 . 0 7 3$ </td><td> $8 . 2 4 0 \pm 0 . 0 8 7$ </td><td> $7 . 5 3 2 \pm 0 . 1 1 3$ </td><td> $8 . 1 5 2 \pm 0 . 0 3 2$ </td></tr><tr><td>CATP (Ours)</td><td> ${ \bf 8 . 8 1 4 \pm 0 . 0 7 3 }$ </td><td> ${ \bf 8 . 6 4 0 \pm 0 . 1 0 4 }$ </td><td> ${ \bf 8 . 5 4 0 \pm 0 . 0 3 9 }$ </td><td> ${ \bf 7 . 8 3 8 \pm 0 . 0 5 8 }$  </td><td> ${ \bf 8 . 4 5 8 \pm 0 . 0 2 0 }$  </td></tr></table>

SDTM (Fang et al., 2025) and TaylorSeer (Liu et al., 2025), both adopting their official default settings. SDTM shortens the denoising sequence, whereas TaylorSeer performs block-wise computation skipping across timesteps, and we therefore report the end-to-end speedup ratio of the entire denoising process relative to the full model, with 10 input reference images.

As shown in Tab. G, both methods alone already improve average generation quality over the full model. Here, SDTM fails to provide a meaningful acceleration, as the fixed context in UMM paradigm renders computation over its targeted denoising sequence non-dominant, leaving the overhead of frequent selection operation insufficiently amortized. Furthermore, when combined with these methods, CATP continues to reduce computational cost, while even improving generation performance by 0.19 and 0.26 points, respectively, primarily due to the elimination of interfering contextual redundancy. We also include representative generation results in Fig. B, where both methods produce erroneous outputs, yet their combination with CATP corrects these failures. These results demonstrate the favorable adaptability of CATP to prevalent techniques, and suggest that appropriate compression may itself provide a promising direction for improving downstream generation quality.

Table G: MacroBench scores and overall denoising speedup of SDTM and TaylorSeer, individually and in combination with CATP, on Bagel. The best results are highlighted in bold.
<table><tr><td>Method</td><td>1-3</td><td>4-5</td><td>6-7</td><td>8-10</td><td>Avg.↑</td><td>Speedup↑</td></tr><tr><td>Full Model</td><td>8.89</td><td>8.51</td><td>7.97</td><td>7.20</td><td>8.14</td><td>1.00×</td></tr><tr><td>SDTM SDTM + CATP (Ours)</td><td>8.95 8.91</td><td>8.79 8.98</td><td>8.41 8.63</td><td>7.65 8.06</td><td>8.45 8.64</td><td>1.04× 1.41×</td></tr><tr><td>TaylorSeer</td><td>8.76</td><td>8.48</td><td>8.21</td><td>7.70</td><td>8.29</td><td>2.39×</td></tr><tr><td>TaylorSeer + CATP (Ours)</td><td>8.78</td><td>8.95</td><td>8.65</td><td>7.81</td><td>8.55</td><td>3.06×</td></tr></table>

![](images/9de22b785c7405fa6c1e42f9e12354e4ae183dfb99fe388e66a89693bfec4642.jpg)  
Place the coffee cup on the kitchen counter and lay the dried leaf beside it.

Figure B: Qualitative generation results by SDTM and TaylorSeer, individually and combined with CATP, on Bagel. The failure/well-preserved regions are marked with red/green boxes.

## B DETAILED DERIVATIONS

In this section, we present detailed derivations for TCAM and FSO. Sec. B.1 first formulates the pooling-induced attention discrepancy, then derives its second-order Taylor expansion with combined Hessian and query variance, yielding the calibrated importance score. Sec. B.2 reformulates this calibration into a simplified form, follow by a clarification of the theoretical basis of exact tiled online Softmax and the analogous online accumulation of the required intermediate statistics.

## B.1 DERIVATION OF THE TAYLOR CALIBRATION

Pooling-Induced Bias. We begin with the exact attention-based metric $s _ { j }$ computed over the complete query sequence $Q ,$ corresponding to Eq. (3) in the main text:

$$
s _ { j } = \frac { 1 } { M } \sum _ { \boldsymbol { q } \in \boldsymbol { Q } } \mathrm { A t t n } ( \boldsymbol { q } ) _ { j } , \quad \mathrm { A t t n } ( \boldsymbol { q } ) = \mathrm { S o f t m a x } ( \frac { \boldsymbol { q } \mathbf { K } ^ { \top } } { \sqrt { d } } ) \in \mathbb { R } ^ { N } .\tag{B.1}
$$

Afterward, partitioning $Q$ into $G = M / w ^ { 2 }$ non-overlapping $w \times w$ windows and applying average pooling yields, for the g-th window,

$$
{ \bar { \pmb q } } _ { g } = \frac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } { \pmb q } _ { g , i } ,\tag{B.2}
$$

which further produces the coarsened window-level score:

$$
p _ { g , j } = \mathrm { A t t n } ( \bar { q } _ { g } ) _ { j } .\tag{B.3}
$$

Accordingly, grouping the queries by window, the original dense attention metric of Eq. (B.1) can be equivalently written as:

$$
s _ { j } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \hat { p } _ { g , j } , \quad \hat { p } _ { g , j } = \frac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } \mathrm { A t t n } ( q _ { g , i } ) _ { j } ,\tag{B.4}
$$

whereas the metric with pooled queries becomes:

$$
\bar { s } _ { j } = \frac { 1 } { G } et { } { ' } \sum _ { g = 1 } ^ { G } \mathrm { A t t n } ( \bar { q } _ { g } ) _ { j } = \frac { 1 } { G } et { } { ' } \sum _ { g = 1 } ^ { G } p _ { g , j } .\tag{B.5}
$$

Although $\bar { s } _ { j }$ can be computed more efficiently, the difference from the exact $s _ { j }$ goes beyond a simple reordering of window-wise averaging and Softmax. Due to the nonlinearity of Softmax, each window incurs an additional non-negligible bias:

$$
\Delta _ { g , j } = \hat { p } _ { g , j } - p _ { g , j } = \frac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } \mathrm { A t t n } ( q _ { g , i } ) _ { j } - p _ { g , j } ,\tag{B.6}
$$

which is identical to Eq. (8) of Sec. 3.3, and aggregating these window-level discrepancies gives:

$$
s _ { j } = \bar { s } _ { j } + \frac { 1 } { G } et { } { ' } \sum _ { g = 1 } ^ { G } \Delta _ { g , j } .\tag{B.7}
$$

Therefore, recovering the accurate importance score reduces to efficiently estimating the windowlevel correction $\Delta _ { g , j }$ , which motivates the following derivation.

Taylor Expansion for Discrepancy Approximation. For the $g \cdot$ -th window, denote the deviation of each query $\mathbf { \delta } _ { q , i }$ from the average-pooled representation $\bar { \pmb q } _ { g }$ as:

$$
\epsilon _ { g , i } = q _ { g , i } - \bar { q } _ { g } ,\tag{B.8}
$$

Given the definition in Eq. (B.2), these deviations sum to zero within the corresponding window:

$$
{ \sum } _ { i = 1 } ^ { w ^ { 2 } } \epsilon _ { g , i } = { \sum } _ { i = 1 } ^ { w ^ { 2 } } q _ { g , i } - w ^ { 2 } \bar { q } _ { g } = \bf { 0 } .\tag{B.9}
$$

Due to the spatial locality of visual representations, queries within each window typically lie close to their pooled center. This motivates a local Taylor expansion of $\mathrm { A t t n } ( \pmb q _ { g , i } ) _ { j }$ around $\bar { \pmb q } _ { g }$ . Using Eq. (B.3), the expansion simplifies to:

$$
\begin{array} { r l r } {  { \mathrm { A t t n } ( \pmb { q } _ { g , i } ) _ { j } = \mathrm { A t t n } ( \bar { \pmb { q } } _ { g } + \epsilon _ { g , i } ) _ { j } = p _ { g , j } + \nabla p _ { g , j } ^ { \top } \epsilon _ { g , i } + \frac { 1 } { 2 } \epsilon _ { g , i } ^ { \top } H _ { g , j } \epsilon _ { g , i } + \pmb { R } ^ { ( 3 ) } } } \\ & { } & \\ & { } & { \approx p _ { g , j } + \nabla p _ { g , j } ^ { \top } \epsilon _ { g , i } + \frac { 1 } { 2 } \epsilon _ { g , i } ^ { \top } H _ { g , j } \epsilon _ { g , i } , } \end{array}\tag{B.10}
$$

where $\nabla p _ { g , j } = \nabla _ { q } \mathrm { A t t n } ( \pmb q ) _ { j } | _ { \pmb q = \bar { \pmb q } _ { g } }$ and ${ H _ { g , j } } = \nabla _ { \pmb { q } } ^ { 2 } \mathrm { A t t n } ( \pmb { q } ) _ { j } | _ { \pmb { q = \bar { q } } _ { \leq } }$ denote the first-order derivative and the second-order Hessian matrix, respectively. Terms beyond the second order, represented by $\pmb { R } ^ { ( 3 ) }$ , are omitted due to their negligible magnitude.

Furthermore, subtracting $p _ { g , j }$ from both sides of Eq. (B.10) and averaging over all queries within the window yields:

$$
\begin{array} { r l } & { \Delta _ { g , j } = \cfrac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } ( \mathrm { A t t n } ( \pmb { q } _ { g , i } ) _ { j } - p _ { g , j } ) } \\ & { \qquad \approx \cfrac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } ( \nabla p _ { g , j } ^ { \top } \pmb { \epsilon } _ { g , i } ) + \cfrac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } ( \cfrac { 1 } { 2 } \epsilon _ { g , i } ^ { \top } \pmb { H } _ { g , j } \pmb { \epsilon } _ { g , i } ) } \\ & { \qquad = \nabla p _ { g , j } ^ { \top } \cfrac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } \epsilon _ { g , i } + \cfrac { 1 } { 2 w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } ( \epsilon _ { g , i } ^ { \top } \pmb { H } _ { g , j } \pmb { \epsilon } _ { g , i } ) . } \end{array}\tag{B.11}
$$

Given Eq. (B.9), the first-order term vanishes exactly. Meanwhile, by defining the population covariance of the within-window query deviations as:

$$
\boldsymbol { \Sigma } _ { g } = \frac { 1 } { w ^ { 2 } } \sum _ { i = 1 } ^ { w ^ { 2 } } \boldsymbol { \epsilon } _ { g , i } \boldsymbol { \epsilon } _ { g , i } ^ { \top } ,\tag{B.12}
$$

and applying the identity $\pmb { x } ^ { \top } \pmb { A } \pmb { x } = \operatorname { T r } ( \pmb { A } \pmb { x } \pmb { x } ^ { \top } )$ , where Tr(·) is the matrix trace, the second-order term can be rewritten as $\textstyle { \frac { 1 } { 2 } } \operatorname { T r } ( H _ { g , j } \Sigma _ { g } )$ . Therefore, Eq. (B.11) yields

$$
\Delta _ { g , j } \approx \frac { 1 } { 2 } \operatorname { T r } ( H _ { g , j } \Sigma _ { g } ) .\tag{B.13}
$$

Hessian of the Softmax Probability. For notational convenience, we denote the attention logit associated with $k _ { j }$ as:

$$
z _ { j } ( \pmb q ) = \frac { \pmb q ^ { \top } \pmb k _ { j } } { \sqrt { d } } ,\tag{B.14}
$$

such that the Softmax-normalized attention probability in Eq. (B.1) becomes:

$$
p _ { j } \equiv \mathrm { A t t n } ( { \pmb q } ) _ { j } = \frac { \exp ( z _ { j } ( { \pmb q } ) ) } { \sum _ { n = 1 } ^ { N } \exp ( z _ { n } ( { \pmb q } ) ) } .\tag{B.15}
$$

Here for brevity, we temporarily omit the window index $g$ and evaluate all quantities at a generic query $\mathbf { \delta } _ { q . }$ The derivative of Softmax w.r.t its logits is:

$$
\frac { \partial p _ { j } } { \partial z _ { n } } = p _ { j } ( \mathbb { I } ( j = n ) - p _ { n } ) ,\tag{B.16}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function. Applying the chain rule to Eq. (B.16) with Eq. (B.14) gives:

$$
\begin{array} { r l } & { \nabla _ { { q } } p _ { j } = \displaystyle \sum _ { n = 1 } ^ { N } \frac { \partial p _ { j } } { \partial z _ { n } } \nabla _ { { q } } z _ { n } = \displaystyle \sum _ { n = 1 } ^ { N } \frac { { k _ { n } } } { \sqrt { d } } p _ { j } \big ( \mathbb { I } \big ( j = n \big ) - p _ { n } \big ) } \\ & { \qquad = \displaystyle \frac { p _ { j } } { \sqrt { d } } \big ( { k _ { j } } - \sum _ { n = 1 } ^ { N } p _ { n } { k _ { n } } \big ) } \\ & { \qquad \equiv \displaystyle \frac { p _ { j } } { \sqrt { d } } c _ { j } , } \end{array}\tag{B.17}
$$

where we define the attention-weighted mean key k<sup>¯</sup> and the centered key vectors $c _ { j }$ as:

$$
{ \bar { \pmb { k } } } = \sum _ { n = 1 } ^ { N } p _ { n } { \pmb { k } } _ { n } , \quad { \pmb { c } } _ { j } = { \pmb { k } } _ { j } - { \bar { \pmb { k } } } .\tag{B.18}
$$

Adopting Eq. (B.17) and Eq. (B.18), we further prepare the Jacobian of $\bar { \pmb k }$ w.r.t q as:

$$
\begin{array} { l } { { \nabla _ { q } \bar { k } = \displaystyle \sum _ { n = 1 } ^ { N } k _ { n } \big ( \nabla _ { q } p _ { n } \big ) ^ { \top } } } \\ { { \displaystyle ~ = \frac { 1 } { \sqrt { d } } \sum _ { n = 1 } ^ { N } p _ { n } k _ { n } \big ( { k _ { n } - \bar { k } } \big ) ^ { \top } } } \\ { { \displaystyle ~ = \frac { 1 } { \sqrt { d } } ( \sum _ { n = 1 } ^ { N } p _ { n } k _ { n } k _ { n } ^ { \top } - \bar { k } \bar { k } ^ { \top } ) } , } \end{array}\tag{B.19}
$$

Differentiating Eq. (B.17) to the second-order gradient and noting that $\nabla _ { q } \pmb { c } _ { j } = - \nabla _ { q } \bar { k }$ now gives:

$$
\begin{array} { l } { { \nabla _ { q } ^ { 2 } p _ { j } = \displaystyle \frac { 1 } { \sqrt { d } } \nabla _ { q } ( p _ { j } \boldsymbol { c } _ { j } ) = \displaystyle \frac { 1 } { \sqrt { d } } ( \boldsymbol { c } _ { j } ( \nabla _ { q } p _ { j } ) ^ { \top } + p _ { j } \nabla _ { q } \boldsymbol { c } _ { j } ) } } \\ { { { } } } \\ { { { } = \displaystyle \frac { 1 } { \sqrt { d } } ( \boldsymbol { c } _ { j } ( \nabla _ { q } p _ { j } ) ^ { \top } - p _ { j } \nabla _ { q } \bar { k } ) } } \\ { { { } = \displaystyle \frac { p _ { j } } { d } ( \boldsymbol { c } _ { j } \boldsymbol { c } _ { j } ^ { \top } - ( \sum _ { n = 1 } ^ { N } p _ { n } \boldsymbol { k } _ { n } \boldsymbol { k } _ { n } ^ { \top } - \bar { k } \bar { k } ^ { \top } ) ) } } \\ { { { } ~ = \displaystyle \frac { p _ { j } } { d } [ ( \boldsymbol { k } _ { j } - \bar { k } ) ( \boldsymbol { k } _ { j } - \bar { k } ) ^ { \top } - \sum _ { n = 1 } ^ { N } p _ { n } ( \boldsymbol { k } _ { n } - \bar { k } ) ( \boldsymbol { k } _ { n } - \bar { k } ) ^ { \top } ] . } } \end{array}\tag{B.20}
$$

Restoring the window index and evaluating at $\bar { \pmb q } _ { g }$ gives $p _ { j }  p _ { g , j }$ and $\begin{array} { r } { \bar { k } \to \bar { k } _ { g } = \sum _ { n = 1 } ^ { N } p _ { g , n } k _ { n } } \end{array}$ This yields the following closed-form Hessian, which is identical to Eq. (11) in Sec. 3.3.

$$
H _ { g , j } = \frac { p _ { g , j } } { d } [ ( k _ { j } - \bar { k } _ { g } ) ( k _ { j } - \bar { k } _ { g } ) ^ { \top } - \sum _ { n = 1 } ^ { N } p _ { g , n } ( k _ { n } - \bar { k } _ { g } ) ( k _ { n } - \bar { k } _ { g } ) ^ { \top } ] .\tag{B.21}
$$

Contracting Hessian with Query Covariance. We first denote the diagonal entries of $\Sigma _ { g }$ by $\sigma _ { g , c } ^ { 2 } ,$ where $c = 1 , \ldots , d ,$ which also corresponds to the channel-wise variances as:

$$
\sigma _ { g , c } ^ { 2 } = \frac { 1 } { w ^ { 2 } } et { } { ' } \sum _ { i = 1 } \epsilon _ { g , i , c } ^ { 2 } .\tag{B.22}
$$

Since cross-channel correlations among the query deviations are typically weak, the off-diagonal covariance terms can be neglected. To further reduce sensitivity to channel-wise variance fluctuations, we approximate the covariance with an isotropic form:

$$
\Sigma _ { g } \approx \sigma _ { g } ^ { 2 } { \cal I } , \quad \sigma _ { g } ^ { 2 } = \frac { 1 } { d } \sum _ { c = 1 } ^ { d } \sigma _ { g , c } ^ { 2 } .\tag{B.23}
$$

Substituting Eq. (B.23) and Eq. (B.20) into Eq. (B.13), and adopting the identity $\operatorname { T r } ( \pmb { x } \pmb { x } ^ { \top } ) = \| \pmb { x } \| _ { 2 } ^ { 2 }$ the approximated discrepancy gives:

$$
\begin{array} { l } { \displaystyle \Delta _ { g , j } \approx \frac { \sigma _ { g } ^ { 2 } } { 2 } \mathrm { T r } ( H _ { g , j } ) } \\ { \displaystyle = \frac { p _ { g , j } \sigma _ { g } ^ { 2 } } { 2 d } \mathrm { T r } [ ( k _ { j } - \bar { k } _ { g } ) ( k _ { j } - \bar { k } _ { g } ) ^ { \top } - \sum _ { n = 1 } ^ { N } p _ { g , n } ( k _ { n } - \bar { k } _ { g } ) ( k _ { n } - \bar { k } _ { g } ) ^ { \top } ] } \\ { \displaystyle = \frac { p _ { g , j } } { 2 d } ( \sigma _ { g } ^ { 2 } \| k _ { j } - \bar { k } _ { g } \| _ { 2 } ^ { 2 } - \sum _ { n = 1 } ^ { N } p _ { g , n } \sigma _ { g } ^ { 2 } \| k _ { n } - \bar { k } _ { g } \| _ { 2 } ^ { 2 } ) } \\ { \displaystyle \equiv \frac { p _ { g , j } } { 2 d } ( r _ { g , j } - \bar { r } _ { g } ) , } \end{array}\tag{B.24}
$$

where we define:

$$
r _ { g , j } = \sigma _ { g } ^ { 2 } \Vert \pmb { k } _ { j } - \bar { \pmb { k } } _ { g } \Vert _ { 2 } ^ { 2 } , \quad \bar { r } _ { g } = \sum _ { n = 1 } ^ { N } p _ { g , n } r _ { g , n } .\tag{B.25}
$$

Calibrated Importance Score. Revisiting the global calibration target in Eq. (B.7), the approximated window-wise discrepancy in Eq. (B.24) yields the calibrated window-level score:

$$
\tilde { p } _ { g , j } = p _ { g , j } + \lambda \frac { p _ { g , j } } { 2 d } ( r _ { g , j } - \bar { r } _ { g } ) ,\tag{B.26}
$$

where λ controls the correction strength. Finally, the overall calibrated importance score formulates:

$$
\tilde { s } _ { j } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \tilde { p } _ { g , j } .\tag{B.27}
$$

This corresponds to Eq. (13) in Sec. 3.3, augmenting the efficient pooled attention with a modest additional second-order correction to improve token discriminability for more reliable pruning.

## B.2 DERIVATION OF THE FAST SCORING OPERATOR

Simplified Window-Wise Correction. We further simplify Eq. (13), equivalently given by Eq. (B.26), in preparation for practical implementation.

First, the centered squared distance $r _ { g , j }$ can be expanded as:

$$
r _ { g , j } = \sigma _ { g } ^ { 2 } \| \pmb { k } _ { j } - \bar { \pmb { k } } _ { g } \| _ { 2 } ^ { 2 } = \sigma _ { g } ^ { 2 } ( \| \pmb { k } _ { j } \| _ { 2 } ^ { 2 } - 2 \pmb { k } _ { j } ^ { \top } \bar { \pmb { k } } _ { g } + \| \bar { \pmb { k } } _ { g } \| _ { 2 } ^ { 2 } ) .\tag{B.28}
$$

Then, given that $\begin{array} { r } { \sum _ { n = 1 } ^ { N } p _ { g , n } = 1 } \end{array}$ and $\begin{array} { r } { \sum _ { n = 1 } ^ { N } p _ { g , n } \pmb { k } _ { n } = \bar { \pmb { k } } _ { g } , } \end{array}$ , the weighted mean distance $\bar { r } _ { g }$ is simplified as:

$$
\begin{array} { r l } & { \bar { r } _ { g } = \displaystyle \sum _ { n = 1 } ^ { N } p _ { g , n } r _ { g , n } } \\ & { \quad = \sigma _ { g } ^ { 2 } ( \sum _ { n = 1 } ^ { N } p _ { g , n } \| \boldsymbol { k } _ { n } \| _ { 2 } ^ { 2 } - 2 ( \sum _ { n = 1 } ^ { N } p _ { g , n } k _ { n } ) ^ { \top } \bar { \boldsymbol { k } } _ { g } + \| \bar { \boldsymbol { k } } _ { g } \| _ { 2 } ^ { 2 } \sum _ { n = 1 } ^ { N } p _ { g , n } ) } \\ & { \quad = \sigma _ { g } ^ { 2 } ( \sum _ { n = 1 } ^ { N } p _ { g , n } \| \boldsymbol { k } _ { n } \| _ { 2 } ^ { 2 } - \| \bar { \boldsymbol { k } } _ { g } \| _ { 2 } ^ { 2 } ) . } \end{array}\tag{B.29}
$$

Subtracting Eq. (B.29) from Eq. (B.28) and substituting the result into Eq. (B.26) ultimately yields:

$$
\tilde { p } _ { g , j } = p _ { g , j } [ 1 + \frac { \lambda \sigma _ { g } ^ { 2 } } { 2 d } ( \| \pmb { k } _ { j } \| _ { 2 } ^ { 2 } - 2 \pmb { k } _ { j } ^ { \top } \bar { \pmb { k } } _ { g } + 2 \| \bar { \pmb { k } } _ { g } \| _ { 2 } ^ { 2 } - \sum _ { n = 1 } ^ { N } p _ { g , n } \| \pmb { k } _ { n } \| _ { 2 } ^ { 2 } ) ] .\tag{B.30}
$$

This is algebraically equivalent to the TCAM formulation. It eliminates the explicit computation of $\bar { r } _ { g } .$ , requiring only the window-level statistics $\bar { k } _ { g }$ and $\begin{array} { r } { u _ { g } = \sum _ { n = 1 } ^ { N } p _ { g , n } \| \pmb { k } _ { n } \| _ { 2 } ^ { 2 } } \end{array}$ , both of which can be accumulated online as in Algo. 1.

Exactness of Online Softmax. To establish the basis for the tiled implementation, we first review the exactness of online Softmax. Without loss of generality, we consider a single query q within window g and let $\boldsymbol { z } = \{ z _ { j } \} _ { j = 1 } ^ { N }$ denote its attention logits.

Softmax is invariant to a uniform shift of its logits, i.e. for any scalar $c ,$

$$
\operatorname { S o f t m a x } ( z ) _ { j } = { \frac { \exp ( z _ { j } - c ) } { \sum _ { n = 1 } ^ { N } \exp ( z _ { n } - c ) } } .\tag{B.31}
$$

In particular, choosing $c = \operatorname* { m a x } _ { j } z _ { j }$ improves numerical stability without changing the resulting probabilities. Online Softmax exploits this property by dynamically updating the reference maximum as key blocks are processed.

Specifically, consistent with the notations in Algo. 1, let the key sequence K be partitioned into $T _ { K }$ blocks, each of size $B _ { K } \times d .$ After processing the first b blocks, define the running maximum m<sub>b</sub> and the normalization factor $\ell _ { b }$ as:

$$
m _ { b } = \underset { j = 1 , \ldots , b B _ { K } } { \operatorname* { m a x } } z _ { j } , \quad \ell _ { b } = \sum _ { j = 1 } ^ { b B _ { K } } \exp ( z _ { j } - m _ { b } ) .\tag{B.32}
$$

When processing a new block, the running maximum changes from $m _ { b - 1 }$ to $m _ { b } .$ , and all previously accumulated terms can then be exactly rescaled to the new reference maximum using:

$$
\alpha = \exp ( m _ { b - 1 } - m _ { b } ) ,\tag{B.33}
$$

yielding:

$$
\ell _ { b } = \alpha \ell _ { b - 1 } + \sum _ { j = ( b - 1 ) B _ { K } + 1 } ^ { b B _ { K } } \exp ( z _ { j } - m _ { b } ) .\tag{B.34}
$$

This recurrence preserves the exact normalization throughout block-wise accumulation. After the first pass till the final block, the resulting $m _ { T _ { K } }$ and $\ell _ { T _ { K } }$ recover each attention probability as:

$$
p _ { j } = \frac { \exp ( z _ { j } - m _ { T _ { K } } ) } { \ell _ { T _ { K } } } = \mathrm { S o f t m a x } ( z ) _ { j } .\tag{B.35}
$$

Thus, tiled online computation is algebraically equivalent to standard Softmax, differing only in the accumulation procedure.

Online Accumulation of TCAM Statistics. Given the shift invariance of Eq. (B.31), any statistic aggregated with normalized Softmax weights is likewise invariant to a uniform logit shift, $e . g .$ for arbitrary $\{ a _ { j } \} _ { j = 1 } ^ { N }$ and scalar c,

$$
{ \sum } _ { j = 1 } ^ { N } p _ { j } { \bf { a } } _ { j } = \frac { { \sum } _ { j = 1 } ^ { N } \exp ( z _ { j } - c ) { \bf { a } } _ { j } } { { \sum } _ { n = 1 } ^ { N } \exp ( z _ { n } - c ) } .\tag{B.36}
$$

Building on this, the two attention-weighted statistics required by TCAM, i.e. $\begin{array} { r } { \bar { \pmb { k } } _ { g } = \sum _ { j = 1 } ^ { N } p _ { g , j } \pmb { k } _ { j } } \end{array}$ and $\begin{array} { r } { \sum _ { n = 1 } ^ { N } p _ { g , n } \| \pmb { k } _ { n } \| _ { 2 } ^ { 2 } } \end{array}$ , can be accumulated jointly within the first online pass. Concretely, after processing the first b key blocks, we maintain the corresponding unnormalized running statistics:

$$
v _ { b } = \sum _ { j = 1 } ^ { b B _ { K } } \exp ( z _ { j } - m _ { b } ) { k _ { j } } , \quad u _ { b } = \sum _ { j = 1 } ^ { b B _ { K } } \exp ( z _ { j } - m _ { b } ) \| { k _ { j } } \| _ { 2 } ^ { 2 } .\tag{B.37}
$$

Both follow the same rescaling rule when the running maximum changes, i.e.,

$$
\begin{array} { l } { { \pmb v _ { b } = \alpha _ { b } { \pmb v } _ { b - 1 } + \sum _ { j = ( b - 1 ) B _ { K } + 1 } ^ { b B _ { K } } \exp ( z _ { j } - m _ { b } ) { \pmb k } _ { j } , } } \\ { { \pmb u _ { b } = \alpha _ { b } { \pmb u } _ { b - 1 } + \sum _ { j = ( b - 1 ) B _ { K } + 1 } ^ { b B _ { K } } \exp ( z _ { j } - m _ { b } ) \| { \pmb k } _ { j } \| _ { 2 } ^ { 2 } . } } \end{array}\tag{B.38}
$$

After the final block, normalization by $\ell _ { T _ { K } }$ exactly recovers the required statistics as:

$$
\bar { \pmb { k } } _ { g } = \frac { \pmb { v } _ { T _ { K } } } { \ell _ { T _ { K } } } = \sum _ { j = 1 } ^ { N } p _ { g , j } \pmb { k } _ { j } , \quad \sum _ { n = 1 } ^ { N } p _ { g , n } \| \pmb { k } _ { n } \| _ { 2 } ^ { 2 } = \frac { \ b { u } _ { T _ { K } } } { \ell _ { T _ { K } } } .\tag{B.39}
$$

With these finalized statistics, a second traversal over the key blocks recomputes the exact probabilities and directly applies the calibration, yielding the final TCAM scores. Consequently, neither the full attention matrix nor the centered-distance matrix needs to be materialized.