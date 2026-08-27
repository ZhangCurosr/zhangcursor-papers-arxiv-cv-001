# A Visual Dependence-Aware Framework for Multimodal Unsupervised Continual Post-Training

Kaichen Li<sup>1,2,3</sup>, Zhilin Zhu<sup>4</sup>, Jianhao Huang<sup>1,2,3</sup>, Zhengqin Lai<sup>4</sup>, Baochen Xiong<sup>1,2,3</sup>, Zibo Shao<sup>1,2,3</sup>, Yaguang Song<sup>2</sup>, Linhui Xiao<sup>2</sup>, Xiaoshan Yang<sup>1,2,3</sup>, Changsheng Xu<sup>1,2,3</sup>

<sup>1</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation,

Chinese Academy of Sciences, Beijing, China

<sup>2</sup>Pengcheng Laboratory, Shenzhen, China

<sup>3</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China <sup>4</sup>Harbin Institute of Technology, Shenzhen, China

## Abstract

In this paper, we explore a novel task of Multimodal Unsupervised Continual Post-Training (MU-CPT), enabling deployed MLLMs to continually evolve from streaming unlabeled data. Existing unsupervised post-training methods for MLLMs typically optimize target tokens uniformly, overlooking their heterogeneous visual dependence (VD). However, we reveal that token-level VD is crucial for MU-CPT. Specifically, its structural distortion serves as an indicator of cross-modal catastrophic forgetting and its inherent heterogeneity acts as a compass to guide new-task learning. Leveraging this property, we propose a Visual Dependence-Aware (VDA) framework with two main components. First, Visually Constrained Optimal Transport (VC-OT) formulates the VD structural distortion of old-task VD during new-task learning as an optimal transport problem to mitigate cross-modal forgetting. By designing a region-aware ground cost and a dependence-stratified transport penalty, it prevents global shifts in visual focus, while strictly prohibiting visual reliance from degenerating into language bias. Second, Visually Modulated Adaptation (VMA) exploits VD heterogeneity to emphasize visually grounded new-task learning, promoting new-task plasticity. Together, our method simultaneously maintains old-task stability and new-task plasticity during challenging MU-CPT. Extensive experiments under our MU-CPT setting validate the efectiveness of VDA.

## Introduction

Multimodal Large Language Models (MLLMs) (Li et al. 2024; Bai et al. 2025b,a; Wang et al. 2025) have achieved remarkable progress by integrating visual perception with language understanding, enabling them to handle complex multimodal inputs and instructions. To further improve MLLMs’ performance on specific tasks or user demands, most existing approaches, such as supervised fine-tuning (SFT) and reinforcement learning (RL), still largely rely on carefully curated and human-annotated multimodal data. As the scope and diversity of real-world applications continue to expand, empowering MLLMs through large-scale human annotation becomes prohibitively expensive and dificult to sustain. This has motivated growing interest in unsupervised post-training for MLLMs (Zhou et al. 2024; Wei et al. 2026), which aims to leverage unlabeled multimodal data for model improvement.

![](images/9a6f9adab756fef1e071fe76d318b31941a312a72ee994051594b495b836b6d5.jpg)  
Figure 1: Comparison between (a) Multimodal Continual Instruction-Tuning and (b) Multimodal Unsupervised Continual Post-Training (MU-CPT), where labeled answers are completely unavailable during training.

Although these unsupervised approaches have made important progress (Zhu et al. 2024a; Singh et al. 2026), they are primarily developed under static, closed-world settings, where unlabeled data are collected in advance from a predefined distribution that remains unchanged (Zhu et al. 2024b, 2026b). However, real-world applications of MLLMs are inherently dynamic (Zhou et al. 2024; Zhu et al. 2024a; Xu et al. 2026). MLLMs deployed in open environments continuously encounter evolving scenarios (Zhu et al. 2026a), yielding non-stationary streams of unlabeled multimodal data. Ideally, MLLMs should continually strengthen their capabilities in emerging scenarios while preserving those learned from earlier experience (Shao et al. 2026b,a). However, existing unsupervised post-training methods (Hu et al. 2025) struggle with non-stationary data distributions. Meanwhile, existing multimodal continual instruction-tuning (MCIT) (Chen et al. 2024; Guo et al. 2025) approaches assume access to abundant and expensive labeled multimodal data, as illustrated in Fig. 1(a). Bridging these two paradigms is therefore essential to enable sustained evolution of MLLMs through unlabeled multimodal data streams. To bridge this gap, we formalize a novel setting termed Multimodal Unsupervised Continual Post-Training (MU-CPT). As illustrated in Fig. 1(b), under MU-CPT setting, MLLMs sequentially encounter distinct multimodal tasks without access to the corresponding answers during continual learning. The model is expected to acquire capabilities for each new task and preserve those learned from previous tasks.

![](images/e810204132f8cec5f0249b45dcec24242b9ea11e997340c67241aaf2b049437d.jpg)  
Figure 2: (a) Text tokens in multimodal data exhibit heterogeneous levels of visual dependence, where red and grey denote high and low visual dependence, respectively. (b) Learning new tasks leads to structural distortion of visual dependence on old tasks.

Existing unsupervised post-training research has primarily concentrated on how to construct optimization target sequences (Zhu et al. 2024a; Wei et al. 2026; Yu et al. 2026). However, these approaches invariably treat the constructed target sequences as monolithic entities, implicitly optimizing all target tokens uniformly, thereby leaving a fine-grained property intrinsic to the learning signals themselves largely unexamined: token-level Visual Dependence (VD) heterogeneity (Ye et al. 2026). By measuring how much real visual input increases a token’s log-likelihood relative to an information-free counterfactual visual input, we observe that text tokens in multi-modal data naturally bifurcate into two regimes (see Figure 2 (a)): visually grounded entities exhibit pronounced VD, whereas language-driven contextual markers exhibit negligible VD. Crucially, as shown in Figure 2(b), we further identify that sequential unsupervised adaptation induces a severe failure mode which we formalize as crossmodal attributional forgetting: the once-concentrated VD on visually grounded tokens (e.g., "photo", "phone") is progressively flattened and redistributed toward language-driven tokens (e.g., "object", "and"), and this structural drift co-occurs with the model abandoning the correct, visually grounded answer for an hallucinated one.

Motivated by these observations, we propose a novel Visual Dependence-Aware (VDA) framework, leveraging token-level VD as a unified basis to harmonize old-task stability and new-task plasticity, a fundamental challenge in continual learning (Parisi et al. 2018; De Lange et al. 2021).

Our framework comprises two synergistic components. First, we innovatively formulate the mitigation ofcross-modal attributional forgetting as a Visually Constrained Optimal Transport (VC-OT) problem, which jointly preserves the overall strength and the structural organization of visual dependencies. Specifically, we explicitly structure the optimal transport matrix through two novel mechanisms: a region-aware cost that regulates internal VD redistribution based on visual spatial similarity, and a cross-set penalty that strictly blocks the erroneous transport of visual reliance toward vision-independent tokens. Beyond preserving old-task visual dependencies, efective continual learning also requires suficient plasticity to acquire new capabilities. Accordingly, we further introduce Visually Modulated Adaptation (VMA), which promotes new-task learning by emphasizing visually grounded tokens while attenuating task-specific language patterns that may interfere with previous tasks. Consequently, these designs enable our framework to maintain strong stability without compromising plasticity throughout the MU-CPT process.

Our contributions are summarized as follows: (1) We formalize Multimodal Unsupervised Continual Post-Training for MLLMs (MU-CPT), a novel and practical setting that enables MLLMs to continuously evolve from non-stationary, unlabeled multimodal data streams. (2) We uncover the phenomenon of cross-modal attributional forgetting caused by token-level visual dependence distortion, establishing VD as a unified signal to balance stability and plasticity. (3) We propose the VDA framework, introducing Visually Constrained Optimal Transport to preserve structural visual grounding and Visually Modulated Adaptation to mitigate linguistic bias. (4) Comprehensive experiments across diverse multimodal tasks validate the efectiveness of our method, setting a strong baseline for unsupervised MLLM evolution.

## Related Work

Continual Learning. Continual learning (CL) seeks to balance new-task plasticity and old-task stability, with representative solutions based on regularization (Kirkpatrick et al. 2017; Li and Hoiem 2017), rehearsal (Rebufi et al. 2017; Lopez-Paz and Ranzato 2017), and parameter isolation (Mallya and Lazebnik 2017); continual self-supervised learning further extends CL to unlabeled streams, mainly for unimodal representation learning (Madaan et al. 2021). Multimodal CL subsequently studies sequential VQA and prompt-based adaptation (Zhang, Zhang, and Xu 2023; Qian et al. 2023). More recently, continual instruction tuning (CIT) extends CL to instruction-following MLLMs trained sequentially on labeled image–instruction–answer data, with EProj and CoIN establishing representative benchmarks (He et al. 2026; Chen et al. 2024). Existing CIT methods mitigate forgetting through selective attention distillation in SEEKR-MLLM (He et al. 2024), global–local expert adaptation in CL-MoE (Huai et al. 2025), old-task-important LoRA regularization in SEFE (Chen et al. 2025), or replay-augmented gradient guidance in DGG (Li et al. 2025). However, these methods rely on labeled answers to define learning and preservation targets, whereas MU-CPT must balance plasticity and stability using only answer-unlabeled multimodal

## streams.

Unsupervised Post-Training. Unsupervised post-training adapts models to unlabeled inputs using self-generated learning signals. For LLMs, LSMI selects high-confidence solutions through self-consistency (Huang et al. 2023), ScPO constructs preferences from response consistency (Prasad et al. 2024), and TLM minimizes input perplexity (Hu et al. 2025). Related studies also exploit semantic or tokenlevel entropy (Zhang et al. 2026; Agarwal et al. 2026). For MLLMs, SeVa contrasts responses generated from original and perturbed images (Zhu et al. 2024a), CSR calibrates self-rewards using image–response relevance (Zhou et al. 2024), MM-UPT derives pseudo-rewards through majority voting (Wei et al. 2026), and TTRV constructs rewards from response frequency and output entropy (Singh et al. 2026); CSRS further stabilizes self-rewarding through retraced sampling and softened frequency rewards (Yu et al. 2026). Although these methods provide diverse unsupervised objectives, they mainly assume static unlabeled datasets and overlook both continual forgetting and token-level VD heterogeneity.

## Methodology

## Problem Formulation

We formulate Multimodal Unsupervised Continual Post-Training (MU-CPT), enabling models to continually evolve on unlabeled multimodal data while preserving previously acquired capabilities, unlike standard continual learning that relies on fully annotated data. Formally, under this setting, we consider an instruction-tuned MLLM, initially parameterized by $\theta _ { 0 } ,$ , continually learning from K multimodal tasks. At stage k, the model only accesses an answer-unlabeled multimodal data set $\mathcal { D } _ { k } \overset { ^ { \cdot } } { = } \{ ( I _ { k } ^ { i } , Q _ { k } ^ { i } ) \} _ { i = 1 } ^ { N _ { k } }$ , where $I _ { k } ^ { i }$ and $Q _ { k } ^ { i }$ denote the i-th image and its associated question, respectively. The overall objective is to acquire capabilities for the current task from $\mathcal { D } _ { k }$ while preventing catastrophic forgetting on previously learned tasks, yielding the updated parameters $\theta _ { k }$ . Upon completing stage k, the updated model $\theta _ { k }$ is evaluated on held-out image-question pairs from all tasks encountered so far. Regardless of the specific form of the unsupervised signal, we denote the optimization target sequence of length L for an (I, Q) pair as $\dot { Y } = ( y _ { 1 } , \dotsc , y _ { L } )$ where $t \in \{ 1 , \ldots , L \}$ indexes the target-token position, and let $\ell _ { t } ( \theta )$ denote the loss associated with token $y _ { t } .$

## Framework Overview

While existing unsupervised methods mainly construct sequence-level targets, they typically optimize all target tokens uniformly, overlooking token-level visual dependence (VD) heterogeneity. In continual learning, this variation provides an intrinsic anchor for balancing stability and plasticity. Accordingly, VDA addresses MU-CPT at the token level rather than only at the sequence level, as illustrated in Figure 3.

To maintain old-task stability, Visually Constrained Optimal Transport (VC-OT) formulates old-task VD distortion as an optimal transport problem. Its region-aware cost and dependence-stratified penalty suppress global shifts in visual focus and transfer toward vision-independent tokens. To preserve new-task plasticity, Visually Modulated Adaptation (VMA) emphasizes visually grounded tokens while attenuating task-specific language patterns that cause cross-task interference. Together, these complementary mechanisms enable robust continual learning from unlabeled multimodal data.

## Token-Level Visual Dependence

To quantify the visual dependence of each target token $y _ { t } \in$ Y, we compare its prediction under real and counterfactual visual conditions. Specifically, keeping the target token and its preceding textual context unchanged, we compute:

$$
\begin{array} { r l } & { \ell _ { t } ^ { \mathrm { r e a l } } ( \theta ) = - \log p _ { \theta } \left( y _ { t } \mid I , \mathcal { P } , y _ { < t } \right) , } \\ & { \ell _ { t } ^ { \mathrm { n u l l } } ( \theta ) = - \log p _ { \theta } \left( y _ { t } \mid I _ { \mathrm { n u l l } } , \mathcal { P } , y _ { < t } \right) , } \end{array}\tag{1}
$$

where $I _ { \mathrm { n u l l } }$ denotes a counterfactual visual input with no information. (See Appendix for details) The token-level visual dependence is then defined as:

$$
D _ { t } ( \theta ) = \frac { \ell _ { t } ^ { \mathrm { n u l l } } ( \theta ) - \ell _ { t } ^ { \mathrm { r e a l } } ( \theta ) } { \ell _ { t } ^ { \mathrm { n u l l } } ( \theta ) + \ell _ { t } ^ { \mathrm { r e a l } } ( \theta ) } .\tag{2}
$$

A positive $D _ { t }$ indicates that the visual input actively facilitates the prediction of the token, whereas a non-positive value suggests that the token is driven primarily by language context.

During continual learning, this token-level metric serves a dual purpose by explicitly bridging visual dependence with the model’s core cross-modal attribution capability. For replayed old-task samples, the structural distortion of VD reflects the collapse of this capability, marking a severe crossmodal attributional forgetting that necessitates VD structural preservation to maintain stability. On the other hand, while pursuing stability alone risks suppressing model adaptability, VD heterogeneity acts as an intrinsic compass to locate visually informative tokens. By directing the optimization focus toward these essential visual cues rather than task-specific language biases, it efectively fosters new-task plasticity. This dual functionality establishes VD as a unified foundation for our subsequent designs.

## Visually Constrained Optimal Transport

Preserving old-task VD is essential to prevent cross-modal attributional forgetting. To this end, we formulate the structural drift between the reference and current old-task VD as a unified optimal transport problem. By leveraging the highly non-uniform damage that VD redistribution in diferent directions inflicts on the model’s cross-modal attribution capability, we introduce a region-aware transport cost and dependence-stratified transport penalty to safely regulate internal VD structural shifts. Furthermore, we explicitly constrain the overall VD strength to prevent the global collapse of visual reliance.

VD Transport Formulation. Since only $D _ { t } > 0$ signifies that visual evidence actively supports token generation, we focus exclusively on positive visual dependence. To extract this non-negative mass for optimal transport while preserving smooth gradients, we smoothly approximate the hard max(·, 0) truncation with a low-temperature Softplus, projecting VD into nonnegative mass:

![](images/be4385e487268c3c39e08fa92cbf0aaafdf0c42c1a0e2e133dec35e86d8808af.jpg)  
Figure 3: Overview of the proposed Visual Dependence-Aware (VDA) framework. (a) Visually Constrained Optimal Transport (VC-OT) preserves old-task VD strength and structure through region-aware and dependence-stratified transport. (b) Visually Modulated Adaptation (VMA) reweights incoming-task tokens according to their VD to promote visually grounded learning. Together, they efectively balance stability and plasticity in MU-CPT.

$$
r _ { t } ^ { m } = \tau _ { D } \mathrm { S o f t p l u s } \left( \frac { D _ { t } ^ { m } } { \tau _ { D } } \right) , \qquad m \in \{ \mathrm { r e f } , \mathrm { c u r } \} .\tag{3}
$$

Here, a size-bounded bufer retains a small subset of old-task samples and their reference VD. For each replayed sample, $D _ { t } ^ { \mathrm { r e f } }$ is computed by the model immediately after learning its source task and cached in the bufer, while $\mathbf { \dot { D } } _ { t } ^ { \mathrm { c u r } }$ is computed by the current trainable model. While this absolute mass $r _ { t } ^ { m }$ will be explicitly constrained later to maintain the overall VD strength, modeling its internal structural flow requires a relative mass distribution. Therefore, we normalize $\boldsymbol { r } _ { t } ^ { m }$ within the target sequence:

$$
\tilde { r } _ { t } ^ { m } = \frac { r _ { t } ^ { m } } { \sum _ { u = 1 } ^ { L } r _ { u } ^ { m } } , \qquad m \in \{ \mathrm { r e f , c u r } \} .\tag{4}
$$

where L represents the length of the target sequence. We treat $\tilde { \mathbf { r } } ^ { \mathrm { r e f } }$ as the source distribution and $\tilde { \mathbf { r } } ^ { \mathrm { c u r } }$ as the target distribution. Specifically, $T _ { i j }$ denotes the VD mass transported from source token i under the reference model to destination token j under the current model, satisfying $\begin{array} { r } { \sum _ { j = 1 } ^ { L } T _ { i j } = \tilde { r } _ { i } ^ { \mathrm { r e f } } } \end{array}$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { L } T _ { i j } = \tilde { r } _ { j } ^ { \mathrm { c u r } } } \end{array}$ . To set up the transport space, we explicitly separate tokens attributed to visual evidence from those driven by language context, partitioning the target sequence into a visually attributed set $\mathcal { H } = \{ t \ | \ \mathbf { \bar { \Gamma } } D _ { t } ^ { \mathrm { r e f } } > \ \mathbf { \bar { 0 } } \}$ and a vision-independent set $\mathcal { L } = \{ t \ | \ D _ { t } ^ { \mathrm { r e f } } \ \stackrel { \ } { = } \ 0 \}$ . Given that the reference mass in L is negligible and does not represent any visual attribution that needs to be preserved, we focus on the structural regularization on transport originating from H. Accordingly, we consider two relevant transport paths: intra-set redistribution within H and cross-set transfer from H to ${ \mathcal { L } } .$

Region-Aware Transport Cost. For the intra-set transport within H, our primary concern is to prevent global shifts in visual focus, where the model’s attention drifts entirely from one region to an unrelated area. This drift signifies a severe forgetting of how the model comprehends the image. Therefore, we permit mild redistribution among visually similar tokens while penalizing such global shifts. Let $\mathbf { a } _ { t , \mathrm { r e f } } ^ { \ell }$ denote the normalized attention distribution of token $t \in \{ 1 , \ldots , L \}$ over all visual tokens at decoder layer ℓ, extracted by the model immediately after learning the sample’s source task and cached with the replay sample. We formulate the regionaware transport cost between tokens i and j as:

$$
d _ { \mathrm { r e g } } ( i , j ) = \frac { 1 } { | \mathcal { G } | } \sum _ { \ell \in \mathcal { G } } \mathrm { J S D } \left( \mathbf { a } _ { i , \mathrm { r e f } } ^ { \ell } , \mathbf { a } _ { j , \mathrm { r e f } } ^ { \ell } \right) .\tag{5}
$$

Here, G specifies the selected intermediate decoder layers. $J S D \in [ 0 , 1 ]$ represents Jensen-Shannon Divergence, which is used to measure the discrepancy in attended regions, a greater diference directly yields a higher transport cost.

Dependence-Stratified Transport Penalty. We next consider cross-set transport from H to L. Since tokens in L are vision-independent under the reference model, assigning VD mass to these positions indicates spurious visual attribution to language-driven tokens. We therefore penalize $\mathcal { H }  \mathcal { L }$ transport with the maximum JSD cost of 1, corresponding to completely non-overlapping visual attention distributions.

Structural Transport Objective. Combining the regionaware intra-set cost and the dependence-stratified cross-set penalty, we define the transport cost matrix as:

$$
C _ { i j } = \left\{ \begin{array} { l l } { d _ { \mathrm { { r e g } } } ( i , j ) , } & { i \in \mathcal { H } , j \in \mathcal { H } , } \\ { 1 , } & { i \in \mathcal { H } , j \in \mathcal { L } , } \\ { 0 , } & { i \in \mathcal { L } . } \end{array} \right.\tag{6}
$$

Since transport originating from $\mathcal { L }$ carries negligible reference mass and no meaningful visual attribution, we set its cost to zero. Let $\mathbf { T } ^ { * }$ denote the optimal transport plan obtained through Sinkhorn iterations. The structural alignment loss is:

$$
\mathcal { L } _ { \mathrm { s t r u c t } } = \langle \mathbf { T } ^ { * } , \mathbf { C } \rangle = \sum _ { i = 1 } ^ { L } \sum _ { j = 1 } ^ { L } T _ { i j } ^ { * } C _ { i j } .\tag{7}
$$

VD Strength Preservation. While $\mathcal { L } _ { \mathrm { s t r u c t } }$ regulates the internal VD organization, learning new tasks may still reduce the overall VD strength. We therefore match the total positive VD mass of the current model to its reference:

$$
\mathcal { L } _ { \mathrm { s t r e n g t h } } = \frac { \left| \sum _ { t = 1 } ^ { L } \boldsymbol { r } _ { t } ^ { \mathrm { c u r } } - \sum _ { t = 1 } ^ { L } \boldsymbol { r } _ { t } ^ { \mathrm { r e f } } \right| } { \sum _ { t = 1 } ^ { L } \boldsymbol { r } _ { t } ^ { \mathrm { r e f } } } .\tag{8}
$$

The complete VC-OT objective is:

$$
{ \mathcal { L } } _ { \mathrm { V C - O T } } = { \mathcal { L } } _ { \mathrm { s t r u c t } } + { \mathcal { L } } _ { \mathrm { s t r e n g t h } } .\tag{9}
$$

## Visually Modulated Adaptation

Although VC-OT preserves old-task knowledge, antiforgetting constraints alone can restrict new-task plasticity (Jung et al. 2020). To balance the two, we introduce Visually Modulated Adaptation (VMA). During new-task training, visually attributed tokens carry task-relevant visual cues, whereas vision-independent tokens often encode taskspecific linguistic templates, such as multiple-choice formats. Overfitting to these templates can induce language bias and cross-task interference; for example, the model may generate option labels for previously learned open-ended questions. VMA modulates each token’s learning signal according to its visual dependence, thereby emphasizing visually grounded knowledge to improve plasticity while suppressing language-bias interference.

Let $\bar { \ell } _ { t } ^ { \mathrm { u } } ( \theta )$ denote the unmodulated token-level loss associated with target token $y _ { t }$ . Based on its current visual dependence defined in Eq. 2, we first compute:

$$
\begin{array} { l } { { \widetilde { w } _ { t } = 1 + \lambda _ { \mathrm { V M A } } \mathrm { R e L U } \left( \mathrm { s g } [ D _ { t } ( \theta ) ] \right) , } } \\ { { \displaystyle w _ { t } = \frac { L \widetilde { w } _ { t } } { \sum _ { u = 1 } ^ { L } \widetilde { w } _ { u } } } , } \end{array}\tag{10}
$$

where sg[·] denotes the stop-gradient operation and $\lambda _ { \mathrm { V M A } }$ controls the modulation strength. The unnormalized modulation factor $\widetilde { w } _ { t }$ amplifies the learning intensity strictly for visually attributed tokens, while retaining a base unit contribution for the rest. Sequence-level normalization ensures the average modulation scale remains neutral, avoiding sampledependent optimization shifts.

The modulated current-task objective is then formulated as:

$$
\mathcal { L } _ { \mathrm { V M A } } = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } w _ { t } \ell _ { t } ^ { \mathrm { u } } ( \theta ) .\tag{11}
$$

By adaptively modulating the relative contribution of visually attributed tokens, VMA encourages the model to absorb efective visual knowledge from the incoming task, while attenuating the influence of task-specific linguistic formatting. Consequently, VMA complements VC-OT by enhancing new-task plasticity without relying on external supervision. To sum up, the overall loss is given by:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { V M A } } + \mathcal { L } _ { \mathrm { V C - O T } } .\tag{12}
$$

## Experiments

## Experimental Setup

Datasets. We evaluate six tasks spanning OCR, scientific diagrams, financial charts, compositional reasoning, driving scenes, and medical VQA: TextVQA (Singh et al. 2019), SciVQA (Borisova, Rauscher, and Rehm 2025), StockQA (Zhao et al. 2025), GQA (Hudson and Manning 2019), DriveLM (Sima et al. 2024), and PMC-VQA (Zhang et al. 2023). We follow the original splits and use only image–question pairs for continual training. All methods use the same random order TextVQA→SciVQA→StockQA→GQA→DriveLM→PMC-VQA (See Appendix for more random orders) and are evaluated on all seen test sets after each stage.

Comparison Methods. We compare with six unsupervised post-training methods—LSMI (Huang et al. 2023), SeVa (Zhu et al. 2024a), ScPO (Prasad et al. 2024), TLM (Hu et al. 2025), MM-UPT (Wei et al. 2026), and TTRV (Singh et al. 2026)—and four MLLM continual-learning methods— DGG (Li et al. 2025), SEFE (Chen et al. 2025), CL-MoE (Huai et al. 2025), and SEEKR-MLLM (He et al. 2024).

Implementation Details. We use Qwen2.5-VL-7B (Bai et al. 2025b) as the primary backbone (See appendix for other model families). For VDA and all continual learning baselines, we perform parameter-eficient tuning using LoRA with a rank of 128. For the unsupervised post-training baselines, we follow the training configurations provided in their oficial implementations. Following TLM (Hu et al. 2025), we use image-conditioned question modeling as the basic unsupervised objective, where the model autoregressively predicts question tokens conditioned on the image and a taskagnostic instruction as in (Zhao et al. 2024). Our VDA is not tied to this objective, with results using other unsupervised signals reported in the appendix. As to other hyperparameters, we set the VD smoothing temperature $\tau _ { D }$ to 0.05, and bufer size to 1000. The reference VD and visual attention of each replay sample is computed once when the sample is inserted into the memory and remains fixed thereafter. Unless otherwise specified, we use AdamW for optimization with a batch size of 1 and a learning rate of $5 \times \mathrm { { i } 0 ^ { - 5 } }$

<table><tr><td>Method</td><td>TextVQA</td><td>SciVQA</td><td>StockQA</td><td>GQA</td><td>DriveLM</td><td>PMC-VQA</td><td>AvgAcc ↑</td></tr><tr><td>Frozen Backbone</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-VL-7B</td><td>79.1</td><td>52.9</td><td>53.3</td><td>60.0</td><td>24.7</td><td>53.5</td><td>53.9</td></tr><tr><td colspan="8">Unsupervised Post-training Methods</td></tr><tr><td>LSMI (EMNLP’23)</td><td>79.5</td><td>63.2</td><td>62.5</td><td>68.3</td><td>29.0</td><td>49.7</td><td>58.7</td></tr><tr><td>SeVa (ACM MM&#x27;24)</td><td>79.9</td><td>57.3</td><td>60.7</td><td>64.0</td><td>27.3</td><td>54.3</td><td>57.2</td></tr><tr><td>ScPO (ICML&#x27;25)</td><td>80.3</td><td>61.7</td><td>64.4</td><td>66.0</td><td>28.6</td><td>51.6</td><td>58.8</td></tr><tr><td>TLM (ICML&#x27;25)</td><td>77.2</td><td>49.3</td><td>43.3</td><td>46.1</td><td>29.8</td><td>42.4</td><td>48.0</td></tr><tr><td>MM-UPT (NeurIPS’25)</td><td>78.8</td><td>54.7</td><td>57.8</td><td>62.4</td><td>24.8</td><td>55.0</td><td>55.6</td></tr><tr><td>TTRV (CVPR&#x27;26)</td><td>79.6</td><td>60.5</td><td>59.2</td><td>64.4</td><td>26.7</td><td>55.9</td><td>57.7</td></tr><tr><td colspan="8">Continual Learning Methods</td></tr><tr><td>SEEKR-MLLM (EMNLP&#x27;24)</td><td>75.4</td><td>61.0</td><td>67.8</td><td>68.8</td><td>31.7</td><td>55.8</td><td>60.1</td></tr><tr><td>CL-MoE (CVPR&#x27;25)</td><td>78.5</td><td>60.5</td><td>62.5</td><td>64.1</td><td>27.7</td><td>54.2</td><td>57.9</td></tr><tr><td>SEFE (ICML&#x27;25)</td><td>79.3</td><td>33.1</td><td>72.5</td><td>56.8</td><td>26.5</td><td>53.1</td><td>53.6</td></tr><tr><td>DGG (CVPR&#x27;26)</td><td>84.4</td><td>54.5</td><td>61.8</td><td>61.7</td><td>28.5</td><td>54.6</td><td>57.6</td></tr><tr><td>VDA (Ours)</td><td>85.9</td><td>63.6</td><td>70.3</td><td>67.6</td><td>32.6</td><td>54.9</td><td>62.5</td></tr></table>

Table 1: Comparison with existing Unsupervised Post-Training methods and Continual learning methods.

Evaluation Metrics. Following standard continuallearning protocols (Wang et al. 2022; Smith et al. 2023), AvgAcc averages final accuracy over all tasks and is our primary stability–plasticity metric. AvgF measures the average performance drop on previous tasks, while AvgLA averages each task’s accuracy immediately after it is learned, diagnosing stability and plasticity, respectively.

<table><tr><td>Method</td><td>AvgAcc ↑</td><td>AvgLA↑</td><td>AvgF↓</td></tr><tr><td>SEEKR-MLLM</td><td>60.1</td><td>62.5</td><td>2.9</td></tr><tr><td>CL-MoE</td><td>57.9</td><td>59.5</td><td>1.8</td></tr><tr><td>SEFE</td><td>53.6</td><td>61.4</td><td>9.4</td></tr><tr><td>DGG</td><td>57.6</td><td>60.8</td><td>3.9</td></tr><tr><td>VDA (Ours)</td><td>62.5</td><td>63.0</td><td>0.6</td></tr></table>

Table 2: Fine-grained comparison of new-task plasticity and old-task stability with continual learning methods.

## Comparison Results

Table 1 reports the final performance after sequential adaptation on six multimodal tasks. VDA achieves the highest AvgAcc of 62.5%, outperforming the strongest unsupervised post-training method, ScPO, by 3.7 points and the strongest continual learning baseline, SEEKR-MLLM, by 2.4 points. Notably, despite learning solely from answerunlabeled multimodal data under continually shifting data distribution, VDA improves the frozen Qwen2.5-VL-7B backbone by 8.6 points, demonstrating its ability to efectively acquire and keep knowledge from non-stationary unlabeled data streams. Across individual tasks, existing baselines exhibit pronounced performance variation, indicating that their adaptation is sensitive to task and domain shifts under answer-free supervision. In contrast, VDA maintains consistently competitive performance across all six benchmarks, demonstrating more robust unsupervised continual learning across heterogeneous multimodal domains.

Furthermore, we conduct a fine-grained comparison with existing supervised CL methods from the perspectives of new-task plasticity and old-task stability, where all methods are adapted to the MU-CPT setting using the same unsupervised learning signal. As shown in Table 2, VDA achieves the highest AvgLA of 63.0% and the lowest forgetting of 0.6, together yielding the best AvgAcc of 62.5%. Among the baselines, SEEKR-MLLM exhibits the strongest newtask plasticity with an AvgLA of 62.5%, but sufers substantially greater forgetting, while CL-MoE attains relatively low forgetting at the cost of a much lower AvgLA of 59.5%. These results demonstrate that VDA improves old-task stability without suppressing new-task adaptation, leading to a more favorable stability–plasticity balance.

<table><tr><td>LVMA</td><td> $\mathcal { L } _ { \mathrm { s t r e n g t h } }$ </td><td> $\mathcal { L } _ { \mathrm { s t r u c t } }$ </td><td></td><td>AvgAcc ↑ AvgLA↑ AvgF↓</td><td></td></tr><tr><td></td><td></td><td></td><td>51.0</td><td>61.7</td><td>12.8</td></tr><tr><td>√</td><td></td><td></td><td>56.9</td><td>63.8</td><td>8.2</td></tr><tr><td>√</td><td>√</td><td></td><td>58.0</td><td>61.5</td><td>4.2</td></tr><tr><td></td><td>√</td><td></td><td>60.1</td><td>62.9</td><td>3.3</td></tr><tr><td></td><td>√</td><td>√</td><td>59.8</td><td>60.3</td><td>0.7</td></tr><tr><td>√</td><td>√</td><td>√</td><td>62.5</td><td>63.0</td><td>0.6</td></tr></table>

Table 3: Ablation study of the components in VDA.

Ablation Study. Table 3 evaluates the contributions of VMA, VD strength preservation, and structural transport. VMA alone increases AvgLA from 61.7 to 63.8 and AvgAcc from 51.0 to 56.9, while reducing AvgF to 8.2, indicating that VD-guided token modulation mainly improves plasticity and partially alleviates forgetting. $\mathcal { L } _ { \mathrm { s t r e n g t h } }$ further lowers AvgF to 4.2 but slightly reduces AvgLA, reflecting the adaptation cost ofstability constraints. Combining it with VMA recovers plasticity and raises AvgAcc to 60.1. Joint strength and structural preservation reduce AvgF to 0.7, but limits AvgLA to 60.3. The complete model achieves the best AvgAcc of 62.5, with AvgLA of 63.0 and AvgF of 0.6. These results show that VMA compensates for the restricted adaptation caused by VD preservation, while the two preservation terms jointly provide strong old-task stability.

![](images/5c482dcfeaa132972670895f32f95e37086c895ade0d41ccf6e75beb57f9abf7.jpg)  
Figure 4: RRAR trajectories on TextVQA throughout continual-learning.

## Detailed Analysis

Analysis of cross-modal comprehension. To examine whether VDA strengthens and preserves cross-modal comprehension throughout continual learning, we track the Relevant Region Attention Ratio (RRAR) (Peng et al. 2026) on TextVQA using its ground-truth bounding boxes provided by(Khayatkhoei et al. 2025). RRAR measures the relative attention allocated to question-relevant visual regions compared with the entire image. As shown in Figure 4, initial adaptation increases RRAR, whereas the baseline (without both VMA and VC-OT) progressively declines as new tasks are learned, eventually falling below the frozen backbone. In contrast, VDA maintains an RRAR of 4.204 after the full sequence, outperforming the baseline by 0.499. Removing VMA or VC-OT reduces the final RRAR to 3.986 and 3.914, respectively, showing that both components contribute to maintaining strong question-relevant visual attribution. These results provide mechanism-level evidence that VDA enhances cross-modal comprehension during adaptation and prevents its degradation across subsequent tasks.

Comparison of VD Preservation Strategies. We further compare VC-OT with several alternative strategies in preserving old-task VD. “No Protection” denotes VDA without VC-OT, while the remaining variants replace VC-OT with token-level L1 matching, or Mass+KL alignment. As shown in Table 4, No Protection exhibit substantial VD drift and severe forgetting, demonstrating the necessity of explicitly preserving old-task VD structures. Token-level L1 achieves the smallest VD drift and the lowest AvgF of 0.4, but only reaches an AvgAcc of 59.1, suggesting that rigid positionwise matching suppresses model’s plasticity. Mass+KL provides a softer distribution-level constraint, yet treats all crosstoken shifts uniformly, without distinguishing visually consistent redistribution from harmful transfer toward vision independent tokens. In contrast, VC-OT achieves comparable VD preservation while using region-aware transport and a dependence-stratified transport penalty to regulate diferent redistribution paths, resulting in a low AvgF and the highest global AvgAcc.

<table><tr><td colspan="5">VD Strength VD Distribution</td></tr><tr><td>VD Protection</td><td>Drift (%) ↓</td><td>Drift ↓</td><td>AvgAcc ↑ AvgF ↓</td><td></td></tr><tr><td>No Protection</td><td>35.94</td><td>0.1052</td><td>56.9</td><td>8.2</td></tr><tr><td>Token-Level L1</td><td>5.08</td><td>0.0187</td><td>59.1</td><td>0.4</td></tr><tr><td>Mass+KL</td><td>14.89</td><td>0.0428</td><td>60.3</td><td>3.4</td></tr><tr><td>VC-OT</td><td>14.27</td><td>0.0385</td><td>62.5</td><td>0.6</td></tr></table>

Table 4: Comparison of diferent VD preservation strategies. VD Strength Drift measures the relative change in total VD. VD Distribution Drift measures the JSD between VD distributions.

![](images/fe61491aed206d66379f9a7cfd89c2c996611a1dd0a5006e31575162526b9957.jpg)

![](images/c8a03f6c38e777d288d7d984c6504dab8842ee66c1111a86b58842cb6fcec438.jpg)  
Figure 5: Hyperparameter Analysis.

Hyperparameter Analysis. Fig. 5 analyzes the efects of the LoRA rank and memory-bufer size. Increasing the LoRA rank initially improves AvgAcc by enhancing new-task adaptation, with the best performance achieved at r=128; further increasing the rank introduces more trainable parameters, leading to greater forgetting and worse AvgAcc. Enlarging the bufer generally improves performance by providing broader coverage of previous tasks, while the gain gradually saturates. A bufer size of 1,000 already achieves a good AvgAcc of 62.5.

## Conclusion

This work introduces Multimodal Unsupervised Continual Post-Training (MU-CPT), where MLLMs continually learn from non-stationary multimodal data without labeled answers. We identify token-level visual dependence (VD) as an intrinsic signal of cross-modal attributional forgetting and propose VDA, which preserves old-task VD through VC-OT and promotes visually grounded adaptation through VMA. Experiments on six tasks show that our proposed framework reduces forgetting while retaining strong plasticity.

Agarwal, S.; Zhang, Z.; Yuan, L.; Han, J.; and Peng, H. 2026. The unreasonable efectiveness of entropy minimization in llm reasoning. Advances in Neural Information Processing Systems, 38: 107150–107180.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025a. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025b. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Borisova, E.; Rauscher, N.; and Rehm, G. 2025. SciVQA 2025: Overview of the first scientific visual question answering shared task. In Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025), 182–210.

Chen, C.; Zhu, J.; Luo, X.; Shen, H. T.; Song, J.; and Gao, L. 2024. Coin: A benchmark of continual instruction tuning for multimodel large language models. Advances in neural information processing systems, 37: 57817–57840.

Chen, J.; Cong, R.; Zhao, Y.; Yang, H.; Hu, G.; Ip, H. H. S.; and Kwong, S. 2025. Sefe: Superficial and essential forgetting eliminator for multimodal continual instruction tuning. arXiv preprint arXiv:2505.02486.

De Lange, M.; Aljundi, R.; Masana, M.; Parisot, S.; Jia, X.; Leonardis, A.; Slabaugh, G.; and Tuytelaars, T. 2021. A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7): 3366–3385.

Guo, H.; Zeng, F.; Xiang, Z.; Zhu, F.; Wang, D.-H.; Zhang, X.-Y.; and Liu, C.-L. 2025. Hide-llava: Hierarchical decoupling for continual instruction tuning of multimodal large language model. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 13572–13586.

He, J.; Guo, H.; Zhu, K.; Tang, M.; and Wang, J. 2026. Continual instruction tuning for large multimodal models. IEEE Transactions on Image Processing.

He, J.; Guo, H.; Zhu, K.; Zhao, Z.; Tang, M.; and Wang, J. 2024. Seekr: Selective attention-guided knowledge retention for continual learning of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 3254–3266.

Hu, J.; Zhang, Z.; Chen, G.; Wen, X.; Shuai, C.; Luo, W.; Xiao, B.; Li, Y.; and Tan, M. 2025. Test-time learning for large language models. arXiv preprint arXiv:2505.20633.

Huai, T.; Zhou, J.; Wu, X.; Chen, Q.; Bai, Q.; Zhou, Z.; and He, L. 2025. Cl-moe: Enhancing multimodal large language model with dual momentum mixture-of-experts for continual visual question answering. In Proceedings of the computer vision and pattern recognition conference, 19608–19617.

Huang, J.; Gu, S.; Hou, L.; Wu, Y.; Wang, X.; Yu, H.; and Han, J. 2023. Large language models can self-improve. In Proceedings ofthe 2023 conference on empirical methods in natural language processing, 1051–1068.

Hudson, D. A.; and Manning, C. D. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6700–6709.

Jung, S.; Ahn, H.; Cha, S.; and Moon, T. 2020. Continual learning with node-importance based adaptive group sparse regularization. Advances in neural information processing systems, 33: 3647–3658.

Khayatkhoei, M.; Chhikara, P.; Ilievski, F.; et al. 2025. Mllms know where to look: Training-free perception of small visual details with multimodal llms. In International Conference on Learning Representations, volume 2025, 68194–68213.

Kirkpatrick, J.; Pascanu, R.; Rabinowitz, N.; Veness, J.; Desjardins, G.; Rusu, A. A.; Milan, K.; Quan, J.; Ramalho, T.; Grabska-Barwinska, A.; et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy ofsciences, 114(13): 3521–3526.

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; et al. 2024. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Li, S.; Gao, M.; Su, T.; Zhang, X.-Y.; and Wang, Z. 2025. Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance. arXiv preprint arXiv:2511.15164.

Li, Z.; and Hoiem, D. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12): 2935–2947.

Lopez-Paz, D.; and Ranzato, M. 2017. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30.

Madaan, D.; Yoon, J.; Li, Y.; Liu, Y.; and Hwang, S. J. 2021. Representational continuity for unsupervised continual learning. arXiv preprint arXiv:2110.06976.

Mallya, A.; and Lazebnik, S. 2017. Packnet: Adding multiple tasks to a single network by iterative pruning. arXiv preprint arXiv:1711.05769.

Parisi, G. I.; Kemker, R.; Part, J. L.; Kanan, C.; and Wermter, S. 2018. Continual lifelong learning with neural networks: A review. arXiv preprint arXiv:1802.07569, 990.

Peng, R.; Wu, X.; Lei, J.; Hou, L.; Ma, Y.; and Li, X. 2026. Deeper Thought, Weaker Aim: Understanding and Mitigating Perceptual Impairment during Reasoning in Multimodal Large Language Models. arXiv:2603.14184.

Prasad, A.; Yuan, W.; Pang, R. Y.; Xu, J.; Fazel-Zarandi, M.; Bansal, M.; Sukhbaatar, S.; Weston, J.; and Yu, J. 2024. Self-consistency preference optimization. arXiv preprint arXiv:2411.04109.

Qian, Z.; Wang, X.; Duan, X.; Qin, P.; Li, Y.; and Zhu, W. 2023. Decouple before interact: Multi-modal prompt learning for continual visual question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2953–2962.

Rebufi, S.-A.; Kolesnikov, A.; Sperl, G.; and Lampert, C. H. 2017. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2001–2010.

Shao, Z.; Xiong, B.; Xu, C.; Xiao, L.; Li, K.; Gong, H.; Li, Y.; Song, Y.; and Yang, X. 2026a. AgentPatch: Coarse-to-Fine Weak-Task Repair for Merging Agentic Multimodal Large Language Models. arXiv preprint arXiv:2608.06699.

Shao, Z.; Xiong, B.; Yang, X.; Song, Y.; Zhang, Q.; Chen, H.; and Xu, C. 2026b. PivotMerge: Bridging Heterogeneous Multimodal Pre-training via Post-Alignment Model Merging. arXiv preprint arXiv:2604.22823.

Sima, C.; Renz, K.; Chitta, K.; Chen, L.; Zhang, H.; Xie, C.; Beißwenger, J.; Luo, P.; Geiger, A.; and Li, H. 2024. Drivelm: Driving with graph visual question answering. In European conference on computer vision, 256–274. Springer.

Singh, A.; Marjit, S.; Lin, W.; Gavrikov, P.; Yeung-Levy, S.; Kuehne, H.; Feris, R.; Doveh, S.; Glass, J.; and Mirza, M. J. 2026. Ttrv: Test-time reinforcement learning for vision language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 33153–33163.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8317–8326.

Smith, J. S.; Karlinsky, L.; Gutta, V.; Cascante-Bonilla, P.; Kim, D.; Arbelle, A.; Panda, R.; Feris, R.; and Kira, Z. 2023. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11909–11919.

Wang, W.; Gao, Z.; Gu, L.; Pu, H.; Cui, L.; Wei, X.; Liu, Z.; Jing, L.; Ye, S.; Shao, J.; Wang, Z.; Chen, Z.; Zhang, H.; Yang, G.; Wang, H.; Wei, Q.; Yin, J.; Li, W.; Cui, E.; Chen, G.; Ding, Z.; Tian, C.; Wu, Z.; Xie, J.; Li, Z.; Yang, B.; Duan, Y.; Wang, X.; Hou, Z.; Hao, H.; Zhang, T.; Li, S.; Zhao, X.; Duan, H.; Deng, N.; Fu, B.; He, Y.; Wang, Y.; He, C.; Shi, B.; He, J.; Xiong, Y.; Lv, H.; Wu, L.; Shao, W.; Zhang, K.; Deng, H.; Qi, B.; Ge, J.; Guo, Q.; Zhang, W.; Zhang, S.; Cao, M.; Lin, J.; Tang, K.; Gao, J.; Huang, H.; Gu, Y.; Lyu, C.; Tang, H.; Wang, R.; Lv, H.; Ouyang, W.; Wang, L.; Dou, M.; Zhu, X.; Lu, T.; Lin, D.; Dai, J.; Su, W.; Zhou, B.; Chen, K.; Qiao, Y.; Wang, W.; and Luo, G. 2025. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Eficiency. arXiv:2508.18265.

Wang, Z.; Zhang, Z.; Ebrahimi, S.; Sun, R.; Zhang, H.; Lee, C.-Y.; Ren, X.; Su, G.; Perot, V.; Dy, J.; et al. 2022. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European conference on computer vision, 631–648. Springer.

Wei, L.; Li, Y.; Wang, C.; Wang, Y.; Kong, L.; Huang, W.; and Sun, L. 2026. First SFT, second RL, third UPT: Continual improving multi-modal LLM reasoning via unsupervised post-training. Advances in Neural Information Processing Systems, 38: 62293–62318.

Xu, C.; Ke, K.; Liu, Z.; Wei, J.; Shao, Z.; Guo, W.; and Yu, C. 2026. EvoMAS: Learning Execution-Time Workflows for Multi-Agent Systems. arXiv preprint arXiv:2605.08769.

Ye, Z.; Li, Q.; Feng, X.; Chen, R.; Li, Z.; Ren, H.; Chen, K.; Tu, D.; and Qin, B. 2026. Not All Tokens See Equally:

Perception-Grounded Policy Optimization for Large Vision-Language Models. arXiv preprint arXiv:2604.01840.

Yu, Y.; Wu, Z.; Chen, Z.; Xu, H.; Liao, Z.; Deng, X.; Liu, Z.; Shi, S.; and Wang, H. 2026. Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling. arXiv preprint arXiv:2604.03647.

Zhang, Q.; Wu, H.; Zhang, C.; Zhao, P.; and Bian, Y. 2026. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. Advances in neural information processing systems, 38: 67345–67372.

Zhang, X.; Wu, C.; Zhao, Z.; Lin, W.; Zhang, Y.; Wang, Y.; and Xie, W. 2023. Pmc-vqa: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415.

Zhang, X.; Zhang, F.; and Xu, C. 2023. Vqacl: A novel visual question answering continual learning setting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19102–19112.

Zhao, H.; Zhou, P.; Gao, D.; Bai, Z.; and Shou, M. Z. 2024. Lova3: Learning to visual question answering, asking and assessment. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Zhao, H.; Zhu, F.; Guo, H.; Wang, M.; Wang, R.; Meng, G.; and Zhang, Z. 2025. Mllm-cl: Continual learning for multimodal large language models. arXiv preprint arXiv:2506.05453.

Zhou, Y.; Fan, Z.; Cheng, D.; Yang, S.; Chen, Z.; Cui, C.; Wang, X.; Li, Y.; Zhang, L.; and Yao, H. 2024. Calibrated self-rewarding vision language models. Advances in Neural Information Processing Systems, 37: 51503–51531.

Zhu, K.; Ge, Z.; Zhao, L.; and Zhang, X. 2024a. Selfsupervised visual preference alignment. arXiv preprint arXiv:2404.10501.

Zhu, Z.; Hong, X.; Ma, Z.; Zhuang, W.; Ma, Y.; Dai, Y.; and Wang, Y. 2024b. Reshaping the online data bufering and organizing mechanism for continual test-time adaptation. In European conference on computer vision, 415–433. Springer.

Zhu, Z.; Ma, Z.; Wang, Y.; Song, Y.; Wang, Y.; and Hong, X. 2026a. Sample-Aware Knowledge Association and Enhancement for Open-Vocabulary Continual Learning. International Journal ofComputer Vision, 134(7): 332.

Zhu, Z.; Wang, Y.; Ma, Z.; Song, Y.; Wang, Y.; and Hong, X. 2026b. Dance Across Shifts: Forward-Facilitation Continual Test-Time Adaptation through Dynamic Style Bridging. arXiv preprint arXiv:2605.18608.