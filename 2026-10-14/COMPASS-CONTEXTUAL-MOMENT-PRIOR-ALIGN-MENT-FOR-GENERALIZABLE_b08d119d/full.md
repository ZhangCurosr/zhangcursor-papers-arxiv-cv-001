# COMPASS: CONTEXTUAL MOMENT PRIOR ALIGN-MENT FOR GENERALIZABLE AIGI DETECTION

Zhan Yang<sup>1∗</sup> Tiancheng Shen<sup>2∗</sup> Weijie Lyu<sup>2</sup> Shuo Chen<sup>2</sup> Bernard Ghanem<sup>3</sup> Yibo Yang<sup>4</sup> Ming-Hsuan Yang<sup>2</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>University of California, Merced <sup>3</sup>KAUST <sup>4</sup>Shanghai Jiao Tong University

<sup>∗</sup>Equal contribution.

## ABSTRACT

Generalizable AI-generated image (AIGI) and deepfake detection remains challenging due to the asymmetry phenomenon, where adaptation to limited training distributions can degrade the transferable representations of Vision Foundation Models (VFMs). Recent subspace adaptation methods mitigate this issue by freezing high-magnitude principal components while adapting the residual subspace. However, such magnitude-based decomposition is data-agnostic and does not explicitly account for target-relevant semantics, potentially leading to semantic misalignment. We propose COMPASS (COntextual Moment Prior Alignment for Subspace Separation), a data-aware subspace adaptation framework that uses a small unlabeled Semantic Proxy Set to estimate target activation statistics. COMPASS incorporates activation covariance into the decomposition of pre-trained weights, preserving target-relevant representations as a stable semantic reference while the residual adapter learns forgery-related variations. The proxy calibration requires no gradient-based optimization. Extensive experiments demonstrate strong generalization across cross-dataset, cross-method, joint crossdataset-method, and open-world cross-generator evaluations, achieving state-ofthe-art performance on both Deepfake and AIGI detection benchmarks. Source code and trained models will be publicly released.

## 1 INTRODUCTION

Recent advances in generative AI (Goodfellow et al., 2020; Ho et al., 2020; Rombach et al., 2022a; Yan et al., 2025c; Chen et al., 2026; Shen et al., 2025) enable hyper-realistic synthesis, posing security risks such as deepfakes and visual disinformation (Korshunov & Marcel, 2018; Thies et al., 2016; Shoaib et al., 2023; Singh & Dhumane, 2025; Easttom, 2025). Consequently, robust AIgenerated image (AIGI) detection is critical (Wang et al., 2020; Rossler et al., 2019; Yan et al., 2025b; Sarkar et al., 2025; Rathore et al., 2021; Fridrich et al., 2003; Kashyap et al., 2017). A key challenge is the asymmetry phenomenon: detectors overfit to limited forgery patterns and biases (Yan et al., 2025b), yielding entangled features that can hinder generalization to unseen manipulations.

To address this issue, recent works explore low-rank fine-tuning of Vision Foundation Models (VFMs) (Radford et al., 2021) via Orthogonal Subspace Decomposition (Wall et al., 2003). EF-FORT (Yan et al., 2025b), a state-of-the-art SVD-based AIGI detector, emphasizes preserving pretrained VFM knowledge for generalization. It applies SVD to decompose the pre-trained weights into principal and residual components, freezing principal components with the largest magnitudes to retain core semantics, while fine-tuning residual components to capture forgery-specific patterns.

The motivation for preserving pre-trained semantics is not to use semantic content itself as a cue for real/fake classification. Rather, real and generated images typically share the same dominant semantic content, while forgery-specific signals appear as variations within this shared context. Preserving the principal semantic subspace therefore provides a stable reference for the residual adapter to learn forgery-related variations without unnecessarily altering the underlying content representation.

![](images/c0e05a9f27e9956047d618dd27e2ef4288da683e86d188586293b0002d38d09d.jpg)  
(a)

![](images/f46fe77bf19cac046f3187ed163b3e37fda4680c25fbf0692c09912d239f4fc7.jpg)

![](images/d053c7b63450a9ec53159afab3fc3a0c680acbfef8ab666a825a81c06e4a3f42.jpg)  
(b)  
Figure 1: Pilot study on feature retention prior to fine-tuning. (a) PCA visualizations of intermediate features across 1K images. Magnitude-based truncation in EFFORT increasingly alters the original feature geometry in deeper layers, with the average cosine similarity to the original CLIP representation dropping to 0.26 in Layer 24, compared with 0.61 for COMPASS. (b) Pairwise similarity density plots using only the principal subspace $( \mathbf { W } _ { p r i n }$ , before fine-tuning). Each point compares the cosine similarity of the same image pair in the original CLIP and principal-only spaces. COM-PASS better preserves the relative similarity structure, achieving a higher Spearman rank correlation $( \rho _ { s } = 0 . 5 5 3 3 )$ than EFFORT $( \rho _ { s } = 0 . 1 3 4 5 )$ . Since truncation and feature re-normalization may shift absolute cosine similarities, we focus on rank preservation rather than proximity to the diagonal; further analysis is provided in section A.1.

However, its magnitude-based implementation has an important limitation: it prioritizes components according to their contribution to global weight reconstruction. Although mathematically well-founded, this criterion is data-agnostic and does not explicitly account for task-specific semantics. In AIGI detection, substantial distribution shifts arise across dataset domains (e.g., faces or human bodies), manipulation types, and generative models. As a result, preserving high-magnitude components does not guarantee retention of information most relevant for robust forgery detection.

Our pilot analysis in fig. 1 illustrates this limitation. Using feature retention, a metric adopted by prior orthogonal adaptation methods (Yan et al., 2025b), we find that magnitude-based decomposition can substantially alter the pre-trained representation retained by the principal subspace. We refer to this mismatch between magnitude-based preservation and target-relevant information as semantic misalignment. The results further suggest that incorporating target-distribution activation statistics can mitigate this mismatch and improve the preservation of pre-trained representations.

Building on this observation, we introduce COMPASS (COntextual Moment Prior Alignment for Subspace Separation), a data-aware subspace adaptation framework that incorporates proxy activation statistics into subspace partitioning. Before fine-tuning, we sample a small unlabeled Semantic Proxy Set (e.g., ∼80 images) from the target-category distribution and compute its activation covariance to guide the SVD of pre-trained weights. This lightweight, zero-backpropagation calibration biases the preserved subspace toward directions emphasized by the proxy activation distribution. The proxy samples are used only to estimate activation statistics for subspace calibration and do not participate in gradient-based optimization.

Notably, this proxy-guided calibration better preserves pre-trained representations than magnitudebased decomposition. Even when the Semantic Proxy Set is constructed from only one or two datasets within the target category, its activation statistics provide a useful estimate of target-relevant activation structure that remains effective across different data sources. By retaining these targetrelevant pre-trained semantics more faithfully, COMPASS establishes a more stable reference for residual fine-tuning, allowing forgery-related variations to be learned with less interference from source-specific content biases. This improved preservation is consistent with the stronger generalization observed across dataset domains, manipulation types, and generative models.

We evaluate COMPASS under three challenging generalization settings tables 1 to 3, where it achieves state-of-the-art performance across (1) cross-dataset scenarios, (2) cross-method scenarios, (3) joint cross-dataset-method scenarios.

The main contributions of this work are:

• We identify an important limitation of magnitude-based orthogonal subspace decomposition for AIGI detection: its data-agnostic partitioning may fail to preserve task-relevant pre-trained information. Our pilot analyses show that incorporating proxy-guided calibration can mitigate this semantic misalignment and improve feature retention.

• We introduce COMPASS (COntextual Moment Prior Alignment for Subspace Separation), which leverages a small unlabeled Semantic Proxy Set to compute activation covariances and uses activation covariance to calibrate subspace preservation before fine-tuning. This lightweight, zerobackpropagation design enables data-aware preservation of pre-trained semantic representations.

• We demonstrate that COMPASS achieves state-of-the-art generalization across cross-dataset, cross-method, and joint cross-dataset-method evaluations. Our results support the use of a betterpreserved semantic reference for residual forgery learning and show consistent improvements across diverse domains, manipulation types, and generative models.

## 2 RELATED WORK

Conventional Approaches for AIGI Detection. Conventional deepfake and synthetic-image detectors primarily exploit task-specific artifacts, including frequency and spatial cues (Li et al., 2021; Liu et al., 2021; Tan et al., 2024b), blending inconsistencies (Shiohara & Yamasaki, 2022; Chen et al., 2022a), reconstruction signals (Cao et al., 2022; Wang et al., 2023; Luo et al., 2024), and content or identity representations (Yan et al., 2023a; Huang et al., 2023). Recent methods further improve artifact-oriented modeling (Liang et al., 2025; Wang et al., 2025; Zhou et al., 2025). Despite strong performance, such task-specific cues may generalize poorly to unseen synthesis processes, motivating increasing use of Vision Foundation Models (VFMs) for more transferable forgery detection.

Vision Foundation Models and Subspace Adaptation. Recent methods leverage VFMs such as CLIP (Radford et al., 2021) to improve generalization in forgery detection (Ojha et al., 2023; Wu et al., 2023; Liu et al., 2024). However, direct fine-tuning may alter pre-trained representations and reduce cross-domain generalization. EFFORT (Yan et al., 2025b) addresses this issue by freezing high-magnitude SVD components while adapting the residual subspace. Its magnitude-based decomposition, however, remains data-agnostic. COMPASS instead uses target activation statistics to guide subspace preservation.

## 3 METHOD

We present COMPASS (COntextual Moment Prior Alignment for Subspace Separation), a dataaware subspace adaptation framework for generalizable AIGI detection. As illustrated in fig. 2, COMPASS uses a small unlabeled Semantic Proxy Set to estimate target activation statistics, which guide the decomposition of pre-trained weights into principal and residual weight components. The principal weight component is kept fixed as a stable semantic reference, while the residual adapter is optimized to learn forgery-related variations.

## 3.1 PRELIMINARIES: MAGNITUDE-BASED SUBSPACE DECOMPOSITION

Given a pre-trained Vision Foundation Model $( e . g .$ , CLIP (Radford et al., 2021)) employed as the backbone for deepfake detection, let $\mathbf { W } \in \mathbb { R } ^ { j _ { o u t } \times d _ { i n } }$ denote the weight matrix of a linear layer. EFFORT (Yan et al., 2025b) decomposes W using Singular Value Decomposition (SVD):

$$
\mathbf { W } = \mathbf { U } \mathbf { S } \mathbf { V } ^ { \top } = \mathbf { U } _ { k } \mathbf { S } _ { k } \mathbf { V } _ { k } ^ { \top } + \mathbf { U } _ { r } \mathbf { S } _ { r } \mathbf { V } _ { r } ^ { \top } ,\tag{1}
$$

where the top-k components are frozen to preserve pre-trained representations, while the residual components are adapted for forgery detection.

![](images/0761aa9fdc969b57fbd4cc1f47629e898fe32b49cc3894cba2e348e4a5280de1.jpg)  
Figure 2: Overall pipeline of COMPASS. Left: A small unlabeled Semantic Proxy Set is used to estimate the activation covariance C. Middle: The pre-trained weights are factorized via covarianceguided SVD into principal weights $\mathbf { W } _ { p r i n }$ and residual weights $\bar { \mathbf { W } } _ { r e s }$ . Right: During fine-tuning, $\bar { \mathbf { W } } _ { p r i n }$ is frozen as a stable semantic reference, while the residual adapter ∆W, initialized from $\mathbf { W } _ { r e s } ,$ is optimized to learn forgery-related variations.

This decomposition follows the standard rank-k reconstruction objective

$$
\operatorname* { m i n } _ { \mathrm { \Gamma } \mathrm { r a n k } ( \mathbf { W } _ { k } ) \leq k } \| \mathbf { W } - \mathbf { W } _ { k } \| _ { F } ^ { 2 } .\tag{2}
$$

Consequently, the retained components are determined solely by their contribution to reconstructing the weight matrix and are independent of the input data distribution. In particular, a large singular value does not necessarily imply that the corresponding direction is important for preserving the representations activated by the target data.

As shown in fig. 1, magnitude-based SVD can substantially alter the VFM representations retained by the principal subspace, illustrating this semantic misalignment and motivating a data-aware response-preservation objective.

## 3.2 PROXY-GUIDED DATA-AWARE SUBSPACE CALIBRATION

Data-Aware Response Preservation. How can the decomposition be made aware of the target data distribution? A direct approach preserves the response of each pre-trained linear layer on targetrelevant inputs. Let $\mathbf { x } \in \mathbb { R } ^ { d _ { i n } }$ denote the input activation to a layer with original and decomposed weights W and $\mathbf { W } _ { k }$ . We therefore consider the response-preservation objective

$$
\operatorname* { m i n } _ { \mathrm {  ~ \scriptstyle { \ r a n k } ~ } ( { \bf W } _ { k } ) \leq k } \mathbb { E } _ { \bf x } \left[ \left\| ( { \bf W } - { \bf W } _ { k } ) { \bf x } \right\| _ { 2 } ^ { 2 } \right] .\tag{3}
$$

This objective is consistent with the motivation of our pilot study in fig. 1, which evaluates whether the preserved principal subspace retains the representations produced by the original VFM.

Assuming that the input activations are mean-centered, let

$$
\mathbf { C } = \mathbb { E } _ { \mathbf { x } } \left[ \mathbf { x } \mathbf { x } ^ { \top } \right]\tag{4}
$$

denote their covariance matrix. The response-preservation objective can then be rewritten as

$$
\mathbb { E } _ { \mathbf { x } } \left[ \left. ( \mathbf { W } - \mathbf { W } _ { k } ) \mathbf { x } \right. _ { 2 } ^ { 2 } \right] = \operatorname { T r } \left[ ( \mathbf { W } - \mathbf { W } _ { k } ) \mathbf { C } ( \mathbf { W } - \mathbf { W } _ { k } ) ^ { \top } \right] = \left. ( \mathbf { W } - \mathbf { W } _ { k } ) \mathbf { C } ^ { 1 / 2 } \right. _ { F } ^ { 2 } .\tag{5}
$$

This form motivates performing the low-rank decomposition in the covariance-weighted space $\mathbf { W C } ^ { 1 / 2 }$ , rather than on W alone. Hence, $\mathbf { C } ^ { 1 / 2 }$ naturally serves as a data-dependent guidance matrix: unlike standard SVD, the decomposition assigns greater importance to directions that are more strongly expressed under the target activation distribution.

Proxy-Based Covariance Estimation. To estimate the target activation covariance in practice, we introduce a small unlabeled Semantic Proxy Set $\mathcal { D } _ { p }$ containing $N _ { p }$ samples drawn from the target-category distribution. Passing these samples through the frozen VFM, we collect the input activations $\mathbf { \bar { X } } _ { p } ^ { } \in \mathbb { R } ^ { d _ { i n } \times N _ { \bar { F } } }$ for each target linear layer. After mean-centering the activations to obtain $\widehat { \mathbf { X } } _ { p } ,$ we estimate the covariance as

$$
\mathbf { C } = \frac { 1 } { N _ { p } } \widehat { \mathbf { X } } _ { p } \widehat { \mathbf { X } } _ { p } ^ { \top } .\tag{6}
$$

We use this covariance as a second-order moment prior that summarizes the activation structure of the proxy distribution. The proxy set is used only to estimate these activation statistics for decomposition and does not participate in gradient-based optimization.

According to the response-preservation objective derived above, $\mathbf { C } ^ { 1 / 2 }$ naturally serves as the proxy guidance matrix for making the decomposition data-aware. In practice, however, our experiments show that directly using C as the guidance matrix provides better generalization than $\mathbf { C } ^ { 1 / 2 }$ (table $5 ( \mathrm { a } ) )$ . This choice is empirical rather than directly implied by the preceding derivation. We hypothesize that the stronger covariance weighting of C places greater emphasis on activation directions that are more strongly expressed under the proxy distribution, thereby providing a stronger guidance signal for subspace decomposition. Accordingly, COMPASS adopts C as the proxy guid ance matrix in the following decomposition.

Covariance-Guided SVD. After selecting C as the proxy guidance matrix, COMPASS incorporates it directly into the weight decomposition. Specifically, we perform SVD on the covariance-guided weight matrix:

$$
\mathbf { W C } = \mathbf { U } ^ { \prime } \mathbf { S } ^ { \prime } \mathbf { V } ^ { \prime } \mathbf { \Sigma } ^ { \top } ,\tag{7}
$$

where $\mathbf { U } ^ { \prime }$ and $\mathbf { V } ^ { \prime }$ contain the left and right singular vectors, respectively, and $\mathbf { S } ^ { \prime }$ contains the singular values in descending order. The original weight matrix can be written as

$$
\mathbf { W } = \mathbf { U } ^ { \prime } \mathbf { S } ^ { \prime } \mathbf { V } ^ { \prime ^ { \top } } \mathbf { C } ^ { - 1 } .\tag{8}
$$

Compared with standard SVD, whose component ordering is determined solely by the weight matrix, this covariance-guided decomposition also incorporates the activation statistics estimated from the proxy distribution. As a result, the resulting components reflect both the structure of the pretrained weights and the target activation statistics estimated by the proxy set. These components are then divided into principal and residual parts for subsequent semantic anchoring and residual fine-tuning.

## 3.3 SEMANTIC ANCHORING AND RESIDUAL FORGERY LEARNING

Given the covariance-guided decomposition, we partition the singular components into principal and residual parts:

$$
\mathbf { W } _ { p r i n } = { \mathbf { U } _ { k } ^ { \prime } } { \mathbf { S } _ { k } ^ { \prime } } { \mathbf { V } _ { k } ^ { \prime } } ^ { \top } \mathbf { C } ^ { - 1 } , \qquad \mathbf { W } _ { r e s } = { \mathbf { U } _ { r } ^ { \prime } } { \mathbf { S } _ { r } ^ { \prime } } { \mathbf { V } _ { r } ^ { \prime } } ^ { \top } \mathbf { C } ^ { - 1 } ,\tag{9}
$$

where k denotes the rank threshold. By construction, $\mathbf { W } _ { p r i n } + \mathbf { W } _ { r e s } = \mathbf { W } .$

Semantic Anchoring. We freeze $\mathbf { W } _ { p r i n }$ throughout fine-tuning to preserve a stable representation of the dominant semantic content. The preserved principal component is not intended to serve as a direct cue for real/fake discrimination; instead, it provides a stable semantic reference within which forgery-related variations can be learned.

Residual Forgery Learning. We introduce a trainable adapter ∆W and initialize it with the residual component obtained from the covariance-guided decomposition:

$$
\Delta { \bf W } ^ { ( 0 ) } = { \bf W } _ { r e s } = { \bf U } _ { r } ^ { \prime } { \bf S } _ { r } ^ { \prime } { \bf V } _ { r } ^ { \prime } { \bf \Psi } ^ { \top } { \bf C } ^ { - 1 } .\tag{10}
$$

Since $\mathbf { W } _ { p r i n } + \Delta \mathbf { W } ^ { ( 0 ) } = \mathbf { W }$ , the decomposed layer is exactly equivalent to the original pre-trained layer before optimization.

During fine-tuning, $\mathbf { W } _ { p r i n }$ remains fixed and only the low-rank parameters of the residual adapter ∆W are updated. The adapted layer is therefore

$$
\begin{array} { r } { \mathbf { Y } = ( \mathbf { W } _ { p r i n } + \Delta \mathbf { W } ) \mathbf { X } . } \end{array}\tag{11}
$$

This residual adaptation maintains the principal weight component as a stable semantic reference while allowing the trainable component to capture forgery-related variations.

Training Objective. Given the labeled source-domain training set $\mathcal { D } _ { s r c } .$ we optimize only ∆W using the standard binary cross-entropy loss.

## 4 EXPERIMENTS

To empirically evaluate COMPASS, we conduct extensive experiments across diverse generalization settings. We first benchmark facial Deepfake detection against 16 methods under cross-dataset, cross-method, and joint cross-dataset-method protocols. We then extend the evaluation to openworld AIGI detection, comparing against 14 competitive detectors across 25 generative models. Finally, we perform a series of ablation and robustness studies to examine the calibration mechanism.

## 4.1 EXPERIMENTAL SETUPS

Implementation Details. Following EFFORT (Yan et al., 2025b), we use CLIP ViT-L/14 (Radford et al., 2021) as the VFM backbone. The unlabeled Semantic Proxy Set is sampled exclusively from the target-category training split and is used only to estimate activation statistics. No test samples are used for proxy-set, thereby avoiding test-set leakage despite the absence of label supervision. We use $N _ { p } = 8 0$ for Deepfake detection and $N _ { p } = 1 5 0 0$ for AIGI detection unless otherwise specified.

During fine-tuning, only the residual adapter ∆W is optimized. We use Adam with a learning rate of $2 \times 1 0 ^ { - 4 }$ and batch sizes of 32 and 48 for Deepfake and AIGI detection, respectively. Following EFFORT, the residual rank is set to 1. All experiments are implemented in PyTorch and trained on a single NVIDIA RTX 4090 GPU.

Evaluation Metrics. We report video-level AUC for Deepfake detection and AP for AIGI detection following AIGIBench (Li et al., 2025b).

Table 1: Video-level AUC results of Cross-dataset and Cross-method evaluations. All detectors are trained on FF++ (c23) and evaluated on other fake data.
<table><tr><td rowspan="2">Methods</td><td colspan="5">Cross-dataset Evaluation</td><td colspan="10">Cross-method Evaluation</td></tr><tr><td>CDF-v1</td><td>CDF-v2</td><td>DFDC</td><td>DFDCP</td><td>Avg.</td><td>UniFace</td><td>BleFace</td><td>MobSwap</td><td>e4s</td><td>FaceDan</td><td>FSGAN</td><td>InSwap</td><td>SimSwap</td><td>Avg.</td></tr><tr><td>F3Net (Qian et al., 2020)</td><td>0.777</td><td>0.789</td><td>0.718</td><td>0.749</td><td>0.758</td><td>0.809</td><td>0.808</td><td>0.867</td><td>0.494</td><td>0.717</td><td>0.845</td><td>0.757</td><td>0.674</td><td>0.746</td></tr><tr><td>SPSL (Liu et al., 2021)</td><td>0.815</td><td>0.799</td><td>0.724</td><td>0.770</td><td>0.777</td><td>0.747</td><td>0.748</td><td>0.885</td><td>0.514</td><td>0.666</td><td>0.812</td><td>0.643</td><td>0.665</td><td>0.710</td></tr><tr><td>SRM (Liu et al., 2021)</td><td>0.793</td><td>0.840</td><td>0.695</td><td>0.728</td><td>0.764</td><td>0.749</td><td>0.704</td><td>0.779</td><td>0.704</td><td>0.659</td><td>0.772</td><td>0.793</td><td>0.694</td><td>0.732</td></tr><tr><td>CORE (Ni et al., 2022)</td><td>0.780</td><td>0.809</td><td>0.721</td><td>0.720</td><td>0.758</td><td>0.871</td><td>0.843</td><td>0.959</td><td>0.679</td><td>0.774</td><td>0.958</td><td>0.855</td><td>0.724</td><td>0.833</td></tr><tr><td>RECCE (Cao et al., 2022)</td><td>0.768</td><td>0.823</td><td>0.696</td><td>0.734</td><td>0.755</td><td>0.898</td><td>0.832</td><td>0.925</td><td>0.683</td><td>0.848</td><td>0.949</td><td>0.848</td><td>0.768</td><td>0.844</td></tr><tr><td>SLADD (Chen et al., 2022b)</td><td>0.802</td><td>0.837</td><td>0.772</td><td>0.756</td><td>0.792</td><td>0.878</td><td>0.882</td><td>0.954</td><td>0.765</td><td>0.825</td><td>0.943</td><td>0.879</td><td>0.794</td><td>0.865</td></tr><tr><td>SBI (Shiohara &amp; Yamasaki, 2022)</td><td>0.831</td><td>0.886</td><td>0.717</td><td>0.848</td><td>0.821</td><td>0.724</td><td>0.891</td><td>0.952</td><td>0.750</td><td>0.594</td><td>0.803</td><td>0.712</td><td>0.701</td><td>0.766</td></tr><tr><td>UCF (Yan et al., 2023a)</td><td>0.780</td><td>0.837</td><td>0.742</td><td>0.770</td><td>0.782</td><td>0.831</td><td>0.827</td><td>0.950</td><td>0.731</td><td>0.862</td><td>0.937</td><td>0.809</td><td>0.647</td><td>0.824</td></tr><tr><td>IID (Huang et al., 2023)</td><td>0.758</td><td>0.838</td><td>0.700</td><td>0.689</td><td>0.746</td><td>0.839</td><td>0.789</td><td>0.888</td><td>0.766</td><td>0.844</td><td>0.927</td><td>0.789</td><td>0.644</td><td>0.811</td></tr><tr><td>LSDA (Yan et al., 2024b)</td><td>0.867</td><td>0.875</td><td>0.701</td><td>0.812</td><td>0.814</td><td>0.872</td><td>0.875</td><td>0.930</td><td>0.694</td><td>0.721</td><td>0.939</td><td>0.855</td><td>0.793</td><td>0.835</td></tr><tr><td>ProDet (Cheng et al., 2024)</td><td>0.909</td><td>0.926</td><td>0.707</td><td>0.828</td><td>0.843</td><td>0.908</td><td>0.929</td><td>0.975</td><td>0.771</td><td>0.747</td><td>0.928</td><td>0.837</td><td>0.844</td><td>0.867</td></tr><tr><td>CDFA (Lin et al., 2024)</td><td></td><td>0.938</td><td>0.830</td><td>0.881</td><td>0.883</td><td>0.762</td><td>0.756</td><td>0.823</td><td>0.631</td><td>0.803</td><td>0.942</td><td>0.772</td><td>0.757</td><td>0.781</td></tr><tr><td>Effort (Yan et al., 2025b)</td><td>0.970</td><td>0.956</td><td>0.843</td><td>0.909</td><td>0.920</td><td>0.962</td><td>0.873</td><td>0.953</td><td>0.983</td><td>0.926</td><td>0.957</td><td>0.936</td><td>0.926</td><td>0.940</td></tr><tr><td>Ours</td><td>0.957</td><td>0.942</td><td>0.870</td><td>0.914</td><td>0.921</td><td>0.980</td><td>0.907</td><td>0.972</td><td>0.993</td><td>0.967</td><td>0.986</td><td>0.968</td><td>0.960</td><td>0.966</td></tr></table>

## 4.2 DEEPFAKE DETECTION

Following the evaluation organization of EFFORT (Yan et al., 2025b), we consider two established generalization settings: cross-dataset evaluation based on DeepfakeBench (Yan et al., 2023b) and cross-method evaluation based on DF40 (Yan et al., 2024c). We further introduce a joint crossdataset-method setting that shifts both the data domain and manipulation method. We compare against 16 representative detectors; detailed baselines and datasets are summarized in tables 1 and 2.

Evaluation Protocols.

• Protocol-1 (Cross-dataset): Models trained on FF++ are evaluated on unseen facial datasets while keeping the forgery setting relatively consistent, measuring robustness to domain shift.

• Protocol-2 (Cross-method): Models are evaluated on unseen manipulation methods from DF40 (Yan et al., 2024c) while keeping the underlying FF++ data domain fixed, isolating the effect of forgery-method shift. We report Face Swapping as the representative category in the main table.

• Protocol-3 (Joint Cross-dataset-method): Both the data domain and manipulation method are shifted simultaneously. Models trained on FF++ are evaluated on Face Reenactment forgeries generated from the CDF-v1 domain in DF40 (Yan et al., 2024c), providing a more challenging double-shift setting.

Results and Analysis. As shown in table 1, COMPASS achieves comparable or improved performance under both single-shift protocols. In cross-dataset evaluation, the gain over EFFORT is marginal (0.921 vs. 0.920 Avg. AUC). This setting changes the facial data domain while retaining relatively familiar manipulation characteristics, allowing both methods to continue exploiting forgery cues learned from the source distribution. In contrast, the advantage of COMPASS becomes more evident under cross-method evaluation, where the manipulation mechanism itself changes. Although the underlying FF++ domain remains familiar, preserving target-relevant pre-trained representations becomes more useful when previously learned forgery cues transfer less directly.

The advantage is largest under the joint cross-dataset-method setting in table 2, where both the image domain and manipulation method are unseen. COMPASS improves the Avg. AUC from 0.710 for EFFORT to 0.829, an absolute gain of 11.9%. These results suggest that the limitation of dataagnostic, magnitude-based preservation becomes increasingly consequential as fewer source-aligned cues remain available, explaining why the improvement is modest under cross-dataset evaluation but substantially larger under cross-method and joint shifts.

Table 2: Video-level AUC results of Joint Cross-dataset-method evaluations. All detectors are trained on FF++ (c23) and evaluated on Face Reenactment forgeries generated from real CDF face data. All baselines are evaluated using the official checkpoints provided by their respective authors.
<table><tr><td>Methods</td><td>FOMM</td><td>FV2V</td><td>Wav2Lip</td><td>MRAA</td><td>OneShot</td><td>PiRender</td><td>TPSMM</td><td>LIA</td><td>DaGan</td><td>SadTalker</td><td>MCNet</td><td>HyperReenact</td><td>Avg.</td></tr><tr><td>EFFNB4 (Tan &amp; Le, 2019)</td><td>0.799</td><td>0.795</td><td>0.683</td><td>0.880</td><td>0.833</td><td>0.759</td><td>0.620</td><td>0.772</td><td>0.602</td><td>0.555</td><td>0.683</td><td>0.637</td><td>0.718</td></tr><tr><td>CNNAUG (Wang et al., 2020)</td><td>0.792</td><td>0.730</td><td>0.559</td><td>0.819</td><td>0.760</td><td>0.708</td><td>0.654</td><td>0.686</td><td>0.606</td><td>0.627</td><td>0.630</td><td>0.654</td><td>0.685</td></tr><tr><td>F3Net (Qian et al., 2020)</td><td>0.689</td><td>0.741</td><td>0.726</td><td>0.927</td><td>0.660</td><td>0.648</td><td>0.552</td><td>0.836</td><td>0.575</td><td>0.452</td><td>0.663</td><td>0.608</td><td>0.673</td></tr><tr><td>FFD (Dang et al., 2020)</td><td>0.752</td><td>0.658</td><td>0.595</td><td>0.905</td><td>0.727</td><td>0.696</td><td>0.535</td><td>0.779</td><td>0.531</td><td>0.461</td><td>0.588</td><td>0.541</td><td>0.647</td></tr><tr><td>SPSL (Liu et al., 2021)</td><td>0.769</td><td>0.773</td><td>0.697</td><td>0.851</td><td>0.742</td><td>0.742</td><td>0.633</td><td>0.681</td><td>0.583</td><td>0.539</td><td>0.635</td><td>0.800</td><td>0.704</td></tr><tr><td>SRM (Liu et al., 2021)</td><td>0.800</td><td>0.810</td><td>0.750</td><td>0.937</td><td>0.804</td><td>0.729</td><td>0.665</td><td>0.856</td><td>0.652</td><td>0.559</td><td>0.735</td><td>0.651</td><td>0.746</td></tr><tr><td>CORE (Ni et al., 2022)</td><td>0.724</td><td>0.678</td><td>0.680</td><td>0.930</td><td>0.714</td><td>0.659</td><td>0.601</td><td>0.853</td><td>0.517</td><td>0.493</td><td>0.620</td><td>0.684</td><td>0.679</td></tr><tr><td>RECCE (Cao et al., 2022)</td><td>0.759</td><td>0.702</td><td>0.846</td><td>0.970</td><td>0.687</td><td>0.703</td><td>0.593</td><td>0.881</td><td>0.517</td><td>0.502</td><td>0.631</td><td>0.730</td><td>0.710</td></tr><tr><td>UCF (Yan et al., 2023a)</td><td>0.774</td><td>0.752</td><td>0.710</td><td>0.948</td><td>0.739</td><td>0.726</td><td>0.570</td><td>0.872</td><td>0.597</td><td>0.461</td><td>0.678</td><td>0.630</td><td>0.631</td></tr><tr><td>Effort (Yan et al., 2025b)</td><td>0.798</td><td>0.709</td><td>0.636</td><td>0.841</td><td>0.755</td><td>0.729</td><td>0.619</td><td>0.797</td><td>0.611</td><td>0.540</td><td>0.638</td><td>0.843</td><td>0.710</td></tr><tr><td>Ours</td><td>0.919</td><td>0.853</td><td>0.715</td><td>0.970</td><td>0.906</td><td>0.873</td><td>0.724</td><td>0.925</td><td>0.771</td><td>0.563</td><td>0.813</td><td>0.919</td><td>0.829</td></tr></table>

## 4.3 AI-GENERATED IMAGE (AIGI) DETECTION

For open-world AIGI detection, we follow Setting-II of AIGIBench (Li et al., 2025b). The setting uses 144K training images comprising real images and synthetic images from ProGAN (Karras et al., 2018) and SDv1.4 (Rombach et al., 2022b), and evaluates generalization across 25 generators spanning diverse model families and generation paradigms.

We compare COMPASS with 14 representative AIGI detectors, covering CNN-, frequency-, VFMbased, and recent state-of-the-art methods. Detailed baseline information and references are provided in section A.6.

As reported in table 3, COMPASS achieves the best average AP of 90.9% across 25 generative models. The improvement is particularly pronounced on several challenging commercial generators. For example, on DALLE-3 (Team, 2024), COMPASS reaches 92.3% AP compared with 60.6% for EF-FORT, an absolute gain of 31.7%. Consistent improvements are also observed on FLUX1-dev (Labs, 2024) and CommunityAI. These results suggest that covariance-guided preservation remains effective under the substantially broader semantic and generative diversity of open-world AIGI detection, where magnitude-based decomposition may preserve less relevant pre-trained information.

## 4.4 ABLATION AND GENERALIZATION ANALYSIS

## 4.4.1 ROBUSTNESS TO PROXY SET SELECTION.

We first examine whether COMPASS is sensitive to the datasets used to construct the Semantic Proxy Set. For each manipulation category, i.e., Face Swapping and Face Reenactment, we randomly select two datasets and sample a total of 80 unlabeled images from their training splits to form the proxy set. We repeat this procedure with different randomly selected dataset pairs and evaluate the resulting models across datasets within the same manipulation category.

As summarized in table 4(a), the performance remains stable across different proxy-source pairs. In FS, the average AUC ranges from 0.956 to 0.969 across four randomly selected pairs, compared with 0.939 for EFFORT. In FR, the corresponding range is 0.871–0.899 across six pairs, compared with 0.808 for EFFORT. These results indicate that COMPASS’s proxy set does not depend strongly on particular datasets .

Table 3: Cross-generator evaluation on representative test sets spanning different generation families, together with the average performance over all 25 datasets included in AIGIBench (Li et al., 2025b). Full per-dataset results are provided in section A.6.
<table><tr><td>Generator</td><td>CNN-Spot</td><td>UnivFD</td><td> $\overline { { { \mathrm { F r e q N e t } } } }$ </td><td>NPR</td><td>Ladeda</td><td>AIDE</td><td>C2P-CLIP</td><td>DFfreq</td><td>SAFE</td><td>Effort</td><td>NS-Net</td><td>FerretNet</td><td>LOTA</td><td>BSF</td><td>Ours</td></tr><tr><td>R3GAN</td><td>52.4</td><td>91.2</td><td>55.8</td><td>61.1</td><td>72.6</td><td>97.1</td><td>91.3</td><td>90.6</td><td>99.2</td><td>98.2</td><td>91.3</td><td>99.0</td><td>94.7</td><td>93.9</td><td>99.7</td></tr><tr><td>StyleGAN-XL</td><td>50.2</td><td>93.3</td><td>83.7</td><td>75.3</td><td>93.1</td><td>93.2</td><td>98.9</td><td>69.5</td><td>90.8</td><td>97.0</td><td>96.1</td><td>95.6</td><td>97.4</td><td>97.2</td><td>98.7</td></tr><tr><td>FaceSwap</td><td>58.3</td><td>52.3</td><td>42.9</td><td>43.6</td><td>40.9</td><td>56.3</td><td>70.4</td><td>64.9</td><td>69.6</td><td>85.8</td><td>62.5</td><td>71.6</td><td>73.4</td><td>66.4</td><td>88.6</td></tr><tr><td>DALLE-3</td><td>68.4</td><td>76.4</td><td>60.2</td><td>70.0</td><td>59.8</td><td>63.1</td><td>56.3</td><td>48.5</td><td>49.5</td><td>60.6</td><td>42.2</td><td>55.4</td><td>44.3</td><td>43.0</td><td>92.3</td></tr><tr><td>FLUX1-dev</td><td>72.1</td><td>79.5</td><td>87.0</td><td>99.0</td><td>98.7</td><td>93.4</td><td>84.3</td><td>98.2</td><td>99.0</td><td>74.4</td><td>94.1</td><td>99.2</td><td>98.0</td><td>97.5</td><td>84.8</td></tr><tr><td>SDXL Infinite-ID</td><td>63.8</td><td>88.0</td><td>95.0</td><td>94.4</td><td>98.5</td><td>95.7</td><td>97.1</td><td>99.1</td><td>99.4</td><td>98.0</td><td>98.2</td><td>99.9</td><td>99.4</td><td>98.0</td><td>94.4 97.2</td></tr><tr><td>CommunityAI</td><td>49.1 56.7</td><td>89.6 73.7</td><td>73.6 63.5</td><td>80.4 62.9</td><td>76.9 56.3</td><td>94.7</td><td>84.2 53.2</td><td>97.4</td><td>98.5 64.3</td><td>85.7 64.3</td><td>91.8 54.0</td><td>99.7</td><td>98.1 49.6</td><td>97.0 57.5</td><td>70.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>61.0</td><td></td><td>64.2</td><td></td><td></td><td></td><td>53.0</td><td></td><td></td><td></td></tr><tr><td>Average of all 25</td><td>67.3</td><td>75.7</td><td>69.9</td><td>73.9</td><td>79.3</td><td>82.7</td><td>82.2</td><td>82.2</td><td>82.9</td><td>87.2</td><td>83.0</td><td>85.8</td><td>82.1</td><td>81.1</td><td>90.9</td></tr></table>

Table 4: Robustness analysis of the Semantic Proxy Set. (a) Sensitivity to randomly selected proxysource datasets. Detailed results are provided in section A.7. (b) Ablation on proxy composition, sample size, and semantic relevance.

(a) Proxy-Source Selection
<table><tr><td>Category</td><td>EFFORT</td><td>COMPASS</td><td>Range</td><td>∆</td></tr><tr><td>Face Swapping</td><td>0.939</td><td> ${ \bf 0 . 9 6 4 \pm 0 . 0 0 5 }$ </td><td>0.956-0.969</td><td>+0.025</td></tr><tr><td>Face Reenactment</td><td>0.808</td><td> $\mathbf { 0 . 8 8 8 \pm 0 . 0 0 9 }$ </td><td>0.871–0.899</td><td>+0.080</td></tr></table>

(b) Proxy-Set Ablation
<table><tr><td>Setting</td><td>Proxy Set</td><td>Ratio</td><td> $N _ { p }$ </td><td>AUC (%)</td></tr><tr><td rowspan="2">Baseline Default</td><td>EFFORT (No Proxy)</td><td></td><td>0</td><td>94.0</td></tr><tr><td>COMPASS (40 runs)</td><td>Mixed</td><td>80</td><td> ${ \bf 9 5 . 8 \pm 0 . 5 }$ </td></tr><tr><td>Composition</td><td>All Real</td><td>100:0</td><td>80</td><td>95.76</td></tr><tr><td rowspan="3">Few-shot</td><td>All Fake</td><td>0:100</td><td>80</td><td>94.95</td></tr><tr><td>Two-shot</td><td>50:50</td><td>2</td><td>94.74</td></tr><tr><td>One-shot</td><td>Random</td><td>1</td><td>94.58</td></tr><tr><td>Unrelated</td><td>ImageNet/COCO</td><td></td><td>80</td><td>93.62</td></tr></table>

We further vary the composition and size of the proxy set in table 4(b). Across 40 random proxy sets, COMPASS achieves an average AUC of 95.8% with a standard deviation of 0.5%. The performance remains above the EFFORT baseline when using only real or only fake samples, and even one- or two-sample proxy sets still provide a modest improvement. In contrast, using unrelated ImageNet/COCO images reduces performance to 93.62%, close to the EFFORT baseline. These results indicate that COMPASS benefits from a small amount of target-relevant semantic context, while being relatively insensitive to the precise real/fake composition of the proxy set.

## 4.4.2 EFFECT OF PROXY SET SIZE.

As shown in fig. 3, COMPASS is effective with a small proxy set and stabilizes as $N _ { p }$ increases. For Deepfake detection, performance peaks at $N _ { p } = 8 0 ( \bar { 0 . 9 6 6 } \mathrm { A U C } )$ , while the more diverse AIGI setting requires more samples, reaching 0.890 AP at $N _ { p } = 1 5 0 0 .$ . This difference likely reflects the broader semantic diversity of open-world AIGI detection compared with the relatively constrained facial domain. Even so, 1,500 unlabeled samples represent only a small fraction of the 100K+ training corpus. These results suggest that COMPASS can estimate useful target activation statistics from comparatively limited unlabeled data, although the required proxy-set size increases with the diversity of the target domain.

## 4.4.3 GUIDANCE MATRIX DESIGN.

We compare different statistics for constructing the proxy guidance matrix in table 5(a). Overall, the full covariance C achieves the best performance among the evaluated variants, outperforming mean-, correlation-, and diagonal-variance-based guidance on both Deepfake and AIGI detection.

Notably, although $\mathbf { C } ^ { 1 / 2 }$ follows naturally from the response-preservation objective in section 3.2, it yields 0.955 AUC and 0.887 AP, compared with 0.966 and 0.890 using C. We therefore adopt

![](images/1dc080ce0d41177171891fcb4070068a5fe6a95d157b5b9b979ddaf9d66815e4.jpg)  
(a) Deepfake Detection

![](images/43dd5519b454ea6fa5469c2d465fb7e8a68aa4d6049ed5613ab8362e15458b63.jpg)  
(b) AIGI Detection  
Figure 3: Sensitivity to Proxy Set Size $( N _ { p } ) .$ . COMPASS improves with additional proxy set samples and gradually stabilizes on both (a) Deepfake Detection and (b) AIGI Detection. The facial domain requires substantially fewer samples than the more semantically diverse open-world AIGI setting.

Table 5: (a) Comparison of different proxy guidance matrices on Deepfake and AIGI detection. (b) Cross-method evaluation across different VFM backbones.  
(a) Proxy Guidance Matrix
<table><tr><td>Method</td><td>Guidance Matrix G</td><td>Deepfake AUC ↑</td><td>AIGI AP ↑</td></tr><tr><td>EFFORT</td><td>Identity (I)</td><td>0.940</td><td>0.872</td></tr><tr><td>Mean</td><td>Rank-1 Mean  $( \bar { \mathbf { X } } _ { p } \bar { \mathbf { X } } _ { p } ^ { \top } )$ </td><td>0.943</td><td>0.873</td></tr><tr><td>Raw</td><td>Raw Correlation  $\begin{array} { r } { ( \frac { 1 } { N _ { p } } \mathbf { X } _ { p } \mathbf { X } _ { p } ^ { \top } ) } \end{array}$ </td><td>0.956</td><td>0.879</td></tr><tr><td>Diag</td><td>Diagonal Variance  $\left( \mathrm { { { \dot { d i a g } } ( { \bf C } ) } } \right)$ </td><td>0.938</td><td>0.874</td></tr><tr><td>Sqrt</td><td>Covariance Root  $( \mathbf { C } ^ { 1 / 2 } )$ </td><td>0.955</td><td>0.879</td></tr><tr><td>COMPASS</td><td>Covariance (C)</td><td>0.966</td><td>0.890</td></tr></table>

(b) VFM Backbone Generalization
<table><tr><td>Backbone</td><td>Pre-training</td><td>Base AUC</td><td>+COMPASS</td><td>∆</td></tr><tr><td>BEiT-v2 (Peng et al., 2022)</td><td>MIM</td><td>0.659</td><td>0.824</td><td>+0.165</td></tr><tr><td>SigLIP (Zhai et al., 2023)</td><td>Contrastive</td><td>0.714</td><td>0.839</td><td>+0.125</td></tr><tr><td>DINOv3 (Siméoni et al., 2025)</td><td>Self-supervised</td><td>0.772</td><td>0.962</td><td>+0.190</td></tr><tr><td>CLIP (Radford et al., 2021)</td><td>Self-supervised</td><td>0.851</td><td>0.966</td><td>+0.115</td></tr></table>

the full covariance as an empirical design choice, suggesting that the stronger covariance weighting provides more effective guidance for data-aware decomposition.

## 4.4.4 GENERALIZATION ACROSS VFM BACKBONES.

We evaluate COMPASS with BEiT-v2, SigLIP, DINOv3 and CLIP under the cross-method setting. As shown in table 5(b), COMPASS improves all three backbones by 0.125–0.190 AUC, suggesting that proxy-guided subspace calibration is not specific to CLIP or a single pre-training paradigm.

Robustness to Input Perturbations. We additionally evaluate JPEG compression, Gaussian blur, Gaussian noise, and resizing. COMPASS maintains higher AUC than EFFORT under all tested perturbations, with an average AUC of 0.948 versus 0.927. Detailed results are provided in section A.8.

## 5 CONCLUSION

We present COMPASS, a data-aware subspace adaptation framework for AI-generated image and deepfake detection. Using a small unlabeled Semantic Proxy Set, COMPASS estimates target activation statistics to guide pre-trained weight decomposition, preserving target-relevant represen tations as a stable semantic reference while the residual adapter learns forgery-related variations. Experiments across cross-dataset, cross-method, joint cross-dataset-method, and open-world AIGI settings demonstrate strong generalization, with particularly large gains under simultaneous domain and manipulation shifts. Ablations further show robustness to proxy selection, proxy set size, and different VFM backbones. Overall, our results support covariance-guided subspace preservation as an effective strategy for generalizable forgery detection.

## REFERENCES

Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, and Xiaokang Yang. End-to-end reconstruction-classification learning for face forgery detection. In CVPR, 2022.

Bar Cavia, Eliahu Horwitz, Tal Reiss, and Yedid Hoshen. Real-time deepfake detection in the realworld. arXiv preprint arXiv:2406.09398, 2024.

Liang Chen, Yong Zhang, Yibing Song, Lingqiao Liu, and Jue Wang. Self-supervised learning of adversarial example: Towards good generalizations for deepfake detection. In CVPR, 2022a.

Liang Chen, Yong Zhang, Yibing Song, Jue Wang, and Lingqiao Liu. Ost: Improving generalization of deepfake detection via one-shot test-time training. In NeurIPS, 2022b.

Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. ICML, 2026.

Jikang Cheng, Zhiyuan Yan, Ying Zhang, Yuhao Luo, Zhongyuan Wang, and Chen Li. Can we leave deepfake data behind in training deepfake detector? NeurIPS, 2024.

Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, and Anil K Jain. On the detection of digital face manipulation. In CVPR, 2020.

Chuck Easttom. Deepfake technology: Emerging threats and security implications. In ICCWS, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Jessica Fridrich, David Soukal, Jan Lukas, et al. Detection of copy-move forgery in digital images. In Proceedings of digital forensic research workshop, 2003.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 2020.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020.

Baojin Huang, Zhongyuan Wang, Jifan Yang, Jiaxin Ai, Qin Zou, Qian Wang, and Dengpan Ye. Implicit identity driven deepfake face swapping detection. In CVPR, 2023.

Tero Karras, Miika Aittala, Samuli Laine, Erik Hark¨ onen, Janne Hellsten, Jaakko Lehtinen, and¨ Timo Aila. Alias-free generative adversarial networks. NeurIPS, 2021.

Terro Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. In ICLR, 2018.

Abhishek Kashyap, Megha Agarwal, and Hariom Gupta. Detection of copy-move image forgery using svd and cuckoo search algorithm. arXiv preprint arXiv:1704.00631, 2017.

Pavel Korshunov and Sebastien Marcel. Deepfakes: a new threat to face recognition? assessment´ and detection. arXiv preprint arXiv:1812.08685, 2018.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

Jiaming Li, Hongtao Xie, Jiahong Li, Zhongyuan Wang, and Yongdong Zhang. Frequency-aware discriminative feature learning supervised by single-center loss for face forgery detection. In CVPR, 2021.

Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, and Fuli Feng. Improving synthetic image detection towards generalization: An image transformation perspective. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pp. 2405– 2414, 2025a.

Ziqiang Li, Jiazhen Yan, Ziwen He, Kai Zeng, Weiwei Jiang, Lizhi Xiong, and Zhangjie Fu. Is artificial intelligence generated image detection a solved problem? arXiv preprint arXiv:2505.12335, 2025b.

Shuqiao Liang, Jian Liu, Renzhang Chen, and Quanlong Guan. Ferretnet: Efficient synthetic image detection via local pixel dependencies. In NeurIPS, 2025.

Yuzhen Lin, Wentang Song, Bin Li, Yuezun Li, Jiangqun Ni, Han Chen, and Qiushi Li. Fake it till you make it: Curricular dynamic forgery augmentations towards general deepfake detection. arXiv preprint arXiv:2409.14444, 2024.

Honggu Liu, Xiaodan Li, Wenbo Zhou, Yuefeng Chen, Yuan He, Hui Xue, Weiming Zhang, and Nenghai Yu. Spatial-phase shallow learning: rethinking face forgery detection in frequency domain. In CVPR, 2021.

Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Jingdong Wang, and Yao Zhao. Forgeryaware adaptive transformer for generalizable synthetic image detection. In CVPR, 2024.

Yunpeng Luo, Junlong Du, Ke Yan, and Shouhong Ding. Lare2: Latent reconstruction error based<sup>ˆ</sup> method for diffusion-generated image detection. In CVPR, 2024.

Midjourney. Midjourney version 6. https://www.midjourney.com/, 2023. Released December 20, 2023.

Yunsheng Ni, Depu Meng, Changqian Yu, Chengbin Quan, Dongchun Ren, and Youjian Zhao. Core: Consistent representation learning for face forgery detection. In CVPRW, 2022.

Utkarsh Ojha et al. Towards universal fake image detectors that generalize across generative models. In CVPR, 2023.

Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366, 2022.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe¨ Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In ECCV, 2020.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Neeraj Kumar Rathore, Neelesh Kumar Jain, Prashant Kumar Shukla, UmaShankar Rawat, and Rachana Dubey. Image forgery detection using singular value decomposition with some attacks. National Academy Science Letters, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In CVPR, 2022a.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models, 2022b.

Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. Faceforensics++: Learning to detect manipulated facial images. In ICCV, 2019.

Syamantak Sarkar, Revoti P Bora, Sudhish N George, and Kiran Raja. Unsupervised and generalizable deepfake detection using singular value decomposition. In EUSIPCO, 2025.

Tiancheng Shen, Zilong Huang, Xiangtai Li, Zhijie Lin, Jiyang Liu, Yitong Wang, Jiashi Feng, Ming-Hsuan Yang, and Jun Hao Liew. Qk-edit: Revisiting attention-based injection in mm-dit for image and video editing. In ICCV, 2025.

Kaede Shiohara and Toshihiko Yamasaki. Detecting deepfakes with self-blended images. In CVPR, 2022.

Mohamed R Shoaib, Zefan Wang, Milad Taleby Ahvanooey, and Jun Zhao. Deepfakes, misinformation, and disinformation in the era of frontier ai, generative ai, and large ai models. In ICCA, 2023.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, et al. Di-´ nov3. arXiv preprint arXiv:2508.10104, 2025.

Sonam Singh and Amol Dhumane. Unmasking digital deceptions: An integrative review of deepfake detection, multimedia forensics, and cybersecurity challenges. MethodsX, 2025.

Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. Frequencyaware deepfake detection: Improving generalizability through frequency space domain learning. In AAAI, 2024a.

Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. Rethinking the up-sampling operations in cnn-based generative network for generalizable deepfake detection. In CVPR, 2024b.

Chuangchuang Tan, Renshuai Tao, Huan Liu, Guanghua Gu, Baoyuan Wu, Yao Zhao, and Yunchao Wei. C2P-CLIP: Injecting category common prompt in CLIP to enhance generalization in deepfake detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 7184–7192, 2025.

Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In ICML, 2019.

OpenAI Team. Dall-e 3 ai image generator. https://dalle3.ai/, 2024.

Justus Thies, Michael Zollhofer, Marc Stamminger, Christian Theobalt, and Matthias Nießner. Face2face: Real-time face capture and reenactment of rgb videos. In CVPR, 2016.

Michael E Wall, Andreas Rechtsteiner, and Luis M Rocha. Singular value decomposition and principal component analysis. In A practical approach to microarray data analysis. Springer, 2003.

Hongsong Wang, Renxi Cheng, Yang Zhang, Chaolei Han, and Jie Gui. Lota: Bit-planes guided ai-generated image detection. In ICCV, 2025.

Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot... for now. In CVPR, 2020.

Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. Dire for diffusion-generated image detection. In ICCV, 2023.

Haiwei Wu, Jiantao Zhou, and Shile Zhang. Generalizable synthetic image detection via languageguided contrastive learning. arXiv preprint arXiv:2305.13800, 2023.

Jiazhen Yan, Fan Wang, Weiwei Jiang, Ziqiang Li, and Zhangjie Fu. Ns-net: Decoupling clip semantic information through null-space for generalizable ai-generated image detection. arXiv preprint arXiv:2508.01248, 2025a.

Jiazhen Yan, Ziqiang Li, Fan Wang, Ziwen He, and Zhangjie Fu. Dual frequency branch framework with reconstructed sliding windows attention for ai-generated image detection. IEEE Transactions on Information Forensics and Security, 2026.

Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, and Weidi Xie. A sanity check for ai-generated image detection. arXiv preprint arXiv:2406.19435, 2024a.

Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. arXiv preprint arXiv:2304.13949, 2023a.

Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, and Baoyuan Wu. Deepfakebench: A comprehensive benchmark of deepfake detection. NeurIPS, 2023b.

Zhiyuan Yan, Yuhao Luo, Siwei Lyu, Qingshan Liu, and Baoyuan Wu. Transcending forgery specificity with latent space augmentation for generalizable deepfake detection. In CVPR, 2024b.

Zhiyuan Yan, Taiping Yao, Shen Chen, Yandan Zhao, Xinghe Fu, Junwei Zhu, Donghao Luo, Li Yuan, Chengjie Wang, Shouhong Ding, et al. Df40: Toward next-generation deepfake detection. arXiv preprint arXiv:2406.13495, 2024c.

Zhiyuan Yan, Jiangming Wang, Zhendong Wang, Peng Jin, Ke-Yue Zhang, Shen Chen, Taiping Yao, Shouhong Ding, Baoyuan Wu, and Li Yuan. Effort: Efficient orthogonal modeling for generalizable ai-generated image detection. ICML, 2025b.

Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025c.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.

Chenming Zhou, Jiaan Wang, Yu Li, Lei Li, Juan Cao, and Sheng Tang. Beyond semantic features: Pixel-level mapping for generalized ai-generated image detection. In AAAI, 2025.

## A APPENDIX

## A.1 MATHEMATICAL DETAIL: SYSTEMATIC SCALING OFFSET IN FEATURE SIMILARITY

As observed in the pilot study (fig. 1(b) of the main paper), the cosine similarities in the compressed subspace systematically shift above the $y = x$ diagonal. We provide a geometric explanation for this similarity shift under principal-subspace projection. The analysis proceeds in three stages.

## A.1.1 CONCENTRATION OF MEASURE IN HIGH-DIMENSIONAL SPACE

We first establish the baseline behavior of feature vectors in a high-dimensional space $\mathbb { R } ^ { d }$ . Let $\mathbf { u } , \mathbf { v } \in \mathbb { R } ^ { d }$ be two randomly distributed feature vectors. For simplicity, assume their elements are independent and identically distributed (i.i.d.) random variables with zero mean and variance $\sigma ^ { 2 }$

The expected inner product of these vectors is:

$$
\mathbb { E } [ \langle \mathbf { u } , \mathbf { v } \rangle ] = \mathbb { E } \left[ \sum _ { i = 1 } ^ { d } u _ { i } v _ { i } \right] = \sum _ { i = 1 } ^ { d } \mathbb { E } [ u _ { i } ] \mathbb { E } [ v _ { i } ] = 0\tag{12}
$$

The variance of the inner product is $d \sigma ^ { 4 }$ , implying its magnitude is approximately $\mathcal { O } ( \sqrt { d } \sigma ^ { 2 } )$

According to the Law of Large Numbers, the squared ℓ -norm of u is:

$$
\| \mathbf { u } \| ^ { 2 } = \sum _ { i = 1 } ^ { d } { u _ { i } ^ { 2 } } \approx d \sigma ^ { 2 } \implies \| \mathbf { u } \| \approx { \sqrt { d } } \sigma\tag{13}
$$

Consequently, the magnitude of the original cosine similarity $S _ { o r i g }$ is:

$$
S _ { o r i g } = \cos ( \theta ) = { \frac { \langle \mathbf { u } , \mathbf { v } \rangle } { \| \mathbf { u } \| \| \mathbf { v } \| } } \sim { \mathcal { O } } \left( { \frac { { \sqrt { d } } \sigma ^ { 2 } } { { \sqrt { d } } \sigma \cdot { \sqrt { d } } \sigma } } \right) = { \mathcal { O } } \left( { \frac { 1 } { \sqrt { d } } } \right)\tag{14}
$$

When d is exceptionally large $( \mathbf { e . g . } , d = 1 0 2 4 ) , 1 / \sqrt { d } \approx 0 . 0 3 1$ . This illustrates the concentration ofmeasure phenomenon: in high-dimensional spaces, random vectors are highly likely to be nearly orthogonal, clustering their cosine similarities near zero.

## A.1.2 ORTHOGONAL DECOMPOSITION VIA SVD TRUNCATION

When we apply SVD truncation to retain the top k principal components, we project the original vectors into a lower-dimensional principal subspace. This acts as an orthogonal decomposition:

$$
\mathbf { u } = \mathbf { u } _ { k } + \mathbf { u } _ { \perp } , \quad \mathbf { v } = \mathbf { v } _ { k } + \mathbf { v } _ { \perp }\tag{15}
$$

where $\mathbf { u } _ { k }$ represents the retained principal semantics and $\mathbf { u } _ { \perp }$ represents the discarded residual components (predominantly high-frequency noise).

By the Pythagorean theorem, the squared norm satisfies:

$$
\| \mathbf { u } \| ^ { 2 } = \| \mathbf { u } _ { k } \| ^ { 2 } + \| \mathbf { u } _ { \perp } \| ^ { 2 }\tag{16}
$$

This strictly dictates that the norm decreases after projection: $\| \mathbf { u } _ { k } \| < \| \mathbf { u } \|$

## A.1.3 MECHANISM OF SIMILARITY INFLATION

We can now compare the cosine similarity before and after compression. The original similarity can be expanded as:

$$
S _ { o r i g } = \frac { \langle \mathbf { u } _ { k } , \mathbf { v } _ { k } \rangle + \langle \mathbf { u } _ { \perp } , \mathbf { v } _ { \perp } \rangle } { \| \mathbf { u } \| \| \mathbf { v } \| }\tag{17}
$$

The similarity in the compressed subspace is:

$$
S _ { c o m p } = \frac { \left. \mathbf { u } _ { k } , \mathbf { v } _ { k } \right. } { \left\| \mathbf { u } _ { k } \right\| \left\| \mathbf { v } _ { k } \right\| }\tag{18}
$$

The core rationale of SVD is to capture directions of maximum variance. Therefore, the shared semantic structures between similar images are predominantly preserved within $\mathbf { u } _ { k }$ and $\mathbf { v } _ { k }$ . Conversely, the discarded residual dimensions $\mathbf { u } _ { \perp }$ and $\mathbf { v } _ { \perp }$ act essentially as uncorrelated highdimensional noise.

Referring back to our first stage, the inner product of such uncorrelated high-dimensional noise vectors approaches zero: $\langle \mathbf { u } _ { \perp } , \mathbf { v } _ { \perp } \rangle \approx 0$ . Consequently, the numerator (inner product) remains largely unchanged after truncation:

$$
\langle { \bf u } , { \bf v } \rangle \approx \langle { \bf u } _ { k } , { \bf v } _ { k } \rangle\tag{19}
$$

However, the denominator (product of norms) strictly and significantly decreases due to dimensionality reduction $( \| \mathbf { u } _ { k } \| \| \mathbf { v } _ { k } \| < \| \mathbf { u } \| \| \mathbf { v } \| )$ . In a fraction, preserving the numerator while decreasing the denominator can lead to an increase in the overall value:

$$
S _ { c o m p } = { \frac { \langle \mathbf { u } _ { k } , \mathbf { v } _ { k } \rangle } { \| \mathbf { u } _ { k } \| \| \mathbf { v } _ { k } \| } } \approx { \frac { \langle \mathbf { u } , \mathbf { v } \rangle } { \| \mathbf { u } _ { k } \| \| \mathbf { v } _ { k } \| } } > { \frac { \langle \mathbf { u } , \mathbf { v } \rangle } { \| \mathbf { u } \| \| \mathbf { v } \| } } = S _ { o r i g }\tag{20}
$$

Conclusion & Empirical Alignment: The systematic scaling and distribution of cosine similarities in fig. 1(b). perfectly align with our mathematical derivation regarding subspace truncation.

First, the similarity points in COMPASS are more concentrated (Spearman rank correlation $\rho =$ 0.5533) compared to the broader scatter of EFFORT (Yan et al., 2025b) $( \rho = 0 . 1 3 4 5 )$ . The higher rank correlation suggests that COMPASS better preserves the relative similarity structure of the original space. By utilizing covariance-guided calibration, COMPASS better preserves the pretrained representation, perfectly preserving the meaningful numerator (shared semantics $\langle { \bf { u } } _ { k } , { \bf { v } } _ { k } \rangle )$ without disrupting structural relationships.

Second, the upward drift of similarities in COMPASS is more pronounced. Magnitude-based SVD prioritizes components according to their singular values. Its reduction of the vector magnitude (the denominator) is inherently limited, leading to a less-shifted but collapsed representation. In contrast, COMPASS discards components based on semantic irrelevance. It intentionally strips away “magnitude-heavy but semantically useless” domain biases. By discarding a much larger proportion of absolute magnitude (reducing the feature norms more strongly), COMPASS can therefore produce a larger upward shift in cosine similarity after normalization.

## A.2 QUALITATIVE ANALYSIS VIA GRAD-CAM

To qualitatively examine the spatial attention patterns produced by EFFORT and COMPASS, we visualize Grad-CAM responses on a representative fake sample. As shown in fig. 4, EFFORT exhibits a relatively dispersed attention pattern that extends to background and clothing regions in addition to the face. In contrast, COMPASS places more attention on the central facial region, including the eyes, nose, and mouth boundaries.

This observation is consistent with our motivation that better preservation of the pre-trained semantic reference can reduce reliance on source-specific contextual cues during residual adaptation. We emphasize that Grad-CAM provides only qualitative evidence; the generalization benefits of COM-PASS are established by the quantitative cross-dataset and cross-method experiments in the main paper.

## A.3 ALGORITHM OVERVIEW

To provide a clearer step-by-step implementation perspective, we summarize the entire workflow of the proposed COMPASS framework in Algorithm 1. Unlike static orthogonal methods that rely on heavy constraint losses during optimization, COMPASS performs proxy-guided subspace calibration prior to fine-tuning. Furthermore, by optimizing only the decomposed low-rank factors of the residual space rather than the full parameter matrix, it inherently maintains a structural bottleneck, preventing overfitting to domain biases.

## A.4 DETAILED IMPLEMENTATION AND REPRODUCIBILITY

To ensure full reproducibility, we provide comprehensive implementation details of the proposed COMPASS framework across two diverse domains.

Input  
EFFORT  
COMPASS (Ours)  
![](images/75f8502014a75e526082b4a6696fa832f07814ee737ced399f2727ac0faff6a2.jpg)  
Figure 4: Grad-CAM Visualization on Deepfake Detection. Left: Input fake image. Middle: EF-FORT exhibits relatively dispersed attention across facial and contextual regions. Right: COMPASS places more attention on the central facial region.

Algorithm 1 COMPASS Approach Algorithm   
Input: Pre-trained weight matrix $W \in \mathbb { R } ^ { d _ { o u t } \times d _ { i n } } ;$ Rank k; Training data $\mathcal { D } _ { s r c } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { M } ;$   
Unlabeled calibration set $\mathcal { D } _ { p }$ of size $N _ { p }$   
Output: Updated weight matrix W   
1: ▷ Step 1: Proxy-Guided Subspace Calibration (Zero-Backpropagation)   
2: Pass $\mathcal { D } _ { c a l }$ through the frozen VFM to extract activations $X _ { p } \in \mathbb { R } ^ { \dot { d } _ { i n } \times N _ { p } }$   
3: Mean-center the activations: $\hat { X } _ { p } = X _ { p } - \bar { X } _ { p }$   
4: Compute the activation covariance: $\begin{array} { r } { \dot { C } = \frac { 1 } { N _ { p } } \dot { X } _ { p } \hat { X } _ { p } ^ { \top } } \end{array}$   
5: ▷ Step 2: Covariance-Guided SVD   
6: Decompose proxy-scaled weights via SVD: $W C = U ^ { \prime } S ^ { \prime } V ^ { \prime } { } ^ { \top }$   
7: Retain the top k singular values and vectors: $\bar { U _ { k } ^ { \prime } } \in \mathbb { R } ^ { \bar { d } _ { o u t } \times k } , S _ { k } ^ { \prime } \in \mathbb { R } ^ { k \times k } , V _ { k } ^ { \prime } \in \mathbb { R } ^ { d _ { i n } \times k }$   
8: Compute the principal weights (Semantic Compass): $W _ { p r i n } = U _ { k } ^ { \prime } S _ { k } ^ { \prime } V _ { k } ^ { \prime } { } ^ { \top } C ^ { - 1 }$   
9: Keep $W _ { p r i n }$ strictly fixed during training   
10: Obtain the residual components for the adapter: $U _ { r } ^ { \prime } , S _ { r } ^ { \prime } , V _ { r } ^ { \prime }$   
11: Initialize trainable low-rank factors:   
12: $U _ { t r a i n }  U _ { r } ^ { \prime }$   
13: $S _ { t r a i n }  S _ { r } ^ { j }$   
14: $V _ { t r a i n }  V _ { r } ^ { \prime } { } ^ { \top } C ^ { - 1 }$   
15: ▷ Step 3: Training Loop   
16: for each epoch do   
17: for each batch in $\mathcal { D } _ { s r c }$ do   
18: ▷ Forward Pass   
19: Compute layer output using $Y = ( W _ { p r i n } + U _ { t r a i n } S _ { t r a i n } V _ { t r a i n } ) X _ { s r c }$   
20: Compute the binary cross-entropy loss $\mathcal { L } _ { B C E }$   
21: ▷ Backward Pass and Optimization   
22: Update only $U _ { t r a i n } , S _ { t r a i n } ,$ and $V _ { t r a i n }$ using gradient descent to minimize $\mathcal { L } _ { C E }$   
23: end for   
24: end for   
25: return Updated weight matrix $W \gets W _ { p r i n } + U _ { t r a i n } S _ { t r a i n } V _ { t r a i n }$

Architecture & Hardware: Consistent with the baseline EFFORT, we adopt the pre-trained CLIP ViT-L/14 (Radford et al., 2021) as our Vision Foundation Model (VFM) backbone. To accommodate the different scales of the training sets, the deepfake detection models are trained on a single NVIDIA RTX 4090 GPU (24GB VRAM), while the large-scale AIGI detection models are trained on a single NVIDIA RTX A6000 GPU (48GB VRAM).

Optimization & Hyperparameters: During the decoupled forgery feature learning phase, the prior-aligned principal weights $W _ { p r i n }$ remain strictly frozen to preserve the principal weight component as a stable semantic reference. We exclusively optimize the low-rank residual adapter. To maintain parameter efficiency and prevent overfitting to domain-specific biases, the residual adapter rank is set to 1. The model is optimized using the Adam optimizer. For the deepfake video detection training phase, the batch size is set to 32 with a learning rate of $2 \times 1 0 ^ { - 4 }$ . For the large-scale openworld AIGI detection task, the batch size is set to 48, with the learning rate adjusted to $1 \times 1 0 ^ { - 4 }$

Calibration Phase: The Semantic Proxy Set is constructed using randomly sampled, unlabeled images from the target-category training distribution. The covariance-guided Covariance-Guided SVD is performed entirely offline prior to the training phase, requiring zero backpropagation and introducing no gradient optimization overhead.

## A.5 COMPUTATIONAL COST AND COMPLEXITY ANALYSIS

A primary advantage of COMPASS is its exceptional parameter efficiency and negligible computational overhead compared to full fine-tuning paradigms.

Parameter Efficiency: By restricting the trainable parameters exclusively to the low-rank residual adapter, COMPASS drastically reduces the optimization burden. Based on our model structure, the trainable parameters account for only 0.38% of the full ViT-L/14 backbone (specifically, approximately 1.18M trainable parameters out of a total 304M parameters).

Table 6: AP Results on AIGIBench. All detectors are trained on the Setting-II of AIGIBench.
<table><tr><td>Generator</td><td>CNN-Spot</td><td>UnivFD</td><td>FreqNet</td><td>NPR</td><td>Ladeda</td><td>AIDE</td><td>C2P-CLIP</td><td>DFfreq</td><td>SAFE</td><td>Effort</td><td>NS-Net</td><td>FerretNet</td><td>LOTA</td><td>BSF</td><td>Ours</td></tr><tr><td>ProGAN</td><td>99.9</td><td>99.9</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.6</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.9</td><td>100.0</td><td>100.0</td></tr><tr><td>R3GAN</td><td>52.4</td><td>91.2</td><td>55.8</td><td>61.1</td><td>72.6</td><td>97.1</td><td>91.3</td><td>90.6</td><td>99.2</td><td>98.2</td><td>91.3</td><td>99.0</td><td>94.7</td><td>93.9</td><td>99.7</td></tr><tr><td>StyleGAN3</td><td>85.2</td><td>84.5</td><td>92.4</td><td>91.7</td><td>96.9</td><td>91.4</td><td>99.5</td><td>98.1</td><td>94.7</td><td>98.9</td><td>99.4</td><td>92.2</td><td>96.5</td><td>95.8</td><td>99.5</td></tr><tr><td>StyleGAN-XL</td><td>50.2</td><td>93.3</td><td>83.7</td><td>75.3</td><td>93.1</td><td>93.2</td><td>98.9</td><td>69.5</td><td>90.8</td><td>97.0</td><td>96.1</td><td>95.6</td><td>97.4</td><td>97.2</td><td>98.7</td></tr><tr><td>StyleSwim</td><td>76.2</td><td>95.2</td><td>91.6</td><td>94.9</td><td>98.5</td><td>89.3</td><td>99.6</td><td>99.5</td><td>96.7</td><td>99.1</td><td>97.8</td><td>99.8</td><td>96.4</td><td>98.6</td><td>99.9</td></tr><tr><td>WFIR</td><td>65.5</td><td>83.0</td><td>57.0</td><td>65.5</td><td>86.9</td><td>90.8</td><td>99.6</td><td>89.0</td><td>60.7</td><td>100.0</td><td>99.4</td><td>98.1</td><td>78.8</td><td>51.5</td><td>98.6</td></tr><tr><td>BlendFace</td><td>73.2</td><td>35.2</td><td>34.0</td><td>34.7</td><td>42.1</td><td>54.2</td><td>48.9</td><td>42.3</td><td>44.1</td><td>66.1</td><td>54.4</td><td>51.7</td><td>45.9</td><td>47.9</td><td>68.9</td></tr><tr><td>E4S</td><td>68.7</td><td>57.0</td><td>34.5</td><td>34.4</td><td>49.3</td><td>44.3</td><td>48.3</td><td>41.0</td><td>45.6</td><td>76.9</td><td>49.4</td><td>42.6</td><td>44.1</td><td>38.4</td><td>79.1</td></tr><tr><td>FaceSwap</td><td>58.3</td><td>52.3</td><td>42.9</td><td>43.6</td><td>40.9</td><td>56.3</td><td>70.4</td><td>64.9</td><td>69.6</td><td>85.8</td><td>62.5</td><td>71.6</td><td>73.4</td><td>66.4</td><td>88.6</td></tr><tr><td>InSwap</td><td>77.5</td><td>40.1</td><td>41.8</td><td>40.7</td><td>47.4</td><td>54.6</td><td>65.6</td><td>57.3</td><td>62.3</td><td>83.1</td><td>67.5</td><td>59.6</td><td>52.4</td><td>49.5</td><td>83.2</td></tr><tr><td>SimSwap</td><td>69.7</td><td>40.3</td><td>41.7</td><td>42.7</td><td>42.3</td><td>62.7</td><td>68.5</td><td>57.7</td><td>59.9</td><td>89.7</td><td>69.5</td><td>68.7</td><td>55.7</td><td>62.9</td><td>88.6</td></tr><tr><td>DALLE-3</td><td>68.4</td><td>76.4</td><td>60.2</td><td>70.0</td><td>59.8</td><td>63.1</td><td>56.3</td><td>48.5</td><td>49.5</td><td>60.6</td><td>42.2</td><td>55.4</td><td>44.3</td><td>43.0</td><td>92.3</td></tr><tr><td>FLUX1-dev</td><td>72.1</td><td>79.5</td><td>87.0</td><td>99.0</td><td>98.7</td><td>93.4</td><td>84.3</td><td>98.2</td><td>99.0</td><td>74.4</td><td>94.1</td><td>99.2</td><td>98.0</td><td>97.5</td><td>84.8</td></tr><tr><td>Midjourney-V6</td><td>59.8</td><td>61.5</td><td>55.9</td><td>76.9</td><td>86.9</td><td>83.0</td><td>84.6</td><td>94.2</td><td>94.5</td><td>90.0</td><td>90.0</td><td>96.5</td><td>91.0</td><td>86.2</td><td>91.4</td></tr><tr><td>GLIDE</td><td>59.7</td><td>80.2</td><td>76.5</td><td>94.3</td><td>95.0</td><td>97.7</td><td>94.9</td><td>97.7</td><td>96.1</td><td>96.5</td><td>96.1</td><td>99.7</td><td>93.9</td><td>98.4</td><td>97.2</td></tr><tr><td>Imagen3</td><td>57.0</td><td>79.3</td><td>80.2</td><td>94.4</td><td>97.2</td><td>95.2</td><td>93.9</td><td>84.7</td><td>97.0</td><td>97.8</td><td>95.9</td><td>97.3</td><td>88.5</td><td>94.0</td><td>96.9</td></tr><tr><td>SD3</td><td>72.7</td><td>87.2</td><td>81.9</td><td>97.2</td><td>98.7</td><td>98.3</td><td>97.1</td><td>97.0</td><td>99.3</td><td>97.9</td><td>98.9</td><td>98.8</td><td>95.1</td><td>96.4</td><td>90.6</td></tr><tr><td>SDXL</td><td>63.8</td><td>88.0</td><td>95.0</td><td>94.4</td><td>98.5</td><td>95.7</td><td>97.1</td><td>99.1</td><td>99.4</td><td>98.0</td><td>98.2</td><td>99.9</td><td>99.4</td><td>98.0</td><td>94.4</td></tr><tr><td>BLIP</td><td>92.8</td><td>95.8</td><td>99.8</td><td>100.0</td><td>99.9</td><td>99.5</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.9</td><td>100.0</td><td>99.9</td></tr><tr><td>Infinite-ID</td><td>49.1</td><td>89.6</td><td>73.6</td><td>80.4</td><td>76.9</td><td>94.7</td><td>84.2</td><td>97.4</td><td>98.5</td><td>85.7</td><td>91.8</td><td>99.7</td><td>98.1</td><td>97.0</td><td>97.2</td></tr><tr><td>PhotoMaker</td><td>57.8</td><td>72.2</td><td>74.2</td><td>53.4</td><td>90.7</td><td>95.6</td><td>69.7</td><td>99.0</td><td>96.9</td><td>69.2</td><td>67.8</td><td>99.1</td><td>97.3</td><td>95.5</td><td>96.7</td></tr><tr><td>Instant-ID</td><td>79.9</td><td>93.5</td><td>86.0</td><td>79.2</td><td>90.4</td><td>96.3</td><td>84.9</td><td>99.9</td><td>94.8</td><td>87.1</td><td>87.7</td><td>100.0</td><td>98.5</td><td>99.0</td><td>98.8</td></tr><tr><td>IP-Adapter</td><td>65.5 50.6</td><td>87.3 55.2</td><td>79.4 58.1</td><td>91.7</td><td>94.3</td><td>95.4</td><td>95.4</td><td>98.8</td><td>94.8</td><td>98.1</td><td>96.2</td><td>99.9</td><td>97.6</td><td>98.7</td><td>91.3</td></tr><tr><td>SocialRF</td><td>56.7</td><td>73.7</td><td>63.5</td><td>68.4 62.9</td><td>68.3 56.3</td><td>65.0</td><td>67.7 53.2</td><td>66.9 64.2</td><td>63.5 64.3</td><td>65.2</td><td>74.6</td><td>68.0</td><td>65.0</td><td>64.0</td><td>65.2</td></tr><tr><td>CommunityAI Average</td><td>67.3</td><td>75.7</td><td>69.9</td><td>73.9</td><td>79.3</td><td>61.0 82.7</td><td>82.2</td><td>82.2</td><td>82.9</td><td>64.3 87.2</td><td>54.0 83.0</td><td>53.0 85.8</td><td>49.6 82.1</td><td>57.5 81.1</td><td>70.3 90.9</td></tr></table>

Training Overhead & Memory Footprint: The proxy-guided subspace calibration step is a highly lightweight, one-time offline process. Extracting features for the minimal proxy set and computing the covariance-guided SVD takes only a few seconds on a single GPU. During the fine-tuning phase, thanks to the low-rank structural bottleneck, COMPASS operates with remarkable resource efficiency:

• Deepfake Detection: Operating on the RTX 4090 with a batch size of 32, the peak GPU memory utilization is tightly constrained to approximately 15 GB. One full epoch is processed rapidly in roughly 16 minutes.

• AIGI Detection: Handling the significantly larger AIGIBench on the RTX A6000 with a batch size of 48, the peak GPU memory footprint is maintained at around 22 GB. Despite the massive scale of the data, one full epoch requires only about 87 minutes.

Inference Complexity (FLOPs): Since the trainable adapter ∆W and the frozen principal weights $W _ { p r i n }$ are strictly parallel linear structures, they can be seamlessly re-parameterized into a single weight matrix $W _ { d e p l o y } = W _ { p r i n } + \Delta W$ during the deployment phase. Consequently, COMPASS introduces absolutely zero additional inference latency or FLOPs overhead compared to the base zero-shot CLIP model, making it exceptionally well-suited for real-time, real-world generative artifact detection scenarios.

## A.6 FULL RESULTS ON AIGIBENCH

We conduct experiments on the comprehensive AIGIBench (Li et al., 2025b). We adopt the standard Setting-II Protocol (Li et al., 2025b) of AIGIBench, which includes 144k images generated by Pro-GAN (Karras et al., 2018) and SDv1.4 (Rombach et al., 2022b), along with real images, to ensure a rigorous assessment. This evaluation encompasses 25 diverse generative models, ranging from GANs (e.g., ProGAN (Karras et al., 2018), StyleGAN3 (Karras et al., 2021)) to the latest advanced Diffusion Models (e.g., Midjourney-V6 (Midjourney, 2023), DALLE-3 (Team, 2024), SD3 (Esser et al., 2024), SDXL (Podell et al., 2023)).

We compare our approach against 14 competitive detectors. This includes widely recognized methods such as CNN-Spot (Wang et al., 2020), UnivFD (Ojha et al., 2023), FreqNet (Tan et al., 2024a), NPR (Tan et al., 2024b), Ladeda (Cavia et al., 2024), AIDE (Yan et al., 2024a), C2P-CLIP (Tan et al., 2025), DFfreq (Yan et al., 2026), SAFE (Li et al., 2025a), EFFORT (Yan et al., 2025b); and the latest SOTA detectors including NS-Net (Yan et al., 2025a), FerretNet (Liang et al., 2025), LOTA (Wang et al., 2025), and BSF (Zhou et al., 2025).

table 6 reports the complete per-generator AP results for COMPASS and 14 representative AIGI detectors. While the main paper presents a subset of representative generators for compactness, the full results show that COMPASS maintains strong overall generalization across the diverse generator set.

Table 7: Detailed robustness analysis of proxy-source selection. For each Face Swapping (FS) or Face Reenactment (FR) setting, two datasets are selected as proxy sources and a total of 80 unlabeled images are sampled from their training splits. We report per-dataset AUC for different proxy-source pairs together with representative baselines.  
(a) Face Swapping (FS) Category
<table><tr><td rowspan="2">Testing Dataset</td><td colspan="3">Baseline</td><td colspan="4">Ours (Calibrated on 2 Datasets)</td></tr><tr><td>CORE</td><td>UCF</td><td>Effort</td><td>UniFace + BleFace</td><td>MobSwap + e4s</td><td>FaceDan + FSGAN</td><td>InSwap + SimSwap</td></tr><tr><td>UniFace</td><td>0.871</td><td>0.831</td><td>0.962</td><td>0.981</td><td>0.974</td><td>0.982</td><td>0.977</td></tr><tr><td>BleFace</td><td>0.844</td><td>0.827</td><td>0.873</td><td>0.941</td><td>0.925</td><td>0.939</td><td>0.928</td></tr><tr><td>MobSwap</td><td>0.959</td><td>0.950</td><td>0.953</td><td>0.971</td><td>0.964</td><td>0.985</td><td>0.969</td></tr><tr><td>e4s</td><td>0.680</td><td>0.732</td><td>0.982</td><td>0.988</td><td>0.982</td><td>0.989</td><td>0.983</td></tr><tr><td>FaceDan</td><td>0.774</td><td>0.862</td><td>0.920</td><td>0.971</td><td>0.958</td><td>0.965</td><td>0.966</td></tr><tr><td>FSGAN</td><td>0.958</td><td>0.937</td><td>0.957</td><td>0.980</td><td>0.973</td><td>0.985</td><td>0.977</td></tr><tr><td>InSwap</td><td>0.855</td><td>0.809</td><td>0.936</td><td>0.946</td><td>0.932</td><td>0.953</td><td>0.948</td></tr><tr><td>SimSwap</td><td>0.724</td><td>0.647</td><td>0.926</td><td>0.948</td><td>0.938</td><td>0.950</td><td>0.954</td></tr><tr><td>Average</td><td>0.833</td><td>0.824</td><td>0.939</td><td>0.966</td><td>0.956</td><td>0.969</td><td>0.963</td></tr></table>

(b) Face Reenactment (FR) Category
<table><tr><td rowspan="2">Testing Dataset</td><td colspan="3">Baseline</td><td colspan="6">Ours (Calibrated on 2 Datasets)</td></tr><tr><td>CORE</td><td>UCF</td><td>Effort</td><td>FOMM +FV2V</td><td>Wav2Lip + MRAA</td><td>OneShot + PiRender</td><td>TPSMM + LIA</td><td>DaGan + SadTalker</td><td>MCNet + HyperReenact</td></tr><tr><td>FOMM</td><td>0.932</td><td>0.889</td><td>0.901</td><td>0.949</td><td>0.945</td><td>0.962</td><td>0.959</td><td>0.958</td><td>0.963</td></tr><tr><td>FV2V</td><td>0.822</td><td>0.794</td><td>0.810</td><td>0.873</td><td>0.852</td><td>0.866</td><td>0.855</td><td>0.899</td><td>0.914</td></tr><tr><td>Wav2Lip</td><td>0.899</td><td>0.874</td><td>0.733</td><td>0.861</td><td>0.881</td><td>0.885</td><td>0.896</td><td>0.882</td><td>0.883</td></tr><tr><td>MRAA</td><td>0.983</td><td>0.978</td><td>0.946</td><td>0.991</td><td>0.986</td><td>0.994</td><td>0.989</td><td>0.993</td><td>0.993</td></tr><tr><td>OneShot</td><td>0.861</td><td>0.818</td><td>0.802</td><td>0.890</td><td>0.898</td><td>0.917</td><td>0.905</td><td>0.916</td><td>0.913</td></tr><tr><td>PiRender</td><td>0.901</td><td>0.862</td><td>0.804</td><td>0.943</td><td>0.916</td><td>0.924</td><td>0.918</td><td>0.935</td><td>0.920</td></tr><tr><td>TPSMM</td><td>0.816</td><td>0.792</td><td>0.758</td><td>0.860</td><td>0.822</td><td>0.865</td><td>0.849</td><td>0.869</td><td>0.888</td></tr><tr><td>LIA</td><td>0.926</td><td>0.913</td><td>0.915</td><td>0.957</td><td>0.947</td><td>0.968</td><td>0.960</td><td>0.974</td><td>0.969</td></tr><tr><td>DaGan</td><td>0.731</td><td>0.753</td><td>0.742</td><td>0.821</td><td>0.789</td><td>0.834</td><td>0.812</td><td>0.830</td><td>0.854</td></tr><tr><td>SadTalker</td><td>0.715</td><td>0.662</td><td>0.594</td><td>0.710</td><td>0.666</td><td>0.701</td><td>0.687</td><td>0.681</td><td>0.672</td></tr><tr><td>MCNet</td><td>0.833</td><td>0.813</td><td>0.769</td><td>0.864</td><td>0.833</td><td>0.876</td><td>0.867</td><td>0.883</td><td>0.899</td></tr><tr><td>HyperReenact</td><td>0.878</td><td>0.747</td><td>0.926</td><td>0.902</td><td>0.913</td><td>0.911</td><td>0.931</td><td>0.928</td><td>0.922</td></tr><tr><td>Average</td><td>0.858</td><td>0.825</td><td>0.808</td><td>0.885</td><td>0.871</td><td>0.892</td><td>0.886</td><td>0.896</td><td>0.899</td></tr></table>

## A.7 DETAILED RESULTS ON PROXY-SOURCE SELECTION

For completeness, we report the full per-dataset results corresponding to the proxy-source robustness analysis in table 4(a). For each manipulation category, we randomly select two datasets and sample a total of 80 unlabeled images from their training splits to construct the Semantic Proxy Set. The proxy images are used only to estimate activation statistics, and no test images are involved in calibration. We repeat this procedure with different dataset pairs and evaluate the resulting model on all datasets within the same manipulation category.

As shown in table 7, COMPASS remains consistently effective across different proxy-source pairs. For Face Swapping, the average AUC ranges from 0.956 to 0.969 across four proxy pairs, compared with 0.939 for EFFORT. For Face Reenactment, the corresponding range is 0.871–0.899 across six proxy pairs, compared with 0.808 for EFFORT. The detailed results support the conclusion in the main paper that COMPASS is relatively insensitive to the particular datasets used to construct a target-relevant proxy set.

## A.8 REAL-WORLD PERTURBATIONS

We evaluated the robustness of COMPASS under severe common unseen corruptions in table 8: JPEG compression (Quality=80), Gaussian Blur $( \sigma = 0 . 5 , k = 5 )$ , Gaussian Noise $( \sigma = 0 . 0 1 )$ , and Resizing (0.75×). In the table below, COMPASS demonstrates remarkable resilience and consistently outperforms the baseline (EFFORT) across all distortions.

Table 8: Robustness evaluation under real-world perturbations.
<table><tr><td>Method</td><td>Clean</td><td>JPEG (80)</td><td>Blur  $\overline { { ( \sigma \mathrm { = } 0 . 5 ) } }$ </td><td>Noise (0.01)</td><td>Resize (0.75×)</td><td>Avg.</td></tr><tr><td>EFFORT</td><td>0.940</td><td>0.927</td><td>0.936</td><td>0.913</td><td>0.920</td><td>0.927</td></tr><tr><td>COMPASS</td><td>0.966</td><td>0.947</td><td>0.949</td><td>0.930</td><td>0.946</td><td>0.948</td></tr></table>