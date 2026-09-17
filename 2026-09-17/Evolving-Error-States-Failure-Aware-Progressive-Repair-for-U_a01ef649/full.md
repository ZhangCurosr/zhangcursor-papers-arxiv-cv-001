# Evolving Error States: Failure-Aware Progressive Repair for Ultrasound Lesion Segmentation

Ziliang Wang<sup>1</sup>, XuJiang Tang<sup>2</sup>, Lu Yuting<sup>3</sup>, Weixin Xu<sup>3</sup>, Yongqiang Zhao<sup>4</sup>, Ying Fu<sup>5</sup>, Kehua Guo<sup>1∗</sup>

<sup>1</sup>Central South University, <sup>2</sup>Yangtze University, <sup>3</sup>Chongqing University, <sup>4</sup>Peking University, <sup>5</sup>Southwest Jiaotong University

## Abstract

Reliability under sparse and heterogeneous failures remains a fundamental challenge for medical image segmentation. High average accuracy can conceal a small set of structurally distinct and clinically consequential errors. Existing post-hoc correction methods alleviate this problem, but typically estimate false-positive and false-negative corrections from the same fixed prediction. This ignores the dynamic evolution of error states and limits the correction of complex cases. Inspired by iterative error feedback in structured prediction, we propose Failure-Aware Progressive Repair (FAPR). FAPR represents the current segmentation mask as a dynamic failure state and models each repair operation as a state-transition operator. Each accepted correction forms a new prediction state for subsequent error diagnosis and repair, enabling later operations to adapt to preceding changes. Conditional routing selectively activates necessary state transitions, while failure replay exposes the model to rare error states. By keeping the base segmentor frozen, FAPR preserves its established segmentation capability while improving dificult cases. Across three public ultrasound lesion segmentation benchmarks, FAPR improves mean DSC by 1.52%. On the very-hard subsets of BUSI and TN3K, the average gain reaches 13.77%.

## Introduction

Encoder–decoder networks, represented by U-Net (Ronneberger, Fischer, and Brox 2015) and U-KAN (Li et al. 2025), have achieved strong average performance in medical image segmentation through multi-scale feature extraction and cross-level feature fusion. However, when applied to ultrasound lesions with weak boundaries, heterogeneous appearance, and substantial shape variation, these models remain susceptible to gross lesion misses, regional undersegmentation, and false-positive predictions. The long-tailed distribution of severe errors further limits their exposure during conventional training. Consequently, high average segmentation accuracy does not necessarily ensure reliable and stable predictions for dificult cases in high-risk clinical settings.

U-Net-style architectures remain a prevalent design paradigm for medical segmentation (Zhou et al. 2018; Luo et al. 2025), and recent models such as U-KAN have demonstrated competitive segmentation capability. Most subsequent studies pursue further gains by adding specialized components or introducing increasingly elaborate network designs, as exemplified by MSAGHNet. However, this architecture-centered line of research is primarily evaluated through aggregate metrics and rarely examines whether the same dificult samples remain unresolved. As a result, improvements in mean DSC can coexist with a persistent tail of gross misses, incomplete lesion coverage, and false-positive errors.

![](images/6dd697b27637d0d27940922fb6a1c03ff1e45dbdda99c7c671794c0e8ef05111.jpg)  
Figure 1: Persistent hard-case tails on BUSI. (a) Persample DSC distributions with model averages and parameter counts. (b) Among cases below 0.70 DSC, 17 are shared by CausalBridgeNet, U-KAN, and MSAGHNet.

Recent studies have therefore introduced post-hoc correction to repair residual errors without redesigning the base segmentor. CausalBridgeNet (Yang et al. 2026), for example, estimates false-positive and false-negative corrections from the initial prediction. However, these signals are inferred in parallel from the same fixed state, ignoring how one correction changes the error context for the next. Figure 1 summarizes this limitation on BUSI: despite stronger representations in U-KAN and MSAGHNet and the additional post-hoc correction in CausalBridgeNet, their low-accuracy tails remain highly overlapping, with 17 cases below 0.70 DSC for all three methods. Thus, neither architectural scaling nor static correction adequately resolves recurrent structured failures. Once a grossly missed region is recovered, subsequent completion and suppression must instead adapt to the updated spatial support.

Inspired by iterative error feedback (Carreira et al. 2016), we reinterpret post-hoc correction as an ordered composition of state-transition operators. Unlike conventional mixtureof-experts systems, in which exchangeable experts independently process the same input and compete for selection (Jacobs et al. 1991; Riquelme et al. 2021; Wang et al. 2025), FAPR assigns heterogeneous experts complementary failure semantics. Each expert diagnoses the latest accepted mask and transforms it into a new prediction state. The repair operators are therefore order-sensitive: recovering a gross miss changes the spatial support available for false-negative completion, which subsequently changes the regions considered by false-positive suppression.

Based on this formulation, we propose Failure-Aware Progressive Repair (FAPR). It decomposes residual errors into gross misses, false-negative incompletion, and false-positive contamination, and organizes the corresponding specialists into an ordered repair chain. Sample- and pixel-level gates control whether and where each transition is accepted, while failure replay provides targeted supervision for rare error states. This design changes expert collaboration from competitive selection to compositional state transition over an evolving segmentation mask.

Our contributions are threefold:

• We introduce a state-dependent formulation of medical segmentation repair, modeling correction as progressive transitions between error states rather than a one-shot update of a fixed prediction. It formalizes how each correction changes the spatial support and error context of subsequent operations, providing a principled basis for progressive repair.

• We instantiate this formulation with three nonexchangeable specialists for gross-miss recovery, falsenegative completion, and false-positive suppression. Dual-level routing controls each transition, while failure replay enables the specialists to learn rare and heterogeneous failure states.

• We validate FAPR across three ultrasound lesion benchmarks and diferent frozen segmentation backbones. Matched controls involving parallel fusion, shared refiners, static-state conditioning, and reversed repair order demonstrate that the gains arise from failure-specific specialization and evolving-state composition rather than additional model capacity.

## Related Work

Medical lesion segmentation. Medical lesion segmentation has progressed from convolutional encoder–decoder networks toward architectures that incorporate attention, global context, and multi-scale feature interaction (Hatamizadeh et al. 2021, 2022; Li et al. 2024; Miao et al. 2023; Xu and Wang 2025; Lu et al. 2025). U-Net established a symmetric encoder–decoder architecture with skip connections for recovering spatial details (Ronneberger, Fischer, and Brox 2015) (U-Net). Attention U-Net introduced attention gates to emphasize task-relevant regions during feature fusion (Oktay et al. 2018) (Attention U-Net). TransUNet combined convolutional representations with Transformer-based global context modeling (Chen et al. 2021) (TransUNet). More recently, MSAGHNet integrated multi-resolution CNN–Transformer representations with scale-guided attention to strengthen feature interaction across spatial scales (Zhu et al. 2025)

(MSAGHNet). Although these architectures progressively improve representation learning and average segmentation accuracy, they remain single-pass predictors and do not explicitly revisit heterogeneous residual failures after the initial prediction.

Post-hoc segmentation correction. Post-hoc refinement improves an existing segmentation without redesigning its primary prediction pathway. SegFix corrects unreliable boundary pixels using learned directions toward more reliable interior regions (Yuan et al. 2020) (SegFix). SegRefiner treats coarse-mask refinement as a generative process and progressively updates pixel labels through discrete diffusion (Wang et al. 2023) (SegRefiner). In medical imaging, HiDif combines a discriminative segmentor with a binary Bernoulli difusion refiner conditioned on its mask prior (Chen et al. 2024) (HiDif ). CausalBridgeNet freezes the foundation model and estimates structured false-positive and false-negative correction signals through a predictive causal reasoning unit (Yang et al. 2026) (CausalBridgeNet). These studies establish the value of correcting an initial prediction, but they either target a local error type, employ a homogeneous generic refiner, or derive multiple correction signals from the same prediction prior. They do not explicitly formulate post-hoc correction as ordered transitions among heterogeneous failure states.

Iterative error feedback. Iterative Error Feedback introduced a self-correcting formulation in which predicted errors are fed back to progressively update an initial structured output (Carreira et al. 2016) (Iterative Error Feedback). Recurrent Mask Refinement subsequently demonstrated in fewshot medical segmentation that the mask produced at one iteration can be reused to recapture changed foreground– background context at the next iteration (Tang et al. 2021) (Recurrent Mask Refinement). FAPR builds on this stateupdating principle but applies it to automatic post-hoc correction with a frozen segmentor. Instead of repeatedly invoking one shared refiner, it assigns gross-miss recovery, false-negative completion, and false-positive suppression to ordered failure-specific experts, so each repair operates on the state produced by its predecessor.

## Method

## Overview

We propose Failure-Aware Progressive Repair (FAPR), a modular post-hoc framework for correcting a frozen medical image segmentor. Unlike one-shot methods that infer all corrections from the initial prediction, FAPR represents the mask as an evolving failure state and updates it through an ordered pathway of gross-miss recovery, false-negative completion, and false-positive suppression.

As shown in Figure 2, Stage 1 produces the initial mask $M _ { 0 }$ and frozen multi-scale features. At stage k, a failurespecific expert predicts a candidate $Q _ { k }$ from the latest state $M _ { k - 1 }$ and the shared features; sample- and pixel-level routing determine whether and where it is incorporated to form $M _ { k } . \mathrm { ~ A ~ }$ lightweight final controller rolls harmful trajectories back to $\bar { M } _ { 0 }$ . Failure replay based on observed Stage-1 errors and controlled perturbations exposes the specialists to rare failures without modifying the base segmentor.

![](images/9ebb067979bc2e7fb104c404100c3cc4a40908f1fb70eed41af957414c946bb0.jpg)  
Figure 2: Overview of FAPR. The frozen base model provides the initial mask $M _ { 0 }$ and a shared multi-scale feature pyramid. Three failure-specific experts sequentially propose complete candidate masks $Q _ { k }$ , and the corresponding gates selectively overlay each candidate on the latest state to produce $M _ { 1 } , M _ { 2 } ,$ and $M _ { 3 }$ . A lightweight risk-aware controller finally accepts ${ \dot { M _ { 3 } } }$ or rolls the prediction back to $M _ { 0 }$

## Backbone-Agnostic Frozen Base Segmentor

FAPR is designed as a post-hoc repair framework rather than an architecture-specific extension. It can be attached to diferent pretrained segmentation models without modifying their original prediction pathways. Given an input image $\check { I } \in \mathbb { R } ^ { 3 \times H \times \mathbf { \check { W } } }$ , a base segmentor $f _ { \theta }$ produces the initial segmentation logits $L _ { 0 }$ and a hierarchy of intermediate visual features:

$$
( L _ { 0 } , \mathcal { F } ) = f _ { \boldsymbol { \theta } } ( I ) , \qquad M _ { 0 } = \sigma ( L _ { 0 } ) ,\tag{1}
$$

where $M _ { 0 } \in [ 0 , 1 ] ^ { H \times W }$ is the initial probability mask and

$$
\mathcal { F } = \{ B , T _ { L } , \dots , T _ { 2 } , T _ { 1 } \}\tag{2}
$$

denotes the feature hierarchy. Here, B represents the bottleneck feature and $T _ { l }$ represents a feature map at the l-th spatial resolution. Once Stage 1 is completed, all parameters θ are frozen, and no gradient from the repair objective is propagated into the base segmentor:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { r e p a i r } } = 0 .\tag{3}
$$

The only interface required by FAPR is therefore the initial mask $M _ { 0 }$ and a set of hierarchical image features F. For encoder-decoder architectures such as U-Net and U-KAN, these features can be directly obtained from their corresponding resolution levels. The repair formulation does not depend on the internal operations used to construct them. Consequently, replacing the base segmentor only changes the feature dimensions exposed through ${ \mathcal F } .$ , while the subsequent failure diagnosis, sequential repair, and state-update process remain unchanged.

In our U-KAN implementation with a $2 5 6 \times 2 5 6$ input, the extracted hierarchy is

$$
\begin{array} { r l } & { T _ { 1 } \in \mathbb R ^ { 1 6 \times 1 2 8 \times 1 2 8 } , \quad T _ { 2 } \in \mathbb R ^ { 3 2 \times 6 4 \times 6 4 } , } \\ & { T _ { 3 } \in \mathbb R ^ { 1 2 8 \times 3 2 \times 3 2 } , \quad T _ { 4 } \in \mathbb R ^ { 1 6 0 \times 1 6 \times 1 6 } , } \\ & { B \in \mathbb R ^ { 2 5 6 \times 8 \times 8 } . } \end{array}\tag{4}
$$

The frozen outputs $( M _ { 0 } , { \mathcal { F } } )$ initialize the repair pathway. At repair stage $k ,$ the corresponding expert receives the latest prediction state $M _ { k - 1 }$ together with the shared feature hierarchy ${ \mathcal F } ,$ rather than repeatedly invoking or updating the base model.

## Failure-State Context

Each specialist receives the frozen feature pyramid and a context representation derived from the current mask. For a probability map M, we construct

$$
C ( M ) = \operatorname { C o n c a t } ( M , B ( M ) , \mathcal { U } ( M ) , A ( M ) ) ,\tag{5}
$$

where B is a morphological boundary band, $\begin{array} { r } { \mathcal { U } ( M ) = 1 - } \end{array}$ $2 | M - 0 . 5 |$ is pixel uncertainty, and $\mathcal A ( M )$ broadcasts the spatial mean of M as an area map. The context makes the repair conditional on the current prediction geometry rather than only on image features.

## Sequential Repair Specialists

At stage k, specialist E<sub>k</sub> predicts a complete candidate mask

$$
Q _ { k } = \sigma \left( \mathcal { E } _ { k } ( \mathcal { F } , C ( M _ { k - 1 } ) ) \right) .\tag{6}
$$

All specialists use a five-level decoder with bilinear upsampling, concatenation with the corresponding frozen skip feature, and two Conv-GroupNorm-GELU blocks per level. This preserves the complete candidate-mask capacity that is necessary for large corrections while assigning each specialist a distinct failure objective.

Strict-miss routing. Before repair, a separately trained and frozen strict-miss router determines whether $\dot { M _ { 0 } }$ lacks reliable lesion support. It jointly analyzes the input image, the base mask, and explicit mask–image statistics ϕ(I, M<sub>0</sub>):

$$
s _ { 1 } = \sigma ( \mathcal { R } _ { \mathrm { m i s s } } ( I , M _ { 0 } , \phi ( I , M _ { 0 } ) ) ) , \qquad S _ { 1 } = \mathbb { I } [ s _ { 1 } \geq \tau _ { \mathrm { m i s s } } ] ,\tag{7}
$$

where $S _ { 1 } ~ \in ~ \{ 0 , 1 \}$ is a hard sample-level decision. Thus, ordinary predictions bypass global relocalization, while only detected gross misses activate the first repair stage.

Tail-sample routing. Unlike gross misses, undersegmentation and over-segmentation usually exhibit gradual rather than binary failure patterns. We therefore employ a shared tail-sample router with stage-specific outputs. Each output is recomputed from the frozen bottleneck feature $B$ and the latest prediction state:

$$
\begin{array} { r l r } & { } & { \mathbf { u } _ { k - 1 } = \left[ \mathrm { G A P } ( B ) , \mathrm { G M P } ( B ) , \psi ( M _ { k - 1 } ) \right] , } \\ & { } & { S _ { k } = \sigma ( \mathcal { R } _ { \mathrm { t a i l } , k } ( \mathbf { u } _ { k - 1 } ) ) , \qquad k \in \{ 2 , 3 \} . } \end{array}\tag{8}
$$

Here, GAP and GMP denote global average and max pooling, respectively, and $\psi ( M _ { k - 1 } )$ contains confidence, foreground-area, and uncertainty statistics of the current mask. Thus, $S _ { 2 } = \mathcal { R } _ { \mathrm { t a i l } , 2 } ( B , M _ { 1 } )$ diagnoses the state after gross-miss repair, whereas $S _ { 3 } = { \mathcal { R } } _ { \operatorname { t a i l } , 3 } ( B , M _ { 2 } )$ diagnoses the state after FN expansion. The soft scores $S _ { 2 } , \bar { S } _ { 3 } \in \bar { ( 0 , 1 ) }$ control the sample-level activation strengths of the corresponding stages, while their pixel-level gates determine the spatial extent of each repair.

Gross-miss repair stage. When $S _ { 1 } = 1$ , the gross-miss expert reconstructs a plausible lesion candidate from the frozen multi-scale features and the current mask context:

$$
Q _ { 1 } = \sigma ( \mathcal { D } _ { \mathrm { m i s s } } ( \mathcal { F } ) + \mathcal { H } _ { \mathrm { m i s s } } ( C ( M _ { 0 } ) ) ) ,\tag{9}
$$

where $\mathcal { D } _ { \mathrm { m i s s } }$ reconstructs global lesion structure and $\mathcal { H } _ { \mathrm { m i s s } }$ encodes the geometry and uncertainty of $M _ { 0 }$ . A pixel gate $P _ { 1 }$ selects the spatial support of the candidate, and the combined gate $G _ { 1 } = S _ { 1 } \odot P _ { 1 }$ updates the state as

$$
M _ { 1 } = \left( 1 - G _ { 1 } \right) \odot M _ { 0 } + G _ { 1 } \odot Q _ { 1 } .\tag{10}
$$

The resulting $M _ { 1 }$ is then passed to the subsequent refinement stages.

False-negative expansion stage. The second expert operates on the updated state $M _ { 1 }$ and recovers lesion regions that remain under-segmented after gross-miss repair. Rather than estimating a correction from the obsolete base mask, it predicts a complete candidate $Q _ { 2 }$ from the shared frozen features and the context of $M _ { 1 }$ :

$$
Q _ { 2 } = \sigma ( { \mathcal { D } } _ { \mathrm { f n } } ( { \mathcal { F } } ) + { \mathcal { H } } _ { \mathrm { f n } } ( C ( M _ { 1 } ) ) ) .\tag{11}
$$

The pixel gate evaluates the proposed change against the latest state:

$$
P _ { 2 } = \sigma ( \mathcal { G } _ { \mathrm { f n } } \left( [ M _ { 1 } , Q _ { 2 } , Q _ { 2 } - M _ { 1 } , ( 1 - M _ { 1 } ) \odot Q _ { 2 } ] \right) ) .\tag{12}
$$

It is combined with the tail-sample score $S _ { 2 }$ to obtain $G _ { 2 } =$ $S _ { 2 } \odot P _ { 2 }$ . The next prediction state is then

$$
M _ { 2 } = \left( 1 - G _ { 2 } \right) \odot M _ { 1 } + G _ { 2 } \odot Q _ { 2 } .\tag{13}
$$

Consequently, expansion is activated only for samples exhibiting an under-segmentation tendency and only at spatial locations supported by the current candidate. The updated mask $M _ { 2 }$ is passed to the false-positive suppression stage.

False-positive suppression stage. The final expert removes unsupported responses from the latest state $M _ { 2 } .$ , including over-segmented regions and isolated false-positive components introduced by either the base model or preceding repairs. It predicts a complete suppression candidate from the shared frozen features and the context of $M _ { 2 }$

$$
Q _ { 3 } = \sigma ( { \mathcal D } _ { \mathrm { f p } } ( { \mathcal F } ) + \mathcal { H } _ { \mathrm { f p } } ( C ( M _ { 2 } ) ) ) .\tag{14}
$$

The corresponding pixel gate compares the candidate with the current prediction state:

$$
P _ { 3 } = \sigma ( \mathcal { G } _ { \mathrm { f p } } \left( [ M _ { 2 } , Q _ { 3 } , Q _ { 3 } - M _ { 2 } , M _ { 2 } \odot ( 1 - Q _ { 3 } ) ] \right) )\tag{15}
$$

Here, $M _ { 2 } \odot ( 1 - Q _ { 3 } )$ explicitly represents the foreground region in the current state that the candidate proposes to remove. Combining the pixel gate with the tail-sample score gives $G _ { 3 } = S _ { 3 } \odot P _ { 3 }$ , and the final repaired state is

$$
M _ { 3 } = \left( 1 - G _ { 3 } \right) \odot M _ { 2 } + G _ { 3 } \odot Q _ { 3 } .\tag{16}
$$

This final transition suppresses unsupported foreground responses only where both sample-level diagnosis and pixellevel evidence permit modification, while retaining the preceding state elsewhere.

## Risk-Aware Rollback

Sequential repair may occasionally perturb an already reliable base mask. We therefore introduce a lightweight controller that decides whether to accept $M _ { 3 }$ or restore $M _ { 0 }$ . It uses the base-mask state and the accumulated repair change:

$$
\mathbf { z } = [ \phi ( M _ { 0 } ) , \Delta ( M _ { 0 } , M _ { 3 } ) ] , \qquad p _ { \mathrm { h a r m } } = \mathcal { R } _ { \eta } ( \mathbf { z } ) ,\tag{17}
$$

where $\phi ( M _ { 0 } )$ contains four base-state descriptors: foreground area ratio, vertical centroid, largest-component ratio, and mean predictive entropy. The change descriptor $\Delta ( M _ { 0 } , M _ { 3 } )$ contains the add–remove balance, area-ratio change, connected-component change, boundary-density change, scale-normalized centroid shift, and low overlap. The controller is supervised by

$$
y ^ { \mathrm { h a r m } } = \mathbb { I } [ \mathrm { D S C } ( M _ { 3 } , Y ) < \mathrm { D S C } ( M _ { 0 } , Y ) ] ,\tag{18}
$$

with sample weight $\begin{array}{c} \begin{array} { r l r } { w } & { { } = } & { \operatorname* { m a x } ( | \operatorname { D S C } ( M _ { 3 } , Y ) } \end{array} -  \end{array}$ $\mathrm { D S C } ( M _ { 0 } , \overset { \bullet } { Y } ) | , 1 0 ^ { - 3 } )$ . Candidate lightweight classifiers are fitted only on deduplicated training predictions. The classifier family and decision threshold are selected only on the validation set by maximizing rollback DSC, subject to a maximum rollback rate of 5%. Both are then frozen before test evaluation. At inference,

$$
r = \mathbb { I } [ p _ { \mathrm { h a r m } } \geq \tau ] , \qquad M _ { \mathrm { o u t } } = ( 1 - r ) M _ { 3 } + r M _ { 0 } .\tag{19}
$$

The controller uses no ground-truth information at inference. It makes one final decision after the complete sequential repair and never rolls back intermediate states.

## Failure Replay and Controlled Mask Corruption

Severe segmentation failures are sparse in the original training distribution. We therefore construct a training-only failure replay set from predictions of the frozen Stage-1 model. For each training pair $( I _ { i } , Y _ { i } )$ , the base segmentor produces $M _ { i } ^ { 0 }$ . Let

$$
d _ { i } ^ { 0 } = \mathrm { D S C } ( M _ { i } ^ { 0 } , Y _ { i } ) .\tag{20}
$$

The hard subset is defined as

$$
\mathcal { H } = \{ ( I _ { i } , Y _ { i } ) \in \mathcal { D } _ { \operatorname { t r a i n } } : d _ { i } ^ { 0 } < 0 . 7 \} .\tag{21}
$$

Each selected example is replayed once together with the original training set:

$$
\mathcal { D } _ { \mathrm { r e p l a y } } = \mathcal { D } _ { \mathrm { t r a i n } } \cup \mathcal { H } .\tag{22}
$$

The discrepancy between $M _ { i } ^ { 0 }$ and $Y _ { i }$ supplies false-negative and false-positive supervision for the routers and pixel gates. Because severe failures are sparse, the Stage-1 prior is replaced with probability $p _ { c } = 0 . 5 5$ by a controlled groundtruth corruption sampled from four modes: gross miss, undersegmentation, over-segmentation, or boundary jitter. These modes are implemented using erosion or cutout, dilation or disconnected blobs, and sparse boundary perturbations. Otherwise, the real Stage-1 prediction is retained. Replay construction uses training annotations only; validation and test samples are excluded, and inference always starts from the frozen segmentor’s prediction.

## Training Objectives

FAPR jointly supervises the candidate masks, sequential states, and routing modules. Let Y denote the ground-truth mask. Each specialist is optimized using region-weighted binary cross-entropy and Tversky loss:

$$
\mathcal { L } _ { \mathrm { e x p } } = \sum _ { k = 1 } ^ { 3 } \left[ \mathcal { L } _ { \mathrm { W B C E } } ( Q _ { k } , Y ; W _ { k } ) + \mathcal { L } _ { \mathrm { T v } } ( Q _ { k } , Y ; \alpha _ { k } , \beta _ { k } ) \right] .\tag{23}
$$

Here, $W _ { k }$ emphasizes the failure region assigned to expert k. The gross-miss and FN experts use recall-oriented supervision, whereas the FP expert uses precision-oriented supervision. All intermediate states $M _ { k }$ and the final output $M _ { 3 }$ are additionally supervised by BCE and Dice losses.

The pixel gates are trained against their corresponding gross-miss, FN, and FP regions. Sample routers use classbalanced binary cross-entropy with failure labels derived from Stage-1 training predictions. A preservation term penalizes changes to correctly predicted regions, while sparsity regularization discourages unnecessary repair. The complete objective is

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { e } \mathcal { L } _ { \mathrm { e x p } } + \lambda _ { s } \displaystyle \sum _ { k = 1 } ^ { 3 } \mathcal { L } _ { \mathrm { s e g } } ( M _ { k } , Y ) + \lambda _ { f } \mathcal { L } _ { \mathrm { s e g } } ( M _ { 3 } , Y ) } \\ { + \lambda _ { g } \mathcal { L } _ { \mathrm { g a t e } } + \lambda _ { r } \mathcal { L } _ { \mathrm { r o u t e r } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r e s } } + \lambda _ { \mathrm { s p } } \mathcal { L } _ { \mathrm { s p a r s e } } , } \end{array}\tag{24}
$$

where $\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { D i c e } }$ . The strict-miss router is pretrained and frozen before specialist training. The risk-aware rollback controller is calibrated separately after sequential repair training and does not propagate gradients to the base segmentor or repair modules.

## Experiments

## Datasets

BUSI. BUSI (Al-Dhabyani et al. 2020) contains breast ultrasound images with benign, malignant, and normal cases. Following the fixed lesion-only split used by our baselines, we use 453 training, 65 validation, and 129 test images.

TN3K. TN3K (Gong et al. 2021) contains 3,493 thyroid nodule ultrasound images. We follow its oficial test split, using 2,519 training, 360 validation, and 614 test images.

BUSIS. BUSIS (Zhang et al. 2022) is a breast ultrasound segmentation benchmark containing 562 de-identified images collected from multiple hospitals. We use 394 images for training, 56 for validation, and 112 for testing.

## Baselines

We compare FAPR with representative segmentation networks and post-hoc refiners. U-Net (Ronneberger, Fischer, and Brox 2015), Attention U-Net (Oktay et al. 2018), CMUNet (Tang et al. 2023), MSAGHNet (Zhu et al. 2025), and U-KAN (Li et al. 2025) cover convolutional, attentionbased, and KAN-based segmentors, while SegRefiner (Wang et al. 2023) represents generic mask refinement. CausalBridgeNet (Yang et al. 2026) is the closest recent correction baseline: its PCRU processes the same prompt-conditioned decoder feature through four parallel causal experts (Jacobs et al. 1991; Riquelme et al. 2021; Wang et al. 2025) and mixes their outputs to estimate FP and FN maps for a oneshot correction of the initial logits. In contrast, FAPR decomposes the correction into three ordered failure-specific stages, each conditioned on the latest accepted mask. Our matched CausalBridgeNet reproduction uses the same frozen U-KAN, data splits, input resolution, and validation-based checkpoint selection as FAPR.

## Evaluation Metrics

We report DSC, mIoU, Precision, and Recall, computed per image and averaged across the evaluation set. Checkpoints are selected by validation DSC and evaluated once on the held-out test set.

## Implementation Details

FAPR is implemented in PyTorch and trained on two NVIDIA RTX PRO 5000 Blackwell GPUs with a global batch size of 16. Table 1 reports averages over seeds 2981, 3407, and 42. The Stage-1 segmentor and pretrained strictmiss router remain frozen. Repair modules are trained for 80 epochs using Adam with an initial learning rate of $8 \times 1 0 ^ { - 5 }$ weight decay of $1 0 ^ { - 4 }$ , cosine decay to $1 \bar { 0 } ^ { - 6 }$ , and gradient clipping at 5. Checkpoints are selected by validation DSC. Replay samples and router labels use training data only, while the final rollback controller is calibrated on the training and validation splits.

<table><tr><td></td><td></td><td colspan="4">BUSI</td><td colspan="4">BUSIS</td><td colspan="4">TN3K</td></tr><tr><td>Method</td><td>Params (M)</td><td>DSC</td><td>mIoU</td><td>Prec.</td><td>Rec.</td><td>DSC</td><td>mIoU</td><td>Prec.</td><td>Rec.</td><td>DSC</td><td>mIoU</td><td>Prec.</td><td>Rec.</td></tr><tr><td>U-Net (Ronneberger et al., 2015)</td><td>31.04</td><td>80.91</td><td>72.16</td><td>81.84</td><td>84.77</td><td>91.18</td><td>84.89</td><td>93.02</td><td>90.88</td><td>80.48</td><td>71.04</td><td>82.26</td><td>84.53</td></tr><tr><td>AttUNet (Oktay et al., 2018)</td><td>34.88</td><td>76.62</td><td>68.09</td><td>79.72</td><td>78.44</td><td>91.04</td><td>84.65</td><td>93.04</td><td>90.66</td><td>77.80</td><td>67.86</td><td>74.94</td><td>87.04</td></tr><tr><td>SegRefiner (Wang et al., 2023)</td><td>124.96</td><td>81.34</td><td>72.46</td><td>83.35</td><td>83.71</td><td>92.63</td><td>86.80</td><td>93.33</td><td>92.76</td><td>81.91</td><td>73.23</td><td>85.87</td><td>84.38</td></tr><tr><td>CMUNet (Tang et al., 2023)</td><td>49.93</td><td>79.95</td><td>71.68</td><td>84.13</td><td>81.45</td><td>91.43</td><td>85.28</td><td>92.86</td><td>91.58</td><td>79.86</td><td>70.15</td><td>78.35</td><td>85.19</td></tr><tr><td>MSAGHNet (Zhu et al., 2025)</td><td>26.61</td><td>81.44</td><td>72.53</td><td>84.04</td><td>81.78</td><td>91.36</td><td>84.76</td><td>92.88</td><td>90.94</td><td>81.01</td><td>71.35</td><td>79.83</td><td>87.55</td></tr><tr><td>U-KAN (Li et al., 2025)</td><td>6.36</td><td>81.24</td><td>72.29</td><td>83.50</td><td>83.35</td><td>92.50</td><td>86.64</td><td>93.25</td><td>92.65</td><td>81.83</td><td>73.10</td><td>85.37</td><td>84.48</td></tr><tr><td>CausalBridgeNet (Yang et al., 2026)</td><td>18.76</td><td>81.77</td><td>73.03</td><td>84.06</td><td>83.44</td><td>92.53</td><td>86.68</td><td>93.14</td><td>92.79</td><td>82.12</td><td>73.47</td><td>85.19</td><td>85.25</td></tr><tr><td>FAPR (Ours)</td><td>17.37</td><td> $\mathbf { 8 3 . 7 8 } _ { \pm 0 . 2 }$ </td><td> $7 5 . 3 4 _ { \pm 0 . 3 }$ </td><td> $\mathbf { 8 6 . 0 8 } _ { \pm 0 . 2 }$ </td><td> $\mathbf { 8 4 . 5 1 } _ { \pm 0 . 2 }$ </td><td> $9 2 . 6 2 _ { \pm 0 . 1 }$ </td><td>86.79±0.2</td><td> $9 3 . 6 6 _ { \pm 0 . 1 }$ </td><td>92.41±0.2</td><td> $\mathbf { 8 3 . 7 2 } _ { \pm 0 . 1 }$ </td><td> $7 5 . 1 5 _ { \pm 0 . 1 }$ </td><td> $\mathbf { 8 5 . 2 9 } _ { \pm 0 . 1 }$ </td><td>86.67</td></tr></table>

Table 1: Comparison on BUSI, BUSIS, and TN3K. Values are percentages and Params denotes the total number of parameter in millions. The FAPR results are averaged over three random seeds; tiny subscripts denote standard deviations.

<table><tr><td>Dataset</td><td>Method</td><td>Very hard</td><td>Hard</td><td>Easy</td></tr><tr><td rowspan="4">BUSI</td><td>Stage 1</td><td>17.04</td><td>42.62</td><td>89.18</td></tr><tr><td>SegRefiner</td><td>17.22</td><td>42.54</td><td>89.32</td></tr><tr><td>CausalBridgeNet</td><td>17.20</td><td>43.80</td><td>89.58</td></tr><tr><td>FAPR</td><td>35.78 ±0.4</td><td>53.69 ±1.0</td><td>90.11 ±0.0</td></tr><tr><td rowspan="4">TN3K</td><td>Stage 1</td><td>25.97</td><td>42.85</td><td>89.60</td></tr><tr><td>SegRefiner</td><td>25.93</td><td>42.82</td><td>89.68</td></tr><tr><td>CausalBridgeNet</td><td>26.99</td><td>43.69</td><td>89.78</td></tr><tr><td>FAPR</td><td>34.76 ±2.7</td><td>49.39 ±0.8</td><td>89.25</td></tr></table>

Table 2: Failure-conditioned DSC. Columns denote veryhard (< 0.5), hard $( < 0 . 7 ) .$ , and easy $( \geq 0 . 7 )$ subsets. Their sizes are BUSI: 9/22/107 and TN3K: 53/102/512.

Under single-image FP32 inference at 256 × 256 resolution, FAPR uses 7.4% fewer parameters than the matched CausalBridgeNet reproduction and reduces Conv/Linear FLOPs, latency, and peak memory by 81.3%, 59.3%, and 73.9%, respectively.

Additional implementation details, code, training scripts, and data splits are provided in the supplementary material. Code and trained weights will be released upon acceptance.

## Efectiveness of FAPR

Tables 1 and 2 report aggregate and failure-conditioned performance on BUSI and TN3K. Hard and very-hard cases satisfy $\mathrm { D S C } ( M _ { 0 } , Y ) < 0 . 7$ and < 0.5, respectively; these labels are used only for analysis.

Using the same frozen U-KAN, FAPR outperforms the matched static correctors on both benchmarks (Table 1). Its advantage is concentrated in the low-accuracy tail: very-hard DSC increases by 18.74 points on BUSI and 8.79 points on TN3K, whereas easy-case DSC changes by only +0.93 and −0.35 points, respectively (Table 2). This separation supports progressive repair of structural failures rather than uniform smoothing of all predictions.

<table><tr><td>Configuration</td><td>Overall</td><td>Easy</td><td>Hard</td><td>Very hard</td></tr><tr><td>FAPR</td><td>83.78 ±0.2</td><td></td><td> $\mathbf { 9 0 . 1 1 } _ { = 0 . 0 } 5 3 . 6 9 _ { + 1 }$  ..0</td><td></td></tr><tr><td>w/o Gross-miss</td><td>83.28</td><td>90.03</td><td>50.49</td><td> $^ { 3 5 . 7 8 } _ { 3 0 . 4 0 }$ </td></tr><tr><td>w/o FN Expansion</td><td>82.48</td><td>89.77</td><td>47.00</td><td>21.58</td></tr><tr><td>w/o FP Suppression</td><td>82.31</td><td>89.84</td><td>45.67</td><td>21.66</td></tr><tr><td>w/o Failure Replay</td><td>82.38</td><td>89.90</td><td>45.83</td><td>18.95</td></tr><tr><td>w/o Controlled Corruption</td><td>82.40</td><td>89.89</td><td>45.97</td><td>19.51</td></tr></table>

Table 3: Specialist and training ablations on the BUSI test set. The complete FAPR row includes the final rollback controller.

## Robustness of FAPR

BUSIS is nearly saturated and contains only one hard test case. FAPR reaches 92.62 DSC and 86.79 mIoU, within 0.01 points of SegRefiner and slightly above the frozen Stage-1 model. Thus, selective routing preserves performance when little repair is needed.

## Roles of the Sequential Specialists

Table 3 isolates the contribution of the repair components. Removing individual specialists lowers very-hard DSC by 5.38–14.20 points, while removing replay or controlled corruption lowers it by 16.27–16.83 points. Easy-case DSC remains within 89.77–90.11, showing that these components primarily repair dificult failures.

Sparse gross-miss routing. The strict-miss router activates for only 3/129 BUSI test cases, yet all three improve after the first repair stage, with a mean $M _ { 0 } \to M _ { 1 }$ gain of 9.75 DSC points. Removing the gross-miss specialist further reduces very-hard DSC by 5.38 points. Thus, the miss route is rarely used but remains important for recovering extreme localization failures.

Low-cost rollback. The final controller uses only ten maskchange descriptors and requires neither additional image encoding nor another segmentation forward pass. Despite its negligible computational cost, it improves DSC by 0.13 and

![](images/cff425e59a9866c804f32233bb9261fc7b93e381f16f3d0c4d94e2b15e3dcc81.jpg)  
Figure 3: Qualitative prediction-state progression on two BUSI test cases. True-positive, false-positive, and false-negative regions are shown in green, red, and blue, respectively. (a) The strict-miss route is activated. (b) The miss route is bypassed, so $M _ { 1 } = M _ { 0 }$ , while the FN-expansion and FP-suppression specialists perform the subsequent repair.

0.004 percentage points on BUSI and TN3K, respectively, providing a conservative safety gain after sequential repair. On the nearly saturated BUSIS benchmark, where hard failures are almost absent, no prediction is rolled back.
<table><tr><td>Configuration</td><td>Total Train. DSC mIoU Hard Very hard (M) (M)</td><td></td><td>DSC</td><td>DSC</td></tr><tr><td>U-KAN-Large (GB16) 25.36 25.36 78.76 68.8547.36</td><td></td><td></td><td></td><td>28.96</td></tr><tr><td>Shared Refiner ×3</td><td></td><td>17.3210.7082.6974.1748.01</td><td></td><td>27.77</td></tr><tr><td>Parallel Expert Fusion</td><td>17.6110.9982.99 74.4148.84</td><td></td><td></td><td>29.21</td></tr><tr><td>Static-M0Cascade</td><td></td><td>17.3710.7582.9874.4550.88</td><td></td><td>33.61</td></tr><tr><td>Reverse Repair Order</td><td></td><td>17.3710.75 82.55 73.92 46.02</td><td></td><td>21.83</td></tr><tr><td>Uniform BCE+Dice</td><td></td><td>17.3710.7582.02 73.3943.90</td><td></td><td>17.49</td></tr><tr><td>FAPR</td><td></td><td>17.3710.7583.7875.34 53.69</td><td></td><td>35.78</td></tr></table>

Table 4: Mechanism ablation study on the BUSI test set.

Table 4 separates model capacity from the proposed repair mechanism. A larger backbone, a repeated shared refiner, and parallel expert fusion all underperform FAPR, showing that the gains do not arise from additional capacity alone. Conditioning every specialist on the static $M _ { 0 } ^ { \dot { } }$ weakens hard-case repair, confirming the importance of evolving-state inputs. Reversing the repair order or replacing failure-specific objectives with uniform supervision causes a larger degradation, demonstrating that the specialists are order-sensitive and non-exchangeable. Together, these controls validate the coordinated roles of failure specialization, progressive state updating, and ordered execution.

Visualization. To better illustrate the repair workflow of FAPR, in Figure 3, the first case activates gross-miss recovery before expansion and suppression, while the second bypasses it and is repaired by the later specialists. These trajectories illustrate that FAPR follows sample-dependent repair paths rather than applying every correction indiscriminately.

<table><tr><td>Dataset Method</td><td></td><td>DSC mIoU Hard</td><td>Very hard</td></tr><tr><td rowspan="2">BUSI</td><td>U-Net Stage1</td><td>80.91 72.16 38.36</td><td>16.38</td></tr><tr><td>U-Net + FAPR</td><td>82.86 74.71 46.47</td><td>30.37</td></tr><tr><td rowspan="2">TN3K</td><td>U-Net Stage1</td><td>80.48 71.04 42.38</td><td>21.83</td></tr><tr><td>U-Net + FAPR</td><td>82.55 73.22 52.19</td><td>37.83</td></tr></table>

Table 5: Backbone-transfer results with a frozen U-Net Stage-1 model.

Backbone transfer. With a frozen U-Net, FAPR improves overall DSC by 1.95 points on BUSI and 2.07 points on TN3K. The gains increase to 8.11/13.99 points on BUSI and 9.81/16.00 points on the hard/very-hard subsets, confirming that the repair mechanism transfers beyond U-KAN and remains focused on the low-accuracy tail.

## Conclusion

We presented FAPR, a post-hoc framework that models ultrasound segmentation correction as progressive predictionstate transitions. Ordered specialists recover gross misses, complete FN regions, and suppress FP regions, while selective routing limits harmful updates. Experiments on three datasets and two frozen backbones show consistent gains concentrated on dificult cases without sacrificing saturated benchmarks. FAPR thus improves dificult predictions without retraining the base segmentor. These findings suggest that residual segmentation failures are better modeled as coupled, evolving states than as independent corrections derived from a fixed prediction. Future work will extend this formulation to stronger volumetric repair for broader medical imaging applications by explicitly modeling 3D spatial continuity and inter-slice dependencies.

## References

Al-Dhabyani, W.; Gomaa, M.; Khaled, H.; and Fahmy, A. 2020. Dataset of breast ultrasound images. Data in brief, 28: 104863.

Carreira, J.; Agrawal, P.; Fragkiadaki, K.; and Malik, J. 2016. Human pose estimation with iterative error feedback. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4733–4742.

Chen, J.; Lu, Y.; Yu, Q.; Luo, X.; Adeli, E.; Wang, Y.; Lu, L.; Yuille, A. L.; and Zhou, Y. 2021. Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.

Chen, T.; Wang, C.; Chen, Z.; Lei, Y.; and Shan, H. 2024. HiDif: Hybrid difusion framework for medical image segmentation. IEEE Transactions on Medical Imaging, 43(10): 3570–3583.

Gong, H.; Chen, G.; Wang, R.; Xie, X.; Mao, M.; Yu, Y.; Chen, F.; and Li, G. 2021. Multi-Task Learning for Thyroid Nodule Segmentation with Thyroid Region Prior. In IEEE International Symposium on Biomedical Imaging, 257–261.

Hatamizadeh, A.; Nath, V.; Tang, Y.; Yang, D.; Roth, H. R.; and Xu, D. 2021. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In International MICCAI brainlesion workshop, 272–284. Springer.

Hatamizadeh, A.; Tang, Y.; Nath, V.; Yang, D.; Myronenko, A.; Landman, B.; Roth, H. R.; and Xu, D. 2022. Unetr: Transformers for 3d medical image segmentation. In Proceedings of the IEEE/CVF winter conference on applications ofcomputer vision, 574–584.

Jacobs, R. A.; Jordan, M. I.; Nowlan, S. J.; and Hinton, G. E. 1991. Adaptive mixtures of local experts. Neural computation, 3(1): 79–87.

Li, C.; Liu, X.; Li, W.; Wang, C.; Liu, H.; Liu, Y.; Chen, Z.; and Yuan, Y. 2025. U-kan makes strong backbone for medical image segmentation and generation. In Proceedings ofthe AAAI conference on artificial intelligence, volume 39, 4652–4660.

Li, C.; Mao, Y.; Liang, S.; Li, J.; Wang, Y.; and Guo, Y. 2024. Deep causal learning for pancreatic cancer segmentation in CT sequences. Neural Networks, 175: 106294.

Lu, Y.; Wang, Z.; Xu, W.; Zhang, W.; Zhao, Y.; Yu, Y.; and Zhang, X. 2025. Layer-wise Noise Guided Selective Wavelet Reconstruction for Robust Medical Image Segmentation. arXiv preprint arXiv:2511.16162.

Luo, Z.; Zhu, X.; Zhang, L.; and Sun, B. 2025. Rethinking unet: Task-adaptive mixture of skip connections for enhanced medical image segmentation. In Proceedings of the AAAI Conference onArtificial Intelligence, volume 39, 5874–5882.

Miao, J.; Chen, C.; Liu, F.; Wei, H.; and Heng, P.-A. 2023. Caussl: Causality-inspired semi-supervised learning for medical image segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, 21426–21437.

Oktay, O.; Schlemper, J.; Folgoc, L. L.; Lee, M.; Heinrich, M.; Misawa, K.; Mori, K.; McDonagh, S.; Hammerla, N. Y.; Kainz, B.; et al. 2018. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.

Riquelme, C.; Puigcerver, J.; Mustafa, B.; Neumann, M.; Jenatton, R.; Susano Pinto, A.; Keysers, D.; and Houlsby, N. 2021. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34: 8583– 8595.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, 234–241. Springer.

Tang, F.; Wang, L.; Ning, C.; Xian, M.; and Ding, J. 2023. CMU-Net: A Strong ConvMixer-Based Medical Ultrasound Image Segmentation Network. In IEEE International Symposium on Biomedical Imaging, 1–5.

Tang, H.; Liu, X.; Sun, S.; Yan, X.; and Xie, X. 2021. Recurrent Mask Refinement for Few-Shot Medical Image Segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3918–3928.

Wang, M.; Ding, H.; Liew, J. H.; Liu, J.; Zhao, Y.; and Wei, Y. 2023. Segrefiner: Towards model-agnostic segmentation refinement with discrete difusion process. arXiv preprint arXiv:2312.12425.

Wang, Z.; Zhang, X.; Li, Z. S.; and Yan, M. 2025. A Region-Aware Dual Latent State Mining Framework for Service Recommendation in Large-Scale Service Networks. IEEE Transactions on Knowledge and Data Engineering.

Xu, W.; and Wang, Z. 2025. SCRNet: Spatial-Channel Regulation Network for Medical Ultrasound Image Segmentation. arXiv preprint arXiv:2508.13899.

Yang, H.; Chen, Y.; Ma, S.; and Guo, F. 2026. Make Foundation Models Trustworthy Again: Causal Fine-Adaptation for Medical Image Segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 27512– 27520.

Yuan, Y.; Xie, J.; Chen, X.; and Wang, J. 2020. Segfix: Model-agnostic boundary refinement for segmentation. In European conference on computer vision, 489–506. Springer.

Zhang, Y.; Xian, M.; Cheng, H.-D.; Shareef, B.; Ding, J.; Xu, F.; Huang, K.; Zhang, B.; Ning, C.; and Wang, Y. 2022. BU-SIS: a benchmark for breast ultrasound image segmentation. In Healthcare, volume 10, 729. MDPI.

Zhou, Z.; Rahman Siddiquee, M. M.; Tajbakhsh, N.; and Liang, J. 2018. Unet++: A nested u-net architecture for medical image segmentation. In International workshop on deep learning in medical image analysis, 3–11. Springer.

Zhu, S.; Li, Y.; Dai, X.; Mao, T.; Wei, L.; and Yan, Y. 2025. A multi-resolution hybrid cnn-transformer network with scale-guided attention for medical image segmentation. IEEE Journal of Biomedical and Health Informatics.