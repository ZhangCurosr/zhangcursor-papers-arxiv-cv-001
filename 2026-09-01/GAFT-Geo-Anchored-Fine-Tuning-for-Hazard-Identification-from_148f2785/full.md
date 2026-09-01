# GAFT: Geo-Anchored Fine-Tuning for Hazard Identification from Rare Failures

Yanran Xu<sup>B</sup>, Chuanhang Qiu, Yue Wang, Wenbo Wu, Zhaoxing Li

University of Southampton, Southampton, United Kingdom {Y.Xu, C.Qiu, Yue.Wang, Wenbo.Wu, Zhaoxing.Li}@soton.ac.uk

Abstract. Of-road navigation can fail when physical structures induce irrecoverable states such as high-centering or entrapment, requiring human interventions. Identifying these structures is crucial, yet challenging. Such failure events are rare and costly to collect, resulting in limited training data. Moreover, the collected data associate frames with outcomes, but do not indicate the visual cues responsible for the failure. Learning directly from these data can therefore exploit scenario-specific visual cues, leading to poor generalization. We propose Geo-Anchored Fine-Tuning (GAFT), a parameter-eficient method that adapts a vision foundation model with a geometry-derived prior. It guides LoRA adaptation by aligning a spatial attention-rollout map with the geometry prior, while preserving pretrained representations. On an interventionverified forest hazard benchmark, across ten independently trained adaptations, GAFT consistently outperforms frozen DINOv2 and supervised PEFT baselines, improving the repeated leave-one-scenario-out mean F<sub>2</sub> from 0.0607 to 0.3757 with statistical significance under paired analysis. Within these independently trained models, the best-performing GAFT model achieves a repeated-LOSO F of 0.570. Code and benchmark: https://github.com/Xu-Yanran/geo\_anchored\_fine\_tuning.

Keywords: Hazard Identification · Of-Road Navigation · Vision Foundation Models · Geometry-Guided Adaptation · Parameter-Eficient Fine-Tuning

## 1 Introduction

For autonomous ground robots operating in unstructured terrain, the presence of a physical structure alone does not determine whether it poses a hazard. The risk posed by a structure depends on its geometry relative to the robot and its path. A standing tree trunk may be visually prominent yet remain avoidable. Conversely, a low fallen log can high-center a rover that drives over it, causing loss of wheel–ground contact. We use hazards to denote structures that can drive a ground robot into such irrecoverable states, including high-centering, entrapment, and sensor damage [32, 29]. Reliable field navigation therefore requires reasoning about the physical consequences of local structures, beyond their object identity or visual saliency.

Learning this distinction is dificult because hazard data are scarce and provide only coarse, outcome-level supervision. Hazard events are costly to collect, often requiring human intervention after a failure, resulting in only a small number of labeled examples. Moreover, these data associate frames with failure outcomes but do not indicate the visual evidence responsible for the hazardous interaction. Consequently, models trained directly on such data, whether from scratch or by adapting a pretrained model using only hazard labels, may exploit scenario-specific cues, such as scene layout, texture, illumination, or obstacle appearance, rather than learning the robot–terrain relationship that makes a structure hazardous [10]. Few-shot and meta-learning methods reduce annotation requirements when suitable meta-training tasks are available [6, 25, 30]. Such tasks are dificult to construct for rare of-road failures spanning diverse robots, terrains, and obstacle configurations. Class-imbalance objectives address label imbalance [20, 23] but do not constrain the evidence learned under outcome-level supervision, leaving shortcut learning unresolved.

Vision foundation models (VFMs) provide a strong starting point because their pretrained features capture broad visual and semantic structure without learning a representation from scarce hazard data alone. Wild visual navigation uses pretrained DINO-ViT features to learn traversability from field experience [21], and self-supervised ViTs provide transferable patch representations [2, 22]. However, generic visual pretraining provides no preference for features that predict the physical consequence of a particular robot–terrain interaction. A VFM may encode trees, foliage, terrain texture, and scene layout well; when it is adapted only on scarce hazard data, a classifier or parameter-eficient adapter can therefore rely on whichever of these cues best predicts the observed scenarios, including scene-specific shortcuts that do not transfer. Data-scarce hazard identification consequently requires an additional inductive bias that preserves transferable visual features while favouring spatial cues relevant to traversal.

We propose Geo-Anchored Fine-Tuning (GAFT), a two-stage parametereficient method that adapts a vision foundation model for hazard identification from rare failures. GAFT uses an RGB–depth traversal dataset to construct a proximity-weighted geometric prior that emphasizes structures protruding above the ground, assigning greater weight to those closer to the robot. During adaptation, attention rollout is aligned with this prior while a representation anchor limits drift from the pretrained representation. The hazard classifier is then trained on scarce RGB frames whose labels are derived from the observed outcomes of robot-terrain interactions.

## Contributions:

1. We introduce a two-stage learning framework that uses an RGB–depth traversal dataset to guide vision foundation model adaptation before training a robot-specific hazard classifier from scarce robot–terrain interaction labels.

2. We propose GAFT, a parameter-eficient adaptation method that aligns spatial attention with a proximity-weighted geometric prior to guide an RGB vision foundation model while preserving pretrained representations.

3. Empirical evaluation on an intervention-verified benchmark shows that GAFT consistently outperforms frozen DINOv2 and supervised PEFT baselines, with statistical significance under paired analysis.

## 2 Related Work

## 2.1 Navigation in Unstructured Environments

Navigation in unstructured environments has often been approached by estimating traversability from the geometry of the surrounding terrain. Geometry-based methods capture slope, clearance, and other structural properties relevant to navigation [3, 7]. However, geometry is only a partial description of traversability: tall grass may be traversable despite its geometric extent, while visually incon spicuous structures near the ground can prevent a robot from moving forward. Vision-based methods therefore learn terrain properties from visual or semantic features [13, 11], rather than treating all geometrically salient structures as obstacles.

The relevance of a visual feature nevertheless depends on the robot and its interaction with the terrain. Self-supervised traversability methods use robot motion and proprioceptive measurements to generate supervision signals, allowing visual predictions to reflect platform-specific traction, mobility, or collision outcomes [27, 18, 9, 21]. Anomaly detection, proprioceptive feedback, and recovery policies further address situations in which a visual estimate is unreliable or an obstacle has been missed [28, 16, 8, 32, 29]. For hazards that can lead to irrecoverable states, however, the interaction used to generate a label may high-center or trap the robot and require manual recovery. Hazard events are therefore time-consuming to collect and remain scarce.

## 2.2 Learning from Sparse Failure Data

With only a small number of hazard events, a classifier can fit the observed scenarios without learning the physical structures associated with failure. A binary hazard label indicates association with a collision, entrapment, or high centering event, but it does not identify which image region contains the relevant structure. Direct training from scratch or fine-tuning with only these labels can therefore rely on texture, illumination, or background correlations that do not transfer to a new environment [10]. Reweighting objectives such as focal loss increase the contribution of rare positive examples [20], but do not identify which image regions contain interaction-relevant structure.

Few-shot and meta-learning methods improve adaptation when related tasks are available during meta-training [24, 30, 26]. Hazard identification does not satisfy this assumption. Whether a log, root, branch, or vegetation is hazardous depends on its pose, the terrain, and the platform-specific capability, rather than on a fixed semantic category. A meta-training collection spanning these cases would require broad coverage of the same hazardous interactions that are costly to collect.

## 2.3 Geometry-Guided Vision Foundation Model Adaptation

Pretrained vision foundation models are a promising starting point because they provide visual representations without requiring a task-specific model to be trained from scratch. Self-supervised ViTs provide transferable visual features [2, 22], and pretrained DINO-ViT features have been efective for wild visual navigation [21]. Combined with parameter-eficient adaptation methods such as LoRA, visual prompt tuning, and AdaptFormer [14, 17, 4], these models reduce the need to train a task-specific network from scratch and make adaptation more feasible when labeled data are limited. However, their pretraining is generic rather than navigation-specific. When adapted only with scarce failure labels, a VFM has no explicit preference for visual evidence that is physically relevant to robot–terrain interaction.

Geometry provides a natural way to introduce such a preference for ground robots. Depth information is widely used in of-road navigation because terrain shape, clearance, and above-ground structure are directly related to navigation [9, 28]. Existing work on geometry-guided VFM adaptation uses geometric information for diferent purposes. FiT3D uses multi-view geometric consistency to improve the 3D awareness of foundation model features [31], while ViTA combines semantic traversability supervision with geometric knowledge to adapt a VFM for traversability estimation [15]. For hazard identification from rare failures, geometric cues can complement PEFT by biasing adaptation toward physical structures that may afect robot motion rather than incidental appearance cues such as background texture, lighting, or scene layout. This motivates geometry-guided adaptation of vision foundation models for hazard identification from rare failures.

## 3 Methodology

## 3.1 Problem Formulation and Overview

GAFT uses two complementary data sources. The first is a traversal dataset without hazard annotations, $\mathcal { D } _ { u } = \{ ( x _ { i } , d _ { i } ) \} _ { i = 1 } ^ { N _ { u } }$ , containing temporally matched RGB frames $x _ { i }$ and depth maps $d _ { i }$ . The second is a much smaller hazard set $\mathcal { D } _ { h } = \{ ( x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { N _ { h } }$ , where $y _ { j } = 1$ denotes a hazard-associated frame, $y _ { j } = 0$ a normal traversal frame.

Stage 1 uses $\mathcal { D } _ { u }$ without hazard labels to construct a proximity-weighted geometric prior from depth relative to an estimated ground plane. This prior guides LoRA adaptation of DINOv2-ViT-S/14, while all original backbone parameters are frozen. Stage 2 freezes the adapted backbone and trains a linear hazard probe on rollout-weighted patch features from $\mathcal { D } _ { h }$

## 3.2 Proximity-Weighted Geometric Prior

For each depth map, valid pixels define a binary mask $W ^ { \mathrm { v a l i d } }$ . We back-project these pixels into 3D and estimate a ground plane with RANSAC. Let $\scriptstyle { \hat { e } } ( u , v )$ be

![](images/eabc5ef76863360773ff62c45367e4f8bd04733c5e00643cd2732bb890f8eff4.jpg)  
Fig. 1. GAFT overview. Stage 1 uses an RGB–depth traversal set without hazard annotations to align a spatial attention-rollout map with a proximity-weighted geometric prior while anchoring the adapted representation to frozen DINOv2. Stage 2 freezes the adapted backbone and fits a linear hazard probe on scarce labeled RGB frames. Deployment uses RGB only.

the signed height of pixel $( u , v )$ above this fitted ground plane, let $d ( u , v )$ be its metric depth, and let $d _ { \mathrm { m i n } }$ be the minimum valid depth in the frame. The geometric target is

$$
\begin{array} { r l } & { M ^ { * } ( u , v ) = W ^ { \mathrm { v a l i d } } ( u , v ) \exp \left[ - \lambda \big ( d ( u , v ) - d _ { \mathrm { m i n } } \big ) \right] } \\ & { \qquad \cdot \ : \mathbf { 1 } \{ \epsilon < \hat { e } ( u , v ) < e _ { \mathrm { m a x } } \} \ : \mathbf { 1 } \{ d ( u , v ) < d _ { \mathrm { m a x } } \} . } \end{array}\tag{1}
$$

The target encodes a proximity-weighted geometric prior: valid pixels within a selected height band above the ground plane receive greater weight when they are closer to the robot, whereas pixels outside this band and distant pixels are suppressed. $M ^ { * }$ is bounded in [0, 1] by the binary masks and the exponential distance decay. The prior supplies spatial supervision for VFM adaptation, while the hazard classifier is trained on labels derived from robot–terrain interaction outcomes.

## 3.3 Geo-Anchored VFM Adaptation

The visual backbone is DINOv2-ViT-S/14. Its pretrained parameters remain frozen; GAFT adds trainable LoRA adapters to the combined QKV projection and, for the QKV+MLP variant, to the feed-forward linear projections in every transformer block. We evaluate QKV-only and QKV+MLP as two adapter-scope configurations.

GAFT aligns the [CLS]-to-patch attention rollout [1] with the geometric prior. Attention rollout aggregates residual-adjusted attention across transformer layers and provides a diferentiable spatial map whose relative patch weights are matched to the geometric prior. For each block, attention weights $\mathbf { \mathcal { A } } _ { l }$ are averaged over heads, mixed with the identity residual connection, row-normalized, and multiplied in layer order:

$$
\begin{array} { l } { \displaystyle \mathbf { a } = \left( \prod _ { l } \mathrm { r o w n o r m } ( 0 . 5 \mathcal { A } _ { l } + 0 . 5 I ) \right) _ { 0 , 1 : } , } \\ { \displaystyle R = \mathrm { n o r m } _ { [ 0 , 1 ] } ( \mathrm { r e s h a p e } ( \mathbf { a } ) ) . } \end{array}\tag{2}
$$

Here a contains the raw CLS-to-patch rollout weights, and R is the per-image min– max normalized patch-level map. During training, R is bilinearly upsampled to the resolution of $M ^ { * }$ . Consequently, $\mathcal { L } _ { \mathrm { g e o } }$ constrains the relative spatial distribution of the rollout rather than its absolute magnitude.

The adaptation objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { g e o } } + \beta _ { k l } \mathcal { L } _ { \mathrm { K L } }\tag{3}
$$

with a depth-weighted MSE alignment term

$$
\mathcal { L } _ { \mathrm { g e o } } = \frac { \sum _ { i , j } W _ { i j } ^ { \mathrm { v a l i d } } \left( R _ { i j } - M _ { i j } ^ { * } \right) ^ { 2 } } { \sum _ { i , j } W _ { i j } ^ { \mathrm { v a l i d } } }\tag{4}
$$

and a KL representation anchor to the frozen reference backbone:

$$
\mathcal { L } _ { \mathrm { K L } } = D _ { \mathrm { K L } } \Big ( \sigma ( \mathbf { z } _ { \mathrm { C L S } } ^ { \mathrm { f r o z e n } } ) \ \Big \| \ \sigma ( \mathbf { z } _ { \mathrm { C L S } } ^ { \mathrm { a d a p t } } ) \Big )\tag{5}
$$

where $\mathbf { z } _ { \mathrm { C L S } }$ denotes the DINOv2 [CLS] embedding and σ applies softmax across its feature dimensions. The KL term is separate from the geometric alignment: the frozen backbone serves as the reference, and the adapted backbone is penalized when its softmax-normalized CLS embedding diverges from the frozen one. This representation anchor discourages large changes in the CLS embedding while still allowing LoRA to adjust the spatial routing induced by the geometric objective.

## 3.4 Hazard Probe and RGB-Only Inference

After adaptation, the DINOv2 backbone and LoRA adapters are frozen. For each RGB frame, the unnormalised CLS-to-patch rollout weights from Eq. 2 pool the final patch tokens:

$$
\mathbf { h } ( x ) = \sum _ { p = 1 } ^ { P } \frac { a _ { p } } { \sum _ { q } a _ { q } } \mathbf { t } _ { p } ,\tag{6}
$$

where P is the number of image patches and $\mathbf { t } _ { p } \in \mathbb { R } ^ { L }$ is the final DINOv2 patch token. A linear probe maps h(x) to a hazard logit and is trained on the labeled hazard set with binary focal loss [20].

![](images/20ea47a25ecdde84b6e56076de1a9135c99ab350336a3ebf2c514582e2b95e49.jpg)  
Fig. 2. Intervention-verified hazards in the RGB benchmark. The first two columns show an elongated, light-coloured fallen log; the last two show a stubby, dark-coloured woody protrusion. The top row gives front and top views; the bottom row gives side and rover-scale views. Red boxes are visual annotations only and are not used for training supervision.

## 4 Experimental Evaluation

## 4.1 Datasets and Protocol

Data sources. The geo-anchored adaptation stage uses 15,223 RGB–depth traversal pairs from ForEnt [19], collected by a Go2 legged robot without hazard labels. Hazard identification uses a separately collected, RGB-only benchmark from a wheeled rover. The corpora share the broad forest navigation domain but difer in robot platform, camera placement, collection sessions, and failure mechanism. This mismatch is intentional: GAFT uses depth only to obtain a generic geometric bias toward nearby above-ground structure, not a failure-mode-specific hazard mask. The two corpora are disjoint: no Stage 2 frame, label, scenario, or depth map is used during Stage 1 adaptation.

Event-based hazard collection. The rover executed the probing-driven Blind-Wayfarer policy in unstructured forest terrain [29], while its RGB stream was logged continuously. A hazard event was recorded only when contact with an obstacle immobilized the rover and required human intervention. We retrospectively extract the associated pre-intervention RGB window as failure-associated evidence; normal frames come from uninterrupted traversal. Thus the positive label denotes an observed robot–terrain failure event rather than a hand-drawn obstacle category or pixel-level annotation. The 124 positive frames were collected from 40 intervention-verified approach runs, each contributing one pre-intervention sequence. Of these, 29 runs involved elongated, light-coloured fallen logs and 11 involved stubby, dark woody protrusions, yielding 67 and 57 positive frames, respectively. The two scenarios difer in colour, shape, and local texture, making LOSO transfer a deliberately hard test of reliance on appearance-specific cues. Figure 2 shows their physical scale and visual ambiguity against leaf litter.

Table 1. Data used for geo-anchored adaptation and hazard identification.
<table><tr><td>Corpus</td><td>Platform/modality</td><td>Supervision/ role</td></tr><tr><td>ForEnt Hazard set</td><td>Go2 /RGB-depth Rover / RGB</td><td>Adaptation; no hazard labels; 15 223 pairs Recovery labels; 124 hazard, 2817 normal</td></tr></table>

Repeated LOSO protocol. For each fixed Stage 1 adapter, we run 50 repeated twodirection Leave-One-Scenario-Out (LOSO) evaluations. Each repetition randomly partitions the original normal frames into two disjoint halves and evaluates both scenario directions. In either direction, all positives from one scenario, together with one normal half, form the outer training set; original positives from the other scenario and the remaining normal half form the outer test set. The roles of the scenarios and normal halves are reversed in the complementary direction. Thus, the hazard scenarios and their positive frames remain fixed across repetitions; only the normal-frame partition, inner folds, and probe fitting are redrawn.

Within every outer training set, grouped five-fold CV selects the F -optimal decision threshold. Twenty augmented variants are used only for each positive training frame; variants of the same source frame remain in the same inner-CV group, and all test sets contain original frames only. The afine probe is reinitialized and trained on the complete outer training set before evaluation. We average the two directional test scores to obtain one repetition score. Its variation reflects Stage 2 resampling and probe-fitting variability. For GAFT, we repeat this complete Stage 2 protocol for each independently seeded Stage 1 adapter. These seeds change the LoRA initialization and subsequent geometryadaptation trajectory; variation across adapter-run means therefore reflects Stage 1 adaptation, whereas variation within an adapter run reflects Stage 2 evaluation.

Primary metric. We report the standard $F _ { \beta }$ score with $\beta = 2 ,$ which weights recall four times more than precision. This reflects the asymmetric cost of hazard detection: a missed hazard can cause mechanical failure, while a false alarm causes a conservative stop. At the observed 1:23 positive-to-negative ratio, a stratified-random predictor has expected $F _ { 2 } { \approx } 0 . 0 4 2$ , which we use as a chancelevel reference.

Baselines. The DINOv2-ViT-S (frozen) baseline uses rollout-weighted features from the frozen backbone with a linear head and represents the unmodified foundation model without task-specific adaptation. The supervised PEFT baselines adapt DINOv2 directly from scarce hazard labels without Stage 1 geometry: Sup-LoRA QKV, Sup-LoRA QKV+MLP, deep visual prompt tuning (VPT), and AdaptFormer. Sup-LoRA QKV+MLP uses the same QKV+MLP adapter scope as the reported GAFT model, testing whether the gain can be explained by adapter capacity alone. GAFT QKV applies geo-anchored adaptation with LoRA on QKV projections only. GAFT QKV+MLP is the reported configuration and also adapts the MLP projections. All learned methods use the same DINOv2 backbone, the same hazard-probe protocol, and RGB-only inference.

![](images/77d450ef2424de58bbf9dcdbb98a341e4785a8951d28a8674dfa257313ceff91.jpg)

![](images/2fdca27c1873fe83db9cb3c0dc8dcb071d84db92b4ae4edb570d393d27740b59.jpg)  
Fig. 3. Repeated-LOSO results. Left: a faint point is one Stage 2 repetition, averaged over the two complementary LOSO directions. For each adapted method, one large coloured circle and bar give the mean ± standard deviation of 50 repetitions with one fixed Stage 1 adapter; their separation shows variation across ten independently trained adapters. Each black square and bar summarizes the adapter-run means. Frozen DINOv2 has no adaptation run, so its diamond and bar summarize its 50 repetitions directly. Right: paired ∆F<sub>2</sub> of GAFT QKV+MLP with hierarchical-bootstrap 95% confidence intervals. “Sup” abbreviates Sup-LoRA.

Implementation details. All RGB–depth pairs use a bottom-centre 448 × 448 crop resized to $2 2 4 \times 2 2 4$ , with camera intrinsics adjusted before back-projection. The RANSAC plane is seeded from the lower 15% of image rows. Unless otherwise stated, the prior uses ϵ = 0.05 m, $e _ { \mathrm { m a x } } = 0 . 5 \mathrm { m }$ , λ = 0.3, and $d _ { \operatorname* { m a x } } = 5$ m. These geometric parameters were fixed before hazard-label training and were not selected on the LOSO test scenarios. The reported GAFT QKV+MLP model uses LoRA rank $r = 3 2$ , α = 0.5, and $\beta _ { k l } = 0 . 0 5$ , for 2.06M trainable adapter parameters. The Stage 2 probe uses focal loss with $\alpha = 0 . 7 5$ and $\gamma = 2 . 0$

## 4.2 Main Results

Table 2 reports aggregate point estimates under this protocol. GAFT achieves mean $F _ { 2 } = 0 . 3 7 5 7$ across ten independently initialized Stage 1 adapters, compared with 0.0607 for frozen DINOv2 and an expected 0.042 for a stratified-random predictor. GAFT QKV reaches 0.2695, and the paired analysis below shows that both GAFT variants reliably outperform frozen DINOv2 and supervised PEFT controls. The supervised PEFT baselines use the same DINOv2 backbone, focal loss, train-only positive augmentation, inner-CV threshold selection, and repeated LOSO protocol as GAFT, but omit the geo-anchored adaptation stage. Despite this matched protocol, Sup-LoRA QKV+MLP reaches only $F _ { 2 } = 0 . 0 1 3 8$ while VPT and AdaptFormer reach 0.0123 and 0.0282. Table 3 explains this gap: supervised PEFT fits the source scenario but transfers poorly to the heldout hazard scenario, so the low test scores are not simply an optimization or threshold-selection failure.

Table 2. LOSO hazard-identification estimates. Adapted methods average ten initialization seeds; Figure 3 shows repeated-LOSO variation and paired confidence intervals.
<table><tr><td>Method</td><td>Adapter params</td><td> $F _ { 2 }$  ↑</td><td></td><td>Recall ↑ Precision ↑</td></tr><tr><td>Stratified random</td><td></td><td>0.042</td><td>0.042</td><td>0.042</td></tr><tr><td>Frozen DINOv2</td><td>none</td><td>0.0607</td><td>0.050</td><td>0.650</td></tr><tr><td>Sup-LoRA QKV</td><td>0.59M</td><td>0.0039</td><td>0.003</td><td>0.069</td></tr><tr><td>Sup-LoRA  $\mathrm { Q K V + M L P }$ </td><td>2.06M</td><td>0.0138</td><td>0.012</td><td>0.119</td></tr><tr><td>VPT</td><td>0.05M</td><td>0.0123</td><td>0.012</td><td>0.086</td></tr><tr><td>AdaptFormer</td><td>0.60M</td><td>0.0282</td><td>0.035</td><td>0.167</td></tr><tr><td>GAFT QKV</td><td>0.59M</td><td>0.2695</td><td>0.239</td><td>0.679</td></tr><tr><td>GAFT QKV+MLP</td><td>2.06M</td><td>0.3757</td><td>0.358</td><td>0.749</td></tr></table>

Table 3. Diagnostic for supervised PEFT: within-scenario fit versus cross-scenario transfer. Inner-CV and outer-train scores use the source scenario of each LOSO direction; the LOSO test score uses its held-out scenario. All values are means over the repeated LOSO evaluations.
<table><tr><td rowspan="2">Method</td><td colspan="2">Within-scenario</td><td>Cross-scenario</td></tr><tr><td>Inner-CV  $F _ { 2 }$ </td><td>Outer-train  $F _ { 2 }$ </td><td>LOSO test  $F _ { 2 }$ </td></tr><tr><td>Sup-LoRA QKV</td><td>0.944</td><td>0.975</td><td>0.0039</td></tr><tr><td>Sup-LoRA  $\mathrm { Q K V + M L P }$ </td><td>0.952</td><td>0.987</td><td>0.0138</td></tr><tr><td>VPT</td><td>0.853</td><td>0.922</td><td>0.0123</td></tr><tr><td>AdaptFormer</td><td>0.832</td><td>0.860</td><td>0.0282</td></tr></table>

Table 3 shows that the poor PEFT scores in Table 2 are not simply explained by failure to fit the source scenario. Every supervised PEFT method obtains high inner-CV and outer-train $F _ { 2 }$ within that scenario, but low $F _ { 2 }$ when LOSO evaluates the other scenario. On the held-out scenario, these models predict hazard for only 0.05–1.51% of frames and have zero recall in 70.2–90.2% of evaluation folds. This pattern is consistent with scenario-specific cue reliance under scarce failure labels, rather than with a simple optimization failure. It does not by itself identify the cues or establish their causal role.

To test whether the gains persist under matched Stage 2 evaluations, we pair methods on the same LOSO direction and repetition. For each Stage 1 adapter run, paired diferences are averaged over its repeated outer evaluations; hierarchical bootstrap confidence intervals then resample adapter runs and paired evaluations.

Across ten seed-matched runs (1,000 paired Stage 2 folds per comparison), Figure 3 shows positive intervals against frozen DINOv2 $( \varDelta F _ { 2 } = + 0 . 3 1 5 0 \ [ 0 . 2 3 5 0 , 0 . 3 9 0 4 ] )$ and matched-capacity Sup-LoRA QKV+MLP (+0.3619 [0.2865, 0.4284]), but the advantage over GAFT QKV is not robust $\left( + 0 . 1 0 6 2 \ [ - 0 . 0 0 7 3 , 0 . 2 1 1 8 ] \right)$

Figure 3 separates the two levels of variation. Each coloured circle and bar gives the mean and standard deviation of 50 Stage 2 repeats for one independently initialized adaptation run; for GAFT, this is one Stage 1 adapter. The black square and bar summarize the resulting run means. The supervised PEFT baselines remain near zero at both levels, whereas GAFT shifts the distribution upward. In this setting, matched adapter capacity alone does not yield cross-scenario performance; coupling it with the geo-anchored objective does. MLP adaptation should nevertheless be interpreted as a capacity choice rather than a statistically isolated contribution.

The broad GAFT range in Figure 3 should not be read as instability of a fixed adapted backbone under repeated LOSO. Holding each Stage 1 adapter fixed, we perform 50 independent Stage 2 repetitions; the resulting standard deviation, averaged across the ten independently trained adapters, is 0.0575 for GAFT QKV and 0.0536 for GAFT QKV+MLP. The wider separation among the corresponding adapter means is therefore introduced when Stage 1 constructs the geometry-adapted backbone. In this implementation, a Stage 1 seed afects both LoRA initialization and the subsequent adaptation trajectory, including minibatch order and checkpoint selection, which are known sources of fine-tuning variability [5, 12].

## 4.3 Ablation Studies

We first test whether the pooling used by the frozen reference explains its weak performance, then examine how KL anchoring strength and LoRA capacity afect performance under the geo-anchored objective.

Pooling control. For frozen DINOv2, CLS pooling performs poorly, whereas mean-patch and rollout pooling yield similar $F _ { 2 }$ values (Table 4). Thus the weak frozen reference is not an artifact of rollout pooling. We retain rollout pooling because it provides the spatial interface through which GAFT applies its Stage 1 geometric supervision.

KL anchoring. In the QKV-only RANSAC setting, the strong-anchor configuration $\beta _ { k l } = 0 . 0 5$ reaches a five-seed mean of $0 . 2 5 8 7 \pm 0 . 1 5 3 5$ , compared with $0 . 0 5 9 6 \pm 0 . 0 6 2 4$ at $\beta _ { k l } = 0 . 0 0 1$ (Table 4). The weak-anchor settings are highly seed-sensitive and do not consistently exceed the frozen baseline. The ablation therefore shows that performance depends on suficiently strong anchoring, without isolating the particular mechanism by which the KL term provides it.

LoRA capacity. We also vary QKV-only LoRA rank at $\beta _ { k l } = 0 . 0 5$ and RANSAC $M ^ { * }$ . In this five-seed ablation, $r = 8$ is weak and seed-sensitive $( 0 . 0 7 6 3 \pm 0 . 0 5 2 7 )$ whereas $r = 1 6$ has a higher and more stable mean $( 0 . 2 3 6 9 \pm 0 . 0 2 3 8 )$ . Rank $r = 3 2$ reaches a slightly higher mean $( 0 . 2 5 8 7 \pm 0 . 1 5 3 5 )$ but with larger seed variance. These exploratory results indicate that very low rank is inadequate for this objective, while increasing capacity alone does not guarantee a stable improvement.

Table 4. Pooling control and ablations. Frozen pooling results are mean ± standard deviation over 50 Stage 2 repetitions. GAFT ablations are mean ± standard deviation across five Stage 1 seeds.
<table><tr><td>Factor</td><td>Setting</td><td> $F _ { 2 } \uparrow$ </td></tr><tr><td rowspan="3">Frozen pooling</td><td>CLS</td><td> $0 . 0 0 9 5 \pm 0 . 0 1 7 0$ </td></tr><tr><td>Mean patch</td><td> $0 . 0 5 8 8 \pm 0 . 0 3 5 6$ </td></tr><tr><td>Rollout</td><td> $0 . 0 6 0 7 \pm 0 . 0 2 0 9$ </td></tr><tr><td rowspan="4"> $\beta _ { k l } , \mathrm { Q K V / R A N S A C }$ </td><td>0.001</td><td> $0 . 0 5 9 6 \pm 0 . 0 6 2 4$ </td></tr><tr><td>0.005</td><td> $0 . 0 8 9 5 \pm 0 . 0 9 1 4$ </td></tr><tr><td>0.010</td><td> $0 . 0 5 2 9 \pm 0 . 0 1 1 8$ </td></tr><tr><td>0.050</td><td> $0 . 2 5 8 7 \pm 0 . 1 5 3 5$ </td></tr><tr><td rowspan="4">Rank, QKV/RANSAC,</td><td> $r = 8$ </td><td> $0 . 0 7 6 3 \pm 0 . 0 5 2 7$ </td></tr><tr><td> $\beta _ { k l } = 0 . 0 5 ~ r = 1 6$ </td><td> $0 . 2 3 6 9 \pm 0 . 0 2 3 8$ </td></tr><tr><td> $r = 3 2$ </td><td> $0 . 2 5 8 7 \pm 0 . 1 5 3 5$ </td></tr><tr><td></td><td></td></tr></table>

## 5 Discussion

The central comparison shows that, in this data-scarce setting, matched adapter capacity and hazard-label supervision alone do not yield cross-scenario hazard identification. Supervised PEFT fits the source scenario but transfers poorly to the held-out scenario, whereas geo-anchored adaptation substantially improves the repeated-LOSO distribution. This contrast supports the role of geo-anchored adaptation as an inductive bias: the supervised PEFT controls learn a sourcescenario decision rule, whereas GAFT improves transfer to the held-out hazard scenario.

The ablations further show that the benefit is conditional. Weak KL anchoring and very low LoRA rank yield weak or seed-sensitive results, while the diference between GAFT QKV and GAFT QKV+MLP is not robust under the paired analysis. Accordingly, QKV+MLP is the reported practical configuration rather than a separate statistically isolated contribution. Alternative representation anchors and more accurate geometric targets remain open design choices.

These results should be interpreted as evidence for ofline RGB hazard identification under rare, intervention-verified field failures. The benchmark contains two verified forest hazard morphologies, so the evaluation measures transfer between these morphologies rather than robustness across arbitrary seasons, platforms, sensor placements, or closed-loop navigation policies. The 50 repeated LOSO evaluations quantify Stage 2 resampling and probe-fitting variation, not independent environments. GAFT is also not claimed to produce a pixel-accurate hazard explanation: attention rollout is used as a diferentiable spatial-routing interface, and $M ^ { * }$ is a proximity-weighted geometric proxy rather than a hazard mask. Closed-loop validation of whether GAFT reduces immobilization, recovery frequency, or mission time remains future work.

## 6 Conclusion

We introduced GAFT, a two-stage method that uses a proximity-weighted geometric prior to adapt an RGB vision foundation model before hazard-label supervision. The prior is available only during Stage 1; Stage 2 and deployment use RGB alone. Across ten independently initialized adaptations and repeated LOSO evaluation, GAFT improves frozen DINOv2 from mean $F _ { 2 } = 0 . 0 6 0 7$ to 0.3757. Matched supervised PEFT controls fit their source scenario but fail to transfer reliably to the held-out one. Together, these results support a narrow conclusion: geo-anchored adaptation can improve RGB-only hazard identification when hazard labels are scarce and adaptation remains anchored to pretrained features.

## References

1. Abnar, S., Zuidema, W.: Quantifying attention flow in transformers. In: Proceedings of the 58th annual meeting of the association for computational linguistics. pp. 4190–4197 (2020)

2. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

3. Chavez-Garcia, R.O., Guzzi, J., Gambardella, L.M., Giusti, A.: Learning ground traversability from simulations. IEEE Robotics and Automation letters 3(3), 1695– 1702 (2018)

4. Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., Luo, P.: Adaptformer: Adapting vision transformers for scalable visual recognition. Advances in Neural Information Processing Systems 35, 16664–16678 (2022)

5. Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., Smith, N.: Finetuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305 (2020)

6. Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: International conference on machine learning. pp. 1126–1135. PMLR (2017)

7. Frey, J., Hoeller, D., Khattak, S., Hutter, M.: Locomotion policy guided traversabil ity learning using volumetric representations of complex environments. In: 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 5722–5729. IEEE (2022)

8. Fu, Z., Kumar, A., Agarwal, A., Qi, H., Malik, J., Pathak, D.: Coupling vision and proprioception for navigation of legged robots. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17273–17283 (2022)

9. Gasparino, M.V., Sivakumar, A.N., Liu, Y., Velasquez, A.E., Higuti, V.A., Rogers, J., Tran, H., Chowdhary, G.: Wayfast: Navigation with predictive traversability in the field. IEEE Robotics and Automation Letters 7(4), 10651–10658 (2022)

10. Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine Intelligence 2(11), 665–673 (2020)

11. Hadsell, R., Sermanet, P., Ben, J., Erkan, A., Scofier, M., Kavukcuoglu, K., Muller, U., LeCun, Y.: Learning long-range vision for autonomous of-road driving. Journal of Field Robotics 26(2), 120–144 (2009)

12. Hayou, S., Ghosh, N., Yu, B.: The impact of initialization on lora finetuning dynamics. Advances in Neural Information Processing Systems 37, 117015–117040 (2024)

13. Howard, A., Turmon, M., Matthies, L., Tang, B., Angelova, A., Mjolsness, E.: Towards learned traversability for robot navigation: From underfoot to the far field. Journal of Field Robotics 23(11-12), 1005–1017 (2006)

14. Hu, E.J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022)

15. Hwang, J.H., Bae, J., Kim, D.W., Lee, Y., Seo, S.W.: From general vision to reliable traversability estimation: Adapting vision foundation models for unstructured outdoor environments. arXiv preprint arXiv:2605.29565 (2026)

16. Ji, T., Sivakumar, A.N., Chowdhary, G., Driggs-Campbell, K.: Proactive anomaly detection for robot navigation with multi-sensor fusion. IEEE Robotics and Au tomation Letters 7(2), 4975–4982 (2022)

17. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: European conference on computer vision. pp. 709–727. Springer (2022)

18. Kahn, G., Abbeel, P., Levine, S.: BADGR: An autonomous self-supervised learningbased navigation system. IEEE Robotics and Automation Letters 6(2), 1312–1319 (2021)

19. Kirdwichai, N., Tarapore, D.: Forent: A multi-modal dataset for characterizing quadruped robot entrapments in forest environments. arXiv preprint arXiv:2606.19675 (2026)

20. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: Proceedings of the IEEE international conference on computer vision. pp. 2980–2988 (2017)

21. Mattamala, M., Frey, J., Libera, P., Chebrolu, N., Martius, G., Cadena, C., Hutter, M., Fallon, M.: Wild visual navigation: Fast traversability learning via pre-trained models and online self-supervision. Autonomous Robots 49(3), 19 (2025)

22. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal (2024)

23. Qiu, C., Middlehurst, M., Holder, C., Bagnall, A.: e-smote: a train set rebalancing algorithm for time series classification. In: International Workshop on Advanced Analytics and Learning on Temporal Data. pp. 1–17. Springer (2025)

24. Qiu, C., Tang, T., Yang, T., Chen, M.: Learning to generalize with latent embedding optimization for few-and zero-shot cross domain fault diagnosis. Expert Systems with Applications 254, 124280 (2024)

25. Snell, J., Swersky, K., Zemel, R.: Prototypical networks for few-shot learning. Advances in neural information processing systems 30 (2017)

26. Wang, Q., Tang, T., Wu, J., Mo, F., Qiu, C., Chen, M.: Masked autoencoders-based fault diagnosis pretraining model with meta fine-tuning and prototype prompting for few-shot fault diagnosis. Knowledge-Based Systems p. 115871 (2026)

27. Wellhausen, L., Dosovitskiy, A., Ranftl, R., Walas, K., Cadena, C., Hutter, M.: Where should i walk? predicting terrain properties from images via self-supervised learning. IEEE Robotics and Automation Letters 4(2), 1509–1516 (2019)

28. Wellhausen, L., Ranftl, R., Hutter, M.: Safe robot navigation via multi-modal anomaly detection. IEEE Robotics and Automation Letters 5(2), 1326–1333 (2020)

29. Xu, Y., Zauner, K.P., Tarapore, D.: Blind-wayfarer: A minimalist, probing-driven framework for resilient navigation in perception-degraded environments. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 19340–19345 (2025). https://doi.org/10.1109/IROS60139.2025.11246357

30. Xue, H., An, Y., Qin, Y., Li, W., Wu, Y., Che, Y., Fang, P., Zhang, M.L.: Toward few-shot learning in the open world: A review and beyond. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(11), 10420–10440 (2025). https://doi.org/10.1109/TPAMI.2025.3594686

31. Yue, Y., Das, A., Engelmann, F., Tang, S., Lenssen, J.E.: Improving 2d feature representations by 3d-aware fine-tuning. In: European Conference on Computer Vision. pp. 57–74. Springer (2024)

32. Zhang, C., Jin, J., Frey, J., Rudin, N., Mattamala, M., Cadena, C., Hutter, M.: Resilient legged local navigation: Learning to traverse with compromised perception end-to-end. In: IEEE International Conference on Robotics and Automation. pp. 34–41 (2024)