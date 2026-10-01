# Beyond Spatial-Domain Supervision: A Relation Constrained Space for Multi-Modal Image Fusion

Zeyu Wang College of Computer Science and Engineering Dalian Minzu University 20231578@dlnu.edu.cn

Haiyu Song<sup>∗</sup> College of Computer Science and Engineering Dalian Minzu University shy@dlnu.edu.cn

Mingyu Ge College of Computer Science and Engineering Dalian Minzu University 2019082204@stu.dlnu.edu.cn

Haoran Duan<sup>∗</sup>   
Department of Automation   
Tsinghua University   
haoran.duan@ieee.org

## Abstract

Multi-modal image fusion (MMIF) aims to form a single image by integrating shared information, preserving complementary cues, and coordinating cross-modal conflicts across modalities. However, due to the absence of ground-truth fused images, existing MMIF supervision commonly uses spatial-domain sources or gradient variants as surrogate ground truth, making the supervision mechanism inherently misaligned with the goal of MMIF and causing pixel-level compromise or modality bias. To address this, we propose a relation-constrained supervision paradigm that moves fusion supervision from the spatial domain to a learned relation space. Rather than relying solely on direct source approximation, we further leverage frozen pretrained representation models as information providers and design a learnable feature adapter to align heterogeneous DINO and CLIP features into a unified supervision space. The adapter infers three relation parameters, namely sharedness, dominance, and coordination radius, which define three losses corresponding to the MMIF’s goal. To make this space reliable, we devise a selfsupervised contrastive ranking objective tailored to the adapter and couple it with the fusion network through alternating optimization. Extensive experiments show that the proposed supervision space yields significant gains regardless of which mainstream backbone the fusion network adopts, offering a supervision paradigm better aligned with the goal of MMIF. Code: github.com/GMY628/RCS-Fusion.

## 1 Introduction

Multi-modal Image Fusion (MMIF) combines complementary information from different modalities into one image [25, 26, 45]. Representative tasks include infrared visible fusion [13, 44], which integrates thermal cues with visible textures, and medical image fusion, which combines anatomical and functional information. MMIF also benefits the object detection [12] and medical segmentation [32].

Recent MMIF methods have advanced architectures, cross-modal interaction, learning strategies, and loss design [50, 31, 2, 30, 39]. Despite this, due to the absence of ground-truth (GT) fused images, existing methods typically treat spatial-domain source images or their gradient variants as surrogate targets for loss computation. This practice has become standard, but introduces a fundamental problem: the supervision mechanism is inherently misaligned with the goal of MMIF.

![](images/a489254a39d34f90c08fc405b1072aca74c63feda4c571c8d3bbb3e6997fe59d.jpg)  
Figure 1: Vanilla MMIF Supervision Paradigm vs Our Paradigm. Our method construct a relationconstraint supervision space tailored for MMIF, leading to better modality balance and fusion quality.

The goal of MMIF is to integrate shared information, preserve modality specific complementary cues, and coordinate conflicts between heterogeneous observations, rather than reconstruct any individual source image. Source images should therefore act as information providers instead of reconstruction targets. Once they are treated as simultaneous GT, the model is forced to satisfy competing constraints from different modalities. This leads to either a pixel-level compromise or a biased result dominated by one modality, as shown in Fig. 1. More importantly, complementary cues, such as infrared thermal targets and visible fine textures, are converted into conflicting supervisory signals.

A natural solution is to move supervision beyond raw spatial domain source images. Recent pretrained representation models [19, 21] provide rich visual features that encode structural and semantic cues. Nevertheless, directly using these features does not solve the problem. Features from different pretrained models are heterogeneous and cannot naturally form a unified supervision space. More critically, ideal MMIF supervision requires not only strong representation capacity, but also explicit relation modeling between modalities: what information is shared, what information is complementary, and how conflicting cues should be coordinated. Without such relation modeling, pretrained features may enrich the representation, but supervision remains insufficiently aligned with fusion.

Motivated by this observation, we propose a relation-constrained supervision paradigm for MMIF. Beyond direct supervision from source images as surrogate ground truths, we further construct an adaptive supervision space from frozen pretrained representation models and constrain it according to cross-modal relations. Our core idea is that a proper fusion supervision space should be defined by three fundamental relations: sharedness, complementarity, and coordination. Sharedness retain information consistently supported by multiple sources. Complementarity preserves modality specific cues that are informative but weak or absent in the other modality. Coordination prevents conflicting responses from driving the fused result excessively toward either source. Together, these relations provide a supervision principle that matches the intrinsic objective of MMIF.

To implement this principle, we design a learnable feature adapter (LFA) that bridges heterogeneous pretrained features and the proposed supervision space. LFA consists of a Feature Alignment Module and a Relation Reasoning Module. The former projects intermediate features from frozen DINO and CLIP into a unified feature space. Within this space, we tailor three losses, namely sharedness, complementarity, and coordination losses, to supervise fusion. The latter predicts three relation parameters that adaptively weight these losses, allowing supervision to reflect the cross-source relationship of each image pair rather than fixed manual coefficients.

Besides, training the adapter is critical yet challenging. An unreliable adapter would produce misleading supervision and thus undermine any fusion model trained with it. However, no ready made training set or established strategy exists for learning such a relation-aware adapter, creating a circular dependency between adapter learning and fusion supervision. We therefore tailor a selfsupervised contrastive training scheme to the adapter and couple it with alternating optimization. Specifically, we pretrain the adapter with a ranking loss on relation scores computed from its outputs, so that matched pairs (positive pairs) score higher than unmatched ones (negative pairs). This enables LFA to learn cross-modal relation estimation without explicit relation annotations. Before alternating optimization, the fusion network is first trained with a conventional fusion loss to reach a baseline level of fusion performance. We then alternate between freezing the adapter to supervise the fusion network with relation losses and freezing the fusion network to train the adapter with fused images under the same ranking objective, encouraging each fused image to score higher with its matched sources than with mismatched ones. This strategy enables effective training of both the adapter and the fusion network. Our contributions are summarized as follows:

•We revisit MMIF supervision from the objective of fusion and propose a relation-constrained paradigm beyond spatial domain source images as surrogate ground truths.

•We define the supervision space through sharedness, complementarity, and coordination, aligning supervision with the requirements of fusion.

•We design a learnable feature adapter that aligns heterogeneous pretrained features and predicts relation parameters to automatically construct the supervision space.

•We propose a self-supervised relation ranking strategy and an alternating optimization scheme to jointly train the adapter and fusion network, forming a bidirectional optimization mechanism.

## 2 Related Work

Supervision Paradigms for MMIF. Recent MMIF research has progressed from reconstructionbased frameworks to recent Transformer, diffusion, and language-guided models [50, 53, 40, 28, 52]. Because no physical ground-truth fused image exists, their performance remains highly dependent on supervision loss design [3, 2]. Existing methods mainly follow source-preservation, perceptual or adversarial, task-driven, and semantic or language-guided paradigms [14, 33, 35, 44]. However, they still use spatial-domain source images as GT or proxy targets for loss computation. This misaligns optimization with the objective of MMIF and often causes pixel-level compromise or modality bias. In contrast, our method abandons this long-standing practice and uses powerful frozen pretrained representation models to build a supervision paradigm better aligned with the goal of MMIF.

Pretrained Visual Knowledge for MMIF. Recent advances in pretrained representation models have made large-scale visual knowledge a promising resource for image fusion. Models such as CLIP and DINOv2 learn rich structural and semantic cues from massive data, and their representations have shown strong transferability in low-level vision tasks [37, 8, 1]. In image fusion, recent studies have explored knowledge-aware or representation-driven fusion through semantic guidance [49, 46, 2], text descriptions [52, 41, 6], or self-supervised representation learning [51, 50, 34]. However, these methods mainly use pretrained knowledge as auxiliary features or side information and do not turn it into direct supervision for fusion. In contrast, our method abandons this long-standing practice and uses external knowledge from powerful frozen pretrained representation models to build a supervision paradigm better aligned with the fundamental objective of MMIF.

## 3 Methodology

Overview. As shown in Fig. 2, our method consists of three tightly coupled components: a learnable feature adapter (LFA), relation-constrained supervision (RCS) for MMIF, and a tailored training strategy. The LFA maps heterogeneous multi-level features from frozen DINO [19] and CLIP [21] into a unified supervision space and infers source relations. Based on these outputs, we construct the RCS with three losses for shared information integration, complementary cues preservation, and conflict coordination. Finally, we train the LFA with a self-supervised contrastive learning strategy and optimize it alternately with the fusion net. Appendix B gives the formal problem statement.

## 3.1 Learnable Feature Adapter (LFA)

The learnable feature adapter (LFA) contains two components: a Feature Alignment Module (FAM) and a Relation Reasoning Module (RRM). For each source image $X \in \{ A , { \bar { B } } \}$ , frozen DINO and CLIP extract three feature maps from DINO layer 4, CLIP layer 6, and DINO layer 8, denoted by $f _ { D } ^ { 4 } ( X ) , f _ { C } ^ { 6 } ( X )$ , and $f _ { D } ^ { 8 } ( X )$ , respectively. FAM takes these three feature maps as input and outputs an aligned representation $y x$ in the unified supervision space. RRM then takes the aligned source pair $( y _ { A } , y _ { B } )$ as input and outputs two pair-conditioned source representations, $z _ { A }$ and $z _ { B }$ , together with three relation parameters: sharedness $m _ { A , B } ,$ dominance $d _ { A , B }$ , and coordination radius $r _ { A , B }$

Feature Alignment Module. To address the mismatch in spatial layout and representation structure across models and feature depths, FAM processes the multi-level features of a source pair $( A , B )$ with a shared projection-and-alignment pipeline. Specifically, for each source image $\bar { X ^ { \circleddash } } \{ \dot { A } , \bar { B } \}$ FAM projects the three feature maps to the same channel dimension and a common token grid:

$$
\begin{array} { r } { \tilde { f } _ { D } ^ { 4 } ( X ) = P _ { D } ^ { 4 } \big ( f _ { D } ^ { 4 } ( X ) \big ) , \quad \tilde { f } _ { C } ^ { 6 } ( X ) = P _ { C } ^ { 6 } \big ( f _ { C } ^ { 6 } ( X ) \big ) , \quad \tilde { f } _ { D } ^ { 8 } ( X ) = P _ { D } ^ { 8 } \big ( f _ { D } ^ { 8 } ( X ) \big ) , } \end{array}\tag{1}
$$

![](images/ce0057ffa1db4680ae0cea3eaf7ae884a8c0d6e9d35dd4f7b0661716500aa72a.jpg)  
Figure 2: Overview of the Learnable Feature Adapter and Relation-Constrained Supervision.

where $P _ { D } ^ { 4 } , P _ { C } ^ { 6 }$ , and $P _ { D } ^ { 8 }$ denote the corresponding projection operators, and ${ \tilde { f } } _ { D } ^ { 4 } ( X ) , { \tilde { f } } _ { C } ^ { 6 } ( X )$ , and $\tilde { f } _ { D } ^ { 8 } ( X )$ denote the projected features. Each projector is implemented as $\mathbf { a \ 1 \times 1 }$ convolution followed by normalization and bilinear resizing, so that the three inputs share the same token layout. The projected features are then flattened into token sequences, concatenated along the channel dimension, and fused by a cross-model aligner:

$$
y _ { X } = G _ { \mathrm { a l i g n } } \Bigl ( [ \widetilde { f } _ { D } ^ { 4 } ( X ) , \widetilde { f } _ { C } ^ { 6 } ( X ) , \widetilde { f } _ { D } ^ { 8 } ( X ) ] \Bigr ) , \quad X \in \{ A , B \} ,\tag{2}
$$

where [·] denotes concatenation and $G _ { \mathrm { a l i g n } }$ denotes a lightweight MLP-based aligner. Applying this shared pipeline to the two source images produces the aligned representations $y _ { A }$ and $y _ { B }$

Relation Reasoning Module. Given the aligned source representations $y _ { A }$ and $y _ { B }$ produced by FAM, RRM infers the relation of the source pair in the unified supervision space. To support this relation inference, RRM forms a pair input by concatenating the two aligned source representations, their element-wise absolute difference, and their element-wise product. The absolute difference captures the magnitude of source discrepancy, while the element-wise product reflects their featurewise agreement. This pair input is first encoded into a shared pair feature ${ q } _ { A , B }$ and then projected to two pair-conditioned source representations $z _ { A }$ and $z _ { B } \mathrm { : }$

$$
q _ { A , B } = E _ { \mathrm { p a i r } } \big ( [ y _ { A } , y _ { B } , | y _ { A } - y _ { B } | , y _ { A } \odot y _ { B } ] \big ) , z _ { A } = H _ { A } \big ( [ q _ { A , B } , y _ { A } ] \big ) , z _ { B } = H _ { B } \big ( [ q _ { A , B } , y _ { B } ] \big ) ,\tag{3}
$$

where $E _ { \mathrm { p a i r } }$ denotes the pair encoder, and $H _ { A }$ and $H _ { B }$ denote two source-specific projection heads. In our implementation, $E _ { \mathrm { p a i r } }$ is a lightweight M $\mathcal { \mathbf { P } }$ with GELU and LayerNorm, and $H _ { A }$ and $H _ { B }$ are two linear heads followed by normalization.

Based on the shared pair feature $q _ { A , B }$ , RRM further predicts three relation parameters:

$$
\begin{array} { r } { m _ { A , B } = \mathrm { S i g m o i d } ( h _ { m } ( q _ { A , B } ) ) , \quad d _ { A , B } = \mathrm { S i g m o i d } ( h _ { d } ( q _ { A , B } ) ) , \quad r _ { A , B } = \mathrm { S o f t p l u s } ( h _ { r } ( q _ { A , B } ) ) , } \end{array}\tag{4}
$$

where $h _ { m } , h _ { d } .$ , and $h _ { r }$ are lightweight prediction heads, each implemented by a linear layer, GELU, and LayerNorm. Accordingly, $m _ { A , B }$ measures sharedness, $d _ { A , B }$ quantifies the tendency toward source $A$ in the non-shared part, with $1 - d _ { A , B }$ giving the complementary contribution of source $B ,$ and $r _ { A , B }$ defines the coordination radius. Together, the representations $z _ { A }$ and $z _ { B }$ and the relation parameters $m _ { A , B } , d _ { A , B } .$ , and $r _ { A , B }$ are used to construct the subsequent supervision space for MMIF.

## 3.2 Relation-Constrained Supervision (RCS) for MMIF

We now use the outputs of LFA to construct relation-constrained supervision for MMIF in the unified supervision space. For a source pair (A, B), RRM provides two pair-conditioned source representations, $z _ { A }$ and $z _ { B } ,$ , together with three relation parameters, $m _ { A , B } , d _ { A , B } ,$ , and $r _ { A , B }$ . We obtain $z _ { F }$ by encoding the fusion image F self-pair $( F , F )$ with the same pair-conditioned LFA.

Since source images contain shared and non-shared information, a fixed target representation is insufficient for fusion supervision. We introduce a pretrained shared token encoder $G _ { \mathrm { s h r } }$ , which is applied with shared weights to the aligned source representations to extract their shared components:

$$
c _ { A } = G _ { \mathrm { s h r } } ( z _ { A } ) , \qquad c _ { B } = G _ { \mathrm { s h r } } ( z _ { B } ) .\tag{5}
$$

Here, $c _ { A }$ and $c _ { B }$ denote source-conditioned shared representations. During pretraining, $G _ { \mathrm { s h r } }$ is encouraged to make them highly consistent, so they retain information commonly supported by both modalities and jointly define the consensus reference. The details are in the Appendix L.

Based on the extracted shared representations, we define the relative offsets of the two source and fused representations as:

$$
u _ { A } = z _ { A } - c _ { A } , \quad u _ { B } = z _ { B } - c _ { B } , \quad u _ { F } = z _ { F } - c _ { F } ,\tag{6}
$$

where $c _ { F } = G _ { \mathrm { s h r } } ( z _ { F } )$ denotes the shared representation extracted from the fused representation. u<sub>A</sub> and u<sub>B</sub> denote the source offsets, and $u _ { F }$ denotes the fusion offset, all measured with respect to the same consensus center. These quantities allow us to define three losses for shared information integration, complementary information preservation, and conflict coordination.

Shared Information Integration. The first requirement of MMIF is to preserve information shared by the two source images. In highly shared regions, the fused representation should stay close to the consensus center rather than drift toward either modality. Accordingly, the shared loss takes the form:

$$
\mathcal { L } _ { \mathrm { s h r } } = \mathcal { D } _ { \mathrm { c o s } } ( c _ { F } , c _ { A } ) + \mathcal { D } _ { \mathrm { c o s } } ( c _ { F } , c _ { B } ) .\tag{7}
$$

where $\mathcal { D } _ { \mathrm { { c o s } } } ( \cdot , \cdot )$ denotes the cosine distance. Reducing the loss requires $c _ { F }$ to align with the shared representations $c _ { A }$ and $c _ { B }$ , which preserves common source content.

Complementary Information Preservation. Fusion should not collapse all information into the shared components. In regions with low sharedness, the fused result should preserve useful non-shared cues from the two sources. The complementary loss is given by:

$$
\mathcal { L } _ { \mathrm { c m p } } = \left( 1 - m _ { A , B } \right) \left[ d _ { A , B } \mathcal { D } _ { \mathrm { c o s } } ( u _ { F } , u _ { A } ) + \left( 1 - d _ { A , B } \right) \mathcal { D } _ { \mathrm { c o s } } ( u _ { F } , u _ { B } ) \right] .\tag{8}
$$

Here, $d _ { A , B }$ controls the relative contributions of the two source residuals to the complementary constraint. The factor $1 - m _ { A , B }$ makes this term focus on regions with low sharedness. When the loss decreases, the fused offset is encouraged to follow the dominance-aware direction defined by the two source offsets, which helps preserve complementary information from both modalities.

Conflict Coordination. Preserving complementary information alone is insufficient under strong cross-modal conflict, because the fused representation may still be driven excessively toward one source. To prevent this, we introduce a coordination constraint that keeps the fused offset within an adaptive range. The coordination loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c r d } } = \operatorname* { m a x } ( 0 , \| u _ { F } \| _ { 2 } - r _ { A , B } ) , } \end{array}\tag{9}
$$

where $r _ { A , B }$ denotes the coordination radius for the source pair $( A , B )$ . This hinge form imposes no penalty when the fused offset stays within the allowed radius. It only penalizes excessive deviation, thereby limiting conflict-driven bias while preserving flexibility for useful complementary cues.

The final relation-constrained supervision is defined as:

$$
\mathcal { L } _ { \mathrm { r e l } } = \mathcal { L } _ { \mathrm { s h r } } + \mathcal { L } _ { \mathrm { c m p } } + \mathcal { L } _ { \mathrm { c r d } } .\tag{10}
$$

In this way, fusion supervision is no longer defined by direct approximation to the source images in the spatial domain. Instead, it is defined by whether the fused representation satisfies the shared, complementary, and coordination requirements induced by source relations in the proposed supervision space. Appendix D.1 analyzes the effects of relation parameters on the supervision losses.

## 3.3 Training Strategy

Self-Supervised Contrastive Learning for the Adapter. Before using LFA to supervise the fusion network, we pre-train the adapter with a self-supervised contrastive ranking objective to make the learned supervision space discriminative for source relations. Since LFA outputs both pair-conditioned representations and relation parameters, the score used for training should depend on both parts. For a source pair $( A , B )$ , we define the source-pair relatedness score as:

![](images/10d443adec7c7e4aeffc68e2c9013f516d3ec77ea87db030cbb30016270a7a45.jpg)  
Figure 3: Overview of the Alternating Optimization Strategy.

$$
s ^ { + } = \frac { \langle w _ { A , B } , \cos ( z _ { A } , z _ { B } ) \rangle } { \| w _ { A , B } \| _ { 1 } + \epsilon } , \quad w _ { A , B } = \sigma ( \lambda _ { m } m _ { A , B } + \lambda _ { d } | 2 d _ { A , B } - 1 | - \lambda _ { r } r _ { A , B } ) ,\tag{11}
$$

where ϵ is a small constant, $\cos ( \cdot , \cdot )$ denotes token-wise cosine similarity, $\langle \cdot , \cdot \rangle$ denotes the inner product, and $\| \cdot \| _ { 1 }$ denotes the $\ell _ { 1 }$ norm used to normalize the weighted aggregation. The function $\sigma ( \cdot )$ denotes the sigmoid function, which maps the relation-aware weights into (0, 1). $\lambda _ { m } , \lambda _ { d } .$ , and $\lambda _ { r }$ are weighting coefficients. In this score, $\cos ( z _ { A } , z _ { B } )$ measures the local agreement between the two pair-conditioned representations, while $w _ { A , B }$ determines how much each token contributes to the final score. The design of $w _ { A , B }$ follows the roles of the three relation parameters: a larger m $^ { , A , B }$ gives higher weight to shared regions, a larger $| 2 d _ { A , B } - 1 |$ gives higher weight to regions with clearer dominance, and a larger $r _ { A , B }$ reduces the weight of regions that require a wider coordination range. As a result, $s ^ { + }$ becomes high only when $z _ { A }$ and $z _ { B }$ are locally consistent at tokens where $^ { m _ { A , B } }$ $d _ { A , B }$ , and $r _ { A , B }$ indicate reliable source relations. For negative source pairs such as $( A , B ^ { - } )$ and $( A ^ { - } , B )$ , the scores $s _ { 1 } ^ { - }$ and $s _ { 2 } ^ { - }$ are computed in the same way after replacing the corresponding source. Appendix E further explains the relatedness scores and ranking objective.

We first use source-side ranking to make the relation space discriminative at the source-pair level. Let $( A , B ^ { - } )$ and $( A ^ { - } , B )$ denote two negative pairs formed by replacing one source with an unpaired sample. The source-side ranking loss is:

$$
\mathcal { L } _ { \mathrm { s s r } } = \frac { 1 } { 2 } \Big ( \operatorname* { m a x } ( 0 , \delta _ { s } - s ^ { + } + s _ { 1 } ^ { - } ) + \operatorname* { m a x } ( 0 , \delta _ { s } - s ^ { + } + s _ { 2 } ^ { - } ) \Big ) ,\tag{12}
$$

where $\delta _ { s }$ denotes the source-side margin. This margin requires the matched pair to score higher than each mismatched pair by a non-trivial gap, instead of only being slightly larger.

Source-side ranking alone is insufficient, because LFA is ultimately used to supervise fused images. Without source-fused ranking, the learned relation space may not remain reliable when fused images are introduced. We first define the token-wise shared and complementary relation scores, denotes as $\rho ^ { s } , \rho ^ { c }$ respectively:

$$
\rho ^ { s } = \cos ( c _ { F } , c _ { A } ) + \cos ( c _ { F } , c _ { B } ) , \quad \rho ^ { c } = d _ { A , B } \cos ( u _ { F } , u _ { A } ) + ( 1 - d _ { A , B } ) \cos ( u _ { F } , u _ { B } ) .\tag{13}
$$

The relatedness between the fused image and its source pair is then defined as:

$$
\tilde { s } ^ { + } = \frac { \langle w _ { A , B } , m _ { A , B } \rho ^ { s } + ( 1 - m _ { A , B } ) \rho ^ { c } \rangle } { \| w _ { A , B } \| _ { 1 } + \epsilon } .\tag{14}
$$

Here, $\rho ^ { s }$ measures the consistency between the fused and source shared components, while $\rho ^ { c }$ measures the preservation of source-specific residuals according to their predicted dominance. Based on this score, the source–fused ranking loss is:

$$
\mathcal { L } _ { \mathrm { s f r } } = \frac { 1 } { 2 } \Big ( \operatorname* { m a x } ( 0 , \delta _ { f } - \tilde { s } ^ { + } + \tilde { s } _ { 1 } ^ { - } ) + \operatorname* { m a x } ( 0 , \delta _ { f } - \tilde { s } ^ { + } + \tilde { s } _ { 2 } ^ { - } ) \Big ) ,\tag{15}
$$

where $\delta _ { f }$ denotes the fused-side margin, analogous to $\delta _ { s }$ in source-side ranking. The overall adapter loss is:

$$
{ \mathcal { L } } _ { \mathrm { a d a p t e r } } = { \mathcal { L } } _ { \mathrm { s s r } } + { \mathcal { L } } _ { \mathrm { s f r } } .\tag{16}
$$

Alternating optimization. Before alternating optimization, the fusion network is first warmed up with a conventional spatial-domain fusion loss to obtain a reasonable baseline performance.

![](images/6af90e7f9235b13808226c60104a66cec66b98fc25e48be94a75e68e88b9b2da.jpg)  
Figure 4: Qualitative comparison for IVF (top two rows) and MIF (bottom two rows).

Specifically, we adopt the commonly used intensity and gradient constraints:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { w a r m } } = \lambda _ { \mathrm { i n t } } \mathcal { L } _ { \mathrm { i n t } } + \lambda _ { \mathrm { g r a d } } \mathcal { L } _ { \mathrm { g r a d } } , } \end{array}\tag{17}
$$

where $\mathcal { L } _ { \mathrm { i n t } } = \| F - \operatorname* { m a x } ( A , B ) \| _ { 1 }$ and $\mathcal { L } _ { \mathrm { g r a d } } = \| \nabla F - \operatorname* { m a x } ( \nabla A , \nabla B ) \| _ { 1 }$ . Here, ∇ denotes the image gradient operator, while $\lambda _ { \mathrm { i n t } }$ and $\lambda _ { \mathrm { g r a d } }$ are hyperparameters. This loss is used only for fusion-network warm-up training and is removed once alternating optimization begins.

With this initialized fusion network, we then switch from conventional spatial-domain supervision to the proposed relation-constrained training scheme. The adapter constructs the supervision space, while the fusion network produces the fused images on which this supervision is applied. If either side is kept fixed throughout training, the learned supervision space and the fusion results may remain mismatched. During the adapter-update stage, the fusion network is frozen, and the current fused image F is used to optimize LFA with $\mathcal { L } _ { \mathrm { a d a p t e r } }$ . During the fusion-update stage, LFA is frozen, and the fusion network is optimized with $\mathcal { L } _ { \mathrm { r e l } }$ . This alternating process allows the adapter to refine the supervision space with current fusion results, while the fusion network progressively adapts to the updated supervision. Appendix A provides the full training procedure.

## 4 Experiments

Datasets and Backbone. For IVF, models are trained on MSRS [16] and tested on $\mathrm { M ^ { 3 } F D }$ [12], RoadScene [36], and MSRS [16]. For MIF, we use the Harvard Medical Dataset [44] for CT-MRI, MRI-PET, and MRI-SPECT fusion. Since our supervision paradigm is architecture-independent, we adopt CNN- [47], Mamba- [15], and Transformer-based [43] fusion backbones and provide their details in the Appendix K.

Training Details. For IVF, we train on MSRS with 1000 pairs. For MIF, we use 200 training pairs on Harvard Medical Dataset. For each task, the fusion net is first warmed up for 20 epochs with the conventional fusion loss. The adapter is then pretrained for 50 epochs with the self-supervised ranking objective. After initialization, alternating optimization is performed for 5 rounds. In each round, the adapter is updated for 30 epochs with the fusion network frozen, and the fusion network is updated for 20 epochs with the adapter frozen. During the fusion-update stage, only the proposed relation-constrained supervision is used. All these training stages use AdamW with an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , batch size 4, and no weight decay. The learning rate is halved every 10 epochs. All experiments are conducted on an NVIDIA RTX 4090 GPU. Parameter settings are in the Appendix K.

Comparison methods and metrics. We compare with EMMA [51], Text-DiFuse [44], Tc-MoA [54], ReFusion [3], C2RF [26], GIF-Net [7], SAGE [33], Omni-Fuse [45], CCF [5], Mask-Difuser [25], CLDyN [39], and ISFusion [30], using AG [9], EN [23], SD [22], MI [20], $Q _ { A B / F } [ 3 8 ] , Q _ { M }$ [29], and $Q _ { P }$ [48] as metrics.

Table 1: Quantitative comparison for IVF and MIF task. The best, second best, and third best are highlighted in red, blue, and green, respectively. “Ours(CNN/Mamba/Transformer)” denotes CNN, Mamba, and Transformer fusion backbones trained under our relation-constrained supervision.
<table><tr><td colspan="2">IVF Task</td><td colspan="6"></td><td colspan="6"></td><td colspan="6"></td><td colspan="6">MSRS</td></tr><tr><td>Methods</td><td></td><td>Pub./Year | AG ↑</td><td>EN ↑</td><td>SD ↑</td><td>MI↑</td><td></td><td>QAB/F ↑</td><td>QM ↑</td><td></td><td>QP↑|AG ↑</td><td>EN↑</td><td>SD ↑</td><td>MI↑</td><td>QAB/F↑</td><td></td><td>QM ↑</td><td></td><td>QP↑|AG↑</td><td></td><td>EN↑</td><td>SD ↑</td><td>MI↑</td><td></td><td>QAB/F↑</td><td>QM ↑</td><td>QP↑</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.432</td><td></td><td></td><td>5.871</td><td>57.569</td><td></td><td>3.771</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EMMA Text-Difuse</td><td>CVPR 24 NIPS 24</td><td>4.779 3.962</td><td>49.127 42.863</td><td></td><td>46.777 48.966</td><td>2.809 1.961</td><td>0.436</td><td></td><td>0.355 0.052</td><td></td><td></td><td></td><td>42.377</td><td></td><td>0.589</td><td>0.512 0.240</td><td>0.460 0.332</td><td>3.788 2.792</td><td></td><td>40.539 30.723</td><td>40.590 39.679</td><td>4.164 1.310</td><td></td><td>0.541</td><td>0.760</td><td>0.496 0.038</td></tr><tr><td>Tc-MoA</td><td>CVPR 24</td><td>4.192</td><td>44.883</td><td></td><td>42.610</td><td>1.043</td><td>0.140 0.413</td><td>0.288 0.556</td><td>0.403</td><td></td><td>1.861 5.180</td><td>20.580 54.156</td><td>40.568 39.030</td><td>2.054</td><td>0.066 0.651</td><td>0.891</td><td>0.513</td><td>3.618</td><td></td><td>38.414</td><td>42.335</td><td></td><td>3.493</td><td>0.118 0.630</td><td>0.288 1.090</td><td>0.483</td></tr><tr><td>ReFusion</td><td>IJCV 24</td><td>5.357</td><td></td><td>59.336</td><td>48.229</td><td>2.726</td><td>0.437</td><td>0.566</td><td>0.397</td><td></td><td>5.338</td><td>57.331</td><td>40.803</td><td>3.333 3.548</td><td>0.637</td><td>0.731</td><td>0.489</td><td></td><td>3.813</td><td>40.641</td><td>42.890</td><td></td><td>4.102</td><td>0.442</td><td>1.085</td><td>0.576</td></tr><tr><td>C2RF</td><td>IJCV 25</td><td>4.989</td><td>52.270</td><td></td><td>45.576</td><td>2.720</td><td>0.519</td><td>0.565</td><td>0.388</td><td></td><td>4.779</td><td>50.329</td><td>39.348</td><td>2.498</td><td>0.223</td><td>0.287</td><td>0.399</td><td></td><td>3.474</td><td>37.153</td><td>41.348</td><td></td><td>1.729</td><td>0.185</td><td>0.330</td><td>0.092</td></tr><tr><td>GIF-Net</td><td>CVPR 25</td><td>5.705</td><td>59.855</td><td></td><td>45.000</td><td>2.236</td><td>0.352</td><td>0.365</td><td>0.252</td><td></td><td>5.435</td><td>52.096</td><td>42.714</td><td>2.529</td><td>0.471</td><td>0.319</td><td>0.319</td><td></td><td>3.379</td><td>36.267</td><td>32.910</td><td></td><td>1.926</td><td>0.367</td><td>0.396</td><td>0.277</td></tr><tr><td>SAGE</td><td>CVPR 25</td><td>3.715</td><td>39.057</td><td></td><td>46.384</td><td>1.035</td><td>0.362</td><td>0.451</td><td>0.332</td><td></td><td>4.885</td><td>50.959</td><td>40.930</td><td>2.851</td><td>0.592</td><td>0.599</td><td>0.434</td><td></td><td>3.342</td><td>35.157</td><td>37.485</td><td></td><td>2.990</td><td>0.539</td><td>0.714</td><td>0.435</td></tr><tr><td>Omni-Fuse</td><td>TPAMI 25</td><td>3.206</td><td>35.212</td><td></td><td>44.830</td><td>2.187</td><td>0.303</td><td>0.396</td><td>0.267</td><td></td><td>4.514</td><td>49.067</td><td>41.029</td><td>3.208</td><td>0.430</td><td>0.362</td><td>0.277</td><td></td><td>3.091</td><td>34.123</td><td>40.043</td><td></td><td>2.664</td><td>0.361</td><td>0.412</td><td>0.217</td></tr><tr><td>CLDyN ISFusion</td><td>CVPR 26</td><td>4.918</td><td></td><td>51.612</td><td>40.644</td><td>1.827</td><td>0.417</td><td>0.475</td><td>0.329</td><td></td><td>5.824</td><td>57.312</td><td>35.254</td><td>3.235</td><td>0.689</td><td>0.756</td><td>0.520</td><td></td><td>3.568</td><td>37.882</td><td>39.377</td><td></td><td>2.111</td><td>0.661</td><td>0.867</td><td>0.484</td></tr><tr><td></td><td>CVPR 26</td><td></td><td>5.320</td><td>54.393</td><td>35.946</td><td>2.345</td><td>0.522</td><td>0.484</td><td>0.306</td><td></td><td>5.080</td><td>52.927</td><td>36.718</td><td>2.188</td><td>0.511</td><td>0.422</td><td>0.343</td><td></td><td>4.124</td><td>42.850</td><td>34.100</td><td>3.303</td><td></td><td>0.491</td><td>0.550</td><td>0.357</td></tr><tr><td>Ours(CNN)</td><td></td><td>5.701</td><td></td><td>59.876</td><td>49.673</td><td>2.962</td><td>0.541</td><td>0.604</td><td>0.411</td><td></td><td>5.478</td><td>57.921</td><td>42.315</td><td>4.238</td><td>0.664</td><td>1.041</td><td>0.529</td><td></td><td>3.842</td><td>40.982</td><td>43.517</td><td>4.301</td><td></td><td>0.651</td><td>1.337</td><td>0.509</td></tr><tr><td>Ours(Mamba) Ours(Transformer)</td><td></td><td></td><td>5.858</td><td>60.742</td><td>50.214</td><td>3.084</td><td>0.552</td><td>0.621</td><td>0.417</td><td></td><td>5.602</td><td>58.736</td><td>42.982</td><td>4.612</td><td>0.671</td><td>1.102</td><td>0.533</td><td>3.902</td><td></td><td>41.427</td><td>43.918</td><td>4.493</td><td></td><td>0.664</td><td>1.462</td><td>0.518</td></tr><tr><td colspan="2"></td><td></td><td>5.861</td><td>61.069</td><td>50.581</td><td>3.179</td><td>0.564</td><td>0.643</td><td></td><td>0.425</td><td>5.666</td><td>59.158</td><td>43.407</td><td>4.837</td><td>0.679</td><td>1.157</td><td>0.540</td><td></td><td>3.953</td><td>41.854</td><td>44.231</td><td>4.651</td><td></td><td>0.678</td><td>1.576</td><td>0.526</td></tr><tr><td></td><td colspan="2">MIF Task</td><td></td><td></td><td></td><td>CT-MRI</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>MRI-PET</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>MRI-SPECT</td><td></td><td></td><td></td></tr><tr><td>Methods</td><td>Pub./Year</td><td></td><td>|AG ↑</td><td>EN↑</td><td>SD ↑</td><td>MI↑</td><td>QAB/F↑</td><td>QM↑</td><td>QP↑|AG ↑</td><td></td><td></td><td>EN↑</td><td>SD ↑</td><td>MI↑</td><td>QAB/F↑</td><td>QM ↑</td><td></td><td>QP↑|AG↑</td><td></td><td>EN↑</td><td>SD ↑</td><td>MI↑</td><td></td><td>QAB/F↑</td><td>QM ↑</td><td>QP↑</td></tr><tr><td>EMMA</td><td></td><td>CVPR 24</td><td>7.317 76.268</td><td></td><td></td><td></td></table>

## 4.1 Comparison with SOTA.

Quantitative Comparison. Table 1 compares our method with recent SOTA methods on IVF and MIF tasks. “Ours(CNN/Mamba/Transformer)” denotes different fusion backbones trained under our relation-constrained supervision. All three variants achieve competitive or superior results across datasets, with the Transformer variant obtaining the best overall performance. These results indicate that relation-space supervision provides a more task-aligned optimization target than spatial-domain source approximation. Qualitative Comparison. Fig. 4 shows that existing methods often suffer from modality bias, either weakening infrared targets or visible textures in IVF, and either suppressing functional uptake or distorting anatomical boundaries in MIF. In contrast, our method preserves salient targets, structural details, and functional regions more faithfully, producing clearer and more balanced fusion results. Full qualitative comparisons are provided in Appendix F.

## 4.2 Analysis of Source-Preservation Update Conflicts

Following task-gradient conflict analysis in multi-task optimization [42, 11, 18, 24], we view infrared and visible source preservation as two MMIF analysis tasks. This analysis is used only for evaluation. For a fused result $F ^ { m }$ , we define the source-preservation objective for source s as:

$$
\begin{array} { r } { \mathcal { L } _ { s } ^ { \mathrm { s p } } = \Vert \boldsymbol { F } ^ { m } - \boldsymbol { I } _ { s } \Vert _ { 1 } + \left( \Vert \boldsymbol { D } _ { x } \boldsymbol { F } ^ { m } - \boldsymbol { D } _ { x } \boldsymbol { I } _ { s } \Vert _ { 1 } + \Vert \boldsymbol { D } _ { y } \boldsymbol { F } ^ { m } - \boldsymbol { D } _ { y } \boldsymbol { I } _ { s } \Vert _ { 1 } \right) , } \end{array}\tag{18}
$$

where $s \in \{ \mathrm { i r } , \mathrm { v i s } \} , D _ { x }$ and $D _ { y }$ are Sobel operators. The corresponding source-induced update is defined as:

$$
u _ { s } ^ { m } = - \frac { \partial \mathcal { L } _ { s } ^ { \mathrm { s p } } } { \partial F ^ { m } } .\tag{19}
$$

We partition the source-induced update maps into spatial patches of size $1 6 \times 1 6$ . For each patch p, the updates at the same spatial location are flattened into vectors $u _ { \mathrm { i r } , p } ^ { m }$ and $u _ { \mathrm { v i s } , p } ^ { m } .$ , respectively. Their directional agreement is measured by:

$$
\rho _ { p } ^ { m } = \cos \left( u _ { \mathrm { i r } , p } ^ { m } , u _ { \mathrm { v i s } , p } ^ { m } \right) .\tag{20}
$$

A negative $\rho _ { p } ^ { m }$ indicates that the two sources induce opposite local update directions at patch p.

Specifically, we evaluate conflicts on the high-disagreement patch set Ω, which is selected only from source-image intensity and gradient discrepancies and is shared by all methods. On Ω, we measure the conflict ratio (CR), conflict strength (CS), and update imbalance (UI) as:

$$
\mathrm { C R } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } \mathbb { I } \big ( \rho _ { p } ^ { m } < 0 \big ) , \quad \mathrm { C S } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } [ - \rho _ { p } ^ { m } ] _ { + } , \quad \mathrm { U I } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } \frac { \big | \| u _ { \mathrm { i r } , p } ^ { m } \| _ { 2 } - \| u _ { \mathrm { v i s } , p } ^ { m } \| _ { 2 } \big | } { \| u _ { \mathrm { i r } , p } ^ { m } \| _ { 2 } + \| u _ { \mathrm { v i s } , p } ^ { m } \| _ { 2 } + \epsilon } .\tag{21}
$$

Lower values indicate fewer source-preservation conflicts and better source-side update balance. Details of the construction of Ω and the three metrics are provided in Appendix G. As shown in Fig. 5 and Table 2, existing methods exhibit stronger conflict responses in high-disagreement regions,

![](images/74807175cf7ba498f3bb57045d7332a0c1e3cad25a6db9e37c9d13c4131ed344.jpg)  
Figure 5: Visualization of conflict map. Warmer colors denote higher source preservation update conflicts, while cooler denote lower conflicts.

Table 2: Quantitative results on source preservation update conflicts in highdisagreement regions on the M<sup>3</sup>FD dataset. CR, CS, and UI denote conflict ratio, conflict strength, and update imbalance, respectively.
<table><tr><td>Method</td><td>|CR (%) ↓ CS↓ UI↓</td></tr><tr><td>Text-DiFuse C2RF</td><td>27.25 0.806 0.4527 16.04 0.715 0.3379</td></tr><tr><td>ReFusion</td><td>20.91 0.784 0.3865</td></tr><tr><td>GIF-Net</td><td>18.36 0.692 0.3514</td></tr><tr><td>SAGE</td><td>15.28 0.641 0.3196</td></tr><tr><td>Omni-Fuse</td><td>17.63 0.667 0.3421</td></tr><tr><td>CLDyN</td><td>13.72 0.584 0.2968</td></tr><tr><td>Ours (CNN)</td><td>7.42 0.315 0.218</td></tr><tr><td>Ours (Mamba)</td><td>6.37 0.287 0.198</td></tr><tr><td>Ours (Transformer)</td><td>4.77 0.265 0.176</td></tr></table>

Table 3: Ablation study of the proposed method on the IVF RoadScene and MIF MRI-PET datasets. The best results are highlighted in bold. More ablation variants are provided in the appendix.
<table><tr><td rowspan="2">Group</td><td rowspan="2">ID</td><td rowspan="2">Description</td><td colspan="6">IVF on RoadScene</td><td colspan="8">MIF on MRI-PET</td></tr><tr><td>AG ↑</td><td>EN↑</td><td>SD↑</td><td>MI↑</td><td> $Q _ { A B / F }$  ↑</td><td> $Q _ { M } \uparrow$ </td><td>QP↑|AG↑</td><td></td><td>EN↑</td><td>SD↑</td><td>MI↑</td><td> $Q _ { A B / F }$ </td><td>↑  $Q _ { M }$ </td><td>↑QP↑</td></tr><tr><td rowspan="5">(a)</td><td>(1)</td><td>w/o LFA</td><td>5.247</td><td>58.932</td><td>47.286</td><td>2.861</td><td>0.506</td><td>0.589</td><td>0.372</td><td>8.721</td><td>94.863</td><td>72.418</td><td>2.321</td><td>0.552</td><td>0.196</td><td>0.414</td></tr><tr><td>(2)</td><td>w/o FAM</td><td>5.516</td><td>60.217</td><td>49.103</td><td>3.041</td><td>0.537</td><td>0.618</td><td>0.401</td><td>9.184</td><td>97.326</td><td>75.861</td><td>2.512</td><td>0.594</td><td>0.218</td><td>0.452</td></tr><tr><td>(3)</td><td>w/o RRM</td><td>5.438</td><td>59.846</td><td>48.624</td><td>2.982</td><td>0.529</td><td>0.611</td><td>0.394</td><td>9.062</td><td>96.814</td><td>75.104</td><td>2.463</td><td>0.586</td><td>0.213</td><td>0.444</td></tr><tr><td>(4)</td><td>w/o RCS</td><td>5.196</td><td>58.714</td><td>47.635</td><td>2.803</td><td>0.497</td><td>0.581</td><td>0.363</td><td>8.604</td><td>94.217</td><td>71.936</td><td>2.276</td><td>0.541</td><td>0.189</td><td>0.405</td></tr><tr><td>(1)</td><td></td><td>5.642</td><td>60.384</td><td>49.427</td><td>3.086</td><td>0.545</td><td>0.626</td><td>0.410</td><td>9.346</td><td>97.942</td><td>76.412</td><td>2.574</td><td>0.607</td><td>0.224</td><td>0.461</td></tr><tr><td rowspan="4">(b)</td><td>(2)</td><td>w/o Lshr wlo Lcmp</td><td>5.471</td><td>60.092</td><td>48.891</td><td>2.947</td><td>0.523</td><td>0.613</td><td>0.392</td><td>9.118</td><td>97.156</td><td>75.386</td><td>2.438</td><td>0.581</td><td>0.216</td><td>0.441</td></tr><tr><td>(3)</td><td>w/o Lcrd</td><td>5.573</td><td>60.263</td><td>49.018</td><td>3.018</td><td>0.532</td><td>0.607</td><td>0.386</td><td>9.241</td><td>97.624</td><td>75.927</td><td>2.496</td><td>0.592</td><td>0.211</td><td>0.433</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(1) (2)</td><td> $\mathrm { w } / \mathrm { o } \ \mathcal { L } _ { \mathrm { s s r } }$ </td><td>5.324 5.386</td><td>59.463</td><td>48.102 48.406</td><td>2.902 2.918</td><td>0.514</td><td>0.598</td><td>0.379</td><td>8.891</td><td>95.842</td><td>73.864</td><td>2.384</td><td>0.567</td><td>0.204</td><td>0.425</td></tr><tr><td colspan="2">(c) (3)</td><td>w/o Lsfr fixed adapter</td><td>5.452</td><td>59.721 59.984</td><td>48.713</td><td>2.971</td><td>0.517 0.526</td><td>0.602 0.609</td><td>0.381 0.389</td><td>8.976 9.087</td><td>96.214 96.936</td><td>74.293 75.021</td><td>2.406 2.451</td><td>0.573 0.583</td><td>0.207 0.212</td><td>0.429 0.438</td></tr><tr><td colspan="3">Full Model (Transformer)</td><td>5.861</td><td></td><td>61.069 50.581</td><td>3.179</td><td>0.564</td><td>0.643</td><td>0.425</td><td>9.690</td><td>99.031 78.072</td><td></td><td>2.667</td><td>0.627</td><td>0.235</td><td>0.481</td></tr></table>

whereas our method reduces CR, CS, and UI overall. These results suggest that RCS coordinates cross-modal fusion more stably and preserves useful source cues with fewer contradictory gradients.

## 4.3 Ablation Study.

We conduct ablation studies on IVF and MIF datasets to verify the effectiveness of our main designs.   
Key results are shown in Table 3, and more variants are provided in the Appendix I.

(a) Effect of LFA. LFA constructs the supervision space from frozen DINO and CLIP features. We evaluate its role through four variants: (1) removing LFA, (2) removing FAM, (3) removing RRM, and (4) replacing RCS with conventional source preservation. Results in Table 3 (a) show clear performance drops in all cases, especially when LFA or RCS is removed. This shows that pretrained features need task-oriented alignment and relation reasoning for effective fusion supervision.

(b) Effect of RCS. RCS defines fusion supervision through shared integration, complementary preservation, and conflict coordination. We examine its role by removing $\mathcal { L } _ { \mathrm { s h r } } , \mathcal { L } _ { \mathrm { c m p } } ,$ and $\mathcal { L } _ { \mathrm { c r d } }$ respectively. Results in Table 3 (b) show consistent performance drops in all cases. This shows that adaptive relation constraints are essential for task-aligned fusion supervision.

(c) Effect of Adapter Training. Adapter training makes the learned supervision space discriminative and compatible with fused images. We evaluate its role through three variants: (1) removing $\mathcal { L } _ { \mathrm { s s r } } .$ (2) removing $\mathcal { L } _ { \mathrm { s f r } }$ , and (3) fixing the adapter after pretraining. Results in Table 3 (c) show clear degradation across these variants. This confirms that the adapter should be pretrained with relation ranking and alternately refined with the fusion network.

## 4.4 Downstream Tasks

We further evaluate downstream perception using fused images. For IVF, all methods are tested with the same pretrained YOLOv12 [27] detector without fine-tuning; for MIF, medical fusion

![](images/b7fe1e87e75e4e8335bab3bed917d978290c683e76761cbb84caa0d5f565879f.jpg)  
Figure 6: Qualitative results on downstream tasks (Object detection and medical image segmentation).

Table 4: Quantitative results of various methods for object detection task on the M<sup>3</sup>FD [12].
<table><tr><td>Object Detection</td><td>|People</td><td>Car</td><td>Bus</td><td>Motorcycle</td><td>Lamp</td><td>Truck</td><td>mAP</td></tr><tr><td>Infrared</td><td>0.304</td><td>0.422</td><td>0.024</td><td>0.549</td><td>0.000</td><td>0.112</td><td>0.235</td></tr><tr><td>Visible</td><td>0.515</td><td>0.603</td><td>0.691</td><td>0.455</td><td>0.585</td><td>0.491</td><td>0.557</td></tr><tr><td>EMMA</td><td>0.450</td><td>0.617</td><td>0.457</td><td>0.469</td><td>0.465</td><td>0.517</td><td>0.496</td></tr><tr><td>Text-Difuse</td><td>0.547</td><td>0.614</td><td>0.781</td><td>0.565</td><td>0.632</td><td>0.545</td><td>0.614</td></tr><tr><td>Tc-MoA</td><td>0.550</td><td>0.687</td><td>0.757</td><td>0.459</td><td>0.565</td><td>0.317</td><td>0.556</td></tr><tr><td>ReFusion</td><td>0.489</td><td>0.663</td><td>0.522</td><td>0.538</td><td>0.597</td><td>0.663</td><td>0.579</td></tr><tr><td>GIF-Net</td><td>0.695</td><td>0.782</td><td>0.756</td><td>0.528</td><td>0.565</td><td>0.477</td><td>0.634</td></tr><tr><td>SAGE</td><td>0.511</td><td>0.760</td><td>0.671</td><td>0.485</td><td>0.542</td><td>0.475</td><td>0.574</td></tr><tr><td>OmniFuse</td><td>0.614</td><td>0.739</td><td>0.795</td><td>0.511</td><td>0.614</td><td>0.543</td><td>0.636</td></tr><tr><td>Mask-Difuser</td><td>0.628</td><td>0.756</td><td>0.802</td><td>0.536</td><td>0.641</td><td>0.579</td><td>0.657</td></tr><tr><td>CLDyN</td><td>0.646</td><td>0.773</td><td>0.764</td><td>0.552</td><td>0.658</td><td>0.621</td><td>0.669</td></tr><tr><td>ISFusion</td><td>0.665</td><td>0.784</td><td>0.736</td><td>0.574</td><td>0.672</td><td>0.684</td><td>0.686</td></tr><tr><td>Ours (CNN)</td><td>0.677</td><td>0.813</td><td>0.709</td><td>0.427</td><td>0.655</td><td>0.753</td><td>0.672</td></tr><tr><td>Ours (Mamba)</td><td>0.683</td><td>0.791</td><td>0.823</td><td>0.591</td><td>0.689</td><td>0.712</td><td>0.715</td></tr><tr><td>Ours (Transformer)</td><td>0.701</td><td>0.781</td><td>0.774</td><td>0.633</td><td>0.669</td><td>0.822</td><td>0.730</td></tr></table>

Table 5: Quantitative results of various methods for medical segmentation task on the BraTS2018 [10].
<table><tr><td>Case</td><td>T1CE</td><td>FLAIR</td><td>EMMA</td><td>Text-Difuse</td><td>Tc-MoA</td></tr><tr><td>IoU [55]</td><td>0.695</td><td>0.657</td><td>0.743</td><td>0.685</td><td>0.689</td></tr><tr><td>Dice [17]</td><td>0.807</td><td>0.770</td><td>0.841</td><td>0.751</td><td>0.829</td></tr><tr><td>Case</td><td>|ReFusion</td><td>C2RF</td><td>GIF-Net</td><td>CCF</td><td>Mask-Difuser</td></tr><tr><td>IoU</td><td>0.639</td><td>0.724</td><td>0.639</td><td>0.661</td><td>0.723</td></tr><tr><td>Dice</td><td>0.757</td><td>0.833</td><td>0.758</td><td>0.552</td><td>0.829</td></tr><tr><td>Case</td><td>CLDyN</td><td>ISFusion</td><td>Ours (CNN)</td><td>Ours (Mamba)</td><td>Ours (Transformer)</td></tr><tr><td>IoU</td><td>0.745</td><td>0.694</td><td>0.782</td><td>0.791</td><td>0.799</td></tr><tr><td>Dice</td><td>0.843</td><td>0.806</td><td>0.846</td><td>0.854</td><td>0.850</td></tr></table>

results are evaluated with UniverSeg [4]. As shown in Fig. 6 and Tables 4–5, our method achieves more reliable detection and segmentation results than competing methods. These gains indicate that relation-constrained supervision better preserves task-relevant cues and improves downstream utility.

## 5 Conclusion

In this paper, we present a relation-constrained supervision paradigm that moves MMIF supervision beyond the spatial domain. This paradigm learns a relation space from frozen pretrained representa tions and aligns supervision with the goal of fusion. Our adapter aligns heterogeneous DINO and CLIP features and infers relation parameters for shared integration, complementary preservation, and conflict coordination. With adapter-tailored contrastive ranking and alternating training, the proposed space is effectively optimized. Extensive experiments validate its effectiveness.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China [No.62401097, 62601484]; Fundamental Research Funds for Central Universities, Dalian Minzu University [No.0854- 53]; Liaoning Province Applied Basic Research Program [2026JH2/101300163]; Liaoning Province Science and Technology Joint Plan (2024JH2/102600113).

## References

[1] Y. Ai, H. Huang, X. Zhou, J. Wang, and R. He. Multimodal prompt perceiver: Empower adaptiveness generalizability and fidelity for all-in-one image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25432–25444, 2024.

[2] H. Bai, J. Zhang, Z. Zhao, Y. Wu, L. Deng, Y. Cui, T. Feng, and S. Xu. Task-driven image fusion with learnable fusion loss. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7457–7468, 2025.

[3] H. Bai, Z. Zhao, J. Zhang, Y. Wu, L. Deng, Y. Cui, B. Jiang, and S. Xu. Refusion: Learning image fusion from reconstruction with learnable loss via meta-learning. International Journal of Computer Vision, 133(5):2547–2567, 2025.

[4] V. I. Butoi, J. J. G. Ortiz, T. Ma, M. R. Sabuncu, J. Guttag, and A. V. Dalca. Universeg: Universal medical image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21438–21451, 2023.

[5] B. Cao, X. Xu, P. Zhu, Q. Wang, and Q. Hu. Conditional controllable image fusion. Advances in Neural Information Processing Systems, 37:120311–120335, 2024.

[6] Z. Cao, Y. Zhong, Z. Wang, and L.-J. Deng. Mmaif: Multi-task and multi-degradation all-in-one for image fusion with language guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11744–11754, 2025.

[7] C. Cheng, T. Xu, Z. Feng, X. Wu, Z. Tang, H. Li, Z. Zhang, S. Atito, M. Awais, and J. Kittler. One model for all: Low-level task interaction is a key to task-agnostic image fusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28102–28112, 2025.

[8] J. Cheng, D. Liang, and S. Tan. Transfer clip for generalizable image denoising. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 25974–25984, 2024.

[9] G. Cui, H. Feng, Z. Xu, Q. Li, and Y. Chen. Detail preserved fusion of visible and infrared images using regional saliency extraction and multi-scale image decomposition. Optics Communications, 341:199–209, 2015.

[10] A. Kori, M. Soni, B. Pranjal, M. Khened, V. Alex, and G. Krishnamurthi. Ensemble of fully convolutional neural network for brain tumor segmentation from magnetic resonance images. In International MICCAI Brainlesion Workshop, pages 485–496. Springer, 2018.

[11] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu. Conflict-averse gradient descent for multi-task learning. In Advances in Neural Information Processing Systems, 2021.

[12] J. Liu, X. Fan, Z. Huang, G. Wu, R. Liu, W. Zhong, and Z. Luo. Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5802–5811, 2022.

[13] J. Liu, G. Wu, Z. Liu, D. Wang, Z. Jiang, L. Ma, W. Zhong, X. Fan, and R. Liu. Infrared and visible image fusion: From data compatibility to task adaption. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(4):2349–2369, 2024.

[14] J. Liu, B. Zhang, Q. Mei, X. Li, Y. Zou, Z. Jiang, L. Ma, R. Liu, and X. Fan. Dcevo: Discriminative cross-dimensional evolutionary learning for infrared and visible image fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2226–2235, 2025.

[15] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, J. Jiao, and Y. Liu. Vmamba: Visual state space model. Advances in neural information processing systems, 37:103031–103063, 2024.

[16] J. Ma, L. Tang, F. Fan, J. Huang, X. Mei, and Y. Ma. Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer. IEEE/CAA Journal of Automatica Sinica, 9(7):1200–1217, 2022.

[17] F. Milletari, N. Navab, and S.-A. Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016.

[18] A. Navon, A. Shamsian, I. Achituve, H. Maron, K. Kawaguchi, G. Chechik, and E. Fetaya. Multi-task learning as a bargaining game. In International Conference on Machine Learning, 2022.

[19] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[20] G. Qu, D. Zhang, and P. Yan. Information measure for performance of image fusion. Electronics letters, 38(7):313–315, 2002.

[21] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[22] Y.-J. Rao. In-fibre bragg grating sensors. Measurement science and technology, 8(4):355–375, 1997.

[23] J. W. Roberts, J. A. Van Aardt, and F. B. Ahmed. Assessment of image fusion procedures using entropy, image quality, and multispectral classification. Journal of Applied Remote Sensing, 2(1):023522, 2008.

[24] D. Senushkin, N. Patakin, A. Kuznetsov, and A. Konushin. Independent component alignment for multi-task learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[25] L. Tang, C. Li, and J. Ma. Mask-difuser: A masked diffusion model for unified unsupervised image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[26] L. Tang, Q. Yan, X. Xiang, L. Fang, and J. Ma. C2rf: Bridging multi-modal image registration and fusion via commonality mining and contrastive learning. International journal of computer vision, 133(8):5262– 5280, 2025.

[27] Y. Tian, Q. Ye, and D. Doermann. Yolov12: Attention-centric real-time object detectors. arXiv preprint arXiv:2502.12524, 2025.

[28] H. Wang, H. Zhang, X. Yi, X. Xiang, L. Fang, and J. Ma. Terf: Text-driven and region-aware flexible visible and infrared image fusion. In Proceedings of the 32nd ACM international conference on multimedia, pages 935–944, 2024.

[29] P.-w. Wang and B. Liu. A novel image fusion metric based on multi-scale analysis. In 2008 9th international conference on signal processing, pages 965–968. IEEE, 2008.

[30] X. Wang, Z. Guan, W. Qian, C. Wang, and R. Ma. Multi-modal image fusion via intervention-stable feature learning. arXiv preprint arXiv:2603.23272, 2026.

[31] Z. Wang, J. Zhang, H. Song, M. Ge, J. Wang, and H. Duan. Highlight what you want: Weakly-supervised instance-level controllable infrared-visible image fusion. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 12637–12647, 2025.

[32] Z. Wang, L. Zhao, J. Zhang, R. Song, H. Song, J. Meng, and S. Wang. Multi-text guidance is important: Multi-modality image fusion via large generative vision-language model. International Journal of Computer Vision, 133(7):4646–4668, 2025.

[33] G. Wu, H. Liu, H. Fu, Y. Peng, J. Liu, X. Fan, and R. Liu. Every sam drop counts: Embracing semantic priors for multi-modality image fusion and beyond. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17882–17891, 2025.

[34] H. Xu, Y. Li, Y. Deng, J. Ma, and G. Liu. Deno-if: Unsupervised noisy visible and infrared image fusion method. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[35] H. Xu, J. Ma, J. Jiang, X. Guo, and H. Ling. U2fusion: A unified unsupervised image fusion network. IEEE transactions on pattern analysis and machine intelligence, 44(1):502–518, 2020.

[36] H. Xu, J. Ma, Z. Le, J. Jiang, and X. Guo. Fusiondn: A unified densely connected network for image fusion. In proceedings ofthe Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.

[37] X. Xu, S. Kong, T. Hu, Z. Liu, and H. Bao. Boosting image restoration via priors from pre-trained models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2900–2909, 2024.

[38] C. S. Xydeas and V. Petrovic. Objective image fusion performance measure. Electronics letters, 36(4):308– 309, 2000.

[39] Z. Yang, Y. Liu, J. Cheng, Z. Zhu, Y. Zhang, and H. Li. Customized fusion: A closed-loop dynamic network for adaptive multi-task-aware infrared-visible image fusion. arXiv preprint arXiv:2604.08924, 2026.

[40] X. Yi, L. Tang, H. Zhang, H. Xu, and J. Ma. Diff-if: Multi-modality image fusion via diffusion model with fusion knowledge prior. Information Fusion, 110:102450, 2024.

[41] X. Yi, H. Xu, H. Zhang, L. Tang, and J. Ma. Text-if: Leveraging semantic text guidance for degradationaware and interactive image fusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27026–27035, 2024.

[42] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, 2020.

[43] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5728–5739, 2022.

[44] H. Zhang, L. Cao, and J. Ma. Text-difuse: An interactive multi-modal image fusion framework based on text-modulated diffusion model. Advances in Neural Information Processing Systems, 37:39552–39572, 2024.

[45] H. Zhang, L. Cao, X. Zuo, Z. Shao, and J. Ma. Omnifuse: Composite degradation-robust image fusion with language-driven semantics. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[46] H. Zhang, X. Zuo, J. Jiang, C. Guo, and J. Ma. Mrfs: Mutually reinforcing image fusion and segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26974– 26983, 2024.

[47] Y. Zhang, Y. Liu, P. Sun, H. Yan, X. Zhao, and L. Zhang. Ifcnn: A general image fusion framework based on convolutional neural network. Information Fusion, 54:99–118, 2020.

[48] J. Zhao, R. Laganiere, and Z. Liu. Performance assessment of combinative pixel-level image fusion based on an absolute feature measurement. Int. J. Innov. Comput. Inf. Control, 3(6):1433–1447, 2007.

[49] W. Zhao, S. Xie, F. Zhao, Y. He, and H. Lu. Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13955–13965, 2023.

[50] Z. Zhao, H. Bai, J. Zhang, Y. Zhang, S. Xu, Z. Lin, R. Timofte, and L. Van Gool. Cddfuse: Correlationdriven dual-branch feature decomposition for multi-modality image fusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5906–5916, 2023.

[51] Z. Zhao, H. Bai, J. Zhang, Y. Zhang, K. Zhang, S. Xu, D. Chen, R. Timofte, and L. Van Gool. Equivariant multi-modality image fusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 25912–25921, 2024.

[52] Z. Zhao, L. Deng, H. Bai, Y. Cui, Z. Zhang, Y. Zhang, H. Qin, D. Chen, J. Zhang, P. Wang, et al. Image fusion via vision-language model. arXiv preprint arXiv:2402.02235, 2024.

[53] Z. Zhao, S. Xu, C. Zhang, J. Liu, P. Li, and J. Zhang. Didfuse: Deep image decomposition for infrared and visible image fusion. arXiv preprint arXiv:2003.09210, 2020.

[54] P. Zhu, Y. Sun, B. Cao, and Q. Hu. Task-customized mixture of adapters for general image fusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7099–7108, 2024.

[55] K. H. Zou, S. K. Warfield, A. Bharatha, C. M. Tempany, M. R. Kaus, S. J. Haker, W. M. Wells III, F. A. Jolesz, and R. Kikinis. Statistical validation of image segmentation quality based on a spatial overlap index1: scientific reports. Academic radiology, 11(2):178–189, 2004.

## A Pseudocode of the Training Procedure

We provide the pseudocode of the proposed training procedure in Algorithm 1 and Algorithm 2. Algorithm 1 summarizes the full optimization pipeline. It first warms up the fusion network with a conventional fusion loss to obtain initial fused images. This warm-up is only used to provide preliminary fused samples for adapter learning. After that, the fusion network is fixed and the adapter is initialized with the self-supervised ranking objective. The later training follows an alternating scheme. In each round, LFA is first updated for several epochs using the current fused images, and the fusion network is then updated for several epochs using only the relation-constrained supervision. The conventional fusion loss is not used in this alternating stage. The shared token encoder $G _ { \mathrm { s h r } }$ is pretrained separately and kept fixed throughout the following optimization procedure.

Algorithm 1 Training procedure of the proposed framework   
Input: Paired source dataset $\mathcal { D } = \{ ( A , B ) \}$   
Modules: Frozen feature extractor Ψ, pretrained shared token encoder $G _ { \mathrm { s h r } } ,$ fusion network $\mathcal { F } _ { \Theta }$   
learnable feature adapter ${ \cal A } _ { \Phi } .$   
1: Initialize parameters Θ and Φ.   
2: Keep Ψ and $G _ { \mathrm { s h r } }$ frozen throughout training.   
// Warm-upfusion network to obtain initialfused images   
3: for epoch = 1 to $E _ { \mathrm { w a r m } }$ do   
4: for each mini-batch $( A , B ) \subset { \mathcal { D } }$ do   
5: $F = { \mathcal { F } } _ { \Theta } ( A , B )$   
6: Update Θ by back-propagating $\mathcal { L } _ { \mathrm { w a r m } } ( F ; A , B )$   
7: end for   
8: end for   
// Initialize adapter withfixedfusion network   
9: Freeze Θ.   
10: for epoch = 1 to $E _ { \mathrm { a d a } }$ do   
11: for each mini-batch $( A , B ) \subset { \mathcal { D } }$ do   
12: Sample unpaired sources $A ^ { - }$ and $B ^ { - }$   
13: $F = \mathcal { F } _ { \Theta } ( \mathring { A } , B )$   
14: $\mathcal { L } _ { \mathrm { a d a p t e r } } \dot { = } \dot { \mathrm { A D A } }$ PTEROBJECTIVE $( A , B , A ^ { - } , B ^ { - } , F ; \Phi , G _ { \mathrm { s h r } } )$   
15: Update Φ by back-propagating L<sub>adapter</sub>   
16: end for   
17: end for   
//Alternating optimization   
18: for round = 1 to $K$ do   
//Adapter update: train LFAfor several epochs with currentfused images   
19: Freeze Θ and unfreeze Φ.   
20: for epoch = 1 to $E _ { \mathrm { a d a } } ^ { \mathrm { a l t } }$ do   
21: for each mini-batch $( A , B ) \subset { \mathcal { D } }$ do   
22: Sample unpaired sources $A ^ { - }$ and $B ^ { - }$   
23: $F = \mathcal { F } _ { \Theta } ( \mathring { A } , B )$   
24: $\mathcal { L } _ { \mathrm { a d a p t e r } } = \mathrm { A D A P T E R O B J E C T I V E } ( A , B , A ^ { - } , B ^ { - } , F ; \Phi , G _ { \mathrm { s h r } } )$   
25: Update Φ by back-propagating $\mathcal { L } _ { \mathrm { a } }$ dapter   
26: end for   
27: end for   
// Fusion update: train FusionNetfor several epochs only with RCS   
28: Freeze Φ and unfreeze Θ.   
29: for epoch = 1 to $E _ { \mathrm { f u s } } ^ { \mathrm { a l t } }$ do   
30: for each mini-batch $( A , B ) \subset { \mathcal { D } }$ do   
31: $F = { \mathcal { F } } _ { \Theta } ( A , B )$   
32: $\mathcal { L } _ { \mathrm { r e l } } = \mathrm { R C S } ( \dot { A } , B , F ; \Phi , G _ { \mathrm { s h r } } )$   
33: Update Θ by back-propagating $\mathcal { L } _ { \mathrm { r e l } }$   
34: end for   
35: end for   
36: end for   
Output: Trained fusion network $\mathcal { F } _ { \Theta }$ and adapter $A _ { \Phi }$

Algorithm 2 details the adapter objective used in both adapter initialization and adapter-update stages. Given a matched source pair, two unpaired sources, and the current fused image, the adapter first computes source-pair scores for the matched and mismatched pairs. These scores are used to form the source–source ranking loss $\mathcal { L } _ { \mathrm { s s r } }$ . The fused image is then encoded by feeding the self-pair $( F , F )$ into the same pair-conditioned adapter, producing the fused representation $z _ { F } .$ . The pretrained shared token encoder $G _ { \mathrm { s h r } }$ further extracts $c _ { A } , c _ { B }$ , and $c _ { F }$ , from which the corresponding residuals $u _ { A } , u _ { B }$ and $u _ { F }$ are constructed. Based on these shared and non-shared representations, the adapter computes one matched source–fused score $\tilde { s } ^ { + }$ and two mismatched source–fused scores $\tilde { s } _ { 1 } ^ { - }$ and $\tilde { s } _ { 2 } ^ { - }$ . These scores are used to form the source–fused ranking loss $\mathcal { L } _ { \mathrm { s f r } }$ , which encourages the fused image to be more compatible with its matched source pair than with mismatched source pairs. The final adapter objective is the sum of these two ranking losses. This design trains the adapter to distinguish matched source relations and to remain reliable when fused images enter the supervision space.

Algorithm 2 ADAPTEROBJECTIVE $( A , B , A ^ { - } , B ^ { - } , F ; \Phi , G _ { \mathrm { s h r } } )$   
Input: Matched source pair $( A , B )$ , unpaired sources $A ^ { - }$ and $B ^ { - } ,$ fused image $F .$   
Modules: Learnable feature adapter $\mathcal { A } _ { \Phi }$ and frozen shared token encoder $G _ { \mathrm { s h r } }$   
Margins: $\delta _ { s } , \delta _ { f }$   
1: Extract $\Psi ( \bar { X } ) \stackrel { , } { = } \{ f _ { D } ^ { 4 } ( X ) , f _ { C } ^ { 6 } ( X ) , f _ { D } ^ { 8 } ( X ) \} \mathrm { f o r } X \in \{ A , B , A ^ { - } , B ^ { - } , F \} .$   
// Source–source ranking   
2: ${ \mathcal { A } } _ { \Phi } ( \Psi ( A ) , \Psi ( B ) ) \to s ^ { \bar { + } } .$   
3: $\mathcal { A } _ { \Phi } ( \Psi ( A ) , \Psi ( B ^ { - } ) )  s _ { 1 } ^ { - } .$   
4: $\mathcal { A } _ { \Phi } ( \Psi ( A ^ { - } ) , \Psi ( B ) )  s _ { 2 } ^ { - } .$   
5: Compute source–source ranking loss:   
$\mathcal { L } _ { \mathrm { s s r } } = \frac { 1 } { 2 } \Big ( \operatorname* { m a x } ( 0 , \delta _ { s } - s ^ { + } + s _ { 1 } ^ { - } ) + \operatorname* { m a x } ( 0 , \delta _ { s } - s ^ { + } + s _ { 2 } ^ { - } ) \Big ) .$ (22)   
// Matched source pair should score higher than mismatched source pairs   
// Source–fused ranking   
6: Encode the fused self-pair: $\mathcal { A } _ { \Phi } ( \Psi ( F ) , \Psi ( F ) ) \to z _ { F } .$   
7: Extract the fused shared representation and residual:   
$c _ { F } = G _ { \mathrm { s h r } } ( z _ { F } ) , \qquad u _ { F } = z _ { F } - c _ { F } .$   
8: For each source pair above, apply $G _ { \mathrm { s h r } }$ to the corresponding pair-conditioned source representa  
tions:   
$c _ { A } = G _ { \mathrm { s h r } } ( z _ { A } ) , \qquad c _ { B } = G _ { \mathrm { s h r } } ( z _ { B } ) ,$   
and construct the corresponding residuals:   
$u _ { A } = z _ { A } - c _ { A } , \qquad u _ { B } = z _ { B } - c _ { B } .$   
9: Compute the matched source–fused score $\tilde { s } ^ { + }$ from $( c _ { A } , c _ { B } , c _ { F } , u _ { A } , u _ { B } , u _ { F } , m _ { A , B } , d _ { A , B } , r _ { A , B } )$   
10: Compute the mismatched source–fused scores $\tilde { s } _ { 1 } ^ { - }$ and $\tilde { s } _ { 2 } ^ { - }$ in the same way using the corresponding   
mismatched source pairs.   
11: Compute source–fused ranking loss:   
$\mathcal { L } _ { \mathrm { s f r } } = \frac { 1 } { 2 } \Big ( \operatorname* { m a x } ( 0 , \delta _ { f } - \tilde { s } ^ { + } + \tilde { s } _ { 1 } ^ { - } ) + \operatorname* { m a x } ( 0 , \delta _ { f } - \tilde { s } ^ { + } + \tilde { s } _ { 2 } ^ { - } ) \Big ) .$ (23)   
// Fused image should score higher with its matched source pair than with mismatched source   
pairs   
12: Compute adapter objective:   
${ \mathcal { L } } _ { \mathrm { a d a p t e r } } = { \mathcal { L } } _ { \mathrm { s s r } } + { \mathcal { L } } _ { \mathrm { s f r } } .$ (24)   
Output: $\mathcal { L } _ { \mathrm { a d a p t e r } } .$

## B Problem Statement and Modeling

Given two source images A and B, the goal of MMIF is to learn a fusion network $\mathcal { F }$ that generates a fused image: $F = \mathcal { F } ( \tilde { A } , B ; \Theta )$ , where Θ denotes the parameters of the fusion network. Under this notation, most existing multimodal image fusion methods follow a spatial source-image supervision

paradigm, which directly uses the source images as $\mathrm { G T }$ to compute loss, defined as:

$$
\arg \operatorname* { m i n } _ { \Theta } \mathcal { L } _ { \mathrm { s p a } } ( F ; A , B ) .\tag{25}
$$

Here, $\mathcal { L } _ { \mathrm { s p a } }$ denotes the losses composed of spatial-domain criteria such as intensity, gradient, and structural similarity. However, A and B are carriers of information, rather than natural ground truth for the fused image. They typically contain shared information, modality-specific information, and potential conflicts simultaneously. Therefore, directly using spatial-domain source images as GT often drives fusion learning toward pixel-level compromise or modality bias, rather than faithfully modeling the integration objective of MMIF. To address this limitation, we argue that an ideal supervisory space for MMIF should align with the fundamental objective of fusion. Since MMIF aims to integrate shared information, preserve complementary information, and handle potential cross-modal conflicts, we construct the proposed supervisory space from these three aspects. In other words, the proposed fusion supervision is defined not by direct spatial-domain approximation to source images, but by whether the fused result satisfies the fundamental requirements of MMIF in a knowledge-driven relational space. Under this formulation, fusion supervision is defined as:

$$
\arg \operatorname* { m i n } _ { \Theta } \underbrace { \mathcal { L } _ { \mathrm { r e l } } } _ { \mathcal { L } _ { \mathrm { s h r } } , \mathcal { L } _ { \mathrm { c m p } } , \mathcal { L } _ { \mathrm { c r d } } } \biggl ( \underbrace { \mathcal { A } \bigl ( \Psi ( A ) , \Psi ( B ) , \Psi ( F ) ; \Phi \bigr ) } _ { \hat { A } , \hat { B } , \hat { F } , m , d , r } \biggr ) .\tag{26}
$$

Here, Φ denotes the parameters of the learnable feature adapter ${ \mathcal { A } } ,$ which outputs aligned features $z _ { A }$ $z _ { B }$ , and $z _ { F }$ in the unified supervision space, along with three relation parameters m, d, and r. Built on these outputs, $\mathcal { L } _ { \mathrm { r e l } }$ denotes the proposed relation-constrained supervision with three components: the shared loss ${ \mathcal { L } } _ { \mathrm { s h r } } ,$ the complementary loss $\mathcal { L } _ { \mathrm { c m p } } .$ , and the coordination loss $\mathcal { L } _ { \mathrm { c r d } }$

## C Why Fusion Supervision Requires Sharedness, Complementarity, and Coordination?

A proper supervision space for MMIF should reflect the objective of fusion rather than the appearance of any single source image. Given a source pair $( A , B )$ , the fused image is expected to integrate information that is jointly supported by both sources, retain useful information that appears mainly in one source, and avoid unstable updates caused by cross-modal conflicts. These requirements naturally lead to three relations: sharedness, complementarity, and coordination.

Sharedness. Sharedness describes the degree to which two sources provide consistent information at a local position. When the two sources contain the same structure or semantic content, the fused representation should not be biased toward either source. Instead, it should preserve their common information. Without modeling sharedness, a supervision loss may treat the two source images as two independent targets and force the fusion model to make a pixel-level compromise. This may weaken common structures or introduce modality bias. Therefore, sharedness is needed to identify where the fused representation should follow a common source reference.

Complementarity. Complementarity describes useful non-shared information provided by different sources. In MMIF, some important cues are visible only in one modality, such as thermal targets in infrared images or fine textures in visible images. A supervision space that only emphasizes shared information would push the fused representation toward a common center and may suppress modality-specific cues. Therefore, complementarity is needed to guide the fused representation to preserve non-shared but useful information.

Coordination. Coordination is required because not all source differences are useful complementarity. Some regions contain strong cross-modal disagreement, noise, or inconsistent responses. If the supervision always forces the fused representation to follow one source or preserve both sources without restriction, the fusion result may be driven excessively toward one modality. This can introduce artifacts or unstable representations. Coordination defines an adaptive tolerance range for the fused offset. It allows useful complementary deviations while suppressing excessive deviations caused by conflict.

Together, the three relations define different but connected aspects of fusion supervision. Sharedness determines where the fused representation should align with common source content. Complementarity determines how non-shared source cues should be preserved. Coordination determines how far the fused representation can deviate under conflict. Thus, the proposed supervision space is not defined by direct spatial approximation to source images, but by whether the fused representation satisfies these three requirements induced by source relations.

## D Additional Analysis of the Relation-Constrained Supervision

This appendix provides a detailed explanation of how the relation parameters affect the supervision structure, the relation-constrained losses, the relatedness scores, and the training objective. The goal is to clarify how the proposed supervision space responds to different source relations. For compactness, $m _ { A , B } , d _ { A , B } .$ , and $r _ { A , B }$ denote the pair-conditioned relation maps without explicitly indexing individual tokens.

## D.1 Effects of Relation Parameters on Supervision

Sharedness $m _ { A , B } .$ . The sharedness parameter $m _ { A , B }$ controls how strongly the information of the source pair should be treated as shared. Unlike directly constructing a shared target from the source representations, the pretrained shared token encoder independently extracts:

$$
c _ { A } = G _ { \mathrm { s h r } } ( z _ { A } ) , \qquad c _ { B } = G _ { \mathrm { s h r } } ( z _ { B } ) , \qquad c _ { F } = G _ { \mathrm { s h r } } ( z _ { F } ) .\tag{27}
$$

The effect of $m _ { A , B }$ therefore appears in the strength of the shared constraint:

$$
\mathcal { L } _ { \mathrm { s h r } } = \mathcal { D } _ { \mathrm { c o s } } ( c _ { F } , c _ { A } ) + \mathcal { D } _ { \mathrm { c o s } } ( c _ { F } , c _ { B } ) .\tag{28}
$$

Since the shared components are explicitly extracted by $G _ { \mathrm { s h r } } , \mathcal { L } _ { \mathrm { s h r } }$ directly enforces consistency between $c _ { F }$ and the two source shared representations without additional weighting by $m _ { A , B }$ . Reducing this loss requires $c _ { F }$ to remain consistent with both $c _ { A }$ and $c _ { B }$ , thereby preserving information commonly supported by the two sources.

Dominance $d _ { A , B }$ . The dominance parameter $d _ { A , B }$ controls the relative contributions of the two source residuals in non-shared regions. After removing the corresponding shared representations,

$$
u _ { A } = z _ { A } - c _ { A } , ~ u _ { B } = z _ { B } - c _ { B } , ~ u _ { F } = z _ { F } - c _ { F } ,\tag{29}
$$

the complementary loss is defined as:

$$
\mathcal { L } _ { \mathrm { c m p } } = \left( 1 - m _ { A , B } \right) \left[ d _ { A , B } \mathcal { D } _ { \mathrm { c o s } } ( u _ { F } , u _ { A } ) + ( 1 - d _ { A , B } ) \mathcal { D } _ { \mathrm { c o s } } ( u _ { F } , u _ { B } ) \right] .\tag{30}
$$

The sharedness $m _ { A , B }$ controls the overall strength of the complementary constraint through the factor $1 - m _ { A , B } . \ \mathbf { A }$ larger $m _ { A , B }$ suppresses the contribution of non-shared information, whereas a smaller $m _ { A , B }$ increases the emphasis on complementary preservation. Within this constraint, $d _ { A , B }$ further determines the relative contribution of the two source residuals. When $d _ { A , B }$ is close to 1, greater emphasis is placed on preserving the non-shared information from source A. When $d _ { A , B }$ is close to $0 ,$ the constraint places greater emphasis on source $B$ . When $d _ { A , B }$ is close to 0.5, the two source residuals contribute more evenly. Thus, $m _ { A , B }$ controls the overall strength of complementary supervision, while $d _ { A , B }$ controls the relative contribution of the two sources without directly mixing their residuals into a single target. In addition, both parameters contribute to the relation-aware weight used for adapter ranking: larger $m _ { A , B }$ indicates stronger shared relations, while larger $| 2 d _ { A , B } - 1 |$ | indicates clearer source dominance, so these regions provide stronger relation evidence during ranking.

Coordination radius $r _ { A , B } .$ . The coordination radius $r _ { A , B }$ controls the allowed magnitude of the fused residual. Its effect appears in the coordination loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c r d } } = \operatorname* { m a x } ( 0 , \| u _ { F } \| _ { 2 } - r _ { A , B } ) . } \end{array}\tag{31}
$$

When $r _ { A , B }$ is small, the magnitude of the fused residual is more tightly constrained, limiting excessive deviation caused by cross-modal conflict. When $r _ { A , B }$ is large, the loss becomes more tolerant and allows a wider residual range, preserving flexibility in regions where stronger non-shared information needs to be retained. However, a large $r _ { A , B }$ also indicates that a wider coordination range is required and the local relation is less stable. Therefore, the relatedness score reduces the contribution of such regions through the negative term $- \lambda _ { r } r _ { A , B }$ in the relation-aware weight. In this way, $r _ { A , B }$ relaxes overly strict supervision in difficult regions while preventing unstable relations from dominating adapter ranking.

## E Why Self-Supervised Contrastive Learning Works for LFA?

LFA must learn pair-conditioned representations and relation parameters before it can supervise the fusion network. Since MMIF has no ground-truth fused image, these outputs cannot be trained with direct target labels. We instead use the pairing structure of source images as a self-supervised signal. For a matched source pair $( A , B )$ , the two images are captured from the same scene and share objects, structures, and semantic layout. For mismatched pairs such as $( A , B ^ { - } )$ and $( A ^ { - } , B )$ , this scene-level correspondence is broken. This provides a natural contrastive signal for adapter learning.

## E.1 Source-Side Contrastive Signal

The source-side signal trains LFA to distinguish matched source pairs from mismatched ones. Instead of requiring an absolute ground-truth relation score, it only requires the matched pair to have a higher relatedness score:

$$
s ^ { + } > s _ { 1 } ^ { - } , \qquad s ^ { + } > s _ { 2 } ^ { - } ,\tag{32}
$$

where $s ^ { + }$ denotes the score of the matched pair $( A , B )$ , while $s _ { 1 } ^ { - }$ and $s _ { 2 } ^ { - }$ denote the scores of $( A , B ^ { - } )$ and $( A ^ { - } , B )$ , respectively. This relative constraint is suitable for LFA because the adapter is expected to learn a relation space, rather than predict a fixed label for each pair. The corresponding source–side ranking loss is defined as:

$$
\mathcal { L } _ { \mathrm { s s r } } = \frac { 1 } { 2 } \Big ( \operatorname* { m a x } ( 0 , \delta _ { s } - s ^ { + } + s _ { 1 } ^ { - } ) + \operatorname* { m a x } ( 0 , \delta _ { s } - s ^ { + } + s _ { 2 } ^ { - } ) \Big ) .\tag{33}
$$

The margin $\delta _ { s }$ prevents weak separation. Thus, LFA cannot satisfy the objective by assigning similar scores to all pairs. It must learn representations and relation parameters that separate matched and mismatched source relations.

## E.2 Coupled Training of Representations and Relation Parameters

The relatedness score is designed to depend on both types of LFA outputs. For a matched source pair $( A , B )$ , the score is defined as:

$$
\begin{array} { r } { s ^ { + } = \frac { \left. w _ { A , B } , \cos \left( z _ { A } , z _ { B } \right) \right. } { \left| \left| w _ { A , B } \right| \right| _ { 1 } + \epsilon } , \quad w _ { A , B } = \lambda _ { m } m _ { A , B } + \lambda _ { d } | 2 d _ { A , B } - 1 | - \lambda _ { r } r _ { A , B } . } \end{array}\tag{34}
$$

Here, cos $( z _ { A } , z _ { B } )$ denotes token-wise cosine similarity between the two pair-conditioned representations, while $w _ { A , B }$ determines how much each token contributes to the final score. The design of $w _ { A , B }$ follows the roles of the three relation parameters: a larger $m _ { A , B }$ gives higher weight to shared regions, a larger $| 2 d _ { A , B } - 1 |$ gives higher weight to regions with clearer dominance, and a larger $r _ { A , B }$ reduces the weight of regions that require a wider coordination range.

This design couples representation learning and relation estimation within the same ranking objective. $\operatorname { I f } z _ { A }$ and $z _ { B }$ are not discriminative, matched and mismatched pairs cannot be separated. If $m _ { A , B } ,$ $d _ { A , B }$ , and $r _ { A , B }$ do not identify reliable relation evidence, unstable regions may dominate the score. Therefore, the ranking objective jointly calibrates the pair-conditioned representations and relation parameters through the same score function.

## E.3 Source-Fused Contrastive Signal

Source-side ranking alone is insufficient because LFA is later used to supervise fused images. If the adapter only learns source–source relations, the learned supervision space may not remain reliable when fused images are introduced. We therefore include the current fused image F in adapter training and require its matched source–fused relation to score higher than the corresponding mismatched relations:

$$
\tilde { s } ^ { + } > \tilde { s } _ { 1 } ^ { - } , \qquad \tilde { s } ^ { + } > \tilde { s } _ { 2 } ^ { - } .\tag{35}
$$

Here, $\tilde { s } ^ { + }$ measures the compatibility between the fused representation and its matched source pair $( A , B )$ , while $\tilde { s } _ { 1 } ^ { - }$ and $ { \tilde { s } } _ { 2 } ^ { - }$ are computed using the mismatched pairs $( A , B ^ { - } )$ and $( A ^ { - } , B )$ , respectively.

The source–fused score follows the same shared and complementary relations used in the relationconstrained supervision. The pretrained shared token encoder first extracts:

$$
\begin{array} { r } { c _ { A } = G _ { \mathrm { s h r } } ( z _ { A } ) , \qquad c _ { B } = G _ { \mathrm { s h r } } ( z _ { B } ) , \qquad c _ { F } = G _ { \mathrm { s h r } } ( z _ { F } ) , } \end{array}\tag{36}
$$

and the corresponding residuals are:

$$
u _ { A } = z _ { A } - c _ { A } , ~ u _ { B } = z _ { B } - c _ { B } , ~ u _ { F } = z _ { F } - c _ { F } .\tag{37}
$$

We then define the token-wise shared and complementary relation scores as:

$$
\rho ^ { s } = \cos ( c _ { F } , c _ { A } ) + \cos ( c _ { F } , c _ { B } ) , \qquad \rho ^ { c } = d _ { A , B } \cos ( u _ { F } , u _ { A } ) + ( 1 - d _ { A , B } ) \cos ( u _ { F } , u _ { B } ) .\tag{38}
$$

The matched source–fused relatedness score is defined as:

$$
\tilde { s } ^ { + } = \frac { \langle w _ { A , B } , m _ { A , B } \rho ^ { s } + ( 1 - m _ { A , B } ) \rho ^ { c } \rangle } { \| w _ { A , B } \| _ { 1 } + \epsilon } .\tag{39}
$$

Here, $\rho ^ { s }$ measures the consistency between the fused and source shared components, while $\rho ^ { c }$ measures the preservation of source-specific residuals according to their predicted dominance. The mismatched scores $\tilde { s } _ { 1 } ^ { - }$ and $\tilde { s } _ { 2 } ^ { - }$ are computed in the same way after replacing the corresponding source and using its pair-conditioned representations and relation parameters. Thus, source–fused ranking does not simply add the fused image as an extra sample. It evaluates the fused representation using the same shared and complementary relations that are later used for fusion supervision.

The corresponding source–fused ranking loss is defined as:

$$
\mathcal { L } _ { \mathrm { s f r } } = \frac { 1 } { 2 } \Big ( \operatorname* { m a x } ( 0 , \delta _ { f } - \tilde { s } ^ { + } + \tilde { s } _ { 1 } ^ { - } ) + \operatorname* { m a x } ( 0 , \delta _ { f } - \tilde { s } ^ { + } + \tilde { s } _ { 2 } ^ { - } ) \Big ) .\tag{40}
$$

The margin $\delta _ { f }$ plays the same role as $\delta _ { s } ,$ , but is applied to source–fused ranking. The full adapter objective is:

$$
{ \mathcal { L } } _ { \mathrm { a d a p t e r } } = { \mathcal { L } } _ { \mathrm { s s r } } + { \mathcal { L } } _ { \mathrm { s f r } } .\tag{41}
$$

## E.4 Sufficient Pretraining and Alternating Optimization

The adapter receives supervision from both source-pair discrimination and source–fused compatibility. For each matched source pair, the source–side ranking loss compares it with two mismatched pairs. This trains LFA to identify reliable source relations. The source–fused ranking loss then introduces the current fused image into adapter learning and requires its matched source relation to score higher than the two mismatched relations. Since these scores depend on the pair-conditioned representations and relation parameters, the two objectives jointly update the representation space and relation estimation. The pretrained shared token encoder $G _ { \mathrm { s h r } }$ remains fixed and provides the shared and residual representations used in source–fused scoring.

The warm-up fusion network provides the initial fused images required by source–fused ranking. These fused images do not need to be optimal. They only provide a starting distribution for adapting LFA from source–source relations to source–fused relations. After warm-up, LFA is trained with the fusion network fixed. The current fused image $F$ is generated by the fixed fusion network and is used in $\mathcal { L } _ { \mathrm { s f r } }$

The following training proceeds alternately. During the adapter-update stage, the fusion network is frozen, and LFA is updated with $\mathcal { L } _ { \mathrm { a d a p t e r } }$ using the current fused images. During the fusion-update stage, LFA is frozen, and the fusion network is optimized only with $\bar { \mathcal { L } } _ { \mathrm { r e l } }$ . No conventional fusion loss is used in this stage. This creates a closed training loop: improved fused images provide more informative source–fused samples for LFA, and an improved LFA provides a more reliable supervision space for the fusion network.

## F Full Qualitative Comparisons

We provide full qualitative comparisons for both MIF and IVF tasks. For MIF, the results are reported on CT-MRI [44], MRI-PET [44], and MRI-SPECT [44] fusion. For IVF, the results are reported on M<sup>3</sup>FD [12], RoadScene [36], and MSRS [16]. Since our supervision paradigm is independent of the fusion architecture, we include CNN-, Mamba-, and Transformer-based variants in all comparisons.

As shown in Fig. 8, our method produces visually reliable medical fusion results across all three MIF settings. In CT-MRI fusion, our variants preserve clear anatomical structures while maintaining soft-tissue contrast from MRI. In MRI-PET and MRI-SPECT fusion, our results retain functional color information and avoid excessive color bleeding or structural distortion. These results show that the proposed supervision can guide different backbones to preserve complementary medical information in a stable manner.

Infrared Visible Image Fusion  
![](images/8295bd5d77d72e4077a71d8d6a20c9735c9c3914d25726c870520fb80a409bd8.jpg)  
Figure 7: Full qualitative comparison for IVF task on $\mathbf { M } ^ { \mathrm { 3 } } \mathbf { F D }$ , RoadScene and MSRS.

Fig. 7 further shows the full qualitative comparison on IVF datasets. Existing methods may suffer from weak thermal targets, over-smoothed textures, or insufficient visible-detail preservation in challenging night and road scenes. In contrast, our CNN, Mamba, and Transformer variants consistently preserve salient infrared targets while keeping visible structures, such as vehicles, pedestrians, roads, and background textures. This indicates that the proposed relation-constrained supervision is not tied to a specific backbone and can provide effective guidance across different fusion architectures.

## G Details of Source-Preservation Update Conflict Analysis

We provide the detailed procedure for the source-preservation update conflict analysis used in Table 2. Following task-gradient conflict analysis in multi-task optimization [42, 11, 18, 24], we use the update directions induced by two source-preservation objectives to examine whether a fused result can accommodate infrared and visible information without strong local conflict. This analysis is only used for evaluation and is not used to train any model.

For a fused result $F ^ { m }$ produced by method $m ,$ , we define the source-preservation objective for source $s \in \{ i r , v i s \}$ as:

$$
\mathcal { L } _ { s } ^ { s p } = \| F ^ { m } - I _ { s } \| _ { 1 } + \lambda _ { g } \left( \| D _ { x } F ^ { m } - D _ { x } I _ { s } \| _ { 1 } + \| D _ { y } F ^ { m } - D _ { y } I _ { s } \| _ { 1 } \right) ,
$$

where $I _ { s }$ denotes the source image, and $D _ { x } , D _ { y }$ are Sobel operators. Here, $\| \cdot \| .$ <sub>1</sub> denotes the mean absolute error over pixels. The induced update direction is:

$$
u _ { s } ^ { m } = - \frac { \partial \mathcal { L } _ { s } ^ { s p } } { \partial F ^ { m } } .
$$

The negative sign converts the loss gradient into the local descent direction. Since both sourcepreservation gradients are negated, the cosine-based conflict criterion is unchanged compared with using the original loss gradients.

For each patch p, we restrict the two update maps to this patch and flatten them into vectors, denoted as $u _ { i r , p } ^ { m }$ and $u _ { v i s , p } ^ { m } .$ Their directional agreement is:

$$
{ \rho } _ { p } ^ { m } = \frac { \langle u _ { i r , p } ^ { m } , u _ { v i s , p } ^ { m } \rangle } { \| u _ { i r , p } ^ { m } \| _ { 2 } \| u _ { v i s , p } ^ { m } \| _ { 2 } + \epsilon } .
$$

A negative $\rho _ { p } ^ { m }$ means that infrared and visible preservation require opposite local updates for the same fused result.

Medical Image Fusion  
![](images/60414d255ed4bd66a022f4879c095e149b01ee7172381f97ef2f91ec043ddc4d.jpg)  
Figure 8: Full qualitative comparison for MIF task on CT-MRI, MRI-PET and MRI-SPECT.

High-disagreement patch set Ω. We evaluate the conflict on high-disagreement regions, where the two source images are more likely to contain competing information. For each patch p, we compute a source discrepancy score:

$$
d _ { p } = \frac { 1 } { | p | } \sum _ { q \in p } \left( | I _ { i r } ^ { q } - I _ { v i s } ^ { q } | + | D _ { x } I _ { i r } ^ { q } - D _ { x } I _ { v i s } ^ { q } | + | D _ { y } I _ { i r } ^ { q } - D _ { y } I _ { v i s } ^ { q } | \right) .
$$

The high-disagreement patch set Ω is defined as the top 30% patches ranked by $d _ { p }$ . Since Ω is computed only from the two source images, it is fixed for each image pair and shared by all compared methods.

Conflict ratio. The conflict ratio is defined as:

$$
\mathrm { C R } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } \mathbb { I } ( \rho _ { p } ^ { m } < 0 ) .
$$

CR measures how frequently opposite source-preservation updates occur. A lower CR means that fewer local regions require conflicting corrections to preserve the two sources.

Conflict strength. The conflict strength is defined as:

$$
\mathrm { C S } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } [ - \rho _ { p } ^ { m } ] _ { + } .
$$

CS measures how severe the conflicts are. Unlike CR, which only counts whether a conflict exists, CS considers the magnitude of negative alignment. A value close to 1 indicates nearly opposite update directions, while a value close to 0 indicates weak or no conflict.

Table 6: Refinement effects of our paradigm on the RoadScene dataset. Original vs. upgraded models (\* denotes our paradigm inserted).
<table><tr><td>Method</td><td>AG↑</td><td> $E N \uparrow$ </td><td> $S D \uparrow$ </td><td> $M I \uparrow$ </td><td> $Q _ { A B / F } \uparrow$ </td><td> $Q _ { M } \uparrow$ </td><td> $Q _ { P } \uparrow$ </td></tr><tr><td>ReFusion [3] ReFusion* Gain</td><td>5.357 5.692 6.3%↑</td><td>59.336 60.108 1.3%↑</td><td>48.229 49.356 2.3%↑</td><td>2.726 2.916 7.0%↑</td><td>0.437 0.523 19.7%↑</td><td>0.566 0.612 8.1%↑</td><td>0.397 0.418 5.3%↑</td></tr><tr><td>EMMA [51] EMMA* Gain</td><td>4.779 5.514 15.4%↑</td><td>49.127 58.742 19.6%↑</td><td>46.777 49.108 5.0%↑</td><td>2.809 2.943 4.8%↑</td><td>0.436 0.518 18.8%↑</td><td>0.432 0.589 36.3%↑</td><td>0.355 0.407 14.6%↑</td></tr><tr><td>C2RF [26] C2RF* Gain</td><td>4.989 5.603 12.3%↑</td><td>52.270 58.196 11.3%↑</td><td>45.576 48.962 7.4%↑</td><td>2.720 2.935 7.9%↑</td><td>0.519 0.548 5.6%↑</td><td>0.565 0.606 7.3%↑</td><td>0.388 0.412 6.2%↑</td></tr></table>

Update imbalance. The update imbalance is defined as:

$$
\mathrm { U I } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } \frac { | \| u _ { i r , p } ^ { m } \| _ { 2 } - \| u _ { v i s , p } ^ { m } \| _ { 2 } | } { \| u _ { i r , p } ^ { m } \| _ { 2 } + \| u _ { v i s , p } ^ { m } \| _ { 2 } + \epsilon } .
$$

UI is not a cosine-based conflict measure. It measures the normalized magnitude gap between the infrared and visible source-preservation updates. This metric checks whether a low conflict score is caused by suppressing one modality. A lower UI indicates that the two source-preservation updates have more balanced magnitudes.

Together, CR, CS, and UI provide complementary evidence. CR measures how often conflicts occur, CS measures how strong these conflicts are, and UI checks whether conflict reduction is achieved without suppressing one source-preservation update.

## H Broader Impact of the Proposed Paradigm

Table 6 evaluates whether our paradigm can serve as a general refinement strategy for existing fusion models. For a fair plug-in evaluation, we keep the original network architecture, training dataset, and training settings of each baseline unchanged. The only modification is to replace its original supervision with our relation-constrained supervision paradigm. We apply this setting to three representative baselines, including ReFusion, EMMA, and C2RF, and report the results on the RoadScene dataset.

The upgraded versions consistently improve all seven metrics, showing that the gains are not tied to a specific backbone. EMMA\* obtains large improvements on EN, $Q _ { M }$ , and $\boldsymbol { Q } _ { P } ,$ , while ReFusion\* and C2RF\* also achieve clear gains on structural and perceptual quality metrics. These results indicate that relation-constrained supervision provides a transferable optimization signal. It can enhance source preservation, structural fidelity, and fusion quality across different model designs. Therefore, the proposed paradigm has broader applicability beyond our own architecture and can improve existing image fusion systems with limited modification.

## I Extended Ablation Study

We provide the full ablation results in Table 7. Besides the key variants reported in the main paper, we further analyze adaptive relation parameters and adapter training choices. These additional results show consistent trends on both IVF and MIF.

(a) Effect of adaptive relation parameters. We examine whether the relation parameters in RCS should be adaptively inferred through three variants: (1) fixing the consensus center, (2) fixing dominance, and (3) fixing the coordination radius. Results in Table 7 show that all three variants underperform the full model. Fixed center weakens the distinction between shared and non-shared information. Fixed dominance prevents the model from adapting to source-specific contributions. Fixed radius reduces the flexibility of conflict coordination. These results show that sharedness, dominance, and coordination radius should be inferred from each source pair.

Table 7: Extended ablation study of the proposed method on the IVF RoadScene and MIF MRI-PET datasets. This table includes additional variants for fixed relation parameters and adapter training strategies. The best results are highlighted in bold.
<table><tr><td rowspan="2" colspan="2">Group |ID</td><td rowspan="2">Description</td><td colspan="7">IVF on RoadScene</td><td colspan="8">MIF on MRI-PET</td></tr><tr><td>AG↑ EN↑</td><td>SD↑</td><td></td><td>MI↑</td><td>QAB/F ↑</td><td>QM↑</td><td>QP↑|</td><td>AG↑</td><td>EN↑</td><td>SD↑</td><td>MI↑</td><td></td><td>QAB/F↑</td><td>QM↑</td><td>QP↑</td></tr><tr><td rowspan="6">(a)</td><td rowspan="3">|(1) (2)</td><td>w/o LFA</td><td>5.247</td><td>58.932</td><td>47.286</td><td>2.861</td><td>0.506</td><td>0.589</td><td>0.372</td><td>8.721</td><td>94.863</td><td>72.418</td><td>2.321</td><td></td><td>0.552</td><td>0.196 0.218</td><td>0.414</td></tr><tr><td></td><td>w/o FAM</td><td>5.516</td><td>60.217 49.103</td><td>3.041</td><td></td><td>0.537</td><td>0.618</td><td>0.401</td><td>9.184</td><td>97.326</td><td>75.861</td><td>2.512</td><td>0.594</td><td></td><td>0.452</td></tr><tr><td>(3)</td><td>w/o RRM</td><td>5.438</td><td>59.846 48.624</td><td>2.982</td><td></td><td>0.529</td><td>0.611</td><td>0.394</td><td>9.062</td><td>96.814</td><td>75.104</td><td>2.463</td><td>0.586</td><td>0.213</td><td>0.444</td></tr><tr><td>(4)</td><td>w/o RCS</td><td>5.196</td><td>58.714</td><td>47.635</td><td>2.803</td><td>0.497</td><td>0.581</td><td>0.363</td><td>8.604</td><td>94.217</td><td>71.936</td><td>2.276</td><td>0.541</td><td></td><td>0.189</td><td>0.405</td></tr><tr><td>(1)</td><td></td><td></td><td>5.642 60.384</td><td>49.427</td><td>3.086</td><td></td><td>0.545</td><td>0.626</td><td>0.410</td><td>9.346</td><td>97.942</td><td>76.412</td><td>2.574</td><td>0.607</td><td>0.224</td><td>0.461</td></tr><tr><td rowspan="8">(b)</td><td>(2)</td><td>w/o Lshr wlo Lcmp</td><td>5.471</td><td>60.092</td><td>48.891</td><td>2.947</td><td>0.523</td><td>0.613</td><td>0.392</td><td>9.118</td><td>97.156</td><td>75.386</td><td>2.438</td><td></td><td>0.581</td><td>0.216 0.441</td></tr><tr><td>39</td><td>wlo Lcrd</td><td>5.573</td><td>60.263</td><td>49.018</td><td>3.018</td><td>0.532</td><td>0.607</td><td>0.386</td><td>9.241</td><td>97.624</td><td>75.927</td><td>2.496</td><td>0.592</td><td>0.211</td><td>0.433</td></tr><tr><td></td><td>fixed center</td><td>5.589</td><td>60.338</td><td>49.214</td><td>3.052</td><td>0.539</td><td>0.621</td><td>0.404</td><td>9.281</td><td>97.735</td><td>76.103</td><td>2.536</td><td>0.601</td><td>0.221</td><td>0.456</td></tr><tr><td>(5)</td><td>fixed dominance</td><td>5.492</td><td>60.041</td><td>48.776</td><td>2.963</td><td>0.526</td><td>0.614</td><td>0.395</td><td>9.146</td><td>97.208</td><td>75.542</td><td>2.455</td><td>0.584</td><td>0.217</td><td>0.445</td></tr><tr><td>(6)</td><td>fixed radius</td><td>5.601</td><td>60.171</td><td>49.006</td><td>3.011</td><td>0.533</td><td>0.609</td><td>0.388</td><td>9.265</td><td>97.531</td><td>75.818</td><td>2.487</td><td>0.590</td><td>0.212</td><td>0.436</td></tr><tr><td>(1)</td><td>w/o Lssr</td><td>5.324</td><td>59.463</td><td>48.102</td><td>2.902</td><td>0.514</td><td>0.598</td><td>0.379</td><td>8.891</td><td>95.842</td><td>73.864</td><td>2.384</td><td>0.567</td><td>0.204</td><td>0.425</td></tr><tr><td>(2) (3)</td><td>w/o Lsfr</td><td>5.386</td><td>59.721</td><td>48.406</td><td>2.918</td><td>0.517</td><td>0.602</td><td>0.381</td><td>8.976</td><td>96.214</td><td>74.293</td><td>2.406</td><td>0.573</td><td>0.207</td><td>0.429</td></tr><tr><td rowspan="7">(c)</td><td></td><td>uniform wi</td><td>5.538</td><td>60.064</td><td>48.957</td><td>3.006</td><td>0.531</td><td>0.612</td><td>0.393</td><td>9.203</td><td>97.382</td><td>75.674</td><td>2.481</td><td>0.589</td><td>0.215</td><td>0.442</td></tr><tr><td>(4)</td><td>w/o adapter pretraining</td><td>5.291</td><td>59.318</td><td>47.946</td><td>2.884</td><td>0.510</td><td>0.593</td><td>0.376</td><td>8.802</td><td>95.417</td><td>73.218</td><td>2.358</td><td>0.561</td><td>0.201</td><td>0.421</td></tr><tr><td>(5)</td><td>fixed adapter</td><td>5.452</td><td>59.984</td><td>48.713</td><td>2.971</td><td>0.526</td><td>0.609</td><td>0.389</td><td>9.087</td><td>96.936</td><td>75.021</td><td>2.451</td><td>0.583</td><td>0.212</td><td>0.438</td></tr><tr><td>(6)</td><td>joint optimization</td><td>5.408</td><td>59.762</td><td>48.531</td><td>2.943</td><td>0.521</td><td>0.604</td><td>0.384</td><td>9.014</td><td>96.573</td><td>74.612</td><td>2.427</td><td>0.578</td><td>0.209</td><td>0.432</td></tr><tr><td></td><td>Full Model (Transformer)</td><td>5.861</td><td>61.069</td><td>50.581</td><td>3.179</td><td>0.564</td><td>0.643</td><td>0.425</td><td>9.690</td><td>99.031</td><td>78.072</td><td>2.667</td><td>0.627</td><td></td><td>0.481</td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.235</td></tr></table>

Table 8: Comparison of runtime, computational cost (GFLOPs), and model size (Params) on IVF and MIF tasks. All values are averaged per image.
<table><tr><td colspan="4">IVF</td><td colspan="4">MIF</td></tr><tr><td>Method</td><td>Time (s)</td><td>GFLOPs (G)</td><td>Params (M)</td><td>Method</td><td>Time (s)</td><td>GFLOPs (G)</td><td>Params (M)</td></tr><tr><td>EMMA</td><td>0.058</td><td>8.861</td><td>1.516</td><td>EMMA</td><td>0.009</td><td>8.861</td><td>1.516</td></tr><tr><td>Text-Difuse</td><td>23.818</td><td>2742.5</td><td>119.460</td><td>Text-Difuse</td><td>24.424</td><td>2742.5</td><td>119.460</td></tr><tr><td>Tc-MoA</td><td>0.543</td><td>61.000</td><td>340.580</td><td>Tc-MoA</td><td>0.161</td><td>61.000</td><td>340.580</td></tr><tr><td>ReFusion</td><td>0.026</td><td>98.344</td><td>1.486</td><td>ReFusion</td><td>0.012</td><td>98.344</td><td>1.486</td></tr><tr><td>C2RF</td><td>0.089</td><td>123.869</td><td>1.325</td><td>C2RF</td><td>0.112</td><td>123.869</td><td>1.325</td></tr><tr><td>GIF-Net</td><td>0.077</td><td>39.814</td><td>0.823</td><td>GIF-Net</td><td>0.084</td><td>39.814</td><td>0.823</td></tr><tr><td>SAGE</td><td>0.020</td><td>29.285</td><td>0.136</td><td>CCF</td><td>31.876</td><td>1114.000</td><td>552.810</td></tr><tr><td>Omni-Fuse</td><td>2.614</td><td>172.923</td><td>78.320</td><td>Mask-Difuser</td><td>0.503</td><td>995.706</td><td>171.262</td></tr><tr><td>CLDyN</td><td>0.272</td><td>174.061</td><td>0.46</td><td>CLDyN</td><td>0.311</td><td>174.06</td><td>0.46</td></tr><tr><td>ISFusion</td><td>0.382</td><td>273.472</td><td>1.372</td><td>ISFusion</td><td>0.401</td><td>273.472</td><td>1.372</td></tr><tr><td>Ours(CNN)</td><td>0.013</td><td>3.284</td><td>0.067</td><td>Ours(CNN)</td><td>0.010</td><td>3.284</td><td>0.067</td></tr><tr><td>Ours(Mamba)</td><td>0.008</td><td>2.137</td><td>0.038</td><td>Ours(Mamba)</td><td>0.006</td><td>2.137</td><td>0.038</td></tr><tr><td>Ours(Transformer)</td><td>0.028</td><td>6.912</td><td>0.142</td><td>Ours(Transformer)</td><td>0.023</td><td>6.912</td><td>0.142</td></tr></table>

(b) Effect of relation-aware scoring. We evaluate the token weighting strategy by replacing the relation-aware weight w<sub>i</sub> with uniform weights. This variant treats all tokens equally during adapter ranking and consistently degrades performance. The result indicates that not all regions contribute equally to relation learning. Tokens with reliable sharedness, clear dominance, and smaller coordination uncertainty should receive higher weights. This verifies the necessity of relation-aware scoring in the adapter objective.

(c) Effect of adapter pretraining. We study the role of adapter pretraining by directly starting alternating optimization from an untrained adapter. This setting leads to a clear performance drop on both IVF and MIF. Since the adapter defines the supervision space, an unstable adapter can provide noisy relation constraints to the fusion network. Pretraining gives LFA a discriminative initialization before it is used for fusion supervision.

(d) Effect of fixed adapter. We further test a fixed-adapter variant, where LFA is pretrained once and then kept frozen during fusion training. This variant performs worse than the full model. The result shows that a static supervision space is insufficient because fused images evolve as the fusion network improves. Updating LFA during alternating optimization allows the supervision space to adapt to current fused results.

(e) Effect of joint optimization. We also compare alternating optimization with joint optimization, where LFA and the fusion network are updated simultaneously. Joint optimization gives lower results than the full model. This suggests that changing the supervision space and the fused output at the same time can introduce unstable mutual feedback. Alternating optimization is more stable because one side is fixed while the other is updated.

## J Model Complexity and Overhead Analysis

Table 8 compares the runtime, computational cost, and parameter size of representative IVF and MIF methods. Overall, our variants maintain a compact model scale and low computational overhead across both tasks. This indicates that the proposed framework does not rely on a heavy fusion backbone to achieve effective fusion performance.

Compared with many fusion methods, our models require fewer parameters and lower computation while keeping competitive inference speed. The three variants also provide different efficiencyperformance trade-offs: the Mamba variant is the most lightweight, the CNN variant offers a balanced design, and the Transformer variant introduces a moderate increase in cost for stronger representation ability. These results show that the proposed relation-constrained supervision can work effectively with lightweight architectures and is suitable for practical fusion scenarios where efficiency matters.

## K Implementation Details

Feature extraction and adapter. We use frozen DINO and CLIP as feature providers, denoted by Ψ. For each input image X, we extract multi-level features from DINO layer 4, CLIP layer 6, and DINO layer 8, denoted by ${ f _ { D } ^ { 4 } ( X ) , f _ { C } ^ { 6 } ( X ) }$ , and $f _ { D } ^ { 8 } ( X )$ , respectively. Each feature map is projected by $\mathbf { a \ 1 \times 1 }$ convolution, followed by normalization and bilinear resizing. The corresponding projection operators are denoted by $P _ { D } ^ { 4 } , P _ { C } ^ { 6 }$ , and $P _ { D } ^ { 8 }$ . The projected features are resized to a common token grid of $1 6 \times 1 6$ and projected to 256 channels. They are then flattened into token sequences and concatenated along the channel dimension. The feature aligner $G _ { \mathrm { a l i g n } }$ is implemented as an MLP with hidden dimension 512 and output dimension 256, producing the aligned representation $y x$

In the relation reasoning module, the pair encoder $E _ { \mathrm { p a i r } }$ takes $\left[ y _ { A } , y _ { B } , | y _ { A } - y _ { B } | , y _ { A } \odot y _ { B } \right]$ as input and produces the pair feature ${ q } _ { A , B }$ . The source-specific heads $H _ { A }$ and $H _ { B }$ produce the pair-conditioned source representations $z _ { A }$ and $z _ { B }$ . The prediction heads $h _ { m } , h _ { d }$ , and $h _ { r }$ estimate sharedness $m _ { A , B }$ , dominance $d _ { A , B } .$ , and coordination radius $r _ { A , B }$ , respectively. The pair encoder is implemented as an MLP with GELU activation and LayerNorm, and the prediction heads are lightweight linear layers. Sharedness and dominance are normalized by the sigmoid function, while the coordination radius is constrained by the softplus function.

Hyperparameter settings. For the relation-aware source-pair score, we set the weighting coefficients to $\lambda _ { m } = 1 . 0 , \lambda _ { d } = 0 . 5$ , and $\lambda _ { r } = 0 . 5$ . The numerical stability constant is set to $\epsilon = 1 \stackrel { - } { \times } 1 0 ^ { - 6 }$ The source-side ranking margin and fused-side ranking margin are set to $\delta _ { \mathrm { s r c } } = 0 . 2$ and $\delta _ { \mathrm { f u s e } } = 0 . 2$ respectively. For the source-preservation update conflict analysis, we use patch size $1 6 \times 1 6$ and define the high-disagreement region Ω as the top 30% patches ranked by source intensity and gradient discrepancy.

Fusion network architecture. As shown in Fig. 9, our FusionNet adopts a two-branch encoderdecoder architecture. The two source images are first processed by two modality-specific encoders, denoted as Encoder A and Encoder B. These two encoders have the same architecture but do not share weights, allowing each branch to learn source-specific feature extraction. Taking the Transformerbased backbone as an example, each encoder is composed of stacked Transformer blocks and extracts a feature from its corresponding source image. The two latent features are then concatenated along the channel dimension and fed into the decoder. The decoder aggregates the concatenated feature through stacked Transformer blocks and a lightweight output head with LeakyReLU and Sigmoid, producing the final fused image.

Under this unified design, we instantiate Transformer- [43], CNN- [47], and Mamba-based [15] fusion backbones to verify that the proposed supervision paradigm is not tied to a specific architecture. The Transformer variant is illustrated in Fig. 9. The CNN and Mamba variants follow the same two-branch encoder-decoder pipeline, and their only difference is that the Transformer blocks are replaced by CNN blocks [47] and VMamba blocks [15], respectively.

![](images/5ead7add8ed34f79a72dd9f056c43c790b929c0c283290ed833c0f415681c77f.jpg)  
Figure 9: Schematic diagram of fusion network.

## L Architecture and Pretraining of the Shared Token Encoder

The shared token encoder $G _ { \mathrm { s h r } }$ is introduced to extract the information consistently supported by the two source modalities. Its design follows a simple principle: the extracted shared representations should be highly consistent across modalities, while the remaining residual representations should contain less correlated source-specific information.

Architecture. $G _ { \mathrm { s h r } }$ is implemented as a lightweight token-wise MLP with shared weights across all modalities. Given an aligned representation $z \in \breve { \mathbb { R } } ^ { N \times C }$ , each token is processed independently as:

$$
G _ { \mathrm { s h r } } ( z ) = W _ { 2 } \phi ( W _ { 1 } \mathrm { L N } ( z ) ) ,\tag{42}
$$

where $\mathrm { L N } ( \cdot )$ denotes LayerNorm, ϕ(·) denotes GELU, and $W _ { 1 }$ and $W _ { 2 }$ are linear projections. The output dimension is kept identical to that of $z ,$ allowing the shared representation to remain in the same supervision space. The same $G _ { \mathrm { s h r } }$ is applied to the two source representations and the fused representation:

$$
\begin{array} { r } { c _ { A } = G _ { \mathrm { s h r } } ( z _ { A } ) , \qquad c _ { B } = G _ { \mathrm { s h r } } ( z _ { B } ) , \qquad c _ { F } = G _ { \mathrm { s h r } } ( z _ { F } ) . } \end{array}\tag{43}
$$

The corresponding residual representations are defined by direct subtraction:

$$
u _ { A } = z _ { A } - c _ { A } , ~ u _ { B } = z _ { B } - c _ { B } , ~ u _ { F } = z _ { F } - c _ { F } .\tag{44}
$$

This design keeps the decomposition explicit and avoids introducing an additional decoder or reconstruction branch.

Correlation-Driven Pretraining. Following the correlation-driven feature decomposition strategy of CDDFuse [50], we pretrain $G _ { \mathrm { s h r } }$ using matched source pairs only. Similar to its common/detail feature decomposition principle, we encourage the shared representations $c _ { A }$ and $c _ { B }$ to be highly correlated, while reducing the correlation between the corresponding residual representations $u _ { A }$ and $u _ { B }$ . We define the correlation between two token representations as:

$$
\mathrm { C C } ( x , y ) = \frac { \left. x - \mu _ { x } , y - \mu _ { y } \right. } { \| x - \mu _ { x } \| _ { 2 } \| y - \mu _ { y } \| _ { 2 } + \epsilon } ,\tag{45}
$$

where $\mu _ { x }$ and $\mu _ { y }$ denote the corresponding mean representations. Following this correlation-driven decomposition principle, the pretraining objective is defined as:

$$
\mathcal { L } _ { G } = 1 - \mathrm { C C } ( c _ { A } , c _ { B } ) + \mathrm { C C } ^ { 2 } ( u _ { A } , u _ { B } ) ,\tag{46}
$$

where the first term encourages $G _ { \mathrm { s h } }$ to retain information consistently represented in both modalities, while the second term suppresses correlated information in the residual components. Different from CDDFuse, which performs decomposition within its fusion architecture, we adapt this principle to the aligned token representations and use it only to pretrain the shared token encoder. Since $u _ { A } = z _ { A } - c _ { A }$ and $u _ { B } = z _ { B } - c _ { B }$ are defined explicitly, the decomposition exactly preserves the original representations and therefore does not require an additional reconstruction loss.

After pretraining, the parameters of $G _ { \mathrm { s h r } }$ are frozen. The same encoder is subsequently applied to $z _ { A } ,$ $z _ { B } ,$ , and $z _ { F }$ during relation-constrained supervision and source–fused ranking. Keeping σ Gsbr $G _ { \mathrm { s h r } }$ fixed provides a stable decomposition of shared and non-shared information while LFA and the fusion network are optimized alternately.

## M Limitations

Although our method achieves strong fusion performance with a compact model design, it still has some limitations. First, the relation-constrained supervision relies on pretrained vision models to construct the supervision space. Its effectiveness may therefore be influenced by the representation quality and domain coverage of the selected pretrained models. Second, the current design mainly focuses on pairwise source relations. Extending it to more complex fusion settings with more than two modalities may require additional relation modeling. Future work will explore more robust supervision construction and broader adaptation to multi-modal and multi-source fusion scenarios.