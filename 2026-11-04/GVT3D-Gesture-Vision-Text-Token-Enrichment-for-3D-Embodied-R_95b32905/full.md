# GVT3D: Gesture-Vision-Text Token Enrichment for 3D Embodied Reference Understanding

Vimarsh Shah<sup>1</sup>, Dulanga Weerakoon<sup>2</sup>, Ajinkya Deshpande<sup>1</sup>, Saksham Agarwal<sup>1</sup>, Nilesh Bhatia<sup>1</sup>, Neel Kiran Naik<sup>1</sup>, Parth Shah<sup>1</sup>, Sravan Danda<sup>1</sup>, Sanjay E. Sarma<sup>3</sup>, Archan Misra<sup>4</sup>, Vigneshwaran Subbaraju<sup>5,6</sup>, Sougata Sen<sup>1</sup>

Abstract— Natural human–robot interaction requires robots to interpret language and gestures within a 3D environment to support tasks such as pick-and-place and object retrieval. 3D Embodied Reference Understanding (3D-ERU) supports this by identifying the 3D location of a target object from a scene (point-cloud) containing a human pointing gesture and an accompanying language description. However, current 3D-ERU models lack alignment of cross-modal representations and incur high processing costs that increase with the pointcloud’s size. To address this, we propose GVT3D, which employs an Embodied Fusion module that jointly models language, gesture, and vision inputs within a shared embedding space, promoting cross-modal alignment. We additionally introduce a foveation approach that selectively processes the point cloud within a coarse region around the estimated pointing direction to reduce computations. GVT3D achieves ∼8% higher accuracy and foveation enables 2.95× faster execution while maintaining comparable accuracy, relative to prior 3D-ERU models.

## I. INTRODUCTION

Fluent, natural interaction between humans and robots requires a robot to accurately interpret the natural linguistic and gestural expressions generated concurrently in natural human communication [1]. In this context, an embodied reference denotes a human instruction comprising a language expression accompanied by an optional pointing gesture that refers to a single target object within a scene [2]; such multimodal instructions are common for pick-and-place or object retrieval robotic tasks associated with home and warehouse logistics (e.g., [3]), as presented in Figure 1. The capability to accurately resolve such commands, through joint reasoning over language and gesture signals, is referred to as ‘embodied reference understanding’.

![](images/be8655756e417d973b7ca7148e695e9043ec07fde2d349db6e21a657da04afe4.jpg)  
(a)

![](images/8f1f571574bfa4221174877a89a7e8ec80c4cd49e3c332945bd9d709ae3533c2.jpg)  
(b)

![](images/b46a81abd4e133ecd1f5144bed7451176c43af26ca6464cbcac1edd9dca3d50d.jpg)  
(c)  
Fig. 1: (a) A practical task requiring 3D-ERU through the fusion of gesture, vision and text. (b) Point-cloud of a human performing gesture. (c) A sample of 3D-ERU from Imputerefer dataset.

Embodied reference understanding has been explored using deep neural network-based approaches that operate on 2D visual inputs captured from ego-centric and exo-centric views [2], [4]. 3D Embodied Reference Understanding (3D-ERU) extends this problem into identifying a target object within a 3D scene, typically represented as a point cloud captured from a global, exo-centric view via LiDARs installed in the infrastructure [5], [6]. Compared to 2D settings, this introduces additional challenges due to spatial depth reasoning, geometric ambiguity, and articulated human pose variability, but has remained relatively underexplored; Ges3ViG [6] and ScanERU [5] being among the few approaches that address this problem in 3D settings. The limited literature on 3D-ERU leaves the following gaps unexplored:

• Lack of Unified Cross-Modal Semantics: Despite substantial progress in grounding linguistic referring expressions, existing methods for 3D-ERU do not explicitly model the semantics of interaction between language and gesture within a common representation space. Prior work on 3D-ERU incorporates gesture only through late fusion, where the estimated pointing direction is used as the final arbiter to re-weight or select candidate regions along the pointing ray, e.g., Ges3ViG [6]. Such formulations treat gesture as an auxiliary signal operating independently of language, rather than truly exploiting the interactions between language and gesture that jointly constrain the identification and localization of the target. Previous studies have also shown that relying solely on pointing can be inaccurate due to systematic human pointing biases and variability in articulation [7], and that the relative importance of language and pointing may depend on varying environmental context [8]. Thus, gestures must be interpreted jointly with language throughout the reasoning process, with the relative importance adjusted based on factors like pointing fidelity, or presence of confounding distractor objects.

• Lack of Computational Efficiency: Existing models for 3D-ERU [5], [6] remain untested in practical real-world scenarios involving downstream tasks/reasoning on top of 3D-ERU being executed on-device by a robot: for example, enabling a robot to rapidly and accurately navigate to the object specified by an embodied reference. Extant 3D-ERU models incur substantial computational overhead that scales with the number of points; we show in Section IV-C that this prohibitive overhead can meaningfully affect robot responsiveness on embedded edge computing platforms. Consequently, 2D ERU models [4], [8] remain widely preferred and adopted, even though 2D reasoning limits the ability to identify the 3D coordinates of the target object in the real world. Computationally efficient, edge-compatible 3D-ERU models are thus needed to increase adoption in practical deployments.

In this paper, we introduce Gesture-Vision-Text Token Enrichment for 3D Embodied Reference Understanding (GVT3D), a new model that explicitly aligns gesture, vision, and language features within a shared embedding space through a novel embodied fusion mechanism, enabling semantically-rich structured cross-modal interaction before any modality-specific score is computed. Rather than relying solely on late fusion or simple feature concatenation, GVT3D combines this joint representation learning with a final learned re-weighting stage, allowing the model to adaptively trust language, gesture, or vision depending on their relative reliability for a given instance, leading to improved accuracy compared to baseline 3D-ERU models. To further address the computational cost of processing large 3D scenes, we propose point cloud foveation, whereby candidate objects are first ranked by how well they align with the pointing direction determined via a lightweight gesture predictor, and the expensive 2D visual encoding step is executed only on the most gesture-consistent proposals. In summary, we make the following key contributions:

1) GVT3D model for 3D-ERU: On the ImputeRefer [6] benchmark, our GVT3D outperforms Ges3ViG (prior SOTA) by achieving 8% higher accuracy, alongside a 1.75× speedup. Our analysis further shows that GVT3D meaningfully aligns the vision, language and gesture embeddings, enabling it to adjust the importance of individual modality for a given input context.

2) Foveation for Efficient 3D-ERU with GVT3D: To address the computational overhead of processing large point clouds found in 3D-ERU data, we propose point cloud foveation-based computation. Compared to Ges3ViG, GVT3D with foveation runs 2.95× faster and reduces energy consumption by 70.76% while improving accuracy by 2.75%.

3) Real-World Prototype of GVT3D with Foveation: We implement a robotic system that integrates GVT3D as its reasoning module and Nav2 [9] for navigation planning to support a representative object retrieval task. In realworld evaluations, foveation-enhanced GVT3D achieves a success rate of 70% across 30 scenes which the model hasn’t seen before, in comparison to 56% achieved by Ges3ViG on the same task.

## II. BACKGROUND AND RELATED WORK

## A. 2D Visual Grounding

Localising image regions from natural language descriptions, commonly known as Referring Expression Comprehension, has been a long-standing challenge in vision-language understanding. Early approaches followed a proposal-based paradigm where candidate regions were generated and matched with language features using multimodal embeddings [10]. More recent work has explored transformer-based architectures that align textual tokens with spatial regions through attention mechanisms [11], [12]. In parallel, large-scale vision-language pretraining has enabled strong semantic alignment between images and text [13]. Despite these advances, most 2D grounding methods assume that language uniquely specifies the referent, ignoring the use of additional cues, such as gestures and gaze, in natural human-robot communication [14].

## B. 2D-Embodied Reference Understanding

Recent work extends referring-expression comprehension by incorporating embodied cues such as pointing gestures alongside language. In the 2D setting, this integration has largely focused on system-level efficiency rather than representation-level fusion. For instance, COSM2IC enhances real-time multimodal instruction understanding [4], involving 2D vision, language and pointing gestures, by introducing adaptive model selection to balance accuracy and computational efficiency, rather than jointly modeling how gesture and language features interact.

## C. 3D Visual Grounding

3D visual grounding extends referring-expression comprehension to 3D scene representations, typically point clouds, requiring reasoning over spatial geometry and object layouts. Early work, such as ScanRefer, introduced largescale datasets and proposal-based grounding pipelines built upon ScanNet scenes [15], [16]. Subsequent work, including ReferIt3D, highlighted the challenges of spatial reasoning in cluttered environments [17]. More recent methods incorporate attention mechanisms to associate language tokens with object-level geometric features [18], or adopt transformerbased architectures for end-to-end grounding [19]. Leveraging vision-language pretraining has further improved performance by aligning rendered multi-view images with textual embeddings [20], while approaches such as PointCLIP adapt CLIP representations for point cloud understanding through multi-view projections [21].

## D. 3D-Embodied Reference Understanding

3D embodied reference understanding studies how agents interpret multimodal instructions that combine language with physical cues like pointing gestures in 3D environments. Ges3ViG originally introduced approaches to incorporate pointing gestures into 3D visual grounding by modeling gesture as a geometric ray extending from the shoulder to fingertip [6]. ScanERU further investigates interactive grounding using human pose signals to derive spatial constraints [5]. However, these methods typically represent gestures using deterministic geometric formulations and integrate gesture information through late fusion strategies. In contrast, our work treats gesture, vision, and language as tightly coupled modalities and jointly models the semantics of their interactions through a unified token-level fusion framework.

## III. GVT3D: GESTURE-VISION-TEXT FOR 3D-ERU

We now present the proposed GVT3D approach, illustrated in Figure 2. GVT3D differs from prior 3D-ERU methods in two ways. (i) we estimate the pointing direction using a learned model instead tracing a shoulder-to-fingertip line, to capture natural arm gesture. (ii) instead of using gesture as a separate signal for refining object selection, GVT3D’s Embodied Fusion Module performs joint reasoning over gesture, language, and vision tokens, allowing each modality to inform others before the target object is chosen.

## A. Stage 1: Per Modality Encoding

The first stage in the GVT3D architecture encodes the three signal streams: vision, language, and gesture.

1) Vision Module: The vision module encodes the scene point cloud via a PointGroup detector [22], which takes a 3D point cloud $\mathbf { P } \in \mathbb { R } ^ { N \times C }$ as input, where N is the number of points and $C = 6$ (three coordinates and three RGB values). The detector outputs: (i) per-point semantic features ${ \bf F } _ { S } \in  \large $ $\mathbb { R } ^ { N \times d _ { \mathrm { d e t } } }$ and scores $\mathbf S _ { S } \in \bar { \mathbb R } ^ { N \times N _ { c } }$ , used for gesture inference; and (ii) proposal features $\mathbf { F } _ { p } ^ { 3 \mathrm { D } } \in \mathbb { R } ^ { M \times d _ { \mathrm { d e t } } }$ and objectness scores $\mathbf { S } _ { p } \in \mathbb { R } ^ { M }$ , used for proposal ranking, where $d _ { \mathrm { d e t } } =$ 32, M is the number of proposals, and $N _ { c }$ is the number of semantic classes. To selectively allocate computation in the complementary 2D branch, we introduce Gesture-Guided Foveation before multi-view depth projection. This selects a subset of proposals for depth projection and CLIP encoding, while preserving $\mathbf { F } _ { p } ^ { 3 \mathrm { D } }$ for all M proposals.

a) Gesture-Guided Foveation: We use the unit pointing direction $\hat { { \bf d } } _ { \mathrm { p r e d } }$ estimated by the Pointing Regressor in the Gestural Module, described in Section III-A.3, to select proposals for 2D visual encoding. For proposal i, let $\mathbf { c } _ { i }$ denote its bounding-box centre and a denote the shoulder joint of the pointing arm. Its pointing bias is

$$
b _ { i } = \frac { ( \mathbf { c } _ { i } - \mathbf { a } ) ^ { \top } \hat { \mathbf { d } } _ { \mathrm { p r e d } } } { \operatorname* { m a x } ( \lVert \mathbf { c } _ { i } - \mathbf { a } \rVert _ { 2 } , \epsilon ) }\tag{1}
$$

where ϵ ensures numerical stability. This is the same pointing bias used subsequently in Adaptive Score Fusion.

When foveation is enabled and a human is detected, proposals are ranked by $b _ { i }$ , and the highest-scoring

$K _ { f }$ number of proposals are selected where $\begin{array} { r l } { K _ { f } } & { { } = } \end{array}$ min $( M , \operatorname* { m a x } ( k , k _ { \operatorname* { m i n } } ) )$ . Here, k is the configured proposal budget and $k _ { \mathrm { m i n } }$ is the minimum number of proposals to retain for 2D encoding. Let $\mathcal { T } _ { f }$ contains the indices of the selected proposals. We construct the binary selection mask $m _ { i } = 1 [ i \in \mathbb { Z } _ { f } ]$ and perform depth projection and CLIP encoding only for proposals for which $m _ { i } = 1$ . Only selected proposals undergo depth projection (described later) and CLIP encoding. This proposal-level foveation preserves the spatial resolution of each generated depth map, while reducing number of proposals processed by the 2D branch.

b) Multi-view depth projection: For each of the selected proposals, the scene point cloud is cropped using the predicted axis-aligned bounding box, centred at the origin, and normalised to a unit sphere. The normalised cloud is then projected via orthographic projection onto $V = 3$ singlechannel depth maps of size $2 2 4 \times 2 2 4$ , with viewpoints uniformly distributed over the sphere. Each pixel stores the normalised depth of the nearest projected point; RGB information is not used.

c) CLIP visual encoding: Each depth map is replicated across three channels and normalised using the standard CLIP image statistics. Depth images from selected, nonempty proposals are encoded in a single forward pass through the frozen CLIP ViT-B/32 visual backbone, yielding per-view features $\mathbf { F } _ { \mathrm { v i e w } } \in \mathbb { R } ^ { M \times V \times d _ { \mathrm { c l i p } } }$ , where $d _ { \mathrm { c l i p } } = 5 1 2$ This approach, inspired by PointCLIP [21], operates directly on raw 3D geometry rather than rendered RGB images.

d) Inter-view adapter: To fuse information across the V viewpoints of proposal i, we introduce a bottleneck MLP with a residual connection. For each proposal, the V view vectors are concatenated along the feature dimension and passed through the MLP:

$$
\mathbf { F } _ { \mathrm { a d a p t } , i } = \mathbf { F } _ { \mathrm { v i e w } , i } + \mathrm { r e s h a p e } \Big ( \mathrm { M L P } _ { \mathrm { b n } } ( \mathrm { f a t t e n } ( \mathbf { F } _ { \mathrm { v i e w } , i } ) ) , ~ V , ~ d _ { \mathrm { c l i p } } \Big )\tag{2}
$$

where flatten $( \mathbf { F } _ { \mathrm { v i e w } , i } ) \in \mathbb { R } ^ { V }$ <sup>·dclip</sup> and $\mathrm { M L P _ { b n } }$ is a two-layer MLP that reduces dimensionality by a factor of 4 before projecting back to $V \cdot d _ { \mathrm { c l i p } }$

e) View aggregator and projector: The adapted view features are average-pooled across the view dimension and passed through a two-layer projection MLP to produce proposal-level 2D features:

$$
\mathbf { F } _ { p , i } ^ { 2 D } = \mathrm { M L P } \left( \frac { 1 } { V } \sum _ { v = 1 } ^ { V } \mathbf { F } _ { \mathrm { a d a p t } , i } ^ { ( v ) } \right) \in \mathbb { R } ^ { M \times d _ { \mathrm { v i s } } } , \qquad d _ { \mathrm { v i s } } = 1 2 8 .
$$

f) Combining visual features: For proposals not selected by foveation $( m _ { i } = 0 ) , F _ { p } ^ { 2 D }$ is set to a zero vector. The 3D proposal features from PointGroup and the 2D features from the PointCLIP pipeline are concatenated and L2-normalised to form the joint visual representation:

$$
\mathbf { F } _ { v } = \ell _ { 2 } ( [ \mathbf { F } _ { p } ^ { \mathrm { 3 D } } \parallel \mathbf { F } _ { p } ^ { \mathrm { 2 D } } ] ) \in \mathbb { R } ^ { M \times ( d _ { \mathrm { d e t } } + d _ { \mathrm { v i s } } ) } = \mathbb { R } ^ { M \times 1 6 0 } ,\tag{3}
$$

where [·∥·] denotes feature-dimension concatenation and $\ell _ { 2 } ( \cdot )$ denotes L2-normalisation. The resulting representation retains all M proposals for further processing.

![](images/a1a1458a653e049661dbb7ef4a19a4e219edc88eebdfc777e590b6813b441e00.jpg)  
Fig. 2: Proposed GVT3D Approach

2) Language Module: We adopt the CLIP based language encoder, replicating the approach adopted in Ges3ViG [6]. The referring expression is tokenized using the CLIP tokenizer, and both token-level and sentence-level representations are extracted from the frozen CLIP text transformer. The token level features are projected through a two-layer MLP to a shared embedding dimension $d _ { t } = 1 2 8 ,$ , yielding word-level features $\mathbf { F } _ { l , t } ~ \in ~ \mathbb { R } ^ { N \mathrm { t o k } \times 1 2 8 }$ . The sentence level representation is obtained via a learned projection head applied to the global CLIP text embedding, resulting in $\mathbf { F } _ { l , s } \in \mathbb { R } ^ { 1 2 8 }$

3) Gestural Module: In contrast to work like Ges3ViG [6], which models pointing gestures using a heuristic shoulder-to-fingertip ray, we introduce a learned pointing direction estimator. In natural human interactions, pointing gestures are rarely perfectly straight arms [7], so a fixed geometric ray does not always reflect the intended target direction. To address this, we propose a gestural module that learns the pointing direction directly from joint positions, rather than assuming a fixed arm geometry.

a) Joint Feature Encoder: This module extracts the human point cloud segment identified by the PointGroup detector predictions. Using per-point semantic features $\mathbf { F } _ { s } \in$ $\mathbb { R } ^ { N \times 3 2 }$ and semantic scores, points classified as human are selected and fed through a PointNet-style joint regressor that predicts soft assignment weights, $\dot { \mathbf { W } } _ { \mathrm { h u m } } \in \mathbb { R } ^ { \breve { N } _ { \mathrm { h u m } } \times N _ { j } }$ for $N _ { j } = 5 5$ SMPL-X joints [23]. Joint 3D coordinates are computed as $\mathbf { J } = \mathrm { S o f t m a x } ( \mathbf { W } _ { \mathrm { h u m } } ^ { T } ) \times \mathbf { P } _ { \mathrm { h u m } }$ and initial joint features as $\mathbf { F } _ { j , \mathrm { i n i t } } ~ = ~ \mathrm { S o f t m a x } ( \mathbf { W } _ { \mathrm { h u m } } ^ { T } ) \times \mathbf { F } _ { s , \mathrm { h u m } } ,$ similar to computation in Point2Skeleton [24]. These initial features are processed through a Joint Feature Extraction module consisting of 1D convolution layers $( 3 2  6 4  1 2 8 )$ with LeakyReLU activations, producing joint-level features $\mathbf { F } _ { j } \in$ $\mathbb { R } ^ { N _ { j } \times 1 2 8 }$ . The module also computes a global aggregate feature $F _ { a g g }$ by average pooling intermediate representations at each layer and concatenating them. This aggregate feature captures the global pose information.

The Joint Feature Extractor’s output is sequentially passed through two self-attention layers, each comprising a multihead attention mechanism $( h = 4 , d _ { k } = d _ { v } = 3 2 )$ . This outputs a set of gesture tokens $\mathbf { G } \in \mathbb { R } ^ { B \times N _ { j } \times 1 2 8 }$ that encodes pose information for the downstream Joint Fusion task.

b) Pointing Regressor: With the computed joint features, we employ a lightweight MLP termed the Pointing Regressor, which takes as input the 3D positions of eight key joints from a set of 55 total human joints, namely the shoulder, elbow, wrist, and index fingertip for both left and right arms. For each arm, we compute shoulder-centred relative offsets:

$$
\begin{array} { r } { \delta _ { \mathrm { a r m } } = \left[ \mathbf { p } _ { \mathrm { e l b o w } } - \mathbf { p } _ { \mathrm { s h o u l d e r } } , \ \mathbf { p } _ { \mathrm { w r i s t } } - \mathbf { p } _ { \mathrm { s h o u l d e r } } , \ \mathbf { p } _ { \mathrm { i n d e x } } \right. } \\ { \left. - \mathbf { p } _ { \mathrm { s h o u l d e r } } , \ \mathbf { p } _ { \mathrm { w r i s t } } - \mathbf { p } _ { \mathrm { e l b o w } } \right] \ \in \ \mathbb { R } ^ { 1 2 } } \end{array}\tag{4}
$$

The offset vectors from both arms are concatenated to form a 24-dimensional input, which is then passed through a threelayer MLP (24 → 64 → 64 → 3) with ReLU activations. The output is L2-normalised to produce the predicted unit pointing direction $\hat { \mathbf { d } } _ { \mathrm { { p r e d } } }$ . This $\dot { \bf d } _ { \mathrm { p r e d } }$ is used to compute proposalwise pointing biases, which guide top-k proposal selection for foveated visual encoding and contribute to adaptive score fusion in the Cross-Modal Matching Module.

The inclusion of $\mathbf { p } _ { \mathrm { w r i s t } } \ : - \ : \mathbf { p } _ { \mathrm { e l b o w } }$ explicitly captures the forearm direction independently of the upper arm. This allows the regressor to learn the relationship between joint geometry and pointing direction directly from the data, rather than assuming a fixed shoulder-to-fingertip line.

## B. Stage 2: Fusion

Next, the projections from the three primary modules in Stage 1 are fused in Stage 2, which we describe next.

1) Embodied Fusion Module: Prior work on 3D-ERU incorporates gestural features using late fusion strategies, where each modality is processed largely independently and combined only at the final stages [6]. Instead, we formalise gesture, vision, and text as three tightly coupled modalities that should be projected into a shared embedding space and conditioned on one another throughout representation learning.

![](images/1cf531397e8d27fb43dd09b328cfca201ccb9e3eeb5564423cd7233c5718c48e.jpg)  
Fig. 3: Robotic Navigation Implementation

We propose a novel feature fusion mechanism termed Embodied Fusion Module. Concretely, we design a 3-way joint self attention computation that enables mutual conditioning across gesture, vision, and text tokens, resulting in enriched modality representations that are better aligned for downstream grounding. We now detail the components of our embodied fusion module:

a) Modality Projection Layer: Each modality has a different native dimensionality: text tokens lie in $\mathbb { R } ^ { d _ { t } } ~ ( d _ { t } =$ 128), gesture tokens in $\mathbb { R } ^ { d _ { g } } ~ ( d _ { g } = 1 2 8 )$ , and vision tokens in $\mathbb { R } ^ { d _ { v } } ~ ( d _ { v } = 1 6 0 )$ . We project each into a common fusion space of dimension $d _ { f } = 2 5 6$ via learned bias-free linear layers as shown below:

$$
\mathbf { T } _ { f } = \mathbf { F } _ { l , t } \mathbf { W } _ { t } , \quad \mathbf { W } _ { t } \in \mathbb { R } ^ { d _ { t } \times d _ { f } } ,\tag{5}
$$

$$
\mathbf { G } _ { f } = \mathbf { G } \mathbf { W } _ { g } , ~ \mathbf { W } _ { g } \in \mathbb { R } ^ { d _ { g } \times d _ { f } } ,\tag{6}
$$

$$
\mathbf { V } _ { f } = \mathbf { F } _ { v } \mathbf { W } _ { v } , ~ \mathbf { W } _ { v } \in \mathbb { R } ^ { d _ { v } \times d _ { f } } .\tag{7}
$$

b) Joint Fusion Layer: The projected tokens are concatenated along the sequence dimension into a single sequence of length $L _ { t } + N _ { j } + K$

$$
\mathbf { S } = [ \mathbf { T } _ { f } ; ~ \mathbf { G } _ { f } ; ~ \mathbf { V } _ { f } ] \in \mathbb { R } ^ { B \times ( L _ { t } + N _ { j } + K ) \times d _ { f } } .\tag{8}
$$

The concatenated sequence is processed by a Transformer encoder layer [25] $( h = 8$ attention heads, FFN dimension $4 d _ { f } )$ . Self-attention over the full sequence enables crossmodal interaction among all tokens.

c) Modality Backprojection Layer: Next, the output sequence is partitioned into the original text, gesture, and vision segments. Each segment is projected back to its native feature dimension using a modality-specific linear layer, followed by a residual connection and layer normalisation:

$$
\mathbf { F } _ { l , t } ^ { \prime } = \mathrm { L N } _ { t } ( \mathbf { F } _ { l , t } + \mathbf { W } _ { t } ^ { \prime } \mathbf { T } _ { f } ^ { \prime } ) ,\tag{9}
$$

$$
\mathbf G ^ { \prime } = \mathrm { L N } _ { g } ( \mathbf G + \mathbf W _ { g } ^ { \prime } \mathbf G _ { f } ^ { \prime } ) ,\tag{10}
$$

$$
\mathbf { F } _ { v } ^ { \prime } = \operatorname { L N } _ { v } ( \mathbf { F } _ { v } + \mathbf { W } _ { v } ^ { \prime } \mathbf { V } _ { f } ^ { \prime } ) .\tag{11}
$$

This process produces enriched modality representations $\mathbf { F } _ { l , t } ^ { \prime } , \mathbf { G } ^ { \prime }$ , and $\mathbf { F } _ { v } ^ { \prime }$ that reside in a shared embedding space. Empirically, the representations exhibit stronger cross-modal alignment compared to the original modality-specific features, indicating effective interaction between the modalities.

To verify that this block actually learns a shared representation, we measured linear Centred Kernel Alignment (CKA) [26] across modality pairs before and after the transformer layer; we present the results in Section IV.

2) Cross Modal Matching Module: This module receives three streams of enriched representations from Embodied Fusion Module: updated text tokens $\mathbf { F } _ { l , t } ^ { \prime } ,$ gesture tokens $\mathbf { G } ^ { \prime } ,$ and vision tokens $F _ { v } ^ { \prime } { . }$ , as well as the pointing direction $\hat { \mathbf { d } } _ { \mathrm { { p r e d } } }$ produced by the Gestural Module.

a) Referential-Vision Cross-Attention: Text tokens $\mathbf { F } _ { l , t } ^ { \prime }$ and gesture tokens $\mathbf { G } ^ { \prime }$ are concatenated to form a unified referential token sequence $\mathbf { F } _ { \mathrm { r e f } } \in \mathbb { R } ^ { ( L _ { t } + N _ { j } ) \times 1 2 8 }$ , while the updated vision tokens are treated as object-level features. These are passed through a transformer-based cross-modal architecture as described in M3DRef-CLIP [20], consisting of alternating self-attention blocks augmented with spatial distance weights [27] and cross-attention blocks (depth = 2, $h = 4 )$ , producing a per-proposal content score $S _ { \mathrm { c o n f } } \in \mathbb { R } ^ { M }$

b) Pointing Bias.: For the late pointing bias $S _ { \mathrm { b } } ,$ the predicted ray $\hat { { \bf d } } _ { \mathrm { p r e d } }$ from the Pointing Regressor of the Gestural Module is used. Given the better-aligned arm (selected by comparing the maximum angular alignment of both arms to the proposals), the cosine similarity between $\hat { { \bf d } } _ { \mathrm { p r e d } }$ and the vector from the anchor shoulder (a) to each proposal centre $( \mathbf { c } _ { i } )$ is computed in Equation 1.

c) Adaptive Score Fusion.: The content score $( S _ { \mathrm { c o n f } } )$ and pointing bias $( S _ { \boldsymbol { \mathrm { b } } } )$ are combined via:

$$
\begin{array} { r l } & { S _ { \mathrm { f i n a l } } = w _ { 0 } \cdot S _ { \mathrm { c o n f } } + w _ { 1 } \cdot S _ { b } , \quad [ w _ { 0 } , w _ { 1 } ] } \\ & { \quad \quad = \mathrm { S o f t m a x } ( \mathbf { M L P } ( \mathbf { F } _ { \mathrm { a g g } } ) ) , } \end{array}\tag{12}
$$

where $\mathbf { F } _ { \mathrm { a g g } }$ is the global aggregate feature from the Gestural Module’s feature extraction branch, reduced to 32 dimensions and passed through a bias weigher that produces a softmax-normalised two-element weight vector. This mechanism was adopted as is from Ges3ViG. However our results show that this late fusion has little effect on GVT3D as it primarily relies on modality alignment.This then returns us our final Object Scores, indicating the likely candidate of the reference expression.

## C. Losses used and Training Method

We use the same losses as prior work [6], [20]: semantic, offset norm, offset direction, and objectness losses from PointGroup, a cross-entropy reference loss, a symmetric contrastive loss, and an L2 joint localisation loss. In addition, we introduce a pointing bootstrap loss $\mathcal { L } _ { \mathrm { b o o t s t r a p } } .$ , a cosine distance between the regressed pointing direction and the geometric shoulder-to-index ray derived from the model’s own predicted joints, applied on first 10 warm-up epochs with linear weight decay, after which it drops to zero; the regressor is trained through the grounding objective. The full model is trained end-to-end in a single stage for 40 epochs, with all losses jointly optimised from start.

TABLE I: Comparison of GVT3D without foveation against prior 3D-ERU models.
<table><tr><td rowspan="2">Model</td><td colspan="3">IoU@0.25</td><td colspan="3">IoU@0.5</td></tr><tr><td></td><td>Unique Multiple</td><td>Overall</td><td>Unique</td><td>Multiple</td><td>Overall</td></tr><tr><td colspan="7">without Gestures</td></tr><tr><td>3DVG-Transformer [27]</td><td>71.56</td><td>31.35</td><td>39.17</td><td>50.66</td><td>21.54</td><td>27.20</td></tr><tr><td>HAM [28]</td><td>67.10</td><td>25.42</td><td>33.51</td><td>48.13</td><td>16.04</td><td>22.27</td></tr><tr><td>3DJCG [29]</td><td>75.93</td><td>40.34</td><td>47.24</td><td>59.19</td><td>30.61</td><td>36.16</td></tr><tr><td>RefMask3D [30]</td><td>72.45</td><td>25.44</td><td>34.57</td><td>64.95</td><td>22.24</td><td>30.54</td></tr><tr><td>M3DRef-CLIP [20]</td><td>77.32</td><td>62.62</td><td>65.53</td><td>60.15</td><td>47.27</td><td>49.78</td></tr><tr><td colspan="7">with Gestures (ImputeRefer)</td></tr><tr><td>ScanERU [5]</td><td>71.6</td><td>31.91</td><td>39.54</td><td>52.79</td><td>23.06</td><td>28.84</td></tr><tr><td>Ges3ViG [6]</td><td>84.6</td><td>67.6</td><td>70.9</td><td>71.0</td><td>55.8</td><td>58.7</td></tr><tr><td>GVT3D w/o Fov. (Ours)</td><td>86.7</td><td>77.4</td><td>79.2</td><td>73.3</td><td>65.2</td><td>66.7</td></tr></table>

## D. System Implementation

We now describe our real-world robotic system, which uses GVT3D for 3D-ERU as shown in Figure 3. Our system comprises a TurtleBot3 Burger connected to an NVIDIA Jetson Orin Nano, which runs the grounding and navigation pipeline. An Orbbec Femto Bolt depth camera captures a 3D point cloud of the environment, including the human performing the pointing gesture.

Given the point cloud, pointing gesture, and language command, GVT3D predicts a 3D bounding box for the referred object. We then select a reachable navigation goal near the predicted target and pass it to Nav2. Using a 2D occupancy map derived from the scene point cloud, together with odometry and LiDAR feedback, Nav2 guides the robot toward the selected goal, as shown in supplementary video.

## IV. EXPERIMENTAL RESULTS

## A. Dataset & Evaluation Metrics

We evaluate on ImputeRefer [6], which extends Scan-Net [15] and ScanRefer [16] indoor scenes with imputed human avatars pointing at a target object, comprising 35,000 instances across 707 scenes. Every instance refers to a single object; the dataset contains no region-level or multi-object references. ImputeRefer’s referring expressions are shorter and more naturalistic than text-only benchmarks (∼ 10 words vs. ∼ 19 in ScanRefer), reflecting how language is underspecified when paired with a gesture. This is similar to practical instructions like “grab that box” (Figure 1). Target distance also varies across scenes.

Following prior work [6], we use the same definition of unique (one object of the target class in the scene) and multiple (multiple same-class distractors), and report IoU@0.25 and IoU@0.5 as the primary metrics, along with latency and energy consumed on the Jetson Orin Nano.

## B. Accuracy Comparison with Prior Work on 3D-ERU

We compare the performance of GVT3D against prior approaches for 3D-ERU, both with and without gestural reasoning, as shown in Table I. Methods listed without Gestures are 3D visual grounding baselines evaluated on the underlying ScanRefer scenes without gesture input; methods listed with Gestures are evaluated on the full ImputeRefer benchmark, including the imputed avatars. The results indicate that GVT3D consistently achieves higher accuracy compared to existing state of the art models.

Among approaches that do not perform gestural reasoning, M3DRef-CLIP achieves the closest performance to GVT3D, but GVT3D outperforms it by 17.7% in IoU@0.5 on the overall ImputeRefer dataset. When compared to Ges3ViG, which supports gestural reasoning but relies primarily on ray tracing for gesture interpretation and lacks cross modal feature alignment, GVT3D achieves an improvement of 8%.

The improvement is particularly pronounced in the Multiple setting, where several instances of the same object class are present in the scene. In this scenario, performance improves from 55.8% to 65.2% at IoU@0.5 (+9.4 points). This gain can be attributed to our embodied fusion module, which enables gesture tokens to interact directly with language and geometric features. Such cross modal interaction allows the model to more effectively disambiguate among multiple candidate objects and accurately identify the intended referent. This performance gain highlights the importance of both explicit cross modal feature alignment and moving beyond ray tracing approaches for gestural reasoning.

## C. Accuracy, Latency, and Energy Trade-off of GVT3D against Prior 3D-ERU Methods

All latency and energy measurements are taken on the Jetson Orin Nano, using 10 warmup scenes to reach a steady operating state, followed by averaging over the same 50 held-out scenes for every model and configuration. Figure 4 illustrates the accuracy, latency, and energy tradeoff of GVT3D at different k values, where k denotes the maximum number of object proposals retained for foveation, compared with prior 3D-ERU approaches. In the accuracy– latency comparison (Figure 4a), GVT3D is 1.75× faster than Ges3ViG even without foveation, demonstrating its runtime efficiency. Foveation reduces accuracy as k decreases. This occurs mostly in dense scenes, where many objects lie close to the pointing direction. In such scenes, the object nearest to the pointing ray is frequently not the target, because the language instruction identifies a different object in the same region. When k is small, foveation can exclude the correct object from the retained proposals, and the model cannot recover it. However at k = 10, GVT3D still outperforms Ges3ViG by 2.75% in IoU@0.5 while running 2.95× faster. Similarly, in the accuracy–energy comparison (Figure 4b), GVT3D without foveation achieves higher accuracy while consuming only 67.11% of the energy required by Ges3ViG. With foveation, GVT3D maintains higher accuracy than Ges3ViG down to k = 10, where it consumes only 29.24% of Ges3ViG’s energy. These results demonstrate that GVT3D achieves higher accuracy with lower latency and energy consumption than Ges3ViG, supporting its suitability for ondevice ERU tasks in robotic deployments.

TABLE II: Ablation studies for GVT3D with ImputeRefer dataset
<table><tr><td rowspan="2">Configuration</td><td colspan="3">IoU@0.25</td><td colspan="3">IoU@0.5</td></tr><tr><td></td><td>Unique Multiple</td><td>Overall</td><td>Unique</td><td>Multiple</td><td>Overall</td></tr><tr><td>w/o Fusion</td><td>80.1</td><td>59.0</td><td>63.2</td><td>68.0</td><td>47.8</td><td>51.6</td></tr><tr><td>w/o Ges.</td><td>65.9</td><td>27.3</td><td>34.8</td><td>57.4</td><td>22.2</td><td>29.0</td></tr><tr><td>w. fixed lang.</td><td>71.7</td><td>68.9</td><td>69.3</td><td>63.1</td><td>58.4</td><td>59.3</td></tr><tr><td>Ges3ViG [6] (w.  $\operatorname { f i x e d } \ \operatorname { l a n g } )$ </td><td>51.1</td><td>44.9</td><td>46.1</td><td>43.9</td><td>36.8</td><td>38.2</td></tr><tr><td>GVT3D w/o Fov. (Ours)</td><td>86.7</td><td>77.4</td><td>79.2</td><td>73.3</td><td>65.2</td><td>66.7</td></tr></table>

![](images/d97ef9b48e32ade05c8d10583ffc8a29fe038fc29ce02b632f36eecad4fa359e.jpg)  
(a) Accuracy vs. latency

![](images/fabf011a9b59022bb977abdfff85ca10badda0de797b010aaf4ad6f8f97e263e.jpg)  
(b) Accuracy vs. energy  
Fig. 4: Accuracy–latency–energy trade-off of GVT3D under varying foveation levels (k), measured on the Jetson Orin Nano. GVT3D even without applying any foveation remains both faster and more energy efficient than prior methods, while GVT3D with foveation achieving further gains in latency and energy efficiency.

## D. Ablation Studies:

Table II presents several ablation studies of GVT3D on the ImputeRefer dataset. In the w/o Fusion setting, we remove the embodied fusion module. Under this configuration, we observe a significant degradation in performance, with IoU@0.5 on the overall dataset dropping by 15.9%. This highlights the importance of cross modal feature alignment.

In the w/o Ges. setting, the human is removed from the scene environment, forcing GVT3D to rely entirely on language based reasoning. In this, we again observe a substantial reduction in performance, with IoU@0.5 decreasing by 38.5% on the overall dataset. This degradation is even more pronounced in the Multiple setting, where the performance drops by 43.5%. This result indicates that gestural reasoning plays a critical role in disambiguating between objects of the same class when multiple candidates are present. Finally, in the w. fixed lang. setting, where the language description for the entire dataset is replaced with a minimal expression ‘this one’, we observe that GVT3D achieves 20% higher accuracy when compared to Ges3ViG. Collectively, these ablation results demonstrate the importance of accurate cross modal fusion, as well as the complementary roles of language and gesture reasoning in 3D-ERU.

## E. Evaluation of Embodied Fusion Module

To verify cross-modal alignment, we measure Linear CKA [26] and Recall@1 before and after the fusion block. As shown in Figures 5 and 6, all modality pairs exhibit large gains: CKA increases by +0.33, +0.46, and +0.18 for vision-text, vision-gesture, and text-gesture respectively, while Recall@1 improves from near-zero to 43.6%, 19.9%, and 15.4%. This is further confirmed by Figure 7 (UMAP projections): initially separated modality clusters draw together after the fusion, with matched triplets forming tight local neighborhoods.

We additionally measured the causal contribution of the language and gesture branches to the final score under various localized sensor degradations, as shown in Figure 8. For each stream we zero it before the fusion block, replay the model’s own fusion, matching and score-blending head, and record the drop $\Delta z _ { m }$ in the z-score of the originally selected proposal within the score distribution over valid proposals; rectifying and normalizing the three drops gives a decision share $s _ { m } = [ \Delta z _ { m } ] _ { + } / \sum _ { m ^ { \prime } } [ \Delta z _ { m ^ { \prime } } ] _ { + }$ that sums to one over language, gesture and vision. Under noise localized to the target, gesture reliance increases as visual evidence degrades. Conversely, when noise is applied around the human, the gesture contribution more than halves while the language share increases from approximately 31% to 44%. This shows the GVT3D can adapt the importance of gesture and language according to context.

![](images/39e8767506bafaa83b0c49d4d73de0a50d380f5ad4d1bad0c4ba6b6cb45c95e6.jpg)

![](images/94d8ef888ab4a45116a5d6a8d6d4f098f67da6c058f6b2822b11bac7f207b10d.jpg)

Fig. 5: Cross-modal Recall@1 before and after fusion across modality pairs (200 val samples). (a) Before fusion  
Fig. 6: Linear CKA between modality pairs before and after the embodied fusion block. (b) After fusion  
![](images/6fcbfdc13c365c861aedef8e94d57c218a0705c757b373e876bb961150e07492.jpg)  
Fig. 7: UMAP of vision, text, and gesture embeddings before (a) and after (b) fusion. Lines connect matched triplets. After fusion, clusters converge and connecting lines become parallel, reflecting consistent cross-modal alignment.

## F. Human-to-Target Distance-based Sensitivity Study

To understand pointing gesture effectiveness variance with distance, we analyze model performance across proxemic zones [31]: personal (0.46–1.22m), social (1.22–3.70m), and public (≥3.70m), shown in Figure 9. Performance stays stable at personal (70.1%) and social (69.7%) zones but drops at public zone (57.3%), where gesture cues become less discriminative. We also evaluate three fixed-weight variants III-B.2.c: content only $( w _ { 0 } { = } 1 , w _ { 1 } { = } 0 )$ , gesture only $( w _ { 0 } { = } 0 , w _ { 1 } { = } 1 )$ , and equal weight $\scriptstyle ( w _ { 0 } = w _ { 1 } = 0 . 5 )$ . Content only $( S _ { c o n f } )$ performs near-identically to the full model, confirming that the model genuinely learns from all three modalities rather than relying on ray-casting. Gesture only $( S _ { b } )$ degrades significantly across all zones, demonstrating that raw pointing rays alone are insufficient for grounding.

These retrieval results confirm that the fusion block enables matched samples from different modalities to locate one another in embedding space, demonstrating that the module aligns the three modalities rather than merely mixing features indiscriminately.

![](images/51807b8b32e44787b5b60ebb6aa2351f23dd7b5ec81463ee6dc4312d928a1d8c.jpg)  
Fig. 8: Causal decision share per modality under $\sigma = 2$ cm localized point-cloud noise, with IoU@0.5.

![](images/f539e6a12674fceef6deabd09cc7bfff8c047d8da5d268a2ee0e50f3a87d648c.jpg)  
Fig. 9: Sensitivity scan for different distances: Personal, Social, Public.

## G. Success Rate Evaluation with GVT3D

We evaluated GVT3D with our robotic prototype in realworld scenes that were not part of the training data with our system as shown in Figure 3. We placed the target between 2.75 m and 9.5 m from the robot across the trials (mean 5.89 m, SD 2.28 m). Before these trials, we fine-tuned the model on 80 samples that we collected from real persons, and we replaced the synthetic persons in the ImputeRefer dataset with these samples.

Two sources of error can occur in this pipeline: (i) GVT3D grounds the wrong object, resulting in an incorrect navigation goal, and (ii) the TurtleBot fails to reach the correct goal due to navigation drift or failure. To assess (i), six participants each viewed 30 scenes showing the real-world environment and the predicted target bounding box, and judged whether the grounding was correct. Overall, GVT3D’s object localization was judged correct in 70% of instances, as compared to 56% of cases in Ges3ViG. To assess (ii), we evaluated navigation performance on the subset of trials where GVT3D grounded the correct object. Across 16 completed trials, the TurtleBot reached within a mean distance of 0.79 m (SD 0.36 m) from the target. Navigation was aborted in 2 trials due to odometry drift.

## V. CONCLUSION

We propose GVT3D, a novel approach for 3D-ERU that localises objects in 3D scenes given language and pointing gestures. Unlike prior methods that rely on heuristic ray tracing for gestural reasoning, GVT3D models gestures as a spatial prior jointly conditioned with language features. We introduce an embodied fusion mechanism that tightly aligns vision, language, and gesture representations within a shared embedding space. Experimental results show that (a) GVT3D achieves ∼8% higher accuracy without foveation and (b) foveation enables 2.95× faster execution while maintaining comparable accuracy, relative to prior 3D-ERU models.

## VI. ACKNOWLEDGMENTS

This research was supported by the Singapore National Research Foundation (NRF), Prime Minister’s Office, Singapore, under its Campus for Research Excellence and

Technological Enterprise (CREATE) programme. The Mens, Manus, and Machina (M3S) is an interdisciplinary research group (IRG) of the Singapore-MIT Alliance for Research and Technology (SMART) centre.

## REFERENCES

[1] A. Kendon, Gesture: Visible action as utterance. Cambridge University Press, 2004.

[2] Y. Chen, Q. Li, D. Kong, Y. L. Kei, S.-C. Zhu, T. Gao, Y. Zhu, and S. Huang, “Yourefit: Embodied reference understanding with language and gesture,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 1385–1395.

[3] Y. Ji, Z. Zhang, D. Tang, Y. Zheng, C. Liu, Z. Zhao, and X. Li, “Foundation models assist in human–robot collaboration assembly,” Scientific Reports, vol. 14, no. 1, p. 24828, Oct 2024.

[4] D. Weerakoon, V. Subbaraju, T. Tran, and A. Misra, “COSM2IC: Optimizing Real-Time Multi-Modal Instruction Comprehension,” IEEE Robotics and Automation Letters, vol. 7, no. 4, 2022.

[5] Z. Lu, Y. Pei, G. Wang, P. Li, Y. Yang, Y. Lei, and H. T. Shen, “ScanERU: Interactive 3D Visual Grounding Based on Embodied Reference Understanding,” in Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024. AAAI Press, 2024.

[6] A. M. Mane, D. Weerakoon, V. Subbaraju, S. Sen, S. E. Sarma, and A. Misra, “Ges3ViG: Incorporating Pointing Gestures into Language-Based 3D Visual Grounding for Embodied Reference Understanding,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[7] Y. Li, X. Chen, H. Zhao, J. Gong, G. Zhou, F. Rossano, and Y. Zhu, “Understanding embodied reference with touch-line transformer.” in ICLR, 2023.

[8] D. Weerakoon, V. Subbaraju, N. Karumpulli, T. Tran, Q. Xu, U.-X. Tan, J. H. Lim, and A. Misra, “Gesture enhanced comprehension of ambiguous human-to-robot instructions,” in Proceedings of the 2020 International Conference on Multimodal Interaction, 2020.

[9] S. Macenski, T. Moore, D. V. Lu, A. Merzlyakov, and M. Ferguson, “From the desks of ROS maintainers: A survey of modern & capable mobile robotics algorithms in the robot operating system 2,” Robotics and Autonomous Systems, 2023.

[10] R. Hu, H. Xu, M. Rohrbach, J. Feng, and K. Saenko, “Natural language object retrieval,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4555–4564.

[11] J. Deng, Z. Yang, T. Chen, W. Zhou, and H. Li, “Transvg: End-to-end visual grounding with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021.

[12] A. Kamath, M. Singh, Y. LeCun, G. Synnaeve, I. Misra, and N. Carion, “Mdetr-modulated detection for end-to-end multi-modal understanding,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 1780–1790.

[13] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[14] L. Jia, M. Chang, Y. Liu, and N. Du, “How humans naturally refer to targets: Understanding multimodal instruction patterns in human–robot interaction,” in Proceedings of the CHI Conference on Human Factors in Computing Systems, ser. CHI ’26, 2026.

[15] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828–5839.

[16] D. Z. Chen, A. X. Chang, and M. Nießner, “Scanrefer: 3d object localization in rgb-d scans using natural language,” in European conference on computer vision. Springer, 2020, pp. 202–221.

[17] P. Achlioptas, A. Abdelreheem, F. Xia, M. Elhoseiny, and L. Guibas, “ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes,” in 16th European Conference Computer Vision–ECCV. Springer, 2020, pp. 422–440.

[18] Z. Yuan, X. Yan, Y. Liao, R. Zhang, S. Wang, Z. Li, and S. Cui, “Instancerefer: Cooperative holistic understanding for visual grounding on point clouds through instance multi-level contextual referring,” in IEEE/CVF International Conference on Computer Vision, 2021.

[19] D. He, Y. Zhao, J. Luo, T. Hui, S. Huang, A. Zhang, and S. Liu, “TransRefer3D: Entity-and-Relation Aware Transformer for Fine-Grained 3D Visual Grounding,” in Proceedings of the 29th ACM international conference on multimedia, 2021, pp. 2344–2352.

[20] Y. Zhang, Z. Gong, and A. X. Chang, “Multi3drefer: Grounding text description to multiple 3d objects,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023.

[21] R. Zhang, Z. Guo, W. Zhang, K. Li, X. Miao, B. Cui, Y. Qiao, P. Gao, and H. Li, “Pointclip: Point cloud understanding by clip,” arXiv preprint arXiv:2112.02413, 2021.

[22] L. Jiang, H. Zhao, S. Shi, S. Liu, C.-W. Fu, and J. Jia, “Pointgroup: Dual-set point grouping for 3d instance segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and Pattern recognition, 2020, pp. 4867–4876.

[23] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A. A. Osman, D. Tzionas, and M. J. Black, “Expressive body capture: 3d hands, face, and body from a single image,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019.

[24] C. Lin, C. Li, Y. Liu, N. Chen, Y.-K. Choi, and W. Wang, “Point2skeleton: Learning skeletal representations from point clouds,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 4277–4286.

[25] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” 2023. [Online]. Available: https://arxiv.org/abs/1706.03762

[26] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in Proceedings of the International Conference on Machine Learning, vol. 97. PMLR, 2019.

[27] L. Zhao, D. Cai, L. Sheng, and D. Xu, “3DVG-Transformer: Relation modeling for visual grounding on point clouds,” in ICCV, 2021.

[28] J. Chen, W. Luo, R. Song, X. Wei, L. Ma, and W. Zhang, “Learning point-language hierarchical alignment for 3d visual grounding,” 2023. [Online]. Available: https://arxiv.org/abs/2210.12513

[29] D. Cai, L. Zhao, J. Zhang, L. Sheng, and D. Xu, “3djcg: A unified framework for joint dense captioning and visual grounding on 3d point clouds,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 464–16 473.

[30] S. He and H. Ding, “RefMask3D: Language-guided transformer for 3D referring segmentation,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 8316–8325.

[31] E. T. Hall, The Hidden Dimension. Garden City, NY: Doubleday, 1966.