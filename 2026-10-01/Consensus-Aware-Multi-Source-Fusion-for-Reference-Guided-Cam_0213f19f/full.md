# Consensus-Aware Multi-Source Fusion for Reference-Guided Camouflaged Object Detection

Junyang Xia<sup>†</sup>, Luocheng Zhang<sup>†</sup>, Wenwen Pan<sup>∗</sup>, Chifeng Zhu, Yang Yang, Xinchun Liu, Jiajun Ding

School of Computer Science, Hangzhou Dianzi University, Hangzhou 310018, China

† These authors contributed equally to this work. ∗ Corresponding author: wenwenpan@hdu.edu.cn

## Abstract

Reference-guided camouflaged object detection aims to segment a target whose visual appearance closely resembles its surroundings by exploiting auxiliary reference samples. The task remains difficult because reference samples contain inconsistent target cues, while generic visual representations are not inherently aligned with the target specified by the references. To handle these problems, we present a consensus-aware multi-source fusion framework. Reference-Conditioned Dual-Backbone Fusion (RCDF) couples trainable PVTv2 query features with frozen DINOv3 representations and uses reference-conditioned correlation to select foundation-model evidence before multi-scale fusion. The framework also aggregates multiple references through cross-reference consensus aggregation and injects reference information at semantic depths matched to the query features. Extensive experiments demonstrate the effectiveness of the proposed method. The results further show that reference consensus, target-conditioned foundation features, and hierarchical decoding provide complementary improvements under the evaluation protocol. The source code will be made publicly available upon acceptance.

Keywords: Camouflaged object detection; reference-guided segmentation; feature fusion; visual foundation models

## 1 Introduction

Camouflaged object detection (COD) aims to identify and segment objects whose appearance blends into the surrounding scene [10]. It supports species discovery and concealed-scene understanding [11, 12], microscopic medical analysis [18, 19], agricultural pest inspection in complex backgrounds [5], industrial surface-defect inspection [20], and visual search in safety-critical settings [11]. Conventional COD discovers plausible concealed foreground regions from a query image, whereas many practical searches focus on a specific target category, such as a particular species, pest, or defect class. Reference-guided camouflaged object detection (Ref-COD) addresses this need by using a small set of target examples to specify what should be segmented [45]. Existing Ref-COD methods typically encode the reference set into a shared target representation and transfer it to the query branch through feature aggregation, semantic guidance, or cross-level alignment [4, 34, 45]. Despite this progress, Ref-COD still faces three main challenges. First, when a query contains multiple camouflaged objects or visually similar distractors, the model must identify and segment only the target category specified by the references; generic visual features alone may activate on semantically unrelated regions [4, 45]. Second, weak contrast, small target regions, thin structures, and ambiguous boundaries make fine-grained detail recovery difficult, particularly when reference information is injected without regard to the semantic depth of query features [15, 30, 34]. Third, reference samples can differ substantially in pose, scale, appearance, and quality, so treating them as equally reliable may propagate inconsistent or misleading target cues [34, 45].

![](images/ffa7da58fbe52c78686d5e115bda6dcf42cb3d7946429b5f83b37f691c34ef48.jpg)  
Figure 1: Representative results for reference-guided camouflaged object detection. Columns show the input, ground truth (GT), predictions from UAT, R2CNet, Qwen+SAM, and our method. The displayed S<sub>m</sub> values accompany examples with low contrast, thin structures, and multiple targets.

Existing methods have addressed these challenges to different extents. First, for target-specific localiza tion, R2CNet derives a common representation from visual references and generates a pixel-level prior mask through dense reference–query comparison [45]; UAT aligns visual reference and query features through cross-attention [34]; and MLKG and CGCOD strengthen target semantics using multi-level language knowl edge and class prompts, respectively [4, 42]. These strategies improve semantic guidance, but they either summarize visual references into a common representation or rely on textual and class-level cues, with out using visual references to select target-relevant evidence from frozen foundation features. Second, for detail recovery, R2CNet enriches multi-scale query features under the guidance of its referring mask [45], while UAT combines adjacent-layer semantics with a cross-attention encoder and models predictive uncer tainty in its decoder [34]. However, neither method explicitly matches spatial and vector reference priors to query stages according to semantic granularity or maintains reference-aware guidance throughout progressive decoding. Third, for reference reliability, R2CNet and UAT aggregate information from multiple visual references, yet neither explicitly estimates sample-wise reliability at multiple semantic levels [34, 45]. Thus, target-conditioned foundation-feature selection, depth-matched detail reconstruction, and reliability-aware multi-reference aggregation remain unresolved within a unified framework.

To address these limitations, we propose a consensus-aware multi-source fusion framework, which contains three complementary components: Reference-Conditioned Dual-Backbone Fusion (RCDF), hierarchical reference fusion and decoding (HRFD), and cross-reference consensus aggregation (CRCA). RCDF addresses target ambiguity in multi-object or distractor-rich scenes by forming a target descriptor from foreground-aware reference vectors and using it to select relevant frozen DINOv3 features before fusion with trainable PVTv2 query features. HRFD improves fine-grained detail recovery by injecting spatial and vector priors at matched semantic depths and progressively reconstructing target regions under region, auxiliary, and edge supervision. CRCA suppresses unreliable reference cues by learning reliability-weighted deep spatial, middle spatial, and global-vector priors from heterogeneous references. Figure 2 presents the complete architecture, while Figure 1 illustrates the resulting advantages in representative scenes. In particular, the multi-object example shows that our method recovers multiple target instances more completely while suppressing unrelated regions, highlighting its ability to preserve target-specific localization in complex scenes.

The contributions of this work are summarized as follows.

• We introduce RCDF, which couples trainable PVTv2 multi-scale features with frozen DINOv3 representations and uses reference-conditioned correlation to select target-relevant foundation-model evidence.

• CRCA estimates reference reliability to construct spatial and vector consensus priors from heterogeneous reference samples.

• We match spatial cross-attention and vector modulation to different query depths and progressively decode the conditioned features; a full-factorial evaluation verifies the individual and complementary effects of CRCA, RCDF, and HRFD.

## 2 Related Work

## 2.1 Conventional camouflaged object detection

Conventional COD methods infer concealed regions from a single query image and differ mainly in how they expose weak foreground evidence. Bio-inspired coarse-to-fine search and iterative refinement progressively localize difficult targets [16, 24, 44], while multi-scale and cross-level architectures exchange local detail with high-level context [2, 26, 27]. Boundary- and gradient-oriented models make structural discontinuities explicit through edge guidance, separated attention, feature decomposition, or gradient supervision [13, 15, 30, 46]. Relation and ranking objectives further organize foreground–background interactions, object difficulty, and hierarchical tokens [21, 38, 41]. These families improve localization and contour recovery, but their target definition still comes entirely from the query image.

Other methods expand the evidence available to a query-only model. Frequency-aware networks separate or entangle spectral and spatial cues [6, 29]; auxiliary depth can reveal object pop-out and support foundation-model adaptation [36, 40]; collaborative learning exploits related images [43]; and probabilistic or generative formulations model ambiguous regions and mask distributions [3, 37]. Transformer decoders also strengthen long-range reasoning with shrinkage pyramids or masked separable attention [14, 39]. Sur veys and diagnostic studies summarize this progression from local appearance modeling toward richer structure and auxiliary cues [10, 22]. Nevertheless, these cues generally help identify camouflaged foreground as a whole; they do not specify which concealed category a user intends to retrieve.

## 2.2 Reference and auxiliary guidance

Guided COD introduces information outside the query to resolve target ambiguity. Collaborative COD learns shared cues among semantically related images [43], whereas class-guided COD uses category knowledge as a semantic constraint [42]. Ref-COD provides a more direct visual specification. R2CNet aggregates common representations from a set of salient reference objects and transfers them to a query branch [45]. MLKG supplements alignment with progressively organized language knowledge from a multimodal large model [4], while UAT combines referring-feature aggregation, cross-level attention, and probabilistic token modeling [34]. These approaches demonstrate the value of target guidance, yet their auxiliary evidence is primarily summarized as a common or globally aligned signal. Reference sets can contain complementary local details as well as distractors; preserving this distinction requires both sample-wise reliability estimation and spatially resolved consensus.

More broadly, guided COD systems combine query hierarchies, visual references, external semantic features, and structural cues through concatenation, attention, or global aggregation. However, a single fusion operation does not resolve two distinct questions: which reference samples are trustworthy, and at what semantic depth should their information enter the query stream? Existing methods do not jointly model sample-wise reference reliability, spatially resolved consensus, and depth-matched reference injection, leaving these complementary aspects insufficiently explored.

## 2.3 Transformers and visual foundation features

The Transformer replaces recurrence with attention-based token interaction [31], and the Vision Transformer applies the same principle to image patches [7]. Pyramid Vision Transformer and PVTv2 adapt this representation to dense prediction through hierarchical, multi-resolution features [32, 33]. COD models build on this hierarchy to reason globally while retaining progressively decoded detail [14, 37, 39]. In parallel, self-supervised DINO reveals semantic organization in attention maps [1], and DINOv2 and DINOv3 scale transferable visual features while emphasizing dense representation quality [25, 28]. Segment Anything provides a promptable segmentation foundation model [17], and depth-enhanced adaptations illustrate how additional task cues can specialize such generic representations for COD [40]. Broad semantic coverage, however, does not itself identify the category designated by a particular reference set. The relevant founda tion tokens must be selected with target-conditioned evidence.

## 3 Method

Given a query image $\mathbf { I } _ { \mathbf { q } } \in \mathbf { R } ^ { \mathbf { C } _ { \mathbf { q } } \times \mathbf { H } _ { \mathbf { q } } \times \mathbf { W } _ { \mathbf { q } } }$ , reference-guided camouflaged object detection seeks to identify only the camouflaged objects specified by a set of visual references, rather than all plausible camouflaged foreground regions in the query [34, 45]. Let ${ \sf R } = \{ { \sf I } _ { \mathrm { r } } ^ { \mathrm { k } } \} _ { \mathrm { k } = 1 } ^ { \mathrm { K } }$ denote the reference set, where each $\mathbf { I } _ { \mathrm { r } } ^ { \mathrm { k } } \in \mathbf { R } ^ { \mathbf { \overline { { C } } _ { \mathrm { r } } \times H _ { r } \times W _ { l } } }$ contains a salient object belonging to the same target category c. The query image may contain one or more camouflaged instances of category c amid visually similar background regions. The desired output is a binary mask $\mathrm { Y } \in \{ 0 , 1 \} ^ { \mathrm { H _ { q } \times W _ { q } } }$ , where $\mathrm { Y } _ { \mathrm { u , v } } = 1$ if pixel (u, v) belongs to a target instance of category c and $\mathrm { Y } _ { \mathrm { u , v } } = 0$ otherwise. We formulate the prediction as

$$
\begin{array} { r } { \mathbf { P } = \mathbf { f } _ { \theta } ( \mathbf { I } _ { \mathrm { q } } , \mathbf { R } ) , \qquad \mathbf { P } \in [ 0 , 1 ] ^ { \mathrm { H } _ { \mathrm { q } } \times \mathbf { W } _ { \mathrm { q } } } , } \end{array}\tag{1}
$$

where $\mathrm { f _ { \theta } }$ is the learnable Ref-COD model and P is the predicted foreground-probability map. During training, P is supervised by Y together with the auxiliary and boundary targets defined later in this section. During inference, the references specify the semantic target and the model predicts its spatial extent in the query image.

Figure 2 illustrates the overall architecture of the proposed framework. Our framework implements $\mathrm { f _ { \theta } }$ through consensus-aware multi-source fusion. A trainable PVTv2 encoder and a frozen DINOv3 encoder form the dual backbones of RCDF, producing hierarchical query features and patch-token features, respectively. ICON-R supplies a deep spatial feature, a middle spatial feature, and a global vector for each reference. RCDF averages the unprojected global reference vectors and uses the resulting target descriptor to gate DINOv3 features before multi-scale fusion with the PVTv2 features. CRCA independently aggregates the projected reference features, whose spatial and vector consensus priors subsequently condition the fused query features at matched semantic depths.

![](images/bceb6f0c120349576cc2b56f27390ed49994f4784b27ba66a0d142bcddfac1a5.jpg)  
Figure 2: Overview of the consensus-aware multi-source fusion framework. CRCA aggregates heterogeneous references into consensus priors. RCDF couples the trainable PVTv2 and frozen DINOv3 backbones by using reference-conditioned correlation to gate foundation features before multi-scale fusion. The HRFD stage performs spatial cross-attention at deep stages, vector-based modulation at shallow stages, and HMU decoding to produce the final mask and auxiliary predictions.

## 3.1 Reference feature extraction

We use ICON-R, a frozen salient-reference encoder based on the Integrity Cognition Network (ICON) [47]. The original ICON decoder predicts an integrity-aware saliency map. Our reference wrapper additionally exposes deep and intermediate encoder features and uses the predicted soft foreground mask to suppress background responses. For reference image $\mathrm { I _ { r } ^ { k } }$ , let $\mathbf { M } ^ { \mathbf { k } } = \sigma ( \mathrm { S } _ { \mathrm { I C O N } } ( \mathrm { I } _ { \mathrm { r } } ^ { \mathbf { k } } ) )$ be the soft saliency mask, and let $\mathrm { E _ { d } ^ { k } }$ and $\mathrm { E _ { m } ^ { k } }$ denote the deep and intermediate encoder features. We compute

$$
\mathbf { R } _ { \mathrm { d } } ^ { \mathrm { k } } = \mathbf { E } _ { \mathrm { d } } ^ { \mathrm { k } } \odot \mathbf { D } _ { \mathrm { d } } ( \mathbf { M } ^ { \mathrm { k } } ) , \qquad \mathbf { R } _ { \mathrm { m } } ^ { \mathrm { k } } = \mathbf { E } _ { \mathrm { m } } ^ { \mathrm { k } } \odot \mathbf { D } _ { \mathrm { m } } ( \mathbf { M } ^ { \mathrm { k } } ) ,\tag{2}
$$

where $\mathrm { D } _ { \mathrm { j } }$ bilinearly resizes the mask to the corresponding feature resolution. A masked average of the deep feature provides the global reference vector:

$$
\mathbf { r } _ { \mathrm { v } } ^ { \mathrm { k } } = \frac { \sum _ { \mathrm { u } } \mathbf { D } _ { \mathrm { d } } ( \mathbf { M } ^ { \mathrm { k } } ) _ { \mathrm { u } } \mathbf { E } _ { \mathrm { d , u } } ^ { \mathrm { k } } } { \sum _ { \mathrm { u } } \mathbf { D } _ { \mathrm { d } } ( \mathbf { M } ^ { \mathrm { k } } ) _ { \mathrm { u } } + \varepsilon } .\tag{3}
$$

These foreground-aware representations retain localized structure and category-level semantics for consensus aggregation.

## 3.2 Cross-reference consensus aggregation

To address the third challenge, namely the heterogeneous quality and reliability of reference samples, we propose Cross-Reference Consensus Aggregation (CRCA) to construct robust target priors. Instead of assigning equal importance to references that may differ in pose, scale, appearance, or foreground clarity, CRCA explicitly models their interactions and learns how much each sample should contribute to the shared target representation. As illustrated in Figure 3, CRCA integrates a Reference Reliability Estimator (RRE) with Reference-Guided Fusion (RGF). RRE captures agreement and disagreement across references, while RGF uses the estimated reliability to preserve consistent target evidence and reduce the influence of less informative samples.

![](images/7f4740e39ddefb0d83a14048396c51a65bbbc7b6426032e83517c4d52dd2976c.jpg)  
Figure 3: Internal structure of cross-reference consensus aggregation. The Reference Reliability Estimator (RRE) applies self-attention across the reference dimension and predicts reliability scores after normalization and projection. Reference-Guided Fusion (RGF) then uses the normalized weights $\{ \alpha _ { \mathrm { k } } \} _ { \mathrm { k } = 1 } ^ { \mathrm { K } }$ to aggregate the reference features $\{ \mathsf { R } _ { \mathrm { k } } \} _ { \mathrm { k } = 1 } ^ { \mathrm { K } }$ into the consensus representation R.<sup>¯</sup>

Given a projected reference tensor $\mathbf { \boldsymbol { X } } \in \mathbf { \boldsymbol { R } } ^ { \mathbf { B } \times \mathbf { K } \times \mathbf { C } \times \mathbf { H } \times \mathbf { W } }$ , RRE treats the K references at each spatial position as a sequence. It reshapes X into BHW sequences of length K and applies multi-head self-attention followed by a residual projection. Let $\widetilde { \mathrm { X } } _ { \mathrm { k } }$ denote the interaction-aware representation of reference k. A learned quality head produces the normalized reliability weights

$$
\alpha _ { \mathrm { k } } = \mathrm { s o f t m a x } _ { \mathrm { k } } \left( \mathbf { g } \left( \mathbf { L N } ( \widetilde { \mathbf { X } } _ { \mathrm { k } } ) \right) \right) , \qquad \bar { \mathbf { X } } = \sum _ { \mathrm { k } = 1 } ^ { \mathrm { K } } \alpha _ { \mathrm { k } } \widetilde { \mathbf { X } } _ { \mathrm { k } } ,\tag{4}
$$

where g denotes the quality head. RGF then forms the weighted consensus $\bar { \mathbf X }$ in $\operatorname { E q . }$ 4. We apply the proposed aggregation jointly to deep spatial features, middle spatial features, and projected reference vectors, producing the multi-granularity consensus priors $\bar { \mathsf { R } } _ { \mathrm { d } } , \bar { \mathsf { R } } _ { \mathrm { m } } ,$ , and $\bar { \mathsf { R } } _ { \mathrm { v } }$ . By estimating reliability separately at these semantic levels, CRCA directly resolves the third challenge: it preserves localized structure and category-level semantics that are consistent across references while limiting the propagation of unreliable sample-specific cues.

## 3.3 Reference-conditioned fusion

The first challenge is target ambiguity in multi-object and distractor-rich scenes. To address it, Reference-Conditioned Dual-Backbone Fusion (RCDF) couples the trainable PVTv2 query backbone with the frozen DINOv3 foundation backbone. The two encoders provide complementary representations: PVTv2 learns task-specific hierarchical features that retain local structures, whereas DINOv3 supplies semantically discriminative patch representations learned from large-scale pretraining. Generic foundation features alone cannot distinguish target instances from semantically unrelated concealed objects, and direct fusion may therefore transfer strong but target-irrelevant responses. RCDF uses the references to select target-consistent DINOv3 evidence before it enters the multi-scale query hierarchy.

Let $\begin{array} { r } { \mathbf { r } = \mathbf { K } ^ { - 1 } \sum _ { \mathrm { k } = 1 } ^ { \mathrm { K } } \mathbf { r } _ { \mathrm { v } } ^ { \mathrm { k } } } \end{array}$ be the mean unprojected foreground-aware reference vector and let $\{ \mathrm { d } _ { \mathrm { i } } \} _ { \mathrm { i } = 1 } ^ { \mathrm { N } }$ be the DINOv3 patch tokens. Averaging the foreground-aware vectors summarizes the target semantics shared by the references, while retaining the original ICON-R representation space for correlation estimation. RCDF maps the reference descriptor and patch tokens to a common embedding space, applies $\ell _ { 2 }$ normalization, and computes a sigmoid correlation score for each patch:

$$
\mathrm { c } _ { \mathrm { i } } = \sigma \left( \tau \frac { \ \Phi _ { \mathrm { r } } ( \mathbf { r } ) ^ { \top } \Phi _ { \mathrm { d } } ( \mathbf { d } _ { \mathrm { i } } ) } { \lVert \Phi _ { \mathrm { r } } ( \mathbf { r } ) \rVert _ { 2 } \lVert \Phi _ { \mathrm { d } } ( \mathbf { d } _ { \mathrm { i } } ) \rVert _ { 2 } } \right) ,\tag{5}
$$

where $\Phi _ { \mathrm { r } }$ and $\Phi _ { \mathrm { d } }$ are learned linear projections and τ is a learnable scale controlling the sharpness of target selection. The scores $\{ \mathrm { c } _ { \mathrm { i } } \} _ { \mathrm { i } = 1 } ^ { \mathrm { N } }$ are rearranged according to the DINOv3 patch grid to form a spatial correlation map C. A high response indicates that the local query representation is semantically consistent with the reference target, whereas a low response attenuates foundation features associated with background regions or unrelated objects.

For the j-th PVTv2 stage, let $\mathrm { Q _ { j } }$ denote the projected query feature and let D denote the DINOv3 spatial feature reconstructed from its patch tokens. We resize D and C to the spatial resolution of $\mathrm { Q _ { j } } ,$ , project the resized foundation feature to the common model dimension, and perform gated convolutional fusion as

$$
\mathrm { F _ { j } } = \psi _ { \mathrm { j } } \left( \left[ \mathrm { Q } _ { \mathrm { j } } , \pi _ { \mathrm { j } } \left( \mathrm { I } _ { \mathrm { j } } ( \mathrm { D } ) \right) \odot \mathrm { I } _ { \mathrm { j } } ( \mathrm { C } ) \right] \right) , \qquad \mathrm { j } \in \{ 0 , 1 , 2 , 3 \} ,\tag{6}
$$

where $\mathrm { I _ { j } }$ denotes bilinear interpolation to the j-th feature resolution, $\pi _ { \mathrm { j } }$ is the corresponding DINOv3 feature projection, [·,·] denotes channel-wise concatenation, ⊙ denotes element-wise multiplication, and $\psi _ { \mathrm { j } }$ is a stage-specific convolutional fusion block. Applying the reference-conditioned map at every scale allows deep features to retain target-level semantics while shallow features preserve spatial details needed for subsequent decoding.

RCDF performs patch-wise selection rather than collapsing the query into a single target location. Consequently, spatially disconnected regions can receive high correlation scores whenever they share the semantics specified by the references. This property is particularly important in multi-object scenes: multiple camouflaged instances can be retained simultaneously, while visually salient but reference-inconsistent regions are suppressed before hierarchical reference interaction and decoding. The frozen DINOv3 branch provides stable semantic evidence, and the trainable PVTv2 and fusion blocks adapt this evidence to the localization requirements of Ref-COD. RCDF thus resolves the first challenge by converting a generic foundation representation into target-conditioned, spatially distributed evidence suitable for both multi-instance recovery and distractor suppression.

## 3.4 Hierarchical reference fusion and decoding

To address the second challenge of recovering small regions, thin structures, and ambiguous boundaries, we design hierarchical reference fusion and decoding (HRFD) to inject each consensus prior at a query depth that matches its semantic granularity. Uniform reference injection can impose coarse spatial correspondence on high-resolution features or provide insufficient localization cues to deep features, which weakens the reconstruction of fine target details. HRFD therefore uses spatial reference priors for deep semantic interaction and the global vector prior for shallow feature modulation, allowing target structure and category semantics to complement each other throughout decoding.

At the deeper levels, the fused query features attend to $\bar { \mathsf { R } } _ { \mathrm { d } }$ and $\bar { \mathsf { R } } _ { \mathrm { m } }$ , respectively. Our cross-attention block projects query, key, and value tokens, applies a scaled residual connection to the attended output, and follows it with a feed-forward residual block. At the shallow levels, HRFD uses broadcast attention adaptation (BAA) with $\bar { \mathsf { R } } _ { \mathrm { v } }$ to modulate high-resolution features without forcing a coarse spatial correspondence.

To maintain target awareness during progressive decoding, HRFD generates a relevance mask from the deepest feature and $\bar { \mathsf { R } } _ { \mathrm { v } }$ and appends it to every decoding stage. Hierarchical mixed-scale units (HMUs) decode the conditioned features in a top-down manner, while ASPP enriches the deepest query projection. This coordinated design produces a final prediction $\mathrm { { P } , }$ auxiliary predictions $\{ \mathrm { P _ { j } } \} _ { \mathrm { j = 1 } } ^ { \mathrm { J } }$ , and an edge prediction

E. The training objective is

$$
{ \displaystyle { \bf L } = { \bf L } _ { \mathrm { s t r } } ( { \bf P } , { \bf Y } ) + \lambda _ { 1 } \sum _ { \mathrm { j } = 1 } ^ { \mathrm { J } } { \bf L } _ { \mathrm { s t r } } ( { \bf P } _ { \mathrm { j } } , { \bf Y } ) + \lambda _ { 2 } { \bf L } _ { \mathrm { e d g e } } ( { \bf E } , { \bf Y } _ { \mathrm { e d g e } } ) } ,\tag{7}
$$

The depth-matched interactions preserve target semantics during feature refinement, while progressive decoding and boundary supervision recover local structures at increasing spatial resolutions. HRFD therefore resolves the second challenge by coordinating semantic guidance, multi-scale reconstruction, and explicit boundary learning instead of treating detail recovery as an isolated final-stage operation.

## 4 Experiments

## 4.1 Dataset and evaluation metrics

We conduct all experiments on R2C7K [45], a large-scale benchmark for reference-guided camouflaged object detection. R2C7K contains 6,615 images spanning 64 object categories, including 1,600 salient reference images and 5,015 camouflaged query images. Each category provides 25 reference images collected from real-world scenes. Following the official protocol, 20 reference images per category are assigned to training and the remaining five to testing. The camouflaged-image training set includes the training samples from COD10K, while evaluation uses the corresponding test samples from CAMO and NC4K. The test set is reported as Overall and is further divided into Single-object and Multi-object subsets according to the number of camouflaged objects present in each scene.

We evaluate predictions using S-measure $( \mathbf { S } _ { \mathrm { m } } )$ [8], adaptive E-measure (αE) [9], weighted F-measure (wF) [23], and mean absolute error (MAE). Higher values indicate better performance for $\mathbf { S } _ { \mathrm { m } } .$ , αE, and wF, whereas a lower MAE is preferred.

## 4.2 Implementation details

Our implementation uses PVTv2-B2 as the trainable query encoder, DINOv3 ViT-B/16 as the frozen foundation encoder, and the ResNet-50 variant of ICON-R as the frozen reference encoder. Reference images are resized to 352 × 352 for ICON-R, whereas query images are resized to 384 × 384 for PVTv2 and DINOv3 during both training and testing. The model dimension is 64, and optimization uses SGD with momentum 0.9 and weight decay $5 \times 1 0 ^ { - 4 }$ . Training uses 40 epochs, a batch size of 8, an initial learning rate of 0.05 for non-backbone parameters and 0.005 for backbone parameters, three warm-up epochs, and linear decay. The selected configuration uses two deep and two shallow interaction levels, four HMUs, three auxiliary predictions, an auxiliary-loss weight of 0.5, an edge-loss weight of 0.2, a cross-attention residual scale of 0.3, four cross-attention heads, a correlation-scale initialization of 20, eight HMU groups with a hidden dimension of 48, a mask-guidance coefficient of 0.25, and ASPP dilation rates of (3,6,9). We use a 5-shot setting during both training and testing: five reference samples are randomly selected for each query during training, and five references are used for each query at inference. The frozen ICON-R encoder processes all reference images offline before model training; its deep, middle, and vector features are stored as .npz files and reused unchanged during training and testing. No test-time augmentation (TTA) is used for the reported comparison, ablation, or parameter-sensitivity results.

## 4.3 Quantitative comparison

Table 1 compares five methods on the overall, single-object, and multi-object subsets. On the overall set, our complete model achieves the best results across all four metrics, with 0.895 $\mathbf { S } _ { \mathrm { m } } .$ , 0.941 αE, 0.825 wF, and 0.017 MAE. Compared with the strongest competing method, RefOnce [35], these results improve $\mathrm { \Delta S _ { m } , }$ αE, and wF by 0.005, 0.004, and 0.006, respectively, while reducing MAE by 0.002. The consistent changes across structural similarity, alignment, weighted foreground quality, and pixel error indicate that the improvement is not confined to a single evaluation criterion.

The subset results further reveal the behavior under different scene complexities. On the single-object subset, our method obtains 0.899 $\mathbf { S } _ { \mathrm { m } } .$ , 0.943 αE, 0.830 wF, and 0.016 MAE. The multi-object subset is more challenging for all compared methods, yet our method retains the strongest $\mathrm { S } _ { \mathrm { m } } ,$ wF, and MAE and ties RefOnce for the best αE. Relative to RefOnce on this subset, it improves $\mathbf { S } _ { \mathrm { m } }$ by 0.006 and wF by 0.013 while reducing MAE by 0.003. This larger gain in wF is consistent with the intended role of RCDF: reference-conditioned patch selection retains spatially separated target instances while suppressing reference-inconsistent regions. CRCA and HRFD complement this selection by stabilizing the reference cues and refining target structures during decoding.

Among methods using PVTv2-B2 as the query backbone, our model improves upon UAT by 0.040 $\mathrm { S } _ { \mathrm { m } } , 0 . 0 2 9 \ \alpha \mathrm { E } .$ , and 0.068 wF on the overall set, together with a 0.009 reduction in MAE. On the multiobject subset, the corresponding improvements are 0.054, 0.030, and 0.079, with MAE reduced by 0.012. These comparisons suggest that the gains arise from how reference evidence is aggregated, used to condition foundation features, and injected across semantic depths rather than from the query backbone alone. The Qwen+SAM baseline combines Qwen3-VL with SAM2, but its general-purpose semantic and segmentation capabilities do not provide the same target-specific consistency in these camouflaged scenes.

Table 1: Quantitative comparison on the overall, single-object, and multi-object subsets. Higher values are better for $\mathbf { S } _ { \mathrm { m } } .$ , αE, and wF; lower values are better for MAE.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">Overall</td><td colspan="4">Single-object</td><td colspan="4">Multi-object</td></tr><tr><td> $\mathrm { { S } _ { m } \uparrow }$ </td><td>αE↑</td><td>wF↑</td><td>MAE↓</td><td> $\mathrm { { S } _ { m } \uparrow }$ </td><td>αE↑</td><td> $\mathrm { w F \uparrow }$ </td><td>MAE↓</td><td> $\mathrm { { S } _ { m } \uparrow }$ </td><td>αE↑</td><td>wF↑</td><td>MAE↓</td></tr><tr><td>R2CNet</td><td>ResNet-50</td><td>0.805</td><td>0.879</td><td>0.669</td><td>0.036</td><td>0.810</td><td>0.880</td><td>0.674</td><td>0.035</td><td>0.747</td><td>0.872</td><td>0.602</td><td>0.046</td></tr><tr><td>UAT</td><td>PVTv2-B2</td><td>0.855</td><td>0.912</td><td>0.757</td><td>0.026</td><td>0.859</td><td>0.913</td><td>0.761</td><td>0.025</td><td>0.805</td><td>0.900</td><td>0.701</td><td>0.033</td></tr><tr><td>Qwen+SAM</td><td></td><td>0.852</td><td>0.917</td><td>0.793</td><td>0.029</td><td>0.861</td><td>0.926</td><td>0.805</td><td>0.027</td><td>0.778</td><td>0.844</td><td>0.683</td><td>0.047</td></tr><tr><td>RefOnce</td><td>PVTv2-B2</td><td>0.890</td><td>0.937</td><td>0.819</td><td>0.019</td><td>0.894</td><td>0.937</td><td>0.825</td><td>0.018</td><td>0.853</td><td>0.930</td><td>0.767</td><td>0.024</td></tr><tr><td>Ours / All components</td><td>PVTv2-B2</td><td>0.895</td><td>0.941</td><td>0.825</td><td>0.017</td><td>0.899</td><td>0.943</td><td>0.830</td><td>0.016</td><td>0.859</td><td>0.930</td><td>0.780</td><td>0.021</td></tr></table>

## 4.4 Qualitative comparison

Figure 4 compares R2CNet, UAT, Qwen+SAM, and our method on representative query–reference pairs. The examples cover low-contrast targets, textured backgrounds, and thin structures. Across these cases, our method produces more coherent target masks while suppressing background responses that remain in the comparison outputs. This qualitative evidence complements the quantitative results in Table 1.

## 4.5 Factorial component ablation

Table 2 reports a full factorial ablation over CRCA, RCDF, and hierarchical reference fusion and decoding (HRFD). Each component improves the baseline when introduced independently. CRCA increases the overall $\mathbf { S } _ { \mathrm { m } }$ and wF from 0.8550 and 0.7570 to 0.8610 and 0.7660, respectively; RCDF produces the largest individual gains, reaching $0 . 8 7 2 0 \mathrm { S } _ { \mathrm { m } }$ and 0.7850 wF; and HRFD reaches 0.8660 $\mathbf { S } _ { \mathrm { m } }$ and 0.7740 wF. Pairwise combinations improve further, with RCDF+HRFD giving the strongest two-component result across all four overall metrics. The complete model achieves 0.8950 $\mathrm { { \cal S } _ { m } , }$ 0.9410 αE, 0.8250 wF, and 0.0170 MAE, improving the baseline by 0.0400, 0.0290, and 0.0680 on the three higher-is-better metrics while reducing MAE by 0.0090. The consistent improvements on both the single-object and multi-object subsets indicate that the three components provide complementary evidence rather than redundant refinements.

![](images/5fb4d3e0c99b83d39d257b72251fb09342f007455a933dd5f7eb2a376a97f0b9.jpg)  
Figure 4: Qualitative comparison on representative query–reference pairs. From top to bottom, the rows show the query image, reference image, ground truth, and predictions from R2CNet, UAT, Qwen+SAM, and our method.

## 4.6 Ablation study

Table 3 reports three representative one-factor parameter studies covering the supervision balance, foundation-feature gating, and multi-scale context configuration. Each study starts from the same completemodel configuration and changes only the parameter named in that block while keeping all other settings fixed. Consequently, the selected row in every block represents the same complete-model run and inten-

Table 2: Full-factorial ablation of CRCA, RCDF, and HRFD on the Overall, Single-object, and Multi-object subsets. Higher values are better for $\mathrm { { \cal S } _ { m } , }$ , αE, and wF; lower values are better for MAE.
<table><tr><td rowspan="2">CRCA</td><td rowspan="2">RCDF</td><td rowspan="2">HRFD</td><td colspan="4">Overall</td><td colspan="4">Single-object</td><td colspan="4">Multi-object</td></tr><tr><td> $\mathbf { S } _ { \mathrm { m } } \uparrow$  1</td><td>αE↑</td><td>wF↑</td><td>MAE↓</td><td> $\mathrm { { S } _ { m } \uparrow }$ </td><td>αE↑</td><td>wF↑</td><td>MAE↓</td><td> $ \mathbf { S } _ { \mathrm { m } } \uparrow$ </td><td>αE↑</td><td>wF↑</td><td>MAE↓</td></tr><tr><td></td><td></td><td></td><td>0.8550</td><td>0.9120</td><td>0.7570</td><td>0.0260</td><td>0.8590</td><td>0.9130</td><td>0.7610</td><td>0.0250</td><td>0.8050</td><td>0.9000</td><td>0.7010</td><td>0.0330</td></tr><tr><td>L</td><td></td><td></td><td>0.8610</td><td>0.9160</td><td>0.7660</td><td>0.0247</td><td>0.8654</td><td>0.9172</td><td>0.7722</td><td>0.0239</td><td>0.8210</td><td>0.9050</td><td>0.7100</td><td>0.0315</td></tr><tr><td></td><td>√</td><td></td><td>0.8720</td><td>0.9230</td><td>0.7850</td><td>0.0229</td><td>0.8762</td><td>0.9242</td><td>0.7905</td><td>0.0222</td><td>0.8340</td><td>0.9120</td><td>0.7350</td><td>0.0288</td></tr><tr><td></td><td></td><td>√</td><td>0.8660</td><td>0.9190</td><td>0.7740</td><td>0.0238</td><td>0.8703</td><td>0.9202</td><td>0.7800</td><td>0.0231</td><td>0.8270</td><td>0.9080</td><td>0.7200</td><td>0.0300</td></tr><tr><td>√</td><td>√</td><td></td><td>0.8810</td><td>0.9320</td><td>0.8060</td><td>0.0208</td><td>0.8847</td><td>0.9331</td><td>0.8111</td><td>0.0202</td><td>0.8480</td><td>0.9220</td><td>0.7600</td><td>0.0258</td></tr><tr><td>√</td><td></td><td>√</td><td>0.8760</td><td>0.9270</td><td>0.7950</td><td>0.0219</td><td>0.8799</td><td>0.9281</td><td>0.8002</td><td>0.0213</td><td>0.8410</td><td>0.9170</td><td>0.7480</td><td>0.0273</td></tr><tr><td></td><td>√</td><td>√</td><td>0.8870</td><td>0.9360</td><td>0.8160</td><td>0.0189</td><td>0.8903</td><td>0.9371</td><td>0.8210</td><td>0.0184</td><td>0.8570</td><td>0.9260</td><td>0.7710</td><td>0.0233</td></tr><tr><td>V</td><td>√</td><td>√</td><td>0.8950</td><td>0.9410</td><td>0.8250</td><td>0.0170</td><td>0.8990</td><td>0.9422</td><td>0.8300</td><td>0.0166</td><td>0.8590</td><td>0.9300</td><td>0.7800</td><td>0.0208</td></tr></table>

![](images/799fba58282d80526f50c833826394e592c96bb2ae621b7369aa0046f1e848ca.jpg)  
Input

![](images/affbbd261678dc1d7b84b63d7dd4beacc43bf2d34e9182a4e54087bab8fafe7e.jpg)  
GT

![](images/c713acb4f13b0bbc343f64ae941af42aa4b5039d93b6bff32f142dc3e45d5461.jpg)  
Baseline

![](images/833dc150b32fdeb36352448ed8dac6db203bff74eca90646c1debec6100e205c.jpg)  
+ CRCA

![](images/16e89f76f8d0f8511cb4d5d57a8f48422d317a6fa0793a149a2b5be6a2f066c7.jpg)  
+ RCDF

![](images/0479e84eec3badb995375ed7b42279109a04f77884b4308a3a146e6730e14a13.jpg)  
+ HRFD（Ours）

Figure 5: Qualitative visualization of progressive component variants. The labels embedded in the figure denote the progressive variants and are distinct from the factorial combinations evaluated in Table 2.

tionally repeats the same performance values. Figures 6 and 7 provide qualitative comparisons for the correlation-scale initialization and ASPP dilation settings, respectively. These studies are separate from the factorial design in Table 2 and serve as controlled configuration-selection evidence rather than as a joint component ablation.

The parameter trends also clarify how the three design choices affect different aspects of prediction. For edge supervision, a weight of 0.20 provides the most balanced result: reducing the weight weakens boundary guidance, whereas an excessively large weight overemphasizes local contours at the expense of region-level consistency. The correlation-scale study shows that small initial scales produce less selective query–reference responses. As illustrated in Fig. 6, these settings often yield diffuse activations around thin structures or incomplete responses over the target extent, while the selected scale produces more concen trated foreground evidence and cleaner boundaries. The dilation study exhibits a complementary pattern. Closely spaced small rates offer limited context variation, whereas the selected (3,6,9) group combines local detail with a broader receptive field. Fig. 7 consequently shows fewer fragmented regions and more complete recovery of low-contrast targets. These observations support using the selected settings together: edge supervision refines contours, correlation scaling sharpens reference-conditioned selection, and the dilation group supplies the multi-scale context needed to preserve object extent.

## 4.7 Computational efficiency

We measure computational efficiency on one NVIDIA GeForce RTX 4090 D GPU with 24 GB of memory. Table 4 summarizes the resulting computational profile. With a batch size of 8, one training iteration takes 140.68 ms on average. Under FP32 inference with a 384 × 384 query and five references, the model processes one query in 24.31 ms, corresponding to 41.13 frames per second (FPS). This inference measurement uses the offline-cached ICON-R reference features described above and therefore excludes reference-feature extraction. The model requires 143.54 GFLOPs (71.77 GMACs) and contains 115.67 million parameters, of which 30.00 million are trainable and 85.67 million remain frozen throughout optimization.

Table 3: Three representative one-factor parameter studies. Each block changes only the named setting, while selected rows repeat the shared complete-model result. Higher values are better for $\mathrm { { \cal S } _ { m } , }$ αE, and wF; lower values are better for MAE.
<table><tr><td>Study</td><td>Setting</td><td> $\mathbf { S } _ { \mathrm { m } } \uparrow$ </td><td> $\alpha \mathrm { { E } \uparrow }$ </td><td> $\mathrm { w F \uparrow }$ </td><td>MAE↓</td></tr><tr><td rowspan="4">Edge-loss weight</td><td>0.05</td><td>0.8884</td><td>0.9402</td><td>0.8162</td><td>0.0174</td></tr><tr><td>0.10</td><td>0.8904</td><td>0.9334</td><td>0.8108</td><td>0.0177</td></tr><tr><td>0.50</td><td>0.8885</td><td>0.9318</td><td>0.8192</td><td>0.0170</td></tr><tr><td>0.20 (selected)</td><td>0.8946</td><td>0.9414</td><td>0.8253</td><td>0.0168</td></tr><tr><td rowspan="4">Correlation-scale initialization</td><td>1</td><td>0.8874</td><td>0.9368</td><td>0.8172</td><td>0.0172</td></tr><tr><td>3</td><td>0.8873</td><td>0.9394</td><td>0.8170</td><td>0.0172</td></tr><tr><td>5</td><td>0.8876</td><td>0.9366</td><td>0.8145</td><td>0.0176</td></tr><tr><td>20 (selected)</td><td>0.8946</td><td>0.9414</td><td>0.8253</td><td>0.0168</td></tr><tr><td rowspan="3">ASPP dilation rates</td><td>(1,3,5)</td><td>0.8875</td><td>0.9386</td><td>0.8156</td><td>0.0182</td></tr><tr><td>(2,4,6)</td><td>0.8890</td><td>0.9329</td><td>0.8112</td><td>0.0179</td></tr><tr><td>(3,6,9) (selected)</td><td>0.8946</td><td>0.9414</td><td>0.8253</td><td>0.0168</td></tr></table>

![](images/e815632fcb148219155ce782841bdc315160715d7a0119bc80bdaba1e9e1f762.jpg)  
Figure 6: Qualitative comparison of correlation-scale initialization. Columns show the input, ground truth, scales of 1, 3, and 5, and the selected scale of 20 (Ours). Red boxes enlarge challenging regions and boundaries.

This profile reflects the intended separation between reusable reference processing and query-time prediction. Reference features can be computed once and reused for multiple queries, so the online path focuses on reference aggregation, query–reference interaction, and progressive decoding. The resulting throughput indicates that the additional reference-guided reasoning remains practical for batch evaluation and interac-

![](images/aace800818a9b3174027b1887e469904a37ceb4f77ade1f76cfad3e4c1af7efa.jpg)  
Input  
GT  
ASPPDilations=(1,3,5)  
ASPPDilations=(2,4,6) ASPPDilations=(3,6,9) (Ours)

Figure 7: Qualitative comparison of ASPP dilation settings. Columns show the input, ground truth, groups (1,3,5) and (2,4,6), and the selected group (3,6,9) (Ours). Red boxes enlarge challenging regions and boundaries.

tive inspection. Moreover, freezing most parameters confines optimization to the task-specific components, reducing the trainable portion of the model while retaining the representation capacity of the pretrained encoders. The reported latency should therefore be interpreted as the cost of processing a query after reference preparation; applications that replace or frequently update the reference set must additionally account for the one-time reference-feature extraction cost.

Table 4: Computational efficiency on one NVIDIA GeForce RTX 4090 D. Inference uses FP32, a 384 × 384 query, five references, and cached ICON-R features; latency excludes reference-feature extraction.
<table><tr><td>Item</td><td>Measurement</td></tr><tr><td>Training iteration time (batch size 8)</td><td>140.68 ms/step</td></tr><tr><td>Inference latency (FP32, 5-shot)</td><td>24.31 ms/image</td></tr><tr><td>Inference throughput</td><td>41.13 FPS</td></tr><tr><td>Computational cost</td><td>143.54 GFLOPs / 71.77 GMACs</td></tr><tr><td>Peak inference memory</td><td>548.6 MiB</td></tr><tr><td>Peak training memory</td><td>7008.8 MiB</td></tr></table>

## 5 Conclusion and Limitation

We presented a consensus-aware multi-source fusion framework for reference-guided camouflaged object detection. CRCA aggregates heterogeneous references into reliability-weighted spatial and vector priors, RCDF uses reference-conditioned correlation to select target-relevant DINOv3 features, and HRFD injects reference evidence at matched semantic depths before progressive decoding. The full-factorial ablation verifies that each component improves the baseline independently and that their combinations yield cumulative gains, with RCDF providing the strongest individual contribution and the complete model performing best across the overall metrics. Parameter studies further identify a consistent operating configuration, while qualitative comparisons show improved localization and boundary recovery on challenging query–reference pairs. Together, these results support reliability-aware reference aggregation, target-conditioned foundationfeature selection, and depth-matched decoding as complementary mechanisms for reference-guided camouflaged object detection. A limitation is that the evaluation uses a single dataset; future work will assess cross-dataset generalization under broader reference conditions.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No. 62402152) and the Zhejiang Province Natural Science Foundation of China (Grant No. LQN25F020017).

## References

[1] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9630–9640, 2021. doi: 10.1109/ ICCV48922.2021.00951.

[2] Geng Chen, Si-Jie Liu, Yu-Jia Sun, Ge-Peng Ji, Ya-Feng Wu, and Tao Zhou. Camouflaged object detection via context-aware cross-level fusion. IEEE Transactions on Circuits and Systems for Video Technology, 32(10):6981–6993, 2022. doi: 10.1109/TCSVT.2022.3178173.

[3] Zhennan Chen, Rongrong Gao, Tian-Zhu Xiang, and Fan Lin. Diffusion model for camouflaged object detection. In ECAI 2023. IOS Press, 2023. doi: 10.3233/FAIA230302.

[4] Shupeng Cheng, Ge-Peng Ji, Pengda Qin, Deng-Ping Fan, Bowen Zhou, and Peng Xu. Large model based referring camouflaged object detection, 2023. URL https://arxiv.org/abs/2311.17122.

[5] Xi Cheng, Youhua Zhang, Yiqiong Chen, Yunzhi Wu, and Yi Yue. Pest identification via deep residual learning in complex background. Computers and Electronics in Agriculture, 141:351–356, 2017. doi: 10.1016/j.compag.2017.08.005.

[6] Runmin Cong, Mengyao Sun, Sanyi Zhang, Xiaofei Zhou, Wei Zhang, and Yao Zhao. Frequency perception network for camouflaged object detection. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1179–1189, 2023. doi: 10.1145/3581783.3612083.

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[8] Deng-Ping Fan, Ming-Ming Cheng, Yun Liu, Tao Li, and Ali Borji. Structure-measure: A new way to evaluate foreground maps. In Proceedings of the IEEE International Conference on Computer Vision, pages 4558–4567, 2017. doi: 10.1109/ICCV.2017.487.

[9] Deng-Ping Fan, Cheng Gong, Yang Cao, Bo Ren, Ming-Ming Cheng, and Ali Borji. Enhanced alignment measure for binary foreground map evaluation. In Proceedings ofthe Twenty-Seventh Inter national Joint Conference on Artificial Intelligence, pages 698–704, 2018. doi: 10.24963/IJCAI.2018/ 97.

[10] Deng-Ping Fan, Ge-Peng Ji, Ming-Ming Cheng, and Ling Shao. Concealed object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6024–6042, 2022. doi: 10.1109/ TPAMI.2021.3085766.

[11] Deng-Ping Fan, Ge-Peng Ji, Peng Xu, Ming-Ming Cheng, Christos Sakaridis, and Luc Van Gool. Advances in deep concealed scene understanding. Visual Intelligence, 1(1), 2023. doi: 10.1007/ s44267-023-00019-6.

[12] Junwei Han, Dingwen Zhang, Gong Cheng, Nian Liu, and Dong Xu. Advanced deep-learning techniques for salient and category-specific object detection: A survey. IEEE Signal Processing Magazine, 35(1):84–100, 2018. doi: 10.1109/MSP.2017.2749125.

[13] Chunming He, Kai Li, Yachao Zhang, Longxiang Tang, Yulun Zhang, Zhenhua Guo, and Xiu Li. Camouflaged object detection with feature decomposition and edge reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22046–22055, 2023. doi: 10.1109/CVPR52729.2023.02111.

[14] Zhou Huang, Hang Dai, Tian-Zhu Xiang, Shuo Wang, Huai-Xin Chen, Jie Qin, and Huan Xiong. Feature shrinkage pyramid for camouflaged object detection with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5557–5566, 2023. doi: 10.1109/CVPR52729.2023.00538.

[15] Ge-Peng Ji, Deng-Ping Fan, Yu-Cheng Chou, Dengxin Dai, Alexander Liniger, and Luc Van Gool. Deep gradient learning for efficient camouflaged object detection. Machine Intelligence Research, 20 (1):92–108, 2023. doi: 10.1007/s11633-022-1365-9.

[16] Qi Jia, Shuilian Yao, Yu Liu, Xin Fan, Risheng Liu, and Zhongxuan Luo. Segment, magnify and reiterate: Detecting camouflaged objects the hard way. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4703–4712, 2022. doi: 10.1109/CVPR52688. 2022.00467.

[17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 3992– 4003, 2023. doi: 10.1109/ICCV51070.2023.00371.

[18] Lin Li, Jingyi Liu, Fei Yu, Xunkun Wang, and Tian-Zhu Xiang. MVDI25K: A large-scale dataset of microscopic vaginal discharge images. BenchCouncil Transactions on Benchmarks, Standards and Evaluations, 1(1):100008, 2021. doi: 10.1016/j.tbench.2021.100008.

[19] Lin Li, Jingyi Liu, Shuo Wang, Xunkun Wang, and Tian-Zhu Xiang. Trichomonas vaginalis segmen tation in microscope images. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2022, pages 68–78. Springer, 2022. doi: 10.1007/978-3-031-16440-8\_7.

[20] Qiwu Luo, Ben Li, Jiaojiao Su, Chunhua Yang, Weihua Gui, Olli Silvén, and Li Liu. CDDNet: Camouflaged defect detection network for steel surface. IEEE Transactions on Instrumentation and Measurement, 73:1–13, 2024. doi: 10.1109/TIM.2023.3336452.

[21] Yunqiu Lv, Jing Zhang, Yuchao Dai, Aixuan Li, Bowen Liu, Nick Barnes, and Deng-Ping Fan. Simultaneously localize, segment and rank the camouflaged objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11586–11596, 2021. doi: 10.1109/ CVPR46437.2021.01142.

[22] Yunqiu Lv, Jing Zhang, Yuchao Dai, Aixuan Li, Nick Barnes, and Deng-Ping Fan. Toward deeper understanding of camouflaged object detection. IEEE Transactions on Circuits and Systemsfor Video Technology, 33(7):3462–3476, 2023. doi: 10.1109/TCSVT.2023.3234578.

[23] Ran Margolin, Lihi Zelnik-Manor, and Ayellet Tal. How to evaluate foreground maps. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2014. doi: 10.1109/CVPR.2014.39.

[24] Haiyang Mei, Ge-Peng Ji, Ziqi Wei, Xin Yang, Xiaopeng Wei, and Deng-Ping Fan. Camouflaged object segmentation with distraction mining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8768–8777, 2021. doi: 10.1109/CVPR46437.2021.00866.

[25] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision, 2024. URL https: //arxiv.org/abs/2304.07193.

[26] Youwei Pang, Xiaoqi Zhao, Tian-Zhu Xiang, Lihe Zhang, and Huchuan Lu. Zoom in and out: A mixedscale triplet network for camouflaged object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2150–2160, 2022. doi: 10.1109/CVPR52688. 2022.00220.

[27] Youwei Pang, Xiaoqi Zhao, Tian-Zhu Xiang, Lihe Zhang, and Huchuan Lu. ZoomNeXt: A unified collaborative pyramid network for camouflaged object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9205–9220, 2024. doi: 10.1109/TPAMI.2024.3417329.

[28] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025. URL https://arxiv.org/abs/2508. 10104.

[29] Yanguang Sun, Chunyan Xu, Jian Yang, Hanyu Xuan, and Lei Luo. Frequency-spatial entanglement learning for camouflaged object detection. In Computer Vision – ECCV 2024, pages 343–360. Springer, 2024. doi: 10.1007/978-3-031-72658-3\_20.

[30] Yujia Sun, Shuo Wang, Chenglizhao Chen, and Tian-Zhu Xiang. Boundary-guided camouflaged object detection. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, pages 1335–1341, 2022. doi: 10.24963/IJCAI.2022/186.

[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

[32] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 548–558, 2021. doi: 10.1109/ICCV48922.2021.00061.

[33] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. PVT v2: Improved baselines with pyramid vision transformer. Computational Visual Media, 8(3):415–424, 2022. doi: 10.1007/s41095-022-0274-8.

[34] Ranwan Wu, Tian-Zhu Xiang, Guo-Sen Xie, Rongrong Gao, Xiangbo Shu, Fang Zhao, and Ling Shao. Uncertainty-aware transformer for referring camouflaged object detection. IEEE Transactions on Image Processing, 34:5341–5354, 2025. doi: 10.1109/TIP.2025.3587579.

[35] Yu-Huan Wu, Zi-Xuan Zhu, Yan Wang, Liangli Zhen, and Deng-Ping Fan. RefOnce: Distilling references into a prototype memory for referring camouflaged object detection, 2025. URL https://arxiv.org/abs/2511.20989.

[36] Zongwei Wu, Danda Pani Paudel, Deng-Ping Fan, Jingjing Wang, Shuo Wang, Cédric Demonceaux, Radu Timofte, and Luc Van Gool. Source-free depth for object pop-out. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1032–1042, 2023. doi: 10.1109/ICCV51070.2023.00101.

[37] Fan Yang, Qiang Zhai, Xin Li, Rui Huang, Ao Luo, Hong Cheng, and Deng-Ping Fan. Uncertaintyguided transformer reasoning for camouflaged object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4126–4135, 2021. doi: 10.1109/ICCV48922. 2021.00411.

[38] Siyuan Yao, Hao Sun, Tian-Zhu Xiang, Xiao Wang, and Xiaochun Cao. Hierarchical graph interaction transformer with dynamic token clustering for camouflaged object detection. IEEE Transactions on Image Processing, 33:5936–5948, 2024. doi: 10.1109/TIP.2024.3475219.

[39] Bowen Yin, Xuying Zhang, Qibin Hou, Bo-Yuan Sun, Deng-Ping Fan, and Luc Van Gool. Camo-Former: Masked separable attention for camouflaged object detection, 2022. URL https://arxiv. org/abs/2212.06570.

[40] Zhenni Yu, Xiaoqin Zhang, Li Zhao, Yi Bin, and Guobao Xiao. Exploring deeper! segment anything model with depth perception for camouflaged object detection. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 4322–4330, 2024. doi: 10.1145/3664647.3681119.

[41] Qiang Zhai, Xin Li, Fan Yang, Chenglizhao Chen, Hong Cheng, and Deng-Ping Fan. Mutual graph learning for camouflaged object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12992–13002, 2021. doi: 10.1109/CVPR46437.2021.01280.

[42] Chenxi Zhang, Qing Zhang, Jiayun Wu, and Youwei Pang. CGCOD: Class-guided camouflaged object detection. In Proceedings ofthe 33rdACM International Conference on Multimedia, pages 4369–4377, 2025. doi: 10.1145/3746027.3755408.

[43] Cong Zhang, Hongbo Bi, Tian-Zhu Xiang, Ranwan Wu, Jinghui Tong, and Xiufang Wang. Collaborative camouflaged object detection: A large-scale dataset and benchmark. IEEE Transactions on Neural Networks and Learning Systems, 35(12):18470–18484, 2024. doi: 10.1109/TNNLS.2023.3317091.

[44] Miao Zhang, Shuang Xu, Yongri Piao, Dongxiang Shi, Shusen Lin, and Huchuan Lu. PreyNet: Preying on camouflaged objects. In Proceedings of the 30th ACM International Conference on Multimedia, pages 5323–5332, 2022. doi: 10.1145/3503161.3548178.

[45] Xuying Zhang, Bowen Yin, Zheng Lin, Qibin Hou, Deng-Ping Fan, and Ming-Ming Cheng. Referring camouflaged object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47 (5):3597–3610, 2025. doi: 10.1109/TPAMI.2025.3532440.

[46] Hongwei Zhu, Peng Li, Haoran Xie, Xuefeng Yan, Dong Liang, Dapeng Chen, Mingqiang Wei, and Jing Qin. I can find you! boundary-guided separated attention network for camouflaged object detection. Proceedings of the AAAI Conference on Artificial Intelligence, 36(3):3608–3616, 2022. doi: 10.1609/aaai.v36i3.20273.

[47] Mingchen Zhuge, Deng-Ping Fan, Nian Liu, Dingwen Zhang, Dong Xu, and Ling Shao. Salient object detection via integrity learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45 (3):3738–3752, 2023. doi: 10.1109/TPAMI.2022.3179526.

## A Additional qualitative analyses

The following visualizations provide mechanism-oriented qualitative evidence for the reference-conditioned feature selection and reference-guided refinement stages. Each panel uses representative examples to illustrate intermediate responses. They complement, rather than replace, the controlled factorial ablation reported in the main manuscript.

## A.1 Reference-conditioned DINOv3 feature selection

Figure A1 illustrates how Reference-Conditioned Dual-Backbone Fusion (RCDF) transforms generic DI NOv3 responses into target-conditioned evidence. The ungated DINOv3 prior can contain broad activations around semantically salient but target-ambiguous regions. The Semantic Gate, which corresponds to RCDF’s reference-conditioned correlation map, attenuates responses that are inconsistent with the reference target while retaining target-aligned structures. The gate maps make this spatial selectivity explicit, with high responses concentrated around regions that agree with the reference-guided target. These examples are qualitative diagnostics and should be interpreted together with the controlled component ablation in the main manuscript.

![](images/fa8d86a840719289fe8f32bf18c49aff7d43f970327377627ca90c2b751c1189.jpg)  
Input

![](images/7207f4e8c11d7d1468e51df3d905ce557204ff2d3c1ab6cd0c0ba8f2edab30ae.jpg)  
GT

![](images/051ff4965f6663d34246c42d09b418d23dfa08f6e9e7405393c5473e43ee0ffd.jpg)  
Baseline

![](images/f4be3eaf8062fbb39af7cefc244c77b44832bc62b16c9319c1bcd730654e6136.jpg)  
+ DINOv3 Prior

![](images/0eba0c24e1aae85dfed8054435aa9e4d5df04649b6542be9c29fe23b27bb568c.jpg)  
+ Semantic Gate

![](images/3085874e15f7eb665566eb0143c91506caf20ab09ef09fa076eb20681c424baf.jpg)  
Gate Map  
Figure A1: Qualitative visualization of reference-conditioned DINOv3 feature selection. From left to right, the columns show the input, ground truth (GT), baseline prediction, output with the DINOv3 prior, output after the Semantic Gate, and gate map. In the present formulation, the Semantic Gate denotes RCDF’s reference-conditioned correlation map.

## A.2 Progressive reference-guided refinement

Figure A2 visualizes progressive refinement from the semantic-gated response to the final prediction. The Semantic Gate implements RCDF’s correlation-based selection, whereas Cross-Attention and Ref. Interac tion denote the reference-guided interactions in hierarchical reference fusion and decoding (HRFD). Later stages suppress extraneous activations, and the rightmost column shows residual errors relative to GT. This visualization qualitatively supports the complementary roles of target-conditioned selection and referenceguided refinement without introducing an independent ablation factor.

![](images/f0481b55aa8c3a1d7601e220bb104de7e1587d56591c736e022e3cfc7eb03af3.jpg)  
GT  
+ Ref. Interaction

Figure A2: Qualitative visualization of progressive reference-guided refinement. From left to right, the columns show the input, GT, outputs after the Semantic Gate, Cross-Attention, and Ref. Interaction, followed by residual error. The latter two stages are the reference-guided interactions in HRFD.