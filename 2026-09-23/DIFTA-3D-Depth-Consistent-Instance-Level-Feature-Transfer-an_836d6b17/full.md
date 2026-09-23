# DIFTA-3D: Depth-Consistent Instance-Level Feature Transfer and Adaptation of DINOv3 for 3D Detection

Linman Wang<sup>1</sup>, ZiFei Zhang<sup>1</sup>, Chunran Zheng<sup>2</sup>, Xiwang Dong<sup>1</sup>, Jiarong Lin<sup>1,∗</sup>

Abstract— RGB-D 3D instance detectors benefit from visual semantics, but the task-specific Faster R-CNN/ResNet branch used by IIFNet3D couples feature extraction to a separately trained 2D detector and its image-domain labels. Replacing that branch with a frozen vision foundation model removes this taskspecific dependency, but may introduce occlusion noise and a mismatch between patch features and geometry-aware detection features. In this work, we investigate this replacement through an adaptation of DINOv3 to the instance-level fusion pipeline of IIFNet3D. At the core of our approach is a depth-consistent feature pipeline that projects scene points into calibrated RGB-D frames, applies a metric depth-residual check, averages the accepted DINOv3 features into an offline point cache, and aggregates the cached features inside proposal-aligned RoI grids. The geometric and bidirectional instance-fusion paths are preserved, while Conservative VAID is evaluated as a low-strength, support-weighted semantic distillation recipe applied only to positive RoIs. We conduct extensive evaluations on ScanNetV2 to assess the proposed transfer recipes. On ScanNetV2, our DINOv3 control achieves mAP scores of 76.15 and 60.93 at IoU thresholds of 0.25 and 0.50, respectively. The Conservative VAID setting achieves mAP scores of 76.59 and 62.16, corresponding to numerical gains of 0.44 and 1.23 points over the control, respectively, in this checkpoint-level recipe comparison. The reported IIFNet3D result of 75.7/63.8 is used only as an external reference because the visual branch and processing protocol differ. Accordingly, we interpret these results as evidence for a controlled transfer recipe rather than as a causal estimate of the individual contributions of VAID or depth filtering.

## I. INTRODUCTION

Three-dimensional instance detection from RGB-D observations is central to indoor scene understanding and embodied robotics [1], [2]. A reliable detector must combine geometric evidence from point clouds with visual semantics that distinguish objects with similar shapes or limited geometric support. Vision foundation models offer a promising source of transferable semantics because their frozen representations can be reused across image domains without training a taskspecific image detector [3]. Yet a general-purpose patch representation is not inherently aligned with a 3D proposal, an occluded image measurement, or a geometry-sensitive detection head. The key challenge is therefore to adapt foundation-model features for instance-level 3D reasoning while preserving the geometric representation that supports detection.

Existing RGB-D 3D detectors exploit point-cloud geometry through voting, grouping, or point-to-image fusion; some also use image features from a task-specific 2D detector [4], [5], [6], [7]. Instance-level fusion is attractive because each proposal can aggregate visual evidence before the 3D detection head. The original IIFNet3D branch has a useful detection prior, but it requires a dedicated 2D detector and does not provide the broad, frozen patch representation provided by DINOv3. Introducing DINOv3 raises three unresolved issues. First, projecting a 3D sample into an RGB frame may retrieve a background patch when the sample is occluded. Second, a fixed ROI grid can provide unstable coverage for thin, boundary, or sparsely observed objects. Third, the semantic distribution of a frozen foundation model differs from that of the detection-specific visual features used to design the fusion module. Our controlled experiments show that direct cross-attention, proposal/ROI gating, residual fusion, and strong visual distillation can all reduce performance. These observations motivate explicit visibility control and task-constrained transfer rather than direct feature substitution.

We address these issues with DIFTA-3D, a depthconsistent instance-level transfer of DINOv3 within IIFNet3D. The method replaces the task-specific visual encoder while retaining the geometric branch, the proposal ROI grid with RoI-Conv pooling, and the bidirectional GGF/SGF fusion path. For each scene point projected into a calibrated frame, we compare the camera-space depth with the depth image and retain the visual sample only when the discrepancy between them falls below a predefined tolerance. Accepted samples are averaged into a 384-dimensional point cache, and unobserved points are set to zero before proposal pooling. On top of this construction, Conservative VAID uses detached DINOv3 features as teachers and applies low-strength cosine distillation only to positive RoIs, with a feature-energy support weight. The central insight is to expose DINOv3’s broad semantics through depth-consistent instance evidence while constraining the transfer to protect the detector’s geometry, rather than treating a foundation-model feature map as a drop-in replacement.

Our ScanNetV2 checkpoint comparison illustrates this positioning. The DINOv3 control obtains 76.15 mAP@0.25 and 60.93 mAP@0.50, whereas Conservative VAID obtains 76.59 and 62.16, respectively. Thus, the observed recipelevel difference is 0.44 points at mAP@0.25 and 1.23 points at mAP@0.50. These differences combine continuation optimization with the auxiliary recipe and are not an isolated estimate of VAID. The reported IIFNet3D result of 75.7/63.8 is included only as an external reference because its visual branch and processing protocol differ from those used in our

experiments.

Our contributions are summarized as follows:

• We identify the feature-distribution and visibility mismatches that arise when a task-specific 2D visual branch is replaced by a frozen DINOv3 encoder in an instance-level RGB-D detector.

• We construct depth-consistent DINOv3 instance features using calibrated depth-residual filtering and proposal-aligned RoI aggregation, and evaluate Conservative VAID as a low-strength, support-weighted semantic regularizer for positive RoIs.

• We provide a controlled ScanNetV2 evaluation and a cross-dataset protocol analysis, reporting categorylevel behavior, high-IoU performance, and negative results for simpler transfer alternatives while separating observed recipe effects from unverified component-level causal effects.

## II. RELATED WORK

Point-based detectors extract local geometric evidence directly from irregular point sets, whereas voting-based methods generate object-center hypotheses from surface points [4], [8]. Transformer-based and sparse-voxel detectors provide complementary mechanisms for modeling long-range context and operating on sparse scenes [9], [10], [11]. These methods establish strong geometric foundations for indoor detection, but thin or weakly sampled surfaces can remain ambiguous when appearance is needed to distinguish instances.

Multimodal detectors combine point-cloud geometry with RGB semantics through point-to-pixel projection, voxel lifting, or cross-modal attention [5], [6], [12]. IIFNet3D performs proposal-level instance-to-instance fusion with geometry-guided and semantics-guided attention [7]. We follow this instance-level design while studying a distinct question: how to replace its task-specific visual branch with DINOv3. This replacement changes both the feature distribution and the reliability of projected evidence. We therefore retain the original fusion path and redesign the visual feature construction and adaptation protocol around depth consistency.

Self-supervised vision encoders such as DINO[13], DINOv2[14], and DINOv3[3] provide patch-level representations that transfer across image domains. Their semantic breadth is useful when a task-specific 2D detector is unavailable. However, patch features are not inherently aware of 3D visibility, proposal boundaries, or the optimization behavior of a geometry-trained detector. In our setting, DINOv3 is consequently used as a frozen teacher, and its evidence is adapted at the instance level rather than used to replace the geometric representation.

Depth consistency and occlusion reasoning have been used to constrain image-to-3D feature lifting [12], [16]. Our setting differs in that visibility is evaluated for proposal ROI samples across calibrated views and then used to construct an instance feature. This formulation supports explicit invalidview handling and provides a basis for analyzing hard rejection and effective coverage, including failure modes involving thin and boundary objects. Together, these distinctions motivate our depth-consistent adaptation of DINOv3 within an instance-level 3D detector.

## III. METHODOLOGY

## A. Overview

DIFTA-3D replaces the task-specific visual branch of IIFNet3D with depth-consistent DINOv3 evidence while preserving the geometric proposal path and the instancefusion blocks. The framework comprises three stages: depthconsistent feature construction, proposal-aligned visual aggregation, and conservative semantic adaptation, as illustrated in Fig. 1.

Let P denote the input point cloud and let $\{ ( I _ { v } , D _ { v } , \Pi _ { v } ) \} _ { v = 1 } ^ { V }$ denote calibrated RGB images, depth images, and camera projection matrices. A geometric encoder extracts point features, and the coarse proposal generator (CPG) produces a set of 3D proposals $\{ B _ { i } \} _ { i = 1 } ^ { N }$ For each proposal, the visual branch constructs an instance descriptor from DINOv3, the bidirectional instance modules fuse it with proposal geometry, and the detection head performs classification and box regression. The following subsections define these stages in detail.

## B. Depth-Consistent DINOv3 Features

The principal ScanNet experiments use an offline pointfeature cache. The frozen DINOv3 ViT-S/16 model produces a 384-channel patch map $F _ { v } = \Phi ( I _ { v } )$ . For an aligned scene point $p _ { j }$ , let $\hat { p } _ { j } ~ = ~ [ p _ { j } ^ { \top } , 1 ] ^ { \top }$ , and let $M _ { v } ^ { c } , M _ { v } ^ { d }$ denote the calibrated color and depth projection matrices, including the inverse scene alignment. For sensor $s \in \{ c , d \}$ , the projection and feature sampling operations are defined as follows.

$$
\begin{array} { r l } & { h _ { j , v } ^ { s } = M _ { v } ^ { s } \hat { p } _ { j } , \quad z _ { j , v } ^ { s } = h _ { j , v , 3 } ^ { s } , } \\ & { u _ { j , v } ^ { s } = h _ { j , v , 1 : 2 } ^ { s } / z _ { j , v } ^ { s } , } \\ & { f _ { j , v } = \mathrm { B i l i n e a r } \big ( F _ { v } , \mathrm { r o u n d } ( u _ { j , v } ^ { c } ) / 1 6 \big ) , } \end{array}\tag{1}
$$

where $\boldsymbol { u } _ { j , v } ^ { s }$ is a two-dimensional pixel coordinate, and $z _ { j , v } ^ { s }$ is camera-space depth. Color coordinates are rounded before sampling, as in the cache builder. Let $a _ { j , v }$ require a positive color depth and an in-bounds rounded color coordinate. With $z _ { j , v } = z _ { j , v } ^ { d }$ and depth observation $d _ { j , v } \mathrm { , }$ the mask is

$$
m _ { j , v } = a _ { j , v } \mathbf { 1 } [ d _ { j , v } > 0 . 0 5 \mathrm { m } ] \mathbf { 1 } [ | z _ { j , v } - d _ { j , v } | \leq \tau _ { d } ] ,\tag{2}
$$

where $\tau _ { d }$ denotes the depth-residual threshold, 1[·] denotes the indicator function, which equals 1 when its condition is satisfied and 0 otherwise. Raw depth images are converted to metric units, and projected depth coordinates are rounded and clamped to the valid image range. If a depth image is unavailable, the cache builder falls back to the front-facing and color-in-bounds test; strict occlusion filtering therefore requires a valid depth input. With $\begin{array} { r } { n _ { j } = \sum _ { v } m _ { j , v } } \end{array}$ , the cached feature is:

$$
\bar { f } _ { j } = \frac { \sum _ { v } m _ { j , v } f _ { j , v } } { \operatorname* { m a x } ( n _ { j } , 1 ) } .\tag{3}
$$

![](images/a8c4d275becc3acdab103e64193fe768f25482f2dbe60d43c44f87863cd5622b.jpg)  
Fig. 1. Overview of DIFTA-3D. The geometric path generates 3D proposals, while projected DINOv3 features are filtered by a depth-residual check before RoI aggregation and bidirectional instance fusion. Conservative VAID uses detached teacher features, visibility weighting, and positive-RoI supervision.

Points with no accepted observation retain a zero feature. This is a point-level cache rule; subsequent learned pooling can still transform the resulting descriptor. During training and evaluation, the cache is loaded together with the point cloud and queried within each proposal. The depth projection and invalid-observation logic are summarized in Fig. 2.

## C. Instance-Level Fusion and Conservative VAID

1) Proposal Generation and RoI-Conv Aggregation.: The inherited geometric stream voxelizes XYZRGB observations and extracts 64-channel BiResNet features. The CPG encoder combines local and superpoint context, and its prediction head produces class scores, centerness, and box parameters for axis-aligned ScanNet bounding boxes. The refinement stage samples a fixed set of proposals per scene. These components are kept unchanged in the DINOv3 experiments.

For an axis-aligned ScanNet proposal with center $b _ { i }$ and dimensions $\ell _ { i } ,$ we construct a proposal-aligned grid of cellcenter queries. With $k \in \{ 0 , \ldots , 6 \} ^ { 3 }$ , the grid and visual pooling operations are defined as follows:

$$
\begin{array} { r l } & { x _ { i , k } = b _ { i } + \ell _ { i } \odot \big ( ( k + 0 . 5 1 ) / 7 - 0 . 5 1 \big ) , } \\ & { ~ q _ { i } = \mathcal { R } _ { 7 } \big ( \mathcal { R } _ { 5 } ( S , \mathcal { G } _ { i } ) \big ) , } \end{array}\tag{4}
$$

where $\mathcal { G } _ { i } = \{ x _ { i , k } \}$ and $s$ is the sparse tensor obtained by voxelizing $\{ ( p _ { j } , \bar { f } _ { j } ) \}$ . The operator $\mathcal { R } _ { 5 }$ queries voxelized grid locations with a kernel-size-five sparse convolution, batch normalization, and ELU. The operator $\mathcal { R } _ { 7 }$ reorganizes these features on the local $7 ^ { 3 }$ grid and applies a kernel-sizeseven convolution and batch normalization at its center. This produces a 128-dimensional descriptor. The same pooling operation is applied to the 64-channel geometric decoder and 390-channel CPG streams; their descriptors are combined using a learned scalar sigmoid weight to obtain $g _ { i }$

The cache construction and proposal pooling follow the sequence specified by Eqs. (1)–(4); the resulting descriptors are then passed to instance fusion.

2) Bidirectional instance fusion.: The pooled descriptors interact through the geometry-guided fusion (GGF) and semantics-guided fusion (SGF) blocks. The retained channelwise gate is applied before cross-attention:

$$
\tilde { g } _ { i } = \gamma _ { i } \odot g _ { i } + ( 1 - \gamma _ { i } ) \odot q _ { i } , \gamma _ { i } = \sigma ( \mathrm { M L P } ( [ g _ { i } ; q _ { i } ] ) )\tag{5}
$$

This is the pre-attention gate in the current DINOv3 configuration; the proposal/ROI gate evaluated as P0 is a separate ablation. Let $E _ { q }$ and $E _ { k }$ be learned MLPs that encode the sine-transformed box center, dimensions, and volume, and let $E _ { c }$ encode the proposal center. Here, ${ \mathcal { A } } ( X , Y ) \equiv$ $\mathrm { M H A } ( X , Y , Y )$ denotes single-head scaled dot-product attention, LN(·) denotes Layer Normalization, and the second argument supplies both the key and value; the implemented two directions can be written as

$$
\begin{array} { r l } & { H _ { g } = \mathrm { L N } \big ( A ( \tilde { G } + E _ { q } , Q + E _ { k } ) + \tilde { G } + E _ { q } \big ) , } \\ & { } \\ & { H _ { q } = \mathrm { L N } \big ( A ( Q + E _ { c } , \tilde { G } + E _ { c } ) + Q + E _ { c } \big ) , } \\ & { \qquad H = H _ { g } + H _ { q } . } \end{array}\tag{6}
$$

where $\tilde { G }$ and $Q$ stack the proposal descriptors within each scene. Each attention block uses 128 channels, one head, a dropout rate of 0.1, and a residual LayerNorm. The output is additive, consistent with the reference implementation; it is not formed by concatenation.

3) Conservative VAID.: Conservative VAID treats the pooled geometric descriptor $s _ { i } = g _ { i }$ as the student and the detached visual descriptor $t _ { i } = \mathrm { s g } ( q _ { i } )$ as the teacher. The offline cache does not store a calibrated proposal visibility count. We therefore use a detached feature-energy proxy. For $e _ { i } = \| t _ { i } \| _ { 2 }$ and local mean e¯, the implementation computes

![](images/42dd7da482f293220e0e3182df115d78eda38a0b24cf1e1c032c8d6ef94bc7c6.jpg)  
Fig. 2. Depth-consistent point-feature construction. Each 3D point is projected independently into calibrated RGB and depth cameras. Frozen DINOv3 patches provide the color feature, while the metric depth sample checks $d > 0 . 0 5$ m and the residual |z − d| ≤ 0.25 m; foreground observations are retained, and occluded background observations are rejected. The comparison shows how depth filtering prevents contaminated features from being aggregated. Accepted observations are averaged into a 384-dimensional offline point cache, with an explicit zero feature assigned when no view is valid. A proposal-aligned $7 \times 7 \times 7$ grid queries this cache, and RoI-Conv pooling produces the instance descriptor q<sub>i</sub>.

$$
r _ { i } = \mathrm { c l i p } \left( \frac { e _ { i } } { 2 \operatorname* { m a x } ( \bar { e } , 1 0 ^ { - 6 } ) } , 0 , 1 \right) , \ c _ { i } = 0 . 2 5 + 0 . 7 5 r _ { i } .\tag{7}
$$

Thus, $c _ { i }$ is a support proxy rather than a measured visibility fraction. Here, e¯ is the mean of $e _ { i }$ over all RoIs in the current batch. For the positive RoIs P selected by the detector’s regression-valid mask, we define the auxiliary objective as

$$
\mathcal { L } _ { \mathrm { v a i d } } = \frac { \sum _ { i \in \mathcal { P } } c _ { i } \left( 1 - \cos ( s _ { i } , \mathrm { s g } ( t _ { i } ) \right) ) } { \operatorname* { m a x } \left( \sum _ { i \in \mathcal { P } } c _ { i } , 1 0 ^ { - 6 } \right) } ,\tag{8}
$$

where cosine similarity is computed between normalized features, and sg denotes stop-gradient. If no positive RoI is present, the auxiliary loss is set to zero. DINOv3 remains frozen, and the auxiliary term introduces neither a second encoder pass nor additional adapter parameters. The total objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { d e t } } + \lambda _ { \mathrm { v a i d } } \mathcal { L } _ { \mathrm { v a i d } } . } \end{array}\tag{9}
$$

The reported conservative setting uses an adaptation learning rate of $1 \times 1 0 ^ { - 5 }$ and $\lambda _ { \mathrm { v a i d } } = 0 . 0 1$ . The low loss weight and positive-RoI restriction make the auxiliary term serve as a semantic-alignment regularizer rather than a replacement for the detection objective.

4) Detection Supervision and Inference: The detector retains supervision at both stages. The CPG objective combines focal classification, centerness, distance-IoU box regression, and Smooth-L1 voting losses. The refinement objective combines classification, Smooth-L1 residual regression, and distance-IoU losses with weights 1, 0.5, and 1, respectively; the refinement and CPG losses are summed to define $\mathcal { L } _ { \mathrm { d e t } }$ The refinement target uses the inherited regression-valid mask, and a previously negative proposal becomes positive when its maximum ground-truth IoU is at least 0.3. At inference, VAID supervision is omitted; the detector uses standard refinement decoding and NMS, with a pre-NMS limit of 1, 000 candidates and an IoU threshold of 0.5.

## IV. EXPERIMENTS

## A. Experimental Setup

We evaluate ScanNetV2 across its 18 classes at IoU thresholds of 0.25 and 0.50. The principal control uses the depth-consistent offline DINOv3 cache and the same detector configuration as Conservative VAID, but without the auxiliary loss. The name “times=8” refers to the RepeatDataset training multiplier, not to eight RGB frames; each cached point feature may aggregate features from all frames available for its scene. Training uses four-GPU distributed data parallelism without validation during training, and evaluation follows the corrected four-GPU protocol. The reported IIFNet3D values serve as external references rather than same-protocol baselines because the visual branch and processing details differ.

All comparisons in this paper use the protocols and evaluators described above. Quantities that are unavailable in an external source are omitted rather than used to support a numerical claim. For provenance, the in-house runs use seed 0, deterministic data-loader settings, and a fixed checkpoint rule with no best-checkpoint selection: each VAID recipe is evaluated after a one-epoch continuation from the archived control checkpoint. The evaluator records the ordered AP vector, IoU threshold, checkpoint, and seed before computing mAP from unrounded values; the reproducibility artifact should expose these records, the evaluator version, configuration, and exact commands.

## B. Datasets and Metrics

1) ScanNetV2: ScanNetV2 contains 1,513 reconstructed indoor scenes, with 1,201 scenes for training and 312 scenes for validation. We follow the 18-category protocol used by the IIFNet3D implementation and report mean average precision at 3D IoU thresholds of 0.25 and 0.50. The corrected inhouse evaluator uses axis-aligned corner IoU for the ScanNet boxes and the 11-point AP calculation implemented in the archived evaluation script. The evaluator computes each mAP from the unrounded 18-class AP vector in the same log and rounds the final scalar to two decimals; the class entries printed in Table I are independently rounded. The Table I rows were regenerated from the archived epoch-17 control and epoch-1 Conservative VAID logs.

TABLE I  
SCANNETV2 CATEGORY-WISE DETECTION RESULTS (MAP@0.25). ORIGINAL-PAPER ROWS ARE EXTERNAL REFERENCES; DINOV3 ROWS USE THE CORRECTED IN-HOUSE EVALUATOR. BOLD ENTRIES MARK THE LARGER OF THE TWO REPORTED DINOV3 VALUES FOR EACH CATEGORY.
<table><tr><td>Method</td><td>cab</td><td>bed</td><td>chr</td><td>sofa</td><td>tbl</td><td>door</td><td>wnd</td><td>bks</td><td>pic</td><td>ctr</td><td>desk</td><td>crt</td><td>frg</td><td>shw</td><td>tol</td><td>snk</td><td>bth</td><td>ofn mAP</td><td></td></tr><tr><td colspan="10">Point Cloud-Driven</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GSDN [11]</td><td>41.6</td><td>82.5</td><td>92.1</td><td>87.0</td><td>61.1</td><td>42.4</td><td>40.7</td><td>51.1</td><td>10.2</td><td>64.2</td><td>71.1</td><td>54.9</td><td>40.0</td><td>70.5</td><td>99.9</td><td>75.5</td><td>93.2</td><td>53.1</td><td>62.8</td></tr><tr><td>VoteNet [4]</td><td>47.7</td><td>88.7</td><td>89.5</td><td>89.3</td><td>62.1</td><td>54.1</td><td>40.8</td><td>54.3</td><td>12.0</td><td>63.9</td><td>69.4</td><td>52.0</td><td>52.5</td><td>73.3</td><td>95.9</td><td>52.0</td><td>92.5</td><td>41.4</td><td>62.9</td></tr><tr><td>Pointformer [17]</td><td>46.7</td><td>88.4</td><td>90.5</td><td>88.7</td><td>65.7</td><td>55.0</td><td>47.7</td><td>55.8</td><td>18.0</td><td>63.8</td><td>69.1</td><td>55.4</td><td>48.5</td><td>66.2</td><td>98.9</td><td>61.5</td><td>86.7</td><td>47.4</td><td>64.1</td></tr><tr><td>MLCNet [18]</td><td>42.5</td><td>88.5</td><td>90.0</td><td>87.4</td><td>63.5</td><td>56.9</td><td>47.0</td><td>56.9</td><td>11.9</td><td>63.9</td><td>76.1</td><td>56.7</td><td>60.9</td><td>65.9</td><td>98.3</td><td>59.2</td><td>87.2</td><td>47.9</td><td>64.5</td></tr><tr><td>BRNet [19]</td><td>49.9</td><td>88.3</td><td>91.9</td><td>86.9</td><td>69.3</td><td>59.2</td><td>45.9</td><td>52.1</td><td>15.3</td><td>72.0</td><td>76.8</td><td>57.1</td><td>60.4</td><td>73.6</td><td>93.8</td><td>58.8</td><td>92.2</td><td>47.1</td><td>66.1</td></tr><tr><td>H3DNet [8]</td><td>49.4</td><td>88.6</td><td>91.8</td><td>90.2</td><td>64.9</td><td>61.0</td><td>51.9</td><td>54.9</td><td>18.6</td><td>62.0</td><td>75.9</td><td>57.3</td><td>57.2</td><td>75.3</td><td>97.9</td><td>67.4</td><td>92.5</td><td>53.6</td><td>67.2</td></tr><tr><td>GroupFree3D [9]</td><td>52.1</td><td>91.9</td><td>93.6</td><td>88.0</td><td>70.7</td><td>60.7</td><td>53.7</td><td>62.4</td><td>16.1</td><td>58.5</td><td>80.9</td><td>67.9</td><td>47.0</td><td>76.3</td><td>99.6</td><td>72.0</td><td>95.3</td><td>56.4</td><td>69.1</td></tr><tr><td>SCGÑet [20]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>69.1</td></tr><tr><td>Objformer [21]</td><td>55.4</td><td>88.7</td><td>93.4</td><td>87.2</td><td>74.1</td><td>61.3</td><td>57.3</td><td>55.5</td><td>17.9</td><td>67.4</td><td>85.1</td><td>74.4</td><td>52.0</td><td>79.8</td><td>97.5</td><td>71.8</td><td>88.2</td><td>57.5</td><td>70.3</td></tr><tr><td>FCAF3D [22]</td><td>57.2</td><td>87.0</td><td>95.0</td><td>92.3</td><td>70.3</td><td>61.1</td><td>60.2</td><td>64.5</td><td>29.9</td><td>64.3</td><td>71.5</td><td>60.1</td><td>52.4</td><td>83.9</td><td>99.9</td><td>84.7</td><td>86.6</td><td>65.4</td><td>71.5</td></tr><tr><td>TR3D [12]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>72.9</td></tr><tr><td>DLLA [23]</td><td>56.0</td><td>86.8</td><td>96.3</td><td>91.5</td><td>74.8</td><td>63.2</td><td>57.2</td><td>65.0</td><td>32.7</td><td>75.8</td><td>82.5</td><td>57.9</td><td>60.7</td><td>83.7</td><td>99.8</td><td>80.2</td><td>90.2</td><td>64.8</td><td>73.8</td></tr><tr><td>SPGroup3D [24]</td><td>58.0</td><td>88.2</td><td>94.2</td><td>93.0</td><td>73.4</td><td>68.4</td><td>65.9</td><td>66.9</td><td>39.3</td><td>72.5</td><td>79.6</td><td>64.2</td><td>64.0</td><td>79.6</td><td>99.8</td><td>77.3</td><td>90.2</td><td>62.2</td><td>74.3</td></tr><tr><td>CAGroup3D [25]</td><td>60.4</td><td>93.0</td><td>95.3</td><td>92.3</td><td>70.0</td><td>68.0</td><td>63.6</td><td>67.3</td><td>40.7</td><td>77.0</td><td>83.9</td><td>69.4</td><td>65.7</td><td>73.0</td><td>100.0</td><td>79.7</td><td>87.0</td><td>66.1</td><td>75.1</td></tr><tr><td>Multi-modal</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MFFVoteNet [26]</td><td>40.5</td><td>89.0</td><td>89.1</td><td>85.5</td><td>64.4</td><td>57.6</td><td>49.8</td><td>58.9</td><td>14.4</td><td>63.4</td><td>69.8</td><td>51.6</td><td>51.6</td><td>71.2</td><td>97.3</td><td>59.5</td><td>91.4</td><td>45.5</td><td>63.9</td></tr><tr><td>PiMAE [27]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>67.6</td></tr><tr><td>TokenFusion [28]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>69.8</td></tr><tr><td>SPGroup3D+FF†</td><td>62.4</td><td>89.9</td><td>94.3</td><td>92.0</td><td>73.3</td><td>69.7</td><td>68.3</td><td>73.6</td><td>44.1</td><td>65.1</td><td>80.2</td><td>62.9</td><td>64.9</td><td>70.6</td><td>100.0</td><td>78.2</td><td>91.9</td><td>63.8</td><td>74.7</td></tr><tr><td>IIFNet3D (ext.)</td><td>62.1</td><td>90.2</td><td>94.9</td><td>92.7</td><td>77.0</td><td>70.5</td><td>68.7</td><td>68.1</td><td>49.5</td><td>63.0</td><td>82.5</td><td>65.4</td><td>64.9</td><td>79.7</td><td>100.0</td><td>77.5</td><td>91.4</td><td>65.3</td><td>75.7</td></tr><tr><td colspan="14">DINOv3 adaptation (current evaluator) 62.3083.25</td></tr></table>

<sup>†</sup> Early-stage fusion in the original paper. Class abbreviations follow the ScanNetV2 labels: cab (cabinet), chr (chair), tbl (table), wnd (window), bks (bookshelf), pic (picture), ctr (counter), crt (curtain), frg (refrigerator), shw (shower), tol (toilet), snk (sink), bth (bathtub), and ofn (other furniture). External and DINOv3 rows use different visual branches and processing protocols. Bold marks the larger of the two reported DINOv3 values in each column; external rows are not used for bolding.

2) SUN-RGBD: SUN-RGBD contains 10, 335 indoor RGB-D images, with 5, 285 training images and 5, 050 validation images. Following the standard 10-category protocol, we report mAP at 3D IoU thresholds of 0.25 and 0.50. The corrected evaluator recomputes rotated 3D IoU from box corners and integrates the precision–recall curve over all recall changes. Thus, the AP implementation is documented separately for the two datasets rather than being assumed to be identical. The original IIFNet3D paper reports only mAP at a 3D IoU threshold of 0.25 for its SUN-RGBD reference row. TABLE II TABLE II

VAID TRAINING-RECIPE COMPARISON ON SCANNETV2.
<table><tr><td>Configuration LR</td><td>λ Support AP25 AP50</td><td></td></tr><tr><td>External reference</td><td></td><td></td></tr><tr><td>IIFNet3D</td><td></td><td>75.70 63.80</td></tr><tr><td>DINOv3 control and training recipes</td><td></td><td></td></tr><tr><td>DINOv3 control</td><td>0 None</td><td>76.15 60.93</td></tr><tr><td>VAID, uniform  $1 0 ^ { - 3 }$ </td><td>0.05 Uniform 70.99 57.80</td><td></td></tr><tr><td>VAID, energy-weighted 10−5 0.01 Energy</td><td></td><td>76.5962.16</td></tr></table>

LR and λ refer to VAID fine-tuning. “Uniform” and “Energy” denote uniform and detached feature-energy support weighting on positive RoIs. AP25/AP50 denote mAP at IoU thresholds of 0.25 and 0.50, respectively. The two VAID rows jointly change the continuation learning rate, loss weight, and support rule; they are not a one-factor ablation. Bold marks the best reported DINOv3 result.

## C. Implementation Details

The detector is implemented with the MMDetection3D framework. The geometric stream uses the BiResNet-based sparse 3D backbone with a voxel size of 0.02 m. The principal ScanNetV2 control uses AdamW with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ , weight decay of $1 \times 1 0 ^ { - 4 }$ , and learning-rate decays at epochs 9, 12, and 15 for 20 epochs. The distributed batch uses four samples per GPU. DINOv3 is a frozen ViT-S/16 encoder with 384-channel cached patch features; the RoI-Conv path maps each stream to 128 channels. The principal cache uses the depth-consistency threshold specified above, and the training configuration uses a RepeatDataset multiplier of eight. The RoI head samples a $7 \times 7 \times 7$ grid, retains 128 proposals per scene, and uses kernel sizes of five and seven in its two sparse pooling stages. Conservative VAID is initialized from the archived control checkpoint and fine-tuned for the reported screening run with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a loss weight of 0.01. The packed train and validation caches occupy 171.9480 decimal GB, and the checkpoint used for the reported ScanNetV2 rows contains 47.5587 million parameters. The logged peak memory is 57,570 MB for the control and 68,876 MB for Conservative VAID; an inference-throughput benchmark was not archived and is therefore not reported.

The original IIFNet3D values in Table I are external references: its task-specific Faster R-CNN branch and processing protocol were not rerun here. The DINOv3 rows are produced by the current evaluator, and unavailable external quantities are omitted.

## D. Evaluation Results

Table II compares the reference detector, the DINOv3 control, and two VAID training recipes. Both VAID variants start from the same control checkpoint and are evaluated after one additional training epoch, using the same depthchecked cache, proposal pooling, and corrected evaluator. Because the recipes jointly change the continuation learning rate, auxiliary-loss weight, and support weighting, this table is a recipe comparison rather than a causal isolation of VAID. The tables use the archived control checkpoint and the first Conservative VAID continuation checkpoint; no multiseed or best-of-epoch selection is claimed. Accordingly, the rows should be read as checkpoint-level observations under a fixed protocol, not as estimates of a population mean or a statistically stable method effect.

1) Main Comparison: The single reported Conservative VAID checkpoint changes mAP0.50 from 60.93 to 62.16 relative to the DINOv3 depth-check control, a difference of 1.23 points, while mAP0.25 changes from 76.15 to 76.59, a difference of 0.44 points. The larger numerical difference at the stricter overlap threshold is an observation about these two checkpoints only; the current measurements do not isolate localization from classification effects, establish statistical significance, or demonstrate that the difference is caused by VAID. The external IIFNet3D report of 75.7/63.8 remains higher at mAP0.50, but it does not constitute a sameprotocol comparison.

2) Training-Recipe Comparison: Standard VAID reaches 70.99 mAP0.25, whereas the conservative recipe reaches 76.59 after the same one-epoch continuation duration. The two runs do not use identical optimization settings: their learning rates, loss weights, and support rules differ. Both restrict distillation to positive RoIs. The recipes jointly change the learning rate, loss weight, and support weighting, so the 5.60-point difference measures their combined effect. The matched equal-budget comparison in Table IV separates detection-only continuation from uniform, energy-weighted, and conservative VAID under the same $1 0 ^ { - 5 }$ learning rate and 0.01 loss weight.

3) Component Isolation: Table III reports the paired cache experiment: applying $| z - d | \leq 0 . 2 5$ m retains 65.8% of observations, reduces mean support from 2.84 to 1.91 points, and improves mAP0.25/0.50 by 0.50/1.18 points over the unfiltered cache. Table IV fixes the continuation learning rate, loss weight, initialization, and one-epoch budget. Detection-only continuation reaches 76.22/61.48, while energy-weighted and conservative VAID reach 76.40/61.78 and 76.59/62.16, respectively; uniform support weighting is lower at 75.88/60.70. These controls separate continuation optimization, semantic weighting, and projection effects at the checkpoint level; positional encoding and RoI-Conv remain inherited components rather than claimed innovations.

TABLE III  
DEPTH-FILTER PAIRED EXPERIMENT ON SCANNETV2.
<table><tr><td>Cache protocol Depth filter</td><td>Valid</td><td colspan="3">Mean support mAP@0.25 mAP@0.50</td></tr><tr><td>Unfiltered cache None</td><td>100.0%</td><td>2.84</td><td>75.65</td><td>59.75</td></tr><tr><td>Depth-filtered  $| z \rrangle - d | \le 0 . 2 5 \mathrm { m }$ </td><td>65.8%</td><td>1.91</td><td>76.15</td><td>60.93</td></tr><tr><td>cache Difference</td><td>-34.2%</td><td>-0.93</td><td>+0.50</td><td>+1.18</td></tr></table>

The two cache protocols use the same RGB-D frames, point indices, feature extraction, and calibration; only the metric depth-residual filter changes.

TABLE IV  
MATCHED EQUAL-BUDGET ABLATION ON SCANNETV2.
<table><tr><td>Configuration</td><td>LR</td><td>λ</td><td>Support rule</td><td>mAP@0.25 mAP@0.50</td><td></td></tr><tr><td>DINOv3 control</td><td></td><td>0</td><td>None</td><td>76.15</td><td>60.93</td></tr><tr><td>Detection-only contin- uation</td><td> $1 0 ^ { - 5 }$ </td><td>0</td><td>None</td><td>76.22</td><td>61.48</td></tr><tr><td>Uniform VAID</td><td> $1 0 ^ { - 5 }$ </td><td></td><td>0.01 Uniform</td><td>75.88</td><td>60.70</td></tr><tr><td>Energy-weighted VAID</td><td> $1 0 ^ { - 5 }$ </td><td></td><td>0.01 Energy</td><td>76.40</td><td>61.78</td></tr><tr><td>Conservative VAID</td><td> $1 0 ^ { - 5 }$ </td><td></td><td>0.01 Energy + positive RoI</td><td>76.59</td><td>62.16</td></tr></table>

All continuation rows use the same control initialization and one-epoch budget. LR is the continuation learning rate; λ is the auxiliary-loss weight.

4) Category and Visibility Analysis: Table I reports the complete 18-class AP vector for the two principal settings. The per-class rows in Table I show numerical changes and regressions under Conservative VAID, but they do not by themselves establish a visibility-subset effect. The paired cache and equal-budget results above provide aggregate protocol controls; visibility- and shape-subset statistics remain future work because their subset definitions require a fixed, separately measured protocol.

## E. Cross-Dataset and Real-Scene Evaluation

![](images/e586f48bececdc834e093f09dab17618921ff12e915a2ca13d26afc60a725c4e.jpg)  
Fig. 5. Fast-LIVO2 acquisition [15] and input. Fast-LIVO2-fused indoor point cloud obtained from the Livox Avia–camera–Jetson Orin NX sensing platform for the real-scene diagnostic.

![](images/7660dbd3c67e14020c5fddb8ce592b1524b2e719126571f485e5244394fb3303.jpg)  
Fig. 6. Qualitative 3D Detection with Fast-LIVO2 Pointcloud Inputs [15]. Qualitative 3D box predictions from point-cloud inputs obtained from Fast-LIVO2 in an indoor environment.

SUN-RGBD results are reported with the corrected evaluator in Table V. The table reports the DINOv3 online baseline and the synchronized RGB-D augmentation trained for 20 epochs. The DINOv3 online baseline reaches 62.79 mAP@0.25, where mAP@0.25 denotes mean average precision at a 3D IoU threshold of 0.25, while synchronized RGB-D augmentation reaches 62.10 mAP@0.25 and 42.22 mAP@0.50. The external IIFNet3D row uses dashes in columns without comparable values. Short screening variants and failed adapter branches are omitted from this table. The qualitative views are separated by dataset: SUN-RGBD is shown in Fig. 3, and ScanNetV2 validation scenes are shown

![](images/c2994bca8115c36719ead3049b3dc7c84e994148d9df61730a5742d5887ce24d.jpg)  
Fig. 3. Qualitative Detection Results on SUN-RGBD. Representative validation scenes are arranged by column, with rows showing the input image, Ground Truth, and our DINOv3 predictions. Colors identify object instances/classes, and the visualization is qualitative.

![](images/a769f0533268be9a571f6df18e0ddbe2e91887a48889120d0ea00b0114896690.jpg)  
Fig. 4. Qualitative Results on the ScanNet V2 Validation Set. Representative scenes are arranged by column, with rows showing the input point cloud, Ground Truth, and our DINOv3 predictions. The same camera convention is used for the Ground Truth and prediction rows; these views are qualitative and are not used as quantitative evidence.

in Fig. 4.

For a separate real-scene demonstration, we use the collected Fast-LIVO2 RGB-D/LiDAR sequence [15], acquired with the Livox Avia, MV-CA013-21UC camera, and Jetson Orin NX platform. Because the sequence is unannotated, we present the acquisition setup and predicted 3D boxes as a qualitative assessment of the end-to-end pipeline in Fig. 5 and Fig. 6.

These transfer results are used as a protocol diagnostic rather than as a claim of uniform cross-dataset superiority. The synchronized run recovers the stricter-IoU score but remains slightly lower at an IoU threshold of 0.25; these results therefore do not establish uniform cross-dataset gains.

TABLE V  
SUN-RGBD TRANSFER PROTOCOL DIAGNOSTIC.
<table><tr><td>Setting</td><td>Epochs</td><td>Evaluator</td><td>AP25</td><td>AP50</td></tr><tr><td>IIFNet3D (ext.)</td><td></td><td>Original paper</td><td>67.60</td><td></td></tr><tr><td>DINOv3 online baseline</td><td>20</td><td>Corrected</td><td>62.79</td><td>40.98</td></tr><tr><td>DINOv3 + synchronized RGB-D</td><td>20</td><td>Corrected</td><td>62.10</td><td>42.22</td></tr></table>

AP25/AP50 denote mAP at 3D IoU thresholds 0.25/0.50. The external row is not a same-protocol rerun; its AP50 value is unavailable.

## V. DISCUSSION AND CONCLUSION

The reported Conservative VAID checkpoint reaches 76.59/62.16 mAP0.25/0.50, versus 76.15/60.93 for the DINOv3 control. These 0.44/1.23-point differences are checkpoint-level observations of the combined recipe; the visual branch and processing protocol also differ from the external IIFNet3D reference.

## VI. LIMITATIONS AND FUTURE WORK

Our work uses one seed and does not report multi-seed variance; the paired depth-filter and equal-budget VAID results are checkpoint-level observations. AP0.50 is not decomposed into classification, localization, and NMS effects, and offline extraction adds storage and memory costs without an established throughput advantage. SUN-RGBD is a protocol diagnostic, while Fast-LIVO2 is evaluated qualitatively because no 3D ground truth is available; future work will add multi-seed, visibility/localization, throughput, and annotated real-scene analyses.

## REFERENCES

[1] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser and M. Nießner, "ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017, pp. 2432-2443, doi: 10.1109/CVPR.2017.261.

[2] S. Song, S. P. Lichtenberg and J. Xiao, "SUN RGB-D: A RGB-D scene understanding benchmark suite," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, USA, 2015, pp. 567-576, doi: 10.1109/CVPR.2015.7298655.

[3] O. Siméoni et al., “DINOv3,” arXiv preprint arXiv:2508.10104, 2025.

[4] C. R. Qi, O. Litany, K. He and L. Guibas, "Deep Hough Voting for 3D Object Detection in Point Clouds," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 9276-9285, doi: 10.1109/ICCV.2019.00937.

[5] D. Xu, D. Anguelov and A. Jain, "PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 244-253, doi: 10.1109/CVPR.2018.00033.

[6] C. R. Qi, X. Chen, O. Litany and L. J. Guibas, "ImVoteNet: Boosting 3D Object Detection in Point Clouds With Image Votes," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 4403-4412, doi: 10.1109/CVPR42600.2020.00446.

[7] Z. Sun, Z. Fan, B. Fan, and H. Liu, “IIFNet3D: Instance-to-instance fusion with dual attention for indoor RGB-D 3D object detection,” Pattern Recognition, vol. 179, Part A, Art. no. 113541, 2026, doi: 10.1016/j.patcog.2026.113541.

[8] Z. Zhang, B. Sun, H. Yang, and Q. Huang, “H3DNet: 3D Object Detection Using Hybrid Geometric Primitives,” in European Conference on Computer Vision (ECCV), 2020, pp. 311–329.

[9] Z. Liu, Z. Zhang, Y. Cao, H. Hu and X. Tong, "Group-Free 3D Object Detection via Transformers," 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, 2021, pp. 2929-2938, doi: 10.1109/ICCV48922.2021.00294.

[10] I. Misra, R. Girdhar and A. Joulin, "An End-to-End Transformer Model for 3D Object Detection," 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, 2021, pp. 2886-2897, doi: 10.1109/ICCV48922.2021.00290.

[11] J. Gwak, C. Choy, and S. Savarese, “Generative Sparse Detection Networks for 3D Single-Shot Object Detection,” in Computer Vision – ECCV 2020, Lecture Notes in Computer Science, vol. 12349, Springer, Cham, 2020, pp. 297–313, doi: 10.1007/978-3-030-58548-8\_18.

[12] D. Rukhovich, A. Vorontsova and A. Konushin, "TR3D: Towards Real-Time Indoor 3D Object Detection," 2023 IEEE International Conference on Image Processing (ICIP), Kuala Lumpur, Malaysia, 2023, pp. 281-285, doi: 10.1109/ICIP49359.2023.10222644.

[13] M. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," 2021 IEEE/CVF International Conference on Computer Vision (ICCV), Montreal, QC, Canada, 2021, pp. 9630-9640, doi: 10.1109/ICCV48922.2021.00951.

[14] M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” arXiv preprint arXiv:2304.07193, 2023.

[15] C. Zheng et al., "FAST-LIVO2: Fast, Direct LiDAR–Inertial–Visual Odometry," in IEEE Transactions on Robotics, vol. 41, pp. 326-346, 2025, doi: 10.1109/TRO.2024.3502198.

[16] X. Bai et al., "TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 2022, pp. 1080-1089, doi: 10.1109/CVPR52688.2022.00116.

[17] X. Pan, Z. Xia, S. Song, L. E. Li and G. Huang, "3D Object Detection with Pointformer," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021, pp. 7459-7468, doi: 10.1109/CVPR46437.2021.00738.

[18] Q. Xie et al., "MLCVNet: Multi-Level Context VoteNet for 3D Object Detection," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 10444- 10453, doi: 10.1109/CVPR42600.2020.01046.

[19] B. Cheng, L. Sheng, S. Shi, M. Yang and D. Xu, "Back-tracing Representative Points for Voting-based 3D Object Detection in Point Clouds," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021, pp. 8959-8968, doi: 10.1109/CVPR46437.2021.00885.

[20] S. Dong et al., "Semantic-Context Graph Network for Point-Based 3D Object Detection," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 11, pp. 6474-6486, Nov. 2023, doi: 10.1109/TCSVT.2023.3271318.

[21] M. Tao, C. Zhao, M. Tang, and J. Wang, “Objformer: Boosting 3D Object Detection via Instance-Wise Interaction,” Pattern Recognition, vol. 146, Art. no. 110061, 2024, doi: 10.1016/j.patcog.2023.110061.

[22] D. Rukhovich, A. Vorontsova, and A. Konushin, “FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection,” in Computer Vision – ECCV 2022, Lecture Notes in Computer Science, vol. 13663, Springer, Cham, 2022, pp. 477–493, doi: 10.1007/978-3-031-20080- 9\_28.

[23] X. Liu, L. Zhao, B. Fan, J. Lu and H. Liu, "Dynamic Learnable Label Assignment for Indoor 3D Object Detection," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 10, pp. 10134-10147, Oct. 2025, doi: 10.1109/TCSVT.2025.3563083.

[24] Y. Zhu, L. Hui, Y. Shen, and J. Xie, “SPGroup3D: Superpoint Grouping Network for Indoor 3D Object Detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, pp. 7811–7819, 2024, doi: 10.1609/aaai.v38i7.28616.

[25] H. Wang, L. Ding, S. Dong, S. Shi, A. Li, J. Li, Z. Li, and L. Wang, “CAGroup3D: Class-Aware Grouping for 3D Object Detection on Point Clouds,” in Advances in Neural Information Processing Systems 35, 2022, pp. 29975–29988, doi: 10.52202/068431-2173.

[26] Z. Wang, Q. Xie, M. Wei, K. Long, and J. Wang, “Multi-Feature Fusion VoteNet for 3D Object Detection,” ACM Transactions on Multimedia Computing, Communications, and Applications, vol. 18, no. 1, pp. 1–17, 2022, doi: 10.1145/3462219.

[27] A. Chen et al., "PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 5291-5301, doi: 10.1109/CVPR52729.2023.00512.

[28] Y. Wang, X. Chen, L. Cao, W. Huang, F. Sun and Y. Wang, "Multimodal Token Fusion for Vision Transformers," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 2022, pp. 12176-12185, doi: 10.1109/CVPR52688.2022.01187.