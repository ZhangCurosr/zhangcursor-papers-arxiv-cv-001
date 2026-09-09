# FPicker: Topology-Guided Evolution for Filament Tracing in Low-SNR Microscopy<sup>⋆</sup>

Tingyin Zhao<sup>1,2</sup> , Mingtao Huang<sup>1,2,\*</sup> , and Yuan Shen<sup>1,2,\*</sup>

<sup>1</sup>Department of Electronic Engineering, Tsinghua University, Beijing, China <sup>2</sup>Beijing National Research Center for Information Science and Technology, Beijing, China

zhaoty25@mails.tsinghua.edu.cn, huangmt@mail.tsinghua.edu.cn, shenyuan\_ee@tsinghua.edu.cn

Abstract. Automating filament tracing in Cryo-Electron Microscopy (Cryo-EM) is essential for 3D helical reconstruction but challenged by intersecting topologies and extremely low Signal-to-Noise Ratios (SNR = $\sigma _ { s } ^ { 2 } / \sigma _ { n } ^ { 2 } < 0 . 1$ or -10 dB). Existing paradigms fail: pixel-wise segmenters sufer from severe topological fracturing, box-based detectors face ghost center drift, sequential trackers derail due to error accumulation, and traditional active contours collapse under artificial closed-curve constraints. To resolve these bottlenecks, we present FPicker, the first topologyguided framework reconciling these incompatibilities. It unifies perception via a center-endpoint representation and an open-curve evolution module to explicitly model non-cyclic connectivity. On simulated benchmarks, FPicker outperforms top baselines by over 40% relative gain in mean spatio-angular precision (mSAP) and reduces topological gap rates by over 60% under extreme noise (−20 dB). By learning intrinsic physical geometry rather than local texture, FPicker demonstrates strong potential as a resilient geometric backbone. Its zero-shot performance on the real-world EMPIAR dataset exhibits robust topological resistance, achieving a state-of-the-art 82.9% mSAP upon fine-tuning. Our results also suggest modeling physical priors is a highly robust path toward bridging the sim-to-real gap in signal-starved scientific imaging. The code is publicly available at: https://github.com/tomzhaosky/FPicker.

Keywords: Filament Tracing · Open-Curve Evolution · Microscopic Imaging · Low-SNR Perception · AI for Science

## 1 Introduction

Elucidating the atomic structure of filamentous proteins, such as amyloid fibrils involved in neurodegenerative diseases, is a cornerstone of modern structural biology and pharmaceutical science [26, 31, 42]. Cryo-Electron Microscopy (Cryo-EM) has revolutionized this field; however, the initial step of filament picking—identifying and tracing individual instances from noisy micrographs, which explicitly enables accurate helical parameter estimation and subsequent 3D helical reconstruction [7, 10]—remains a laborious bottleneck. While automated picking for globular particles is well-established, filaments pose a unique topological challenge: they appear as flexible, continuous curvilinear structures that frequently run parallel, overlap, or intersect with one another [36,40]. Tracing such complex structures is a fundamental problem that pervades biomedical vision, encompassing tasks like microtubule tracking in fluorescence microscopy and DNA strand tracing in nanoscale imaging [8,25,37]. However, Cryo-EM represents an extreme case of this challenge because the filaments are embedded in vitreous ice, resulting in an extremely low Signal-to-Noise Ratio $( \mathrm { S N R } = \sigma _ { s } ^ { 2 } / \sigma _ { n } ^ { 2 }$ < 0.1 or -10 dB) where the target signal is often indistinguishable from background noise at a local level.

![](images/c2a8adf509eb035a714b04038ab13c67e99ac47fe58558ef4f04ceeb60a10c3a.jpg)  
Fig. 1: Visualizing the topological gap and FPicker evolution. Left: Geometric incompatibility of existing paradigms under extreme noise. Box-based detectors sufer from ghost centers (top), while segmentation models succumb to topological fracturing (bottom). Right: FPicker paradigm. Stage I (top) generates a linear topological prior via a center-endpoint representation. Stage II (bottom) iteratively refines this rigid prior (T = 1 → 3) via open-curve evolution. Please zoom in to see the details.

Existing vision paradigms face fundamental geometric incompatibilities here. Box-based detectors (e.g., crYOLO [38], YOLOv8 [12], etc.) trace filaments by linking sequential boxes. However, for thin flexible curves, a bounding box inherently encapsulates mostly background noise. This forces manual tuning of box sizes and causes the geometric centers to drift into signal-free regions (the ghost center degeneracy), leading to broken trajectories. Conversely, segmentation models (e.g., Topaz [3], SAM 2 [29], etc.) rely on a dense segment-thenskeletonize pipeline. While these models efectively extract intersecting contours, 2D projections force distinct filaments to share spatial pixels, creating ambiguities in disentangling individual instances and yielding merged topologies.

To explicitly model continuous topologies, methods turn to sequential tracking or active contours, yet both stumble in extreme noise. Sequential trackers [19, 33] iteratively grow filaments but inherently derail into the signal-free background due to Markovian error accumulation. While active contour models (e.g., Deep Snake [28], CurveGCN [18], etc.) utilize Graph Convolutional Networks (GCNs) [13] to holistically resist such local derailment, they sufer from two fatal flaws. First, standard box-based initializations force them to inherit the aforementioned ghost center problem, causing optimization collapse. Second, directly applying cyclic boundaries to open filaments creates an artificial tension pulling the tips together—a phenomenon we term shrinking degeneracy.

In this paper, we propose to break the closed loop, as conceptually illustrated in Fig. 1. We introduce FPicker (Fiber/Filament Picker), a specialized framework designed for open-topology evolution in Cryo-EM. By replacing box priors with a center-endpoint representation and driving deformation via open-boundary graph convolutions, we ensure the model captures the continuous essence of protein filaments without succumbing to tip retraction. Our contributions are:

– We introduce a center-endpoint representation to resolve the geometric ambiguity and ghost center problem inherent in box-based detection, efectively anchoring flexible filaments in extreme noise.

– We propose an open-curve evolution module to explicitly solve the fundamental endpoint shrinking limitation of traditional active contour models when applied to non-cyclic structures.

– We design a bridge curriculum strategy tailored for end-to-end low-SNR training, successfully decoupling the learning of local deformation physics from global localization variance.

Empirically, by optimizing within a strict physical forward model (Cryo-Sim), FPicker demonstrates strong potential as a resilient geometric backbone. It delivers efective zero-shot transfer and state-of-the-art fine-tuning performance on real-world EMPIAR micrographs. These results suggest that explicitly modeling intrinsic physical priors ofers a fundamentally more robust path for AI for Science in signal-starved domains than brute-force data scaling.

## 2 Related Works

## 2.1 Vector and Box-based Detection

Box-driven detectors (e.g., crYOLO, EPicker, YOLOv8/v10, RT-DETR, CenterNet, etc.) [12, 38, 39, 45–47] excel at globular particles. To trace continuous filaments, they typically detect local segments and link their box centers. We identify a geometric incompatibility here: representing 1D flexible curves with 2D static bounding boxes is intrinsically ill-posed. Because a box inherently encapsulates mostly background noise rather than the thin filament itself, it necessitates tedious manual tuning of anchor sizes. More critically, the geometric center of these boxes frequently falls into the signal-free background (the ghost center degeneracy), derailing accurate linking and feature extraction. FPicker departs from this paradigm by anchoring directly onto the topological backbone.

## 2.2 Segmentation-based Approaches

The segment-then-skeletonize paradigm, widely adopted in general vision and tubular tracing (e.g., Topaz, U-Net, SAM 2, DeepVesselNet, etc.) [2,3,22,23,29, 30, 35, 41], avoids box constraints. While efective at extracting overall filament contours, these methods struggle to distinguish individual instances. Because Cryo-EM micrographs are inherently 2D projections, intersecting filaments share identical spatial pixels. Consequently, even when advanced continuity constraints are applied to mitigate fracturing, grid-bound representations inevitably merge distinct instances at 2D crossovers. Furthermore, non-diferentiable skeletonization decouples topology from learning, preventing the network from utilizing global connectivity to heal noise-induced gaps or resolve junctions.

## 2.3 Explicit Topological Tracking and Evolution

Methods explicitly modeling topology fall into two paradigms: sequential tracking and active contours. Sequential trackers [4, 19, 20, 33], including those utilizing Bayesian probabilistic inference [9], iteratively extend trajectories from local seeds. However, under extreme Cryo-EM noise, misguided local gradients cause trackers to irreversibly derail into the background due to Markovian error accumulation. Conversely, deep active contours (e.g., Deep Snake, CurveGCN, BoundaryFormer, DSAC, RLS, etc.) [14, 15, 18, 24, 28] holistically deform explicit graphs, naturally resisting local derailment. Yet, directly adapting them to open filaments faces two fatal bottlenecks: an initialization trap (inheriting box-based ghost centers) and a manifold mismatch (cyclic convolutions artificially shrink open endpoints together). Ultimately, FPicker synergizes the opentopology essence of sequential tracking with the holistic deformation of active contours, resolving their respective bottlenecks via a center-endpoint prior and strictly open-boundary evolution.

## 3 Methods

## 3.1 Overview

The overall architecture of FPicker is illustrated in Fig. 2. We formulate filament tracing as a unified, end-to-end trainable framework that maps a raw Cryo-EM micrograph I to a set of precise, open skeletal trajectories F. The pipeline follows a coarse-to-fine paradigm, consisting of a shared feature extractor and two specialized stages, which are all gradient accessible:

1. Shared Feature Extraction: A deep encoder-decoder backbone [6,21,43] processes the micrograph I into a high-resolution semantic feature map F ∈ R<sup>H/s×W/s×C</sup> (stride s). F serves as the unified representation for detection and vertex-level refinement.

![](images/aacdf3ff1c76d076be6a0841771fcdbefba8f8ff162fc467af78eb1e739bee96.jpg)  
Fig. 2: FPicker Architecture. The shared backbone extracts a high-resolution feature map F to drive a two-stage pipeline. Stage I generates linear skeletal priors using a center-endpoint representation. Stage II iteratively refines these priors via an opencurve snake module. During training, a bridge curriculum strategy (dashed box) and an auxiliary geometric head ensure stable convergence in extreme noise conditions.

2. Proposal Generation (Stage I): Operating on the feature map F, a multihead branch predicts dense topological heatmaps and center-endpoint fields, aggregating them into linear skeletal priors $\hat { S } _ { \mathrm { i n i t } }$ to anchor instances despite extreme noise.

3. Topological Refinement (Stage II): The linear skeletons serve as the initial state for the open-curve evolution module. For each instance, vertexspecific features are bilinearly sampled from F. A graph convolutional network (GCN) then iteratively deforms the rigid priors into flexible curves that snap to the true protein skeletal trajectories.

To bridge the optimization gap between these stages in low-SNR regimes, we introduce a bridge curriculum strategy and auxiliary geometric fields to enforce structural consistency during the training process.

## 3.2 Topology-Aware Proposal Generation

To enforce structural alignment, we redefine the target as a topological centroid $\mathbf { c } _ { i } .$ , strictly constrained to the filament trajectory $\mathcal { G } _ { i } ( \boldsymbol { \ell } )$ at the arc-length median: $\mathbf { c } _ { i } = \mathcal { G } _ { i } ( L _ { i } / 2 )$ . This ensures the detector anchors onto discriminative protein density rather than background artifacts.

To extract discrete instances, we apply a $, 3 \times 3$ max-pooling operation over the topological heatmaps for non-maximum suppression (NMS) [47]. Crucially, by operating directly on the smoothed semantic feature space, this topological peak extraction is highly stable and completely circumvents the need for bounding box regression. Local peaks form discrete valid centroids $\mathbf { c } _ { i }$

Complementing this, we regress dense displacement fields $\{ \hat { \mathbf { d } } _ { \mathrm { s t a r t } } , \hat { \mathbf { d } } _ { \mathrm { e n d } } \}$ from these centers. By uniformly interpolating N vertices between $\mathbf { c } _ { i } + \mathbf { d } _ { \mathrm { s t a r t } }$ and $\mathbf { c } _ { i } + \hat { \mathbf { d } } _ { \mathrm { e n d } }$ , we establish the linear topological prior $\hat { S } _ { \mathrm { i n i t } } .$ , which acts as a geometric attractor, defining a rotation-aware capture range that ensures even highcurvature sinusoidal fibrils fall within the subsequent GCN’s receptive field.

## 3.3 Open-Curve Evolution

While the linear skeleton $\hat { S } _ { \mathrm { i n i t } }$ provides a coarse localization, biological filaments possess non-linear elasticity and variable curvatures. To capture these fine-grained geometries, we adapt the active contour framework into a topologyspecific open-curve evolution module. Unlike segmentation-based refinement, our approach treats the filament as a continuous, deformable entity, ensuring axial integrity even in fragmented signal regions.

Graph Construction and Feature Sampling. Given a predicted linear skeleton $\hat { S } _ { \mathrm { i n i t } }$ generated from the topological centroid cˆ and displacement vectors $\{ \hat { \mathbf { d } } _ { \mathrm { s t a r t } } , \hat { \mathbf { d } } _ { \mathrm { e n d } } \}$ , we uniformly sample N vertices $\hat { \mathcal { V } } = \{ \hat { \mathbf { v } } _ { i } \} _ { i = 1 } ^ { N }$ along the axis. Unlike traditional closed-polygon snakes, we construct an open graph connecting each internal vertex $\hat { \mathbf { v } } _ { i } \ ( 1 < i < N )$ only to its immediate neighbors.

For each $\hat { \mathbf { v } } _ { i } ~ = ~ ( \hat { x } _ { i } , \hat { y } _ { i } )$ sampled along the initial skeleton, we extract a feature vector $\mathbf { f } _ { i }$ from F via bilinear interpolation: $\begin{array} { r l } { \mathbf { f } _ { i } } & { { } = } \end{array}$ Interpolate $( \mathbf { F } , \hat { \mathbf { v } } _ { i } )$ . Interpolating directly along the skeleton yields a deformable receptive field precisely tailored to the true trajectory. This feature-centric approach avoids explicit geometric embeddings prone to noise-induced drift and bypasses the ghost center vulnerability inherent in box-based RoI pooling.

![](images/336544f2718cbe0e66dc61408431ce1ba8a9366e248ea62f66f57f80ebcdfc63.jpg)

Open-Boundary Graph Convolution (OBC). Standard active contour implementations employ circular convolution, enforcing periodic boundary conditions where vertex $v _ { N }$ connects to $v _ { 1 }$ . This introduces a phantom tension for open filaments, causing tip retraction. To achieve open-curve evolution for continuous filaments, we reformulate the traditional cyclic GCNs by strictly imposing open-boundary conditions during feature aggregation.

Fig. 3: Dual-Stream Architecture. Local (OBC) and Global feature streams fuse to guide deformation.

Mathematically, we employ replicate padding at the graph boundaries. For a graph signal $\mathbf { f } \in \mathbb { R } ^ { N }$ , the feature aggregation for a vertex i is constrained to its geodesic neighbors:

$$
\mathbf { f } _ { i } ^ { \prime } = \sum _ { j = - d } ^ { d } \mathbf { W } _ { j } \cdot \mathbf { f } _ { \operatorname* { m a x } ( 1 , \operatorname* { m i n } ( N , i + j ) ) }\tag{1}
$$

This truncation breaks the gradient flow between logically distant endpoints, setting the tension at the tips to zero and preventing the shrinking degeneracy.

Iterative Deformation with Global Structural Anchoring. In extreme noise, local receptive fields are often dominated by stochastic background fluctuations, causing individual vertices to drift. To ensure structural consistency without relying on fragile coordinate assumptions, we introduce a global structural anchoring mechanism.

At each iteration t, vertex features $\mathbf { F } ^ { ( t ) }$ are processed through two parallel streams, as illustrated in Fig. 3. The local stream extracts geometric afinities via stacked OBC layers, while the global stream employs a max-pooling operation to capture the most salient signal. Crucially, because the shared feature F is explicitly regularized by auxiliary low-pass geometric fields in Section 3.4, this pooling operation naturally resists high-frequency noise disturbance:

$$
\mathbf { h } _ { i } ^ { \mathrm { l o c a l } } = \boldsymbol { \varPhi } _ { \mathrm { G C N } } ( \mathbf { f } _ { i } ^ { ( t ) } ) , \quad \mathbf { h } ^ { \mathrm { g l o b a l } } = \operatorname* { m a x } _ { j = 1 } ^ { N } \{ \boldsymbol { \varPhi } _ { \mathrm { i n i t } } ( \mathbf { f } _ { j } ^ { ( t ) } ) \}\tag{2}
$$

We posit that in low-SNR regimes, data-dependent priors outperform independent ones. Traditional positional embeddings impose rigid absolute coordinates that mislead when the initial snake drifts due to noise. In contrast, our $\mathbf { h } ^ { \mathrm { \varepsilon } }$ global is data-dependent: it dynamically aggregates the strongest filament existence signals (e.g., from a clearly visible segment) regardless of absolute position.

This global descriptor acts as a semantic anchor, broadcasting high-confidence structural cues to vertices in low-signal regions. The deformation ofset is then predicted by applying a learnable weight matrix $\mathbf { W _ { \mathrm { o f f s e t } } }$ and bias $\mathbf { b } _ { \mathrm { o f f s e t } }$ to the concatenated features:

$$
\hat { \mathbf { v } } _ { i } ^ { ( t + 1 ) } = \hat { \mathbf { v } } _ { i } ^ { ( t ) } + \Delta \hat { \mathbf { v } } _ { i } ^ { ( t ) } , \quad \Delta \hat { \mathbf { v } } _ { i } ^ { ( t ) } = \mathbf { W } _ { \mathrm { o f f s e t } } \big [ \mathbf { h } _ { i } ^ { \mathrm { l o c a l } } ~ \mathbf { h } ^ { \mathrm { g l o b a l } } \big ] ^ { \top } + \mathbf { b } _ { \mathrm { o f f s e t } }\tag{3}
$$

By conditioning deformation on this global context, the network efectively hallucinates the correct trajectory for noise-submerged vertices, preventing the fragmentation typical of local-only methods.

(a) Original Micrograph (Low SNR)  
![](images/cfb214fd6da7a6cb3375346d3f79df23b4d08eb6fd055241285bfc3688d608ba.jpg)

(b) Predicted Heatmap (HM)  
![](images/31d10cbbedb25a8b18bd38430054e2bc3d5b429774a4ef2ed83851c771d8d1d7.jpg)

(c) Predicted Distance Field (D)  
![](images/76d7bdc214d04a6dd98f93068986eb15d3af1377d5237cd2cf8d02db93b35e90.jpg)

(d) Predicted Angle Field (A)  
![](images/1a986e5aa1ecd8ce37969bd56f154e75009e3ded2b5ed4974e3660ad008910ca.jpg)  
Fig. 4: Visualization of Auxiliary Fields. (a) Raw input micrograph. (b) Predicted topological centroid heatmap. (c) Distance Field (D<sup>ˆ</sup>), revealing smooth tubular manifolds. (d) Angle Field (A<sup>ˆ</sup>) in HSV space (Hue: tangent direction; Value: distance). The continuous color gradients demonstrate robust learning of intrinsic filament geometry.

## 3.4 Auxiliary Geometric Field Learning.

While the evolution module efectively captures the filament’s elasticity, its convergence heavily relies on the quality of the underlying feature map. In extreme low-SNR regimes, intensity-based features are prone to local minima. To regularize the backbone and enforce structural awareness, we attach an auxiliary geometric branch to the backbone to regress two dense geometric fields: a Distance Field (D) representing Euclidean proximity to the nearest trajectory, and a 2-channel Angle Field (A) encoding local tangent directions (cos θ, sin θ).

Inspired by DeepLSD [27], our dual-field representation establishes robust dense geometric supervision. While conceptually resembling Local Shape Descriptors (LSDs) [32], FPicker departs from their algorithmic role: rather than strengthening local pixel classifications, our fields act as continuous low-pass regularizers guiding sparse center-endpoint proposals and holistic open-curve graph evolution, as visually demonstrated in Fig. 4. Consequently, unless supported by a coherent center-endpoint prior, the impact of spurious auxiliary-field artifacts is mitigated, rendering our framework resilient to Contrast Transfer Function (CTF) oscillations inherent in Cryo-EM defocus imaging.

## 3.5 The Bridge Curriculum Strategy

End-to-end training in extreme noise faces a cold start dilemma: the evolution module requires stable initialization to learn topology, yet the early-stage detector yields chaotic proposals. To prevent optimization collapse, we propose the bridge curriculum strategy, designed to decouple the learning of local deformation physics from global localization variance. We employ a transition regulator $P _ { \mathrm { g t } }$ to modulate the input $S _ { \mathrm { { i n p u t } } }$

$$
\begin{array} { r } {  { S _ { \mathrm { i n p u t } } } = \beta (  { S _ { \mathrm { g t } } } + \epsilon ) + ( 1 - \beta )  { \hat { S } _ { \mathrm { p r e d } } } , \quad \mathrm { w h e r e } \quad \beta \sim \mathrm { B e r n o u l l i } ( P _ { \mathrm { g t } } ) } \end{array}\tag{4}
$$

where ϵ denotes Gaussian perturbation. Prior to this element-wise modulation, both the ground-truth skeleton ${ \mathcal S } _ { \mathrm { g t } }$ and the predicted linear prior $\hat { S } _ { \mathrm { p r e d } }$ are uniformly resampled to exactly N equidistant vertices $( \mathrm { e . g . } , N = 1 2 8 )$ to guarantee strict dimensional alignment. Unlike standard teacher forcing, this stochastic injection forces the evolution module to first master the physical laws of feature attraction $( \mathrm { i . e . }$ , snapping to ridge intensity) within a controlled capture range $( P _ { \mathrm { g t } } = 1$ for the first 10 epochs), independent of detection drift. As the detector stabilizes, we linearly anneal $P _ { \mathrm { g t } } \to 0$ over the subsequent 110 epochs, gradually exposing the refinement module to the inherent variance of predicted priors until fully autonomous inference is achieved.

## 3.6 Loss Function

The FPicker framework is trained via a multi-task objective function. The total loss $\mathcal { L } _ { t o t a l }$ is defined as a weighted sum of detection, geometric, and evolution

components, ensuring the model balances coarse localization with fine-grained structural alignment:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { h m } } { \mathcal { L } } _ { \mathrm { h m } } + \lambda _ { \mathrm { r e g } } { \mathcal { L } } _ { \mathrm { r e g } } + \lambda _ { \mathrm { e n d s } } { \mathcal { L } } _ { \mathrm { e n d s } } + \lambda _ { \mathrm { a u x } } { \mathcal { L } } _ { \mathrm { a u x } } + \lambda _ { \mathrm { e v o l } } { \mathcal { L } } _ { \mathrm { e v o l } }\tag{5}
$$

where λ are coeficients used to balance the contribution of each task.

Topology-Anchored Heatmap Loss $\left( \mathcal { L } _ { \mathbf { h m } } \right)$ . To handle extreme foregroundbackground imbalance, we employ a modified Focal Loss [17]. Inspired by the objective formulation of CenterNet [47], our loss is tailored to anchor strictly onto the topological centroids $\mathbf { c } _ { i } = \mathcal { G } _ { i } ( L _ { i } / 2 )$ rather than ill-posed bounding box centers. Let $\hat { Y } ( \mathbf { p } )$ and $Y ( \mathbf { p } )$ denote the predicted and ground-truth Gaussian maps at spatial location $\mathbf { p } \in \varOmega$ . The loss is defined as:

$$
{ \mathcal { L } } _ { \mathrm { h m } } = - { \frac { 1 } { N } } \sum _ { \mathbf { p } \in { \mathcal { Q } } } { \left\{ ( 1 - { \hat { Y } } ( \mathbf { p } ) ) ^ { \alpha } \log ( { \hat { Y } } ( \mathbf { p } ) ) \right. } _ { ( 1 - Y ( \mathbf { p } ) ) ^ { \beta } \left( { \hat { Y } } ( \mathbf { p } ) \right) ^ { \alpha } \log ( 1 - { \hat { Y } } ( \mathbf { p } ) ) } { \left. \quad { \mathrm { o t h e r w i s e } } \quad \right. }\tag{6}
$$

where N is the number of valid filaments, and $\alpha = 2 , \beta = 4$ are hyper-parameters controlling the penalty landscape around the topological medians.

Regression Losses $( { \mathcal { L } } _ { \mathbf { r e g } }$ and $\pmb { \mathcal { L } _ { \mathrm { e n d s } } } )$ . The center ofset map oˆ and endpoint vector map d<sup>ˆ</sup> are trained using L1 loss. Crucially, to avoid noise interference, we apply a sparse supervision strategy: the loss is computed only at the ground-truth center locations $\mathbf { c } _ { k }$

$$
\mathcal { L } _ { \mathrm { e n d s } } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left\| \hat { \mathbf { d } } ( \mathbf { c } _ { k } ) - \mathbf { d } _ { k } \right\| _ { 1 } , \quad \mathcal { L } _ { \mathrm { r e g } } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left\| \hat { \mathbf { o } } ( \mathbf { c } _ { k } ) - \left( \frac { \mathbf { c } _ { k } } { s } - \lfloor \frac { \mathbf { c } _ { k } } { s } \rfloor \right) \right\| _ { 1 }\tag{7}
$$

where $s$ is the output stride. This forces the network to focus its capacity on valid structural instances.

Auxiliary Geometric Loss $\left( \mathcal { L } _ { \bf a u x } \right)$ . Unlike the sparse regression heads, the auxiliary fields provide dense supervision. We compute the L1 diference between the predicted fields $( \hat { \mathcal { D } } , \hat { \mathbf { A } } )$ and the ground truth $( \mathcal { D } , \mathbf { A } )$ over the entire valid fiber mask M:

$$
\mathcal { L } _ { \mathrm { a u x } } = \frac { 1 } { | M | } \sum _ { ( x , y ) \in M } \left( | \hat { \mathcal { D } } _ { x y } - \mathcal { D } _ { x y } | + \| \hat { \mathbf { A } } _ { x y } - \mathbf { A } _ { x y } \| _ { 2 } \right)\tag{8}
$$

This term acts as a regularizer, preventing the backbone from overfitting to background noise patterns.

Open-Curve Evolution Loss $( \mathcal { L } _ { \mathrm { e v o l } } )$ . To capture the flexible geometry of filaments while maintaining structural integrity, we define the topological refinement loss $\mathcal { L } _ { \mathrm { e v o l } } = \mathcal { L } _ { \mathrm { f i t } } + \lambda _ { \mathrm { u n i } } \mathcal { L } _ { \mathrm { u n i } }$ . Here, ${ \mathcal { L } } _ { \mathrm { f i t } }$ employs the Smooth-L1 criterion to minimize the discrepancy between evolved vertices V<sup>ˆ</sup> and ground-truth points V, while ${ \mathcal { L } } _ { \mathrm { u n i } }$ penalizes edge-length variance to prevent vertex clustering in noisedominated regions:

$$
\mathcal { L } _ { \mathrm { f i t } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { S m o o t h } _ { \mathrm { L } 1 } ( \hat { \mathbf { v } } _ { i } , \mathbf { v } _ { i } ) , \quad \mathcal { L } _ { \mathrm { u n i } } = \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N - 1 } | \| \hat { \mathbf { v } } _ { i + 1 } - \hat { \mathbf { v } } _ { i } \| _ { 2 } - \bar { d } |\tag{9}
$$

where $\bar { d }$ denotes the average predicted edge length. This combined objective ensures that the open curve not only accurately snaps to the true skeleton topology but also maintains a physically plausible, equidistant vertex distribution, which is critical for bridging fragmented signals in extreme low-SNR environments.

## 4 Experiments

## 4.1 Experimental Setup

Datasets and Benchmarks. Lacking open-source datasets with pixel-level filament annotations, we eschew large-scale supervised training on real data. We utilize: (1) Cryo-Sim: An electron optics simulation engine generating 20,000 micrographs (explicitly modeling CTF physics like spherical aberration, amplitude contrast, and B-factor decay; full details in the Supplementary Material). It contains micrographs across High (SNR 0.1, -10 dB), Medium (0.05, -13 dB), and Extreme (0.01, -20 dB) noise regimes. (2) Custom-EMPIAR: A realworld dataset of 500 manually annotated micrographs from EMPIAR-10230 and EMPIAR-10340 [5, 11, 44], containing ∼5,500 individual filaments. Models are trained/tested on Cryo-Sim, evaluated zero-shot on EMPIAR, and finally finetuned on EMPIAR. Both datasets can be found in open-sourced materials.

Implementation & Metrics. FPicker is implemented in PyTorch and trained on 5 RTX 4090 GPUs using Adam (LR 1e − 4) for 200 epochs. To comprehensively assess performance, we first establish mSAP (mean spatioangular precision) as our core matching criterion, evaluating spatial Chamfer distance [1] under a strict angular tolerance $( \varDelta \theta \ < \ 1 5 ^ { \circ } )$ . Based on mSAP true positives, we compute standard detection metrics: F1-score and gap rate $( 1 - \mathrm { m A R } )$ . For structural continuity, we evaluate standard clDice [34] alongside our proposed fp-clDice, which incorporates an exponential fragmentation penalty $( \exp ( - 0 . 1 ( N _ { \mathrm { f r a g } } - 1 ) ) )$ to heavily penalize dashed predictions. Finally, we report P-Ang (penalized angle error), assigning a 90<sup>◦</sup> penalty to unrecovered instances to prevent selection bias. Preliminary variance analysis of FPicker performance is also included in the supplementary material.

## 4.2 Comparison with State-of-the-Arts

We benchmark FPicker against three dominant paradigms: Box-based Detection (crYOLO, YOLOv8), Pixel-wise Segmentation (Topaz, U-Net, SegFormer, SAM 2), and Closed Active Contours (Deep Snake, CurveGCN). While Table 1 reports the quantitative performance across diferent noise regimes, Fig. 5 provides the corresponding qualitative results that visually demonstrate FPicker’s robustness under extreme noise.

Ghost Center Collapse (Box-based Methods). In High SNR regimes, box-based methods like YOLOv8 and crYOLO perform robustly (mSAP 78.5% and 75.2%). However, performance degrades drastically under Extreme noise (- 20 dB), with YOLOv8 and crYOLO plummeting to 28.5% and 25.4% mSAP, respectively. This confirms our hypothesis regarding the ghost center problem: for thin, curved filaments buried in noise, the geometric center of a bounding box often resides in the background signal. Without a strong visual feature at the anchor point, the detector struggles to converge, resulting in severe recall failure (e.g., YOLOv8 Gap Rate 45.2%).

![](images/3a30ca1e9ce9234ed7bab59cd44aa38196b37edd0fec1349c77251a632f4ea03.jpg)  
Fig. 5: Qualitative Comparison. Box detectors show severe discontinuities, while segmenters merge instances at crossovers. Closed contours perform poorly at endpoints, whereas FPicker disentangles junctions and recovers continuous skeletons under extreme noise. Colors distinguish instances. Please zoom in to see the details.

Topological Fracturing (Segmentation Methods). While U-Net and SegFormer maintain better localization than box detectors at Medium SNR (mSAP ∼62-64%), they sufer from severe topological inconsistency. This is quantified by their high Gap Rates (18.5% and 16.2% respectively) compared to FPicker (6.3%). The pixel-wise independence assumption causes these models to treat noise fluctuations as boundaries, resulting in dashed line predictions that require heavy post-processing to repair. Despite using a skeletal outline slightly larger than the ground truth—an exceptionally strong prompt bordering on explicit localization—SAM 2 yields an mSAP of only 12.1% at Extreme SNR. SAM 2 fundamentally fails because its attention mechanism relies on highfrequency, texture-based afinities, which systematically collapse in the textureless, shot-noise-dominated Cryo-EM domain.

Table 1: Main Comparison across SNR regimes on Cryo-Sim. Comparisons are grouped by paradigm. F1: F1-Score (% ↑), clD: clDice (% ↑), mSAP: Spatio-Angular Precision (% ↑), fp-clD: Fragmentation-Penalized clDice (% ↑), Gap: Gap Rate (% ↓), P-Ang: Penalized Tangent Error (<sup>◦</sup> ↓).
<table><tr><td rowspan="2">Method</td><td rowspan="2">F1↑</td><td colspan="5">High SNR (0.1/-10 dB) clD↑ mSAP↑ fp-clD↑</td><td colspan="5">Medium SNR (0.05/-13 dB) clD↑ mSAP↑ fp-clD↑</td><td rowspan="2"></td><td colspan="5">Extreme SNR (0.01/-20 dB) clD↑ mSAP↑ fp-clD↑</td></tr><tr><td></td><td></td><td></td><td></td><td>Gap↓ P-Ang↓</td><td>F1↑</td><td></td><td></td><td></td><td>Gap↓ P-Ang↓</td><td>F1↑</td><td></td><td></td><td></td><td>Gap↓ P-Ang↓</td></tr><tr><td>Box/Vector-based Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>crYOLO [38]</td><td>79.5</td><td>86.8</td><td>75.2</td><td>85.4</td><td>6.8</td><td>14.2</td><td>55.4 64.1</td><td>48.6</td><td>52.1</td><td>24.5</td><td>48.5</td><td>35.2</td><td>44.1</td><td>25.4</td><td>31.5</td><td>48.2</td><td>64.4</td></tr><tr><td>YOLOv8 [12]</td><td>82.0</td><td>89.2</td><td>78.5</td><td>88.1</td><td>5.1</td><td>12.5</td><td>58.5 68.2</td><td>52.4</td><td>56.5</td><td>20.2</td><td>45.1</td><td>38.5</td><td>48.2</td><td>28.5</td><td>35.4</td><td>45.2</td><td>62.1</td></tr><tr><td>Segmentation-based Models†</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Topaz [3]</td><td>78.5</td><td>86.5</td><td>72.1</td><td>85.4</td><td>6.8</td><td>14.8</td><td>54.2</td><td>64.5 48.1</td><td>52.6</td><td>24.5</td><td>49.8</td><td>34.6</td><td>44.5</td><td>24.2</td><td>31.8</td><td>48.5</td><td>65.4</td></tr><tr><td>U-Net [30]</td><td>89.1</td><td>92.5</td><td>86.4</td><td>91.2</td><td>2.4</td><td>8.2</td><td>68.5</td><td>75.4</td><td>62.1</td><td>61.2 18.5</td><td>32.4</td><td>46.5</td><td>55.4</td><td>38.2</td><td>42.1</td><td>38.5</td><td>55.2</td></tr><tr><td>SegFormer [41]</td><td>89.8</td><td>93.1</td><td>87.1</td><td>91.8</td><td>2.1</td><td>7.9</td><td>71.2 77.1</td><td>64.8</td><td>63.5</td><td>16.2</td><td>29.8</td><td>49.2</td><td>58.1</td><td>41.5</td><td>45.3</td><td>35.4</td><td>52.8</td></tr><tr><td>SAM 2 [29]</td><td>60.1</td><td>68.5</td><td>65.2</td><td>55.2</td><td>45.0</td><td>22.1</td><td>42.1</td><td>48.5 35.4</td><td>28.2</td><td>52.6</td><td>55.4</td><td>18.5</td><td>25.4</td><td>12.1</td><td>15.2</td><td>75.2</td><td>82.5</td></tr><tr><td>Active Contours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Deep Snake [28]</td><td>90.5</td><td>93.5</td><td>88.2</td><td>92.5</td><td>1.5</td><td>9.8</td><td>75.4</td><td>81.5</td><td>68.2</td><td>71.4 12.5</td><td>26.2</td><td>54.5</td><td>62.4</td><td>45.8</td><td>51.2</td><td>28.5</td><td>45.2</td></tr><tr><td>CurveGCN [18]</td><td>90.2</td><td>92.8</td><td>88.5</td><td>91.5</td><td>1.8</td><td>10.5</td><td>76.8</td><td>82.1 71.5</td><td>73.2</td><td>11.8</td><td>24.5</td><td>56.8</td><td>64.1</td><td>48.2</td><td>53.5</td><td>26.2</td><td>42.8</td></tr><tr><td>Ours (Ablation)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FPicker (ResNet-50)</td><td>92.5</td><td>94.2</td><td>89.8</td><td>93.5</td><td>1.5</td><td>2.1</td><td>87.2</td><td>88.8 89.0</td><td>84.5</td><td>7.1</td><td>19.9</td><td>75.4</td><td>81.2</td><td>72.1</td><td>72.5</td><td>12.2</td><td>24.5</td></tr><tr><td>FPicker (Swin-T)</td><td>93.8</td><td>95.5</td><td>91.5</td><td>94.8</td><td>0.8</td><td>1.6</td><td>88.5</td><td>89.4 90.5</td><td>83.9</td><td>7.3</td><td>19.4</td><td>78.2</td><td>83.5</td><td>74.8</td><td>75.8</td><td>10.5</td><td>21.8</td></tr><tr><td>FPicker (DLA-34)</td><td>94.5</td><td>96.1</td><td>92.1</td><td>95.8</td><td>0.7</td><td>1.4</td><td>89.6</td><td>90.7</td><td>91.1</td><td>89.6 6.3</td><td>15.5</td><td>80.1</td><td>85.2</td><td>76.5</td><td>78.5</td><td>9.8</td><td>20.5</td></tr></table>

<sup>†</sup> Note: Segmentation maps are skeletonized using standard medial axis algorithms, which has been widely applied in medical image analysis.

The Shrinking Degeneracy (Closed Active Contours). Deep Snake efectively mitigates fracturing (Gap Rate 12.5% at Medium SNR), validating the benefit of topological modeling. However, it fails geometrically. Its Penalized Angle Error is significantly high (26.2<sup>◦</sup> at Medium SNR) compared to FPicker (15.5<sup>◦</sup>). This is the direct consequence of the endpoint shrinking problem: the cyclic convolution forces the open filament tips to pull towards each other, distorting the tangent direction at the extremities.

FPicker Superiority. FPicker (DLA-34) achieves decisive superiority, particularly in the Extreme SNR regime. It maintains a low Gap Rate (9.8%) and acceptable Angle Error (20.5<sup>◦</sup>), proving that the open-curve evolution successfully decouples internal smoothness from endpoint tension.

## 4.3 Sim-to-Real Transfer on EMPIAR

To evaluate real-world applicability, we benchmark the Sim-to-Real transfer capability on the Custom-EMPIAR dataset (Table 2). Recognizing the extreme domain gap, we evaluate all models in two phases: purely zero-shot (trained only on Cryo-Sim) and fine-tuned (adapted on EMPIAR).

The substantial performance drop across all models in the zero-shot phase highlights the severe noise distribution of real micrographs. Yet, FPicker maintains the highest topological resilience. In the zero-shot regime, FPicker achieves a clDice of 42.4% and an mSAP of 28.6%. While the zero-shot precision reflects the immense domain gap, FPicker heavily outperforms the fine-tuned versions of established detectors (e.g., crYOLO reaches 25.4% and Topaz achieves 26.2% mSAP) in recall and topology conservation. Meanwhile, texture-reliant foundation models like SAM 2 experience severe zero-shot degradation (1.5% mSAP).

Table 2: Transfer Performance of FPicker (DLA-34) on the Custom-EMPIAR Real-World Dataset. Format: Zero-Shot / Fine-Tuned
<table><tr><td>Method</td><td colspan="2">F1↑</td><td colspan="2">clD↑</td><td colspan="2">mSAP↑</td><td colspan="2">fp-clD↑</td><td colspan="2">Gap↓</td><td colspan="2">P-Ang↓</td></tr><tr><td>crYOLO</td><td>8.2</td><td>/38.6</td><td>15.4</td><td>/33.2</td><td>5.1</td><td>/25.4</td><td>6.5</td><td>/18.5</td><td>83.4</td><td>/56.5</td><td>81.2</td><td>70.4</td></tr><tr><td>YOLOv8</td><td>10.5</td><td>42.1</td><td>18.2</td><td>38.4</td><td>6.5</td><td>29.8</td><td>8.4</td><td>21.6</td><td>80.5</td><td>51.2</td><td>77.8</td><td>66.5</td></tr><tr><td>Topaz</td><td>7.8</td><td>39.4</td><td>15.1</td><td>34.1</td><td>4.8</td><td>26.2</td><td>6.2</td><td>19.1</td><td>84.1</td><td>55.4</td><td>81.6</td><td>69.2</td></tr><tr><td>U-Net</td><td>15.2</td><td>56.8</td><td>25.8</td><td>49.6</td><td>11.2</td><td>43.2</td><td>13.5</td><td>28.5</td><td>70.5</td><td>40.2</td><td>65.2</td><td>58.4</td></tr><tr><td>SegFormer</td><td>17.6</td><td>60.5</td><td>28.5</td><td>53.4</td><td>12.5</td><td>47.8</td><td>14.8</td><td>33.2</td><td>67.8</td><td>34.5</td><td>63.4</td><td>55.2</td></tr><tr><td>SAM 2</td><td>3.5</td><td>43.2</td><td>6.8</td><td>40.1</td><td>1.5</td><td>29.5</td><td>2.4</td><td>22.4</td><td>92.4</td><td>50.8</td><td>85.2</td><td>70.5</td></tr><tr><td>Deep Snake</td><td>22.4</td><td>68.5</td><td>32.5</td><td>55.2</td><td>19.8</td><td>/58.6</td><td>21.4</td><td>43.5</td><td>58.5</td><td>23.4</td><td>64.2</td><td>48.5</td></tr><tr><td>CurveGCN</td><td>24.5</td><td>71.4</td><td>34.8</td><td>56.5</td><td>22.4</td><td>62.5</td><td>23.5</td><td>46.8</td><td>55.2</td><td>20.5</td><td>61.5</td><td>45.2</td></tr><tr><td>FPicker</td><td>31.5</td><td>81.4 42.4</td><td></td><td>71.4 28.6</td><td></td><td>82.9 34.2</td><td></td><td>67.9 48.2</td><td></td><td>6.8 52.7</td><td></td><td>/13.7</td></tr></table>

Table 3: Fine-tuning Eficiency of FPicker (DLA-34) on the Custom-EMPIAR.
<table><tr><td># real imgs</td><td>5</td><td>10 50</td><td>100 200</td></tr><tr><td>F1↑</td><td></td><td>45.3 56.1 69.8 75.4 79.2</td><td></td></tr><tr><td>clD↑</td><td></td><td>51.4 58.6 64.2 67.8 69.5</td><td></td></tr><tr><td>mSAP↑</td><td></td><td>41.3 55.4 70.1 76.580.4</td><td></td></tr><tr><td>fp-clD↑</td><td></td><td>46.2 54.360.5 63.6 65.8</td><td></td></tr><tr><td>Gap↓</td><td></td><td>32.2 22.314.510.88.5</td><td></td></tr><tr><td>P-Ang↓</td><td></td><td>42.1 31.3 21.4 18.2 15.6</td><td></td></tr></table>

Upon fine-tuning, FPicker reaches a state-of-the-art mSAP of 82.9%, decisively surpassing other fine-tuned baselines. This robust sim-to-real bridging indicates that our open-curve evolution successfully learns the noise-invariant physical geometry of filaments, rather than overfitting to synthetic artifacts, allowing it to seamlessly adapt its visual filters to real-world situations.

## 4.4 Data Eficiency and the Sim-to-Real Paradigm

To further investigate the impact of limited annotations in Cryo-EM, we evaluate the data eficiency of our fine-tuning mechanism. As quantified in Table 3, FPicker exhibits a rapid performance gain even with sparse training data: finetuning with just 10 images nearly doubles the zero-shot mSAP (28.6 → 55.4), and expanding to 50 images yields an mSAP of 70.1. This robust few-shot response validates its baseline feasibility for actual laboratory usage.

Fundamentally, this data-eficient adaptation suggests that the network essentially aligns with the underlying physical manifold rather than memorizing domain-specific noise. While training from scratch under extreme noise frequently leads to optimization collapse, optimizing within a physical forward model (Cryo-Sim) establishes a stable geometric prior. Consequently, on realworld EMPIAR data, the network primarily adapts its low-level visual filters rather than relearning filament definitions from sparse annotations. This observation points toward a promising Sim-to-Real paradigm for biological imaging: explicitly modeling physical priors can help mitigate the reliance on expensive manual annotations.

Nevertheless, FPicker currently serves as a preliminary proof of concept. Fully bridging the gap between synthetic physics and real-world structural discovery remains an open challenge, requiring future explorations into more complex graph and curve topologies to handle broader geometric variations.

## 4.5 Ablation Studies

To verify the individual contribution of each core component in FPicker, we conduct comprehensive ablation studies, as detailed in Table 4.

Center-Endpoint Representation. Baseline Model A (box prior, closed curves) collapses under Extreme noise (-20 dB) with 2.5% mSAP and a 92.5% gap rate. Using our center-endpoint (C-E) representation (Model B) surges EMPIAR mSAP to 52.5% and reduces the Extreme SNR gap rate to 65.4%. This confirms C-E avoids the ghost center degeneracy by anchoring predictions directly onto the topological trajectory instead of the ambiguous background.

Table 4: Component ablation across Cryo-Sim and Custom-EMPIAR. We progressively validate Initialization (Init), Topology (Topo), Bridge Curriculum, and Auxiliary loss (Aux).
<table><tr><td rowspan="2">Model</td><td colspan="4">Init</td><td rowspan="2">F1↑</td><td rowspan="2">Medium SNR (0.05/-13 dB) clD↑ mSAP↑ fp-clD↑ Gap↓ P-Ang↓</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">F1↑</td><td rowspan="2"></td><td rowspan="2">Extreme SNR (0.01/-20 dB) clD↑ mSAP↑fp-clD↑ Gap↓ P-Ang↓</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">F1↑</td><td rowspan="2">EMPIAR (Fine-Tuned) clD↑ mSAP↑fp-clD↑ Gap↓ P-Ang↓</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td>Topo Bridge</td><td></td></tr><tr><td>A (Base)|</td><td>Box</td><td>Closed</td><td></td><td>Aux</td><td>62.1</td><td>70.5</td><td>55.4</td><td>68.2</td><td>25.5</td><td>52.1</td><td>5.8</td><td>15.2</td><td></td><td>3.8</td><td>92.5</td><td>87.5</td><td>38.4</td><td>35.1</td><td>24.5</td><td>22.8</td><td>60.4</td><td>75.2</td></tr><tr><td>B</td><td>C-E</td><td>Closed</td><td></td><td></td><td>73.2</td><td>79.8</td><td>68.5</td><td>76.8</td><td>18.2</td><td>45.2</td><td>18.5</td><td>35.4</td><td>2.5 12.5</td><td>15.2</td><td>65.4</td><td></td><td>58.6</td><td></td><td>56.4</td><td></td><td></td><td>54.8</td></tr><tr><td>C</td><td>C-E</td><td>Open</td><td></td><td></td><td>75.8</td><td>82.5</td><td>70.2</td><td>79.4</td><td>16.5</td><td>19.4</td><td>32.4</td><td>40.2</td><td>35.6</td><td>18.4</td><td>58.5</td><td>75.1 60.8</td><td>60.1</td><td>60.8</td><td>52.5 55.2</td><td>48.4 52.1</td><td>34.2 30.4</td><td>42.3</td></tr><tr><td>D</td><td>C-E</td><td>Open</td><td></td><td>√</td><td>84.1</td><td>88.2</td><td>82.5</td><td>86.2</td><td>10.5</td><td>18.5</td><td>52.4</td><td>62.5</td><td>45.2</td><td>48.5</td><td>35.8</td><td>42.1</td><td>68.5</td><td>67.2</td><td>71.4</td><td>62.5</td><td>22.4</td><td>30.5</td></tr><tr><td>E (Full)]</td><td>C-E</td><td>Open</td><td>√</td><td>√</td><td>89.6</td><td>90.7</td><td>91.1</td><td>89.6</td><td>6.3</td><td>15.5</td><td>80.1</td><td>85.2</td><td>76.5</td><td>78.5</td><td>9.8</td><td>20.5</td><td>81.4</td><td>71.4</td><td>82.9</td><td>67.9</td><td>6.8</td><td>13.7</td></tr></table>

Open-Curve Evolution. Model C introduces the open-curve evolution module. While F1-score gains are moderate, P-Ang drops precipitously (45.2<sup>◦</sup> to 19.4<sup>◦</sup> at Medium SNR; 54.8<sup>◦</sup> to 42.3<sup>◦</sup> on EMPIAR). This proves explicit open-boundary graph convolutions eliminate cyclic phantom tension, solving the shrinking degeneracy at filament endpoints.

## Auxiliary Geometric Fields. Adding

the auxiliary geometric loss (Model D) yields a transformative leap under Extreme noise: mSAP jumps from 35.6% to 45.2%, and gap rate nearly halves. Regressing continuous distance and angle fields forces the backbone to learn a lowpass structural filter, preventing overfitting to high-frequency Poisson noise and ensuring stable evolution guidance.

Bridge Curriculum Strategy. Comparing Model D with the full FPicker (Model E), this curriculum proves vital. Early chaotic proposals otherwise trap refinement in local minima, plateauing Extreme SNR mSAP at 45.2%. Regulating initialization variance allows Model E to reach 76.5% mSAP at Extreme SNR

![](images/a200824b513f6be010002041939e0133790c68d79f63ca84423492ee3394d745.jpg)  
Fig. 6: Iteration Analysis. The solid line represents Cryo-Sim, and the dashed one represents EMPIAR.

and 82.9% on EMPIAR, confirming that decoupling localization variance from topological learning is essential.

Iteration Refinement. Fig. 6 analyzes GCN deformation iterations (T). Performance evolves from a coarse fit at T = 1 (78.5% mSAP) and saturates around T = 3 (91.1% mSAP). Inference speed remains highly eficient (38.3 to 36.9 FPS), demonstrating that global structural anchoring achieves rapid, stable convergence without prohibitive computational overhead.

## 5 Limitations

FPicker precludes unified modeling of branches (e.g., Y-junctions), but true molecular branching is biologically absent in target Cryo-EM filaments. Apparent Y-junctions are merely 2D projection overlaps; resolving them as distinct filaments aligns with 3D reconstruction. Extending FPicker via dynamic node degrees for vascular or neural branching remains future work.

Furthermore, while FPicker’s strong continuity prior successfully bridges noise-induced gaps, it occasionally over-merges aligned fragments (Fig. 5h). Although this Over-Merging Rate (OMR, the fraction of predicted curves covering ≥ 2 GT instances) is intrinsically low (7.96%), explicitly repelling competing tips without sacrificing noise resistance remains an open challenge.

## 6 Conclusions

FPicker presents an efective pathway for addressing geometric incompatibilities in signal-limited regimes. By substituting rigid box priors with a center-endpoint representation and uniting holistic graph evolution with strictly open-boundary constraints, FPicker successfully overcomes local tracking derailment and cyclic shrinking, bridging the long-standing topological gap in filament tracing.

Empirically, FPicker achieves 76.5% mSAP and a 9.8% gap rate under Extreme noise (-20 dB), preventing ghost center collapse and endpoint shrinking. Training on Cryo-Sim cultivates a scalable geometric backbone, delivering efective zero-shot transfer and a state-of-the-art 82.9% fine-tuned mSAP on Custom-EMPIAR. This proves modeling intrinsic physical priors outstrips brute-force data scaling in scientific imaging [16].

## Acknowledgements

This work is supported by the National Key Research and Development Program of China under Grant 2025YFF0515300. We also acknowledge the support of the Deng Feng Fund from the School of Information Science and Technology, Tsinghua University. FPicker is a follow-up work to EPicker, developed by our group for particle extraction in Cryo-EM images. We extend our sincere gratitude to all researchers in the SGroup whose foundational eforts and prior contributions paved the way for this research.

## References

1. Barrow, H.G., Tenenbaum, J.M., Bolles, R.C., Wolf, H.C.: Parametric correspondence and chamfer matching: Two new techniques for image matching (1977)

2. Bastani, F., He, S., Abbar, S., Alizadeh, M., Balakrishnan, H., Chawla, S., Madden, S., DeWitt, D.: Roadtracer: Automatic extraction of road networks from aerial images. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4720–4728 (2018)

3. Bepler, T., Morin, A., Rapp, M., Brasch, J., Shapiro, L., Noble, A.J., Berger, B.: Positive-unlabeled convolutional neural networks for particle picking in cryoelectron micrographs. Nature methods 16(11), 1153–1160 (2019)

4. Dai, T., Dubois, M., Arulkumaran, K., Campbell, J., Bass, C., Billot, B., Uslu, F., De Paola, V., Clopath, C., Bharath, A.A.: Deep reinforcement learning for subpixel neural tracking. In: International conference on medical imaging with deep learning. pp. 130–150. PMLR (2019)

5. Falcon, B., Zhang, W., Schweighauser, M., Murzin, A.G., Vidal, R., Garringer, H.J., Ghetti, B., Scheres, S.H., Goedert, M.: Tau filaments from multiple cases of sporadic and inherited alzheimer’s disease adopt a common fold. Acta neuropathologica 136(5), 699–708 (2018)

6. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

7. He, S., Scheres, S.H.: Helical reconstruction in relion. Journal of structural biology 198(3), 163–176 (2017)

8. Holmes, E.P., Gamill, M.C., Provan, J.I., Wiggins, L., Rusková, R., Whittle, S., Catley, T.E., Main, K.H., Shephard, N., Bryant, H.E., et al.: Quantifying complexity in dna structures with high resolution atomic force microscopy. Nature Communications 16(1), 5482 (2025)

9. Huang, M., Li, T., Huang, W., Li, X., Shen, Y.: A bayesian method for tracing filamentous structures in cryo-electron microscopy images. In: 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). pp. 1–4. IEEE (2025)

10. Huang, M., Ma, J., Fu, X., Wang, H., Shen, Y., Li, X.: Accurate helical parameter estimation based on cylindrical unrolling. Structure 33(9), 1603–1613 (2025)

11. Iudin, A., Korir, P.K., Salavert-Torres, J., Kleywegt, G.J., Patwardhan, A.: Empiar: a public archive for raw electron microscopy image data. Nature methods 13(5), 387–388 (2016)

12. Jocher, G., Chaurasia, A., Qiu, J.: Ultralytics yolov8 (2023), version 8.0.0, url: https://github.com/ultralytics/ultralytics, accessed: 2026-06-24

13. Kipf, T.N., Welling, M.: Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016)

14. Lazarow, J., Xu, W., Tu, Z.: Instance segmentation with mask-supervised polygonal boundary transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4382–4391 (2022)

15. Le, T.H.N., Quach, K.G., Luu, K., Duong, C.N., Savvides, M.: Reformulating level sets as deep recurrent neural network approach to semantic segmentation. IEEE Transactions on Image Processing 27(5), 2393–2407 (2018)

16. LeCun, Y., et al.: A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review 62(1), 1–62 (2022)

17. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: Proceedings of the IEEE international conference on computer vision. pp. 2980–2988 (2017)

18. Ling, H., Gao, J., Kar, A., Chen, W., Fidler, S.: Fast interactive object annotation with curve-gcn. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5257–5266 (2019)

19. Liu, C., Jiang, Y., Zheng, N.: Netracer: A topology-aware iterative tracing approach for tubular structure extraction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20593–20602 (2025)

20. Liu, S., Zhang, D., Liu, S., Feng, D., Peng, H., Cai, W.: Rivulet: 3d neuron morphology tracing with iterative back-tracking. Neuroinformatics 14(4), 387–401 (2016)

21. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10012–10022 (2021)

22. Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B.: Segment anything in medical images. Nature communications 15(1), 654 (2024)

23. Ma, J., Li, F., Wang, B.: U-mamba: Enhancing long-range dependency for biomedical image segmentation. arXiv preprint arXiv:2401.04722 (2024)

24. Marcos, D., Tuia, D., Kellenberger, B., Zhang, L., Bai, M., Liao, R., Urtasun, R.: Learning deep structured active contours end-to-end. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 8877–8885 (2018)

25. Masoudi, S., Razi, A., Wright, C.H., Gatlin, J.C., Bagci, U.: Instance-level microtubule tracking. IEEE transactions on medical imaging 39(6), 2061–2075 (2020)

26. Nogales, E., Mahamid, J.: Bridging structural and cell biology with cryo-electron microscopy. Nature 628(8006), 47–56 (2024)

27. Pautrat, R., Barath, D., Larsson, V., Oswald, M.R., Pollefeys, M.: Deeplsd: Line segment detection and refinement with deep image gradients. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17327– 17336 (2023)

28. Peng, S., Jiang, W., Pi, H., Li, X., Bao, H., Zhou, X.: Deep snake for real-time instance segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8533–8542 (2020)

29. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)

30. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

31. Scheres, S.H., Ryskeldi-Falcon, B., Goedert, M.: Molecular pathology of neurodegenerative diseases by cryo-em of amyloids. Nature 621(7980), 701–710 (2023)

32. Sheridan, A., Nguyen, T.M., Deb, D., Lee, W.C.A., Saalfeld, S., Turaga, S.C., Manor, U., Funke, J.: Local shape descriptors for neuron segmentation. Nature methods 20(2), 295–303 (2023)

33. Shin, S.Y., Summers, R.M.: Deep reinforcement learning for small bowel path tracking using diferent types of annotations. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 549–559. Springer (2022)

34. Shit, S., Paetzold, J.C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J.P., Bauer, U., Menze, B.H.: cldice-a novel topology-preserving loss function for tubular structure segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16560–16569 (2021)

35. Tetteh, G., Efremov, V., Forkert, N.D., Schneider, M., Kirschke, J., Weber, B., Zimmer, C., Piraud, M., Menze, B.H.: Deepvesselnet: Vessel segmentation, centerline prediction, and bifurcation detection in 3-d angiographic volumes. Frontiers in Neuroscience 14, 592352 (2020)

36. Vargas, J., Modrego, A., Canabal, H., Martin-Benito, J.: Semantic segmentationbased detection algorithm for challenging cryo-electron microscopy rnp samples. Frontiers in molecular biosciences 11, 1473609 (2024)

37. Wagner, T., Lusnig, L., Pospich, S., Stabrin, M., Schönfeld, F., Raunser, S.: Two particle-picking procedures for filamentous proteins: Sphire-cryolo filament mode and sphire-striper. Biological Crystallography 76(7), 613–620 (2020)

38. Wagner, T., Merino, F., Stabrin, M., Moriya, T., Antoni, C., Apelbaum, A., Hagel, P., Sitsel, O., Raisch, T., Prumbaum, D., et al.: Sphire-cryolo is a fast and accurate fully automated particle picker for cryo-em. Communications biology 2(1), 218 (2019)

39. Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., et al.: Yolov10: Real-time end-to-end object detection. Advances in neural information processing systems 37, 107984–108011 (2024)

40. Wu, J.G., Yan, Y., Zhang, D.X., Liu, B.W., Zheng, Q.B., Xie, X.L., Liu, S.Q., Ge, S.X., Hou, Z.G., Xia, N.S.: Machine learning for structure determination in single-particle cryo-electron microscopy: A systematic review. IEEE Transactions on Neural Networks and Learning Systems 33(2), 452–472 (2021)

41. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: Segformer: Simple and eficient design for semantic segmentation with transformers. Advances in neural information processing systems 34, 12077–12090 (2021)

42. Yang, Y., Arseni, D., Zhang, W., Huang, M., Lövestam, S., Schweighauser, M., Kotecha, A., Murzin, A.G., Peak-Chew, S.Y., Macdonald, J., et al.: Cryo-em structures of amyloid-β 42 filaments from human brains. Science 375(6577), 167–172 (2022)

43. Yu, F., Wang, D., Shelhamer, E., Darrell, T.: Deep layer aggregation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2403–2412 (2018)

44. Zhang, W., Tarutani, A., Newell, K.L., Murzin, A.G., Matsubara, T., Falcon, B., Vidal, R., Garringer, H.J., Shi, Y., Ikeuchi, T., et al.: Novel tau filament fold in corticobasal degeneration. Nature 580(7802), 283–287 (2020)

45. Zhang, X., Zhao, T., Chen, J., Shen, Y., Li, X.: Epicker is an exemplar-based continual learning approach for knowledge accumulation in cryoem particle picking. Nature Communications 13(1), 2468 (2022)

46. Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J.: Detrs beat yolos on real-time object detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16965–16974 (2024)

47. Zhou, X., Wang, D., Krähenbühl, P.: Objects as points. arXiv preprint arXiv:1904.07850 (2019)