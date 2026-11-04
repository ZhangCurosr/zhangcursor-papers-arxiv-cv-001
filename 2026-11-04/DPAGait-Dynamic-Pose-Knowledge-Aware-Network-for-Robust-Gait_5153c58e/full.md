# DPAGait: Dynamic Pose-Knowledge-Aware Network for Robust Gait Recognition in the Wild

Zenan Lin<sup>a</sup>, Junduan Huang<sup>b</sup>, Kun Liu<sup>a,∗</sup>, M. Saad Shakeel<sup>a</sup>, Wenxiong Kang<sup>a,c,d,∗</sup>

<sup>a</sup>School of Automation Science and Engineering, South China University of Technology, Guangzhou 510641, China

<sup>b</sup>School ofArtificial Intelligence, South China Normal University, Foshan 528225, China

<sup>c</sup>Pazhou Laboratory, Guangzhou 510335, China

<sup>d</sup>School of Future Technology, South China University of Technology, Guangzhou 510641, China

## Abstract

Gait biometrics aims at identity recognition by appearance or pose features captured during unconstrained human walking. Existing methods predominantly rely on binary silhouette images, inherently characterized by low information entropy, and fail to fully utilize the dynamic information in gait sequences, limiting their ability to handle cross-time variations efectively. Consequently, these methods encounter significant challenges in the wild. They are susceptible to variations in clothing, carried objects, and viewpoints, as well as dynamic environmental conditions, all of which lead to substantial degradation in recognition accuracy and robustness. To overcome these challenges, we propose a pose-knowledge-aware approach specifically designed for robust gait recognition in the wild. First, we propose a novel Dynamic Pose-Parsing enhanced Silhouette (DPPS) gait representation paradigm that substantially enhances the information entropy and effectively captures fine-grained behavioral cues by integrating pose-parsing cues with traditional silhouettes. Subsequently, we develop a dual-branch Dynamic Pose-Knowledge-Aware Gait network (DPAGait). The framework consists of two complementary branches: (1) a 2D dynamic pose-aware branch that extracts pose-enhanced features from our proposed DPPS representations, and (2) a 3D Skinned Multi-Person Linear (SMPL) branch that leverages the 3D SMPL model to capture rich and comprehensive 3D dynamic pose information. Finally, the features are integrated to facilitate accurate and robust gait recognition. Extensive experiments on four challenging in-the-wild gait datasets, including SUSTech1K, CCPG, Gait3D, and GREW, demonstrate the effectiveness and generalizability of the proposed DPPS representation. The complete framework

is further evaluated on SUSTech1K, CCPG, and Gait3D, where SMPL information is available or can be consistently estimated, achieving competitive performance under diverse real-world conditions. The source code is available at https://github.com/Lin-ZN/DPAGait.

Keywords: Biometrics; Behavioral Identification; Gait Recognition; Gait Representations;   
Human Motion Analysis.

## 1. Introduction

Gait recognition is a biometric technique that identifies individuals based on their walking patterns. Owing to its unique advantages in non-cooperative, long-distance, and low-resolution scenarios, gait recognition has attracted increasing attention in recent years [1–4]. Unlike other biometric traits such as face [5], finger vein [6, 7], palmprint [8], and hand gesture [9], gait recognition does not require subject cooperation and is inherently dificult to disguise or imitate. These characteristics make it particularly well-suited for applications in public security, surveillance, forensic investigation, and medical rehabilitation [3, 10]. The release of large-scale gait datasets such as GREW [3], Gait3D [11], and SUSTech1K [12], along with the development of opensource toolkits like OpenGait [13] and its associated benchmark model, has greatly advanced gait recognition research through systematic evaluations and standardized comparisons.

Despite substantial progress achieved under controlled indoor conditions, gait recognition in unconstrained real-world environments (i.e., in-the-wild) remains highly challenging [11, 12, 14, 15]. Numerous studies have shown that many SOTA approaches, such as GaitSet [2] and GaitGL [16], which perform well on indoor datasets like CASIA-B [17] and OU-MVLP [18], sufer significant performance degradation when deployed on outdoor datasets such as Gait3D [11] and SUSTech1K [12]. This performance gap largely stems from the complexity of real-world conditions, where factors like clothing variation, inconsistent lighting, dynamic occlusion, and viewpoint changes impair the stability of appearance features and hinder the extraction of robust discriminative representations.

Table 1: Comparison of diferent gait representations.
<table><tr><td>Aspect</td><td>Sils</td><td>Pose</td><td>SMPL</td><td>Our DPPS</td></tr><tr><td>Representation</td><td>人</td><td>人</td><td>火</td><td>入</td></tr><tr><td>Modality</td><td>Image</td><td>Point/Graph</td><td>3D Mesh</td><td>Image</td></tr><tr><td>Body Shape</td><td>V</td><td>X</td><td>√</td><td>√</td></tr><tr><td rowspan="2">Structural</td><td>Silhouette</td><td>Sparse</td><td>Dense Mesh</td><td>Fine-grained</td></tr><tr><td>Contour</td><td>Joints</td><td>Vertices</td><td>Contour &amp; Parts</td></tr><tr><td>Discriminability</td><td>Medium</td><td>Low</td><td>High</td><td>High</td></tr><tr><td>Size</td><td>64×64</td><td>17× 2</td><td>85D/ 6890× 3</td><td>64×64</td></tr></table>

Most gait recognition methods rely on binary silhouettes (abbreviated as Sils) [2, 19], which, despite capturing body contours, sufer from low information entropy and structural sparsity [15, 20], as shown in Table1. Most local patches in binary silhouettes exhibit uniform pixel value distributions, either all 0s or all 255s. These regions encode neither body shape nor structural semantics, leading to embedding vectors that are efectively zero or constant (Unit Vectors), as shown in Fig. 1. In DeepGaitV2 [15], the authors termed these as dumb patch, where the input provides little or no information to the model. Pose skeletons ofer strong priors and better robustness to viewpoints, but lack dense temporal cues. Skinned Multi-Person Linear (SMPL) models provide richer 3D information but are high-dimensional and not directly suitable for recognition [21]. ParsingGait [20] addressed this by proposing high-entropy semantic representations. These trends highlight the need for more informative, compact, and identity-discriminative modalities under real-world conditions.

Beyond appearance, gait encodes behavioral biometric traits such as stride frequency and limb motion amplitude [22–24], which are often invariant to clothing and viewpoint changes. These cues carry strong identity signals. However, prior works [23, 25] highlight that appearance and pose features are often entangled, necessitating a more interpretable fusion strategy. Also, SMPLbased methods [11, 22] have demonstrated the potential of modeling 3D dynamics, but often sufer from coarse feature fusion and lack explicit structural guidance. To this end, better utilization of compact SMPL parameters for dynamic modeling remains an open challenge [26].

![](images/bdb7e0ed339638ee99a2f0fac4ca022abb6a2c22b0360a98fbac158b23762a42.jpg)  
Figure 1: Visualization of DPPS patch semantics. The highlighted region in the silhouette exhibits homogeneous pixel values (e.g., all 255), producing a nearly constant patch embedding (unit vector), thus carrying minimal discriminative information. In contrast, our DPPS preserves structural variations within the same region, yielding more informative dense vectors. This visualization highlights the fine-grained spatial structure encoded in our DPPS gait representation. Conv3×3 denotes a 3 × 3 convolution kernel.

To address the aforementioned challenges, we introduce a high-entropy representation termed Dynamic Pose-Parsing Enhanced Silhouette (DPPS), which overlays pose structures onto silhouettes to embed semantic topology and structural cues. Fig. 1 demonstrates how the same patch region, when enhanced via DPPS, retains clear intra-patch variations. These variations correspond to meaningful joint and limb patterns, thereby increasing the local entropy of pixel values. The resulting dense embeddings are no longer uniform, but capture fine-grained motion-aware cues and convey stronger discriminative power. This semantic compensation mechanism allows DPPS to mitigate the dumb patch issue by embedding structure-aware cues directly into the pixel domain. As a result, the general gait recognition networks can benefit from the augmented representational richness, particularly in occluded or ambiguous walking conditions.

Furthermore, to better capture 3D dynamic behaviors from temporal gait sequences, we propose a dual-branch Dynamic Pose-Knowledge-Aware Gait network (DPAGait). The two branches of DPAGait capture complementary semantics and are fused to form a unified, discriminative gait representation. Specifically, the 2D pose-aware branch focuses on structure-enhanced appearance cues derived from DPPS, modeling fine-grained spatial patterns along body contours and articulated parts. In parallel, the 3D SMPL branch encodes pose dynamics and identity-related behavioral characteristics in a canonical 3D space, ofering robustness to clothing, viewpoint, and occlusion variations. By integrating these complementary sources through our residual attention fusion module, DPAGait produces a holistic representation that excels under diverse in-the-wild conditions.

The main contributions of our research are summarized as follows:

• We propose a novel DPPS representation, which embeds pose-derived structural cues and RankSVM-based local motion dynamics into the silhouette foreground while preserving its original contour and single-channel form. Diferent from pose heatmaps, parsing maps, and direct multimodal fusion, DPPS performs spatially aligned structure-motion integration at the representation level and remains directly compatible with existing silhouette-based gait recognition backbones.

• We design a dynamic pose-knowledge-aware dual-stream network that efectively captures both structural and behavioral cues, leveraging the complementary nature of contour and 3D dynamics to enhance structural and motion modeling.

• Extensive experiments on four large-scale in-the-wild gait datasets demonstrate the efectiveness and generalizability of DPPS, while evaluations of the complete DPAGait framework on SUSTech1K, CCPG and Gait3D further validate the benefits of complementary 2D/3D gait modeling.

## 2. Related Work

## 2.1. Gait Representations

Gait representation serves as a fundamental component of gait recognition systems that directly impacts the model’s capacity for discriminative feature learning and robustness under real-world conditions. Existing approaches typically rely on silhouettes [2, 19, 27], skeletons [14], flow [4], dynamic templates such as Gait Energy Images (GEI) [28] and parsing-based representations [20].

Among these, the binary silhouettes and skeleton sequences remain the most widely adopted representations. Silhouettes ofer a compact abstraction of human shape but inherently sufer from low information entropy, often discarding critical pose details during binarization. Moreover, their performance degrades in challenging scenarios due to foreground jitter, background noise, and occlusion. Temporal templates like GEI and CGI also fall short in modeling subtle temporal dynamics due to their averaging nature. Skeleton-based representations provide a robust encoding of human pose structures and exhibit resilience to appearance variations. However, they lack expressiveness for identity-related shape cues, such as body proportion and contour fidelity. Recent advances attempt to address these issues: SkeletonGait [14] introduces a Gaussian-based skeletal heatmap to enhance structural detail, while ParsingGait [20] proposes the Gait Parsing Sequence (GPS), extracting fine-grained part-level parsing maps for more semantically meaningful features. DPGait [29] extends sparse skeleton representations with human surface points derived from body-mesh priors, and encodes skeleton and surface points as multi-channel heatmaps for dense pose-based gait modeling. Other eforts have explored multi-modal or sensor-based representations such as LiDAR point clouds to enhance 3D structural perception [12]. However, these approaches often require heavy pre-processing or computationally expensive segmentation networks, limiting their practical deployment in large-scale systems. To this end, we propose a new silhouette representation, DPPS, which fuses skeletal topology with contour appearance in a single image. The DPPS not only retains structural semantics from the pose skeleton but also preserves the appearance fidelity of silhouette contours. This unified design provides a compact, robust, and semantically rich representation tailored for real-world gait recognition.

## 2.2. Gait Recognition Methods

Gait recognition methods are generally categorized into two major paradigms: model-based and appearance-based approaches.

Model-based gait recognition methods recognize individuals by explicitly modeling human structural or geometric characteristics. Common representations include 2D/3D skeleton sequences, mesh-based human models, and 3D point clouds [30–32]. Early works such as Urtasun et al. [30] proposed 3D temporal motion models for gait analysis, while Yamauchi et al.

[33] employed human body fitting to 3D volumetric data for person recognition. Recent advances have further improved the representational power of structural modeling. PoseGait [31] integrates 3D skeletons with human prior knowledge to achieve robustness across clothing variations and view changes. GaitGraph [32] employs Graph Convolutional Networks to extract temporal gait features from skeletal graphs, highlighting the structural advantages of graph modeling. Despite these advances, model-based methods often underperform their appearance-based counterparts on in-the-wild datasets. This performance gap primarily arises from their limited ability to capture appearance-driven gait cues, such as body shape and silhouette dynamics, which are critical under complex real-world conditions.

Appearance-based gait recognition methods aim to extract identity-relevant features from raw video sequences, focusing primarily on external contours and body shape dynamics [2, 15, 19, 24, 34–37]. These approaches are particularly efective in low-resolution or uncontrolled environments. The key challenge lies in modeling both spatial and temporal aspects of gait to learn discriminative and stable identity features. Previous works such as GaitSet [2] introduced a setbased representation of gait sequences with frame-level max-pooling, while GaitPart [19] further emphasized local body regions using a micromotion capture module to enhance temporal granularity. Fan et al. [15] proposed DeepGaitV2, which unified multiple backbone strategies to improve robustness in complex scenarios. DyGait [34] explicitly models dynamic part-level features over time, and GaitGCI [35] leverages counterfactual learning to suppress confounding regions and highlight informative patterns. RSANet [36] addresses silhouette quality variation via a relativesequence quality assessment framework. In contrast to these methods, our approach jointly models dynamic pose knowledge-enhanced appearance features and pose-aware behavioral features derived from the SMPL model. The proposed dual-stream architecture enables fine-grained and structurally guided representation learning, ofering improved robustness and discriminability under in-the-wild conditions.

## 3. Methodology

## 3.1. Dynamic Pose-Parsing enhanced Silhouette Generation

To construct a semantically structured gait representation, we design a novel semantic image that fuses silhouette contours with pose estimation results, referred to as the Dynamic Pose-Parsing enhanced Silhouette (DPPS). The generation process is shown in Fig. 2. The binary silhouette image is denoted as $S i l s \in \{ 0 , 2 5 5 \} ^ { H \times W }$ , where H and W are the height and width of the image, respectively. The corresponding set of 2D human keypoints is represented as $\mathbf { K } ~ = ~ \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { J }$ where $J = 1 7$ is the number of joints and $( x _ { i } , y _ { i } )$ indicates the pixel coordinates of the i-th keypoint. The skeletal connectivity is defined by an edge set $\mathbf { E } = \{ ( i , j ) \} \subseteq \{ 1 , . . . , J \} \times \{ 1 , . . . , J \}$ , which follows the standard topology used in the COCO dataset [38, 39]. The process of generating the Pose-Parsing (PP) image is formulated as an image rendering function $G : ( \mathbf { K } , \mathbf { E } ) \mapsto \mathbb { R } ^ { H \times W \times 3 }$ consisting of two main components:

• Keypoint Rendering: For each joint $( x _ { i } , y _ { i } ) \in { \textbf { K } } \in { \mathbb { R } } ^ { J \times 2 }$ , we draw a circle with a fixed radius $r = 8$ on a blank canvas of size $H \times W$ , enhancing the visibility of critical joints and improving robustness to minor pose estimation errors (typically within 2-4 pixels.)

• Limb Connecting: For each skeletal edge $( i , j ) \in \mathbf { E }$ , a colored line of fixed thickness $l = 5$ connects $( x _ { i } , y _ { i } )$ and $( x _ { j } , y _ { j } )$ , providing explicit topological structure and supporting partaware modeling.

Through this process, we obtain an RGB pose-parsing image that visually encodes both keypoint positions K and skeletal connections E in a semantically interpretable manner.

To inject dynamic pose context, we employ the RankSVM method [40, 41] on the generated pose-parsing sequence $\{ I _ { 1 } ^ { P P } , \ldots , I _ { T } ^ { P P } \}$ to generate the DPPS. The RankSVM optimizes a ranking objective function, as shown in Eq. (1), to learn a feature vector d<sup>∗</sup> that represents the temporal dynamics of the video sequence (specific formula derivation and more detailed information can be

found in the supplementary materials.)

$$
\begin{array} { l } { \displaystyle \mathbf { d } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { d } } E ( \mathbf { d } ) } \\ { \displaystyle E ( \mathbf { d } ) = \frac { \lambda } { 2 } \| \mathbf { d } \| ^ { 2 } + } \\ { \displaystyle \frac { 2 } { T ( T - 1 ) } \sum _ { q > t } \operatorname* { m a x } \{ 0 , 1 - S ( q \mid \mathbf { d } ) + S ( t \mid \mathbf { d } ) \} , } \end{array}\tag{1}
$$

where $S ( t \mid \mathbf { d } )$ is the ranking score that measures the temporal position of frame t.

We divide the input video sequence into overlapping segments of length $n = 3$ with a stride of 1, enabling the model to capture the temporal evolution of short-term motion patterns within each local window. Let $\psi ( I _ { t } )$ denote the pixel-level feature vector of the t-th frame, and $\begin{array} { r } { V _ { t } = \frac { 1 } { t } \sum _ { \tau = 1 } ^ { t } \psi ( I _ { \tau } ) } \end{array}$ be the temporal average. The RankSVM learns a dynamic descriptor $\mathbf { d } ^ { * }$ for each window as shown in Eq. (2).

$$
{ \bf d } ^ { * } \propto \sum _ { q > t } ( V _ { q } - V _ { t } ) = \sum _ { t = 1 } ^ { T } \gamma _ { t } \psi ( I _ { t } ) ,\tag{2}
$$

where $\begin{array} { r } { \gamma _ { t } ~ = ~ \sum _ { i = t } ^ { T } \frac { 1 } { i } ( 2 i - T ) } \end{array}$ Each dynamic descriptor $\mathbf { d } ^ { * } \in \mathbb { R } ^ { H \times W \times 3 }$ is reshaped and normalized to the same size as input frames, forming a temporally fused image summarizing motion trends across consecutive Pose-Parsing frames, termed as Dynamic Pose-Parsing (DPP) representation. It ofers several advantages: (a) it compresses both short-term motion and pose-aware structure into a single image, enabling lightweight inference with 2D CNNs; (b) it preserves fine-grained joint movements by emphasizing inter-frame diferences.

While skeleton representations efectively encode the internal structural configuration of the human body and are robust to clothing and appearance variations, they inherently lack explicit information about individual body shapes. This omission may result in suboptimal recognition performance, particularly in scenarios where shape cues are important for identity discrimination. In contrast, silhouette images preserve the external body contours but omit finer structural semantics, making them complementary to skeletons. To fully leverage their complementary nature, we propose to fuse the generated DPP image with the original binary silhouette (S ils). To ensure that the injected structural information remains confined within valid silhouette regions, we introduce a conditional mask defined as Eq. (3):

$$
\mathcal { M } ( x , y ) = \mathbb { 1 } _ { S i l s ( x , y ) = 2 5 5 } \cdot \mathbb { 1 } _ { D P P ( x , y ) \neq 2 5 5 } ,\tag{3}
$$

![](images/44ad1053c0528b92e72139723d547885bf17d20f5b06765a0dbf9da8372e0156.jpg)  
Figure 2: Overview of the proposed DPAGait network. The network consists of two parallel branches: the 2D dynamic pose-aware branch takes DPPS as input to extract 2D pose-aware features, while the 3D SMPL branch leverages SMPL parameters to model 3D pose-aware behavioral representations. The two branches are fused using the Residual Attention Fusion (RAFusion) module for final gait recognition through joint optimization with reconstruction loss, triplet loss, and cross-entropy loss.

![](images/fe463ceeb5f6b2971740f2f3299cff36c198337671937ccc8ad7abbabbda2699.jpg)  
Figure 3: The proposed Residual Attention Fusion (RAFusion) module.

where 1<sub>·</sub> denotes the indicator function, which turns into 1 if the corresponding pixel in the silhouette image is white (i.e., part of the foreground), and the same pixel in DPP image is not entirely white (i.e., contains meaningful skeletal content, noting that the background of DPP image is uniformly white with pixel value 255). Using mask M, we generate the final Dynamic Pose-Parsing enhanced Silhouette (DPPS) by Eq. (4):

$$
D P P S \left( x , y \right) = \left\{ \begin{array} { l l } { { D P P ( x , y ) , } } & { { { \mathrm { i f } } \ M ( x , y ) = 1 } } \\ { { } } & { { } } \\ { { { } } } & { { { } } } \\ { { { \cal S } i l s ( x , y ) , } } & { { { \mathrm { o t h e r w i s e } } . } } \end{array} \right.\tag{4}
$$

This strategy selectively overlays skeletal information onto the silhouette only within valid human regions, preserving the original body contour while injecting rich pose semantic structure. By introducing richer local structural variations within the silhouette foreground, DPPS provides more informative local patterns for subsequent feature extraction. The DPPS representation embeds structural semantics without compromising contour integrity, facilitating the learning of discriminative and stable gait features in the wild.

## 3.2. Dual-branch Dynamic Pose-Knowledge-Aware Network

To further enhance the modeling of diverse gait-related modalities, we propose a dual-branch Dynamic Pose-Knowledge-Aware Gait network (DPAGait), consisting of a 2D dynamic poseaware branch and a 3D SMPL branch, as shown in Fig. 2.

## 3.2.1. 2D Dynamic Pose-Aware Branch

The 2D dynamic pose-aware branch is designed to extract pose knowledge-enhanced features from the DPPS representations. Unlike conventional silhouette images, the DPPS preserves the complete outer contour of the human body while explicitly encoding key skeletal structures, enabling the model to perceive shape and geometric pose information simultaneously. This branch focuses on learning fine-grained features enhanced with dynamic pose priors.

As illustrated in Fig. 2, we adopt DeepGaitV2-P3D [15] as the backbone for this branch. DeepGaitV2-P3D consists of five components, including an initial convolutional layer (C0) followed by four stages (S1 - S4). The C0 block applies a 3 × 3 2D convolution to extract low-level features. Stage 1 employs standard 2D residual blocks for shallow spatial encoding, while stage 2 to stage 4 leverage Pseudo-3D (P3D) residual units. Each P3D block decouples 3D convolution into a combination of $1 \times 3 \times 3$ spatial and $3 \times 1 \times 1$ temporal convolutions, balancing expressive power and computational eficiency. This design is particularly advantageous for modeling largescale in-the-wild gait datasets. We further apply Temporal Pooling (TP) and Horizontal Pyramid Pooling (HPP) [15] to capture multiscale local spatial cues by Eq. (5):

$$
\mathbf { F } _ { T P } = T P ( \mathbf { F } _ { S 4 } ) , \quad \mathbf { F } _ { A } = H P P ( \mathbf { F } _ { T P } ) .\tag{5}
$$

The resulting representation $\mathbf { F } _ { A }$ embeds both the pose-guided structural semantics and temporal dynamics, providing a discriminative and stable feature set for downstream fusion with the 3D SMPL branch.

## 3.2.2. 3D SMPL Branch

The 3D SMPL branch is designed to extract identity-related behavioral representations from 3D human body models, focusing on modeling individual motion patterns in the spatiotemporal domain. The role of this branch is to leverage SMPL-based 3D pose information to provide complementary behavioral cues that are dificult to explicitly characterize using the 2D DPPS representation.

For the Gait3D dataset, we directly use the provided SMPL parameters, while for SUSTech1K and CCPG, we estimate the SMPL parameters from RGB videos via ROMP [42]. The pose parameters θ encode dynamic 3D pose motion, and the SMPL’s strong statistical priors enable reliable full-body pose recovery, even under partial occlusion. This parameterization ofers several advantages for in-the-wild gait recognition, such as occlusion robustness, viewpoint and scale normalization, and behavioral stability. Consequently, even when the same subject appears under diferent outdoor conditions and clothing, SMPL-driven reconstruction preserves the intrinsic pose dynamics critical for identity discrimination.

To model temporal behaviors, we adopt a GRAB-inspired [43] motion encoder-decoder framework. The 3D SMPL branch maps $\pmb \theta$ into a compact latent vector using a three-layer residual encoder adapted from GrabNet [43], followed by a decoder that reconstructs pose motion representations. This design captures compact, identity-relevant motion cues for robust gait recognition.

As shown in Fig. 2, given the input pose sequence $\theta ,$ the $\delta ^ { c }$ encode it into a latent embedding, as shown in Eq. (6):

$$
\begin{array} { r } { z = \mathscr { E } ^ { c } ( \pmb { \theta } ) \in \mathbb { R } ^ { d } , \quad d = 1 6 } \end{array}\tag{6}
$$

Here, $z$ denotes the learned pose-aware behavioral representation, capturing high-level identityrelevant information from dynamic motion. The decoder not only enforces reconstruction supervision via loss $\mathcal { L } _ { r e c o n }$ , but also guides the latent distribution to be more semantically structured and identity-aware. Also, we design a lightweight behavior guide module, consisting of a fully connected (FC) layer, batch normalization (BN), and Temporal Pooling (TP) layer, to get behavior representation $\mathbf { F } _ { B } ,$ , as shown in Eq. (7):

$$
\mathbf { F } _ { B } = T P ( B N ( F C ( z ) ) )\tag{7}
$$

To efectively combine complementary features from the two branches, we design a Residual Attention Fusion (RAFusion) module, as shown in Fig. 3. Unlike conventional fusion methods such as simple concatenation or vanilla attention, our design introduces a learnable gate that adaptively emphasizes informative features from each modality while preserving the characteristic representation through a residual pathway. Given the two input features $\mathbf { F } _ { A } , \mathbf { F } _ { B } \in \mathbb { R } ^ { C \times D }$ , where $C = 2 5 6$ is the number of channels and $D = 1 6$ the length of the feature, we first concatenate them to form Eq. (8):

$$
\mathbf { F } _ { c a t } = C o n c a t e n a t i o n ( \mathbf { F } _ { A } , \mathbf { F } _ { B } ) \in \mathbb { R } ^ { 2 C \times D }\tag{8}
$$

We then apply a gating network $\mathcal { G } ( \cdot )$ consisting of two 1D convolutional layers with intermediate channel reduction, batch normalization, ReLU activation, and a final sigmoid to get Eq. (9):

$$
\mathbf { G } = \mathcal { G } ( \mathbf { F } _ { c a t } ) \in [ 0 , 1 ] ^ { C \times D }\tag{9}
$$

The final fused feature is computed via Eq. (10):

$$
\mathbf { F } _ { f u s e d } = \mathbf { F } _ { A } + \mathbf { G } \odot ( \mathbf { F } _ { B } - \mathbf { F } _ { A } ) \in \mathbb { R } ^ { C \times D }\tag{10}
$$

$\mathbf { F } _ { f u s e d }$ captures complementary cues from both streams and is further passed through a BNNeck [44] for final identity recognition and inference. This formulation encourages residual-style modulation of the feature $\mathbf { F } _ { A }$ by selectively injecting useful diferences from $\mathbf { F } _ { B } ,$ , while suppressing

noisy or redundant patterns. The gating network $\mathcal { G } ( \cdot )$ serves as a soft attention map that adaptively reweights the feature diference. This design preserves alignment and allows the fusion process to be easily integrated into the backbone with minimal computation overhead.

## 3.2.3. Loss Function

During training, we introduce a reconstruction loss to enforce semantic consistency in the latent space, as shown in Eq. (11):

$$
\mathcal { L } _ { r e c o n } = \| \hat { \theta } - \theta \| _ { 1 }\tag{11}
$$

where $\theta$ is the ground-truth pose, and $\hat { \boldsymbol { \theta } }$ is the decoder’s output. At inference time, the decoder is discarded, and only the fused latent vector is used for downstream recognition. Compared to using raw pose parameters, the latent representation z encodes more robust and compact identity features, mitigating the efects of noisy or jittery pose inputs.

We adopt a combined loss function comprising cross-entropy loss $\mathcal { L } _ { c e }$ for identity supervision and triplet loss $\mathcal { L } _ { t r i p l a t e }$ to enforce intra-class compactness and inter-class separability in the embedding space. The final training objective is in Eq. (12):

$$
\begin{array} { r } { \mathcal { L } _ { t o t a l } = \mathcal { L } _ { r e c o n } + \mathcal { L } _ { c e } + \mathcal { L } _ { t r i p l a t e } } \end{array}\tag{12}
$$

This multi-objective optimization ensures discriminative and robust gait representations suitable for real-world recognition scenarios.

## 4. Experiments

## 4.1. Datasets

We evaluate our method on four challenging in-the-wild gait recognition benchmarks: SUSTech1K [12], CCPG [45], Gait3D [11], and GREW [3].

SUSTech1K. An outdoor multimodal dataset with 1,050 subjects and 25,239 sequences captured by synchronized RGB and LiDAR devices.

CCPG. A benchmark proposed for cloth-changing person ReID and gait recognition. It includes 200 identities and 16K sequences with fine-grained clothing variation labels.

Gait3D. A large-scale gait dataset collected from 39 surveillance cameras in real-world scenes. It contains 4,000 identities and over 25,000 sequences with frame-wise silhouettes, 2D/3D pose annotations, and SMPL fits.

GREW. A real-world dataset constructed from unconstrained videos. It contains 26K identities and 128K sequences, with rich attributes and camera diversity.

## 4.2. Evaluation Metrics

These datasets reflect the practical complexities of real-world gait recognition and serve as rigorous benchmarks for evaluating model generalization and robustness. We adopt commonly used evaluation metrics consistent with prior works, such as Rank-1 accuracy (R-1), Rank-5 accuracy (R-5), Rank-10 accuracy (R-10), mean Average Precision (mAP), and mean Inverse Negative Penalty (mINP).

## 4.3. Implementation Details

All experiments are conducted on a deep learning server equipped with four NVIDIA RTX 3090 GPUs (each with 24 GB of memory). Our method is implemented using the PyTorch framework and is modularly developed based on the OpenGait toolbox [13] for eficient extension and integration. During training, we adopt Stochastic Gradient Descent (SGD) as the optimizer, with an initial learning rate of 0 1, a momentum factor of 0 9, and a weight decay of $5 \times 1 0 ^ { - 4 }$ . We apply a range of data augmentation techniques to the images during training, including perspective transformation, random horizontal flipping, rotation, cropping, and random erasing [13]. All input images are resized or cropped to a fixed resolution of $6 4 \times 6 4$ to ensure consistency across the dataset.

## 4.4. Comparisons with State-of-the-Art Methods

As shown in Table 2 and Table 3, our proposed DPPS demonstrates clear advantages in both recognition accuracy and computational eficiency compared to other single-modality methods. Specifically, on the SUSTech1K dataset, DPPS achieves the Rank-1 accuracy of 81.3%, surpassing DeepGaitV2-Parsing (77.3%) and DeepGaitV2-Flow (80.5%), while requiring less than half of their computational cost 9.5 GFLOPs vs. 21.9 GFLOPs and 21.0 GFLOPs, respectively, as reported in Table 4. On the GREW and Gait3D datasets, DPPS also achieved the highest Rank-1 accuracy, at 79.2% and 81.1% respectively. On the CCPG dataset, DPPS attains a Rank-1 accuracy of 85.9%, which is also competitive with representative RGB-based methods. It is worth noting that RGB inputs inherently contain rich appearance cues, but they can also raise privacy concerns in real-world identity recognition. In contrast, DPPS leverages structural and motion semantics from non-RGB modalities, maintaining competitive accuracy while preserving the privacy advantages of silhouette-based representations.

Table 2: Evaluation with diferent attributes on SUSTech1K and CCPG datasets.
<table><tr><td rowspan="2">Modality</td><td rowspan="2">Method</td><td colspan="2">SUSTech1K</td><td colspan="5">CCPG</td></tr><tr><td>Probe Sequence (R-1) NM BG CL CR UB UN OC NT</td><td>Overall R-1 R-5</td><td>CL R-1 mAP</td><td>UP R-1 mAP</td><td>DN R-1 mAP</td><td>BG R-1 mAP</td><td>Mean R-1</td></tr><tr><td rowspan="3">Skeleton</td><td>GaitGraph2 [46]</td><td>22.2 18.2 6.818.6 13.4 19.2 27.3 16.4</td><td>18.6 40.2</td><td>5.0 2.5</td><td>5.3 4.0</td><td>5.8 4.2</td><td>6.2 4.6</td><td>5.6</td></tr><tr><td>GaitTR [47]</td><td>33.3 31.5 21.0 30.4 22.7 34.6 44.9 23.5</td><td>30.856.0</td><td>15.7 9.7</td><td>18.316.1</td><td>18.516.4</td><td>17.515.3</td><td>17.5</td></tr><tr><td>SkeletonGait [14]</td><td>55.0 51.0 24.749.9 42.3 52.0 62.843.9</td><td>50.1 72.6</td><td>40.420.8</td><td>48.535.8</td><td>53.040.3</td><td>61.748.1</td><td>50.9</td></tr><tr><td rowspan="5">Sils</td><td>GaitSet [2]</td><td>69.1 68.2 37.4 65.0 63.1 61.0 67.2 23.0</td><td>65.0 84.8</td><td>60.247.4</td><td>65.2 63.2</td><td>65.1 61.9</td><td>68.565.8</td><td>64.8</td></tr><tr><td>GaitPart [19]</td><td>62.2 62.3 31.1 59.5 57.2 54.8 57.2 21.7</td><td>59.2 80.8</td><td>64.348.0</td><td>67.863.2</td><td>68.663.8</td><td>71.766.4</td><td>68.1</td></tr><tr><td>GaitGL [16]</td><td>67.1 66.2 35.9 63.3 61.6 58.1 66.6 17.9</td><td>63.1 82.8</td><td>66.342.7</td><td>74.364.5</td><td>67.758.0</td><td>80.4 70.1</td><td>72.2</td></tr><tr><td>GaitBase [13]</td><td>81.577.5 49.6 75.875.5 76.781.4 25.9</td><td>76.1 89.4</td><td>71.658.5</td><td>75.071.6</td><td>76.873.3</td><td>78.675.3</td><td>75.5</td></tr><tr><td>DeepGaitV2 [15]</td><td>83.5 79.5 46.3 76.8 79.1 78.5 81.1 27.3</td><td>77.4 90.2</td><td>78.662.1</td><td>84.881.5</td><td>80.778.1</td><td>89.2 85.8</td><td>83.3</td></tr><tr><td rowspan="2">Parsing</td><td>GaitBaseP [13]</td><td>80.3 71.1 32.5 72.6 66.3 73.581.7 40.7</td><td>71.7 88.4</td><td>59.1 44.1</td><td>62.157.5</td><td>66.861.6</td><td>68.164.2</td><td>64.0</td></tr><tr><td>DeepGaitV2P [15]</td><td>84.777.329.4 78.275.880.2 87.943.3</td><td>77.391.3</td><td>69.651.7</td><td>75.871.4</td><td>75.871.6</td><td>83.379.3</td><td>76.1</td></tr><tr><td>Flow</td><td>DeepGaitV2f [15]</td><td>84.4 82.854.979.382.4 79.889.735.080.592.5</td><td></td><td>68.849.5</td><td>73.367.5</td><td>75.068.0</td><td>76.871.1</td><td>73.5</td></tr><tr><td>Dense Pose</td><td>DPGait [29]</td><td>90.3 85.7 64.3 85.3 83.7 88.8 93.8 63.1|85.8 95.3|88.6</td><td></td><td></td><td>94.0</td><td>93.9</td><td>94.3</td><td>92.7</td></tr><tr><td rowspan="2">RGB</td><td>GaitEdge [48]</td><td></td><td></td><td>66.9 =</td><td>74.0</td><td>70.6</td><td>77.1</td><td>72.2</td></tr><tr><td>BigGait [49]</td><td></td><td></td><td>82.6</td><td>85.9</td><td>87.1</td><td>93.1</td><td>87.2</td></tr><tr><td>DPPS(Ours)</td><td>DeepGaitV2 [15]</td><td>85.082.6 51.8 81.083.2 82.4 86.8 36.2|81.3 92.881.365.9</td><td></td><td></td><td></td><td>86.883.5|84.981.3</td><td>90.4 87.1</td><td>85.9</td></tr><tr><td rowspan="2">Sils+Skeleton</td><td>BiFusion [50]</td><td>69.8 62.3 45.4 60.9 54.3 63.5 77.8 33.7</td><td>62.183.4</td><td>62.646.7</td><td>67.664.1</td><td>66.361.9</td><td>66.061.9</td><td>65.6</td></tr><tr><td></td><td>SkeletonGait++ [14]|85.1 82.9 46.6 81.9 80.882.5 86.2 47.5</td><td>81.3 95.5</td><td>79.1 63.6</td><td>83.981.2</td><td>81.779.3</td><td>89.987.0</td><td>83.7</td></tr><tr><td rowspan="2">Sils+Parsing</td><td>XGait [51]</td><td></td><td></td><td>72.859.5</td><td>77.074.3</td><td>79.175.7</td><td>80.578.2</td><td>77.4</td></tr><tr><td>MultiGait++ [52]</td><td>90.5 86.6 43.1 85.8 87.3 86.7 92.6 44.7|85.4 94.8|80.764.6</td><td></td><td></td><td></td><td></td><td>85.782.5 84.1 80.5 90.587.0</td><td>85.3</td></tr><tr><td>DPPS+Parsing</td><td>MultiGait++ [52]</td><td>92.088.2 49.5 81.688.7 88.8 94.3 46.5|87.2 95.6|82.3 67.0 |86.4 83.8 |84.982.0|90.4 87.5</td><td></td><td></td><td></td><td></td><td></td><td>86.0</td></tr><tr><td rowspan="2">Sils+Parsing+Flow</td><td>MultiGaits+p+f [52]</td><td>90.087.844.2 86.2 88.3 87.7 91.844.8|86.0 95.181.365.2</td><td></td><td></td><td>87.286.3</td><td>82.980.3</td><td>90.687.4</td><td>85.5</td></tr><tr><td>MultiGait++ [52]</td><td>92.089.4 50.4 87.6 89.789.1 93.4 45.1</td><td>87.4 95.6|83.968.5</td><td></td><td></td><td>89.085.8 |86.082.5</td><td>91.588.9</td><td>87.6</td></tr><tr><td>Sils+DPPS+Flow</td><td>MultiGait++ [52]</td><td>91.5 88.6 60.1 87.2 90.4 88.2 92.5 39.1|87.6 95.2|85.9 71.5 |88.9 86.4 |88.1 85.1 |92.0 89.3</td><td></td><td></td><td></td><td></td><td></td><td>88.7</td></tr><tr><td>DPPS+Parsing+Flow</td><td>MultiGait++ [52]</td><td>91.989.353.088.089.789.8 94.3 47.2|87.8 95.9|85.070.2</td><td></td><td></td><td></td><td>88.485.7|86.483.3</td><td>90.988.2</td><td>87.7</td></tr><tr><td rowspan="2">Sils+SMPL</td><td>SMPLGait [11]</td><td>79.1 77.749.7 73.7 74.5 73.380.1 29.6|</td><td>74.6 89.7</td><td>62.650.8</td><td>66.764.3</td><td>67.064.5</td><td>67.464.5</td><td>65.9</td></tr><tr><td>DPAGait(Ours)</td><td>87.8 84.6 52.8 81.6 86.3 85.2 88.5 29.4</td><td>82.5 92.6</td><td>80.362.1</td><td>85.580.2</td><td>83.377.9</td><td>91.086.8</td><td>84.8</td></tr><tr><td rowspan="2">DPPS+SMPL</td><td>SMPLGait [11]</td><td>81.6 78.7 53.6 77.0 76.6 75.2 83.5 36.1</td><td>77.6 91.6</td><td>64.353.4</td><td>66.865.2</td><td>69.566.8</td><td>68.4 65.8</td><td>67.3</td></tr><tr><td>DPAGait(Ours)</td><td>89.886.857.984.789.188.2 92.136.7</td><td>85.6 94.3</td><td>82.465.2</td><td>86.682.6</td><td>85.3 80.8</td><td>91.4 87.8</td><td>86.4</td></tr></table>

Table 3: Comparative results on two outdoor in-the-wild gait datasets, GREW and Gait3D.
<table><tr><td rowspan="2">Modality</td><td rowspan="2">Method</td><td rowspan="2">GREW R-1 R-5 R-10</td><td colspan="2">Gait3D R-1 R-5 mAP mINP</td></tr><tr><td></td><td></td></tr><tr><td rowspan="9">Sils</td><td>GaitSet [2]</td><td>46.3 63.670.3</td><td>36.7 58.3 30.0</td><td>17.3</td></tr><tr><td>GaitPart [19]</td><td>44.0 60.7 67.3</td><td>28.2 47.6 21.6</td><td>12.4</td></tr><tr><td>GaitGL [16]</td><td>47.3 63.6 69.3</td><td>29.7 48.5 22.3</td><td>13.6</td></tr><tr><td>GaitBase [13]</td><td>60.1 75.5 80.4</td><td>64.6</td><td></td></tr><tr><td>QAGait [53]</td><td>59.1 74.0 79.2</td><td>67.0 81.5 56.5</td><td></td></tr><tr><td>DyGait [34]</td><td>74.1 83.2 86.8</td><td>66.3 80.8 56.4</td><td>37.3</td></tr><tr><td>MTSGait [11]</td><td>55.3 71.3 76.9</td><td>48.7 67.1 37.6</td><td>21.9</td></tr><tr><td>HSTL [24]</td><td>62.7 76.6 81.3</td><td>61.3 76.3 55.5</td><td>34.8</td></tr><tr><td>GaitGCI [35]</td><td>68.5 80.8 84.9</td><td>50.3 68.5 39.5</td><td>24.3</td></tr><tr><td rowspan="3">Skeleton</td><td>DeepGaitV2 [15]</td><td>77.7 88.9 91.8</td><td>74.488.0</td><td>65.8 39.2</td></tr><tr><td>GaitGraph2 [46]</td><td>33.5</td><td>11.1</td><td></td></tr><tr><td>GaitTR [47] SkeletonGait [14]</td><td>54.5</td><td>6.6</td><td></td></tr><tr><td>Flow</td><td>AttenGait [4]</td><td>77.4 87.9 91.0</td><td>38.1 56.7 28.9</td><td>16.1</td></tr><tr><td>DPPS(Ours)</td><td>DeepGaitV2 [15]</td><td>70.7 82.986.9</td><td></td><td>1</td></tr><tr><td rowspan="3">Sils+Skeleton</td><td></td><td>79.2 89.4 92.3</td><td>81.1 92.1 74.6</td><td>56.6</td></tr><tr><td>GaiRef [54] MSAFF [55]</td><td>53.0 67.973.0</td><td>49.0 49.3 40.7</td><td>25.3</td></tr><tr><td>SkeletonGait++ [14]</td><td>57.4 73.078.3 85.8 92.694.3</td><td>48.1 66.6 38.5 77.6 89.4 70.3</td><td>23.5 42.6</td></tr><tr><td rowspan="3">Sils+Parsing</td><td>XGait [51]</td><td></td><td></td><td></td></tr><tr><td>MultiGaits+P [52]</td><td></td><td>80.5 91.9 73.3</td><td>55.4</td></tr><tr><td>MultiGait++s+P [52]</td><td></td><td>83.0 94.5 78.6 85.4 94.9 80.5</td><td>62.7 65.2</td></tr><tr><td>DPPS+Parsing</td><td>MultiGait++ [52]</td><td></td><td>85.6 94.2 78.7</td><td>61.4</td></tr><tr><td>Sils+SMPL</td><td>SMPLGait [11]</td><td></td><td>46.3 64.5 37.2</td><td>22.2</td></tr><tr><td rowspan="2"></td><td>DPAGait(Ours)</td><td></td><td>74.5 87.3 66.2</td><td>46.6</td></tr><tr><td>SMPLGait [11]</td><td></td><td>59.3 78.8 50.6</td><td>33.1</td></tr><tr><td>DPPS+SMPL</td><td>DPAGait(Ours)</td><td></td><td>81.6 91.8 74.8</td><td>56.9</td></tr></table>

Table 4: GFLOPs comparison of diferent modalities at 256 × 256 resolution. The calculation protocol follows MultiGait++ [52]. “DI” denotes the Dynamic Image generation method [40].
<table><tr><td>Modality</td><td>Method</td><td>Backbone</td><td>GFLOPs</td></tr><tr><td>Parsing</td><td>SCHP</td><td>A-CE2P</td><td>21.9</td></tr><tr><td>Flow</td><td>VideoFlow</td><td>ResNet-like</td><td>21.0</td></tr><tr><td>DPPS</td><td>HRNet + DI</td><td>HRNet [39]</td><td>9.5</td></tr></table>

We further include the recent dense pose-based DPGait [29] for comparison. By augmenting sparse skeleton keypoints with human surface points, DPGait achieves 85.8% and 92.7% Rank-1 accuracy on SUSTech1K and CCPG, respectively, demonstrating the efectiveness of dense pose modeling. In comparison, DPPS follows a diferent representation paradigm: rather than increasing the density of pose points, it embeds pose-derived structural and dynamic cues into the silhouette foreground while retaining a compact single-channel representation that is directly compatible with existing silhouette-based gait recognition backbones. Although DPGait achieves higher in-domain accuracy on SUSTech1K and CCPG, DPPS provides a simpler silhouette-compatible representation and demonstrates improved cross-dataset transferability, as further validated in Table 5.

Furthermore, DPPS demonstrates promising generalization ability when integrated into multimodal architectures. In the powerful MultiGait++ backbone, replacing silhouette, parsing, and flow with DPPS significantly improves Rank-1 accuracy, indicating that DPPS efectively complements other modalities through semantically enriched yet lightweight representations. Compared to SMPLGait, which also utilizes SMPL parameters, DPAGait achieves an 11.0% performance improvement on SUSTech1K (85.6% vs. 74.6%), a 20.5% improvement on CCPG (86.4% vs. 65.9%), and a 35.3% improvement on Gait3D (81.6% vs. 46.3%), highlighting the advantages of dynamic pose knowledge modeling and two-stream fusion based on residual attention mechanisms. These results collectively demonstrate that DPPS provides a compact yet discriminative representation method, ofering a promising alternative when dense parsing or optical flow annotations are dificult to obtain. To further evaluate the efectiveness of the dual-branch architecture, we introduce DPPS into SMPLGait, which also adopts a dual-branch framework with 2D gait representations and 3D SMPL parameters. As shown in Table 2 and Table 3, replacing silhouette with DPPS improves the Rank-1 accuracy of SMPLGait from 74.6% to 77.6% on SUSTech1K, from 65.9% to 67.3% on CCPG, and from 46.3% to 59.3% on Gait3D. These consistent improvements on another existing dual-branch framework provide further experimental support for the efectiveness of dual-branch modeling with DPPS and SMPL.

![](images/8178103929f7a02acc1dc9ca7f5dbf069f46ed6fedeeb77593fa085b3f6c1830.jpg)  
Figure 4: Comparison of Rank-1 accuracy between DPPS and silhouette inputs across representative single-stream gait recognition models on four benchmark datasets.

## 4.5. Comparison Between DPPS and Silhouette Inputs

As shown in Fig. 4, we evaluate the efectiveness of the proposed DPPS representation by comparing it with traditional silhouette inputs across several representative single-stream baseline models, including GaitSet [2], GaitPart [19], GaitGL [16], GaitBase [13], and DeepGaitV2 [15]. Experimental results consistently demonstrate that DPPS improves gait recognition performance across all four benchmark datasets, using five baseline models, thereby validating its discriminative ability under diverse and challenging conditions.

Table 5: Cross-dataset performance comparison among SUSTech1K, Gait3D, and CCPG. All models are trained only on the source dataset and directly evaluated on the target dataset without fine-tuning or domain adaptation. Results are reported in Rank-1 accuracy (%).
<table><tr><td rowspan="3">Model</td><td colspan="3">Train on SUSTech1K</td><td colspan="4">Train on Gait3D</td><td colspan="3">Train on CCPG</td></tr><tr><td colspan="2">CCPG</td><td>Gait3D</td><td colspan="2">SUSTech1K</td><td colspan="2">CCPG</td><td colspan="2">SUSTech1K</td><td>Gait3D</td></tr><tr><td>CL</td><td>BG</td><td>Overall</td><td>OC</td><td>Overall</td><td>CL</td><td>BG</td><td>OC</td><td>Overall</td><td>Overall</td></tr><tr><td>SMPLGait [11]</td><td>14.7</td><td>41.0</td><td>12.9</td><td>35.7</td><td>33.8</td><td>11.1</td><td>59.5</td><td>10.9</td><td>11.5</td><td>8.6</td></tr><tr><td>DPGait [29]</td><td>7.9</td><td>28.7</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>18.9</td><td>14.3</td><td>一</td></tr><tr><td>DPAGait (Ours)</td><td>15.0</td><td>52.1</td><td>26.3</td><td>56.4</td><td>47.3</td><td>15.0</td><td>83.8</td><td>32.1</td><td>31.4</td><td>28.8</td></tr></table>

On SUSTech1K, DPPS yields remarkable performance gains over silhouette inputs, particularly for GaitGL, which improves by 14.1 percentage points in Rank-1 accuracy. On Gait3D, DPPS demonstrates consistent improvements across all baseline datasets, with an average performance improvement of up to 8.4 percentage points. Even on Gait3D, the robust DeepGaitV2 backbone achieves a further 6.7 percentage points increase (from 74.4% to 81.1%). Similarly, on GREW, the average improvement in DPPS is as high as 4.8 percentage points.

These consistent gains across datasets stem primarily from the structural priors and semantic completeness provided by DPPS. Unlike traditional silhouettes that only encode coarse outer contours, DPPS embeds human skeletal topology within the silhouette foreground, efectively recovering missing local body regions (e.g., occluded limbs or motion-blurred parts) and strengthening the representation of anatomically essential joints. This skeleton-guided enrichment provides a geometry-aware prior that stabilizes silhouette quality and guides alignment. Moreover, by fusing structural semantics with silhouette contours, DPPS substantially increases the input’s information entropy, enabling downstream networks to capture more motion cues with limited receptive fields.

Table 6: Ablation study of static pose-silhouette fusion and RankSVM-based dynamic encoding with diferent temporal window length n on SUSTech1K, CCPG, and Gait3D. The stride is fixed at s = 1 for all dynamic encoding settings. Results are reported in Rank-1 accuracy (%).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Temporal Window Length</td><td rowspan="2"></td><td rowspan="2">CCPG Gait3D</td></tr><tr><td>SUSTech1K</td></tr><tr><td>PPS</td><td>一</td><td>81.0</td><td>85.7 79.2</td></tr><tr><td rowspan="3">RankSVM-Based Dynamic Encoding (DPPS)</td><td>2</td><td>81.2</td><td>85.7 79.6</td></tr><tr><td>3</td><td>81.3</td><td>85.9 81.1</td></tr><tr><td>4</td><td>80.7</td><td>85.3 80.0</td></tr></table>

## 4.6. Cross-Dataset Evaluation

To further evaluate the generalization ability of DPAGait, we conduct cross-dataset experiments among SUSTech1K, Gait3D, and CCPG. Following the cross-domain protocol in DPGait [29], each model is trained only on the source dataset and evaluated directly on an unseen target dataset, without target-domain fine-tuning or adaptation. As shown in Table 5, DPAGait consistently outperforms SMPLGait and DPGait across all reported cross-dataset settings. For example, when trained on Gait3D and evaluated on SUSTech1K, DPAGait achieves an overall Rank-1 accuracy of 47.3%, compared with 33.8% for SMPLGait. When trained on CCPG and evaluated on Gait3D, DPAGait improves Rank-1 accuracy from 8.6% to 28.8%. These results provide further evidence that DPAGait can learn gait representations that transfer more efectively across unseen dataset domains.

## 4.7. Ablation Experiments

## 4.7.1. Efectiveness of Dynamic Pose-Parsing Encoding

To further investigate the contribution of the dynamic pose-parsing encoding, we evaluate DPPS across diferent temporal window lengths while keeping the stride fixed at s = 1. The stride is kept unchanged to maintain a consistent temporal sampling density and a comparable number of generated dynamic representations, because increasing the stride would reduce the number of generated samples. As shown in Table 6, the best performance is achieved with n = 3, obtaining Rank-1 accuracies of 81.3%, 85.9%, and 81.1% on SUSTech1K, CCPG, and Gait3D, respectively. Compared with static PPS, RankSVM-based dynamic encoding provides further improvements, particularly on Gait3D. When the window length is further increased to n = 4, the performance slightly decreases, indicating that an excessively long local window may weaken the modeling of fine-grained short-term gait dynamics. Therefore, we adopt n = 3 and s = 1 as the default settings.

Table 7: Ablation studies of the dual-branch architecture on three datasets under Rank-1 accuracy.
<table><tr><td rowspan="2">Method</td><td colspan="2">2D Pose-Aware 3D SMPL</td><td colspan="2">SUSTech1K CCPG Gait3D</td></tr><tr><td>Branch</td><td>Branch</td><td></td><td></td></tr><tr><td>Sche.1</td><td>Sils</td><td>X</td><td>77.4</td><td>83.3 74.4</td></tr><tr><td>Sche.2 Sche.3</td><td>Skeleton</td><td>X</td><td>47.3 81.0</td><td>60.5 22.4 79.2</td></tr><tr><td>Sche.4</td><td>PPS DPPS</td><td>X X</td><td>81.3</td><td>85.7 85.9 81.1</td></tr><tr><td></td><td></td><td>SMPL</td><td>85.6</td><td></td></tr><tr><td>Sche.5(Ours)</td><td>DPPS</td><td></td><td></td><td>86.4 81.6</td></tr></table>

## 4.7.2. Efectiveness of Dual-Branch Architecture

As shown in Table 7, in single-stream settings using either traditional silhouettes (Sche.1) or skeletons (Sche.2) as input modalities, the recognition performance is relatively lower, highlighting the limitations of sparse structural cues in representing gait identity. Taking PPS with static explicit pose parsing (Sche.3) as input leads to some improvement, indicating partial benefits from skeletal priors. When replacing the silhouette with our proposed DPPS representation (Sche.4), the accuracy improves further. This validates that DPPS provides higher information entropy and more explicit dynamic motion cue guidance for gait recognition.

We then introduce the 3D SMPL branch (Sche.5) and achieve the best results. The 3D SMPL branch enhances the network in several key aspects: (i) Occlusion robustness. The strong statistical priors of SMPL regularize partial or noisy observations, enabling plausible full-body reconstruction even when limbs are missing or self-occluded. (ii) Appearance invariance. Operating in a clothing-agnostic mesh space efectively suppresses interference from garments or carried items. (iii) Viewpoint normalization. 3D articulation in canonical coordinates mitigates variations caused by camera perspective. (iv) Behavioral stability. Temporal pose parameters θ encode identity-consistent motion dynamics such as stride and cadence. Through this complementary integration and collaborative modeling, the dual-branch framework jointly exploits dynamic 2D cues and stable 3D behavioral priors, yielding more discriminative and environment-invariant gait representations for real-world scenarios.

Table 8: Comparison of diferent fusion strategies on three datasets under Rank-1 accuracy.
<table><tr><td>Method</td><td>SUSTech1K</td><td>CCPG</td><td>Gait3D</td></tr><tr><td>Mean</td><td>85.2</td><td>86.0</td><td>79.8</td></tr><tr><td>Concatenation</td><td>85.2</td><td>85.5</td><td>80.5</td></tr><tr><td>SMPLGait-style Fusion [11]</td><td>85.3</td><td>84.6</td><td>79.1</td></tr><tr><td>Attention [14]</td><td>83.6</td><td>82.1</td><td>80.3</td></tr><tr><td>RAFusion(Ours)</td><td>85.6</td><td>86.4</td><td>81.6</td></tr></table>

Incorporating the SMPL branch yields varying performance gains across datasets, as shown in Table 7. This variation may be attributed to the dataset-dependent complementarity between the 2D DPPS representation and 3D SMPL-based motion cues. The 3D pose dynamics may provide a larger additional benefit for SUSTech1K. Importantly, the overall performance gain does not arise solely from DPPS. As shown in Table 2 and Table 3, under the same Sils+SMPL inputs, DPA-Gait outperforms SMPLGait by 7.9, 18.9, and 28.2 percentage points on SUSTech1K, CCPG, and Gait3D, respectively, further demonstrating the efectiveness of the proposed dual-branch framework.

## 4.7.3. Comparison among Diferent Fusion Modules

As summarized in Table 8, our proposed Residual Attention Fusion module consistently achieves the best Rank-1 accuracy across all three datasets, outperforming representative fusion strategies including Mean, Concatenation, SMPLGait-style fusion, and Attention Fusion methods. This performance gain highlights the importance of jointly modeling complementary cues from DPPS and SMPL features while avoiding the feature dilution or misalignment issues commonly observed in naive fusion schemes.

The limitations of the alternative fusion baselines are interpretable. First, Mean and Concatenation fusion implicitly treat DPPS and SMPL as equally reliable sources and aggregate them uniformly. This coarse integration tends to wash out fine-grained, modality-specific cues, and it cannot adaptively down-weight noisy signals. Second, SMPLGait-style Fusion reshapes the silhouette and SMPL features onto the same spatial grid forcibly and then performs element-wise multiplication. While this design enforces a strict alignment between the two streams, the aggressive spatial warping can distort the original SMPL representation and partially destroy its physically meaningful structure, leading to information loss. Additionally, the Attention Fusion method improves upon integration by learning element-wise importance weights and applying a weightedsum aggregation. However, the absence of a residual pathway makes the fusion process prone to instability: it may oversuppress informative motion cues or over-amplify noisy activations, as shown by the visualization heatmap in Fig. 5. In contrast, our RAFusion introduces a learnable residual-attention mechanism that models the diference between modalities while preserving essential identity cues. This design enables the network to selectively compensate for missing or unreliable information, rather than averaging or blindly attending to features. The residual formulation stabilizes optimization and maintains structural consistency, particularly when one modality becomes unreliable.

Beyond quantitative improvements, the qualitative heatmap visualization in Fig. 5 further demonstrates the superiority of RAFusion. Under normal conditions, RAFusion produces clear and compact activation responses around limb-swing and torso regions, which are well aligned with key discriminative gait cues. Under occlusion or night illumination, baseline fusion methods exhibit dispersed or noisy activations, while RAFusion continues to highlight coherent and semantically meaningful regions. This suggests that RAFusion efectively mitigates modality noise and preserves behavior-related dynamics, validating its role as a robust and adaptive fusion mechanism for multi-modal gait recognition.

![](images/da4ad9fb16612e24edaf2ea754f09b6cdfc0ad2a86423509f1e360be8593d783.jpg)  
Figure 5: The heatmaps generated by DeepGaitV2, Attention Fusion, and our RAFusion methods (from the second to the last row) on three datasets under diferent conditions: Normal, Night, and Occlusion.

## 5. Conclusion

In this work, we propose a dynamic pose-knowledge-aware gait recognition framework, which integrates DPPS representation and SMPL-driven modeling. This design efectively enhances the robustness and accuracy of gait recognition in complex real-world scenarios. The DPPS module enriches the input modality with structural semantics, while the 3D SMPL branch facilitates the learning of identity-discriminative behavioral motion cues. Extensive experiments on multiple benchmark datasets validate the efectiveness and generalizability of the proposed method. In future work, we plan to explore more eficient strategies for modeling joint-level motion patterns and develop finer-grained, learnable human parsing mechanisms, as well as a better fusion strategy, to further enhance the expressiveness of multimodal gait representation and fusion.

## 6. Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant 62376100.

## References

[1] H. Li, Y. Qiu, H. Zhao, J. Zhan, R. Chen, T. Wei, Z. Huang, Gaitslice: A gait recognition model based on spatio-temporal slice features, Pattern Recognition 124 (2022) 108453.

[2] H. Chao, K. Wang, Y. He, J. Zhang, J. Feng, Gaitset: Cross-view gait recognition through utilizing gait as a deep set, IEEE transactions on pattern analysis and machine intelligence 44 (7) (2021) 3467–3478.

[3] Z. Zhu, X. Guo, T. Yang, J. Huang, J. Deng, G. Huang, D. Du, J. Lu, J. Zhou, Gait recognition in the wild: A benchmark, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 14789–14799.

[4] F. M. Castro, R. Delgado-Escaño, R. Hernández-García, M. J. Marín-Jiménez, N. Guil, Attengait: Gait recognition with attention and rich modalities, Pattern Recognition 148 (2024) 110171.

[5] F. Wang, X. Xiang, J. Cheng, A. L. Yuille, Normface: L2 hypersphere embedding for face verification, in: Proceedings of the 25th ACM international conference on Multimedia, 2017, pp. 1041–1049.

[6] W. Jia, W. Xia, B. Zhang, Y. Zhao, L. Fei, W. Kang, D. Huang, G. Guo, A survey on dorsal hand vein biometrics, Pattern Recognition 120 (2021) 108122.

[7] J. Huang, S. Bhattacharjee, S. Marcel, W. Kang, Study of full-view finger vein biometrics on redundancy analysis and dynamic feature extraction, IEEE Transactions on Information Forensics and Security (2025) 1–1.

[8] L. Fei, B. Zhang, Y. Xu, W. Jia, J. Wen, J. Wu, Precision direction and compact surface type representation for 3d palmprint identification, Pattern Recognition 87 (2019) 237–247.

[9] W. Song, W. Kang, L. Wang, Z. Lin, M. Gan, Video understanding-based random hand gesture authentication, IEEE Transactions on Biometrics, Behavior, and Identity Science 4 (4) (2022) 453–470.

[10] Z. Mu, F. M. Castro, M. J. Marin-Jimenez, N. Guil, Y.-R. Li, S. Yu, Resgait: The real-scene gait dataset, in: 2021 IEEE international joint conference on biometrics (IJCB), IEEE, 2021, pp. 1–8.

[11] J. Zheng, X. Liu, W. Liu, L. He, C. Yan, T. Mei, Gait recognition in the wild with dense 3d representations and a benchmark, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 20228–20237.

[12] C. Shen, C. Fan, W. Wu, R. Wang, G. Q. Huang, S. Yu, Lidargait: Benchmarking 3d gait recognition with point clouds, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1054–1063.

[13] C. Fan, S. Hou, J. Liang, C. Shen, J. Ma, D. Jin, Y. Huang, S. Yu, Opengait: A comprehensive benchmark study for gait recognition towards better practicality, IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

[14] C. Fan, J. Ma, D. Jin, C. Shen, S. Yu, Skeletongait: Gait recognition using skeleton maps, in: Proceedings of the AAAI conference on artificial intelligence, Vol. 38, 2024, pp. 1662–1669.

[15] C. Fan, S. Hou, Y. Huang, S. Yu, Exploring deep models for practical gait recognition, arXiv preprint arXiv:2303.03301 (2023).

[16] B. Lin, S. Zhang, X. Yu, Gait recognition via efective global-local feature representation and local temporal aggregation, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 14648–14656.

[17] S. Yu, D. Tan, T. Tan, A framework for evaluating the efect of view angle, clothing and carrying condition on gait recognition, in: 18th international conference on pattern recognition (ICPR’06), Vol. 4, IEEE, 2006, pp. 441–444.

[18] N. Takemura, Y. Makihara, D. Muramatsu, T. Echigo, Y. Yagi, Multi-view large population gait dataset and its performance evaluation for cross-view gait recognition, IPSJ transactions on Computer Vision and Applications 10 (2018) 1–14.

[19] C. Fan, Y. Peng, C. Cao, X. Liu, S. Hou, J. Chi, Y. Huang, Q. Li, Z. He, Gaitpart: Temporal part-based model for gait recognition, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 14225–14233.

[20] J. Zheng, X. Liu, S. Wang, L. Wang, C. Yan, W. Liu, Parsing is all you need for accurate gait recognition in the wild, in: Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 116–124.

[21] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, M. J. Black, Smpl: A skinned multiperson linear model, in: Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2023, pp. 851–866.

[22] X. Li, Y. Makihara, C. Xu, Y. Yagi, End-to-end model-based gait recognition using synchronized multi-view pose constraint, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 4106–4115.

[23] Z. Zhang, L. Tran, X. Yin, Y. Atoum, X. Liu, J. Wan, N. Wang, Gait recognition via disentangled representation learning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4710–4719.

[24] L. Wang, B. Liu, F. Liang, B. Wang, Hierarchical spatio-temporal representation learning for gait recognition, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE, 2023, pp. 19582–19592.

[25] X. Li, Y. Makihara, C. Xu, Y. Yagi, M. Ren, Gait recognition via semi-supervised disentangled representation learning to identity and covariate features, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 13309–13319.

[26] X. Li, Y. Makihara, C. Xu, Y. Yagi, Multi-view large population gait database with human meshes and its performance evaluation, IEEE Transactions on Biometrics, Behavior, and Identity Science 4 (2) (2022) 234–248.

[27] Z. He, W. Wang, J. Dong, T. Tan, Temporal sparse adversarial attack on sequence-based gait recognition, Pattern Recognition 133 (2023) 109028.

[28] J. Han, B. Bhanu, Individual recognition using gait energy image, IEEE transactions on pattern analysis and machine intelligence 28 (2) (2005) 316–322.

[29] W. Lang, S. Hou, Y. Huang, Beyond sparse keypoints: Dense pose modeling for robust gait recognition, in: Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 669–678.

[30] R. Urtasun, P. Fua, 3d tracking for gait characterization and recognition, in: Sixth IEEE International Conference on Automatic Face and Gesture Recognition, 2004. Proceedings., IEEE, 2004, pp. 17–22.

[31] R. Liao, S. Yu, W. An, Y. Huang, A model-based gait recognition method with body pose and human prior knowledge, Pattern Recognition 98 (2020) 107069.

[32] T. Teepe, A. Khan, J. Gilg, F. Herzog, S. Hörmann, G. Rigoll, Gaitgraph: Graph convolutional network for skeleton-based gait recognition, in: 2021 IEEE international conference on image processing (ICIP), IEEE, 2021, pp. 2314–2318.

[33] K. Yamauchi, B. Bhanu, H. Saito, Recognition of walking humans in 3d: Initial results, in: 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, IEEE, 2009, pp. 45–52.

[34] M. Wang, X. Guo, B. Lin, T. Yang, Z. Zhu, L. Li, S. Zhang, X. Yu, Dygait: Exploiting dynamic representations for high-performance gait recognition, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 13424–13433.

[35] H. Dou, P. Zhang, W. Su, Y. Yu, Y. Lin, X. Li, Gaitgci: Generative counterfactual intervention for gait recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 5578–5588.

[36] G. Peng, Y. Wang, S. Zhang, R. Li, Y. Zhao, A. Li, Rsanet: Relative-sequence quality assessment network for gait recognition in the wild, Pattern Recognition 161 (2025) 111219.

[37] K. Ma, Y. Fu, C. Cao, S. Hou, Y. Huang, D. Zheng, Learning visual prompt for gait recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 593–603.

[38] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, C. L. Zitnick, Microsoft coco: Common objects in context, in: Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, Springer, 2014, pp. 740–755.

[39] J. Wang, K. Sun, T. Cheng, B. Jiang, C. Deng, Y. Zhao, D. Liu, Y. Mu, M. Tan, X. Wang, et al., Deep high-resolution representation learning for visual recognition, IEEE transactions on pattern analysis and machine intelligence 43 (10) (2020) 3349–3364.

[40] H. Bilen, B. Fernando, E. Gavves, A. Vedaldi, S. Gould, Dynamic image networks for action recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 3034–3042.

[41] Z. Lin, W. Song, W. Kang, Multiscale super-images for dynamic hand gesture authentication, IEEE Transactions on Circuits and Systems for Video Technology (2025).

[42] Y. Sun, Q. Bao, W. Liu, Y. Fu, M. J. Black, T. Mei, Monocular, one-stage, regression of multiple 3d people, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 11179–11188.

[43] O. Taheri, N. Ghorbani, M. J. Black, D. Tzionas, Grab: A dataset of whole-body human grasping of objects, in: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, Springer, 2020, pp. 581–600.

[44] H. Luo, Y. Gu, X. Liao, S. Lai, W. Jiang, Bag of tricks and a strong baseline for deep person re-identification, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2019, pp. 0–0.

[45] W. Li, S. Hou, C. Zhang, C. Cao, X. Liu, Y. Huang, Y. Zhao, An in-depth exploration of person re-identification and gait recognition in cloth-changing conditions, in: Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13824– 13833.

[46] T. Teepe, J. Gilg, F. Herzog, S. Hörmann, G. Rigoll, Towards a deeper understanding of skeleton-based gait recognition, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 1569–1577.

[47] C. Zhang, X.-P. Chen, G.-Q. Han, X.-J. Liu, Spatial transformer network on skeleton-based gait recognition, Expert Systems 40 (6) (2023) e13244.

[48] J. Liang, C. Fan, S. Hou, C. Shen, Y. Huang, S. Yu, Gaitedge: Beyond plain end-to-end gait recognition for better practicality, in: European conference on computer vision, Springer, 2022, pp. 375–390.

[49] D. Ye, C. Fan, J. Ma, X. Liu, S. Yu, Biggait: Learning gait representation you want by large vision models, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 200–210.

[50] Y. Peng, K. Ma, Y. Zhang, Z. He, Learning rich features for gait recognition by integrating skeletons and silhouettes, Multimedia Tools and Applications 83 (3) (2024) 7273–7294.

[51] J. Zheng, X. Liu, B. Zhang, C. Yan, J. Zhang, W. Liu, Y. Zhang, It takes two: Accurate gait recognition in the wild via cross-granularity alignment, in: Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 8786–8794.

[52] D. Jin, C. Fan, W. Chen, S. Yu, Exploring more from multiple gait modalities for human identification, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 4120–4128.

[53] Z. Wang, S. Hou, M. Zhang, X. Liu, C. Cao, Y. Huang, P. Li, S. Xu, Qagait: Revisit gait recognition from a quality perspective, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, 2024, pp. 5785–5793.

[54] H. Zhu, W. Zheng, Z. Zheng, R. Nevatia, Gaitref: Gait recognition with refined sequential skeletons, in: 2023 IEEE International Joint Conference on Biometrics (IJCB), IEEE, 2023, pp. 1–10.

[55] S. Zou, J. Xiong, C. Fan, C. Shen, S. Yu, J. Tang, A multi-stage adaptive feature fusion neural network for multimodal gait recognition, IEEE Transactions on Biometrics, Behavior, and Identity Science 6 (4) (2024) 539–549.