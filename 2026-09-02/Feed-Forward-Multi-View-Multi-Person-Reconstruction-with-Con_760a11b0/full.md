# Feed-Forward Multi-View Multi-Person Reconstruction with Contrastive Human-Aware 3D Representation

Yuanwang Yang<sup>1†</sup>, Buzhen Huang<sup>1†</sup>, Zongxuan Ren<sup>1</sup>, Jing Huang<sup>1</sup>, Kun Li<sup>1\*</sup>

<sup>1</sup>College of Intelligence and Computing, Tianjin University, 135 Yaguan Road, Tianjin, 300350, Tianjin, China.

\*Corresponding author(s). E-mail(s): lik@tju.edu.cn; Contributing authors: yyw@tju.edu.cn; hbz@tju.edu.cn; xuan591847902@163.com; hj00@tju.edu.cn; <sup>†</sup>Equal contribution.

## Abstract

Multi-view human reconstruction has been extensively studied under simplified settings, yet scaling these methods to robust and eficient multi-person reconstruction in unconstrained environments requires a more general and scalable modeling paradigm. Existing bottom-up methods often rely on accurate camera calibration and explicit cross-view matching, and therefore struggle in multi-person scenarios with severe occlusions and ambiguities. We attribute these limitations to the lack of an explicit and view-agnostic 3D representation for jointly reasoning about human structure and multi-view consistency. Based on this observation, we propose a new top-down paradigm that maintains a unified, instance-centric human-aware 3D space, enabling simultaneous camera calibration, cross-view association, and human reconstruction via crossmodal contrastive learning. Specifically, observations from multiple views are lifted and fused directly into this shared 3D space, where geometric structure, visual appearance, and human-centric semantic cues are jointly encoded at the instance level. To enhance the discriminability and consistency of the 3D representation, we introduce a spatial contrastive learning strategy that aligns 3D features corresponding to the same human instance across diferent views and modalities, while separating features from diferent instances. This design allows correspondence reasoning, semantic aggregation, and instance discrimination to be performed natively in the 3D space, which enforces cross-view consistency and

improves robustness under severe occlusions. Based on the learned instance-aware 3D representation, we recover structured human body models in a feed-forward manner by regressing SMPL parameters from instance-level 3D human tokens. Extensive experiments demonstrate that adopting a unified human-aware 3D feature space as the core representation leads to robust, accurate, and eficient multi-view human reconstruction in challenging real-world scenarios. The code and data will be made publicly available.

Keywords: Unconstrained 3D reconstruction, Multi-person pose and shape estimation, Human-scene interaction

Input Uncalibrated, Multi-view Images  
![](images/c01bb75c84ab36a3b5bf27414f354a137bd47a8a3694c17ef70458ff35698dd7.jpg)

![](images/eff094204538796a0beb3701a7209c69cf496048cbb0053fb15983d9bd1e02aa.jpg)  
Fig. 1: We propose a unified, human-aware 3D representation learned via cross-modal contrastive learning. This representation provides a persistent and view-agnostic 3D space in which camera calibration, cross-view association, and human reconstruction can be jointly performed.

## 1 Introduction

Reconstructing 3D humans from multiple camera views is a long-standing problem with wide applications in VR/AR, sports analysis, and human–computer interaction. Although extensive research has been devoted to this topic, achieving robust and eficient reconstruction in complex real-world scenarios remains challenging. Traditional multi-view systems [1–4] rely heavily on accurate camera calibration and well-controlled environments, making them fragile under practical conditions such as imperfect setups, heavy occlusions, or large-scale scenes (Fig.1).

The majority of existing multi-view human reconstruction approaches [3, 5, 6] follow a bottom-up “2D-to-3D” paradigm. They first extract 2D cues (e.g., keypoints, silhouettes, or per-view body parameters) from each image, establish cross-view correspondences, and then fuse these observations into a 3D representation. As one of the dominant approaches, triangulation-based methods directly associate and fuse multi-view 2D observations with camera geometry, but minor inaccuracies in camera parameters can propagate and result in global geometric inconsistencies [3–5, 7]. Although some works [6, 8–10] relax the reliance on precise camera geometry with semantic association, they remain inherently vulnerable to occlusion, as missing or corrupted 2D observations directly undermine correspondence estimation and subsequent feature fusion. In addition, these methods often depend on iterative optimization or matching procedures, which are computationally expensive and dificult to scale to complex scenes or real-time applications.

These challenges are not merely implementation issues, but are inherent to the bottom-up paradigm, which reasons about 3D structure through fragmented and viewdependent 2D observations. In contrast, humans and objects exist and interact in a shared 3D space, and a top-down approach enforces multi-view consistency by reasoning directly within a unified 3D representation rather than across independent image views. This motivates a shift in perspective: instead of treating 3D reconstruction as the result of fusing 2D predictions, we advocate for maintaining an explicit, shared 3D representation as the central reasoning space throughout the reconstruction process.

In this work, we propose a new top-down paradigm that maintains a unified humanaware 3D space to enable simultaneous camera calibration, cross-view association, and human reconstruction. With this 3D space, multi-view information is lifted and fused at the feature level, where geometric structure, appearance cues, and human-centric semantics are jointly encoded in a shared representation. To enhance the discriminability and consistency of the 3D representation, a spatial contrastive learning strategy is further introduced. Consequently, this 3D feature space serves as a persistent, viewagnostic set of spatial tokens, and thus cross-view association, occlusion handling, and instance reasoning can be performed natively within the 3D space.

Specifically, given uncalibrated multi-view images, we initialize the 3D space with a geometry-aware prior [11] and subsequently finetune the representation through diferentiable 3D Gaussian Splatting (3DGS) rendering [12]. After initialization, the 3D representation encodes only limited human prior knowledge and lacks explicit instance discriminability. To this end, we further introduce a spatial contrastive learning strategy to improve the consistency and discriminability of the 3D feature space. In particular, human points in the 3D space are re-projected into pose, appearance, and geometry fields to sample expressive human semantics. We then apply contrastive learning across sampled features from diferent views and instances to enforce semantic consistency. Meanwhile, the sampled features are used to regress identity embeddings, which are further contrasted across the 3D and 2D spaces. Spatial contrastive learning not only enriches the human prior knowledge within the 3D space, but also improves geometric alignment (e.g., camera parameters) by enforcing semantic consistency. With the learned 3D representation, structured SMPL parameters [13] are finally regressed in a feed-forward manner from instance-aware 3D human tokens.

In summary, our main contributions are as follows:

• We propose a novel multi-view human reconstruction framework that explicitly maintains a unified 3D feature space as the central representation, which enables feed-forward and simultaneous calibration, association and reconstruction.

• We design a human-aware 3D representation and refinement strategy that integrates multi-view geometry, appearance, and human semantics at the representation level, thereby substantially improving robustness to occlusion and camera calibration errors.

• We introduce a spatial contrastive learning strategy that leverages semantic consistency to enhance geometric alignment and reconstruction accuracy in multi-view, multi-person scenarios.

## 2 Related Work

## 2.1 Human Reconstruction from Unconstrained Cameras

Real-world deployments often involve unknown, unsynchronized, or distorted cameras, as well as sparse and wide-baseline viewpoints. The key challenge is that uncalibrated cameras prevent direct enforcement of multi-view geometric consistency, so both camera parameters and 3D humans must be inferred from weak and noisy cues; this coupling is particularly brittle under occlusion, wide baselines, distortion, and unsynchronized viewpoints. To deal with these unconstrained settings, several methods [8, 10, 14–16] jointly estimate camera parameters and 3D human poses. HSfM [10] uses SMPL-derived anthropometric constraints as metric references and jointly optimizes scene structure, human meshes, and camera parameters through bun dle adjustment, addressing both scale ambiguity and missing calibration. Yu et al. [8] map per-view image features to a canonical human surface that is invariant to viewpoint, pose and fuse them via confidence-weighted attention, enabling reconstruction without camera calibration. Additional approaches [15] incorporate depth or 3D reidentification cues for cross-view association but require extra sensors. EasyRet3D [17] addresses uncalibrated multi-view multi-human reconstruction by explicitly modeling cross-view identity association through a tracking-based formulation. However, it introduces a dependency on explicit tracking quality and can be sensitive to association errors in crowded or heavily occluded scenes. Kineo [16] focuses on self-calibration and geometric recovery by jointly estimating camera intrinsics/extrinsics and 3D structure from weak observations, leveraging robust sampling, temporal consistency, and anthropometric priors to resolve scale and depth ambiguities. Although these methods demonstrate the feasibility of calibration-free reconstruction, they still rely on accurate 2D detections, robust cross-view matching, or reliable dense semantic correspondences.

## 2.2 Multi-View 3D Human Pose and Shape Estimation.

As a foundational component of 3D human reconstruction, estimating pose from multiple views has been extensively studied [18]. Early approaches [18] follow a bottom-up pipeline: 2D poses are detected independently per view and subsequently triangulated or fitted by kinematic models to recover 3D joint locations. Such pipelines are highly sensitive to 2D detection noise and often fail under heavy occlusion or multiperson interaction. To mitigate errors introduced during 2D association, voxel-based methods [4, 19] aggregate multi-view features into a unified 3D space and perform inference directly on volumetric representations, achieving improved robustness at the cost of substantial computation and memory. Transformer-based methods further enable end-to-end multi-view reasoning. For instance, MvP [5] leverages learnable joint queries and projective attention to fuse cross-view geometry, while MV-SSM [20] introduces state-space modeling to enhance generalization to unseen camera layouts. Meanwhile, multi-view human mesh reconstruction has been widely studied, particularly in controlled environments where intrinsic and extrinsic camera parameters are known [21–23]. Classical optimization-based methods extend single-view body models (e.g., SMPL/SMPL-X [13]) by jointly refining camera poses, keypoints, silhouettes, and mesh parameters, achieving metrically consistent meshes when calibration and initialization are reliable [24, 25]. More recent learning-based approaches aggregate multi-view features to directly regress a coherent mesh [26].

## 2.3 Feed-forward 3D Reconstruction for Scene and Humans.

Feed-forward 3D reconstruction [11, 27–32] has recently emerged as an eficient method for predicting dense 3D structures from images directly, without relying on iterative optimization. VGGT introduces a feed-forward multi-view geometry framework that directly predicts camera parameters together with dense geometric outputs (e.g., depth and point-based 3D representations) from one to many images in a single forward pass, avoiding expensive test-time optimization. Building on this capability, recent works increasingly treat VGGT-style predictors as a front-end to bootstrap geometry and camera estimates, and then plug these initialization signals into downstream tasks such as rendering, tracking, interaction understanding, or task-specific refinement. Meanwhile, recent work has begun to reconstruct humans, cameras, and the surrounding scene in a shared metric coordinate system, rather than treating human mesh recovery and scene reconstruction as separate problems. HSfM [10] tackles this problem from sparse, uncalibrated multi-view images by jointly reconstructing human meshes, the surrounding scene, and cameras, using human observations to stabilize camera estimation and scene geometry. SynCHMR [33] addresses monocular videos by combining human motion recovery with a SLAM-style formulation to estimate metric camera motion, scene geometry, and a consistent human trajectory in a unified coordinate system. JOSH [34] performs optimization-based 4D human–scene reconstruction from monocular videos by leveraging human–scene contact constraints to jointly refine human motion, camera poses, and dense scene geometry, while its distilled variant JOSH3R [34] enables eficient feed-forward inference trained from JOSH pseudo-labels. Human3R [35] builds on CUT3R [36] and performs feed-forward human–scene reconstruction from monocular videos, jointly predicting multi-person SMPL-X bodies, camera motion, and dense scene geometry in a shared world coordinate system.Recently, HAMSt3R [37] extends MASt3R [38] with human-aware heads and performs fully feed-forward reconstruction, and it predicts dense 3D point maps directly from sparse views, enabling eficient joint recovery of the scene and people. Successfully combines feed-forward 3D reconstruction with joint optimization of the scene and human bodies. In contrast, we construct a unified implicit 3D embedding space that jointly models pose, appearance, and spatial relationships in an endto-end manner. This enables direct cross-view correspondence learning and feature fusion in 3D space, substantially improving the robustness of multi-human SMPL reconstruction under unconstrained settings.

![](images/43ed930cb2ef718f04470e870ced7a3a63d7f254e602be51088a989fc47f9b94.jpg)  
Fig. 2: Overview. Given a set of multi-view unconstrained images, our method first initializes a 3D space using a geometry-aware prior and 3D Gaussian Splatting (a). To further enhance human prior knowledge and instance-level discriminability for the 3D representation, spatial contrastive learning is applied by sampling across diferent semantic fields (b). Finally, camera and human representations are jointly optimized within the 3D space and regressed in a feed-forward fashion (c).

## 3 Method

As shown in Fig.2, we present a human-aware multi-view 3D reconstruction framework that explicitly maintains and optimizes a unified 3D feature space as an intermediate representation. Instead of reasoning directly in the 2D image domain or regressing parametric body models from per-view features, our approach encodes geometric structure, semantic information, and human identity within this shared 3D space. This design allows camera calibration, cross-view association, and human reconstruction to be addressed jointly in a feed-forward pipeline. Details of the proposed framework are described in the following sections.

## 3.1 Human-aware Unified 3D Feature Space Initialization

Directly operating on 2D features is inherently fragile under occlusions, viewpoint changes, and inter-view misalignment. While multi-view aggregation alleviates some of these issues, ambiguity remains when reasoning purely in image space. We therefore lift image features into 3D and maintain an explicit spatial representation, allowing information to be aggregated, filtered, and refined in a geometry-aware manner, which naturally mitigates these issues.

## 3.1.1 Construction of 3D Feature Space.

As shown in Fig.2 (a), given a set of uncalibrated images $\{ I _ { v } \} _ { v = 1 } ^ { V }$ from V views, we first construct a view-consistent 3D feature tokens:

$$
\mathcal { T } = \{ ( { \bf p } _ { i } , { \bf f } _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{1}
$$

where $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ denotes a 3D location and $\mathbf { f } _ { i } \in \mathbb { R } ^ { C }$ represents a learned 3D feature embedding. N is the number of points and varies with the complexity of the scene.

To initialize the 3D feature space, we employ a pretrained VGGT model [11] as a geometry-aware prior. Given each input image $I _ { v } { \mathrm { . } }$ , VGGT predicts a depth map $D _ { v } ,$ , a per-pixel confidence map $C _ { v } ,$ a dense feature map $F _ { v }$ and per-view camera parameters $p _ { v } \colon$

$$
( D _ { v } , C _ { v } , F _ { v } , p _ { v } ) = \Phi _ { \mathrm { V G G T } } ( I _ { v } ) .\tag{2}
$$

Using the predicted depth and camera parameters, each pixel $\mathbf { u } \in \mathbb { R } ^ { 2 }$ is unprojected into 3D space:

$$
\begin{array} { r } { \mathbf { x } _ { v } ( \mathbf { u } ) = \pi ^ { - 1 } ( \mathbf { u } , D _ { v } ( \mathbf { u } ) ; p _ { v } ) . } \end{array}\tag{3}
$$

Thus, the image features $F _ { v } ( { \bf u } )$ and corresponding confidence weights $C _ { v } ( { \mathbf { u } } )$ can be mapped into 3D space.

$$
\mathbf { f } _ { v } ( \mathbf { u } ) = F _ { v } ( \mathbf { u } ) ,\tag{4}
$$

$$
\begin{array} { r } { c _ { v } ( \mathbf { u } ) = C _ { v } ( \mathbf { u } ) . } \end{array}\tag{5}
$$

This process yields a dense, view-dependent 3D representation, but directly operating on it is computationally expensive and prone to errors from noisy depth estimation and redundant observations. To obtain a more stable and compact representation, we aggregate neighboring 3D points from multiple views using a confidence-weighted soft aggregation scheme, where aggregation weights are computed as

$$
w _ { i j } = \boldsymbol { c } _ { i } \cdot \boldsymbol { c } _ { j } \cdot \exp \Big ( - \frac { \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \Big ) ,\tag{6}
$$

where $\sigma$ is a fixed hyper-parameter controlling the spatial aggregation range, and the fused point center $\mathbf { p } _ { i }$ and feature $\mathbf { f } _ { i }$ are then defined as

$$
\mathbf { p } _ { i } = \frac { \sum _ { j } \tilde { w } _ { i j } \mathbf { x } _ { j } } { \sum _ { j } \tilde { w } _ { i j } } , \quad \mathbf { f } _ { i } = \frac { \sum _ { j } \tilde { w } _ { i j } \mathbf { f } _ { j } } { \sum _ { j } \tilde { w } _ { i j } } .\tag{7}
$$

In contrast to voxelization [19, 39], our representation operates directly on continuous 3D points using soft spatial aggregation, avoiding explicit discretization into fixed voxel grids. This formulation is more flexible with respect to varying scene scales and irregular point distributions. With the aggregation, the 3D feature space is viewagnostic and spatially anchored, forming a robust 3D feature tokens that integrate multi-view evidence while suppressing noisy or inconsistent observations.

## 3.1.2 Human-aware 3D Feature Decoding.

Since the 3D features from VGGT only describe geometry cues, we then explicitly encode human-related semantics into the 3D feature space. Each fused 3D feature token $( \mathbf { p } _ { i } , \mathbf { f } _ { i } )$ serves as a shared latent representation from which multiple attributes are decoded. Specifically, a lightweight MLP $g _ { \theta }$ maps $\mathbf { f } _ { i }$ to Gaussian attributes [12]:

$$
\begin{array} { r } { ( \rho _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } , \mathbf { h } _ { i } ^ { \mathrm { S H } } , z _ { i } ) = g _ { \theta } ( \mathbf { f } _ { i } ) , } \end{array}\tag{8}
$$

where $\rho _ { i }$ denotes density, $\mathbf { s } _ { i }$ scale, $\mathbf { q } _ { i }$ rotation quaternion, $\mathbf { h } _ { i } ^ { \mathrm { S H } }$ spherical harmonic coeficients, and $z _ { i }$ is a human-related logit.

The Gaussian opacity is obtained via a bounded monotonic mapping:

$$
\alpha _ { i } = \psi ( \rho _ { i } ) ,\tag{9}
$$

while the human confidence is defined as

$$
h _ { i } = \sigma ( z _ { i } ) .\tag{10}
$$

The human confidence is defined at the 3D level rather than per image. This enables consistent human reasoning across views and prevents conflicting predictions caused by occlusion or partial visibility in individual images. Moreover, both geometric, appearance, and semantic attributes are decoded from the same 3D feature space. This allows the representation to jointly encode scene structure and human-related semantics, rather than treating human segmentation as a separate 2D task.

## 3.1.3 Initial 3D Space Training.

We train the initial 3D space by learning Gaussian parameters and the human confidence field, without introducing instance-level supervision or SMPL regression.

Specifically, each 3D token is associated with a Gaussian primitive parameterized by its position $\mathbf { p } _ { i } ,$ , scale ${ \bf s } _ { i } ,$ , rotation $\mathbf { q } _ { i } ,$ , opacity $\alpha _ { i }$ , and spherical harmonic coeficients $\mathbf { h } _ { i } ^ { \mathrm { S H } }$ . Given a camera view $v ,$ we render color images by splatting these Gaussians onto the image plane. For a pixel x, the rendered color is computed as

$$
\hat { \mathrm { \bf ~ I } } _ { v } ( \mathrm { \bf x } ) = \sum _ { i } \mathrm { \bf c } _ { i } ( \mathrm { \bf x } ) \alpha _ { i } \mathcal { G } _ { i } ( \mathrm { \bf x } ) \prod _ { j < i } \big ( 1 - \alpha _ { j } \mathcal { G } _ { j } ( \mathrm { \bf x } ) \big ) ,\tag{11}
$$

where $\mathcal { G } _ { i } ( \mathbf { x } )$ denotes the projected Gaussian footprint and $\mathbf { c } _ { i } ( \mathbf { x } )$ is the view-dependent color obtained from spherical harmonics. The photometric reconstruction loss $\mathcal { L } _ { \mathrm { r g b } }$ supervises Gaussian appearance by comparing rendered images with the input views:

$$
\mathcal { L } _ { \mathrm { r g b } } = \sum _ { v } \sum _ { \mathbf { x } } \big \| \hat { \mathbf { I } } _ { v } ( \mathbf { x } ) - \mathbf { I } _ { v } ( \mathbf { x } ) \big \| _ { 1 } ,\tag{12}
$$

where $\hat { \mathbf { I } } _ { v }$ denotes the image rendered from Gaussians in view $v .$

To supervise human confidence in 3D, we render a soft human mask by modulating Gaussian opacity with the predicted confidence:

$$
\tilde { \alpha } _ { i } = \alpha _ { i } \cdot h _ { i } .\tag{13}
$$

Rasterizing $\tilde { \alpha } _ { i }$ yields a per-view human mask $\hat { M } _ { v }$ , which is supervised by available 2D masks $M _ { v }$ using a binary cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { m a s k } } = \sum _ { v } \mathrm { B C E } ( \hat { M } _ { v } , M _ { v } ) .\tag{14}
$$

We further introduce a regularization term $\mathcal { L } _ { \mathrm { r e g } }$ to stabilize Gaussian optimization and encourage spatially coherent human confidence. Specifically, we penalize excessively opaque Gaussians, regularize Gaussian scales to avoid degenerate footprints, and enforce local smoothness of the 3D human confidence field:

$$
\mathcal { L } _ { \mathrm { r e g } } = \lambda _ { \alpha } \sum _ { i } \alpha _ { i } ^ { 2 } + \lambda _ { s } \sum _ { i } \| \log \mathbf { s } _ { i } \| _ { 1 } + \lambda _ { h } \sum _ { i } \sum _ { j \in \mathcal { N } ( i ) } ( h _ { i } - h _ { j } ) ^ { 2 } ,\tag{15}
$$

where $\mathcal { N } ( i )$ denotes the spatial neighbors of token $i \ ( \mathrm { e . g . }$ , within a fixed radius in 3D).

## 3.2 3D Feature Enhancement with Spatial Contrastive Learning

The 3D feature space in Sec.3.1 provides stable anchors but may lose fine-grained details and lacks explicit instance discriminability. We therefore enhance the 3D token features and learn instance-discriminative fields with spatial contrastive learning by selectively sampling from more human semantics. Importantly, both enhancement and instance reasoning are defined on the shared 3D token field, which mitigates viewdependent ambiguities from occlusion and truncation.

## 3.2.1 Human Semantics Sampling

After initializing the 3D space, the resulting 3D features contain only limited humanrelated information; therefore, we further exploit richer semantic cues from the image domain. Specifically, we adopt ROMP [40] features $P _ { v }$ to provide additional pose information, since they encode pixel-wise human pose and body structure cues. The appearance cues from DINOv3 [41] $A _ { v }$ are also utilized. In addition, since the aggregation adjust the point positions, we further sampling from the VGGT feature maps $F _ { v }$

However, not all 3D tokens require semantic enhancement. We only focus on the tokens with high human confidence $h _ { i } .$ which are more likely to lie on human surfaces:

$$
\mathcal { H } = \{ i \mid h _ { i } > \tau _ { h } \} ,\tag{16}
$$

where $\tau _ { h }$ is a confidence threshold. Restricting the sampling points to H significantly reduces computational cost and prevents background clutter from contaminating the

![](images/87c79353a53fcadd99de323ae2238e5abb495cf2183eba7454a0e5ea8b21b58a.jpg)  
Fig. 3: Human points in the 3D space are projected into diferent semantic fields to obtain features for spatial contrastive learning.

refinement process. Tokens outside H retain their original geometric features and continue to contribute to scene reconstruction.

For each selected 3D token $( \mathbf { p } _ { i } , \mathbf { f } _ { i } )$ and each camera view $v ,$ we project the 3D position $\mathbf { p } _ { i }$ onto the image plane:

$$
\begin{array} { r } { \mathbf { u } _ { i } ^ { ( v ) } = \pi ( \mathbf { p } _ { i } ; p _ { v } ) , } \end{array}\tag{17}
$$

where $\pi ( \cdot )$ denotes the camera projection function defined by the estimated camera parameters.

To ensure reliable feature sampling, we perform visibility-aware filtering. A reprojection is considered valid if it satisfies two conditions: (i) the projected location lies within image boundaries, and (ii) the projected depth is consistent with the predicted depth map.

Formally, the visibility indicator $m _ { i } ^ { ( v ) }$ is defined as

$$
m _ { i } ^ { ( v ) } = \mathbb { I } ( \mathbf { u } _ { i } ^ { ( v ) } \in \Omega ) \cdot \mathbb { I } \big ( | z _ { i } ^ { ( v ) } - D _ { v } ( \mathbf { u } _ { i } ^ { ( v ) } ) | < \epsilon \big ) ,\tag{18}
$$

where Ω denotes the image domain, $z _ { i } ^ { ( v ) }$ is the projected depth of $\mathbf { p } _ { i }$ in view $v , D _ { v } ( \cdot )$ is the predicted depth map, and ϵ is a tolerance threshold.

As shown in ${ \mathrm { F i g . 3 } } ,$ by sampling diferent semantic maps $P _ { v } , A _ { v } .$ , and $F _ { v }$ with each valid reprojection point u<sub>i</sub>, we then obtain pose, appearance and geometry features $P _ { v } ( \mathbf { u } _ { i } ) , A _ { v } ( \mathbf { u } _ { i } )$ , and $F _ { v } ( \mathbf { u } _ { i } )$

## 3.2.2 Cross-modal Feature Fusion

The sampled features $P _ { v } ( \mathbf { u } _ { i } ) , A _ { v } ( \mathbf { u } _ { i } )$ , and $F _ { v } ( \mathbf { u } _ { i } )$ are concatenated as $\mathbf { g } _ { i } ^ { ( v ) }$ and then aggregated across multiple views to refine the original 3D feature. Each view contributes with a confidence weight proportional to its visibility:

$$
w _ { i } ^ { \left( v \right) } = \frac { m _ { i } ^ { \left( v \right) } } { \sum _ { v ^ { \prime } } m _ { i } ^ { \left( v ^ { \prime } \right) } + \delta } ,\tag{19}
$$

where $\delta$ is a small constant for numerical stability.

The refined 3D feature is computed via a residual fusion formulation:

$$
\tilde { \mathbf { f } } _ { i } = \mathbf { f } _ { i } + \sum _ { v } w _ { i } ^ { ( v ) } \cdot \phi \big ( [ \mathbf { f } _ { i } , \mathbf { g } _ { i } ^ { ( v ) } ] \big ) ,\tag{20}
$$

where $\phi ( \cdot )$ is a lightweight fusion network implemented as an MLP, and [·] denotes feature concatenation.

This residual design preserves the stability of the original 3D representation while selectively injecting high-level semantic information. Importantly, fusion is performed independently for each 3D token, enabling scalable and parallel refinement.

By selectively sampling and fusing 2D features through reprojection, we compensate for the insuficient human information and enable the refined 3D feature space to better align with articulated human structure. Moreover, repeated reprojection and semantic supervision implicitly enforce cross-view consistency, allowing the method to naturally scale to higher-resolution features and additional modalities without modifying the core architecture.

## 3.2.3 3D Instance Embedding and Spatial Contrastive Learning

Based on the refined 3D features, we further learn instance-aware representations to distinguish multiple humans in the shared 3D space. For each 3D token, we predict a compact instance embedding using a lightweight projection head:

$$
\mathbf { e } _ { i } = \frac { h _ { \phi } ( \tilde { \mathbf { f } } _ { i } ) } { \| h _ { \phi } ( \tilde { \mathbf { f } } _ { i } ) \| _ { 2 } } ,\tag{21}
$$

where $h _ { \phi }$ denotes an MLP and $\ell _ { 2 }$ normalization enforces a bounded embedding space. The projection head $h _ { \phi } ( \cdot )$ is optimized during contrastive learning, while the backbone features $\mathbf { f } _ { i }$ remain fixed. Instance embeddings are only computed for tokens with high human confidence, which suppresses background interference and reduces unnecessary computation.

To learn discriminative embeddings, we adopt a spatial contrastive learning objective that links 3D tokens with 2D instance supervision. Each 3D token is associated with a Gaussian primitive parameterized by its fixed geometric attributes (position, scale, rotation, and opacity) and its instance embedding. During rendering, the instance embedding is treated as the feature carried by the Gaussian and projected onto the image plane via Gaussian splatting. Geometric parameters are detached during this process to prevent degenerate solutions.

Formally, the rendered instance embedding map at pixel x is given by

$$
\mathbf { E } ( \mathbf { x } ) = \sum _ { i } \mathbf { e } _ { i } \alpha _ { i } \mathcal { G } _ { i } ( \mathbf { x } ) \prod _ { j < i } \big ( 1 - \alpha _ { j } \mathcal { G } _ { j } ( \mathbf { x } ) \big ) ,\tag{22}
$$

which softly aggregates contributions from visible 3D tokens along each viewing ray. This diferentiable rendering establishes a correspondence between 3D tokens and 2D pixels, allowing instance supervision to be propagated from image space back to the shared 3D representation.

We apply the same contrastive objective under three complementary supervision domains: intra-view, cross-view, and 3D-space contrastive learning. Intra-view contrastive learning is performed within each image to improve local instance discrimination and suppress ambiguity caused by overlapping humans in image space. Cross-view contrastive learning aligns embeddings belonging to the same person across diferent camera views, improving view consistency and cross-view association robustness. Finally, 3D-space contrastive learning directly regularizes the shared 3D representation after multi-view feature aggregation, enforcing identity consistency in the unified 3D space. These three forms of supervision are complementary: intraview supervision improves local separability, cross-view supervision improves identity consistency across viewpoints, while 3D-space supervision stabilizes the globally aggregated instance-aware representation. In all cases, embeddings belonging to the same person are encouraged to be close, while embeddings from diferent persons are separated. The contrastive loss is defined as

$$
\mathcal { L } _ { \mathrm { s c } } = - \sum _ { i } \log \frac { \exp ( \mathbf { e } _ { i } ^ { \top } \mathbf { u } _ { y _ { i } } ) } { \sum _ { k } \exp ( \mathbf { e } _ { i } ^ { \top } \mathbf { u } _ { y _ { k } } ) } ,\tag{23}
$$

where $y _ { i }$ denotes the instance label and $\mathbf { u } _ { y _ { i } }$ is the prototype embedding of instance y, computed as the mean of embeddings assigned to that instance. The same formulation in $\operatorname { E q }$ . 23 is used in all three settings, difering only in how instance labels are obtained.

The final spatial contrastive learning loss is defined as a weighted combination of the three terms:

$$
\mathcal { L } _ { \mathrm { S C L } } = \lambda _ { \mathrm { i v } } \mathcal { L } _ { \mathrm { s c } } ^ { \mathrm { i v } } + \lambda _ { \mathrm { c v } } \mathcal { L } _ { \mathrm { s c } } ^ { \mathrm { c v } } + \lambda _ { \mathrm { 3 D } } \mathcal { L } _ { \mathrm { s c } } ^ { \mathrm { 3 D } } ,\tag{24}
$$

where $\lambda _ { \mathrm { i v } } , ~ \lambda _ { \mathrm { c v } }$ , and $\lambda _ { \mathrm { 3 D } }$ balance the contributions of intra-view, cross-view, and 3D-space supervision.

By enforcing instance consistency directly in the shared 3D feature space, spatial contrastive learning enables robust multi-person separation across views and provides instance-aware 3D tokens for subsequent SMPL regression.

In addition to instance-level contrastive learning, we further introduce a semantic contrastive objective on $P _ { v } ( \mathbf { u } _ { i } ) , \ A _ { v } ( \mathbf { u } _ { i } )$ , and $F _ { v } ( \mathbf { u } _ { i } )$ to improve the consistency of semantic representations across views. The key observation is that, for a given person, semantic features extracted from diferent viewpoints should remain consistent, even under occlusion or partial visibility. Such consistency is essential for learning a coherent 3D feature space that captures both spatial structure and semantic meaning.

Concretely, we treat the refined 3D features as semantic descriptors of the underlying 3D tokens. For each human-related token, its 3D location is projected into multiple camera views, and only valid projections are selected based on visibility. Semantic features sampled from diferent views but corresponding to the same 3D token and the same identity are encouraged to be similar, while features associated with diferent identities are pushed apart. The contrastive loss is the same as Eq. (23). This identity-aware contrastive supervision is applied directly to semantic features, rather than instance embeddings.

By enforcing semantic consistency across views at the 3D level, the proposed objective improves the model’s ability to extract stable semantic representations over the entire 3D space. As a result, the learned features exhibit stronger view invariance and spatial coherence, which benefits subsequent SMPL regression.

## 3.2.4 Training with Spatial Contrastive Learning

During the contrastive learning stage, Gaussian parameters and the 3D feature tokens continue to be optimized jointly.

The spatial contrastive learning loss $\mathcal { L } _ { \mathrm { S C I } }$ is applied to both instance embeddings and semantic features sampled via multi-view reprojection. Instance-level contrastive supervision encourages 3D tokens belonging to the same person to share similar embeddings, while separating tokens from diferent individuals. In parallel, semantic contrastive supervision enforces view-consistent semantic representations for the same identity across diferent viewpoints.

Photometric reconstruction loss $\mathcal { L } _ { \mathrm { r g b } }$ is retained throughout this stage to preserve geometric consistency and prevent drift in the underlying 3D structure. By jointly optimizing geometry, appearance, and contrastive objectives, the model learns a discriminative and spatially coherent 3D feature space that supports robust multi-person reasoning.

## 3.3 SMPL Parameter Regression from Human Tokens

With the refined 3D representation, we then recover structured human body models for each person in the scene. Instead of iterative fitting, we adopt a feed-forward regression formulation that predicts SMPL parameters directly from the learned 3D representation.

## 3.3.1 Person-token pooling with human-guided weighting

Let $\{ ( \mathbf { p } _ { i } , \tilde { \mathbf { f } } _ { i } ) \} _ { i = 1 } ^ { N }$ denote the fused 3D tokens, where $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ is the 3D location and $\tilde { \mathbf { f } } _ { i } \in \mathbb { R } ^ { C }$ is the refined feature (Sec.3.2.2). We also obtain a human confidence score $h _ { i } \in [ 0 , 1 ]$ for each token and a pseudo instance label $\ell _ { i } \in \{ 0 , 1 , \ldots , P \}$ , where $\ell _ { i } = 0$ indicates background and $\ell _ { i } = p$ indicates the p-th person instance.

We first construct a point descriptor by concatenating 3D features and coordinates, followed by a projection MLP:

$$
\mathbf { d } _ { i } = \psi \big ( [ \widetilde { \mathbf { f } } _ { i } ; \mathbf { p } _ { i } ] \big ) \in \mathbb { R } ^ { T } ,\tag{25}
$$

where $\psi ( \cdot )$ denotes a lightweight MLP and $T$ is the token dimension. We then pool a per-person token by confidence-weighted averaging over all tokens associated with the same person:

$$
\mathbf { t } _ { p } = \frac { \sum _ { i = 1 } ^ { N } \mathbb { I } [ \ell _ { i } = p ] w _ { i } \mathbf { d } _ { i } } { \sum _ { i = 1 } ^ { N } \mathbb { I } [ \ell _ { i } = p ] w _ { i } + \epsilon } , \qquad w _ { i } = h _ { i } ^ { \gamma } ,\tag{26}
$$

where $\gamma$ is a hyper-parameter controlling how strongly we emphasize high-confidence human tokens, and ϵ is a small constant. This design efectively suppresses noisy or floating points by down-weighting low-confidence tokens.

We additionally keep the accumulated weight

$$
s _ { p } = \sum _ { i = 1 } ^ { N } \mathbb { I } [ \ell _ { i } = p ] \ w _ { i } ,\tag{27}
$$

which serves as a validity indicator for whether person $p$ is reliably observed and can be used to mask invalid instances during training.

## 3.3.2 Feed-forward SMPL parameter regression

Given the pooled person token $\mathbf { t } _ { p } ,$ we regress SMPL parameters with a small MLP head:

$$
\begin{array} { r } { { \bf z } _ { p } = \rho ( { \bf t } _ { p } ) , } \end{array}\tag{28}
$$

followed by linear predictors for pose, shape, and translation:

$$
\begin{array} { r } { \hat { \pmb { \theta } } _ { p } = \mathbf { W } _ { \theta } \mathbf { z } _ { p } , \quad \hat { \pmb { \beta } } _ { p } = \mathbf { W } _ { \beta } \mathbf { z } _ { p } , \quad \hat { \mathbf { t } } _ { p } = \mathbf { W } _ { t } \mathbf { z } _ { p } , } \end{array}\tag{29}
$$

where $\hat { \pmb { \theta } } _ { p } \in \mathbb { R } ^ { 1 4 4 }$ denotes the 6D pose parameters, $\hat { \boldsymbol { \beta } } _ { p } \in \mathbb { R } ^ { 1 0 }$ denotes the shape coefi cients, and $\hat { \mathbf { t } } _ { p } \in \mathbb { R } ^ { 3 }$ denotes the global translation. Optionally, we also predict a global scale parameter $\hat { s } _ { p }$ (implemented as a positive scalar via sof tplus) to compensate for scale ambiguity in monocular or normalized coordinate systems:

$$
\hat { s } _ { p } = \mathrm { s o f t p l u s } ( \mathbf { W } _ { s } \mathbf { z } _ { p } ) + \epsilon .\tag{30}
$$

The regression outputs define an SMPL mesh for each person instance, enabling downstream tasks such as multi-view rendering, silhouette supervision, and instancelevel evaluation without iterative optimization.

## 3.3.3 Training with SMPL Regression

In the final stage, we train the SMPL regression head while continuing to optimize the Gaussian representation with photometric supervision. Given the pooled person tokens (Sec.3.3), the network predicts SMPL pose, shape, and translation parameters for each person instance.

The SMPL regression is supervised using a combination of 2D reprojection, 3D joint alignment, and silhouette consistency:

$$
\mathcal { L } _ { \mathrm { s m p l } } = \lambda _ { \mathrm { j 2 d } } \mathcal { L } _ { \mathrm { j 2 d } } + \lambda _ { \mathrm { j 3 d } } \mathcal { L } _ { \mathrm { j 3 d } } + \lambda _ { \mathrm { s e g } } \mathcal { L } _ { \mathrm { s e g } } .\tag{31}
$$

The 2D reprojection loss $\mathcal { L } _ { \mathrm { j 2 d } }$ penalizes the discrepancy between projected SMPL joints and ground-truth 2D keypoints across views. When available, the 3D joint loss $\mathcal { L } _ { \mathrm { j 3 d } }$ directly supervises predicted SMPL joints in 3D space. In addition, a silhouette loss $\mathcal { L } _ { \mathrm { s e g } }$ is applied by rendering the predicted SMPL meshes into each view and matching them with the corresponding human masks, which helps constrain body extent and resolve depth ambiguities under occlusions.

During this stage, photometric supervision $\mathcal { L } _ { \mathrm { r g b } }$ is retained to preserve geometric and appearance consistency of the Gaussian representation. This joint optimization enables accurate recovery of structured SMPL parameters from the instance-aware 3D feature space.

## 4 Experiments

We evaluate the proposed framework on challenging multi-view human reconstruction benchmarks under unconstrained conditions. Our experiments are designed to validate the efectiveness of maintaining a unified human-aware 3D feature space, as well as the contributions of reprojection-based enhancement, spatial contrastive learning, and 3D token–based SMPL regression.

## 4.1 Datasets

EgoHumans [42] is a multi-camera, multi-person dataset designed for evaluating human pose estimation and reconstruction in real-world environments. It consists of video sequences captured by multiple synchronized cameras, where two to four individuals perform everyday activities. The dataset covers a wide range of scene scales and layouts, including both indoor and outdoor environments, and exhibits frequent interperson occlusions, viewpoint variations, and camera placement irregularities. These characteristics make EgoHumans particularly suitable for evaluating the robustness of multi-view reconstruction methods under realistic conditions. Following common practice, we construct train/test splits at the scene level: for each scene, one sequence is randomly selected for testing, while the remaining sequences are used for training.

OcMotion [43] is a multi-camera dataset focusing on single-person motion under severe object-induced occlusions. It contains diverse everyday activities where the human body is partially or heavily occluded by surrounding objects, resulting in incomplete and view-dependent observations. OcMotion provides a complementary evaluation setting that emphasizes robustness to dynamic occlusion and missing visual evidence. For OcMotion, we follow the oficial dataset split as defined in the original benchmark.

## 4.2 Metrics

We evaluate both human reconstruction accuracy and camera estimation quality using standard metrics.

For human reconstruction, we report CA-MPJPE (Camera-aligned MPJPE, meters) first aligns the estimated camera parameters with the ground-truth camera parameters, and then computes MPJPE between the predicted and ground-truth 3D joints. We additionally report GA-MPJPE (Procrustes-aligned MPJPE, meters), which aligns the predicted multi-person configuration with the ground-truth multiperson layout as a whole, and evaluates the relative spatial relationships among all individuals. We further report PA-MPJPE (Group-aligned MPJPE, meters), which independently applies Procrustes alignment to each predicted person and evaluates per-person pose accuracy for articulated human reconstruction. These metrics together provide a more comprehensive and interpretable evaluation of both local pose accuracy and global multi-person spatial structure.

For camera evaluation, we report AE (angular error) and s-TE (scale-normalized translation error) to measure rotation and translation accuracy, respectively. We additionally report s-CCA@10 and AUC@10, which evaluate camera consistency within a fixed angular threshold. Lower AE and s-TE, and higher s-CCA@10 and AUC@10 indicate more accurate and consistent camera estimation.

Table 1: Quantitative comparison of multi-view human reconstruction accuracy on OcMotion.
<table><tr><td>Method</td><td>CA-MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>Multi-HMR [26]</td><td></td><td>0.06</td></tr><tr><td>DMMR [9]</td><td></td><td>0.04</td></tr><tr><td>HSfM [10]</td><td>0.35</td><td>0.05</td></tr><tr><td>Ours</td><td>0.19</td><td>0.04</td></tr></table>

## 4.3 Results

We compare our method with representative state-of-the-art multi-view reconstruction approaches on the EgoHumans [42] and OcMotion [43] dataset. In particular, we include comparison with HSfM [10], a recent framework for jointly reconstructing people, scene structure, and camera parameters from multi-view images. HSfM represents a strong baseline that integrates geometry and human modeling in a unified optimization framework.

Tables 1 and 2 report quantitative results on OcMotion and EgoHumans under diferent occlusion settings, respectively, while Table 3 presents camera pose estimation accuracy.

Table 1 reports results on OcMotion, which involves challenging single-person scenarios with severe object-induced occlusions. Our method achieves the best CA-MPJPE and competitive PA-MPJPE, outperforming Multi-HMR, DMMR and HSfM in global reconstruction accuracy. Table 2 further evaluates performance on the EgoHumans dataset under non-severely and severely occluded scenarios. In particular, we define severe occlusion as the presence of a subject occluded in more than 50% of the views. All remaining samples are grouped into the non-severely occluded subset. Across all settings, our method consistently achieves strong CA-MPJPE and GA-MPJPE, indicating robust multi-person spatial reconstruction under complex occlusions. Compared to Multi-HMR and DMMR, our method better preserves global spatial consistency, especially under heavy occlusion where cross-view correspondence becomes highly ambiguous. HSfM shows competitive inter-person structure modeling but sufers from larger absolute reconstruction errors. Overall, our method achieves consistently strong global reconstruction accuracy while maintaining competitive or superior local pose accuracy across both datasets and occlusion levels.

Table 2: Quantitative comparison of multi-view human reconstruction accuracy on EgoHumans under non-severely and severely occluded settings.
<table><tr><td rowspan="2">Method</td><td colspan="3">Non-severely occluded</td><td colspan="3">Severely occluded</td></tr><tr><td></td><td>CA-MPJPE↓ GA-MPJPE↓</td><td>PA-MPJPE↓</td><td>CA-MPJPE↓</td><td>GA-MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>Multi-HMR [26]</td><td>-</td><td>0.85</td><td>0.10</td><td>一</td><td>1.23</td><td>0.12</td></tr><tr><td>DMMR [9]</td><td></td><td>0.54</td><td>0.23</td><td></td><td>0.81</td><td>0.20</td></tr><tr><td>HSfM [10]</td><td>0.84</td><td>0.34</td><td>0.08</td><td>1.18</td><td>0.53</td><td>0.13</td></tr><tr><td>Ours</td><td>0.80</td><td>0.21</td><td>0.07</td><td>0.89</td><td>0.28</td><td>0.14</td></tr></table>

Table 3: Quantitative comparison of Camera pose accuracy on EgoHumans and OcMotion.
<table><tr><td rowspan="2">Method</td><td colspan="5">EgoHumans</td><td colspan="3">OcMotion</td></tr><tr><td>AE↓</td><td>s-TE↓</td><td>s-CCA@10↑</td><td>AUC@10↑</td><td>AE↓</td><td>s-TE↓</td><td>s-CCA@10↑</td><td>AUC@10↑</td></tr><tr><td>UnCaliPose[44]</td><td>130.68</td><td>4.38</td><td>0.02</td><td>0.23</td><td>154.43</td><td>4.31</td><td>0.01</td><td>0.19</td></tr><tr><td>HSfM[10]</td><td>29.58</td><td>0.77</td><td>0.17</td><td>0.62</td><td>0.97</td><td>0.13</td><td>0.50</td><td>0.80</td></tr><tr><td>Ours</td><td>0.60</td><td>0.03</td><td>0.95</td><td>0.94</td><td>0.36</td><td>0.04</td><td>0.99</td><td>0.94</td></tr></table>

As shown in Table 3, our method consistently achieves substantially lower camera pose errors across all metrics on both EgoHumans and OcMotion. In particular, the improvements in AE and s-TE indicate accurate estimation of absolute camera orientation and translation, while the high s-CCA@10 and AUC@10 scores demonstrate strong cross-view consistency. These results suggest that maintaining a unified 3D feature space enables reliable joint reasoning over camera geometry and scene structure without explicit calibration.

Figure 4 presents qualitative reprojection comparisons. HSfM tends to prioritize relative spatial relationships between individuals, which often leads to inaccurate reprojection results and visible misalignments in the image plane. Multi-HMR produces accurate reprojections in the input views but lacks explicit 3D reasoning, resulting in inconsistent reconstructions across diferent viewpoints. In comparison, our method produces accurate and consistent reprojections across views, reflecting improved multi-view coherence and pose estimation.

Finally, we visualize the reconstructed 3D scene together with the recovered SMPL bodies in Figure 5. The results show that our framework is able to reconstruct both scene geometry and human bodies in a coherent and physically plausible manner, further validating the efectiveness of maintaining a unified human-aware 3D feature space.

![](images/347f01a1a92ef340084be8900e643cd61337817d490cce641ecf1483d4a164b7.jpg)  
Input

![](images/b1a5e15625686e1e9633c9c2881aaf044e7a94cb7545200056516cd3abbbfc6c.jpg)  
HSfM

![](images/08a755ae9d3d3a5546b7bde60353fa33686736248c5248cac6e8be8dbdf5d53f.jpg)  
Multi-HMR

![](images/2b6ba3d54f5800ee1588eba350ef2ea12d6e467d0a94a9823bc0dbc84a5146d2.jpg)  
Ours  
Fig. 4: Qualitative Comparison of Multi-View Human Reconstruction.

![](images/cf9538f4fd6aaa4a7259aca8ba2b978acbef5f9c69e6ecf95893db54110917ce.jpg)  
(a) EgoHumans Scene 1

![](images/c7358a313fd8b241f8bab908a4c0b26699ce65ae11e9fdd18aff00cf8f32fc83.jpg)  
(b) EgoHumans Scene 2

![](images/d16bef42325da386030db7c37a9b4fb61aa7908c7108d29c30ece27f8cf1bfce.jpg)  
(c) OcMotion Scene 1

![](images/4086734e6c360d9681ac7573f0fe9f99e092cc95aa31fbd7a7d76636bc7c6e68.jpg)  
(d) OcMotion Scene 2  
Fig. 5: Qualitative results of the proposed method on EgoHumans and OcMotion. We visualize the reconstructed 3D scene together with the recovered SMPL bodies from multiple views. The results demonstrate that our method produces spatially coherent human reconstructions.

Table 4: Runtime breakdown (s) of the proposed method under diferent numbers of input frames.
<table><tr><td>Input Frames</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td></tr><tr><td>3D Scene Reconstruction</td><td>0.099</td><td>0.131</td><td>0.219</td><td>0.382</td><td>0.812</td><td>1.886</td></tr><tr><td>Instance-level Human Segmentation</td><td>0.064</td><td>0.184</td><td>0.471</td><td>0.501</td><td>0.501</td><td>0.488</td></tr><tr><td>SMPL Regression</td><td>0.046</td><td>0.046</td><td>0.048</td><td>0.052</td><td>0.084</td><td>0.180</td></tr><tr><td>Total</td><td>0.209</td><td>0.361</td><td>0.729</td><td>0.936</td><td>1.397</td><td>2.554</td></tr></table>

## 4.4 Runtime and Memory Analysis

We further analyze the computational eficiency of the proposed framework under diferent numbers of input views, and compare it with a representative optimizationbased baseline HSfM.

Since all components in our framework are implemented in a feed-forward regression manner, the overall computational cost is primarily determined by the number of input frames rather than iterative optimization or test-time refinement. This makes the proposed method fundamentally diferent from optimization-based reconstruction pipelines.

Table 4 reports the runtime breakdown of diferent stages in our pipeline, including 3D scene reconstruction, instance-level human segmentation, and SMPL regression. We observe that the 3D scene reconstruction stage scales with the number of input views due to additional depth estimation and feature lifting from VGGT. The instancelevel human segmentation stage depends on the number of reconstructed humanrelated 3D points and gradually stabilizes as spatial density converges. The SMPL regression module remains lightweight and relatively stable across diferent input sizes.

Overall, although runtime increases with more input views, the growth is sub-linear rather than strictly proportional, indicating good scalability in multi-view settings. In contrast, optimization-based methods such as HSfM require iterative refinement, leading to significantly higher computational cost as shown in Table 5.

We further report a comparison of runtime and GPU memory consumption between our method and HSfM. As shown in Table 5 , our approach achieves substantially lower runtime while maintaining comparable or higher memory eficiency under increasing numbers of input views. This demonstrates that the proposed framework provides a favorable trade-of between reconstruction accuracy and computational eficiency in multi-view scenarios.

## 4.5 Cross-view Association Evaluation

To further evaluate the quality of cross-view instance association, we further apply the AIDP [45] metrics for cross-view association evaluation. AIDP evaluates the precision of subject association across views by computing pairwise identity matching accuracy over all view pairs and averaging the results. This metric directly reflects the consistency of instance-level correspondence in multi-view settings.

Table 5: Runtime and peak GPU memory comparison with HSfM under diferent numbers of input frames.
<table><tr><td>Method</td><td>Metric</td><td>1</td><td>2</td><td>4</td><td>8</td></tr><tr><td>HSfM</td><td>Time (s)</td><td>170</td><td>177</td><td>199</td><td>220</td></tr><tr><td>HSfM</td><td>Mem. (GB)</td><td>4.62</td><td>5.13</td><td>6.29</td><td>10.79</td></tr><tr><td>Ours</td><td>Time (s)</td><td>0.209</td><td>0.361</td><td>0.729</td><td>0.936</td></tr><tr><td>Ours</td><td>Mem. (GB)</td><td>8.26</td><td>8.72</td><td>11.20</td><td>15.82</td></tr></table>

Table 6: Cross-view instance association evaluation using AIDP metric. Higher is better.
<table><tr><td>Method</td><td>AIDP ↑</td></tr><tr><td>HSfM [10]</td><td>75.68</td></tr><tr><td>VGGT-based Pose-Aware ReID</td><td>82.02</td></tr><tr><td>Ours</td><td>98.54</td></tr></table>

We first compare our method with HSfM under this evaluation protocol. In addition, to further examine whether geometric cues alone are suficient for reliable association, we construct a VGGT-based Pose-Aware ReID baseline. Specifically, given detected human bounding boxes, we extract keypoint predictions and confidence scores from VGGT, and perform confidence-weighted aggregation to obtain compact pose-centric descriptors. Cross-view identity association is then performed via nearest-neighbor matching based on Euclidean distance.

As shown in Table 6 , our method achieves the highest AIDP score, outperforming both HSfM and the VGGT-based baseline. This indicates that geometry- or posebased heuristics can provide reasonable cues for association, but remain limited under challenging multi-view conditions. In contrast, our spatial contrastive learning explicitly enforces instance-level consistency in the unified 3D feature space, leading to more robust cross-view identity alignment.

## 4.6 Ablation Study

## 4.6.1 Architecture Ablation

We conduct ablation studies on EgoHumans to analyze the contributions of key components in the proposed framework. In particular, we investigate how diferent design choices afect multi-view human perception, instance consistency, and SMPL regression from the unified 3D feature space.

Specifically, we consider the following variants. w/o unified 3D feature space removes explicit 3D perception and directly regresses human bodies from per-view 2D feature maps. w/o reprojection enhancement disables reprojection-based pose feature sampling and regresses SMPL parameters solely from the initial 3D feature space containing geometric cues. w/o SCL removes spatial contrastive learning and aggregates features based on 2D instance segmentation without enforcing instance consistency in 3D.

Table 7: Ablation study on key components of the proposed framework on EgoHumans.
<table><tr><td>Method</td><td>CA-MPJPE↓</td><td>GA-MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>w/o unified 3D feature space</td><td>3.12</td><td>3.28</td><td>2.53</td></tr><tr><td>w/o reprojection enhancement</td><td>1.28</td><td>1.42</td><td>1.74</td></tr><tr><td>w/o SCL</td><td>1.94</td><td>0.49</td><td>0.37</td></tr><tr><td>Ours</td><td>0.95</td><td>0.30</td><td>0.18</td></tr></table>

Table 7 summarizes the ablation results. Removing the unified 3D feature space leads to a significant performance drop. Without explicit 3D perception, the model relies solely on view-dependent 2D features, which are highly sensitive to occlusion and viewpoint variation. As a result, it becomes dificult to associate observations of the same individual across views, leading to degraded reconstruction accuracy.

Disabling reprojection-based enhancement also degrades performance. In this setting, SMPL parameters are regressed only from the initial 3D feature space that mainly encodes geometric information. Without injecting pose-related image features via reprojection, optimization becomes more dificult and convergence is less stable, resulting in inferior reconstruction quality.

Removing spatial contrastive learning further impacts performance. When instance consistency is enforced only through 2D instance segmentation, feature aggregation becomes vulnerable to segmentation noise and view-dependent errors. This leads to contaminated 3D features and reduced robustness, especially in the presence of occlusion and clutter.

## 4.6.2 Robustness to Geometry Initialization

To further evaluate the robustness of our framework to geometry initialization, we replace the VGGT backbone with AnySplat [46] , a representative geometry-aware reconstruction method that provides depth and camera priors with Gaussian Splattingbased refinement.

Both VGGT and AnySplat are geometry-aware reconstruction pipelines that produce compatible initializations of depth, features, and camera parameters, making them suitable for evaluating the sensitivity of our framework to diferent geometry backbones. Unlike VGGT, which directly predicts geometry and feature representations, AnySplat introduces an additional rendering-based refinement stage, resulting in slightly diferent camera and point cloud quality. Nevertheless, both methods share a similar underlying geometry foundation, allowing a fair comparison of backbone robustness.

As shown in Table 8 , the performance remains highly consistent under both initialization strategies, with only marginal diferences across all evaluation metrics. This indicates that our method is largely robust to the choice of geometry initialization.

Table 8: Robustness to geometry initialization. We compare VGGT-based initialization with an alternative AnySplat-based initialization.
<table><tr><td>Geometry Initialization</td><td>CA-MPJPE↓</td><td>GA-MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>w/ AnySplat [46]</td><td>1.02</td><td>0.37</td><td>0.19</td></tr><tr><td>VGGT (default)</td><td>0.95</td><td>0.30</td><td>0.18</td></tr></table>

The results further suggest that our method is robust to the choice of geometry initialization, and benefits primarily from the unified 3D feature representation and spatial contrastive learning.

## 4.6.3 Analysis of 3D Human Awareness

The limitations of relying solely on 2D instance segmentation further motivate a deeper analysis of human perception in 3D. Although 2D segmentation-based aggregation can achieve high coverage and include most human-related features, it exhibits structural weaknesses when applied to multi-view 3D reconstruction.

In practice, segmentation errors near object boundaries and occlusion regions are common. When such masks are lifted into 3D, these inaccuracies lead to enlarged selection regions and ray-like leakage into the background, contaminating the aggregated 3D features. Moreover, 2D instance segmentation does not naturally provide consistent instance identities across views. Associating masks belonging to the same person from diferent viewpoints requires additional cross-view matching, which is highly sensitive to occlusion, truncation, and segmentation noise. Inconsistent instance assignments across views further degrade feature aggregation and undermine multi-view coherence.

These observations indicate that accurate human perception in multi-view settings cannot be reliably achieved by inheriting instance information from 2D segmentation alone. Instead, human awareness and instance consistency should be learned directly in the shared 3D feature space, where multi-view observations can be aggregated and regularized in a geometry-aware manner. In the following, we further validate this design choice.

On a single-person setting with per-view masks and ground-truth SMPL models, we compare the proposed learned 3D human confidence with a non-learned baseline that fuses 2D masks into 3D via visibility-aware averaging. Ground-truth human labels for 3D tokens are derived based on their distance to the SMPL surface.

Table 9 reports token-level Precision, Recall, F1 score, and PR-AUC. The 2D mask fusion baseline achieves very high recall but sufers from extremely low precision, indi cating severe background leakage. In contrast, the learned 3D confidence substantially improves precision and F1 score while maintaining competitive recall, and achieves a significantly higher PR-AUC, demonstrating better global separability and calibration.

Figure 6 provides qualitative evidence. Directly lifting 2D masks into 3D introduces ray-like artifacts and background noise around occlusion boundaries, whereas the learned 3D confidence yields a spatially compact and coherent human region. These results indicate that human awareness is not merely inherited from 2D segmentation, but emerges as a spatially consistent property learned in the unified 3D feature space, forming a reliable foundation for subsequent human-guided modules.

Table 9: Comparison between learned 3D human confidence and 2D mask fusion on the single-person dataset. Evaluation is performed at the token level using ground-truth SMPL supervision.
<table><tr><td>Method</td><td>Precision ↑</td><td>Recall ↑</td><td>F1 ↑</td><td>PR-AUC ↑</td></tr><tr><td>2D mask fusion</td><td>0.420</td><td>0.989</td><td>0.590</td><td>0.006</td></tr><tr><td>Ours (3D learned)</td><td>0.886</td><td>0.825</td><td>0.855</td><td>0.863</td></tr></table>

![](images/eab9d0692c4222b6c20dc3399ab5df85fe35353991c7cef1119edd4d8a8a64b5.jpg)  
(a) GT

![](images/f08a4ed19ab16e943bb43fdf6ef6a37d3572c6234cdeaaac01ac96f142dc732b.jpg)  
(b) 2D mask fusion

![](images/4fe41dd19b41d2ca16013950a51d7405a36c86a52f7749e9393caf9d79df6c49.jpg)  
(c) Ours  
Fig. 6: Qualitative comparison of 3D human confidence fields. While 2D mask fusion covers most human regions, it introduces substantial false positives in background and occluded areas. In contrast, the learned 3D confidence produces a spatially compact and clean human region with significantly reduced background noise.

## 5 Conclusion

In this paper, we presented a novel feed-forward framework for multi-view multi-person 3D reconstruction. Departing from conventional pipelines that rely on view-dependent 2D reasoning or tightly coupled optimization, our method explicitly maintains a shared 3D representation where geometry, semantics, and human identity are jointly encoded and progressively refined. By initializing the 3D space with a geometry-aware prior and 3D Gaussian representation, the framework establishes stable spatial anchors under unconstrained camera settings. Building upon this foundation, spatial contrastive learning enhances the discriminability and consistency of the 3D features by leveraging multi-view semantic cues, enabling robust multi-person separation and cross-view association in the presence of occlusion and partial visibility. Structured human body models are then recovered in a feed-forward manner directly from instance-aware 3D tokens, avoiding iterative fitting and allowing camera parameters and human representations to be jointly optimized. Extensive experiments on challenging multi-view benchmarks demonstrate that the proposed approach achieves competitive or superior human reconstruction accuracy while significantly improving camera estimation quality.

While the present work focuses on static multi-view reconstruction, the proposed framework can be naturally extended to 4D dynamic settings by incorporating temporal association of 3D tokens across sequential frames. In particular, the instance-aware 3D representation provides a structured and identity-consistent basis that can naturally support temporal correspondence, enabling potential extensions to articulated motion and moving-camera scenarios. We further note that the current formulation does not explicitly model temporal evolution, and therefore future work may explore temporal update mechanisms such as token propagation, merging, and splitting within the same 3D Gaussian representation.

We also acknowledge that our framework relies on VGGT as a geometry-aware foundation model for initializing depth, features, and camera parameters. While VGGT provides strong generalization for constructing the initial 3D feature space, its performance is not perfect and may be limited in cases of large viewpoint gaps, low-texture regions, or challenging depth discontinuities, which can afect downstream reconstruction quality. Nevertheless, our method is not tightly coupled to VGGT itself and operates on the constructed 3D feature representation. Therefore, it can be naturally integrated with alternative or improved geometry foundation models in future work.

We believe that extending the framework toward full 4D reconstruction and improving robustness under more challenging geometry initialization remain important and promising directions for future research.

Funding This work was supported in part by National Key R&D Program of China (2023YFC3082100), the Open Project Program of State Key Laboratory of Virtual Reality Technology and Systems, Beihang University (No.VRLAB2026C04), and Science Fund for Distinguished Young Scholars of Tianjin (No.22JCJQJC00040).

Data availability The datasets used in this study are publicly available. Detailed descriptions and corresponding references are provided in Section 4.1.

## References

[1] Joo, H., Liu, H., Tan, L., Gui, L., Nabbe, B., Matthews, I., Kanade, T., Nobuhara, S., Sheikh, Y.: Panoptic studio: A massively multiview system for social motion capture. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 3334–3342 (2015)

[2] Ionescu, C., Papava, D., Olaru, V., Sminchisescu, C.: Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence 36(7), 1325–1339 (2013)

[3] Dong, J., Fang, Q., Jiang, W., Yang, Y., Huang, Q., Bao, H., Zhou, X.: Fast and robust multi-person 3d pose estimation and tracking from multiple views. IEEE transactions on pattern analysis and machine intelligence 44(10), 6981– 6992 (2021)

[4] Tu, H., Wang, C., Zeng, W.: Voxelpose: Towards multi-camera 3d human pose estimation in wild environment. In: European Conference on Computer Vision, pp. 197–212 (2020). Springer

[5] Zhang, J., Cai, Y., Yan, S., Feng, J., et al.: Direct multi-view multi-person 3d pose estimation. Advances in Neural Information Processing Systems 34, 13153–13164 (2021)

[6] Huang, B., Ju, J., Shu, Y., Wang, Y.: Simultaneously recovering multi-person meshes and multi-view cameras with human semantics. IEEE Transactions on Circuits and Systems for Video Technology 34(6), 4229–4242 (2023). Authors: Buzhen Huang; Jingyi Ju; Yuan Shu; Yangang Wang

[7] Huang, C., Jiang, S., Li, Y., Zhang, Z., Traish, J., Deng, C., Ferguson, S., Da Xu, R.Y.: End-to-end dynamic matching network for multi-view multi-person 3d pose estimation. In: European Conference on Computer Vision, pp. 477–493 (2020). Springer

[8] Yu, Z., Zhang, L., Xu, Y., Tang, C., Tran, L., Keskin, C., Park, H.S.: Multiview human body reconstruction from uncalibrated cameras. Advances in Neural Information Processing Systems 35, 7879–7891 (2022)

[9] Huang, B., Ju, J., Shu, Y., Wang, Y.: Simultaneously recovering multi-person meshes and multi-view cameras with human semantics. IEEE Transactions on Circuits and Systems for Video Technology 34(6), 4229–4242 (2024)

[10] M¨uller, L., Choi, H., Zhang, A., Yi, B., Malik, J., Kanazawa, A.: Reconstructing people, places, and cameras. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21948–21958 (2025)

[11] Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294–5306 (2025)

[12] Kerbl, B., Kopanas, G., Leimk¨uhler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)

[13] Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J.: Smpl: A skinned multi-person linear model. In: Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pp. 851–866 (2023)

[14] Xu, Y., Kitani, K.: Multi-view multi-person 3d pose estimation with uncalibrated camera networks. In: BMVC (2022)

[15] Li, Y.-J., Xu, Y., Khirodkar, R., Park, J., Kitani, K.: Multi-person 3d pose estimation from multi-view uncalibrated depth cameras. arXiv preprint arXiv:2401.15616 (2024)

[16] Javerliat, C., Raimbaud, P., Lavou´e, G.: Kineo: Calibration-free metric motion capture from sparse rgb cameras. arXiv preprint arXiv:2510.24464 (2025)

[17] Yin, J.O., Li, T., Wang, J., Zhang, Y., Yuille, A.: Easyret3d: Uncalibrated multiview multi-human 3d reconstruction and tracking. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 3128–3137 (2025). IEEE

[18] Chen, X., Wang, L., Wang, Q., Tang, J.: 3d human pose estimation and tracking based on feedback mechanism. In: 2024 7th International Conference on Intelligent Robotics and Control Engineering (IRCE), pp. 292–296 (2024). IEEE

[19] Luo, J., Xie, S., Quan, T., Ren, X., Miao, Y.: Voxel-based multi-person multi-view 3d pose estimation in operating room. Applied Sciences 15(16), 9007 (2025)

[20] Chharia, A., Gou, W., Dong, H.: Mv-ssm: Multi-view state space modeling for 3d human pose estimation. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11590–11599 (2025)

[21] Furukawa, Y., Ponce, J.: Accurate, dense, and robust multiview stereopsis. IEEE transactions on pattern analysis and machine intelligence 32(8), 1362–1376 (2009)

[22] Hu, S., Hong, F., Pan, L., Mei, H., Yang, L., Liu, Z.: Sherf: Generalizable human nerf from a single image. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9352–9364 (2023)

[23] Starck, J., Hilton, A.: Surface capture for performance-based animation. IEEE computer graphics and applications 27(3), 21–31 (2007)

[24] Bogo, F., Kanazawa, A., Lassner, C., Gehler, P., Romero, J., Black, M.J.: Keep it smpl: Automatic estimation of 3d human pose and shape from a single image. In: European Conference on Computer Vision, pp. 561–578 (2016). Springer

[25] Li, Z., Oskarsson, M., Heyden, A.: 3d human pose and shape estimation through collaborative learning and multi-view model-fitting. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1888–1897 (2021)

[26] Baradel, F., Armando, M., Galaaoui, S., Br´egier, R., Weinzaepfel, P., Rogez, G., Lucas, T.: Multi-hmr: Multi-person whole-body human mesh recovery in a single shot. In: European Conference on Computer Vision, pp. 202–218 (2024). Springer

[27] Elflein, S., Zhou, Q., Leal-Taix´e, L.: Light3r-sfm: Towards feed-forward structurefrom-motion. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 16774–16784 (2025)

[28] Yang, J., Sax, A., Liang, K.J., Henaf, M., Tang, H., Cao, A., Chai, J., Meier, F., Feiszli, M.: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21924–21935 (2025)

[29] Cabon, Y., Stofl, L., Antsfeld, L., Csurka, G., Chidlovskii, B., Revaud, J., Leroy, V.: Must3r: Multi-view network for stereo 3d reconstruction. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1050–1060 (2025)

[30] Jang, W., Weinzaepfel, P., Leroy, V., Agapito, L., Revaud, J.: Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1071–1081 (2025)

[31] Li, W., Liu, S., Qiao, P., Dou, Y.: Mono3r: Exploiting monocular cues for geometric 3d reconstruction. In: Proceedings of the 33rd ACM International Conference on Multimedia, pp. 11081–11090 (2025)

[32] Liu, S., Li, W., Qiao, P., Dou, Y.: Regist3r: Incremental registration with stereo foundation model. In: Proceedings of the 33rd ACM International Conference on Multimedia, pp. 4484–4493 (2025)

[33] Zhao, Y., Wang, T.Y., Raj, B., Xu, M., Yang, J., Huang, C.-H.P.: Synergistic global-space camera and human reconstruction from videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1216–1226 (2024)

[34] Liu, Z., Lin, J., Wu, W., Zhou, B.: Joint optimization for 4d human-scene reconstruction in the wild. arXiv preprint arXiv:2501.02158 (2025)

[35] Chen, Y., Chen, X., Xue, Y., Chen, A., Xiu, Y., Pons-Moll, G.: Human3r: Everyone everywhere all at once. arXiv preprint arXiv:2510.06219 (2025)

[36] Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10510–10522 (2025)

[37] Rojas, S., Armando, M., Ghanem, B., Weinzaepfel, P., Leroy, V., Rogez, G.: Hamst3r: Human-aware multi-view stereo 3d reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5027– 5037 (2025). Authors: Sara Rojas; Matthieu Armando; Bernard Ghanem; Philippe Weinzaepfel; Vincent Leroy; Gr´egory Rogez

[38] Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r. In: European Conference on Computer Vision (ECCV), pp. 71–91 (2024)

[39] Zhang, Y., Wang, C., Wang, X., Liu, W., Zeng, W.: Voxeltrack: Multi-person 3d human pose estimation and tracking in the wild. IEEE Transactions on Pattern

[40] Sun, Y., Bao, Q., Liu, W., Fu, Y., Black, M.J., Mei, T.: Monocular, one-stage, regression of multiple 3d people. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11179–11188 (2021)

[41] Sim´eoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khali dov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

[42] Khirodkar, R., Bansal, A., Ma, L., Newcombe, R., Vo, M., Kitani, K.: Egohumans: An ego-centric 3d multi-human benchmark. In: ICCV (2023)

[43] Huang, B., Shu, Y., Ju, J., Wang, Y.: Occluded human body capture with self-supervised spatial-temporal motion prior. arXiv preprint arXiv:2207.05375 (2022)

[44] Xu, Y., Kitani, K.: Multi-view multi-person 3d pose estimation with uncalibrated camera networks. In: British Machine Vision Conference (BMVC) (2022)

[45] Gan, Y., Han, R., Yin, L., Feng, W., Wang, S.: Self-supervised multi-view multihuman association and tracking. In: Proceedings of the 29th ACM International Conference on Multimedia, pp. 282–290 (2021)

[46] Jiang, L., Mao, Y., Xu, L., Lu, T., Ren, K., Jin, Y., Xu, X., Yu, M., Pang, J., Zhao, F., et al.: Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on Graphics (TOG) 44(6), 1–16 (2025)