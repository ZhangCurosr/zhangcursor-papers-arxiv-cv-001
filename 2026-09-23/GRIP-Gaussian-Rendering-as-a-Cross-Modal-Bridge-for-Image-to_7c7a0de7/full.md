# GRIP: Gaussian Rendering as a Cross-Modal Bridge for Image-to-Point Cloud Registration

Karim Slimani<sup>1[0009−0007−4791−7173]</sup>, Catherine Achard<sup>1[0000−0002−5790−0830]</sup>,

Eric Marchand<sup>2[0000−0001−7096−5236]</sup>, and Brahim Tamadazte<sup>1[0000−0002−4668−3092]</sup>

1 ISIR, Sorbonne Univ., CNRS , INSERM, Paris, France Corresponding Author: karim.slimani@isir.upmc.fr <sup>2</sup> Univ Rennes, Inria, CNRS, Irisa, Rennes, France

Abstract. This paper introduces GRIP, a pose-conditioned refinement framework for pixel-to-point matching and 2D to 3D registration. Given an initial coarse pose estimate, GRIP addresses the structural mismatch between grid based image descriptors and unordered point cloud descriptors by softly rendering learned 3D point features onto the image grid through Gaussian feature splatting. The rendered point derived feature map is then fused with image features by a pixel aligned transformer, enabling visual semantic and geometric cues to interact in a shared 2D representation. The refined features are decoded and propagated to finer resolutions for dense correspondence estimation and final pose refinement. Experiments on RGB D Scenes V2 and 7 Scenes demonstrate state of the art inlier ratio and competitive registration recall, with stronger performance under stricter evaluation thresholds.

Keywords: 2D-3D Registration · Pixel-to-point Matching · Gaussian Feature Splatting

## 1 Introduction

Image-to-point cloud registration is a fundamental task in computer graphics, computer vision, and robotics, with applications including robotic navigation [38], 3D reconstruction [4], and augmented reality [22, 25]. It aims to estimate the rigid transformation that registers a 3D point cloud with a partially overlapping 2D camera image.

This problem is highly challenging due to the data’s heterogeneous nature. Captured through diferent modalities, images primarily provide visual cues such as color, texture, and semantics, whereas point clouds provide geometric cues such as spatial structure and local properties. Traditional 2D-3D registration methods typically follow a detect-then-match pipeline composed of three main steps: repeatable keypoints detection, pixel-to-point correspondence estimation, and transformation estimation. In these methods, keypoint detection and correspondence search rely on visual saliency and local geometric structures, while robust estimators, most notably PnP-RANSAC [10, 18], are used to recover the final rigid pose. For a long time, such methods dominated cross-modal registration. However, challenging conditions such as sparse data, limited overlap, and large viewpoint changes make repeatable keypoint detection across these two diferent domains dificult, thereby degrading performance. To overcome these limitations, 2D3D-MATR [19] introduced a detection-free pipeline. Inspired by state-of-the-art methods in image matching [35] and 3D point cloud registration [30, 43], 2D3D-MATR divides the matching process into two complementary levels: coarse-level image patch and point-node matching and dense pixelto-point matching. This paradigm inspired recent learning methods [6,19,27,41] which adopted the same coarse-to-fine matching strategy coupled with PnP-RANSAC for scene registration [11, 17].

Despite these advances, detection-free 2D-3D matching still relies largely on direct feature comparison between two heterogeneous representations. In these methods, image descriptors are typically extracted using Feature Pyramid Network (FPN) [21] and ResNet [12] backbones, and therefore lie on a regular 2D grid that primarily encodes visual appearance and semantic context. In contrast, point cloud descriptors are predicted using backbones such as KPFCNN [36], which encode geometric attributes in an unorganized 3D domain. Although existing methods [19, 27, 41] adopt transformers to model cross-modal interactions at the coarsest level, the two modalities remain structurally misaligned during matching: image descriptors are arranged on a dense 2D grid, whereas point cloud descriptors are indexed over an unordered 3D points. As a result, these methods enhance contextual communication but do not explicitly unify the representation space in which cross-modal matching is performed. Moreover, since transformers are introduced only at the coarse level [19, 41] and outside the backbone, fine-level features from the two modalities fail to capture cross-modal cues. This modality gap makes dense correspondence estimation challenging, especially under partial overlap, occlusions, and imperfect coarse alignment. Furthermore, once an initial pose is available, it provides a valuable geometric cue that can directly relate 3D points to image locations. However, existing coarseto-fine pipelines do not fully exploit this pose to construct an image-aligned point cloud representation before refining the matching process.

In this paper, we present GRIP, a pose conditioned refinement framework for image to point cloud registration based on feature splatting [15,39]. Starting from an initial coarse pose estimate, GRIP renders learned 3D point features into the image plane, producing an image aligned representation of the point cloud. This representation serves as a cross modal bridge, enabling pixel aligned interaction between visual features and 3D derived geometric features prior to final correspondence estimation. GRIP therefore changes the interaction space before final matching: instead of refining descriptors only in their original heterogeneous domains, it first renders learned 3D features onto the image plane and then performs cross modal interaction within a shared pixel aligned representation. This representation is subsequently used to refine dense correspondences and recover the final pose, as detailed in Section 3.

In a nutshell, the main contributions of this paper are as follows:

– We introduce a Point-to-Pixel feature rendering module that converts unordered point-cloud descriptors into an image-aligned feature map.

– We propose a Pixel-Aligned Interaction Transformer that benefits from this intermediate representation and bidirectionally fuses geometric and visualsemantic cues on a shared 2D grid.

– We design a hierarchical propagation strategy that injects the fused crossmodal representation into both image and point-cloud decoders, improving coarse and dense 2D-3D matching and pose refinement.

– We achieve state-of-the-art matching inlier ratio and competitive registration recall, with stronger performance under stricter thresholds.

## 2 Related works

Recent 2D-3D registration methods draw inspiration from both image matching and 3D point cloud registration. Image matching evolved from handcrafted descriptors [24, 32] to learning-based methods [7, 8], with transformer-based architectures such as SuperGlue [33] improving correspondence estimation through global context aggregation. Similarly, point cloud registration progressed from ICP and its derivatives [2, 42, 44] to deep learning approaches such as DCP [40], before recent coarse-to-fine transformer frameworks became the dominant paradigm [30, 35, 43].

Building upon these advances, 2D-3D registration has evolved from detectand-match pipelines based on handcrafted features [20] to recent detection-free learning frameworks [19, 41]. Inspired by learning-free predecessors [34], which extract local features using SIFT [24], 2D3DMatchNet [9] introduces a learningbased approach relying on PointNet [29] and CNNs to jointly predict descriptors for 2D and 3D keypoints. Moving beyond handcrafted detectors, P2-Net [37] proposes a dual-fully-convolutional framework that maps 2D and 3D inputs into a shared latent space to jointly detect and describe keypoints. However, as highlighted by 2D3D-MATR [19], existing inter-modality methods either rely on scene-specific coordinate regression, thereby limiting generalization to novel scenes, or follow detect-then-match pipelines that are hindered by unstable crossmodal keypoint repeatability and low inlier ratios.

2D3D-MATR [19] introduced a detection-free coarse-to-fine framework for 2D-3D registration. Multi-resolution image and point cloud features interact through a transformer [1] at the coarsest level to establish correspondences that subsequently guide dense pixel-to-point matching. The good performance of 2D3D-MATR on challenging indoor benchmarks such as RGB-D Scenes V2 [17] and 7-Scenes [11] inspired several subsequent methods to adopt a similar coarseto-fine design. In particular, Dif-Reg [41] extends 2D3D-MATR with a difusionbased matching framework that preserves the original pipeline and employs a transformer-based denoising module to iteratively refine the coarse matching matrix. Dif<sup>2</sup>I2P [27] further leverages a depth-conditioned difusion model to distill cross-modal knowledge and bridge the gap between 2D and 3D features while maintaining the same coarse-to-fine paradigm. More recently, $\mathrm { R ^ { 2 3 } N e t }$ [6] addressed the issue of non-overlapping and low-quality matching regions inherent to coarse-to-fine 2D-3D registration by first identifying informative regions in both the image and the point cloud through a reinforcement-learning-based High-Value Zone Reinforced Selection module.

In contrast to these approaches, which mainly refine matching scores or crossmodal descriptors within the original feature domains, GRIP explicitly modifies the representation space by rendering 3D descriptors onto the image plane prior to cross-modal interaction.

## 3 Method

## 3.1 Problem Statement

Let I be an RGB image and $\mathbf { P } = \{ \mathbf { p } _ { 1 } , \dots , \mathbf { p } _ { N } \} \subset \mathbb { R } ^ { 3 }$ be a point cloud. We assume that I and P are partially overlapping, such that a set of valid 2D-3D correspondences exists: $\mathcal { C } ^ { g t } = \{ ( \mathbf { p } _ { k } , \mathbf { u } _ { k } ) \ | \ \mathbf { p } _ { k } \in \mathbb { R } ^ { 3 } , \mathbf { u } _ { k } \in \mathbb { R } ^ { 2 } \}$ where $\mathbf { u } _ { k }$ are the 2D pixel coordinates. The goal of 2D-3D registration is to estimate the rigid transformation $\hat { \mathbf { T } } ( \hat { \mathbf { R } } , \hat { \mathbf { t } } ) \in \hat { \mathrm { S E } } ( 3 )$ , that minimizes the reprojection error between these correspondences:

$$
\hat { \mathbf { R } } , \hat { \mathbf { t } } = \underset { \mathbf { R } , \mathbf { t } } { \arg \operatorname* { m i n } } \ \sum _ { ( \mathbf { p } _ { k } , \mathbf { u } _ { k } ) \in \mathcal { C } ^ { g t } } \left\| \mathbf { u } _ { k } - \boldsymbol { \pi } \big ( \mathbf { R } \mathbf { p } _ { k } + \mathbf { t } ; \mathbf { K } \big ) \right\| _ { 2 } ^ { 2 }\tag{1}
$$

where $\pi ( { \bf \partial } \cdot { \bf \partial } ; { \bf K } ) : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ denotes the perspective projection from 3D camera coordinates to 2D image coordinates using the camera intrinsic matrix K.

The proposed method follows a two-stage strategy. Stage 1 estimates an initial transformation 3.2, while the main contribution lies in Stage 2, which performs pose-conditioned rendering-based refinement. Stage 2 comprises Pointto-Pixel Feature Rendering, Pixel-Aligned Interaction, Hierarchical Cross-Modal Feature Propagation, and final correspondence and pose estimation, as detailed in 3.3.

## 3.2 Coarse Registration (Initialization)

Given an image $\mathbf { I } \in \mathbb { R } ^ { H \times W \times 3 }$ and a point cloud $\mathbf { P } \in \mathbb { R } ^ { N \times 3 }$ , the first stage estimates an initial transformation between the two modalities. It follows the coarse-to-fine matching paradigm of 2D3D-MATR [19] and provides the initialization for the proposed rendering-based refinement module. Multi-resolution features are extracted using modality-specific backbones: KPFCNN [36] for the point cloud, and a ResNet-FPN backbone [12,21] for the image, with the deepest image features augmented by DINOv2 descriptors [28].

At the coarse level, we denote the image-patch and point-node features by $\hat { \mathbf { F } } ^ { \mathrm { I } } \in \mathbb { R } ^ { \hat { H } \times \hat { W } \times \hat { C } }$ and $\hat { \mathbf { F } } ^ { \mathrm { P } } \in \mathbb { R } ^ { \hat { N } \times \hat { C } }$ , respectively. At the fine level, the corresponding features are denoted by $\mathbf { F } ^ { \mathrm { I } } \in \dot { \mathbb { R } ^ { H \times W \times \check { C } } }$ and $\mathbf { F } ^ { \mathrm { P } } \in \mathbb { R } ^ { N \times \dot { C } }$ . A cross-modal transformer [19] refines the coarse features $\hat { \mathbf { F } } ^ { \mathrm { I } }$ and $\hat { \mathbf { F } } ^ { \mathrm { P } }$ to produce context-aware descriptors $\hat { \mathbf { H } } ^ { \mathrm { I } } \in \mathbb { R } ^ { \hat { H } \hat { W } \times \hat { C } }$ and $\hat { \mathbf { H } } ^ { \mathrm { P } } \in \mathbb { R } ^ { \hat { N } \times \hat { C } }$ . Coarse image-patch-to-point-node correspondences are obtained by computing the pairwise similarity:

![](images/a13c67c13d8d48f3d280ab773c75855e4695bd709365fe224df44724d40210de.jpg)  
Fig. 1: Summary of the proposed GRIP method.

$$
\hat { \mathbf { S } } _ { i j } ^ { c } = \left. \hat { \mathbf { h } } _ { i } ^ { \mathrm { P } } , \hat { \mathbf { h } } _ { j } ^ { \mathrm { I } } \right. ,\tag{2}
$$

where $\hat { \mathbf { h } } _ { i } ^ { \mathrm { P } }$ denotes the i-th coarse point feature and $\hat { \mathbf { h } } _ { { i } } ^ { \mathrm { I } }$ denotes the j-th image patch feature. A top-k selection is then applied to $\hat { \mathbf { S } } ^ { c } ,$ yielding a set of coarse correspondences $\hat { \mathcal { C } } = \{ ( \hat { \bf p } _ { m } , \hat { \bf u } _ { m } ) \} _ { m = 1 } ^ { N _ { c } } ,$ , where each coarse pair links an image patch $\hat { \mathbf { u } } _ { m }$ to a point node $\hat { \mathbf { p } } _ { m } .$ , and $N _ { c }$ is the number of selected correspondences.

The coarse correspondences are then used to restrict dense pixel-to-point matching to local patch pairs. For each coarse pair $\left( \hat { \mathbf { p } } _ { m } , \hat { \mathbf { u } } _ { m } \right)$ , the associated local 3D points and local image pixels are retrieved from the fine-resolution representations. Dense similarities are computed between their fine-level descriptors, and a mutual top-k selection is applied to obtain a set of dense correspondences $\mathcal { C } _ { m }$ The final dense correspondence set is obtained by aggregating all local matches as $\textstyle { \mathcal { C } } = \bigcup _ { m = 1 } ^ { N _ { c } } { \mathcal { C } } _ { m }$

In our experiments, the initial transformation $\mathbf { T } _ { 0 } \in \mathrm { S E } ( 3 )$ is then estimated from C using PnP-RANSAC [10, 18] with a maximum of 2,000 iterations. This transformation serves as the geometric guide for the feature rendering-based refinement stage described below.

## 3.3 Feature Rendering-based Refinement

A key design principle of the proposed GRIP is that reliable 2D-3D correspondences should be refined via an intermediate representation that spatially aligns image and point cloud features, rather than only through direct cross-modal descriptor comparison. Accordingly, GRIP consists of four main components: i) the Point-to-pixel Feature Renderer converts unordered point-cloud descriptors into an image-aligned feature map via pose-aware feature splatting using the initial pose $\mathbf { T } _ { 0 } . \mathrm { ~ i i } )$ The Pixel-Aligned Interaction Transformer refines image features and rendered 3D-derived features through bidirectional cross-modal attention on a shared 2D grid. iii) The Hierarchical Cross-Modal Feature Propagation maps the refined rendered features back to the sparse 3D nodes, yielding updated point cloud descriptors enriched with visual information. iv) Matching and Pose Estimation uses the refined descriptors to recover improved dense 2D-3D correspondences and estimate the final rigid transformation.

Point-to-pixel feature renderer Given coarse point features $\hat { \mathbf { F } } ^ { \mathrm { P } } \in \mathbb { R } ^ { \hat { N } \times \hat { C } }$ associated with points $\hat { \textbf { P } } \in \mathbb { R } ^ { \hat { N } \times 3 }$ , the Point-to-Pixel Feature Renderer uses the initial pose $\mathbf { T } _ { 0 }$ to softly splat point features onto the coarse image grid, transforming an unordered point representation into an image-aligned feature map [15, 39]. This soft and diferentiable assignment is particularly beneficial under imperfect pose initialization. In contrast to hard projection, which assigns each 3D feature to a single pixel and is therefore prone to misalignment under small pose perturbations, Gaussian splatting distributes feature responses over a local image neighbourhood, enabling nearby tokens to retain informative geometric cues. For each coarse 3D point $\hat { \mathbf { p } } _ { i } \in \hat { \mathbf { P } }$ , we instantiate one renderable Gaussian primitive, containing the attributes required by the splatting renderer: a 3D Gaussian shape, an opacity value, and a feature vector to be projected onto the image grid. The spatial support of this primitive is an anisotropic 3D Gaussian $\mathcal { G } _ { i }$ defined by a center $\mu _ { i }$ and a covariance matrix $\Sigma _ { i }$ . Following anisotropic Gaussian splatting [15,39], the covariance matrix is parameterized by a rotation matrix $\mathbf { Q } _ { i }$ and a scaling matrix $\mathbf { \Lambda } _ { \Lambda _ { i } }$ as $\pmb { \Sigma } _ { i } = \mathbf { Q } _ { i } \pmb { \Lambda } _ { i } \pmb { \Lambda } _ { i } ^ { \top } \mathbf { Q } _ { i } ^ { \top }$

The primitive parameters can then be defined as $\theta _ { i } = \{ \mu _ { i } , \mathbf { s } _ { i } , \mathbf { r } _ { i } , \alpha _ { i } , \mathbf { f } _ { i } ^ { G s } \}$ Here, $\mu _ { i } = \hat { \mathbf { p } } _ { i } + \varDelta \mathbf { p } _ { i }$ is the Gaussian center, where $\varDelta \mathbf { p } _ { i }$ is a learned ofset. The vector $\mathbf { s } _ { i }$ defines the anisotropic scale, with $\mathbf { \Lambda } \Lambda _ { i } = \mathrm { d i a g } ( \mathbf { s } _ { i } )$ , while $\mathbf { r } _ { i }$ parameterizes the Gaussian rotation matrix $\mathbf { Q } _ { i }$ . The scalar $\alpha _ { i }$ denotes the opacity, and $\mathbf { f } _ { i } ^ { G s } \in$ $\mathbb { R } ^ { C _ { G s } }$ is the feature vector carried by the primitive and splatted onto the image grid. All these attributes are predicted from the input point feature $\hat { \mathbf { f } } _ { i } ^ { \mathrm { P } }$ as:

$$
\mathbf { f } _ { i } ^ { G s } = \hat { \mathbf { f } } _ { i } ^ { \mathrm { P } } \mathbf { W }\tag{3}
$$

where $\mathbf { W } \in \mathbb { R } ^ { \hat { C } \times C _ { G s } }$ is a learned projection. The remaining primitive parameters are predicted as functions of the point descriptor $\hat { \mathbf { f } } _ { i } ^ { \mathrm { P } }$ :

$$
\varDelta \mathbf { p } _ { i } = h _ { \theta } ^ { p } ( \hat { \mathbf { f } } _ { i } ^ { \mathrm { P } } ) , \quad \mathbf { r } _ { i } = h _ { \theta } ^ { r } ( \hat { \mathbf { f } } _ { i } ^ { \mathrm { P } } ) , \quad \mathbf { s } _ { i } = \mathrm { s o f t p l u s } ( h _ { \theta } ^ { s } ( \hat { \mathbf { f } } _ { i } ^ { \mathrm { P } } ) ) , \quad \alpha _ { i } = \sigma \big ( h _ { \theta } ^ { \alpha } ( \hat { \mathbf { f } } _ { i } ^ { \mathrm { P } } ) \big ) .\tag{4}
$$

where $h _ { \theta } ^ { p } : \mathbb { R } ^ { \hat { C } } \to \mathbb { R } ^ { 3 }$ predicts the center ofset, $h _ { \theta } ^ { r } : \mathbb { R } ^ { \hat { C } }  \mathbb { R } ^ { 6 }$ predicts a continuous 6D rotation representation following [45], $h _ { \theta } ^ { s } : \mathbb { R } ^ { \hat { C } }  \mathbb { R } ^ { 3 }$ predicts the anisotropic scale, and $h _ { \theta } ^ { \alpha } : \mathbb { R } ^ { \hat { C } }  \mathbb { R }$ predicts the opacity, with $\sigma ( \cdot )$ denoting the sigmoid function. The ofset prediction head $h _ { \theta } ^ { p }$ is initialized to zero, and no explicit regularization is imposed on the predicted ofsets. The predicted rotation parameter $\mathbf { r } _ { i }$ is converted into a valid rotation matrix $\mathbf { Q } _ { i }$ , while $\mathbf { s } _ { i }$ defines the diagonal scaling matrix $\mathbf { \Lambda } \Lambda _ { i } = \mathrm { d i a g } ( \mathbf { s } _ { i } )$ .

At this point, one can exploit the known camera intrinsics K and the initial pose $\mathbf { T } _ { 0 }$ to build an organized representation of the point cloud $\hat { \mathbf { F } } ^ { R e } \in$ $\mathbb { R } ^ { \hat { H } \times \hat { W } \times C _ { G s } }$ by rendering the attributes of each primitive into the coarse image plane:

$$
\hat { \mathbf { F } } ^ { R e } = \operatorname { R e } \left( \{ \mu _ { i } , \mathbf { Q } _ { i } , \boldsymbol { \Lambda } _ { i } , \alpha _ { i } , \mathbf { f } _ { i } ^ { G s } \} _ { i = 1 } ^ { \hat { N } } \mid \hat { \mathbf { K } } , \mathbf { T } _ { 0 } \right) .\tag{5}
$$

where Re denotes the Gaussian feature renderer [15,39]. Before rendering, points with invalid camera-frame depth are discarded, and primitives projected far outside the image domain are filtered out. The renderer projects each remaining primitive into the image plane using the scaled coarse intrinsics $\hat { \mathbf { K } } .$ , derived from $\mathbf { K }$ , and the initial pose $\mathbf { T } _ { 0 }$ , computes its projected Gaussian footprint, and alphacomposites the feature vectors carried by the visible primitives, as in [39]. The efective opacity contribution of primitive $\theta _ { i }$ at location uˆ is defined as:

$$
g _ { i } ( \hat { \mathbf { u } } ) = \alpha _ { i } \exp \left( - \frac { 1 } { 2 } ( \hat { \mathbf { u } } - \hat { \mathbf { u } } _ { i } ) ^ { \top } ( \Sigma _ { i } ^ { 2 D } ) ^ { - 1 } ( \hat { \mathbf { u } } - \hat { \mathbf { u } } _ { i } ) \right)\tag{6}
$$

where $\hat { \mathbf { u } } _ { i } = \pi ( \mathbf { R } _ { 0 } \mu _ { i } + \mathbf { t } _ { 0 } ; \hat { \mathbf { K } } )$ is the projected center of the i-th primitive with the rotation $\mathbf { R } _ { 0 }$ and the translation $\mathbf { t } _ { \mathrm { 0 } }$ from $\mathbf { T } _ { 0 }$ and $\Sigma _ { i } ^ { 2 D }$ denotes the projected 2D covariance of the i-th Gaussian primitive. The rendered feature is then computed as:

$$
\hat { \mathbf { F } } ^ { R e } ( \hat { \mathbf { u } } ) = \sum _ { i \in \mathcal { N } ( \hat { \mathbf { u } } ) } T _ { i } ( \hat { \mathbf { u } } ) g _ { i } ( \hat { \mathbf { u } } ) \mathbf { f } _ { i } ^ { G s }\tag{7}
$$

where $\mathcal { N } ( \hat { \mathbf { u } } )$ denotes the set of projected primitives contributing to location uˆ and $T _ { i }$ the accumulated transmittance is defined as:

$$
T _ { i } ( \hat { \mathbf { u } } ) = \prod _ { j < i } \left( 1 - g _ { j } ( \hat { \mathbf { u } } ) \right)\tag{8}
$$

The primitives are sorted in front-to-back order according to their depth in the camera view. The resulting feature map $\hat { \mathbf { F } } ^ { R e }$ is thus an image-aligned representation of the point cloud features. It can therefore interact with the image backbone features $\hat { \mathbf { F } } ^ { \mathrm { I } }$ . The shared coarse grid enables pixel-aligned crossattention between rendered point features and 2D image features, as detailed below.

Pixel-aligned interaction transformer The main purpose of this central block is to exploit the organized structure of the rendered point cloud feature map $\hat { \mathbf { F } } ^ { R e } \in \mathbb { R } ^ { \bar { \hat { H } } \times \hat { W } \times C _ { G s } }$ to enable information exchange between the two modalities. The rendered descriptor $\hat { \mathbf { F } } ^ { R e }$ captures local geometric structure in the point cloud while the image feature map $\hat { \mathbf { F } } ^ { \mathrm { \hat { I } } } \in \mathbb { R } ^ { \hat { H } \times \hat { W } \times \check { C } }$ mainly encodes visual appearance and semantic context. Since both representations are organized on the same coarse image grid, they can interact in a shared pixel-aligned space. This interaction allows the rendered point cloud features to integrate visual context from the image, while the image features, in turn, are enhanced with geometry-aware cues from the point cloud.

To this end, we map both feature maps into a shared feature dimension ${ \hat { C } } .$ The rendered point cloud feature map and the image feature map are then flattened into token sequences and passed through linear projections as:

$$
{ } ^ { 1 } \hat { \mathbf { F } } ^ { R e } = \mathrm { F l a t t e n } ( \hat { \mathbf { F } } ^ { R e } ) \mathbf { W } ^ { \mathrm { P } } , \quad { } ^ { 1 } \hat { \mathbf { F } } ^ { \mathrm { I } } = \mathrm { F l a t t e n } ( \hat { \mathbf { F } } ^ { \mathrm { I } } ) \mathbf { W } ^ { \mathrm { I } }\tag{9}
$$

where $\mathbf { W } ^ { \mathrm { P } } \in \mathbb { R } ^ { \hat { C } _ { G s } \times \hat { C } }$ and ${ \bf W } ^ { \mathrm { I } } \in \mathbb { R } ^ { \hat { C } _ { I } \times \hat { C } }$ are learned modality-specific linear projection matrices, and $^ 1 \hat { \mathbf { F } } ^ { R e } , ^ { 1 } \hat { \mathbf { F } } ^ { \mathrm { I } } \in \mathbb { R } ^ { \hat { H } \hat { W } \times \hat { C } }$ . Since both token sequences are defined on the same coarse image grid, we associate each token with its normalized 2D coordinate $p x _ { i } \in [ - 1 , 1 ] ^ { 2 }$ , making the attention aware of the pixel location of each token while preserving their alignment on the same grid. These coordinates are encoded with a Fourier positional embedding [26, 31], denoted as $\gamma ( \cdot )$ , and projected to the same feature dimension:

$$
\mathbf { e } _ { i } = \mathbf { W } ^ { p o s } \gamma ( p x _ { i } )\tag{10}
$$

We perform bidirectional shifted cross-attention [23], $i . e . ,$ , from point cloud to image and from image to point cloud, to update both feature maps with the same algorithm. For instance, consider the point cloud-to-image direction, where image tokens query the rendered point cloud tokens. Let the output of one attention layer be the feature matrix $\mathbf { Z } \in \mathbb { R } ^ { \hat { H } \hat { W } \times \hat { C } }$ . It is then given by:

$$
\mathbf { Z } _ { i } = \sum _ { j \in \mathcal { W } ( i ) } a _ { i , j } \left( { ^ { 1 } \hat { \mathbf { f } } _ { j } ^ { R e } \mathbf { W } ^ { V } } \right) ,\tag{11}
$$

where ${ ^ 1 \hat { \mathbf { f } } _ { j } ^ { R e } } \in { \mathbb { R } } ^ { \hat { C } }$ denotes the $j \mathrm { - t h }$ rendered point-cloud token, i.e., the j-th row of ${ } ^ { 1 } \hat { \mathbf { F } } ^ { R e }$ , and $\mathcal { W } ( i )$ denotes the local shifted window associated with token i, $\mathbf { W } ^ { V }$ is the value projection matrix, and $a _ { i , j }$ denotes the attention weight between the i-th image token and the j-th rendered point cloud token within this window. The attention weights are computed as:

$$
a _ { i , j } = \mathrm { s o f t m a x } _ { j } \left( \frac { \left( \left( ^ { 1 } \hat { \mathbf { f } } _ { i } ^ { \mathrm { I } } + \mathbf { e } _ { i } \right) \mathbf { W } ^ { Q } \right) \left( ( ^ { 1 } \hat { \mathbf { f } } _ { j } ^ { R e } + \mathbf { e } _ { j } ) \mathbf { W } ^ { K } \right) ^ { \top } } { \sqrt { C } } \right)\tag{12}
$$

where $\mathbf { W } ^ { Q }$ and $\mathbf { W } ^ { K }$ are the query and key projection matrices, respectively. This bidirectional interaction enables image and rendered point-derived tokens to selectively incorporate both geometric and semantic information, helping the model capture repeated patterns across modalities and facilitating feature matching. The transformer outputs refined rendered point cloud and image feature maps, denoted as $\hat { \mathbf { Z } } ^ { R e } \in \tilde { \mathbb { R } ^ { \hat { H } \times \hat { W } \times \hat { C } } }$ and $\hat { \mathbf { Z } } ^ { \mathrm { I } } \in \mathbb { R } ^ { \hat { H } \times \hat { W } \times \hat { C } }$ , respectively. These features are then decoded and propagated to finer resolutions, as detailed below.

Hierarchical cross-modal feature propagation At this stage, we progressively propagate coarse point clouds and image feature maps to finer resolutions. On the image side, $\hat { \mathbf { Z } } ^ { \mathrm { I } }$ is first projected by a fusion Feed-Forward Network (FFN) and concatenated with the original coarse image descriptor $\hat { \mathbf { F } } ^ { \mathrm { I } }$ to produce $\hat { \mathbf { V } } ^ { \mathrm { I } } \in \mathbb { R } ^ { \hat { H } \hat { W } \times 2 \hat { C } }$ :

$$
\hat { \mathbf { V } } ^ { \mathrm { I } } = [ \hat { \mathbf { F } } ^ { \mathrm { I } } , \phi _ { \mathrm { I } } ( \hat { \mathbf { Z } } ^ { \mathrm { I } } ) ]\tag{13}
$$

where $[ \cdot , \cdot ]$ denotes channel-wise concatenation. The fusion function $\phi _ { \mathrm { I } }$ is implemented as an FFN, consisting of a LayerNorm layer followed by two linear projections with a GELU non-linearity in between. The resulting image features $\hat { \mathbf { V } } ^ { \mathrm { I } }$ are then fed into an FPN-style 2D decoder, which progressively propagates enhanced information to finer image resolutions following the first-stage architecture described in Section 3.2.

In the same spirit, we fuse the coarse point cloud descriptor $\hat { \mathbf { F } } ^ { \mathrm { P } } \in \mathbb { R } ^ { \hat { N } \times \hat { C } }$ with the Pixel-Aligned Interaction Transformer output $\hat { \mathbf { Z } } ^ { R e } \in \mathbb { R } ^ { \hat { H } \times \hat { W } \times \hat { C } }$ . To do so, we first need to recover the feature vector associated with each coarse node $\hat { \mathbf { p } } _ { i }$ . Each node is projected onto the coarse image grid using the intrinsics and the initial transformation $\mathbf { T } _ { 0 }$ . Thereby, the corresponding feature vector for the node $\hat { { \bf p } } _ { i }$ is aggregated from $\hat { \mathbf { Z } } ^ { R e }$ via bilinear grid sampling, which extracts features from the surrounding pixels of its 2D projection. The resulting pointwise feature map is denoted as $\hat { \mathbf { Z } } ^ { \mathrm { P } } \in \mathbb { R } ^ { \hat { N } \times \hat { C } }$ . We then compute the fused point cloud features $\hat { \mathbf { V } } ^ { \mathrm { P } } \in \mathbb { R } ^ { \hat { N } \times 2 \hat { C } }$ as:

$$
\hat { \mathbf { V } } ^ { \mathrm { P } } = [ \hat { \mathbf { F } } ^ { \mathrm { P } } , \phi _ { \mathrm { P } } ( \hat { \mathbf { Z } } ^ { \mathrm { P } } ) ]\tag{14}
$$

where the fusion function $\phi _ { \mathrm { P } }$ shares the same architecture as $\phi _ { \mathrm { I } }$ . The resulting coarse feature map $\hat { \mathbf { V } } ^ { \mathrm { P } }$ is then fed into a KPFCNN [36] decoder, following the first-stage architecture described in Section 3.2. The dense image and point cloud feature maps, denoted as $\mathbf { V } ^ { \mathrm { I } } \in \mathbb { R } ^ { H \times W \times C }$ and ${ \bf V } ^ { \mathrm { P } } \in \mathbb { R } ^ { N \times C }$ , respectively, are used to guide the matching and registration processes, as explained below.

Matching and pose estimation Given the refined coarse features $\hat { \mathbf { V } } ^ { \mathrm { I } } \in$ $\mathbb { R } ^ { \hat { H } \hat { W } \times 2 \hat { C } }$ and $\hat { \mathbf { V } } ^ { \mathrm { P } } \in \mathbb { R } ^ { \hat { N } \times 2 \hat { C } }$ , together with the decoded dense features ${ \bf V } ^ { \mathrm { I } } \in  \qquad $ $\mathbb { R } ^ { H W \times C }$ and $\mathbf { V } ^ { \mathrm { P } } \in \mathbb { R } ^ { N \times C }$ , we perform the same coarse to fine matching strategy as described in Section 3.2. We first compute a coarse similarity matrix between the refined point cloud nodes and image patches:

$$
\hat { \mathbf { S } } _ { i j } = \left. \hat { v } _ { i } ^ { \mathrm { P } } , \hat { v } _ { j } ^ { \mathrm { I } } \right.\tag{15}
$$

where $\hat { v } _ { i } ^ { \mathrm { P } }$ and $\hat { v } _ { j } ^ { \mathrm { I } }$ denote the refined coarse descriptors of the i-th point cloud node and the $j { \cdot } \mathrm { { \dot { \ t h } } }$ image patch, respectively. A masked top-k selection is then applied to $\hat { \textbf { S } } \mathrm { t o }$ obtain a set of coarse patch correspondences $\hat { \mathcal { C } } = ( \hat { \mathbf { p } } _ { m } , \hat { \mathbf { u } } _ { m } ) _ { m = 1 } ^ { N _ { c } }$

For each selected coarse correspondence, we retrieve the associated local image pixel and point cloud, and compute dense similarities between their decoded fine-level descriptors. The dense similarity between a point descriptor $v _ { i } ^ { \mathrm { P } }$ and an image descriptor $v _ { j } ^ { \mathrm { I } }$ is defined as

$$
{ \bf S } _ { i j } = \frac { \left. v _ { i } ^ { \mathrm { P } } , v _ { j } ^ { \mathrm { I } } \right. } { \tau }\tag{16}
$$

where $\tau$ is a temperature parameter. Mutual top-k selection is then applied within each matched patch pair to recover the dense pixel-to-point correspondence set C<sup>∗</sup>. Finally, the 2048 highest-scoring correspondences are used for pose estimation. We evaluate two final pose estimators. The first is a score-guided PnP-RANSAC variant. Instead of sampling minimal sets uniformly [10], each correspondence is assigned a confidence score derived from the dense similarity matrix S. Correspondences are sorted by decreasing confidence, and minimal PnP sets are first sampled from high-confidence subsets. This prioritization enables reliable matches to be tested earlier, reducing the maximum number of RANSAC iterations to 500, compared with the 50,000 iterations used by the methods reported in Section 4, such as [14,19, 41]. Each pose hypothesis is evaluated over all correspondences using the reprojection error with an 8 pixel inlier threshold. The second variant is the weighted PnP solver of [3], which directly uses the correspondence confidence scores S to estimate the transformation in a small number of iterations, i.e., 3 in our case.

## 3.4 Implementation details

The method is implemented in PyTorch and trained on an NVIDIA Tesla V100 GPU with 32GB of memory. For both stage 1 and 2 training, we adopt the same coarse-to-fine circle loss supervision commonly used in recent image-topoint cloud registration methods [19,41]. The network is trained with the Adam optimizer [16] using a learning rate of $1 0 ^ { - 4 }$ , which is decayed by a factor of 0.95 at each epoch. Stage 1 is trained for 20 epochs, and all its parameters are frozen during stage 2. Stage 2 is then trained for 20 epochs with $T _ { 0 }$ predicted by the pretrained Stage 1. For stage 1, the 2D backbone is implemented as a 4-stage ResNet [12] with FPN [21], with output dimensions {128, 128, 256, 512}. The point cloud backbone is implemented as a 3-stage KPFCNN [36], with dimensions {128, 256, 512}, yielding $\hat { C } = 5 1 2$ at the coarsest level. For stage 2, the point cloud features used for Gaussian feature splatting are projected to a 128- dimensional feature space, $i . e . , C _ { G s } = 1 2 8$ . The Pixel-Aligned Interaction Transformer module is implemented with shifted-window bidirectional cross-attention using a window size of 32. The image and point cloud fusion heads $\phi _ { \mathrm { I } }$ and ϕ are implemented as separate FFNs, each composed of a normalization layer, two linear projections, and a GELU activation. The input image resolution is $4 8 0 \times 6 4 0$ , and the coarsest feature grid has resolution $3 4 \times 4 5$

## 4 Experimental Validation

To evaluate the performance of GRIP, we follow state of the art image to point cloud registration methods [5, 27, 41] and report results on the RGB-D Scenes V2 [17] and 7-Scenes [11] datasets under the Registration Recall (RR), Inlier Ratio (IR), and Feature Matching Recall (FMR) metrics.

## 4.1 Datasets

RGB-D Scenes V2. Introduced by Lai et al. [17], RGB-D Scenes V2 contains 14 indoor RGB-D video sequences, covering both tabletop objects and large furniture. Following prior works [19, 27, 41], we construct the 2D-3D registration benchmark by fusing one point cloud fragment from every 25 consecutive depth frames and pairing it with one RGB image sampled every 25 frames. Only pairs with an overlap ratio of at least 30% are retained. Scenes 1-8 are used for training, scenes 9 and 10 for validation, and scenes 11-14 for testing. This split yields 1,748 training pairs, 236 validation pairs, and 497 test pairs.

7-Scenes. Similarly, Glocker et al. [11] introduced a challenging indoor RGB-D dataset with greater scale and viewpoint variations across scenes than RGB-D Scenes V2. The dataset consists of 46 tracked RGB-D sequences captured from 7 indoor scenes using a handheld Kinect RGB-D camera at a resolution of 640 × 480. We follow the same procedure as above to process the input images and point clouds. Following [19,27], we retain only image-point cloud pairs with at least 50% overlap. Using the oficial sequence split yields 4048 training pairs, 1011 validation pairs, and 2304 testing pairs.

## 4.2 Results

RGB-D Scenes V2 As reported in Table 1, GRIP achieves the best matching performance on RGB-D Scenes V2 in terms of inlier ratio (IR). In particular, GRIP reaches an IR of 60.9%, outperforming the second-best method, R<sup>23</sup>Net [6], by 17.5 percentage points. Compared with [19], which obtains 32.4% IR, this corresponds to a relative improvement of approximately 88%, demonstrating the efectiveness of the proposed rendering-based refinement for producing geometrically consistent correspondences.

Regarding registration, following prior work [19, 41], we report RR at a 10 cm threshold as the primary metric. GRIP ranks second overall on this metric, reaching 84.6% RR with EPro-PnP and 84.9% RR with score-guided RANSAC, while Dif-Reg [41], which adopts a difusion process, obtains 85.7%. This indicates that the substantial gains in correspondence quality translate into highly competitive registration recall. Moreover, although GRIP has a slightly higher network forward time than Dif-Reg, its total inference time is lower because the final pose estimation stage is substantially more eficient. Under identical hardware conditions, GRIP reduces total inference latency by 40.2% with EPro-PnP and by 27.8% with score-guided RANSAC compared to Dif-Reg (Table 3).

7-Scenes. Similarly, following previous works [19, 41], we evaluate the same metrics on the 7-Scenes dataset. The results reported in Table 2 show trends consistent with those observed on RGBD Scenes V2. In particular, GRIP achieves the best IR, reaching 71.9%. Compared with the 2D3DMATR baseline, which achieves an IR of 50.1%, this represents an absolute improvement of 21.8 percentage points and a relative improvement of approximately 43.5%. The secondbest method is $\mathrm { R ^ { 2 3 } N e t }$ , at 54.9%, which is 17.0 percentage points lower than our method. This substantial gain in IR indicates that GRIP produces significantly more geometrically consistent 2D-to-3D correspondences, as qualitatively illustrated in Fig. 3. Although the Feature Matching Recall (FMR) of GRIP is slightly lower than the best reported value, its substantially higher IR indicates that the predicted correspondences are more geometrically consistent once a suficient set of matches is established. This improved correspondence quality directly benefits pose estimation. GRIP achieves an RR of 84.1% with EPro-PnP and 84.2% with score-guided RANSAC, outperforming all compared methods. These results show that the proposed method not only increases the number of tentative matches but also improves their geometric accuracy, leading to more robust and successful registration.

Table 1: Evaluation results on RGB-D V2. Bold numbers highlight the best performance; the second best are underlined.  
Table 2: Evaluation results on 7-Scenes. Bold numbers highlight the best performance; the second best are underlined.
<table><tr><td>Model</td><td>IR ↑ FMR↑ RR↑</td></tr><tr><td>P2-Net [37] Predator-2D3D [14] 2D3D-MATR [19]</td><td>12.2 59.6 38.4 15.7 65.9 30.2 32.4 90.8 56.4</td></tr><tr><td>Diff²I2P [27] Diff-Reg [41]</td><td>36.9 77.1 60.5 37.7 91.4 85.7</td></tr><tr><td>R23Net [6] GRIP - EPro-PnP</td><td>43.4 93.6 77.0 60.9 91.8 84.6</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>IR ↑FMR ↑RR ↑</td></tr><tr><td rowspan=1 colspan=1>P2-Net [37]</td><td rowspan=1 colspan=1>31.7  79.0  65.7</td></tr><tr><td rowspan=1 colspan=1>Predator-2D3D [14]</td><td rowspan=1 colspan=1>23.4  77.5  48.5</td></tr><tr><td rowspan=1 colspan=1>2D3D-MATR [19]</td><td rowspan=1 colspan=1>50.1  92.1  75.8</td></tr><tr><td rowspan=1 colspan=1>Diff2I2P [27]</td><td rowspan=1 colspan=1>53.2  92.2  83.0</td></tr><tr><td rowspan=2 colspan=1>Diff-Reg [41]R23Net [6]</td><td rowspan=1 colspan=1>52.0  91.6  83.8</td></tr><tr><td rowspan=1 colspan=1>54.9 93.2  83.8</td></tr><tr><td rowspan=2 colspan=1> $\overline { { G R I P \cdot \mathbf { E P r o - P n P } } }$  $G R I P \mathrm { ~ - ~ } \mathbf { R A N S A C ^ { * } }$ </td><td rowspan=1 colspan=1>71.9  89.8  84.1</td></tr><tr><td rowspan=1 colspan=1>71.9  89.8  84.2</td></tr></table>

![](images/abb0e9f7e4e0fca838a182c97ce5e8233c78c716a8139cb92de01a64080cd7f2.jpg)  
(a) Registration Recall evolution on RGBD V2.

![](images/e22718cd968d57a1dd1ebe75cba3dd0510e8b0866a0481bdcc191c324afc11c1.jpg)  
(b) Registration Recall evolution on 7Scenes.  
Fig. 2: Registration Recall for increasing threshold on RGBD V2 and 7Scenes.

In addition, we compute the RR for both datasets using thresholds ranging from 2 cm to 10 cm. As shown in Fig. 2, GRIP achieves higher RR under stricter thresholds, demonstrating its superior registration precision. Under more relaxed thresholds, Dif-Reg gradually closes the gap with GRIP and surpasses it on RGBD Scenes V2, regardless of the pose estimator used.

Table 3: Inference time cost comparison on RGBD V2 dataset.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="2">Registration Recall Mean inference time (seconds)</td></tr><tr><td>Model Pose</td><td>Total</td></tr><tr><td>Diff-Reg [41]</td><td>85.7</td><td>0.702 0.634</td><td>1.336</td></tr><tr><td>GRIP - EPro-PnP (Ours)</td><td>84.6</td><td>0.790 0.009</td><td>0.799</td></tr><tr><td>GRIP - RANSAC*(Ours)</td><td>84.9</td><td>0.790 0.175</td><td>0.965</td></tr></table>

![](images/6dd712e7072643809ddf4ccfe38b0f5b3694c68bde11ada1e8156e2558f9f039.jpg)

Fig. 3: Qualitative comparison of matching results on selected 7Scenes examples. Each row shows the same image/point-cloud pair 512 correspondences with the highest score are represented. Green indicates inlier matches, while red indicates outliers. Table 4: Ablation on RGB-D V2.
<table><tr><td>Variant</td><td>DINO</td><td>Feat. Spl.</td><td>PAIT</td><td>Hier. Dec.</td><td>Solver</td><td>IR</td><td>RR</td></tr><tr><td>Baseline</td><td>x</td><td>x</td><td>x</td><td>x</td><td>RANSAC</td><td>32.4</td><td>56.4</td></tr><tr><td>Our Stage 1</td><td>√</td><td>x</td><td>x</td><td>X</td><td>RANSAC</td><td>44.2</td><td>78.1</td></tr><tr><td>Our Stage 1</td><td>✓</td><td>x</td><td>X</td><td>x</td><td>Score-guided</td><td>44.2</td><td>77.3</td></tr><tr><td>Stage 2 w/o PAIT</td><td>√</td><td>√</td><td>x</td><td>√</td><td>RANSAC</td><td>42.5</td><td>64.7</td></tr><tr><td>GRIP RANSAC</td><td></td><td>√</td><td>√</td><td>√</td><td>RANSAC</td><td>60.9</td><td>83.9</td></tr><tr><td>GRIP wo dec.</td><td>√</td><td></td><td></td><td>x</td><td>Score-guided</td><td>52.4</td><td>83.9</td></tr><tr><td>GRIP + Hard</td><td>√</td><td>Hard</td><td>L</td><td>√</td><td>Score-guided</td><td>59.3</td><td>82.3</td></tr><tr><td>GRIP</td><td>√</td><td></td><td>√</td><td>√</td><td>Score-guided</td><td>60.9</td><td>84.9</td></tr><tr><td>Oracle (1-stage)</td><td></td><td></td><td></td><td></td><td>RANSAC</td><td>99.0</td><td>100.0</td></tr><tr><td>Oracle (2-stage)</td><td>1</td><td>V</td><td></td><td>L</td><td>Score-guided</td><td>78.0</td><td>99.9</td></tr></table>

## 4.3 Ablation Study

Table 4 shows that the complete GRIP pipeline improves IR/RR from 44.2/78.1 with the DINOv2 based Stage 1 to 60.9/84.9, corresponding to gains of +16.7 IR and +6.8 RR. The ablations indicate complementary contributions from Gaussian splatting, PAIT, and hierarchical decoding. PAIT provides the largest improvement by enabling cross modal interaction on the rendered feature grid, while the hierarchical decoder propagates the refined representation to fine matching. Replacing Gaussian splatting with hard point to pixel projection slightly decreases IR/RR to 59.3/82.3, with no substantial runtime reduction (882 ms vs. 895 ms). Overall, these results support the contribution of each component to the complete refinement pipeline.

## 4.4 Discussion and Future Directions

Experiments on RGB-D Scenes V2 and 7-Scenes demonstrate the efectiveness of GRIP for cross-modal image-to-point cloud matching, with particularly strong performance under stricter registration thresholds. For a fair comparison, the results of 2D3D-MATR and Dif-Reg were reproduced using their oficial implementations in the same experimental environment as GRIP. Overall, the results confirm that the proposed rendering-based refinement improves the geometric consistency of the predicted correspondences.

The main limitation of GRIP is its dependence on the initial pose. Since feature splatting uses $\mathbf { T } _ { 0 }$ to project 3D features into the image plane, this initialization must be suficiently accurate to produce meaningful rendered features. When the initial pose is degenerate or highly inaccurate, the rendered representation becomes poorly aligned with the image observations. As a result, the pixel-aligned transformer may receive uninformative or misleading evidence, which can lead to incorrect correspondences and degrade the final pose. This behavior is particularly evident on the Stairs scene of the 7-Scenes dataset, where the weak baseline initialization performance, i.e., a RR of 28.4% and an IR of 18.1%, also limits the improvement achieved by GRIP, which reaches a RR of 31.1% and an IR of 28.6%. These results suggest that the proposed method should be viewed as a pose-conditioned rendering-based refinement framework whose performance depends on the quality of the initial pose, rather than as a solution to arbitrary initialization.

A promising direction for future work is therefore to reduce the dependence on a single initial transformation. One possibility is to integrate GRIP into a generative or difusion-based registration framework [13,27,41], where the pose is progressively refined from a noisy transformation on the SE(3) manifold. Another direction is to sample multiple pose candidates around the initial transformation. Each candidate can produce its own feature-splatting representation, enabling the model to assess multiple alignment hypotheses simultaneously. This raises the likelihood of obtaining at least one informative rendering, efectively broadening the convergence basin and mitigating the impact of poor initializations.

Acknowledgments This work was supported by the French ANR program MARSurg (ANR-21-CE19-0026).

This work was performed using HPC resources from GENCI-IDRIS (Grant 2026- AD011015228R1)

## 5 Conclusion

In this paper, we introduced GRIP, a pose-conditioned rendering-based refinement framework for 2D-3D feature matching and registration. The main objective of our approach is to reduce the structural gap between grid-based image descriptors and unordered point cloud descriptors. To this end, we proposed a feature-splatting module that renders learned 3D point features into the image plane, producing an image-aligned point-derived feature map. A pixel-aligned interaction transformer then jointly refines image features and 3D-derived rendered features via bidirectional cross-modal attention on a shared 2D grid. The refined rendered features are finally sampled back to the 3D nodes, enabling improved dense 2D-3D correspondence estimation and final pose refinement. The refined representation improves dense correspondence quality, achieving the highest IR on both evaluated benchmarks and competitive RR, particularly under stricter thresholds. Future work will focus on reducing the dependence on the initial pose through multi-hypothesis or generative refinement.

## References

1. Bello, I., Zoph, B., Vaswani, A., Shlens, J., Le, Q.V.: Attention augmented convolutional networks. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3286–3295 (2019)

2. Besl, P.J., McKay, N.D.: Method for registration of 3-d shapes. In: Sensor fusion IV: control paradigms and data structures. vol. 1611, pp. 586–606 (1992)

3. Chen, H., Wang, P., Wang, F., Tian, W., Xiong, L., Li, H.: Epro-pnp: Generalized end-to-end probabilistic perspective-n-points for monocular object pose estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2781–2790 (2022)

4. Chen, J., Wei, X., Liang, X., Xu, H., Zhou, L., He, W., Ma, Y., Yin, Y.: High precision 3d reconstruction and target location based on the fusion of visual features and point cloud registration. Measurement 243, 116455 (2025)

5. Cheng, Z., Deng, J., Li, X., Yin, B., Zhang, T.: Bridge 2d-3d: Uncertainty-aware hierarchical registration network with domain alignment. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 2491–2499 (2025)

6. Cheng, Z., Liao, B., Deng, J., Yin, X., Li, X., Chen, Y., Yin, B., Zhang, T.: Rethinking 2d-3d registration: A novel network for high-value zone selection and representation consistency alignment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 39052–39063 (2026)

7. DeTone, D., Malisiewicz, T., Rabinovich, A.: Superpoint: Self-supervised interest point detection and description. In: IEEE/CVF CVPR. pp. 224–236 (2018)

8. Dusmanu, M., Rocco, I., Pajdla, T., Pollefeys, M., Sivic, J., Torii, A., Sattler, T.: D2-net: A trainable cnn for joint detection and description of local features. arXiv preprint arXiv:1905.03561 (2019)

9. Feng, M., Hu, S., Ang, M., et al.: 2d3d-matchnet: Learning to match keypoints across 2d image and 3d point cloud. In: IEEE ICRA. pp. 4790–4796 (2019)

10. Fischler, M.A., Bolles, R.C.: Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM 24, 381–395 (1981)

11. Glocker, B., Izadi, S., Shotton, J., Criminisi, A.: Real-time rgb-d camera relocalization. In: 2013 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). pp. 173–179. IEEE (2013)

12. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

13. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

14. Huang, S., Gojcic, Z., Usvyatsov, M., et al.: Predator: Registration of 3d point clouds with low overlap. In: IEEE/CVF CVPR. pp. 4267–4276 (2021)

15. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., et al.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139–1 (2023)

16. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

17. Lai, K., Bo, L., Fox, D.: Unsupervised feature learning for 3d scene labeling. In: 2014 IEEE International Conference on Robotics and Automation (ICRA). pp. 3050–3057. IEEE (2014)

18. Lepetit, V., Moreno-Noguer, F., Fua, P.: Ep n p: An accurate o (n) solution to the p n p problem. IJCV 81(2), 155–166 (2009)

19. Li, M., Qin, Z., Gao, Z., Yi, R., Zhu, C., Guo, Y., Xu, K.: 2d3d-matr: 2d-3d matching transformer for detection-free registration between images and point clouds. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14128–14138 (2023)

20. Li, Y., Snavely, N., Huttenlocher, D., Fua, P.: Worldwide pose estimation using 3d point clouds. In: European conference on computer vision. pp. 15–29. Springer (2012)

21. Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2117–2125 (2017)

22. Liu, W., Lai, B., Wang, C., Bian, X., Yang, W., Xia, Y., Lin, X., Lai, S.H., Weng, D., Li, J.: Learning to match 2d images and 3d lidar point clouds for outdoor augmented reality. In: 2020 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW). pp. 654–655. IEEE (2020)

23. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10012–10022 (2021)

24. Lowe, D.G.: Distinctive image features from scale-invariant keypoints. International journal of computer vision 60(2), 91–110 (2004)

25. Marchand, E., Uchiyama, H., Spindler, F.: Pose estimation for augmented reality: a hands-on survey. IEEE transactions on visualization and computer graphics 22(12), 2633–2651 (2015)

26. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021)

27. Mu, J., Ren, C., Zhang, W., Pan, L., Zhang, X.P., Gao, Y.: Dif2i2p: Diferentiable image-to-point cloud registration with difusion prior. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 25777–25787 (2025)

28. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

29. Qi, C.R., Su, H., et al.: Pointnet: Deep learning on point sets for 3d classification and segmentation. In: IEEE/CVF CVPR. pp. 652–660 (2017)

30. Qin, Z., Yu, H., Wang, C., et al.: Geometric transformer for fast and robust point cloud registration. In: IEEE/CVF CVPR. pp. 11143–11152 (2022)

31. Qin, Z., Yu, H., Wang, C., Peng, Y., Xu, K.: Deep graph-based spatial consistency for robust non-rigid point cloud registration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5394–5403 (2023)

32. Rublee, E., Rabaud, V., Konolige, K., Bradski, G.: Orb: An eficient alternative to sift or surf. In: 2011 International conference on computer vision. pp. 2564–2571. Ieee (2011)

33. Sarlin, P.E., DeTone, D., et al.: SuperGlue: Learning feature matching with graph neural networks. In: IEEE/CVF CVPR (2020)

34. Sattler, T., Leibe, B., Kobbelt, L.: Eficient & efective prioritized matching for large-scale image-based localization. IEEE Trans. PAMI 39(9), 1744–1756 (2016)

35. Sun, J., Shen, Z., Wang, Y., Bao, H., Zhou, X.: Loftr: Detector-free local feature matching with transformers. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8922–8931 (2021)

36. Thomas, H., Qi, C.R., Deschaud, J.E., Marcotegui, B., Goulette, F., Guibas, L.J.: Kpconv: Flexible and deformable convolution for point clouds. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 6411–6420 (2019)

37. Wang, B., Chen, C., Cui, Z., Qin, J., Lu, C.X., Yu, Z., Zhao, P., Dong, Z., Zhu, F., Trigoni, N., et al.: P2-net: Joint description and detection of local features for pixel and point matching. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16004–16013 (2021)

38. Wang, G., Zheng, Y., Wu, Y., Guo, Y., Liu, Z., Zhu, Y., Burgard, W., Wang, H.: End-to-end 2d-3d registration between image and lidar point cloud for vehicle localization. IEEE Transactions on Robotics (2025)

39. Wang, J., Zhang, Z., He, J., Xu, R.: Pfgs: High fidelity point cloud rendering via feature splatting. In: European Conference on Computer Vision. pp. 193–209. Springer (2024)

40. Wang, Y., Solomon, J.M.: Deep closest point: Learning representations for point cloud registration. In: IEEE/CVF ICCV (2019)

41. Wu, Q., Jiang, H., Luo, L., et al.: Dif-reg: Difusion model in doubly stochastic matrix space for registration problem. In: ECCV. pp. 160–178 (2024)

42. Yang, J., Li, H., Campbell, D., et al.: Go-icp: A globally optimal solution to 3d icp point-set registration. EEE Trans. Pattern Anal. Mach. Intell. 38, 2241–2254 (2015)

43. Yu, H., Li, F., Saleh, M., Busam, B., Ilic, S.: Cofinet: Reliable coarse-to-fine correspondences for robust pointcloud registration. Advances in Neural Information Processing Systems 34, 23872–23884 (2021)

44. Zhang, J., Yao, Y., Deng, B.: Fast and robust iterative closest point. EEE T. Pattern Anal. Mach. Intell. 44, 3450–3466 (2021)

45. Zhou, Y., Barnes, C., Lu, J., Yang, J., Li, H.: On the continuity of rotation representations in neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5745–5753 (2019)