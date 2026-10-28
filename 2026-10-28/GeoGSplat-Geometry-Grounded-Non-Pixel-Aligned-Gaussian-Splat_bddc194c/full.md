# GeoGSplat: Geometry-Grounded Non-Pixel-Aligned Gaussian Splatting

Jiale Wang<sup>1,3∗</sup> Chen Zhao<sup>2∗</sup> Haoyang Zhang<sup>3</sup> Yanhao Wu<sup>1,3</sup> Fei He<sup>3</sup> Yanhu Shan<sup>3</sup> Rui Wu<sup>3</sup> Wei Ke<sup>1</sup> Tong Zhang<sup>4‡</sup>

<sup>1</sup>Xi’an Jiaotong University <sup>2</sup>EPFL <sup>3</sup>Horizon Robotics <sup>4</sup>University of the Chinese Academy of Sciences

![](images/65134724b72d58607b0d32b2509aeea61d0506e37224a3a0475ef1b188e2e9fc.jpg)  
Figure 1: GeoGSplat introduces a geometry-grounded non-pixel-aligned Gaussian Splatting paradigm. It achieves high-quality 3D reconstruction and novel view synthesis by incorporating reliable geometric priors through complementary Anchor and Triplane representations. Larger bubble size indicates a larger number of Gaussians.

## ABSTRACT

Feed-forward Gaussian Splatting enables efficient novel view synthesis, but pixelaligned prediction makes the primitive count grow with image resolution and view count, leading to substantial redundancy and increased computational cost. Non-pixel-aligned methods address this limitation by predicting a fixed number of Gaussians. Without explicit ray and depth constraints, these methods yield misplaced Gaussian centers, causing rendering artifacts under large viewpoint changes. To address this issue, we introduce GeoGSplat, a geometrygrounded feed-forward model that explicitly incorporates geometric priors into the non-pixel-aligned framework. A dual-branch encoder constructs Anchors and Triplane to capture coarse 3D structure and fine-grained scene features, respectively. Subsequently, a hierarchical decoder progressively decodes dense 3D points and Gaussian primitives, enabling 3D reconstruction and novel view synthesis. GeoGSplat surpasses both state-of-the-art pixel-aligned and non-pixelaligned Gaussian Splatting methods on DTU and NRGBD in terms of novel view synthesis and reconstruction. The advantage of our geometry-grounded paradigm further increases as target viewpoints move farther from the input cameras, demonstrating stronger generalization to large viewpoint changes. Project Page: https://comedy0913.github.io/projects/geogsplat

## 1 INTRODUCTION

Feed-forward Gaussian Splatting (Charatan et al., 2024; Chen et al., 2024; Zhang et al., 2024; Ren et al., 2026) aims to directly predict Gaussian primitives from a set of images, supporting novel view synthesis (NVS) and avoiding expensive per-scene optimization (Kerbl et al., 2023). Most existing methods (Charatan et al., 2024; Chen et al., 2024) conduct pixel-aligned predictions, where each input pixel is associated with a predicted Gaussian primitive. Such methods inherit explicit geometric constraints from per-pixel depth prediction, which restricts Gaussian centers to a geometrically meaningful 3D surface and thus reduces the ambiguity of appearance learning during rasterization. However, this paradigm couples Gaussian primitives to both image resolution and the number of input views, resulting in considerable computational overhead. As illustrated in Fig. 2, the number of predicted primitives grows rapidly with the number of input views. Moreover, over lapping regions across images produce multiple Gaussians for the same scene content, introducing substantial redundancy. Some recent methods (Ren et al., 2026; Li et al., 2026a;b), adopt a nonpixel-aligned paradigm that predicts a fixed number of Gaussians. These methods typically decode Gaussians from a set of learnable tokens that aggregate multi-view image features through attention. While this formulation removes the scaling dependency of pixel-aligned prediction, it also discards the geometric constraints provided by per-pixel depth. Primitives are learned from photometric supervision, leaving the Gaussian centers underconstrained and resulting in inaccurate geometry and overfitting to the observed viewpoints. Such Gaussians may produce plausible renderings near the input cameras, but they tend to yield strong rendering artifacts under larger viewpoint changes.

![](images/f7ac27077548ce869b786b514bf93638d3b3d163d95c59383beb2c6f609171ba.jpg)  
Figure 2: Comparison of pixel-aligned, non-pixel-aligned, and our GeoGSplat paradigms. Pixel-aligned methods rely on explicit geometry priors but produce an increasing number of primitives (Prim.) with more input views. Non-pixel-aligned methods predict a fixed number of primitives but lack geometric priors. GeoGSplat combines fixed-size prediction with explicit geometric priors.

To address this limitation, we introduce GeoGSplat, a geometry-grounded non-pixel-aligned framework consisting of a dual-branch structured encoder and a geometry-guided hierarchical decoder. The encoder captures complementary scene properties through Anchors and Triplane. Anchors comprise sparse 3D spheres with learnable centers and adaptive radii, providing coarse geometric constraints on the scene structure. Triplane comprises three orthogonal 2D feature maps that encode fine-grained high-dimensional scene features queryable at arbitrary 3D coordinates. Building on the two complementary representations, the decoder progressively converts sparse geometric priors into dense 3D points and Gaussian primitives. We first perform Anchor-to-Point decoding by sampling Triplane features within each Anchor region to predict a set of dense 3D points. We then apply Point-to-Gaussian decoding, re-querying the Triplane at the decoded 3D points to predict corresponding Gaussian primitives. This hierarchical design establishes interaction between Anchors and Triplane, using Anchor geometry to constrain point reconstruction and employing fine-grained Triplane features to enable geometric refinement and appearance learning.

We conduct experiments on DTU (Jensen et al., 2014) and NRGBD (Azinovic et al., 2022) for´ object-level and scene-level evaluations, respectively. Our GeoGSplat outperforms both non-pixelaligned and pixel-aligned competitors on these datasets in terms of NVS and 3D reconstruction. Furthermore, we evaluate GeoGSplat under a challenging setting by progressively moving the target cameras farther from the input cameras. Its performance advantage becomes increasingly pronounced under larger viewpoint changes, showing stronger robustness and generalization under large viewpoint changes enabled by our geometric priors. Our contributions are summarized below.

• We introduce a geometry-grounded non-pixel-aligned paradigm for feed-forward Gaussian Splatting that incorporates explicit geometric priors, enabling high-quality NVS and 3D reconstruction in an efficient manner.

• We propose a dual-branch structured encoder that constructs Anchors and Triplane to capture complementary coarse 3D structure and fine-grained high-dimensional scene features, respectively.

• We develop a geometry-guided hierarchical decoder that progressively decodes dense 3D points and Gaussian primitives by querying Triplane features under Anchor-based geometric constraints.

## 2 RELATED WORKS

Pixel-aligned feed-forward Gaussian Splatting. Significant progress has been made in feedforward Gaussian Splatting (Charatan et al., 2024; Chen et al., 2024; Zhang et al., 2024; Ren et al., 2026; Szymanowicz et al., 2024), predicting Gaussian primitives from images without per-scene optimization. Most existing approaches follow a pixel-aligned mechanism (Chen et al., 2024; Xu et al., 2025; Charatan et al., 2024), predicting a Gaussian primitive for each image pixel. Such predictions benefit from explicit geometric constraints imposed by the corresponding camera rays and per-pixel depths, which restrict Gaussian centers to plausible 3D locations. However, the pixelaligned pipeline ties the number of Gaussians to the input image resolution and the number of views, leading to considerable redundancy and computational cost.

Non-pixel-aligned models. Non-pixel-aligned methods remove correspondences between image pixels and Gaussian primitives, instead predicting Gaussians from aggregated image features or scene representations (Wang et al., 2026; Zhang et al., 2026; Li et al., 2026a; Ren et al., 2026; Zhao et al., 2026; Moreau et al., 2026; Li et al., 2026b). However, removing pixel-to-Gaussian correspondences also eliminates the geometric constraints provided by per-pixel ray and depth. In this context, Gaussian centers tend to be inaccurately placed, leading to suboptimal solutions that overfit the observed viewpoints. While these primitives may still produce plausible renderings for novel views close to the input cameras, their geometric errors become increasingly exposed under large viewpoint changes, resulting in strong rendering artifacts. To tackle this issue, we introduce a new non-pixel-aligned paradigm that injects explicit geometric priors, enabling more robust NVS.

Scene Representations. Existing feed-forward methods adopt various scene representations (Yao et al., 2018; Gu et al., 2020; Yu et al., 2021; Wang et al., 2021; Chen et al., 2021; Jiang et al., 2025a; Johari et al., 2022; Wang et al., 2024; Leroy et al., 2024; Wang et $\mathrm { { a l . } }$ , 2025; Xiang et al., 2026). These representations play a critical role in Gaussian prediction. For instance, voxels and anchors constrain the spatial locations at which primitives are predicted (Lu et al., 2024; Fridovich-Keil et al., 2022); Triplane (Hong et al., 2024; Zhang et al., 2024; Chan et al., 2022) provides structured representations, where fine-grained high-dimensional scene features can be encoded and queried. Motivated by the complementarity of these two representations, GeoGSplat combines Anchors and Triplane, using Anchors to provide coarse geometric constraints and Triplane to supply fine-grained scene features.

## 3 METHOD

## 3.1 PRELIMINARIES

Given $N _ { I }$ posed images $\{ ( \mathbf { I } _ { i } , \pmb { \pi } _ { i } ) \} _ { i = 1 } ^ { N _ { I } }$ , feed-forward Gaussian Splatting predicts a set of 3D Gaussian primitives in a single pass. Here, $\mathbf { I } _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ is an RGB image and $\pi _ { i }$ denotes its camera parameters. The predicted Gaussian set $\mathcal { G }$ is defined as:

$$
\begin{array} { r } { \mathcal { G } = \{ \mathbf { g } _ { j } \} _ { j = 1 } ^ { N } , \qquad \mathbf { g } _ { j } = ( \pmb { \mu } _ { j } , \alpha _ { j } , \mathbf { s } _ { j } , \mathbf { q } _ { j } , \mathbf { c } _ { j } ) , } \end{array}\tag{1}
$$

where N denotes the number of predicted Gaussians, $\mu _ { j } , \alpha _ { j } , \mathbf { s } _ { j } , \mathbf { q } _ { j }$ , and $\mathbf { c } _ { j }$ represent the center, opacity, scale, rotation, and spherical-harmonic coefficients of $\mathbf { g } _ { j }$ (Kerbl et al., 2023).

Pixel-aligned prediction. For each pixel $\mathbf { u } ~ = ~ ( x , y )$ in image I<sub>i</sub>, pixel-aligned methods (Chen et al., 2024) predict a depth z and obtain its Gaussian center µ by camera unprojection $\mu \mathbf { \Sigma } =$ $\mathrm { U n p r o j } ( \mathbf { u } , z , \pmb { \pi } _ { i } )$ , where Unproj(·) indicates the 2D to 3D unprojection. Since each pixel is mapped to one Gaussian, the total number of Gaussians is $N = N _ { I } \times \mathbf { \bar { } } H \times W$ and therefore grows with both the input resolution and the number of views.

Non-pixel-aligned prediction. Non-pixel-aligned methods (Ren et al., 2026; Li et al., 2026b) aggregate the input images into multi-view features $\mathbf { F } _ { \mathrm { i m g } }$ and use $N _ { Q }$ learnable tokens Q to directly decode Gaussians as $\mathbf { \bar { \mathcal { G } } } = f ( \mathbf { F } _ { \mathrm { i m g } } , \mathbf { Q } )$ , where $N = \breve { N } _ { Q } \times N _ { \mathscr { G } / Q }$ and $N _ { \mathcal { G } / Q }$ denotes the number of Gaussians predicted per token. This fixed-count formulation enables efficient prediction, but discards the geometric priors that have proven effective in pixel-aligned models.

![](images/fbf1ea1e209fd58d25948a9ce42bbc0ff9b00034e8fdcead547985e0e82a94ee.jpg)  
Figure 3: Pipeline of GeoGSplat. Image features are encoded in parallel into Anchors and Triplane. Anchors comprise sparse 3D spheres with learnable centers and adaptive radii, accounting for the coarse scene structure; Triplane consists of three orthogonal 2D feature maps that encode finegrained high-dimensional scene features. Anchor-to-Point Decoding (blue) gathers Triplane features within each Anchor region to predict offsets and recover dense 3D points; Point-to-Gaussian Decoding (orange) re-queries the Triplane at these points to predict the corresponding Gaussian primitives.

To retain a fixed Gaussian count while keeping explicit geometric priors, we introduce GeoGSplat, a geometry-grounded non-pixel-aligned prediction paradigm. As shown in Fig. 3, GeoGSplat first constructs M Anchors A and a Triplane $\tau$ from $\bar { \mathbf { F } } _ { \mathrm { i m g } }$ through a dual-branch structured encoder. Subsequently, a geometry-guided hierarchical decoder is proposed to expand each Anchor into K dense 3D points and decode the corresponding K Gaussians. As M and $\bar { K }$ are fixed, $N = M \times K$ remains independent of $( N _ { I } , H , W )$

## 3.2 DUAL-BRANCH STRUCTURED ENCODER

Image Feature Extraction. For each view, RGB and their corresponding Plucker ray (Plucker,¨ 1865) embeddings are summed and processed by an image encoder, implemented as a standard Vision Transformer (Dosovitskiy et al., 2021):

$$
\mathbf { F } _ { \mathrm { i m g } } = \operatorname { E n c o d e r } \left( \left[ \operatorname { P E } _ { \mathrm { r g b } } ( \mathbf { I } _ { i } ) + \operatorname { P E } _ { \mathrm { r a y } } ( \operatorname { R a y } ( \boldsymbol { \pi } _ { i } ) ) \right] _ { i = 1 } ^ { N _ { I } } \right) , \quad \mathbf { F } _ { \mathrm { i m g } } \in \mathbb { R } ^ { N _ { I } n _ { p } \times C } .\tag{2}
$$

where $n _ { p }$ is the number of patches per view, $\mathrm { P E } _ { \mathrm { r g b } } ( \cdot )$ and $\mathrm { P E } _ { \mathrm { r a y } } ( \cdot )$ indicate standard ViT patch embedding modules applied to RGB patches and Plucker rays, respectively.¨

Dual-Branch Feature Aggregation. To extract features for Anchor and Triplane construction, we introduce two sets of learnable queries, denoted as $\mathbf { Q } ^ { A } \in \mathbb { R } ^ { M \times C }$ and $\mathbf { Q } ^ { T } \in \mathbb { R } ^ { n _ { T } \times C }$ , respectively, where $n _ { T } = 3 S _ { 0 } ^ { 2 }$ and $S _ { 0 }$ is the initial plane resolution. For the Triplane branch, we additionally introduce spatial positional embeddings $\mathbf { \bar { P } } ^ { T } \in \mathbb { R } ^ { n _ { T } \times C }$ to encode the 2D grid coordinates and identities of the three orthogonal planes. The two sets of queries aggregate the image features through a shared attention module $\mathrm { A t t } ( \cdot )$ (Carion et al., 2020) as

$$
\mathbf { F } ^ { A } = \operatorname { A t t } ( \mathbf { Q } ^ { A } , \mathbf { F } _ { \mathrm { i m g } } ) , \mathbf { F } ^ { A } \in \mathbb { R } ^ { M \times C } , \mathbf { F } ^ { T } = \operatorname { A t t } ( \mathbf { Q } ^ { T } + \mathbf { P } ^ { T } , \mathbf { F } _ { \mathrm { i m g } } ) , \mathbf { F } ^ { T } \in \mathbb { R } ^ { n _ { T } \times C } .\tag{3}
$$

The resulting $\mathbf { F } ^ { A }$ and $\mathbf { F } ^ { T }$ are subsequently used to construct Anchors and Triplane.

Anchor Construction. To provide coarse geometric constraints on the scene structure, we construct M Anchors $\mathbf { \mathcal { A } } = \{ ( \mathbf { p } _ { i } , r _ { i } ) \} _ { i = 1 } ^ { \mathbf { \dot { M } } }$ from $\mathbf { F } ^ { A }$ . Each Anchor is represented by a 3D sphere with center $\mathbf { p } _ { i }$ and adaptive radius $r _ { i } .$ . Specifically, we initialize the Anchor centers from a set of reference points $\mathcal { Z } = \{ \bar { \mathbf { z } _ { i } } \} _ { i = 1 } ^ { M }$ with broad spatial coverage, then refine them according to $\mathbf { F } ^ { A }$ . The reference points $\mathcal { Z }$ are randomly sampled once from the normalized 3D volume and kept fixed throughout training. We predict an offset from $\mathbf { F } _ { i } ^ { A }$ with a linear projection Linear $\cdot ( \cdot )$ . The offset is then applied to the corresponding reference point $\mathbf { z } _ { i }$ to obtain the center $\mathbf { p } _ { i } \colon$

$$
\mathbf { p } _ { i } = \operatorname { t a n h } \bigl ( \mathbf { z } _ { i } + \mathrm { L i n e a r } ( \mathbf { F } _ { i } ^ { A } ) \bigr ) , \qquad \mathbf { p } _ { i } \in ( - 1 , 1 ) ^ { 3 } ,\tag{4}
$$

where $\operatorname { t a n h } ( { \cdot } )$ keeps the predicted centers within the normalized scene volume. Given the refined Anchor centers, we further determine their spatial extents using adaptive radii to accommodate the non-uniform spatial distribution of 3D scene content. We initialize the radius $r _ { i }$ using the median distance from $\mathbf { p } _ { i }$ to the four nearest Anchor centers. The radius is then adjusted with a bounded factor predicted from $\mathbf { F } _ { i } ^ { A }$ as:

$$
r _ { i } = d _ { i } \big ( \gamma _ { \operatorname* { m i n } } + \left( \gamma _ { \operatorname* { m a x } } - \gamma _ { \operatorname* { m i n } } \right) \sigma \big ( \mathrm { L i n } \mathbf { e } \mathbf { a } \mathbf { r } \big ( \mathbf { F } _ { i } ^ { A } \big ) \big ) \big ) , \qquad r _ { i } \in \mathbb { R } _ { + } ,\tag{5}
$$

where $d _ { i }$ defines the median distance, $( \gamma _ { \mathrm { m i n } } , \gamma _ { \mathrm { m a x } } )$ bound the learned multiplier, and $\sigma ( \cdot )$ is the sigmoid function. $d _ { i }$ adapts the Anchor extent to the spatial distribution of scene geometry, while the learned factor makes it responsive to image content. Together, these Anchors form a coarse 3D representation of the scene observed in the input images.

Triplane Construction. To complement the coarse structure with fine-grained features, we build Triplane from $\mathbf { F } ^ { T }$ through progressive refinement. First, we reshape $\mathbf { F } ^ { T } \in \mathbb { R } ^ { 3 S _ { 0 } ^ { 2 } \times C }$ to an initial Triplane $\mathbf { T } _ { S _ { 0 } } \in \mathbb { R } ^ { 3 \times S _ { 0 } \times S _ { 0 } \times C }$ . Two refinement stages are then applied to increase the plane resolution from $S _ { 0 }$ to $S = 4 S _ { 0 }$ . Each refinement comprises bilinear upsampling by a factor of two $\mathrm { { U p } _ { 2 } ( \cdot ) }$ and a Triplane refinement block $\mathcal { R } _ { s } ( \cdot , \cdot )$

$$
\mathbf { T } _ { 2 s } = \mathcal { R } _ { s } ( \mathrm { U p } _ { 2 } ( \mathbf { T } _ { s } ) , \mathbf { F } _ { \mathrm { i m g } } ) , \qquad \mathbf { T } _ { 2 s } \in \mathbb { R } ^ { 3 \times 2 s \times 2 s \times C ^ { ' } } ,\tag{6}
$$

where s and $C ^ { ' }$ denote the plane resolution and feature dimension, respectively. $\mathcal { R } _ { s } ( \cdot , \cdot )$ combines residual convolution (He et al., 2016), cross-plane feature fusion (Wang et al., 2023), and imageconditioned cross-attention. The final Triplane is defined as $\mathcal { T } = \{ \hat { \mathbf { T } } ^ { x y } , \check { \mathbf { T } } ^ { x z } , \mathbf { T } ^ { y z } \} , \mathbf { T } \overset { \prime \prime } { \in } \mathbb { R } ^ { S \times S \times \check { C } _ { T } }$ Given 3D coordinates $\mathbf { x } \in [ - 1 , 1 ] ^ { 3 }$ , we project them onto each plane and sample features at the corresponding 2D locations. The Triplane features $\phi ( \mathbf { x } )$ are then obtained via concatenation. Such a query process is defined as

$$
\phi _ { a } ( \mathbf { x } ) = \mathrm { B i l i n e a r } ( \mathbf { T } ^ { a } , \mathrm { P r o j } ( \mathbf { x } ) ) , \quad \phi ( \mathbf { x } ) = \left[ \phi _ { x y } ( \mathbf { x } ) ; \phi _ { x z } ( \mathbf { x } ) ; \phi _ { y z } ( \mathbf { x } ) \right] \in \mathbb { R } ^ { 3 C _ { T } } .\tag{7}
$$

where $a \in \{ x y , x z , y z \}$ identifies a plane, Proj(·) maps the 3D coordinates to 2D coordinates on a plane, and $\mathrm { B i l i n e a r } ( \cdot , \cdot )$ samples the corresponding features by bilinear interpolation. The queried features are utilized in the following decoder for reconstruction and Gaussian primitive prediction.

## 3.3 GEOMETRY-GUIDED HIERARCHICAL DECODER

To enforce geometric priors throughout decoding, we propose a hierarchical decoder with two stages: Anchor-to-Point decoding reconstructs dense 3D geometry, and Point-to-Gaussian decoding builds Gaussian primitives upon the recovered geometry for appearance modeling.

Anchor-to-Point Decoding. To decode dense points within each Anchor, we initialize reference points and then refine them using scene features from the Triplane. Similar to Anchor construction, these points are randomly initialized once from the normalized 3D volume and kept fixed throughout training. For each Anchor, the reference points $\bar { \mathbf { X } } _ { i }$ are established within its local region defined by $r _ { i } .$ . We then query the Triplane to obtain geometry features as $\mathbf { F } _ { i } ^ { \mathrm { g e o } } = \phi ( \bar { \mathbf { X } } _ { i } ) \in \breve { \mathbb { R } } ^ { K \times 3 C _ { T } }$ , where $\phi ( \cdot )$ denotes the Triplane sampling function in Eq. (7). To refine $\bar { \mathbf { X } } _ { i }$ based on $\mathbf { F } _ { i } ^ { \mathrm { g e o } }$ , we introduce K learnable tokens $\mathbf { Q } ^ { \mathrm { { o f f } } }$ and apply cross-attention to predict coordinate offsets:

$$
\Delta { \bf X } _ { i } = \mathrm { C r o s s A t t } \left( { \bf Q } _ { i } ^ { \mathrm { o f f } } , { \bf F } _ { i } ^ { \mathrm { g e o } } , { \bf F } _ { i } ^ { \mathrm { g e o } } \right) .\tag{8}
$$

We scale the offsets by $r _ { i }$ and obtain K points $\mathbf { X } _ { i }$ as $\mathbf { X } _ { i } = \bar { \mathbf { X } } _ { i } + r _ { i } \Delta \mathbf { X } _ { i }$ , where $\mathbf { X } _ { i } \in \mathbb { R } ^ { K \times 3 }$ . The final dense points are defined as $\mathcal { P } = [ \bar { \mathbf { X } _ { 1 } } ; . . . ; \mathbf { X } _ { M } ] \in \mathbb { R } ^ { N \times 3 }$ with $N = M \times K$

Point-to-Gaussian Decoding. To further decode G from the dense points $\mathcal { P }$ , we start by querying $\tau$ at each refined point $\mathbf { x } _ { i } \in \mathcal { P }$ . A shared MLP is then employed to produce the Gaussian features as $\begin{array} { r } { \mathbf { z } _ { j } = \mathrm { M L P } ( [ \mathbf { x } _ { j } ; \boldsymbol { \phi } ( \mathbf { x } _ { j } ) ] ^ { * } , } \end{array}$ ). The corresponding $\mathbf { g } _ { j }$ is predicted from $\mathbf { z } _ { j }$ through a Gaussian head as

$$
( \Delta \pmb { \mu } _ { j } , \alpha _ { j } , \mathbf { s } _ { j } , \mathbf { q } _ { j } , \mathbf { c } _ { j } ) = \mathrm { H e a d } _ { G } ( \mathbf { z } _ { j } ) .\tag{9}
$$

The final Gaussian center is computed as $\pmb { \mu } _ { j } = \mathbf { x } _ { j } + \Delta \pmb { \mu } _ { j }$

## 3.4 TRAINING LOSS

The geometric priors provided by the aforementioned Anchors and dense points allow us to enhance the supervision by utilizing ground-truth geometry. Therefore, we employ a geometry loss defined as the symmetric Chamfer Distance $\mathrm { C D } ( \cdot , \cdot )$

$$
\mathcal { L } _ { \mathrm { g e o } } = \mathrm { C D } ( \mathcal { P } _ { A } , \mathcal { P } _ { \mathrm { g t } } ^ { A } ) + \mathrm { C D } ( \mathcal { P } , \mathcal { P } _ { \mathrm { g t } } ) ,\tag{10}
$$

where $\mathcal { P } _ { A }$ denotes the Anchor centers, P indicates the decoded dense 3D points, and $( \mathcal { P } _ { \mathrm { g t } } ^ { A } , \mathcal { P } _ { \mathrm { g t } } )$ are the ground-truth counterparts. We unproject and fuse the valid depth points from the input views. We then apply FPS (Qi et al., 2017) to obtain $\mathcal { P } _ { \mathrm { g t } } \in \mathbb { R } ^ { N \times 3 }$ and $\mathcal { P } _ { \mathrm { g t } } ^ { \dot { A } } \in \mathbb { R } ^ { M \times 3 }$ , matching the predicted points $\mathcal { P }$ and Anchor centers $\mathcal { P } _ { A }$ , respectively. This geometry supervision enables reliable scene-wide Anchor placement and local point refinement.

Furthermore, we render the predicted Gaussians at the input cameras and compute the standard photometric loss (Wang et al., 2004; Zhang et al., 2018) as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e n d e r } } = \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { s } \mathcal { L } _ { \mathrm { S S I M } } + \lambda _ { p } \mathcal { L } _ { \mathrm { L P I P S } } . } \end{array}\tag{11}
$$

The complete training loss is defined as $\mathcal { L } = \mathcal { L } _ { \mathrm { g e o } } + \mathcal { L } _ { \mathrm { r e n d e r } }$ . In practice, we first train with geometric supervision alone for 15K iterations, and then add the rendering loss. This allows the model to establish stable geometry before jointly optimizing geometry and appearance.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Implementation details. The training resolution is set to $2 5 6 \times 2 5 6$ . The image encoder contains 8 ViT blocks, and the shared attention module contains 12 attention blocks. We use M = 4096 Anchor tokens and predict $K \ : = \ : 1 6$ dense points per Anchor, yielding a fixed set of 65,536 Gaussians. Triplane is progressively upsampled from resolution 64 to 128 and then 256. We train for 320K iterations using AdamW (Loshchilov & Hutter, 2017), with a peak learning rate of $4 \times 1 0 ^ { - 4 }$ and BF16 precision. Training is performed on 128 NVIDIA L20 GPUs with a global batch size of 128. During training, the network takes 4 input images, and the renderings are produced at the input cameras, without sampling additional target views.

Datasets. We train GeoGSplat on five public datasets: ScanNet++ (Yeshwanth et al., 2023), ARKitScenes (Dehghan et al., 2021), BlendedMVS (Yao et al., 2020), CO3D (Reizenstein et al., 2021), and WildRGB-D (Xia et al., 2024). We perform evaluations on DTU (Jensen et al., 2014) and NRGBD (Azinovic et al., 2022) for object- and scene-level reconstruction, respectively. For novel´ view synthesis, each evaluation sample contains 4 novel views. We use 4 context views by default and also report results with 2 and 12 input views in the Appendix A.2.

Evaluation protocol. We evaluate NVS using PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018). The input and target resolutions are set to $2 5 6 \times 2 5 6$ for all methods except AnyS plat, whose input images are resized to $4 4 8 \times 4 4 8$ to match its input resolution requirement. 3D reconstruction is evaluated using Chamfer distances (Tenenbaum et al., 1977) from predicted points to ground truth (Pred→GT) and from ground truth to predicted points (GT→Pred), as well as thei average, denoted as CD. For GeoGSplat, these metrics are computed on the predicted dense points, while for the other methods, they are computed on the predicted Gaussian centers. Notably, the results of GeoGSplat computed on dense points and Gaussian centers are nearly identical, as reported in Appendix A.4. For a fair comparison, we define ground-truth point clouds in a normalized coordinate system. We then transform predicted point clouds from the camera coordinate system to the same coordinate system for CD computation. All evaluated methods use ground-truth camera poses for the input images, except AnySplat, which uses predicted poses by default. Notably, we also report results of AnySplat with ground-truth poses, denoted as AnySplat<sup>∗</sup>.

## 4.2 NOVEL VIEW SYNTHESIS AND GEOMETRIC RECONSTRUCTION

Tables 1 and 2 report NVS and reconstruction results with 4 input views, and Fig. 4 provides qualitative comparisons. We compare our GeoGSplat against pixel-aligned methods, including MVSplat (Chen et al., 2024), DepthSplat (Xu et al., 2025), and AnySplat (Jiang et al., 2025b). Although

Table 1: Novel view synthesis with four input views. NUM denotes the number of Gaussians used for rendering. The best PSNR, SSIM, and LPIPS are shown in bold.
<table><tr><td rowspan="2">Method</td><td colspan="4">DTU</td><td colspan="4">NRGBD</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NUM</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>NUM</td></tr><tr><td>MVSplat</td><td>14.44</td><td>0.520</td><td>0.401</td><td>158K</td><td>18.53</td><td>0.609</td><td>0.319</td><td>255K</td></tr><tr><td>DepthSplat</td><td>15.46</td><td>0.599</td><td>0.331</td><td>158K</td><td>22.98</td><td>0.792</td><td>0.170</td><td>255K</td></tr><tr><td>AnySplat</td><td>16.60</td><td>0.630</td><td>0.166</td><td>170K</td><td>20.12</td><td>0.690</td><td>0.138</td><td>643K</td></tr><tr><td>AnySplat*</td><td>17.54</td><td>0.700</td><td>0.148</td><td>166K</td><td>20.52</td><td>0.756</td><td>0.140</td><td>624K</td></tr><tr><td>VolSplat</td><td>15.86</td><td>0.602</td><td>0.306</td><td>152K</td><td>22.22</td><td>0.757</td><td>0.216</td><td>246K</td></tr><tr><td>TokenGS</td><td>17.71</td><td>0.643</td><td>0.354</td><td>65K</td><td>23.82</td><td>0.844</td><td>0.173</td><td>65K</td></tr><tr><td>Ours</td><td>20.48</td><td>0.771</td><td>0.137</td><td>65K</td><td>24.78</td><td>0.847</td><td>0.120</td><td>65K</td></tr></table>

AnySplat aggregates Gaussians through voxelization, it relies on per-pixel depth prediction, and the Gaussian count is still coupled to both image resolution and the number of input views. We also compare against non-pixel-aligned methods TokenGS (Ren et al., 2026) and VolSplat (Wang et al., 2026).

Novel view synthesis. Under the same Gaussian budget, GeoGSplat substantially improves NVS quality over TokenGS on both datasets. On DTU, GeoGSplat improves PSNR from 17.71 to 20.48 dB and reduces LPIPS from 0.354 to 0.137. On NRGBD, PSNR improves from 23.82 to 24.78 dB and LPIPS decreases from 0.173 to 0.120. Notably, even compared with pixel-aligned approaches that predict considerably more Gaussians, GeoGSplat still achieves the best PSNR, SSIM, and LPIPS. For example, AnySplat<sup>∗</sup> uses 624K Gaussians on NRGBD, nearly 10× more than GeoGSplat, yet attains lower NVS quality.

Table 2: 3D reconstruction with four input views. Pred→GT and GT→Pred denote Chamfer distances from predicted points to ground-truth points and vice versa, respectively. CD indicates their average. The best results are shown in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">DTU</td><td colspan="3">NRGBD</td></tr><tr><td>Pred→GT↓</td><td>GT→Pred↓</td><td>CD↓</td><td>Pred→GT↓</td><td>GT→Pred↓</td><td>CD↓</td></tr><tr><td>MVSplat</td><td>0.271</td><td>0.061</td><td>0.166</td><td>0.091</td><td>0.080</td><td>0.085</td></tr><tr><td>DepthSplat</td><td>0.130</td><td>0.066</td><td>0.098</td><td>0.093</td><td>0.058</td><td>0.076</td></tr><tr><td>AnySplat</td><td>0.048</td><td>0.032</td><td>0.040</td><td>0.063</td><td>0.046</td><td>0.055</td></tr><tr><td>AnySplat*</td><td>0.033</td><td>0.016</td><td>0.025</td><td>0.046</td><td>0.038</td><td>0.042</td></tr><tr><td>VolSplat</td><td>0.091</td><td>0.035</td><td>0.063</td><td>0.061</td><td>0.059</td><td>0.060</td></tr><tr><td>TokenGS</td><td>0.108</td><td>0.057</td><td>0.082</td><td>0.515</td><td>0.039</td><td>0.277</td></tr><tr><td>Ours</td><td>0.019</td><td>0.016</td><td>0.018</td><td>0.017</td><td>0.017</td><td>0.017</td></tr></table>

3D reconstruction. The NVS improvement is accompanied by significantly more accurate geometric reconstruction. As shown in Table 2, GeoGSplat reduces CD from 0.082 to 0.018 on DTU and from 0.277 to 0.017 on NRGBD compared with TokenGS. It also achieves the lowest CD among all compared methods on both datasets. Notably, TokenGS obtains a Pred→GT distance of 0.515 despite a GT→Pred distance of only 0.039 on NRGBD, indicating that many predicted centers are displaced from the ground-truth geometry. GeoGSplat reduces both directional distances to 0.017. Compared with pixel-aligned baselines, such as AnySplat<sup>∗</sup>, GeoGSplat also achieves lower CD, demonstrating accurate reconstruction without relying on the expensive pixel-aligned predictions.

Qualitative comparison. Fig. 4 provides a visual comparison of the synthesized novel views. TokenGS (Ren et al., 2026) produces blurred object markings on DTU and blurred textures on NRGBD. AnySplat<sup>∗</sup> (Jiang et al., 2025b) yields sharper renderings, but suffers from duplicated or distorted structures, particularly on NRGBD, due to inconsistent depth predictions across views. In contrast, GeoGSplat preserves both fine appearance details and structurally faithful scene geometry. Such observations show the effectiveness of our geometry-guided non-pixel-aligned paradigm. Our explicit geometric constraints improve geometry over TokenGS, reducing blurred appearance, and non-pixel-aligned prediction implicitly aligns information across views, avoiding duplicated structures caused by the inconsistent depth.

GT  
AnySplat∗  
TokenGS  
Ours  
GT  
AnySplat∗  
TokenGS  
![](images/f43cb372b7c0c183b434a4b42ca302a7e189d1c0e2126a2d0d46512021ae7f24.jpg)  
Ours  
Figure 4: Qualitative novel view synthesis with four input views. Columns show ground truth, AnySplat<sup>∗</sup>, TokenGS, and GeoGSplat from left to right. Red boxes highlight blurred details in TokenGS and shifted or duplicated structures in AnySplat. In contrast, GeoGSplat preserves finer details and more accurate scene structures in these regions.

## 4.3 HOW GEOMETRY AFFECTS NOVEL VIEW SYNTHESIS

Considering that inaccurate geometry may still produce plausible renderings from viewpoints close to the input views, we examine whether the impact of geometric errors becomes more apparent under larger viewpoint changes. We keep the context views fixed and compare GeoGSplat and TokenGS along the same target-view trajectories on NRGBD. Please refer to the Appendix B.6 for more details.

![](images/bf8b44aa00a02d664e8916c86cd3eb520b769300daa7dab6620017d73a6728f1.jpg)

![](images/2e26d04754d78fce7f27451e789253ec50be4e6106c71c8d7988b4223ce1b345.jpg)  
Figure 5: NVS under varying viewpoint changes on NRGBD. The plots show PSNR (left) and the performance advantage over TokenGS (right) at different target-context distances. The performance gap between GeoGSplat and TokenGS becomes larger as the target-context distance increases.

As shown in Fig. 5, both methods achieve similar performance for target views close to the context views. As the distance increases, the performance gap between GeoGSplat and TokenGS becomes larger, with GeoGSplat maintaining higher PSNR. These results suggest that TokenGS can produce plausible renderings near the input views despite inaccurate geometry, but its geometric errors become increasingly apparent as the viewpoint changes. In contrast, GeoGSplat benefits from reliable geometry, leading to more robust NVS under larger viewpoint changes. Fig. 6 provides a detailed comparison of the recovered geometry on NRGBD. TokenGS predicts Gaussian centers that are sub stantially far from the ground-truth points, consistent with its high Pred→GT distance reported in Sec. 4.2. These outliers remain after removing low-opacity Gaussians, indicating that they are not simply caused by low-contribution primitives. Points unprojected from TokenGS rendered depth also remain displaced from the ground truth. In contrast, GeoGSplat reconstructs points that are much closer to the ground-truth geometry.

## 4.4 ABLATION STUDIES

We conduct comprehensive ablations on the key components of GeoGSplat and report the results in Table 3. All variants use the same data and evaluation protocol, ensuring fair comparisons. For computational efficiency, we conduct all ablation studies using a smaller GeoGSplat model. Please refer to the Appendix B.7 for the detailed model architecture.

![](images/a849300b0cda4e5ec7e9ef786b4a34a3481fd5a3871ff8929c06ee5a1a463c92.jpg)  
Figure 6: 3D reconstruction on NRGBD. From left to right: ground truth, TokenGS Gaussian centers, opacity-filtered TokenGS centers, points unprojected from depth rendered by TokenGS, and GeoGSplat dense points. TokenGS contains substantial geometric outliers that remain after opacity filtering, while its depth-based reconstruction is also displaced from the ground truth. GeoGSplat predicts points closer to the ground-truth geometry and provides more complete scene coverage.

Table 3: Ablation studies. Full uses both representations, both decodings, and both geometry losses. The best result in each group is shown in bold.
<table><tr><td rowspan="2">Variant</td><td colspan="4">DTU</td><td colspan="4">NRGBD</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>CD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>CD↓</td></tr><tr><td>w/o Anchor</td><td>18.67</td><td>0.700</td><td>0.188</td><td>0.031</td><td>22.46</td><td>0.782</td><td>0.185</td><td>0.030</td></tr><tr><td>w/o Triplane</td><td>18.63</td><td>0.663</td><td>0.253</td><td>0.031</td><td>22.47</td><td>0.733</td><td>0.229</td><td>0.031</td></tr><tr><td>Full</td><td>19.79</td><td>0.728</td><td>0.183</td><td>0.019</td><td>23.70</td><td>0.798</td><td>0.164</td><td>0.019</td></tr><tr><td>w/o Anchor-to-Point Decoding</td><td>18.32</td><td>0.653</td><td>0.276</td><td>0.032</td><td>21.81</td><td>0.703</td><td>0.276</td><td>0.041</td></tr><tr><td>w/o Point-to-Gaussian Decoding</td><td>18.71</td><td>0.720</td><td>0.186</td><td>0.031</td><td>22.35</td><td>0.768</td><td>0.189</td><td>0.031</td></tr><tr><td>Full</td><td>19.79</td><td>0.728</td><td>0.183</td><td>0.019</td><td>23.70</td><td>0.798</td><td>0.164</td><td>0.019</td></tr><tr><td>w/o Geometry Losses</td><td>13.60</td><td>0.518</td><td>0.343</td><td>0.218</td><td>18.53</td><td>0.600</td><td>0.301</td><td>0.135</td></tr><tr><td>w/o Dense-Point Loss</td><td>18.76</td><td>0.705</td><td>0.187</td><td>0.026</td><td>23.42</td><td>0.791</td><td>0.166</td><td>0.025</td></tr><tr><td>w/o Anchor Loss</td><td>19.60</td><td>0.724</td><td>0.178</td><td>0.020</td><td>23.82</td><td>0.801</td><td>0.161</td><td>0.022</td></tr><tr><td>Full</td><td>19.79</td><td>0.728</td><td>0.183</td><td>0.019</td><td>23.70</td><td>0.798</td><td>0.164</td><td>0.019</td></tr></table>

Dual-Branch Structured Encoder. We first investigate the effectiveness of the dual-branch encoder. w/o Anchor and w/o Triplane remove the Anchor and Triplane constructions, respectively, and directly predict points and Gaussians from $\mathbf { F } ^ { A }$ and $\mathbf { F } ^ { T }$ . The two variants incur noticeable performance drops on DTU and NRGBD. In particular, PSNR drops by more than 1 dB on both datasets. These results validate the complementary roles of Anchors and Triplane in our encoder.

Geometry-Guided Hierarchical Decoder. We next examine the progressive decoding process. w/o Anchor-to-Point Decoding directly regresses dense points from $\mathbf { \dot { F } } ^ { A }$ without querying Triplane. w/o Point-to-Gaussian Decoding removes the query process at the Gaussian prediction stage and predicts Gaussians from $\mathbf { F } ^ { A }$ . Both variants perform worse than Full, demonstrating the benefit of progressively refining predictions through the two-stage Triplane querying mechanism.

Geometry Supervision. Since our framework is grounded in explicit geometric priors, we further investigate the effect of geometric supervision. Removing the geometry losses causes severe performance drops, demonstrating the importance of geometric supervision for maintaining reliable geometric priors in GeoGSplat. Removing either loss also degrades performance, with the dense-point loss having a larger impact on both NVS and geometry. Full achieves the lowest CD, showing that supervising both Anchor centers and dense points contributes to accurate geometric reconstruction.

## 5 CONCLUSION

We presented GeoGSplat, a geometry-grounded non-pixel-aligned framework for feed-forward Gaussian Splatting that maintains a fixed Gaussian budget while incorporating explicit geometric priors. GeoGSplat combines a dual-branch structured encoder, which represents the scene through complementary Anchors and Triplane, with a geometry-guided hierarchical decoder that progressively reconstructs dense 3D points before predicting Gaussian primitives. Experiments on DTU and NRGBD demonstrate consistent improvements over both pixel-aligned and non-pixel-aligned baselines in NVS and 3D reconstruction, with the advantage becoming more significant under large viewpoint changes. These results highlight the importance of explicit geometric grounding for robust and generalizable feed-forward Gaussian Splatting.

## REFERENCES

Dejan Azinovic, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies.´ Neural rgb-d surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6280–6291, 2022.

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5460–5469, 2022.

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European Conference on Computer Vision, pp. 213–229, 2020.

Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16123–16133, 2022.

David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19457–19467, 2024.

Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 14104–14113, 2021.

Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, pp. 370–386, 2024.

Afshin Dehghan, Gilad Baruch, Zhuoyuan Chen, Yuri Feigin, Peter Fu, Thomas Gebauer, Daniel Kurz, Tal Dimry, Brandon Joffe, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. NeurIPS Datasets and Benchmarks, 2(6):16, 2021.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations, 2021.

Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5491–5500, 2022.

Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong Tan, and Ping Tan. Cascade cost volume for high-resolution multi-view stereo and stereo matching. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2492–2501, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. In International Conference on Learning Representations, volume 2024, pp. 50678–50702, 2024.

Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. Large scale multiview stereopsis evaluation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 406–413, 2014.

Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, et al. Rayzer: A self-supervised large view synthesis model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4918–4929, 2025a.

Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on Graphics, 44(6):1–16, 2025b.

Mohammad Mahdi Johari, Yann Lepoittevin, and Franc¸ois Fleuret. Geonerf: Generalizing nerf with geometry priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18344–18347, 2022.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, George Drettakis, et al. 3d gaussian splat-¨ ting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139–1, 2023.

Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud. Grounding image matching in 3d with mast3r. ˆ In European Conference on Computer Vision, pp. 71–91, 2024.

Wenyu Li, Sidun Liu, Tongrui Hu, Peng Qiao, and Yong Dou. Locusgs: Spatially grounded tokens for feed-forward 3d gaussian splatting. arXiv preprint arXiv:2608.12825, 2026a.

Yinglong Li, Donghui Shen, Xiaoyu Zhang, Zhichao Ye, Hongyu Wu, Aimin Hao, Guofeng Zhang, and Haomin Liu. Querysplat: Decoupling geometry and appearance representations in 3dgs prediction. arXiv preprint arXiv:2608.01186, 2026b.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20654–20664, 2024.

Arthur Moreau, Richard Shaw, Michal Nazarczuk, Jisu Shin, Thomas Tanay, Zhensong Zhang, Songcen Xu, and Eduardo Perez-Pellitero. Off the grid: Detection of primitives for feed-forward´ 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11756–11766, 2026.

Julius Plucker. I. on a new geometry of space. Proceedings of the Royal Society of London, pp. 53–58, 1865.

Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017.

Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10881–10891, 2021.

Jiawei Ren, Michal Tyszkiewicz, Jiahui Huang, and Zan Gojcic. Tokengs: Decoupling 3d gaussian prediction from pixels with learnable tokens. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10208–10217, 2024.

RC Bolles HG Barrow JM Tenenbaum, RC Bolles, HC Wolf, et al. Parametric correspondence and chamfer matching: Two new techniques for image matching. In Proceedings ofthe International Joint Conference on Artificial Intelligence, pp. 659–663, 1977.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5294–5306, 2025.

Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multiview image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4690–4699, 2021.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20697–20709, 2024.

Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4563–4573, 2023.

Weijie Wang, Yeqing Chen, Zeyu Zhang, Hengyu Liu, Haoxiao Wang, Zhiyuan Feng, Wenkang Qin, Feng Chen, Jia-Wang Bian, Zheng Zhu, et al. Volsplat: Rethinking feed-forward 3d gaussian splatting with voxel-aligned prediction. In European Conference on Computer Vision, pp. 1–19, 2026.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600– 612, 2004.

Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22378–22389, 2024.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14419–14429, 2026.

Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16453–16463, 2025.

Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In European Conference on Computer Vision, pp. 785–801, 2018.

Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1787–1796, 2020.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 12–22, 2023.

Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4576–4585, 2021.

Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, pp. 1–19, 2024.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 586–595, 2018.

Xiaoxue Zhang, Xiaoxu Zheng, Yixuan Yin, Tiao Zhao, Kaihua Tang, Michael Bi Mi, Zhan Xu, and Dave Zhenyu Chen. Anchorsplat: Feed-forward 3d gaussian splatting with 3d geometric priors. arXiv preprint arXiv:2604.07053, 2026.

Qitao Zhao, Hao Tan, Qianqian Wang, Sai Bi, Kai Zhang, Kalyan Sunkavalli, Shubham Tulsiani, and Hanwen Jiang. E-rayzer: Self-supervised 3d reconstruction as spatial visual pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7525–7535, 2026.

## A MORE EXPERIMENTS

## A.1 COMPARISON WITH RETRAINED TOKENGS

The evaluation of TokenGS in Sec. 4.2 uses the publicly released model, which differs from GeoGSplat in training data. To further isolate the effect of our geometry-grounded design, we retrain TokenGS under the same setting as GeoGSplat. Both variants are trained under the same training data, input resolution, Gaussian budget, and training schedule. TokenGS-R uses the original photometric loss, while TokenGS-R + Geometry loss additionally uses the geometry loss on the Gaussian centers. The results are reported in Table 4.

Table 4: Comparison with retrained TokenGS under four input views. Both TokenGS variants use the same setting as GeoGSplat and differ only in whether the geometry loss is applied.
<table><tr><td rowspan="2">Variant</td><td colspan="4">DTU</td><td colspan="4">NRGBD</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>CD↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>CD↓</td></tr><tr><td>TokenGS-R</td><td>13.75</td><td>0.571</td><td>0.386</td><td>0.277</td><td>22.48</td><td>0.728</td><td>0.234</td><td>0.180</td></tr><tr><td>TokenGS-R + Geometry loss</td><td>14.41</td><td>0.608</td><td>0.304</td><td>0.044</td><td>24.17</td><td>0.797</td><td>0.158</td><td>0.029</td></tr><tr><td>GeoGSplat</td><td>20.48</td><td>0.771</td><td>0.137</td><td>0.018</td><td>24.78</td><td>0.847</td><td>0.120</td><td>0.017</td></tr></table>

Both retrained TokenGS variants perform substantially worse than GeoGSplat on both datasets. Adding the geometry loss improves both NVS and geometric accuracy, but a substantial gap remains. The results indicate that the gains of GeoGSplat do not simply come from adding training data or geometric supervision, but from the geometry-grounded architecture itself.

## A.2 GENERALIZATION ACROSS INPUT VIEW COUNTS

We extend the evaluation by testing the methods on 2 and 12 input views without retraining, and report the results in Table 5. With 2 input views, GeoGSplat achieves the highest PSNR among the compared methods on both datasets, and the lowest CD on NRGBD. With 12 input views, GeoGSplat maintains its leading performance, further improving PSNR to 22.53 dB on DTU and 25.21 dB on NRGBD, and reducing CD to 0.014 and 0.015, respectively. It is worth noting that GeoGSplat keeps a fixed set of 65K Gaussians for both input settings. These results demonstrate that GeoGSplat can generalize across varying numbers of input views without retraining, while maintaining accurate geometry and high-quality NVS.

Table 5: Evaluation under varying input view counts on DTU and NRGBD. All methods are tested with 2 and 12 input views without retraining. We report PSNR (P) for novel view synthesis, Chamfer distance (CD) for 3D reconstruction, and the number of predicted Gaussians (NUM).  
(a) DTU  
(b) NRGBD
<table><tr><td rowspan="2">Method</td><td colspan="3">2 Views</td><td colspan="3">12 Views</td><td rowspan="2">Method</td><td colspan="3">2 Views</td><td colspan="3">12 Views</td></tr><tr><td>P↑</td><td>CD↓</td><td>NUM</td><td>P↑</td><td>CD↓</td><td>NUM</td><td>P↑</td><td>CD↓</td><td>NUM</td><td>P↑</td><td>CD↓</td><td>NUM</td></tr><tr><td>DepthSplat</td><td>14.01</td><td>0.224</td><td>78K</td><td>17.06</td><td>0.035</td><td>480K</td><td>DepthSplat</td><td>20.16</td><td>0.102</td><td>127K</td><td>22.12</td><td>0.090</td><td>766K</td></tr><tr><td>AnySplat*</td><td>15.94</td><td>0.034</td><td>91K</td><td>18.71</td><td>0.019</td><td>352K</td><td>AnySplat</td><td>17.28</td><td>0.045</td><td>325K</td><td>22.37</td><td>0.035</td><td>1665K</td></tr><tr><td>TokenGS</td><td>14.57</td><td>0.155</td><td>65K</td><td>18.30</td><td>0.073</td><td>65K</td><td>TokenGS</td><td>21.84</td><td>0.330</td><td>65K</td><td></td><td>23.900.250</td><td>65K</td></tr><tr><td>Ours</td><td>17.53</td><td>0.037</td><td>65K</td><td>22.53</td><td>0.014</td><td>65K</td><td>Ours</td><td>22.01 0.024</td><td></td><td>65K</td><td></td><td>25.21 0.015</td><td>65K</td></tr></table>

## A.3 EXPERIMENTS ON MIP-NERF 360

We conducted additional experiments on Mip-NeRF 360 (Barron et al., 2022), a large-scale scenelevel dataset. As shown in Table 6, GeoGSplat achieves the best PSNR of 21.59 dB and the lowest CD of 0.015 on Mip-NeRF 360, showing that our geometry-grounded design generalizes well to large-scale scenes.

Table 6: Comparison on Mip-NeRF 360 with 12 input views. The best results are shown in bold.
<table><tr><td></td><td>MVSplat</td><td>DepthSplat</td><td> $\mathrm { \ A n y { S p l a t } ^ { * } }$ </td><td>VolSplat</td><td>TokenGS</td><td>Ours</td></tr><tr><td>PSNR↑</td><td>16.24</td><td>16.06</td><td>18.65</td><td>19.79</td><td>21.51</td><td>21.59</td></tr><tr><td>CD↓</td><td>0.106</td><td>0.063</td><td>0.081</td><td>0.058</td><td>0.132</td><td>0.015</td></tr></table>

## A.4 CHAMFER DISTANCE COMPARISON BETWEEN DENSE POINTS AND GAUSSIAN CENTERS

We compare the Chamfer distance computed on the dense points and the final Gaussian centers after Point-to-Gaussian decoding. As shown in Table 7, the two measurements are nearly identical across all settings. The predicted Gaussian offsets adjust Gaussian centers for better rendering quality, while preserving the high-quality geometry reconstructed by the dense points.

Table 7: Consistency between dense points and Gaussian centers. We report Chamfer distance computed on the dense points from Anchor-to-Point decoding and the final Gaussian centers after Point-to-Gaussian decoding.
<table><tr><td>Dataset</td><td>Views</td><td>Dense Points CD↓</td><td>Gaussian Centers CD↓</td></tr><tr><td>DTU</td><td>2V</td><td>0.037148</td><td>0.037130</td></tr><tr><td>DTU</td><td>4V</td><td>0.017984</td><td>0.017879</td></tr><tr><td>DTU</td><td>12V</td><td>0.014296</td><td>0.014189</td></tr><tr><td>NRGBD</td><td>2V</td><td>0.023718</td><td>0.023623</td></tr><tr><td>NRGBD</td><td>4V</td><td>0.017406</td><td>0.017485</td></tr><tr><td>NRGBD</td><td>12V</td><td>0.015325</td><td>0.015472</td></tr></table>

## A.5 ADDITIONAL VISUALIZATIONS

We provide more visualizations in Fig. 7, where we can observe that our GeoGSplat reconstructs cleaner Gaussian centers and renders higher-quality images than TokenGS.

## B IMPLEMENTATION DETAILS

## B.1 SHARED ATTENTION MODULE

The attention module $\mathrm { A t t } ( \cdot )$ used in Eq. 3 consists of a cross-attention layer, a self-attention layer, and a feed-forward network (FFN). Given query tokens Q and image features $\mathbf { F } _ { \mathrm { i m g } } .$ , we first perform cross-attention to aggregate image features:

$$
\mathbf { H } _ { 1 } = \mathbf { Q } + \mathrm { M H A } \left( \mathbf { Q } , \mathbf { F } _ { \mathrm { i m g } } , \mathbf { F } _ { \mathrm { i m g } } \right) ,\tag{12}
$$

where $\mathrm { M H A } ( \cdot , \cdot , \cdot )$ denotes multi-head attention. We then apply self-attention among the query tokens:

$$
\mathbf { H } _ { 2 } = \mathbf { H } _ { 1 } + \mathrm { M H A } \left( \mathbf { H } _ { 1 } , \mathbf { H } _ { 1 } , \mathbf { H } _ { 1 } \right) .\tag{13}
$$

The output is further refined by an FFN:

$$
\mathbf { H } _ { 3 } = \mathbf { H } _ { 2 } + \mathrm { F F N } \left( \mathbf { H } _ { 2 } \right) ,\tag{14}
$$

where FFN(·) consists of two linear layers. The same attention module is shared between the Anchor and Triplane branches, while the two branches use independent query tokens.

## B.2 TRIPLANE REFINEMENT BLOCK

Each refinement stage in Eq. (6) consists of three operations: a residual convolution that refines each plane locally, a cross-plane fusion that exchanges information between the three planes, and an image-conditioned cross-attention that re-injects image evidence at the new resolution.

![](images/065148610c2555e7549c0d4fb8907a03ee62177967d396834682c32bd941d678.jpg)  
Figure 7: Comparison of reconstruction and novel view synthesis.

First, the Triplane resolution is doubled, followed by two residual convolutional blocks:

$$
{ \bf Z } ^ { a } \gets { \bf Z } ^ { a } + \mathrm { C o n v } _ { 3 \times 3 } \left( \mathrm { C o n v } _ { 3 \times 3 } ( { \bf Z } ^ { a } ) \right) ,\tag{15}
$$

where $\mathbf { Z } ^ { a }$ denotes the feature map of plane $a \in \{ x y , x z , y z \}$ , and $\mathrm { C o n v } _ { 3 \times 3 }$ denotes a $3 \times 3$ convolution. We then perform cross-plane fusion by exchanging features between the three orthogonal planes. For instance, the xy and xz planes share the x-axis. We average the features of the xz plane along the z-axis and replicate the resulting features along the y-axis to match the xy plane:

$$
\mathbf { P } ^ { x z  x y } = \mathrm { E x p a n d } _ { y } ( \mathrm { M e a n } _ { z } ( \mathbf { Z } ^ { x z } ) ) .\tag{16}
$$

where Mean averages along the z-axis, and Expan $\mathrm { l } _ { y }$ replicates the resulting features along the y-axis. The same operation is applied to the other pairs of orthogonal planes. The pooled features from the other two planes are concatenated with the target plane and fused by convolution:

$$
\mathbf { Z } ^ { a } \gets \mathbf { Z } ^ { a } + \operatorname { C o n v } ( [ \mathbf { Z } ^ { a } ; \mathbf { P } ^ { b  a } ; \mathbf { P } ^ { c  a } ] ) ,\tag{17}
$$

where $\{ b , c \} = \{ x y , x z , y z \} \backslash \{ a \}$ . Finally, image features are re-injected at each resolution through image-conditioned cross-attention:

$$
\mathbf { Z } \gets \mathbf { Z } + \operatorname { C r o s s A t t } \left( \mathbf { Z } , \mathbf { F } _ { \mathrm { i m g } } , \mathbf { F } _ { \mathrm { i m g } } \right) ,\tag{18}
$$

where Z stacks the features of the three planes. The refined features are reshaped into three orthogonal planes to obtain $\mathbf { T } _ { 2 s }$

Across the refinement stages, the Triplane is progressively refined from $6 4 ^ { 2 }$ to $1 2 8 ^ { 2 }$ and $2 5 6 ^ { 2 }$ , with channel dimensions of 1024, 768, and 384, respectively.

## B.3 REFERENCE INITIALIZATION

We use fixed spatial references for both Anchor center prediction and Anchor-to-Point prediction. For Anchor centers, we randomly sample M reference centers $\{ \mathbf { p } _ { i } ^ { 0 } \} _ { i = 1 } ^ { M }$ from the normalized scene

space. The reference centers are sampled once at initialization and then kept fixed throughout training.

For Anchor-to-Point prediction, we randomly sample K normalized reference offsets $\{ \delta _ { k } \} _ { k = 1 } ^ { K }$ within the Anchor region. The same set of offsets is sampled once at initialization and shared across all Anchors and training samples. The corresponding reference points are defined as

$$
{ \bf q } _ { i k } = { \bf p } _ { i } + r _ { i } \delta _ { k } .\tag{19}
$$

## B.4 DATASET-SPECIFIC VIEW SAMPLING

The five training datasets have different view organizations, and we therefore use dataset-specific sampling strategies to construct four input views. For ScanNet++ and ARKitScenes, which consist of continuous indoor camera trajectories, we first randomly select a reference view and then sample three additional views within a predefined pose-distance range, avoiding views that are either too close or too far from the reference. For BlendedMVS, we first define minimum and maximum temporal gaps, then randomly select a value within this range to determine the interval between the first and last frames; views are then randomly sampled from within this interval. For CO3D and WildRGB-D, where each sequence contains views of a single object from different viewpoints, we uniformly sample four distinct views from the same sequence.

## B.5 NORMALIZED GEOMETRIC EVALUATION

We evaluate geometry in a common normalized coordinate system defined by the context cameras and ground-truth geometry. The context camera poses determine the orientation and origin of this coordinate system, while the ground-truth point cloud determines its scale. We then transform the ground-truth point cloud and all context cameras into this frame, and normalize the scene to fit within $[ - 1 , 1 ] ^ { 3 }$ . For each method, the predicted points are transformed into the same frame using the known correspondence between its context cameras and the ground-truth context cameras. Thus, both ground-truth and predicted points are evaluated in a shared coordinate system.

## B.6 TARGET-CONTEXT DISTANCE

We use the target-context distance to characterize the viewpoint change in Sec. 4.3. For each target view, it is defined as the Euclidean distance between the target camera center and the nearest context camera center.

As the distance increases, the target view may contain regions that are unseen in all context views. These regions are not observed in the input views. Thus, we compute PSNR only on target pixels that are visible in at least one context view, so that the metric directly reflects the rendering quality of reconstructed regions.

## B.7 DETAILED MODEL ARCHITECTURE

We use two model settings in our experiments. The smaller model is used for all ablation studies to reduce computational cost, while the main-table model is used for comparison with prior methods. Their detailed architectures are summarized in Table 8.

Table 8: Architecture comparison between the smaller model used for ablations and the model used in the main tables.
<table><tr><td>Component</td><td>Ablation model</td><td>Main-table model</td></tr><tr><td>Input resolution</td><td>256×256</td><td>256×256</td></tr><tr><td>Patch size</td><td>8×8</td><td>8×8</td></tr><tr><td>Image encoder blocks</td><td>6</td><td>8</td></tr><tr><td>Shared Attention blocks</td><td>8</td><td>12</td></tr><tr><td>Triplane resolutions</td><td>64→128→256</td><td>64→128→256</td></tr><tr><td>Anchors × points per Anchor</td><td>4096×16</td><td>4096×16</td></tr><tr><td>Global batch size</td><td>64</td><td>128</td></tr></table>