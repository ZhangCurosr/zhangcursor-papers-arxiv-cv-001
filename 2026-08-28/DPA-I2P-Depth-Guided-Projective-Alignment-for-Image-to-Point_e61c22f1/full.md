# DPA-I2P: Depth-Guided Projective Alignment for Image-to-Point-Cloud Registration in Autonomous Driving

Wenxin Zhang<sup>∗</sup>, Hang Li<sup>∗</sup>, Zhiwei Xu<sup>†</sup>, Qiankun Dong<sup>∗</sup>, Gang Wang<sup>∗</sup>, Tao Li<sup>∗</sup>

<sup>∗</sup>Nankai University, Tianjin, China

wenxinzhang@mail.nankai.edu.cn, lihangws2000@gmail.com, {qiankund, ganggang, litao}@nankai.edu.cn <sup>†</sup>Haihe Lab of ITAI, Tianjin 300459, China

xuzhiwei2001@ict.ac.cn

Abstract—Image-to-Point Cloud Registration aims to estimate the camera pose of a given image within a 3D scene point cloud, which is a fundamental task in autonomous driving and large-scale outdoor localization. Recent implicit correspondence learning methods have improved registration performance by learning cross-modal alignment in an end-to-end framework, leading to more accurate camera pose estimation. However, due to the inherent modality discrepancy between images and sparse LiDAR point clouds, reliable cross-modal correspondence learning remains challenging. To address this issue, we propose Depth-Guided Projective Alignment for Image-to-Point-Cloud Registration (DPA-I2P). Unlike naive depth or feature concatenation, Ray-Conditioned Metric Depth Encoding (RMDE) and Projection-Consistent Vision Lifting (PVL) exploit depth and visual cues in a structured, geometry-aware manner. In addition, Cross-Modal Query Pruning (CQP) suppresses unreliable queries during early refinement to improve matching stability. Experiments on KITTI and nuScenes demonstrate the effectiveness of the proposed method. On KITTI, DPA-I2P reduces RTE and RRE by 45.0% and 55.6% over the strongest implicit baseline, respectively. On nuScenes, DPA-I2P also improves registration accuracy over the evaluated baselines, suggesting better transferability to different driving scenes.

Index Terms—image-to-point cloud registration, implicit correspondence learning, depth-guided projective alignment, autonomous driving

## I. INTRODUCTION

Image-to-Point Cloud (I2P) Registration aims to estimate the camera pose of an image within a 3D scene point cloud, which is a fundamental task for autonomous driving applications such as visual localization [1], camera relocalization [2], SLAM [3], motion planning [4], and 3D reconstruction [5]. Despite significant progress in recent years, accurate Imageto-Point Cloud Registration remains challenging due to the inherent modality gap between dense 2D image observations and sparse 3D point cloud representations.

Existing I2P methods mainly rely on explicit correspondence estimation [6]–[9], direct pose regression [10], [11], or implicit correspondence learning [12]–[14]. Among them, implicit correspondence learning methods achieve stronger robustness by avoiding brittle hand-crafted matching and enabling end-to-end optimization. However, current methods still struggle to establish geometrically consistent cross-modal representations across heterogeneous modalities, especially in large-scale outdoor scenes with sparse LiDAR observations.

![](images/b28ea2c040515a62436225df195f67cbcba84868ecab0bdfe68170fb3cff9bad.jpg)  
Fig. 1. Motivation of DPA-I2P. Existing implicit correspondence learning methods suffer from weak cross-modal feature consistency. DPA-I2P integrates depth and projection cues to construct a discriminative cross-modal feature space for more reliable correspondence exploration.

The fundamental limitation lies in cross-modal feature inconsistency caused by the modality gap. Image features are often dominated by visual appearance without sufficient metricscale geometric awareness, resulting in ambiguous matching in repetitive or low-texture regions. In contrast, point cloud features encode rich 3D geometric structures but lack reliable visual grounding, making cross-modal association unstable under coarse pose estimation. Consequently, the inconsistent feature space easily leads to correspondence ambiguity, which further introduces optimization instability during iterative pose refinement and degrades camera pose estimation accuracy. Therefore, the key challenge is to construct geometry-aware image representations and visually grounded point representations for reliable implicit correspondence learning (Fig. 1).

Accurate correspondence reasoning requires metric-scale depth awareness and direction cues to capture camera geometry. Recent monocular metric depth estimation methods, particularly UniDepthV2 [15], provide effective geometric priors for narrowing the modality gap between images and point clouds. However, metric depth alone is insufficient to explicitly represent visual geometry. Therefore, we consider metric depth, camera ray directions, and projection consistency to construct geometry-aware image representations and projection-consistent point features to improve cross-modal feature alignment and matching stability. In this way, we propose DPA-I2P, a depth-aware implicit correspondence learning framework for Image-to-Point Cloud Registration. Specifically, Ray-Conditioned Metric Depth Encoding (RMDE) incorporates depth cues into image representations to enhance imageside geometric awareness. Projection-Consistent Vision Lifting (PVL) enriches point representations with visual information. RMDE and PVL jointly construct a more discriminative crossmodal feature space for implicit correspondence learning. In addition, Cross-Modal Query Pruning (CQP) suppresses unreliable candidates during early refinement, improving correspondence refinement for robust pose estimation.

The main contributions of this paper are summarized as follows:

• We propose DPA-I2P, a depth-aware implicit correspondence learning framework for robust end-to-end Imageto-Point Cloud Registration. By integrating metric depth and projection-aware visual cues, DPA-I2P constructs a more discriminative cross-modal feature space for correspondence learning.

• We introduce RMDE and PVL to enhance cross-modal geometric consistency. RMDE injects metric depth cues into image features to improve their geometric awareness, while PVL transfers projection-consistent visual information to point features.

• We propose CQP to suppress unreliable candidates during early correspondence refinement, improving correspondence stability and robustness in pose estimation.

• Extensive experiments on KITTI demonstrate that DPA-I2P achieves state-of-the-art registration accuracy, reducing RTE and RRE by 45.0% and 55.6%, respectively, over the strongest implicit baseline. Qualitative results on nuScenes further demonstrate its robustness and crossscene generalization capability.

## II. RELATED WORK

## A. Image-to-Point Cloud Registration

Existing Image-to-Point Cloud (I2P) Registration methods can generally be divided into three categories: explicit correspondence-based methods, correspondence-free methods, and implicit correspondence learning methods.

Explicit correspondence-based methods [6]–[9] first establish 2D–3D correspondences and then estimate camera pose via PnP-RANSAC [16], [17]. To improve robustness, recent works explore stronger feature representations and matching strategies. For example, FreeReg [18] introduces diffusion-based and depth-enhanced features, while CFI2P [19] adopts quantity-aware coarse-to-fine matching.

Other methods such as DeepI2P [10], EFGHNet [20], CoFiI2P [21], and GraphI2P [22] further improve robustness via overlap reasoning or structured matching. However, they remain sensitive to cross-modal feature inconsistency and correspondence quality.

Correspondence-free methods [10], [11] directly regress camera pose from global cross-modal features without explicit matching. While simplifying the pipeline, they often suffer from unstable performance under large viewpoint changes or weak geometric constraints.

More recently, implicit correspondence learning methods [12]–[14] unify bridge the gap between explicit matching and direct pose regression by learning latent query-based correspondences for end-to-end pose estimation. These methods typically rely on a differentiable probabilistic PnP solver for final pose estimation. However, they still struggle to learn discriminative representations during early correspondence exploration. In contrast, our method constructs a more discriminative feature space by integrating metric depth and projection cues into image and point representations.

## B. Monocular Metric Depth Estimation

Monocular depth estimation provides important geometric priors for 3D vision tasks. Early methods such as MiDaS [23] and DPT [24] predict relative depth from monocular images, but lack metric-scale consistency, limiting their use in geometry-sensitive tasks such as Image-to-Point Cloud Registration.

Recent methods directly predict metric-scale depth for more reliable geometric reasoning. We adopt UniDepthV2 [15] due to its strong cross-domain generalization ability and accurate metric depth prediction. It also produces pixel-wise confidence, which serves as an indicator of prediction reliability. In our framework, metric depth and confidence cues are incorporated into both image and point representations. Confidence further guides RMDE to emphasize reliable geometry and improves view-consistent vision lifting in PVL, leading to more stable correspondence learning.

## III. METHOD

## A. Overview

Given a 3D scene point cloud $P \in \mathbb { R } ^ { N \times 3 }$ and an image $I \in \mathbb { R } ^ { H \times W \times 3 }$ from the same scene, Image-to-Point Cloud Registration aims to estimate the camera pose $T _ { g t } = [ R _ { g t } | t _ { g t } ]$

An overview of DPA-I2P is illustrated in Fig. 2. Given an image and a point cloud, the framework first extracts multiscale image and point features and obtains an initial coarse pose estimate [12].To exploit the monocular depth prior in a structured geometry-aware manner rather than through simple depth concatenation, RMDE constructs uncertainty-aware rayconditioned depth features to enhance image representations, while PVL injects projection-consistent visual cues into point features under the coarse pose prior. CQP is further introduced during early refinement stages to suppress unreliable correspondence matching. Finally, the refined implicit correspondences are used for camera pose estimation.

![](images/5f5a58886e7dcc91340aab7bdfd7a6f7b0e1559530dcbd31d1f80d14967a771f.jpg)  
Fig. 2. Overview of DPA-I2P. Given an image and point cloud, the framework first integrates metric depth cues with a coarse projection-consistent prior via RMDE and PVL for feature enhancement, together with an additional CQP module for query refinement. It then performs implicit correspondence learning for final pose regression

## B. Ray-Conditioned Metric Depth Encoding

Image features contain strong visual cues but limited metricscale geometric awareness, which weakens early correspondence reasoning in repetitive or low-texture regions, especially in large-scale outdoor driving environments with sparse Li-DAR observations and significant viewpoint variations. To address this issue, we introduce Ray-Conditioned Metric Depth Encoding (RMDE), which converts monocular metric depth into a ray-aware geometric representation before cross-modal correspondence learning.

Given an input image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , UniDepthV2 provides a metric depth map $\mathbf { \bar { \mathbf { \Gamma } } } _ { D } ~ \in ~ \mathbb { R } ^ { H \times W }$ and a confidence map $C \in \mathbb { R } ^ { H \times W }$ , both of which are frozen during training. Let $\mathbf { F } _ { I } ^ { l } ~ \in ~ \mathbb { R } ^ { C _ { l } \times H _ { l } \times W _ { l } }$ denote the stage-l image feature. The depth and confidence maps are resized to the same resolution, producing $D ^ { l }$ and $C ^ { l }$ . We normalize the confidence map as $\bar { C } ^ { l } \ = \ \bar { \mathrm { N o r m } } ( C ^ { l } )$ , where a larger value indicates a more reliable depth prediction.

The camera intrinsics are adapted to the feature resolution: $\begin{array} { r } { f _ { x } ^ { l } = f _ { x } \frac { W _ { l } } { W } , f _ { y } ^ { l } = f _ { y } \frac { H _ { l } } { H } , c _ { x } ^ { l } = c _ { x } \frac { W _ { l } } { W } } \end{array}$ , and $\begin{array} { r } { c _ { y } ^ { l } = c _ { y } \frac { H _ { l } } { H } } \end{array}$ . For each feature location $i = ( u _ { i } ^ { l } , v _ { i } ^ { l } )$ , we compute the camera-ray coordinate as

$$
r _ { x , i } ^ { l } = \frac { u _ { i } ^ { l } - c _ { x } ^ { l } } { f _ { x } ^ { l } } , \quad r _ { y , i } ^ { l } = \frac { v _ { i } ^ { l } - c _ { y } ^ { l } } { f _ { y } ^ { l } } ,\tag{1}
$$

and define the corresponding ray direction as

$$
\tilde { \mathbf { d } } _ { i } ^ { l } = \left[ r _ { x , i } ^ { l } , r _ { y , i } ^ { l } , 1 \right] ^ { \top } , \quad \mathbf { d } _ { i } ^ { l } = \frac { \tilde { \mathbf { d } } _ { i } ^ { l } } { \lVert \tilde { \mathbf { d } } _ { i } ^ { l } \rVert _ { 2 } } .\tag{2}
$$

Instead of directly concatenating the depth map with image features, RMDE samples N candidate points along each viewing ray using the predicted metric depth as the reference surface. Specifically, we define an uncertainty-aware sampling radius

$$
s _ { i } ^ { l } = s _ { \mathrm { m i n } } ^ { l } + ( s _ { \mathrm { m a x } } ^ { l } - s _ { \mathrm { m i n } } ^ { l } ) ( 1 - \bar { C } _ { i } ^ { l } ) ,\tag{3}
$$

where $s _ { \mathrm { m i n } } ^ { l }$ and $s _ { \mathrm { m a x } } ^ { l }$ are stage-specific positive constants. A higher confidence produces a narrower sampling range, while a lower confidence allows a broader ray interval. Given a set of fixed offsets $\{ \rho _ { k } \} _ { k = 1 } ^ { N }$ with $\rho _ { k } \in [ - 1 , 1 ]$ , the sampled depth values are

$$
z _ { i , k } ^ { l } = \operatorname* { m a x } \left( D _ { i } ^ { l } + \rho _ { k } s _ { i } ^ { l } , \ \epsilon \right) , \quad k = 1 , \ldots , N .\tag{4}
$$

The corresponding 3D point in the camera coordinate system is then obtained by

$$
\mathbf { p } _ { i , k } ^ { l } = z _ { i , k } ^ { l } \tilde { \mathbf { d } } _ { i } ^ { l } .\tag{5}
$$

For each sampled point, we encode its 3D location, ray direction, and depth offset with respect to the predicted surface. The normalized depth offset is defined as

$$
\begin{array} { r } { \Delta z _ { i , k } ^ { l } = \log ( z _ { i , k } ^ { l } + \epsilon ) - \log ( D _ { i } ^ { l } + \epsilon ) . } \end{array}\tag{6}
$$

The ray-conditioned geometric descriptor of the k-th sampled point is

$$
\begin{array} { r } { \mathbf { h } _ { i , k } ^ { l } = \Big [ \gamma _ { p } ( \mathbf { p } _ { i , k } ^ { l } ) ; \gamma _ { d } ( \mathbf { d } _ { i } ^ { l } ) ; \gamma _ { z } ( \Delta z _ { i , k } ^ { l } ) ; { \bar { C } _ { i } ^ { l } } \Big ] , } \end{array}\tag{7}
$$

where $\gamma _ { p } ( \cdot ) , \gamma _ { d } ( \cdot )$ , and $\gamma _ { z } ( \cdot )$ denote lightweight positional encodings for 3D location, ray direction, and depth offset, respectively. A shared encoder maps the descriptor to a sampled ray feature:

$$
\mathbf { e } _ { i , k } ^ { l } = \phi ^ { l } ( \mathbf { h } _ { i , k } ^ { l } ) , \quad \mathbf { e } _ { i , k } ^ { l } \in \mathbb { R } ^ { C _ { l } } .\tag{8}
$$

To aggregate sampled features along the ray, we assign larger weights to samples closer to the predicted metric depth and make the weighting less peaked when the depth confidence is low:

$$
\tilde { w } _ { i , k } ^ { l } = \bar { C } _ { i } ^ { l } \exp \left( - \frac { | \Delta z _ { i , k } ^ { l } | } { \tau _ { l } + \epsilon } \right) + ( 1 - \bar { C } _ { i } ^ { l } ) ,\tag{9}
$$

$$
w _ { i , k } ^ { l } = \frac { \tilde { w } _ { i , k } ^ { l } } { \sum _ { j = 1 } ^ { N } \tilde { w } _ { i , j } ^ { l } } .\tag{10}
$$

The ray-conditioned metric depth encoding for location i is obtained by a confidence-weighted aggregation:

$$
{ \mathbf e } _ { i } ^ { l } = \sum _ { k = 1 } ^ { N } w _ { i , k } ^ { l } { \mathbf e } _ { i , k } ^ { l } .\tag{11}
$$

Finally, RMDE injects the aggregated ray-aware geometric representation into the image feature through a gated residual update. Let $\mathbf { f } _ { I , i } ^ { l }$ denote the image feature at location i. The residual gate is computed as

$$
\mathbf { g } _ { i } ^ { l } = \sigma \left( \eta ^ { l } \left( [ \mathbf { f } _ { I , i } ^ { l } ; \mathbf { e } _ { i } ^ { l } ; \bar { C } _ { i } ^ { l } ] \right) \right) ,\tag{12}
$$

where $\eta ^ { l } ( \cdot )$ is a lightweight projection layer. The enhanced image feature is

$$
\hat { \mathbf { f } } _ { I , i } ^ { l } = \mathbf { f } _ { I , i } ^ { l } + \bar { C } _ { i } ^ { l } \left( \mathbf { g } _ { i } ^ { l } \odot \mathbf { e } _ { i } ^ { l } \right) .\tag{13}
$$

The enhanced image feature map is then reconstructed as

$$
\begin{array} { r } { \hat { \mathbf { F } } _ { I } ^ { l } = \mathrm { R e s h a p e } \left( \{ \hat { \mathbf { f } } _ { I , i } ^ { l } \} _ { i = 1 } ^ { H _ { l } W _ { l } } \right) . } \end{array}\tag{14}
$$

In this way, RMDE does not merely append monocular depth to image features. Instead, it represents each image token as an uncertainty-aware local ray distribution around the predicted metric surface, allowing image features to carry camera-aware geometric cues before cross-modal correspondence learning.

## C. Projection-Consistent Vision Lifting

Although RMDE enhances image-side geometry, point features remain view-independent and lack vision consistency with the image branch. We therefore project 3D points onto the image feature plane using the coarse pose prior and lift projection-consistent vision cues into point features.

Let $\mathbf { X } _ { I } ^ { l } = \mathrm { F l a t t e n } ( \hat { \mathbf { F } } _ { I } ^ { l } ) \in \mathbb { R } ^ { H _ { l } W _ { l } \times \bar { C } _ { I } ^ { l } }$ denote the flattened image tokens, and let $\mathbf { F } _ { P } ^ { l } \in \mathbb { R } ^ { N _ { l } \times C _ { P } ^ { l } }$ denote the stage-l point features. Given the coarse pose $\mathbf { T } _ { f } = ( \mathbf { R } _ { f } , \mathbf { t } _ { f } )$ and camera intrinsics K, each point $\mathbf { p } _ { i } ^ { l }$ is projected onto the stage-l image feature plane as

$$
\bar { \mathbf { u } } _ { i } ^ { l } = \lfloor \Gamma ^ { l } ( \pi ( \mathbf { p } _ { i } ^ { l } ; \mathbf { K } , \mathbf { R } _ { f } , \mathbf { t } _ { f } ) ) \rceil ,\tag{15}
$$

with validity indicator $m _ { i } ^ { l } \in \{ 0 , 1 \}$ . The corresponding image feature is gathered by $\mathbf { z } _ { i } ^ { i } = \mathcal { G } ( \mathbf { X } _ { I } ^ { l } , \bar { \mathbf { u } } _ { i } ^ { l } )$ , where G(·) denotes center-cell feature gathering, and projected into the point feature space as $\bar { \mathbf { z } } _ { i } ^ { l } = \psi ^ { l } ( \mathbf { z } _ { i } ^ { l } )$

To reduce sensitivity to coarse pose errors, the vision cue is incorporated through a bounded residual update:

$$
\hat { \mathbf { f } } _ { P , i } ^ { l } = \mathbf { f } _ { P , i } ^ { l } + \delta ^ { l } m _ { i } ^ { l } \big ( \operatorname { t a n h } ( \mathbf { w } ^ { l } ) \odot \bar { \mathbf { z } } _ { i } ^ { l } \big ) ,\tag{16}
$$

where $\mathbf { w } ^ { l } \in \mathbb { R } ^ { C _ { P } ^ { l } }$ is a learnable channel-wise modulation vector and $\delta ^ { l } \in \{ 0 , 1 \}$ is a stage indicator. PVL is enabled only in early stages and disabled during late refinement stages to avoid noisy vision priors. Invalid projections $( m _ { i } ^ { l } = 0 )$ preserve the original point features. The enhanced point features are

$$
\hat { \mathbf { F } } _ { P } ^ { l } = \{ \hat { \mathbf { f } } _ { P , i } ^ { l } \} _ { i = 1 } ^ { N _ { l } } .\tag{17}
$$

## D. Cross-Modal Query Pruning

Although RMDE and PVL improve cross-modal representations, early correspondence queries may still drift toward geometrically unsupported regions. To stabilize correspondence exploration, we introduce a projective support prior.

$\operatorname { L e t } T ^ { l - 1 } = ( R ^ { l - 1 } , t ^ { l - 1 } )$ denote the pose estimate before the l-th refinement stage. Given the stage-wise point set $\{ \mathbf { p } _ { i } ^ { l } \} _ { i = 1 } ^ { N _ { l } } .$ each point is projected onto the stage-l feature plane as

$$
\hat { \mathbf { u } } _ { i } ^ { l } = \Gamma ^ { l } ( \pi ( \mathbf { p } _ { i } ^ { l } ; K , R ^ { l - 1 } , t ^ { l - 1 } ) ) ,\tag{18}
$$

with validity indicator $m _ { i } ^ { l } \in \{ 0 , 1 \}$ . Based on the projected points, we construct a dense support prior:

$$
\mathcal { S } ^ { l } ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { N _ { l } } m _ { i } ^ { l } \exp \left( - \frac { \lVert \boldsymbol { x } - \hat { \mathbf { u } } _ { i } ^ { l } \rVert _ { 2 } ^ { 2 } } { 2 \sigma _ { l } ^ { 2 } } \right) .\tag{19}
$$

After normalization, the support prior is flattened into $\mathbf { s } ^ { l } = \mathrm { F l a t t e n } ( S ^ { l } )$ . Using the flattened image tokens ${ \bf X } _ { I } ^ { l } \ \in$ R ${ _ { \parallel } } _ { l } W _ { l } \times C _ { l }$ and image-side queries $Q _ { \mathrm { i m g } } ^ { l } ~ \mathbf { \bar { \in } } ~ \mathbb { R } ^ { N _ { q } \times C _ { l } }$ , we compute the support-aware query-token similarity:

$$
B _ { I } ^ { l } = \frac { Q _ { \mathrm { i m g } } ^ { l } ( \mathbf { X } _ { I } ^ { l } ) ^ { \top } } { \sqrt { C _ { l } } } + \beta _ { l } \log ( \mathbf { s } ^ { l } + \epsilon ) ,\tag{20}
$$

where $\beta _ { l }$ is a stage-dependent weight and ϵ avoids numerical instability.

The image-side heatmap is then computed as $\begin{array} { r l } { H _ { I } ^ { l } } & { { } = } \end{array}$ Softmax $( B _ { I } ^ { \bar { l } } )$ , and the corresponding 2D keypoints are obtained by $\dot { K } _ { I } ^ { l } = H _ { I } ^ { l } E _ { I } ^ { l }$ , where $E _ { I } ^ { l } \ \in \ \mathbb { R } ^ { H _ { l } W _ { l } \times 2 }$ denotes the stage-wise image coordinate matrix.

For regions with extremely low support, we further construct a conservative pruning mask $M _ { \mathrm { s u p } } ^ { l } = \mathbb { I } ( \mathbf { s } ^ { l } < \tau _ { l } )$ , where $\tau _ { l }$ is a small threshold. Query pruning is applied only during early refinement stages and gradually relaxed later.

## E. Loss Function

The metric depth network remains frozen during training. The overall objective jointly optimizes pose regression and support-guided query regularization.

Let the ground-truth pose be $T _ { \mathrm { g t } } = ( R _ { \mathrm { g t } } , t _ { \mathrm { g t } } )$ , the coarse pose be $T _ { f } ~ = ~ ( R _ { f } , t _ { f } )$ , and the refined pose at stage l be $\bar { T } ^ { l } = ( \hat { R } ^ { l } , \bar { t } ^ { l } )$ . The pose regression loss is defined as

$$
\mathcal { L } _ { \mathrm { p o s e } } ( T , T _ { \mathrm { g t } } ) = \Vert \hat { t } ^ { l } - t _ { \mathrm { g t } } \Vert _ { 1 } + \lambda _ { R } d _ { R } ( \hat { R } ^ { l } , R _ { \mathrm { g t } } ) ,\tag{21}
$$

where $d _ { R } ( \cdot , \cdot )$ denotes the geodesic rotation distance and $\lambda _ { R }$ balances translation and rotation terms.

To align early correspondence exploration with projective support, we introduce a support-guided regularization loss:

$$
\mathcal { L } _ { \mathrm { s u p } } = \sum _ { l \in \mathcal { E } } \frac { 1 } { N _ { q } } \sum _ { q = 1 } ^ { N _ { q } } \mathrm { K L } \left( H _ { I , q } ^ { l } \parallel \tilde { \mathbf { s } } _ { q } ^ { l } \right) .\tag{22}
$$

where E denotes the set of early refinement stages, and $\tilde { \mathbf { s } } _ { q } ^ { l }$ denotes the normalized projective support distribution for query q.

The overall objective is

$$
\mathcal { L } = \lambda _ { c } \mathcal { L } _ { \mathrm { p o s e } } ( T _ { c } , T _ { \mathrm { g t } } ) + \lambda _ { p } \sum _ { l = 1 } ^ { L } \omega _ { l } \mathcal { L } _ { \mathrm { p o s e } } ( T , T _ { \mathrm { g t } } ) + \lambda _ { s } \mathcal { L } _ { \mathrm { s u p } } ,\tag{23}
$$

where ω denotes the stage weight and $\lambda _ { c } , \lambda _ { p } , \lambda _ { s }$ balance different loss terms. Following [12], the coarse pose branch is first optimized independently, after which the entire framework is trained end-to-end.

## IV. EXPERIMENTS

In this section, we first introduce the implementation details. Then, we present quantitative results on the KITTI dataset and qualitative visualizations on both KITTI and nuScenes datasets. Finally, we conduct a series of ablation studies to verify the effectiveness of each component.

## A. Implementation Details

Our framework employs a 4-stage ResNet [25] with FPN [26] as the image backbone and a 4-stage KPFCNN [27] as the point backbone, both outputting 128-dimensional features. We utilize 4-head attention [28] across all layers, with $N _ { q } = 1 2 8$ correspondence queries and a refinement depth of $L = 3$ . metric depth and confidence maps are pre-computed offline using UniDepthV2 and kept frozen to provide geometric priors for correspondence learning, without fine-tuning the depth estimator. Specifically, RMDE is applied at all four scales, while PVL and CQP are utilized only in the first two stages. The model is trained for 40 epochs using the Adam optimizer (batch size 4, initial learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 6 } )$ on a single NVIDIA RTX 4090 GPU.

## B. Datasets and Evaluation Metrics

1) KITTI [29].: We evaluate on the KITTI dataset, comprising 22 synchronized image–LiDAR sequences, and 11 sequences provide ground-truth calibration files. Following the standard protocol [9], sequences 0–8 are used for training and sequences 9–10 for evaluation. Misregistration transformations are generated using a 2D translation on the ground plane within ±10 m and a rotation around the up-axis with no limited range. Input images are resized to $1 6 0 \times 5 1 2$ , and point clouds are downsampled to 40,960 points.

2) nuScenes [30].: We further conduct qualitative evaluation on the nuScenes benchmark using the official 150 test scenes. Image and point cloud pairs are obtained via the nuScenes SDK, where point clouds are accumulated from adjacent frames to ensure sufficient density. Input images are resized to $1 6 0 \times 3 2 0$ , and point clouds are uniformly downsampled to 40,960 points.

3) Evaluation Metrics.: We adopt average Relative Translational Error (RTE), average Relative Rotation Error (RRE), and registration accuracy (Acc). Acc is defined as the proportion of registrations satisfying RTE < 2 m and $\mathrm { R R E } < 5 ^ { \circ }$

## C. Registration Accuracy

1) Quantitative Comparison.: We report the quantitative results in Table I. DPA-I2P achieves the best performance across all evaluation metrics on the KITTI dataset. Compared with ICLI2P [12], our method reduces RTE from 0.20 m to 0.11 m and RRE from 1.24<sup>◦</sup> to 0.55<sup>◦</sup>, while achieving the highest registration accuracy of 99.70%. We attribute the superior performance to three factors. RMDE reduces ambiguous image matching by introducing metric depth and ray-direction cues. PVL improves cross-modal feature compatibility by lifting projection-consistent vision cues to point features. CQP further stabilizes early correspondence refinement by suppressing projectively unsupported matching.

2) Qualitative Comparison.: Qualitative comparisons on KITTI and nuScenes are shown in Fig. 3 and Fig. 4, respectively. For visualization, we project the point clouds into the image space using the predicted poses from different methods and the camera intrinsics, where colors encode depth. Compared with VP2P-Match [9] and ICLI2P [12], our method achieves more accurate registration results in challenging driving scenes. We further provide 2D–3D correspondence visualizations in Fig. 5. The projected points of our method exhibit more accurate alignment with image structures, indicating that the proposed implicit correspondence learning framework can establish more reliable 2D–3D correspondences.

## D. Ablation Study

In this section, we conduct ablation studies on the KITTI dataset to validate the effectiveness of each component and analyze several key design choices in our framework.

We evaluate three variants in Table II: (1) w/o RMDE removes ray-conditioned metric depth encoding; (2) w/o PVL disables Projection-Consistent Vision Lifting in the point branch; (3) w/o CQP removes cross-modal query pruning and retains all 2D–3D matching during early refinement.

0.14 m/0.97<sup>◦</sup>  
TABLE I  
REGISTRATION ACCURACY ON THE KITTI AND NUSCENES DATASETS. LOWER IS BETTER FOR RTE AND RRE, HIGHER IS BETTER FOR ACC.
<table><tr><td rowspan="2">Methods</td><td colspan="3">KITTI</td><td colspan="3">nuScenes</td></tr><tr><td>RTE↓</td><td>RRE↓</td><td>Acc.↑</td><td>RTE↓</td><td>RRE↓</td><td>Acc.↑</td></tr><tr><td>Grid Cls.+PnP [10]</td><td>3.64±3.46</td><td>19.19±28.96</td><td>11.22</td><td>3.02±2.40</td><td>12.66±21.01</td><td>2.45</td></tr><tr><td>DeepI2P(3D) [10]</td><td>4.06±3.54</td><td>24.73±31.69</td><td>3.77</td><td>2.88±2.12</td><td>20.65±12.24</td><td>2.26</td></tr><tr><td>DeepI2P(2D) [10]</td><td>3.59±3.21</td><td>11.66±18.16</td><td>25.95</td><td>2.78±1.99</td><td>4.80±6.21</td><td>38.10</td></tr><tr><td>CorrI2P [8]</td><td>3.78±65.16</td><td>5.89±20.34</td><td>72.42</td><td>3.04±60.76</td><td>3.73±9.03</td><td>49.00</td></tr><tr><td>VP2P-Match [9]</td><td>0.75±1.13</td><td>3.29±7.99</td><td>83.04</td><td>0.89±1.44</td><td>2.15±7.03</td><td>88.33</td></tr><tr><td>ICLI2P [12]</td><td>0.20±0.21</td><td>1.24±2.34</td><td>97.49</td><td>0.63±0.44</td><td>2.13±3.75</td><td>90.94</td></tr><tr><td>DPA-I2P</td><td>0.11±0.12</td><td>0.55±0.67</td><td>99.70</td><td>0.54±0.37</td><td>1.92±3.81</td><td>92.02</td></tr></table>

![](images/c7a5f2a5e92211875689378df7ecab2f585827857769cd8881e5e0a871816cb5.jpg)

![](images/8fa4aff01e9f985a92d845cde1b53fb565a2d65363dd77ff455a9bc562ae3693.jpg)

![](images/4575ec33a48a1735286473b590a419539152e0a5530f8783e2f58ec02dcb5191.jpg)

![](images/38fcd3d318ec4a74bc40534d5b4c4282de03a9daafee421606586301b2662e02.jpg)  
Fig. 3. Qualitative comparison of image-to-point cloud registration results on the KITTI dataset.

![](images/d54eeda1b62aed012e84586cae381403c1cbec09ead870233f029c4a628eda1e.jpg)

![](images/5b0958deec5279ef6d15c9772259f1e05c2b75fd4af404f3f25654eec71fb5aa.jpg)  
0.57 m/2.29<sup>◦</sup>

![](images/608ea7f3b3035ab75c65ba2fa973e2623cd08107fb043c29a6d0c5219eb75bb6.jpg)  
0.28 m/1.69<sup>◦</sup>

![](images/7665eef08645e2dbff6141d1c33cb4a86fd3316cf7a4b654becedce5605a9d29.jpg)  
Fig. 4. Qualitative comparison of image-to-point cloud registration results on the nuScenes dataset.

1) Effects of the RMDE.: As shown in Table II, with RMDE, the performance on the KITTI dataset is improved on all metrics. Specifically, RTE decreases from 0.12 m to 0.11 m, RRE decreases from 0.58<sup>◦</sup> to 0.55<sup>◦</sup>, and Acc increases from 99.61% to 99.70%. This demonstrates that incorporating metric depth and ray-direction cues consistently improves image representations before cross-modal interaction.

2) Effects of the PVL.: Without PVL, RRE increases from 0.55<sup>◦</sup> to 0.74<sup>◦</sup>, which is the largest degradation among all variants, while other metrics remain relatively stable. This suggests that Projection-Consistent Vision Lifting mainly benefits rotation estimation by improving cross-modal geometric alignment.

3) Effects of the CQP.: Without CQP, RTE increases from 0.11 m to 0.18 m and Acc drops from 99.70% to 98.82%, showing the most significant performance degradation. This indicates that suppressing unreliable matching during early refinement is crucial for stable correspondence learning.

TABLE II  
THE EFFECT OF EACH DESIGN IN OUR FRAMEWORK.
<table><tr><td>Methods</td><td>RTE (m)↓</td><td>RRE (°)↓</td><td>Acc. (%)↑</td></tr><tr><td>w/o RMDE</td><td>0.12±0.11</td><td>0.58±0.72</td><td>99.61</td></tr><tr><td>w/o PVL</td><td>0.12±0.14</td><td>0.74±0.87</td><td>99.46</td></tr><tr><td>w/o CQP</td><td>0.18±0.22</td><td>0.65±0.51</td><td>98.82</td></tr><tr><td>DPA-I2P (Full)</td><td>0.11±0.12</td><td>0.55±0.67</td><td>99.70</td></tr></table>

4) Component Analysis of RMDE.: Table III analyzes different ways of exploiting the same frozen UniDepthV2 prior in the image branch, while keeping the remaining framework unchanged. Directly concatenating the raw metric depth map provides only a marginal improvement over the no-depth baseline, reducing RRE from 0.58<sup>◦</sup> to 0.57<sup>◦</sup> and increasing Acc. from 99.61% to 99.63%. This indicates that the metric depth prior itself is useful, but naive depth concatenation is insufficient to fully exploit its geometric information.

![](images/4091b837c5e1a2aec9cad0dd9c6e126365831fbbfbe6461ce0d30e4a3f92b2f0.jpg)  
0.09m / 0.45<sup>◦</sup>

![](images/1562b0810954b103e1bf27c1094bf1f82c161d8b97439a57a41ffc558007d1b1.jpg)

![](images/35558f8c6bed7ba31479640062bd613fd0bc717b08cb3246eba76d3c01b468d4.jpg)  
0.04m / 0.61<sup>◦</sup>

0.08m / 0.37<sup>◦</sup>  
![](images/c3da47b8e752e2da890bf60f2cff1c0901fb65d1cecc882b6fd0eeb6ff54c280.jpg)  
0.08m / 0.26<sup>◦</sup>  
Fig. 5. Visual illustration of 2D-3D correspondences and registration accuracy. Green lines represent correct correspondences.

Introducing camera-ray direction into the depth representation further improves the performance, yielding an RRE of 0.57<sup>◦</sup> and an Acc. of 99.64%. This suggests that metric depth becomes more informative when explicitly coupled with camera geometry rather than treated as an independent scalar cue. Encoding the depth as a camera-aware surface point further reduces RRE to 0.56<sup>◦</sup> and improves Acc. to 99.66%, indicating that explicitly representing the 3D location associated with each image token provides more effective geometric supervision.

We then extend the single surface estimate to a local neighborhood by uniformly sampling multiple points along each viewing ray. The resulting variant further improves the performance to an RRE of 0.56<sup>◦</sup> and an Acc. of 99.67%, supporting the use of a local ray distribution instead of relying on a single predicted surface point. Removing confidenceaware aggregation slightly degrades the performance, with Acc. decreasing to 99.66% and RRE increasing to 0.56<sup>◦</sup>. This indicates that confidence provides a complementary benefit by reducing the influence of less reliable monocular depth predictions.

Finally, the full RMDE achieves the best performance, with an RTE/RRE/Acc. of $0 . 1 1 \pm 0 . 1 2 ~ \mathrm { m } / 0 . 5 5 \pm 0 . 6 7 ^ { \circ } / 9 9 . 7 0 \%$ Overall, the results show that the gain of RMDE does not come merely from appending an external depth prior, but from progressively structuring the prior through camera-ray geometry, local ray sampling, and confidence-aware aggregation before cross-modal correspondence refinement.

5) Effects of the Pruning Schedule.: Table IV shows that applying query pruning only in the first two refinement stages achieves the best performance, reducing RTE from 0.18 m to 0.11 m and RRE from $0 . 6 5 ^ { \circ }$ to 0.55<sup>◦</sup> compared with no pruning. In contrast, applying pruning throughout all stages leads to inferior results, suggesting that later refinement requires more flexible local correspondence exploration rather than overly strong projective constraints.

TABLE III  
ABLATION STUDY OF RMDE COMPONENTS ON KITTI. ALL VARIANTS USE THE SAME FROZEN UNIDEPTHV2 PRIOR WHEN DEPTH IS INVOLVED, AND THE REMAINING FRAMEWORK IS KEPT UNCHANGED.
<table><tr><td>RMDE variant</td><td>RTE (m)↓</td><td>RRE (°)↓</td><td>Acc. (%)↑</td></tr><tr><td>No depth prior</td><td>0.12±0.11</td><td>0.58±0.72</td><td>99.61</td></tr><tr><td>Raw depth concatenation</td><td>0.12±0.12</td><td>0.57±0.70</td><td>99.63</td></tr><tr><td>Depth + ray direction</td><td>0.12±0.12</td><td>0.57±0.69</td><td>99.64</td></tr><tr><td>Surface-point encoding</td><td>0.12±0.12</td><td> $0 . 5 6 { \pm } 0 . 6 9$ </td><td>99.66</td></tr><tr><td>Uniform ray sampling w/o confidence</td><td> $0 . 1 1 { \pm } 0 . 1 2$ </td><td> $0 . 5 6 { \pm } 0 . 6 8$ </td><td>99.67</td></tr><tr><td>Full RMDE</td><td> $\mathbf { 0 . 1 1 \bot 0 . 1 } 2$ </td><td> $\mathbf { 0 . 5 5 \pm 0 . 6 7 }$ </td><td>99.70</td></tr></table>

TABLE IV  
ABLATION STUDIES ON QUERY PRUNING STRATEGIES ON THE KITTI DATASET.
<table><tr><td>Query strategy</td><td>RTE (m)↓</td><td>RRE (°)↓</td><td>Acc. (%)↑</td></tr><tr><td>No query pruning</td><td>0.18±0.22</td><td>0.65±0.51</td><td>98.82</td></tr><tr><td>Early-only</td><td>0.11±0.12</td><td> $\mathbf { 0 . 5 5 \bot 0 . 6 7 }$ </td><td>99.70</td></tr><tr><td>All-stage</td><td> $0 . 1 4 \pm 0 . 4 3$ </td><td> $0 . 6 6 { \pm } 0 . 8 7$ </td><td>98.90</td></tr></table>

## E. Efficiency Analysis

We evaluate the efficiency of DPA-I2P on the KITTI dataset. All methods are evaluated under the same input setting, where the image is resized to 160 × 512 and the point cloud is downsampled to 40,960 points. Neural inference, including feature extraction, cross-modal correspondence refinement, and differentiable pose solving, is performed on an NVIDIA GeForce RTX 4090 GPU. Following our implementation setting, we use a batch size of 4 and report the average network size, GPU memory consumption, and end-to-end inference time per image–point cloud pair. Since the metric depth and confidence maps are pre-computed offline using the frozen UniDepthV2 model, this preprocessing cost is excluded from the reported online inference time.

The results are summarized in Table V. Classification-based methods such as Grid Cls. achieve lower computational and memory overhead, as they do not perform dense cross-modal correspondence refinement. Recent implicit correspondence learning methods require additional computation to establish and refine cross-modal correspondences, resulting in moderately higher resource consumption. DPA-I2P has a network size of 179.93 MB, uses 11.15 GB of GPU memory, and takes 36.81 ms for end-to-end inference. Compared with ICLI2P, these values increase from 175.92 MB to 179.93 MB, from 10.74 GB to 11.15 GB, and from 35.12 ms to 36.81 ms, corresponding to increases of approximately 2.3%, 3.8%, and 4.8%, respectively. Meanwhile, DPA-I2P achieves substantially improved registration accuracy, as reported in Table I. The relatively small increase in model size, GPU memory usage, and inference time indicates that the proposed RMDE, PVL, and CQP modules introduce only moderate additional

## TABLE V

EFFICIENCY COMPARISON ON THE KITTI DATASET. NETWORK SIZE, GPU MEMORY, AND INFERENCE TIME ARE MEASURED ON AN NVIDIA   
GEFORCE RTX 4090 GPU. THE REPORTED INFERENCE TIME INCLUDES FEATURE EXTRACTION, CORRESPONDENCE REFINEMENT, AND DIFFERENTIABLE POSE SOLVING. OFFLINE METRIC DEPTH PRE-COMPUTATION IS NOT INCLUDED.

<table><tr><td>Methods</td><td>Network size (MB)</td><td>Mem. (GB)</td><td>Inference (ms)</td></tr><tr><td>Grid Cls. [10] DeepI2P [10]</td><td>100.75 100.12</td><td>2.39 2.01</td><td>11.20 7.55</td></tr><tr><td>CorrI2P [8]</td><td>141.07</td><td>2.88</td><td>13.75</td></tr><tr><td>VP2P-Match [9]</td><td>146.42</td><td>17.17</td><td>30.63</td></tr><tr><td>ICLI2P [12]</td><td>175.92</td><td>10.74</td><td>35.12</td></tr><tr><td>DPA-I2P (Ours)</td><td>179.93</td><td>11.15</td><td>36.81</td></tr></table>

overhead, while the overall inference efficiency remains within the range of recent implicit correspondence learning methods.

## V. CONCLUSION

We propose DPA-I2P, a depth-aware implicit correspondence learning framework for Image-to-Point Cloud Registration. RMDE and PVL improve cross-modal feature representation using metric depth and projection-consistent vision cues, while CQP enhances matching stability by suppressing unreliable queries. Experiments on KITTI and nuScenes demonstrate superior performance, robustness, and cross-dataset generalization.

## ACKNOWLEDGMENT

## REFERENCES

[1] M. Pietrantoni, M. Humenberger, T. Sattler, and G. Csurka, “Segloc: Learning segmentation-based representations for privacy-preserving visual localization,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 15 380–15 391.

[2] A. Moreau, N. Piasco, M. Bennehar, D. Tsishkou, B. Stanciulescu, and A. de La Fortelle, “Crossfire: Camera relocalization on self-supervised features from an implicit representation,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 252– 262.

[3] L. Lipson and J. Deng, “Multi-session slam with differentiable widebaseline pose optimization,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 19 626– 19 635.

[4] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang et al., “Planning-oriented autonomous driving,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 17 853–17 862.

[5] Z. Li, T. Muller, A. Evans, R. H. Taylor, M. Unberath, M.-Y. Liu, and¨ C.-H. Lin, “Neuralangelo: High-fidelity neural surface reconstruction,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 8456–8465.

[6] M. Feng, S. Hu, M. H. Ang, and G. H. Lee, “2d3d-matchnet: Learning to match keypoints across 2d image and 3d point cloud,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 4790–4796.

[7] Q.-H. Pham, M. A. Uy, B.-S. Hua, D. T. Nguyen, G. Roig, and S.-K. Yeung, “Lcd: Learned cross-domain descriptors for 2d-3d matching,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 07, 2020, pp. 11 856–11 864.

[8] S. Ren, Y. Zeng, J. Hou, and X. Chen, “Corri2p: Deep image-to-point cloud registration via dense correspondence,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 3, pp. 1198– 1208, 2022.

[9] J. Zhou, B. Ma, W. Zhang, Y. Fang, Y.-S. Liu, and Z. Han, “Differentiable registration of images and lidar point clouds with voxelpoint-topixel matching,” Advances in Neural Information Processing Systems, vol. 36, pp. 51 166–51 177, 2023.

[10] J. Li and G. H. Lee, “Deepi2p: Image-to-point cloud registration via deep classification,” arXiv preprint arXiv:2104.03501, 2021.

[11] G. Yao, X. Li, Y. Xuan, and Y. Pan, “Mafreei2p: A matching-free image-to-point cloud registration paradigm with active camera pose retrieval,” in 2024 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2024, pp. 1–6.

[12] X. Li, W. Yang, J. Deng, Z. Cheng, X. Zhou, and T. Zhang, “Implicit correspondence learning for image-to-point cloud registration,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 16 922–16 931.

[13] J. Mu, C. Ren, W. Zhang, L. Pan, X.-P. Zhang, and Y. Gao, “Diff 2 i2p: Differentiable image-to-point cloud registration with diffusion prior,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 25 777–25 787.

[14] Z. Cheng, J. Deng, X. Li, X. Yin, B. Liao, B. Yin, W. Yang, and T. Zhang, “Ca-i2p: Channel-adaptive registration network with global optimal selection,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 27 739–27 749.

[15] L. Piccinelli, C. Sakaridis, Y.-H. Yang, M. Segu, S. Li, W. Abbeloos, and L. Van Gool, “Unidepthv2: Universal monocular metric depth estimation made simpler,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[16] V. Lepetit, F. Moreno-Noguer, and P. Fua, “Ep n p: An accurate o (n) solution to the p n p problem,” International journal of computer vision, vol. 81, no. 2, pp. 155–166, 2009.

[17] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981.

[18] H. Wang, Y. Liu, B. Wang, Y. Sun, Z. Dong, W. Wang, and B. Yang, “Freereg: Image-to-point cloud registration leveraging pretrained diffusion models and monocular depth estimators,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 28 484–28 507.

[19] G. Yao, Y. Xuan, Y. Chen, and Y. Pan, “Quantity-aware coarse-to-fine correspondence for image-to-point cloud registration,” IEEE Sensors Journal, vol. 24, no. 20, pp. 33 826–33 837, 2024.

[20] Y. Jeon and S.-W. Seo, “Efghnet: A versatile image-to-point cloud registration network for extreme outdoor environment,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7511–7517, 2022.

[21] S. Kang, Y. Liao, J. Li, F. Liang, Y. Li, X. Zou, F. Li, X. Chen, Z. Dong, and B. Yang, “Cofii2p: Coarse-to-fine correspondences-based image to point cloud registration,” IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 10 264–10 271, 2024.

[22] L. Bie, S. Pan, S. Li, Y. Zhao, and Y. Gao, “Graphi2p: Image-topoint cloud registration with exploring pattern of correspondence via graph learning,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 22 161–22 171.

[23] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 3, pp. 1623–1637, 2020.

[24] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2021, pp. 12 159–12 168.

[25] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[26] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in 2017 IEEE conference on computer vision and pattern recognition (CVPR). IEEE, 2017, pp. 936–944.

[27] H. Thomas, C. R. Qi, J.-E. Deschaud, B. Marcotegui, F. Goulette, and L. J. Guibas, “Kpconv: Flexible and deformable convolution for point clouds,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 6411–6420.

[28] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[29] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012, pp. 3354–3361.

[30] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuscenes: A multimodal dataset for autonomous driving,” in 2020 IEEE/CVF conference on

computer vision and pattern recognition (CVPR). IEEE, 2020, pp.   
11 618–11 628.