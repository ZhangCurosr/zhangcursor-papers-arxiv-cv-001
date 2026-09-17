# IRIS: Implicit Rendering Maters for Pose-Free Novel View Synthesis

Wenyu Li<sup>†</sup>, Sidun Liu<sup>†</sup>, Peng Qiao<sup>\*</sup>, Yong Dou<sup>\*</sup>, and Tongrui Hu

National University of Defense Technology

Changsha, China

## Abstract

Novel view synthesis from unposed multi-view images remains challenging, as the model must jointly learn scene representations and camera parameters without pose supervision. Existing approaches largely fall into two extremes: implicit latent-space rendering is flexible and easy to optimize, but often yields weakly grounded camera estimation; explicit 3D representations provide stronger geometric grounding, but introduce heavier parameterization and more fragile optimization. In this paper, we present IRIS, a fully self-supervised framework that provides a practical middle ground between these two paradigms. Instead of decoding free latent tokens or reconstructing fully explicit 3D primitives, IRIS represents the scene as a latent neural field and renders novel views by querying this field under self-predicted cameras. Specifically, projected features from reference views are aggregated at sampled 3D points to form point-wise latent features, which are then composed along target rays for rendering. This design preserves the flexibility and optimization stability of implicit modeling, while introducing stronger geometric structure than unconstrained latent rendering. Extensive experiments show that IRIS achieves strong novel view synthesis quality with competitive pose accuracy under fully self-supervised learning. Our project page: https://leo-frank.github.io/IRIS/.

## CCS Concepts

• Computing methodologies → Reconstruction.

## Keywords

Novel View Synthesis, 3D Reconstruction

ACM Reference Format:

Wenyu Li<sup>†</sup>, Sidun Liu<sup>†</sup>, Peng Qiao<sup>\*</sup>, Yong Dou<sup>\*</sup>, and Tongrui Hu. 2026. IRIS: Implicit Rendering Matters for Pose-Free Novel View Synthesis. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3767308.3835537

## 1 Introduction

Novel view synthesis is a fundamental problem in computer vision and computer graphics, aiming to recover scene representation from limited observations and support high-quality rendering at target viewpoints. In recent years, this area has witnessed remarkable progress, ranging from radiance field representations such as NeRF[21], to explicit scene representations such as 3D Gaussian Splatting[16], and further to feed-forward models[7, 12, 13, 25, 32, 39] for view synthesis without scene-specific optimization.

Most of these methods rely on known or pre-estimated camera poses[4, 7, 13, 32], which directly afect the quality of target-view rendering. Their dependence on external camera acquisition still poses clear limitations. In practice, camera parameters are often obtained through Structure-from-Motion pipelines[27, 28], which not only increase computational cost and engineering complexity, but also become unstable in the presence of weak textures, heavy occlusions, or large viewpoint changes. Moreover, such dependence on external pose estimation limits the scalability of these methods to large collections of raw multi-view data. As a result, directly learning from unposed images, namely pose-free or self-calibrated novel view synthesis[11, 24], is becoming increasingly important.

Existing pose-free methods[9, 11, 41] have shown that it is feasible to jointly learn scene representations and camera parameters without pose supervision, but their rendering designs still largely fall into two extremes. The pioneering RayZer [11] adopts a flexible implicit latent-space renderer and is easier to optimize. However, its predicted cameras are more likely to degenerate into latent variables that merely support internal rendering compatibility, rather than corresponding to physically grounded geometry. Its follow-up, E-RayZer [41], obtains more physically-grounded camera spaces by representing the scene with explicit 3D Gaussians, but at the cost of lower rendering quality and more sensitive optimization. This raises a natural question: can one simultaneously retain the advantages of both sides, namely high rendering quality and more physically-grounded camera estimation?

Our key insight is that geometric constraints remain essential for self-supervised novel view synthesis. However, such constraints need not be introduced only through fully explicit 3D primitives. Instead, they can also arise from how an implicit scene representation is structured and rendered under self-predicted cameras. Based on this observation, we propose IRIS, which represents the scene as a latent neural field and renders novel views by querying this field under self-predicted cameras. Concretely, rather than decoding free latent tokens as in RayZer or reconstructing fully explicit 3D primitives as in E-RayZer, IRIS builds a latent neural field as the scene representation. Novel views are rendered by querying this field along target rays: at each sampled 3D point, projected features from reference views are aggregated into a point-wise latent feature, and these point-wise features are further composed along the ray to predict the target color. This design preserves the flexibility and trainability of implicit modeling, while introducing stronger geometric structure than unconstrained latent rendering.

<table><tr><td>Property</td><td>RayZer[11]</td><td>E-RayZer[41]</td><td>IRIS</td></tr><tr><td>Ordered-input NVS</td><td>●</td><td>0</td><td></td></tr><tr><td>Unordered-input NVS</td><td>O</td><td>0</td><td></td></tr><tr><td>Grounded camera</td><td>O</td><td>●</td><td></td></tr><tr><td>Trainability</td><td></td><td></td><td></td></tr></table>

Table 1: High-level comparison ofRayZer, E-RayZer, and IRIS. , , and denote strong, moderate, and weak performance, respectively.

Experiments show that IRIS achieves strong novel view synthesis quality while maintaining competitive pose accuracy under the fully self-supervised setting. Compared with prior methods, IRIS is more robust to unordered inputs while avoiding the optimization dificulty of fully explicit 3D representations. Our main contributions can be summarized as follows: (1) We propose IRIS, a fully self-supervised framework for novel view synthesis from unposed multi-view images, based on a latent neural field representation and field querying under self-predicted cameras. (2) We present a field-based alternative to existing self-supervised rendering designs, which avoids both unconstrained latent decoding and fully explicit 3D primitives. (3) We show through experiments that IRIS achieves strong novel view synthesis quality and pose accuracy.

## 2 Related Work

In this section, we briefly review prior work on novel view synthesis (NVS). From the perspective of pose dependency, existing methods can be broadly categorized into three groups: (1) Pose-Conditioned NVS. These methods assume camera poses are available as input and synthesize novel views from posed images. (2) Pose-Supervised NVS. These methods do not require poses at inference time, but still rely on pose annotations or pose-related supervision during training. (3) Fully Self-Supervised NVS. These methods learn both camera pose estimation and novel view synthesis directly from unlabeled images, without using pose annotations in either training or inference.

Pose-Conditioned NVS. These methods assume camera poses are available at inference time and synthesize novel views from posed images. Representative examples include Neural Radiance Fields (NeRF [22]) and 3D Gaussian Splatting (3DGS [16]) However, both original methods require costly per-scene optimization, limiting practicality. To address this issue, later works developed generalizable pose-conditioned frameworks that amortize reconstruction and rendering across scenes. These methods can be broadly grouped into three categories (1) Radiance-field-based methods [4, 7, 8, 32, 34, 39] query radiance or latent properties at arbitrary 3D locations by combining spatial coordinates with image features from multiple source views, and perform ray-based rendering for target-view synthesis. (2) 3DGS-based methods [3, 5, 6, 12, 40, 43] directly predict 3D Gaussian primitives as the scene representation and render novel views through diferentiable splatting. (3) Neural network-only methods [14, 26] remove explicit 3D inductive biases and instead learn a latent rendering function directly from data. Although these methods substantially reduce optimization cost and improve eficiency, their applicability is still limited by the requirement for accurate camera poses at test time. This limitation motivates later work that further relaxes pose dependency during training and inference.

Pose-Supervised NVS. A second line of research relaxes the reliance on camera poses at inference time, while still using pose supervision during training. Compared with pose-conditioned methods, these methods learn scene representations directly from 2D images instead of taking camera poses as input. Trained on largescale posed image collections, such methods have shown promising performance in novel view synthesis. For example, LEAP [10] and PF-LRM [33] construct neural volumes in the canonical view coordinate system and perform neural rendering from the inferred representation. NoPoSplat [37] reconstructs scene Gaussians in the coordinate system of the first view directly from input images without requiring camera poses as input. Nevertheless, these methods still rely on ground-truth poses during training, which limits their scalability to fully unposed image collections.

Self-Supervised NVS. Fully self-supervised NVS aims to eliminate the reliance on camera poses during both training and inference, and instead nstead learns camera estimation and novel view synthesis. In practice, camera poses are often estimated using Structure-from-Motion (SfM) pipelines such as COLMAP [29], but these methods can be brittle under sparse-view or wide-baseline settings. Even when many views are available, incremental SfM may still discard dificult image regions by filtering feature correspondences that are inconsistent with the current reconstruction. With the diferentiability of rendering methods, it is possible to optimize camera parameters jointly with scene representations. Early works such as NeRF– [35] show that camera poses can be estimated for forward-facing scenes by jointly optimizing poses and radiance fields from simple initialization, while BARF [19] improves optimization stability through a coarse-to-fine registration strategy. More recently, RayZer [11] removes explicit 3D inductive bias and adopts latent rendering, achieving strong synthesis quality for ordered image sequences. However, its learned pose space is only weakly grounded in physically interpretable 3D geometry, which limits its 3D awareness. E-RayZer [41] moves one step further by introducing explicit 3D Gaussian reconstruction into the self-supervised setting. While such explicit 3D representations ofer stronger geometric grounding, they also make training substantially more dificult. To stabilize training, E-RayZer further adopts a fine-grained visual-overlap-based curriculum that starts from high-overlap samples and gradually expands to harder ones, while adaptively aligning heterogeneous data sources. Our method is also situated in the fully self-supervised setting, but difers from prior approaches by seeking a better balance between geometric grounding, rendering quality, and optimization stability: instead of relying on unconstrained latent rendering or explicit 3D Gaussian reconstruction alone, we learn a latent neural field to represent the scene.

## 3 Approach

In the fully self-supervised setting, our goal is to achieve strong novel view synthesis while learning camera parameters and scene representations directly from unlabeled multi-view images. We first revisit the two representative self-calibrated paradigms, namely RayZer[11] and E-RayZer[41], and clarify their key diferences. We then present IRIS, which combines camera prediction with a latent field representation.

![](images/412205eaa13e0776e8ed28bc114c2cc25cfe6ef0205001ba22335ef24f727ce8.jpg)  
Figure 1: Given a set of unordered input images, our method predicts camera poses and synthesizes high-quality novel views, while being trained entirely on images without camera annotations. Compared with prior methods, our approach produces sharper and more faithful renderings. The rightmost panels visualize the predicted camera poses.

## 3.1 Preliminaries: Implicit or Explicit

Given a set of unposed multi-view images $\boldsymbol { \mathcal { I } } = \left\{ I _ { i } \right\} _ { i = 1 } ^ { V }$ , following prior self-supervised learning frameworks, we split the input into two non-overlapping subsets: a reference set $\boldsymbol { \mathcal { I } } _ { \mathrm { r e f } }$ used for scene inference, and a target set ${ \cal T } _ { \mathrm { t g t } }$ used for photometric supervision. Let $\boldsymbol { \mathcal { V } } _ { \mathrm { r e f } }$ and $\mathcal { V } _ { \mathrm { t g t } }$ denote the index sets of the reference and target views, respectively, with $V _ { \mathrm { r e f } } = | \mathcal { V } _ { \mathrm { r e f } } |$

RayZer first patchifies all images into image tokens $f ,$ and introduces one learnable camera token per view, denoted by $p \in \mathbb { R } ^ { V \times d } .$ A transformer-based camera estimator updates them jointly:

$$
\{ f ^ { * } , p ^ { * } \} = E _ { \mathrm { c a m } } ( \{ f , p \} ) .
$$

A canonical reference view � is selected, and the relative pose of each view � with respect to � is predicted as

$$
\begin{array} { r } { \pi _ { i } = \mathrm { M L P } _ { \mathrm { p o s e } } ( [ p _ { i } ^ { * } , p _ { c } ^ { * } ] ) \in \mathbb { R } ^ { 9 } , \qquad P _ { i } = \Phi _ { \mathrm { S E } ( 3 ) } ( \pi _ { i } ) , } \end{array}
$$

where $\pi _ { i }$ consists of a 6D rotation parameterization and a 3D translation and $P _ { i } \in S E ( 3 )$ denotes the camera pose. RayZer further predicts a shared focal length from the canonical camera token,

$$
f o c a l = \mathrm { M L P _ { f o c a l } } ( { p _ { c } ^ { * } } ) ,
$$

which defines a shared intrinsic matrix � for all views, yielding the predicted camera parameters

$$
\mathcal { P } = \{ ( P _ { i } , K ) \} _ { i = 1 } ^ { V } .
$$

Each predicted camera is converted into a pixel-aligned Plücker ray map

$$
R _ { i } ^ { \mathrm { p l k } } = \Pi _ { \mathrm { p l k } } ( P _ { i } , K ) .
$$

For the reference-view subset $\scriptstyle { \mathcal { I } } _ { \mathrm { r e f } }$ , RayZer linearly tokenizes the predicted Plücker ray maps into ray tokens $r _ { \mathrm { r e f : } }$ fuses them with the corresponding image tokens $f _ { \mathrm { r e f } }$

$$
\begin{array} { r } { x _ { \mathrm { r e f } } = \mathrm { M L P } _ { \mathrm { f u s e } } ( \left[ f _ { \mathrm { r e f } } , r _ { \mathrm { r e f } } \right] ) , } \end{array}
$$

and predicts the latent scene representation from learnable scene tokens � via

$$
\{ z ^ { * } , x _ { \mathrm { r e f } } ^ { * } \} = E _ { \mathrm { s c e n e } } ( \{ z , x _ { \mathrm { r e f } } \} ) , \qquad z _ { \mathrm { s c e n e } } = z ^ { * } .
$$

Finally, the target views are rendered from the latent scene representation and the target-view ray tokens,

$$
\hat { I } _ { \mathrm { t g t } } = f _ { \phi } ^ { \mathrm { r e n d } } ( z _ { \mathrm { s c e n e } } , r _ { \mathrm { t g t } } ) ,
$$

and the model is trained with photometric supervision on target views:

$$
\mathcal { L } = \sum _ { ( I , \hat { I } ) \in ( \mathcal { I } _ { \mathrm { f g t } } , \hat { \mathcal { I } } _ { \mathrm { f g t } } ) } \Big ( \mathrm { M S E } ( \hat { I } , I ) + \lambda _ { \mathrm { p e r c } } \mathrm { P e r c e p } ( \hat { I } , I ) \Big ) .
$$

E-RayZer follows the same overall pose-first formulation as RayZer, but replaces RayZer’s implicit latent scene representation with explicit 3D Gaussians [16]. Using the same fused image–ray tokens $x _ { \mathrm { r e f } }$ as in RayZer, E-RayZer predicts the Gaussian scene as

$$
G _ { \mathrm { r e f } } = \left( f _ { \omega } ^ { \mathrm { g a u s s } } \circ E _ { \mathrm { s c e n e } } \right) \left( x _ { \mathrm { r e f } } \right)
$$

Here, $E _ { \mathrm { s c e n e } } ( x _ { \mathrm { r e f } } )$ denotes the scene encoding stage that performs multi-view aggregation and produces updated latent tokens, while $f _ { \omega } ^ { \mathrm { g a u s s } } ( \cdot )$ is a lightweight Gaussian decoder that maps these tokens to per-pixel Gaussian parameters $\{ g _ { i } \ = \ ( d _ { i } , q _ { i } , \stackrel { \textstyle - } { C _ { i } } , s _ { i } , \alpha _ { i } ) \} _ { i = 1 } ^ { V _ { \mathrm { r e f } } H W }$ along the reference-view rays. Each Gaussian $g _ { i }$ is parameterized by distance along the ray $d _ { i } ,$ orientation $q _ { i }$ , spherical-harmonic color coeficients $C _ { i } ,$ scale $s _ { i } ,$ , and opacity $\alpha _ { i } .$ . The target views are then rendered using the predicted target-view cameras

$$
P _ { \mathrm { { t g t } } } = \{ ( K , P _ { i } ) \mid i \in \mathcal { V } _ { \mathrm { { t g t } } } \} , \qquad \hat { I } _ { \mathrm { { t g t } } } = \pi ( \mathcal { G } _ { \mathrm { { r e f } } } , P _ { \mathrm { { t g t } } } )
$$

where � (·) denotes diferentiable Gaussian splatting.

Comparison and Motivation. RayZer benefits from flexible latent-space rendering and favorable optimization, but because camera prediction, scene inference, and rendering are jointly learned in latent space, the predicted cameras mainly need to remain mutually compatible with the internal renderer. As a result, the learned camera space is not necessarily physically grounded, and may partially degenerate into a latent variable that mainly serves rendering compatibility. By replacing latent rendering with explicit 3D Gaussians and diferentiable splatting, E-RayZer achieves a more geometrically grounded camera space. This improvement, however, comes at the cost of a heavier representation, a more sensitive optimization process, and slightly weaker rendering quality. These two paradigms therefore expose a clear trade-of between optimization flexibility and geometric grounding. Our goal is to preserve the optimization advantages of latent scene modeling, while replacing unconstrained latent decoding with a rendering interface that is more strongly constrained by multi-view geometric consistency.

## 3.2 IRIS Model

Our Insight. The above comparison suggests that some form of geometric inductive bias remains essential for self-supervised novel view synthesis. Without it, camera prediction and scene representation may drift toward a latent solution that is internally compatible with the renderer, but not physically grounded. At the same time, simply adopting fully explicit 3D representations is not ideal either, since it introduces a heavier representation and a more fragile optimization process. Our key idea is therefore to introduce geometric structure in a lighter-weight manner: instead of relying on unconstrained latent decoding as in RayZer, or fully explicit 3D primitives as in E-RayZer, IRIS represents the scene as a latentfield, which is queried and rendered under self-predicted cameras.

Shared Multi-View Encoder. Following RayZer and E-RayZer, IRIS adopts a pose-first formulation: camera parameters are predicted first and then used to define the scene for rendering. Unlike RayZer, however, IRIS does not use a dedicated camera estimator followed by a separate scene reconstructor conditioned on fused image–ray tokens. Instead, we use a single shared multi-view encoder to jointly produce per-view visual features and camera tokens, and directly retain the encoded view features for the subsequent latent-field representation. We instantiate the shared encoder with Muskie [18], since Muskie is a backbone designed for multi-view inputs that processes all input views jointly rather than encoding each frame independently. Given the input images $\boldsymbol { \mathcal { I } } = \{ I _ { i } \} _ { i = 1 } ^ { V }$ we tokenize each view into patch tokens and prepend one learnable camera token $\mathbf { \nabla } \mathcal { P } i$ to each view. The resulting multi-view tokens are processed jointly by the shared encoder:

$$
\{ f _ { i } , \ p _ { i } ^ { * } \} _ { i = 1 } ^ { V } = E _ { \theta } ^ { \mathrm { e n c } } ( \mathcal { I } ) ,
$$

where $f _ { i } \in \mathbb { R } ^ { h w \times d }$ denotes the encoded feature map of the �-th view, and $p _ { i } ^ { * } \in \mathbb { R } ^ { d }$ denotes its updated camera token. A camera head then predicts camera parameters in the same way as RayZer: a canonical reference view � is selected, each view pose is regressed relative to � from the updated camera tokens, and a single shared intrinsic matrix � is predicted for all views from the canonical token. We denote the resulting camera set by $\mathcal { P } = \{ ( P _ { i } , K ) \} _ { i = 1 } ^ { V }$

Scene Representation. Unlike RayZer and E-RayZer, IRIS does not reconstruct a separate set of latent scene tokens or explicit 3D primitives. Instead, IRIS directly preserves the reference-view features produced by the shared encoder and uses them, together with the self-predicted cameras, as a latent scene memory. Formally, we denote this memory by

$$
S = \{ ( f _ { k } , K , P _ { k } ) \} _ { k \in \mathcal { V } _ { \mathrm { r e f } } }
$$

This scene memory induces a latent field, whose value at a 3D query point � is obtained by projecting � into all reference views under the predicted cameras, sampling the corresponding viewwise features, and aggregating them across views. Concretely, for each reference view $k ,$ we first obtain

$$
v _ { k } ( x ) = \Pi _ { k } ( x ; f _ { k } , P _ { k } , K ) ,
$$

where $\Pi _ { k } ( \cdot )$ denotes projection followed by feature sampling from the �-th reference-view feature map. We then define the latent feature of � as

$$
\tilde { f } ( \boldsymbol x ) = F _ { S } ( \boldsymbol x ) = \mathcal { T } _ { \mathrm { v i e w } } \big ( \boldsymbol v _ { 1 } ( \boldsymbol x ) , \dots , \boldsymbol v _ { V _ { \mathrm { r e f } } } ( \boldsymbol x ) \big ) ,
$$

where $\mathcal { T } _ { \mathrm { v i e w } }$ is a transformer that fuses the projected features from all reference views into a single point-wise latent feature. In this sense, $F _ { S }$ is a latent field induced on the fly from multi-view features under the self-predicted cameras.

Rendering. Given a target ray $\boldsymbol { r } = \left( o _ { r } , d _ { r } \right)$ , we sample � points along the ray,

$$
x _ { m } = o _ { r } + z _ { m } d _ { r } , \qquad m = 1 , \ldots , M ,
$$

and query the latent field at each sampled point:

$$
\tilde { f } _ { m } = F _ { S } ( x _ { m } ) .
$$

We then compose the resulting point-wise features along the ray with a second transformer,

$$
h _ { r } = \mathcal { T } _ { \mathrm { r a y } } ( \tilde { f } _ { 1 } , \ldots , \tilde { f } _ { M } ) ,
$$

where $\mathcal { T } _ { \mathrm { r a y } }$ is a transformer that aggregates the sampled point features into a ray-wise feature. Unlike classical volumetric rendering, which predicts explicit color and density for each sample and combines them with a fixed rendering rule[22], $\mathcal { T } _ { \mathrm { r a y } }$ learns how the queried latent features should be composed to form the final ray representation. The rendered color is then predicted as

$$
\hat { C } ( r ) = f _ { \mathrm { r g b } } ( h _ { r } ) .
$$

We train IRIS with self-supervised photometric loss on target rays:

$$
\mathcal { L } = \sum _ { r \in \mathcal { R } _ { \mathrm { t g t } } } \Vert \hat { C } ( r ) - C ( r ) \Vert _ { 2 } ^ { 2 } .
$$

Since both latent-field querying and ray-wise rendering are conditioned on the self-predicted cameras, this objective jointly supervises camera estimation, scene representation learning, and rendering in a fully self-supervised manner.

Overall, IRIS can be viewed as a middle ground between RayZer and E-RayZer. Unlike RayZer, it does not rely on unconstrained latent decoding for rendering; unlike E-RayZer, it avoids fully explicit 3D primitives and their associated optimization dificulty. Instead, IRIS introduces geometric structure by constraining how the latent scene representation is queried and rendered under self-predicted cameras.

![](images/396ed280e591fd29b7eaf5e6f8662c499fe82b00a1cd83434c611e021af01b96.jpg)  
Figure 2: Overview of our method. Given unposed multi-view images, IRIS first predicts camera parameters and extracts latent reference features with a shared multi-view encoder, from which a latent neural field is constructed. Novel views are rendered by querying this field along target rays under the self-predicted cameras, where point-wise features are aggregated across reference views and then composed along each ray for color prediction.

## 4 Experiments

## 4.1 Experimental Setup

We report evaluation results for quality of novel view synthesis and pose estimation on several datasets to demonstrate the efectiveness of our method.

Implementation Details. We use Muskie[18] as the visual backbone and fine-tune it during training. For rendering, we sample 64 points within a fixed depth range along each ray. During training, we randomly sample a subset of rays for eficiency, while at inference time full-resolution images are rendered in chunks. We optimize the model with AdamW and a base learning rate of $4 \times 1 0 ^ { - 5 }$ , together with a warmup schedule. The encoder and pose predictor are trained with a 0.1× learning rate, while the renderer uses the full rate. Training is conducted on 8 A100 GPUs. In practice, training on DL3DV takes roughly 7 days, while training on the mixed-dataset setting takes about 14 days.

Training and Testing Data. We report results under both single-dataset and multi-dataset training settings. For singledataset training, models are trained exclusively on Re10K [42] or DL3DV [20]. For multi-dataset training, we train on a mixture of Re10K [42], DL3DV [20], CO3Dv2 [23], ARKitScenes [2], and WildRGBD [36], covering a diverse set of indoor and outdoor scenes. This training setup is comparable to that of E-RayZer[41] in terms of both dataset scale and diversity. For evaluation, we mainly benchmark both pose estimation and novel view synthesis on the DL3DV [20] test set, NRGBD [1], 7Scenes [30], and ScanNet++ [38]. Each test sample contains 24 frames, with 16 used as reference and the remaining 8 as target views. We evaluate pose estimation over all frames, while NVS quality is reported only on the 8 target views. To better reflect real-world usage, our evaluation primarily uses randomly sampled input images that do not assume any fixed temporal or pose ordering; we denote this setting as Rand. In addition, since we find that RayZer is only applicable to ordered image sequences due to its internal image index embeddings, we also report results on ordered input sequences, denoted as Seq, to enable a more complete comparison with RayZer.

Baselines. We compare with baselines from three categories: pose-conditioned methods (MVSplat [5] and LVSM [13]), posesupervised methods (NoPoSplat [37]), and self-supervised methods (SelfSplat [15], SPFSplat [9], RayZer [11], and E-RayZer [41]). We do not include comparisons with PF-LRM[33], since its oficial implementation is not publicly available. We note that NoPoSplat and SelfSplat are primarily designed for two-view input and show limited scalability to the multi-view setting. Moreover, among the self-supervised baselines, only RayZer and E-RayZer have been trained at relatively large scale—RayZer on DL3DV, and E-RayZer on an even broader mixture of datasets—whereas other methods (e.g., SPFSplat) are mainly pretrained on Re10K [42], whose camera trajectories are considerably less diverse. Therefore, we regard RayZer and E-RayZer as the primary baselines in our comparisons.

Evaluation Metrics. We evaluate novel view synthesis using the standard metrics PSNR, SSIM, and LPIPS. For pose estimation, we report relative pose accuracy (RPA) at thresholds of 5<sup>◦</sup>, 15<sup>◦</sup>, and 30<sup>◦</sup>, where the relative pose error is defined as the maximum of the rotation error and the translation-direction error between the predicted and ground-truth relative poses.

## 4.2 Results

Main Results. We first evaluate our method under the singledataset training setting, where the model is trained on a single

Ours

RayZer

E-RayZer  
Ours  
GT  
![](images/45e85693f2af40f61b69ae00f56f2bc18ff8653a59a5c83815045296916074c2.jpg)

Figure 3: Qualitative comparisons of pose estimation and rendering quality. We visualize all poses of the predicted trajectory as colored frustums, while for the COLMAP trajectory we display only every third pose as a black frustum to reduce overlap.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Self-supervised?</td><td colspan="3">Pose Accuracy</td><td colspan="3">Novel View Synthesis</td></tr><tr><td>@5°↑</td><td>@15° ↑</td><td>@30°↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td rowspan="2">Re10K</td><td>SPFSplat</td><td>X(MASt3R[17])</td><td>0.673</td><td>0.928</td><td>0.961</td><td>24.418</td><td>0.148</td><td>0.788</td></tr><tr><td>Ours</td><td>√</td><td>0.590</td><td>0.884</td><td>0.997</td><td>31.272</td><td>0.117</td><td>0.920</td></tr></table>

Table 2: Comparison under the single-source training setting on Re10K[42].

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">Pose Accuracy</td><td colspan="3">Novel View Synthesis</td></tr><tr><td>@5°↑</td><td>@15° ↑</td><td>@30°↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td rowspan="3">DL3DV[20] (Seq.)</td><td>RayZer</td><td>0.000</td><td>0.010</td><td>0.089</td><td>26.286</td><td>0.169</td><td>0.814</td></tr><tr><td>E-RayZer</td><td>0.658</td><td>0.824</td><td>0.902</td><td>21.187</td><td>0.272</td><td>0.701</td></tr><tr><td>Ours</td><td>0.600</td><td>0.869</td><td>0.918</td><td>25.402</td><td>0.215</td><td>0.791</td></tr><tr><td rowspan="3">DL3DV[20] (Rand.)</td><td>RayZer</td><td></td><td></td><td>Fail</td><td></td><td></td><td></td></tr><tr><td>E-RayZer</td><td>0.656</td><td>0.811</td><td>0.895</td><td>21.124</td><td>0.271</td><td>0.701</td></tr><tr><td>Ours</td><td>0.563</td><td>0.863</td><td>0.924</td><td>25.324</td><td>0.215</td><td>0.791</td></tr><tr><td rowspan="2">NRGBD[1]</td><td>E-RayZer</td><td>0.362</td><td>0.800</td><td>0.916</td><td>26.476</td><td>0.144</td><td>0.860</td></tr><tr><td>Ours</td><td>0.437</td><td>0.861</td><td>0.979</td><td>31.404</td><td>0.128</td><td>0.921</td></tr><tr><td rowspan="2">ScanNet++[38]</td><td>E-RayZer</td><td>0.019</td><td>0.252</td><td>0.545</td><td>21.992</td><td>0.259</td><td>0.744</td></tr><tr><td>Ours</td><td>0.036</td><td>0.333</td><td>0.627</td><td>22.938</td><td>0.276</td><td>0.763</td></tr><tr><td rowspan="2">7Scenes[30]</td><td>E-RayZer</td><td>0.312</td><td>0.729</td><td>0.882</td><td>26.727</td><td>0.175</td><td>0.870</td></tr><tr><td>Ours</td><td>0.278</td><td>0.749</td><td>0.856</td><td>30.783</td><td>0.140</td><td>0.907</td></tr></table>

Table 3: Comparison under the single-source training setting on DL3DV[20]. RayZer fails when testing with random-order views.

dataset. We begin with Re10K, a widely used benchmark with relatively regular camera trajectories. As shown in Tab. 2 and Fig. 4, compared with SPFSplat, our method yields weaker pose accuracy but substantially better novel view synthesis quality. We attribute the pose gap to the strong MASt3R-based[17] initialization used by SPFSplat, which introduces additional geometric priors through correspondence-based supervision and therefore makes the comparison less directly aligned with a fully self-supervised setting. We do not include RayZer[11] and E-RayZer[41] in this comparison, since pretrained weights on Re10K are not publicly available for these methods.

We next evaluate on DL3DV [20], which provides a more challenging benchmark due to its greater scene diversity and more varied camera motion. As shown in the first two rows of Tab. 3, compared with E-RayZer, our method achieves overall comparable pose accuracy while improving novel view synthesis quality by a clear margin. This suggests that our learned representation is more efective for rendering under self-predicted cameras. We further provide qualitative comparisons in Fig. 3. We take E-RayZer as the primary comparison baseline on this benchmark, since it is the most relevant large-scale self-supervised method and is explicitly designed to address the geometric limitations of RayZer. It is also worth noting that RayZer shows a clear dependence on sequence regularity, as its image-index embeddings provide a strong cue for shortcut learning via frame interpolation. Consistent with this observation, RayZer performs reasonably well under the sequential setting, but fails when the views are randomly ordered.

Ours

GT

![](images/8847bc04c42014cb08c7c956c79723522b0209116b7da390b47c2e7caa50a643.jpg)  
Figure 4: Qualitative comparisons with SPFSplat[9] on Re10K[42]. Our method produces sharper and more structurally consistent renderings, achieving lower LPIPS.

Beyond the in-domain comparison on DL3DV, we further study whether the learned representation can generalize to unseen datasets. The cross-dataset results in the remaining rows of Tab. 3 show that our method consistently demonstrates stronger generalization in novel view synthesis than E-RayZer, while maintaining broadly comparable pose accuracy. Notably, these evaluation image sets do not assume any fixed temporal or pose ordering, but are instead constructed to better reflect realistic multi-view captures. Since RayZer relies on ordered inputs and is therefore not suitable for this unordered cross-dataset setting, we do not treat it as a primary comparison method in the following evaluations.

Scaling to more training data. We further investigate whether enlarging the training distribution with mixed-source data can improve cross-dataset generalization. Comparing Tab. 4 with the DL3DV-only results in Tab. 3, we find that naively training on the mixed-source set does not consistently outperform single-source training for either E-RayZer or our method. Although our method still maintains clear advantages over E-RayZer in novel view synthesis on most benchmarks, the gains brought by mixed-source training are not stable, and in several cases the performance is even weaker than that of the DL3DV-only model. These observa tions suggest that simply increasing training data diversity does not automatically lead to better cross-dataset transfer for pose-free reconstruction and synthesis. A likely reason is that the mixed setting introduces substantially larger variations in scene statistics, camera motion patterns, and image distributions across datasets, making it harder for the model to learn a unified representation that generalizes well across domains. Overall, while our method remains competitive under mixed-source training, robust generalization across heterogeneous data sources likely requires more dedicated mechanisms than straightforward dataset mixing alone.

Comparison with supervised methods. We further compare our method with pose-supervised baselines on Re10K. As shown in Tab. 5, both LVSM [13] and MVSplat [5] rely on camera poses during both training and inference, whereas our method is trained without pose annotations and does not require camera poses at test time. Despite this weaker supervision, our method achieves the best PSNR and SSIM by a clear margin, outperforming both supervised baselines in reconstruction fidelity. Compared with LVSM, our method still achieves the best PSNR/SSIM and competitive LPIPS, indicating that it can learn an efective scene representation for high-quality novel view synthesis without explicit pose supervision. These results suggest that strong multi-view rendering quality can emerge from self-supervised learning alone, without relying on posed inputs or camera labels.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">Pose Accuracy</td><td colspan="3">Novel View Synthesis</td></tr><tr><td>@5° ↑</td><td>@15° ↑</td><td>@30°↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td rowspan="2">DL3DV[20]</td><td>E-RayZer</td><td>0.551</td><td>0.766</td><td>0.861</td><td>20.450</td><td>0.305</td><td>0.666</td></tr><tr><td>Ours</td><td>0.514</td><td>0.824</td><td>0.893</td><td>24.042</td><td>0.237</td><td>0.763</td></tr><tr><td>NRGBD[1]</td><td>E-RayZer Ours</td><td>0.280 0.360</td><td>0.782 0.885</td><td>0.913 0.984</td><td>25.837 30.879</td><td>0.173 0.114</td><td>0.842 0.922</td></tr><tr><td>ScanNet++[38]</td><td>E-RayZer Ours</td><td>0.011 0.033</td><td>0.262 0.284</td><td>0.545 0.588</td><td>21.950 22.191</td><td>0.267 0.293</td><td>0.743 0.749</td></tr><tr><td>7Scenes[30]</td><td>E-RayZer Ours</td><td>0.323 0.296</td><td>0.696 0.717</td><td>0.966 0.791</td><td>26.778 28.767</td><td>0.184 0.157</td><td>0.864 0.898</td></tr></table>

Table 4: Comparison under the mixed-source training setting.Comparison with supervised methods

![](images/1a9c048f71b3af48aa6530d8d000f460993f28f815ea75b4a23cacee299cce4f.jpg)

Figure 5: Qualitative comparison with supervised methods LVSM[13] and MVSplat[5].
<table><tr><td>Method</td><td>Training Supervision</td><td>Inference w. Camera Poses</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>LVSM[13]</td><td>2D + Camera</td><td>√</td><td>27.603</td><td>0.866</td><td>0.110</td></tr><tr><td>MVSplat[5]</td><td>2D + Camera</td><td>√</td><td>20.257</td><td>0.747</td><td>0.241</td></tr><tr><td>Ours</td><td>2D</td><td>x</td><td>31.272</td><td>0.920</td><td>0.117</td></tr></table>

Table 5: Comparison with pose-supervised baselines on Re10K[42].

## 4.3 Ablation studies

Backbone initialization. Tab. 6 shows that backbone initialization is critical for both pose estimation and novel view synthesis. Initializing from Muskie[18] yields substantially stronger performance than random initialization across all metrics, indicating that the gain comes from a stronger multi-view prior. This choice is motivated by the fact that Muskie is encouraged to discover crossview correspondences and to produce view-consistent features with stronger geometric awareness. Such a property is particularly suitable for our setting: the backbone is expected not only to extract per-view appearance features, but also to provide a feature space in which information from diferent views can be reliably aligned and aggregated. This initialization plays an important practical role in stabilizing optimization and improving reconstruction quality. Besides, freezing the Muskie backbone causes performance to collapse, showing that efective initialization alone is insuficient and that adapting the backbone to the target task is essential. We also find that DINOv3 performs poorly in this setting. A likely reason is that DINOv3 extracts features on a per-frame basis and lacks explicit cross-view interaction, making its features less suitable for regressing camera parameters in multi-view reconstruction.

<table><tr><td rowspan="2">Initialization</td><td rowspan="2">Frozen?</td><td colspan="3">Pose Accuracy</td><td rowspan="2"></td><td colspan="3">Novel View Synthesis</td></tr><tr><td>@5° ↑</td><td>@15° ↑</td><td>@30° ↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td>Random</td><td>x</td><td>0.197</td><td>0.459</td><td>0.589</td><td>22.851</td><td></td><td>0.300</td><td>0.707</td></tr><tr><td>DINOv3[31]</td><td>x</td><td>0.014</td><td>0.151</td><td>0.322</td><td></td><td>11.523</td><td>0.561</td><td>0.200</td></tr><tr><td>Muskie[18]</td><td>√</td><td>0.002</td><td>0.038</td><td>0.167</td><td></td><td>15.060</td><td>0.563</td><td>0.284</td></tr><tr><td>Muskie[18]</td><td>x</td><td>0.563</td><td>0.863</td><td>0.924</td><td></td><td>25.324</td><td>0.215</td><td>0.791</td></tr></table>

Table 6: Ablation on backbone initialization, evaluated on DL3DV [20].

<table><tr><td rowspan="2">Method</td><td colspan="3">Pose Accuracy</td><td colspan="3">Novel View Synthesis</td></tr><tr><td></td><td>@5°↑ @15° ↑</td><td>@30°↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td>3DGS</td><td colspan="6">Fail</td></tr><tr><td>Volume Rendering</td><td>0.494</td><td>0.829</td><td>0.990</td><td>28.058</td><td>0.176</td><td>0.868</td></tr><tr><td>Ours</td><td>0.497</td><td>0.877</td><td>0.995</td><td>31.272</td><td>0.117</td><td>0.920</td></tr></table>

Table 7: Ablation on the renderer design, evaluated on Re10K[42].

<table><tr><td rowspan="2">Train Setting</td><td rowspan="2">Method</td><td rowspan="2">Self-supervised ?</td><td colspan="3">Novel View Synthesis</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td rowspan="6">Re10K (2-view)</td><td>MVSplat[5]</td><td>x</td><td>22.720</td><td>0.168</td><td>0.823</td></tr><tr><td>LVSM[13]</td><td>x</td><td>27.580</td><td>0.120</td><td>0.895</td></tr><tr><td>NoPoSplat[37]</td><td>x</td><td>22.865</td><td>0.178</td><td>0.770</td></tr><tr><td>SPFSplat[9]</td><td>x</td><td>22.851</td><td>0.171</td><td>0.771</td></tr><tr><td>SelfSplat[15]</td><td>√</td><td>20.747</td><td>0.253</td><td>0.748</td></tr><tr><td>Ours</td><td>√</td><td>23.448</td><td>0.259</td><td>0.782</td></tr><tr><td>DL3DV (multi-view)</td><td>E-RayZer[41] Ours</td><td>√ √</td><td>14.899 23.708</td><td>0.287 0.251</td><td>0.703 0.794</td></tr></table>

Table 8: Ablation in the sparse-view case. Pose accuracy is omitted since it is not required in this setting. Methods such as MVSplat, LVSM, NoPoSplat, and SPFSplat rely on pose supervision or external supervised initialization.

Renderer. Tab. 7 studies the efect of renderer design. Replacing our implicit renderer with a standard volume-rendering formulation still yields reasonable pose accuracy, suggesting that the model can recover useful geometric cues under this design. However, the rendering quality drops consistently across all NVS metrics, indicating that our implicit renderer provides a more efective and flexible decoding interface for high-fidelity view synthesis. We further replace the scene representation with a 3DGS-style renderer, but observe training failure even with Muskie-based initialization. This observation is consistent with our findings on RayZer[11] and E-RayZer[41]: although careful data scheduling can partially alleviate the issue, optimizing an explicit 3DGS representation in a fully self-supervised setting remains highly unstable.

![](images/54865ef1aebfea5265a1cda6bfa64a0a0f19430d8274dafa675f4fc29097b3de.jpg)  
Figure 6: Rendered images and corresponding depth maps

## 4.4 Depth Visualization

Although our renderer does not explicitly predict per-sample density as in classical volume rendering, we can extract a soft depth map from the ray aggregation module. For a target ray $r ,$ let $\bar { \{ \boldsymbol { z } _ { m } \} } _ { m = 1 } ^ { M ^ { - } }$ denote the sampled depths and $\{ w _ { m } ( \boldsymbol { r } ) \} _ { m = 1 } ^ { M }$ the per-sample weights returned by the final ray-transformer block. These weights are obtained from the attention distribution of the last ray-aggregation layer and treated as a soft importance distribution. We compute

$$
D ( r ) = \sum _ { m = 1 } ^ { M } w _ { m } ( r ) z _ { m } .
$$

Applying this computation to all target rays yields the relative depth map in Fig. 6. Since training is pose-free and uses a normalized ray-sampling range, this depth should be interpreted as relative soft depth rather than metric depth.

Sparse-view case. Tab. 8 reports results on Re10K with only two input views. In this highly sparse setting, methods that rely on pose supervision or external supervised initialization, such as LVSM[13], MVSplat[5], NoPoSplat[37], and SPFSplat[9], remain strong baselines, with LVSM achieving the best overall performance. Among the self-supervised methods trained directly in the 2-view regime, our method consistently outperforms SelfSplat[15] and achieves the strongest PSNR and SSIM, although there is still a gap to the best pose-supervised model. This suggests that the extreme two-view setting remains challenging for fully self-supervised reconstruction. More importantly, when trained with multi-view inputs and tested with only two views, our method clearly surpasses E-RayZer[41] by a large margin. This indicates that the scene representation learned from multi-view training transfers more efectively to sparse-view inference, and retains strong rendering capability even when the number of available input views is drastically reduced.

## 5 Conclusion

We presented IRIS, a fully self-supervised framework for novel view synthesis from unposed multi-view images. IRIS represents the scene as a latent neural field queried and rendered under selfpredicted cameras, providing a practical middle ground between flexible latent modeling and geometrically grounded 3D reasoning. Experiments demonstrate strong rendering quality, competitive pose accuracy, and favorable generalization. We hope this work ofers useful insights to the community on designing future selfsupervised 3D vision models.

## 6 Appendix

Definition ofPose Error. Given a set ofpredicted and ground-truth camera-to-world poses, $\{ \hat { \mathbf { T } } _ { i } ^ { \mathrm { c } 2 \mathrm { w } } \} _ { i = 1 } ^ { B }$ and $\{ \mathbf { T } _ { i } ^ { \mathrm { c 2 w } } \} _ { i = 1 } ^ { B }$ , we first convert them to world-to-camera poses:

$$
\hat { \mathbf { T } } _ { i } ^ { \mathrm { w 2 c } } = ( \hat { \mathbf { T } } _ { i } ^ { \mathrm { c 2 w } } ) ^ { - 1 } , \qquad \mathbf { T } _ { i } ^ { \mathrm { w 2 c } } = ( \mathbf { T } _ { i } ^ { \mathrm { c 2 w } } ) ^ { - 1 } .
$$

We then evaluate all unordered view pairs

$$
\mathcal { P } = \{ ( i , j ) ~ | ~ 1 \le i < j \le B \} .
$$

For each pair $( i , j ) \in \mathscr { P }$ , the relative pose is defined as

$$
\begin{array} { r } { \hat { \mathbf { T } } _ { i  j } = \hat { \mathbf { T } } _ { i } ^ { \mathrm { w } 2 \mathrm { c } } ( \hat { \mathbf { T } } _ { j } ^ { \mathrm { w } 2 \mathrm { c } } ) ^ { - 1 } , \qquad \mathbf { T } _ { i  j } = \mathbf { T } _ { i } ^ { \mathrm { w } 2 \mathrm { c } } ( \mathbf { T } _ { j } ^ { \mathrm { w } 2 \mathrm { c } } ) ^ { - 1 } . } \end{array}
$$

Let

$$
\hat { \mathbf { T } } _ { i  j } = [ \begin{array} { c c } { \hat { \mathbf { R } } _ { i  j } } & { \hat { \mathbf { t } } _ { i  j } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} ] , \qquad \mathbf { T } _ { i  j } = [ \begin{array} { c c } { \mathbf { R } _ { i  j } } & { \mathbf { t } _ { i  j } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} ] ,
$$

where $\hat { \mathbf { R } } _ { i  j } , \mathbf { R } _ { i  j } \in S O ( 3 )$ and $\hat { \bf t } _ { i  j } , { \bf t } _ { i  j } \in \mathbb { R } ^ { 3 }$

The rotation error is measured by the geodesic angle between the predicted and ground-truth relative rotations:

$$
e _ { R } ( i , j ) = \operatorname { a r c c o s } ( \frac { \operatorname { T r } ( \mathbf { R } _ { i  j } ^ { \top } \hat { \mathbf { R } } _ { i  j } ) - 1 } { 2 } ) \cdot \frac { 1 8 0 } { \pi } .
$$

For translation, we only evaluate the direction and ignore the global scale. Let

$$
c _ { t } ( i , j ) = \frac { \mathbf { t } _ { i  j } ^ { \top } \hat { \mathbf { t } } _ { i  j } } { \Vert \mathbf { t } _ { i  j } \Vert _ { 2 } \Vert \hat { \mathbf { t } } _ { i  j } \Vert _ { 2 } } .
$$

Since the relative translation is treated as sign-ambiguous in our implementation, the translation-direction error is defined as

$$
e _ { t } ( i , j ) = \operatorname* { m i n } \bigl ( \operatorname { a r c c o s } ( c _ { t } ( i , j ) ) , \pi - \operatorname { a r c c o s } ( c _ { t } ( i , j ) ) \bigr ) \cdot \frac { 1 8 0 } { \pi } .
$$

Equivalently, this can be viewed as the unsigned angular error between the two translation directions.

Following the implementation, the relative pose error for each pair is defined as the maximum of the rotation and translationdirection errors:

$$
e _ { \mathrm { p o s e } } ( i , j ) = \operatorname* { m a x } \bigl ( e _ { R } ( i , j ) , e _ { t } ( i , j ) \bigr ) .
$$

Given an angular threshold $\delta \in \{ 5 ^ { \circ } , 1 5 ^ { \circ } , 3 0 ^ { \circ } \}$ , the relative pose accuracy (RPA) is computed as

$$
\mathrm { R P A } @ \delta = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } 1 \big [ e _ { \mathrm { p o s e } } ( i , j ) \le \delta \big ] ,
$$

where 1[·] denotes the indicator function.

Compare with standard volume rendering. Both standard volume rendering and our method can be written under a unified raydecoding form:

$$
\hat { C } ( r ) = \mathcal { R } \big ( \{ g ( x _ { m } , r ) \} _ { m = 1 } ^ { M } \big ) , \qquad x _ { m } = o _ { r } + z _ { m } d _ { r } ,
$$

where $\boldsymbol { r } = \left( o _ { r } , d _ { r } \right)$ is a target ray, $\{ x _ { m } \} _ { m = 1 } ^ { M }$ are the sampled 3D points along the ray, ${ \mathfrak { g } } ( x _ { m } , r )$ denotes the per-sample representation to be composed, and $\mathcal { R } ( \cdot )$ denotes the ray-wise composition rule.

For standard volume rendering, the scene is parameterized by explicit radiance-field quantities:

$$
g _ { \mathrm { v o l } } ( x _ { m } , r ) = ( \sigma _ { m } , c _ { m } ) , \qquad ( \sigma _ { m } , c _ { m } ) = F _ { \mathrm { v o l } } ( x _ { m } , d _ { r } ) ,
$$

where $\sigma _ { m }$ and $c _ { m }$ denote the density and color at the �-th sample. The final pixel color is obtained by a fixed hand-crafted compositing rule:

$$
\hat { C } _ { \mathrm { v o l } } ( r ) = \sum _ { m = 1 } ^ { M } T _ { m } \alpha _ { m } c _ { m } ,
$$

with

$$
\alpha _ { m } = 1 - \exp ( - \sigma _ { m } \delta _ { m } ) , \qquad T _ { m } = \prod _ { n = 1 } ^ { m - 1 } ( 1 - \alpha _ { n } ) .
$$

In contrast, our method does not predict explicit density and color at each sampled point. Instead, it constructs a latent field from multi-view features and uses the queried latent feature as the per-sample representation:

$$
g _ { \mathrm { I R I S } } ( x _ { m } , r ) = \tilde { f } _ { m } , \qquad \tilde { f } _ { m } = F _ { S } ( x _ { m } ) ,
$$

where

$$
F _ { S } ( x _ { m } ) = \mathcal { T } _ { \mathrm { v i e w } } \big ( v _ { 1 } ( x _ { m } ) , \dots , v _ { V _ { \mathrm { r e f } } } ( x _ { m } ) \big ) .
$$

These point-wise latent features are then composed along the ray by a learnable ray-wise aggregator:

$$
\begin{array} { r } { h _ { r } = \mathcal { T } _ { \mathrm { r a y } } ( \tilde { f } _ { 1 } , \cdot \cdot \cdot , \tilde { f } _ { M } ) , \qquad \hat { C } _ { \mathrm { I R I S } } ( r ) = f _ { \mathrm { r g b } } ( h _ { r } ) . } \end{array}
$$

Therefore, the key diference is that standard volume rendering uses explicit physical quantities $( \sigma , c )$ together with a fixed alpha-compositing rule, whereas our method uses latent point-wise features together with a learned ray-wise composition function.

Besides Table 7 in the main paper, Fig. 8 shows that replacing our learned implicit renderer with standard volume rendering produces visibly blurrier results and consistently worse quantitative performance. We attribute this gap to the fact that volume rendering uses a fixed hand-crafted composition rule, while our renderer learns to compose view-conditioned latent features adaptively along each ray, yielding a more expressive decoding interface for high-fidelity synthesis. This phenomenon has also been observed in prior work. For example, GNT[7, 32] showed that learned ray-wise rendering can better capture fine structures and appearance efects. Our result extends this observation to a new regime: unlike prior evidence obtained in posed settings, we verify it in a fully pose-free setting, where camera parameters themselves are learned under self-supervision.

Formal comparison of scene modeling and rendering. Table 9 summarizes the main diferences among RayZer, E-RayZer, and IRIS in scene modeling and rendering. RayZer represents the scene with latent tokens and renders target views using a learned decoder. E-RayZer instead predicts explicit 3D Gaussians and renders them by diferentiable splatting. In contrast, IRIS induces a latent neural field from multi-view features and performs learnable ray-wise rendering. This comparison highlights that our method difers from prior self-supervised paradigms in both the scene representation and the rendering interface.

More Qualitative comparisons. We present more qualitative comparisons in Fig. 7. x

<table><tr><td>Method</td><td>Scene modeling</td><td>Rendering</td></tr><tr><td>RayZer</td><td> $z _ { \mathrm { s c e n e } } = f _ { \psi } ^ { \mathrm { s c e n e } } \left( z _ { 0 } ^ { \mathrm { s c e n e } } , x _ { \mathrm { r e f } } \right)$ </td><td> $\hat { I } _ { \mathrm { t g t } } = f _ { \phi } ^ { \mathrm { r e n d } } \Bigl ( z _ { \mathrm { s c e n e } } , \mathrm { L i n e a r } \Bigl ( R _ { \mathrm { t g t } } ^ { \mathrm { p l k } } \Bigr ) \Bigr )$ </td></tr><tr><td>E-RayZer</td><td> $\mathcal { G } _ { \mathrm { r e f } } = f _ { \omega } ^ { \mathrm { g a u s s } } \circ f _ { \psi ^ { \prime } } ^ { \mathrm { s c e n e } } ( x _ { \mathrm { r e f } } )$ </td><td> $\hat { I } _ { \mathrm { t g t } } = \pi \big ( G _ { \mathrm { r e f } } , P _ { \mathrm { t g t } } \big )$ </td></tr><tr><td>IRIS</td><td> $\tilde { f } ( \boldsymbol { x } ) = \mathcal { T } _ { \mathrm { v i e w } } \Big ( v _ { 1 } ( \boldsymbol { x } ) , \dots , v _ { V _ { \mathrm { r e f } } } ( \boldsymbol { x } ) \Big )$ </td><td> $\hat { C } ( \boldsymbol { r } ) = f _ { \mathrm { r g b } } \circ \mathcal { T } _ { \mathrm { r a y } } \left( \tilde { f } _ { \mathrm { l } } , \dots , \tilde { f } _ { M } \right)$ </td></tr></table>

Table 9: Comparison of RayZer, E-RayZer, and IRIS in scene modeling and rendering.

E-RayZer  
Ours  
RayZer  
E-RayZer  
Ours  
GT  
![](images/95db9ade678fd8f6b7d61b28099ed0a4f9dce836622b1dac5be5b309ff7ff6fd.jpg)  
Figure 7: More qualitative comparisons of pose estimation and rendering quality. We visualize all poses of the predicted trajectory as colored frustums, while for the COLMAP trajectory we display only every third pose as a black frustum to reduc overlap.

Implementation details. For ray sampling, we use a fixed normalized depth range of [0.1, 1.0] in all experiments. We note that, in the fully self-supervised pose-free setting, the recovered camera translations are only defined up to a global scale, so the absolute scene depth is not directly meaningful. In this sense, the depth range mainly serves as a canonical sampling interval rather than a metric depth prior. We choose [0.1, 1.0] empirically, and find it suficient for stable training and rendering quality across datasets. Our implementation is mainly based on the GNT-style aggregation design. However, unlike GNT[32], which alternates multiple view and ray transformer blocks, our renderer uses only a single view aggregation stage to compute point-wise latent features and a single ray aggregation stage to decode the final ray color. In practice, the point-wise latent feature computation starts by projecting a 3D query point � into each reference view and sampling the corresponding feature

$$
v _ { k } ( x ) = \Pi _ { k } ( x ; f _ { k } , P _ { k } , K ) , \qquad k \in \mathcal { V } _ { \mathrm { r e f } } .
$$

Besides the sampled feature itself, we also compute a relative geometric cue $\Delta _ { k } ( x )$ and a binary validity mask $m _ { k } ( x )$ . The cue $\Delta _ { k } ( x )$ provides geometry-aware information about the relation between the target ray and the �-th reference-view observation, while $m _ { k } ( x )$ indicates whether the projected point falls into a valid image region and should participate in aggregation. The sampled features are first projected to the model width and max-pooled across views for initialization, and are then fused by a single view transformer to produce the point-wise latent feature $\tilde { f } ( x ) = F _ { S } ( x )$

Algorithm 1 Computing point-wise latent features   
Require: A 3D query point $x ;$ latent scene memoryAb   
$S = \{ ( f _ { k } , K , P _ { k } ) \} _ { k \in \mathcal { V } _ { \mathrm { r e f } } }$   
Require: View-transformer blocks $\{ \mathcal { T } _ { \mathrm { v i e w } } ^ { ( l ) } \} _ { l = 1 } ^ { L }$   
1: for each reference view $k \in \mathcal { V } _ { \mathrm { r e f } }$ do   
2: Project � into view � and sample the corresponding feature:   
$v _ { k } ( x ) = \Pi _ { k } ( x ; f _ { k } , P _ { k } , K )$   
3: Compute the relative geometric cue $\Delta _ { k } ( x )$ between the   
target ray and view �   
4: Compute the visibility / validity mask $m _ { k } ( x )$   
5: end for   
6: Project sampled view-wise features to the model width:   
$u _ { k } ^ { ( 0 ) } ( x ) = \phi _ { \mathrm { p r o j } } \bigl ( v _ { k } ( x ) \bigr ) , \qquad k = 1 , \ldots , V _ { \mathrm { r e f } }$   
7: Initialize by max-pooling across reference views:   
$q ^ { ( 0 ) } ( x ) = \operatorname* { m a x } _ { k \in \mathcal { V } _ { \mathrm { r e f } } } u _ { k } ^ { ( 0 ) } ( x )$   
8: for � = 1 to � do   
9: Update the point token by aggregating multi-view features:   
$\begin{array} { r } { q ^ { ( l ) } ( x ) = \mathcal { T } _ { \mathrm { v i e w } } ^ { ( l ) } \big ( q ^ { ( l - 1 ) } ( x ) , \{ u _ { k } ^ { ( 0 ) } ( x ) \} _ { k = 1 } ^ { V _ { \mathrm { r e f } } } , \{ \Delta _ { k } ( x ) \} _ { k = 1 } ^ { V _ { \mathrm { r e f } } } , \{ m _ { k } ( x ) \} _ { k = 1 } ^ { V _ { \mathrm { r e f } } } \big ) } \end{array}$   
10: end for   
11: Define the point-wise latent feature as   
$\tilde { f } ( x ) = F _ { S } ( x ) = q ^ { ( L ) } ( x )$   
12: return $\tilde { f } ( \boldsymbol x )$

Algorithm 2 Rendering a target ray from point-wise latent features   
Require: A target ray $\boldsymbol { r } = \left( o _ { r } , d _ { r } \right)$   
Require: Precomputed point-wise latent features $\{ \tilde { f } _ { m } ^ { ( 0 ) } \} _ { m = 1 } ^ { M }$ and   
their sample locations $\{ x _ { m } \} _ { m = 1 } ^ { M }$   
Require: Ray-transformer blocks $\{ ( \psi ^ { ( l ) } , \mathcal { T } _ { \mathrm { r a y } } ^ { ( l ) } ) \} _ { l = 1 } ^ { L }$   
1: Encode the sample locations and the target-ray direction:   
$e _ { m } ^ { \mathrm { p o s } } = \gamma _ { \mathrm { p o s } } ( x _ { m } ) , \qquad e ^ { \mathrm { d i r } } = \gamma _ { \mathrm { d i r } } ( d _ { r } )$   
2: for $l = 1$ to � do   
Update each point feature with point/ray encodings:   
$\tilde { f } _ { m } ^ { ( l - \frac { 1 } { 2 } ) } = \psi ^ { ( l ) } \bigl ( \tilde { f } _ { m } ^ { ( l - 1 ) } , e _ { m } ^ { \mathrm { p o s } } , e ^ { \mathrm { d i r } } \bigr ) ,$ $m = 1 , \ldots , M$   
4: Aggregate the point-wise features along the ray:   
$( \tilde { f } _ { 1 } ^ { ( l ) } , \ldots , \tilde { f } _ { M } ^ { ( l ) } ) = \mathcal { T } _ { \mathrm { r a y } } ^ { ( l ) } \big ( \tilde { f } _ { 1 } ^ { ( l - \frac { 1 } { 2 } ) } , \ldots , \tilde { f } _ { M } ^ { ( l - \frac { 1 } { 2 } ) } \big )$   
5: end for   
6: Normalize and pool the final point features:   
$h _ { r } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathrm { N o r m } ( \tilde { f } _ { m } ^ { ( L ) } )$   
7: Predict the final ray color:   
$\hat { C } ( r ) = f _ { \mathrm { r g b } } ( h _ { r } )$   
8: return $\hat { C } ( \boldsymbol { r } )$

For ray aggregation, given sampled points $\{ x _ { m } \} _ { m = 1 } ^ { M }$ on a target ray $\boldsymbol { r } = \left( o _ { r } , d _ { r } \right)$ , we first compute their point-wise latent features $\{ \tilde { f } _ { m } \} _ { m = 1 } ^ { M }$ and then inject sinusoidal encodings of both the sample locations and the target-ray direction:

![](images/2aef996bda35994c0217d6b889671a161cb67b927501557393a5f8b0e302570c.jpg)

Figure 8: Compared with standard volume rendering, implicit rendering improves quality of novel view synthesis.
<table><tr><td>Method</td><td>Self-supervised?</td><td>Mutli-view capable?</td><td>Remarks</td></tr><tr><td>MVSplat</td><td>x</td><td>x</td><td rowspan="3"></td></tr><tr><td>LVSM</td><td>x</td><td>√</td></tr><tr><td>PF-LRM</td><td>x</td><td>√</td></tr><tr><td>NoPoSplat</td><td>x</td><td>x</td><td rowspan="3">Only trained on Re10K</td></tr><tr><td>SelfSplat</td><td> $\checkmark$ </td><td>x</td></tr><tr><td>SPFSplat</td><td>X(MASt3R init)</td><td>√</td></tr><tr><td>RayZer</td><td> $\checkmark$ </td><td>√</td><td rowspan="2"></td></tr><tr><td>E-RayZer</td><td> $\checkmark$ </td><td> $\checkmark$ </td></tr></table>

<table><tr><td>M</td><td>N</td><td>E-RayZer (ms)</td><td>RayZer (ms)</td><td>IRIS (s)</td></tr><tr><td>4</td><td>1</td><td>31.27</td><td>36.37</td><td>2.74</td></tr><tr><td>8</td><td>2</td><td>32.77</td><td>52.01</td><td>7.92</td></tr><tr><td>12</td><td>4</td><td>44.67</td><td>80.63</td><td>20.75</td></tr><tr><td>16</td><td>8</td><td>70.96</td><td>133.35</td><td>49.24</td></tr></table>

Table 11: Inference latency for baselines and ours.

$$
e _ { m } ^ { \mathrm { p o s } } = \gamma _ { \mathrm { p o s } } ( x _ { m } ) , \qquad e ^ { \mathrm { d i r } } = \gamma _ { \mathrm { d i r } } ( d _ { r } ) .
$$

These enriched point features are then composed by a single ray transformer into a ray-wise representation $h _ { r }$ , from which the final color is predicted by $f _ { \mathrm { r g b } }$

Inference eficiency. To evaluate inference eficiency, we benchmark all methods on a fixed test scene and vary the numbers of context and target views, denoted by � and $N ,$ respectively. For each $( M , N )$ setting, we construct a single-scene batch by selecting � context frames and � target frames, move the batch to GPU, and measure the forward-pass latency using CUDA events. Following the benchmark script, each configuration is first warmed up for two runs and then timed for four runs, and we report representative mean latency values in Table 11. By default, the benchmark measures the model forward pass itself and excludes additional method-specific input preparation overhead, so the reported numbers primarily reflect the rendering cost of each method under the corresponding (�, �) configuration. As shown in Table 11, E-RayZer is the fastest among the three methods at all representative operating points, while RayZer is moderately slower but remains in the millisecond regime. Our method is noticeably more expensive, and its latency increases more rapidly as both � and � grow. This trend is expected, since our renderer explicitly performs learned view aggregation and ray aggregation. As a result, the computational cost grows with both the number of target views and the amount of reference views used for each ray. At the same time, we stress that this additional cost is closely tied to the stronger rendering capability of our method. Our method adopts a finer-grained rendering process over sampled 3D points and multi-view features. In this sense, the higher latency mainly reflects a quality–eficiency trade-of.

## References

[1] Dejan Azinović, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. 2022. Neural RGB-D Surface Reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6290–6301.

[2] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Jofe, Daniel Kurz, Arik Schwartz, and Elad Shulman. 2021. ARKitScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile RGB-D Data. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1). https://openreview.net/forum?id=tjZjv\_qh\_CE

[3] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent Sitzmann. 2024. pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction. In Proc. CVPR

[4] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. 2021. MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo. In Proc. ICCV.

[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. 2024. MVSplat: Eficient 3D Gaussian Splatting from Sparse Multi-View Images. arXiv 2403.14627 (2024).

[6] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat Jen Cham, and Jianfei Cai. 2024. MVSplat360: Benchmarking 360 Generalizable 3D Novel View Synthesis from Sparse Views. In Proceedings ofAdvances in Neural Information Processing Systems (NeurIPS).

[7] Wenyan Cong, Hanxue Liang, Peihao Wang, Zhiwen Fan, Tianlong Chen, Mukund Varma, Yi Wang, and Zhangyang Wang. 2023. Enhancing NeRF akin to Enhancing LLMs: Generalizable NeRF Transformer with Mixture-of-View-Experts. In ICCV.

[8] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).

[9] Ranran Huang and Krystian Mikolajczyk. 2025. No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views. arXiv preprint arXiv: 2508.01171 (2025).

[10] Hanwen Jiang, Zhenyu Jiang, Yue Zhao, and Qixing Huang. 2023. LEAP: Liberate Sparse-view 3D Modeling from Camera Poses. ArXiv 2310.01410 (2023).

[11] Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, et al. 2025. RayZer: A Self-supervised Large View Synthesis Model. arXiv preprint arXiv:2505.00702 (2025).

[12] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. 2025. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Transactions on Graphics (TOG) 44, 6 (2025), 1–16.

[13] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. 2024. LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias. arXiv 2410.17242 (2024).

[14] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. 2025. LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=QQBPWtvtcn

[15] Gyeongjin Kang, Jisang Yoo, Jihyeon Park, Seungtae Nam, Hyeonsoo Im, Sangheon Shin, Sangpil Kim, and Eunbyung Park. 2025. SelfSplat: Pose-free and 3D prior-free generalizable 3D Gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference. 22012–22022.

[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics 42, 4 (July 2023). https://repo-sam.inria.fr/fungraph/3dgaussian-splatting/

[17] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. 2024. Grounding Image Matching in 3D with MASt3R. arXiv preprint arXiv:2406.09756 (2024).

[18] Wenyu Li, Sidun Liu, Peng Qiao, Yong Dou, and Tongrui Hu. 2025. Muskie: Multi-view Masked Image Modeling for 3D Vision Pre-training. arXiv:2511.18115 [cs.CV] https://arxiv.org/abs/2511.18115

[19] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. 2021. BARF: Bundle-Adjusting Neural Radiance Fields. In IEEE International Conference on Computer Vision (ICCV).

[20] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. 2024. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22160–22169.

[21] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In Proc. ECCV.

[22] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

[23] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. 2021. Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction. In International Conference on Computer Vision.

[24] Mehdi S. M. Sajjadi, Aravindh Mahendran, Thomas Kipf, Etienne Pot, Daniel Duckworth, Mario Lučić, and Klaus Gref. 2023. RUST: Latent Neural Scene Representations from Unposed Imagery. CVPR (2023).

[25] Mehdi S. M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Gref, Noha Radwan, Suhani Vora, Mario Lucic, Daniel Duckworth, Alexey Dosovitskiy, Jakob Uszkoreit, Thomas Funkhouser, and Andrea Tagliasacchi. 2022. Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations. CVPR (2022). https://srt-paper.github.io/

[26] Mehdi S. M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Gref, Noha Radwan, Suhani Vora, Mario Lucic, Daniel Duckworth, Alexey Dosovitskiy, Jakob Uszkoreit, Thomas A. Funkhouser, and Andrea Tagliasacchi. 2021. Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations. CoRR abs/2111.13152 (2021).

[27] Johannes Lutz Schönberger and Jan-Michael Frahm. 2016. Structure-from-Motion Revisited. In Proc. CVPR.

[28] Johannes Lutz Schönberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. 2016. Pixelwise View Selection for Unstructured Multi-View Stereo. In Proc. ECCV.

[29] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Kon rad Schindler, Marc Pollefeys, and Andreas Geiger. 2017. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 3260–3269.

[30] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. 2013. Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images. In 2013 IEEE Conference on Computer Vision and Pattern Recognition. 2930–2937. doi:10.1109/CVPR.2013.377

[31] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. 2025. Dinov3. arXiv preprint arXiv:2508.10104 (2025).

[32] Mukund Varma T, Peihao Wang, Xuxi Chen, Tianlong Chen, Subhashini Venugopalan, and Zhangyang Wang. 2023. Is Attention All That NeRF Needs?. In The Eleventh International Conference on Learning Representations. https: //openreview.net/forum?id=xE-LtsE-xx

[33] Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan, Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai Zhang. 2024. PF-LRM: Pose-Free Large Reconstruction Model for Joint Pose and Shape Prediction. In ICLR.

[34] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P. Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas A. Funkhouser. 2021. IBRNet: Learning Multi-View Image-Based Rendering. In Proc. CVPR.

[35] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. 2021. NeRF−−: Neural Radiance Fields Without Known Camera Parameters. arXiv preprint arXiv:2102.07064 (2021).

[36] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. 2024. RGBD Objects in the Wild: Scaling Real-World 3D Object Learning from RGB-D Videos. arXiv:2401.12592 [cs.CV]

[37] Botao Ye, Sifei Liu, Haofei Xu, Li Xueting, Marc Pollefeys, Ming-Hsuan Yang, and Peng Songyou. 2024. No Pose, No Problem: Surprisingly Simple 3D Gaussian

Splats from Sparse Unposed Images. arXiv preprint arXiv:2410.24207 (2024).

[38] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. 2023. ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes. In Proceedings ofthe International Conference on Computer Vision (ICCV).

[39] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. 2021. PixelNeRF: Neural Radiance Fields from One or Few Images. In Proc. CVPR.

[40] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. 2024. GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting. European Conference on Computer Vision (2024).

[41] Qitao Zhao, Hao Tan, Qianqian Wang, Sai Bi, Kai Zhang, Kalyan Sunkavalli, Shubham Tulsiani, and Hanwen Jiang. 2026. E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training. In CVPR.

[42] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyfe, and Noah Snavely. 2018. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817 (2018).

[43] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. 2025. Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats. In Proceedings of the IEEE/CVF International Conference on Computer Vision.