# AquaFlow: A Monocular Gaussian Splatting SLAM for Underwater Streaming Reconstruction

Yingxiang Xu<sup>∗</sup>, Kerui Ren<sup>∗</sup>, Wenqi Guo, Changjian Jiang, Tao Lu, Linning Xu, Mulin Yu<sup>†</sup>

Abstract—Recent monocular 3D Gaussian Splatting (3DGS) streaming reconstruction methods have achieved impressive performance by balancing reconstruction quality and efficiency. However, extending these frameworks to underwater scenes remains challenging due to severe visual degradation, such as light attenuation and scattering, which degrades camera pose tracking and distorts scene geometry. To address these challenges, we propose AquaFlow, a monocular Gaussian Splatting streaming reconstruction framework for efficient and high-fidelity underwater reconstruction. Specifically, AquaFlow fine-tunes a 3D vision foundation model on large-scale underwater data for robust pose and pointmap estimation, and introduces a mediumguided incremental Gaussian initialization strategy for streaming mapping. Furthermore, we develop a streaming-compatible hybrid scene representation that integrates structured, distanceconditioned neural Gaussians with a physics-inspired optical model to compensate for underwater image formation effects, enabling accurate scene reconstruction. We evaluate AquaFlow on a comprehensive dataset of 62 diverse underwater trajectories, collected from both public benchmarks and in-the-wild web videos across various scales. Extensive experiments demonstrate that AquaFlow achieves state-of-the-art tracking and rendering performance, reducing average localization error by 13.2% and improving PSNR by 4.74 dB compared to WaterSplat-SLAM.

Index Terms—Underwater Reconstruction, Streaming Reconstruction, 3D Gaussian Splatting, SLAM, Feed-Forward Models.

## I. INTRODUCTION

IGH-RESOLUTION, photorealistic underwater 3D reconstruction is fundamental to benthic habitat mapping, underwater resource assessment, marine infrastructure inspection, and autonomous subsea navigation [1]–[5]. Yet, scalable and efficient 3D reconstruction from continuous underwater video remains severely underexplored. This slow progress is primarily because complex optical attenuation and scattering induce severe visual degradation, making it extremely challenging to rapidly estimate stable camera poses and achieve high-accuracy scene reconstruction within short timeframes [6]. Despite these challenges, monocular cameras

remain a highly desirable solution: compared to bulky sonar or multi-sensor suites, they serve as an exceptionally lightweight, cost-effective, and flexible sensing modality that captures rich visual textures while minimizing vehicle weight, drag, and hydrodynamic impact [7]–[10].

3D Gaussian Splatting (3DGS) [11] provide an efficient representation for high-fidelity rendering and open new opportunities for underwater reconstruction. However, existing underwater 3DGS methods are predominantly offline [12]–[15], requiring complete image sequences alongside pre-estimated camera poses and sparse point clouds as input, while incurring severe computational overheads for Gaussian optimization and densification even setting aside the lengthy preprocessing time of Structure-from-Motion (SfM). Moreover, existing methods are predominantly evaluated on a limited number of smallscale scenes, leaving their scalability to extended video sequences and their generalizability across diverse, complex underwater topologies largely unproven.

To overcome these efficiency and scalability bottlenecks, streaming reconstruction has emerged as a promising paradigm, with recent advances in 3DGS-based SLAM [16]– [18] providing a solid technical foundation to incrementally update explicit scene representations for time-efficient mapping and high-quality novel-view synthesis. However, these methods are primarily tailored for terrestrial environments, and directly applying them to underwater scenarios introduces two tightly coupled challenges: robust pose tracking and mediumaware scene modeling. First, wavelength-dependent attenuation, non-uniform illumination, and weakly textured sandy or rocky seafloors substantially reduce cross-frame visual correspondence. The resulting pose errors and trajectory drift destabilize dense point-map initialization and propagate into the Gaussian representation, ultimately degrading both geometry and rendering quality. Although recent SLAM systems [19]– [21] employ 3D vision foundation models to improve visual matching, their predominantly terrestrial training data limit their ability to generalize to subsea imagery governed by complex light transport. Second, the clear-medium assumption underlying standard 3DGS breaks down underwater, where image formation is strongly affected by distance-dependent spectral attenuation and volumetric backscattering [22]. Without modeling these effects, 3DGS primitives absorb them into static Gaussian colors, leading to geometric distortions and rendering artifacts.

To address these challenges, we present AquaFlow, a monocular 3D Gaussian Splatting SLAM framework that combines underwater-adapted geometric priors with physicsguided hybrid scene modeling for scalable, high-fidelity subsea reconstruction (Fig. 1). For robust tracking, AquaFlow finetunes a 3D vision foundation model on large-scale subsea corpora, thereby distilling robust geometric priors for accurate pose tracking and dense point-map estimation. For online mapping, we develop a streaming-compatible hybrid scene-medium representation that integrates distance-aware neural Gaussian primitives with an explicit underwater image formation model. During streaming, the physical model guides medium-aware Gaussian initialization in unobserved regions, while distance-encoded Gaussian parameters adaptively capture fine-grained underwater imaging effects not fully represented by the explicit formulation. By unifying explicit optical modeling with implicit neural representations, AquaFlow decouples intrinsic scene geometry from complex (ii) Gaussian-based optical degradation, enabling highly accurate and photorealistic subsea mapping.

![](images/4a61910be74b0adc48490f9fd7300ed0bcb8e0c6759ad2f26471432d2f08b949.jpg)  
Fig. 1. Overview of the AquaFlow framework. Given a long, continuous underwater video sequence, AquaFlow performs online streaming 3D reconstruction, rapidly estimating camera poses, dense pointmaps, a highly compact 3D Gaussian map, and physical medium parameters. To achieve this, AquaFlow seamlessly integrates domain-adapted underwater 3D geometric priors, a medium-guided Gaussian initialization scheme, and a hybrid scene representation combining structured neural Gaussians with physics-based medium modeling, ultimately enabling accurate subsea structural recovery and photorealistic novel view synthesis.

Our main contributions are summarized as follows:

• We fine-tune a 3D vision foundation model on extensive subsea datasets to bridge the terrestrial-to-underwater domain gap, providing robust geometric priors that empower camera pose tracking and dense depth initialization.

• We design a streaming-compatible hybrid scene representation combining distance-aware neural Gaussians with a physical underwater optical model, effectively disentangling intrinsic appearance from distance-dependent attenuation and backscattering.

• We compile a comprehensive underwater evaluation benchmark containing 62 diverse real-world sequences (25 from public datasets and 37 challenging in-thewild subsea videos). Extensive experiments validate that AquaFlow achieves state-of-the-art accuracy in both camera tracking and high-fidelity 3D reconstruction.

## II. RELATE WORKS

## A. Underwater Visual SLAM and Pose Estimation

To tackle severe image degradation, scattering, and low illuminance in underwater environments, classical visual SLAM systems often suffer from feature tracking failures or misalignments. Existing studies primarily address these issues from a 2D perspective by applying self-adaptive color or turbidity restoration [23] and lightweight GANs [24] as pre-processing backbones to restore image quality, replacing conventional hand-crafted features with domain-tailored learned descriptors [25], [26], or leveraging deep optical flow and spatial attention mechanisms [27], [28] to adapt to weakly textured underwater regions. However, these 2D methods only yield sparse point clouds and coarse scene geometry, while neglecting the strong generalizability and dense 3D structure prediction capabilities of modern 3D vision foundation models for downstream matching and pose estimation in degraded underwater regions.

## B. Offline Underwater 3D Gaussian Splatting Reconstruction

To achieve photorealistic novel view synthesis, recent efforts have extended 3DGS to underwater environments by jointly optimizing Gaussian primitives and explicit physical light transport models from pre-computed Structure-from-Motion (SfM) inputs. For instance, methods like Water-Splatting [12] and SeaSplat [13] integrate image formation models into 3DGS to disentangle scene radiance from attenuation and backscatter, while UW-GS [29] and RUSplatting [14] incorporate depth- or range-dependent priors to restore appearance and geometry. However, as scene scale expands, these offline paradigms demand excessive computation time for offline Gaussian densification, pruning, and optimization, failing to accommodate the continuous streaming video nature of practical underwater data collection. Furthermore, they heavily rely on simplified physical medium formulations that fail to capture complex underwater imaging dynamics, and are typically evaluated only on narrow, small-scale datasets with sparse frame counts.

## C. Streaming Gaussian Splatting Reconstruction

Combining 3DGS with SLAM enables real-time state estimation to guide Gaussian initialization and rapid scene learning, presenting a promising paradigm for online underwater reconstruction [30]–[33]. However, current general streaming reconstruction algorithms still face significant challenges when applied to underwater environments, including severe pose drift, erroneous Gaussian growth, and suboptimal scene representation. Meanwhile, WaterSplat-SLAM [34], the concurrent underwater streaming method, attempts to address underwater degradation by leveraging pre-computed semantic segmentation to guide frame tracking and Gaussian optimization. Nevertheless, its heavy reliance on semantic masks renders it vulnerable to segmentation failures and computationally unfeasible in complex or low-visibility underwater scenes.

![](images/94d3d7ec924ceee2d84b49fd82a04c7446182e0fb39a79b51ec56dc85a0cd3a3.jpg)  
Fig. 2. Overview of the AquaFlow streaming 3D reconstruction pipeline. Given incoming monocular underwater frames, AquaFlow first predicts dense pointmaps and descriptors using a domain-adapted 3D foundation model to establish reliable pixel correspondences for relative pose optimization. Based on match counts and parallax, frames are dynamically categorized into keyframes, mapping frames, or common frames. Keyframes and mapping frames trigger medium-guided incremental initialization, where an online learned physical medium model removes attenuation and backscatter to yield clean base colors and structure-aware Gaussian seeds. Finally, these primitives are organized into a distance-aware structured neural Gaussian representation and decoded via the LGPD module, which is jointly optimized alongside explicit physical medium modeling for high-fidelity rendering.

## III. METHODOLOGY

We address the problem of incrementally recovering a highfidelity 3D underwater scene from a single monocular RGB video. Given an incoming sequence $\{ I _ { t } \} _ { t = 1 } ^ { \mathbf { \overline { { \cal N } } } }$ our objective is to jointly estimate the camera trajectories $\{ \mathbf { \bar { T } } _ { t } \} _ { t = 1 } ^ { N } \in S E ( 3 )$ construct a dense Gaussian scene map $\{ \mathcal { G } _ { j } \} _ { j = 1 } ^ { M }$ , and resolve the optical parameters governing underwater medium attenuation and backscatter.

To overcome severe domain shifts and complex mediuminduced optical degradation inherent in underwater environments, AquaFlow decomposes the streaming reconstruction process into four synergistic modules: 1) an underwateradapted 3D geometric prior (Sec. III-A), 2) tracking and global optimization (Sec. III-B), 3) a hybrid scene-medium neural representation (Sec. III-C), and 4) an explicit joint optimization objective (Sec. III-D).

## A. Generalizable Underwater 3D Prior

1) Data Collection and Processing: To address the lack of generalizable multi-view geometric foundation models for underwater environments, we construct the first large-scale, highly diverse underwater 3D dataset tailored for model adaptation. Spanning a wide variety of synthetic and real-world underwater environments, including ocean floors, coral reefs, canyons, and submerged structures—our dataset provides rich geometric supervision.

For synthetic data [35], [36], camera poses and depth maps are directly extracted from simulator annotations. For realworld sequences, we employ domain-specific strategies: sweetcorals [37] is reconstructed via COLMAP-based photogrammetry [38], while FLSea-stereo [39] combines stereo depth estimation with feature-based pose-only bundle adjustment. All image-depth pairs are undistorted and calibrated to enforce rigid geometric consistency. To eliminate noisy supervision, we apply joint geometric and semantic masking to remove dynamic objects, invalid depth regions, specular highlights, and scattering artifacts. Finally, valid pixels are back-projected to evaluate cross-view spatial overlap; only frame pairs with sufficient overlap (≥ 30%) and verified geometric consistency are retained for training.

2) Model finetuning: To establish a robust 3D vision foundation model tailored for underwater environments, we select MASt3R [40] as our base architecture due to its rapid inference speed, exceptional stability in small-baseline geometric estimation, and strong alignment with front-end tracking requirements in SLAM frameworks. Given a pair of uncalibrated underwater images $\mathbf { I } _ { i } , \mathbf { I } _ { j } \in \mathbb { R } ^ { H \times W \times 3 }$ , MASt3R predicts the 3D geometry, per-pixel descriptors, and their associated confidence maps in a single forward pass:

$$
( \mathbf { X } _ { i } ^ { i } , \mathbf { X } _ { i } ^ { j } , \mathbf { D } _ { i } ^ { i } , \mathbf { D } _ { i } ^ { j } , \mathbf { Q } _ { i } ^ { i } , \mathbf { Q } _ { i } ^ { j } , \mathbf { C } _ { i } ^ { i } , \mathbf { C } _ { i } ^ { j } ) = f _ { \pmb { \theta } } ( \mathbf { I } _ { i } , \mathbf { I } _ { j } ) ,\tag{1}
$$

where $\mathbf { X } _ { i } ^ { i } , \mathbf { X } _ { i } ^ { j } \in \mathbb { R } ^ { H \times W \times 3 }$ denote the dense 3D pointmaps of images $\mathbf { I } _ { i }$ and $\mathbf { I } _ { j }$ expressed in the local coordinate frame of camera i. The terms $\mathbf { D } _ { i } ^ { i } , \mathbf { D } _ { i } ^ { j } \ \in \ \mathbb { R } ^ { H \times W \times d }$ represent the per-pixel feature descriptor maps for images $\mathbf { I } _ { i }$ and $\mathbf { I } _ { j } ,$ respectively. To handle underwater noise, $\bar { \mathbf { Q } _ { i } ^ { i } } , \mathbf { Q } _ { i } ^ { j } \in \mathbb { R } ^ { H \times W \times 1 }$ provide local confidence estimates for the predicted 3D point coordinates, while $\mathbf { C } _ { i } ^ { i } , \mathbf { C } _ { i } ^ { j } \in \mathbb { R } ^ { H \times W \times 1 }$ represent the descriptor confidences used for robust feature matching.

We fine-tune MASt3R on our dataset of 224K calibrated underwater two-view image pairs. We omit scale normalization to directly optimize metric-scale 3D predictions and the overall objective combines 3D regression and feature matching:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { c o n f } } + \beta \mathcal { L } _ { \mathrm { m a t c h } }\tag{2}
$$

where ${ \mathcal { L } } _ { \mathrm { c o n f } }$ optimizes metric pointmap regression along with a log-confidence regularization term. The matching objective ${ \mathcal { L } } _ { \mathrm { m a t c h } }$ leverages an InfoNCE contrastive loss over ground-truth point correspondences established according to their proximity in 3D space [40].

## B. Tracking and GLobal Optimization

With the underwater 3D prior, the tracking frontend estimates relative transformations and predicts dense point clouds as new frames arrive in a stream. For an incoming frame $\mathbf { I } _ { t } ,$ we first execute $f _ { \pmb { \theta } } ( \mathbf { I } _ { t } , \mathbf { I } _ { k } )$ by passing $\mathbf { I } _ { t }$ together with the latest keyframe $\mathbf { I } _ { k }$ into our fine-tuned foundation model. This feed-forward inference directly yields the dense pointmaps $( \mathbf { X } _ { t } ^ { t } , \mathbf { X } _ { t } ^ { k } )$ , per-pixel descriptors $( \mathbf { D } _ { t } ^ { t } , \mathbf { D } _ { t } ^ { k } )$ , and their associated confidence maps in the coordinate frame of camera t. To establish a set of reliable pixel correspondences $\mathcal { M } _ { t , k }$ between I<sub>t</sub> and $\mathbf { I } _ { k }$ , we perform dense descriptor matching using $\mathbf { D } _ { t } ^ { t } , \mathbf { D } _ { t } ^ { k }$ combined with reciprocal ray-error optimization over predicted 3D geometries [19]. Subsequently, the relative transformation $\mathbf { T } _ { k t } \ \in \ \mathrm { S i m } ( 3 )$ is estimated by minimizing the confidenceweighted pointmap alignment residual:

$$
\mathbf { T } _ { k t } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { T } _ { k t } } \sum _ { ( u , v ) \in \mathcal { M } _ { t , k } } \rho \left( \left\| \mathbf { X } _ { k } ^ { k } ( v ) - \mathbf { T } _ { k t } \mathbf { X } _ { t } ^ { t } ( u ) \right\| _ { 2 } \right) ,\tag{3}
$$

where $\mathbf { X } _ { t } ^ { t } ( u )$ and $\mathbf { X } _ { k } ^ { k } ( v )$ denote the predicted 3D point coordinates at matched pixel positions u and v, respectively and $\rho ( \cdot )$ represents the Huber robust loss function.

Specifically, keyframes are spawned when the size of the reliable match set $| \mathcal { M } _ { t , k } |$ falls below a predefined tracking threshold $\tau _ { \mathrm { k e y f r a m e } } .$ This triggers the insertion of $\mathbf { I } _ { t }$ into the backend pose graph to jointly optimize all absolute keyframe poses $\{ \mathbf { T } _ { i } \} _ { i \in \mathcal { K } }$ across active spatial edges $( i , j ) \in \mathcal { E }$ via:

$$
\{ \mathbf { T } _ { i } ^ { * } \} = \operatorname* { m i n } _ { \{ \mathbf { T } _ { i } \} } \sum _ { ( i , j ) \in { \mathcal { E } } ( u , v ) \in { \mathcal { M } } _ { i , j } } \rho \left( \| \mathbf { X } _ { i } ^ { i } ( v ) - \mathbf { T } _ { i } \mathbf { T } _ { j } ^ { - 1 } \mathbf { X } _ { j } ^ { j } ( u ) \| _ { 2 } \right) ,\tag{4}
$$

where K denotes the set of keyframes, $\mathbf { X } _ { i } ^ { j } ( u )$ and $\mathbf { X } _ { i } ^ { i } ( v )$ represent the 3D point predictions for matched pixel pair $( u , v ) \in \mathcal { M } _ { i , j }$ in their local camera frames, and $\mathbf { T } _ { i } \in \mathrm { S E } ( 3 )$ denotes the absolute pose of keyframe i. The robust Huber loss $\rho ( \cdot )$ effectively mitigates false correspondences caused by underwater optical degradation. Mapping frames are selected when the average pixel parallax calculated over matched points in $\mathcal { M } _ { t , k }$ with high geometric confidence $\mathbf { Q } _ { t } ^ { t }$ exceeds a baseline threshold $\tau _ { \mathrm { m a p f r a m e } } ,$ , initiating the insertion of new

3D Gaussians into high-certainty areas; and common frames, which fail both criteria, are utilized solely to update and refine existing Gaussian parameters.

## C. Hybrid Scene Representation

1) Distance-aware Structured Neural Gaussian: In our framework, for each Gaussian primitive $\mathcal { G } _ { j }$ in the dense map $\{ \mathcal { G } _ { j } \} _ { j = 1 } ^ { M }$ , its position $\mu _ { j }$ and opacity $\alpha _ { j }$ are explicitly learned, whereas its structural attributes (rotation $q _ { j }$ and scale $s _ { j } )$ along with the 0-th order SH component $\mathbf { c } _ { 0 , j }$ are dynamically decoded via a structure-aware, distance-guided perception decoder. This design bypasses the limitation of standard 3DGS, whose sparse, low-degree SH representations fail to capture complex, distance-dependent underwater color degradation. Specifically, the perception decoder processes structured Gaussian tokens to decode these attributes as residual increments relative to their initial states, effectively compensating for the coarse initialization inherent in streaming reconstruction. Consequently, each Gaussian primitive explicitly models appearance and geometry under both viewing direction and observation distance, enabling robust continuous scene modeling.

Concretely, let $\mathbf { p } _ { \mathrm { c a m } }$ denote the camera center and $\pmb { \mu } _ { i }$ the center of the i-th Gaussian. We define its observation distance and viewing direction as

$$
d _ { i } = \| { \pmb { \mu } } _ { i } - \mathbf { p } _ { \mathrm { c a m } } \| _ { 2 } , \qquad \mathbf { v } _ { i } = \frac { \pmb { \mu } _ { i } - \mathbf { p } _ { \mathrm { c a m } } } { d _ { i } } .\tag{5}
$$

Here, $d _ { i }$ approximates the light propagation distance in the water medium, while $\mathbf { v } _ { i }$ describes the observation direction. They are concatenated into an observation condition:

$$
\mathbf { o } _ { i } = \left[ \mathbf { v } _ { i } , d _ { i } \right] ^ { \top } \in \mathbb { R } ^ { 4 } .\tag{6}
$$

To model nonlinear observation-dependent appearance variations, we apply sinusoidal encoding to $\mathbf { o } _ { i } \mathbf { : }$

$$
\begin{array} { r } { \gamma ( \mathbf { o } _ { i } ) = \left[ \mathbf { o } _ { i } , \left\{ \sin ( 2 ^ { l } \pi \mathbf { o } _ { i } ) , \cos ( 2 ^ { l } \pi \mathbf { o } _ { i } ) \right\} _ { l = 0 } ^ { L - 1 } \right] , } \end{array}\tag{7}
$$

where $L$ is the number of frequency bands. This encoding enables the representation to implicitly capture complex, rangedependent underwater imaging phenomena that are otherwise difficult to model explicitly.

To maintain a stable Gaussian map and faithfully preserve continuous underwater terrains, such as sandy beds and mounds, we further enforce local structural organization during streaming reconstruction. Gaussians are incremently voxelized by their 3D centers and grouped within each voxel, with each group containing at most $K$ primitives. Each Gaussian has a learnable local feature $\mathbf { f } _ { i } ^ { 1 } ,$ , while all Gaussians in the same group share a global feature $\mathbf { f } _ { g ( i ) } ^ { \mathrm { g } }$ that encodes structural context. Together with the encoded observation condition $\gamma ( \mathbf { o } _ { i } )$ , the input feature of the i-th Gaussian is defined as

$$
\begin{array} { r } { \mathbf { h } _ { i } = \left[ \mathbf { f } _ { i } ^ { \mathrm { l } } , \mathbf { f } _ { g ( i ) } ^ { \mathrm { g } } , \gamma ( \mathbf { o } _ { i } ) \right] ^ { \top } , } \end{array}\tag{8}
$$

where $g ( i )$ denotes the group index.

We predict Gaussian parameters with a lightweight Local Gaussian Parameter Decoder (LGPD), which operates on each voxelized Gaussian group, as shown in Fig. 2. Given at most K

Gaussians in a group, LGPD first projects their input features into token embeddings:

$$
\mathbf { e } _ { i } = \operatorname { L i n e a r } ( \mathbf { h } _ { i } ) .\tag{9}
$$

The token sequence is processed by a lightweight learnable neural network with a validity mask $\mathbf { m } _ { 1 : K }$ for padded entries:

$$
{ \bf z } _ { 1 : K } = \mathrm { S e l f - A t t e n t i o n } \left( { \bf e } _ { 1 : K } , { \bf m } _ { 1 : K } \right) ,\tag{10}
$$

which enables local context aggregation among neighboring Gaussians, thereby significantly enhancing the network’s capacity to model non-linear underwater medium effects.

Each output token is then decoded into residual geometry and color updates:

$$
[ \Delta \mathbf { r } _ { i } , \Delta \mathbf { s } _ { i } ] = \mathrm { M L P } _ { \mathrm { g e o } } ( \mathbf { z } _ { i } ) , \qquad \Delta \mathbf { c } _ { i } = \mathrm { M L P } _ { \mathrm { c o l o r } } ( \mathbf { z } _ { i } ) .\tag{11}
$$

Here, $\Delta \mathbf { r } _ { i } , \Delta \mathbf { s } _ { i }$ , and $\Delta \mathbf { c } _ { i }$ are applied as residual corrections to the initialized Gaussian parameters, producing the updated rotation, scale, and color of the i-th Gaussian. This residual formulation improves the stability of streaming optimization.

2) Underwater Medium Modeling: Underwater image formation is mainly governed by light attenuation and backscattering. Inspired by the simplified image formation model in [41], we model the observed color as the combination of attenuated scene radiance and accumulated backscatter:

$$
\mathbf { I } = \mathbf { J } \cdot \exp ( - \beta _ { \mathrm { a } } z ) + \mathbf { B } ( z ) ,\tag{12}
$$

where J denotes the medium-free scene radiance rendered from the Gaussian map, z denotes the depth along the camera ray estimated from the Gaussian-rendered depth, $\beta _ { \mathrm { a } }$ is the RGB attenuation coefficient vector, and $\mathbf { B } ( z )$ represents the backscatter term.

To enhance the representation capability of our model, we assume that the medium parameters are directional-dependent rather than isotropic. Consequently, we predict the attenuation coefficient conditioned directly on the camera orientation: Given the camera rotation $\mathbf { R } _ { t } \in \mathbb { R } ^ { 3 \times 3 }$ of frame t, we flatten it into a pose vector:

$$
\mathbf { p } _ { t } = \mathrm { v e c } ( \mathbf { R } _ { t } ) \in \mathbb { R } ^ { 9 } .\tag{13}
$$

Using the same sinusoidal encoding as in Eq. (7), we obtain the pose embedding $\gamma ( \mathbf { p } _ { t } )$ ), and decode the frame-specific attenuation coefficient as

$$
\beta _ { \mathrm { a } } ^ { t } = \mathrm { M L P } _ { \mathrm { a } } ( \gamma ( \mathbf { p } _ { t } ) ) .\tag{14}
$$

For backscatter, we adopt a physics-inspired RGB formulation:

$$
\mathbf { B } ( z ) = \mathbf { B } _ { \infty } \odot ( \mathbf { 1 } - \exp ( - \mathbf { b } z ) ) + \mathbf { B } _ { \mathrm { r e s } } \odot \exp ( - \mathbf { d } z ) ,\tag{15}
$$

where $\mathbf { B } _ { \infty } \in \mathbb { R } ^ { 3 }$ denotes the asymptotic backscatter intensity vector at infinite distance, b $\in \mathbb { R } ^ { 3 }$ controls the channel-wise backscatter accumulation rate, $\mathbf { B } _ { \mathrm { r e s } } \in \mathbb { R } ^ { 3 }$ models the neardistance residual backscatter vector, and d $\in \mathbb { R } ^ { 3 }$ represents the channel-wise residual decay parameter vector. The parameters $\{ \mathbf { B } _ { \infty } , \mathbf { b } , \mathbf { B } _ { \mathrm { r e s } } , \mathbf { d } \}$ are learnable channel-wise vectors.

3) Medium-guided Incremental Gaussian Initialization: For keyframes and mapping frames selected by the frontend, as described in Section III-B, new Gaussians are inserted to incorporate newly observed scene regions. Instead of initializing them directly from degraded underwater images, we use the online learning medium model to obtain medium-reduced observations.

Given an underwater image $\mathbf { I } _ { t } ,$ the camera pose estimated by the frontend, and the medium parameters accumulated from previous mapping steps, we estimate the attenuation coefficient $\beta _ { \mathrm { a } } ^ { t }$ and the backscatter term $\mathbf { B } ( z )$ . The medium-reduced image is recovered as

$$
\hat { \mathbf { J } } _ { t } = \left( \mathbf { I } _ { t } - \mathbf { B } ( z ) \right) \odot \exp \left( \beta _ { \mathrm { a } } ^ { t } z \right) ,\tag{16}
$$

where z denotes the depth along the camera ray rendered from the current Gaussian map, and ⊙ denotes element-wise multiplication. Compared with the raw underwater observation, $\hat { \mathbf { J } } _ { t }$ provides cleaner appearance cues for Gaussian initialization.

To determine where new Gaussians should be inserted, we compare the medium-reduced observation $\hat { \mathbf { J } } _ { t }$ with the medium-free rendering $\tilde { \mathbf { J } } _ { t }$ from the current Gaussian map. We compute a structure-aware insertion probability using a Laplacian-of-Gaussian operator:

$$
\begin{array} { r l } & { P _ { \mathrm { a } } ( u , v ) = \operatorname* { m a x } \Big ( \operatorname* { m i n } \big ( \left\| \nabla ^ { 2 } G _ { \sigma } * \hat { \mathbf { J } } _ { t } ( u , v ) \right\| _ { 2 } , 1 \big ) } \\ & { \qquad - \operatorname* { m i n } \big ( \left\| \nabla ^ { 2 } G _ { \sigma } * \tilde { \mathbf { J } } _ { t } ( u , v ) \right\| _ { 2 } , 1 \big ) , 0 \Big ) , } \end{array}\tag{17}
$$

where $G _ { \sigma }$ is a Gaussian kernel with standard deviation σ, ∇<sup>2</sup> is the Laplacian operator, and ∗ denotes convolution. Pixels with $P _ { \mathrm { a } } ( u , v ) > \tau _ { \mathrm { a } }$ are selected for Gaussian insertion. For each newly inserted Gaussian, the base color is initialized from the medium-reduced image $\hat { \mathbf { J } } _ { t }$ rather than the raw underwater image $\mathbf { I } _ { t } .$ This base color is kept fixed, while the neural-enhanced Gaussian representation predicts residual color updates during subsequent optimization.

## D. Loss Function

After initialization, we jointly optimize the Gaussian attributes, local and global Gaussian features, LGPD parameters, and underwater medium parameters through differentiable rendering.

Specifically, to compensate for boundary lens distortions and prioritize central view fidelity, a radial decay kernel $\mathbf { K _ { \mathrm { r a d } } } { \bf \bar { \Psi } } \in \mathbb { R } ^ { H \times W }$ weights the spatial loss terms. For common frames, we construct an adaptive binary validity mask $\mathbf { M } \in \{ 0 , 1 \} ^ { H \times W }$ based on the weighted residual to filter out outliers:

$$
\mathbf { M } ( u , v ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f } \ \exists c \in \{ R , G , B \} , } \\ & { \mathbf { K } _ { \mathrm { r a d } } ( u , v ) \cdot \left| \hat { \mathbf { I } } _ { c } ( u , v ) - \mathbf { I } _ { c } ( u , v ) \right| > \tau _ { e } , } \\ { 1 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{18}
$$

where <sup>ˆ</sup>I and I are the rendered and ground-truth images. For keyframes and mapping frames, M is set to an all-ones matrix.

Using M, we compute the masked rendered image $\hat { \mathbf { I } } _ { \mathbf { M } } =$ M ⊙ <sup>ˆ</sup>I, ground-truth image $\mathbf { I _ { M } } = \mathbf { M } \odot \mathbf { I } ,$ rendered inverse depth $\hat { \mathbf { D } } _ { \mathbf { M } } = \mathbf { M } \odot \hat { \mathbf { D } }$ , and masked monocular depth prior ${ \bf D _ { M } } = { \bf M } \odot { \bf D }$ , where the raw depth prior D is provided by our underwater 3D vision foundation model. The total optimization objective $\mathcal { L } _ { \mathrm { t o t a l } }$ combines the color, depth, and regularization terms:

![](images/d90cca0936cc6798cd4986de11fc73d746aa70c058f8d350bda9595a9471f3e5.jpg)  
Fig. 3. Overview and taxonomy of our proposed underwater evaluation benchmark. The dataset spans broad physical spatial scales and rich topological diversity, encompassing characteristic benthic environments such as shallow sandy beds, canyons, coral reefs, rocky terrains, submerged caves, and artificial ruins. Furthermore, it incorporates diverse camera trajectories (e.g., straight paths, loops, and erratic 3D motions) across a wide range of frame lengths: 14 short sequences (0–50 frames), 13 medium-short sequences (50–100 frames), 20 medium sequences (100–200 frames), 3 long sequences (200–400 frames), and 12 extended-range sequences (> 400 frames) to comprehensively assess online tracking and long-term mapping stability.

TABLE I  
COMPOSITION OF UNDERWATER TRAINING IMAGE PAIRS. THE DATASETS COVER DIVERSE SYNTHETIC AND REAL UNDERWATER SCENES.
<table><tr><td>Dataset</td><td>Pairs</td><td>Type</td><td>Scene Content</td></tr><tr><td>TartanAir-Ocean</td><td>76,387</td><td>Syn.</td><td>Seafloor, terrain, industrial, wrecks</td></tr><tr><td>MIMIR-UW</td><td>44,721</td><td>Syn.</td><td>Seafloor, algae, industrial, ruins</td></tr><tr><td>UWStereo</td><td>26,611</td><td>Syn.</td><td>Corals, vessels, structures, ROVs</td></tr><tr><td>FLSea-stereo</td><td>6,576</td><td>Real</td><td>Seafloor, canyons, corals, artifacts</td></tr><tr><td>sweet-corals</td><td>69,978</td><td>Real</td><td>Corals, complex biomes &amp; structures</td></tr><tr><td>Total</td><td></td><td></td><td>224,273 Mixed Varied benthic biomes and subsea structures</td></tr></table>

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = ( 1 - \lambda _ { \mathrm { s s i m } } ) \mathcal { L } _ { 1 } + \lambda _ { \mathrm { s s i m } } \mathcal { L } _ { \mathrm { s s i m } } } \\ { + \lambda _ { d } \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { s } \mathcal { L } _ { \mathrm { s c a l e } } , \quad } \end{array}\tag{19}
$$

where the individual loss terms are defined as:

$$
\mathcal { L } _ { 1 } = \frac { \sum _ { u , v } \mathbf { K } _ { \mathrm { r a d } } ( u , v ) \cdot \left| \hat { \mathbf { I } } _ { \mathbf { M } } ( u , v ) - \mathbf { I } _ { \mathbf { M } } ( u , v ) \right| } { H W } ,\tag{20}
$$

$$
\mathcal { L } _ { \mathrm { d e p t h } } = \frac { \sum _ { u , v } \mathbf { K } _ { \mathrm { r a d } } ( u , v ) \cdot \left| \hat { \mathbf { D } } _ { \mathbf { M } } ( u , v ) - \mathbf { D } _ { \mathbf { M } } ( u , v ) \right| } { H W }\tag{21}
$$

$$
\mathcal { L } _ { \mathrm { s s i m } } = 1 - \mathrm { S S I M } ( \hat { \mathbf { I } } _ { \mathbf { M } } , \mathbf { I } _ { \mathbf { M } } ) ,\tag{22}
$$

$$
\mathcal { L } _ { \mathrm { s c a l e } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \prod _ { j = 1 } ^ { 3 } s _ { i , j } .\tag{23}
$$

Here, $\lambda _ { \mathrm { s s i m } } , ~ \lambda _ { d } ,$ , and $\lambda _ { s }$ denote the balancing weights for structural similarity, depth consistency, and Gaussian scale regularization across N Gaussians, respectively.

## IV. EXPERIMENT

## A. Experiment Setup

1) Test Datasets & Evaluation Metrics: To evaluate streaming underwater reconstruction under open-world conditions, we assemble a comprehensive 62-sequence test suite, combining 25 sequences from curated public benchmarks with 37 in-the-wild internet sequences. As illustrated in Fig. 3, our test benchmark exhibits high diversity across visual scenes, physical scales, and trajectory lengths.

Specifically, the public subset spans both small- and largescale underwater scenarios: the small-scale domain includes 4 sequences from SeaThru-NeRF [22] and 4 sequences from S-UW [29], while the large-scale domain comprises 4 sequences from Canyons and 8 sequences from RedSea (both sourced from FLSea [39]), alongside 5 sequences from UW-Stereo-VI [42]. Collectively, these public sequences cover a broad spectrum of underwater scene types, including seafloor terrain, coral reefs, caves, submerged vehicles, and man-made structures. To further push system limits under unconstrained real-world conditions, the internet subset incorporates 37 sequences that extend this diversity across unconstrained scene categories, spatial scales, and trajectory lengths. For these uncalibrated internet sequences, camera intrinsics and pseudo ground-truth camera poses are reconstructed via structurefrom-motion (SfM) using COLMAP [38], accompanied by meticulous manual inspection of point-cloud geometric consistency to guarantee evaluation fidelity.

Reconstruction quality is quantitatively assessed using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) [43], and Learned Perceptual Image Patch Similarity (LPIPS) [44]. Camera tracking performance is evaluated via the Root Mean Square Error (RMSE) of Absolute Trajectory Error (ATE) after rigid trajectory alignment. In addition, we measure reconstruction time and per-frame processing latency to rigorously compare the computational overhead against both streaming and off-line baselines.

TABLE II  
QUANTITATIVE RECONSTRUCTION RESULTS ON REAL-WORLD UNDERWATER DATASETS. WE REPORT PSNR, SSIM, AND LPIPS. HIGHER PSNR/SSIM AND LOWER LPIPS INDICATE BETTER PERFORMANCE. THE BEST AND SECOND-BEST RESULTS ARE IN BOLD AND UNDERLINE.
<table><tr><td rowspan="2">Method</td><td colspan="3">Canyons</td><td colspan="3">RedSea</td><td colspan="3">UW-Stereo-VI</td><td colspan="3">SeaThru-NeRF</td><td colspan="3">S-UW</td><td colspan="3">Internet Data</td></tr><tr><td>PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UW-GS</td><td>27.33</td><td>0.916</td><td>0.288</td><td>23.78</td><td>0.809</td><td>0.316</td><td>27.87</td><td>0.903</td><td>0.366</td><td>25.79 0.858</td><td></td><td>0.260</td><td>20.55</td><td>0.733</td><td>0.279</td><td>23.56 0.786</td><td>0.311</td></tr><tr><td>OnTheFly-NVS</td><td>27.20</td><td>0.894</td><td>0.317</td><td>22.37</td><td>0.699</td><td>0.387</td><td>29.32</td><td>0.888</td><td>0.385 22.12</td><td>0.719</td><td>0.324</td><td>19.51</td><td>0.623</td><td>0.335</td><td>23.42</td><td>0.746</td><td>0.339</td></tr><tr><td>HI-SLAM2</td><td>29.88</td><td>0.898</td><td>0.269</td><td>23.66</td><td>0.716</td><td>0.448</td><td>27.69 0.856</td><td>0.423</td><td>25.33</td><td>0.745</td><td>0.471</td><td>21.42</td><td>0.685</td><td>0.367</td><td>25.13</td><td>0.769</td><td>0.369</td></tr><tr><td>MonoGS</td><td>27.83</td><td>0.875</td><td>0.460</td><td>20.89</td><td>0.644</td><td>0.708</td><td>25.87 0.861</td><td>0.480</td><td>13.82</td><td>0.524</td><td>0.589</td><td>9.70</td><td>0.261</td><td>0.667</td><td>16.47</td><td>0.607</td><td>0.623</td></tr><tr><td>S3PO-GS</td><td>29.62</td><td>0.910</td><td>0.369</td><td>22.47</td><td>0.669</td><td>0.562</td><td>29.96</td><td>0.900 0.450</td><td>19.48</td><td>0.609</td><td>0.493</td><td>19.48</td><td>0.569</td><td>0.459</td><td>22.82</td><td>0.743</td><td>0.491</td></tr><tr><td>ARTDECO</td><td>31.83</td><td>0.933</td><td>0.262</td><td>25.41</td><td>0.808</td><td>0.300</td><td>36.22</td><td>0.913</td><td>0.337 23.75</td><td>0.689</td><td>0.378</td><td>23.05</td><td>0.703</td><td>0.302</td><td>26.96</td><td>0.821</td><td>0.279</td></tr><tr><td>WaterSplat-SLAM</td><td>29.33</td><td>0.885</td><td>0.303</td><td>22.61</td><td>0.726</td><td>0.401</td><td>27.06</td><td>0.896</td><td>0.432 24.43</td><td>0.736</td><td>0.492</td><td>19.59</td><td>0.605</td><td>0.539</td><td>23.22</td><td>0.747</td><td>0.355</td></tr><tr><td>Ours</td><td>34.53</td><td>0.955</td><td>0.199</td><td>26.77</td><td>0.836</td><td>0.272</td><td>36.31</td><td>0.920</td><td>0.303</td><td>24.83 0.739</td><td>0.346</td><td>23.73</td><td>0.741</td><td>0.277</td><td>28.53</td><td>0.852</td><td>0.241</td></tr></table>

![](images/091fe883547b957afe7a65a0a50460b447b1fb8527e7b30aec0a263c1d6935a9.jpg)  
Fig. 4. Qualitative comparisons of AquaFlow against baselines [16]–[18], [33], [34]. The top three rows highlight AquaFlow’s superior modeling capabilit for distant underwater scenes and intricate seafloor details (e.g., ground markings and discarded vehicle textures). The bottom three rows demonstrate our strength in recovering complex subsea structural topologies (e.g., underwater caves, shipwreck ruins and canyons), where domain-adapted 3D geometric priors and appearance representations effectively preserve sharp object boundaries and detailed structures compared to baseline methods.

2) Baselines: We benchmark AquaFlow against state-ofthe-art methods across multiple tracking, rendering, and efficiency settings in challenging underwater environments. For camera pose estimation, we compare localization accuracy against optical-flow-based DROID-SLAM [45], vision foundation model-based SLAMs (MASt3R-SLAM [46], VGGT-SLAM [20], and VGGT-SLAM2 [21]), as well as UFEN-SLAM [25] which is a domain-specific underwater SLAM framework that replaces classical ORB feature extraction with learned underwater-adapted features. For rendering quality, we evaluate against streaming novel view synthesis frameworks, including MonoGS [30], S3PO-GS [18], OnTheFly-NVS [16], and HI-SLAM2 [17]. We also include UW-GS [22], an offline underwater Gaussian Splatting system, whose optimization iterations are carefully adjusted to ensure a fair and reasonable comparison between streaming and offline rendering capabilities. Furthermore, ARTDECO [33] and WaterSplat-SLAM [34] serve as strong unified baselines, where both camera tracking accuracy and novel view reconstruction fidelity are benchmarked concurrently. Finally, computational runtime and streaming efficiency are quantitatively evaluated against UW-GS, ARTDECO, and WaterSplat-SLAM.

3) 3D Vision Foundation Model Training Data: To adapt dense geometry prediction and 3D reconstruction vision foundation models [40] to underwater environments, we construct a dedicated training dataset comprising 224,273 valid image pairs, curated and repurposed from five public underwater datasets (Tab. I). The selected sequences contain typical underwater visual degradations, such as color attenuation haze, scattering, low contrast, and texture loss, while maintaining sufficient viewpoint overlap and temporal continuity for reliable geometric supervision. By strategically combining synthetic data (providing large-scale, consistent geometry) with real-world scenes (introducing natural medium degradation and realistic visual features), this dataset enhances MASt3R’s geometric representation while drastically reducing domain shifts when deployed in real underwater conditions.

TABLE III  
QUANTITATIVE EFFICIENCY AND GAUSSIAN NUMBER COMPARISON. WE REPORT SCENE TRAINING TIME (TIME (S) ↓), TRAINING THROUGHPUT (FPS ↑), AND GAUSSIAN COUNT (#GS (K) ↓). THE BEST AND SECOND-BEST RESULTS ARE IN BOLD AND UNDERLINE, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="2">FLsea_canyons</td><td colspan="2"></td><td colspan="2">FLsea_redsea</td><td colspan="2">underwater-stereo-vi</td><td colspan="2">seathruNerf</td><td colspan="2"></td><td colspan="2">S-UW</td><td colspan="2">Internet data</td></tr><tr><td>Time↓</td><td>FPS↑ #GS(k)↓</td><td></td><td>Time↓ FPS↑</td><td></td><td>#GS(k)↓</td><td>Time↓ FPS↑</td><td>#GS(k)↓</td><td> Time↓ FPS↑ #GS(k)↓</td><td></td><td></td><td></td><td></td><td> Time↓ FPS↑ #GS(k)↓</td><td>Time↓ FPS↑</td><td>#GS(k)↓</td></tr><tr><td>UW-GS</td><td>1896.0</td><td>0.26</td><td>952.0</td><td>4174.0</td><td>0.12</td><td>3120.0</td><td>557.6 0.31</td><td>750.9</td><td>227.5</td><td>0.10</td><td>707.8</td><td>245.2</td><td>0.10</td><td>1032.0</td><td>557.0 0.06</td><td>1132.9</td></tr><tr><td>ARTDECO</td><td>498.4</td><td>1.00</td><td>464.2</td><td>593.3</td><td>0.84</td><td>1006.4</td><td>212.8 0.74</td><td>733.9</td><td>50.6</td><td>0.43</td><td>917.4</td><td>39.8</td><td>0.60</td><td>1214.2</td><td>115.0 0.92</td><td>705.2</td></tr><tr><td>WaterSplat-SLAM 1562.9</td><td></td><td>0.32</td><td>13.8</td><td>1029.0</td><td>0.48</td><td>26.9</td><td>757.8 0.21</td><td>7.4</td><td>75.9</td><td>0.29</td><td>33.8</td><td>88.5</td><td>0.27</td><td>37.1</td><td>147.6 0.71</td><td>750.0</td></tr><tr><td>Ours</td><td>467.1</td><td>1.08</td><td>610.5</td><td>709.4</td><td>0.69</td><td>975.7</td><td>207.2 0.76</td><td>581.9</td><td>38.5</td><td>0.56</td><td>790.2</td><td>47.3</td><td>0.51</td><td>1157.8</td><td>153.9 0.68</td><td>758.6</td></tr></table>

TABLE IV

QUANTITATIVE LOCALIZATION (TRAJECTORY) RESULTS ON UNDERWATER DATASETS. WE REPORT ATE RMSE (M) ↓. THE BEST AND SECOND-BEST RESULTS ARE IN BOLD AND UNDERLINE, RESPECTIVELY.
<table><tr><td>Method</td><td>Canyons RedSea UW-Stereo-VI SeaThru-NeRF S-UW Internet</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UFEN-SLAM</td><td>1.107</td><td>1.226</td><td>0.978</td><td>1.637</td><td>1.570</td><td>1.477</td></tr><tr><td>DROID-SLAM</td><td>0.935</td><td>0.737</td><td>2.969</td><td>0.376</td><td>0.069</td><td>1.057</td></tr><tr><td>MASt3r-SLAM</td><td>0.978</td><td>1.357</td><td>0.872</td><td>1.579</td><td>1.270</td><td>0.997</td></tr><tr><td>VGGT-SLAM</td><td>3.079</td><td>2.747</td><td>1.965</td><td>0.237</td><td>0.469</td><td>2.261</td></tr><tr><td>VGGT-SLAM2</td><td>0.732</td><td>1.206</td><td>0.520</td><td>0.240</td><td>0.470</td><td>1.339</td></tr><tr><td>ARTDECO</td><td>0.559</td><td>0.991</td><td>0.325</td><td>0.372</td><td>0.245</td><td>0.593</td></tr><tr><td>WaterSplat-SLAM</td><td>0.432</td><td>0.882</td><td>0.350</td><td>0.335</td><td>0.139</td><td>0.636</td></tr><tr><td>Ours</td><td>0.422</td><td>0.958</td><td>0.157</td><td>0.182</td><td>0.211</td><td>0.478</td></tr></table>

4) Implementation Details.: We fine-tune MASt3R on 16 NVIDIA RTX 4090 GPUs for 30 epochs with a per-GPU batch size of 2 and gradient accumulation over 4 iterations. Optimization is performed using AdamW with $\beta _ { \mathrm { A d a m W } } =$ (0.9, 0.95) and a loss weight of $\beta = 0 . 0 7 5$ . We apply a 4- epoch linear warmup followed by a cosine learning rate decay schedule from $1 \times 1 0 ^ { - 6 }$ down to $1 \times 1 0 ^ { - 7 }$

All experiments are conducted on an Intel i9-14900K CPU and a single NVIDIA RTX 4090 GPU. Following standard protocols, every 8th frame is held out for evaluation, with its pose tracked for trajectory assessment. For implementation details, we set the voxel size to 0.2 m, the token capacity per voxel group to $N _ { \mathrm { t o k e n } } = 3 2$ , and the positional encoding bandwidth to $L = 4$ . Keyframe selection, mapping parallax, and Gaussian insertion thresholds are configured as τ<sub>keyframe</sub> = 0.7, τ<sub>mapframe</sub> = 30 pixels, and $\tau _ { \mathrm { a } } ~ = ~ 1 . 0$ , respectively. For optimization, the loss weight for $\mathcal { L } _ { \mathrm { l 1 } }$ is set to $0 . 8 . \lambda _ { \mathrm { s s i m } } = 0 . 2 $ $\lambda _ { d } ~ = ~ 0 . 0 1$ for depth loss, and $\lambda _ { s } ~ = ~ 0 . 0 1$ for scaling regularization.

## B. Comparasion

1) Reconstruction Results Analysis: Tab. II reports the quantitative reconstruction results across six real-world underwater benchmarks. Among these, AquaFlow achieves stateof-the-art reconstruction performance on five datasets, outperforming existing off-line reconstruction and 3DGS-based SLAM methods by consistently improving PSNR and SSIM while reducing LPIPS. Although AquaFlow yields slightly lower metrics than UW-GS on SeaThru-NeRF, UW-GS is an off-line method that relies on substantially longer optimization time (Tab. III). Noticeably, SeaThru-NeRF is a small-scale benchmark where each scene contains only around 20 images focused on localized coral environments. This performance discrepancy further indicates that current off-line underwater methods tend to overfit to such small, single public benchmark datasets, thereby confirming the necessity and effectiveness of our extensive generalization evaluation across larger and more diverse real-world environments.

Fig. 4 illustrates the qualitative reconstruction comparisons. As shown in the top three rows, owing to our explicit physical modeling of underwater light transport phenomena and the structured distance-conditioned Gaussian representation, AquaFlow achieves superior modeling of long-range objects as well as fine seafloor details and textures. The bottom three rows present reconstruction results in scenes with complex structural topologies (e.g., underwater caves and shipwreck ruins). Benefiting from our accurate appearance representation, the incorporated underwater geometric priors effectively constrain and enhance the structural geometry, enabling AquaFlow to faithfully preserve sharp boundaries and intricate structures where baseline approaches fail.

2) Tracking Results Analysis: Tab. IV presents quantitative trajectory localization performance. AquaFlow achieves the lowest overall mean error of 0.401 m, consistently outperforming optical flow-based SLAM (DROID-SLAM), 3D vision foundation models (e.g., MASt3r-SLAM, VGGT-SLAM2), and underwater descriptor matching methods (UFEN-SLAM). While conventional baselines struggle with severe underwater visual degradation, our method leverages 3D underwater geometric priors to ensure robust pose estimation. Notably, AquaFlow achieves top tracking performance on four out of six real-world datasets, highlighting the strong generalization capability of our underwater 3D geometric prior across diverse underwater environments.

3) Efficiency Analysis: Tab. III evaluates the computational efficiency and memory footprint across six underwater benchmarks. We compare AquaFlow against the offline optimization baseline UW-GS as well as state-of-the-art streaming methods (ARTDECO and WaterSplat-SLAM). While delivering superior overall reconstruction quality compared to UW-GS, AquaFlow operates significantly faster, achieving substantial speedups in per-frame optimization time. Notably, this speed gap is even wider in practice, as our report excludes the heavy preprocessing runtime required by UW-GS for COLMAP pose estimation and Depth Anything V2 [47] depth extraction. Compared to streaming baselines, AquaFlow maintains comparable or superior runtime efficiency while achieving significantly higher reconstruction accuracy. These efficiency gains stem from our lightweight LGPD network architecture and a medium-aware Gaussian initialization strategy, which precisely adds Gaussians where needed and properly initializes their attributes to avoid redundant computation.

TABLE V  
ABLATION STUDY ON THE FLSEA-CANYONS DATASET. WE REPORT PSNR, SSIM, AND LPIPS. HIGHER PSNR/SSIM AND LOWER LPIPS INDICATE BETTER RECONSTRUCTION QUALITY.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>GS(k)</td></tr><tr><td>Full model</td><td>34.47</td><td>0.953</td><td>0.209</td><td>601.6</td></tr><tr><td>w/o UW finetuning</td><td>33.41</td><td>0.943</td><td>0.247</td><td>608.3</td></tr><tr><td>w/o medium init.</td><td>33.91</td><td>0.949</td><td>0.223</td><td>406.8</td></tr><tr><td>w/o dist. embedding</td><td>32.56</td><td>0.943</td><td>0.232</td><td>629.6</td></tr><tr><td>w/o LGPD</td><td>33.92</td><td>0.951</td><td>0.214</td><td>653.2</td></tr></table>

![](images/a67a50784fb517b6510719d0b532131d49c8fd1ebb87f46f3dc35c9c7796578d.jpg)  
Fig. 5. Qualitative ablation comparison on rendered color and depth maps. Compared to baseline variants lacking specific modules (e.g., medium initialization, underwater fine-tuning, distance encoding, or LGPD), disabling any component introduces structural geometric errors or scene blurring. In contrast, our full model faithfully recovers both accurate 3D geometry and realistic underwater appearance.

4) Ablation Study: To evaluate the contribution of each key design, we conduct ablation studies on the FLSea-Canyons dataset (Tab. V and Fig. 5). Quantitatively, our full model achieves optimal performance across all metrics, whereas removing any core component leads to consistent performance drops. Specifically, we construct variant baselines by removing medium-aware initialization (w/o medium init.), observation distance embeddings (w/o dist. embedding), domain-adapted weights (w/o UW finetuning), and replacing the LGPD module with an equivalent MLP (w/o LGPD).

Qualitatively, Fig. 5 and Tab. V visually confirm the individual contribution of each component in AquaFlow. Specif ically, disabling medium-aware initialization (w/o medium init.) yields sparse, under-dense initial seeds due to severe underwater attenuation and low contrast, ultimately leading to an insufficient modeling capability of Gaussian primitives and noticeable scene structural loss. Removing domain adaptation (w/o UW finetuning) severely impairs the learned 3D geometric priors, resulting in corrupted depth maps with noisy boundaries and pronounced geometry collapse, which further directly degrades the final rendering quality. Meanwhile, omitting observation distance embeddings (w/o dist. embedding) prevents the model from perceiving distance-dependent light scattering, causing apparent color shifts and contrast degradation in synthesized views. Finally, replacing the LGPD module with a standard MLP (w/o LGPD) weakens the physical decoupling between local water turbidity and global illumination, leading to fine-detail blurring and loss of high-frequency textures. In contrast, our full model seamlessly integrates all modules to achieve the most visually pleasing and geometrically accurate reconstruction.

## V. CONCLUSION

In this work, we present AquaFlow, a monocular 3D Gaussian Splatting streaming reconstruction framework tailored for complex underwater environments. By adapting a 3D vision foundation model to large-scale underwater data, AquaFlow achieves robust camera pose tracking and geometric prior estimation. For streaming mapping, we incorporate a medium-guided incremental Gaussian initialization strategy along with a streaming-compatible hybrid scene representation, effectively modeling range-dependent attenuation, scattering, and complex physical image formation phenomena. Extensive evaluations across 62 real-world underwater trajectories demonstrate that AquaFlow achieves state-of-the-art localization and reconstruction performance, outperforming strong offline and streaming baselines while maintaining high time efficiency.

Building upon these capabilities, our future work will proceed in two directions. First, we plan to further enhance the zero-shot generalization of underwater visual foundation models by scaling up training datasets and further incorporating physics-inspired underwater priors into the training process. Second, we aim to extend our reconstruction framework to accommodate dynamic underwater environments, such as moving marine organisms, drifting floating particles, and flowinduced structural deformations (e.g., swaying vegetation or flexible cables), thereby broadening its applicability to a wider range of unconstrained underwater video inputs.

## REFERENCES

[1] Y. Song, D. Nakath, M. She, and K. Koser, “Optical imaging and image¨ restoration techniques for deep ocean mapping: a comprehensive survey,” Journal of Photogrammetry, Remote Sensing and Geoinformation Science (PFG), vol. 90, pp. 243–267, 2022.

[2] A.-C. Wolfl, H. Snaith, S. Amirebrahimi, C. W. Devey, B. Dorschel,¨ V. Ferrini, V. A. Huvenne, M. Jakobsson, J. Jencks, G. Johnston et al., “Seafloor mapping–the challenge of a truly global ocean bathymetry,” Frontiers in Marine Science, vol. 6, p. 434383, 2019.

[3] W. Wang, B. Joshi, N. Burgdorfer, K. Batsosc, A. Q. Lid, P. Mordohaia, and I. Rekleitisb, “Real-time dense 3d mapping of underwater environments,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 5184–5191.

[4] A. Bodenmann, B. Thornton, R. Nakajima, and T. Ura, “Methods for quantitative studies of seafloor hydrothermal systems using 3d visual reconstructions,” Robomech Journal, vol. 4, no. 1, p. 22, 2017.

[5] J. D. Hernandez, K. Isteni ´ c, N. Gracias, N. Palomeras, R. Campos,ˇ E. Vidal, R. Garcia, and M. Carreras, “Autonomous underwater navigation and optical mapping in unknown natural environments,” Sensors, vol. 16, no. 8, p. 1174, 2016.

[6] K. Hu, T. Wang, C. Shen, C. Weng, F. Zhou, M. Xia, and L. Weng, “Overview of underwater 3d reconstruction technology based on optical images,” Journal of Marine Science and Engineering, vol. 11, no. 5, p. 949, 2023.

[7] J. Qin, M. Li, D. Li, J. Zhong, and K. Yang, “A survey on visual navigation and positioning for autonomous uuvs,” Remote Sensing, vol. 14, no. 15, p. 3794, 2022.

[8] M. Ferrera, J. Moras, P. Trouve-Peloux, and V. Creuze, “Real-time´ monocular visual odometry for turbid and dynamic underwater environments,” Sensors, vol. 19, no. 3, p. 687, 2019.

[9] B. Chemisky, E. Nocerino, F. Menna, M. Nawaf, and P. Drap, “A portable opto-acoustic survey solution for mapping of underwater targets,” The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. 43, pp. 651–658, 2021.

[10] B. Joshi, M. Xanthidis, S. Rahman, and I. Rekleitis, “High definition, inexpensive, underwater mapping,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 1113–1121.

[11] B. Kerbl, G. Kopanas, T. Leimkuhler, G. Drettakis ¨ et al., “3d gaussian splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[12] H. Li, W. Song, T. Xu, A. Elsig, and J. Kulhanek, “WaterSplatting: Fast underwater 3D scene reconstruction using gaussian splatting,” arXiv preprint arXiv:2408.08206, 2024. [Online]. Available: https: //arxiv.org/abs/2408.08206

[13] D. Yang, J. J. Leonard, and Y. Girdhar, “SeaSplat: Representing underwater scenes with 3D gaussian splatting and a physically grounded image formation model,” in IEEE International Conference on Robotics and Automation, 2025. [Online]. Available: https: //arxiv.org/abs/2409.17345

[14] Z. Jiang, H. Wang, G. Huang, B. Seymour, and N. Anantrasirichai, “RUSplatting: Robust 3D gaussian splatting for sparse-view underwater scene reconstruction,” in British Machine Vision Conference, 2025. [Online]. Available: https://arxiv.org/abs/2505.15737

[15] W. Xing, J. Chen, Z. Yang, C. Lin, J. Dong, C. Chen, X. Zhou, and M. Han, “UW-3DGS: Underwater 3D reconstruction with physics-aware gaussian splatting,” arXiv preprint arXiv:2508.06169, 2025. [Online]. Available: https://arxiv.org/abs/2508.06169

[16] A. Meuleman, I. Shah, A. Lanvin, B. Kerbl, and G. Drettakis, “Onthe-fly reconstruction for large-scale novel view synthesis from unposed images,” ACM Transactions on Graphics (TOG), vol. 44, no. 4, pp. 1–14, 2025.

[17] W. Zhang, Q. Cheng, D. Skuddis, N. Zeller, D. Cremers, and N. Haala, “Hi-slam2: Geometry-aware gaussian slam for fast monocular scene reconstruction,” IEEE Transactions on Robotics, vol. 41, pp. 6478–6493, 2025.

[18] C. Cheng, S. Yu, Z. Wang, Y. Zhou, and H. Wang, “Outdoor monocular slam with global scale-consistent 3d gaussian pointmaps,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 26 035–26 044.

[19] R. Murai, E. Dexheimer, and A. J. Davison, “MASt3R-SLAM: Realtime dense slam with 3d reconstruction priors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[20] D. Maggio, H. Lim, and L. Carlone, “Vggt-slam: Dense rgb slam optimized on the sl (4) manifold,” Advances in Neural Information Processing Systems, vol. 38, pp. 129 839–129 867, 2026.

[21] D. Maggio and L. Carlone, “Vggt-slam 2.0: Real time dense feedforward scene reconstruction,” arXiv preprint arXiv:2601.19887, 2026.

[22] D. Levy, A. Peleg, N. Pearl, D. Rosenbaum, D. Akkaynak, S. Korman, and T. Treibitz, “SeaThru-NeRF: Neural radiance fields in scattering media,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. [Online]. Available: https://arxiv.org/abs/2304.07743

[23] G. Chen, G. Du, C. Yang, Y. Xu, C. Wu, H. Hu, F. Dong, and J. Zeng, “An underwater visual slam system with adaptive image enhancement,” Ocean Engineering, vol. 326, p. 120896, 2025.

[24] Z. Zheng, Z. Xin, Z. Yu, and S.-K. Yeung, “Real-time gan-based image enhancement for robust underwater monocular slam,” Frontiers in Marine Science, vol. 10, p. 1161399, 2023.

[25] J. Yang, M. Gong, G. Nair, J. H. Lee, J. Monty, and Y. Pu, “Knowledge distillation for feature extraction in underwater vslam,” arXiv preprint arXiv:2303.17981, 2023.

[26] Z. Wang, Q. Zhang, Y. Hu, and B. Zheng, “Nerf-enhanced visual–inertial slam for low-light underwater sensing,” Journal of Marine Science and Engineering, vol. 14, no. 1, p. 46, 2025.

[27] X. Wang, X. Fan, Y. Liu, Y. Xin, and P. Shi, “Eum-slam: An enhancing underwater monocular visual slam with deep learning-based optical flow estimation,” IEEE Transactions on Instrumentation and Measurement, 2025.

[28] Y. Wu, Y. Li, W. Luo, and X. Ding, “Raem-slam: A robust adaptive endto-end monocular slam framework for auvs in underwater environments,” Drones, vol. 9, no. 8, p. 579, 2025.

[29] H. Wang, N. Anantrasirichai, F. Zhang, and D. Bull, “Uw-gs: Distractoraware 3d gaussian splatting for enhanced underwater scene reconstruction,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 3280–3289.

[30] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian splatting slam,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 18 039–18 048.

[31] T. Deng, Y. Chen, L. Zhang, J. Yang, S. Yuan, J. Liu, D. Wang, H. Wang, and W. Chen, “Compact 3d gaussian splatting for dense visual slam,” arXiv preprint arXiv:2403.11247, 2024.

[32] E. Sandstrom, G. Zhang, K. Tateno, M. Oechsle, M. Niemeyer, Y. Zhang,¨ M. Patel, L. Van Gool, M. Oswald, and F. Tombari, “Splat-slam: Globally optimized rgb-only slam with 3d gaussians,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1680–1691.

[33] G. Li, K. Ren, L. Xu, Z. Zheng, C. Jiang, X. Gao, B. Dai, J. Pu, M. Yu, and J. Pang, “Artdeco: Toward high-fidelity on-the-fly reconstruction with hierarchical gaussian structure and feed-forward guidance,” in The Fourteenth International Conference on Learning Representations, 2026.

[34] K. Wang, S. Zou, C. Jiang, Y. Dai, S. Chen, S. Shen, and G. Wang, “Watersplat-slam: Photorealistic monocular slam in underwater environment,” IEEE Robotics and Automation Letters, 2026.

[35] W. Wang, D. Zhu, X. Wang, Y. Hu, Y. Qiu, C. Wang, Y. Hu, A. Kapoor, and S. Scherer, “Tartanair: A dataset to push the limits of visual slam,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020, pp. 4909–4916.

[36] O. Alvarez-Tu <sup>´</sup> n˜on, H. Kanner, L. R. Marnet, H. X. Pham, J. le Fevre Se-´ jersen, Y. Brodskiy, and E. Kayacan, “Mimir-uw: A multipurpose synthetic dataset for underwater navigation and inspection,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 6141–6148.

[37] S. Nozdrenkov, Mujiyanto, R. R. Zedta, Fakhrurrozi, T. Barker, M. D. Rahman, A. S. Samusamu, F. Rochman, L. Peters, D. R. Rachmawati, D. O. Johan, D. J. Craggs, and P. M. Sweet, “sweet-corals (revision bd316dc),” 2025. [Online]. Available: https: //huggingface.co/datasets/wildflow/sweet-corals

[38] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[39] Y. Randall and T. Treibitz, “FLSea: Underwater visualinertial and stereo-vision forward-looking datasets,” arXiv preprint arXiv:2302.12772, 2023. [Online]. Available: https://arxiv.org/abs/2302. 12772

[40] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3d with mast3r,” in European conference on computer vision. Springer, 2024, pp. 71–91.

[41] D. Akkaynak and T. Treibitz, “A revised underwater image formation model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 6723–6732.

[42] B. Joshi, S. Rahman, M. Kalaitzakis, B. Cain, J. Johnson, M. Xanthidis, N. Karapetyan, A. Hernandez, A. Q. Li, N. Vitzilaios et al., “Experimental comparison of open source visual-inertial-based state estimation algorithms in the underwater domain,” in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 7227–7233.

[43] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[44] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595.

[45] Z. Teed and J. Deng, “DROID-SLAM: Deep visual slam for monocular, stereo, and rgb-d cameras,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 16 558–16 569.

[46] R. Murai, E. Dexheimer, and A. J. Davison, “MASt3R-SLAM: Realtime dense SLAM with 3D reconstruction priors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. [Online]. Available: https://arxiv.org/abs/2412.12392

[47] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Advances in Neural Information Processing Systems, vol. 37, pp. 21 875–21 911, 2024.