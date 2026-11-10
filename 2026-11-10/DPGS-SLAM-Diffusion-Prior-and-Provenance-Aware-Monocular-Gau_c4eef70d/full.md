# DPGS-SLAM: Diffusion-Prior and Provenance-Aware Monocular Gaussian Splatting SLAM for Dynamic Visual Sensing

Yifan Luo, Takafumi Matsumaru, Senior Member, IEEE, Vibekananda Dutta, Member, IEEE, and Xin He, Member, IEEE

Abstract— This paper presents DPGS-SLAM (Diffusion-Prior and Provenance-Aware Monocular Gaussian Splatting SLAM) for dense mapping in environments containing dynamic objects. Dynamic observations can contaminate camera tracking and scene reconstruction in visual SLAM (simultaneous localization and mapping). Although 3DGS (3D Gaussian splatting) enables efficient dense mapping, existing 3DGS-SLAM systems may exhibit instability when dynamic objects violate their underlying static-scene assumption. Excluding dynamic observations can reduce their influence on tracking and mapping, but may leave occluded static backgrounds missing or weakly constrained. Our proposed

![](images/76f92492fd9a9f0fe6abc29b6f16f61fbd6a40d89fd3b9229ac1ca2b5310bd3b.jpg)

DPGS-SLAM is a monocular RGB SLAM framework that supports provisional representations of insufficiently observed static backgrounds through asynchronous diffusion-based background completion and provenance-aware Gaussian mapping. Dynamic regions are identified through optical-flow-based motion discovery and temporally consistent mask propagation by SAM2 (segment anything model 2). For selected keyframes, a fine-tuned diffusion inpainting model generates plausible background appearance in occluded regions, while Metric3D v2 predicts metric depth from the completed images to initialize provisional Gaussian primitives. Provenance labels distinguish generated primitives from observation-supported primitives, excluding generated content from camera-pose estimation and observed-map optimization. When previously occluded backgrounds become visible, eligible generated primitives are removed, and observation-supported primitives are independently initialized from real RGB observations and predicted depth. Experimental results demonstrated competitive camera tracking, with average ATE RMSE (absolute trajectory error, measured as root mean square error) values of 1.2 cm on TUM RGB-D and 2.7 cm on BONN RGB-D. On StaDy4D, DPGS-SLAM improved average PSNR (peak signal-to-noise ratio) by 4.60 dB over WildGS-SLAM using only monocular RGB input.

Index Terms— 3-D Gaussian splatting (3DGS), dynamic scene reconstruction, simultaneous localization and mapping (SLAM).

## I. INTRODUCTION

SLAM), a monocular RGB SLAM (simultaneous localization and mapping) framework for dynamic visual sensing, designed to address insufficient real observation support for static-background reconstruction through asynchronous staticbackground completion via diffusion-based inpainting [1], [2] and provenance-aware Gaussian mapping. Here, inpainting refers to the general technique of synthesizing image content within masked regions, whereas background completion refers to the specific task of generating plausible staticbackground content in regions occluded by dynamic objects. A real observation of a static-background surface is defined as an RGB measurement acquired when that surface is directly visible to the camera. Insufficient real observation support occurs when such measurements are absent or too sparsely distributed across the available views to reliably constrain the reconstruction of the corresponding surface.

Achieving reliable SLAM in dynamic environments remains a long-standing challenge in the computer vision community. Many traditional SLAM systems, such as ORB-SLAM (2015) [3] and DSO (2017) [4], are built upon the assumption of a rigid scene and therefore tend to suffer from tracking drift or reconstruction instability when independently moving objects are present.

Recently, NeRF (neural radiance fields) (2020) [5] -based volumetric representations have been explored as a densemapping paradigm for SLAM, due to their strong photometric modeling capability and high-quality novel-view synthesis. However, NeRF-SLAM methods typically incur high computational costs and limited rendering efficiency, and often rely on additional depth sensors or predefined dynamic priors, such as DN-SLAM (2023) [6] and RoDyn-SLAM (2024) [7], to handle dynamic regions. These constraints restrict their applicability in purely monocular RGB settings and limit their robustness in complex dynamic scenes.

More recently, 3DGS (3D Gaussian splatting) (2023) [8] has emerged as an efficient and expressive scene representation for dense scene modeling. By modeling scenes as a collection of learnable 3D Gaussian primitives and enabling fast, differentiable rasterization, 3DGS has been widely adopted in recent SLAM systems, such as SplaTAM (2024) [9] and MonoGS (2024) [10], achieving high-fidelity mapping in static environments at substantially lower computational costs. Nevertheless, existing 3DGS-SLAM approaches typically assume a static scene, leading to temporally inconsistent updates of Gaussian primitives in dynamic regions, which accumulate over time and contaminate the reconstructed map.

This behavior results in tracking instability, persistent noise in long-term mapping, and ghosting artifacts, thereby limiting the applicability of 3DGS-SLAM in real-world dynamic scenarios.

In dynamic SLAM systems such as MaskFusion [11], DG-SLAM [12], and RoDyn-SLAM [7], detected dynamic regions are excluded or downweighted during tracking and mapping. While this strategy reduces the influence of moving objects from directly contaminating camera-pose estimation and map optimization, it does not supply static-background observations for the masked regions. Consequently, backgrounds that remain insufficiently observed, whether under persistent or intermittent occlusion, may be missing or only sparsely represented in the reconstructed map. When only sparse or inconsistent static observations are available, their reconstructed appearance and geometry may also be weakly constrained, resulting in local blur, ghosting, or unstable Gaussian structures. Therefore, suppression alone removes unreliable dynamic observations but does not provide the missing background information needed for reconstruction.

To address this limitation, our DPGS-SLAM introduces asynchronous diffusion-based background completion together with provenance-aware Gaussian mapping. Here, provenanceaware denotes the explicit labeling of Gaussian primitives as observation-supported or diffusion-generated, with each label determining whether a primitive participates in optimization and how it is updated.

For selected keyframes, a diffusion inpainting model generates plausible background content for regions masked as dynamic, providing an appearance prior where direct observations are unavailable. The completion process runs asynchronously, without blocking camera tracking or observedmap optimization. Metric3D v2 (2024) [13] predicts metric depth for the completed regions, allowing the generated content to be lifted into 3D and represented by provisional Gaussian primitives.

These generated Gaussian primitives remain fixed and are excluded from camera-pose estimation and observed-map optimization. When the corresponding background becomes visible in subsequent keyframes, a depth-aware projective rule removes eligible generated Gaussian primitives, while observation-supported Gaussian primitives are independently initialized from real RGB observations and predicted depth. In this way, DPGS-SLAM provides provisional representations of insufficiently observed static backgrounds while keeping generated Gaussian primitives separate from camera tracking and observation-supported map optimization.

Our main contributions are as follows:

• We design a motion-aware dynamic-region identification pipeline for motion mask generation that integrates periodic optical-flow-based motion discovery with SAM2 (segment anything model 2) (2025) [14] for mask refinement and temporal propagation, supporting dynamicobject discovery and spatiotemporally consistent motion masks.

• We introduce an asynchronous diffusion-based background completion strategy for static-background regions with insufficient observation support. A fine-tuned diffusion inpainting model generates plausible background appearance for selected keyframes, while Metric3D v2 (2024) [13] predicts depth to initialize provisional Gaussian primitives without blocking camera tracking or observed-map optimization.

• We propose a provenance-aware Gaussian mapping strategy that explicitly distinguishes observationsupported primitives from diffusion-generated primitives. Generated primitives are excluded from camera-pose estimation and observed-map optimization. As previously occluded backgrounds become visible, eligible generated primitives are selectively removed using a depth-aware projective rule, while observation-supported primitives are independently initialized from real RGB observations and predicted depth.

• We evaluated DPGS-SLAM on TUM RGB-D [15], BONN RGB-D [16], and StaDy4D [17] using monocular RGB input. Our DPGS-SLAM achieved average ATE (absolute trajectory error) RMSE (root mean squared error) values of 1.2 cm and 2.7 cm on TUM RGB-D and BONN RGB-D, respectively, demonstrating competitive camera tracking. On the three evaluated StaDy4D sequences, it improved average PSNR (peak signal-to-noise ratio) by 4.60 dB over WildGS-SLAM (2025) [18], while quantitative ablation studies and qualitative comparisons further support the effectiveness of background completion and observation-based replacement.

The rest of this paper is organized as follows. Section II reviews diffusion-based 3D reconstruction and scene completion, 3D Gaussian splatting SLAM, and dynamic visual SLAM. Section III presents our proposed method, DPGS-SLAM. Section IV presents the experimental setup, comparative results, ablation studies, and limitations. Section V concludes this paper.

## II. RELATED WORKS

## A. Diffusion-Based 3D Reconstruction and Scene Completion

Diffusion models learn data distributions through iterative denoising [1], while latent diffusion models perform this process in a compressed latent space to enable efficient conditional image generation and inpainting [2]. Their ability to generate plausible content in unobserved regions has motivated their application to 3D reconstruction. GS-GS (2025) [19] uses diffusion-generated views to refine Gaussian representations, while Zhong et al. (2025) [20] introduce scene-grounded video diffusion for sparse-view reconstruction. RI3D (2025) [21] employs separate diffusion priors for repairing observed regions and inpainting missing content.

Diffusion-based 3D scene completion further addresses missing or occluded regions. NeRFiller (2024) [22] uses multi-view diffusion inpainting for NeRF completion, while InFusion (2024) [23] introduces diffusion-based depth completion for initializing missing Gaussian primitives. SplatFill (2026) [24] combines diffusion inpainting with depth-aware supervision, and Lai et al. (2025) [25] employ diffusionbased depth completion for egocentric reconstruction under dynamic occlusions. Although these approaches improve reconstruction completeness, diffusion-generated content may lack direct observation support, requiring careful integration into incremental SLAM to avoid affecting camera tracking and observed-map optimization.

## B. 3D Gaussian Splatting SLAM Systems

3DGS [8] represents scenes as collections of anisotropic Gaussian primitives and enables efficient differentiable rasterization without ray marching. Owing to its rendering efficiency and reconstruction quality, 3DGS has been widely adopted in dense SLAM systems, including SplaTAM (2024) [9], MonoGS (2024) [10], and Photo-SLAM (2024) [26], achieving accurate tracking and high-quality mapping in static environments. However, these systems are primarily developed and evaluated under static-scene assumptions.

To handle dynamic environments, recent works extend 3DGS-SLAM to dynamic scenarios, including DG-SLAM (2024) [12] and GassiDy (2025) [27]. These approaches identify and suppress dynamic observations through motion masks, rendering residuals, or adaptive Gaussian primitive management. WildGS-SLAM (2025) [18] further incorporates learned per-pixel uncertainty into tracking and Gaussian map optimization to reduce the influence of dynamic observations. DRG-SLAM (2026) [28] combines geometric motion verification and temporal consistency to exclude dynamic observations from RGB-D tracking and Gaussian mapping. While these strategies reduce contamination from moving objects, dynamic suppression alone does not provide observation support for persistently occluded static backgrounds. As a result, such regions may remain missing, sparsely represented, or weakly constrained in the reconstructed Gaussian map.

## C. Dynamic Visual SLAM Systems

Many classical visual SLAM systems, such as DSO [4] and ORB-SLAM3 (2021) [29], are developed under the assumption of static scenes. The presence of dynamic objects contradicts this assumption by disrupting feature correspondences and photometric consistency, often resulting in tracking drift and reconstruction failure. To address this limitation, a common strategy is to identify and exclude dynamic regions, reconstructing only the static components of the environment.

Dynamic region detection is commonly based on geometric consistency checks, such as StaticFusion (2018) [30] and ReFusion [16], or optical-flow-based motion estimation, such as Motion removal (2018) [31] and FlowFusion (2020) [32]. However, these approaches can produce noisy and temporally unstable masks under occlusions or inaccurate geometry and motion estimates. Semantic-based methods further exploit object detection or semantic segmentation priors like Mask-SLAM (2018) [33], but semantic category information alone cannot reliably distinguish between moving and stationary instances. Hybrid approaches combining geometric, motion, and semantic cues, such as DynaSLAM (2018) [34] and SG-SLAM (2022) [35], improve dynamic-region identification, although consistent separation of dynamic and static regions remains challenging under occlusions and intermittent observations. More recently, DROID-W (2026) [36] introduces uncertaintyaware bundle adjustment to jointly refine camera poses, depth, and dynamic uncertainty.

In many dynamic SLAM systems [7], [12], [34], dynamic observations are excluded or suppressed during tracking and mapping. While hard exclusion prevents detected dynamic pixels from directly contributing to pose estimation and map optimization, masking alone does not provide static-background information for regions that remain unobserved.

Beyond hard exclusion, Zhou et al. (2025) [37] combine dynamic pixel-set masking with residual losses for soft suppression, while DROID-W (2026) [36] incorporates dynamic uncertainty into bundle adjustment. Dy3DGS-SLAM (2025) [38] addresses artifacts associated with pruning dynamic Gaussian primitives. These strategies reduce dynamic interference but do not supply missing static-background observations; soft suppression may also retain residual dynamic contamination.

DPGS-SLAM therefore combines optical-flow-based motion discovery with temporally propagated segmentation and performs asynchronous diffusion-based background completion to provisionally represent backgrounds with insufficient real observation support. Provenance-aware mapping isolates generated Gaussian primitives from camera-pose estimation and observed-map optimization; when real static observations become available, eligible generated primitives are removed and observation-supported primitives are independently initialized.

![](images/2ea5f595c8c466d1f5e4041ae8c239bf1978161fb453921f03bf4f9d514fd06a.jpg)  
Fig. 1: System overview – Our framework consists of (a) motion mask generation (Section III-A), (b) asynchronous diffusionbased background completion (Section III-B), and (c) 3DGS-based SLAM (Section $\operatorname { I I I - C } ) ,$ , including (c.1) camera tracking, (c.2) keyframing and co-visibility, and (c.3) provenance-aware mapping. Dynamic regions are identified using motion discrepancie revealed by optical flow, with SAM2 [14] providing temporally consistent motion masks, while tracking uses observed static regions and background completion provides provisional representations for mapping.

## III. METHODOLOGY

The overall pipeline is illustrated in Fig. 1, which consists of (a) motion mask generation (Section III-A), (b) asynchronous diffusion-based background completion (Section III-B), and (c) 3DGS-based SLAM (Section III-C).

## A. Motion Mask Generation

The process of motion mask generation, roughly shown in Fig. 1 (a), is further presented in greater detail in Fig. 2. Given RGB frames $\{ \mathbf { I } _ { t } \} _ { t = 1 } ^ { T }$ , where $\mathbf { I } _ { t } ~ \in ~ \mathbb { R } ^ { H \times W \times 3 }$ , we estimate binary masks $\mathbf { \bar { M } } _ { t } \in \{ 0 , 1 \} ^ { H \times W }$ identifying dynamic foreground regions to exclude from static-background reconstruction. Motion discovery is performed every S frames using only RGB input, independently of SLAM pose estimates and keyframe selection.

At each detection frame r satisfying $\Delta + 1 \le r \le T - 1$ a pretrained OCLR [39] model processes $\Delta$ consecutive pairs of optical flow fields:

$$
\begin{array} { r } { \mathbf { F } _ { s } ^ { + } = \mathcal { F } ( \mathbf { I } _ { s } , \mathbf { I } _ { s + 1 } ) , } \\ { \mathbf { F } _ { s } ^ { - } = \mathcal { F } ( \mathbf { I } _ { s } , \mathbf { I } _ { s - 1 } ) , } \end{array}\tag{1}
$$

where $s \in \{ r - \Delta + 1 , \ldots , r \}$ and $\mathcal { F }$ is a pretrained optical flow estimator. Each pair shares the source-frame grid, and the two two-channel flow fields are concatenated into a fourchannel input per frame, matching the input configuration of the pretrained OCLR network. Amodal predictions (of partially hidden or occluded objects) by OCLR are composed using its estimated layer ordering; only the resulting modal masks at frame r are used as object candidates.

Candidates are associated with propagated masks by SAM2 [14] at frame r by greedy one-to-one matching in descending order of intersection over union (IoU), rejecting pairs below an IoU threshold. Matched objects retain their IDs, while unmatched candidates receive new IDs. Candidate bounding boxes and interior positive points prompt SAM2 to refine or initialize the corresponding masks. The refined masks are used for the current frame and subsequent forward propagation. Once registered as dynamic, objects remain masked during temporary stops while segmented as visible. Let ${ \mathcal { O } } _ { t }$ denote the object IDs registered up to frame t and $\mathbf { M } _ { t } ^ { ( i ) }$ its estimated visible mask, set to zero when no visible region is predicted. The combined mask is

![](images/2fcd4f3047809b132fbb13015862b01e5feed82693e733d8a889fae486c54236.jpg)  
Fig. 2: Motion mask generation (Section III-A): OCLR [39] periodically identifies motion candidates from local opticalflow windows. Association with SAM2 [14] predictions generates ID-aware box and point prompts for refining existing masks and initializing new objects. SAM2 propagates registered objects forward to produce temporally consistent dynamic-region masks.

$$
\mathbf { M } _ { t } = \bigvee _ { i \in \mathcal { O } _ { t } } \mathbf { M } _ { t } ^ { ( i ) } ,\tag{2}
$$

where $\vee$ denotes pixel-wise logical OR; an empty union yields an all-zero mask.

A complete flow window requires $\Delta + 2$ RGB frames; detection at r therefore waits for ${ \bf { I } } _ { r + 1 }$ . SLAM starts at the first detection frame $t _ { \mathrm { i n i t } }$ , with preceding frames used only for motion initialization. Image–mask pairs are submitted chronologically after their masks are finalized, and submitted masks are not revised. After the last detection, SAM2 propagates already registered objects over the remaining frames. The static observation mask is

$$
{ \bf V } _ { t } = { \bf V } _ { t } ^ { \mathrm { r g b } } \odot { \bf V } _ { t } ^ { \mathrm { d e p t h } } \odot ( 1 - { \bf M } _ { t } ) ,\tag{3}
$$

where $\mathbf { V } _ { t } ^ { \mathrm { r g b } }$ and $\mathbf { V } _ { t } ^ { \mathrm { d e p t h } }$ indicate valid RGB pixels and valid monocular depth predictions, respectively, and ⊙ denotes element-wise multiplication.

![](images/8e336509faeb5ff3cb08d4359884e703571f4c18c91fb9b8a258332169f22bfc.jpg)  
Fig. 3: LoRA-based fine-tuning (Section III-B.1). Static background images and synthetic occlusion masks provide training pairs for the Stable Diffusion inpainting model. Only the LoRA [40] adapters are optimized using the noiseprediction loss, while the base U-Net, VAE, and text encoder remain frozen.

## B. Asynchronous Diffusion-based Background Completion

The process of asynchronous diffusion-based background completion roughly shown in Fig. 1 (b) is explained in detail.

1) Model and LoRA-based Fine-tuning: We fine-tuned a latent diffusion inpainting model initialized from Stable Diffusion Inpainting (2022) [2] to reconstruct static-background content in motion-masked regions. The variational autoencoder (VAE) encoder ${ \mathcal { E } } _ { \mathrm { V A E } }$ , VAE decoder $\mathcal { D } _ { \mathrm { V A E } }$ , text encoder $\tau$ and the pretrained denoising U-Net parameters $\theta _ { 0 }$ are frozen, while only the Low-Rank Adaptation (LoRA) (2022) [40] parameters ϕ inserted into the U-Net are optimized. During inference, diffusion is invoked only for selected keyframes and is executed asynchronously in an independent worker, so camera tracking does not synchronously wait for completion. In the following, two-dimensional masks are broadcast across RGB channels, and 1 − M denotes the element-wise mask complement. Norms of image and latent arrays are evaluated after vectorization.

Training samples are constructed from static background images using synthetic occlusion masks. The LoRA finetuning pipeline is illustrated in Fig. 3. Given a ground-truth (GT) static image $\mathbf { I ^ { \mathrm { g t } } } \in \mathbb { R } ^ { H \times W \times 3 }$ and a sampled binary training mask $\mathbf { M } ^ { \mathrm { { t r } } } \in \{ 0 , 1 \} ^ { H \times W }$ , the masked image is

$$
{ \bf \cal I } ^ { \mathrm { m a s k } } = \left( 1 - { \bf M } ^ { \mathrm { t r } } \right) \odot { \bf \cal I } ^ { \mathrm { g t } } .\tag{4}
$$

The ground-truth and masked images are encoded into latent space as

$$
{ \bf z } _ { 0 } = \mathcal { E } _ { \mathrm { V A E } } \left( { \bf I } ^ { \mathrm { g t } } \right) , \qquad { \bf z } ^ { \mathrm { m a s k } } = \mathcal { E } _ { \mathrm { V A E } } \left( { \bf I } ^ { \mathrm { m a s k } } \right) .\tag{5}
$$

Both latent arrays have dimensions $H _ { z } \times W _ { z } \times C _ { z } ,$ , where $H _ { z } .$ $W _ { z }$ , and $C _ { z }$ denote the latent height, width, and number of channels, respectively.

The binary training mask is resized to the latent resolution as

$$
{ { \bf { m } } ^ { \mathrm { { l a t } } } } = \boldsymbol { S } \left( { { \bf { M } } ^ { \mathrm { { t r } } } } \right) ,\tag{6}
$$

where $\boldsymbol { S } ( \cdot )$ denotes spatial resampling and $\begin{array} { r l } { \mathbf { m } ^ { \mathrm { l a t } } } & { { } \in } \end{array}$ $[ 0 , 1 ] ^ { H _ { z } \times \dot { W _ { z } } \times 1 }$

We use an ϵ-prediction diffusion formulation. At diffusion timestep $n ,$ , Gaussian noise is added to the ground-truth latent as

$$
\begin{array} { r } { \mathbf { z } _ { n } = \sqrt { \bar { \alpha } _ { n } } \mathbf { z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { n } } \epsilon . } \end{array}\tag{7}
$$

Here, $\begin{array} { r } { \bar { \alpha } _ { n } = \prod _ { j = 1 } ^ { n } \alpha _ { j } } \end{array}$ is the cumulative signal-retention coefficient determined by the diffusion noise schedule, and $\alpha _ { j }$ denotes the signal-retention coefficient at diffusion timestep $j .$ The noise ϵ has the same shape as $\mathbf { z } _ { 0 }$ , with entries independently sampled from $\mathcal { N } ( 0 , 1 )$

Let $\mathbf { c } = \mathcal { T } ( p )$ denote the text-conditioning embedding of prompt p. Following the inpainting formulation, the denoising U-Net receives the noisy latent together with the maskedimage latent and the latent-resolution mask:

$$
{ \hat { \boldsymbol { \epsilon } } } = \epsilon _ { \theta _ { 0 } , \phi } \left( [ { \bf z } _ { n } , { \bf z } ^ { \mathrm { m a s k } } , { \bf m } ^ { \mathrm { l a t } } ] , n , { \bf c } \right) ,\tag{8}
$$

where [·] denotes channel-wise concatenation. The LoRA parameters are optimized using the standard diffusion noiseprediction objective

$$
\mathcal { L } _ { \mathrm { d i f f } } = \| \hat { \epsilon } - \epsilon \| _ { 2 } ^ { 2 } .\tag{9}
$$

2) Asynchronous Keyframe Inference: Background completion is invoked only for selected keyframes that contain sufficiently large dynamic regions with insufficient coverage from the current Gaussian map. The asynchronous backgroundcompletion pipeline is illustrated in Fig. 4. Let ${ \bf C } _ { t } \mathrm { ~  ~ \in ~ }$ $\{ 0 , 1 \} ^ { H \times W }$ denote its rendered coverage at the frame pose. For single-channel maps, $A ( u , v )$ denotes the value of array A at horizontal coordinate u and vertical coordinate v in the pixel domain $\Omega = \{ 1 , \dots , W \} \times \{ 1 , \dots , H \}$ . Coverage is defined by

$$
C _ { t } ( u , v ) = \mathbb { I } \big [ A _ { t } ^ { \mathrm { a l l } } ( u , v ) \geq \tau _ { \mathrm { c o v } } \big ] ,\tag{10}
$$

where $\mathbf { A } _ { t } ^ { \mathrm { a l l } } \ \in \ [ 0 , 1 ] ^ { H \times W }$ is the accumulated-opacity image rendered from the combined observed and generated Gaussian map $\mathcal { G } _ { \mathrm { O } } \cup \mathcal { G } _ { \mathrm { G } } , \tau _ { \mathrm { c o v } } \in ( 0 , 1 ]$ is the coverage threshold, and $\mathbb { I } [ \cdot ]$ is the indicator function. We define the masked region with insufficient map coverage as

$$
\mathbf { U } _ { t } = \mathbf { M } _ { t } \odot \left( \mathbf { 1 } _ { H \times W } - \mathbf { C } _ { t } \right) ,\tag{11}
$$

![](images/611c2c151933b0402302f090c3bf3df09a23af056c97f14d5b06a58e8337c712.jpg)  
Fig. 4: Asynchronous keyframe inference (Section III-B.2). Eligible keyframes undergo asynchronous background completion using the LoRA-adapted inpainting model with all model parameters fixed. After VAE decoding, compositing retains generated pixels within the full motion mask and preserves original pixels elsewhere. Tracking and observedmap optimization proceed without waiting for completion, and completed frames are excluded from camera-pose optimization.

where $\mathbf { 1 } _ { H \times W }$ is the all-one array and $U _ { t } ( u , v ) = 1$ indicates a pixel with insufficient map coverage. Its normalized area is

$$
a _ { t } = \frac { 1 } { H W } \sum _ { v = 1 } ^ { H } \sum _ { u = 1 } ^ { W } U _ { t } ( u , v ) .\tag{12}
$$

A diffusion task is scheduled only if

$$
a _ { t } \geq \tau _ { \mathrm { c o m p } } ,\tag{13}
$$

where $\tau _ { \mathrm { c o m p } }$ is the minimum unknown-area threshold.

For each selected keyframe, the full-resolution masked image $\left( 1 - \mathbf { M } _ { t } \right) \odot \mathbf { I } _ { t }$ and its motion mask ${ { \bf { M } } _ { t } }$ are passed to the diffusion inpainting model, producing an inpainted image $\hat { \mathbf { I } } _ { t } .$ . Only the generated pixels within the masked region are retained:

$$
\mathbf { B } _ { t } = \mathbf { M } _ { t } \odot \hat { \mathbf { I } } _ { t } + \left( 1 - \mathbf { M } _ { t } \right) \odot \mathbf { I } _ { t } .\tag{14}
$$

Thus, all unmasked pixels remain unchanged. Completion is executed asynchronously, and the tracking and observed-map optimization loops do not wait for its result. The backgroundcompleted frame $\mathbf { B } _ { t }$ is never used for camera pose estimation.

![](images/02de130a09514a04063dcbb5fcbfd320cb0f5630f1f1972e0f511b6156ad21bc.jpg)  
Fig. 5: Gaussian primitives initialization (Section III-C.2). At the initial SLAM frame, Metric3D v2 predicts depth from the original RGB image. Valid static pixels are back-projected using the predicted depth and camera intrinsics to initialize the observed Gaussian map $\mathcal { G } _ { \mathrm { O } }$

## C. Provenance-aware 3DGS-SLAM Pipeline

1) 3D Gaussian-based Map Representation: We represent the scene using the 3DGS formulation, in which the environment is modeled as a set of anisotropic 3D Gaussian primitives:

$$
\mathcal { G } = \{ G _ { i } = ( \pmb { \mu } _ { i } , \pmb { \Sigma } _ { i } , o _ { i } , \mathbf { h } _ { i } , q _ { i } ) \ | \ i = 1 , \ldots , N \} .\tag{15}
$$

Here, N is the current number of primitives. Each Gaussian primitive is parameterized by its center $\pmb { \mu } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ , covariance $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ , scalar opacity $o _ { i } ~ \in ~ ( 0 , 1 )$ , and spherical harmonic coefficients $\mathbf { h } _ { i } \in \mathbb { R } ^ { \bar { 3 } \times ( L + 1 ) ^ { 2 } }$ representing viewdependent appearance, where L denotes the maximum order of the spherical harmonics expansion [8]. The provenance label $q _ { i } \in \{ \mathrm { O B S E R V E D } $ , GENERATED} distinguishes primitives initialized from real static RGB pixels and diffusion-generated RGB pixels. Accordingly, we denote the observed and generated Gaussian subsets by $\mathcal { G } _ { \mathrm { O } }$ and $\mathcal { G } _ { \mathrm { G } }$ , respectively, with $\mathcal { G } = \mathcal { G } _ { \mathrm { O } } \cup \mathcal { G } _ { \mathrm { G } }$

Given a camera pose $\mathbf { T } _ { \mathrm { C W } } ^ { t } \in \mathrm { S E } ( 3 )$ (rotation + translation), world-frame primitives are transformed into the camera frame and rendered using the standard differentiable rasterization and alpha-compositing procedure of 3DGS [8]. The subscript CW denotes the world-to-camera transformation, and WC denotes its inverse.

2) Gaussian Primitives Initialization: For the initial SLAM frame at time $t _ { \mathrm { i n i t } }$ , inspired by WildGS-SLAM [18], we initialize the observation-supported Gaussian map from valid static pixels selected by $\mathbf { V } _ { t _ { \mathrm { i n i t } } }$ . Metric depth $\tilde { \mathbf { D } } _ { t _ { \mathrm { i n i t } } } ^ { \mathbf { \Gamma } } \in \mathbb { R } ^ { H \times W }$ is predicted from the original frame $\mathbf { I } _ { t _ { \mathrm { i n i t } } }$ using Metric3D $\mathbf { v } 2$ [13], and the valid pixels are back-projected using the camera intrinsics to initialize $\mathcal { G } _ { \mathrm { O } }$ . The resulting primitives are assigned $q _ { i } = \mathrm { O B S E R V E D }$ and provide the starting point cloud for subsequent tracking and mapping (see Fig. 5). All depth maps represent camera-frame z-coordinates, with finite positive predictions selected by the depth-validity mask.

3) Camera Tracking: Given a new frame at time t, we optimize only the camera pose $\mathbf { T } _ { \mathrm { C W } } ^ { t }$ , while keeping all Gaussian parameters fixed (see Fig. 1 (c.1)). Only observation-supported

Gaussian primitives and valid static pixels are used for pose estimation:

$$
\mathcal { L } _ { \mathrm { p h o } } ^ { t } = \left. \mathbf { V } _ { t } \odot \left[ \mathcal { R } _ { \mathrm { r g b } } \left( \mathcal { G } _ { \mathrm { O } } , \mathbf { T } _ { \mathrm { C W } } ^ { t } \right) - \mathbf { I } _ { t } \right] \right. _ { 1 } ,\tag{16}
$$

$$
\mathcal { L } _ { \mathrm { g e o } } ^ { t } = \left. \mathbf { V } _ { t } \odot \left[ \mathcal { R } _ { \mathrm { d e p t h } } \left( \mathcal { G } _ { \mathrm { O } } , \mathbf { T } _ { \mathrm { C W } } ^ { t } \right) - \tilde { \mathbf { D } } _ { t } \right] \right. _ { 1 } .\tag{17}
$$

Here, $\mathcal { R } _ { \mathrm { r g b } }$ and $\mathcal { R } _ { \mathrm { d e p t h } }$ produce color and depth arrays of dimensions $H \times W \times 3$ and $H \times W$ , respectively, and $\tilde { \mathbf { D } } _ { t }$ is predicted from the original frame $\mathbf { I } _ { t }$ by Metric3D v2 [13]. The final tracking objective is

$$
\mathcal { L } _ { \mathrm { t r a c k } } ^ { t } = \lambda _ { \mathrm { p h o } } \mathcal { L } _ { \mathrm { p h o } } ^ { t } + \lambda _ { \mathrm { g e o } } \mathcal { L } _ { \mathrm { g e o } } ^ { t } ,\tag{18}
$$

where $\lambda _ { \mathrm { p h o } }$ and $\lambda _ { \mathrm { g e o } }$ balance the photometric and geometric residuals. Camera pose optimization is performed using gradient descent for a fixed number of iterations $N _ { \mathrm { t r a c k } }$ per frame. Generated Gaussian primitives and dynamic pixels do not contribute gradients to camera pose optimization.

4) Keyframing and Co-visibility: To avoid optimizing all frames, we maintain a fixed-size sliding window ${ \mathcal W } _ { \mathrm { c } }$ of keyframe indices selected based on a co-visibility criterion (see Fig. 1 (c.2)). For each incoming frame, we estimate covisibility by measuring the intersection over union (IoU) of the sets of visible observation-supported Gaussian primitives between the current frame and the latest keyframe. If this score falls below a threshold, or if the camera has moved significantly relative to the median rendered depth of the observed map in the latest keyframe, the current frame is selected as a keyframe.

Real static pixels from a newly selected keyframe are immediately sent to the mapping backend. Keyframes satisfying the completion criterion in Section III-B are additionally submitted to the asynchronous diffusion worker.

5) Provenance-aware Mapping: During mapping, camera poses estimated during tracking are fixed, and G is optimized using multi-view static observations (see Fig. 1 (c.3)). Following the predicted-depth-guided Gaussian initialization strategy of WildGS-SLAM [18], valid static pixels from each new keyframe f are randomly subsampled:

$$
\mathbf { N } _ { f } = { \mathrm { S a m p l e } } _ { \rho } ( \mathbf { V } _ { f } ) ,\tag{19}
$$

where Sampl $^ { \mathrm { ~ \small ~ 3 ~ } } \rho$ selects a fraction $\rho$ of valid static pixels. Selected pixels are back-projected using predicted depth $\ddot {  Ḋ \mathbf Ḋ \ Ḋ \ Ḍ Ḍ } _ { f }$ and the estimated camera pose to initialize new Gaussian primitives with $q _ { i } = \mathrm { O B S E R V E D }$ , with appearance initialized from the original RGB image $\mathbf { I } _ { f }$ . The same procedure applies to both ordinary static regions and newly revealed backgrounds.

We use the current sliding window W together with two randomly sampled past keyframe indices W to reduce forgetting of the global map, and define $\mathcal { W } = \mathcal { W } _ { \mathrm { c } } \cup \mathcal { W } _ { \mathrm { I } }$ . Reusing the photometric and geometric losses in (16) and (17), the mapping objective is

$$
\begin{array} { r l r } {  { \mathcal L _ { \mathrm { m a p } } = \sum _ { f \in \mathcal W } ( \lambda _ { \mathrm { p h o } } \mathcal L _ { \mathrm { p h o } } ^ { f } + \lambda _ { \mathrm { g e o } } \mathcal L _ { \mathrm { g e o } } ^ { f } ) } } \\ & { } & { + \lambda _ { \mathrm { i s o } } \sum _ { G _ { i } \in \mathcal G _ { \mathrm { O } } } \| \mathbf s _ { i } - \bar { s } _ { i } \mathbf 1 _ { 3 } \| _ { 1 } , } \end{array}\tag{20}
$$

where $\mathbf { s } _ { i } ~ = ~ [ s _ { i , x } , s _ { i , y } , s _ { i , z } ] ^ { \top } ~ \in ~ \mathbb { R } ^ { 3 }$ denotes the anisotropic scaling vector, ${ \bar { s } } _ { i } = \frac { 1 } { 3 } ( s _ { i , x } + s _ { i , y } + s _ { i , z } )$ is its mean, and $\lambda _ { \mathrm { i s o } }$ controls the isotropic regularization strength, following MonoGS (2024) [10]. The same residuals depend on both the map and camera poses: tracking optimizes the poses, whereas mapping optimizes the observed Gaussian parameters. Unlike MonoGS, we retain a geometric loss under monocular RGB input using depth predicted by Metric3D v2 (2024) [13].

When an asynchronous completion becomes available, Metric3D v2 [13] predicts depth from the completed frame:

$$
\hat { \mathbf { D } } _ { t } = \mathrm { M e t r i c } 3 \mathrm { D } ( \mathbf { B } _ { t } ) .\tag{21}
$$

Here, t denotes the source keyframe index, irrespective of when the result becomes available. The source mask $\mathbf { U } _ { t }$ and pose are saved when the task is submitted. At integration, coverage is recomputed from the current map at the saved pose; only pixels in $\mathbf { U } _ { t }$ that remain uncovered according to the recomputed coverage mask and have finite positive completeddepth predictions are used to initialize generated geometry.

For a pixel $( u , v )$ in this region, let $\bar { \pmb x } = [ u , v , 1 ] ^ { \top }$ be its homogeneous image coordinate. The camera- and world-frame points are

$$
\begin{array} { r l } & { \hat { \boldsymbol { p } } _ { \mathrm { c } , t } ( \boldsymbol { u } , \boldsymbol { v } ) = \hat { D } _ { t } ( \boldsymbol { u } , \boldsymbol { v } ) \mathbf { K } ^ { - 1 } \bar { \boldsymbol { x } } , } \\ & { \hat { \boldsymbol { p } } _ { \mathrm { w } , t } ( \boldsymbol { u } , \boldsymbol { v } ) = \mathbf { R } _ { \mathrm { W C } } ^ { t } \hat { \boldsymbol { p } } _ { \mathrm { c } , t } ( \boldsymbol { u } , \boldsymbol { v } ) + \boldsymbol { t } _ { \mathrm { W C } } ^ { t } . } \end{array}\tag{22}
$$

Here, $\mathbf { K } \in \mathbb { R } ^ { 3 \times 3 }$ is the intrinsic matrix for the image resolution used, and $\mathbf { R } _ { \mathrm { W C } } ^ { t } \in \mathrm { S O } ( 3 )$ (rotation) and $t _ { \mathrm { W C } } ^ { t } \in \mathbb { R } ^ { 3 }$ are the rotation and translation of $\mathbf { T } _ { \mathrm { W C } } ^ { t } = ( \mathbf { T } _ { \mathrm { C W } } ^ { t } ) ^ { - \mathrm { i } }$ . The points initialize provisional Gaussian primitives in $\mathcal { G } _ { \mathrm { G } } ^ { } ,$ , with appearance taken from $\mathbf { B } _ { t }$ and provenance $q _ { i } = \mathrm { G E N E R A T E D }$ . These primitives remain fixed until they are removed when valid static observations become available and are excluded from camera pose optimization and the observed-map loss.

Generated primitives are removed using a depth-aware projective rule when subsequent keyframes provide valid static observations. For a generated primitive $G _ { i }$ and a subsequent keyframe f, its camera-frame center and depth are

$$
\begin{array} { r } { \pmb { p } _ { i , f } = \mathbf { R } _ { \mathrm { C W } } ^ { f } \pmb { \mu } _ { i } + \pmb { t } _ { \mathrm { C W } } ^ { f } , \qquad z _ { i , f } = [ \pmb { p } _ { i , f } ] _ { 3 } , } \end{array}\tag{23}
$$

where $\mathbf { R } _ { \mathrm { C W } } ^ { f }$ and $\pmb { t } _ { \mathrm { C W } } ^ { f }$ are the rotation and translation of $\mathrm { \mathbf { T } } _ { \mathrm { C W } } ^ { f }$ . For $z _ { i , f } ~ > ~ 0$ , let ${ \pmb u } _ { i , f } ~ = ~ \mathrm { r o u n d } ( \pi ( { \bf K } { p } _ { i , f } ) )$ be its nearest projected pixel, where $\pi ( [ x , y , z ] ^ { \top } ) = [ x / z , \overset { . } { y } / z ] ^ { \top }$ and rounding is component-wise. Let Ω denote the valid image pixel domain. The generated primitives to be deleted are

$$
\mathcal { G } _ { \mathrm { G } } ^ { \mathrm { d e l } } ( f ) = \left\{ G _ { i } \in \mathcal { G } _ { \mathrm { G } } \left| \begin{array} { l l } { z _ { i , f } > 0 , \quad u _ { i , f } \in \Omega , } \\ { V _ { f } ( u _ { i , f } ) = 1 , } \\ { z _ { i , f } \leq \tilde { D } _ { f } ( u _ { i , f } ) + \tau _ { \mathrm { r e p l a c e } } } \end{array} \right. \right\} \mathrm { , }\tag{24}
$$

where $\tau _ { \mathrm { r e p l a c e } } \geq 0$ is a depth tolerance in meters. This criterion removes generated primitives whose projected centers lie at or in front of the predicted static surface, or within $\tau _ { \mathrm { r e p l a c e } }$ behind it. Primitives farther behind the surface are retained, as their geometry remains unverified from the current view. The criterion uses projected Gaussian centers as an approximation of primitive-level visibility. Newly revealed static pixels are reconstructed through (19), while existing observed primitives are refined by (20). New observed primitives use only realframe RGB and predicted depth and do not inherit parameters from deleted generated primitives.

Together, static-pixel tracking and mapping, asynchronous background completion, and observation-based replacement provide provisional representations of occluded backgrounds while keeping generated primitives outside camera pose optimization. Scene visualization renders $\mathcal { G } _ { \mathrm { O } } \cup \mathcal { G } _ { \mathrm { G } }$ , whereas tracking and observed-map optimization render only G<sub>O</sub>.

## IV. EXPERIMENTS AND RESULTS

## A. Experimental Setup

1) Datasets: We evaluated our DPGS-SLAM on two widely used dynamic RGB-D benchmarks: (a) TUM RGB-D (2012) [15] dataset and (b) BONN RGB-D dynamic (2019) [16] dataset. Both datasets capture indoor scenes using handheld RGB-D cameras and provide ground-truth camera trajectories. Following prior work on dynamic SLAM such as GassiDy (2025) [27] and WildGS-SLAM (2025) [18], we selected sequences involving human motion and object interactions to evaluate robustness in dynamic scenes.

We additionally evaluated static-background reconstruction on three sequences from the StaDy4D (2026) [17] dataset. StaDy4D is a synthetic dataset that provides paired static and dynamic RGB sequences rendered along identical camera trajectories. The dynamic sequences were used as SLAM inputs, while the corresponding static sequences provided ground-truth (GT) references for evaluating the reconstructed background and were not involved in tracking or mapping optimization.

For qualitative ablation comparisons, we used the wandering sequence from WildGS-SLAM iPhone (2025) [18] and freiburg3 sitting halfsphere from TUM RGB-D [15], representing limited background observations under rapid viewpoint changes and prolonged human occlusion, respectively.

Across all evaluations, our DPGS-SLAM used only monocular RGB images and camera intrinsics as input; neither sensor nor ground-truth depth was used for tracking, mapping, or background completion. Ground-truth depth from StaDy4D was used only for evaluation, including the construction of regional masks and quantitative assessment of geometric accuracy.

For LoRA-based fine-tuning of the diffusion inpainting model for static-background completion (Section III-B), we used indoor scene images sampled from ScanNet (2017) [44] and ScanNet++ (2023) [45] datasets. Synthetic occlusion masks were applied to these images to construct the maskedimage training pairs described in (4)–(9).

2) Metrics: Camera tracking accuracy was evaluated using the ATE (absolute trajectory error), reported as RMSE (root mean squared error) in centimeters. Following WildGS-SLAM [18] and DROID-W [36], we evaluated the complete camera trajectory, including non-keyframes, after aligning the estimated trajectory with the ground-truth using evo (2017) [46] with Sim(3) Umeyama alignment [47].

To assess reconstruction quality under dynamic scenes, we report standard photometric rendering metrics commonly adopted in 3DGS-SLAM (like GassiDy [27] and WildGS-SLAM [18]), including PSNR (peak signal-to-noise ratio), SSIM (structural similarity index) [48], and LPIPS (learned perceptual image patch similarity) [49]. On StaDy4D dataset, full-frame renderings were compared with paired static RGB frames at matching time steps, providing a backgroundreconstruction reference that is independent of the dynamic input frames.

WildGS-SLAM (2025) [18]  
TABLE I: Camera tracking accuracy measured by ATE RMSE (↓ [cm]) on BONN RGB-D and TUM RGB-D.
<table><tr><td rowspan="2">Paradigm</td><td rowspan="2">Methods</td><td rowspan="2">Type</td><td colspan="5">BONN RGB-D [16]</td><td colspan="5">TUM RGB-D [15]</td></tr><tr><td>ps.tr</td><td>ps.tr2</td><td>balloon</td><td>mv.box2</td><td>Avg.</td><td>wk.xyz</td><td>wk.hf</td><td>wk.st</td><td>st.hf</td><td>Avg.</td></tr><tr><td rowspan="4">Traditional SLAM methods</td><td>DynaSLAM (2018) [34]</td><td>S</td><td>6.1</td><td>7.8</td><td>3.0</td><td>3.9</td><td>5.2</td><td>1.7</td><td>2.6</td><td>0.7</td><td>2.8</td><td>2.0</td></tr><tr><td>ReFusion (2019) [16]</td><td>D</td><td>28.9</td><td>46.3</td><td>17.5</td><td>17.9</td><td>27.7</td><td>9.9</td><td>10.4</td><td>1.7</td><td>11.0</td><td>8.3</td></tr><tr><td>ORB-SLAM3 (2021) [29]</td><td>S</td><td>70.7</td><td>77.9</td><td>5.8</td><td>3.1</td><td>39.4</td><td>28.1</td><td>30.5</td><td>2.0</td><td>2.6</td><td>15.8</td></tr><tr><td>NICE-SLAM (2022) [41]</td><td>D</td><td>24.5</td><td>53.6</td><td>24.4</td><td>8.3</td><td>27.7</td><td>113.8</td><td>X</td><td>88.2</td><td>45.0</td><td>一</td></tr><tr><td rowspan="4">NeRF-based SLAM methods</td><td>ESLAM (2023) [42]</td><td>D</td><td>48.0</td><td>51.4</td><td>22.6</td><td>17.7</td><td>34.9</td><td>45.7</td><td>60.8</td><td>93.6</td><td>3.6</td><td>50.9</td></tr><tr><td>RoDyn-SLAM (2024) [7]</td><td>D</td><td>14.5</td><td>13.8</td><td>7.9</td><td>12.6</td><td>12.2</td><td>8.3</td><td>5.6</td><td>1.7</td><td>4.4</td><td>5.0</td></tr><tr><td>Gaussian-SLAM (2023) [43]</td><td>D</td><td>109.2</td><td>118.7</td><td>65.2</td><td>31.7</td><td>81.2</td><td>133.7</td><td>80.7</td><td>19.1</td><td>5.4</td><td>59.7</td></tr><tr><td>SplaTAM (2024) [9]</td><td>D</td><td>138.4</td><td>126.3</td><td>35.8</td><td>22.0</td><td>80.6</td><td>149.2</td><td>157.8</td><td>85.3</td><td>14.0</td><td>101.6</td></tr><tr><td rowspan="5">3DGS-based SLAM methods</td><td>GassiDy (2025) [27]</td><td>D</td><td>10.3</td><td>13.0</td><td>2.6</td><td>5.4</td><td>7.8</td><td>3.5</td><td>3.7</td><td>0.6</td><td>2.4</td><td>2.6</td></tr><tr><td>WildGS-SLAM (2025) [18]</td><td>D</td><td>3.9</td><td>3.1</td><td>2.8</td><td>2.5</td><td>3.1</td><td>1.3</td><td>1.6</td><td>0.4</td><td>2.0</td><td>1.3</td></tr><tr><td>DROID-W (2026) [36]</td><td>D</td><td>3.3</td><td>2.9</td><td>2.6</td><td>2.3</td><td>2.8</td><td>1.2</td><td>1.6</td><td>0.5</td><td>1.4</td><td>1.2</td></tr><tr><td>DPGS-SLAM (Ours)</td><td>D</td><td>3.5</td><td>3.1</td><td>2.0</td><td>2.1</td><td>2.7</td><td>1.1</td><td>1.5</td><td>0.4</td><td>1.8</td><td>1.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Notation: “–” denotes absence of mention, and “X” indicates tracking failures. S and D denote sparse and dense SLAM, respectively.

Input GT (ground-truth)  
SplaTAM (2024) [9]  
MonoGS (2024) [10]  
DROID-W (2026) [36]  
![](images/c34c3b5c03f9c33e8d9fa1884c83d5a3f47caec106c37c0ee4077bd37b771e34.jpg)  
Fig. 6: Visual comparison of the rendered images and corresponding camera trajectories on TUM RGB-D (wk.xyz) and BONN RGB-D (ps.tr2) datasets. Orange boxes highlight artifacts in baseline renderings, including ghosting, blurring, and local geometric distortions.

3) Baselines: We compared our DPGS-SLAM against the state-of-the-art methods from different SLAM paradigms: Traditional ones (DynaSLAM (2018) [34], ReFusion (2019) [16], ORB-SLAM3 (2021) [29]) including both sparse and dense SLAM methods, Neural implicit ones (NICE-SLAM (2022) [41], ESLAM (2023) [42], RoDyn-SLAM (2024) [7]) as dense but NeRF-based SLAM, and 3DGS-based ones (Gaussian-SLAM (2023) [43], MonoGS (2024) [10],

SplaTAM (2024) [9], as well as dynamic-oriented methods GassiDy (2025) [27], WildGS-SLAM (2025) [18] and DROID-W (2026) [36]) as dense SLAM similar to ours. This selection supports a comparison of both tracking robustness and reconstruction quality in dynamic environments.

For the tracking comparison in Table I, baseline results were collected from established evaluations on the TUM RGB-D and BONN RGB-D benchmarks, as reported in RoDyn-SLAM [7], GassiDy [27], and DROID-W [36]. The corresponding input modalities and method variants followed these published benchmarks, with average ATE RMSE values recalculated over the four selected sequences of each dataset. For StaDy4D, MonoGS [10], WildGS-SLAM [18], and DROID-W [36] were evaluated using their public implementations with identical dynamic RGB inputs and mapping optimization settings. DROID-W’s optional Gaussian mapping module was enabled (mapping.enable: True) for reconstruction evaluation.

![](images/6f930ce34547c0a5af61845cc10d275097e44904a4bcf7fb0b73b632d0dbfba1.jpg)  
Fig. 7: Qualitative comparison of dynamic-region estimates on BONN RGB-D dataset. From left to right, each row shows the input RGB frame, the RoDyn-SLAM motion mask, the DROID-W dynamic uncertainty map, and the DPGS-SLAM motion mask. Orange boxes highlight regions with incorrect motion segmentation (e.g., false positives or missed dynamic areas).

4) Implementation Details: All experiments were conducted on a workstation equipped with an NVIDIA RTX A6000 GPU. Motion discovery was performed every S = 10 frames using $\Delta = 1 0$ consecutive pairs of forward and backward optical flows, with an object-association IoU threshold of 0.5. We used a sliding window of $| \mathcal { W } _ { \mathrm { c } } | = 1 0$ keyframes together with two randomly sampled past keyframes. Observation-supported Gaussian primitives were initialized by randomly sampling a fraction $\rho = 0 . 0 6 2 5$ of valid static pixels. Background completion was triggered when $a _ { t } \ge \tau _ { \mathrm { c o m p } } = 0 . 0 3$ , with $\tau _ { \mathrm { c o v } } =$ 0.5 used to identify insufficient map coverage, prioritizing sizable unsupported regions while limiting requests for small residual gaps. The replacement tolerance $\tau _ { \mathrm { r e p l a c e } } = 0 . 0 5$ m allows a small margin behind the predicted static surface to accommodate depth discrepancies while retaining generated geometry farther behind it. The loss weights in (18) and (20) were set to $\lambda _ { \mathrm { p h o } } = 0 . 9 , \lambda _ { \mathrm { g e o } } = 0 . 1$ , and $\lambda _ { \mathrm { i s o } } = 1 0 .$ . Camera pose optimization used Adam [50], with learning rates of 0.003 for rotation and 0.001 for translation, and $N _ { \mathrm { t r a c k } } = 1 0 0$ iterations per frame. Map optimization used 150 iterations per update. For LoRA-based fine-tuning, we used a LoRA rank of 8 and a learning rate of $1 0 ^ { - 4 }$ , while diffusion inference used 30 denoising steps with a guidance scale of 7.5. The remaining Gaussian-parameter optimization settings followed a previous study [8]. The numerical settings were selected empirically through preliminary experiments and, unless otherwise specified, kept fixed across all sequences.

## B. Camera Tracking Accuracy Analysis

We evaluated the camera tracking accuracy on dynamic scenes from the BONN RGB-D dynamic and TUM RGB-D datasets using the ATE RMSE metric.

1) Quantitative Results: As shown in Table I, our DPGS-SLAM achieved the lowest average ATE RMSE of 2.7 cm on the four selected BONN RGB-D sequences. On TUM RGB-D, it achieved an average ATE RMSE of 1.2 cm, matching DROID-W at the reported precision. This tracking performance is supported by motion-mask-based exclusion of dynamic observations and photometric–geometric pose optimization, as further examined in Table IV. Camera tracking uses only observation-supported Gaussian primitives, while diffusion-generated primitives remain excluded from pose optimization.

2) Trajectory Analysis: As shown in Fig. 6, our DPGS-SLAM maintains closer alignment with the ground-truth trajectory and exhibits reduced drift in the illustrated sequences, compared with SplaTAM and MonoGS. On the wk.xyz sequence of TUM RGB-D, DPGS-SLAM and DROID-W exhibit similar trajectory alignment, while WildGS-SLAM shows larger local deviations. On the $p s . t r 2$ sequence of BONN RGB-D, all three methods closely follow the ground-truth trajectory, with minor local deviations.

## C. Map and Motion Mask Quality Analysis

1) Dynamic-scene Rendering Quality on TUM RGB-D and BONN RGB-D: Fig. 6 also presents qualitative comparisons of rendered images on TUM RGB-D and BONN RGB-D datasets. Compared with the evaluated baselines, our DPGS-SLAM produces reconstructions with fewer ghosting artifacts in regions affected by dynamic objects.

2) Motion Mask Quality on BONN RGB-D: We qualitatively compared the motion masks from RoDyn-SLAM [7] with the dynamic uncertainty maps from DROID-W [36] and the motion masks from DPGS-SLAM on the same RGB frames from two BONN RGB-D sequences (mv box2 and balloon). As shown in Fig. 7, the DPGS-SLAM masks align more closely with the visible moving objects in the illustrated frames.

3) Static-background Reconstruction on StaDy4D: To quantitatively assess static-background reconstruction under dynamic occlusions, we conducted experiments on three sequences from StaDy4D benchmark [17]: scene T07 022 06 cctv, scene T10 024 00 car, and scene T03 016 07 pedestrian. We compared our DPGS-SLAM with MonoGS [10], WildGS-SLAM [18], and DROID-W [36]. All methods used dynamic RGB sequences as input, while the paired static RGB frames were used only as ground-truth references for reconstruction evaluation. For the rendering comparisons involving DROID-W, we enabled its optional Gaussian splatting mapping module (mapping.enable: True) in the official implementation. All evaluated methods used the same number of mapping optimization iterations.

As shown in Table II, our DPGS-SLAM achieved the highest PSNR and SSIM and the lowest LPIPS among the compared methods across all three evaluated sequences. Compared with WildGS-SLAM, DPGS-SLAM improved the average PSNR by 4.60 dB. The average SSIM increased by 7.97 percentage points, while the average LPIPS (×100) decreased by 2.97. These results mean better photometric, structural, and perceptual agreement with the static-background references.

Qualitative comparisons are provided in Fig. 8, showing the dynamic inputs, paired static ground-truth, and reconstructed renderings for visual assessment of the occluded background regions.

Dynamic Input  
Static GT  
MonoGS (2024) [10]  
WildGS (2025) [18]  
DROID-W (2026) [36]  
Ours  
![](images/d3c065079064eaab2075e97e45e20f63423ed9b468442a1550537b8bc55b216e.jpg)  
Fig. 8: Qualitative comparison of static-background reconstruction on StaDy4D: scene T07 022 06 cctv (top), scene T10 024 00 car (middle), and scene T03 016 07 pedestrian (bottom). All methods used dynamic RGB sequences for SLAM. The first two columns show the dynamic input and corresponding static ground-truth, followed by the rendered images reconstructed using each method. Orange boxes highlight artifacts in baseline renderings, including ghosting, blurring, and residual moving objects.

TABLE II: Static-background reconstruction quality on StaDy4D using dynamic RGB inputs and paired static groundtruth references.
<table><tr><td>Metrics</td><td>Scene</td><td>MonoGS (2024) [10]</td><td>WildGS (2025) [18]</td><td>DROID-W (2026) [36]</td><td>DPGS-SLAM (Ours)</td></tr><tr><td rowspan="3"></td><td>T0722_cctv 16.6</td><td></td><td>18.9</td><td>21.8</td><td>25.2</td></tr><tr><td>PSNR↑ [dB] T1024_car 16.3</td><td></td><td>19.9</td><td>19.7</td><td>24.5</td></tr><tr><td>T0316-ped 12.6</td><td></td><td>24.8</td><td>24.0</td><td>27.7</td></tr><tr><td rowspan="3">SSIM↑ [%]</td><td>T0722_cctv 68.4</td><td></td><td>75.5</td><td>76.7</td><td>83.7</td></tr><tr><td>T1024_car 60.2</td><td></td><td>69.3</td><td>70.0</td><td>80.1</td></tr><tr><td>T0316-ped 56.1</td><td></td><td>84.0</td><td>83.2</td><td>88.9</td></tr><tr><td rowspan="3">LPIPS↓ (×100)</td><td>T0722_cctv 37.5</td><td></td><td>29.7</td><td>30.0</td><td>28.4</td></tr><tr><td>T1024_car 45.5</td><td></td><td>26.2</td><td>26.7</td><td>20.1</td></tr><tr><td>T0316-ped 62.1</td><td></td><td>19.4</td><td>23.4</td><td>17.9</td></tr></table>

Scene abbreviations: T0722 cctv: scene T07 022 06 cctv; T1024 car: scene T10 024 00 car; T0316 ped: scene T03 016 07 pedestrian. Notation: SSIM and LPIPS are multiplied by 100.

4) Runtime Comparison on StaDy4D: To evaluate computational efficiency, we compared the average per-frame tracking and mapping times of DPGS-SLAM with MonoGS [10], WildGS-SLAM [18], and DROID-W [36] on the three StaDy4D sequences. All methods were evaluated on the same NVIDIA RTX A6000 GPU using identical dynamic RGB inputs. Tracking used 100 iterations per frame for DPGS-SLAM, at most 100 for MonoGS, and the original configurations for WildGS-SLAM and DROID-W. For mapping, all methods used an average of 150 optimization iterations per keyframe, with DROID-W’s Gaussian mapping module enabled. Tracking time includes pose estimation and associated online preprocessing, while mapping time includes Gaussian primitives initialization and optimization. Per-frame times were computed from the accumulated execution time of each stage divided by the total number of processed frames, including non-keyframes. GPU execution was synchronized for timing, and results were averaged over three runs per sequence, as reported in Table III.

TABLE III: Average per-frame tracking and mapping times across three StaDy4D sequences using dynamic RGB inputs.
<table><tr><td>Methods</td><td>| Tracking↓ [ms/frame]</td><td>Mapping↓ [ms/frame]</td></tr><tr><td>MonoGS (2024) [10]</td><td>579</td><td>1134</td></tr><tr><td>WildGS-SLAM (2025) [18]</td><td>372</td><td>3587</td></tr><tr><td>DROID-W (2026) [36]</td><td>249</td><td>1661</td></tr><tr><td>DPGS-SLAM (Ours)</td><td>510</td><td>1205</td></tr></table>

As shown in Table III, DPGS-SLAM achieved an average tracking time of 510 ms/frame and a mapping time of 1205 ms/frame. Its mapping time is comparable to MonoGS and lower than WildGS-SLAM and DROID-W, while tracking remains slower than the latter two methods.

The main tracking overhead arises from iterative pose optimization (292.1 ms/frame). Motion mask generation requires 179.5 ms/frame, including optical flow estimation (141.8 ms/frame), OCLR (6.2 ms/frame), and SAM2 (30.0 ms/frame). However, parallel execution reduces the blocking time imposed by motion mask generation on tracking to 50.5 ms/frame. Diffusion model loading incurs a onetime cost of 5790 ms, while inference takes an average of 390 ms per selected keyframe. As background completion runs asynchronously, it does not block tracking or observed-map optimization, limiting its impact on processing latency.

## D. Ablation Study

1) Tracking Ablation: To assess the contributions of motion mask generation and geometric constraints to camera tracking, we evaluated three ablated configurations on the four dynamic TUM RGB-D sequences. W/o motion masks disables both OCLR and SAM2, allowing all valid pixels to participate in tracking and mapping. OCLR only generates motion masks using OCLR for every frame, without SAM2-based refinement or temporal propagation. W/o geometric loss removes the geometric term from (18), retaining only the photometric loss for camera pose optimization. All other applicable settings remain unchanged. Table IV reports the ATE RMSE for each configuration.

TABLE IV: Tracking ablation results on four dynamic TUM RGB-D sequences (ATE RMSE ↓ [cm]).
<table><tr><td>Configuration</td><td>wk.xyz</td><td>wk.hf</td><td>wk.st</td><td>st.hf</td><td>Avg.</td></tr><tr><td>W/o motion masks</td><td>2.6</td><td>3.4</td><td>2.1</td><td>4.2</td><td>3.1</td></tr><tr><td>OCLR only</td><td>2.0</td><td>3.1</td><td>1.3</td><td>3.0</td><td>2.4</td></tr><tr><td>W/o geometric loss</td><td>2.3</td><td>3.1</td><td>1.1</td><td>2.9</td><td>2.3</td></tr><tr><td>Full DPGS-SLAM</td><td>1.1</td><td>1.5</td><td>0.4</td><td>1.8</td><td>1.2</td></tr></table>

Notation: ATE RMSE is reported in centimeters. Avg. denotes the mean across the four sequences.

As shown in Table IV, the full DPGS-SLAM achieved the lowest average ATE RMSE of 1.2 cm. Without motion masks, the error increased to 3.1 cm, while using OCLR alone reduces it to 2.4 cm, demonstrating the benefit of SAM2- based refinement and temporal propagation. Removing the geometric loss increased the error to 2.3 cm, confirming the contribution of predicted-depth constraints to camera tracking. These results indicate that accurate motion masking and photometric–geometric optimization jointly support the tracking performance.

2) Static-background Reconstruction Ablation: We compared five mapping configurations using the same tracking module, depth predictor, keyframe selection policy, and Gaussianparameter optimization settings. Hard mask excludes pixels within detected dynamic regions from Gaussian primitives initialization and mapping supervision. Soft mask allows Gaussian primitives initialization from these pixels and assigns their mapping residuals a weight of 0.1, while valid pixels outside the masked regions retain unit weight. Adaptive weighting incorporates WildGS-style [18] learned perpixel uncertainty weighting into the same mapping backend. These three variants omit background completion. W/o replacement retains background completion, provenance-based isolation of generated primitives, and independent initialization of observation-supported primitives, but disables deletion of generated Gaussian primitives. Full DPGS-SLAM enables all proposed components.

Table V compares PSNR, SSIM, and LPIPS inside and outside undilated ground-truth dynamic-object masks shared across methods, using paired static RGB references. The masks were derived from paired dynamic and static groundtruth depth maps by identifying pixels where dynamic objects occlude static surfaces. PSNR is computed from regional RGB MSE, while spatial SSIM and AlexNet (2017) [51] - based LPIPS maps are computed on full images and averaged within each region. Results were averaged over frames within each sequence and then equally across the three StaDy4D sequences. Inside the masks, Full DPGS-SLAM improved PSNR by 9.9 dB and SSIM by 32.2 percentage points over adaptive weighting, while reducing LPIPS (×100) from 51.9 to 36.3. Compared with w/o replacement, it improved PSNR by 5.6 dB and SSIM by 17.5 percentage points, while reducing LPIPS (×100) from 43.5 to 36.3. Improvements outside the masks are comparatively limited, indicating that the reconstruction gains are primarily concentrated in regions affected by dynamic occlusions. The comparison with w/o replacement further supports the effectiveness of removing eligible generated Gaussian primitives when real observations become available.

TABLE V: Ablation results inside and outside dynamic-object masks on three StaDy4D sequences.
<table><tr><td rowspan="2">Configuration</td><td colspan="3">Inside mask</td><td colspan="3">Outside mask</td></tr><tr><td>PSNR↑ [dB]</td><td>SSIM↑ [%]</td><td>LPIPS↓ (×100)</td><td>PSNR↑ [dB]</td><td>SSIM↑ [%]</td><td>LPIPS↓ (×100)</td></tr><tr><td>Hard mask</td><td>11.5</td><td>43.7</td><td>50.5</td><td>25.2</td><td>80.2</td><td>23.6</td></tr><tr><td>Soft mask (0.1)</td><td>13.9</td><td>45.4</td><td>51.0</td><td>25.0</td><td>81.4</td><td>22.7</td></tr><tr><td>Adaptive weighting</td><td>12.2</td><td>41.5</td><td>51.9</td><td>25.3</td><td>80.1</td><td>22.6</td></tr><tr><td>W/o replacement</td><td>16.5</td><td>56.2</td><td>43.5</td><td>25.9</td><td>82.7</td><td>22.0</td></tr><tr><td>Full DPGS-SLAM</td><td>22.1</td><td>73.7</td><td>36.3</td><td>26.1</td><td>84.8</td><td>21.4</td></tr></table>

Notation: Entries are averaged over frames within each sequence and then equally across sequences. SSIM and LPIPS are multiplied by 100.

Figure 9 presents qualitative comparisons on wandering (WildGS-SLAM iPhone dataset) and freiburg3 sitting halfsphere (TUM RGB-D dataset), representing insufficient background observations caused by rapid viewpoint changes and human occlusion, and prolonged occlusion by a seated person, respectively. In both sequences, hard and soft masking produce coarse, blurred backgrounds due to insufficient valid observations to constrain Gaussian refinement, while adaptive weighting retains human-shaped ghosting from residual dynamic observations. Background completion provides provisional representations in these insufficiently observed regions. In wandering, disabling replacement allows diffusion-generated Gaussian primitives to obscure parts of the background subsequently reconstructed from real observations. Full DPGS-SLAM mitigates this issue by removing eligible generated primitives and independently initializing observation-supported Gaussian primitives.

3) Geometric Accuracy Evaluation: To evaluate geometric errors introduced by background completion and depth prediction, we compared Metric3D v2 [13] predictions from diffusion-completed and paired static ground-truth RGB images against ground-truth depth at completion keyframes. We further evaluated rendered depth from three mapping configurations: hard mask, w/o replacement, and Full DPGS-SLAM, as defined in Section IV-D.2.

For each Gaussian map, opacity-normalized expected camera-axis depth was rendered at ground-truth viewpoints and evaluated after global trajectory alignment. We computed coverage-penalized AbsRel separately inside and outside the undilated ground-truth dynamic-object masks, using all pixels with ground-truth depth in [0.1, 80) m. Missing, nonfinite, or nonpositive predictions, as well as rendered predictions with accumulated opacity below 0.5, were set to zero, contributing a relative error of 1 per pixel. Scores were averaged over frames within each sequence and then equally across the three sequences (Table VI).

As shown in Table VI, Metric3D v2 achieved comparable depth accuracy on diffusion-completed and static groundtruth RGB images, with masked-region AbsRel values of 0.053 and 0.049, respectively. This indicates limited additional depth error from background completion. For Gaussian map rendering, hard masking produces a substantially higher AbsRel of 0.523 inside the masks, while w/o replacement and Full DPGS-SLAM reduce it to 0.078 and 0.073, respectively. The improvement reflects more accurate and complete reconstruction under the missing-depth penalty. Observation-based replacement provides a further reduction in masked-region error, although the difference is relatively small. Outside the masks, all three mapping configurations achieve similar accuracy (0.086–0.089), indicating that the geometric benefits are primarily concentrated in regions affected by dynamic occlusions.

Input RGB  
Hard mask  
Soft mask (0.1)  
Adaptive weighting  
W/o replacement  
Full DPGS-SLAM  
![](images/5275912bf85aa25057660d3eb725105cd4e1b8d7f64c42ca3e702be3f7ccfb35.jpg)  
Fig. 9: Qualitative ablation comparison on wandering (WildGS-SLAM iPhone dataset, top) and freiburg3 sitting halfsphere (TUM RGB-D dataset, bottom). The first column shows the input RGB frames, followed by renderings from five configurations. The sequences illustrate background reconstruction under limited observations and prolonged occlusion, respectively, with orange boxes highlighting blurring, ghosting, and background occlusion artifacts.

TABLE VI: Geometric accuracy on three StaDy4D sequences (coverage-penalized AbsRel ↓).
<table><tr><td>Configuration</td><td></td><td>|Inside mask Outside mask</td></tr><tr><td colspan="3">Depth prediction</td></tr><tr><td>Metric3D v2 (static GT RGB)</td><td>0.049</td><td>0.053</td></tr><tr><td>Metric3D v2 (diffusion-completed RGB)</td><td>0.053</td><td>0.051</td></tr><tr><td colspan="3">Gaussian map rendering</td></tr><tr><td>Hard mask</td><td>0.523</td><td>0.087</td></tr><tr><td>W/o replacement</td><td>0.078</td><td>0.086</td></tr><tr><td>Full DPGS-SLAM</td><td>0.073</td><td>0.089</td></tr></table>

## E. General Discussion and Limitations

The proposed system DPGS-SLAM relies only on monocular RGB input. The diffusion inpainting model was finetuned offline using LoRA [40], without additional scenespecific fine-tuning on the evaluated sequences. Metric3D v2 [13] provides predicted depth for tracking, mapping, and provisional Gaussian initialization. Periodic optical-flow-based motion discovery may introduce a brief delay in detecting newly entering moving objects, allowing temporary dynamic interference. Such interference may be mitigated by subsequent motion detection and incremental map updates.

Provenance-aware mapping excludes generated Gaussian primitives from camera-pose estimation and observed-map optimization, and selectively deletes them when valid static observations become available. However, persistently unob served regions may retain inaccurate generated appearance or geometry. Asynchronous background completion also incurs computational overhead and does not guarantee real-time performance.

## V. CONCLUSIONS

We present DPGS-SLAM, a monocular RGB 3D Gaussian splatting SLAM system for dynamic environments that provides provisional representations of static backgrounds with insufficient support from real RGB observations. The system combines periodic optical-flow-based motion discovery with SAM2 to obtain temporally consistent motion masks.

For map reconstruction, diffusion-based inpainting enables asynchronous background completion, while provenanceaware Gaussian mapping integrates the generated content as provisional Gaussian primitives. Generated primitives are excluded from camera-pose estimation and observed-map optimization. When valid static observations become available, eligible generated primitives are removed using a depthaware projective rule, while observation-supported Gaussian primitives are independently initialized. Experimental results demonstrated competitive camera tracking, with average ATE RMSE values of 1.2 cm on TUM RGB-D and 2.7 cm on BONN RGB-D. On the three evaluated StaDy4D sequences, DPGS-SLAM improved average PSNR by 4.60 dB over WildGS-SLAM, while ablation studies further support the benefit of observation-based replacement for static-background reconstruction. Future work will explore tighter integration of dynamic object modeling and unified optimization formulations to further extend the applicability of the proposed framework.

## REFERENCES

[1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[2] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR). ieee, 2022, pp. 10 674–10 685.

[3] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “Orb-slam: A versatile and accurate monocular slam system,” IEEE trans on robotics (T-RO), vol. 31, no. 5, pp. 1147–1163, 2015.

[4] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE trans on pattern analysis and machine intelligence, vol. 40, no. 3, pp. 611–625, 2017.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in Computer Vision - ECCV 2020 (LNCS 12346), 2020, pp. 405–421.

[6] C. Ruan, Q. Zang, K. Zhang, and K. Huang, “Dn-slam: A visual slam with orb features and nerf mapping in dynamic environments,” IEEE Sensors J, vol. 24, no. 4, pp. 5279–5287, 2023.

[7] H. Jiang, Y. Xu, K. Li, J. Feng, and L. Zhang, “Rodyn-slam: Robust dynamic dense rgb-d slam with neural radiance fields,” IEEE Robotics and Automation Letters (RA-L), vol. 9, no. 9, pp. 7509–7516, 2024.

[8] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering.” ACM Trans on Graphics (TOG), vol. 42, no. 4, pp. 1–14, 2023.

[9] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, “Splatam: Splat track & map 3d gaussians for dense rgb-d slam,” in 2024 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 357–21 366.

[10] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian splatting slam,” in 2024 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18 039–18 048.

[11] M. Runz, M. Buffier, and L. Agapito, “Maskfusion: Real-time recognition, tracking and reconstruction of multiple moving objects,” in 2018 IEEE intn’l symp on mixed and augmented reality (ISMAR), 2018, pp. 10–20.

[12] Y. Xu, H. Jiang, Z. Xiao, J. Feng, and L. Zhang, “Dg-slam: Robust dynamic gaussian splatting slam with hybrid pose optimization,” in Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 2024, pp. 51 577–51 596.

[13] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C. Shen, and S. Shen, “Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation,” IEEE Trans on pattern analysis and machine intelligence, vol. 46, no. 12, pp. 10 579–10 596, 2024.

[14] N. Ravi et al., “SAM 2: Segment anything in images and videos,” in 13th Intn’l Conf on Learning Representations (ICLR 2025), no. 92 (44 pages), 2025.

[15] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of rgb-d slam systems,” in 2012 IEEE/RSJ Intn’l Conf on Intelligent Robots and Systems (IROS), 2012, pp. 573– 580.

[16] E. Palazzolo, J. Behley, P. Lottes, P. Giguere, and C. Stachniss, “Refusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals,” in 2019 IEEE/RSJ Intn’l Conf on Intelligent Robots and Systems (IROS), 2019, pp. 7855–7862.

[17] H.-T. Tsui, Y.-R. Tuan, E. Lai, and C.-Y. Wang, “Stady4d: Towards complete 4d static-dynamic reconstruction with sigma,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2026, pp. 4697–4707.

[18] J. Zheng, Z. Zhu, V. Bieri, M. Pollefeys, S. Peng, and I. Armeni, “Wildgs-slam: Monocular gaussian splatting slam in dynamic environments,” in 2025 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 11 461–11 471.

[19] H. Kong, X. Yang, and X. Wang, “Generative sparse-view gaussian splatting,” in 2025 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 26 745–26 755.

[20] Y. Zhong, Z. Li, D. Z. Chen, L. Hong, and D. Xu, “Taming video diffusion prior with scene-grounding guidance for 3d gaussian splatting from sparse inputs,” in 2025 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 6133–6143.

[21] A. Paliwal, X. Zhou, W. Ye, J. Xiong, R. Ranjan, and N. K. Kalantari, “RI3D: Few-shot gaussian splatting with repair and inpainting diffusion priors,” in 2025 IEEE/CVF Intn’l Conf on Computer Vision (ICCV), 2025, pp. 25 094–25 103.

[22] E. Weber, A. Holynski, V. Jampani, S. Saxena, N. Snavely, A. Kar, and A. Kanazawa, “NeRFiller: Completing scenes via generative 3d inpainting,” in 2024 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 731–20 741.

[23] Z. Liu, H. Ouyang, Q. Wang, K. L. Cheng, J. Xiao, K. Zhu, N. Xue, Y. Liu, Y. Shen, and Y. Cao, “InFusion: Inpainting 3d gaussians via learning depth completion from diffusion prior,” 2024. [Online]. Available: https://arxiv.org/abs/2404.11613

[24] M. Dahaghin, M. G. Padalkar, M. Toso, A. Del Bue, and V. Murino, “SplatFill: 3d scene inpainting via depth-guided gaussian splatting,” in Pattern Recognition - ICPR 2026 (LNCS 16827), 2026, pp. 19–33.

[25] R. Lai, H. Liu, and H. Wu, “Diffusion-driven 3d gaussian splatting for occlusion-free egocentric scene reconstruction,” in 7th ACM Intn’l Conf on Multimedia in Asia (MMAsia 2025), 2025, pp. 1–7.

[26] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-slam: Realtime simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,” in 2024 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 584–21 593.

[27] L. Wen, S. Li, Y. Zhang, Y. Huang, J. Lin, F. Pan, Z. Bing, and A. Knoll, “Gassidy: Gaussian splatting slam in dynamic environments,” in 2025 IEEE Intn’l Conf on Robotics and Automation (ICRA), 2025, pp. 8471– 8477.

[28] Y. Li, S. Wen, and F. R. Yu, “Drg-slam: Dynamic-aware gaussian splatting slam for rgb-d scenes,” IEEE Robotics and Automation Letters, 2026.

[29] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. M. Montiel, and J. D. Tardos, “Orb-slam3: An accurate open-source library for visual, visual-´ inertial, and multimap slam,” IEEE Trans on Robotics (T-RO), vol. 37, no. 6, pp. 1874–1890, 2021.

[30] R. Scona et al., “Staticfusion: Background reconstruction for dense rgbd slam in dynamic environments,” in 2018 IEEE intn’l conf on robotics and automation (ICRA), 2018, pp. 3849–3856.

[31] Y. Sun, M. Liu, and M. Q.-H. Meng, “Motion removal for reliable rgb-d slam in dynamic environments,” Robotics and Autonomous Systems, vol. 108, pp. 115–128, 2018.

[32] T. Zhang, H. Zhang, Y. Li, Y. Nakamura, and L. Zhang, “Flowfusion: Dynamic dense rgb-d slam based on optical flow,” in 2020 IEEE intn’l conf on robotics and automation (ICRA), 2020, pp. 7322–7328.

[33] M. Kaneko, K. Iwami, T. Ogawa, T. Yamasaki, and K. Aizawa, “Maskslam: Robust feature-based monocular slam by masking using semantic segmentation,” in 2018 IEEE/CVF Confon Computer Vision and Pattern Recognition Workshops (CVPRW), 2018, pp. 258–266.

[34] B. Bescos, J. M. Facil, J. Civera, and J. Neira, “Dynaslam: Tracking,´ mapping, and inpainting in dynamic scenes,” IEEE Robotics and Automation Letters (RA-L), vol. 3, no. 4, pp. 4076–4083, 2018.

[35] S. Cheng, C. Sun, S. Zhang, and D. Zhang, “Sg-slam: A real-time rgb-d visual slam toward dynamic scenes with semantic and geometric information,” IEEE Trans on Instrumentation and Measurement, vol. 72, pp. 1–12, 2023.

[36] M. Li, Z. Zhu, M. Pollefeys, and D. Barath, “Droid-slam in the wild,” in 2026 IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2026, pp. 36 498–36 508.

[37] H. Zhou, J. Chen, and Z. Li, “Dynamic slam with 3-d gaussian splatting supporting monocular sensing,” IEEE Sensors Journal, vol. 25, no. 12, pp. 21 900–21 910, 2025.

[38] M. Li, Y. Zhou, H. Zhou, X. Hu, F. Roemer, H. Wang, and A. Osman, “Dy3dgs-slam: Monocular 3d gaussian splatting slam for dynamic environments,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 14 572–14 578.

[39] J. Xie, W. Xie, and A. Zisserman, “Segmenting moving objects via an object-centric layered representation,” in Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022, pp. 28 023–28 036.

[40] E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in 10th Intn’l Conf on Learning Representations (ICLR 2022), no. 29 (13 pages), 2022.

[41] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, “Nice-slam: Neural implicit scalable encoding for slam,” in 2022 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 12 786–12 796.

[42] M. M. Johari, C. Carta, and F. Fleuret, “Eslam: Efficient dense slam system based on hybrid representation of signed distance fields,” in 2023 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 17 408–17 419.

[43] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-slam: Photo-realistic dense slam with gaussian splatting,” 2023. [Online]. Available: https://arxiv.org/abs/2312.10070

[44] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in 2017 IEEE conference on computer vision and pattern recognition (CVPR). IEEE, 2017, pp. 2432–2443.

[45] C. Yeshwanth, Y.-C. Liu, M. Nießner, and A. Dai, “Scannet++: A highfidelity dataset of 3d indoor scenes,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 12–22.

[46] M. Grupp, “evo: Python package for the evaluation of odometry and SLAM,” https://github.com/MichaelGrupp/evo, 2017.

[47] S. Umeyama, “Least-squares estimation of transformation parameters between two point patterns,” IEEE Trans on pattern analysis and machine intelligence, vol. 13, no. 4, pp. 376–380, 1991.

[48] E. Alexiou and T. Ebrahimi, “Towards a point cloud structural similarity metric,” in 2020 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), 2020, pp. 1–6.

[49] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in 2018 IEEE/CVF Conf on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595.

[50] D. P. Kingma and J. L. Ba, “Adam: A method for stochastic optimization,” in 3rd Intn’l Conf on Learning Representations (ICLR 2015), no. 11 (15 pages), 2015.

[51] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Communications of the ACM, vol. 60, no. 6, pp. 84–90, 2017.