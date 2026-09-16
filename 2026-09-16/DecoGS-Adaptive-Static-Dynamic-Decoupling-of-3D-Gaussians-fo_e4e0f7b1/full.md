# DecoGS: Adaptive Static-Dynamic Decoupling of 3D Gaussians for Free-Viewpoint Video Streaming

Idil Sulo University of Bonn, Almetra

Alexey Supikov V3DEO

Ilke Demir Cauth AI

Sainan Liu Intel Labs

![](images/3621ca16e1337be1f5135c1b4f4c7539c79b8ffb00ed5be9009ab254b208e1b6.jpg)  
3DGStream

![](images/45e4b33e06739fd7ec84c41656ee08ace1677fe767c3ae139cec81c1365d8af2.jpg)  
DecoGS (Ours)

![](images/30d62239ba6ee1e7edf36fd88698765c2387fb580886845b7f1c4f3d8631266e.jpg)  
Per-frame Train Time (s)  
Figure 1. Quality and efficiency comparison. DecoGS achieves superior reconstruction quality while maintaining efficient streaming performance. The left images illustrate high-fidelity rendering results produced by DecoGS in dynamic regions. The right plot compares DecoGS with prior streaming methods [11, 17, 18, 20, 24, 55, 72] demonstrating the effectiveness of static–dynamic decoupling.

## Abstract

Streaming 3D reconstruction demands both speed and temporalfidelity, goals that existing methods undermine by updating every Gaussian every frame, even in static regions. We present DecoGS, a method for efficient online training of 3D Gaussiansfrom streaming videos. Unlike prior methods that update the entire scene indiscriminately, DecoGS introduces an adaptive mechanism that selectively focuses optimization on spatiotemporal regions exhibiting motion or photometric changes. This targeted training strategy eliminates redundant updates that cause flickering and drift in nominally static regions, while enabling fast, high-fidelity scene updates. The pipeline further integrates region-aware Gaussian management through gradient gating and efficient visibility filtering to maintain temporal coherence and a compact memory footprint. On N3DV and MeetRoom, DecoGS achieves 34.55 and 31.60 dB PSNR respectively, outperforming all streaming and offline baselines, while rendering at 261 FPS with 70× lower temporalflicker than the best prior method, requiring no large-scale pretraining.

## 1. Introduction

Free-Viewpoint Video (FVV) reconstruction from multiview captures represents a fundamental challenge in immersive media, powering emerging applications in VR and AR. By enabling interactive and photorealistic exploration of dynamic scenes, FVV transcends traditional 2D video limitations, offering unprecedented viewer agency. Streamingbased FVV, with applications in remote telepresence and sports broadcasting, delivers such responsiveness at scale. This necessitates online 3D reconstruction that incrementally refines scene representations as new frames arrive, with a minimal latency in comparison to offline optimization, making it suitable for interactive applications.

The emergence of Neural Radiance Fields (NeRFs) [42] enabled photorealistic scene reconstruction from multiview images. Following this, synthesizing novel views gained significant popularity and several NeRF-based works [14, 16, 33, 35–37, 46–49, 59, 70] attempted reconstruction of dynamic scenes for FVVs. However, most are limited by either requiring the full video before training (offline) or by failing to maintain temporal consistency as frames arrive (online), reducing utility for streaming.

Recent 3D Gaussian Splatting (3DGS) [29] methods further improve novel view synthesis with real-time rendering. However, adapting these static techniques to streaming video remains challenging as new frames continuously reveal scene changes, requiring adaptation that is both fast and temporally consistent. Building on 3DGS, offline training methods [38, 66, 71, 75] demonstrate high-quality reconstruction from pre-recorded sequences. However, these methods require frames at all time steps to be present prior to training and scene reconstruction. Therefore, they are not practical for live streaming scenarios in which frames arrive sequentially over time. Recently, [62, 68] explore separating static and dynamic scene components for dynamic reconstruction. However, these approaches operate in offline training settings that assume access to the full sequence and therefore do not address incremental optimization for streaming scenarios. Current online training methods try to tackle these challenges in streaming scenarios. IGS [72] reduces per-frame latency by pretraining large motion networks that infer Gaussian updates directly from video streams. While reducing per-frame processing time, it requires 192 GPU-hours of pretraining, transferring the computational burden rather than eliminating it. Another line of work [24, 55] performs per-scene optimization by incrementally updating Gaussian parameters as new frames arrive. While effective for local adaptation, they re-optimize all Gaussians, causing gradient accumulation in static regions that manifests as temporal artifacts.

We observe that in dynamic scenes, more than half of Gaussians do not require updates. Exploiting this sparsity is the key idea behind DecoGS: an incremental streaming framework that restricts optimization to dynamic regions. DecoGS provides an adaptive mechanism that estimates moving and deforming regions per frame, and restricts optimization to Gaussians within these regions while freezing gradient flow in stationary areas. This design suppresses redundant updates and concentrates optimization on the most informative regions, allowing DecoGS to train effectively while updating only as few as 35% of the Gaussians per iteration. Combined with region-aware Gaussian management, adaptive update scheduling, and efficient visibility filtering, DecoGS produces compact and temporally coherent representations that enable real-time rendering and scalable deployment, while improving reconstruction quality. In summary, our contributions include

• a real-time streaming 3D reconstruction method that achieves superior quality with efficient online training,

• an adaptive static-dynamic decoupling module that identifies and freezes stable scene regions,

• a dynamic selection framework that limits the number of updated Gaussians per iteration, preserving temporal coherence via gradient gating and visibility filtering.

Experiments on two real-world datasets across 9 dynamic scenes validate that DecoGS’s reconstruction quality of 34.55 (N3DV) and 31.60 (MeetRoom) PSNR surpasses SOTA online and offline methods, also providing the fastest rendering with 260-261 FPS. Furthermore, we quantify the temporal coherence visible in our supplemental videos with 0.003/0.004 mTV on two scenes, +0.079/+0.099 improvement over the second best approach. DecoGS establishes a new direction for practical and resource-efficient 3D Gaussian learning from continuous video streams, with a competitive advantage in rendering quality and training time.

## 2. Related Work

## 2.1. Novel View Synthesis for Static Scenes

Synthesizing novel views of static scenes has long been a fundamental topic in computer vision and computer graphics, with early approaches [5, 7, 13, 21, 32, 52] achieving novel-view generation through view interpolation and image-based warping. With the advent of deep learning, Neural Radiance Fields (NeRF) [42] revolutionized the field by representing continuous scene radiance via coordinate-based neural networks, producing photorealistic novel views from sparse input images. A wide range of subsequent methods improved NeRF in different dimensions by accelerating training [9, 10, 15, 25, 45, 54], enabling real-time rendering [12, 19, 23, 51, 65, 77], and enhancing reconstruction quality for complex or sparse scenes [2–4, 8, 41, 43, 48, 57, 64, 69, 74, 78]. However, the original volumetric formulation of NeRF requires dense sampling and neural network evaluation along each camera ray, which introduces inherent trade-offs among speed, quality, and storage efficiency. To address these constraints, 3D Gaussian Splatting (3DGS) [29] introduced a point-based representation in which each scene element is modeled as a Gaussian primitive with learnable opacity, scale, and spherical harmonic color coefficients, enabling real-time, high-fidelity view synthesis. Building on this foundation, our work leverages 3D Gaussian representations not only for static view synthesis but also for efficient online training on continuous video streams, enabling dynamic Free-Viewpoint Video (FVV) reconstruction with minimal computational overhead.

## 2.2. Free-Viewpoint Videos for Dynamic Scenes

Extending novel view synthesis from static to dynamic scenes has become an active area of research, with early progress driven by NeRF-based dynamic representations [1, 6, 14, 16, 27, 28, 33–37, 46–48, 50, 53, 56, 58, 59, 61, 63, 73, 80]. These approaches model motion by conditioning neural fields on time or deformation networks, but remain limited by slow optimization and heavy volumetric rendering. Following 3DGS[29], several works have explored dynamic Gaussian formulations [26, 30, 38, 66, 71, 75, 76] integrating 3DGS’s real-time capabilities into temporal modeling frameworks. Recently, [22, 40, 62, 67, 68, 71] attempt to separate dynamic and static Gaussian points and introduce external models to segment foreground and back ground areas. ClipGStream [39] further scales offline reconstruction to long sequences and large-scale motion via clip-level optimization. Although these methods achieve high-fidelity reconstruction, they typically rely on offline optimization over full video sequences, making them unsuitable for applications demanding fast updates and low latency, such as FVV and immersive VR/AR streaming.

To overcome this limitation, streaming-based approaches have been proposed. Among these, earlier methods such as StreamRF [33], NeRFPlayer [53] and ReRF [60] utilize NeRF-based dynamic representations, while recent streaming approaches [24, 55, 72] utilize Gaussian representations. These methods formulate dynamic reconstruction as an incremental learning problem, updating the scene representation as new frames arrive. In particular, 3DGStream [55] models Gaussian motion between frames, significantly improving rendering speed and memory efficiency; however, it fails to represent dynamic areas with high quality. Several works [11, 17, 18, 20, 24] model the motion in a similar fashion, focusing on compactness to reduce storage cost. Among these, QUEEN [20] improves streaming efficiency by encoding temporal Gaussian updates with quantized residuals and sparse position updates, focusing on compression of changes rather than explicitly separating static and dynamic Gaussians during optimization. Furthermore, these methods still perform per-frame optimization over all Gaussians, leading to substantial computational overhead and artifacts arising from updates in static regions. Recent variants, such as IGS [72], aim to further reduce per-frame latency and improve reconstruction quality by pretraining a motion network to predict Gaussian deformation directly from streaming video. While effective, this approach requires extensive offline training prior to incremental learning, and falls short in generalization and quality of fine details. In contrast, DecoGS introduces a lightweight, adaptive formulation that updates only the dynamic areas of the scene, heavily suppressing the emergence of artifacts and improving temporal consistency.

## 3. Preliminary

3D Gaussian Splatting (3DGS) [29] represents static scenes as a collection of anisotropic 3D Gaussians (3DGs), where the color of each pixel is obtained through pointbased alpha blending. Each 3DG is parameterized by a center $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ and a covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ . For every 3D point $x \in \mathbb { R } ^ { 3 }$ , the 3DG is defined as:

$$
G ( x ; { \pmb \mu } , { \Sigma } ) = e ^ { - \frac 1 2 ( x - { \pmb \mu } ) ^ { T } { \Sigma } ^ { - 1 } ( x - { \pmb \mu } ) } ,\tag{1}
$$

where covariance matrix Σ is given by, with scale matrix s

and rotation matrix R:

$$
\Sigma = \mathbf { R } \operatorname { d i a g } ( \mathbf { s } ) \operatorname { d i a g } ( \mathbf { s } ) ^ { T } \mathbf { R } ^ { T } .\tag{2}
$$

For rendering, the 3DG is projected onto the 2D space and the Gaussians covering a pixel are sorted based on their depth. The color of the pixel c is obtained using point-based alpha blending using the color of the i-th Gaussian, $c _ { i } ,$ , opacity value $\alpha _ { i }$ computed after particle projection:

$$
\mathbf { c } = \sum _ { i = 1 } ^ { n } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) ,\tag{3}
$$

Neural Transformation Cache (NTC). Following [55], we employ NTC, a lightweight MLP-based network that predicts per-Gaussian transformation changes to capture scene dynamics. At each timestep, NTC estimates spatia and rotational residuals $( d _ { r o t } , d \mu )$ conditioned on the Gaussian’s latent attributes, enabling rapid adaptation of positions and orientations without full retraining. Unlike directly optimizing all 3DGs, NTC operates as a compact motion prior that is queried and updated online, substantially reducing computational overhead. Its outputs are later committed back to the explicit Gaussian parameters during densification, forming a bridge between neural field prediction and explicit 3D representation refinement.

## 4. Method

Given multi-view video streams as input, DecoGS constructs photo-realistic FVV streams via online training of an incremental Gaussian Splatting framework. We begin by training a static 3DGS model at the initial time step $t = 0 .$ For each subsequent timestep $t > 0 ,$ , we apply an efficient Adaptive Static-Dynamic Decoupling strategy (Sec. 4.1) to isolate motion regions in 2D image space. Using this information, we perform Dynamic Gaussian Selection (Sec. 4.2), which identifies the subset of Gaussians affected by dynamic content. The selected subset is then refined through Selective Gaussian Optimization (Sec. 4.3), while all static Gaussians remain frozen to preserve stability and reduce redundancy. Finally, we describe a Focus-Aware Loss Function (Sec. 4.4) that emphasizes dynamic areas during optimization, further improving temporal coherence and visual sharpness. Overview is provided in Fig. 2.

## 4.1. Adaptive Static-Dynamic Decoupling

At each time step t, we receive a new set of multi-view frames $\{ I _ { t } ^ { v } \}$ from cameras $v \in \{ 1 , \ldots , V \}$ . We first compute a difference mask $\tilde { M } _ { t } ^ { v }$ in the pixel space between consecutive frames $I _ { t - 1 } ^ { v }$ and $I _ { t } ^ { v }$ to identify dynamic regions. This step identifies photometrically changed regions without optical flow or learned features, relying instead on motion and appearance changes being well-approximated by per-pixel intensity differences.

![](images/d830625658c12f9254bb492ff44cfe76024c55a9bfc772dd2ce6cdb9bf56fb4e.jpg)  
Figure 2. Overview of DecoGS. Given a set of multi-view video streams, DecoGS reconstructs an FVV stream of the dynamic scene. At $t = 0 ,$ we optimize a set of initial 3D Gaussians, and for each $t > 0$ , we compute a pixel-wise difference mask to isolate dynamic regions, dilated via max-pooling to capture motion boundaries. Throughout training, we optimize the NTC for 250 steps to rotate and translate Gaussians from the previous timestep $t = T - 1 .$ . At step 150, we employ selective densification that limits the spawning of Gaussians only to the selected areas, enabling the emergence of new objects. From this step onward, we optimize the selected Gaussians jointly with NTC using a focus-aware loss that concentrates the optimization to the detected ROI.

Formally, $\tilde { M } _ { t } ^ { v }$ is computed for each view v as follows:

$$
D _ { t } ^ { v } ( x , y ) = \operatorname* { m a x } _ { i _ { c } } | I _ { t } ^ { v } ( x , y , i _ { c } ) - I _ { t - 1 } ^ { v } ( x , y , i _ { c } ) | ,\tag{4}
$$

$$
M _ { t } ^ { v } ( x , y ) = \Im [ D _ { t } ^ { v } ( x , y ) > \tau ] ,\tag{5}
$$

where $\tau$ is an intensity threshold, $i _ { c }$ is a color channel. To account for motion boundaries and slight localization errors, we spatially dilate $M _ { t } ^ { v }$ via a max-pooling operation:

$$
\tilde { M } _ { t } ^ { v } = \mathrm { M a x P o o l } ( M _ { t } ^ { v } , r ) ,\tag{6}
$$

with kernel radius $r ,$ yielding a pixel-wise selection mask used to identify dynamic Gaussians.

## 4.2. Dynamic Gaussian Selection

After detecting ROIs in 2D, we need to associate these regions with 3D Gaussians of interest. To seek a low-cost and efficient streaming procedure that does not increase the overall computation, we use camera parameters to project Gaussian centers from 3D world space to 2D image space.

Projection of 3D Gaussians. To associate Gaussians with image-space changes, we project 3D centers $\pmb { \mu } _ { i } ^ { 3 D } \in \mathbb { R } ^ { 3 }$ into camera coordinates via calibrated intrinsics/extrinsics:

$$
\pmb { \mu } _ { i } ^ { 2 D , v } = \Pi _ { v } ( \pmb { \mu } _ { i } ^ { 3 D } ) = \binom { \pmb { u } _ { i } ^ { v } } { v _ { i } ^ { v } } = \binom { f _ { x } \frac { X _ { i } ^ { v } } { Z _ { i } ^ { v } } + c _ { x } } { f _ { y } \frac { Y _ { i } ^ { v } } { Z _ { i } ^ { v } } + c _ { y } } ,\tag{7}
$$

where $( X _ { i } ^ { v } , Y _ { i } ^ { v } , Z _ { i } ^ { v } )$ are the coordinates of the Gaussian in camera space, and $( f _ { x } , f _ { y } , c _ { x } , c _ { y } )$ denote the intrinsic parameters. This projection efficiently identifies visible Gaussians with positive depth and avoids costly procedures such as 2D-to-3D motion feature lifting.

Mask-based Gaussian Selection. Given the per-view dilated masks $\tilde { M } _ { t } ^ { v }$ from Sec. 4.1, we select Gaussians whose projected centers fall inside at least $\kappa$ views’ active pixels. Let $S _ { t }$ denote the selected set:

$$
S _ { t } = \big \{ i \big | \sum _ { v = 1 } ^ { V } \Im \bigl [ \tilde { M } _ { t } ^ { v } \bigl ( \pmb { \mu } _ { i } ^ { \mathrm { 2 D } , v } \bigr ) = 1 \bigr ] \ge \kappa \big \} .\tag{8}
$$

In practice, we set $\kappa = 2$ . Additionally, we intersect this selection with per-view visibility filters to suppress updates to occluded Gaussians, yielding the final global Gaussian mask. Gradients for 3DGs outside this mask are blocked:

$$
{ \frac { \partial { \mathcal { L } } } { \partial \theta _ { i } } } = 0 , \quad { \mathrm { f o r ~ } } i \notin S _ { t } .\tag{9}
$$

## 4.3. Selective Gaussian Optimization

NTC with Gradient Gating. Following [45, 55] due to its compactness and efficiency, we extend NTC with a gradient gating mechanism. During incremental training, the model retains all Gaussians for rendering but only backpropagates gradients to $S _ { t }$ . This targeted optimization suppresses redundant updates in static areas while preserving details in dynamic ones. For each iteration:

$$
\theta _ { i } \gets \left\{ \begin{array} { l l } { \theta _ { i } - \eta \nabla _ { \theta _ { i } } \mathcal { L } , } & { \mathrm { i f ~ } i \in \mathcal { S } _ { t } , } \\ { \theta _ { i } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{10}
$$

where $\eta$ is the learning rate. This gating is implemented efficiently as an in-place tensor operation, avoiding overhead in both forward and backward passes. We observe that the gradient gating mechanism significantly reduces updates to 3D Gaussians in static regions, such as the walls and ceiling, while concentrating training updates in dynamic regions.

Optimizer State Reset. Gradient gating alone is insufficient to prevent drift: even when parameter updates are blocked, Adam’s first and second moment buffers $m _ { i } , v _ { i }$ for static Gaussians continue to accumulate stale signal from dynamic frames, corrupting future optimization steps when those Gaussians later enter $S _ { t }$ . We therefore zero these buffers after every update for all Gaussians outside $S _ { t }$ :

$$
m _ { i } \gets 0 , \quad v _ { i } \gets 0 , \quad \quad \forall i \notin S _ { t } ,\tag{11}
$$

This prohibits gradient history from dynamic frames to corrupt the optimization of static Gaussians in future steps.

Selective Densification. The selective transformation of Gaussians covers a large portion of dynamics in the scenes by managing occlusions and disappearances in subsequent timesteps. Still, this approach falls short in capturing the emergence of new objects, such as a smartphone being taken out of a pocket. It is not feasible to generate an extensive number of additional Gaussians, as this would create intractable growth. Therefore, a reliable strategy for estimating the emergence of new Gaussians is required.

[55] proposes a mechanism to capture every potential location where new objects might emerge by tracking the view-space positional gradient during the training of the NTC. We extend this by adding a selective densification process that restricts the spawned Gaussians to the ROI. First, we prune and clone only within the ROI. This limits the substantial growth of Gaussians, spawned only in the required regions. Empirically, Gaussians selected for densification never exceed 35% of the dynamic scene. For highgradient regions where new 3D Gaussians are spawned, we set a lower gradient threshold of $\tau _ { d } ~ = ~ 0 . 0 0 0 1$ . Gradient and radius accumulators used by 3DGS densification are zeroed outside. To prevent off-ROI drift, any added Gaussian whose projections fall outside all per-view ROIs is repositioned to its parent’s location before being committed to the scene. Finally, we perform a mask re-sync. Since the number of points changes after the prune-and-clone step, the external mask is resized to match the new set of 3D Gaussians.

## 4.4. Focus-aware Loss Function

Although the centers of the selected Gaussians lie within the ROI, updating the Gaussian can affect regions outside these boundaries, depending on its size. Therefore, we do not strictly limit the optimization to this region. Instead, to emphasize dynamic regions, we introduce a focus-weighted reconstruction loss that gives higher weight to pixels within the ROI. The overall loss combines a global $\ell _ { 1 }$ term, a focus-weighted $\ell _ { 1 }$ term, and a focus-only SSIM term:

$$
\begin{array} { r } { \mathcal { L } = ( 1 - \lambda _ { \mathrm { s s i m } } ) \left[ \beta _ { 1 } \mathcal { L } _ { 1 } + \beta _ { 2 } \mathcal { L } _ { 1 } ^ { \mathrm { f o c u s } } \right] + \lambda _ { \mathrm { s s i m } } \mathcal { L } _ { \mathrm { s s i m } } ^ { \mathrm { f o c u s } } , } \end{array}\tag{12}
$$

$$
\mathcal { L } _ { 1 } ^ { \mathrm { f o c u s } } = \| \tilde { M } _ { t } ^ { v } \odot ( I _ { t } ^ { v } - \hat { I } _ { t } ^ { v } ) \| _ { 1 } ,\tag{13}
$$

$$
\mathcal { L } _ { \mathrm { s s i m } } ^ { \mathrm { f o c u s } } = 1 - \mathrm { S S I M } ( \tilde { M } _ { t } ^ { v } \odot ( I _ { t } ^ { v } - \hat { I } _ { t } ^ { v } ) ) ,\tag{14}
$$

and $\beta _ { 1 } , \beta _ { 2 }$ balance $\ell _ { 1 }$ terms. SSIM is applied exclusively to the focus region, since gradients outside the ROI are already

suppressed by the gradient-gating mechanism.

## 5. Experiments

## 5.1. Setup

Datasets. We perform our experiments on two real-world dynamic scene datasets, i.e. N3DV dataset [35] and Meet-Room dataset [33]. Each dataset reserves 1 camera view as a held-out test set, while the remaining views are used for training. N3DV dataset [35] is captured via 21 multiview cameras and includes dynamic scenes recorded at a resolution of $2 7 0 4 \times 2 0 2 8$ at 30 FPS. For a fair comparison with methods that pre-train on sequences from the same dataset, we report results on the N3DV test sequences over 300 frames. The MeetRoom dataset [33] is captured via 13 multi-view Azure Kinect cameras and includes dynamic scenes recorded at a resolution of 1280 × 720 and 30 FPS. This dataset presents more challenging scenarios with rapid changes and motion blur.

Implementation Details. We utilize 3DGS [29] and NTC using InstantNGP [44, 45]. For the training of the initial 3DGS, we adapt the learning rates from the N3DV dataset defaults and use the same parameters for the MeetRoom dataset. To suppress any noise that might occur during the transformation of the initial static areas, we train the NTC at 500 iterations only for the initial frame. For the remainder of the video sequence, we train NTC for 250 iterations and limit the Gaussian optimization to the last 100 iterations after selective densification. Gaussians corresponding to static areas remain unchanged throughout training. Therefore, we perform a full training of the original 3DGS. All experiments are conducted on an NVIDIA RTX 4090 GPU. We provide further details in Supp. Sec. B.

## 5.2. Baselines

We compare our approach to both state-of-the-art online and offline training methods for dynamic scene reconstruction. Offline methods [16, 38, 66, 68, 71, 75] use Gaussian primitives or hex-plane representations for the entire scene, but require the complete video sequence before training begins, an assumption incompatible with live streaming. In contrast, online methods [11, 17, 18, 20, 24, 31, 33, 55, 72] employ per-frame optimization, suitable for streaming by design. Among these, IGS [72] requires heavy pre-training prior to per-frame optimization on 4 scenes of N3DV. Therefore, we additionally provide mean PSNR across the test sequences “cut beef” and “sear steak” for a fair comparison. Similarly, we compare MoRGS [31] only on N3DV via its self-reported results, as no public implementation is available (Supp. Sec. C.1).

Table 1. Comparison on N3DV of offline and online 3DGS methods at $1 3 5 2 \times 1 0 1 4 .$ Best and second best are highlighted. Pretrain-free (✓/✗) indicates whether a method operates without any pretraining beyond initialization. (†) marks results reproduced by us with the official code in the same experimental environment.
<table><tr><td>Method</td><td>PSNR ↑ Test (dB)</td><td>(dB)</td><td></td><td>(FPS)</td><td>PSNR↑ SSIM↑ Render↑ Pre-train-free Per-frame↓ (√IX)</td><td>Train (s)</td></tr><tr><td>Offline Training</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>K-Planes [16]</td><td>32.17</td><td>31.63</td><td></td><td>0.15</td><td>x</td><td></td></tr><tr><td>Realtime-4DGS [75]</td><td>33.94</td><td>32.01</td><td>0.972</td><td>114</td><td>x</td><td></td></tr><tr><td>4DGS [66]</td><td>32.70</td><td>31.15</td><td>0.950</td><td>30</td><td>X</td><td>7.8</td></tr><tr><td>SpaceTime-GS [38]</td><td>33.71</td><td>32.05</td><td>0.972</td><td>140</td><td>x</td><td></td></tr><tr><td>Saro-GS [71]</td><td>33.90</td><td>32.15</td><td></td><td>40</td><td>x</td><td></td></tr><tr><td>Swift4D [68]</td><td></td><td>32.23</td><td>0.972</td><td>125</td><td>X</td><td>5</td></tr><tr><td colspan="7">Online Training</td></tr><tr><td>StreamRF [33]</td><td>32.09</td><td>30.68</td><td></td><td>8.3</td><td></td><td>15</td></tr><tr><td>4DGC [24]</td><td>33.32</td><td>31.58</td><td></td><td>168</td><td></td><td>50</td></tr><tr><td>IGS [72]</td><td>34.15</td><td></td><td></td><td>204</td><td>X (192 hours)</td><td>3.35</td></tr><tr><td>HiCoM† [18]</td><td>33.22</td><td>32.08</td><td>0.953</td><td>255</td><td></td><td>6.6</td></tr><tr><td>ReCon-GS [17]</td><td>33.92</td><td>32.66</td><td>0.956</td><td>250</td><td>1</td><td>6.4</td></tr><tr><td>ComGS [11]</td><td>33.64</td><td>32.12</td><td></td><td>147</td><td></td><td>43</td></tr><tr><td>QUEEN [20]</td><td>33.72</td><td>32.01</td><td>0.946</td><td>248</td><td></td><td>7.9</td></tr><tr><td>MoRGS [31]</td><td></td><td>32.53</td><td>0.950</td><td>200</td><td></td><td></td></tr><tr><td>3DGStream† [55]</td><td>33.03</td><td>31.35</td><td>0.948</td><td>261</td><td></td><td>8.4</td></tr><tr><td>Ours</td><td>34.55</td><td>33.36</td><td>0.956</td><td>261</td><td></td><td>7.8</td></tr></table>

Table 2. Comparison on MeetRoom of streaming methods at $1 2 8 0 \times 7 2 0$ . Train denotes seconds at each iteration during incremental training Best and second best are highlighted.
<table><tr><td rowspan=1 colspan=5>PSNR ↑ SSIM↑  Train ↓  Render ↑Method(dB)                (s)     (FPS)</td></tr><tr><td rowspan=1 colspan=1>4DGC [24]</td><td rowspan=1 colspan=1>28.08</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>49.8</td><td rowspan=1 colspan=1>213</td></tr><tr><td rowspan=1 colspan=1>HiCoM† [18]</td><td rowspan=1 colspan=1>29.57</td><td rowspan=1 colspan=1>0.944</td><td rowspan=1 colspan=1>3.9</td><td rowspan=1 colspan=1>236</td></tr><tr><td rowspan=1 colspan=1>ReCon-GS [17]</td><td rowspan=1 colspan=1>30.84</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.9</td><td rowspan=1 colspan=1>256</td></tr><tr><td rowspan=1 colspan=1>ComGS [11]</td><td rowspan=1 colspan=1>31.49</td><td rowspan=1 colspan=1>0.955</td><td rowspan=1 colspan=1>28.3</td><td rowspan=1 colspan=1>98</td></tr><tr><td rowspan=1 colspan=1>3DGStream† [55]</td><td rowspan=1 colspan=1>30.79</td><td rowspan=1 colspan=1>0.950</td><td rowspan=1 colspan=1>4.9</td><td rowspan=1 colspan=1>260</td></tr><tr><td rowspan=1 colspan=1>IGS [72]</td><td rowspan=1 colspan=1>30.13</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>252</td></tr><tr><td rowspan=1 colspan=1>DecoGS (Ours)</td><td rowspan=1 colspan=1>31.60</td><td rowspan=1 colspan=1>0.955</td><td rowspan=1 colspan=1>4.3</td><td rowspan=1 colspan=1>260</td></tr></table>

Table 3. Temporal consistency in static regions. Following regions of [79], we report PSNR, SSIM and mTV (masked Total Variation) with standard deviations.
<table><tr><td rowspan="2">Method</td><td colspan="2">Coffee Martini</td><td colspan="2">Flame Steak</td></tr><tr><td>PSNR ↑</td><td> $\mathrm { m T V } _ { \times 1 0 0 } \downarrow$ </td><td>PSNR ↑</td><td> $\mathrm { m T V } _ { \times 1 0 0 } \downarrow$ </td></tr><tr><td>3DGStream</td><td> $\overline { { 2 7 . 7 5 _ { \pm 0 . 3 6 } } }$ </td><td> $0 . 2 1 3 _ { \pm 0 . 0 4 0 }$ </td><td> $\overline { { 3 3 . 4 7 _ { \pm 1 . 0 1 } } }$ </td><td> $\overline { { 0 . 1 4 0 _ { \pm 0 . 0 2 6 } } }$ </td></tr><tr><td>3DGStream w. [79]</td><td>29.54±0.06</td><td> $0 . 0 8 2 _ { \pm 0 . 0 7 0 }$ </td><td> $3 4 . 8 6 _ { \pm 0 . 3 7 }$ </td><td> $0 . 1 0 3 { \scriptstyle \pm 0 . 0 4 6 }$ </td></tr><tr><td>DecoGS (Ours)</td><td>31.15±0.24</td><td>0.003±0.003</td><td> $3 5 . 5 4 _ { \pm 1 . 0 3 }$ </td><td> $0 . 0 0 4 { \scriptstyle \pm 0 . 0 1 3 }$ </td></tr></table>

![](images/f90c808055198dd6a5259013dfee5916856bd1805fa30a0122b9b3fc0a13655b.jpg)  
IGS

![](images/a9e6724bbb19ff478c447751cf71dd9bcbfc83ec0f5f84213ad81a5ead4e7e4c.jpg)  
3DGStream

![](images/9d2dc21ff4fb79b213f20afcba16c20c17abf4d26d910ba53334089af1dca130.jpg)  
DecoGS

![](images/3250ab63ce37efc61d17a2c006ada9f59df09ae3d23aa128a4f768277a73f544.jpg)  
GT  
Figure 3. Qualitative comparison across streaming baselines IGS [72] and 3DGStream [55] on sear steak scene of N3DV dataset and the discussion scene of MeetRoom dataset. Our method achieves high-quality reconstruction both in static and dynamic regions.

## 5.3. Comparisons

Quantitative Analysis. We conduct quantitative comparisons to benchmark DecoGS on N3DV and MeetRoom datasets. In Tab. 1, we present peak signal-to-noise ratio (PSNR) for test and all sequences, rendering speed, whether pre-training is required prior to streaming, and per-frame training time. To highlight the superior quality of DecoGS, we compare it with both offline training methods that require the entire video sequence and online methods that target video streaming. For each scene, the metrics are computed as averages over 300 frames of the sequence. In addition, we provide per-scene results for all N3DV sequences in Supp. Sec. C.1. While maintaining comparable rendering speed to 3DGStream, our method achieves higher rendering quality with a +2.01 dB increase in PSNR over all scenes. Notably, each baseline concentrates on a different point of the design space. IGS [72] attains the lowest perframe training time by amortizing motion estimation into a large pre-trained network, trading it off against 192 GPUhours of pre-training before streaming can start; this cost does not appear in DecoGS, a property shared by the remaining baselines. Despite requiring no pre-training, we still outperform IGS by +0.40 dB in PSNR over the two test sequences. Among recently introduced online methods, 4DGC [24], HiCoM [18], ComGS [11], and ReCon-GS [17] mainly concentrate on compactness and storage, and therefore trade off either per-frame training time (50 and 43 seconds for 4DGC and ComGS) or reconstruction quality (33.22 and 33.92 dB for HiCoM and ReCon-GS). DecoGS instead concentrates on reconstruction quality and rendering speed, outperforming all of them on both PSNR metrics, as well as QUEEN [20] and the flow-guided MoRGS [31], while providing the fastest rendering at 261 FPS. Our method achieves state-of-the-art rendering quality for N3DV compared with both streaming and offline methods.

To demonstrate the generalization of our method, we additionally conduct experiments on the MeetRoom dataset of StreamRF [33] and report quantitative comparisons across FVV streaming baselines. As shown in Tab. 2, our method achieves state-of-the-art rendering quality compared to the baselines while maintaining fast online training and realtime rendering capabilities. Compared with 3DGStream, our method preserves on-par training time while achieving a +0.81 dB increase in PSNR, demonstrating our effectiveness in eliminating 3D Gaussians corresponding to static areas. Among the recent baselines, ComGS [11] is the closest competitor at 31.49 dB; however, its focus on compactness comes at the cost of 28.3 seconds of per-frame training and 98 FPS rendering, whereas DecoGS surpasses it with 31.60 dB while training in 4.3 seconds per frame and rendering at 260 FPS. To validate our claims, we also report and compare PSNRs for the dynamic regions in Supp. Sec. C.1 for the MeetRoom scenes.

Beyond global PSNR, we evaluate temporal stability in static regions using masked Total Variation (mTV) [79], which measures per-pixel intensity change across consecutive frames. Tab. 3 reveals the most striking result: DecoGS achieves near-zero mTV (0.003 and 0.004) on the “coffee martini” and “flame steak” scenes, reducing temporal flicker up to ∼ 70× over 3DGStream and outperforming the dedicated stabilization of [79] without any additional post-processing step. This near-zero flicker is a direct consequence of gradient gating: static Gaussians are never updated, so they cannot drift. Per-scene mTV comparisons across the full N3DV dataset are provided in Supp. Sec. C.1.

Qualitative Analysis. We present qualitative comparisons across representative dynamic scenes from the N3DV and MeetRoom datasets. As in Fig. 3, our method consistently yields sharper and more temporally coherent reconstructions. In highly dynamic regions (hand motion, facial expression, and smoke from the steak) DecoGS captures fine appearance variations while keeping background consistency. Static objects, such as the snowman on the plate, remain artifact-free precisely because their Gaussians are never updated. In contrast, 3DGStream [55] exhibits noticeable ghosting and temporal flicker due to frame-wise optimization over the entire Gaussian set, while IGS [72] suffers from poor generalization on the discussion scene of MeetRoom due to pre-training on a limited subset of N3DV.

The proposed Adaptive Static-Dynamic Decoupling plays a key role in improving spatial clarity: dynamic regions (e.g., moving limbs or objects) are reconstructed with enhanced detail, while static backgrounds remain stable. Moreover, our Selective Gaussian Optimization suppresses redundant updates in non-changing areas, preventing the cumulative artifacts observed in other online methods. As in the “sear steak” and “discussion” scenes, DecoGS maintains fine surface texture and correct object geometry under large inter-frame motion, highlighting its ability to localize and track motion-aware Gaussians effectively. Fig. 4 further illustrates temporal stability through space-time scanlines over static regions: 3DGStream produces a noisy spatiotemporal profile with flickering artifacts [79], whereas DecoGS maintains a clean, consistent profile throughout the sequence. Overall, DecoGS achieves photorealistic rendering quality and temporal smoothness, demonstrating clear advantages for real-time FVV streaming.

## 5.4. Ablation Study

We conduct an ablation study to analyze the impact of individual components in DecoGS. We evaluate our method across different architectural and training configurations on the N3DV dataset, as shown in Tab. 4.

<table><tr><td>Select</td><td>Focus</td><td>Gate</td><td>PSNR (dB) ↑</td></tr><tr><td></td><td>一</td><td>一</td><td>28.61</td></tr><tr><td>√</td><td></td><td>一</td><td>30.81</td></tr><tr><td>√</td><td>√</td><td></td><td>30.83</td></tr><tr><td>√</td><td>√</td><td>√</td><td>30.92</td></tr></table>

Table 4. Component impact on flame salmon scene of N3DV.
<table><tr><td>Setting</td><td>PSNR (dB) ↑</td></tr><tr><td>3DGStream (30 FPS)</td><td>33.39</td></tr><tr><td>No sampling (30 FPS)</td><td>34.40</td></tr><tr><td>Sampling rate = 5 (6 FPS)</td><td>33.68</td></tr><tr><td>Sampling rate = 10 (3 FPS)</td><td>33.25</td></tr></table>

Table 5. Effect of temporal sampling on sear steak of N3DV.

Dynamic Gaussian Selection. We begin by evaluating the effect of our dynamic 3D Gaussian selection module (Sec. 4.2), which identifies and updates only a sparse set of Gaussians whose 2D projections fall inside view-specific ROIs. Without this selection, the system defaults to dense optimization over all Gaussians, leading to slower updates and degradation of static regions. Enabling selection alone improves PSNR from 28.61dB to 30.81dB, validating that localized updates offer a better tradeoff between adaptation and consistency, and confirming that the key bottleneck in prior methods is indiscriminate optimization, not their choice of motion representation or loss function.

![](images/8624268675a5ee12699368b3a4ce31dc5525a1d3685d2a815a410be7dddb892e.jpg)  
Figure 4. Scanlines across MeetRoom and N3DV datasets. Each image shows a 1 pixel line over the static area across the video.

Focus-Aware Loss. We introduce a dynamic regionweighted loss in Sec. 4.4 that increases the contribution of ROI pixels during training. This modification marginally improves the PSNR to 30.83 dB and yields visibly sharper edges in motion regions, where sub-pixel errors are perceptually significant. Qualitative gains are pronounced in motion-rich sequences (e.g., flame in “flame steak”), where standard loss formulations tend to underfit faint boundaries.

Gradient Gating and Static Suppression. When combined with gradient gating in Sec. 4.2, which blocks updates for Gaussians outside of the dynamic region, PSNR further increases to 30.92dB. This configuration enforces strong background stability while focusing learning on changing content. Without gating, we observe subtle ghosting and bleeding in static areas, particularly on walls and stationary surfaces, that accumulate over time. These results confirm the importance of isolating dynamic regions both during selection and optimization.

Sensitivity to Frame Rate. While in pre-recorded captures all frames are available, in real-world streaming scenarios, frame drops are possible. This requires robustness to varying sample rates. To test this, we study the impact of training frame rate (Tab. 5). As it decreases from 30 FPS to 6 FPS and 3 FPS, the reconstruction quality degrades from 34.40 dB to 33.68 dB and 33.25 dB respectively. Even at 6 FPS, DecoGS (33.68 dB) outperforms 3DGStream (33.39 dB) on the same scene, while the 3 FPS variant remains on par, demonstrating that DecoGS is robust to sparse frame-rates without significant quality loss.

Mask Parameter Sensitivity. The difference mask is governed by two parameters: intensity threshold τ and dilation radius r. We sweep τ and r on the “discussion” scene and report mean PSNR in Fig. 5. The method is robust around τ=25, r=48, which we adopt as the default. A too-conservative mask (r=24) under-covers motion boundaries and consistently underperforms, while over-dilation (r=96) introduces marginal noise. Notably, performance curve is relatively flat across the tested range, confirming that DecoGS requires no per-scene hyperparameter tuning, which is a practical advantage over methods that use scenespecific optical flow or segmentation thresholds.

![](images/f6983618251dd76b99239458e2d629c3fb51ec57b2b1c9ccd4b6419eb387c350.jpg)  
Figure 5. Sensitivity of Stage 2 PSNR to intensity threshold τ and dilation radius r on the discussion scene. DecoGS is robust around τ=25, r=48 (our default).

## 5.5. Limitations and Future Work

DecoGS opens several promising avenues for future work. Current change detection can be extended with semantic or foundation model cues to enable intent-aware dynamic region detection that distinguishes transient objects from persistent background without heuristics. Coupling Gaussian representations with language features would further unlock semantic FVV streaming, enabling querying and editing directly in Gaussian space. Finally, all existing online FVV methods, including DecoGS, assume fixed, synchronized, and calibrated cameras; robustness to camera drift and exposure changes remains open for the field.

## 6. Conclusion

DecoGS demonstrates a fundamental finding: in streaming 3D Gaussian reconstruction, doing less is more. By identifying that fewer than 35% of Gaussians require updating at any given frame, and building a targeted pipeline around this sparsity, DecoGS achieves state-of-the-art quality with 70× lower temporal flicker, without pretraining, without optical flow, and without increasing per-frame training time. By isolating regions that change over time through fast image-space differencing and updating only a sparse set of 3D Gaussians per frame, DecoGS keeps static regions frozen while preserving temporal coherence and background sharpness. Our focus-aware loss further enhances detail in dynamic areas, and selective densification adapts to scene changes.

Experimental results on multiple real-world dynamic scene datasets demonstrate that DecoGS outperforms prior streaming and offline methods in both quality and efficiency. Notably, our approach achieves higher reconstruction quality while maintaining comparable training and rendering speeds, without large-scale pretraining. More broadly, DecoGS suggests that selective optimization may be a general principle for efficient incremental scene learning, with applicability beyond Gaussian splatting to any representation that accumulates gradient state over time.

## 7. Acknowledgments

We thank Justin Solomon and the team of volunteers for organizing the MIT Summer Geometry Initiative (SGI) and for fostering the collaboration that led to this work.

## References

[1] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil Kim. Hyperreel: High-fidelity 6-dof video with rayconditioned sampling. In CVPR, 2023. 2

[2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, 2021. 2

[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, 2022.

[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. In ICCV, 2023. 2

[5] Chris Buehler, Michael Bosse, Leonard McMillan, Steven Gortler, and Michael Cohen. Unstructured lumigraph rendering. In Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques, 2001. 2

[6] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In CVPR, 2023. 2

[7] Jin-Xiang Chai, Xin Tong, Shing-Chow Chan, and Heung-Yeung Shum. Plenoptic sampling. In Proceedings ofthe 27th Annual Conference on Computer Graphics and Interactive Techniques, 2000. 2

[8] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In ICCV, 2021. 2

[9] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In ECCV, 2022. 2

[10] Anpei Chen, Zexiang Xu, Xinyue Wei, Siyu Tang, Hao Su, and Andreas Geiger. Dictionary fields: Learning a neural basis decomposition. ACM TOG, 2023. 2

[11] Jiacong Chen, Qingyu Mao, Youneng Bao, Xiandong Meng, Fanyang Meng, Ronggang Wang, and Yongsheng Liang. Motion matters: Compact gaussian streaming for freeviewpoint video reconstruction. NeurIPS, 2025. 1, 3, 5, 6, 7

[12] Zhiqin Chen, Thomas Funkhouser, Peter Hedman, and Andrea Tagliasacchi. Mobilenerf: Exploiting the polygon rasterization pipeline for efficient neural field rendering on mobile architectures. In CVPR, 2023. 2

[13] Abe Davis, Marc Levoy, and Fredo Durand. Unstructured light fields. In Computer Graphics Forum, 2012. 2

[14] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia, 2022. 1, 2

[15] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2022. 2

[16] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In CVPR, 2023. 1, 2, 5, 6

[17] Jiaye Fu, Qiankun Gao, Chengxiang Wen, Yanmin Wu, Siwei Ma, Jiaqi Zhang, and Jian Zhang. Recon-gs: Continuum-preserved gaussian streaming for fast and com pact reconstruction of dynamic scenes. NeurIPS, 2025. 1, 3, 5, 6, 7

[18] Qiankun Gao, Jiarui Meng, Chengxiang Wen, Jie Chen, and Jian Zhang. Hicom: Hierarchical coherent motion for dynamic streamable scenes with 3d gaussian splatting. NeurIPS, 2024. 1, 3, 5, 6, 7

[19] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In ICCV, 2021. 2

[20] Sharath Girish, Tianye Li, Amrita Mazumdar, Abhinav Shrivastava, Shalini De Mello, et al. Queen: Quantized efficient encoding of dynamic gaussians for streaming free-viewpoint videos. NeurIPS, 2024. 1, 3, 5, 6, 7

[21] Steven J Gortler, Radek Grzeszczuk, Richard Szeliski, and Michael F Cohen. The lumigraph. In Seminal Graphics Pa pers: Pushing the Boundaries, Volume 2, pages 453–464. 2023. 2

[22] Bing He, Yunuo Chen, Guo Lu, Qi Wang, Qunshan Gu, Rong Xie, Li Song, and Wenjun Zhang. S4d: Streaming 4d real-world reconstruction with gaussians and 3d control points. arXiv preprint, 2024. 3

[23] Peter Hedman, Pratul P Srinivasan, Ben Mildenhall, Jonathan T Barron, and Paul Debevec. Baking neural radiance fields for real-time view synthesis. In ICCV, 2021. 2

[24] Qiang Hu, Zihan Zheng, Houqiang Zhong, Sihua Fu, Li Song, Xiaoyun Zhang, Guangtao Zhai, and Yanfeng Wang. 4dgc: Rate-aware 4d gaussian compression for efficient streamable free-viewpoint video. In CVPR, 2025. 1, 2, 3, 5, 6, 7

[25] Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen Ma. Tri-miprf: Tri-mip representation for efficient anti-aliasing neural radiance fields. In ICCV, 2023. 2

[26] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In CVPR, 2024. 2

[27] Yuheng Jiang, Zhehao Shen, Yu Hong, Chengcheng Guo, Yize Wu, Yingliang Zhang, Jingyi Yu, and Lan Xu. Robust dual gaussian splatting for immersive human-centric volumetric videos. ACM TOG, 2024. 2

[28] Yuheng Jiang, Zhehao Shen, Penghao Wang, Zhuo Su, Yu Hong, Yingliang Zhang, Jingyi Yu, and Lan Xu. Hifi4g: High-fidelity human performance rendering via compact gaussian splatting. In CVPR, 2024. 2

[29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG, 2023. 1, 2, 3, 5

[30] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting. In ECCV, 2024. 2

[31] Wonjoon Lee, Sungmin Woo, Donghyeong Kim, Jungho Lee, Sangheon Park, and Sangyoun Lee. Morgs: Efficient per-gaussian motion reasoning for streamable dynamic 3d scenes. CVPR, 2026. 5, 6, 7

[32] Marc Levoy and Pat Hanrahan. Light field rendering. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 441–452. 2023. 2

[33] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and Ping Tan. Streaming radiance fields for 3d video synthesis. NeurIPS, 2022. 1, 2, 3, 5, 6, 7

[34] Ruilong Li, Julian Tanke, Minh Vo, Michael Zollhofer,¨ Jurgen Gall, Angjoo Kanazawa, and Christoph Lassner.¨ Tava: Template-free animatable volumetric actors. In ECCV, 2022.

[35] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In CVPR, 2022. 1, 5

[36] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In CVPR, 2021.

[37] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. Dynibar: Neural dynamic image-based rendering. In CVPR, 2023. 1, 2

[38] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In CVPR, 2024. 2, 5, 6

[39] Jie Liang, Jiahao Wu, Chao Wang, Jiayu Yang, Xiaoyun Zheng, Kaiqiang Xiong, Zhanke Wang, Jinbo Yan, Feng Gao, and Ronggang Wang. Clipgstream: Clip-stream gaussian splatting for any length and any motion multi-view dynamic scene reconstruction. CVPR, 2026. 3

[40] Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-Phuoc, Douglas Lanman, James Tompkin, and Lei Xiao. Gaufre: Gaussian deformation fields for real-time dynamic novel view synthesis. In WACV, 2025. 3

[41] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In CVPR, 2021. 2

[42] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 2021. 1, 2

[43] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P Srinivasan, and Jonathan T Barron. Nerf in the dark: High dynamic range view synthesis from noisy raw images. In CVPR, 2022. 2

[44] Thomas Muller. tiny-cuda-nn, 2021.¨ 5

[45] Thomas Muller, Alex Evans, Christoph Schied, and Alexan-¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 2022. 2, 4, 5

[46] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In ICCV, 2021. 1, 2

[47] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: a higher dimensional representation for topologically varying neural radiance fields. ACM TOG, 2021.

[48] Sungheon Park, Minjung Son, Seokhwan Jang, Young Chun Ahn, Ji-Yeon Kim, and Nahyup Kang. Temporal interpolation is all you need for dynamic neural radiance fields. In CVPR, 2023. 2

[49] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In CVPR, 2021. 1

[50] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In CVPR, 2021. 2

[51] Christian Reiser, Rick Szeliski, Dor Verbin, Pratul Srinivasan, Ben Mildenhall, Andreas Geiger, Jon Barron, and Peter Hedman. Merf: Memory-efficient radiance fields for realtime view synthesis in unbounded scenes. ACM TOG, 2023. 2

[52] Heung-Yeung Shum and Li-Wei He. Rendering with con centric mosaics. In Proceedings of the 26th Annual Conference on Computer Graphics and Interactive Techniques, pages 299–306, 1999. 2

[53] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerf player: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE TVCG, 2023. 2, 3

[54] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In CVPR, 2022. 2

[55] Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei Zhao, and Wei Xing. 3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic freeviewpoint videos. In CVPR, 2024. 1, 2, 3, 4, 5, 6, 7

[56] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhofer, Christoph Lassner, and Christian Theobalt. Non-¨ rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In ICCV, 2021. 2

[57] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. IEEE TPAMI, 2024. 2

[58] Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei Song, and Huaping Liu. Mixed neural voxels for fast multi view video synthesis. In ICCV, 2023. 2

[59] Liao Wang, Qiang Hu, Qihan He, Ziyu Wang, Jingyi Yu, Tinne Tuytelaars, Lan Xu, and Minye Wu. Neural residual radiance fields for streamably free-viewpoint videos. In CVPR, 2023. 1, 2

[60] Liao Wang, Qiang Hu, Qihan He, Ziyu Wang, Jingyi Yu, Tinne Tuytelaars, Lan Xu, and Minye Wu. Neural residual radiance fields for streamably free-viewpoint videos. In CVPR, 2023. 3

[61] Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski, and Noah Snavely. Tracking everything everywhere all at once. In ICCV, 2023. 2

[62] Rui Wang, Quentin Lohmeyer, Mirko Meboldt, and Siyu Tang. Degauss: Dynamic-static decomposition with gaussian splatting for distractor-free 3d reconstruction. In CVPR, 2025. 2, 3

[63] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In CVPR, 2022. 2

[64] Felix Wimbauer, Nan Yang, Christian Rupprecht, and Daniel Cremers. Behind the scenes: Density fields for single view reconstruction. In CVPR, 2023. 2

[65] Suttisak Wizadwongsa, Pakkapon Phongthawee, Jiraphon Yenphraphai, and Supasorn Suwajanakorn. Nex: Real-time view synthesis with neural basis expansion. In CVPR, 2021. 2

[66] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, 2024. 2, 5, 6

[67] Jiahao Wu, Rui Peng, Jianbo Jiao, Jiayu Yang, Luyang Tang, Kaiqiang Xiong, Jie Liang, Jinbo Yan, Runling Liu, and Ronggang Wang. Localdygs: Multi-view global dynamic scene modeling via adaptive local implicit feature decoupling. In ICCV, 2025. 3

[68] Jiahao Wu, Rui Peng, Zhiyan Wang, Lu Xiao, Luyang Tang, Jinbo Yan, Kaiqiang Xiong, and Ronggang Wang. Swift4d: Adaptive divide-and-conquer gaussian splatting for compact and efficient reconstruction of dynamic scene. ICLR, 2025. 2, 3, 5, 6

[69] Jamie Wynn and Daniyar Turmukhambetov. Diffusionerf: Regularizing neural radiance fields with denoising diffusion models. In CVPR, 2023. 2

[70] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. Space-time neural irradiance fields for free-viewpoint video. In CVPR, 2021. 1

[71] Jinbo Yan, Rui Peng, Luyang Tang, and Ronggang Wang. 4d gaussian splatting with scale-aware residual field and adaptive optimization for real-time rendering of temporally com plex dynamic scenes. In ACMMM, 2024. 2, 3, 5, 6

[72] Jinbo Yan, Rui Peng, Zhiyan Wang, Luyang Tang, Jiayu Yang, Jie Liang, Jiahao Wu, and Ronggang Wang. Instant gaussian stream: Fast and generalizable streaming of dynamic scene reconstruction via gaussian splatting. In CVPR, 2025. 1, 2, 3, 5, 6, 7

[73] Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ramanan, Andrea Vedaldi, and Hanbyul Joo. Banmo: Building animatable 3d neural models from many casual videos. In CVPR, 2022. 2

[74] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Im proving few-shot neural rendering with free frequency regularization. In CVPR, 2023. 2

[75] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Realtime photorealistic dynamic scene representation and render ing with 4d gaussian splatting. arXiv preprint, 2023. 2, 5, 6

[76] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high fidelity monocular dynamic scene reconstruction. In CVPR, 2024. 2

[77] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In ICCV, 2021. 2

[78] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In CVPR, 2021. 2

[79] Youngsik Yun, Jeongmin Bae, Hyunseung Son, Seoha Kim, Hahyun Lee, Gun Bang, and Youngjung Uh. Compensating spatiotemporally inconsistent observations for online dy namic 3d gaussian splatting. In SIGGRAPH, 2025. 6, 7

[80] Fuqiang Zhao, Wei Yang, Jiakai Zhang, Pei Lin, Yingliang Zhang, Jingyi Yu, and Lan Xu. Humannerf: Efficiently gen erated human radiance field from sparse inputs. In CVPR, 2022. 2