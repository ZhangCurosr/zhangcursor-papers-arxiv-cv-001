# FaceSnap: Real-Time Personalized Lightstage Facial Performance Capture

Rukhshanda Hussain<sup>1,2</sup> Noe Artru´ <sup>1,2</sup> Emeline Got<sup>2</sup>\* Luiz Gustavo Hafemann<sup>2</sup>\* Alexandre Messier<sup>2</sup> Brandon Dearlove<sup>2</sup> Rafael M. O. Cruz<sup>1</sup> Abdallah Dib<sup>2</sup> Eric Granger<sup>1</sup>

<sup>1</sup> Ecole de Technologie Sup <sup>´</sup> erieure ´ <sup>2</sup> Ubisoft La Forge

![](images/a31c54c91eb34a28fef7ceffc06d0d29ce12dee4e628bf2a3d21bebbb4156e94.jpg)  
Figure 1. FaceSnap produces high-fidelity geometry and appearance with topologically consistent results. Left: A one-time multi-view optimization builds a personalized subject model. Right: The personalized model enables real-time facial performance capture of new sequences from a single camera.

## Abstract

Lightstage facial capture produces production-quality digital humans, but it is resource and labor-intensive. Multicamera setups, hours of computation, and massive data storage create bottlenecks that hinder iterative workflows. This paper introduces FaceSnap, an end-to-endframework that streamlines capture via a two-stage approach. First, a one-time multi-view optimization from a range-of-motion sequence builds a personalized model encoding both geometry and expression-dependent appearance. This model then enables high-fidelity real-time facial performance capture from a single monocular lightstage camera, with no further multi-view capture required. FaceSnap jointly estimates geometry and dynamic 4K texture at 83 fps. The 4K texture is produced by a novel personalized residual upscaler that recovers subject-specific high-frequency detail, which generic upscalersfail to capture. FaceSnap achieves

geometric accuracy competitive with full per-frame multiview optimization while outperforming feed-forward methods trained on production-quality 3D data, all from a single camera view. Finally, we introduce Multi4D, a public benchmark for evaluating 4D facial reconstruction methods in lightstage environments, enabling topology-invariant geometric comparison across methods.

## 1. Introduction

Modern game engines and high-end visual effects pipelines increasingly demand digital human characters of unprecedented realism. To achieve this, lightstages [17], equipped with dozens of synchronized cameras and controllable illumination, have become the standard for capturing facial geometry and reflectance. While lightstages capture highly detailed data, the associated workflows remain resource-intensive and time-consuming: camera and light calibration require several hours, photogrammetric processing of hundreds of high-resolution images calls for days of computation and terabytes of storage, and manual postprocessing is still required to register, clean, and retopologize meshes into production-ready assets. Prior efforts to address these challenges fall into two main categories: generalized and subject-specific approaches. Generalized methods aim to work across individuals without per-subject optimization. This includes both lightstage-trained models, such as [8, 42, 44], which depend on extensive lightstage datasets and incur high capture costs, and ”in-the-wild” approaches [3, 15, 16, 18, 25, 28, 50, 52, 55, 57, 61] that leverage publicly available monocular images to avoid specialized hardware. The former requires extensive capture sessions in the lightstage, while the latter simplifies capture, but still fails to reproduce the fine geometrical details and accurate reflectance needed for production-quality results.

In contrast, subject-specific methods [30, 51, 53, 54, 59] approach facial reconstruction by optimizing for individual subjects, producing high-fidelity results without requiring extensive multi-subject datasets. Among these, Gaussian splatting-based methods [36] provide visually impressive renderings but produce assets incompatible with mesh based production pipelines. While some approaches attempt to extract registered, engine-friendly meshes from these splatting volumes [43], they remain offline procedures that require the full lightstage for every capture, along with per-session optimization. As a result, a practical bottleneck persists: capturing each new facial performance requires repeating the full multi-view capture and processing pipeline. Some methods reduce this repeated cost by learning a personalized, video-drivable asset [60], but rely on synchronized high-speed photometric capture, a specialized configuration that limits adoption. This paper focuses on building a more standard multi-camera capture setup and introduces several practical simplifications to facilitate this style of capture.

We introduce FaceSnap, an end-to-end framework that amortizes the cost of high-fidelity facial capture. It involves a single, one-time multi-view session in the lightstage, which builds a personalized tracking model, after which every subsequent performance is captured in real time from a single lightstage camera. By front-loading the expensive multi-camera step into a reusable model, FaceSnap eliminates the need for repeated full-array captures that make current workflows impractical for iterative production. Figure 1 shows results for both stages.

The key contributions of our work are: (1) FaceSnap, a fully automated, end-to-end pipeline for facial performance capture, in which a one-time multi-view optimization builds a personalized tracking model that enables real-time tracking from a single monocular lightstage camera. This amortizes the cost of multi-camera capture into a reusable model, eliminating the need for repeated full-array sessions that make current workflows impractical for iterative production. (2) A novel one-time optimization that, through our regularization scheme, outperforms state-ofthe-art reconstruction methods that require multi-array cameras. (3) A real-time tracker, driven by the personalized model, that estimates geometry and 4K dynamic appearance from an expression code alone. At its core, a personalized two-stage residual upscaler recovers subject-specific high-frequency detail, allowing the tracker to outperform monocular baselines in texture fidelity while running in real time. (4) Multi4D, a public benchmark based on Multiface [58] for 4D facial tracking in lightstage environments that enables topology-agnostic comparison across methods. This gap has prevented rigorous evaluation in prior work. Multi4D provides raw scan sequences, evaluation metrics source code for reproducible research.

## 2. Related Works

While lightstage capture provides high-fidelity facial data, obtaining meshes in a pre-defined topology remains challenging. Production workflows require manual mesh retopology using tools like Wrap4D [24]. Conventional optimization-based approaches [4, 10–12, 31, 32] automate this via multi-view stereo reconstruction and non-rigid ICP registration to deform canonical models [49].

Recent implicit representations [47] have been explored for photorealistic head avatars [14, 26, 46], with Gaussian splatting methods [30, 51, 53, 54, 59] showing impressive results. However, these lack explicit geometry and texture control for production pipelines. Topo4D [43] addresses this by outputting engine-compatible meshes and textures, but requires re-optimization for each new sequence and multi-camera lightstage capture. Moreover, Topo4D struggles to track rapid expression changes.

(a) Learning-based Methods: To address slow optimization-based methods, several data-driven approaches [1, 9, 34, 40, 42, 44] have been proposed. These methods take multi-view images as input and directly decode meshes in a consistent topology. However, these learning-based methods are limited by their dependence on large amounts of collected scans (e.g. 64 subjects in [42, 44]), which are expensive to collect. Moreover, they are camera-sensitive and trained on specific lightstage configurations—when camera positions change, the model requires complete retraining on similarly extensive datasets [40], limiting their practical usability. To avoid the need for specialized hardware, monocular methods leverage large-scale face images available in-the-wild and parametric models like FLAME [41] to reconstruct facial geometry from single images or videos [3, 15, 16, 19, 21, 25, 28, 50, 52, 55, 57, 61]. Methods like EMOCA [16] and DECA [25] predict 3DMM coefficients and use differentiable rendering [35, 38] to minimize a photometric loss. Some works go beyond 3DMM while still being regularized by it [20, 57, 64]. While these approaches bypass lightstage capture requirements, the quality remains below production standards due to the limited representation of parametric morphable models.

![](images/ea969793de6c662d5554fd8909b0b27230b88167d6741539863d76cdd6d1f691.jpg)  
Figure 2. Overview of the FaceSnap method. From multi-view Lightstage captures, FaceSnap: (i) performs multi-stage optimization to obtain registered facial animation, and (ii) trains subject-specific models that estimate geometry and texture. These models enable high fidelity monocular face capture of new performances.

(b) Personalized Models for Performance Capture: Learning-based methods require expensive data, while optimization methods need re-processing for each capture session. Personalized models [2, 5, 13, 39, 45, 60] have been proposed to reduce capture sessions. As an example, SPARK [2] builds personalized models from in-the-wild videos without Lightstage data but achieves results below production quality. Using lightstage data, Lombardi et al. [45] proposed Deep Appearance Models from VR cameras, while Zhang et al. [60] produce high-quality meshes and textures with wrinkle maps. Our work differs in three ways: (1) methods from [45, 60] rely on pre-registered meshes from off-the-shelf trackers, whereas we provide end-to-end mesh tracking from lightstage data; (2) their focus is appearance modeling, while ours streamlines facial capture from lightstage for both geometry and textures; and (3) [60] uses different capture hardware than traditional lightstage, introducing additional hardware constraints.

## 3. Proposed FaceSnap Method

As illustrated in Figure 2, FaceSnap’s two-stage approach first performs a one-time multi-view optimization on a range-of-motion sequence. The registered meshes are used as priors to build a personalized subject model encoding geometry and expression-dependent appearance (Sec. 3.1– 3.2), thus enabling real-time tracking from a single Lightstage camera for any subsequent performance (Sec. 3.3).

## 3.1. One-time Multi-Stage Optimization

This step takes as input multi-view images $\{ I _ { 1 } ^ { t } , I _ { 2 } ^ { t } , \ldots , I _ { N } ^ { t } \}$ captured from N cameras over time $t \in [ 0 , S ]$ . A fully automated photogrammetry step first produces a sequence of raw scans $\{ \mathcal { R } ^ { t } \}$ with appearance textures $\{ A ^ { t } \}$ , and a subject neutral mesh $\mathcal { M } _ { \mathrm { n e u t r a l } }$ in consistent topology is assumed provided. The goal is to generate registered meshes $\{ \mathcal { M } ^ { t } \}$ in canonical space with consistent topology while preserving photogrammetry detail. Starting from $M _ { \mathrm { n e u t r a l } } ,$ our pipeline comprises three stages: (1) global registration, (2) base fitting, and (3) vertex refinement (see Figure 3). Building upon classical mesh optimization [41], our key improvements are a non-linear expression model (Sec. 3.1.1), robust vertex refinement (Sec. 3.1.2), and specialized loss functions including a lip contour loss (Sec. 3.1.3) that together enhance convergence stability and geometric fidelity. Global Registration: Procrustes alignment [22] is applied on neutral frame, $t \ = \ 0 ,$ , estimating a similarity transform $( \mathbf { R } _ { \mathrm { g } } , \mathbf { T } _ { \mathrm { g } } , s _ { \mathrm { g } } )$ that minimizes 3D landmark distances between $\mathcal { R } ^ { 0 }$ and $\mathcal { M } _ { \mathrm { n e u t r a l } }$ , then applying it to align the full sequence $\{ \mathcal { R } ^ { t } \}$ to canonical coordinates $\{ \widetilde { \mathcal { R } } ^ { t } \}$ (Figure 3, Part A). Full details are provided in the supplementary.

## 3.1.1. Base Fitting

Following global initialization, base fitting deforms the neutral mesh $\mathcal { M } _ { \mathrm { n e u t r a l } }$ per frame by optimizing a lowdimensional latent expression code $\mathbf { z } _ { \mathrm { e x p } } ^ { t }$ (Figure 3, Part B), which also regularizes the optimization. The deformation comes from a pretrained expression decoder $D _ { \mathrm { e x p } }$ (architecture follows [33]) that produces geometric deformations conditioned on the subject’s neutral geometry, preserving identity-specific facial structure. The low-dimensional latent space provides effective regularization during optimization. We also optimize a per-frame rigid transformation $( \mathbf { R } ^ { t } , \mathbf { T } ^ { t } ) \in S E ( 3 )$ to refine head pose and compensate for alignment imperfections. The deformed mesh at frame t is:

$$
\widetilde { \mathcal { M } } ^ { t } = \mathbf { R } ^ { t } \cdot \left[ \mathcal { M } _ { \mathrm { n e u t r a l } } + D _ { \mathrm { e x p } } ( \mathcal { M } _ { \mathrm { n e u t r a l } } , z _ { \mathrm { e x p } } ^ { t } ) \right] + \mathbf { T } ^ { t }\tag{1}
$$

We jointly optimize expression code $z _ { \mathrm { e x p } } ^ { t }$ and pose parameters $( \mathbf { R } ^ { t } , \mathbf { T } ^ { t } )$ to minimize point-to-surface distance adaptively between transformed mesh $\widetilde { \mathcal { M } } ^ { t }$ and rigidly aligned raw scan $\mathcal { R } ^ { t }$ , ensuring temporal consistency by initializing $z _ { \mathrm { e x p } } ^ { t }$ and $( \mathbf { R } ^ { t } , \mathbf { T } ^ { t } )$ from the previous frame. Section 3.1.3 describes the complete loss function used for this stage.

![](images/3e61c1a25f5f9fa71c2fc7545bdb8fbc4354f46360611e2cb1a4c97f36ccc2bd.jpg)  
Figure 3. The multi-stage optimization of FaceSnap consists of (A) global registration, which rigidly aligns the raw sequence to the canonical coordinate space, producing aligned sequence $\{ \widetilde { \mathbf { R } } ^ { t } \}$ that serves as the target for subsequent stages. Then, for each frame, (B) basefitting deforms the mesh $\mathcal { M } _ { \mathrm { n e u t r a l } }$ by optimizing rigid transformation and expression parameters constrained by a non-linear expression decoder. Finally, (C) vertex refinement allows per-vertex displacements to obtain the registered mesh $\mathcal { M } ^ { t }$

## 3.1.2. Vertex Refinement

After the base fitting with the non-linear expression model, we proceed to a vertex refinement step to recover more subtle details and achieve better fitting (Figure 3, Part C). While the parametric model provides a good foundation, it cannot accurately represent all possible expressions across different individuals. For this, we introduce per-vertex displacement vectors $\delta ^ { t } \in \mathbb { R } ^ { N _ { v } \times 3 }$ that allow for more precise local adjustments at each frame $t ,$ where $N _ { v }$ is the number of vertices. Each vertex i has a 3D displacement vector $\delta _ { i } ^ { t }$ optimized to capture fine-scale deformations. The final mesh is computed as:

$$
\mathcal { M } ^ { t } = \widetilde { \mathcal { M } } ^ { t } + \delta ^ { t }\tag{2}
$$

This unconstrained deformation can easily lead to implausible geometry and local minima. We therefore introduce regularization constraints (Section 3.1.3), balancing detail capture and anatomical plausibility. The Part-based edge preservation, normal consistency, and Laplacian regularization prevent distortion while capturing subject-specific details beyond the parametric expression model.

Automatic key framing with temporal smoothing. To prevent temporal drift, we employ automatic key framing that resets the per-vertex displacement field $\delta ^ { t }$ when a neutral pose is detected. The current mesh M<sup>t</sup> is compared against the neutral geometry $\mathcal { M } _ { \mathrm { n e u t r a l } }$ using L2 distance on a frontal face mask. When the average distance falls below the threshold $\tau _ { f } .$ we trigger a reset and retroactively smooth displacements over the previous N frames using linear decay $\textstyle \alpha _ { i } = { \frac { i + 1 } { N } }$ for frame i back from the reset. This creates gradual transitions.

## 3.1.3. Loss Functions

The following loss functions are used in both optimization stages. For notation convenience, we denote $\mathcal { X } ^ { t }$ as the mesh being optimized: $\mathcal X ^ { t } = \widetilde { \mathcal M } ^ { t }$ during base fitting and $\mathcal { X } ^ { t } = \mathcal { M } ^ { t }$ during vertex refinement.

Adaptive point-to-surface distance. We build on the point-to-surface distance objective of [8] and introduce two modifications for greater robustness and efficiency:

(\*1) Instead of using all mesh vertices, we uniformly sample N points $\mathbf { \mathcal { S } } ^ { t } = \{ \mathbf { s } _ { i } \} _ { i = 1 } ^ { N }$ from the raw scan. This sampling strategy ensures balanced spatial coverage regardless of the irregular vertex distribution in the raw scan, while significantly reducing computational cost. We also discard mouth and eye interior vertices by rendering the textured raw mesh and applying a face segmentation [62] to identify and discard vertices falling inside these regions, preventing these vertices from producing erroneous correspondences in the point-to-surface distance. (2) We introduce an adaptive correspondence weighting w that automatically discards unreliable correspondences. The point-to-surface loss is then:

$$
\mathcal { L } _ { \mathrm { p 2 s } } = \sum _ { i = 1 } ^ { N } w _ { i } \cdot \mathbf { G M } \left( \lVert \mathbf { s } _ { i } - \mathbf { q } _ { i } \rVert ^ { 2 } , \sigma \right)\tag{3}
$$

where $\mathbf { q } _ { i }$ is the closest point on $\mathcal { X } ^ { t }$ found by barycentric projection onto the nearest triangle face and GM is the Geman–McClure robustifier [29]. The adaptive weights are defined as:

$$
w _ { i } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \mathrm { G M } \big ( \lVert \mathbf { s } _ { i } - \mathbf { q } _ { i } \rVert ^ { 2 } , \boldsymbol { \sigma } \big ) < \tau _ { s } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{4}
$$

with $\tau _ { s }$ a distance threshold. This step removes corrupted or severely misaligned correspondences, particularly in regions where photogrammetry is unreliable (e.g., hair boundaries, specular highlights).

Lip contour loss. Accurate lip tracking is critical for realistic facial animation. Traditional landmark-based approaches rely on a few sparse points that require fixed correspondences without semantic meaning, making them insufficient for capturing the complex deformations of lip contours. To address this limitation, we introduce a 2D chamfer distance loss that captures the complete outline of the lip without requiring any point correspondences.

Our method uses a facial segmentation network [37] to identify upper and lower lip regions, then applies contour detection to find the boundaries of each lip region. To extract the inner lip boundary, we group the detected contour points by x-coordinate and select the bottom-most points from the upper lip contour and the top-most points from the lower lip contour for each x-coordinate. This process extracts the inner edge of the lips, providing a 2D contour representation denoted as $Q .$ . The chamfer distance loss is formulated as:

$$
\mathcal { L } _ { \mathrm { l i p } } = \frac { 1 } { | P | } \sum _ { p \in P } \operatorname* { m i n } _ { q \in Q } \| p - q \| ^ { 2 } + \frac { 1 } { | Q | } \sum _ { q \in Q } \operatorname* { m i n } _ { p \in P } \| q - p \| ^ { 2 }\tag{5}
$$

where $P$ represents the corresponding contour points from the registered mesh $\mathcal { X } ^ { t }$ back-projected to 2D using the camera perspective. This bidirectional formulation enables flexible contour matching without fixed correspondences, accurately capturing lip deformations.

Regularization losses. To ensure robust and physically plausible mesh optimization, we employ three regularization terms. The Part-Based Edge Preservation Loss $\mathcal { L } _ { \mathrm { e d g e } }$ [8] prevents sliding effects by constraining edge vectors between optimization iterations, with spatially-varying weights emphasizing critical facial regions like eyes and lips. The Laplacian Loss ${ \mathcal { L } } _ { \mathrm { l a p } }$ [48] promotes smooth deformations by preserving local differential properties encoded in Laplacian coordinates, particularly important for challenging photogrammetry regions such as eyebrows and eyelids. The Normal Consistency Loss ${ \mathcal { L } } _ { \mathrm { n c } }$ encourages smooth surface reconstruction by penalizing inconsistent orientations between adjacent face normals. These regularization terms collectively maintain mesh topology consistency while allowing for natural deformations during the optimization. Full details are in the supp. material.

Overall Objective Function. The total loss function is formulated as a weighted combination of individual losses:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { p 2 s } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { l i p } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { l a p } } + \lambda _ { 4 } \mathcal { L } _ { \mathrm { e d g e } } + \lambda _ { 5 } \mathcal { L } _ { \mathrm { n c } }\tag{6}
$$

The weighting coefficients $\lambda _ { i }$ are tuned differently for each optimization stage and provided in the supp. material.

## 3.1.4. Texture Extraction

To obtain high-quality appearance texture set $\{ \mathcal T ^ { t } \}$ , at 4K resolution, for each registered mesh $\mathcal { M } ^ { t }$ , we use the spatial correspondence established during registration to map texture information from the raw scan $\mathcal { R } ^ { t }$ to the registered mesh $\mathcal { M } ^ { t }$ . Specifically, we project the photogrammetry texture onto the consistent UV parameterization of the template topology. This is achieved by finding the nearest surface point on the raw scan for each vertex in the registered mesh and sampling the corresponding texture coordinates. This results in a sequence of meshes with both geometric consistency and photorealistic textures.

## 3.2. Personalized Tracking Model

After optimization (subsection 3.1) on the subject’s ROM session, we obtain registered meshes $\{ \mathcal { M } ^ { t } \}$ and textures $\{ \mathcal T ^ { t } \}$ , used to train an expression dependent personalized tracker comprising a geometry model and two-stage appearance model.

## 3.2.1. Geometry Model

To create a compact geometric representation, we compress registered meshes $\{ \mathcal { M } ^ { t } \} _ { t = 1 } ^ { T }$ into a 256-dimensional PCA subspace. This representation retains over 99% of geometric variance, indicating that facial expressions exhibit strong linear correlations that can be effectively captured without requiring more complex nonlinear models. Each frame is represented by coefficients $\mathbf { c } ^ { t } \in \mathbb { R } ^ { 2 5 6 }$ . We tested 256 and 1024 dimensions; both retained ${ > } 9 9 \%$ variance with negligible quality difference, so we use 256 for efficiency.

We train a ResNet encoder $E _ { \mathrm { g e o m } } : \mathbb { R } ^ { H \times W \times 3 } \ \xrightarrow { \cdot }$ <sub>R</sub>256 that maps $5 1 2 \times 5 1 2$ monocular frames directly to PCA coefficients.To enable direct mesh reconstruction from these coefficients, we append a frozen PCA decoder layer $F$ $\mathbb { R } ^ { 2 5 6 }  \mathbb { R } ^ { 3 N _ { v } }$ initialized with the PCA basis as weights and mean shape as bias. The encoder is trained exclusively on ROM sequences using the registered meshes from the optimization stage as ground truth with L2 loss: $\mathcal { L } _ { t r a c k } =$ $\lVert \bar { \boldsymbol { \mathcal { M } } } ^ { t } - F ( E _ { \mathrm { g e o m } } ( I ^ { t } ) ) \rVert _ { 2 } ^ { 2 } .$ where $I ^ { t }$ is the input image.

## 3.2.2. Dynamic Appearance

We recover production-quality dynamic appearance in two stages: a lightweight expression-conditioned decoder that synthesizes low-resolution texture maps from the learned geometric representation, followed by a personalized upscaler that lifts these maps to 4K resolution in real time, as shown in Figure 4.

Stage 1: Expression-Conditioned Texture Synthesis Given the subject profile built in subsubsection 3.2.1, we train an expression-dependent texture decoder $D _ { \mathrm { t e x } } : \mathbb { R } ^ { d } $ R<sup>512×512×3</sup> that generates texture maps conditioned solely on the geometric PCA coefficients $\mathbf { c } ^ { t }$ extracted from that profile. This design explicitly captures the correlation between facial geometry and appearance — wrinkles, specular highlights, and skin deformations are all encoded in $\mathbf { c } ^ { t }$ — without requiring image-based supervision at inference time. The decoder is trained with a masked L1 loss in UV space:

$$
\mathcal { L } _ { \mathrm { t e x } } = \| \mathbf { M } \odot ( \mathcal { T } ^ { t } - D _ { \mathrm { t e x } } ( \mathbf { c } ^ { t } ) ) \| _ { 1 } ,\tag{7}
$$

where $\mathcal { T } ^ { t }$ is the ground-truth UV texture for frame t and M is a frontal face mask that concentrates supervision on critical regions. No additional capture is required beyond the single Lightstage session used to build the subject profile.

Stage 2: Personalized Residual Upscaler Off-the-shelf super-resolution models introduce smoothing artifacts on facial textures and fail to preserve high-frequency details such as skin pores and subsurface structures, as they have no knowledge of the subject’s specific appearance. We therefore train a subject-specific upscaling network $U _ { \theta }$ that lifts the 512×512 output of Stage 1 directly to a 4096×4096 texture.

![](images/b099798c345b82aa83d50276fb872197b699f6e0992cf4ab348d7b4d29e654fe.jpg)  
Figure 4. Two-stage dynamic appearance pipeline. (A) Low-res synthesis: PCA coefficients $\mathbf { c } ^ { t }$ are decoded via deconvolution layers into a low-res dynamic texture $\hat { \mathcal { T } } ^ { t }$ , supervised with masked L1 against ground-truth UV textures $\mathcal { T } ^ { t }$ . (B) Personalized upscaling: The offset $\hat { \mathcal { T } } ^ { t } - \mathcal { T } _ { \mathrm { n e u t } } ^ { \mathrm { L R } }$ is lifted to 4K via residual RepConv blocks and added to $\mathcal { T } _ { \mathrm { n e u t } } ^ { \mathrm { H R } }$ , recovering fine-scale expression detail. Both stages train solely on ROM data.

Rather than predicting absolute pixel values, $U _ { \theta }$ performs residual prediction: it estimates a pixel-space offset $\Delta ^ { t }$ over the subject’s neutral high-resolution texture $\mathcal { T } _ { \mathrm { n e u t } } ^ { \mathrm { H R } } .$

$$
\begin{array} { r } { \hat { \mathcal { T } } _ { \mathrm { H R } } ^ { t } = \mathcal { T } _ { \mathrm { n e u t } } ^ { \mathrm { H R } } + U _ { \theta } ( \hat { \mathcal { T } } ^ { t } ) , } \end{array}\tag{8}
$$

where $\hat { \mathcal { T } } ^ { t } = D _ { \mathrm { t e x } } ( \mathbf { c } ^ { t } )$ is the Stage 1 prediction. This formulation focuses the network exclusively on expressiondependent, high-frequency deviations from the neutral state, substantially simplifying the learning problem and improving reconstruction of fine-grained detail. The network is supervised with a masked L1 loss in UV space:

$$
\mathcal { L } _ { \mathrm { u p } } = \Vert \mathbf { M } ^ { \mathrm { H R } } \odot ( \mathcal { T } _ { \mathrm { H R } } ^ { t } - \mathcal { \hat { T } } _ { \mathrm { H R } } ^ { t } ) \Vert _ { 1 } ,\tag{9}
$$

where ${ \bf { M } } ^ { \mathrm { H R } }$ is the topology mask upsampled to 4K resolution.

The backbone of $U _ { \theta }$ is a RepConv architecture [27], which employs multi-branch convolutions during training — combining 3×3, 1×1, and identity paths in parallel — to maximize representational capacity. At inference, all branches are fused into a single convolution via structural reparameterization [6], recovering the speed of a plain network with no architectural overhead. This makes $U _ { \theta }$ both highly expressive for subject-specific detail recovery and efficient enough for real-time performance.

![](images/a2b6dbd4a93299a81196523d81a252c1a875117ba628da8debfa1ed3c442024f.jpg)  
Figure 5. Visual comparison of mesh reconstruction quality across different subjects on the Multi4D benchmark.

Table 1. Geometry comparison of multi-view face optimization methods using point-to-surface distance (mm, lower is better), ± per-frame standard deviation over each sequence.
<table><tr><td>Method</td><td>S1</td><td>S2</td><td>S3</td><td>S4</td><td>S5</td><td>S6</td><td>Overall</td></tr><tr><td>FLAME Fitting</td><td>0.643 ±0.151</td><td>1.35 ±0.237</td><td>0.625 ±0.194</td><td>0.600 ±0.207</td><td>0.672 ±0.0965</td><td>0.683 ±0.182</td><td>0.793</td></tr><tr><td>Topo4D</td><td>1.28 ±0.465</td><td>1.71 ±0.780</td><td>1.21 ±0.382</td><td>1.07 ±0.504</td><td>0.641 ±0.175</td><td>1.21 ±0.483</td><td>1.24</td></tr><tr><td>Ours (Base Fitting)</td><td>0.814 ±0.436</td><td>0.844 ±0.450</td><td>0.796 ±0.412</td><td>0.711 ±0.331</td><td>0.614 ±0.155</td><td>0.817 ±0.357</td><td>0.779</td></tr><tr><td>Ours (Full)</td><td>0.584 ±0.394</td><td>0.505 ±0.404</td><td>0.477 ±0.364</td><td>0.420 ±0.278</td><td>0.240 ±0.125</td><td>0.508 ±0.345</td><td>0.471</td></tr></table>

## 3.3. Real-Time Monocular Tracking

At inference, the pretrained geometry encoder of our tracker $E _ { \mathrm { g e o m } }$ regresses PCA coefficients $\mathbf { c } ^ { t }$ directly from a single lightstage camera frame, which are then decoded into a dynamic texture $\hat { \mathcal { T } } ^ { t }$ via $D _ { \mathrm { t e x } }$ and upscaled to 4K by $U _ { \theta }$ using our personalized upscaler. This unified approach delivers production-quality geometry and appearance in real time without any multi-camera capture or complex optimization.

## 4. Datasets and Benchmark

Multi4D Benchmark. We propose Multi4D, a new benchmark based on the Multiface dataset [58] to enable topology invariant mesh geometry comparison. It contains 4D lightstage captures of 6 subjects with varying facial characteristics performing a partial range of motion sequence between ∼1150 to ∼1500 frames each (∼7600 frames total). Since Multiface does not provide ground truth scans, we extract raw scans using photogrammetry software [23] for each frame to serve as ground truth. For evaluation, we sample 300K vertices from each raw scan and clip them to include only the frontal facial region for point-to-surface distance computation. We publicly release the evaluation source code and ground-truth raw scans for all subjects. More details in the supp. material.

Custom Lightstage Dataset. We capture data using a lightstage setup [17] with 24 time-synchronized, calibrated cameras at 60 FPS and 4K resolution. Our dataset includes 3 subjects performing: (1) Range of Motion (ROM) sequences (∼5 minutes) following FACS principles for profile creation, and (2) expressive sequences with semantically distinct spoken sentences (happy, angry, articulated, 20 seconds each) for tracking evaluation.

![](images/943d68f126859aa476ddb4d7937907021dc113bc0b7cc66e4ec7145a5008d0af.jpg)  
Figure 6. Qualitative geometric comparison for different methods.

## 5. Evaluation and Discussion

## 5.1. Evaluation of our Multi-Stage Optimization

We evaluate geometric accuracy of our optimization (Sec. 3.1) against FLAME fitting [41], which fits a FLAME model directly to raw photogrammetry scans, and Topo4D [43], a Gaussian splatting-based per-frame optimization requiring a manually registered neutral mesh, partwise facial annotations, and 8K texture and multi-view Lightstage images as input. Both share a comparable initialization burden to ours. We report point-to-surface distance against ground-truth photogrammetry scans on Multi4D in Table 1.

Our optimization achieves the best overall accuracy (0.47mm), improving 40.7% over FLAME fitting (0.79mm) and 62.2% over Topo4D (1.24mm). Topo4D shows high mean error and variance across subjects (range: 0.64– 1.71mm), indicating sensitivity to rapid expression dynamics in Multiface sequences. FLAME is constrained by parametric model capacity, with reduced accuracy in perceptually critical region; lips, eyes, jaw, and chin, as seen in Figure 5. The optimization first fits a parametric expression model (base fitting), then refines per-vertex displacements to recover fine-scale detail (vertex refinement). The gap between base fitting and full (0.78mm→0.47mm) confirms that vertex refinement meaningfully improves geometric fidelity beyond what the parametric model alone can capture. Ablations of notable loss components, including the lip contour loss, are provided in the supplementary material.

## 5.2. Evaluation of Real-Time Tracker

In this section, we evaluate our real-time tracker on lightstage data. We build a personalized tracking model from the ROM sequence using the lightstage data described in section 4 (custom lightstage data), and assess it on 9 expressive sequences, along three axes.

Table 2. Geometric comparison of face tracking methods using average point-to-surface distance (mm, lower is better), ± average per-frame standard deviation over each sequence.
<table><tr><td rowspan="2">Method</td><td colspan="3">Subject 1</td><td colspan="3">Subject 2</td><td colspan="3">Subject 3</td><td rowspan="2">Overall</td></tr><tr><td>Artic.</td><td>Angry</td><td>Happy</td><td>Artic.</td><td>Angry</td><td>Happy</td><td>Artic.</td><td>Angry</td><td>Happy</td></tr><tr><td colspan="10">Per-frame,Multi-View Optimization</td></tr><tr><td>FLAME Fitting</td><td>0.691 ±0.124 0.685 ±0.059 0.721 ±0.046</td><td></td><td></td><td>1.08 ±0.172</td><td>1.05 ±0.047</td><td></td><td>1.07 ±0.043 0.982 ±0.078 0.978 ±0.073 0.976 ±0.036 0.910 ±0.174</td><td></td><td></td><td></td></tr><tr><td>Topo4D</td><td>0.899 ±0.123 0.908 ±0.085 0.854 ±0.083</td><td></td><td></td><td>1.09 ±0.152</td><td>1.02 ±0.096</td><td>1.02 ±0.129</td><td>0.903 ±0.169 0.861 ±0.122 0.822 ±0.071 0.920 ±0.143</td><td></td><td></td><td></td></tr><tr><td colspan="10">Feed-forward Methods</td></tr><tr><td>SEREP</td><td>1.78 ±0.269</td><td>1.91 ±0.209</td><td>2.10 ±0.265</td><td>1.53 ±0.304</td><td>1.40 ±0.311</td><td>1.30 ±0.289</td><td>1.16 ±0.294</td><td>1.43 ±0.317</td><td>1.65 ±0.316</td><td>1.59 ±0.409</td></tr><tr><td>DECA</td><td>3.06 ±1.42</td><td>2.73 ±2.10</td><td>3.33 ±1.44</td><td>1.83 ±1.08</td><td>2.23 ±5.22†</td><td>2.30 ±5.48†</td><td>1.58 ±0.747</td><td>2.12 ±1.01</td><td>2.21 ±2.01</td><td>2.38 ±2.82</td></tr><tr><td>FLAME_BACKBONE</td><td>1.11 ±0.231</td><td>2.22 ±0.732</td><td>2.44 ±0.634</td><td>1.29 ±0.113</td><td>1.45 ±0.308</td><td>1.46 ±0.196</td><td>1.26 ±0.166</td><td>1.74 ±0.402</td><td>2.40 ±0.549</td><td>1.75 ±0.663</td></tr><tr><td>TOPO4D,BACKBONE</td><td>1.44 ±0.335</td><td>2.12 ±0.693</td><td>2.50 ±0.617</td><td>1.32 ±0.198</td><td>1.80 ±0.368</td><td>1.48 ±0.169</td><td>1.47 ±0.266</td><td>2.02 ±0.549</td><td>3.09 ±0.697</td><td>1.97 ±0.752</td></tr><tr><td>Ours (Tracking)</td><td>0.966 ±0.314</td><td>1.08 ±0.211</td><td>1.33 ±0.132</td><td>0.867 ±0.175</td><td>0.754 ±0.129</td><td>0.897 ±0.195 0.687 ±0.189</td><td></td><td>0.786 ±0.269</td><td>0.842 ±0.311</td><td>0.923 ±0.294</td></tr></table>

† DECA std exceeds the mean on these sequences, reflecting frequent failure frames.

Comparison to monocular trackers We compare our monocular real-time tracker against state-of-the-art monocular tracking methods, DECA and SEREP, which are inference-based models trained on large-scale data.Both the methods are provided the subject neutral. For more details please see the supp mat.

Reference comparison to multi-view optimization As an upper-bound reference, we also compare against FLAME fitting and Topo4D, which recover geometry through perframe optimization over full camera array. Unlike our tracker, which infers geometry from a single camera at test time, these methods use all views and optimize each frame independently. We include them to show that our singlecamera inference remains on par with full multi-view optimization, despite operating monocularly.

Effect of the backbone Our personalized tracking model is built on the registered meshes produced by our onetime optimization (subsection 3.1). To isolate the contribution of this step, we replace it with alternative registration methods while keeping the downstream pipeline fixed. Specifically, we use FLAME fitting to register the ROM sequence, then build a personalized tracking model from the resulting meshes following subsection 3.2; we apply the same procedure with Topo4D. We refer to these variants as FLAME BACKBONE and TOPO4D BACKBONE. Since only the registration backbone differs, this comparison directly measures its effect on final tracker performance (details in supp. mat).

Results. FaceSnap (0.92mm) outperforms all feed-forward methods by a substantial margin. Notably, it surpasses SEREP (1.59mm) even though both receive identical neutral input and production-quality training data, confirming that subject-specific lightstage capture cannot be replaced by generic feed-forward methods. The backbone variants FLAME BACKBONE (1.75mm) and TOPO4D BACKBONE (1.97mm) perform worse than FaceSnap despite sharing its architecture, showing that tracker quality is driven by the quality of the registered meshes used to train the personalized model, not by architecture alone. Finally, FaceSnap is on par with the multi-view per-frame optimization baselines (FLAME: 0.91mm, Topo4D: 0.92mm), while using

![](images/71f72054b01cf6d5c5f95392949afba5364308a574e8f32e21efebea6acddac3.jpg)  
Figure 7. Qualitative comparison of appearance across subjects for FaceSnap and baseline methods on the custom lightstage dataset.

Table 3. Appearance quality comparison across subjects and expressions, aggregated over 3 camera views. Shaded row = our method. ↑ higher is better, ↓ lower is better. <sup>∗</sup>Best perceptual similarity (LPIPS).
<table><tr><td></td><td></td><td colspan="3">Subject 1</td><td colspan="3">Subject 2</td><td colspan="3">Subject 3</td><td></td></tr><tr><td>Metric</td><td>Method</td><td>Art.</td><td>Ang.</td><td>Hap.</td><td>Art.</td><td>Ang.</td><td>Hap.</td><td>Art.</td><td>Ang.</td><td>Hap.</td><td>Overall</td></tr><tr><td>SSIM ↑</td><td>Topo4D</td><td>0.955</td><td>0.952</td><td>0.952</td><td>0.978</td><td>0.979</td><td>0.978</td><td>0.943</td><td>0.940</td><td>0.939</td><td>0.957</td></tr><tr><td></td><td>Upscaler (Face-Tuned)</td><td>0.963</td><td>0.962</td><td>0.959</td><td>0.980</td><td>0.982</td><td>0.980</td><td>0.952</td><td>0.947</td><td>0.944</td><td>0.963</td></tr><tr><td></td><td>Ours (Personalised)</td><td>0.960</td><td>0.958</td><td>0.955</td><td>0.979</td><td>0.980</td><td>0.979</td><td>0.946</td><td>0.941</td><td>0.939</td><td>0.960</td></tr><tr><td></td><td>Topo4D</td><td>39.8</td><td>39.8</td><td>39.6</td><td>40.6</td><td>41.1</td><td>41.2</td><td>38.9</td><td>38.3</td><td>38.5</td><td>39.8</td></tr><tr><td>PSNR ↑</td><td>Upscaler (Face-Tuned)</td><td>38.0</td><td>37.2</td><td>37.0</td><td>38.9</td><td>38.9</td><td>39.0</td><td>37.1</td><td>36.5</td><td>36.6</td><td>37.7</td></tr><tr><td></td><td>Ours (Personalised)</td><td>37.7</td><td>37.0</td><td>36.7</td><td>38.8</td><td>38.8</td><td>39.0</td><td>36.9</td><td>36.4</td><td>36.4</td><td>37.5</td></tr><tr><td></td><td>Topo4D</td><td>0.0543</td><td>0.0528</td><td>0.0579</td><td>0.0329</td><td>0.0300</td><td>0.0316</td><td>0.0627</td><td>0.0670</td><td>0.0632</td><td>0.0503</td></tr><tr><td>LPIPS ↓</td><td>Upscaler (Face-Tuned)</td><td>0.0723</td><td>0.0758</td><td>0.0807</td><td>0.0538</td><td>0.0542</td><td>0.0531</td><td>0.0758</td><td>0.0807</td><td>0.0813</td><td>0.0698</td></tr><tr><td></td><td>Ours (Personalised)*</td><td>0.0516</td><td>0.0560</td><td>0.0608</td><td>0.0312</td><td>0.0296</td><td>0.0310</td><td>0.0579</td><td>0.0620</td><td>0.0669</td><td>0.0497</td></tr></table>

only a single-camera video sequence. However, as shown in Figure $^ { 6 , }$ mean point-to-surface scores alone do not capture localized errors in perceptually critical regions where FaceSnap preserves lip shape and identity detail better.

## 5.3. Evaluation of Personalized Residual Upscaler

We evaluate the fidelity of the tracker’s predicted 4K texture (see section 3.2.2) via novel-view synthesis. For each subject (3 identities, 9 expressive sentences in total), we render the ground-truth photogrammetry scans from three novel viewpoints at 4K resolution and use these as reference. We compare three configurations spanning a range of capture and supervision costs: (i) Topo4D, which performs per-frame photometric optimization over the full multi-view array; (ii) our monocular tracker with ESRGAN [56], a state-of-the-art upscaler fine-tuned on a large-scale lightstage dataset of 1,000 subjects; and (iii) our monocular tracker with our personalized upscaler, trained solely on the target subject’s ROM data.

Despite operating from a single monocular view, our personalized upscaler matches Topo4D’s full multi-view capture in perceptual quality, achieving comparable LPIPS (0.0497 vs. 0.0503). SSIM is near-saturated across all methods (<0.01 spread), confirming that it does not discriminate at this fidelity. FaceSnap’s lower PSNR relative to Topo4D (37.52 vs. 39.75 ,dB) is consistent with the perception-distortion tradeoff [7]: methods optimizing for perceptual quality necessarily sacrifice some pixel-level fidelity. The contrast is decisive against the generic upscaler: our personalized model improves LPIPS by a substantial margin (0.0497 vs. 0.0698 for ESRGAN, fine-tuned on 1,000 subjects), demonstrating that single-subject ROM training recovers high-frequency identity detail that largescale generic training cannot, as shown in Figure 7.

## 5.4. Real-Time Performance and Efficiency

FaceSnap’s real-time monocular tracker predicts full geometry and 4K texture in 12 ms per frame on an NVIDIA RTX A6000, comprising geometry inference (0.6 ms), appearance decoding (2.9 ms), and 4K upscaling (8.5 ms). Measured as end-to-end time to produce a tracked frame, this is a 5,000× speedup over Topo4D (∼ 60 s) and a 25,000 × speedup over the MVS+ICP registration used in production pipelines [24] (300–600 s depending on expression complexity). Crucially, these gains come from a single camera rather than a full multi-view array: beyond runtime, FaceSnap removes per-session multi-camera capture and photogrammetry processing entirely.

## 6. Conclusion

We presented FaceSnap, an end-to-end framework that amortizes the cost of high-fidelity lightstage facial capture into a reusable personalized subject model. Our one-time multi-stage optimization achieves superior geometric accuracy through non-linear expression modeling, vertex refinement, and a novel lip contour loss. The personalized tracker matches full multi-view optimization baselines at 83 fps from a single camera, with subject-specific appearance recovering high-frequency detail that generic models struggle to reproduce. We additionally introduce Multi4D, a public benchmark enabling topology-invariant evaluation of 4D facial reconstruction methods. Together, these contributions reduce the infrastructure and iteration cost of production facial capture while providing standardized tools for future research. Limitations and future work are discussed in the supplementary material.

## 7. Acknowledgments

This work was made possible through the financial support of the Natural Sciences and Engineering Research Council of Canada (NSERC) and MITACS, awarded through the Alliance-Mitacs program under Grant No. ALLRP 589317- 23. We thank the teams behind Topo4D [43], FLAME Fitting [41], and DECA [25] for their open-source releases, and the authors of SEREP [33] for sharing their code with us. These resources enabled the comparisons and analyses reported in this work. We also thank the MultiFace team [58] for making their dataset available to build our benchmark upon.

## References

[1] Riza Alp Guler, George Trigeorgis, Epameinondas Antonakos, Patrick Snape, Stefanos Zafeiriou, and Iasonas Kokkinos. Densereg: Fully convolutional dense shape regression in-the-wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6799–6808, 2017. 2

[2] Kelian Baert, Shrisha Bharadwaj, Fabien Castan, Benoit Maujean, Marc Christie, Victoria Abrevaya, and Adnane Boukhayma. Spark: Self-supervised personalized real-time monocular face capture. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12, 2024. 3

[3] Anil Bas, William AP Smith, Timo Bolkart, and Stefanie Wuhrer. Fitting a 3d morphable model to edges: A comparison between hard and soft correspondences. In Computer Vision–ACCV 2016 Workshops: ACCV 2016 International Workshops, Taipei, Taiwan, November 20-24, 2016, Revised Selected Papers, Part II 13, pages 377–391. Springer, 2017. 2

[4] Thabo Beeler, Fabian Hahn, Derek Bradley, Bernd Bickel, Paul A Beardsley, Craig Gotsman, Robert W Sumner, and Markus H Gross. High-quality passive facial performance capture using anchor frames. ACM Trans. Graph., 30(4):75, 2011. 2

[5] Shrisha Bharadwaj, Yufeng Zheng, Otmar Hilliges, Michael J Black, and Victoria Fernandez-Abrevaya. Flare:

Fast learning of animatable and relightable mesh avatars. arXiv preprint arXiv:2310.17519, 2023. 3

[6] Kartikeya Bhardwaj, Milos Milosavljevic, Liam O’Neil, Dibakar Gope, Ramon Matas, Alex Chalfin, Naveen Suda, Lingchuan Meng, and Danny Loh. Collapsible linear blocks for super-efficient super resolution. Proceedings of machine learning and systems, 4:529–547, 2022. 6, 14

[7] Yochai Blau and Tomer Michaeli. The perception-distortion tradeoff. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6228–6237, 2018. 8

[8] Timo Bolkart, Tianye Li, and Michael J. Black. Instant multi-view head capture through learnable registration. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 768–779, 2023. 2, 4, 5, 12

[9] Timo Bolkart, Tianye Li, and Michael J Black. Instant multiview head capture through learnable registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 768–779, 2023. 2

[10] James Booth, Anastasios Roussos, Stefanos Zafeiriou, Allan Ponniah, and David Dunaway. A 3d morphable model learnt from 10,000 faces. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 5543–5552, 2016. 2

[11] Derek Bradley, Wolfgang Heidrich, Tiberiu Popa, and Alla Sheffer. High resolution passive facial performance capture. In ACM SIGGRAPH 2010 papers, pages 1–10. 2010.

[12] Chen Cao, Yanlin Weng, Shun Zhou, Yiying Tong, and Kun Zhou. Facewarehouse: A 3d facial expression database for visual computing. IEEE Transactions on Visualization and Computer Graphics, 20(3):413–425, 2013. 2

[13] Chen Cao, Vasu Agrawal, Fernando De La Torre, Lele Chen, Jason Saragih, Tomas Simon, and Yaser Sheikh. Realtime 3d neural facial animation from binocular video. ACM Transactions on Graphics (TOG), 40(4):1–17, 2021. 3

[14] Prashanth Chandran and Gaspard Zoss. Anatomically constrained implicit face models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2220–2229, 2024. 2

[15] Feng-Ju Chang, Anh Tuan Tran, Tal Hassner, Iacopo Masi, Ram Nevatia, and Gerard Medioni. Expnet: Landmark-free, deep, 3d facial expressions. In 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018), pages 122–129. IEEE, 2018. 2

[16] Radek Daneˇcek, Michael J Black, and Timo Bolkart. Emoca:ˇ Emotion driven monocular face capture and animation. In Proceedings of the IEEE/CVF Conference on Computer Vi sion and Pattern Recognition, pages 20311–20322, 2022. 2

[17] Paul Debevec. The light stages and their applications to photoreal digital actors. SIGGRAPH Asia, 2(4):1–6, 2012. 1, 6

[18] Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set. In Proceedings ofthe IEEE/CVF conference on computer vi sion and pattern recognition workshops, pages 0–0, 2019. 2

[19] Abdallah Dib, Cedric Thebault, Junghyun Ahn, Philippe-Henri Gosselin, Christian Theobalt, and Louis Chevallier.

Towards high fidelity monocular face reconstruction with rich reflectance using self-supervised learning and ray tracing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12819–12829, 2021. 2

[20] Abdallah Dib, Junghyun Ahn, Cedric Thebault, Philippe-Henri Gosselin, and Louis Chevallier. S2f2: Self-supervised high fidelity face reconstruction from monocular image. arXiv preprint arXiv:2203.07732, 2022. 3

[21] Abdallah Dib, Luiz Gustavo Hafemann, Emeline Got, Trevor Anderson, Amin Fadaeinejad, Rafael M. O. Cruz, and Marc-Andre Carbonneau. Mosar: Monocular semi-supervised´ model for avatar reconstruction using differentiable shading. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1770–1780, 2024. 2

[22] Ian L Dryden and Kanti V Mardia. Statistical shape analysis: with applications in R. John Wiley & Sons, 2016. 3, 12

[23] Epic Games. Realityscan. https : / / www . realityscan.com/en-US, 2025. Accessed: 2025-07- 24. 6, 15

[24] FaceForm. Faceform – 3d face scanning and reconstruction software, 2025. Accessed: 2025-07-25. 2, 8, 16

[25] Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart. Learning an animatable detailed 3d face model from in-thewild images. ACM Transactions on Graphics (ToG), 40(4): 1–13, 2021. 2, 9

[26] Guy Gafni, Justus Thies, Michael Zollhofer, and Matthias Nießner. Dynamic neural radiance fields for monocular 4d facial avatar reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8649–8658, 2021. 2

[27] Ganzorig Gankhuyag, Kihwan Yoon, Jinman Park, Haeng Seon Son, and Kyoungwon Min. Lightweight real-time image super-resolution network for 4k images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 1746–1755, 2023. 6, 14

[28] Baris Gecer, Stylianos Ploumpis, Irene Kotsia, and Stefanos Zafeiriou. Ganfit: Generative adversarial network fitting for high fidelity 3d face reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1155–1164, 2019. 2

[29] Stuart Geman and Donald E. McClure. Statistical methods for tomographic image reconstruction. In Proceedings ofthe 46th Session ofthe International Statistical Institute, Bulletin ofthe ISI, 1987. 4

[30] Simon Giebenhain, Tobias Kirschstein, Martin Runz, Lour-¨ des Agapito, and Matthias Nießner. Npga: Neural parametric gaussian avatars. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 2

[31] Pei-Lun Hsieh, Chongyang Ma, Jihun Yu, and Hao Li. Unconstrained realtime facial performance capture. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 1675–1683, 2015. 2

[32] Penglei Ji, Hanchao Li, Luyan Jiang, and Xinguo Liu. Lightweight multi-view topology consistent facial geometry and reflectance capture. In Computer Graphics International Conference, pages 139–150. Springer, 2021. 2

[33] Arthur Josi, Luiz Gustavo Hafemann, Abdallah Dib, Emeline Got, Rafael MO Cruz, and Marc-Andre Carbonneau. Serep: Semantic facial expression representation for robust in-the-wild capture and retargeting. arXiv preprint arXiv:2412.14371, 2024. 3, 9

[34] Harim Jung, Myeong-Seok Oh, and Seong-Whan Lee. Learning free-form deformation for 3d face reconstruction from in-the-wild images. In 2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pages 2737–2742. IEEE, 2021. 2

[35] Hiroharu Kato, Deniz Beker, Mihai Morariu, Takahiro Ando, Toru Matsuoka, Wadim Kehl, and Adrien Gaidon. Differentiable rendering: A survey. arXiv preprint arXiv:2006.12057, 2020. 2

[36] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 2

[37] Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. Sapiens: Foundation for human vision models. arXiv preprint arXiv:2408.12569, 2024. 4

[38] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives fo high-performance differentiable rendering. ACM Transac tions on Graphics (ToG), 39(6):1–14, 2020. 2

[39] Jiaman Li, Zhengfei Kuang, Yajie Zhao, Mingming He, Karl Bladin, and Hao Li. Dynamic facial asset and rig generation from a single scan. ACM Trans. Graph., 39(6):215–1, 2020. 3

[40] Jing Li, Di Kang, and Zhenyu He. Grape: Generalizable and robust multi-view facial capture. In European Conference on Computer Vision, pages 403–418. Springer, 2024. 2

[41] Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 36(6):194–1, 2017. 2, 3, 7, 9, 12, 15

[42] Tianye Li, Shichen Liu, Timo Bolkart, Jiayi Liu, Hao Li, and Yajie Zhao. Topologically consistent multi-view face inference using volumetric sampling. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3824–3834, 2021. 2

[43] Xuanchen Li, Yuhao Cheng, Xingyu Ren, Haozhe Jia, Di Xu, Wenhan Zhu, and Yichao Yan. Topo4d: Topologypreserving gaussian splatting for high-fidelity 4d head capture. In European Conference on Computer Vision, pages 128–145. Springer, 2024. 2, 7, 9, 15, 16

[44] Shichen Liu, Yunxuan Cai, Haiwei Chen, Yichao Zhou, and Yajie Zhao. Rapid face asset acquisition with recurrent feature alignment. ACM Transactions on Graphics (TOG), 41 (6):1–17, 2022. 2

[45] Stephen Lombardi, Jason Saragih, Tomas Simon, and Yaser Sheikh. Deep appearance models for face rendering. ACM Transactions on Graphics (ToG), 37(4):1–13, 2018. 3

[46] Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. Mixture of volumetric primitives for efficient neural rendering. ACM Transactions on Graphics (ToG), 40(4):1–13, 2021. 2

[47] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2

[48] Andrew Nealen, Takeo Igarashi, Olga Sorkine, and Marc Alexa. Laplacian mesh optimization. In Proceedings of the 4th international conference on Computer graphics and interactive techniques in Australasia and Southeast Asia, pages 381–389, 2006. 5

[49] Alice J O’Toole, Theodore Price, Thomas Vetter, James C Bartlett, and Volker Blanz. 3d shape and 2d surface textures of human faces: The role of “averages” in attractiveness and age. Image and Vision Computing, 18(1):9–19, 1999. 2

[50] Stylianos Ploumpis, Evangelos Ververas, Eimear O’Sullivan, Stylianos Moschoglou, Haoyang Wang, Nick Pears, William AP Smith, Baris Gecer, and Stefanos Zafeiriou. Towards a complete 3d morphable model of the human head. IEEE transactions on pattern analysis and machine intelligence, 43(11):4142–4160, 2020. 2

[51] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, and Matthias Nießner. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20299–20309, 2024. 2

[52] George Retsinas, Panagiotis P Filntisis, Radek Danecek, Victoria F Abrevaya, Anastasios Roussos, Timo Bolkart, and Petros Maragos. 3d facial expressions through analysis-byneural-synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2490–2501, 2024. 2

[53] Shunsuke Saito, Gabriel Schwartz, Tomas Simon, Junxuan Li, and Giljoo Nam. Relightable gaussian codec avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130–141, 2024. 2

[54] Kartik Teotia, Hyeongwoo Kim, Pablo Garrido, Marc Habermann, Mohamed Elgharib, and Christian Theobalt. Gaussianheads: End-to-end learning of drivable gaussian head avatars from coarse-to-fine representations. ACM Transactions on Graphics (TOG), 43(6):1–12, 2024. 2

[55] Justus Thies, Michael Zollhofer, Marc Stamminger, Christian Theobalt, and Matthias Nießner. Face2face: Real-time face capture and reenactment of rgb videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2387–2395, 2016. 2

[56] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1905–1914, 2021. 8

[57] Zidu Wang, Xiangyu Zhu, Tianshuo Zhang, Baiqin Wang, and Zhen Lei. 3d face reconstruction with the geometric guidance of facial part segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1672–1682, 2024. 2, 3

[58] Cheng-hsin Wuu, Ningyuan Zheng, Scott Ardisson, Rohan Bali, Danielle Belko, Eric Brockmeyer, Lucas Evans, Tim-

othy Godisart, Hyowon Ha, Xuhua Huang, Alexander Hy pes, Taylor Koska, Steven Krenn, Stephen Lombardi, Xiaomin Luo, Kevyn McPhail, Laura Millerschoen, Michal Perdoch, Mark Pitts, Alexander Richard, Jason Saragih, Junko Saragih, Takaaki Shiratori, Tomas Simon, Matt Stewart, Autumn Trimble, Xinshuo Weng, David Whitewolf, Chenglei Wu, Shoou-I Yu, and Yaser Sheikh. Multiface: A dataset for neural face rendering. In arXiv, 2022. 2, 6, 9, 15

[59] Ye Yuan, Xueting Li, Yangyi Huang, Shalini De Mello, Kok Nagano, Jan Kautz, and Umar Iqbal. Gavatar: Animatable 3d gaussian avatars with implicit mesh learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 896–905, 2024. 2

[60] Longwen Zhang, Chuxiao Zeng, Qixuan Zhang, Hongyang Lin, Ruixiang Cao, Wei Yang, Lan Xu, and Jingyi Yu. Videodriven neural physically-based facial asset for production. ACM Transactions on Graphics (TOG), 41(6):1–16, 2022. 2, 3

[61] Tianke Zhang, Xuangeng Chu, Yunfei Liu, Lijian Lin, Zhendong Yang, Zhengzhuo Xu, Chengkun Cao, Fei Yu, Changyin Zhou, Chun Yuan, and Yu Li. Accurate 3d face reconstruction with facial component tokens. In Proceedings ofthe IEEE/CVF International Conference on Computer Vi sion (ICCV), pages 9033–9042, 2023. 2

[62] Yinglin Zheng, Hao Yang, Ting Zhang, Jianmin Bao, Dong dong Chen, Yangyu Huang, Lu Yuan, Dong Chen, Ming Zeng, and Fang Wen. General facial representation learn ing in a visual-linguistic manner. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18697–18709, 2022. 4

[63] Zhenglin Zhou, Huaxia Li, Hong Liu, Nanyang Wang, Gang Yu, and Rongrong Ji. Star loss: Reducing semantic ambiguity in facial landmark detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15475–15484, 2023. 12, 15

[64] Xiangyu Zhu, Chang Yu, Di Huang, Zhen Lei, Hao Wang, and Stan Z Li. Beyond 3dmm: Learning to capture high fidelity 3d face shape. IEEE Transactions on Pattern Analy sis and Machine Intelligence, 45(2):1442–1457, 2022. 3

## Supplementary Material

## A. Ablation Studies

## A.1. Lip Countour Loss ablation

To demonstrate the effectiveness of our proposed lip contour loss, we conduct an ablation study by removing ${ \mathcal { L } } _ { \mathrm { l i p } }$ (eq. 5 in the paper) from eq. 6 and replacing it with a traditional sparse landmark loss using 51 fixed facial keypoints (we exclude the first 17 landmarks, corresponding to the jaw and chin contour as these boundary points are inherently ambiguous.) As shown in Figure 1, our contour-based approach significantly outperforms the landmark-based baseline in capturing subtle lip movements and maintain accurate lip contact. Unlike landmark-based methods that require fixed point correspondences, our approach avoids the correspondence problem entirely, enabling more flexible and accurate lip deformation, critical for high-fidelity facial animation.

## A.2. Regularization Loss Ablation

To validate the importance of our regularization terms, we conduct an ablation study by removing all regularization losses $( \mathcal { L } _ { \mathrm { e d g e } } , \mathcal { L } _ { \mathrm { l a p } }$ , and ${ \mathcal { L } } _ { \mathrm { n c } } )$ from the overall objective function (Eq. 6 in the paper). As demonstrated in Figure $^ { 2 , }$ the absence of these regularization terms leads to severe mesh degradation artifacts. Without the part-based edge preservation loss $\mathcal { L } _ { \mathrm { e d g e } } ,$ we observe prominent sliding effects that produce inconsistent edge flows, particularly noticeable around critical facial regions (The edge flows are visualized as wireframe overlays on top of the geometry in Figure 2). The removal of the normal consistency loss ${ \mathcal { L } } _ { \mathrm { n c } }$ results in flipped face normals that create unrealistic surface orientations. Additionally, without the Laplacian loss ${ \mathcal { L } } _ { \mathrm { l a p } }$ , the mesh exhibits jagged surface artifacts and unnatural discontinuities that violate smooth deformation assumptions. These results clearly demonstrate that our regularization losses are essential for maintaining mesh topology consistency and ensuring physically plausible facial reconstructions during optimization.

## B. Multi-Stage Optimization Details

Our multi-stage optimization takes multi-view lightstage images and raw photogrammetry scans as input and produces a sequence of registered meshes in a canonical topology, serving as pseudo-GT for the personalized subject model. The pipeline comprises three sequential stages; global registration, base fitting, and vertex refinement, each building on the previous. Here, we provide additional details for each stage.

## B.0.1. Global Registration

The goal of global registration is to bring the raw photogrammetry sequence $\{ \mathcal { R } ^ { t } \}$ into the canonical coordinate space defined by the subject neutral mesh $\mathcal { M } _ { \mathrm { n e u t r a l } }$ , so that subsequent per-frame optimization stages operate in a consistent space. We estimate a similarity transform $( \mathbf { R } _ { \mathrm { g } } , \mathbf { T } _ { \mathrm { g } } , s _ { \mathrm { g } } )$ via Procrustes alignment [22] on the neutral frame $t = 0 .$

Landmarks on $\mathcal { R } ^ { 0 }$ are obtained by rendering the raw scan with its appearance texture into a 2D image, applying a 2D landmark detector [63], and back-projecting detections to 3D by intersecting rays with $\mathcal { R } ^ { 0 }$ . Following [41], we exclude the first 17 landmarks corresponding to the jaw and chin contour, as these boundary points are inherently ambiguous due to self-occlusion and soft tissue deformation. The estimated transform is then applied to the full sequence $\{ \mathcal { R } ^ { t } \}$ , producing the canonically aligned sequence $\{ \widetilde { \mathcal { R } } ^ { t } \}$ used as the registration target for all subsequent stages.

## B.1. Regularizaton losses

The following regularization losses are used in Eq. 6 of the paper.

Part-Based Edge Preservation Loss. To preserve the local structure of the mesh and prevent vertex sliding effects, we use a part-based edge preservation loss, as in [8], that constraint the edge vectors of the optimized mesh $\mathcal { X } ^ { t }$ to the previous mesh $\bar { \chi ^ { t - 1 } }$

$$
\mathcal { L } _ { \mathrm { e d g e } } = \sum _ { e \in E } w _ { e } \| \mathbf { e } ^ { t } - \mathbf { e } ^ { t - 1 } \| ^ { 2 }\tag{1}
$$

where $E$ represents the set of mesh edges, $\mathbf { e } ^ { t }$ and $\mathbf { e } ^ { t - 1 }$ are edge vectors obtained from $\mathcal { X } ^ { t }$ and $\mathcal { X } ^ { t - 1 }$ respectively. $w _ { e }$ are spatially-varying weights that emphasize preservation in critical facial regions such eye region and lips.

Laplacian Loss. To promote smooth deformations and penalize irregular or jaggy surface areas, especially for regions known to be challenging for photogrammetry reconstruction, such as eyebrows and eyelids, we incorporate a Laplacian regularization:

$$
\mathcal { L } _ { \mathrm { l a p } } = \Vert \delta _ { i } - \delta _ { i } ^ { \mathrm { n e u t r a l } } \Vert ^ { 2 }\tag{2}
$$

where $\delta _ { i }$ is the Laplacian coordinate for vertex $i ,$ and $\delta _ { i } ^ { n e u t r a l }$ is the corresponding Laplacian coordinate in the neutral mesh $\mathcal { M } _ { \mathrm { n e u t r a l } }$ The Laplacian coordinates effectively encode the local differential properties of the mesh, and preserving these properties leads to natural deformations.

Normal Consistency Loss. To encourage smooth surface reconstruction and consistent face orientations, we apply a normal consistency loss to the reconstructed mesh $\mathcal { X } ^ { t }$

$$
\mathcal { L } _ { \mathrm { n c } } = \sum _ { ( f _ { 0 } , f _ { 1 } ) \in \mathcal { E } } \left( 1 - \cos ( { \bf n } _ { 0 } , { \bf n } _ { 1 } ) \right) ,\tag{3}
$$

![](images/cffdafc0c531241347425b773284eed9b3c843309cdde17fd2f471ba6ad4bc85.jpg)  
Figure 1. Ablation study demonstrating the importance of the lip contour loss for accurate lip tracking. From left to right: Input image, result with lip contour loss, result without lip contour loss. Note the improved lip closure and contour preservation when using the lip contour loss.

where $f _ { 0 }$ and $f _ { 1 }$ are adjacent triangular faces sharing an edge, n<sub>0</sub> and ${ \bf n } _ { 1 }$ are their respective face normals and E is the set of all adjacent face pairs.

## C. Implementation Details

Our FaceSnap framework is implemented in PyTorch with CUDA enabled GPU.

## C.1. Multi-stage Optimization

For the multi-stage optimization (Sec. 3.1 in the paper), we use gradient descent across the the different steps (base fitting and vertex refinement). For base fitting stage (refer to Eq. 6 in the paper), we perform 100 gradient descent step, with learning rate equal to 0.01, with the following regularization terms: $\lambda _ { 1 } ^ { \mathrm { b a s \bar { e } } } ~ = ~ 1 , \lambda _ { 2 } ^ { \mathrm { b a s e } } ~ = ~ 1 e ^ { - 0 6 } , \lambda _ { 3 } ^ { \mathrm { b a s e } } ~ =$ $2 , \lambda _ { 4 } ^ { \mathrm { b a s e } } = 1 0 0 0 0 , \lambda _ { 5 } ^ { \mathrm { b a s e } } = 2 . 0$ . For the vertex refinement stage (refer to Eq. 6 in the paper), we perform 300 gradient descent step, with learning rate equal to 0.0001 with the following regularization losses: $\lambda _ { 1 } ^ { \mathrm { v e r t } } = 1 , \lambda _ { 2 } ^ { \mathrm { v e r t } } = 1 e ^ { - 0 6 }$ $\lambda _ { 3 } ^ { \mathrm { v e r t } } = 2 , \lambda _ { 4 } ^ { \mathrm { v e r t } } = 2 0 0 0 0 , \lambda _ { 5 } ^ { \mathrm { v e r t } } = 2 . 0$ . The threshold for the automatic keyframe detector is $\tau _ { f } = 0 . 1 8$ cm. For the adaptive point-to-surface loss, we use $\tau _ { s } = 0 . 2$ cm and $\sigma =$ 1.

## C.2. Personalized tracking Model

Geometry Model: For the geometry model (Sec. 3.2.1 in the paper), we use a lightweight ResNet18 encoder, trained for 100 epochs with Adam optimizer $( \beta _ { 1 } ~ = ~ 0 . 9 , \beta _ { 2 } ~ =$ 0.999), with a learning rate equal to $1 \times 1 0 ^ { - 4 }$ , and a batch size of 8.

Dynamic Appearance Model: The model consists of two stages; low resolution prediction and high resolution upscaling.

Expression-Conditioned Texture Synthesis (Low Res.) (see Sec. 3.2.2, stage 1, in the paper) consists of two main components: a feature projection network and a convolutional decoder.

First, the model (Table 1) projects 256-dimensional PCA coefficients through three dense layers into a 2048- dimensional embedding. Then a convolutional decoder, progressively upsample from $8 \times 8 \ \mathrm { t o } \ 5 1 2 \times 5 1 2$ resolution (Table 2). Each stage uses transposed convolution for upsampling followed by regular convolutions with

![](images/edf25291a2ae8cb02ecb357825fbafac9fb88f39e958d9a167c62bb51992aad9.jpg)  
Figure 2. Ablation study demonstrating the importance of regularization losses. Top row: tracking results without regularization losses showing severe mesh degradation artifacts including inconsistent edge flows and surface discontinuities. Bottom row: tracking with ful regularization showing stable mesh topology and smooth deformations.

Table 1. Dense Feature Projection Layers
<table><tr><td>Layer</td><td>Input</td><td>Output</td><td>Activation</td></tr><tr><td>Linear 1 Linear 2</td><td>256 512 1024</td><td>512 1024</td><td>LeakyReLU LeakyReLU</td></tr></table>

Table 2. Convolutional Decoder Architecture (per branch)
<table><tr><td>Stage</td><td>Operations</td><td>Resolution</td><td>Channels</td></tr><tr><td>1</td><td> $\overline { { \mathrm { ~ C o n v T + C o n v } } }$ </td><td> $\overline { { 8 \times 8 \to 1 6 \times 1 6 } }$ </td><td> $\overline { { 3 2 \to 6 4 \to 1 2 8 } }$ </td></tr><tr><td>2</td><td> $\mathrm { C o n v T } + 2 \mathrm { \times C o n v }$ </td><td> $1 6 \times 1 6 \to 3 2 \times 3 2$ </td><td> $1 2 8 \to 1 2 8 \to 6 4 \to 1 9 2$ </td></tr><tr><td>3</td><td> $\mathrm { C o n v T } + 2 \mathrm { \times C o n v }$ </td><td> $3 2 \times 3 2 \to 6 4 \times 6 4$ </td><td> $1 9 2 \to 1 9 2 \to 9 6 \to 1 2 8$ </td></tr><tr><td>4</td><td> $\mathrm { C o n v T } + 2 \mathrm { \times C o n v }$ </td><td> $6 4 \times 6 4 \to 1 2 8 \times 1 2 8$ </td><td> $1 2 8 \to 1 2 8 \to 6 4 \to 6 4$ </td></tr><tr><td>5</td><td> $\mathrm { C o n v T } + 2 \mathrm { \times C o n v }$ </td><td> $1 2 8 \times 1 2 8 \to 2 5 6 \times 2 5 6$ </td><td> $6 4  6 4  4 2  3 2$ </td></tr><tr><td>6</td><td> $\mathrm { C o n v T } + 2 \mathrm { \times C o n v }$ </td><td> $2 5 6 \times 2 5 6 \to 5 1 2 \times 5 1 2$ </td><td> $3 2  4 2  2 1  3$ </td></tr></table>

BatchNorm and LeakyReLU activations. ConvT denotes ConvTranspose2d. All convolutional layers are initialized with weights from N(0, 0.0002) and biases uniformly distributed in [0.001, 0.0015]. BatchNorm weights are initialized with N(1, 0.02) and biases set to 0.01. The model is trained using the Adam optimizer with a learning rate of $5 \times 1 0 ^ { - 4 }$ and a batch size of 16.

High-Resolution Personalized Residual Upscaler: (see Sec. 3.3.2, stage 2 in the paper) Rather than predicting the full 4096 × 4096 UV texture directly, the model predicts a subject-specific residual offset that is added to a pre-computed neutral texture. The network (Table 3) takes as input a $5 1 2 \times 5 1 2$ low-resolution residual UV map; the difference between the current expression texture and the neutral texture and upscales it by a factor of 8× using a re-parameterizable convolutional backbone [6] followed by PixelShuffle. The backbone consists of a head block, four residual feature-refinement blocks, and a transition block with a skip connection from the head output, i.e. y = Transition(Backbone(y<sub>0</sub>)+y<sub>0</sub>), where $\mathbf { y } _ { 0 } = \mathrm { H e a d } ( \mathbf { x } )$ Each RepBlock [27] uses a 3×3 convolution through a 256- channel bottleneck followed by a 1×1 projection; at inference, both branches are folded into a single 3×3 convolution via structural re-parameterization, incurring no additional runtime cost. The model is trained with a masked L1 loss (restricted to the facial foreground region) using AdamW $( \mathrm { l r } = 5 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 9 9 )$ , weight decay 10<sup>−4</sup>) with cosine annealing and mixed-precision training.

Table 3. High-Resolution Residual Upscaler Architecture
<table><tr><td>Stage</td><td>Module</td><td>Channels</td><td>Resolution</td></tr><tr><td>Head</td><td>RepBlock (ReLU)</td><td> $\overline { { 3 \to 6 4 } }$ </td><td> $\overline { { 5 1 2 \times 5 1 2 } }$ </td></tr><tr><td>Backbone</td><td>4× RepBlock (ReLU)</td><td> $6 4  6 4$ </td><td> $5 1 2 \times 5 1 2$ </td></tr><tr><td>Transition</td><td>RepBlock (linear)</td><td> $6 4  1 9 2$ </td><td> $5 1 2 \times 5 1 2$ </td></tr><tr><td>Upsample</td><td>PixelShuffle (×8)</td><td> $1 9 2  3$ </td><td> $5 1 2 \times 5 1 2 \to 4 0 9 6 \times 4 0 9 6$ </td></tr></table>

![](images/d5564ec55f77f1c86272b217c88424584a7d33927a9248e27614a02fa63f74e6.jpg)  
Figure 3. Multi4D benchmark camera setup: Sample images from the 38 cameras with different expressions used to run photogrammetry to obtain the raw scans used as ground truth for evaluation.

## D. Benchmark Design

Producing the data: We design the Multi4D benchmark to enable direct comparison in the topology space of each method without requiring retargeting, establishing a standardized protocol for geometry evaluation. Multi4D represents the first publicly available benchmark designed to evaluate 4D facial capture methods on range of motion sequences, facilitating fair comparisons between approaches with different underlying topologies. To obtain the raw meshes used as ground truth, we run a commercial photogrammetry tool [23] on 38 camera views from the Multiface dataset [58], covering 360 degrees around the full head with most cameras focused on the front and side views (please refer to Section 5 in the paper). This comprehensive multi-view reconstruction provides high-fidelity reference geometry for evaluation.

Metric calculation: We evaluate every other frame for efficiency and sample 300,000 points uniformly from each raw scan. Points are clipped to a subject-specific axis-aligned bounding box covering the frontal facial region (forehead, eyes, ears, cheeks, and jaw). The bounding box is defined as six scalars [x<sub>min</sub>, x<sub>max</sub>, y<sub>min</sub>, y<sub>max</sub>, z<sub>min</sub>, z<sub>max</sub>] in raw scan space and is set per subject; the exact values are provided in the released evaluation code. Point-to-surface distance is then computed between these clipped points and each method’s reconstructed mesh. The full protocol described in Appendix E applies to both the Multi4D benchmark (Sec. 5.1) and the expressive sequence evaluation (Sec. 5.2).

We note that for Topo4D [43] evaluation, we exclude cameras capturing back views since their method only optimizes for the frontal facial region. For the FLAME fitting method [41], we employ an off-the-shelf landmark detector [63] to extract facial landmarks that are used by FLAME fitting process to rigidly aligning raw scans to their canonical coordinate system. Finally, we publicly release the evaluation source code and the ground-truth raw scans (obtained via photogrammetry) for each ROM sequence of the five subjects.

## E. Point-to-Surface Evaluation Protocol

Both geometry evaluations (Sec. 5.1 and Sec. 5.2) follow the sampling and metric protocol described above. The key additional consideration for Sec. 5.2 is method alignment. Alignment to metric space. For per-frame multiview optimization methods (FLAME Fitting, Topo4D, and our optimization), the known camera parameters bring each reconstructed mesh directly into raw scan space. For tracking-based methods (SEREP, DECA, FLAME BACKBONE, TOPO4D BACKBONE, and Ours), direct ICP to the raw scan is unreliable due to topological differences. Instead, each tracking method is aligned to its corresponding per-frame optimization result using point-topoint ICP (SVD-based, 200 iterations) on the first frame, then applied rigidly to all frames: SEREP, DECA, and Ours align to our optimization; FLAME BACKBONE aligns to FLAME Fitting; TOPO4D BACKBONE aligns to Topo4D.

## F. Limitations and future work

Although FaceSnap achieves fast and efficient facial performance capture and outperforms state-of-the-art methods in both geometric accuracy and texture appearance quality, several aspects could benefit from further improvements. Our dynamic appearance model, optimized for real-time performance using a simple CNN architecture, operates at 512×512 resolution and may lose fine details such as microwrinkles during the decoding process, which is a notable constraint compared to offline methods that can operate at higher resolutions. While the proposed personalized upscaler recovers recovers most fine details in 4K, regions near forehead are challenging.

Additionally, the lip region presents particular challenges for our method, exhibiting subtle temporal artifacts due to occasional errors in photogrammetry or the automated face segmentation. Range of Motion (ROM) capture requires the subject to return to neutral state periodically to allow our automatic keyframing to detect neutral poses and reset the tracking to avoid drift. Finally, our evaluation is limited in subject diversity: optimization geometry is evaluated on Multi4D (6 subjects, Table 1, main paper), while monocular tracking and appearance are evaluated on our custom lightstage dataset (Table 2 and 3 using 3 subjects, 3 expressive sequences each, 9 sequences total). Broader validation across skin tone, age, and facial hair is needed to confirm generalization, particularly for the real-time tracking component; this is primarily constrained by the cost and limited availability of lightstage capture sessions rather than a methodological limitation.

As future work, we aim to incorporate facial intrinsic recovery to enable relighting by decomposing skin reflectance from light. Furthermore, enhancing our dynamic appearance model to produce higher resolution outputs with correctives to avoid temporal artifacts while maintaining the real-time performance aspect—which was a core design choice—represents an important direction for future development. Expanding evaluation to a larger and more demographically diverse subject pool is another important direction we leave for future work.

## G. Topo4D Initialization

Topo4D [43] requires that the first frame of the sequence be perfectly aligned and deformed to match the raw scan before initiating their optimization process. To achieve this initialization, we use a commercial tool Wrap4D [24], the same tool used by the original paper, to obtain a registered mesh and 8K texture for the first frame which is then used to start the optimization process. The wrapping process involves loading the raw scan and Topo4D template mesh, excluding certain polygons corresponding to eyes and mouth interior from registration. Wrap4D performs a two-stage alignment: first rigid alignment to transform the raw scan into the coordinate system of the template mesh, followed by non-rigid deformation to accurately fit the template to the raw scan geometry. Following the authors’ recommendations, we scale the wrapped mesh to match Topo4D’s template mesh scale and adjust camera calibration accordingly.