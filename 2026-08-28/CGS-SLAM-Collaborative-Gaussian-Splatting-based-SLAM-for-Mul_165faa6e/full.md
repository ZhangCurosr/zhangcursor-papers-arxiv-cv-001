# CGS-SLAM: Collaborative Gaussian Splatting based SLAM for Multi-Agent Reconstruction

Jean-Daniel de Ambrogi<sup>1,2</sup> , Aladine Chetouani<sup>1</sup> , Vincent Nguyen<sup>3</sup> , and Aurélien Chateigner<sup>2</sup>

<sup>1</sup> Université Sorbonne Paris Nord, L2TI, UR 3043, F-93430, Villetaneuse, France <sup>2</sup> SAS IMPACT, Orléans, France

<sup>3</sup> Université d’Orléans, INSA CVL, LIFO, UR 4022, Orléans, France

Abstract. Recent advances in SLAM have leveraged 3DGS for photorealistic reconstruction and novel view synthesis. However, most methods rely on RGB-D input, which is unavailable on consumer-grade smartphones, and few integrate 3DGS within a collaborative framework. Therefore, we present CGS-SLAM, a hybrid decentralized/centralized system enabling multi-agent 3DGS SLAM using only RGB and inertial data. Each agent performs local tracking with inertial data as a motion prior and reconstructs a scaled map using a metric monocular depth estimator (Depth Pro). Keyframe encodings are shared among agents, enabling dynamic keyframing in regions of spatial overlaps with other agents, enhancing submap alignment. Afterwards, a central server aligns submaps using VGGT as a view alignment model. This bidirectional communication keeps communication cost low during mapping and global reconstruction in dificult GNSS-denied environments. Experiments on multiple datasets demonstrate competitive tracking performance, improved rendering quality over state-of-the-art methods, and accurate submap alignment.

Keywords: VI-SLAM · Gaussian Splatting · Collaborative SLAM

## 1 Introduction

In high-stakes operational scenarios such as hostage rescue, disaster response, or reconnaissance missions, law enforcement and military units are frequently required to navigate unfamiliar, GNSS<sup>4</sup>-denied environments under time-critical constraints. These operations often involve rapid deployment across confined or structurally complex spaces, with minimal prior knowledge of the terrain and coordination among multiple, concurrently operating teams. To enable efective decision-making, there is a need for a rapid mapping system able to fuse the local contributions of each team into a globally consistent, shared representation.

This problem falls within the domain of Simultaneous Localization and Mapping (SLAM), which enables reconstruction of spatial environments while concurrently localizing the exploratory agent within them. However, most existing

SLAM systems rely on explicit depth priors [9], either from dedicated depth sensors or via dense structure-from-motion (SfM) pipelines [28, 56], to compensate for the inherent ambiguity of monocular vision. Although efective, these approaches often incur significant computational overhead, rendering them impractical for deployment on resource-constrained platforms, limiting live operation, since they require both time and parallax-inducing motion at initialization.

In practice, the current electronic equipment available to tactical units is ruggedized consumer-grade smartphones, limited to RGB cameras and inertial measurement units (IMUs). They typically do not provide any depth measure through LiDAR or Time-of-Flight (ToF) cameras. This sensor constraint narrows the design space for viable SLAM solutions, as it precludes the use of active depth sensors.

Furthermore, many existing systems output point clouds or mesh-based reconstructions, both of which are sensitive to trajectory coverage. In scenarios where exploration is sparse, these representations often exhibit significant topological incompleteness and are hard to interpret. Recent advances in 3D Gaussian Splatting (3DGS) [19] ofer an alternative. While it is not immune to sparse coverage either, its continuous, opacity-weighted formulation does not require a topological decision in unobserved regions, unlike an explicit surface, and with its photorealistic representations and fast rendering it allows easier interpretation.

We present CGS-SLAM, a new Visual-Inertial SLAM (VI-SLAM) approach targeting this specific sensor-constrained regime. Our method reconstructs dense, continuous local maps on each agent, which are then aligned on a server using a sparse reconstruction model to produce a globally consistent shared map enabling real-time situational awareness for task force teams, even in GNSS-denied, resource-constrained environments.

Our main contributions, forming a global SLAM solution, are:

– A monocular inertial local SLAM system, leveraging a metric monocular depth estimation model, producing metrically scaled 3DGS maps from the very first frame.

– A light-weight collaborative framework for constrained networks, ofering dynamic keyframing capabilities for enhanced sub-map alignment.

– A rapid submap alignment process, leveraging VGGT to produce an immediate pose estimation of local submaps.

## 2 Related Work

## 2.1 Gaussian Splatting

Accurate, robust, and semantically meaningful environmental representation lies at the core of SLAM systems. Researchers thus increasingly turned to neural representations as the foundational primitive for scene reconstruction. 3D Gaussian Splatting (3DGS), introduced by Kerbl et al. [19], has rapidly transformed the field of 3D scene representation. It has found applications in mainstream industries such as film making [14] or medical education [30] due to its ability to provide photorealistic rendering at high frame rates. Several recent works [10, 15, 29, 43, 48–50] have shown the feasibility of using 3DGS as a primary mapping representation in SLAM systems. However, most of these methods are heavily dependent on depth priors. While Sun et al. [43] proposed an RGB-only SLAM configuration, their method relies on relative monocular depth estimation, which provides inconsistent estimations during exploration and thus can lead to catastrophic map collapse. In our work, motivated by the need for a real-time rendering method, we adopt 3DGS as our core 3D representation. Further, we replace depth sensors with a recent state-of-the-art metric monocular depth estimation model (MMDE), Depth Pro [5], that provides consistent at scale depth predictions, enabling accurate 3DGS reconstruction without LiDAR, ToF or stereo cameras.

## 2.2 Monocular SLAM

Given that most consumer devices lack the capability to capture explicit depth priors, SLAM research has increasingly focused on monocular-only solutions [9, 18, 25]. Early approaches relied on feature-based visual methods, extracting and matching hand-crafted image features for pose estimation and mapping [7, 20]. However, these methods often exhibited limited robustness in environments with sparse texture, varying illumination, or rapid camera motion, which adversely afected depth estimation accuracy under such conditions. Subsequent works [6, 24] have replaced traditional feature extractors with deep learning-based alternatives. Some have bypassed explicit feature matching altogether, regressing camera poses directly using end-to-end neural networks. More recently, advances in monocular depth estimation (MDE) have been integrated into SLAM pipelines [27,43,51], enabling the replacement of external depth priors with depth maps predicted by specialized deep models. While these approaches have demonstrated success in producing geometrically coherent reconstructions, they typically sufer from a lack of global scale consistency, as MDE models inherently provide only relative depth with respect to the input image. Furthermore, dynamic scenes, illumination changes, or motion-induced artifacts can cause temporal inconsistencies in predicted depth maps across frames, leading to erroneous inputs for the mapping module and compromising overall trajectory and structure estimation. To mitigate these issues, we rely on the metric scale of our MMDE, thus improving the temporal consistency of depth prediction during exploration.

## 2.3 Collaborative SLAM

Collaborative or multi-agent SLAM, which enables multiple agents to simultaneously localize and map an environment through information exchange, has attracted increasing interest but remains a challenging and relatively underdeveloped field compared to single-agent SLAM, due to technical complexities such as inter-agent data association, communication constraints, and map merging [22]. While collaboration has been investigated using traditional map representations such as point clouds or meshes, few approaches utilize neural representations. Hu et al. [16] enabled collaborative mapping through a neural point-based representation anchored to camera keyframes, fused across agents by a distributedto-centralized learning scheme. Yugay et al. [50] proposed a multi-agent SLAM system based on Gaussian Splatting, achieving high-fidelity results with multiple agents; however it relies on RGB-D inputs requiring LiDAR sensors. Gao et al. [10] introduced a collaboration strategy for wide-area scene capture using Unmanned Aerial Vehicles (UAV), focusing on stitching together local maps from each UAV. Nonetheless, this method assumes the availability of GNSS data and does not address the case where it is absent, thus leaving the SLAM tracking problem unaddressed. Finally, Xu et al. [48] proposed an RGB-D-based method leveraging Multi-Agent Gaussian Consensus achieving global consistency. However, it relies on intensive communication between agents, aiming for global consensus on overlapping areas and continuously refining Gaussians. Moreover it aligns submaps through Mahalanobis distance on point clouds, thus relying heavily on accurate mapping produced from RGB-D input.

## 3 Method

![](images/e1a7bb60e3c5ae3f061a263cd75318cb40fb45c96e261004251b9aa84b2241f2.jpg)  
Fig. 1: The global architecture of our framework. Our method allows multiple agents to evolve into an unknown environment and locally reconstruct it. The keyframes encodings are sent to the server, which broadcast them to all agents. Afterwards all agents share their local maps to the server which merge them into a single global map.

We propose a hybrid decentralized–centralized architecture for light-weight collaborative Gaussian-Splatting SLAM. An overview of our framework is illustrated by Fig. 1. Each agent independently reconstructs a local map from its own limited sensor suite, employing a decentralized approach while periodically exchanging keyframe encodings through a central server, allowing a dynamic keyframing process. After all agents finish their exploration, the centralized server fuses the individual submaps into a globally consistent map.

In the sequel, we first describe the environment perception method, through IMU and depth estimation. Then we explore our framework for the local reconstruction method. Finally we detail the global consistency approach through our alignment pipeline.

## 3.1 Environment Perception

Sensor Fusion In a monocular setup, where no reliable depth information is available, a basic approach is to assume constant velocity between consecutive frames [1,8,11,36,44] as an initial motion estimate. Methods then rely on visual clues to optimize the pose of the camera through dense structure from motion [7,13]. However, this assumption is not robust and can introduce significant errors in the predicted pose, especially in the presence of rapid camera movements or when the frame rate is low. Visual Inertial SLAM approaches [4, 7, 23, 32] have been proposed to tackle these limitations by leveraging IMU data as a more accurate a priori. We adopt this approach with the aim of using all sensors that can help our method to have a better perception of the scene, and then estimate an accurate prior.

Thus, before optimizing the camera pose for each new frame, we compute an initial pose estimate from IMU preintegration. The method to integrate and fuse these sensor signals is detailed in the supplementary material. Considering that the IMU sensors have a significantly higher sensing frequency than the RGB camera, we chain the relative transformation throughout all IMU measurement up to the timestamps of the new tracked RGB frame. Finally the camera pose $\mathbf { P } _ { j }$ at the timestamp of the new RGB frame is estimated from the previous camera pose $\mathbf { P } _ { i }$ through:

$$
\mathbf { P } _ { j } = \left( \prod _ { t = i } ^ { j } t ^ { { t - 1 } } \mathbf { T } _ { C } \right) \mathbf { P } _ { i }\tag{1}
$$

Where i denotes the timestamp of the last tracked camera frame, j represents the timestamps of the new tracked frame and $_ { t } ^ { t - 1 } \mathbf { T } _ { C }$ is the translation matrix between the two consecutive timestamps t and $t - 1$

Monocular Depth Estimation In order to circumvent the need for scale recovery through post-hoc alignment or global optimization, we employ in our experiments a metric-MDE (MMDE). The metrical dimension of the model also ensures temporal consistency across frames where relative depth estimates may drift due to exposure variations or motion blur. At each keyframe (selection detailed in Sec. 3.2), depth is inferred using this model and subsequently integrated into the SLAM pipeline as a proxy for true metric depth. This approach eliminates the reliance on batch-based methods that require accumulating multiple keyframes to perform deep bundle adjustment, thereby enabling truly online operation from the very first frame while maintaining geometric consistency.

## 3.2 Local Reconstruction

Each agent constructs its local 3D map using synchronized RGB video and IMU data. As described in Sec. 3.1, we fuse these complementary signals from the IMU to initialize the camera pose $P _ { i } ,$ which serves as a coarse pose prior during the tracking phase, described in the next subsection. During this phase, we align in a frame-to-model manner the tracked pose to the Gaussian map G. Afterwards a frame is promoted to keyframe status if it fulfills certain conditions detailed below and then contributes to the map G through projections of pixels using an MMDE described in Sec. 3.1. We then optimize the map G through Eq. (5). Recent keyframes are stored in a sliding window for local bundle adjustment, and jointly optimized to preserve geometric consistency over time.

Upon completion of local exploration, the agent transmits its local map, including keyframes, Gaussian parameters, and camera intrinsics, to the server for global reconstruction described in Sec. 3.3

Tracking Since tracking quality depends on the geometric accuracy of the map, we complement the photometric term with a depth term. The metric model bounds the variation of the predicted depth range across consecutive estimates, but does not remove it entirely; we therefore compute the depth loss using the Pearson correlation coeficient (Eq. (2)) between the estimated depth map $D _ { e }$ and the rendered depth map $D _ { r }$ . Sun et al. [43] reported the benefit of this approach over an L1 depth term; we adopt their approach.

$$
\mathcal { L } _ { \mathrm { d e p t h } } = 1 - \rho ( D _ { e } , D _ { r } ) = 1 - \frac { \mathrm { C o v } ( D _ { e } , D _ { r } ) } { \sqrt { \mathrm { V a r } ( D _ { e } ) \mathrm { V a r } ( D _ { r } ) } } .\tag{2}
$$

During Tracking, the Gaussian map is held constant, and the camera pose is optimized using the following composite loss function:

$$
\mathcal { L } _ { \mathrm { t r a c k } } = \mathrm { M a s k } _ { O ( G , T _ { c } ) } \left( \mathcal { L } _ { \mathrm { p h o t o } } + \lambda _ { D } \mathcal { L } _ { \mathrm { d e p t h } } \right) , \mathrm { a n d } 0 \leq \lambda _ { D } \leq 1\tag{3}
$$

where $\mathcal { L } _ { \mathrm { p h o t o } }$ is the L1 loss between the ground truth and the rendered image and $\mathrm { M a s k } _ { O ( G , T _ { c } ) }$ is a masking function defined as:

$$
\mathrm { M a s k } _ { O ( G , T _ { c } ) } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \ O ( G , T _ { c } ) > 0 . 9 9 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{4}
$$

This function keeps pixels whose computed opacity $O ( G , T _ { c } )$ is greater than 0.99, a threshold we found efective to retain only high opacity regions during the optimization.

Finally, tracking is not applied to the first frame, as no Gaussian map has yet been constructed. Instead, the identity matrix is used as the initial camera transformation. The pose tracking is thus relative to the position of the first frame.

## Mapping

Initialization Upon the first frame, pixels are back-projected into 3D space using the initial depth estimate provided by the metric depth model. This yields a scale-consistent initialization of the scene geometry, anchoring the mapping process to a metric reference frame. The 3DGS technique used in our method is detailed in the supplementary material.

Keyframe Selection The Gaussian representation is incrementally densified as new keyframes are selected based on a multi-criterion information gain metric. Specifically, it is designed to balance geometric coverage, temporal consistency, and computational eficiency. A new keyframe is instantiated according to the following conditions, checked in the following order:

– Spatial displacement: To enforce temporal sparsity and avoid excessive keyframe insertion during slow or static motion, we enforce a minimum frame interval between consecutive keyframes. Thus, if the frame index is diferent from the last keyframe index by less than s frames, the frame is discarded. Otherwise, a new keyframe is triggered if the current frame index exceeds the index of the last keyframe by at least k frames. This ensures that keyframes are spaced suficiently apart in time, reducing computational overhead while maintaining suficient temporal resolution for robust tracking and mapping. Suficient information gain: To ensure that each new keyframe contributes meaningful reconstruction information, we evaluate the spatial overlap between the current view and the most recent keyframe. Specifically, we render the depth map from the latest keyframe’s pose using the current Gaussian representation and identify pixels with valid depth and high silhouette confidence (i.e., presence mask $> 0 . 9 9 )$ . These pixels are back-projected into 3D space to form a point cloud, which is then tested for co-visibility under the current camera pose. If the fraction of co-visible Gaussians does not exceed a predefined threshold (here 90%), the frame is promoted to keyframe status.

Dynamic keyframing: Throughout exploration, each agent continuously receives, through the server, the keyframe encodings computed by the other agents (Sec. 3.3). When an agent traverses an area previously mapped by another one, the encoding of the incoming frame exhibits high similarity to one of these received encodings. We leverage this similarity as an indicator of spatial overlap between local reconstructions, and dynamically increase the keyframe sampling density within the overlapping region to enhance global consistency during subsequent map fusion.

Map Optimization The previously selected keyframes are retained within a sliding window of fixed extent, over which periodic map optimization is performed to mitigate catastrophic forgetting and preserve geometric consistency throughout the reconstruction.

With $\mathcal { L } _ { \mathrm { p h o t o } }$ the same loss as in $\operatorname { E q . } \ ( 3 )$ and $\begin{array} { r } { \mathcal { L } _ { \mathrm { D - S S I M } } = \frac { 1 - \mathrm { S S I M } } { 2 } } \end{array}$ , the optimization of the Gaussian representation is governed by the following composite loss function:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { m a p p i n g } } = \lambda _ { C } \mathcal { L } _ { \mathrm { p h o t o } } + \lambda _ { S } \mathcal { L } _ { \mathrm { D - S S I M } } + \lambda _ { D } \mathcal { L } _ { \mathrm { d e p t h } } , } \\ & { \quad \quad \quad \quad \mathrm { w i t h } \ \lambda _ { C } , \lambda _ { S } , \lambda _ { D } \in ( 0 ; 1 ) \ \mathrm { a n d } \ \lambda _ { C } + \lambda _ { S } + \lambda _ { D } = 1 } \end{array}\tag{5}
$$

Following [19, 43], we employ the structural dissimilarity index (D-SSIM) [47] as a perceptual regularization term to enhance rendering fidelity.

Finally, a local bundle adjustment is performed over the keyframes retained in the sliding window. This step jointly refines camera poses and Gaussian parameters to ensure geometric coherence, efectively adapting the trajectory and map structure to the evolving scene representation.

## 3.3 Global Consistency

Dynamic Keyframing. Submap alignment relies on the existence of keyframe pairs observing a common region. Our agents actively bias their keyframe distribution towards such regions: each agent transmits the encoding of every new keyframe to the server, which caches them and broadcasts them to all other connected agents. Each agent matches incoming encodings against its own local keyframe database, and a match triggers the densification policy of Sec. 3.2. Only fixed-size descriptors are exchanged; the mechanism therefore does not require a shared coordinate frame. The efect on alignment is twofold. Overlapping regions yield more candidate pairs for the keyframe association step described below, and each candidate pair comes with a denser set of temporal neighbours, which better conditions the subsequent pose refinement. The message protocol and the payload format are detailed in the supplementary material.

Submap Alignment Upon receiving the local map of each client (including RGB keyframes), the server initiates global map reconstruction by identifying pairs of co-visible keyframes across distinct local maps. These corresponding keyframe pairs are established using NetVLAD [2] visual descriptors. Keyframes with similar NetVLAD embeddings are considered candidates for spatial correspondence, under the assumption that visually similar frames are likely to capture overlapping regions of the environment. If no similar frames are found between two submaps, these submaps are not merged.

For each candidate pair $\mathrm { ( c l i e n t _ { 1 } , c l i e n t _ { 2 } ) }$ , we extract the keyframe images pairs, as well as several neighboring keyframes (from our experiments we choose 6 per submap, 3 before and after the keyframe). We feed them into VGGT [46] to obtain camera extrinsic matrices $\mathbf { E } _ { i } \in \mathbb { R } ^ { 4 \times 4 }$ , which map camera coordinates to a specific and temporary VGGT world frame. These extrinsics are used with the locally tracked poses $\mathbf { T } _ { C _ { i } } ^ { S _ { i } }$ (camera → submap) to compute an estimation of an initial rigid transformation $\mathbf { T } _ { S _ { 2 } } ^ { S _ { 1 } }$ that aligns the second submap into the first’s coordinate frame. This transformation estimation is detailed in the supplementary material.

Since sub-maps from our front-end are metric-scale while VGGT operates up to relative scale, $\mathbf { T } _ { S _ { 2 } } ^ { S _ { 1 } }$ contains the relative scale diference. We decompose $\mathbf { T } _ { S _ { 2 } } ^ { \bar { S _ { 1 } } }$ into rotation R, translation t and scale s, which is discarded, retaining only the rigid transformation $( \mathbf { R } , \mathbf { t } ) \in S E ( 3 )$

This initial transformation is then refined via gradient-based optimization over the full set of neighboring keyframes. The optimization minimizes the geometric error between corresponding camera poses under the current transformation:

$$
\mathcal { L } ( \mathbf { T } ) = \sum _ { j } \big \| \mathbf { T } \cdot \mathbf { P } _ { j } - \mathbf { Q } _ { j } \ \big \| ^ { 2 } ,\tag{6}
$$

where $\mathbf { P } _ { j }$ are the source poses (transformed VGGT poses) and $\mathbf { Q } _ { j }$ are the target poses (SLAM poses). The optimization is performed over the 6-DoF parameters of T. Finally, this pose-based estimate is refined by a residual rigid correction, initialized at identity and optimized with Adam through the differentiable rasterizer, which minimizes the depth discrepancy between the two sub-maps rendered from a common keyframe pose.

Merging Submaps Once the optimal transformation $\mathbf { T } _ { S _ { 2 } } ^ { S _ { 1 } }$ is computed, the second client’s Gaussian model is transformed into the first’s coordinate frame via:

$$
\mathbf { G } _ { 2 } ^ { \prime } = \mathbf { T } _ { S _ { 2 } } ^ { S _ { 1 } } \cdot \mathbf { G } _ { 2 } ,\tag{7}
$$

Applying $\mathbf { T } _ { S _ { 2 } } ^ { S _ { 1 } }$ to every Gaussian of client 2 brings its whole sub-map into the reference frame of client 1.

Once aligned, each local Gaussian map is incrementally merged into the global map using the computed transformation without further refinement to keep the process fast, at the cost of duplicated Gaussians on overlapping regions. All submaps are merged into the reference frame of the first completed and received local map, and the process is repeated as new local maps are received.

## 4 Experiments

## 4.1 Datasets

A wide range of datasets have been proposed to benchmark SLAM [3, 12, 17, 21, 34, 35, 37, 41, 43, 45, 54], each tailored to specific sensor configurations, motion profiles, and environmental conditions. However, only a limited subset of these datasets provide synchronized monocular video sequences paired with IMU measurements under handheld operation in indoor environments. Multi-agent datasets, such as the ones derived from [38, 40], do not provide synced IMU data. Testing our pipeline on these datasets with constant velocity assumption would contaminate the evaluation, and using depth data would invalidate our lightweight-sensing promise. To evaluate our method, we select two datasets providing IMU, TUM RGB-D Dataset [41] and UT-MM Dataset [43]. While TUM RGB-D Dataset ofers RGB-D sequences with ground-truth trajectories and accelerometer data, UT-MM Dataset consists of multiple RGB-D sequences with synchronized IMU sensors. Although it does not provide explicit multi-agent trajectories, all sequences were recorded within the same physical environment, enabling synthetic multi-agent evaluation.

## 4.2 Experimental Set-up

All experiments were conducted on a consumer-grade laptop, a Lenovo ThinkPad T15g Gen 1 equipped with an NVIDIA GeForce RTX 2080 Max-Q GPU and an Intel® Core™ i7-10750H CPU. This hardware configuration achieves a sustained frame rate of 10 FPS during tracking and mapping tasks. The client-side tracking and mapping module operates within approximately 1 GB of VRAM, making it suitable for deployment on portable workstations.

We use Depth Pro [5] as our MMDE for its depth quality. But this choice was at the cost of ofloading it to the centralized server as it is too heavy for our laptop. Our pipeline is nonetheless MMDE-agnostic: we additionally evaluated Depth Anything v3, which replaces both the MMDE and the view alignment model for the submap alignment (VGGT). But it did not match the combination of Depth Pro and VGGT, results are available in the supplementary material.

The centralized server was equipped with an NVIDIA RTX 4090 GPU, producing depth estimation with Depth Pro at ∼169 ms per keyframe. The keyframe-encoding payload (described in the supplementary material) was 65 KB at 1–2 keyframes/s on UT-MM, yielding <130 KB/s uplink per agent, making it suitable for exchanges under degraded network conditions. The final maps were between 25 and 75 MB.

## 4.3 Performance of Monocular Single Agent Exploration

To evaluate the impact of integrating a metric depth model as a substitute for direct depth sensing, we conduct several evaluation comparisons. First, as illustrated in Fig. 2, the adoption of a modern metric depth estimation model yields qualitative improvements over data from consumer grade LiDAR which can be noisy. Furthermore, the metric depth model enables an accurate scaling of the

![](images/ed8f1c66e95af50aff589bdb3d6e2d7c972167e6623f52120354a0078f80ba2e.jpg)

![](images/e49845a2ab3345bffb30e9f618335a9917de1fd8505916c8c3e57b755fde9635.jpg)

![](images/399fff8a6b0a41a196b625b307009d4c92e72a08a089989cc4727c9600f39f7a.jpg)  
Fig. 2: Comparison of the rendering between LiDAR (left), MiDaS [33] depth estimator (middle) and Depth-Pro (right) based Gaussian maps on scene fast-straight from UT-MM. Our method exhibits reduced linear distortion along the horizontal axis. Even when compared against LiDAR measurements, our method demonstrates superior consistency against sensor artifacts found in the left panel.

reconstructed scene. In Fig. 3 we compare the raw trajectory estimation between our method and MM3DGS, we can see that our method accurately predicts the scale of the trajectory, and is close to the GT. We also analyze the trajectory estimation performance of our framework under monocular conditions, specifically when no initial depth measurement is available to calibrate the scale of the first depth estimate. In such scenarios, the system must infer scale purely from visual cues and the predicted depth prior.

![](images/6632600c2c18ff3aa43184d675e1b9c2657ee2027837a25e97bb43a6d2363962.jpg)  
Fig. 3: Raw trajectory output (TUM, fr1/desk), i.e. when results are not rotated, scaled nor translated to fit at best the ground truth, as usually done to evaluate ATE-RMSE. We only align the origin. Our method ofers a trajectory almost at scale with the ground truth, while MM3DGS is way of with an overall size of a third of the GT.

Table 1 presents a comparative evaluation of ATE RMSE across multiple sequences under both RGBD, as reference, and monocular configurations. The results indicate that our approach achieves competitive tracking accuracy compared to methods that utilize LiDAR or RGBD inputs. In particular, we show competitive results with Magic-SLAM, the SOTA RGBD-based 3DGS SLAM method. This shows that a metric MDE can serve as a viable and efective replacement for direct depth sensing. To assess the visual fidelity of the recon-
<table><tr><td>Method</td><td>Map</td><td>Setup</td><td>fr1/desk</td><td>fr1/desk2</td><td>fr1/room</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg</td></tr><tr><td>iMAP [42]</td><td>Neural</td><td>RGBD</td><td>4.9*</td><td></td><td></td><td>2.0*</td><td>58*</td><td></td></tr><tr><td>NiceSLAM [55]</td><td>Neural</td><td>RGBD</td><td>4.26*</td><td>4.99*</td><td>34.49*</td><td>31.73*</td><td>30*</td><td>21.09</td></tr><tr><td>CP-SLAM [16]</td><td>Neural</td><td>RGBD</td><td> $7 . 8 4 ^ { * }$ </td><td></td><td></td><td>3.93*</td><td></td><td></td></tr><tr><td>Go-SLAM [53]</td><td>Neural</td><td>RGB</td><td>1.6*</td><td>2.8*</td><td>5.2*</td><td>1.0*</td><td>2.23</td><td>2.56</td></tr><tr><td>GS-SLAM [49]</td><td>3DGS</td><td>RGBD</td><td>3.3*</td><td></td><td></td><td>1.3*</td><td></td><td></td></tr><tr><td>MAGIC-SLAM [50]</td><td>3DGS</td><td>RGBD</td><td>4.71</td><td>3.91</td><td>14.45</td><td>1.15</td><td>15.64</td><td>7.97</td></tr><tr><td>Mac-Ego3D [48]</td><td>3DGS</td><td>RGBD</td><td>25.53</td><td>30.81</td><td>22.50</td><td>1.39</td><td>11.62</td><td>18.37</td></tr><tr><td>MM3DGS [43]</td><td>3DGS</td><td>RGB</td><td>3.35</td><td>6.54</td><td>5.8</td><td>1.24</td><td>231.42</td><td>49.67</td></tr><tr><td>Ours</td><td>3DGS</td><td>RGB + IMU</td><td>4.2</td><td>5.12</td><td>5.3</td><td>1.14</td><td>20.42</td><td>7.24</td></tr></table>

Table 1: Tracking results against SOTA neural methods (top) and 3DGS based methods (bottom) on TUM Dataset (ATE-RMSE ↓cm). Results marked with \* are provided by the original paper. Our method does not use depth measurements and still performs as well as the RGBD-based 3DGS SLAM methods. The first , second , and third ranks are highlighted accordingly on 3DGS based SLAM; Go-SLAM provides the best results on all scenes.

structed scenes, we compute the PSNR, SSIM [47] and LPIPS [52] between the original keyframes and their corresponding renderings generated by the final Gaussian map, evaluated at the estimated camera poses. As summarized in Tab. 2, our approach, which employs a metric MDE, consistently outperforms the monocular Gaussian splatting based SLAM system MM3DGS on all evaluated sequences and matches the RGB-D baselines on most, Magic-SLAM remaining ahead on fr1/room. Fig. 4 illustrates this rendering quality through visual is achieved without compromising tracking accuracy. As previously established, our method maintains competitive trajectory estimation performance relative to other neural depth based approaches. Thus, the integration of a metric depth model delivers a dual benefit: it preserves the robustness of monocular SLAM while elevating the quality of the final reconstruction to a level that rivals that of sensor aided systems.

examples on two scenes. Importantly, this improvement in rendering fidelity  
![](images/1a843bf534cbcaf563a9d34b6f76b24ec61b1ceadb36659d9b5ada9bd871bf12.jpg)  
Fig. 4: From left to right: RGB GT, RGB rendering, Depth GT, Depth rendering and Depth predicted from fr1/desk (TUM) and Fast-Straight (UT-MM) scenes with our method on monocular settings. We achieve clear rendering fidelity and spatial consistency with the GT.

<table><tr><td></td><td colspan="3">Magic-SLAM</td><td colspan="3">MAC-Ego3D</td><td colspan="3">MM3DGS</td><td colspan="3">Ours</td></tr><tr><td>Scene</td><td>PSNR SSIM LPIPS</td><td></td><td></td><td> PSNR SSIM LPIPS</td><td></td><td></td><td></td><td></td><td></td><td>|PSNR SSIM LPIPS PSNR SSIM LPIPS</td><td></td><td></td></tr><tr><td>Square-1</td><td>18.46</td><td>0.77</td><td>0.37</td><td>20.60</td><td>0.18</td><td>0.76</td><td>16.06</td><td>0.57</td><td>0.44</td><td>18.13</td><td>0.64</td><td>0.43</td></tr><tr><td>Square-2</td><td></td><td></td><td></td><td>20.84</td><td>0.17</td><td>0.78</td><td>17.50</td><td>0.60</td><td>0.42</td><td>18.08</td><td>0.62</td><td>0.39</td></tr><tr><td>Ego-centric</td><td>19.10</td><td>0.79</td><td>0.30</td><td>14.38</td><td>0.39</td><td>0.61</td><td>22.81</td><td>0.82</td><td>0.22</td><td>23.34</td><td>0.86</td><td>0.20</td></tr><tr><td>Fast-Straight</td><td>22.45</td><td>0.86</td><td>0.28</td><td>21.04</td><td>0.20</td><td>0.77</td><td>21.58</td><td>0.78</td><td>0.29</td><td>24.51</td><td>0.82</td><td>0.26</td></tr><tr><td>fr1/desk</td><td>16.57 0.68</td><td></td><td>0.47</td><td>15.66 0.67</td><td></td><td>0.38</td><td>14.73</td><td>0.56</td><td>0.60</td><td>19.44</td><td>0.70</td><td>0.38</td></tr><tr><td>fr1/desk2</td><td>17.04</td><td>0.70</td><td>0.46</td><td>14.66</td><td>0.66</td><td>0.38</td><td>14.88</td><td>0.57</td><td>0.51</td><td>18.53</td><td>0.62</td><td>0.43</td></tr><tr><td>fr1/room</td><td>15.56</td><td>0.62</td><td>0.52</td><td>16.67</td><td>0.68</td><td>0.36</td><td>13.80</td><td>0.48</td><td>0.52</td><td>15.41</td><td>0.52</td><td>0.48</td></tr><tr><td>fr2/xyz</td><td>19.56</td><td>0.79</td><td>0.33</td><td>19.46</td><td>0.74</td><td>0.26</td><td>23.09</td><td>0.78</td><td>0.24</td><td>23.24</td><td>0.84</td><td>0.20</td></tr><tr><td>fr3/office</td><td>16.94</td><td>0.71</td><td>0.44</td><td>19.23</td><td>0.74</td><td>0.28</td><td>10.71</td><td>0.37</td><td>0.60</td><td>18.39</td><td>0.67</td><td>0.42</td></tr></table>

Table 2: Rendering comparison on TUM and UT-MM datasets. PSNR\delimter "3278 , SSIM\delimter "3278 and LPIPS\delimter "3279 . Magic-SLAM and MAC-Ego3D are RGBD, while MM3DGS and ours are RGB-only. We perform better than MM3DGS on all scenes, and achieve similar quality with the RGBD methods. Magic-SLAM failed on Square-2 scene.

We analyze the impact of IMU pre-integration by comparing against a constant velocity baseline, where angular velocity and linear acceleration are kept constant from the previous tracking step. Table 3 reports both the Absolute Trajectory Error (ATE) and PSNR metrics.

IMU integration provides substantial tracking improvements on challenging sequences with rapid motions. On Fast-straight, our IMU-enabled method reduces ATE from 12.85 cm to 6.07 cm, a 53% reduction while simultaneously improving PSNR by 2.73 dB. Similarly, on Square-1, ATE decreases from 55.43 cm to 42.33 cm.

<table><tr><td rowspan="2">Method</td><td colspan="2">Square1</td><td rowspan="2">Ego-centric Square2</td><td colspan="2" rowspan="2">Fast-straight</td><td rowspan="2">Avg</td></tr><tr><td>ATE</td><td>PSNR ATE PSNR ATE</td></tr><tr><td>MM3DGS (CV)</td><td>59.48 16.0</td><td>4.09</td><td>23.1567.2017.51</td><td>PSNR ATE 25.78</td><td>PSNR ATE 21.71</td><td>PSNR 39.14 19.59</td></tr><tr><td>Ours (CV)</td><td>55.43</td><td>17.0</td><td>5.02 23.52</td><td>62.84 17.96</td><td>12.85 21.78</td><td>34.03 20.06</td></tr><tr><td>MM3DGS (IMU)</td><td>47.05</td><td>16.06</td><td>3.52 22.81 68.50</td><td>17.50</td><td>16.78 21.58</td><td>33.96 19.49</td></tr><tr><td>Ours (IMU)</td><td>42.33 18.13</td><td></td><td>4.26 23.34 55.24 18.08</td><td>6.07</td><td>24.51</td><td>26.98 21.02</td></tr></table>

Table 3: Contribution of the IMU pre-integration compared to the Constant Velocity (CV) Assumption during the tracking phase. All methods are in monocular settings, our method outperforms MM3DGS in most cases. ATE RMSE is in cm and PSNR is in dB. The first , second , and third ranks are highlighted accordingly.

Notably, our method with constant velocity (ATE: 12.85 cm) outperforms MM3DGS with IMU integration (ATE: 16.78 cm) on Fast-straight, suggesting that the proposed spatial alignment and map initialization contribute independently to tracking accuracy. The combination of IMU pre-integration with our initialization achieves the best overall performance, reducing ATE relative to the constant velocity baseline while establishing good rendering quality across all sequences.

## 4.4 Performance on Global Reconstruction

To validate our collaborative back-end, we evaluate the accuracy of submap alignment using the UT-MM dataset [43] as a multi-agent benchmark. All sequences in the dataset were captured within the same physical environment, the Anna Hiss Gymnasium at the University of Texas. This spatial consistency allows us to treat the dataset as a multi agent benchmark. We can evaluate trajectory alignment across independently captured runs within a shared coordinate frame. Our results indicate that trajectory alignment succeeds in the majority of cases, validating the feasibility of multi agent reconstruction under these constraints. As illustrated in Fig. 5, our framework successfully aligns trajectories even with significant heading diferences. It is worth noting that our method completes the alignment process in less than 10 seconds, allowing to obtain a global map almost directly upon receiving final submap. For example with Square-1 & Faststraight: <1s NetVLAD matching + 1s VGGT inference + ∼3 s refinement = ∼5 s total per submap pair.

While submap fusion is performed once exploration ends, collaboration itself is online: keyframe encodings are broadcast continuously, and each agent uses them to detect overlap with other trajectories and to locally increase its keyframe density in those regions. We deliberately avoid tightly-coupled multi-agent bundle adjustment: its bandwidth requirements are incompatible with the degradednetwork regime we target. Tab. 4 isolates this contribution. On Sq1&FastStr the gain is limited, as every FastStr keyframe is already covisible and aligned; on the two other configurations, disabling dynamic keyframing degrades global ATE by 43% and 72%.

![](images/001fdd2feba7805e9c7c36f587534a2110e03e163155a30153ef210d79ef09b3.jpg)  
ATE-RMSE : 37,33cm

![](images/b1b4b03c30516530a4a8546b28cdf07d74b2ae4de96f80ed00994341af20cb60.jpg)  
ATE-RMSE : 18,12cm

![](images/33b40061a6b99b0debc837999ce5d1e4a1b6ddc72512ceb77f49917a3a9a1ee3.jpg)

Fig. 5: Submap alignment comparison on diferent scenes association from the UT-MM dataset. Each trajectory is plotted after transformed by the server sub-map alignment process. Our method achieves sub-meter spatial alignment with the ground truth (translation error < 0.40 m), and can work with multiple ( > 2) agents exploring the same scene.
<table><tr><td>Scenes</td><td>with Dynamic KF</td><td>without Dynamic KF</td></tr><tr><td>Sq1&amp;EgoCtr</td><td>37.33</td><td>53.40</td></tr><tr><td>Sq1&amp;FastStr</td><td>18.12</td><td>19.08</td></tr><tr><td>FastStr&amp;Sq1&amp;EC</td><td>39.63</td><td>68.13</td></tr></table>

Table 4: Dynamic KF impact on ATE (cm) on global map. Enabling it brings improved alignment on all scenes. It has limited contribution on Sq1&FastStr as the second trajectory is rectilinear and forward, thus all the keyframes carry the same information

## 5 Conclusion

We presented a novel Gaussian Splatting-based SLAM system that leverages a metric monocular depth estimation model (Depth Pro) to construct scaleconsistent 3D models without requiring dedicated depth sensors, such as LiDAR, which are typically unavailable on consumer-grade hardware. To further enhance tracking robustness and accelerate convergence, we integrate IMU data and show its contribution. Additionally, we introduce an approach for rapid submap alignment using VGGT, enabling collaborative mapping. Our method demonstrates improvements over state-of-the-art approaches, achieving both superior reconstruction quality and reduced tracking error compared to other monocular 3DGS methods, as well as comparable quality to RGB-D systems while relying only on RGB and inertial data. In future work, we aim to employ a single model for both depth estimation and camera pose estimation and reduce its computational requirements to enable fully onboard execution on consumer smartphones, paving the way for a completely integrated SLAM solution.

## References

1. Alcantarilla, P.F., Bergasa, L.M., Dellaert, F.: Visual odometry priors for robust ekf-slam. In: 2010 IEEE International Conference on Robotics and Automation. pp. 3501–3506 (2010). https://doi.org/10.1109/ROBOT.2010.5509272

2. Arandjelović, R., Gronat, P., Torii, A., Pajdla, T., Sivic, J.: Netvlad: Cnn architecture for weakly supervised place recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 40(6), 1437–1451 (2018). https://doi.org/ 10.1109/TPAMI.2017.2711011

3. Baruch, G., Chen, Z., Dehghan, A., Feigin, Y., Fu, P., Gebauer, T., Kurz, D., Dimry, T., Jofe, B., Schwartz, A., Shulman, E.: ARKitscenes: A diverse realworld dataset for 3d indoor scene understanding using mobile RGB-d data. In: Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1) (2021), https://openreview.net/forum?id=tjZjv\_ qh\_CE

4. Bloesch, M., Omari, S., Hutter, M., Siegwart, R.: Robust visual inertial odometry using a direct ekf-based approach. In: 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 298–304 (2015). https://doi.org/ 10.1109/IROS.2015.7353389

5. Bochkovskiy, A., Delaunoy, A., Germain, H., Santos, M., Zhou, Y., Richter, S., Koltun, V.: Depth pro: Sharp monocular metric depth in less than a second. In: International Conference on Learning Representations. vol. 2025, pp. 75602– 75637 (2025), https://proceedings.iclr.cc/paper\_files/paper/2025/file/ bc8b2058fd96978a4146f18298cb2d39-Paper-Conference.pdf

6. Bruno, H.M.S., Colombini, E.L.: Lift-slam: A deep-learning feature-based monocular visual slam method. Neurocomputing 455, 97–110 (2021). https://doi.org/ 10.1016/j.neucom.2021.05.027

7. Campos, C., Elvira, R., Gómez, J.J., Montiel, J.M.M., Tardós, J.D.: Orb-slam3: An accurate open-source library for visual, visual–inertial, and multimap slam. IEEE Transactions on Robotics 37(6), 1874–1890 (2021). https://doi.org/10. 1109/TRO.2021.3075644

8. Chatterjee, A., Ray, O., Chatterjee, A., Rakshit, A.: Development of a reallife ekf based slam system for mobile robots employing vision sensing. Expert Systems with Applications 38(7), 8266–8274 (2011). https://doi.org/https: //doi.org/10.1016/j.eswa.2011.01.007, https://www.sciencedirect.com/ science/article/pii/S0957417411000273

9. Gaia, J., Orosco, E., Rossomando, F., Soria, C.: Mapping the landscape of slam research: A review. IEEE Latin America Transactions 21(12), 1313–1336 (2023). https://doi.org/10.1109/TLA.2023.10305240

10. Gao, Y., Dai, Y., Li, H., Ye, W., Chen, J., Chen, D., Zhang, D., He, T., Zhang, G., Han, J.: Cosurfgs: 3d surface gaussian splatting with collaborative distributed learning for large-scale scene reconstruction. International Journal of Computer Vision 134(5), 195 (apr 2026). https://doi.org/10.1007/s11263-025-02627-9

11. Gu, T., Zhang, J., Liu, Y.: Accurate monocular slam initialization via structural line tracking. Sensors 23 (12 2023). https://doi.org/10.3390/s23249870

12. Handa, A., Whelan, T., McDonald, J., Davison, A.J.: A benchmark for rgb-d visual odometry, 3d reconstruction and slam. In: 2014 IEEE International Conference on Robotics and Automation (ICRA). pp. 1524–1531 (2014). https://doi.org/10. 1109/ICRA.2014.6907054

13. Herrera-Granda, E.P., Torres-Cantero, J.C., Pelufo-Ordóñez, D.H.: Monocular visual slam, visual odometry, and structure from motion methods applied to 3d reconstruction: A comprehensive survey (9 2024). https://doi.org/10.1016/j. heliyon.2024.e37356

14. Hery, J.: The environments of dune: Prophecy through the gaussian splat. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Talks. SIGGRAPH Talks ’25, Association for Computing Machinery, New York, NY, USA (2025). https://doi.org/10.1145/3721239. 3734124

15. Hu, J., Chen, X., Feng, B., Li, G., Yang, L., Bao, H., Zhang, G., Cui, Z.: Cg-slam: Eficient dense rgb-d slam in a consistent uncertainty-aware 3d gaussian field. In: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (eds.) Computer Vision – ECCV 2024. pp. 93–112. Springer Nature Switzerland, Cham (2025). https://doi.org/10.1007/978-3-031-72698-9\_6

16. Hu, J., Mao, M., Bao, H., Zhang, G., Cui, Z.: Cp-slam: Collaborative neural pointbased slam. NeurIPS (2023)

17. Kaveti, P., Gupta, A., Giaya, D., Karp, M., Keil, C., Nir, J., Zhang, Z., Singh, H.: Challenges of indoor slam: A multi-modal multi-floor dataset for slam evaluation. In: 2023 IEEE 19th International Conference on Automation Science and Engineering (CASE). pp. 1–8 (2023). https://doi.org/10.1109/CASE56687.2023. 10260618

18. Kazerouni, I.A., Fitzgerald, L., Dooly, G., Toal, D.: A survey of state-of-the-art on visual slam (11 2022). https://doi.org/10.1016/j.eswa.2022.117734

19. Kerbl, B., Kopanas, G., Leimkuehler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4) (Jul 2023). https: //doi.org/10.1145/3592433

20. Klein, G., Murray, D.: Parallel tracking and mapping for small AR workspaces. In: Proc. Sixth IEEE and ACM International Symposium on Mixed and Augmented Reality (ISMAR’07). Nara, Japan (November 2007)

21. Klenk, S., Chui, J., Demmel, N., Cremers, D.: Tum-vie: The tum stereo visualinertial event dataset. In: 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 8601–8608 (2021). https://doi.org/10.1109/ IROS51168.2021.9636728

22. Lajoie, P.Y., Ramtoula, B., Wu, F., Beltrame, G.: Towards collaborative simultaneous localization and mapping: A survey of the current research landscape. Field Robotics 2, 971–1000 (2022). https://doi.org/10.55417/fr.2022032

23. Leutenegger, S.: Okvis2: Realtime scalable visual-inertial slam with loop closure (2022). https://doi.org/arXiv:2202.09199

24. Li, D., Shi, X., Long, Q., Liu, S., Yang, W., Wang, F., Wei, Q., Qiao, F.: Dxslam: A robust and eficient visual slam system with deep features. In: 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 4958–4965 (2020). https://doi.org/10.1109/IROS45743.2020.9340907

25. Li, S., Zhang, D., Xian, Y., Li, B., Zhang, T., Zhong, C.: Overview of deep learning application on visual slam (9 2022). https://doi.org/10.1016/j.displa.2022. 102298

26. Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Zhao, Y., Peng, S., Guo, H., Zhou, X., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=yirunib8l8

27. Liu, F., Huang, M., Ge, H., Tao, D., Gao, R.: Unsupervised monocular depth estimation for monocular visual slam systems. IEEE Transactions on Instrumentation and Measurement 73, 1–13 (2024). https://doi.org/10.1109/TIM.2023.3342210

28. Longuet-Higgins, H.C.: A computer algorithm for reconstructing a scene from two projections. Nature 293, 133–135 (09 1981). https://doi.org/10.1038/293133a0

29. Matsuki, H., Murai, R., Kelly, P.H.J., Davison, A.J.: Gaussian splatting slam. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18039–18048 (2024). https://doi.org/10.1109/CVPR52733.2024. 01708

30. Niedermayr, S., Neuhauser, C., Petkov, K., Engel, K., Westermann, R.: Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices. In: Linsen, L., Thies, J. (eds.) Vision, Modeling, and Visualization. The Eurographics Association (2024). https://doi.org/10.2312/vmv.20241195

31. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024), https://openreview.net/forum?id=a68SUt6zFt, featured Certification

32. Pak, G., Kim, E.: Vigs slam: Imu-based large-scale rgb-d 3-d gaussian splatting slam. IEEE Transactions on Instrumentation and Measurement 75, 5010510– 5010510 (2026). https://doi.org/10.1109/TIM.2026.3693421

33. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., Koltun, V.: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(3), 1623–1637 (2022). https://doi.org/10.1109/TPAMI.2020.3019967

34. Sarlin, P.E., Dusmanu, M., Schönberger, J.L., Speciale, P., Gruber, L., Larsson, V., Miksik, O., Pollefeys, M.: Lamar: Benchmarking localization and mapping for augmented reality. In: Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VII. p. 686–704. Springer-Verlag (2022). https://doi.org/10.1007/978-3-031-20071-7\_40

35. Schubert, D., Goll, T., Demmel, N., Usenko, V., Stuckler, J., Cremers, D.: The tum vi benchmark for evaluating visual-inertial odometry. In: 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). p. 1680–1687. IEEE (Oct 2018). https://doi.org/10.1109/iros.2018.8593419

36. Servières, M., Renaudin, V., Dupuis, A., Antigny, N.: Visual and visual-inertial slam: State of the art, classification, and experimental benchmarking. Journal of Sensors 2021(1), 2054828 (2021). https://doi.org/10.1155/2021/2054828

37. Sewtz, M., Fanger, Y., Luo, X., Bodenmuller, T., Triebel, R.: Indoormcd: A benchmark for low-cost multi-camera slam in indoor environments. IEEE Robotics and Automation Letters 8, 1707–1714 (3 2023). https://doi.org/10.1109/LRA.2023. 3236840

38. Shotton, J., Glocker, B., Zach, C., Izadi, S., Criminisi, A., Fitzgibbon, A.: Scene coordinate regression forests for camera relocalization in rgb-d images. In: 2013 IEEE Conference on Computer Vision and Pattern Recognition. pp. 2930–2937 (2013). https://doi.org/10.1109/CVPR.2013.377

39. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S.E., Ramamonjisoa, M., Massa, F., HAZIZA, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C.,

Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jegou, H., Labatut, P., Bojanowski, P.: DINOv3. Transactions on Machine Learning Research (2026), https://openreview.net/forum?id=2NlGyqNjns, featured Certification

40. Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J., Mur-Artal, R., Ren, C., Verma, S., Clarkson, A., Yan, M., Budge, B., Yan, Y., Pan, X., Yon, J., Zou, Y., Leon, K., Carter, N., Briales, J., Gillingham, T., Mueggler, E., Pesqueira, L., Savva, M., Batra, D., Strasdat, H.M., Nardi, R.D., Goesele, M., Lovegrove, S., Newcombe, R.: The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797 (2019)

41. Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D.: A benchmark for the evaluation of rgb-d slam systems. In: 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 573–580 (2012). https://doi.org/10. 1109/IROS.2012.6385773

42. Sucar, E., Liu, S., Ortiz, J., Davison, A.J.: imap: Implicit mapping and positioning in real-time. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 6209–6218 (2021). https://doi.org/10.1109/ICCV48922.2021. 00617

43. Sun, L.C., Bhatt, N.P., Liu, J.C., Fan, Z., Wang, Z., Humphreys, T.E., Topcu, U.: Mm3dgs slam: Multi-modal 3d gaussian splatting for slam using vision, depth, and inertial measurements. In: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 10159–10166 (2024). https://doi.org/10.1109/ IROS58592.2024.10802389

44. Tourani, A., Bavle, H., Sanchez-Lopez, J.L., Voos, H.: Visual slam: What are the current trends and what to expect? (12 2022). https://doi.org/10.3390/ s22239297

45. Wang, C., Dai, Y., El-Sheimy, N., Wen, C., Retscher, G., Kang, Z., Lingua, A.: Progress on isprs benchmark on multisensory indoor mapping and positioning. In: International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences - ISPRS Archives. vol. 42, pp. 1709–1713. International Society for Photogrammetry and Remote Sensing (6 2019). https://doi.org/10.5194/ isprs-archives-XLII-2-W13-1709-2019

46. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: VGGT: Visual Geometry Grounded Transformer . In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5294–5306. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2025). https://doi.org/10.1109/ CVPR52734.2025.00499

47. Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861

48. Xu, X., Xue, F., Zhao, S., Pan, Y., Scherer, S., Huang, X.: Mac-ego3d: Multi-agent gaussian consensus for real-time collaborative ego-motion and photorealistic 3d reconstruction. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 854–863 (2025). https://doi.org/10.1109/CVPR52734. 2025.00088

49. Yan, C., Qu, D., Xu, D., Zhao, B., Wang, Z., Wang, D., Li, X.: Gs-slam: Dense visual slam with 3d gaussian splatting. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19595–19604 (2024). https://doi.org/10.1109/CVPR52733.2024.01853

50. Yugay, V., Gevers, T., Oswald, M.R.: MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM . In: 2025 IEEE/CVF Conference on Computer Vision and

Pattern Recognition (CVPR). pp. 6741–6750. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2025). https://doi.org/10.1109/CVPR52734.2025.00632

51. Zhang, H., Uchiyama, H., Ono, S., Kawasaki, H.: Motslam: Mot-assisted monocular dynamic slam using single-view depth estimation. In: 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 4865–4872 (2022). https://doi.org/10.1109/IROS47612.2022.9982280

52. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 586–595 (2018). https://doi.org/10.1109/CVPR.2018.00068

53. Zhang, Y., Tosi, F., Mattoccia, S., Poggi, M.: Go-slam: Global optimization for consistent 3d instant reconstruction. In: 2023 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 3704–3714 (2023). https://doi.org/10.1109/ ICCV51070.2023.00345

54. Zhao, H., Chen, J., Wang, L., Lu, H.: Arkittrack: A new diverse dataset for tracking using mobile rgb-d data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5126–5135 (June 2023)

55. Zhu, Z., Peng, S., Larsson, V., Xu, W., Bao, H., Cui, Z., Oswald, M.R., Pollefeys, M.: Nice-slam: Neural implicit scalable encoding for slam. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)

56. Özyeşil, O., Voroninski, V., Basri, R., Singer, A.: A survey of structure from motion. Acta Numerica 26, 305–364 (2017). https : / / doi . org / 10 . 1017 / S096249291700006X

# Supplementary Material: CGS-SLAM — Collaborative Gaussian Splatting based SLAM for Multi-Agent Reconstruction

## A Representation: 3DGS

This section details the Gaussian splatting technique used in our pipeline, it follows the original implementation by Kerbl et al. [19]. The underlying 3D scene is modeled using a collection of parametric Gaussian distributions. Each Gaussian model is denoted $G = \{ G _ { i } \} _ { i = 1 } ^ { N }$ and is characterized by the following parameters: a 3D position $\mu _ { i } \in { \mathbb { R } } ^ { 3 }$ , a rotation and scaling matrix R and S, representing the covariance matrix $\textstyle \mathcal { L } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ through the eigenvalue decomposition, an opacity value $\alpha _ { i } \in [ 0 , 1 ]$ and a color vector $c _ { i } \in \mathbb { R } ^ { 3 }$

The probability density function of a Gaussian centered at $\mu$ with covariance Σ is given by:

$$
G ( x ) = e ^ { - \frac { 1 } { 2 } ( x - \mu ) ^ { T } \Sigma ^ { - 1 } ( x - \mu ) } ,\tag{S1}
$$

where $x \in \mathbb { R } ^ { 3 }$ represents a spatial point.

To maintain the symmetry and positive semi-definiteness of Σ during optimization, an eigenvalue decomposition is employed. This allows to express Σ as:

$$
\boldsymbol { \Sigma } = \boldsymbol { R } \boldsymbol { S } \boldsymbol { S } ^ { T } \boldsymbol { R } ^ { T } ,\tag{S2}
$$

where R is an orthonormal rotation matrix and S is a diagonal matrix of scaling factors. This decomposition ensures numerical stability and physical plausibility of the Gaussian shapes.

To render the scene, the 3D Gaussians are projected onto the 2D image plane using a process known as splatting. The resulting 2D covariance matrix $\Sigma ^ { \prime }$ is computed as:

$$
\begin{array} { r } { \Sigma ^ { \prime } = J W \Sigma W ^ { T } J ^ { T } , } \end{array}\tag{S3}
$$

where W is the world-to-view transformation matrix, and J is the Jacobian of the linearized projection mapping.

Given a set of Gaussians G and a camera pose $T _ { c } ,$ the color C of a pixel is computed by blending the contributions of all Gaussians that project onto the pixel, ordered by increasing depth. The blending formula is:

$$
C ( G , T _ { c } ) = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{S4}
$$

where $c _ { i }$ is the color of the i-th Gaussian, and $\alpha _ { i }$ is the local opacity sampled from the 2D projection of that Gaussian at the pixel location.

The final pixel opacity O is defined analogously:

$$
O ( G , T _ { c } ) = \sum _ { i = 1 } ^ { N } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) .\tag{S5}
$$

Because the entire rendering process is diferentiable with respect to the Gaussian parameters, we can optimize $G$ using gradient-based methods. Specifically, we minimize the $L _ { 1 }$ reconstruction loss between the rendered image $C ( G , T _ { c } )$ and a reference ground-truth image I:

$$
\mathcal { L } _ { \mathrm { p h o t o } } = \| I - C ( G , T _ { c } ) \| _ { 1 } .\tag{S6}
$$

## B Message passing between clients and server

To enhance the selection of keyframe candidates for the alignment process described in Sec. 3.3, we introduce a lightweight, scalable message-passing framework that enables inter-agent coordination without requiring centralized mapping or full trajectory sharing. The server operates in an idle state until it receives initialization messages from individual agents, each of which begins by transmitting its intrinsic camera parameters for potential heterogeneity in sensor models across clients. Once initialized, the server assumes a broadcast role: it collects and caches compact keyframe encodings from all active agents, then disseminates them in real time to all connected clients. This enables each agent to continuously monitor for spatial overlaps with other agents’ trajectories by comparing incoming encodings against its own local keyframe database. As the keyframe encoder, we adopt NetVLAD [2], a lightweight, descriptor-based architecture widely validated for place recognition and visual localization tasks. No incorrect submap association occurred on the datasets evaluated here, but robustness in visually repetitive environments remains to be quantified. Future work will consider stronger encoders such as DINOv3 [39], following [50] and its use of DINOv2 encodings [31]. To ensure low-latency communication and minimal bandwidth overhead, we have designed a compact, extensible message protocol. The architecture supports multiple message types, including initialization, keyframe encoding, pose updates, and alignment triggers while maintaining a minimal fixed-size header and variable-length payload. The message layout is shown in Fig. S1.

![](images/e706bf149737730614854acdfd368e3545875e9a70886f606c918a0265bd00f9.jpg)  
Fig. S1: Our communication protocol uses a custom message architecture enabling eficient, low-bandwidth communication of keyframe encodings.

## C IMU pre-integration

Most consumer-grade devices embed 6-degree-of-freedom (6-DOF) IMUs which include accelerometers and gyroscopes that measure linear acceleration, ${ \bf a } \ =$ $[ { \ddot { x } } , { \ddot { y } } , { \ddot { z } } ]$ , and angular velocity, $\dot { \Theta } = [ \dot { \alpha } , \dot { \beta } , \dot { \gamma } ]$ , in 3D space. The displacement between two consecutive time steps, $\varDelta \mathbf { p } _ { t } ^ { t - 1 }$ , expressed in the coordinate frame of the previous time step, can be computed as:

$$
\Delta \mathbf { p } _ { t } ^ { t - 1 } = \mathbf { v } _ { t - 1 } \cdot \varDelta t + \frac 1 2 \mathbf { a } _ { t } \cdot ( \varDelta t ) ^ { 2 } ,\tag{S7}
$$

where the velocity $\mathbf { v } _ { t - 1 }$ is obtained by numerically integrating the previous acceleration over time:

$$
\mathbf { v } _ { t - 1 } = \mathbf { v } _ { t - 2 } + \mathbf { a } _ { t - 1 } \cdot \varDelta t .\tag{S8}
$$

Similarly, the angular displacement $\varDelta \Theta _ { t } ^ { t - 1 }$ , again expressed in the previous frame, is given by:

$$
\varDelta \Theta _ { t } ^ { t - 1 } = \dot { \Theta } _ { t - 1 } \cdot \varDelta t .\tag{S9}
$$

Using these estimates, the relative transformation ${ \bf \Phi } _ { t } ^ { t - 1 } \mathbf T _ { I }$ between the two IMU frames can be constructed as:

$$
\mathbf { \Lambda } _ { t } ^ { t - 1 } \mathbf { T } _ { I } = \left[ \mathbf { \Lambda } _ { t } ^ { t - 1 } \mathbf { R } \mid \varDelta \mathbf { p } _ { t } ^ { t - 1 } \right] ,\tag{S10}
$$

where $_ t ^ { t - 1 }$ R is a rotation matrix derived from the angular displacement $\varDelta \Theta _ { t } ^ { t - 1 }$

To express this transformation in the camera coordinate frame, we apply the static transformation $^ C _ { I } \mathbf { T }$ between the IMU and the camera, yielding the relative camera transformation:

$$
\mathbf { \Lambda } _ { t } ^ { t - 1 } \mathbf T _ { C } = \mathbf { \Lambda } _ { I } ^ { C } \mathbf T \cdot \mathbf { \Lambda } _ { t } ^ { t - 1 } \mathbf T _ { I } .\tag{S11}
$$

## D Implementation Details

Table S1 lists the specific hyperparameters and parameters used throughout our method for training, tracking, and global mapping. The mapping process uses a set of balanced loss weights $\left( \lambda _ { D } , \lambda _ { C } , \lambda _ { S } \right)$ and densification intervals to maintain a high-fidelity representation of the scene while preventing Gaussian explosion. For tracking, we enforce a strict pixel opacity threshold and covisibility constraints to ensure frame-to-frame pose accuracy. Finally, the global consistency parameters, including dedicated learning rates for global position and rotation, are used during the optimization phase of the submap alignment process.

## E Two-Stage Procrustes Alignment

Given a pair of matching keyframes $( f _ { 1 } , f _ { 2 } )$ from clients 1 and 2 respectively, we collect their temporal neighbors $\mathcal { N } _ { 1 } , \mathcal { N } _ { 2 }$ to form two point sets. For robustness to rotation ambiguity, we augment each point set with a virtual point ofset along the camera’s up-axis:

$$
\mathbf { p } _ { i } ^ { + } = \mathbf { p } _ { i } + \lambda \cdot \mathbf { u } _ { i } ,\tag{S12}
$$

<table><tr><td></td><td>Keyframing</td><td>Value</td></tr><tr><td>Mapping Value</td><td>Covisibility threshold</td><td>0.9</td></tr><tr><td>Learning Rates (LR)</td><td rowspan="2">Min KF s Max KF k</td><td>2 5</td></tr><tr><td> $\lambda _ { D }$  0.05</td><td></td></tr><tr><td> $\lambda _ { C }$  0.75</td><td rowspan="2">Tracking Value</td></tr><tr><td> $\lambda _ { S }$  0.2</td></tr><tr><td>Camera t / q LR 0.001 / 0.003</td><td> $\lambda _ { D }$  0.001</td></tr><tr><td>Feature / Opacity LR 0.0025 / 0.05</td><td>Pixel opacity threshold 0.99</td></tr><tr><td>Scaling / Rotation LR 0.001 / 0.001</td><td>Global Consistency Value</td></tr><tr><td>Densification</td><td>Translation LR (global) 0.01</td></tr><tr><td>Densification interval 40 Pruning interval 50</td><td>Rotation LR (global) 0.01</td></tr><tr><td>Max steps 30,000</td><td>Neighbor frames 3</td></tr><tr><td></td><td>Iterations 300</td></tr></table>

Table S1: System parameters used in our experimental setup.

where $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ is the camera position, $\mathbf { u } _ { i }$ is the local y-axis (up-vector) of the camera frame, and λ is set to the mean inter-camera distance within the point set. Let $\mathcal { P } _ { k } = \{ \mathbf { p } _ { i } \} _ { i \in \{ f _ { k } \} \cup \mathcal { N } _ { k } } \cup \{ \mathbf { p } _ { f _ { k } } ^ { + } \}$ denote the augmented point set for client k.

Stage 1: VGGT → Sub-map 1. We estimate a similarity transform $\mathbf { T } _ { 1 } \in$ Sim(3) that aligns the VGGT camera positions of client 1’s keyframes to their corresponding sub-map poses. Denoting by t(M) the translation component of a $4 \times 4$ matrix M, we solve:

$$
\mathbf { T } _ { 1 } = \mathop { \arg \operatorname* { m i n } } _ { \mathbf { T } \in { \cal S } i m ( 3 ) } \sum _ { \mathbf { p } \in \mathcal { P } _ { 1 } ^ { \mathrm { v g g t } } } \left\| \mathbf { T } \cdot \mathbf { p } - \tilde { \mathbf { p } } \right\| ^ { 2 } ,\tag{S13}
$$

where $\mathcal { P } _ { 1 } ^ { \mathrm { v g g t } }$ contains positions extracted from inverted VGGT extrinsics $\{ \mathbf { t } ( \mathbf { E } _ { i } ^ { - 1 } ) \}$ and $\tilde { \mathbf { p } }$ denotes the corresponding sub-map position $\{ \mathbf { t } ( ( \mathbf { T } _ { C _ { i } } ^ { S _ { 1 } } ) ^ { - 1 } ) \}$

Stage 2: Transformed VGGT → Sub-map 2. We apply ${ \bf T } _ { 1 }$ to client $2 \mathrm { { : } s }$ VGGT poses, bringing them into the coordinate frame of sub-map 1. We then estimate $\mathbf { T } _ { 2 }$ aligning these transformed poses to sub-map 2:

$$
\mathbf { T } _ { 2 } = \mathop { \arg \operatorname* { m i n } } _ { \mathbf { T } \in S i m ( 3 ) } \sum _ { \mathbf { p } \in \mathcal { P } _ { 2 } ^ { \mathrm { v g g t } } } \left\| \mathbf { T } \cdot \left( \mathbf { T } _ { 1 } \cdot \mathbf { p } \right) - \tilde { \mathbf { p } } \right\| ^ { 2 } ,\tag{S14}
$$

where $\tilde { \mathbf { p } }$ now refers to sub-map 2 positions $\{ \mathbf { t } ( ( \mathbf { T } _ { C _ { i } } ^ { S _ { 2 } } ) ^ { - 1 } ) \}$

Both stages are solved via Procrustes analysis and subsequently refined by gradient descent on the full pose error, including orientation. Since $\mathbf { T } _ { 1 }$ maps VGGT coordinates to sub-map 1 and $\mathbf { T } _ { 2 }$ maps sub-map 1 coordinates (via the transformed VGGT poses) to sub-map 2, the composition $\mathbf { T } _ { 2 } \circ \mathbf { T } _ { 1 }$ efectively transforms from VGGT to sub-map 2. The alignment from sub-map 2 to submap 1 is therefore:

$$
\boxed { \mathbf { T } _ { S _ { 2 } } ^ { S _ { 1 } } = \mathbf { T } _ { 2 } ^ { - 1 } }\tag{S15}
$$

## F Experiments with Depth Anything V3

In the aim of reducing the dependency on two deep neural network models, we tested our method on Depth Anything V3 [26] (DA3) which produces both metric depth maps and camera pose estimation. It could then replace Depth Pro for the MMDE part of the local mapping and VGGT for submap alignment in the global consistency. Tab. S2 reports the comparison, and Fig. S2 a corresponding trajectory. However, our results showed lower quality for the local mapping part, and thus contamination of the submap alignment process, which made us choose to keep the dual model approach.

<table><tr><td></td><td colspan="4">Depth Pro</td><td colspan="4">Depth Anything V3</td></tr><tr><td>Scene</td><td>ATE-RMSE PSNR SSIM LPIPS ATE-RMSE PSNR SSIM LPIPS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Square-1</td><td>42.33</td><td>18.13</td><td>0.64</td><td>0.43</td><td>61.42</td><td>16.42</td><td>0.53</td><td>0.45</td></tr><tr><td>Square-2</td><td>55.24</td><td>18.08</td><td>0.62</td><td>0.39</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Ego-centric</td><td>4.26</td><td>23.34</td><td>0.86</td><td>0.20</td><td>26.55</td><td>18.62</td><td>0.69</td><td>0.35</td></tr><tr><td>Fast-Straight</td><td>6.07</td><td>24.51</td><td>0.82</td><td>0.26</td><td>12.63</td><td>20.54</td><td>0.68</td><td>0.48</td></tr><tr><td>Slow-Straight-2</td><td>5.01</td><td>25.85</td><td>0.84</td><td>0.24</td><td>6.18</td><td>25.66</td><td>0.84</td><td>0.24</td></tr><tr><td>fr1/desk</td><td>4.2</td><td>19.44</td><td>0.70</td><td>0.38</td><td>3.26</td><td>21.83</td><td>0.77</td><td>0.29</td></tr><tr><td>fr1/desk2</td><td>5.12</td><td>18.53</td><td>0.62</td><td>0.43</td><td>7.83</td><td>20.18</td><td>0.72</td><td>0.36</td></tr><tr><td>fr1/room</td><td>5.3</td><td>15.41</td><td>0.52</td><td>0.48</td><td>10.36</td><td>14.84</td><td>0.48</td><td>0.49</td></tr><tr><td>fr2/xyz</td><td>1.14</td><td>23.24</td><td>0.84</td><td>0.20</td><td>1.32</td><td>25.40</td><td>0.84</td><td>0.19</td></tr><tr><td>fr3/office</td><td>20.42</td><td>18.39</td><td>0.67</td><td>0.42</td><td>33.82</td><td>18.32</td><td>0.63</td><td>0.46</td></tr></table>

Table S2: Tracking (ATE-RMSE ↓) and Rendering (PSNR \delimter "32 78 , SSIM\delimter "32 78 and LPIPS\delimter "32 79 ) comparison between Depth Pro (DP) and Depth Anything V3 (DA3) setup on TUM and UT-MM datasets. DA3 is competitive and occasionally better on the small TUM sequences, but degrades sharply on the large UT-MM scenes and fails on Square-2, which is why we retain the dual-model setup

![](images/b18b873bcb42a83647ab783330927c388789df326e54dea3a7a06406c66285c2.jpg)

![](images/e5b09ce8678f271f5996d118cc2f057b663d754ea4dca36295392bfed76fee75.jpg)

![](images/6a86778462425c645b7b6ab6e8bbe388492ea8dbd8acf8175e2cdd6ac16511a2.jpg)  
Fig. S2: Trajectory comparison between Depth Pro and Depth Anything configuration. We can see that the Depth Pro configuration achieves better tracking, with better handling of turns.

## G Tests on Replica Multiagent Dataset with the constant velocity hypothesis

Replica Multiagent [16, 40] provides no inertial measurements, so running CGS-SLAM on it requires replacing the IMU prior with a constant-velocity assumption. Table 3 quantifies the cost of this substitution on UT-MM (7.05 cm of average ATE). The results below are reported for completeness and are not directly comparable to the main evaluation.

As reported in Tab. S3, results are bimodal rather than uniformly degraded. Apt-0 and Apt-2/agent-0 stay within the range of our IMU-based results, while Apt-1 and Apt-2/agent-1 diverge above one meter (Tab. S3 and Fig. S3). Since both agents of Apt-2 observe the same scene, the separation follows the trajectories rather than the environment. The diverging runs are those crossing between rooms, where a doorframe passes within a few centimeters of the camera. Our depth estimation is unreliable at such close range on synthetic imagery, and the resulting keyframe injects grossly mis-scaled geometry into the Gaussian map; without an inertial prior to constrain the following poses, tracking has no way to recover. An example of this misprediction is visible on the last line of Fig. S4. The failure is therefore driven by near-field depth estimation rather than by the collaborative back-end, and it is confined to a component our pipeline treats as interchangeable (Section F).

Scene Agent ATE-RMSE PSNR SSIM LPIPS
<table><tr><td>Apt-0</td><td>0 1</td><td>12.33 15.06</td><td>26.96 25.20</td><td>0.79 0.81</td><td>0.29 0.24</td></tr><tr><td rowspan="2">Apt-1</td><td>0</td><td>199.39</td><td>12.92</td><td>0.55</td><td>0.49</td></tr><tr><td>1</td><td>122.20</td><td>17.65</td><td>0.62</td><td>0.43</td></tr><tr><td rowspan="2">Apt-2</td><td>0</td><td>19.54</td><td>25.20</td><td>0.77</td><td>0.28</td></tr><tr><td>1</td><td>150.09</td><td>18.83</td><td>0.72</td><td>0.34</td></tr></table>

Table S3: Tracking (ATE-RMSE ↓cm) and Rendering (PSNR \delimter "3278 dB, SSIM\delimter "3278 and LPIPS \delimter "3279 ) on Replica Dataset under the constant velocity hypothesis (no IMU).

![](images/277121669379cff4ab2423b86d414c8488afe3be2258505a3adca10ed416230d.jpg)

![](images/4e5ce66f51291bdc36b7f21b92016c34debfe464888f1f36789643e479604147.jpg)

![](images/69852ad9253fed06ede8c7687163fc0788480be5974c5e8f2cc2c76db141142b.jpg)  
Fig. S3: Top view of trajectories from global alignment of submaps from ReplicaMultiagent. The tracking error from the lack of IMU contaminates the submap alignments, resulting in poor ATE-RMSE.

![](images/ea9da2805d5792479a3d06c89a2ed7dbb5059eb272f3721f1e31c66440144dd9.jpg)  
Fig. S4: Rendering examples of the scenes from Apt-0 and Apt-1 of the Replica dataset.

## H Supplementary Results

For completeness, tracking results of Magic-SLAM and MAC-Ego3D on the full UT-MM dataset are reported in Tab. S4. We also provide supplementary rendering results in Fig. S5

<table><tr><td>Method</td><td>Config</td><td>Square-1</td><td>Square-2</td><td>Ego-Centric-1 Fast-Straight</td><td></td></tr><tr><td>Magic-SLAM</td><td>RGBD</td><td>6.24</td><td>X</td><td>4.78</td><td>5.72</td></tr><tr><td>MAC-Ego3D</td><td>RGBD</td><td>40.07</td><td>77.24</td><td>39.08</td><td>19.44</td></tr><tr><td>Ours</td><td>RGB+IMU</td><td>42.33</td><td>55.24</td><td>4.26</td><td>6.07</td></tr></table>

<table><tr><td colspan="4">Slow-Straight-1 Slow-Straight-2</td><td>Ego-Drive</td><td>Ego-Centric-2</td></tr><tr><td>Magic-SLAM</td><td>RGBD</td><td>8.49</td><td>5.94</td><td>8.01</td><td>7.16</td></tr><tr><td>MAC-Ego3D</td><td>RGBD</td><td>13.82</td><td>16.02</td><td>X</td><td>19.31</td></tr><tr><td>Ours</td><td>RGB+IMU</td><td>1.43</td><td>5.01</td><td>15.86</td><td>5.50</td></tr></table>

Table S4: Single agent tracking evaluation ATE (cm). Ours (RGB) against Magic-SLAM (RGB-D) and MAC-Ego3D (RGB-D) on UT-MM scenes.

Tab. S4 extends the single-agent evaluation to the full UT-MM dataset. Ours is the only system to complete all eight sequences: Magic-SLAM diverges on Square-2 and MAC-Ego3D on Ego-Drive. Performance is governed by the motion profile of each sequence. On the straight and localized sequences we stay in the range of the RGB-D baselines without using a depth sensor. The Square sequences, which chain four sharp turns, are the exception, and they are dificult for every evaluated system (MAC-Ego3D: 40.07/77.24 cm; MM3DGS: 47.05/68.50 cm, Tab. 3). Sharp turns provide little translational parallax, so heading error accumulates; Magic-SLAM recovers on Square-1 through loop closure and global bundle adjustment, which we omit for bandwidth reasons. Robustness to rotation-dominated motion is the main avenue for improvement.

![](images/5f43a1be5cdcbcad218c0e1cb7c272952a8cf81b134085e33ca3507580be97e1.jpg)  
Fig. S5: Rendering results of monocular single agent exploration on UT-MM and TUM dataset with Depth Pro as MMDE. We can see that the rendering provides good results, but also some small artifacts. Leveraging a MMDE allows to bypass some of the poorly detailed area of the depth ground truth from UT-MM.