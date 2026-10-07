# 4D-HOF: HAND-OBJECT FLOW MATCHING FORFEED-FORWARD 4D INTERACTION RECONSTRUCTION

Shiqi Li<sup>1</sup> Sean Cho<sup>1</sup> Yijie Li<sup>1</sup> Fengzhi Guo<sup>1</sup> Bowen Wen<sup>2</sup> Cheng Zhang<sup>1</sup>

<sup>1</sup>Texas A&M University <sup>2</sup>NVIDIA

https://tamu-visual-ai.github.io/4D-HOF

![](images/f419380866de68242142e88d51a0cb85151889ae4c0822ffc584e1266b84634a.jpg)  
Figure 1: 4D Hand-Object Flow Matching (4D-HOF). Given a monocular video of a hand interacting with an unseen object, we learn a flow matching model for accurate and efficient 4D hand-object interaction reconstruction. Our method generalizes across diverse and challenging scenarios, including ambiguous depth (headset), occlusion (kettle), complex articulated hand poses (bar), and large object rotation (camera). Moreover, 4D-HOF is substantially faster than prior optimization-based methods due to our feed-forward design.

## ABSTRACT

Existing methods for 4D hand-object reconstruction often rely on costly persequence optimization, while generative approaches typically synthesize interactions from random noise, which can lead to unstable interaction prediction. We introduce 4D-HOF, a feed-forward framework that reconstructs 4D hand-object interactions from coarse but informative estimates produced by vision foundation models. Concretely, we learn a conditional flow matching model that transports foundation-model-derived hand-object states toward an interaction manifold, allowing the model to correct errors in translation, rotation, and alignment in a feed-forward manner. A key advantage of our generative formulation is that it naturally enables test-time guidance within the transport process. Rather than applying a separate post-hoc optimization after reconstruction, we directly steer the evolving generative states using physical interaction constraints and observed 2D evidence, allowing the reconstruction to be refined as part of the generative process itself. By training the generative model on diverse datasets, 4D-HOF generalizes robustly to challenging in-the-wild scenarios. Experiments on out-of-domain benchmarks show that 4D-HOF achieves state-of-the-art performance, producing more stable and accurate 4D hand-object reconstructions.

## 1 INTRODUCTION

Reconstructing 4D hand-object interaction (HOI) from monocular video is a fundamental problem for understanding human behavior and enabling a variety of applications such as robot learning (He et al., 2025; Zhong et al., 2025; Paliwal et al., 2026), extended reality (Marchand et al., 2015), and human-computer interaction (Pavlovic et al., 2002). Despite recent progress (Fan et al., 2024; Wang et al., 2025b; Shi et al., 2026; Xu et al., 2026), recovering accurate hand-object geometry and interaction remains challenging due to diverse object geometry and pose, severe occlusion, and the inherent ambiguity of monocular observations.

To address these challenges, existing approaches mainly fall into two lines. Optimization-based methods (Fan et al., 2024; Wang et al., 2025b; Shi et al., 2026; Aytekin et al., 2026b; Zhu et al., 2025) formulate HOI reconstruction as a per-sequence fitting problem, enforcing interaction constraints through manually designed objectives such as contact, rendering, and interaction stability. While effective, these methods are computationally expensive and often brittle under imperfect initialization or occlusion. Generative methods (Ye et al., 2023; 2024; Christen et al., 2024; Li et al., 2025; Ye et al., 2026) learn priors over HOI distributions from large-scale data, yet often start from random noise or rely on category-level priors with closed-set vocabularies, making it difficult to preserve instance-specific geometry and pose cues. This raises a key question: can we retain and learned priors of generative modeling while starting from informative observations rather than random noise?

Vision foundation models provide a natural starting point. Recent large-scale pretrained models can estimate hand pose, object geometry, depth, and scene structure from monocular inputs (Carion et al., 2026; Oquab et al., 2023; Piccinelli et al., 2025; Lin et al., 2025b; Wen et al., 2024; Pavlakos et al., 2024; Potamias et al., 2025). These models produce coarse yet informative estimates that capture rich geometric cues. Indeed, recent works have begun to incorporate foundation models into HOI reconstruction (Liu et al., 2025b; Wang et al., 2026; Lin et al., 2025a; Xu et al., 2026). However, they typically treat these estimates as initialization and rely on downstream heuristic optimization to enforce consistency, which reintroduces inefficiency and instability.

In this work, we instead couple foundation-model perception directly with generative modeling. We use them to define the source distribution of a conditional generative transport process. Starting from coarse but informative hand-object states, we learn how these states should evolve toward physically plausible interactions. Specifically, we adopt flow matching and stochastic interpolates (Liu et al., 2022b; Shi et al., 2023; Albergo et al., 2025) to construct a generative bridge that transports the initial estimates toward an HOI manifold. To ground refinement in observed evidence and exploit interaction dynamics, the bridge is conditioned on both the input video frames and the temporal context of the initial 4D hand-object sequence. This formulation enables unified correction of geometry, pose, and alignment in a feed-forward manner, while preserving instance-specific cues from the input. Importantly, the generative formulation also gives us a natural mechanism for test-time control to further enhance fine-grained physical plausibility and consistency with the 2D observation. Our method, termed 4D-HOF, is a feed-forward framework that takes foundation-model estimates as input and jointly refines hand-object interactions. We summarize our contributions as follows:

• We formulate 4D HOI reconstruction as conditional generative transport from foundation-modelderived hand-object state initialization to physically plausible interactions, avoiding random noise as initialization to flow matching and costly per-sequence optimization.

• We design a conditional flow matching model that jointly refines hand-object geometry, pose, and alignment using visual, geometric, and temporal cues.

• We introduce test-time guidance enabled by the generative transport process, which steers evolving states with image, interaction, and temporal constraints without separate post-hoc optimization.

• Experiments on 4D HOI benchmarks and challenging in-the-wild scenarios demonstrate stateof-the-art performance in hand estimation, object reconstruction, interaction consistency, and temporal stability, while running over 5 times faster than prior methods.

## 2 RELATED WORK

Hand-Object Reconstruction. Early image-based methods jointly regress hand and object meshes using parametric hand priors and contact constraints (Hasson et al., 2019; Ye et al., 2022). Videobased methods typically optimize each sequence with hand-crafted geometric and interaction objectives (Hasson et al., 2021; Wen et al., 2023; Fan et al., 2024; Wang et al., 2025b; Lin et al., 2025a; Shi et al., 2026; Aytekin et al., 2026a; On et al., 2025; Liu et al., 2025a; Aboukhadra et al., 2026), incurring high computational costs and sensitivity to initialization. Feed-forward approaches accelerate inference (Chen et al., 2026b) but emphasize geometry without explicitly modeling interaction dynamics. Our method instead learns an interaction prior that maps coarse estimates to physically plausible interactions for feed-forward reconstruction.

Foundation Models for 3D Understanding. Foundation models provide strong priors for segmentation (Carion et al., 2026), visual representations (Oquab et al., 2023), monocular and multi-view geometry (Kong et al., 2026; Piccinelli et al., 2025; Lin et al., 2025b; Wen et al., 2025; 2026; Wang et al., 2025a), object pose (Wen et al., 2024), hand reconstruction (Pavlakos et al., 2024; Potamias et al., 2025), and 3D generation (Chen et al., 2026a; Xiang et al., 2026). Recent HOI reconstruction pipelines use these models for initialization (Wang et al., 2026; Lin et al., 2025a; Xie et al., 2026), but independently estimated components often have inconsistent scale, depth, and orientation. Existing methods resolve these discrepancies through additional optimization (Shi et al., 2026; Aytekin et al., 2026a); we learn to jointly refine the estimates without per-sequence optimization.

Generative Priors for Hand-Object Interaction. Generative models capture HOI distributions through diffusion (Ho et al., 2020; Christen et al., 2024; Li et al., 2025) or language modeling (Huang et al., 2025a), primarily for interaction synthesis. Complementary denoising methods learn spatiotemporal priors to refine noisy HOI trajectories (Zhou et al., 2022; Liu & Yi, 2024; Luo et al., 2024), demonstrating the potential of learned priors for HOI correction. For reconstruction, Diff-HOI (Ye et al., 2023) and G-HOP (Ye et al., 2024) incorporate generative priors, but they rely on noise-driven generation or category-level priors with closed-set vocabulary. Most closely related, CHOIR (Xu et al., 2026) learns a flow-matching prior to correct relative hand-object 1D depth displacement along the viewing ray from initializations, while contact-aware optimization refines the full interaction. Building on flow matching and stochastic interpolants (Liu et al., 2022b; Shi et al., 2023; Albergo et al., 2025), we instead learn a conditional bridge that jointly refines entire HOI states, making the learned prior the primary reconstruction mechanism, complemented by test-time guidance.

Agentic Real-to-Sim. Recent advances in agentic AI motivate its integration into 3D reconstruction pipelines. AgentSTAR (Mazur et al., 2026) couples visual reasoning with numerical optimization in a render-and-compare loop to recover object geometry, articulation, and motion, but does not explicitly model hand–object interactions. Inspired by GPT-6 Astra’s capabilities, we conduct pilot experiments on agentic HOI reconstruction, detailed in the Appendix B.4. Our preliminary findings suggest that this approach is promising but incurs substantial runtime and costs. Our proposed feed-forward method offers an efficient reconstruction module that agents could use to accelerate this process.

## 3 APPROACH

## 3.1 PROBLEM FORMULATION

4D HOI reconstruction. Given an N-frame monocular RGB video ${ \mathcal { T } } = \{ \mathbf { I } ^ { i } \} _ { i = 1 } ^ { N }$ , where a hand interacts with an object of arbitrary shape, our goal is to reconstruct the underlying 4D hand-object interaction. We aim to reconstruct the object mesh $\mathbf { m } _ { o }$ with corresponding per-frame 6-DoF poses $\mathbf { o } ^ { i } = ( \mathbf { R } ^ { o } , \mathbf { t } ^ { o } )$ , where $\mathbf { R } ^ { o } \in S O ( 3 )$ and translation $\mathbf { t } ^ { o } \in \mathbb { R } ^ { 3 }$ . For the hand, we estimate the per-frame MANO (Romero et al., 2022) parameters, consisting of shape $\beta$ and pose $\mathbf { h } ^ { i } = ( \mathbf { R } ^ { h } , \mathbf { t } ^ { h } )$ ). We denote the full interaction trajectory as ${ \mathcal { S } } = \{ { \mathbf { s } } ^ { i } \} _ { i = 1 } ^ { N }$ , where $\mathbf { s } ^ { i } = [ \mathbf { h } ^ { i } , \mathbf { o } ^ { i } ]$

Flow matching for HOI reconstruction. Flow matching (Lipman et al., 2022; Liu et al., 2022a) formulates generative modeling as learning a continuous transport from a source distribution $p ( \mathbf { x } _ { 0 } )$ to a target data distribution $p ( \mathbf { x } _ { 1 } )$ . Let ${ \bf x } _ { 0 } \sim p ( { \bf x } _ { 0 } )$ and $\mathbf { x } _ { 1 } \sim p ( \mathbf { x } _ { 1 } )$ . A stochastic interpolant defines intermediate states $\mathbf { x } _ { t } = ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } + \gamma ( t ) { \boldsymbol { \epsilon } } $ , where ${ \bf \bar { \Phi } } \mathrm { ~  ~ \cdot ~ } t \in [ 0 , 1 ] , \epsilon \sim \mathcal { N } ( 0 , { \bf I } )$ , and $\gamma ( t )$ is a noise schedule. Flow matching learns a velocity field $\mathbf { v } _ { \theta } ( \mathbf { x } , t )$ such that the trajectory follows ${ d \mathbf { x } ( t ) } / { d t } = \mathbf { v } _ { \theta } ( \mathbf { x } ( t ) , t )$ , transporting samples from $p ( \mathbf { x } _ { 0 } )$ to $p ( \mathbf { x } _ { 1 } )$ . In practice, $\mathbf { v } _ { \theta }$ can be a neural network $( \mathrm { e . g . }$ ., DiT (Peebles & Xie, 2023)) trained to match the conditional drift induced by the interpolate via the following objective:

$$
\mathcal { L } _ { \theta } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } , \epsilon } \left[ \| \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { t } ) / ( 1 - t ) \| _ { 2 } ^ { 2 } \right] .\tag{1}
$$

In the context of HOI, the target distribution $p ( \mathbf { x } _ { 1 } )$ corresponds to physically plausible hand-object interactions, which can be learned from large-scale HOI datasets (Wang et al., 2024; Yang et al., 2022; Swamy et al., 2023; Hampali et al., 2020). While several recent works adopt generative models for HOI, they often start from random noise (Ye et al., 2022; 2026), depend on object templates (Ye et al., 2024), or target the task of grasp synthesis (Wu et al., 2026) instead of full 4D HOI reconstruction.

![](images/651ef7b112d1a006b86891357d1f863aea1cca2bb5a5f257fc08d4c9cafa5f1e.jpg)  
Figure 2: 4D-HOF overview. Given a monocular video, we construct initial hand-object states by leveraging foundation models, including (a) contextual scene parsing, (b) object reconstruction, (d) hand reconstruction, and (c) depth alignment for recovering metric geometry, producing coarse HOI initialization. We then formulate HOI refinement as a conditional generative bridge matching problem that takes the HOI initialization together with RGB and 3D cues as input and outputs the refined HOI reconstruction. At inference time, test-time guidance adjusts the evolving states to better satisfy physical and image-space constraints.

Bridge matching. A key limitation of vanilla flow matching is that it assumes a fixed Gaussian prior, whose stochastic nature can hinder physical accuracy in many deterministic tasks. To address this, the Bridge Matching (Albergo et al., 2025) framework alleviates this by learning transport between arbitrary distributions $\pi _ { 0 }$ and $\pi _ { 1 }$ . Given $\mathbf { x } _ { 0 } \sim \pi _ { 0 }$ and $\mathbf { x } _ { 1 } \sim \pi _ { 1 }$ , the trajectory can be written as

$$
\begin{array} { r } { d \mathbf { x } _ { t } = \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) d t + \sigma ( t ) d \mathbf { W } _ { t } , } \end{array}\tag{2}
$$

where $\mathbf { v } _ { \theta }$ denotes the learnable drift and $\sigma ( t )$ controls the noise level. A common instantiation is the Brownian bridge with $\sigma ( t ) = \sigma \sqrt { t ( 1 - t ) }$ . The velocity field is trained by matching the drift:

$$
\mathcal { L } _ { \mathrm { b r i d g e } } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \left[ \left. \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) - ( \mathbf { x } _ { 1 } - \mathbf { x } _ { t } ) / ( 1 - t ) \right. _ { 2 } ^ { 2 } \right] .\tag{3}
$$

When $\sigma ( t ) = 0$ , Eq (3) reduces to deterministic flow matching. By replacing the Gaussian prior with a more plausible initial distribution, it reduces the stochasticity inherent in noise-driven generation. A key remaining question therefore is the choice of the source distribution. In this work, we define it using foundation-model-derived hand and object states, which will be described in the next section.

## 3.2 HAND AND OBJECT STATES INITIALIZATION

The goal is to leverage foundation models to obtain initial hand and object geometry and their per-frame interaction initialization $\mathbf { s } _ { 0 } ^ { i } = [ \mathbf { h } _ { 0 } ^ { i } , \mathbf { o } _ { 0 } ^ { i } ]$ , where i indexes video frames. For brevity, we will omit the frame index i. These estimates provide coarse yet informative hand and object pose, which serve as the starting point of our conditional generative bridge. While foundation models has made significant progress, they treat hand, object, and the context separately and do not consider their intricate interactions. To address this, we introduce the following methods for reconstructing hand, object, and their interaction. Please refer to Appendix for the detailed process.

Contextual scene parsing. We first perform holistic 3D scene understanding to reduce ambiguity in monocular reconstruction and generate outputs for object and hand states estimation. Specifically, for each frame, we obtain the object mask M via vision language grounding (Comanici et al., 2025) and segmentation models (Carion et al., 2026). We also estimate depth $\mathbf { D } _ { i }$ and camera parameters using MoGe-3 (Kong et al., 2026), which provide geometric cues for both hand and object reconstruction. These outputs establish spatial consistency across frames. Figure 2-(a) illustrates the main idea.

Hand reconstruction and state estimation. For hand initialization, we employ WiLoR (Potamias et al., 2025) to estimate per-frame MANO (Romero et al., 2022) parameters and 2D keypoints, with hand bounding boxes extracted from masks. While WiLoR provides reliable hand articulation and rotation estimates, its translation prediction is limited by the weak-perspective camera assumption, which cannot accurately recover metric camera-space depth. To address this, we solve a lightweight closed-form least-squares to estimate the hand translation. Figure 2-(d) shows the pipeline.

Object reconstruction and state estimation. We estimate a canonical object mesh m<sub>o</sub> and temporally consistent per-frame poses $\{ \mathbf { o } ^ { i } \} _ { i = 1 } ^ { N }$ . We sample sparse keyframes with diverse viewpoints and reconstruct the mesh in normalized space using pre-trained multi-view generative models (Comanici et al., 2025; Zhao et al., 2025). Compared with single-view generation (Chen et al., 2026a), this reduces ambiguity under occlusion and improves texture quality. We then align the object with the scene using depth cues and an enhanced FoundationPose (Wen et al., 2024), as shown in Figure 2-(b). The resulting states provide plausible structural alignment but remain susceptible to occlusion-induced instability and generative artifacts, motivating subsequent bridge matching refinement.

## 3.3 4D HOI FLOW MATCHING

After we obtained initialized hand-object states ${ \bf s } _ { 0 }$ from the input video sequence, we learn a conditional generative model that transports these states toward physically plausible interactions. We parameterize the conditional velocity field $\mathbf { v } _ { \theta } ( \mathbf { s } _ { t } , t )$ using a neural network that takes as input the current state $\mathbf { s } _ { t } .$ , flow time $t ,$ and multi-modal conditions derived from the input video.

Conditional bridge matching. As discussed in Section 3.1, directly applying Eq. (1) to HOI reconstruction is challenging due to the large solution space and ambiguity of monocular observations. To address this issue, we inject an initialization state $\mathbf { s } _ { 0 }$ through the bridge matching formulation in Eq. (3), which guides the transport process toward physically plausible hand-object configurations.

We further encode geometric structure by representing both the hand and object as point clouds. Specifically, we sample points from the object mesh $\mathbf { m } _ { o }$ and use MANO vertices parameterized by shape coefficients $\beta$ for the hand representation. These point clouds are processed using a Point Transformer V3 backbone (Wu et al., 2024) to capture spatial relationships between the hand and object. To incorporate visual context, we extract image features from each frame using a pretrained DINOv2 encoder (Oquab et al., 2023). For efficient 4D modeling, we aggregate multi-frame features and feed them into a DiT backbone for temporal reasoning and conditional flow prediction. Detailed architecture is shown in Figure S1. The bridge matching objective in Eq. (3) can be written as:

$$
\mathcal { L } _ { \mathrm { h o f } } = \mathbb { E } _ { \mathbf { s } _ { 0 } , \mathbf { s } _ { 1 } , t } \left[ \left\| \mathbf { v } _ { \theta } ( \mathbf { s } _ { t } , t , \mathbf { m } _ { o } , \beta , \mathcal { T } ) - ( \mathbf { s } _ { 1 } - \mathbf { s } _ { t } ) / ( 1 - t ) \right\| _ { 2 } ^ { 2 } \right] ,\tag{4}
$$

where $\mathbf { m } _ { o }$ denotes the object mesh, $\beta$ the hand shape, and $\mathcal { T }$ the sequence of input images. This objective learns to transport the initialized states ${ \bf s } _ { 0 }$ toward the target interaction distribution.

Training objectives. We introduce three auxiliary losses to improve reconstruction quality: 3D position loss $\mathcal { L } _ { 3 \mathrm { D } }$ , interaction loss ${ \mathcal { L } } _ { \mathrm { i n t e r } } $ , and regularization loss $\bar { \mathcal { L } _ { \mathrm { r e g } } }$

We apply regression-based (Pavlakos et al., 2024) 3D position losses $\mathcal { L } _ { 3 \mathrm { D } }$ to the terminal estimate $\mathbf { s } _ { 1 } ,$ , recovered from the generated trajectory since flow matching predicts velocities rather than states. The 3D loss consists of three components: (1) a hand loss supervising the 3D MANO joints $\mathbf { J } _ { 3 \mathrm { D } } , ( 2 )$ an object loss supervising the transformed object point cloud $\mathbf { P } _ { \mathrm { : } }$ and (3) a relative loss supervising the spatial relationship between the hand and the object through their relative Euclidean distances:

$$
\mathcal { L } _ { \mathrm { 3 D } } = \| { \bf J } _ { \mathrm { 3 D } } - { \bf J } _ { \mathrm { 3 D } } ^ { * } \| _ { 1 } + \| { \bf P } - { \bf P } ^ { * } \| _ { 1 } + \| { \bf D } ( { \bf P } , { \bf J } _ { \mathrm { 3 D } } ) - { \bf D } ( { \bf P } ^ { * } , { \bf J } _ { \mathrm { 3 D } } ^ { * } ) \| _ { 1 } ,\tag{5}
$$

where P is the point cloud sampled from the transformed object mesh $\mathbf { m } _ { o } , \mathbf { D } ( \mathbf { P } , \mathbf { J } _ { 3 \mathrm { D } } )$ is the distance matrix for each transformed object point-hand joint pair. The asterisk denotes the ground truth.

To encourage physically plausible interactions, we further incorporate contact and penetration-based constraints (Jiang et al., 2021), denoted by ${ \mathcal { L } } _ { \mathrm { { i n t e r } } }$ , between the hand mesh and object point cloud:

$$
\mathcal { L } _ { \mathrm { i n t e r } } = \sum _ { j \in \Omega } d ( p _ { j } , \mathcal { H } ) + \sum _ { j \in \mathcal { P } } d ( p _ { j } , \mathcal { H } ) ,\tag{6}
$$

where $d ( p _ { j } , \mathcal { H } )$ is the distance between object point $p _ { j }$ to MANO mesh surface $\mathcal { H } , \Omega$ is the contact set derived from ground-truth pose and $\mathcal { P }$ is the object point set that penetrates into the hand mesh. The first term encourages contact at ground-truth contact regions, while the second penalizes penetration.

In addition, we empirically observe that the hand estimates provided by the initialization are accurate in most cases. Therefore, the refinement process should make only subtle adjustments to improve physical plausibility, rather than performing unconstrained optimization that may introduce undesirable artifacts. To this end, we introduce a regularization term, $\mathcal { L } _ { \mathrm { r e g } } ,$ , on hand articulation related channels $\mathbf { v } _ { \theta , \mathrm { h a n d } }$ to preserve the initialized hand motion during refinement:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e g } } = \left\| \mathbf { v } _ { \theta , \mathrm { h a n d } } ( \mathbf { s } _ { t } , t , \mathbf { m } _ { o } , \beta , \mathcal { T } ) \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{7}
$$

The overall training objective combines the bridge matching loss with the auxiliary terms:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { h o f } } + \mathcal { L } _ { \mathrm { 3 D } } + \mathcal { L } _ { \mathrm { i n t e r } } + \mathcal { L } _ { \mathrm { r e g } } .\tag{8}
$$

Test-time guidance. An additional benefit of our bridge matching formulation is that the continuous transport process provides a flexible interface for test-time control. During inference, task-specific constraints can be directly incorporated by steering intermediate states along the learned transport trajectory, without performing expensive per-sequence optimization. This allows us to efficiently impose additional geometric and interaction constraints, such as contact, temporal consistency, and image-space alignment, while preserving the interaction prior learned by the model. Formally, we incorporate the following lightweight test-time guidance $G ( \mathbf { s } _ { t } )$ into the bridge matching process:

$$
\begin{array} { r } { \mathbf { s } _ { t } = \mathbf { s } _ { t } - \nabla _ { \mathbf { s } _ { t } } G ( \mathbf { s } _ { t } ) . } \end{array}\tag{9}
$$

The guidance function $G$ combines three complementary objectives: silhouette guidance $G _ { \mathrm { s i l } }$ for image alignment, interaction guidance $G _ { \mathrm { i n t e r } }$ for physical correctness, and temporal smoothness guidance $\bar { G } _ { \mathrm { t e m p } }$ for motion consistency.

The silhouette guidance $G _ { \mathrm { s i l } }$ aligns the projected object geometry with the observed segmentation. We sample object surface points and project them onto the image plane to obtain U. Let B denote sampled boundary pixels of the visible object mask, and let M be the union of the object mask and the dilated hand mask. The guidance is defined as

$$
G _ { \mathrm { s i l } } = \frac { 1 } { 2 | \mathcal { U } | } \sum _ { \mathbf { u } \in \mathcal { U } } \rho ( D _ { \mathcal { M } } ( \mathbf { u } ) + D _ { \Psi } ( \mathbf { u } ) ) + \frac { 1 } { 2 | \mathcal { B } | } \sum _ { \mathbf { b } \in \mathcal { B } } \rho ( D _ { \mathcal { U } } ( \mathbf { b } ) ) ,\tag{10}
$$

where $D _ { \mathcal { M } } ( \mathbf { u } )$ is the distance from u to the admissible region M, $D _ { \Psi } ( \mathbf { u } )$ projections outside the image, and $D _ { \mathcal { U } } ( \mathbf { b } )$ is the distance from b to its nearest projected point in U. Here, ρ denotes the smooth L1 penalty. The first term penalizes projections outside the object or hand-occluded regions, while the second term encourages coverage of the visible object boundary.

Our interaction guidance $G _ { \mathrm { i n t e r } }$ follows the interaction loss ${ \mathcal { L } } _ { \mathrm { { i n t e r } } }$ used during training, encouraging contact while penalizing interpenetration. Since ground-truth contact annotations are unavailable at test time, we use the learned contact prediction model HACO (Jung & Lee, 2026) to infer the contact index set Ω. This allows guidance to adapt to diverse hand-object interaction patterns without relying on manually predefined fixed contact regions (Fan et al., 2024; Hasson et al., 2019).

To further reduce temporal jitter, we introduce temporal smoothness guidance $G _ { \mathrm { t e m p } }$ that penalizes the velocities and accelerations of the hand and object:

$$
G _ { \mathrm { t e m p } } = \| \nabla \mathbf { s } ^ { i } \| _ { 2 } ^ { 2 } + \| \nabla ^ { 2 } \mathbf { s } ^ { i } \| _ { 2 } ^ { 2 } ,\tag{11}
$$

where $\nabla$ and $\nabla ^ { 2 }$ denote first- and second-order temporal finite differences, respectively. Translational increments are computed as differences between consecutive positions, while rotational increments are represented by the axis-angle vectors of the corresponding relative rotations. Second-order differences are computed from consecutive increments.

We apply guidance only during the later stages of flow-matching integration, when the predicted state is already close to a plausible interaction and requires only minor corrections. By concentrating refinement on these later steps, the guidance makes targeted adjustments that improve image alignment, interaction validity, and temporal consistency while introducing limited computational overhead.

Table 1: Results on HO3D-v3. Our method shows strong improvements on interaction metrics, which are often underreported in previous methods (Wang et al., 2025b; Shi et al., 2026). Hand and object accuracy are determined by foundation models and our method performs on par with AGILE (code not available).
<table><tr><td rowspan="2">Method</td><td colspan="4">Interaction</td><td>Hand</td><td colspan="3">Object</td><td>Robustness</td></tr><tr><td> $\mathrm { C D } _ { h } \downarrow$ </td><td> $\textrm { I V } _ { \downarrow }$ </td><td>CA↑</td><td>JM↓</td><td>MPJPE↓</td><td>CD↓</td><td>F5↑</td><td>F10↑</td><td>SR↑</td></tr><tr><td>HOLD</td><td>18.66</td><td>18.76</td><td>89.50</td><td>10.43</td><td>22.09</td><td>1.11</td><td>81.75</td><td>92.42</td><td>100.0</td></tr><tr><td>MagicHOI</td><td>21.81</td><td>17.41</td><td>83.00</td><td>21.66</td><td>7.38</td><td>0.90</td><td>76.74</td><td>91.59</td><td>83.3</td></tr><tr><td>AGILE</td><td>15.81</td><td></td><td></td><td></td><td>3.92</td><td>0.27</td><td>86.63</td><td>97.77</td><td>100.0</td></tr><tr><td>CHOIR</td><td>42.37</td><td>22.90</td><td>90.99</td><td>0.92</td><td>21.12</td><td>0.84</td><td>63.87</td><td>90.29</td><td>94.4</td></tr><tr><td>4D-HOF (ours)</td><td>5.98</td><td>13.44</td><td>97.27</td><td>1.61</td><td>4.40</td><td>0.28</td><td>89.11</td><td>97.13</td><td>100.0</td></tr></table>

## 3.4 DATA PREPARATION AND MODEL TRAINING

We train our model on multiple large-scale HOI datasets, including HO-Cap (Wang et al., 2024), OakInk (Yang et al., 2022), and SHOWMe (Swamy et al., 2023), which covering diverse interactions and object categories. Training excludes all evaluation benchmarks, and we use identical model weights across test datasets to assess cross-dataset generalization.

To efficiently leverage the available data and prevent overfitting to specific dataset pattern, we construct training pairs by perturbing the ground-truth HOI states $\mathbf { s } _ { 1 }$ with a randomly sampled noise (see Appendix for details). Unlike conventional generative modeling that transports samples from pure Gaussian noise, our formulation learns a conditional transport process between imperfect yet structured initialization states and physically plausible HOI states. This matches the practical inference setting, where modern foundation models already provide strong but imperfect estimates.

## 4 EXPERIMENTS

## 4.1 SETUP

Datasets. We evaluate our approach on two standard benchmarks: HO3D-v3 (Hampali et al., 2020) and DexYCB (Chao et al., 2021). HO3D-v3 features complex interactions resulting in severe occlusions, and DexYCB emphasizes tabletop grasping across diverse subjects. Following AGILE’s evaluation protocol (Shi et al., 2026), we use 18 HO3D-v3 and 20 DexYCB sequences; we refer readers to AGILE for sequence selection and evaluation details.

Baselines. We compare with state-of-the-art monocular HOI reconstruction methods: HOLD (Fan et al., 2024) jointly optimizes implicit hand-object fields; MagicHOI (Wang et al., 2025b) improves object geometry with novel-view synthesis priors. AGILE (Shi et al., 2026) combines agentic initialization with heuristic optimization, and CHOIR (Xu et al., 2026) integrates foundation-model initialization, learned relative-depth correction, and contact-aware joint optimization.

Metrics. We evaluate five complementary aspects. Hand accuracy: mean per-joint position error (MPJPE) [mm] for local articulation. Object accuracy: bidirectional Chamfer distance (CD) [cm<sup>2</sup>] and F-scores at 5/10 mm (F5/F10) [%] after ICP alignment of scale, rotation, and translation. Interaction consistency: hand-relative Chamfer distance $\left( \mathrm { C D } _ { h } \right) [ \mathrm { c m } ^ { 2 } ]$ (Fan et al., 2024), computed in the hand-root frame; interpenetration volume $\mathrm { ( I V ) [ c m ^ { 3 } ] }$ , measuring hand-object overlap; and contact accuracy (CA) [%] (Jiang et al., 2021), the fraction of frames whose contact states match the ground truth. Temporal stability: hand-joint trajectory jerk magnitude $\left( \mathbf { J } \mathbf { M } \right) [ \mathbf { m } / \mathrm { s } ^ { 3 } ]$ , with lower values indicating smoother motion. Robustness: per-sequence success rate (SR) [%] (Shi et al., 2026), requiring both hand and object reconstruction to complete without failure.

## 4.2 4D HOI RECONSTRUCTION COMPARISON

We compare 4D-HOF against state-of-the-art methods on HO3D-v3 (Table 1) and DexYCB (Table 2), and show qualitative results in Figure 3 and the supplementary video.

CHOIR

Table 2: Results on DexYCB (Chao et al., 2021) dataset.
<table><tr><td rowspan="2">Method</td><td>Hand</td><td colspan="2">Object</td><td></td><td>Interaction Robustness</td><td></td></tr><tr><td>MPJPE</td><td>CD↓</td><td>F5↑</td><td>F10↑</td><td> $\mathrm { C D } _ { h } \downarrow$ </td><td>SR↑</td></tr><tr><td>HOLD</td><td>30.86</td><td>19.30</td><td>33.20</td><td>54.94</td><td>170.90</td><td>45.0</td></tr><tr><td>MagicHOI</td><td>21.20</td><td>2.05</td><td>45.67</td><td>67.14</td><td>661.90</td><td>25.0</td></tr><tr><td>AGILE</td><td>19.06</td><td>0.52</td><td>83.219</td><td>95.43</td><td>94.60</td><td>100.0</td></tr><tr><td>CHOIR</td><td>27.24</td><td>0.84</td><td>79.469</td><td>93.09</td><td>134.74</td><td>100.0</td></tr><tr><td>4D-HOF (ours)</td><td>6.84</td><td>0.45</td><td>83.62</td><td>95.81</td><td>48.85</td><td>100.0</td></tr></table>

Table 3: Computational cost.
<table><tr><td>Method</td><td>Preproc.</td><td>Optim.</td><td>Total</td></tr><tr><td>4D-HOF</td><td>~23 min</td><td>~1 min</td><td>~24 min</td></tr><tr><td>HOLD</td><td>~65 min</td><td>~24 h</td><td>~25.1 h</td></tr><tr><td>MagicHOI</td><td>~25 min</td><td>~2 h</td><td>~2.4 h</td></tr><tr><td>AGILE</td><td>21–29 min ~1.75 h 2.1–2.2 h</td><td></td><td></td></tr><tr><td>CHOIR</td><td>~11min</td><td>~19min ~30min</td><td></td></tr></table>

![](images/7644d6a40cf036d14df25d2de76ab7a99ae0dbded5ef879a616de2cdbe331150.jpg)  
Ground Truth  
4D-HOF (Ours)  
HOLD  
MagicHOI

Figure 3: Qualitative results on HO3D-v3 (Hampali et al., 2020) and DexYCB (Chao et al., 2021). 4D-HOF consistently outperforms HOLD (Fan et al., 2024) and MagicHOI (Wang et al., 2025b) in preserving accurate geometry and hand-object interactions. Please see our supplementary video for detailed comparisons.

Results on HO3D-v3. 4D-HOF attains the lowest hand-to-object Chamfer distance and interpenetration volume among all methods, improving over the next best baseline by 62.2% in CD . The jerk magnitude is reduced to a level 6.4× lower than HOLD (Fan et al., 2024) and 13.4× lower than MagicHOI (Wang et al., 2025b), confirming that our learned interaction prior produces temporally stable trajectories without explicit smoothness regularization. Our method further achieves the highest contact ratio among reporting methods and the strongest F5 among all baselines.

Results on DexYCB. DexYCB contains diverse subjects performing tabletop grasps, where prior baselines suffer from large scale and translation errors. 4D-HOF yields a particularly pronounced margin across all the metrics comparing with other baselines.

Computational cost. We compare the inference time in Table 3. Existing methods rely on expensive test-time optimization and typically require hours to process a 200-frame sequence. In contrast, 4D-HOF performs reconstruction through a feed-forward generative transport process, reducing the refinement stage to approximately one minute while maintaining competitive reconstruction quality.

## 4.3 ABLATION STUDY AND ANALYSIS

We conduct ablation studies on HO3D-v3 (Hampali et al., 2020) to validate the design choices.

How to use vision foundation models? We first study different strategies for constructing the initial hand-object states (Table 4). (a) Replacing our multi-view object generation with SAM3D (Chen et al., 2026a) significantly degrades interaction quality, showing the importance of temporally consistent object reconstruction. (b) Using FoundationPose (Wen et al., 2024) in pure tracking mode also fails under severe occlusions, where a single mistracked frame can corrupt the entire sequence. (c) Directly using vision foundation model outputs without our generative bridge yields poor alignment and contact. (d) Adding guidance (but without bridge matching model) offers only modest improvements, suggesting the need for a learned interaction prior to refine independently estimated HOI states.

Table 4: Ablation of vision foundation models.
<table><tr><td rowspan="2">Method Variants</td><td colspan="4">Interaction</td><td>Hand</td></tr><tr><td> $\mathrm { C D } _ { h } \downarrow$ </td><td> $\textrm { I V } _ { \downarrow }$ </td><td>CA↑</td><td>JM↓</td><td>MPJPE↓</td></tr><tr><td>4D-HOF (full model)</td><td>5.98</td><td>13.44</td><td>97.27</td><td>1.61</td><td>4.40</td></tr><tr><td>(a) SAM3D object recon.</td><td>262.19</td><td>9.87</td><td>64.40</td><td>1.75</td><td>6.09</td></tr><tr><td>(b) FPose default tracking</td><td>15646.78</td><td>9.39</td><td>75.74</td><td>2.07</td><td>5.00</td></tr><tr><td>(c) VFM only</td><td>46.58</td><td>11.06</td><td>78.07</td><td>3.95</td><td>4.03</td></tr><tr><td>(d) VFM + guidance</td><td>41.75</td><td>11.89</td><td>81.53</td><td>3.71</td><td>4.03</td></tr></table>

Table 5: Ablation of the bridge matching model.
<table><tr><td rowspan="2">Method Variants</td><td colspan="4">Interaction</td><td>Hand</td></tr><tr><td> $\mathrm { C D } _ { h } \downarrow$ </td><td> $\textrm { I V } _ { \star }$ </td><td>CA↑</td><td>JM↓</td><td>MPJPE↓</td></tr><tr><td>4D-HOF (full model)</td><td>5.98</td><td>13.44</td><td>97.27</td><td>1.61</td><td>4.40</td></tr><tr><td>(a) w/o temporal design</td><td>18.93</td><td>14.23</td><td>95.64</td><td>4.26</td><td>4.15</td></tr><tr><td>(b) w/o explicit encoding</td><td>11.92</td><td>14.76</td><td>96.41</td><td>1.62</td><td>4.40</td></tr><tr><td>(c) w/o object optimization</td><td>90.49</td><td>9.63</td><td>62.23</td><td>2.31</td><td>5.06</td></tr><tr><td>(d) w/o hand optimization</td><td>12.34</td><td>13.05</td><td>96.02</td><td>3.95</td><td>4.03</td></tr><tr><td>(e) w/o guidance</td><td>8.03</td><td>13.39</td><td>96.62</td><td>1.71</td><td>4.37</td></tr></table>

Table 7: Ablation studies on guidance terms.

Table 6: Ablation studies on loss functions.
<table><tr><td rowspan="2">L Variants</td><td colspan="3">Interaction</td><td rowspan="2">Hand</td></tr><tr><td> $\mathrm { C D } _ { h } \downarrow$ </td><td>IV↓ CA↑</td><td>JM↓</td></tr><tr><td>4D-HOF</td><td>5.98</td><td>13.44 97.27</td><td>1.61</td><td>MPJPE↓ 4.40</td></tr><tr><td>w/o  $\mathcal { L } _ { 3 \mathrm { D } }$ </td><td>7.68</td><td>15.94 96.27</td><td>2.93</td><td>3.98</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { i n t e r } }$ </td><td>6.02</td><td>13.78</td><td>97.00 1.63</td><td>4.60</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r e g } }$ </td><td>6.71</td><td>13.78</td><td>97.15 1.10</td><td>5.64</td></tr></table>

<table><tr><td rowspan="2">G Variants</td><td colspan="3">Interaction</td><td>Hand</td></tr><tr><td> $\mathrm { C D } _ { h } \downarrow$ </td><td>IV↓ CA↑</td><td>JM↓</td><td>MPJPE↓</td></tr><tr><td>4D-HOF</td><td>5.98</td><td>13.44 97.27</td><td>1.61</td><td>4.40</td></tr><tr><td>w/o  $G _ { \mathrm { s i l } }$ </td><td>7.51</td><td>13.85 97.06</td><td>1.72</td><td>4.75</td></tr><tr><td>w/o  $G _ { \mathrm { i n t e r } }$ </td><td>7.81</td><td>16.00 95.07</td><td>1.63</td><td>4.74</td></tr><tr><td> $\mathrm { w } / \mathrm { o } G _ { \mathrm { t e m p } }$ </td><td>6.35</td><td>14.53</td><td>97.29 1.91</td><td>4.74</td></tr></table>

How to design the HOI flow matching model? We next analyze the design of the conditional bridge (Table 5). (a): Removing temporal modeling produces unstable 4D trajectories, suggesting that independent frame-wise refinement cannot capture interaction dynamics. (b): Removing the explicit hand-object structural encoding consistently degrades interaction metrics, highlighting the importance of explicitly modeling relative hand-object geometry. (c) and (d): Optimizing only the hand or only the object also leads to worse interaction quality, indicating that hand and object states must be refined jointly to resolve their mutual ambiguity. (e): Removing test-time guidance degrades contact and interaction quality, showing that guidance provides effective test-time control toward geometrically and physically consistent HOI states. Notably, these variants leave the object geometry unchanged, while contact quality and physical interaction degrade significantly.

How to supervise 4D-HOF training? We ablate each auxiliary losses in Table 6. Removing $\mathcal { L } _ { 3 \mathrm { D } }$ causes severe degradation across most metrics, showing that explicit 3D supervision is critical for HOI reconstruction. Removing ${ \mathcal { L } } _ { \mathrm { i n t e r } }$ reduces contact quality and hand pose accuracy, indicating that the contact loss encourages meaningful physical interaction. Removing $\mathcal { L } _ { \mathrm { r e g } }$ leads to the largest drop in hand pose, suggesting that regularization helps preserve the hand state from foundation models.

How to guide 4D-HOF inference? We ablate each test-time guidance term in Table 7. Removing $G _ { \mathrm { s i l } }$ degrades all metrics, suggesting that image-space alignment helps refine the reconstructed interaction. Removing $G _ { \mathrm { i n t e r } }$ leads to the largest increase in interpenetration and the largest drop in contact accuracy, demonstrating its importance for plausible interactions. Removing $G _ { \mathrm { t e m p } } ^ { - }$ causes the largest increase in motion jitter, confirming that temporal guidance improves motion consistency.

## 5 CONCLUSION AND FUTURE WORKS

We propose 4D-HOF, a novel feed-forward framework for 4D hand-object interaction reconstruction from monocular video. Our key innovation is to replace noise-driven generation with initialization derived from foundation models, and to learn a conditional generative bridge that transports these initial states toward physically plausible interactions. Our formulation enables joint refinement of hand, object, and their interactions in a feed-forward manner. Experiments show that our approach achieves strong performance in interaction quality while maintaining competitive hand and object accuracy, with significantly improved inference efficiency. One limitation, shared by many foundation model-based methods, is its sensitivity to initialization failures (Figure S6). Our current formulation focuses on single-hand interactions with rigid objects. Extending 4D-HOF to more complex settings, such as bimanual interactions and articulated or deformable objects, is an important future direction.

## REFERENCES

Ahmed Tawfik Aboukhadra, Marcel Rogge, Nadia Robertini, Abdalla Arafa, Jameel Malik, Ahmed Elhayek, and Didier Stricker. Ghost: Fast category-agnostic hand-object interaction reconstruction from rgb videos using gaussian splatting. arXiv preprint arXiv:2603.18912, 2026.

Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal ofMachine Learning Research, 26(209):1–80, 2025.

Ayce Idil Aytekin, Xu Chen, Zhengyang Shen, Thabo Beeler, Helge Rhodin, Rishabh Dabral, and Christian Theobalt. Grasp in gaussians: Fast monocular reconstruction of dynamic hand-object interactions. arXiv preprint arXiv:2604.12929, 2026a.

Ayce Idil Aytekin, Helge Rhodin, Rishabh Dabral, and Christian Theobalt. Follow my hold: Handobject interaction reconstruction through geometric guidance. In Thirteenth International Conference on 3D Vision, 2026b.

Blender Online Community. Blender: A 3d modelling and rendering package, 2026. URL https: //www.blender.org.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International conference on learning representations, volume 2026, pp. 138846–138923, 2026.

Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, et al. Dexycb: A benchmark for capturing hand grasping of objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9044–9053, 2021.

Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, et al. Sam 3d: 3dfy anything in images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7220–7232, 2026a.

Yuantao Chen, Jiahao Chang, Chongjie Ye, Chaoran Zhang, Zhaojie Fang, Chenghong Li, and Xiaoguang Han. Forehoi: Feed-forward 3d object reconstruction from daily hand-object interaction videos. arXiv preprint arXiv:2602.06226, 2026b.

Sammy Christen, Shreyas Hampali, Fadime Sener, Edoardo Remelli, Tomas Hodan, Eric Sauser, Shugao Ma, and Bugra Tekin. Diffh2o: Diffusion-based synthesis of hand-object interactions from textual descriptions. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Zicong Fan, Maria Parelli, Maria Eleni Kadoglou, Xu Chen, Muhammed Kocabas, Michael J Black, and Otmar Hilliges. Hold: Category-agnostic 3d reconstruction of interacting hands and objects from video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 494–504, 2024.

Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. Honnotate: A method for 3d annotation of hand and object poses. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 3196–3206, 2020.

Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J Black, Ivan Laptev, and Cordelia Schmid. Learning joint reconstruction of hands and manipulated objects. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 11807–11816, 2019.

Yana Hasson, Gul Varol, Cordelia Schmid, and Ivan Laptev. Towards unconstrained joint hand-object¨ reconstruction from rgb videos. In 2021 International Conference on 3D Vision (3DV), pp. 659–668. IEEE, 2021.

Jiawei He, Danshi Li, Xinqiang Yu, Zekun Qi, Wenyao Zhang, Jiayi Chen, Zhaoxiang Zhang, Zhizheng Zhang, Li Yi, and He Wang. Dexvlg: Dexterous vision-language-grasp model at scale. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14248–14258, 2025.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Mingzhen Huang, Fu-Jen Chu, Bugra Tekin, Kevin J Liang, Haoyu Ma, Weiyao Wang, Xingyu Chen, Pierre Gleize, Hongfei Xue, Siwei Lyu, et al. Hoigpt: Learning long-sequence hand-object interaction with language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7136–7146, 2025a.

Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mv-adapter: Multi-view consistent image generation made easy. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 16377–16387, 2025b.

Hanwen Jiang, Shaowei Liu, Jiashun Wang, and Xiaolong Wang. Hand-object contact consistency reasoning for human grasps generation. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 11107–11116, 2021.

Daniel Jung and Kyoung Mu Lee. Learning dense hand contact estimation from imbalanced data. Advances in Neural Information Processing Systems, 38:120351–120384, 2026.

Lingyu Kong, Ruicheng Li, Ruicheng Wang, Sicheng Xu, Chengtang Yao, Jianfeng Xiang, and Jiaolong Yang. Moge-3: Fine-detail monocular geometry estimation with self-guided sparse volumetric refinement. arXiv e-prints, pp. arXiv–2607, 2026.

Muchen Li, Sammy Christen, Chengde Wan, Yujun Cai, Renjie Liao, Leonid Sigal, and Shugao Ma. Latenthoi: On the generalizable hand object motion generation with latent hand diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17416–17425, 2025.

Dixuan Lin, Tianyou Wang, Zhuoyang Pan, Yufu Wang, Lingjie Liu, and Kostas Daniilidis. Zero-shot reconstruction of in-scene object manipulation from video. arXiv preprint arXiv:2512.19684, 2025a.

Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025b.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022a.

Xingchao Liu, Lemeng Wu, Mao Ye, and Qiang Liu. Let us build bridges: Understanding and extending diffusion generative models. arXiv preprint arXiv:2208.14699, 2022b.

Xingyu Liu, Pengfei Ren, Qi Qi, Haifeng Sun, Zirui Zhuang, Jing Wang, Jianxin Liao, and Jingyu Wang. Generalizable hand-object modeling from monocular rgb images via 3d gaussians. Advances in Neural Information Processing Systems, 38:127318–127340, 2025a.

Xueyi Liu and Li Yi. Geneoh diffusion: Towards generalizable hand-object interaction denoising via denoising diffusion. arXiv preprint arXiv:2402.14810, 2024.

Yumeng Liu, Xiaoxiao Long, Zemin Yang, Yuan Liu, Marc Habermann, Christian Theobalt, Yuexin Ma, and Wenping Wang. Easyhoi: Unleashing the power of large models for reconstructing handobject interactions in the wild. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 7037–7047, 2025b.

Haowen Luo, Yunze Liu, and Li Yi. Physics-aware hand-object interaction denoising. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2341–2350, 2024.

Eric Marchand, Hideaki Uchiyama, and Fabien Spindler. Pose estimation for augmented reality: a hands-on survey. IEEE transactions on visualization and computer graphics, 22(12):2633–2651, 2015.

Kirill Mazur, Nikita Karaev, Matthew Chang, Jitendra Malik, and Nur Muhammad Shafiullah. AgentSTAR: Agentic shape tracking and reconstruction from monocular videos, 2026. URL https://arxiv.org/abs/2609.24487.

Jeongwan On, Kyeonghwan Gwak, Gunyoung Kang, Junuk Cha, Soohyun Hwang, Hyein Hwang, and Seungryul Baek. Bigs: Bimanual category-agnostic interaction reconstruction from monocular videos via 3d gaussian splatting. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 17437–17447, 2025.

OpenAI. ChatGPT (6-astra) [large language model]. https://chatgpt.com, 2026. Accessed: 2026-09-25.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Bhawna Paliwal, Haritheja Etukuru, William Liang, Pieter Abbeel, Nur Muhammad Mahi Shafiullah, and Jitendra Malik. Do as i do: Dexterous manipulation data from everyday human videos. arXiv preprint arXiv:2606.19333, 2026.

Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing hands in 3d with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9826–9836, 2024.

Vladimir I Pavlovic, Rajeev Sharma, and Thomas S. Huang. Visual interpretation of hand gestures for human-computer interaction: A review. IEEE Transactions on pattern analysis and machine intelligence, 19(7):677–695, 2002.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou. Wilor: End-to-end 3d hand localization and reconstruction in-the-wild. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 12242–12254, 2025.

Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: Modeling and capturing hands and bodies together. arXiv preprint arXiv:2201.02610, 2022.

Jin-Chuan Shi, Binhong Ye, Tao Liu, Xiaoyang Liu, Yangjinhui Xu, Junzhe He, Zeju Li, Hao Chen, and Chunhua Shen. Agile: Hand-object interaction reconstruction from video via agentic generation. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–11, 2026.

Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrodinger¨ bridge matching. Advances in neural information processing systems, 36:62183–62223, 2023.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Anilkumar Swamy, Vincent Leroy, Philippe Weinzaepfel, Fabien Baradel, Salma Galaaoui, Romain Bregier, Matthieu Armando, Jean-Sebastien Franco, and Gr ´ egory Rogez. Showme: Benchmarking´ object-agnostic hand-object 3d reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pp. 1935–1944, 2023.

Eugene Valassakis and Guillermo Garcia-Hernando. Handdgp: Camera-space hand mesh prediction with differentiable global positioning. In European Conference on Computer Vision, pp. 479–496. Springer, 2024.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294–5306, 2025a.

Jikai Wang, Qifan Zhang, Yu-Wei Chao, Bowen Wen, Xiaohu Guo, and Yu Xiang. Ho-cap: A capture system and dataset for 3d reconstruction and pose tracking of hand-object interaction. arXiv preprint arXiv:2406.06843, 2024.

Shibo Wang, Haonan He, Maria Parelli, Christoph Gebhardt, Zicong Fan, and Jie Song. Magichoi: Leveraging 3d priors for accurate hand-object reconstruction from short monocular video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5957–5968, 2025b.

Zikai Wang, Zhilu Zhang, Yiqing Wang, Hui Li, and Wangmeng Zuo. Arthoi: Taming foundation models for monocular 4d reconstruction of hand-articulated-object interactions. arXiv preprint arXiv:2603.25791, 2026.

Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Muller, Alex Evans, Dieter¨ Fox, Jan Kautz, and Stan Birchfield. Bundlesdf: Neural 6-dof tracking and 3d reconstruction of unknown objects. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 606–617. IEEE, 2023.

Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 17868–17879, 2024.

Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, and Stan Birchfield. Foundationstereo: Zero-shot stereo matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5249–5260, 2025.

Bowen Wen, Shaurya Dewan, and Stan Birchfield. Fast-foundationstereo: Real-time zero-shot stereo matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7513–7524, 2026.

Thomas Wimmer, Prune Truong, Marie-Julie Rakotosaona, Michael Oechsle, Federico Tombari, Bernt Schiele, and Jan Eric Lenssen. Anyup: Universal feature upsampling. In International Conference on Learning Representations, volume 2026, pp. 140700–140720, 2026.

Xiaofei Wu, Yi Zhang, Yumeng Liu, Yuexin Ma, Yujiao Shi, and Xuming He. Affordgrasp: Crossmodal diffusion for affordance-aware grasp synthesis. arXiv preprint arXiv:2603.08021, 2026.

Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 4840–4851, 2024.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14419–14429, 2026.

Xianghui Xie, Bowen Wen, Chang Yan, Hesam Rabeti, Jiefeng Li, Ye Yuan, Gerard Pons-Moll, and Stan Birchfield. CARI4D: Category agnostic 4d reconstruction of human-object interaction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2026.

Hao Xu, Yilin Liu, Yinqiao Wang, Chi-Wing Fu, and Niloy J Mitra. Choir: Contact-aware 4d hand-object interaction reconstruction. arXiv preprint arXiv:2605.20992, 2026.

Lixin Yang, Kailin Li, Xinyu Zhan, Fei Wu, Anran Xu, Liu Liu, and Cewu Lu. Oakink: A large-scale knowledge repository for understanding hand-object interaction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20953–20962, 2022.

Yufei Ye, Abhinav Gupta, and Shubham Tulsiani. What’s in your hands? 3d reconstruction of generic objects in hands. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3895–3905, 2022.

Yufei Ye, Poorvi Hebbar, Abhinav Gupta, and Shubham Tulsiani. Diffusion-guided reconstruction of everyday hand-object interaction clips. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 19717–19728, 2023.

Yufei Ye, Abhinav Gupta, Kris Kitani, and Shubham Tulsiani. G-hop: Generative hand-object prior for interaction reconstruction and grasp synthesis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1911–1920, 2024.

Yufei Ye, Jiaman Li, Ryan Rong, and C Karen Liu. Whole: World-grounded hand-object lifted from egocentric videos. arXiv preprint arXiv:2602.22209, 2026.

Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.

Yiming Zhong, Qi Jiang, Jingyi Yu, and Yuexin Ma. Dexgrasp anything: Towards universal robotic dexterous grasping with physics awareness. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22584–22594, 2025.

Keyang Zhou, Bharat Lal Bhatnagar, Jan Eric Lenssen, and Gerard Pons-Moll. Toch: Spatio-temporal object-to-hand correspondence for motion refinement. In European Conference on Computer Vision, pp. 1–19. Springer, 2022.

Zhifan Zhu, Siddhant Bansal, Shashank Tripathi, and Dima Damen. Reconstructing objects along hand interaction timelines in egocentric video. arXiv preprint arXiv:2512.07394, 2025.

## Table of Contents in Appendix

A Implementation Details 16   
A.1 Hand and Object State Initialization 16   
A.1.1 Contextual scene parsing. 16   
A.1.2 Hand reconstruction and state estimation. 16   
A.1.3 Object reconstruction and state estimation. 16   
A.2 4D HOI Flow Matching 17   
A.2.1 Model architecture 17   
A.2.2 Training data curation 18   
A.2.3 Training details . 18   
B More Experimental Results 18   
B.1 HO3D-v3 MagicHOI’s setting 18   
B.2 Comparison of Object Generation 18   
B.3 Do more HOI data help? 19   
B.4 Comparison with GPT 6 Astra Ultra (as of Sep 20, 2026) 20   
C Failure Case Analysis 22   
D VLM Prompt 22   
D.1 Interaction object reasoning . 22   
D.2 Multi-view image generation 22   
D.3 Texture selection 23

## A IMPLEMENTATION DETAILS

In this section, we describe additional details of our method. We first present the hand and object state initialization pipeline based on various foundation models in Section A.1, followed by details of the flow-matching model in Section A.2.

## A.1 HAND AND OBJECT STATE INITIALIZATION

## A.1.1 CONTEXTUAL SCENE PARSING.

Given an input video I, we first employ a VLM (Comanici et al., 2025) to understand the interaction and identify the object manipulated by the human hand. Since processing all video frames is computationally prohibitive, we uniformly sample five frames from the video and prompt the VLM to describe the target object. The resulting object description, together with the prompt “hand,” is then provided to SAM 3 (Carion et al., 2026) for segmentation.

## A.1.2 HAND RECONSTRUCTION AND STATE ESTIMATION.

For hand estimation, we use WiLoR (Potamias et al., 2025) to predict the MANO (Romero et al., 2022) parameters and 2D keypoints. We then recover the hand translation in camera coordinates using the camera intrinsics from MoGe-3 (Kong et al., 2026; Valassakis & Garcia-Hernando, 2024). Specifically, for each joint, the perspective projection yields the 2D position $( u , v )$ of each 3D joint following equations:

$$
u = f _ { x } \frac { \mathbf { J } _ { x } + \mathbf { t } _ { x } } { \mathbf { J } _ { z } + \mathbf { t } _ { z } } + c _ { x } ,\tag{S1}
$$

$$
v = f _ { y } \frac { \mathbf { J } _ { y } + \mathbf { t } _ { y } } { \mathbf { J } _ { z } + \mathbf { t } _ { z } } + c _ { y } ,\tag{S2}
$$

where $( \mathbf { J } _ { x } , \mathbf { J } _ { y } , \mathbf { J } _ { z } )$ denotes the position of 3D joint in the canonical hand space, $\left( \mathbf { t } _ { x } , \mathbf { t } _ { y } , \mathbf { t } _ { z } \right)$ represents the global hand translation, and $f _ { x } , f _ { y } , c _ { x } , c _ { y }$ are the camera intrinsic parameters. By aggregating constraints from all joints, we construct a linear system and solve it using least squares.

## A.1.3 OBJECT RECONSTRUCTION AND STATE ESTIMATION.

In order to automatically select a compact yet informative set of observations for the object from the video, we propose a visual feature–based frame selection strategy. Specifically, we extract image features using a pre-trained DINOv2 (Oquab et al., 2023) image encoder. Since DINO only yields patch-level embeddings, we also apply AnyUp (Wimmer et al., 2026) to enhance their spatial resolution. We then pool the upsampled features within the object mask region to obtain an object descriptor for each frame.

After having the object descriptors, the next step is picking the frames that are visually different. The frame with the largest object mask is first chosen as the anchor, and Farthest Point Sampling (FPS) is performed in the descriptor space to select a visual diverse set of frames. These selected frames are used as context prompts for canonical view generation, and the synthesized views are subsequently fed into a multi-view-to-3D model (Zhao et al., 2025) to recover the object mesh. Although the object geometry is generally reliable, textures produced by generative models are often unsatisfactory, preserving coarse color spots but failing on fine-grained patterns and text. To address this limitation, we construct two types of texture maps: one obtained directly from the generative model, which is suitable for unstructured objects with simple appearance, and another derived by projecting the canonical views (Huang et al., 2025b), which better preserves detailed patterns for structured objects. We render the object under multiple views using both texture types and employ a VLM to automatically select the optimal variant.

To better leverage FoundationPose (Wen et al., 2024) for object pose estimation, we initialize the pose on the frame with the largest object mask and propagate it bidirectionally throughout the sequence. Instead of relying on the default tracking mode of FoundationPose, in which a single occluded frame may lead to catastrophic error accumulation, we perform independent per-frame registration while explicitly rejecting pose hypotheses that deviate significantly from the previous estimate. Specifically, we use the geodesic distance between adjacent frame rotations as the filtering criterion and set the threshold to $6 0 ^ { \circ }$

![](images/49847b0eaca7e11a9425c28f1835cd30503b36f7bc1f11f4fd3757b0057e43f3.jpg)  
Figure S1: Model architecture. Our model adopts a DiT-style architecture for conditional flow matching. The latent hand-object state is processed through temporal self-attention, point-stream cross-attention using Point Transformer V3 features, and visual cross-attention using DINO features. Adaptive layer normalization conditioned on the timestep is applied throughout the network to modulate the generative transport process.

We further employ silhouette-based filtering to further enforce visual consistency. For each pose hypothesis, we render an object mask $\tilde { \mathbf { M } } ^ { o }$ and compute its intersection-over-union (IoU) with the segmentation mask ${ \bf M } ^ { o }$ . To reduce the influence of hand occlusion, we exclude the hand mask $\mathbf { M } ^ { h }$ region before computing the IoU:

$$
\mathrm { I o U } = \frac { ( \tilde { \mathbf { M } } ^ { o } \cap \sim \mathbf { M } ^ { h } ) \cap ( \tilde { \mathbf { M } } ^ { o } \cap \sim \mathbf { M } ^ { h } ) } { ( \tilde { \mathbf { M } } ^ { o } \cap \sim \mathbf { M } ^ { h } ) \cup ( \tilde { \mathbf { M } } ^ { o } \cap \sim \mathbf { M } ^ { h } ) } .\tag{S3}
$$

We retain only the top 30 pose hypotheses with the highest IoU scores. To further improve robustness, we enable both RGB and RGB-D modes in the pose refinement and scoring stages of FoundationPose, and compute the final pose score as the average of the two modalities.

## A.2 4D HOI FLOW MATCHING

## A.2.1 MODEL ARCHITECTURE

Our flow matching model transports the noisy hand and object pose from the foundation model-based initialization to a plausible interaction pose manifold. Since we aim to reconstruct the 4D hand object interaction, a natural design is directly handling the sequence data, which benefits from the temporal information interaction. Specifically, our model takes a hand and object pose sequence as input. In each frame, the hand pose and object pose are concatenated together and tokenized by a linear layer. This pose token sequence is then passed thought several stacked Transformer blocks, after that the hand and object pose in each frame are decoded from the corresponding pose token.

There are three attention operations in each Transformer block. Our block starts with self-attention operation along the temporal dimension with RoPE (Su et al., 2024) for temporal embedding. This role of this layer is to utilize the temporal corrections in the sequence. In a motion sequence, the pose divergence usually small for nearby frames and large for far away frames, besides the motion evolution in the sequence should be smooth. By learning large scale data, this layer can make the pose sequences coherent and smooth. The second layer is a cross-attention layer between the 3D HOI point cloud sequence and pose token. The motivation of this operation is to inject explicit 3D spatial information to the model, since it’s impossible to understand the hand object interaction solely from a set of bare rotations and translation. To prepare 3D cues, we use the pose results from the initialization stage to transform the MANO hand and object, then use a Point Transformer V3 (Wu et al., 2024) to encode them. In each frame, we use MANO vertices as hand point cloud and sample points on the object surface as the object point cloud. We treat the two components as a single point cloud and add a class label to each point. After feature extraction, a max pooling operation is applied to construct the final HOI point cloud token sequence. By this cross attention, the model can obtain explicit 3D information which boosts joint optimization. The third one is another cross attention with the 2D image features, since we hope the reconstruction results respect the observation. For each video frame, we use a frozen DINOv2-B to extract visual features and use Plucker ray to add camera information to each patch embedding. To balance the computational cost, we only apply cross attention within each frame. A standard feed-forward layer is added at the end of block, and adaptive layer norm is used before each attention and FFN later to inject flow-step information.

Table S1: Results on HO3D-v3 (Hampali et al., 2020) following MagicHOI’s setting.
<table><tr><td rowspan="2">Method</td><td>Hand</td><td colspan="4">Object</td><td rowspan="2">Interaction</td></tr><tr><td>MPJPE↓</td><td>CD↓</td><td>F5↑</td><td>F10↑</td><td>RS↓</td></tr><tr><td>iHOI</td><td>27.75</td><td>2.37</td><td>35.78</td><td>62.11</td><td>0.17</td><td>25.45</td></tr><tr><td>DiffHOI</td><td>16.02</td><td>2.30</td><td>39.59</td><td>64.49</td><td>0.13</td><td>33.33</td></tr><tr><td>EasyHOI</td><td>16.69</td><td>1.86</td><td>46.10</td><td>70.92</td><td>0.28</td><td>19.55</td></tr><tr><td>HOLD</td><td>30.79</td><td>1.31</td><td>57.20</td><td>80.23</td><td>0.62</td><td>21.28</td></tr><tr><td>MagicHOI</td><td>4.62</td><td>0.87</td><td>69.72</td><td>92.15</td><td>0.11</td><td>2.39</td></tr><tr><td>CHOIR</td><td>5.55</td><td>0.77</td><td>72.33</td><td>96.03</td><td>0.10</td><td>2.41</td></tr><tr><td>4D-HOF (ours)</td><td>4.60</td><td>0.24</td><td>90.24</td><td>98.13</td><td>0.13</td><td>2.17</td></tr></table>

## A.2.2 TRAINING DATA CURATION

We initially explored training with initializations produced by our foundation model pipeline, but observed two limitations. First, object symmetries introduce ambiguities in the estimated rotations, complicating the construction of interpolation paths for bridge matching and subsequent training. Second, initialization errors exhibit dataset-specific biases, limiting the effectiveness of these initializations as a training source. We therefore adopt a dataset-agnostic perturbation strategy that synthetically corrupts ground-truth HOI states to emulate imperfect initializations.

We introduce synthetic noise to the ground-truth HOI states to emulate the imperfect initializations generated by the foundation models. To perturb the object pose, we sample a large offset (3.14 rad, 0.2 m) across 20% of the frames to simulate catastrophic pose estimation failures, while applying a smaller offset (0.1 rad, 0.02 m) to the remaining 80%. For the hand parameters, the translation vector is perturbed using noise sampled from a uniform spatial distribution (0.02, 0.02, 0.1). Meanwhile, the global orientation and the local pose of the hand are perturbed by adding Gaussian noise with a standard deviation of 0.1.

## A.2.3 TRAINING DETAILS

We optimize our model using AdamW with a learning rate of 0.0001 and a weight decay of 0.0001, regulated by a cosine learning rate scheduler with a warmup phase. The training process spans 20 epochs with a batch size of 16. During feature extraction, input images are resized to 224×224 before being encoded by DINOv2. To establish an explicit 3D encoding, we uniformly sample 4096 points from the object mesh surface. Finally, the temporal sequence length is set to 16 frames during training and extended to 128 frames during inference. The entire training process takes approximately 30 hours on a single NVIDIA H100 GPU.

## B MORE EXPERIMENTAL RESULTS

## B.1 HO3D-V3 MAGICHOI’S SETTING

We empirically observe that some methods perform better on shorter clips, motivating an additional evaluation using the same sequences and frames as MagicHOI (Wang et al., 2025b). Table S1 and Figure S2 present the quantitative and qualitative results, respectively. Under this protocol, 4D-HOF achieves the best results on five of the six metrics, with substantial improvements in object reconstruction accuracy.

## B.2 COMPARISON OF OBJECT GENERATION

We compare our multiview-based object generation pipeline with SAM3D (Chen et al., 2026a), with quantitative and qualitative results presented in Table S2 and Figure S3, respectively. Limited to

![](images/6008e6c43359efe6909c78214da3bcd13b01e8713c537370284d83691de73005.jpg)

![](images/db0f30c762c3b44b9b8a8f480544be7ff06634334dbd9cb433613684072e699d.jpg)

![](images/50b2bf5d80818cbc85e597ae32650bfe603dd0a76834804edbb5786349e54d61.jpg)

![](images/4eb00d09df3d1cb1d328ff20245f0fc5e82b07305858f02c673fc5412b49ef69.jpg)

![](images/689eff09847a1979ca22943fb931fc82c0d52f9da3c858e1d7d4903b31e539e3.jpg)

![](images/b1b630850f9520158a61b8a44c9c538b9836bafcbe35e11a40f29a98ffb05ed5.jpg)

![](images/3878064b78161a1bf55862c9f8eef0df69f0c2bfbafc5d34d3d1f14b40ab8587.jpg)

![](images/501af2acd3a79cbc69493b0b98dd59b26559d0521441360b867fcdd44f9277c5.jpg)

![](images/4edac4e163416a9623459d7ca9d94c9343bc65bb5c1bffc99e82165039c22a56.jpg)

![](images/b10ca78b9af721e3641c79ce902c0a16f1325085465a52c2c4a2215b7b1dd3c6.jpg)

![](images/13c01f986f22d4581ac55131a98a1c01352c20a0cd0f7bedc88356a03a93a66c.jpg)  
Input Frame

![](images/38bbff36fe65053c2e91cd34fc09dd3d9e5f84550c79c66b74f9960214a7c628.jpg)  
Ground Truth

![](images/b9bb38a3431b85f554912a5191c0e95d8acf82c4296389750d7b880e9de00618.jpg)  
4D-HOF (Ours)

![](images/2022ee268bfce1d9c6adf69997d71078d274b5262f0f9c7af16bdb4881bbfa22.jpg)  
HOLD

![](images/4bfb8c07cf4fab917cf8db26642a3e55ceda10d7590aa814c41e991267d713c4.jpg)  
MagicHOI

![](images/9a3d1b942c948b2051aeca62f0d42cb91cedb5a11e8e063a18d2de193ad8c5ad.jpg)  
CHOIR

Figure S2: Qualitative results on HO3D-v3 (Hampali et al., 2020) following MagicHOI’s setting.  
Table S2: Object generation comparison. Our multiview-based pipeline outperforms SAM3D across all metrics, demonstrating the benefit of complementary visual observations for object reconstruction.
<table><tr><td>Methods</td><td>CD↓</td><td>F5↑</td><td>F10↑</td><td>RS↓</td></tr><tr><td>SAM3D</td><td>0.47</td><td>77.54</td><td>96.20</td><td>0.17</td></tr><tr><td>4D-HOF (Ours)</td><td>0.28</td><td>89.11</td><td>97.13</td><td>0.12</td></tr></table>

a single input image, SAM3D produces less reliable textures, which can compromise downstream pose estimation with FoundationPose, where rendered object appearance provides a key cue for pose evaluation. By combining complementary observations from multiple views with a powerful image generation model, our pipeline produces objects with higher geometric accuracy and texture fidelity.

## B.3 DO MORE HOI DATA HELP?

To investigate the impact of training data volume, we evaluate our model’s performance on the HO3D dataset using the $\mathrm { C D } _ { h }$ metric (Figure S4). The results indicate that increasing the size of the training dataset yields consistent performance gains, demonstrating the scalability of our approach.

![](images/139a442821b921ce343fc951e04327d72d516b78a681984b68653a15ff7f416f.jpg)  
GT

![](images/a61dca5858033cdda565c4826f3aea8aec0496a6dfc335cb1625e7541f18534e.jpg)  
4D-HOF

![](images/e7c112f8098a7bcf58338e173f118f03bba7fd0ebacfccb9ccbee707bb51c480.jpg)  
SAM 3D

![](images/ba5004b04f67fb8920ad13efbb825d2f0885375c5f2cb79b672883dbc1ef904d.jpg)

![](images/6a91dda702cef134896cef30d2801bcefb2e2decb0abded66f97efbb7403ee0c.jpg)

![](images/38710c0c8e767b7967d6b95c916f9015d46c24497e4d29ec01f283ba8c62751d.jpg)  
GT

![](images/dd099b29a7e63c61fed6d070a39bd699f02f1066b889a02f0655b2c2d2a3c47c.jpg)  
4D-HOF

![](images/90c691d87f41e515aad4bce05c6e2701625c6f8fceccc65ada99abc8d2ffe55e.jpg)

![](images/208cd2f530dd482230381487779c254ba184a66014aeed223a9f4bf451a915eb.jpg)  
SAM 3D

Figure S3: Qualitative comparison of object generation. We show two views of each object reconstructed by 4D-HOF and SAM3D (Chen et al., 2026a) alongside the ground truth. Our pipeline better preserves object geometry and texture details, including labels and fine structures. Ground-truth object widths are provided for scale reference.  
![](images/0cc365727eaeb3ae4bdface824df56251d4b1b8ff740fab2d11bf39df6dfc78c.jpg)  
Figure S4: Dataset scaling analysis for bridge matching model training.

## B.4 COMPARISON WITH GPT 6 ASTRA ULTRA (AS OF SEP 20, 2026)

Recent GPT-6 (OpenAI, 2026) models have demonstrated strong capabilities in 3D reasoning and tool use. To provide a broader and more rigorous evaluation of our model, we constructed an agentic baseline by prompting GPT-6-Astra with Ultra reasoning effort and providing access to Foundation-Pose (Wen et al., 2024), WiLoR (Potamias et al., 2025), MoGe-3 (Kong et al., 2026), Blender (Blender Online Community, 2026), and related tools. We tasked the agent with reconstructing 4D hand–object interactions from the ABF12 and SM2 sequences of HO3D dataset , without imposing execution-time or token-budget limits. Our prompt is illustrated in B.4. Qualitative comparisons are shown in Figure S5. Although GPT-6 recovers plausible coarse hand and object poses, it exhibits noticeable hand–object separation and weaker cross-view contact consistency. In contrast, our method better pre serves instance-specific object geometry and produces tighter, more physically plausible hand–object configurations: in SM2, the fingers more consistently wrap around the bottle, while in ABF12, our reconstruction better captures the complex fingertip contact and provide consistent result across viewpoints. These results suggest that, despite GPT-6’s strong general-purpose 3D reasoning ability, the proposed pipeline provides more accurate geometry and more coherent hand–object interactions.

Beyond reconstruction quality, the agentic baseline incurs substantial runtime and resource overhead. In our experiments, a single run typically required several hours and consumed approximately 35% of the weekly usage allowance associated with our \$100 subscription plan, making large-scale evaluation impractical under this budget. These observations highlight the challenges of applying general-purpose agentic systems to efficient, contact-consistent HOI reconstruction. Our feed-forward module could serve as a specialized component within such systems to accelerate reconstruction while improving geometric and interaction consistency.

![](images/daa68d9158d5c3eae0725635fb374246b9ccf5ed57dabcd00aae052f7241e689.jpg)  
Figure S5: Qualitative comparison between GPT6-Astra and our 4D-HOF.

<table><tr><td>GPT6 HOI Reconstruction Prompt</td></tr><tr><td>Two third-person-view human hand-object interaction videos is available at: data/ABF12.mp4, data/SM2.mp4</td></tr><tr><td>For each video, reconstruct: 1. A temporally consistent MANO hand sequence. 2. A textured mesh for each manipulated object. 3. A per-frame 6DoF object pose sequence.</td></tr><tr><td>The hand and object should share a consistent metric coordinate system. Minimize hand–object penetration and separation, keep motion smooth, preserve plausible physical contact, and ensure rendered hand/object meshes align closely with the original RGB frames in 2D. Handle occlusion and tracking loss without large pose jumps, scale changes, or</td></tr><tr><td>identity switches. If additional Python packages are needed, create a virtual environment in the current directory rather than modifying the system environment. Blender and Blender-mcp are installed and</td></tr><tr><td>enabled. - Use MoGe-3 (https://github.com/microsoft/MoGe) for monocular depth estimation. - Use WiLoR (https://github.com/rolpotamias/WiLoR) for MANO hand reconstruction, plus additional optimization procedures. - Use FoundationPose (https://github.com/NVlabs/FoundationPose) for object 6DoF pose</td></tr><tr><td>estimation, and make suitable modifications if you need. Create textured mesh and adjust mesh scale before running the model. You can use the depth from MoGe 3 as input. Output: - MANO parameters, joints, and vertices.</td></tr><tr><td>- Textured object meshes. - Per-frame object poses and confidence. - Camera parameters and coordinate-system documentation. - RGB overlay videos and necessary 3D/diagnostic visualizations. - README and other documents containing setup, commands, methods, optimization details, output paths, validation results, and known limitations.</td></tr></table>

![](images/e152d3686c2200465fa61ec570b6bcb070859e289ab52fe0425e357ee42328c5.jpg)  
Figure S6: Failure analysis. When the estimated point map contains severe artifacts or out-of-distribution predictions, the resulting initialization may become unreliable, which can affect the flow-matching refinement process and lead to reconstruction failures.

## C FAILURE CASE ANALYSIS

Our pipeline is reliant on several vision foundation models, with the monocular depth estimator, MoGe-3, acting as a critical geometric anchor. MoGe-3 (Kong et al., 2026) recovers the object’s metric scale to match the hand and coarsely aligns both components within a unified coordinate space. Consequently, if the predicted point map exhibits significant artifacts (Figure S6), these inaccuracies cascade directly into the object scale and pose estimation modules. Such highly corrupted, out-ofdistribution initializations ultimately misguide the flow-matching refinement model, resulting in catastrophic reconstruction failures.

## D VLM PROMPT

Our hand and object states initialization pipeline employs VLM for multiple purposes, including identifying the manipulated object, generating multi-view images, and selecting object textures. We describe each task below and provide the corresponding prompts.

## D.1 INTERACTION OBJECT REASONING

Given an input video, the VLM identifies the object manipulated by the hand and generates a textual description. This description is then passed to SAM3 as a prompt for automatic object segmentation.

Object Identification Prompt

Describe the object interaction with hand, only return in [color object name] format.

## D.2 MULTI-VIEW IMAGE GENERATION

We select object keyframes from the input video and provide them to an image generation model to synthesize canonical views of the object. The placeholder {description} in the prompt is filled with the object description obtained in the preceding reasoning step.

Canonical View Generation Prompt

Generate the front, back, left, right, top, bottom canonical views of this {description}. The generated views MUST perfectly match the exact shape, texture, color, and geometric details of the object in the reference images. Complete the occluded regions. Present as a 2\*3 grid, front, back, left in the upper row, right, top, bottom in the lower row. Pure white background. The image must not contain text indicating viewpoint direction.

After each generation attempt, the VLM acts as a critic to assess whether the generated image satisfies the specified requirements. If the image fails this evaluation, we regenerate it.

## Generated View Evaluation Prompt

You are an expert evaluator. Look at the provided generated image at the very end. It is supposed to be a 2x3 image grid showing the front, back, left, right, top, and bottom canonical views of the object in the reference images.

Does the generated image strictly follow these requirements:

1. It is exactly a 2x3 grid.

2. Background is pure white.

3. No text indicating viewpoint direction.

4. Identity and texture are preserved.

Answer EXACTLY ’Yes’ or ’No’ without any explanation.

If both consecutive attempts fail, the VLM selects the better of the two candidates, allowing the pipeline to proceed.

## Candidate Image Selection Prompt

I have two generated images at the end, each supposed to be a 2x3 grid showing canonical views of the same object from the preceding reference images. Image 1 is the previous best. Image 2 is the newly generated one.

Compare them based on these criteria:

1. Exactly a 2x3 grid layout.

2. Pure white background.

3. No text or labels.

4. High visual quality and accurate canonical views.

Which image strictly follows these instructions better and has better quality?

Return EXACTLY ’Image 1’ or ’Image 2’ without any explanation or markdown formatting.

## D.3 TEXTURE SELECTION

Given the reconstructed object geometry, we obtain two candidate texture maps through direct generation and image projection, respectively. We apply each texture map to the geometry and render the resulting textured object from multiple viewpoints. The VLM then compares these multi-view renderings and selects the better texture.

## Texture Selection Prompt

I have two 2x3 image grids showing canonical views of the same 3D object’s geometry. Image 1 represents Model A. Image 2 represents Model B. Important: The 3D geometry is identical in both models, but the poses or viewpoints might differ between the two images. Please act as a 3D texture quality judge. Ignore any pose or orientation differences, and focus strictly on comparing the texture quality. Look for detail, sharpness, realism, and a lack of noise or artifacts. Which model has better texture quality?

Return EXACTLY ’Model A’ or ’Model B’ without any explanation or markdown formatting.