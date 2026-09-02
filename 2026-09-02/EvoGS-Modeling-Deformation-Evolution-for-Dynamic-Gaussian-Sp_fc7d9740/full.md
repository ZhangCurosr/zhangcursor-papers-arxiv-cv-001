# EvoGS: Modeling Deformation Evolution for Dynamic Gaussian Splatting

Wei Dong , Shahram Shirani , Jun Chen , Han Zhou<sup>†</sup>

The Department of Electrical & Computer Engineering, McMaster University

![](images/68e9c6e1a060428c478255b3e878c2ff225f9a3e7d8ca937efed81c7a2566c8a.jpg)  
Figure 1: Existing dynamic 3DGS methods often estimate deformations separately at each timestamp, which can lead to temporal artifacts and unstable reconstruction. EvoGS maintains persistent deformation states and performs evolution-based correction. The qualitative results show that integrating EvoGS improves different dynamic Gaussian backbones, producing sharper details and better-preserved geometry.

## Abstract

Recent extensions of3D Gaussian Splatting (3DGS) enable real-time novel view synthesis in dynamic scenes by learning timeconditioned Gaussian deformations. However, existing MLP-based methods typically estimate deformations independently at each timestamp, making them less robust to large or abrupt motions. To address this issue, we propose EvoGS, a 3DGSbased dynamic reconstructionframework that models Gaussian deformation as a temporal evolution process. EvoGS maintains persistent deformation states for each Gaussian, extrapolates future states from historical deformation states, and corrects the predictions with MLP-derived observations. The correction is adaptively weighted using a temporal residual memory and evolution statistics such as deformation velocity and trajectory deviation. To further improve reconstruction quality, EvoGS introduces deformation-aware densification. Clone and split operations are performed along corrected deformation directions, while an uncertainty-aware strategy suppresses densification for Gaussians with unstable deformation histories. Experiments show that EvoGS improves dynamic novel view synthesis quality and achieves competitive performance across benchmarks.

• Computing methodologies → Rendering; Computer graphics; Neural networks;

## 1. Introduction

As a fundamental task in computer vision, dynamic 3D scene reconstruction is to reconstruct and render scenes involving object motion with temporal consistency, geometric accuracy, and high visual fidelity, which has wide applications including augmented realityvirtual reality (ARVR), autonomous navigation, and commercial advertising. Recent years have witnessed significant progress driven by Neural Radiance Fields (NeRF [MST<sup>∗</sup>20]) and its extensive variants [DZY<sup>∗</sup>21, GSKH21, PSH<sup>∗</sup>21, PSB<sup>∗</sup>21, PCP-MMN21, FYW<sup>∗</sup>22a], which model scenes using time-conditioned MLPs to represent geometry and appearance over time. While powerful in capturing complex motion, these methods are notoriously slow due to the cost of volumetric ray marching and implicit representation, rendering them unsuitable for real-time applications.

To address the efficiency bottleneck, 3D Gaussian Splatting (3DGS [KKLD23]) has emerged as a new paradigm for realtime rendering with high-quality results. By representing the scene as a collection of textured 3D Gaussians projected onto the image plane, 3DGS allows for real-time view synthesis [KKLD23, SJL<sup>∗</sup>24, KVN24, ZDC25]. Recent extensions of 3DGS into dynamic settings attempt to model scene deformation over time by learning per-Gaussian offsets using MLPs. Notable works such as 4D-GS [WYF<sup>∗</sup>24], SC-GS [HSY<sup>∗</sup>24], D-3DGS [YGZ<sup>∗</sup>24], MotionGS [ZLC<sup>∗</sup>24], Grid4D [XFYX24], DASH [CHW<sup>∗</sup>25], MoDec-GS [KKJ<sup>∗</sup>25] adopt frame-wise regression networks to estimate deformation at each timestep (Fig. 1). However, these methods typically treat each frame independently, without modeling temporal context in the input sequences. This unavoidably results in inconsistent deformation trajectories across frames and perceptual flickering in the rendered video sequences (see Fig. 2).

To address this issue, we propose EvoGS, a dynamic 3D Gaussian Splatting framework designed to model the temporal EVOlution of Gaussian deformations. Unlike existing MLP-based approaches that query the deformation of each timestamp independently, EvoGS treats deformation estimation as an evolving state modeling problem. For each Gaussian primitive, we maintain a persistent deformation state buffer that records corrected deformation states across training timesteps. Given historical states, EvoGS extrapolates a predicted deformation state for the current timestep and further refines it with an observation derived from a deformation MLP. The correction is adaptively weighted according to a temporal residual memory and motion evolution statistics, such as deformation velocity and angular deviation, allowing the model to balance historical prediction and current observation. Furthermore, EvoGS leverages the corrected deformation states for deformation-aware densification, where clone and split operations are performed along corrected deformation directions instead of relying only on local perturbations around the parent Gaussian. To avoid expanding unreliable primitives, we also introduce an uncertainty-aware densification criterion that suppresses densification for Gaussians exhibiting unstable deformation evolution.

Our contributions are summarized as follows:

⋄ We propose EvoGS, a 3DGS-based framework that models Gaussian deformation as a temporal evolution process for dynamic scene reconstruction.

⋄ We introduce persistent deformation state modeling, where historical corrected states are stored, extrapolated, and corrected using MLP-derived observations for more reliable estimation.

⋄ We develop an adaptive correction strategy driven by temporal residual memory and deformation evolution statistics, including deformation velocity and angular deviation.

⋄ We propose deformation-aware densification, which guides clone/split operations using corrected deformations and suppresses densification for Gaussians with unstable deformation histories.

⋄ Extensive experiments demonstrate that EvoGS improves dynamic view synthesis quality across multiple benchmarks.

## 2. Related Works

Dynamic NeRF. Dynamic novel view synthesis remains challenging due to temporal variation in input images. The success of NeRF [MST<sup>∗</sup>20] in representing static scenes with implicit neural fields has inspired its adaptation to dynamic settings. Some methods incorporate temporal information implicitly through timeconditioned inputs or latent embeddings [DZY<sup>∗</sup>21, GSKH21], while others [PSH<sup>∗</sup>21, PSB<sup>∗</sup>21, PCPMMN21, FYW<sup>∗</sup>22a, K<sup>∗</sup>25] learn explicit deformation fields that transform timestamped 3D points to a canonical space. Additional approaches separate scenes into static and dynamic parts [SCL 23], apply 4D grid-based structures [FKMW<sup>∗</sup>23, LSW<sup>∗</sup>22, WTL<sup>∗</sup>23, CJ23a] or use keyframes for redundancy reduction [LSZ<sup>∗</sup>22a, AHR<sup>∗</sup>23]. Despite progress, spatial-temporal coupling and the computational overhead of temporal modeling remain fundamental challenges.

Dynamic Gaussian Splatting. Gaussian Splatting (GS) [KKLD23] excels in rendering efficiency and fidelity for complex scenes. Recent advances have extended 3DGS for dynamic scenes [QLW<sup>∗</sup>25, FCL<sup>∗</sup>24, LZL<sup>∗</sup>25, KYW25, YYPZ23, YGZ<sup>∗</sup>24, WYF<sup>∗</sup>24, CJ23b, HSY<sup>∗</sup>24, SLX<sup>∗</sup>25, ZLZ<sup>∗</sup>25, LTH<sup>∗</sup>25, LGH<sup>∗</sup>24, XWZ<sup>∗</sup>25, YPW<sup>∗</sup>25, LPL<sup>∗</sup>25]. Yang et al. [YGZ<sup>∗</sup>24] employ an MLP to generate a time-dependent deformation field over the canonical Gaussian space. Wu et al. [WYF<sup>∗</sup>24] propose a spatial-temporal encoder with multi-resolution Hex-Planes [CJ23b] and a compact MLP to learn Gaussian deformation. SC-GS [HSY<sup>∗</sup>24] predict time-varying transformations of sparse control points via an MLP, whose interpolated effects drive Gaussian motion. E-D3DGS [BKY<sup>∗</sup>24] employs per-Gaussian and temporal embeddings for fine-grained motion modeling. While these methods can model frame-wise motion, they typically rely on temporally conditioned networks without modeling motion dynamics across frames, which may lead to temporal inconsistencies in complex scenes. MotionGS [ZLC<sup>∗</sup>24] addresses this via a motion-guided Gaussian deformation framework supervised by decoupled optical flow. However, its reliance on externally estimated optical flow limits its ability to robustly capture consistent motion under uncertain or ambiguous dynamics.

Temporal State-Space Modeling. State-space models and Kalman-inspired neural networks provide a principled perspective for modeling temporal evolution through recursive prediction and correction. Early deep Kalman-style models combine latent statespace formulations with deep generative or discriminative networks [KSS15, HPZ<sup>∗</sup>16, FKPW17], enabling end-to-end learning of temporal dynamics. KalmanNet [BRSE21] preserves the recursive structure of classical Kalman filtering while learning adaptive gains from data, and Horvath et al. [HOY23] further analyze the approximation ability of deep Kalman filters in non-Markovian settings. Given that directly employing image restoration methods [ZCLL22,ZDL<sup>∗</sup>24,ZDL<sup>∗</sup>25,DZZ<sup>∗</sup>24,DZLC26] to video contents introduces inconsistency across frames, KEEP [FLL24] introduces a Kalman-inspired method for stable video face restoration. More recently, KFD-NeRF [ZLN<sup>∗</sup>24] introduces a Kalman-style update into dynamic NeRF by using locally linear motion prediction to guide deformation learning. Although these methods demonstrate the benefit of recursive state modeling, their formulations are not directly designed for Gaussian primitives in dynamic 3DGS, where each Gaussian undergoes time-varying deformation and densification. In contrast, EvoGS models Gaussian deformation as a persistent temporal evolution process. Instead of independently regressing deformations at each timestamp, EvoGS maintains per-Gaussian deformation states, extrapolates future states from historical trajectories, and adaptively corrects them using MLP-derived observations and evolution statistics.

![](images/d8b199230d9d2820947fa3a1b81dc1ae279f70a491f1bbce54edeb2cc2e2bc6e.jpg)  
Figure 2: Under a fixed camera pose with varying timesteps $( \Delta t = 0 . 0 1 )$ , the baseline 4DGS $I W Y F ^ { * } 2 4 J$ produces blurry and temporally inconsistent results, where the shell shape graduallyfades acrossframes. In the baseline result, the colored arrows highlight typical artifacts: yellow indicates geometric distortion, while blue marks blurred wood textures.In contrast, EvoGS preserves sharper details, more stable geometry, and temporally coherent appearances in dynamic regions through persistent temporal deformation evolution. This evolution process also indirectly reduces ambiguity in static regions by mitigating motion-induced inconsistencies, leading to better overall reconstruction.

![](images/d01de4e06d5581360b9ac1e92a8b633610d1650da48b6bfea4db157e1a7b20eb.jpg)  
Figure 3: Qualitative comparison of Gaussian deformation trajectories. The baseline 4DGS exhibits scattered motion traces with localjitter, crossings, and outlier trajectories around the moving subject.In contrast, EvoGS(4DGS) yields more compact and structurealigned trajectories over time.The reduced trajectory dispersion suggests more stable deformation evolution. Notably, only Gaussians satisfying the visibility, opacity, motion-magnitude, and trackability criteria are visualized.

## 3. Preliminary

## 3.1. Radiance Modeling via 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) [KKLD23] represents a scene using a set of spatially continuous, anisotropic volumetric primitives $\mathcal { G } = \{ \pmb { g } _ { i } | i = 1 , . . . , N \}$ , where N represents the number of Gaussian primitives. Each point ${ \pmb { g } } _ { j }$ in the scene is modeled as a Gaussian blob parameterized by: ${ \bf { \mathbf { \mathbf { \mathbf { \mathbf { \mathbf { \mathit { g } } } } } } } } _ { i } =$ $\left\{ \mu _ { i } \in \mathbb { R } ^ { 3 } , s _ { j } \in \mathbb { R } ^ { 3 } , \pmb { q } _ { i } \in \mathbb { R } ^ { 4 } , \pmb { \sigma } _ { i } \in ( 0 , 1 ) , \pmb { c } _ { j } \in \mathbb { R } ^ { 3 } \right\}$ , where $\pmb { \mu } _ { i }$ is the 3D center, $\mathbf { } _ { s _ { i } }$ is the scale vector encoded as a diagonal covariance, $\pmb { q } _ { i }$ is the quaternion representing rotation, $\sigma _ { i }$ is the opacity, and $c _ { j }$ denotes its spherical harmonics-encoded color. Notably, the scale and rotation jointly determine the 3D covariance matrix of each Gaussian $\pmb { \Sigma } _ { i } \dot { = } \pmb { q } _ { i } \pmb { s } _ { i } \pmb { s } _ { i } ^ { T } \pmb { q } _ { i } ^ { T }$

For rendering, each $\pmb { g } _ { i }$ is projected into the 2D space using the following 2D covariance matrix $\pmb { \Sigma } ^ { \prime } \colon \pmb { \Sigma } _ { i } ^ { \prime } = \pmb { J } \pmb { V } \pmb { \Sigma } _ { i } \pmb { V } ^ { T } \pmb { J } ^ { T }$ , where J is the Jacobian of the affine approximation of the projective transformation, V symbolizes the view matrix, transitioning from world to camera coordinates. Then, the resulting 2D ellipsoids are rasterized and composited via alpha blending based on depth ordering:

$$
\begin{array} { c } { { \displaystyle C ( { \pmb p } ) = \sum _ { i = 1 } ^ { N } T _ { i } \alpha _ { i } c _ { i } , \quad T _ { i } = \Pi _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , } } \\ { { \alpha _ { i } = \sigma _ { i } e ^ { - \frac { 1 } { 2 } ( { \pmb p } - { \pmb \mu } _ { i } ^ { \prime } ) ^ { T } \Sigma ^ { \prime } ( { \pmb p } - { \pmb \mu } _ { i } ^ { \prime } ) } , } } \end{array}\tag{1}
$$

where $C ( \pmb { p } )$ denotes the rendered color of pixel $_ p$ on the image plane, and $\mu _ { i } ^ { \prime }$ represents the coordinates of 3D Gaussian center projected onto the 2D image plane. This pipeline circumvents volumetric ray integration and enables real-time differentiable rendering.

## 3.2. Temporal State Prediction and Correction

Temporal state-space models provide a natural framework for estimating a latent state from historical predictions and current observations. A classical example is the Kalman filter, which considers a linear state-space model:

$$
{ \pmb x } _ { t } = { \pmb F } { \pmb x } _ { t - 1 } + { \pmb w } _ { t } , \quad { \pmb z } _ { t } = { \pmb H } { \pmb x } _ { t } + { \pmb m } _ { t } ,\tag{2}
$$

where x<sub>t</sub> is the latent state, z<sub>t</sub> the observation, $\pmb { F }$ the statetransition matrix, H the observation matrix, and ${ \pmb w } _ { t } \sim \mathcal { N } ( { \bf 0 } , Q )$ , $\scriptstyle m _ { t } \sim { \mathcal { N } } ( 0 , R )$ denote process and measurement noise with covariances $\ b { Q }$ and $\scriptstyle { \mathbf { } } ,$ respectively. The latent state is recursively estimated through a prediction–correction procedure:

$$
\begin{array} { r } { \hat { { \pmb x } } _ { t } ^ { - } = { \pmb F } \hat { { \pmb x } } _ { t - 1 } , \qquad \hat { { \pmb x } } _ { t } = \hat { { \pmb x } } _ { t } ^ { - } + { \pmb K } _ { t } \left( { \pmb z } _ { t } - { \pmb H } \hat { { \pmb x } } _ { t } ^ { - } \right) , } \end{array}\tag{3}
$$

![](images/d0c6a3ba0db6884a20a6cdda21b7ea974ca89efe059356d1b80921a83478a2f3.jpg)  
Figure 4: The overall architecture ofEvoGS. Instead ofthe standard modelingfor Gaussian deformation using observation MLP, we propose the temporal deformation evolution mechanism to temporally learn Gaussian deformation field across time. Furthermore, we develop the deformation-aware densification strategy, which initializes newly created Gaussians based on the updated deformation and prevents densifi cation in regions with high deformation uncertainty.

with $H = I ,$ the update becomes $\hat { \pmb { x } } _ { t } = ( { \pmb I } - { \pmb K } _ { t } ) \hat { \pmb { x } } _ { t } ^ { - } + { \pmb K } _ { t } { \pmb z } _ { t }$ , which shows that the correction step adaptively interpolates between the historical prediction and the current observation.

In this work, we use this recursive prediction-correction principle as a conceptual basis for modeling Gaussian deformation evolution instead of implementing a classical Kalman filter or propagating covariance matrices. This is because explicitly maintaining per-Gaussian covariance matrices would introduce substantial memory and computational overhead for the large and dynamically changing Gaussian set, while reliable process and observation covariances are also difficult to define for nonlinear, non-rigid deformations.

## 4. Method

The overall framework of EvoGS is illustrated in Fig. 4. Specifically, EvoGS models dynamic Gaussian deformation as a temporal evolution process rather than independent per-timestamp regression, commonly used in existing approaches. For each Gaussian, EvoGS maintains persistent deformation states, extrapolates the current deformation from historical corrected states, and adaptively refines it using an MLP-derived observation. Sec. 4.1 introduces the persistent deformation buffer and the historical state extrapolation used to obtain the evolution prior. Sec. 4.2 describes the observation network with tri-plane spatial and B-spline temporal encoders, as well as the adaptive correction module based on innovation memory and motion evolution statistics. Furthermore, we develop the uncertainty-aware Gaussians control strategy to adjust the density of Gaussians for areas that exhibits with high motion uncertainty in Sec. 4.3. Moreover, different from the original Gaussians clone and split mechanism proposed in 3DGS [KKLD23], we propose the deformation-aware clone and split process for these new instantiated Gaussians. The optimization details of our method are discussed in Sec. 4.4.

## 4.1. Deformation Evolution with Persistent States

Given a canonical 3D Gaussian primitive ${ \bf \it _ { g } } _ { i } ,$ dynamic 3DGS methods usually predict its time-dependent deformation with a timeconditioned network. Instead of treating the deformation at each timestamp as an independent network output, EvoGS models Gaussian deformation as a persistent temporal state. We denote the deformation state of $\mathbf { \sigma } _ { g _ { i } }$ at timestamp t as $\pmb { d } _ { i } ^ { t } \in \mathbb { R } ^ { D }$ , which can represent position, scale, rotation, or other deformation offsets depending on the adopted dynamic 3DGS backbone.

Persistent Deformation States. For each Gaussian primitive ${ \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { }$ EvoGS maintains a persistent deformation buffer

$$
\boldsymbol { B } _ { g _ { i } } = \{ \tilde { d } _ { g _ { i } } ^ { 1 } , \tilde { d } _ { g _ { i } } ^ { 2 } , . . . , \tilde { d } _ { g _ { i } } ^ { T } \} ,\tag{4}
$$

where $\tilde { d } _ { g _ { i } } ^ { t }$ denotes the corrected deformation state of $\pmb { g } _ { i }$ at timestamp t. Given a target timestamp t, EvoGS queries a local historical window from this buffer:

$$
\mathcal { H } _ { g _ { i } } ^ { t } = \{ \tilde { d } _ { g _ { i } } ^ { t - 1 } , \tilde { d } _ { g _ { i } } ^ { t - 2 } \} \subset \mathcal { B } _ { g _ { i } } ,\tag{5}
$$

which provides a persistent motion history for each Gaussian and serves as the basis for predicting its current deformation. After the correction step in Sec. 4.2, the updated state $\tilde { d } _ { g _ { i } } ^ { t }$ is written back to the buffer and reused for subsequent timestamps.

Historical State Evolution. Given the local historical window $\mathcal { H } _ { i } ^ { t } ,$ EvoGS predicts an evolution prior for the current deformation state from the corrected historical states. We adopt a second-order temporal extrapolation:

$$
\begin{array} { r } { \hat { d } _ { g _ { i } } ^ { t } = 2 \tilde { d } _ { g _ { i } } ^ { t - 1 } - \tilde { d } _ { g _ { i } } ^ { t - 2 } , } \end{array}\tag{6}
$$

where $\hat { d } _ { i } ^ { t }$ denotes the predicted deformation state before incorporating the current observation. This design explicitly propagates the historical motion trend stored in the persistent deformation buffer. Since complex non-rigid motion may deviate from this extrapolated prior, EvoGS further corrects it using the current MLP-derived observation through the adaptive correction module in Sec. 4.2.

Discussion. The second-order extrapolation is used only as a lightweight short-horizon prior, rather than as the final deformation estimate or a strict constant-velocity motion assumption. When the motion deviates from this prior, especially under non-rigid or sudden changes, the current MLP-derived observation is used to correct the prediction through the adaptive correction module. Therefore, the extrapolation provides temporal continuity, while the observation branch compensates for deviations from the local motion trend. The current formulation assumes approximately uniform temporal spacing, as used by the evaluated benchmarks.

![](images/d5b3dbc35bf6c12ef6b49b3b47d63653c763ea55523d250bc17eeb2c5dc7feaf.jpg)  
(a) Vanilla Clone and Split process  
(b) Deformation-Aware Clone and Split Process  
Figure 5: Illustration of the proposed deformation-aware clone-and-split process. Compared with the vanilla strategy, our method integrates deformation cues to initialize new Gaussians with motion-adaptive geometry, leading to more consistent and stable updates acrossframes.

## 4.2. Adaptive Deformation Correction

The historical state evolution in Sec. 4.1 provides an evolution prior $\hat { d } _ { i } ^ { t }$ for each Gaussian primitive. However, this extrapolated prior may become inaccurate under abrupt or non-rigid motion. To account for such deviations, EvoGS further corrects the predicted state using a deformation observation estimated from the current timestamp.

Observation Network with Tri-plane Spatial and B-spline Temporal Encoders. We use an observation network Φ to estimate the current deformation observation:

$$
z _ { g _ { i } } ^ { t } = \Phi ( g _ { i } , t ) ,\tag{7}
$$

where $z _ { g _ { i } } ^ { t }$ denotes the MLP-derived deformation observation for Gaussian g at timestamp t. Following the dynamic 3DGS backbone, Φ predicts the same type of deformations. To improve efficiency and temporal smoothness, we parameterize Φ with tri-plane spatial features and a B-spline temporal encoder. Specifically, spatial features are sampled from the XY, XZ, and YZ feature planes according to the canonical Gaussian position, while the timestamp is encoded by a uniform cubic B-spline basis. The sampled spatial features and temporal embedding are then decoded by a lightweight MLP to produce $z _ { i } ^ { t } .$ . Compared with directly using sinusoidal positional encodings and a large MLP, this design preserves local spatial structure and provides smooth temporal interpolation.

Evolution-aware Correction Weight. The reliability of the extrapolated prior and the current observation varies across Gaussians and timestamps. EvoGS therefore predicts an adaptive correction weight from temporal evolution statistics.

(1) Temporal Residual Memory. We first compute the predictionobservation residual: $\pmb { r } _ { g _ { i } } ^ { t } = \pmb { z } _ { g _ { i } } ^ { t } - \hat { d } _ { g _ { i } } ^ { t }$ . To make the correction stable over time, we maintain a temporal residual memory:

$$
\begin{array} { r } { { \pmb m } _ { g _ { i } } ^ { t } = \beta { \pmb m } _ { g _ { i } } ^ { t - 1 } + ( 1 - \beta ) { \pmb r } _ { i } ^ { t } , } \end{array}\tag{8}
$$

where $\beta$ is a momentum coefficient. This memory records recent prediction uncertainty and prevents the correction from being dominated by a single noisy observation.

(2) Observation Instability. Based on the positional component in the persistent deformation buffer $\tilde { d } _ { g _ { i , x } } ^ { t - 3 } , \tilde { d } _ { g _ { i , x } } ^ { t - 2 } , \tilde { d } _ { g _ { i , x } } ^ { t - 1 }$ , we compute the magnitudes of deformation velocities across successive frames, $( \| \boldsymbol { v } _ { g _ { i } } ^ { t - 1 } \| _ { 2 }$ and $\| { \boldsymbol { v } } _ { g _ { i } } ^ { t - 2 } \| _ { 2 } )$ , as defined in Eq. 9, where large temporal displacements suggest abrupt motion and reduced predictive reliability.

(3) Angular Deviation. We further calculate the angle between adjacent motion vectors, given in Eq. 10. Significant directional changes imply motion irregularity and are treated as proxies for

elevated uncertainty.

$$
\| \pmb { v } _ { g _ { i } } ^ { \tau } \| _ { 2 } = \| \tilde { d } _ { g _ { i , x } } ^ { \tau } - \tilde { d } _ { g _ { i , x } } ^ { \tau - 1 } \| _ { 2 } , \qquad \tau \in \{ t - 1 , t - 2 \} ,\tag{9}
$$

$$
\Theta _ { g _ { i } } ^ { t } = \operatorname { a r c c o s } \left( \frac { \langle v _ { g _ { i } } ^ { t - 2 } , v _ { g _ { i } } ^ { t - 1 } \rangle } { \lVert v _ { g _ { i } } ^ { t - 2 } \rVert \cdot \lVert v _ { g _ { i } } ^ { t - 1 } \rVert + \epsilon } \right) ,\tag{10}
$$

where ϵ is set to $1 0 ^ { - 6 }$ for numerical stability. These uncertaintyrelated representations are fed to a lightweight MLP Ψ to estimate the gain: $\mathbf { \hat { { K } } } _ { g _ { i } } ^ { t } = \Psi ( \pmb { m } _ { g _ { i } } ^ { t } , \lVert \pmb { v } _ { g _ { i } } ^ { t - 2 } \rVert _ { 2 } , \lVert \bar { \pmb { v } } _ { g _ { i } } ^ { t - 1 } \rVert _ { 2 } , \mathbf { \bar { \theta } } _ { g _ { i } } ^ { t } )$ . Ψ adopts a threelayer MLPs with hidden size 32 and ReLU activations. A final sigmoid activation in Ψ ensures the predicted gain lies within [0, 1]. The final corrected deformation state is obtained by interpolating between the extrapolated prior and the current observation:

$$
\tilde { d } _ { g _ { i } } ^ { t } = \left( 1 - K _ { g _ { i } } ^ { t } \right) \odot \hat { d } _ { g _ { i } } ^ { t } + K _ { g _ { i } } ^ { t } \odot z _ { g _ { i } } ^ { t } ,\tag{11}
$$

where $\odot$ denotes element-wise multiplication. The corrected state $\tilde { d } _ { g _ { i } } ^ { t }$ is used for rendering and written back to the deformation buffer for subsequent state evolution. For the first three timesteps, we skip Eq. 11 and use the extrapolated prior directly.

## 4.3. Deformation-adaptive Control of Gaussians

To maintain representation fidelity and stabilize optimization in dynamic scenes, we introduce a deformation-adaptive densification module with two components: (i) an uncertainty-aware trigger that avoids densifying unstable Gaussians; (ii) a deformation-aware clone/split strategy that aligns newly created Gaussians with the underlying motion.

Uncertainty-aware Trigger. In 3DGS-style pipelines, densification is typically decided solely by the gradient magnitude $\rho _ { i } ,$ which may prematurely densify Gaussians under motion ambiguity. We augment the trigger with two uncertainty cues derived from Sec. 4.2: the prediction–observation residual $r _ { g _ { i } } ^ { t }$ and the angular deviation (defined in Eq. 10). Concretely, a Gaussian $\mathbf { \pmb { g } } _ { i }$ is densified only if

$$
\rho _ { i } < \tau _ { \rho } , \qquad r _ { i } ^ { t } < \tau _ { r } , \qquad \theta _ { i } ^ { t } < \tau _ { \theta } ,\tag{12}
$$

where $\tau _ { \rho }$ is the standard gradient threshold and $( \tau _ { r } , \tau _ { \Theta } )$ are fixed across scenes to avoid overfitting. Small $r _ { i } ^ { t }$ and $\mathsf { \boldsymbol { \theta } } _ { i } ^ { t }$ indicate temporally consistent motion and a reliable prior, reducing the chance of propagating uncertainty.

Deformation-aware Clone/Split. Standard densification operates at the source Gaussian’s current position: a clone duplicates a small Gaussian with parameters identical to the source, and a split replaces a large Gaussian with two smaller components (e.g., size scaled by 1/1.6) whose positions are sampled from a Gaussian defined by the source’s mean and covariance. To better align with motion, we first apply a partial deformation to the source,

![](images/0335dd6736a7ad70f7a49d05d0de4990c160a77a204c374b5159169879d100a5.jpg)  
4DGS

![](images/1bbe4c361531bace12ec99206d0f3a4f74b0a0bfadabfcc1b49314138b1d1805.jpg)  
EvoGS(4DGS)

![](images/1a2344162b2c3f68643f80b7a5bf8423f66b171240b52b3e070c98c536d716e3.jpg)  
DASH

![](images/1ed8ad2a33d9fa218a2163ae09a8be65b02d92071faf5a9c84f65b2db0f2d704.jpg)  
EvoGS(DASH)

![](images/38d9d6054dd70385480ec291dcebb7ecb634ca1796872b3b2713db182795910f.jpg)  
GT  
Figure 6: Visual comparisons on Neu3D dataset. The zoomed views reveal that the baselines produce over-smoothed structures on the metal strap (red box), the spinach leaves (green box), and the ring details on the clothes (blue box). Incorporating our temporal deformation evolution modeling substantially sharpens these structures, restores richer local contrast, and yields more faithful fine-scale geometry and appearance across both backbones.

$$
\pmb { g } _ { i }  \pmb { g } _ { i } + \alpha \tilde { d } _ { g _ { i } } ^ { t } , \qquad \alpha = 0 . 5 ,\tag{13}
$$

and then perform clone/split at the updated location as shown in Fig. 5. The resulting Gaussians inherit shape/opacity/color following standard rules.

Buffer Maintenance during Gaussian Control. Since EvoGS maintains persistent deformation states for each Gaussian, the state buffers must be updated consistently when the Gaussian set changes during densification and pruning. When new Gaussians are instantiated by clone or split, we initialize their deformation buffers by inheriting the corrected deformation history from their parent Gaussians. Specifically, for a newly created Gaussian $\mathbf { \pmb { g } } _ { n e w }$ from parent ${ \pmb g } _ { P } ,$ , we set

$$
B _ { g _ { n e w } }  B _ { g _ { p } } , \qquad m _ { g _ { n e w } }  m _ { g _ { p } } ,\tag{14}
$$

where B denotes the persistent deformation buffer and m denotes the innovation memory used for adaptive correction. This inheritance provides newly instantiated Gaussians with a valid local motion history, avoiding cold-start artifacts in subsequent state extrapolation. During pruning, the deformation buffers and innovation memories are filtered using the same validity mask as the Gaussian parameters, ensuring that the temporal states remain aligned with the active Gaussian set.

![](images/1523cb3eeea62a5f7a8220c91bcd0c226757266da10de3c8e7c59459b58e7819.jpg)  
MoDec

![](images/0e66495340e0d929138157ce07e3c6e5966cabe54d95601e6b49839a2f49ad44.jpg)  
EvoGS(MoDec)

Figure 7: MoDec produces local blurred artifacts, while EvoGS yields cleaner reconstruction with sharper structural details.

## 4.4. Optimization

To optimize Gaussians G, the observation network Φ, and the gain estimator Ψ, the render loss is adopted by comparing the rendered image at different time steps with ground truth reference images using a combination of $\mathcal { L } _ { 1 }$ loss and D-SSIM loss following baseline methods. Besides, specific in our method, our training objective integrates two auxiliary loss terms to enhance the deformation estimation process, targeting complementary aspects of estimation accuracy and predictive stability.

Deformation supervision. To ensure that the final corrected deformation offset aligns with the local motion evidence encoded in the observation, we apply direct supervision by minimizing their discrepancy for all N Gaussians:

$$
\mathcal { L } _ { \mathrm { s u p } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \big \lVert \tilde { d } _ { g _ { i } } ^ { t } - z _ { g _ { i } } ^ { t } \big \rVert _ { 2 } ^ { 2 } .\tag{15}
$$

This term encourages the corrected deformation state to remain consistent with the observation-driven displacement, providing a per-frame corrective constraint for the adaptive deformation correction.

It is important when the prediction or prior motion is unreliable, allowing the network to compensate through observations.

Temporal prior regularization. The extrapolated prior $\hat { d } _ { i } ^ { t }$ is computed from the persistent deformation buffer and contains no additional learnable parameters. We therefore use it as a temporal reference to weakly regularize the current MLP-derived observation:

$$
\mathcal { L } _ { \mathrm { p r i o r } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s g ( \sigma _ { i } ) \left| \left| s g ( \hat { d } _ { g _ { i } } ^ { t } ) - z _ { g _ { i } } ^ { t } \right| \right| _ { 2 } ^ { 2 } ,\tag{16}
$$

where $s g ( \cdot )$ denotes the stop-gradient operation. This term optimizes the observation network Φ by discouraging abrupt deviations from the historical motion prior in visible regions, while the adaptive correction module still determines how much the final state should rely on the prior or the current observation.

Table 1: Quantitative comparisons on Neu3D dataset. All methods are re-trained and metrics are evaluated at 1352 × 1014 resolution, complexity results are obtained using one H200 GPU. The best , second best and third best performances are colored. EvoGS consistently outperforms all baselines by substantial margins, demonstrating the effectiveness of our temporal deformation evolution modeling. Furthermore, with the proposed deformation-aware Gaussian control strategy, EvoGS achieves improved efficiency, yielding faster rendering speed and shorter training time.
<table><tr><td rowspan="2">Methods</td><td colspan="3">coffee_martini</td><td colspan="3">cook_spinach</td><td colspan="2">cut_roasted_beef</td><td colspan="3">flame_salmon_1</td></tr><tr><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td>|PSNR↑ SSIM↑ LPIPS↓|</td><td></td><td></td><td>PSNR↑ SSIM↑ LPIPS↓|</td><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>4DGS [WYF*24] EvoGS(4DGS)</td><td>28.41</td><td>0.908</td><td>0.162</td><td>32.22</td><td>0.938</td><td>0.170</td><td>29.68</td><td>0.932 0.164</td><td>29.27</td><td>0.913</td><td>0.155</td></tr><tr><td rowspan="7">Grid4D [XFYX24] EvoGS(Grid4D) Motion-GS [ZLC*24] EvoGS(Motion-GS) MoDec-GS [ZLC*24]</td><td>29.68</td><td>0.915</td><td>0.159</td><td>33.11</td><td>0.943</td><td>0.158</td><td>32.82 0.949</td><td>0.149</td><td>30.14</td><td>0.922</td><td>0.146</td></tr><tr><td>28.19</td><td>0.899</td><td>0.175</td><td>32.30</td><td>0.945</td><td>0.149</td><td>29.79 0.918</td><td>0.190</td><td>29.05</td><td>0.910</td><td>0.166</td></tr><tr><td>30.35</td><td>0.919</td><td>0.156</td><td>32.81</td><td>0.949</td><td>0.147</td><td>33.26 0.950</td><td>0.145</td><td>30.60</td><td>0.923</td><td>0.150</td></tr><tr><td>27.14</td><td>0.900</td><td>0.156</td><td>31.16</td><td>0.943</td><td>0.149</td><td>30.74 0.942</td><td>0.151</td><td>27.10</td><td>0.907</td><td>0.148</td></tr><tr><td>28.56</td><td>0.919</td><td>0.149</td><td>31.77</td><td>0.947</td><td>0.147</td><td>31.96 0.948</td><td>0.146</td><td>28.20</td><td>0.919</td><td>0.143</td></tr><tr><td></td><td>0.918</td><td>0.142</td><td>28.78</td><td>0.938</td><td>0.154</td><td>26.42 0.926</td><td>0.156</td><td>29.05</td><td>0.924</td><td>0.133</td></tr><tr><td>28.34 32.04</td><td>0.941</td><td>0.131</td><td>29.60</td><td>0.946</td><td>0.150</td><td>29.57 0.946</td><td>0.145</td><td>31.96</td><td>0.946</td><td>0.125</td></tr><tr><td>EvoGS(MoDec-GS) DASH [CHW*25]</td><td>26.95</td><td>0.895</td><td>0.173</td><td>27.65</td><td>0.913 0.193</td><td>27.32</td><td>0.918 0.194</td><td></td><td>27.06</td><td>0.900 0.171</td></tr><tr><td>EvoGS(DASH)</td><td>27.79</td><td>0.905</td><td>0.169</td><td>28.24</td><td>0.914</td><td>0.192</td><td>27.93 0.921</td><td>0.180</td><td>27.98</td><td>0.909</td><td>0.161</td></tr><tr><td>SpeeDe3DGS [TYH*26]</td><td>26.13</td><td>0.887</td><td>0.191</td><td>29.77</td><td>0.928</td><td>0.189</td><td>23.86 0.905</td><td>0.217</td><td>27.12</td><td>0.911</td><td>0.152</td></tr><tr><td>EvoGS(SpeeDe3DGS)</td><td>26.89</td><td>0.898</td><td>0.187</td><td>30.54</td><td>0.935</td><td>0.181</td><td>29.83 0.928</td><td>0.198</td><td>28.37</td><td>0.915</td><td>0.169</td></tr><tr><td>Methods</td><td colspan="2">flame_steak</td><td colspan="3"></td><td colspan="2"></td><td colspan="2">Complexity (coffee_martini)</td><td></td></tr><tr><td rowspan="2">4DGS [WYF*24]</td><td></td><td>|PSNR↑ SSIM↑ LPIPS↓|PSNR↑ SSIM↑ LPIPS↓|PSNR↑ SSIM↑ LPIPS↓|Num of Gaussians↓</td><td></td><td></td><td>sear_steak</td><td></td><td></td><td>Average</td><td></td><td></td><td></td></tr><tr><td colspan="2">29.99</td><td>0.158</td><td colspan="2">32.52</td><td colspan="2"></td><td colspan="2"></td><td></td><td>FPS↑ Training Time↓</td></tr><tr><td>EvoGS(4DGS)</td><td>33.55</td><td>0.941 0.953</td><td>0.142</td><td>33.84</td><td>0.949 0.955</td><td>0.153 0.140</td><td>30.35 0.930</td><td>0.160</td><td>125K</td><td>41</td><td>24 min</td></tr><tr><td>Grid4D [XFYX24]</td><td>32.32</td><td>0.952</td><td>0.139</td><td>32.86</td><td></td><td></td><td>32.19 0.940</td><td>0.149</td><td>122 K</td><td>56</td><td>13 min</td></tr><tr><td>EvoGS(Grid4D)</td><td>33.83</td><td>0.959</td><td>0.132</td><td>33.77</td><td>0.957</td><td>0.135</td><td>30.75 0.930</td><td>0.159 0.143</td><td>164 K 151 K</td><td>217</td><td>49 min</td></tr><tr><td>Motion-GS [ZLC*24]</td><td>31.37</td><td>0.952</td><td>0.134</td><td>28.29</td><td>0.961 0.941</td><td>0.130 0.142</td><td>32.44 0.944 29.30 0.931</td><td>0.147</td><td>728 K</td><td>244</td><td>37 min</td></tr><tr><td>EvoGS(Motion-GS)</td><td>32.10</td><td>0.956</td><td>0.132</td><td>29.54</td><td>0.947</td><td>0.139</td><td>30.36 0.939</td><td>0.143</td><td>637 K</td><td>26 58</td><td>363 min 287 min</td></tr><tr><td>MoDec-GS [KKJ*25]</td><td>24.21</td><td>0.854</td><td>0.313</td><td>23.61</td><td>0.907</td><td>0.166</td><td>26.74 0.911</td><td>0.177</td><td>98K</td><td>27</td><td>41 min</td></tr><tr><td>EvoGS(MoDec-GS)</td><td>28.84</td><td>0.953</td><td>0.144</td><td>30.55</td><td>0.958</td><td>0.138</td><td>30.43 0.948</td><td>0.139</td><td>91 K</td><td>45</td><td>36 min</td></tr><tr><td>DASH [CHW*25]</td><td>27.59</td><td>0.924</td><td>0.181</td><td>29.98</td><td>0.930</td><td>0.174</td><td>27.76 0.913</td><td>0.181</td><td>206 K</td><td>159</td><td>40 min</td></tr><tr><td>EvoGS(DASH)</td><td>28.15</td><td>0.930</td><td>0.174</td><td>30.54</td><td>0.936</td><td>0.167</td><td>28.44 0.919</td><td>0.174</td><td>197 K</td><td>175</td><td>35 min</td></tr><tr><td>SpeeDe3DGS [TYH*26]</td><td>25.51</td><td>0.919</td><td>0.194</td><td>27.14</td><td>0.922</td><td>0.193</td><td>26.59 0.912</td><td>0.189</td><td>40 K</td><td>224</td><td>44 min</td></tr><tr><td>EvoGS(SpeeDe3DGS)</td><td>29.96</td><td>0.940</td><td>0.174</td><td>29.04</td><td>0.933</td><td>0.175</td><td>29.11 0.925</td><td>0.181</td><td>37 K</td><td>246</td><td>41 min</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Therefore, the final optimization objective is a weighted sum of the above components:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e n d e r } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { s u p } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { p r i o r } } ,\tag{17}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are hyper-parameters, and we set $\lambda _ { 1 } = \lambda _ { 2 } = 0 .$ 5 in our implementations. Notably, $\mathcal { L } _ { \mathrm { s u p } }$ and $\mathcal { L } _ { \mathrm { p r e d } }$ are calculated within the deformation network without the extra forward rendering or backpropagation over Gaussians, achieving superior performance with minimal overhead.

## 5. Experiments

## 5.1. Experiment Setup

Datasets and Metrics. To assess the effectiveness of our method, we evaluate EvoGS on several datasets (NeRF-DS [YLL23], HyperNeRF (vrig subset) [PSH 21], and Neu3D [LSZ 22b]). NeRF-DS consists of seven real-world video sequences, with camera trajectories reconstructed using COLMAP [SF16]. Following the protocol in [WYF<sup>∗</sup>24], we estimate camera poses and initialize Gaussians for HyperNeRF and Neu3D datasets. Training is conducted on the official training set, with evaluation carried out on the corresponding test scenes. We adopt the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM [WBS<sup>∗</sup>04]) to assess pixel-level accuracy, and use Learned Perceptual Image Patch Similarity (LPIPS [ZIE 18], VGG) for perceptual quality evaluation.

Implementation Details. We conduct all the experiments on one H200 GPU device with Pytorch [Pas19] framework. The inputs to the gain estimator are directly concatenated, including the temporal residual memory, the two deformation-velocity magnitudes, and the angular deviation, and then fed into the three-layer MLP. The output is a scalar correction weight for each Gaussian, which is shared across its deformation components. For the first two timestamps, where insufficient historical states are available for secondorder extrapolation, we use the current observation to initialize the predicted state. At the third timestamp, second-order extrapolation is enabled with a fixed correction weight K=0.5. The learned adaptive correction is activated from the fourth timestamp, when sufficient historical states are available to compute the velocity and angular-deviation statistics. No additional warm-up stage is used. In the process of Gaussian densification, newly created Gaussians directly inherit the persistent deformation buffer and residual memory of their corresponding parent Gaussians for cloning; for splitting, the parent states are replicated for all generated child Gaussians, while their spatial parameters are initialized according to the standard split operation. We build our EvoGS by integrating our modular temporal deformation evolution process into several existing SOTA baselines (D-3DGS [YGZ<sup>∗</sup>24], 4DGS [WYF<sup>∗</sup>24], SCGS [HSY<sup>∗</sup>24], MAGS [GZL<sup>∗</sup>24], Grid4D [XFYX24], MoDec-GS [KKJ<sup>∗</sup>25], DASH [CHW<sup>∗</sup>25], SpeeDe3DGS [TYH<sup>∗</sup>26]). To ensure a fair comparison, the same deformation-aware densification strategy is applied to all integrated baselines and our training configurations (e.g., total training iterations and learning rate schedules) strictly follows the baselines. Furthermore, we replace the original deformation MLP/module in baselines with our observation network and temporal evolution/correction module and employ reduced channel dimensions for Φ and Ψ (half of that of the original deformation module), thereby constraining their model capacity to be lower than that of the original deformation MLPs used in these baselines.

Table 2: Quantitative comparison on NeRF-DS dataset. Our temporal deformation evolution modeling replaces the original deformation modules in various deformable Gaussian frameworks (e.g., 4DGS, SCGS, MAGS, D-3DGS), forming their EvoGS variants. These EvoGS models consistently achieve higher PSNR/SSIM and lower LPIPS across diverse dynamic scenes, demonstrating the robustness and broad applicability of our temporal deformation evolution modeling.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Sieve</td><td colspan="3">Plate</td><td colspan="3">Bell</td><td colspan="3">Press</td></tr><tr><td>PSNR(↑) SSIM(↑) LPIPS(↓) PSNR(↑) SSIM(↑) LPIPS(↓) PSNR(↑) SSIM(↑) LPIPS(↓) PSNR(↑) SSIM(↑) LPIPS(↓)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3D-GS [KKLD23]</td><td>23.16</td><td>0.8203</td><td>0.2247</td><td>16.14</td><td>0.6970</td><td>0.4093</td><td>21.01</td><td>0.7885</td><td>0.2503</td><td>22.89</td><td>0.8163</td><td>0.2904</td></tr><tr><td>TiNeuVox [FYW*22b]</td><td>21.49</td><td>0.8265</td><td>0.3176</td><td>20.58</td><td>0.8027</td><td>0.3317</td><td>23.08</td><td>0.8242</td><td>0.2568</td><td>24.47</td><td>0.8613</td><td>0.3001</td></tr><tr><td>HyperNeRF [PSH*21]</td><td>25.43</td><td>0.8798</td><td>0.1645</td><td>18.93</td><td>0.7709</td><td>0.2940</td><td>23.06</td><td>0.8097</td><td>0.2052</td><td>26.15</td><td>0.8897</td><td>0.1959</td></tr><tr><td>NeRF-DS [YLL23]</td><td>25.78</td><td>0.8900</td><td>0.1472</td><td>20.54</td><td>0.8042</td><td>0.1996</td><td>23.19</td><td>0.8212</td><td>0.1867</td><td>25.72</td><td>0.8618</td><td>0.2047</td></tr><tr><td>4DGS [WYF*24]</td><td>24.33</td><td>0.8625</td><td>0.1583</td><td>18.07</td><td>0.7215</td><td>0.3218</td><td>24.44</td><td>0.8237</td><td>0.1725</td><td>24.57</td><td>0.8611</td><td>0.2039</td></tr><tr><td>EvoGS(4DGS)</td><td>25.74</td><td>0.8827</td><td>0.1484</td><td>20.33</td><td>0.7968</td><td>0.2009</td><td>25.96</td><td>0.8525</td><td>0.1530</td><td>26.03</td><td>0.8699</td><td>0.1836</td></tr><tr><td>SCGS [HSY*24] EvoGS(SCGS)</td><td>24.93 26.25</td><td>0.8622 0.8875</td><td>0.1563</td><td>19.08 20.55</td><td>0.7488</td><td>0.3620 0.1901</td><td>25.11 26.44</td><td>0.8425</td><td>0.1628</td><td>25.40</td><td>0.8609</td><td>0.1943</td></tr><tr><td>MAGS [GZL*24]</td><td></td><td></td><td>0.1474</td><td></td><td>0.8127</td><td></td><td></td><td>0.8674</td><td>0.1451</td><td>26.33</td><td>0.8681</td><td>0.1827</td></tr><tr><td>EvoGS(MAGS)</td><td>24.82 26.23</td><td>0.8673 0.8866</td><td>0.1533 0.1474</td><td>18.83 20.99</td><td>0.7413 0.8211</td><td>0.3486 0.1872</td><td>24.92 26.35</td><td>0.8417 0.8652</td><td>0.1641</td><td>25.02 26.28</td><td>0.8644</td><td>0.1985</td></tr><tr><td>D-3DGS [YGZ*24]</td><td>25.24</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.1481</td><td></td><td>0.8712</td><td>0.1816</td></tr><tr><td>EvoGS(D-3DGS)</td><td>26.46</td><td>0.8686 0.8916</td><td>0.1505 0.1454</td><td>19.23 21.28</td><td>0.7523 0.8241</td><td>0.3576 0.1831</td><td>25.38 26.82</td><td>0.8473 0.8705</td><td>0.1593 0.1419</td><td>25.40 26.81</td><td>0.8609 0.8745</td><td>0.1943 0.1789</td></tr><tr><td>Methods</td><td colspan="4">Cup</td><td></td><td></td><td></td><td>Basin</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="4">PSNR(↑) SSIM(↑) LPIPS(↓) PSNR(↑) SSIM(↑) LPIPS(↓) PSNR(↑) SSIM(↑) LPIPS(↓) PSNR(↑) SSIM(↑) LPIPS(↓)</td><td>As</td><td></td><td></td><td></td><td></td><td></td><td>Average</td><td></td></tr><tr><td>3D-GS [KKLD23]</td><td>21.71</td><td>0.8304</td><td>0.2548</td><td>22.69</td><td>0.8017</td><td>0.2994</td><td>18.42</td><td>0.7170</td><td>0.3153</td><td>20.86</td><td>0.7816</td><td>0.2920</td></tr><tr><td>TiNeuVox [FYW*22b]</td><td>19.71</td><td>0.8109</td><td>0.3643</td><td>21.26</td><td>0.8289</td><td>0.3967</td><td>20.66</td><td>0.8145</td><td>0.2690</td><td>21.61</td><td>0.8241</td><td>0.3195</td></tr><tr><td>HyperNeRF [PSH*21] NeRF-DS [YLL23]</td><td>24.59</td><td>0.8770</td><td>0.1650</td><td>25.58</td><td>0.8949</td><td>0.1777</td><td>20.41</td><td>0.8199</td><td>0.1911</td><td>23.45</td><td>0.8488</td><td>0.1991</td></tr><tr><td>4DGS [WYF*24]</td><td>24.91</td><td>0.8741</td><td>0.1737</td><td>25.13</td><td>0.8778</td><td>0.1741</td><td>19.96</td><td>0.8166</td><td>0.1855</td><td>23.60</td><td>0.8494</td><td>0.1816</td></tr><tr><td>EvoGS(4DGS)</td><td>23.85 24.83</td><td>0.8802 0.8887</td><td>0.1645 0.1542</td><td>25.26 26.33</td><td>0.8732</td><td>0.1942 0.1799</td><td>18.66 19.99</td><td>0.7885</td><td>0.2048</td><td>22.74</td><td>0.8301</td><td>0.2029</td></tr><tr><td>SCGS [HSY*24]</td><td>24.44</td><td></td><td></td><td></td><td>0.8868</td><td></td><td></td><td>0.7994</td><td>0.1876</td><td>24.17</td><td>0.8538</td><td>0.1725</td></tr><tr><td>EvoGS(SCGS)</td><td>25.56</td><td>0.8875</td><td>0.1579</td><td>25.75</td><td>0.8796</td><td>0.1877</td><td>19.24</td><td>0.7902</td><td>0.1911</td><td>23.42</td><td>0.8388</td><td>0.2017</td></tr><tr><td></td><td></td><td>0.8934</td><td>0.1501</td><td>26.80</td><td>0.8911</td><td>0.1735</td><td>20.65</td><td>0.8218</td><td>0.1819</td><td>24.65</td><td>0.8631</td><td>0.1673</td></tr><tr><td>MAGS [GZL*24]</td><td>24.35</td><td>0.8824</td><td>0.1608</td><td>25.58</td><td>0.8756</td><td>0.1906</td><td>19.17</td><td>0.7868</td><td>0.1938</td><td>23.24</td><td>0.8371</td><td>0.2014</td></tr><tr><td>EvoGS(MAGS)</td><td>25.28</td><td>0.8923</td><td>0.1524</td><td>26.53</td><td>0.8886</td><td>0.1759</td><td>20.61</td><td>0.8212</td><td>0.1821</td><td>24.61</td><td>0.8637</td><td>0.1678</td></tr><tr><td>D-3DGS [YGZ*24]</td><td>24.82</td><td>0.8909</td><td>0.1542</td><td>26.01</td><td>0.8822</td><td>0.1837</td><td>19.69</td><td>0.7935</td><td>0.1886</td><td>23.68</td><td>0.8422</td><td>0.1983</td></tr><tr><td>EvoGS(D-3DGS)</td><td></td><td></td><td></td><td></td><td></td><td>0.1704</td><td>20.98</td><td>0.8251</td><td>0.1755</td><td></td><td></td><td></td></tr><tr><td></td><td>25.89</td><td>0.8951</td><td>0.1472</td><td>27.06</td><td>0.8936</td><td></td><td></td><td></td><td></td><td>25.04</td><td>0.8678</td><td>0.1632</td></tr></table>

## 5.2. Comparisons and Results

Quantitative Comparisons. As observed in Tab 1, Tab. 2, and Tab. 3, our EvoGS consistently surpasses the baselines across all Neu3D, NeRF-DS, and HyperNeRF scenes, yielding markedly higher PSNR/SSIM and lower LPIPS. These gains substantiate the effectiveness of our temporal deformation evolution modeling and deformation-aware densification, and further attest to both the method’s real-world generalization and its plug-and-play applicability. Notably, these improvements are attained without increasing computational cost: all networks (Φ and Ψ) remain lightweight and our deformation-aware triggering decreases the number of

Gaussians, delivering shorter training time and higher FPS. We also measure the GPU memory used by the deformation-state buffer and the peak GPU memory during training and evaluation. For EvoGS(Grid4D) on the coffee\_martini scene of Neu3D, the deformation-state buffer only occupies 3.6 GB of GPU memory, while the peak GPU memory during training/evaluation is 16/8 GB. Furthermore, compared to FreeTimeGS [WYX<sup>∗</sup>25] which achieves 32.25 dB PSNR and 0.946 SSIM over Neu3D, our EvoGS variants based on Grid4D and 4DGS achieve 32.44/32.19 dB PSNR and 0.944/0.940 SSIM, respectively. These results show that EvoGS remains competitive with recent fully explicit approaches, while our method targets a complementary setting by improving temporal deformation modeling in deformation-based dynamic Gaussian frameworks and can be integrated into multiple such backbones.

Visual Comparisons. Fig. 6, Fig. 8, Fig. 9, Fig. 7 highlight the perceptual quality of our method under diverse real-world motions. Across different real-world scenes, EvoGS exhibits fewer ghosting artifacts and sharper structural recovery. These visual improvements are attributed to our temporal deformation evolution modeling and gated densification, which together help stabilize deformation estimates and avoid amplifying noise. Furthermore, we fix the viewpoint and visualize novel views with varying time steps in Fig. 2 and 10, which clearly presents the temporal consistency cross different time steps. The baseline (4DGS) suffers from poor temporal consistency, as evidenced by serious structure distortions at multiple time steps. This suggests that, without explicit crosstemporal constraints, directly employing an MLP to predict deformation tends to degenerate into time- and view-specific mappings rather than a temporally coherent field. In comparison, our developed temporal deformation evolution modeling helps achieve a more consistent geometry reconstruction across various time steps.

![](images/20737b0e72f656a64b9b4876194d187bca51ed8887eb6a8b747ec0bdf8223f1b.jpg)  
Figure 8: Qualitative comparison on the NeRF-DS dataset. The baselines (4DGS and MAGS) produce severely distorted object contours, where the plate boundary and arm silhouette are noticeably warped. The EvoGS-augmented variants (EvoGS(4DGS) and EvoGS(MAGS)) preserve much more stable geometry and sharper outlines, demonstrating that the proposed temporal deformation evolution modeling con sistently stabilizes geometry and appearance across diverse dynamic Gaussianframeworks.

Table 3: Quantitative comparisons on the vrig subset of HyperNeRF, reported per scene. Metrics are evaluated at 536 × 960 resolution. Our Kalman-inspired deformation modeling can be seamlessly plugged into existing deformable Gaussian frameworks (e.g., 4DGS, D-3DGS, Grid4D, and MoDec-GS). The resulting variants (EvoGS(4DGS), EvoGS(D-3DGS), EvoGS(Grid4D), EvoGS(MoDec-GS)) consistently surpass their baselines in both PSNR and SSIM, highlighting the adaptability and generalization capability of our deformation design.
<table><tr><td rowspan="2">Method</td><td colspan="2">3D Printer</td><td colspan="2">Chicken</td><td colspan="2">Broom</td><td colspan="2">Banana</td><td colspan="2">Average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>HyperNeRF [PSH*21]</td><td>20.04</td><td>0.6308</td><td>27.43</td><td>0.6312</td><td>19.52</td><td>0.2114</td><td>22.13</td><td>0.7209</td><td>22.28</td><td>0.5486</td></tr><tr><td>TiNeuVox [FYW*22b]</td><td>22.84</td><td>0.7316</td><td>28.24</td><td>0.7923</td><td>21.37</td><td>0.3138</td><td>24.46</td><td>0.6419</td><td>24.23</td><td>0.6199</td></tr><tr><td>MotionGS [ZLC*24]</td><td>20.63</td><td>0.6611</td><td>22.87</td><td>0.6817</td><td>21.04</td><td>0.3026</td><td>26.57</td><td>0.8408</td><td>22.78</td><td>0.6216</td></tr><tr><td>D-3DGS [YGZ*24]</td><td>20.56</td><td>0.6413</td><td>22.84</td><td>0.6121</td><td>20.58</td><td>0.3517</td><td>26.04</td><td>0.8314</td><td>22.50</td><td>0.6091</td></tr><tr><td>EvoGS(D-3DGS)</td><td>21.63(+1.07) 0.6903(+0.0490)</td><td></td><td></td><td></td><td></td><td>24.39(+1.55) 0.7305(+0.1184)21.89(+1.31) 0.3904(+0.0387)2</td><td></td><td>27.46(+1.42)0.8702(+0.0388)</td><td></td><td>23.84(+1.34)0.6704(+0.0613)</td></tr><tr><td>4DGS [WYF*24]</td><td>22.05</td><td>0.7118</td><td>28.76</td><td>0.8115</td><td>22.14</td><td>0.3713</td><td>28.07</td><td>0.8617</td><td>25.26</td><td>0.6891</td></tr><tr><td>EvoGS(4DGS)</td><td>23.06(+1.01) 0.7504(+0.0386)</td><td></td><td></td><td>29.64(+0.88)0.8504(+0.0389)</td><td></td><td>22.89(+0.75)0.4003(+0.0290)</td><td></td><td>28.92(+0.85)0.9002(+0.0385)</td><td>26.13(+0.87)0.7253(+0.0362)</td><td></td></tr><tr><td>Grid4D [XFYX24]</td><td>22.43</td><td>0.7235</td><td>29.14</td><td>0.8431</td><td>22.33</td><td>0.3739</td><td>27.92</td><td>0.8625</td><td>25.45</td><td>0.7007</td></tr><tr><td>EvoGS(Grid4D)</td><td>23.85(+1.42)0.7937(+0.0702)30.27(+1.13)0.8826(+0.0395)23.29(+0.96)0.4007(+0.0268)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>28.85(+0.93)0.9074(+0.0449)26.57(+1.11) 0.7461(+0.0454)</td><td></td></tr><tr><td>MoDec-GS [KKJ*25]</td><td>22.03</td><td>0.7023</td><td>28.24</td><td>0.8116</td><td>21.62</td><td>0.4033</td><td>26.24</td><td>0.8409</td><td>24.53</td><td>0.6895</td></tr><tr><td>EvoGS(MoDec-GS)</td><td>23.36(+1.33) 0.7628(+0.0605)</td><td></td><td></td><td>29.66(+1.42)0.8653(+0.0537)</td><td></td><td>22.79(+1.17)0.4284(+0.0251)</td><td></td><td>28.35(+2.11)0.9016(+0.0607)</td><td></td><td>26.04(+1.51)0.7395(+0.0500)</td></tr></table>

Fig. 3 also indicates that EvoGS learns a more effective dynamic Gaussian representation, where more Gaussians remain trackable in motion-critical areas. Although some residual floating artifacts can still be observed around the plate in Fig. 8 (EvoGS(4DGS), EvoGS(MAGS)), this is mainly due to the highly degraded geometry produced by the underlying baselines (4DGS, MAGS), where the object structure is not reliably reconstructed. Our temporal deformation evolution modeling substantially improves the reconstruction by recovering more coherent object geometry and more stable appearance over time, rather than explicitly performing outlier removal or floater suppression. Therefore, a small number of residual floaters may remain in challenging regions, but the overall geometry, boundary consistency, and temporal coherence are clearly improved compared with the original baselines.

EvoGS(MoDecGS)  
EvoGS(Grid4D)  
MoDecGS  
Grid4D  
![](images/204f4f1409a9b0004a223c701c6adb7e79e4747b1818c3f4e92a1d5e802836a5.jpg)  
Figure 9: Qualitative comparisons on a complex scene with thin structures and sharp planar geometry (3D printer scene in HyperNeRF dataset). The zoomed crops highlight that our EvoGS variants built upon various baselines enhance boundary localization and structural fidelity: vertical edges on the cutting mat become straighter and less wavy, the black cable and metallicframe exhibitfewer double edges and ringing artifacts, and small high-contrast details are better preserved, demonstrating consistent improvements across compared baselines.

## 5.3. Ablation Study

To evaluate the effectiveness of our proposed EvoGS framework, we conduct a series of ablation experiments on the NeRF-DS dataset, considering several configurations: 1) replacing the temporal deformation evolution process with simple observation MLP (Baseline); 2) only utilizing the observation network with tri-plane spatial and B-spline temporal encoders (discussed in Sec. 4.2); 3) replacing the persistent deformation state buffer by querying the observation MLP with different timesteps; 4) utilizing the temporal deformation evolution process (w/o deformation-aware densification); 5) adding the uncertainty-aware densification triggering mechanism in Eq. 12 to 4); 6) adding our designed deformationaware clone and split strategy in Eq. 13 to 5). The average quantitative results are reported in Tab. 4.

When temporal deformation evolution modeling is fully removed, the deformations at different time steps are determined by the observation MLP. Due to the lack of motion modeling across time, compared to the average performance of our method presented in Tab. 2, this setting leads to a marked decrease in PSNR and LPIPS. When the tri-plane spatial and B-spline temporal encoders are equipped with the observation MLP, we observe enhanced performance (+0.3 PNSR) over the baseline. The comparison between 3) and 4) demonstrates the positive contributions of the deformation state buffer. Furthermore, compared with the original densification triggering or clone & split strategy, the improved performances in configurations 5) and 6) indicate that our proposed deformation-aware densification not only stabilizes the optimization but also enhances the representational fidelity of dynamic scenes. We additionally compare prediction-only (K=0), fixed blending (K=0.5), observation-only (K=1), and the learned adaptive correction, with all other settings kept unchanged. The learned adaptive correction achieves 25.04 dB PSNR / 0.868 SSIM, followed by fixed blending (24.64 dB / 0.863), while predictiononly performs worst (23.52 dB / 0.842). These results show that neither relying solely on historical prediction nor using a fixed fusion rule is optimal, and that dynamically balancing the historical prior and current observation provides the best performance.

Table 4: Ablation results on NeRF-DS dataset (Baseline: D-3DGS). Removing any key components from our full EvoGS leads to reduced performance, highlighting the positive contribution of each module in our design.
<table><tr><td>Configurations</td><td>|PSNR ↑ SSIM ↑ LPIPS ↓</td><td></td></tr><tr><td>1) Baseline (D-3DGS)</td><td>23.68</td><td>0.8422 0.1983</td></tr><tr><td>2) Tri-plane and B-spline</td><td>23.99</td><td>0.8520 0.1747</td></tr><tr><td>3) temporal evolution without buffer</td><td>24.14</td><td>0.8535 0.1728</td></tr><tr><td>3) temporal evolution</td><td>24.55</td><td>0.8632 0.1697</td></tr><tr><td>4) temporal evolution+Eq. 12</td><td>24.76</td><td>0.8659 0.1662</td></tr><tr><td>5) temporal evolution+Eq. 12, Eq. 13</td><td>25.04</td><td>0.8678 0.1632</td></tr></table>

![](images/143d5760f78c5d83b4a450dd0bbccf273573937166c0bc82b9432f89d38943fb.jpg)  
Figure 10: Visualizations of novel views rendered with the same camera pose across consecutive timesteps on the HyperNeRF (vrig) dataset. Specifically, the baseline D-3DGS fails to represent the moving broom. In contrast, EvoGS achieves good reconstructions ofthe moving broom and background structures acrossframes.

## 6. Conclusion and Limitation

Conclusion. In this paper, we present EvoGS, a 3DGS-based dynamic reconstruction framework that models Gaussian deformation as a temporal evolution process. Instead of independently estimating deformations at each timestamp, EvoGS maintains persistent deformation states for each Gaussian and extrapolates the current deformation from historical corrected states. The extrapolated prior is then adaptively refined with an MLP-derived observation using temporal residual memory and motion evolution statistics. Furthermore, we introduce a deformation-aware densification mechanism that performs clone and split operations along corrected deformation directions while suppressing unstable Gaussians with uncertain motion histories. Extensive experiments demonstrate that EvoGS improves dynamic novel view synthesis quality and achieves competitive performance across multiple benchmarks.

Limitation. Our design targets short-horizon updates under approximately uniform frame spacing. Our formulation focuses on short-horizon deformation evolution and may become less reliable under very long occlusions, where errors in the historical states can accumulate before reliable observations become available again. Severe topology changes also remain constrained by the representational capability of the underlying deformation-based backbone.

## References

[AHR<sup>∗</sup>23] ATTAL B., HUANG J.-B., RICHARDT C., ZOLLHOEFER M., KOPF J., O’TOOLE M., KIM C.: Hyperreel: High-fidelity 6-dof video with ray-conditioned sampling. In CVPR (2023). 2

[BKY<sup>∗</sup>24] BAE J., KIM S., YUN Y., LEE H., BANG G., UH Y.: Per-gaussian embedding-based deformation for deformable 3d gaussian splatting. In ECCV (2024). 2

[BRSE21] BECKER M., REVACH D., SHLEZINGER N., ELDAR Y. C.: Kalmannet: Neural network architecture for kalman filtering from data. IEEE TSP (2021). 2

[CHW<sup>∗</sup>25] CHEN J., HU Z., WU P., ZHU H., LI H., SUN X.: Dash: 4d hash encoding with self-supervised decomposition for real-time dynamic scene rendering. In ICCV (2025). 2, 7

[CJ23a] CAO A., JOHNSON J.: Hexplane: A fast representation for dynamic scenes. In CVPR (2023). 2

[CJ23b] CAO A., JOHNSON J.: Hexplane: A fast representation for dynamic scenes. In CVPR (2023). 2

[DZLC26] DONG W., ZHOU H., LIN J., CHEN J.: Zero-reference joint low-light enhancement and deblurring via visual autoregressive modeling with vlm-derived modulation. In AAAI (2026). 2

[DZY<sup>∗</sup>21] DU Y., ZHANG Y., YU H.-X., TENENBAUM J. B., WU J.: Neural radiance flow for 4d view synthesis and video processing. In ICCV (2021). 1, 2

[DZZ<sup>∗</sup>24] DONG W., ZHOU H., ZHANG Y., LIU X., CHEN J.: Ecmamba: Consolidating selective state space model with retinex guidance for efficient multiple exposure correction. In NeurIPS (2024). 2

[FCL<sup>∗</sup>24] FAN C.-D., CHANG C.-W., LIU Y.-R., LEE J.-Y., HUANG J.-L., TSENG Y.-C., LIU Y.-L.: Spectromotion: Dynamic 3d reconstruction of specular scenes. arXiv preprint arXiv:2410.17249 (2024). 2

[FKMW<sup>∗</sup>23] FRIDOVICH-KEIL S., MEANTI G., WARBURG F. R., RECHT B., KANAZAWA A.: K-planes: Explicit radiance fields in space, time, and appearance. In CVPR (2023). 2

[FKPW17] FRACCARO M., KAMRONN S., PAQUET U., WINTHER O.: A disentangled recognition and nonlinear dynamics model for unsupervised learning. In NeurIPS (2017). 2

[FLL24] FENG R., LI C., LOY C. C.: Kalman-inspired feature propagation for video face super-resolution. In ECCV (2024). 2

[FYW<sup>∗</sup>22a] FANG J., YI T., WANG X., XIE L., ZHANG X., LIU W., NIESSNER M., TIAN Q.: Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia (2022). 1, 2

[FYW<sup>∗</sup>22b] FANG J., YI T., WANG X., XIE L., ZHANG X., LIU W., NIESSNER M., TIAN Q.: Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia (2022). 8, 9

[GSKH21] GAO C., SARAF A., KOPF J., HUANG J.-B.: Dynamic view synthesis from dynamic monocular video. In ICCV (2021). 1, 2

[GZL<sup>∗</sup>24] GUO Z., ZHOU W., LI L., WANG M., LI H.: Motion-aware 3d gaussian splatting for efficient dynamic scene reconstruction. IEEE TCSVT (2024). 7, 8

[HOY23] HORVATH B., OBERHAUSER B., YANG H.: Deep kalman filters can filter. arXiv preprint arXiv:2310.19603 (2023). 2

[HPZ<sup>∗</sup>16] HAARNOJA T., PONG V., ZHOU X. C., ABBEEL P., LEVINE S.: Backprop kf: Learning discriminative deterministic state estimators. In NeurIPS (2016). 2

[HSY<sup>∗</sup>24] HUANG Y.-H., SUN Y.-T., YANG Z., LYU X., CAO Y.-P., QI X.: Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In CVPR (2024). 2, 7, 8

[K<sup>∗</sup>25] KUMAR A., ET AL.: Dynamode-nerf: Motion-aware deblurring neural radiance field for dynamic scenes. In CVPR (2025), pp. 21728– 21738. 2

[KKJ<sup>∗</sup>25] KWAK S., KIM J., JEONG J. Y., CHEONG W.-S., OH J., KIM M.: Modec-gs: Global-to-local motion decomposition and temporal interval adjustment for compact dynamic 3d gaussian splatting. In CVPR (2025). 2, 7, 9

[KKLD23] KERBL B., KOPANAS G., LEIMKÜHLER T., DRETTAKIS G.: 3d gaussian splatting for real-time radiance field rendering. ACM TOG (2023). 2, 3, 4, 8

[KSS15] KRISHNAN R. G., SHALIT U., SONTAG D.: Deep kalman filters. In Proceedings of the 32nd International Conference on Machine Learning (ICML) (2015). 2

[KVN24] KATSUMATA K., VO D. M., NAKAYAMA H.: A compact dynamic 3d gaussian representation for real-time dynamic view synthesis. In ECCV (2024). 2

[KYW25] KONG H., YANG X., WANG X.: Efficient gaussian splatting for monocular dynamic scene rendering via sparse time-variant attribute modeling. In Proceedings of the AAAI Conference on Artificial Intelligence (2025). 2

[LGH<sup>∗</sup>24] LU Z., GUO X., HUI L., CHEN T., YANG M., TANG X., ZHU F., DAI Y.: 3d geometry-aware deformable gaussian splatting for dynamic view synthesis. In CVPR (2024). 2

[LPL<sup>∗</sup>25] LI W., PAN X., LIN J., LU P., FENG D., SHI W.: Frpgs: Fast, robust, and photorealistic monocular dynamic scene reconstruction with deformable 3d gaussians. IEEE TCSVT (2025). 2

[LSW<sup>∗</sup>22] LI L., SHEN Z., WANG Z., SHEN L., TAN P.: Streaming radiance fields for 3d video synthesis. NeurIPS (2022). 2

[LSZ<sup>∗</sup>22a] LI T., SLAVCHEVA M., ZOLLHOEFER M., GREEN S., LASSNER C., KIM C., SCHMIDT T., LOVEGROVE S., GOESELE M., NEWCOMBE R., ET AL.: Neural 3d video synthesis from multi-view video. In CVPR (2022). 2

[LSZ<sup>∗</sup>22b] LI T., SLAVCHEVA M., ZOLLHOEFER M., GREEN S., LASSNER C., KIM C., SCHMIDT T., LOVEGROVE S., GOESELE M., NEWCOMBE R., ET AL.: Neural 3d video synthesis from multi-view video. In CVPR (2022). 7

[LTH<sup>∗</sup>25] LI X., TONG J., HONG J., ROLLAND V., PETERSSON L.: Dgns: Deformable gaussian splatting and dynamic neural surface for monocular dynamic 3d reconstruction. In ACMMM (2025). 2

[LZL<sup>∗</sup>25] LU Y., ZHOU Y., LIU D., LIANG T., YIN Y.: Bard-gs: Bluraware reconstruction of dynamic scenes via gaussian splatting. arXiv preprint arXiv:2503.15835 (2025). 2

[MST<sup>∗</sup>20] MILDENHALL B., SRINIVASAN P. P., TANCIK M., BARRON J. T., RAMAMOORTHI R., NG R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV (2020). 1, 2

[Pas19] PASZKE A.: Pytorch: An imperative style, high-performance deep learning library. arXiv preprint arXiv:1912.01703 (2019). 7

[PCPMMN21] PUMAROLA A., CORONA E., PONS-MOLL G., MORENO-NOGUER F.: D-nerf: Neural radiance fields for dynamic scenes. In CVPR (2021). 1, 2

[PSB<sup>∗</sup>21] PARK K., SINHA U., BARRON J. T., BOUAZIZ S., GOLD-MAN D. B., SEITZ S. M., MARTIN-BRUALLA R.: Nerfies: Deformable neural radiance fields. In ICCV (2021). 1, 2

[PSH<sup>∗</sup>21] PARK K., SINHA U., HEDMAN P., BARRON J. T., BOUAZIZ S., GOLDMAN D. B., MARTIN-BRUALLA R., SEITZ S. M.: Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. ACM TOG (2021). 1, 2, 7, 8, 9

[QLW<sup>∗</sup>25] QINGMING L., LIU Y., WANG J., LYU X., WANG P., WANG W., HOU J.: Modgs: Dynamic gaussian splatting from casually-captured monocular videos with depth priors. In ICLR (2025). 2

[SCL<sup>∗</sup>23] SONG L., CHEN A., LI Z., CHEN Z., CHEN L., YUAN J., XU Y., GEIGER A.: Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE TVCG (2023). 2

[SF16] SCHÖNBERGER J. L., FRAHM J.-M.: Structure-from-motion revisited. In CVPR (2016). 7

[SJL<sup>∗</sup>24] SUN J., JIAO H., LI G., ZHANG Z., ZHAO L., XING W.: 3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos. In CVPR (2024). 2

[SLX<sup>∗</sup>25] SONG R., LIANG C., XIA Y., ZIMMER W., CAO H., CAE-SAR H., FESTAG A., KNOLL A.: Coda-4dgs: Dynamic gaussian splatting with context and deformation awareness for autonomous driving. In ICCV (2025). 2

[TYH<sup>∗</sup>26] TU A., YING H., HANSON A., LEE Y., GOLDSTEIN T., ZWICKER M.: Speede3dgs: Speedy deformable 3d gaussian splatting with temporal pruning and motion grouping. In CVPR (2026). 7

[WBS<sup>∗</sup>04] WANG Z., BOVIK A. C., SHEIKH H. R., , SIMONCELLI E. P.: Image quality assessment: From error visibility to structural similarity. In IEEE TIP (2004). 7

[WTL<sup>∗</sup>23] WANG F., TAN S., LI X., TIAN Z., SONG Y., LIU H.: Mixed neural voxels for fast multi-view video synthesis. In ICCV (2023). 2

[WYF<sup>∗</sup>24] WU G., YI T., FANG J., XIE L., ZHANG X., WEI W., LIU W., TIAN Q., WANG X.: 4d gaussian splatting for real-time dynamic scene rendering. In CVPR (2024). 2, 3, 7, 8, 9

[WYX<sup>∗</sup>25] WANG Y., YANG P., XU Z., SUN J., ZHANG Z., CHEN Y., BAO H., PENG S., ZHOU X.: Freetimegs: Free gaussian primitives at anytime anywhere for dynamic scene reconstruction. In CVPR (2025). 8

[XFYX24] XU J., FAN Z., YANG J., XIE J.: Grid4d: 4d decomposed hash encoding for high-fidelity dynamic gaussian splatting. In NeurIPS (2024). 2, 7, 9

[XWZ<sup>∗</sup>25] XU W., WENG W., ZHANG Y., XU R., XIONG Z.: Eventboosted deformable 3d gaussians for dynamic scene reconstruction. In ICCV (2025). 2

[YGZ<sup>∗</sup>24] YANG Z., GAO X., ZHOU W., JIAO S., ZHANG Y., JIN X.: Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In CVPR (2024). 2, 7, 8, 9

[YLL23] YAN Z., LI C., LEE G. H.: Nerf-ds: Neural radiance fields for dynamic specular objects. In CVPR (2023). 7, 8

[YPW<sup>∗</sup>25] YAN J., PENG R., WANG Z., TANG L., YANG J., LIANG J., WU J., WANG R.: Instant gaussian stream: Fast and generalizable streaming of dynamic scene reconstruction via gaussian splatting. In CVPR (2025). 2

[YYPZ23] YANG Z., YANG H., PAN Z., ZHANG L.: Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642 (2023). 2

[ZCLL22] ZHOU S., CHAN K. C., LI C., LOY C. C.: Towards robust blind face restoration with codebook lookup transformer. In NeurIPS (2022). 2

[ZDC25] ZHOU H., DONG W., CHEN J.: Lita-gs: Illumination-agnostic novel view synthesis via reference-free 3d gaussian splatting and physical priors. In CVPR (2025). 2

[ZDL<sup>∗</sup>24] ZHOU H., DONG W., LIU X., LIU S., MIN X., ZHAI G., CHEN J.: Glare: Low light image enhancement via generative latent feature based codebook retrieval. In ECCV (2024). 2

[ZDL<sup>∗</sup>25] ZHOU H., DONG W., LIU X., ZHANG Y., ZHAI G., CHEN J.: Low-light image enhancement via generative perceptual priors. In AAAI (2025). 2

[ZIE<sup>∗</sup>18] ZHANG R., ISOLA P., EFROS A. A., SHECHTMAN E., WANG O.: The unreasonable effectiveness of deep features as a perceptual metric. In CVPR (2018). 7

[ZLC<sup>∗</sup>24] ZHU R., LIANG Y., CHANG H., DENG J., LU J., YANG W., ZHANG T., ZHANG Y.: Motiongs: Exploring explicit motion guidance for deformable 3d gaussian splatting. In NeurIPS (2024). 2, 7, 9

[ZLN<sup>∗</sup>24] ZHAN Y., LI Z., NIU M., ZHONG Z., NOBUHARA S., NISHINO K., ZHENG Y.: Kfd-nerf: Rethinking dynamic nerf with kalman filter. In ECCV (2024). 2

[ZLZ<sup>∗</sup>25] ZHANG X., LIU Z., ZHANG Y., GE X., HE D., XU T., WANG Y., LIN Z., YAN S., ZHANG J.: Mega: Memory-efficient 4d gaussian splatting for dynamic scenes. In ICCV (2025). 2