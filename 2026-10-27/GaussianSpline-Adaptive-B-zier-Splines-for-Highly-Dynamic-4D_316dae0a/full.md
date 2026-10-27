# GaussianSpline: Adaptive Bézier Splines for Highly Dynamic 4D Gaussian Splatting

Tom Yachil

Andrei Sharf

yachil@post.bgu.ac.il

asharf@cs.bgu.ac.il

Ben-Gurion University of the Negev, Israel

![](images/c156e2b480ffd1c1ac70fbb515cedb8956213240d568aadc8888fedf7b45360a.jpg)  
Figure 1: Demonstration of GaussianSpline on two highly dynamic 4D scenes (top and bottom rows). Left-to-right (columns) are the ground truth, reconstruction using splines with uniform-width control points (middle), and our adaptive splines with non-uniform temporal support (right). Zoom-ins highlight that GaussianSpline provides significantly better reconstructions ofrapidly changing motion.

## Abstract

We present GaussianSpline, a dynamic 4D Gaussian Splatting method that models per-Gaussian motion using adaptive, nonuniform piecewise cubic Bézier splines. Unlike approaches based on global deformations,fixed-orderpolynomials, or uniformly parameterized splines, GaussianSpline jointly learns the Bézier control points and the temporal partition of each Gaussian’s trajectory. This adaptive representation allocates greater temporal capacity to highly dynamic regions while maintaining compact trajectories for near-static regions, enabling high-fidelity reconstruction of complex and rapidly changing motion. To jointly achieve globally smooth motion and local adaptivity, we enforce $G ^ { 1 }$ continuity between consecutive Bézier segments using a reflection-based constraint on their control points. This produces smooth,fully differentiable trajectories while preserving the flexibility of non-uniform temporal parameterization. Experiments on the Neural 3D Video benchmark, the Google Immersive dataset [BFO 20], as well as on synthetic scenes demonstrate that GaussianSpline substantially improves reconstruction quality in scenes with highly non-uniform and challenging dynamics, while maintaining competitive performance on general dynamic scenes compared with state-of-the-art 4D Gaussian Splatting methods

CCS Concepts

• Computing methodologies  Rendering;

## 1. Introduction

Reconstructing dynamic 3D scenes from multi-view video is an emerging problem in computer graphics and vision, with applications in virtual reality, telepresence, and film production. The recent success of 3D Gaussian Splatting (3DGS) [KKLD23] has demonstrated that explicit, differentiable representations based on anisotropic Gaussians can achieve high-quality reconstruction and real-time rendering of static 3D scenes, motivating their extension to dynamic 4D content. However, moving from static to dynamic Gaussian Splatting introduces a fundamental challenge: the representation must simultaneously capture the spatial structure of the scene and the complex temporal evolution of potentially millions of Gaussians. In particular, accurately modeling highly dynamic regions requires flexible motion trajectories, while avoiding unnecessary temporal degrees of freedom in near-static regions.

Existing dynamic Gaussian Splatting methods typically rely on neural deformation fields or fixed spatial-temporal representations. Neural deformation approaches augment canonical Gaussians with MLP-based motion predictors [HSY 23, YGZ 24, LKL 25], introducing additional per-Gaussian computation during rendering. Structured 4D approaches instead factorize spacetime using voxel grids or plane-based representations [XPL<sup>∗</sup>24, WYF<sup>∗</sup>24], but rely on predefined spatial-temporal resolutions. These representations therefore face a trade-off between computational or memory cost and reconstruction fidelity, particularly for scenes with highly nonuniform dynamics, where different regions require different levels of temporal modeling.

To avoid the overhead of neural deformation networks, explicit trajectory-based methods [LCLX24, LDZY24] model Gaussian motion using global parametric functions, such as polynomials or Fourier bases. More recently, piecewise spline representations have been explored for trajectory modeling, offering local control and compact parameterization [PBB<sup>∗</sup>25, YHOL25]. However, these approaches typically employ a uniform temporal parameterization across Gaussian trajectories, allocating the same temporal capacity regardless of the underlying motion. This limits their ability to efficiently capture the highly heterogeneous dynamics of realworld scenes, where different regions may exhibit slow, smooth motion or rapid and complex changes.

To address these limitations, we introduce GaussianSpline, a per-Gaussian trajectory-based 4D Gaussian Splatting framework that models motion using adaptive piecewise cubic Bézier splines with learnable, non-uniform temporal parameterization. Each Gaussian is assigned an independent trajectory initialized with a minimal number of spline segments. During optimization, the Bézier control points and segment intervals are jointly learned, allowing each trajectory to allocate temporal resolution according to its local motion. Rapidly changing motion is captured with finer temporal intervals, while slowly varying motion is represented with broader intervals, avoiding unnecessary temporal degrees of freedom and yielding a compact representation.

To ensure smooth motion across spline segments, we enforce $G ^ { 1 }$ continuity using a reflection-based control-point formulation with direction-locked tangents and learnable magnitude scaling. This yields smooth, fully differentiable trajectories while preserving local flexibility in motion speed and curvature. GaussianSpline introduces no auxiliary neural networks and integrates directly into the standard 3DGS optimization and densification pipeline, providing a compact and efficient representation for highly non-uniform dynamics.

Experiments on the Neural 3D Video benchmark, the Google Immersive dataset [BFO<sup>∗</sup>20], and synthetic scenes demonstrate that GaussianSpline significantly improves reconstruction quality for scenes with complex, highly non-uniform motion, while maintaining competitive performance on general dynamic scenes compared with state-of-the-art 4D Gaussian Splatting methods. Our contributions are summarized as follows:

Learnable adaptive temporal parameterization. We jointly learn Bézier control points and non-uniform segment intervals to adapt temporal resolution to each Gaussian’s local motion.

Smooth motion transitions. We introduce a reflection-based $G ^ { 1 }$ continuity constraint with learnable tangent magnitudes for smooth and flexible trajectory modeling.

Efficient 4D representation. Our adaptive trajectories integrate seamlessly into the standard 3DGS framework, providing a compact and efficient representation of dynamic scenes.

## 2. Related Work

In the following we discuss recent related works on reconstruction of 3D dynamic scenes from multi-view video.

Dynamic neural radiance fields. Dynamic extensions of NeRF [MST 20] represent a scene as a time-varying radiance field, either conditioning the field directly on time [LSZ<sup>∗</sup>22] or warping each observation into a canonical frame through a learned deformation [PCPMMN21,PSH<sup>∗</sup>21], with later work improving robustness on casually captured monocular video [LGM<sup>∗</sup>23].

These methods produce high-quality free-viewpoint video, but inherit NeRF’s volumetric ray marching: rendering a single pixel requires hundreds of network evaluations along a ray, so training takes hours to days and playback remains far from real time [LSZ<sup>∗</sup>22]. Grid- and plane-based factorizations [FKMW<sup>∗</sup>23] reduce this cost substantially, yet the scene stays an implicit field sampled at a fixed spatial-temporal resolution, so motion is never an explicit quantity that can be allocated where the scene actually moves.

Dynamic 3DGS The emergence of 3D Gaussian Splatting (3DGS) [KKLD23] has significantly advanced neural rendering by replacing implicit volumetric sampling with explicit, differentiable rasterization of anisotropic Gaussians. Building on this success, recent work has focused on extending 3DGS to dynamic scenes by modeling temporal deformation of Gaussian attributes [LKLR24, LCLX24, WYF<sup>∗</sup>24].

Luiten et al. [LKLR24] directly optimize per-timestep Gaussian positions and rotations under local rigidity constraints, enabling dense 6-DOF tracking without an explicit deformation model. In contrast, MLP-based approaches learn continuous deformation fields in a canonical space. Deformable 3DGS [YGZ<sup>∗</sup>24] maps Gaussians from canonical to observation space via a neural deformation network, while GauFRe [LKL<sup>∗</sup>25] introduces a forwardwarping formulation with an explicit static component to decouple moving and stationary regions. Although expressive, these methods require per-Gaussian MLP evaluations at render time, leading to significant computational overhead [LCLX24]. SC-GS $\mathrm { [ H S Y ^ { * } } 2 3 ]$ mitigates this issue by combining deformation networks with sparse spatial constraints to reduce inference cost.

Another line of work improves efficiency through structured spacetime factorization. Originating from NeRF-based representations such as K-Planes $\mathrm { [ F K M W ^ { * } 2 3 ] }$ , these methods decompose the 4D domain into low-dimensional grids or plane representations. This idea has been adopted in 3DGS extensions such as 4DGS $[ \mathrm { W Y F ^ { * } } 2 4 ]$ and 4K4D $[ \bar { \mathrm { X P L } } ^ { * } 2 4 ]$ , which model dynamic Gaussian attributes using structured grid-based embeddings. While computationally efficient, these methods are inherently limited by discrete resolution, leading to a trade-off between memory usage and reconstruction fidelity [WYF<sup>∗</sup>24, LCLX24].

Explicit trajectory modeling. More recent methods abandon neural deformation networks in favor of explicit motion trajectories.

Spacetime Gaussians (STG) [LCLX24] represent Gaussian motion using polynomial trajectories, improving temporal smoothness and efficiency. Gaussian-Flow [LDZY24] further extends this idea by combining polynomial and Fourier-based representations in a dual-domain formulation, eliminating the need for neural networks while improving expressiveness.

However, fixed-degree polynomial models assign uniform temporal capacity to all Gaussians and may struggle to represent complex, non-linear motion patterns [LCLX24]. This limitation has motivated spline-based representations as a more flexible alternative.

Spline-based formulations provide smooth and locally controllable representations of motion trajectories. SplineGS $\left[ \mathrm { P B B } ^ { * } 2 5 \right]$ introduces piecewise cubic Hermite splines with motion-adaptive control-point pruning, while also reducing reliance on precomputed camera parameters. However, its formulation targets monocular video sequences rather than multi-view dynamic 4D reconstruction, as considered in our work.

More closely related to our setting, Yoon et al. [YHOL25] propose a method, also termed SplineGS, that employs NURBS to represent Gaussian trajectories as combinations of learned representative motion bases, improving temporal smoothness and spatial coherence in multi-view 4D dynamic scenes.

However, both approaches rely on uniform spline parameterizations with fixed control-point distributions. In contrast, our approach jointly learns adaptive, locally refined control-point distributions based on motion complexity. This allows temporal capacity to be allocated where it is most needed, providing a more efficient representation of highly non-uniform motion and improving reconstruction quality with fewer Gaussian splats.

## 3. Method

We first review the standard 3D Gaussian Splatting representation and then extend it to dynamic scenes by assigning each Gaussian a learnable piecewise cubic Bézier trajectory. Unlike fixed temporal parameterizations, our formulation jointly optimizes the spline parameters and the temporal placement of control points, allowing each Gaussian to adapt its temporal resolution to the underlying motion (see Figure 2).

3D Gaussian Splatting 3D Gaussian Splatting (3DGS) represents a scene as a set of anisotropic Gaussians, each parameterized by its position µ, covariance matrix Σ, opacity o, and viewdependent color c represented using spherical harmonics. Each Gaussian is defined as:

$$
G ( \mathbf { x } ) = \exp \left( - \frac { 1 } { 2 } ( \mathbf { x } - \boldsymbol { \mu } ) ^ { \top } \Sigma ^ { - 1 } ( { \mathbf { x } } - \boldsymbol { \mu } ) \right) ,\tag{1}
$$

with covariance parameterized as:

$$
\boldsymbol { \Sigma } = \mathbf { R } \mathbf { S } \mathbf { S } ^ { \top } \mathbf { R } ^ { \top } ,\tag{2}
$$

where S and R denote learnable scaling and rotation matrices, respectively.

For rendering, Gaussians are projected onto the image plane and rasterized using differentiable alpha compositing, where Gaussians are accumulated in depth order:

$$
\mathbf { C } = \sum _ { i \in N } \mathbf { c } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{3}
$$

Starting from an SfM point cloud, all Gaussian parameters are optimized end-to-end through differentiable rendering, while adaptive density control dynamically splits, clones, and prunes Gaussians during training. We refer to [KKLD23] for a detailed description of the 3DGS formulation.

4D Gaussian Splatting for dynamic scenes. To reconstruct a dynamic 4D scene, we lift static Gaussian splats to a time-varying representation over a normalized timeline $t \in [ 0 , 1 ]$ . In our formulation, only the Gaussian position $\mu ( t )$ and rotation $\mathbf { R } ( t )$ are timedependent, while the scale s, opacity o, and color c are shared across all timestamps. The time-varying covariance is therefore given by:

$$
\Sigma ( t ) = \mathbf { R } ( t ) \mathbf { S } \mathbf { S } ^ { \top } \mathbf { R } ( t ) ^ { \top } ,\tag{4}
$$

and the Gaussian at time t is obtained by evaluating Eq. (1) with $\mu ( t )$ and $\Sigma ( t )$

This design reflects the observation that many real-world motions can be approximated as piecewise locally rigid deformations, with temporal variation primarily captured by translation and rotation. Restricting temporal variation to these degrees of freedom yields a compact yet flexible representation for dynamic scene modeling.

Non-uniform Cubic Bézier Spline Trajectories To model the motion trajectory of a Gaussian splat over time, we use a piecewise cubic Bézier spline defined by $K + 1$ joints $\left\{ \mathbf { J } _ { 0 } , \mathbf { J } _ { 1 } , \ldots , \mathbf { J } _ { K } \right\}$ 2K interior control points $\{ \mathbf { C } _ { k , 0 } , \mathbf { C } _ { k , 1 } \} _ { k = 0 } ^ { K - 1 }$ , and K non-uniform interval widths $\left\{ \mathbf { w } _ { 0 } , \mathbf { w } _ { 1 } , \ldots , \mathbf { w } _ { K - 1 } \right\}$ . Note that all spline parameters are differentiable and jointly optimized within our framework (see Figure 2 (middle)).

![](images/506c66304ee911681dcc4fe50ff6e9ec6e1c4627458643683454fa3073f7260d.jpg)  
Figure 2: Pipeline overview. Left: multi-view videoframes sampled at random timestamps and the initial SfMpoint cloud. Middle: three per-Gaussian piecewise cubic Bézier trajectories with learnable, non-uniform temporal intervals. Right: Gaussians evolve along their learned trajectories to reconstruct the 4D scene, with rasterized frames compared against ground truth and the resulting photometric loss backpropagated to the spline parameters and Gaussian attributes.

![](images/31b1b940850688cd0065c0fbc68c861f31142ac561bcf9728f2d8295fc247c53.jpg)

![](images/2a9afadcd8b96f6a604d6cefeab8e9ffec07db44a43da6632df84e5e37897072.jpg)  
Figure 3: Real-data reconstruction of a 1D curve using uniform $( l e f t )$ and our adaptive non-uniform splines $( r i g h t )$ . Our method jointly learns the temporal interval placement, allocating more control points to high-curvature regions and fewer to slowly varying regions. This adaptive parameterization yields a significantly more accurate reconstruction (red curve) of the target samples (black dots).

To enforce monotonic partition of the entire temporal domain based on the learned adaptive non-uniform intervals we define:

$$
t _ { k } = \sum _ { j = 0 } ^ { k - 1 } \tilde { w } _ { j } ,\tag{5}
$$

where $\begin{array} { r } { \tilde { w } _ { k } = \frac { \exp ( \mathbf { w } _ { k } ) } { \sum _ { i = 0 } ^ { K - 1 } \exp ( \mathbf { w } _ { j } ) } } \end{array}$ is segment $k \mathrm { { s } }$ interval width, t<sub>k</sub> is $\mathbf { J } _ { k } \mathbf { \bar { s } }$ timestamp, and the joints timestamp sequence $0 = t _ { 0 } < t _ { 1 } < \cdots <$ $t _ { K } = 1$ is a valid monotonic partition of [0, 1] by construction.

To evaluate the trajectory at a global timestamp $t \in [ 0 , 1 ]$ , we locate the active segment k such that $t _ { k } \leq t < t _ { k + 1 }$ and compute the local parameter $\begin{array} { r } { u \ = \ \frac { t - t _ { k } } { \tilde { w } _ { k } } } \end{array}$ which maps u to [0, 1) within the active segment. The position $\mu ( t )$ is then obtained by evaluating the cubic Bézier curve on segment $k \mathrm { { s } }$ points, given by:

$$
\mu ( t ) = \left( 1 - u \right) ^ { 3 } \mathbf { J } _ { k } + 3 \left( 1 - u \right) ^ { 2 } u \mathbf { C } _ { k , 0 } + 3 ( 1 - u ) u ^ { 2 } \mathbf { C } _ { k , 1 } + u ^ { 3 } \mathbf { J } _ { k + 1 }\tag{6}
$$

Learnable interval widths. Rather than using a uniform partition of the time domain [0, 1], we parameterize each spline using learnable interval-width logits $\left\{ \mathbf { w } _ { 0 } , \mathbf { w } _ { 1 } , \ldots , \mathbf { w } _ { K - 1 } \right\}$ , which define a non-uniform temporal segmentation. Initialized to zero, these logits yield a uniform partition at the beginning of training and are subsequently optimized jointly with the spline joints and control points.

This allows each Gaussian to adapt its temporal resolution to the underlying motion, allocating finer intervals to rapidly changing motion and broader intervals to slowly varying motion, thereby improving reconstruction fidelity over an equivalent uniformly parameterized spline (see Figure 3).

Learnable motion speed. $C ^ { 0 }$ continuity is ensured by shared joints between adjacent segments. Rather than enforcing the stricter $\dot { \boldsymbol { C } } ^ { 1 }$ continuity, which couples both the direction and speed of motion across neighboring segments through continuity of the first derivative, we impose $\bar { G } ^ { 1 }$ continuity. This preserves the tangent direction at each internal junction while allowing its magnitude to vary through a learnable positive scalar $\lambda _ { k } > 0 \colon$

$$
{ \bf C } _ { k , 0 } = { \bf J } _ { k } + \lambda _ { k } \left( { \bf J } _ { k } - { \bf C } _ { k - 1 , 1 } \right) , \qquad k = 1 , \ldots , K - 1 .\tag{7}
$$

The per-junction scales $\lambda _ { k }$ are initialized to 1, recovering $C ^ { 1 }$ continuity at the start of training, and are jointly optimized with the spline parameters. This relaxation allows motion speed to vary smoothly across segments while preserving kink-free trajectories and tangent continuity throughout the spline.

Per-Gaussian polynomial rotations. To compute the per-Gaussian time-dependent rotation matrix $R ( t )$ , we follow [LCLX24] and parameterize the raw quaternion as a first-degree polynomial, mapping its normalized form to the rotation group:

$$
q ( t ) = q _ { 0 } + q _ { 1 } t , \quad R ( t ) = \mathcal { R } \bigg ( \frac { q ( t ) } { \| q ( t ) \| } \bigg )\tag{8}
$$

![](images/c58dc70a25f37bb9c3c4f8b0fb957a56098b6c805bd53bee1c03868a0e9024c8.jpg)  
Figure 4: Qualitative evaluation on five dynamic scenes (rows): checker sphere (frame 59), soft contact (frame $4 2 ) ,$ cook spinach (frame 47), coffee martini (frame 0), and flames (frame 40), with zoom-ins. From left to right: ground truth, Gaussian-Flow, STG, SplineGS and our method. Zoom-ins demonstrate that our adaptive non-uniform splines better preserve fine details in moving objects while maintaining sharpness in near-static regions, whereas competing methods produce motion blur or losefine details.

where $q _ { 0 }$ and $q _ { 1 }$ denote the base quaternion and its linear velocity, respectively, and $\mathcal { R } ( \cdot )$ is the standard mapping from unit quaternions to rotation matrices.

Training. In each training iteration, we pick a random frame index from the training set, calculate its timestamp and select a subset of camera views captured at this timestamp. Then, every Gaussian is evaluated at this timestamp through its active segment (Eq. 6), and the photometric loss w.r.t. the selected views is backpropagated to the spline and Gaussian parameters.

Together with the static initialization (all joints and control points set to $\mathbf { x } _ { 0 } ) ,$ this generates the Gaussian motion trajectories. Note that in near-static regions the gradients average or are near zero, leaving the trajectory approximately flat and consuming no temporal capacity.

Furthermore, we apply Gaussians adaptive densification steps, i.e., split/clone/prune operations in early stage training iterations (as in [KKLD23]). Afterwards, the set of Gaussians remains fixed, and only the spline parameters (joints, control points, widths, $G ^ { 1 }$ scales) and the Gaussian attributes (opacity, color, covariance) are optimized.

## 4. Experiments and Results

Implementation Details. We train for 30k iterations using Adam on a single RTX 4090, optimizing an $\mathcal { L } _ { 1 }$ loss with a D-SSIM term $( \lambda = 0 . 2 )$ , SH degree 3, and batch size 2. Following 3DGS, the position learning rate decays exponentially from $1 . 6 \times 1 0 ^ { - 4 }$ to $1 . 6 \times 1 0 ^ { - 6 }$ , while the learning rates for features, opacity, scaling, and rotation are $2 . 5 { \times } 1 0 ^ { - 3 } , \bar { 5 } { \times } 1 0 ^ { - 2 } , 5 { \times } 1 0 ^ { - 3 }$ , and $1 0 ^ { - 3 }$ , respectively. The spline joint-width learning rate is set to $0 . 3 \times$ the position learning rate. Densification is performed every 100 iterations from iteration 500 to 9k, using a gradient threshold of $2 \times 1 0 ^ { - 4 }$ and a maximum of six densification events. Opacity is reset every 3k iterations, and Gaussians with opacity below 0.005 or scale above 0.1 the scene extent are pruned. Unless otherwise stated, we use K = 3 spline segments with non-uniform joint spacing and $G ^ { 1 }$ continuity, initialized from approximately 8k COLMAP SfM points from a single frame on N3DV.

Baselines We compare our method with three other SOTA methods: Gaussian-Flow [LDZY24], STG (Spacetime Gaussian) [LCLX24] and SplineGS [YHOL25].

We select these methods because they target the same multi-view dynamic 4D reconstruction setting and, like our approach, explicitly model Gaussian motion trajectories. They therefore provide relevant baselines for evaluating different approaches to temporal motion representation. STG represents Gaussian motion using polynomial trajectories, Gaussian-Flow combines polynomial and Fourier-based representations, and SplineGS employs piecewise cubic Hermite splines with motion-adaptive control-point pruning. These methods constitute the current set of SOTA approaches that explicitly model Gaussian motion trajectories in the multi-view dynamic 4D reconstruction setting, providing direct baselines for evaluating our adaptive temporal parameterization on highly nonuniform motion.

In our experiments, we use the default training schedules of 30K, 30K, and 20K iterations for Gaussian-Flow, STG, and SplineGS, respectively. Initialization varies by scene type: for real-world sequences, we initialize from SfM, as ground-truth geometry is unavailable, while for synthetic scenes, we initialize from the groundtruth point clouds. All methods use the same views, frames, and image resolution, and are evaluated at their respective final checkpoints.

For STG, we use the without temporal opacity configuration defined in [LCLX24], in which each Gaussian’s opacity remains constant over time. This setting enables a direct comparison of the explicit Gaussian trajectories and motion. Allowing temporal opacity could otherwise represent motion implicitly by fading Gaussians in and out rather than explicitly moving them through space. Since trajectory-based methods model motion through explicit Gaussian trajectories, we retain constant opacity for a consistent comparison.

Datasets. The Neural 3D Video (N3DV) dataset [LSZ<sup>∗</sup>22] comprises multi-view recordings of dynamic real-world scenes captured by 21 synchronized cameras at a resolution of 2704 2028 and 30 FPS. The six publicly available scenes contain approximately 10 seconds of video and feature complex non-rigid motion. Following prior work [WYF<sup>∗</sup>24, LDZY24], we downsample the images to 1352 1014.

We evaluate our method also on the Google Immersive light field video dataset [BFO<sup>∗</sup>20] (flame scene), captured by an outwardfacing rig of 46 synchronized fisheye cameras mounted on a hemispherical dome. We use the 45 calibrated cameras, holding out camera\_0001 for evaluation and training on the remaining 44, over 50 frames. Following STG [LCLX24], the fisheye images are undistorted to a pinhole model and downsampled to 1280 960. The scene features a flamethrower whose rapidly evolving, semitransparent flame contrasts with a largely static background.

We additionally evaluate on two ray-traced synthetic dynamic scenes, checker sphere and soft contact, with known ground-truth geometry. Both scenes use the same camera rig of 20 static cameras spanning 90 in azimuth across two rings, with 60 frames captured at 30 FPS. Images are rendered at 2704 2028 and downsampled to $1 3 5 2 \times 1 0 1 4$ , with the central view held out for evaluation.

checker sphere features a checker-patterned sphere moving laterally within a five-plane room, following two sinusoidal motion segments with different frequencies.

soft contact contains 16 balls with different masses (m $\propto r ^ { 3 } )$ undergoing collisions with an invisible box. Contacts are modeled using a penalty spring-damper over a finite 0.2 s interval, producing continuous velocities with large, localized accelerations. Both scenes are procedurally generated, providing per-frame groundtruth geometry for quantitative evaluation.

Qualitative Evaluation. Figure 1 compares between reconstruction with uniform-width splines (middle) and our adaptive splines with non-uniform temporal support (right). The top and bottom rows show checker sphere and cook spinach, respectively. The zoomed-in regions demonstrate the benefit of adaptive temporal allocation: GaussianSpline better recovers the earth’s spheregeometry, texture, and shadow (top), as well as the hands, fingers, and gripper (bottom), while uniform splines produce blurred or incomplete details.

Figure 4 provides qualitative comparisons on five dynamic scenes (rows): two synthetic sequences with known ground truth 3D motion, checker sphere (frame 59) and soft contact (frame 42), and three real-world sequences: cook spinach (frame 47) and coffee martini (frame 0) from N3DV, and flames (frame 40) from Google Immersive. Columns show the ground truth, Gaussian-Flow, STG, SplineGS, and ours with zoom-ins.

On the checker sphere, we focus on a rapidly horizontally moving sphere with a checker pattern. Gaussian-Flow blurs the quadrant boundaries, STG completely distorts the sphere and its shadow, while SplineGS blurs the facets and completely misses the red facet. In contrast, GaussianSpline preserves the quadrant boundaries, sphere geometry, and elliptical shadow, closely matching the ground truth.

soft contact represents the most challenging case, with rapid collisions between small spheres. Gaussian-Flow fails to preserve the boundaries of closely colliding spheres, producing bleeding artifacts between them. STG collapses colliding spheres into unified smears, while SplineGS produces inter-sphere artifacts and blur. In contrast, GaussianSpline preserves the individual sphere boundaries and geometry.

The cook spinach real-world scene contains regions with different motion patterns. The hand and grippers move continuously, while the dog remains largely static before turning its head in the second half of the sequence. Zoom-ins show that all other methods blur the dog’s fur, whereas ours produces the sharpest reconstruction. This improvement comes from our adaptive partition, which concentrates temporal resolution around the localized head motion rather than allocating capacity uniformly across the sequence.

Table 1: Quantitative evaluation on the two synthetic scenes (checker sphere and soft contact, and the three real-world scenes (cook spinach and coffee martini from N3DV, and flames from Google Immersive [BFO<sup>∗</sup>20]). We compare our method with three other SOTA: Gaussian-Flow, STG and SplineGS using PSNR, SSIM and LPIPS error metrics. The best result per scene and metric is shown in bold; underlined entries marks the second best.
<table><tr><td rowspan="2">Method</td><td colspan="3">checker sphere</td><td colspan="3">soft contact</td><td colspan="3">cook spinach</td><td colspan="3">coffee martini</td><td colspan="3">flames</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Gaussian-Flow</td><td>37.42</td><td>0.9875</td><td>0.0495</td><td>31.10</td><td>0.9829</td><td>0.0308</td><td>24.65</td><td>0.8771</td><td>0.2293</td><td>23.42</td><td>0.8571</td><td>0.2398</td><td>21.69</td><td>0.7453</td><td>0.3878</td></tr><tr><td>STG</td><td>34.78</td><td>0.9920</td><td>0.0198</td><td>23.42</td><td>0.9623</td><td>0.0551</td><td>31.93</td><td>0.9516</td><td>0.1393</td><td>26.80</td><td>0.9066</td><td>0.1628</td><td>23.96</td><td>0.8888</td><td>0.1680</td></tr><tr><td>SplineGS</td><td>40.37</td><td>0.9953</td><td>0.0375</td><td>24.55</td><td>0.9805</td><td>0.0336</td><td>32.67</td><td>0.9528</td><td>0.2007</td><td>27.73</td><td>0.9059</td><td>0.2188</td><td>21.39</td><td>0.7309</td><td>0.4742</td></tr><tr><td>Ours</td><td>41.18</td><td>0.9956</td><td>0.0151</td><td>26.84</td><td>0.9845</td><td>0.0289</td><td>32.92</td><td>0.9522</td><td>0.1531</td><td>27.99</td><td>0.9087</td><td>0.1743</td><td>27.42</td><td>0.9068</td><td>0.1498</td></tr></table>

In coffee martini, all other methods fail to accurately reconstruct the round cap and its pointed tip, either blurring them (Gaussian-Flow and STG) or distorting their geometry (SplineGS). In contrast, ours is the only method that faithfully reconstructs both the cap’s rounded geometry and sharp tip.

Finally, in flames, the rapidly evolving flame and the moving hand holding the can present the greatest challenges. Gaussian-Flow, STG, and SplineGS collapse the flame into a saturated, blurry blob and smear the hand and can, whereas our method preserves more of the flame’s tongue-like structure and reconstructs the hand and can with the highest sharpness (see zoom-ins).

Quantitative Evaluation. Table 1 summarizes the quantitative evaluation of our method in comparison with the other three SOTA methods: Gaussian-Flow, STG, and SplineGS on the five sequences, i.e., checker sphere, soft contact, cook spinach, coffee martini, and flames.

Every method is trained with its own default iteration budget: Gaussian-Flow and STG use 30k iterations on all five sequences; SplineGS uses 20k on checker sphere, cook spinach, coffee martini and flames, and 40k on soft contact; our method uses 20k on checker sphere, 40k on soft contact, and 30k on the three real captures.

Our method achieves the highest PSNR on all three real-world scenes, with a margin of 3.46 dB on flames, as well as on the synthetic checker sphere scene, and ranks second on soft contact.

For SSIM, ours achieves the highest score on both synthetic scenes, i.e., checker sphere, and soft contact, as well as on coffee martini andflames, and ranks second on cook spinach.

For LPIPS, it achieves the best score on both synthetic scenes ( checker sphere and soft contact) and on flames, and ranks second on cook spinach and coffee martini, with only a small difference from the top-performing method in both cases.

Note that while our method ranks first or second across all tests and metrics, its performance is notably stronger on the synthetic scenes. This is expected, as our method is specifically designed to handle highly non-uniform dynamics. Our synthetic scenes were deliberately designed to exhibit such heterogeneous temporal dynamics, providing a controlled setting in which to evaluate the benefits of adaptive temporal allocation.

Ablation Study. Table 2 ablates the two design choices of our temporal parameterization: the spline order K and the placement of the joints. For each order, we compare uniform spacing with our learnable non-uniform spacing.

Non-uniform spacing consistently outperforms (or matches) uniform spacing across all spline orders. Interestingly, increasing the spline order does not improve reconstruction quality. Instead, the lowest-order spline with non-uniform spacing achieves the best overall results, suggesting that adaptive joint placement is more important than spline order. Moreover, the performance gap between non-uniform and uniform spacing decreases as the spline order increases, as expected, since higher-order splines provide greater flexibility even with uniformly spaced joints.

Table 2: Ablation on spline order K with uniform and non-uniform joint spacing. All metrics are averaged over five scenes (cut roasted beef, cook spinach, sear steak,flame steak, coffee martini).
<table><tr><td>K</td><td>Joints</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="2">2</td><td>Uniform</td><td>31.71</td><td>0.9459</td><td>0.0537</td></tr><tr><td>Non-uniform</td><td>32.08</td><td>0.9481</td><td>0.0540</td></tr><tr><td rowspan="2">3</td><td>Uniform</td><td>30.83</td><td>0.9441</td><td>0.0565</td></tr><tr><td>Non-uniform</td><td>31.74</td><td>0.9469</td><td>0.0551</td></tr><tr><td rowspan="2">4</td><td>Uniform</td><td>31.61</td><td>0.9454</td><td>0.0579</td></tr><tr><td>Non-uniform</td><td>31.61</td><td>0.9446</td><td>0.0572</td></tr></table>

## 5. Conclusions and limitations.

We presented GaussianSpline, a dynamic 4D Gaussian Splatting method in which each Gaussian follows a piecewise cubic Bézier trajectory with a learnable, non-uniform temporal partition. With a fixed segment budget, trajectories allocate temporal resolution where motion is fast and use larger intervals where it is slow. A differentiable reflection-based $G ^ { \dagger }$ constraint ensures smooth trajectories without restricting local motion speed, and the method integrates directly into the standard 3DGS optimization and densification pipeline without auxiliary networks.

Our ablation study shows that non-uniform spacing consistently outperforms or matches uniform spacing across spline orders, while increasing the order provides little benefit (Table 2). GaussianSpline achieves the best overall results on checker sphere and leads PSNR on real N3DV and Google Immersive scenes.

Our approach has several limitations. First, while our spline representation is effective for a wide range of motion, it can struggle when the dynamics become highly transient. In Soft Contact, for example, rapid collisions produce large accelerations over very short intervals that are difficult to capture accurately with a fixed, small number of spline segments. A promising direction is therefore adaptive segment insertion, guided by local curvature or reconstruction residuals, to dynamically increase temporal resolution where needed.

Second, our trajectories are purely temporal and optimized independently for each Gaussian, without explicit spatial motion priors or coupling between neighboring primitives. As a result, rigid and articulated structures must be learned through individual Gaussian trajectories, while topological changes, such as the appearance and disappearance of geometry, are handled implicitly through opacity and densification.

Finally, our evaluation is limited to the publicly available N3DV scenes, one Google Immersive scene, and two synthetic sequences, with each baseline trained using its default optimization schedule. Broader evaluation is desirable, but diverse multi-view dynamic datasets with dense temporal sampling and accurate ground truth remain scarce. As more public datasets become available, evaluation across a wider range of motions and scene dynamics will provide a more comprehensive assessment.

## References

[BFO<sup>∗</sup>20] BROXTON M., FLYNN J., OVERBECK R., ERICKSON D., HEDMAN P., DUVALL M., DOURGARIAN J., BUSCH J., WHALEN M., DEBEVEC P.: Immersive Light Field Video with a Layered Mesh Representation. ACM Transactions on Graphics (Proc. SIGGRAPH) 39, 4 (2020), 86:1–86:15. 1, 2, 6, 7

[FKMW<sup>∗</sup>23] FRIDOVICH-KEIL S., MEANTI G., WARBURG F. R., RECHT B., KANAZAWA A.: K-Planes: Explicit Radiance Fields in Space, Time, and Appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), pp. 12479–12488. 2, 3

[HSY<sup>∗</sup>23] HUANG Y.-H., SUN Y.-T., YANG Z., LYU X., CAO Y.-P., QI X.: SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes. arXiv preprint arXiv:2312.14937 (2023). 2, 3

[KKLD23] KERBL B., KOPANAS G., LEIMKÜHLER T., DRETTAKIS G.: 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics (TOG) (2023). 2, 3, 5

[LCLX24] LI Z., CHEN Z., LI Z., XU Y.: Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2024), pp. 8508–8520. 2, 3, 4, 6

[LDZY24] LIN Y., DAI Z., ZHU S., YAO Y.: Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024), pp. 21136–21145. 2, 3, 6

[LGM<sup>∗</sup>23] LIU Y.-L., GAO C., MEULEMAN A., TSENG H.-Y., SARAF A., KIM C., CHUANG Y.-Y., KOPF J., HUANG J.-B.: Robust Dynamic Radiance Fields. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023). 2

[LKL<sup>∗</sup>25] LIANG Y., KHAN N., LI Z., NGUYEN-PHUOC T., LANMAN D., TOMPKIN J., XIAO L.: GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis. In Proc. IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV) (2025). 2

[LKLR24] LUITEN J., KOPANAS G., LEIBE B., RAMANAN D.: Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis. In 3DV (2024). 2

[LSZ<sup>∗</sup>22] LI T., SLAVCHEVA M., ZOLLHÖFER M., GREEN S., LASS-NER C., KIM C., SCHMIDT T., HERBST S., SARAGIH J., LUVIZON D.:

Neural 3D Video Synthesis from Multi-view Video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022), pp. 5521–5531. 2, 6

[MST<sup>∗</sup>20] MILDENHALL B., SRINIVASAN P. P., TANCIK M., BARRON J. T., RAMAMOORTHI R., NG R.: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In European Conference on Computer Vision (ECCV) (2020). 2

[PBB<sup>∗</sup>25] PARK J., BUI M.-Q. V., BELLO J. L. G., MOON J., OH J., KIM M.: SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (CVPR) (June 2025), pp. 26866–26875. 2, 3

[PCPMMN21] PUMAROLA A., CORONA E., PONS-MOLL G., MORENO-NOGUER F.: D-NeRF: Neural Radiance Fields for Dynamic Scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021), pp. 10318–10327. 2

[PSH<sup>∗</sup>21] PARK K., SINHA U., HEDMAN P., BARRON J. T., BOUAZIZ S., GOLDMAN D. B., MARTIN-BRUALLA R., SEITZ S. M.: Hyper-NeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields. ACM Transactions on Graphics (TOG) 40, 6 (2021). 2

[WYF<sup>∗</sup>24] WU G., YI T., FANG J., XIE L., ZHANG X., WEI W., LIU W., TIAN Q., WANG X.: 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2024), pp. 20310– 20320. 2, 3, 6

[XPL<sup>∗</sup>24] XU Z., PENG S., LIN H., HE G., SUN J., SHEN Y., BAO H., ZHOU X.: 4K4D: Real-Time 4D View Synthesis at 4K Resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024). 2, 3

[YGZ<sup>∗</sup>24] YANG Z., GAO X., ZHOU W., JIAO S., ZHANG Y., JIN X.: Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. In CVPR (2024). 2

[YHOL25] YOON J., HAN S., OH J., LEE M.: SplineGS: Learning Smooth Trajectories in Gaussian Splatting for Dynamic Scene Reconstruction. In The Thirteenth International Conference on Learning Representations (2025). 2, 3, 6