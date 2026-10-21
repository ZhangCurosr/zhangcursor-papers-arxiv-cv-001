# Can a Display Be Used for Rapid LiDAR-Camera Extrinsic Calibration?

Zuhao Zou, Graduate Student Member, IEEE, Xiangcheng Hu, Jianhao Jiao, and Jin Wu, Senior Member, IEEE

Abstract—ScreenCalib enables light detection and ranging (LiDAR)–camera extrinsic calibration using an ordinary programmable display, providing known visual markers and controlled geometry while reducing dedicated target preparation. The camera observes displayed AprilTags, whereas the LiDAR measures the physical screen surface and boundary, requiring indirect cross-modal association. Display geometry introduces association ambiguities, while indoor clutter complicates extraction. ScreenCalib addresses these challenges through tag-guided display extraction, geometric hypothesis generation and refinement, and surrounding scene edges for ambiguity resolution. The singlepose mode requires only one static placement. The joint multipose mode combines independent static observations through cross-pose consensus and jointly refines the shared extrinsics and tag scale, reducing sensitivity to tag-size measurement errors without requiring continuous motion or inter-pose trajectory estimation. Experiments in simulation and on three different sensor platforms cover mechanical and nonrepetitive LiDARs in clean and cluttered environments. Benchmark results show that ScreenCalib outperforms competitive calibration-board-based methods in translation accuracy. Joint multi-pose refinement further reduces mean rotation and translation errors across the three platforms and improves rendered-map alignment. Evaluation across readily available displays, including cell phones, tablets, laptops, computer monitors, and televisions, further examines ScreenCalib’s versatility as a practical alternative to dedicated calibration targets. The source code is available at https://github.com/ZuhaoZou2026/ScreenCalib.

Index Terms—Camera calibration, light detection and ranging (LiDAR), multisensor systems, robot sensing systems.

## I. INTRODUCTION

Accurate extrinsic calibration between light detection and ranging (LiDAR) sensors and cameras is fundamental to multimodal perception [1], multimodal place recognition [2], and three-dimensional reconstruction [3]. In practical robotic systems, however, calibration is constrained not only by estimation accuracy, but also by how easily reliable calibration observations can be obtained during deployment. Existing methods typically make a trade-off between these two requirements. As illustrated in Fig. 1, target-based methods deliberately introduce known and repeatable geometric structures, but require dedicated target preparation and deployment. Targetless methods substantially reduce this burden, but the available calibration constraints depend on the surrounding environment and sensor motion. When the scene does not provide sufficiently informative geometry or texture, or when the motion excitation is inadequate, the calibration problem can become weakly constrained, leading to unstable estimates, degraded accuracy, or even calibration failure. This raises a straightforward practical question: can we achieve reliable and repeatable LiDAR–camera calibration without carrying and setting up a dedicated calibration target?

Target-based LiDAR–camera calibration uses specially designed structures, such as planar boards, circular holes, or checkerboards, to establish explicit cross-modal correspondences [4]–[6]. The known and repeatable target geometry (as priors) generally provides well-constrained and reliable calibration. However, this comes with a relatively high deployment burden, as the target must be fabricated or printed, accurately measured, transported, and positioned for each calibration.

Targetless methods occupy the other end of this tradeoff. They avoid dedicated calibration objects by exploiting natural scene appearance, geometric structures, sensor motion, or learned cross-modal correspondences [7], [8]. This reduces deployment burden, but the resulting calibration strongly depends on scene structure and motion excitation; insufficient or repetitive observations can lead to unstable estimates or calibration failure. Some methods also rely on accurate initialization, odometry, or trajectory optimization, further increasing sensitivity to acquisition conditions.

ScreenCalib is designed to reduce this dependency without returning to a specially manufactured calibration target. Instead of carrying and setting up a dedicated board, we use an ordinary display as a programmable cross-modal calibration interface. The displayed pattern provides known and repeatable visual geometry to the camera, while the physical display surface and its finite boundary provide geometric structure to the LiDAR. The displayed content can be electronically generated and adjusted according to sensing distance, field of view, and image resolution, without printing, attaching, or machining a dedicated target. As illustrated in Fig. 1, this places ScreenCalib between conventional target-based and targetless calibration: it preserves a user-provided and repeatable calibration structure while substantially reducing the deployment burden associated with a dedicated target.

Importantly, however, a display is not simply an electronic replacement for a conventional calibration board. In conventional target-based calibration, both sensing modalities are typically associated through the same physical target features. In ScreenCalib, the two modalities observe different manifestations of the display. The camera directly observes the displayed AprilTags, whereas the LiDAR generally cannot observe these visual fiducials and instead measures the physical screen surface and boundary. The cross-modal relationship must therefore be established indirectly through their shared display geometry. This is the first practical difficulty.

![](images/3b29aa318d5d73522762da6ab740aa452711f77fec6ecc38f06d3cafed01858c.jpg)  
Fig. 1. Comparison of target-based, ScreenCalib, and targetless LiDAR–camera calibration. Target-based methods provide known and repeatable geometry and direct cross-modal association, but require dedicated target preparation. Targetless methods minimize deployment burden by exploiting natural scene structure, but the available calibration geometry depends on the environment. ScreenCalib uses an ordinary programmable display to retain known calibration geometry while reducing dedicated target preparation. Its main challenges are the indirect association between camera and LiDAR observations and the cluttered environments in which displays are commonly located.

The second one is that, the planar geometry of the display inherently introduces association ambiguities: multiple permutations of the four image and LiDAR corners may produce apparently plausible initial transformations. These characteristics make display-based calibration substantially different from directly applying a conventional calibration-board pipeline.

The third practical difficulty arises from where displays are normally used. Unlike a dedicated calibration target that can be placed in a clean acquisition area, displays are commonly located in offices, laboratories, and other cluttered indoor environments. Nearby objects can generate competing image edges and planar LiDAR clusters, while the display boundary may be sparsely or noisily sampled, especially by nonrepetitive solid-state LiDARs.

Furthermore, ScreenCalib requires a measured AprilTag side length as input, and this measurement may be inaccurate. In single-pose data, LiDAR measurement noise can inflate points near the display boundary, making it difficult to jointly estimate the tag size and extrinsics accurately. This motivates joint multi-pose calibration.

To address these challenges, we propose ScreenCalib, a display-based LiDAR–camera calibration framework with single-pose (SP) and multi-pose (MP) modes. The main contributions are as follows:

• Robust Display Extraction in Two Different Modalities. We use an ordinary programmable display as a calibration target. On the camera side, AprilTag detections define a reliable polygon around the displayed pattern. Because the pattern and outer display boundaries are parallel in the display plane, the projected pattern edges guide the search for the outer display boundary in the image. The AprilTag-derived depth prior rangefilters LiDAR points for display-cloud extraction. The detected tags also provide a pixel-to-meter ratio, which, together with the image-polygon size, gives a metric size cue for LiDAR display extraction. Experiments in cluttered environments demonstrate the robustness of display extraction.

• Single-pose calibration with indirect cross-modal association. We develop a complete calibration pipeline requiring only one static display placement. Multiple camera–LiDAR display associations are explicitly retained, initialized from the display geometry, refined by finite-support registration, and validated using surrounding cross-modal scene structures. This enables rapid calibration even when the display is surrounded by cluttered indoor objects.

• Joint multi-pose calibration. We extend the framework to multiple independently captured static display observations without requiring continuous motion or interpose trajectory estimation. Cross-pose consensus selection identifies a consistent initialization from single-pose extrinsic proposals. Joint optimization then combines image-edge, plane, and normal constraints to refine a shared LiDAR–camera transformation and a correction to the user-measured AprilTag size.

• Carefully designed experimental evaluation. We examine complementary aspects of the framework: simulation with ground truth; three real LiDAR–camera systems, including mechanical and nonrepetitive LiDARs; clean and cluttered environments; indirect real-world accuracy assessment against Kalibr-estimated inter-camera extrinsics; qualitative assessment through rendered point-cloud alignment; sensitivity to user-measured AprilTag size;

![](images/6e5c8f78e7a16a07d033091841dc06b6acb3248a537c86575dade107366463ba.jpg)  
Fig. 2. Examples of display devices used in the ScreenCalib evaluation, each showing the calibration pattern: (a) cell phone, (b) tablet, (c) laptop, (d) computer monitor, and (e) television.

and calibration across five display types (Fig. 2).

## II. RELATED WORK

## A. Target-Based Methods

Target-based LiDAR–camera calibration establishes explicit geometric correspondences using calibration objects with known patterns or structures. Existing approaches differ in their features, preparation requirements, and how targets are isolated from the scene.

Zhou et al. [4] combine checkerboard plane and boundary constraints to estimate extrinsics from a single pose, requiring reliable LiDAR edge extraction. Verma et al. [9] use checkerboard centers and normals, with user-specified board dimensions and spatial filtering bounds.

Beltran et al. [5] register a circular-hole target against´ multilayer LiDAR data using manual spatial bounds.

Dong et al. [10] address solid-state LiDAR calibration through a multi-target, multi-shot strategy: pseudoimages provide corner correspondences for coarse initialization, followed by refinement using multiple target planes. LCE-Calib [6] is also a multi-shot method, tracking a checkerboard across frames and accumulating point-to-plane constraints for initialization before refinement with plane and edge constraints. These methods thus exploit multiple target observations in different ways: Dong et al. use several patterned targets, whereas LCE-Calib aggregates repeated observations of a tracked checkerboard.

ScreenCalib instead uses an ordinary display, pairing camera-observed AprilTags with the LiDAR-visible screen surface and boundary. Its multi-pose mode combines independently captured static placements through cross-pose consensus and joint refinement of the shared extrinsics and tag scale.

## B. Targetless Methods

Targetless LiDAR–camera calibration avoids dedicated calibration objects by exploiting natural scene appearance, geometric features, or sensor motion. These approaches simplify target preparation, but their applicability depends on the availability of the measurements needed to constrain calibration.

Pandey et al. [11] maximize mutual information between LiDAR reflectivity and image intensity, requiring sufficient statistical dependence between the two measurements. Geometric approaches align natural features: Yuan et al. [7] use edges formed by intersecting surfaces, while Zhang et al. [12] use line features for online calibration. These methods require reliably associated and sufficiently distributed scene features.

CMRNext [13] predicts image-to-LiDAR correspondences and uncertainties, followed by robust perspective-n-point (PnP) estimation, and supports unseen sensor configurations without retraining. MDPCalib [14] combines odometry-based initialization with learned correspondences and requires reliable odometry and adequate motion.

iKalibr [8] performs targetless spatiotemporal calibration of heterogeneous sensors through dynamic initialization and continuous-time optimization. It requires informative motion and sufficient trajectory observations.

Han et al. [15] propose UniCalib, which converts camera images and sparse LiDAR points into unified dense depth maps and estimates probabilistic flow to represent cross-modal correspondence uncertainty. A reliability map and perceptually weighted sparse-flow loss suppress unreliable regions, improving robustness to occlusions and dynamic objects.

Compared with targetless methods, ScreenCalib uses an ordinary programmable display to provide known, repeatable geometric constraints, reducing dependence on natural scene structure, texture, and sensor-motion excitation. Thus, ScreenCalib combines the low deployment burden of targetless methods with reliable geometric and marker constraints. It requires neither LiDAR reflectivity nor trajectory estimation. Surrounding-scene geometry helps distinguish competing single-pose hypotheses when the display outline is ambiguous.

## III. PROBLEM FORMULATION

The pipeline of the proposed method is shown in Fig. 3. Furthermore, this section formulates the problem of estimating the extrinsic transformation of a rigidly mounted LiDAR–camera system from observations of an AprilTag-patterned display. It also introduces the key variables and geometric objects used throughout the method, whose notation is summarized in Tab. I.

## A. Coordinate Frames and Calibration Variables

Let $L , C ,$ and M denote the LiDAR, camera, and display coordinate frames, respectively. The display frame is attached to the displayed pattern, with its z-axis normal to the display plane. A point expressed in frame A is denoted by $^ A \mathbf { p }$ , and ${ ^ { A } { \bf T } _ { B } }$ maps coordinates from frame B to frame A.

The unknown extrinsic is $\mathbf { T } = { ^ { C } } \mathbf { T } _ { L } = ( \mathbf { R } , \mathbf { t } ) \in \mathrm { S E } ( 3 )$ where ${ \bf R } = { \bf \nabla } ^ { C } { \bf R } _ { L }$ and $\mathbf { t } = ^ { C } \mathbf { t } _ { L }$ , satisfying

$$
{ \bf \Delta } ^ { C } { \bf p } = { \bf R } ^ { L } { \bf p } + { \bf t } .\tag{1}
$$

![](images/3f81908db6271f0b28d8097211401dca3685d261f3ef73d26f016b4e04443c0c.jpg)  
Fig. 3. The ScreenCalib calibration pipeline.

<table><tr><td>TABLE I KEY NOTATIONS</td></tr><tr><td>Notation Explanation</td></tr><tr><td> $L , C ,$  M LiDAR, camera, and display coordinate frames.  $^ A \mathbf { p }$  A point expressed in coordinate frame A.</td></tr><tr><td> ${ ^ A \mathbf { T } } _ { B }$  Transformation mapping coordinates from frame B to frame A.  $\mathbf { T }$  LiDAR-to-camera extrinsic, for simplicity,  ${ \textbf { T } } = { \bf \nabla } ^ { C } { \bf T } _ { L } { \bf \nabla } = { \bf \nabla }$   $( \mathbf { R } , \mathbf { t } ) \in \mathrm { S E } ( 3 )$ </td></tr><tr><td> $\mathbf { R } , \mathbf { t }$  Rotation and translation components of the LiDAR-to-camera extrinsic.</td></tr><tr><td> $\pi _ { C }$  Calibrated projection into the undistorted image.  $s _ { 0 }$  Nominal AprilTag side length used to set metric scale.</td></tr><tr><td> $C _ { \mathfrak { Q } }$  Virtual display cloud reconstructed in the camera frame.  ${ \boldsymbol { L } } { \boldsymbol { \mathcal { Q } } } _ { c }$  The c-th candidate display cloud extracted from the LiDAR</td></tr><tr><td>observations, expressed in the LiDAR frame.  $\mathbf { u } _ { \cdot } ^ { \mathrm { d i s p } }$  The j-th 2D corner of the display polygon in the image.  $\mathbf { \Pi } _ { r } ^ { \mathbf { u } } { } _ { j }$   $^ { L } \mathbf { b } _ { c , j }$  The j-th 3D corner of the c-th LiDAR display candidate.</td></tr></table>

Camera intrinsics and distortion parameters are calibrated beforehand. All images are undistorted upon receipt, and $\pi _ { C }$ projects points into these undistorted images. The user measures the physical side length $s _ { 0 }$ of an AprilTag directly on the display to provide the nominal metric scale.

## B. Calibration Overview

The inputs are camera images and LiDAR scans, collected under static placement. Camera intrinsics and distortion are calibrated beforehand.

The display is a shared planar target. AprilTag correspondences recover its plane, and boundary extraction yields the four image corners $\{ \mathbf { u } _ { j } ^ { \mathrm { d i s p } } \} _ { j = 1 } ^ { 4 }$ . Intersecting rays through this polygon with the tag-derived plane reconstructs the cameraframe display cloud ${ \bf \bar { \Psi } } _ { C } { \bf \Psi } _ { \mathcal { Q } }$ . LiDAR plane fitting and clustering produce candidate display clouds $\left\{ { \bf { \mathcal { L } } } { \bf { \mathcal { Q } } } _ { c } \right\}$ , whose 3D corners $\{ { ^ { L } \mathbf { b } _ { c , j } } \} _ { j = 1 } ^ { 4 }$ are paired with the image corners using the permutations in Sec. V-A.

These measurements support extrinsic estimation. Sec. V handles one pose; Sec. VI combines poses and refines the nominal tag size $s _ { 0 }$

## IV. DISPLAY OBSERVATION EXTRACTION

Fig. 3 illustrates how camera observations guide LiDAR processing. AprilTag detections recover the display plane and provide a camera-frame depth prior and pixel-to-meter scale. We use the depth prior to range-filter LiDAR points; the detected 2D display polygon, together with the scale, provides a prior on the display’s physical width and height. These complementary cues help identify plausible LiDAR display surfaces and produce more reliable calibration measurements.

## A. AprilTag-Based Display Plane Estimation

Corners of the displayed AprilTags [16] are detected using Kalibr [17], [18]. Their identities associate image corners $\mathbf { u } _ { j } ^ { \mathrm { t a g } }$ with nominal pattern coordinates $^ M { \bf q } _ { j }$ . These correspondences determine the display pose ${ \mathbf { T } } _ { M } ~ = ~ ^ { C } { \mathbf { T } } _ { M } ~ = ~ ( { \mathbf { R } } _ { M } , { \mathbf { t } } _ { M } )$ through calibrated PnP, where $\mathbf { R } _ { M } = { } ^ { C } \mathbf { R } _ { M }$ and $\mathbf { t } _ { M } = \mathbf { \epsilon } ^ { C } \mathbf { t } _ { M }$ The projection model and image points use a consistent distortion convention.

We represent the display plane by $( ^ { C } \mathbf { n } ) ^ { \top C } \mathbf { p } = ^ { C } d ,$ where $c _ { \mathbf { n } }$ is a unit normal and $\dot { c } _ { d }$ is the signed plane offset. Since the pattern lies on the $z = 0$ plane of the display frame M, its plane parameters are

$$
{ \bf \Delta } ^ { C } { \bf n } = { \bf R } _ { M } { \bf e } _ { 3 } , \qquad { \bf \Delta } ^ { C } d = \left( { \bf \Delta } ^ { C } { \bf n } \right) ^ { \top } { \bf t } _ { M } ,\tag{2}
$$

where ${ \mathbf e } _ { 3 } = [ 0 , 0 , 1 ] ^ { \top }$ . For nominal tag centers ${ { \mathbf { } } ^ { M } } { \mathbf { c } } _ { i } .$ , the finite, positive camera-axis depths $z _ { i } ~ = ~ { \bf e } _ { 3 } ^ { \top } ( { \bf R } _ { M } { } ^ { M } { \bf c } _ { i } + { \bf t } _ { M } )$ are averaged to obtain $\begin{array} { r } { \bar { z } = N _ { \mathrm { t a g } } ^ { - 1 } \sum _ { i = 1 } ^ { N _ { \mathrm { t a g } } } { z _ { i } } } \end{array}$ , where $N _ { \mathrm { t a g } }$ counts valid centers. This depth prior guides LiDAR filtering in Sec. IV-C.

Together with the image boundary extracted next, this plane defines the virtual display cloud ${ } ^ { C } \dot { \mathcal { Q } } \dot { : }$ camera rays through the detected display polygon are intersected with the plane, as detailed in Sec. IV-C.

## B. Image-Based Display Boundary Extraction

In undistorted image coordinates, the farthest tag corner in each quadrant about their centroid forms an initial quadrilateral. Invalid quadrilaterals are rejected. Each edge is searched outward by shifting its midpoint along the edge normal. Candidate segments are oriented toward the vanishing point of the corresponding opposite edges when reliable; otherwise, the initial edge is translated along its normal.

Candidates are scored by counting nearby pixels with strong Sobel gradients approximately normal to the segment. $\mathbf { A }$ minimum displacement and comparison with the unshifted score suppress the pattern boundary. After non-maximum suppression, the first retained peak restricts the search interval, within which the farthest candidate with sufficient support relative to the best score is selected.

Nearby Canny edges [19] refine the selected lines. Their intersections yield four cyclically ordered display corners $\{ \mathbf { u } _ { j } ^ { \mathrm { d i s p } } \} _ { j = 1 } ^ { 4 }$ enclosing the image region Ω. Polygons failing convexity, image-boundary, or geometric consistency checks are rejected. Perspective can break parallelism, so oppositeedge directions provide only a weak check.

## C. Depth-Guided LiDAR Display Extraction

The accumulated LiDAR points are restricted to

$$
\alpha _ { 1 } \bar { z } < \| ^ { L } \mathbf { p } \| _ { 2 } \leq \alpha _ { 2 } \bar { z } , \qquad 0 < \alpha _ { 1 } < \alpha _ { 2 } ,\tag{3}
$$

where $\alpha _ { 1 }$ and $\alpha _ { 2 }$ define a coarse range interval for compact sensor configurations. Camera-axis depth and LiDAR range are generally different quantities.

After optional voxel downsampling, iterative random sample consensus (RANSAC) [20] extracts candidate planes and removes their inliers from the residual cloud. Euclidean clustering separates disconnected regions within each plane. Components with sufficient support are retained and their planes are refitted, producing candidate display clouds $\left\{ { \bf \Psi } ^ { L } { \mathcal { Q } } _ { c } \right\}$ indexed by c.

## D. Display Geometry Reconstruction and Validation

For a pixel ${ \textbf { u } } \in \Omega .$ , its calibrated bearing vector ${ { \mathbf { \mathit { C } } } _ { \mathbf { \mathit { r } } \left( \mathbf { \mathit { u } } \right) } }$ intersects the display plane at

$$
{ } ^ { C } \mathbf { p } ( \mathbf { u } ) = \lambda ( \mathbf { u } ) ^ { C } \mathbf { r } ( \mathbf { u } ) , \qquad \lambda ( \mathbf { u } ) = \frac { { } ^ { C } d } { ( { } ^ { C } \mathbf { n } ) ^ { \top } { } ^ { C } \mathbf { r } ( \mathbf { u } ) } .\tag{4}
$$

Only finite intersections with $\lambda ( \mathbf { u } ) ~ > ~ 0$ and denominators sufficiently far from zero are retained. These intersections form the camera-frame virtual display cloud, $^ C \mathcal { Q } = \{ ^ { C } \mathbf { q } _ { i } \}$ ; applying them to the image corners recovers the display boundary.

For each LiDAR-frame candidate display cloud $^ L { \mathcal { Q } } _ { c }$ , principal component analysis (PCA) defines a local in-plane coordinate system. The farthest projected point in each quadrant about the centroid is selected as a candidate corner. These corners are ordered cyclically and mapped onto the fitted plane, yielding $\{ { ^ { L } \mathbf { b } _ { c , j } } \} _ { j = 1 } ^ { 4 }$ , where $^ L \mathbf { b } _ { c , j }$ is the j-th corner of the c-th candidate display expressed in the LiDAR frame. Candidate rectangles must have sufficient extent, approximately parallel opposite edges, and side lengths within the prescribed displaysize bounds.

The plane $( ^ { C } \mathbf { n } , ^ { C } d )$ , image corners $\{ \mathbf { u } _ { j } ^ { \mathrm { { d i s p } } } \}$ , reconstructed cloud $\bar { c } _ { \mathcal { Q } }$ , and LiDAR clouds $\{ { ^ L } { \mathcal Q } _ { c } \}$ with their corners $\{ { ^ { L } \mathbf { b } _ { c , j } } \}$ are the inputs to the subsequent extrinsic estimation in Secs. V and VI. The alternating 3D edge lengths of these corners provide candidate display width and height, which can be checked for consistency with the 2D corner spans using the pixel-to-meter scale estimated from the AprilTag, helping reject incorrect LiDAR clouds.

## V. SINGLE-POSE EXTRINSIC CALIBRATION

The rectangular display outline does not uniquely determine the corner association, and several LiDAR plane candidates may satisfy the geometric checks. We therefore generate alternative hypotheses, reject implausible solutions, and refine the remaining transformations before final selection. Fig. 4 contrasts a correct hypothesis with three incorrect cases addressed by the rejection and validation steps below. The acquisition index is omitted in this section.

## A. Geometric Hypothesis Generation

Both the 3D LiDAR candidate corners $\{ { ^ { L } \mathbf { b } _ { c , j } } \} _ { j = 1 } ^ { 4 }$ and the 2D image polygon vertices $\{ \mathbf { u } _ { j } ^ { \mathrm { d i s p } } \} _ { j = 1 } ^ { 4 }$ are cyclically ordered. However, they leave the cross-modal starting corner and traversal direction ambiguous.

We enumerate four cyclic shifts under both forward and reversed traversal, producing eight adjacency-preserving corner permutations. Each plane candidate and corner permutation defines a 3D–2D association. We use h to index the resulting hypotheses across all plane candidates; any retained PnP branches are indexed separately. Hypotheses are initialized and checked as described below.

## B. Extrinsic Initialization and Hypothesis Rejection

For each hypothesis h of the c-th display candidate, using its corner pair correspondence, the extrinsic is initialized by solving the PnP problem:

$$
\begin{array} { r l } {  { \mathbf { T } _ { h } ^ { \mathrm { P n P } } = \arg \underset { \mathbf { T } } { \operatorname* { m i n } } } } \\ & { \sum _ { j = 1 } ^ { 4 } \| \boldsymbol { \pi } _ { C } ( \mathbf { R } ^ { \textit { L } } \mathbf { b } _ { c , j } + \mathbf { t } ) - \mathbf { u } _ { \sigma ( j ) } ^ { \mathrm { d i s p } } \| _ { 2 } ^ { 2 } . } \end{array}\tag{5}
$$

Reject any hypothesis with a corner behind the camera.

After the positive-depth check, hypothesis rejection uses two criteria: the initial PnP reprojection residual and the implied LiDAR–camera separation. The dimensionless reprojection error $\epsilon _ { \mathrm { r e p } , h }$ is the root-mean-square pixel error over the four correspondences, divided by the mean image-polygon edge length <sup>¯</sup>ℓ. Hypotheses with $\epsilon _ { \mathrm { r e p } , h } > \tau _ { \mathrm { r e p } }$ are discarded, as illustrated in Fig. 4(c), where the proposed corner association yields a large initial reprojection residual.

We also require $\| \mathbf { t } _ { h } ^ { \mathrm { P n P } } \| _ { 2 } ~ \le ~ B _ { \operatorname* { m a x } }$ , where $B _ { \mathrm { m a x } }$ is a coarse sensor-separation bound supplied from the physical configuration. This rejects implausibly large baselines, such as the case in Fig. 4(d), without specifying the translation direction or an initial rotation. It can eliminate some front– back association errors, but does not guarantee rejection of every incorrect hypothesis.

![](images/5e22027963dc74b350e1d9d3770bf77f8f2540de430841223139abbc296a0c92.jpg)  
(a) Correct hypothesis

![](images/dd6ec76438c2c23452465b478f6955d71e3ee038e8d807e9ebd550a70d8b7e8a.jpg)

![](images/df6e2d57bfcfa36ce29077b2117358f66ebe3317ad4d8ccee64a2d0f6f457eb3.jpg)  
(c) Large PnP residual

(b) Scene edge misalignment  
![](images/52e9e34bddace7c804a2841dd593788af8ab6ab66f293b4bb259358d0eb63feb.jpg)  
(d) Implausible baseline  
Fig. 4. Hypothesis cases: (a) correct hypothesis, (b) scene-edge misalignment, (c) large initial PnP residual, and (d) implausible LiDAR–camera baseline.

For notational simplicity, the hypothesis index h is omitted during geometric refinement, where each retained hypothesis is processed independently. It is reinstated in scene-edge validation to compare the retained hypotheses.

## C. Geometric Refinement

Each retained transformation is refined using iterative closest point (ICP) registration of the virtual display cloud reconstructed in the camera frame $^ C \mathcal { Q }$ with its associated LiDAR cloud, denoted by $^ L \mathcal { Q }$ for the current hypothesis.

For the ICP refinement below, $m = 0 , 1 , . . .$ . denotes the iteration index, and $\mathbf { T } ^ { 0 }$ is initialized by the PnP estimate above, i.e., $\mathbf { T } ^ { 0 } = \mathbf { T } ^ { \mathrm { P n P } }$

The ICP algorithm alternates between nearest-neighbor correspondence estimation and rigid-transformation updates. At iteration $m ,$ each transformed source point ${ { \mathbf { } } ^ { C } } { \mathbf { q } } _ { i } { \mathbf { \bar { \xi } } } \in \mathbf { \Xi } ^ { C } { \mathcal { Q } }$ is matched to its nearest target point ${ \cal L } _ { \mathbf q _ { i } }$ in $^ L \mathcal { Q } .$ Matches exceeding a prescribed distance threshold are excluded, leaving valid correspondence indices $A _ { m }$ . With these correspondences fixed, the rigid increment is estimated as

$$
\begin{array} { r l } & { \Delta { \mathbf { T } } ^ { m } = \arg \underset { \Delta { \mathbf { T } } ^ { m } } { \operatorname* { m i n } } } \\ & { \qquad \quad \displaystyle \sum _ { i \in \mathcal { A } _ { m } } \left\| \Delta { \mathbf { R } } ^ { m } \left( { \mathbf { R } } ^ { m } \right) ^ { \top } \left( { ^ { C } } { \mathbf { q } } _ { i } - { \mathbf { t } } ^ { m } \right) + \Delta { \mathbf { t } } ^ { m } - { ^ { L } } { \mathbf { q } } _ { i } \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{6}
$$

Here, $\Delta { \bf T } ^ { m } = ( \Delta { \bf R } ^ { m } , \Delta { \bf t } ^ { m } )$ acts on points in the LiDAR frame. Let ${ \bf T } ^ { m } ~ = ~ ( { \bf R } ^ { m } , { \bf t } ^ { m } )$ denote the current LiDARto-camera estimate; its inverse maps camera points into the LiDAR frame. The left-multiplicative update is $( \mathbf { T } ^ { m + 1 } ) ^ { - 1 } =$ $\Delta \mathbf { T } ^ { m } ( \mathbf { T } ^ { m } ) ^ { - 1 }$ , and iteration continues until the increment is sufficiently small or the iteration limit is reached. After ICP convergence, the refined LiDAR-to-camera estimate is $\mathbf { T } ^ { \mathrm { I C P } }$

The finite display boundaries provide in-plane alignment information, whose strength depends on boundary coverage and point sampling. ICP convergence alone therefore does not establish a correct association or strong constraints in every degree of freedom. Candidates must converge with sufficient correspondence support and acceptable mean squared alignment error. Sensor-separation and positive-depth checks are repeated after refinement.

## D. Scene Edge Matching Validation

Using a common rectangular display as an example, its LiDAR cloud may align similarly well with the reconstructed camera-frame display cloud under both the true extrinsic transformation and an alternative hypothesis differing by a $1 8 0 ^ { \circ }$ rotation about the display normal. A square display introduces additional ambiguities at $9 0 ^ { \circ }$ and $2 7 0 ^ { \circ }$ . Display geometry alone may therefore be insufficient to distinguish these hypotheses. Fig. 4(a)–(b) illustrates this distinction: the correct hypothesis is consistent with the display and surrounding scene, whereas an incorrect hypothesis can retain nice display alignment while misaligning scene edges. We therefore evaluate the alignment between surrounding-scene depth edges and image edges to distinguish the remaining hypotheses.

For each geometrically valid hypothesis $h ,$ , the accumulated LiDAR cloud is projected into the image using its refined transformation $\mathbf { T } _ { h } ^ { \mathrm { { I C P } } }$ and the calibrated projection model $\pi _ { C }$ A depth image $D _ { h }$ is constructed by retaining the nearest positive camera-axis depth at each projected pixel. Depth discontinuities between valid neighboring pixels form a binary depth-edge map $E _ { h } ^ { D }$ . Boundaries caused by missing depth and edges within the display region Ω, expanded by a seven-pixel margin, are excluded. Independently, Gaussian smoothing followed by Canny detection produces the binary image-edge map $E ^ { I }$

Let $\mathcal { E } _ { h } ^ { D }$ denote the eligible depth-edge pixels in $E _ { h } ^ { D }$ . The surrounding-scene alignment cost is

$$
\begin{array} { r l r } {  { S _ { h } = \frac { 1 } { | \mathcal { E } _ { h } ^ { D } | } \sum _ { \mathbf { u } \in \mathcal { E } _ { h } ^ { D } } d ( \mathbf { u } ) , } } \\ & { } & { d ( \mathbf { u } ) = \{ \begin{array} { l l } { \displaystyle \| \mathbf { u } - \mathbf { v _ { u } } \| _ { 2 } , } & { \mathrm { i f ~ a ~ m a t c h ~ i s ~ f o u n d } , } \\ { \displaystyle \tau _ { d } , } & { \mathrm { o t h e r w i s e } . } \end{array}  } \end{array}\tag{7}
$$

Here, $\mathbf { v _ { u } }$ is the nearest image-edge pixel in $E ^ { I }$ within $\tau _ { d }$ pixels of u. Thus, $d ( \mathbf { u } )$ is the capped image-edge distance, independent of hypothesis h; h only affects the eligible depthedge set. The threshold $\tau _ { d }$ defines both the maximum nearestpoint search distance and the penalty for unmatched edge pixels; by default, it is set to 20 pixels. Therefore, lower costs indicate better alignment.

The final single-pose estimate is selected as

$$
h ^ { \ast } = \arg \operatorname* { m i n } _ { h \in \mathcal { H } } S _ { h } , \qquad \widehat { \mathbf { T } } ^ { \mathrm { S P } } = \mathbf { T } _ { h ^ { \ast } } ^ { \mathrm { I C P } } ,\tag{8}
$$

where H contains the hypotheses that pass initialization, geometric refinement, and validation checks. Fig. 5 illustrates how the surrounding-scene alignment cost distinguishes two geometrically plausible hypotheses.

![](images/65b841c965d8a7af6c671d7c00c41ceec4d9614190f6912aff2c0075f0a8d858.jpg)  
Fig. 5. Surrounding-scene depth-edge and image-edge alignment for two competing hypotheses. Hypothesis 1 has a lower alignment cost $( S _ { h } = 6 . 7 4$ pixels) than hypothesis $2 \ ( \boldsymbol { S } _ { h } = 1 1 . 2 8$ pixels), so it is selected.

## VI. JOINT MULTI-POSE EXTRINSIC CALIBRATION

Single-pose (SP) calibration estimates the extrinsic from one static acquisition. Joint multi-pose (MP) calibration combines static acquisitions of the same rigid sensor pair to refine shared extrinsics and AprilTag scale. Sensors remain stationary within each acquisition but may be repositioned between poses. Secs. IV and V provide a validated display cloud and initial proposal for every pose.

## A. Cross-Pose Extrinsic Consensus Selection

Individual proposals may be inaccurate because of imperfect display extraction, geometric ambiguity, or tag-size error. We select a consensus extrinsic initialization by comparing the proposals across poses, then jointly refine it using observations from all acquired poses.

For pose $k ,$ let $\widehat { \mathbf { T } } _ { k } = ( \widehat { \mathbf { R } } _ { k } , \widehat { \mathbf { t } } _ { k } )$ be its single-pose estimate (Equ. 8); this estimate serves as the pose’s proposal.

The j-th pose supports the i-th pose if their rotational and translational differences are both within the corresponding consistency thresholds $( i \neq j )$ . Accordingly, the support set of the i-th pose is defined as

$$
\begin{array} { r } { S _ { i } = \left\{ j \mid j \neq i \ \wedge \theta _ { i j } \leq \tau _ { R } \ \wedge \delta _ { i j } \leq \tau _ { t } \right\} ; } \\ { \theta _ { i j } = \left\| \mathrm { L o g } \left( \widehat { \mathbf { R } } _ { i } ^ { \top } \widehat { \mathbf { R } } _ { j } \right) ^ { \vee } \right\| _ { 2 } , \delta _ { i j } = \left\| \widehat { \mathbf { t } } _ { i } - \widehat { \mathbf { t } } _ { j } \right\| _ { 2 } . } \end{array}\tag{9}
$$

where $\tau _ { R }$ and $\tau _ { t }$ are the rotation and translation consistency thresholds, respectively. By default, $\tau _ { R }$ is $1 ^ { \circ }$ and $\tau _ { t }$ is $2$ cm.

If no proposal has a supporting pose, calibration terminates and reports an error asking the user to check whether the target display is fully observed by both the camera and LiDAR.

We select the proposal with the most supporters. For the same support number case, select the smallest supporting score $s _ { i }$ one:

$$
s _ { i } = \frac { 1 } { \left| S _ { i } \right| } \sum _ { j \in S _ { i } } \left[ \left( \frac { \theta _ { i j } } { \tau _ { R } } \right) ^ { 2 } + \left( \frac { \delta _ { i j } } { \tau _ { t } } \right) ^ { 2 } \right] .\tag{10}
$$

A pose does not support itself. The consensus set ${ \mathcal { C } } =$ $S _ { \mathrm { s e e d } } \cup \{ \mathrm { s e e d } \}$ is used to select the initialization, and its members need not be mutually consistent. The selected consensus extrinsic initializes joint refinement, which also uses observations from poses outside C. A pose may fail to support the consensus because the wrong hypothesis survives, even when the display itself was correctly detected. The consensus extrinsic provides an initialization that allows these outlier poses to contribute their local edge and point-plane constraints during joint optimization.

## B. Tag Scale Correction

Let $s _ { 0 }$ be the user-measured AprilTag side length. We model its correction by $\beta = \exp ( \eta ) > 0$ , initialized at $\beta = 1$ , so the corrected side length is $s = \beta s _ { 0 }$ . The displayed AprilTag retains the same physical side length across all acquisitions used in one joint calibration.

Because $s _ { 0 }$ sets the metric scale of the virtual display cloud, its measurement error scales the reconstructed points and can bias the extrinsic estimate.

The scale correction is applied to each camera-derived display point ${ \mathbf { } } C _ { \mathbf { q } _ { i } } \in { \mathbf { } } ^ { C } \mathcal { Q }$ as

$$
{ \bf \Xi } ^ { C } { \bf q } _ { i } ( \beta ) = \beta { \bf \Lambda } ^ { C } { \bf q } _ { i } ,\tag{11}
$$

This adjusts each reconstructed point’s depth along its pixel’s viewing ray while preserving the ray direction. The displayplane normal remains unchanged, and its offset scales by $\beta .$

## C. Joint Extrinsic and Scale Refinement

The proposed joint MP method estimates $\mathbf { T } = \left( \mathbf { R , t } \right)$ and shared scale $\beta$ using image-edge, point-to-plane, and planenormal terms. Using the residuals defined below, the four named objective components are

$$
\begin{array} { l } { { \displaystyle F _ { \mathrm { A l i g m e n t } } = \frac { 1 } { | \mathcal I | } \sum _ { k \in \mathbb V } \displaystyle \frac { 1 } { | \mathcal E _ { k } | } \sum _ { i \in \mathcal E _ { k } } \rho \Bigl ( ( r _ { k i } ^ { \mathrm { A l g m e n t } } ) ^ { 2 } \Bigr ) \ : , } } \\ { { \displaystyle F _ { \mathrm { f l C P } } = \frac { 1 } { | \mathcal I | } \sum _ { k \in \mathbb V } \displaystyle \frac { 1 } { | \mathcal I _ { k } | } \sum _ { j \in \mathcal A _ { k } } \rho \Bigl ( ( r _ { k j } ^ { \mathrm { L C P } } ) ^ { 2 } \Bigr ) \ : , } } \\ { { \displaystyle F _ { \mathrm { N o m e n a l } } = \frac { 1 } { | \mathcal V | } \sum _ { k \in \mathbb V } \rho \Bigl ( \| \mathbf r _ { k } ^ { \mathrm { N o m e n t } } \| _ { 2 } ^ { 2 } \Bigr ) \ : , } } \\ { { \displaystyle F _ { \mathrm { I n g S t a e P r i o n } } = \left( \frac { \beta - 1 } { \sigma _ { \beta } } \right) ^ { 2 } , } } \\ { { \displaystyle F _ { \mathrm { J o i n t e } } = F _ { \mathrm { A l i g m e n t } } = + F _ { \mathrm { I t C P } } } } \\ { { \displaystyle F _ { \mathrm { S t a r n g } } = \frac { 1 } { F _ { \mathrm { N o m e n t } } + 1 } \ : F _ { \mathrm { I n S t a r n g } } } } \end{array}\tag{12}
$$

The joint calibration result is obtained by solving

$$
\left( \widehat { \mathbf { T } } ^ { \mathrm { M P } } , \widehat { \beta } \right) = \underset { \mathbf { T } , \beta } { \arg \operatorname* { m i n } } \quad F _ { \mathrm { J o i n t } } ( \mathbf { T } , \beta ) .\tag{13}
$$

For pose k, $\mathcal { E } _ { k }$ is the set of dynamically associated depth– image edge pairs that pass the pixel-distance gate, and $\mathcal { M } _ { k }$ is the set of dynamically associated point–plane correspondences that pass the distance gate. Their cardinalities normalize the corresponding sums, and $\rho$ is the Huber robust loss [21].

All acquired poses are considered for joint refinement, regardless of membership in the consensus set. Correspondence gates select measurements consistent with the current extrinsic estimate, limiting the influence of inconsistent observations. Here, V denotes poses with sufficient valid gated correspondences to evaluate the data terms.

In the objective above, the indexed residuals with k are expanded as follows with k omitted:

$$
r _ { i } ^ { \mathrm { A l i g n m e n t } } = \frac { d ( \pi _ { K } ( \mathbf { R e } _ { i } + \mathbf { t } ) ) } { \sigma _ { \mathrm { e } } } ,\tag{14}
$$

$$
r _ { j } ^ { \mathrm { I C P } } = \frac { ( ^ { C } \mathbf { n } ) ^ { \top } ( \mathbf { R } \mathbf { p } _ { j } + \mathbf { t } ) - \beta ^ { C } d } { \sigma _ { \mathrm { I C P } } } ,\tag{15}
$$

$$
\mathbf { r } ^ { \mathrm { N o r m a l } } = \frac { \mathbf { R } ^ { L } \mathbf { n } - { C } _ { \mathbf { n } } } { \sigma _ { n } } ,\tag{16}
$$

Here, d denotes the capped image-edge distance $d _ { k }$ defined in Equ. 7 from the Canny edge map for pose $k ,$ and $\pi _ { K }$ denotes the undistorted camera projection. The samples $\mathbf { p } _ { j }$ and $\mathbf { e } _ { i }$ are expressed in the LiDAR frame, and the plane parameters are pose-specific. The vectors $c _ { \mathbf { n } }$ and $L _ { \mathbf { n } }$ are the unit displayplane normals in the camera and LiDAR frames, respectively, and $c _ { d }$ is the signed plane offset in the camera frame. The extensively-validated and fixed positive scales $\sigma _ { \mathrm { e } } , \sigma _ { n } ,$ , σ<sub>ICP</sub>, and $\sigma _ { \beta }$ normalize the four residual terms.

The four terms provide complementary constraints:

• Cross-Modal Edge Alignment: Starting from the consensus extrinsic, we project the LiDAR points into a projective depth image, detect edges across the full image area in both this depth image and the RGB image, and penalize the local distance between the two edge sets, following the construction approach in Sec. V-D.

• Point-to-Plane ICP: This term keeps transformed Li-DAR display points on the virtual plane.

• Plane Normal Difference: This term refines extrinsic orientation by aligning the transformed LiDAR normal with the camera-plane normal.

• Tag Size Prior: This term allows the tag-size scale $\beta$ to be optimized, adjusting the depth of each virtual display point along its pixel’s viewing ray, while regularizing $\beta$ toward 1.

Remark 1: Edge-inflated LiDAR points can bias single-pose scale and extrinsic estimates. Complementary static poses can reduce this bias during joint extrinsic and tag-scale refinement.

## VII. EXPERIMENTS

We evaluate ScreenCalib through both simulated and realworld experiments, which are designed to benchmark Screen-Calib’s single-pose (SP) and multi-pose (MP) modes in terms of accuracy, robustness, and practical versatility. Because simulated sequences provide exact ground-truth extrinsics and controllable scene conditions, they offer the clearest quantitative evidence of calibration accuracy. Although real-world sequences lack exact LiDAR–camera ground truth, they expose the methods to real sensor noise and enable accuracy comparisons across different LiDAR–camera configurations. Finally, tests across various display types and controlled AprilTag-size errors examine whether ScreenCalib remains effective with different displays and imperfectly measured AprilTag sizes.

TABLE II  
ROTATION AND TRANSLATION CALIBRATION ERRORS ON SIMULATED SEQUENCES  
Device 1: Ouster (noise standard deviation: 1.0 cm) + Gazebo camera
<table><tr><td>Error</td><td>Zhou et al.</td><td>Beltrán et al.</td><td>Ours (SP)</td><td>LCE- Calib</td><td>Ours (MP)</td></tr><tr><td>Rot.  $\overline { { ( ^ { \circ } ) } }$ </td><td>1.25</td><td>0.52</td><td>0.36</td><td>0.39</td><td>0.34</td></tr><tr><td>Trans. (cm)</td><td>1.35</td><td>1.71</td><td>0.66</td><td>1.45</td><td>0.40</td></tr></table>

<table><tr><td colspan="6">Device 2: Velodyne (noise standard deviation: 1.0 cm) + Gazebo camera</td></tr><tr><td>Error</td><td>Zhou et al.</td><td>Beltrán et al.</td><td>Ours (SP)</td><td>LCE- Calib</td><td>Ours (MP)</td></tr><tr><td>Rot. (°)</td><td>0.32</td><td>0.25</td><td>0.28</td><td>0.14</td><td>0.20</td></tr><tr><td>Trans. (cm)</td><td>1.81</td><td>1.83</td><td>0.51</td><td>0.32</td><td>0.27</td></tr></table>

· × denotes that the method failed to produce a valid calibration result.  
· Beltran et al. requires manually-tuned calibration-cloud bounds along all three axes.´

We compare ScreenCalib with Zhou et al. [4], Beltran et´ al. [5], two highly cited works, and LCE-Calib [6], a recent multi-pose calibration method. We use the publicly available GitHub implementations of Beltran et al. and LCE-Calib,´ and Zhou et al.’s MATLAB implementation. To ensure a fair comparison, we use the same simulated and real-world data collection environments, place the calibration boards and displays in similar configurations, and use boards with the default designs and dimensions reported in their publications.

Zhou et al., Beltran et al., and the proposed SP variant use´ a single static pose (single-shot calibration), whereas LCE-Calib and the proposed MP variant use multiple static poses (multi-shot calibration).

## A. Calibration Accuracy Benchmark on Simulated Data

The simulated environments provide ground-truth LiDAR– camera extrinsics, allowing both rotational and translational calibration errors to be evaluated. Tab. II compares the proposed single-pose (SP) and joint multi-pose (MP) variants with the baseline methods for Ouster–camera and Velodyne–camera configurations. In both configurations, noise with a standard deviation of 1.0 cm is added to the LiDAR points.

Tab. II reports SP rotation/translation errors of 0.36<sup>◦</sup>/0.66 cm (Ouster) and 0.28<sup>◦</sup>/0.51 cm (Velodyne). SP gives the lowest single-shot translation errors in both cases and the lowest Ouster rotation error; Beltran et al.´ slightly outperform it on Velodyne (0.25<sup>◦</sup> vs. 0.28<sup>◦</sup>). MP errors are 0.34<sup>◦</sup>/0.40 cm and 0.20<sup>◦</sup>/0.27 cm, respectively. MP gives the lowest translation errors in both cases and the lowest Ouster rotation error, while LCE-Calib leads on Velodyne rotation (0.14<sup>◦</sup>).

## B. Calibration Accuracy Benchmark on Real-World Data

Tab. III compares rotation (degrees) and translation (centimeters) errors for the proposed method and three baselines across real-world sequences. The three LiDAR–camera systems are shown in Fig. $6 ;$ Device 2 uses USB cameras.

Ground Truth. Real-world sequences lack ground-truth LiDAR–camera extrinsics. CAD models provide only nominal poses: manufacturing tolerances, assembly errors, or base bending may shift the actual sensor poses. We therefore do not treat CAD extrinsics as ground truth.

TABLE III  
INTER-CAMERA ROTATION AND TRANSLATION ERRORS ON REAL-WORLD SEQUENCES FROM THREE LIDAR–CAMERA DEVICES  
Device 1: Livox Avia + 2 FLIR cameras
<table><tr><td>Error</td><td></td><td>Zhou et al.</td><td>Beltrán et al.</td><td>Ours (SP)</td><td>LCE- Calib</td><td>Ours (MP)</td></tr><tr><td>Env. Clean</td><td>Rot. (°)</td><td>4.29, 7.84</td><td>X</td><td>0.81, 0.96</td><td>X</td><td>0.85, 0.99</td></tr><tr><td>Messy</td><td>Rot. (°)</td><td>3.03, 4.14</td><td>X</td><td>2.92, 1.15</td><td>×</td><td>1.22, 1.76</td></tr><tr><td>Clean</td><td>Trans. (cm)</td><td>6.35, 29.01</td><td>X</td><td>3.55, 3.95</td><td>X</td><td>1.90, 2.55</td></tr><tr><td>Messy</td><td>Trans. (cm)</td><td>11.80, 9.91</td><td>X</td><td>11.32, 4.54</td><td>X</td><td>2.12, 3.47</td></tr><tr><td>Mean</td><td>Rot. (°)</td><td>4.83</td><td>X</td><td>1.46</td><td>x</td><td>1.21</td></tr><tr><td></td><td>Trans. (cm)</td><td>14.27</td><td>×</td><td>5.84</td><td>×</td><td>2.51</td></tr></table>

Device 2: Livox Mid-360 + 2 USB cameras
<table><tr><td rowspan=2 colspan=1>Env. Error</td><td rowspan=2 colspan=3>Zhou    Beltrán    Ourset al.     et al.     (SP)</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>Calib</td><td rowspan=1 colspan=1>(MP)</td></tr><tr><td rowspan=1 colspan=1>CleanRot. (°)</td><td rowspan=1 colspan=1>4.29, 5.50</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>0.46, 0.36</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0.15, 0.45</td></tr><tr><td rowspan=1 colspan=1>Messy Rot. (°)</td><td rowspan=1 colspan=1>62.03, 51.09</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>0.34, 0.37</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>0.43, 0.14</td></tr><tr><td rowspan=2 colspan=1>CleanTrans. (cm)MessyTrans. (cm)</td><td rowspan=1 colspan=1>6.63, 1.81</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0.25, 0.42</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0.19, 0.91</td></tr><tr><td rowspan=1 colspan=1>48.91, 21.98</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>0.72, 0.56</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>0.42, 0.24</td></tr><tr><td rowspan=1 colspan=1>MeanRot. (°)</td><td rowspan=1 colspan=1>30.73</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>0.29</td></tr><tr><td rowspan=1 colspan=1>Trans. (cm)</td><td rowspan=1 colspan=1>19.83</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0.44</td></tr></table>

Device 3: Ouster 64 + 2 Hikvision cameras
<table><tr><td rowspan=2 colspan=1>Env.Error</td><td rowspan=2 colspan=3>Zhou    Beltrán    Ourset al.     et al.     (SP)</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>Calib</td><td rowspan=1 colspan=1>(MP)</td></tr><tr><td rowspan=1 colspan=1>CleanRot. (°)</td><td rowspan=1 colspan=1>3.74, 5.78</td><td rowspan=1 colspan=1>2.47, 2.56</td><td rowspan=1 colspan=1>3.36, 4.30</td><td rowspan=1 colspan=1>3.40, 3.51</td><td rowspan=1 colspan=1>3.27, 2.07</td></tr><tr><td rowspan=1 colspan=1>Messy Rot. (°)</td><td rowspan=1 colspan=1>36.60, 72.87</td><td rowspan=1 colspan=1>3.14, 2.94</td><td rowspan=1 colspan=1>1.32, 3.55</td><td rowspan=1 colspan=1>14.36, 5.12</td><td rowspan=1 colspan=1>1.84, 3.98</td></tr><tr><td rowspan=1 colspan=1>CleanTrans. (cm)</td><td rowspan=1 colspan=1>48.88, 37.64</td><td rowspan=1 colspan=1>6.93, 6.57</td><td rowspan=1 colspan=1>2.44, 3.48</td><td rowspan=1 colspan=1>1.73, 1.22</td><td rowspan=1 colspan=1>2.44, 1.88</td></tr><tr><td rowspan=1 colspan=1>MessyTrans. (cm)</td><td rowspan=1 colspan=1>13.17, 33.31</td><td rowspan=1 colspan=1>5.73, 6.67</td><td rowspan=1 colspan=1>2.58, 2.97</td><td rowspan=1 colspan=1>32.92, 5.75</td><td rowspan=1 colspan=1>3.69, 1.92</td></tr><tr><td rowspan=2 colspan=1>MeanRot. (°)Trans. (cm)</td><td rowspan=1 colspan=1>29.75</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>3.13</td><td rowspan=1 colspan=1>6.60</td><td rowspan=1 colspan=1>2.79</td></tr><tr><td rowspan=1 colspan=1>33.25</td><td rowspan=1 colspan=1>6.48</td><td rowspan=1 colspan=1>2.87</td><td rowspan=1 colspan=1>10.41</td><td rowspan=1 colspan=1>2.48</td></tr></table>

· × denotes that the method failed to produce a valid calibration result. · For each device, we collect four sequences: two in clean environments and two in cluttered (messy) environments. The reported means are the arithmetic averages over these four sequences.  
· Beltran et al. requires manually-tuned calibration-cloud bounds along all three axes.´

TABLE IV  
KALIBR INTER-CAMERA EXTRINSICS AND COMPARISON WITH CAD DESIGN (TRANSLATION IN CENTIMETERS; ROTATION IN DEGREES)
<table><tr><td rowspan="2"></td><td colspan="3">Kalibr</td><td rowspan="2">CAD baseline</td><td colspan="2" rowspan="2">Kalibr roll pitch yaw anglel</td><td rowspan="2">CAD angle</td></tr><tr><td> $t _ { x }$ </td><td> $t _ { y }$   $t _ { z }$ </td><td>baseline</td></tr><tr><td>Device 1</td><td>25.00 0.141.34</td><td></td><td>25.03</td><td>25.00</td><td>0.49-0.45 0.28</td><td>0.72</td><td>0.00</td></tr><tr><td>Device 2]</td><td>4.240.01 -0.02</td><td></td><td>4.24</td><td>4.25</td><td>0.10 0.86 -0.01 0.86</td><td></td><td>0.00</td></tr><tr><td>Device 3|</td><td>17.62-0.02 0.08</td><td></td><td>17.62</td><td>17.75</td><td>-0.52 0.14 -1.30 1.41</td><td></td><td>0.00</td></tr></table>

Following prior work [22]–[24], we use Kalibr-estimated inter-camera extrinsics [25] as the reference and assess LiDAR–camera accuracy through inter-camera error.

To assess the reliability of the Kalibr results as a reference for indirect evaluation, we compare the estimated camera baselines with the CAD values. As shown in Tab. IV, the Kalibr camera baselines differ from the CAD values by only 0.03, 0.01, and 0.13 cm for Devices 1, 2, and 3, respectively. For each method, we compose its two LiDAR–camera transformations through the shared LiDAR frame and compute the resulting inter-camera rotational and translational errors relative to the Kalibr reference.

Benchmark. According to Tab. III, neither Beltran et al.’s´ method nor LCE-Calib is applicable to the Livox Avia or Mid-360 LiDAR. In particular, Beltran et al.’s publication and re-´ leased implementation assume that “measurements distributed into different layers, as typical in mechanical devices based on rotating mirrors” [5].

Using Kalibr as the reference, our MP variant achieves the lowest mean translation error among the three baseline methods and our SP variant across all three devices (2.51, 0.44, and 2.48 cm). It also achieves the lowest mean rotation error on Devices 1 and 2 (1.21<sup>◦</sup> and 0.29<sup>◦</sup>); on Device 3, its mean rotation error (2.79<sup>◦</sup>) is nearly identical to Beltran et´ al.’s lowest result (2.78<sup>◦</sup>).

![](images/d34b1acf15433b28ae863d3c28c71d0329ec7e28c3aa9d357e98e383828b3b8b.jpg)  
Fig. 6. The three real-world LiDAR–camera systems evaluated in Tab. III: Device 1, Livox Avia with two FLIR cameras; Device 2, Livox Mid-360 with two USB cameras; and Device 3, Ouster 64 with two Hikvision cameras. Red arrows indicate the camera baselines and their CAD lengths.

Considering only the single-shot methods, our SP variant achieves the lowest mean translation error across all three devices (5.84, 0.49, and 2.87 cm) and the lowest mean rotation error on Devices 1 and 2 (1.46<sup>◦</sup> and 0.38<sup>◦</sup>). On Device 3, its mean rotation error (3.13<sup>◦</sup>) is slightly higher than Beltran et´ al.’s (2.78<sup>◦</sup>), but substantially lower than Zhou et al.’s (29.75<sup>◦</sup>).

Calibration Environment. Tab. III also shows that nearby clutter affects the baseline methods to varying degrees. For example, Zhou et al.’s method exhibits large rotation errors on the Mid-360 and Ouster systems, while LCE-Calib shows increased errors on cluttered Ouster sequences. In contrast, the proposed SP and MP variants generally maintain low errors across clean and cluttered trials, indicating greater robustness to objects surrounding the calibration display.

## C. Rendered Map Quality Benchmark on Real-World Data

This experiment compares the two variants of our method: single-pose (SP) calibration and joint multi-pose (MP) calibration. We render Device 2 data using the extrinsic parameters estimated by each variant; Fig. 7(a) shows the MP result, and Fig. 7(b) shows the SP result. The red box in Fig. 7(b) highlights a region where the SP map is slightly less well aligned, illustrating the qualitative improvement provided by MP. This observation is consistent with the lower mean rotation and translation errors achieved by MP in Tab. III.

## D. Robustness Across Programmable Displays

We test five displays (Fig. 2) using the three Device 2 trials in Tab. III; the errors in Tab. V use the Kalibr inter-camera reference (Sec. VII-B).

Tab. V shows that our method in SP mode can provide reasonable calibration results with the tablet, laptop, computer monitor, and television. However, the cell phone does not provide usable results in all three trials, likely because the LiDAR noise at its edges is significant compared with its small size. Among the displays, the television, which has the largest display area, provides the best mean calibration: 0.16<sup>◦</sup> in rotation and 0.30 cm in translation.

Displays support single-capture calibration, but small phones may be less reliable.

![](images/6b68c1ce1bb1d87a4db50c34f832c79376f92db738066117edc2132c1d2ce8f6.jpg)  
Fig. 7. Qualitative comparison of rendered maps generated using the extrinsic calibration parameters estimated by our single-pose (SP) and joint multi-pose (MP) methods: (a) MP and (b) SP. The red box in (b) marks a region with slightly poorer alignment in the SP result. Maps are rendered from real-world data collected by Device 2 in an indoor scene.

TABLE V  
SINGLE-POSE (SP) INTER-CAMERA ROTATION AND TRANSLATION ERRORS ACROSS DISPLAYS
<table><tr><td rowspan="2">Display</td><td rowspan="2">Size (cm×cm)</td><td colspan="4">Rot. (°) / Trans. (cm)</td><td rowspan="2">Mean</td></tr><tr><td>Trial 1</td><td>Trial 2</td><td></td><td>Trial 3</td></tr><tr><td>Cell phone</td><td>16×7</td><td>4.81  /  1.44</td><td>×/×</td><td></td><td>×/ ×</td><td>× ×</td></tr><tr><td>Tablet</td><td>25×18</td><td>0.12 /0.12</td><td>2.49</td><td>/3.53</td><td>1.63 / 0.61</td><td>1.41 1.42</td></tr><tr><td>Laptop</td><td>30×20</td><td>0.23 / 0.28</td><td>0.10 / 0.23</td><td></td><td>0.47 / 0.44</td><td>0.27 0.32</td></tr><tr><td>Computer monitor</td><td> $7 0 \times 5 0$ </td><td>0.46 / 0.34</td><td>0.36</td><td>/0.42</td><td>0.37 / 0.56</td><td>0.40 0.44</td></tr><tr><td>Television</td><td>168×96</td><td>0.13 / 0.45</td><td>0.27 / 0.10</td><td></td><td>0.07  / 0.35</td><td>0.16 / 0.30</td></tr></table>

× denotes that the method failed to produce a valid calibration result.

## E. Sensitivity to User-Measured AprilTag Size

We test tag-size robustness on a simulated sequence with known extrinsics and true AprilTag size. The input size ranges from −10% to +10% in 2% steps; for each MP run, we record calibration errors and the optimized $\beta .$

Fig. 8 plots (a) extrinsic rotation error, (b) translation error, and (c) estimated tag size. Plots (a) and (b) show calibration error versus input tag-size error; (c) compares estimated and true tag size. The orange boxes mark the −4% to +6% inputerror range, where $\beta$ keeps the scale close to ground truth.

Our method performs best for input tag-size errors from −4% to +6%, with mean rotation and translation errors of 0.169<sup>◦</sup> and 0.643 cm. It remains valid beyond this range, but larger tag-size errors reduce accuracy.

Plot (b) also shows that disabling $\beta$ optimization (blue curve) mainly affects the estimated extrinsic translation. This is because the tag-size scale factor $\beta$ and extrinsic translation along the camera optical axis are coupled.

## VIII. CONCLUSION

This paper presented ScreenCalib, a LiDAR–camera extrinsic calibration framework that uses an ordinary programmable display to retain controlled and repeatable calibration geometry while reducing dedicated target preparation and handling.

(a) Extrinsic Rotation Calibration Error  
![](images/368655acc26191398337c36af1606207e005336815a055efd5ab734f90c1e4fb.jpg)

(b) Extrinsic Translation Calibration Error  
![](images/bfa92a360524c1410732f0471366ae9fca10f0804fcc65c1b69d5c90d67caa63.jpg)

(c) Estimated Tag Size  
![](images/252c2a4fafa0b43f03ebb59a9e4c7125dfca073ee758bec451efabd1154704f7.jpg)  
Fig. 8. Sensitivity of the multi-pose (MP) method to AprilTag-size measurement errors: (a) extrinsic rotation error, (b) extrinsic translation error, and (c) estimated tag size, shown with and without β optimization.

Its central challenge is that the camera observes the displayed AprilTags, whereas the LiDAR measures the physical screen surface and boundary. ScreenCalib connects these different observations through shared display geometry: tagguided image-boundary extraction reconstructs the cameraframe display, while tag-derived depth and metric size cues guide LiDAR display extraction in cluttered environments. To address ambiguous corner associations, the single-pose mode retains competing hypotheses, refines them through finitesupport registration, and uses surrounding scene edges to distinguish geometrically plausible alternatives. The multi-pose mode combines independently captured static observations through cross-pose consensus and jointly refines the shared extrinsics and tag scale, reducing sensitivity to user-measured tag-size errors without requiring continuous motion or inter-

pose trajectory estimation.

Experiments in simulation and on three real LiDAR–camera systems, covering mechanical and nonrepetitive LiDARs in clean and cluttered environments, support the effectiveness of these designs. Relative to single-pose calibration, multi-pose refinement reduced mean rotation and translation errors against the Kalibr-derived inter-camera reference on all three systems and improved rendered-map alignment. The tag-size sensitivity study further demonstrated the benefit of scale correction, while tests across five display types showed both the versatility of the approach and its limitations: small displays can provide insufficient reliable boundary measurements relative to LiDAR noise. These results establish ordinary displays as a practical calibration alternative when suitable screen geometry is available, with one static placement enabling rapid calibration and additional placements improving accuracy. Future work will investigate joint intrinsic and extrinsic refinement and improved handling of sparse, noisy display boundaries.

## REFERENCES

[1] S. Vora, A. H. Lang, B. Helou, and O. Beijbom, “PointPainting: Sequential fusion for 3D object detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 4604–4612.

[2] Z. Zou, C. Zheng, C. Yuan, S. Zhou, K. Xue, and F. Zhang, “iBTC: An image-assisting binary and triangle combined descriptor for place recognition by fusing LiDAR and camera measurements,” IEEE Robot. Autom. Lett., vol. 9, no. 12, pp. 10858–10865, 2024.

[3] C. Jiang, R. Gao, K. Shao, Y. Wang, R. Xiong, and Y. Zhang, “LI-GS: Gaussian splatting with LiDAR incorporated for accurate large-scale reconstruction,” IEEE Robot. Autom. Lett., vol. 10, no. 2, pp. 1864– 1871, 2025.

[4] L. Zhou, Z. Li, and M. Kaess, “Automatic extrinsic calibration of a camera and a 3D LiDAR using line and plane correspondences,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2018, pp. 5562–5569.

[5] J. Beltran, C. Guindel, A. de la Escalera, and F. Garc´ ´ıa, “Automatic extrinsic calibration method for LiDAR and camera sensor setups,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 10, pp. 17677–17689, 2022.

[6] J. Jiao, F. Chen, H. Wei, J. Wu, and M. Liu, “LCE-Calib: Automatic LiDAR-frame/event camera extrinsic calibration with a globally optimal solution,” IEEE/ASME Trans. Mechatronics, vol. 28, no. 5, pp. 2988– 2999, 2023.

[7] C. Yuan, X. Liu, X. Hong, and F. Zhang, “Pixel-level extrinsic self calibration of high resolution LiDAR and camera in targetless environments,” IEEE Robot. Autom. Lett., vol. 6, no. 4, pp. 7517–7524, 2021.

[8] S. Chen, X. Li, S. Li, Y. Zhou, and X. Yang, “iKalibr: Unified targetless spatiotemporal calibration for resilient integrated inertial systems,” IEEE Trans. Robot., vol. 41, pp. 1618–1638, 2025.

[9] S. Verma, J. S. Berrio, S. Worrall, and E. Nebot, “Automatic extrinsic calibration between a camera and a 3D LiDAR using 3D point and plane correspondences,” in Proc. IEEE Intell. Transp. Syst. Conf. (ITSC), 2019, pp. 3906–3912.

[10] Y. Dong, Y. Liu, L. Li, H. Deng, J. Tang, and J. Li, “Automatic LiDAR– camera extrinsic calibration using pseudoimage and multiple targets,” IEEE Trans. Instrum. Meas., vol. 73, Art. no. 7005113, 2024.

[11] G. Pandey, J. R. McBride, S. Savarese, and R. M. Eustice, “Automatic targetless extrinsic calibration of a 3D LiDAR and camera by maximizing mutual information,” Proc. AAAI Conf. Artif. Intell., vol. 26, no. 1, pp. 2053–2059, 2012.

[12] X. Zhang, S. Zhu, S. Guo, J. Li, and H. Liu, “Line-based automatic extrinsic calibration of LiDAR and camera,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2021, pp. 9347–9353.

[13] D. Cattaneo and A. Valada, “CMRNext: Camera to LiDAR matching in the wild for localization and extrinsic calibration,” IEEE Trans. Robot., vol. 41, pp. 1995–2013, 2025.

[14] K. Petek, N. Vodisch, J. Meyer, D. Cattaneo, A. Valada, and W. Burgard,¨ “Automatic target-less camera-LiDAR calibration from motion and deep point correspondences,” IEEE Robot. Autom. Lett., vol. 9, no. 11, pp. 9978–9985, 2024.

[15] S. Han, X. Zhu, J. Wu, X. Cai, W. Yang, H. Yu, and G.-S. Xia, “UniCalib: Targetless LiDAR-camera calibration via probabilistic flow on unified depth representations,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2026, pp. 1906–1915.

[16] E. Olson, “AprilTag: A robust and flexible visual fiducial system,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2011, pp. 3400–3407.

[17] L. Oth, P. Furgale, L. Kneip, and R. Siegwart, “Rolling shutter camera calibration,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2013, pp. 1360–1367.

[18] J. Rehder, J. Nikolic, T. Schneider, T. Hinzmann, and R. Siegwart, “Extending Kalibr: Calibrating the extrinsics of multiple IMUs and of individual axes,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2016, pp. 4304–4311.

[19] J. Canny, “A computational approach to edge detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 8, no. 6, pp. 679–698, Nov. 1986, doi: 10.1109/TPAMI.1986.4767851.

[20] M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography,” Commun. ACM, vol. 24, no. 6, pp. 381–395, 1981.

[21] P. J. Huber, “Robust estimation of a location parameter,” Ann. Math. Statist., vol. 35, no. 1, pp. 73–101, 1964, doi: 10.1214/aoms/1177703732.

[22] L. F. T. Fu, N. Chebrolu, and M. Fallon, “Extrinsic calibration of camera to LIDAR using a differentiable checkerboard model,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2023, pp. 1825–1831.

[23] Z. Bai, G. Jiang, and A. Xu, “LiDAR-camera calibration using line correspondences,” Sensors, vol. 20, no. 21, Art. no. 6319, 2020.

[24] E. Giacomini, L. Brizi, L. Di Giammarino, O. Salem, P. Perugini, and G. Grisetti, “Ca<sup>2</sup>Lib: Simple and accurate LiDAR-RGB calibration using small common markers,” Sensors, vol. 24, no. 3, Art. no. 956, 2024.

[25] P. Furgale, J. Rehder, and R. Siegwart, “Unified temporal and spatial calibration for multi-sensor systems,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2013, pp. 1280–1286.

![](images/3de45107163695bce35e684cad9fb042a14118e4d7bad869aeea7f82f632aa03.jpg)

Zuhao Zou received the B.Eng. degree in automatic control system engineering from the University of Sheffield in 2016 and the M.Sc. degree in computer science, specializing in computer vision and SLAM, from University College London in 2017. He received the Ph.D. degree from mechanical engineering at the University of Hong Kong and works in the MaRS Lab under Prof. Fu Zhang. His research spans SLAM, sensor fusion, loop detection, and place recognition.

Xiangcheng Hu received the B.Sc. degree from the North University of China in 2017 and the M.S. degree from Beihang University in 2020. He is pursuing the Ph.D. degree in electronic and computer engineering at The Hong Kong University of Science and Technology.

Jianhao Jiao received the Ph.D. degree in robotics from The Hong Kong University of Science and Technology in 2021. He is a researcher at Lion Rock AI Lab, China Merchants Group. His research interests include autonomous navigation, multi-sensor SLAM, place recognition, and embodied intelligence.

![](images/ce2af5b44cc965d94e1391567e8ec74dd60cf6050c540732b214d119e8b4b0c3.jpg)

Jin Wu received the Ph.D. degree in electronic and computer engineering from The Hong Kong University of Science and Technology in 2024 and the bachelor’s degree in applied science from the University of Electronic Science and Technology of China. He is a Professor at the University of Science and Technology Beijing. His research interests include pose and attitude estimation, integrated navigation, sensor calibration, and multi-sensor fusion. He is an Technical Editor of IEEE/ASME TRANS-ACTIONS ON MECHATRONICS (TMECH), and an

Associate Editor of International Journal of Robotics Research (IJRR).