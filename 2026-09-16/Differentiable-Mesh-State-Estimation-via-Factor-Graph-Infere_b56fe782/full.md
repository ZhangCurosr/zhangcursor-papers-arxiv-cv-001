# Differentiable Mesh State Estimation via Factor Graph Inference for Deformable Object Reconstruction

Lidia Al-Zogbi<sup>1∗</sup>, Fangjie Li<sup>2∗</sup>, Samuel Tobin<sup>3</sup>, James Ferguson<sup>4</sup>, Nithesh Kumar<sup>4</sup>, Alejandro Chara<sup>5</sup>, Kuan-I Chung<sup>2</sup>, Mingxing Rao<sup>2</sup>, Ayberk Acar<sup>2</sup>, Susheela Sharma Stern<sup>4</sup>, Robert Webster<sup>4</sup>, Daniel Moyer<sup>2</sup>, Alan Kuntz<sup>4</sup>, Caleb Rucker<sup>3</sup>, Tucker Hermans<sup>6,7</sup> and Jie Ying Wu<sup>2</sup>

Abstract— Estimating deformable object states remains a fundamental challenge in robotics and simulation. We propose a novel factor graph-based framework for probabilistic mesh state estimation of deformable objects. The method directly updates a tetrahedral mesh—a rich and physically-grounded representation of an environment—by combining physics priors, noisy sensor measurements, and temporal smoothness constraints within a unified probabilistic formulation. The estimation problem is posed as a nonlinear least-squares optimization and solved using Levenberg–Marquardt. Ex vivo central-airway obstruction experiments and simulations on deforming cube models demonstrate reliable and accurate reconstruction under both rigid motion and deformation, highlighting the potential of this probabilistic approach for principled, measurement-driven mesh state estimation in deformable object reconstruction.

## I. INTRODUCTION

Updating time-varying tetrahedral meshes from point cloud observations is a fundamental inverse problem at the intersection of computational geometry, state estimation, and robotics, enabling compliant manipulation [1], predictive simulation [2], and real-time digital twinning [3]. Challenges include high-dimensionality of mesh states; partial observability; nonlinear measurement operators that induce nonconvex objectives; and the trade-off between measurement fidelity and physics-based regularization.

Existing approaches address subsets of these challenges but lack unified solutions. Kalman-based methods provide principled uncertainty propagation but rely on local linearization that breaks under large deformations and can scale quadratically with state size [4], [5]. Online filters process measurements incrementally but cannot exploit global temporal structure or revise past estimates to improve future predictions. Physics-based variational methods enforce material consistency but require accurate constitutive parameters, are poorly conditioned under measurement noise, and struggle with real-time use [6]. Data-driven approaches offer tractability but provide no guarantees of physical plausibility or generalization beyond training [7].

We propose a factor graph formulation [8], [9] that addresses these limitations through joint spatiotemporal optimization. Our approach treats mesh estimation as a nonlinear least-squares problem where states represent vertex positions, and factors encode physics-based priors, measurement likelihoods, and temporal consistency. This representation enables systematic multi-modal data fusion, and computational efficiency through sparse matrix structures inherent in mesh connectivity.

## II. RELATED WORK

Factor-graph state estimation provides a natural probabilistic foundation for representing robotic perception and reconstruction problems as structured inference. In this formulation, the posterior distribution over the unknown state is expressed as a product of local factors, where each factor encodes a measurement, prior, motion constraint, or physical relationship among a subset of variables. This representation has been widely adopted in robotics for simultaneous localization and mapping, visual-inertial estimation, and smoothing because it exposes the sparse structure of the estimation problem and enables efficient nonlinear least-squares optimization [10], [11]. While traditional applications often estimate robot poses, landmarks, or calibration parameters, the same formulation can be extended to richer environment states. In our work, the estimated variable is not a lowdimensional pose, but an explicit mesh representation of the deformable object or environment. This allows the posterior over the mesh state to be constructed from heterogeneous factors, including geometric measurement terms, temporal consistency, boundary conditions, and mechanics-informed priors.

Differentiable geometric optimization has similarly become an important tool for estimating 3D structure from observations. Many geometric estimation problems can be written as optimization problems over residuals defined on points, surfaces, images, meshes, or implicit fields, and can be solved using Gauss-Newton, Levenberg-Marquardt, or gradient-based optimization. General-purpose solvers such as Ceres support nonlinear least-squares formulations with automatic differentiation, while differentiable 3D libraries such as PyTorch3D provide differentiable operators for meshes, point clouds, rendering, and geometric losses [12], [13]. These tools have enabled analysis-by-synthesis and gradient based reconstruction pipelines in which the geometric state is updated directly to reduce observation error. Our formulation builds on this perspective by defining residuals that are differentiable with respect to the mesh state itself. As a result, point-to-surface measurements, temporal regularization terms, and stiffness-based priors can all contribute gradients to a common optimization problem, enabling mesh reconstruction to be treated as differentiable inference rather than as a purely procedural registration step.

Deformable environment modeling addresses the challenge of estimating objects or scenes whose geometry changes over time. Classical approaches include non-rigid registration, deformation graphs, and surface regularization methods that recover a deformation field or deformed geometry from sparse or dense observations. Coherent Point Drift formulates non-rigid point-set registration probabilistically by aligning one point set to another while enforcing coherent motion of the transformed points [14]. Embedded deformation represents shape change using a sparse graph of local transformations that can deform complex geometry through direct manipulation [15], while as-rigid-as-possible surface modeling regularizes deformation by encouraging local transformations to remain close to rigid [16]. Dense reconstruction systems such as DynamicFusion further demonstrate that non-rigid scene geometry can be reconstructed and tracked in real time from RGB-D observations by jointly estimating geometry and a deformation field [17]. These methods are highly relevant to deformable reconstruction, but they often treat deformation primarily as geometric alignment, tracking, or warping. In contrast, our work frames deformable object reconstruction as state estimation over an explicit mesh-valued environment state, allowing measurements and priors to be expressed as probabilistic factors within a unified inference problem.

Physics-informed estimation provides another important foundation for deformable reconstruction because purely geometric alignment can produce deformations that match observations locally but are physically implausible globally. Deformable modeling has long incorporated mechanical principles such as elasticity, stiffness, internal forces, damping, and constraints to describe how non-rigid bodies respond to forces and boundary conditions [18]. In computer graphics and simulation, such models have been used to produce stable and realistic deformable motion, while in robotics and perception they provide priors that constrain the space of admissible environment states. For mesh-based reconstruction, stiffness or mechanics-inspired regularization can encode the intuition that neighboring vertices, material regions, or constrained boundaries should deform in a physically consistent manner. Our approach incorporates this idea by using stiffness-informed factors as probabilistic priors over the mesh state. Rather than relying only on geometric correspondence, the estimator combines observation likelihoods with mechanical regularization and temporal consistency, producing a posterior estimate that is both measurement-driven and physically constrained. This is especially important for deformable object reconstruction during manipulation and cutting, where observations may be sparse, partial, noisy, or locally ambiguous.

## III. FACTOR GRAPH FORMULATION

## A. Baseline Deformation Model

State Representation: We represent the deformable object as a tetrahedral mesh with N vertices, where the state variables are the 3D coordinates of all mesh vertices. We assume the mesh connectivity is given a priori and does not change over time. At each time step t, the mesh state is $\mathbf { X } _ { t } \stackrel { \cdot } { = } \left[ \mathbf { x } _ { t , 1 } ^ { \top } , \ldots , \mathbf { x } _ { t , N } ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { 3 N }$ , with each vertex position $\mathbf { x } _ { t , i } \in \mathbb { R } ^ { 3 }$

Factor Graph Formulation: In a factor graph, each factor encodes a probabilistic constraint on selected state variables, represented by a residual function and weighted by its information matrix (the inverse covariance). We consider three principal factor types:

• Physics Prior Factor: Enforces physically plausible nodal displacements by using the finite element stiffness matrix of the mesh as the information matrix $\Lambda _ { t } ^ { \mathrm { p h y s } } =$ ${ \bf K } _ { t } ( { \bf X } _ { t } )$ . Note the dependence of $\mathbf { K } _ { t }$ on the current state $\mathbf { X } _ { t }$

We model temporal evolution of the mesh as a probabilistic transition between consecutive states $\mathbf { \bar { \rho } } P ( \mathbf { X } _ { t } | \mathbf { X } _ { t - 1 } ) = \mathcal { N } ( \mathbf { X } _ { t } | \hat { \mathbf { X } } _ { t } , \mathbf { K } _ { t } ^ { - 1 } )$ . The residual for the prior factor is:

$$
\mathbf { r } _ { t } ^ { \mathrm { p h y s } } ( \mathbf { X } _ { t } ) = \hat { \mathbf { X } } _ { t } - \mathbf { X } _ { t } ,\tag{1}
$$

where a deformable simulator [19] predicts $\hat { \mathbf { X } } _ { t }$ from $\mathbf { X } _ { t - 1 }$

• Point-to-Surface Measurement Factor: Penalizes error between measurements and mesh surface. We associate every measurement point $\mathbf { m } _ { t , k } \ \in \ \mathbb { R } ^ { 3 }$ with its closest point on the mesh surface at time step t, where $k \in [ 1 , K ]$ and K is the number of measurements. Let $\tau$ denote the set of surface triangles with vertex indices $( a , b , c )$ . For each measurement $\mathbf { m } _ { t , k }$ we identify the surface triangle whose projection is closest to the measurement point:

$$
a ^ { \star } , b ^ { \star } , c ^ { \star } = \operatorname * { a r g m i n } _ { ( a , b , c ) \in \mathcal { T } } \big \| \Pi _ { \bigtriangleup ( a , b , c ) } ( \mathbf { m } _ { t , k } ) - \mathbf { m } _ { t , k } \big \| ^ { 2 } ,\tag{2}
$$

where $\Pi _ { \triangle ( a , b , c ) } ( \cdot )$ is the closest-point projection operator onto the triangle with vertices $\left( \mathbf { x } _ { t , a } , \mathbf { x } _ { t , b } , \mathbf { x } _ { t , c } \right)$ This projection operator admits a closed-form solution in barycentric coordinates $( \alpha , \beta , \gamma )$ , with $\alpha + \beta + \gamma =$ $1 , \alpha , \beta , \gamma \ge 0$ obtained by standard point-to-triangle projection algorithms [20]. The k-th measurement residual is:

$$
\mathbf { r } _ { t , k } ^ { \mathrm { m e a s } } = \left( \alpha \mathbf { x } _ { t , a ^ { \star } } + \beta \mathbf { x } _ { t , b ^ { \star } } + \gamma \mathbf { x } _ { t , c ^ { \star } } \right) - \mathbf { m } _ { t , k } ,\tag{3}
$$

which can be used to form the concatenated residual vector across all measurements as $\mathbf { r } _ { t } ^ { \mathrm { m e a s } } =$ $\left[ \mathbf { r } _ { t , 1 } ^ { \mathrm { m e a s } \top } , . . . , \mathbf { r } _ { t , K } ^ { \mathrm { m e a s } \top } \right] ^ { \top } . \ \Lambda _ { t } ^ { \mathrm { m e a s } }$ denotes the information matrix encoding the sensor noise model for all measurements at time t.

When the closest triangle $( a ^ { \star } , b ^ { \star } , c ^ { \star } )$ remains fixed, the projection is differentiable with respect to its three vertex positions.

• Temporal Smoothness Factor: Connects consecutive states to encourage temporal smoothness of the mesh. The residual is thus:

$$
{ \bf r } _ { t } ^ { \mathrm { t e m p } } ( { \bf X } _ { t } ) = { \bf X } _ { t } - { \bf X } _ { t - 1 } ,\tag{4}
$$

penalizing large inter-frame displacements. The associated information matrix $\boldsymbol { \Lambda } _ { t } ^ { \mathrm { t e m p } }$ controls the strength of this smoothness prior, and is chosen as a scaled identity matrix assuming isotropic uncertainty across vertices.

## B. Optimization

The maximum a posteriori (MAP) estimation problem can be reformulated as a nonlinear least-squares objective [9] over all mesh states $\mathbf { X } = \{ \mathbf { X } _ { 0 } , \ldots , \mathbf { X } _ { t } \}$

$$
\mathbf { X } ^ { \star } = \operatorname * { a r g m i n } _ { \mathbf { X } } \sum _ { t } \left( \mathcal { L } _ { t } ^ { \mathrm { p h y s } } + \mathcal { L } _ { t } ^ { \mathrm { m e a s } } + \mathcal { L } _ { t } ^ { \mathrm { t e m p } } \right) .\tag{5}
$$

As time advances, new states $\mathbf { X } _ { t + 1 }$ and their associated factors are appended to the graph, so the optimization dimension grows while preserving past information. Each factor i contributes a quadratic cost $\begin{array} { r } { \mathscr { L } _ { i } = \frac { 1 } { 2 } \mathbf { r } _ { i } ( \mathbf { X } ) ^ { \top } \Lambda _ { i } \mathbf { r } _ { i } ( \mathbf { X } ) } \end{array}$ where $\mathbf { r } _ { i } ( \mathbf { X } )$ is the residual vector and $\Lambda _ { i }$ the associated information matrix. The loss function is solved iteratively using the damped Levenberg–Marquardt [21] algorithm.

This work represents a prototype implemented in Python 3.12, using the PyTorch library [22] for automatic differentiation and accelerated numerical computations.

## IV. EXPERIMENTS

## A. Simulation-Based

1) Ablation Studies: We performed an ablation study to evaluate the contribution of each factor type in the proposed mesh state estimation framework. The study was conducted using forward-simulated deforming-cube experiments with unit side length, including: (a) rigid-body translation, (b) deformation with an SPD stiffness prior, and (c) deformation with a non-SPD stiffness prior. For each case, the tetrahedral mesh state was reconstructed from synthetic surface measurements sampled from the deformed mesh and perturbed with Gaussian noise. The forward-simulation vertices were used as ground truth, and reconstruction accuracy was quantified using the vertex RMSE between the estimated and ground-truth mesh states. We used a cube with 1 meter side length for the simulation.

We evaluated five factor configurations: temporal smoothness only, measurement only, temporal and measurement factors, simulation and measurement factors, and the full model combining simulation, measurement, and temporal factors. The temporal factor penalizes inter-frame vertex displacement, the measurement factor penalizes point-tosurface discrepancy between noisy observations and the current mesh surface, and the simulation factor incorporates physics-based information from the forward simulator. The physics prior information matrix was obtained from the simulation stiffness matrix, the measurement factors used identity information matrices assuming independent observations, and the temporal smoothness factors used scaled identity matrices.

Errors were computed over three vertex sets: all vertices, all external vertices, and unobserved external vertices. The last metric is particularly important because it evaluates whether the estimator can infer the state of surface regions that are not directly constrained by measurements. Across the simulated cases, the overall RMSE increased for the deformable experiments, particularly when using the non-SPD prior, likely due to mesh warping and the simplified identity measurement covariance. Nevertheless, the solution remained stable across selected time steps, with relatively small reconstruction error even in poorly conditioned settings.

## B. Ex Vivo Studies

1) Clinical Motivation: Central airway obstruction (CAO) is a disorder of increasing prevalence and is associated with significant morbidity and mortality [23]. Therapeutic tools such as laser ablation and electrocautery, delivered through rigid or flexible bronchoscopes, can enable removal of obstructing tumor tissue and restoration of airway patency [23]. However, bronchoscopic intervention for CAO remains technically challenging: the operative workspace is narrow, the anatomy is deformable, visibility may be partial, and complications such as bleeding, airway perforation, hypoxia, or loss of airway control can be fatal [24], [25]. These challenges motivate robotic systems that can maintain an accurate estimate of airway and tumor geometry during tissue manipulation and cutting.

2) Experimental Setup: In this work, we use an ex vivo CAO model to evaluate whether the proposed factor graph formulation can estimate deformable tissue state from partial observations during robot-assisted intervention. The goal is not only to reconstruct visible surface geometry, but also to infer the volumetric deformation of the CAO model so that unobserved regions remain registered to the robotic system throughout deformation.

We performed the ex vivo study on a CAO phantom using a Virtuoso robotic system (Virtuoso Surgical, Inc., Nashville, TN, USA), as outlined in Fig. 1. Following our previous work [26], [27], we use sheep trachea to mimic human anatomy. We cut and insert a small piece of chicken breast into the trachea to mimic tumor that causes around 50% occlusion in the trachea. It is placed inside the airway through small incisions, secured with super glue.

During the experiment, we used the method outlined in [28] to register the segmented pre-operative CT into the robot coordinate frame, which is used to initialize the tissue simulation method. During the experiment, we manually deformed the tumor with the robotic tool. We then used a xCAT CT system (Xoran Technologies LLC, Ann Arbor, MI, USA) to obtain a snapshot ground truth shape of the deformed anatomy. We also capture the relevant input factors to the factor graph algorithm at 1 Hz. We then run the factor graph algorithm for ten time steps, and use the final step output for evaluation. These intra-operative CT volumes are registered to the pre-operative CT using fiducial markers rigidly attached to the exterior of the CAO phantom. We had 3 CAO phantom models, with 3 pushes on each phantom, giving 9 experimental trials in total. They were excluded from the MDE training data.

![](images/9f293b523493f2fa1134efb25342396c6d04bda58347ef96cb66c8b1d0288b30.jpg)  
Fig. 1. A) The Virtuoso robot system used for the ex vivo experiment. B) The sheep trachea used as CAO phantom. C) Endoscopic view of the CAO tumor, with and without tool interaction. There are additional red fiducials in the phantoms, although not used in this experiment.

3) Factors: The ex vivo factor graph used the same volumetric mesh state representation as the simulation studies, with $\mathbf { X } _ { t }$ denoting the tetrahedral CAO mesh in the robot base frame at time t. Similar to the simulation case, two primary factors were used: an XPBD-based simulation factor and a metric monocular-depth-estimation (MDE) point-cloud measurement factor.

The MDE factor incorporates a point cloud reconstructed from an MDE network model, constituting the point-tosurface measurement factor above. It is built on top of a nonmetric depth estimator, DepthAnything-v2 [29]. This depth estimator is in turn built on a DINOv2 [30] backbone and teacher model, which is a generic natural vision foundation model. DINOv2 is a Vision Transformer [31] architecture model with contrastive self-supervised pre-training tasks; DepthAnything-v2 supplements this with additional synthetic scene training on depth specific tasks and a Teacher-Student distillation. Our present metric MDE additionally fine-tunes this model in two stages, first across all weights and second across the depth prediction head. For both of these stages we train using 6000 video frames, with supervision in the form of depth information derived from CT registered to the robotic viewport, using the $\ell _ { 2 }$ Euclidean loss instead of SigLog loss (for relative non-metric depth).

We further incorporate a segmentation network to segment out the tumor only, to remove non-tumor depth predictions, such as that of the tool and the trachea. The segmentation network consists of (1) a frozen SAM2 [32] Hiera image encoder with lightweight adapters for efficient fine-tuning, (2) multi-branch convolutional feature modules that refine and unify multi-scale features extracted by the encoder, and (3) a U-Net-style [33] decoder that progressively fuses the refined multi-scale features through skip connections to generate the final segmentation. During training, only the adapters, the feature refinement modules, and the decoder are trainable, while the pretrained SAM2 encoder remains frozen.

TABLE I  
SIMULATION RESULTS (MEAN ± STANDARD DEVIATION, MM)
<table><tr><td>Factor configuration</td><td>All</td><td>External</td><td>Unobserved external</td></tr><tr><td>Temporal only</td><td> $\overline { { 5 . 5 \pm 3 . 4 } }$ </td><td> $\overline { { 5 . 3 \pm 3 . 3 } }$ </td><td> $\overline { { 5 . 3 \pm 3 . 2 } }$ </td></tr><tr><td>Measurement only</td><td> $2 8 0 \pm 2 2 0$ </td><td> $3 0 0 \pm 2 4 0$ </td><td> $3 5 0 \pm 2 8 0$ </td></tr><tr><td>Temporal + measurement</td><td> $1 2 0 \pm 9 . 0$ </td><td> $9 8 \pm 9 . 1$ </td><td> $1 1 0 \pm 1 1$ </td></tr><tr><td>Simulation + measurement</td><td> $0 . 7 3 \pm 0 . 2 2$ </td><td> $0 . 7 7 \pm 0 . 2 1$ </td><td> $0 . 6 7 \pm 0 . 2 3$ </td></tr><tr><td>Full model</td><td> $0 . 7 3 \pm 0 . 2 2$ </td><td> $0 . 7 6 \pm 0 . 2 1$ </td><td> $0 . 6 7 \pm 0 . 2 3$ </td></tr></table>

The uncertainty of the MDE factor is set empirically based on its error evaluation.

4) Quantitative Evaluation: For quantitative evaluation, the optimized mesh was sampled to produce an estimated point cloud $\hat { \mathcal { P } } _ { t } = \{ \hat { \mathbf { p } } _ { t , i } \} _ { i = 1 } ^ { N _ { t } }$ , which was compared against a reference point cloud $\mathcal { \tilde { P } } _ { t } ^ { \mathrm { r e f } } = \{ \mathbf { p } _ { t , j } ^ { \mathrm { r e f } } \} _ { j = 1 } ^ { M _ { t } }$ in the robot base frame. We first computed the one-sided nearest-neighbor error from the estimated reconstruction to the reference:

$$
d _ { \hat { \mathcal { P } }  \mathcal { P } ^ { \mathrm { r e f } } } = \frac { 1 } { N _ { t } } \sum _ { i = 1 } ^ { N _ { t } } \operatorname* { m i n } _ { j }  \hat { \mathbf { p } } _ { t , i } - \mathbf { p } _ { t , j } ^ { \mathrm { r e f } }  _ { 2 } .\tag{6}
$$

The reverse error was similarly computed as:

$$
d _ { \mathcal { P } ^ { \mathrm { r e f } }  \hat { \mathcal { P } } } = \frac { 1 } { M _ { t } } \sum _ { j = 1 } ^ { M _ { t } } \operatorname* { m i n } _ { i }  \mathbf { p } _ { t , j } ^ { \mathrm { r e f } } - \hat { \mathbf { p } } _ { t , i }  _ { 2 } .\tag{7}
$$

The symmetric Chamfer distance was then defined as:

$$
d _ { \mathrm { C D } } = \frac { 1 } { 2 } \left( d _ { \hat { \mathcal { P } } \to \mathcal { P } ^ { \mathrm { r e f } } } + d _ { \mathcal { P } ^ { \mathrm { r e f } } \to \hat { \mathcal { P } } } \right) .\tag{8}
$$

We additionally report the root-mean-square nearestneighbor error:

$$
d _ { \mathrm { R M S E } } = \sqrt { \frac { 1 } { N _ { t } } \sum _ { i = 1 } ^ { N _ { t } } \operatorname* { m i n } _ { j } \left. \hat { { \mathbf p } } _ { t , i } - { \mathbf p } _ { t , j } ^ { \mathrm { r e f } } \right. _ { 2 } ^ { 2 } } .\tag{9}
$$

To capture worst-case geometric disagreement while reducing sensitivity to isolated outliers, we also report the 95thpercentile distance:

$$
\begin{array} { r } { d _ { \mathrm { H 9 5 } } = \operatorname* { m a x } \left( Q _ { 0 . 9 5 } \left( \left\{ \operatorname* { m i n } _ { j } \left\| \hat { \mathbf { p } } _ { t , i } - \mathbf { p } _ { t , j } ^ { \mathrm { r e f } } \right\| _ { 2 } \right\} _ { i = 1 } ^ { N _ { t } } \right) , \right. } \\ { \left. Q _ { 0 . 9 5 } \left( \left\{ \operatorname* { m i n } _ { i } \left\| \mathbf { p } _ { t , j } ^ { \mathrm { r e f } } - \hat { \mathbf { p } } _ { t , i } \right\| _ { 2 } \right\} _ { j = 1 } ^ { M _ { t } } \right) \right) . } \end{array}\tag{10}
$$

where $Q _ { 0 . 9 5 } ( \cdot )$ denotes the 95th percentile. Together, these metrics quantify both average reconstruction accuracy and localized geometric disagreement between the estimated and reference point clouds.

TABLE II  
SURFACE RECONSTRUCTION ERRORS OVER THE FULL AND UNOBSERVED SURFACES (MEAN ± STANDARD DEVIATION, MM).
<table><tr><td></td><td colspan="3">Full Surface</td><td colspan="3">Unobserved Surface</td></tr><tr><td>Method</td><td> $d _ { C D }$ </td><td> $\underline { { d _ { R M S E } } }$ </td><td> $d _ { H 9 5 }$ </td><td> $d _ { C D }$ </td><td> $\underline { { d _ { R M S E } } }$ </td><td> $d _ { H 9 5 }$ </td></tr><tr><td>Simulation</td><td> $\overline { { 1 . 9 7 \pm 0 . 6 0 } }$ </td><td> $\overline { { 2 . 7 7 \pm 0 . 8 4 } }$ </td><td> $\overline { { 5 . 2 8 \pm 1 . 4 6 } }$ </td><td> $\overline { { 1 . 9 2 \pm 0 . 5 9 } }$ </td><td> $\overline { { 2 . 5 7 \pm 0 . 8 2 } }$ </td><td> $\overline { { 5 . 0 9 \pm 1 . 5 8 } }$ </td></tr><tr><td>Factor Graph</td><td> ${ \bf 1 . 6 1 \pm 0 . 6 7 }$ </td><td> ${ \bf 2 . 4 1 \pm 0 . 9 6 }$ </td><td> ${ \bf 4 . 3 4 \pm 1 . 6 3 }$ </td><td> ${ \bf 1 . 5 1 \pm 0 . 6 5 }$ </td><td> ${ \bf 2 . 2 3 \pm 0 . 7 2 }$ </td><td> ${ \bf 4 . 1 2 \pm 1 . 4 0 }$ </td></tr></table>

TABLE III

SURFACE RECONSTRUCTION ERRORS WITHIN THE MDE OBSERVABLEREGION (MEAN ± STANDARD DEVIATION, MM).
<table><tr><td>Method</td><td> $\overline { { d _ { C D } } }$ </td><td> $\_ d _ { R M S E }$ </td><td> $d _ { H 9 5 }$ </td></tr><tr><td>Simulation</td><td> $\overline { { 1 . 8 4 \pm 0 . 7 1 } }$ </td><td> $\overline { { 2 . 5 7 \pm 1 . 1 4 } }$ </td><td> $\overline { { 4 . 8 1 \pm 1 . 8 0 } }$ </td></tr><tr><td>Factor Graph</td><td> ${ \bf 1 . 6 3 \pm 0 . 7 8 }$ </td><td> $2 . 3 8 \pm 1 . 2 6$ </td><td> ${ \bf 4 . 2 6 \pm 1 . 7 6 }$ </td></tr><tr><td>MDE</td><td> $2 . 1 6 \pm 0 . 8 6$ </td><td> ${ \bf 1 . 9 4 \pm 0 . 4 1 }$ </td><td> $7 . 4 4 \pm 4 . 8 5$ </td></tr></table>

As shown in Fig. 2, we only consider the lesion surface for metric computation. This is because the lesion-trachea interface is considered to be fixed, and hence including it in the measurement will skew the error. Additionally, there is no ground truth correspondence between the sub-surface vertices and the CT scan.

To further analyze the effect of the MDE measurements on the mesh region not observable by MDE, we separate the mesh into 2 regions. We perform ray-tracing from the camera optical center to the sampled MDE point cloud. The intersection of the rays and $\hat { \mathcal { P } } _ { t }$ and $\mathcal { P } _ { t } ^ { \mathrm { r e f } }$ defines a proxy subset of the point clouds that would have been directly observable by MDE.

We also compare the outputs from factor graph, against its contributing factors, tissue simulation and MDE measurements.

## V. RESULTS AND DISCUSSION

## A. Simulation

Table I summarizes the contribution of each factor type to mesh-state reconstruction over the first 15 time steps of the deformable-cube shear experiment. The measurementonly configuration produced the largest errors, with a corresponding-vertex RMSE of $2 8 0 \pm 2 2 0$ mm over all vertices and $3 5 0 \pm 2 8 0$ mm over unobserved external vertices. Adding temporal smoothness reduced these errors to $1 2 0 \pm 9 . 0$ mm and $1 1 0 \pm 1 1$ mm, respectively. However, this configuration remained poorly constrained, and several optimization steps reached the maximum iteration limit.

The temporal-only configuration produced smaller errors of $5 . 5 \pm 3 . 4$ mm over all vertices and $5 . 3 \pm 3 . 2$ mm over unobserved external vertices. This result should be interpreted cautiously: without an absolute factor, the copyforward initialization leaves the mesh at its initial state, which remains close to the ground truth over this relatively short portion of the deformation trajectory.

Incorporating the simulation prior produced the lowest reconstruction errors. The simulation-and-measurement configuration achieved errors of 0.73±0.22 mm over all vertices, $0 . 7 7 \pm 0 . 2 1$ mm over external vertices, and $0 . 6 7 \pm 0 . 2 3$ mm over unobserved external vertices. The full model produced nearly identical errors of $0 . 7 3 \pm 0 . 2 2$ mm, $0 . 7 6 \pm 0 . 2 1$ mm, and $0 . 6 7 \pm 0 . 2 3$ mm, respectively. These results indicate that the physics-based simulation prior is the principal contributor to accurate reconstruction in this experiment, while the temporal factor provides no measurable improvement once the simulation prior is present. This effect is illustrated in Fig. 2.

![](images/0743440df3fe2a4ef7c465fb2a862151d29a8579370e04b193259f9562b37a42.jpg)  
Fig. 2. A) The side view of the factors as input to the factor graph, along with the ground truth. B) The factor graph output super-imposed.

## B. Ex Vivo Studies

Table II describes the alignment of the estimated tumor surfaces by factor graph and tissue simulation against the ground truth tumor state. The full factor graph achieves a lower Chamfer distance than tissue simulation alone. Notably, this improvement is also observed in regions not directly observed by the MDE measurement. This suggests that the tissue-mechanics prior encoded by the stiffness matrix can help propagate information from observed regions to constrain the unobserved mesh state within the factor graph framework.

Table III outlines the result for the MDE observable region. Here, factor graph displays the lowest $d _ { C D }$ and $d _ { \mathrm { H 9 5 } }$ , MDE has the lowest d<sub>RMSE</sub>. This suggests that MDE has higher local precision, whereas the lower two-way $d _ { C D }$ and $d _ { H 9 5 }$ of the factor graph indicate better coverage of the tumor surface. This may be attributed to the factor graph explicitly modeling the complete tumor mesh. Near regions viewed at grazing angles, the MDE point cloud becomes increasingly sparse over the physical surface, and small image-space or pose misalignments can correspond to larger surface displacements. Consequently, the groundtruth-to-MDE component of the bidirectional metrics can be disproportionately affected in these regions.

Taken together, these results suggest the complementary roles of the two information sources within the factor graph framework. MDE provides locally precise but spatially limited measurements, while tissue simulation provides a global tissue-mechanics prior that constrains the overall deformation state. By jointly incorporating these sources, the factor graph can leverage accurate local observations while propagating their information through the mechanical prior to improve estimation of the complete tissue state, including regions without direct MDE observations.

## VI. CONCLUSIONS

Preliminary experiments show that the proposed factor graph formulation closely tracks ground-truth deformations in simulation and ex vivo CAO experiments, highlighting its promise for mesh-based state estimation of deformable objects. Future work will incorporate sensor noise models beyond identity-weighted information matrices, extend validation to additional physical experiments, and address complex cases such as mesh topology changes during cutting [34]. By unifying measurements and physics within a probabilistic framework, this approach has the potential to transform how robots perceive, predict, and interact with deformable environments.

[1] H. Yin, A. Varava, and D. Kragic, “Modeling, learning, perception, and control methods for deformable object manipulation,” Science Robotics, vol. 6, no. 54, p. eabd8803, 2021.

[2] S. El Hadramy, N. Padoy, and S. Cotin, “Hyperu-mesh: Real-time deformation of soft-tissues across variable patient-specific parameters,” in Computational Biomechanics for Medicine workshop, pp. 129–139, Springer, 2024.

[3] H. Koo, H. Kim, H. Shin, and M. Kim, “Comprehensive review of digital twin technology for deformable objects: Integration of modeling, rendering, and simulation,” IEEE Access, 2025.

[4] N. A. Piga, U. Pattacini, and L. Natale, “A differentiable extended kalman filter for object tracking under sliding regime,” Frontiers in Robotics and AI, vol. 8, p. 686447, 2021.

[5] S. Dambreville, Y. Rathi, and A. Tannenbaum, “Tracking deformable objects with unscented kalman filtering and geometric active contours,” in 2006 American Control Conference, pp. 6–pp, IEEE, 2006.

[6] V. E. Arriola-Rios, P. Guler, F. Ficuciello, D. Kragic, B. Siciliano, and J. L. Wyatt, “Modeling of deformable objects for robotic manipulation: A tutorial and review,” Frontiers in Robotics and AI, vol. 7, p. 82, 2020.

[7] J. Gao, W. Chen, T. Xiang, A. Jacobson, M. McGuire, and S. Fidler, “Learning deformable tetrahedral meshes for 3d reconstruction,” Advances in neural information processing systems, vol. 33, pp. 9936– 9947, 2020.

[8] H.-A. Loeliger, “An introduction to factor graphs,” IEEE Signal Processing Magazine, vol. 21, no. 1, pp. 28–41, 2004.

[9] F. Dellaert, “Factor graphs and gtsam: A hands-on introduction,” Georgia Institute of Technology, Tech. Rep, vol. 2, no. 4, 2012.

[10] F. Dellaert and M. Kaess, “Factor graphs for robot perception,” Foundations and Trends in Robotics, vol. 6, no. 1–2, pp. 1–139, 2017.

[11] F. Dellaert, “Factor graphs and gtsam: A hands-on introduction,” Tech. Rep. GT-RIM-CP&R-2012-002, Georgia Institute of Technology, 2012.

[12] S. Agarwal, K. Mierle, and T. C. S. Team, “Ceres Solver,” Oct. 2023.

[13] N. Ravi, J. Reizenstein, D. Novotny, T. Gordon, W.-Y. Lo, J. Johnson, and G. Gkioxari, “Accelerating 3d deep learning with pytorch3d,” in Proceedings of the Fourth International Workshop on Multimedia Content Analysis in Sports, pp. 1–4, Association for Computing Machinery, 2020.

[14] A. Myronenko and X. Song, “Point set registration: Coherent point drift,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 12, pp. 2262–2275, 2010.

[15] R. W. Sumner, J. Schmid, and M. Pauly, “Embedded deformation for shape manipulation,” ACM Transactions on Graphics, vol. 26, no. 3, p. 80, 2007.

[16] O. Sorkine and M. Alexa, “As-rigid-as-possible surface modeling,” in Proceedings of the Eurographics Symposium on Geometry Processing, pp. 109–116, 2007.

[17] R. A. Newcombe, D. Fox, and S. M. Seitz, “Dynamicfusion: Reconstruction and tracking of non-rigid scenes in real-time,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 343–352, 2015.

[18] D. Baraff and A. Witkin, “Large steps in cloth simulation,” in Proceedings of the 25th Annual Conference on Computer Graphics and Interactive Techniques, pp. 43–54, Association for Computing Machinery, 1998.

[19] S. Tobin, Y. Wu, F. Liu, and C. Rucker, “Efficient steady-state tissue simulation for surgical robotics: A first-order position-based method,” in 2025 International Symposium on Medical Robotics (ISMR), pp. 211–217, IEEE, 2025.

[20] W. Heidrich, “Computing the barycentric coordinates of a projected point,” Journal of Graphics Tools, vol. 10, no. 3, pp. 9–12, 2005.

[21] J. J. Moré, “The levenberg-marquardt algorithm: implementation and theory,” in Numerical analysis: proceedings of the biennial Conference held at Dundee, June 28–July 1, 1977, pp. 105–116, Springer, 2006.

[22] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[23] A. Ernst, D. Feller-Kopman, H. D. Becker, and A. C. Mehta, “Central airway obstruction,” American journal of respiratory and critical care medicine, vol. 169, no. 12, pp. 1278–1297, 2004.

[24] D. L. Stahl, K. M. Richard, and T. J. Papadimos, “Complications of bronchoscopy: A concise synopsis,” International journal of critical illness and injury science, vol. 5, no. 3, p. 189, 2015.

[25] J. B. Gafford, S. Webster, N. Dillon, E. Blum, R. Hendrick, F. Maldonado, E. A. Gillaspie, O. B. Rickman, S. D. Herrell, and R. J. Webster III, “A concentric tube robot system for rigid bronchoscopy: A feasibility study on central airway obstruction removal: Gafford et al.,” Annals of biomedical engineering, vol. 48, no. 1, pp. 181–191, 2020.

[26] A. Acar, M. Smith, L. Al-Zogbi, T. Watts, F. Li, H. Li, N. Yilmaz, P. M. Scheikl, J. F. d’Almeida, S. Sharma, et al., “From monocular vision to autonomous action: Guiding tumor resection via 3d reconstruction,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 21714–21720, IEEE, 2025.

[27] H. Li, J. Wang, N. Kumar, J. d’Almeida, D. Lu, A. Acar, J. Han, Q. Yang, T. E. Ertop, J. Y. Wu, et al., “Automated segmentation of central airway obstruction from endoscopic video stream with deep learning,” in Medical Imaging 2025: Image-Guided Procedures, Robotic Interventions, and Modeling, vol. 13408, pp. 113–119, SPIE, 2025.

[28] A. Acar, F. Li, S. S. Stern, L. Al-Zogbi, H. Li, K. J. Oguine, D. Isik, B. Burkhart, J. F. d’Almeida, R. J. Webster, et al., “Perseus: perception with semantic endoscopic understanding and slam,” International Journal of Computer Assisted Radiology and Surgery, pp. 1–10, 2026.

[29] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Advances in neural information processing systems, vol. 37, pp. 21875–21911, 2024.

[30] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[31] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[32] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, et al., “Sam 2: Segment anything in images and videos, 2024,” arXiv preprint arXiv:2408.00714, vol. 3, 2024.

[33] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention, pp. 234–241, Springer, 2015.

[34] E. Heiden, M. Macklin, Y. Narang, D. Fox, A. Garg, and F. Ramos, “Disect: A differentiable simulation engine for autonomous robotic cutting,” arXiv preprint arXiv:2105.12244, 2021.