# LagrangeGS: Non-Conservative Lagrangian System on Dynamic 3D Gaussian Splatting

Shogo Sato shg.sato@ntt.com

Takuhiro Kaneko takuhiro.kaneko@ntt.com

Human Informatics Laboratories NTT, Inc.   
Kanagawa, Japan

0Shoichiro Takeda <sup>2</sup>shoichiro.takeda@ntt.com

<sup>g</sup>Tomoyasu Shimada <sup>u</sup><sub>tomoyasu.shimada@ntt.com</sub>

A<sub>Riku</sub> <sub>Inoue</sub>

<sup>4</sup>riku.inoue@ntt.com

<sup>2</sup>Kazuhiko Murasaki

]kazuhiko.murasaki@ntt.com

V<sub>Ryuichi</sub> <sub>Tanida</sub> C<sub>ryuichi.tanida@ntt.com</sub>

![](images/c0432423adfc6652d44241538c7650359d933ebc2c293d373f1fac674c770a76.jpg)

![](images/9c4bd86e3b341773563b4a360679d73fed9c614a4c4a5f53ea6d261f6e873b2c.jpg)  
<sup>a</sup>Figure 1: LagrangeGS formulates dynamic scenes as a non-conservative Lagrangian system parameterized by potential energy U, initial velocity v<sub>0</sub>, and non-conservative forces Q, enabling (a) long-term extrapolation, (b) time reversal, and (c) physics-based editing.

## Abstract

Dynamic 3D Gaussian Splatting (3DGS) achieves photorealistic reconstruction of time-varying scenes, and recent physics-aware extensions improve extrapolation by explicitly predicting velocity fields. However, these extensions merely fit vector fields to visual deformations without satisfying Lagrangian mechanics, leading to three major issues: (i) physically inconsistent trajectories, (ii) lack of time-reversibility, and (iii) geometric collapse during long-term extrapolation. In this paper, we propose LagrangeGS, which formulates dynamic 3DGS as a non-conservative Lagrangian system. While this Lagrangian formulation fundamentally solves (i), a direct application of general LNNs to dynamic 3DGS requires a large velocity-Hessian inversion for millions of Gaussian particles. To overcome this computational bottleneck, we approximate the velocity-Hessian as an identity matrix, decoupling particle dynamics for computational tractability. For (ii), we restrict the non-conservative forces to be explicitly time independent, enabling consistent backward integration. Finally, to address (iii), we introduce local rigid alignment that regularizes particle trajectories. Extensive evaluations on dynamic scene benchmarks demonstrate that LagrangeGS enables stable long-term extrapolation, consistent time reversal, and counterfactual physics-based editing without retraining.

## 1 Introduction

Physical modeling of dynamic 3D scenes is a key enabler for downstream applications such as robot motion planning and embodied artificial intelligence (AI). These advanced applications demand not only visual fidelity but also physics manipulations: recovering past trajectories from current states (time reversal) and simulating "what-if" scenarios under altered environments (counterfactual physics-based editing). The emergence of Neural Radiance Fields (NeRF) [16] and 3D Gaussian Splatting (3DGS) [7] has driven rapid progress in static 3D scene reconstruction. While their dynamic extensions successfully render timevarying scenes within the training window (interpolation) [13, 17, 22, 25], they operate in a purely visual domain without physics priors. Consequently, performance beyond the training window (extrapolation) is limited, and physics manipulations remain ill-defined. To bridge this gap, recent physics-aware extensions explicitly predict velocity [10] or acceleration [9] fields. However, these extensions merely fit vector fields to visual deformations without satisfying Lagrangian mechanics. This fundamental limitation leads to three major issues: (i) physically inconsistent trajectories, (ii) lack of time-reversibility, and (iii) geometric collapse during long-term extrapolation.

To address these issues, we propose LagrangeGS, a Lagrangian Neural Network (LNN) [2] based framework that formulates dynamic 3DGS as a non-conservative Lagrangian system explicitly parameterized by potential energy, initial velocity, and non-conservative forces. This Lagrangian formulation fundamentally solves (i) and formally defines physics manipulations (Figure 1). However, the direct application of conventional LNN [23] to dynamic 3DGS requires an intractable velocity-Hessian inversion for calculating the velocity of millions of Gaussian particles during forward integration. To overcome this computational bottleneck, we approximate the velocity-Hessian as an identity matrix to avoid computing its inverse. This operation physically corresponds to assigning a uniform unit mass to all Gaussian particles and decoupling their inertial dynamics. To address (ii), we restrict the non-conservative forces to be explicitly time independent, enabling consistent time reversal. We refer to this architecture, which combines the decoupled particle dynamics and the time-independent system, as the Time-independent Decoupled LNN (TD-LNN). Finally, to address (iii), we propose a local rigid alignment (LRA), which binds the decoupled particles into locally rigid structures via a Kabsch solver [6, 21].

We validate LagrangeGS on the Dynamic Object [8] and Dynamic Indoor Scene [8] benchmarks across short-term extrapolation, long-term extrapolation, time reversal, and counterfactual physics-based editing. Our contributions are summarized as follows:

• We propose LagrangeGS, which formulates dynamic 3DGS as a non-conservative Lagrangian system. This Lagrangian formulation fundamentally solves physically inconsistent trajectories and defines physics manipulations.

• We introduce two key architectural designs: (1) TD-LNN, which approximates the velocity-Hessian as an identity matrix for computational tractability and employs a time-independent system for consistent time reversal; and (2) LRA, which binds the particles into locally rigid structures to prevent geometric collapse.

• We demonstrate that LagrangeGS enables stable long-term extrapolation, consistent time reversal, and counterfactual physics-based editing without retraining.

## 2 Related Work

Dynamic 3D Scene Representation Various models have been proposed for dynamic 3D scene representation, ranging from NeRF-based models [13, 17] to 3DGS-based models [22]. DefGS [25] introduces Gaussian deformation networks to handle scene dynamics, yet they operate in a purely visual domain without physics priors, leading to limited extrapolation performance. To enable extrapolation, recent approaches introduce physics priors into dynamic 3DGS through two main strategies: vector fields and external simulation. Models adopting the first strategy predict velocity fields in NVFi [8] and FreeGave [10], or acceleration fields in TRACE [9]. Other methods apply continuous-time regularization via Neural ODEs [1] (ParticleGS [19]) or Hamiltonian formulations [4] (NeHaD [18]). However, these models impose physical constraints by merely fitting vector fields to visual deformations. Since they do not satisfy Lagrangian mechanics, they suffer from physically inconsistent trajectories and geometric collapse during long-term extrapolation as well as provide no theoretical foundation for physics manipulations. The second strategy integrates the Material Point Method (MPM) [5, 20] into the optimization loop [11], as seen in PhysGaussian [24] and Physics3D [14]. While these simulation-based methods satisfy Lagrangian mechanics, embedding an MPM solver within the optimization loop is computationally expensive, and often leads to vanishing or exploding gradients. In contrast, LagrangeGS requires no external simulator, yet satisfies Lagrangian mechanics.

Neural Dynamics Representations Modeling continuous state evolution via neural differential equations, such as Neural ODEs [1], has gained significant traction. HNNs [4] embed symplectic structures to conserve total energy, yet they require canonical coordinates that are often difficult to rigorously define for complex 3D scenes. LNNs [2] relax this requirement by learning the Lagrangian $\mathcal { L } ( q , \dot { q } )$ directly in generalized coordinates, though vanilla LNNs are strictly limited to conservative systems. To accommodate non-conservative behaviors, models like Deep Lagrangian Networks (DeLaN) [15] and the Generalised LNN (G-LNN) [23] introduce explicit external forces to Lagrangian mechanics. Crucially, G-LNN’s standard formulation introduces two fatal flaws for 3DGS applications: first, it necessitates the inversion of a velocity-Hessian matrix $\partial ^ { 2 } \mathcal { L } / \partial \dot { q } ^ { 2 }$ at each integration step, making it computationally intractable for the millions of Gaussian particles of a standard dynamic 3DGS scene. Second, its explicit time-dependent force $F ( q , { \dot { q } } , t )$ , making consistent time reversal mathematically ill-defined. In our model, we approximate the velocity-Hessian as an identity matrix to avoid the intractable computation. Additionally, we restrict the non-conservative forces to be explicitly time independent $Q ( q , \dot { q } )$ , enabling exact backward integration.

## 3 Methods

## 3.1 Preliminary

Task formulation. The goal of dynamic 3D scene rendering is to synthesize photorealistic images of a time-varying scene from arbitrary novel viewpoints. The input consists of a set of RGB video frames $\mathsf { \bar { \{ } }  I _ { \tau } ^ { 1 } , I _ { \tau } ^ { 2 } , \ldots , I _ { \tau } ^ { N } \}$ captured by N synchronized cameras at discrete time steps such as $\tau = 0 , 0 . 0 1 , \ldots , \tau _ { \mathrm { m a x } } \in \mathbb { R } _ { + }$ , where $\tau _ { \mathrm { m a x } }$ denotes the maximum time step of the observed sequence. The system aims to render images at an arbitrary target time $t ^ { \prime } \in \mathbb { R }$ and novel viewpoint $\nu ^ { \prime }$ . Conventional benchmarks primarily evaluate temporal interpolation $( t ^ { \prime } \in [ 0 , \tau _ { \operatorname* { m a x } } ] )$ and short-term extrapolation $( t ^ { \prime } \in ( \tau _ { \operatorname* { m a x } } , \tau _ { \operatorname* { m a x } } + \Delta ]$ with a small $\Delta )$ . In this work, we additionally formulate three challenging physics manipulations: (i) long-term extrapolation $( t ^ { \prime } \in ( \tau _ { \operatorname* { m a x } } , 3 \tau _ { \operatorname* { m a x } } ] )$ , (ii) time reversal $( t ^ { \prime } \in [ - \tau _ { \operatorname* { m a x } } , 0 ) )$ , and (iii) counterfactual physics-based editing, where the underlying dynamics are explicitly modified by scaling physical parameters such as potential energy, initial velocity, and non-conservative forces.

3D Gaussian Splatting [7]. Following vanilla 3DGS, a static scene at the initial time $\tau = 0$ is represented as a set of 3D Gaussian particles, $\mathcal { G } _ { 0 } = \{ g _ { 0 } ^ { 1 } , g _ { 0 } ^ { 2 } , \ldots , g _ { 0 } ^ { M } \}$ . Each Gaussian $g _ { 0 } ^ { j }$ is parameterized by its mean position $q _ { 0 } ^ { j } \in \mathbb { R } ^ { 3 }$ , a covariance matrix derived from a quaternion $r _ { 0 } ^ { j } \in \mathbb { R } ^ { 4 }$ , a scaling factor $s _ { 0 } ^ { j } \in \mathbb { R } ^ { 3 }$ , an opacity $\sigma _ { 0 } ^ { j } \in \mathbb { R }$ , and a color coefficient $c _ { 0 } ^ { j } \in \mathbb { R } ^ { d _ { c } }$ , where $d _ { c }$ is the dimension of the color space. The Gaussian set is optimized to reconstruct the initial frames $\{ I _ { 0 } ^ { 1 } , I _ { 0 } ^ { 2 } , \ldots , I _ { 0 } ^ { N } \}$ by minimizing the photometric loss with the Structural Similarity Index Measure (SSIM):

$$
\mathcal { L } _ { \mathrm { i m g } } = \sum _ { i = 1 } ^ { N } ( 1 - \lambda ) \Vert \hat { I } _ { 0 } ^ { i } - I _ { 0 } ^ { i } \Vert _ { 1 } + \lambda \left( 1 - \mathrm { S S I M } ( \hat { I } _ { 0 } ^ { i } , I _ { 0 } ^ { i } ) \right) ,\tag{1}
$$

where $\hat { I } _ { 0 } ^ { i }$ and $I _ { 0 } ^ { i }$ are the rendered image from the i-th camera at the initial frame and its corresponding ground truth.

Deformation GS (DefGS) [25]. To model scene dynamics, a deformation network $f _ { d }$ is learned to predict deviations from the initial state $( \tau = 0 )$ . Specifically, given a time $t ^ { \prime } .$ , the network predicts the position displacement $\delta q _ { t ^ { \prime } } ^ { j }$ , quaternion displacement $\delta r _ { t ^ { \prime } } ^ { j }$ , and scaling factor displacement $\delta s _ { t ^ { \prime } } ^ { j }$ for each Gaussian particle, yielding the deformed state directly

$$
q _ { t ^ { \prime } } ^ { j } = q _ { 0 } ^ { j } + \delta q _ { t ^ { \prime } } ^ { j } , \quad r _ { t ^ { \prime } } ^ { j } = r _ { 0 } ^ { j } + \delta r _ { t ^ { \prime } } ^ { j } , \quad s _ { t ^ { \prime } } ^ { j } = s _ { 0 } ^ { j } + \delta s _ { t ^ { \prime } } ^ { j } ,\tag{2}
$$

while opacity $\sigma _ { t ^ { \prime } } ^ { j }$ and color $c _ { t ^ { \prime } } ^ { j }$ are typically kept time-invariant $( \sigma _ { t ^ { \prime } } ^ { j } = \sigma _ { 0 } ^ { j } , c _ { t ^ { \prime } } ^ { j } = c _ { 0 } ^ { j } )$ . The deformation network $f _ { d }$ is also trained with the photometric loss for all time frames. Although DefGS reconstructs the observed motion faithfully, it relies solely on fitting visual deformations without physics priors, leading to limited extrapolation performance and ill-defined physics manipulations.

Generalized Lagrangian Neural Networks (G-LNN) [23]. To satisfy Lagrangian mechanics, we apply LNN to dynamic 3DGS. Given M particles with generalized coordinates $q = [ q _ { t } ^ { 1 } , q _ { t } ^ { 2 } , \ldots , q _ { t } ^ { M } ] ^ { T } \in \mathbb { R } ^ { 3 M }$ and generalized velocities $\dot { q } = [ \dot { q } _ { t } ^ { 1 } , \dot { q } _ { t } ^ { 2 } , \hdots , \dot { q } _ { t } ^ { M } ] ^ { T } \in \mathbb { R } ^ { 3 M }$ at an arbitrary continuous time $t \in \mathbb { R }$ , a standard LNN learns a Lagrangian $\mathcal { L } ( q , \dot { q } )$

![](images/02ad7197b363b99d01c95ea212d80fdcc145a1e3a88088fd5e9c0220efa67ef5.jpg)  
Figure 2: Overview of LagrangeGS. (i) Gaussian particles are partitioned into distinct objects via pre-trained model trajectory. (ii) TD-LNN computes per-particle accelerations $\ddot { q } = - \nabla _ { q } U + Q$ , combining the decoupled particle dynamics for avoiding intractable computation and the time-independent system for exact backward integration. (iii) To prevent geometric collapse, LRA binds the decoupled particles into locally rigid structures.

$$
\mathcal { L } ( q , \dot { q } ) = T ( q , \dot { q } ) - U ( q ) ,\tag{3}
$$

where T and U denote the kinetic and potential energies, respectively. The system state evolves through the Euler-Lagrange equation:

$$
{ \frac { d } { d t } } \left( { \frac { \partial { \mathcal { L } } } { \partial { \dot { q } } } } \right) - { \frac { \partial { \mathcal { L } } } { \partial q } } = 0 ,\tag{4}
$$

which enforces conservative dynamics. The G-LNN introduces a learned non-conservative force $F ( q , { \dot { q } } , t )$ to the right-hand side to accommodate dissipation and external interactions:

$$
{ \frac { d } { d t } } \left( { \frac { \partial { \mathcal { L } } } { \partial { \dot { q } } } } \right) - { \frac { \partial { \mathcal { L } } } { \partial q } } = F ( q , { \dot { q } } , t ) .\tag{5}
$$

Expanding the total time derivative using the chain rule yields $\begin{array} { r } { \frac { \partial ^ { 2 } \mathcal { L } } { \partial \dot { q } ^ { 2 } } \ddot { q } + \frac { \partial ^ { 2 } \mathcal { L } } { \partial \dot { q } \partial q } \dot { q } - \frac { \partial \mathcal { L } } { \partial q } = F ( q , \dot { q } , t ) } \end{array}$ Solving for the physical acceleration $\ddot { q } \in  { \mathbb { R } } ^ { 3 M }$ requires isolating it:

$$
\ddot { q } = \left( \frac { \partial ^ { 2 } \mathcal { L } } { \partial \dot { q } ^ { 2 } } \right) ^ { - 1 } \left[ \frac { \partial \mathcal { L } } { \partial q } - \frac { \partial ^ { 2 } \mathcal { L } } { \partial \dot { q } \partial q } \dot { q } + F ( q , \dot { q } , t ) \right] .\tag{6}
$$

For a general LNN, the kinetic term $T ( \mathbf { q } , { \dot { \mathbf { q } } } )$ may couple the velocities of different degrees of freedom. Consequently, the velocity-Hessian $\partial ^ { 2 } L / \partial \dot { \bf q } ^ { 2 }$ is not necessarily diagonal even when Cartesian coordinates are used. For M Gaussian particles, directly evaluating Eq. (6) may therefore require solving a coupled 3M-dimensional system at every integration step, which is impractical for dynamic 3DGS containing millions of particles.

## 3.2 LagrangeGS

Overall architecture. The overall architecture of LagrangeGS is illustrated in Figure 2. Starting from an initial 3D Gaussian representation $\mathcal { G } _ { 0 }$ at $\tau = 0$ , we (i) extract the per-particle trajectories $\{ q _ { \tau } ^ { j } \}$ from the pre-trained $f _ { d }$ and partition Gaussian particles into objects based on the trajectories, (ii) distill the per-object trajectories into an TD-LNN with potential energy, initial velocity, and non-conservative forces, and (iii) align the candidate positions predicted by the TD-LNN with LRA. Finally, we refine both the TD-LNN parameters and the Gaussian parameters with the photometric loss. At inference time, per-Gaussian rotations are derived from the Kabsch rotations, while scaling, opacity, and color remain time-invariant.

Time-independent Decoupled LNN (TD-LNN). To avoid an intractable velocity-Hessian inversion for millions of Gaussian particles, we assign a uniform unit mass to all Gaussian particles and decouple their inertial dynamics to simplify the kinetic energy as $T ( \dot { q } ) =$ $\textstyle { \frac { 1 } { 2 } } | | { \dot { q } } | | ^ { 2 }$ . Substituting this prior into the Lagrangian definition (Eq. 3), the momentum term simplifies to $\begin{array} { r } { \frac { \partial \mathcal { L } } { \partial \dot { q } } = \dot { q } } \end{array}$ . Consequently, the velocity-Hessian $\textstyle { \frac { \partial ^ { 2 } { \mathcal { L } } } { \partial { \dot { q } } ^ { 2 } } }$ rigorously reduces to the identity matrix I, and the cross-Hessian $\begin{array} { r } { \frac { \partial ^ { 2 } \mathcal { L } } { \partial \dot { q } \partial q } = 0 } \end{array}$ . Furthermore, the potential term derivative becomes $\begin{array} { r } { \frac { \partial \mathcal { L } } { \partial \boldsymbol { q } } = - \nabla _ { \boldsymbol { q } } U ( \boldsymbol { q } ) } \end{array}$ . Unlike standard G-LNNs, we restrict our non-conservative force to be explicitly time independent, such that $F ( q , \dot { q } , t ) = Q ( q , \dot { q } )$ . By substituting these reduced terms back into Eq. 6, the acceleration for each particle j transparently reduces to a closed form:

$$
\ddot { q } = - \nabla _ { q } U ( q ) + Q ( q , \dot { q } ) .\tag{7}
$$

This formulation depends only on a scalar potential $U : \mathbb { R } ^ { 3 M } \xrightarrow { } \mathbb { R }$ and a time-independent non-conservative force $Q : \mathbb { R } ^ { 3 \bar { M } } \times \mathbb { R } ^ { 3 M } \to \mathbb { R } ^ { \bar { 3 } M }$ . Crucially, it entirely bypasses the intractable full-system matrix inversion. Both $U$ and $Q$ are parameterized by MLPs [2]; the last layer of $Q$ is zero-initialized so training starts from a purely conservative baseline $( Q \equiv 0 )$ and learns the forcing correction incrementally. A single $U$ is shared across all particles within an object. This structural choice allows the TD-LNN to easily scale to millions of Gaussian particles. Each particle’s state is time-integrated independently with the semi-implicit Euler scheme:

$$
\dot { q } _ { \tau + 1 } ^ { j } = \dot { q } _ { \tau } ^ { j } + \ddot { q } _ { \tau } ^ { j } \Delta \tau , \qquad q _ { \tau + 1 } ^ { j } = q _ { \tau } ^ { j } + \dot { q } _ { \tau + 1 } ^ { j } \Delta \tau .\tag{8}
$$

Because $Q ( q , \dot { q } )$ has no explicit time dependence, exact backward integration $( q _ { 0 } ^ { j } , \dot { q } _ { 0 } ^ { j } ) \mapsto$ $( q _ { - \tau _ { \mathrm { m a x } } } ^ { j } , \dot { q } _ { - \tau _ { \mathrm { m a x } } } ^ { j } )$ becomes mathematically well-posed.

Local Rigid Alignment (LRA). To prevent the geometric collapse during long-term extrapolation and compensate for the decoupled dynamics, we introduce a LRA that binds the decoupled particles into locally rigid structure. First, we define a local rigid support $\boldsymbol { S } _ { j }$ for each Gaussian particle $j$ using its k-nearest neighbors in an initial state $( \tau = 0 )$ . At rollout time from $\tau \textrm { t o } \tau + 1$ , we obtain the intermediate candidate positions $\{ \tilde { q } _ { \tau + 1 } ^ { j } \}$ from the TD-LNN. For each particle $j ,$ we structurally align these candidates onto a locally rigid motion of its support at time τ via a differentiable Kabsch fit:

$$
\big ( R _ { \tau + 1 } ^ { j } , c _ { \tau + 1 } ^ { j } \big ) = \operatorname * { a r g m i n } _ { R \in S O ( 3 ) , c \in \mathbb { R } ^ { 3 } } \sum _ { l \in \mathcal { S } _ { j } } \big \| R \big ( q _ { \tau } ^ { l } - \bar { q } _ { \tau , S j } \big ) + c - \tilde { q } _ { \tau + 1 } ^ { l } \big \| ^ { 2 } ,\tag{9}
$$

where $\bar { q } _ { \tau , S _ { j } }$ is the centroid of the support at time τ. The finalized physical position for particle j at time $\tau + 1$ is explicitly given by $q _ { \tau + 1 } ^ { j } = R _ { \tau + 1 } ^ { j } \left( q _ { \tau } ^ { j } - \bar { q } _ { \tau , S _ { j } } \right) + c _ { \tau + 1 } ^ { j }$ . To maintain visual consistency, its rotation is updated by extracting the unit quaternion from $R _ { \tau + 1 } ^ { j }$ and applying it to the rotation $r _ { \tau + 1 } ^ { j }$ . Because the Singular Value Decomposition (SVD)-based Kabsch solver is fully differentiable, gradients from the photometric loss can propagate to the TD-LNN-integrated candidate positions of the entire support. This local alignment prevents geometric collapse while smoothly accommodating global continuous deformations.

![](images/8c1bfd388fb99abf686e3f3ed6ac46b951ac67dd2313fa1862c4869742e8422f.jpg)  
Figure 3: Overview of the physics manipulations. (i) Extrapolation: future states $( 0 < \tau )$ are computed by forward integration. (ii) Time reversal: Past states $( \tau < 0 )$ are recovered by backward integration using a negative time step $- \Delta \tau$ . (iii) Counterfactual physics-based editing: Dynamic properties are manipulated by scaling the initial velocity (α), learned potential $( \beta )$ , and time-independent non-conservative force $( \gamma )$

Optimization. LagrangeGS is trained in three sequential stages.

(Stage 1) Trajectory distillation. We minimize the rollout Mean Squared Error (MSE) between the integrated trajectories $\{ q _ { \tau } ^ { j } \}$ predicted by the TD-LNN and the pre-trained source trajectories $\{ q _ { \tau } ^ { \prime j } \}$ extracted from $f _ { d } \mathbf { \cdot }$

$$
\mathcal { L } _ { \mathrm { d i s t i l } } = \frac { 1 } { M \tau _ { \mathrm { m a x } } } \sum _ { \tau = 1 } ^ { \tau _ { \mathrm { m a x } } } \sum _ { j = 1 } ^ { M } \left. q _ { \tau } ^ { j } - q _ { \tau } ^ { \prime j } \right. ^ { 2 } + \lambda _ { Q } \mathbb { E } \left[ \left. Q \right. ^ { 2 } \right] ,\tag{10}
$$

The second term is a small $\ell _ { 2 }$ penalty that forces the dynamics onto the conservative potential U wherever possible, ensuring the non-conservative force Q only activates for genuine dissipation or external interactions. The initial velocity $\dot { q } _ { 0 } ^ { j }$ is a learnable parameter initialized by the source trajectories.

(Stage 2) TD-LNN fine-tuning. We freeze all components except $( U , Q , \{ \dot { q } _ { 0 } ^ { j } \} )$ and minimize the photometric loss (Eq. 1). Crucially, this stage allows LagrangeGS to systematically escape the overfitting of the source trajectories. Because the Kabsch alignment (Eq. 9) is differentiable, gradients flow from the photometric loss directly back to the TD-LNN parameters, supporting the model to discover true physical dynamics rather than merely memorizing pretrained deformations. Truncated Backpropagation Through Time (TBPTT) limits gradient propagation over long rollout sequences, while spatial particle subsampling randomly samples Gaussian particles during training to reduce memory consumption.

(Stage 3) Gaussian parameter finetuning. Finally, we freeze the TD-LNN parameters $( U , Q , \{ \dot { q } _ { 0 } ^ { j } \} )$ and refine only the Gaussian parameters $( q _ { 0 } ^ { j } , r _ { 0 } ^ { j } , s _ { 0 } ^ { j } , \sigma _ { 0 } ^ { j } , c _ { 0 } ^ { j } )$ using the photometric loss. This strategically absorbs residual visual errors without corrupting the rigorously learned physics.

## 3.3 Physics manipulations

The Lagrangian formulation in LagrangeGS enables long-term extrapolation and the physics manipulations at inference time without retraining:

Long-term extrapolation. For time steps beyond the training window $( t ^ { \prime } > \tau _ { \operatorname* { m a x } } )$ , we directly extend the forward integration of Eq. 8 with the potential $U ( q )$ and non-conservative force $Q ( q , \dot { q } )$ . Because Q has no explicit time dependence, it preserves time-translation symmetry and supporting stable long-term extrapolation.

![](images/1e8e8d26ca4ebc2493d69ea9f09021b6652abef12669a9523d1cfe9216f3ee7d.jpg)  
Figure 4: Learned trajectories of FreeGave and LagrangeGS with the potential $U ( q )$ for the "Fan" and "Falling Ball" scenes. While the rollouts of FreeGave exhibit erratic fluctuations, LagrangeGS ensures consistent trajectories satisfying Lagrangian mechanics. Without structural priors, $U ( q )$ captures the underlying physics: a rotationally symmetric potential for the Fan, and a uniform gravitational gradient for the Falling Ball.

Time reversal. To reconstruct past states of the scene $( t ^ { \prime } < 0 )$ , we solve the system backward in time by reversing the semi-implicit Euler integration with a negative time step $- \Delta \tau .$ , as shown in Figure 3:

$$
\dot { q } _ { \tau - 1 } ^ { j } = \dot { q } _ { \tau } ^ { j } - \ddot { q } _ { \tau } ^ { j } \Delta \tau , \qquad q _ { \tau - 1 } ^ { j } = q _ { \tau } ^ { j } - \dot { q } _ { \tau - 1 } ^ { j } \Delta \tau .\tag{11}
$$

Because the system is time independent, backward integration is mathematically well-posed, enabling exact state recovery $( \dot { q } _ { 0 } ^ { j } , \dot { q } _ { 0 } ^ { j } ) \mapsto ( q _ { - \tau _ { \operatorname* { m a x } } } ^ { j } , \dot { q } _ { - \tau _ { \operatorname* { m a x } } } ^ { j } )$

Counterfactual physics-based editing. As shown in Figure 3, we can directly manipulate the learned dynamic properties at inference time by applying explicit scaling factors $( \alpha , \beta , \gamma ) \in \mathbb { R } ^ { 3 }$ to the initial velocity, potential gradient, and non-conservative force, respectively. The edited acceleration becomes:

$$
\ddot { q } = - \beta \nabla _ { q } U ( q ) + \gamma Q ( q , \dot { q } ) ,\tag{12}
$$

paired with the scaled initial state $( q _ { 0 } ^ { j } , \alpha \dot { q } _ { 0 } ^ { j } )$ . Modulating these physical parameters allows us to seamlessly simulate counterfactual environments, such as scaling the initial velocity $( \alpha \neq 1 )$ , weakening the effective field strength $( \beta < 1 )$ , or adjusting the non-conservative dissipation/driving effects $( \gamma \neq 1 )$ .

## 4 Experiments

## 4.1 Experimental setup

Datasets. We evaluate LagrangeGS on two dynamic 3D scene benchmarks. The Dynamic Object dataset [8] contains six multi-view scenes featuring both globally rigid and locally deformable motions. The Dynamic Indoor Scene dataset [8] comprises four scenes in which multiple rigid objects move simultaneously against a static background. Both benchmarks expose held-out time windows to evaluate temporal extrapolation, in addition to held-out camera views for novel-view synthesis.

Baselines. We compare LagrangeGS with two types of dynamic-scene methods. The first type models the scene as a deformation of an initial representation: T-NeRF [17], D-NeRF [17], NSFF [12], and DefGS [25]. The second type additionally introduces physical intermediates: NVFi [8], TRACE [9], and FreeGave [10]. All baselines are evaluated on the same train/test splits as the proposed method.

## S.SATO ETAL.: LAGRANGEGS

![](images/fc02bef1c1e7ae9ca14aed12409572c3791752a779664afafb30da3de28f6dbf.jpg)  
Figure 5: Extrapolation on the Fan scene. Training spans t = 0.0–0.75. Upon entering extrapolation, DefGS instantly suffers from unnatural motion or complete stagnation, while FreeGave exhibits geometric drift at t = 1.80 and catastrophic volume divergence at t = 2.25. Distilling both backbones via LagrangeGS (Ours) mitigates these artifacts, sustaining physically consistent and smooth rotation across the entire extended horizon.

Pre-trained models. LagrangeGS is a universal physical framework that extracts its initial kinematic trajectories from a pre-trained dynamic 3DGS model. To demonstrate its robustness against variations in initial kinematic quality, we apply LagrangeGS using two representative models: DefGS [25] (purely visual) and FreeGave [10] (velocity-regularized). For each source, we train using the official implementation and default hyperparameters.

Evaluation metrics. We measure photorealistic rendering quality using standard metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). We qualitatively assess the performance on longterm extrapolation, time reversal, and counterfactual physics-based editing.

Implementation details. The potential energy U and the non-conservative force Q in the TD-LNN are predicted by four-layer MLPs with 128 hidden units and Sigmoid-weighted Linear Unit (SiLU) activations. The last layer of Q is zero-initialized to enforce a conservative dynamics baseline at the start of fine-tuning. We utilize a single NVIDIA DGX Spark for distillation and fine-tuning. Full hyperparameter settings and detailed network architectures are provided in the supplementary material. On a DGX Spark, the complete LagrangeGS pipeline requires 3.5 h for training and 90 s for testing, compared with 2.5 h and 60 s for FreeGave. Training memory increases from 4.7 GB to 9.6 GB due to differentiable multistep rollout, while both methods require 3.3 GB during testing.

![](images/f7fff50761a5f1441a5f9a712fcebd935a1aa442981d06c808a1742b22c86a4f.jpg)  
Figure 6: Extrapolation on the Factory scene. Training spans $t = 0 . 0 { - } 0 . 7 5$ . Upon extrapolation, DefGS immediately degenerates into an indistinct haze, while FreeGave suffers from geometric collapse for $t \geq 1 . 8 0$ . LagrangeGS prevents this long-rollout collapse for both backbones, preserving the can geometry.

## 4.2 Visual analysis of trajectories and potentials

Before evaluating the extrapolation capabilities, we first visually analyze Gaussian particle trajectories and the learned potential fields on representative scenes. As shown in Fig. 4, the FreeGave trajectories exhibit erratic fluctuations and fail to satisfy Lagrangian mechanics, while LagrangeGS yields physically consistent trajectories. Additionally, the learned potentials successfully capture the underlying scene physics without structural priors: recovering a rotationally symmetric harmonic potential for the Fan rotation, and a uniform gradient field strictly corresponding to gravitational acceleration for the Falling Ball.

## 4.3 Extrapolation

We evaluate the extrapolation capabilities of LagrangeGS and the baselines by rolling out the learned dynamics beyond the training window $( t > t _ { \operatorname* { m a x } } )$ . The results are visualized in Fig. 5 and Fig. 6. Note that the training window spans $t = 0 . 0 \mathrm { - } 0 . 7 5$ ; hence, we consider $t = 0 . 0 \ – 0 . 7 5$ as interpolation, $t = 0 . 7 5 – 1 . 0$ as short-term extrapolation, and $t > 1 . 0$ as longterm extrapolation. DefGS suffers from unnatural motion immediately upon extrapolation, while FreeGave exhibits geometric collapse and volume divergence at later time steps. In contrast, LagrangeGS (DefGS) preserves physically consistent and smooth motion across the entire extended horizon. LagrangeGS (FreeGave) also maintains stable motion and prevents the artifacts observed in the original FreeGave.

Table 1: Short-term extrapolation performance on dynamic scene benchmarks. LagrangeGS improves the extrapolation performance of DefGS. Conversely, when applied to FreeGave, photometric metrics exhibit a marginal drop. This is a plausible consequence of strict physical regularization: by satisfying Lagrangian mechanics across all particles, LagrangeGS deliberately prunes non-physical visual artifacts such as shadows.
<table><tr><td rowspan="2"></td><td colspan="3">Dynamic Object Dataset</td><td colspan="3">Dynamic Indoor Scene Dataset</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>T-NeRF [口]</td><td>13.818</td><td>0.739</td><td>0.324</td><td>22.242</td><td>0.700</td><td>0.363</td></tr><tr><td>D-NeRF 口</td><td>14.660</td><td>0.737</td><td>0.312</td><td>20.791</td><td>0.692</td><td>0.349</td></tr><tr><td>NSFF 口</td><td></td><td></td><td></td><td>24.163</td><td>0.795</td><td>0.289</td></tr><tr><td>TiNeuVox [0]</td><td>19.612</td><td>0.940</td><td>0.073</td><td>21.029</td><td>0.770</td><td>0.281</td></tr><tr><td>NVFi [日]</td><td>27.594</td><td>0.972</td><td>0.036</td><td>29.745</td><td>0.876</td><td>0.204</td></tr><tr><td>TRACE []</td><td>31.597</td><td>0.987</td><td>0.009</td><td>34.824</td><td>0.965</td><td>0.054</td></tr><tr><td rowspan="3">DefGS [] LagrangeGS (DefGS)</td><td>19.849</td><td>0.949</td><td>0.045</td><td>21.380</td><td>0.819</td><td>0.188</td></tr><tr><td>24.900</td><td>0.965</td><td>0.037</td><td>31.370</td><td>0.930</td><td>0.103</td></tr><tr><td>diff (+5.051)</td><td>(+0.016)</td><td>(-0.008)</td><td>(+9.990)</td><td>(+0.111)</td><td>(-0.085)</td></tr><tr><td>FreeGave []</td><td>31.987</td><td>0.990</td><td>0.007</td><td>35.019</td><td>0.966</td><td>0.051</td></tr><tr><td>LagrangeGS (FreeGave)</td><td>28.940</td><td>0.977</td><td>0.020</td><td>33.097</td><td>0.955</td><td>0.059</td></tr><tr><td>diff</td><td>(-3.047)</td><td>(-0.013)</td><td>(+0.013)</td><td>(-1.922)</td><td>(-0.011)</td><td>(+0.008)</td></tr></table>

Quantitatively, as shown in Table 1, LagrangeGS improves the photometric metrics when applied to DefGS, whereas a moderate degradation is observed when initialized from Free-Gave. To isolate whether this degradation originates from appearance or geometry, we additionally compute PSNR on binarized foreground silhouettes. On Dynamic Object, the FreeGave–LagrangeGS gap is nearly unchanged between RGB and silhouette PSNR (−3.2 vs. −3.2 dB), indicating that the degradation is primarily geometric. In contrast, on Dynamic Indoor Scene, the gap decreases substantially from −2.5 dB in RGB to −0.6 dB in silhouette PSNR, indicating that the degradation is mainly caused by appearance effects such as moving shadows.

To evaluate long-term extrapolation under constrained conditions, we artificially truncate the training window to $t < 0 . 5$ and $t < 0 . 3$ for FreeGave and LagrangeGS (FreeGave), and evaluate their performance during the original extrapolation window $( 0 . 7 5 < t \leq 1 . 0 )$ As shown in Table 2, the performance gap between the two models narrows as the training window is shortened. Especially under extreme truncation $( t < 0 . 3 )$ in the Dynamic Object dataset, LagrangeGS outperforms FreeGave, demonstrating its resilience to temporal overfitting and its structural stability in data-sparse regimes.

To quantitatively evaluate geometric stability during long-term extrapolation, we measure the foreground pixel-area extent relative to that at t = 0, since unstable rollouts typically manifest as object expansion or divergence. On the Dynamic Object dataset, LagrangeGS remains within 1.1× its initial foreground extent throughout the rollout, whereas FreeGave expands up to 1.9×. This result quantitatively confirms that LagrangeGS suppresses the geometric divergence observed in long-term rollout.

Table 2: Extrapolation under truncated observation windows $( t < 0 . 5 , t < 0 . 3 )$ . By artificially shortening the training horizon (originally $t _ { m a x } \sim 0 . 7 5 )$ , the performance gap between FreeGave and LagrangeGS narrows as the training window is shortened. Especially for the extreme truncation $( t < 0 . 3 )$ in the Dynamic Object dataset, LagrangeGS outperforms Free-Gave, demonstrating its resilience in long-term extrapolation under data-sparse regimes.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Dynamic Object Dataset</td><td colspan="3">Dynamic Indoor Scene Dataset</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="2">Full</td><td>FreeGave LagrangeGS</td><td>31.987</td><td>0.990</td><td>0.007</td><td>35.019</td><td>0.966</td><td>0.051</td></tr><tr><td>diff</td><td>28.940 (-3.047)</td><td>0.977 (-0.013)</td><td>0.020 (+0.013)</td><td>33.097</td><td>0.955</td><td>0.059</td></tr><tr><td rowspan="2"> $t < 0 . 5$ </td><td>FreeGave</td><td>24.874</td><td>0.970</td><td>0.027</td><td>(-1.922) 25.757</td><td>(-0.011)</td><td>(+0.008)</td></tr><tr><td>LagrangeGS</td><td>24.601</td><td>0.962</td><td>0.032</td><td>24.713</td><td>0.823 0.821</td><td>0.214 0.218</td></tr><tr><td rowspan="2"></td><td>diff</td><td>(-0.273)</td><td>(-0.008)</td><td>(+0.005)</td><td>(-1.044)</td><td>(-0.002)</td><td>(+0.004)</td></tr><tr><td>FreeGave</td><td>21.341</td><td>0.960</td><td>0.038</td><td>23.677</td><td>0.789</td><td>0.246</td></tr><tr><td rowspan="2"> $t < 0 . 3$ </td><td>LagrangeGS</td><td>22.272</td><td>0.949</td><td>0.042</td><td>22.975</td><td>0.786</td><td>0.246</td></tr><tr><td>diff</td><td>(+0.913)</td><td>(-0.011)</td><td>(+0.004)</td><td>(-0.702)</td><td>(-0.003)</td><td>(+0.000)</td></tr></table>

![](images/2c94d5204b92c41e2e0c5d4300221d5252352522dd3e0519facfe53a807483f9.jpg)  
Figure 7: Time-reversal rollout on the "Telescope" scene via LagrangeGS. Past states $( t < 0 )$ are recovered exactly by integrating the same trained TD-LNN backward with a negative time step $- \Delta t$ , ensuring visual and structural consistency with forward rollouts $( t > 0 )$ . Red arrows denote per-particle velocity vectors; their exact directional inversion across $t = 0$ confirms that LagrangeGS captures genuine continuous mechanics rather than a learned animation shortcut.

## 4.4 Time reversal

We validate the time-reversal capability of LagrangeGS by rolling out the learned TD-LNN backward in time with a negative time step −∆t. As verified by the per-particle velocity vectors shown in Fig. 7, the velocity directions invert exactly across the temporal boundary $( t = 0 )$ . Consequently, the recovered past states at $t = - 0 . 3 , - 0 . 6 , - 0 . 9$ exhibit structural and visual consistency with the corresponding forward rollout trajectories at $t = + 0 . 3 , + 0 . 6 , + 0 . 9$ . To quantify this consistency, we perform a forward–backward cycle test by integrating each trajectory from $t = 0$ to $t _ { \mathrm { m a x } }$ and subsequently back to $t = 0$ . We measure the final position error $\left\| \mathbf { q } _ { 0 } - \hat { \mathbf { q } } _ { 0 } \right\|$ normalized by the total traveled distance. The mean cycle error is only 1.3% on Dynamic Object and 0.19% on Dynamic Indoor Scene, demonstrating strong forward–backward consistency despite the absence of an exact time-reversal guarantee.

Table 3: Ablation study. The Stage 2 (TD-LNN) and Stage 3 (Gaussian parameter) finetuning contribute to visual scores. While LRA is essential for preventing geometric collapse, its absence marginally inflates PSNR in the Indoor dataset; this merely reflects unconstrained particles overfitting to non-rigid visual phenomena like moving shadows. Furthermore, because the Indoor scenes exhibit strictly conservative motions, the non-conservative force Q is redundant, meaning its removal avoids over-parameterization and slightly benefits PSNR.
<table><tr><td rowspan="2"></td><td colspan="3">Dynamic Object Dataset</td><td colspan="3">Dynamic Indoor Scene Dataset</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Full</td><td>28.940</td><td>0.977</td><td>0.020</td><td>33.097</td><td>0.955</td><td>0.059</td></tr><tr><td>w/o stage3</td><td>27.608</td><td>0.975</td><td>0.018</td><td>27.804</td><td>0.934</td><td>0.079</td></tr><tr><td>w/o stage2,3</td><td>27.471</td><td>0.974</td><td>0.021</td><td>29.011</td><td>0.938</td><td>0.077</td></tr><tr><td>w/o LRA</td><td>26.964</td><td>0.974</td><td>0.022</td><td>34.261</td><td>0.957</td><td>0.057</td></tr><tr><td>w/o Force Q</td><td>24.971</td><td>0.965</td><td>0.032</td><td>33.303</td><td>0.956</td><td>0.054</td></tr></table>

## 4.5 Counterfactual physics-based editing

We demonstrate the counterfactual physics-based editing capability of LagrangeGS by directly rescaling the learned potential field $U$ at inference time. As shown in Fig. 8, decreasing the scaling factor $\beta$ weakens the effective field and slows the ball’s descent, whereas increasing $\beta$ accelerates the motion.

We further quantify whether this manipulation follows the intended physical change. For the Falling Ball scene, scaling the learned potential by $\beta$ changes the falling time according to the expected relation $t \propto \beta ^ { - 1 / 2 }$ . Fitting the simulated falling times yields $\bar { t } \propto \beta ^ { - 0 . 4 8 9 }$ with $R = 0 . 9 9 9$ , closely matching the theoretical exponent of −0.5. This demonstrates that the learned potential supports quantitatively calibrated physics-based editing rather than merely producing visually plausible trajectory changes.

LagrangeGS also supports scaling the initial velocity and the non-conservative force, enabling counterfactual simulations such as varying the initial impulse or dissipation/driving effects.

## 4.6 Ablation studies

We perform an ablation study for key components of LagrangeGS. Table 3 summarizes the results. TD-LNN fine-tuning (Stage 2) and Gaussian parameter refinement (Stage 3) consistently improve rendering fidelity. LRA is essential for preventing geometric collapse during long-term extrapolation and improving visual scores in the Dynamic Object dataset. However, its absence marginally inflates the scores in the Indoor dataset. The lack of LRA allows the particles to overfit to non-rigid visual phenomena, such as moving shadows, within the expressivity limits of $U , Q .$ , and $\dot { q } _ { 0 }$ . The non-conservative force $Q$ significantly benefits the Dynamic Object dataset, which contains non-conservative motions; however, it is redundant for the strictly conservative motions in the Indoor dataset, where its removal avoids overparameterization and slightly benefits photometric scores.

## 4.7 Limitations

While LagrangeGS successfully constrains dynamic 3DGS within Lagrangian mechanics, it exhibits two key limitations. First, the current TD-LNN requires distillation from a pretrained dynamic 3DGS model, since the unstable nature of the initial random trajectories makes it difficult to directly train from scratch. Future work could explore more robust training strategies or curriculum learning to enable direct end-to-end training. Second, the current formulation assumes decoupled particle dynamics, which may not capture complex interactions such as collisions or contact forces. Extending the framework to incorporate more sophisticated kinetic priors or interaction models could further enhance its applicability to a wider range of dynamic scenes.

![](images/1701c63c9baaccca8fbe3bc88e9b6eec18fd34382c1f79cd28f145edbb51b3bb.jpg)  
Figure 8: Inference-time counterfactual physics-based editing on the "Falling Ball" scene. The dynamics are systematically manipulated without retraining by rescaling the learned potential field $( U ^ { \prime } = \beta \cdot U )$ . The baseline $( \beta = 1 . 0 )$ reproduces the true observed trajectory. Decreasing $\beta$ simulates a weakened gravitational environment, slowing the descent, while increasing $\beta$ strengthens the effective gravity field. This predictable and continuous modulation exposes the learned $U ( q )$ as an explicitly editable physical property rather than an uninterpretable visual latent feature.

## 5 Conclusion

In this paper, we introduced LagrangeGS, a novel framework that integrates Lagrangian mechanics into dynamic 3DGS. LagrangeGS features two key components: an TD-LNN, which approximates the velocity-Hessian as an identity matrix for computational tractability and employs a time-independent system for consistent time reversal; and LRA, which binds the decoupled particles into locally rigid structures to prevent geometric collapse. Our experiments on two dynamic scene benchmarks demonstrate that LagrangeGS improves long-term extrapolation and enables consistent time reversal and counterfactual physics-based editing. This work establishes an explicit formulation of dynamic scenes as Lagrangian systems, paving the way for physically grounded dynamic modeling and manipulation in 3DGS.

## References

[1] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Proc. NeurIPS, 31, 2018.

[2] Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, and Shirley Ho. Lagrangian neural networks. arXiv preprint arXiv:2003.04630, 2020.

[3] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In Proc. SIGGRAPH Asia, pages 1–9, 2022.

[4] Samuel Greydanus, Misko Dzamba, and Jason Yosinski. Hamiltonian neural networks. Proc. NeurIPS, 32, 2019.

[5] Yuanming Hu, Jiancheng Liu, Andrew Spielberg, Joshua B Tenenbaum, William T Freeman, Jiajun Wu, Daniela Rus, and Wojciech Matusik. Chainqueen: A real-time differentiable physical simulator for soft robotics. In Proc. ICRA, pages 6265–6271. IEEE, 2019.

[6] Wolfgang Kabsch. A solution for the best rotation to relate two sets of vectors. Foundations ofCrystallography, 32(5):922–923, 1976.

[7] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph., 42 (4):139–1, 2023.

[8] Jinxi Li, Ziyang Song, and Bo Yang. NVFi: Neural Velocity Fields for 3D Physics Learning from Dynamic Videos. Proc. NeurIPS, 36:34723–34751, 2023.

[9] Jinxi Li, Ziyang Song, and Bo Yang. TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos. In Proc. ICCV, pages 8820–8829, 2025.

[10] Jinxi Li, Ziyang Song, Siyuan Zhou, and Bo Yang. FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity. In Proc. CVPR, pages 12433–12443, 2025.

[11] Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. Pac-nerf: Physics augmented continuum neural radiance fields for geometry-agnostic system identification. arXiv preprint arXiv:2303.05512, 2023.

[12] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proc. CVPR, pages 6498–6508, 2021.

[13] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. DynI-BaR: Neural Dynamic Image-Based Rendering. In Proc. CVPR, pages 4273–4284, 2023.

[14] Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan. Physics3d: Learning physical properties of 3d gaussians via video diffusion. arXiv preprint arXiv:2406.04338, 2024.

[15] Michael Lutter, Christian Ritter, and Jan Peters. Deep lagrangian networks: Using physics as model prior for deep learning. arXiv preprint arXiv:1907.04490, 2019.

[16] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. Communications ofthe ACM, 65(1):99–106, 2021.

[17] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural Radiance Fields for Dynamic Scenes. In Proc. CVPR, pages 10318– 10327, 2021.

[18] Hai-Long Qin, Sixian Wang, Guo Lu, and Jincheng Dai. Neural Hamiltonian Deformation Fields for Dynamic Scene Rendering. In Proc. SIGGRAPH Asia, pages 1–11, 2025.

[19] Jinsheng Quan, Qiaowei Miao, Yichao Xu, Zizhuo Lin, Ying Li, Wei Yang, Zhihui Li, and Yawei Luo. ParticleGS: Learning Neural Gaussian Particle Dynamics from Videos for Prior-free Physical Motion Extrapolation. arXiv preprint arXiv:2505.20270, 2025.

[20] Deborah Sulsky, Zhen Chen, and Howard L Schreyer. A particle method for historydependent materials. Computer methods in applied mechanics and engineering, 118 (1-2):179–196, 1994.

[21] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on pattern analysis and machine intelligence, 13(4): 376–380, 1991.

[22] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. In Proc. CVPR, pages 20310–20320, 2024.

[23] Shanshan Xiao, Jiawei Zhang, and Yifa Tang. Generalized lagrangian neural networks. arXiv preprint arXiv:2401.03728, 2024.

[24] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physics-integrated 3d gaussians for generative dynamics. In Proc. CVPR, pages 4389–4398, 2024.

[25] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. In Proc. CVPR, pages 20331–20341, 2024.