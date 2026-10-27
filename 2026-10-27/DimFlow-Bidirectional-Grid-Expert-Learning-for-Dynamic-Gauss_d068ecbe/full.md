# DimFlow: Bidirectional Grid–Expert Learning for Dynamic Gaussian Reconstruction

Gabriel Adineera and Jinoh Kim<sup>∗</sup>

Abstract— Dynamic Gaussian reconstruction requires compact scene features to support coordinated changes in geometry and appearance. A factorized grid provides efficient local queries, but accurate interpolation alone does not ensure that the resulting dynamic attributes remain compatible across space and time. We introduce DimFlow, a framework that learns these features through bidirectional interaction between a multiresolution grid and a coordinate-based expert branch. Featurelevel adversarial alignment connects their distributions, adaptive fusion transfers complementary information, and activationmasked residual feedback refines the grid during training. Four attribute experts organize deformation, rotation, opacity, and temporal information before Gaussian rendering. In the aggregate comparison across D-NeRF, HyperNeRF, and DAVIS, DimFlow achieves 39.73 dB PSNR, improving on 4D-GS by 1.92 dB while matching its reported 85 FPS rendering speed. The model occupies 18.9 MB, compared with 18.3 MB for 4D-GS. Component and attribute-expert ablations reveal complementary contributions from feature alignment, adaptive exchange, and temporal processing.

## I. INTRODUCTION

Reconstructing a moving scene requires more than rendering each observed frame accurately. A useful representation must also describe how scene elements change between observations, while retaining enough detail for a new view at a chosen time. This combination matters for interactive visualization, dynamic scene understanding, and robotic perception. The central difficulty is coordinating spatial detail, motion, and appearance within a representation that can still be queried efficiently.

Neural radiance fields [1] establish a continuous formulation of scene appearance. Dynamic extensions [2], [3] introduce time-dependent mappings, while explicit feature representations [4]–[7] reduce the cost of querying space and time. Gaussian splatting [8] changes the rendering substrate by projecting explicit primitives into the image. Building on this representation, 4D-GS [9] combines canonical Gaussians with compact spatiotemporal features and a deformation decoder. These developments make dynamic reconstruction substantially more practical.

The representation and its optimization nevertheless play different roles. A compact grid can store local variations efficiently, but image supervision must still organize those features into a useful decoding space. Likewise, a shared coordinate network can express regularities across the scene without automatically providing the local detail that a grid can retain. Simply placing both components in a pipeline leaves a further question: what information should they exchange, and how should that exchange influence learning?

![](images/1f26eb04cd14955a4dd864106a4f6e232d9a24733675070c85f85c1958c5f331.jpg)  
Fig. 1. Two feature views of the same scene. Two-dimensional projections of grid and coordinate-expert features without and with adversarial alignment. Each panel contains 350 samples from each branch. Distributional overlap is interpreted together with the reconstruction ablation.

DimFlow addresses this question through bidirectional grid–expert learning. The grid branch supplies interpolated local features at multiple resolutions. A coordinate-based expert branch produces a complementary representation of the same spacetime query. During the forward computation, adaptive fusion transfers information between the two feature views. During optimization, adversarial alignment and feature residuals guide the grid toward a representation that remains compatible with the expert branch. Here “bidirectional” denotes forward feature enhancement and backward optimization feedback.

We separate this feature interaction from attribute decoding. Four specialized branches organize deformation, rotation, opacity, and temporal information. Their gate controls how the corresponding responses contribute to a Gaussian update. This distinction makes the design easier to interpret: the coordinate branch supports feature learning, whereas the attribute experts structure the dynamic decoder. Temporal processing further connects neighboring feature states instead of treating every timestamp as an unrelated prediction.

The aggregate comparison reaches 39.73 dB PSNR with 85 FPS rendering and an 18.9 MB representation. Relative to 4D-GS, the PSNR improves by 1.92 dB at the same reported frame rate, with 0.6 MB additional storage. The ablations examine the feature-learning components and the attribute experts in separate settings. Our contributions are:

• A grid–expert formulation that connects efficient local feature queries to a shared coordinate representation through feature-level adversarial alignment.

• A bidirectional interaction mechanism combining adaptive forward fusion, activation-masked feedback, and temporally coupled features for dynamic attribute decoding.

• An evaluation of reconstruction quality, rendering efficiency, and component contributions, including a separate analysis of deformation and temporal experts.

## II. RELATED WORK

## A. Dynamic Radiance Fields and Explicit Representations

NeRF [1] represents a static scene through a continuous radiance field. NeRF++ [10] and mip-NeRF [11] address different limitations of the underlying representation and sampling process. For dynamic scenes, D-NeRF [2] learns a deformation to canonical space, while HyperNeRF [3] uses an additional embedding to represent topological variation. Neural scene flow fields [12] connect scene appearance with time-dependent motion. These methods establish the importance of sharing information between observations rather than independently reconstructing each frame.

Explicit and factorized features offer a different quality– cost tradeoff. TiNeuVox [4] combines voxels with temporal information; TensoRF [13] factorizes radiance-field representations. HexPlane [5] and K-Planes [6] use coordinate-plane features to represent higher-dimensional fields compactly. MSTH [7] separates static and dynamic contributions through masked hash encodings. DimFlow adopts the efficient featurequery perspective and focuses on how the stored features are organized by learning and exchange.

Compact visual representations also arise in other settings. HTC-VLM [14] combines appearance features and discrete semantic anchors for token compression; video encoders built on image representations [15] retain frame evidence before temporal refinement. DimFlow instead compresses and queries a dynamic scene field for Gaussian rendering.

## B. Gaussian-Based Dynamic Reconstruction

3D Gaussian Splatting [8] enables efficient differentiable rendering with explicit anisotropic primitives. Dynamic extensions must additionally account for time-varying geometry and visibility. Deformable 3D Gaussians [16] and 4D-GS [9] learn temporal changes to Gaussian representations. DG-Mesh [17] couples Gaussian representations with mesh reconstruction. DrivingGaussian [18] studies composite dynamic scenes in autonomous driving, while Gaussian representations have also attracted interest in robotics [19]. Our evaluation concerns reconstruction and rendering; it does not evaluate downstream navigation or control.

DepthSSC [20] aligns monocular depth and spatial evidence for voxel completion. The Tri-MARF framework in 3D-Agent [21] integrates views, text, and point clouds for object annotation. The ActPhysCause challenge [22] evaluates actionconditioned future videos. These tasks add semantic or causal requirements beyond reconstructing observed dynamic appearance.

## C. Expert Specialization and Adversarial Feature Learning

Mixture-of-experts models [23]–[25] allocate representational capacity through specialized branches. Switch-NeRF [26] learns scene decomposition with experts, and model-agnostic NeRF mixtures [27] study expert combinations for rendering. Beyond rendering, KABB [28] selects experts, OSC [29] adapts communication, GAM-Agent [30] uses uncertainty to trigger visual-reasoning debate, and Agora [31] exchanges structured visual uncertainty. MAT-Agent [32] coordinates training choices; failure-driven workflow refinement [33] edits workflows from counterexamples, while DrDiff [34] schedules experts for text diffusion.

Markets of Minds [35] provides a particularly relevant distinction: expert assistance is valued relative to a default continuation, with repair benefits and regression risks considered under shared budgets and service capacity. Difficulty identifies possible need, whereas allocation must also account for competing uses of the same resources. DimFlow’s attribute gate blends responses within one model. Section IV-E uses this distinction to clarify what its weights and ablations establish.

Adversarial learning [36] provides a way to compare learned distributions. GANeRF [37] uses discriminators to optimize radiance fields, and NeRF–GAN distillation [38] connects generative and radiance-field representations. Dim-Flow places the discriminator between two feature branches. Distribution alignment, pointwise feature consistency, and image reconstruction therefore supply distinct constraints on the same representation.

Visual faithfulness has different operational meanings across tasks. CF-VLM [39] uses counterfactual edits for causal sensitivity, MM-CoT [40] tests visual consistency and logical coherence, and top-down semantic refinement [41] plans image captions. Our evaluation measures rendered-image agreement and dynamic diagnostics rather than languagegeneration fidelity.

## III. BIDIRECTIONAL GRID–EXPERT LEARNING

## A. Dynamic Gaussian Representation

Given images $I _ { t }$ and their camera parameters, we optimize a scene representation that can render a view at time t. A canonical Gaussian has center $\mu _ { i }$ , rotation $R _ { i }$ , scale $s _ { i } ,$ opacity $\alpha _ { i } ,$ , and view-dependent appearance $c _ { i } .$ The dynamic decoder predicts updates to its attributes. The covariance is

$$
\Sigma _ { i } ( t ) = R _ { i } ( t ) \mathrm { d i a g } \big ( s _ { i } ( t ) ^ { 2 } \big ) R _ { i } ( t ) ^ { \top } .\tag{1}
$$

We use the projected Gaussian and front-to-back compositing formulation of Gaussian splatting [8]. For pixel $p ,$ let $a _ { i } ( p , t )$ be the opacity-weighted projected footprint of the ith contributing Gaussian. Then

$$
C _ { t } ( \boldsymbol { p } ) = \sum _ { i \in \mathcal { N } _ { t } ( \boldsymbol { p } ) } T _ { i } ( \boldsymbol { p } , t ) a _ { i } ( \boldsymbol { p } , t ) c _ { i } , \quad T _ { i } = \prod _ { j < i } ( 1 - a _ { j } ) .\tag{2}
$$

The ordering is by depth. This separates scene attributes, their projection, and visibility-aware image formation.

![](images/d99c39d3b57896b6f2837e7578f3d7d0476a67609343398530061f7b2176beb1.jpg)  
Fig. 2. DimFlow overview. Grid and coordinate features are adaptively fused before four gated attribute experts predict dynamic Gaussian updates. Canonical Gaussians provide the reference state, and differentiable splatting produces the requested view. The lower band summarizes distribution alignment, activation-masked feature feedback, and temporal regularity used during training. All illustrated planes and Gaussian clouds are schematic.

## B. Complementary Feature Branches

a) Multi-resolution grid.: For a spacetime query $( x , t )$ the local branch queries a hierarchy of feature fields $\dot { \mathcal { M } } ^ { ( \ell ) }$ We denote the interpolation and collection operator by $\mathcal { Q } _ { \ell } \colon$

$$
F _ { g } ^ { ( \ell ) } ( x , t ) = \mathcal { Q } _ { \ell } ( \mathcal { M } ^ { ( \ell ) } , x , t ) .\tag{3}
$$

The factorized representation uses the coordinate planes

$$
\mathcal { P } = \{ x y , x z , y z , x t , y t , z t \} .\tag{4}
$$

Each plane is queried with its corresponding coordinate pair. The query operator applies the interpolation native to each plane and collects its sampled features. Level-specific projections map features to a common channel dimension before fusion:

$$
F _ { g } = W _ { g } [ \psi _ { 1 } ( { F } _ { g } ^ { ( 1 ) } ) ; \ldots ; \psi _ { L } ( { F } _ { g } ^ { ( L ) } ) ] + b _ { g } .\tag{5}
$$

The hierarchy provides local information at different spatial scales without storing a separate full scene for every frame.

b) Coordinate expert.: The complementary branch evaluates shared MLPs on spatial and temporal encodings:

$$
F _ { e } ^ { ( \ell ) } ( x , t ) = G _ { e } ^ { ( \ell ) } ( [ \gamma ( x ) ; \tau ( t ) ] ) ,\tag{6}
$$

$$
\tau ( t ) = [ \sin ( 2 ^ { k } \omega t ) , \cos ( 2 ^ { k } \omega t ) ] _ { k = 0 } ^ { K - 1 } .\tag{7}
$$

The same network parameters serve many queries, providing a different inductive bias from locally stored features. The coordinate generator uses a five-layer MLP with GELU activation. Channel-wise level weights combine the responses as $\begin{array} { r } { F _ { e } = \sum _ { \ell } \beta ^ { ( \ell ) } \odot F _ { e } ^ { ( \ell ) } } \end{array}$ . The expert features are learned jointly with reconstruction. Both branches are projected to compatible dimensions wherever they are compared or fused.

## C. Distribution Alignment and Residual Feedback

a) Feature discrimination.: At level ℓ, a discriminator $D _ { \ell }$ distinguishes coordinate-expert features from grid features.

Its multilevel objective is

$$
\begin{array} { r l r } {  { V ( D ) = \sum _ { \ell = 1 } ^ { L } \omega _ { \ell } \big [ \mathbb { E } _ { F _ { e } } \log D _ { \ell } ( F _ { e } ) } } \\ & { } & { + \mathbb { E } _ { F _ { g } } \log ( 1 - D _ { \ell } ( F _ { g } ) ) \big ] , } \\ & { } & { \omega _ { \ell } = \frac { 2 ^ { \ell } - 1 } { 2 ^ { L } - 1 } . } \end{array}\tag{8}
$$

(9)

The discriminator maximizes V while holding feature samples fixed. During representation learning, the grid branch minimizes the grid-side adversarial term

$$
\mathcal { L } _ { \mathrm { a d v } } = \sum _ { \ell } \omega _ { \ell } \mathbb { E } _ { F _ { g } } \log ( 1 - D _ { \ell } ( F _ { g } ) ) ,\tag{10}
$$

alongside the reconstruction objectives. Thus discriminator fitting and feature-generator optimization have different roles. The discriminator is a training component, rather than an additional rendering stage.

b) Pointwise correspondence.: Adversarial alignment compares distributions and does not enforce agreement at a specific spacetime query. We also retain a direct consistency objective,

$$
\mathcal { L } _ { \mathrm { c o n } } = \sum _ { \ell } \left. F _ { g } ^ { ( \ell ) } - F _ { e } ^ { ( \ell ) } \right. _ { 1 } .\tag{11}
$$

The feedback path emphasizes active expert channels through

$$
M ^ { ( \ell ) } = \mathbf { 1 } \{ | F _ { e } ^ { ( \ell ) } | > \tau _ { a } \} , \qquad \tau _ { a } = 0 . 0 5 ,\tag{12}
$$

$$
r ^ { ( \ell ) } = M ^ { ( \ell ) } \odot ( F _ { g } ^ { ( \ell ) } - F _ { e } ^ { ( \ell ) } ) .\tag{13}
$$

Its magnitude identifies feature discrepancies used in refinement. The reverse path returns optimization signals to the feature representation.

## D. Adaptive Forward Exchange and Temporal Processing

a) Channel-wise fusion.: Let $a ^ { ( \ell ) } ~ \in ~ [ 0 , 1 ] ^ { C }$ denote the adaptive grid weight, conditioned on feature and scenecomplexity information. The fused feature is

$$
z ^ { ( \ell ) } = a ^ { ( \ell ) } \odot F _ { g } ^ { ( \ell ) } + ( 1 - a ^ { ( \ell ) } ) \odot F _ { e } ^ { ( \ell ) } .\tag{14}
$$

Equivalently, the expert-injection weight is $1 - a ^ { ( \ell ) }$ . Motion information includes velocity and acceleration magnitudes:

$$
c _ { \mathrm { m o t } } ^ { ( \ell ) } = \sigma \Big ( \alpha _ { v } \left\| \dot { x } ^ { ( \ell ) } \right\| _ { 2 } + \alpha _ { a } \left\| \ddot { x } ^ { ( \ell ) } \right\| _ { 2 } \Big ) .\tag{15}
$$

Here $x ^ { ( \ell ) } ( t )$ is the moving state used to characterize the region, rather than a fixed query coordinate. Deformation descriptors provide complementary information about that state. A channel allocation generated from the complexity signals controls the blend. These signals condition the fusion rule rather than imposing that every moving region must use the same branch. In particular, the gate is not a calibrated probability that an expert will improve reconstruction.

For a fixed gate, the displacement from the grid feature obeys

$$
\begin{array} { r } { \left\| z ^ { ( \ell ) } - F _ { g } ^ { ( \ell ) } \right\| _ { 2 } \leq \left\| F _ { e } ^ { ( \ell ) } - F _ { g } ^ { ( \ell ) } \right\| _ { 2 } . } \end{array}\tag{16}
$$

This follows from the componentwise bound $0 \leq 1 - a _ { c } \leq 1$ It describes the forward blend; it does not establish stability of the learned gate or guarantee that expert injection improves every query.

b) Temporal feature coupling.: For neighboring feature states, a finite-difference update combines local averaging with a motion trend:

$$
\widetilde { z } _ { t } = \frac { 1 } { 2 } \big ( z _ { t } + z _ { t - 1 } \big ) + \eta \frac { z _ { t - 1 } - z _ { t - 2 } } { \Delta t } .\tag{17}
$$

Here $\eta$ controls the trend contribution and depends on timestamp scaling. Temporal regularization penalizes abrupt feature changes:

$$
\mathcal { L } _ { \mathrm { t e m p } } = \sum _ { t } \| \partial \boldsymbol { z } _ { t } / \partial t \| _ { 1 } .\tag{18}
$$

This encourages a coherent feature sequence but cannot enforce physically correct motion through an occlusion or a topology change.

## E. Attribute Experts and Joint Optimization

The decoder organizes four specialized responses for deformation, rotation, opacity, and temporal information. Given the fused representation $z ,$ their weights and responses are

$$
\pi = \operatorname { s o f t m a x } ( g ( z , \tau ( t ) ) ) , \quad h _ { k } = E _ { k } ( z ) , \quad k = 1 , \ldots , 4 .\tag{19}
$$

A decoder D combines these responses to predict the Gaussian update:

$$
\Delta \vartheta = { \cal D } ( \pi _ { 1 } h _ { 1 } , \pi _ { 2 } h _ { 2 } , \pi _ { 3 } h _ { 3 } , \pi _ { 4 } h _ { 4 } ) .\tag{20}
$$

The aggregation preserves attribute types when decoding branch responses into Gaussian parameters. The updated Gaussians are rendered using Equation (2).

The rendering term is the mean squared error between rendered and observed pixels. Representation and decoder learning uses

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { r } \mathcal { L } _ { \mathrm { r e n d e r } } + \lambda _ { a } \mathcal { L } _ { \mathrm { a d v } } + \lambda _ { c } \mathcal { L } _ { \mathrm { c o n } } + \lambda _ { t } \mathcal { L } _ { \mathrm { t e m p } } , } \end{array}\tag{21}
$$

with nonnegative normalized weights. Discriminator updates remain separate from minimizing this objective. Rendering

Algorithm 1 Grid–expert reconstruction and rendering   
Require: Posed video, canonical Gaussians, feature fields.   
1: for each training batch do   
2: Query grid and coordinate features at sampled $( x , t )$   
3: Update the feature discriminator with fixed samples.   
4: Form adaptive fused features and temporal context.   
5: Evaluate four attribute experts and update Gaussians.   
6: Render the current camera view.   
7: Optimize representation and decoder using Equa  
tion (21); return feature-residual feedback to the grid.   
8: end for   
9: Render a requested (camera, t) from the learned repre  
sentation and decoder.

trains the useful scene representation, consistency connects matching queries, and the adversarial term shapes the gridside feature distribution.

## IV. PROPERTIES OF FEATURE EXCHANGE

The formulation separates three questions: whether the two feature branches have compatible distributions, whether corresponding queries agree, and whether their combination supports accurate rendering. We derive simple properties of the feature operations to clarify how these constraints interact.

## A. Distribution Matching Does Not Fix Correspondence

Consider a finite set of queries $q _ { i } ~ = ~ ( x _ { i } , t _ { i } )$ sampled uniformly. A permutation $\rho$ of their coordinate-expert features preserves the empirical feature distribution:

$$
F _ { g } ( q _ { i } ) = F _ { e } ( q _ { \rho ( i ) } ) \implies \{ F _ { g } ( q _ { i } ) \} _ { i = 1 } ^ { n } = \{ F _ { e } ( q _ { i } ) \} _ { i = 1 } ^ { n } .\tag{22}
$$

A feature discriminator observing only these samples cannot distinguish the branches from this distribution alone. Yet the feature assigned to a particular scene location can be wrong whenever $\rho ( i ) \neq i .$ . Pointwise consistency and image reconstruction therefore retain roles that distribution alignment cannot replace.

Conversely, matching individual features without a useful rendering objective can make the branches agree on an uninformative representation. The coordinate branch is consequently trained with the scene, and its usefulness is determined jointly with reconstruction. This also explains why visual overlap in Figure 1 is interpreted together with the component ablation, rather than as an independent accuracy guarantee.

## B. Adaptive Gates Add a Learning Path

The bounded displacement in Equation (16) holds for the value of the current gate. If that gate depends on the queried features, its derivative also influences learning. With $q = ( x , t )$ and channel-wise Jacobians, differentiating Equation (14) gives

$$
\begin{array} { r } { J _ { z } = \mathrm { d i a g } ( a ) J _ { g } + \mathrm { d i a g } ( 1 - a ) J _ { e } } \\ { + \mathrm { d i a g } ( F _ { g } - F _ { e } ) J _ { a } . \qquad } \end{array}\tag{23}
$$

![](images/6100a2f728f728001655cc749f1ab18352bb6569db6c4b42e53f788f5bd92e5b.jpg)  
Fig. 3. Temporal averaging and trend compensation. Analytical frequency response of Equation (17) for unit-spaced time indices. Increasing η preserves more rapid variation, illustrating why trend compensation and temporal regularization serve different roles. The curves are derived from the stated operator.

The last term depends on both branch disagreement and gate sensitivity. Small feature disagreement can reduce this contribution, but bounded gate values alone do not bound $J _ { a } .$ Feature matching and adaptive selection thus influence different parts of the learning problem.

The same distinction matters for interpretation. A large motion descriptor can change a gate response without proving that one branch is superior at that location. Rendering supervision determines whether the resulting attribute update is useful. This motivates evaluating adaptive weighting by removing it from the complete system, while avoiding a universal rule that complex motion always benefits from stronger expert injection.

## C. Temporal Averaging and Trend Compensation

For a constant-velocity feature trajectory, $z _ { t } = z _ { t - 1 } + \Delta t v .$ Equation (17) reduces to

$$
\begin{array} { r } { \widetilde { z } _ { t } = z _ { t } + \left( \eta - \frac { 1 } { 2 } \Delta t \right) v . } \end{array}\tag{24}
$$

The averaging component introduces half a time-step of lag; setting $\eta = \Delta t / 2$ compensates that lag for this trajectory. With unit-spaced time indices this corresponds to $\eta = 0 . 5$ The interpretation depends on time normalization and does not assume that arbitrary scene motion is linear.

The derivative term can also transmit noise or respond poorly to acceleration. Temporal coupling therefore balances local regularity against motion responsiveness. An abrupt appearance change caused by occlusion should not automatically be treated as an erroneous fluctuation. The separate temporalexpert ablation probes the behavior of the full learned system under this tradeoff.

For a discrete temporal frequency $\nu ,$ let $\kappa = \eta / \Delta t$ . The response of the linear update is

$$
\begin{array} { r } { H _ { \kappa } ( \nu ) = \frac 1 2 + ( \frac 1 2 + \kappa ) e ^ { - \mathrm { i } \nu } - \kappa e ^ { - 2 \mathrm { i } \nu } . } \end{array}\tag{25}
$$

It preserves constant features because $H _ { \kappa } ( 0 ) = 1$ . At the highest discrete frequency, $| H _ { \kappa } ( \pi ) | = 2 | \kappa |$ , so the update is not uniformly attenuating for arbitrary trend weights. Figure 3 visualizes this distinction for three coefficients. Temporal regularization remains useful even when the averaging and trend terms accurately track a slowly changing feature trajectory.

For unit-spaced time and $\eta = 0 . 5 ,$ , the gain is $\sqrt { 1 + \sin ^ { 2 } \nu }$ reaching $\sqrt { 2 }$ at $\nu = \pi / 2 .$ . The complete peak is therefore relevant to interpreting motion responsiveness and is included in the plotted range.

## D. Training and Rendering Costs

The discriminator adds feature-classification work during training. Rendering instead queries the learned feature branches, combines attribute responses, updates Gaussians, and performs splatting. A gate that scales four expert responses does not by itself imply sparse execution: evaluating all four responses still incurs all four branch costs. The framerate comparison measures the resulting pipeline and should not be interpreted as evidence of a particular dispatch policy.

Similarly, the parameter footprint includes the stored scene representation and its decoder, whereas optimization also uses activations and training-only components. Model storage, peak training memory, and rendering latency describe different resources. We report the available storage and timing measurements separately instead of combining them into a single efficiency score.

a) Resolution and capacity.: For six coordinate planes with comparable resolution $R _ { \ell }$ and $D _ { \ell }$ channels at level $\ell ,$ the stored feature count scales as

$$
N _ { \mathrm { p l a n e } } = \mathcal { O } \left( \sum _ { \ell = 1 } ^ { L } 6 R _ { \ell } ^ { 2 } D _ { \ell } \right) .\tag{26}
$$

A dense four-dimensional field at the same resolutions instead scales with $\sum _ { \ell } R _ { \ell } ^ { 4 } D _ { \ell }$ . The factorized query interface therefore changes how additional scene capacity can be stored, while the coordinate network and attribute branches add shared parameters. This scaling explains the motivation for retaining a compact grid rather than substituting a separate dense field at every time. It does not determine the measured model size, which also includes Gaussian attributes, decoder weights, and implementation choices.

b) Multilevel loss scale.: The coefficients $\omega _ { \ell }$ in Equation (9) emphasize the higher levels, but they are not probabilities over levels. In particular,

$$
\sum _ { \ell = 1 } ^ { L } \omega _ { \ell } = \frac { 2 ^ { L + 1 } - 2 - L } { 2 ^ { L } - 1 } .\tag{27}
$$

Their sum is generally different from one. Likewise, an $L _ { 1 }$ feature sum depends on the number of compared samples and channels, while a mean pixel loss uses a different normalization. Normalizing the outer coefficients in Equation (21) does not make the numerical scales of these objectives identical. Matching feature dimensions, sampling consistently, and monitoring the individual terms remain important when interpreting a joint optimization trajectory.

These distinctions also limit what can be concluded from a training-loss plot alone. A lower weighted objective can result from a change in its component balance and need not imply a proportional improvement in PSNR or temporal behavior. The experiments consequently compare rendered image metrics and separate expert diagnostics directly, while the representation analysis explains why the corresponding constraints are complementary.

## E. Expert Value and Resource Allocation

The assistance-value perspective of Markets of Minds [35] separates an expert’s potential usefulness from the difficulty of its input. A continuous reconstruction analogue would compare a default prediction $\widehat { I _ { 0 } }$ with a prediction $\widetilde { I _ { k } }$ receiving a candidate expert response:

$$
\Delta _ { k } ( q ) = \mathbb { E } \Big [ \ell ( \widehat { I } _ { 0 } , I ) - \ell ( \widehat { I } _ { k } , I ) \mid q \Big ] .\tag{28}
$$

Positive $\Delta _ { k }$ indicates expected error reduction; a response that degrades an already useful prediction has negative value. Allocation under a compute budget would additionally require execution costs, feasibility constraints, and estimates made before observing the assisted result.

This comparison clarifies the scope of the current gate. Its normalized coefficients control attribute responses but do not estimate $\Delta _ { k }$ or enforce a shared service-capacity constraint. The ablations measure average changes after removing components in their evaluation settings; they do not calibrate query-specific intervention values. Extending the model to budgeted execution would require such measurements and must account for interactions between attribute branches. The present results support coordinated feature learning, without assuming that a larger residual or gate weight certifies greater marginal value.

## V. EXPERIMENTS

## A. Evaluation Settings

a) Data and hardware.: We evaluate synthetic motion with D-NeRF [2] and real sequences with HyperNeRF [3] and DAVIS [42]. The aggregate comparison covers eight D-NeRF scenes, eight HyperNeRF scenes, and ten DAVIS sequences. Experiments use NVIDIA A100 GPUs. The compared methods are K-Planes [6], TiNeuVox-B [4], Hex-Plane [5], 4D-GS [9], MSTH [7], and DG-Mesh [17]. Baseline training follows the corresponding configurations used in the comparison.

b) Metrics.: We report PSNR, SSIM [43], and LPIPS [44] for image quality, together with training time, rendering frame rate, and model storage. Higher PSNR and SSIM and lower LPIPS indicate better image agreement. Results include the reported variation at the displayed precision. The aggregate table and the two ablation settings have different evaluation populations and are analyzed separately.

The attribute-expert experiment additionally reports opticalflow consistency error (FlowsME), temporal SSIM fluctuation (TSV), and elastic deformation error (EDE). These are diagnostic quantities from that setting, rather than additional terms in the image-quality average. That experiment uses AdamW with an initial learning rate of $3 \times 1 0 ^ { - 4 }$ and reports means over five runs.

## B. Reconstruction Quality and Efficiency

Table I reports the aggregate comparison. DimFlow achieves 39.73 dB PSNR, compared with 37.81 dB for 4D-GS and 38.32 dB for DG-Mesh. The corresponding improvements are 1.92 and 1.41 dB. DimFlow also reaches 0.98 SSIM and 0.02 LPIPS, matching the best values at the displayed precision. The improvements are therefore most clearly distinguished by PSNR; the rounded SSIM and LPIPS values contain several ties.

The frame rate is 85 FPS, equal to the 4D-GS result in this comparison. DimFlow uses 18.9 MB of storage, versus 18.3 MB for 4D-GS: a 0.6 MB increase, or about 3.3%. Its reported training time is eight minutes, compared with seven for 4D-GS. The quality gain consequently comes with a modest increase in training time and storage in this setting, while preserving the reported rendering rate. MSTH has the shortest reported training time, and 4D-GS has the smallest reported footprint.

The baselines illustrate different tradeoffs. K-Planes and TiNeuVox-B occupy the slower end of the reported rendering comparison, while the explicit Gaussian methods achieve higher frame rates. MSTH has no reported frame rate or storage in this table and is omitted from the speed scatter plot. Figure 5 visualizes the quality–speed relationship and separately summarizes component contributions; the two panels use different experimental settings.

a) Qualitative observations.: Figure 4 aligns the Cat and Dancer views using a common layout. The comparisons expose local appearance differences and changes around the moving body and silhouette. The ground-truth frames may themselves contain motion blur, so apparent sharpness alone is not a complete measure of reconstruction accuracy. These static views complement the numerical comparison; temporal claims are assessed through the expert diagnostics rather than inferred from a single image.

## C. Component Contributions

The component study in Table II uses the D-NeRF setting with a 39.73 dB full-model baseline. All five removals decrease PSNR and worsen LPIPS. Removing the coordinateexpert framework gives the largest PSNR decrease, 2.03 dB, and raises LPIPS from 0.02 to 0.05. The grid alone therefore does not reproduce the behavior of the coordinated featurelearning system in this experiment.

Removing adversarial alignment decreases PSNR by 1.88 dB, while disabling adaptive weighting decreases it by 1.61 dB. These ablations address different parts of the method: distribution alignment changes how the feature space is learned, whereas adaptive weighting changes which feature information reaches the decoder. Their effects should not be added, since each removal starts from the complete model and may also change how the remaining modules adapt.

Without temporal feature flow, PSNR decreases by 1.79 dB and LPIPS increases to 0.04. Removing bidirectional exchange gives a 1.71 dB decrease and 0.03 LPIPS. Together, the results support retaining both the forward integration of complementary features and the learning signals that connect those features across modules and time. They do not isolate an independently causal effect of every individual feedback operation.

TABLE I  
AGGREGATE RECONSTRUCTION QUALITY AND EFFICIENCY ACROSS D-NERF, HYPERNERF, AND DAVIS. BOLD MARKS THE BEST VALUE, INCLUDING TIES AT THE DISPLAYED PRECISION. TRAINING TIME IS IN MINUTES WITH VARIATION IN SECONDS; MISSING ENTRIES ARE UNREPORTED.
<table><tr><td>Method</td><td>PSNR ↑ dB</td><td> ${ \bf S S I M \uparrow }$ </td><td> $\mathrm { L P I P S } \downarrow$ </td><td> $\operatorname { T r a i n } \downarrow$   $\operatorname* { m i n } \pm \mathrm { ~ s ~ }$ </td><td> $\mathrm { F P S ~ \uparrow }$ </td><td>Storage ↓ MB</td></tr><tr><td>K-Planes [6]</td><td> $3 5 . 7 1 \pm \mathrm { 0 . 1 4 }$ </td><td> $0 . 9 7 \pm \mathrm { 0 . 0 0 }$ </td><td> $0 . 0 5 \pm 0 . 0 1$ </td><td> $4 9 \pm _ { 4 2 }$ </td><td> $0 . 9 7 \pm \mathrm { 0 . 0 0 }$ </td><td> $4 1 8 . 2 \pm 0 . 0 0$ </td></tr><tr><td>TiNeuVox-B [4]</td><td> $3 5 . 4 2 \pm 0 . 2 7$ </td><td> $0 . 9 7 \pm \mathrm { 0 . 0 1 }$ </td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td> $2 6 \pm { } 3 9$ </td><td> $1 . 5 \pm 0 . 0 0$ </td><td> $4 8 . 1 \pm 0 . 0 0$ </td></tr><tr><td>HexPlane [5]</td><td> $3 5 . 2 1 \pm \mathrm { 0 . 1 2 }$ </td><td> $0 . 9 7 \pm \mathrm { 0 . 0 0 }$ </td><td> $0 . 0 4 \pm 0 . 0 0$ </td><td> $1 0 \pm 2 7$ </td><td> $2 . 5 \pm 0 . 0 0$ </td><td> $3 7 . 9 \pm \mathrm { 0 . 0 0 }$ </td></tr><tr><td>4D-GS [9]</td><td> $3 7 . 8 1 \pm \mathrm { 0 . 1 1 }$ </td><td> $\mathbf { 0 . 9 8 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 0 2 \Pi _ { \pm 0 . 0 0 } }$ </td><td> ${ \bf 7 \pm 3 9 }$ </td><td> ${ \bf 8 5 \pm 0 . 0 0 }$ </td><td> ${ \bf 1 8 . 3 \pm 0 . 0 0 }$ </td></tr><tr><td>MSTH [7]</td><td> $3 4 . 8 8 \pm \mathrm { { 0 . 0 9 } }$ </td><td> $\mathbf { 0 . 9 8 \Pi _ { \pm 0 . 0 0 } }$ </td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td> ${ \bf 5 \pm } { \bf 1 8 }$ </td><td></td><td></td></tr><tr><td>DG-Mesh [17]</td><td> $3 8 . 3 2 \pm 0 . 1 3$ </td><td> $0 . 9 7 \pm \mathrm { 0 . 0 1 }$ </td><td> ${ \bf 0 . 0 2 \pm 0 . 0 0 }$ </td><td> $_ { 8 \pm 2 1 }$ </td><td> $6 8 \pm 0 . 0 0$ </td><td> $3 2 . 1 \pm 0 . 0 0$ </td></tr><tr><td>DimFlow</td><td> $\mathbf { 3 9 . 7 3 \bot 0 . 0 6 }$ </td><td> $\mathbf { 0 . 9 8 \Pi _ { \pm 0 . 0 0 } }$ </td><td> $\mathbf { 0 . 0 2 \Pi _ { \pm 0 . 0 0 } }$ </td><td> $_ { 8 \pm 3 2 }$ </td><td> ${ \bf 8 5 \pm 0 . 0 0 }$ </td><td> $1 8 . 9 \pm \mathrm { { 0 . 0 0 } }$ </td></tr></table>

4D-GS  
DimFlow  
DG-Mesh

Ground truth  
![](images/061c97ab546a716b137b065471aab339b42af9d9b64984e3ed922733c5c53386.jpg)

![](images/8a9623add35cbd233c3712c672b81c53de723d7fdb170d7dd297b0471256f8fc.jpg)

![](images/79a7ac91638b82d437199c5d96b6894ed11b2a273f5617337e0c21a514ee0d21.jpg)

![](images/9833ed9a1b9015f81a69609ff59bd843fc208817cdb976a25bc01589ad6172b2.jpg)  
Fig. 4. Qualitative reconstruction views. Cat and Dancer comparisons highlight local appearance, silhouettes, and motion boundaries. Ground-truth views can include motion blur. The complete eight-method comparison is available in the supplementary material.

(a) Reconstruction quality and speed  
![](images/f1d37ff0ce814e2d92a2e887f747a84a684cdb6c91adc0b68adc6459477935a7.jpg)

(b) Component contributions  
![](images/87c23ccd66064de2366d242cacb9bcbde66e1764425126ff665f6f73a90305df.jpg)  
Fig. 5. Quality, speed, and component effects. (a) Aggregate PSNR and frame rate from Table I; vertical error bars show the reported PSNR variation. (b) PSNR decrease for component removals in Table II. The panels summarize separate evaluation settings. MSTH is absent from (a) because its frame rate was not reported.

a) Feature-space view.: The projections in Figure 1 show greater visual overlap between grid and coordinateexpert samples with adversarial alignment. This observation is consistent with the intended distributional role of the discriminator, but a t-SNE projection is not a substitute for a discrepancy measure in the full feature space. The quantitative ablation supplies the corresponding reconstruction comparison.

TABLE II  
COMPONENT ABLATIONS ON D-NERF. ∆ IS THE PSNR DECREASE FROM THE FULL MODEL, CALCULATED FROM THE DISPLAYED MEANS.
<table><tr><td>Variant</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>∆</td></tr><tr><td>Full model</td><td> $\mathbf { 3 9 . 7 3 \pm 0 . 0 6 }$ </td><td> $\mathbf { 0 . 9 8 \mathop { \pm 0 . 0 0 } }$ </td><td> $\mathbf { 0 . 0 2 \Pi _ { \pm 0 . 0 0 } ^ { } }$ </td><td>0.00</td></tr><tr><td>w/o adversarial alignment</td><td> $3 7 . 8 5 \pm 0 . 1 0$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td><td> $0 . 0 4 \pm 0 . 0 1$ </td><td>1.88</td></tr><tr><td>w/o adaptive weighting</td><td> $3 8 . 1 2 \pm 0 . 0 8$ </td><td> $0 . 9 7 \pm \mathrm { 0 . 0 0 }$ </td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td>1.61</td></tr><tr><td>w/o temporal feature flow</td><td> $3 7 . 9 4 \pm 0 . 0 9$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td><td> $0 . 0 4 \pm 0 . 0 0$ </td><td>1.79</td></tr><tr><td>w/o bidirectional exchange</td><td> $3 8 . 0 2 \pm 0 . 0 7$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td>1.71</td></tr><tr><td>w/o coordinate expert</td><td> $3 7 . 7 0 \pm 0 . 1 1$ </td><td> $0 . 9 6 \pm 0 . 0 1$ </td><td> $0 . 0 5 \pm 0 . 0 1$ </td><td>2.03</td></tr></table>

TABLE III

ATTRIBUTE-EXPERT ABLATIONS. THE FULL-MODEL BASELINE IN THISSETTING IS 38.79 DB. ∆ PSNR IS CALCULATED RELATIVE TO THISBASELINE.
<table><tr><td>Metric</td><td>Full model</td><td>No deformation</td><td>No temporal</td></tr><tr><td>PSNR ↑</td><td>38.79</td><td>36.53</td><td>37.12</td></tr><tr><td>SSIM ↑</td><td>0.98</td><td>0.95</td><td>0.96</td></tr><tr><td>LPIPS ↓</td><td>0.03</td><td>0.05</td><td>0.04</td></tr><tr><td>FlowsME ↓</td><td>0.12</td><td>0.31</td><td>0.58</td></tr><tr><td>TSV ↓</td><td>0.015</td><td>0.028</td><td>0.030</td></tr><tr><td>EDE↓</td><td>0.08</td><td>0.21</td><td>0.16</td></tr><tr><td> $\Delta \mathrm { \ P S N R } \downarrow$ </td><td>0.00</td><td>2.26</td><td>1.67</td></tr></table>

## D. Deformation and Temporal Experts

The expert-specific study in Table III has a separate fullmodel baseline of 38.79 dB. Suppressing the deformation expert yields 36.53 dB, a decrease of 2.26 dB; suppressing the temporal expert yields 37.12 dB, a decrease of 1.67 dB. These differences are calculated directly from the values in that table. They are not measured against the 39.73 dB aggregate or component baseline.

Removing the deformation expert increases EDE from 0.08 to 0.21 and flow error from 0.12 to 0.31. Removing the temporal expert raises flow error to 0.58 and TSV from 0.015 to 0.030. The latter doubles the reported temporal fluctuation metric. The flow error becomes about 4.83× its baseline value, rather than a small percentage change. These changes distinguish a loss of geometric modeling capacity from a stronger degradation of temporal consistency.

The image metrics also worsen: deformation removal changes SSIM from 0.98 to 0.95 and LPIPS from 0.03 to 0.05; temporal removal gives 0.96 SSIM and 0.04 LPIPS. The specialists thus have overlapping consequences for rendered appearance while showing different sensitivities in the additional diagnostics. The evidence supports complementary attribute roles within this decoder, without implying that the four branches are statistically independent.

## VI. DISCUSSION AND LIMITATIONS

DimFlow depends on the information present in the input views and their camera poses. A coordinate prior and temporal regularity cannot recover unobserved motion uniquely, and excessive coupling can smooth a rapid change that should be preserved. Occlusion boundaries, substantial topology changes, and fluid-like motion remain difficult cases. The coordinate expert supplies a learned representation, not a guarantee of physical consistency.

![](images/caa70dd61321abc5bde2e1ab9992edaa56b14e074f919a249c83b9ce508e21a3.jpg)

![](images/ebeab288aaf74fe655c6bcf856c69342737bc527292858a12d591d25703bee5a.jpg)  
Fig. 6. Expert-specific sensitivity. PSNR and optical-flow consistency error from Table III. “No def.” removes the deformation expert; “No temp.” removes the temporal expert. The full-model baseline here is distinct from the aggregate comparison.

The aggregate image metrics do not reveal every sequencelevel failure. Likewise, a feature projection cannot prove that individual Gaussian trajectories are correct. Frame rate and storage depend on image resolution, the number of primitives, and the feature configuration; the reported aggregate should not be interpreted as deployment latency for every camera setup. Applications to mapping, navigation, or closed-loop control require additional task-specific evaluation.

Evaluation must also match the intended task. The NTIRE super-resolution challenge [45] distinguishes reconstruction fidelity from perceptual quality; scaling laws for deepfake detection [46] study coverage of real domains and forgery methods. These criteria address different failure modes from dynamic view synthesis. The current PSNR comparison does not establish perceptual preference, forensic generalization, or causal future prediction. Calibrated expert usefulness, broader sparse-view evaluation, and richer motion priors remain useful extensions.

Global image metrics and local expert value operate at different scales. Paired interventions, holding the decoder and camera view fixed, could measure both reconstruction gains and response costs, helping distinguish useful specialization from compensation between attribute branches. Any routing policy would additionally require validation across motion regimes and camera trajectories.

## VII. CONCLUSION

DimFlow connects a compact grid representation with coordinate-expert learning through feature-level alignment and bidirectional exchange. Adaptive fusion, residual feedback, and temporal processing organize the features supplied to specialized dynamic-attribute branches. The aggregate results improve PSNR while retaining the reported rendering speed of 4D-GS, with a small increase in storage. Component and expert ablations identify complementary contributions from representation learning and dynamic specialization. These findings support treating feature interaction as a central part of efficient dynamic Gaussian reconstruction.

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” 2020. [Online]. Available: https://arxiv.org/abs/2003.08934

[2] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “D-NeRF: Neural radiance fields for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 10 318–10 327. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2021/html/Pumarola\_ D-NeRF\_Neural\_Radiance\_Fields\_for\_Dynamic\_Scenes\_CVPR\_ 2021\_paper.html

[3] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, and S. M. Seitz, “Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields,” 2021. [Online]. Available: https://arxiv.org/abs/2106.13228

[4] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. Nießner, and Q. Tian, “Fast dynamic radiance fields with time-aware neural voxels,” in SIGGRAPH Asia 2022 Conference Papers, ser. SA ’22. ACM, Nov. 2022, pp. 1–9. [Online]. Available: http://dx.doi.org/10.1145/3550469.3555383

[5] A. Cao and J. Johnson, “HexPlane: A fast representation for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 130–141. [Online]. Available: https: //openaccess.thecvf.com/content/CVPR2023/html/Cao\_HexPlane\_A\_ Fast\_Representation\_for\_Dynamic\_Scenes\_CVPR\_2023\_paper.html

[6] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, “K-Planes: Explicit radiance fields in space, time, and appearance,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 479–12 488. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2023/ html/Fridovich-Keil\_K-Planes\_Explicit\_Radiance\_Fields\_in\_Space\_ Time\_and\_Appearance\_CVPR\_2023\_paper.html

[7] F. Wang, Z. Chen, G. Wang, Y. Song, and H. Liu, “Masked space-time hash encoding for efficient dynamic scene reconstruction,” in Advances in Neural Information Processing Systems, 2023. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2023/hash df31126302921ca9351fab73923a172f-Abstract-Conference.html

[8] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3D gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph 3d-gaussian-splatting/

[9] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4D gaussian splatting for real-time dynamic scene rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 310–20 320. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/html/Wu\_ 4D\_Gaussian\_Splatting\_for\_Real-Time\_Dynamic\_Scene\_Rendering\_ CVPR\_2024\_paper.html

[10] K. Zhang, G. Riegler, N. Snavely, and V. Koltun, “Nerf++: Analyzing and improving neural radiance fields,” 2020. [Online]. Available: https://arxiv.org/abs/2010.07492

[11] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields,” 2021. [Online]. Available: https://arxiv.org/abs/2103.13415

[12] Z. Li, S. Niklaus, N. Snavely, and O. Wang, “Neural scene flow fields for space-time view synthesis of dynamic scenes,” 2021. [Online]. Available: https://arxiv.org/abs/2011.13084

[13] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance fields,” 2022. [Online]. Available: https://arxiv.org/abs/2203. 09517

[14] J. Zhang, X. Guo, K. Cai, Q. Lv, Y. Fan, W. Chai, J. Wang, and K. Wang, “Hybrid token compression for vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 31 889–31 899. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2026/html/ Zhang\_Hybrid\_Token\_Compression\_for\_Vision-Language\_Models\_ CVPR\_2026\_paper.html

[15] J. Zhang, W. Wang, L. Cai, L. Yuan, Y. Wang, and M.-H. Yang, “Video encoders built on image representations,” 2026, arXiv preprint arXiv:2610.06616. [Online]. Available: https://arxiv.org/abs/2610.06616

[16] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,” 2023. [Online]. Available: https://arxiv.org/abs/2309.13101

[17] I. Liu, H. Su, and X. Wang, “Dynamic gaussians mesh: Consistent mesh reconstruction from dynamic scenes,” in International Conference on Learning Representations, 2025. [Online]. Available: https://arxiv.org/abs/2404.12379

[18] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, “Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,” 2024. [Online]. Available: https://arxiv.org/abs/2312.07920

[19] S. Zhu, G. Wang, X. Kong, D. Kong, and H. Wang, “3d gaussian splatting in robotics: A survey,” 2024. [Online]. Available: https://arxiv.org/abs/2410.12262

[20] J. Yao, J. Zhang, X. Pan, T. Wu, and C. Xiao, “DepthSSC: Monocular 3D semantic scene completion via depth-spatial alignment and voxel adaptation,” in Proceedings of the Winter Conference on Applications of Computer Vision, 2025, pp. 2154–2163. [Online]. Available: https://openaccess.thecvf.com/content/WACV2025/html/ Yao\_DepthSSC\_Monocular\_3D\_Semantic\_Scene\_Completion\_via\_ Depth-Spatial\_Alignment\_and\_WACV\_2025\_paper.html

[21] J. Zhang, Y. Fan, Z. Wen, J. Wang, and K. Wang, “3D-Agent: A tri-modal multi-agent responsive framework for comprehensive 3D object annotation,” in Advances in Neural Information Processing Systems, vol. 38, 2025. [Online]. Available: https://proceedings.nips.cc/paper\_files/paper/2025/hash/ 6030db5195150ac86d942186f4abdad8-Abstract-Conference.html

[22] J. Zhang, Q. Lyu, R. Gao, E. Liu, T. Mo, Z. Liang, S. Liu, X. Wu, X. Li, W. Wang, G. Wang, K. Wang, T. Chen, P. Torr, and L. Lin, “The LoViF 2026 ActPhysCause challenge: A baseline report on action-conditioned video world models,” ECCV Workshop on Low-Level Vision Frontiers (LoViF), challenge report, 2026. [Online]. Available: https://openreview.net/forum?id=vbxZVVKuxh

[23] D. Eigen, M. Ranzato, and I. Sutskever, “Learning factored representations in a deep mixture of experts,” 2014. [Online]. Available: https://arxiv.org/abs/1312.4314

[24] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. S. Pinto, D. Keysers, and N. Houlsby, “Scaling vision with sparse mixture of experts,” 2021. [Online]. Available: https://arxiv.org/abs/2106.05974

[25] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” 2022. [Online]. Available: https://arxiv.org/abs/2101.03961

[26] Z. Mi and D. Xu, “Switch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields,” in International Conference on Learning Representations, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:259298532

[27] F. D. Sario, R. Renzulli, E. Tartaglione, and M. Grangetto, “Boost your nerf: A model-agnostic mixture of experts framework for high quality and efficient rendering,” 2024. [Online]. Available: https://arxiv.org/abs/2407.10389

[28] J. Zhang, Z. Huang, Y. Fan, N. Liu, M. Li, Z. Yang, J. Yao, J. Wang, and K. Wang, “KABB: Knowledge-aware bayesian bandits for dynamic expert coordination in multi-agent systems,” in International Conference on Machine Learning, 2025. [Online]. Available: https://arxiv.org/abs/2502.07350

[29] J. Zhang, Y. Fan, K. Cai, X. Sun, and K. Wang, “OSC: Cognitive orchestration through dynamic knowledge alignment in multi-agent LLM collaboration,” in Findings of the Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguistics, 2025, pp. 6320–6337. [Online]. Available: https://aclanthology.org/2025.findings-emnlp.335/

[30] J. Zhang, Y. Fan, W. Lin, R. Chen, H. Jiang, W. Chai, J. Wang, and K. Wang, “GAM-Agent: Game-theoretic and uncertainty-aware collaboration for complex visual reasoning,” in Advances in Neural Information Processing Systems, vol. 38, 2025. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ d38512ba4121046784528ad9984af74a-Abstract-Conference.html

[31] J. Zhang, Y. Fan, K. Cai, J. Yang, J. Yao, J. Wang, G. Qu, Z. Chen, and K. Wang, “Why keep your doubts to yourself? trading visual uncertainties in multi-agent bandit systems,” in International Conference on Learning Representations, 2026. [Online]. Available: https://arxiv.org/abs/2601.18735

[32] J. Zhang, K. Cai, Y. Fan, N. Liu, and K. Wang, “MAT-agent: Adaptive multi-agent training optimization,” in The Thirty-ninth

Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=YDWRTYgR79

[33] J. Zhang, K. Cai, Q. Zeng, N. Liu, S. Fan, Z. Chen, and K. Wang, “Failure-driven workflow refinement,” 2025, arXiv preprint arXiv:2510.10035. [Online]. Available: https://arxiv.org/abs/2510.10035

[34] J. Zhang, Y. Fan, K. Cai, Z. Huang, X. Sun, J. Wang, C. Tang, and K. Wang, “DrDiff: Dynamic routing diffusion with hierarchical attention for breaking the efficiency-quality trade-off,” in Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2025, pp. 9320–9340. [Online]. Available: https://aclanthology.org/2025.emnlp-main.474/

[35] J. Zhang, X. Wu, W. Wang, M. Wang, and M.-H. Yang, “Markets of minds: A 51-billion-state economy of intelligence,” 2026, arXiv preprint arXiv:2610.32190. [Online]. Available: https://arxiv.org/abs/2610.32190

[36] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” 2014. [Online]. Available: https://arxiv.org/abs/1406.2661

[37] B. Roessle, N. Müller, L. Porzi, S. R. Bulò, P. Kontschieder, and M. Nießner, “Ganerf: Leveraging discriminators to optimize neural radiance fields,” ACM Trans. Graph., vol. 42, no. 6, nov 2023. [Online]. Available: https://doi.org/10.1145/3618402

[38] M. Shahbazi, E. Ntavelis, A. Tonioni, E. Collins, D. P. Paudel, M. Danelljan, and L. V. Gool, “Nerf-gan distillation for efficient 3d-aware generation with convolutions,” 2023. [Online]. Available: https://arxiv.org/abs/2303.12865

[39] J. Zhang, K. Cai, Y. Fan, J. Wang, and K. Wang, “CF-VLM: Counterfactual vision-language fine-tuning,” in Advances in Neural Information Processing Systems, vol. 38, 2025. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 541b6d155146d142a5e10d787d1d430f-Abstract-Conference.html

[40] J. Zhang, K. Cai, X. Guo, S. Liu, Q. Lv, R. Chen, J. Yang, Y. Fan, X. Sun, J. Wang, Z. Chen, L. Lin, and K. Wang, “MM-CoT: A benchmark for probing visual chain-of-thought reasoning in multimodal models,” 2025, arXiv preprint arXiv:2512.08228. [Online]. Available: https://arxiv.org/abs/2512.08228

[41] J. Zhang, K. Cai, J. Yang, J. Wang, C. Tang, and K. Wang, “Top-down semantic refinement for image captioning,” 2025, arXiv preprint arXiv:2510.22391. [Online]. Available: https://arxiv.org/abs/2510.22391

[42] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung, “A benchmark dataset and evaluation methodology for video object segmentation,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 724–732.

[43] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[44] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” 2018. [Online]. Available: https://arxiv.org/abs/1801.03924

[45] Z. Chen, K. Liu, J. Wang, X. Yan, J. Li, Z. Zhang, J. Gong, J. Li, L. Sun, X. Liu, R. Timofte, Y. Zhang et al., “The fourth challenge on image super-resolution (×4) at NTIRE 2026: Benchmark results and method overview,” 2026, arXiv preprint arXiv:2604.14558. [Online]. Available: https://arxiv.org/abs/2604.14558

[46] W. Wang, J. Zhang, L. Cai, T. Xiao, Y. Wang, and M.-H. Yang, “Scaling laws for deepfake detection,” 2025, arXiv preprint arXiv:2510.16320. [Online]. Available: https://arxiv.org/abs/2510.16320