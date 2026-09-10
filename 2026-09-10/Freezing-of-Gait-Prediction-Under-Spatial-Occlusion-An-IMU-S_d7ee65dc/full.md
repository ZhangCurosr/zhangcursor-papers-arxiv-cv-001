# Freezing of Gait Prediction Under Spatial Occlusion: An IMU-Supervised Cross-Modal Distillation Approach

Chandan Biswas D NeuroAI Fusion Labs Kolkata, India chandan@neuroailabs.in

Aryan Singh   
NeuroAI Fusion Labs   
Kolkata, India   
aryan@neuroailabs.in

Anabik Pal Indian Institute of Science Education and Research Berhampur, Odisha anabikpal@iiserbpr.ac.in

Abstract—Parkinson’s disease is a progressive neurodegenerative disorder in humans characterised by the gradual deterioration of movement control. Automated freezing-of-gait (FOG) detection supports the objective assessment of gait-related motor impairment. Two common approaches are used for the FOG prediction: (i) analysing video recordings of the patient’s movements and (ii) analysing data collected using inertial measurement unit (IMU) wearable sensors attached to the patient’s lower limbs. Video-based approaches may suffer from detection errors during continuous turning-in-place tasks because the lower limbs undergo substantial geometric self-occlusion, which can degrade pose-estimation accuracy. In contrast, IMU-based approaches are generally less affected by visual occlusion; however, they can be difficult to deploy outside clinical or laboratory settings because the sensors must be attached securely and remain in place throughout the assessment.

Motivated by this, we propose a cross-modal subspace distillation framework to mitigate the limitations of unimodal FOG detection by combining IMU accuracy with video-based practicality. Specifically, we extract invariant latent topologies from a pretrained kinematic oracle to structurally supervise a non-encoded visual architecture during training. To resolve periods of severe spatial occlusion, a dual-stream visual model probabilistically fuses skeletal graph nodes and continuous spatial pixels, dynamically shifting reliance to uninterrupted pixel boundaries precisely as joint tracking confidence drops. Evaluated against a public, multi-modal sequence dataset of Parkinson’s individuals executing continuous 360<sup>◦</sup> turns, empirical results demonstrate that applying sensory boundary topologies strictly mitigates tracking evaluation entropy. Consequently, our constrained optimisation confirms that highly precise FOG prediction bounds can be achieved over zero-wearable inference environments.

Index Terms—Freezing of Gait, Cross-Modal Distillation, Geometric Self-Occlusion, Supervised Contrastive Learning, Zero-Wearable Inference.

## I. INTRODUCTION

Parkinson’s disease (PD) is a progressive neurodegenerative disorder that leads to cardinal motor symptoms (bradykinesia, resting tremor, rigidity, postural instability) and frequent gait disturbances such as freezing of gait. This disease is the second most common neurodegenerative disease, and the WHO estimated that, in 2019, PD resulted in 5.8 million disability-adjusted life years, an increase of 81% since 2000, and caused 329,000 deaths, an increase of over 100% since

![](images/99b24287603c8aef9d692159da31a14b826da2a9dfbe38215f337e8bb291c32e.jpg)  
Fig. 1: Manifestation of visual self-occlusion during a turning-inplace protocol.

2000 <sup>1</sup>. Prolonged FOG episodes correlate strongly with a higher probability of falls and injuries. Objective assessment of FOG is an important clinical requirement for evaluating motor symptom progression in PD.

Recent research shows that technology focused on estimating spatial movement trajectories using physical sensors provides reliable, high-precision FOG assessment results. However, it is challenging to deploy such technology outside controlled laboratory conditions, as the IMU affixed to patients’ lower limbs are uncomfortable to wear continuously. In contrast, smartphone-based video acquisition for analysis is a viable option for easy deployment. However, deriving diagnostic kinematic measurements solely from continuous optical fields can introduce substantial errors, particularly during clinical evaluations focused on turning-in-place tasks. As demonstrated in Figure 1, calculating spatial joint geometry via camera hardware introduces physical uncertainties during periods of spatial self-occlusion. Specifically, when one leg rotates in front of the other, standard skeletal joint algorithms estimate coordinates with diminished confidence limits, result ing in signal noise. A standard learning classifier, completely deprived of deterministic structural points, fails to resolve subtle FOG kinematic characteristics across obscured visual bounds.

To construct an effective defence against these physical ambiguities without assuming an extensive increase in hardware parameters, we formulate an alternative approach exploiting information from distinct data domains. Instead of relying solely on explicit optical patterns to infer motion properties, our workflow utilizes a synchronized multi-modal representation logic during the training phase. We postulate that leveraging a verified hardware signal—such as an IMU embedding subspace containing high task dependency correlations—acts as an explicit mapping mechanism. By projecting a mathematically regularized multi-task constraint, we train an underlying probabilistic video classifier to approximate an expert target metric space rather than modelling individual corrupted geometric constraints sequentially. By applying cross-modal latent alignment, the parametrizations acquired on the unified representations constrain visual and tabular classification functions structurally analogous to sensory arrays. Consequently, the diagnostic assessment process performs uninterrupted inference estimates, completely decoupled from explicit tracking boundaries, without imposing direct dependencies on sensor input.

Contributions: In summary, the key contributions of this paper are as follows:

• We detail a cross-modal subspace alignment training pipeline transferring domain properties from 1-D spatial kinematics directly to unannotated spatio-temporal video subsets.

• We introduce an adaptive structural evaluation limit via parameter-weight fusion distributions based explicitly upon dynamic skeleton coordinate confidences (to solve systematic lower limb tracking corruption conditions probabilistically).

• We conduct evaluation experiments isolating metrics resilient to highly sparse clinical variables demonstrating empirical event classification bounds nearing upper bounds measured over attached sensors independent of external observation limitations.

## II. RELATED WORK

This section reviews related work across three research directions: Kinematic and Vision-Based FOG prediction, Latent Space Alignment and Cross-modal Distillation, and Coordinate Representation under Visual Uncertainty.

## A. Kinematic and Vision-Based FOG prediction

Quantifying rapid temporal movement artifacts, such as the 3 − 8 Hz festination signatures characteristic of Freezing of Gait (FOG), typically utilizes hardware sensor networks. Deep learning evaluations of Inertial Measurement Unit (IMU) arrays validate high correlation limits to medical benchmarks [1]. Operating directly on physical inertial bounds allows standard classification margins to routinely exceed a predictive fidelity of 90% without subjective filtering [2]. The public repository formalized by [3] verifies these operational limits but reiterates the inherent deployment friction regarding continuous device attachment at inference time.

Transitioning from sensory to markerless video limits restricts objective spatial observation [4]. Studies examining RGB frame derivations generally constrain visual input to strictly frontal or sagittal movement vectors, minimizing temporal coordinate crossing. Applying analogous evaluation paradigms directly to non-linear sequences—such as a dynamic 360<sup>◦</sup> continuous rotational condition—causes systematic observation breakdown within unmodified computer vision classifiers due to an inability to mathematically reconcile heavy geometric occlusion points.

## B. Latent Space Alignment and Cross-modal Distillation

When disparate observational domains generate unequal functional precision, restricting structural entropy in the subordinate domain improves limits generalization. The mechanism of parameter distillation, fundamentally defining limits by transferring distribution outputs across heterogeneous modalities [5], constitutes the mathematical baseline of teacherstudent topologies. Scaling beyond logit regressions, joint representation environments evaluate probability estimations utilizing symmetrical contrastive distance [6]. Applied specifically to action recognition, architectures minimizing vector variations mapping RGB streams into analogous positional domains—such as Audio or skeletal topologies [7]—enable zerodependency modality inference limits. In standard literature, dual-stream architectures process components conditionally with equal likelihood metrics. We extend this by configuring an inherently rigid IMU embedding as an invariant parameter subset, constraining sequence derivations purely based upon a rigid spatial Oracle alignment formulated directly across dynamic turning vectors.

## C. Coordinate Representation under Visual Uncertainty

Defining physical topologies for non-pixel networks usually adopts coordinate representations encoded mathematically via Spatio-Temporal Graph Convolutional Networks (ST-GCN) [8]. Baseline ST-GCN parameters assume an uninterrupted series of structural input locations modeling adjacency weights conditionally over fixed node matrices. Such modeling paradigms degrade strictly monotonically when node estimations operate beneath probabilistic noise bounds. Deterministic graph propagation does not possess dynamic bounds resolving variables when target confidence margins estimate null configurations (C ≈ 0). Work regarding noisy topological extraction normally operates through mathematical imputation bounds, modeling linear node continuation states independent of raw pixel parameters [9]. Diverging from rigid topological corrections, our parameter configuration accepts spatial uncertainty physically through expected node activation ranges. Treating spatial topology deterministically across unobstructed bounds and relying proportionately upon continuous raw matrices exclusively when tracking limits fall below minimal prediction margins standardizes objective extraction through complete blockage intervals.

## III. PROPOSED METHOD

In this section, we describe the details of our proposed cross-modal learning framework for FOG prediction. The framework is designed to train a vision-and-language model

under the strict supervision of an informative kinematic subspace (derived from inertial sensors) to compensate for visual information loss during periods of severe physical occlusion.

## A. Problem Formulation and Synchronisation

Let $\mathcal { D } = \{ ( \mathbf { X } _ { V } ^ { ( i ) } , \mathbf { X } _ { I } ^ { ( i ) } , \mathbf { x } _ { T } ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ be the synchronised training dataset consisting of N data instances. For each data instance i, $\mathbf { X } _ { V } ^ { ( i ) } \in \mathbb { R } ^ { F \times H \times W \times 3 }$ represents a sequence of $F$ visual frames in a spatial resolution of $H \times W$ . Let $\mathbf { X } _ { I } ^ { ( i ) } \in \mathbb { R } ^ { S \times 6 }$ represent the temporally synchronised sequence of S inertial measurements from the triaxial accelerometer and gyroscope.

To introduce contextual metadata (e.g., age, medication state, clinical rating scale) into the formulation, let $\mathbf { x } _ { T } ^ { ( i ) }$ represent an encoded text prompt denoting the patient’s clinical profile. The variable $y ^ { ( i ) } \in \{ 0 , 1 \}$ } represents the ground-truth cluster label for the FOG class. During inference on the client side, exclusively the visual sequence $\mathbf { X } _ { V }$ is available. The tabular clinical metadata $\left( { { \bf { x } } _ { T } } \right)$ is strictly unavailable during deployment to guarantee fully unobtrusive, zero-dependency evaluations. Consequently, the research objective is to estimate a parameterized transformation of the visual inputs such that they project onto an informative kinematic and clinical latent subspace, optimizing the prediction of $\boldsymbol y ^ { ( i ) }$ without requiring continuous sensory or tabular inputs at inference.

## B. The Kinematic Teacher Oracle

Due to the physical characteristics of the FOG condition, a dedicated kinematic representation contains a highly discriminative subspace. We leverage an independently pre-trained deep neural network as an expert extractor. Let $E _ { I } ( \cdot ; \theta _ { I } )$ denote the function projecting the inertial inputs into a $p \textmd { - }$ dimensional Euclidean space as follows:

$$
\mathbf { z } _ { I } = E _ { I } ( \mathbf { X } _ { I } ; \theta _ { I } ) , \quad \mathbf { z } _ { I } \in \mathbb { R } ^ { p }\tag{1}
$$

Throughout our multi-modal training procedure, we treat the parameters $\theta _ { I }$ as constant (frozen). The objective is to utilise $\mathbf { z } _ { I }$ as the reference vector capturing complete kinematic properties, specifically forcing the latent representations generated by the non-encoded vision and textual networks $( \mathbf { z } _ { V }$ and ${ \bf z } _ { T } )$ to align geometrically with this hardware boundary during optimisation.

## C. Probabilistic Dual-Stream Vision Model

Visual assessment of the turning-in-place task suffers from frequent self-occlusions, particularly when one leg crosses the other. Because a strict deterministic mapping from skeletal coordinates to a label introduces error during periods of heavy occlusion, we frame the extraction of visual features from a probabilistic perspective.

Let the visual stream be composed of two parallel embeddings. A spatio-temporal Convolutional 3D model $( E _ { r g b } )$ computes a latent representation over dense pixels yielding $\mathbf { v } _ { r g b } \in \mathbb { R } ^ { p }$ . Simultaneously, a skeleton pose estimator extracts a graph of joint coordinates over $F$ frames. We use a Graph Convolutional Network $( E _ { s k } )$ to yield $\mathbf { v } _ { s k } \in \mathbb { R } ^ { p }$

However, since pose tracking fails systematically under selfocclusion, treating $\mathbf { v } _ { s k }$ equally across all sequences leads to information degradation. We consider the skeletal output to be corrupted by noise, modelled as an uncertainty random variable. Let $c _ { j } ^ { t } ~ \in ~ [ 0 , 1 ]$ be the observed confidence score from the pose estimation framework for the $j ^ { \mathrm { t h } }$ tracking joint at frame t. We hypothesise that the occurrence of valid topological information follows a distribution proportionate to the observed coordinate confidence. We model the estimated validity $C _ { V }$ of a window of skeletal data as a simple expected confidence value:

$$
\mathbb { E } [ C _ { V } ] = \frac { 1 } { F \cdot K } \sum _ { t = 1 } ^ { F } \sum _ { j = 1 } ^ { K } c _ { j } ^ { t }\tag{2}
$$

where $K$ refers to the number of skeletal tracking nodes. We thus treat the actual feature vector for the given data instance as an expected value from a weighted mixture between the dense pixel stream (resistant to missing nodes) and the skeletal stream (susceptible to occlusion) yielding the unified vision embedding $\mathbf { z } _ { V } \in \mathbb { R } ^ { p }$ , formally defined as:

$$
\mathbf { z } _ { V } = \alpha \cdot \mathbf { v } _ { s k } + ( 1 - \alpha ) \cdot \mathbf { v } _ { r g b } ,\tag{3}
$$

where the combination coefficient α is derived from a parameterized sigmoid transformation evaluated over the tracking validity $\mathbb { E } [ C _ { V } ]$

$$
\alpha = \frac { 1 } { 1 + \exp ( - ( \omega _ { c } \cdot \mathbb { E } [ C _ { V } ] + b _ { c } ) ) } ,\tag{4}
$$

where $\omega _ { c }$ and $b _ { c }$ act as explicitly trainable scaling and shift parameters updated alongside Θ. As geometric spatial overlap deprives the skeleton matrices of continuous resolution bounds $( \mathbb { E } [ C _ { V } ]$ is low), the adaptive scalar threshold evaluates past shifting thresholds causing α to systematically decay towards zero. Consequently, $\mathbf { z } _ { V }$ transitions mapping inference boundaries dynamically into unbroken pixel topologies.

## D. Supervised Cross-Modal Subspace Distillation

To address the situation where explicitly annotated temporal bounds of FOG micro-movements are difficult to estimate purely from video, we supervise the fused vision embedding $\mathbf { z } _ { V }$ using the kinematics oracle $\mathbf { z } _ { I }$ . In parallel, let $\mathbf { z } _ { T } \in \mathbb { R } ^ { p }$ denote a vector obtained from $E _ { t x t } ( \mathbf { x } _ { T } ; \boldsymbol { \theta } _ { T } )$ , modeling the metadata subspace of the text features.

Applying standard contrastive alignment unconditionally across random batch segments generates false negative structural penalization, where distinct sequence bounds matching equivalent target pathology (i.e., multiple FOG blocks within the same batch) mathematically repulse. To resolve this, we learn a parameterized joint-space function applying a multitarget Supervised Contrastive Loss (SupCon).

Let ${ \mathcal { P } } ( i ) ~ = ~ \{ k ~ \in ~ \{ 1 . . . M \} ~ | ~ y ^ { ( k ) } ~ = ~ y ^ { ( i ) } \}$ define the explicit index set identifying matching inference classes relative to an evaluated anchor instance (i) spanning a batch of size M. Integrating the cardinality $| \mathcal { P } ( i )$ |, we define the temperature-scaled similarity alignment function targeting the

IMU representation subspace independent of inter-instance isolation penalties as:

$$
\mathcal { L } _ { s u p } ^ { V  I } = \sum _ { i = 1 } ^ { M } \frac { - 1 } { | \mathcal { P } ( i ) | } \sum _ { k \in \mathcal { P } ( i ) } \log \frac { \exp ( \mathbf { z } _ { V } ^ { ( i ) } \cdot \mathbf { z } _ { I } ^ { ( k ) } / \tau ) } { \sum _ { j = 1 } ^ { M } \exp ( \mathbf { z } _ { V } ^ { ( i ) } \cdot \mathbf { z } _ { I } ^ { ( j ) } / \tau ) } .\tag{5}
$$

Similarly, to force the vision manifold to become sensitive to varying states of the clinical profiles of the patient cohort without class collisions, we perform simultaneous latent space alignment with respect to the text component:

$$
\mathcal { L } _ { s u p } ^ { V  T } = \sum _ { i = 1 } ^ { M } \frac { - 1 } { | \mathcal { P } ( i ) | } \sum _ { k \in \mathcal { P } ( i ) } \log \frac { \exp ( \mathbf { z } _ { V } ^ { ( i ) } \cdot \mathbf { z } _ { T } ^ { ( k ) } / \tau ) } { \sum _ { j = 1 } ^ { M } \exp ( \mathbf { z } _ { V } ^ { ( i ) } \cdot \mathbf { z } _ { T } ^ { ( j ) } / \tau ) } ,\tag{6}
$$

where τ serves as a temperature scaling scalar influencing the penalty range of the estimated likelihoods of unaligned features.

Finally, a classifier matrix $\Theta _ { C }$ maps the distilled representation to the set of desired target cluster labels representing FOG classes. We optimize the sum of standard cross-entropy and the supervised alignment constraints. Formally speaking, given the FOG class $y ^ { ( i ) }$ , the unified loss formulation executed in a back-propagation gradient scheme translates to:

$$
J ( \Theta ) = - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \log P ( y ^ { ( i ) } | \mathbf { z } _ { V } ^ { ( i ) } ; \Theta _ { C } ) + \lambda \mathcal { L } _ { s u p } ^ { V  I } + \gamma \mathcal { L } _ { s u p } ^ { V  T }\tag{7}
$$

where λ and $\gamma \in [ 0 , 1 ]$ refer to penalty combination constants parameterizing the level of forced mutual information constraint applied from the oracle distributions onto the primary training mechanism. Post optimization of $J ( \Theta )$ , during validation operations executed on non-encoded client environments, the dependencies on both the inertial streams $( { \bf z } _ { I } )$ and explicit tabular clinical profiles $( { \bf z } _ { T } )$ are entirely bypassed. The classification matrix $\Theta _ { C }$ relies solely on the distilled visual topology $\mathbf { z } _ { V }$ , which has mathematically absorbed the expected bounds of the unavailable Oracle domains.

## E. Algorithmic Optimisation Formulation

The objective of our optimisation procedure is to yield a vision-based predictor parameter space $\boldsymbol { \Theta } = \{ \theta _ { r g b } , \theta _ { s k } , \Theta _ { C } \}$ that operates independently of inertial measurements during the inference stage. We aggregate the mathematical formalisations constructed across Equations 1 to 7 into a cohesive batch-wise update workflow.

Algorithm 1 details the explicit procedure for modelling the expectations under self-occlusion conditions during K-means assignment analogies and aligning the subsequent subspace manifolds. For any given mini-batch of size M, operations mapping sequence topologies to coordinate configurations iteratively estimate bounding limits derived from the invariant hardware oracle constants, $\theta _ { I }$ and $\theta _ { T }$ . During each parameter evaluation loop, a subset calculation strictly regulates confidence vectors proportionate to available temporal unoccluded node observations (lines 9-10). Subsequent backward accumulation formulates gradient dependencies entirely supervised by both spatial coordinate variance and latent kinematic states defined within the unified joint objective scalar J(Θ).

```powershell
Algorithm 1: Cross-modal Informative Subspace Dis
tillation Framework
Input: Synchronised multi-modal dataset
$\bar { \mathcal { D } } = \{ ( \mathbf { X } _ { V } ^ { ( i ) } , \mathbf { X } _ { I } ^ { ( i ) } , \mathbf { x } _ { T } ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ , Batch size
M, Max training epochs T, Optimiser learning
rate $\eta ,$ InfoNCE temperature τ, Objective
weighting parameters $\lambda , \gamma$
Output: Optimised classification target parameter
states $\boldsymbol { \Theta } = \{ \theta _ { r g b } , \theta _ { s k } , \Theta _ { C } \}$
Initialise Θ from normal random uniform bounds
Set auxiliary parameters $\theta _ { I } , \theta _ { T }$ mapping to Oracle $E _ { I }$
and $E _ { t x t }$ to constant vectors → (requires $\nabla = \mathbf { 0 } )$
for $t = 1 , \dots , T$ do
for each batch subset $B \subset D$ such that $| B | = M$
do
for $i = 1 , \dots , M$ do
// Phase I: Constant Subspace
Extractions
$\mathbf { z } _ { I } ^ { ( i ) }  E _ { I } ( \mathbf { X } _ { I } ^ { ( i ) } ; \boldsymbol { \theta } _ { I } )$ // Constant
inertial target metric
projection
$\mathbf { z } _ { T } ^ { ( i ) }  E _ { t x t } ( \mathbf { x } _ { T } ^ { ( i ) } ; \boldsymbol { \theta } _ { T } )$ // Constant
clinical string topology
formulation
// Phase II: Probabilistic
Visual Encoding Extraction
$\mathbf { v } _ { r g b } ^ { ( i ) }  E _ { r g b } ( \mathbf { X } _ { V } ^ { ( i ) } ; \theta _ { r g b } )$
$\mathbf { v } _ { s k } ^ { ( i ) } , \mathcal { C } ^ { ( i ) } \gets E _ { s k } ( \mathbf { X } _ { V } ^ { ( i ) } ; \theta _ { s k } ) \quad / /$ where C
encapsulates bounded spatial
scores $\{ c _ { j } \}$
$/ /$ Phase III: Estimation of
Uncertainty due to
Occlusions
$\begin{array} { r } { \mathbb { E } [ C _ { V } ] ^ { ( i ) } \longleftarrow \frac { 1 } { F \cdot K } \sum _ { f = 1 } ^ { F } \sum _ { j = 1 } ^ { K } c _ { j } ^ { ( i ) t } } \end{array}$
α<sup>(i)</sup> $\begin{array} { r } { \overleftarrow { + } \frac { 1 } { 1 + \exp ( - ( \omega _ { c } \cdot \mathbb { E } [ C _ { V } ] ^ { ( i ) } + b _ { c } ) ) } } \end{array}$
// Parameterized adaptive
expected margin limits
$\mathbf { z } _ { V } ^ { ( i ) }  \alpha ^ { ( i ) } \cdot \mathbf { v } _ { s k } ^ { ( i ) } + ( 1 - \alpha ^ { ( i ) } ) \cdot \mathbf { v } _ { r g b } ^ { ( i ) }$
$\hat { P } ^ { ( i ) } \gets P ( y ^ { ( i ) } = 1 | \mathbf { z } _ { V } ^ { ( i ) } ; \boldsymbol { \Theta } _ { C } )$
// Phase IV: Cross-modal Batch
Cost Calculation and
Supervised Target
Distillation
Compute SupCon subspace topology mapping
margins $\mathcal { L } _ { s u p } ^ { \bar { V } \to I }$ according to Equation 5
Compute SupCon textual constraint boundaries
$\mathcal { L } _ { s u p } ^ { V  T }$ according to Equation 6
$J ( \Theta ) $
$\begin{array} { r } { - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \log \left( \hat { P } _ { y = y ^ { ( i ) } } ^ { ( i ) } \right) + \lambda \mathcal { L } _ { s u p } ^ { V \to I } + \gamma \mathcal { L } _ { s u p } ^ { V \to T } } \end{array}$
// Phase V: Weight Optimisation
Application via standard
optimisation
$\Theta  \Theta - \eta \nabla _ { \Theta } J ( \Theta )$
return Θ
```

![](images/79c23f16378384c4b441f1c60e1aa363ec9eaa0db4f7d4690473522f43420124.jpg)  
Fig. 2: Schematic of the proposed Cross-Modal Subspace Distillation Framework. During optimisation, invariant spatial limits map from physical hardware bounds $( \mathbf { X } _ { I } )$ and tabular metadata text (x<sub>T</sub>). Under severe geometric visual occlusions (where empirical tracking validation limits drop, $\mathbb { E } [ C _ { V } ]  0 ) ,$ representation calculation shifts probabilistically back towards standard optical derivatives $( { \bf v } _ { r g b } )$ . Test deployment evaluates bounds isolated to zero-hardware derivations, processing solely through the inference matrix constraints.

## IV. EXPERIMENTAL SETUP

In this section, we present the evaluation framework designed to validate our cross-modal distillation methodology. We describe the dataset formulation, model parameter configurations, and the comparative baselines. We specifically establish a metric space tailored to the imbalanced occurrence probability of FOG events.

## A. Dataset and Protocol

We evaluate our proposed workflow on a public multimodal FOG dataset<sup>2</sup> [3] consisting of 35 individuals diagnosed with idiopathic Parkinson’s disease. The protocol tasks participants with a $3 6 0 ^ { \circ }$ turning-in-place protocol evaluated over intervals of 2 minutes per experimental session. Clinical metadata assessments, inclusive of the New Freezing of Gait Questionnaire (NFOG-Q) and the Unified Parkinson’s Disease Rating Scale (UPDRS-III), function as categorical variables initialising x<sub>T</sub> .

To formalise the temporal sequence synchronisation, let $F _ { V }$ denote the operational sampling frequency of the camera sensor (30 Hz) and $F _ { I }$ dictate the synchronous frequency of the hardware Inertial Measurement Unit (128 Hz). The continuous observation matrices are discretely partitioned into uniform sub-windows evaluated against a shifting duration boundary $\Delta t \in \{ 1 . 0 , 1 . 5 , 2 . 0 , 3 . 0 \}$ seconds. All formulations enforce a 50% uniform sliding overlap.

Formally, an isolated observation sample $w _ { k } ( \Delta t )$ contains a precisely synchronised dimensional tuple:

$$
w _ { k } ( \Delta t ) = \{ \mathbf { X } _ { V } ^ { ( k ) } \in \mathbb { R } ^ { ( F _ { V } \cdot \Delta t ) \times H \times W \times 3 } , \mathbf { X } _ { I } ^ { ( k ) } \in \mathbb { R } ^ { ( F _ { I } \cdot \Delta t ) \times 6 } \}\tag{8}
$$

Table I explicitly formalises the resultant sequence parameters spanning these discrete intervals. An evaluated window maps to a binary target assignment $y ^ { ( k ) } = 1$ unconditionally if any sequence bounded within $w _ { k }$ overlaps an expert-annotated FOG instance. Due to the intrinsically intermittent biological occurrence of movement arrest, narrowing or extending temporal boundary extraction non-linearly manipulates overall distribution entropy. Consequently, Table I highlights the pronounced categorical probability imbalance limiting baseline vision algorithms, thus motivating our explicit cross-modal mapping configuration.

TABLE I: Dataset parameter distributions structured conditionally across shifting temporal inference windows $( \Delta t )$ Evaluated strictly utilising a 50% transition stride. Instances isolate discrete spatial tensor bounds versus inertial mappings per FOG objective alignment.
<table><tr><td>Window</td><td>Overlap</td><td>Frames  $( F _ { V } )$ </td><td>IMU Vectors (FI)</td><td>Total (N)</td><td>Pos. (y = 1)</td><td>Neg. (y = 0)</td></tr><tr><td>1.0 sec</td><td>50%</td><td>30</td><td>128</td><td>16,969</td><td>3,532</td><td>13,437</td></tr><tr><td>1.5 sec</td><td>50%</td><td>45</td><td>192</td><td>11,289</td><td>2,466</td><td>8,823</td></tr><tr><td>2.0 sec</td><td>50%</td><td>60</td><td>256</td><td>8,449</td><td>1,920</td><td>6,529</td></tr><tr><td>3.0 sec</td><td>50%</td><td>90</td><td>384</td><td>5,609</td><td>1,372</td><td>4,237</td></tr></table>

## B. Implementation Details

Our proposed learning framework utilises independent embedding models for each modality. For the inertial extractor $E _ { I } ( \cdot )$ , we employ a 1D Convolutional Neural Network consisting of three successive convolutional and max-pooling blocks, projecting to a vector dimension $p = 5 1 2$ . This model is pre-trained exclusively on $\mathbf { X } _ { I }$ to reach an optimal latent topology and frozen before cross-modal training. For the visual input, the dense pixel network $( E _ { r g b } )$ corresponds to a pretrained R3D-18 (ResNet-3D) backbone. The skeletal pipeline estimates 17 joints per frame; the spatial dependencies among these tracking coordinates are encoded utilising a Spatio-Temporal Graph Convolutional Network $( E _ { s k } )$ . The clinical metadata prompts are mapped using a pre-trained clinical language Transformer, where the mean pooling of the ultimate hidden layer produces $\mathbf { z } _ { T } \in \mathbb { R } ^ { 5 1 2 }$

For the objective function parameters detailed in Equation 7, the batch size M is set to 32. We conduct a grid search for the information-constraint penalties λ and $\gamma$ within the continuous range [0, 1]. We set the InfoNCE temperature scalar $\tau \ : = \ : 0 . 0 7$ as per the normalised projection heuristics. The complete unified model parameters Θ are updated employing the Adam optimiser with a learning rate $\eta = 1 0 ^ { - 4 }$ spanning a maximum of 50 epochs.

## C. Baselines

To estimate the degree of information leakage and compensation managed by our proposed distillation method, we test it against specific unimodal and feature-fusion baselines.

a) RGB-Only.: A naive visual classifier that trains exclusively on raw frames $E _ { r g b } ( \mathbf { X } _ { V } )$ using standard Cross-Entropy optimization.

b) Skeleton-Only.: A temporal prediction architecture mapping only the estimated coordinates via $E _ { s k } ( \cdot )$ . It lacks mechanisms to deduce latent kinematics from coordinates with low prediction confidence (e.g., crossing-leg occlusion).

c) Vision Early-Fusion.: An architecture directly mapping the combined visual elements $\left[ \mathbf { v } _ { r g b } , \mathbf { v } _ { s k } \right]$ without distillation constraints mapped from the inertial Oracle.

d) Apex (IMU Oracle).: The 1D-CNN IMU network evaluated natively. Analogous to non-private estimations representing an idealised upper-bound (the Apex-line), this defines the maximum expected retrieval limit utilising attached hardware, providing a point of reference for our synthesised cross-modal visual performance.

## D. Evaluation Metrics and Parameters

Given that the probability of the complementary condition (normal walking, $y \ : = \ : 0 )$ strictly dominates the occurrence of the condition of interest (FOG, y = 1), computing gross binary accuracy misrepresents algorithm robustness. If an estimator predicts a sequence as $y = 0$ indefinitely, empirical accuracy would remain spuriously high. Consequently, we discard raw accuracy and adopt alternative metrics defined from an information theoretic classification context.

First, we utilise the Matthews Correlation Coefficient (MCC), a symmetric measurement independent of cluster mass. Let $T P , T N , F P $ , and FN represent the components of a discrete confusion matrix; MCC is modelled as:

$$
\mathbf { M C C } = { \frac { T P \times T N - F P \times F N } { { \sqrt { ( T P + F P ) ( T P + F N ) ( T N + F P ) ( T N + F N ) } } } }\tag{9}
$$

Furthermore, we utilise the harmonic aggregate between class precision and sensitivity, measured by the F1-Score, specifically isolated to the positive indicator class $F _ { 1 } ( y = 1 )$ . Lastly, to capture sensitivity limits beyond single operational thresholds, we measure the area under the Precision-Recall curve (AUPRC).

We record measurements across two specific contextual formulations:

1) Frame-level inference: Evaluation is executed strictly per discrete window $w _ { k } .$ , computing if the explicit alignment overlaps identically.

2) Event-level (Bout) inference: Designed to capture clinical alerting efficiency, where the objective implies issuing an indicator corresponding to a prolonged kinematic freeze. An instance mapping indicates 1 (True Positive event) if:

$$
\exists k \in { \mathcal { K } } _ { \mathrm { e v e n t } } : P ( y ^ { ( k ) } = 1 | \mathbf { z } _ { V } ^ { ( k ) } ; \Theta ) > 0 . 5\tag{10}
$$

where $\kappa _ { \mathrm { e v e n t } }$ maps the discrete sub-windows comprising a single ground-truth temporal FOG event block.

## V. RESULTS AND DISCUSSION

In this section, we present the evaluation of the proposed cross-modal distillation method. First, we compare the clustering effectiveness of our proposed framework with the baselines defined in Section IV. Next, we analyse the performance bounds of the modalities, utilising both frame-level and eventlevel metrics. Finally, we formulate a qualitative analysis evaluating the stability of the feature expectation mapping under occluded spatial states.

## A. Quantitative Analysis

Table II summarises the optimal results across the differing modalities. To mitigate classification biases arising from thresholding under heavy data imbalance, the optimal point for each formulation was derived by maximising the F1-Score of the positive FOG class on the validation set.

As an anticipated reference point, the unimodal Inertial (IMU Oracle) model yields an apex clustering alignment. Relying solely on $E _ { I } ( \mathbf { X } _ { I } ; \theta _ { I } )$ , it achieves an AUPRC of 0.7993, defining an operational upper bound achievable only with constant sensor attachment.

Conversely, naive visual configurations indicate substantial information degradation. The Skeleton-Only classifier demonstrates minimum effectiveness $( M C C = 0 . 3 2 1 8 )$ . This occurs because missing target nodes yield zero-vectors, compromising continuous latent representation distances during iterations of occluded $3 6 0 ^ { \circ }$ turning. Early fusion $( \mathbf { v } _ { s k } \oplus \mathbf { v } _ { r g b } )$ only marginalises the prediction error without effectively bounding the FOG geometry within R<sup>p</sup>.

It is observable that the proposed cross-modal distillation configuration drastically minimises the loss relative to the IMU Oracle, whilst demanding exclusively visual input during inference. The application of Equation 5 effectively enforces latent projection topology, reducing misclassification entropy for visual representations. Quantitatively, this reduces the variance within positive sample margins, resulting in an increase of 0.1854 on the event-level F1 metric in comparison to the generic early-fusion baseline.

TABLE II: Comparison of classification architectures against the Oracle bound and naive visual baselines. Evaluations isolate FOG as the primary metric class. Bold text indicates optimal effectiveness exclusive of the hardware Oracle.
<table><tr><td>Configuration</td><td>Inference Input</td><td>Frame-F1</td><td>Event-F1</td><td>AUPRC</td><td>MCC</td></tr><tr><td colspan="6">Naive Vision Baselines</td></tr><tr><td>RGB-Only</td><td>Raw frames</td><td>0.4218</td><td>0.3976</td><td>0.4821</td><td>0.2745</td></tr><tr><td>ST-GCN</td><td>Node Graph</td><td>0.4687</td><td>0.4512</td><td>0.5876</td><td>0.3218</td></tr><tr><td>Vision Early-Fusion</td><td>Frames + Nodes</td><td>0.5364</td><td>0.5089</td><td>0.6814</td><td>0.3976</td></tr><tr><td colspan="6">The Sensor Oracle (Upper Bound)</td></tr><tr><td>Apex-line</td><td>X (IMU)</td><td>0.7456</td><td>0.7179</td><td>0.7993</td><td>0.6880</td></tr><tr><td colspan="6">Proposed Framework</td></tr><tr><td>Cross-modal Distilled</td><td>Frames+Nodes</td><td>0.7126</td><td>0.6943</td><td>0.7750</td><td>0.6128</td></tr></table>

The distribution of events presented in Table II exposes an important divergence between discrete metrics. While both metrics show consistent improvement with cross-modal distillation, Frame-F1 consistently exceeds Event-F1 across all configurations (as further demonstrated in Table III). This pattern is expected because frame-level classification requires exact temporal alignment with annotated FOG windows, whereas event-level detection aggregates predictions over entire FOG bouts. The event-level F1 scores are slightly lower due to the strict requirement that at least one frame within each bout exceeds the classification threshold - any missed detection within a bout counts as an event-level failure. Nevertheless, the proposed method achieves Event-F1 of 0.6943, substantially outperforming all baseline configurations and representing a clinically meaningful improvement in bout-level detection.

## B. Sensitivity to Temporal Limits

Isolating high-frequency trembling artefacts relative to rigid spatial macro-poses requires evaluating structural limits across continuous bounds. We evaluate the proposed crossmodal methodology against shifting temporal observation parameters, restricting input inference windows to $\Delta t \_ { \in }$ $\{ 1 . 0 , 1 . 5 , 2 . 0 , 3 . 0 \}$ seconds.

Narrow evaluation matrices $( \Delta t = 1 . 0 s )$ demonstrate suboptimal Event-level metrics $( F 1 = 0 . 6 7 1 2 )$ . Structurally, 1.0s durations frequently split transitional gait sequences symmetrically, depriving the convolutional spatial representations of completed acceleration limits characteristic of arrested inertia. Conversely, executing classification limits mapped to continuous large spans $( \Delta t = 3 . 0 s )$ artificially dilutes explicit high-frequency $3 - 8$ Hz micro-festinations against broad, unobstructed walking sequences occurring within the extended observation window, escalating prediction boundary entropy.

We identify empirical stabilisation optimally within $\Delta t =$ 1.5s margins (as formulated in Table II), validating that preserving roughly 4 to 10 sequential FOG oscillation vectors isolates coordinate variances maximising convergence for contrastive Oracle projections constraints.

## C. Visual Interpretability under Spatial Occlusion

In standard analytical constraints devoid of physical blockages, explicitly mapping joint geometries offers optimal inference validation. However, empirical performance under dynamic $3 6 0 ^ { \circ }$ occlusion loops shows that deterministic graph evaluation frameworks are mathematically suboptimal in these conditions.

By implementing the expected coordinate continuity limitation $\mathbb { E } [ C _ { V } ]$ structured in Section III, spatial geometry mapping actively defaults against structural degradation limitations. Specifically, tracking metrics confirming minimum optical joint overlap logically converge parameters to bound sequences utilising standard $\mathbb { R } ^ { p }$ limits $( \mathrm { l i m } _ { \alpha  0 } { \bf z } _ { V } \approx { \bf v } _ { r g b } )$

To objectively assess the spatial fidelity of the learned representations under these mathematical constraints, we examine how the latent representations correspond to the observed visual inputs. The results indicate that enforcing only the physical topological constraints can produce fragmented spatial responses, particularly in regions affected by occlusion, where missing limb trajectories may lead to spurious activations and reduced localisation accuracy.

In contrast, incorporating the expected visual representation through the cross-modal constraints produces more spatially coherent responses. The resulting activations are concentrated around anatomically relevant regions, particularly near the ankle and lower-limb areas associated with FOG-related motion patterns. This suggests that the cross-modal supervision helps the model maintain meaningful spatial representations despite partial occlusion and tracking discontinuities.

## D. Modal Penalty Ablation and Manifold Topology

The multi-task optimisation mapping formalised in Equation 7 establishes λ and γ as critical hyperparameters explicitly constraining the degree of topology overlap imposed upon the visual subspace $( \mathbf { z } _ { V } )$ . To evaluate the mathematical necessity of maintaining simultaneous dual-modal Oracles during training, we perform isolated ablations.

As structured in Table III, setting the categorical textual boundary $\gamma = 0$ collapses the textual contextualisation, depriving the evaluation constraints of baseline patient severity metadata. Conversely, nullifying the inertial metric limit $( \lambda = 0 )$ degrades the vision manifold strictly back to baseline spatial bounds heavily subject to coordinate occlusion (analogous to early-fusion approximations). Optimal cluster discrimination converges predictably when maintaining equilibrium across non-visual constraints $( \lambda \approx \gamma \approx 1 . 0 )$ , ensuring that topological deviations measured during high-occlusion transitions (derived via $\mathbb { E } [ C _ { V } ]$ decay) undergo immediate penalty applications respective of hardware and profile norms.

To validate spatial geometry bounding, Figure 3 formulates the two-dimensional mapping representation of the derived 512-dimension sequence space $\mathbb { R } ^ { p }$ . Processed across nontraining target splits utilising t-SNE (t-Distributed Stochastic Neighbor Embedding), unconstrained configurations $\begin{array} { r l } { ( \lambda } & { { } = } \end{array}$ $0 , \gamma = 0 )$ plot continuously interleaved sample representations. Integrating multi-modal constraints $( \lambda = 1 . 0 , \gamma = 1 . 0 )$ isolates independent positive feature mass mappings, objectively preventing geometric ambiguity surrounding latent phase estimations in continuous inference logic.

TABLE III: Parameter evaluation resolving objective combination constraints λ (Inertial Projection Weight) and $\gamma$ (Language Projection Weight). Performance measured utilising temporal boundaries $\Delta t = 1 . 5 s$ . Emboldened state highlights optimal inference stability utilised in proposed structural deployments.
<table><tr><td>λ (IMU Bound)</td><td>γ (Text Bound)</td><td>Event-F1</td><td>Frame-F1</td><td>MCC</td></tr><tr><td>0.0</td><td>0.0</td><td>0.4217</td><td>0.4583</td><td>0.2841</td></tr><tr><td>1.0</td><td>0.0</td><td>0.5632</td><td>0.5874</td><td>0.4187</td></tr><tr><td>0.0</td><td>1.0</td><td>0.3715</td><td>0.3946</td><td>0.2113</td></tr><tr><td>1.0</td><td>1.0</td><td>0.6943</td><td>0.7126</td><td>0.6128</td></tr><tr><td>0.5</td><td>1.0</td><td>0.6218</td><td>0.6485</td><td>0.4872</td></tr></table>

![](images/6ace611332a443c0ad5cbdf967111a5b711ddfb9f5f2595e36b859a4623f463c.jpg)  
Fig. 3: Visual interpretation of t-SNE derived projections of the final $\mathbf { z } _ { V }$ layer output distribution limits across inference test windows. Integrating objective inertial bounds (b) resolves severe sample overlap evident in unobstructed baseline estimations (a), minimising topological uncertainties during validation execution sequences.

## E. Computational Complexity for Edge Deployment

Transitioning structural diagnostic analogies to distributed patient device platforms mathematically necessitates evaluating operational overhead calculations bounded specifically outside massive data centre logic environments. Eliminating concurrent measurement synchronisation arrays reduces execution variables extensively relative to multi-sensor data handling.

The deployment iteration executed upon inference environments fundamentally discards the dual $\theta _ { I }$ and $\theta _ { T }$ parameters utilised during structural generation logic defined over Algorithm 1. Evaluating $\mathbf { X } _ { V }$ through spatial convolutions relative to limited subset frame quantities mitigates persistent processing latency calculations. To establish rigorous reproducibility limits, execution latency is measured utilising a workstation-grade NVIDIA RTX A5500 graphics processing unit. Evaluating consecutive matrix distributions against strict timing limits (Table IV), the isolated spatial constraints yield estimation times restricted to 6.91 millisecond evaluation margins. Floating Point Operation counts confirm that decoupled modality boundaries ensure clinical scale implementations do not inherently necessitate explicit continuous telemetry array overhead calculations, successfully matching the strict zerowearable evaluation goals previously introduced in Section I.

TABLE IV: Inference matrix limits mapping objective floating point approximations versus required classification temporal margins processing standard bounded subsets. Execution parameter times map bounds measured natively outside parallel processing bounds utilising an NVIDIA RTX A5500 (per query block configuration, $\Delta t = 1 . 5 s )$
<table><tr><td>Pipeline Mode</td><td>FLOPs (109)</td><td>Params (106)</td><td>Lat. Per Eval (ms)</td></tr><tr><td>Standard Vision Early-Fusion</td><td>45.86</td><td>36.74</td><td>6.54</td></tr><tr><td>Kinematic Baseline Sensor Network (θ1)</td><td>0.01</td><td>0.34</td><td>0.52</td></tr><tr><td>Proposed Distillation Architecture</td><td>45.86</td><td>36.21</td><td>6.91</td></tr></table>

## VI. CONCLUSIONS AND FUTURE WORK

In this paper, we addressed the problem of predicting FOG events under conditions of severe spatial occlusion, typical of a continuous $3 6 0 ^ { \circ }$ turning-in-place task. Recognising that standard deterministic visual feature extraction degrades systematically without physical sensor constraints, we proposed a probabilistic cross-modal distillation framework. By mathematically enforcing alignment between a visual-textual student space and an expert kinematic oracle space $( { \bf z } _ { I } )$ , we transferred robust inertial feature topology directly onto the vision model.

Experimental evaluations demonstrate that formulating the fused visual representation, $\mathbf { z } _ { V }$ , as a function of the expected spatial coordinate confidence, E $\Im [ C _ { V } ]$ , mitigates noise propagation during lower-limb occlusion. The proposed methodology substantially lowers classification entropy during inference. Consequently, our approach approaches the apex boundary established by hardware sensors, validating that precise FOG detection can operate using non-encoded video streams devoid of continuous wearable components.

In future work, we plan to relax the assumption of deterministic temporal synchronisation between the hardware and visual signals during the training phase. We aim to explore probabilistic, unsupervised domain adaptation to model instances where strict multi-modal alignment boundaries are unknown. Furthermore, extending this informative subspace optimisation to unconstrained, in-the-wild environments—where explicit clinical attribute metadata $( { \bf z } _ { T } )$ is latent or incomplete—stands as a logical progression towards formalising fully unobtrusive clinical assessments.

## REFERENCES

[1] M. Bachlin, M. Plotnik, D. Roggen, I. Maidan, J. M. Hausdorff, N. Giladi,¨ and G. Troster, “Wearable assistant for parkinson’s disease patients¨ with the freezing of gait symptom,” IEEE Transactions on Information Technology in Biomedicine, vol. 14, no. 2, pp. 436–446, 2010.

[2] D. Li, Y. Sun, Z. Yao, J. Wang, S. Wang, and X. Yang, “Improved deep learning technique to detect freezing of gait in parkinson’s disease based on wearable sensors,” Electronics, vol. 9, no. 11, p. 1919, 2020.

[3] C. Ribeiro De Souza, R. Miao, J. Avila De Oliveira, A. C. De Lima-<sup>´</sup> Pardini, D. Fragoso De Campos, C. Silva-Batista, L. Teixeira, S. Shokur, B. Mohamed, and D. B. Coelho, “A public data set of videos, inertial measurement unit, and clinical scales of freezing of gait in individuals with parkinson’s disease during a turning-in-place task,” Frontiers in Neuroscience, vol. 16, p. 832463, 2022.

[4] M. Mancini, V. V. Shah, S. Stuart, C. Curtze, F. B. Horak, D. Safarpour, and J. G. Nutt, “Measuring freezing of gait during daily-life: an opensource, wearable sensors approach,” Journal of NeuroEngineering and Rehabilitation, vol. 18, no. 1, pp. 1–11, 2021.

[5] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[6] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.

[7] F. M. Thoker and J. Gall, “Cross-modal knowledge distillation for action recognition,” in 2021 IEEE International Conference on Image Processing (ICIP). IEEE, 2021, pp. 6–10.

[8] S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[9] J. Song et al., “Modeling uncertainty in skeletal tracking for robust motion prediction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.