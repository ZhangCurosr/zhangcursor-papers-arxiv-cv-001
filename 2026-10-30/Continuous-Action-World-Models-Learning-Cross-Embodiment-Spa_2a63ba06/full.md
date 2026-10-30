# Continuous-Action World Models: Learning Cross-Embodiment Spatiotemporal Dynamics Priors

Liangyu Li<sup>\*</sup> Qingwen Liu<sup>†</sup>

## Abstract

We introduce Continuous-Action World Models (CoAWM), a framework for learning cross-embodiment spatiotemporal dynamics priors that support robot action generation across embodiments and execution frequencies. At its core is a shared latent function space that captures the spatial and temporal structure of robot actions. We learn this space through two stages. First, function pretraining brings trajectories from different embodiments and sampling frequencies into a common representation, learning shared dynamics priors through action reconstruction and consistency across temporal sampling views. By accommodating heterogeneous robot trajectories, this stage broadens the range of data sources available for pretraining and facilitates scaling to larger and more diverse datasets. Second, we connect the pretrained function space to a worldmodel backbone, enabling observations, language instructions, and the current robot state to condition the prediction ofa latent actionfunction. A single decoder takes any query time and returns the corresponding end-effector action. By separating shared dynamics priors from task-conditioned action selection and temporal sampling, our approach provides a unifiedfoundationfor transferring robot experience across embodiments and controlfrequencies.

## 1. Introduction

Robots with different bodies can perform similar tasks, yet their experience is recorded through different action interfaces and control frequencies [3, 13, 24]. A reaching motion may be demonstrated by one manipulator and executed by another, with different command sequences governing a related physical behavior. This diversity raises a central question for robot world models: how can experience collected across embodiments be used to learn shared spatiotemporal dynamics priors that guide action generation? Answering this question would broaden the sources of experience available to a model, allowing robot learning to benefit from data beyond a single embodiment or collection setup.

The challenge is not simply to combine more trajectories. Robot datasets differ in what their actions specify, how those actions are grounded in the robot’s state, and when commands are issued. The same sequence index can correspond to different physical times, while numerically similar commands can have different meanings across action interfaces. A common storage format alone does not resolve these differences. To learn from heterogeneous experience, a model needs a representation that captures reusable spatial and temporal structure while retaining the information required to interpret and execute an action on a particular robot. For a world model, this representation must also connect shared experience to action selection in the current scene and task.

Our key insight is to learn the evolution of an action as a function, rather than treating its sampled commands as the fundamental object of representation. A latent action function encodes how a motion unfolds in space and time; individual commands are obtained by querying that function (Fig. 2). This provides a common learning space for trajectories sampled at different frequencies. More importantly, it separates two complementary learning problems: acquiring dynamics priors from heterogeneous trajectories, and selecting an action using the current interaction context. The former can draw on a broader pool of robot experience, while the latter connects these priors to the observationconditioned capabilities of a world model.

We introduce Continuous-Action World Models (CoAWM) to realize this idea through two stages (Fig. 1). First, we pretrain a shared latent function space using trajectories from different embodiments and sampling frequencies. A function encoder and decoder learn spatiotemporal dynamics priors through action reconstruction and consistency across temporal sampling views. This pretraining stage expands the range of trajectory sources that can contribute to representation learning, providing a path toward larger and more diverse training datasets. Second, we connect the pretrained function space to a world-mode backbone. Visual observations, language instructions, and the current robot state condition a prediction head that produces a latent action function, which a shared decoder converts into end-effector actions through time queries. CoAWM thus connects cross-embodiment prior learning with task-conditioned action generation, while allowing the execution frequency to be specified through the query times.

![](images/159849503dfee8d4f996ff51b0e78df37a6185d679fdbeac18cd8314d75d9212.jpg)  
Figure 1. Overview of continuous action-function learning and deployment. Training Stage A learns shared spatiotemporal priors from typed trajectories across embodiments and temporal sampling grids. Training Stage B connects observation-conditioned latent prediction to the learned function space using latent and physical reconstruction supervision. At inference, current multimodal observations and visual features condition latent prediction; the function decoder combines the predicted latent with physical query times and observed context to produce embodiment-specific control commands.

## 2. Related Work

## 2.1. Continuous Action and Function Representations

Action representations determine how a policy organizes motion over time. ACT predicts action chunks [34], Diffusion Policy generates action sequences through denoising [10], and FAST compresses temporal structure into frequency-domain tokens [26]. Continuous action representations capture how motion evolves over time and allow actions to be queried at different execution times. Dynamical Movement Primitives encode motor behavior through attractor dynamics and learned forcing terms [16], Probabilistic Movement Primitives model distributions over trajectories [25], and ProDMPs connect these dynamic and probabilistic formulations [19]. These foundations establish motion trajectories as structured objects from which reusable action priors can be learned.

Recent visuomotor policies explore different function families and decoding mechanisms. FLASH represents trajectories with a Legendre basis and history-anchored flow [2], while B-spline Policy uses control points and knots to parameterize actions [14]. Neural Implicit Action Fields learn modulated continuous-time action functions and supervise their temporal derivatives [20]. Periodicactivation implicit networks provide a general foundation for coordinate-queryable neural functions and their derivatives [27]. These efforts to represent actions motivate us to learn shared spatiotemporal dynamics priors through a continuous function representation.

A latent function representation captures how motion evolves over time. A decoder takes this representation and a time query to produce the corresponding action. Querying the same representation at different times yields action sequences at different temporal resolutions. FunDiff combines function encoding, latent diffusion, and continuous decoding for physical-field generation [30]. Function-Space Diffusion for Motion Planning models trajectory distributions using temporally correlated noise and boundarycompatible operators [7]. CAT learns a fixed-size set of continuous trajectory latents over a physical-time window and uses frequency-aware positional encoding with a causal action decoder to accommodate different control rates [31]. These works establish important precedents for learning continuous representations and transferring between temporal resolutions. CoAWM focuses on learning shared spatiotemporal dynamics priors from heterogeneous robot trajectories and connecting the resulting function space to world-model-conditioned action generation.

## 2.2. Cross-Embodiment Learning and Action Priors

Cross-embodiment learning expands the experience available for robot policy training. Open X-Embodiment brings together robot datasets and studies cross-robot policies [24]. Octo develops generalist pretraining and adaptation to new observation and action spaces [13], and $\pi _ { 0 }$ combines a vision-language model with flow-based action generation for general robot control [3]. These approaches make heterogeneous data a practical resource for learning behavior across tasks and robot systems. A central representation question is which structure can be shared while maintaining a meaningful connection to each robot’s executable actions.

Latent-action learning addresses this question by separating reusable action knowledge from specific command spaces. LAPA learns latent actions from video before adapting them to robot control [32], and UniVLA learns taskcentric latent actions for embodiment transfer [5]. LAC-WM learns shared latent actions from visual transitions with auxiliary motion reconstruction for cross-embodiment world modeling, and reports benefits from increasing the diversity of pretraining embodiments [15]. These results motivate shared action priors for both control and prediction. CoAWM approaches this problem through action trajectories: a function autoencoder learns their spatial and temporal structure, and an observation-conditioned head connects the pretrained function space to a world-model backbone. Its pretraining interface is designed to accommodate trajectories from multiple embodiments and sampling frequencies, broadening the sources of data available for learning dynamics priors.

## 2.3. World Models for Action Generation and Execution

World models connect experience to predictions that support action selection. V-JEPA 2 adapts self-supervised video representations for action-conditioned prediction and planning [1]. LAWM learns latent actions through futureframe modeling [28], while Learning Latent Action World Models In The Wild investigates continuous-valued latent actions for prediction and controllable interaction [12]. villa-X combines latent-action modeling with visual and proprioceptive forward dynamics and robot-action prediction [9]. Alongside LAC-WM [15], these approaches highlight the role of an action representation in linking predicted transitions to executable behavior.

LaWAM conditions action generation on predicted latent visual subgoals and explicitly supports physical-time alignment and mixed-frequency training [8]. Its mixed-

![](images/2db9a3d670cb8480c7a2f93c2c3eec61cfb139193ed4c9cf3bfad72dfef56250.jpg)

![](images/41305d9a7bbdcb17ccaa205ba7b155f27e7c224e2763e7747351bae49214ccfe.jpg)

![](images/d097a93585fb2f8d497e6051eec66a61f693ff208c532460d43573e5b2b48c0f.jpg)  
Figure 2. An action interval and its sampling grid are different objects. Schematic of a continuous scalar position reference. The first two panels query the same function over the same physical duration with different grids. The third changes the duration: the geometric progression is preserved, but its velocity and acceleration change. A reusable function representation must retain physical timing while allowing the execution grid to vary.

## 3. Method

frequency setting retains native-rate action sequences with variable lengths and uses physical-time encoding to align them. AHA-WAM separates a slow video planner from a fast action executor through observation-guided context routing and horizon-adaptive offset training [6]. AW-0 extends world-action modeling with joint code, video, and action generation [22]. CoAWM learns a shared latent function space through cross-embodiment pretraining and connects it to world model and action generation. Given visual observations, language instructions, and the current robot state, the model predicts a latent function in this pretrained space. The decoder then takes time queries to produce executable actions. This design makes the pretrained dynamics priors directly available to action generation and allows a single predicted function to produce action sequences at different execution frequencies.

Execution introduces complementary questions about continuity, latency, and commitment. Real-Time Chunking overlaps prediction and execution through inference-time inpainting [4], Legato learns native continuation for actionchunking flow policies [21], and REMAC uses masked action chunking to address real-time execution [29]. Mixture of Horizons combines action predictions over multiple horizons [17]; PACE uses kinematic phases to select execution boundaries [23], and DVAC adapts the executed prefix using denoising variance [11]. Reliable Action-Chunk Extension models when to switch according to predicted subskill transitions [33], whereas Reachability-aware Accelerated Chunk Execution retimes predicted state paths under physical constraints [18]. These methods concern how a plan is committed, continued, or retimed during execution. CoAWM complements these execution mechanisms by separating the latent action function, the times at which it is queried, and the physical timing of the represented motion.

## 3.1. Overview

Continuous-Action World Models (CoAWM) connect world model and action generation through a pretrained space of action functions. The central representation is a latent function that describes how an action evolves over physical time. A decoder takes this latent function and time queries to produce the corresponding actions, enabling action generation at different execution frequencies through queries at the corresponding time points.

CoAWM uses two training stages. Stage A learns an action-function encoder and decoder from timestamped robot trajectories. A Mixture-of-Transformers (MoT) encoder combines specialized processing of end-effector position, orientation, and gripper commands with joint attention across these action components. Training aligns representations of the same motion observed at different frequencies. This provides a shared function space for learning from trajectories collected across robot embodiments and at different frequencies, broadening the data available for pretraining. Stage B connects the learned function space to a worldmodel backbone. The prediction head combines VLM features, predicted future visual features, and robot proprioception, with current visual features contributing through an output residual. It predicts a function in the pretrained space. Expert trajectories supervise both its latent representation and the actions obtained by decoding it. At inference, the decoder queries the predicted latent function at time points determined by the desired execution frequency.

## 3.2. Continuous Action Function Representation

A function latent with a physical-time interface. We represent an action sequence as timestamped observations $\boldsymbol { \mathcal { A } } = \{ ( \tau _ { i } , a _ { i } , m _ { i } ) \} _ { i = 1 } ^ { N }$ , where $a _ { i }$ is an end-effector command and $m _ { i }$ specifies valid components. Physical time τ = 0 denotes the current observation. Let c denote known execution information, including the current end-effector anchor and validity information. An encoder $E _ { \phi }$ maps the sequence to a latent function $Z ,$ and a decoder $D _ { \psi }$ queries it at physical time τ:

$$
Z = E _ { \phi } ( A , c ) , \qquad a ( \tau ) = D _ { \psi } ( Z , \tau , c ) .\tag{1}
$$

The latent has a fixed size, while the timestamps used to observe or query the motion can vary.

The implementation uses an external time coordinate u offset from the physical clock by $\Delta t ,$ the delay between the current observation and the first action target. The physicaltime notation above corresponds to

$$
\begin{array} { c } { { \tau = u + \Delta t , } } \\ { { D _ { \psi } ( Z , \tau , c ) = D _ { \psi } ^ { \mathrm { i m p l } } ( Z , \tau - \Delta t , c ) . } } \end{array}\tag{2}
$$

The encoder $E _ { \phi }$ above also denotes a physical-time interface: its implementation receives external labels $u _ { i } ~ =$ $\tau _ { i } - \Delta t$ . Encoder timestamps and decoder queries are each converted to physical time once at their respective module boundaries. The current-state anchor therefore corresponds to external query $u = - \Delta t$ . The execution segment spans 1.2 s of physical time.

Mixture-of-Transformers for Action Function Encoding. The MoT encoder brings position (XYZ), orientation (Q), and gripper (G) components into a shared latent function representation. In the dual-arm implementation, these streams contain six position coordinates, eight quaternion coordinates, and two gripper commands. Each stream combines the corresponding components of both arms. Input tokens incorporate timestamps and validity masks, so the encoder can process different temporal views without treating missing observations as valid commands.

The encoder first aggregates trajectory tokens into learned latent queries through typed Perceiver blocks. Latent MoT blocks then exchange information across the three action-component streams. Each stream has its own query, key, value, and output projections, normalization layers, and feed-forward networks. Attention is computed jointly over all streams, allowing the specialized Transformer parameters to model each action component while shared attention captures their temporal coordination. The bottleneck contains 96 tokens of width 256, with 32 tokens allocated to each modality.

Independent time queries. The decoder forms a query from the requested physical time, its Fourier features, a modality embedding, and known execution information. Typed cross-attention blocks read the latent memory and produce XYZ, Q, and G outputs. Each requested timestamp is evaluated through its own query-to-memory attention. Consequently, an action at a given timestamp does not depend on the number or ordering of the other requested timestamps. A single latent can therefore produce actions at different execution frequencies through the corresponding time queries.

Anchored geometric readout. Physical time zero corresponds to the observed current state. For each valid arm, the position readout uses the current position $p _ { 0 }$ as its anchor:

$$
p ( \tau ) = p _ { 0 } + \widetilde { p } ( \tau ) - \widetilde { p } ( 0 ) ,\tag{3}
$$

where $\widetilde { p }$ is the decoder’s position output before anchoring, expressed on the physical clock. Let $x ( \tau )$ be its normalized quaternion output and $q _ { 0 }$ the current orientation. The orientation readout is

$$
q ( \tau ) = \mathrm { n o r m a l i z e } \left( q _ { 0 } \otimes r ( 0 ) ^ { - 1 } \otimes r ( \tau ) \right) ,\tag{4}
$$

where $\otimes$ denotes quaternion multiplication. These readouts express predicted motion relative to the observed pose and recover that pose at time zero. The hard anchor applies to position and orientation. Gripper commands use a separate time-indexed knot readout: the decoder predicts gripper values at native knots, and a query selects the nearest knot. This accommodates discrete gripper changes alongside time-queryable pose commands.

## 3.3. Learning Cross-Embodiment Dynamics Priors

Stage A learns reusable dynamics priors from action trajectories. Its inputs are timestamped actions and known execution information, allowing action data to contribute to pretraining without paired language instructions or camera observations. The MoT encoder integrates XYZ/Q/G streams through specialized Transformer parameters and shared attention. This common action organization, together with validity masks and execution context, provides the representational interface for incorporating trajectories from different embodiments.

Learning across observation frequencies. For each trajectory, we construct two observations $\boldsymbol { \mathcal { A } } ^ { ( 1 ) }$ and $\boldsymbol { \mathcal { A } } ^ { ( 2 ) }$ at different sampling rates. Both describe the same underlying motion on the same physical clock. The shared encoder produces

$$
\begin{array} { r } { Z ^ { ( 1 ) } = E _ { \phi } ( \mathcal { A } ^ { ( 1 ) } , c ) , \qquad Z ^ { ( 2 ) } = E _ { \phi } ( \mathcal { A } ^ { ( 2 ) } , c ) . } \end{array}\tag{5}
$$

Each latent receives reconstruction supervision at its source timestamps and on an additional query grid. The grid contains 73 points at 60 Hz over external coordinates $u \in [ 0 , 1 . 2 ]$ s, mapped to physical time by $\tau = u + \Delta t$

Let $\Delta p$ denote the six-dimensional position error of both arms, $\Delta g _ { L }$ and $\Delta g _ { R }$ their gripper errors, and $\theta _ { L }$ and $\theta _ { R }$ their shortest rotation errors on SO(3). For each arm, $\theta =$ $\lVert \log ( q ^ { - 1 } \hat { q } ) \rVert _ { 2 }$ is the geodesic angle in radians. For view v and query subset S, reconstruction is

$$
\begin{array} { r } { E _ { S } ^ { ( v ) } = \mathrm { m e a n } _ { m _ { S } ^ { ( v ) } } \displaystyle \left[ \frac { \lVert \Delta p \rVert _ { 2 } ^ { 2 } } { 6 } + \frac { \theta _ { L } ^ { 2 } + \theta _ { R } ^ { 2 } } { 2 } \right. } \\ { \displaystyle \left. + \frac { \Delta g _ { L } ^ { 2 } + \Delta g _ { R } ^ { 2 } } { 2 } \right] . } \end{array}\tag{6}
$$

The mean uses the subset’s own Boolean mask over batch and query indices. Position is measured in meters, and the three terms use their native scales. Source and grid losses are reduced separately and assigned equal weight. The two-view objective is

$$
\mathcal { L } _ { A } = \frac { 1 } { 2 } \sum _ { v = 1 } ^ { 2 } \left( E _ { \mathrm { s o u r c e } } ^ { ( v ) } + E _ { \mathrm { g r i d } } ^ { ( v ) } \right)\tag{7}
$$

The latent MSE averages over batch, tokens, and latent dimensions. Reconstruction teaches the decoder how motion evolves over time; consistency aligns the representations obtained from different observation frequencies. Together, they establish a shared function space that Stage B can address directly.

A scalable trajectory interface. Separating actionfunction pretraining from task-specific observation-toaction learning makes trajectory data usable through a common representation. Additional sources enter through timestamped end-effector trajectories, component validity, and known execution information. Shared attention in the MoT encoder captures coordination among action components, while the explicit context preserves information needed to interpret the motion for a particular robot. This is the mechanism through which CoAWM connects crossembodiment data collection to the scaling of dynamicsprior learning.

## 3.4. Connecting the Function Space to a World Model

Stage B connects the pretrained function space to LaWAM’s visual world prediction. Current RGB observations from the head and two wrist cameras, together with a language instruction, are processed by the VLM to produce features $h _ { \mathrm { v l m } }$ and the original latent-action prediction $\hat { a } _ { \mathrm { l a t e n t } }$ . The current images also yield LAM visual tokens $h _ { t }$ . The retained world decoder predicts future visual tokens:

$$
\hat { h } _ { t + 1 } = \mathrm { L A M D e c o d e r } ( h _ { t } , \hat { a } _ { \mathrm { l a t e n t } } ) .\tag{8}
$$

Here $\hat { a } _ { \mathrm { l a t e n t } }$ drives the original world prediction, while $\hat { Z }$ below is the action-function latent decoded into robot commands. The head combines projected VLM features, predicted future visual tokens, and proprioceptive state s as its cross-attention memory:

$$
M = \operatorname { c o n c a t } ( P _ { v } h _ { \mathrm { v l m } } , P _ { f } \hat { h } _ { t + 1 } , P _ { s } s ) .\tag{9}
$$

Learned latent queries attend to M through the head’s Transformer decoder. Let H denote the resulting hidden tokens after LayerNorm. Current LAM visual tokens enter a separate attention residual, with H as queries and projected current tokens as keys and values:

$$
\hat { Z } = W _ { \mathrm { o u t } } H + W _ { \mathrm { r e s } } \mathrm { A t t n } ( H , P _ { c } h _ { t } , P _ { c } h _ { t } ) .\tag{10}
$$

The residual output projection $W _ { \mathrm { r e s } }$ is initialized to zero. The head thus combines task semantics and predicted visual evolution in its main memory, then incorporates current visual information at the function-latent output. Proprioception enters the head memory; the decoder’s execution information c remains a separate interface for time-query decoding and anchoring.

Expert supervision in function space. During training, the frozen action encoder maps an expert future trajectory $\mathcal { A } ^ { \ast }$ to a target latent:

$$
Z ^ { * } = \mathrm { s g } [ E _ { \phi } ( A ^ { * } , c ) ] ,\tag{11}
$$

where sg stops gradients. For each batch sample $b ,$ we flatten the full predicted and target latents as $x _ { b } = \mathrm { v e c } ( \hat { Z } _ { b } )$ and $y _ { b } = \mathrm { v e c } ( Z _ { b } ^ { \ast } )$ . The latent objective is

$$
\mathcal { L } _ { \mathrm { l a t e n t } } = \mathrm { m e a n } _ { b } \left[ 1 - \frac { x _ { b } ^ { \top } y _ { b } } { \| x _ { b } \| _ { 2 } \| y _ { b } \| _ { 2 } } \right] .\tag{12}
$$

The alignment objective computes cosine similarity from the predicted and target latents, and the action decoder directly receives the predicted latent $\hat { Z } .$

Supervision through decoded actions.

Using position, rotation-angle, and gripper errors as defined in Stage $\mathbf { A } ,$ decoded-action supervision uses the scales $s _ { p } = 0 . 0 1 \mathrm { m } , s _ { q } = 0 . 0 5$ rad, and $s _ { g } = 0 . 0 5$ :

$$
\begin{array} { c } { { \mathcal { L } _ { \mathrm { p h y s } } = \displaystyle \frac { 1 } { 3 } \Big [ \mathrm { m e a n } \left( ( \Delta p / s _ { p } ) ^ { 2 } \right) } } \\ { { + \mathrm { m e a n } \left( ( \theta / s _ { q } ) ^ { 2 } \right) } } \\ { { + \mathrm { m e a n } \left( ( \Delta g / s _ { g } ) ^ { 2 } \right) \Big ] . } } \end{array}\tag{13}
$$

Stage B optimizes

$$
\mathcal { L } _ { B } = \mathcal { L } _ { \mathrm { l a t e n t } } + 0 . 1 \mathcal { L } _ { \mathrm { p h y s } } .\tag{14}
$$

Latent alignment connects the prediction to the pretrained function representation, while action supervision ties it to the commands produced at native target times. The function encoder and decoder remain frozen. Gradients through the decoder train the latent prediction pathway and backbone adapters, preserving the function space while adapting its use to the task. The retained world-prediction branch supplies features to the head and remains frozen.

## 3.5. Time-Query Action Execution

At each replanning step, CoAWM uses the current observation to predict one latent function and anchors its pose readout to the observed current state.

For frequency f, we query the nominal endpoint of each command interval. For $k = 0 , \ldots , f T _ { \mathrm { e x e c } } - 1$ , where $T _ { \mathrm { e x e c } }$ denotes the execution-segment duration, the physical query and corresponding external decoder input are

$$
\begin{array} { l } { \displaystyle \tau _ { k } = \frac { k + 1 } { f } , \qquad u _ { k } = \frac { k + 1 } { f } - \Delta t , } \\ { \displaystyle a _ { k } = D _ { \psi } ( \hat { Z } , \tau _ { k } , c ) . } \end{array}\tag{15}
$$

Native execution uses 30 Hz.

The inference path consists of observation encoding, visual world prediction, latent-function prediction, and timequery decoding. The expert trajectory and function encoder are used only for training. This organization connects pretrained dynamics priors to task-aware action generation and frequency-specific execution through a shared function interface.

## 4. Experiments

We evaluate continuous action functions from four perspectives: representation accuracy, closed-loop execution, cross-frequency querying, and action quality. We first examine whether the function space can accurately represent robot actions, then evaluate task performance after alignment with LaWAM, focusing on the effect of the continuous representation on the smoothness of initial execution. (These experiments provide a rapid validation of the method; a subsequent arXiv version will expand the training scale and perform full fine-tuning.)

## 4.1. Representation Accuracy of Continuous Actions

A continuous function representation preserves action details while providing a unified query interface for different time points. On 8,192 held-out evaluation windows, the function autoencoder achieves a position reconstruction error of 2.503 mm and a rotation error of 0.2295<sup>◦</sup>. This result shows that the function space can represent the original action trajectories with high accuracy, providing a representation foundation for subsequent world-model alignment (Fig. 3).

## 4.2. Smoother Initial Execution at Similar Movement Speeds

Continuous action functions significantly reduce abrupt changes in motion during initial execution. Across 16 tasks and 800 pairs of matched initial conditions, we measure motion metrics over the first second of each execution. Compared with LaWAM, position jerk RMS decreases from

![](images/694de0812f5f1c7d1fa88f931240d29fcddffb699050ad40bdb41efc42372645.jpg)  
Figure 3. Mean position and rotation reconstruction errors of the function autoencoder on 8,192 held-out evaluation windows used for model selection.

150.87 to 95.46, a reduction of 36.7%; rotation jerk RMS decreases from 746.06 to 341.73, a reduction of 54.2%. The velocity RMS values of the two methods are 0.338 and 0.332, respectively, showing that continuous action functions reduce abrupt changes in motion without reducing action execution speed (Fig. 4).

This result demonstrates the improvement in initial execution quality provided by continuous action representations. In the same evaluation, our method achieves a task success rate of 75.1%. However, we find that substantial jerk tails remain over longer execution windows, and plan to further address action stability during sustained execution and replanning in subsequent optimization.

## 4.3. Multitask Manipulation and Cross-Frequency Execution

The aligned function policy can perform a variety of bimanual manipulation tasks. In evaluations with 50 trials per task, it achieves a success rate of 94% on the two-bottle task, 90% on both left and right A2B tasks, 88% on the burger task, and 86% on the hammer task. In a separate preliminary evaluation covering 50 tasks with three trials per task, both methods complete all three trials on 30 tasks.

By changing the time query points, the same function policy can output actions at 10, 15, 20, 30, 50, and 60 Hz while keeping the execution segment and replanning period fixed at 1.2 s. The corresponding numbers of successes are 116, 121, 117, 119, 116, and 110, with 150 trials evaluated at each frequency. These results demonstrate crossfrequency execution within a fixed physical-time window (Fig. 5).

## 4.4. Reducing Execution Energy Consumption

We further use positive mechanical work based on PD torque estimates to measure energy consumption during execution. Among 16 prespecified scenarios, on the 13 scenarios in which both methods succeed, the function policy reduces the mean estimated positive mechanical work from 506.66 to 392.13, a reduction of 22.6%, with lower values in 12 scenarios (Fig. 6).

Smoother initial execution

![](images/0aed8fe6700203822367e40cde1717c3f2c2e43aa572a40736a772530ca11a29.jpg)

![](images/747b56c0f239635cef390de4d9fa3776444bf64aa41fca06be0037f1ed2111b9.jpg)

![](images/ff11887d80327b7906bb51fd93992f9c71fd28866088d3f162cc8eb68a1226a0.jpg)  
Figure 4. Initial execution quality during the first 1 s across 800 paired trials in 16 tasks. The highlighted percentages show the reduction in translation jerk RMS, the reduction in rotation jerk RMS, and the retained translation speed RMS relative to LaWAM, respectively. Metrics describe actual end-effector motion, with equal task weighting and both arms pooled.

## One function, multiple execution frequencies

![](images/3112e8ea54d89d9d55bdb630b669309deb9260b71b9f5dddbaa6096404c54c8a.jpg)  
(b) Selected tasks

![](images/1ac384213ade54ba927175d392b334d32b1c4d5a972b1471fad948ff27b34ccc.jpg)  
Figure 5. (a) CoAWM success rates at six execution frequencies, evaluated with 150 trials per frequency across 50 tasks (three trials pe task), with the execution segment and replanning period fixed at 1.2 s. (b) Five selected tasks from a separate 16-task evaluation with 50 trials per task. The panels therefore use different evaluation cohorts and denominators.

## 5. Discussion

Continuous function representations separate the learning of action structure from the choice of execution frequency. The model learns how actions evolve over physical time, while the execution side obtains commands at the required frequency through time queries. Execution frequency can therefore be adjusted while keeping the action-segment duration and replanning period unchanged, allowing the same action representation to serve different execution clocks.

This representation also makes action quality an important evaluation dimension alongside task success rate. The current experiments demonstrate high trajectory reconstruction accuracy and smoother initial execution at similar movement speeds. At the same time, offline reconstruction quality, closed-loop task success rate, and stability during sustained execution have different optimization requirements. Considering these metrics together enables a more complete evaluation of the practical effects of aligning function representations with a world model.

Execution work across paired scenes  
![](images/46eb0a9ac3f9e6e5d9d22488c5a58227e297c73bf92078391fc41cfecd51466e.jpg)  
Figure 6. Estimated positive mechanical work for the 13 jointly successful scenes among 16 prespecified scenes, computed from PD torque estimates and measured joint velocities. Open blue points denote LaWAM and filled red points denote CoAWM; connecting segments identify paired scenes. The two time columns report physical execution duration in seconds using the same method colors.

The shared function space provides a unified interface for expanding the training scale. During function pretraining, we will introduce action trajectories from more embodiments and collection frequencies; during world-model alignment, we will further perform full fine-tuning, using visual predictions and task information to learn task-related action functions. Subsequent experiments will examine how larger-scale data and training affect task performance, cross-embodiment transfer, and stability during sustained execution.

## 6. Conclusion

We present CoAWM, which connects action representation learning, world-model prediction, and time-query execution through a shared function space. Two-stage training uses the spatiotemporal structure in trajectories for task-related action generation and supports different execution frequencies through time queries. Preliminary experiments demonstrate the reconstruction accuracy, bimanual task execution capabilities, and improvements in initial action quality of this representation, providing a foundation for further expanding the scale of data and training.

## References

[1] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv preprint arXiv:2506.09985, 2025. 3

[2] Jiaqi Bai, Jindou Jia, Yuxuan Hu, Gen Li, Xiangyu Chen, Tuo An, Kuangji Zuo, and Jianfei Yang. FLASH: Efficient Visuomotor Policy via Sparse Sampling. arXiv preprint arXiv:2605.15492, 2026. 3

[3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164, 2024. 1, 3

[4] Kevin Black, Manuel Y. Galliker, and Sergey Levine. Real-Time Execution of Action Chunking Flow Policies. arXiv preprint arXiv:2506.07339, 2025. Version 2. 4

[5] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Learning to Act Anywhere with Task-centric Latent Actions. In Proceedings ofRobotics: Science and Systems, 2025. 3

[6] Jisong Cai, Long Ling, Shiwei Chu, Zhongshan Liu, Jiayue Kang, Zhixuan Liang, Wenjie Xu, Yinan Mao, Weinan Zhang, Xiaokang Yang, Ru Ying, Ran Zheng, and Yao Mu. AHA-WAM:Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing. arXiv preprint arXiv:2606.09811, 2026. 4

[7] Zinuo Chang, Yipu Chen, Byoungwoo Park, Hongzhe Yu, and Yongxin Chen. Function-Space Diffusion for Motion Planning. arXiv preprint arXiv:2607.02977, 2026. 3

[8] Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, Yuanbo Xu, and Chao Yu. LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies. arXiv preprint arXiv:2606.15768, 2026. 3

[9] Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, Jianyu Chen, and Jiang Bian. villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models. arXiv preprint arXiv:2507.23682, 2025. 3

[10] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin CM Burchfiel, and Shuran Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. In Robotics: Science and Systems XIX, 2023. 3

[11] Xiangdong Feng, Yuxuan Cheng, Chen Shi, Boyao Han, Yuxuan Yan, Yitong Hong, Zhuotao Tian, and Li Jiang. De-

noising Tells When to Replan: Denoising-Variance Adap tive Chunking for Flow-Based Robot Policies. arXiv preprint arXiv:2606.03847, 2026. Version 1. 4

[12] Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, and Michael Rabbat. Learning Latent Action World Models In The Wild. arXiv preprint arXiv:2601.05230, 2026. 3

[13] Dibya Ghosh, Homer Rich Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Quan Vuong, Ted Xiao, Pannag R Sanketi, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In Proceed ings of Robotics: Science and Systems, Delft, Netherlands, 2024. 1, 3

[14] Xiaoshen Han, Haoyu Xiong, Haonan Chen, Chaoqi Liu, Antonio Torralba, Yuke Zhu, and Yilun Du. B-spline Pol icy: Accelerating Manipulation Policies via B-spline Action Representations. arXiv preprint arXiv:2607.09648, 2026. 3

[15] Huang Huang, Sriram Yenamandra, Arjun Majumdar, Elie Aljalbout, Tushar Nagarajan, Tsung-Yen Yang, Akshara Rai, Michael Rabbat, Li Fei-Fei, Jiajun Wu, Tingfan Wu, and Franziska Meier. Cross-Embodiment Robot Foundation World Models with Latent Actions. In Proceedings of the 43rd International Conference on Machine Learning, pages 48175–48191. PMLR, 2026. 3

[16] Auke Jan Ijspeert, Jun Nakanishi, Heiko Hoffmann, Peter Pastor, and Stefan Schaal. Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors. Neural Computation, 25:328–373, 2013. 3

[17] Dong Jing, Gang Wang, Jiaqi Liu, Weiliang Tang, Zelong Sun, Yunchao Yao, Zhenyu Wei, Yunhui Liu, Zhiwu Lu, and Mingyu Ding. Mixture of Horizons in Action Chunking. arXiv preprint arXiv:2511.19433, 2025. Version 2. 4

[18] Sunwoo Kim, Jeongjun Kim, and Joseph J. Lim. Time Opti mal Execution of Action Chunk Policies Beyond Demonstration Speed. In International Conference on Learning Repre sentations, 2026. 4

[19] Ge Li, Zeqi Jin, Michael Volpp, Fabian Otto, Rudolf Lioutikov, and Gerhard Neumann. ProDMPs: A Unified Perspective on Dynamic and Probabilistic Movement Primitives. arXiv preprint arXiv:2210.01531, 2022. 3

[20] Haoyun Liu, Jianzhuang Zhao, Xinyuan Chang, Tianle Shi, Chuanzhang Meng, Jiayuan Tan, Feng Xiong, Tong Lin, Dongjie Huo, Mu Xu, SongLin Dong, Zhiheng Ma, Yihong Gong, and Sheng Zhong. Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models. arXiv preprint arXiv:2603.01766, 2026. 3

[21] Yufeng Liu, Hang Yu, Juntu Zhao, Bocheng Li, Di Zhang, Mingzhu Li, Wenxuan Wu, Yingdong Hu, Junyuan Xie, Jun liang Guo, Dequan Wang, and Yang Gao. Learning Native Continuation for Action Chunking Flow Policies. arXiv preprint arXiv:2602.12978, 2026. Version 2. 4

[22] Zhuoyang Liu, Dantong Niu, Botong Zhang, Boning Shao, Jing Wang, Ruijie Zheng, Yuvan Sharma, Runyu Lu, Jiaming Liu, Renrui Zhang, Roei Herzig, Yutong Bai, Ken Gold berg, Danfei Xu, Yuke Zhu, Guanzhi Wang, Linxi Jim Fan,

and Trevor Darrell. Agentic World Action Models. arXiv preprint arXiv:2610.30136, 2026. 4

[23] Junnan Nie, Jiayi Li, Chenghao Liu, Junyi Lao, Jiachen Zhang, Tianle Zhang, Liang Lin, and Songfang Huang. PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking. arXiv preprint arXiv:2606.00537, 2026. Version 3. 4

[24] Open X-Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khaz atsky, Anant Rai, Anchit Gupta, Andrew Wang, Andrey Kolobov, Anikait Singh, Animesh Garg, Aniruddha Kem bhavi, Annie Xie, Anthony Brohan, Antonin Raffin, Archit Sharma, Arefeh Yavary, Arhan Jain, Ashwin Balakr ishna, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Scholkopf, Blake Wulfe, Brian Ichter, Cewu¨ Lu, Charles Xu, Charlotte Le, Chelsea Finn, Chen Wang, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Christopher Agia, Chuer Pan, Chuyuan Fu, Coline Devin, Danfei Xu, Daniel Morton, Danny Driess, Daphne Chen, Deepak Pathak, Dhruv Shah, Dieter Buchler, Di-¨ nesh Jayaraman, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Ethan Foster, Fangchen Liu, Federico Ceola, Fei Xia, Feiyu Zhao, Felipe Vieira Frujeri, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Gilbert Feng, Giulio Schiavi, Glen Berseth, Gregory Kahn, Guangwen Yang, Guanzhi Wang, Hao Su, Hao-Shu Fang, Haochen Shi, Henghui Bao, Heni Ben Amor, Henrik I Christensen, Hiroki Furuta, Homanga Bharadhwaj, Homer Walke, Hongjie Fang, Huy Ha, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jae hyung Kim, Jaimyn Drake, Jan Peters, Jan Schneider, Jasmine Hsu, Jay Vakil, Jeannette Bohg, Jeffrey Bingham, Jeffrey Wu, Jensen Gao, Jiaheng Hu, Jiajun Wu, Jialin Wu, Jiankai Sun, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jimmy Wu, Jingpei Lu, Jingyun Yang, Jitendra Malik, Joao˜ Silverio, Joey Hejna, Jonathan Booher, Jonathan Tompson,´ Jonathan Yang, Jordi Salvador, Joseph J. Lim, Junhyek Han, Kaiyuan Wang, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Black, Kevin Lin, Kevin Zhang, Kiana Ehsani, Ki ran Lekkala, Kirsty Ellis, Krishan Rana, Krishnan Srinivasan, Kuan Fang, Kunal Pratap Singh, Kuo-Hao Zeng, Kyle Hatch, Kyle Hsu, Laurent Itti, Lawrence Yunliang Chen, Lerrel Pinto, Li Fei-Fei, Liam Tan, Linxi ”Jim” Fan, Lionel Ott, Lisa Lee, Luca Weihs, Magnum Chen, Marion Lepert, Marius Memmel, Masayoshi Tomizuka, Masha Itk ina, Mateo Guaman Castro, Max Spero, Maximilian Du, Michael Ahn, Michael C. Yip, Mingtong Zhang, Mingyu Ding, Minho Heo, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Muhammad Zubair Irshad, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J Joshi, Niko Suenderhauf, Ning Liu, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Osbert Bastani, Pannag R Sanketi, Patrick ”Tree” Miller, Patrick Yin, Paul

Wohlhart, Peng Xu, Peter David Fagan, Peter Mitrano, Pierre Sermanet, Pieter Abbeel, Priya Sundaresan, Qiuyu Chen, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Mart´ın-Mart´ın, Rohan Baijal, Rosario Scalise, Rose Hendrix, Roy Lin, Runjia Qian, Ruohan Zhang, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Shan Lin, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shubham Tulsiani, Shuran Song, Sichun Xu, Siddhant Haldar, Siddharth Karamcheti, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Subramanian Ramamoorthy, Sudeep Dasari, Suneel Belkhale, Sungjae Park, Suraj Nair, Suvir Mirchandani, Takayuki Osa, Tanmay Gupta, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Thomas Kollar, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Trinity Chung, Vidhi Jain, Vikash Kumar, Vincent Vanhoucke, Vitor Guizilini, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiangyu Chen, Xiaolong Wang, Xinghao Zhu, Xinyang Geng, Xiyuan Liu, Xu Liangwei, Xuanlin Li, Yansong Pang, Yao Lu, Yecheng Jason Ma, Yejin Kim, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Yilin Wu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yongqiang Dou, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yue Cao, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunchu Zhang, Yunfan Jiang, Yunshuang Li, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zehan Ma, Zhuo Xu, Zichen Jeff Cui, Zichen Zhang, Zipeng Fu, and Zipeng Lin. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv preprint arXiv:2310.08864, 2023. 1, 3

[25] Alexandros Paraschos, Christian Daniel, Jan R Peters, and Gerhard Neumann. Probabilistic Movement Primitives. In Advances in Neural Information Processing Systems, 2013. 3

[26] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient Action Tokenization for Vision-Language-Action Models. arXiv preprint arXiv:2501.09747, 2025. 3

[27] Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit Neural Representations with Periodic Activation Functions. arXiv preprint arXiv:2006.09661, 2020. 3

[28] Bahey Tharwat, Yara Nasser, Ali Abouzeid, and Ian Reid. Latent Action Pretraining Through World Modeling. arXiv preprint arXiv:2509.18428, 2025. 3

[29] Haoxuan Wang, Gengyu Zhang, Yan Yan, Yuzhang Shang, Ramana Rao Kompella, and Gaowen Liu. Real-Time Robot Execution with Masked Action Chunking. arXiv preprint arXiv:2601.20130, 2026. Version 1. 4

[30] Sifan Wang, Zehao Dou, Siming Shan, Tong-Rui Liu, and Lu Lu. FunDiff: diffusion models over function spaces for physics-informed generative modeling. Nature Communications, 17:5749, 2026. 3

[31] Tong Yang, Jingkai Jia, Yuecheng Xu, Xueyao Chen, Chi Zhang, and Wenqiang Zhang. Trajectory-Level Continu ous Action Representation for Robotic Manipulation. arXiv preprint arXiv:2608.24111, 2026. 3

[32] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent Action Pretraining from Videos. arXiv preprint arXiv:2410.11758, 2024. 3

[33] Seonghoon Yu, Dongwon Kim, HyungRok Jung, Yoonjae Baek, Byung-kwan Lee, Suha Kwak, and Jeany Son. When to Switch: Reliable Action-Chunk Extension for Vision-Language-Action Models. arXiv preprint arXiv:2610.05719, 2026. Version 1. 4

[34] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Robotics: Science and Systems XIX, 2023. 3