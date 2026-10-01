# Beyond the Current Scene: Event-Referential Grasping with Active View Selection

Hyunjoon Lee<sup>1\*</sup> Haebeom Jung<sup>1\*</sup> Eunsung Cha<sup>1</sup> Daeun Lee<sup>1</sup> Yu-Chiang Frank Wang<sup>2</sup> Jaesung Choe<sup>2</sup> Jaesik Park<sup>1†</sup> <sup>1</sup>Seoul National University <sup>2</sup>NVIDIA

![](images/29b2e1b60d59e6664f71194dc03fbd1c4dc410e2738fdcc6a598b66bccbf07b5.jpg)

![](images/af29ea8e6f14149a4a79ae1da75d3dbba217dada6bf6e8d54813853ac3be265b.jpg)  
Fig. 1: Event-referential grasping. (A) VoLo [1], a VLM orchestrator, correctly identifies the target of the instruction and passes its text description to either a VLA (π<sub>0.5</sub> [2]) or a WAM (Cosmos3-Nano-Policy [3]). Nevertheless, both downstream models fail to ground and grasp the occluded target. (B) Our system instead initializes a spatial belief from past observations, actively selects a viewpoint that reveals the target, and successfully grasps it.

Abstract— A robot that observes people interacting with objects should be able to carry out later requests that refer back to those interactions. Such requests may specify a grasp target by the role it played in a past event rather than by its name or appearance. Moreover, the target may no longer be visible when the robot is asked to act. We present BeyondCSe, a zero-shot robotic grasping system for this event-referential setting. Given the event history and the current scene, the system identifies the requested object or part and localizes it for grasping. If the target is occluded, it combines an event prior recovered from the history with current scene geometry to select camera viewpoints likely to reveal the target. The system uses pretrained models without additional task-specific training. In real-robot experiments with a single wrist-mounted RGB-D camera, it achieves grasp success rates of 76% and 77% for initially visible and occluded targets, respectively, compared with 40% and 55% for the strongest baseline in each condition. On four additional scenes with heavy occlusion, it increases grasp success rates from 75% to 95% while reducing the mean number of views from 3.35 to 2.20, compared with an active-perception baseline given the target’s ground-truth 3D bounding box.

## I. INTRODUCTION

Real-world workspaces have histories: before a robot is asked to act, objects may have been moved, used, and put away. An instruction can refer to this history, as in “Pick up the object I just used.” We call such an instruction event referential: it identifies a target object or part by its role in a past event rather than by attributes in the current scene. The robot must therefore resolve the reference to the correct instance using the event history. If the target is no longer visible, the history must provide not only its identity but also spatial evidence of where it went. The robot can then use this evidence to reach a viewpoint from which the target is visible and graspable.

Language-guided manipulation methods map an instruction and the robot’s current observations to a grasp [2], [3], [4], [5], [6], [7]. These methods typically resolve targets described by name, category, appearance, or part within the current scene. A target defined by its role in a past event, however, cannot be identified from the current scene alone. Recent systems incorporate event history by conditioning actions on past observations [8], [9] or by passing a reasoned target description to a policy [1]. These approaches can recover the identity of a target that is no longer visible, but target identification alone does not provide the current visual evidence needed for grasping. Our method addresses this gap by using the event history to determine both the intended object and where the camera should move to see it. As shown in Figure 1, VoLo [1] correctly identifies the target and delegates grasping to either a vision-languageaction (VLA) model [2] or a world action model (WAM) [3]. Yet neither downstream model can grasp the target while it remains occluded. Our system instead selects a viewpoint that reveals the target before grasping it.

Active perception supplies this missing visual evidence by selecting new viewpoints likely to reveal a hidden target. Prior work derives view scores from the current scene using geometric information gain [10], predicted grasp affordance [11], or object–occluder relations when the target has never been seen [12]. The current scene alone, however, provides limited guidance about an object displaced during an earlier event. Our event-conditioned prior instead uses the target’s own past observations to estimate where it went. Our view-selection objective further models the visibility of likely target regions, discounting candidate views whose rays pass through unobserved space rather than treating that space as free.

To this end, we present BeyondCSe, a zero-shot grasping system that grounds event-referential targets using the event history and current scene, actively seeking new viewpoints when necessary. The video reasoning first associates the instruction with the target in the event history and then attempts to ground it in the current observation. If grounding succeeds, the module passes an action point to the grasping backend. Otherwise, it passes earlier target observations to event-conditioned active perception. These observations are combined with current scene geometry to initialize a volumetric target belief. The active-perception module scores candidate viewpoints by combining the target belief with transmittance-aware visibility and updates the belief after each new observation until the target is found or the sensing budget is exhausted.

Our main contributions are as follows:

• A zero-shot grasping pipeline that grounds eventreferential objects or parts and recovers an event prior with an off-the-shelf multimodal large language model (MLLM) and point tracker.

• A probabilistic volumetric formulation integrating an event-conditioned spatial prior with transmittance-aware visibility for Bayesian belief updates and view selection.

• A real-robot evaluation of grasping and active perception with a wrist-mounted RGB-D camera, including the effects of event-history evidence on search cost and grasping success.

## II. RELATED WORK

## A. Language-Guided Grasping and Grounding

LERF-TOGO [4] and GraspSplats [5] ground language queries in 3D representations built from multiple views.

For point-based grounding, GraspMolmo [7] adapts a pointgrounding MLLM to predict a task-oriented grasp point from a single image, while Point2Act [6] distills multiview MLLM point predictions into a 3D relevancy field. At the system level, VoLo [1] coordinates VLAs, perception models, and action primitives for long-horizon manipulation, including tasks involving memory and complex references. Our setting goes beyond current scene grounding by identifying an object or part through its role in an event history and using its event prior to guide search under occlusion.

## B. History-Dependent Manipulation and Episodic Memory

EgoLoc [13] recovers the past 3D location of an object specified by an image query, while VideoAgent [14] retrieves frames to answer questions about a video. For robot navigation, ReMEmbR [15] queries a robot’s spatio-temporal memory to generate navigation goals. For manipulation, MemoryVLA [8] incorporates past observations into action generation, and MemER [9] selects relevant keyframes to generate instructions for a low-level policy. Complementing these methods, RoboMME [16] systematically evaluates history-dependent manipulation, including ordinal references and memory of temporarily hidden objects. Unlike memory conditioned policies, our system uses off-the-shelf models to recover explicit 3D observations of a target specified by an event-referential instruction without task-specific training.

## C. Active Perception for Occluded-Target Search

Active view planning for grasping selects viewpoints based on geometric information gain [10], predicted grasp affordance [11], or uncertainty in a calibrated grasp-success model [17]. For severe occlusion, VISO-Grasp [12] reasons about object–occluder relations in the current scene to adjust the viewpoint or remove an inferred occluder. Learned approaches couple active perception with manipulation: SaPaVe [18] predicts camera and manipulation actions, while ActiveVLA [19] selects virtual views rendered from reconstructed 3D input. These methods derive search guidance from current observations, grasp predictions, or generic semantic associations. Our system instead derives an instance-specific belief from the target’s event history and updates this belief after each observation. The system then selects viewpoints based on the visibility of likely target locations, discounting rays through unobserved space.

## III. METHOD

## A. Problem Formulation

Given an RGB-D video V of a person’s tabletop event history and a language instruction ℓ issued after the events end, our system aims to obtain a 3D action point $p ^ { \star }$ in the robot base frame. The video is recorded by a wrist-mounted camera that remains stationary during recording, and its final frame serves as the robot’s initial observation $( I _ { 0 } , D _ { 0 } )$ The robot may then move the camera to acquire additional observations $( I _ { t } , D _ { t } )$ , indexed by step t, with camera intrinsics and camera-to-base transform known at every step. The instruction identifies an object or part through a past event.

![](images/0fc17b5f5eb30f951a2a63a709ab93a741d60abd54e20d45c07d11ad5af131ce.jpg)  
Fig. 2: Overview of BeyondCSe. Given an event-referential instruction and a recorded video, the video reasoning identifies and grounds the target. Historical 3D target observations recovered from the video are denoted by P. If the target is occluded, event-conditioned active perception builds a volumetric belief from prior observations and current geometry, then selects visibility-aware viewpoints until the target can be grasped or search terminates.

The target may be indistinguishable from other objects in the initial image or occluded from the initial view.

## B. System Overview

The two modules in Fig. 2 are connected through the target description and spatial observations. If video reasoning (Sec. III-C) obtains a valid 3D action point from the initial observation, the system passes it to grasp validation, skipping target-location search. Otherwise, recovered historical 3D observations are passed to event-conditioned active perception (Sec. III-D) to initialize a target belief together with geometry from the initial observation. New observations support target pointing and update the map and belief; once the target is confirmed, the system proceeds to grasp validation. The pipeline uses an off-the-shelf MLLM [20] and point tracker [21] without additional task-specific training.

## C. Video Reasoning

1) Selecting the referent: Figure 3(A) shows the inputs and outputs of the four stages. RECORD receives the video without the instruction and forms a time-ordered event record $\mathcal { E } ~ = ~ \left( e _ { 1 } , \ldots , e _ { N } \right)$ , where N is the number of recorded events. Each event $\boldsymbol { e } _ { i } = ( a _ { i } , o _ { i } , r _ { i } )$ describes an action, its object, and another involved object, if any. SELECT reads this record and the instruction to produce an appearance description of the requested object or part (target) and an event description that distinguishes it (cue). The target is passed to POINT, and the cue to LOCATE.

2) Grounding in the initial image: LOCATE uses the video and cue to propose a box in $I _ { 0 }$ . POINT receives the target and the corresponding crop, and returns an action pixel. The crop narrows the search region and enlarges small parts, as illustrated by the example in Fig. 3(B). If crop-based pointing returns no usable point, the search region is expanded; if no usable box is available, the full image is used. The returned pixel is mapped to the full initial image $I _ { 0 }$ and lifted to $p ^ { \star }$ using valid depth and the camera pose.

(A) Reasoning Pipeline  
(B) Pointing Example  
![](images/7c5523b8447cebaa75a19c0ee11066c54de0325bce4020f2268d8c665e24ac61.jpg)  
Fig. 3: Video reasoning. (A) Inputs and outputs of RECORD (R), SELECT (S), LOCATE (L), and POINT (P). (B) Direct video-to-point, full-image (R+S+P), and crop-based (R+S+L+P) predictions in the same close-up region. Red dots indicate predicted results; the latter two configurations share RECORD/SELECT outputs.

3) Recovering historical locations: When pointing in the initial image returns no usable point, we search sampled past frames from recent to earlier ones for pixels matching the target. We track these candidates together and reject trajectories that remain observed at the end of the video. The remaining candidates are checked for appearance, most recent first, and the first to pass is selected. We lift its visible track samples with valid depth into the robot base frame to

![](images/b2c2af6e1ac8d5d8efff4a77db83bde1918a2d550e970de1aea4e49bb2744c39.jpg)  
Fig. 4: Event-conditioned active perception. The event track and initial RGB-D observation initialize the target belief and map. Belief-guided view search then acquires physical views, each yielding a current observation that updates both in closed loop. After target confirmation, grasp validation requests another view only when the observed geometry is insufficient.

obtain

$$
\mathcal { P } = ( q _ { 1 } , \ldots , q _ { m } ) ,\tag{1}
$$

Here, m is the number of valid 3D observations, and $q _ { m }$ is the last one. The 3D event track $\mathcal { P }$ initializes the target belief in Sec. III-D.

## D. Event-Conditioned Active Perception

Figure 4 summarizes the loop: at $t { = } 0$ , the event history and current observation initialize the target belief and map. Each view acquired by belief-guided view search then provides the next observation, which updates the map and belief. Once the target is confirmed, grasp validation either returns a plan or requests a target-centered refinement view.

1) Event-Conditioned Belief Initialization: We construct the initial map $\mathcal { M } _ { 0 }$ as a Truncated Signed Distance Function (TSDF) from the initial RGB-D observation $( I _ { 0 } , D _ { 0 } )$ and define $\Omega \subset \mathbb { R } ^ { 3 }$ as the geometrically admissible workspace, retaining regions unresolved by occlusion or missing depth. The map represents observed geometry, whereas $b _ { t } ( x )$ is the target-location probability density over Ω after observation step t and integrates to one. The stationary target has no intervening motion model. We initialize this belief from the 3D event track P in Sec. III-C, without requiring a current target mask, bounding box, or center.

Let $\gamma : [ 0 , L ] \to \mathbb { R } ^ { 3 }$ be the piecewise-linear path through ${ \mathcal { P } } _ { : }$ parameterized by arc length, with $\gamma ( L ) = q _ { m }$ and total length $\begin{array} { r } { L = \sum _ { i = 2 } ^ { m } \| q _ { i } - q _ { i - 1 } \| _ { 2 } } \end{array}$ . Using the terminal segment of length $h = L / \rho$ for $\rho \geq 1$ , we set

$$
\mu _ { e } = q _ { m } , \qquad \Delta ( s ) = \gamma ( s ) - q _ { m } ,\tag{2}
$$

$$
C _ { \mathrm { t a i l } } = \frac { 1 } { h } \int _ { L - h } ^ { L } \Delta ( s ) \Delta ( s ) ^ { \top } d s ,\tag{3}
$$

$$
\Sigma _ { e } = \delta _ { 0 } ^ { 2 } { \bf I } _ { 3 } + C _ { \mathrm { t a i l } } .\tag{4}
$$

Thus, the prior is centered at the last observation and shaped by the terminal path without extrapolating it. Here, $\mathbf { I } _ { 3 }$ is the

$3 \times 3$ identity matrix, and the subscript e denotes the eventconditioned prior. The parameter $\delta _ { 0 } > 0$ sets the minimum spread, and $C _ { \mathrm { t a i l } } ~ = ~ 0$ for a single observation or zerolength path. With $\mathcal { N } _ { \Omega }$ denoting the Gaussian restricted and normalized over Ω, we use

$$
b _ { 0 } ( x ) = ( 1 - \epsilon ) \cdot \mathcal { N } _ { \Omega } ( x ; \mu _ { e } , \Sigma _ { e } ) + \epsilon \cdot \frac { { \bf 1 } _ { \Omega } ( x ) } { | \Omega | } .\tag{5}
$$

Here, 1<sub>Ω</sub> denotes the indicator of Ω, and |Ω| denotes its volume. The mixture weight $0 ~ < ~ \epsilon ~ < ~ 1$ assigns nonzero probability throughout Ω, allowing the search to recover when the historical estimate is inaccurate. Without valid 3D history, $b _ { 0 }$ is uniform over Ω.

2) Belief-Guided View Search: We generate candidate views around high-probability belief regions and retain the feasible set $\Xi _ { t }$ after kinematic, self-collision, and scenecollision checks. For each $\xi \in \Xi _ { t }$ , we render $\widehat { D } _ { t } ^ { \xi }$ from $\mathcal { M } _ { t }$ and evaluate the negative-observation likelihood $\mathcal { L } _ { t } ^ { - } ( x ; \xi ) \in$ $( 0 , 1 ]$ at each target-location hypothesis $x \in \Omega$ . This lookahead likelihood uses the same depth-consistency and targetmiss factors as the subsequent Bayesian update.

When a ray is not terminated by an observed surface, its visibility confidence is attenuated according to the distance traversed through unobserved space:

$$
T _ { t } ( x ; \xi ) = \kappa + ( 1 - \kappa ) \exp [ - \sigma _ { t } d _ { u , t } ( x ; \xi ) ] .\tag{6}
$$

Here, $d _ { u , t } ( x ; \xi ) \ge 0$ is the ray length through unobserved space, $\sigma _ { t } \geq 0$ controls attenuation, and $0 ~ < ~ \kappa ~ \leq ~ 1$ sets a nonzero transmittance floor. Inspired by the accumulated volumetric transmittance used in NeRF [22], we use $T _ { t }$ without learning a radiance field: it instead measures confidence in a hypothetical observation through unobserved space. Together, these components form a coherent probabilistic loop: the event-conditioned spatial prior structures the initial target belief, nonzero observation likelihoods revise it without hard exclusions, and volumetric transmittance discounts views whose apparent informativeness depends on unobserved space. We score each view by the target-belief mass that a negative observation is expected to downweight:

(A) Visible Target  
![](images/1e3e2ec4fd6dc23791e925d67f97a562b836911ec3aed87bb9bbd651306dd59c.jpg)  
Fig. 5: Examples of event-referential instructions. Each row shows sampled frames from the event history video, the initial observation for robot execution, and the instruction issued after recording. The final video frame serves as the initial observation. (A) The requested cup handle is visible, but identifying the cup covering the pear requires the event history. (B) The cube is moved first and is occluded by the wall in the initial observation.

$$
\xi _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { \xi \in \Xi _ { t } } S _ { t } ( \xi ) ,\tag{7}
$$

$$
S _ { t } ( \xi ) = \int _ { \Omega } b _ { t } ( x ) \left[ 1 - \mathcal { L } _ { t } ^ { - } ( x ; \xi ) \right] T _ { t } ( x ; \xi ) d x .\tag{8}
$$

We execute the highest-scoring view that admits a valid motion plan.

3) Map and Belief Update: After executing $\xi _ { t } ^ { \star }$ , we register the acquired keyframe $( I _ { t + 1 } , D _ { t + 1 } , \xi _ { t } ^ { \star } )$ , integrate its depth into $\mathcal { M } _ { t + 1 }$ , and update the same belief scored in Eq. (8). We combine a depth-consistency likelihood $\mathcal { L } _ { \mathrm { d e p } , t + 1 } ( x )$ with a target-miss likelihood $\mathcal { L } _ { \mathrm { m i s s } , t + 1 } ( x )$ when POINT returns NOT-VISIBLE:

$$
b _ { t + 1 } ( x ) \propto b _ { t } ( x ) \mathcal { L } _ { \mathrm { d e p } , t + 1 } ( x ) \mathcal { L } _ { \mathrm { m i s s } , t + 1 } ( x ) .\tag{9}
$$

The depth likelihood remains neutral wherever the observation provides no valid geometric evidence. The target-miss likelihood combines predicted visibility with the probability that POINT misses a visible target. Both likelihoods take values in (0, 1], so negative observations downweight rather than rule out locations.

At each belief-guided view, POINT queries the target description from SELECT. A lifted pixel yields $p ^ { \star }$ and ends target search. Otherwise, the updated posterior is rescored until no feasible view remains or the active view budget is exhausted.

4) Target Confirmation and View Refinement: Once $p ^ { \star }$ is confirmed, we generate and validate grasp candidates for target association and motion feasibility. Following Breyer et al. [10], incomplete target geometry triggers a feasible targetcentered view, map integration, and grasp regeneration. Unlike their setting with a provided target box, our target region becomes available only after event-conditioned search confirms the target. The bounded loop terminates with an executable grasp or abstains if none is found within the active-view budget.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Hardware Setup: We conduct all real-world experiments using a ROBOTIS OMY-F3M robot equipped with a wrist-mounted Intel RealSense D435i RGB-D camera. During video capture, the robot holds the wrist-mounted camera at a fixed observation pose, and the final RGB-D frame becomes the initial observation. All perception models run on a workstation with a single NVIDIA GeForce RTX 4090 GPU. All methods share the robot, camera calibration, joint limits, collision checks, and grasp generation and execution.

2) Dataset: Each episode consists of a recorded interaction video, an initial wrist-camera RGB-D observation, and an instruction. We distinguish visible and occluded conditions by whether the requested object or part is visible in the initial observation. Our tabletop scenes contain common objects such as produce, cans, mugs, containers, flowers, markers, and toy objects, with other objects serving as potential distractors. The events include taking objects from containers, placing objects in sequence, and rearranging objects in different temporal orders. Instructions refer to objects through event order, such as the object moved first or last, and to object parts, such as a flower stem or cup handle. In the occluded condition, scene occluders hide the queried target from the initial camera view. Figure 5 shows example episodes with event-based references and part queries under both conditions.

The visible condition comprises 10 scene–query pairs across four scenes. The occluded condition comprises 10 scenes with two queries per scene, giving 20 scene–query

Reasoned instruction

Original instruction

![](images/cd93cb50a9acc67983c4b01f933eff6b9972b123367af79c61816c10f814eb92.jpg)

![](images/e0b53c24e218d65a9dfbd0445b9d07d1dfe08f0f20b327aedfbea0a3edac0c88.jpg)

![](images/b2870275fb39352cff556f0d0feca3f12e997e2d62583e5cc6d89c791ea83721.jpg)

![](images/45a308010c9ca90e279a558f7420af24c4ac597606d171ebc2daa233b4fb3979.jpg)  
Fig. 6: Real-world grasping performance. Targets are visible or occluded in the initial wrist-camera view, with 50 and 100 trials per method, respectively. Reasoned instructions are grounding queries generated by Qwen3-VL-8B [20] from the history video and original instruction, shared across baselines while retaining their native grounding backbones. LERF\*, P2A, GS, and GM denote LERF-TOGO [4], Point2Act [6], GraspSplats [5], and GraspMolmo [7], respectively. P2A<sup>†</sup> uses RGB-D reconstruction. Colored segments indicate successful grasps or the first unsuccessful stage. Ours is repeated as a common full-system reference across instruction conditions. N/A denotes an unevaluated condition.

TABLE I: Grasp success (%) over 150 trials.
<table><tr><td>Instruction</td><td>LERF*</td><td>P2A</td><td>GS</td><td>P2A†</td><td>Ours</td></tr><tr><td>Original</td><td>0.0</td><td>9.3</td><td>0.0</td><td>22.7</td><td>76.7</td></tr><tr><td>Reasoned</td><td>18.7</td><td>19.3</td><td>34.7</td><td>50.0</td><td></td></tr></table>

pairs. We conduct five real-world trials per scene–query pair, yielding 50 visible and 100 occluded trials, for a total of 150 trials. These 30 scene–query pairs form the evaluation set for comparison with zero-shot grasping methods. To compare our event-conditioned active perception module against other active-perception methods, we record four additional heavily occluded scenes. In these scenes, every queried target is initially out of sight, concealed inside a basket or behind other scene objects, and remains invisible from a fixed bird’seye view. Two examples are shown in Fig. 7(A). All methods within each comparison use the same scenes and queries.

3) Implementation Details: In the original instruction condition, grasping baselines receive current scene observations and the original instruction. In the reasoned instruction condition, a separate Qwen3-VL-8B [20] reasoner converts the history video and instruction into a grounding query. We generate this query once per episode and instruction and share it across baselines while retaining their visual grounding backbones. It is generated separately from the pointing instruction in our video reasoning pipeline (Sec. III-C). Our system uses the same Qwen3-VL-8B for all MLLM stages. We uniformly sample 96 frames from each event history video and use deterministic decoding with temperature zero. For Eqs. (4) and (5), we set $\rho ~ = ~ 2 , ~ \delta _ { 0 } ~ = ~ 0 . 0 5 ~ \mathrm { ~ m ~ }$ , and $\epsilon \ : = \ : 0 . 1$ . For Eq. (6), we set $\kappa ~ = ~ 0 . 3 3$ and estimate $\sigma _ { t }$ from the occupied fraction of voxels newly resolved by the initial depth observation, using $2 . 0 ~ \mathrm { m } ^ { - 1 }$ when this estimate is unavailable. A new map keyframe is added after 0.10 m of camera translation or $1 5 ^ { \circ }$ of rotation. Following the grasp proposal procedure of LERF-TOGO [4], we generate

TABLE II: Real-world active perception results and belief ablations on four heavily occluded scenes (S1–S4), five trials each. View counts are averaged over trials, including failures. Grasp success aggregates all 20 trials.
<table><tr><td rowspan="2">Method</td><td colspan="4">Scene</td><td colspan="2">Overall</td></tr><tr><td>S1</td><td>S2</td><td>S3</td><td>S4</td><td>Views ↓</td><td>Grasp Success ↑</td></tr><tr><td>Breyer et al. [10]</td><td>4.2</td><td>4.2</td><td>2.2</td><td>2.8</td><td>3.35</td><td>15/20 (75%)</td></tr><tr><td>Ours w/o belief weights</td><td>4.0</td><td>4.0</td><td>5.6</td><td>5.0</td><td>4.65</td><td>16/20 (80%)</td></tr><tr><td>Ours w/o transmittance</td><td>2.8</td><td>2.0</td><td>5.2</td><td>2.4</td><td>3.10</td><td>18/20 (90%)</td></tr><tr><td>Ours w/o event prior</td><td>3.8</td><td>4.0</td><td>4.2</td><td>4.0</td><td>4.00</td><td>19/20 (95%)</td></tr><tr><td>Ours</td><td>2.4</td><td>2.0</td><td>2.4</td><td>2.0</td><td>2.20</td><td>19/20 (95%)</td></tr></table>

AnyGrasp [23] candidates from virtual views, pool them, and apply non-maximum suppression to duplicate poses.

4) Metrics: We report successful trials out of all trials for 3D localization, planning, and grasping. For localization evaluation, a human annotator defines a 3D oriented bounding box for each target object or part in the robot base frame. These annotations are withheld from all methods. Localization succeeds when the predicted 3D point lies within the corresponding annotated box without an additional distance margin, and planning succeeds when a feasible grasp plan is found for the localized target. Grasp success measures whether the robot successfully grasps the instructed target. These metrics reflect successive stages: localization enables planning, and a feasible plan enables grasp execution.

## B. Comparison with Zero-Shot Grasping Methods

We compare with LERF-TOGO [4], GraspSplats [5], Point2Act and Point2Act<sup>†</sup> [6], and GraspMolmo [7]. LERF-TOGO and Point2Act use RGB inputs, while GraspSplats, GraspMolmo, and Point2Act<sup>†</sup> use RGB-D. Each baseline is evaluated with both original and reasoned instructions. Multiview baselines receive 30 predefined observations covering the scene. GraspMolmo uses only the initial RGB-D view and is not evaluated when the target is occluded in that view, as indicated by N/A in Fig. 6. Our method starts from the same initial scene and selects additional views as needed. Figure 6 compares system performance under visible and occluded conditions, while Tab. I reports grasp success pooled across all 150 trials.

(A) Heavily occluded scenes  
![](images/9bb1d2ebe3e7b1a79475c238818e718b8a10b72c182a6b11000bbf3eb36d1df9.jpg)

(B) Visualization of view selection  
![](images/d8bfeb95a39819a00ef18e80cd126ba2ed1c55d82f918758badfc624c24e5108.jpg)  
Fig. 7: Real-world active perception. (A) Examples of heavily occluded scenes in which the queried targets are not visible from a fixed bird’s-eye view. Boxed regions indicate locations where the object can be hidden. (B) Camera viewpoints acquired during target search. In this example, our method reveals the target with just one additional view, whereas Breyer et al. requires exploring multiple additional viewpoints.

With original instructions, the MLLM-based Point2Act and GraspMolmo achieve higher localization success on visible targets than LERF-TOGO and GraspSplats, although none of these baselines receives the event history. Providing reasoned instructions improves localization and grasping success for every evaluated baseline.

As shown in Fig. 6, our method localizes 43/50 visible and 88/100 occluded targets, exceeding the strongest baselines by 32 and 20 percentage points, respectively. LOCATE restricts the pointing region to reduce distractor ambiguity, with Fig. 3(B) illustrating its use for a small target part. Grasping succeeds in 38/50 visible and 77/100 occluded trials, compared with 20/50 and 55/100 for the strongest baselines in each condition. Across all 150 trials, our method achieves 76.7% grasp success, compared with 50.0% for the strongest baseline using reasoned instructions (Tab. I). For occluded targets, our pipeline combines active view search with additional observation of local target geometry when needed for grasp validation.

![](images/a743d4c79919b528a8c06afb42960fdeed8a5351abff0c59a67c13e342d53a1c.jpg)  
Fig. 8: Video reasoning examples on EgoDex [25] and EPIC-KITCHENS [26]. Each example pairs sampled event history and an instruction with target-related excerpts from the pipeline stages. Image panels show expanded proposal crops and enlarged final-frame regions, with predicted points.

## C. Active Perception and Belief Ablations

We compare our method with an adapted implementation of the target-driven active-view method of Breyer et al. [10] on these four scenes (S1–S4 in Tab. II).

1) Breyer et al.: Following the original method, we provide Breyer et al. with the required ground-truth 3D bounding box of the target and retain the original TSDFbased ray-casting objective. We replace VGN [24] with the AnyGrasp [23] generator and planning stack shared with our method. All methods stop at the first feasible target grasp, with target association based on the provided box for the baseline and the confirmed target region for ours and its ablations.

2) Belief Ablations: We compare three variants in Tab. II while keeping candidate views, feasibility checks, target confirmation, and the grasp pipeline fixed. Without belief weights replaces the posterior-weighted objective in Eq. (8) with uniform visibility over target hypotheses not yet cleared by depth. The posterior is still updated, but its probability mass no longer affects view ranking. Without transmittance sets σ<sub>t</sub> = 0 in Eq. (6), making $T _ { t } = 1$ and removing attenuation along sight lines through unobserved space. Without event prior replaces the history-derived initial belief with a uniform density while retaining subsequent belief updates and the view objective.

Our full method achieves 95% grasp success with 2.20 views on average, versus 75% success and 3.35 views for Breyer et al. despite its privileged target box. Although the baseline’s box removes localization ambiguity, it neither reveals occluded target geometry nor guarantees grasp success.

Without belief weights, the mean view count is highest at 4.65 and grasp success rate falls to 80%, suggesting that treating all remaining hypotheses equally wastes views on unlikely regions and reduces robustness. Without transmittance, the mean view count rises to 3.10, with S3 requiring 5.2 views, consistent with optimistic scoring of unknownspace sight lines. Without the event prior, grasp success matches that of the full method, but the mean view count rises from 2.20 to 4.00, showing the cost of searching without history-derived spatial guidance. Fig. 7 illustrates these differences in search behavior: in this example, our method reaches a target-revealing viewpoint directly, while the baseline explores additional viewpoints.

## D. Qualitative Video Reasoning on Egocentric Videos

To examine video reasoning beyond our recorded robot scenes, we apply the same pipeline to selected egocentric clips from EgoDex [25] and EPIC-KITCHENS [26]. Without dataset-specific prompt changes, the pipeline selects targets referred to by the order of plate placement or washing and produces 2D points on the requested objects in the final frames, as shown in Fig. 8. In the EPIC-KITCHENS example, which includes large camera viewpoint changes, LOCATE proposes an inaccurate region, but POINT identifies the carrot within the expanded crop. These examples illustrate use beyond our capture setup while showing that region estimation in the final frame can remain imprecise.

## V. CONCLUSION

To fulfill requests referring to a person’s prior interactions, a robot must infer the intended target from the event history, then ground and grasp it in the current scene. We present BeyondCSe, a zero-shot system that connects video reasoning with active view selection for event-referential grasping. The system identifies the requested object or part from the event history and, when the target is occluded, combines the recovered event prior with current scene geometry to obtain the observations needed for grasping. Real-robot experiments demonstrate higher grasp success than the baselines for both visible and occluded targets. Additional experiments under heavy occlusion show that the event prior reduce the number of observations while achieving a higher grasp success rate than the active-perception baseline. Together, these results demonstrate the value of event history for both target identification and spatial search. Our formulation assumes that the target remains stationary during search. Extending the system to interactions in which objects continue to move remains future work.

## REFERENCES

[1] S. Chen, H. Hadfield, A. Zook, M. A. Uy, C. H. Song, E. Coumans, X. Yang, F. Ladhak, Q. Qu, S. Birchfield, et al., “Volo: A physical orchestrator for open-vocabulary long-horizon manipulation,” arXiv preprint arXiv:2606.07723, 2026.

[2] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al., “π<sub>0.5</sub>: a vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

[3] N. Agarwal, A. Ali, J. Allen, M. Antolini, A. Aubame, A. Azzolini, J. Bai, M. Bala, Y. Balaji, J. Bapst, et al., “Cosmos 3: Omnimodal world models for physical ai,” arXiv preprint arXiv:2606.02800, 2026.

[4] A. Rashid, S. Sharma, C. M. Kim, J. Kerr, L. Y. Chen, A. Kanazawa, and K. Goldberg, “Language embedded radiance fields for zero-shot task-oriented grasping,” in Conference on Robot Learning, vol. 229. PMLR, 2023, pp. 178–200.

[5] M. Ji, R.-Z. Qiu, X. Zou, and X. Wang, “Graspsplats: Efficient manipulation with 3d feature splatting,” in Conference on Robot Learning, vol. 270. PMLR, 2025, pp. 1443–1460.

[6] S. M. Kim, H. Heo, J. Kim, Y. Lee, and Y. M. Kim, “Point2act: Efficient 3d distillation of multimodal llms for zero-shot context-aware grasping,” ICRA, 2025.

[7] A. Deshpande, Y. Deng, J. Salvador, A. Ray, W. Han, J. Duan, R. Hendrix, Y. Zhu, and R. Krishna, “Graspmolmo: Generalizable task-oriented grasping via large-scale synthetic data generation,” in CoRL, vol. 305, 2025, pp. 2983–3007.

[8] H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang, “Memoryvla: Perceptual-cognitive memory in vision-language-action models for robotic manipulation,” in ICLR, vol. 2026, 2026, pp. 18 567–18 602.

[9] A. Sridhar, J. Pan, S. Sharma, and C. Finn, “Scaling up memory for robotic control via experience retrieval,” in ICLR, vol. 2026, 2026, pp. 97 142–97 166.

[10] M. Breyer, L. Ott, R. Siegwart, and J. J. Chung, “Closed-loop nextbest-view planning for target-driven grasping,” in IROS. IEEE, 2022, pp. 1411–1416.

[11] X. Zhang, D. Wang, S. Han, W. Li, B. Zhao, Z. Wang, X. Duan, C. Fang, X. Li, and J. He, “Affordance-driven next-best-view planning for robotic grasping,” arXiv preprint arXiv:2309.09556, 2023.

[12] Y. Shi, D. Wen, G. Chen, E. Welte, S. Liu, K. Peng, R. Stiefelhagen, and R. Rayyes, “Viso-grasp: vision-language informed spatial object-centric 6-dof active view planning and grasping in clutter and invisibility,” in IROS. IEEE, 2025, pp. 14 931–14 938.

[13] J. Mai, A. Hamdi, S. Giancola, C. Zhao, and B. Ghanem, “Egoloc: Revisiting 3d object localization from egocentric videos with visual queries,” in ICCV. IEEE, 2023, pp. 45–57.

[14] X. Wang, Y. Zhang, O. Zohar, and S. Yeung-Levy, “Videoagent: Longform video understanding with large language model as agent,” in ECCV. Springer, 2024, pp. 58–76.

[15] A. Anwar, J. Welsh, J. Biswas, S. Pouya, and Y. Chang, “Remembr: Building and reasoning over long-horizon spatio-temporal memory for robot navigation,” in ICRA. IEEE, 2025, pp. 2838–2845.

[16] Y. Dai, H. Fu, J. Lee, Y. Liu, H. Zhang, J. Yang, C. Finn, N. Fazeli, and J. Chai, “Robomme: Benchmarking and understanding memory for robotic generalist policies,” arXiv preprint arXiv:2603.04639, 2026.

[17] B. Lei, W. Jiang, and K. Daniilidis, “ActiveGrasp: Information-guided active grasping with calibrated energy-based model,” in CVPR, 2026.

[18] M. Liu, E. Zhou, C. Chi, Y. Han, S. Rong, L. Chen, P. Wang, Z. Wang, and S. Zhang, “Sapave: Towards active perception and manipulation in vision-language-action models for robotics,” arXiv preprint arXiv:2603.12193, 2026.

[19] Z. Liu, Y. Gu, Y. Wang, X. Xue, and Y. Fu, “Activevla: Injecting active perception into vision-language-action models for precise 3d robotic manipulation,” arXiv preprint arXiv:2601.08325, 2026.

[20] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[21] N. Karaev, Y. Makarov, J. Wang, N. Neverova, A. Vedaldi, and C. Rupprecht, “Cotracker3: Simpler and better point tracking by pseudo-labeling real videos,” in ICCV. IEEE, 2025, pp. 1–10.

[22] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[23] H.-S. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y. Xie, and C. Lu, “Anygrasp: Robust and efficient grasp perception in spatial and temporal domains,” T-RO, vol. 39, no. 5, pp. 3929–3945, 2023.

[24] M. Breyer, J. J. Chung, L. Ott, S. Roland, and N. Juan, “Volumetric grasping network: Real-time 6 dof grasp detection in clutter,” in CoRL, 2020.

[25] R. Hoque, P. Huang, D. Yoon, J. Zhang, et al., “Egodex: Learning dexterous manipulation from large-scale egocentric video,” in ICLR, vol. 2026, 2026, pp. 4218–4237.

[26] D. Damen, H. Doughty, G. M. Farinella, S. Fidler, A. Furnari, E. Kazakos, D. Moltisanti, J. Munro, T. Perrett, W. Price, et al., “The epic-kitchens dataset: Collection, challenges and baselines,” TPAMI, vol. 43, no. 11, pp. 4125–4141, 2020.