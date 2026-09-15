# LG-VLN: A ZERO-SHOT VISION-AND-LANGUAGE NAVIGATION FRAMEWORK WITH LANGGRAPH STATE ORCHESTRATION

Jianhe Zhao<sup>1</sup>, Yanhua Qiu<sup>1</sup>∗, Zhiyu Zhang<sup>2</sup>, Zibo Zhao<sup>3</sup>, Jinhua Xie<sup>1</sup>

<sup>1</sup>School of Geodesy and Geomatics, Wuhan University, Wuhan, China

<sup>2</sup>School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan, China <sup>3</sup>Academy of Advanced Interdisciplinary Studies, Wuhan University, Wuhan, China

## ABSTRACT

Continuous-environment vision-and-language navigation (VLN-CE) requires interpreting naturallanguage instructions in unseen 3D environments and executing continuous low-level actions. Existing methods often depend on LiDAR, panoramic cameras, or extra sensors; separate geometricmapping and semantic-navigation visual representations can cause long-trajectory spatial-semantic inconsistencies. We propose LG-VLN, a monocular zero-shot framework with shared visual features and LangGraph-based state orchestration. An online feed-forward 3D reconstruction network predicts depth, camera poses, and dense point clouds for agent-pose estimation and global map fusion. Geometry and navigation share dense CleanDIFT features: semantic consistency rejects incorrect inter-frame correspondences, while target-instance constraints define visual references whose similarity combines with local BLIP-2 image-text relevance to form a semantic value map. LangGraph represents instruction parsing, geometric perception, semantic value updates, path planning, action execution, and failure recovery as a directed state graph with conditional transitions, persistent state, and modular recovery mechanisms. On a fixed 550-episode subset of the R2R-CE val-unseen split, LG-VLN achieves 21.3% success and 12.1% success weighted by path length. Ablations show shared semantic features improve navigation, further boosted by combining visual similarity and image-text relevance. Results establish shared visual representations and explicit state orchestration as efective for zero-shot VLN-CE using monocular RGB alone. Code will be publicly released for reproducibility.

Keywords Vision-and-Language Navigation · Shared Visual Features · LangGraph State Graph

## 1 Introduction

Vision-and-Language Navigation (VLN) requires an agent to interpret natural-language instructions and reach a target location in an unfamiliar three-dimensional environment, making it an important task in embodied intelligence [1]. Early VLN methods typically operate on discrete navigation graphs, where agents move between predefined waypoints along known topological connections [1]. Vision-and-Language Navigation in Continuous Environments (VLN-CE), in contrast, assumes neither a prior map nor predefined navigation points. Agents must instead generate low-level control actions directly from sensory observations while navigating through continuous space [2, 3]. This setting more closely reflects the conditions faced by real-world robots and requires close integration of environmental perception, instruction understanding, and action selection [2]. Recent advances in vision-language foundation models have also increased interest in zero-shot VLN-CE [4, 5, 6, 7, 8]. These methods use pretrained models to parse instructions, identify semantically relevant locations, and select actions without access to prior maps or task-specific training, allowing them to navigate unseen environments and follow previously unseen instructions.

Despite recent progress, existing zero-shot VLN-CE methods still face several limitations in system design. Many approaches rely on additional sensors or observation modalities, such as depth sensors, panoramic images, and odometry, to recover environmental geometry, generate candidate waypoints, and estimate the robot’s pose [2, 9, 5, 8]. While these inputs can improve spatial perception and path planning, they also increase system complexity and the cost of real-world deployment. Another limitation lies in how semantic value maps are constructed. Most existing methods measure the similarity between the language instruction and either the current observation or local image regions, then project the resulting scores onto a two-dimensional map. Directly mapping language similarity to 2D regions often produces coarse spatial representations in which neighboring pixels receive nearly identical values. This limits finegrained semantic grounding and makes it dificult to distinguish the target from contextual or irrelevant regions within the same view [5, 8, 10]. Some methods aggregate historical observations through weighted fusion across frames, but they generally do not maintain explicit correspondences between pixels or target regions over time, nor do they explicitly enforce temporal consistency [8, 10]. Changes in viewpoint, target scale, or occlusion can therefore cause semantic values to drift, making them less reliable for fine-grained local action selection. These local perception issues are compounded by limited system-level control over long-horizon navigation. VLN-CE tasks may involve multiple sub-instructions that must be completed in sequence, while errors such as false target detections, unreachable subgoals, local planning failures, and execution failures can occur along the way [11, 12]. Previous studies have addressed parts of this problem through instruction decomposition, progress estimation, historical memory, phase transitions, and replanning [13, 14, 12]. However, without a unified and verifiable state representation, it remains dificult for the system to identify persistent errors and recover from false detections, unreachable subgoals, or repeated planning failures. As instructions become longer and navigation episodes extend over greater distances and durations, errors in perception, planning, and control can accumulate, reducing the stability of the overall system [11].

Beyond perception and planning, a VLN-CE system must also track sub-instruction progress, map states, and action feedback throughout navigation, while recovering from failures when necessary. These elements can be maintained in a shared state that records task progress, the active sub-instruction, the current map state, and action history. With this representation, perception, mapping, constraint evaluation, path planning, and action execution can be coordinated through well-defined state transitions. Recent studies have distributed navigation functions across multiple collaborating agents or exposed them through tool-calling interfaces, suggesting that modular workflows provide a practical structure for embodied navigation systems [15, 16].

We implement this stateful navigation workflow using LangGraph [17]. Shared state, conditional transitions, controlled loops, and checkpointing allow the system to explicitly track sub-instruction transitions, perception scheduling, navigation phases, and recovery states [17, 18]. We define the node functions, shared state, and transition conditions according to the requirements of VLN-CE and organize them into a unified workflow. LangGraph primarily serves to coordinate state management across modules and to maintain an explicit record of module execution and state changes during navigation.

In summary, we propose LG-VLN, a monocular zero-shot Vision-and-Language Navigation framework based on shared dense visual features. Unlike existing methods that use separate feature extraction pipelines for 3D reconstruction and semantic navigation, LG-VLN supports both tasks with a common set of dense difusion features. During mapping, semantic consistency is used to reject incorrect cross-frame correspondences, reducing point-cloud outliers and improving global geometric consistency. During navigation, an open-vocabulary vision-language model identifies semantic reference pixels according to the active sub-instruction, and the difusion features propagate this information across frames to construct a dense semantic value map. At the system level, LG-VLN organizes perception, constraint evaluation, planning, and execution as computational nodes within a shared-state workflow controlled by a graph-structured state machine. This architecture coordinates sub-instruction transitions, supports failure recovery, and provides a unified mechanism for managing state throughout navigation.

## The main contributions are summarized as follows:

1. LG-VLN operates using only monocular RGB observations, without requiring active depth sensors, LiDAR, or panoramic cameras. The depth estimates, camera poses, and dense point clouds needed for navigation are produced online by a feed-forward three-dimensional reconstruction network, reducing hardware requirements and calibration complexity.

2. LG-VLN also uses a shared visual representation for geometric reconstruction and semantic navigation. The same dense difusion features are used to filter incorrect correspondences during mapping and to construct the semantic value map during navigation. Sharing the feature space allows geometric reasoning and semantic guidance to remain consistent across these two stages.

3. For long-horizon navigation, we formulate VLN-CE as a stateful workflow and implement its control logic as a graphstructured state machine using LangGraph. The workflow explicitly tracks sub-instruction transitions, conditional control flow, navigation failures, and checkpoint states. This design provides a unified mechanism for tracing execution, inspecting system states, and implementing recovery strategies.

## 2 Related Work

## 2.1 Vision-Language Navigation

Vision-language navigation (VLN) has evolved from end-to-end methods on discrete navigation graphs to continuous environments that require memory, hierarchical planning, and low-level control. ETPNav maintains an online topological graph of previously visited waypoints and decomposes VLN-CE into high-level topological planning and low-level obstacle avoidance [9]. MapGPT represents an online topological map in language, enabling a large language model (LLM) to incorporate global spatial structure when planning multistep waypoint candidates [19]. These methods show that compact topological memory can reduce short-sighted decisions based only on local observations and provide useful spatial context for long-horizon planning.

Recent work has also formulated VLN as an explicit reasoning process with LLMs and vision-language models (VLMs). NavGPT performs zero-shot action prediction using observation descriptions, trajectory history, and candidate directions [4]. NavGPT-2 aligns visual features with an LLM and combines them with a navigation policy network, reducing the performance gap between general-purpose models and specialized VLN systems [20]. DiscussNav coordinates multiple experts for instruction understanding, environmental perception, and task completion estimation, reducing the instability associated with a single self-reasoning process [21]. Although these methods make navigation decisions more interpretable, their reasoning outputs still need to be grounded in stable and executable motion goals in continuous environments.

Several methods connect language reasoning with continuous control through value maps and constraint-aware planning. VLFM, or Vision-Language Frontier Maps, combines VLM predictions with an occupancy map to construct a language-conditioned frontier value map for zero-shot semantic navigation [5]. InstructNav uses a Dynamic Chainof-Navigation to handle diferent forms of natural-language navigation instructions and converts language plans into robot trajectories using multi-source value maps [6]. CA-Nav formulates zero-shot VLN-CE as constraint-aware subinstruction completion and dynamically updates its plan through constraint state management and cross-modal value maps [8]. However, existing value estimation methods typically operate on images, 2D bounding boxes, or sparse frontiers. These coarse representations provide limited pixel-level semantic grounding, making it dificult to reliably distinguish instruction-relevant targets from background context. Some methods also assume access to depth, panoramic observations, or explicit poses, which does not match the sensing conditions of monocular RGB-only navigation.

Long-horizon tasks and restricted observations pose additional challenges for VLN. LH-VLN introduces a multistage, long-distance navigation benchmark and uses multigranularity dynamic memory to retain information across subtasks [11]. StreamVLN models navigation as online decision-making over a video stream and balances historical context with inference eficiency through rapidly updated context and slowly updated memory [12]. For monocular observations, NaVid formulates VLN as video-to-action prediction [22], while MonoDream uses a unified navigation representation and latent panoramic imagination to recover spatial information that is missing from narrow-field-of view inputs [23]. Prior work has examined semantic reasoning, mapping, and long-context modeling, but geometric perception and semantic navigation are still commonly handled by separate pipelines with distinct visual features. Longhorizon execution is also often managed through rigid control loops or implicit history representations. To address these limitations, we unify geometric perception and semantic decision-making in a shared dense visual feature space. Combined with a stateful graph orchestration mechanism, this design explicitly manages long-horizon execution for monocular zero-shot VLN-CE.

## 2.2 Feed-Forward 3D Reconstruction

Feed-forward 3D reconstruction estimates depth, point clouds, camera parameters, and other 3D representations from a single image, multiple views, or video in one or a few forward passes. Traditional Structure from Motion (SfM) and Simultaneous Localization and Mapping (SLAM) pipelines rely on feature matching, pose-graph optimization, and iterative refinement for each scene. Recent visual geometry foundation models instead learn from large-scale data with Transformer architectures and generalize more efectively across scenes.

For monocular geometry estimation, Depth Anything V2 improves the robustness and detail of depth prediction through large-scale pseudo-labeling on real images and supervision from synthetic data [24]. UniDepth and UniDepthV2 estimate metric 3D structure and camera-related representations from a single image, with a focus on zero-shot generalization across domains [25, 26]. MoGe formulates open-domain monocular geometry estimation as afine-invariant point-map prediction and jointly recovers depth, point clouds, and field of view [27]. These models reduce the need for active depth sensors in robotic systems. However, frame-wise predictions do not explicitly enforce temporal consistency and therefore cannot guarantee consistent scale and pose across a sequence.

Multiview models further integrate camera estimation, correspondence matching, and dense reconstruction. DUSt3R formulates two-view and multiview reconstruction as point-map regression, predicting dense 3D structure without known camera parameters [28]. Building on DUSt3R, MASt3R introduces dense local features and reciprocal matching to improve correspondence estimation under large viewpoint changes [29]. CUT3R maintains a persistent internal state while processing continuous image streams and produces online point maps in a shared coordinate system, making it well suited to the continuous perception requirements of robotic systems [30].

More general models have begun to predict geometry, trajectories, and renderable representations within a unified framework. VGGT, or the Visual Geometry Grounded Transformer, uses a unified Transformer to estimate camera parameters, depth, point maps, and point tracks [31]. VGGT-SLAM incorporates these outputs into dense RGB SLAM for continuous pose estimation and submap alignment [32]. AnySplat extends feed-forward reconstruction to 3D Gaussian Splatting for eficient novel-view synthesis [33], while UniForward combines Gaussian Splatting with open-vocabulary semantic field reconstruction [34]. This line of work reflects a shift from isolated depth estimation toward joint modeling of geometry, semantics, and renderable representations. These models provide a natural basis for spatial memories that navigation systems can update online. Our work reuses their visual features for both geometric mapping and language-guided navigation, avoiding redundant visual encoding.

## 2.3 Agent Orchestration Frameworks

Robotic systems have long used finite-state machines, behavior trees, and execution graphs to represent task stages and state transitions. Behavior trees support hierarchical composition and reactive reevaluation, making them well suited to interpretable conditional branches and local recovery procedures [35]. Execution monitoring methods also address deviation detection, diagnosis, and recovery [36], while behavior trees can organize fault-handling actions [37]. Task and Motion Planning (TAMP) jointly considers symbolic task constraints and geometric feasibility [38]. However, these methods generally assume predefined skills and explicit task models, and they do not specify how open-vocabulary semantics, online maps, and action histories should be maintained within a shared state.

Embodied language agents combine high-level language planning with environmental feedback and action feasibility. SayCan filters language-model-generated steps according to skill afordances [39]. Inner Monologue iteratively revises plans based on success signals and scene feedback [40]. PaLM-E incorporates continuous visual observations into an embodied multimodal language model [41]. Research on active perception further suggests that sensing frequency and observation actions should depend on task requirements [42]. These studies address language planning, feedback-based correction, and task-driven perception, but pay less attention to how geometric maps, semantic evidence, progress states, and action histories are jointly maintained over long-horizon navigation.

LLM-based agent systems typically organize functional modules through roles, messages, and task dependencies. MetaGPT encodes standard operating procedures as collaboration protocols among roles [43]. AutoGen represents language models, human participants, and tools as customizable conversational components [44]. GPTSwarm models a multi-agent system as an optimizable node–edge computation graph [45]. These frameworks provide general abstractions for modular reasoning and task allocation, but their internal states rarely correspond directly to action outcomes, spatial constraints, or environmental changes during robotic execution.

At the runtime level, ExoFlow supports persistent execution and failure recovery for Directed Acyclic Graph (DAG) workflows [46], while LangGraph provides a graph-based abstraction for stateful, multi-participant LLM applica tions [17]. Their checkpointing and state reentry mechanisms provide useful design references for long-running workflows. However, the fault-tolerance mechanisms of general-purpose workflow systems do not directly address the needs of embodied navigation. Continuous VLN instead requires an auditable execution graph that supports environmentdriven state updates, conditional transitions, checkpoint recovery, and task-level replanning within a unified runtime.

## 3 Method

## 3.1 Problem Formulation and Spatial Representation

## 3.1.1 Task Definition

We consider zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE). Given a naturallanguage instruction I, the agent receives a monocular RGB observation $\mathbf { I } _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ at decision step t and selects an action $a _ { t } \in { \mathcal { A } }$ , where $\mathcal { A } = \{ \alpha _ { \mathrm { F } } , \alpha _ { \mathrm { L } } , \alpha _ { \mathrm { R } } , \alpha _ { 0 } \}$ corresponds to moving forward, turning left, turning right, and stopping, respectively. Forward and turning actions use fixed translational and angular increments.

Unlike conventional VLN-CE systems that rely on simulator-provided depth or navigation-specific training, our agent operates without depth observations, prior maps, or task-specific navigation training. All geometric information needed for mapping and planning is inferred online from monocular RGB observations. After each motion, we also compare the displacement implied by the issued action with that estimated from visual geometry. This motion-consistency signal is used to detect unreliable traversability estimates and navigation failures.

## 3.1.2 Spatial Representation

We maintain a world coordinate frame W, a body frame $B _ { t }$ , a camera frame $\mathcal { C } _ { t }$ , and an agent-centered local map frame $\mathcal { L } _ { t } .$ . The agent pose at decision step t is represented by the body-to-world transformation $\mathcal { T } _ { B _ { t } } ^ { \mathcal { W } }$ , which specifies the orientation and position of the body frame $\hat { B _ { t } }$ with respect to the world frame W. The world frame is initialized from the first reliable geometric observation, with its vertical axis aligned with the estimated ground normal. The horizontal axes of the local map remain aligned with W, while its origin moves with the agent. Thus, the map orientation remains fixed as the agent changes its heading.

The transformation from the current camera frame to the world frame is denoted by $\mathcal { T } _ { \mathcal { C } _ { t } } ^ { \mathcal { W } }$ . Since monocular reconstruction may exhibit small frame-to-frame scale variations even after approximate metric initialization, inter-frame registration is modeled as a bounded similarity transformation in Sim(3), with the scale factor restricted to a predefined interval. Detailed conversions among camera, body, pixel, and map coordinates are provided in Appendix A.1.

## 3.1.3 Global Map and Local Window

Navigation relies on two complementary spatial representations. An extensible global bird’s-eye-view map, referred to as the Global Top-Down Map (GTM) and denoted by $\mathbf { G } _ { t } ,$ , stores geometric and semantic information accumulated throughout the episode, while a fixed-size local window centered on the agent is used for waypoint generation and short-range planning. Both representations are aligned with W.

At each decision step, observations are first reconstructed in the current camera frame, transformed into the world frame, and then projected onto the local map. The updated local state is subsequently fused into the GTM. This design preserves long-term spatial memory while keeping local planning computationally tractable.

## 3.2 System Framework

We propose a monocular zero-shot VLN-CE framework that converts natural-language instructions into executable navigation through four interacting components: instruction constraint tracking, semantic-geometric reconstruction, world-aligned semantic mapping, and hierarchical navigation. A stateful execution graph implemented with LangGraph coordinates these components and maintains persistent navigation state across perception, planning, action execution, and recovery. Fig. 1 provides an overview of the framework.

At the beginning of each episode, the instruction is decomposed into an ordered sequence of sub-instructions. Each sub-instruction is associated with explicit completion constraints that describe relevant objects, locations, turns, and relative movement directions. These constraints define both the semantic targets for navigation and the criteria used to assess navigation progress.

For spatial perception, VGGT reconstructs monocular geometry from incoming RGB observations [31]. Dense semantic features from CleanDIFT [47], a denoised variant of difusion-based correspondence features [48], are used to reject semantically inconsistent geometric correspondences before inter-frame registration. The reconstructed geometry is then transformed into a world-aligned bird’s-eye-view map that accumulates traversability, occupancy, exploration, and semantic-value information.

Semantic guidance comes from two complementary visual signals. CleanDIFT provides dense correspondence-based visual relevance, while BLIP-2 estimates image–text relevance for the active sub-instruction. The two signals are fused based on observation confidence and accumulated in the global value map.

Navigation follows a hierarchical design. The high-level planner generates reachable semantic waypoint candidates from the local map and selects a target based on instruction relevance, navigation history, and temporal consistency. The low-level planner then uses the Fast Marching Method (FMM) [49] to compute a collision-free short-term goal and converts it into discrete motion commands.

The system operates in a closed loop, with each new observation updating the geometry, semantic evidence, instruction progress, and navigation state. When the execution graph detects sustained lack of progress, repeated visits, or heading oscillation, control is transferred to a recovery subgraph that revises perception or the navigation target before returning to normal execution.

![](images/409283c0ba22112833b386982a0cb58fb46021ddca61df7769056bfedcd7b511.jpg)  
Fig. 1. Overview of the proposed monocular zero-shot VLN-CE framework.

## 3.3 Stateful Navigation Graph with LangGraph

A central component of our framework is a stateful navigation graph that explicitly models the interactions among perception, reasoning, planning, execution, and recovery. Rather than running these modules as a fixed sequential pipeline, LangGraph maintains a persistent shared state and dynamically routes execution based on navigation progress and failure conditions.

## 3.3.1 Stateful Execution Model

At step t, the shared state contains the active sub-instruction, current pose, global map, navigation history, and graphlevel control status. We denote this state compactly by $\mathbf { s } _ { t } .$ . Given the current observation $o _ { t } ,$ , graph execution is written as

$$
n _ { t } = \rho ( \mathbf { s } _ { t } ) , \qquad \mathbf { s } _ { t } ^ { + } = F _ { n _ { t } } \big ( \mathbf { s } _ { t } , o _ { t } \big ) ,\tag{1}
$$

where $\rho$ selects the active graph branch and $F _ { n _ { i } }$ denotes the state transition performed by that branch. The observation $o _ { t }$ contains the current RGB frame together with the outcome of the preceding action.

The parent graph contains four semantic branches: initialization, normal navigation, recovery, and termination. These branches share the same persistent geometric map and navigation history, allowing recovery to revise the current decision process without discarding spatial information accumulated before the failure.

This graph structure serves two purposes. It separates high-level execution logic from individual perception and planning modules, allowing each module to operate on a consistent navigation state. It also makes non-monotonic navigation behavior explicit: the agent can return from planning to perception, temporarily leave the main navigation loop for recovery, and then resume the interrupted sub-instruction without restarting the episode.

## 3.3.2 Initialization Subgraph

The initialization subgraph prepares the language, geometric, and orientation states needed for subsequent navigation. It first decomposes the instruction and aligns open-vocabulary entities with the representations used for object detection and visual retrieval. It then estimates the metric reconstruction scale from ground observations and performs an initial panoramic scan.

The panoramic scan collects complementary geometric and semantic observations from multiple headings. For subinstructions without an explicit turning requirement, the agent uses the accumulated semantic evidence to select a favorable initial orientation. If the instruction explicitly requires a turn, the specified direction takes precedence and remains an unsatisfied directional constraint rather than being overridden by semantic orientation selection.

Initialization is complete only when instruction parsing, semantic alignment, metric scale recovery, and the initial observation scan have all produced valid states. The agent then proceeds with incremental mapping and waypoint planning. Detailed scale-validity criteria and initialization parameters are provided in Appendix A.3.

## 3.3.3 Closed-Loop Navigation

After initialization, the agent repeatedly performs geometric updates, semantic perception, constraint evaluation, waypoint selection, local planning, and action execution.

Geometric reconstruction and pose estimation are updated from every valid observation because they define the metric state used for mapping and control. More expensive open-vocabulary semantic perception is triggered only when needed. A new semantic observation is requested when the active sub-instruction changes, after recovery, or when the viewpoint has changed suficiently to provide new task-relevant evidence. Otherwise, the most recent semantic observation is retained in the graph state. The detailed scheduling policy is provided in Appendix A.2

After perception, the new geometric and semantic observations are projected into the shared world-aligned map. The instruction module then evaluates the completion constraints associated with the active sub-instruction. Once all constraints are satisfied, the graph advances to the next sub-instruction, suppresses stale target-specific semantic evidence, and requests a fresh semantic observation for the new navigation stage.

If the active sub-instruction remains incomplete, the updated semantic map is used to generate candidate waypoints. A high-level waypoint is selected from the reachable region, and FMM then computes a short-term geometric target. The corresponding discrete action is executed, and its outcome is written back to the graph state, completing the perception– planning–action loop.

## 3.3.4 Navigation Failure Detection

Continuous navigation can fail even when a locally valid path exists. Common failure modes include repeated attempts to traverse an obstacle that has been incorrectly classified as traversable, negligible displacement after forward actions, repeated visits to the same neighborhood, oscillation between similar waypoints, and persistent turning without spatial progress.

We detect these conditions using the estimated trajectory, recent actions, waypoint progress, and motion-consistency observations. Rather than treating a single unsuccessful action as a failure, the graph detects persistent loss of progress over a temporal window. This avoids interrupting normal navigation because of transient reconstruction errors or isolated collisions. Detailed detection criteria and thresholds are provided in Appendix A.2.

Once persistent failure is detected, the current waypoint is suspended and control is transferred to the recovery subgraph.

## 3.3.5 Recovery Subgraph

The recovery subgraph progressively increases the extent to which the current navigation hypothesis is revised. It consists of three recovery levels.

Viewpoint Recovery. The agent first performs a panoramic observation at its current location and reevaluates target localization and instruction relevance across the newly observed headings. This stage addresses failures caused by a limited field of view, temporary occlusion, or an unfavorable camera orientation. If reliable task-relevant evidence is recovered, a new waypoint is generated and the graph returns directly to the main navigation loop.

Semantic Recovery. If viewpoint recovery remains inconclusive, the graph invalidates target-specific semantic observations associated with the current hypothesis and reconstructs them from the latest visual evidence. Object localization, visual references, and instruction-conditioned relevance are then reevaluated while the persistent geometric map is retained.

Navigation Redirection. When the active semantic target does not provide a reliable reachable waypoint, the graph temporarily shifts to geometric exploration. A reachable frontier or unexplored location within the safe connected region is selected as an intermediate target. This target is used only to obtain a new viewpoint and expand the observable space; it does not replace the active instruction constraint.

After successful recovery, temporary planning state is cleared, while the global geometric map, semantic memory, and long-term navigation history are preserved. Semantic perception is refreshed before the graph resumes the interrupted sub-instruction. If repeated recovery attempts remain unsuccessful, the episode is eventually marked as unrecoverable. Additional recovery details are provided in Appendix A.7.

## 3.4 Semantics-Enhanced Monocular 3D Reconstruction

Accurate spatial reasoning is challenging in VLN-CE when only monocular RGB observations are available. We therefore construct an online metric representation by combining feed-forward 3D reconstruction with semantics-aware correspondence filtering. VGGT provides dense geometric predictions, while CleanDIFT filters out visually plausible but semantically inconsistent inter-frame matches.

## 3.4.1 Feed-Forward Reconstruction and Metric Scale

For each RGB observation $\mathbf { I } _ { t } ,$ , VGGT predicts dense relative depth, a 3D point map, and geometric confidence [31]. Since monocular reconstruction does not inherently determine metric scale, we estimate an initial scale using the known camera mounting height $h _ { \mathrm { { c a m } } }$

During initialization, high-confidence points on the observed ground are used to fit a local ground plane. Let $\hat { \mathbf { n } } _ { t }$ and $\hat { b } _ { t }$ denote the unit normal and ofset of the fitted plane, respectively. Since the plane normal is normalized, $| \hat { b } _ { t } |$ gives the camera-to-ground-plane distance in the relative reconstruction space. The resulting per-frame scale estimate is

$$
\hat { s } _ { t } = \frac { h _ { \mathrm { c a m } } } { | \hat { b } _ { t } | } .\tag{2}
$$

During the initialization scan, observations that satisfy the ground-support, plane-fitting, and scale-validity criteria are retained. We denote the corresponding set of valid frames by $\mathcal { F } _ { \mathrm { v a l i d } }$ and estimate the common initial metric scale robustly as

$$
s _ { 0 } = \mathrm { m e d i a n } \left( \left\{ \hat { s } _ { t } ~ | ~ t \in \mathcal { F } _ { \mathrm { v a l i d } } \right\} \right) .\tag{3}
$$

The predicted relative depth and 3D point maps are then rescaled by $s _ { 0 }$ to establish a common approximate metric reference for mapping and navigation. Using multiple initial headings makes the scale estimate less sensitive to local occlusions, ground-segmentation errors, and degenerate plane observations. Detailed ground-plane fitting, frame-validity criteria, and initialization procedures are provided in Appendix A.3.

## 3.4.2 Semantic Outlier Rejection with CleanDIFT

For frames used in inter-frame registration, CleanDIFT extracts dense semantic features from the input images. The resulting feature maps are interpolated to the input-image resolution and $\ell _ { 2 }$ -normalized along the feature dimension, yielding a unit semantic descriptor $\mathbf { f } _ { t } ( \mathbf { u } )$ at each pixel u.

For each candidate 3D correspondence generated by the geometric matching module between consecutive frames, the system measures semantic consistency using the CleanDIFT descriptors at the corresponding image locations. Correspondences with low semantic similarity are rejected, and the remaining correspondences are further filtered using the geometric confidence predicted by VGGT.

This procedure combines geometric reliability from VGGT with semantic consistency from CleanDIFT. Geometric confidence removes points with unreliable reconstruction, while semantic consistency helps reject geometrically plausible correspondences between semantically diferent regions, particularly in repetitive structures and weak-texture areas.

## 3.4.3 Inter-Frame Similarity Transformation Estimation and Global Fusion

Our backend optimization follows the general design of [32]. Although metric scale is initialized during the initialization stage, frame-wise feed-forward reconstruction may still exhibit residual scale variations. We therefore model inter-frame registration as a constrained Sim(3) transformation. For a geometrically consistent 3D correspondence, points in consecutive frames are expected to satisfy

$$
\mathbf { x } _ { i } ^ { t } \simeq s _ { t - 1 , t } \mathbf { R } _ { t - 1 , t } \mathbf { x } _ { i } ^ { t - 1 } + \mathbf { t } _ { t - 1 , t } , \qquad | \log s _ { t - 1 , t } | \leq \epsilon _ { s } ,\tag{4}
$$

where $\mathbf { x } _ { i } ^ { t - 1 }$ and $\mathbf { x } _ { i } ^ { t }$ denote corresponding 3D points in the previous and current camera frames, respectively, $s _ { t - 1 , t } > 0$ is the inter-frame scale factor, $\mathbf { R } _ { t - 1 , t } ~ \in ~ \mathrm { S O } ( 3 )$ is the relative rotation, and $\mathbf { t } _ { t - 1 , t } \ \in \ \mathbb { R } ^ { 3 }$ is the relative translation. The constraint |log $s _ { t - 1 , t } | \ \leq \ \epsilon _ { s }$ bounds frame-to-frame scale variation and prevents a single registration step from introducing an excessive scale change.

The semantically and geometrically filtered correspondences are used to robustly estimate the constrained similarity transformation. RANSAC [50] provides an initial estimate while rejecting geometrically inconsistent matches, after which the transformation parameters are refined with a confidence-weighted robust objective. Details of correspondence construction, semantic filtering, RANSAC estimation, correspondence weighting, and robust Sim(3) refinement are provided in Appendix A.3.

The estimated inter-frame transformation is then used to update the current camera pose and register the reconstructed geometry in the world-aligned map. If a reliable registration cannot be obtained, the previous pose estimate is retained and global map fusion is temporarily suspended. Detailed failure-handling and map-fusion criteria are provided in Appendix A.3.

## 3.5 Instruction Decomposition and Constraint Tracking

## 3.5.1 Semantics-Guided Sub-instruction Decomposition

VLN instructions often describe a sequence of spatially distinct operations. Reasoning over the full instruction at every step can introduce landmarks or actions that are irrelevant to the current navigation stage, potentially interfering with visual grounding [51]. We therefore represent each instruction as an ordered sequence of sub-instructions, each associated with an explicit set of completion constraints.

Candidate boundaries are identified using navigation verbs, sequential connectives, changes in spatial relations, and syntactic structure. A lightweight LLM then normalizes the resulting segments and extracts task-relevant entities and relations. This hybrid decomposition limits reliance on unconstrained language generation while retaining the flexibility needed to handle open-vocabulary instructions.

For each sub-instruction $I _ { k }$ , we construct a constraint set $\mathcal { C } _ { k }$ containing object, location, and direction constraints as needed. Explicit turning commands and relative movement directions are represented separately: turning commands specify a change in orientation, whereas relative movement directions specify displacement with respect to the heading at the start of the sub-instruction. Instructions that require backward motion are normalized into a turn followed by forward movement, allowing the low-level action space to remain unchanged.

## 3.5.2 Sub-instruction Constraint Satisfaction

At decision step t, the graph evaluates all constraints associated with the active sub-instruction $I _ { k _ { t } }$ . The completion state is defined as

$$
\chi _ { t } ^ { \mathrm { d o n e } } = \prod _ { c \in \mathcal { C } _ { k _ { t } } } \mathsf { S a t } _ { t } ( c ) ,\tag{5}
$$

where $\mathbf { S a t } _ { t } ( c ) \in \{ 0 , 1 \}$ indicates whether constraint c is supported by the visual observations, estimated trajectory, and action history available up to step t. The graph proceeds to the next sub-instruction only after all active constraints are satisfied.

Object Constraints. For open-vocabulary landmarks, we use Grounding DINO to obtain language-conditioned detections [52], followed by SAM 2 [53] to refine the corresponding instance regions. Valid pixels in each mask are associated with reconstructed depth values and projected into the world frame. The target position is robustly estimated from the resulting 3D points and updated as new semantic evidence becomes available. An object constraint is considered satisfied when the target has been observed with suficient confidence and the agent has reached the required spatial relationship to it.

Location Constraints. Location constraints represent transitions between semantic regions rather than individual visual detections. We use visual question answering to determine whether the current observation is consistent with a target region, with BLIP-2 image–text relevance used as a fallback when needed [54]. Rather than classifying each frame independently, the graph maintains a short-term region state that records whether the agent is entering, remaining inside, leaving, has previously visited, or is traversing a region. A traversal constraint is therefore satisfied only by a consistent sequence of entry, presence, and exit events, rather than by a single positive classification. The full state transition rules are given in Appendix A.4.

Direction Constraints. For each sub-instruction, the agent’s heading at activation is recorded as the reference orientation. Turning constraints are evaluated using the accumulated heading change relative to this reference. Relative movement-direction constraints, by contrast, are evaluated from the direction of the estimated displacement over a short temporal window. Separating these two cases prevents an in-place rotation from satisfying instructions such as “move to the left” and prevents lateral motion from being interpreted as an explicit “turn left” command. The angular ranges and temporal parameters used for diferent direction constraints are reported in Appendix A.4.

Once a sub-instruction is completed, the graph advances to the next sub-instruction, downweights semantic values associated with the previous target, and refreshes target-specific visual evidence. The geometric exploration history is retained across this transition.

## 3.6 Constraint-Guided Semantic Value Estimation

The active instruction constraints are transformed into a spatial signal for waypoint selection. We obtain this signal by integrating dense visual correspondence with image–text relevance, which provide complementary semantic cues.

Let ${ \bf A } _ { t }$ denote the CleanDIFT-based relevance map between the current observation and the visual references associated with the active constraint, and let $\mathbf { B } _ { t }$ denote the BLIP-2 image–text relevance map. The available semantic observations are fused as

$$
\mathbf { E } _ { t } = \frac { \beta _ { a } q _ { t } ^ { a } \mathbf { A } _ { t } + \beta _ { b } q _ { t } ^ { b } \mathbf { B } _ { t } } { \beta _ { a } q _ { t } ^ { a } + \beta _ { b } q _ { t } ^ { b } + \epsilon } ,\tag{6}
$$

where $q _ { t } ^ { a }$ and $q _ { t } ^ { b }$ represent the reliability of the two semantic sources, and $\beta _ { a } , \beta _ { b }$ determine their relative contributions.   
If one source is unavailable, the remaining source is used to construct the evidence map.

The semantic evidence is then weighted by observation confidence. This confidence accounts for geometric validity, viewing direction, observation range, surface geometry, traversability, and pose reliability. Specifically, observations close to the optical axis and locations supported by reliable reconstruction and registration are assigned higher weights, whereas obstacle regions and geometrically uncertain pixels are down-weighted.

To improve temporal consistency, historical semantic observations are reprojected into the current image using the esti mated inter-frame transformation. Let $\widetilde { \mathbf { V } } _ { t } ^ { \mathrm { i m g } }$ and $\widetilde { \mathbf { Q } } _ { t } ^ { \mathrm { v a l } }$ denote the current-frame value and confidence maps, respectively, and let $\widehat { \mathbf { V } } _ { t - 1  t }$ and $\widehat { \mathbf { Q } } _ { t - 1  t }$ e edenote the corresponding historical estimates after reprojection into the current view.

The navigation mask $\mathbf { M } _ { t } ^ { \mathrm { n a v } }$ first excludes non-traversable pixels and pixels occluded by foreground obstacles. Reprojections within the remaining valid regions are further filtered by image-boundary, positive-depth, and occlusionconsistency constraints, with the resulting validity encoded by ${ \bf M } _ { t } ^ { \mathrm { r e p } }$ . The current and historical observations are then combined through confidence-weighted averaging:

$$
\mathbf { V } _ { t } ^ { \mathrm { i m g } } = \mathbf { M } _ { t } ^ { \mathrm { n a v } } \odot \frac { \widetilde { \mathbf { Q } } _ { t } ^ { \mathrm { v a l } } \odot \widetilde { \mathbf { V } } _ { t } ^ { \mathrm { i m g } } + \eta \mathbf { M } _ { t } ^ { \mathrm { r e p } } \odot \widehat { \mathbf { Q } } _ { t - 1  t } \odot \widehat { \mathbf { V } } _ { t - 1  t } } { \widetilde { \mathbf { Q } } _ { t } ^ { \mathrm { v a l } } + \eta \mathbf { M } _ { t } ^ { \mathrm { r e p } } \odot \widehat { \mathbf { Q } } _ { t - 1  t } + \epsilon } ,\tag{7}
$$

and the corresponding confidence map is updated as

$$
\mathbf { Q } _ { t } ^ { \mathrm { v a l } } = \mathbf { M } _ { t } ^ { \mathrm { n a v } } \odot \mathrm { c l i p } ( \widetilde { \mathbf { Q } } _ { t } ^ { \mathrm { v a l } } + \eta \mathbf { M } _ { t } ^ { \mathrm { r e p } } \odot \widehat { \mathbf { Q } } _ { t - 1  t } , 0 , 1 ) .\tag{8}
$$

Here, $\eta \in [ 0 , 1 ]$ controls the contribution decay of historical observations. Additional implementation details regarding camera-axis confidence, reprojection validity, and local-block interpolation are provided in Appendix A.5.

## 3.7 Global Bird’s-Eye-View Semantic-Geometric Map

The global map $\mathbf { G } _ { t }$ provides a shared representation for geometric reconstruction, semantic grounding, and navigation planning. It contains traversability, occupancy, exploration, observation-count, semantic-value, and dense semanticfeature layers. The exploration state is derived from accumulated observations, while semantic values and features are associated with the active instruction.

For each observation, reconstructed 3D points are transformed by $\mathcal { T } _ { \mathcal { C } _ { t } } ^ { \mathcal { W } }$ and rasterized into the agent-centered local map.   
Ground-supported points contribute to the traversability layer, whereas elevated structures contribute to occupancy.   
When conflicting evidence falls within the same cell, occupancy takes precedence to maintain conservative navigation.

The motion-consistency signal provides an additional source of geometric supervision. If the agent issues a forward command but the estimated pose shows substantially less displacement than expected, cells immediately along the attempted direction are marked as potentially occupied. This feedback allows the map to correct visually inferred traversable regions that repeatedly prove impassable during execution.

The semantic-value and semantic-feature layers are projected using the same coordinate transformation. Repeated observations of the same grid cell are fused according to observation confidence, assigning greater weight to geomet rically reliable measurements obtained from favorable viewpoints. Semantic features are re-normalized after fusion. When the active sub-instruction changes, semantic values associated with the previous target are attenuated rather than discarded, preserving useful exploration context while giving priority to the new target.

A fixed-size local window is extracted from the global map for waypoint selection and FMM planning. Further details of map fusion are provided in Appendix A.6.

## 3.8 Waypoint Selection and Local Planning

## 3.8.1 Reachable Region and Candidate Generation

Waypoint generation is restricted to locations that are both traversable and safely reachable from the current agent position. Let $\Omega _ { t } ^ { \mathrm { o c c } }$ denote the occupied cells in the local map. We define

$$
\mathcal { R } _ { t } = \mathrm { C C } \left( \{ \mathbf { x } \mid \mathbf { N } _ { t } ^ { \mathrm { n a v } } ( \mathbf { x } ) = 1 , \ \mathbf { N } _ { t } ^ { \mathrm { o c c } } ( \mathbf { x } ) = 0 , \ \mathrm { d i s t } ( \mathbf { x } , \boldsymbol { \Omega } _ { t } ^ { \mathrm { o c c } } ) > d _ { \mathrm { s a f e } } \} , \mathbf { p } _ { t } \right) ,\tag{9}
$$

where $\mathrm { C C } ( \cdot , { \bf p } _ { t } )$ returns the connected component containing the current agent position, and $d _ { \mathrm { s a f e } }$ specifies the obstacleclearance margin.

Within $\mathcal { R } _ { t }$ , the semantic-value and semantic-feature layers are grouped into spatially coherent regions using a semanticsaugmented SLIC clustering [55]. Each valid region yields a representative waypoint near its geometric center. Regions that are too small, unsafe, unreachable, or strongly associated with a previously detected navigation loop are discarded.

This region-level representation is more robust than selecting individual high-value cells because semantic value maps may contain local peaks caused by viewpoint changes or reconstruction noise. Spatial aggregation provides more stable long-range targets while preserving fine-grained instruction relevance.

## 3.8.2 Semantic Waypoint Selection

$\mathbf { V } _ { t } ^ { \mathrm { m a p } }$ denote the map-level semantic value layer. Let $\mathcal { A } _ { t } ^ { \mathrm { w p } }$ denote the remaining candidate waypoints. The semantic score of each candidate w is computed by averaging $\mathbf { V } _ { t } ^ { \mathrm { f n a p } }$ over its associated region $\mathcal { U } ( \mathbf { w } )$ . The highest-scoring candidate is

$$
J _ { t } ( \mathbf { w } ) = \frac { 1 } { | \mathcal { U } ( \mathbf { w } ) | } \sum _ { \mathbf { x } \in \mathcal { U } ( \mathbf { w } ) } \mathbf { V } _ { t } ^ { \mathrm { m a p } } ( \mathbf { x } ) , \qquad \widehat { \mathbf { w } } _ { t } = \arg \operatorname* { m a x } _ { \mathbf { w } \in \mathcal { A } _ { t } ^ { \mathrm { w p } } } J _ { t } ( \mathbf { w } ) .\tag{10}
$$

Replacing the current waypoint whenever a new candidate receives a slightly higher score can lead to oscillatory behavior. We therefore apply waypoint hysteresis: the previous target is retained as long as it remains safely reachable, ha not yet been reached, and its updated semantic score remains close to that of the best newly generated candidate. A new target is selected only when the previous waypoint becomes invalid, is reached, loses substantial semantic relevance, or is associated with a navigation failure.

The graph also stores waypoint history in the world frame. When the agent revisits a previously explored neighborhood, candidates that would repeat an earlier unsuccessful decision are suppressed. If loop suppression removes all candidates, the planner falls back to the highest-valued safe candidate available before history-based filtering rather than terminating navigation.

During the final navigation stage, a confidently detected target can directly refine the stopping waypoint. Its instance mask is combined with valid reconstructed depth to estimate the 3D target position, whose ground-plane projection is matched to the nearest reachable map cell. This target-specific refinement is applied only when suficient geometric and semantic evidence is available.

## 3.8.3 FMM Planning and Action Execution

The selected high-level waypoint is projected onto the current local map and supplied to FMM, which computes a geodesic distance field over $\mathcal { R } _ { t } .$ . A short-term goal is then selected within the reachable local region by following the descending direction of this field for a fixed planning horizon.

The low-level controller compares the direction to the short-term goal with the current agent heading. If the angular diference is within the controller tolerance, the agent executes α<sub>F</sub>; otherwise, it executes α<sub>L</sub> or α<sub>R</sub> according to the sign of the angular error. Reaching an intermediate waypoint triggers waypoint regeneration rather than a transition to the next sub-instruction.

Stopping is handled separately from waypoint arrival. The stop action is issued only when the final sub-instruction is complete and the agent has reached the required neighborhood of the final target. The navigation episode terminates when the stop action is executed or when the predefined action budget is exhausted.

## 4 Experiments

## 4.1 Experimental Setup

We conduct all simulation experiments in Habitat-Sim [3] using scenes from the Matterport3D (MP3D) dataset [56]. All experiments run on a system equipped with a single NVIDIA RTX 5090 GPU with 32 GB of memory, an Intel Xeon Platinum 8470Q CPU, and 90 GB of RAM, running Ubuntu 22.04. We follow the standard R2R-CE evaluation protocol [1, 2] and evaluate on the val-unseen split to assess the agent’s ability to generalize to unseen indoor environments.

We use a fixed evaluation subset constructed from the R2R-CE val-unseen split. Specifically, we sample 50 navigation episodes from each of the 11 unseen scenes, resulting in 550 episodes in total. The same subset is used throughout the experiments, including the main comparisons, ablation studies, and analyses of navigation eficiency, depth estimation and 3D reconstruction, and pose estimation.

For the main comparisons, we reproduce each method on this subset using its publicly released configuration. All methods are evaluated in the same Habitat-Sim environment with identical task definitions, action spaces, and evaluation metrics. Using the same episodes and evaluation setup across methods ensures a consistent comparison while keeping the computational cost of repeated evaluations manageable.

We do not use data from the R2R-CE training split to train the navigation policy or construct task-specific data for finetuning. The language model and visual foundation models in our system rely on publicly available pretrained weights and are used only for instruction understanding and scene-level semantic reasoning. The agent receives only first-person monocular RGB images as visual input and does not use auxiliary information from Habitat, such as ground-truth depth, semantic segmentation, ground-truth poses, or global maps. Instead, the system estimates the depth, camera poses, and map representations required for navigation online.

The agent operates in a low-level discrete action space consisting of FORWARD, TURNLEFT, TURNRIGHT, and STOP. A FORWARD action moves the agent by 0.25 m, while each TURNLEFT or TURNRIGHT action rotates it by 30◦. Before navigation begins, the agent first obtains a panoramic observation, either using a panoramic camera or by rotating a monocular camera through a full 360◦. The resulting panoramic observations are then used for metric-scale estimation and initial map construction. This initialization stage is treated as a pre-navigation sensing procedure and is excluded from the navigation action budget and evaluation trajectory. The navigation episode starts from the agent state reached after this initialization stage. From this post-initialization state, all methods are evaluated on the same 550-episode subset using an identical budget of 150 low-level navigation actions per episode.

We evaluate navigation performance using four standard R2R-CE metrics: Success Rate (SR), Oracle Success Rate (OSR), Success weighted by Path Length (SPL), and Navigation Error (NE). We use a success threshold of 3.0 m. SR measures the proportion of episodes in which the agent stops within 3.0 m of the goal. OSR measures the proportion of episodes whose trajectories reach at least one position within 3.0 m of the goal, regardless of where the agent ultimately stops. SPL combines success with the ratio of the shortest-path length to the executed trajectory length, capturing navigation eficiency. NE is the average distance from the agent’s final position to the goal. Higher SR, OSR, and SPL indicate better performance, while lower NE indicates more accurate final stopping. All metrics are computed from the navigation trajectory after the initial scale calibration stage; the trajectory generated during calibration is excluded.

## 4.2 Experimental Results

Table 1 compares LG-VLN with four representative zero-shot VLN methods, including NavGPT, Open-Nav, Instruct-Nav, and CA-Nav. NavGPT is evaluated on a preconstructed discrete navigation graph, whereas Open-Nav, InstructNav, and CA-Nav are evaluated in continuous environments. Since these methods difer in their sensor configurations, visual inputs, navigation settings, and action spaces, we also report their key system settings to clarify the conditions under which the results are obtained.

NavGPT [4] converts visual observations at candidate nodes into natural-language descriptions, representing both scene information and candidate-node attributes as text. It performs zero-shot navigation on a preconstructed discrete naviga tion graph (Nav-graph). Open-Nav [7] uses panoramic RGB-D observations and a waypoint prediction module, with an open-source language model performing step-by-step navigation reasoning. InstructNav [6] performs zero-shot path planning using multi-source value maps; in its egocentric setting, it relies on RGB-D observations and groundtruth poses. CA-Nav [8] takes egocentric RGB-D images and odometry as input and generates navigation waypoints through constraint-aware sub-instruction management and value-map updates. In contrast, LG-VLN uses only egocentric monocular RGB images and employs VGGT to estimate depth, camera poses, and the spatial map online during navigation.

The Eficient LLM Usage column indicates whether a method avoids querying the language model at every low-level navigation step. The Task-Specific Training Module column indicates whether a method relies on waypoint predictors, navigation policies, or other task-specific networks trained using navigation data, environment topology, or related task data. General-purpose vision and language foundation models pretrained on generic data and kept frozen during navigation are not counted as task-specific training modules.

Table 1. Main zero-shot VLN-CE results on a fixed 550-episode subset of the R2R-CE val-unseen split; NavGPT performs navigation over a preconstructed discrete navigation graph, whereas the other methods operate in the continuous Habitat-Sim environment.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Sensor Configuration</td><td rowspan="2">Efficient LLM Usage</td><td rowspan="2">Egocentric Obs.</td><td rowspan="2">Task-Specific Training Module</td><td rowspan="2">Online Geometry SR↑ SPL↑ OSR↑ NE↓ Estimation</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td>NavGPT† [4]</td><td>Panoramic RGB</td><td>×</td><td>×</td><td>X</td><td>×</td><td>31.8 26.8</td><td></td><td>40.9 9.75</td></tr><tr><td>Open-Nav [7]</td><td>Panoramic RGB-D</td><td>X</td><td>X</td><td>√</td><td>X</td><td>26.4 23.2</td><td></td><td>31.6 7.99</td></tr><tr><td>InstructNav [6]</td><td>RGB-D + GT Pose</td><td>X</td><td>√</td><td>X</td><td>X</td><td>15.1</td><td>6.7</td><td>25.1 7.84</td></tr><tr><td>CA-Nav [8]</td><td>RGB-D + Odometry</td><td>√</td><td>√</td><td>X</td><td>X</td><td>22.4</td><td>9.3</td><td>43.7 8.76</td></tr><tr><td>LG-VLN</td><td>Monocular RGB</td><td>√</td><td>√</td><td>X</td><td>√</td><td>21.312.1</td><td></td><td>29.5 7.85</td></tr></table>

NavGPT is evaluated on a preconstructed discrete navigation graph rather than in the continuous environment. Its reported metrics are therefore provided for reference only and are not directly comparable with those of the continuous-environment methods. LG-VLN additionally assumes a known camera mounting height $h _ { \mathrm { { c a m } } }$ for metric-scale initialization.

NavGPT and Open-Nav achieve relatively strong SR, reaching 31.8% and 26.4%, respectively, along with competitive SPL. NavGPT, however, does not navigate from online sensor observations in a continuous environment. Instead, it plans over a preconstructed discrete navigation graph (Nav-graph), whereas all other methods operate directly in the continuous Habitat-Sim environment. Open-Nav combines panoramic RGB-D observations with a waypoint predictor trained on VLN datasets, benefiting from both panoramic visual information and task-specific navigation priors.

InstructNav, CA-Nav, and LG-VLN all use egocentric observations, which more closely reflect practical robotic nav igation settings. CA-Nav and LG-VLN also avoid querying the LLM at every low-level navigation step, reducing the latency caused by repeated language-model inference. Under this setting, LG-VLN achieves an SR of 21.3% and an SPL of 12.1% using only monocular RGB input. Its SR is close to that of CA-Nav (22.4%), which uses RGB-D observations and odometry, and is notably higher than that of InstructNav (15.1%), which uses ground-truth poses.

Among the methods that operate with egocentric observations in the continuous environment, CA-Nav achieves an OSR of 43.7%, compared with 29.5% for LG-VLN, a gap of 14.2 percentage points. LG-VLN, however, achieves a higher SPL of 12.1%, compared with 9.3% for CA-Nav. This diference between OSR and SPL is closely related to the methods’ exploration and stopping behavior.

OSR only requires the trajectory to reach a position within 3.0 m of the goal at any point during navigation. It is therefore sensitive to the extent of exploration and whether the trajectory enters the goal vicinity. CA-Nav uses RGB D observations and real-time odometry, providing stable metric information for occupancy-grid distance estimation and supporting more aggressive frontier selection based on reachability. LG-VLN, in contrast, estimates depth and camera poses online from monocular RGB observations. Its initial metric scale $s _ { 0 }$ remains fixed after calibration, while bounded inter-frame Sim(3) registration compensates for residual frame-to-frame scale fluctuations. As a result, LG-VLN adopts a more conservative exploration radius and frontier-selection policy. This reduces the likelihood that its trajectory enters the goal vicinity during exploration and may contribute to the 14.2 percentage-point OSR gap between LG-VLN and CA-Nav.

Once LG-VLN determines that the goal is nearby and enters its recovery or termination procedure, locally consistent relative depth estimates support waypoint selection without substantially increasing the traveled distance. CA-Nav explores more aggressively, increasing the chance of entering the goal vicinity and thereby improving OSR, but this behavior can also produce longer trajectories. The resulting increase in path length lowers SPL and may explain why CA-Nav achieves a lower SPL despite its higher OSR.

The methods also difer in how they obtain geometric information. InstructNav uses RGB-D observations and groundtruth poses, whereas CA-Nav relies on RGB-D observations and real-time odometry. LG-VLN instead estimates depth, camera poses, and the spatial map online. It therefore does not require a depth sensor, odometry, a prior map, or task-specific modules trained on VLN data. These results suggest that online geometric perception and mapping can provide suficiently reliable spatial information for continuous navigation while reducing dependence on additional sensors. More broadly, they indicate that competitive VLN performance can be achieved using only monocular RGB observations.

## 4.3 Ablation Study

We conduct ablation experiments on the R2R-CE val-unseen evaluation subset to assess the contribution of each module to overall performance. All ablations use the same parameter settings, with only the module under study removed or replaced.

## 4.3.1 Analysis of Shared Visual Feature Enhancement

The semantic value map combines two complementary visual signals: visual reference similarity from CleanDIFT and image–text relevance from BLIP-2. CleanDIFT captures fine-grained correspondences between regions in the current observation and the target visual reference, while BLIP-2 measures the semantic relevance of the current observation to the active sub-instruction. The full model combines these signals to estimate the semantic values of candidate regions.

We vary the fusion weights $\beta _ { a }$ and $\beta _ { b }$ to assess the contribution of each signal. Setting $\beta _ { a } = 0$ removes the CleanDIFT term from the semantic value map, leaving only the BLIP-2-based image–text relevance. Conversely, setting $\beta _ { b } = 0$ retains only the CleanDIFT-based visual similarity. This ablation isolates the roles of the two signals in constructing the semantic value map; shared visual features used by other components of the framework remain unchanged.

Table 2. Ablation on the two semantic sources of the value map (550-episode subset of R2R-CE val-unseen).
<table><tr><td>Mode</td><td> $\beta _ { a } \wedge \beta _ { b }$ </td><td>SR↑</td><td>SPL↑</td><td>OSR↑</td><td>NE↓</td></tr><tr><td>Full LG-VLN</td><td>1.0 / 1.0</td><td>21.3</td><td>12.1</td><td>29.5</td><td>7.85</td></tr><tr><td>w/o CleanDIFT value</td><td>0.0 / 1.0</td><td>3.0</td><td>2.1</td><td>7.8</td><td>8.70</td></tr><tr><td>w/o BLIP-2 value</td><td>1.0 / 0.0</td><td>12.0</td><td>9.3</td><td>16.2</td><td>8.92</td></tr></table>

As shown in Table 2, the full LG-VLN model achieves an SR of 21.3%, an SPL of 12.1%, and an OSR of 29.5% on the evaluation subset. Removing either semantic signal consistently reduces navigation performance, supporting their complementary roles in semantic value estimation. Without BLIP-2, SR decreases to 12.0% and OSR to 16.2%, indicating that image–text relevance provides useful instruction-conditioned semantic information beyond CleanDIFTbased visual correspondence.

The performance degradation is larger when CleanDIFT is removed, with SR dropping to 3.0%, SPL to 2.1%, and OSR to 7.8%. This suggests that fine-grained visual correspondence plays an important role in distinguishing spatially relevant regions in the proposed value estimator, while BLIP-2 alone is insuficient to provide reliable navigation guidance under the current value-map construction and waypoint-selection pipeline. This result should therefore be interpreted within the proposed fusion framework rather than as a general comparison between the standalone capabilities of CleanDIFT and BLIP-2.

NE varies less than SR and OSR across the three configurations, ranging from 7.85 to 8.92 m. This is consistent with the diferent definitions of these metrics. NE is a continuous measure of the final distance to the goal, whereas SR depends on whether the final position lies within the success threshold and OSR on whether the trajectory enters the success region at any point. Overall, the results support the complementary use of CleanDIFT for fine-grained visual correspondence and BLIP-2 for instruction-conditioned semantic relevance. This ablation evaluates only their contributions to semantic value-map construction; the efect of shared CleanDIFT features on the geometric front-end is analyzed separately.

## 4.3.2 Analysis of Online Geometric Estimation

To evaluate how online geometric estimation afects VLN performance, we conduct ablation experiments on the same evaluation subset under three geometric settings. The Oracle setting uses ground-truth depth and poses provided by Habitat. The Hybrid setting estimates depth online with VGGT while using ground-truth poses from Habitat. In our setting, both depth and poses are estimated online with VGGT and used for online 3D reconstruction.

All three settings use the same navigation parameters, semantic modules, and planning modules, with only the geometric inputs changed. The resulting performance diferences therefore primarily reflect the efects of depth and pose estimation quality on navigation.

Table 3. Efect of Online Geometric Estimation on Navigation Performance
<table><tr><td>Mode</td><td>Depth Source</td><td>Pose Source</td><td>SR↑</td><td>SPL↑</td><td>NE↓</td></tr><tr><td>Oracle</td><td>GT Depth</td><td>GT Pose</td><td>24.9</td><td>17.7</td><td>6.80</td></tr><tr><td>Hybrid</td><td>Pred.</td><td>GT Pose</td><td>24.2</td><td>15.2</td><td>7.04</td></tr><tr><td>Ours</td><td>Pred.</td><td>Pred.</td><td>21.3</td><td>12.1</td><td>7.85</td></tr></table>

Table 3 reports the results of the three configurations evaluated on the same set of episodes. Navigation performance gradually declines as ground-truth geometric information is replaced by online estimates. From Oracle to Hybrid and Ours, SR decreases from 24.9% to 24.2% and 21.3%, respectively, while SPL decreases from 17.7% to 15.2% and 12.1%. NE increases from 6.80 m in the Oracle setting to 7.85 m in our setting. These results indicate that errors in online depth and pose estimation can afect both navigation success and path eficiency.

The gap between Hybrid and Oracle mainly reflects the efect of VGGT depth estimation errors. Replacing ground-truth depth with VGGT estimates reduces SR and SPL by 0.7 and 2.5 percentage points, respectively, while increasing NE by 0.24 m. The reduction in SR is relatively small compared with the performance gaps among the complete navigation systems reported in Table 1. This suggests that, given continuous-view observations and initial scale calibration, VGGT depth estimates preserve suficiently accurate relative depth and local geometric structure to support state-machine constraints, value-map accumulation, and waypoint selection.

The gap between Ours and Hybrid mainly reflects the efect of online pose estimation errors. Replacing ground-truth poses with online estimates further reduces SR and SPL by 2.9 and 3.1 percentage points, respectively, while increasing NE by 0.81 m. This increase in NE is substantially larger than that caused by replacing ground-truth depth (0.81 m vs. 0.24 m). The comparison therefore suggests that pose estimation errors have a greater efect on the final navigation error than depth estimation errors under the current setup.

## 4.4 Navigation Eficiency Analysis

LG-VLN includes several computationally intensive components, particularly geometric reconstruction and dense semantic feature extraction. We therefore measure the runtime of the main perception and planning modules to quantify the computational cost of the system. All measurements are collected on the same hardware using identical input settings.

Table 4. Runtime per call for the main perception and planning modules.
<table><tr><td>Module</td><td>Time/Call ↓</td><td>Description</td></tr><tr><td>VGGT</td><td>178 ms</td><td>Depth, pose, and point cloud estimation, executed for every frame</td></tr><tr><td>CleanDIFT</td><td>179 ms</td><td>Dense semantic feature extraction for inter-frame registration</td></tr><tr><td>Grounding DINO + SAM2</td><td>602 ms</td><td>Open-vocabulary detection and instance mask generation, executed when perception is triggered</td></tr><tr><td>BLIP-2 ITM</td><td>52–72 ms</td><td>Image-text matching, executed when perception is triggered</td></tr><tr><td>GTM + FMM</td><td>86 ms</td><td>Map fusion and local planning</td></tr></table>

As shown in Table 4, Grounding DINO + SAM2 is the most computationally expensive semantic perception component, requiring approximately 602 ms per call. VGGT and CleanDIFT require 178 ms and 179 ms per call, respectively, resulting in similar computational costs. BLIP-2 ITM requires 52–72 ms for image–text matching for each short-text candidate, while GTM + FMM takes approximately 86 ms for map fusion and local planning.

LG-VLN does not execute every semantic model at each navigation step. In the current implementation, VGGT and CleanDIFT are run on every frame to support geometric reconstruction and inter-frame feature association. Grounding DINO + SAM2 and BLIP-2 ITM are invoked only when a semantic perception update is needed. Detection, segmentation, and image–text matching are repeated when the target category changes, the active sub-instruction switches, or the current observation satisfies the perception-update criterion. For consecutive observations within the same subinstruction, the system reuses existing perception results when no update is required, avoiding redundant computation.

## 4.5 Analysis of the Navigation Process

To examine how LG-VLN perceives the environment, estimates navigation values, and selects waypoints, we visualize intermediate results from two representative episodes at decision step 20, as shown in Fig. 2. The two examples illustrate diferent stages of navigation. In the first, the target landmark has not yet entered the agent’s view, whereas in the second, it is already visible. Comparing these cases shows how the navigation behavior shifts from exploration toward targetdirected movement.

![](images/9769efc37e97340f4b960168863ce4dfd2343fc82c2d8e13fa5b49686a5450b9.jpg)  
Fig. 2. Visualization of two representative navigation episodes at decision step 20. The active sub-instructions in the top and bottom rows are “exit through double doors” and “walk toward the big wall of windows,” respectively. From left to right, each row shows the current RGB observation and active sub-instruction, the pixel-wise CleanDIFT score map, the fused semantic value map, and the top-down navigation state derived from the online reconstruction map. The CleanDIFT and semantic value maps use the JET colormap, with warmer colors indicating higher responses or values. In the top-down view, gray regions denote traversable space, green boundaries denote walls or obstacles, white contours show SLIC superpixel boundaries, and colored dots mark the centroids of the five highest-valued superpixels. The historical trajectory changes gradually from blue at the starting position to green at the current position.

In Task 1, the active sub-instruction is “exit through double doors,” but the target doors are not yet visible in the current RGB observation. Because the target lies outside the field of view, the fused semantic value map does not contain a clear local high-value region. The system instead relies on accumulated map information and the surrounding geometry to continue frontier exploration. CleanDIFT still produces relatively strong responses around visually distinctive structures, such as carpets, furniture boundaries, and door frames, indicating that it captures local appearance patterns even when they are not directly related to the current target. In the top-down view, high-value waypoint candidates are mainly located near traversable frontiers of the reconstructed map. At this stage, the agent remains in exploration mode, and its planned motion follows the geometry of the observed environment.

In Task 2, the active sub-instruction is “walk toward the big wall of windows,” and the target landmark is already visible. With stronger correspondence between the instruction and the current observation, the fused semantic value map shows a clear high-value region around the windows, while regions farther from the target receive lower values. This distribution indicates that the value map can distinguish target-relevant regions once suficient visual evidence is available. CleanDIFT also responds strongly to the window frames, boundaries, and nearby floor regions, providing local visual cues for semantic value fusion. In the top-down view, the highest-value waypoint candidates are concentrated in the direction of the windows and distributed across traversable space. The agent’s trajectory follows the same direction, indicating that the fused visual-semantic representation provides useful guidance for local planning.

These two examples illustrate the complementary roles of CleanDIFT and the semantic value map. CleanDIFT provides local visual similarity cues from textures, boundaries, and other fine-grained structures. The semantic value map combines these cues with the language instruction, accumulated map information, and the current navigation state to produce a spatial value distribution for planning. When the target is not visible, accumulated map information supports continued exploration. Once the target enters the field of view, visual evidence and language semantics reinforce the corresponding regions, concentrating high values near the target and directing waypoint selection toward it. After projection onto the map and SLIC superpixel clustering, the selected high-value candidates remain within traversable regions and align with the agent’s direction of motion. Together, these examples suggest that the proposed value fusion and waypoint generation procedure balances target relevance with navigational feasibility.

## 4.6 Depth Estimation and 3D Reconstruction Results

## 4.6.1 Depth Estimation Results

We use the metric depth provided by Habitat as ground truth to evaluate the dense depth estimates. Let $\hat { d } _ { t }$ denote the raw depth prediction of VGGT at frame $t , \tilde { d } _ { t }$ the corresponding metric depth after applying the global scale factor $s _ { 0 }$ estimated during navigation initialization, and $d _ { t } ^ { g t }$ the Habitat ground-truth depth map in meters. The metric depth used during navigation is given by

$$
\tilde { d } _ { t } = s _ { 0 } \hat { d } _ { t } .\tag{11}
$$

Ground-truth depth is not available during navigation for frame-wise scale or ofset correction. We therefore evaluate the depth estimates using the same fixed global scale $s _ { 0 }$ used at deployment, comparing $\tilde { d } _ { t }$ with $d _ { t } ^ { g t }$ . The resulting depth estimation errors are reported in Table 5.

To distinguish errors from global scale estimation from those associated with local relative depth structure, Table 6 also reports AbsRel under two scale-alignment settings. $\mathrm { \bf A b s R e l } _ { s _ { 0 } }$ applies the global scale factor $s _ { 0 }$ estimated during initialization to all frames, matching the deployment setting used for navigation. In contrast, $\mathbf { A b s R e l } _ { \mathrm { m e d } }$ aligns each frame independently before computing the error. Because this frame-wise alignment requires ground-truth depth, it i used only for post-hoc analysis.

Table 6 further reports the same metrics for pixels satisfying $d _ { \mathrm { g t } } < 5$ m, providing a separate measure of depth estimation accuracy at the short ranges most relevant to navigation.

Table 5. Depth estimation error over all valid observation frames. Metrics are computed per frame and then aggregated across frames.
<table><tr><td>Statistics</td><td>AbsRel↓</td><td>RMSE (m)↓</td><td>MAE (m)↓</td><td>SILog↓</td><td>Pearson↑</td></tr><tr><td>mean</td><td>0.5874</td><td>0.6339</td><td>0.5371</td><td>0.1486</td><td>0.9107</td></tr><tr><td>median</td><td>0.2643</td><td>0.4863</td><td>0.3941</td><td>0.1128</td><td>0.9723</td></tr><tr><td>std</td><td>1.0531</td><td>0.5314</td><td>0.4949</td><td>0.1176</td><td>0.1851</td></tr><tr><td>25%</td><td>0.1322</td><td>0.2696</td><td>0.2078</td><td>0.0691</td><td>0.9260</td></tr><tr><td>75%</td><td>0.4752</td><td>0.8248</td><td>0.6974</td><td>0.1890</td><td>0.9869</td></tr></table>

Table 6. Depth estimation errors under diferent scale alignment settings.
<table><tr><td>Statistics</td><td> $\mathbf { A b s R e l } _ { s _ { 0 } }$ </td><td> $\mathbf { A b s R e l } _ { \mathrm { m e d } }$ </td><td> $\mathbf { A b s R e l } _ { s _ { 0 } } ^ { < 5 \mathrm { m } }$ </td><td> $\mathbf { A b s R e l } _ { \mathrm { m e d } } ^ { < 5 \mathrm { m } }$ </td></tr><tr><td>mean</td><td>0.5874</td><td>0.0493</td><td>0.5875</td><td>0.0484</td></tr><tr><td>median</td><td>0.2643</td><td>0.0319</td><td>0.2629</td><td>0.0299</td></tr><tr><td>25%</td><td>0.1322</td><td>0.0186</td><td>0.1302</td><td>0.0174</td></tr><tr><td>75%</td><td>0.4752</td><td>0.0586</td><td>0.4747</td><td>0.0558</td></tr></table>

All metrics are computed over valid ground-truth pixels in each frame and then aggregated across frames. Table 5 therefore reflects frame-to-frame variability rather than a global pixel-level distribution. The fixed-scale AbsRel is strongly right-skewed, with a mean of 0.5874 and a median of only 0.2643, together with a large standard deviation of 1.0531. This pattern suggests that the mean is driven by a relatively small number of frames with severe scale errors.

Table 6 further separates scale errors from errors in relative depth structure. After independent median alignment for each frame, AbsRel decreases from 0.5874 to 0.0493 on average and from 0.2643 to 0.0319 at the median. A similar reduction is observed for pixels with ${ d _ { \mathrm { { g t } } } < 5 , \mathrm { { m } } }$ . The median SILog of0.1128 and Pearson correlation of0.9723 provide further evidence that most of the error comes from frame-to-frame scale variation rather than inaccurate relative depth structure.

The same pattern appears in Table 3. Replacing ground-truth depth with VGGT depth while retaining ground-truth poses results in only a small drop in SR. Approximately uniform changes in scale largely preserve bearings, depth ordering, and traversability structure, making navigation less sensitive to scale errors than metric reconstruction. Residual scale variation can still afect fixed metric thresholds, however, and the use of a single global scale remains a limitation.

For qualitative evaluation, we select two representative R2R-CE navigation episodes and use the first valid observation frame from each episode. In Fig. 3, the first three columns show the Habitat ground-truth depth, the scale-corrected depth estimate, and the pixel-wise absolute error map. The fourth column plots the estimated depth against the groundtruth depth, together with the $y = x$ reference line and the tolerance region corresponding to $\bar { \delta } < \bar { 1 } . 2 5$ . These two frames are near-range observations selected for visualization and are not representative of the full-split statistics reported in Table 5.

![](images/fa0a5c8c61a02f67562ab9ab483ea030ff4930520ca104cb8b759a1de98fb910.jpg)  
Fig. 3. Qualitative depth estimation results on representative R2R-CE frames. Each row corresponds to one observation frame and shows, from left to right: (a) the Habitat ground-truth depth map; (b) the estimated depth map after global metric-scale calibration; (c) the pixel-wise absolute error map; and (d) a scatter plot of the predicted depth against the ground-truth depth, including the dashed $y = x$ reference line and the tolerance band corresponding to $\delta < 1 . 2 5$

Fig. 3 shows that the estimated depth in both examples is strongly correlated with the ground-truth depth. However, the scatter plots remain systematically ofset from the $y = x$ reference line, indicating that the fixed global scale does not fully match the efective scale of each frame. Large absolute errors occur mainly near object boundaries and depth discontinuities, where the predicted depth tends to be spatially smoother than the ground truth.

Across all valid frames, the mean and median Pearson correlation coeficients are 0.9107 and 0.9723. The high correlation indicates that the predicted depth generally preserves relative depth variation and scene structure even when its absolute metric scale is inaccurate.

## 4.6.2 3D Reconstruction Results

We qualitatively evaluate the 3D reconstructions obtained with VGGT on two representative navigation episodes with diferent spatial layouts, as shown in Fig. 4. In both cases, reconstruction uses only consecutive monocular RGB frames, without additional sensing modalities such as depth, IMU, or odometry. The accumulated point clouds are visualized from the X–Z top-down view and the X–Y side view to examine the scene layout, reconstructed geometry, and consistency across frames.

![](images/7bc5330841cddade5d3a01514c334d9bd24ce6bdc64a931c8d2a2b1411460b26.jpg)  
Fig. 4. 3D reconstruction results for two representative indoor scenes with diferent spatial layouts. The X–Z panels show top-down views of the accumulated point clouds, highlighting the planar scene layout and spatial consistency. The X–Y panels show side views, illustrating scene height, depth variation, and local geometric structure.

The example on the left shows an indoor room with a relatively regular layout. In the X–Y side view, the main structures, including the walls, floor, and large pieces of furniture, are largely recovered, with clear separation across diferent depth ranges. The X–Z top-down view preserves the room boundaries and the relative positions of the furniture, without obvious large-scale ghosting or structural misalignment.

The example on the right shows a more complex scene containing nearby furniture, a long corridor, and arched windows. The reconstruction preserves the main geometric contours of both the foreground furniture and the distant corridor while maintaining a largely continuous spatial layout. Some scattered points remain near wall boundaries and thin structures, but they do not introduce substantial geometric discontinuities.

Inter-frame consistency is maintained through Sim(3) alignment between adjacent local reconstructions and pointcloud filtering based on geometric confidence and CleanDIFT semantic consistency. The alignment compensates for diferences in rotation, translation, and scale across local reconstructions, while the filtering step suppresses points arising from low-confidence predictions and incorrect matches. Low-texture regions and depth discontinuities can still produce locally sparse reconstructions or isolated outliers.

Overall, these examples show that VGGT can recover the main geometric structures of indoor scenes from consecutive monocular RGB images and integrate them into relatively continuous point clouds in a unified coordinate system. The reconstructed geometry provides a practical spatial representation for subsequent value-map construction and local path planning.

## 4.7 Pose Estimation Accuracy Evaluation

The proposed system relies on online camera pose estimates for map updates and local path planning during navigation. In geometrically ambiguous scenes, incorrect correspondences can reduce pose estimation accuracy. To mitigate this problem, we use CleanDIFT features to filter correspondences based on semantic consistency. We quantitatively eval uate the resulting pose estimates using Absolute Trajectory Error (ATE) and the translational component of Relative Trajectory Error (RTE).

We evaluate the pose estimation pipeline under two configurations. In the w/o CleanDIFT setting, inter-frame Sim(3) registration relies only on geometric RANSAC. In the w/ CleanDIFT setting, CleanDIFT semantic consistency is used to filter the correspondences before geometric RANSAC. The Sim(3) transformation is then re-estimated from the remaining semantically consistent matches.

For evaluation, the estimated pose at each frame is compared with the corresponding ground-truth pose provided by Habitat. After Sim(3) alignment, ATE is computed as the root mean squared error (RMSE) between the estimated and ground-truth trajectories. RTE is computed from the relative translation error between consecutive frames and is also reported as RMSE. For each metric, we average the RMSE across all navigation episodes to obtain the overall result.

Table 7. Pose estimation accuracy on the 550-episode evaluation subset.
<table><tr><td>Mode</td><td>ATE RMSE (m)</td><td>RTE RMSE (m)</td></tr><tr><td>w/o CleanDIFT</td><td>0.322</td><td>0.150</td></tr><tr><td>w/ CleanDIFT</td><td>0.318</td><td>0.092</td></tr></table>

Table 7 shows that the CleanDIFT semantic consistency constraint substantially improves relative pose estimation. The RTE RMSE decreases from 0.150 m to 0.092 m, corresponding to a 38.7% reduction, whereas the ATE RMSE is essentially unchanged (0.322 m versus 0.318 m). Since RTE more directly reflects local frame-to-frame motion accuracy, the larger improvement suggests that CleanDIFT mainly enhances inter-frame registration rather than global trajectory consistency.

In indoor scenes with repetitive structures or weak texture, geometric matching may produce ambiguous correspondences that are not fully rejected by RANSAC. CleanDIFT introduces semantic consistency as an additional criterion for correspondence filtering, reducing geometrically plausible but semantically incorrect matches. This improves the reliability of correspondences used for relative pose estimation and consequently reduces local translation errors. The limited improvement in ATE is expected because CleanDIFT operates primarily on local pairwise alignment rather than explicitly optimizing the complete trajectory.

## 5 Conclusion

This work addresses zero-shot VLN-CE using only a monocular RGB camera and presents LG-VLN, a framework that shares dense visual features across geometric estimation and semantic navigation. A feed-forward 3D reconstruction network estimates depth, camera poses, and dense point clouds online, which are used to update geometric and semantic maps in a unified world coordinate system. To connect geometric mapping with semantic navigation, CleanDIFT features are used in both stages. In the geometric front-end, semantic consistency filters inter-frame correspondences to reduce the efect of incorrect matches on pose estimation and map fusion. During navigation, target-conditioned visual reference similarity is combined with local image–text relevance from BLIP-2 to construct a semantic value map for the active sub-instruction.

LG-VLN also maintains an explicit shared state that records task progress, map states, perception results, and action feedback. LangGraph organizes instruction decomposition, constraint evaluation, semantic perception, incremental mapping, waypoint selection, path planning, action execution, and failure recovery into a directed workflow. Conditional routing governs sub-instruction transitions, termination decisions, and recovery procedures, allowing the navigation modules to operate through a common state and control interface.

On a fixed 550-episode subset of the R2R-CE val-unseen split, LG-VLN achieves an SR of 21.3%, an SPL of 12.1%, and an NE of 7.85 m without task-specific training data, ground-truth depth or poses, odometry, or prior maps. Ablation results show that pixel-level visual similarity from CleanDIFT and local image–text relevance from BLIP-2 provide complementary semantic signals. The same dense CleanDIFT features also support geometric correspondence filtering, allowing them to contribute to both geometric estimation and semantic value construction. The results further indicate that online pose estimation errors have a greater overall efect on navigation performance than depth prediction errors, with a particularly clear efect on NE. The diference in SR is relatively small under the current single-run evaluation, so a stronger conclusion would require more episodes or repeated runs. Overall, these results show that zero-shot VLN-CE can be performed from monocular RGB observations alone by combining online geometric estimation, shared visual features, and semantic value fusion.

## References

[1] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid, S. Gould, and A. van den Hengel, “Vision-and Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 3674–3683.

[2] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments,” in Proceedings of the European Conference on Computer Vision (ECCV), 2020, pp. 104–120.

[3] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik, D. Parikh, and D. Batra, “Habitat: A Platform for Embodied AI Research,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 9338–9346.

[4] G. Zhou, Y. Hong, and Q. Wu, “NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7641–7649.

[5] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation,” in Proceedings ofthe IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 42–48.

[6] Y. Long, W. Cai, H. Wang, G. Zhan, and H. Dong, “InstructNav: Zero-Shot System for Generic Instruction Navigation in Unexplored Environments,” in Proceedings ofthe Conference on Robot Learning (CoRL), 2024.

[7] Y. Qiao, W. Lyu, H. Wang, Z. Wang, Z. Li, Y. Zhang, M. Tan, and Q. Wu, “Open-Nav: Exploring Zero-Shot Vision-and Language Navigation in Continuous Environments with Open-Source LLMs,” in Proceedings ofthe IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 6710–6717.

[8] Chen K, An D, Huang Y, et al. Constraint-aware zero-shot vision-language navigation in continuous environments[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[9] An D, Wang H, Wang W, et al. Etpnav: Evolving topological planning for vision-language navigation in continuous environ ments[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024, 47(7): 5130-5145.

[10] J. Chen, B. Lin, X. Liu, L. Ma, X. Liang, and K.-Y. K. Wong, “Afordances-Oriented Planning Using Foundation Models for Continuous Vision-Language Navigation,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, 2025.

[11] X. Song, W. Chen, Y. Liu, V. Chan, G. Li, and L. Lin, “Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[12] M. Wei, C. Wan, X. Yu, T. Wang, Y. Yang, X. Mao, C. Zhu, W. Cai, H. Wang, Y. Chen, X. Liu, and J. Pang, “StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling,” arXiv preprint arXiv:2507.05240, 2025.

[13] Z. Zhan, L. Yu, S. Yu, and G. Tan, “MC-GPT: Empowering Vision-and-Language Navigation with Memory Map and Reasoning Chains,” arXiv preprint arXiv:2405.10620, 2024.

[14] Zeng S, Qi D, Chang X, et al. Janusvln: Decoupling semantics and spatiality with dual implicit memory for vision-language navigation[C]//International Conference on Learning Representations. 2026, 2026: 33001-33026.

[15] L. Luo and Q. Bai, “MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN,” arXiv preprint arXiv:2603.03024, 2026.

[16] Y. Li, C. Li, H. Shi, J. Luo, J. Cai, M. Yang, and T. Qin, “AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness,” arXiv preprint arXiv:2606.10577, 2026.

[17] LangChain, “LangGraph: A Low-Level Orchestration Framework for Building Stateful Agents,” 2024. [Online]. Available: https://github.com/langchain-ai/langgraph. [Accessed: Sep. 11, 2026].

[18] J. Bellver-Soler, S. Ramos-Varela, A. Guragain, R. Córdoba, and L. F. D’Haro, “ORCHESTRA: AI-Driven Microservices Architecture to Create Personalized Experiences,” in Proceedings of the 16th International Workshop on Spoken Dialogue Systems Technology (IWSDS), 2026, pp. 158–167.

[19] J. Chen, B. Lin, R. Xu, Z. Chai, X. Liang, and K.-Y. K. Wong, “Mapgpt: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024, pp. 9796–9810.

[20] G. Zhou, Y. Hong, Z. Wang, X. E. Wang, and Q. Wu, “NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models,” in Proceedings ofthe European Conference on Computer Vision (ECCV), 2024, pp. 260–278.

[21] Y. Long, X. Li, W. Cai, and H. Dong, “Discuss Before Moving: Visual Language Navigation via Multi-Expert Discussions,” in Proceedings ofthe IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 17380–17387.

[22] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and H. Wang, “NaVid: Video-Based VLM Plans the Next Step for Vision-and-Language Navigation,” in Proceedings ofRobotics: Science and Systems (RSS), 2024.

[23] Wang S, Wang Y, Fan Z, et al. Monodream: Monocular vision-language navigation with panoramic dreaming[C]//Proceedings ofthe AAAI Conference on Artificial Intelligence. 2026, 40(12): 10074-10082.

[24] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth Anything V2,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.

[25] L. Piccinelli, Y.-H. Yang, C. Sakaridis, M. Segu, S. Li, L. Van Gool, and F. Yu, “UniDepth: Universal Monocular Metric Depth Estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 10106–10116.

[26] Piccinelli L, Sakaridis C, Yang Y H, et al. Unidepthv2: Universal monocular metric depth estimation made simpler[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[27] R. Wang, S. Xu, C. Dai, J. Xiang, Y. Deng, X. Tong, and J. Yang, “MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[28] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: Geometric 3D Vision Made Easy,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20697–20709.

[29] V. Leroy, Y. Cabon, and J. Revaud, “Grounding Image Matching in 3D with MASt3R,” in Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[30] Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa, “Continuous 3D Perception Model with Persistent State,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[31] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual Geometry Grounded Transformer,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[32] Maggio D, Lim H, Carlone L. Vggt-slam: Dense rgb slam optimized on the sl (4) manifold[J]. Advances in Neural Information Processing Systems, 2026, 38: 129839-129867.

[33] Jiang L, Mao Y, Xu L, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views[J]. ACM Transactions on Graphics (TOG), 2025, 44(6): 1-16.

[34] Q. Tian, X. Tan, J. Gong, Y. Xie, and L. Ma, “UniForward: Unified 3D Scene and Semantic Field Reconstruction via Feed Forward Gaussian Splatting from Only Sparse-View Images,” arXiv preprint arXiv:2506.09378, 2025.

[35] M. Iovino, E. Scukins, J. Styrud, P. Ögren, and C. Smith, “A Survey of Behavior Trees in Robotics and AI,” Robotics and Autonomous Systems, vol. 154, art. no. 104096, 2022.

[36] O. Pettersson, “Execution Monitoring in Robotics: A Survey,” Robotics and Autonomous Systems, vol. 53, no. 2, pp. 73–88, 2005.

[37] R. Wu, S. Kortik, and C. H. Santos, “Automated Behavior Tree Error Recovery Framework for Robotic Systems,” in Proceedings ofthe IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 6898–6904.

[38] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-Pérez, “Integrated Task and Motion Planning,” Annual Review ofControl, Robotics, and Autonomous Systems, vol. 4, pp. 265–293, 2021.

[39] Brohan A, Chebotar Y, Finn C, et al. Do as i can, not as i say: Grounding language in robotic afordances[C]//Conference on robot learning. Pmlr, 2023: 287-318.

[40] Huang W, Xia F, Xiao T, et al. Inner monologue: Embodied reasoning through planning with language models[J]. arXiv preprint arXiv:2207.05608, 2022.

[41] Driess D, Xia F, Sajjadi M S M, et al. Palm-e: An embodied multimodal language model[J]. arXiv preprint arXiv:2303.03378, 2023.

[42] R. Bajcsy, Y. Aloimonos, and J. K. Tsotsos, “Revisiting Active Perception,” Autonomous Robots, vol. 42, no. 2, pp. 177–196, 2018.

[43] S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, J. Wang, C. Zhang, Z. Wang, S. Yau, Z. Lin, L. Zhou, C. Ran, L. Xiao, C. Wu, and J. Schmidhuber, “MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework,” in Proceedings ofthe International Conference on Learning Representations (ICLR), 2024.

[44] Wu Q, Bansal G, Zhang J, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation[J]. arXiv preprint arXiv:2308.08155, 2023.

[45] Zhuge M, Wang W, Kirsch L, et al. Language agents as optimizable graphs[J]. arXiv preprint arXiv:2402.16823, 2024.

[46] S. Zhuang, S. Wang, E. Liang, Y. Cheng, and I. Stoica, “ExoFlow: A Universal Workflow System for Exactly-Once DAGs,” in Proceedings ofthe 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2023, pp. 269–286.

[47] N. Stracke, S. A. Baumann, K. Bauer, F. Fundel, and B. Ommer, “CleanDIFT: Difusion Features without Noise,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 117–127.

[48] L. Tang, M. Jia, Q. Wang, C. P. Phoo, and B. Hariharan, “Emergent Correspondence from Image Difusion,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 1363–1389.

[49] J. A. Sethian, “A Fast Marching Level Set Method for Monotonically Advancing Fronts,” Proceedings ofthe National Academy ofSciences, vol. 93, no. 4, pp. 1591–1595, 1996.

[50] M. A. Fischler and R. C. Bolles, “Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography,” Communications ofthe ACM, vol. 24, no. 6, pp. 381–395, 1981.

[51] Z. He, L. Wang, S. Li, Q. Yan, C. Liu, and Q. Chen, “A Multi-Level Attention Network with Sub-Instructions for Continuous Vision-and-Language Navigation,” Applied Intelligence, vol. 55, no. 7, 2025.

[52] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang, “Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection,” in Proceedings ofthe European Conference on Computer Vision (ECCV), 2024.

[53] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer, “SAM 2: Segment Anything in Images and Videos,” in Proceedings of the International Conference on Learning Representations (ICLR), 2025.

[54] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models,” in Proceedings ofthe 40th International Conference on Machine Learning (ICML), vol. 202, 2023, pp. 19730–19742.

[55] R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua, and S. Süsstrunk, “SLIC Superpixels Compared to State-of-the-Art Super pixel Methods,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 11, pp. 2274–2282, 2012.

[56] Chang A, Dai A, Funkhouser T, et al. Matterport3d: Learning from rgb-d data in indoor environments[J]. arXiv preprint arXiv:1709.06158, 2017.

## A Implementation Details

This appendix provides implementation details that complement the method description in the main paper. We focus on coordinate conventions, event-driven perception scheduling, monocular scale initialization, constraint evaluation, semantic-value construction, map fusion, and the recovery policy. Routine software-level control flow is omitted unless it afects the behavior or reproducibility of the proposed method.

## A.1 Coordinate Systems and Map Projection

We use $\mathcal { T } _ { a } ^ { b }$ to denote the transformation from coordinate frame a to coordinate frame b. Because the world-aligned map is maintained in a similarity-coordinate representation, transformations between the reconstructed camera frames and the world frame are represented in Sim(3). A similarity transformation is written as

$$
\mathcal { T } _ { a } ^ { b } = \left( s _ { a } ^ { b } , \mathbf { R } _ { a } ^ { b } , \mathbf { t } _ { a } ^ { b } \right) \in \mathrm { S i m } ( 3 ) ,\tag{A.1}
$$

and acts on a 3D point according to

$$
{ \mathbf { } } ^ { b } { \mathbf { x } } = s _ { a } ^ { b } { \mathbf { R } } _ { a } ^ { b a } { \mathbf { x } } + { \mathbf { t } } _ { a } ^ { b } .\tag{A.2}
$$

Here, $s _ { a } ^ { b } \ > \ 0$ denotes the scale factor, $\mathbf { R } _ { a } ^ { b } \in \mathrm { S O } ( 3 )$ denotes the rotational component, and ${ \bf t } _ { a } ^ { b } \in \mathbb { R } ^ { 3 }$ denotes the translation expressed in frame b. Standard composition and inversion of similarity transformations are used throughout the system.

At decision step t, the agent state in the world-aligned map is represented by the body-to-world similarity transformation $T _ { B _ { t } } ^ { \mathcal { W } } \in \mathrm { S i m } ( 3 )$ , written in homogeneous form as

$$
\mathcal { T } _ { \mathcal { B } _ { t } } ^ { \mathcal { W } } = \left[ { \begin{array} { c c } { s _ { t } ^ { \mathcal { W } } \mathbf { R } _ { t } ^ { \mathcal { W } } } & { \mathbf { p } _ { t } ^ { \mathcal { W } } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} } \right] \in \mathrm { S i m } ( 3 ) ,\tag{A.3}
$$

where $s _ { t } ^ { \mathcal { W } } > 0$ denotes the scale associated with the current world-map representation, $\mathbf { R } _ { t } ^ { \mathcal { W } } \in \mathrm { S O } ( 3 )$ denotes the orientation of the body frame $B _ { t }$ with respect to the world frame W, and $\mathbf { p } _ { t } ^ { \mathcal { W } } = ( x _ { t } , y _ { t } , z _ { t } ) ^ { \top } \in \mathbb { R } ^ { 3 }$ denotes the origin of the body frame expressed in the world frame.

The scale component $s _ { t } ^ { \mathcal { W } }$ is a property of the similarity-based map representation rather than a physical degree of freedom of the agent. It accounts for residual scale variations between monocular reconstructions while preserving the rotational and translational components required for navigation. The body and camera frames are related through a fixed camera-to-body extrinsic transformation $\dot { T } _ { \mathcal { C } } ^ { B }$ . The extrinsic transformation represents a rigid calibration and therefore belongs to SE(3), which is naturally embedded in Sim(3) with unit scale. The camera-to-world transformation is consequently obtained by

$$
\mathcal { T } _ { \mathcal { C } _ { t } } ^ { \mathcal { W } } = \mathcal { T } _ { \mathcal { B } _ { t } } ^ { \mathcal { W } } \circ \mathcal { T } _ { \mathcal { C } } ^ { \mathcal { B } } , \qquad \mathcal { T } _ { \mathcal { C } _ { t } } ^ { \mathcal { W } } \in \mathrm { S i m } ( 3 ) ,\tag{A.4}
$$

with the inverse world-to-camera transformation given by

$$
\mathcal { T } _ { \mathcal { W } } ^ { \mathcal { C } _ { t } } = \left( \mathcal { T } _ { \mathcal { C } _ { t } } ^ { \mathcal { W } } \right) ^ { - 1 } .\tag{A.5}
$$

Here, $T _ { \mathcal { C } } ^ { B } \in \mathrm { S E } ( 3 )$ is fixed throughout an episode and does not introduce any additional scale variation. For a pixel $\boldsymbol { p } = \left( u , v \right)$ with homogeneous coordinate $\bar { \mathbf p } = [ u , v , 1 ] ^ { \top }$ , the corresponding reconstructed point in the camera frame is obtained from its predicted depth as

$$
\mathbf { x } _ { t } ^ { \mathcal { C } } ( p ) = Z _ { t } ( p ) \mathbf { K } _ { \mathrm { c a m } } ^ { - 1 } \bar { \mathbf { p } } ,\tag{A.6}
$$

where $Z _ { t } ( p )$ denotes the reconstructed depth and $\mathbf { K } _ { \mathrm { c a m } }$ is the camera intrinsic matrix. The corresponding point in the world-aligned map is then obtained through the similarity transformation

$$
\mathbf { x } _ { t } ^ { \mathcal { W } } ( p ) = \mathcal { T } _ { \mathcal { C } _ { t } } ^ { \mathcal { W } } \left( \mathbf { x } _ { t } ^ { \mathcal { C } } ( p ) \right) .\tag{A.7}
$$

Thus, the same world-aligned similarity representation is used for accumulating reconstructed geometry and for projecting semantic observations into the global map. For ground-plane mapping, we use the horizontal world coordinates $( x , z )$ . The horizontal component of the agent position is therefore

$$
\begin{array} { r } { \mathbf { p } _ { t , \mathrm { h } } ^ { \mathcal { W } } = \left( x _ { t } , z _ { t } \right) ^ { \top } . } \end{array}\tag{A.8}
$$

A point on the ground plane is represented by $\mathbf { q } ^ { \mathcal { W } } = ( x , z ) ^ { \top }$ . Because the local map remains aligned with the horizontal axes of the world frame, its coordinates are obtained by translating the world coordinates by the current horizontal agent position:

$$
\mathbf { q } ^ { \mathcal { L } _ { t } } = \mathbf { q } ^ { \mathcal { W } } - \mathbf { p } _ { t , \mathrm { h } } ^ { \mathcal { W } } = \left( x - x _ { t } , z - z _ { t } \right) ^ { \top } .\tag{A.9}
$$

The local-map orientation is therefore fixed to the world frame, while its origin follows the agent. Agent rotation does not rotate the local map. For a map resolution of r meters per cell and local-map center $\mathbf { 0 } = \left( o _ { r } , o _ { c } \right)$ , a horizontal world coordinate is projected to the local grid as

$$
\pi _ { t } ( \mathbf { q } ^ { \mathcal { W } } ) = \left( \mathrm { r o u n d } \left( \frac { z - z _ { t } } { r } \right) + o _ { r } , \mathrm { r o u n d } \left( \frac { x - x _ { t } } { r } \right) + o _ { c } \right) ,\tag{A.10}
$$

with the corresponding inverse mapping from a grid-cell center given by

$$
\pi _ { t } ^ { - 1 } ( i , j ) = ( x _ { t } + ( j - o _ { c } ) r , \ z _ { t } + ( i - o _ { r } ) r ) ^ { \top } .\tag{A.11}
$$

For temporal reprojection, the transformation from the previous camera frame $\mathcal { C } _ { t - 1 }$ to the current camera frame $\mathcal { C } _ { t }$ is obtained from the corresponding world-frame similarity transformations:

$$
\mathcal { T } _ { \mathcal { C } _ { t - 1 } } ^ { \mathcal { C } _ { t } } = \left( \mathcal { T } _ { \mathcal { C } _ { t } } ^ { \mathcal { W } } \right) ^ { - 1 } \circ \mathcal { T } _ { \mathcal { C } _ { t - 1 } } ^ { \mathcal { W } } \in \mathrm { S i m } ( 3 ) .\tag{A.12}
$$

This relative transformation captures the rotation, translation, and residual scale variation between consecutive monocular reconstructions. It is used consistently for inter-frame geometric registration, geometric consistency verification, and reprojection of historical semantic observations into the current camera view.

## A.2 Perception Scheduling and Navigation Failure Detection

Event-driven semantic perception. Geometric reconstruction is updated for every valid RGB observation, whereas open-vocabulary semantic perception is invoked only when it is expected to provide substantially new information. Let $t _ { \mathrm { s e m } }$ denote the time of the most recent semantic update. A new semantic observation is requested at episode initialization, immediately after a sub-instruction transition, after recovery, or whenever the cached semantic observation does not correspond to the active sub-instruction.

Semantic perception is also refreshed when the viewpoint has changed suficiently since $t _ { \mathrm { s e m } } .$ . Specifically, we trigger an update after a translation of at least $\tau _ { p }$ or an accumulated heading change of at least $\tau _ { \psi } .$ . An additional update is requested when the current constraint decision is ambiguous, for example, when an object detection score is close to its acceptance threshold or when a location constraint has not accumulated suficient positive or negative evidence.

A semantic cache is therefore considered valid only if it was generated for the currently active sub-instruction. Thi prevents object categories, visual references, or relevance estimates associated with the previous navigation stage from influencing the current one.

The viewpoint thresholds are chosen to maintain suficient spatial coverage. If $\phi _ { h }$ denotes the horizontal field of view of the camera, we use $\tau _ { \psi } < \phi _ { h }$ . Likewise, if the efective perception range is lower-bounded by $d _ { \mathrm { m i n } }$ , the translational spacing is chosen such tha

$$
\tau _ { p } < d _ { \mathrm { m i n } } \tan \left( \frac { \phi _ { h } } { 2 } \right) .\tag{A.13}
$$

In our experiments, $\tau _ { \psi } = 3 0 ^ { \circ }$ and $\tau _ { p } = 0 . 3 0 \mathrm { m }$ .

Motion consistency. After each action, the expected displacement is compared with the motion estimated from visual registration. Forward actions that repeatedly produce substantially less translation than the commanded step provide evidence that the corresponding local free-space estimate may be incorrect. Cells immediately along the attempted motion direction are therefore assigned additional obstacle evidence during map fusion.

The same signal also serves as an indicator of navigation stagnation. We do not declare a failure from a single inconsistent action because occasional reconstruction errors, collisions, or imperfect correspondence estimates can occur during normal navigation.

Persistent failure detection. The graph monitors four complementary indicators over short temporal windows: inefective forward motion, spatial recurrence, heading oscillation, and lack of waypoint progress. Spatial recurrence is detected when the current pose returns to the neighborhood of a recently visited pose. Heading oscillation captures repeated alternating or large heading changes that do not produce meaningful translational progress. Initial panoramic scans and explicit direction-changing constraints are excluded from this test.

Waypoint progress is evaluated only while the high-level target remains unchanged. Let $d _ { t } ^ { \mathrm { g e o } }$ denote the geodesic distance from the current position to the active waypoint. The agent is considered to make progress if this distance decreases suficiently over the monitoring window or if progress is made toward satisfying the active instruction constraints during the same interval. When the waypoint changes, the progress window is restarted for the new target.

A small stagnation counter aggregates these signals over time. Negative events increase the counter, whereas sustained efective motion and waypoint progress decrease it. Recovery is invoked only when the counter exceeds the stagnation threshold, avoiding unnecessary interruptions caused by isolated failures.

## A.3 Monocular Scale Recovery and Registration

Ground-based metric initialization. VGGT produces geometry with an initially unknown metric scale. We recover this scale from the known camera mounting height $h _ { \mathrm { { c a m } } }$ using ground observations collected during initialization.

For each initialization view, ground pixels with suficient VGGT confidence are back-projected into the camera frame:

$$
\mathcal { G } _ { t } = \left\{ \hat { Z } _ { t } ( p ) \mathbf { K } _ { \mathrm { c a m } } ^ { - 1 } \bar { \mathbf { p } } \enspace \middle | \enspace \mathbf { M } _ { t } ^ { \mathrm { g n d } } ( p ) = 1 , \enspace C _ { t } ^ { \mathrm { v g g t } } ( p ) > \tau _ { \mathrm { g e o } } \right\} .\tag{A.14}
$$

RANSAC is first used to remove non-ground points. A plane $\widehat { \mathbf { n } } _ { t } ^ { \top } \mathbf { x } + \widehat { b } _ { t } = 0$ is then fitted to the remaining points, with VGGT geometric confidence used as the fitting weight.

For a normalized plane normal, $| \widehat { b } _ { t } |$ gives the reconstructed camera-to-ground distance. Each valid frame therefore yields a scale estimate

$$
\widehat { s } _ { t } = \frac { h _ { \mathrm { c a m } } } { \left| \widehat { b } _ { t } \right| } , \qquad s _ { 0 } = \mathrm { m e d i a n } \left( \left\{ \widehat { s } _ { t } \ | \ t \in \mathcal { F } _ { \mathrm { v a l i d } } \right\} \right) .\tag{A.15}
$$

A scale observation is accepted only if the ground plane contains at least $N _ { \operatorname* { m i n } } ^ { \mathrm { g n d } }$ inliers, its normal is suficiently close to the expected vertical direction, the weighted plane-fitting residual is below the prescribed tolerance, and the resulting scale lies within the admissible interval $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ]$

Scale observations are collected throughout the initial panoramic scan. The nominal initialization uses $T _ { \mathrm { i n i t } } ~ = ~ 1 2$ headings. If fewer than three reliable scale estimates are available, additional observations are collected until at leas three valid measurements have been obtained. Their median $s _ { 0 }$ is then used as the episode-level metric scale.

## A.3.1 CleanDIFT-based Semantic Consistency Filtering

For each candidate correspondence in $\mathcal { C } _ { t } ^ { 0 }$ generated by the geometric matching module between consecutive frames, the system measures semantic consistency using the CleanDIFT descriptors at the corresponding image locations. The semantic similarity is computed as

$$
\gamma _ { i } = \frac { { { \bf { f } } _ { t - 1 } } { { \left( { { \bf { u } } _ { i } ^ { t - 1 } } \right) } ^ { \top } } { { \bf { f } } _ { t } } \left( { { \bf { u } } _ { i } ^ { t } } \right) } { { { \left\| { { { \bf { f } } _ { t - 1 } } \left( { { \bf { u } } _ { i } ^ { t - 1 } } \right) } \right\| } _ { 2 } } { { \left\| { { { \bf { f } } _ { t } } \left( { { \bf { u } } _ { i } ^ { t } } \right) } \right\| } _ { 2 } } } .\tag{A.16}
$$

Since the descriptors are $\ell _ { 2 } \cdot$ -normalized, $\gamma _ { i }$ is equivalent to their inner product.

The semantically and geometrically consistent correspondence set is defined as

$$
\mathcal { C } _ { t } ^ { \mathrm { s e m } } = \left\{ c _ { i } \in \mathcal { C } _ { t } ^ { 0 } \mid \gamma _ { i } \geq \tau _ { \mathrm { s e m } } , C _ { i } ^ { t - 1 } \geq \tau _ { \mathrm { g e o } } , C _ { i } ^ { t } \geq \tau _ { \mathrm { g e o } } \right\} ,\tag{A.17}
$$

where $\tau _ { \mathrm { s e m } }$ and $\tau _ { \mathrm { g e o } }$ denote the semantic similarity and geometric-confidence thresholds, respectively, and $C _ { i } ^ { t - 1 }$ and $C _ { i } ^ { t }$ are the VGGT geometric confidence values of the two corresponding 3D points.

This filtering combines semantic consistency from CleanDIFT with geometric reliability from VGGT. The semantic criterion removes correspondences that are visually inconsistent despite geometric agreement, while geometric confidence suppresses points with unreliable reconstruction quality. The semantic threshold is selected such that $\tau _ { \mathrm { s e m } } > 0$ ensuring that the similarity values used in subsequent correspondence weighting remain non-negative.

Similarity alignment. Given the filtered correspondence set $\mathcal { C } _ { t } ^ { \mathrm { s e m } }$ , RANSAC first provides a robust initial estimate of the inter-frame similarity transformation while removing geometrically inconsistent correspondences. The transfor mation is then refined by solving a confidence-weighted robust optimization problem:

$$
\left( \hat { s } _ { t - 1 , t } , \hat { \mathbf { R } } _ { t - 1 , t } , \hat { \mathbf { t } } _ { t - 1 , t } \right) = \operatorname* { a r g m i n } _ { \substack { s _ { t - 1 , t } , \mathbf { R } _ { t - 1 , t } , \mathbf { t } _ { t - 1 , t } } } \sum _ { \substack { c _ { i } \in \mathcal { C } _ { t } ^ { \mathrm { s m } } } } w _ { i } \rho \left( \left\| \mathbf { x } _ { i } ^ { t } - \left( s _ { t - 1 , t } \mathbf { R } _ { t - 1 , t } \mathbf { x } _ { i } ^ { t - 1 } + \mathbf { t } _ { t - 1 , t } \right) \right\| _ { 2 } ^ { 2 } \right) ,\tag{A.18}
$$

where $\hat { s } _ { t - 1 , t } , \hat { \mathbf { R } } _ { t - 1 , t } ,$ and $\hat { \mathbf { t } } _ { t - 1 , t }$ denote the estimated scale, rotation, and translation, respectively, and $\rho ( \cdot )$ represents the robust loss function.

Each correspondence is weighted according to both geometric confidence and semantic consistency:

$$
w _ { i } = \operatorname* { m i n } \left( C _ { i } ^ { t - 1 } , C _ { i } ^ { t } \right) \cdot \gamma _ { i } ,\tag{A.19}
$$

where $w _ { i }$ measures the contribution of correspondence $c _ { i }$ to the robust optimization. Taking the minimum of the two geometric confidence values ensures that a correspondence receives a lower weight when either endpoint is unreliable, while $\gamma _ { i }$ reduces the influence of semantically inconsistent matches.

The estimated transformation is accepted only if the RANSAC consensus set contains at least $N _ { \mathrm { m i n } } ^ { \mathrm { i n l i e r } }$ valid correspondences and the estimated scale satisfies the bounded scale-variation constraint:

$$
\begin{array} { r } { \left| \log \hat { s } _ { t - 1 , t } \right| \leq \epsilon _ { s } . } \end{array}\tag{A.20}
$$

If RANSAC fails or the number of valid inliers is below $N _ { \mathrm { m i n } } ^ { \mathrm { i n l i e r } }$ , the previous pose estimate is retained, and the current geometric observation is excluded from fusion into the global metric map.

## A.4 Constraint Evaluation Details

The constraint manager uses diferent temporal evidence models for object, location, and direction constraints. Instead of switching a sub-instruction based on a single semantic prediction, it accumulates evidence across observations and combines it with the estimated trajectory.

Object constraints. Grounding DINO provides open-vocabulary bounding boxes, and SAM2 refines each accepted detection into an instance mask. Mask pixels with valid reconstructed geometry are transformed into the world frame, and the target position is robustly estimated from the resulting 3D points.

To reduce isolated false detections, an object constraint is confirmed only after consistent support is obtained across multiple semantic perception events. The target must also satisfy the spatial relationship specified by the instruction. For ordinary approach-type constraints, the agent must be within the efective target range $r _ { \mathrm { o b j } }$ before the constraint i marked as complete. The target position is updated whenever additional reliable observations become available.

Location constraints. For each semantic region $\ell ,$ we maintain a compact state

$$
q _ { \ell } ^ { t } \in \{ \mathrm { u n k n o w n } , \mathrm { i n s i d e } , \mathrm { o u t s i d e } \} .\tag{A.21}
$$

BLIP-based VQA provides the primary region-membership observation, while BLIP-2 image–text relevance is used when direct VQA evidence is unavailable or unreliable.

Raw predictions are temporally filtered. A transition into the inside state requires $\kappa _ { \mathrm { i n } }$ consecutive positive observations, whereas a transition from inside to outside requires $\kappa _ { \mathrm { o u t } }$ consecutive negative observations. The system also records whether the region has been previously visited and whether a stable entry has occurred during the current sub-instruction.

This compact state representation supports the location expressions required by VLN instructions. An entry constraint is satisfied by a stable transition into the region; an exit constraint requires a stable transition from inside to outside; and a visited constraint remains satisfied once a stable visit has occurred. A traversal constraint is stricter: the agent must enter the region during the current sub-instruction, remain inside it for the required duration, and subsequently exit. Therefore, a single positive VQA prediction cannot satisfy a traversal instruction.

When a new sub-instruction becomes active, long-term region visitation information is preserved, while sub-instructionspecific entry and duration records are reset.

Turn constraints. Explicit turning commands are evaluated from heading changes. When sub-instruction k starts, the current heading $\psi _ { k } ^ { \mathrm { r e f } }$ is stored. A turn constraint is satisfied only when the accumulated signed heading change relative to this reference falls within the angular interval associated with the requested turn. These constraints depend only on orientation and do not require translational motion.

Relative motion constraints. Instructions such as “move left” or “continue forward” describe displacement relative to the heading at the beginning of the sub-instruction. Let $\mathbf { v } _ { \mathrm { r e f } , k }$ denote this reference heading on the ground plane. Over a temporal window of length τ, we compute

$$
\begin{array} { r } { \begin{array} { r l } { \mathbf { v } _ { \mathrm { d i s p } } ^ { t } = \mathbf { p } _ { t } ^ { \mathcal { W } } - \mathbf { p } _ { t - \tau } ^ { \mathcal { W } } , } & { \quad \theta _ { t } ^ { \mathrm { m o t } } = \operatorname { a r c c o s } \frac { \left. \mathbf { v } _ { \mathrm { r e f } , k } , \mathbf { v } _ { \mathrm { d i s p } } ^ { t } \right. } { \left\| \mathbf { v } _ { \mathrm { r e f } , k } \right\| _ { 2 } \left\| \mathbf { v } _ { \mathrm { d i s p } } ^ { t } \right\| _ { 2 } } . } \end{array} } \end{array}\tag{A.22}
$$

Direction is evaluated only when $\| \mathbf { v } _ { \mathrm { d i s p } } ^ { t } \| _ { 2 } \geq \epsilon _ { \mathrm { d i s p } }$ . The side of the reference direction is determined from

$$
\begin{array} { r } { \varsigma _ { t } = v _ { \mathrm { r e f } , k , x } \Delta z _ { t } - v _ { \mathrm { r e f } , k , z } \Delta x _ { t } . } \end{array}\tag{A.23}
$$

We classify motion using angular boundaries of $1 5 ^ { \circ }$ and $1 2 0 ^ { \circ }$ . Displacements with $\theta _ { t } ^ { \mathrm { m o t } } < 1 5 ^ { \circ }$ are classified as forward, whereas $\theta _ { t } ^ { \mathrm { { m o t } } } \geq 1 2 0 ^ { \circ }$ are classified as backward. Intermediate directions are classified as left or right according to the sign of $\varsigma _ { t } .$

This displacement-based definition is intentionally separated from turn evaluation. An in-place rotation therefore cannot satisfy a relative motion constraint, and lateral displacement cannot satisfy an explicit turn constraint. Backward instructions are normalized by the instruction parser into a turn followed by forward motion, while the backward category above is retained only for interpreting observed displacement.

## A.5 Construction of the Semantic Value Map

Source normalization. The CleanDIFT relevance map and BLIP-2 image–text relevance map are independently normalized to [0, 1] over geometrically valid pixels before fusion in Eq. (6). Pixels removed by geometric confidence, visibility, or traversability filtering are excluded from the normalization process.

When the valid value range of a relevance map is smaller than $\delta _ { x } .$ , the map is considered insuficiently discriminative for spatial guidance, and the corresponding source reliability is set to zero. This prevents nearly uniform similarity responses from dominating semantic-value estimation simply because their absolute scores are high.

Visual-reference relevance. CleanDIFT visual-reference maps are constructed on the native feature grid. Instance masks associated with the active object or landmark are resized to this grid and used to aggregate reference descriptors. Dense cosine similarity with the resulting reference descriptor produces $\mathbf { A } _ { t } .$ which is then resized to the navigation image resolution using bilinear interpolation.

For BLIP-2, the current observation is evaluated against the language representation of the active constraint. Local image regions are scored independently and interpolated to obtain the dense relevance map $\mathbf { B } _ { t }$ . The source reliability terms $q _ { t } ^ { a }$ and $q _ { t } ^ { b }$ in Eq. (6) are set to zero whenever the corresponding source is unavailable or fails the validity checks described above.

Observation confidence. The observation-confidence map integrates geometric validity, pose reliability, viewing direction, observation distance, surface consistency, and traversability. Pixels near the optical axis and those supported by reliable reconstruction and registration receive higher weights, whereas uncertain geometry, grazing-angle observations, obstacle regions, and unreliable reprojections are suppressed.

Historical image-space semantic values are transferred to the current frame using Eq. (A.12). A reprojected sample is retained only if the transformed 3D point has positive depth, lies within the current image boundary, and is consistent with the current reconstructed depth under the occlusion-consistency tolerance. Samples violating any of these conditions are removed by ${ \bf M } _ { t } ^ { \mathrm { r e p } }$ before the confidence-weighted temporal update in Eq. (7).

This reprojection is used to stabilize semantic evidence across nearby viewpoints rather than to maintain an indefinitely persistent image-space memory. Long-term semantic information is instead stored in the world-aligned global map.

## A.6 Map Fusion and Candidate Waypoint Generation

## A.6.1 Local Window and Global Fusion

The fixed-size local map is centered on the current agent position and remains aligned with the world axes. As the agent moves, the local window is translated according to Appendix A.1. Cells entering the window are recovered from the extensible global map whenever previous observations are available; otherwise, they are initialized as unknown.

Traversability and occupancy evidence are accumulated in separate layers. Occupancy is represented using a temporally decayed log-odds update:

$$
\ell _ { t } ( { \mathbf { x } } ) = \lambda _ { \mathrm { o c c } } \ell _ { t - 1 } ( { \mathbf { x } } ) + \ell _ { \mathrm { o b s } } ( { \mathbf { x } } ) ,\tag{A.24}
$$

where $0 < \lambda _ { \mathrm { o c c } } \leq 1$ . Occupied observations contribute positive log-odds evidence, while free-space support is maintained independently in the traversability layer. The occupancy probability is recovered through the logistic transform.

This separation allows outdated obstacle evidence to decay while preventing uncertain free-space observations from immediately overriding previously observed obstacles. After independent fusion, occupied cells mask the traversability layer, ensuring that occupancy takes precedence when the two sources disagree.

The action–motion consistency signal provides an additional geometric correction mechanism. When repeated forward commands produce negligible visually estimated displacement, cells immediately along the attempted path receive additional occupancy evidence. This correction only modifies the local geometric interpretation of the failed motion and does not afect semantic values.

Semantic values are fused into the same world-aligned cells using observation confidence. Dense semantic features are averaged across repeated observations and re-normalized after fusion. Observation count and exploration state are maintained independently, allowing semantic confidence to be distinguished from geometric coverage.

When a sub-instruction changes, target-specific semantic values from the previous stage are attenuated rather than removed. The geometric map and exploration history remain unchanged.

## A.6.2 Semantic-Enhanced Candidate Generation

Waypoint candidates are generated only within the safe reachable region $\mathcal { R } _ { t }$ defined in Eq. (9). Instead of selecting isolated local maxima, we partition the reachable semantic map into spatially coherent regions.

The clustering distance combines spatial proximity, semantic value, and dense semantic-feature similarity:

$$
d _ { \mathrm { S L I C } } = d _ { \mathrm { s p a t i a l } } + \lambda _ { 1 } d _ { \mathrm { v a l u e } } + \lambda _ { 2 } d _ { \mathrm { s e m } } , \qquad d _ { \mathrm { s e m } } = 1 - \left. \mathbf { F } _ { t } ^ { \mathrm { m a p } } ( \mathbf { x } ) , \mathbf { F } _ { t } ^ { \mathrm { m a p } } ( \mathbf { x } ^ { \prime } ) \right. .\tag{A.25}
$$

Each cluster is intersected with $\mathcal { R } _ { t }$ and further divided into 8-connected components. Components with areas smaller than $A _ { \mathrm { m i n } }$ are discarded. For each remaining component, the navigable cell closest to its geometric center is selected as the representative waypoint.

This procedure separates candidate construction from candidate scoring. The semantic score used by the high-level planner remains the region-level score defined in Eq. (10).

## A.6.3 Waypoint History and Loop Suppression

The planner maintains recent pairs of agent positions and selected waypoints in world coordinates. A newly generated waypoint is considered a loop candidate when the agent returns to the neighborhood of a previous decision state and the candidate is also close to the waypoint selected at that state. The corresponding distance tolerances are $d _ { \mathrm { c y c , p o s } }$ and $d _ { \mathrm { c y c , w p } } .$

Loop candidates are suppressed only during waypoint selection; the semantic and geometric maps remain unchanged. If history filtering removes all otherwise valid candidates, the planner falls back to the highest-scoring safe candidate before loop suppression. This prevents the history mechanism from introducing an artificial dead end.

Waypoint hysteresis is applied after candidate scoring. The previous waypoint is retained as long as it remains reachable, unreached, and semantically competitive with newly generated candidates. It is replaced only when it becomes unsafe, is reached, loses suficient semantic support, or is associated with a detected navigation failure.

## A.7 LangGraph Execution and Recovery

LangGraph implements the stateful execution policy described in Sec. 3.3. The graph maintains persistent episode state instead of treating perception, planning, and recovery as independent operations.

Persistent and transient state. Persistent state includes the active sub-instruction index, estimated camera and body poses, the global semantic-geometric map, constraint states, exploration history, and waypoint history. These variables are preserved across transitions between normal navigation and recovery.

Transient state contains the currently selected waypoint, immediate low-level plans, perception-cache status, and recovery-control variables. These quantities can be invalidated and recomputed without discarding the spatial knowl edge accumulated earlier in the episode.

Recovery policy. Recovery follows the three levels described in Sec. 3.3.5. The levels are executed progressively rather than simultaneously.

Viewpoint recovery is attempted first because it preserves all persistent semantic and geometric states. If it does not recover suficiently reliable target evidence, semantic recovery is invoked. Navigation redirection is used only when the current semantic hypothesis still cannot produce a useful reachable waypoint.

The frontier selected in ${ \mathrm { R } } _ { 3 }$ serves as an intermediate exploration target rather than a replacement for the instruction goal. Reaching this target therefore triggers perception refresh and resumes the interrupted sub-instruction.

Table 8. Summary of the recovery policy.
<table><tr><td>Stage</td><td>Revision</td><td>Return condition</td></tr><tr><td> $\operatorname { R } _ { 1 } { \mathrm { : } }$  Viewpoint recovery</td><td>Perform a panoramic observation at the current pose and recompute instruction-relevant visual evidence from the newly observed headings.</td><td>Reliable task-relevant evidence provides a valid reachable waypoint.</td></tr><tr><td> $\operatorname { R } _ { 2 } \colon$  Semantic recovery</td><td>Invalidate semantic observations associated with the current target hypothesis and reconstruct grounding, visual references, and relevance esti- mates from recent observations.</td><td>A refreshed semantic hypothesis provides a valid reachable waypoint.</td></tr><tr><td> $\operatorname { R } _ { 3 } \colon$  Navigation redirection</td><td>Temporarily select a reachable frontier or unex- plored location within the safe connected region while preserving the active instruction.</td><td>The intermediate target provides a new viewpoint that allows normal semantic planning to resume.</td></tr></table>

After successful recovery, the global map, accumulated semantic memory, constraint history, and waypoint history are preserved. The temporary waypoint, pending low-level action, and stagnation state are cleared. Semantic perception is then forced to refresh before returning to the normal navigation loop. Geometry, map projection, and local planning are recomputed from the updated state when necessary.

Each recovery level is attempted conservatively to avoid repeatedly applying the same inefective correction. If recovery remains unsuccessful after $R _ { \mathrm { m a x } }$ attempts, the episode is marked as unrecoverable and terminated. In our experiments, $R _ { \mathrm { m a x } } = 3$

## B Model and Hyperparameter Configuration

We evaluate the proposed system in the zero-shot setting on the 550-episode subset of the R2R-CE val-unseen split described in Sec. 4.1. All perception and foundation-model components use publicly available pretrained weights, and no R2R-CE navigation trajectories are used for task-specific training.

Table 9 summarizes the main thresholds and navigation parameters required to reproduce the proposed method. Parameters used only for numerical stability are set to standard implementation defaults and omitted unless they influence decision boundaries.

Table 9. Core model and hyperparameter settings.
<table><tr><td>Parameter</td><td>Value</td><td>Description</td></tr><tr><td></td><td colspan="2">Geometric Reconstruction and Registration</td></tr><tr><td>CleanDIFT similarity threshold  $\tau _ { \mathrm { s e m } }$ </td><td>0.65</td><td>Minimum cosine similarity required to retain semantically consistent inter- frame correspondences.</td></tr><tr><td>VGGT confidence threshold  $\tau _ { \mathrm { g e o } }$ </td><td>0.10</td><td>Minimum geometric confidence used for ground-plane estimation and cor- respondence filtering.</td></tr><tr><td>Point-cloud confidence percentile</td><td>25%</td><td>Points within the lowest confidence quartile are removed before high- confidence geometric processing.</td></tr><tr><td>RANSAC inlier threshold Initialization scan directions</td><td>0.10 m</td><td>Metric residual threshold used for robust inter-frame geometric verification. Uniformly spaced headings used during the initial panoramic scan and scale</td></tr><tr><td> $T _ { \mathrm { i n i t } }$ </td><td>12</td><td>initialization.</td></tr><tr><td colspan="5">Semantic Perception and Constraint Evaluation</td></tr><tr><td>Grounding DINO detection threshold</td><td>0.25</td><td>Minimum confidence required for open-vocabulary candidate detections.</td></tr><tr><td>Effective object range  $r _ { \mathrm { o b j } }$ </td><td>2.0 m</td><td>Maximum horizontal distance for completing ordinary object-approach con- straints.</td></tr><tr><td>Object temporal support  $K _ { \mathrm { o b j } }$ </td><td>5</td><td>Number of semantic perception events used for temporal denoising of object constraints.</td></tr><tr><td>Region debounce  $\kappa _ { \mathrm { i n } } , \kappa _ { \mathrm { o u t } }$ </td><td>3</td><td>Number of consecutive observations required for stable location entry and exit.</td></tr><tr><td>Semantic translation trigger  $\tau _ { p }$ </td><td>0.30 m</td><td>Maximum translation interval between routine semantic perception re- freshes.</td></tr><tr><td>Semantic rotation trigger  $\tau _ { \psi }$  Forward-motion angular boundary</td><td> $3 0 ^ { \circ }$   $1 5 ^ { \circ }$ </td><td>Maximum heading change between routine semantic perception refreshes. Maximum deviation from the reference heading for forward displacement</td></tr><tr><td></td><td></td><td>classification.</td></tr><tr><td>Backward-motion angular boundary</td><td> $1 2 0 ^ { \circ }$ </td><td>Minimum deviation from the reference heading for backward displacement classification.</td></tr><tr><td colspan="5">Mapping and Navigation</td></tr><tr><td>Map resolution</td><td>0.05 m/cell</td><td>Spatial resolution of the world-aligned bird&#x27;s-eye-view map.</td></tr><tr><td>Waypoint arrival threshold  $d _ { \mathrm { w p } }$  Safe-traversal margin  $d _ { \mathrm { s a f e } }$ </td><td>0.25 m 0.25 m</td><td>Distance below which an intermediate waypoint is considered reached. Obstacle-clearance margin used when constructing the safe reachable re-</td></tr><tr><td></td><td></td><td>gion.</td></tr><tr><td>Final-target distance threshold</td><td>3.00 m</td><td>Distance criterion used with final constraint completion before issuing STOP.</td></tr><tr><td>Cycle pose threshold  $d _ { \mathrm { c y c , p o s } }$ </td><td>0.50 m</td><td>Maximum position distance for matching a previous navigation decision state.</td></tr><tr><td>Cycle waypoint threshold  $d _ { \mathrm { c y c , w p } }$ </td><td>0.30 m</td><td>Maximum waypoint distance for identifying a repeated waypoint decision.</td></tr><tr><td>Yaw-oscillation threshold  $\delta _ { \psi }$ </td><td> $2 5 ^ { \circ }$ </td><td>Minimum per-step heading change counted by the oscillation detector.</td></tr><tr><td>Maximum recovery attempts  $R _ { \mathrm { m a x } }$ </td><td>3</td><td>Maximum number of recovery attempts before an episode is declared unre- coverable.</td></tr></table>