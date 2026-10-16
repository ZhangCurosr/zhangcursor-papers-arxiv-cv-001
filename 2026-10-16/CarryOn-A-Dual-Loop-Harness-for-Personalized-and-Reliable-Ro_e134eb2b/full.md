# CarryOn: A Dual-Loop Harness for Personalized and Reliable Robot Assistance

Zixuan Wang<sup>1∗</sup> Yuqi Liu<sup>2∗</sup> Yuxin Chen<sup>1∗</sup> Sengqiao Yang<sup>2</sup> Jinhui Ye<sup>1</sup>

Pengguang Chen<sup>3</sup> Changsheng Lu<sup>1</sup> Shu Liu<sup>3</sup> Bei Yu<sup>2</sup> Jiaya Jia<sup>1,3</sup>

HKUST<sup>1</sup> CUHK<sup>2</sup> SmartMore<sup>3</sup>

https://github.com/JIA-Lab-research/CarryOn

## Abstract

While generalist vision-language-action (VLA) models have enabled diverse open-world manipulation, real-world robotic assistance requires two critical capabilities they currently lack: personalization and reliable execution. First, executing an instruction like ‘make me a sandwich’ requires grounding subtask decomposition in persistent user preferences rather than relying on a single generic solution. Second, reliable deployment demands explainable safety assurance that proactively prevents hazards and gracefully recovers from occasional physical failures. To address these gaps, we present CarryOn, a dual-loop harness designed to augment a frozen VLA policy. The planning loop maintains persistent user-level and scene-level context that is continually grounded against new observations and revised through interaction, enabling personalized task selection and execution grounding across conversations and changing scenes. The execution loop actively supervises policy rollouts, shielding actions from predicted hazardous contact. When execution fails, the system invokes reusable control-based recovery skills that restore task progress and are refined through reflection on earlier failures. On a real robot and in simulation, CarryOn dispatches preference-consistent subtasks across users and scenes, reduces the violation rate of an already safetyfinetuned VLA from 52.6% to 6.1%, and increases recovery success from 13% to 80% when refined skills are reused.

![](images/5cf4df0d3091839136cea1595dc08dc809dbf488dd75f573117931050659bd75.jpg)  
Figure 1. Serving food to a vegetarian user. (1) The user asks only, “Serve me food.” (2) CarryOn retrieves the context from memory. (3) The planner dispatches a sequence of subtasks to a frozen VLA policy: open the pink cover and, if the carrot is underneath, place it in the bowl. (4) During execution, a proactive shield steers the gripper and the carried carrot around an obstacle, and reactive recovery re-grasps the carrot to resume the task

## 1 Introduction

Robot assistance ultimately aims to serve people in their everyday lives. Generalist vision-language-action (VLA) models have advanced this goal by learning to execute diverse manipulation instructions across tasks and environments [1–3]. Large-scale robot datasets further support learning across tasks and embodiments [4–6]. However, progres in instruction-conditioned manipulation alone does not bridge the gap to everyday assistance. A robot must know both what to do to meet the user’s needs in the current environment and how to carry it out reliably as physical conditions change. Bridging this gap requires a system that connects context-sensitive task decisions with reliable execution.

A harness is the system surrounding a base model that orchestrates execution and decides how the model plans, perceives and manages context, stores artifacts, and evaluates results [7]. Language-agent systems illustrate how memory and reflection can support decisions across interactions [8, 9]. In the field of robotics, hierarchical VLA systems connect high-level requests to manipulation instructions [10]. Recent concurrent works on embodied harnesses use memory to steer frozen VLAs and coordinate heterogeneous policies [11, 12], or evolve runtime critics and recovery skills from execution experience [13]. These works mainly focus on how to improve VLA models’ performance in out-of-distribution (OOD) scenarios. However, for everyday assistance, reliable execution must also be coupled with deciding what is appropriate for the user in the current situation.

We study this connection through personalized robot assistance, which exposes two challenges. First, task decisions must remain aligned with the user’s needs as interactions and scenes change, and the same request can require differen actions. Learning preferences or identifying personal objects [14, 15] provides context, while in reality the context can be dynamically changed and thus requires revision in stored memory. Second, the resulting plan must be supported by reliable physical execution: impending hazardous contact calls for intervention [16] to ensure safety, while a failure that has already occurred calls for repair [17] to ensure robustness. The harness must connect these responses to the intended task and retain failure experience to improve later recovery.

Consider a user who asks a robot to “serve me food” (Fig. 1). The robot should respect an earlier vegetarian preference without requiring it to be repeated in the query. It may look beneath a remembered cover for a carrot, but finding meat instead must change its original plan. Even when the ingredient is suitable, an obstacle or a lost grasp can interrupt delivery to the bowl. The harness must preserve the user’s dietary requirement when observations change the plan, restore progress when a graspfails, and use thatfailure to improve later recovery.

We present CarryOn, a dual-loop harness that connects adaptive task decisions with reliable execution for personalized robot assistance. Its central design makes task context and recovery behavior persistent, revisable parts of the runtime around a frozen VLA. The planning loop determines what to do for the user and scene, and the execution loop protects and repairs the active subtask. Visual feedback connects them, informing plan revision and determining whether execution should advance or invoke recovery while retaining the task goal.

The planning loop supports contextual adaptation through memories. User context expresses needs and preferences, while scene context helps determine where and how they can be fulfilled. The planner combines both with the current multimodal request and available observations to select an endpoint and supported subtasks. Dialogue revises user context, and observations trigger explicit scene-memory updates before subsequent plans proceed. Memory thus participates in a feedback loop: it guides action, while the resulting evidence updates what later plans can rely on.

The execution loop supports physical reliability through fast proactive shielding and slower reactive recovery. A fast proactive shield predicts gripper and carried-object motion and filters commands against geometric contact constraints. After a failure, slower, event-triggered recovery skills repair the physical state so the interrupted task can continue. Offline reflection then uses failures of the current recovery version to revise its parameters and strategies, retaining successive versions in a skill library for later attempts. Immediate protection and reactive repair operate within an episode, and accumulated experience changes recovery behavior across attempts. This design keeps the VLA fixed while allowing the recovery pathway to improve through repeated reflection.

Contributions. In this paper, we present a dual-loop harness, called CarryOn, with the features of personalization and reliable execution. The planning loop of the harness connects persistent user and scene contexts with observation driven revision, supporting personalized task decisions across interactions, scenes, and query modalities, while the execution loop combines proactive geometric shielding with reusable and iteratively refined recovery skills. Real-world studies demonstrate the effectiveness of our harness in planning preference-oriented tasks across diverse users and scenes. Notably, our method strongly increases recovery success rates from 13% to 80% using refined skills, and reduces the constraint-violation rate from 52.6% to 6.1%.

## 2 Related Work

Personalization through persistent context. TidyBot learns placement preferences that generalize to later requests [14], while Bring My Cup personalizes through reference-image memory and visual prompting [15]. Embodied personalization studies memory across user interactions [18, 19], and language agents retrieve and consolidate experience for subsequent decisions [8, 9]. Unlike previous works, CarryOn connects persistent preferences with revisable scene knowledge: user memory constrains task selection, while the scene memory guides later actions.

Harnesses for robot policy execution. Code as Policies generates programs over perception and control APIs [20]. Language-model planners ground instructions through affordances, feedback, and multimodal observations [21–26], while uncertainty-aware planners decide when to act or seek information [27]. Hierarchical VLA systems connect instructions to visuomotor execution [10]; RoboHarness uses memory to orchestrate heterogeneous policies [12]. Harness VLA combines a frozen contact-rich primitive with analytic operations and memory-guided retries [11]. In contrast, CarryOn retains the VLA to execute manipulation and couples persistent personalization with action filtering and physical repair. Both loops share the active task, preserving the user-conditioned goal through recovery and using observed outcomes to govern task progression.

Recovery and adaptation from failure. Vision-language models provide semantic failure feedback [28], while FailSafe trains failure reasoning and recovery from generated failure recovery cases [29]. RecoveryChaining learns local policies that return control to nominal controllers [17], and Zetta evolves runtime critics and recovery skills around a frozen policy [13]. Language agents accumulate experience through memory and reusable skills [30–32]. In our method, CarryOn confines refinement to control-based recovery: offline reflection revises supported parameters and strategies, retaining successive skills for later attempts. This separates immediate repair from experience accumulation while leaving the VLA and geometric shield fixed.

Geometric protection during manipulation. Latent-space safety filters use reachability analysis in learned world models [33]; AnySafe adapts such filters to parameterized constraints [34]. VLSA adds a barrier-function safety layer [16], and Code-as-Monitor generates executable constraints for failure monitoring [35]. In our design, the proactive shield predicts the spatial response to policy-predicted actions and filters actions according to minimumdistance constraints between the gripper or carried object and task-specific obstacles. This provides a check before hazardous contact, complementing recovery after failure without online learning.

## 3 Method

CarryOn uses a dual-loop harness to adapt task decisions to the user and environment and to support reliable physical execution. The planning loop turns a request into subtasks using persistent user and scene knowledge, then updates that knowledge through dialogue and observation. The execution loop runs a pretrained VLA, checks its actions for collision risk, and invokes control-based recovery skills when manipulation fails. Experience is retained in two forms: context for future planning and recovery experience that supports successive revisions of a versioned skill library. Fig. 2 summarizes the runtime, and the following sections describe how each form of experience is used and revised.

## 3.1 Overview of CarryOn

Given a request c and an optional reference image, the planner combines user memory $M ^ { u }$ , scene memory $M ^ { s }$ , and descriptions of the available skills to produce a plan. Each planned subtask specifies an instruction $\ell _ { i } ,$ an endpoint that provides the requested skill, and optional dependencies on earlier subtasks. Endpoints include manipulation stations and information-gathering tools, so a plan can request an observation or external information before deciding what to manipulate. At a manipulation endpoint, the frozen VLA predicts action chunks $U _ { t } = \pi _ { \theta } ( o _ { t } , \ell _ { i } )$ from observation $o _ { t }$ and the subtask instruction $\ell _ { i } .$ A geometric shield filters these chunks using task-dependent contact constraints before execution. The Visual Judge compares recent camera images with the scene at the start of the subtask to assess progress toward $\ell _ { i }$ . It reports running, success, or failed: success advances the plan, while failure invokes recovery on the same subtask. Observation results return to the planner for plan and memory updates.

![](images/c06ff061b3a2c7cd18cc54f2accc33210a4d2b3a7a8d7d1c0b738fa62d91fef9.jpg)  
Figure 2. Two loops, one task context. The planning loop grounds a request in persistent context and dispatches supported subtasks. The execution loop retains the VLA for manipulation, with proactive shielding before hazardous contact and recovery after failure. A shared visual judge advances the plan on success or triggers local recovery on failure. Recovery attempts to repair the physical state so the VLA can continue the interrupted subtask. Post-episode reflection uses failure evidence to revise recovery skills, retaining successive versions in the skill library.

## 3.2 The Planning Loop: Personalization across Interactions and Scenes

## 3.2.1 Planning with persistent, revisable context

A personalized plan must satisfy the user’s needs and account for the environment in which the task will be carried out. Since the current request may leave both kinds of information unstated, we maintain user and scene memories as complementary planning context. User memory supplies preferences and object references established in earlier conversations. We store preferences, habits, and reference images of the user’s objects in user-specific profiles, which enter the planner context with each request. The planner is instructed to apply the requesting user’s profile when selecting subtasks. The profile is combined with the current instruction, so planning accounts for both retained preferences and the user’s present request.

Scene memory provides planning context for deciding where and how to carry out the task in the current environment. We store object inventories and descriptions by place. The planner combines these records with the available skill descriptions to select an endpoint and decide whether an observation is needed before manipulation. If the target is hidden or its location needs checking, the planner requests an observation. The returned facts enter the planning context and are combined with the user context to revise subsequent subtasks.

## 3.2.2 Updating context from dialogue and observation

Personalized assistance must adapt as user needs and the surrounding environment change. This requires the robot to distinguish lasting changes in user preferences from temporary requests, and to revise its understanding of the scene as new evidence becomes available. We update user memory from dialogue and scene memory from observations to keep this planning context current. For user memory, the planner is instructed to store lasting preferences during the interaction and keep temporary instructions in the current conversation. After a plan terminates, a background summarizer reviews the dialogue using the same criterion to retain user facts not yet recorded. When the user revises or retracts a preference, the corresponding profile entry is replaced or removed, so later plans use the updated requirements.

The environment can also change independently of the user’s instructions. New observations must therefore revise the robot’s understanding of the scene, both for the current task and for later requests. We require a scene-memory update whenever an observation reveals the target or finds its remembered location empty or stale. The runtime retains the observation and records the need for an update. The planner uses this evidence to specify an update\_scene operation in its next plan, revising the scene record and removing outdated information. If the planner omits this required operation, a plan validator returns the plan to the planner for revision.

## 3.3 The Execution Loop: Proactive Protection and Reactive Recovery

Reliable physical assistance requires both safety and robustness: the robot must intervene before hazardous contact and restore progress after disturbances interrupt a task. A policy trained to complete the task may still fall short of either requirement. Two observations motivate intervention around the VLA: The safety-finetuned $\pi _ { 0 . 5 }$ checkpoint, trained on 19,664 safe demonstrations [36], still violates constraints in 52.6% of our 900 evaluation episodes, compared with 71.1% for the original checkpoint (Section 4.2). In a separate LIBERO-Object policy-only probe, frozen StarVLA [37] completes 26/200 episodes (13.0%) after a grasped object is displaced by 12 cm (Fig. 3).

![](images/f073e3f04c6185073fec48fba29835f6ad0eb87bb6e6a9d2f8c2459f42302c63.jpg)  
(a) Example of perturbation

![](images/2c71bd49b6efac4943e9a53c4a4bbc22784393e16c23b49b89a81e7d7302a3e4.jpg)  
(b) VLA performance under perturbation  
Figure 3. (a) Schematic of the perturbation, varying object displacement relative to the initial lifted position. (b) Policy-only response to object displacement on LIBERO-Object. Blue bars show task success, and the upper line shows the decrease from undisturbed success in percentage points.

We therefore divide execution support into fast proactive shielding and slower, event-triggered reactive recovery. The shield checks actions predicted by the VLA before execution and filters them according to predicted geometric contact risk. Shielding reduces risks in the proposed motion, but surrounding environment may still introduce disturbance and change the physical state. Reactive recovery acts after the Judge detects a subtask failure, suspending the VLA and invoking a control-based skill to attempt repair before continue the same subtask. Both interventions operate within an episode, at action and subtask levels respectively. Offline reflection subsequently uses unresolved failures to refine the current recovery version for later attempts, while the VLA remains fixed.

## 3.3.1 Proactive shielding within an episode

A VLA policy may predict actions that cause unintended contact even when the task instruction itself is appropriate. Safe execution therefore requires checking the physical consequences of these actions before execution. We implement proactive protection with a geometric action filter, the shield, between the policy and controller. For each action chunk, it uses current RGB-D observations and the target and destination descriptions to construct obstacle sets, predict gripper and carried-object motion, and filter the actions. The output is an executable action prefix or, if the first action is blocked, a short retreat.

Grasping and placement require contact with the target and destination, so treating every scene point as an obstacle would prevent the task itself. We construct separate obstacle sets for the gripper and carried object by identifying which observed geometry belongs to these task roles. A VLM infers descriptions of the target and destination from the task instruction. The scene builder then uses text-prompted segmentation to associate these descriptions with image masks and projects the masked depth pixels into three-dimensional point sets. It removes robot self-geometry using the robot model and current state and identifies the target’s local supporting surface from geometry beneath it. Contact rules exclude permitted target, support, and destination geometry from each body’s obstacle set. Other points, including unassigned geometry, remain obstacles. Appendix C describes the grounding and contact rules used in the safety evaluation.

Motion prediction then estimates the end-effector trajectory resulting from the VLA’s predicted actions. Proposed action increments alone do not describe the realized robot displacement: the controller response and ongoing motion also affect where the gripper moves. For the next H actions $U = \left( u _ { 0 } , \dots , u _ { H - 1 } \right)$ , the shield predicts end-effector poses using executed-motion history H<sub>t</sub> and a short-horizon response model. Translation is approximated by critically damped dynamics,

$$
\ddot { p } = \omega _ { n } ^ { 2 } ( p ^ { \mathrm { g o a l } } - p ) - 2 \omega _ { n } \dot { p } ,\tag{1}
$$

where $p$ denotes end-effector position, with velocity $\dot { p }$ and acceleration ${ \ddot { p } } .$ The controller target $p ^ { \mathrm { g o a l } }$ is obtained by adding the scaled translational component of the VLA-predicted action to the end-effector position at the start of the control step. The natural frequency $\omega _ { n }$ sets the response rate. Starting from the observed motion state, we predict position and, through an analogous rotation model, orientation over the action chunk. The shield then propagates the gripper and any inferred carried-object geometry with these poses, so the next stage checks distances from the moving bodies to obstacles.

Action filtering uses these predicted body positions to decide how much of the action chunk can execute. At step $k ,$ the shield computes the distance $d _ { a , k }$ between each body a (the gripper or carried object) and its obstacle set. It releases only the actions before the first predicted step at which any distance reaches or falls below its threshold $\delta _ { a }$ . The resulting prefix length is

$$
g = \operatorname* { m i n } \Bigl ( \{ k \in \{ 0 , \ldots , H - 1 \} : \operatorname* { m i n } _ { a } ( d _ { a , k } - \delta _ { a } ) \leq 0 \} \cup \{ H \} \Bigr ) .\tag{2}
$$

When $g > 0 ,$ , it releases $\left( u _ { 0 } , \ldots , u _ { g - 1 } \right)$ , and $g = H$ permits the entire checked chunk. When $g = 0 ,$ , it withholds the candidate actions and commands a short retreat upward and away from the forbidden point nearest the end effector. It then reassesses using fresh observations. History is updated only from actions actually executed and their observed motion, so withheld actions are not treated as changes to the physical state.

## 3.3.2 Reactive recovery with reusable skills

Robust task completion requires recovering from disturbances that have already interrupted manipulation. At this point, filtering the next action may reduce further contact risk but cannot by itself restore task progress. We therefore implement reactive recovery with control-based skills that attempt to repair the physical state before the VLA retries the interrupted subtask. The recovery skill library initially contains one shared base skill for all targets. When the Judge detects a subtask failure, the runtime suspends the VLA and invokes the applicable recovery behavior from the current library version, using retained revisions where available or base version otherwise.

The base skill proceeds through observation, retrieval, verification, and resumption. For a dropped object, it first moves the arm to an observation configuration and re-localizes the target. It generates grasp candidates, executes a grasp, then lifts and checks holding from the measured gripper width. A passing check allows the arm to carry the object to a stand-off pose near the destination, where a second check assesses whether the grasp was retained during transport. If this check passes, the VLA resumes the original instruction from that pose. If retrieval fails or the second holding check does not pass, the recovery attempt is marked as failed. The runtime then attempts to return to the available grasp candidate’s pre-grasp pose, or to the recorded pre-failure pose if no candidate is available, and opens the gripper on arrival. Both paths continue in the current scene within the remaining episode budget, and the Judge assesses task completion separately from recovery and resumption.

## 3.3.3 Refining recovery skills across attempts

Failures can recur when recovery repeats an unsuitable grasp configuration or an ineffective sequence of operations. Improving recovery can therefore require changing an operation’s numerical settings or the procedure in which it is used. We use offline reflection to refine the current skill version from unresolved failures. Between evaluation rounds, reflectors inspect selected failed attempts using scene and target images, grasp-stage measurements, and the version used. Parameter memory (P) records changes to numerical settings, for example, a revised contact tool-center-point height is applied when constructing a grasp command. Strategy memory (S) records changes to the recovery control flow, such as adding retries or VLM-assisted re-grounding. With re-grounding enabled, the runtime segments current object candidates and asks a VLM to compare them with the target reference image. The selected mask is then passed to the grasp-generation routine.

After reflection produces proposed revisions, an admission check rejects unsupported formats and out-of-range numerical values. Accepted P/S updates are incorporated into the current recovery procedure and saved as the next library version. The following round evaluates that version, and its unresolved failures guide further reflection and refinement. Successive rounds build on the preceding version while the VLA weights remain fixed. Appendix D.2 details how the records are used, and Section 4.3 specifies failure selection and the two-round protocol.

![](images/0fb98cecb430f25e787f1d6775b851f762da21b1977408af71bba3047af9073f.jpg)  
Figure 4. Simulation settings. (a)–(d) RoboLab scenes for personalized planning and memory revision; (e)–(f) LIBERO-Safet tasks for proactive shielding; (g)–(h) controlled perturbations in LIBERO and recorded failure states in LIBERO-Recovery.

## 4 Experiments

We evaluate the planning, shielding, and recovery mechanisms. Real-robot studies test their use in physical assistance (Fig. 5); controlled simulation studies extend the evaluation across scenes, safety constraints, and disturbances (Fig. 4).

Table 1. Real-robot results. Planning rows count correct subtasks; execution rows count completed episodes.
<table><tr><td>Study</td><td>Condition</td><td>Result</td></tr><tr><td rowspan="3">Sandwich Preparation</td><td>Textual</td><td>24 /24</td></tr><tr><td>Memory</td><td>24 / 24</td></tr><tr><td>Preference update</td><td>20 / 24</td></tr><tr><td rowspan="3">Activity-dependent Packing No-habit</td><td>Memory</td><td>16 / 16</td></tr><tr><td></td><td>16 / 20</td></tr><tr><td>Isolation</td><td>16 / 18</td></tr><tr><td rowspan="3">Covered ingredient, scene memory</td><td>Write and retrieve</td><td>12 / 12</td></tr><tr><td>Search without memory</td><td>8 / 10</td></tr><tr><td>Revise stale memory</td><td>9 / 12</td></tr></table>

Execution loop
<table><tr><td>Condition Result</td></tr><tr><td>Obstacle avoidance</td></tr><tr><td>Raw policy 0/ 20</td></tr><tr><td>Ours, contact-free 20 / 20</td></tr><tr><td>Recovery after object removal</td></tr><tr><td>RAW 3 / 20</td></tr><tr><td>BASE (z 0.00, 1 attempt) 9 / 20</td></tr><tr><td>FULL (z −0.02, 3 attempts) 18 / 20</td></tr></table>

## 4.1 Personalization across Interactions and Scenes

We test whether persistent context supports appropriate task decisions, remains useful after updates, and reduces scene search. Real-robot studies report subtask dispatch accuracy: the fraction of dispatched subtasks judged correct under task requirements and applicable user preferences (Table 1). RoboLab complements these studies with controlled endpoint selection and memory revision.

Sandwich preparation: retaining and revising preferences. This experiment tests whether user memory preserves user preferences across conversations and supports plan changes when those preferences are revised. The robot prepares sandwiches from toast, lettuce, and ham across four ingredient layouts. We compare three conditions: Textual specifies how to make the sandwich directly in the query; Memory introduces the user’s preference in an earlier conversation, restarts the program, and then asks only “Make me a sandwich”; Preference update uses another conversation to revise the user’s dietary preference before the same request. Conversations also contain unrelated remarks, requiring only relevant preferences to be retained. Dispatch accuracy is 100% (24/24) for both Textual and Memory, and 83.3% (20/24) after updates. These results show that the system retains user preferences across conversations and follows revised preferences in 20 of 24 dispatched subtasks.

Activity-dependent packing: combining profiles, intent, and online information. This experiment tests whether the planner can model the user’s profile with more complex information to make personalized decisions in a multi-user setting. The robot selects from a cap, cup, umbrella, charger, and towel for different activities such as running, visiting the gym, going to the park, or commuting to work. The planner queries the current weather at the user’s residence through online search and combines it with the requested activity and personal habits. Two users have distinct residences and habits, including preferences that differ from generic packing choices. Memory supplies the user’s profile through conversation before restarting the program; No-habit supplies the same information without personal habits and relies on commonsense reasoning for planning; Isolation registers both profiles before one user’s request to test profile isolation. Required and forbidden items are judged against recorded weather and preferences. Dispatch accuracies are 100% (16/16), 80.0% (16/20), and 88.9% (16/18), respectively. These results support combining persistent user profiles, current intent, and online information for personalized decisions in a multi-user setting

![](images/00dbb70a1261ca42b0f2622ec26ab882c8a955cae4d4a60d7653f852d0e48876.jpg)  
Figure 5. Real-robot settings. A Franka Research 3 with external and wrist-mounted cameras. (a) Covered-ingredient retrieval before and after lid opening; (b) sandwich preparation; (c) activity-dependent packing; (d) safe execution; (e) failure recovery.

Covered ingredients: retrieving and correcting scene memory. This experiment tests whether scene memory guides the search for hidden targets and is revised when observations contradict remembered locations. Lettuce, toast, and ham are hidden beneath colored lids, and the robot is asked to transfer a named ingredient to a plate. Write and retrieve provides its location before a restart, followed by a request without a location hint. Search without memory requires successive lid opening and observation before finding the target. Revise stale memory swaps ingredients after memory formation to test if memory can be revised. Dispatch accuracies are 100% (12/12), 80.0% (8/10), and 75.0% (9/12), respectively. Remembered locations support retrieval, while fresh observations can correct the stale context used for planning. Recorded failure cases include misidentification and omitted subtasks.

Multimodal requests and memory revision across scenes. We test whether memory enables efficient scene selection for named and visually specified personal objects across four endpoints in RoboLab [38]: garage, dining table, kitchen, and study desk. The planner must select where to act for a target specified by name or photograph, allowing users to refer to personal objects without a precise verbal description. With no initial scene view, the planner requests endpoint images or consults stored scene memory. Each text/image condition contains 88 requests. Image queries never name the object: without memory, the photograph accompanies the instruction go pick up this; with memory, go pick up mine refers to a photograph supplied in an earlier conversation as this is mine. Memory raises correct selections in both query modes and reduces latency (Table 2a). These configurations support both named and remembered visual references while reducing the search required to select a scene.

Table 2. RoboLab planning without physical execution. (a) Static scenes: 88 requests per condition; latency is the median time to dispatch. (b) Relocated targets: 88 request pairs per query type. Memory corrected requires both writing the new location and removing the stale association.

(a) Query support and efficiency
<table><tr><td>Query</td><td>Correct /88</td><td>Time (s)</td></tr><tr><td>Text</td><td>77</td><td>28.6</td></tr><tr><td>Image</td><td>77</td><td>34.6</td></tr><tr><td>Text + memory</td><td>88</td><td>7.0</td></tr><tr><td>Image + memory</td><td>88</td><td>7.1</td></tr></table>

(b) Closed-loop scene-memory revision
<table><tr><td>Query</td><td>Memory corrected</td><td>Follow-up dispatch</td></tr><tr><td>Text</td><td>76/88</td><td>83/ 88</td></tr><tr><td>Image</td><td>77/88</td><td>85 / 88</td></tr></table>

We additionally conduct a revision experiment to test whether new observations correct persistent scene records and whether later requests can use the revised memory for endpoint selection. We relocate targets while keeping their old locations in memory. A valid memory correction must write the new location and remove the stale association. Corrections succeed in 76/88 text and 77/88 image trials; follow-up dispatch is correct in 83/88 and 85/88, respectively (Table 2b). The two measures support observation-driven revision and later reuse of scene knowledge. Analysis. The above studies evaluate the lifecycle of personalized context: acquiring requirements, retaining them across interactions, and revising them after change. The results support connecting memory to planning and observation, rather than treating stored profiles and locations as permanently valid instructions.

## 4.2 Safety during Execution

Geometric checks beyond safety fine-tuning. We test whether runtime shielding reduces unsafe contact beyond improvements from training with safe demonstrations. LIBERO-Safety [36] provides tabletop spatial avoidance (TSA) and free-space hand-object avoidance (FSHOA), with 450 episodes per suite. We evaluate the original and safety-finetuned (SFT) $\pi _ { 0 . 5 }$ , with and without our shield, alongside adapted VLSA [16], and extend the comparison to OpenVLA-OFT [39] and VLAct [40] to test transfer across policy backbones (Table 3). The official SFT $\pi _ { 0 . 5 }$ checkpoint was released by [36] and was trained on 19,664 safe demonstrations for collision-free execution. We report violation-free task success rate (VF) and violation rate (VIOL.).

Table 3. Physical safety by suite and difficulty on LIBERO-Safety. Each entry is VF ↑ / Viol. ↓ (%): violation-free task success and episodes with at least one constraint violation. Each suite–level cell contains 150 episodes.
<table><tr><td rowspan="2">Method</td><td colspan="3">TSA</td><td colspan="3">FSHOA</td><td rowspan="2">Overall All</td></tr><tr><td>L0</td><td>L1</td><td>L2</td><td>L0</td><td>L1</td><td>L2</td></tr><tr><td>VLAct [40]</td><td>0.0 / 93.3</td><td>20.7 / 30.7</td><td>0.0 / 85.3</td><td>0.0 / 76.0</td><td>5.3 / 47.3</td><td>0.0/ 83.3</td><td>4.3 / 69.3</td></tr><tr><td> $\mathrm { V L A c t } + \mathrm { V L S A }$ </td><td>17.3 / 66.7</td><td>20.0 / 20.0</td><td>10.7 / 58.7</td><td>1.3 / 52.0</td><td>15.3 / 24.0</td><td>13.3 / 47.3</td><td>13.0 / 44.8</td></tr><tr><td>VLAct + Ours</td><td>20.0/25.3</td><td>20.0 / 24.7</td><td>20.0 / 28.0</td><td>11.3 / 26.0</td><td>10.7 / 15.3</td><td>1.3 / 37.3</td><td>13.9 / 26.1</td></tr><tr><td>OpenVLA-OFT [39]</td><td>0.0/67.3</td><td>20.7 / 38.7</td><td>6.0/ 61.3</td><td>0.0/ 78.7</td><td>14.7 / 52.0</td><td>0.0/ 70.7</td><td>6.9 / 61.4</td></tr><tr><td>OpenVLA-OFT + VLSA</td><td>12.7 / 47.3</td><td>19.3 / 22.7</td><td>6.7 / 45.3</td><td>0.7 / 46.7</td><td>18.0 / 46.0</td><td>16.7 / 44.0</td><td>12.3 / 42.0</td></tr><tr><td>OpenVLA-OFT + Ours</td><td>20.0 / 16.0</td><td>19.3 / 8.0</td><td>19.3 / 12.0</td><td>7.3 / 20.7</td><td>18.0/ 10.0</td><td>4.7 / 36.7</td><td>14.8 / 17.2</td></tr><tr><td>π0.5 [41]</td><td>0.0 / 84.7</td><td>26.7 / 34.7</td><td>0.0 / 84.0</td><td>1.3 / 72.0</td><td>12.0 / 66.0</td><td>0.0/85.3</td><td>6.7 / 71.1</td></tr><tr><td> $\pi _ { 0 . 5 } + \mathrm { V L S A }$ </td><td>8.0 / 66.7</td><td>26.7 / 10.7</td><td>5.3 / 65.3</td><td>2.7 / 52.0</td><td>18.7 / 45.3</td><td>14.7 / 58.7</td><td>12.7 / 49.8</td></tr><tr><td> $\pi _ { 0 . 5 } + \mathrm { O u r s }$ </td><td>18.7 / 18.7</td><td>24.0 / 17.3</td><td>20.0 / 16.0</td><td>10.7 / 27.3</td><td>12.0/ 18.0</td><td>2.0/ 38.7</td><td>14.6 / 22.7</td></tr><tr><td> $\pi _ { 0 . 5 } – \mathrm { S F T } \ [ 3 6 ]$ </td><td>17.3 /69.3</td><td>40.7 / 23.3</td><td>24.0/ 61.3</td><td>20.0 / 76.0</td><td>35.3 / 4.7</td><td></td><td>25.3 / 52.6</td></tr><tr><td> $\pi _ { 0 . 5 } – \mathrm { S F T } + \mathrm { V L S A }$ </td><td>15.3 / 56.7</td><td>38.7 / 16.0</td><td>19.3 / 42.0</td><td>19.3 / 51.3</td><td>30.7 / 6.0</td><td>14.7 / 80.7 29.3 / 54.0</td><td></td></tr><tr><td> $\pi _ { 0 . 5 } – \mathrm { S F T + O u r s }$ </td><td>20.0/ 13.3</td><td>40.0 /6.0</td><td>23.3 / 4.7</td><td>18.7 / 4.7</td><td>35.3 / 4.7</td><td>16.0/ 3.3</td><td>25.4 /37.7 25.6 / 6.1</td></tr></table>

Despite safety fine-tuning, 52.6% of episodes incur violations. Shielding reduces this to 6.1%, compared with 37.7% for adapted VLSA, while VF remains similar at 25.3% without shielding and 25.6% with it. For the original checkpoint, violations decrease from 71.1% to 22.7%, compared with 49.8% for adapted VLSA, while VF increases from 6.7% to 14.6%. The same shield also reduces violations with OpenVLA-OFT, from 42.0% under VLSA to 17.2%, while increasing VF from 12.3% to 14.8%. On VLAct, our shield reduces violations from 44.8% under VLSA to 26.1%, with a smaller VF increase from 13.0% to 13.9%, supporting cross-backbone violation reduction.

For deployment cost, measured mean per-invocation times are 21.06 ms for geometric filtering, 47.82 ms for VLA inference, and 170.82 ms for perception. Perception dominates these measurements, while filtering contributes a smaller computational component.

Real-world safe execution. The physical study tests whether the proactive shield reduces contact during real-robot execution when an obstacle is introduced. The real-robot policy is trained on carrot relocation without obstacles. With an inserted obstacle, contact-free completion increases from 0/20 for the raw policy to 20/20 with shielding (Table 1), showing repeatable intervention against collision hazards.

![](images/9ce4ab0b3e524f1c0bf1441315afc63cc4420ac4f3536b3e6dc90c3b4d79296e.jpg)  
Figure 6. Integrated food-serving runs. (a)–(d) Memory-guided lid opening and shielded transport. (e)–(h) Recovery restores a lost grasp before task completion. (i)–(l) Revealing meat triggers a scene-memory update and refusal under the vegetarian preference.

## 4.3 Recovery and Improvement from Failure Experience

Recovery from perturbation This experiment tests whether the recovery skill restores task progress after disturbance and whether using successively refined skills in reflection-and-retry rounds increases cumulative completion. The disturbance-response results (Fig. 3) motivate control-based recovery when perturbations undermine VLA execution.

The experiments are conducted on the LIBERO-Object task suite with a 12 cm positive-x hold-loss perturbation. R0 uses the shared base recovery skill with empty strategy (S) and parameter (P) memories. After each round, reflection uses up to three failed episodes with recovery evidence per object. Baselines and our reflection process use Qwen3-VL-235B-A22B-Instruct. Harness VLA uses the same Qwen3-VL-4B-OFT VLA checkpoints as our system, while CaP-X does not require a VLA.

Table 4. Recovery on 200 LIBERO episodes with perturbation. R1/R2 include additional reflection and retries. API tokens accumulate execution and reflection calls per original episode.
<table><tr><td>System / Stage</td><td>Cumulative API tokens / 200 (%)</td><td>per episode</td></tr><tr><td>Qwen3-VL-OFT CaP-X [42] Harness VLA [11]</td><td>26 (13.0) 10 (5.0)</td><td>1,124,819 3,829,605</td></tr><tr><td>CarryOn R0: base skill</td><td>101 (50.5) 58 (29.0)</td><td>21,371</td></tr><tr><td>CarryOn R1: one reflection</td><td>141 (70.5)</td><td>34,393</td></tr><tr><td>CarryOn R2: two reflections</td><td>160 (80.0)</td><td>41,109</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 5. Recovery on 770 LIBERO-Recovery Object cases. R1/R2 include additional reflection and retries. API tokens accumulate execution and reflection calls per original episode.
<table><tr><td>System / Stage</td><td>Cumulative API tokens / 770 (%) per episode</td></tr><tr><td></td><td>0</td></tr><tr><td>Pure VLA CarryOn R0: base skill</td><td>247 (32.08) 385 (50.00)</td></tr><tr><td></td><td>2,674 5,492</td></tr><tr><td>CarryOn R1: one reflection CarryOn R2: two reflections</td><td>569 (73.90) 586 (76.10) 7,159</td></tr></table>

With base recovery, R0 completes 58/200 episodes (29.0%). R1 resolves 83/142 previous failures, raising cumulative completion to 70.5%; R2 resolves another 19/59, reaching 80.0% (Table 4). The second reflection-and-retry round extends recovery beyond the first revision, and the full iterative system achieves the highest completion rate.

To assess the VLM-query cost of recovery-focused refinement, we measure billed API tokens including both reflection rounds and all retries, amortized over the original episodes. Our recorded usage is 41,109 tokens per episode, including 2,609 for reflection, compared with the recorded baseline totals in Table 4. Higher cumulative completion together with lower recorded VLM-token usage supports recovery-focused refinement as a more efficient method.

Recovery from recorded failure states. We further evaluate recovery from natural execution-induced failure states using LIBERO-Recovery [43]. The evaluation covers 77 Object scenes with five initially closed-gripper and five initially open-gripper trials each, yielding 770 cases without additional runtime disturbances. Since failure states are supplied, we disable the Judge and invoke recovery after scene initialization. Pure VLA and R0 share the same Qwen3-VL-4B-OFT VLA checkpoint.

Base recovery raises completion from 32.08% to 50.00%, a gain of 17.92 percentage points under the same rollout budget. R1 resolves 184/385 previous failures, increasing cumulative completion to 73.90%; R2 resolves another 17/201, reaching 76.10% (Table 5). Final cumulative API usage is 7,158.77 tokens per original case, including reflection and retries. The above results support iterative recovery from recorded natural failure states.

Real-world recovery. This experiment tests whether base recovery and retained skill revisions improve real-robot task completion after object removal. An operator takes the lifted object and places it at one of five locations, with two orientations and two repetitions (20 episodes per condition). We compare the raw policy, base recovery, and a revised skill combining a −0.02 m grasp-height offset with up to three attempts and renewed perception. Task completion rises from 3/20 to 9/20 and 18/20, respectively (Table 1). External repair improves completion over the raw policy, and retained revisions improve it further across the tested placements.

Analysis. The execution studies support complementary interventions around a frozen VLA: shielding reduces contact violations, while recovery restores progress after disruption. Reflection-and-retry rounds extend cumulative completion, and revisions improve real-robot repair. Together, the two paths combine immediate protection with reusable failure recovery skills.

## 4.4 Integrated Assistance on the Real Robot

Setup. This study examines whether personalized planning, proactive shielding, and reactive recovery operate as one system in a long-horizon assistance task, where each component acts on what the others produce. In the food-serving scenario of Fig. 1, a user with a stored vegetarian preference asks the robot to serve lunch from a covered pan. The robot must look under the lid, decide whether the food may be served, and carry it past an obstacle while an operator disturbs the transport. A second trial places meat under the lid while scene memory still records a carrot. All real-robot experiments use a Qwen3-VL-4B-OFT policy [37]; the policy for this task was trained on lid opening and food pickup without obstacles.

Results. In the carrot run (Fig. 6a–h), the robot uncovers the pan guided by scene memory and starts the transfer. The shield avoids potential collisions with the rice cooker obstacle during transport. After the carrot is dropped into the pan as a perturbation, the VLA is not able to pick it up again. The Judge triggers the recovery procedure, which re-grasps the carrot and hands control back to the VLA to place it in the bowl. In the meat run (Fig. 6i–l), the robot uncovers the pan and finds a contradiction with scene memory, then replaces the stale carrot entry with the observed meat entry. The system then declines to serve anything as the food currently on the table does not meet the user’s dietary requirements The accompanying video shows both runs.

Analysis. These runs illustrate how CarryOn preserves the user’s intent while adapting both the plan and its execution. Scene memory guides where to look, while fresh visual evidence and the stored dietary preference determine whether serving should proceed. In the carrot run, the obstacle and lost grasp disrupt execution without invalidating the goal: shielding adjusts the transfer motion, and Judge-triggered recovery restores the grasp before returning control to the VLA to finish serving. In the meat run, the newly revealed food instead invalidates the planned transfer under the same dietary constraint. Visual feedback therefore changes the planning decision and corrects the persistent scene record, rather than merely checking whether a manipulation succeeded. The benefit of the dual-loop design is this distinction between repairing an interrupted, appropriate task and stopping a task that new evidence makes inappropriate. Together, the runs provide evidence that contextual decisions, action-level protection, and physical repair can support a common assistance goal around a frozen VLA.

## 5 Conclusion

We presented CarryOn, a dual-loop harness for personalized and reliable robot assistance. The planning loop maintains revisable user and scene memories to guide task decisions across interactions, query modalities, and execution endpoints. The execution loop combines proactive geometric shielding with reactive recovery, using offline reflection to refine recovery skills while keeping the underlying VLA policy frozen. Real-robot and simulation studies evaluate these complementary capabilities, from preference-conditioned assistance and scene-memory revision to runtime protection and recovery after disturbance. Together, they show that persistent context and physical feedback extend robot assistance beyond isolated task execution. Future work will evaluate longer-term adaptation and the transfer of recovery experience across tasks, while improving the balance between protective intervention and task completion.

## References

[1] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[2] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[3] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[4] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[5] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[6] Konstantinos Bousmalis, Giulia Vezzani, Dushyant Rao, Coline Devin, Alex X Lee, Maria Bauza, Todor Davchev, Yuxiang Zhou, Agrim Gupta, Akhil Raju, et al. Robocat: A self-improving foundation agent for robotic manipulation. arXiv preprint arXiv:2306.11706, 1(8), 2023.

[7] Lilian Weng. Harness engineering for self-improvement. lilianweng.github.io, July 2026.

[8] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

[9] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.

[10] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417, 2025.

[11] Yixian Zhang, Huanming Zhang, Feng Gao, Xiao Li, Zhihao Liu, Chunyang Zhu, Jiaxing Qiu, Yuchen Yan, Jiyuan Liu, Wenhao Tang, et al. Harness vla: Steering frozen vlas into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448, 2026.

[12] Jinbang Huang, Yuanzhao Hu, Zhiyuan Li, Ran Qi, Yixin Xiao, Zhanguang Zhang, Mark Coates, Tongtong Cao, and Yingxue Zhang. Roboharness: Memory-driven orchestration of heterogeneous robot policies for long-horizon planning. arXiv preprint arXiv:2607.18060, 2026.

[13] Xin Ding, Liang Mi, Mingzhe Huang, Zixuan Wang, Chao Zhang, Zixu Hao, Fu Chen, Xiangyu Li, Yikai Zheng, Yaoyu Guo, et al. Zetta ζ: An efficient closed-loop embodied harness for self-evolving physical intelligence. arXiv preprint arXiv:2608.16590, 2026.

[14] Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, and Thomas Funkhouser. Tidybot: Personalized robot assistance with large language models. Autonomous Robots, 47(8):1087–1102, 2023.

[15] Sangoh Lee, Sangwoo Mo, and Wook-Shin Han. Bring my cup! personalizing vision-language-action models with visual attentive prompting. arXiv preprint arXiv:2512.20014, 2025.

[16] Songqiao Hu, Zeyi Liu, Shuang Liu, Jun Cen, Zihan Meng, Shihefeng Wang, Xiang Li, and Xiao He. Vlsa: Vision-language-action models with plug-and-play safety constraint layer. arXiv preprint arXiv:2512.11891, 2025.

[17] Shivam Vats, Devesh K Jha, Maxim Likhachev, Oliver Kroemer, and Diego Romeres. Recoverychaining: Learning local recovery policies for robust manipulation. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 9776–9783. IEEE, 2025.

[18] Taeyoon Kwon, Dongwook Choi, Hyojun Kim, Sunghwan Kim, Seungjun Moon, Beong-woo Kwak, Kuan-Hao Huang, and Jinyoung Yeo. Embodied agents meet personalization: Investigating challenges and solutions through the lens of memory utilization. In International Conference on Learning Representations, volume 2026, pages 4790–4835, 2026.

[19] Jeongeun Lee, Chanyoung Park, and Dongha Lee. Personalizing embodied multimodal large language model agents over long-term user interactions. arXiv preprint arXiv:2605.26256, 2026.

[20] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International conference on robotics and automation (ICRA), pages 9493–9500. IEEE, 2023.

[21] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

[22] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. Progprompt: Generating situated robot task plans using large language models. arXiv preprint arXiv:2209.11302, 2022.

[23] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022.

[24] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.

[25] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

[26] Y Zhu et al. Vima: General robot manipulation with multimodal prompts. In International Conference on Learning Representations (ICLR), 2023.

[27] Allen Z Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, et al. Robots that ask for help: Uncertainty alignment for large language model planners. arXiv preprint arXiv:2307.01928, 2023.

[28] Borong Zhang, Yuhao Zhang, Jiaming Ji, Yingshan Lei, Juntao Dai, Yuanpei Chen, and Yaodong Yang. Safevla: Towards safety alignment of vision-language-action model via constrained learning. Advances in Neural Information Processing Systems, 38:153335–153373, 2026.

[29] Zijun Lin, Jiafei Duan, Haoquan Fang, Dieter Fox, Ranjay Krishna, Cheston Tan, and Bihan Wen. Failsafe: Reasoning and recovery from failures in vision-language-action models. arXiv preprint arXiv:2510.01642, 2025.

[30] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

[31] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642, 2024.

[32] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anand kumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

[33] Kensuke Nakamura, Lasse Peters, and Andrea Bajcsy. Generalizing safety beyond collision-avoidance via latent-space reachability analysis. arXiv preprint arXiv:2502.00935, 2025.

[34] Sankalp Agrawal, Junwon Seo, Kensuke Nakamura, Ran Tian, and Andrea Bajcsy. Anysafe: Adapting latent safety filters at runtime via safety constraint parameterization in the latent space. arXiv preprint arXiv:2509.19555, 2025.

[35] Enshen Zhou, Qi Su, Cheng Chi, Zhizheng Zhang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, and He Wang. Code-as-monitor: Constraint-aware visual programming for reactive and proactive robotic failure detection. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6919–6929. IEEE, 2025.

[36] Rongxu Cui, Zongzheng Zhang, Jingrui Pang, Haohan Chi, Jinbang Guo, Saining Zhang, Shaoxuan Xie, Xin Jin, Yao Mu, Jiaolong Yang, et al. Libero-safety: A comprehensive benchmark for physical and semantic safety in vision-language-action models. arXiv preprint arXiv:2606.23686, 2026.

[37] StarVLA Community. Starvla: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

[38] Jenai Xuning Yang, Rishit Dagli, Alex Zook, Hugo Hadfield, Ankit Goyal, Stan Birchfield, Fabio Ramos, and Jonathan Tremblay. Robolab: A high-fidelity simulation benchmark for analysis of task generalist policies. arXiv preprint arXiv:2604.09860, 2026.

[39] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[40] Senqiao Yang, Chengyao Wang, Yuxin Chen, Zixuan Wang, Longxiang Tang, Haokun Gui, Jinhui Ye, Changsheng Lu, Xiaoyang Wu, Mingkang Zhu, et al. Beyond data scaling: Representation-centric continued pre-training for vision-language-action models. arXiv preprint arXiv:2608.27550, 2026.

[41] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[42] Letian Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Dantong Niu, Fei-Fei Li, et al. Cap-x: A framework for benchmarking and improving coding agents for robot manipulation. arXiv preprint arXiv:2603.22435, 2026.

[43] Lin Liu, Zhicheng Bao, Lu Zhang, Ziying Song, Wu Yang, Shuai Tao, Wulong Liu, and Huchuan Lu. Libero-recover: Beyond task success towards failure recovery in robotic manipulation models. arXiv preprint arXiv:2609.05178, 2026.

# Appendix

## A System Implementation and Runtime Coordination

This appendix specifies how control passes between the planner, the VLA, the Judge, the shield, and recovery while a request is handled (Fig. 2). Each planner output is a plan: a goal and a set of subtasks, each of which sends an instruction to an endpoint. Invalid plans, including those that omit a required update\_scene (Appendix B.3), are returned to the planner for repair. Algorithm 1 summarizes one request. A subtask without dependencies starts immediately, and a subtask that depends on earlier subtasks starts in sequence.

Algorithm 1 Handling one request in the dual-loop harness   
Require: request c; user memory M<sup>u</sup>; scene memory M<sup>s</sup>; endpoints E; skill library L   
1: x ← BUILDCONTEXT(c, M<sup>u</sup>, M<sup>s</sup>, E)   
2: P ← PLANNER(x), repaired until valid   
3: for all subtask ℓ ∈ P whose dependencies succeeded do   
4: if ℓ<sub>i</sub> writes memory then   
5: write the record to M<sup>s</sup> or M<sup>u</sup> ▷ update\_scene, update\_user\_memory   
6: else if ℓ calls a non-physical skill then   
7: run the skill; append its result to x ▷ e.g., web query, scene check   
8: else ▷ physical skill executed by the VLA: manipulation or observation step   
9: dispatch ℓ to the VLA; start the Judge   
10: while the job is running do   
11: execute SHIELD(π<sub>θ</sub>(o<sub>t</sub>, ℓ<sub>i</sub>)) ▷ Section 3.3.1   
12: if Judge reports success then mark ℓ<sub>i</sub> succeeded; append what it revealed to x   
13: else if Judge reportsfailed then   
14: suspend the VLA; run the active recovery skill ▷ Section 3.3.2   
15: resume the VLA on the original ℓ<sub>i</sub>; restart the Judge   
16: end if   
17: end while   
18: end if   
19: if ℓ revealed new scene contents then replan; the next plan must record them with update\_scene   
20: end if   
21: end for   
22: on termination: M<sup>u</sup> ← SUMMARIZE(conversation, M<sup>u</sup>)

## B Persistent User and Scene Memory

This appendix gives the storage format of both memories, their update paths, and logged examples showing how a revision changes the next plan

## B.1 Storage and Planner Injection

Both memories are Markdown files of ## <section> blocks with - key: value lines (Table 6). Both are rendered into the planner context on every request. All user files are rendered under per-user headings, and the planner prompt instructs it to apply only the requesting user’s facts.

## B.2 User Memory from Dialogue

User memory has two write paths. During a turn, the planner can record a preference the user states explicitly. In addition, after each plan finishes, a background summarizer reviews the recent conversation and writes durable facts

Table 6. Memory schemas.
<table><tr><td></td><td>User memory Mª</td><td>Scene memory M⁸</td></tr><tr><td>Unit Content</td><td>One file per user; one section per category Facts in profile : * (identity, habits, preferences),</td><td>One file; one section per place, named by the planner - objects: and- description: lines</td></tr><tr><td></td><td>profile:avoid: * (standing prohibitions), entity : * (owned objects, including reference_image)</td><td></td></tr><tr><td>Example</td><td>## profile:eating_habit/- diet: vegetarian; does not eat meat</td><td>## frying_pan_under_pink_lid/- objects: carrot</td></tr><tr><td>Evidence</td><td>Dialogue</td><td>Observations</td></tr><tr><td>Writers Updates</td><td>Summarizer; planner via update_user_memory Latest value per key; retraction by delete</td><td>Planner via update_scene only Section replaced; a stale entry is retracted by rewriting</td></tr></table>

that are not yet stored. Both follow the same rule for separating durable facts from temporary statements. A new value for a key replaces the old one, and a superseded preference is retracted.

Durability rule in the summarizer prompt (excerpt, real-robot development branch)   
DURABILITY FILTER (the most important rule):   
Only write facts that will still be true in future sessions. Evidence must   
be a lasting preference/attribute explicitly stated, or clearly meant as a   
standing instruction ("always cook 2 eggs from now on").   
- Do NOT write one-off / session-scoped wishes ("I don't want meat today",   
"no salt this time") or volatile data (weather, time, robot/task state).   
- Negatives use the same filter: profile:avoid:<sub>\*</sub> only for a STANDING prohibition,   
never one-off negatives.

For example, when Bob says “I take my charger with me everywhere, even to the gym. Also, I need to leave by five today.”, the profile stores profile:habit/charger and nothing about leaving at five.

We show an example below to demonstrate how preference change is reflected in the user profile. In the sandwich experiment, the same request yields different plans before and after the stored diet changes.

Sandwich demonstration: plans under preference change   
# Profile: vegetarian   
CarryOn> Make me a sandwich.   
[plan] I'll make you a vegetarian sandwich with toast and lettuce.   
- t1: toast / t2: lettuce / t3: toast   
# Profile after update: diet: non-vegetarian; prefers ham and lettuce in sandwiches   
CarryOn> Make me a sandwich.   
[plan] I'll make you a sandwich with ham and lettuce using the available ingredients.   
- t1: toast / t2: ham / t3: lettuce / t4: toast

## B.3 Scene Memory from Observation

An observation that reveals new contents creates an outstanding record, and the validator rejects the next plan unless it contains an update\_scene subtask. The RoboLab example below shows a stale location being revised and reused. Memory lists a Rubik’s cube at the garage bench, but the cube is on the dining table. The first request checks the remembered station, rewrites it, finds the cube elsewhere, and records the new location; the follow-up dispatches directly.

RoboLab relocated-target pair (text query)   
# Scene memory before: ## franka\_garage\_bench / - objects: rubiks cube   
[plan] I'll check the garage bench for the Rubik's cube first.   
- t1 @ check\_endpoint -> {"present": false, ...}   
[plan] I'll update the record for the garage bench and then check the dining table.   
- t1 @ update\_scene: Record that the garage bench currently holds no Rubik's cube.   
- t2 @ check\_endpoint -> {"present": true, "confidence": 0.95, "scene\_update\_required": true,   
"revealed\_place": "franka\_dining\_table", ...}   
[plan] I'll record that the Rubik's cube is on the dining table and then pick it up.   
- t1 @ update\_scene: Record the presence of the Rubik's cube at the dining table.   
- t2 @ franka\_dining\_table: Pick up the Rubik's cube from the dining table.   
# Scene memory after: ## franka\_dining\_table / - objects: rubiks cube   
# ## franka\_garage\_bench / - objects:   
# Follow-up request (new process, memory carried over):   
[plan] I will pick up the Rubik's cube from the dining table.

## B.4 Remembered Reference Images

A reference image is stored as an entity attribute: the file is copied into the memory store, and a reference\_image fact pointing to it is written under entity:<name>, either by the summarizer when the model tags an entity fact with an attached image or by an explicit /remember\_image <path> <entity> command. A later request can then refer to the entity (“go pick up mine”), and the stored image is available to the planner through the user-memory block.

## C Proactive Shielding

The LIBERO-Safety scene builder assigns task roles to observed geometry before applying contact rules. It receives target and destination names from the instruction. Text-prompted SAM3 segmentation associates these names with image masks; the corresponding depth pixels are back-projected into the world frame. When text grounding returns no object mask, the scene builder can use a VLM point prompt followed by point-prompted segmentation. The self-mask removes robot and mount pixels. The target’s source support is obtained as a local geometric patch beneath the grounded target.

The contact rule exempts intended target, source-support, and destination contact for the bodies to which those permissions apply. Remaining scene obstacles are checked geometrically. Gripper geometry is propagated from each predicted end-effector pose. A holding-state latch uses measured gripper width and the executed gripper command to determine when carried-object geometry should be included. The inferred held-object point set is transformed using its attachment pose, so carried-object contact is checked in addition to gripper contact. If the first candidate action is blocked, the controller executes a three-control-step retreat with an away-from-obstacle component and an upward component. The retreat direction uses the obstacle point nearest the current end-effector reference point, and the filtering distance uses the surfaces of the predicted bodies, as distinguished in Fig. 7.

Table 7 gives the parameters of the response model and decision rule on LIBERO-Safety. Figure 7 illustrates action filtering and retreat.

Table 7. Shield parameters on LIBERO-Safety.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Controller</td><td>Operational-space pose control, kp = 150, damping ratio 1, control period 0.05 s; ωn = √150 ≈ 12.25 s−1</td></tr><tr><td>Prediction horizon H</td><td>5 actions</td></tr><tr><td>Intervention threshold δ</td><td>6.67 cm (1.51 cm margin + largest displacement over 5 steps)</td></tr></table>

## D Recovery Skills and Reflection

This appendix describes additional details of the recovery and reflection mechanism.

![](images/83abab88732310a729ba3a7d2d90749f719f163277f2e4069b9d75364b88f590.jpg)

![](images/ed06d37f057dc0498e9386f511714fa672b2459198eeae2cd380a64bf8c0c183.jpg)  
Figure 7. Geometric action filtering and retreat. Each example shows a geometric side view (left), the observed scene (upper right), and predicted gripper–obstacle distances (lower right); gray spheres depict current robot geometry and purple outlines depict predicted gripper geometry. (a) Actions before the first violation of the dashed distance threshold are released; (b) if the first action is blocked, the robot retreats and rechecks. Open triangles mark obstacle points nearest the predicted gripper surface for filtering, whereas the filled triangle n is nearest the current end-effector reference point and determines retreat. The two thin dashed arrows in (b), arranged head-to-tail, show the component away from n and the added upward bias; the thick solid arrow shows their resulting retreat direction.

## D.1 Base Recovery Skill

The base skill attempts to restore the physical state needed to continue the interrupted subtask (Algorithm 2). After suspending the VLA, it moves the arm to an observation configuration, locates the target, and generates grasp candidates for retrieval. A check of measured gripper width after lifting determines whether to proceed with transport. If this check passes, the skill locates the destination, moves to a stand-off pose near it, and checks holding again. Passing the second check allows the VLA to resume the original instruction from that pose.

Recovery fails if the skill cannot locate the target or destination, its grasp attempts do not establish a hold, a required motion cannot be completed, or the second holding check does not pass. In these cases, the skill stops and the runtime attempts to return the arm to the available grasp candidate’s pre-grasp pose, or to the recorded pre-failure pose if no candidate exists. It opens the gripper after arrival, then requests VLA resumption with the original instruction and remaining episode budget. Resumption is still requested if the fallback motion fails. The Judge continues to assess task completion; a successful recovery only enables the task to continue.

## D.2 Applying a Stored Recovery Version

Each recovery version preserves the recovery procedure and the strategy (S) and parameter (P) revisions from offline reflection for use in later attempts. S changes the sequence or conditions of recovery operations, while P adjusts their numerical settings, such as grasp height. Recovery uses the revisions stored in the current library version, with the base skill supplying operations or settings that have not yet been revised. Between rounds, reflection builds on this version using the remaining failures and saves further revisions for subsequent attempts.

The strategy revisions address different failure sources in simulation and on the real robot. In simulation, we found target grounding to be a recovery bottleneck in scenes with visually unrealistic object appearances. The evaluated S revision therefore enables VLM-assisted re-grounding only for objects that are hard to ground and lead to recovery failure. During recovery, a VLM receives the target name and reference image together with a fresh scene image showing numbered candidate masks, and selects the mask used for grasp generation. Images from earlier failed attempts inform the offline reflector’s decision to enable this step. On the real robot, S also adds retries with renewed perception, allowing another grasp attempt to use an updated target location.

Algorithm 2 Base recovery skill (hold loss)   
Require: Judgefailed verdict for the current subtask   
1: suspend the VLA; move to the initial configuration; open the gripper   
2: ground the task target   
3: build grasp candidates for the target ▷ short axis, then long axis   
4: for each candidate do   
5: execute the grasp; lift and check holding from measured width   
6: if held then break   
7: end if   
8: end for   
9: if held then   
10: ground the destination; move to a pre-place stand-off; re-check holding   
11: end if   
12: if recovery failed, including the second holding check then   
13: attempt to return to the candidate’s pre-grasp pose, or to the recorded pre-failure pose if no candidate exists   
14: open the gripper only after reaching the fallback pose   
15: end if   
16: request VLA resumption with the original instruction and remaining budget ▷ Judge assesses task completion

## D.3 A Complete Revision Chain

We provide a BBQ sauce example across reflection rounds in the LIBERO perturbation study. Both memories start empty. After each round, the reflector reads up to three failed recoveries of the object and decides whether to write an S entry and whether to update the P entry. The next round reruns the failed episodes with the new memory version.

• R0 (empty memory): 3/20. In the three failures selected for M1, all six grasp attempts were planned at a contact height of 66.0 mm and all descents jammed at 109.0–113.2 mm.

• M1. S write: “Algorithm-selected crop shows tomato ketchup bottle, not BBQ sauce reference”, obtained by the reflector’s visual judgment. P update to 110.0 mm: “raising z to 110.0 mm tests a shallower approach.”

• R1: 3/17. Descents now reached the planned height instead of jamming, but the object often slipped during the lift.

• M2. S skip (“crops match reference BBQ sauce bottle”); P update to 105.0 mm (“a testable hypothesis to improve grasp stability”).

• R2: 11/14. Cumulative completion for the task rises from 3 to 17 of 20.

## E Experimental Configuration

Table 8 lists the models used for each role.

Table 8. Models per role. Language and vision-language models are called through an API; the VLA runs locally.
<table><tr><td>Role</td><td>LIBERO perturbation / LIBERO-Recovery</td><td>Real robot / RoboLab</td></tr><tr><td>Planner</td><td>Qwen3-VL-32B-Instruct</td><td>Qwen3-VL-32B-Instruct</td></tr><tr><td>Visual Judge</td><td>Qwen3-VL-32B-Instruct</td><td>Qwen3-VL-32B-Instruct</td></tr><tr><td>Reflectors</td><td>Qwen3-VL-235B-A22B-Instruct</td><td>Qwen3-VL-235B-A22B-Instruct</td></tr><tr><td>VLA</td><td>StarVLA Qwen3-VL-OFT (LIBERO checkpoint)</td><td>StarVLA Qwen3-VL-OFT (task-specific fine-tunes)</td></tr></table>

## F Agent Prompts

## F.1 Agent Inputs and Outputs

Table 9 summarizes what each language-model module receives and returns.

Table 9. Inputs and outputs of the language-model modules.
<table><tr><td>Module</td><td>Input</td><td>Output</td></tr><tr><td colspan="3">Planning loop</td></tr><tr><td>Planner</td><td>User request, user and scene memory, camera images, A plan (goal, reply to the user, subtasks with results of earlier subtasks</td><td>endpoint, instruction, and dependencies), including memory updates</td></tr><tr><td>Background summarizer</td><td>Dialogue of a finished plan, current user memory</td><td>Durable user facts (Appendix B.2)</td></tr><tr><td colspan="3">Execution loop</td></tr><tr><td>Visual Judge</td><td>Current subtask, camera images, earlier observations Keep waiting, advance, or invoke recovery; what an</td><td>observation step revealed</td></tr><tr><td>S reflector</td><td>Reference image, scene image, selected mask and crop of each failed recovery</td><td>Enable re-grounding for the object, or skip</td></tr><tr><td>P reflector</td><td>height, gripper-width trace</td><td>Same, plus grasp-stage frames, planned and reached New grasp height for the object, or no update</td></tr></table>

## F.2 Agent Prompts

This section lists the prompts of the core modules. The planner and Judge prompts are instantiated from the real-robot studies, and the reflector prompts from the LIBERO perturbation study; the summarizer’s durability rule is excerpted in Appendix B.2. Other studies use the same prompts except for the available endpoints and benchmark-specific inputs. Runtime content, such as memories, endpoint descriptions, and images, is replaced by «SLOT: ...» placeholders. [SYSTEM MESSAGE] and [USER MESSAGE] separate the messages of a request.

Planner   
[SYSTEM MESSAGE]   
You are CarryOn, a high-level robot dispatch agent.   
You will be given:   
(a) a user instruction, AND/OR   
(b) one or more "events" describing prior subtask outcomes -- success, failures, or timeouts.   
When the configured planner provider is a VLM, you may also receive current   
camera/image inputs out-of-band in the model request. Use those visual inputs   
as scene context when choosing endpoints, ordering subtasks, and writing   
instructions.   
Each event may or may not require action. Your job is to decide, given the   
overall goal so far, what (if anything) CarryOn should dispatch NEXT.   
You only know an endpoint by its natural-language description and optional   
skill\_library. The VLA behind each endpoint is a black box -- do NOT invent   
structured action parameters or motor commands.   
==========   
OUTPUT FORMAT (this is REQUIRED -- the parser depends on it)   
====   
You MUST first reason out loud, THEN emit a single JSON document.   
1. Wrap your reasoning in a <reasoning>...</reasoning> block. In it, walk   
through:   
- what the user / event is asking for,   
- which endpoints look relevant and why,   
- the order/dependencies you intend,   
- any uncertainty or fallback you want to flag.   
This block is for your own thinking; CarryOn logs it but does not   
parse it.   
2. After </reasoning>, emit ONE JSON object matching the schema below.   
No markdown fences, no extra prose after it.   
JSON schema:   
{   
"goal": "<string -- short restatement of what THIS plan tries to achieve>",

```snap
"user_reply": "<string -- brief 1-2 sentence message shown DIRECTLY to the user>",
"subtasks": [
"id": "<string, e.g. t1>",
"endpoint": "<endpoint name from the provided list>",
"instruction": "<REQUIRED non-empty string; exact skill_library entry when one exists>",
"objective": "<completion criterion for the visual judge; see rules>",
"purpose": "observe | act -- why this step is in the plan; see rules",
"inputs": { }, // optional extra context, object
"depends_on": ["<id>"] // optional, defaults to []
Rules:
Always set "user_reply": a short, natural acknowledgment or status line the
user will read in the terminal (what you understood / what you will do).
Keep it to 1-2 sentences. Do NOT put JSON, endpoint names, or subtask ids
there. Match the user's language when practical.
When the user states a DURABLE preference, habit, identity fact, or owned
object attribute and is NOT asking to manipulate the world right now:
dispatch `update_user_memory` (follow that endpoint's description for
category/key/value, VALUE QUALITY, profile:avoid:<sub>*</sub>, inputs.facts, and
delete/retract). Acknowledge briefly. One-off / session-scoped wishes stay
in short-term context only -- never write them to memory. If the new fact
SUPERSEDES stored profile entries, retract the obsolete keys in the same
call.
IDENTITY ATTRIBUTION (several people share this robot): a speaker who says
who they are <<SLOT: example>> is the ACTIVE USER until someone else does.
<sub>*</sub> every durable fact you persist from them MUST carry their id: put
"user_id" inside each element of inputs.facts, e.g. <<SLOT: example fact with user_id, category,
key, and value>>;
<sub>*</sub> spell it as the given name in lowercase; if a `# user: <id>` section for
that person exists, reuse that exact id -- two spellings split one person
into two profiles;
<sub>*</sub> use ONLY that user's facts when planning. Never apply another user's
facts, even though every user's section is visible: two people can hold
opposite habits, and using the wrong one is a wrong plan;
<sub>*</sub> with nobody identified, act for the default user.
NO UNSOLICITED MANIPULATION (mandatory): dispatch arm /
search ONLY when THIS turn explicitly asks for a physical action now
(<<SLOT: example>>, "put X on Y"). Do NOT invent
robot busywork because:
<sub>*</sub> <<SLOT: example>> / objects are visible in the camera, OR
<sub>*</sub> the user stated a standing preference about <<SLOT: example domain>>
(<<SLOT: example>>, "from now on...", "always include..."), OR
<sub>*</sub> small-talk is mixed into the same message (phone battery, weather,
traffic, etc.).
A preference-only or chitchat turn is never an assembly plan.
If nothing durable needs storing and no robot action is requested (small-talk,
informational events, goal already done, waiting), emit an EMPTY "subtasks"
array. For reactions, leave user_reply empty unless you have something brief
and useful to tell the user.
Use ONLY endpoint names from the provided list.
EVERY subtask on EVERY endpoint MUST have a non-empty "instruction" string.
An endpoint without skill_library accepts a concise free-form instruction;
"no skill_library" NEVER means instruction may be empty. For memory/local
endpoints, state the bookkeeping operation briefly (for example,
"Record the stated scene fact") while putting its structured data in inputs.
Each subtask.id must be unique; depends_on must reference ids from THIS plan.
The DAG must be acyclic.
Do NOT include fields other than
id/endpoint/instruction/objective/purpose/inputs/depends_on inside each
subtask.
"instruction" and "objective" are TWO SEPARATE AUDIENCES. Never merge them.
<sub>*</sub> "instruction" is dispatched VERBATIM to the VLA. When the endpoint has a
skill_library, instruction MUST be an exact library entry (see next rule).
Do NOT append completion criteria, comparisons to earlier subtasks, or
judging hints -- learned policies degrade on untrained phrasings.
"objective" is read ONLY by the visual judge. One sentence naming the NEW
visible end state that means THIS subtask is done (not a state already
true before it started). Required for VLA / arm endpoints; optional for
non-visual ones (memory, queries, waits). Reuse the SAME object noun as
instruction / skill_library. The judge prefers objective over instruction when
they disagree, so a synonym here will make it look for the wrong object.
SKILL LIBRARY SELECTION (mandatory when the endpoint lists skill_library):
instruction MUST be copied VERBATIM from ONE listed entry. The ONLY allowed
```

edit is filling declared {placeholders} (e.g. {object} -> cup). Do NOT invent object names, colors, destinations, or phrasings from the camera or user wording. Vision and memory choose WHICH library entry to dispatch; they must not rewrite the instruction text. Endpoints without skill\_library may use concise natural language matching that endpoint's description.

\- AMBIGUOUS VISUAL NAMES: skill\_library is a naming reference, not evidence that an object exists or has a particular identity. Lighting, shade and wear can make appearance or names ambiguous. Only when position, task role and observation continuity support the same physical object, prefer its corresponding library name and retain the visual description as an attribute. Apply this to judge/reveal reports too: do not create a second scene entity or conclude there is no applicable skill solely because a report used a descriptive synonym. Do not merge objects merely because names sound similar. Clear contradictory visual evidence takes precedence over naming expectations. When there are multiple plausible instances or identity remains uncertain, report that uncertainty rather than force a match. Newly revealed contents must be reported independently: <<SLOT: example>>. An absent object is still absent. Once the matching skill is selected, copy its instruction verbatim as required above; do not rewrite it to use the descriptive synonym.

\- IF NOTHING IN THE LIBRARY DOES WHAT YOU INTEND, SAY SO: emit EMPTY subtasks and name what is missing in user\_reply. Never dispatch a different entry because it "still advances the goal" -- a different entry is a different action, and the operator reads your reply believing it describes what the robot did.

\- SAY WHAT YOU DISPATCH: user\_reply and the subtasks must describe the SAME act. If the reply announces one thing while the instruction does another, the plan is wrong however sound each half looks alone.

- For VLA / learned-policy endpoints, each subtask is ONE execution of ONE skill\_library entry. Do NOT batch repeats into plurals ("two objects"); emit chained single-object subtasks at the skill's natural granularity.   
- RESOURCE SERIALIZATION (mandatory): RESOURCE SERIALIZATION (mandator ):

\- A physical arm endpoint is single-slot; the executor FIFO serializes concurrent same-arm jobs -- do NOT invent depends\_on just for arm contention. Use depends\_on ONLY for a true prerequisite: the later subtask needs the earlier one to succeed (explicit "then"/"after that", stacked assembly, or a \${tN.result...} placeholder). Upstream failure cancels dependents. Independent same-arm picks (e.g. <<SLOT: example>> -- either is useful alone) MUST be separate roots with empty depends\_on so one failure does not cancel the others; the executor runs them one-at-a-time in plan order. - Independent subtasks on different endpoints may remain parallel. - NEVER substitute the target object. Copy the instruction's object noun verbatim even if it is missing from skill\_library or the camera, or looks harder than an alternative. Wrong-object execution cannot be undone downstream; not-found fails fast and can be corrected.

\- CANONICALIZE applies only when filling a skill\_library {placeholder} or when the endpoint has no skill\_library: once the target kind is unambiguous, drop identification-only qualifiers (extra color/size/ownership/position hints) that are not part of the chosen library entry. Never strip words that appear in the selected skill\_library string itself: if the entry names a colour, size or side, that word is part of the instruction and stays.

\- <<SLOT: deployment-specific rule for the workstation camera>>

\- NEVER dispatch a manipulation on an object that is NOT visible in the live image and whose whereabouts SCENE MEMORY does not record. \`skill\_library lists what the arm CAN do, never what is present right now: an exactly matching entry is NOT evidence the object is there. Emit the subtask(s) that would make the object observable FIRST, then the manipulation once it is actually in view. When the target's

location is unknown among candidate places, the CURRENT response must emit exactly ONE subtask that makes ONE candidate place observable. A LATER planner call, after a fresh image, decides whether to inspect another place or manipulate the now-visible target. Dispatching the target manipulation before that observation "because the instruction is clear" is the error this rule exists to stop.

\- A user's statement about current world layout is SCENE MEMORY, not a durable user-profile fact: with no physical action requested, dispatch \`update\_scene for it.

\- COVER EVERY PHASE the user explicitly chained ("then", "after that", "next"): emit subtasks for ALL of them; the plan "goal" must restate the full end-to-end objective.

PLANNING RUNS IN ROUNDS: ONE PLANNER RESPONSE is ONE observation round. When your subtasks finish you are asked again, with a fresh camera frame and this turn's results. LABEL EVERY SUBTASK: set

"purpose" to "observe" when the point of the step is to find something out and what you do next depends on what it shows, or to "act" when the need for the step is already established. Two limits follow from the label and the PARSER ENFORCES BOTH -- a plan may hold AT MOST ONE "observe" subtask, and NOTHING may depend on an "observe" subtask, directly or down a chain. So when

you do not know which of several candidates holds what you are after, emit ONE observation and stop there: do not queue the other candidates, and do not queue the manipulation that would follow, not even with depends\_on. Those are next round's decision, taken from a frame that actually answers the question. This does NOT weaken COVER EVERY PHASE above: phases the USER chained are known in advance and still go in one plan, because none of them waits on an observation you have not made. Emit an EMPTY "subtasks" list once the goal is reached.   
- If \`web\_query\` is available and the user asks for current / external / weather / internet-backed info, dispatch it instead of a robot endpoint (see that endpoint's description for inputs.kind / location / query).   
- <<SLOT: task-specific rules, e.g. a weather check before packing on the real robot>>   
- Placeholders: \${t1.result.<path>}\$ may appear in instruction or inputs when referring to earlier subtasks IN THIS PLAN.

LIVE VISION AUTHORITY: historical actions may no longer describe the current scene; determine current object state and goal completion from the attached live camera image, not from prior results or past verdicts.

SCENE MEMORY (last established content per place, as markdown):   
<<SLOT: rendered scene memory markdown, one '## <place>' entry per place, e.g. '- objects: ...'; if   
empty, this header is replaced by the single line 'SCENE MEMORY: (empty)'>>   
How SCENE MEMORY works:   
- SCENE MEMORY is last-known reference context, not guaranteed current state. It records one entry per named place.   
- A "place" is any region whose contents cannot be read off the current view (enclosed, obscured, occluded, or out of frame). YOU choose its name from what identifies it in the view; the name is a memory key, not a configured constant, and there is no pre-declared list of places to copy from.   
- If the user STATES what is at a place, that settles WHAT TO RECORD: dispatch \`update\_scene\` and do not stage a trip whose only point is to confirm it. The record is settled; the world is not.

\- AN ENTRY LICENSES THE LOOK, NOT THE GRAB. It says what was true WHEN IT WAS WRITTEN, and the workspace can have changed since without anyone saying so. So an entry tells you WHERE to go and never what you will find there: a step you take because an entry says the target is there has purpose "observe" -- you have not seen the target, and what you actually find decides what comes next. Reaching for the target in the same plan is reaching for something you have only been told about.

- If the task needs the contents of a place you cannot observe, plan the ORDINARY subtasks that make it observable -- whatever the scene and the registered endpoints afford -- and then dispatch \`update\_scene\` with what was actually observed. There is no separate search endpoint; decompose it yourself. An existing entry does not excuse this: it told you where to look first, and what you then SEE is what gets recorded.   
- WHAT YOU JUST SAW IS PERISHABLE; THE RECORD IS NOT. A later turn sees only

the record, so an observation you never write down is lost, and an entry you have just disproved will send that turn back to the same place. Whenever what you find differs from what an entry says -- or concerns a place with no entry -- dispatch \`update\_scene\` for it IN THE SAME PLAN as the next physical step, never as a plan of its own. Record both halves of what you learned: the place that did not hold what was recorded, and the place that did.

- Record only what was established (told or observed). Never write a guess.   
- When an action makes a place observable, compare its contents with the fresh camera image. The fresh visual observation overrides scene memory. If the contents differ, dispatch \`update\_scene\` with the new contents. Never use scene memory to resolve uncertainty in the current image.

\- A step result carrying \`scene\_update\_required\` is an OUTSTANDING RECORD. That step saw a place and nothing has written it down; the description lives only in this turn's results, while the record is what the next turn reads. Your NEXT plan must carry an \`update\_scene\` for that place, saying what that step's \`revealed\_contents\` actually said -- not what you expected it to say, and not a summary that drops the part which surprised you. You are the only one who can do this; no other step will. THE PARSER ENFORCES THIS: while a record is outstanding, a plan without an \`update\_scene\` subtask is rejected. Send an \`update\_scene\` correction IN THE SAME PLAN as the next physical step, never as a plan of its own: a plan that only writes memory ends the turn, and the work the correction was meant to enable never happens.

Available endpoints:

<<SLOT: JSON array of registered endpoints (name, transport, description, tags, skill\_library)>>

[USER MESSAGE]

<<SLOT: recent dialogue and the user request, with the live camera frame attached>>

## Visual Judge

[SYSTEM MESSAGE]   
You are CarryOn VLM Judge, a visual progress supervisor for a robot manipulation task. You receive the currently executing task, endpoint metadata, job ids, timing context, previous observations, and camera inputs.

Your job is to decide whether the planner should keep waiting, advance to the next task, or invoke a correction module.

Return STRICT JSON only. Do not include markdown or prose.

The context may include original\_user\_request. It is the user's original input and is provided as read-only background reference. The Planner has already decomposed that request: instruction and subtask\_objective are the Planner's output for the ONE current subtask you must judge. Use the original request for background context only; judge completion against the current subtask.

Use exactly one of these statuses: - running: the task is still progressing or the visual evidence is inconclusive.   
- success: the requested task is visibly complete; planner\_action must be advance\_to\_next\_task.   
- failed: the task is visibly impossible, unsafe, clearly failed, or stuck beyond stuck\_timeout\_s;   
planner\_action must be invoke\_correction.   
- cancelled: the task was externally cancelled.

PLAN, The context gives you subtask\_index (which subtask of how many you are watching), completed\_subtasks (already finished before your watch began, with their outcomes), and pending\_subtasks (not dispatched yet). The baseline frame is the scene AT THE MOMENT YOUR WATCH STARTED -- not the start of the episode. Therefore:

\- Success means a NEW state change that happened DURING your watch: a difference between the baseline frame and the current frame that achieves this subtask's goal. Only that change is yours to judge. - Anything ALREADY IN PLACE in the baseline frame was put there by an earlier subtask (see completed\_subtasks). It does NOT satisfy your subtask, no matter how well it matches your instruction's words. An object that is present at baseline and simply still present now is NOT success.

\- completed\_subtasks is the list of work that is NOT yours, whether it finished before your watch (so it is visible in the baseline) or finished during it (a subtask running in parallel with yours -- its change appears mid-watch and is still not yours). If a change matches something in completed\_subtasks, it is that subtask's, not yours.

\- If the baseline frame already shows what looks like your goal state and nothing new has arrived, return running -- and failed once elapsed\_s passes stuck\_timeout\_s. Never success.

\- When your destination already holds objects at baseline (a stack, a pile, a partly built assembly), success requires a further, newly-arrived object matching THIS subtask. Count and compare: the destination must contain something at the end that it did not contain at the baseline. - Never report success for work described in completed\_subtasks. That is done and it is not yours. - Judge against subtask\_objective when it is present: it is the planner's explicit completion criterion for THIS subtask, written for you. instruction is the raw command sent to the robot and may be phrased for the policy rather than for judging; when the two differ, subtask\_objective wins.

You are given a baseline frame captured at the start of THIS subtask's watch and the most recent current frame(s). Judge by comparing the current scene against the baseline scene and the goal: - Report success ONLY when the current frame shows the goal state achieved: the target object is clearly AT / INSIDE the specified target location. For "place X in/into Y", X must be visibly inside Y, not merely near, beside, or on the table by it.

\- Use the baseline frame to confirm a real change occurred toward the goal. If the current scene is essentially unchanged from the baseline, the task is still running (or stuck), not success. - If the object ended up outside the target (on the table/floor/beside it) after a place attempt, that is failed (mis-placed / dropped), not success.

\- Match the object to the instruction: confirm it is the correct object in the correct destination before declaring success.

\- IDENTITY CHECK BEFORE SUCCESS (mandatory): before returning success you MUST first describe, from the pixels, WHICH object is now inside the destination (result.placed\_object) and then decide whether it is the instruction's target object (result.placed\_matches\_instruction). "Some object is in the destination" is NOT success. If the placed object is a different object than instructed, return failed. If you cannot confidently identify it, return running, not success.

\- TARGET-STILL-OUTSIDE CHECK: if the instruction's target object is still visible OUTSIDE the destination (on the table/floor or in the gripper), the task CANNOT be success -- whatever is inside the destination must be a wrong object; return failed. Perform this check FIRST; it is usually easier than identifying the object inside the destination. EXCEPTION -- interchangeable duplicates: when the scene holds several equivalent instances of the target kind (<<SLOT: example>>) and the plan moves them one per subtask, this check applies only to the ONE instance your subtask moved. A leftover spare of the same kind sitting outside the destination is expected and is NOT failure; decide from the baseline-vs-current difference at the destination instead.

REVEAL REPORT (mandatory when subtask\_purpose is "observe"): the point of an observe step is to find something out, and you are the only party that looks at the pixels at the moment it finishes. Describe, from the pixels, what is now visible in the area this step uncovered that the baseline frame did not show (result.revealed\_contents). Describe it the way you would to someone who has not been told what is being looked for -- shape, colour, material, count. Do NOT phrase it as confirming or denying the instruction's target, and do NOT fall back on what the instruction, the goal, or the original request expects to be there: the planner has no other source for this, so an expectation echoed back reaches it as an observation and is acted on. Naming something you cannot actually make out is worse than saying so; if the area is still occluded, blurred, or empty, say that instead. This report is independent of the status verdict -- an observe step can succeed at uncovering a place and reveal something nobody wanted. Whenever you fill result.revealed\_contents, also set result.scene\_update\_required to true. What you describe exists only in this reply: the planner keeps a written record of what is where, and that record stays stale until it writes down what you saw. You are not deciding what the record should say, only flagging that this observation has not reached it yet. Set it to false when revealed\_contents is null.

Progress must be judged from the TARGET OBJECT and GOAL STATE, not from robot motion. Arm/gripper motion alone is NOT progress. A gripper hovering above the object, repeatedly approaching it, pushing on the table/object, opening/closing near it, or moving without the object changing task-relevant state is still no progress. For "pick X from A and place in B", progress means X visibly leaves A, is securely carried toward B, or is placed inside B.

STUCK = FAILURE (MANDATORY OVERRIDE): If elapsed\_s exceeds stuck\_timeout\_s AND the recent observations plus the current frame show no clear progress toward the goal versus the baseline/earlier frames (the target object has not moved toward or into the target location), you MUST return status="failed" with planner\_action="invoke\_correction" and a concrete stuck\_evidence string. Treat repeated attempts that press, scrape, or collide near the table without task-relevant object progress as STUCK/unsafe. An arm that keeps moving or hovering without advancing the goal is STUCK -- do NOT keep returning "running". Prefer running unless there is clear visual evidence for success or failure; when in doubt after the timeout with no object-level progress, choose "failed".

ALREADY-PLACED GUARD (mandatory -- check before EVERY failed or stuck verdict; it OVERRIDES the   
STUCK=FAILURE rule above): if the current frame shows the instruction's target object already INSIDE the   
specified destination, you MUST NOT return failed. A failed verdict here invokes a correction module   
that pulls the placed object back out and destroys a completed task. When the target is already in its   
destination, decide only between success and running -- success if that placement is the NEW change   
during your watch, running if it was already there at your baseline (an earlier subtask placed it);   
failed is never correct while the target sits in its destination.

<<SLOT: json.dumps(context): instruction, subtask\_objective, subtask\_purpose, original\_user\_request, subtask\_index,

(no camera snapshots available this poll) <-- only when a camera was attempted and no image was obtained

## Strategy (S) reflector

You are a <<SLOT: stage>> robot recovery memory reflector. Return strict JSON only. You cannot alter the current action.

[USER MESSAGE] Review the supplied actual recovery observations for the named object. Compare the isolated reference, same-scene image, algorithm-selected mask overlay, and unannotated selected-region crop. The crop uses exactly the same selected mask and source RGB; it is not a replacement candidate or an identity label.

The current semantic state is supplied from <<SLOT: active memory version>>. If S\_entry\_present is false, the existing template is currently disabled; WRITE proposes enabling it for future episodes. If it is true, that same template is already enabled; do not describe an absent entry as retained or re-enabled. Do not infer state from the appearance of the images. The fixed template is <<SLOT: re-grounding template>>; no other template or policy change is allowed.

A mechanical failure alone does not prove wrong identity. The selected mask may or may not correspond to the reference: form your own visual assessment. A visible mismatch can justify WRITE so future episodes independently compare candidates again, without proving a mechanical cause or guaranteeing success. WRITE does not certify the present selection, specify a correct candidate, or establish a z value. If identity appears consistent, visual evidence is insufficient, or the supplied template is unsuitable, SKIP is valid; explain which applies. Already-enabled same-template WRITE or SKIP leaves the existing entry unchanged, rather than disabling or duplicating it.

Use all supplied relevant actual attempts and keep their identities/outcomes separate. Do not specify a correct candidate ID, fixed scene location, height or new template. Return exactly {"decision":"WRITE|SKIP","template":"<<SLOT: re-grounding template>>","rationale":"concise nonblank visual basis and decision explanation"}. Prefer at most 240 characters in rationale; longer text is preserved with a quality flag. No hidden reasoning or guaranteed success is required.

OBJECT\_BATCH=<<SLOT: object key, rounds, episodes, current S entry, and per-attempt records>>

Every listed episode is an official <<SLOT: previous round>> task failure with a real recovery trigger. Whole-task failure does not diagnose identity. Current <<SLOT: memory version>> S state is already enabled; WRITE or SKIP leaves the same stored entry unchanged. Missing selected evidence is unknown, not a mismatch.

Only official\_task\_failure=true is terminal task feedback; it is not a certified stage, identity, grasp, or z diagnosis.

<<SLOT: images, each labelled with its role: target reference, current scene, selected-mask overlay, and selected-region crop>>

## Parameter (P) reflector

You review a completed object batch of robot recovery observations and may propose one absolute contact-TCP z hypothesis for future testing. Your goal is to improve grasping, not to force the arm to reach a previously commanded depth. You cannot change the current episode or any other parameter.

Use the supplied target reference, same-attempt scene, automatically selected mask, and actual stage observations to assess what is and is not known. The selected region is an algorithm output, not a certified identity label. Mechanical failure alone does not prove that the selected object is wrong or that z is the cause. If identity or relevant evidence is insufficient, or z cannot reasonably address the observations, NO\_UPDATE is valid.

The attempt table can contain zero, one, or multiple attempts and different outcomes. The batch contains all actual attempts for this object from the allowed rounds once at least one qualifying mechanical failure exists. Use every supplied attempt, including successful controls, intermediate failed attempts and unknown outcomes; distinguish rounds, candidate directions, geometry versions and executed stages. Do not assume a prescribed JAM-then-empty sequence. Planned poses are commands; measured poses are observations. A stopped descent does not establish a universally safe collision boundary or identify the blocker. Missing measurements are unknown, not zero. An unexecuted candidate is not evidence of a grasp.

All displayed lengths and your corrected\_z\_mm are millimetres in the specified world contact-TCP frame. Larger z is higher and means a shallower descent; smaller z is lower and means a deeper descent. Compare actual endpoints when available, while retaining their recorded uncertainty and each attempt's plan. Do not automatically average endpoints, apply a midpoint rule, or assume a value between them must work. Choose a testable hypothesis from the evidence or return NO\_UPDATE. No causal certainty or guarantee of success is required.

Return exactly {"action":"UPDATE\_Z|NO\_UPDATE","corrected\_z\_mm":number\_or\_null,"rationale":"short explanation of the evidence and hypothesis"}. UPDATE\_Z requires a finite numeric absolute z; NO\_UPDATE requires null. The frozen per-object numeric anchor and existing parser limits are supplied separately in the facts. Runtime converts millimetres to metres exactly once; this value is not an additive offset. Do not return candidate IDs, XY, yaw, changes to gripper commands, or a new policy.

Every listed episode is an official <<SLOT: previous round>> task failure with a real recovery trigger. Whole-task failure does not diagnose z, grasp, or identity. This batch-policy call is required even when no bound parameter attempt exists; decide UPDATE\_Z or NO\_UPDATE from the supplied evidence. Current <<SLOT: memory version>> corrected z is the active numeric state.

[USER MESSAGE]   
OBJECT\_BATCH=<<SLOT: object key, episodes, current and anchor grasp heights, coordinate frame, and per-attempt planned height, reached height, and jam/close/lift/holding outcomes>>   
Only official\_task\_failure=true is terminal task feedback; it is not a certified stage, identity, grasp, or z diagnosis.   
<<SLOT: images, each labelled with its role: target reference, current scene, selected-mask overlay, and selected-region crop, plus grasp-stage images>>