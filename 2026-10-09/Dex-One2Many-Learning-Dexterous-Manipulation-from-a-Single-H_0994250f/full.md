![](images/4e389d960628ff95285e96ce4ae822b94a05f6e8c677d615190668436e3bef83.jpg)

# Dex-One2Many: Learning Dexterous Manipulation from a Single Human Demonstration

Jusuk Lee<sup>1\*</sup>, Sungha Kim<sup>1\*</sup>, Yeonsoo Park<sup>1\*</sup>, Jonguk Cheon<sup>1</sup>, Yoonkyo Jung<sup>2</sup>, Yongjun You<sup>1</sup>, H. Jin Kim<sup>1</sup>, Jia-Bin Huang<sup>2</sup>, Furong Huang<sup>2,3</sup>, Youngseok Jang<sup>4†</sup>, Seungjae Lee<sup>2†</sup>

<sup>1</sup>Seoul National University <sup>2</sup>University of Maryland, College Park <sup>3</sup>All Purpose AI <sup>4</sup>KAIST

https://dex-one2many.github.io

While learning dexterous manipulation from a single human video ofers a promising alternative to costly robot demonstrations, many recent methods predominantly imitate demonstrated motions. Such strict motion matching often limits generalization to initial object poses, goal poses, and grasps not shown in the video. Alternatively, discovering a policy via reinforcement learning (RL) allows for broad generalization, but without prior guidance, it struggles with high-dimensional exploration in complex, multi-stage tasks. To address these coupled generalization and exploration challenges, we present Dex-One2Many, a real-to-sim-to-real framework that learns a generalizable dexterous manipulation policy from a single human video. Our key insight is to abstract the video into sequential scene graphs that guide RL, enabling eficient exploration while preserving broad generalizability. The graphs serve as generative constraints for sampling diverse reset states and provide dense rewards for each stage. Because the graphs constrain relations rather than exact poses, these reset states cover object poses and grasps beyond the video, while initializing each stage from them with dense rewards keeps exploration short and guided. Trained entirely in simulation, Dex-One2Many transfers zero-shot to a real multi-fingered hand. Across five tool-use and manipulation tasks, Dex-One2Many exceeds baselines by 6.5% in seen configurations, while its robust generalization widens this gap to 71% in unseen scenarios.

One human video  
Real-world execution across many unseen configurations  
![](images/b8aa62c1d853160429feb118acb0b3b5650105986618f941283fa5fd75af8195.jpg)  
Figure 1: From one human video to many dexterous behaviors. Given one human video of a manipulation task, Dex-One2Many learns a dexterous policy entirely in simulation and deploys it zero-shot in the real world. By training across diverse simulated task configurations, the resulting policy executes the task across many unseen configurations beyond the execution shown in the video.

## 1. Introduction

Dexterous manipulation is a promising direction for enabling robots to perform complex object interactions, but collecting enough high-quality robot demonstrations remains a major bottleneck. For high-degree-of freedom hands, teleoperation interfaces are particularly costly to design and operate, and the collected motions are tied to a specific robot morphology. Human videos ofer a way around this bottleneck: they are easy to collect without a robot or a teleoperation rig and contain rich visual and kinematic information about hand–object interactions. Recent methods exploit this to learn dexterous manipulation from just a single human video (Lum et al., 2025, Paliwal et al., 2026, Han et al., 2026, Zhu et al., 2026).

However, these single-video methods mostly convert the video into execution-level guidance, such as reconstructed object trajectories (Lum et al., 2025), retargeted hand motions (Paliwal et al., 2026, Han et al., 2026), or contact cues (Zhu et al., 2026). Such guidance facilitates high-dimensional dexterous policy learning, but also ties the resulting policy to the motion and spatial configuration shown in the video. Even for the same task, these methods generalize poorly to initial object poses, goal poses, and grasps not shown in the video.

Learning a dexterous policy that generalizes beyond the execution shown in the video raises two tightly coupled challenges. The first is task abstraction: deciding what to preserve from the video and what to leave free to vary. Preserving too much reduces the policy to replaying the recorded motion; preserving too little leaves the task underspecified. The second is a hard exploration problem: the more that is left free to vary, the more of the state space the policy must explore. Without execution-level guidance, reinforcement learning (RL) struggles to discover multi-finger, contact-rich interactions in this high-dimensional space, especially in multi-stage tasks.

To address both challenges, we present Dex-One2Many, a real-to-sim-to-real framework that learns a generalizable dexterous manipulation policy from a single human video (Fig. 1). Our key idea is to extract not the video’s motion but its task structure, a stage-wise scene graph specifying which hand–object and object–object relations must hold, and in what order, while leaving RL to learn how the hand acts. To this end, we propose two techniques (Fig. 2).

First, Stage-Wise Scene Graph Abstraction reduces the video to the relations the task requires, such as the hand holding an object or an object resting inside a container, and to the order in which they occur. Object poses, goal poses, and grasps remain free to vary. The abstraction splits the video into stages when these relations change and encodes each stage as a graph of predicates over the hand, objects, and their functional parts. Each stage graph describes not one recorded state but a set of states that all preserve the task.

Second, Scene-Graph-Grounded RL uses each stage graph as a generative constraint: it samples reset states that satisfy the stage’s relations, with diverse multi-finger grasps, and keeps only those that pass our simulation checks. One video thus becomes many training states across the whole workspace. The accepted states form the initial-state distribution of an MDP whose observations and rewards are derived from the graph. Starting episodes at every stage eases exploration: the policy can practice the last stage before mastering the first, rather than reaching it only by chance. Within each stage, predicate-derived rewards give a dense signal of progress toward satisfying the stage’s relations, without reference to a demonstrated trajectory. The policy therefore learns the task structure rather than the recorded execution.

Trained entirely in simulation, the resulting policy completes the demonstrated task zero-shot on a real multi-fingered hand across initial and goal object poses not shown in the video. We evaluate five tasks spanning direct object manipulation and tool use on a UR3 arm with a 20-DoF Wuji 1 hand. On configurations with initial and goal poses not shown in the video, Dex-One2Many achieves a 60–85% real-world success rate, while the baselines remain at or below 15%. Our main contributions are as follows:

• Stage-Wise Scene Graph Abstraction: We represent a human demonstration as a stage-wise scene graph that retains task-defining structure—the required hand–object and object–object relations and their ordering—while discarding execution-specific motion.

• Scene-Graph-Grounded RL: We use each stage’s scene graph to generate diverse reset states through the task, enabling the policy to explore alternative grasps and motions while grounding its observations and rewards from the graph.

• Zero-shot sim-to-real transfer on unseen configurations: Across five real-world tasks, policies learned from a single human video achieve 75% mean success on configurations not shown in the input video, exceeding the next-best baseline by 71 percentage points.

## 2. Related Work

Generalizing dexterous policies beyond the execution shown in a single human video raises two key challenges identified in the introduction: task abstraction and hard exploration. We review the closest work in these two directions, along with broader related work, in Appendix A.

Task Structure Extraction from Human Videos. Human videos have been used to learn reusable visual representations (Nair et al., 2022, Lee et al., 2026c), rewards (Ma et al., 2023, Kumar et al., 2022, Huang et al., 2024a), point tracks (Bharadhwaj et al., 2024), latent plans (Wang et al., 2023a), or action priors (Shaw et al., 2022) from large video collections. In the single-video setting, however, generalization depends largely on how the video is abstracted. Prior work abstracts the video into keypoints, keyframes, object-centric graphs, or object pose trajectories, then adapts the demonstrated execution to a new scene by warping or tracking it, or by synthesizing imitation data around it (Gao et al., 2023, Sieb et al., 2020, Heppert et al., 2024, Zhu et al., 2024, Li et al., 2024, Zhou et al., 2025, Tang et al., 2025, Heppert et al., 2026). Because what transfers is the demonstrated motion, adaptation extends only as far as that motion can be reshaped. Relational task specifications provide a more abstract alternative and are well established in symbolic and scene-graph planning (Fikes and Nilsson, 1971, Rana et al., 2023), and vision-language models have extracted symbolic task plans from a human video (Wake et al., 2024) and stage-wise relational constraints from language (Huang et al., 2024b). Building on this relational view, our Stage-Wise Scene Graph Abstraction extracts task structure from the single video, while discarding execution details, encoding each stage as a graph whose edges are predicates stating which hand–object and object–object relations must hold, in the order the video shows. Unlike ORION’s graphs (Zhu et al., 2024), our stage-wise scene graph carries no coordinates or keypoint trajectories from the video, so each graph describes not one state but the set of all states that satisfy its predicates. Rather than a plan to execute, a reference to adapt, or a cost to optimize, each graph serves as a constraint that sampled training states must satisfy, leaving object poses, goal poses, and grasps free to vary. RL then learns how the hand acts instead of inheriting the demonstrated motion.

Learning Dexterous Manipulation from a Single Human Video. Dexterous manipulation poses a hard exploration problem for RL because multi-fingered hands must coordinate many joints through contact-rich interactions (Rajeswaran et al., 2018). Prior methods that learn dexterous manipulation from human video therefore extract execution-level guidance, such as object pose trajectories used as a tracking reward (Lum et al., 2025, Zhu et al., 2026, Sharma et al., 2026, Chen et al., 2025b, Mandi et al., 2026), retargeted hand motions used as references to imitate or refine (Qin et al., 2022, Li et al., 2025, Pan et al., 2025, Paliwal et al., 2026, Han et al., 2026, Guzey et al., 2025), or demonstrated contacts (Zhu et al., 2026, Sharma et al., 2026). This guidance makes exploration tractable but ties the resulting behavior to the demonstrated motion, even when another initial object pose, goal pose, or grasp requires a diferent execution of the same task. As a result, their generalization is largely limited to local variations around the demonstrated configuration (Lum et al., 2025, Han et al., 2026, Zhu et al., 2026).<sup>1</sup> Removing execution-level guidance, however, creates the second challenge: how to make RL explore eficiently from task structure alone. Resetbased RL alleviates hard exploration by starting episodes near a known goal (Florensa et al., 2017) or along a demonstration (Resnick et al., 2018, Salimans and Chen, 2018, Tao et al., 2024, Bauza et al., 2025), so the policy learns from later states before it can reach them. Demonstration-based resets, however, inherit the demonstrated execution. More recent approaches such as OmniReset (Yin et al., 2026) and Video2Policy (Ye et al., 2025) reduce this dependence by generating reset states or task programs, but rely on user-specified task information or video-derived object states and have only been demonstrated with parallel-jaw grippers. Our approach grounds resets, observations, and rewards in a stage-wise scene graph derived from the human video, so it derives the task structure without imitating the execution.

![](images/7acee67b2f7d2d6057a9b42d9c56f395f2793d0f06ccc16ca4d6adb826538342.jpg)  
Figure 2: Overview of Dex-One2Many. (1) From a single human video, a VLM extracts a stage-wise scene graph of object relations. (2) Each stage graph constrains the sampling of object poses and synthesized grasps, yielding verified reset states for every stage. An RL policy trained from these resets with predicate-derived rewards acts in task space through a QP controller. (3) The policy transfers zero-shot to unseen initial poses, goal poses, and grasps on real hardware.

## 3. Method

Given a single demonstration of a manipulation task, our goal is to learn a dexterous policy that generalizes to diverse spatial configurations, including randomized object poses and novel grasps, beyond the original video. To achieve this, we first abstract the demonstration into a stage-wise scene graph (Sec. 3.1). We then formulate policy learning as a Markov decision process (MDP) grounded in this graph, with observations, rewards, and an initial-state distribution constructed from it without task-specific design (Sec. 3.2). The learned policy then transfers zero-shot to the real robot (Sec. 3.3). Fig. 2 summarizes the full pipeline.

## 3.1. Task Abstraction as a Stage-wise Scene Graph

We abstract the human video as a sequence of stage-wise scene graphs that preserve which relations must hold and in what order, while allowing object poses, goal poses, grasps, and trajectories to vary. Each graph represents one stage of the task: its nodes identify the hand, task-relevant objects, and their functional parts, while its edges describe the relations characterizing that stage. When the same task is performed from diferent configurations, the poses and trajectories may difer, but the required relations and their order remain the same. We refer to this abstraction as Stage-Wise Scene Graph Abstraction.

To encode the relations carried by the scene-graph edges without fixing the poses observed in the video, we use a predefined, task-independent predicate vocabulary (Table 3). Some predicates specify relations that must hold, such as grasp, contact, and inside, while others describe proximity toward the next relation to be established, such as pre\_grasp and pre\_inside. The former describe what must hold in the current stage, whereas the latter indicate which relation should be established next. Together, these predicates describe task progression without fixing the specific configuration shown in the video.

We recover these stage-wise graphs from the video using a vision-language model (VLM) that abstracts the task directly. Given the video and the predicate vocabulary, the VLM identifies task-relevant entities and functional parts as graph nodes and infers the predicate relations between them over time. We introduce a new stage whenever the predicate relations change, either because a new relation is added or because a relation being approached becomes established, such as pre\_grasp transitioning to grasp. This yields an ordered sequence $\mathcal { G } _ { 1 : K } = \left( \mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { K } \right)$ . Fig. 2 illustrates how the stage-wise scene graph captures task progression in the sweeping task: a new stage is introduced when pre\_grasp becomes grasp; in later stages, the established grasp and contact relations remain as constraints, while the toy cat is moved toward the dustpan until the new inside relation is established.

## 3.2. Scene-graph-grounded Reinforcement Learning

Given the scene-graph sequence $\mathcal { G } _ { 1 : K }$ from Sec. 3.1, we formulate policy learning as a finite-horizon MDP $\mathcal { M } = \langle { \mathcal { S } } , { \mathcal { A } } , p , r , \gamma , \rho _ { 0 } , T \rangle$ . Here, $\gamma$ and T denote the discount factor and episode horizon; the simulator provides transition dynamics $p ,$ and $\mathcal { A }$ consists of desired palm and fingertip velocities tracked by a low-level QP controller (Sec. 3.3). The scene graph defines the policy observation o from ${ \mathcal { S } } _ { z }$ , the reward $r ,$ and the initial-state distribution , as described below. $\rho _ { 0 }$

Observation from the scene graph. We construct the policy observation o from the task-relevant state associated with the scene graph’s nodes and edges. Nodes are represented by role: the hand node provides robot proprioception, object nodes encode object geometry and 6D pose, and functional-part nodes localize regions such as the grasp region. Edges capture the relative 6D pose between connected entities. Because these encodings are shared across tasks, the same observation construction applies to diferent task graphs. Concatenating these node and edge features forms the policy observation, making the observation design reusable (see Appendix C.1 for details).

Task reward from the scene graph. Each scene-graph edge encodes a predicate between two entities, which we map to a geometric or physical reward. Geometric predicates use the distance metric in Table 3 as a dense progress signal, a 3D Cartesian distance for spatial relations (e.g., to a grasp region for pre\_grasp or a receptive region for pre\_inside; Fig. 3(a)) and a joint-space distance for seat. For predicates not naturally represented by such a distance, we use a predicate-specific physical measure. Specifically, we evaluate grasp using CHORD’s force-closure reward in 6D wrench space (Zhu et al., 2026), augmented with a grasp-quality measure (Lynch and Park, 2017) to ensure grasp stability (see Appendix C.2 for details and discussion).

(a) Distance functions for geometric predicates  
![](images/605bc80216f6269cae7b17a78cbd85bab1dfc97cc6c5e24dcdd23e4501d527d9.jpg)

![](images/7a8c34d9f3e168f7e7cc0913c11cdbb66a57bca947ebe719230d8f628bfe79ca.jpg)  
Figure 3: Using scene-graph predicates for policy learning. (a) Predicates that can be expressed geometrically are mapped to Cartesian-space distance functions. (b) The predicates in a stage-wise scene graph determine which reward terms are assembled into the task reward. (c) The same assembly rule is instantiated for direct and tool-use manipulation from their respective scene graphs.

With a reward defined for each predicate, we construct the task reward from the stage-wise scene graph. As the task progresses to a new stage, the reward for the newly introduced predicate is added to the task reward (Fig. 3(b)). Because each predicate has a fixed reward definition, we apply the same construction across diferent tasks. Fig. 3(c) shows this shared construction for direct and tool-use manipulation tasks: although they involve diferent entities and predicate sequences, their task rewards are assembled from the same predicate-level reward functions.

Initial-state distribution from the stage-wise scene graphs. While predicate rewards guide progress from a state, the stage-wise scene graph defines the initial-state distribution by determining which parts of the task the policy can practice during training. Since each stage graph $\mathcal { G } _ { k }$ fixes relations but not poses or grasps, we construct $\rho _ { 0 }$ by using it as a generative constraint. For geometric predicates, the same feature distance that provides their dense progress signal also serves as a placement rule: we position the source node relative to its target node so that the condition in Table 3 holds, applying these rules along the edges of $\mathcal { G } _ { k }$ (Fig. 10, Appendix C.3). For the final stage, we first sample a state that satisfies the goal predicate and then perturb it into a near-goal state, since an already successful reset provides less useful learning signal. Episodes start uniformly from such resets across all stages, so the policy practices late interactions without first reaching them from the task start, which reduces the exploration burden of long-horizon tasks (Yin et al., 2026).

For grasp, the placement rule is grasp synthesis. Instead of fixing a single grasp, we place the hand at one of many multi-finger grasps generated by Dexonomy (Chen et al., 2025a) within the object’s grasp region. The policy thus encounters diverse hand–object configurations during training(Appendix C.3.2). Each grasp also includes the finger command $q _ { \mathrm { c m d } }$ that closes the hand on the object, which we restore at initialization so the grasp holds from the first step. Before a sampled state enters $\rho _ { 0 } .$ , we instantiate it in simulation and keep it only if it passes our checks for scene collision and reset stability, using grasps certified beforehand under external forces (Appendix C.3.3).

## 3.3. Zero-shot Sim-to-Real Transfer

We train the policy with PPO (Schulman et al., 2017) using an asymmetric actor–critic (Pinto et al., 2018), in which the critic receives privileged observations and the actor receives only the scene-graph-grounded observation available on the real robot. To enable zero-shot transfer, we randomize physical properties such as object mass and friction during training (Appendix C.5) and address the sim-to-real gap in perception and control. On the perception side, we perturb the actor observations during training with object-pose noise, perceived-size noise, and observation delay. At deployment, we estimate object poses from two RGB-D cameras using FoundationPose++ (Yan and Chu, 2025) and fuse the estimates with depth-consistency weights that down-weight unreliable views (Appendix D). On the control side, the policy outputs desired palm and fingertip velocities, which a QP low-level controller (Lee et al., 2026a) maps to joint velocities while enforcing kinematic and deployment-time safety constraints. This lets us adjust joint-velocity limits and add collision-avoidance constraints at deployment without retraining the policy. Appendix C.4 provides further details on the QP controller and demonstrates a real-world hardware failure prevented by the QP constraints.

## 4. Experiments

In this section, we evaluate Dex-One2Many in simulation and on real hardware. Our experiments address three questions:

(Q1) Does Dex-One2Many generalize to initial and goal configurations not shown in the single human video?

(Q2) Does Dex-One2Many apply across diverse dexterous embodiments with diferent kinematics?

(Q3) Which design choices of Dex-One2Many are critical to its performance?

## 4.1. Experimental Setup

Tasks. We evaluate Dex-One2Many on five manipulation tasks that span direct object manipulation and tool use (Fig. 4). The direct-manipulation tasks (Doll, Can, Stamp) require the robot to grasp an object and move it to a target configuration, while the tool-use tasks (Hammer, Sweep) require the robot to grasp and manipulate a tool to interact with another object.

Hardware Setup. We conduct all real-world experiments on a 6-DoF UR3 robot arm equipped with a 20-DoF Wuji 1 dexterous hand. Two Intel RealSense D435i RGB-D cameras capture the workspace from complementary viewpoints. We estimate object poses from these observations using FoundationPose++ (Yan and Chu, 2025).

Baselines. We compare our Dex-One2Many against two prior methods that learn dexterous manipulation from a single human video. These methods difer in what information they extract from the human demonstration and how they use it for policy learning. Do as I Do (Paliwal et al., 2026) reconstructs the 4D hand–object interaction and retargets the recovered human hand motion to the robot through trajectory optimization. CHORD (Zhu et al., 2026) uses the demonstrated object-state evolution as an imitation reward and augments it with contact-wrench guidance that encourages the robot to reproduce the demonstrated trajectories.

![](images/7395152b5769f263097bb11629b3dcf793ea8a3c39b574454439730df57ed86a.jpg)  
Figure 4: Generalization beyond the human video in simulation and the real world. Success rates are reported for configurations shown in the input human video (Seen, top) and configurations with diferent initial and goal poses not shown in the video (Unseen, bottom), in both simulation and the real world. The bottom row illustrates the five evaluated tasks; N/A denotes an experiment unavailable due to a hardware failure.

## 4.2. Does Dex-One2Many Generalize Beyond the Human Video?

Evaluation protocol. We evaluate generalization in both simulation and the real world using the same Seen/Unseen protocol. For each method and task, we evaluate 30 episodes in simulation and 30 trials on the real robot: 10 in Seen configurations and 20 in Unseen configurations. The Seen configurations reproduce the initial and goal poses shown in the input human video, whereas the Unseen configurations use diferent initial and goal poses not present in the video. All evaluation episodes start with an empty hand and require the policy to execute the full task from reaching to task completion. We deploy real-world policies zero-shot, without real-world data or fine-tuning.

Results. Figure 4 compares performance on Seen and Unseen configurations in simulation and the real world. On Seen configurations, the baselines can achieve high success on several tasks, showing that execution-level guidance extracted from the human video can reproduce the demonstrated task configuration. The diference becomes pronounced on Unseen configurations. Across the evaluated baselines, success remains at or below 15% in both simulation and the real world, whereas Dex-One2Many achieves 85–100% success in simulation d 6 h l b

Figure 5 further isolates generalization to unseen initial object positions on the Can task. With the goal fixed, the baselines succeed primarily near the demonstrated object position, whereas Dex-One2Many maintains high success across a substantially broader range of the workspace. This shows that the performance gap is not limited to a particular set of evaluation configurations, but reflects broader spatial generalization beyond the demonstrated execution. Additional results across the remaining tasks are provided in Appendix E.1.

![](images/fae9cbaa9d713db4e131a1d2fc56dd108af9acfd3119faa4ddd6caa0bb495d8d.jpg)  
Figure 5: Generalization to unseen initial object positions on the Can task. In simulation with the goal fixed, Dex-One2Many succeeds across a broader range of initial positions than the baselines.

The baselines derive execution-level guidance from the demonstrated hand or object motion, which is tied to the execution observed in the video. In contrast, Dex-One2Many uses stage-wise scene graphs as generative constraints for diverse task-consistent training states, allowing the policy to learn the same task structure across diferent configurations beyond the demonstrated execution.

## 4.3. Does Dex-One2Many Apply Across Diverse Dexterous Embodiments?

Evaluation protocol. We evaluate Dex-One2Many in simulation across four arm–hand embodiments with diferent kinematics: UR3 + Wuji 1, UR5e + Sharpa Wave, UR5e + Wuji 2, and UR5e + Allegro. For each embodiment, we only replace the robot URDF and synthesize a corresponding grasp set, keeping the task specification and all other components unchanged. We train a separate policy for each embodiment and report success rates over 500 evaluation episodes per task, with initial object positions sampled within the reachable part of $( x , y ) \in [ 0 . 3 , 0 . 7 ] \times [ - 0 . 1 7 , 0 . 4 ] \mathrm { m }$

Results. Table 1 reports task-wise success rates and the mean across five tasks for four distinct dexterous embodiments. Despite substantial diferences in arm–hand kinematics, Dex-One2Many achieves consistently high performance across all four tested embodiments, with mean success rates ranging from 94.6% to 96.8%. Appendix E.3 provides rollouts across all five tasks and multiple configurations. Figure 6 further shows that the learned execution is embodimentspecific. Under the same task specification, diferent hands adopt diferent grasp types, and the same hand can change its grasp as the object configuration changes. Rather than retargeting a fixed human grasp, Dex-One2Many exposes RL to diverse grasps synthesized for each morphology, and the policy learns which grasp suits each scene configuration. Appendix E.2 provides additional grasp examples across embodiments and configurations.

Table 1: Simulation performance across diverse dexterous embodiments. Success rates (%) over 500 episodes per task.
<table><tr><td colspan="3">UR3 UR5e UR5e UR5e Wuji 1 Sharpa Wuji 2 Allegro</td></tr><tr><td>Doll</td><td>88.4 98.4</td><td>97.2 97.2</td></tr><tr><td>Can 93.6</td><td>97.4</td><td>95.2 92.2</td></tr><tr><td>Stamp 98.6</td><td>96.6</td><td>97.0 98.8</td></tr><tr><td>Hammer 94.6</td><td>94.2</td><td>97.6 91.8</td></tr><tr><td>Sweep 98.0</td><td>96.6</td><td>97.0 100.0</td></tr><tr><td>Mean 94.6</td><td>96.6</td><td>96.8 96.0</td></tr></table>

![](images/1443dfeaa1f068d73f0d0bcb17fa0b4523c2e19904b8fc5521e84aa42a11e4e7.jpg)  
(a) UR3 + Wuji 1

![](images/ff5cd6c0f8d73f7c2c8bdd2d722927c789641425308000a4ecb197c313f943d2.jpg)  
(b) UR5e + Sharpa Wave

![](images/4d4f77da2c4c3f09392bfb5eaf8fd950d7b098763ed81113fa08923e405c60db.jpg)  
(c) UR5e + Wuji 2

![](images/9517185a1dd226ca801a1abc8c0d1938a3b8d6952a62d8c4636c3e270bd96327.jpg)  
(d) UR5e + Allegro

Figure 6: Diverse grasp strategies across dexterous embodiments and object configurations on the Doll task. Across the 33-type grasp taxonomy (Feix et al., 2015), diferent embodiments learn diferent grasp types, and the same embodiment adapts its grasp across object configurations.  
![](images/aae9eb45fabf595581d86954927cbe10b5f353fc5b56be615774b2274f54dfa4.jpg)  
Dex-One2Many (Ours) w/o stage-wise reset w/o dense reward w/o q<sub>cmd</sub>  
(a) Component ablations

![](images/20657a882194d7212da7142903ca070a955c25aae202acbfe7870464922caa41.jpg)  
49,152 envs (Ours) 24,576 envs 12,288 envs  
(b) Training scale  
Figure 7: Ablation studies. (a) Removing each component. (b) Varying the number of environments.

## 4.4. Which Design Choices Are Critical to Dex-One2Many?

Evaluation protocol. We conduct all ablation studies on the Sweep task in simulation. Success is measured on full-task episodes from empty-hand starts. We study two aspects of Dex-One2Many: the contribution of its learning components and the efect of training scale. For component ablations, we remove one component at a time while keeping all other training settings unchanged. For the training-scale analysis, we vary only the number of parallel environments used for PPO training.

Component ablations. From the human video, Dex-One2Many extracts a stage-wise scene graph that specifies the task relations at each stage. We examine three design choices in how this task structure is used for policy learning. (1) Removing stage-wise resets (w/o stage-wise reset) drops success after 2k PPO updates from over 90% to about 16%, showing that starting from intermediate stages is key to exploring multi-stage tasks. (2) Removing predicate-derived rewards (w/o dense reward) keeps the success rate at 0% throughout training, showing that dense feedback toward the required task relations is critical for guiding exploration. (3) Finally, omitting the verified closing command $( \mathrm { w } / \mathrm { o } ~ q _ { c m d } )$ lowers success to about 48% after 2K PPO updates and induces frequent grasp slips and drops, showing that restoring active contact forces is vital for stable transitions across stages.

Efect of training scale. Increasing the number of parallel environments substantially improves learning speed and final performance. With 49,152 environments, Dex-One2Many reaches over 90% success, whereas training with 24,576 or 12,288 environments remains below 25% within the same number of PPO updates. These results show that large-scale parallel experience collection substantially benefits policy learning under our diverse reset distribution.

## 5. Conclusion

We presented Dex-One2Many, a real-to-sim-to-real framework for learning generalizable dexterous manipulation from a single human video. Rather than treating the demonstrated motion as an execution reference, Dex-One2Many abstracts the video into a stage-wise scene graph that preserves the task structure while leaving object poses, goal poses, grasps, and motion paths free to vary. Scene-Graph-Grounded RL then uses this structure to construct diverse stage-wise reset states and predicate-derived rewards, making exploration tractable without inheriting the demonstrated trajectory. Across five manipulation tasks, the resulting policies generalize to configurations not shown in the video and transfer zero-shot to a real multi-fingered hand. These results suggest that extracting what must be achieved from a human video, while allowing RL to discover how to achieve it, provides a promising route from a single demonstration to diverse dexterous behaviors.

Limitations and future work. Although Dex-One2Many substantially improves generalization across unseen configurations and applies to diverse hand morphologies, several limitations remain. First, a sim-to-real gap remains because real-world deployment relies on estimated 6D object poses. Pose-estimation noise and diferences between simulated and real object geometry can degrade policy performance, particularly during contact-rich interactions. Second, our real-to-sim pipeline reconstructs only rigid objects, so articulated objects would require manually built assets. Since a single video may not reveal the underlying articulation structure, such as joint locations, axes, and joint limits, automatic real-to-sim reconstruction remains dificult. Inferring these structural kinematics directly from unconstrained video is an important avenue for future work.

## 6. Acknowledgment

This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [No. RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)], the Technology Innovation Program (RS-2025-25453780, Development of a National Humanoid AI Robot Foundation Model for Multi-Task Applications) funded by the Ministry of Trade, Industry & Energy (MOTIE, Korea), and Samsung Electronics Robotics eXperience (RX). We thank Jinu Pahk and Kangin Lee from Tommoro Robotics for providing the Wuji 1 hand used in our real-world experiments.

## References

Ananye Agarwal, Shagun Uppal, Kenneth Shaw, and Deepak Pathak. Dexterous functional grasping. In Conference on Robot Learning (CoRL), 2023.

OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1):3–20, 2020.

Lars Ankile, Anthony Simeonov, Idan Shenfeld, Marcel Torne, and Pulkit Agrawal. From imitation to refinement-residual rl for precise assembly. In IEEE International Conference on Robotics and Automation (ICRA), 2025.

Seung Hyeon Bang, Carlos Arribalzaga Jové, and Luis Sentis. Rl-augmented mpc framework for agile and robust bipedal footstep locomotion planning and control. In IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids), 2024.

Maria Bauza, Jose Enrique Chen, Valentin Dalibard, Nimrod Gileadi, Roland Hafner, Murilo F. Martins, Joss Moore, Rugile Pevceviciute, Antoine Laurens, Dushyant Rao, Martina Zambelli, Martin Riedmiller, Jon Scholz, Konstantinos Bousmalis, Francesco Nori, and Nicolas Heess. Demostart: Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots. In IEEE International Conference on Robotics and Automation (ICRA), 2025.

Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In European Conference on Computer Vision (ECCV), 2024.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International Conference on Learning Representations (ICLR), 2026.

Jiayi Chen, Yuxing Chen, Jialiang Zhang, and He Wang. Task-oriented dexterous hand pose synthesis using diferentiable grasp wrench boundary estimator. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

Jiayi Chen, Yubin Ke, Lin Peng, and He Wang. Dexonomy: Synthesizing all dexterous grasp types in a grasp taxonomy. In Robotics: Science and Systems (RSS), 2025a.

Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, et al. Sam 3d: 3dfy anything in images. In Conference on Computer Vision and Pattern Recognition (CVPR), 2026a.

Yifei Chen, Shihan Lu, and Colgate Kevin Lynch. Robust in-hand manipulation via priors in reinforcement learning and mechanical design. arXiv preprint arXiv:2607.12105, 2026b.

Zerui Chen, Shizhe Chen, Etienne Arlaud, Ivan Laptev, and Cordelia Schmid. ViViDex: Learning vision-based dexterous manipulation from human videos. In IEEE International Conference on Robotics and Automation (ICRA), 2025b.

Prithwish Dan, Kushal Kedia, Angela Chao, Edward Weiyi Duan, Maximus Adrian Pace, Wei-Chiu Ma, and Sanjiban Choudhury. X-sim: Cross-embodiment learning via real-to-sim-to-real. In Conference on Robot Learning (CoRL), 2025.

Thomas Feix, Javier Romero, Heinz-Bodo Schmiedmayer, Aaron M Dollar, and Danica Kragic. The grasp taxonomy of human grasp types. IEEE Transactions on human-machine systems, 46(1):66–77, 2015.

Carlo Ferrari and John Canny. Planning optimal grasps. In IEEE International Conference on Robotics and Automation (ICRA), 1992.

Richard E Fikes and Nils J Nilsson. Strips: A new approach to the application of theorem proving to problem solving. Artificial intelligence, 2(3-4):189–208, 1971.

Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, and Pieter Abbeel. Reverse curriculum generation for reinforcement learning. In Conference on Robot Learning (CoRL), 2017.

Siddhant Gangapurwala, Mathieu Geisert, Romeo Orsolino, Maurice Fallon, and Ioannis Havoutis. Rloc: Terrain-aware legged locomotion using reinforcement learning and optimal control. IEEE Transactions on Robotics, 38(5):2908–2927, 2022.

Jianfeng Gao, Zhi Tao, Noémie Jaquier, and Tamim Asfour. K-vil: Keypoints-based visual imitation learning. IEEE Transactions on Robotics, 39(5):3888–3908, 2023.

Gemini Team, Google. Gemini 3.1 pro: A smarter model for your most complex tasks. https://blog. google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/, 2026. Accessed: 2026-09-26.

Irmak Guzey, Yinlong Dai, Georgy Savva, Raunaq Bhirangi, and Lerrel Pinto. Bridging the human to robot dexterity gap through object-oriented rewards. In IEEE International Conference on Robotics and Automation (ICRA), 2025.

Yunhai Han, Jianuo Qiu, Linhao Bai, Ziyu Xiao, Zihang Zeng, Yangcen Liu, Zhaodong Yang, Shalin Jain, Wenrui Ma, Jiaqi Fu, Yuqian Zheng, Manisha Natarajan, Muhammad Zubair Irshad, Kenneth Shaw, Matthew Gombolay, Zsolt Kira, and Harish Ravichandar. Video2sim2real: Full-stack autonomous dexterous skill acquisition from a single human video. arXiv preprint arXiv:2606.08828, 2026.

Ankur Handa, Arthur Allshire, Viktor Makoviychuk, Aleksei Petrenko, Ritvik Singh, Jingzhou Liu, Denys Makoviichuk, Karl Van Wyk, Alexander Zhurkevich, Balakumar Sundaralingam, et al. Dextreme: Transfer of agile in-hand manipulation from simulation to reality. In IEEE International Conference on Robotics and Automation (ICRA), 2023.

Nick Heppert, Max Argus, Tim Welschehold, Thomas Brox, and Abhinav Valada. Ditto: Demonstration imitation by trajectory transformation. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024.

Nick Heppert, Minh Quang Nguyen, and Abhinav Valada. Scaling single human demonstrations for imitation learning using generative foundational models. In IEEE International Conference on Robotics and Automation (ICRA), 2026.

Lei Huang, Weijia Cai, Zihan Zhu, Chen Feng, Helge Rhodin, and Zhengbo Zou. Virl: Self-supervised visua graph inverse reinforcement learning. In Conference on Robot Learning (CoRL), 2024a.

Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. In Conference on Robot Learning (CoRL), 2024b.

Kushal Kedia, Tyler Ga Wei Lum, Jeannette Bohg, and C. Karen Liu. Simtoolreal: An object-centric policy for zero-shot dexterous tool manipulation. In Robotics: Science and Systems (RSS), 2026.

Sateesh Kumar, Jonathan Zamora, Nicklas Hansen, Rishabh Jangir, and Xiaolong Wang. Graph inverse reinforcement learning from diverse videos. In Conference on Robot Learning (CoRL), 2022.

Ho Jae Lee, Yonghyeon Lee, Alexander Alexiev, Tzu-Yuan Lin, Se Hwan Jeon, and Sangbae Kim. Learning reactive dexterous grasping via hierarchical task-space rl planning and joint-space qp control. arXiv preprint arXiv:2605.03363, 2026a.

Jayjun Lee, Jessica Yin, Asif Rana, Nicholas Blauch, Sam Mady, Mohak Bhardwaj, Nima Fazeli, Nathan Ratlif, Karl Van Wyk, and Ankur Handa. Adept: Accelerating dexterity via pre-training and post-training using reinforcement learning. In Conference on Robot Learning (CoRL), 2026b.

Jusuk Lee, Seungjae Lee, Jonghun Shin, Hoseong Jung, Sungha Kim, Daesol Cho, H Jin Kim, Jia-Bin Huang, and Furong Huang. Dynaflip: Rethinking robotics perception via tri-modal-dynamics guided representation. arXiv preprint arXiv:2605.30350, 2026c.

Jinhan Li, Yifeng Zhu, Yuqi Xie, Zhenyu Jiang, Mingyo Seo, Georgios Pavlakos, and Yuke Zhu. Okami: Teaching humanoid robots manipulation skills through single video imitation. In Conference on Robot Learning (CoRL), 2024.

Kailin Li, Puhao Li, Tengyu Liu, Yuyang Li, and Siyuan Huang. Maniptrans: Eficient dexterous bimanual manipulation transfer via residual learning. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Tengyu Liu, Zeyu Liu, Ziyuan Jiao, Yixin Zhu, and Song-Chun Zhu. Synthesizing diverse and physically stable grasps with arbitrary hand structures using diferentiable force closure estimator. IEEE Robotics and Automation Letters, 7(1):470–477, 2021.

Tyler Ga Wei Lum, Olivia Y. Lee, C. Karen Liu, and Jeannette Bohg. Crossing the human-robot embodiment gap with sim-to-real rl using one human demonstration. In Conference on Robot Learning (CoRL), 2025.

Tyler Ga Wei Lum, Kushal Kedia, C. Karen Liu, and Jeannette Bohg. Play2perfect: What matters in dexterous play pretraining for precise assembly? In Conference on Robot Learning (CoRL), 2026.

Kevin M Lynch and Frank C Park. Modern robotics. Mechanics, Planning, and Control, 2017.

Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. In International Conference on Learning Representations (ICLR), 2023.

Zhao Mandi, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, and Shuran Song. Dexmachina: Functional retargeting for bimanual dexterous manipulation. In International Conference on Machine Learning (ICML), 2026.

Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Munoz, Xinjie Yao, René Zurbrügg, Nikita Rudin, et al. Isaac lab: A gpu-accelerated simulation framework for multi-modal robot learning. arXiv preprint arXiv:2511.04831, 2025.

Ken Museth, Jef Lait, John Johanson, Jef Budsberg, Ron Henderson, Mihai Alden, Peter Cucka, David Hill, and Andrew Pearce. Openvdb: an open-source data structure and toolkit for high-resolution volumes. In ACM SIGGRAPH 2013 Courses, SIGGRAPH ’13, 2013.

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. In Conference on Robot Learning (CoRL), 2022.

Bhawna Paliwal, Haritheja Etukuru, William Liang, Pieter Abbeel, Nur Muhammad Mahi Shafiullah, and Jitendra Malik. Do as i do: Dexterous manipulation data from everyday human videos. arXiv preprint arXiv:2606.19333, 2026.

Chaoyi Pan, Changhao Wang, Haozhi Qi, Zixi Liu, Homanga Bharadhwaj, Akash Sharma, Tingfan Wu, Guanya Shi, Jitendra Malik, and Francois Hogan. Spider: Scalable physics-informed dexterous retargeting. arXiv preprint arXiv:2511.09484, 2025.

Juhan Park, Taerim Yoon, Seungmin Kim, Joong-Gil Kim, Wontae Ye, Jeongeun Park, Yoonbyung Chai, Geonwoo Cho, Geunwoo Cho, Dohyeong Kim, Kyungjae Lee, Yong-Jae Kim, and Sungjoon Choi. Learning dexterous grasping from sparse taxonomy guidance. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026.

Lerrel Pinto, Marcin Andrychowicz, Peter Welinder, Wojciech Zaremba, and Pieter Abbeel. Asymmetric actor critic for image-based robot learning. In Robotics: Science and Systems (RSS), 2018.

Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang. DexMV: Imitation learning for dexterous manipulation from human videos. In European Conference on Computer Vision (ECCV), 2022.

Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. In Robotics: Science and Systems (RSS), 2018.

Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning. In Conference on Robot Learning (CoRL), 2023.

Cinjon Resnick, Roberta Raileanu, Sanyam Kapoor, Alexander Peysakhovich, Kyunghyun Cho, and Joan Bruna. Backplay: "man muss immer umkehren". arXiv preprint arXiv:1807.06919, 2018.

Tim Salimans and Richard Chen. Learning montezuma’s revenge from a single demonstration. arXiv preprint arXiv:1812.03381, 2018.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Satvik Sharma, Samrat Sahoo, Huang Huang, Fei-Fei Li Jiajun Wu, Dorsa Sadigh, and Jeannette Bohg. One demonstration, many objects: Generalizing manipulation via local contact geometry. arXiv preprint arXiv:2609.01938, 2026.

Kenneth Shaw, Shikhar Bahl, and Deepak Pathak. Videodex: Learning dexterity from internet videos. In Conference on Robot Learning (CoRL), 2022.

Maximilian Sieb, Zhou Xian, Audrey Huang, Oliver Kroemer, and Katerina Fragkiadaki. Graph-structured visual imitation. In Conference on Robot Learning (CoRL), 2020.

Aaditya Singh et al. OpenAI GPT-5 system card, 2025. URL https://arxiv.org/abs/2601.03267.

Chao Tang, Anxing Xiao, Yuhong Deng, Tianrun Hu, Wenlong Dong, Hanbo Zhang, David Hsu, and Hong Zhang. Mimicfunc: Imitating tool manipulation from a single human video via functional correspondence. In Conference on Robot Learning (CoRL), 2025.

Stone Tao, Arth Shukla, Tse kai Chan, and Hao Su. Reverse forward curriculum learning for extreme sample and demonstration eficiency in reinforcement learning. In International Conference on Learning Representations (ICLR), 2024.

Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta, and Pulkit Agrawal. Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation. In Robotics: Science and Systems (RSS), 2024.

Sébastien Valette and Jean-Marc Chassery. Approximated centroidal voronoi diagrams for uniform polygonal mesh coarsening. Computer Graphics Forum, 23(3):381–389, 2004.

Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, and Katsushi Ikeuchi. Gpt-4v (ision) for robotics: Multimodal task planning from human demonstration. IEEE Robotics and Automation Letters, 9 (11):10567–10574, 2024.

Weikang Wan, Haoran Geng, Yun Liu, Zikang Shan, Yaodong Yang, Li Yi, and He Wang. Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist specialist learning. In International Conference on Computer Vision (ICCV), 2023.

Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, and Anima Anandkumar. Mimicplay: Long-horizon imitation learning by watching human play. In Conference on Robot Learning (CoRL), 2023a.

Ruicheng Wang, Jialiang Zhang, Jiayi Chen, Yinzhen Xu, Puhao Li, Tengyu Liu, and He Wang. Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation. In IEEE International Conference on Robotics and Automation (ICRA), 2023b.

Xinyue Wei, Minghua Liu, Zhan Ling, and Hao Su. Approximate convex decomposition for 3d meshes with collision-aware concavity and tree search. ACM Transactions on Graphics (TOG), 41(4):1–18, 2022.

Zhaoming Xie, Xingye Da, Buck Babich, Animesh Garg, and Michiel van de Panne. Glide: Generalizable quadrupedal locomotion in diverse environments with a centroidal model. In International workshop on the algorithmic foundations of robotics, pages 523–539. Springer, 2022.

Yinzhen Xu, Weikang Wan, Jialiang Zhang, Haoran Liu, Zikang Shan, Hao Shen, Ruicheng Wang, Haoran Geng, Yijia Weng, Jiayi Chen, et al. Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

Wenhao Yan and Jie Chu. Foundationpose++: Simple tricks boost foundationpose performance in highdynamic scenes. https://github.com/teal024/FoundationPose-plus-plus, 2025.

Max Yang, Chenghua Lu, Alex Church, Yijiong Lin, Christopher J. Ford, Haoran Li, Efi Psomopoulou, David A. W. Barton, and Nathan F. Lepora. Anyrotate: Gravity-invariant in-hand object rotation with sim-to-real touch. In Conference on Robot Learning (CoRL), 2024.

Weirui Ye, Fangchen Liu, Zheng Ding, Yang Gao, Oleh Rybkin, and Pieter Abbeel. Video2policy: Scaling up manipulation tasks in simulation through internet videos. arXiv preprint arXiv:2502.09886, 2025.

Patrick Yin, Tyler Westenbroek, Zhengyu Zhang, Joshua Tran, Ignacio Dagnino, Eeshani Shilamkar, Numfor Mbiziwo-Tiapo, Simran Bagaria, Xinlei Liu, Galen Mullins, Andrey Kolobov, and Abhishek Gupta. Emergent dexterity via diverse resets and large-scale reinforcement learning. In International Conference on Learning Representations (ICLR), 2026.

Haoqi Yuan, Ziye Huang, Ye Wang, Chuan Mao, Chaoyi Xu, and Zongqing Lu. Demograsp: Universal dexterous grasping from a single demonstration. In International Conference on Learning Representations (ICLR), 2026.

Hui Zhang, Zijian Wu, Linyi Huang, Sammy Christen, and Jie Song. Robust dexterous grasping of general objects. In Conference on Robot Learning (CoRL), 2025.

Huayi Zhou, Ruixiang Wang, Yunxin Tai, Yueci Deng, Guiliang Liu, and Kui Jia. You only teach once: Learn one-shot bimanual robotic manipulation from video demonstrations. In Robotics: Science and Systems (RSS), 2025.

Xinghao Zhu, Zixi Liu, Shalin Jain, Chenran Li, Milad Noori, Michael Andres Lin, Huihua Zhao, John Welsh, Mrinal Verghese, Wei Liu, Tingwu Wang, Xingye Da, Zhengyi Luo, Vishal Kulkarni, Naema Bhatti, Yuke Zhu, Linxi Fan, Bowen Wen, Danfei Xu, Soha Pouya, and Yan Chang. Learning dexterous manipulation using contact wrench guidance from human demonstration. arXiv preprint arXiv:2607.00033, 2026.

Yifeng Zhu, Arisrei Lim, Peter Stone, and Yuke Zhu. Vision-based manipulation from single human video with open-world object graphs. Autonomous Robots, 2024.

René Zurbrügg, Andrei Cramariuc, and Marco Hutter. Graspqp: Diferentiable optimization of force closure for diverse and robust dexterous grasping. In Conference on Robot Learning (CoRL), 2025.

## Appendix

A Additional Related Works 19   
B Task Abstraction as a Stage-wise Scene Graph 20   
B.1 Predicate Vocabulary 20   
B.2 VLM Query 21   
B.3 Asset Preparation and Functional Regions 26   
B.4 Physical Properties and Functional Region Localization 26   
C Scene-graph-grounded Reinforcement Learning 28   
C.1 Observations 28   
C.2 Rewards 28   
C.3 Reset Generation 30   
C.3.1 Predicate Samplers and Placement Order . 30   
C.3.2 Grasp Synthesis 31   
C.3.3 Transition Feasibility and Physical Validity 31   
C.4 Hierarchical decomposition with a QP low level. 32   
C.5 Simulation Training Details 34   
D Real-World Perception 35   
E Additional Experimental Results 36   
E.1 Generalization to Unseen Initial Object Positions . 36   
E.2 Diverse Grasp Strategies Across Embodiments and Configurations 41   
E.3 Simulation Rollouts Across Diverse Dexterous Embodiments 44

## A. Additional Related Works

Demonstration-seeded RL in simulation. A parallel line makes multi-stage exploration tractable by injecting demonstration-derived information into RL in simulation. Prior methods use this information as an action prior (Torne et al., 2024, Ankile et al., 2025), as object-pose progress targets (Dan et al., 2025), or to generate and refine task-specific reset and reward programs (Ye et al., 2025). In each case, the demonstration guides exploration by specifying how the policy should act, progress, or be rewarded. Instead, we use the demonstration to identify how object relations change at the predicate level encoded in the stage-wise scene graph over time. These changes define the task stages, from which we construct diverse, physically valid reset states across all intermediate stages, including synthesized multi-finger grasps. These stage-wise reset states allow episodes to start directly from intermediate stages of the task, reducing exploration burden. Furthermore, exposing the policy to multiple configurations results in a more generalizable policy. Thus, the video specifies the stage-wise relations the policy must achieve and their temporal order, rather than providing demonstrated actions, motion targets, or a reward program to follow.

Task-Aware Grasp Synthesis and Learned Grasp Acquisition. TaskDexGrasp conditions static dexterous hand-pose synthesis on task semantics, while Dexterous Functional Grasping combines semantic afordances with a learned grasp-acquisition controller (Chen et al., 2024, Agarwal et al., 2023). A broader line learns generalizable grasp proposal, acquisition, or maintenance from synthetic grasps and demonstrations (Xu et al., 2023, Wan et al., 2023, Yuan et al., 2026, Park et al., 2026, Zhang et al., 2025). These methods primarily target robust grasp acquisition across diverse objects, whereas our goal is to solve the full manipulation task. We use grasp synthesis to generate diverse reset states and train the policy via reinforcement learning to acquire the grasp and execute subsequent manipulation stages.

Sim-to-Real Dexterous Manipulation. Simulation-based reinforcement learning has been successfully applied to real-world dexterous in-hand orientation (Andrychowicz et al., 2020). Subsequent work has extended this setting toward faster control (Handa et al., 2023), more accessible hardware, and tactile feedback (Yang et al., 2024). Recent systems move toward longer-horizon interaction. SimToolReal (Kedia et al., 2026) trains a policy to move tools toward target poses and executes unseen tasks by tracking the full 6D tool trajectory extracted from a human video. Because this target pose sequence is assumed available during execution, and our task setting does not provide such trajectory information at execution time, the two settings are not directly comparable. Play2Perfect (Lum et al., 2026) and ADEPT (Lee et al., 2026b) instead focus on reusable dexterous pretraining: Play2Perfect learns from task-agnostic play before fine-tuning on downstream assembly tasks, while ADEPT pretrains generic object-reposing skills and adapts them to insertion and placement. These approaches complement to ours, as they study how pretraining supports downstream manipulation, whereas we focus on learning a new task from a single human video.

Force Closure and Grasp-Quality Objectives. Force closure is widely used to evaluate and synthesize stable multi-finger grasps. Recent methods formulate diferentiable force-closure objectives to generate physically stable dexterous grasps across objects and hand embodiments (Liu et al., 2021, Wang et al., 2023b, Zurbrügg et al., 2025).

Force-closure measures have also been used as reinforcement learning rewards. CHORD (Zhu et al., 2026) introduces a force-closure reward for learning dexterous manipulation when demonstration-derived contact guidance is unreliable. We adopt this force-closure term by augmenting it with a weakest-direction graspquality measure to encourage stable grasps. While concurrent work similarly uses grasp quality as a dense reward to strengthen weak wrench directions during in-hand manipulation (Chen et al., 2026b), we augment the force-closure reward with a weakest-direction grasp-quality measure to acquire stable grasps as part of a multi-stage manipulation task.

## B. Task Abstraction as a Stage-wise Scene Graph

This section details how we obtain the stage-wise scene graph from a human video. Sec. B.1 lists the predicate vocabulary. Sec. B.2 describes how a single VLM query converts the video into the stage-wise scene graph, including the input prompt, an example output, and the query time and cost. Sec. B.3 describes how we reconstruct object assets from the demonstration, estimate their physical properties, and localize their functional regions. The pretrained models used at each step are summarized in Table 2.

Table 2: Pretrained models used in task abstraction, asset preparation, and real-world perception.
<table><tr><td>Step</td><td>Model</td><td>Input → Output</td><td>Sec.</td></tr><tr><td>Scene-graph extraction</td><td>Gemini 3.1 Pro</td><td>video → entities, parts, stage sequence</td><td>B.2</td></tr><tr><td>Instance segmentation</td><td>SAM 3.1</td><td>first frame, visual_phrase → object B.3 mask</td><td></td></tr><tr><td>Mesh reconstruction</td><td>SAM 3D Objects</td><td>RGB, mask, metric point map → metric B.3 mesh</td><td></td></tr><tr><td>Object pose tracking</td><td>FoundationPose++</td><td>RGB-D stream, mesh → 6D pose</td><td>D</td></tr><tr><td>Physical parameters</td><td>GPT-5</td><td>category, size → mass, friction</td><td>B.4</td></tr><tr><td>Part localization</td><td>SAM 3</td><td>rendered views, part name → part region B.4</td><td></td></tr></table>

Table 3: Predicate vocabulary. Geometric features, distance metrics, and constraints used to instantiate the predicates in our tasks. Paired pre-relation and terminal predicates share the same distance metric and difer only in their constraints, except for pre\_grasp/grasp.
<table><tr><td>Predicate</td><td>Reference A</td><td>Target B</td><td>Distance metric</td><td>Constraint</td></tr><tr><td>pre_on_top on_top</td><td>top support disk  $\mathcal { D } _ { A }$ </td><td>bottom point  $p _ { B }$ </td><td> $d _ { \mathrm { t o p } } = d ( p _ { B } , { \mathcal { D } } _ { A } )$ </td><td> $\epsilon _ { \mathrm { t o p } } < d _ { \mathrm { t o p } } \leq \varphi _ { \mathrm { t o p } }$   $d _ { \mathrm { t o p } } \leq \epsilon _ { \mathrm { t o p } }$ </td></tr><tr><td>pre_inside inside</td><td>interior box  $B _ { A }$ </td><td>root point  $p _ { B }$ </td><td> $d _ { \mathrm { i n } } = d ( p _ { B } , { \cal B } _ { A } )$ </td><td> $\epsilon _ { \mathrm { i n } } < d _ { \mathrm { i n } } \leq \varphi _ { \mathrm { i n } }$   $d _ { \mathrm { i n } } \leq \epsilon _ { \mathrm { i n } }$ </td></tr><tr><td>pre_contact contact</td><td>center  $p _ { A }$ </td><td>working point  $p _ { B } ^ { \mathrm { w o r k } }$ </td><td> $d _ { \mathrm { c o n } } = \| p _ { A } - p _ { B } ^ { \mathrm { w o r k } } \| _ { 2 }$ </td><td> $\epsilon _ { \mathrm { c o n } } < d _ { \mathrm { c o n } } \leq \varphi _ { \mathrm { c o n } }$   $d _ { \mathrm { c o n } } \leq \epsilon _ { \mathrm { c o n } }$ </td></tr><tr><td>pre_seat seat</td><td>goal qgoal</td><td>joint position qB</td><td> $d _ { \mathrm { s e a t } } = | q _ { B } - q _ { \mathrm { g o a l } } |$ </td><td> $\epsilon _ { \mathrm { s e a t } } < d _ { \mathrm { s e a t } } \leq \varphi _ { \mathrm { s e a t } }$   $d _ { \mathrm { s e a t } } \leq \epsilon _ { \mathrm { s e a t } }$ </td></tr><tr><td>pre_grasp grasp</td><td>grasp-region  $p _ { A } ^ { \mathrm { g r a s p } }$  grasp set  $\mathcal { G } _ { A }$ </td><td>fingertips  $P _ { B } ^ { \mathrm { t i p } }$  hand configuration  $h _ { B }$ </td><td> $d _ { \mathrm { p r e - g r a s p } } = d _ { \mathrm { t i p } } ( P _ { B } ^ { \mathrm { t i p } } , p _ { A } ^ { \mathrm { g r a s p } } )$ </td><td> $\ell _ { g } \leq d _ { \mathrm { p r e - g r a s p } } \leq u _ { g }$   $\bar { h _ { B } } \in \bar { \mathcal { G } } _ { A }$ </td></tr></table>

Notation. $d ( p , S ) = \mathrm { m i n } _ { s \in S } \| p - s \| _ { 2 }$ denotes the Euclidean distance from a point p to a set S. $\mathcal { D } _ { A }$ and $\overline { { { B } _ { A } } }$ denote the top support disk and interior box of reference entity A, respectively. For relation $c , \epsilon _ { c }$ denotes its satisfaction tolerance, while φ denotes the outer sampling bound of the corresponding pre-relation. $P _ { B } ^ { \mathrm { t i p } }$ denotes the fingertip positions, and $d _ { \mathrm { t i p } }$ denotes the fingertip-to-grasp-region-center distance used for pre\_grasp. $\ell _ { g }$ and $u _ { g }$ denote its lower and upper sampling bounds. $\mathcal { G } _ { A }$ denotes the verified grasp set for object $A ,$ and h denotes the hand configuration.

## B.1. Predicate Vocabulary

Table 3 lists the predicates in our vocabulary, which cover common relations in tabletop manipulation. Each geometric predicate also has an approach form, denoted by the prefix pre-, which holds while the relation is being approached but not yet established.

## B.2. VLM Query

Given the single human demonstration of each task (Fig. 8), recorded at 640 × 480, we query Gemini 3.1 Pro (gemini-3.1-pro-preview) (Gemini Team, Google, 2026) with the entire video at medium media resolution; videos under 20,MB are sent inline, and larger ones through the file-upload API. The prompt (Listing 1) is identical for all tasks: the only task-specific input is the video, and the vocabulary in the prompt is the one in Table 3. A single query takes 12–21s and costs less than \$0.05, so extracting all five task graphs takes under 90s and \$0.20 in total (Table 6).

## Time

(a) Doll  
![](images/7931efe1ddb050c3cbb6c522dbf68248d657c9eb9911fde52fe34d0f7ee8792e.jpg)  
Figure 8: Single human demonstrations for the five tasks. Each row shows the human video used to specify one task, with frames ordered from left to right over time.

Table 4: Task suite. Object roles: held is grasped by the robot, site receives another object, acted-on is manipulated only through a tool, andfixture holds an object at the start. The predicate chain extracted by the VLM defines K stages, each with its own reset distribution. Objects marked with <sup>†</sup> use a CAD mesh instead of a reconstructed one (Appendix B.3).
<table><tr><td>Task</td><td>Description</td><td>Objects (role)</td><td>K</td><td>Predicate chain (stage 1 → K)</td><td>Type</td></tr><tr><td>Doll</td><td>Pick up a Pikachu plush toy and place it inside an orange pot.</td><td>pikachu (held), pot† (site)</td><td>5</td><td>∅ → pre-grasp(hand, pikachu) → grasp(hand, pikachu) → pre-inside(pikachu, pot)</td><td>direct</td></tr><tr><td>Can</td><td>Pick up the coke can and place it in a bucket.</td><td>can (held), bucket† (site)</td><td>5</td><td>∅ → pre-grasp(hand, can) → grasp(hand, can) → pre-inside(can, bucket)</td><td>direct</td></tr><tr><td>Stamp</td><td>Grasp the stamp and press it down onto the stack of papers.</td><td>stamp† (held), paper (site)</td><td>5</td><td>∅ → pre-grasp(hand, stamp) → grasp(hand, stamp) → pre-on_top(stamp, paper) → success: on_top(stamp, paper)</td><td></td></tr><tr><td></td><td>Use a hammer to strike a peg inserted into a block.</td><td>hammer† (held), peg† (acted-on), block† (site), holder† (fixture)</td><td></td><td>fixed_to(hammer, holder) → pre-grasp(hand, hammer) → grasp(hand, hammer) → pre-contact(hammer, peg) → contact(hammer, peg)</td><td>tool-use</td></tr><tr><td>Sweep</td><td>Sweep the toy cat into the dustpan using the brush.</td><td>brush (held), toy (acted-on), dustpan (site)</td><td>7</td><td>→ success: seated(peg, block) ∅ → pre-grasp(hand, brush) → grasp(hand, brush) → pre-contact(brush, toy) → contact(brush, toy)</td><td>tool-use</td></tr></table>

Table 5: Structure of the VLM output and the use of each field in our pipeline. Indented entries are fields of the object above them, and [] marks a list with one entry per object, mount, or stage.
<table><tr><td>Field</td><td>Content</td><td>Used for</td></tr><tr><td colspan="3">task name, description</td></tr><tr><td></td><td>Task name and one-line summary</td><td>Generated first to condition the subsequent fields; not used by the pipeline</td></tr><tr><td colspan="3">objects[] id</td></tr><tr><td>category</td><td>Identifier of each task-relevant object Object class</td><td>Nodes; key for all per-object data Hint in the physical-parameter query</td></tr><tr><td>visual_phrase roles</td><td>Text description of the object&#x27;s appearance Role of the object (e.g., held, site)</td><td>Object segmentation Scene slots in reset generation and RL</td></tr><tr><td>upright grasping-part, working_part</td><td>Whether the object is normally kept upright Names of functional parts</td><td>Restricts sampled stable poses to the upright class Text prompts for functional-region localization</td></tr><tr><td>working_direction</td><td>How a tool meets its target (down or horizontal)</td><td>Contact placement</td></tr><tr><td colspan="3">scene_graph initial_state[]</td></tr><tr><td></td><td>Relations that already hold in the first frame</td><td>First stage graph; kept in the resets of later stages until the related object moves</td></tr><tr><td>stages []</td><td>One predicate per stage, in temporal order</td><td>Stage sequence for backward reset generation; the last stage sets the near-goal target</td></tr><tr><td>goal_condition</td><td>Predicates that must all hold at success</td><td>RL success test (first predicate)</td></tr><tr><td></td><td></td><td></td></tr></table>

The VLM returns a JSON object following a fixed response schema, whose fields and their use in our pipeline are summarized in Table 5. Listing 2 shows the output for the Sweep task, in which a brush is used to push a toy cat into a dustpan, and Table 4 summarizes the resulting objects, roles, and stage sequences for all five tasks. Table 4 lists the objects, their roles, and the stage sequence extracted by the VLM from each video.

Listing 1: VLM prompt for stage-wise scene graph extraction. {predicate\_list} is filled with the predicates of Table 3.   
You are analyzing a human demonstration video for robot learning (real2sim).   
The human hand in the video corresponds to a ROBOT END-EFFECTOR. Refer to it as "ee".   
Do NOT list the hand as an object; it is the agent, not a manipulable object.   
Extract, grounded ONLY in what is visually observable:   
1. TASK: concise name and one-line description.   
2. OBJECTS: the manipulable objects of interest (NOT the hand). For each: a short id,   
a category, a visual\_phrase, and roles.   
category vs visual\_phrase –- these are DIFFERENT and both required:   
- "category": the clean object class used to search an asset library ("kettle", "mug",   
"knife"). A superordinate is fine here.   
- "visual\_phrase": what you would type into an open-vocabulary SEGMENTATION model to   
select exactly this object in the video frame. 2-4 words, concrete and appearance  
bearing: colour and/or material plus the specific noun ("red enamel kettle", "white   
ceramic mug", "wooden cutting board"). NEVER an abstract superordinate ("container",   
"object", "item", "tool") –- those ground to nothing. If two objects in the frame share   
a noun, the phrase must be what tells them apart.   
Do NOT include the workspace surface itself (worktable, desk, floor, walls) –- it   
is part of the robot cell, not an object. But DO include a fixed object that holds   
another object off that surface (a knife holder, a dish rack, a tray): it is not   
manipulated, yet the scene is wrong without it.   
Likewise include every object that another object is, or ends up, "seated" in   
(inserted, driven, or screwed into it: the bottle a cork is pressed into, the   
socket a plug is pushed into), even if the hand never touches it.   
roles –- assign ALL that apply, judged from what visibly happens in the video:   
- "held": the hand grasps and holds this object at some point   
(both directly-placed objects and tools count).   
- "site": another object comes to rest ON or IN this object   
(plate under food, cutting board, container receiving something).   
- "acted\_on": a HELD object acts on this object through contact   
(food being cut, dough being rolled, a window being wiped).   
- "fixture": this object is already holding another object up when the demo   
begins, and the hand never touches it (a holder, a rack, a tray).   
"site" and "fixture" differ in WHEN: a site receives an object as a result   
of the manipulation, a fixture is holding one before it starts. The plate a   
cake is placed on is a site; the holder the knife rests in is a fixture.   
A pot that is carried and later receives food is ["held", "site"].   
Also determine per object:   
- "upright": true if the object is normally stored/used standing upright   
(cup, bottle, vase, jar → true; plate, cutting board, spatula, knife → false).   
- "grasping\_part": for "held" objects, the part the hand grasps.   
- "working\_direction": for "held" tool-use objects, how the working part moves   
against the target at the moment of contact –- "down" if the tool is raised and   
brought down onto it (pestle, potato masher, cookie cutter), "horizontal" if the tool is pushed   
sideways across a surface (squeegee, spatula, rolling pin). This decides the   
contact pose the sampler builds; getting it wrong produces a tool that slides sideways   
when it should come down. null if not tool-use.   
"working\_part": for "held" objects that physically contact another   
object to do work (tool-use), the functional surface that does the work.   
null if the held object is placed directly (not tool-use).   
PART NAMING –- these two strings are fed verbatim to an open-vocabulary   
segmenter as text prompts. Name the SMALLEST region that actually carries   
the part's role, and describe ONLY that region.

Write "<attribute> <part> of the <object>" with EXACTLY ONE attribute, and   
that attribute must belong to the region you are naming –- not to anything   
attached to it. If the handle is the part, describe the handle alone: its   
own colour, its own material. Say nothing about the head, the shaft, the   
body, or the object's overall look, even when those features are the most   
obvious thing in the demo. Every word describing a neighbouring region is   
an invitation for the mask to spread into that region.   
Keep the <object> slot bare –- the plain object noun, never "the red mug"   
or "the wooden-handled knife".   
NEVER enumerate. If the region itself shows two colours, pick the one that   
covers most of it –- do not write both. A colour the region shares with a   
neighbouring region is fine: name the region as it actually looks. Never   
borrow a colour from an adjacent part just to stand apart from it.   
good: "blue handle of the screwdriver" (grip region only)   
"black handle of the spatula" (same black as the blade –- still   
right, that is what it looks like)   
"rubber blade of the squeegee" (blade only, not the frame)   
"metal blade of the knife"   
"black grip of the pan"   
bad: "handle" (no attribute –- bleeds outward)   
"silver handle of the ladle" (silver is the neck; the grip   
the hand holds is black –- a colour   
borrowed from the neighbour)   
"black handle of the silver pan" (attribute on the object –-   
mask jumps to the body)   
"wooden handle of the metal knife" (describes the blade too)   
"blue and red handle of the screwdriver" (enumeration –- matches   
the whole screwdriver)   
"blue rubber ergonomic handle of the screwdriver" (piled attributes)   
Use only attributes actually visible in the demo; never invent a colour.   
If no attribute applies to the region on its own, write just   
"<part> of the <object>". null if not applicable.   
3. SCENE\_GRAPH: express it with PREDICATES ONLY (do not write free-form sentences),   
in three parts: initial\_state, goal\_condition, stages.   
initial\_state: what is already mounted on what in the FIRST frame, before the   
hand moves. Most demos start with everything simply lying on the table –- in   
that case return an empty list. Emit an entry only when an object begins the   
demo \*\*held off the table by another object\*\* in a definite resting pose: a   
ladle hanging on its rack, a part seated in a tray, a bit sitting in a holder.   
The only predicate here is "fixed\_to": a is the object being held, b is the   
thing holding it. Nothing else –- this list is not about what the hand does.   
- arg 'a' must be an object that LATER MOVES (it appears in the   
stages). An object that never moves needs no entry: it is simply part   
of the scene.   
- arg 'b' is the holder. Give it the "fixture" role in OBJECTS.   
- Do not describe an object resting directly on the table or the ground.   
- Do not describe a part that is already inserted, driven, or screwed into   
another object. That is a joint, not a mount, and it is not expressed here.   
goal\_condition and stages –- allowed predicates: {predicate\_list}.   
goal\_condition: the set of predicates that must ALL hold at success (goal state).   
stages: the relations that become established during the demo, one per entry,   
in the temporal order in which they are established. List only relations that   
become TRUE; do NOT add entries for approaching, reaching, or moving toward   
something –- approach phases are derived automatically from these relations.   
Use "ee" as arg 'a' for grasp.   
Be precise about when each relation is actually established (e.g. distinguish   
real contact from the tool merely hovering near the target).   
AGENCY TRANSFER: when a source acts on a target through "grasp" or "contact", the   
target becomes arg 'a' of the subsequent stages –- it is the thing that moves   
next, not the original actor. (ee grasps spatula → spatula becomes arg 'a'; spatula contacts   
pancake → pancake becomes arg 'a' for the move onto the plate.)

Choose predicates from the list; never invent new ones.   
- "inside" vs "seat": use "seat" ONLY when the object ends up DRIVEN/PRESSED INTO   
the other one so the two become a single rigid assembly (a cork pressed into a   
bottle, a plug pushed into a socket, a drawer pushed shut). Use "inside" when the   
object merely ends up within an open container and could be lifted straight out   
(fruit placed in a basket, an item dropped in a bin).   
(grasp->grasp, contact->contact, place->on\_top/inside/seat).

Do not hallucinate objects not present. Return only valid JSON matching the schema.

## Listing 2: VLM output for the Sweep task

```jsonl
{
"task": {
"name": "sweep toy into dustpan",
"description": "Use a brush to sweep a small plush toy into a dustpan."
},
"objects": [
{
"id": "brush",
"category": "brush",
"visual_phrase": "blue brush",
"roles": ["held"],
"upright": false,
"grasping_part": "black handle of the brush",
"working_part": "blue bristles of the brush",
"working_direction": "horizontal"
},
{
"id": "dustpan",
"category": "dustpan",
"visual_phrase": "blue dustpan",
"roles": ["site"],
"upright": false,
"grasping_part": null,
"working_part": null,
"working_direction": null
},<sub>{</sub>
"id": "toy",
"category": "plush toy",
"visual_phrase": "grey cat plush",
"roles": ["acted_on"],
"upright": true,
"grasping_part": null,
"working_part": null,
"working_direction": null
}
],
"scene_graph": {
"initial_state": [],
"stages": [
{"step": 1, "predicate": "grasp", "args": {"a": "ee", "b": "brush"}},
{"step": 2, "predicate": "contact", "args": {"a": "brush", "b": "toy"}},
{"step": 3, "predicate": "inside", "args": {"a": "toy", "b": "dustpan"}}
],
"goal_condition": [
{"predicate": "inside", "args": {"a": "toy", "b": "dustpan"}}
]
}
}
```

Table 6: VLM query cost per task.
<table><tr><td>Task</td><td>Video [s]</td><td>Latency [s]</td><td>Cost [$]</td></tr><tr><td>Doll</td><td>7.07</td><td>21.2</td><td>0.045</td></tr><tr><td>Can</td><td>5.30</td><td>11.5</td><td>0.024</td></tr><tr><td>Stamp</td><td>4.97</td><td>12.0</td><td>0.026</td></tr><tr><td>Hammer</td><td>4.00</td><td>18.7</td><td>0.039</td></tr><tr><td>Sweep</td><td>5.77</td><td>18.5</td><td>0.037</td></tr><tr><td>Total</td><td>27.11</td><td>81.9</td><td>0.171</td></tr></table>

## B.3. Asset Preparation and Functional Regions

We reconstruct each task-relevant object from the first frame of the demonstration; the full video is used only by the VLM query. The object’s visual phrase from the VLM output serves as the text prompt for SAM 3.1 (Carion et al., 2026), which returns the object’s instance mask. Given the RGB image, the mask, and the metric point map from the depth image of the same frame, SAM 3D Objects (Chen et al., 2026a) reconstructs the object mesh. For objects marked in Table 4, we instead use an available CAD mesh.

Each mesh, reconstructed or CAD, is post-processed into a watertight simulation mesh and a set of convex pieces (Table 7). The convex pieces serve as collision geometry for grasp synthesis, and the watertight mesh serves as the collision mesh in Isaac Sim (Mittal et al., 2025).

Table 7: Mesh post-processing.
<table><tr><td>Step</td><td>Tool</td><td>Parameters</td></tr><tr><td>Convex decomposition</td><td>CoACD (Wei et al., 2022)</td><td>concavity threshold 0.05; resolution 2000; MCTS 20 nodes, 150 iterations, depth 3;  $r v _ { k } = 0 . 3 ;$  manifold preprocessing only for non-manifold input (resolution 50); hull merging on; no hull-count limit; minimum part volume 10−10 (-mv); random seed from the system clock</td></tr><tr><td>Watertight remesh</td><td>VDB (Museth et al., 2013))</td><td>CoACD manifold mode (Open- applied to the normalised mesh; dual-marching-cubes level set 0.1</td></tr><tr><td>Simplification</td><td>2004)</td><td>ACVD (Valette and Chassery, 2000 vertices, gradation 1.5, manifold output</td></tr></table>

## B.4. Physical Properties and Functional Region Localization

The VLM query provides only semantic attributes of each object (Table 5); physical parameters and functional regions are obtained as follows.

Physical parameters. Mass, mass range, and friction coeficient are estimated after scene-graph extraction by a separate text-only language-model query (gpt-5 (Singh et al., 2025)), which receives the id, category, and measured longest axis of each object(Listing 3).

Here, {obj\_lines} holds one line per object: its id, category, and measured longest axis in cm. During RL, the object mass is sampled from $[ m - \Delta m , m + \Delta m ]$ and the static friction from [0.9<sub>µ</sub>, 1.1<sub>µ</sub>] with a floor of 0.05, and the dynamic friction is set to 0.9 times the static friction (Table 10). Physical verification (in Sec. C.3.3) use the worst case of these ranges, namely the maximum mass and the minimum friction.

Table 8: Localization of grasping and working regions.
<table><tr><td>Step</td><td>Setting</td></tr><tr><td>Views</td><td>34 views: elevations —35°, 10°, 35°, 60°× 8 azimuths (45° steps), plus top (85°) and bottom (—85°); rendered twice (textured and uniform grey); 768 ×768 px, vertical field of view 45°, camera distance 2.6× the half bounding-</td></tr><tr><td>Detection</td><td>box diagonal SAM 3; the part string is one text concept; instance score floor 0.15, mask threshold 0.5</td></tr><tr><td>Negative concepts</td><td>grasping only: blade, cutting edge, spike, bristles; their masks are dilated by 12 px and removed</td></tr><tr><td>Fusion</td><td>a vertex is visible in a view if its depth is within 2% of the bounding-box diagonal of the z-buffer; score = positive votes / visible views, over views with at least one detection; vertex kept if seen in ≥ 2 views and score &gt; 0.3</td></tr><tr><td>Transfer to mesh</td><td>each simulation-mesh face takes the maximum score of the nearest triangle of the reconstructed mesh; face kept if score ≥ 0.25; grasping region: largest connected component</td></tr><tr><td>Outputs</td><td>per-face mask and object-frame bounding box (centre, extents) of the grasping region; for grasp synthesis on the table, the mask is restricted to faces whose lowest vertex is ≥ 5 mm above the table</td></tr><tr><td>Working surfaces</td><td>working-region faces clustered by adjacency with normal deviation ≤ 5°; area-weighted plane fit per cluster; clusters with RMS residual ≤ 6 × 10−5 m and area ≥ 90% of the largest are kept (centre, normal, half-extents)</td></tr></table>

Listing 3: Prompt for physical-parameter estimation   
This is a domain-randomization setup for a robot simulation. The objects   
below were 3D-scanned from real instances, so the longest-axis size is already fixed by   
measurement -- do NOT re-estimate the size.   
For each object, answering as "an object of that category whose measured size is that value",   
give:   
1. mass\_g: typical mass at that size (g, empty / dry weight).   
2. mass\_range\_g: +/- variation of the mass (g).   
3. friction\_coeff: static friction coefficient against a flat surface   
(plastic \~0.3, wood \~0.4, rubber \~0.8, ceramic \~0.4, metal \~0.3, fabric \~0.5).   
Objects:   
{obj\_lines}   
Answer in JSON, keeping each id exactly as given.

Grasp and working regions. We localize each part name from the VLM output on the object mesh in three steps (Table 8). We first render the mesh from 34 viewpoints around the object and segment the part in each view with SAM 3, using the part name as a text prompt. We then project the 2D masks back onto the mesh and keep the vertices that are detected in most of the views in which they are visible. Finally, the grasping region is taken as the largest connected component, and the working region is fitted with planar surfaces that represent the tool’s working faces. To keep tool heads out of the grasping region, masks of generic tool-head concepts (e.g., blades and bristles) are removed from it. The grasping region restricts grasp synthesis and provides the approach target of the reward, and the working surfaces are used by the contact sampler and the tool reward.

Support patches. Support patches are the upward-facing flat regions of a mesh on which another object can rest. Faces whose normal has a vertical component above 0.99 (within about 8<sup>◦</sup> of vertical) are grouped by height with a 5 mm tolerance, and groups with less than 5% of the largest area are dropped. Each patch stores its height, centre, and radius.

## C. Scene-graph-grounded Reinforcement Learning

In this section, we make an detail explanation for Sec. 3.2.

## C.1. Observations

Scene-graph-grounded observation construction. The scene graph identifies the task-relevant entities, their roles, relations, and functional regions. We map the demonstrated hand to the robot end-efector and assign the remaining objects according to their roles in the task. For direct manipulation, the observation contains the end-efector, the manipulated object, and the receptive object. For tool-use manipulation, the held object is instead the tool, and an additional acted-on (interactive) object is introduced (Fig. 9).

Object and functional-part nodes provide their local geometry, such as object size, grasp regions, and working surfaces. Functional parts are associated with the hand or another object according to their role in the interaction. We represent a grasp region by its center and bounding-box extents, and a working surface by its center, unit normal vector of its surface, and the extent of the surface.

Each scene-graph edge contributes a relative 6D pose between its connected entities. Importantly, this relative pose is defined according to the predicate encoded by the edge: the predicate determines which geometric features of the two entities are used, following the same feature definitions used for the predicate distance functions. Thus, an edge may relate object centers, functional regions, or other predicate-specific features rather

![](images/308c3f5dc853bd948aba932594b5766103d78ce92c0fef1c31c9e6c83008032b.jpg)  
Figure 9: Scene-graph-grounded observation. The policy observation is constructed from scene-graph nodes and their relative 6D-pose edges for (a) direct and (b) tool-use manipulation.

than simply the two object centers. All node, part, and edge features are concatenated to form the policy observation.

## C.2. Rewards

The grasp predicate is evaluated in 6D wrench space rather than by a Cartesian distance. Following CHORD (Zhu et al., 2026), we approximate the friction cone at each contact with a set of edge forces, each inducing a 6D primitive wrench consisting of force and torque. We then evaluate how well these primitive wrenches support a set of sampled directions in wrench space. Let $\sigma _ { j , t }$ denote the support along sampled direction j at time t. CHORD’s force-closure reward measures the fraction of directions whose support exceeds a threshold:

$$
r _ { \mathrm { f c } } ( s _ { t } , a _ { t } ) = \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \mathbb { I } [ \sigma _ { j } > \epsilon ] .
$$

where B is the number of sampled wrench directions and ϵ is the minimum support threshold.

In our setting, we use $r _ { \mathrm { f c } }$ to guide grasp acquisition without a retargeted hand trajectory or virtual-object controller (Mandi et al., 2026). Directional coverage alone, however, can increase for contacts that do not form a stable grasp; for example, the policy may push the brush against the toy cat instead of first grasping

the brush. We therefore augment $r _ { \mathrm { f c } }$ with the Ferrari–Canny grasp quality (Ferrari and Canny, 1992), which measures support along the weakest wrench direction:

$$
Q _ { t } = \operatorname* { m i n } _ { j } \sigma _ { j t } .\tag{1}
$$

The resulting grasp reward is

$$
r _ { \mathrm { g r a s p } } ( o _ { t } , a _ { t } ) = r _ { \mathrm { f c } } ( o _ { t } , a _ { t } ) + \mathbb { I } \left[ N _ { \mathrm { f i n g e r } , t } \geq 2 \right] \operatorname* { m a x } ( Q _ { t } , 0 ) ,\tag{2}
$$

where $N _ { \mathrm { f i n g e r } , t }$ is the number of distinct fingers in contact with the object. The additional quality term favors stable multi-finger grasps by encouraging support even along the weakest wrench direction.

Progress rewards from spatial predicates. For predicates whose relations can be expressed as a Cartesian distance, we define dense progress rewards from reductions in that distance. Following Kedia et al. (2026), we measure progress relative to the smallest distance reached so far in the episode. For $\mathtt { p r e - g r a s p } ( A , B )$ let $d _ { A B , t }$ denote the current distance and $d _ { A B , t } ^ { * }$ the smallest distance reached up to time t. We define

$$
r _ { \mathrm { p r e - g r a s p } , t } ( A , B ) = \operatorname* { m a x } \big ( d _ { A B , t - 1 } ^ { * } - d _ { A B , t } , 0 \big ) ,
$$

so that reward is produced only when the policy makes new progress toward the relation. Using improvement over the best distance reached so far avoids rewarding stationary behavior: once the distance stops decreasing, the progress reward becomes zero. We use the same progress construction for the other distance-based predicates. For contact predicate, we additionally include an alignment term between the tool’s working region and the desired direction of interaction.

Task reward composition. We next describe how the reward terms are combined during training. We use a grasp indicator $\mathbb { I } _ { \mathrm { h o l d } , t }$ to determine whether the object is currently lifted by a valid grasp. Using the same grasp-quality criterion as above, we define

$$
\begin{array} { r } { \mathbb { I } _ { \mathrm { h o l d } , t } = \mathbb { I } \left[ N _ { \mathrm { f i n g e r } , t } \ge 2 \right] \mathbb { I } [ Q _ { t } > 0 ] \mathbb { I } [ z _ { t } - z _ { 0 } > h _ { \mathrm { l i f t } } ] , \qquad h _ { \mathrm { l i f t } } = 5 \mathrm { c m } , } \end{array}\tag{3}
$$

Thus, $\mathbb { I } _ { \mathrm { h o l d } , t }$ is active only while at least two distinct fingers are in contact, the grasp provides positive support along its weakest wrench direction and the target is lifted above $h _ { \mathrm { l i f t } }$

Let $g$ denote the first stage whose scene graph contains a grasp predicate, and let ${ \boldsymbol { r } } _ { i , t }$ denote the reward driven by the predicates associated with stage $\mathcal { G } _ { i }$ . At the grasp stage, we additionally use a lift reward before the hold condition is satisfied:

$$
r _ { g , t } = r _ { \mathrm { g r a s p } , t } + \left( 1 - \mathbb { I } _ { \mathrm { h o l d } , t } \right) r _ { \mathrm { l i f t } , t } .\tag{4}
$$

The complete task reward is then

$$
r _ { t } = \sum _ { i \leq g } r _ { i , t } + \mathbb { I } _ { \mathrm { h o l d } , t } \sum _ { i > g } r _ { i , t } + r _ { \mathrm { s u c c e s s } , t } + r _ { \mathrm { r e g } , t } ,\tag{5}
$$

where rewards through the grasp stage are always available, while rewards associated with subsequent manipulation stages are activated only when the object satisfies the hold condition in Eq. 3. Here, $r _ { \mathrm { s u c c e s s } , t }$ is the terminal success bonus, and $r _ { \mathrm { r e g } , t }$ penalizes large actions and joint velocities.

![](images/c97918a3d5213be3f35b2f428ac58b07942e8bfedca2e5d4f9c8901578fe1df7.jpg)  
Figure 10: Scene-graph-constrained reset generation. Given the stage-wise scene graph and the assets, we instantiate each task stage as diverse reset states. The object and hand assets provide geometry and physical properties, while functional part nodes identify regions such as grasp and working regions. Together, these are used to construct a grasp set, restricted to the specified grasp region when available. For each stage, predicates in the scene-graph act as generative constraints: geometric predicates determine where related entities can be sampled, while the grasp predicate samples a valid hand–object configuration from the grasp set. Repeating this process produces diverse simulation reset states that satisfy the relations specified by each stage graph.

## C.3. Reset Generation

This section details how we construct the stage-wise initial-state distribution $\rho _ { 0 }$ of Sec. 3.2. Fig. 10 summarizes the process. Each stage graph $\mathcal { G } _ { k }$ specifies which relations must hold at that point of the task but leaves object poses and grasps free, so many geometrically distinct states satisfy it. We generate such states in two steps. First, we sample states that satisfy the predicates of the stage graph, using each predicate as a rule for placing one node relative to another (Sec. C.3.1); the grasp predicate is instantiated from a set of synthesized dexterous grasps (Sec. C.3.2). Second, we keep only the states that remain usable for the rest of the task and are physically valid in simulation (Sec. C.3.3). We retain 10,000 verified states per stage, which form its reset distribution. The resulting states are therefore not random configurations of the scene, but diverse instantiations of the same task structure.

## C.3.1. Predicate Samplers and Placement Order

Each predicate in Table 3 is defined on geometric features of its two nodes, and the same features determine where one node can be placed relative to the other. For example, on\_top restricts the bottom point of the source object to the support patch of the target, and inside restricts the root point of the source object to the interior box of the target. We use these conditions as placement rules: to satisfy a predicate, we sample the pose of its source node within the region that its target node and the predicate define.

To construct a full stage, we resolve its predicates in dependency order along the edges of $\mathcal { G } _ { k }$ . The anchor, a node that is the target but not the source of any predicate, is placed first. We then follow the edges from the anchor and place each source node relative to its already placed target. Finally, unconstrained nodes, which appear in no predicate of the stage, are sampled in the remaining free space. Fig. 10 illustrates this process for two stages of the Sweep task. The stage in the top row contains contact(brush, toy cat) and grasp(hand, brush). The toy cat is the anchor, so it is placed first; the brush is then placed relative to it so that its bristles touch the toy cat, and the hand is placed on the brush handle at a grasp from ${ \mathcal { H } } _ { \mathrm { g r a s p } }$ (Sec. C.3.2). The dustpan appears in no predicate of this stage and is sampled last in the remaining free space. The stage in the bottom row additionally contains pre-inside(toy cat, dustpan), so the dustpan becomes the anchor: it is placed first, the toy cat is placed relative to it without being inside it, and the brush and the hand follow as above. For the final stage, we first sample a state that satisfies the goal predicate and then perturb it into a near-goal state, following OmniReset (Yin et al., 2026), since an already successful reset provides less useful learning signal.

## C.3.2. Grasp Synthesis

A valid grasp is critical for stage-wise resets: every stage that contains grasp presupposes a securely held object, so an invalid grasp invalidates all of them. We synthesize grasp candidates with Dexonomy (Chen et al., 2025a), which takes a robot hand, an object mesh, and a grasp type from a 33-type taxonomy, and returns diverse grasps of that type. When the object has a grasp region (Appendix B.4), we restrict the synthesis to that region, so that, for example, the brush is grasped at its handle rather than its bristles. For each grasp, we also store the finger command $q _ { \mathrm { c m d } }$ that closes the hand on the object and restore it at initialization together with the hand configuration. The corresponding pre-grasp state is obtained by linearly interpolating the finger joints toward the open configuration, to 30–90% of the grasp configuration, and retracting the hand by 1–5 cm along its approach axis.

## C.3.3. Transition Feasibility and Physical Validity

Satisfying the predicates of one stage does not guarantee that a state remains usable later in the task. We therefore check compatibility at the grasp and stage levels, and verify every state physically.

Grasp level. A grasp that is stable in isolation may still place the hand in a configuration from which the rest of the task cannot be completed. Some grasps are valid when the object is picked up but fail in later stages. For example, when an object must be placed inside a narrow container, a grasp that wraps the whole object holds it securely at pickup, but brings the fingers into collision with the container at the goal. Because every stage after the grasp inherits the same grasp, such a grasp makes all of these stages unusable. We therefore select grasps by their compatibility with the whole task rather than by stability alone. For each grasp type, we instantiate its candidates in every stage that contains grasp and apply the physical checks below. We rank the grasp types by the fraction of candidates that pass across all such stages, and the candidates of the top-n types $( n = 4 )$ form the grasp set $\mathcal { H } _ { \mathrm { g r a s p } }$ . Grasp types that hold the object but cannot attain the goal relation, such as the one in Fig. 11 (3), receive a low pass rate and are excluded. The resulting grasp set is therefore conditioned on the entire task, not only on how well each grasp holds the object.

Stage level. Sampling each stage independently can produce states that satisfy their own stage graph yet cannot lead to the next stage. For example, a pre-grasp state might place a pan with its handle pointing away from the robot, where no verified grasp of the next stage is reachable. Such states are valid locally but ill-posed for the task as a whole. We prevent them by generating the stages backward, starting from the goal stage. After the states of stage k are verified, we carry the poses of their anchor objects and their grasps into stage k − 1, and resample only the remaining entities under the predicates of stage k − 1. Each state of stage k − 1 is thus constructed around a physically verified state of stage k rather than in isolation, so the policy can always progress from one stage to the next.

![](images/f285bdf137ae5b46cb73e87fddd77d1a554b3747fe79da87d31f978dc4e4738f.jpg)

![](images/66b7e760c0848929685f9f820d7b486cd1b22708c336c5a936e983f54f316849.jpg)

![](images/9f92ecb60f049aadb6876daaf3e564450a2c760c4c525a33935c88c6b71e8f39.jpg)  
Figure 11: Physical verification of reset candidates in the Sweep task. (1) and (2) (top: pass, bottom: fail), and (3) is enforced over grasp types. (1) Scene collision: the robot makes no invalid contact with the table or other objects; the failing candidate collides with the dustpan. (2) Reset stability: objects settle under gravity and the grasped object stays in the hand under external wrenches; the failing grasp drops the brush. (3) Goal compatibility: the goal relation remains attainable with the candidate grasp; the failing candidate cannot bring the toy cat inside the allowed region of the dustpan.

Physical validity. Every candidate state is instantiated in simulation with the full robot and must pass two checks (Fig. 11). (1) Scene collision: the robot makes no invalid contact with the table or other objects. (2) Reset stability: at the end of a 6 s episode, all objects have been at rest for the last 2 s, and in grasped states the object is still held by at least two hand links. Goal compatibility is enforced at the grasp level rather than per state: grasp types whose candidates cannot attain the goal relation fail these checks in the goal stage and are removed by the ranking above (Fig. 11 (3)). All checks use the worst-case physical parameters of the domain randomization, the maximum mass and the minimum friction (Appendix B.4), so that verified states remain valid across the randomized range. Grasp robustness is verified once per grasp before reset generation: with only the hand and the object, the grasp must hold the object under gravity in six directions and under an external force of min(0.5 m g, 0.5 N) with a 0.02 m lever along each of the six axes. The pass rate of each grasp type across all grasped stages provides the ranking at the grasp level above, and only states that pass all checks are retained and propagated backward to the previous stage.

## C.4. Hierarchical decomposition with a QP low level.

Zero-shot sim-to-real deployment can expose the robot to hardware-specific constraints that are dificult to anticipate during simulation training. For example, excessive joint velocities may lead to unsafe motions, while contacts that are benign in simulation can damage the physical system. In our setup, contact between the Wuji 1 hand coupler and the table can disengage the coupler from the robot.

To accommodate deployment-time safety constraints without retraining the policy, we separate learned taskspace control from constrained joint-space execution. Following prior work (Xie et al., 2022, Gangapurwala et al., 2022, Bang et al., 2024), we let the RL policy act in task space and delegate kinematic resolution to a model-based low-level controller. Specifically, we adopt the QP velocity-IK formulation of Lee et al. (2026a), which maps the policy’s palm and fingertip velocity commands to joint velocities subject to kinematic and safety constraints.

![](images/5bcdc88419d58b86c5a88727c69c523e64324d73433ded8835b6a00ce96efbe5.jpg)  
Figure 12: QP constraints address a hardware-specific sim-to-real gap. Hand–table contact is benign in simulation because coupler disengagement is not modeled, but the same contact can disengage the Wuji 1 hand coupler on the physical robot. Without the QP low-level controller (top), the learned policy encounters this failure mode. With the QP controller (bottom), a hand–table clearance constraint prevents the collision without retraining the task-space policy.

Controller formulation. We write superscripts for the frame in which a quantity is expressed and subscripts for the associated body or point. B and P denote the robot base and palm frames, and $\mathcal { F }$ denotes the set of fingertips. The policy outputs

$$
a = \left( \xi _ { P } ^ { B } , \{ v _ { i } ^ { \mathcal { P } } \} _ { i \in \mathcal { F } } \right) ,\tag{6}
$$

where $\xi _ { P } ^ { B } \in \mathbb { R } ^ { 6 }$ is the desired palm twist expressed in the robot base frame and $\boldsymbol { v } _ { i } ^ { \mathcal { P } } \in \mathbb { R } ^ { 3 }$ is the desired linear velocity of fingertip i relative to the palm. The resulting action space has dimension $6 + 3 | \mathcal { F } |$

The low-level QP computes the joint velocity that best realizes these task-space commands:

$$
\begin{array} { r l } & { \dot { q } _ { \mathrm { d e s } } = \underset { \dot { q } } { \arg \operatorname* { m i n } } \ \left\| \xi _ { P } ^ { \mathcal { B } } - J _ { P } ^ { \mathcal { B } } ( q ) \dot { q } \right\| _ { 2 } ^ { 2 } + \displaystyle \sum _ { i \in \mathcal { F } } \left\| v _ { i } ^ { \mathcal { P } } - J _ { i } ^ { \mathcal { P } } ( q ) \dot { q } \right\| _ { 2 } ^ { 2 } + \lambda \| \dot { q } \| _ { 2 } ^ { 2 } } \\ & { \qquad \mathrm { s . t . } \quad \varphi _ { c } ( q ) + \nabla \varphi _ { c } ( q ) ^ { \top } \dot { q } \Delta t \geq \varepsilon _ { \mathrm { c o l } } , \qquad c \in \mathcal { C } , } \\ & { \qquad q _ { \mathrm { m i n } } \leq q + \dot { q } \Delta t \leq q _ { \mathrm { m a x } } , \qquad \dot { q } _ { \mathrm { m i n } } \leq \dot { q } \leq \dot { q } _ { \mathrm { m a x } } . } \end{array}\tag{7}
$$

Here, $J _ { P } ^ { B }$ maps joint velocities to the palm twist, while $J _ { i } ^ { \mathcal { P } }$ maps them to the palm-relative velocity of fingertip i. Because the fingertip velocities are defined relative to the palm, the arm columns of $J _ { i } ^ { \mathcal { P } }$ vanish, so the corresponding objective terms control only the finger joints. The constraints enforce collision clearance, joint-position limits, and joint-velocity limits over the control interval ∆t. Here, $\varphi _ { c }$ denotes the signed distance for a monitored collision pair $c \in { \mathcal { C } }$ and $\varepsilon _ { \mathrm { c o l } }$ is the required clearance.

Deployment-time constraint adaptation. The separation between task-space policy outputs and constrained joint-space execution allows us to modify the feasible joint motion at deployment without changing the learned policy. In particular, we can tighten $\dot { q } _ { \mathrm { m i n } }$ and $\dot { q } _ { \mathrm { m a x } }$ to reduce execution speed on the physical robot, while preserving the task-space behavior learned in simulation. We can also introduce additional collision constraints that were not required during simulation training.

Table 9: Policy optimization configuration. We use the same configuration for all tasks and embodiments. The learning rate follows an adaptive KL schedule initialized at $1 0 ^ { - 4 }$ and clipped to $[ 1 0 ^ { - 5 } , 1 0 ^ { - 2 } ]$
<table><tr><td colspan="2">PPO</td><td colspan="2">Network</td></tr><tr><td>Parallel environments</td><td>49,152</td><td>Actor recurrence</td><td>LSTM, 1 layer, width 1024</td></tr><tr><td>Rollout horizon</td><td>32 steps (3.2 s)</td><td>Actor recurrent norm</td><td> Layer norm</td></tr><tr><td>Learning epochs</td><td>5</td><td>Actor MLP</td><td>[1024,1024,512,512]</td></tr><tr><td>Mini-batches</td><td>4</td><td>Critic</td><td>MLP [1024, 1024, 512,512]</td></tr><tr><td>Clip range</td><td>0.2</td><td>Activation</td><td>ELU</td></tr><tr><td>Discount γ</td><td>0.99</td><td>Action distribution</td><td>Gaussian, learned log σ</td></tr><tr><td>GAE λ</td><td>0.95</td><td>Initial σ</td><td>1.0</td></tr><tr><td>Entropy coefficient</td><td>0.006</td><td>Observation history</td><td></td></tr><tr><td>Value-loss coefficient</td><td colspan="3">1.0 (clipped)</td></tr><tr><td>Optimizer</td><td colspan="3">Adam</td></tr><tr><td>Initial learning rate</td><td colspan="3"> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Schedule</td><td colspan="3">Adaptive, KL* = 0.01</td></tr><tr><td>Gradient-norm clip</td><td colspan="3">1.0</td></tr><tr><td>Reset states per stage</td><td colspan="3">10,000</td></tr></table>

For our real-world setup, we add a hand–table clearance constraint

$$
\varphi _ { \mathrm { t a b l e } } ( q ) + \nabla \varphi _ { \mathrm { t a b l e } } ( q ) ^ { \top } \dot { q } \Delta t \ge \varepsilon _ { \mathrm { t a b l e } } ,\tag{8}
$$

where $\varphi _ { \mathrm { t a b l e } }$ measures the minimum signed distance between the hand and the table and $\varepsilon _ { \mathrm { t a b l e } }$ specifies the required clearance. This constraint modifies only the QP at deployment; the RL policy remains unchanged.

Figure 12 illustrates why this separation is useful in our physical system. Without the hand–table clearance constraint, the learned policy can drive the Wuji 1 hand coupler into the table, which we observed to disengage the coupler during real-world execution. Adding the clearance constraint prevents this failure mode without additional policy training. Thus, the QP provides a mechanism for incorporating deploymentspecific hardware and safety requirements while preserving the task-space policy trained in simulation.

## C.5. Simulation Training Details

Simulation setup. We train all policies in Isaac Sim with Isaac Lab using the GPU PhysX pipeline. The simulator runs at 120 Hz, and the policy acts every 12 simulation steps, resulting in a control frequency of 10 Hz. Each episode lasts 16 s, corresponding to 160 policy steps. At every policy step, we solve the QP in Sec. C.4 in batch across all parallel environments to obtain the desired joint velocities. The four evaluated embodiments have 22–28 actuated joints and four or five fingertips. Accordingly, the task-space action dimension is 18 for four-finger hands and 21 for five-finger hands.

Policy optimization. We optimize the policy with PPO (Schulman et al., 2017) under an asymmetric actor–critic. The actor receives only observations available at deployment and uses a single-layer LSTM with layer normalization followed by a four-layer MLP. The critic is memoryless and receives privileged simulation state. We use the same optimization and network configuration across all tasks and embodiments (Table 9).

Domain randomization. During simulation training, we vary physical properties and deliberately add noise to the actor’s observations to improve robustness to real-world diferences. Also we introduce observation and action delays to deal with sim2real gap. Table 10 summarizes these variations.

Table 10: Domain randomization. Multipliers are relative to the original simulation values. Ranges are sampled uniformly unless otherwise stated.
<table><tr><td>Property</td><td>How it is varied</td></tr><tr><td>Physical properties</td><td></td></tr><tr><td>Object mass</td><td>Between the minimum and maximum mass estimated for each object</td></tr><tr><td>Object static friction</td><td>Up to 10% above or below the estimated friction</td></tr><tr><td>Object dynamic friction</td><td>Up to 10% above or below the estimated friction</td></tr><tr><td>Table friction</td><td>Static: 0.3–0.6; dynamic: 0.2–0.5</td></tr><tr><td>Table mass</td><td>0.5–1.5 times the original mass</td></tr><tr><td>Hand stiffness and damping</td><td>0.8–1.25 times the original values</td></tr><tr><td>Errors in actor observations</td><td></td></tr><tr><td>object position</td><td>Gaussian error: 1 cm standard deviation per axis; capped at 3 cm per axis</td></tr><tr><td>static object position</td><td>Gaussian error: 0.5 cm standard deviation per axis; capped at 1.5 cm per axis  $+ 5 ^ { \circ }$ </td></tr><tr><td>Object orientation</td><td>Rotate about a uniformly sampled direction by an angle between  $- 5 ^ { \circ }$  and</td></tr><tr><td>Perceived object size</td><td>Up to 10% larger or smaller, equally in all directions</td></tr><tr><td>Delays</td><td></td></tr><tr><td>Object-pose observation</td><td>0-150 ms</td></tr><tr><td>Robot-state observation</td><td>0–50 ms</td></tr><tr><td>Action execution</td><td>0-50 ms</td></tr></table>

## D. Real-World Perception

The policy observes object poses rather than images (Sec. 3.2), so at deployment we estimate the 6D pose of each task-relevant object in real time. Two RGB-D cameras (Intel RealSense D435i) observe the workspace from complementary viewpoints, and FoundationPose++ (Yan and Chu, 2025) tracks each object independently in each camera, using the same mesh as in simulation. Because the hand frequently occludes the object during manipulation, a single view is often unreliable, and we fuse the two estimates as follows.

Multi-view pose fusion. We fuse the FoundationPose++ estimates $T _ { 1 } , T _ { 2 }$ from the two cameras according to how well each explains its own depth image. For camera c, we render the object mesh at $T _ { c }$ and compute the fraction of its projected silhouette $\Omega _ { c } ,$ including the part outside the image, whose sensor depth is valid and within τ of the rendered depth:

$$
w _ { c } = \frac { 1 } { | \Omega _ { c } | } \sum _ { u \in \Omega _ { c } } \mathbb { I } \big [ | D _ { c } ( u ) - \hat { D } _ { c } ( u ) | < \tau \big ] .\tag{9}
$$

Occluded or truncated views thus receive low weights. Because the attainable value difers between objects and cameras even without occlusion (a white sheet, for instance, returns no depth over much of its surface), each weight is normalized by its unoccluded baseline $w _ { c } ^ { 0 } .$ , the median over the first 15 frames: $\bar { w } _ { c } =$ min $( w _ { c } / \operatorname* { m a x } ( w _ { c } ^ { 0 } , 0 . 1 ) , 1 )$ . The fused pose interpolates $T _ { 1 }$ and $T _ { 2 }$ with $\alpha ~ = ~ \hat { w } _ { 2 } ^ { k } / \left( \hat { w } _ { 1 } ^ { k } + \hat { w } _ { 2 } ^ { k } \right)$ , linearly in translation and by slerp in rotation. A camera is excluded when $\bar { w } _ { c } < w _ { \operatorname* { m i n } }$ and readmitted only once $\bar { w } _ { c } > w _ { \mathrm { r e } }$ , and a pose that deviates from the higher-weighted one by more than 3 cm or $2 0 ^ { \circ }$ is left out of the blend, so that a lost track or a symmetric flip is never averaged. If both cameras are excluded, the pose is extrapolated with a constant-velocity model for up to 0.5 s and then held. The fused pose also serves as the next tracking prior of both cameras, which re-initializes a camera after an occlusion. We use $\tau = 1 . 5 \mathrm { c m }$ $k = 3 , w _ { \mathrm { m i n } } = 0 . 2 5$ , and $w _ { \mathrm { r e } } = 0 . 3 5$

## E. Additional Experimental Results

## E.1. Generalization to Unseen Initial Object Positions

To further analyze spatial generalization beyond the human video, we fix the goal configuration and vary the initial object position across the feasible workspace. Figures 13–17 visualize initial-position generalization across the five evaluation tasks.

![](images/552f29e32967073024f23701062cbb5652a6ceb10dcabe82d23aff37503ff802.jpg)  
Figure 13: Generalization to unseen initial object positions on the Doll task. With the goal fixed, Dex-One2Many succeeds over a substantially broader range of initial positions than the baselines.

![](images/5b0f594c7a0e4fee950033a06f4ae02ec7eb4a27e60cd54077ff97df2d02dca9.jpg)  
Figure 14: Generalization to unseen initial object positions on the Can task. With the goal fixed, Dex-One2Many succeeds over a substantially broader range of initial positions than the baselines.

![](images/4c0a5d38a8939e8069fa2d35e9caf9ce4e7c1d168c8af995143fe6ee02276591.jpg)  
Figure 15: Generalization to unseen initial object positions on the Stamp task. With the goal fixed, Dex-One2Many succeeds over a substantially broader range of initial positions than the baselines.

![](images/8eb227caddba1a028959f478759604b324654ee17d5ee5219b35c87d0e50a7d9.jpg)  
Figure 16: Generalization to unseen initial object positions on the Hammer task. With the goal fixed, Dex-One2Many succeeds over a substantially broader range of initial positions than the baselines.

![](images/86fd9394f056d60aece5b0dcdf810775d94dd1fb07a879ceb2b27b2c1b6720d6.jpg)  
Figure 17: Generalization to unseen initial object positions on the Sweep task. With the goal fixed, Dex-One2Many succeeds over a substantially broader range of initial positions than the baselines.

## E.2. Diverse Grasp Strategies Across Embodiments and Configurations

The task specification extracted from the human video does not prescribe a specific grasp configuration. Instead, our reset generation pipeline exposes the policy to diverse multi-finger grasps drawn from the 33-type grasp taxonomy, allowing the grasp strategy to adapt to the robot embodiment and the current object configuration.

Figures 18–20 show representative grasp strategies learned for the Doll, Can, and Stamp tasks. Diferent dexterous embodiments realize diferent grasp types for the same task, reflecting their distinct hand morphologies and kinematics. Moreover, even within the same embodiment, the learned grasp can change as the object configuration varies. These qualitative results illustrate that Dex-One2Many does not constrain the robot to a fixed grasp inherited from the human demonstration, but instead allows embodiment- and configuration-specific grasp strategies to emerge during policy learning.

![](images/6198b89f60a73a0b85c5def624d0639d2dd048183e436b213492f07b84e0a2f3.jpg)  
(a) UR3 + Wuji 1

![](images/ca84663918dfa10bb337c5577d0336bb53b3251083a58beede47a6405545eacb.jpg)  
(b) UR5e + Sharpa Wave

![](images/c2909c827cf9f078c1af45bc1d425bbd53a544a2a2e7e37de063f83b9013fe48.jpg)  
(c) UR5e + Wuji 2

![](images/e25d79e006cfe59d776e3533e5fc73ae860c3ba5a054b0243297b0fd0c8ad7ce.jpg)  
(c) UR5e + Allegro  
Figure 18: Diverse grasp strategies for the Doll task. Learned grasp types vary across dexterous embodiments and object configurations.

![](images/b0bec596d568f7b048f903f3bb2dc5e6e7c6f9321a98df3fd84db39e5859e387.jpg)  
(a) UR3 + Wuji 1

![](images/7633c2f532ec165bdca475928eee40509e000c5526d96425e03a7cf6704e9736.jpg)  
(b) UR5e + Sharpa Wave

![](images/56cb19fce93c2bed92ac57bb50db5602692b117a84f58e8fb2dcd0009d0e33f3.jpg)  
(c) UR5e + Wuji 2

![](images/a1233878fa4d4873bc917fbcde01d73521365746830779697423cbd44803667b.jpg)  
(c) UR5e + Allegro  
Figure 19: Diverse grasp strategies for the Can task. Learned grasp types vary across dexterous embodiments and object configurations.

![](images/82f6e9e0f7be3ba14a52decaf7d9854a5ba8da3f13b416cd32ee273ea6abcbe3.jpg)  
(a) UR3 + Wuji 1

![](images/eaeecff2ae146e94afbfecc469bc6a104bbce8b4559a2b3f6c43ceebc484698e.jpg)  
(b) UR5e + Sharpa Wave

![](images/b3fa2aa191ed2f3aa3924a9c0930b7829bbddaa49a0a8a11674aa49a1c797f78.jpg)  
(c) UR5e + Wuji 2

![](images/858424e7c49c363a96b0f3183ff80c488ab32414bf2b77bb037cf3cf48311a50.jpg)  
(c) UR5e + Allegro  
Figure 20: Diverse grasp strategies for the Stamp task. Learned grasp types vary across dexterous embodiments and object configurations.

## E.3. Simulation Rollouts Across Diverse Dexterous Embodiments

To complement the quantitative evaluation in Table 1, we visualize successful simulation rollouts across four dexterous embodiments for all five tasks. For each embodiment and task, we show three rollouts initialized from distinct task configurations. Each row corresponds to one configuration, and the columns show the execution over time from left to right. Figure 21–25 show that separately trained policies execute the same task configuration across embodiments with diferent kinematics and under diverse configurations.

![](images/b2db7ba94909f4b51a21918cf6316eec8509e51a8c1c0ac985d530c3bb899132.jpg)  
Figure 21: Successful simulation rollouts across diverse dexterous embodiments for the Doll task. For each of four dexterous embodiments, we show three successful rollouts from distinct task configurations. Each row corresponds to one configuration, and columns show the execution over time from left to right.

![](images/69e3134fa23df902b09030fc33a48f2c7463f6c617f99b01fca27919f02b5a16.jpg)  
Figure 22: Successful simulation rollouts across diverse dexterous embodiments for the Can task. For each of four dexterous embodiments, we show three successful rollouts from distinct task configurations. Each row corresponds to one configuration, and columns show the execution over time from left to right.

![](images/93f8b64ea84e361951d329e8fb31503237a690abe4b47e058308c4fe3dd2575a.jpg)  
Figure 23: Successful simulation rollouts across diverse dexterous embodiments for the Stamp task. For each of four dexterous embodiments, we show three successful rollouts from distinct task configurations. Each row corresponds to one configuration, and columns show the execution over time from left to right.

![](images/5197f3d26ea81f4a24a1e523048edb426bf6596afb2221a1f877d9cbf9d17e38.jpg)  
Figure 24: Successful simulation rollouts across diverse dexterous embodiments for the Hammer task. For each of four dexterous embodiments, we show three successful rollouts from distinct task configurations. Each row corresponds to one configuration, and columns show the execution over time from left to right.

![](images/cbaa3bf85ed0489d7ce5fcc0802f12b08a74b21c4bf827a1a004b170d660ce7f.jpg)  
Figure 25: Successful simulation rollouts across diverse dexterous embodiments for the Sweep task. For each of four dexterous embodiments, we show three successful rollouts from distinct task configurations. Each row corresponds to one configuration, and columns show the execution over time from left to right.