# Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation

Zihan Wang<sup>1,2</sup> Zhen Wu<sup>1</sup> Pieter Abbeel<sup>1,2†</sup> Rocky Duan<sup>1†</sup> Jitendra Malik<sup>1,2†</sup> Carmelo Sferrazza<sup>1†</sup> C. Karen Liu<sup>1,4†</sup> Guanya Shi<sup>1,3†</sup> Angjoo Kanazawa<sup>1,2†</sup>

<sup>1</sup>Amazon FAR <sup>2</sup>UC Berkeley <sup>3</sup>Carnegie Mellon University <sup>4</sup>Stanford † FAR Team Co-Leads

Single Real-world Video

Many Counterfactual Generated Videos

![](images/896434fc0ef803c7e963bd9550a8e1376d6d1ac25d9032318e7bc63a9ab494fa.jpg)  
One Policy, Unseen Objects: Zero-Shot Pick–Carry–Drop in the Real World

![](images/38f2c08792eeab98d675a2e56e714169421112d412a2d08f8427b41e8c9a7a19.jpg)  
Figure 1: PRISM leverages video-to-video (V2V) generation to expand a few real videos into diverse counterfactual interactions—interactions that did not occur in the source videos but could have occurred with different objects. Reconstruction and retargeting yield physically plausible robot–object trajectories for training a unified depth-based humanoid policy that picks up, carries, and drops diverse objects zero-shot in real world deployment. Project website: prism-real2sim2real.github.io.

Abstract: Teaching humanoids loco-manipulation skills, such as carrying diverse objects, via visual imitation is a promising path toward generalist robots. However, collecting diverse, high-quality interaction videos, such as clips that clearly

show a person’s full body and unoccluded interactions with objects, poses a practical barrier to scaling this approach. We propose PRISM, a real-to-sim-to-real framework that overcomes this limitation by amplifying a handful of real videos into a large, diverse training set. PRISM first generates hundreds of diverse “counterfactual” human–object interaction videos via video-to-video (V2V) generation from a few exemplar real videos. Our contact-anchored real-to-sim pipeline then reconstructs both human and object motions, retargeting this imperfect video data into physically plausible trajectories. The intra-class variability across these counterfactual videos lets us train a single policy that generalizes to unseen objects within each category. We demonstrate the full pipeline by deploying this policy on a real robot without any real-world fine-tuning. Using only onboard depth observations, our humanoid picks up, carries, and drops objects—including boxes, barrels, bins, and balls—across novel instances, sizes, and initial configurations.

Keywords: Loco-manipulation, Real-to-Sim-to-Real, Humanoid

## 1 Introduction

How can humanoids learn to interact with the diverse objects encountered in everyday life? Visual imitation learning offers a promising route: human demonstrations provide examples of the coordinated whole-body motions needed to approach, lift, carry, and drop various objects. Recent real-to-sim-to-real pipelines [1] have enabled humanoids to acquire contextual whole-body locomotion skills directly from human videos. These advances suggest a path toward generalist humanoids that learn a broad repertoire of locomotion and manipulation skills from internet-scale video data.

However, acquiring diverse, high-quality human–object interaction videos remains a practical bottleneck to scaling visual imitation. Internet videos are abundant, but their content and framing are shaped by human viewing preferences. Filtering Internet videos to find demonstrations that clearly show the person’s full body and how they interact with objects is therefore costly and impractical at scale. Yet, such interaction data does exist implicitly in modern video generative models [2], whose learned priors over human motion and interactions can be used to synthesize diverse training videos.

To address this data bottleneck, we propose PRISM, a real-to-sim-to-real framework that uses videoto-video (V2V) generation to expand a few real videos into diverse human–object interactions. We call these generated clips “counterfactual videos”: they depict interactions that did not occur in the source videos but could have occurred with different objects. From only four real-world videos of humans carrying boxes, we generate 256 counterfactual videos depicting the same task with boxes, balls, bins, and barrels of varying geometries and initial configurations. Our contact-anchored realto-sim pipeline reconstructs human motion, the static scene, and dynamic object geometry and motion in a unified world frame, then retargets these imperfect reconstructions into physically plausible humanoid–object trajectories. With these trajectories, we train a single depth-based policy that picks up, carries, and drops unseen instances of these categories zero-shot in the real world, validating the full pipeline. Our policy also generalizes to categories absent from the generated videos (Fig. 1).

The core technical challenge is to obtain physically plausible robot demonstrations despite errors introduced by both counterfactual video generation and monocular reconstruction. Our key insight is to use contact signal as a shared constraint across reconstruction, retargeting, and policy learning. In the first reconstruction stage, human–object contact couples the two motions: human motion guides the object trajectory, while object contact helps correct errors in the reconstructed human pose. During the retargeting stage, sparse contact anchors specify where robot end-effectors should contact the object, guiding the refinement of noisy reconstructions into physically plausible robot–object trajectories while accommodating morphological differences. During policy learning, these anchors define contact rewards for a privileged teacher policy that co-tracks robot and object motion. We then distill it into a depth-based policy with joystick control for zero-shot sim-to-real deployment.

We summarize our contributions as follows: (1) We introduce a new learning paradigm that uses counterfactual video generation for scaling human–object interaction experience. (2) We propose a contact-anchored real-to-sim and retargeting pipeline that turns monocular reconstructions into physically plausible robot–object demonstrations. (3) We demonstrate the first real-to-sim-to-real pipeline that trains a unified whole-body humanoid policy that generalizes across diverse objects and enables joystick-controlled pick-up, carry, and drop behaviors from onboard depth observations.

## 2 Related Work

Humanoid Loco-Manipulation. Humanoid loco-manipulation has been widely studied in both computer graphics [3, 4] and robotics [5, 6, 7], where the goal is to coordinate whole-body locomotion and object interaction simultaneously. Some humanoid systems [5] learn interaction skills by tracking human motion references, but require dense reference motions or object trajectories during inference, limiting generalization to unseen objects and interaction configurations. More recent methods [8, 9, 10] reduce this dependency by learning more autonomous interaction policies from sparse goals or interaction-centric representations. Similarly, PRISM trains a unified whole-body humanoid policy that enables zero-shot joystick-controlled pick-up, carry, and drop behaviors from onboard depth observations, without reference or motion-capture systems during deployment.

Learning from Human Video. Recent works learn humanoid skills from human videos [5, 11], including terrain-aware locomotion via joint human–scene reconstruction [1] and dynamic object interactions [5, 12]. However, capturing or curating diverse, high-quality interaction videos at scale remains challenging. PRISM addresses this complementary challenge by expanding a few real videos into diverse human–object interactions through video-to-video generation. With these data, we train a single policy that picks up, carries, and drops diverse objects zero-shot in the real world.

Video Model as Data++. Recent works have explored video generation for robot learning under two main paradigms. One line uses video models as inference-time planners or action generators, synthesizing future visual rollouts to guide closed-loop manipulation [13, 14, 2]. While such methods provide flexible visual reasoning, they require expensive test-time generation and may suffer from an executability gap between plausible videos and physically feasible robot actions. Another line uses video models offline to synthesize robot demonstrations or augment robot datasets before policy learning [15, 16]. PRISM follows the offline-generation paradigm, but differs in generating counterfactual human interaction videos through grounded V2V generation rather than directly generating robot videos from text or images. By conditioning on real demonstrations, V2V preserves realistic motion, lighting, and affordance-aware contacts, while a simple unified prompt can scale one exemplar into diverse object categories, poses, and intra-class variations by multiple sampling.

## 3 Real-to-Sim Data Acquisition

Given a monocular video depicting a human interacting with a static scene and a dynamic object, our goal is to recover camera parameters $( K , \{ T _ { i } ^ { c } \} )$ , a static scene mesh $M _ { s }$ , a dynamic object with its metric mesh $M _ { o }$ and its per-frame poses $\{ T _ { i } ^ { o } \}$ , and 4D SMPL-X [17] motion, all in a unified world frame. Then we retarget the recovered data to humanoid–object trajectories for policy learning.

## 3.1 Counterfactual Interaction Video Generation

Scaling real-to-sim learning with online videos remains challenging. Although internet videos are abundant, their content and framing are shaped by human viewing preferences. Filtering for diverse clips with clear full-body views, visible human–object interactions, and stable viewpoints is therefore costly and impractical at scale. We use video-to-video (V2V) generation to expand a small set of suitable real videos into diverse counterfactual interaction videos—interactions that did not occur in the source videos but could have occurred with different objects. Given a seed video and a category-level text prompt, we ask the model to replace the manipulated object while preserving the original background, lighting, camera viewpoint, and coarse task structure. This video conditioning grounds generation in a real scene and task while allowing object and human behavior to vary.

![](images/76cc79bf36b6ddcae4b5c32306e8ee7abd49d2ea2c1acb93cb65f53cf61fd500.jpg)  
Figure 2: PRISM Real-to-sim Overview. PRISM turns counterfactual human–object videos into deployable humanoid loco-manipulation skills. It reconstructs the camera, human motion, object geometry, and 6D object motion in a shared world frame, using contact points to constrain object pose optimization under monocular ambiguity. The same anchors are used in retargeting to preserve interaction phases and match robot end-effectors to intended contact points. The retargeted demonstrations train a privileged co-tracking teacher, which is distilled into a depth-based student policy Contact-anchored Retargeting conditioned on onboard depth and joystick commands for zero-shot sim-to-real deployment.

Importantly, the counterfactual variation is not limited to object geometry or appearance. When the object category, size, pose, or placement changes, the object’s manipulation affordances also change, and the human behavior adapts accordingly: the generated person may bend lower, adjust hand spacing, adapt the contact strategy based on the object’s affordances, or carry the object differently. <sup>interactions</sup>Thus each generated clip provides an object-conditioned interaction strategy, rather than merely oringpairing the same human motion with a new object. This allows PRISM to obtain behavior-level <sup>Proprioception</sup>variations in interaction data that are difficult to hard-code through geometry-level augmentation alone. Learning such adaptations from scratch would require exhaustive task-specific RL reward design; PRISM instead uses the video model as an offline prior over plausible human adaptations.

In practice, we record four real-world seed videos and use each as a grounded template for V2V generation. We prompt SeedDance 2.0 [18] to replace the manipulated object with a box, bin, barrel, or ball while preserving the original background, lighting, camera viewpoint, and temporal continuity (Appendix A). Category-level prompts allow the model to vary object geometry, appearance, and pose while adapting human behavior accordingly. With 16 samples per category, each seed yields 64 counterfactual videos, expanding four real recordings into 256 videos for our real-to-sim pipeline.

## 3.2 Contact-Anchored Real-to-Sim

To imitate human motion from monocular video, we must disentangle it from camera-induced image motion and recover it in a consistent world coordinate frame. CRISP [19] integrates an human mesh recovery (HMR) network with visual SLAM to reconstruct a metrically-consistent human–scene– camera representation from monocular video, including camera intrinsics $K \in \mathbb { R } ^ { 3 \times 3 }$ , per-frame camera poses $T _ { i } = [ R _ { i } \ | \ t _ { i } ] \in S E ( 3 )$ , a metric-scale scene point cloud ${ \tilde { P } } ,$ and temporally aligned 4D human motion in a unified world coordinate frame. We use it as backend and extend it to reconstruct both the geometry and motion of dynamic objects. While CRISP [20] focuses on human motion and the surrounding environment, imitating human–object interactions additionally requires disentangling dynamic object motion from camera motion. We use it as our backend and extend it to recover both the geometry and motion of dynamic objects in the same world coordinate frame.

Object Geometry and Motion Reconstruction. Given dynamic object masks from SAM 2 [21], we reconstruct the object geometry with SAM3D [22], together with the camera poses and metric depth from CRISP, yielding an object mesh $\mathcal { M } _ { o }$ and initial world pose $T _ { o } ^ { t = 0 } \in S E ( 3 )$ . Rather than tracking the object independently with a visual 6D tracker such as FoundationPose [23], which is sensitive to hand and body occlusions in monocular video (Table 3), we exploit human–object contact to regularize both motions: human motion provides a strong prior for the object trajectory, while the object in turn constrains inaccurate human joint estimates. For pick–carry–drop interactions, the object remains supported by the scene outside contact and follows the human during stable contact.

We detect contact from human motion clues, avoiding manual contact annotations [5, 12]. The detected contact points also regularize human motion during IK by replacing inaccurate joint targets with contact-consistent constraints. During each contact phase $[ t _ { 1 } , t _ { 2 } ]$ , we anchor the object to the SMPL-X palm using the palm-relative transform at $t _ { 1 }$ , held fixed throughout the phase:

$$
T _ { o } ^ { t } = T _ { \mathrm { p a l m } } ^ { t } T _ { \mathrm { p a l m }  o } = T _ { \mathrm { p a l m } } ^ { t } ( T _ { \mathrm { p a l m } } ^ { t _ { 1 } } ) ^ { - 1 } T _ { o } ^ { t _ { 1 } } , \qquad T _ { \mathrm { p a l m }  o } = ( T _ { \mathrm { p a l m } } ^ { t _ { 1 } } ) ^ { - 1 } T _ { o } ^ { t _ { 1 } } , \qquad t \in [ t _ { 1 } , t _ { 2 } ] .
$$

Thus, human motion propagates the object trajectory during contact, while object contact, in turn, provides geometric constraints that correct errors in the reconstructed human motion.

Contact-Anchored Retargeting. Prior retargeting methods [6] preserve human–object relations but assume clean, physically plausible inputs. Monocular reconstructions often violate this assumption (pink in Fig. 2), causing retargeting to preserve the reconstruction errors. We therefore propose contact-anchored retargeting, which treats the reconstructed full-body human motion as a kinematic reference while using object-frame contact anchors as the reliable interaction interface. These anchors specify where the robot end-effectors should act on the object, allowing the retargeting solver to correct noisy reconstructions into physically plausible robot–object motions.

For each contact phase, we derive contact anchors from the reconstructed human–object geometry in Stage 1. Specifically, we intersect the line segment connecting the SMPL-X left and right palm centers with the object mesh. Following interaction-preserving retargeting [6], we keep the constrained IK and add a contact-anchor term to the original objective. At each frame, we solve:

$$
\begin{array} { r } { d q _ { a } ^ { \star } = \underset { d q _ { a } \in \mathcal { C } ( q _ { a } ) } { \arg \operatorname* { m i n } } \ E _ { \mathrm { b a s e } } ( q _ { a } , d q _ { a } ) + w _ { c } \left. x _ { \mathrm { e e f } } ( q _ { a } ) + J _ { \mathrm { e e f } } ( q _ { a } ) d q _ { a } - c \right. _ { 2 } ^ { 2 } . } \end{array}\tag{1}
$$

Here $E _ { \mathrm { b a s e } }$ denotes the interaction mesh matching from [6], c is the target contact point estimated from the human–object reconstruction in Fig. 2, and $x _ { \mathrm { { e e f } } } ( q _ { a } ) + J _ { \mathrm { { e e f } } } ( q _ { a } ) d q _ { a }$ is the linearized end effector position. We update $q _ { a }  q _ { a } + d q _ { a } ^ { \star }$ each frame and use it to warm-start the next frame.

One might attribute these artifacts to the reconstruction pipeline. We take a different view: imperfect reconstruction is unavoidable in monocular real-to-sim stage, as no existing pipeline can recover physically plausible human–object motions. Our contact-anchored real-to-sim system closes this gap by optimizing sufficiently close reconstructions into physically plausible robot demonstrations.

## 4 Learning a Unified Visuomotor Interaction Policy

Using the reconstructed robot-object trajectories (details are in Appendix C), our goal is to train a unified humanoid policy capable of pick-up, carry, and drop behaviors across diverse objects. The policy receives onboard depth observations and joystick commands, and autonomously performs object interaction behaviors conditioned on the perceived object geometry and placement. Similar to prior visuomotor systems [24, 25], we adopt a two-stage training framework. We first train a privileged co-tracking teacher policy in simulation using full-state observations. We then distill this teacher into a unified depth-based student policy using a combination of DAgger [26] and reinforcement learning, enabling zero-shot sim-to-real deployment. An overview is shown in Fig. 2.

## 4.1 Training a Co-Tracking Teacher Policy

We formulate humanoid object interaction as a co-tracking problem, where the policy simultaneously tracks both the humanoid motion and the dynamic object trajectory reconstructed from monocular videos. We refer readers to our codebase for our motion-tracking training details.

Pose Variation  
![](images/bb069aced4509634ae892ea9df246524faaf3de12ba4f537c177393e303691fd.jpg)  
Scale Variation

Intra-Category Variation  
![](images/0aa2b07f522eb1c38bd4e259aee2934219dbd526c418667608a8c163b07fba52.jpg)  
Approach-Distance Variation

![](images/63088adcc6f3bc74bb089f89ef585387034f9ac291bf29d8f2a2aeda38cc7657.jpg)

![](images/fac9970777bb08cf4ab409d95d484e8082971bec6f39a7b3da881f7dcc699209.jpg)  
Figure 3: Policy rollout under real-world object variations. Our single unified policy zero-shot transfers to the real robot across object-pose, intra-category, scale, and approach-distance variations. The policy is agnostic to object placement and initial pose (top-left), handles objects with different topology and appearance within the same category (top-right), and generalizes to large scale variations (bottom-left). It further adapts to object distance using onboard depth, either approaching (up to 4.3 feet!) before grasping or directly grasping when the object is within reach (bottom-right).

Observations. Teacher observations include reference motion, robot proprioception, the previous action, and both current and target object states, represented by object pose and 3D bounding-box dimensions, making the policy explicitly aware of both the desired robot motion and the object state.

Rewards and Terminations. The reward primarily consists of robot pose tracking, object pose tracking, action rate, joint limits, and collision penalties. We additionally introduce a contact-aware interaction reward using the contact anchors from Sec 3.2. Specifically, we encourage the robot end-effectors to reach the desired contact positions and exceed a predefined contact-force threshold:

$$
R _ { \mathrm { c o n t a c t } , i } = \exp \left( - \frac { \| \mathbf { p } _ { \mathrm { e e f } , i } - \mathbf { p } _ { \mathrm { t a r g e t } , i } \| _ { 2 } } { \sigma _ { \mathrm { p o s } } } \right) \cdot \operatorname* { m i n } \left( \exp \left( \frac { \| \mathbf { F } _ { \mathrm { c o n t a c t } , i } \| _ { 2 } - F _ { \mathrm { t h r e s } } } { \sigma _ { \mathrm { f r c } } } \right) , 1 \right) .\tag{2}
$$

where $\mathbf { p } _ { \mathrm { e e f } , i }$ denotes the end-effector position, $\mathrm { p } _ { \mathrm { t a r g e t } , i }$ is the reconstructed target contact point, and $\mathbf { F } _ { \mathrm { c o n t a c t } , i }$ is the contact force vector. This contact-aware reward stabilizes grasping and carry behaviors under noisy monocular reconstructions and substantially improves interaction quality during policy learning. We also adopt early termination and domain randomization following [6].

Table 1: Cross-domain evaluation. OMOMO-trained policies perform well on held-out OMOMO interactions but transfer poorly to PRISM. PRISM-trained policies transfer to OMOMO and unseen PRISM-OOD categories; PRISM-ID evaluates the 80 training interactions.
<table><tr><td>Training Data</td><td>OMOMO Train</td><td>OMOMO Test</td><td>PRISM ID</td><td>PRISM OOD</td></tr><tr><td>OMOMO (90)</td><td>94.44%</td><td>91.67%</td><td>23.75%</td><td>12.50%</td></tr><tr><td>PRISM ID(80)</td><td>98.89%</td><td>100%</td><td>96.25%</td><td>72.92%</td></tr></table>

## 4.2 Distilling a Unified Depth-Based Student Policy

The privileged teacher learns stable object interactions in simulation but depends on information unavailable on real hardware. We therefore distill it into a deployable depth-based student policy using only onboard perception and joystick commands, without MOCAP system or reference motion.

Distillation. We distill the teacher policy using a combination of DAgger and PPO objectives:

$$
\begin{array} { r } { \mathcal { L } = \lambda \mathcal { L } _ { \mathrm { P P O } } + ( 1 - \lambda ) \mathcal { L } _ { D } , } \end{array}\tag{3}
$$

where $\mathcal { L } _ { D }$ denotes the DAgger imitation loss and $\mathcal { L } _ { \mathrm { P P O } }$ is the RL objective. Following [24], we use a curriculum that gradually increases λ during training and keeps $\lambda = 0 . 9$ for the final 20K iterations.

Observations. The student receives proprioception, joystick commands, and onboard depth. At deployment, we estimate depth offboard from stereo images using Fast-FoundationStereo [27]. We find that this substantially reduces the sim-to-real gap in depth observations. Proprioception includes angular velocity, joint states, and the previous action. Joystick commands comprise a relative root command $c _ { t } ^ { \mathrm { j s } } = [ \Delta x _ { t } , \Delta y _ { t } , \Delta \mathrm { y a w } _ { t } ]$ and a binary drop command $b _ { t } ^ { \mathrm { d r o p } } \in \{ 0 , 1 \}$ , derived during training from the reference trajectory and carry-end time (Appendix D). In simulation, we use a nominal $3 7 ^ { \circ }$ camera pitch, randomize camera pose, and inject depth noise, dropout, holes, edge artifacts, and offsets. The critic uses current robot, object, and reference states without history.

Warm Start. We warm-start from a 23K-iteration checkpoint trained with the same method on box-only data, restoring only actor weights before training on all categories. Training from scratch succeeds; warm starting accelerates convergence and is used for our released checkpoint.

Training Details. We train 3-layer MLP teacher and student policies for 40K and 28K iterations, respectively, using 4096 environments per GPU on 8 NVIDIA L40S GPUs. Distillation uses teacher rollouts as motion references instead of the original kinematic trajectories. See Appendix D.

## 5 Results

Overview. We demonstrate that humanoid robots can learn skills that generalize to diverse objects by imitating pure counterfactual generated videos. We first evaluate the effectiveness of our reconstructed data by comparing with the policy trained with OMOMO [4] data only. Next, we evaluate the real-world performance over 20+ diverse objects. Finally, we conduct necessary ablation study to validate the necessity of each module in our framework design. All evaluations are conducted in MuJoCo [28] under a sim-to-sim setting, except for Sec. 5.2 reports real-world robot experiments.

## 5.1 Reconstruction Quality from Real-to-Sim

Evaluation Data. We process and filter OMOMO to retain 63 high-quality human–object interaction sequences, then double the data by adding an object-scaled variant for each sequence using OmniRetarget [6]. This gives 126 sequences, with 90 for OMOMO-Train and 36 held out as OMOMO-Test. For PRISM, the reported comparison uses an earlier student distilled from 80 teacher rollouts and evaluates 80 PRISM-ID interactions and 48 PRISM-OOD interactions with chairs, tables, lamps, and monitors. Appendix C distinguishes this evaluation from the updated training configuration.

Evaluation Metrics. For the distilled student policy, we derive joystick commands from the reference motion and render depth observations in simulation to drive the policy. A rollout is successful if the robot completes pick–carry–drop and places the object at the desired target position.

Table 2: Real-world evaluation. We evaluate our policy on real-world objects. Each in-domain category contains 3 different objects (see Fig. 7 for details). Each object is tested with 5 trials.
<table><tr><td></td><td colspan="4">In-domain</td><td colspan="8">Out-of-domain</td></tr><tr><td>Metric</td><td>Ball</td><td>Bin</td><td>Barrel</td><td>Box</td><td>Paper towel</td><td>Helmet</td><td>Table</td><td>Backpack</td><td>Lamp</td><td>Chair</td><td>Kettle</td><td>Chick toy</td></tr><tr><td>Success / Trials</td><td>12 / 15</td><td>14 /15</td><td>14 /15</td><td>15 / 15</td><td>5/5</td><td>3/5</td><td>5/5</td><td>5/5</td><td>3/5</td><td>4/5</td><td>4/5</td><td>3/5</td></tr><tr><td>Success Rate</td><td>80%</td><td>93%</td><td>93%</td><td>100%</td><td>100%</td><td>60%</td><td>100%</td><td>100%</td><td>60%</td><td>80%</td><td>80%</td><td>60%</td></tr></table>

Table 3: Ablation study. We progressively add contact-anchored pose reconstruction, retargeting, and contact rewards. Each component improves task success on PRISM-ID and PRISM-OOD.
<table><tr><td></td><td>Contact-anchored Pose</td><td>Contact-anchored Retargeting</td><td>Contact Reward</td><td>Suc. (ID)</td><td>Suc. (OOD)</td></tr><tr><td>Baseline</td><td>x</td><td>x</td><td>x</td><td>22.50%</td><td>12.50%</td></tr><tr><td>+ anchored object pose</td><td>√</td><td>x</td><td>x</td><td>58.75%</td><td>29.17%</td></tr><tr><td>+ contact-aware retargeting</td><td>√</td><td>√</td><td>x</td><td>86.25%</td><td>68.75%</td></tr><tr><td>+ contact reward (PRISM)</td><td>√</td><td>√</td><td>√</td><td>96.25%</td><td>72.92%</td></tr></table>

Discussion. OMOMO-trained policies perform well across OMOMO sequences, but transfer poorly to PRISM interactions, especially with out-of-domain objects. We identify three main failure modes: the policy struggles with unseen depth observations, overfits to close-range interactions and attempts to grasp before approaching distant objects, and fails to stabilize carry on novel object geometries. These failures suggest that OMOMO alone lacks the variation in perception, object configuration, and interaction dynamics required for robust loco-manipulation. In contrast, PRISM-trained policies generalize across both domains, demonstrating the broader transferability of PRISM data.

## 5.2 Real-World Deployment

We deploy our controller at 50 Hz on a 29-DoF Unitree G1, with PD gains following [29]. A headmounted D435i camera captures stereo images at 30 Hz, and Fast-FoundationStereo [27] estimates depth on an external computer connected to the robot via Ethernet. A human operator provides joystick commands. We freely place each object in front of the robot, varying its pose and distance across five trials. A trial succeeds if the robot reaches and grasps the object, then carries it stably for at least 3 m without dropping it or falling. We evaluate all test objects (Fig. 7) zero-shot, without using their scans or reconstructions for training. Videos are available on our project website.

Discussion. We find that Fast-FoundationStereo substantially reduces the depth sim-to-real gap. We manually tilt the neck about 10<sup>◦</sup> upward from its default position, relying on camera-pose randomization during training to tolerate imprecise calibration. During bending, the neck sometimes resets to default or deviates from its target angle, producing out-of-distribution depth observations.

## 5.3 Ablation Study

Using eight seeds at submission, V2V outperforms geometric augmentation with fewer demonstrations (80 vs. 103; Appendix F). We then ablate the key real-to-sim-to-real components on this expanded dataset. Figure 4 separately visualizes the initial object poses of 137 generated clips and four real-video seeds. Following [19], a good reconstruction for robot learning should be physically plausible, simulatable and useful for policy training. Therefore, we use the reconstructed trajectories to train policies in simulation, and report the downstream task success rate as the main metric. We define the baseline as: initialize object geometry and poses by SAM3D [22], then track it by FoundationPose [23], and retarget it into simulation-ready robot–object demonstrations with OmniRetarget [6]. Table 3 shows that better real-to-sim data directly benefits downstream policy learning. The baseline produces inaccurate robot–object interactions that limit effective policy learning. Anchored object pose and contact-aware retargeting progressively improve the data quality thus lead to stronger downstream performance. The contact reward further improves the policy by encouraging stable, task-relevant robot–object contacts during teacher and student training.

Robustness and Generalization. For reconstruction, we select seeds with stable views, limited occlusion, and task-compatible motions; V2V needs only simple prompts. At scale, heterogeneousasset setup in Isaac Lab limits simulation and training throughput. Our policy succeeds zero-shot on a 35<sup>◦</sup> ramp and a 0.43 m elevated support (Fig. 5), likely due to grasp-height overlap with tall training objects, but fails at 45<sup>◦</sup> and 0.45 m. Training on elevated V2V data may extend this range.

![](images/13722a0e1d64b0eb61cdbe95bc26ba60a4b525c6d7e904cdf58b16f5a2cf385f.jpg)

![](images/ce0c5e7e33c226bb5286f810cede63b81aa53b92238452fbe381e372ddf3d074.jpg)

Figure 4: Initial object-pose coverage. Object poses relative to G1 in 137 generated clips and four upright seed videos. Left: planar positions of generated clips (circles) and seeds (squares). Right: pooled yaw counts in six 30<sup>◦</sup> bins modulo 180<sup>◦</sup>. Colors distinguish upright and lying-down objects.  
![](images/e1e1b53bfb9fd1e4f90dd0c09aca8efb835e1285df5c16de6a8187178ef4aa9d.jpg)  
Figure 5: Zero-shot elevated pick-up. Though trained only on flat terrain, our policy picks up a box on a 35<sup>◦</sup> ramp (left) and from a 0.43 m elevated support (right), without additional policy tuning.

## 6 Conclusion

We presented PRISM, a real-to-sim-to-real framework that expands a few real videos into diverse humanoid demonstrations. Contact-anchored reconstruction and retargeting turn imperfect counterfactual videos into physically plausible robot–object trajectories. The resulting unified depth-based policy picks up, carries, and drops unseen objects zero-shot in the real world. These results highlight video generation as a practical data source for generalizable whole-body interaction.

## 7 Limitations and Future Work

Our pipeline delivers encouraging real-world results, yet several practical weaknesses remain.

Reconstruction and Retargeting. Monocular 4D human–object–scene recovery remains challenging, especially when preserving consistent contacts and interaction dynamics. Reconstruction errors can propagate through retargeting and compromise robot demonstrations. Although Astra was unavailable during the development of this work, such coding agents could automate and iteratively refine reconstruction from counterfactual videos. Combining these agents with contact-aware retargeting and simulation-based validation may improve the reliability of robot demonstrations.

Data Scale. Four seed videos yield 256 generated clips, 137 feasible trajectories, and 129 successful teacher rollouts. This scale is insufficient to characterize scaling behavior, with reconstruction and retargeting remaining practical bottlenecks. Future work should jointly scale generation and trajectory recovery to study how demonstration quantity and quality affect zero-shot generalization.

Object Physics. We use category-specific nominal masses and mesh-derived inertias, with coupled mass–inertia scaling and randomized friction (Appendix E). We do not identify instance-specific surface properties or internal mass distributions, which can cause sim-to-real contact mismatch.

Simulation Limitations. We model dynamic objects as rigid bodies. Thus the policy can struggle with articulated or deformable objects whose contact geometry may change during interaction. For example, foldable chairs may shift through internal joints, causing unstable grasps or loss of control.

## Acknowledgments

We thank Chung Min Kim, Arthur Allshire, Hongsuk Choi, Isabella Yu, Junyi Zhang, Jacob Berg, Yen-Jen Wang, Sirui Chen, Charlie Cheng, Jiashun Wang, Siheng Zhao, Youjian Huang, JC Hu, Haochen Wang, Haozhi Qi and Qitao Zhao for their support and valuable feedback.

## References

[1] A. Allshire, H. Choi, J. Zhang, D. McAllister, A. Zhang, C. M. Kim, T. Darrell, P. Abbeel, J. Malik, and A. Kanazawa. Visual imitation enables contextual humanoid control. arXiv preprint arXiv:2505.03729, 2025.

[2] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[3] Z. Wu, J. Li, P. Xu, and C. K. Liu. Human-object interaction from human-level instructions. In ICCV, 2025.

[4] J. Li, J. Wu, and C. K. Liu. Object motion guided human motion synthesis. ACM Transactions on Graphics (TOG), 42(6):1–11, 2023.

[5] H. Weng, Y. Li, N. Sobanbabu, Z. Wang, Z. Luo, T. He, D. Ramanan, and G. Shi. Hdmi: Learning interactive humanoid whole-body control from human videos. arXiv preprint arXiv:2509.16757, 2025.

[6] L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza, C. K. Liu, R. Duan, and G. Shi. Omniretarget: Interaction-preserving data generation for humanoid whole-body locomanipulation and scene interaction. arXiv preprint arXiv:2509.26633, 2025.

[7] Y.-J. Wang, J. Li, S. Chen, T. E. Truong, P. Xu, P. Abbeel, R. Duan, K. Sreenath, A. Kanazawa, C. Sferrazza, et al. Vlk: Learning humanoid loco-manipulation from synthetic interactions in reconstructed scenes. arXiv preprint arXiv:2606.30645, 2026.

[8] Y. Lin, J. Cui, Y. Li, B. Jia, Y. Zhu, and S. Huang. Lessmimic: Long-horizon humanoid interaction with unified distance field representations. arXiv preprint arXiv:2602.21723, 2026.

[9] X. He, S. Xu, X. Li, R. Dong, L. Bian, Y.-X. Wang, and L.-Y. Gui. Ultra: Unified multimodal control for autonomous humanoid whole-body loco-manipulation. arXiv preprint arXiv:2603.03279, 2026.

[10] J. Wang, M. E. Mungai, H. Li, J. P. Sleiman, J. Hodgins, and F. Farshidian. Generalizing from references using a multi-task reference and goal-driven rl framework. arXiv preprint arXiv:2602.20375, 2026.

[11] J. Mao, S. Zhao, S. Song, T. Shi, J. Ye, M. Zhang, H. Geng, J. Malik, V. Guizilini, and Y. Wang. Learning from massive human videos for universal humanoid pose control. arXiv preprint arXiv:2412.14172, 2024.

[12] Y. Wang, Q. Zhao, Y. F. Lau, R. Yu, H. W. Tsui, Q. Chen, J. Wang, J. Pang, and P. Tan. Humanx: Toward agile and generalizable humanoid interaction skills from human videos. arXiv preprint arXiv:2602.02473, 2026.

[13] B. Chen, T. Zhang, H. Geng, C. Zhang, P. Li, K. Song, W. T. Freeman, J. Malik, P. Abbeel, R. Tedrake, et al. Large video planner enables generalizable robot control. arXiv preprint arXiv:2512.15840, 2025.

[14] H. Bharadhwaj, D. Dwibedi, A. Gupta, S. Tulsiani, C. Doersch, T. Xiao, D. Shah, F. Xia, D. Sadigh, and S. Kirmani. Gen2act: Human video generation in novel scenarios enable generalizable robot manipulation. arXiv preprint arXiv:2409.16283, 2024.

[15] S. Patel, S. Mohan, H. Mai, U. Jain, S. Lazebnik, and Y. Li. Robotic manipulation by imitating generated videos without physical demonstrations. arXiv preprint arXiv:2507.00990, 2025.

[16] J. Jang, S. Ye, Z. Lin, J. Xiang, J. Bjorck, Y. Fang, F. Hu, S. Huang, K. Kundalia, Y.-C. Lin, et al. Dreamgen: Unlocking generalization in robot learning through video world models. arXiv preprint arXiv:2505.12705, 2025.

[17] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A. A. A. Osman, D. Tzionas, and M. J. Black. Expressive body capture: 3D hands, face, and body from a single image. In CVPR, 2019.

[18] T. Seedance, D. Chen, L. Chen, X. Chen, Y. Chen, Z. Chen, Z. Chen, F. Cheng, T. Cheng, Y. Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

[19] Z. Wang, J. Wang, J. Tan, Y. Zhao, J. Hodgins, S. Tulsiani, and D. Ramanan. Crisp: Contact-guided real2sim from monocular video with planar scene primitives. arXiv preprint arXiv:2512.14696, 2025.

[20] Z. Wang, J. Wang, J. Tan, Y. Zhao, J. K. Hodgins, S. Tulsiani, and D. Ramanan. Contact-guided real2sim from monocular video with planar scene primitives. In The Fourteenth International Conference on Learning Representations, 2026.

[21] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland,¨ L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollar,´ and C. Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. URL https://arxiv.org/abs/2408.00714.

[22] S. D. Team, X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, A. Lin, J. Liu, Z. Ma, A. Sagar, B. Song, X. Wang, J. Yang, B. Zhang, P. Dollar, G. Gkioxari, M. Feiszli, and J. Malik. Sam 3d: 3dfy anything in images. ´ arXiv preprint arXiv:2511.16624, 2025. URL https://arxiv.org/abs/2511.16624.

[23] B. Wen, W. Yang, J. Kautz, and S. Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects, 2024. URL https://arxiv.org/abs/2312.08344.

[24] Z. Wu, X. Huang, L. Yang, Y. Zhang, K. Sreenath, X. Chen, P. Abbeel, R. Duan, A. Kanazawa, C. Sferrazza, et al. Perceptive humanoid parkour: Chaining dynamic human skills via motion matching. arXiv preprint arXiv:2602.15827, 2026.

[25] Y. Kuang, S. Park, K. Fragkiadaki, and S. Tulsiani. Dex4d: Task-agnostic point track policy for sim-to-real dexterous manipulation. arXiv preprint arXiv:2602.15828, 2026.

[26] S. Ross, G. Gordon, and D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

[27] B. Wen, S. Dewan, and S. Birchfield. Fast-FoundationStereo: Real-time zero-shot stereo matching. CVPR, 2026.

[28] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In IROS, 2012.

[29] Q. Liao, T. E. Truong, X. Huang, Y. Gao, G. Tevet, K. Sreenath, and C. K. Liu. Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion. arXiv preprint arXiv:2508.08241, 2025.

[30] Z. Wang, J. Tan, T. Khurana, N. Peri, and D. Ramanan. Monofusion: Sparse-view 4d reconstruction via monocular fusion. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8252–8263. IEEE, 2025.

## A Counterfactual Video Generation Details

We record four real-world seed videos of humans carrying boxes, selecting clips with clear full-body views, limited occlusion, and stable viewpoints. Using the SeedDance 2.0 web interface [18], we condition generation on each seed video and a category-level text prompt. The prompt replaces the manipulated object while preserving the background, lighting, camera viewpoint, and coarse task structure. These counterfactual videos depict interactions that could have occurred with different objects, allowing both object properties and human behavior to vary.

We generate 16 samples for each of four categories—boxes, bins, barrels, and balls—per seed video. This yields 64 counterfactual videos per seed and 256 videos in total. All generated videos are subsequently processed by our contact-anchored real-to-sim pipeline.

Prompt template. Figure 6 shows the shared template. We instantiate <CLS> with a target object category without specifying an individual instance’s geometry, size, or pose, allowing the video model to generate variation within each category and adapt the human interaction accordingly.

<VIDEO>   
In a real-world continuous footage. Preserve the reference video’s   
original background, lighting and camera viewpoint.   
replace the box with <CLS>, pick it up and carry with two hands.  
Figure 6: Video-to-video prompt template. <VIDEO> denotes the seed video, and <CLS> specifies a box, bin, barrel, or ball. The highlighted instruction changes the manipulated object while allowing the human interaction to adapt.

## B Real-World Test Objects

We test unseen instances from the four generated object categories and objects from categories absent from the counterfactual videos (Fig. 7). No scans or reconstructions of these real-world test objects are used for training. Following Sec. 5.2, each object is tested in five trials with varied initial poses and distances. A trial succeeds if the robot reaches and grasps the object, then carries it stably for at least 3 m without dropping it or falling. Table 2 reports the resulting success rates.

In-Domain Objects  
![](images/bc22226a34c92dba6ea2a503fb4195a562833d4d0152a09164179aefd917eff6.jpg)

Out-of-Domain Objects  
![](images/427f8cac68722b8c31c56a989f9d5c6dfe7f21c7b7b2cf263b8962d004cc7bb8.jpg)  
Figure 7: Real-world test objects. In-domain objects are unseen instances of boxes, bins, barrels, and balls. Out-of-domain objects belong to categories absent from the generated training videos. Both groups vary in appearance, geometry, weight, and scale.

## C Training Data and Simulation Evaluation

Reconstruction and retargeting. Of the 256 generated videos, 137 yield feasible robot–object trajectories through contact-anchored reconstruction and retargeting. The remaining sequences fail the constrained solver, mainly because of severe collisions. In particular, some generated objects are too large for the Unitree G1 humanoid to carry without object–body collisions.

We train the privileged co-tracking teacher on these 137 trajectories for 40K iterations and obtain 129 successful teacher rollouts. These rollouts provide the robot and object motion references for student distillation, replacing the original kinematic trajectories with interactions executed in simulation. We train the unified depth-based student for 28K iterations using this set of 129 demonstrations.

Reported evaluation setting. The cross-domain results in the main paper use an earlier student distilled from 80 successful rollouts of a 20K-iteration teacher checkpoint. Its PRISM-ID evaluation set comprises these 80 interactions. These results correspond to the earlier training configuration; results for the updated student trained on 129 demonstrations are not included in that comparison.

Simulation evaluation data. For the OMOMO comparison, we retain 63 human–object interaction sequences and add an object-scaled variant of each using OmniRetarget [6]. The resulting 126 sequences are split into 90 training sequences and 36 held-out test sequences. We additionally reconstruct 48 PRISM out-of-domain interactions, with 12 instances each from tables, chairs, lamps, and monitors. These held-out categories are absent from the generated training data.

Simulation evaluation protocol. We evaluate the distilled student in MuJoCo [28], rendering depth observations and deriving joystick commands from the reference motion. Success requires completing pick–carry–drop and placing the object at the desired target position. This simulation metric includes the final placement, whereas the real-world metric above measures reaching, grasping, and stable carrying over at least 3 m.

## D Training and Distillation Hyperparameters

Both teacher and student policies use 3-layer MLPs and are trained for 40K and 28K iterations, respectively. Each stage runs 4096 environments per GPU on 8 NVIDIA L40S GPUs (32,768 environments in total). Tables 4–6 report observations, rewards, and hyperparameters. Object physics and payload sensitivity are detailed in Appendix E.

Depth observations and deployment. The student actor receives proprioception, joystick commands, and onboard depth. Joystick commands specify a relative planar root position and yaw, together with a binary drop command. During training, these are derived from the reference root trajectory and carry-end time. Depth is encoded by a small CNN into a 32-dimensional feature. At deployment, Fast-FoundationStereo [27] runs on an external computer linked to the robot via Ethernet, estimating depth from the head-mounted D435i stereo images. We find that this substantially reduces the sim-to-real gap in depth observations. The camera operates at 30 Hz and the policy at 50 Hz; the student transfers zero-shot without reference motion or external motion capture.

Distillation and initialization. We combine DAgger imitation with PPO, progressively increasing the PPO coefficient and retaining λ = 0.9 during the final 20K iterations. Successful teacher rollouts supply the motion references. We first train a student for 23K iterations using the same method on box-only data. We then restore only its actor weights, leaving the critic and optimizer freshly initialized, and train for 28K iterations on boxes, bins, barrels, and balls. Training from scratch also succeeds; warm starting accelerates convergence and is used for our released checkpoint. We randomize camera pose and corrupt depth with noise, dropout, holes, edge artifacts, and offsets.

Contact rewards. Both teacher training and the student’s PPO objective use object-frame contact anchors from reconstruction and retargeting. Rewards encourage reaching these targets with sufficient contact force. Table 5 lists the teacher reward weights and scales.

## D.1 Observation and Reward Specifications

Table 4: Observation spaces. T/S denote teacher/student, and A/C denote actor/critic. $\checkmark ^ { \dagger }$ denotes privileged student-critic inputs that are not used by the deployment policy.
<table><tr><td>Input</td><td>Dim.</td><td>T-A</td><td>T-C</td><td>S-A</td><td>S-C</td></tr><tr><td colspan="6">Commands and references</td></tr><tr><td>Motion command  $m _ { t }$ </td><td>58</td><td>√</td><td>√</td><td></td><td> $\checkmark ^ { \dagger }$ </td></tr><tr><td>Reference root position</td><td>3</td><td>一</td><td>√</td><td>一</td><td> $\checkmark ^ { \dagger }$ </td></tr><tr><td>Reference root orientation</td><td>6</td><td>√</td><td>√</td><td>一</td><td> $\surd ^ { \dagger }$ </td></tr><tr><td>Tracked body positions</td><td>42</td><td>一</td><td>√</td><td>一</td><td> $\surd ^ { \dagger }$ </td></tr><tr><td>Tracked body orientations</td><td>84</td><td>一</td><td>√</td><td>一</td><td> $\checkmark ^ { \dagger }$ </td></tr><tr><td>Sparse root command  $c _ { t } ^ { \mathrm { j s } }$ </td><td>3</td><td>1</td><td>一</td><td>√</td><td></td></tr><tr><td>Drop button  $b _ { t } ^ { \mathrm { d r o p } }$ </td><td>1</td><td>一</td><td>一</td><td>√</td><td></td></tr><tr><td colspan="6">Robot proprioception and actions</td></tr><tr><td>Base linear velocity</td><td>3</td><td>一</td><td>√</td><td>一</td><td> $\surd ^ { \dagger }$ </td></tr><tr><td>Base angular velocity</td><td>3</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Joint positions</td><td>29</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Joint velocities</td><td>29</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Previous action</td><td>29</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td colspan="6">Object state and perception</td></tr><tr><td>Object pose  $( p _ { o } , R _ { o } )$ </td><td>9</td><td>√</td><td>√</td><td></td><td> $\checkmark ^ { \dagger }$ </td></tr><tr><td>Target pose  $( p _ { g } , R _ { g } )$ </td><td>9</td><td>√</td><td>√</td><td>一</td><td> $\checkmark ^ { \dagger }$ </td></tr><tr><td>Object size</td><td>3</td><td>√</td><td>√</td><td>一</td><td> $\surd ^ { \dagger }$ </td></tr><tr><td>Object linear velocity</td><td>3</td><td>一</td><td>√</td><td>一</td><td> $\surd ^ { \dagger }$ </td></tr><tr><td>Object angular velocity</td><td>3</td><td>一</td><td>一</td><td></td><td> $\checkmark ^ { \dagger }$ </td></tr><tr><td>Depth image / CNN latent</td><td> $5 8 \times 8 7  3 2$ </td><td>一</td><td>一</td><td>√</td><td></td></tr><tr><td>Total input Deployment</td><td></td><td>175 train only</td><td>310 train only</td><td> $9 4 + 3 2 = 1 2 6$  √</td><td>313</td></tr></table>

Table 5: Main co-tracking reward weights for teacher training. Tracking terms use exponential kernels with the listed scale parameters.
<table><tr><td>Reward term</td><td>Weight</td><td>Parameters</td></tr><tr><td>Root position tracking</td><td>0.5</td><td> $\sigma = 0 . 3$ </td></tr><tr><td>Root orientation tracking</td><td>0.5</td><td> $\sigma = 0 . 4$ </td></tr><tr><td>Full-body position tracking</td><td>1.0</td><td> $\sigma = 0 . 3$ </td></tr><tr><td>Full-body orientation tracking</td><td>1.0</td><td> $\sigma = 0 . 4$ </td></tr><tr><td>Body linear velocity tracking</td><td>1.0</td><td> $\sigma = 1 . 0$ </td></tr><tr><td>Body angular velocity tracking</td><td>1.0</td><td> $\sigma = 3 . 1 4$ </td></tr><tr><td>Object position tracking</td><td>1.0</td><td> $\sigma = 0 . 3$ </td></tr><tr><td>Object orientation tracking</td><td>1.0</td><td> $\sigma = 0 . 4$ </td></tr><tr><td>Offline wrist target guidance</td><td>5.0</td><td>Object-frame contact points;  $\sigma = 0 . 0 8$ </td></tr><tr><td>Offline force-gated contact guidance</td><td>10.0</td><td>Force threshold 1.0; force  $\sigma = 1 0 . 0 ; \mathrm { c o n - }$  tact schedule relaxation 5 steps</td></tr><tr><td>Action-rate penalty</td><td>-0.1</td><td>Squared action difference</td></tr><tr><td>Joint-limit penalty</td><td>-10.0</td><td>Soft joint limit 0.9</td></tr><tr><td>Foot/ankle object-contact penalty</td><td>-0.5</td><td>Threshold 1.0</td></tr><tr><td>Lower-body undesired contact penalty</td><td>-0.1</td><td>Threshold 1.0</td></tr></table>

## D.2 Optimization and Distillation Settings

Table 6: Training and distillation hyperparameters. Settings for the teacher and 129-rollout student; Appendix C describes the earlier 80-rollout evaluation.
<table><tr><td>Hyperparameter</td><td>Co-tracking teacher</td><td>Depth-based student</td></tr><tr><td>Training data</td><td>Contact-anchored retargeted trajectories</td><td>Success-filtered teacher rollouts</td></tr><tr><td>Number of clips</td><td>137</td><td>129</td></tr><tr><td>Total environments</td><td>4096 × 8</td><td>4096 × 8</td></tr><tr><td>Policy input</td><td>Motion command, robot state, object state</td><td>Sparse root command, proprioception, buttons, depth</td></tr><tr><td>Object input</td><td>Current pose, target pose, object size</td><td>Depth observations; no explicit object state</td></tr><tr><td>History stacking</td><td>None</td><td>None</td></tr><tr><td>Actor network</td><td>MLP [512, 256, 128]</td><td>3-layer MLP with depth latent input</td></tr><tr><td>Critic network</td><td>MLP [512, 256, 128]</td><td>MLP with privileged state</td></tr><tr><td>Activation</td><td>ELU</td><td>ELU</td></tr><tr><td>Depth input</td><td>None</td><td>106 × 60 warped to 58 × 87</td></tr><tr><td>Depth encoder</td><td>None</td><td>Small CNN, 32-D latent</td></tr><tr><td>Camera range</td><td>None</td><td>0.3-3.0m</td></tr><tr><td>Camera pitch</td><td>None</td><td>37°</td></tr><tr><td>Depth noise</td><td>None</td><td>Hole prob. 0.2, noise std. 0.03m, offset std. 0.03m</td></tr><tr><td>Rollout length / minibatches</td><td>24/4</td><td>24/4</td></tr><tr><td>Learning epochs</td><td>7</td><td>5</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Actor / critic learning rate</td><td>1.0 × 10−5 / 1.0 × 10−5</td><td> $\mathrm { 7 . 0 \times 1 0 ^ { - 5 } / 7 . 0 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Discount factor</td><td>γ = 0.99</td><td>γ = 0.99</td></tr><tr><td>GAE parameter</td><td>λ = 0.95</td><td>λ = 0.95</td></tr><tr><td>PPO clip range</td><td>0.2</td><td>0.2</td></tr><tr><td>Value loss coefficient</td><td>1.0</td><td>1.0</td></tr><tr><td>Entropy coefficient</td><td>0.005 -3</td><td>0</td></tr><tr><td>Weight decay</td><td>10</td><td>Not used</td></tr><tr><td>Max gradient norm</td><td>1.0</td><td>1.0</td></tr><tr><td>Initial policy std.</td><td>1.0</td><td>0.01</td></tr><tr><td>Training iterations</td><td>40,000</td><td>28,000</td></tr><tr><td>Distillation loss</td><td>None</td><td>MSE behavior cloning</td></tr><tr><td>BC / DAgger coefficients</td><td>None / None</td><td>1.0/1.0</td></tr><tr><td>PPO coefficient schedule</td><td>None</td><td>0.1 → 0.9, step 0.1 every 500 iterations</td></tr><tr><td>Teacher-action rollout mix</td><td>None</td><td>0</td></tr><tr><td>Teacher action clip</td><td>None</td><td>8.0</td></tr><tr><td>Contact-aware command</td><td>None</td><td>Peak-height mode, α = 0.91, smoothing 5 steps</td></tr><tr><td>Sampling curriculum</td><td>Success-rate adaptive clip weighting</td><td>Adaptive timestep sampling</td></tr><tr><td>Warm start</td><td></td><td>Box-only, 23K iterations; actor weights only</td></tr><tr><td>Reset schedule</td><td>Random episode initialization</td><td>Start-at-zero prob. 0.2 → 1.0</td></tr><tr><td>Default-pose prepend</td><td>0.2s</td><td>Not used</td></tr><tr><td>Episode length</td><td>Motion-dependent</td><td>8.0s</td></tr><tr><td>Domain randomization</td><td>Object physics (Appendix E), restitution, COM, joint bias, pushes</td><td>Object physics (Appendix E), restitution, COM, joint bias, camera noise, pushes</td></tr><tr><td>External pushes</td><td>Interval [0.5, 2.0]s; max velocity [0.7, 0.7, 0.25, 0.7, 0.7, 1.0]</td><td>Interval [0.5, 2.0]s; max velocity [0.7, 0.7, 0.25, 0.7, 0.7, 1.0]</td></tr></table>

## E Object Physics and Payload Sensitivity

Category-specific masses $m _ { 0 }$ (Table 7) and mesh-derived inertias $\mathbf { I } _ { 0 }$ scale as $( m , { \bf I } ) = s ( m _ { 0 } , { \bf I } _ { 0 } )$ $s \sim \mathcal { U } ( 0 . 3 3 , 3 . 0 )$ . Shared friction is static $\mu _ { s } \sim \mathcal { U } ( 0 . 1 , 0 . 7 )$ and dynamic $\mu _ { d } = r \mu _ { s }$ , with r ∼ U(0.7, 0.99). We do not identify instance-specific surface properties or mass distributions.

Table 7: Object masses (kg). Each mesh’s inertia shares the mass scaling factor.
<table><tr><td>Category</td><td>Balls</td><td>Bins</td><td>Boxes</td><td>Barrels</td></tr><tr><td>Nominal mass</td><td>0.5</td><td>1.0</td><td>1.0</td><td>1.5</td></tr><tr><td>Training range</td><td>[0.165, 1.50]</td><td>[0.330, 3.00]</td><td>[0.330, 3.00]</td><td>[0.495, 4.50]</td></tr></table>

Payload sensitivity. Real-world test objects vary in shape and mass distribution, weighing 0.11 kg (box) to 4.99 kg (chair). Table 8 evaluates the earlier 80-rollout student (Appendix C).

Table 8: Payload sensitivity. Simulation success (%) across five payload ranges.
<table><tr><td>Payload (kg)</td><td>0.1-1.0</td><td>1.0-2.0</td><td>2.0-3.0</td><td>3.0-4.0</td><td>4.0-5.0</td></tr><tr><td>PRISM-ID</td><td>98.75</td><td>97.50</td><td>96.25</td><td>93.75</td><td>91.25</td></tr><tr><td>PRISM-OOD</td><td>77.08</td><td>72.92</td><td>68.75</td><td>62.50</td><td>64.58</td></tr></table>

## F V2V versus Geometric Augmentation

These submission results use eight seed videos (Seed-1X). Seed-17X adds 16 yaw/xy/scale variants per seed $( 8 \times 1 7 = 1 3 6 ;$ 103 retained after depth-camera visibility filtering). All methods share training budgets and evaluation protocols. V2V uses the earlier 80-rollout student; these results do not evaluate the current four-seed, 129-rollout configuration.

Table 9: V2V ablation at submission. Success (%); training sizes in parentheses.
<table><tr><td>Train\Eval</td><td>Seed-1X</td><td>Seed-17X</td><td>PRISM-ID</td><td>PRISM-OOD</td></tr><tr><td>Seed-1X (8)</td><td>100.00</td><td>58.25</td><td>17.50</td><td>6.25</td></tr><tr><td>Seed-17X (103)</td><td>100.00</td><td>98.06</td><td>36.25</td><td>22.92</td></tr><tr><td>V2V (80)</td><td>100.00</td><td>100.00</td><td>96.25</td><td>72.92</td></tr></table>

V2V improves PRISM-ID/OOD performance with fewer demonstrations than Seed-17X.

## G Additional Pipeline Analysis

## G.1 Contact-Anchor Reliability

For two-handed carrying, 2D overlap gives contact timing; mesh intersections with the segment between palm centers give object-frame anchors (Sec. 3.2). Pose/mesh errors can cause penetration and IK failure. We retain non-penetrating IK solutions and full-trajectory teacher rollouts (Appendix C), without formal contact or feasibility guarantees.

## G.2 Pipeline Runtime

Table 10 reports mean times per 8-second clip. Reconstruction and retargeting use one NVIDIA L40S GPU; V2V generation latency is recorded separately.

Table 10: Pipeline runtime. Mean processing time per 8-second clip.
<table><tr><td>Stage</td><td>V2V generation</td><td>Reconstruction</td><td>Retargeting</td></tr><tr><td>Mean time (s)</td><td>191.76</td><td>414.5</td><td>71.6</td></tr></table>

## G.3 Additional Interaction Examples

Figure 8 shows ball-rolling and door-opening reconstructions given suitable contact points and object representations. The door uses SAM3D [22] for geometry and Codex for articulation. These qualitative examples are not policy or real-world evaluations; tool use, regrasping, and broader deformable-object manipulation and extreme dynamics as in ExoRecon [30] remain untested.

![](images/427db4bfcfcb9c777ae08f400b676e24e57f82eb26794f50b4c6ccd6a7e45f92.jpg)  
Figure 8: Additional interactions. Input frames and reconstructions for ball rolling (left) and door opening (right), with human–object contacts highlighted.