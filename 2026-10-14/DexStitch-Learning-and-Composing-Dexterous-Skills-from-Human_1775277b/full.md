# DexStitch: Learning and Composing Dexterous Skills from Human Videos via Contact-Aware Robotic Inpainting

Byeonggyeol Choi<sup>1,∗</sup>, Seongmin Jung<sup>1,2,∗</sup>, Unghui Nam<sup>1,∗</sup>, Youngbo Song<sup>1,∗</sup>, Sunhong Min<sup>3,∗</sup>, MinHyung Lee<sup>3</sup>, Byonghyo Shim<sup>1</sup>, and Yongseok Lee<sup>3,†</sup>

<sup>1</sup>Seoul National University <sup>2</sup>Robotics Lab, Hyundai Motor Company <sup>3</sup>DGIST

![](images/a1abb16cff0cbbbe0614d78f4fceedcf86539af61f2f7f272a5c8def2b8f2785.jpg)  
Fig. 1. System overview. We propose DexStitch, a system that cuts human videos into skills and actions, converts them into a dexterous robot dataset, and stitches the learned skills back together in a sequential order.

Abstract— A long-horizon manipulation task consists of primitive skills and the transitions connecting them. Learning skills that can be executed in sequence requires supervision for both, which is costly to collect on a robot. Human videos offer a scalable source of demonstrations, but transferring them to robots requires extracting robot-compatible observations and actions together with their skill structure. We present DexStitch, a framework that cuts long-horizon human videos into skills and transitions, converts them into aligned robot observation–action–skill data, and stitches the learned skills into a requested sequence. A skill classifier first segments the videos into manipulation skills and transitions. DexStitch then converts the human demonstrations into robot observations and actions while retaining their corresponding skill labels. During conversion, per-finger human–object contacts guide dexterous hand retargeting and robotic inpainting, preserving contact geometry and finger–object occlusion ordering. To stitch the learned skills into a sequence, a shared skill-conditioned policy is pretrained on the resulting synthetic data and finetuned on a mixture of synthetic data and a small set of single-skill teleoperation episodes. Conditioning on the current and next skills guides each transition toward the following skill, while predicted skill progression enables closed-loop switching. On a held-out session, the classifier agrees with manual annotations on 92.4% of frames. Across four household skills, DexStitch achieves 80.0% single-skill success and 77.5% sequence success.

## I. INTRODUCTION

Long-horizon manipulation tasks consist of primitive skill and the transitions connecting them: a robot tidying a desk picks up objects, opens drawers, and places items inside. Performing such tasks requires learning individual skills and composing them into a sequence. Successful composition also requires the motions that connect successive skills, since the state left by one skill may differ from the starting states encountered by the next during training [1]. The robot must also determine when a transition is complete and the next skill can begin [2]. Human videos naturally capture both manipulation skills and the transitions between them, offering a readily available source of demonstrations. To turn these demonstrations into robot-compatible data, we need to recover both how to act and which skill each action belongs to. This requires robot-compatible observations and actions that preserve the demonstrated interactions, together with aligned skill annotations that organize the sequential execution.

Collecting the action and skill supervision through robot teleoperation and manual annotation is costly, whereas human videos are easier to record but provide no robot actions and skill labels [3]. Existing approaches address this gap in two ways. One line learns a shared latent skill space across human and robot embodiments, enabling a policy trained on robot data to follow a human demonstration [4], [5]. These methods transfer skill structure through latent representations rather than producing explicitly skill-labeled robot observation–action pairs from human videos. Another line retargets human motion and replaces the human embodiment with a rendered robot, producing robot-domain demonstrations [6], [7], [8], [9]. However, this conversion does not by itself identify the skills and transitions within each trajectory. Together, these approaches motivate a framework that converts human demonstrations into robot-compatible observations and actions while making their skill structure explicit.

We present DexStitch (Fig. 1), which cuts long-horizon human videos into skills and transitions, converts them to the robot domain, and stitches them back together in a requested order. A user defines the skill set, including transitions as ordinary skills, and annotates a set of videos frame by frame. A classifier trained on these annotations segments the remaining videos, agreeing with held-out manual labels of frames. Because transitions are expensive to cover individually—one for every ordered skill pair—we record long-horizon human videos that contain many transitions at once.

Robot actions are obtained by converting the human videos. Retargeting to a dexterous hand must preserve both the contacting fingers and their contact points; due to handshape differences, matching fingertip positions alone may miss the points touched by the human. From RGB video alone, without an object mesh or simulator, DexStitch predicts per-finger surface contact and uses it to pull retargeted fingertips onto the contact points. It also determines which fingers are occluded by the object when inpainting the human and rendering the robot. Because conversion is frame-wise, the skill labels transfer directly to the synthetic robot data.

Our skill-conditioned policy learns both skills and transitions in two stages. It is pretrained on the synthetic data from manually and classifier-labeled videos, then finetuned with a small set of teleoperated single-skill episodes. Thus, transitions are supervised solely by human video. Conditioning on the current and next skills directs each transition toward the next skill’s initial state, while an auxiliary prediction of skill completion advances the sequence. At test time, the same classifier extracts a skill sequence from a human prompt video, and the robot executes the corresponding skills in order.

We evaluate DexStitch on four household manipulation skills. We first ask whether the synthetic robot data teaches executable skills, comparing against skill learning and compositing baseline [5]. Ablations remove each data source, the transition skill, and the next-skill conditioning. We then qualitatively compare the converted videos against Masquerade [7] and a variant without contact-guided occlusion handling. Finally, we measure how accurately the classifier segments held-out human video against two vision–language models.

Our contributions are threefold.

• Skill and action supervision from human video. We convert long-horizon human videos into robot-domain observation–action pairs while preserving their framelevel skill annotations, providing supervision for both manipulation skills and the transitions between them. Using RGB video alone, our pipeline produces robotexecutable training data without object meshes or simulation.

• Contact-aware human-to-robot conversion. We use per-finger human–object contact as a shared cue for retargeting and visual compositing, preserving multifinger grasp geometry while resolving finger–object occlusions during robot rendering.

• Sequential composition of reusable skills. We construct an explicit, reusable repertoire of robot-executable skills, including transitions, from the converted data. Our policy conditions on the current and next skills to execute target-aware transitions, and predicts skill completion for closed-loop sequential composition.

## II. RELATED WORK

## A. Learning Skill Structure from Human Video

Long-horizon human demonstrations contain behavioral structure that can be represented as manipulation skills. Prior work has studied such structure in play and unsegmented robot demonstrations, learning latent plans [10], reusable skill primitives [11], and continually expanding skill libraries [12]. More recent approaches extend this perspective to human video. MimicPlay [13] distills human play into latent plans that guide a robot controller trained on teleoperation. XSkill [4] and UniSkill [5] learn cross-embodiment latent skills from unlabeled human and robot videos, and sequence them to imitate a human prompt video in one shot. In both, the policy receives a continuous latent. XSkill’s prototypes cover the motion between manipulations too, but unlabeled, and neither policy is told which skill comes next. DexStitch takes the opposite choice: a skill set is fixed in advance, with the transition as one of its labels. Human videos are then automatically segmented into the labels.

## B. Transferring Human Videos to Robot Execution

Human videos provide abundant manipulation demonstrations [14], [15] but lack embodiment-specific robot actions and visually differ from observations encountered by the robot. Motion-retargeting approaches recover robotcompatible actions from human motion, but the corresponding frames still depict the human embodiment [16], [17], [18], [19]. These methods therefore rely on additional robot data to ground policy learning in robot-domain observations. Robotic inpainting instead removes the human embodiment and renders the retargeted robot into the source video [6], [7], [8], [9]. Phantom [6] and EgoEngine [9] pair the edited frames with the retargeted actions and train policies on them without robot demonstrations, while Masquerade [7] and H2R [8] use them to pretrain visual representations that robot demonstrations then ground. In all four, a demonstration is one trajectory, with no skill or transition boundaries marked. DexStitch also trains on the edited frames paired with the retargeted actions, and adds a small set of teleoperated episodes only in finetuning.

![](images/73ec6f1b8c746aac126589b65eed07c9240b3419abd89bb4e09049da19a9fbea.jpg)  
Fig. 2. Contact-aware Human-Robot Data Conversion Pipeline. Given an egocentric human demonstration, DexStitch estimates the hand pose and hand–object contact while removing the human embodiment from the input frames. The contact information then guides robot retargeting and determines the object–robot occlusion order during compositing. Together, the retargeted robot and the inpainted scene produce a robot-domain visual sequence.

Compositing must also order the rendered robot in depth against the object it holds, or fingers that should curl behind the object are drawn in front of it. Phantom [6] masks the overlay with depth from an RGB-D camera, and Ego-Engine [9] recovers the ordering from a reconstructed digital twin, at the cost of an object mesh, per-frame depth, and a physics simulator for the actions. Contact-aware retargeting methods recover the demonstrated interaction geometry during motion transfer [20], [21], [22], but also take their contact from an object model, scanned or reconstructed, and use it only for the actions. DexStitch estimates per-finger contact on the hand surface from the RGB video alone [23], with neither depth nor an object model, and uses the same signal both to refine the retargeted hand configuration and to identify which fingers pass behind the object during compositing.

## C. Skill Composition and Transition

Composing skills in sequence means each skill starts where the previous one ended, which may be a state it never saw in training [24]. Prior work bridges this gap with dedicated transition policies [1], a learned transition-feasibility function that finetunes and switches sub-policies [2], or a motion planner that carries the end-effector between atomic skills served by one multi-task policy [25]. Other approaches compose discrete skills without a transition module: SDP [26] lets a router assign a skill to each state, which conditions a shared diffusion policy, and SOF [27] plans over skills learned from action-free video, recovering actions with a separate module. In both, the transition is implicit. DexStitch labels the transition as a skill and runs it with the same policy, so it is neither implicit nor a separate module. The policy is told which skill comes next.

## III. HUMAN-ROBOT DATA CONVERSION

DexStitch converts egocentric human video into robotdomain observation–action pairs. Conversion proceeds in three stages (Fig. 2): estimating the hand motion and contact, retargeting it to the robot, and rendering the robot into the frames in place of the human. The converted video and the retargeted robot trajectory constitute the training data for the policy. Because conversion is frame-wise, each converted frame keeps the skill label of its source frame.

## A. Hand Pose and Contact Estimation

We first reconstruct the 3D hand motion with HaWoR [28], which fits the MANO hand model [29] in camera coordinates, giving the hand keypoints J<sub>t</sub> and the wrist pose $\mathbf { T } _ { \mathrm { w r i s t } , t } ^ { \mathrm { c a m } }$ at time t. We then estimate human–object contact with HaCo [23], which predicts a contact probability over the MANO vertices. Vertex probabilities are pooled by finger and thresholded with hysteresis to obtain temporally stable perfinger contact states. For each contacting finger, we retain its contacting vertices as $\mathbf { C } _ { t } \subset \mathbb { R } ^ { 3 }$ . These contact points constrain the retargeting in Sec. III-B and the occlusion ordering in Sec. III-C.

## B. Contact-Aware Retargeting

We derive the robot trajectory from the reconstructed hand motion: the keypoints $\mathbf { J } _ { t }$ determine the robot hand articulation, while the wrist pose $\mathbf { T } _ { \mathrm { w r i s t } , t } ^ { \mathrm { c a m } }$ specifies the target pose of the robot wrist. We retarget the hand in two stages. The first stage applies DexPilot [30], which maps the MANO keypoints $\mathbf { J } _ { t }$ to the robot hand by matching wrist-to-fingertip vectors. This provides a kinematic initialization, but the robot fingertips are not guaranteed to reach the demonstrated contact points. The second stage therefore refines the robot joint angles using the estimated contacts $\mathbf { C } _ { t }$ . For each contacting finger, we minimize the distance from each human contact point to the nearest point on the robot fingertip surface, restricted to surface points whose normal is aligned with the normal at the human contact point. A quadratic regularizer keeps the solution close to the first stage, and the optimization is solved with L-BFGS-B [31], giving the hand configuration ${ \bf q } _ { t } ^ { \mathrm { h a n d } }$

For the arm, the fixed camera-to-base transform $\mathbf { T } _ { \mathrm { b a s e } } ^ { \mathrm { c a m } }$ is first calibrated once for the setup with EasyHeC [32]. We then solve inverse kinematics for $\mathbf { T } _ { \mathrm { w r i s t } , t } ^ { \mathrm { c a m } }$ to obtain the arm joint angles ${ \bf q } _ { t } ^ { \mathrm { a r m } }$ , and stack the two:

$$
\mathbf { x } _ { t } = [ \mathbf { q } _ { t } ^ { \mathrm { a r m } } \ | \ \mathbf { q } _ { t } ^ { \mathrm { h a n d } } ] .
$$

![](images/f032890e40a32c2ea984a275c8675486ba749c068feab09564596aaf9254b891.jpg)  
Fig. 3. Skill Learning and Composition Pipeline. (a) A frozen V-JEPA 2 encoder and a lightweight classification head recover the ordered skill sequence from a human demonstration. (b) A shared skill-conditioned policy predicts robot actions and skill progression from the current observation and the current–next skill pair. (c) At inference time, the policy executes the given skill sequence in closed loop, advancing to the next entry according to its predicted skill progression.

Because each frame is processed independently, we smooth the per-frame configurations over time with a median filter followed by a Savitzky–Golay filter. The result is the robot trajectory

$$
{ \mathbf X } _ { R } = \{ { \mathbf x } _ { t } \} _ { t = 1 } ^ { T } ,
$$

temporally aligned with the human demonstration over its T frames.

## C. Inpainting and Compositing

To produce robot-domain observations aligned with the retargeted trajectory, we remove the human embodiment from each source frame and composite a rendering of the robot while preserving the scene and manipulated objects. SAM2 [33] tracks two sets of masks through the video: the human hand and arm, and the manipulated objects. We inpaint the human mask with ProPainter [34], keeping the object mask protected. The robot is then rendered at configuration $\mathbf { X } _ { t } ,$ placed by $\mathbf { T } _ { \mathrm { b a s e } } ^ { \mathrm { c a m } }$ and viewed with the intrinsics of the camera that recorded the demonstration.

A direct overlay can place the rendered robot entirely on top of the manipulated object, producing incorrect occlusion around grasping fingers. To determine which fingers are behind the object, we take the per-finger contact states from Sec. III-A. We assume that when the hand wraps an object, the thumb lies in front of it and the other fingers curl behind, forming an anti-podal grasp. The non-thumb contacting fingers are therefore the ones to hide: we project their 3D contact points into the image and dilate them into a connected region. Wherever this region overlaps the object mask, the object is drawn in front of the robot.

Each frame is composited in four layers, from back to front: the inpainted background, the robot fingers the object hides, the object taken from the source frame, and the remaining robot pixels. The object layer is copied from the source rather than the inpainted plate, since inpainting alters exactly the region where the hand met the object. Applying this procedure to all frames produces the converted visual sequence $\{ \tilde { I } _ { t } ^ { R } \} _ { t = 1 } ^ { T }$ , where the tilde marks a quantity converted from the human video and the superscript its robot domain. Paired with the retargeted robot trajectory $\mathbf { X } _ { R } ,$ , these frames provide robot observations $\tilde { o } _ { t } ^ { R }$ and action targets $\tilde { a } _ { t } ^ { R } :$ , yielding

$$
\tilde { \tau } = \{ ( \tilde { o } _ { t } ^ { R } , \tilde { a } _ { t } ^ { R } ) \} _ { t = 1 } ^ { T } .
$$

## IV. SKILL LEARNING AND COMPOSITION

We learn to imitate human-demonstrated skills on the robot using the synthetic robot data of Sec. III-C together with a small set of real robot demonstrations. Our framework consists of a skill classifier and a skill-conditioned policy, which are jointly used at inference time to execute and compose the demonstrated skills.

## A. Skill Classifier

We train a skill classifier to recover the sequence of skills from a human demonstration. The skill set includes both manipulation behaviors and the transition behavior connecting them, with transition treated as an ordinary skill label.

1) Architecture Design and Training: We keep the pretrained V-JEPA 2 video encoder [35] frozen and attach a small MLP head to its representation. The encoder emits one token per pair of frames sampled at 4 FPS, so the head sees the video at $2 \mathrm { H z } ;$ it reads a causal window of the eight most recent tokens, four seconds of context, and predicts the skill at the latest one. Freezing the encoder leaves only 296K trainable parameters, which is what allows the classifier to be fitted from the handful of annotated sessions available.

Given a long horizon human video, the classifier is trained to predict a skill label at each token,

$$
\hat { s } _ { t } ^ { H } = C _ { \phi } ( I _ { t } ^ { H } ) ,
$$

and each token’s label is held across the frames it was computed from.

2) Skill Annotation Generation: With the trained skill classifier, we annotate the human video with the skill labels. We represent a human demonstration as a video $\{ I _ { t } ^ { H } \} _ { t = 1 } ^ { T }$ and annotate per-frame skill labels $\{ s _ { t } \} _ { t = 1 } ^ { T }$ , where $s _ { t } \in \mathbf { S }$ We smoothen the frame-wise sequence with temporal filters. This consolidates consecutive predictions of the same label into an ordered sequence of K skill segments

$$
q = ( s _ { 1 } ^ { \prime } , . . . , s _ { K } ^ { \prime } ) .
$$

The generated skill labels are then combined with the synthetic robot data to train the robot-executable policy.

## B. Skill-Conditioned Policy

To reproduce the skills observed in human demonstrations, we learn a single policy that maps each demonstrated skill to the corresponding robot behavior. The synthetic robot data of Sec. III-C provides robot observations and actions together with their corresponding human skill labels,

$$
\mathbf { D } _ { \mathrm { s y n } } = \{ ( \tilde { o } _ { t } ^ { R } , \tilde { a } _ { t } ^ { R } , s _ { t } ) \} _ { t = 1 } ^ { T } ,
$$

from which the policy learns both individual skills and the motions connecting them.

1) Architecture Design: We condition a shared policy on the current and subsequent skills to support their sequential execution. Given the current robot observation $o _ { t } ,$ , current skill $s _ { t } ,$ and next skill $s _ { t } ^ { \mathrm { n e x t } }$ , the policy predicts an action chunk of H steps together with its skill progression:

$$
\pi _ { \boldsymbol { \theta } } \left( a _ { t : t + H } , \hat { s } _ { t : t + H } \mid o _ { t } , s _ { t } , s _ { t } ^ { \mathrm { n e x t } } \right) .
$$

The current skill specifies the behavior to execute, while the next skill provides context for how the behavior should terminate and connect to what follows. The predicted labels $\hat { s } _ { t : t + H }$ capture the progression of the execution and are used for closed-loop skill switching during composition. We instantiate $\pi _ { \theta }$ with Diffusion Policy [36], where $o _ { t }$ consists of the robot camera observation and the current arm and hand configuration.

2) Training: We first train the policy exclusively on the synthetic robot data $\mathbf { D } _ { \mathrm { s y n } }$ . This phase lets the policy acquire the full skill repertoire and the inter-skill motions observed in the long-horizon human demonstration. Because $\mathbf { D } _ { \mathrm { s y n } }$ covers the complete demonstrated sequence, the policy learns the transitions without any teleoperated demonstration of them.

We then continue training on a mixture of $\mathbf { D } _ { \mathrm { s y n } }$ and a small set of teleoperated robot demonstrations $\mathbf { D } _ { \mathrm { t e l e } }$ . The real robot data grounds the pretrained policy in the target embodiment and corrects residual discrepancies introduced during conversion, while the synthetic robot data maintains coverage of the human-derived skills and transitions. This mixed training combines the broad supervision available from human videos with the execution fidelity of limited real-robot data.

## C. Compositional Inference

Given an new ordered sequence of skills whether by a human demonstration video or script, the robot executes the entries of skills $q$ in order using the skill-conditioned policy.

If a human demonstration is given, the skill classifier first produces an ordered sequence $\boldsymbol { q } = ( s _ { 1 } ^ { \prime } , \ldots , s _ { K } ^ { \prime } )$ , equivalent to the ordered sequence given by script. While executing $s _ { k } ^ { \prime } .$ the policy is conditioned on the current and next entries $( s _ { k } ^ { \prime } , \ddot { s } _ { k + 1 } ^ { \prime } )$ and predicts both robot actions and skill labels.

Execution advances through q using the policy’s skill predictions rather than a fixed duration. When the policy consistently predicts the next skill $s _ { k + 1 } ^ { \prime } .$ the current skill is considered complete and the policy conditions are advanced to $( s _ { k + 1 } ^ { \prime } , s _ { k + 2 } ^ { \prime } )$ . We aggregate skill predictions over consecutive time steps to prevent isolated predictions from triggering premature switches. Repeating this process allows the robot to follow the skill composition extracted from the human demonstration in closed loop.

## V. EXPERIMENTS

## A. Experimental Setup

a) Hardware: We conduct all real-robot experiments using an RB5-850E robot arm equipped with an xHand dexterous hand. Visual observations are captured with an Intel RealSense D435i, of which only the RGB stream is used.

b) Tasks: We consider four household manipulation skills: picking and placing a green box, picking and placing a red box, handing over a cup, and stacking bowls.

c) Dataset: Our dataset consists of approximately 240 long-horizon human demonstration episodes in which the four skills are performed in varying orders. Our pipeline converts these into robot observation–action trajectories. We additionally collect a small set of teleoperated robot demonstrations—about 8 per skill (∼32 in total), each executing a single skill in isolation—which are used to finetune the policy.

d) Implementation: Our policy is a diffusion policy [36]. Visual features are extracted by a ResNet-18 encoder (ImageNet-pretrained), and the action sequence is denoised by a 1D convolutional U-Net conditioned on the diffusion timestep and the observation. The observation at each step is a $2 2 4 \times 2 2 4$ RGB image from the camera together with a 33-D proprioceptive-and-skill state: the wrist pose (3-D position and 6-D rotation), 12 hand joint angles, and the current- and next-skill one-hot vectors $( 6 + 6 )$ . The policy predicts a 33-D target of the same layout; at execution the 6-DoF arm command is obtained by inverse kinematics from the predicted wrist pose and the 12 hand joints are commanded directly, yielding an 18-DoF robot action. We use an observation history of 2 steps and a prediction horizon of 32, of which the first 16 actions are executed, and run the policy at 20 Hz. The forward process uses a DDPM schedule; at inference we denoise with 10 DDIM steps.

We train with Adam $( \beta = ( 0 . 9 5 , 0 . 9 9 9 )$ , weight decay $1 0 ^ { - 6 } )$ , a cosine learning-rate schedule, batch size 32, and mixed precision on a single NVIDIA A100. The policy is first pretrained for 120k steps on the converted human→robot trajectories (240 episodes) at a learning rate of $1 0 ^ { - 4 }$ , then finetuned for 70k steps at $2 . 5 \times 1 0 ^ { - 5 }$ . For finetuning we mix the 32 teleoperated episodes, so that teleoperated data constitutes various ratio(50%,75%,90%,100%) of the finetuning frames.

![](images/e6a95a1259c6c0cae964178054214595470422dcf839337f31f1f586a81f9956.jpg)  
Ground Truth  
Ours  
Ours w/o occlusion  
Masquerade

Fig. 4. Synthetic robot observations at the moment of grasp. Three grasps from one demonstration: holding a sponge (top), lifting a carton (middle), and hanging a mug on the rack (bottom). Ours generates realistic robot demonstration images, preserving the hand–object occlusion that the other two lose.  
![](images/9e682f37e7e606699a00b0b7bc9faa1c8418984fd9dedc2bb14b28767ef6bfe7.jpg)

![](images/63bdbc4e69c1f0326b592e314ecd4d7691db5637c0dc4e4b2f1ec0fef1a11cf5.jpg)  
Fig. 5. End-effector trajectories for the redbox skill. Robot base-frame trajectories (m) of the teleoperated demonstrations (gray), our policy (blue), and UniSkill (red), shown in 3D and top-down (x–y) views.

e) Baselines: We compare against two baselines targeting complementary parts of our system. For human-torobot data conversion, we compare with Masquerade [7]. For reusable skill learning and composition, we compare with UniSkill [5]. For a controlled comparison, UniSkill uses the same 120k-step pretraining and 70k-step finetuning on the same converted human and teleoperated data as ours, differing only in the skill representation.

## B. Skill Learning and Composition

We evaluate the learned skills in two settings against UniSkill [5] under the same skill set (Table I): a singleskill setting, in which each skill is executed on its own for 5 rollouts, and a sequence setting, consisting of 10 four-skill sequence rollouts (i.e., 40 skill executions, 10 per skill).

On single skills, UniSkill performs moderately (45.0% average), showing that its skill representation captures the individual behaviors. However, UniSkill provides no explicit mechanism for the transitions between skills, and at the sequence level—where the policy must move from one skill to the next—its success collapses to 0%. DexStitch instead models transitions through an explicit transition skill together with next-skill conditioning, and stays reliable both on single skills (80.0%) and in sequences (77.5%). Figure 5 illustrates this for the RedBox skill: DexStitch’s end-effector trajectories closely follow the teleoperated demonstrations, whereas UniSkill’s, though broadly similar in shape, remain offset from the demonstrated motion and do not complete the grasp. We attribute this gap to the skill representation rather than to data or training budget: both methods use the same synthetic robot data, the same 120k-step pretraining, and the same teleoperated finetuning, and differ only in how skills are represented—in particular, whether the transitions between skills are modeled.

## C. Conversion Quality

We qualitatively compare our synthetic robot observations with Masquerade [7] and a variant of DexStitch without contact-guided occlusion handling, on the same source demonstrations and with particular emphasis on regions where the robot hand interacts with manipulated objects. Figure 4 shows three representative grasps—holding a sponge, lifting a snack carton, and hanging a mug—at the moment of contact.

Masquerade’s parallel-jaw gripper cannot express the demonstrated dexterous grasps and composites the gripper over the object it holds, so the resulting images preserve neither the intended hand configuration nor the hand–object occlusion. Removing contact-guided occlusion handling retains the retargeted dexterous hand but drops the occlusion reasoning: the hand is drawn over the object, and the grasp reads as hovering rather than closing. Our full method both reproduces the demonstrated finger configuration and renders the object over the fingers whose contact points fall inside its mask, yielding images that read as a robot actually performing the demonstrated grasp. These comparisons indicate that contact provides a useful cue for preserving interaction consistency when converting human demonstrations to the robot embodiment, particularly around hand–object interactions where human and robot geometry differ.

TABLE I  
SKILL LEARNING AND COMPOSITION. SUCCESS RATE (%) FOR EACH SKILL, EVALUATED BOTH IN ISOLATION (SINGLE-SKILL, 5 TRIALS EACH) AND WITHIN FOUR-SKILL SEQUENCES. CUP: HANGING A CUP; STACK: STACKING BOWLS.
<table><tr><td></td><td colspan="5">Single-skill</td><td colspan="5">Sequence</td></tr><tr><td>Method</td><td>Green</td><td>Red</td><td>Cup</td><td>Stack</td><td>Avg.</td><td>Green</td><td>Red</td><td>Cup</td><td>Stack</td><td>Total</td></tr><tr><td>UniSkill [5]</td><td>80</td><td>20</td><td>20</td><td>60</td><td>45.0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>Ours (DexStitch)</td><td>100</td><td>100</td><td>40</td><td>80</td><td>80.0</td><td>80</td><td>70</td><td>80</td><td>80</td><td>77.5</td></tr></table>

TABLE II

SKILL CLASSIFICATION ON THE HELD-OUT SESSION. ALL METHODS ANSWER THE SAME CLOSED-SET QUESTION UNDER AN IDENTICAL PROTOCOL (SEE TEXT)
<table><tr><td>Method</td><td>Acc. ↑ Macro-F1 ↑ Edit ↑ F1@50 ↑</td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-VL-7B (zero-shot)</td><td>65.8</td><td>66.0</td><td>70.3</td><td>41.0</td></tr><tr><td>InternVL3-8B (zero-shot)</td><td>57.8</td><td>56.7</td><td>47.5</td><td>28.5</td></tr><tr><td>V-JEPA 2 + MLP (Ours)</td><td>92.4</td><td>92.7</td><td>99.6</td><td>99.0</td></tr></table>

![](images/ec1e8e6308b167891814e59c9ecce747685896bc31a835de02a882618b76afad.jpg)  
Fig. 6. Confusion Matrix of Skill Classifier.

## D. Skill Classifier Evaluation

We compare our classifier against Qwen2.5-VL-7B [37] and InternVL3-8B [38] prompted zero-shot with the same skill set. Both baselines are queried at the same 2 Hz decision times from the same causal window, and all three are scored by the same code on the same frames. The VLM baselines see eight 256-token frames and none abstain. Of the 240 human episodes, 216 are used to train the classifier and the remaining 24 for evaluation (18,792 frames). Besides accuracy and macro-F1 we report segmentation quality, since the classifier exists to cut demonstrations into segments rather than to label isolated moments: edit score compares the predicted skill sequence while ignoring extent, and F1@50 requires at least half overlap with the ground-truth segment.

Table II shows the gap widening from roughly thirty points on frame accuracy to a factor of two to three on F1@50. The baselines label over half the frames correctly but cannot hold one label across a skill, so their predictions fragment and the boundaries do not survive an overlap test; ours recovers 242 segments against 241 in the ground truth, matching the skill order in every episode but one. The failure concentrates on transitions (F1 0.89 ours, 0.41 Qwen, 0.18 InternVL, Fig. 6): with an object in hand the baselines commit to a manipulation skill rather than recognizing motion between skills. Indeed, the baselines classify the manipulation skills far more accurately than the transitions (Fig. 6), so what they lack is not visual recognition of the skills but a judgement about temporal state. Since compositional inference consumes segment boundaries rather than individual frame labels, this margin is what makes the extracted sequences usable.

TABLE III  
ABLATION STUDY. OVERALL TASK SUCCESS (%) IN THE FOUR-SKILL sequence SETTING. OURS USES A 90% TELEOPERATION RATIO WITH NEXT-SKILL CONDITIONING.
<table><tr><td>Variant</td><td>Task Success</td></tr><tr><td>Training data w/o Human data w/o Teleop data</td><td>35.0 5.0</td></tr><tr><td>Skill conditioning w/o Transition w/o Next Skill</td><td>10.0</td></tr><tr><td>Finetuning ratio 50% teleop</td><td>7.5 10.0</td></tr><tr><td>75% teleop 100% teleop</td><td>20.0 50.0</td></tr><tr><td>Ours</td><td>77.5</td></tr></table>

## E. Ablation Studies

All ablations are evaluated in the four-skill sequence setting and compared against our full model (Table III); every variant uses the same 90% teleoperation finetuning ratio.

a) Training data: We ablate the two training data sources while keeping the architecture fixed (Table III). Using only teleoperated data (w/o Human data) reaches 35.0%, while using only the synthetic robot data (w/o Teleop data) reaches 5.0%; both fall far below the full model’s 77.5%. Neither source alone suffices: the synthetic robot data provides broad skill coverage, while the teleoperated data grounds the behaviors in the robot’s own execution.

b) Skill conditioning: We ablate the skill conditioning while keeping the training data unchanged (Table III). Removing the explicit transition skill (w/o Transition) lowers success to 10.0%, and removing next-skill conditioning (w/o Next Skill) to 7.5%. Both degrade most at transitions, where the same transition must lead to different subsequent behaviors depending on the skill that follows; the explicit transition skill and the next-skill signal give the policy the context to move toward the appropriate next skill.

c) Finetuning data ratio: We vary the proportion of teleoperated robot data in the finetuning mixture, using the synthetic robot data as the remainder. As shown in Table III, task success rises from 10.0% at a 50% teleoperation ratio to a peak of 77.5% at 90%, and then drops to 50.0% at 100% (teleoperation only). The peak at a 90% ratio shows that retaining a small fraction of the synthetic robot data during finetuning outperforms finetuning on teleoperated data alone. We therefore adopt this 90% ratio for DexStitch. This indicates that the synthetic robot data provides complementary supervision that teleoperated data alone does not, reducing the amount of robot-collected data required for successful execution.

## VI. CONCLUSION

We presented DexStitch, a framework for learning reusable robot skills from long-horizon human demonstrations. DexStitch converts long-horizon human demonstrations into robot-domain observation–action data using contact-aware retargeting and compositing, and learns manipulation and transition skills with a skill-conditioned policy. A skill classifier extracts skill sequences from human video, enabling the robot to execute skill sequences in closed loop. Our experiments show that human demonstrations can provide both executable robot supervision and reusable skill structure.

Two limits remain. First, the compositing assumes a wrap grasp, with the thumb in front of the object and the other fingers curled behind it, so pinch and top-down grasps fall outside that rule; covering a wider grasp taxonomy is left to future work. Second, composition is evaluated on a single fixed four-skill order, so evaluating orderings held out from training would test the compositional claim directly.

## REFERENCES

[1] Y. Lee, et al., “Composing complex skills by learning transition policies,” in Proceedings of International Conference on Learning Representations, 2019.

[2] Y. Chen, et al., “Sequential dexterity: Chaining dexterous policies for long-horizon manipulation,” in Proceedings of The 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Tan, et al., Eds., vol. 229. PMLR, 06–09 Nov 2023, pp. 3809–3829.

[3] K. Zakka, et al., “XIRL: Cross-embodiment inverse reinforcement learning,” in Conference on Robot Learning. PMLR, 2022, pp. 537– 546.

[4] M. Xu, et al., “XSkill: Cross embodiment skill discovery,” in Proceedings of The 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Tan, et al., Eds., vol. 229. PMLR, 06–09 Nov 2023, pp. 3536–3555.

[5] H. Kim, et al., “UniSkill: Imitating human videos via crossembodiment skill representations,” in 9th Annual Conference on Robot Learning, 2025.

[6] M. Lepert, et al., “Phantom: Training robots without robots using only human videos,” in Conference on Robot Learning (CoRL), Seoul, Korea, 2025.

[7] M. Lepert, et al., “Masquerade: Learning from in-the-wild human videos using data-editing,” in 2026 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2026.

[8] G. Li, et al., “H2R: A human-to-robot data augmentation for robot pre-training from videos,” arXiv preprint arXiv:2505.11920, 2025.

[9] Y. Liu, et al., “EgoEngine: From egocentric human videos to high-fidelity dexterous robot demonstrations,” arXiv preprint arXiv:2606.12604, 2026.

[10] C. Lynch, et al., “Learning latent plans from play,” in Conference on robot learning. Pmlr, 2020, pp. 1113–1132.

[11] Y. Zhu, et al., “Bottom-up skill discovery from unsegmented demonstrations for long-horizon robot manipulation,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4126–4133, 2022.

[12] W. Wan, et al., “LOTUS: Continual imitation learning for robot manipulation through unsupervised skill discovery,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 537–544.

[13] C. Wang, et al., “MimicPlay: Long-horizon imitation learning by watching human play,” in Proceedings ofThe 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023, pp. 201–221.

[14] K. Grauman, et al., “Ego4D: Around the world in 3,000 hours of egocentric video,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 18 995–19 012.

[15] S. Nair, et al., “R3M: A universal visual representation for robot manipulation,” in Proceedings of The 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 205. PMLR, 2023, pp. 892–909.

[16] S. Kareer, et al., “EgoMimic: Scaling imitation learning via egocentric video,” arXiv preprint arXiv:2410.24221, 2024.

[17] R.-Z. Qiu, et al., “Humanoid policy ∼ human policy,” in Proceedings ofThe 9th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 305. PMLR, 2025, pp. 2888–2906.

[18] R. Yang, et al., “EgoVLA: Learning vision-language-action models from egocentric human videos,” arXiv preprint arXiv:2507.12440, 2025.

[19] R. Zheng, et al., “EgoScale: Scaling dexterous manipulation with diverse egocentric human data,” arXiv preprint arXiv:2602.16710, 2026.

[20] X. Lin, et al., “DexFlow: A unified approach for dexterous hand pose retargeting and interaction,” arXiv preprint arXiv:2505.01083, 2025.

[21] Z. Mandi, et al., “DexMachina: Functional retargeting for bimanual dexterous manipulation,” arXiv preprint arXiv:2505.24853, 2025.

[22] J. Hsieh, et al., “DexMan: Learning bimanual dexterous manipulation from human and generated videos,” arXiv preprint arXiv:2510.08475, 2025.

[23] D. Jung et al., “Learning dense hand contact estimation from imbalanced data,” in Advances in Neural Information Processing Systems, 2026.

[24] Y. Lee, et al., “Adversarial skill chaining for long-horizon robot manipulation via terminal state regularization,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, A. Faust, et al., Eds., vol. 164. PMLR, 08–11 Nov 2022, pp. 406–416.

[25] Y. Yang, et al., “LiLo-VLA: Compositional long-horizon manipulation via linked object-centric policies,” arXiv preprint arXiv:2602.21531, 2026.

[26] Z. Gu, et al., “Learning diffusion policy from primitive skills for robot manipulation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 22, 2026, pp. 18 262–18 270.

[27] H.-C. Fang, et al., “Learning skills from action-free videos,” arXiv preprint arXiv:2512.20052, 2025.

[28] J. Zhang, et al., “HaWoR: World-space hand motion reconstruction from egocentric videos,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 1805–1815.

[29] J. Romero, et al., “Embodied hands: modeling and capturing hands and bodies together,” ACM Trans. Graph., vol. 36, no. 6, Nov. 2017.

[30] A. Handa, et al., “DexPilot: Vision-based teleoperation of dexterous robotic hand-arm system,” in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 9164–9170.

[31] R. H. Byrd, et al., “A limited memory algorithm for bound constrained optimization,” SIAM Journal on scientific computing, vol. 16, no. 5, pp. 1190–1208, 1995.

[32] L. Chen, et al., “EasyHeC: Accurate and automatic hand-eye calibration via differentiable rendering and space exploration,” IEEE Robotics and Automation Letters, vol. 8, no. 11, pp. 7234–7241, 2023.

[33] N. Ravi, et al., “Sam 2: Segment anything in images and videos,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 28 085–28 128.

[34] S. Zhou, et al., “ProPainter: Improving propagation and transformer for video inpainting,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 10 477–10 486.

[35] M. Assran, et al., “V-JEPA 2: Self-supervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985, 2025.

[36] C. Chi, et al., “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.

[37] S. Bai, et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.

[38] J. Zhu, et al., “InternVL3: Exploring advanced training and testtime recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.