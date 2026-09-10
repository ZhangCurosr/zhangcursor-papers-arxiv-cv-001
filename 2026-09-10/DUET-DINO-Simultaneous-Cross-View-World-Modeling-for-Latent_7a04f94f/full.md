# DUET-DINO: Simultaneous Cross-View World Modeling for Latent Planning in Robot Manipulation

Nisarga Nilavadi<sup>1</sup>, Ralf Romer¨ <sup>2</sup>, Moritz Reuss<sup>3,4</sup>, Michael Krawez<sup>1</sup>, Tobias Julg¨ <sup>1</sup>, Angela P. Schoellig<sup>2,5</sup>, Rudolf Lioutikov<sup>3,5</sup>, Wolfram Burgard<sup>1,5</sup>

Abstract— Action-conditioned latent world models predict future visual representations, enabling zero-shot goal-conditioned robot planning and control. However, their predictions for finegrained spatial and rotational actions are unreliable for full 7- DoF end-effector control. To address this gap, we introduce DUET-DINO, a simultaneous cross-view latent world model that jointly learns action-conditioned predictions from static side- and wrist-camera observations through cross-view conditioning. By exploiting complementary global scene and grippercentric information, DUET-DINO enables latent planning over the full 7-DoF action space. Across spatially diverse reach, orientation-intensive angled-reach, and multi-goal grasp-andlift tasks, DUET-DINO consistently outperforms single-view and independent dual-view baselines, achieving 92% success on reach, 72.5% on angled-reach, and 60.0% on lift tasks. DUET-DINO is trained from scratch on DROID and RoboArena datasets and generalizes robustly under visual distribution shifts. We further show that while V-JEPA 2 wrist-view predictions underestimate visual dynamics induced by fine-grained actions, DINOv3 predictions better capture action-conditioned scene changes, leading to stronger downstream planning. The code and model checkpoints will be open-sourced. Project page: https://utn-air.github.io/DUET-DINO

## I. INTRODUCTION

World models have emerged as a promising direction to address generalization in robotics by learning representations of the world and its dynamics from large-scale web data [1]– [3]. Through self-supervised training on billions of examples, world models can improve generalization across diverse environments and implicitly capture aspects of real-world dynamics, such as how objects move and interact. In robot manipulation, world models can predict future states as latent representations [2], [4], pixel-level observations [1], [5], [6], or rewards [6], thereby providing information about the consequences of actions. These predictions can be leveraged in various ways, such as for planning [2], [7], policy training [8], [9], or safety verification [10], [11]. Latent world models [2], [4], [12] are particularly attractive for closed-loop planning and control: they produce task-relevant features without modeling photorealistic details, enabling much faster rollouts than pixel-space generators.

Despite these compelling properties, applying latent world models to real-world manipulation remains challenging.

![](images/f75026448b78fd9bf2292d53bd1d442d7058eab6f7fc3a80315e2843adf1ae21.jpg)  
Fig. 1. Comparison of goal-conditioned 7-DoF action planning toward the target object (banana) between our proposed DUET-DINO world model and V-JEPA 2-AC. DUET-DINO progressively moves the end-effector toward the goal, while V-JEPA 2-AC fails to approach the target.

While existing action-conditioned predictors [2], [4] can be conditioned on the full 7-DoF end-effector action space, their rollouts primarily capture coarse translational motion over short horizons. Accordingly, prior deployment setups have focused on translation-only action planning from predicted rollouts for relatively simple pick-and-place downstream tasks. In contrast, full 7-DoF action planning requires reliable prediction of fine-grained translational, rotational, and gripper-state changes, which are critical for precise manipulation tasks such as spatially diverse reaching, angled grasping, orientation-intensive pick-and-place, and insertion [7]. Consequently, tasks that require planning over the full endeffector action space remain out of reach for current static single-view latent world models.

To address this gap, we introduce a tightly modeled dualview approach, Simultaneous Cross-View World Modeling for Latent Planning in Robot Manipulation (DUET-DINO), for planning full 7-DoF end-effector motions. DUET-DINO goes beyond prior latent world models [2], [4] by jointly modeling scene changes from a side-view camera that captures global workspace context and a wristmounted camera that observes gripper-centric geometry. The two streams are conditioned on each other in the latent space via cross-attention, enabling each view predictor to exploit complementary cues from the other while remaining specialized to its own camera. At deployment, DUET-DINO plans over full 7-DoF action sequences by jointly minimizing the dual-view goal cost across both latent streams. On hardware and simulated benchmarks including spatially diverse, orientation-intensive, and multi-goal angled-lift tasks, DUET DINO consistently outperforms static single-view and naive dual-view baselines.

In summary, our main contributions are the following:

1) We propose DUET-DINO, a latent world model that jointly trains side- and wrist-view action-conditioned predictors on cross-view-conditioned observation latents, enabling each view-specific predictor to leverage complementary information for improved downstream planning.

2) We extend latent world-model planning to the full 7-DoF end-effector space by jointly optimizing over predicted side and wrist-view latent rollouts using a normalized dual-view goal cost.

3) We show the effect of pretrained latent representations on action-conditioned predictions, finding that V-JEPA 2 underestimates wrist-view visual changes, while DINOv3 better captures fine-grained motion.

4) We introduce spatially diverse, orientation-intensive, and multi-goal manipulation tasks in the RoboLab simulator [13] that expose the 7-DoF prediction bottlenecks of latent world models.

5) We conduct extensive evaluations across simulation and hardware on spatial reach, orientation-intensive angled reach, and sequential angled-grasp and lift tasks, including diverse backgrounds, distractors, and perturbations. DUET-DINO achieves 92% success on reach, 72.5% on angled-reach, and 60% on lift tasks, consistently outperforming single-view and independent dual-view baselines.

## II. RELATED WORK

## A. World Models in Robotics

Generative video world models. Foundation-scale video generators such as Cosmos [1] and Wan [14] use diffusionand flow-based models to synthesize photorealistic video, and a growing number of works adapt this capability to robot manipulation. DreamGen [15] synthesizes neural trajectories for policy training, recovering latent pseudo-actions to facilitate generalization. WorldGym [16] uses an autoregressive video generator as a proxy environment for scoring policies via Monte Carlo rollouts, while Ctrl-World [5] combines pose-conditioned memory with multi-view video diffusion to rank policies and produce imagined rollouts for supervised fine-tuning. DiWA [8] pushes this further to fully offline reinforcement learning for adapting diffusion policies inside a learned world model. While generative world models excel at visual fidelity, reconstructing every pixel is computationally expensive and prone to hallucinations of physically implausible scenes [17].

Latent world models. A complementary line of work avoids pixel reconstruction by predicting representations in a learned latent space. Joint-Embedding Predictive Architectures (JEPAs), as introduced for images by I-JEPA [18] and extended to video by V-JEPA [19], leverage masked feature-prediction objectives to produce strong, transferable representations without any reconstruction or hand-crafted augmentation. V-JEPA 2 [2] scales this recipe further and adds an action-conditioned predictor (V-JEPA 2-AC) using under 62 hours of robot interaction. V-JEPA 2.1 [4] introduces a dense predictive loss and deep self-supervision to better capture fine-grained features. DINO-WM [12] demonstrates that patch features from a frozen DINOv2 encoder can support action-conditioned latent prediction and zero-shot planning, and DINOv3 [20] pushes dense self-supervised features to a larger scale. A key limitation of existing latent world models is that they largely rely on a single external camera and struggle with fine-grained 7-DoF control, an open challenge that we explicitly target in this work.

## B. Visual Planning with World Models

Planning in pixel vs. latent space. One family of methods plans directly in pixel space by generating future videos and extracting actions from them. UniPi [21] casts decision making as text-conditioned video generation followed by an inverse-dynamics model; SuSIE [22] uses a pretrained image-editing diffusion model to propose subgoal images for a low-level goal-conditioned policy; and CLOVER [23] closes the loop by combining a text-conditioned video diffusion model with a measurable error space and a feedbackdriven controller. These methods directly couple planning quality to the fidelity of generated pixels. An alternative is to plan in the latent space of a latent world model: V-JEPA 2-AC [2] and DINO-WM [12] formulate manipulation as latent-feature matching to a goal embedding and optimize action sequences directly against this latent cost. Wang et al. [24] show that explicitly regularizing latent trajectories to be locally straight improves the geometry of the latent space and makes Euclidean distances a better proxy for cost.

Action proposals and optimization. A common choice is to sample actions from a Gaussian proposal and iteratively refine them using the cross-entropy method (CEM) [2], [12], [25]–[27], which is derivative-free and well-suited to highly non-convex objectives. Other works learn structured proposal distributions [28] or rely on a vision-language model to propose camera trajectories for spatial reasoning [29]. Optimization can also blend sampling and gradients: [26] interleave CEM with gradient steps to escape local optima, and [27] shows that simple MPPI-style updates already enable dexterous in-hand manipulation when paired with sufficiently accurate dynamics. Recent works plan from a single external camera and primarily address translation-dominated tasks. Rotational and gripper actions remain challenging because their effects are degraded in latent predictions [2], [7]. In addition, goal cost is poorly informed by a single global view. DUET-DINO addresses this bottleneck by conditioning the global side- and gripper-centric wrist observation views on each other in latent space, and performing CEM planning with a goal cost that aggregates both views, enabling reliable planning over the full 7-DoF action space.

## III. METHODOLOGY

## A. Problem Formulation

We approach tabletop manipulation using synchronized camera streams from a static side camera and a dynamic wrist-mounted camera. Given an observation $o _ { t } = ( o _ { t } ^ { \mathrm { s i d e } } , o _ { t } ^ { \mathrm { w r i s t } } )$ consisting of two camera images ${ o _ { t } ^ { \mathrm { s i d e } } , o _ { t } ^ { \mathrm { w r i s t } } \in \mathbb { R } ^ { H \times W \times C } }$ , as well as the full end-effector action $a _ { t } \in \mathbb { R } ^ { 7 }$ at timestep $t ,$ we aim to predict future latent representations $\hat { z } _ { t + T } ^ { \mathrm { s i d e } } , \hat { z } _ { t + T } ^ { \mathrm { w r i s t } }$ induced by the executed actions over horizon T. The predicted latents should capture visual features from both camera views, enabling zero-shot goalconditioned planning for manipulation.

![](images/0692dcbc77029806382ecab2ab1a1798f108f5c2615aed0b8bdff7f57eb13185.jpg)  
Fig. 2. Architecture and training of DUET-DINO. The pretrained encoder maps side- and wrist-view observations into latent space. Cross-attention blocks condition each view on complementary information from the other view in the latent space. The view-specific predictor heads then predict future latents conditioned on the action and end-effector state. The dual-view prediction loss $\mathcal { L } _ { \mathrm { p r e d } } ( \bar { \phi } , \psi )$ combines the teacher-forcing loss computed over all one-step predictions in the clip with an autoregressive loss over the K-step rollout predictions.

## B. DUET-DINO: Simultaneous Cross-View World Modeling

The proposed world model consists of three components. First, a pretrained visual encoder maps side- and wrist-view images to latent representations. Next, cross-attention blocks allow each view-specific latent to query the complementaryview latents. Finally, using the updated latent representations, separate predictor heads predict future latents for their respective views, conditioned on the same robot action. We keep the visual encoder frozen and train the cross-attention blocks and predictor heads jointly from scratch on large-scale robotic manipulation datasets. An overview of DUET-DINO is shown in Fig. 2.

The side- and wrist-view observations $o _ { t } ^ { \mathrm { s i d e } }$ and $o _ { t } ^ { \mathrm { w r i s t } }$ are independently encoded using the frozen DINOv3 encoder $E _ { \theta }$ which was pretrained on large-scale visual data, yielding patch-wise latent representations

$$
z _ { t } ^ { v } = E _ { \theta } ( o _ { t } ^ { v } ) \in \mathbb { R } ^ { P \times D } v \in \{ \mathrm { s i d e } , \mathrm { w r i s t } \} .\tag{1}
$$

where P is the number of image patches, and D is the encoder’s latent dimension per patch. We then feed these latents through learnable cross-attention blocks, $\mathcal { C } _ { \phi _ { \mathrm { s i d e } } }$ and $\mathcal { C } _ { \phi _ { \mathrm { w r i s t } } }$ , where the latents of each view query the latents of the other view. Within each cross-attention block, the targetview latents serve as queries, while the other-view latents serve as keys and values

$$
\begin{array} { r } { \tilde { z } _ { t } ^ { \mathrm { s i d e } } = \mathcal { C } _ { \phi _ { \mathrm { s i d e } } } \left( Q = z _ { t } ^ { \mathrm { s i d e } } , K = z _ { t } ^ { \mathrm { w r i s t } } , V = z _ { t } ^ { \mathrm { w r i s t } } \right) , } \\ { \tilde { z } _ { t } ^ { \mathrm { w r i s t } } = \mathcal { C } _ { \phi _ { \mathrm { w r i s t } } } \left( Q = z _ { t } ^ { \mathrm { w r i s t } } , K = z _ { t } ^ { \mathrm { s i d e } } , V = z _ { t } ^ { \mathrm { s i d e } } \right) . } \end{array}\tag{2}
$$

The resulting cross-view-conditioned latents, $\tilde { z } _ { t } ^ { \mathrm { s i d e } }$ and $\tilde { z } _ { t } ^ { \mathrm { w r i s t } }$ are passed to their respective state- and action-conditioned predictor heads, $P _ { \psi _ { \mathrm { s i d e } } }$ and $P _ { \psi _ { \mathrm { w r i s t } } }$ , for which we adopt the architecture introduced by [2]. The predictor heads remain view-specific, with their inputs having already been updated through cross-view latent conditioning. This allows each predictor to preserve its own view-specific dynamics while using complementary information from the other view to predict the next latent representation as

$$
\hat { z } _ { t + 1 } ^ { v } = P _ { \psi _ { v } } \left( \tilde { z } _ { t } ^ { v } , a _ { t } , s _ { t } \right) \qquad v \in \{ \mathrm { s i d e } , \mathrm { w r i s t } \} .\tag{3}
$$

Training: We train the cross-attention blocks and predictor heads of DUET-DINO from scratch on a largescale dataset $\mathcal { D } = \{ ( o _ { t : t + T } ^ { i } , a _ { t : t + T - 1 } ^ { i } ) \} _ { i = 1 } ^ { N }$ containing clips of length T of observation-action pairs. For both camera views $v \in \{ \mathrm { s i d e , w r i s t } \}$ , we combine two loss functions for training: a teacher-forcing (one-step prediction) loss $\mathcal { L } _ { \mathrm { T F } } ^ { v }$ and an autoregressive rollout loss $\mathcal { L } _ { \mathrm { A R } } ^ { v }$ . We compute the latter over a rollout horizon of $K = 2$ , starting from the first frame of the clip [2]. The overall dual-view prediction loss for joint training of the cross-attention blocks and the view-specific predictor heads is given by

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p r e d } } ( \phi , \psi ) = \mathbb { E } _ { ( o _ { t : t + T } , a _ { t : t + T - 1 } ) \sim \mathcal { D } } \sum _ { v \in \{ \mathrm { s i d e } , \mathrm { w r i s t } \} } } \\ & { \Big [ \underbrace { \frac { 1 } { T - 1 } \sum _ { t = 2 } ^ { T } { \lVert \hat { z } _ { t } ^ { v } - z _ { t } ^ { v } \rVert } _ { \mathcal { L } _ { \mathrm { T F } } ^ { v } } } _ { \mathcal { L } _ { \mathrm { T F } } ^ { v } } + \underbrace { \frac { 1 } { K } \sum _ { k = 2 } ^ { K + 1 } { \lVert \hat { z } _ { k , \mathrm { A R } } ^ { v } - z _ { k } ^ { v } \rVert } _ { \mathcal { L } _ { \mathrm { A R } } ^ { v } } } _ { \mathcal { L } _ { \mathrm { A R } } ^ { v } } \Big ] . } \end{array}\tag{4}
$$

## C. Visual Planning

Once trained, we leverage DUET-DINO for zero-shot planning of full end-effector motions. Our objective is to generate a sequence of actions driving the robot to a goal observation $o _ { \mathrm { { g } } } = ( o _ { \mathrm { { g } } } ^ { \mathrm { { s i d e } } } , o _ { \mathrm { { g } } } ^ { \mathrm { { w r i s t } } } )$ . We formulate this problem in the dual learned latent space of DUET-DINO, with the goal embeddings given by $z _ { \mathrm { g } } ^ { \mathrm { s i d e } } ~ = ~ E _ { \theta } ( o _ { \mathrm { g } } ^ { \mathrm { s i d e } } ) , ~ z _ { \mathrm { g } } ^ { \mathrm { w r i s t } } ~ =$ $E _ { \theta } { \left( o _ { \mathrm { g } } ^ { \mathrm { w r i s t } } \right) }$ . Denoting $z _ { t } = ( z _ { t } ^ { \mathrm { s i d e } } , z _ { t } ^ { \mathrm { w r i s t } } )$ and considering a fixed receding horizon of length H, we aim to iteratively solve the optimal control problem

![](images/6813f91009c21612ebfc6903b858e40ec3335ce62957ba0666e44dff4ac43d5a.jpg)  
Fig. 3. Overview of the proposed simultaneous cross-view world modeling method, showing predicted dual-view latent rollouts, CEM action planning using predicted and goal latents, and the normalized dual-view planning cost $J ( z _ { \mathrm { g } } , \hat { z } _ { T } )$

$$
\begin{array} { r l } { \underset { a _ { 0 : H - 1 } } { \operatorname* { m i n } } } & { J ( z _ { \mathbf { g } } , \hat { z } _ { H } ) } \\ { \mathrm { s . t . } } & { \hat { z } _ { \tau + 1 } = \mathbf { D U E T - D I N O } ( \hat { z } _ { \tau } , a _ { \tau } ) , \tau = 0 , \dots , H - 1 , } \\ & { \hat { z } _ { 0 } = z _ { t } , } \end{array}\tag{5}
$$

where $J ( z _ { \mathrm { g } } , \hat { z } _ { H } )$ measures the distance between the predicted and goal latents. We solve the optimal control problem (5) using CEM [2], [7]. At each CEM iteration, we sample N action sequences $a _ { t : t + H - 1 } ^ { 1 : N } \ \sim \ N ( \mu , \Sigma )$ from a Gaussian proposal distribution with mean µ and variance Σ and roll out the actions inside DUET-DINO to obtain imagined latent trajectories $\tau ^ { 1 : N } = ( s _ { t } , z _ { t } , a _ { t } , \hat { z } _ { t + 1 } ^ { 1 : N } , a _ { t + 1 } ^ { 1 : N } , \dots , \hat { z } _ { t + H } ^ { \overline { { 1 } } : N } )$ Then, we can assess the quality of each action sequence by computing the corresponding deviations from the goal latents as

$$
\ell _ { i } ^ { \mathrm { { s i d e } } } = \Vert \hat { z } _ { t + H } ^ { \mathrm { { s i d e } , i } } - z _ { \mathrm { { g } } } ^ { \mathrm { { s i d e } } } \Vert _ { 1 } , \ell _ { i } ^ { \mathrm { { w r i s t } } } = \Vert \hat { z } _ { t + H } ^ { \mathrm { { w r i s t } , i } } - z _ { \mathrm { { g } } } ^ { \mathrm { { w r i s t } } } \Vert _ { 1 } .\tag{6}
$$

The overall dual-view planning cost is obtained by summing the two per-view costs and normalizing by the dual-view cost of a zero-action rollout over the same prediction horizon, i.e.,

$$
J _ { i } ( z _ { \mathrm { g } } , \hat { z } _ { t + H } ) = \frac { \ell _ { i } ^ { \mathrm { s i d e } } + \ell _ { i } ^ { \mathrm { w r i s t } } } { ( \ell ^ { \mathrm { s i d e } } + \ell ^ { \mathrm { w r i s t } } ) _ { 0 } } \qquad \mathcal { I } = \{ J _ { i } \} _ { i = 1 } ^ { N } .\tag{7}
$$

We adopt a weighted top-k CEM update, where the selected candidates are weighted using a softmax over their negative costs, assigning larger weights to lower-cost action sequences when updating the proposal distribution.

## IV. EXPERIMENTS

This section describes the dataset processing, world-model training procedure, experimental setup, evaluation protocol, and results for DUET-DINO and the baseline methods.

## A. Implementation and Training Details

We train and evaluate our world-model predictors on two large-scale robot manipulation datasets: DROID [30] and RoboArena [31]. After filtering the data, we obtain 62,877 DROID and 5,856 RoboArena diverse tabletop manipulation trajectories. DROID comprises 86 diverse manipulation tasks across 564 scenes and 52 buildings at 13 institutions, covering a broad range of everyday objects, tasks, and environments. Unlike prior work that trains only on DROID [2], [4], we include data from RoboArena, collected by rolling out different policies, to expose the model to a broader distribution of behaviors, scenes, and failure modes than DROID alone. Each batch samples 8-frame video clips with synchronized side- and wrist-view observations at 4 FPS from 1280×720 images and resizes them to 256×256 for V-JEPA 2 encoders and 224×224 for DINOv3 encoders. At each time step, the predictor is conditioned on a 7D end-effector action $\mathbf { a } _ { t } \ = \ \mathbf { s } _ { t + 1 } - \mathbf { s } _ { t } \ = \ [ \Delta x , \Delta y , \Delta z , \Delta r , \Delta p , \Delta y , \Delta g ]$ where $\Delta x , \Delta y , \Delta z$ denote Cartesian translation, $\Delta r , \Delta p , \Delta y$ denote changes in end-effector roll, pitch, and yaw, and $\Delta g$ is the change in gripper state.

We train baseline world-model predictors that vary in visual encoder, camera view, and architecture. Specifically, we consider independently trained single-view predictors for the side and wrist cameras, alongside our proposed DUET dual-view predictor. Each predictor architecture is evaluated with two frozen pretrained visual encoder families: V-JEPA 2 [2], using a ViT-G/16 backbone with 1B parameters, and DINOv3 [20], using a ViT-H+/16 backbone with 840M parameters. We optimize all world-model variants using AdamW for 120k steps with a batch size of 256 with a learning rate of $4 . 2 5 \times 1 0 ^ { - 4 }$ and a weight decay of 0.04. The learning rate is warmed up from $7 . 5 \times 1 0 ^ { - 5 }$ over the first 4.5k steps and annealed to zero over the final 30k steps. The total computational cost of DUET-DINO over 120k training steps across four GPUs is $5 \times 1 0 ^ { 2 0 }$ FLOPs. The combined training cost of all baseline and ablation models, excluding DUET-DINO, is $2 . 8 \times 1 0 ^ { 2 1 }$ FLOPs.

## B. Experimental Setup

We evaluate latent-world-model planning on manipulation tasks that require full 7-DoF end-effector control. These tasks include reach, angled reach, angled grasp and angled lift-tohome. We organize the baseline and proposed world models into three predictor categories:

1) Single-view: These models predict latent representations for either the side view or the wrist view using a single view-specific predictor.

2) Independent dual-view: This baseline combines separately trained side- and wrist-view predictors, without cross-attention or joint optimization between the two.

3) DUET dual-view: Our proposed model jointly trains two view-specific predictor heads together with crossview attention blocks.

In addition, we evaluate all tasks using the official V-JEPA 2- $\mathbf { A } \mathbf { C } ^ { * }$ checkpoint [2], which was trained exclusively on the left side-view DROID data for 94.5k optimization steps. Fig. 3 shows an overview of the prediction and planning pipeline. Goal observations from the corresponding camera views specify each task. For single-view predictors, the goal is provided from either the static side camera or the wrist camera, whereas dual-view predictors use paired side- and wrist-view goal observations. Single-view predictors plan using the corresponding normalized view-specific cost, while dual-view predictors use the combined planning cost, as described in Sec. III-C. Unlike prior work [2], [4], which uses 3-DoF end-effector actions with gripper control, our planner samples candidate action sequences over the full 7- DoF end-effector action space. For all tasks in simulation, planning consistently uses 800 action samples, weighted 50 top-k candidates, and 2 CEM iterations. For each evaluation run, we use a run-specific CEM seed shared across all model variants, while different runs use different seeds.

The proposed tasks are evaluated in the RoboLab simulator [13], which provides visually realistic and diverse backgrounds, object assets, configurable camera views, and the DROID robot setup, making it well-suited for rigorous evaluation of latent planning for manipulation. The DROID setup consists of a Franka arm equipped with a Robotiq 2F-85 gripper, a wrist-mounted camera, and a static side-view camera. Note that the right side-view camera is used consistently for both training and evaluation. The end-effector actions produced by the planner are transformed into joint commands using inverse kinematics.

## C. Reach Task

The experimental setup for the reach task consists of ten semantically distinct target objects distributed across the breakfast table scene with a HomeOfficeBackground, as shown in Fig. 4. We define a reach run to be successful if the end-effector position $p _ { \mathrm { e } }$ is within a distance of $\epsilon _ { p }$ to the goal position $p _ { \mathrm { g } } ,$ in 100 steps, i.e., i $\mathrm { ~ f ~ } \| p _ { \mathrm { e } } - p _ { \mathrm { g } } \| _ { 2 } \leq \dot { \epsilon } _ { p }$ . We define $\epsilon _ { p } = 5 $ cm for all items. The final position error (FPE) is the Euclidean distance computed at the final planning step, regardless of task success, and therefore includes residual error after successful task completion.

![](images/dd6a369a73c98bc42c58d8ba0f7d7a48f02475ab146e0da467e5c7eed73364d1.jpg)  
Fig. 4. Static right side-camera view showing the robot in its home pose and the target objects used in the reach tasks: (1) Coffee Pot, (2) Coffee Can, (3) Banana, (4) Juice Carton, (5) Yogurt Cup, (6) Apple, (7) Orange, (8) Bagel, (9) White Pitcher, and (10) Ceramic Mug.

TABLE I  
AGGREGATE REACH SUCCESS RATE (SR) AND FINAL POSITION ERROR (FPE) OVER 100 EVALUATION RUNS PER MODEL ACROSS TEN TASKS.
<table><tr><td>Side</td><td>Wrist</td><td>Predictor</td><td>SR (%) ↑</td><td>FPE (cm) ↓</td></tr><tr><td colspan="5">V-JEPA 2 Encoder</td></tr><tr><td> $\checkmark$ </td><td></td><td> $\mathrm { V } { \mathrm { - J } } \mathrm { E P A } \ 2 { \mathrm { - A C } } ^ { \ast }$ </td><td>55%</td><td> $1 8 . 6 { \pm } 2 3 . 2 $ </td></tr><tr><td> $\checkmark$ </td><td></td><td>Single-view</td><td>20%</td><td> $3 4 . 4 { \pm } 1 7 . 9$ </td></tr><tr><td> $^ -$ </td><td> $\checkmark$ </td><td>Single-view</td><td>37%</td><td> $3 2 . 5 { \pm } 2 6 . 6 $ </td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>Independent</td><td>37%</td><td> $3 2 . 5 { \pm } 2 4 . 4 $ </td></tr><tr><td> $\checkmark$ </td><td></td><td>DUET-VJEPA</td><td>41%</td><td> $3 9 . 0 { \pm } 3 8 . 6 $ </td></tr><tr><td colspan="5">DINOv3 Encoder</td></tr><tr><td> $\checkmark$ </td><td> $^ -$ </td><td>Single-view</td><td>18%</td><td> $2 8 . 0 { \pm } 1 5 . 6 $ </td></tr><tr><td> $^ -$ </td><td> $\checkmark$ </td><td>Single-view</td><td>79%</td><td> $1 3 . 1 { \pm } 2 0 . 3 $ </td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>Independent</td><td>78%</td><td> $9 . 8 \pm 1 3 . 8 $ </td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>DUET-DINO</td><td>92%</td><td>5.4±7.4</td></tr></table>

Table I shows the reach performance of single-view and dual-view predictors in latent planning. V-JEPA $2 { \cdot } \mathrm { A C ^ { * } }$ with our visual planning achieves a higher success rate than all trained V-JEPA 2 baselines, suggesting that longer training on more diverse data degraded spatial planning. The trained side-view predictors achieve low success rates and high FPE, suggesting limited ability to spatially plan end-effector actions across different regions of the table. Successful runs were primarily observed for target objects located near the robot’s home pose, such as objects (10) and (5). The V-JEPA 2 wrist-view predictor shows a moderate improvement over its side-view. In contrast, the DINOv3 wrist-view predictor shows a larger performance gain. Independent dualview models provide negligible gains over the corresponding wrist-view predictors, highlighting the difficulty of naively combining views. The DUET-VJEPA predictor improves over its corresponding independent dual-view model in terms of success rate. Despite this improvement, its FPE remains high because, in failed reach (3)Banana runs, the planned end-effector actions drive the robot substantially away from the target object, resulting in a large FPE. Our DUET-DINO predictor achieves the strongest overall spatial planning performance, reaching a success rate of 92%. Notably, DUET-DINO also succeeds on the challenging reach (1)Coffee Pot runs near the table corner, which requires coordinated translational and rotational motion under kinematic constraints, while the target is outside the wrist camera’s field of view at the robot’s home pose. [2] reports 16 s per planning step for a single-view predictor using more CEM iterations, while DUET-DINO requires 15–17 s with two predictors in our experiment setup. Overall, dual-view models are roughly 2× slower than single-view models, and V-JEPA 2 dualview variants are about 1.3× slower than the corresponding DINOv3 models.

TABLE II  
AGGREGATE ANGLED-REACH SUCCESS RATE (SR), FINAL POSITION ERROR (FPE), AND FINAL ANGULAR ERROR (FAE) OVER 40 EVALUATION RUNS PER MODEL ACROSS FOUR TASKS.
<table><tr><td>Side</td><td></td><td>Wrist Predictor</td><td></td><td>SR (%) ↑ FPE (cm) ↓</td><td>FAE (°) ↓</td></tr><tr><td colspan="6">V-JEPA 2 Encoder</td></tr><tr><td></td><td></td><td> $\mathrm { V } { \mathrm { - J } } \mathrm { E P A } \ 2 { \mathrm { - } } \mathrm { A C } ^ { \ast }$ </td><td>2.5%</td><td> $3 . 6 { \pm } 1 . 5 $ </td><td> $7 2 . 6 { \pm } 3 2 . 2$ </td></tr><tr><td> $\checkmark$ </td><td></td><td>Single-view</td><td>25.0%</td><td> $3 . 6 { \pm } 1 . 6 $ </td><td> $5 1 . 9 { \pm } 3 1 . 9$ </td></tr><tr><td> $^ -$ </td><td> $\checkmark$ </td><td>Single-view</td><td>25.0%</td><td> $3 6 . 7 { \pm } 2 2 . 5$ </td><td> $6 7 . 9 { \pm } 3 7 . 7 $ </td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>Independent</td><td>25.0%</td><td> $1 8 . 4 \pm 1 5 . 7$ </td><td> $6 6 . 7 { \pm } 3 6 . 8 $ </td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>DUET-VJEPA</td><td>25.0%</td><td> $1 5 . 7 { \pm } 2 0 . 6 $ </td><td> $6 9 . 7 { \scriptstyle \pm 4 1 . 7 }$ </td></tr><tr><td colspan="6">DINOv3 Encoder</td></tr><tr><td> $\checkmark$ </td><td></td><td>Single-view</td><td>25.0%</td><td> $5 . 3 { \pm } 1 . 5 $ </td><td> $6 7 . 1 { \pm } 4 0 . 2 $ </td></tr><tr><td> $^ -$ </td><td></td><td>Single-view</td><td>62.5%</td><td> $8 . 3 { \pm } 7 . 2 $ </td><td> $4 7 . 4 { \pm } 5 7 . 5$ </td></tr><tr><td> $\checkmark$ </td><td></td><td>Independent</td><td>55.0%</td><td> $3 . 2 { \pm } 2 . 5 $ </td><td> $4 8 . 8 { \pm } 5 4 . 4 $ </td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>DUET-DINO</td><td>72.5%</td><td> ${ \bf 1 . 9 2 1 . 2 }$ </td><td> $\mathbf { 2 2 . 7 \pm 3 2 . 2 }$ </td></tr></table>

## D. Angled Reach Task

To evaluate orientation planning, we define a set of orientation-intensive angled-reach tasks. We define an angled reach run as successful if the translational and rotational errors satisfy $\| p _ { \mathrm { e } } - p _ { \mathrm { g } } \| _ { 2 } \le \epsilon _ { p }$ and $\theta ( q _ { \mathrm { e } } , q _ { \mathrm { g } } ) \leq \epsilon _ { \theta }$ , where $q _ { \mathrm { e } }$ and $q _ { \mathrm { g } }$ are the actual and goal orientation expressed as quaternions, and the orientation error is given by $\theta ( q _ { \mathrm { e } } , q _ { \mathrm { g } } ) =$ $2 \cos ^ { - 1 } \left( | q _ { \mathrm { e } } \cdot q _ { \mathrm { g } } | \right)$ . We set the tolerances to $\epsilon _ { p } = 0 . 0 5$ m and $\epsilon _ { \theta } = 0 . 2 1$ (corresponding to $1 2 ^ { \circ } )$ , which must be satisfied within 100 steps for a run to be counted as successful. The final angular error (FAE) is computed at the final planning step, regardless of whether a run is successful.

Fixed Background: First, we define four orientationintensive angled-reach tasks across four distinct tabletop scenes in a HomeOfficeBackground. Two tasks primarily require vertical reaching along the z-axis, followed by clockwise and counterclockwise yaw rotations, respectively. The remaining two tasks require coordinated translational motion in $( x , \ y , \ z )$ , together with clockwise and counterclockwise yaw rotations. As shown in Table II, V-JEPA $2 { \cdot } \mathrm { A C ^ { * } }$ achieves a lower success rate than the trained V-JEPA 2 side-view baseline on angled-reach tasks, suggesting that while prior work performs well for coarse translational planning, it struggles with fine-grained control. The trained side-view predictors fail on the most rotationintensive angled-reach tasks and succeed only on the simpler rotation cases. Since the V-JEPA 2 wrist-view predictor also performs poorly, the corresponding independent dual-view and DUET-VJEPA predictors show no improvement. The DINOv3 wrist-view predictor shows substantially stronger performance, suggesting that DINOv3 latent features capture orientation changes effectively and that the corresponding action-conditioned predictor learns these dynamics well. Although the DINOv3 independent dual-view model improves FPE by leveraging the side-view predictor, its FAE increases when both views are combined, resulting in a lower success rate. DUET-DINO reduces both final position and angular errors, resulting in a higher success rate.

TABLE III  
AGGREGATE ANGLED-REACH SUCCESS RATE (SR), FINAL POSITION ERROR (FPE), AND FINAL ANGULAR ERROR (FAE) UNDER VISUAL DISTRIBUTION SHIFTS OVER 100 EVALUATION RUNS PER MODEL.
<table><tr><td>Side</td><td></td><td>Wrist Predictor</td><td>SR (%) ↑</td><td>FPE (cm) ↓</td><td>FAE (°) ↓</td></tr><tr><td colspan="6">DINOv3 Encoder</td></tr><tr><td></td><td></td><td>Single-view</td><td>25%</td><td> $1 5 . 0 { \pm } 1 0 . 0 \ $ </td><td> $5 4 . 3 { \pm } 4 2 . 4$ </td></tr><tr><td>√</td><td>√</td><td>Independent</td><td>45%</td><td> $3 . 3 { \pm } 2 . 4 $ </td><td> $4 7 . 3 { \pm } 4 0 . 8 $ </td></tr><tr><td> $\checkmark$ </td><td></td><td>DUET-DINO</td><td>63%</td><td> ${ \bf 2 . 6 \pm 2 . 1 }$ </td><td> ${ \bf 3 0 . 9 \pm 3 6 . 5 }$ </td></tr></table>

TABLE IV

AGGREGATE ANGLED-REACH SUCCESS RATES (SR), FINAL POSITION ERRORS (FPE), AND FINAL ANGULAR ERRORS (FAE) OVER 30 HARDWARE EVALUATION RUNS ACROSS THREE TASKS.
<table><tr><td></td><td></td><td>Side Wrist Predictor</td><td></td><td>SR (%) ↑ FPE (cm) ↓</td><td> $\mathbf { F A E } \left( { } ^ { \circ } \right) \downarrow$ </td></tr><tr><td colspan="6">DINOv3 Encoder</td></tr><tr><td></td><td> $\checkmark$ </td><td>Single-view</td><td>3.3%</td><td> $2 4 . 9 { \pm } 2 1 . 0 \ $ </td><td> $3 9 . 9 { \pm } 3 1 . 5 $ </td></tr><tr><td>√</td><td></td><td>Independent</td><td>16.7%</td><td> $\mathbf { 1 4 . 3 \pm 1 7 . 2 }$ </td><td> $4 0 . 0 { \pm } 4 2 . 0 $ </td></tr><tr><td></td><td> $\checkmark$ </td><td>DUET-DINO</td><td>26.7%</td><td> $1 4 . 7 { \pm } 1 7 . 1 $ </td><td> $\mathbf { 2 8 . 0 { \pm } 1 7 . 7 }$ </td></tr></table>

Diverse Backgrounds and Distractor Objects: Next, we rigorously evaluate the best-performing models on the same orientation-intensive tasks under visual distribution shifts across five diverse backgrounds and tabletop objects. For each of five backgrounds, we vary the tabletop scene from one target object to five tabletop objects, including visually similar distractors. We evaluate these variations across four angled-reach tasks, resulting in $5 \times 5 \times 4 = 1 0 0$ evaluation runs per model. Table III shows that DUET-DINO remains robust under these visual variations, while the wrist-view predictor exhibits a noticeable drop in success rate. We observe that the overall angled-reach planning performance is particularly strong for asymmetrically shaped target objects.

Hardware Runs: Finally, we evaluate the best-performing DINOv3-based predictors on three angled-reach tasks using a real DROID robot setup. We selected challenging target poses near the table corners with both clockwise and counterclockwise target orientations. Our DROID hardware setup consists of a table-mounted Franka Research 3 robot equipped with a Robotiq 2F-85 gripper, a ZED 2i camera for the static side view, and a ZED Mini camera for the wrist view. We use the Robot Control Stack [32] for realworld robot control and VLAgents [33] to serve the policy. Results in Table IV show that DUET-DINO achieves the highest success rate on the challenging angled-reach poses, outperforming the independent dual-view predictor. Due to the computational cost and latency of CEM planning on hardware, we use one CEM iteration with 500 candidate actions and 15 planning steps. This reduced planning budget contributes to the lower hardware success rates relative to the simulation results. For safety, actions resulting in collisions or kinematic singularities were rejected, and runs were terminated if unsafe actions were proposed for three consecutive planning steps. The wrist-view predictor produced unsafe actions more frequently, further contributing to its lower success rate on hardware.

TABLE V  
AGGREGATE STAGE-WISE SUCCESS RATES (SR) FOR ANGLED-LIFT TASKS OVER 40 EVALUATION RUNS ACROSS FOUR TASKS.
<table><tr><td>Side</td><td>Wrist</td><td>Predictor</td><td>Angled-Reach SR (%) ↑</td><td>Lift SR (%) ↑</td></tr><tr><td colspan="5">DINOv3 Encoder</td></tr><tr><td></td><td></td><td>Single-view</td><td>57.5%</td><td>20.0%</td></tr><tr><td>√</td><td>√</td><td>Independent</td><td>60.0%</td><td>57.5%</td></tr><tr><td></td><td></td><td>DUET-DINO</td><td>75.0%</td><td>60.0%</td></tr></table>

## E. Angled Grasp and Angled Lift-to-Home

We further extend the angled-reach tasks in the HomeOfficeBackground with grasping and lifting subtasks, thereby evaluating the models on sequential multigoal planning. We allocate 60 planning steps for angled reaching (successful if the pose error is below 15 cm and 20<sup>◦</sup>), 10 steps for grasping, and 60 steps for lifting the grasped object to the initial robot home pose (successful if the end-effector position is within 20 cm of the home position while maintaining the grasp). Most of the DUET-DINO runs succeed in all sub-tasks, thereby completing the full lift task. The DINOv3 wrist-view predictor often succeeds at angled reaching and grasping but loses task context during the liftto-home phase, resulting in lift failure.

## F. Ablation

Cross-view Cross Attention: We train an ablated DUET-DINO variant without the cross-attention blocks. This reduces the reach performance of DUET-DINO in Table I to 81% SR with 10.7±15.4 cm FPE, and the angled-reach performance in Table II to 42.5% SR with 4.3±3.1 cm FPE and 55.5±52.8<sup>◦</sup> FAE. This confirms the importance of crossview attention for fine-grained dual-view planning.

Camera Perturbations: The side-view predictor learns action-conditioned visual dynamics from a fixed camera viewpoint, i.e., robot joint motion during reaching and object motion in contact-rich interactions. We therefore evaluate robustness to camera-pose shifts on ReachBananaTask over 30 runs, perturbing the camera by ±20 cm in $x / y ,$ ±10 cm in $z ,$ and $\pm 1 1 . 5 ^ { \circ }$ in roll, pitch, and yaw, using matched perturbation seeds across models’ runs. Under these shifts, DUET-DINO and the DINOv3 independent dualview predictor achieve 66.7% SR, compared with 46.7% for DUET-VJEPA and 40.0% for the V-JEPA 2 independent dual-view predictor, while both single-view side predictors fail with 0% SR. These results highlight the sensitivity of side-view planning to camera-pose shifts and the robustness gained from incorporating the wrist view.

![](images/3bbf13dc21808a351ecc4f1dc2b390acbb7f425a881065d5fb8f617426602137.jpg)

Fig. 5. Mean latent changes captured by frozen encoders, DINOv3 actionconditioned predictors, and V-JEPA 2 action-conditioned predictors over 30 runs of an angled-reach task.  
![](images/a3465fed78dfadf481237b9afa4163579049d589481efe6561d0675ff2749dd9.jpg)  
Fig. 6. Patch-wise latent distances for DINOv3 and V-JEPA 2 feature maps. Lower $\ell _ { 1 }$ distances indicate greater similarity to the selected reference patch. Frames at t and t + 1 are shown after resizing to the respective encoder inputs. Feature maps are visualized at 14 × 14 for DINOv3 and 16 × 16 for V-JEPA 2. DINOv3 produces 196 patches of 1280 dimensions, whereas V-JEPA 2 produces 256 patches of 1408 dimensions.

Training Data: To study the effect of training data, we train DUET-DINO on DROID only for 94.5k steps following the prior work [2]. The DROID-only DUET-DINO achieves 78% SR on the reach tasks in Table I, showing the benefit of additional training data on spatial planning. On the angled-reach tasks in Table II, the DROID-only DUET-DINO achieves 72.5% SR, showing no improvement in orientation planning.

## G. Embeddings Analysis

To understand why DINOv3-based predictors outperform those based on V-JEPA 2, we analyze the underlying latent representations. Using an angled-reach rollout, we encode all frames with the frozen DINOv3 and V-JEPA 2 encoders and compute the $\ell _ { 1 }$ norm between consecutive feature maps. This measures the sensitivity of each representation space to visual changes over action steps. As shown in Fig. 5, first column, V-JEPA 2 and DINOv3 encoder features exhibit latent changes of similar magnitude, with slightly larger side-view changes for V-JEPA 2. We then measure the scene dynamics produced by the predictors by computing the $\ell _ { 1 }$ distance between the encoded input features and the action-conditioned predicted next-step features. In the second column, the DINOv3 side- and wrist-view predictions follow the ground-truth temporal pattern, while generally overestimating their magnitude due to prediction error. In contrast, the last column shows that the V-JEPA 2 wrist-view predictor underestimates latent changes, while the side-view predictor better captures the magnitude, suggesting weaker modeling of fine-grained action-conditioned dynamics by V-JEPA 2 wrist predictor.

In Fig. 6, we analyze wrist-view patch correspondences by selecting a target-object latent patch at time t and computing its $\ell _ { 1 }$ distance to all patches at t, t+1, and the predicted t+1 feature maps. Lower distances indicate higher feature similarity. DINOv3 embeddings show coherent correspondences over semantically related object regions, and its AC-predictor closely preserves the encoded t + 1 pattern. In contrast, V-JEPA 2 embedding exhibits noisier correspondences, with predicted low-distance regions also appearing in semantically unrelated areas.

## V. CONCLUSION

We present DUET-DINO, a cross-view latent world model for action-conditioned prediction and planning in robot manipulation. By combining cross-view latent conditioning with DINO representations, DUET-DINO provides an effective approach for modeling scene changes induced by translational, rotational, and gripper actions. The evaluations are computationally expensive as CEM evaluates many candidate actions through world models at each step. While this enables rigorous world-model evaluation under diverse action proposals, it limits real-time control. Our future work will leverage DUET-DINO on a broader range of real-world tasks and use action proposals from generalist robot policies, such as vision-language-action (VLA) models, to reduce the planning search space and enable faster closed-loop control.

## VI. ACKNOWLEDGEMENTS

We thank Volker Schneider for support with the hardware setup and 3D-printing, and Prasanna Bhat for helping with real-world evaluations. This work has been partially supported by the German Federal Ministry of Research, Technology and Space (BMFTR) under the Robotics Institute Germany (RIG). Ralf Romer gratefully acknowledges sup-¨ port from the German Research Foundation (DFG) within the RTG project ConVeY, funded by grant GRK 2428. The authors acknowledge the HPC resources provided by the Erlangen National HPC Center (NHR@FAU) under the BayernKI project no. v106be.

## REFERENCES

[1] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding et al., “Cosmos world foundation model platform for physical AI,” https://arxiv.org/abs/2501.03575, 2025.

[2] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. Robert Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, and N. Ballas, “V-jepa 2: Self-supervised video models enable understanding, prediction and planning,” https://arxiv.org/abs/2506.09985, 2025.

[3] B. Hou, G. Li, J. Jia, T. An, X. Guo, S. Leng, H. Geng, Y. Ze, T. Harada, P. Torr et al., “World model for robot learning: A comprehensive survey,” https://arxiv.org/abs/2605.00080, 2026.

[4] L. Mur-Labadia, M. Muckley, A. Bar, M. Assran, K. Sinha, M. Rabbat, Y. LeCun, N. Ballas, and A. Bardes, “V-JEPA 2.1: Unlocking dense features in video self-supervised learning,” https://arxiv.org/abs/2603. 14482, 2026.

[5] Y. Guo, L. X. Shi, J. Chen, and C. Finn, “Ctrl-World: A controllable generative world model for robot manipulation,” in Proc. of the Int. Conf. on Learning Representations (ICLR), 2026.

[6] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M.-Y. Liu, C. Finn et al., “Cosmos Policy: Fine-tuning video models for visuomotor control and planning,” https://arxiv.org/abs/ 2601.16163, 2026.

[7] B. Terver, T.-Y. Yang, J. Ponce, A. Bardes, and Y. LeCun, “What drives success in physical planning with joint-embedding predictive world models?” https://arxiv.org/abs/2512.24497, 2025.

[8] A. L. Chandra, I. Nematollahi, C. Huang, T. Welschehold, W. Burgard, and A. Valada, “DiWA: Diffusion policy adaptation with world models,” in Proc. of the Conf. on Robot Learning (CoRL), 2025.

[9] Z. Jiang, K. Liu, Y. Qin, S. Tian, Y. Zheng, M. Zhou, C. Yu, H. Li, and D. Zhao, “World4RL: Diffusion world models for policy refinement with reinforcement learning for robotic manipulation,” https://arxiv. org/abs/2509.19080, 2025.

[10] Y. Li, Y. Zhu, J. Wen, C. Shen, and Y. Xu, “WorldEval: World model as real-world robot policies evaluator,” https://arxiv.org/abs/2505.19017, 2025.

[11] Gemini Robotics Team, K. Choromanski, C. Devin, Y. Du, D. Dwibedi, R. Gao, A. Jindal, T. Kipf, S. Kirmani, I. Leal et al., “Evaluating Gemini Robotics policies in a Veo world simulator,” https://arxiv.org/abs/2512.10675, 2025.

[12] G. Zhou, H. Pan, Y. LeCun, and L. Pinto, “DINO-WM: World models on pre-trained visual features enable zero-shot planning,” in Proc. of the Int. Conf. on Machine Learning (ICML), 2025.

[13] X. Yang, R. Dagli, A. Zook, H. Hadfield, A. Goyal, S. Birchfield, F. Ramos, and J. Tremblay, “RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies,” in Proc. of Robotics: Science and Systems (RSS), 2026.

[14] Wan Team, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced large-scale video generative models,” https://arxiv.org/abs/2503.20314, 2025.

[15] J. Jang, S. Ye, Z. Lin, J. Xiang, J. Bjorck, Y. Fang, F. Hu, S. Huang, K. Kundalia, Y.-C. Lin et al., “DreamGen: Unlocking generalization in robot learning through video world models,” in Proc. of the Conf. on Robot Learning (CoRL), 2025.

[16] J. Quevedo, A. K. Sharma, Y. Sun, V. Suryavanshi, P. Liang, and S. Yang, “WorldGym: World model as an environment for policy evaluation,” in Proc. of the Int. Conf. on Learning Representations (ICLR), 2026.

[17] Z. Mei, T. Yin, M. Baker, O. Shorinwa, and A. Majumdar, “World models that know when they don’t know: Controllable video generation with calibrated uncertainty,” https://arxiv.org/abs/2512.05927, 2025.

[18] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in Proc. of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[19] A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas, “Revisiting feature prediction for learning visual representations from video,” Transactions on Machine Learning Research, 2024.

[20] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “DINOv3,” https://arxiv.org/abs/2508.10104, 2025.

[21] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel, “Learning universal policies via text-guided video generation,” in Advances in Neural Information Processing Systems, 2023.

[22] K. Black, M. Nakamoto, P. Atreya, H. R. Walke, C. Finn, A. Kumar, and S. Levine, “Zero-shot robotic manipulation with pre-trained image-editing diffusion models,” in Proc. of the Int. Conf. on Learning Representations (ICLR), 2024.

[23] Q. Bu, J. Zeng, L. Chen, Y. Yang, G. Zhou, J. Yan, P. Luo, H. Cui, Y. Ma, and H. Li, “Closed-loop visuomotor control with generative expectation for robotic manipulation,” in Advances in Neural Information Processing Systems, 2024.

[24] Y. Wang, O. Bounou, G. Zhou, R. Balestriero, T. G. Rudner, Y. LeCun, and M. Ren, “Temporal straightening for latent planning,” https://arxiv. org/abs/2603.12231, 2026.

[25] R. Y. Rubinstein, “Optimization of computer simulation models with rare events,” European Journal of Operational Research, 1997.

[26] H. Bharadhwaj, K. Xie, and F. Shkurti, “Model-predictive control via cross-entropy and gradient-based optimization,” in Proc. of the Learning for Dynamics & Control Conference (L4DC), 2020.

[27] A. Nagabandi, K. Konolige, S. Levine, and V. Kumar, “Deep dynamics models for learning dexterous manipulation,” in Proc. of the Conf. on Robot Learning (CoRL), 2020.

[28] C. Gao, H. Zhang, Z. Xu, C. Zhehao, and L. Shao, “FLIP: Flow-centric generative planning as general-purpose manipulation world model,” in Proc. of the Int. Conf. on Learning Representations (ICLR), 2025.

[29] Y. Yang, J. Liu, Z. Zhang, S. Zhou, R. Tan, J. Yang, Y. Du, and C. Gan, “MindJourney: Test-time scaling with world models for spatial reasoning,” in Advances in Neural Information Processing Systems, 2025.

[30] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis et al., “DROID: A large-scale in-the-wild robot manipulation dataset,” in Proc. of Robotics: Science and Systems (RSS), 2024.

[31] P. Atreya, K. Pertsch, T. Lee, M. J. Kim, A. Jain, A. Kuramshin, C. Eppner, C. Neary, E. Hu, F. Ramos et al., “RoboArena: Distributed real-world evaluation of generalist robot policies,” in Proc. of the Conf. on Robot Learning (CoRL), 2025.

[32] T. Julg, P. Krack, S. Bien, Y. Blei, K. Gamal, K. Nakahara, J. Hechtl,¨ R. Calandra, W. Burgard, and F. Walter, “Robot Control Stack: A lean ecosystem for robot learning at scale,” in Proc. of the IEEE Int. Conf. on Robotics & Automation (ICRA), 2026.

[33] T. Julg, K. Gamal, N. Nilavadi, P. Krack, S. Bien, M. Krawez,¨ F. Walter, and W. Burgard, “VLAgents: A policy server for efficient vla inference,” https://arxiv.org/abs/2601.11250, 2026.