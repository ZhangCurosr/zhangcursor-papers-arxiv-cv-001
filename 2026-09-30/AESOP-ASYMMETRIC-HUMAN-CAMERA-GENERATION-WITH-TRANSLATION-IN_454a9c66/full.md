# AESOP: ASYMMETRIC HUMAN–CAMERA GENERATION WITH TRANSLATION-INTENSITY CONTROL

Jingzhong Lin<sup>1,\*</sup>, Zhanke Wang<sup>2</sup>, Heng Li<sup>3</sup>, Wenxiang Liu<sup>1</sup> Zhao Zhang<sup>1</sup>, Kecheng Tang<sup>1</sup>, Dongdong Xiang<sup>1</sup>, Changbo Wang<sup>1</sup> Di Kang<sup>4</sup>, Chunchao Guo<sup>4</sup>, Linchao Bao<sup>4</sup>, Gaoqi He<sup>1,†</sup>

<sup>1</sup>East China Normal University

<sup>2</sup>Peking University

<sup>4</sup>Tencent

<sup>3</sup>Sun Yat-sen University

## ABSTRACT

Human motion defines an action, while a camera trajectory determines how it is presented. Camera generation for a given human motion and joint human–camera generation are usually treated as separate tasks, although both share an asymmetric dependency: human motion can be generated independently, whereas the camera responds to the realized action. We introduce AESOP, a unified framework with an independent human pathway and a shared human-conditioned camera generator. Its asymmetric architecture serves both tasks while preserving the human output during camera generation. Although human context anchors the shot to the action and camera text describes its movement, translation intensity remains underspecified. We therefore construct trajectory pairs that differ in camera translation magnitude while sharing human motion and camera text, then use these pairs to learn an explicit intensity condition. Experiments on the PulpMotion dataset demonstrate strong camera distributional and framing quality in both tasks and effective control over camera translation intensity.

## 1 INTRODUCTION

Text-conditioned human motion generation can now produce diverse, natural actions that follow language descriptions (Guo et al., 2022; Tevet et al., 2023; Zhang et al., 2024; Chen et al., 2023; Zhang et al., 2023; Guo et al., 2024). Authoring an animation also requires choosing how the action is viewed. Cinematography treats framing and camera movement as constructive choices that organize screen space, direct attention, and shape how viewers experience an event (Bordwell et al., 2024; Brown, 2022). Human motion and camera trajectories therefore play complementary but not interchangeable roles: human motion defines the action, whereas camera trajectory presents that action for the viewer. A human-aware camera is therefore part of motion storytelling.

Prior work approaches this relationship through two broad formulations. Conditional camera methods adapt a trajectory or shot plan to an available human motion (Courant et al., 2024; Kizil et al., 2026), leaving human synthesis outside the task. Coupled systems synthesize human motion and camera trajectories within one generative process (Courant et al., 2026; Cheng et al., 2026), treating their generation as interdependent. These formulations leave an opportunity to unify both tasks around the distinct roles of human action and camera response.

We introduce AESOP (Asymmetric Encoding and Synthesis of Observer Paths). Two observations guide our design. First, human action can be determined independently of the camera, while both tasks can share a camera generator conditioned on supplied or generated human motion. Second, human context and camera text specify an action and a shot description but leave translation intensity underspecified. Together, these observations motivate a shared asymmetric architecture and explicit geometric supervision for camera control.

![](images/492d0da5729523f012e42b3c5f8c170b442355506f4a19c27571a5cb672b5760.jpg)  
Figure 1: AESOP supports (a) camera generation for a given human motion, (b) joint human–camera generation, and (c) continuous camera translation-intensity control (the amount of camera travel) in both tasks without changing the supplied or generated human motion.

For the architecture, we propose an independent human representation and generator together with a shared human-conditioned camera generator through a common latent interface. Camera training freezes the human pathway; joint inference completes human generation before sampling the camera trajectory. This one-way dependency serves both tasks while preserving the supplied or generated human action during camera generation.

To learn explicit translation intensity, we construct intensity-paired augmentation (IPA): weaker and stronger camera trajectories which share human motion and camera text. IPA contracts or expands smooth displacement about the camera center at the first frame, and geometry and visibility checks screen the targets. Auxiliary direction-paired augmentation (DPA) supplies oppositedirection Truck and Dolly examples. After original-pair camera training, DPA and then IPA are introduced through continuation while retaining earlier data.

On the PulpMotion dataset, AESOP achieves strong camera distributional and framing quality in both tasks. It reduces given-human r-FPD by 49.9% relative to DIRECTOR-C and joint camera FDC by 87.8% relative to PulpMotion DiT. Its learned intensity condition controls camera travel.

Our contributions are threefold. (1) We introduce a unified asymmetric framework with an independent human pathway and a shared human-conditioned camera generator for given-human and joint generation. (2) We learn explicit camera translation-intensity control from geometry-grounded intensity pairs, with directional pairs providing auxiliary supervision. (3) We evaluate both tasks, demonstrating substantially improved camera distributional and framing quality, strong human distribution quality, and effective camera travel intensity control.

## 2 RELATED WORK

## 2.1 HUMAN MOTION GENERATION

Text-to-motion research has expanded from synthesizing plausible action clips to supporting more detailed and flexible authoring. Paired motion–language data such as HumanML3D (Guo et al., 2022) and MotionMillion (Fan et al., 2025) made semantic fidelity and motion quality central evaluation targets. Diffusion-based models (Tevet et al., 2023; Zhang et al., 2024; Chen et al., 2023) and discrete generative models (Zhang et al., 2023; Guo et al., 2024) offer complementary approaches, with masked autoregressive diffusion extending continuous motion modeling (Meng et al., 2025). As clip synthesis improved, the focus broadened to instructions and controls beyond a single action label: finer supervision addresses spatial and temporal detail (Wu et al., 2025), streaming generation accommodates incoming instructions (Zhao et al., 2025; Xiao et al., 2025), and larger-scale training expands action coverage and controllability (Wen et al., 2025; Rempe et al., 2026). Video-generation priors also support motion synthesis (Lin et al., 2026).

## 2.2 HUMAN-AWARE CAMERA GENERATION

Camera synthesis accounts for the subject it depicts. For an available performance, the problem is to generate a compatible viewpoint and trajectory: DIRECTOR (Courant et al., 2024) introduces character-aware text-to-camera generation, while DanceCamera3D (Wang et al., 2024b) studies camera movement conditioned on dance and music. This line of work establishes human motion as useful camera context. When the performance itself is also generated, fixing human motion as input is insufficient. PulpMotion (Courant et al., 2026) incorporates screen-space framing into human–camera generation, and Towards Storytelling Animations (Cheng et al., 2026) studies joint synthesis of their temporal behavior. AESOP uses a shared complete-sequence human interface to support both supplied and generated performances.

## 2.3 EXPLICIT AND GEOMETRY-GROUNDED CAMERA CONTROL

Geometric camera interfaces support composition constraints (Lino & Christie, 2015); learned keyframing systems combine camera styles with keyframe and velocity control (Jiang et al., 2021). GenDoP generates camera trajectories from text, optionally conditioned on initial-frame RGB-D observations (Zhang et al., 2025). Actor-relative framing in Auteur (Kizil et al., 2026) and visual preference optimization in VERTIGO (Li et al., 2026) improve how directing intent is expressed or followed. Even with human context, however, one instruction can admit many trajectories, leaving the strength and direction of movement ambiguous. Explicit trajectory interfaces address this ambiguity by supplying the geometry itself: MotionCtrl (Wang et al., 2024a) and CameraCtrl (He et al., 2025) condition video generation on camera paths. This shifts trajectory authoring to the user. AESOP instead supplies training examples that vary direction or translation intensity while holding human motion fixed. Paired directional instructions and a continuous intensity scalar expose these geometric choices without requiring a target path.

## 3 METHOD

## 3.1 PRELIMINARIES

Problem definition. Let $H = \left( h _ { 0 } , \ldots , h _ { T - 1 } \right)$ denote human motion and $C = ( c _ { 0 } , \dots , c _ { T - 1 } )$ a synchronized camera trajectory, with text instructions $T _ { H }$ and $T _ { C }$ . Following PulpMotion (Courant et al., 2026), human motion data encode root motion and heading, local joint positions, and joint rotations; camera trajectory data encode human-relative position, camera rotation, field of view, and translation velocity. Given-human camera generation maps $( H , T _ { C } )$ to C. Joint human–camera generation maps $( T _ { H } , T _ { C } )$ to $( H , C )$ . Both tasks accept an optional translation-intensity condition $^ { a , }$ with $a = 1$ specifying default behavior and smaller or larger values requesting weaker or stronger camera-center translation.

Flow matching for human and camera generation. For flow time $\sigma \in [ 0 , 1 ]$ , clean latent z and Gaussian noise ϵ, define $z _ { \sigma } = ( 1 - \sigma ) z + \sigma \epsilon$ and target velocity $\epsilon - z$ (Lipman et al., 2023). The human and camera flows minimize

$$
\begin{array} { r l } & { \mathcal { L } _ { H } ^ { \mathrm { F M } } = \mathbb { E } \Big [ \| v _ { \phi } ( z _ { H , \sigma } , \sigma , T _ { H } ) - ( \epsilon _ { H } - z _ { H } ) \| _ { 2 } ^ { 2 } \Big ] , } \\ & { \mathcal { L } _ { C } ^ { \mathrm { F M } } = \mathbb { E } \Big [ \| v _ { \theta } ( z _ { C , \sigma } , \sigma , z _ { H } , T _ { C } , a ) - ( \epsilon _ { C } - z _ { C } ) \| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{1}
$$

Here $z _ { H }$ and $z _ { C }$ are normalized human and camera latents, and the camera flow receives the complete, fixed human latent sequence. Sampling integrates the predicted velocity from noise at $\sigma = 1$ to a clean latent at $\sigma = 0$

## 3.2 UNIFIED ASYMMETRIC HUMAN–CAMERA GENERATION

AESOP learns its asymmetric representation in Stage 1 and the human and camera generators in Stage 2. Geometry-grounded pairs provide translation-intensity and auxiliary direction supervision.

![](images/f743dac962071913506d86ceb0301f849b6cbafba7c48cb5e9e2fca77b0be00f.jpg)  
Figure 2: AESOP overview. (A) Separate encoders produce 128-channel human and 64-channel camera latents; the camera decoder reads their concatenation. (B,C) Human generation supplies the complete latent sequence to both human decoding and camera conditioning. Given-human generation instead encodes the supplied motion. (D,E) Human and camera Transformer blocks.

As illustrated in Fig. 2(B,C), AESOP factorizes generation as

$$
p ( H , C \mid T _ { H } , T _ { C } , a ) = p _ { \phi } ( H \mid T _ { H } ) p _ { \theta } ( C \mid H , T _ { C } , a ) .\tag{2}
$$

Both tasks use the same conditional camera model $p _ { \theta } ( C ~ \vert ~ H , T _ { C } , a )$ . Given-human generation encodes the supplied motion, whereas joint generation first samples the complete human latent from $p _ { \phi } ( H \mid T _ { H } )$ and reuses it for camera conditioning and human decoding.

Asymmetric human–camera representation. Stage 1 (Fig. 2(A)) first trains a human encoder– decoder and then freezes it while training the camera encoder–decoder:

$$
\begin{array} { l } { { z _ { H } = E _ { H } ( H ) , \quad \hat { H } = D _ { H } ( z _ { H } ) , } } \\ { { z _ { C } = E _ { C } ( C ) , \quad \hat { C } = D _ { C } ( [ \mathrm { s g } ( z _ { H } ) ; z _ { C } ] ) . } } \end{array}\tag{3}
$$

Here $[ \cdot ; \cdot ]$ denotes channel concatenation and sg stops gradients. The camera encoder maps camera trajectory data to a latent sequence, and the camera decoder combines this sequence with the frozen human latents. Human reconstruction depends only on $z _ { H }$ . Supplementary Section B.1 illustrates the difference from PulpMotion’s symmetric representation. Stage 1 uses reconstruction and temporal-difference losses, with additional human root and yaw terms; the full objectives appear in Supplementary Section B.2.

Independent human generation. In Stage 2, we train the human flow (Fig. 2(B,D)) with the objective in Eq. 1, then freeze its parameters before camera training. At inference, the human flow samples a complete latent sequence from human text and Gaussian noise, and the frozen human decoder reconstructs the motion. For fixed human text and noise, camera guidance, intensity and camera noise leave the generated human motion unchanged.

Human-aware camera generation. During training, the camera flow (Fig. 2(C,E)) minimizes $\mathcal { L } _ { C } ^ { \mathrm { F M } }$ over θ, conditioned on the complete 128-channel encoded ground-truth human sequence while predicting velocity in the 64-channel camera latent space. The human flow and Stage 1 representation remain frozen. Training has three phases within Stage 2: original-pair training, DPA continuation, and IPA continuation. Both continuation phases mix original and augmented pairs at predefined sampling ratios, with DPA pairs retained during IPA training. Table S3 gives the schedule.

At inference, given-human generation uses the encoded supplied motion, while joint generation uses the completed human-flow latent directly (Fig. 2(B,C)). The same camera flow and decoder serve both tasks. Camera training drops text with probability 0.1 while retaining human context and intensity, enabling classifier-free guidance (Ho & Salimans, 2021) between the text-conditioned velocity $v _ { \theta } ^ { c }$ and text-dropped velocity $v _ { \theta } ^ { u }$

$$
v _ { \theta } ^ { g } = v _ { \theta } ^ { u } + g \big ( v _ { \theta } ^ { c } - v _ { \theta } ^ { u } \big ) .\tag{4}
$$

The text-conditioned and text-dropped velocity predictions use the same human latent, intensity $^ { a , }$ noisy camera latent, and flow time. Here $g$ is the camera-text guidance weight.

## 3.3 GEOMETRY-GROUNDED CAMERA CONTROL

Ordinary paired data rarely show how the same action should be filmed when only camera direction or translation intensity changes. DPA constructs opposite-direction targets, while IPA constructs targets at different translation intensities in physical camera space (Fig. 3).

human: A person stands still and crosses their arms. camera: The camera shifts <direction> along its local horizontal axis.

![](images/aaa767b80ea5eae16c7ab71493054af79c0f8f08608b417c579c47537edc7312.jpg)  
(a) Original: left  
Gray camera: shared initial state C<sub>0</sub>.

human: A person stands still and looks around. camera: The camera follows a counterclockwise arc around the person while keeping them in view.

![](images/452025c8043e87e8f4a81c1e792705f478462dffed91147254b276583c6f15f7.jpg)  
(b) Reversed: right

![](images/63b5f85b00ae3c71e351b8fa459898c8710215d46c2e680037c389cbc317102a.jpg)  
(c) Reduced intensity

![](images/a0648654665f81b483a51808689bc6f6d361ea1d2dc9fba6c754fa207838de74.jpg)  
(d) Increased intensity  
Orange: a < 1; teal: a = 1. Orange: a > 1; teal: a = 1.  
Figure 3: Training target construction of DPA and IPA. (a,b) Opposite directions share human motion and the initial camera state. (c,d) Intensity variants retain human motion and camera text.

Data construction. We mine high-confidence atomic Truck and Dolly events for oppositedirection pairs, and translation-active trajectories for weaker/stronger intensity variants. Both con structions fix human motion. Geometry, visibility, framing, dynamics and representation checks screen the targets; Supplementary Section C details the construction and acceptance criteria for both DPA and IPA.

Frames are indexed by $t = 0 , \ldots , T - 1$ . A direction event occupies $[ t _ { s } , t _ { e } ] \subseteq [ 0 , T - 1 ]$ which may cover a subinterval of the sequence. For direction pairing, let $q ^ { + } ( t )$ be the original signed event coordinate over $[ t _ { s } , t _ { e } ] $ lateral camera-center position for trucking, or camera–human radial distance for dolly translation. The opposite target $q ^ { - } ( t )$ reflects its displacement about the event start. For intensity control, we smooth the original world-space camera-center path $p _ { t }$ to obtain its low-frequency component s<sub>t</sub>; $\boldsymbol { r } _ { t } = \boldsymbol { p } _ { t } - \boldsymbol { s } _ { t }$ contains the remaining local fluctuations. The two transformations are

$$
\begin{array} { r l } & { q ^ { - } ( t ) = 2 q ^ { + } ( t _ { s } ) - q ^ { + } ( t ) , \qquad t \in [ t _ { s } , t _ { e } ] , } \\ & { ~ p _ { t } ^ { ( a ) } = p _ { 0 } + a ( s _ { t } - s _ { 0 } ) + \operatorname* { m i n } ( a , 1 ) ( r _ { t } - r _ { 0 } ) . } \end{array}\tag{5}
$$

Direction targets (Fig. 3a–b) use paired minimal text templates that differ only in direction and preserve event timing, the displacement magnitude of $q ,$ initial camera state, rotation, and field of view; an endpoint offset maintains continuity after the event. Constructed intensity targets (Fig. $^ { 3 \mathrm { c } , \mathrm { d } ) }$ preserve camera text, rotation, field of view, and sequence timing. They recover $p _ { t }$ exactly at $a = 1$ and amplify only the smooth component when $a > 1$ , avoiding amplification of residual jitter.

We also select low-translation sources and pair each unchanged camera target with both $a = 1$ and a non-default intensity. These unchanged-target pairs enter the final training phase alongside active variants, teaching the model to retain low-translation shots under intensity changes. We embed intensity with a two-layer MLP, $e _ { a } = W _ { 2 } \mathrm { S i L U } ( W _ { 1 } [ a - 1 , ( a - 1 ) ^ { 2 } ] ^ { \top } )$ , and add $e _ { a }$ to the camera flow’s timestep condition (Fig. 2(C)). The learned projections $W _ { 1 } , W _ { 2 }$ are bias-free and $W _ { 2 }$ is zeroinitialized, giving $e _ { 1 } = 0$ throughout training and an initially inactive intensity pathway.

Paired supervision. Both members of each control pair contribute to the flow-matching objective in Eq. 1, sharing human context, Gaussian noise, flow time and camera-text dropout decisions. DPA changes the directional prompt with the target; IPA retains the prompt and changes the intensity label. During target construction, candidate intensities are sampled in $0 < a < 1$ and $1 < a < 2 ,$ with values nearer $a = 1$ proposed more often. Each retained translation-active source supplies one accepted weaker target and one accepted stronger target.

## 4 EXPERIMENTS

We evaluate camera quality in given-human and joint human–camera generation, human generation quality, and geometry-grounded camera control.

## 4.1 EXPERIMENTAL SETUP

Data and implementation. We use the public PulpMotion split (Courant et al., 2026), with 162,760 training sequences and 4,053 test sequences, evaluating both tasks at valid sequence lengths. Geometry-grounded training uses 601 direction sources, 8,000 translation-active sources and 2,000 low-translation sources from the training split. Unless stated otherwise, AESOP uses 50-step Euler sampling per stream, human guidance 1, camera guidance $g = 1 . 5$ and translation intensity $a = 1$ Both tasks share the camera model. Supplementary Sections B.1 and B.2 give architecture and optimization details.

Metrics. Table 1 reports camera distribution distance (FDC), camera-text alignment (CLaTr), movement-caption F1 (Courant et al., 2024), and PRDC recall in the frozen CLaTr embedding space (Naeem et al., 2020). Framing is measured by r-FPD, the distribution distance of projected human joints in normalized image coordinates, and Out (%), the valid-frame rate at which none of nine key joints is in front of the camera and inside the image (Courant et al., 2026). Camera-center ADE measures trajectory error (m) for reconstruction and given-human generation; rotation error (<sup>◦</sup>) compares orientations with the reference camera, including in joint generation. Human metrics use a common frozen evaluator (Petrovich et al., 2023): distribution distance $\mathrm { F D } _ { \mathrm { T M R } }$ , text alignment TMR and PRDC. CLaTr and TMR report mean nonnegative text cosine similarity scaled by 100. Table S2 adds reconstruction diagnostics.

Repeated-run results use mean<sup>±SD</sup>, where SD denotes sample standard deviation. Generation and reconstruction comparisons aggregate three independent training seeds; direction-pair results aggregate three paired camera-noise sets from one training seed. Control sweeps fix one checkpoint and share sampling noise across settings.

## 4.2 COMPARISON WITH PRIOR METHODS

Baselines. Given-human comparisons include DIRECTOR-C (Courant et al., 2024), DanceCamera3D (Wang et al., 2024b), CCD (Jiang et al., 2024), and our CCD-H adaptation. DIRECTOR-C conditions on camera text and the human root-translation trajectory; DanceCamera3D replaces music with camera text. CCD and CCD-H share a human-relative camera representation. CCD uses a text-only network; CCD-H additionally reads human joints and replaces the text prefix with text cross-attention. PulpMotion (Courant et al., 2026) uses DiT (Peebles & Xie, 2023) and MAR (Li et al., 2024) backbones for joint human–camera generation. Table 1 uses original-data endpoints for all baselines. Supplementary Section D.1 details adaptations and training exposure.

Quantitative comparison. AESOP leads both tasks in FDC, CLaTr and caption F1 (Table 1). Given-human FDC drops from DIRECTOR-C’s 7.29 to 5.97; joint FDC drops from PulpMotion

Table 1: Quantitative comparison. AE rows report reconstruction; the remaining model rows report generation. AESOP (no aug.) uses original pairs, (+DPA) adds direction pairs, and full AESOP further adds intensity pairs. Human scores are shared across AESOP camera-training phases. Bold marks the best displayed value within each reconstruction/task group, excluding GT; blue marks AESOP.
<table><tr><td colspan="5">camera distribution &amp; alignment</td><td colspan="4">camera framing &amp; geometry</td><td colspan="2">human distribution &amp; alignment</td></tr><tr><td>Method</td><td>FDC↓</td><td>CLaTr↑</td><td></td><td>F1↑ Recallc↑</td><td>r-FPD↓</td><td>Out↓</td><td>ADE↓</td><td>Rot.↓</td><td>FDTMR↓</td><td>TMR↑</td></tr><tr><td>GT reference</td><td>≈ 0</td><td>70.237</td><td>0.945</td><td>1.00</td><td>0.003</td><td>0.71</td><td>0.000</td><td>0.002</td><td>≈ 0</td><td>18.398</td></tr><tr><td>PulpMotion AE (sym.)</td><td>16.59±3.52</td><td>59.91±1.38</td><td>80.739±0.032</td><td>0.973±0.005</td><td>0.133±0.02</td><td>3.93±0.55</td><td>0.110±0.01</td><td> $1 . 3 7 ^ { \pm 0 . 1 0 }$ </td><td>38.79±9.99</td><td>920.15±0.28</td></tr><tr><td>AESOP AE (asym.)</td><td>0.11±0.10 70.10±0.13 0.941±0.001</td><td></td><td></td><td> 0.999±0.0004</td><td>40.089±0.01</td><td>2.93±0.28 0.029±0.0006</td><td></td><td>0.41±0.01</td><td>10.05±0.89</td><td>9 17.56±0.12</td></tr><tr><td>Given-human camera generation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DanceCamera3D</td><td></td><td>180.08±5.2420.92±0.54 0.252±0.002</td><td></td><td></td><td></td><td>0.600±0.012.692±0.4821.72±2.33</td><td>2.995±0.07 63.52±1.19</td><td></td><td></td><td></td></tr><tr><td>CCD</td><td></td><td>359.70±6.02 9.53±0.19 0.118±0.004</td><td></td><td>0.632±0.021.937±0.19</td><td></td><td>0.63±0.10</td><td>2.891±0.0475.58±2.09</td><td></td><td></td><td></td></tr><tr><td>CCD-H</td><td></td><td>375.97±39.4311.32±1.580.134±0.020</td><td></td><td>0.706±0.06 2.942±2.35</td><td></td><td>0.51±0.04</td><td>2.712±0.1468.49±2.60</td><td></td><td></td><td></td></tr><tr><td>DIRECTOR-C</td><td></td><td>7.29±0.4264.33±0.280.848±0.001</td><td></td><td>0.897±0.0031.993±0.0410.26±0.22</td><td></td><td></td><td> $2 . 6 8 7 ^ { \pm 0 . 0 7 } 5 8 . 1 6 ^ { \pm 0 . 3 6 }$ </td><td></td><td></td><td></td></tr><tr><td>AESOP (no aug.)</td><td></td><td>6.99±0.2069.60±0.330.905±0.006</td><td></td><td>0.866±0.011.160±0.0712.56±0.43</td><td></td><td></td><td> $1 . 5 7 7 ^ { \pm 0 . 0 1 } 3 3 . 0 0 ^ { \pm 0 . 4 6 }$ </td><td></td><td></td><td></td></tr><tr><td>AESOP (+ DPA)</td><td>6.03±0.1770.65±0.280.915±0.004</td><td></td><td></td><td>0.860±0.0041.001±0.0311.51±0.21</td><td></td><td></td><td> $1 . 5 1 5 ^ { \pm 0 . 0 1 } 3 2 . 1 0 ^ { \pm 0 . 3 6 }$ </td><td></td><td></td><td></td></tr><tr><td>AESOP</td><td>5.97±0.16 70.72±0.32 0.916±0.005</td><td></td><td></td><td>0.863±0.010.998±0.04 11.47±0.22</td><td></td><td></td><td> $\mathbf { 1 . 5 1 2 ^ { \pm 0 . 0 1 } 3 2 . 0 6 ^ { \pm 0 . 4 1 } }$ </td><td></td><td></td><td></td></tr></table>

<table><tr><td>PulpMotion DiT</td><td>83.80±3.3044.92±2.360.592±0.027</td><td>0.711±0.027.772±0.73 35.39±1.28</td><td></td><td>66.76±1.82366.21±21.5224.20±0.97</td></tr><tr><td>PulpMotion MAR</td><td>131.55±9.1438.64±2.420.514±0.033</td><td>0.769±0.038.169±0.6539.08±2.14</td><td></td><td> $7 0 . 5 3 ^ { \pm 1 . 4 5 }$  319.98±7.3821.57±0.87</td></tr><tr><td>AESOP (no aug.)</td><td>11.39±0.67 68.94±0.34 0.880±0.002</td><td>0.826±0.01 0.665±0.03  $9 . 1 3 ^ { \pm 0 . 2 4 }$ </td><td></td><td> $7 1 . 5 6 ^ { \pm 0 . 5 3 }$  100.98±1.9120.46±0.34</td></tr><tr><td>AESOP (+ DPA)</td><td>10.38±0.4769.76±0.330.887±0.005</td><td>0.824±0.010.576±0.02  $\mathbf { 8 . 4 5 ^ { \pm 0 . 0 8 } }$ </td><td></td><td> $7 1 . 8 2 ^ { \pm 0 . 8 8 }$  100.98±1.9120.46±0.34</td></tr><tr><td>AESOP</td><td>10.18±0.50 69.81±0.34 0.891±0.006</td><td> $0 . 8 2 0 ^ { \pm 0 . 0 1 } 0 . 5 8 1 ^ { \pm 0 . 0 2 }$   $8 . 4 7 ^ { \pm 0 . 0 4 }$ </td><td></td><td> $7 1 . 8 7 ^ { \pm 0 . 8 5 }$  100.98±1.91 20.46±0.34</td></tr></table>

## (a) Given-human camera generation

![](images/3f54b0e622cf932960093a39d0df5f893b563aa908cd953400dcacfa0f1fa777.jpg)  
human: A person walks forward, turns his head to the left, and continues walking. camera: The camera performs a pull-out motion throughout the entire shot.

(b) Joint human–camera generation  
![](images/764a5b5f2b63a831a0cd8e8cd31d8bd66931b365d06be13063baa0aa18145e97.jpg)  
human: A person runs forward, extends their arms outward, and then jumps into an elevator.  
camera: The camera executes a push-in motion throughout the entire shot.

Figure 4: Qualitative comparison on test examples. (a) Given-human camera generation with a pull-out prompt. (b) Joint human–camera generation with a push-in prompt. Each method has a global view and three selected camera projections, ordered from top to bottom in time. Global views show human–camera geometry at individually fitted scales; projections show framing over time.

Qualitative comparison. Figure 4 shows AESOP following the pull-out prompt while retaining the subject, whereas DIRECTOR-C and DanceCamera3D crop the subject and CCD misses the requested movement. In the joint example, AESOP progresses from full-body to upper-body framing; DiT alternates between cropped and empty views, while MAR largely misses the human. Supplementary Section H provides more comparisons; Section G shows external examples from HumanML3D and HY-Motion.

(a) Given-human camera generation  
![](images/80185d3a2bf8964e6a1d8d89da68a78f636cfd285a184b9f4c7207fee1e39e0a.jpg)  
human: A person stands still and gestures with both hands, moving them up and down. camera: The camera moves right with a trucking motion during the entire shot.

(b) Joint human–camera generation  
![](images/2a75b9f48fbe7a2a39da7a9006c7287614f3d3943f8fbfecb080f473c02bd483.jpg)  
human: A person sits on a stool, moves their left hand towards their face, and then places it back on the instrument camera: The camera gradually moves closer to the subject throughout the shot.  
Figure 5: Translation-intensity examples with shared human context, text and camera noise. Outputs are regenerated at each intensity; numbers report first-to-last camera-center displacement. Fading snapshots indicate temporal order. Global views are fitted separately to each trajectory's spatial extent.

DiT’s 83.80 to 10.18, while CLaTr rises from 44.92 to 69.81. Without augmentation, AESOP attains FDC 6.99/11.39 for given-human/joint generation, below DIRECTOR-C and PulpMotion DiT, respectively. DPA further improves camera quality; IPA adds intensity control while largely preserving generation quality. For framing, AESOP reduces r-FPD from DIRECTOR-C’s 1.993 to 0.998 in given-human generation and from DiT’s 7.772 to 0.581 in joint generation. Human $\mathrm { F D } _ { \mathrm { T M R } }$ is 100.98 versus DiT/MAR’s 366.21/319.98, with lower TMR alignment. CCD-H achieves the lowest given-human Out.

Table 2: Human distributional fidelity and coverage in joint generation in the common frozen human embedding space. Bold marks the best value in each column.
<table><tr><td>Method</td><td>Precision ↑</td><td>Recall ↑</td><td>Density ↑</td><td>Coverage ↑</td></tr><tr><td>PulpMotion DiT</td><td>0.809±0.02</td><td>0.215±0.02</td><td>0.748±0.05</td><td>0.465±0.02</td></tr><tr><td>PulpMotion MAR</td><td>0.802±0.005</td><td>0.312±0.01</td><td>0.705±0.02</td><td>0.504±0.002</td></tr><tr><td>AESOP</td><td>0.879±0.002</td><td>0.835±0.01</td><td>0.933±0.02</td><td>0.785±0.01</td></tr></table>

Table 2 evaluates human distributional fidelity and coverage. AESOP improves all four PRDC measures, with recall of 0.835 versus 0.215/0.312 for DiT/MAR. Higher precision accompanies this broader coverage, complementing the lower human distribution distance in Table 1.

## 4.3 REPRESENTATION AND CAMERA CONTROL

Representation quality. The AE rows in Table 1 measure reconstruction by encoding and decoding reference sequences. AESOP reduces camera-center ADE to 0.029 m and rotation error to 0.41<sup>◦</sup>, compared with 0.110 m and 1.37<sup>◦</sup> for PulpMotion. It also yields lower human and camera distribution distances, whereas PulpMotion has higher human-text alignment (20.15 versus 17.56). Supplementary Table S2 reports additional human positional and camera-motion errors.

Translation intensity. Figures 5 and 6 show qualitative examples and an eleven-value scan from a = 0.5 to 1.5 (step 0.1). Path length sums camera-center travel; net displacement measures endpoint distance. Mean path length rises from 0.647 to 1.023 m (given-human) and from 0.367 to

0.612 m (joint). We require travel to be nondecreasing in all ten adjacent intervals: 81.50%/81.37% of given-human/joint clips satisfy this for path length, and 78.61%/78.46% for net displacement. For the 2,106 clips with reference net displacement ≥ 0.2 m, path-length rates reach 99.19%/98.86%. Supplementary Section E reports endpoint results.

CLaTr and F1 increase alongside camera travel, while higher r-FPD and Out indicate a framing trade-off. Given-human ADE and rotation error in both tasks vary little. Given-human FDC is lowest near a = 0.8, whereas joint FDC decreases across the grid.

![](images/fe85cb76ea1fe1f458b650f8febc3a594c8943cf610af8df08704bdbcefa6b55.jpg)  
Figure 6: Translation-intensity response and camera quality. Columns group travel distance, distribution, text alignment, framing and reference error. Results use one training seed and shared human context, text and camera noise across intensities. ADE is reported for given-human generation. Vertical dashed lines mark a = 1.

Geometry-grounded training. We evaluate direction accuracy on original and direction-reversed prompts, comparing translation signs over reference-active axes and time steps. After DPA continuation, AESOP reaches 84.33%/83.11% combined accuracy in given-human/joint generation, exceeding the matched factual-pair control by 1.67/1.38 percentage points (Supplementary Table S6). Accuracy improves in both prompt arms: on reversed prompts, DPA scores 79.00%/79.58%, compared with 77.71%/78.51% for factual continuation. IPA then adds continuous intensity control, with similar default camera quality to the +DPA endpoint in Table 1.

## 4.4 USER STUDY

Table 3: Overall-quality preference for AE-SOP (%). Scores average clip-level responses, with ties receiving half credit. Intervals are participant-cluster 95% CIs.
<table><tr><td>Compared with</td><td>Pref.</td><td>95% CI</td></tr><tr><td>Given-human</td><td></td><td></td></tr><tr><td>CCD</td><td>75.3</td><td>[64.8, 85.5]</td></tr><tr><td>DanceCamera3D</td><td>99.2</td><td>[97.5, 100.0]</td></tr><tr><td>DIRECTOR-C</td><td>86.6</td><td>[81.5, 91.5]</td></tr><tr><td>Joint</td><td></td><td></td></tr><tr><td>PulpMotion DiT</td><td>86.3</td><td>[80.1, 91.9]</td></tr><tr><td>PulpMotion MAR</td><td>92.3</td><td>[86.9, 96.9]</td></tr></table>

We conducted an anonymized pairedcomparison study with 29 participants. Each participant completed 20 given-human and 18 joint comparisons, with balanced method pairs and randomized presentation. Participants rate overall camera quality for given-human generation and overall human–camera quality for joint generation. Preference, averaged over clips and equally over opponents, is 87.0% (95% CI: 83.8–90.8) and 89.3% (84.4–93.6), respectively. Table 3 gives the five pairwise comparisons; Supplementary Section F details the protocol and individual criteria.

## 5 CONCLUSION

AESOP unifies given-human camera generation and joint human–camera generation through an independent human pathway and a shared human-conditioned camera generator. Geometry-grounded intensity pairs teach explicit translation-intensity control, complemented by auxiliary direction supervision. Experiments on PulpMotion show strong camera distributional and framing quality in both tasks, while the learned intensity condition adjusts camera travel.

Limitations and future directions. The sequence-level intensity condition limits control over individual events, while complete human context and offline sampling limit streaming generation. Future directions include event-wise intensity control and causal, incremental camera generation.

## REFERENCES

David Bordwell, Kristin Thompson, and Jeff Smith. Film Art: An Introduction. McGraw Hill, 13th edition, 2024.

Blain Brown. Cinematography: Theory and Practice: For Cinematographers and Directors. Routledge, 4th edition, 2022.

Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18000–18010, 2023.

Boyuan Cheng, Yingjie Xi, Rui He, Jinhe Na, Ying Cao, Pengjie Wang, Jian J. Zhang, and Xiaosong Yang. Towards storytelling animations: Joint synthesis of human and camera motions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 38376–38386, 2026.

Robin Courant, Nicolas Dufour, Xi Wang, Marc Christie, and Vicky Kalogeiton. E.T. the exceptional trajectories: Text-to-camera-trajectory generation with character awareness. In European Conference on Computer Vision, pp. 464–480, 2024.

Robin Courant, Xi Wang, David Loiseaux, Marc Christie, and Vicky Kalogeiton. Pulp motion: Framing-aware multimodal camera and human motion generation. In The Fourteenth International Conference on Learning Representations, pp. 98487–98516, 2026.

Ke Fan, Shunlin Lu, Minyue Dai, Runyi Yu, Lixing Xiao, Zhiyang Dou, Junting Dong, Lizhuang Ma, and Jingbo Wang. Go to zero: Towards zero-shot motion generation with million-scale data. In IEEE/CVF International Conference on Computer Vision, pp. 13336–13348, 2025.

Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3D human motions from text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5142–5151, 2022.

Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. MoMask: Generative masked modeling of 3D human motions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1900–1910, 2024.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for video diffusion models. In The Thirteenth International Conference on Learning Representations, pp. 100433–100464, 2025.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In Conference on Neural Information Processing Systems Workshop on Deep Generative Models and Downstream Applications, 2021.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020.

Hongda Jiang, Marc Christie, Xi Wang, Libin Liu, Bin Wang, and Baoquan Chen. Camera keyframing with style and control. ACM Transactions on Graphics, 40(6):1–13, 2021.

Hongda Jiang, Xi Wang, Marc Christie, Libin Liu, and Baoquan Chen. Cinematographic camera diffusion model. Computer Graphics Forum, 43(2):e15055, 2024.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In Advances in Neural Information Processing Systems, volume 35, pp. 26565–26577, 2022.

Muhammed Burak Kizil, Enes Sanli, Niloy J. Mitra, Xuelin Chen, Erkut Erdem, Aykut Erdem, and Duygu Ceylan. Auteur: Language-driven cinematographic framing for human-centric video generation. arXiv preprint arXiv:2606.01900, 2026.

Mengtian Li, Yuwei Lu, Feifei Li, Chenqi Gan, Zhifeng Xie, and Xi Wang. VERTIGO: Visual preference optimization for cinematic camera trajectory generation. In European Conference on Computer Vision, pp. 78–96, 2026.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In Advances in Neural Information Processing Systems, volume 37, pp. 56424–56445, 2024.

Jing Lin, Ruisi Wang, Junzhe Lu, Ziqi Huang, Guorui Song, Ailing Zeng, Xian Liu, Chen Wei, Wanqi Yin, Qingping Sun, Zhongang Cai, Lei Yang, and Ziwei Liu. The quest for generalizable motion generation: Data, model, and evaluation. In International Conference on Learning Representations, 2026.

Christophe Lino and Marc Christie. Intuitive and efficient camera control with the toric space. ACM Transactions on Graphics, 34(4):1–12, 2015.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Transactions on Graphics, 34(6):248:1– 248:16, 2015.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In The Seventh International Conference on Learning Representations, 2019.

Zichong Meng, Yiming Xie, Xiaogang Peng, Zeyu Han, and Huaizu Jiang. Rethinking diffusion for text-driven human motion generation: Redundant representations, evaluation, and masked autoregression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 27859–27871, 2025.

Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo. Reliable fidelity and diversity metrics for generative models. In International Conference on Machine Learning, volume 119, pp. 7176–7185. Proceedings of Machine Learning Research, 2020.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, pp. 4172–4185, 2023.

Mathis Petrovich, Michael J. Black, and Gül Varol. TMR: Text-to-motion retrieval using contrastive 3D human motion synthesis. In IEEE/CVF International Conference on Computer Vision, pp. 9454–9463, 2023.

Mathis Petrovich, Or Litany, Umar Iqbal, Michael J. Black, Gül Varol, Xue Bin Peng, and Davis Rempe. Multi-track timeline control for text-driven 3D human motion generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop on Human Motion Generation, pp. 1911–1921, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, volume 139, pp. 8748–8763. Proceedings of Machine Learning Research, 2021.

Davis Rempe, Mathis Petrovich, Ye Yuan, Haotian Zhang, Xue Bin Peng, Yifeng Jiang, Tingwu Wang, Umar Iqbal, David Minor, Michael de Ruyter, Jiefeng Li, Chen Tessler, Edy Lim, Eugene Jeong, Sam Wu, Ehsan Hassani, Michael Huang, Jin-Bey Yu, Chaeyeon Chung, Lina Song, Olivier Dionne, Jan Kautz, Simon Yuen, and Sanja Fidler. Kimodo: Scaling controllable human motion generation. arXiv preprint arXiv:2603.15546, 2026.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In The Ninth International Conference on Learning Representations, 2021.

Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. Human motion diffusion model. In The Eleventh International Conference on Learning Representations, 2023.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH Conference Papers, pp. 1–11, 2024a.

Zixuan Wang, Jia Jia, Shikun Sun, Haozhe Wu, Rong Han, Zhenyu Li, Di Tang, Jiaqing Zhou, and Jiebo Luo. DanceCamera3D: 3D camera movement synthesis with music and dance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7892–7901, 2024b.

Yuxin Wen, Qing Shuai, Di Kang, Jing Li, Cheng Wen, Yue Qian, Ningxin Jiao, Changhai Chen, Weijie Chen, Yiran Wang, Jinkun Guo, Dongyue An, Han Liu, Yanyu Tong, Chao Zhang, Qing Guo, Juan Chen, Qiao Zhang, Youyi Zhang, Zihao Yao, Cheng Zhang, Hong Duan, Xiaoping Wu, Qi Chen, Fei Cheng, Liang Dong, Peng He, Hao Zhang, Jiaxin Lin, Chao Zhang, Zhongyi Fan, Yifan Li, Zhichao Hu, Yuhong Liu, Linus, Jie Jiang, Xiaolong Li, and Linchao Bao. HY-Motion 1.0: Scaling flow matching models for text-to-motion generation. arXiv preprint arXiv:2512.23464, 2025.

Bizhu Wu, Jinheng Xie, Meidan Ding, Zhe Kong, Jianfeng Ren, Ruibin Bai, Rong Qu, and Linlin Shen. FineMotion: A dataset and benchmark with both spatial and temporal annotation for finegrained motion generation and editing. In IEEE/CVF International Conference on Computer Vision, pp. 13837–13846, 2025.

Lixing Xiao, Shunlin Lu, Huaijin Pi, Ke Fan, Liang Pan, Yueer Zhou, Ziyong Feng, Xiaowei Zhou, Sida Peng, and Jingbo Wang. MotionStreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space. In IEEE/CVF International Conference on Computer Vision, pp. 10086–10096, 2025.

Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14730–14740, 2023.

Mengchen Zhang, Tong Wu, Jing Tan, Ziwei Liu, Gordon Wetzstein, and Dahua Lin. GenDoP: Auto-regressive camera trajectory generation as a director of photography. In IEEE/CVF International Conference on Computer Vision, pp. 18229–18239, 2025.

Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. MotionDiffuse: Text-driven human motion generation with diffusion model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(6):4115–4128, 2024.

Kaifeng Zhao, Gen Li, and Siyu Tang. DartControl: A diffusion-based autoregressive motion model for real-time text-driven motion control. In International Conference on Learning Representations, pp. 23569–23592, 2025.

Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5738–5746, 2019.

## SUPPLEMENTARY CONTENTS

A Disclosure of AI Use 14   
B Representation and Implementation 14   
B.1 Symmetric and asymmetric representations 14   
B.2 Implementation, training objectives and inference 15   
C DPA and IPA Data Construction 16   
C.1 Direction-paired augmentation 16   
C.2 Intensity-paired augmentation 17   
D Evaluation and Baseline Protocols 19   
D.1 Baseline protocols . . 19   
D.2 Direction-paired training 19   
E Translation-Intensity Control 20   
F User Study 20   
G Qualitative results on external motions and prompts 22   
H More Qualitative Results 24

The authors conceived the research ideas, designed the methods, implemented the core code, and analyzed the experimental results. Under the authors’ direction, generative AI tools assisted with supporting code implementation, experimental execution, threshold selection for DPA/IPA data construction, and language refinement. The authors made the final research decisions and take full re sponsibility for the methods, implementation, experimental results, analyses, and final manuscript.

## B REPRESENTATION AND IMPLEMENTATION

## B.1 SYMMETRIC AND ASYMMETRIC REPRESENTATIONS

PulpMotion’s symmetric representation. PulpMotion’s shared encoder (Courant et al., 2026) first maps human motion and camera trajectory to a joint latent, which is then split into human and camera components:

$$
z _ { H C } ^ { P } = E _ { H C } ( H , C ) , \quad ( z _ { H } ^ { P } , z _ { C } ^ { P } ) = \mathrm { s p l i t } ( z _ { H C } ^ { P } ) , \quad z _ { F } ^ { P } = W z _ { H C } ^ { P } .\tag{S1}
$$

Independent human and camera decoders reconstruct their respective inputs from $z _ { H } ^ { P }$ and $z _ { C } ^ { P } .$ A third decoder reconstructs framing from the auxiliary latent $z _ { F } ^ { P } ,$ , obtained by a learned linear map W. The shared encoder and all three reconstruction branches are trained together. Both latents can depend on both inputs: camera reconstruction has access to human context, while human reconstruction also depends on camera input through the shared encoder. Consequently, human representation learning is coupled to camera training, and human encoding requires camera input.

(a) PulpMotion: symmetric representation  
![](images/341b93bb0f0720263fec5586c15fbb84a72597f3183d6d2c651dfd206540168d.jpg)

(b) AESOP: asymmetric representation  
![](images/6b42a293fe8335e817c38f34c04f9ba06b2bcdb5a536db96be58c8fcdbf92132.jpg)  
Figure S1: human–camera representation dependencies. (a) PulpMotion’s symmetric representation encodes both inputs with a shared encoder, reconstructs each from its latent, and learns an auxiliary framing latent from their concatenation (Courant et al., 2026). (b) AESOP separately encodes the two inputs. The human decoder reads only human latents; the camera decoder concatenates camera latents with frozen human latents (circled C). Human representation training is completed before camera training.

Table S1: Training and reconstruction dependencies. Both representations provide human context for camera reconstruction. AESOP additionally separates human learning from camera training and makes human reconstruction independent of camera input.
<table><tr><td></td><td>Training</td><td colspan="2">Reconstruction</td></tr><tr><td>Method</td><td>Independent human learning</td><td>camera-independent human</td><td>human-aware camera</td></tr><tr><td></td><td>PulpMotion (symmetric) × Both streams trained together</td><td>X</td><td>√</td></tr><tr><td>AESOP (asymmetric)</td><td>√ Train human first, then freeze</td><td>√</td><td>√</td></tr></table>

AESOP’s asymmetric representation. AESOP retains PulpMotion’s human and camera feature format but uses separate encoders (Fig. S1). Its human decoder reads only the human latent; the camera decoder reads the concatenated human and camera latents.

Table S2 evaluates deterministic encoding and decoding. AESOP has lower human positional, translation-velocity and jerk reconstruction errors, while PulpMotion has lower rotation-velocity error. Root-aligned human error is smaller than global error for both representations, indicating the contribution of root-trajectory deviations. Table 1 reports distributional and camera-pose reconstruction quality.

Table S2: Autoencoder reconstruction on the test split. Sym./asym. denote PulpMotion's symmetric and $\mathrm { { A E S O P } _ { \mathrm { { S } } } }$ asymmetric autoencoders. Motion errors are clip-balanced and measured in mm/s, <sup>◦</sup>/s and $\mathrm { { m } / \mathrm { { s } ^ { 3 } } }$ , respectively. MPJPE, ADE and FDE are in meters; aligned MPJPE removes root translation. Bold marks the best value between the autoencoders.
<table><tr><td rowspan="2">Autoencoder</td><td colspan="4">human reconstruction</td><td colspan="4">camera reconstruction</td></tr><tr><td>Global MPJPE↓</td><td>MPJPE↓</td><td>Aligned Root ADE↓</td><td>Root FDE↓</td><td>FDE↓</td><td>Trans. vel.↓</td><td> $\mathrm { R o t . \ v e l . } \downarrow$ </td><td>Trans. jerk↓</td></tr><tr><td>PulpMotion (sym.)</td><td> $0 . 1 6 2 0 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 0 6 1 7 ^ { \pm 0 . 0 1 }$ </td><td> $0 . 1 3 7 6 ^ { \pm 0 . 0 1 }$ </td><td> $0 . 3 2 8 3 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 1 6 7 7 ^ { \pm 0 . 0 2 }$ </td><td> $1 2 4 . 2 8 ^ { \pm 3 . 0 2 }$ </td><td> $\mathbf { 1 0 . 5 5 ^ { \pm 0 . 1 0 } }$ </td><td> $1 5 4 . 6 7 ^ { \pm 1 . 5 4 }$ </td></tr><tr><td>AESOP (asym.)</td><td> $\mathbf { 0 . 1 3 0 4 ^ { \pm 0 . 0 1 } }$  </td><td> $\mathbf { 0 . 0 4 7 9 ^ { \pm 0 . 0 0 3 } }$  一</td><td> $\mathbf { 0 . 1 0 6 3 ^ { \pm 0 . 0 1 } }$  </td><td> $\mathbf { 0 . 2 6 1 1 ^ { \pm 0 . 0 2 } }$  </td><td> $\mathbf { 0 . 0 3 5 6 ^ { \pm 0 . 0 0 2 } }$  一</td><td> $\mathbf { 1 0 . 1 1 ^ { \pm 0 . 8 1 } }$ </td><td> $1 0 . 9 8 ^ { \pm 0 . 2 1 }$ </td><td> $\mathbf { 8 . 5 5 ^ { \pm 0 . 4 0 } }$ </td></tr></table>

## B.2 IMPLEMENTATION, TRAINING OBJECTIVES AND INFERENCE

Per-frame features. We use PulpMotion’s per-frame feature format (Courant et al., 2026), whose human features follow the SMPL-based representation of Petrovich et al. (2024); SMPL is defined by Loper et al. (2015). Each human frame has 199 features describing root height, planar root velocity, yaw velocity, local joint rotations and local joint positions. Each camera frame has 14 features: two field-of-view angles, three camera-minus-human relative-position channels, a six dimensional rotation representation (Zhou et al., 2019) and three world-space translation increments.

Latent interface. For a sequence of T frames, the human and camera encoders produce $\lceil T / 4 \rceil$ latent tokens with 128 and 64 channels, respectively. The human decoder reads only the 128-channel human latent; the camera decoder reads the 192-channel concatenation of human and camera latents.

Noncausal width-256 temporal encoders use stride four; transposed-convolution decoders crop to the requested length. A valid-frame mask marks actual sequence frames and excludes batch padding from losses; latent masks retain $\lceil T / 4 \rceil$ tokens. Both flow Transformers have 12 layers, width 512, eight attention heads, FFN multiplier four and dropout 0.1. Camera blocks apply self-attention, camera-text attention, full-human attention and FFN in order, using frozen 512-dimensional CLIP text features (Radford et al., 2021).

For stream $k ~ \in ~ \{ H , C \}$ , let $\widetilde { z } _ { k }$ denote the encoder output and $z _ { k }$ the normalized latent used by the flow. Normalization uses training-only channel statistics $\mu _ { k } , s _ { k }$ and standardized-stream mean/covariance factors $m _ { k } , L _ { k }$ . Cholesky whitening uses ridge $1 0 ^ { - 4 }$ ; its inverse precedes decoding:

$$
z _ { k } = L _ { k } ^ { - 1 } \big ( ( \widetilde { z } _ { k } - \mu _ { k } ) / s _ { k } - m _ { k } \big ) , \qquad \widetilde { z } _ { k } = \mu _ { k } + s _ { k } \odot \big ( m _ { k } + L _ { k } z _ { k } \big ) .\tag{S2}
$$

Invalid positions are zeroed after either operation. Physical decoding integrates de-normalized translation increments and anchors the first camera center using the decoded relative vector and human root (Courant et al., 2026), using predicted features for initialization. Human-feature, relative-center and translation-increment normalization are distinct from latent whitening.

Stage 1 objectives. Let M mark valid frames and $M _ { \Delta , t } = M _ { t } M _ { t - 1 } { \mathrm { ~ f o r ~ } } t = 1 , \dots , T - 1$ . Angle brackets average selected scalar elements. For normalized features $H , C _ { \cdot }$ , cumulative de-normalized yaw $\psi ,$ , and decoded root trajectory $r ,$ the complete reconstruction losses are

$$
\begin{array} { r l } & { \mathcal { L } _ { H } ^ { \mathrm { r e c } } = \langle \ell _ { \mathrm { S L 1 } } ( \hat { H } , H ) \rangle _ { M } + \langle ( \Delta \hat { H } - \Delta H ) ^ { 2 } \rangle _ { M _ { \Delta } } } \\ & { \qquad + 1 0 ^ { - 3 } \langle 1 - \cos ( \hat { \psi } - \psi ) \rangle _ { M } + 3 \times 1 0 ^ { - 3 } \langle \ell _ { \mathrm { S L 1 } } ( \hat { r } , r ) \rangle _ { M } , } \\ & { \mathcal { L } _ { C } ^ { \mathrm { r e c } } = \langle \ell _ { \mathrm { S L 1 } } ( \hat { C } , C ) \rangle _ { M } + \langle ( \Delta \hat { C } - \Delta C ) ^ { 2 } \rangle _ { M _ { \Delta } } . } \end{array}\tag{S3}
$$

Stage 2 objectives and optimization. We sample $u \sim \mathcal { U } ( 0 , 1 )$ , set $\sigma = 5 u / ( 1 + 4 u )$ and draw $\epsilon \sim \mathcal { N } ( 0 , I )$ . The velocity target $v ^ { * }$ follows Eq. 1. For $n _ { b }$ valid latent tokens and stream dimension $D ,$ define

$$
e _ { b } = \frac { \sum _ { d , t } M _ { b , t } ( \boldsymbol { v } _ { b , d , t } - \boldsymbol { v } _ { b , d , t } ^ { * } ) ^ { 2 } } { D n _ { b } } , \quad \mathcal { L } _ { \mathrm { o r i g i n a l } } = \frac { 1 } { B } \sum _ { b } e _ { b } , \quad \mathcal { L } _ { \mathrm { c o n t i n u a t i o n } } = \frac { \sum _ { b } n _ { b } e _ { b } } { \sum _ { b } n _ { b } } .\tag{S4}
$$

Continuation weights clips by length; original training weights them equally. Each continuation batch has 120 rows, with two rows per control pair. During the 35K DPA phase, direction-pair slots rise from zero to five over the first 2K updates, remain at five through 20K, fall to two through 30K and then to one. During the 15K IPA phase, each batch retains two DPA pair slots; active IPA slots rise to four and null IPA slots to one during the first 1K updates.

Table S3: Training schedule. The final phase additionally trains the intensity MLP at $1 0 ^ { - 4 } .$
<table><tr><td>Stage</td><td>Phase</td><td>Updates</td><td>Batch</td><td>Learning rate</td><td>Frozen components</td></tr><tr><td>Reconstruction</td><td>human autoencoder</td><td>210K</td><td>128</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>None</td></tr><tr><td></td><td>camera autoencoder</td><td>210K</td><td>128</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>human autoencoder</td></tr><tr><td>Generation</td><td>human flow</td><td>105K</td><td>128</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>Autoencoders</td></tr><tr><td></td><td>Original camera</td><td>105K</td><td>128</td><td> $1 0 ^ { - 4 }$ </td><td>Autoencoders, human flow</td></tr><tr><td></td><td> $\mathrm { c a m e r a + D P A }$ </td><td>35K</td><td>120</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td>Autoencoders, human flow</td></tr><tr><td></td><td> $\mathrm { c a m e r a + D P A + I P A }$ </td><td>15K</td><td>120</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td>Autoencoders, human flow</td></tr></table>

Stage 1 uses AdamW (Loshchilov & Hutter, 2019) with $\beta = ( 0 . 9 , 0 . 9 9 9 )$ , zero weight decay, a 1K step linear warmup and cosine decay to $1 0 ^ { - 6 }$ per phase. Stage 2 uses AdamW with $\beta = ( 0 . 9 , 0 . 9 5 )$ weight decay 0.01, gradient clipping 1.0 and EMA decay 0.9999. The human flow warms up for 2K steps and decays its rate by 0.1 at step 80K; original camera training uses a constant rate. Continuations use BF16 and 1K-step warmup.

Inference and conditioning. Sampling follows Section 4.1 with solver shift five and camera-text guidance from Eq. 4. The intensity MLP described in Section 3 has 64 hidden units and a 512- dimensional output added to the camera timestep condition.

## C DPA AND IPA DATA CONSTRUCTION

We construct paired camera targets from the 162,760-clip PulpMotion training pool. DPA changes translation direction and its text condition; IPA changes translation intensity while keeping camera text fixed. Both preserve human motion, camera rotation, field of view and timing; the human latent and decoded output are unchanged across each constructed pair. Figures S2 and S3 illustrate the three construction stages and the retained source counts. Table S4 lists the acceptance rules.

## C.1 DIRECTION-PAIRED AUGMENTATION

1. Pre-screening. From the training pool, DPA selects 22,783 whole-shot captions containing one Truck or Dolly event. Source-motion checks retain 9,143 clips with stable, sufficient translation. Truck uses a fixed camera-local right axis, obtained by averaging the first min $( 8 , \lfloor T / 5 \rfloor )$ right vectors, projecting onto the ground plane and normalizing. Dolly uses camera–human root distance.

2. Augmentation. On the accepted interval $[ t _ { s } , t _ { e } ]$ , we reflect the signed coordinate as $q _ { t } ^ { \prime } = 2 q _ { t _ { s } } -$ q<sub>t</sub>. Truck preserves orthogonal components; Dolly preserves the human-to-camera unit direction. The trajectory is unchanged before the event and carries its endpoint offset afterward. Relativecenter and velocity features are recomputed. Paired minimal text templates differ only in direction.

3. Post-checks. Physical checks assess geometry, framing, visible direction effect and non-target leakage, retaining 781 constructed sources (8.54% of the 9,143 eligible sources). Reconstruction checks assess fidelity, direction, response and visibility, yielding 601 sources: 375 Dolly and 226 Truck. These supply 601 original/opposite training pairs with 1,202 targets. Figure S2 illustrates why the decoded direction is checked as well as the raw target.

The DPA visible-effect score is $V = \operatorname* { m a x } ( \Delta x / 0 . 0 8 , \Delta \alpha / 6 ^ { \circ } )$ for Truck and $V = \Delta h / 0 . 1 2$ for Dolly. Here ∆x is peak projected subject-center separation in image-width units, ∆α is peak humanrelative azimuth separation, and $\Delta h = \operatorname* { m a x } _ { t } | h _ { t } ^ { \prime } / h _ { t } - 1 |$ is peak projected box-height ratio change, comparing original/opposite targets within the event. Decoded retention compares this score after versus before reconstruction.

![](images/d1556132a0d523cdd6f840503a599569c4ae3451b43f90e2c88e9a58629ea3ce.jpg)  
Figure S2: DPA construction and validation. Left: the original Dolly-in trajectory. Middle: its opposite Dolly-out target, with human motion, camera rotation, field of view and timing fixed. Right: a reconstruction rejected because its direction change exceeds the acceptance threshold after decoding. The gold point marks the initial camera center; global views show the starting human pose and full trajectory, and projections show the synchronized final frame. The three stages summarize source filtering, direction reversal and target checks; 601 sources pass the final reconstruction check.

## C.2 INTENSITY-PAIRED AUGMENTATION

1. Pre-screening. The active branch selects trajectories with sufficient translation to provide weaker/stronger targets. The null branch selects low-translation trajectories and pairs an unchanged target with different intensities, teaching the model to retain low-translation shots under intensity changes. Motion, visibility and distance checks, together with DPA-source exclusion, yield 42,844 active and 75,863 null candidates. Deterministic hash ranking selects 36,000 active sources for target construction.

2. Augmentation. An 11-frame triangular filter with reflection padding decomposes camera centers into $p _ { t } = s _ { t } + r _ { t }$ . We apply Eq. 5 and recompute relative-center and velocity features from the transformed trajectory.

For IPA target construction, candidate labels are sampled from $0 < a < 1$ for weaker targets and $1 <$ $a < 2$ for stronger targets. Each interval is divided into four equal-width bins, ordered from smaller to larger a. Their sampling probabilities are (0.1, 0.2, 0.3, 0.4) below one and (0.4, 0.3, 0.2, 0.1) above one, favoring labels closer to a = 1. A fixed-seed sampler selects a bin and then a continuous value within it. For each active source, we try up to eight labels per interval and keep the first weaker and first stronger target that pass the checks below. Null sources retain the unchanged camera target while receiving alternating weaker and stronger labels from the same distributions. The eleven-level evaluation uses the narrower range $a \in [ 0 . 5 , 1 . 5 ]$ , in steps of 0.1.

3. Post-checks. We retain active sources with an accepted target on both sides of $a = 1$ . Physical and construction-representation reconstruction checks yield 14,066 sources (39.07% of the 36,000 constructed sources). After reserving source-video groups for development, the final training set contains 8,000 active sources, yielding 16,000 camera targets labeled $a \neq 1$ , and 2,000 null sources. Development control variants are kept separate from control training and the test split.

![](images/ecd5cbe1e324d2b926581959acd2acedb928738d4b010df7b72997f7142e36e0.jpg)  
Figure S3: IPA target construction. Reduced, original and increased translation share human motion and camera text. Camera travel changes from 0.356 to 0.634 to 0.786 m in this example. The gold point marks the initial camera center; global views and synchronized final-frame projections show the resulting geometry and framing. The three stages summarize the active branch, which supplies 16,000 training targets from 8,000 sources. The accompanying null branch supplies 2,000 low-translation sources with unchanged targets.

Table S4: Construction acceptance criteria for DPA and IPA, grouped by source selection, target geometry, framing and reconstruction.
<table><tr><td>Operation</td><td>Acceptance rules</td></tr><tr><td colspan="2">DPA: direction pairs</td></tr><tr><td>Select source events</td><td>Duration  $\geq 4 5$  frames; same-sign velocity fraction  $\geq 0 . 8 0 ;$  signed net displacement/path  $\geq 0 . 6 0 .$  Truck displacement  $\geq 0 . 2 0$  m; Dolly displacement  $\geq \operatorname* { m a x } ( 0 . 2 0 \mathrm { m } , 0 . 1 0 \rho _ { t _ { s } } ) .$ </td></tr><tr><td>Check target geometry</td><td>Opposite-target human distance  $> 0 . 2 5$  m; opposite/original response ∈ [0.80, 1.25]. Rotation/FOV unchanged; Truck vertical shift  $\leq 1 0 ^ { - 5 }$  m and radial relative change  $\leq 0 . 1 0 ;$  Dolly radial-direction change  $\leq 0 . 0 5 ^ { \circ }$ </td></tr><tr><td>Check framing and effect</td><td>No zero-visible frame; visibility  $\geq 0 . 8 0 ;$  opposite outscreen increase  $\leq 0 . 0 5 .$  Truck: projected center  $\mathrm { s h i f t } \geq 0$  08 image width or viewpoint change  $\geq 6 ^ { \circ }$  Dolly: peak box-height ratio change  $\geq 0 . 1 2 .$ </td></tr><tr><td>Check reconstructed targets</td><td>Feature MSE ≤ 0.01181; both direction signs correct; decoded/raw response ratios ∈ [0.70, 1.30]. No framing failure;  $V _ { \mathrm { d e c } } \geq 1$  and  $V _ { \mathrm { d e c } } / V _ { \mathrm { r a w } } \geq 0 . 7 0 .$ </td></tr><tr><td colspan="2">IPA: intensity pairs</td></tr><tr><td>Select sources</td><td>Duration  $\geq 1 \ : \mathbf { s } .$  Active: path ≥ 0.4118 m, span  $\geq 0 . 1 5$  m, max-step/path  $\leq 0 . 2 0 .$  Null: path ≤ 0.10 m. Source visibility  $\geq 0 . 8 0$  and human distance  $\geq 0 . 2 5$  m; exclude DPA sources.</td></tr><tr><td>Check magnitude and direction</td><td>Active only: absolute requested/actual path-ratio error  $\leq 0 . 0 8 ;$  path/speed-ratio difference  $\leq 0 . 0 3 ;$  direction cosine  $\geq 0 . 9 0$  for  $a \geq 0 . 2 5 ; | a - 1 | \times$  original path  $\geq 0 . 1 5 \ : \mathrm { m } .$ </td></tr><tr><td>Check dynamics and framing</td><td>Active only: speed/acceleration/jerk  $\mathsf { p 9 5 } \leq 0 . 1 4 1 6$  m/frame,  $0 . 0 3 4 9 9$  m/frame2, 0.01315  $\mathrm { m } / \mathrm { f r a m e ^ { 3 } }$  Human distance ∈ [0.25, 12.7716] m; visibility decrease  $\leq 0 . 0 5 ;$ </td></tr><tr><td>Check reconstruction</td><td>severe-outscreen increase  $\leq 0 . { \dot { 0 } } 3 .$  Active: construction-representation decoded path-ratio error  $\leq 0 . 1 5 .$ </td></tr></table>

Construction checks. Table S4 groups the acceptance criteria. Dynamics limits use the 99.5th percentiles of a 1,000-clip training-data pilot, in per-frame units. A joint is visible when its projection is finite, lies inside the image and has camera-space depth $> 1 \bar { 0 } ^ { - 4 }$ m. DPA visibility averages this indicator over joints and frames; outscreen is its complement. IPA visibility counts frames with any visible joint; severe-outscreen counts frames with fewer than half the joints visible. The decoded IPA gate uses the construction representation, and null targets remain unchanged.

## D EVALUATION AND BASELINE PROTOCOLS

## D.1 BASELINE PROTOCOLS

We follow the evaluation setup in Section 4.1 and the AESOP training schedule in Table S3. Givenhuman evaluation supplies reference human motion; joint evaluation uses each method's generated motion. Table S5 lists adaptations and sampling configurations.

Table S5: Baseline inputs, conditioning and sampling. For PulpMotion, the tuple denotes conditional-text guidance, autoregressive-context guidance, and auxiliary projection guidance, respectively. A negative autoregressive-context weight selects standard CFG; zero auxiliary projection weight disables that guidance term. MAR generates autoregressively, with a diffusion sampler inside each iteration. EDM, DDIM and DDPM follow Karras et al. (2022), Song et al. (2021) and Ho et al. (2020), respectively.
<table><tr><td>Method</td><td>Inputs and conditioning</td><td>Sampling and guidance</td></tr><tr><td>DIRECTOR-C</td><td>Camera text and human root trajectory</td><td>EDM: 10 Euler/Heun steps; camera CFG 1.4</td></tr><tr><td>DanceCamera3D</td><td>Camera text and full human motion; text replaces music</td><td>DDIM: 50 steps, η = 1; human/camera CFG 1.75/1</td></tr><tr><td>CCD</td><td>Camera text; text-only network</td><td>DDPM: 1,000 steps; camera CFG 2</td></tr><tr><td>CCD-H</td><td>Camera text and human joints; text cross-attention</td><td>DDPM: 1,000 steps; camera CFG 2</td></tr><tr><td>PulpMotion DiT</td><td>Human and camera text; symmetric representation</td><td>DDPM: 50 steps;  $( g _ { c } , g _ { m } , g _ { z } ) = ( 1 1 , - 1 , 0 )$ </td></tr><tr><td>PulpMotion MAR</td><td>Human and camera text; symmetric representation</td><td>18 autoregressive iterations, each with 50 DDPM steps;  $( g _ { c } , g _ { m } , g _ { z } ) = ( 3 . 5 , 2 , 0 )$ </td></tr><tr><td>AESOP</td><td>Human motion and camera text (given); human text and camera text (joint)</td><td>Euler: 50 steps per stream; human/camera CFG 1/1.5</td></tr></table>

Generative training presents 13.44M examples to each camera-only baseline and 26.88M to each PulpMotion joint model. AESOP presents 13.44M examples to each human/camera flow, followed by 6M camera continuation examples. These counts include repeated presentations and both members of control pairs; Table S3 gives AESOP's phase durations and batch sizes.

Human-relative camera decoding. CCD and CCD-H decode camera position relative to the supplied human head in a heading-aligned frame, with a head-directed look-at rotation adjusted by predicted screen position (Jiang et al., 2024). AESOP and PulpMotion anchor camera position to the human root at the first frame and accumulate world-space increments (Courant et al., 2026).

## D.2 DIRECTION-PAIRED TRAINING

Paired sampling and scoring. Each prompt arm contains the same 3,194 clips with active reference translation, giving 6,388 clip–prompt cases per task. Original and reversed prompts are processed by the same text encoder; human latents, camera noise and valid-length masks are shared within each pair. Given-human uses reference motion; joint generation samples the human first.

Direction accuracy measures the sign of each reference-active translation axis at fixed reference time positions. Each clip first averages sign correctness over those positions and axes, then clips receive equal weight. Reversed targets negate the corresponding reference signs. Reference-inactive positions are excluded; generated still predictions at active positions count as errors. Combined accuracy averages the two prompt arms.

Table S6: Direction accuracy (%). DPA and factual continuations share the starting checkpoint, 35K updates, batch allocation and sampling schedule; factual pairs replace directional counterfactuals in the control. Bold marks the best mean in each column. Section D.2 defines paired sampling and scoring.
<table><tr><td></td><td colspan="3">Given-human generation</td><td colspan="3">Joint generation</td></tr><tr><td>Checkpoint</td><td>Original↑</td><td>Reversed↑</td><td>Combined↑</td><td>Original↑</td><td>Reversed↑</td><td>Combined↑</td></tr><tr><td>Camera original</td><td>86.38±0.86</td><td>75.66±0.68</td><td> $8 1 . 0 2 ^ { \pm 0 . 1 5 }$ </td><td> $8 3 . 1 5 ^ { \pm 0 . 1 3 }$ </td><td>76.42±0.43</td><td> $7 9 . 7 8 ^ { \pm 0 . 2 8 }$ </td></tr><tr><td>35K factual-pair control</td><td> $8 7 . 6 0 ^ { \pm 0 . 4 3 }$ </td><td> $7 7 . 7 1 ^ { \pm 0 . 1 3 }$ </td><td> $8 2 . 6 6 ^ { \pm 0 . 1 7 }$ </td><td> $8 4 . 9 5 ^ { \pm 1 . 1 3 }$ </td><td> $7 8 . 5 1 ^ { \pm 0 . 1 7 }$ </td><td> $8 1 . 7 3 ^ { \pm 0 . 5 1 }$ </td></tr><tr><td>+ DPA</td><td> ${ \bf 8 9 . 6 6 ^ { \pm 0 . 6 7 } }$ </td><td> $\mathbf { 7 9 . 0 0 ^ { \pm 0 . 2 2 } }$ </td><td>一  $\mathbf { 8 4 . 3 3 ^ { \pm 0 . 2 8 } }$ </td><td> ${ \bf 8 6 . 6 3 ^ { \pm 0 . 7 7 } }$ </td><td>_  $\mathbf { 7 9 . 5 8 ^ { \pm 0 . 5 0 } }$ </td><td>_  ${ \bf 8 3 . 1 1 ^ { \pm 0 . 1 4 } }$ </td></tr></table>

## E TRANSLATION-INTENSITY CONTROL

For the endpoint change from a = 1 to 1.5, path length is nondecreasing for 97.31% of given-human test outputs and 97.01% of joint test outputs.

## F USER STUDY

Study design. The anonymous study follows the task allocation in Section 4.4 and includes baseline–baseline comparisons. Each questionnaire receives a randomized, balanced assignment with equal target exposure per method pair, balanced task order, and randomized left–right presentation and trial order. Figure S4 shows synchronized camera and spatial views.

Evaluation criteria. Both tasks assess camera-text alignment, camera motion quality and framing. Given-human trials additionally ask for overall camera quality; joint trials add human-text alignment, human motion quality and overall human–camera quality. Each criterion uses five responses from “A much better” to “B much better,” including a tie, plus “Cannot judge.” Overall quality is judged directly. Both videos must reach 90% playback coverage before a trial can be saved.

Analysis. For each clip, method pair and criterion, we compute preference as $( W + 0 . 5 T ) / ( W +$ $T + L )$ , where W, T and L count wins, ties and losses. “Cannot judge” responses are counted separately. Within each task and criterion, pairwise scores average equally over clips, and methodlevel scores average equally over opponents. Overall quality is the primary outcome for each task. We estimate 95% confidence intervals using 10,000 participant-cluster bootstrap resamples with the stimulus set fixed.

Results. The 29 complete questionnaires comprise 580 given-human and 522 joint trials, covering all 120/60 clip–pair comparisons. Of 5,452 criterion responses, 5,238 are judgeable and 214 are “Cannot judge.”

Figure S5 reports all task-specific criteria. AESOP has the highest preference on each criterion: joint human-text alignment and motion-quality scores are 80.4% and 80.9%, while camera alignment, motion-quality and framing scores range from 87.5% to 89.9%. Section 4.4 and Table 3 report overall preference.

![](images/40f8a1d9e13f7332fa91d9ed3e82074df65c7b6ee6fc0cf0e16f00b044b1ef8d.jpg)

![](images/3e7c67a194c81b3a7b70b4fb7cf34e8e083937a8dfa6faaa2a6a879b3fa05245.jpg)

<table><tr><td>Criterion</td><td>A much better</td><td>A slightly better</td><td>About equal</td><td>B slightly better</td><td>B much better</td><td>Cannot judge</td></tr><tr><td>▼ Camera· Text match Which camera movement better matches the described type, direction, and temporal</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td> Camera · Motion quality</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>▼ Camera • Framing Which camera view maintains more appropriate subject placement, size, cropping, and surrounding space over time?</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td> Camera · Overall quality</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>← Previous</td><td></td><td>Current choices saved on this device</td><td></td><td></td><td>Save &amp; continue →</td><td></td></tr></table>

(a) Given-human camera generation

<table><tr><td>Criterion</td><td>A much better</td><td>A slightly better</td><td>About equal</td><td>B slightly better</td><td>B much better</td><td>Cannot judge</td></tr><tr><td> Human · Text match</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td> Human · Motion quality</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>▶ Camera • Text match</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td> Camera · Motion quality</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>▶ Camera · Framing</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>Human &amp; camera Overall quality</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>← Previous</td><td colspan="4">Current choices saved on this device</td><td colspan="2">Save &amp; continue →</td></tr></table>

(b) Joint human–camera generation

Figure S4: User-study interface. Anonymized A/B videos show the generated camera view above an external spatial view. Participants compare text alignment, motion quality, framing and taskspecific overall quality. The joint task additionally evaluates human motion.  
![](images/2c75a055a02d2f02a2e80f883d8853300f85098a2b1096eb06fe2b5c17a3aa7c.jpg)

![](images/e2c4fcff72cc62dcfb63a8670cff41db09e0a0f195a4b254c6636079cc17f14b.jpg)  
Figure S5: User-study preferences across all criteria. Points show clip-balanced, equal-opponent preference scores; bars show 95% participant-cluster bootstrap intervals. The vertical dashed line marks 50% preference. Shading highlights the separately rated overall-quality endpoints.

## G QUALITATIVE RESULTS ON EXTERNAL MOTIONS AND PROMPTS

We demonstrate the same PulpMotion-trained AESOP model in three settings: camera generation from HumanML3D motions, joint generation from HumanML3D action descriptions, and camera generation from externally synthesized human motions.

Ground-truth human motions. We adapt HumanML3D (Guo et al., 2022) motions to the human representation and generate cameras conditioned on these motions and camera prompts. Figure S6 shows a static camera observing running in place, a pull-out widening the view of a walking and turning figure, and trucking accompanying jumping and spinning.

Joint generation from action descriptions. We use ground-truth HumanML3D action descriptions to generate both human motion and camera trajectory (Figure S7). A pull-in tightens the view of arm stretching, while a static camera observes an approaching runner.

Synthesized human motions. We condition the same camera generator on motions synthesized by HY-Motion. Figure S8 shows static views for bowing and rhythmic stepping, a pull-out for a surfboard pop-up, and a pull-in for stepping back and blocking.

## Given-human Camera generation

![](images/602628f164723c9b75570d9335c20d71403123a675013b423f6c2f2c348e7a21.jpg)  
Human: a person runs on the spot. Camera: The camera remains static throughout the shot.

![](images/e74bd43e55f07430b61ffce7be7fc833330e16d2adf86fa37cfcaaf015fae927.jpg)  
Human: a figure walks and spins on their heel. Camera: The camera pulls out throughout the shot.

![](images/05bf332d473c6dd1112917a357bb448b75c8d3a8cad0b9eec86efe9e5bfe2c6c.jpg)  
Human: a person jumps up and down once. Camera: The camera trucks right throughout the shot.

![](images/1058f1d83d0d6a0e0c919d7016d1e10d81f92eb26b52774c0b9702aefc1e1677.jpg)  
Human: a person is spinning and a circle and then kicks his red foot. Camera: The camera trucks right throughout the shot.

Figure S6: Camera generation conditioned on HumanML3D motions. Each example shows a global view and three successive camera views; lighter poses indicate earlier frames. Global views are fitted separately to each scene.

## Joint Human–Camera generation

![](images/8e3632a107f2871d81e7521437c70867eac33673e3bb26665e449b0978554cd1.jpg)  
Human: the man sits on a stool and looks around. Camera: The camera pulls out throughout the shot.

![](images/d4c09a66ef692a2649347b77afbd00fdc96ce7841fe53ab8ea1a3d27d791ea0d.jpg)  
Human: person stretches both arms up and then put arms down. Camera: The camera pulls in throughout the shot.

![](images/bc91eea7cf6d5704fab9d77b1c3b0153bb1771d88c866a3222b0b2517492c945.jpg)  
Human: a person runs forward and stops short. Camera: The camera remains static throughout the shot.

![](images/96fae3ac226f0f29bc8c0fb03873e379b53c9204c1c8f405c4cbbffbc65cca57.jpg)  
Human: a person walks three steps to his right, then five steps to his left, and finally three steps to his right. Camera: The camera trucks right throughout the shot.

Figure S7: Joint generation from HumanML3D action descriptions. Each example shows generated human motion and camera trajectory in a global view and three successive camera views.

## Camera generation from HY­Motion outputs

![](images/544cba32e1c6ac9fae06827c84b860d0d83f3a1e3b488ae6dba4c0b04c2e7604.jpg)  
Human: a person bows down facing the front then bows down facing the right. Camera: The camera remains static throughout the shot.

![](images/18d79291b3dfa9fc10b3b109b420b3e4a614ba91861a443f6b4e9d2170d684c8.jpg)  
Human: A person swings their arms alternately to the rhythm while stepping. Camera: The camera remains static throughout the shot.

![](images/7ced9ac24f2b3fa86263d5919b0aae656f90920222557634d4b5f0100eec8dce.jpg)  
Human: A person does a pop-up on a surfboard. Camera: The camera pulls out throughout the shot.

![](images/3656634d220cc475a65a0a427b504de454a16c9f40d94a8f1baf80d2e0abedb7.jpg)  
Human: A person steps back with their left foot, then blocks with their right arm. Camera: The camera pulls in throughout the shot.

Figure S8: Camera generation conditioned on HY-Motion outputs. Four synthesized human motions are paired with static, pull-out and pull-in camera prompts. Each example includes a global view and three successive camera views.

## H MORE QUALITATIVE RESULTS

Figures S9 and S10 show two additional examples for each task. Each method shows a global view and three camera projections in temporal order from top to bottom.

(a) Example 1  
![](images/8b13bc193319c335c19c33e303ecda92973606b13e08fe213014a7a51cf472ad.jpg)  
human: A person bends forward, lifts their right leg, and then extends their right arm upwards. camera: The camera starts with a boom top shot.

(b) Example 2  
![](images/fa09152a091c901c6b23d60b20e019af4a20259bd2546db632dcd38f6e57ea81.jpg)  
human: A person walks forward along the sidewalk, maintaining a steady pace.  
camera: The camera performs a pull-out motion throughout the entire shot.

Figure S9: Additional given-human camera generation examples. Global views are fitted separately to each trajectory.

(a) Example 1  
![](images/a4b3e94cfa322c6891102c20ab15038e3cfac732b010176b84592034bd266a80.jpg)  
human: A person walks forward, swinging their arms naturally. camera: The camera moves right along a dolly or truck during the entire shot.

(b) Example 2  
![](images/10ee365f8bca876f84162150b82f39869378f35fb465002f084c211515f12a17.jpg)  
human: A person sits on a stool, moves their left hand towards their face, and then places it back on the instrument. camera: The camera gradually moves closer to the subject throughout the shot.

Figure S10: Additional joint human–camera generation examples. Global views are fitted separately to each trajectory.