# FashMotion: Ultra-Fast Long-Term 3D Motion Generation

Kaixing Yang , Xulong Tang<sup>‡</sup> , Ronghui Li<sup>‡</sup> , Xiangyue Zhang , Ziqiao Peng , Mingshuang Luo, Chenxi Zhao , Haoyu Wu , Ju Fan , Hongyan Liu<sup>†</sup> , Puwei Wang<sup>†</sup> , Zijiao Zeng<sup>†</sup> , and Jun He

Abstract—3D human motion generation plays a crucial role in game development, extended reality (XR), digital humans, and embodied simulation. Beyond motion generation, practical pipelines include rigging, retargeting, and rendering, making ultra-fast long-term 3D motion generation without compromising quality valuable for real-time interactive applications. Despite promising progress, high-quality long-horizon motion generation under tight inference budgets remains challenging, owing to 2 coupled bottlenecks: (1) generative strategies that require many sampling steps for high-quality 3D motion generation and (2) model architectures with substantial per-step computational and memory costs for long sequences. We present FlashMotion, a co-designed framework for high-quality long-horizon 3D motion generation under a strict inference budget, covering text-to-motion, music-to-dance, and speech-to-gesture tasks. To enhance trajectory-level efficiency, we introduce a 3D motion-space MeanFlow formulation with kinematic and dynamic consistency constraints for few-step generation, together with an Adaptive Soft masking mechanism for temporally coherent motion editing. To improve per-step efficiency, we design a lightweight nonautoregressive BiMamba architecture with condition-guided state selection and AdaIN-based condition fusion mechanism, enhancing crossmodal alignment and generalization to longer motion sequences. Extensive experiments demonstrate that FlashMotion achieves overall state-of-the-art performance in both quality and efficiency across textto-motion, speech-to-gesture, and music-to-dance generation. Code is available at FlashMotion.

Index Terms—Digital Humans, 3D Motion Generation, Efficient Motion Generation, Long-Horizon Generation.

## 1 INTRODUCTION

modal motion modeling [1], [2], 3D human motion generation has attracted increasing interest for translating text, speech, and music into expressive human motion. This

capability supports applications in robot motion generation, extended reality (XR) character animation, and mobile motion interaction (Fig. 7). These applications require responsive and sustained interaction. As a key enabler of such interaction, a broadly applicable 3D motion generation method should satisfy 3 key requirements: (1) Motion Quality: generating physically plausible and semantically expressive motion that aligns with the input conditions; (2) Generation Efficiency: achieving low inference latency and memory consumption to leave sufficient computational resources for other pipeline stages, including rigging, retargeting, and rendering; and (3) Long-Horizon Generation: sustaining high motion fidelity and smooth temporal transitions over extended sequences.

Across text-to-motion, speech-to-gesture, and music-todance tasks, 3D human motion generation has advanced along 2 complementary dimensions. (1) Generative Strategy. Representative strategies include direct motion regression [3], [4], adversarial learning [5], [6], autoregressive generation [7], [8], diffusion [9], [10], and flow matching [11], [12], with different trade-offs between generation quality and sampling efficiency. (2) Model Architecture. Temporal backbones range from recurrent networks, including LSTMs [13], [14], to Transformers [4], [7] and Mamba-based models [15], [16], with different capabilities for temporal modeling and conditional integration. However, highquality long-horizon generation under a strict inference budget remains challenging due to 2 coupled bottlenecks: (1) generative strategies that require many sampling steps for high-quality 3D motion generation and (2) model architectures with substantial per-step computational and memory costs for long sequences. These costs leave limited computational resources for other production stages, such as rigging, retargeting, and rendering, potentially degrading the user experience in interactive applications.

To this end, we present FlashMotion, a co-designed framework for high-quality long-horizon 3D motion generation under a strict inference budget. The framework supports diverse conditional motion generation tasks, including text-to-motion, music-to-dance, and speech-to-gesture. Our key observation is that end-to-end generation efficiency is jointly determined by (1) the number of sampling steps and (2) the computational cost of each step. Optimizing either factor alone limits practical speedups: few-step sampling remains costly with an expensive backbone, while a lightweight architecture still incurs substantial latency when many sampling steps are required. FlashMotion therefore jointly optimizes trajectory-level and per-step efficiency. To improve trajectory-level efficiency, we introduce a 3D motion-space MeanFlow formulation with kinematic and dynamic consistency constraints for few-step generation, along with an Adaptive Soft masking mechanism for temporally coherent motion editing. To improve per-step efficiency, we design a lightweight non-autoregressive architecture that combines a BiMamba-based backbone with condition-guided state selection and AdaIN-based condition fusion. Together, FlashMotion achieves state-of-the-art motion quality across all 3 tasks, generating physically plausible, semantically expressive motion at leading throughput with a compact ∼60M-parameter model. Fig. 1 presents quality–throughput comparisons with representative baselines (top) and conditioned motion examples (bottom). Panels (a)-(c) report results on BABEL, BEAT2, and FineDance, respectively. FPS is measured for 10,000-frame outputs on an RTX 3090 at batch size 1.

![](images/628049a3a40b8e2f7bc8d83a4b0ae0a08a14f237b79ab1cd90a8671a90cd7fe4.jpg)

![](images/329ea470a8daa12afffa5cf7419d841546eb2d7c53426dffda260ad3bcd51b08.jpg)

![](images/99e70aafa335d5ca5c59da4630e2d3b0f52bc7b4aa1ebbc4ca824d133dda6b3d.jpg)

![](images/a93ae78d13436d7876cba8f8c454db71a5bdb9b96cb3ebcda530d45216b31971.jpg)  
p ... performing ...(a) Text-to-Motion

Speech-to-Gesture Music-to-Dance elevatoElevator passed bPassed by a smileA smile richJazz  
![](images/292142cd3635e6802c5a7728a033b96b24ce8ae101eac88c182d4f8cae549f09.jpg)  
(b) Speech-to-Gesture

![](images/7b75682bfdc60b0de2d4ccdf63963948875a9bd03edceddb9ec44a4efd2b9744.jpg)  
(c) Music-to-Dance  
<sup>“...</sup> <sup>elevator</sup> <sup>...”</sup> <sup>“...</sup> <sup>passed</sup> <sup>by</sup> <sup>...”</sup> <sup>“...</sup> <sup>a</sup> <sup>smile</sup> <sup>...” ...</sup> <sup>energetic</sup> <sup>jazz</sup> <sup>with</sup> <sup>rich</sup> <sup>rhythmic</sup> <sup>variation</sup> <sup>...</sup>Fig. 1. FlashMotion generates physically plausible, semantically expressive motion at leading throughput with a compact ∼60M-parameter model. Top: quality–throughput comparisons with representative baselines; bottom: conditioned motion examples. Panels (a)–(c) use BABEL, BEAT2, and FineDance. FPS is measured for 10,000-frame outputs (RTX 3090, batch size 1).

Trajectory-level efficiency via generative strategy. To reduce sampling steps without sacrificing motion quality, FlashMotion combines a 3D motion-space MeanFlow formulation with kinematic and dynamic consistency constraints. From a kinematic perspective, coherent 3D human motion requires coordinated joint trajectories and smooth temporal evolution, motivating a generative process that continuously transforms noise into structured motion. Flow Matching provides a natural framework for modeling this transformation through a learned velocity field [17], [18]. However, standard Flow Matching learns instantaneous velocities, whereas few-step sampling requires accurate finiteinterval updates, making generation sensitive to numerical integration errors. MeanFlow [19] addresses this limitation by modeling interval-averaged velocity. Nevertheless, directly predicting velocity makes it less straightforward to impose structural constraints on the resulting 3D motion. Since 3D human motion representation is subject to tightly coupled kinematic and dynamic constraints, errors in largestep updates can manifest as temporal jitter, root drift, foot sliding, and implausible articulations. This motivates predicting clean motion explicitly, where motion-specific supervision can be applied directly, while deriving the velocity required for sampling from the prediction. We therefore introduce a 3D motion-space MeanFlow formulation that takes the noisy motion $z _ { t }$ and the time pair $( t , r )$ as inputs and predicts clean motion $\hat { z } _ { 0 } .$ . During training, we construct the corresponding intermediate state $\hat { z } _ { r }$ by interpolating between $z _ { t }$ and ${ \hat { z } } _ { 0 } ,$ and use its displacement from $z _ { t }$ to parameterize the interval-averaged velocity for MeanFlow training. This formulation also enables direct supervision in 3D motion space: reconstruction and forward-kinematics losses encourage faithful motion recovery and kinematic consistency, while velocity and dynamics losses regularize temporal evolution. At inference, the predicted clean motion is converted into the corresponding interval-averaged velocity for few-step ODE sampling. In addition, FlashMotion provides flexible motion editing to support human-inthe-loop authoring and responsive streaming applications. Specifically, we apply an Adaptive Soft mask during sampling, which gradually relaxes the constraints over time and alleviates the boundary discontinuities caused by hard masks in the few-step setting.

Per-step efficiency via model architecture. We design a lightweight non-autoregressive BiMamba architecture with condition-guided state selection and AdaIN-based condition fusion. BiMamba is well suited to temporally coherent 3D human motion generation because its sequential inductive bias better captures local temporal dependencies than attention-only backbones, while retaining linear-time complexity ${ \mathcal { O } } ( n )$ compared with the $\mathcal { O } ( n ^ { 2 } )$ cost of Transformers [20]. Its bidirectional design further improves temporal coherence and condition-motion alignment, and the nonautoregressive formulation avoids the latency of sequential decoding while mitigating exposure bias in autoregressivebased methods [8], [21] and boundary artifacts in iterative inpainting-based methods [22], [23]. Condition-guided state selection regulates temporal information flow, while AdaIN modulates motion features through frame-wise scaling and shifting. Shared projections support variable sequence lengths without constructing pairwise cross-attention matrices. Together, these architectural choices reduce per-step computational and memory costs while supporting highquality long-horizon motion generation.

This work extends our previous work, FlowerDance [12], in 4 aspects: (1) Task Coverage: generalizing the method to conditional 3D human motion generation, including textto-motion, speech-to-gesture, and music-to-dance; (2) Generative Strategy: reformulating MeanFlow in 3D motion space through explicit clean-motion prediction; (3) Model Architecture: introducing condition-guided state selection and AdaIN-based condition fusion to enhance cross-modal alignment and long-sequence generalization; and (4) Experiments: expanding evaluations across tasks and targeted ablation studies, alongside application demonstrations.

In summary, our main contributions are as follows:

We present FlashMotion, a co-designed framework for high-quality long-horizon 3D motion generation under a strict inference budget. By jointly optimizing trajectory-level sampling efficiency and per-step inference efficiency, FlashMotion achieves state-ofthe-art (SOTA) performance in both quality and efficiency across text-to-motion, speech-to-gesture, and music-to-dance generation.

We propose a few-step 3D motion generation strategy comprising: (1) a 3D motion-space MeanFlow formulation with kinematic and dynamic consistency constraints, and (2) Adaptive Soft masking for coherent motion editing.

We design a lightweight non-autoregressive architecture for efficient long-horizon motion generation, comprising: (1) a BiMamba-based backbone, (2) a condition-guided state selection mechanism, and (3) AdaIN-based cross-modal alignment.

## 2 RELATED WORK

## 2.1 Flow Matching Generative Models

Flow Matching (FM) [17] learns continuous transport from noise to data, with substantial methodological advances [18], [19] and broad generative applications [24], [25], while its adoption in 3D human motion generation remains nascent [12], [26], [27].

Methodological Advances. FM enables simulation-free training through velocity regression [17]. Rectified Flow [18] promotes straighter transport trajectories through reflow, while optimal-transport conditional flow matching (OT-CFM) [28] uses minibatch transport couplings to simplify learned flows. To reduce sampling cost, Consistency Flow Matching [29] encourages velocity consistency, Flow Map Matching [30] learns mappings between pairs of generation times, and Shortcut Models [31] condition predictions on the desired step size. Complementing these approaches, GMFlow [32] models multimodal velocity distributions with Gaussian mixtures. MeanFlow [19] instead learns intervalaveraged velocities, directly targeting finite-interval updates for 1- or few-step generation.

Applications Across Domains. FM has demonstrated its scalability across diverse generative tasks. In image generation, Stable Diffusion 3 [24] combines rectified flow with large-scale multimodal Transformer training. Video applications include sparsely conditioned video prediction [33], audio-driven portrait animation with FLOAT [34], and multiscale video synthesis with Pyramid Flow Matching [35]. In audio, Voicebox [25] and Audiobox [36] employ FM for speech and audio generation, while Matcha-TTS [37] and VoiceFlow [38] explore efficient text-to-speech synthesis.

Few studies have explored FM for 3D human motion synthesis, yet ultra-few-step long-horizon generation remains challenging: (1) approximately 50 steps for Flow-Motion [11], Match-TTSG [27], and UMF’s prior generation stage [39], and (2) 10 steps for Motion Flow Matching [26], FloodDiffusion [40], and FlowerDance [12].

## 2.2 3D Motion Generation

3D human motion generation has made substantial progress across text-to-motion, speech-to-gesture, and music-todance tasks. We review existing methods from 2 complementary perspectives: generative strategy, which determines how motion is synthesized, and model architecture, which determines how temporal dependencies and conditioning signals are modeled.

Generative Strategy. Representative strategies include: (1) Direct Regression, which directly predicts motion from conditioning signals, including Language2Pose [3] for textto-motion, Gesticulator [41] for speech-to-gesture, and FACT [4] for music-to-dance; (2) Adversarial Learning, which encourages realistic motion through discriminator-based training, including the trimodal gesture model [6] for speech-to-gesture, and Choreography cGAN [42], Dancing to Music [43], and CoheDancers [5] for music-driven dance generation; (3) Autoregressive Generation, which synthesizes motion sequentially, including T2M-GPT [7] for text-tomotion, and Dance Revolution [44], Bailando [21], and MEGA [8] for music-to-dance, as well as CoDancers [45] for group choreography; (4) Diffusion, which models conditional motion distributions through iterative denoising [46], including MDM [9] and MLD [47] for text-to-motion, DiffGesture [10] and PersonaGesture [48] for speech-to-gesture, EDGE [22] and CustomDance [49] for dance generation and editing, and BiTDiff [50] for conducting motion; and (5) Flow Matching, which learns continuous transformations from noise to motion, including Motion Flow Matching [26] and FlowMotion [11] for text-driven motion, the unified speechgesture synthesis framework [27], and FlowerDance [12] for few-step music-to-dance generation.

Model Architecture. Representative temporal backbones include: (1) Recurrent Networks, which capture temporal dependencies through sequential hidden-state updates, as exemplified by ERD [13] and the residual RNN [14] for human motion prediction, and the recurrent decoder in Dance Revolution [44] for music-to-dance; (2) Transformers, which use attention for temporal and conditional modeling, including T2M-GPT [7] for text-to-motion, DiffGesture [10] and StreamTalk [51] for speech-to-gesture, and FACT [4] and EDGE [22] for music-to-dance; and (3) Mamba-Based Models, which exploit selective state-space modeling for efficient sequence processing, including Motion Mamba [52] for text-to-motion, MambaTalk [15] and MambaGesture [53] for speech-to-gesture, and Flower-Dance [12], TokenDance [16], and the hybrid Mamba-Transformer model MATHDance [54] for music-to-dance.

![](images/ad56ad0f9011851993cb96a48f3d2a9657a7c267f3350debcb943ea8723391ea.jpg)

![](images/4386e502c9e6e2fda18c15a6441a89eb11020f509ce1ad30b6f09a88bca5a25b.jpg)

![](images/36f49afd5d0255f282b01348cf158eaea603439a1342ada85bf7be43aa985151.jpg)  
Fig. 2. Overview of FlashMotion. (a) Inference flow with clean-motion prediction and MeanFlow updates. (b) Training model inputs and output: noisy motion, conditions, and timesteps are mapped to the clean-motion target. (c) Shared BiMamba/CoBiMamba block. Dashed purple paths denote optional conditioning, omitted in BiMamba and used in CoBiMamba for condition-guided state selection.

However, high-quality long-horizon generation under a strict inference budget remains challenging due to 2 coupled bottlenecks: (1) generative strategies that require many sampling steps for high-quality 3D motion generation and (2) model architectures with substantial per-step computational and memory costs for long sequences. These costs leave limited computational resources for other production stages, such as rigging, retargeting, and rendering, potentially degrading the user experience in interactive applications.

## 3 METHODOLOGY

## 3.1 Overview

Since condition encoding, motion retargeting, and rendering also consume the end-to-end latency budget, motion generation must remain highly efficient. For example, interactively creating and revising an extended dance or action sequence requires rapid generation of long, coherent motions so that users can promptly preview and refine the results. Ultrafast long-term 3D motion generation that preserves physical plausibility, conditional alignment, and motion expressiveness is therefore essential for responsive avatar animation.

Given a conditioning sequence $C = ( c _ { i } ) _ { i = 1 } ^ { N } ,$ our objective is to generate a corresponding 3D motion sequence $\begin{array} { r c l } { { M } } & { { = } } & { { ( m _ { i } ) _ { i = 1 } ^ { N } , } } \end{array}$ where $c _ { i } \ \in \ \mathbb { R } ^ { d _ { c } }$ and $\begin{array} { r l r } { m _ { i } } & { { } \in } & { \mathbb { R } ^ { \hat { d _ { m } } } } \end{array}$ denote the condition and motion features at frame i, respectively. We consider text, speech, and music conditioning, with a separate model trained for each task. (1) Text-to-Motion. For text description $q _ { k }$ and its frame interval $I _ { k } ,$ $c _ { i } = E _ { \mathrm { t e x t } } ( q _ { k } ) \in \mathbb { R } ^ { 4 0 9 6 }$ uses UMT5-XXL [55]; $m _ { i } \in \mathbb { R } ^ { 2 6 3 }$ follows HumanML3D [56] at 20 FPS. (2) Speech-to-Gesture. The condition $c _ { i }$ combines frame-aligned 1024-dimensional HuBERT speech features and 300-dimensional text features; $m _ { i } \in \mathbb { R } ^ { 3 3 3 }$ comprises local 6D rotations [57] for 55 SMPL-X joints and 3 root-motion components at 30 FPS. (3) Musicto-Dance. $c _ { i } ~ \in ~ \mathbb { R } ^ { 3 5 }$ contains frame-aligned Librosa music features $[ 5 8 ] ; m _ { i } \in \mathbb { R } ^ { 1 4 7 }$ comprises 3D root motion (horizontal displacements and height) and 24-joint 6D rotations [57] at 30 FPS.

To meet this strict inference budget in practice, we codesign FlashMotion across both the generative strategy and model architecture, as summarized in Fig. 2. (1) At the generative-strategy level, we formulate MeanFlow in 3D motion space by directly predicting clean motion and deriving the interval-averaged velocity for few-step sampling, with auxiliary kinematic and dynamic constraints to preserve motion fidelity. (2) At the model-architecture level, we develop a lightweight non-autoregressive BiMamba backbone with condition-guided state selection and AdaINbased condition fusion mechanism, enhancing cross-modal alignment and generalization to longer motion sequences. (3) To support real-time human-in-the-loop motion authoring and responsive streaming applications under few-step sampling, we further introduce an Adaptive Soft masking strategy that enables coherent motion editing without additional training.

## 3.2 Generative Strategy

MeanFlow. From a kinematic perspective, 3D human motion consists of continuous trajectories of root translation and joint rotations, coupled through skeletal structure and temporal dependencies [59], [60]. Generating coherent motion therefore requires jointly modeling spatial coordination and temporal evolution, motivating continuous refinement of the entire sequence from unstructured noise toward a condition-consistent trajectory. Flow Matching (FM) [17] learns a velocity field for an ordinary differential equation (ODE) transporting Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ to motion $x \sim p _ { \mathrm { d a t a } }$ . Training uses the linear conditional path $z _ { t } = \mathrm { F P } ( x , t )$ , with time $t \sim U ( 0 , 1 )$

$$
\mathrm { F P } ( x , t ) = ( 1 - t ) x + t \epsilon\tag{1}
$$

The conditional path velocity is $v _ { \mathrm { c o n d } } = \epsilon - x$ . We regress v<sub>θ</sub> against this sample-wise target; the optimal regressor estimates the marginal velocity field $v ( z _ { t } , \bar { t } ) = \mathbb { E } [ v _ { \mathrm { c o n d } } \mid z _ { t } , t ]$ :

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \big [ \| v _ { \theta } ( z _ { t } , t ) - v _ { \mathrm { c o n d } } \| ^ { 2 } \big ] .\tag{2}
$$

At inference, generation starts from an initial prior sample $z _ { 1 }$ and iteratively integrates the learned velocity field $v _ { \theta } ( z _ { t } , t )$ until reaching $z _ { \mathrm { 0 } }$ in the data distribution:

$$
\begin{array} { c } { { v _ { \theta } ( z _ { t } , t ) = \displaystyle \frac { d z _ { t } } { d t } , } } \\ { { z _ { 0 } = z _ { 1 } + \displaystyle \int _ { 1 } ^ { 0 } v _ { \theta } ( z _ { \tau } , \tau ) d \tau . } } \end{array}\tag{3}
$$

Even when conditional flows are designed to be straight, latent trajectories in the high-dimensional space are often highly curved. When using large integration intervals, Flow Matching [17], which models the instantaneous velocity field, cannot accurately capture these curved trajectories, leading to degraded generation quality. Thus, Mean-Flow [19] replaces instantaneous velocity $v ( z _ { t } , t )$ with the interval mean velocity $u ( z _ { t } , r , t )$ , leading to a training objective aligned with the actual ODE inference procedure:

$$
u ( z _ { t } , r , t ) = \frac { 1 } { t - r } \int _ { r } ^ { t } v ( z _ { \tau } , \tau ) d \tau ,\tag{4}
$$

where $( r , t )$ are 2 timesteps randomly sampled from a uniform distribution $U ( 0 , { \bar { 1 } } )$ with the constraint $t \ > \ r .$ Differentiating Eq. 4 with respect to t yields the MeanFlow identity:

$$
u ( z _ { t } , r , t ) = v ( z _ { t } , t ) - ( t - r ) \frac { d } { d t } u ( z _ { t } , r , t ) .\tag{5}
$$

A velocity estimator $u _ { \theta }$ is trained to satisfy the MeanFlow identity:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M F } } = \mathbb { E } \big [ \| u _ { \theta } ( z _ { t } , r , t ) - \mathrm { s g } ( u _ { \mathrm { t g t } } ) \| _ { 2 } ^ { 2 } \big ] , } \\ { \mathrm { w h e r e } \quad u _ { \mathrm { t g t } } = v _ { \mathrm { c o n d } } - ( t - r ) D _ { t } u _ { \theta } , } \\ { D _ { t } u _ { \theta } = \partial _ { z } u _ { \theta } v _ { \mathrm { c o n d } } + \partial _ { t } u _ { \theta } . \qquad } \end{array}\tag{6}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient. Notably, this reduces to the FM loss when $r = t$ . During inference, MeanFlow uses a constant interval-averaged velocity for each update. The general update and its equivalent integral form over the full interval are:

$$
z _ { r } = z _ { t } - ( t - r ) u ( z _ { t } , r , t ) ,\tag{7}
$$

Training. Although MeanFlow has shown promising results in image generation [19], video rendering [61], and audio synthesis [62], directly applying it to few-step 3D motion generation is non-trivial. A key reason is that these modalities often exhibit explicit local continuity in their native representations, such as spatially neighboring pixels, adjacent video frames, or temporally continuous waveforms, which provides a relatively direct structure for learning smooth generative fields. In contrast, SMPL-based 3D motion is represented as a sequence of pose parameters whose temporal continuity is explicit across frames, while the intra-frame dependencies among joints are only implicitly encoded by the human kinematic chain. Under large integration intervals, small errors in the predicted velocity field can therefore accumulate over time and disrupt these latent kinematic constraints, leading to temporal jitter, global drift, foot sliding, and implausible articulations. Moreover, the vanilla MeanFlow objective only constrains the averaged velocity field, without explicitly regularizing the recovered motion in kinematic or physical spaces.

To enable direct motion-level supervision, FlashMotion predicts the clean-motion estimate $\hat { z } _ { 0 }$ from the current state $z _ { t } ,$ frame-aligned conditioning sequence $C ,$ and time interval $( r , t )$ . The interval-averaged velocity is then derived as:

$$
u _ { \theta } ( z _ { t } , r , t ; C ) = \frac { z _ { t } - \hat { z } _ { 0 } } { t } .\tag{8}
$$

We optimize the induced velocity using the MeanFlow objective in Eq. (6), with JVP-based differentiation, $t ^ { 2 }$ time weighting, and adaptive loss weighting.

Empirically, training with ${ \mathcal { L } } _ { \mathrm { M F } }$ alone still remains unstable and fails to converge in our SMPL-based experiments, as corroborated by the results in Sec. 4.5. We therefore add reconstruction supervision to anchor the clean-motion estimate to ground truth:

$$
\mathcal { L } _ { \mathrm { r e c } } = \mathbb { E } \big [ \| \hat { z } _ { 0 } - z _ { 0 } \| _ { 2 } ^ { 2 } \big ] .\tag{9}
$$

To improve kinematic consistency, we match the recovered 3D joint positions to ground truth, encouraging spatially coordinated poses:

$$
\mathcal { L } _ { \mathrm { k i n } } = \mathbb { E } \big [ \| \hat { P } - P \| _ { 2 } ^ { 2 } \big ] .\tag{10}
$$

Here, $\hat { P } = \mathcal { K } ( \hat { z } _ { 0 } )$ and $P = \kappa ( z _ { 0 } )$ , where $\kappa$ recovers joint positions from 3D rotation representations.

To improve temporal dynamics, we match joint velocities and accelerations to ground truth, encouraging temporally coherent motion evolution:

$$
\mathcal { L } _ { \mathrm { d y n } } = \mathbb { E } \big [ \| \Delta \hat { P } - \Delta P \| _ { 2 } ^ { 2 } + \beta \| \Delta ^ { 2 } \hat { P } - \Delta ^ { 2 } P \| _ { 2 } ^ { 2 } \big ] .\tag{11}
$$

Here, $\Delta$ denotes frame-wise finite differences, and $\beta$ balances velocity and acceleration supervision.

Overall, the shared training objective is:

$$
{ \mathcal { L } } = \lambda _ { \mathrm { m f } } { \mathcal { L } } _ { \mathrm { M F } } + \lambda _ { \mathrm { r e c } } { \mathcal { L } } _ { \mathrm { r e c } } + \lambda _ { \mathrm { k i n } } { \mathcal { L } } _ { \mathrm { k i n } } + \lambda _ { \mathrm { d y n } } { \mathcal { L } } _ { \mathrm { d y n } } .\tag{12}
$$

Inference. Starting from Gaussian noise $\epsilon ,$ we progressively generate motion over uniformly spaced time intervals. At each step, we predict the clean motion zˆ<sub>0</sub> and obtain $z _ { r }$ by linearly interpolating between $\hat { z } _ { 0 }$ and the current noisy state $z _ { t } .$ . This update is equivalent to Eq. 7, without explicit velocity computation or additional noise injection.

Motion Editing. Extending the inpainting idea in diffusionbased motion generation [22], [23], FlashMotion supports training-free constraint-based editing with multiple sampling steps; 1-step editing is not supported. This capability serves 2 complementary uses. (1) Human-in-the-loop motion authoring. Users can fill temporal gaps or modify selected joint trajectories, such as interpolating between dance clips or adjusting hand paths (Fig. 6). (2) Responsive Streaming Applications. Temporal inpainting enables sequential window extension by constraining each new window with the previous window’s tail. This allows Flash-Motion to adapt to incoming conditions for conversational avatars or interactive NPCs. With few sampling steps, fixed hard masks may cause dynamic mismatches at constraint boundaries. We therefore apply an Adaptive Soft mask, gradually relaxing the influence of known regions to improve coupling between constrained and unconstrained motions. Given a joint-wise or temporal constraint $\boldsymbol { x } ^ { \prime } \in \mathbb { R } ^ { N \times d _ { m } }$ with positions indicated by a binary mask $M \in \{ 0 , 1 \} ^ { N \times d _ { m } }$ at time t, where N is the sequence length and $d _ { m } = 2 6 3 , 3 3 3$ and 147 for text-to-motion, speech-to-gesture, and music-todance, respectively, FlashMotion performs constraint-aware denoising as follows:

$$
z _ { t } : = ( t \cdot M ) \odot \mathrm { F P } ( x ^ { \prime } , t ) + ( 1 - ( t \cdot M ) ) \odot z _ { t } ,\tag{13}
$$

where ⊙ denotes the Hadamard product, and known regions follow the constraint flow.

## 3.3 Model Architecture

Overview. Given the flow-path state $z _ { t } ,$ conditioning sequence $C ,$ and interval $( r , t ) ,$ , FlashMotion predicts a cleanmotion estimate $\hat { z } _ { 0 } ,$ , which is converted into the intervalaveraged velocity for the ODE solver (Fig. 2). Condition features are projected into a shared hidden dimension and processed by stacked BiMamba blocks. For music and speech, genre and speaker embeddings, respectively, provide additional style guidance through gated modulation [81]. The current time and interval length $t , t - r$ are separately encoded using sinusoidal embeddings and MLPs, then combined by addition. The motion estimator comprises stacked blocks integrating condition-guided Bi-Mamba (CoBiMamba) for temporal modeling, FiLM [82] for time conditioning, and AdaIN-based condition fusion (CoFusion) [83] for channel-wise condition modulation. For long-term generation, FlashMotion jointly predicts all motion frames in a non-autoregressive pass at each sampling step. These components work synergistically to enable FlashMotion to generate high-quality motion efficiently.

Bidirectional Mamba. Our choice of Mamba is motivated by the following considerations: (1) Human motion requires strong local temporal continuity. Standard selfattention without positional information is permutationequivariant; positional encodings provide sequence-order information [20]. Owing to its inherent sequential inductive bias, Mamba [84] has demonstrated strong performance in modeling fine-grained local dependencies [15], [53]. (2) Mamba has only ${ \mathcal { O } } ( n )$ computational complexity compared to the $O ( n ^ { 2 } )$ complexity of attention-based Transformers [20]. (3) Mamba supports non-autoregressive generation, which not only further improves efficiency but also mitigates the exposure-bias problem in autoregressive methods [8], [21] and alleviates transition artifacts at segment boundaries in inpainting-based approaches [22], [23].

Furthermore, condition–motion dependencies benefit from bidirectional temporal context, but standard Mamba models sequences only in a single direction. To address this limitation, we introduce Bidirectional Mamba (BiMamba) and condition-guided BiMamba (CoBiMamba), which capture information from both temporal directions. As shown in Fig. 2, the sequence is processed by forward and backward Mamba branches; their outputs are fused by addition and refined through a multiplicative skip connection, enhancing gradient flow and feature preservation. This bidirectional design improves motion coherence and strengthens condition–motion alignment.

Condition-Guided State Selection. Mamba is built upon state space models (SSMs), which capture temporal dependencies through recurrent hidden-state updates. Specifically, its selective SSM (S6) uses input-dependent parameters to regulate information retention and propagation, while a hardware-aware parallel scan enables efficient computation [84]. Unlike S4 [85], whose learned SSM coefficients are invariant across sequence positions, Mamba makes $B _ { i } ,$ $C _ { i } ,$ and $\Delta _ { i }$ input-dependent through shared learned projections, while retaining a shared, input-independent learnable matrix A. The core SSM recurrence at frame i is:

$$
\begin{array} { l } { { h _ { i } = \bar { A } _ { i } h _ { i - 1 } + \bar { B } _ { i } x _ { i } , } } \\ { { y _ { i } = C _ { i } h _ { i } . } } \end{array}\tag{14}
$$

Here, i indexes motion frames, distinct from the generative time $t ;$ the block’s skip connection, gating, and output projection are omitted from this core formulation. Following the Mamba implementation, we use a diagonal A, an inputdependent discretization step $\Delta _ { i } ,$ and simplified input discretization:

$$
\begin{array} { l } { { \bar { A } _ { i } = \exp ( \Delta _ { i } A ) , } } \\ { { \bar { B } _ { i } = \Delta _ { i } B _ { i } . } } \end{array}\tag{15}
$$

These operations apply independently per channel, with $B _ { i }$ and $C _ { i }$ shared across channels. A hardware-aware parallel scan efficiently evaluates the recurrence, using projection weights and $\dot { A }$ shared across frames.

For condition encoding, BiMamba derives its selection coefficients from its own condition features. In the motion backbone, CoBiMamba jointly uses locally processed motion features ${ \tilde { X } } _ { i }$ and aligned, time-modulated condition features $G _ { i } { \mathrm { : } }$

$$
\begin{array} { r l } & { B _ { i } = f _ { B } ( { \tilde { X } } _ { i } ) + g _ { B } ( G _ { i } ) , \quad C _ { i } = f _ { C } ( { \tilde { X } } _ { i } ) + g _ { C } ( G _ { i } ) , } \\ & { \Delta _ { i } = \mathrm { s o f t p l u s } \big ( W _ { \Delta } [ f _ { \Delta } ( { \tilde { X } } _ { i } ) + g _ { \Delta } ( G _ { i } ) ] + b _ { \Delta } \big ) . } \end{array}\tag{16}
$$

Here, ${ \tilde { X } } _ { i }$ denotes motion features after input projection, local convolution, and activation. The learned projections $f$ and $g$ are shared across frames; $f _ { \Delta }$ and $g _ { \Delta }$ produce low-dimensional intermediate features expanded by the same $W _ { \Delta . }$ , with a learned bias $b _ { \Delta }$ . A remains conditionindependent. Conditioning thus regulates memory retention through $\Delta _ { i } ,$ , information injection through $B _ { i } ,$ and state readout through $C _ { i } ,$ enabling motion-state updates to follow changing action instructions or rhythmic cues.

Channel-Level Condition Fusion. To integrate text, speech, or music conditions efficiently, we use AdaIN-based residual modulation [83]. Frame-wise condition embeddings predict bounded scale and shift residuals for temporally normalized motion features, which are rescaled with their original statistics. Zero-initialized modulation preserves the identity mapping at initialization. This design offers 3 advantages: (1) Length-agnostic generation: shared frame-wise projections introduce no length-specific parameters. Combined with our non-autoregressive BiMamba backbone, this allows FlashMotion to directly generate full long sequences even when trained on short motion windows; (2) Efficient fusion: for fixed feature widths, frame-wise projections and modulation have $O ( N )$ computational complexity, avoiding the $O ( N ^ { 2 } )$ pairwise interactions of cross-attention in our frame-aligned setting; and (3) Expressive conditioning: compared with simple additive fusion, condition-dependent scaling and shifting provide both multiplicative and additive control, enabling amplification, suppression, and offset of motion-feature channels.

TABLE 1  
Quantitative comparison on the BABEL [63] dataset for long-sequence text-to-motion generation. Bold and underlined values denote the best and second-best generated results, excluding ground truth.
<table><tr><td>Method</td><td>Venue</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>MM↓</td><td>FID↓</td><td>∆Div↓</td><td>∆PJ↓</td><td>∆AUJ↓</td><td>FPS@1K↑</td><td>FPS@10K↑</td></tr><tr><td>Ground Truth</td><td>一</td><td>28.504</td><td>43.900</td><td>53.161</td><td>4.316</td><td>1</td><td>0.000</td><td>0.00000</td><td>0.00000</td><td>一</td><td></td></tr><tr><td>MDM [9]</td><td>ICLR&#x27;23</td><td>23.706</td><td>38.571</td><td>48.695</td><td>4.655</td><td>1.374</td><td>0.129</td><td>0.36381</td><td>1.05195</td><td>284</td><td>299</td></tr><tr><td>ST2M [64]</td><td>ICCV&#x27;23</td><td>25.793</td><td>39.681</td><td>48.900</td><td>4.415</td><td>1.450</td><td>0.589</td><td>0.25851</td><td>1.56450</td><td>2772</td><td>3417</td></tr><tr><td>DoubleTake [65]</td><td>ICLR&#x27;24</td><td>18.894</td><td>29.861</td><td>37.929</td><td>4.836</td><td>1.805</td><td>0.918</td><td>0.28694</td><td>1.00752</td><td>185</td><td>180</td></tr><tr><td>FlowMDM [66]</td><td>CVPR&#x27;24</td><td>19.483</td><td>30.783</td><td>39.033</td><td>4.847</td><td>0.712</td><td>0.173</td><td>0.07197</td><td>0.27793</td><td>24</td><td>22</td></tr><tr><td>MoMask [67]</td><td>CVPR&#x27;24</td><td>25.259</td><td>39.390</td><td>49.751</td><td>4.546</td><td>1.101</td><td>0.271</td><td>0.80482</td><td>1.79723</td><td>941</td><td>1014</td></tr><tr><td>MMM [68]</td><td>CVPR&#x27;24</td><td>26.936</td><td>39.533</td><td>47.709</td><td>4.612</td><td>1.823</td><td>0.034</td><td>0.91381</td><td>1.97710</td><td>1332</td><td>1350</td></tr><tr><td>BAMM [69]</td><td>ECCV&#x27;24</td><td>26.849</td><td>40.139</td><td>49.855</td><td>4.487</td><td>1.270</td><td>0.230</td><td>0.77382</td><td>1.75389</td><td>144</td><td>152</td></tr><tr><td>DART [70]</td><td>ICLR&#x27;25</td><td>26.105</td><td>39.924</td><td>48.485</td><td>4.424</td><td>1.241</td><td>0.261</td><td>0.05516</td><td>0.33090</td><td>75</td><td>75</td></tr><tr><td>MotionStreamer [71]</td><td>ICCV&#x27;25</td><td>23.676</td><td>37.492</td><td>46.842</td><td>4.762</td><td>2.216</td><td>0.023</td><td>0.01293</td><td>0.14735</td><td>10</td><td>10</td></tr><tr><td>FloodDiffusion [40]</td><td>CVPR&#x27;26</td><td>25.109</td><td>38.924</td><td>48.372</td><td>4.491</td><td>0.542</td><td>0.199</td><td>0.12539</td><td>1.18581</td><td>34</td><td>35</td></tr><tr><td>FlashMotion (1-step)</td><td>Ours</td><td>26.371</td><td>40.083</td><td>48.851</td><td>4.456</td><td>1.630</td><td>0.093</td><td>0.00663</td><td>0.21474</td><td>29767</td><td>173 172</td></tr><tr><td>FlashMotion (4-step)</td><td>Ours</td><td>29.134</td><td>42.829</td><td>51.683</td><td>4.340</td><td>0.907</td><td>0.210</td><td>0.00027</td><td>0.01295</td><td>10711</td><td>46303</td></tr></table>

TABLE 2

Quantitative comparison on FineDance [72].
<table><tr><td>Method</td><td>Venue</td><td>FIDk ↓</td><td>FID_g ↓ | DIVk ↑</td><td></td><td>DIVg ↑|</td><td>BAS↑ |</td><td>|FPS@10K↑</td></tr><tr><td>Ground Truth</td><td>=</td><td>1</td><td>=</td><td>9.94</td><td>7.54</td><td>0.201</td><td>=</td></tr><tr><td>FACT [4]</td><td>ICCV&#x27;21</td><td>113.38</td><td>97.05</td><td>3.36</td><td>6.37</td><td>0.183</td><td>27</td></tr><tr><td>MNET [73]</td><td>CVPR&#x27;22</td><td>104.71</td><td>90.31</td><td>3.12</td><td>6.14</td><td>0.186</td><td>2311</td></tr><tr><td>Bailando [21]</td><td>CVPR&#x27;22</td><td>82.81</td><td>28.17</td><td>7.74</td><td>6.25</td><td>0.202</td><td>673</td></tr><tr><td>EDGE [22]</td><td>CVPR&#x27;23</td><td>94.34</td><td>50.38</td><td>8.13</td><td>6.45</td><td>0.212</td><td>589</td></tr><tr><td>FineNet [72]</td><td>ICCV′23</td><td>65.15</td><td>23.81</td><td>5.84</td><td>5.19</td><td>0.219</td><td>1137</td></tr><tr><td>Lodge [74]</td><td>CVPR&#x27;24</td><td>50.00</td><td>35.52</td><td>5.67</td><td>4.96</td><td>0.226</td><td>884</td></tr><tr><td>MEGA [8]</td><td>NeurIPS&#x27;25</td><td>50.00</td><td>13.02</td><td>6.23</td><td>6.27</td><td>0.226</td><td>340</td></tr><tr><td>FlowerDance [12]</td><td>ECCV&#x27;26</td><td>29.73</td><td>19.59</td><td>8.42</td><td>7.18</td><td>0.232</td><td>4145</td></tr><tr><td>FlashMotion (1-step)</td><td>Ours</td><td>18.32</td><td>28.48</td><td>9.48</td><td>7.54</td><td>0.237</td><td>159 800</td></tr><tr><td>FlashMotion (4-step)</td><td>Ours</td><td>15.15</td><td>21.13</td><td>9.55</td><td>6.58</td><td>0.240</td><td>40545</td></tr></table>

## 4 EXPERIMENT

## 4.1 Experimental Setup

Text-to-Motion. (1) Dataset. BABEL [63] provides approximately 43 hours of motion covering over 250 action categories with temporally localized annotations; we resample motions to 20 FPS. Following prior work [66], [70], [86], we evaluate a Mixed set of 311 validation sequences (201-1030 frames) containing approximately 300 transitions between text prompts. (2) Evaluation. Following [70], we assess semantic alignment using R@1/2/3 and MM, distributional quality using FID, diversity using ∆Div, and transition continuity using ∆PJ and ∆AUJ.

Music-to-Dance. (1) Dataset. We use the publicly released 7.7-hour FineDance subset [72], covering 16 dance genres at 30 FPS. Following prior FineDance-based evaluations [8], [12], [16], we extract 84 overlapping clips from 19 test recordings. (2) Evaluation. Following prior work [8], [12], we assess distributional quality using $\mathrm { \bar { F I D } } _ { k }$ and $\mathrm { F I D } _ { g } ,$ diversity using $\mathrm { D I V } _ { k }$ and $\mathrm { D I V } _ { g } ^ { - } ,$ rhythmic alignment using BAS.

Speech-to-Gesture. (1) Dataset. We use BEAT2 [75] at 30 FPS, which contains 60 hours of synchronized speech and holistic motion from 25 speakers. Following [51], [79], we evaluate on the 265 test recordings in the official BEAT2 English split, using each recording’s full aligned speech–motion span. (2) Evaluation. Following [51], we assess distributional quality

TABLE 3  
Speech-to-gesture comparison on BEAT2 [75].
<table><tr><td rowspan=1 colspan=1>Method              Venue</td><td rowspan=1 colspan=1>FGD↓  BC↑ DIV↑</td><td rowspan=1 colspan=1>FPS@10K↑</td></tr><tr><td rowspan=2 colspan=1>Ground Truth        一CaMN [76]          ECCV&#x27;22</td><td rowspan=2 colspan=1>0.44458.37140.51200.20005.5800</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>7955</td></tr><tr><td rowspan=2 colspan=1>Audio2Photoreal [77] CVPR&#x27;24EMAGE [75]         CVPR&#x27;24</td><td rowspan=1 colspan=1>0.84900.32606.2400</td><td rowspan=1 colspan=1>115</td></tr><tr><td rowspan=1 colspan=1>0.69200.28406.0600</td><td rowspan=1 colspan=1>1839</td></tr><tr><td rowspan=1 colspan=1>RAG-Gesture [78]    CVPR&#x27;25</td><td rowspan=1 colspan=1>0.48700.51409.9400</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=2 colspan=1>SemTalk [79]         ICCV&#x27;25EchoMask [80]       MM&#x27;25</td><td rowspan=1 colspan=1>0.35600.51008.4090</td><td rowspan=1 colspan=1>358</td></tr><tr><td rowspan=1 colspan=1>0.56600.49509.2990</td><td rowspan=1 colspan=1>510</td></tr><tr><td rowspan=2 colspan=1>FlashMotion (1-step)OursFlashMotion (4-step) Ours</td><td rowspan=1 colspan=1>0.47560.48016.8688</td><td rowspan=1 colspan=1>135179</td></tr><tr><td rowspan=1 colspan=1>0.32420.60069.1950</td><td rowspan=1 colspan=1>34655</td></tr></table>

using FGD, speech-motion rhythmic alignment using BC, and motion diversity using DIV.

Efficiency Evaluation. Across all tasks, we benchmark methods on an otherwise idle NVIDIA RTX 3090 GPU with an Intel Xeon Gold 5218 CPU (2.30 GHz). For our measurements, we compute generation FPS at batch size 1 across 1k or 10k output sequence lengths from the mean runtime of 5 GPU-synchronized runs after 1 full warm-up run. Timing covers only generation between predefined input and output representations, excluding external modality encoding, retargeting, and IK.

## 4.2 Quantitative Evaluation

Generation Quality. Tables 1, 2, and 3 compare generation quality for text-to-motion, music-to-dance, and speech-togesture, respectively. For text-to-motion, FlashMotion leads R@1/2/3 and MM, while reducing ∆PJ and $\Delta \mathrm { A U J }$ by 97.9% and 91.2% relative to the strongest external baselines. These closer-to-ground-truth transition dynamics, improved semantic alignment, and competitive FID establish overall state-of-the-art performance in long-horizon action composition. For music-to-dance, FlashMotion reduces $\mathrm { F I D } _ { k }$ by 49.0% and improves $\mathrm { D I V } _ { k }$ by 13.4% over FlowerDance [12], with a more modest 3.4% gain in BAS. Its 1-step variant also matches ground-truth $\mathrm { D I V } _ { g } ,$ supporting overall stateof-the-art dance quality across motion fidelity, diversity, and

... waves to the viewer with their right hand....

... bends knees slightly..., then squats and stands ....

takes a big jump forward and lands steadily.

![](images/70584b3f42936571848ad306f7c156231077f5297a6d445400edc46d5de15356.jpg)  
Fig. 3. Text-to-motion comparison under a shared sequence of 3 prompts.

rhythmic alignment. For speech-to-gesture, FlashMotion achieves the lowest FGD and highest BC, improving over SemTalk [79] and RAG-Gesture [78] by 8.9% and 16.8%, respectively. Combined with diversity close to ground truth, these results yield an overall state-of-the-art balance of motion fidelity, synchronization, and diversity. Overall, Flash-Motion combines 3D motion-oriented MeanFlow with kinematic and dynamic regularization and a condition-guided BiMamba architecture with AdaIN fusion to achieve superior 3D motion generation quality. These advantages are strongest with 4-step inference, while even 1-step generation remains competitive across complex dance, expressive speech-driven gestures, and compositional text-driven actions, demonstrating the applicability of this design across conditioning modalities.

Generation Efficiency. The FPS columns in Tables 1, 2, and 3 report generation throughput. At 10<sup>4</sup> frames, 4-step FlashMotion achieves 13.6×, 9.8×, and 4.4× the throughput of the fastest external T2M, M2D, and S2G baselines (ST2M, FlowerDance, and CaMN), respectively, with even higher throughput in 1 step. Figure 1 visualizes the quality– throughput trade-off across all 3 tasks. On BABEL, 4-step throughput rises from 10,711 to 46,303 FPS as sequence length increases from 10<sup>3</sup> to 10<sup>4</sup> frames, consistent with efficient long-sequence processing by our non-autoregressive BiMamba backbone. This substantial efficiency lead benefits from 2 complementary designs: (1) 3D motion-oriented MeanFlow enables high-quality generation in a few steps; (2) the Mamba-based architecture naturally supports nonautoregressive long-sequence generation, while conditionguided state selection and AdaIN fusion further preserve motion quality, conditional alignment, and temporal continuity.

## 4.3 Qualitative Evaluation

We qualitatively compare FlashMotion with representative baselines across 3 motion generation tasks. (1) Text-to-Motion. Each method receives 3 consecutive text prompts, each specifying a 5-second action. As shown in Fig. 3, FlashMotion follows the changing instructions while preserving articulated movements and plausible poses. In contrast, FlowMDM [66] exhibits limited motion variation and unnatural joint configurations, MotionStreamer [71] continues similar gestures despite prompt changes, and FloodDiffusion [40] produces unnatural movements and abrupt transitions. (2) Speech-to-Gesture. Given the same 15-second speech clip, FlashMotion generates expressive gestures that follow speech rhythm while maintaining coordinated, physically plausible body movements (Fig. 5). RAG-Gesture [78] exhibits root drift, SemTalk [79] shows poor foot-ground contact, and EchoMask [80] produces unnatural foot motion and implausible root rotations. (3) Music-to-Dance. Given the same 15-second music clip, FlashMotion produces varied, rhythmically aligned choreography with coordinated whole-body articulation and plausible foot contact (Fig. 4). Lodge [74] repeats similar movement patterns, MEGA [8] exhibits poor foot– floor contact, and FlowerDance [12] occasionally produces physically implausible poses. Overall, these visual comparisons demonstrate FlashMotion’s consistent advantages in conditional alignment, motion expressiveness, and physical plausibility across all 3 tasks. These results benefit from our complementary designs: kinematic and dynamic supervision promote plausible and smooth movements, while condition-guided state selection and AdaIN fusion align motion with diverse input conditions.

... a steady groove, synthesized sounds... in a pop style....

![](images/e6c26b9d38a83901253d34eaaa5b4f0842be6d9f9376c182b2ebedd9908bdadc.jpg)  
Fig. 4. Music-to-dance comparison on the same music excerpt.

## 4.4 User Study

Motion’s inherent subjectivity makes user feedback essential for evaluating generated movements. To complement objective metrics, we conduct a user study across T2M, S2G, and M2D, with 40 conditions per task and 30 human reviewers. All reviewers provided informed consent before participating. Ratings use a 5-point scale (1: Terrible; 2: Poor; 3: Fair; 4: Good; 5: Great) and assess Motion Quality (MQ, plausibility and diversity), Motion Synchronization (MS, alignment with text, speech, or music), and Motion Continuity (MC, smooth inter-segment transitions and jitterfree motion within segments). Table 4 reports mean ratings, with FlashMotion scoring highest on every criterion across all 3 tasks. For T2M, it exceeds the strongest baseline for each criterion by 0.16, 0.24, and 0.01 points in MQ, MS, and MC, respectively, indicating stronger perceived motion quality and text alignment with comparable continuity. For S2G, the corresponding gains are 0.35, 0.21, and 0.14 points, showing the largest MQ improvement across tasks alongside better speech alignment and smoother gestures. For M2D, MQ, MS, and MC reach 3.98, 3.31, and 3.99, exceeding the strongest baselines by 0.01, 0.03, and 0.04 points, respectively; these smaller margins indicate a modest but consistent advantage across the criteria. These perceptual gains are consistent with our design: kinematic and dynamic supervision encourage plausible poses and smooth motion, while condition-guided state selection and AdaIN fusion promote cross-modal alignment, jointly improving motion quality, synchronization, and continuity. These results support FlashMotion’s advantage in human preferences for motion quality, conditional alignment, and temporal continuity. See the appendix for study details.

0 -……-… “... assigned to ... the elevator to be a guide... people passed ...”

![](images/d9af4f20d1e77deb5d73cdf9e683446abc8d7775bc23d3351969b0f63a901a32.jpg)  
Fig. 5. Speech-to-gesture comparison on the same speech excerpt.

## 4.5 Ablation Study

Additional qualitative results appear in the appendix and demo video. We conduct ablations on text-to-motion, whose broad motion coverage and established metrics enable comprehensive evaluation of components shared across tasks.

Generative Strategy. Table 5 compares MeanFlow with Flow Matching [18] and Diffusion [9] using 1, 2, and 4 sampling steps. All variants use the same clean 3D motion prediction target and model architecture. Increasing MeanFlow from 1 to 4 steps improves retrieval and FID, with closerto-ground-truth transition dynamics. At 4 steps, MeanFlow achieves the highest retrieval scores and lowest FID, reducing FID by 15.5% and 18.1% relative to Flow Matching and diffusion, respectively. Its jerk-curve discrepancy is also 91.9% lower than that of Flow Matching. These results demonstrate the effectiveness of our 3D motion-oriented MeanFlow strategy for high-quality few-step generation.

Training Objectives. Table 7 evaluates MeanFlow-only training, removal of kinematic or dynamic regularization, and the w/o. MeanFlow Loss variant under 4-step sampling. Following MeanFlow [19], the w/ Uncond variant drops conditioning with a probability of 10% during training to jointly learn conditional and unconditional generation. MeanFlow-only training diverges, highlighting the need for direct motion-level supervision. Training with 10% condition dropout (w/ Uncond) increases FID from 0.907 to 2.232 and lowers R@1 from 29.134 to 25.456, indicating reduced motion fidelity and conditional alignment. All variants use the same model architecture. Removing kinematic supervision increases FID from 0.907 to 0.983 and ∆AUJ from 0.01295 to 0.06503. Removing dynamic supervision lowers R@1 from 29.134 to 27.393 and increases ∆PJ from 0.00027 to 0.00097. The w/o. MeanFlow Loss variant also weakens motion fidelity and continuity. The full objective achieves the best R@1/2, FID, and ∆PJ, supporting the complementary benefits of generative learning, motion reconstruction, and kinematic and dynamic regularization. Model Backbone. Table 6 compares BiMamba, Mamba [84], and Transformer [20], including its Non-AR and Streamline modes. Non-AR directly generates long sequences after training on short windows, whereas Streamline extends motion through prefix-constrained windowed inpainting [22]. The Transformer Streamline variant uses positional encodings within each window, whereas the Transformer Non-AR variant omits them for direct length extrapolation. Mamba and BiMamba also omit positional embeddings. All variants use the same MeanFlow-based strategy with 4-step sampling. BiMamba achieves the best retrieval and smallest boundary gaps, improving R@1 over Mamba by 4.25 percentage points and reducing ∆AUJ by 98.0%. BiMamba also achieves lower FID and diversity gap than Mamba. Both Transformer variants have much larger boundary gaps than BiMamba. These results demonstrate BiMamba’s advantage in combining strong semantic alignment with closer-to-ground-truth transition dynamics, supporting its use as the backbone of FlashMotion.

TABLE 4  
User study ratings (mean ± SD; 1–5, higher is better).
<table><tr><td></td><td>Method</td><td>MQ↑</td><td>MS↑</td><td>MC↑</td></tr><tr><td rowspan="5">T2M</td><td>DART</td><td> $3 . 4 5 \pm 0 . 9 0$ </td><td> $3 . 1 2 \pm 0 . 9 8$ </td><td> $3 . 6 5 \pm 0 . 8 2$ </td></tr><tr><td>FloodDiffusion</td><td> $3 . 6 2 \pm 0 . 8 9$ </td><td> $3 . 3 5 \pm 0 . 9 9$ </td><td> $3 . 8 5 \pm 0 . 8 6$ </td></tr><tr><td>FlowMDM</td><td> $3 . 6 2 \pm 0 . 8 6$ </td><td> $3 . 3 6 \pm 0 . 9 5$ </td><td> $3 . 5 0 \pm 0 . 8 7$ </td></tr><tr><td>FlashMotion</td><td> ${ \bf 3 . 7 8 \pm 0 . 8 6 }$ </td><td> ${ \bf 3 . 6 0 \pm 0 . 9 3 }$ </td><td> ${ \bf 3 . 8 6 \pm 0 . 8 0 }$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">S2G</td><td>CaMN</td><td> $3 . 3 3 \pm 0 . 8 9$ </td><td> $3 . 0 2 \pm 1 . 0 0$ </td><td> $3 . 7 7 \pm 0 . 8 1$ </td></tr><tr><td>EMAGE</td><td> $3 . 4 2 \pm 0 . 9 2$ </td><td> $2 . 8 7 \pm 0 . 9 8$ </td><td> $3 . 9 8 \pm 0 . 8 4$ </td></tr><tr><td>RAG-Gesture</td><td> $3 . 5 5 \pm 0 . 8 7$ </td><td> $3 . 0 5 \pm 0 . 9 0$ </td><td> $3 . 8 8 \pm 0 . 8 1$ </td></tr><tr><td>FlashMotion</td><td> ${ \bf 3 . 9 0 \pm 0 . 8 3 }$ </td><td> ${ \bf 3 . 2 6 \pm 0 . 9 9 }$ </td><td> ${ \bf 4 . 1 2 \pm 0 . 7 8 }$ </td></tr><tr><td rowspan="4">M2D</td><td>FlowerDance</td><td> $3 . 9 2 \pm 0 . 8 8$ </td><td> $3 . 2 7 \pm 0 . 9 8$ </td><td> $3 . 9 5 \pm 0 . 8 3$ </td></tr><tr><td>MEGA</td><td> $3 . 9 3 \pm 0 . 8 5$ </td><td> $3 . 2 2 \pm 0 . 9 6$ </td><td> $3 . 9 3 \pm 0 . 8 0$ </td></tr><tr><td>FineNet</td><td> $3 . 9 7 \pm 0 . 8 8$ </td><td> $3 . 2 8 \pm 0 . 9 7$ </td><td> $3 . 9 1 \pm 0 . 8 3$ </td></tr><tr><td>FlashMotion</td><td> ${ \bf 3 . 9 8 \pm 0 . 8 6 }$ </td><td> ${ \bf 3 . 3 1 \pm 0 . 9 8 }$ </td><td> ${ \bf 3 . 9 9 \pm 0 . 8 0 }$ </td></tr></table>

TABLE 5

Generative strategy ablation on BABEL. Bold denotes the best result at each sampling budget.
<table><tr><td></td><td>S</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>MM↓</td><td>FID↓</td><td>∆Div↓</td><td>∆PJ↓</td><td>∆AUJ↓</td></tr><tr><td rowspan="3">Diffusion</td><td></td><td>24.622</td><td>36.510</td><td>44.539</td><td>4.482</td><td>2.166</td><td>0.646</td><td>0.00095</td><td>0.09766</td></tr><tr><td>2</td><td>25.021</td><td>37.398</td><td>45.858</td><td>4.475</td><td>1.682</td><td>0.527</td><td>0.01483</td><td>0.24482</td></tr><tr><td>4</td><td>25.954</td><td>38.570</td><td>46.985</td><td>4.433</td><td>1.107</td><td>0.471</td><td>0.01270</td><td>0.20795</td></tr><tr><td rowspan="3">Flow Matching</td><td>1</td><td>25.628</td><td>38.228</td><td>46.632</td><td>4.388</td><td>2.009</td><td>0.463</td><td>0.00024</td><td>0.09359</td></tr><tr><td>2</td><td>26.394</td><td>39.476</td><td>47.931</td><td></td><td>4.4211.302</td><td></td><td>0.3020.01168</td><td>0.18384</td></tr><tr><td>4</td><td>27.460</td><td>40.683</td><td>49.149</td><td>4.379</td><td>1.074</td><td></td><td>0.2480.00995</td><td>0.16019</td></tr><tr><td rowspan="3">MeanFlow</td><td>1</td><td>26.371</td><td>40.083</td><td>48.851</td><td></td><td>4.4561.630</td><td>0.093</td><td>0.00663</td><td>0.21474</td></tr><tr><td>2</td><td>28.388</td><td>41.606</td><td>50.527</td><td></td><td>4.4381.192</td><td>0.004 0.00539</td><td></td><td>0.09123</td></tr><tr><td>4</td><td>29.134</td><td>42.829</td><td>51.683</td><td></td><td>4.3400.907</td><td></td><td>0.2100.00027</td><td>0.01295</td></tr></table>

TABLE 6

Motion backbone ablation on BABEL (4-step).
<table><tr><td>Backbone</td><td>Inference</td><td>R@1↑ R@2↑ R@3↑ MM↓ FID↓</td><td>∆Div↓ ∆PJ↓ ∆AUJ ↓</td></tr><tr><td>Transformer</td><td>Non-AR</td><td>24.805 37.09345.400 4.427 1.937</td><td>0.609 1.01509 10.76431</td></tr><tr><td></td><td>Streamline</td><td>25.992 38.953 47.588 4.3411.666</td><td>0.6251.03962 11.80452</td></tr><tr><td>Mamba</td><td>Non-AR</td><td>24.881 38.000 46.5834.538 0.917</td><td>0.232 0.03950 0.63527</td></tr><tr><td>BiMamba</td><td>Non-AR</td><td>29.134 42.82951.6834.3400.907</td><td>0.210 0.00027 0.01295</td></tr></table>

Model Components. Table 8 compares additive fusion, AdaIN [83], condition-guided state selection (Scan), and their combinations under 4-step sampling. Replacing addition with AdaIN improves retrieval and FID, with closer-to-ground-truth transition dynamics. Combining

TABLE 7  
Training objective ablation on BABEL (4-step).
<table><tr><td>Loss setting</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑ MM↓</td><td></td><td>FID↓</td><td>∆Div↓</td><td>∆J↓</td><td>∆AUJ↓</td></tr><tr><td>w/o Dynamic</td><td>27.393</td><td>41.426</td><td>650.905</td><td></td><td>4.3240.928</td><td>0.360</td><td>0.00097</td><td>0.01241</td></tr><tr><td>w/o Kinematic</td><td>28.613</td><td>42.53051.813</td><td></td><td></td><td>4.3860.983</td><td>0.090</td><td>0.00201</td><td>0.06503</td></tr><tr><td>MeanFlow only</td><td>NaN</td><td>NaN</td><td>NaN</td><td></td><td>NaN NaN</td><td>NaN</td><td>NaN</td><td>NaN</td></tr><tr><td>w/o MeanFlow</td><td>27.46040.683 49.149</td><td></td><td></td><td>4.3791.074</td><td></td><td></td><td>0.2480.00995</td><td>0.16019</td></tr><tr><td>w/ Uncond</td><td></td><td>25.456 38.893 47.860</td><td></td><td>4.5162.232</td><td></td><td></td><td>0.311 0.00241</td><td>0.06472</td></tr><tr><td>Full Model</td><td>29.134</td><td>42.829</td><td>51.683</td><td>4.340</td><td>0.907</td><td>0.210</td><td>0.00027</td><td>0.01295</td></tr></table>

TABLE 8

Condition fusion and state selection ablation on BABEL (4-step).
<table><tr><td>Component</td><td>R@1↑ R@2↑</td><td>R@3↑</td><td>MM↓</td><td>FID↓</td><td>∆Div↓</td><td>∆PJ↓ ∆AUJ↓</td></tr><tr><td>Add only</td><td>27.208</td><td>40.909 50.120</td><td>4.307</td><td>1.073</td><td>0.262</td><td>0.00105 0.06135</td></tr><tr><td>AdaIN only</td><td>28.220</td><td>42.575 51.215</td><td>4.342</td><td>0.973</td><td>0.462 0.00101</td><td>0.01925</td></tr><tr><td>Scan only</td><td>27.849</td><td>42.472 51.349</td><td>4.444</td><td>1.616</td><td>0.027 0.00680</td><td>0.10705</td></tr><tr><td>Add + Scan</td><td>26.090</td><td>40.318 49.643</td><td>4.380</td><td>1.283</td><td>0.483 0.00152</td><td>0.06991</td></tr><tr><td>AdaIN + Scan</td><td>29.134</td><td>42.829 51.683</td><td>4.340</td><td>0.907</td><td>0.210 0.00027</td><td>0.01295</td></tr></table>

TABLE 9

Timestep sampling strategy ablation on BABEL (4-step).
<table><tr><td>Distribution</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>MM↓</td><td>FID↓</td><td>∆Div↓</td><td>∆J↓</td><td>∆AUJ↓</td></tr><tr><td>Beta(0.5,0.5)</td><td>27.908</td><td>41.656</td><td>51.041</td><td>4.350</td><td>0.731</td><td>0.173</td><td>0.00216</td><td>0.03406</td></tr><tr><td>Beta(2.0, 2.0)</td><td>24.421</td><td>36.698</td><td>45.243</td><td>4.535</td><td>2.359</td><td>0.363</td><td>0.00844</td><td>0.13800</td></tr><tr><td>Beta(1.0, 2.0)</td><td>11.656</td><td>20.542</td><td>28.004</td><td>5.300</td><td>26.028</td><td>4.388</td><td>0.00413</td><td>0.08032</td></tr><tr><td>Beta(2.0, 1.0)</td><td></td><td>24.221 36.962</td><td>46.106</td><td>4.415</td><td>2.685</td><td>0.846</td><td>0.01100</td><td>0.18491</td></tr><tr><td>Uniform</td><td></td><td>29.13442.829</td><td>51.683</td><td>4.340</td><td>0.907</td><td></td><td>0.2100.00027</td><td>0.01295</td></tr></table>

AdaIN with Scan yields the highest retrieval scores, lowest FID, and smallest boundary gaps. Relative to AdaIN alone, adding Scan reduces FID by 6.8% and ∆PJ and ∆AUJ by 73.3% and 32.7%, respectively. Scan alone achieves a smaller diversity gap but weaker fidelity and continuity. These results show complementary conditioning: AdaIN modulates feature channels, while condition-guided scanning regulates state updates, improving semantic control and transition coherence.

Timestep Sampling Strategy. Timestep sampling is an important design choice in generative models. We compare Uniform sampling with 4 Beta alternatives emphasizing intermediate timesteps Beta(2.0, 2.0), endpoints Beta(0.5, 0.5), high-noise timesteps Beta(2.0, 1.0), and low-noise timesteps Beta(1.0, 2.0). Each distribution is used for both training time-pair sampling and 4-step inference. Table 9 shows that Uniform achieves the overall best results across reported metrics, supporting its use as our default. This advantage likely stems from balancing structural recovery at high noise with fine temporal refinement at low noise, addressing the coupled spatial and temporal demands of 3D motion generation.

## 4.6 Motion Editing

Motion editing enables fine-grained control in interactive 3D motion generation when text or audio alone cannot specify local motion details. Users can demonstrate desired movements, reconstruct them through human mesh recovery [87], and use the recovered motion as constraints for completion or refinement. (1) Qualitative Analysis. We showcase music-to-dance, where user-performed dance fragments provide intuitive local constraints. As shown in Fig. 6, our Adaptive Soft mask generates smooth in-between motions from specified start and end clips. Hard masking still produces reasonable motions within the gap but introduces abrupt boundary discontinuities that disrupt dance fluency. (2) Quantitative Analysis. We evaluate 30 FineDance windows from 19 test recordings using 3 matched seeds, 120-frame gaps, and 4-step sampling. Seam measures mean boundary velocity innovation, and Jerk measures mean interior 3rd-derivative magnitude (both lower is better); Activity is the generated-to-GT ratio of root-relative joint RMS speed (target 1). We compare naive binary Hard [9], [22], constantweight Fixed Soft [65] (w = 0.5), and our Adaptive Soft. Table 10 shows that Adaptive Soft achieves the lowest mean Seam and Jerk, reducing them by 84.9% and 4.2% relative to Hard while retaining comparable activity, demonstrating improved motion fluidity while maintaining fidelity. These results demonstrate FlashMotion’s superior boundary continuity over competing methods, making it better suited to a broader range of applications, including windowed streaming generation and human-in-the-loop motion authoring.

(a) Robot Motion Generation  
![](images/325c0c1f678ff6da87f2479a637c81a35a26aa8376288cb589197d581d7f9722.jpg)  
Fig. 6. Given start (yellow) and end (red) motions, FlashMotion generates smooth in-between motions with the Adaptive Soft mask (green), whereas a hard mask (blue) causes teleportation artifacts. For clarity, we showcase music-to-dance as a representative task.

TABLE 10  
Quantitative evaluation of inpainting transition quality.
<table><tr><td>Mask</td><td> $\mathsf { S e a m \downarrow } \left( \mathrm { m } / \mathrm { s } \right)$ </td><td> $\mathrm { J e r k \downarrow } ( \mathrm { m } / \mathrm { s } ^ { 3 } )$ </td><td>Activity (→ 1)</td></tr><tr><td>GT</td><td> $0 . 2 2 2 \pm 0 . 1 5 4$ </td><td> $1 5 5 . 0 4 \pm 1 0 9 . 6 6$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Hard [9], [22]</td><td> $2 . 0 6 6 \pm 0 . 8 3 4$ </td><td> $2 5 4 . 7 0 \pm 5 4 . 9 7$ </td><td> $1 . 3 3 7 \pm 0 . 6 1 9$ </td></tr><tr><td>Fixed Soft [65]</td><td> $1 . 6 5 7 \pm 0 . 8 6 2$ </td><td> $2 5 1 . 2 9 \pm 8 0 . 9 4$ </td><td> $1 . 3 2 9 \pm 0 . 6 1 9$ </td></tr><tr><td>Adaptive Soft</td><td> $\mathbf { 0 . 3 1 2 \pm 0 . 2 7 1 }$ </td><td> $\mathbf { 2 4 3 . 8 8 \pm 9 5 . 3 8 }$ </td><td> $1 . 3 2 7 \pm 0 . 6 1 9$ </td></tr></table>

![](images/4e88bf6ebc30090881e754f66e5fd406f20b5030ceb7a738c7366fa5442cf79c.jpg)

![](images/bc22ce1505aff83c847d121ce0b790317e0a00c2cf420ba8b17e5ad92be3bfee.jpg)  
(b) XR Character Animation

![](images/0af59c5dd26737d56fd80e1feb251c8997817b79715b610f9e9d7feaf76dfb1c.jpg)  
(c) Mobile Motion Interaction  
Fig. 7. With lightweight post-processing, FlashMotion supports various downstream applications, including (a) robot motion generation, (b) XR character animation, and (c) mobile motion interaction. For clarity, we showcase text-to-motion as a representative task.

## 4.7 Downstream Applications

With lightweight post-processing, FlashMotion supports robot motion generation, XR character animation, and mobile motion interaction. We showcase text-to-motion as a representative task (Fig. 7). (1) Robot Motion Generation. Using a GPT-6 Astra-assisted implementation, we retarget generated human motion to a G1 humanoid and track it with a pretrained controller in MuJoCo. The boxingto-kick example demonstrates the transfer of languageconditioned, coordinated whole-body motion to physical simulation. This workflow can reduce manual trajectory authoring and facilitate robot behavior prototyping. (2) XR Character Animation. We map generated motion to virtual characters through skeletal retargeting. The zombie-walking example demonstrates semantic control of character behavior in an immersive XR environment. As character animation is central to embodied interaction, this workflow can lower motion-authoring costs and support scalable interactive content creation. (3) Mobile Motion Interaction. We convert spoken requests into text conditions and retarget the generated motion to a mobile avatar. With cached text embeddings, quantization, and optimized operators, Flash-Motion generates 200 motion frames using 4-step sampling in approximately 400 ms on iPhone 17, enabling responsive interaction while reducing reliance on cloud inference.

## 5 CONCLUSION

In conclusion, we present FlashMotion, a co-designed framework for high-quality long-horizon 3D motion generation under a strict inference budget. For generative strategy, FlashMotion proposes a 3D motion-space Mean-Flow formulation with kinematic and dynamic consistency constraints, and an Adaptive Soft masking mechanism for temporally coherent motion editing. For model architecture, FlashMotion designs a lightweight non-autoregressive Bi-Mamba architecture with condition-guided state selection and AdaIN-based condition fusion, enhancing cross-modal alignment and generalization to longer motion sequences. Experiments demonstrate that FlashMotion achieves overall state-of-the-art (SOTA) performance in both quality and efficiency across text-to-motion, speech-to-gesture, and music-to-dance tasks.

Potential directions for future research include: (1) Fine-Grained Motion Generation. Modeling facial expressions and finer finger articulation could enable more detailed and expressive human animation. (2) Robust Generation at Scale. Reducing motion artifacts in larger-scale training and longer-sequence generation could improve reliability for production use. (3) Efficient Real-World Deployment. Hardware-aware optimization and end-to-end pipeline integration could facilitate efficient deployment and broader adoption in robotics, XR, and mobile interaction.

## REFERENCES

[1] K. Yang, J. Zhu, X. Tang, Z. Peng, X. Zhang, C. Chen, P. Wang, J. Wu, X. Chu, H. Liu et al., “Omnidance: Multimodal driven dance video generation with large-scale internet data,” in European Conference on Computer Vision. Springer, 2026, pp. 571–588.

[2] K. Yang, X. Zhou, X. Tang, R. Diao, H. Liu, J. He, and Z. Fan, “Beatdance: A beat-based model-agnostic contrastive learning framework for music-dance retrieval,” in Proceedings of the 2024 International Conference on Multimedia Retrieval, 2024, pp. 11–19.

[3] C. Ahuja and L.-P. Morency, “Language2pose: Natural language grounded pose forecasting,” in 2019 International conference on 3D vision (3DV). IEEE, 2019, pp. 719–728.

[4] R. Li, S. Yang, D. A. Ross, and A. Kanazawa, “Ai choreographer: Music conditioned 3d dance generation with aist++,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). Ieee, 2021, pp. 13 381–13 392.

[5] K. Yang, X. Tang, H. Wu, B. Qin, H. Liu, J. He, and Z. Fan, “Cohedancers: Enhancing interactive group dance generation through music-driven coherence decomposition,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 6663–6671.

[6] Y. Yoon, B. Cha, J.-H. Lee, M. Jang, J. Lee, J. Kim, and G. Lee, “Speech gesture generation from the trimodal context of text, audio, and speaker identity,” ACM Transactions on Graphics (TOG), vol. 39, no. 6, pp. 1–16, 2020.

[7] J. Zhang, Y. Zhang, X. Cun, Y. Zhang, H. Zhao, H. Lu, X. Shen, and Y. Shan, “Generating human motion from textual descriptions with discrete representations,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 14 730– 14 740.

[8] K. Yang, X. Tang, Z. Peng, Y. Hu, J. He, and H. Liu, “Megadance: Mixture-of-experts architecture for genre-aware 3d dance generation,” Advances in Neural Information Processing Systems, vol. 38, pp. 10 947–10 969, 2025.

[9] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-Or, and A. H. Bermano, “Human motion diffusion model,” arXiv preprint arXiv:2209.14916, 2022.

[10] L. Zhu, X. Liu, X. Liu, R. Qian, Z. Liu, and L. Yu, “Taming diffusion models for audio-driven co-speech gesture generation,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 10 544–10 553.

[11] M. C. Cuba, V. do Carmo Mel´ıcio, and J. P. Gois, “Flowmotion: Target-predictive conditional flow matching for jitter-reduced text-driven human motion generation,” Computers & Graphics, p. 104374, 2025.

[12] K. Yang, X. Tang, Z. Peng, X. Zhang, C. Chen, X. Zhou, P. Wang, H. Liu, and J. He, “Flowerdance: Meanflow for efficient and refined 3d dance generation,” in European Conference on Computer Vision. Springer, 2026, pp. 56–74.

[13] K. Fragkiadaki, S. Levine, P. Felsen, and J. Malik, “Recurrent network models for human dynamics,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 4346–4354.

[14] J. Martinez, M. J. Black, and J. Romero, “On human motion prediction using recurrent neural networks,” in 2017 IEEE Conference on computer vision and pattern recognition (CVPR). IEEE, 2017, pp. 4674–4683.

[15] Z. Xu, Y. Lin, H. Han, S. Yang, R. Li, Y. Zhang, and X. Li, “Mambatalk: Efficient holistic gesture synthesis with selective state space models,” Advances in Neural Information Processing Systems, vol. 37, pp. 20 055–20 080, 2024.

[16] Z. Yang, K. Yang, and X. Tang, “Tokendance: Token-to-token music-to-dance generation with bidirectional mamba,” arXiv preprint arXiv:2603.27314, 2026.

[17] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[18] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” arXiv preprint arXiv:2209.03003, 2022.

[19] Z. Geng, M. Deng, X. Bai, Z. Kolter, and K. He, “Mean flows for one-step generative modeling,” Advances in Neural Information Processing Systems, vol. 38, pp. 75 460–75 482, 2026.

[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[21] L. Siyao, W. Yu, T. Gu, C. Lin, Q. Wang, C. Qian, C. C. Loy, and Z. Liu, “Bailando: 3d dance generation by actor-critic gpt with choreographic memory,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11 050–11 059.

[22] J. Tseng, R. Castellon, and K. Liu, “Edge: Editable dance generation from music,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 448–458.

[23] X. Liu, X. Dong, S. Qian, D. Kanojia, W. Wang, and Z. Feng, “Gcdance: Genre-controlled music-driven 3d full body dance generation,” IEEE Transactions on Multimedia, 2026.

[24] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini,¨ Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first international conference on machine learning, 2024.

[25] M. Le, A. Vyas, B. Shi, B. Karrer, L. Sari, R. Moritz, M. Williamson, V. Manohar, Y. Adi, J. Mahadeokar et al., “Voicebox: Text-guided multilingual universal speech generation at scale,” Advances in neural information processing systems, vol. 36, pp. 14 005–14 034, 2023.

[26] V. T. Hu, W. Yin, P. Ma, Y. Chen, B. Fernando, Y. M. Asano, E. Gavves, P. Mettes, B. Ommer, and C. G. Snoek, “Motion flow matching for human motion synthesis and editing,” arXiv preprint arXiv:2312.08895, 2023.

[27] S. Mehta, R. Tu, S. Alexanderson, J. Beskow, E. Sz<sup>´</sup> ekely, and´ G. E. Henter, “Unified speech and gesture synthesis using flow matching,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 8220–8224.

[28] A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio, “Improving and generalizing flowbased generative models with minibatch optimal transport,” arXiv preprint arXiv:2302.00482, 2023.

[29] L. Yang, Z. Zhang, Z. Zhang, X. Liu, M. Xu, W. Zhang, C. Meng, S. Ermon, and B. Cui, “Consistency flow matching: Defining straight flows with velocity consistency,” arXiv preprint arXiv:2407.02398, 2024.

[30] N. M. Boffi, M. S. Albergo, and E. Vanden-Eijnden, “Flow map matching with stochastic interpolants: A mathematical framework for consistency models,” arXiv preprint arXiv:2406.07507, 2024.

[31] K. Frans, D. Hafner, S. Levine, and P. Abbeel, “One step diffusion via shortcut models,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 34 668–34 684.

[32] H. Chen, K. Zhang, H. Tan, Z. Xu, F. Luan, L. Guibas, G. Wetzstein, and S. Bi, “Gaussian mixture flow matching models,” arXiv preprint arXiv:2504.05304, 2025.

[33] A. Davtyan, S. Sameni, and P. Favaro, “Efficient video prediction via sparsely conditioned flow matching,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 23 206–23 217.

[34] T. Ki, D. Min, and G. Chae, “Float: Generative motion latent flow matching for audio-driven talking portrait,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 14 699–14 710.

[35] Y. Jin, Z. Sun, N. Li, K. Xu, H. Jiang, N. Zhuang, Q. Huang, Y. Song, Y. Mu, and Z. Lin, “Pyramidal flow matching for efficient video generative modeling,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 23 378–23 402.

[36] A. Vyas, B. Shi, M. Le, A. Tjandra, Y.-C. Wu, B. Guo, J. Zhang, X. Zhang, R. Adkins, W. Ngan et al., “Audiobox: Unified audio generation with natural language prompts,” arXiv preprint arXiv:2312.15821, 2023.

[37] S. Mehta, R. Tu, J. Beskow, E. Sz<sup>´</sup> ekely, and G. E. Henter, “Matcha-´ tts: A fast tts architecture with conditional flow matching,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 11 341–11 345.

[38] Y. Guo, C. Du, Z. Ma, X. Chen, and K. Yu, “Voiceflow: Efficient text-to-speech with rectified flow matching,” in ICASSP 2024- 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 11 121–11 125.

[39] G. Huang and O. Celiktutan, “Unified number-free text-to-motion generation via flow matching,” arXiv preprint arXiv:2603.27040, 2026.

[40] Y. Cai, Y. Wu, K. Li, Y. Zhou, B. Zheng, and H. Liu, “Flooddiffusion: Tailored diffusion forcing for streaming motion generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 2295–2304.

[41] T. Kucherenko, P. Jonell, S. Van Waveren, G. E. Henter, S. Alexandersson, I. Leite, and H. Kjellstrom, “Gesticulator: A framework¨ for semantically-aware speech-driven gesture generation,” in Proceedings of the 2020 international conference on multimodal interaction, 2020, pp. 242–250.

[42] Y.-F. Huang and W.-D. Liu, “Choreography cgan: generating dances with music beats using conditional generative adversarial networks,” Neural Computing and Applications, vol. 33, no. 16, pp. 9817–9833, 2021.

[43] H.-Y. Lee, X. Yang, M.-Y. Liu, T.-C. Wang, Y.-D. Lu, M.-H. Yang, and J. Kautz, “Dancing to music,” Advances in neural information processing systems, vol. 32, 2019.

[44] R. Huang, H. Hu, W. Wu, K. Sawada, M. Zhang, and D. Jiang, “Dance revolution: Long-term dance generation with music via curriculum learning,” in International conference on learning representations, 2020.

[45] K. Yang, X. Tang, R. Diao, H. Liu, J. He, and Z. Fan, “Codancers: Music-driven coherent group dance generation with choreographic unit,” in Proceedings of the 2024 International Conference on Multimedia Retrieval, 2024, pp. 675–683.

[46] K. Yang, J. Zhu, X. Tang, Z. Peng, X. Zhang, P. Wang, J. Wu, X. Chu, H. Liu, and J. He, “Mace-dance: Motion-appearance cascaded experts for music-driven dance video generation,” in Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 2026, pp. 1–12.

[47] X. Chen, B. Jiang, W. Liu, Z. Huang, B. Fu, T. Chen, and G. Yu, “Executing your commands via motion diffusion in latent space,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 18 000–18 010.

[48] X. Zhang, Y. Cai, K. Li, K. Yang, Y. Zhou, Z. Li, X. Chu, J. Zhang, and H. Liu, “Personagesture: Single-reference cospeech gesture personalization for unseen speakers,” arXiv preprint arXiv:2605.06064v1, 2026, version 1. [Online]. Available: https://arxiv.org/abs/2605.06064v1

[49] X. Tang, K. Yang, X. Guo, B. Prabhakaran, and R. Alghofaili, “Customdance: Customized 3d dance generation with coarse-to-fine human-centered interactive control,” arXiv preprint arXiv:2608.06722, 2026.

[50] T. Jia, K. Yang, X. Yang, X. Tang, K. Qiu, S. Wei, and Y. Zhao, “Bitdiff: Fine-grained 3d conducting motion generation via bimambatransformer diffusion,” arXiv preprint arXiv:2604.04395, 2026.

[51] J. Li, X. Zhang, J. Zhang, K. Yang, and S. Hoi, “Streamtalk: Streaming co-speech gesture generation with key-pose anchoring,” in European Conference on Computer Vision. Springer, 2026, pp. 581– 600.

[52] Z. Zhang, A. Liu, I. Reid, R. Hartley, B. Zhuang, and H. Tang, “Motion mamba: Efficient and long sequence motion generation,” in European Conference on Computer Vision. Springer, 2024, pp. 265–282.

[53] C. Fu, Y. Wang, J. Zhang, Z. Jiang, X. Mao, J. Wu, W. Cao, C. Wang, Y. Ge, and Y. Liu, “Mambagesture: Enhancing co-speech gesture generation with mamba and disentangled multi-modality fusion,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 10 794–10 803.

[54] K. Yang, X. Tang, Z. Peng, Y. Hu, X. Zhang, P. Wang, H. Liu, J. He, and Z. Fan, “Mathdance: Mamba-transformer architecture with uniform tokenization for high-quality 3d dance generation,” arXiv preprint arXiv:2505.14222, 2025.

[55] H. W. Chung, N. Constant, X. Garcia, A. Roberts, Y. Tay, S. Narang, and O. Firat, “Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining,” arXiv preprint arXiv:2304.09151, 2023.

[56] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and L. Cheng, “Generating diverse and natural 3d human motions from text,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 5142–5151.

[57] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the continuity of rotation representations in neural networks,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019, pp. 5738–5746.

[58] B. McFee, C. Raffel, D. Liang, D. P. Ellis, M. McVicar, E. Battenberg, O. Nieto et al., “librosa: Audio and music signal analysis in python.” SciPy, vol. 2015, no. 18-24, p. 7, 2015.

[59] A. Seth, J. L. Hicks, T. K. Uchida, A. Habib, C. L. Dembia, J. J. Dunne, C. F. Ong, M. S. DeMers, A. Rajagopal, M. Millard et al., “Opensim: Simulating musculoskeletal dynamics and neu-

romuscular control to study human and animal movement,” PLoS computational biology, vol. 14, no. 7, p. e1006223, 2018.

[60] T. Flash and N. Hogan, “The coordination of arm movements: an experimentally confirmed mathematical model,” The journal of Neuroscience, vol. 5, no. 7, pp. 1688–1703, 1985.

[61] B. Park, B.-H. Kim, and H. Chung, “Flashrender: Few-step generative rendering via camera-controlled video meanflow,” arXiv preprint arXiv:2609.03563, 2026.

[62] X. Li, J. Liu, Y. Liang, Z. Niu, W. Chen, and X. Chen, “Meanaudio: Fast and faithful text-to-audio generation with mean flows,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 14 378–14 393.

[63] A. R. Punnakkal, A. Chandrasekaran, N. Athanasiou, A. Quiros-Ramirez, and M. J. Black, “Babel: Bodies, action and behavior with english labels,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2021, pp. 722–731.

[64] S. Li, S. Zhuang, W. Song, X. Zhang, H. Chen, and A. Hao, “Sequential texts driven cohesive motions synthesis with natural transitions,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 9464–9474.

[65] Y. Shafir, G. Tevet, R. Kapon, and A. Bermano, “Human motion diffusion as a generative prior,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 8717–8733.

[66] G. Barquero, S. Escalera, and C. Palmero, “Seamless human motion composition with blended positional encodings,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 457–469.

[67] C. Guo, Y. Mu, M. G. Javed, S. Wang, and L. Cheng, “Momask: Generative masked modeling of 3d human motions,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 1900–1910.

[68] E. Pinyoanuntapong, P. Wang, M. Lee, and C. Chen, “Mmm: Generative masked motion model,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 1546–1555.

[69] E. Pinyoanuntapong, M. U. Saleem, P. Wang, M. Lee, S. Das, and C. Chen, “Bamm: Bidirectional autoregressive motion model,” in European Conference on Computer Vision. Springer, 2024, pp. 172– 190.

[70] K. Zhao, G. Li, and S. Tang, “Dartcontrol: A diffusion-based autoregressive motion model for real-time text-driven motion control,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 23 569–23 592.

[71] L. Xiao, S. Lu, H. Pi, K. Fan, L. Pan, Y. Zhou, Z. Feng, X. Zhou, S. Peng, and J. Wang, “Motionstreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 10 086–10 096.

[72] R. Li, J. Zhao, Y. Zhang, M. Su, Z. Ren, H. Zhang, Y. Tang, and X. Li, “Finedance: A fine-grained choreography dataset for 3d full body dance generation,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 10 200–10 209.

[73] J. Kim, H. Oh, S. Kim, H. Tong, and S. Lee, “A brand new dance partner: Music-conditioned pluralistic dancing controlled by multiple dance genres,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 3480–3490.

[74] R. Li, Y. Zhang, Y. Zhang, H. Zhang, J. Guo, Y. Zhang, Y. Liu, and X. Li, “Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 1524–1534.

[75] H. Liu, Z. Zhu, G. Becherini, Y. Peng, M. Su, Y. Zhou, X. Zhe, N. Iwamoto, B. Zheng, and M. J. Black, “Emage: Towards unified holistic co-speech gesture generation via expressive masked audio gesture modeling,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 1144–1154.

[76] H. Liu, Z. Zhu, N. Iwamoto, Y. Peng, Z. Li, Y. Zhou, E. Bozkurt, and B. Zheng, “Beat: A large-scale semantic and emotional multimodal dataset for conversational gestures synthesis,” in European conference on computer vision. Springer, 2022, pp. 612–630.

[77] E. Ng, J. Romero, T. Bagautdinov, S. Bai, T. Darrell, A. Kanazawa, and A. Richard, “From audio to photoreal embodiment: Synthesizing humans in conversations,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 1001–1010.

[78] M. H. Mughal, R. Dabral, M. C. Scholman, V. Demberg, and C. Theobalt, “Retrieving semantics from the deep: an rag solution

for gesture synthesis,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 16 578– 16 588.

[79] X. Zhang, J. Li, J. Zhang, Z. Dang, J. Ren, L. Bo, and Z. Tu, “Semtalk: Holistic co-speech motion generation with frame-level semantic emphasis,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 13 761–13 771.

[80] X. Zhang, J. Li, J. Zhang, J. Ren, L. Bo, and Z. Tu, “Echomask: Speech-queried attention-based mask modeling for holistic cospeech motion generation,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 10 827–10 836.

[81] J. Arevalo, T. Solorio, M. Montes-y Gomez, and F. A. Gonz´ alez,´ “Gated multimodal units for information fusion,” arXiv preprint arXiv:1702.01992, 2017.

[82] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[83] X. Huang and S. Belongie, “Arbitrary style transfer in real-time with adaptive instance normalization,” in 2017 IEEE international conference on computer vision (ICCV). IEEE, 2017, pp. 1510–1519.

[84] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[85] A. Gu, K. Goel, and C. Re, “Efficiently modeling long sequences´ with structured state spaces,” arXiv preprint arXiv:2111.00396, 2021.

[86] N. Athanasiou, M. Petrovich, M. J. Black, and G. Varol, “Teach: Temporal action composition for 3d humans,” in 2022 International Conference on 3D Vision (3DV). IEEE, 2022, pp. 414–423.

[87] M. Kocabas, N. Athanasiou, and M. J. Black, “Vibe: Video inference for human body pose and shape estimation,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 5252–5262.