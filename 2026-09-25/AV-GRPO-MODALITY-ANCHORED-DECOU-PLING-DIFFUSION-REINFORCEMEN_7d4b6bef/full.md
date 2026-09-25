# AV-GRPO: MODALITY-ANCHORED DECOU-PLING DIFFUSION REINFORCEMENT LEARNING FOR JOINT AUDIO-VIDEO GENERATION

Zhiyu Xu<sup>1,2</sup> Weilong Yan<sup>3</sup> Yufei Shi<sup>4</sup> Shiyang Li<sup>5</sup> Yihao Liu<sup>1</sup> Kin-Man Lam<sup>2,\*</sup> Yuewen Cao<sup>1,\*</sup>

<sup>1</sup>Shanghai AI Laboratory <sup>2</sup>The Hong Kong Polytechnic University

<sup>3</sup>National University of Singapore <sup>4</sup>Nanyang Technological University

<sup>5</sup>Zhejiang University <sup>\*</sup>Corresponding authors.

## ABSTRACT

Recent years have witnessed remarkable progress in joint audio-video generation. Nevertheless, existing models still suffer from limited per-modality fidelity, inadequate text-modality alignment, and weak cross-modal synchronization. Reinforcement-learning-based post-training offers a promising avenue for addressing these shortcomings. However, naively extending such approaches to joint audio-video generation is challenging. Heterogeneous multimodal rewards entangle the learning signals of the two modalities and obscure credit assignment. Jointly optimizing both modality towers also incurs substantial computational cost despite their distinct optimization dynamics. Furthermore, the difficulty of synchronization evaluation varies with the sampled counter-part modality, making fair reward comparison difficult. To address these challenges, we propose AV-GRPO, a modality-anchored online diffusion reinforcement learning framework, together with 5DAV, a fully decoupled and difficulty-controllable training dataset. AV-GRPO integrates three key components: (1) modality-anchored rollouts that disentangle learning signals while reducing anchor-induced difficulty variation; (2) trajectory-locked frozen-tower optimization that reduces training costs and redirects credit assignment, thereby simplifying optimization. and (3) adaptive objectives and perturbation strengths tailored to modality-specific dynamics and sample difficulty. Collectively, these designs transform coupled multi-modal preference learning into a set of conditional unimodal subproblems, enabling more accurate reward attribution and more effective synchronization optimization. We construct 5DAV, a dataset decoupled along five dimensions, to facilitate systematic training. Experiments on JavisBench and VABench demonstrate that AV-GRPO consistently outperforms LTX-2.3 in generation quality, semantic alignment, and cross-modal synchronization under both LoRA and full fine-tuning settings. Extensive ablation studies further validate the effectiveness of the proposed designs. Our code and data is available at AV-GRPO.

## 1 INTRODUCTION

Joint audio–video generation has advanced rapidly through unified and interacting dual-stream mod els (HaCohen et al., 2026; Low et al., 2025; Liu et al., 2026c; Team et al., 2026). Yet generated audio and video still struggle to achieve high quality together: either stream may contain perceptual artifacts, one or both may not reflect the text prompt, and otherwise plausible streams may depict mismatched events or drift out of sync. Scaling pretraining alone does not directly prioritize these failures. Reward-guided post-training instead turns perceptual and semantic evaluators into learning signals, with demonstrated benefits in language and visual generation (Rafailov et al., 2023; Shao et al., 2024; Liu et al., 2026a). Extending it to joint generation requires improving both streams while preserving their semantic and temporal dependence.

A natural baseline treats each generated audio–video pair as one policy output. For every prompt, it samples groups of paired rollouts, evaluates audio quality and alignment, video quality and alignment, and cross-modal synchronization, aggregates the heterogeneous rewards into a group-relative advantage, and updates both towers (Shao et al., 2024; Liu et al., 2026a; Zheng et al., 2026; Xue et al., 2025). This straightforward joint optimization has three difficulties.

First, mixed rewards complicate model optimization and fair reward comparison. Audio quality, visual quality, text alignment, and synchronization may favor different candidates, making their joint optimization difficult. Moreover, synchronization rewards depend on the sampled counterpart: matching audio to a steady scene can be easier than frequent visual events. A higher score may therefore reflect an easier counterpart as well as better alignment, making comparisons across jointly varying pairs less controlled. Second, joint updates incur high memory costs and can misassign credit. Backpropagation and training states are required for both towers, yet a shared advantage updates both even when its improvement primarily comes from one modality. Their interactions throughout denoising further complicate assigning that improvement to the responsible tower. Third, different modalities have different optimization dynamics. Their pretrained capabilities, latent scales, reward sensitivities, and learning speeds differ. Shared objectives and noise strengths can favor one branch or destabilize the other, making a common configuration unsuitable.

We propose AV-GRPO, which alternates audio-anchored video optimization and video-anchored audio optimization (Figure 1). First, modality-anchored rollouts disentangle rewards and control comparison conditions. Each group shares one complete anchor trajectory while independently sampling the target modality. The anchor reward can be omitted, simplifying optimization to target-modality quality and alignment plus synchronization against a common counterpart. This controls anchor-induced difficulty variation within each group, while refreshing anchors across groups preserves diverse conditions. Second, trajectory locking and tower freezing significantly reduce memory costs and direct credit to the target tower. The anchor states remain fixed throughout each rollout and its policy update, and only the target tower is optimized. Applying the conditional advantage to this tower avoids updating both branches with the same signal; freezing the counterpart also reduces backward-pass and training-state costs. Third, hyperparameter decoupling accommodates different branch dynamics. We use modality-specific reward compositions, noise strengths, and loss weightings to support exploration and stable optimization in each tower. Alternating the target modality allows both towers to improve under controlled cross-modal conditions.

To support controlled post-training, we introduce 5DAV, a training set of 5,760 prompts organized along five independently specified dimensions. We evaluate AV-GRPO on JavisBench and VABench against LTX-2.3 and GDPO under both LoRA and full fine-tuning. The results show broad gains in perceptual quality, text alignment, cross-modal coherence, and synchronization, while ablations test the dataset and the alternating schedule. The frozen-tower strategy also enables full-parameter post-training of the 22B LTX-2.3 model on eight NVIDIA A800 GPUs. Our contributions are:

• We propose AV-GRPO, a modality-anchored reinforcement learning framework that alternates controlled modality-wise updates for clearer reward attribution and balanced audio– video improvement.

• We develop three complementary mechanisms: (i) modality-anchored rollouts that disentangle reward objectives and enable controlled synchronization comparisons; (ii) trajectory locking and tower freezing that isolate modality-wise credit assignment while reducing memory overhead; and (iii) decoupled optimization hyperparameters that accommodate asymmetric audio–video learning dynamics.

• We introduce 5DAV, a five-dimensionally decoupled training set. Experiments on Javis-Bench and VABench show broad gains in generation quality, text–modality alignment, and synchronization, supported by ablations.

## 2 METHOD

## 2.1 PRELIMINARIES

Joint audio–video generation models produce a video and its audio track from a text prompt c within a single denoising process. (HaCohen et al., 2026; Liu et al., 2026c)

![](images/a58e52b30d76fa60cfec4e7c9d1d2d8d99b41835d3b292b24c79febe5b716d6d.jpg)  
Figure 1: Overview of the AV-GRPO training pipeline.

The prevailing approach is to represent video and audio in separate latent spaces, denoted $x ^ { \mathrm { V } }$ and $x ^ { \mathrm { A } }$ , and noise them following rectified flow,

$$
\boldsymbol { x } _ { t } ^ { m } = \left( 1 - t \right) \boldsymbol { x } _ { 0 } ^ { m } + t \boldsymbol { \epsilon } ^ { m } , \qquad \boldsymbol { \epsilon } ^ { m } \sim \mathcal { N } ( 0 , I ) , \quad m \in \{ \mathrm { A } , \mathrm { V } \} ,\tag{1}
$$

where the timestep t is shared by the two modalities and the noises are independent. The denoiser has one stream per modality, which we call the two towers and whose parameters we write as $\theta = \left( \theta _ { \mathrm { A } } , \theta _ { \mathrm { V } } \right)$ ; the towers exchange information through cross-modal attention or other mechanisms, so the velocity predicted by either tower depends on the current latents of both modalities:

$$
\mathrm { d } x _ { t } ^ { m } = v _ { \theta } ^ { m } \bigl ( x _ { t } ^ { \mathrm { A } } , x _ { t } ^ { \mathrm { V } } , t , c \bigr ) \mathrm { d } t , \qquad m \in \{ \mathrm { A } , \mathrm { V } \} .\tag{2}
$$

At sampling time, both modalities integrate Eq. (2) in parallel from t = 1 to $t = 0$

Flow-GRPO. Online reinforcement learning with GRPO (Shao et al., 2024) optimizes the stepby-step sampler as a policy: each denoising step is an action, and the reward is assigned to the final sample. This requires the sampling process to be stochastic, so that different samples can be drawn for the same prompt, and the per-step transition probability to be computable, so that there is a probability ratio to adjust. The deterministic ODE of Eq. (2) satisfies neither: the same initial noise always yields the same sample, and computing its transition probability is computationally expensive due to divergence estimation. Flow-GRPO (Liu et al., 2026a) therefore replaces the ODE with an SDE that has the same marginal distribution at every timestep, introducing stochasticity without changing the distribution the pretrained model generates. Discretized over T steps, the sample advances by

$$
x _ { t + \Delta t } = x _ { t } + \Bigl [ v _ { \theta } + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \bigl ( x _ { t } + ( 1 - t ) v _ { \theta } \bigr ) \Bigr ] \Delta t + \sigma _ { t } \sqrt { | \Delta t | } \epsilon , \qquad \sigma _ { t } = a \sqrt { \frac { t } { 1 - t } } ,\tag{3}
$$

where $v _ { \theta }$ abbreviates $v _ { \theta } ( x _ { t } , t , c ) , \epsilon \sim \mathcal { N } ( 0 , I )$ controls how strongly the sampling is perturbed. Each step of Eq. (3) is a Gaussian transition $\pi _ { \theta } ( x _ { t + \Delta t } \mid x _ { t } , c )$ . With this sampler, Flow-GRPO draws G trajectories per prompt and maximizes

$$
J ( \theta ) = \mathbb { E } \Big [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { T } \sum _ { t } \Big ( \operatorname* { m i n } \big ( r _ { t } ^ { i } \hat { A } ^ { i } , \mathrm { c l i p } ( r _ { t } ^ { i } , 1 - \varepsilon , 1 + \varepsilon ) \hat { A } ^ { i } \big ) - \beta D _ { \mathrm { K L } } \big ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } \big ) \Big ) \Big ] ,\tag{4}
$$

where $\hat { A } ^ { i } = ( R ^ { i } - \mathrm { m e a n } _ { i } R ^ { i } ) / \mathrm { s t d } _ { i } R ^ { i }$ is the advantage of trajectory i, obtained by standardizing the reward of its final sample, $\dot { R ^ { i } } = R ( x _ { 0 } ^ { i } , c )$ , within the group, and indicates whether the sample is better or worse than the others in its group; r<sup>i</sup> is the probability ratio between the current policy and the sampling policy at this step, i.e., a ratio of two Gaussian densities; ε is the clip range; β is the KL coefficient and $\pi _ { \mathrm { r e f } }$ the pretrained model, for which the KL term has a closed form proportional to $\| v _ { \theta } - v _ { \mathrm { r e f } } \| ^ { 2 }$

## 2.2 AV-GRPO

AV-GRPO applies the Flow-GRPO update of Eq. (4) to one tower at a time. In an audio-anchored phase, it fixes one audio trajectory, samples a group of video trajectories, and updates only the video tower; the video-anchored phase is symmetric (Figure 1). The phases alternate every n training steps. Let $m \ \in \ \{ \mathrm { A } , \mathrm { V } \}$ denote the anchor modality, m¯ the target, and $\tau ^ { m } = ( x _ { 1 } ^ { m } , \dots , x _ { 0 } ^ { m } )$ a complete denoising trajectory.

Modality-Anchored Rollouts. For prompt c, we first generate one audio–video pair by sampling both towers with Eq. (3). We retain the full trajectory $\tau _ { \mathrm { a n c } } ^ { \bar { m } }$ of modality m and independently sample G target trajectories. At each step, its anchor state $x _ { \mathrm { a n c } , t } ^ { m }$ replaces the state of modality m in crossmodal attention. The i-th target advances by

$$
x _ { t + \Delta t } ^ { \bar { m } , i } = x _ { t } ^ { \bar { m } , i } + \left[ v _ { \theta } ^ { \bar { m } } + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \big ( x _ { t } ^ { \bar { m } , i } + ( 1 - t ) v _ { \theta } ^ { \bar { m } } \big ) \right] \Delta t + \sigma _ { t } \sqrt { | \Delta t | } \epsilon ^ { \bar { m } , i } , \qquad \epsilon ^ { \bar { m } , i } \sim \mathcal { N } ( 0 , I ) ,\tag{5}
$$

where $v _ { \theta } ^ { \bar { m } }$ is evaluated at $( x _ { t } ^ { \bar { m } , i } , x _ { \mathrm { a n c } , t } ^ { m } , t , c )$ . This gives a conditional Gaussian transition $\pi _ { \theta } ^ { \bar { m } } ( x _ { t + \Delta t } ^ { \bar { m } } \mid$ $x _ { t } ^ { \bar { m } } , x _ { \mathrm { a n c } , t } ^ { m } , c )$ and G final pairs $( x _ { \mathrm { a n c } , 0 } ^ { m } , x _ { 0 } ^ { \bar { m } , i } )$ . The anchor is common to the group, whereas the target sample varies.

Reward Disentanglement. Each pair receives a target-modality quality and text-alignment reward $R _ { \bar { m } }$ and a synchronization reward $R _ { \mathrm { s y n c } }$ (Section 3.2.1). In a joint group, changes in either modality can alter the synchronization score. Here, the anchor reward is constant across all candidates and disappears under group centering, leaving

$$
R ^ { i } = R _ { \bar { m } } ( x _ { 0 } ^ { \bar { m } , i } ) + R _ { \mathrm { s y n c } } ( x _ { 0 } ^ { \bar { m } , i } , x _ { \mathrm { a n c , 0 } } ^ { m } ) .\tag{6}
$$

Meanwhile, this controls the variation in synchronization difficulty of the anchor modality: within a group, all video (or audio) samples share the same audio (or the same video), enabling fair comparison and more targeted learning of audio–video synchronization.

This grouping changes the conditional policy being optimized, rather than merely removing a reward term. For a fixed anchor trajectory, the corresponding KL-regularized subproblem can be written as

$$
\begin{array} { r } { \mathcal { L } _ { \overline { { m } } } ( \pi \mid \tau _ { \mathrm { a n c } } ^ { m } , c ) = \mathbb { E } _ { \tau ^ { \overline { { m } } } \sim \pi ^ { \overline { { m } } } ( \cdot \mid \tau _ { \mathrm { a n c } } ^ { m } , c ) } [ R _ { \overline { { m } } } + R _ { \mathrm { s y n c } } ] - \beta D _ { \mathrm { K L } } \big ( \pi ^ { \overline { { m } } } ( \cdot \mid \tau _ { \mathrm { a n c } } ^ { m } , c ) \big ) \| \pi _ { \mathrm { r e f } } ^ { \overline { { m } } } ( \cdot \mid \tau _ { \mathrm { a n c } } ^ { m } , c ) \big ) } \end{array}\tag{7}
$$

The video and audio phases solve the two symmetric conditional subproblems. Holding the anchor fixed makes their reward comparisons controlled, while refreshing it between groups prevents training on a single counterpart. The formal relation between these conditional objectives and a joint KL-regularized objective is developed in Appendix A.

For comparison, the joint formulation optimizes a distribution over paired trajectories, ${ \mathcal { L } } _ { \mathrm { j o i n t } } ( P ) =$ $\begin{array} { r l } { \mathbb { E } _ { P } [ r ( \tau ^ { \mathrm { V } } , \tau ^ { \mathrm { A } } , c ) ] - \beta D _ { \mathrm { K L } } ( P \| P _ { 0 } ) } \end{array}$ . Its samples change both audio and video at once, whereas each AV-GRPO phase conditions on one realized trajectory and changes only the other. Fixing the complete trajectory matters because the two towers interact at every denoising step: a fixed final audio or video sample alone would leave the intermediate cross-modal states uncontrolled. The conditional view explains why the within-group reward has a clearer interpretation without assuming that the two generation streams are independent.

Trajectory Locking and Tower Freezing. The same anchor states are reused throughout each rollout and during its policy update. We freeze $\theta _ { m }$ and update only $\theta _ { \bar { m } }$ , retaining gradient and optimizer states for the active tower while the anchor supplies a fixed cross-modal condition. Resampling the anchor for each group exposes the target to varied counterpart trajectories; switching towers every n steps lets both branches improve. For the fixed anchor, the conditional GRPO objective is

$$
\begin{array} { r l r } {  { \mathcal { I } _ { \bar { m } } ( \theta ) = \mathbb { E } _ { c , \tau _ { \mathrm { a n c } } ^ { m } , \{ \tau ^ { \bar { m } , i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ^ { \bar { m } } ( \cdot | \tau _ { \mathrm { a n c } } ^ { m } , c ) } \Bigg [ \frac { 1 } { G T } \sum _ { i = 1 } ^ { G } \sum _ { t } \big ( } } \\ & { } & { \operatorname* { m i n } ( r _ { t } ^ { \bar { m } , i } \hat { A } ^ { \bar { m } , i } , \mathrm { c l i p } ( r _ { t } ^ { \bar { m } , i } , 1 - \varepsilon , 1 + \varepsilon ) \hat { A } ^ { \bar { m } , i } \big ) - \beta D _ { \mathrm { K L } } \big ( \pi _ { \theta } ^ { \bar { m } } \| \pi _ { \mathrm { r e f } } ^ { \bar { m } } \big ) ) \Bigg ] , } \end{array}\tag{8}
$$

where $r _ { t } ^ { \bar { m } , i }$ is the current-to-old conditional transition-probability ratio; both policies and the reference are conditioned on the anchor. Appendix A analyzes an idealized counterpart: under a common reference joint law and contraction assumptions, alternating exact conditional Gibbs kernels converge to the joint KL-regularized optimum.

Freezing the anchor also separates the costs of generating a condition from those of learning under it. An anchor trajectory must still be sampled for each group, but its tower does not require backward activations or optimizer updates in that phase. The target tower sees the same anchor at all G comparisons, allowing reward variation to reflect its sampled trajectories rather than changes in the counterpart. Alternating phases then updates each tower under the other’s latest distribution instead of permanently fixing either one.

## 2.3 ADAPTIVE NOISE CLIPPING FOR ROBUST SAMPLING

The diffusion coefficient $\sigma _ { t } = a \sqrt { t / ( 1 - t ) }$ in Eq. (3) grows near $t \ = \ 1$ . On our coarse grid, the resulting stochastic increment can overwhelm the latent signal during noising and destabilize training. Adaptive Noise Clipping (ANC) bounds its standard deviation using

$$
s _ { t } = \sigma _ { t } \sqrt { | \Delta t | } , \qquad \lambda _ { t } = \operatorname* { m i n } \biggl ( 1 , \frac { \tau } { s _ { t } + \delta } \biggr ) , \qquad \tilde { \sigma } _ { t } = \lambda _ { t } \sigma _ { t } ,\tag{9}
$$

where $\tau > 0$ is a clipping threshold and $\delta > 0$ prevents division by zero. We substitute ${ \tilde { \sigma } } _ { t }$ for $\sigma _ { t }$ in both the stochastic term and the drift correction of Eq. (3):

$$
x _ { t + \Delta t } = x _ { t } + \left[ v _ { \theta } + \frac { \tilde { \sigma } _ { t } ^ { 2 } } { 2 t } \big ( x _ { t } + ( 1 - t ) v _ { \theta } \big ) \right] \Delta t + \tilde { \sigma } _ { t } \sqrt { | \Delta t | } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) .\tag{10}
$$

When clipping is inactive, Eq. (10) reduces to the original update; otherwise the stochastic term scales by $\lambda _ { t }$ and the drift correction by $\lambda _ { t } ^ { 2 }$ . We use 15 sampling steps and find ANC stabilizes training at larger noise strengths.

## 2.4 HYPERPARAMETER DECOUPLING AND TRAINING STABILITY

A shared configuration can limit exploration in one tower while destabilizing the other. AV-GRPO naturally supports independent optimization configurations through its alternating frozen-tower updates, allowing noise strengths and objectives to be tailored to each modality. For LTX-2.3, we use $a _ { \mathrm { V } } = 0 . 0 2$ and $a _ { \mathrm { A } } = 0 . 8$ in $\sigma _ { t } = a \sqrt { t / ( 1 - t ) }$ : too little noise limits exploration, whereas too much can break denoising.

With the original Flow-GRPO objective, the video tower becomes over-exposed after a few iterations. The late denoising stages, which control brightness and fine detail, receive weak gradients; changing only the scalar KL coefficient does not resolve this imbalance. Following LongCat-Video (Team et al., 2025), we reweight the video policy and KL terms at each step:

$$
\mathcal { T } _ { \mathrm { v i d e o } } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \left( \lambda _ { \mathrm { p o l i c y } } ( t , \Delta t ) \cdot \mathcal { L } _ { \mathrm { p o l i c y } } ( \theta ) - \beta \lambda _ { \mathrm { K L } } ( t , \Delta t ) \cdot D _ { \mathrm { K L } } ( \pi _ { \theta } ^ { \mathrm { v i d e o } } | | \pi _ { \mathrm { r e f } } ^ { \mathrm { v i d e o } } ) \right) \right] ,\tag{11}
$$

The weighting functions are defined as:

$$
\lambda _ { \mathrm { p o l i c y } } ( t , \Delta t ) = \sqrt { \frac { t } { \Delta t ( 1 - t ) } } , \qquad \lambda _ { \mathrm { K L } } ( t , \Delta t ) = \frac { t } { \Delta t ( 1 - t ) } .\tag{12}
$$

The audio tower retains the original weighting. Appendix C gives the full objective and gradient derivation.

## 3 EXPERIMENT

## 3.1 5DAV DATASET

To support GRPO optimization for joint audio-video generation models, we construct a 5D decoupled training set (5DAV) that is fully disentangled across five dimensions: semantic hierarchy, sound source type, synchronization difficulty, temporal complexity, and instruction granularity. Through Cartesian product of these dimensions, our dataset achieves comprehensive capability coverage with controllable difficulty levels. The five dimensions are defined as follows:

Prompt: There is a large building on fire with intense flames ... The building is surrounded by water ...   
A loud explosion is heard, fol owed by the sound of fire crackling and burning. <sup>T</sup>he sky is clear, and ...

![](images/4c7694c42537d333e7d057f7a744f9d5b5884e4af27b5b87323607db0a6e21f9.jpg)  
Prompt: A child pushes a toy truck, says “Vroom,” pul s it back, says “Beep,” then crashes it into a block and shouts “Boom.”  
Prompt: A woman is playing the violin in an orchestra setting. . . The background shows other musicians . . . . a stained glass window behind them. The sound of the violin blends with the orchestra's accompaniment.

![](images/9e2010b7c8562466947aa1ee374e0ca50ba51435a00aa38cd848055470817365.jpg)  
Prompt: A smal stream flows over rocks and grass, with the water clear and the rocks covered in moss. A cartoonish green character with ... and a single eye jumps into the water making a croaking sound.

![](images/b063bb5dc1b06a2413175b4ddbae4d369140205cde7e7d479d946a6df8da2458.jpg)

![](images/5cdc108e6a6be1b7207061de8bbfd81db29b331a824c5900bf7dcf8f9aa4b8ff.jpg)  
Figure 2: Qualitative comparison of LTX-2.3 and AV-GRPO (full). For each example, rows 2– 3 show the video frames and corresponding audio waveform generated by the original LTX-2.3 model; rows 4–5 show those generated by AV-GRPO (full). Bold text in the prompt highlights the visual events and sound cues that require audio–video synchronization.

• Semantic Hierarchy (W): Entity-level (single object/person/animal and its inherent sound), Scene-level (static scene with background atmosphere), Event-level (dynamic interactions and causal events among multiple entities).

• Sound Source Type (H): Human voice, Environmental and object sounds, Music, Mixed sources (two or more types).

• Synchronization Difficulty (D): Irrelevant (audio and video independent), Category matching (audio category matches the scene), Frame-level synchronization (audio and video fully aligned), Physical causal synchronization (e.g., pitch rising as a train approaches).

• Temporal Complexity (T): Single-event (single scene/event, no scene transition), Multievent (multiple scenes/causal events, with scene transitions).

• Instruction Granularity (C): Weak constraint (core semantics only), Medium constraint (explicit sound source/scene/event), Strong constraint (fine-grained details including precise timing, spatial position, pitch/volume variations).

The five-dimensional decomposition yields $3 \times 4 \times 4 \times 2 \times 3 = 2 8 8$ orthogonal categories, with 20 prompts generated per category, resulting in a total of 5,760 data samples. This design promotes comprehensive coverage of audio-video generation application scenarios. Moreover, the sampling ratio of categories within each dimension can be flexibly adjusted according to the requirements of different training stages, enabling the dataset to adapt to diverse optimization objectives and model iteration paces. This design makes the training process traceable and model weaknesses localizable, providing clear guidance for data selection and reward function design.

## 3.2 EXPERIMENTAL SETUP

Training configuration. All LoRA and full-parameter runs use a single node of eight NVIDIA A800 80 GB GPUs. LTX-2.3 22B generates 544 × 960 clips of 97 frames at 24 FPS with 15 denoising steps. The video tower is updated first, and the active tower switches every four steps. Appendix B provides the remaining implementation details.

## 3.2.1 REWARD MODELS AND REWARD COMPOSITION

For video-tower optimization, we use VideoAlign (Liu et al., 2026b), CLIP (Radford et al., 2021), and DeSync (Iashin et al., 2024) to assess video quality, video–prompt alignment, and audio–video synchronization, respectively. For audio-tower optimization, we use Audiobox Aesthetics (Tjandra et al., 2025), CLAP (Wu et al., 2023), and DeSync to assess audio quality, audio–prompt alignment, and audio–video synchronization, respectively.

Following GDPO (Liu et al., 2026e), the video score sums standardized VideoAlign, CLIP, and negative DeSync, whereas the audio score sums standardized audio aesthetics, CLAP, and negative DeSync (Since lower DeSync is better, its sign is reversed). We standardize the resulting score again within the group to obtain the trajectory advantage. For audio, a CLAP guardrail uses only the alignment advantage when the group’s average CLAP falls below a threshold; this prevents quality and synchronization gains from masking a loss of prompt fidelity. Appendix D gives the equations.

## 3.2.2 EVALUATION BENCHMARKS

We adopt JavisBench (Liu et al., 2026c) and VABench (Hua et al., 2026) as our evaluation benchmarks to comprehensively assess the generated audio-video content.

JavisBench evaluates generated samples across four complementary dimensions: (1) Unimodal Generation Fidelity, measured by Visual Quality (VQ) and Audio Quality (AQ), assessing the perceptual quality of each modality independently; (2) Text-Modal Alignment, where ImageBind similarity, CLIP score, and CLAP score are employed to measure the semantic consistency between the textual prompt and each generated modality; (3) Audio-Video Semantic Coherence, evaluated via Audio-Video ImageBind similarity (AV-IB) and AVH Score to quantify cross-modal semantic alignment; (4) Audio-Video Synchronization, assessed by JavisScore and DeSync to measure temporal correspondence between audio and video streams.

VABench organizes its evaluation into two paradigms. The first paradigm relies on expert models to provide objective perceptual quality assessments, including speech quality and naturalness (SpeechQual&Nat), audio aesthetics (AudioAesthetic), and lip synchronization accuracy (Lip-Sync). The second paradigm leverages Multimodal Large Language Models (MLLMs) to simulate human judgments on complex audio-video semantics, covering high-level criteria such as global Alignment, Artistic quality (Artistry), and Expressiveness.

## 3.3 MAIN RESULTS AND ANALYSIS

## 3.3.1 QUANTITATIVE ANALYSIS

Table 1 and 2 report the results of AV-GRPO under both LoRA and full fine-tuning on JavisBench and VABench. Compared with LTX-2.3 (22B) and GDPO, AV-GRPO consistently improves nearly all metrics under both settings, with full fine-tuned model attaining the best overall performance.

In terms of perceptual quality, the full fine-tuned model improves AQ on JavisBench from 5.097 to 5.798 and Audio Aesthetics on VABench from 3.319 to 3.631 over LTX-2.3. For semantic alignment, it boosts CLIP on JavisBench from 0.318 to 0.327 and CLAP from 0.408 to 0.468. For crossmodality consistency and synchronization, substantial gains are observed across the board: AVH Score, DeSync, and AV-Align on JavisBench, as well as Lip Sync and DeSync on VABench, all exhibit significant improvements. In all these aspects, AV-GRPO consistently outperforms GDPO by a considerable margin.

The two training regimes show different strengths. Full fine-tuning yields the highest JavisBench AQ and AV-IB scores (5.798 and 0.247), while LoRA attains a slightly higher CLIP score (0.330 versus 0.327) and a lower DeSync value (0.554 versus 0.607). The improvements are therefore broad rather than uniform: LoRA’s VQ is 5.816, below the base model’s 5.855, and the full model’s VABench visual-realism score is 4.395 versus 4.399 for the base model. These exceptions do not change the gains in most quality and alignment measures, but they show that different modalities and training regimes retain distinct tradeoffs.

Compared with GDPO, AV-GRPO (full) raises JavisBench AV-IB from 0.224 to 0.247 and JavisScore from 0.202 to 0.222, reducing DeSync from 0.708 to 0.607. On VABench, its Lip Sync increases from 1.439 to 1.646 and DeSync decreases from 0.726 to 0.542. These comparisons focus on cross-modal measures, where controlling the ref-modality is most relevant to the learning signal.

Table 1: Main results on JavisBench. Best results are in bold, second-best are underlined. VQ: Visual Quality, AQ: Audio Quality. (↑: higher is better; ↓: lower is better).
<table><tr><td rowspan="2">Model</td><td rowspan="2">size</td><td colspan="2">AV-Quality</td><td colspan="4">Text-Consistency</td><td colspan="2">AV-Consistency</td><td colspan="3">AV-Synchrony</td></tr><tr><td>VQ↑</td><td>AQ↑</td><td>TV-IB↑</td><td>TA-IB↑</td><td>CLIP↑</td><td>CLAP↑</td><td>AV-IB↑</td><td>AVHScore↑</td><td>JavisScore↑</td><td>DeSync↓</td><td>AV-align↑</td></tr><tr><td>-T2A+A2V</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TempoTKn</td><td>1.3B</td><td></td><td></td><td>0.084</td><td></td><td>0.205</td><td></td><td>0.139</td><td>0.122</td><td>0.103</td><td>1.532</td><td></td></tr><tr><td>TPoS</td><td>1.0B</td><td></td><td></td><td>0.201</td><td></td><td>0.229</td><td></td><td>0.124</td><td>0.129</td><td>0.095</td><td>1.493</td><td></td></tr><tr><td>-T2V+V2A</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReWaS</td><td>0.6B</td><td></td><td></td><td></td><td>0.123</td><td></td><td>0.280</td><td>0.110</td><td>0.104</td><td>0.079</td><td>1.071</td><td></td></tr><tr><td>See&amp;Hear</td><td>0.4B</td><td></td><td></td><td></td><td>0.129</td><td></td><td>0.263</td><td>0.160</td><td>0.143</td><td>0.112</td><td>1.099</td><td></td></tr><tr><td>FolleyCrafter</td><td>1.2B</td><td></td><td></td><td></td><td>0.149</td><td></td><td>0.383</td><td>0.193</td><td>0.186</td><td>0.151</td><td>0.952</td><td></td></tr><tr><td>MMAudio</td><td>0.1B</td><td></td><td></td><td></td><td>0.160</td><td></td><td>0.407</td><td>0.198</td><td>0.182</td><td>0.150</td><td>0.849</td><td></td></tr><tr><td>-T2AV</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JavisDiT</td><td>3.1B</td><td>1.291</td><td>4.478</td><td>0.263</td><td>0.143</td><td>0.302</td><td>0.391</td><td>0.197</td><td>0.179</td><td>0.154</td><td>1.039</td><td></td></tr><tr><td>UniVerse-1</td><td>6.4B</td><td>1.357</td><td>4.839</td><td>0.272</td><td>0.111</td><td>0.309</td><td>0.245</td><td>0.104</td><td>0.098</td><td>0.077</td><td>0.929</td><td></td></tr><tr><td>JavisDiT++</td><td>2.1B</td><td>1.462</td><td>5.049</td><td>0.282</td><td>0.164</td><td>0.316</td><td>0.424</td><td>0.198</td><td>0.184</td><td>0.159</td><td>0.832</td><td></td></tr><tr><td>LTX-2.3</td><td>22B</td><td>5.855</td><td>5.097</td><td>0.284</td><td>0.143</td><td>0.318</td><td>0.408</td><td>0.212</td><td>0.201</td><td>0.183</td><td>0.757</td><td>0.354</td></tr><tr><td>LTX-2.3+GDPO</td><td>22B</td><td>5.870</td><td>5.406</td><td>0.283</td><td>0.165</td><td>0.313</td><td>0.431</td><td>0.224</td><td>0.217</td><td>0.202</td><td>0.708</td><td>0.360</td></tr><tr><td>LTX-2.3+AV-GRPO(LoRA)</td><td>22B</td><td>5.816</td><td>5.633</td><td>0.289</td><td>0.174</td><td>0.330</td><td>0.449</td><td>0.236</td><td>0.235</td><td>0.213</td><td>0.554</td><td>0.382</td></tr><tr><td>LTX-2.3+AV-GRPO(full)</td><td>22B</td><td>5.902</td><td>5.798</td><td>0.288</td><td>0.185</td><td>0.327</td><td>0.468</td><td>0.247</td><td>0.241</td><td>0.222</td><td>0.607</td><td>0.398</td></tr></table>

Table 2: Main results on VABench. Best results are in bold, second-best are underlined. For all metrics except DeSync, higher is better; for DeSync, lower is better.
<table><tr><td>Model</td><td>Speech Q&amp;N</td><td>Audio Aes</td><td>T-V Align</td><td>T-A Align</td><td>A-V Align</td><td>Lip Sync</td><td>DeSync↓</td><td>Alignment</td><td>Express- siveness</td><td>Visual Realism</td><td>Audio Realism</td><td>Artistry</td></tr><tr><td>LTX-2.3</td><td>1.383</td><td>3.319</td><td>0.211</td><td>0.398</td><td>0.227</td><td>1.351</td><td>0.800</td><td>4.455</td><td>4.371</td><td>4.399</td><td>3.830</td><td>3.766</td></tr><tr><td>LTX-2.3+GDPO</td><td>1.465</td><td>3.452</td><td>0.210</td><td>0.424</td><td>0.243</td><td>1.439</td><td>0.726</td><td>4.481</td><td>4.409</td><td>4.413</td><td>3.849</td><td>3.784</td></tr><tr><td>LTX-2.3+AV-GRPO(lora)</td><td>1.487</td><td>3.553</td><td>0.217</td><td>0.446</td><td>0.261</td><td>1.585</td><td>0.594</td><td>4.512</td><td>4.450</td><td>4.402</td><td>3.865</td><td>3.792</td></tr><tr><td>LTX-2.3+AV-GRPO(full)</td><td>1.510</td><td>3.631</td><td>0.215</td><td>0.468</td><td>0.272</td><td>1.646</td><td>0.542</td><td>4.510</td><td>4.434</td><td>4.395</td><td>3.878</td><td>3.798</td></tr></table>

## 3.3.2 QUALITATIVE ANALYSIS

Figure 2 showcases several representative cases. As shown in the figure, compared with the original model, AV-GRPO-trained LTX-2.3 achieves improvements across different aspects: per-modality fidelity, modality-text alignment, and cross-modal synchronization.

The examples span event-driven effects, music performance, object sounds, and natural ambience. In each row, the video frames and corresponding waveform are shown together so that visual changes can be compared with the timing and character of the generated audio. These cases complement the aggregate benchmark scores by illustrating the kinds of cross-modal relationships optimized through the shared-anchor comparisons.

## 3.4 ABLATION STUDY

## 3.4.1 DATASET

To demonstrate the effectiveness of our proposed dataset, we conduct a comparative experiment against the well-known audio-video dataset VGGSound (Chen et al., 2020) under full finetuning settings. All training configurations are kept identical across experiments. The results on JavisBench-mini are reported in Figure 3.

We conduct ablation studies on the difficulty levels of Synchronization Difficulty (D) and Instruction Granularity (C). Specifically, we compare the easiest and most difficult categories within each of these two dimensions (denoted as D/C easiest/hardest) while keeping all other settings identical. The experimental results are reported in Figure 3. The results demonstrate that the decoupling and categorical partitioning of dimensions and difficulty levels contribute substantially to the model’s performance.

The left panel compares the two training sets across quality, text alignment, cross-modal consistency, and synchronization metrics; the middle panel reports the relative change when an easy or hard level of C or D is removed. Together, these views test both the value of the complete category grid and whether its difficulty levels supply distinct training signals. In particular, changes in the synchronization measures under the D removals show why a single aggregate benchmark score is insufficient to assess the data design.

![](images/d66586a1300fda28000c10fc050bfd706ced309f37316a765763b203bc311be2.jpg)  
Figure 3: Left: Radar chart comparing performance on the 5DAV and VGGSound datasets. Middle: Metrics are reported as relative percentage changes with respect to the baseline. Right: Comparison of model performance with different alternating step intervals.

## 3.4.2 ALTERNATING MODALITY-WISE OPTIMIZATION

To verify the effectiveness of alternating optimization, we conduct ablation studies on whether to apply alternating optimization, and provide experimental results with varying switch intervals (i.e., the number of training steps per modality before switching to the other). All experiments are conducted on JavisBench-mini, as shown in Figure 3. The results demonstrate that alternating optimization substantially benefits the performance of both towers. Under our current configuration, switching every four steps achieves relatively optimal performance.

The right panel includes no alternation and intervals of two, four, and eight steps. The four-step schedule gives the strongest overall balance across the plotted measures, although individual metrics need not peak at the same interval. This comparison motivates the switching interval used for the main LoRA and full-parameter runs.

## 4 RELATED WORK

## 4.1 JOINT AUDIO–VIDEO GENERATION

Early audio–visual generation methods typically adopt asymmetric pipelines that synthesize one modality conditioned on the other. For example, FoleyCrafter (Zhang et al., 2026b) and MMAudio (Cheng et al., 2025) generate temporally aligned audio from video. While such methods benefit from strong unimodal priors, their asymmetric formulations limit direct bidirectional co-generation. Recent works instead generate audio and video within a unified process. JavisDiT (Liu et al., 2026c), UniVerse-1 (Wang et al., 2025), Ovi (Low et al., 2025) and LTX-2 (HaCohen et al., 2026) employ interacting dual-stream architectures with cross-modal fusion. These advances primarily focus on model architecture, data construction, and large-scale pretraining. In contrast, our work studies reward-guided post-training of joint audio–video models, aiming to improve both modalities without entangling their learning signals.

## 4.2 REINFORCEMENT LEARNING-BASED POST-TRAINING

Reward-guided post-training has been widely explored for diffusion and flow-based generation. DDPO (Black et al., 2023) and DPOK (Fan et al., 2023) formulate reverse denoising as a sequential decision process and optimize diffusion models using policy gradients. More recently, Flow-GRPO (Liu et al., 2026a) converts deterministic flow sampling into an equivalent stochastic process for online GRPO. DiffusionNFT (Zheng et al., 2026) instead performs online reward optimization through the forward process. For multiple objectives, GDPO (Liu et al., 2026e) independently normalizes individual rewards before aggregation, but does not resolve cross-modal credit assignment when both modalities vary simultaneously.

## 5 CONCLUSION

We presented AV-GRPO, a modality-anchored reinforcement learning framework for joint audio– video post-training. By combining anchored rollouts with alternating frozen-tower optimization,

AV-GRPO disentangles rewards and controls comparison conditions, reduces training overhead, redirects credit assignment, and adapts to modality-specific dynamics and sample difficulty. We also introduced 5DAV, a five-dimensionally decoupled dataset for controllable post-training. Experiments on JavisBench and VABench show consistent improvements over LTX-2.3 and GDPO in perceptual quality, text alignment, and audio–video coherence and synchronization under both LoRA and full fine-tuning. These results demonstrate an effective and scalable approach to improving coupled audio–video generators.

## ACKNOWLEDGMENTS

This work is supported by Shanghai Artificial Intelligence Laboratory.

## REFERENCES

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. 2023.

Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audiovisual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 721–725. IEEE, 2020.

Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, and Yuki Mitsufuji. MMAudio: Taming multimodal joint training for high-quality video-to-audio synthesis. In CVPR, 2025.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, P. Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. ArXiv, abs/2305.16381, 2023. URL https: //api.semanticscholar.org/CorpusID:258947323.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

Daili Hua, Xizhi Wang, Bohan Zeng, Xinyi Huang, Hao Liang, Junbo Niu, Xinlong Chen, Quanqing Xu, and Wentao Zhang. Vabench: A comprehensive benchmark for audio-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23345–23355, 2026.

Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchronization from sparse cues. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5325–5329. IEEE, 2024.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026a.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. Advances in Neural Information Processing Systems, 38:82155–82192, 2026b.

Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Jiebo Luo, Ziwei Liu, Hao Fei, and Tat-Seng Chua. Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. In ICLR, 2026c.

Kai Liu, Yanhao Zheng, Kai Wang, Shengqiong Wu, Rongjunchen Zhang, Jiebo Luo, Dimitrios Hatzinakos, Ziwei Liu, Hao Fei, and Tat-Seng Chua. Javisdit++: Unified modeling and optimization for joint audio-video generation. In The Fourteenth International Conference on Learning Representations, 2026d.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen,´ Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. ArXiv, abs/2601.05242, 2026e. URL https://api.semanticscholar. org/CorpusID:284543897.

Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin backbone cross-modal fusion for audiovideo generation. arXiv preprint arXiv:2510.01284, 2025.

Navonil Majumder, Chia-Yu Hung, Deepanway Ghosal, Wei-Ning Hsu, Rada Mihalcea, and Soujanya Poria. Tango 2: Aligning diffusion-based text-to-audio generations through direct preference optimization. Proceedings ofthe 32nd ACM International Conference on Multimedia, 2024. URL https://api.semanticscholar.org/CorpusID:269149104.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Meituan LongCat Team, Bairui Wang, Bin Xiao, Bo Zhang, Bolin Rong, Borun Chen, Chang Wan, Chao Zhang, Chen Huang, Chen Chen, et al. Longcat-flash-omni technical report. arXiv preprint arXiv:2511.00279, 2025.

OpenMOSS Team, Donghua Yu, Mingshu Chen, Qi Chen, Qi Luo, Qianyi Wu, Qinyuan Cheng, Ruixiao Li, Tianyi Liang, Wenbo Zhang, et al. Mova: Towards scalable and synchronized video audio generation. arXiv preprint arXiv:2602.08794, 2026.

Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025.

Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025.

Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Guohui Zhang, Xiaoxiao Ma, Jie Huang, Hang Xu, Hu Yu, Siming Fu, Yuming Li, Zeyue Xue, Lin Song, Haoyang Huang, Nan Duan, and Feng Zhao. Omninft: Modality-wise omni diffusion reinforcement for joint audio-video generation. ArXiv, abs/2605.12480, 2026a. URL https: //api.semanticscholar.org/CorpusID:288257189.

Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, Bin Liu, and Kai Chen. Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. International Journal ofComputer Vision, 134(1):46, 2026b.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. In International Conference on Learning Representations, volume 2026, pp. 134129–134150, 2026.

## A PROOF OF THE CONVERGENCE AND JOINT OPTIMALITY OF AV-GRPO

This section studies KL-regularized reinforcement learning fine-tuning of audio-video two-tower diffusion models under the protocol of “alternately fixing a single-modality complete denoising trajectory”. We introduce a common reference joint law $P _ { 0 }$ , and specify the two single-tower operation kernels as the regular conditional kernels of $P _ { 0 } ;$ the reward r is bounded measurable, and $\beta > 0$ . We define the exponentially tilted joint law $\mathrm { d } P ^ { * } / \mathrm { d } P _ { 0 } \propto e ^ { r / \beta }$ , and construct the two conditional Gibbs optimal kernels $K _ { v } ^ { * } , K _ { a } ^ { * }$ . Under the Dobrushin contraction condition $\delta _ { v } \delta _ { a } < 1$ , we prove: $P ^ { * }$ is the unique global optimum of the joint KL-regularized objective; $K _ { v } ^ { * } , K _ { a } ^ { * }$ are the unique optimal kernels of the conditional subproblems, respectively, and are automatically compatible; the systematic-scan Gibbs chain with kernels $K _ { v } ^ { * } , K _ { a } ^ { * }$ has a unique stationary distribution ${ \bar { P } } ^ { * }$ ; starting from any initial joint law, the joint laws of both sampling phases converge geometrically in total variation to $P ^ { * }$ with an explicit convergence rate. If the model implements $\check { K } _ { v } ^ { * } , K _ { a } ^ { * }$ via a non-interfering exact kernel oracle, then both the training rollouts and the inference distribution using the same alternating full-trajectory protocol possess the above convergence properties.

## A.1 PROBLEM SETUP AND BASIC ASSUMPTIONS

## A.1.1 SPACES AND THE REFERENCE JOINT LAW

Fix the text condition c, which is omitted in the following. Let $\mathcal { X }$ be the space of complete video denoising trajectories, and $\mathcal { V }$ the space of complete audio denoising trajectories. Assume that $\mathcal { X } , \mathcal { y }$ are spaces equipped with fixed Polish topologies, with their Borel σ-algebras denoted by $B _ { \mathcal { X } } , B _ { \mathcal { Y } }$ respectively. Thus $\mathcal { X } \times \mathcal { V }$ is also a Polish space, and regular conditional probability kernels exist. Let $\mathcal { P } ( \mathcal { X } ) , \mathcal { P } ( \mathcal { Y } ) , \mathcal { P } ( \mathcal { X } \times \mathcal { Y } )$ denote the corresponding sets of probability measures.

Assumption 1 (Common reference joint law H0). There exists a reference joint probability measure $P _ { 0 } \in \mathcal P ( \mathcal X \times \mathcal y )$ . The actual reference kernel for “running the video tower with the audio trajectory fixed”, $\dot { K } _ { v } ^ { 0 } : \mathcal { V } \times B _ { \mathcal { X } } \to [ 0 , 1 ]$ , and the reference kernel for “running the audio tower with the video trajectory fixed”, $K _ { a } ^ { 0 } : \mathcal { X } \times \dot { \mathcal { B } _ { y } }  [ 0 , 1 ]$ , are two versions of the regular conditional distributions of $P _ { 0 } \colon$

$$
K _ { v } ^ { 0 } ( \cdot \mid y ) = P _ { 0 } ( \mathrm { d } x \mid y ) , \quad P _ { 0 , Y } - \mathrm { a . s . } , \qquad K _ { a } ^ { 0 } ( \cdot \mid x ) = P _ { 0 } ( \mathrm { d } y \mid x ) , \quad P _ { 0 , X } - \mathrm { a . s . } ,
$$

and we select a jointly measurable version of them that is defined on all of $\mathcal { X } , \mathcal { y }$ . Hence

$$
P _ { 0 } ( A \times B ) = \int _ { B } K _ { v } ^ { 0 } ( A \mid y ) P _ { 0 , Y } ( \mathrm { d } y ) = \int _ { A } K _ { a } ^ { 0 } ( B \mid x ) P _ { 0 , X } ( \mathrm { d } x ) .
$$

## A.1.2 REWARD AND THE JOINT OBJECTIVE

Let $r : \mathcal { X } \times \mathcal { Y } $ R be a bounded Borel measurable function, $\beta > 0$ , and let $M = \| r \| _ { \infty }$ . Define the joint objective

$$
J ( P ) = \mathbb { E } _ { P } [ r ] - \beta D _ { \mathrm { K L } } ( P \parallel P _ { 0 } ) , \qquad P \in \mathcal { P } ( \mathcal { X } \times \mathcal { Y } ) ,
$$

with the convention that $D _ { \mathrm { K L } } ( P \parallel P _ { 0 } ) = \infty$ when $P \not \ll P _ { 0 }$ , so that $J ( P ) = - \infty$

## A.1.3 CONDITIONAL SUBPROBLEM OBJECTIVES

Fix $y \in \mathcal { V } ;$ ; for any $q \in { \mathcal { P } } ( { \mathcal { X } } )$ , define the video subproblem

$$
L _ { v } ( q ; y ) = \int _ { \mathcal { X } } r ( x , y ) q ( \mathrm { d } x ) - \beta D _ { \mathrm { K L } } \big ( q \lVert K _ { v } ^ { 0 } ( \cdot \lfloor y ) \big ) .
$$

Fix $x \in { \mathcal { X } } ;$ for any $q \in \mathcal { P } ( \mathcal { Y } )$ , define the audio subproblem

$$
L _ { a } ( q ; x ) = \int _ { y } r ( x , y ) q ( \mathrm { d } y ) - \beta D _ { \mathrm { K L } } \bigl ( q \bigr | \mathrm { ~ } K _ { a } ^ { 0 } ( \cdot \mid x ) \bigr ) .
$$

## A.2 EXPONENTIAL TILTING AND CONDITIONAL GIBBS OPTIMAL KERNELS

## A.2.1 JOINT EXPONENTIAL TILTING

Define the normalizing constant and the probability measure

$$
Z = \int _ { \mathcal { X } \times \mathcal { Y } } e ^ { r ( x , y ) / \beta } P _ { 0 } ( \mathrm { d } x , \mathrm { d } y ) , \qquad \frac { \mathrm { d } P ^ { * } } { \mathrm { d } P _ { 0 } } ( x , y ) = Z ^ { - 1 } e ^ { r ( x , y ) / \beta } .
$$

Since $r$ is bounded, $e ^ { - M / \beta } \le Z \le e ^ { M / \beta }$ , hence $0 < Z < \infty$ , and $P ^ { * }$ and $P _ { 0 }$ are mutually absolutely continuous. Their marginals are denoted by $P _ { X } ^ { * }$ and $P _ { Y } ^ { * }$ , respectively.

## A.2.2 CONDITIONAL NORMALIZING CONSTANTS AND OPTIMAL KERNELS

Define

$$
Z _ { v } ( y ) = \int _ { \mathcal { X } } e ^ { r ( x , y ) / \beta } K _ { v } ^ { 0 } ( \mathrm { d } x \mid y ) , \qquad Z _ { a } ( x ) = \int _ { \mathcal { Y } } e ^ { r ( x , y ) / \beta } K _ { a } ^ { 0 } ( \mathrm { d } y \mid x ) .
$$

By the measurability theorem for kernel integrals, $Z _ { v } , Z _ { a }$ are measurable, and $e ^ { - M / \beta } \leq$ $\dot { Z _ { v } } ( y ) , Z _ { a } ( x ) \le e ^ { M / \beta }$ , so the denominators are everywhere positive and finite. Define the $\mathrm { \ o p t i - }$ mal conditional kernels

$$
K _ { v } ^ { * } ( \mathrm { d } x \mid y ) = Z _ { v } ( y ) ^ { - 1 } e ^ { r ( x , y ) / \beta } K _ { v } ^ { 0 } ( \mathrm { d } x \mid y ) ,
$$

$$
K _ { a } ^ { * } ( \mathrm { d } y \mid x ) = Z _ { a } ( x ) ^ { - 1 } e ^ { r ( x , y ) / \beta } K _ { a } ^ { 0 } ( \mathrm { d } y \mid x ) .
$$

The integrands are jointly measurable, and the normalizing constants are measurable and everywhere positive; therefore $K _ { v } ^ { * } , K _ { a } ^ { * }$ are probability kernels defined on all conditional states.

## A.3 JOINT KL VARIATIONAL IDENTITY

Lemma 1 (Joint Gibbs variational identity). For any $P \in \mathcal P ( \mathcal x \times \mathcal y )$ , in the extended real-valued sense,

$$
J ( P ) = \beta \log Z - \beta D _ { \mathrm { K L } } ( P \parallel P ^ { * } ) .
$$

Therefore $P ^ { * }$ is the unique global optimum of J over all joint probability measures, with optimal value β log Z.

Proof. Since $\mathrm { d } P ^ { * } / \mathrm { d } P _ { 0 } = e ^ { r / \beta } / Z$ has strictly positive finite upper and lower bounds, $P ^ { * }$ and $P _ { 0 }$ are equivalent. Therefore $P \ll P _ { 0 }$ if and only if $P \ll P ^ { * }$ . If this condition does not hold, both sides are −∞, and the identity holds. Assume $P \ll P _ { 0 }$ . By the Radon–Nikodym chain rule, $P \mathrm { - a . s }$ we have

$$
\log { \frac { \mathrm { d } P } { \mathrm { d } P ^ { * } } } = \log { \frac { \mathrm { d } P } { \mathrm { d } P _ { 0 } } } - { \frac { r } { \beta } } + \log Z .
$$

Integrating both sides against $P$ and multiplying by $\beta ,$ we obtain

$$
\beta D _ { \mathrm { K L } } ( P \parallel P ^ { * } ) = \beta D _ { \mathrm { K L } } ( P \parallel P _ { 0 } ) - \mathbb { E } _ { P } [ r ] + \beta \log Z .
$$

Because r is bounded, the two KL terms are either simultaneously finite or simultaneously $\infty ,$ so the above identity is unambiguous in the extended real-valued sense. Rearranging yields

$$
J ( P ) = \beta \log Z - \beta D _ { \mathrm { K L } } ( P \parallel P ^ { * } ) \le \beta \log Z .
$$

Finally, $D _ { \mathrm { K L } } ( P \parallel P ^ { * } ) = 0$ if and only if $P = P ^ { * }$ . Hence $P ^ { * }$ is the unique optimum. □

## A.4 OPTIMAL KERNELS OF THE CONDITIONAL SUBPROBLEMS

Lemma 2 (Conditional Gibbs optimality). For each $y \in \mathcal { V } , K _ { v } ^ { * } ( \cdot \mid y )$ is the unique global optimum of $L _ { v } ( \cdot ; y )$ over $\mathcal { P } ( \mathcal { X } )$ , with optimal value β log $Z _ { v } ( y )$ . For each $\bar { x } \in \mathcal { X } , K _ { a } ^ { * } ( \bar { \cdot } \mid \bar { x } )$ is the unique global optimum of the audio subproblem, with optimal value β log $Z _ { a } ( x )$ .

Proof. Fix y, and replace $P _ { 0 } , P ^ { * } , Z , r$ by $K _ { v } ^ { 0 } ( \cdot \mid y ) , K _ { v } ^ { * } ( \cdot \mid y ) , Z _ { v } ( y ) , r ( \cdot , y )$ , respectively. The Radon–Nikodym computation of Lemma 1 applies verbatim, giving

$$
L _ { v } ( q ; y ) = \beta \log Z _ { v } ( y ) - \beta D _ { \mathrm { K L } } \big ( q \parallel K _ { v } ^ { * } ( \cdot \mid y ) \big ) .
$$

KL is nonnegative and vanishes only when $q = K _ { v } ^ { * } ( \cdot \mid y )$ , so the conclusion holds. The audio side is entirely symmetric. □

## A.5 AUTOMATIC COMPATIBILITY OF THE TWO OPTIMAL CONDITIONAL KERNELS

Lemma 3 (Marginals and regular conditional kernels of $P ^ { * } )$ .

$$
P _ { Y } ^ { * } ( \mathrm { d } y ) = \frac { Z _ { v } ( y ) } { Z } P _ { 0 , Y } ( \mathrm { d } y ) , \qquad P _ { X } ^ { * } ( \mathrm { d } x ) = \frac { Z _ { a } ( x ) } { Z } P _ { 0 , X } ( \mathrm { d } x ) .
$$

Hence $P _ { Y } ^ { * }$ is equivalent to $P _ { 0 , Y }$ , and $P _ { X } ^ { * }$ is equivalent to $P _ { 0 , X }$ . The selected $K _ { v } ^ { * } , K _ { a } ^ { * }$ are, respectively, versions ofthe video and audio regular conditional distributions of $P ^ { * }$

Proof. For any $B \in B _ { \mathcal { V } }$ , using the decomposition of $P _ { 0 }$ via $K _ { v } ^ { 0 }$ :

$$
P _ { Y } ^ { * } ( B ) = Z ^ { - 1 } \int _ { B } \int _ { \mathcal X } e ^ { r / \beta } K _ { v } ^ { 0 } ( \mathrm { d } x \mid y ) P _ { 0 , Y } ( \mathrm { d } y ) = Z ^ { - 1 } \int _ { B } Z _ { v } ( y ) P _ { 0 , Y } ( \mathrm { d } y ) .
$$

Thus $P _ { Y } ^ { * } ( \mathrm { d } y ) = Z ^ { - 1 } Z _ { v } ( y ) P _ { 0 , Y } ( \mathrm { d } y ) . ~ Z _ { v }$ has strictly positive finite upper and lower bounds, so the two audio marginals are equivalent. The video marginal formula follows analogously.

Next, taking $A \in B _ { X } , B \in B _ { y }$ , we have

$$
\int _ { B } K _ { v } ^ { * } ( A \mid y ) P _ { Y } ^ { * } ( \mathrm { d } y ) = Z ^ { - 1 } \int _ { B } \int _ { A } \boldsymbol { e } ^ { r / \beta } K _ { v } ^ { 0 } ( \mathrm { d } x \mid y ) P _ { 0 , Y } ( \mathrm { d } y ) = P ^ { * } ( A \times B ) .
$$

This is precisely the definition of $K _ { v } ^ { * }$ being a regular conditional distribution of $P ^ { * } , K _ { a } ^ { * }$ is symmetric. □

## A.6 ALTERNATING KERNELS, MARGINAL OPERATORS, AND THE SYSTEMATIC-SCANCHAIN

## A.6.1 KERNEL–MEASURE MIXING

For $\mu \in \mathcal P ( \mathcal Y ) , \nu \in \mathcal P ( \mathcal X )$ ), define the joint distributions of the two phases

$$
M _ { v } \mu ( \mathrm { d } x , \mathrm { d } y ) = K _ { v } ^ { * } ( \mathrm { d } x \mid y ) \mu ( \mathrm { d } y ) , \qquad M _ { a } \nu ( \mathrm { d } x , \mathrm { d } y ) = \nu ( \mathrm { d } x ) K _ { a } ^ { * } ( \mathrm { d } y \mid x ) .
$$

Define the corresponding marginal operators

$$
V \mu ( \mathrm { d } x ) = \int _ { \mathcal { y } } K _ { v } ^ { * } ( \mathrm { d } x \mid y ) \mu ( \mathrm { d } y ) , \quad \quad W \nu ( \mathrm { d } y ) = \int _ { \mathcal { x } } K _ { a } ^ { * } ( \mathrm { d } y \mid x ) \nu ( \mathrm { d } x ) .
$$

Here the Y-marginal of $M _ { v } \mu$ is $\mu ,$ and its X-marginal is $V \mu ;$ the X-marginal of $M _ { a } \nu$ is $\nu ,$ and its Y-marginal is Wν. Let $F = W V : \mathcal { P } ( \mathcal { V } )  \mathcal { P } ( \mathcal { \bar { y } } )$

## A.6.2 ONE COMPLETE GIBBS SWEEP

Starting from any joint law $Q \in { \mathcal { P } } ( { \mathcal { X } } \times { \mathcal { Y } } )$ , first keep y and resample $x ^ { \prime } \sim K _ { v } ^ { * } ( \cdot \mid y )$ , then keep $x ^ { \prime }$ and resample $y ^ { \prime } \overset { \cdot \cdot } { \sim } K _ { a } ^ { * } ( \cdot \mid x ^ { \prime } )$ . Denote one round of transition by $\bar { G ; }$ then

$$
Q G = M _ { a } ( V Q _ { Y } ) , \qquad ( Q G ) _ { Y } = W V Q _ { Y } = F Q _ { Y } .
$$

In particular, the joint law after one round depends only on the Y-marginal of the input joint law.

## A.6.3 INVARIANCE OF $P ^ { * }$

By Lemma 3, $P ^ { * } = M _ { v } P _ { Y } ^ { * } = M _ { a } P _ { X } ^ { * }$ , and moreover $P _ { X } ^ { * } = V P _ { Y } ^ { * } , P _ { Y } ^ { * } = W P _ { X } ^ { * }$ . Therefore

$$
P ^ { * } G = M _ { a } ( V P _ { Y } ^ { * } ) = M _ { a } P _ { X } ^ { * } = P ^ { * } .
$$

So $P ^ { * }$ is a stationary distribution of the alternating Gibbs chain. This step requires no ergodicity or convergence assumptions.

## A.7 DOBRUSHIN CONTRACTION AND THE BASIC CONTRACTION LEMMA

## A.7.1 TOTAL VARIATION AND DOBRUSHIN COEFFICIENTS

Define

$$
\| \mu - \mu ^ { \prime } \| _ { \mathrm { T V } } = \operatorname* { s u p } _ { A } | \mu ( A ) - \mu ^ { \prime } ( A ) | \in [ 0 , 1 ] ,
$$

$$
\delta _ { v } = \operatorname* { s u p } _ { y , y ^ { \prime } } \Vert K _ { v } ^ { * } ( \cdot \mid y ) - K _ { v } ^ { * } ( \cdot \mid y ^ { \prime } ) \Vert _ { \mathrm { T V } } , \qquad \delta _ { a } = \operatorname* { s u p } _ { x , x ^ { \prime } } \Vert K _ { a } ^ { * } ( \cdot \mid x ) - K _ { a } ^ { * } ( \cdot \mid x ^ { \prime } ) \Vert _ { \mathrm { T V } } .
$$

Assumption 2 (Verifiable weak coupling/contraction condition H1).

$$
\rho = \delta _ { v } \delta _ { a } < 1 .
$$

This condition is used to derive uniqueness and convergence. The supremum here is taken with respect to the full-domain conditional kernel versions selected above; H1 is an assumption on these versions.

## A.7.2 MARKOV KERNEL CONTRACTION LEMMA

Lemma 4 (Dobrushin contraction). Let $K : S \times B _ { T }   0 , 1 $ ] be a probability kernel, and $\delta ( K ) =$ $\begin{array} { r } { \operatorname* { s u p } _ { z , z ^ { \prime } } \| K ( \cdot \mid z ) - K ( \cdot \mid z ^ { \prime } ) \| _ { \mathrm { T V } } } \end{array}$ . Thenfor any $\eta , \eta ^ { \prime } \in \bar { \mathcal { P } } ( S )$

$$
\| \eta K - \eta ^ { \prime } K \| _ { \mathrm { T V } } \leq \delta ( K ) \| \eta - \eta ^ { \prime } \| _ { \mathrm { T V } } .
$$

Proof. Let $\sigma = \eta - \eta ^ { \prime } . \mathrm { I f } \ t = \lVert \eta - \eta ^ { \prime } \rVert _ { \mathrm { T V } } = 0$ , the conclusion is obvious. Otherwise the Jordan decomposition of σ satisfies $\sigma ^ { + } = t p , \sigma ^ { - } = t q$ , where $p , q$ are probability measures. For any $C \in B _ { T }$ , let $f _ { C } ( z ) = K ( C \mid z )$ ; then

$$
| ( \eta K - \eta ^ { \prime } K ) ( C ) | = t \left| \int f _ { C } ~ \mathrm { d } p - \int f _ { C } ~ \mathrm { d } q \right| \le t ( \operatorname* { s u p } _ { z } f _ { C } ( z ) - \operatorname* { i n f } _ { z } f _ { C } ( z ) ) \le t \delta ( K ) .
$$

Taking the supremum over C yields the conclusion.

## A.7.3 APPLICATION TO V, W

$$
\begin{array} { r l } & { \| \boldsymbol { V } \mu - \boldsymbol { V } \mu ^ { \prime } \| _ { \mathrm { T V } } \leq \delta _ { v } \| \mu - \mu ^ { \prime } \| _ { \mathrm { T V } } , \qquad \| \boldsymbol { W } \nu - \boldsymbol { W } \nu ^ { \prime } \| _ { \mathrm { T V } } \leq \delta _ { a } \| \nu - \nu ^ { \prime } \| _ { \mathrm { T V } } , } \\ & { \qquad \| \boldsymbol { F } \mu - \boldsymbol { F } \mu ^ { \prime } \| _ { \mathrm { T V } } \leq \rho \| \mu - \mu ^ { \prime } \| _ { \mathrm { T V } } , \qquad \rho = \delta _ { v } \delta _ { a } < 1 . } \end{array}
$$

## A.8 EXISTENCE, UNIQUENESS, AND IDENTIFICATION OF THE SELF-CONSISTENT MARGINALS

Lemma 5 (Unique self-consistent marginals). Under H1, $F = W V$ is a contraction mapping on the complete metric space $( \mathcal { P } ( \mathcal { V } ) , \Vert \cdot \Vert _ { \mathrm { T V } } )$ . Hence there exists a unique $\mu ^ { \ast } \in \mathcal { P } ( \mathcal { V } )$ satisfying $F \mu ^ { * } = \mu ^ { * }$ . Let $\nu ^ { * } = V \mu ^ { * } ;$ then $( \mu ^ { * } , \nu ^ { * } )$ is the unique solution ofthe equations $\mu = W \nu , \nu = V \mu .$

Proof. Finite signed measures form a Banach space under the total variation norm, and $\mathcal { P } ( \mathcal { V } )$ is a closed subset thereof, so $( \mathcal { P } ( \mathcal { V } ) , \mathrm { T V } )$ is complete. By Lemma 4, the Lipschitz constant of F is at most $\rho < 1$ . The Banach fixed point theorem yields the unique fixed point $\mu ^ { * } ;$ ; and for any $\mu _ { 0 } \in \mathcal { P } ( \mathcal { V } )$

$$
\begin{array} { r } { \| F ^ { n } \mu _ { 0 } - \mu ^ { * } \| _ { \mathrm { T V } } \leq \rho ^ { n } \| \mu _ { 0 } - \mu ^ { * } \| _ { \mathrm { T V } } . } \end{array}
$$

Let $\nu ^ { * } = V \mu ^ { * } ;$ ; then $W \nu ^ { * } = W V \mu ^ { * } = F \mu ^ { * } = \mu ^ { * } , \operatorname { s o } \left( \mu ^ { * } , \nu ^ { * } \right)$ is self-consistent. Conversely, any self-consistent pair $( \mu , \nu )$ satisfies $\mu = W \nu = W V \mu = F \mu$ , hence $\mu = \mu ^ { * } ;$ ; and then $\nu = V \mu =$ $V \mu ^ { * } = \nu ^ { * }$ □

Remark 1 (Identification with the marginals of the jointly optimal distribution). By Lemma 3, the two marginals of $P ^ { * }$ satisfy

$$
P _ { X } ^ { * } = V P _ { Y } ^ { * } , \qquad P _ { Y } ^ { * } = W P _ { X } ^ { * } = W V P _ { Y } ^ { * } = F P _ { Y } ^ { * } .
$$

So $P _ { Y } ^ { * }$ is a fixed point of F. By uniqueness, $\mu ^ { * } = P _ { Y } ^ { * } , \nu ^ { * } = P _ { X } ^ { * }$ . Thus the self-consistent marginals not only exist and are unique, but are exactly the marginals of the jointly KL-optimal distribution.

## A.9 THE UNIQUE STATIONARY JOINT DISTRIBUTION

Lemma 6 (Unique stationary distribution of the alternating chain). Under H0, bounded reward, and H1, the stationary distribution of the systematic-scan Gibbs chain G exists and is unique, and the unique stationary distribution is $P ^ { * }$

Proof. Existence has already been proved in A.6.3: $P ^ { * } G = P ^ { * }$

We now prove uniqueness. Let $Q$ be any stationary distribution, i.e., $Q = Q G$ . By the structure of one round of updates,

$$
Q = M _ { a } ( V Q _ { Y } ) , \qquad Q _ { Y } = W V Q _ { Y } = F Q _ { Y } .
$$

So $Q _ { Y }$ is a fixed point of $F$ . Lemma 5 gives $Q _ { Y } = \mu ^ { * } = P _ { Y } ^ { * }$ . Substituting back into the joint distribution formula:

$$
Q = M _ { a } ( V \mu ^ { * } ) = M _ { a } \nu ^ { * } = M _ { a } P _ { X } ^ { * } = P ^ { * } .
$$

Therefore $P ^ { * }$ is the unique stationary distribution.

Remark 2 (Strict distinction among optimality, stationarity, and convergence). That $P ^ { * }$ is the unique joint optimum follows from the KL variational identity and does not require $\rho < 1$ ; the conditional compatibility of $K _ { v } ^ { * } , K _ { a } ^ { * }$ follows from the common $P _ { 0 }$ and the common reward tilting, and does not require $\rho < 1 ;$ ; that ${ \check { P } } ^ { * }$ is a stationary distribution follows from the two conditional kernels of $P ^ { * }$ , and does not require $\rho < 1 ;$ ; uniqueness of the stationary distribution follows from the strict contraction of $F = W { \dot { V } }$ , and requires $\rho < 1$ ; geometric convergence from any initial value follows from Banach iteration and the TV isometry, and requires $\rho < 1$

## A.10 GEOMETRIC ROLLOUT CONVERGENCE FROM ARBITRARY INITIAL LAWS

## A.10.1 THE TWO SAMPLING PHASES

Take any initial joint law $Q _ { 0 }$ , and let $\mu _ { 0 } = Q _ { 0 , Y }$ . For $n \geq 1$ , recursively define

$$
Q _ { n } ^ { v } = M _ { v } \mu _ { n - 1 } , \qquad \nu _ { n } = V \mu _ { n - 1 } ,
$$

$$
Q _ { n } ^ { a } = M _ { a } \nu _ { n } , \qquad \mu _ { n } = W \nu _ { n } = F \mu _ { n - 1 } .
$$

$Q _ { n } ^ { v }$ is the joint law after the video resampling; $Q _ { n } ^ { a }$ is the joint law after the subsequent audio resampling, and is also the chain distribution after one complete sweep.

## A.10.2 TV ISOMETRIC EMBEDDING OF THE MIXING OPERATORS

$M _ { \tau }$ keeps y as is and only randomly generates x. Hence the non-expansiveness of Markov kernels gives $\lVert M _ { v } \mu - M _ { v } \mu ^ { \prime } \rVert _ { \mathrm { T V } } \leq \lVert \mu - \mu ^ { \prime } \rVert _ { \mathrm { T V } }$ , while taking the Y -marginal gives the reverse inequality. Thus

$$
\lVert M _ { v } \mu - M _ { v } \mu ^ { \prime } \rVert _ { \mathrm { T V } } = \lVert \mu - \mu ^ { \prime } \rVert _ { \mathrm { T V } } .
$$

Similarly, $\| M _ { a } \nu - M _ { a } \nu ^ { \prime } \| _ { \mathrm { T V } } = \| \nu - \nu ^ { \prime } \| _ { \mathrm { T V } }$

Theorem 1 (Geometric TV convergence of the two phases). Let $\rho = \delta _ { v } \delta _ { a } < 1$ . Then for any $Q _ { 0 }$ and $n \geq 1 .$

$$
\| \mu _ { n } - P _ { Y } ^ { * } \| _ { \mathrm { T V } } \leq \rho ^ { n } \| \mu _ { 0 } - P _ { Y } ^ { * } \| _ { \mathrm { T V } } ,
$$

$$
\begin{array} { r } { \| \nu _ { n } - P _ { X } ^ { * } \| _ { \mathrm { T V } } \leq \delta _ { v } \rho ^ { n - 1 } \| \mu _ { 0 } - P _ { Y } ^ { * } \| _ { \mathrm { T V } } , } \end{array}
$$

$$
\lVert Q _ { n } ^ { v } - P ^ { * } \rVert _ { \mathrm { T V } } \leq \rho ^ { n - 1 } \lVert \mu _ { 0 } - P _ { Y } ^ { * } \rVert _ { \mathrm { T V } } ,
$$

$$
\begin{array} { r } { \| Q _ { n } ^ { a } - P ^ { * } \| _ { \mathrm { T V } } \leq \delta _ { v } \rho ^ { n - 1 } \| \mu _ { 0 } - P _ { Y } ^ { * } \| _ { \mathrm { T V } } . } \end{array}
$$

When $\rho = 0$ and $n = 1$ , we take $\rho ^ { 0 } = 1$ by convention.

Proof. The first inequality is the Banach iteration. The second follows from $\nu _ { n } = V \mu _ { n - 1 }$ and the $\delta _ { v }$ -contraction of $V .$ Since $P ^ { * } = M _ { v } P _ { Y } ^ { * } = M _ { a } P _ { X } ^ { * }$ , combining the two isometry identities yields the third and fourth inequalities. □

## A.11 MAIN THEOREM

Theorem 2 (Idealized AV-GRPO Gibbs theorem). Fix c. Let X , Y be Polish trajectory spaces, and suppose H0 holds; r is bounded measurable, and $\beta > 0$ . Define $P ^ { * } , K _ { v } ^ { * } , K _ { a } ^ { * }$ from $P _ { 0 }$ and r. Assume the selectedfull-domain versions satisfy $\rho = \delta _ { v } \delta _ { a } < 1$ . Then:

1. $P ^ { * }$ is the unique global optimum of the joint objective $J ( P ) = \mathbb { E } _ { P } [ r ] - \beta D _ { \mathrm { K L } } ( P ~ \| ~ P _ { 0 } )$ , with optimal value β log Z;

2. $K _ { v } ^ { * } , K _ { a } ^ { * }$ are, respectively, the unique optimal kernels ofthe conditional KL subproblems when the counterpart trajectory is fixed, and they are two versions ofthe regular conditional distributions $o f P ^ { * } { } _ { ; }$

3. the alternating Gibbs chain has a unique stationary distribution $P ^ { * } ,$

4. the self-consistent marginal equations $\mu = W \nu , \nu = V \mu$ have a unique solution $( P _ { Y } ^ { * } , P _ { X } ^ { * } ) ,$

5. starting from any initial joint law, the joint laws of the video phase and the audio phase both converge to $P ^ { * }$ in total variation at the explicit rates ofTheorem 1.

Proof. Conclusion (1) is Lemma 1. Conclusion (2) follows from Lemma 2 and Lemma 3. The invariance of $P ^ { * }$ follows from A.6.3; Lemma 6 then uses Dobrushin contraction to prove its uniqueness, giving (3). Lemma 5 identifies the unique self-consistent marginals, giving (4). Finally, Theorem 1 gives the geometric TV convergence of both phases, giving (5). All conclusions have been proved by the preceding lemmas. □

## A.12 CONNECTION WITH ALTERNATING TRAINING

Assumption 3 (exact-kernel non-interference oracle H2). The video tower policy class contains the selected full-domain kernel $K _ { v } ^ { * }$ , and the audio tower policy class contains $K _ { a } ^ { * } .$ . The video update oracle returns, for each $y ,$ the unique optimal kernel $\bar { K _ { v } ^ { * } } ( \cdot \mid \bar { y } )$ of Lemma 2; the audio update oracle returns, for each $x , K _ { a } ^ { * } ( \cdot \mid x )$ . Furthermore, we assume independently that updating the parameters of either tower does not change the conditional kernel already realized by the other tower.

Corollary 1 (Idealized training–inference consistency). Under the main theorem and H2, after the video tower and the audio tower each complete one exact update, the two policy kernels arefixed as $K _ { v } ^ { * } , K _ { a } ^ { * }$ , respectively. Thereafter, thejoint distributions ofthe video phase and the audio phase ofthe training rollouts, as well as the inference distribution adopting the same full-trajectory alternating resampling protocol, all converge geometrically in TV to ${ \bar { P } } ^ { * }$ from any initial joint law. The limiting joint distribution attains the global optimal value β log Z ofJ.

Proof. By Lemma 2, the unique optimal solutions of the pointwise conditional subproblems are $K _ { v } ^ { * }$ and $K _ { a } ^ { * }$ , respectively. H2 guarantees that the two towers realize these full-domain kernels after updating, and that subsequent updates do not change the other already-realized kernel. Therefore, after each tower has been updated once, all subsequent sampling is described exactly by the fixed operators $V , W , M _ { v } , M _ { a }$ . Theorem 2(5) directly gives the geometric TV convergence of both phases and the inference chain to $P ^ { * }$ ; Theorem 2(1) gives the objective value of $P ^ { * }$ □

## A.13 CONCLUSION

Under the common reference joint law H0, bounded reward, $\beta > 0$ , and the Dobrushin contraction condition $\rho = \delta _ { v } \delta _ { a } < 1$ , this section has proved:

1. the exponentially tilted joint law $P ^ { * }$ is the unique global optimum of the joint KL-regularized objective, with optimal value β log $Z ;$

2. the two conditional Gibbs optimal kernels $K _ { v } ^ { * } , K _ { a } ^ { * }$ are the unique optimal kernels of the conditional subproblems, respectively, are automatically compatible, and are the regular conditional distributions of $P ^ { * }$ ;

3. the systematic-scan Gibbs chain with kernels $K _ { v } ^ { * } , K _ { a } ^ { * }$ has a unique stationary distribution $P ^ { * }$

4. the self-consistent marginal equations have a unique solution, which is exactly the two marginals of $P ^ { * }$ ;

5. starting from any initial joint law, the joint laws of the two sampling phases converge geometrically in total variation to $P ^ { * }$ , with explicit rates;

6. if the model implements $K _ { v } ^ { * } , K _ { a } ^ { * }$ via a non-interfering exact-kernel oracle, then both the training rollouts and the inference distribution possess the same geometric convergence properties.

The proof is closed under the idealized assumptions: optimality comes from the KL variational identity, compatibility comes from the common reference joint law, and uniqueness and convergence come from Dobrushin contraction.

## B TRAINING DETAILS

All training runs, including LoRA and full fine-tuning, are conducted on a single 8-GPU A800 (80 GB) node. All prompts in the training dataset are fully randomly shuffled (rather than sampled in category-ordered sequences). Given a sampled prompt, the LTX-2.3 model generates audio-video clips with resolution $5 4 4 \times 9 6 0$ , 97 frames, and 24 FPS via 15-step denoising inference using the default random seed (42), producing 8 samples per prompt. The composition of the reward used to evaluate each audio-video sample is described in Appendix D. For LoRA training, the learning rate starts from $3 \times 1 0 ^ { - 6 }$ and decays to zero following a cosine schedule for 480 training steps (The total run is 1440 steps, i.e., all data is seen once by each of the two towers: $5 7 6 0 \times 2 7 8 = 1 4 4 0 )$ The noise level for sampling of the video tower is set to 0.02 with a KL coefficient of 0.01; for the audio tower, the sampling noise level is 0.8 and the KL coefficient is 0.002. GDPO is trained for 480 steps under identical hyperparameters and the same data ordering. Since GDPO directly generates paired audio–video samples and updates both towers simultaneously, the same data is used to train each tower twice. For full fine-tuning, the learning rate starts at $\mathrm { i \times 1 0 ^ { - 6 } }$ with cosine decay to zero over 480 steps (keeping the same total step count). The noise levels and KL coefficients for both towers are kept identical to those used in LoRA training. Alternating optimization is adopted: training switches between the two towers every 4 steps, with 8 prompts consumed per step and the video tower updated first. Within each round, the video tower is first trained on $4 \times 8 = 3 2$ prompts. Training then switches to the audio tower, which is updated using the same set of 32 prompts, before proceeding to the next round with fresh data.

## C FLOW-GRPO AND LONGCAT-VIDEO FRAMEWORK

In this section, we present the complete details of the Flow-GRPO (Liu et al., 2026a) and the LongCat-Video (Team et al., 2025) framework.

## C.1 FLOW-GRPO

Flow-GRPO converts the deterministic flow ODE into an equivalent SDE that preserves identical marginal distributions with the original model at all timesteps. The reversed SDE for rectified flow is formulated as:

$$
d \pmb { x } _ { t } = \left[ \pmb { v } _ { t } ( \pmb { x } _ { t } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \big ( \pmb { x } _ { t } + ( 1 - t ) \pmb { v } _ { t } ( \pmb { x } _ { t } ) \big ) \right] d t + \sigma _ { t } d \pmb { w } .\tag{13}
$$

where dw denotes the Wiener process increment, and $\sigma _ { t }$ is the time-dependent noise schedule. Flow-GRPO sets $\begin{array} { r } { \sigma _ { t } = a \sqrt { \frac { t } { 1 - t } } . } \end{array}$ , where scalar hyperparameter a controls the stochastic intensity. Applying Euler-Maruyama discretization to the SDE yields the iterative update formula:

$$
{ { x } _ { t + \Delta t } } = { { x } _ { t } } + \left[ { { v } _ { \theta } } ( { { x } _ { t } } , t ) + \frac { { { \sigma } _ { t } ^ { 2 } } } { 2 t } { { \left( { { x } _ { t } } + { { ( 1 - t ) } { v } _ { \theta } } ( { { x } _ { t } } , t ) \right) } } \right] \Delta t + { { \sigma } _ { t } } \sqrt { \Delta t } \epsilon ,\tag{14}
$$

with $\epsilon \sim \mathcal { N } ( 0 , I )$ introducing Gaussian noise. From Equation (2), the transition distribution $\pi _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t + \Delta t } \mid \mathbf { x } _ { t } , \boldsymbol { c } )$ is an isotropic Gaussian distribution. Thus, the KL divergence between current policy and reference policy admits a closed-form solution:

$$
D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) = \frac { \Delta t } { 2 } \left( \frac { \sigma _ { t } ( 1 - t ) } { 2 t } + \frac { 1 } { \sigma _ { t } } \right) ^ { 2 } \| \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) - \pmb { v } _ { \mathrm { r e f } } ( \pmb { x } _ { t } , t ) \| ^ { 2 } .\tag{15}
$$

Flow-GRPO adopts the group-relative formulation for advantage estimation. Given condition c, the model samples a group of G reverse trajectories. All timesteps of one trajectory share the identical advantage computed via intra-group reward normalization:

$$
\hat { A } _ { t } ^ { i } = \frac { R ( { \pmb x } _ { 0 } ^ { i } , c ) - \mathrm { m e a n } ( \{ R ( { \pmb x } _ { 0 } ^ { i } , c ) \} _ { i = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R ( { \pmb x } _ { 0 } ^ { i } , c ) \} _ { i = 1 } ^ { G } ) } .\tag{16}
$$

The importance sampling ratio characterizes the probability ratio between old and new policy:

$$
r _ { t } ^ { i } ( \theta ) = \frac { p _ { \theta } ( \pmb { x } _ { t - 1 } ^ { i } \mid \pmb { x } _ { t } ^ { i } , c ) } { p _ { \theta _ { \mathrm { o l d } } } ( \pmb { x } _ { t - 1 } ^ { i } \mid \pmb { x } _ { t } ^ { i } , c ) } .\tag{17}
$$

Flow-GRPO optimizes the policy by maximizing the following objective:

$$
\mathcal { I } _ { \mathrm { H o w - G R P O } } ( \theta ) = \mathbb { E } _ { c , \pi ^ { \otimes d } , \{ \mathbf { x } ^ { i } \} _ { i = 1 } ^ { G } } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \left( \operatorname* { m i n } \left( r _ { t } ^ { i } \hat { A } _ { t } ^ { i } , \mathrm { c l i p } ( r _ { t } ^ { i } , 1 - \varepsilon , 1 + \varepsilon ) \hat { A } _ { t } ^ { i } \right) \right. \right.\tag{18}
$$

## C.2 LONGCAT-VIDEO

The gradient of the policy loss $\mathcal { L } _ { \mathrm { p o l i c y } } ( \theta ) = r _ { t } ^ { i } ( \theta ) \hat { A } _ { t } ^ { i }$ with respect to parameters $\theta$ can be derived. The gradient computation proceeds as follows:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p o l i c y } } ( \boldsymbol { \theta } ) = \hat { A } _ { t } ^ { i } \nabla _ { \boldsymbol { \theta } } r _ { t } ^ { i } ( \boldsymbol { \theta } ) .\tag{19}
$$

$$
\nabla _ { \theta } r _ { t } ^ { i } ( \theta ) = \frac { p _ { \theta } \left( x _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c \right) } { p _ { \theta _ { \mathrm { o u t } } } \left( x _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c \right) } \nabla _ { \theta } \log p _ { \theta } \left( x _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c \right) = \nabla _ { \theta } \log p _ { \theta } \left( x _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c \right) .\tag{20}
$$

Combining these results yields the policy gradient:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p o l i c y } } ( \boldsymbol { \theta } ) = \hat { A } _ { t } ^ { i } r _ { t } ^ { i } ( \boldsymbol { \theta } ) \nabla _ { \boldsymbol { \theta } } \log p _ { \boldsymbol { \theta } } \left( x _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , c \right) .\tag{21}
$$

The score function $\nabla _ { \theta }$ log p<sub>θ</sub> $\left( \boldsymbol { x } _ { t - 1 } \mid \boldsymbol { x } _ { t } , \boldsymbol { c } \right)$ is computed next. The conditional distribution is Gaussian:

$$
p _ { \theta } \left( x _ { t - 1 } \mid x _ { t } , c \right) = \mathcal { N } \left( x _ { t - 1 } ; \mu _ { \theta } \left( x _ { t } , t , c \right) , \sigma _ { t } ^ { 2 } \Delta t \mathbf { I } \right) .\tag{22}
$$

$$
\nabla _ { \boldsymbol { \theta } } \log { p _ { \boldsymbol { \theta } } } = \frac { 1 } { \sigma _ { t } ^ { 2 } \Delta t } \left( x _ { t - 1 } - \mu _ { \boldsymbol { \theta } } \right) \cdot \nabla _ { \boldsymbol { \theta } } \mu _ { \boldsymbol { \theta } } .\tag{23}
$$

From the SDE sampling process, the following reparameterization holds:

$$
x _ { t - 1 } = \mu _ { \theta } + \sigma _ { t } \sqrt { \Delta t } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{24}
$$

Substituting:

$$
\nabla _ { \theta } \log { p _ { \theta } } = \frac { 1 } { \sigma _ { t } ^ { 2 } \Delta t } \left( \sigma _ { t } \sqrt { \Delta t } \boldsymbol { \epsilon } \right) \cdot \nabla _ { \theta } \mu _ { \theta } = \frac { 1 } { \sigma _ { t } \sqrt { \Delta t } } \boldsymbol { \epsilon } \cdot \nabla _ { \theta } \mu _ { \theta } .\tag{25}
$$

$$
\mu _ { \theta } = x _ { t } + \left[ v _ { \theta } \left( x _ { t } , t , c \right) + \frac { \sigma _ { t } ^ { 2 } } { 2 t } \left( x _ { t } + ( 1 - t ) v _ { \theta } \left( x _ { t } , t , c \right) \right) \right] ( - \Delta t )\tag{26}
$$

Simplifying the drift term:

$$
\begin{array} { l } { \displaystyle \mathrm { d r i f t } = v _ { \theta } + \frac { \sigma _ { t } ^ { 2 } } { 2 t } x _ { t } + \frac { \sigma _ { t } ^ { 2 } } { 2 t } ( 1 - t ) v _ { \theta } } \\ { \displaystyle = v _ { \theta } \left( 1 + \frac { \sigma _ { t } ^ { 2 } ( 1 - t ) } { 2 t } \right) + \frac { \sigma _ { t } ^ { 2 } } { 2 t } x _ { t } } \end{array}\tag{27}
$$

Thus:

$$
\mu _ { \theta } = x _ { t } - \Delta t \cdot \mathrm { d r i f t }\tag{28}
$$

Taking the gradient with respect to θ (noting that $x _ { t }$ is constant):

$$
\nabla _ { \theta } \mu _ { \theta } = - \Delta t \cdot \nabla _ { \theta } \mathrm { d r i f t } = - \Delta t \cdot \left( 1 + \frac { \sigma _ { t } ^ { 2 } ( 1 - t ) } { 2 t } \right) \nabla _ { \theta } v _ { \theta }\tag{29}
$$

Substituting into $\nabla _ { \theta }$ log p<sub>θ</sub>:

$$
\begin{array} { l } { \displaystyle \nabla _ { \theta } \log p _ { \theta } = \frac { 1 } { \sigma _ { t } \sqrt { \Delta t } } \epsilon \cdot \left[ - \Delta t \cdot \left( 1 + \frac { \sigma _ { t } ^ { 2 } ( 1 - t ) } { 2 t } \right) \nabla _ { \theta } v _ { \theta } \right] } \\ { \displaystyle = - \frac { \sqrt { \Delta t } } { \sigma _ { t } } \left( 1 + \frac { \sigma _ { t } ^ { 2 } ( 1 - t ) } { 2 t } \right) \epsilon \cdot \nabla _ { \theta } v _ { \theta } } \end{array}\tag{30}
$$

Therefore, the gradient of the policy loss is:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p o l i c y } } ( \boldsymbol { \theta } ) = \hat { A } _ { t } ^ { i } r _ { t } ^ { i } ( \boldsymbol { \theta } ) \cdot \left[ - \frac { \sqrt { \Delta t } } { \sigma _ { t } } \left( 1 + \frac { \sigma _ { t } ^ { 2 } ( 1 - t ) } { 2 t } \right) \boldsymbol { \epsilon } \cdot \nabla _ { \boldsymbol { \theta } } \boldsymbol { v } _ { \boldsymbol { \theta } } \right]\tag{31}
$$

Substituting $a = 1$ and $\begin{array} { r } { \sigma _ { t } = \sqrt { \frac { t } { 1 - t } \left( { \bf s } _ { 0 } \sigma _ { t } ^ { 2 } = \frac { t } { 1 - t } \right) } } \end{array}$ . The coefficient term is computed as:

$$
1 + { \frac { \sigma _ { t } ^ { 2 } ( 1 - t ) } { 2 t } } = 1 + { \frac { { \frac { t } { 1 - t } } \cdot ( 1 - t ) } { 2 t } } = 1 + { \frac { 1 } { 2 } } = { \frac { 3 } { 2 } }\tag{32}
$$

And the scaling term:

$$
{ \frac { { \sqrt { \Delta t } } } { \sigma _ { t } } } = { \frac { { \sqrt { \Delta t } } } { \sqrt { \frac { t } { 1 - t } } } } = { \sqrt { \Delta t } } \cdot { \sqrt { \frac { 1 - t } { t } } } = { \sqrt { \frac { \Delta t ( 1 - t ) } { t } } }\tag{33}
$$

Substituting these simplifications gives the final policy gradient expression:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p o l i c y } } ( \boldsymbol { \theta } ) = - \frac { 3 } { 2 } \hat { A } _ { t } ^ { i } \sqrt { \frac { \Delta t ( 1 - t ) } { t } } \boldsymbol { \epsilon } \cdot \nabla _ { \boldsymbol { \theta } } \boldsymbol { v } _ { \boldsymbol { \theta } }\tag{34}
$$

A reweighting coefficient is introduced, defined as:

$$
\lambda _ { \mathrm { p o l i c y } } ( t , \Delta t ) = \kappa ( t , \Delta t ) ^ { - 1 } = \sqrt { \frac { t } { \Delta t ( 1 - t ) } }\tag{35}
$$

The reweighted policy loss becomes:

$$
\mathcal { L } _ { \mathrm { p o l i c y , r e w e i g h t e d } } ( \theta ) = \lambda _ { \mathrm { p o l i c y } } ( t , \Delta t ) \cdot \mathcal { L } _ { \mathrm { p o l i c y } } ( \theta )\tag{36}
$$

This yields the modified gradient:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p o l i c y , r e w e i g h t e d } } ( \boldsymbol { \theta } ) = - \frac { 3 } { 2 } \hat { \boldsymbol { A } } _ { t } ^ { i } \cdot \boldsymbol { \epsilon } \cdot \nabla _ { \boldsymbol { \theta } } v _ { \boldsymbol { \theta } }\tag{37}
$$

Similarly, the gradient of the KL divergence term can be derived as:

$$
\nabla _ { \theta } D _ { \mathrm { K L } } ( \theta ) = \Delta t \cdot \frac { 9 } { 4 } \cdot \frac { 1 - t } { t } \cdot ( \boldsymbol { v } _ { \theta } - \boldsymbol { v } _ { \mathrm { r e f } } ) \cdot \nabla _ { \theta } \boldsymbol { v } _ { \theta }\tag{38}
$$

This expression reveals that the KL loss gradient suffers from the same scaling issues as the policy loss gradient. To address this, a KL reweighting coefficient is also introduced:

$$
\lambda _ { \mathrm { K L } } ( t , \Delta t ) = k _ { \mathrm { K L } } ( t , \Delta t ) ^ { - 1 } = \frac { t } { \Delta t ( 1 - t ) }\tag{39}
$$

The reweighted KL loss becomes:

$$
{ \mathcal { L } } _ { \mathrm { K L , r e w e i g h t e d } } ( \theta ) = \lambda _ { \mathrm { K L } } ( t , \Delta t ) \cdot D _ { \mathrm { K L } } ( \theta )\tag{40}
$$

yielding the simplified gradient:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { K L , r e w e i g h t e d } } ( \theta ) = \frac { 9 } { 4 } \cdot ( v _ { \theta } - v _ { \mathrm { r e f } } ) \cdot \nabla _ { \theta } v _ { \theta }\tag{41}
$$

Based on the reweighting coefficients for the policy loss and KL loss, the revised GRPO objective function is as follows:

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { c \sim \mathcal { C } , t ^ { \prime } \sim \mathcal { U } ( 0 , T ^ { \prime } - 1 ) } , \displaystyle \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \Bigg ( \lambda _ { \mathrm { p o i i c y } } \left( \frac { t ^ { \prime } } { T } , \Delta \frac { t ^ { \prime } } { T } \right) \cdot \mathcal { L } _ { \mathrm { p o i i c y } } ( \theta ) } \\ { - \beta \lambda _ { \mathrm { K L } } \left( \frac { t ^ { \prime } } { T } , \Delta \frac { t ^ { \prime } } { T } \right) \cdot D _ { \mathrm { K L } } \left( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } \right) \Bigg ) \Bigg ] } \end{array}\tag{42}
$$

## D REWARD MODELS AND REWARD COMPOSITION

For video-tower optimization, we use VideoAlign (Liu et al., 2026b), CLIP (Radford et al., 2021), and DeSync (Iashin et al., 2024) to assess video quality, video–prompt alignment, and audio–video synchronization, respectively. For audio-tower optimization, we use Audiobox Aesthetics (Tjandra et al., 2025), CLAP (Wu et al., 2023), and DeSync to assess audio quality, audio–prompt alignment, and audio–video synchronization, respectively.

Following GDPO, we first compute group-wise normalized advantages for each individual metric, then sum them to obtain a cumulative score per sample, and finally perform a second group-wise normalization to obtain the final advantage $\hat { A } _ { t } ^ { i }$

Specifically, when training the video tower with audio anchored, the cumulative score for video sample i is:

$$
S _ { \mathrm { v i d e o } } ^ { i } = \mathrm { n o r m } ( \mathrm { V Q } ^ { i } ) + \mathrm { n o r m } ( \mathrm { C L I P } ^ { i } ) + \mathrm { n o r m } ( \mathrm { - D e s y n c } ^ { i } ) ,
$$

where norm $\begin{array} { r } { ( r ^ { i } ) = \frac { r ^ { i } - \mathrm { m e a n } ( \{ r ^ { i } \} _ { i = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ r ^ { i } \} _ { i = 1 } ^ { G } ) } } \end{array}$ denotes group-wise standardization. Note that for the DeSync metric, the lower the better. The final advantage is:

$$
\hat { A } _ { \mathrm { v i d e o } } ^ { i } = \operatorname { n o r m } \bigr ( \{ S _ { \mathrm { v i d e o } } ^ { i } \} _ { i = 1 } ^ { G } \bigr ) .
$$

Symmetrically, when training the audio tower with video anchored as $\tau _ { v } ,$ the cumulative score for audio sample i is:

$$
S _ { \mathrm { a u d i o } } ^ { i } = \mathrm { n o r m } ( \mathrm { A Q } ^ { i } ) + \mathrm { n o r m } ( \mathrm { C L A P } ^ { i } ) + \mathrm { n o r m } ( \mathrm { - D e s y n c } ^ { i } ) ,
$$

with the final advantage:

$$
\hat { A } _ { \mathrm { a u d i o } } ^ { i } = \mathrm { n o r m } \big ( \{ S _ { \mathrm { a u d i o } } ^ { i } \} _ { i = 1 } ^ { G } \big ) .
$$

In practice, however, we observe that when training the audio tower, the model is prone to reward hacking: it easily finds a shortcut that sacrifices the CLAP score while substantially boosting AQ and Desync. This behavior severely undermines audio-prompt semantic alignment and is unacceptable.

Inspired by GDPO, we introduce a semantic alignment guardrail for CLAP. Let the group-wise average CLAP score be $\begin{array} { r } { \bar { r } _ { \mathrm { C L A P } } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \mathrm { C L A P } ^ { i } } \end{array}$ , with a predefined threshold $\eta _ { \mathrm { C L A P } } .$ . Only when $\bar { r } _ { \mathrm { C L A P } } \geq \eta _ { \mathrm { C L A P } }$ are AQ and Desync incorporated into the advantage; otherwise, the CLAP advantage serves directly as the final advantage. Formally:

$$
\begin{array} { r } { \hat { A } _ { \mathrm { a u d i o } } ^ { i } = \left\{ \begin{array} { l l } { \mathrm { n o r m } \big ( \{ S _ { \mathrm { a u d i o } } ^ { i } \} _ { i = 1 } ^ { G } \big ) , } & { \bar { r } _ { \mathrm { C L A P } } \geq \eta _ { \mathrm { C L A P } } , } \\ { \mathrm { n o r m } \big ( \{ { C L A P } ^ { i } \} _ { i = 1 } ^ { G } \big ) , } & { \bar { r } _ { \mathrm { C L A P } } < \eta _ { \mathrm { C L A P } } . } \end{array} \right. } \end{array}
$$

When the group-average CLAP falls below the threshold, the model receives learning signals exclusively from CLAP, guiding it to prioritize audio-prompt semantic alignment before attending to other metrics.

## E MORE QUALITATIVE RESULTS

Prompt: A welder is working in a workshop. ... He is using a welding torch to join two metal pieces together. Bright sparks and flashes of light are visible as he works.

![](images/a54e2034bf8c6dbfe15b004144a01392f233b21f61ad5d4177a846c8e34aa366.jpg)

Prompt: A yellow front-end loader is shown. It has a large bucket filled with dirt. The loader is on a construction site, and the sky in the background is blue with some clouds

![](images/63b557d57dfa84b5221ee0acc20f78867b42c0e6fb1dbc77866f67b9dc375d8c.jpg)

![](images/7971acc7da87a2bfaeb7bb1be60200e00eece23e2fce4965e7438a51b108e7ae.jpg)

Prompt: The video shows an aerial view of a city. There are various buildings, roads, and qreen spaces. ... The roads are laid out in a pattern, ...

Prompt: The video shows a serene forest scene. .. In the middle ot the forest, there's a small body of water, possibly a pond or a stream, reflecting the trees and the sky...

![](images/4cfa0026536f042535891d4e8450b5fd6ccd9a93fb7b583622291eb9929bcab8.jpg)

![](images/a85fbd10bd395b65c6c3f384a079cbb43cdd7d7434f9bdacd61656509198cebf.jpg)

Prompt: The trees have pink and white flowers, ... In the background, there are traditional Chinese-style buildings with sloping roofs. .. There are also some birds flying in the sky.

![](images/5f1c75e3a4ab598f63c238cfaab6fa867e7c7c77a49ccc0e6b57cc4a4cc61945.jpg)

Prompt: A man with a beard and mustache is sitting down, wearing a checkered shirt. He is holding a small object in his hand and appears to be looking at it closely. ...

Prompt: The sound is of water gurgling and bubbling, which is typical of underwater environments. There are no other distinct sounds.

![](images/4ff98b31acf487d827c7b91132fc9f052ffe7752d0e9df2d3fee74b9c6edf246.jpg)

![](images/8cd12a487b0b6931c72d0239ed1ef30eef6c6d246e64f4eefce18b3c50d919f1.jpg)

Prompt: The video shows a Lego scene with a character holding a green lightsaber. The character is in a dark, industrial-looking environment with green lighting. ...

![](images/03e509dc63b8e0c84cf53db7d748366d608763c5c0a6d00c38fc6607c54357cb.jpg)

Prompt: The video shows a modern. two-story house at night. The house has a clean, minimalist design with large windows and a flat roof. ... The house is illuminated by warm lights. ..

![](images/8b0ea333539cc77b03093992e04ad4d906231b1b15fc30dc91b51a67bdcb7fd2.jpg)  
Figure 4: More qualitative results of AV-GRPO (full).