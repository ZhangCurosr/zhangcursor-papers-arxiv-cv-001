# DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models

Yohan Choi<sup>1</sup>, Min-Jun Kim<sup>1</sup>, Jin-Sung Kim<sup>1</sup>, Yong-Jae Kim<sup>2</sup>, Youn-Hee Han<sup>1,∗</sup>

Abstract— Vision-based legged locomotion methods assume clean depth at training time and rely on hand-tuned postprocessing filters at deployment. However, filter parameters are rarely disclosed, hindering reproducibility, and performance degrades substantially when depth noise is left unaddressed. Building noise robustness directly into the learning pipeline would eliminate this dependency. While such robustness has been explored for proprioceptive inputs, analogous approaches for depth perception remain largely absent in legged locomotion. We propose DAWN (Denoising and Alignment in World models for Noise-robustness), a noise-robust perception framework for legged locomotion, which builds noise robustness directly into a world model via two modifications: (1) feeding noisy depth to the encoder while keeping clean depth as the reconstruction target, forcing the model to implicitly denoise its input; and (2) applying contrastive learning to align the latent states of noisy and clean depth. Importantly, DAWN is not tied to a specific noise model, requiring no manual tuning to the noise distribution at deployment. Furthermore, it incurs no additional inference cost over existing world model-based methods. Without any manual filter calibration—relying solely on the learned noise-robust representation—DAWN achieves zeroshot quadruped parkour on a Unitree Go1: traversing stairs up to 18 cm, clearing gaps up to 70 cm, and mounting steps up to 45 cm from raw depth observations. Ablation studies show that denoising and contrastive alignment contribute at complementary levels—reconstruction and representation, respectively— and yield additive gains when combined. Videos and code are available at: https://dawn-parkour.github.io/

## I. INTRODUCTION

Reinforcement learning (RL) has enabled quadruped robots to traverse diverse terrains through sim-to-real transfer [1], [2]. A blind policy with only proprioceptive input can traverse moderate terrains such as slopes and stairs [3], [4], but fails on obstacles that require perceiving terrain geometry in advance, such as gaps and steps [5], [6]. Depth cameras provide direct geometric information about such terrains and have become the primary sensor for vision-based locomotion [6], [5]. Recent depth-based methods have demonstrated increasingly agile parkour on low-cost quadrupeds, from climbing obstacles up to 0.55 m and leaping gaps up to 0.85 m to jumping obstacles exceeding twice the robot’s height [7], [8], [9]. These and most other depth-based locomotion methods rely on Intel RealSense D435/D435i cameras [7], [8], [9], [10], [11]. However, the handling of sensor noise at deployment is rarely discussed.

Despite this progress, most vision-based locomotion methods assume clean depth during training and defer noise handling to post-processing filters at deployment [9], [8], [7], [5]. Even when depth degradation is acknowledged, common remedies remain outside the main learning pipeline— proprioceptive fallback [6] or a separate learned reconstruction module [10]—suggesting that the standard training procedure alone does not produce noise-robust policies. Moreover, while post-processing filters are widely used at inference [7], [8], [9], [11], their specific parameters are seldom reported. The librealsense2 pipeline for the D435i [12] exposes six filter types with 12–24 interdependent parameters. Their optimal values vary with illumination, surface material, and scene depth, making consistent reproduction across environments difficult.

Prior work has quantified the severity of this issue [13], [14], but the approaches explored so far operate at the input level: domain randomization [15], hand-crafted augmentation [13], [14], or separate denoising modules [10]. Handcrafted filters cannot address scene-dependent noise with fixed parameters [16], and prior depth denoising methods exhibit mismatches with current stereo sensors [17]. These input-level augmentations provide limited representational capacity [18]. Meanwhile, world model-based denoising has been applied only to proprioception [19], [20], where noise is low-dimensional and approximately i.i.d. Depth images, by contrast, exhibit spatially structured noise whose statistics vary with scene geometry and illumination [21], [12]—a fundamentally different regime that prior approaches have not addressed.

In this work, we propose DAWN (Denoising and Alignment in World models for Noise-robustness), a perception framework for legged locomotion that embeds noise robustness into the latent space of a world model through two modifications to the Recurrent State-Space Model (RSSM) architecture of Lai et al. [9]. Specifically, DAWN introduces an input–target mismatch in the RSSM, where the encoder receives noisy depth but the decoder is supervised with clean depth, and applies SimCLR [22]-based contrastive alignment between the latent representations of noisy and clean depth. Unlike fixed-parameter filters that require scene-specific tuning, our RSSM encoder learns a nonlinear mapping that discards noise conditioned on scene context, trained end-toend with the locomotion policy. Critically, both modifications apply only at training time, adding no inference cost over the base world model. Notably, DAWN does not rely on a specific noise model and needs no tuning to the noise distribution at deployment.

In simulation, DAWN achieves 96.9% average success rate across stairs, gaps, and steps from raw depth, closing 77% of the gap between the depth-based baseline WMP [9] and the clean-depth oracle, and degrading far less than baselines under out-of-distribution noise. Deployed zero-shot on a real Unitree Go1 without any manual filter calibration, DAWN traverses stairs up to 18 cm, gaps up to 70 cm, and steps up to 45 cm in both indoor and outdoor environments.

Below, we summarize our main contributions:

1) DAWN, a noise-robust perception framework that builds depth denoising directly into the world model, eliminating environment-dependent filter tuning at deployment.

2) Zero-shot sim-to-real quadruped parkour on a Unitree Go1 across stairs, gaps, and steps, without man ual filter calibration.

3) Systematic ablations confirming that RSSM denoising and contrastive alignment operate at complementary levels—reconstruction and representation—and yield additive gains when combined.

These results suggest that noise robustness for depth-based legged locomotion can be achieved through learned representation rather than hand-engineered filtering.

## II. RELATED WORK

We review four lines of work relevant to DAWN: visual legged locomotion, depth sim-to-real transfer, world models for locomotion, and contrastive learning for locomotion.

## A. Visual Legged Locomotion

Sim-to-real RL [1], [2] combined with privileged learning [23], [3] has become the dominant paradigm for legged locomotion. Policies with only proprioceptive input can traverse moderate terrains such as slopes and stairs [3], but fail on obstacles that require perceiving terrain geometry in advance, such as gaps and steps [5], [6]. Depth cameras provide direct geometric information for such terrains, and vision-based parkour has progressed from climbing obstacles up to 1.5× the robot height [7] to jumps exceeding 2× the robot height [8]. Lai et al. [9] introduced a world modelbased approach that bypasses the teacher–student distillation pipeline, learning perception and policy end-to-end via the RSSM. These methods commonly train with clean depth and defer noise handling to hand-tuned post-processing filters at deployment.

## B. Depth Sim-to-Real Transfer

Domain randomization [15] is widely used for sim-toreal transfer, but even large-scale automatic domain randomization does not fully close all sim-to-real gaps [24]. Keselman et al. [12] reported the official characteristics of the RealSense D400 series, and Ahn et al. [21] provided an empirical noise model of the D435. Sweeney et al. [16] analyzed scene-dependent filter behavior, and Hu et al. [17] examined the applicability of prior depth denoising methods to current stereo sensors. Liu et al. [25] proposed a systematic depth simulation pipeline, and Sun et al. [14] and Zhuang et al. [11] introduced physically informed noise synthesis. Sun et al. [13] reported a 56 p.p. success rate gain through eight depth augmentations, quantifying the severity of depth noise. Hoeller et al. [10] addressed noisy depth with a learned terrain reconstruction module that fuses six depth cameras and LiDAR. This approach, however, requires auxiliary hardware and is optimized independently of the downstream policy, preventing end-to-end learning.

Existing work focuses on input-level noise modeling or separate denoising modules, both of which require scenespecific filters or auxiliary components at deployment. Repurposing the reconstruction objective of a world model to remove filter dependency entirely has not been explored.

## C. World Models for Locomotion

Following Ha and Schmidhuber [26], Hafner et al. [27] introduced the RSSM, which evolved through subsequent iterations [28], [29], [30]. Wu et al. [31] demonstrated learning locomotion from scratch on a real robot using a world model.

Gu et al. [19] applied noisy-to-clean reconstruction of proprioception to achieve noise-robust locomotion, and Sun et al. [20] introduced gradient cutoff to protect denoising quality. These works, however, address only proprioception, whose noise is low-dimensional and approximately i.i.d. Depth noise is fundamentally different: it concentrates at depth discontinuities and grows nonlinearly with distance [21], [12], requiring architectural considerations for processing spatially structured features.

## D. Contrastive Learning for Locomotion

Contrastive representation learning originated with van den Oord et al. [32] and advanced in the vision domain through He et al. [33] and Chen et al. [22]. In RL, Srinivas et al. [34] demonstrated improved sample efficiency for pixelbased control. For locomotion, Long et al. [35] applied prototypical representation alignment, Mousa et al. [36] used contrastive triplet loss for teacher–student alignment, and Lu et al. [37] improved sim-to-real transfer for humanoid locomotion. These approaches target proprioceptive or general visual representations; enforcing invariance to depth noise via contrastive learning has not been explored.

## III. METHOD

DAWN builds upon the RSSM architecture of Lai et al. [9] and introduces two modifications for depth noise robustness: (1) a denoising reconstruction objective that feeds noisy depth to the encoder while keeping clean depth as the reconstruction target, and (2) a contrastive loss that aligns latent states from noisy and clean observations. An overview is shown in Fig. 1.

![](images/e311fa87b32303933f1f28b63a88ed4b933b33b891a31ceae25d9393c30c2328.jpg)  
Fig. 1: Overview of the DAWN framework. DAWN’s two modifications are highlighted in green: the denoising reconstruction loss $\mathcal { L } _ { \mathrm { d e n o i s e } }$ between the decoder output and the clean depth target, and the contrastive loss ${ \mathcal { L } } _ { \mathrm { c o n t r a s t } }$ applied to the projection head (MLP).

## A. Preliminaries: World Model Learning

We define the observation at timestep t as $x _ { t } = ( d _ { t } , o _ { t } ^ { \mathsf { p } } )$ where $d _ { t }$ is the depth image and $o _ { t } ^ { \mathfrak { p } }$ is the proprioception. The depth image is updated every k steps, and $a _ { t }$ denotes the joint position target action.

Lai et al. [9] propose World Model-based Perception (WMP), an end-to-end framework that adopts an RSSM following Hafner et al. [29]. The RSSM consists of four components parameterized by ϕ:

$$
\begin{array} { r } { \mathrm { S e q u e n c e ~ m o d e l } ; \quad h _ { t } = f _ { \phi } \big ( h _ { t - k } , z _ { t - k } , a _ { t - k : t - 1 } \big ) } \end{array}\tag{1}
$$

$$
\begin{array} { r l } { \operatorname { E n c o d e r : } } & { { } z _ { t } \sim q _ { \phi } ( . \mid h _ { t } , x _ { t } ) } \end{array}\tag{2}
$$

$$
\mathrm { D y n a m i c s ~ p r e d i c t o r : } \quad \hat { z } _ { t } \sim p _ { \phi } ( \cdot \mid h _ { t } )\tag{3}
$$

$$
\mathrm { D e c o d e r : } \quad \hat { x } _ { t } \sim p _ { \phi } ( \cdot \mid h _ { t } , z _ { t } )\tag{4}
$$

where $h _ { t }$ is a deterministic state computed by a Gated Recurrent Unit (GRU) [38]-based sequence model $( 1 ) , z _ { t }$ is the stochastic state incorporating the current observation $( 2 )$ $\hat { z } _ { t }$ is the prior predicted without observation access (3), and $\hat { x } _ { t }$ is the reconstructed observation (4).

These components are jointly optimized by minimizing:

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { W M P } } = \mathbb { E } \Bigl [ \sum _ { t } \underbrace { - \ln p _ { \phi } ( x _ { t } \mid z _ { t } , h _ { t } ) } _ { \mathrm { r e c o n s t r u c t i o n } } } \quad } & { } \\ & { +  \beta \mathrm { K L } \bigl [ q _ { \phi } ( z _ { t } \mid h _ { t } , x _ { t } ) \bigr \rvert \bigr \rvert p _ { \phi } ( z _ { t } \mid h _ { t } ) \bigr ] ] , } \end{array}\tag{5}
$$

where $\beta$ is a hyperparameter. The reconstruction term encourages $z _ { t }$ to retain sufficient information about $x _ { t } .$ , while the KL term regularizes the posterior toward the prior. The policy is trained with Proximal Policy Optimization [39] using the deterministic state $h _ { t }$ as input.

In WMP, both the encoder input and the reconstruction target are the same clean observation $x _ { t }$ . Depth noise at deployment is handled by post-processing filters applied to raw sensor readings. DAWN removes this filter dependency by modifying how the RSSM is trained, without changing its architecture.

## B. Denoising Reconstruction Objective

We denote clean and noisy depth as $d _ { t } ^ { \mathrm { c } }$ and $d _ { t } ^ { \mathrm { n } }$ , respectively, and write the corresponding observations as $x _ { t } ^ { \mathrm { c } } = ( d _ { t } ^ { \mathrm { c } } , o _ { t } ^ { \mathrm { p } } )$ and $x _ { t } ^ { \mathrm { n } } = ( d _ { t } ^ { \mathrm { n } } , o _ { t } ^ { \mathrm { p } } )$ . The noisy depth $d _ { t } ^ { \mathrm { n } }$ is generated by a noise model.

In WMP’s standard training, the encoder input and the decoder target are identical:

$$
z _ { t } \sim q _ { \phi } ( \cdot \mid h _ { t } , x _ { t } ^ { \mathrm { c } } ) , \qquad \hat { x } _ { t } \approx x _ { t } ^ { \mathrm { c } } .\tag{6}
$$

DAWN replaces the encoder input with the noisy observation while keeping the clean observation as the reconstruction target:

$$
z _ { t } ^ { \mathrm { n } } \sim q _ { \phi } ( \cdot \mid h _ { t } , x _ { t } ^ { \mathrm { n } } ) , \qquad \hat { x } _ { t } \approx x _ { t } ^ { \mathrm { c } } .\tag{7}
$$

Comparing (6) and (7), the only change is in the encoder input: from $\boldsymbol { x } _ { t } ^ { \mathrm { c } }$ to $\boldsymbol { x } _ { t } ^ { \mathrm { n } }$ . The decoder must still reconstruct the clean observation, which forces the encoder to map noisy input to a latent state from which clean depth can be recovered.

This input–target mismatch turns the standard reconstruction loss into a denoising objective. Substituting (7) into the WMP loss (5) yields the DAWN reconstruction loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d e n o i s e } } = \mathbb { E } \Bigl [ \underset { t } { \sum } \underbrace { - \ln p _ { \phi } ( x _ { t } ^ { \mathrm { c } } \mid z _ { t } ^ { \mathrm { n } } , h _ { t } ) } _ { \mathrm { d e n o i s i n g } } } \\ & { \quad \quad \quad + \left. \beta \mathrm { K L } \bigl [ q _ { \phi } ( z _ { t } ^ { \mathrm { n } } \mid h _ { t } , x _ { t } ^ { \mathrm { n } } ) \bigr \rvert \big \rvert p _ { \phi } ( z _ { t } \mid h _ { t } ) \bigr ] \right] . } \end{array}\tag{8}
$$

The KL term constrains the information capacity of $\boldsymbol { z } _ { t } ^ { \mathrm { n } } .$ . Since reconstructing $\boldsymbol { x } _ { t } ^ { \mathrm { c } }$ does not require any information about the noise ϵ in $\boldsymbol { x } _ { t } ^ { \mathrm { n } }$ , the encoder discards noise-specific features under this capacity constraint and retains only terrainrelevant geometry. This behavior follows directly from the information bottleneck principle [40]: when the target is $\boldsymbol { x } _ { t } ^ { \mathrm { c } } ,$ the mutual information $I ( z _ { t } ^ { \mathrm { n } } ; \epsilon )$ does not contribute to reducing the reconstruction loss and is suppressed by the KL penalty.

TABLE I: Comparison of WMP and DAWN training configurations. The RSSM architecture and policy training are identical; only the encoder input and loss terms differ.
<table><tr><td></td><td>WMP</td><td>DAWN</td></tr><tr><td>Encoder input</td><td></td><td> $\begin{array} { l } { x _ { t } ^ { \mathrm { n } } } \\ { x _ { t } ^ { \mathrm { c } } } \end{array}$ </td></tr><tr><td>Decoder target</td><td> $\begin{array} { l } { x _ { t } ^ { \mathrm { c } } } \\ { x _ { t } ^ { \mathrm { c } } } \end{array}$ </td><td></td></tr><tr><td>Reconstruction loss</td><td> ${ \mathcal { L } } _ { \mathrm { W M P } }$ </td><td> $\mathcal { L } _ { \mathrm { d e n o i s e } }$ </td></tr><tr><td>Contrastive loss</td><td></td><td> $\mathcal { L } _ { \mathrm { { c o n t r a s t } } }$ </td></tr><tr><td>Post-processing filter</td><td>Required</td><td>1 Not required</td></tr></table>

This learned compression provides a practical advantage over hand-crafted post-processing filters. Filter pipelines operate with a fixed set of parameters that require scene-specific tuning, whereas the encoder learns a nonlinear mapping trained end-to-end with the policy and conditioned on scene context. Because the denoising objective already drives the encoder to retain only task-relevant geometry, no additional filter stage is required at deployment.

## C. Contrastive Latent Alignment

The denoising objective encourages noise-invariant reconstruction but does not explicitly constrain the latent space structure. We add a contrastive loss to directly align the encoder outputs from clean and noisy observations.

We write $z _ { t } ^ { \mathrm { c } } \sim q _ { \phi } ( \cdot \mid h _ { t } , x _ { t } ^ { \mathrm { c } } )$ and $z _ { t } ^ { \mathrm { n } } \sim q _ { \phi } ( \cdot \mid h _ { t } , x _ { t } ^ { \mathrm { n } } )$ for the posterior states encoded from clean and noisy inputs at the same timestep. A projection head $g _ { \psi }$ maps these posteriors to a contrastive embedding space: $v _ { t } ^ { \mathrm { c } } = g _ { \psi } ( h _ { t } , z _ { t } ^ { \mathrm { c } } )$ and $v _ { t } ^ { \mathrm { n } } =$ $g _ { \psi } ( h _ { t } , z _ { t } ^ { \mathfrak { n } } )$

From N parallel environments within a batch, the clean and noisy embeddings of the same scene $i , \ ( v _ { i } ^ { \mathrm { c } } , v _ { i } ^ { \mathrm { n } } )$ , form a positive pair, while those from different scenes $i \neq j$ form negative pairs. The Normalized Temperature-scaled Cross Entropy (NT-Xent) loss is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n t r a s t } } = - \log \frac { \exp ( \sin ( v _ { i } ^ { \mathrm { c } } , v _ { i } ^ { \mathrm { n } } ) / \tau ) } { \displaystyle \sum _ { j = 1 } ^ { 2 N } \mathbf { 1 } _ { [ j \neq i ] } \exp ( \sin ( v _ { i } ^ { \mathrm { c } } , v _ { j } ) / \tau ) } , } \end{array}\tag{9}
$$

where $v _ { i } ^ { \mathrm { c } }$ is the anchor, sim $( \cdot , \cdot )$ denotes cosine similar-$\mathrm { i t y } , \ \mathbf { 1 } _ { [ j \neq i ] }$ is an indicator function excluding the anchor, $\{ v _ { j } \} _ { j = 1 } ^ { 2 \breve { N } ^ { ' } } \stackrel { \cdot } { = } \{ v _ { 1 } ^ { \mathrm { c } } , \ldots , v _ { N } ^ { \mathrm { c } } , v _ { 1 } ^ { \mathrm { n } } , \ldots , v _ { N } ^ { \mathrm { n } } \}$ is the set of all embeddings in the batch, and $\tau$ is the temperature parameter.

Following Chen et al. [22], the contrastive loss is applied to the projected embeddings $v _ { t }$ rather than directly to $z _ { t } .$ preventing the encoder from collapsing toward the contrastive objective and preserving diverse information useful for policy learning. At deployment, the projection head $g _ { \psi }$ is removed. The contrastive loss serves only as a trainingtime regularizer that shapes the latent geometry; the inference pipeline remains identical to WMP.

## D. Total Training Objective

The full DAWN loss combines the denoising reconstruction objective (8) with the contrastive alignment loss (9):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D A W N } } = \mathcal { L } _ { \mathrm { d e n o i s e } } + \lambda \mathcal { L } _ { \mathrm { c o n t r a s t } } , } \end{array}\tag{10}
$$

where λ controls the relative weight of the contrastive term. Setting $\lambda = 0$ and replacing $\boldsymbol { x } _ { t } ^ { \mathrm { n } }$ with $\boldsymbol { x } _ { t } ^ { \mathrm { c } }$ in (8) recovers the original WMP loss (5).

Table I summarizes the differences between WMP and DAWN. The RSSM architecture and the policy training procedure remain unchanged; DAWN modifies only the training signals.

## IV. EXPERIMENTAL RESULTS

## A. Experimental Setup

Simulation. Training is conducted in IsaacLab [41] with 4,096 parallel environments. The control frequency is 50 Hz, and depth images at $6 4 \times 6 4$ resolution are updated every $k ~ = ~ 5$ steps (0.1 s). All experiments are repeated with three seeds, and 100 episodes per condition are evaluated, reporting mean±standard deviation.

Real-world. A Unitree Go1 is equipped with an Intel RealSense D435i depth camera and an Nvidia Jetson NX, with all RealSense built-in filters disabled so that the policy receives raw depth. Policies trained in simulation are transferred zero-shot without additional fine-tuning. We evaluate the success rate over 10 trials per difficulty level across both indoor and outdoor environments.

Depth Noise Simulation. DAWN does not require tuning to a specific noise model. We use a D435i-characteristic model [21] capturing three phenomena, validated in §IV-F. We denote the intermediate depth image as ${ \tilde { d } } .$ The noise model applies the following filters to $d _ { t } ^ { \mathrm { c } }$ to produce the final noisy depth $d _ { t } ^ { \mathrm { n } }$

• Gaussian sensor noise adds zero-mean noise:

$$
\tilde { d } = d _ { t } ^ { \mathrm { c } } + \eta , \quad \eta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \quad \sigma = 0 . 0 1 \mathrm { m } .\tag{11}
$$

• Edge dropout simulates stereo matching failure at depth discontinuities with quadratic probability:

$$
P _ { \mathrm { d r o p } } ( x , y ) = P _ { \mathrm { m a x } } \cdot \mathrm { c l i p } \left( \frac { G - G _ { \mathrm { t h } } } { G _ { \mathrm { s a t } } - G _ { \mathrm { t h } } } , 0 , 1 \right) ^ { 2 } ,\tag{12}
$$

where $G = \| \nabla \tilde { d } \|$ is the spatial gradient magnitude, $G _ { \mathrm { t h } } = 0 . 0 4$ is the minimum gradient for edge detection, $G _ { \mathrm { s a t } } ~ = ~ 0 . 2 0$ is the gradient at which the dropout probability saturates, and $P _ { \mathrm { m a x } } ~ = ~ 0 . 6 0$ is the upper bound on dropout probability. Dropped pixels are filled by $3 \times 3$ max pooling.

• Far particle noise models infrared (IR)–ambient light interference:

$$
P _ { \mathrm { p a r t i c l e } } ( x , y ) = r _ { p } \cdot \mathrm { c l i p } \left( \frac { \tilde { d } - d _ { \mathrm { t h } } } { 0 . 2 } , 0 , 1 \right) ,\tag{13}
$$

where $r _ { p } ~ = ~ 0 . 0 0 3$ is the maximum particle rate and $d _ { \mathrm { t h } } = 0 . 3$ is the depth threshold beyond which particles are applied.

Terrains. We use four terrain types, each with five difficulty levels: Slope (6–22<sup>◦</sup>), Stair (6–18 cm), Gap (10–90 cm), and Step (10–54 cm). Slope is included only in the baseline comparison (Table II) and excluded from ablation and noise

TABLE II: Baseline comparison (noise ×1.0). SR (%, ↑) and TE (m/s, ↓) are reported. Averaged over all difficulty levels, 3 seeds × 100 episodes.
<table><tr><td rowspan="2">Method</td><td colspan="2">Slope (6–22°)</td><td colspan="2">Stair (6–18 cm)</td><td colspan="2">Gap (10–90 cm)</td><td colspan="2">Step (10–54 cm)</td></tr><tr><td>SR</td><td>TE</td><td>SR</td><td>TE</td><td>SR</td><td>TE</td><td>SR</td><td>TE</td></tr><tr><td>Blind</td><td> $9 8 . 9 { \pm } 3 . 8 \ $ </td><td> $0 . 0 2 5 { \scriptstyle \pm . 0 0 4 }$ </td><td> $7 4 . 6 { \pm } 4 4 . 5 $ </td><td> $0 . 0 6 4 { \pm } . 0 6 6$ </td><td> $2 6 . 2 { \pm } 4 4 . 0 $ </td><td> $0 . 1 5 4 { \pm } . 0 7 3$ </td><td> $3 6 . 2 { \pm } 4 3 . 4 $ </td><td> $0 . 9 3 6 { \pm } . 0 7 1$ </td></tr><tr><td>EP</td><td> $1 0 0 . 0 { \pm } 0 . 0 \ \mathrm { \Omega }$ </td><td> $0 . 0 1 1 { \pm } . 0 0 3$ </td><td> $8 2 . 2 { \pm } 3 5 . 4 $ </td><td> $0 . 0 3 1 { \pm } . 0 4 2$ </td><td> $8 3 . 5 { \pm } 3 5 . 9 \ \qquad $ </td><td> $0 . 0 7 2 { \scriptstyle \pm . 0 4 8 }$ </td><td> $9 2 . 6 { \pm } 2 3 . 2 $ </td><td> $0 . 0 3 8 { \pm } . 0 3 1$ </td></tr><tr><td>WMP</td><td> $1 0 0 . 0 { \pm } 0 . 0 \ \mathrm { \Omega }$ </td><td> $0 . 0 1 0 { \scriptstyle \pm . 0 0 3 }$ </td><td> $8 9 . 7 { \pm } 3 0 . 4 $ </td><td> $0 . 0 2 6 { \pm } . 0 3 8$ </td><td> $8 8 . 6 { \pm } 3 1 . 8 $ </td><td> $\mathbf { 0 . 0 5 8 } { \scriptstyle \pm . 0 3 4 }$ </td><td> $9 5 . 9 { \pm } 1 9 . 8 $ </td><td> $0 . 0 2 6 { \pm } . 0 1 8$ </td></tr><tr><td>WMP w/ N</td><td> $9 9 . 9 { \pm } 3 . 2 $ </td><td> $0 . 0 0 9 { \scriptstyle \pm . 0 0 3 }$ </td><td> $9 4 . 5 { \pm } 2 2 . 8 $ </td><td> $0 . 0 2 3 { \scriptstyle \pm . 0 3 6 }$ </td><td> $9 3 . 7 { \pm } 2 4 . 3 $ </td><td> $0 . 0 6 7 { \scriptstyle \pm . 0 2 9 }$ </td><td> $9 5 . 3 { \pm } 2 1 . 2 $ </td><td> $\mathbf { 0 . 0 2 5 { \pm } . 0 2 1 }$ </td></tr><tr><td>DAWN</td><td> ${ \bf 9 9 . 9 2 2 . 5 }$ </td><td> $\mathbf { 0 . 0 0 8 } { \pm . 0 0 2 }$ </td><td> ${ \bf 9 6 . 6 { \pm } 1 8 . 2 }$ </td><td> $\mathbf { 0 . 0 1 5 { \pm } . 0 2 0 }$ </td><td> $\mathbf { 9 7 . 2 { \pm } 1 6 . 6 }$ </td><td> $0 . 0 6 5 { \pm } . 0 3 1$ </td><td> $\mathbf { 9 7 . 0 { \pm } 1 7 . 0 }$ </td><td> $0 . 0 3 4 { \pm } . 0 2 0$ </td></tr><tr><td>Oracle</td><td> $1 0 0 . 0 { \pm } 0 . 0 \ \mathrm { \Omega }$ </td><td> $0 . 0 0 8 { \pm } . 0 0 2$ </td><td> $9 8 . 2 { \pm } 1 3 . 3 $ </td><td> $0 . 0 1 3 { \pm } . 0 1 5$ </td><td> $9 8 . 8 { \pm } 1 0 . 9 \ $ </td><td> $0 . 0 5 4 { \pm } . 0 2 8$ </td><td> $9 8 . 5 { \pm } 1 2 . 2 \ $ </td><td> $0 . 0 2 0 { \pm } . 0 1 4$ </td></tr></table>

The terrain-wise pattern isolates where noise hurts. Slope yields near-perfect SR for every method, including Blind: with no depth discontinuities, proprioception alone suffices and depth noise is inconsequential. The picture reverses on terrains defined by geometric edges. Blind collapses on Gap and Step, confirming that advance depth perception is required once the robot must anticipate an obstacle it cannot feel. WMP, the clean-depth world-model baseline we build on, degrades most on Stair and Gap, where edge dropout removes depth precisely at the boundaries the policy relies on—Stair through consecutive edges subject to cumulative

Table II reports baseline results under the default $\times 1 . 0$ noise, averaged across all difficulty levels over three seeds. We use this comparison to answer whether depth denoising must be built into the learning pipeline or can be deferred to the training data and to post-processing filters.

TABLE III: Summary of compared methods. Noise: depth noise applied during training. Denoise: reconstruction target set to clean depth. Contrastive: contrastive learning applied.
<table><tr><td>Method</td><td>Noise Denoise</td><td>Contrastive</td></tr><tr><td>Blind</td><td>一 一</td><td>一</td></tr><tr><td>EP [8]</td><td>X</td><td>X X</td></tr><tr><td>WMP [9]</td><td>X X</td><td>X</td></tr><tr><td>WMP w/ N</td><td>√ X</td><td>X</td></tr><tr><td>DAWN w/o C</td><td>√ √</td><td>×</td></tr><tr><td>DAWN w/o D</td><td>√ X</td><td>√</td></tr><tr><td>DAWN</td><td>√ √</td><td>√</td></tr><tr><td>Oracle</td><td>一 一</td><td>一</td></tr></table>

## B. Baseline Comparison

Metrics. We report Success Rate (SR, %) and velocity Tracking Error (TE, m/s). SR measures the percentage of episodes in which the robot successfully traverses the terrain without falling. TE is the mean squared error between the commanded velocity and the measured velocity capturing how precisely the robot tracks the desired motion.

sweep analyses, as its lack of depth discontinuities yields similar performance across all methods.

Compared Methods. Table III summarizes the compared methods. WMP w/ N adds noise training to WMP. DAWN w/o C and DAWN w/o D are ablation variants that remove contrastive learning and RSSM denoising, respectively, yielding denoising-only and contrastive-only variants.

TABLE IV: Noise scale parameters. ×1.0 is the default setting.
<table><tr><td>Parameter</td><td>×1.0</td><td>×1.5</td><td>×2.0</td></tr><tr><td>Gaussian σ (m)</td><td>0.01</td><td>0.015</td><td>0.02</td></tr><tr><td>Edge  $P _ { \mathrm { m a x } }$ </td><td>0.60</td><td>0.80</td><td>0.90</td></tr><tr><td>Particle  $r _ { p }$ </td><td>0.003</td><td>0.0045</td><td>0.006</td></tr></table>

dropout, Gap through a width estimate that needs both rims visible at once. Step is more forgiving, since a single discontinuity leaves the height change recoverable from depth values on either side. EP, the distillation-based policy, trails WMP on every perceptive terrain, indicating that the teacher–student pipeline carries less usable geometry to the policy than the end-to-end world model even before noise is considered.

Across Stair/Gap/Step, DAWN reaches 96.9% average SR, closing 77% of the gap that noise opens between WMP and the clean-depth Oracle. WMP w/ N recovers part of this gap through noise exposure alone but stays below DAWN, showing that seeing noise during training is not equivalent to structuring the representation against it. The trend carries to tracking: DAWN attains the lowest TE on Slope and Stair, cutting WMP’s Stair error by 42%, which we attribute to denoised depth producing more precise foot placement. On Gap and Step, TE differences among the world-model methods fall within overlapping variance, indicating that under ×1.0 noise the corrupted boundaries surface primarily in SR rather than in tracking.

## C. Ablation Study

Fig. 2 (left) isolates the contribution of each modification by tracking average SR over Stair/Gap/Step as difficulty increases. We compare five methods: the full DAWN; its two single-component variants (DAWN w/o C, denoising only; DAWN w/o D, contrastive only); and two WMP baselines that differ only in noise training—WMP w/ N, trained with noisy depth, and WMP, trained on clean depth.

All five methods perform near-ceiling through the low difficulties, where obstacles are small and the surviving depth is sufficient regardless of how it is processed. The methods separate only from level 4 onward: larger obstacles subtend wider depth discontinuities, producing broader dropout regions that make the quality of the recovered geometry the limiting factor. We therefore read the ablation at the hardest level, where the noise stress is greatest. There, DAWN reaches 88.2% SR. Measured as increments over noiseonly training, denoising alone adds 5.7 p.p. and contrastive alignment alone 4.6 p.p., while the two together add 11.5 p.p.—more than the sum of the parts, so the components are complementary rather than redundant.

![](images/ba7a4d64c04b704b7fe33c87887cf96d92a37f4a2f8580265c8dda96633ca247.jpg)

![](images/b51c62e19529143acd98fbc53b45286c41f7582871e5b0fa2678aa3b70bd5026.jpg)

![](images/aa8f32b851e59b6f1e65b0c39d9a4cca788ad36bf7d64c2ceaafa0445e2a2077.jpg)

Fig. 2: Simulation analysis. Ablation study showing average SR over Stairs/Gap/Step by difficulty level (noise ×1.0), mean of 3 seeds (left). Noise robustness analysis showing average SR over Stairs/Gap/Step at the highest difficulty as the noise scale increases from ×1.0 to $\times 2 . 0 .$ , averaged over 3 seeds (center). t-SNE visualization of encoder latent states; colors indicate terrain type, shading indicates input condition (filled: clean, open: noisy) (right).  
![](images/07abb4a8333dd03681549c3c91fe24053e531963136177520d560e16f11634fb.jpg)  
Fig. 3: Comparison of simulated clean depth, simulated noisy depth (×1.0), and real D435i depth across three terrains (Stairs, Gap, Step). The simulated noise patterns closely match real sensor output.

The complementarity follows from how the two objectives interact: denoising restructures the latent space toward noisefree geometry, giving contrastive alignment a stable target to match, while alignment in turn regularizes the encoder output so the decoder reconstructs from a cleaner state. The two objectives provide additive performance gains when combined.

![](images/6ad1dc1d40078859176d7e689178fdb58cb10fef5d9b6b47fee70c57bf3029be.jpg)  
Fig. 4: Real-world depth reconstruction on three terrains (Stairs, Gap, Step). For each terrain, the top row is the raw D435i depth input and the bottom row (dashed border) is DAWN’s decoder output at the same timestep, shown at t = k, 2k, . . . , 5k, 15k, and 25k.

## D. Noise Robustness Analysis

The ×1.0 results establish robustness at the noise level seen during training. Fig. 2 (center) sweeps the noise scale from ×1.0 to ×2.0 at the hardest difficulty (see Table IV); since all policies are trained at ×1.0, the higher scales are out-of-distribution. Across the sweep DAWN loses 6.5 p.p. versus 17.5 for clean-trained WMP, and the DAWN–WMP margin widens from 14.0 p.p. at ×1.0 to $2 4 . 9 \mathsf { p . p }$ . at ×2.0, with the DAWN–WMP w/ N margin growing in step: the baselines lose ground fastest exactly where noise is most severe, whereas DAWN’s advantage grows. We attribute this to where each method acts: domain randomization fits the encoder to the noise levels it has seen and extrapolates poorly beyond them, while denoising and contrastive alignment reshape the latent space toward noise-free geometry—a target that does not move as the input noise intensifies. This margin behavior anticipates the real-world results (§IV-H), where outdoor IR interference pushes sensor noise past the indoor regime and the clean-trained baselines degrade most.

Difficulty (cm)  
Difficulty (cm)  
![](images/0e8b3b432072e740db63e2f80310b0e76f74a9be89ae6ab5532a197bc41d53c1.jpg)

![](images/4ae71bea49ad1c69b01605936e1523ec2dada997099b9f1539029f94cf0cb154.jpg)

![](images/7f8e319b997b26857e6d4143df5d7bfa17e8aa24fb5dbe08b820bd4b6ebb0235.jpg)

![](images/b28c2d4452bb3a4ed1234ae43921b5f94af7e98e6641eba8115ed1cf020748b6.jpg)

![](images/4e665545abd6b8bba8f564a1d501cf686ec195f6fe0103e98040f60de67be65f.jpg)

![](images/cb36d6f3d5bafebd8b4a584592db6a1a884515bc486e9cf143304de490796d6d.jpg)  
Difficulty (cm)

![](images/ab76da94500c393dff93e98d2f6059f3c647b6872b9020ab9bb32291ed37a5bd.jpg)  
Fig. 5: Real-world experimental results. Three methods (DAWN, WMP w/ N, WMP) × three terrains (Stairs, Gap, Step) × two environments (Indoor, Outdoor). SR (%) over 10 trials per difficulty level.

## E. Latent Space Analysis

Fig. 2 (right) shows a t-SNE visualization of the concatenated $\left( h _ { t } , z _ { t } \right)$ . For each terrain type, the clean and noisy clusters overlap, indicating that the encoder maps both to overlapping regions of the latent space rather than separating them by noise condition. This means the encoder has learned to discard depth noise and extract only terrain geometry, achieving the noise invariance that DAWN targets. Meanwhile, different terrain types form well-separated clusters, confirming that the encoder preserves terrain discriminability.

## F. Noise Model Validation

Fig. 3 compares simulated clean depth, simulated noisy depth (×1.0), and real D435i depth across three terrain views. The simulated noise visually matches real sensor output, particularly the missing pixels at depth discontinuities and far particle artifacts at distance, validating that the noise model captures the dominant characteristics of the D435i.

## G. Real-World Depth Reconstruction

Fig. 4 shows DAWN’s RSSM decoder reconstructing clean depth from real D435i input across consecutive timesteps on Stair, Gap, and Step. This confirms that the deterministic state $h _ { t }$ has learned to discard sensor noise. Since the policy conditions on $h _ { t } ,$ it operates on noisy observations without requiring external filters.

## H. Real-World Deployment

Fig. 5 presents real-robot results on a Unitree Go1 in indoor and outdoor environments, with SR measured over 10 trials per difficulty level. In indoor environments, all three methods perform comparably, holding near 100% SR and dropping to 90% only at the hardest Stair (18 cm) and Gap (70 cm); Step shows no degradation at any level. The performance gap emerges outdoors, where sunlight-induced IR interference and surface material variation amplify depth corruption. On Stair at 18 cm, DAWN achieves 80% vs. WMP w/ N 70% and WMP 60%. On Gap at 70 cm, DAWN reaches 60%, WMP w/ N 40%, and WMP 30%. On Step at 45 cm, DAWN records 70% vs. WMP w/ N and WMP both at 50%. This preserves the three-way ordering observed in simulation (§IV-D): DAWN leads, noise-trained WMP w/ N sits between, and clean-trained WMP trails. Noise exposure during training therefore helps outdoors but does not substitute for DAWN’s noise-robust representation, which absorbs the added IR and surface-material corruption without any filter tuning. Fig. 6 shows qualitative snapshots of outdoor deployment across diverse terrains and surface materials.

![](images/61ab91142704d3948c18a9fbefa1c2ad937d225e4330464b6684fb6505832ff3.jpg)  
Fig. 6: Outdoor deployment snapshots: curb climbing on stone terrain, stair descending, slope descending, gravel walking, grass walking, and slope ascending.

## V. CONCLUSION

We presented DAWN, a noise-robust perception framework for quadruped parkour that builds depth noise robustness into the world model’s existing mechanisms, requiring no external modules or environment-specific tuning. DAWN passes noisy depth to the RSSM encoder while keeping clean depth as the reconstruction target, forcing the model to implicitly denoise its input. A contrastive loss further aligns noisy and clean representations at the encoder level. Together, these modifications remove the need for environmentdependent filter tuning at deployment while adding no inference overhead. In simulation, DAWN achieved 96.9% average success rate across stairs, gaps, and steps—close to the clean-depth Oracle—and these improvements transferred consistently to a real Unitree Go1 via zero-shot deployment. Ablation confirmed that the two modifications serve complementary roles: denoising restructures the representation space toward clean geometry, while contrastive learning enforces noise-robust encoding, yielding a compound gain when combined. The current noise model targets the D435i; extending it to other depth sensors is a natural direction for future work.

## ACKNOWLEDGMENT

This research was supported by Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (2018R1A6A1A03025526).

## REFERENCES

[1] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, “Learning agile and dynamic motor skills for legged robots,” Sci. Robot., vol. 4, no. 26, 2019.

[2] J. Tan, T. Zhang, E. Coumans, A. Iscen, Y. Bai, D. Hafner, S. Bohez, and V. Vanhoucke, “Sim-to-real: Learning agile locomotion for quadruped robots,” in Proc. Robot.: Sci. Syst. (RSS), 2018.

[3] A. Kumar, Z. Fu, D. Pathak, and J. Malik, “RMA: Rapid motor adaptation for legged robots,” in Proc. Robot.: Sci. Syst. (RSS), 2021.

[4] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning quadrupedal locomotion over challenging terrain,” Science Robotics, vol. 5, no. 47, p. eabc5986, 2020.

[5] A. Agarwal, A. Kumar, J. Malik, and D. Pathak, “Legged locomotion in challenging terrains using egocentric vision,” in Proc. Conf. Robot Learn. (CoRL), 2022.

[6] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning robust perceptive locomotion for quadrupedal robots in the wild,” Sci. Robot., vol. 7, no. 62, 2022.

[7] Z. Zhuang, Z. Fu, J. Wang, C. Atkeson, S. Schwertfeger, C. Finn, and H. Zhao, “Robot parkour learning,” in Proc. Conf. Robot Learn. (CoRL), 2023.

[8] X. Cheng, K. Shi, A. Agarwal, and D. Pathak, “Extreme parkour with legged robots,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024.

[9] H. Lai, J. Cao, J. Xu, H. Wu, Y. Lin, T. Kong, Y. Yu, and W. Zhang, “World-model-based perception for visual legged locomotion,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2025.

[10] D. Hoeller, N. Rudin, D. Sako, and M. Hutter, “ANYmal parkour: Learning agile navigation for quadrupedal robots,” Sci. Robot., vol. 9, no. 88, p. eadi7566, 2024.

[11] Z. Zhuang, S. Yao, and H. Zhao, “Humanoid parkour learning,” in Proc. Conf. Robot Learn. (CoRL), 2024.

[12] L. Keselman, J. I. Woodfill, A. Grunnet-Jepsen, and A. Bhowmik, “Intel RealSense stereoscopic depth cameras,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPR-W), 2017.

[13] W. Sun, Y. Su, L. Huang, A. Zhang, D. Wei, M. San, D. Tian, E. Cao, F. Yan, E. Xie, and Z. Xie, “Now you see that: Learning end-to-end humanoid locomotion from raw pixels,” arXiv preprint arXiv:2602.06382, 2026.

[14] J. Sun, G. Han, P. Sun, W. Zhao, J. Cao, J. Wang, Y. Guo, and Q. Zhang, “DPL: Depth-only perceptive humanoid locomotion,” arXiv preprint arXiv:2510.07152, 2025.

[15] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring deep neural networks from simulation to the real world,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2017.

[16] C. Sweeney, G. Izatt, and R. Tedrake, “A supervised approach to predicting noise in depth images,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2019.

[17] J. Hu, C. Bao, M. Ozay, C. Fan, Q. Gao, H. Liu, and T. L. Lam, “Deep depth completion from extremely sparse data: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 7, pp. 8244–8264, 2023.

[18] M. Laskin, K. Lee, A. Stooke, L. Pinto, P. Abbeel, and A. Srinivas, “Reinforcement learning with augmented data,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.

[19] X. Gu, Y.-J. Wang, X. Zhu, C. Shi, Y. Guo, Y. Liu, and J. Chen, “Advancing humanoid locomotion: Mastering challenging terrains with denoising world model learning,” in Proc. Robot.: Sci. Syst. (RSS), 2024.

[20] W. Sun, L. Chen, Y. Su, B. Cao, Y. Liu, and Z. Xie, “Learning humanoid locomotion with world model reconstruction,” arXiv preprint arXiv:2502.16230, 2025.

[21] M. S. Ahn, H. Chae, D. Noh, H. Nam, and D. Hong, “Analysis and noise modeling of the Intel RealSense D435 for mobile robots,” in Proc. Int. Conf. Ubiquitous Robots (UR), 2019, pp. 707–711.

[22] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Proc. Int. Conf. Mach. Learn. (ICML), 2020.

[23] D. Chen, B. Zhou, V. Koltun, and P. Krähenbühl, “Learning by cheating,” in Proc. Conf. Robot Learn. (CoRL), 2019, pp. 66–75.

[24] OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang, “Solving Rubik’s Cube with a robot hand,” arXiv preprint arXiv:1910.07113, 2019.

[25] X. Liu, C. Zhang, G. Wang, R. Zhang, and X. Ji, “RaSim: A rangeaware high-fidelity RGB-D data simulation pipeline for real-world applications,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 17 057–17 064.

[26] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, 2018.

[27] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in Proc. Int. Conf. Mach. Learn. (ICML), 2019, pp. 2555–2565.

[28] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[29] D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba, “Mastering Atari with discrete world models,” in Proc. International Conference on Learning Representations (ICLR), 2021.

[30] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse domains through world models,” Nature, vol. 640, no. 8059, pp. 647– 653, 2025.

[31] P. Wu, A. Escontrela, D. Hafner, K. Goldberg, and P. Abbeel, “Day-Dreamer: World models for physical robot learning,” in Proc. Conf. Robot Learn. (CoRL), 2022.

[32] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[33] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020.

[34] A. Srinivas, M. Laskin, and P. Abbeel, “CURL: Contrastive unsupervised representations for reinforcement learning,” in Proc. Int. Conf. Mach. Learn. (ICML), 2020.

[35] J. Long, Z. Wang, Q. Li, L. Cao, J. Gao, and J. Pang, “HIM: Hybrid internal model for agile legged locomotion,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2024.

[36] A. Mousa, N. Karavis, M. Caprio, W. Pan, and R. Allmendinger, “TAR: Teacher-aligned representations via contrastive learning for quadrupedal locomotion,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2025, pp. 11 669–11 676.

[37] Y. Lu, R. Yang, Q. Kou, M. Chen, T. Fan, P. Cui, Y. Dong, and P. Lu, “Contrastive representation learning for robust sim-to-real transfer of adaptive humanoid locomotion,” arXiv preprint arXiv:2509.12858, 2025.

[38] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” arXiv preprint arXiv:1406.1078, 2014.

[39] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[40] N. Tishby, F. C. Pereira, and W. Bialek, “The information bottleneck method,” arXiv preprint physics/0004057, 2000.

[41] M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang, P. Du, A. Serrano-Muñoz, X. Yao, R. Zurbrugg, N. Rudin, et al., “Isaac Lab: A GPUaccelerated simulation framework for multi-modal robot learning,” arXiv preprint arXiv:2511.04831, 2025.