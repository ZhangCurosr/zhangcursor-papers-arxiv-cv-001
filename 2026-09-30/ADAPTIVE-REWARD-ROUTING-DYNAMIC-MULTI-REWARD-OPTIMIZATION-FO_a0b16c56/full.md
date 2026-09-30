# ADAPTIVE REWARD ROUTING: DYNAMIC MULTI-REWARD OPTIMIZATION FOR JOINT AUDIO-VIDEO DIFFUSION VIA FORWARD-PROCESS RL

Songlin Yang<sup>1</sup>, Xiaotong Zhao<sup>2</sup>, Jiacheng Zhang<sup>3</sup>, Zhe Wang<sup>1</sup>, Toyota Li<sup>2</sup>, Eric Liu<sup>2</sup>, Alan Zhao<sup>2</sup>, Anyi Rao<sup>1</sup>

<sup>1</sup>MMLab@HKUST, The Hong Kong University of Science and Technology <sup>2</sup>Tencent Video, <sup>3</sup>The University of Hong Kong

## ABSTRACT

Multi-reward guided reinforcement learning (i.e., RL) offers a promising way to improve joint audio-video diffusion models along several objectives, including modality-specific quality, cross-modal semantic alignment, and temporal synchronization. Its effectiveness, however, depends on two quantities that change during training: where reward-driven updates should act, and how competing rewards should be coordinated. Existing methods tend to rely on fixed routing and reward weights, failing to track evolving model functions. To address these limitations, we propose Adaptive Reward Routing to jointly adapt update locations and reward coordination during forward-process RL (i.e., DiffusionNFT) of joint audio-video diffusion models. Our method consists of two components. (i) Cross-Modal Influence-Guided Routing (Localizing Updates): We use bidirectional cross-attention responses as an efficient proxy for evolving cross-modal influence, dynamically reweighting token-aware losses and scaling gradients across crossmodal layers without additional model interventions. (ii) Preference-Preserving Modality-Aware Reweighting (Coordinating Rewards): We preserve predefined weights as preference priors and use branch-specific reward-gradient interactions as residual corrections after warm-up. This resolves evolving conflicts without letting dominant rewards suppress weak but essential objectives. Extensive experiments demonstrate consistent improvements in modality quality, semantic consistency, and audio-video synchronization over strong RL baselines. Ablations and mechanism analyses further validate the complementary benefits of adaptive update routing and reward coordination.

## 1 INTRODUCTION

Recent advances in joint audio-video diffusion models (HaCohen et al., 2026) have enabled the generation of visual and audio content from text prompts. However, high-quality joint generation must simultaneously satisfy modality-specific visual and audio quality, cross-modal semantic alignment, and temporal synchronization, which are difficult to capture with a single supervised objective. Reward-guided diffusion reinforcement learning (RL), including GRPO-based methods (Guo et al., 2025; Liu et al., 2026a) and DiffusionNFT (Zheng et al., 2025), therefore provides a promising paradigm by expressing these requirements through multiple reward signals.

However, reward-guided RL of joint audio-video diffusion models remains challenging, as it involves dynamic multi-reward optimization along two coupled dimensions: where should rewarddriven updates act, and how should multiple rewards be coordinated? (i) Dynamic Reward Routing: Where to Optimize. Joint audio-video models contain modality-specific branches coupled through cross-attention. Reward routers first determine the responsible modality branches, while the resulting branch-level updates must be localized across tokens and cross-modal interaction layers. OmniNFT (Zhang et al., 2026) recognizes these but fixes its layer routing based on the base model. However, our probing of the base and OmniNFT-trained checkpoints in Fig. 1(a) and (b) shows that cross-modal functions and gradient flows evolve during fine-tuning, making static routing progressively stale. (ii) Dynamic Reward Coordination: How to Balance. Rewards frequently disagree on the same sample, as shown in Fig. 1(c), and their appropriate balance changes throughout optimization. GDPO (Liu et al., 2026e) normalizes each reward but combines them through fixed weights, leaving conflicts unadapted. MARBLE (Zhao et al., 2026) adjusts weights using gradient geometry, but its coefficients reflect gradient compatibility rather than importance aligned with user preferences, as shown in Fig. 1(d). Effective post-training therefore requires conflict-aware adaptation anchored by user-defined priorities.

![](images/7ac75a3482bd9849a603aa5c1f5401947bca3d5567010054a047b3124a0428b6.jpg)  
Figure 1: Joint audio-video post-training requires adaptive update routing and reward coordination. (a) Forward KV Ablation. We ablate cross-modal K/V outputs by layer range and measure DeSync (lower is better; larger degradation indicates greater synchronization importance). Divergent base/trained profiles reveal shifting synchronization-critical layers. (b) Backward Gradient Analysis. We measure layer-wise ℓ norms of Q (within-branch) and K/V (cross-modal) gradients in both branches; shifted base/trained peaks show evolving cross-modal gradient paths. (c) Reward Conflicts. Rewards conflict when their per-prompt normalized preferences have opposite signs, with larger off-diagonal values indicating more disagreement. The conflicts show that fixed branch assignment is insufficient. (d) Gradient-Based Weight Drift. We track weights derived solely from reward-gradient geometry, where near-zero trajectories indicate objective suppression. The vanishing AV-DeSync weight shows that geometry-only weighting can discard essential objectives. Together, these results motivate dynamic routing and preference-anchored reward coordination.

Together, these challenges call for an approach that adapts both reward coordination and update routing as the model evolves. We therefore propose Adaptive Reward Routing for forward-process RL (i.e., DiffusionNFT (Zheng et al., 2025)) of joint audio-video diffusion models. It has two components. (i) Cross-Modal Influence-Guided Routing uses bidirectional cross-attention responses to locate reward-driven updates. Layer aggregation yields token weights that emphasize cross-modally influential locations, while token aggregation yields layer scales that preserve gradients through influential cross-modal pathways. (ii) Preference-Preserving Modality-Aware Reweighting estimates reward conflicts within the branch responsible for each objective. After warm-up, it uses the resulting coefficients as residual corrections to predefined reward weights, adapting to changing conflicts without overriding user priorities.

Our experiments establish three findings. First, Adaptive Reward Routing consistently improves modality quality, semantic alignment, and audio-video synchronization of joint audio-video diffusion models. Second, controlled ablations verify the complementary contributions of token- and layer-level routing, branch-aware conflict estimation, residual preference correction, and warm-up. Third, mechanism analyses with direct path interventions confirm that the cross-attention response proxy identifies functionally important layers and tokens, while routes frozen at initialization become stale as training progresses.

Contributions. (i) We formulate joint audio-video diffusion RL as dynamic multi-reward optimization over two coupled dimensions (i.e., where modality-conditioned reward updates should ac and how multiple rewards should be coordinated) and empirically reveal the limitations of static solutions. (ii) We propose Adaptive Reward Routing, which unifies cross-modal response-guided token/layer localization with preference-preserving, modality-aware reward coordination. (iii) We provide comprehensive comparisons, ablations, and mechanism analyses demonstrating that adapting both dimensions enables more stable and effective joint audio-video post-training.

## 2 RELATED WORK

Joint Audio-Video Generation. Video generation Yang et al. (2026a; 2023) has progressed from image diffusion models with temporal modules (Blattmann et al., 2023; Guo et al., 2024) to large diffusion Transformers (Kong et al., 2024), with flow matching (Wan et al., 2025; HaCohen et al., 2024) and compressed latents improving efficiency (Yang et al., 2026b). Joint audio-video sys tems connect pretrained experts through cross-modal projections (Wang et al., 2025), use unified diffusion Transformers (Liu et al., 2026c;d), or couple separate streams through bidirectional crossattention (HaCohen et al., 2026). These heterogeneous branches enable mutual conditioning but make reward responsibility and gradient routing less obvious than in a single-stream model.

Reinforcement Learning for Diffusion Models. GRPO (Guo et al., 2025) estimates relative advantages without a critic. Flow-GRPO (Liu et al., 2026a) and DanceGRPO (Xue et al., 2025) extend online optimization to flow-based generation through stochastic sampling. DiffusionNFT (Zheng et al., 2025) instead optimizes the forward process using implicit positive and negative policies. OmniNFT (Zhang et al., 2026) adds modality-wise credit assignment for joint audio-video generation. We retain its forward-process formulation but replace fixed routing rules with token- and layer-level routes recomputed from the current model.

Multi-Reward Optimization. Fixed scalarization cannot react to changing conflicts. GDPO (Liu et al., 2026e) preserves reward-specific signals through decoupled normalization, but still uses predefined aggregation weights. Multi-task methods instead seek common descent directions (Desid´ eri,´ 2012; Sener & Koltun, 2018), project conflicting gradients (Yu et al., 2020), or optimize local agreement (Liu et al., 2021). MARBLE (Zhao et al., 2026) adapts this idea to diffusion RL. Because gradient compatibility alone does not encode objective importance or modality responsibility, we estimate conflicts within each modality branch and use them as residuals to preference priors.

## 3 PROBLEM FORMULATION AND PRELIMINARIES

We study reward-guided post-training of joint audio-video diffusion models (i.e., LTX-2 (HaCohen et al., 2026)) under DiffusionNFT (Zheng et al., 2025), which can be formulated as multi-modal, multi-reward forward-process reinforcement learning. We use $m \in \mathcal { M } = \{ v , a \}$ for modality, n for rollout sample, k for reward, i for token, l for Transformer block, and t for flow-matching timestep.

Joint Audio-Video Flow Matching. LTX-2 (HaCohen et al., 2026) uses separate audio and video streams under a shared timestep. Each latent follows the standard linear interpolation $x _ { t } ^ { m } = ( 1 -$ $t ) x _ { 0 } ^ { m } + t x _ { 1 } ^ { m }$ , with $x _ { 1 } ^ { m } \sim \mathcal { N } ( 0 , I )$ , and the model predicts the two velocity fields jointly. The streams exchange information through bidirectional cross-attention:

$$
o _ { a  v } ^ { l , t } = \mathrm { A t t n } \Big ( Q _ { v } ( h _ { v } ^ { l , t } ) , K _ { a } ( h _ { a } ^ { l , t } ) , V _ { a } ( h _ { a } ^ { l , t } ) \Big ) , \quad o _ { v  a } ^ { l , t } = \mathrm { A t t n } \Big ( Q _ { a } ( h _ { a } ^ { l , t } ) , K _ { v } ( h _ { v } ^ { l , t } ) , V _ { v } ( h _ { v } ^ { l , t } ) \Big ) .\tag{1}
$$

The gated audio-to-video (A2V) and video-to-audio (V2A) outputs are added to the video and audio streams, respectively.

Diffusion Forward-Process Reinforcement Learning. DiffusionNFT (Zheng et al., 2025) constructs implicit positive and negative policies from the updated and trainable velocity predictors:

$$
v _ { \theta } ^ { + } = ( 1 - \beta ) v ^ { \mathrm { u p d a t e d } } + \beta v _ { \theta } , \qquad v _ { \theta } ^ { - } = ( 1 + \beta ) v ^ { \mathrm { u p d a t e d } } - \beta v _ { \theta } .\tag{2}
$$

For each prompt, the updated policy generates a group of N samples. The reward of sample n is converted to a group-relative advantage,

$$
A ^ { ( n ) } = \frac { R ^ { ( n ) } - \mu _ { R } } { \sigma _ { R } + \varepsilon } , \qquad r ^ { ( n ) } = \frac { 1 } { 2 } + \frac { 1 } { 2 } \mathrm { c l i p } \left( \frac { A ^ { ( n ) } } { A _ { \mathrm { m a x } } } , - 1 , 1 \right) ,\tag{3}
$$

![](images/304ae1f5a3714d012c42bd26733ba7f4b72906e62f65c3d4ca29772da6d7e6d5.jpg)  
Figure 2: Overview of Adaptive Reward Routing. Our framework adapts reward-driven optimization at four levels: reward reweighting, modality-branch assignment, token-level credit allocation, and layer-wise cross-modal gradient routing.

where $\mu _ { R }$ and $\sigma _ { R }$ are computed within the rollout group. Thus $r ^ { ( n ) } > 1 / 2$ favors the positive policy, whereas $r ^ { ( n ) } < 1 / 2$ favors the negative policy. The resulting objective is

$$
\mathcal { L } _ { \mathrm { N F T } } = \mathbb { E } _ { n , t } \left[ r ^ { ( n ) } \| v _ { \theta } ^ { + } ( x _ { t } ^ { ( n ) } , c , t ) - u ^ { ( n ) } \| _ { 2 } ^ { 2 } + ( 1 - r ^ { ( n ) } ) \| v _ { \theta } ^ { - } ( x _ { t } ^ { ( n ) } , c , t ) - u ^ { ( n ) } \| _ { 2 } ^ { 2 } \right] .\tag{4}
$$

This advantage requires no learned value function: it states only whether a sample performs above or below its peers for the same prompt.

Multi-Reward Optimization. Let $\mathcal { K } = \mathcal { K } _ { v } \cup \mathcal { K } _ { a } \cup \mathcal { K } _ { c }$ denote video, audio, and cross-modal rewards. Eq. 3 is applied independently to each reward, producing $A _ { k } ^ { ( n ) }$ . GDPO (Liu et al., 2026e) combines them using predefined weights, $\begin{array} { r } { A _ { \mathrm { G D P O } } ^ { ( n ) } = \sum _ { k } \omega _ { k } ^ { \mathrm { p r i o r } } A _ { k } ^ { ( n ) } } \end{array}$ . MARBLE (Zhao et al., 2026) instead chooses simplex weights that minimize the norm of the weighted sum of normalized reward gradients. GDPO therefore preserves explicit preferences but cannot adapt to conflicts, while MARBLE adapts to local gradient geometry but does not encode preference or modality responsibility.

## 4 METHOD: ADAPTIVE REWARD ROUTING

## 4.1 OVERVIEW

We propose Adaptive Reward Routing, a forward-process RL framework that adapts both reward priorities and routing locations, as shown in Fig. 2. It contains two components. First, Cross-Modal Influence-Guided Routing (Sec. 4.2) determines where the update should act by adapting token weights and layer-wise cross-modal gradient flow. Second, Preference-Preserving Modality-Aware Reweighting (Sec. 4.3) determines how rewards should be combined within the video and audio branches. As shown in Algorithm 1, the complete optimization flow is

$$
\begin{array} { r } { \left\{ A _ { k } \right\} \xrightarrow [ { ( \mathrm { S e c . } 4 . 3 ) } ] { \mathrm { ~ r e w a r d ~ r e w e i g h t i n g } } \left\{ \omega _ { m , k } A _ { k } \right\} \xrightarrow [ { ( \mathrm { S e c . } 4 . 3 ) } ] { \mathrm { b r a n c h ~ r o u t i n g } } \left( A _ { v } , A _ { a } \right) \xrightarrow [ { ( \mathrm { S e c . } 4 . 2 ) } ] { \mathrm { t o k e n ~ r o u t i n g } } \mathcal { L } \xrightarrow [ { ( \mathrm { S e c . } 4 . 2 ) } ] { \mathrm { l a y e r ~ r o u t i n g } } \nabla _ { \theta } \mathcal { L } . } \end{array}\tag{5}
$$

Intuitively, reward reweighting decides how strongly each objective contributes, branch routing assigns objectives to target modalities, token routing selects where each modality loss is emphasized, and layer routing controls how the resulting gradient crosses modality boundaries.

Algorithm 1 Adaptive Reward Routing   
Require: Policy v , updated policy $v ^ { \mathrm { u p d a t e d } }$ , and smoothed coefficients $\bar { \gamma } _ { m }$   
1: for each training round e do   
2: Generate a group of samples with $v ^ { \mathrm { u p d a t e d } }$ and collect pre-gate A2V/V2A responses during sampling   
3: Evaluate $\{ \bar { R } _ { k } ^ { ( n ) } \}$ and independently normalize each reward to obtain $\{ A _ { k } ^ { ( n ) } \}$   
4: Compute token weights $\{ \lambda _ { m , i } ^ { ( n ) } \}$ using Eq. 7 and layer routes $\{ \alpha _ { m } ^ { l } \}$ using Eq. 8   
5: Compute $\{ \omega _ { m , k } \}$ from the cached $\bar { \gamma } _ { m }$ using $\mathrm { E q . 1 0 }$   
6: Construct $A _ { m } ^ { ( n ) }$ using Eq. 11 and set $r _ { m } ^ { ( n ) } = r ( A _ { m } ^ { ( n ) } )$   
7: $\mathbf { i f } e \geq e _ { \mathrm { w a r m } }$ and e is a refresh round then   
8: Probe reward gradients through their responsible branches with uniform token weights   
9: Solve the branch-wise MARBLE problem and update $\bar { \gamma } _ { m }$ for the next round   
10: end if   
11: Compute $\ell _ { m , i } ^ { ( n ) } , \mathcal { L } _ { m } ^ { \mathrm { p o l i c y } }$ , and $\mathcal { L } ( \boldsymbol { \theta } )$ using Eqs. 12–14   
12: Backpropagate through the routed KV paths in Eq. 9 and update θ   
13: Update $\chi ^ { \mathrm { { u p d a t e d } } }$ according to the DiffusionNFT policy-update schedule   
14: end for

## 4.2 CROSS-MODAL INFLUENCE-GUIDED ROUTING

A direct measure of directional influence would disable A2V or V2A and compare the velocity predictions. Repeating this intervention during training would require extra model evaluations. We instead use a quantity already produced by the forward pass: the pre-gate response of the corresponding cross-attention path. For target token i,

$$
d _ { v , i } ^ { l , t } = \| o _ { a  v , i } ^ { l , t } \| _ { 2 } , \qquad d _ { a , i } ^ { l , t } = \| o _ { v  a , i } ^ { l , t } \| _ { 2 } .\tag{6}
$$

These directional responses are collected over an intermediate-to-late denoising window and detached before policy optimization. Sec. 5.4 validates their relationship to direct interventions.

Token-Level Routing. For each target token, we average its responses over the selected timesteps and cross-modal blocks. After percentile-clipped min–max normalization (Norm ), the score becomes a positive loss weight:

$$
\lambda _ { m , i } = 1 + \left( \lambda _ { \operatorname* { m a x } } - 1 \right) \operatorname { N o r m g } \left( \frac { 1 } { | \mathcal { B } | | \mathcal { T } | } \sum _ { l \in \mathcal { B } } \sum _ { t \in \mathcal { T } } d _ { m , i } ^ { l , t } \right) .\tag{7}
$$

Audio responses are normalized globally, while video responses are normalized within each frame to prevent frame-level magnitude differences from dominating the weights. These weights are applied to the token-level negative-aware loss in Sec. 4.4.

Layer-Level Routing. For each layer, we instead average the same response over tokens and selected timesteps. Let $\widetilde { \delta } _ { m } ^ { l }$ denote this layer score after min–max normalization across blocks. We convert it to a soft detachment coefficient

$$
\alpha _ { m } ^ { l } = ( 1 - \widetilde { \delta } _ { m } ^ { l } ) ^ { 1 / \tau } .\tag{8}
$$

For a source key or value tensor $X \in \{ K , V \}$ , the routed representation is

$$
\widetilde { X } _ { \bar { m }  m } ^ { l , t } = \alpha _ { m } ^ { l } \mathrm { s g } ( X _ { \bar { m } } ^ { l , t } ) + ( 1 - \alpha _ { m } ^ { l } ) X _ { \bar { m } } ^ { l , t } .\tag{9}
$$

This operation leaves the forward value unchanged but scales its backward gradient by $1 - \alpha _ { m } ^ { l }$ Strongly influential layers retain more gradient, while weakly coupled layers are increasingly detached. A2V and V2A are routed independently.

## 4.3 PREFERENCE-PRESERVING MODALITY-AWARE REWEIGHTING

Motivation. Predefined reward weights express what the user wants, but they cannot react to reward conflicts. Gradient-based coefficients react to conflicts, but may suppress a weak objective because its early gradient is noisy or incompatible. We combine the two rather than choosing one.

Implementation. Each reward is probed only through the branch it supervises: video and audio rewards use their respective branches, while cross-modal rewards use both. MARBLE then produces a conflict-aware coefficient $\gamma _ { m , k }$ within each branch. Token routing is disabled during these probes so that the measured geometry is not biased by the current token weights. After warm-up, the smoothed coefficient provides a residual correction to the prior:

$$
\omega _ { m , k } = \left\{ \begin{array} { l l } { \omega _ { m , k } ^ { \mathrm { p r i o r } } , } & { e < e _ { \mathrm { w a r m } } , } \\ { ( 1 - \kappa ) \omega _ { m , k } ^ { \mathrm { p r i o r } } + \kappa C _ { m } \bar { \gamma } _ { m , k } , } & { e \geq e _ { \mathrm { w a r m } } . } \end{array} \right.\tag{10}
$$

Here $C _ { m }$ rescales the simplex coefficients to preserve the total prior weight within branch $m ,$ , and $\bar { \gamma } _ { m } \gets \rho \bar { \gamma } _ { m } + ( 1 - \rho ) \gamma _ { m } ^ { * }$ smooths successive estimates. The prior therefore sets a nonzero floor, while the residual term adapts to current conflicts.

## 4.4 TRAINING OBJECTIVE

The adaptive reward weights first produce a separate advantage for each modality branch:

$$
A _ { m } ^ { ( n ) } = \sum _ { k \in { \cal K } _ { m } \cup { \cal K } _ { c } } \omega _ { m , k } A _ { k } ^ { ( n ) } , \qquad m \in \{ v , a \} .\tag{11}
$$

Cross-modal rewards are included in both branches. We then map each branch advantage to an optimality probability $r _ { m } ^ { ( n ) } = r ( A _ { m } ^ { ( n ) } )$ ) using Eq. 3. For token i of sample $n ,$ the negative-aware loss is

$$
\ell _ { m , i } ^ { ( n ) } = r _ { m } ^ { ( n ) } \frac { \| v _ { \theta , m , i } ^ { + } - u _ { m , i } \| _ { 2 } ^ { 2 } } { w _ { m } ^ { + , ( n ) } + \varepsilon } + \big ( 1 - r _ { m } ^ { ( n ) } \big ) \frac { \| v _ { \theta , m , i } ^ { - } - u _ { m , i } \| _ { 2 } ^ { 2 } } { w _ { m } ^ { - , ( n ) } + \varepsilon } .\tag{12}
$$

Here $w _ { m } ^ { \pm , ( n ) }$ is the detached mean absolute residual of the corresponding policy, averaged over all tokens and feature dimensions of modality m. The token routing weights then form the modality loss

$$
\mathcal { L } _ { m } ^ { \mathrm { p o l i c y } } = \mathbb { E } _ { n } \left[ \frac { \sum _ { i \in \mathbb { Z } _ { m } } \lambda _ { m , i } ^ { ( n ) } \ell _ { m , i } ^ { ( n ) } } { \sum _ { i \in \mathbb { Z } _ { m } } \lambda _ { m , i } ^ { ( n ) } } \right] .\tag{13}
$$

Finally, we combine the two branches and regularize them toward the fixed reference policy:

$$
\mathcal { L } ( \theta ) = \sum _ { m \in \mathcal { M } } \mathcal { L } _ { m } ^ { \mathrm { p o l i c y } } + \lambda _ { \mathrm { K L } } \sum _ { m \in \mathcal { M } } \mathcal { L } _ { \mathrm { K L } , m } ( \theta ) .\tag{14}
$$

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Backbones and Training Data. We evaluate Adaptive Reward Routing on two joint audio-video diffusion backbones, LTX-2 (19B) and LTX-2.3 (22B) (HaCohen et al., 2026). Both models employ separate audio and video streams connected through bidirectional cross-attention, making them suitable for studying adaptive reward localization across modalities, tokens, and layers. For reward-guided post-training, we use 19,487 audio-video prompts collected from a VGGSoundderived (Chen et al., 2020) corpus. Each record contains modality-specific audio and video descriptions together with a joint audio-video prompt.

Reward Models. Following the multi-objective evaluation dimensions of joint audio-video generation, we optimize five complementary reward signals: (i) Video Quality: VideoAlign (Liu et al., 2026b) and HPSv3 (Ma et al., 2025); (ii) Audio Quality: AudioBox Aesthetics (Tjandra et al., 2025); (iii) Text-Audio Alignment: CLAP (Wu et al., 2023); and (iv) Audio-Video Synchronization: DeSync (Iashin et al., 2024), which we convert into a higher-is-better AV-DeSync during training, while reporting the original lower-is-better DeSync metric at evaluation time.

Baselines. We compare against complementary reference settings that cover the principal dimensions of multimodal, multi-reward optimization. (i) No Post-Training: the pretrained backbone establishes the performance before reward-guided optimization. (ii) Fixed Reward Coordination: GDPO (Liu et al., 2026e) independently normalizes reward-wise advantages but aggregates them using fixed weights. (iii) Conflict-Aware Reward Coordination: MARBLE (Zhao et al., 2026) dynamically adjusts reward weights according to global gradient conflicts, without modality-specific routing. (iv) Static Multimodal Routing: OmniNFT (Zhang et al., 2026), designed specifically for DiffusionNFT-based joint audio-video post-training, introduces modality-wise credit assignment, layer-wise gradient surgery, and region-wise reweighting, but fixes its routing strategy using the base model. OmniNFT\* denotes the checkpoint released by the original authors.

![](images/3f179b1127358870c125e9682e1e369004806918f6f4a67d9ef85e0893d4b583.jpg)  
In a modern news studio, a female anchor in a dark blazer stands before a large screen displaying a nighttime accident. Responders attend to an injured person near a damaged silver car, illuminated by flashing emergency lights. An ‘ai’ watermark is in the bottom-right. Maintaining a serious expression, she speaks: “德克萨斯州发生了一起事故，造成两人受 伤，暂无死亡报告。”

![](images/739c758dbea99cc562908010cd7dbccfd2ba84ff7f89a5840d05a791be364b50.jpg)  
A close-up shows a whimsical humanoid character made of green leafy vegetables in a sunlit garden. With large white eyes and leafy hands, he raises his arms, gestures animatedly. He then leans eagerly toward the camera and says, “Oh, how did you find me? Although it is a bit inappropriate, for the sake of your health, I hope you remember to eat more vegetables.”

![](images/0e91f65fd547b2bf0f8a6d644544e7a89c6fcdcfbfff277900f3abacf2e12d8a.jpg)  
In a medium shot with cool tones, two men in light blue prison shirts lean against a rough stone wall under diffused daylight. The older Black man on the right, whose shirt bears the number “302”. Beside him, a white man with short brown hair, wearing the number “574”. The older man says gently, “But that doesn’t mean you’re a murderer.” The younger man replies, “Someone else must have killed my wife.”.

![](images/a68af1e84cc451c26d6ac68b2d7cf14a9611ea683daec7fa89906d4db68d861b.jpg)  
A low close medium shot of a German shepherd in a fenced backyard on packed earth and patchy grass, body angled off-frame; its dense tan-and-black coat has a dark saddle marking and ears pricked upright, with a weathered timber fence and blurred hedge behind it under flat overcast light. The dog barks sharply several times, then pants, as deep dog barks arrive in an uneven series over quiet outdoor ambience with breaths between calls

Figure 3: Qualitative comparison of joint audio-video generation. For each prompt, we show five temporally ordered frames generated by LTX-2, OmniNFT, and Ours. The examples cover Chinese news delivery, stylized English speech, two-speaker dialogue, and dog barking. Our method better preserves the requested subjects and scene composition while reducing identity and appearance drift throughout the generated sequence.  
Table 1: Main results on JavisBench. (Mean ± std over 3 seeds.)
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="2">AV-Quality</td><td colspan="4">Text-Consistency</td><td colspan="2">AV-Consistency</td><td colspan="2">AV-Synchrony</td></tr><tr><td>VQ↑</td><td>AQ↑</td><td>TV-IB↑</td><td>TA-IB↑</td><td>CLIP↑</td><td>CLAP↑</td><td>AV-IB↑</td><td>AVHScore↑</td><td>JavisScore↑</td><td>DeSync↓</td></tr><tr><td rowspan="7">LTX-2</td><td>Base Model</td><td>1.883</td><td>5.201</td><td>0.265</td><td>0.143</td><td>0.312</td><td>0.358</td><td>0.180</td><td>0.177</td><td>0.153</td><td>0.604</td></tr><tr><td>+ GDPO</td><td>2.722 ±0.0136</td><td>5.450 ±0.0217</td><td>0.261 ±0.0020</td><td>0.138 ±0.0018</td><td>0.312 ±0.0014</td><td>0.347±0.0031</td><td>0.174 ±0.0025</td><td>0.175 ±0.0022</td><td>0.155±0.0020</td><td>0.671 ±0.0104</td></tr><tr><td>+ MARBLE</td><td>2.384 ±0.0124</td><td>5.100 ±0.0193</td><td>0.265 ±0.0016</td><td>0.138 ±0.0019</td><td>0.311 ±0.0013</td><td>0.365 ±0.0028</td><td>0.182 ±0.0023</td><td>0.182 ±0.0026</td><td>0.158 ±0.0021</td><td>0.618 ±0.0092</td></tr><tr><td>+ OmniNFT</td><td>3.136 ±0.0108</td><td>5.614 ±0.0176</td><td>0.265 ±0.0018</td><td>0.145 ±0.0016</td><td>0.312 ±0.0011</td><td>0.416 ±0.0024</td><td>0.222 ±0.0019</td><td>0.219 2±0.0021</td><td>0.195 ±0.0017</td><td>0.390 ±0.0075</td></tr><tr><td>+ OmniNFT*</td><td>3.278</td><td>5.609</td><td>0.254</td><td>0.165</td><td>0.314</td><td>0.422</td><td>0.220</td><td>0.219</td><td>0.193</td><td>0.378</td></tr><tr><td>+ Ours</td><td>3.336 ±0.0094</td><td>5.868 ±0.0158</td><td>0.268 ±0.0014</td><td>0.167±0.0013</td><td>0.314 ±0.0011</td><td>0.425 ±0.0020</td><td>0.235 ±0.0017</td><td>0.234 ±0.0019</td><td>0.206 ±0.0015</td><td>0.341 ±0.0062</td></tr><tr><td>Base Model</td><td>2.032</td><td>5.218</td><td>0.271</td><td>0.151</td><td>0.308</td><td>0.387</td><td>0.205</td><td>0.202</td><td>0.175</td><td>0.504</td></tr><tr><td rowspan="5">LTX-2.3</td><td>+ GDPO</td><td>2.929</td><td>5.251</td><td>0.272</td><td>0.144</td><td>0.309</td><td>0.376</td><td>0.218</td><td>0.199</td><td>0.178</td><td></td></tr><tr><td>+ MARBLE</td><td>±0.0142 2.582</td><td>±0.0225 5.176</td><td>±0.0021</td><td>±0.0020</td><td>±0.0015 0.309</td><td>±0.0033 0.394</td><td>±0.0027 0.217±0.0022</td><td>±0.0025 0.207</td><td>±0.0022 0.182</td><td>0.560 ±0.0113</td></tr><tr><td>+ OmniNFT</td><td>±0.0130 3.489</td><td>±0.0206 5.693</td><td>0.271±0.0018 0.271</td><td>0.147 ±0.0017 0.163</td><td>±0.0014 0.316</td><td>±0.0029 0.449</td><td>0.250</td><td>±0.0028 0.238</td><td>±0.0020</td><td>0.496 ±0.0098</td></tr><tr><td>+ OmniNFT*</td><td>±0.0115 3.537</td><td>±0.0181 5.702</td><td>±0.0016 0.260</td><td>±0.0015 0.174</td><td>±0.0010 0.311</td><td>±0.0025 0.456</td><td>±0.0021 0.252</td><td>±0.0023 0.249</td><td>0.224 ±0.0018 0.221</td><td>0.369 ±0.0079</td></tr><tr><td>+ Ours</td><td>3.599 0.0089</td><td>5.979 ±0.0151</td><td>0.274 ±0.0013</td><td>0.177 ±0.0012</td><td>0.314 ±0.0011</td><td>0.460 ±0.0019</td><td>0.267 ±0.0016</td><td>0.266 ±0.0017</td><td>0.236 ±0.0014</td><td>0.335 0.302 0.0058</td></tr></table>

Evaluation. We evaluate on the full JavisBench benchmark (Liu et al., 2026d), which contains 10,140 prompts spanning diverse audio-video generation scenarios. Generated outputs are normalized to the benchmark protocol of four seconds, 24 FPS, and 16-kHz audio. We report four groups of metrics: (i) AV-Quality (Liu et al., 2026b), including Visual Quality (VQ) and Audio Quality (AQ); (ii) Text-Consistency, including Text-Video and Text-Audio ImageBind similarity (TV-IB and TA-IB) (Girdhar et al., 2023), CLIP (Radford et al., 2021), and CLAP; (iii) AV-Consistency (Girdhar et al., 2023), including AV-IB and AVHScore; and (iv) AV-Synchrony, including JavisScore (Liu et al., 2026c) and DeSync.

![](images/202e69d4abd7e47cb44ca70abb6d1f41e0f6637c02eb5abd6460db3a2c013644.jpg)  
(b) Ablation Study

![](images/05dbff2e49c46b02d9d374ff38230e97d00112555e3e76a82ba462ea157acc70.jpg)

![](images/dc7ba2eea764c8b9edb7b3a5b1889cc9b6bb9bccd984c8d91be2565de77eff23.jpg)

![](images/557eda5c8edc92304da50608a3b205556164ada2b273a84cb6654fd5fa962a33.jpg)

![](images/9a9bb426b77e60b6f1c08ef8aaf4145c7233c5191c57f7c52ef4ac768214044c.jpg)

![](images/a85842e73c702169547df95b760dbc256d1228982de9b4098bc7a8c26ed01267.jpg)  
(a) Reward Dynamics Across Training

![](images/e0172a17f3d16e5712071315f23f3aa6fbdec468b1ff0856ad68c41b8fcd06ac.jpg)

![](images/4dffa187c8059ce6d3c237c50e13c3518acbf88fd63351844a104bc19cca084c.jpg)  
Figure 4: Reward dynamics and component ablations. (a) Training trajectories of five individual rewards (AudioBox, AV-DeSync, CLAP, HPSv3, and VideoAlign) and their average for GDPO, MARBLE, OmniNFT, and Ours. Average denotes the arithmetic mean of the five normalized rewards. (b) Ablation studies of the routing and weighting designs. Starting from GDPO or MARBLE, respectively, each added component yields progressive gains, while the complete method achieves the strongest overall performance.

## 5.2 MAIN RESULTS AND TRAINING DYNAMICS

Fig. 3, Tab. 1, and Fig. 4 summarize the generation quality, benchmark performance, and optimization behavior of our method, respectively. (i) Qualitative Results. Fig. 3 covers diverse audio-video scenarios, including multilingual speech, a stylized speaking character, a two-speaker exchange, and animal vocalization. LTX-2 exhibits noticeable subject and appearance drift, particularly in the character and animal examples, while OmniNFT improves prompt fidelity but retains temporal inconsistencies. Our method maintains more stable identities and scene structures while preserving the visual actions associated with speech, dialogue, and barking. (ii) Quantitative Results. As shown in Tab. 1, our method achieves the strongest overall performance on both LTX-2 and LTX-2.3, obtaining the best result on nine of the ten metrics under each backbone. For each backbone, GDPO, MARBLE, OmniNFT, and Ours are independently trained with three random seeds under the same data, LoRA, and optimization budgets, and the table reports their arithmetic means. The Base Model and OmniNFT\* are fixed checkpoints evaluated under the same generation and evaluation protocol, with OmniNFT\* denoting the checkpoint released by its original authors. GDPO improves visual quality but degrades several audio and synchronization metrics, revealing the imbalance caused by fixed reward aggregation. MARBLE alleviates reward conflicts globally, and OmniNFT introduces modality-aware optimization, but neither adapts both reward coordination and update routing to the evolving model. Our method consistently improves modality quality, semantic consistency, crossmodal consistency, and synchronization, with the same trend across both backbones. (iii) Training Dynamics. Fig. 4(a) shows that our method reaches the highest average reward while maintaining favorable trajectories across all five component rewards. In contrast, the baselines make less balanced progress across audio, video, and synchronization objectives. This indicates that the final gains arise from coordinated multi-reward optimization rather than improving one objective at the expense of others.

## 5.3 ABLATION STUDIES

(i) Cross-Modal Influence-Guided Routing. The routing ablation in Tab. 2 progressively introduces token weighting and layer scaling over GDPO. Token weighting improves local credit assignment by emphasizing tokens with stronger cross-modal responses, while layer scaling further improves consistency and synchronization by preserving gradients through influential interaction layers. Combining the two produces the strongest routing-only configuration, confirming that tokenand layer-level adaptation are complementary. (ii) Preference-Preserving Reward Coordination.

Table 2: Results of the ablation studies on JavisBench using the LTX-2 backbone. Gray-shaded rows isolate the reward-coordination components from the MARBLE baseline and do not inherit the routing stack. (Mean ± std over 3 seeds.)
<table><tr><td rowspan="2">Study</td><td rowspan="2">Configuration</td><td colspan="2">AV-Quality</td><td colspan="5">Text-Consistency</td><td colspan="2">AV-Consistency</td><td colspan="2">AV-Synchrony</td></tr><tr><td>VQ↑</td><td>AQ↑</td><td>TV-IB↑</td><td>TA-IB↑</td><td>CLIP↑</td><td></td><td>CLAP↑</td><td>AV-IB↑</td><td>AVHScore↑</td><td>JavisScore↑</td><td>DeSync↓</td></tr><tr><td rowspan="3">Routing</td><td>+ Token Weighting</td><td>3.008 ±0.0127</td><td>5.663 ±0.0202</td><td>0.263 ±0.0018</td><td>0.156±0.0016</td><td>0.313</td><td>±0.0013 0.388</td><td>±0.0027 0.210</td><td>±0.0022</td><td>0.201 ±0.0024</td><td>0.179 ±0.0019</td><td>0.482 ±0.0088</td></tr><tr><td>+ Layer Scale</td><td>3.192 ±0.0111</td><td>5.784 ±0.0188</td><td>0.264 ±0.0017</td><td>0.161 ±0.0015</td><td>0.313 ±0.0012</td><td>0.409</td><td>0.221 ±0.0025</td><td>±0.0020 0.227</td><td>±0.0022</td><td>0.189 ±0.0018</td><td>0.366 ±0.0072</td></tr><tr><td>+ Token + Layer</td><td>3.315 ±0.0103</td><td>5.839 ±0.0171</td><td>0.266 ±0.0015</td><td>0.162 ±0.0014</td><td>0.313 ±0.0011</td><td>0.411</td><td>±0.0023 0.226</td><td>±0.0018</td><td>0.229 0.0020</td><td>0.190 ±0.0016</td><td>0.343 ±0.0065</td></tr><tr><td rowspan="3">Weighting</td><td>+ Branch-Aware</td><td>2.612 ±0.0131</td><td>5.290 ±0.0211</td><td>0.264 ±0.0019</td><td>0.149 ±0.0017</td><td>0.311±0.0014</td><td>0.384</td><td>±0.0030 0.198</td><td>±0.0024 0.188</td><td>±0.0027</td><td>0.177±0.0021</td><td>0.492 ±0.0096</td></tr><tr><td>+ Residual</td><td>2.891±0.0118</td><td>5.526±0.0196</td><td>0.264 ±0.0016</td><td>0.158±0.0018</td><td>0.312±0.0013</td><td>0.393</td><td>±0.0026 0.199</td><td>±0.0021 0.199</td><td>±0.0023</td><td>0.180±0.0019</td><td>0.455 ±0.0084</td></tr><tr><td>+ Warm-Up</td><td>2.999 ±0.0106</td><td>5.791 ±0.0179</td><td>0.265 ±0.0017</td><td>0.157 ±0.0014</td><td>0.312±0.0012</td><td>0.404</td><td>±0.0024 0.209</td><td>±0.0019 0.201</td><td>±0.0021 0.183</td><td>±0.0017</td><td>0.369 ±0.0070</td></tr><tr><td></td><td>Routing + Weighting (Ours)</td><td>3.336 ±0.0094</td><td>5.868 ±0.0158</td><td>0.268 ±0.0014</td><td>0.167 ±0.0013</td><td>0.314 ±0.0011</td><td>0.425</td><td>±0.0020 0.235</td><td>0.234 0.0017</td><td>0.0019</td><td>0.206 ±0.0015</td><td>0.341 ±0.0062</td></tr></table>

![](images/60ac396befb7aa32b52da500a83ff24bf72116104ce3295054381ee36e3d0ebc.jpg)

![](images/37532b55ea876030bffb0343bf188e5cbc00764479b5a5df56fe750f0fe2881f.jpg)

![](images/155795d637cd3e6f67d3b40424037efacfde71cad7ddfef0e6166edfb9aa0e98.jpg)  
Figure 5: The proxy correctly identifies influential cross-modal layers and tokens. (a) Each point is one layer. The x-axis is the layer’s mean pre-gate A2V/V2A response norm, while the y-axis is the relative change in the corresponding final video/audio velocity after disabling that layer. Both axes are min–max normalized across the 48 layers for visualization. Points near the diagonal indicate agreement between the proxy and direct intervention. (b) Higher correlation means better agreement with the current model. The current proxy stays accurate, whereas the proxy fixed at initialization becomes inaccurate as training changes the model. (c) We block the cross-modal information received by the top-scoring, random, or bottom-scoring 10% of target tokens. A larger final prediction change means that the blocked tokens were more influential. The top-scoring tokens consistently cause the largest change, confirming that the proxy correctly identifies important tokens.

The gray rows isolate the weighting components from the MARBLE baseline without inheriting the routing stack. Branch-aware balancing assigns reward interactions to their responsible modality branches, while residual mixing preserves the predefined preference prior instead of replacing it with gradient-derived coefficients. Warm-up further stabilizes this adaptation by delaying dynamic reweighting until the estimated gradient relationships become reliable. (iii) Complementarity and Reward Trade-Offs. Individual components may favor different objectives, so intermediate configurations do not necessarily improve every metric monotonically. Nevertheless, progressively incorporating routing and weighting produces a stronger overall balance across quality, semantic consistency, and synchronization, as also reflected in Fig. 4(b). Because the routing chain keeps the GDPO weighting fixed and the weighting chain inherits no routing, each chain isolates a single axis. The complete model performs best overall, demonstrating that adaptive update localization and preference-preserving reward coordination address distinct but complementary failure modes.

## 5.4 VALIDATING THE CROSS-MODAL INFLUENCE PROXY

We verify whether the response proxy identifies the cross-modal paths that actually affect the model output. At four training checkpoints, we disable each of the 48 A2V or V2A blocks separately while fixing the prompt, noisy latent, and timestep; a larger change in the final prediction indicates a more influential path. (i) Layer-Level Fidelity: The proxy closely recovers the intervention-based layer ranking, with Spearman correlations of 0.98 for A2V and 0.97 for V2A. Since a value close to 1 means nearly identical rankings, the proxy reliably identifies which layers matter (Fig. 5(a)). (ii) Dynamic Tracking: The proxy recomputed from the current model remains above 0.96 throughout fine-tuning, whereas the proxy frozen at initialization falls to 0.56 and 0.38. Thus, influential layers shift during training, and a fixed routing map becomes stale (Fig. 5(b)). (iii) High-Score Tokens Matter More: We block the cross-modal responses of the top-scoring, random, or bottom-scoring

10% of target tokens. Blocking the top-scoring group changes the final prediction 1.62× more for A2V and 1.74× more for V2A than blocking an equally sized random group. Since the ablation size is identical, this result directly shows that higher proxy scores identify tokens with greater functional influence.

## 6 CONCLUSION

This work reframes multi-reward post-training for joint audio-video generation as a dynamic creditassignment problem. As optimization reshapes the model’s cross-modal functions, both the token/layer localization of modality-conditioned updates and the relative strengths of reward signals should evolve accordingly. Adaptive Reward Routing embodies this principle by tracking changing cross-modal influence and reward conflicts throughout training, while preserving user-defined preferences. Its consistent gains across backbones, metrics, and ablations demonstrate the importance of adapting the optimization process in step with the model itself. More broadly, our findings point toward multimodal learning systems in which feedback is not routed by a fixed recipe, but continually reorganized as the model acquires new capabilities.

Limitations and Future Work. (i) Reward Models: No established unified reward jointly captures modality quality, semantic consistency, and temporal synchronization. Human-preference models (Huang et al., 2026) provide a complementary overall signal but do not replace fine-grained, modality-specific supervision. Developing a comprehensive audio-video reward remains an important direction. (ii) RL Framework: We adopt DiffusionNFT for stable, direct supervision of sampled flow-matching timesteps without reverse-process likelihood estimation. Our routing can extend to other diffusion objectives through their token losses and gradient paths, while reward coordination requires only reward-wise gradients. (iii) Architectural Scope: We validate our method on both dual-stream LTX backbones and the unified single-stream JavisDiT++ backbone. Routing applies when modality tokens are identifiable and directional interaction responses can be isolated, whereas reward coordination is architecture-independent. Models with inseparable modality repre sentations or inaccessible interaction responses remain future work.

## REFERENCES

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audiovisual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 721–725. IEEE, 2020.

Jean-Antoine Desid´ eri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization.´ Comptes Rendus. Mathematique ´ , 350(5-6):313–318, 2012.

Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind one embedding space to bind them all. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15180–15190. IEEE, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. International Conference on Learning Representations, 2024.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

Yinming Huang, Shuyuan Tu, Xi Yan, Zihan Yang, Jianhua Han, Xu Hang, Yu-Gang Jiang, and Zuxuan Wu. Va-judger: Reward modeling from human preference feedback for joint video-audio generation, 2026. URL https://arxiv.org/abs/2608.18607.

Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchro nization from sparse cues. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5325–5329. IEEE, 2024.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in neural information processing systems, 34:18878–18890, 2021.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026a.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. Advances in Neural Information Processing Systems, 38:82155–82192, 2026b.

Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Jiebo Luo, Ziwei Liu, Hao Scofield Fei, et al. Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. In International Conference on Learning Representations, volume 2026, pp. 139160–139194, 2026c.

Kai Liu, Yanhao Zheng, Kai Wang, Shengqiong Wu, Rongjunchen Zhang, Jiebo Luo, Dimitrios Hatzinakos, Ziwei Liu, Hao Scofield Fei, and Tat-Seng Chua. Javisdit++: Unified modeling and optimization for joint audio-video generation. In International Conference on Learning Representations, volume 2026, pp. 150592–150618, 2026d.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, and Pavlo Molchanov. GDPO: Group reward-decoupled normalization policy optimization for multi-reward RL optimization. In Forty-third International Conference on Machine Learning, 2026e.

Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15086–15095. IEEE, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in Neural Information Processing Systems, 31, 2018.

Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic assessment for speech, music and sound. In 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 1–8. IEEE, 2025.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025.

Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Songlin Yang, Wei Wang, Jun Ling, Bo Peng, Xu Tan, and Jing Dong. Context-aware talking-head video editing. In Proceedings of the 31st ACM International Conference on Multimedia, pp. 7718–7727, 2023.

Songlin Yang, Zhe Wang, Xuyi Yang, Songchun Zhang, Xianghao Kong, Taiyi Wu, Xiaotong Zhao, Ran Zhang, Alan Zhao, and Anyi Rao. Shotverse: Advancing cinematic camera control for textdriven multi-shot video creation. arXiv preprint arXiv:2603.11421, 2026a.

Songlin Yang, Haobin Zhong, Ruilin Zhang, Xiaotong Zhao, Shuai Li, Kai Zheng, Xuyi Yang, Zhe Wang, Zhenchen Tang, Yang Li, et al. Evalverse: Pipeline-aware and expert-calibrated benchmarking for professional cinematic video generation. arXiv preprint arXiv:2605.23271, 2026b.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in neural information processing systems, 33: 5824–5836, 2020.

Guohui Zhang, XiaoXiao Ma, Jie Huang, Hang Xu, Hu Yu, Siming Fu, Yuming Li, Zeyue Xue, Lin Song, Haoyang Huang, et al. Omninft: Modality-wise omni diffusion reinforcement for joint audio-video generation. arXiv preprint arXiv:2605.12480, 2026.

Canyu Zhao, Hao Chen, Yunze Tong, Yu Qiao, Jiacheng Li, and Chunhua Shen. Marble: Multiaspect reward balance for diffusion rl. arXiv preprint arXiv:2605.06507, 2026.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.