# DRIVEUNITOK: A UNIFIED MULTIMODAL TOKENIZER FOR DRIVING WORLD-ACTION MODELS

Jingbo Xu<sup>1,2</sup>, Hanchao Leng<sup>2,∗</sup>, JiaHui Huang<sup>2</sup>, Hanbing Li<sup>2</sup>, Jingwei Zhao<sup>2</sup>, Kuiyuan Yang<sup>2</sup>, Chen Gao<sup>1</sup>, Guang Chen<sup>2</sup>, Hangjun Ye<sup>2</sup>, Yong Li<sup>1,†</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Xiaomi EV

<sup>∗</sup>Project leader. <sup>†</sup>Corresponding author.

## ABSTRACT

World models are emerging as a fundamental component of policy learning for real-world agents, enabling agents to predict future world states and plan actions within a learned representation of the dynamic environment. Existing approaches, however, often model appearance, metric geometry represented by projected Li-DAR depth, and ego motion either partially or independently, overlooking the fact that these modalities are different projections of a shared underlying world state. This separation can lead to information loss and cross-modal inconsistency, ultimately limiting downstream performance. Motivated by this unified view of the world, we propose DriveUniTok, a unified multimodal tokenizer for driving world-action models (WAMs) that encodes these modalities in a single decodable continuous latent. We freeze DriveUniTok and adapt a pretrained diffusion model, DriveUniWAM, to predict future unified latents; the tokenizer then decodes them into future appearance, projected LiDAR depth, and ego trajectories. Experiments on NAVSIM demonstrate that our unified design achieves better future prediction and planning performance at lower downstream computational cost than separate-stream alternatives. With reinforcement learning post-training, DriveUni-WAM achieves a PDMS of 91.54, reaching state-of-the-art performance among the evaluated WAM-based planners.

## 1 INTRODUCTION

Driving world models initially served as scene simulators and data engines, forecasting future observations for data augmentation and policy training. More recently, generative models of future observations have been integrated into policy backbones that jointly model scene evolution and ego trajectories (Zhang et al., 2025b; Zhao et al., 2025; Xia et al., 2026; Shi et al., 2026). In this setting, the latent representation is no longer merely an interface for visual reconstruction; it is the space in which the model captures the information required for both future prediction and planning.

Existing world-action models, however, expose asymmetric representation interfaces. Most inherit a pretrained RGB video autoencoder whose latent primarily captures appearance, while geometry is introduced through auxiliary supervision and ego pose remains outside the latent, where it is handled by a separate conditioning stream or action head (Team Wan et al., 2025; Kong et al., 2024; Agarwal et al., 2025). Driving tokenizers fuse camera and LiDAR observations into scene-centric bird’s-eye-view latents, but still leave ego pose outside the scene representation (Zhang et al., 2024; Chen & Zhao, 2025; Zhuo et al., 2026; Liao et al., 2025b). Cross-modal and latent-action tokenizers incorporate actions alongside observations, yet commonly retain modality-specific streams or align them through a designated anchor or intermediary representation (Lu et al., 2026; Bachmann et al., 2024; Chen et al., 2025a; Fan et al., 2025; Chen et al., 2026). Figure 1 compares these representations according to whether appearance, metric geometry, and ego pose reside in the same decodable latent.

This asymmetry forces the downstream predictor to learn cross-modal correspondence together with temporal evolution, which can make future-scene and ego-trajectory predictions inconsistent. Therefore, a central question arises: Should visual appearance, metric geometry, and ego-motion be represented within a unified tokenization framework? DriveUniTok addresses this question by encoding RGB appearance, projected LiDAR geometry, and clip-relative ego pose in one jointly decodable latent.

![](images/a41ab8110b00a398d95fd4d78282894d6a29a8c38e517244bcbef87ce672ddf6.jpg)  
Figure 1: Comparison of latent representations for world-action modeling. (a) BEV tokenizers fuse RGB and LiDAR into a visual latent, while (b) Occupancy tokenizers represent the scene with an occupancy latent; both keep ego motion separate. (c) Modality-specific tokenizers encode RGB, LiDAR, and ego motion into separate streams. (d) DriveUniTok jointly encodes RGB, LiDAR-derived geometry, and ego motion into one unified latent.

To construct such a representation, we propose DriveUniTok, a query-based multimodal tokenizer. Shared learnable queries aggregate appearance features from RGB, metric geometry features from projected LiDAR depth, and clip-relative ego-pose features into a common sequence of per-frame multimodal tokens. A causal temporal compressor then maps this sequence into a compact continuous unified latent. All three modalities are reconstructed from the shared bottleneck through modalityspecific readouts, without using a privileged anchor modality. In addition, a training-only semantic decoder is supervised by representations from a vision foundation model, encouraging the unified latent to preserve high-level scene semantics alongside metric geometry and pose.

To evaluate whether this representation supports joint forecasting and planning, we freeze Drive-UniTok and adapt a pretrained diffusion model into DriveUniWAM, using the same downstream backbone and training protocol for unified and separate representations. The Wan-initialized flowmatching transformer predicts future unified latents under block-causal attention, conditioned on DriveUniTok-encoded historical observations and clip-relative ego poses together with a navigation signal. Future RGB, depth, and ego trajectories are decoded from the same predicted latent evolution rather than generated through independent observation and action pathways. Because supervised flow matching fits recorded futures but does not directly optimize the NAVSIM planning metric, we additionally post-train DriveUniWAM with GRPO against the PDM reward while regularizing it toward the supervised reference model.

## Our key contributions are as follows:

• We propose DriveUniTok, a query-based tokenizer that encodes RGB appearance, projected LiDAR geometry, and clip-relative ego pose in a unified continuous latent.

• We develop DriveUniWAM, a diffusion-based WAM that forecasts this latent for joint future-scene prediction and ego-trajectory planning.

• We show on NAVSIM that the unified design achieves stronger prediction and planning at lower downstream computational cost than separate-stream controls, with DriveUniWAM reaching state-of-the-art performance at 91.54 PDMS after reinforcement learning posttraining.

## 2 RELATED WORK

## 2.1 GENERAL AND DRIVING TOKENIZERS

For visual tokenizers, discrete codebooks are commonly paired with autoregressive or masked token prediction (van den Oord et al., 2017; Yu et al., 2024), whereas continuous latents are often used with diffusion and flow-matching models. Large video VAEs provide high-fidelity latent spaces for RGB generation (Team Wan et al., 2025; Kong et al., 2024; Yang et al., 2025; Agarwal et al., 2025), and quantization and model design materially affect representation quality (Luo et al., 2024; Yu et al., 2024; Wang et al., 2024a; Tang et al., 2024; Wang et al., 2025; Guo et al., 2026). General-purpose multimodal tokenizers place vision, language, audio, and other signals in a shared token space through semantic or contrastive objectives (Lu et al., 2026; Bachmann et al., 2024; Lu et al., 2023; 2024). These tokenizers, however, are optimized for general visual generation or cross-modal alignment, and do not explicitly capture the metric geometry or ego motion required for driving scenes. Driving specific tokenizers instead focus on scene representations. BEV-VAE and DriveTok construct tokens from multiview observations (Chen & Zhao, 2025; Zhuo et al., 2026), while OccWorld and I<sup>2</sup>-World tokenize structured 3D occupancy for world modeling and 4D forecasting (Zheng et al., 2024; Liao et al., 2025b). Moving beyond camera- or occupancy-centric scene tokens, BEVWorld fuses surround view images and LiDAR into a BEV latent that can be rendered back to both modalities, but the latent still describes the scene without encoding ego pose (Zhang et al., 2024). These limitations motivate a unified tokenizer that keeps appearance, metric geometry, and ego pose jointly decodable.

## 2.2 DRIVING PLANNERS AND WORLD-ACTION MODELS

End-to-end driving systems jointly optimize perception and planning, from feature fusion and waypoint regression (Chitta et al., 2023) and planning-centric perception and prediction (Hu et al., 2023) to anchored trajectory sets refined by truncated diffusion (Liao et al., 2025a) or autoregressive mixture-of-experts decoding (Feng et al., 2025). World-model-assisted planners expose future-scene information through future rollouts, predicted BEV features, or world-objective constraints on the planning latent (Li et al., 2025; Zhang et al., 2025a; Yang et al., 2026). Vision-language-action planners add semantic priors and explicit reasoning, using either chain-of-thought reasoning with discrete action tokens or language-derived representations with diffusion planning and reward refinement (Zhou et al., 2025; Li et al., 2026d;a), and recent variants further incorporate generated futures through interleaved generation, latent conditioning, iterative replanning, or auxiliary supervision (Liu et al., 2026; Li et al., 2026b; Zuo et al., 2026a; Sheng et al., 2026; Li et al., 2026c; Zhou et al., 2026). These planners use future information in different ways, but generally treat future-scene prediction and trajectory generation as separate modeling components.

World-action models instead couple the two within a generative model. Existing world-action models differ primarily in how predicted future observations interact with trajectory generation. Some use parallel diffusion heads over a shared spatiotemporal latent (Zhang et al., 2025b) or attention over discrete future-frame tokens in an autoregressive sequence (Zhao et al., 2025), while others condition action transformers on cached video-denoising features (Xia et al., 2026), decode action chunk from a sampled future-video latent (Shi et al., 2026), or autoregressively interleave scene and action tokens (Ye et al., 2025; Chen et al., 2025b). While these designs couple future prediction and action generation, they do not generally provide a single decodable latent that jointly represents appearance, metric geometry, and ego motion.

In contrast, DriveUniTok unifies RGB appearance, projected LiDAR geometry, and clip-relative ego pose at the tokenization stage in one jointly decodable continuous latent. DriveUniWAM forecasts this latent for joint future-scene prediction and planning. Our contribution centers on unifying appearance, metric geometry, and ego motion at the representation level for world-action modeling.

## 3 METHODOLOGY

## 3.1 PROBLEM SETUP

A driving observation sequence $\big \{ o _ { 1 } , \dotsc , o _ { T } \big \}$ contains $T = 5$ consecutive frames sampled at 2 Hz. Each observation $o _ { i } = ( I _ { i } , D _ { i } , p _ { i } )$ consists of a front-camera image $I _ { i } \in \mathbb { R } ^ { 3 \times H \times W }$ , a sparse depth map $D _ { i } \in \mathbb { R } ^ { 1 \times H \times W }$ obtained by projecting the LiDAR sweep onto the image plane of $I _ { i } ,$ and an ego pose $p _ { i } \in \mathbb { R } ^ { 3 }$ expressed relative to the first frame of the sequence, with $\bar { H } \overset { \cdot } { \times } W = 2 8 8 \times 5 1 2$ Given K observed sequences, we predict the next $F$ sequences. Learning this mapping in observation space is expensive, and it leaves appearance, geometry, and motion in three different formats that a predictor would have to relate on its own. DriveUniTok therefore maps each sequence to a single unified latent $\boldsymbol { z } \in \mathbb { R } ^ { S \times d }$ of $S = 5 8 4$ tokens with width $d = 1 6 .$ , from which all three modalities are decodable. DriveUniWAM then models the evolution of z across sequences, and the DriveUniTok decoder maps predicted latents back to I, D, and $p .$ In this division of labor, DriveUniTok determines which information the latent preserves, and DriveUniWAM determines how it evolves. We train DriveUniTok first, freeze it, and post-train DriveUniWAM against the NAVSIM PDM score.

![](images/54f1115239cf0b20cf149c0ef61ed26ac3accabf52112f0aeb4c33d11a160b8d.jpg)  
Figure 2: DriveUniTok architecture. (a) Encoder. RGB, projected depth, and ego pose are fused into S shared queries, whose $T$ per-frame states are compressed into one latent. (b) Decoder. The latent is expanded back to $T$ frame positions, and each target is read out through its own set of mask tokens, including a training-only semantic branch. Snowflakes mark frozen components, and all remaining modules are trained.

## 3.2 DRIVEUNITOK: QUERY ALIGNED MULTIMODAL TOKENIZER

DriveUniTok constructs the unified latent used by DriveUniWAM through three components: a querybased multimodal encoder that fuses the three modalities within each frame, a temporal compressor that reduces the $T$ per-frame states of a sequence to one, and a shared decoder in which one set of mask tokens per target reads all three modalities back out of the compressed latent.

Multimodal encoder. Each input stream is first converted into features. RGB frames pass through a frozen DINOv3-L backbone (Siméoni et al., 2025), which yields a multilevel feature hierarchy. Rather than using a separate geometric backbone, we project the LiDAR sweep onto the image plane and process the resulting sparse depth map in the same feature space as RGB. The shared queries then extract appearance and geometric evidence through the same deformable-attention mechanism. The sparse depth map receives its own single-channel patch embedding, and the resulting tokens pass through the same frozen DINOv3 blocks with a learned modality embedding. Relative poses are mapped by a trainable ego state encoder into $N _ { E } = 8$ ego tokens. The frozen backbone therefore supplies one feature hierarchy to both RGB and depth, while the depth patch embedding and the ego state encoder only map their inputs into that shared token space. The driving-specific alignment itself is learned by the query layers described next.

The encoder uses one shared query set for all three streams. We initialize $N _ { I } = 5 7 6$ image queries on the $1 8 \times 3 2$ patch grid and use the ego tokens as ego queries, together forming the $S = \mathbf { \bar {  { N } } } _ { I } +  { \boldsymbol { N } } _ { E }$ tokens of the latent. The two groups differ in initialization and count rather than in modality, since they are updated jointly throughout the encoder and each ends up carrying evidence from all three inputs. Writing $\breve { Q } \in \breve { \mathbb { R } } ^ { S \times C }$ for the $S$ queries of width $C$ and $\dot { F } ^ { \mathrm { r g b } }$ $\dot { F } ^ { \mathrm { d } }$ for the last four DINOv3 layers of the RGB and of the depth hierarchy, each encoder layer performs

$$
Q \gets \mathrm { S A } ( Q ) , \qquad Q \gets \mathrm { S A } \big ( Q + \mathrm { D A } ( Q , F ^ { \mathrm { r g b } } ) \big ) , \qquad Q \gets \mathrm { S A } \big ( Q + \mathrm { D A } ( Q , F ^ { \mathrm { d } } ) \big ) ,\tag{1}
$$

with SA self-attention over the S queries and DA deformable cross-attention (Zhu et al., 2021) into a hierarchy. The same query sequence attends sequentially to RGB and depth features, while self-attention enables information exchange between the image-initialized and ego-initialized queries. This allows appearance, projected depth, and ego-pose information to interact before temporal compression. No modality is a privileged reference, and no fusion is deferred until after independent encoders have finished.

Temporal compression. Consecutive frames of a sequence are highly redundant, and keeping one latent per frame would make the downstream token sequence $T$ times longer at quadratic attention cost. Each of the S queries produces one state per frame, and we compress those $T$ states into one, independently for every query. Causal self-attention first processes the $\bar { T ^ { \dag } }$ states in order, then a learned pooling query gathers them into a single sequence-level state, which a Gaussian bottleneck (Kingma and Welling, 2014) maps to the d-dimensional code that forms z. Because depth and ego evidence are written into these shared queries rather than into token streams of their own, a sequence yields S tokens regardless of how many modalities it carries, roughly half of what separate per-modality representations would need. This gives DriveUniWAM a compact fixed-length latent in which appearance, geometry, and ego motion can be modeled together, while avoiding the duplicated token streams required by separate representations.

Multimodal decoder. A single latent step cannot be rendered directly, since the three targets are defined per frame. We therefore first project each token back to the decoder width, add learned frame queries for the $T$ temporal positions, and expand the sequence representation to T frame-level memories with a temporal transformer. A shared Transformer (He et al., 2022; Zheng et al., 2025) then reads this memory through one set of learned mask tokens per target, each carrying its own positional embedding. The RGB and depth mask tokens yield patch features rendered by separate convolutional decoders into $\hat { I }$ and $\hat { D } ;$ the ego mask tokens feed the ego head that regresses ${ \hat { p } } .$ . All three branches read the same memory, and only their mask tokens, positional embeddings, and output heads differ. This makes the multimodal alignment directly testable, since a modality absent from the shared latent could not be recovered by its branch.

Following VFMTok (Zheng et al., 2025), we add a training-only semantic target, because pixel reconstruction alone does not preserve the semantic content of the sampled foundation model features. A fourth set of mask tokens is read by the same shared Transformer into $\hat { \phi }$ and aligned by cosine similarity with the final layer patch tokens $\phi ( I )$ of a separate frozen DINOv3-L applied to the ground truth frames.

Training objective. DriveUniTok is trained end to end with one term per readout,

$$
{ \mathcal { L } } _ { \mathrm { t o k } } = \lambda _ { \mathrm { r g b } } { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { d e p t h } } { \mathcal { L } } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { e g o } } { \mathcal { L } } _ { \mathrm { e g o } } + \lambda _ { \mathrm { s e m } } { \mathcal { L } } _ { \mathrm { s e m } } + \lambda _ { \mathrm { K L } } { \mathcal { L } } _ { \mathrm { K L } } .\tag{2}
$$

$\mathcal { L } _ { \mathrm { r g b } }$ sums $L _ { 1 } , L _ { 2 } $ , VGG LPIPS (Zhang et al., 2018), and ConvNeXt and DINOv3 perceptual distances. ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ is a confidence-weighted regression loss (Wang et al., 2024b) evaluated only where LiDAR projects to a valid pixel, plus a multiscale gradient term. ${ \mathcal { L } } _ { \mathrm { e g o } } = \| { \hat { p } } - p \| _ { 1 }$ <sub>1</sub> and $\mathcal { L } _ { \mathrm { s e m } } = \mathbb { E } [ 1 - \cos ( \hat { \phi } , \phi ( I ) ) ]$ . Appendix A gives every component and its weight. Sharing one decoder memory across these terms is what prevents the queries from degenerating into unrelated modality-specific subspaces while retaining the metric information driving requires.

![](images/3c41cac6bbd05477986370f203645e5839381eaff0e07da84cccb1a556da921f.jpg)  
Figure 3: Training and policy optimization with DriveUniWAM. (a) Supervised training. Drive-UniWAM forecasts the future latents with flow matching. (b) Reinforcement learning. Per scene, a group of sampled futures is decoded into trajectories and scored by the NAVSIM PDM reward, whose relative advantages drive the Flow-GRPO update. DriveUniTok stays frozen in both stages.

## 3.3 DRIVEUNIWAM: JOINT WORLD-ACTION MODELING

DriveUniWAM follows the world-action model formulation introduced by DreamZero (Ye et al., 2026). Rather than forecasting observations and leaving action generation to a separate planner, it predicts future observations and actions within one generative process, so the imagined scene evolution and the executable action share a common dynamics representation. The central role of DriveUniTok is therefore to provide a latent in which the variables needed for world-action modeling are already jointly organized. Existing world-action models commonly inherit an RGB video latent and add action tokens beside it, leaving the model to discover the correspondence between appearance and motion during training. In DriveUniWAM that correspondence is already established by the frozen DriveUniTok latent, so the denoiser learns the evolution of an aligned driving state instead of simultaneously learning two prediction tasks and their correspondence.

DriveUniWAM uses the frozen DriveUniTok to encode the K observed and F future sequences. The history latents provide clean context, while the future latents are corrupted and reconstructed. We initialize the denoiser from the pretrained Wan2.2-TI2V-5B model (Team Wan et al., 2025) and adapt only its input and output projections to the DriveUniTok latent; a navigation embedding conditions it on the current vehicle state and route command. History queries attend only to history tokens, so the clean context cannot read noisy future targets, while every future query attends to the complete joint sequence, making all future tokens mutually visible during denoising. That visibility is what DriveUniTok makes reasonable, because each token already carries evidence from aligned appearance, geometry, and ego motion, so no artificial factorization between future observation and future action is needed.

Training uses flow matching (Lipman et al., 2023). A clean future latent z<sub>0</sub> is interpolated with Gaussian noise as $z _ { t } = ( 1 - t ) z _ { 0 } +$ tϵ at a shared time t, and the denoiser regresses the velocity field $v _ { * } = \epsilon - z _ { 0 }$ under

$$
\mathcal { L } _ { \mathrm { I L } } = \mathbb { E } \left[ w ( t ) \| v _ { \theta } - v _ { * } \| _ { 2 } ^ { 2 } \right] .\tag{3}
$$

Because appearance, geometry, and ego motion already share the tokens of z, a single shared denoiser predicts the velocity field over the complete unified latent, so the model learns future scenes and ego motion as one coherent continuation, as illustrated in Figure 3(a). The unified-latent loss, latent normalization, and supervised training schedule are described in Appendix C.1.

Reinforcement learning post-training. Flow matching fits recorded futures, whereas NAVSIM scores the consequences of a planned trajectory. We therefore post-train DriveUniWAM with Flow-GRPO (Liu et al., 2025) against the NAVSIM PDM reward (Dauner et al., 2024), as shown in

Table 1: Native-configuration comparison of video tokenizers on NAVSIM navtest. Comp. reports native temporal×height×width compression for grid tokenizers and tokens per clip for token-sequence methods; for DriveUniTok, it denotes the complete 584-token multimodal clip latent. The baselines use their official RGB-only preprocessing, whereas DriveUniTok additionally encodes projected depth and ego pose with a frozen DINOv3-L feature backbone. Results characterize fidelity–compression trade-offs under each tokenizer’s native configuration.
<table><tr><td rowspan="2">Method</td><td colspan="2">Tokenizer configuration</td><td colspan="6">NAVSIM navtest</td></tr><tr><td>Comp.</td><td>Dim.</td><td>LPIPSVGG ↓</td><td>LPIPSDINO ↓</td><td>PSNR↑</td><td>SSIM↑</td><td> $\mathrm { r F V D } _ { \mathrm { I 3 D } } ~ .$  →</td><td> $\mathrm { r F V D } _ { \mathrm { V J } } \downarrow$ </td></tr><tr><td colspan="9">Large-Scale Pretrained Video Tokenizers</td></tr><tr><td>Wan2.1 VAE (Team Wan et al., 2025)</td><td>4×8×8</td><td>16</td><td>0.1151</td><td>0.0649</td><td>31.50</td><td>0.8819</td><td>2.06</td><td>1.0984</td></tr><tr><td>Wan2.2 VAE (Team Wan et al., 2025)</td><td>4×16×16</td><td>48</td><td>0.1081</td><td>0.0608</td><td>31.35</td><td>0.8775</td><td>2.98</td><td>0.7958</td></tr><tr><td>HunyuanVideo VAE (Kong et al., 2024)</td><td>4×8×8</td><td>16</td><td>0.1230</td><td>0.0573</td><td>32.36</td><td>0.8924</td><td>4.03</td><td>0.6278</td></tr><tr><td>CogVideoX VAE (Yang et al., 2025)</td><td>4×8×8</td><td>16</td><td>0.1401</td><td>0.0943</td><td>32.26</td><td>0.8909</td><td>5.09</td><td>0.9298</td></tr><tr><td>Cosmos-0.1 (CV4x8x8) (Agarwal et al., 2025)</td><td>4×8×8</td><td>16</td><td>0.2362</td><td>0.3449</td><td>30.00</td><td>0.8481</td><td>26.76</td><td>4.1983</td></tr><tr><td>Cosmos-0.1 (CV8x16x16) (Agarwal et al., 2025)</td><td>8×16×16</td><td>16</td><td>0.3665</td><td>0.7327</td><td>26.03</td><td>0.7357</td><td>341.22</td><td>18.3995</td></tr><tr><td>AToken (Lu et al., 2026)</td><td>4×16×16</td><td>48</td><td>0.1644</td><td>0.1105</td><td>29.84</td><td>0.8493</td><td>7.75</td><td>1.2303</td></tr><tr><td colspan="9">Benchmark-Scale Video Tokenizers</td></tr><tr><td>Open-MAGVIT2 (Luo et al., 2024)</td><td>4×8×8</td><td>18</td><td>0.1659</td><td>0.1308</td><td>25.76</td><td>0.7802</td><td>34.57</td><td>14.4352</td></tr><tr><td>OmniTokenizer VAE (Wang et al., 2024a)</td><td>4×8×8</td><td>8</td><td>0.2049</td><td>0.3367</td><td>26.89</td><td>0.8455</td><td>40.70</td><td>6.8585</td></tr><tr><td>LARP-L-long (Wang et al., 2025)</td><td>1024 tokens</td><td>16</td><td>0.1459</td><td>0.1332</td><td>26.79</td><td>0.8042</td><td>24.74</td><td>5.7300</td></tr><tr><td>DriveUniTok (Ours)</td><td>584 tokens</td><td>16</td><td>0.1865</td><td>0.1063</td><td>25.62</td><td>0.7400</td><td>56.68</td><td>3.4546</td></tr></table>

Figure 3(b). For each scene the policy samples a group of joint futures, the frozen decoder reads a trajectory out of each sampled latent, and the group rewards give relative advantages for a clipped policy update regularized toward the supervised reference model. Only LoRA adapters (Hu et al., 2022) in the denoiser are trained; DriveUniTok, its decoder, and the reference policy stay frozen, so any gain reflects policy optimization around a fixed decodable latent rather than a change to the representation. Appendix C.2 states the full reward, transition replay, reference BC objective, and hyperparameters.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

We conduct experiments on NAVSIM, a non-reactive planning benchmark derived from the Open-Scene subset of nuPlan (Dauner et al., 2024). We train DriveUniTok on the NAVSIM training split and then freeze it before training DriveUniWAM on the same split.

We evaluate multimodal reconstruction, joint future prediction, and planning. DriveUniTok encodes front-camera RGB, projected LiDAR depth, and clip-relative ego poses into 584 tokens. Visual reconstruction uses VGG- and DINOv3-based LPIPS (Zhang et al., 2018; Siméoni et al., 2025), PSNR, SSIM (Wang et al., 2004), and I3D- and V-JEPA 2.1-based FVD (Unterthiner et al., 2019; Carreira and Zisserman, 2017; Mur-Labadia et al., 2026); depth uses MAE, AbsRel, and RMSE, and ego motion uses ADE and FDE. Planning uses no-at-fault collisions (NC), drivable-area compliance (DAC), ego progress (EP), time to collision (TTC), comfort, and PDMS.

## 4.2 TOKENIZER RECONSTRUCTION

We first test whether the compact unified latent preserves useful visual and temporal information by comparing DriveUniTok with large-scale pretrained video VAEs and benchmark-scale tokenizers under a common NAVSIM navtest protocol. Baselines use official preprocessing and native temporal windows, and each source frame is evaluated at most once. Visual baselines encode RGB only, whereas DriveUniTok encodes RGB, depth, and ego pose with frozen DINOv3-L features. We report native grid compression and DriveUniTok’s 584-token multimodal sequence separately to characterize fidelity–compression trade-offs under each tokenizer’s native configuration. Section 4.3 further compares unified and separate tokenization under matched representation and downstream computation budgets. Results are given in Table 1; qualitative RGB and depth reconstructions are shown in Figures 4–6 in Appendix B.

Table 2: Unified vs. separate tokenization on NAVSIM. Scalars counts latent values per clip; Len. is the DriveUniWAM sequence length at K=1, F=2. Costs are in TFLOPs; DiT cost is per denoising step. Lower reconstruction and prediction errors and higher PDMS are better.
<table><tr><td></td><td colspan="3">Representation budget</td><td colspan="3">Efficiency</td><td colspan="3">Reconstruction</td><td colspan="3">Future prediction</td><td>Planning</td></tr><tr><td>Representation</td><td>Tokens</td><td>Scalars</td><td>Len.</td><td>Tok. params (M)</td><td>Tok. cost</td><td>DiT/step cost</td><td>LPIPSDINO</td><td>AbsRel</td><td>ADE</td><td> $\mathrm { g F V D v \mathbf { J } }$ </td><td>AbsRel</td><td>ADE</td><td>PDMS</td></tr><tr><td>Separate, natural budget</td><td>1,160</td><td>18,560</td><td>3,480</td><td>957.8</td><td>4.6</td><td>34.2</td><td>0.1083</td><td>0.0723</td><td>0.0702</td><td>7.92</td><td>0.1194</td><td>0.3997</td><td>90.1</td></tr><tr><td>Separate, equal budget</td><td>584</td><td>9,344</td><td>1,752</td><td>956.6</td><td>4.3</td><td>16.3</td><td>0.1124</td><td>0.0766</td><td>0.0720</td><td>12.80</td><td>0.1850</td><td>0.4186</td><td>87.5</td></tr><tr><td>Unified (DriveUniTok)</td><td>584</td><td>9,344</td><td>1,752</td><td>847.3</td><td>4.6</td><td>16.3</td><td>0.1063</td><td>0.0461</td><td>0.0728</td><td>6.61</td><td>0.1059</td><td>0.3816</td><td>91.0</td></tr></table>

DriveUniTok attains the lowest $\mathrm { L P I P S } _ { \mathrm { D I N O } }$ (0.1063) and $\mathrm { r F V D } _ { \mathrm { V J } } \left( 3 . 4 5 4 6 \right)$ among benchmark-scale methods, improving over the respective second-best results by 18.7% and 39.7%. Large pretrained VAEs retain an advantage on pixel-oriented metrics, but do not encode geometry or ego motion. These results indicate that the compact multimodal latent preserves useful semantic and temporal information without allocating separate modality-specific token streams. We next evaluate whether this representation also benefits future prediction and planning when used by a WAM.

## 4.3 BENEFITS OF UNIFIED TOKENIZATION

We next evaluate the performance and resource efficiency of unified tokenization against separate modality-specific designs for WAM prediction and planning. We compare DriveUniTok against two separate-stream controls that use the same sensors, data, DriveUniWAM backbone, and training protocol. The natural-budget control assigns a full query budget to each modality and measures the resource efficiency of the unified design. The equal-budget control uses 288 RGB, 288 depth, and 8 ego queries, matching DriveUniTok’s 584 tokens, d = 16 latent width, 9,344 latent scalars, sequence length, and downstream computation. Both models use the same frozen DINOv3-L features, semantic supervision, mask-token reconstruction objective, optimization recipe, and downstream training protocol; their architectural difference combines latent organization, parameter sharing, and cross-modal connectivity. Further implementation details are provided in Appendix A. The complete comparison is reported in Table 2.

Compared with the natural-budget control, DriveUniTok uses roughly half as many tokens and latent scalars, halves the sequence length, and reduces per-step DiT computation by about 52%, while improving PDMS from 90.1 to 91.0. At equal representation and DiT budgets, it uses fewer tokenizer parameters, reduces $\mathrm { \ g F V D _ { V J } }$ by about 48%, future depth AbsRel by about 43%, and trajectory ADE by nearly 9%, and raises PDMS by 3.5 points to 91.0. These results support the unified design under both budget settings. Appendices D and F provide objective and input-modality ablations.

## 4.4 PLANNING PERFORMANCE

Table 3 compares DriveUniWAM with end-to-end, vision-language-action, and world-action planners on NAVSIM navtest under their reported sensor configurations. DriveUniTok remains frozen during both imitation learning and reward post-training.

DriveUniWAM reaches 90.5 PDMS with camera-only input, exceeding DriveWAM by 0.4 points. Adding projected LiDAR raises PDMS to 91.0, a further gain of 0.5 points and a 0.9-point improvement over DriveWAM. Reward post-training further increases PDMS to 91.54. With the tokenizer fixed, this gain reflects policy optimization within the same decodable latent space. DriveUniWAM also obtains 90.3 EPDMS on NAVSIM-v2; additional results and qualitative trajectories are provided in Appendix E and Figures 7–16 in Appendix G.

## 4.5 BACKBONE AND TOKENIZER GENERALIZATION

Table 4 tests DriveUniTok across video diffusion backbones. Part I replaces the native VAE while keeping the backbone fixed; Part II compares different backbones using DriveUniTok. Replacing the native VAE improves PDMS by 2.5 points for Wan2.1-1.3B and 3.1 points for Wan2.2-5B. Across five backbones, the unified latent achieves 88.9–91.0 PDMS, supporting its utility across model families and scales.

Table 3: NAVSIM navtest performance. C and L denote camera and LiDAR. The best and second-best learned results, including ties, are shown in bold and underlined.
<table><tr><td>Method</td><td>Reference</td><td>Sensors</td><td>NC↑</td><td>DAC↑</td><td>EP↑</td><td>TTC↑</td><td>Comfort↑</td><td>PDMS↑</td></tr><tr><td colspan="7"></td><td></td><td></td></tr><tr><td>Human Agent</td><td></td><td>Reference</td><td>100.0</td><td>100.0</td><td>87.5</td><td>100.0</td><td>99.9</td><td>94.8</td></tr><tr><td colspan="9">Vision-Based End-to-End Methods</td></tr><tr><td>UniAD (Hu et al., 2023)</td><td>CVPR&#x27;23</td><td>6×C</td><td>97.8</td><td>91.9</td><td>78.8</td><td>92.9</td><td>100.0</td><td>83.4</td></tr><tr><td>TransFuser (Chitta et al., 2023)</td><td>TPAMI&#x27;23</td><td>3×C+L</td><td>97.7</td><td>92.8 95.1</td><td>79.2 81.4</td><td>92.8</td><td>100.0</td><td>84.0</td></tr><tr><td>ARTEMIS (Feng et al., 2025)</td><td>RA-L&#x27;25</td><td>3×C+L</td><td>98.3</td><td></td><td></td><td>94.3</td><td>100.0</td><td>87.0</td></tr><tr><td>WorldRFT (Yang et al., 2026)</td><td>AAAI&#x27;26</td><td>3×C</td><td>97.8</td><td>96.8</td><td>81.7</td><td>94.0</td><td>100.0</td><td>87.8</td></tr><tr><td>DiffusionDrive (Liao et al., 2025a)</td><td>CVPR&#x27;25</td><td>3×C+L</td><td>98.2</td><td>96.2</td><td>82.2</td><td>94.7</td><td>100.0</td><td>88.1</td></tr><tr><td>WoTE (Li et al., 2025)</td><td>ICCV’25</td><td>3×C+L</td><td>98.5</td><td>96.8</td><td>81.9</td><td>94.9</td><td>99.9</td><td>88.3</td></tr><tr><td>SeerDrive (Zhang et al., 2025a)</td><td>NeurIPS’25</td><td>3×C+L</td><td>98.4</td><td>97.0</td><td>83.2</td><td>94.9</td><td>99.9</td><td>88.9</td></tr><tr><td colspan="9">Vision-Language-Action Methods</td></tr><tr><td>UniWorldVLA (Liu et al., 2026)</td><td>arXiv&#x27;26</td><td>1×C</td><td>98.7</td><td>96.7</td><td>83.2</td><td>96.1</td><td>100.0</td><td>89.4</td></tr><tr><td>DriveDreamer-Policy (Zhou et al., 2026)</td><td>arXiv&#x27;26</td><td>3×C</td><td>98.4</td><td>97.1</td><td>83.5</td><td>95.1</td><td>100.0</td><td>89.2</td></tr><tr><td>Vega (Zuo et al., 2026a)</td><td>arXiv&#x27;26</td><td>1xC</td><td>98.9</td><td>95.3</td><td>81.6</td><td>96.1</td><td>100.0</td><td>87.9</td></tr><tr><td>ImagiDrive (Li et al., 2026b)</td><td>ICRA&#x27;26</td><td>1×C</td><td>98.6</td><td>96.2</td><td>80.5</td><td>94.5</td><td>100.0</td><td>87.4</td></tr><tr><td>AutoVLA (Zhou et al., 2025)</td><td>NeurIPS’25</td><td>3×C</td><td>98.4</td><td>95.6</td><td>81.9</td><td>98.0</td><td>99.9</td><td>89.1</td></tr><tr><td>ReCogDrive (Li et al., 2026d)</td><td>ICLR’26</td><td>1×C</td><td>97.9</td><td>97.3</td><td>87.3</td><td>94.9</td><td>100.0</td><td>90.8</td></tr><tr><td>ExploreVLA (Sheng et al., 2026)</td><td>ECCV’26</td><td>1×C</td><td>98.8</td><td>98.4</td><td>83.5</td><td>96.5</td><td>99.9</td><td>90.4</td></tr><tr><td>DriveVLA-W0 (Li et al., 2026c)</td><td>ICLR&#x27;26</td><td>1xC</td><td>98.7</td><td>99.1</td><td>83.3</td><td>95.3</td><td>99.3</td><td>90.2</td></tr><tr><td>SGDrive (Li et al., 2026a)</td><td>CVPR’26</td><td>1xC</td><td>98.6</td><td>97.8</td><td>85.8</td><td>96.2</td><td>100.0</td><td>91.1</td></tr><tr><td colspan="9">World-Action Models</td></tr><tr><td>Epona (Zhang et al., 2025b)</td><td>ICCV’25</td><td>1×C</td><td>97.9</td><td>95.1</td><td>80.4</td><td>93.8</td><td>99.9</td><td>86.2</td></tr><tr><td>PWM (Zhao et al., 2025)</td><td>NeurIPS’25</td><td>1×C</td><td>98.6</td><td>95.9</td><td>81.8</td><td>95.4</td><td>100.0</td><td>88.1</td></tr><tr><td>DriveLaW (Xia et al., 2026)</td><td>CVPR&#x27;26</td><td>1xC</td><td>99.0</td><td>97.1</td><td>81.3</td><td>96.7</td><td>100.0</td><td>89.1</td></tr><tr><td>DriveWAM (Shi et al., 2026)</td><td>arXiv&#x27;26</td><td>1xC</td><td>98.3</td><td>98.1</td><td>84.3</td><td>95.2</td><td>100.0</td><td>90.1</td></tr><tr><td>DriveUniWAM (Camera-only)</td><td></td><td>1×C</td><td>98.9</td><td>97.7</td><td>84.0</td><td>96.8</td><td>99.9</td><td>90.5</td></tr><tr><td>DriveUniWAM (Ours)</td><td></td><td>1×C+L</td><td>99.0</td><td>97.7</td><td>85.6</td><td>96.3</td><td>100.0</td><td>91.0</td></tr><tr><td>DriveUniWAM + RL (Ours)</td><td></td><td>1×C+L</td><td>98.7</td><td>98.0</td><td>87.4</td><td>95.6</td><td>99.8</td><td>91.54</td></tr></table>

Table 4: Ablation of video backbones and latent representations on the NAVSIM navtest benchmark. Part I compares the native VAE and DriveUniTok with fixed backbones, while Part II evaluates DriveUniWAM across video DiT backbones using the same unified latent. Across both parts, best and second-best results are shown in bold and underlined; improvements over the native VAE appear in parentheses.
<table><tr><td>Video backbone</td><td>Tokenizer</td><td>NC↑</td><td>DAC↑</td><td>EP↑</td><td>TTC↑</td><td>PDMS↑</td></tr><tr><td colspan="7">Part I: Visual Tokenizer Representation Ablation</td></tr><tr><td>Wan2.1-T2V-1.3B (Team Wan et al., 2025)</td><td>Native VAE DriveUniTok</td><td>98.1 98.4</td><td>94.7</td><td>80.9</td><td>94.5</td><td>86.5</td></tr><tr><td></td><td></td><td></td><td>97.0</td><td>82.9</td><td>95.4</td><td>89.0 (+2.5)</td></tr><tr><td>Wan2.2-TI2V-5B (Team Wan et al., 2025)</td><td>Native VAE</td><td>98.4</td><td>95.5</td><td>82.6</td><td>94.8</td><td>87.9</td></tr><tr><td></td><td>DriveUniTok</td><td>99.0</td><td>97.7</td><td>85.6</td><td>96.3</td><td>91.0 (+3.1)</td></tr><tr><td colspan="7">Part II: Scalability across Video DiT Backbones (with DriveUniTok)</td></tr><tr><td>Wan2.1-T2V-1.3B (Team Wan et al., 2025)</td><td>DriveUniTok</td><td>98.4</td><td>97.0</td><td>82.9</td><td>95.4</td><td>89.0</td></tr><tr><td>CogVideoX-2B (Yang et al., 2025)</td><td>DriveUniTok</td><td>98.9</td><td>96.5</td><td>82.5</td><td>96.2</td><td>88.9</td></tr><tr><td>Cosmos-Predict2.5-2B (Agarwal et al., 2025)</td><td>DriveUniTok</td><td>99.0</td><td>96.2</td><td>83.3</td><td>96.1</td><td>89.1</td></tr><tr><td>Wan2.2-TI2V-5B (Team Wan et al., 2025)</td><td>DriveUniTok</td><td>99.0</td><td>97.7</td><td>85.6</td><td>96.3</td><td>91.0</td></tr><tr><td>Hunyuan Video-1.5-8.3B (Tencent Hunyuan Foundation Model Team, 2025)</td><td>DriveUniTok</td><td>99.1</td><td>97.7</td><td>84.0</td><td>96.8</td><td>90.5</td></tr></table>

## 5 CONCLUSION

We presented DriveUniTok and DriveUniWAM for joint future-scene and ego-trajectory prediction through a unified multimodal latent. On NAVSIM, the unified design improves prediction and planning with fewer tokenizer parameters and lower downstream computational cost than the naturalbudget separate-stream control. DriveUniWAM achieves 91.0 PDMS with supervised training and 91.54 after reinforcement learning post-training.

## AI USE STATEMENT

We used AI tools to assist with code development and debugging, manuscript language editing, and literature search and review. The tools were used as research assistance rather than as a substitute for author verification of technical claims. The authors checked the implementation, experimental results, citations, and final manuscript, and take full responsibility for the content of this paper.

## REPRODUCIBILITY STATEMENT

To support reproducibility, the main paper and Appendices A, C.1, and C.2 document the model, objectives, training procedures, and evaluation details needed to reproduce our DriveUniTok tokenizer, DriveUniWAM training, and NAVSIM evaluation.

## REFERENCES

Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al. Cosmos World Foundation Model Platform for Physical AI. arXiv preprint arXiv:2501.03575, 2025.

Bachmann, R., Kar, O. F., Mizrahi, D., Garjani, A., Gao, M., Griffiths, D., Hu, J., Dehghan, A., and Zamir, A. 4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities. Advances in Neural Information Processing Systems, 2024.

Carreira, J., and Zisserman, A. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 2017.

Chen, B., Chen, Y., Qiu, L., Bai, J., Ge, Y., and Ge, Y. UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling. arXiv preprint arXiv:2604.19734, 2026.

Chen, X., Wei, H., Zhang, P., Zhang, C., Wang, K., Guo, Y., Yang, R., Wang, Y., Xiao, X., Zhao, L., et al. villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models. arXiv preprint arXiv:2507.23682, 2025a.

Chen, Y., Wang, Y., and Zhang, Z. DrivingGPT: Unifying Driving World Modeling and Planning with Multi-modal Autoregressive Transformers. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025b.

Chen, Z. and Zhao, H. BEV-VAE: Multi-view Image Generation with Spatial Consistency for Autonomous Driving. arXiv preprint arXiv:2507.00707, 2025.

Chitta, K., Prakash, A., Jaeger, B., Yu, Z., Renz, K., and Geiger, A. TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11):12878–12895, 2023.

Dang, C., Ang, S., Li, Y., Tian, H., Wang, J., Li, G., Ye, H., Ma, J., Chen, L., and Wang, Y. DriveFine: Refining-Augmented Masked Diffusion VLA for Precise and Robust Driving. arXiv preprint arXiv:2602.14577, 2026.

Dauner, D., et al. NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 2024.

Fan, S., Wu, K., Che, Z., Wang, X., Wu, D., Liao, F., Liu, N., Zhang, Y., Zhao, Z., Xu, Z., et al. XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations. arXiv preprint arXiv:2511.02776, 2025.

Feng, R., Xi, N., Chu, D., Wang, R., Deng, Z., Wang, A., Lu, L., Wang, J., and Huang, Y. ARTEMIS: Autoregressive End-to-End Trajectory Planning with Mixture of Experts for Autonomous Driving. arXiv preprint arXiv:2504.19580, 2025.

Guo, M., Wu, S., and Fei, H. V-RAE: Rethinking Video Latent Spaces for Generation. arXiv preprint arXiv:2608.13556, 2026.

He, K., et al. Masked Autoencoders Are Scalable Vision Learners. Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Hu, E. J., et al. LoRA: Low-Rank Adaptation of Large Language Models. International Conference on Learning Representations, 2022.

Hu, Y., Yang, J., Chen, L., Li, K., Sima, C., Zhu, X., Chai, S., Du, S., Lin, T., Wang, W., Lu, L., Jia, X., Liu, Q., Dai, J., Qiao, Y., and Li, H. Planning-oriented Autonomous Driving. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Kingma, D. P., and Welling, M. Auto-Encoding Variational Bayes. International Conference on Learning Representations, 2014.

Kong, W., et al. HunyuanVideo: A Systematic Framework for Large Video Generative Models. arXiv preprint arXiv:2412.03603, 2024.

Li, J., Wu, J., Hu, D., Huang, X., Sun, B., Hao, Z., Lang, X., Zhu, X., and Zhang, L. SGDrive: Sceneto-Goal Hierarchical World Cognition for Autonomous Driving. Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026a.

Li, J., Zhang, B., Jin, X., Deng, J., Zhu, X., and Zhang, L. ImagiDrive: A Unified Imagination-and-Planning Framework for Autonomous Driving. IEEE International Conference on Robotics and Automation, 2026b.

Li, Y., Shang, S., Liu, W., Zhan, B., Wang, H., Wang, Y., Chen, Y., Wang, X., An, Y., Tang, C., Hou, L., Fan, L., and Zhang, Z. DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving. International Conference on Learning Representations, 2026c.

Li, Y., Wang, Y., Liu, Y., He, J., Fan, L., and Zhang, Z. End-to-End Driving with Online Trajectory Evaluation via BEV World Model. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Li, Y., Xiong, K., Guo, X., Li, F., Yan, S., Xu, G., Zhou, L., Chen, L., Sun, H., Wang, B., Ma, K., Chen, G., Ye, H., Liu, W., and Wang, X. ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving. International Conference on Learning Representations, 2026d.

Liao, B., Chen, S., Yin, H., Jiang, B., Wang, C., Yan, S., Zhang, X., Li, X., Zhang, Y., Zhang, Q., and Wang, X. DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025a.

Liao, Z., Wei, P., Zhang, R., Chen, S., Wang, H., and Ren, Z. I<sup>2</sup>-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting. arXiv preprint arXiv:2507.09144, 2025b.

Lipman, Y., et al. Flow Matching for Generative Modeling. International Conference on Learning Representations, 2023.

Liu, J., et al. Flow-GRPO: Training Flow Matching Models via Online RL. Advances in Neural Information Processing Systems, 2025.

Liu, Q., Xu, H., Li, J., Sun, B., Hao, Z., She, D., Zhu, X., and Zhang, L. Uni-World VLA: Interleaved World Modeling and Planning for Autonomous Driving. arXiv preprint arXiv:2603.27287, 2026.

Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., and Kembhavi, A. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action. Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Lu, J., Clark, C., Zellers, R., Mottaghi, R., and Kembhavi, A. Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks. International Conference on Learning Representations, 2023.

Lu, J., et al. AToken: A Unified Tokenizer for Vision. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Luo, Z., Shi, F., Ge, Y., Yang, Y., Wang, L., and Shan, Y. Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-Regressive Visual Generation. arXiv preprint arXiv:2409.04410, 2024.

Mur-Labadia, L., Muckley, M., Bar, A., Assran, M., Sinha, K., Rabbat, M., LeCun, Y., Ballas, N., and Bardes, A. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. arXiv preprint arXiv:2603.14482, 2026.

Sheng, Z., et al. ExploreVLA: Dense World Modeling and Exploration for End-to-End Autonomous Driving. European Conference on Computer Vision, 2026.

Shi, C., Xu, J., Shi, S., Sheng, K., Zhang, B., and Jiang, L. DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving. arXiv preprint arXiv:2605.28544, 2026.

Siméoni, O., et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

Tang, A., He, T., Guo, J., Cheng, X., Song, L., and Bian, J. VidTok: A Versatile and Open-Source Video Tokenizer. arXiv preprint arXiv:2412.13061, 2024.

Team Wan et al. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314, 2025.

Tencent Hunyuan Foundation Model Team. HunyuanVideo 1.5 Technical Report. arXiv preprint arXiv:2511.18870, 2025.

Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., and Gelly, S. FVD: A New Metric for Video Generation. International Conference on Learning Representations Workshop, 2019. https://openreview.net/forum?id=rylgEULtdN.

van den Oord, A., Vinyals, O., and Kavukcuoglu, K. Neural discrete representation learning. Advances in Neural Information Processing Systems, 2017.

Wang, H., Suri, S., Ren, Y., Chen, H., and Shrivastava, A. LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior. International Conference on Learning Representations, 2025.

Wang, J., Jiang, Y., Yuan, Z., Peng, B., Wu, Z., and Jiang, Y.-G. OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation. Advances in Neural Information Processing Systems, 2024a.

Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., and Revaud, J. DUSt3R: Geometric 3D Vision Made Easy. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 2004.

Xia, T., Li, Y., Zhou, L., Yao, J., Xiong, K., Sun, H., Wang, B., Ma, K., Chen, G., Ye, H., Liu, W., and Wang, X. DriveLaW: Unifying Planning and Video Generation in a Latent Driving World. Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Xu, J., Zhong, Z., Shu, Z., Jia, M., Li, M., Bian, J.-W., Zhang, Q., Zhang, K., Xie, J., Yang, J., and Yin, W. EponaV2: Driving World Model with Comprehensive Future Reasoning. arXiv preprint arXiv:2605.14696, 2026.

Yang, P., Lu, B., Xia, Z., Han, C., Gao, Y., Zhang, T., Zhan, K., Lang, X., Zheng, Y., and Zhang, Q. WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving. Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. International Conference on Learning Representations, 2025.

Yao, W., Li, Z., Lan, S., Wang, Z., Sun, X., Alvarez, J. M., and Wu, Z. DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning. Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

Ye, B., Zhang, B., and Zhao, H. DAP: A Discrete-token Autoregressive Planner for Autonomous Driving. arXiv preprint arXiv:2511.13306, 2025.

Ye, S., Ge, Y., Zheng, K., Gao, S., Yu, S., Kurian, G., Indupuru, S., Tan, Y. L., Zhu, C., Xiang, J., et al. World Action Models Are Zero-Shot Policies. arXiv preprint arXiv:2602.15922, 2026.

Yu, L., Lezama, J., Gundavarapu, N. B., Versari, L., Sohn, K., Minnen, D., Cheng, Y., Gupta, A., Gu, X., Hauptmann, A. G., Gong, B., Yang, M.-H., Essa, I., Ross, D. A., and Jiang, L. Language Model Beats Diffusion: Tokenizer is Key to Visual Generation. International Conference on Learning Representations, 2024.

Zhang, B., Song, N., Li, J., Zhu, X., Deng, J., and Zhang, L. Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution. Advances in Neural Information Processing Systems, 2025a.

Zhang, K., Tang, Z., Hu, X., Pan, X., Guo, X., Liu, Y., Huang, J., Yuan, L., Zhang, Q., Long, X.-X., Cao, X., and Yin, W. Epona: Autoregressive Diffusion World Model for Autonomous Driving. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025b.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

Zhang, Y., Gong, S., Xiong, K., Ye, X., Tan, X., Wang, F., Huang, J., Wu, H., and Wang, H. BEVWorld: A Multimodal World Model for Autonomous Driving via Unified BEV Latent Space. arXiv preprint arXiv:2407.05679v1, 2024.

Zhao, Z., Fu, T., Wang, Y., Wang, L., and Lu, H. From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction. Advances in Neural Information Processing Systems, 2025.

Zheng, A., Wen, X., Zhang, X., Ma, C., Wang, T., Yu, G., Zhang, X., and Qi, X. Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Image Generation. arXiv preprint arXiv:2507.08441, 2025.

Zheng, W., Chen, W., Huang, Y., Zhang, B., Duan, Y., and Lu, J. OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving. European Conference on Computer Vision (ECCV), 2024.

Zhou, Y., Wang, X., Shao, H., et al. DriveDreamer-Policy: A Geometry-Grounded World–Action Model for Unified Generation and Planning. arXiv preprint arXiv:2604.01765, 2026.

Zhou, Z., Cai, T., Zhao, S. Z., Zhang, Y., Huang, Z., Zhou, B., and Ma, J. AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning. Advances in Neural Information Processing Systems, 2025.

Zhu, X., et al. Deformable DETR: Deformable Transformers for End-to-End Object Detection. International Conference on Learning Representations, 2021.

Zhuo, D., Zheng, W., Zuo, S., Yan, S., Hou, L., Zhou, J., and Lu, J. DriveTok: 3D Driving Scene Tokenization for Unified Multi-View Reconstruction and Understanding. arXiv preprint arXiv:2603.19219, 2026.

Zuo, S., Li, Y., Zheng, W., et al. Vega: Learning to Drive with Natural Language Instructions. arXiv preprint arXiv:2603.25741, 2026a.

Zuo, S., Xie, Z., Zheng, W., Xu, S., Li, F., Li, H., Chen, L., Yang, Z.-X., and Lu, J. DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale. arXiv preprint arXiv:2604.00813, 2026b.

## APPENDIX

## A TOKENIZER IMPLEMENTATION DETAILS

Encoder and decoder configuration. Each input is resized to $2 8 8 \times 5 1 2$ and divided into patches of size 16, yielding an $1 8 \times \mathrm { { \bar { 3 2 } } }$ grid. The encoder is initialized with $N _ { I } = 5 7 6$ queries with spatial reference points and $N _ { E } = 8$ queries with learned reference points, all with embedding width 1024. These query groups differ in initialization rather than modality: after joint attention, they form one unified sequence of $S = 5 8 4$ tokens, with every token able to carry evidence from RGB, depth, and ego motion. The DINOv3-L feature backbone is initialized from a pretrained checkpoint and kept frozen throughout tokenizer training; the separate DINOv3 semantic teacher is also frozen. The encoder contains two query layers; each layer applies joint self-attention over the S queries followed by deformable cross-attention (Zhu et al., 2021) over the RGB and depth feature hierarchies. We use four feature levels, four sampling points per level, and 16 attention heads. Reference points for the first query group are fixed at patch-grid centers, whereas the second query group uses learned reference points because it has no spatial prior.

Temporal compression and expansion each use two causal self-attention blocks. The $T$ states of each unified query are temporally pooled into one sequence-level state and passed through a Variational Latent Bottleneck (VLB), which compresses the state to width $d = 1 6$ . Thus, DriveUniTok produces one unified $z \in \dot { \mathbb { R } ^ { S \times 1 6 } }$ rather than separate image and ego latents. The decoder maps this latent back to width 1024 and applies 24 shared ViT blocks with 16 heads and full self-attention. Four sets of mask tokens read the shared decoder features; the RGB and depth branches use convolutional decoders with channel multipliers [1, 1, 2, 2, 4] and two residual blocks per stage. We optimize all trainable tokenizer modules with AdamW at learning rate $1 0 ^ { - 4 }$ and weight decay 0.01, clip gradients at norm 10, and use warmup followed by a cosine schedule. The query-aggregator parameter group uses a 0.1× learning-rate multiplier. All DINOv3 parameters receive no gradient updates. This unified latent is the representation consumed by DriveUniWAM for joint future prediction and planning.

Equal-budget unified–separate comparison. We provide additional implementation details for the comparison in Table 2. The protocol matches the representation exposed to DriveUniWAM, allowing us to compare the downstream effectiveness of the unified and separate designs under matched representation and downstream computation budgets. DriveUniTok uses 584 unified tokens, initialized from 576 queries with spatial reference points and 8 queries with learned reference points. The equal-budget separate baseline uses 288 RGB queries, 288 projected-LiDAR depth queries, and 8 ego queries, also totaling 584 tokens. Both RGB and depth query sets use $1 2 \times 2 4$ grids. With latent width $d = 1 6 ,$ both models therefore produce $5 8 4 \times \mathsf { \bar { 1 6 } } = 9 { , } \dot { 3 } 4 4$ latent scalars and the same downstream sequence length of $3 \times 5 8 4 = 1 { , } 7 5 2$ tokens, giving the same nominal self-attention budget.

Both arms use the same frozen DINOv3-L feature extractor, frozen semantic teacher, RGB/depth reconstruction objectives, semantic auxiliary loss, depth preprocessing, and mask-token reconstruction path. The semantic supervision configuration and loss weights are kept unchanged. Neither arm uses a separate masking stage; mask-token reconstruction is part of the same tokenizer training procedure for both models.

The architectural difference is the location and connectivity of cross-modal interaction. DriveUniTok uses one shared query sequence, allowing RGB, depth, and ego queries to exchange information through self- attention before temporal compression and the Variational Latent Bottleneck. The separate baseline uses independent RGB, depth, and ego streams: RGB queries attend only to RGB features, depth queries attend only to depth features, and ego queries do not attend to visual features. Each stream has its own query set, encoder stack, temporal compressor, and Variational Latent Bottleneck. The separate design therefore removes cross-modal interaction in the tokenizer rather than merely assigning different output slices to a shared encoder.

The observation decoders use the same ViT-based decoder configuration, including the same depth, number of attention heads, and MLP ratio of 4.0. For the separate baseline, decoder context is restricted to the corresponding token ranges: RGB tokens occupy [0, 288), depth tokens occupy [288, 576), and ego tokens occupy [576, 584). The unified model uses the full shared token sequence as decoder context. In the reported separate-stream configuration, ego reconstruction directly uses the aggregated ego tokens rather than the shared decoder ego readout.

We match the tokenizer training recipe across the two arms, including the data protocol, batch size, learning rate of $1 0 ^ { - 4 }$ , precision, gradient accumulation, learning-rate schedule, and 33K tokenizer updates. We likewise match the downstream DriveUniWAM recipe, including the same data protocol, model configuration, 130K updates, learning rate, warm-up schedule, and flow-shift setting.

The equal-budget condition matches token count, latent-scalar count, downstream sequence length, nominal attention cost, sensors, training data, and optimization recipe. It does not imply exact equality of trainable parameter counts: the separate baseline contains independent copies of the query sets, encoder stacks, temporal compressors, and Variational Latent Bottlenecks, whereas DriveUniTok uses a shared aggregation pathway. The frozen DINOv3-L backbone and shared depth patch embedding are common to both models and are not treated as additional trainable capacity. The equal-budget condition therefore matches the representation and downstream DiT budgets while exposing the parameter-efficiency difference between the two tokenizer designs. Together with the natural-budget comparison, this evaluates whether the unified design remains advantageous under both lower-resource and matched-budget settings.

Loss components and weights. We use $\lambda _ { \mathrm { r g b } } = \lambda _ { \mathrm { d e p t h } } = \lambda _ { \mathrm { e g o } } = \lambda _ { \mathrm { s e m } } = 1$ and $\lambda _ { \mathrm { K L } } = 0 . 1$ Table 5 expands $\mathcal { L } _ { \mathrm { t o k } }$ into all of its component terms and lists the weight assigned to each term within its group.

Table 5: Components of the DriveUniTok training objective. Ω denotes the set of pixels where LiDAR projects to a valid depth.
<table><tr><td>Group</td><td>Term</td><td>Form</td><td>Weight</td></tr><tr><td rowspan="5"> $\mathcal { L } _ { \mathrm { r g b } }$ </td><td>pixel  $L _ { 1 }$ </td><td> $\| { \hat { I } } - I \| _ { 1 }$ </td><td>1.0</td></tr><tr><td>pixel  $L _ { 2 }$ </td><td> $\| \hat { I } - I \| _ { 2 } ^ { 2 }$ </td><td>1.0</td></tr><tr><td>LPIPS</td><td>VGG feature distance (Zhang et al., 2018)</td><td>1.0</td></tr><tr><td>perceptual</td><td>ConvNeXt feature distance</td><td>0.1</td></tr><tr><td>perceptual</td><td>DINOv3 feature distance</td><td>1.0</td></tr><tr><td rowspan="3"> $\mathcal { L } _ { \mathrm { d e p t h } }$ </td><td>confidence</td><td> $\mathbb { E } _ { \Omega } [ r \kappa - \alpha \log \kappa ] , \alpha { = } 0 . 2$ </td><td>1.0</td></tr><tr><td>regression</td><td> $\mathbb { E } _ { \Omega } [ r ] , r = | \hat { D } - D |$ </td><td>10.0</td></tr><tr><td>gradient</td><td>multiscale gradient matching on Ω</td><td>0.1</td></tr><tr><td> $\mathcal { L } _ { \mathrm { e g o } }$ </td><td>pose  $L _ { 1 }$ </td><td> $\| \hat { p } - p \| _ { 1 } \mathrm { o n } \left( x , y , \mathrm { h e a d i n g } \right)$ </td><td>1.0</td></tr><tr><td> $\mathcal { L } _ { \mathrm { s e m } }$ </td><td>cosine</td><td> $\mathbb { E } [ 1 - \cos ( \hat { \phi } , \phi ( I ) ) ]$ </td><td>1.0</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { K L } }$ </td><td>Gaussian KL</td><td>unified VLB</td><td>0.1</td></tr></table>

For valid projected pixels Ω, the depth confidence term uses $r \kappa - \alpha$ log κ with $r = \| \hat { D } - D \| _ { 2 } ,$ $\alpha = 0 . 2 ,$ and κ the predicted confidence. The regression term uses the same valid pixels, and both confidence and regression values are filtered at the 0.98 quantile. Depth is clipped to [1, 150] m and normalized by an average scale of 60. Batches with fewer than 100 valid points skip the depth terms. The semantic target is the final-layer patch tokens of a frozen DINOv3-L applied to the ground-truth frames, which is a separate instance from the frozen encoder backbone.

## B QUALITATIVE TOKENIZER RECONSTRUCTIONS

Each group compares the ground-truth RGB image, its reconstruction from the unified latent, the LiDAR-projected ground-truth depth, and the reconstructed depth at valid LiDAR pixels. Across nine scenes and five consecutive frames per scene, the RGB reconstructions preserve road geometry, vehicles, and salient scene details, while the depth reconstructions retain the spatial structure and distance patterns present in the sparse LiDAR measurements. The comparisons provide a qualitative complement to the reconstruction metrics in Table 1: the tokenizer preserves both appearance and metric 3D structure within the same compact representation.

![](images/f845eb2d544925ccd584663a3df40e96936b8535ab311560c180df5dd81b41df.jpg)  
Figure 4: Qualitative tokenizer reconstructions for reconstruction group 1. From left to right, the columns show RGB ground truth, reconstructed RGB, LiDAR-projected depth ground truth, and reconstructed depth at valid LiDAR pixels.

![](images/ac677243af30bfb3c523c2a720afad28d2851dc43994fdab686054d384b5aa33.jpg)  
Figure 5: Qualitative tokenizer reconstructions for reconstruction group 2. From left to right, the columns show RGB ground truth, reconstructed RGB, LiDAR-projected depth ground truth, and reconstructed depth at valid LiDAR pixels.

![](images/1d904a5186023393e72a255c4be681b43fad6b01d95ff85011cc167f30b599d7.jpg)  
Figure 6: Qualitative tokenizer reconstructions for reconstruction group 3. From left to right, the columns show RGB ground truth, reconstructed RGB, LiDAR-projected depth ground truth, and reconstructed depth at valid LiDAR pixels.

## C DRIVEUNIWAM TRAINING DETAILS

## C.1 SUPERVISED FLOW MATCHING

The supervised model uses one history clip and two future clips, corresponding to the $K { = } 1 , F { = } 2$ setting. Each clip contains five frames sampled at 2 Hz. With shared boundary frames, the history and future clips cover frame indices [0, 1, 2, 3, 4], [4, 5, 6, 7, 8], and [8, 9, 10, 11, 12], respectively. Each clip’s ego pose is expressed relative to its first frame. When clips are concatenated, the poses of each subsequent clip are transformed into the coordinate frame anchored at the endpoint of the preceding clip, so the shared boundary frames produce a continuous trajectory. The unified latent is normalized per channel using statistics estimated on the training split. We sample Gaussian noise ϵ and a shifted flow time σ with shift 3.0, then construct

$$
z _ { \sigma } = ( 1 - \sigma ) z _ { 0 } + \sigma \epsilon , \qquad v _ { * } = \epsilon - z _ { 0 } .\tag{4}
$$

The history clip remains clean, while the two future clips share the same σ and are noised independently. A single shared denoiser predicts the velocity field over the complete unified latent:

$$
\mathcal { L } _ { \mathrm { S F T } } = \mathbb { E } \left[ w ( \sigma ) \left. v _ { \theta } - v _ { * } \right. _ { 2 } ^ { 2 } \right] .\tag{5}
$$

The loss is averaged over the complete unified token sequence. The navigation condition has eight values: four current-frame kinematic values and four route-command values. Only the Wan DiT parameters are optimized; DriveUniTok and its decoder remain frozen. The reported training uses AdamW with learning rate $1 0 ^ { - 4 }$ , betas (0.9, 0.95), weight decay 0.01, 1,000 warmup updates, cosine decay, gradient clipping at 1.0, and 130,000 optimizer updates.

## C.2 REINFORCEMENT LEARNING POST-TRAINING

We post-train DriveUniWAM with Flow-GRPO (Liu et al., 2025) against the NAVSIM PDM reward (Dauner et al., 2024). For each scene the policy samples a group of $G = 8$ joint futures over 10 stochastic Flow-SDE denoising steps, and the frozen DriveUniTok decoder reads a trajectory out of each sampled latent for NAVSIM to score. For rewards $\{ R _ { i } \} _ { i = } ^ { G }$ the group relative advantage is

$$
A _ { i } = \frac { R _ { i } - \mathrm { m e a n } _ { j } ( R _ { j } ) } { \mathrm { s t d } _ { j } ( R _ { j } ) + \varepsilon } .\tag{6}
$$

Let $x _ { i , k - 1 }$ denote the sampled next state from $x _ { i , k }$ in the denoising chain. With the transition likelihood ratio $r _ { i , k } ( \theta ) = \exp [ \log p _ { \theta } ( x _ { i , k - 1 } \mid x _ { i , k } ) - \log p _ { \mathrm { o l d } } ( x _ { i , k - 1 } \mid x _ { i , k } ) ]$ we optimize the clipped objective

$$
\mathcal { L } _ { \mathrm { p o l i c y } } = - \mathbb { E } _ { i , k } \left[ \gamma ^ { N - k - 1 } \operatorname* { m i n } \left( r _ { i , k } ( \theta ) A _ { i } , \operatorname { c l i p } ( r _ { i , k } ( \theta ) , 1 - \delta , 1 + \delta ) A _ { i } \right) \right] ,\tag{7}
$$

where N is the number of denoising steps, $\gamma = 0 . 6$ weights transitions by their contribution to the final sample, and $\delta = 0 . 0 2$ . Since the trajectory is decoded from the unified latent, the likelihood ratio is computed over the transitions of the complete token sequence. A frozen reference model supplies a behavior cloning regularizer over the unified latent transition mean:

$$
\mathcal { L } _ { \mathrm { r e f } } = \frac { \| \mu _ { \theta } ^ { z } - \mu _ { \mathrm { r e f } } ^ { z } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } ,\tag{8}
$$

This regularizer keeps the policy close to the reference model on the decodable unified-latent manifold. The final objective is $\mathcal { L } _ { \mathrm { R L } } = \dot { \mathcal { L } } _ { \mathrm { p o l i c y } } + \lambda _ { \mathrm { r e f } } \mathcal { L } _ { \mathrm { r e f } }$ , with $\lambda _ { \mathrm { r e f } } = 0 . 1$ . We train rank-16, scale-32 LoRA adapters (Hu et al., 2022) on the attention projections for four inner epochs per rollout at learning rate $5 \times 1 0 ^ { - 5 }$ . DriveUniTok, its decoder, and the reference policy remain frozen throughout post-training.

## D TRAINING OBJECTIVE ABLATION

This ablation isolates the masking and semantic-supervision objectives from the representation comparison. During tokenizer training on NAVSIM, intra-modal and cross-modal masks are sampled randomly for each training example. Intra-modal masking removes patches or tokens within an observed modality and reconstructs them from the remaining evidence in that modality. Cross-modal masking hides one target modality and reconstructs it from the others. Semantic supervision refers to the training-only semantic feature reconstruction branch; when it is removed, the RGB DINO perceptual objective is retained, so the ablation affects only the semantic readout. All variants use the same unified architecture, latent budget, NAVSIM training split, optimization budget, and full-input NAVSIM navtest evaluation protocol. The results are summarized in Table 6.

Table 6: Training-objective ablation on NAVSIM navtest. Each row removes the named objectives from the full recipe. All variants share the architecture, 584 × 16 latent, and complete camera, depth, and ego inputs at evaluation. Best values, including ties, are bold.
<table><tr><td></td><td colspan="4">RGB</td><td colspan="2">Depth</td><td colspan="2">Ego</td></tr><tr><td>Variant</td><td> $\mathrm { L P I P S } _ { \mathrm { D I N O } }$ </td><td>↓ PSNR↑</td><td>SSIM↑</td><td>rFVDvJ↓</td><td>AbsRel↓</td><td>RMSE↓</td><td>ADE↓</td><td>FDE↓</td></tr><tr><td>w/o masked reconstruction and semantic supervision</td><td>0.1540</td><td>24.55</td><td>0.68</td><td>8.40</td><td>0.0695</td><td>9.30</td><td>0.1450</td><td>0.2670</td></tr><tr><td>w/o intra-modal masking</td><td>0.1215</td><td>25.20</td><td>0.72</td><td>4.85</td><td>0.0520</td><td>8.00</td><td>0.0900</td><td>0.1650</td></tr><tr><td>w/o cross-modal masking</td><td>0.1298</td><td>25.02</td><td>0.72</td><td>5.60</td><td>0.0565</td><td>8.30</td><td>0.1040</td><td>0.1910</td></tr><tr><td>w/o semantic supervision</td><td>0.1195</td><td>25.55</td><td>0.74</td><td>4.28</td><td>0.0480</td><td>7.72</td><td>0.0775</td><td>0.1420</td></tr><tr><td>Full training objective</td><td>0.1063</td><td>25.62</td><td>0.74</td><td>3.45</td><td>0.0461</td><td>7.57</td><td>0.0728</td><td>0.1329</td></tr></table>

Table 7: Performance on the NAVSIM-v2 navtest benchmark. Human Agent is included as a reference. DriveUniWAM (Ours) denotes the imitation-trained DriveUniWAM with a frozen DriveUniTok tokenizer. TLC denotes traffic-light compliance, HC denotes history comfort, EC denotes extended comfort, and EPDMS is the NAVSIM-v2 aggregate score. The best and second-best learned results, including ties, are shown in bold and underlined.
<table><tr><td>Method</td><td>Reference</td><td>NC↑</td><td>DAC↑</td><td>DDC↑</td><td>TLC↑</td><td>EP↑</td><td>TTC↑</td><td>LK↑</td><td>HC↑</td><td>EC↑</td><td>EPDMS↑</td></tr><tr><td colspan="10">Reference</td></tr><tr><td>Human Agent</td><td></td><td>100.0</td><td>100.0</td><td>99.8</td><td>100.0</td><td>87.4</td><td>100.0</td><td>100.0</td><td>98.1</td><td>90.1</td><td>94.5</td></tr><tr><td colspan="10">Vision-Based End-to-End Methods</td></tr><tr><td>TransFuser (Chitta et al., 2023)</td><td>TPAMI&#x27;23</td><td>96.90</td><td>89.90</td><td>97.80</td><td>99.70</td><td>87.10</td><td>95.40</td><td>92.70</td><td>98.30</td><td>87.20</td><td>76.7</td></tr><tr><td>DriveSuprim (Yao et al., 2026)</td><td>AAAI&#x27;26</td><td>97.50</td><td>96.50</td><td>99.40</td><td>99.60</td><td></td><td></td><td>95.50</td><td>98.30</td><td>77.00</td><td>83.1</td></tr><tr><td>DiffusionDrive (Liao et al., 2025a)</td><td>CVPR&#x27;25</td><td>98.20</td><td>95.90</td><td>99.40</td><td>99.80</td><td>88.40 87.50</td><td>96.60 97.30</td><td>96.80</td><td>98.30</td><td>87.70</td><td>84.5</td></tr><tr><td colspan="10">Vision-Language-Action Methods</td></tr><tr><td>ReCogDrive (Li et al., 2026d)</td><td>ICLR&#x27;26</td><td>98.30</td><td>95.20</td><td>99.50</td><td>99.80</td><td>87.10</td><td>97.50</td><td>96.60</td><td>98.30</td><td>86.50</td><td>83.6</td></tr><tr><td>SGDrive (Li et al., 2026a)</td><td>CVPR&#x27;26</td><td>98.60</td><td>94.30</td><td>99.50</td><td>99.90</td><td>86.00</td><td>97.90</td><td>96.10</td><td>98.30</td><td>85.90</td><td>86.2</td></tr><tr><td>DriveFine (Dang et al., 2026)</td><td>arXiv&#x27;26</td><td>98.70</td><td>97.30</td><td>99.50</td><td>99.80</td><td>88.70</td><td>97.80</td><td>97.70</td><td>98.40</td><td>83.80</td><td>89.7</td></tr><tr><td>DriveVLA-W0 (Li et al., 2026c)</td><td>ICLR&#x27;26</td><td>98.50</td><td>99.10</td><td>98.00</td><td>99.70</td><td>86.40</td><td>98.10</td><td>93.20</td><td>97.90</td><td>58.90</td><td>86.1</td></tr><tr><td colspan="10">World-Action Models</td></tr><tr><td>DVGT-2 (Zuo et al., 2026b)</td><td>arXiv&#x27;26</td><td>98.70</td><td>97.90</td><td>99.70</td><td>99.90</td><td>87.90</td><td>98.00</td><td>98.20</td><td>98.40</td><td>77.00</td><td>89.6</td></tr><tr><td>EponaV2 (Xu et al., 2026)</td><td>arXiv&#x27;26</td><td>98.50</td><td>97.40</td><td>99.50</td><td>99.90</td><td>87.90</td><td>98.10</td><td>97.70</td><td>98.20</td><td>77.40</td><td>88.9</td></tr><tr><td>DriveLaW (Xia et al., 2026)</td><td>CVPR&#x27;26</td><td>98.70</td><td>96.90</td><td>99.60</td><td>99.80</td><td>87.50</td><td>98.30</td><td>97.60</td><td>98.40</td><td>77.40</td><td>88.6</td></tr><tr><td>DriveUniWAM (Ours)</td><td></td><td>98.63</td><td>98.11</td><td>99.60</td><td>99.88</td><td>88.11</td><td>98.51</td><td>97.83</td><td>98.18</td><td>80.65</td><td>90.3</td></tr></table>

This ablation complements the equal-budget comparison by examining how the training objectives affect the quality of the unified latent itself. The full recipe achieves the best results across all reported RGB, depth, and ego metrics. Compared with omitting masked reconstruction and semantic supervision, it reduces LPIPS<sub>DINO</sub> from 0.1540 to 0.1063, rFVD<sub>VJ</sub> from 8.4000 to 3.4546, depth AbsRel from 0.0695 to 0.0461, and trajectory ADE from 0.1450 to 0.0728. These results indicate that both the cross-modal reconstruction objective and the auxiliary semantic target contribute to the quality of the unified latent used by DriveUniWAM.

## E NAVSIM-V2 PLANNING RESULTS

We further evaluate the planning model under the NAVSIM-v2 protocol, which includes additional scene- and motion-compliance criteria. Table 7 shows that DriveUniWAM reaches 90.3 EPDMS while maintaining high scores across the individual metrics. This result supports the effectiveness of the unified latent representation for WAM planning: it preserves scene evidence, semantic structure, and the metric ego state needed to evaluate future action sequences.

Table 8: Modality sensitivity for reconstruction and joint future generation on NAVSIM navtest. C, L, and T denote camera, LiDAR/depth, and ego-trajectory modalities. Input and reconstruction Output follow the original tokenizer masking setup; the DriveUniWAM block evaluates future generation under the same input condition. Best results are highlighted in bold.
<table><tr><td></td><td></td><td colspan="4">Tokenizer Reconstruction</td><td colspan="4">Joint Future Generation</td></tr><tr><td>Input</td><td>Output</td><td>LPIPSDINO ↓</td><td>rFVDvJ↓</td><td>Depth AbsRel↓</td><td>Ego ADE↓</td><td>LPIPSDINO ↓</td><td>gFVDvJ↓</td><td>Depth AbsRel↓</td><td>Traj. ADE↓</td></tr><tr><td>C</td><td>C+L+T</td><td>0.1671</td><td>8.8645</td><td>0.1274</td><td>1.0512</td><td>0.1170</td><td>12.1153</td><td>0.2185</td><td>3.1991</td></tr><tr><td>L</td><td>L+T</td><td>一</td><td>一</td><td>0.1202</td><td>0.3350</td><td>一</td><td>一</td><td>0.2934</td><td>4.5854</td></tr><tr><td>T</td><td>T</td><td>一</td><td>一</td><td></td><td>0.2978</td><td>一</td><td>一</td><td></td><td>2.3534</td></tr><tr><td>C+L</td><td>C+L+T</td><td>0.1070</td><td>3.6430</td><td>0.0463</td><td>0.0876</td><td>0.0985</td><td>8.8587</td><td>0.1669</td><td>2.6611</td></tr><tr><td>C+L+T</td><td>C+L+T</td><td>0.1063</td><td>3.4546</td><td>0.0461</td><td>0.0728</td><td>0.0955</td><td>6.6076</td><td>0.1059</td><td>0.3816</td></tr></table>

## F MODALITY SENSITIVITY

We finally vary the available input modalities to examine how different sources of evidence contribute to the unified latent, reconstruction, and joint future generation. Table 8 reports evaluation-time modality sensitivity under five input conditions, with the Input and Output configurations following the tokenizer masking setup. The DriveUniWAM block uses the same input condition as the tokenizer evaluation, allowing the effect of each modality to be tracked from representation quality to downstream prediction. Because these variants mask modalities at evaluation time rather than retraining with modality dropout, we do not interpret them as a complete sensor-failure robustness study.

Using all three input modalities yields the strongest overall reconstruction and future-generation results, indicating that the unified latent benefits from jointly integrating complementary visual, geometric, and motion evidence.

## G QUALITATIVE NAVSIM PLANNING VISUALIZATIONS

The qualitative comparisons cover both vision-language-action and world-action modeling paradigms. Each group contains a front-view observation followed by predictions from AutoVLA and ReCog-Drive (VLA methods), PWM (a world-action baseline), DriveUniWAM (Ours), and DriveUniWAM with reinforcement learning post-training. AutoVLA produces reasonable paths in straight-road scenes with simple geometry, but can under-steer at intersections and turns, resulting in large turning radii or endpoints that deviate from the ground-truth route. ReCogDrive generally makes better use of road topology, while PWM captures the broad scene-conditioned motion trend but can retain local curvature or endpoint errors in more demanding cases. DriveUniWAM further improves trajectory shape, turning curvature, and endpoint placement. After reinforcement learning post-training, the predicted paths show greater stability in complex intersections, turns, merges, and scenes with an explicit navigation direction: the model adjusts its direction earlier and better matches the ground-truth curvature, position, and final heading. Overall, the visualizations show that DriveUniWAM produces feasible trajectories across both comparison paradigms, while reinforcement learning post-training further improves their agreement with driving feasibility.

![](images/814bbb8675d48e3b3a85e11936a3ff6573f327efc7863e4e3db1f088b570ef35.jpg)  
Figure 7: Qualitative NAVSIM navtest visualizations for scene group 1. From left to right, the columns show the front-view observation, AutoVLA, ReCogDrive, PWM, DriveUniWAM (Ours), and DriveUniWAM with reinforcement learning post-training. AutoVLA and ReCogDrive are visionlanguage-action methods, whereas PWM is a world-action baseline.

![](images/2962af3d0fc177daa0975961dcfdf3bc639d4e1e51adcf4077db43a82b5f3ecd.jpg)  
Figure 8: Qualitative NAVSIM navtest visualizations for scene group 2. The columns follow the ordering defined in Fig. 7.

![](images/5e89a0bf2f1ed045a36bf5a836f089d8464557eb1404460082cae4bc79fa33e5.jpg)  
Figure 9: Qualitative NAVSIM navtest visualizations for scene group 3. The columns follow the ordering defined in Fig. 7.

![](images/a8c3fcba59bd3d042c02c84234f6d1b9cf8db01b7934dd5a987e5d2b38c2a958.jpg)  
Figure 10: Qualitative NAVSIM navtest visualizations for scene group 4. The columns follow the ordering defined in Fig. 7.

![](images/762cd9273cc5f37eeb1e904d01196dae377749f92c8eb4e6460d63c69eff9c39.jpg)  
Figure 11: Qualitative NAVSIM navtest visualizations for scene group 5. The columns follow the ordering defined in Fig. 7.

![](images/e77898e97ce1232ef97b6eb16b6f45a48b55e49e9c5f6f4caf8398a38f7c60ab.jpg)  
Figure 12: Qualitative NAVSIM navtest visualizations for scene group 6. The columns follow the ordering defined in Fig. 7.

![](images/1c49c7a351cb2dec7034d92fdf000c5e72e45e4e6f18d33653fa351c4b1510ef.jpg)  
Figure 13: Qualitative NAVSIM navtest visualizations for scene group 7. The columns follow the ordering defined in Fig. 7.

![](images/0b5ed11005505ed6adf651ff58fe38e8339f75dbcdffa1edddad9575c88f41b5.jpg)  
Figure 14: Qualitative NAVSIM navtest visualizations for scene group 8. The columns follow the ordering defined in Fig. 7.

![](images/8ce236f2997f87ae2e63220360b51bb45d35eb48f306387e46aae39ca2e55b2f.jpg)  
Figure 15: Qualitative NAVSIM navtest visualizations for scene group 9. The columns follow the ordering defined in Fig. 7.

![](images/9503d0d6889359c764206f80ec856b928ca1529b41a997128ca512edbecdaedb.jpg)  
Figure 16: Qualitative NAVSIM navtest visualizations for scene group 10. The columns follow the ordering defined in Fig. 7.