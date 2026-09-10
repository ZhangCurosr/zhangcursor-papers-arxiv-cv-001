# Decoupled Self-Forcing Distillation for Streaming Talking Head Generation

Yanru An<sup>1</sup>, Ruiyan Wang<sup>1</sup>, Wenwu Wei<sup>1</sup>, Rui Bu<sup>2</sup>, Qi Wang<sup>2</sup>, Hongwei Hu<sup>2</sup> Zhengxue Cheng<sup>1</sup>, Rong Xie<sup>1</sup>, Li Song<sup>1</sup>, Wenjun Zhang<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Ant Group

![](images/abf07a6588310b5a9cc840ad0fb88a130e77766e55528c547a5269f4cff7d091.jpg)  
Figure 1: Given a reference face, driving audio, and a motion caption, Motar autoregressively synthesizes highly lip-synced talking-head video through two parallel streams: one generating ID-disentangled motion latents, the other rendering them into video. Each frame is produced as its audio arrives.

## Abstract

Streaming talking-head generation produces each frame as its driving audio arrives, yet fidelity and eficiency have so far pulled in opposite directions: end-to-end methods condition a video difusion model on audio directly and achieve high qual ity but only at large scale, while cheaper two-stage methods generate an intermediate motion representation and trail in fidelity. We argue the cost of the former lies in the target of fusion: the video latent is dominated by identity, appearance and background, none of which audio bears on, so coupling audio to every pixel blurs detail and wastes capacity. We instead fuse conditions in a low-dimensional identity-disentangled motion space, routing audio and motion captions by their temporal granularity, and generate motion latents with a small causal autoregressive transformer that a pretrained difusion renderer turns into video. Conditions thus control video transitively, and high fidelity no longer requires a large backbone. Streaming this decomposition needs both models to be causal, and the exposure-bias problem could be solved by self-forcing given a bidirectional teacher. But there is no such teacher in motion space. Our decoupled self-forcing distillation resolves both models under one frozen teacher: conditioned on motion, it distills the renderer into a block-causal student; unconditionally, it scores rendered rollouts against real videos, supervising motion by the video it produces. This lifts the fidelity ceiling from the motion generator onto the stronger renderer. The two models run as parallel causal streams, reaching 15.4 FPS at 1.3 s latency with no quality degradation.

## 1 Introduction

Audio-driven talking-head generation has advanced rapidly with video difusion models (VDMs) (Xu et al. 2024a; Chen et al. 2025c; Lin et al. 2025; Chen et al. 2025b), yet real-time applications like live streaming demand streaming synthesis, in which each frame is generated at low latency as its driving audio arrives. Existing methods fall into two families. Endto-end methods finetune a pretrained VDM (Wan et al. 2025; Yang et al. 2025b) with cross-attention for condition injection (Huang et al. 2025; Sun et al. 2026; Wang et al. 2025; Ki et al. 2026) and use autoregressive (AR) distillation (Yin et al. 2025, 2024; Huang et al. 2026) to make it few-step and block-causal. Two-stage methods instead generate a compact motion representation from audio and hand pixels to a small renderer (Chu et al. 2025; Chen and Liu 2025; Li et al. 2025; Ling et al. 2024). The two trade against each other: end-toend methods reach the highest fidelity but introduces a large number of parameters, while two-stage methods have lower computational costs but trail in fidelity (Wang et al. 2025).

We trace this tension to where conditions are aligned. Endto-end methods align audio to the video latent, and blame the resulting cost on the dificulty of multimodal fusion. We argue the dificulty lies not in the fusion operation but in its target: the video latent is dominated by identity, texture and lighting conditions, none of which is causally determined by audio. Coupling audio to every pixel forces it to interact with texture, lighting and identity; trained at scale, the model regresses these entangled factors toward their conditional mean, blurring fine detail much as a mean-seeking objective would. The natural target is not pixels but head motion, a low-dimensional, identity-free manifold where alignment is leakage-free and cheap. We therefore build on the twostage paradigm. Its known weakness is a fidelity gap: existing methods optimize the motion generator entirely within motion space, isolated from the renderer that consumes its output, so fidelity is capped by the motion generator rather than the far stronger renderer.

We propose Motar, a two-stage streaming talking-head generation method. We adopt the identity-disentangled motion latent of X-NeMo (Zhao et al. 2025) and generate it with a causal transformer whose per-frame difusion head (Li et al. 2024) models continuous values without the quantization error of a discrete codebook (Chu et al. 2025). A hierarchical conditioning scheme fuses audio and a motion caption by a Q-Former (Li et al. 2023) into a global coarse condition, while per-frame audio drives a local branch for lip synchronization in the d-dimensional motion space, keeping it lightweight. The key advantage is indirect but decisive: motion controls video and audio controls motion, so conditions control video transitively. High fidelity thus needs no large backbone; the renderer need not know what angry means, only what a motion latent looks like in pixels.

Streaming this decomposition exposes two exposure-bias problems: the AR motion model drifts on its own rollout, and the bidirectional renderer must be distilled into a few-step block-causal one (Yin et al. 2025). Self-forcing (Huang et al. 2026) is the standard remedy, but presupposes a frozen bidirectional teacher, which exists in video space and not in motion space. Our decoupled self-forcing distillation resolves both under the same frozen X-NeMo teacher: its conditional form distills the renderer via distribution matching (Yin et al. 2024); its unconditional form scores rendered rollouts against real videos, backpropagating through the frozen renderer into the motion model. Motion is thus optimized by its videospace output rather than its latent-space precision, lifting the fidelity ceiling onto the renderer and countering the diversity collapse of regression-based self-forcing.

Our contributions are three-fold:

• We relocate multimodal fusion to a low-dimensional motion manifold, giving transitive control over video that decouples fidelity from backbone scale.

• We design a causal motion generator with a continuous difusion head and hierarchical conditioning separating coarse text–audio control from frame-level lip sync.

• We propose decoupled self-forcing distillation, resolving two exposure-bias problems under one frozen teacher and lifting the two-stage fidelity ceiling onto the renderer, yielding a fully streaming two-stream causal pipeline.

## 2 Related Work

## 2.1 Audio-Driven Talking Head Generation

Early methods targeted lip synchronization (Prajwal et al. 2020), and were later extended toward expressive faces and natural head motion (Zhang et al. 2023; Wang et al. 2021).

Difusion-based approaches now split along two axes. Endto-end methods fine-tune a pretrained VDM to attend to audio directly (Xu et al. 2024a; Chen et al. 2025c; Tian et al. 2024; Lin et al. 2025; Zheng et al. 2024; Ji et al. 2025; Chen et al. 2025b), achieving strong lip-sync and naturalness. Twostage methods instead generate a compact identity-agnostic motion representation from audio and delegate pixels to a renderer, using either explicit keypoints (Li et al. 2025; Guo et al. 2024) or learned disentangled latents (Liu et al. 2024; Xu et al. 2024b; Ki, Min, and Chae 2025; Chen et al. 2025a), cutting cost by orders of magnitude. The standard objection to the latter family is that it is bottlenecked by the fidelity of the generated motion, and so trails end-to-end methods in naturalness (Wang et al. 2025). We attribute this to how the decomposition is trained: existing two-stage methods optimize the motion generator entirely within motion space, in isolation from the renderer that consumes its output. We close this gap by supervising the motion generator with a video-level distribution matching objective.

## 2.2 Streaming and Real-Time Avatars

Two routes lead to real time. One distills a large bidirectional VDM into a causal, few-step student (Low and Wang 2025; Huang et al. 2025; Sun et al. 2026; Wang et al. 2025; Ki et al. 2026), which streams but only at large-parameter scale and, in several cases, across multiple GPUs; READ (Wang et al. 2026) accelerates the backbone yet remains nonautoregressive and cannot stream at all. The other makes the motion generator autoregressive, which is lightweight, but couples it to a renderer whose expressiveness is structurally bounded — either an explicit 3D avatar driven by parametric coeficients (Chu et al. 2025; Chu and Harada 2024), whose expressions are confined to the span of a fixed basis, or a warping-based decoder (Chen and Liu 2025; Wang et al. 2024), which transports pixels frame by frame and struggles with fine detail and temporal consistency. Either way, only one end of the pipeline is temporally causal. We pair an implicit motion latent with a difusion renderer and causalize both, running them as two parallel streams.

## 2.3 Autoregressive Difusion Distillation

AR models trained with teacher forcing degrade over long rollouts, as inference exposes them to their own compounding error. Difusion Forcing (Chen et al. 2024) conditions at arbitrary noise levels; CausVid (Yin et al. 2025) distills a block-causal VDM with DMD (Yin et al. 2024), but still accumulates error over long horizons; Self-Forcing (Huang et al. 2026) closes the gap directly, unrolling the student on its own predictions during training and matching its output distribution to a frozen bidirectional teacher, with subsequent work pushing this to minute-scale (Yang et al. 2025a; Zhu et al. 2026). Every method in this line presupposes that such a teacher exists. The premise holds in video latent space, where large pretrained bidirectional difusion models are available, but fails in motion latent space, where no pretrained bidirectional motion teacher exists and an AR motion generator therefore has exposure bias with no score to correct it against. We resolve both cases under a single frozen video teacher.

![](images/f7e700c98cd0b4a127aa7cd12554cf43eb2420b0480e94d961fb05c667e79524.jpg)  
Figure 2: Overview of Motar. (a) An AR motion transformer models causal dependencies over motion latents, with an eficient difusion head producing continuous-valued latents; a user-written motion caption and the driving audio are fused into a global condition query by a Q-Former projector. (b) Hierarchical conditioning: the global query is injected by full cross-attention for coarse, sequence-level control, while raw audio embeddings are injected by windowed cross-attention for frame-level lip articulation. (c) Decoupled self-forcing distillation: the frozen bidirectional teacher G supervises both branches by distribution matching. Conditioned on motion, it distills G into a block-causal student $G _ { \phi } ;$ unconditionally, it matches the rendered motion rollout against the distribution of real talking videos.

## 3 Method

The overview ofour proposed Motar is shown in Fig. 2. Given a reference image $\mathbf { I } _ { \mathrm { r e f } } .$ , a driving audio $c _ { a } ,$ , and an optional motion caption $c _ { t }$ describing the desired emotion, amplitude and head movement, our goal is to synthesize a talking-head video in which the subject speaks and emotes accordingly while preserving the appearance of $\mathbf { I } _ { \mathrm { r e f } }$

Motion–rendering decomposition. Following Sec. 1, we align conditions with motion rather than pixels. We adopt the pretrained identity-disentangled motion encoder ${ \mathcal { E } } _ { m }$ of X-NeMo (Zhao et al. 2025), which maps a video $\mathbf { V } = \left( \mathbf { I } _ { 1 } , \ldots , \mathbf { I } _ { T } \right)$ to a 1D motion latent sequence ${ \bf M } =$ $( \mathbf { m } _ { 1 } , \ldots , \mathbf { m } _ { T } ) , \mathbf { m } _ { t } \in \mathbb { R } ^ { d }$ , paired with a difusion decoder G that reconstructs the video from motion latents and a reference frame, $G ( \mathbf { M } , \mathbf { I } _ { \mathrm { r e f } } ) \approx { \bf V }$ . Since ${ \mathcal { E } } _ { m }$ is identitydisentangled, M retains only the low-dimensional dynamics of expression and head motion, to which $c _ { a }$ and $c _ { t }$ are causally related. The task splits into

$$
\mathbf { M } = \mathcal { F } _ { \mathrm { m o t i o n } } ( c _ { a } , c _ { t } ) , \qquad \mathbf { V } = G ( \mathbf { M } , \mathbf { I } _ { \mathrm { r e f } } ) ,\tag{1}
$$

where the first prediction task is low-dimensional and identity-free, needing only a small model (Sec. 3.1), and the second is handled by G, distilled into a causal student in

Sec. 3.3. Since G decodes latents locally, the two run as parallel streams, rendering frame t as soon as $\mathbf { m } _ { t }$ is generated.

## 3.1 Autoregressive Motion Latent Generation

$\mathcal { F } _ { \mathrm { m o t i o n } }$ predicts $\mathbf { m } _ { t } \in \mathbb { R } ^ { d }$ from the history $\mathbf { m } _ { < t }$ and the conditions. As $\mathbf { m } _ { t }$ is continuous rather than discrete, we follow MAR (Li et al. 2024) and replace the softmax head with a small conditional difusion head $v _ { \phi } ,$ avoiding the quantization error imposed by a VQ codebook on facial motion (Chu et al. 2025) and separating causal sequence modeling (the backbone $f _ { \theta } )$ from distribution modeling $( v _ { \phi } )$ With $\mathbf { z } _ { t } = f _ { \theta } ( \mathbf { m } _ { < t } , \mathbf { c } ^ { g } , \mathbf { c } _ { t } ^ { l } )$ the conditioning vector from the backbone, the head is trained with a v-prediction objective,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { \tau , \epsilon } \big [ \| v _ { \phi } ( \mathbf { m } _ { t } ^ { \tau } , \tau , \mathbf { z } _ { t } ) - \mathbf { v } ^ { \mathbf { g t } } \| _ { 2 } ^ { 2 } \big ] , } \end{array}\tag{2}
$$

where $\tau$ is the difusion timestep and m<sup>τ</sup> the noised target. The backbone consists of N causal transformer blocks with RoPE(Su et al. 2024), unrolled at inference with a rolling KV cache; we cache pre-RoPE projections and re-apply RoPE relative to the current window, keeping training and streaming inference consistent.

Hierarchical conditioning. How the two modalities enter $f _ { \theta }$ is dictated by their temporal granularity. Audio is tied to lip movement frame by frame, meaningful only at frame resolution; a motion caption asserts a sequence-level property with no frame-level timestamp. Routing both through one cross-attention pathway would force a frame-level signal and a sequence-level one to share attention weights. We therefore split conditioning into two branches. The global branch fuses text embeddings $\mathbf { \bar { e } } ^ { \mathrm { t x t } }$ and audio embeddings $\mathbf { e } ^ { \mathrm { a u d } }$ through a Q-Former (Li et al. 2023) with learnable queries into a compact sequence ${ \bf c } ^ { g } = g _ { \psi } ( { \bf e } ^ { \mathrm { t x t } } , { \bf e } ^ { \mathrm { a u d } } )$ that every frame attends to; sharing these queries across the sequence biases it toward coarse, sequence-level control such as emotion and amplitude. The local branch lets frame t attend only to a short audio window ${ \bf c } _ { t } ^ { l } = { \bf e } _ { t - w _ { a } : t + w _ { a } } ^ { \mathrm { a u d } }$ with window size $w _ { a } = 2 .$ supplying the resolution for lip sync; applied independently per frame, it needs no cross-frame cache. Each block applies causal self-attention, global and local cross-attention, and an FFN, each pre-normalized and gated by a learnable residual scale (Touvron et al. 2021). Because fusion happens in the $d -$ dimensional motion space, $g _ { \psi }$ and the cross-attention layers cost only a fraction of the parameters such modules need on a video backbone. The backbone is pretrained with teacher forcing (TF), conditioning $\mathbf { z } _ { t }$ on ground-truth history.

Bounded look-ahead for streaming. Neither branch is strictly causal as the global query summarizes a window of audio and the local branch peeks $w _ { a }$ frames ahead, yet both preserve streaming with bounded latency since their look-ahead is bounded and independent of sequence length. In practice the global branch is fed a sliding window rather than the whole utterance: audio arrives faster than motion is generated, so this window is available without stalling, and the caption is compressed by the Q-Former into an embedding carrying no fine-grained future content. The local peek is a deliberate concession: a phoneme spans several frames, so a window sharpens lip sync at a fixed, small delay. The model is thus not strictly frame-causal but has bounded lookahead, keeping first-frame latency constant and inference indefinitely streamable while improving quality.

## 3.2 Self-Forcing of the Motion Generator

At inference the model conditions on its own predictions, so any error in $\hat { \mathbf { m } } _ { t }$ compounds. Closing this gap requires fine-tuning on the model’s own rollout, which must be differentiable end to end. Backpropagating through a multi-step denoiser at every AR step is computationally prohibitive, as the graph grows with the product of sequence length and denoising steps. We therefore first distill $v _ { \phi }$ into a single-step consistency sampler $v _ { \phi } ^ { \mathrm { o n e - s t e p } }$ (Luo et al. 2023), which both enables real-time generation and makes the rollout afordable. During rollout, the ground-truth frame is replaced by the model’s own prediction with probability $p _ { \mathrm { s f } }$ , annealed $0  1$ , with $v _ { \phi } ^ { \mathrm { o n e - s t e p } }$ frozen so gradients flow into $f _ { \theta }$ alone; as $p _ { \mathrm { s f } } \to 1$ the backbone is directly exposed to its own compounding error.

Why regression alone fails. The natural supervision is a masked regression against ground truth,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m s e } } = \frac { 1 } { | \mathcal { V } | } \sum _ { t \in \mathcal { V } } \| \hat { \mathbf { m } } _ { t } - \mathbf { m } _ { t } ^ { \mathrm { g t } } \| _ { 2 } ^ { 2 } , } \end{array}\tag{3}
$$

where V indexes rolled-out frames. But Eq. 3 is meanseeking: it drives the model toward the conditional expectation of plausible next frames rather than a sample from their distribution, collapsing motion amplitude and diversity over a long rollout.

Adversarial loss in motion space. We therefore add an adversarial term (Goodfellow et al. 2014) matching the rollout to the real motion distribution rather than its mean. A discriminator D judges whether an entire sequence is plausible, with no per-frame target to collapse onto:

$$
\mathcal { L } _ { \mathrm { a d v } } = \mathbb { E } _ { \mathbf { m } ^ { \mathrm { g t } } } [ \log D ( \mathbf { m } ^ { \mathrm { g t } } ) ] + \mathbb { E } _ { \hat { \mathbf { m } } } [ \log ( 1 - D ( \hat { \mathbf { m } } ) ) ] .\tag{4}
$$

D operates on motion sequences unconditionally, penalizing rollouts whose temporal statistics degenerate. ${ \mathcal { L } } _ { \mathrm { m s e } }$ is kept as a weak anchor, and $w _ { \mathrm { a d v } }$ ramped in once D warms up. This constrains motion within its own space; whether it renders into a plausible video is addressed in Sec. 3.3.

## 3.3 Decoupled Self-Forcing Distillation

Exposure bias is standardly addressed by distilling a causal student against a bidirectional teacher. In our decomposition, $\mathcal { F } _ { \mathrm { m o t i o n } }$ is autoregressive but has no bidirectional teacher in its own space, while G is such a teacher but is bidirectional and multi-step and must be made block-causal and few-step. We resolve both by Self-Forcing against the same frozen bidirectional $G ;$ the two branches difer in one respect: whether the score model is conditioned on motion.

Branch 1: distilling a causal renderer. We convert $G \ ' \mathrm { s }$ temporal attention from bidirectional to block-causal with a rolling KV cache over past blocks, leaving per-frame spatial and reference-conditioning layers unchanged. The student $G _ { \phi }$ is unrolled block by block on its own blocks as at inference, and supervised by DMD: a frozen G is the real-score model, and an online copy $s _ { \mathrm { f a k e } } .$ , updated by a denoising loss on the student’s output, tracks its distribution. Both are conditioned on ground-truth motion m, as the task is to reproduce the teacher’s rendering of a given motion:

$$
\nabla _ { \hat { \mathbf { x } } _ { 0 } } \mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { v i d e o } } \propto s _ { \mathrm { f a k e } } \big ( \hat { \mathbf { x } } _ { \tau } \mid \mathbf { m } , \mathbf { I } _ { \mathrm { r e f } } \big ) - s _ { \mathrm { r e a l } } \big ( \hat { \mathbf { x } } _ { \tau } \mid \mathbf { m } , \mathbf { I } _ { \mathrm { r e f } } \big ) ,\tag{5}
$$

where $\hat { \mathbf { x } } _ { 0 } = G _ { \phi } ( \mathbf { x } _ { \tau } , \tau , \mathbf { m } , \mathbf { I } _ { \mathrm { r e f } } )$ and $\hat { \mathbf { x } } _ { \tau }$ its renoised counterpart. The gradient approaches zero only when $G _ { \phi } \mathbf { \vec { s } }$ distribution matches $\mathit { G } \mathit { \mathbf { \hat { s } } . \ G _ { \phi } }$ is then frozen for all subsequent training.

Branch 2: matching motion in video space. A motion sequence is meaningful only insofar as it renders into a plausible video, yet Sec. 3.2 supervises it entirely within motion space — the fidelity gap two-stage methods are criticized for. With a diferentiable renderer, we instead supervise motion by the video it produces: we decode a window of the rollout $\hat { \mathbf { m } } _ { k : k + w }$ through the frozen $G _ { \phi }$ into $\hat { \mathbf { x } } _ { 0 }$ and apply DMD,

$$
\nabla _ { \hat { \mathbf { x } } _ { 0 } } \mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { m o t i o n } } \propto s _ { \mathrm { f a k e } } \big ( \hat { \mathbf { x } } _ { \tau } ~ | ~ \mathbf { I } _ { \mathrm { r e f } } \big ) - s _ { \mathrm { r e a l } } \big ( \hat { \mathbf { x } } _ { \tau } ~ | ~ \mathbf { I } _ { \mathrm { r e f } } \big ) ,\tag{6}
$$

$$
\frac { \partial \mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { m o t i o n } } } { \partial \theta } = \frac { \partial \mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { m o t i o n } } } { \partial \hat { \mathbf { x } } _ { 0 } } \cdot \frac { \partial \hat { \mathbf { x } } _ { 0 } } { \partial \hat { \mathbf { m } } } \cdot \frac { \partial \hat { \mathbf { m } } } { \partial \theta } ,\tag{7}
$$

where θ denotes $\mathcal { F } _ { \mathrm { m o t i o n } } \mathrm { ^ { * } s }$ parameters. Since $G _ { \phi }$ and $\mathbf { I } _ { \mathrm { r e f } }$ are fixed, motion is the only free variable, and the video-level gradient is transported back onto it.

Why this branch must be unconditional.Conditioning the score models on the generated mˆ , as in Eq. 5, fails because G is a faithful conditional renderer: given any motion it renders a consistent video and assigns it high density. The two scores then agree, the gradient vanishes, and no force acts on motion. The failure is structural: $s _ { \mathrm { r e a l } } ( \cdot \mid \hat { \mathbf { m } } )$ asks whether xˆ is a plausible rendering of mˆ , and the answer is always yes, never whether mˆ is itself plausible. Conditioning on ground-truth motion is no remedy either: the video would be evaluated under a condition it was not rendered from, reintroducing a per-frame target and the mean-seeking behaviour of Eq. 3. We therefore drop the motion condition, obtaining an unconditional score model by fine-tuning G on real videos with the motion condition removed, retaining only $\mathbf { I } _ { \mathrm { r e f } } ,$ so that $s _ { \mathrm { r e a l } } ( \cdot \ | \ \mathbf { I } _ { \mathrm { r e f } } )$ models the marginal distribution of real talking videos. A collapsed motion now renders to a near-static video, which this model assigns low density, and the gradient is non-zero.

Complementary supervision in two spaces. The full objective for $\mathcal { F } _ { \mathrm { m o t i o n } }$ combines three terms,

$$
\mathcal { L } _ { \mathrm { m o t i o n } } = \mathcal { L } _ { \mathrm { m s e } } + w _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } } + w _ { \mathrm { d m d } } \mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { m o t i o n } } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { m s e } }$ is an anchor that keeps mˆ in the region where the score models are reliable before the distribution-level terms dominate. The other two are both distribution-matching but act in diferent spaces. Applying ${ \mathcal { L } } _ { \mathrm { a d v } }$ directly to the motion latents is computationally inexpensive and allows the full rollout to be evaluated jointly, thereby capturing long-range temporal statistics such as amplitude and rhythm. $\mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { m o t i o n } }$ operates on the rendered video, evaluating exactly what the viewer sees and penalizing motion that is well-formed yet renders poorly, but observes only a short window per step since rendering is expensive. Supervising in both spaces covers what either alone would miss, lifting the fidelity ceiling: motion is no longer optimized in latent space alone but by the video it renders into, so system quality is bounded by the renderer $G _ { \phi }$ rather than the motion generator $\mathcal { F } _ { \mathrm { m o t i o n } } .$

## 4 Experiments

## 4.1 Experimental Setup

Dataset. We combine MEAD (Wang et al. 2020) and Hallo3- data (Cui et al. 2025), covering both acted emotional and in-the-wild talking-head videos. All clips are resampled to 25 fps and cropped to $5 1 2 \times 5 1 2$ . We use Qwen2.5-VL-7B (Bai et al. 2025) to generate a textual caption for each clip, describing the speaker’s emotion and head pose movement; caption accuracy was verified by manual inspection on a random subset.

Implementation Details. We extract 512-D motion latents using the pretrained X-NeMo (Zhao et al. 2025) encoder, with text and audio features encoded by umT5-base (Raffel et al. 2020) and wav2vec2-base (Baevski et al. 2020), respectively. The motion generator comprises 8 causal transformer blocks (dim 512) paired with a 2-layer MLP difusion head, and consists of only 77M parameters. Training follows a four-stage schedule: (1) teacher-forced pre-training of the backbone, (2) distillation of the difusion head into a single-step consistency sampler, (3) self-forcing fine-tuning unrolled over 256 frames (transitioning from ${ \mathcal { L } } _ { \mathrm { m s e } }$ to ${ \mathcal { L } } _ { \mathrm { a d v } } ) ,$ and (4) decoupled self-forcing distillation where the motion generator is supervised by the distilled block-causal renderer $G _ { \phi }$ via $\mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { m o t i o n } }$ (rendering a 64-frame window). $G _ { \phi }$ is distilled following Self-Forcing, using 64-length rolling KV-cache, 8- frame denoising blocks and 4-step sampling. All training is conducted with Adam optimizer on 4 NVIDIA A100 GPUs. More details are provided in the Supplementary Material.

Baselines. We compare our model against state-of-theart talking head generation methods, including two-stage approaches (SadTalker (Zhang et al. 2023) and AniPortrait (Wei, Yang, and Wang 2024)) and end-to-end frameworks (Hallo3 (Cui et al. 2025), StableAvatar (Tu et al. 2025), AvatarForcing (Ki et al. 2026), and LiveAvatar (Huang et al. 2025)). We additionally report the reconstruction oracle, formulated as $G ( \mathbf { M } ^ { \mathrm { g t } } , \mathbf { I } _ { \mathrm { r e f } } )$ , which decodes ground-truth motion latents and serves as an upper bound for any method based on this latent decomposition.

Metrics. We evaluate visual quality using FID and FVD, and measure identity preservation with CSIM (Deng et al. 2019a). Lip synchronization is assessed via $\Delta \mathrm { S y n c - C }$ and ∆Sync-D (Chung and Zisserman 2016), which denote the absolute diferences between the computed sync scores and the ground-truth values of the test set. Emotion controllability is quantified by E-FID (Tian et al. 2024), computed as the Fréchet Distance of expression parameters extracted following (Deng et al. 2019b). Finally, computational eficiency is evaluated in terms of first-frame latency (Lat.), steady-state throughput and parameter count (Param.) with each method generating a 200-frames video with BF16 precision on a single NVIDIA H200 GPU.

## 4.2 Quantitative Comparison

Table 1 shows that our method leads decisively on lip synchronization: ∆Sync-C and ∆Sync-D are the best on both subsets, on par with the reconstruction oracle. Since the oracle is driven by GT motion, matching it means almost no synchronization error originates in our motion generator.

SadTalker achieves the highest FPS and the lowest firstframe latency as a lightweight CNN-based method, at the cost of significantly inferior visual quality. Our method runs at 15.4 FPS, faster than all difusion-based baselines. While AvatarForcing exhibits lower first-frame latency, it underperforms our method across most quality metrics. The additional latency in our method mainly comes from the Q-Former fusion module and the generation of the first block of motion latent, and is further reduced in the audio-only setting where the fusion module is omitted. Despite this overhead, the firstframe latency is only around one second, making it suited for real-time interactive applications.

On visual quality, FID, FVD and E-FID are competitive but do not lead, most visibly on the Hallo3 subset. This is expected: our renderer is built on Stable Difusion(SD) backbone, weaker than the DiT-based backbones ofthe end-to-end baselines, so absolute visual fidelity is capped below theirs. The crucial comparison is not against those baselines but against the oracle, which uses the teacher renderer. Metrics of Ours are close to $G ( \mathbf { M } ^ { \mathrm { g t } } , \mathbf { I } _ { \mathrm { r e f } } )$ , and where we slightly exceed it is attributed to the oracle being trained on a diferent dataset. This proximity is the intended result showing the motion generator is not the bottleneck, and that our method raises the capability ceiling to the renderer’s, which the ablations in Sec. 4.4 confirm directly. A stronger renderer would lift visual fidelity without any change to the motion side.

<table><tr><td>Method</td><td>FID↓</td><td>FVD↓</td><td>CSIM↑</td><td>E-FID↓</td><td>∆Sync-C↓</td><td>∆Sync-D↓</td><td>FPS↑</td><td>Lat.(s)↓</td><td>Param.</td></tr><tr><td>SadTalker</td><td>101.6 / 83.9</td><td>346.3 / 519.1</td><td>0.869 / 0.883</td><td>0.372 / 0.425</td><td>3.39 / 0.43</td><td>3.57 / 0.42</td><td>23.8</td><td>0.21</td><td>0.2B</td></tr><tr><td>AniPortrait</td><td>43.5 / 38.5</td><td>216.5 / 385.8</td><td>0.904 / 0.901</td><td>0.108 / 0.258</td><td>1.56 / 1.82</td><td>1.66 / 1.73</td><td>0.83</td><td>242</td><td>1.3B</td></tr><tr><td>Hallo3</td><td>68.5 / 42.3</td><td>239.5 /395.0</td><td>0.839 / 0.773</td><td>0.095 / 0.189</td><td>4.28 / 0.76</td><td>3.01 / 0.26</td><td>0.31</td><td>157</td><td>8.9B</td></tr><tr><td>StableAvatar</td><td>46.1 / 45.8</td><td>514.8 / 662.3</td><td>0.801 / 0.771</td><td>0.294 / 0.261</td><td>1.09 / 2.35</td><td>0.56 / 2.80</td><td>0.86</td><td>230</td><td>1.7B</td></tr><tr><td>LiveAvatar</td><td>31.9 / 40.7</td><td>242.8 / 443.1</td><td>0.812 / 0.806</td><td>0.511 / 0.389</td><td>3.92 / 0.67</td><td>2.79 / 0.01</td><td>0.38</td><td>43.3</td><td>16.3B</td></tr><tr><td>AvatarForcing</td><td>46.7 / 42.8</td><td>351.7 / 340.5</td><td>0.768 / 0.728</td><td>0.240 / 0.192</td><td>2.66 / 1.34</td><td>2.13 / 1.77</td><td>14.8</td><td>0.61</td><td>1.4B</td></tr><tr><td>Motar(Ours)</td><td>45.0 / 50.5</td><td>185.9 / 424.2</td><td>0.894 / 0.831</td><td>0.109 / 0.280</td><td>0.08 / 0.06</td><td>0.42 / 0.37</td><td>15.4</td><td>1.39</td><td> $7 7 \mathrm { M } + 1 . 7 \mathrm { B }$ </td></tr><tr><td> $G ( \mathbf { M } ^ { \mathrm { g t } } , \mathbf { I } _ { \mathrm { r e f } } )$ </td><td>44.2/ 50.2</td><td>185.1 / 467.6</td><td>0.899 / 0.766</td><td>0.097 / 0.292</td><td>0.11 / 0.45</td><td>0.28 / 0.53</td><td>0.82</td><td>247</td><td>1.7B</td></tr></table>

Table 1: Quantitative comparison on MEAD and Hallo3 public subsets. Results are reported as MEAD / Hallo3. Best results are bolded and second-best results are underlined. The GT Sync-C is 1.68 in MEAD and 4.93 in Hallo3-data. The GT Sync-D is 12.22 in MEAD and 8.90 in Hallo3-data. ∆Sync-C and ∆Sync-D denote the absolute diference relative to GT. Param. denotes the parameter count of the backbone. Oracle $\dot { G } ( \mathbf { M } ^ { \mathrm { g t } } , \mathbf { I } _ { \mathrm { r e f } } )$ only serves as a reference and do not participate in comparison.

## 4.3 Qualitative Comparison

Figure 3 compares our method against StableAvatar, LiveAvatar and AvatarForcing, all built on the Wan backbone(Wan et al. 2025), with the articulated phoneme marked in red at each frame. Two diferences are consistent across examples.

First, lip synchronization. Our lip shapes track the highlighted phoneme frame by frame, whereas the baselines drift slightly. This matches the quantitative gap in ∆Sync, and follows from the hierarchical conditioning: the local branch ties each frame directly to a short audio window, giving the frame-level resolution that a single global pathway cannot.

Second, facial detail. The end-to-end baselines couple audio to the full video latent through cross-attention, so the conditioning signal interacts with every pixel with texture and lighting included, neither of which audio should govern. Trained at scale, the model regresses these entangled factors toward their conditional mean, blurring fine detail much as a mean-seeking objective would. Our method never exposes these factors to audio since conditioning acts only on the identity- and appearance-free motion latent, while all pixel-level detail is carried by the reference image through the frozen renderer. Fine texture is thus preserved almost verbatim and stays stable as the head moves.

These two properties are why our method matches the perceptual quality of these larger DiT-based baselines despite a weaker backbone and faster inference: by restricting audio to motion, it spends its capacity where the conditioning signa belongs, rather than defending appearance against it.

## 4.4 Ablation Study

In this section, we conduct ablation study on our proposed methods to demonstrate their efectiveness. More details and results are provided in the Supplementary Material.

Training recipe. We ablate the training recipe of $\mathcal { F } _ { \mathrm { m o t i o n } }$ on 256-frame rollouts, isolating each term of Eq. 8. Besides MSE, frame-wise cosine similarity (CosSim), and Fréchet motion distance (FMD) in latent space, we report two rolloutlevel ratios to ground truth, both ideally equal to 1: Std-R, the ratio of the generated sequence’s temporal standard deviation to GT, capturing motion amplitude; and VC-R(velocity consistency ratio), the same ratio on frame-to-frame diferences, capturing motion pace.

<table><tr><td>Variant</td><td>MSE↓</td><td>CosSim↑</td><td>FMD↓</td><td>Std-R↑</td><td>VC-R↑</td></tr><tr><td>TF(50 steps)</td><td>0.518</td><td>0.637</td><td>30.33</td><td>1.001</td><td>1.208</td></tr><tr><td>CM(1 step)</td><td>0.595</td><td>0.696</td><td>44.21</td><td>0.831</td><td>0.435</td></tr><tr><td> $\mathrm { S F w } / \mathcal { L } _ { \mathrm { m s e } }$ </td><td>0.259</td><td>0.840</td><td>29.93</td><td>0.494</td><td>0.433</td></tr><tr><td> $+ { \mathcal { L } } _ { \mathrm { a d v } }$ </td><td>0.263</td><td>0.839</td><td>18.21</td><td>0.590</td><td>0.542</td></tr><tr><td> $+ \mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { m o t i o n } }$ </td><td>0.298</td><td>0.826</td><td>17.07</td><td>0.704</td><td>0.644</td></tr></table>

Table 2: Ablation on the motion generator training recipe. We only compare Std-R and VC-R on SF variants as rollouts of TF variants collapse.

The two variants without self-forcing, the teacher-forced model (TF) and its consistency-distilled one-step version (CM) never see their own rollout in training, so error compounds at inference and MSE degrades. Their rollout-level ratios are unreliable as quality signals: TF’s Std-R of 1.001 and VC-R of 1.208 do not indicate faithful dynamics but severe drift, whose erratic motion happens to inflate these ratios; CM instead collapses toward static output. We therefore read Std-R and VC-R only among the self-forced variants.

Self-forcing with ${ \mathcal { L } } _ { \mathrm { m s e } }$ removes exposure bias and sharply improves MSE, CosSim and FMD, but its regression objective is mean-seeking: the rollout collapses toward static, under-animated motion, giving the lowest Std-R (0.494) and VC-R (0.433) among SF variants. Adding ${ \mathcal { L } } _ { \mathrm { a d v } }$ evaluates the rollout’s distribution rather than a per-frame target, restoring amplitude and pace and halving FMD. Adding L<sup>motion</sup><sub>DMD</sub> improves FMD, Std-R and VC-R further as it is the only term supervising motion by the video it renders into. Both distribution-matching terms slightly raise MSE and CosSim, but this is by design rather than a cost: MSE rewards regressing toward the per-frame mean, precisely the collapse we aim to escape, so recovering amplitude and diversity necessarily moves MSE away from its minimum. FMD, Std-R and VC-R, which measure distributional rather than point-wise fidelity, are the metrics aligned with our goal, and all three improve.

Figure 4 isolates exposure bias directly. TF models accumulate error rapidly along the rollout while SF variants grow far more slowly (a), confirming that self-forcing suppresses drift. Velocity is complementary (b): SF-MSE falls well below GT as its mean-seeking objective flattens dynamics, whereas ${ \mathcal { L } } _ { \mathrm { a d v } }$ and $\mathcal { L } _ { \mathrm { D M D } } ^ { \mathrm { m o t i o n } }$ each pull it back toward GT. The two TF models fail in opposite ways: the 50-step teacher overshoots GT velocity, its drift surfacing as erratic oversized motion; the 1-step distilled model collapses to near-static, as distillation strips the sampling stochasticity that gave the teacher its dynamics, leaving exposure bias flatten.

![](images/8ea119f15496debd384fa1687818ad920c5cb7c96a2bed50266281849883f5a4.jpg)

![](images/2415c4a6b50ba3b6fc7d4d18e3eb6a9f8ef9540953eb2d49b22956aaa25da655.jpg)  
Figure 3: Qualitative comparison with Wan-based end-to-end methods. For each result we show five frames with the spoken word beneath; the phoneme being articulated is highlighted in red (e.g. the “oo” in “look”), so that lip shape can be checked against the sound at each frame. Our method produces the tightest audio–lip alignment and the sharpest facial detail.

![](images/d07ea454bd45ac5d220296f25ce1a24aa973b8162697f38231e6c3e6c35a2d93.jpg)  
(a) Cumulative error.

![](images/e15f11b2b3140d38f0ae4a421fbc42415b8776b29108ed9629767d81f2068a50.jpg)  
(b) Inter-frame velocity.  
Figure 4: Exposure bias across variants. TF models drift over the rollout (a); SF variants curb this drift, and distributionmatching objectives restore GT-level velocity (b).

Hierarchical conditioning. We ablate the two conditioning pathways of $\mathcal { F } _ { \mathrm { m o t i o n } }$ in Table 3.

<table><tr><td>Variant</td><td>MSE↓</td><td>CosSim↑</td><td>FMD↓</td><td>Std-R↑</td><td>VC-R↑</td></tr><tr><td>w/o Global</td><td>0.314</td><td>0.807</td><td>35.08</td><td>0.483</td><td>0.443</td></tr><tr><td>w/o Local</td><td>0.588</td><td>0.725</td><td>50.44</td><td>0.578</td><td>0.282</td></tr><tr><td>Full</td><td>0.298</td><td>0.826</td><td>17.07</td><td>0.704</td><td>0.644</td></tr></table>

Table 3: Ablation on hierarchical conditioning.

The two pathways fail in near-orthogonal ways. Removing local audio conditioning nearly doubles MSE and sharply cuts velocity while barely touching amplitude, consistent with its role in frame-level lip alignment rather than motion scale. Removing the global Q-Former branch leaves MSE comparatively intact but collapses amplitude, consistent with its role as a sequence-level prior on motion intensity rather than fine-grained details. Both ablations more than double FMD, showing that either pathway alone is insuficient for distributional fidelity even when per-frame error looks acceptable, and the two branches are complementary.

## 5 Conclusion

We presented Motar, a streaming talking-head framework that fuses audio and text in an identity-disentangled motion space rather than the video latent, so that conditions control the video transitively and high fidelity no longer requires a large backbone. A hierarchical conditioning scheme routes the two modalities by their temporal granularity, using a global branch for sequence-level emotion and amplitude and a windowed local branch for frame-level lip articulation, so that a motion generator of only 77M parameters achieves precise, controllable synthesis. To make the decomposition stream, our decoupled self-forcing distillation resolves the exposure bias of both the autoregressive motion generator and the renderer under a single frozen teacher, lifting the fidelity ceiling from the motion generator onto the far stronger renderer. The two models run as parallel causal streams, reaching high throughput and low latency.

## References

Baevski, A.; Zhou, Y.; Mohamed, A.; and Auli, M. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33: 12449–12460.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Chen, B.; and Liu, H. 2025. DyStream: Streaming Dyadic Talking Heads Generation via Flow Matching-based Autoregressive Model. arXiv preprint arXiv:2512.24408.

Chen, B.; Liu, T.; Chen, Q.; Chen, X.; and Zheng, Z. 2025a. IMTalker: Eficient Audio-driven Talking Face Generation with Implicit Motion Transfer. arXiv preprint arXiv:2511.22167.

Chen, B.; Martí Monsó, D.; Du, Y.; Simchowitz, M.; Tedrake, R.; and Sitzmann, V. 2024. Difusion forcing: Next-token prediction meets full-sequence difusion. Advances in Neural Information Processing Systems, 37: 24081–24125.

Chen, Y.; Liang, S.; Zhou, Z.; Huang, Z.; Ma, Y.; Tang, J.; Lin, Q.; Zhou, Y.; and Lu, Q. 2025b. Hunyuanvideoavatar: High-fidelity audio-driven human animation for multiple characters. arXiv preprint arXiv:2505.20156.

Chen, Z.; Cao, J.; Chen, Z.; Li, Y.; and Ma, C. 2025c. Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, 2403–2410.

Chu, X.; Goswami, N.; Cui, Z.; Wang, H.; and Harada, T. 2025. Artalk: Speech-driven 3d head animation via autoregressive model. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, 1–9.

Chu, X.; and Harada, T. 2024. Generalizable and animatable gaussian head avatar. Advances in Neural Information Processing Systems, 37: 57642–57670.

Chung, J. S.; and Zisserman, A. 2016. Out of time: automated lip sync in the wild. In Asian conference on computer vision, 251–263. Springer.

Cui, J.; Li, H.; Zhan, Y.; Shang, H.; Cheng, K.; Ma, Y.; Mu, S.; Zhou, H.; Wang, J.; and Zhu, S. 2025. Hallo3: Highly dynamic and realistic portrait image animation with video difusion transformer. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 21086–21095.

Deng, J.; Guo, J.; Xue, N.; and Zafeiriou, S. 2019a. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4690–4699.

Deng, Y.; Yang, J.; Xu, S.; Chen, D.; Jia, Y.; and Tong, X. 2019b. Accurate 3D Face Reconstruction With Weakly-Supervised Learning: From Single Image to Image Set. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 285–295.

Goodfellow, I. J.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y. 2014. Generative adversarial nets. Advances in neural information processing systems, 27.

Guo, J.; Zhang, D.; Liu, X.; Zhong, Z.; Zhang, Y.; Wan, P.; and Zhang, D. 2024. Liveportrait: Eficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168.

Huang, X.; Li, Z.; He, G.; Zhou, M.; and Shechtman, E. 2026. Self forcing: Bridging the train-test gap in autoregressive video difusion. Advances in Neural Information Processing Systems, 38: 167283–167308.

Huang, Y.; Guo, H.; Wu, F.; Wang, W.; Zhang, S.; Huang, S.; Gan, Q.; Liu, L.; Zhao, S.; Chen, E.; et al. 2025. Live avatar: Streaming real-time audio-driven avatar generation with infinite length. arXiv preprint arXiv:2512.04677.

Ji, X.; Hu, X.; Xu, Z.; Zhu, J.; Lin, C.; He, Q.; Zhang, J.; Luo, D.; Chen, Y.; Lin, Q.; et al. 2025. Sonic: Shifting focus to global audio perception in portrait animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 193–203.

Ki, T.; Jang, S.; Jo, J.; Yoon, J.; and Hwang, S. J. 2026. Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 18074–18084.

Ki, T.; Min, D.; and Chae, G. 2025. Float: Generative motion latent flow matching for audio-driven talking portrait. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14699–14710.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, 19730–19742. PMLR.

Li, T.; Tian, Y.; Li, H.; Deng, M.; and He, K. 2024. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37: 56424–56445.

Li, T.; Zheng, R.; Yang, M.; Chen, J.; and Yang, M. 2025. Ditto: Motion-space difusion for controllable realtime talking head synthesis. In Proceedings of the 33rd ACM International Conference on Multimedia, 9704–9713.

Lin, G.; Jiang, J.; Yang, J.; Zheng, Z.; Liang, C.; Zhang, Y.; and Liu, J. 2025. Omnihuman-1: Rethinking the scalingup of one-stage conditioned human animation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 13847–13858.

Ling, J.; Wang, Y.; Xue, H.; Xie, R.; and Song, L. 2024. Posetalk: Text-and-audio-based pose control and motion refinement for one-shot talking head generation. arXivpreprint arXiv:2409.02657.

Liu, T.; Chen, F.; Fan, S.; Du, C.; Chen, Q.; Chen, X.; and Yu, K. 2024. Anitalker: Animate vivid and diverse talking faces through identity-decoupled facial motion encoding. In Proceedings of the 32nd ACM International Conference on Multimedia, 6696–6705.

Low, C.; and Wang, W. 2025. Talkingmachines: Real-time audio-driven facetime-style video via autoregressive difusion models. arXiv preprint arXiv:2506.03099.

Luo, S.; Tan, Y.; Huang, L.; Li, J.; and Zhao, H. 2023. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378.

Prajwal, K.; Mukhopadhyay, R.; Namboodiri, V. P.; and Jawahar, C. 2020. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the 28th ACM international conference on multimedia, 484–492.

Rafel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140): 1–67.

Su, J.; Ahmed, M.; Lu, Y.; Pan, S.; Bo, W.; and Liu, Y. 2024. RoFormer: Enhanced transformer with Rotary Position Embedding. Neurocomputing, 568: 127063.

Sun, Z.; Peng, Z.; Ma, Y.; Chen, Y.; Zhou, Z.; Zhou, Z.; Zhang, G.; Zhang, Y.; Zhou, Y.; Lu, Q.; et al. 2026. Streamavatar: Streaming difusion models for real-time interactive human avatars. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10887–10897.

Tian, L.; Wang, Q.; Zhang, B.; and Bo, L. 2024. Emo: Emote portrait alive generating expressive portrait videos with audio2video difusion model under weak conditions. In European Conference on Computer Vision, 244–260. Springer.

Touvron, H.; Cord, M.; Sablayrolles, A.; Synnaeve, G.; and Jégou, H. 2021. Going deeper with image transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 32–42.

Tu, S.; Pan, Y.; Huang, Y.; Han, X.; Xing, Z.; Dai, Q.; Luo, C.; Wu, Z.; and Jiang, Y.-G. 2025. StableAvatar: Infinite-Length Audio-Driven Avatar Video Generation. arXiv:2508.08248.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

Wang, H.; Weng, Y.; Du, J.; Xu, H.; Wu, X.; He, S.; Yin, B.; Liu, C.; Gao, J.; and Liu, Q. 2026. Read: Real-time and eficient asynchronous difusion for audio-driven talking head generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 9766–9774.

Wang, H.; Weng, Y.; Du, J.; Xu, H.; Wu, X.; He, S.; Yin, B.; Liu, C.; and Liu, Q. 2025. REST: Difusion-based Realtime End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation. arXiv preprint arXiv:2512.11229.

Wang, K.; Wu, Q.; Song, L.; Yang, Z.; Wu, W.; Qian, C.; He, R.; Qiao, Y.; and Loy, C. C. 2020. MEAD: A Large-scale Audio-visual Dataset for Emotional Talking-face Generation. In ECCV.

Wang, S.; Li, L.; Ding, Y.; Fan, C.; and Yu, X. 2021. Audio2head: Audio-driven one-shot talking-head generation with natural head motion. arXiv preprint arXiv:2107.09293.

Wang, Y.; Yang, D.; Bremond, F.; and Dantcheva, A. 2024. Lia: Latent image animator. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12): 10829–10844.

Wei, H.; Yang, Z.; and Wang, Z. 2024. AniPortrait: Audio-Driven Synthesis of Photorealistic Portrait Animation. arXiv:2403.17694.

Xu, M.; Li, H.; Su, Q.; Shang, H.; Zhang, L.; Liu, C.; Wang, J.; Yao, Y.; and Zhu, S. 2024a. Hallo: Hierarchical audiodriven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801.

Xu, S.; Chen, G.; Guo, Y.-X.; Yang, J.; Li, C.; Zang, Z.; Zhang, Y.; Tong, X.; and Guo, B. 2024b. Vasa-1: Lifelike audio-driven talking faces generated in real time. Advances in Neural Information Processing Systems, 37: 660–684.

Yang, S.; Huang, W.; Chu, R.; Xiao, Y.; Zhao, Y.; Wang, X.; Li, M.; Xie, E.; Chen, Y.; Lu, Y.; et al. 2025a. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622.

Yang, Z.; Teng, J.; Zheng, W.; Ding, M.; Huang, S.; Xu, J.; Yang, Y.; Hong, W.; Zhang, X.; Feng, G.; et al. 2025b. Cogvideox: Text-to-video difusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, 83048–83077.

Yin, T.; Gharbi, M.; Zhang, R.; Shechtman, E.; Durand, F.; Freeman, W. T.; and Park, T. 2024. One-step difusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6613–6623.

Yin, T.; Zhang, Q.; Zhang, R.; Freeman, W. T.; Durand, F.; Shechtman, E.; and Huang, X. 2025. From slow bidirectional to fast autoregressive video difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22963–22974.

Zhang, W.; Cun, X.; Wang, X.; Zhang, Y.; Shen, X.; Guo, Y.; Shan, Y.; and Wang, F. 2023. Sadtalker: Learning realistic 3d motion coeficients for stylized audio-driven single image talking face animation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 8652– 8661.

Zhao, X.; Xu, H.; Song, G.; Xie, Y.; Zhang, C.; Li, X.; Luo, L.; Suo, J.; and Liu, Y. 2025. X-nemo: Expressive neural motion reenactment via disentangled latent attention. arXiv preprint arXiv:2507.23143.

Zheng, L.; Zhang, Y.; Guo, H.; Pan, J.; Tan, Z.; Lu, J.; Tang, C.; An, B.; and Yan, S. 2024. Memo: Memory-guided difusion for expressive talking video generation. arXiv preprint arXiv:2412.04448.

Zhu, H.; Zhao, M.; He, G.; Su, H.; Li, C.; and Zhu, J. 2026. Causal Forcing: Autoregressive Difusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation. arXiv preprint arXiv:2602.02214.