# DreamX-Creator 1.0: Democratizing Native Audio-Video Generation at 2K Resolution

Jiashu Zhu<sup>∗</sup>, Yanhao Zheng<sup>∗</sup>, Ruitian Tian<sup>∗</sup>, Rujing Dang<sup>∗</sup>, Shen Zhang, Bingze Song, Jiachen Lei, Ruimin Lin, Jiahong Wu<sup>†</sup>, Xiangxiang Chu

DreamX Team, Alibaba Group <sup>∗</sup>Equal contribution. <sup>†</sup>Project Lead.

Recent video generators often omit audio or synthesize it in a separate stage, limiting reciprocal modeling of visual dynamics and acoustic events. We present DreamX-Creator 1.0, a compact native joint audio-video generation system centered on a 7B generator. Conditioned on a first frame and a text prompt, the generator jointly denoises modality-specialized audio and video streams. The streams are processed independently in the first half of the network and coupled in the latter half through Gated Cross-Modal Attention, whose token- and head-wise output gates modulate each active cross-modal attention-head output. A unified Audio-Video Data System constructs and filters temporally coherent clips, produces structured multimodal annotations, and organizes clips into capability-oriented data pools. Progressive Joint Training comprises two audio-video pre-training stages followed by High-Quality Finetuning. Audio-Video Reinforcement Learning further post-trains the generator with Modality-Aware Multimodal Feedback that routes video-, audio-, and cross-modal feedback to the corresponding streams. For high-resolution output, our Autoregressive 1-Step 2K Refinement pipeline adapts a bidirectional multi-step teacher into an autoregressive multi-step refiner and distills it into a student requiring one denoising evaluation per temporal chunk. Overall, DreamX-Creator 1.0 achieves native, synchronized audio-video generation with performance competitive with state-of-the-art open-source systems. By releasing our compact 7B generator and 2K Refiner, we seek to democratize native audio-video generation and provide an accessible foundation for future research in unified audio-video generative modeling.

![](images/d48bea482b073382c4113c165e7342824fe3c694379aa434ac86daa7eb1ed31f.jpg)  
Figure 1 DreamX-Creator 1.0. A native joint generator couples visual motion, speech, sound efects, and ambience, followed by multimodal post-training and a 2K-oriented video refinement stage.

## Contents

1 Introduction 3   
2 Audio-Video Data System   
2.2 Data Filtering .   
2.3 Data Annotation   
2.4 Capability Taxonomy 6   
3 Native Joint Audio-Video Generation 7   
3.1 Architecture Overview . 7   
3.2 Gated Cross-Modal Attention 7   
3.3 Progressive Joint Training 9   
3.4 Optimization Details 9   
4 Audio-Video Reinforcement Learning 10   
4.1 Overview 10   
4.2 Modality-Aware Multimodal Feedback 10   
4.3 Training 11   
5 Autoregressive 1-Step 2K Refinement 12   
5.1 Bidirectional Refiner 12   
5.2 Autoregressive Refiner 12   
5.3 DMD Distillation 13   
6 Evaluation 13   
6.1 Settings 13   
6.2 Quantitative Evaluation 13   
6.3 Refiner Comparison 15   
6.4 User Study 15   
7 Related Work 16   
7.1 Audio-Video Generation Models . . . . 16   
7.2 From Single-Modality to Joint Audio-Video Preference Alignment . 16   
7.3 Video Refinement and Few-Step Distillation . 17   
8 Conclusion 17

## 1 Introduction

Generative video has advanced rapidly in visual fidelity, motion quality, and duration, but audio is still often omitted (Team et al., 2026a,b) or synthesized in a separate stage. Directional pipelines have achieved strong video-to-audio generation (Luo et al., 2023; Zhang et al., 2024; Cheng et al., 2025) and audio-driven human animation (Tian et al., 2024; Lin et al., 2025; Gao et al., 2025). However, these formulations treat one modality as a fixed condition and the other as the generated output, limiting reciprocal interaction when visual dynamics and acoustic events are jointly determined. Examples include speech and mouth motion, visible impacts and their associated sounds, and scene-level relationships among ambience, music, and camera motion.

Joint audio-video generation instead models both modalities inside the generative process. Early work coupled audio and video denoisers within joint difusion models (Ruan et al., 2023; Xing et al., 2024; Wang et al., 2024; Liu et al., 2024). More recent native generators retain modality-specialized representations while exchanging information through dual-stream backbones, expert-based architectures, or unified multimodal transformers (Low et al., 2025; HaCohen et al., 2026; Wang et al., 2025a; Ji et al., 2026; SII-OpenMOSS Team, 2026; Cheng et al., 2026; Liu et al., 2026; Zhang et al., 2026b). Together, these systems show that synchronized sound and video need not be produced as a post-hoc cascade.

This progress exposes four unresolved challenges. First, raw audio-video data require reliable clip segmentation, modality-specific quality filtering, synchronization assessment, and joint multimodal annotation. Shot transitions and unrelated background audio can otherwise introduce spurious cross-modal correlations. Second, cross-modal interaction must be introduced without overwhelming modality-specific representations. Its required strength varies across layers, attention heads, tokens, and samples, while diferent content exhibits diferent directional dependencies between audio and video. Third, likelihood- or flow-based training does not directly optimize perceptual quality, prompt adherence, cross-modal semantic consistency, or fine-grained audio-video synchronization. Fourth, joint latent generation and 2K refinement impose diferent computationa requirements: the former benefits from a practical latent resolution, whereas the latter must recover spatial detail while maintaining the generated content, motion, and audio-synchronized timing.

Model scale and availability remain practical barriers to research on native audio-video generation. Many existing systems use tens of billions of parameters, provide only hosted access, or are dificult to reproduce. We therefore introduce DreamX-Creator 1.0, a compact native joint audio-video system centered on a 7B generator and accompanied by an autoregressive 1-step 2K Refiner. We publicly release both components to support reproducible evaluation and downstream adaptation.

The comparison in table 1 isolates three system-level properties relevant to research accessibility and deployment scope: downloadable model weights, native joint audio-video generation, and an oficially supported output path at 2K resolution or above. Among the listed systems with a disclosed total backbone size, DreamX-Creator 1.0 is the smallest model that combines all three properties.

Table 1 Capability comparison among representative contemporary video generators.

<table><tr><td>Model</td><td>Backbone params. (total / active)</td><td>Weight availability</td><td>Generation modality</td><td>≥2K availability</td></tr><tr><td>LTX-2.3 (Lightricks, 2026a,b)</td><td>22B</td><td>Open</td><td>Joint A/V</td><td>Local (up to 4K)</td></tr><tr><td>MOVA (SII-OpenMOSS Team, 2026)</td><td>32B / 18B</td><td>Open</td><td>Joint A/V</td><td>Not available</td></tr><tr><td>LingBot-Video (LingBot Team, 2026)</td><td>30B/3B</td><td>Open</td><td>Video only</td><td>Not available</td></tr><tr><td>MAGI-2 Preview (Sand AI, 2026)</td><td>114B / 6B</td><td>Open</td><td>Joint A/V</td><td>Not available</td></tr><tr><td>MiniMax H3 (MiniMax, 2026)</td><td>33B</td><td>Base only</td><td>Joint A/V</td><td>Online only</td></tr><tr><td>Seedance 2.5 (ByteDance Seed, 2026; Dreamina, 2026)</td><td>N/D</td><td>Not released</td><td>Joint A/V</td><td>Online only</td></tr><tr><td>Kling AI 3.0 (Kuaishou Technology, 2026a,b)</td><td>N/D</td><td>Not released</td><td>Joint A/V</td><td>Online only</td></tr><tr><td>DreamX-Creator 1.0</td><td>7B</td><td>Open</td><td>Joint A/V</td><td>Local</td></tr></table>

Availability is based on publicly documented capabilities as of August 27, 2026. “Open” denotes downloadable model parameters and does not imply an OSI-approved license; “Base only” indicates that only the base-generation checkpoint is downloadable. In the ≥2K column, “Local” denotes an oficially supported local generation, refinement, or upscaling path; “Online only” denotes availability exclusively through an oficial web service, cloud platform, or API; and “Not available” indicates that no oficial support was identified. MiniMax H3 releases H3-Base for local 768p generation, whereas H3-Context-IR and H3-Regenerate-2K remain online-only. LTX-2.3 releases both its generation checkpoints and x2 latent spatial upscaler weights, with oficial support for local output up to 4K. Parameter counts refer to the generation backbone; active parameters are additionally reported for sparse models. N/D: not disclosed; A/V: audio-video.

At the core of DreamX-Creator 1.0 is a native joint generator with modality-specialized audio and video streams. The first half of the network processes the two streams independently, while the latter half introduces paired audio-to-video (A2V) and video-to-audio (V2A) paths through Gated Cross-Modal Attention. The paths operate on positions mapped to a shared temporal coordinate system, with temporal rotary position encoding applied to their cross-modal queries and keys. Following the output-gating design studied in gated attention (Qiu et al., 2025), each enabled path uses a token- and head-wise sigmoid output gate conditioned on both the target hidden state and the cross-modal attention output. A per-sample direction mask selects the active path or paths, whereas the gate scales each attention-head output before head concatenation, output projection, and residual addition.

The complete DreamX-Creator 1.0 system is supported by a unified Audio-Video Data System. The base generator is trained with Progressive Joint Training, comprising two audio-video pre-training stages followed by High-Quality Finetuning. The internal A2V, V2A, and Joint conditioning configurations are mixed throughout training and do not define separate inference tasks. Audio-Video Reinforcement Learning then post-trains the generator using Modality-Aware Multimodal Feedback that routes video-, audio-, and crossmodal advantages to the corresponding streams. Finally, 2K Refiner adapts a bidirectional multi-step teacher into an autoregressive multi-step refiner and distills it into an autoregressive 1-step student. At inference, it enhances the low-resolution video to 2K with one denoising evaluation per temporal chunk while leaving the audio stream unchanged.

![](images/690a179d32c072e8cbda1b5a1bb51b659eb769e80df68ffd015bdd22eb5886dc.jpg)  
Figure 2 The DreamX-Creator 1.0 pipeline, comprising native joint audio-video generation, Audio-Video Reinforcement Learning post-training, and Autoregressive 1-Step 2K Refinement.

## Our main contributions are:

• Audio-Video Data System. We build a unified pipeline for clip construction and filtering, structured multimodal annotation, and capability-oriented data organization to support targeted audio-video training.

• Native Joint Audio-Video Generation. We develop a native joint generator with Gated Cross-Modal Attention and Progressive Joint Training, followed by Audio-Video Reinforcement Learning with Modality-Aware Multimodal Feedback.

• Autoregressive 1-Step 2K Refinement. We adapt a bidirectional multi-step teacher into an autoregressive multi-step refiner and distill it into a 1-step student that requires one denoising evaluation per temporal chunk.

• A Compact 7B Open-Weight Model. We publicly release our 7B native joint audio-video generator and 2K Refiner. The generator is the smallest open-weight native joint model by disclosed total backbone size in table 1 and achieves top-tier benchmark performance.

## 2 Audio-Video Data System

Training a general-purpose native audio-video generation model requires a corpus with high perceptual quality, reliable cross-modal correspondence, and diverse audio-visual interactions. However, raw videos collected from heterogeneous sources are often noisy, weakly aligned, and highly imbalanced in their audio-visual content. We therefore develop a unified data system that transforms raw videos into training-ready clips through staged filtering for quality and alignment, followed by structured multimodal annotation. We further organize the curated clips according to their cross-modal dependency patterns, allowing diferent types of audio-visual data to provide supervision tailored to diferent training objectives. We describe each component of the system below.

![](images/cbe6cdaa903c345b7b823247fc4c3463c5cffbbc8cd54194814c3dcc3abba1df.jpg)  
Figure 3 Overview of the data system for DreamX-Creator 1.0, including heterogeneous data collection, clip construction and filtering, structured multimodal annotation, and capability-oriented data organization.

## 2.1 Data Collection

We construct our audio-visual corpus by integrating multiple public datasets, including Koala-36M (Wang et al., 2025c), VGGSound (Chen et al., 2020), AudioSet (Gemmeke et al., 2017), OpenHumanVid (Li et al., 2024), SpeakerVid-5M (Zhang et al., 2026c), Action-100M (Chen et al., 2026), and Talker-T2AV (Ye et al., 2026), together with internally collected data. These heterogeneous sources provide complementary coverage across diferent content domains and audio-visual interactions.

## 2.2 Data Filtering

We first preprocess the collected videos into valid audio-visual clips by segmenting them at scene boundaries with PySceneDetect (Castellano), removing near-silent segments based on audio RMS energy, and discarding invalid or low-resolution samples. As scene-boundary localization may be imperfect, we further trim three frames from both ends of each segment to reduce boundary artifacts and potential cross-shot contamination. We then apply a staged filtering pipeline to assess visual quality, motion, audio quality, and cross-modal synchronization.

Multidimensional quality assessment. We use Q-Align to evaluate visual perceptual quality (Wu et al., 2023), UniMatch-based optical flow to estimate motion magnitude (Xu et al., 2023), and Audiobox Aesthetics to assess audio quality (Tjandra et al., 2025).

Table 2 Capability-oriented data pools for audio-video generation.
<table><tr><td>Pool</td><td>Primary content</td></tr><tr><td>Speech and dialogue</td><td>Visible speech, conversations, Audio provides fine-grained cues for fa- A2V and joint and speaker interactions cial and articulatory dynamics</td></tr><tr><td>Actions and Foley</td><td>Human actions, object interac- Visible events provide strong cues for V2A and joint tions, impacts, tools, and vehi- temporally aligned sound generation cles</td></tr><tr><td>Ambience and music Environmental</td><td>ambience, Visual context provides scene-level cues V2A and joint crowds, weather, and back- for sustained acoustic content ground music</td></tr><tr><td>General cinematic</td><td>Mixed scenes without a single Mixed cross-modal dependencies pro- A2V, V2A, and joint dominant audio-visual interac- vide broad supervision for general audio- visual modeling</td></tr></table>

Audio-visual alignment assessment. We assess temporal alignment at both general and speech-specific levels. Synchformer is used to evaluate synchronization between temporally correlated audio and visual events (Iashin et al., 2024), while SyncNet is additionally applied to clips with visible speech for fine-grained lip-audio assessment (Chung and Zisserman, 2016). Accordingly, general audio-visual clips are filtered based on Synchformer, whereas visible-speech clips are further required to satisfy the SyncNet-based lip-sync criterion.

## 2.3 Data Annotation

Following filtering, we construct structured multimodal annotations that jointly capture visual content, acoustic events, cross-modal relationships, and spoken text. Our pipeline consists of joint audio-visual understanding followed by caption refinement and fusion.

We first employ Qwen3-Omni-30B-A3B-Instruct to jointly analyze the video and audio streams (Xu et al., 2025). Unlike pipelines that caption the two modalities independently and merge them afterward (SII-OpenMOSS Team, 2026; Huang et al., 2025a), joint multimodal perception allows visual and acoustic evidence to constrain each other, helping reduce hallucinated or cross-modally inconsistent descriptions. The resulting annotation describes subjects, actions, scene context, camera view, speech, music, sound events, and ambience, while organizing audio-visual events in temporal order and distinguishing visually grounded sounds from of-screen audio.

To improve the accuracy of spoken content, we additionally use Qwen3-ASR-1.7B to transcribe speech (Shi et al., 2026). The multimodal annotation and ASR transcript are then jointly provided to Qwen3.6-27B (Qwen Team, 2026), which consolidates them into a coherent and fluent caption while preserving the temporal structure and spoken content. The resulting caption is used as the conditioning signal for audio-visual generation.

## 2.4 Capability Taxonomy

As shown in Table 2, we organize the annotated clips into four capability-oriented pools according to their content characteristics and cross-modal dependency patterns. Since diferent audio-visual content provides distinct forms of cross-modal supervision, we group clips by their dominant supervision signals to support targeted learning of different generation capabilities. Specifically, we use Qwen3.6-27B (Qwen Team, 2026) to infer the dominant cross-modal supervision pattern from each structured audio-visual annotation and assign the clip to the corresponding capability-oriented pool.

This capability-oriented organization enables us to construct data according to the supervision required by diferent generation objectives, rather than sampling uniformly from a heterogeneous corpus. By concentrating clips with strong and relevant cross-modal signals, the resulting pools support more targeted capability learning while preserving the diversity needed for general-purpose audio-visual generation.

![](images/c6627f7aa279205961bad5eb2ad13355b38fef3e0127428bc5c2d7fea3f6b427.jpg)  
Figure 4 Content distribution of the training dataset.

In addition, we analyze the content distribution of the training data, as shown in Figure 4. Speech accounts for 45.0% of the data, while event sounds constitute another 33.4%, including sounds associated with human actions, object interactions, transportation, and other physical events. The remaining samples cover music, natural sounds, and mixed content, providing broad supervision beyond speech-centric scenarios for general audio-visual generation.

## 3 Native Joint Audio-Video Generation

## 3.1 Architecture Overview

Given a first frame and a text prompt, DreamX-Creator 1.0 jointly denoises video and audio latent streams. Each stream retains its own token rate, positional encoding, and transformer backbone, while receiving text conditioning from a shared text encoder through a modality-specific conditioning path.

The first half of the network processes the two streams independently. The latter half adds paired audioto-video (A2V) and video-to-audio (V2A) cross-attention paths. A2V uses video queries with audio keys and values; V2A reverses these roles. When both paths are active, they read the same pre-fusion states and compute their residual updates in parallel. A direction mask selects one path for a directional mode or both paths for Joint mode. A2V and V2A are therefore internal cross-modal paths rather than standalone inference tasks.

Because the streams have diferent token rates, we map their positions to a shared temporal coordinate system and apply temporal rotary position encoding to cross-modal queries and keys. This provides time-aware attention without resampling either latent sequence or replacing its modality-specific positional encoding (Low et al., 2025; HaCohen et al., 2026; SII-OpenMOSS Team, 2026; Ji et al., 2026).

Training modes. Each sample is assigned an A2V, V2A, or Joint mode. The mode specifies the active cross-modal path, relative corruption ordering, and stop-gradient rule. Let $m _ { a  v } , m _ { v  a } \in \{ 0 , 1 \}$ denote the direction masks:

$$
\begin{array} { r l } { \mathrm { A 2 V : } } & { \sigma _ { v } > \sigma _ { a } , \quad ( m _ { a  v } , m _ { v  a } ) = ( 1 , 0 ) , } \\ { \mathrm { V 2 A : } } & { \sigma _ { a } > \sigma _ { v } , ( m _ { a  v } , m _ { v  a } ) = ( 0 , 1 ) , } \\ { \mathrm { J o i n t : } } & { \sigma _ { v } = \sigma _ { a } , ( m _ { a  v } , m _ { v  a } ) = ( 1 , 1 ) . } \end{array}\tag{1}
$$

A larger σ denotes stronger corruption, so A2V and V2A use a noisier target stream and a cleaner conditioning stream. Both streams retain their flow-matching losses. Let $h ^ { v }$ and $h ^ { a }$ denote the pre-fusion hidden states. The cross-modal updates are

$$
\begin{array} { r l } & { \Delta h ^ { v } = m _ { a  v } \mathrm { ~ A t t n } \Big ( Q ( h ^ { v } ) , K ( \widehat { h } ^ { a } ) , V ( \widehat { h } ^ { a } ) \Big ) , } \\ & { \Delta h ^ { a } = m _ { v  a } \mathrm { ~ A t t n } \Big ( Q ( h ^ { a } ) , K ( \widehat { h } ^ { v } ) , V ( \widehat { h } ^ { v } ) \Big ) , } \end{array}\tag{2}
$$

For A2V, ${ \widehat h } ^ { a } = \operatorname { s g } ( h ^ { a } )$ ; for V2A, ${ \widehat h } ^ { v } = \operatorname { s g } ( h ^ { v } )$ ; Joint mode uses the original hidden states in both paths. Stop-gradient is applied before the key and value projections. It blocks the target-stream loss from updating the conditioning backbone through cross-modal attention while leaving the attention parameters trainable; the conditioning backbone is still trained by its own flow-matching loss.

## 3.2 Gated Cross-Modal Attention

Within each latter-half block, cross-modal attention follows the modality-specific self-attention and text cross-attention transformations.

More specifically, let $X \in \mathbb { R } ^ { L _ { x } \times d _ { x } }$ denote the target stream and $Y \in \mathbb { R } ^ { L _ { y } \times d _ { y } }$ the source stream of a generic cross-modal path. Each direction uses its own projections to map the two streams to a common multi-head attention space:

![](images/6936f3fc8fcf4bc1fdca3b45c2a881b9d142c87e99e524ac279aceaae25c97e9.jpg)

![](images/32429738c8f4d1c85cef67bea7e2a4390a735fb4bdaa3d41530d0ba270bcf378.jpg)  
Figure 5 Architecture of the joint audio-video generator. The two streams are processed independently in the first half and interact through gated A2V and V2A cross-attention in the latter half.

$$
\begin{array} { l l } { \bar { X } = \mathrm { L N } _ { x } ( X ) , } & { \bar { Y } = \mathrm { L N } _ { y } ( Y ) , } \\ { Q = \mathrm { R M S N o r m } ( W _ { Q } \bar { X } ) , } & { K = \mathrm { R M S N o r m } ( W _ { K } \bar { Y } ) , ~ V = W _ { V } \bar { Y } . } \end{array}\tag{3}
$$

After splitting Q, K, and V into attention heads, temporal RoPE is applied using coordinates $\tau _ { x }$ and $\tau _ { y }$ in the shared temporal coordinate system. For head h, the cross-modal attention output is

$$
\begin{array} { l l } { { \widetilde { Q } _ { h } = \mathrm { R o P E } _ { t } ( Q _ { h } , \tau _ { x } ) , } } & { { \widetilde { K } _ { h } = \mathrm { R o P E } _ { t } ( K _ { h } , \tau _ { y } ) , } } \\ { { \ A _ { h } = \mathrm { s o f t m a x } \left( \frac { \widetilde { Q } _ { h } \widetilde { K } _ { h } ^ { \top } } { \sqrt { d _ { h } } } + M _ { y } \right) , } } & { { C _ { h } = A _ { h } V _ { h } , } } \end{array}\tag{4}
$$

where $M _ { y }$ masks padded positions beyond the valid source-stream length. For $\operatorname { A 2 V } , ( X , Y ) = ( h ^ { v } , h ^ { a } )$ ; for V2A, $( \bar { X , } \bar { Y } ) = ( h ^ { a } , h ^ { v } )$ .

Following the output-gating placement studied in gated attention (Qiu et al., 2025), we apply a token- and head-wise sigmoid output gate after scaled dot-product attention and before the output projection. Unlike a query-only gate, our gate depends on both the target token $x _ { i }$ and its cross-modal attention output $C _ { i , h } \colon$

$$
\begin{array} { r l } & { g _ { i , h } = \mathrm { s i g m o i d } \big ( [ W _ { x } \mathrm { L N } ( x _ { i } ) ] _ { h } + \mathbf { w } _ { c } ^ { \top } \mathrm { L N } ( C _ { i , h } ) + b _ { h } \big ) , } \\ & { \Delta x _ { i } = W _ { o } ( \mathrm { C o n c a t } _ { h } [ g _ { i , h } \odot C _ { i , h } ] ) , } \\ & { \quad x _ { i } ^ { \prime } = x _ { i } + \Delta x _ { i } . } \end{array}\tag{5}
$$

Here, $W _ { x }$ produces one gate logit per head from the target hidden state, ${ \bf w } _ { c }$ maps the normalized head-wise attention output to a scalar, and $b _ { h }$ is a head-specific bias. Thus $g _ { i , h } \in ( 0 , 1 )$ scales each attention-head output before head concatenation, output projection, and residual addition.

The output gate and direction mask act at diferent levels: the gate scales each active attention-head output, while the direction mask selects whether the complete A2V or V2A residual is added. The gate is therefore an output modulation mechanism rather than token routing or path selection.

## 3.3 Progressive Joint Training

Training consists of two audio-video pre-training stages followed by High-Quality Finetuning. Across all stages, we mix the internal A2V, V2A, and Joint conditioning configurations defined above. Pre-training uses a fixed mixture, while High-Quality Finetuning adjusts their relative emphasis according to the predominant cross-modal dependency of the training content. These configurations determine the conditioning paths used during training and do not define separate inference tasks.

Stage 1: LoRA-Based AV Pre-training. We initialize from modality-specific video and audio backbones and train on broad-coverage paired data. The first half of both backbones remains frozen, rank-256 LoRA adapters are applied to the latter half, and the cross-modal attention modules and output gates are optimized directly.

Stage 2: Full-Parameter AV Pre-training. The Stage 1 LoRA weights are merged into their respective backbones. Training then continues on broad-coverage paired data with both difusion-transformer backbones and all cross-modal modules jointly optimized.

Stage 3: High-Quality Finetuning. We construct a curated finetuning subset by filtering candidate clips using audio-video synchronization metrics, modality-specific aesthetic scores for both video and audio, spatial resolution, and clip duration. To obtain temporally coherent clips, we further apply OmniShotCut (Wang et al., 2026) for fine-grained shot-boundary detection and transition labeling, and discard candidate segments that contain or span a detected transition. We then finetune the full-parameter model on the retained pairs. The first finetuning run inherits the group-wise learning rates from Stage 2; subsequent runs progressively reduce them.

## 3.4 Optimization Details

We optimize the video and audio streams using flow matching (Lipman et al., 2023). For modality $m \in \{ v , a \}$ let $z _ { 0 } ^ { m }$ denote a clean latent, $\epsilon ^ { m } \sim \mathcal { N } ( 0 , I )$ independently sampled Gaussian noise, and $\sigma _ { m }$ its modality-specific noise level. The interpolated latent and target velocity are

$$
z _ { \sigma _ { m } } ^ { m } = ( 1 - \sigma _ { m } ) z _ { 0 } ^ { m } + \sigma _ { m } \epsilon ^ { m } , \qquad u _ { m } ^ { * } = \epsilon ^ { m } - z _ { 0 } ^ { m } .\tag{6}
$$

Let $N _ { m }$ and $d _ { m }$ denote the number of valid latent tokens and their feature dimension for modality m, respectively. We define the token- and feature-normalized flow-matching objective as

$$
\mathcal { L } _ { \mathrm { F M } } ^ { m } = \frac { 1 } { N _ { m } d _ { m } } \sum _ { j = 1 } ^ { N _ { m } } \left. \widehat { u } _ { \boldsymbol { \theta } , j } ^ { m } - \boldsymbol { u } _ { m , j } ^ { * } \right. _ { 2 } ^ { 2 } , \qquad \mathcal { L } _ { \mathrm { A V } } = \lambda _ { v } \mathcal { L } _ { \mathrm { F M } } ^ { v } + \lambda _ { a } \mathcal { L } _ { \mathrm { F M } } ^ { a } ,\tag{7}
$$

where $\lambda _ { v }$ and $\lambda _ { a }$ balance the two modalities. The two streams use independently sampled Gaussian noise tensors. A base timestep is drawn from a discrete 1,000-timestep flow schedule with a shift factor of 5.0, and the modality-specific noise levels follow the directional ordering in equation (1). The two audio-video pre-training stages use $\lambda _ { v } = \lambda _ { a } = 0 . 5$ , whereas High-Quality Finetuning uses $\lambda _ { v } = 0 . 5$ and $\lambda _ { a } = 0 . 1$

Optimizer. In Stage 1, the latter-half LoRA parameters use a learning rate of $1 \times 1 0 ^ { - 4 }$ , while the cross-modal attention and output-gate parameters use $2 \times 1 0 ^ { - 5 }$ . LoRA uses rank 256, a scaling factor of 128, and zero dropout. After merging the adapters, Stage 2 updates the video and audio backbone parameters with $1 \times 1 0 ^ { - 5 }$ and the cross-modal modules with $2 \times 1 0 ^ { - 5 }$ . High-Quality Finetuning begins with the same group-wise rates and progressively reduces them across successive runs.

All stages use AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 )$ and $\epsilon = 1 0 ^ { - 1 0 }$ . Weight decay is zero for the Stage 1 LoRA and cross-modal parameter groups. As specified in our training recipe, Stage 2 uses $1 0 ^ { - 5 }$ for the backbone parameters and zero for the cross-modal modules, while High-Quality Finetuning uses zero weight decay for both groups. The two pre-training stages use a 200-step linear warm-up followed by constant group-wise learning rates. Training uses bfloat16 mixed precision and clips the gradient norm at 1.0. The complete training pipeline uses a comparatively modest GPU-day budget.

## 4 Audio-Video Reinforcement Learning

## 4.1 Overview

The native joint generator described in Section 3 is trained with audio and video flow-matching objectives. This training enables the model to capture the joint audio-video distribution and establish bidirectional interaction between the two modality streams. However, minimizing these flow-matching objectives does not directly optimize the properties most relevant to human perception, such as visual and acoustic fidelity, prompt adherence, cross-modal semantic consistency, and precise temporal correspondence between visible events and their associated sounds.

We therefore introduce an audio-visual reinforcement-learning stage to post-train DreamX-Creator 1.0 with multimodal feedback. This stage improves visual and audio quality, semantic consistency, and temporal synchronization while preserving the capabilities and diversity of the base model. Both streams remain within the joint generator, improving their coordination without an additional cascaded generation stage.

A central challenge is that an audio-video sample is evaluated along multiple dimensions that may not improve together. A candidate can contain high-quality video but weak audio, or plausible content in both modalities but incorrect event timing. Consequently, we retain decomposed feedback signals rather than immediately reducing all criteria to a single global score. This design follows the modality-aware optimization principle explored by OmniNFT (Zhang et al., 2026a), while adapting it to the bidirectional architecture and context-aware interaction modules of DreamX-Creator 1.0.

## 4.2 Modality-Aware Multimodal Feedback

A single global reward is insuficient for joint audio-video post-training because the quality of the two modalities may vary independently. A candidate with strong visual quality may contain weak audio, while a perceptually plausible pair may still exhibit incorrect semantic or temporal correspondence. We therefore decompose the feedback into video-specific, audio-specific, and cross-modal components. Let $A _ { v } , A _ { a }$ , and $A _ { a v }$ denote their normalized advantages. The supervision routed to the two generation streams is

$$
\widetilde { A } _ { v } = A _ { v } + A _ { a v } , \qquad \widetilde { A } _ { a } = A _ { a } + A _ { a v } .\tag{8}
$$

The modality-specific terms improve the corresponding video or audio stream, whereas the shared cross-modal term jointly supervises both streams and their bidirectional interaction modules. This separation prevents an improvement in one modality from masking degradation in the other and preserves synchronization as a shared optimization objective.

Audio-visual synchronization is mainly determined by a small number of regions, such as visible mouths and sound-producing objects. We use the video-to-audio responses from the selected interaction blocks to estimate the relevance of each video token. To account for response-scale diferences across frames, the scores are normalized within each frame and converted into token weights:

$$
w _ { i } ^ { ( e ) } = 1 + \alpha _ { e } \mathrm { ~ s i g m o i d } \left( \frac { s _ { i } - \mu _ { f ( i ) } } { \sigma _ { f ( i ) } + \epsilon } \right) .\tag{9}
$$

Here, $s _ { i }$ is the cross-modal relevance score of video token i, and $f ( i )$ denotes the frame containing that token. The terms $\mu _ { f ( i ) }$ and $\sigma _ { f ( i ) }$ are the mean and standard deviation of the relevance scores within the same frame, while ϵ ensures numerical stability. The resulting weight $w _ { i } ^ { ( e ) }$ determines the contribution of token i at training step e. The coeficient $\alpha _ { e }$ is gradually increased from zero to a predefined maximum during warmup. Thus, training begins with nearly uniform token weights and progressively focuses on regions that contribute more strongly to audio-visual synchronization.

We also apply depth-dependent gradient scaling to the audio-to-video pathway. Gradients entering the audio stream are attenuated more strongly in shallow blocks and progressively restored in deeper interaction blocks. Together, regional weighting and directional gradient scaling improve audiovisual alignment while preserving the pretrained modality-specific representations.

![](images/ed7e3be0696866c3a2beef4e0352d6c1f2e3c98927c86ad460813cb65dc6e993.jpg)  
Figure 6 Modality-aware reinforcement learning for joint audio-video generation. Video- and audio-specific rewards are routed to their corresponding streams, while the shared audio-visual synchronization reward jointly optimizes both streams and their cross-modal interactions.

## 4.3 Training

We initialize the reinforcement-learning policy from the pretrained DreamX-Creator 1.0 base model. The training set contains 1,000 first-frame–prompt pairs, where the first frame specifies the initial visual appearance and scene structure, and the prompt describes the expected visual dynamics and accompanying audio. For each condition (I, c), the current policy generates a group of G joint audio-video candidates:

$$
\left\{ \left( x _ { v } ^ { ( j ) } , x _ { a } ^ { ( j ) } \right) \right\} _ { j = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid I , c ) ,\tag{10}
$$

where $\pi _ { \theta _ { \mathrm { o l d } } }$ denotes the rollout policy, while $\boldsymbol { x } _ { v } ^ { ( j ) }$ and $x _ { a } ^ { ( j ) }$ denote the generated video and audio components of the j-th candidate, respectively. Each candidate is evaluated independently using video-quality, audio-quality, prompt-consistency, and audio-visual synchronization rewards. The rewards are normalized within candidates generated from the same condition to obtain relative video, audio, and cross-modal advantages.

The resulting advantages are routed according to their corresponding modalities. Video-specific feedback supervises the video stream, audio-specific feedback supervises the audio stream, and cross-modal feedback is shared by both streams and their bidirectional interaction modules. Synchronization-relevant regions identified by deeper cross-modal responses are assigned greater importance, while gradient propagation through shallow modality-specialized blocks is conservatively controlled to protect the visual and acoustic priors learned during pretraining. The overall training objective is written as

$$
{ \mathcal { L } } _ { \mathrm { R L } } = \lambda _ { v } { \mathcal { L } } _ { v } ( A _ { v } + \lambda _ { a v } ^ { v } A _ { a v } ) + \lambda _ { a } { \mathcal { L } } _ { a } ( A _ { a } + \lambda _ { a v } ^ { a } A _ { a v } ) + \lambda _ { \mathrm { r e g } } { \mathcal { L } } _ { \mathrm { r e g } } ( \pi _ { \theta } , \pi _ { \mathrm { b a s e } } ) ,\tag{11}
$$

where $A _ { v } , A _ { a } ,$ and $A _ { a v }$ denote the group-normalized video, audio, and cross-modal advantages, respectively. The regularization term constrains the updated policy toward the pretrained base model to preserve generation diversity and prevent reward over-optimization.

Training alternates between grouped candidate generation, multimodal reward evaluation, and policy optimization. The updated policy is periodically used to refresh the rollout model, forming an online generation– evaluation–update loop. The post-trained model retains the same first-frame-conditioned joint audio-video generation interface as the base model.

## 5 Autoregressive 1-Step 2K Refinement

High-resolution video generation requires both coherent motion and rich spatial detail. Directly generating 2K videos with the joint audio-visual model is computationally expensive, while operating at a tractable resolution leaves the output short of fine textures and sharp boundaries. We therefore introduce 2K Refiner, an autoregressive 1-step 2K refinement stage. The joint generator first produces a coherent low-resolution (LR) video, and 2K Refiner enhances it to 2K resolution while preserving the generated content, motion, and audio-synchronized timing.

At inference, 2K Refiner refines the video sequentially over temporal chunks. For each chunk, the model performs a single denoising evaluation conditioned on the LR video information and the previously refined high-resolution (HR) chunks. This design avoids the cost of multi-step difusion sampling at 2K resolution and makes refinement scalable to long videos. The audio stream is not modified during this process.

Our training pipeline consists of three stages. We first train a bidirectional multi-step difusion teacher for high-quality video refinement. We then adapt it into an autoregressive multi-step refiner, which serves as the intermediate model for 1-step distillation. Finally, we distill the autoregressive refiner into a 1-step student with DMD. The deployed 2K Refiner is this final autoregressive 1-step student.

![](images/94f1b2164c05180622b02733e1d0c73f6e49c2e088dde074bb6f73330e0a2b3e.jpg)  
Figure 7 Training pipeline of 2K Refiner. A bidirectional multi-step refinement teacher is adapted into an autoregressive multi-step refiner and then distilled into an autoregressive 1-step 2K student. The final model performs one denoising evaluation per temporal chunk refinement.

## 5.1 Bidirectional Refiner

We begin with a bidirectional difusion-transformer teacher trained for video refinement. The teacher operates in the latent space and is conditioned on the LR input video. Its bidirectional spatiotemporal attention allows it to exploit both past and future frames, providing a strong refinement prior for recovering temporally consistent details.

The teacher is trained with flow matching (Lipman et al., 2023). For each HR training video, we sample a noise level and corrupt the clean HR latent toward Gaussian noise. The teacher is then optimized to predict the corresponding velocity field conditioned on the LR video. This objective trains the model to progressively transform noisy HR latents into clean refined latents while remaining faithful to the LR input.

Degradation curriculum. Training pairs are synthesized by degrading HR videos, following real-world VSR practice (Chan et al., 2022b). We start from mild degradations to encourage structure preservation, and progressively introduce stronger blur, noise, compression artifacts, resampling artifacts, geometric distortion, and temporally correlated corruptions. The last category is important because the inputs to 2K Refiner are generated videos rather than raw camera captures. Generated videos often contain flickering textures, local motion jitter, and unstable fine structures. Including such corruptions during training helps the teacher learn temporal repair rather than merely spatial sharpening.

## 5.2 Autoregressive Refiner

The bidirectional teacher produces high-quality refinements, but running a multi-step bidirectional model at 2K resolution is too costly. We therefore factorize HR refinement autoregressively over temporal chunks. Instead of denoising the entire HR video jointly, the model refines one chunk at a time, using the LR video information and the previously refined HR context.

This causal factorization reduces high-resolution computation while preserving temporal continuity through the autoregressive context. It also matches the intended deployment setting: during inference, each new chunk is refined after the previous chunks have already been produced.

We convert the bidirectional teacher into an autoregressive multi-step refiner by conducting teacher forcing: the current chunk is denoised conditioned on the LR video and ground-truth past HR chunks. The resulting autoregressive refiner retains the refinement quality of the bidirectional teacher while matching the temporal structure of the final student.

## 5.3 DMD Distillation

The autoregressive multi-step refiner is still too slow for deployment, since each chunk requires multiple denoising evaluations. We distill it into a 1-step student using Distribution Matching Distillation (DMD) (Yin et al., 2024b), following the DMD2 formulation (Yin et al., 2024a). DMD matches the output distribution of the student to the refinement distribution of the multi-step refiner, rather than forcing the student to imitate intermediate denoising states.

At each autoregressive step, the 1-step student predicts the current HR chunk from Gaussian noise, the LR video, and its own previously refined chunks. Since the student uses its own outputs as temporal context at inference, training it only with ground-truth past chunks would create exposure bias. Small refinement errors could then accumulate across chunks, leading to flicker, over-sharpening, or temporal drift. We therefore train the student under self-rollout. During distillation, complete videos are generated by the student itself, and the DMD objective is applied to these generated rollouts. This aligns the training distribution with the inference distribution and encourages the student to correct errors induced by its own previous refinements.

The distilled objective combines distribution matching with pixel-space supervision. The diference between real-score and fake-score provides the distribution-matching signal that pushes the student toward the teacherdefined refinement distribution. In addition, we decode the generated latents with a frozen VAE decoder and apply perceptual and reconstruction losses against the HR ground truth. We use DISTS as the perceptual loss and an $\ell _ { 2 }$ reconstruction loss for pixel-level consistency, with both loss weights set to one in all experiments. The LR video condition keeps the refinement faithful to the generated content, while distribution matching supplies realistic high-frequency detail. As a result, 2K Refiner performs eficient 1-step 2K refinement without changing the motion or audio-aligned timing produced by the joint generator.

## 6 Evaluation

## 6.1 Settings

While benchmarks such as VMBench (Huang et al., 2023; Ling et al., 2025) efectively evaluate general video generation quality—measuring core attributes like motion and temporal consistency—they are not designed to assess audio quality or audio-visual alignment. Consider this, we evaluate our model on Verse-Bench (Wang et al., 2025a), which consists of three evaluation sets covering diferent audio-video generation scenarios. Set 1 and Set 2 cover diverse general audio-visual events, whereas Set 3 focuses on speech-centric scenarios with visible speakers. Since Verse-Bench provides separate video and audio descriptions for each sample, we use Qwen3.6-27B (Qwen Team, 2026) to merge them into a single unified prompt that jointly describes the visual content and the corresponding acoustic events. The unified prompts are then used as the text conditions for joint audio-video generation.

## 6.2 Quantitative Evaluation

We evaluate the generated audio-video samples from four complementary aspects:

• Video quality. We report the video-quality score (VQ), which jointly considers perceptual quality and identity consistency. Perceptual quality is evaluated using Aesthetic Predictor (discus0434, 2024), MUSIQ (Ke et al., 2021), and MANIQA (Yang et al., 2022), while identity consistency is measured using DINOv3 (Siméoni et al., 2025).

• Audio quality. We adopt Audiobox Aesthetics (Tjandra et al., 2025) and report its four dimensions: Content Enjoyment (CE), Content Usefulness (CU), Production Complexity (PC), and Production Quality (PQ), which characterize complementary aspects of generated audio quality.

• Speech generation. For the speech-focused Set 3, we report Word Error Rate (WER) to evaluate the correctness of generated speech content and SyncNet-based LSE-C (Chung and Zisserman, 2016) to measure lip-audio synchronization.

• Audio-visual alignment. We evaluate both semantic and temporal correspondence between the two modalities. ImageBind similarity (IB) (Girdhar et al., 2023) measures audio-visual semantic consistency, while the Synchformer-based DeSync score (Iashin et al., 2024) measures temporal synchronization.

To distinguish comparisons with the five established research baselines from those with substantially larger recent open-weight systems, we report the two groups separately. Table 3 compares DreamX-Creator 1.0 with NAVA (Ji et al., 2026), UniAVGen (Zhang et al., 2026b), Ovi (Low et al., 2025), and the two DaVinci-MagiHuman variants (Chern et al., 2026), whereas Table 4 isolates the comparison with LTX-2.3 (Lightricks, 2026a,b) and MiniMax-H3 (MiniMax, 2026).

Table 3 Preliminary quantitative comparison with NAVA, UniAVGen, Ovi, and the DaVinci-MagiHuman variants. Metrics are grouped by evaluation target, and arrows indicate the preferred direction for each metric. Bold and underlined values denote the best and second-best results within this table, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Params</td><td colspan="2">Video</td><td colspan="2">Audio Aesthetics</td><td></td><td>ASR</td><td>Lip Sync</td><td colspan="2">AV-Align</td></tr><tr><td>VQ↑</td><td>CE↑</td><td>CU↑</td><td>PC↓</td><td>PQ↑</td><td>WER↓</td><td>LSE-C↑</td><td>IB↑</td><td>DeSync↓</td></tr><tr><td>NAVA</td><td>6.3B</td><td>0.6116</td><td>4.7695</td><td>6.0053</td><td>2.3027</td><td>6.3779</td><td>0.1574</td><td>7.7261</td><td>0.2853</td><td>0.2342</td></tr><tr><td>UniAVGen</td><td>7.1B</td><td>0.6644</td><td>4.5938</td><td>5.8605</td><td>2.1534</td><td>6.7643</td><td>0.1661</td><td>4.9556</td><td>0.1556</td><td>0.5371</td></tr><tr><td>Ovi</td><td>10B</td><td>0.6542</td><td>4.7756</td><td>5.9832</td><td>2.1280</td><td>6.1427</td><td>0.1053</td><td>7.3095</td><td>0.1846</td><td>0.4730</td></tr><tr><td>DaVinci-MagiHuman-256p</td><td>15B</td><td>0.6067</td><td>4.9288</td><td>6.0619</td><td>2.4117</td><td>6.2781</td><td>0.1311</td><td>5.7972</td><td>0.2565</td><td>0.4150</td></tr><tr><td>DaVinci-MagiHuman-512p</td><td>15B</td><td>0.6013</td><td>4.9361</td><td>6.0720</td><td>2.4156</td><td>6.2836</td><td>0.1228</td><td>6.8997</td><td>0.2653</td><td>0.4010</td></tr><tr><td>Ours</td><td>7B</td><td>0.6568</td><td>4.7210</td><td>6.0818</td><td>2.1739</td><td>6.3519</td><td>0.1112</td><td>7.8018</td><td>0.2608</td><td>0.1902</td></tr><tr><td>Ours (RL)</td><td>7B</td><td>0.6573</td><td>4.7463</td><td>6.0929</td><td>2.4072</td><td>6.4094</td><td>0.1232</td><td>7.8361</td><td>0.2677</td><td>0.1351</td></tr><tr><td>Ours (Refiner)</td><td>7B</td><td>0.6930</td><td>4.7463</td><td>6.0929</td><td>2.4072</td><td>6.4094</td><td>0.1232</td><td>7.6979</td><td>0.2675</td><td>0.1731</td></tr></table>

VQ denotes the reported video-quality aggregate. CE, CU, PC, and PQ are the AudioBox Aesthetics dimensions Content Enjoyment, Content Usefulness, Production Complexity, and Production Quality. DeSync is the Synchformer-based audiovisual desynchronization score used in the MOVA-style evaluation. PC, WER, and DeSync are lower-is-better under this protocol; VQ, CE, CU, PQ, LSE-C, and IB are higher-is-better. Precision, duration, and inference settings are retained in model names where supplied.

Table 4 Preliminary quantitative comparison with the larger open-weight LTX-2.3 and MiniMax-H3 systems. Metrics are grouped by evaluation target, and arrows indicate the preferred direction for each metric.
<table><tr><td>Model</td><td>Params</td><td>Video VQ↑</td><td>CE↑</td><td>Audio Aesthetics CU↑</td><td>PC↓</td><td>PQ↑</td><td>ASR WER↓</td><td>Lip Sync LSE-C↑</td><td>AV-Align IB↑</td><td>DeSync↓</td></tr><tr><td>LTX-2.3</td><td>22B</td><td>0.6285</td><td>5.1876</td><td>6.4033</td><td>2.6585</td><td>6.7169</td><td>0.1113</td><td>7.6768</td><td>0.3081</td><td>0.2412</td></tr><tr><td>MiniMax-H3 (open source)</td><td>33B</td><td>0.6429</td><td>5.2776</td><td>6.5847</td><td>2.4428</td><td>7.0140</td><td>0.1619</td><td>8.7354</td><td>0.3119</td><td>0.2708</td></tr><tr><td>Ours</td><td>7B</td><td>0.6568</td><td>4.7210</td><td>6.0818</td><td>2.1739</td><td>6.3519</td><td>0.1112</td><td>7.8018</td><td>0.2608</td><td>0.1902</td></tr><tr><td>Ours (RL)</td><td>7B</td><td>0.6573</td><td>4.7463</td><td>6.0929</td><td>2.4072</td><td>6.4094</td><td>0.1232</td><td>7.8361</td><td>0.2677</td><td>0.1351</td></tr><tr><td>Ours (Refiner)</td><td>7B</td><td>0.6930</td><td>4.7463</td><td>6.0929</td><td>2.4072</td><td>6.4094</td><td>0.1232</td><td>7.6979</td><td>0.2675</td><td>0.1731</td></tr></table>

VQ denotes the reported video-quality aggregate. CE, CU, PC, and PQ are the AudioBox Aesthetics dimensions Content Enjoyment, Content Usefulness, Production Complexity, and Production Quality. DeSync is the Synchformer-based audiovisual desynchronization score used in the MOVA-style evaluation. PC, WER, and DeSync are lower-is-better under this protocol; VQ, CE, CU, PQ, LSE-C, and IB are higher-is-better. Precision, duration, and inference settings are retained in model names where supplied.

Compared with the larger LTX-2.3 (22B) (Lightricks, 2026a,b) and MiniMax-H3 (33B) (MiniMax, 2026), our 7B model remains behind on several metrics. Relative to Ours (RL), LTX-2.3 is higher on CE (5.1876 vs. 4.7463), CU (6.4033 vs. 6.0929), PQ (6.7169 vs. 6.4094), and IB (0.3081 vs. 0.2677). MiniMax-H3 further leads on CE (5.2776), CU (6.5847), PQ (7.0140), LSE-C (8.7354), and IB (0.3119), compared with 4.7463, 6.0929, 6.4094, 7.8361, and 0.2677 for Ours (RL), respectively. These gaps show that the current 7B model has not reached across-the-board parity with the two larger open-weight baselines, particularly in audio aesthetics and cross-modal semantics, and it also remains behind MiniMax-H3 in lip synchronization. The favorable VQ and DeSync results therefore represent a trade-of rather than overall superiority; closing the remaining gap will require stronger audio post-training and cross-modal alignment.

## 6.3 Refiner Comparison

Table 5 shows that our refiner achieves the best overall trade-of among the refinement methods. It obtains the highest MUSIQ and MANIQA scores, 0.7073 and 0.4382, respectively, and a competitive aesthetic score of 0.4911. In addition to improving perceptual quality, our method also achieves the best LSE-C and IB among the refinement methods, together with a low DeSync score of 0.1731. These results indicate that our refiner improves perceptual video quality while better retaining semantic consistency and audio-visual alignment, leading to a more balanced refinement performance.

Table 5 Comparison of the baseline generator and video restoration/refinement methods. Arrows indicate the preferred direction for each metric.
<table><tr><td>Model</td><td>Aesthetic↑</td><td>MUSIQ↑</td><td>MANIQA↑</td><td>LSE-C↑</td><td>IB↑</td><td>DeSync↓</td></tr><tr><td>Baseline</td><td>0.4444</td><td>0.5895</td><td>0.3006</td><td>7.8361</td><td>0.2677</td><td>0.1351</td></tr><tr><td>FlashVSR</td><td>0.4940 (+.0496)</td><td>0.7000 (+.1105)</td><td>0.4180 (+.1174)</td><td>7.6772 (−.1589)</td><td></td><td>0.2627(−.0050) 0.1604(+.0253)</td></tr><tr><td>SeedVR</td><td>0.4672(+.0228)</td><td> $0 . 6 7 1 0 \scriptstyle ( + . 0 8 1 5 )$ </td><td>0.3586 (+.0580)</td><td> $7 . 3 4 5 6 \ : ( - . 4 9 0 5 )$ </td><td> $0 . 2 6 4 0 \scriptstyle ( - . 0 0 3 7 )$ </td><td> $0 . 1 8 5 1 \ : ( + . 0 5 0 0 )$ </td></tr><tr><td>LTX-2.5 Refiner</td><td>0.4824(+.0380)</td><td>0.6200 (+.0305)</td><td>0.3377(+.0371)</td><td> $7 . 6 4 8 9 \ : ( - . 1 8 7 2 )$ </td><td> $\underline { { 0 . 2 6 7 0 } } \ : ( - . 0 0 0 7 )$ </td><td> $0 . 2 1 6 6 \left( + . 0 8 1 5 \right)$ </td></tr><tr><td>Ours Refiner</td><td>0.4911(+.0467)</td><td></td><td></td><td>0.7073(+.1178) 0.4382(+.1376) 7.6979(−.1382)</td><td> $\mathbf { 0 . 2 6 7 5 \ : ( - . 0 0 0 2 ) }$ </td><td>0.1731 (+.0380)</td></tr></table>

Higher is better for Aesthetic, MUSIQ, MANIQA, LSE-C, and IB. Lower is better for DeSync. Parenthesized values report the raw diference relative to the Baseline (method minus Baseline); green and red indicate improved and degraded performance, respectively. Therefore, a positive DeSync diference is red because it indicates worse synchronization. Bold and underlined values denote the best and second-best results among the restoration/refinement methods, respectively; the gray Baseline row is excluded from this ranking.

## 6.4 User Study

To complement automatic metrics, we conduct a blind side-by-side human preference study. Each trial compares DreamX-Creator 1.0 with one baseline under the same prompt and playback setting. The comparison cases are randomly sampled from Verse-Bench. The model identities are anonymized and the left-right order is randomized. Assessors report whether DreamX-Creator 1.0 wins, ties, or loses along four dimensions: text-video alignment (TV-Align), audio-video alignment (AV-Align), video quality, and audio quality. We report all percentages from the perspective of DreamX-Creator 1.0.

![](images/189e9a5a32edc23c25dc0d8ea52df561eed68f30e71578c76429152574fd8431.jpg)  
Figure 8 Human preference study comparing DreamX-Creator 1.0 with Ovi, UniAVGen, NAVA, and DaVinci (top), and with Wan2.7, Kling v3, and MiniMax-H3 (bottom). Each horizontal stacked bar reports Win/Tie/Lose percentages from the perspective of DreamX-Creator 1.0 under blind side-by-side comparison.

As shown in figure 8, DreamX-Creator 1.0 records more wins than losses in every criterion–baseline pair against Ovi, UniAVGen, NAVA, and DaVinci. The advantage is most pronounced in video quality, with win rates of 64.2%, 73.7%, 61.7%, and 68.2%, respectively, and remains consistent in AV-Align, where win rates range from 41.8% to 59.2%. TV-Align is closer against Ovi and NAVA, with tie rates of 47.4% and 45.7%, respectively, but DreamX-Creator 1.0 still receives more wins than losses in both comparisons.

Against the industrial-scale systems, the observed gap is limited in several dimensions, particularly video quality: DreamX-Creator 1.0 wins/loses 45.5%/41.3% against Wan2.7 and 48.8%/40.7% against Kling v3, while the comparison with MiniMax-H3 is close at 39.5%/18.7%/41.8% win/tie/lose. TV-Align also yields more losses than wins, although tie rates are high (43.2%–53.0%), and losses exceed wins on AV-Align and audio quality against all three systems. These results should be interpreted within the scope of the sampled subset: highly dynamic and compositionally complex scenes are under-represented, so the study does not fully probe the capabilities of these industrial systems. The findings therefore show that our 7B model remains competitive on the comparatively less challenging cases covered by this study, rather than establishing across-the-board parity; audio fidelity and cross-modal alignment remain the main areas for improvement.

## 7 Related Work

## 7.1 Audio-Video Generation Models

Audiovisual synthesis is often posed directionally—video-to-audio (Luo et al., 2023; Zhang et al., 2024; Cheng et al., 2025) or audio-to-video for talking-head/human animation (Tian et al., 2024; Lin et al., 2025; Gao et al., 2025; Yang et al., 2026b,a)—which performs well when one modality can be treated as a fixed timeline, but precludes reciprocal co-generation. Joint difusion models and difusion transformers instead sample audio and video together, progressing from coupled denoisers (Ruan et al., 2023; Xing et al., 2024) to more integrated backbones and expert compositions (Wang et al., 2024; Liu et al., 2024; Low et al., 2025; HaCohen et al., 2026; SII-OpenMOSS Team, 2026; Cheng et al., 2026; Wang et al., 2025a; Zhang et al., 2026b) and toward shared representations with explicit alignment and physical-time modeling (Ji et al., 2026; Liu et al., 2025b, 2026). Achieving both semantic correspondence and fine-grained synchronization remains central, motivating physical-time encodings, cross-modal attention, discriminative synchrony objectives (Low et al., 2025; Wang et al., 2025a; Liu et al., 2026; Iashin et al., 2024; Chung and Zisserman, 2016), and selective fusion mechanisms such as gated attention (Qiu et al., 2025). DreamX-Creator 1.0 follows this trend by keeping modality-specific streams while making bidirectional exchange context dependent.

## 7.2 From Single-Modality to Joint Audio-Video Preference Alignment

Difusion and flow models (Chu et al., 2025; Yu et al., 2026b; Lei et al., 2026, 2023) primarily fit the data distribution and do not directly optimize perceptual quality, instruction following, or human preference, motivating a line of post-training work that aligns generators to explicit rewards. At the image level, methods difer mainly in their optimization paradigm: DDPO casts denoising as a sequential decision process and applies policy-gradient reinforcement learning (Black et al., 2023), Difusion-DPO derives a direct preference objective from chosen/rejected pairs without an explicit reward model (Wallace et al., 2024), and D3PO extends direct preference optimization along the full denoising trajectory to avoid training a separate reward model (Yang et al., 2024).

Extending alignment from images to video adds a temporal dimension, so rewards must evaluate not only single-frame aesthetics and text consistency but also temporal coherence, motion quality, subject consistency, and long-range dynamics. VADER backpropagates diferentiable rewards through video difusion (Prabhudesai et al., 2024), VideoDPO constructs preference pairs tailored to video generation (Liu et al., 2025c), and reinforcement-learning formulations such as DanceGRPO and Flow-GRPO adapt group-relative policy optimization to difusion and flow models (Xue et al., 2025; Liu et al., 2025a), while DifusionNFT recasts reward-driven fine-tuning as negative-aware forward-process training (Zheng et al., 2025). These methods are efective for video, yet they still optimize only the visual modality and cannot directly optimize joint audio-video multi-modal tasks.

Joint audio-video generation raises distinct alignment challenges: a single sample simultaneously involves visual quality, audio quality, text consistency, cross-modal semantics, and temporal synchronization; diferent rewards may conflict; a single global reward cannot clearly attribute improvements to the audio stream, the video stream, or the interaction module; and fine-grained synchronization typically occurs only in local regions such as lip movements, collisions, and sounding objects. JavisDiT++ proposes AV-DPO with chosen/rejected pairs reflecting visual, acoustic, semantic, cross-modal, and synchronization criteria (Liu et al., 2026), and OmniNFT extends negative-aware forward-process training to the multi-modal setting (Zhang et al., 2026a). Such audiovisual alignment introduces additional pitfalls: improving one modality can hide regressions in the other, and optimizing a synchrony score can exploit evaluator artifacts—so controlled same-clip temporal shifts and held-out human comparisons remain necessary to distinguish genuine alignment improvements from reward-model shortcutting. Existing work proves the feasibility of audio-video preference optimization, but how to perform stable post-training for a joint generator with modality-specific backbones and a bidirectional interaction module remains open. DreamX-Creator 1.0 addresses this with (1) a first-frame-conditioned RL setting; (2) decomposed audio, video, and cross-modal feedback that keeps quality, instruction adherence, semantic correspondence, event timing, and lip synchronization as separable, auditable components; and (3) integration with DreamX-Creator 1.0’s bidirectional context-aware interaction.

## 7.3 Video Refinement and Few-Step Distillation

Video restoration and super-resolution propagate information across frames for detail and temporal consistency. BasicVSR++ and RealBasicVSR use recurrent propagation for clean and real-world degradations (Chan et al., 2022a,b), and VRT introduces a recurrent restoration transformer (Liang et al., 2024). Difusion-based refiners (StableVSR, Upscale-A-Video, VEnhancer) leverage generative priors for realistic enhancement (Rota et al., 2024; Zhou et al., 2024; He et al., 2024), while large-scale systems such as VideoGigaGAN and SeedVR target more detailed and general restoration (Zhang et al., 2025; Wang et al., 2025b); SeedVR in particular is a 2.48B-parameter difusion transformer with shifted-window attention that performs generic real-world video restoration at arbitrary length and resolution (Wang et al., 2025b). Recent industrial and interactive systems push this further: LPM scales generative video restoration with progressive training and temporal-pyramid inference for arbitrarily long videos in production (Zeng et al., 2026), SparkVSR lets users super-resolve sparse keyframes and then propagates the high-resolution priors to the full video (Yu et al., 2026a), and UltraFlash cascades streaming components for real-time high-resolution generation and super-resolution (roughly 30 FPS at 1K) (Luxury et al., 2026).

Separately, few-step generation uses distillation to reduce sampling cost. DMD and DMD2 match teacher and student distributions, and LADD improves high-resolution eficiency (Yin et al., 2024b,a; Sauer et al., 2024). These distillation principles have been carried into video super-resolution: DUO-VSR applies distributionmatching distillation with a dual-stream design for one-step inference, achieving roughly a 50× speed-up over SeedVR-7B (Lv et al., 2026), FlashVSR is the first difusion-based one-step streaming VSR, combining three-stage distillation, locality-constrained sparse attention, and a tiny conditional decoder to reach real-time ∼17 FPS at 768×1408 (Zhuang et al., 2025), and LiteVSR adds a lightweight adapter (11.25% trainable parameters) with time-dependent cross-attention aligning intermediate denoising states, compatible with single-step inference (Cao et al., 2026). For autoregressive video, Self-Forcing and Causal Forcing reduce train–test mismatch by training on the model’s own generated context (Huang et al., 2025b; Zhu et al., 2026). Our refiner operates under a diferent contract: the full low-resolution clip is available before refinement, enabling bidirectional full-clip attention for both teacher and few-step student. We therefore distill under the ofline refinement setting rather than imposing a causal constraint.

## 8 Conclusion

We presented DreamX-Creator 1.0, an open framework for joint audio-video generation. Its implemented generator preserves dedicated audio and video streams, aligns them on a shared timeline, and exchanges information through bidirectional attention regulated by a hidden- and context-dependent gate. Contentadaptive directional training and full-model consolidation place A2V, V2A, and joint generation within one flow-matching framework. The broader system connects this generator to two next steps: multimodal reinforcement learning for user-facing alignment and a bidirectional few-step Refiner targeting 2K output. In this initial report these stages are stated as designs with explicit validation requirements, not as completed empirical claims. By releasing the evaluated model variant, inference stack, and evaluation tools with explicit provenance, our goal is to broaden access to synchronized audio-video generation and high-resolution generative video research.

## References

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training difusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. https://arxiv.org/abs/2305.13301.

ByteDance Seed. One-take creation, flexible referencing: Introducing Seedance 2.5. Oficial model announcement, 2026. https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5.

Yu Cao, Ziquan Liu, Zhensong Zhang, Jiankang Deng, Shaogang Gong, and Jifei Song. Litevsr: Lightweight adaptation of frozen difusion transformers for video super-resolution, 2026. https://arxiv.org/abs/2606.09250.

Brandon Castellano. PySceneDetect. https://github.com/Breakthrough/PySceneDetect.

Kelvin C. K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. BasicVSR++: Improving video superresolution with enhanced propagation and alignment. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022a. https://arxiv.org/abs/2104.13371.

Kelvin C. K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeofs in real-world video super-resolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022b. https: //arxiv.org/abs/2111.12704.

Delong Chen, Tejaswi Kasarla, Yejin Bang, Mustafa Shukor, Willy Chung, Jade Yu, Allen Bolourchi, Theo Moutakanni, and Pascale Fung. Action100m: A large-scale video action dataset. arXiv preprint arXiv:2601.10592, 2026.

Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. VGGSound: A large-scale audio-visual dataset. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2020. https://arxiv.org/abs/2004.14368.

Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander G. Schwing, and Yuki Mitsufuji. MMAudio: Taming multimodal joint training for high-quality video-to-audio synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. https://arxiv.org/abs/2412.15322.

Shihao Cheng, Jiaxu Zhang, Quanyue Song, Shansong Liu, Zhizhi Guo, Xiaolei Zhang, Chi Zhang, Xuelong Li, and Zhigang Tu. Unison: Harmonizing motion, speech, and sound for human-centric audio-video generation. arXiv preprint arXiv:2605.08729, 2026. https://arxiv.org/abs/2605.08729.

Ethan Chern, Hansi Teng, Hanwen Sun, Hao Wang, Hong Pan, Hongyu Jia, Jiadi Su, Jin Li, Junjie Yu, Lijie Liu, et al. Speed by simplicity: A single-stream architecture for fast audio-video generative foundation model. arXiv preprint arXiv:2603.21986, 2026.

Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding, 2025. https://arxiv.org/abs/2503.06132.

Joon Son Chung and Andrew Zisserman. Out of time: Automated lip sync in the wild. In Asian Conference on Computer Vision Workshops, 2016. https://arxiv.org/abs/1609.00204.

discus0434. Aesthetic predictor v2.5. https://github.com/discus0434/aesthetic-predictor-v2-5, 2024.

Dreamina. Seedance 2.5: 4k video generation with audio. Oficial product page, 2026. https://dreamina.capcut.com/ seedance/seedance-2-5.

Yue Gao et al. Wan-S2V: Audio-driven cinematic video generation. arXiv preprint arXiv:2508.18621, 2025. https: //arxiv.org/abs/2508.18621.

Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2017.

Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. https://arxiv.org/abs/2305.05665.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, et al. LTX-2: Eficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026. https://arxiv.org/abs/2601.03233.

Jingwen He et al. VEnhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667, 2024. https://arxiv.org/abs/2407.07667.

Xiaohu Huang, Hao Zhou, Qiangpeng Yang, Shilei Wen, and Kai Han. Jova: Unified multimodal learning for joint video-audio generation. arXiv preprint arXiv:2512.13677, 2025a.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. arXiv preprint arXiv:2506.08009, 2025b. https://arxiv.org/abs/2506.08009.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models, 2023. https://arxiv.org/abs/2311.17982.

Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Eficient synchronization from sparse cues. arXiv preprint arXiv:2401.16423, 2024. https://arxiv.org/abs/2401.16423.

Longbin Ji, Guan Wang, Xuan Wei, Chenye Yang, Xiangrui Liu, Zhenyu Zhang, Shuohuan Wang, Yu Sun, and Jingzhou He. Native audio-visual alignment for generation. arXiv preprint arXiv:2605.30073, 2026. https: //arxiv.org/abs/2605.30073.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 5128–5137. IEEE, 2021.

Kuaishou Technology. Kling AI 3.0: Native multimodal video and audio generation. Oficial model announcement, 2026a. https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-launches-30-model-ushering-era-w here-everyone-can-be.

Kuaishou Technology. Kling AI 3.0: Native 4k video output. Second Quarter 2026 results announcement, 2026b. https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-technology-announces-second-quarter-a nd-interim-2026.

Jiachen Lei, Qinglong Wang, Peng Cheng, Zhongjie Ba, Zhan Qin, Zhibo Wang, Zhenguang Liu, and Kui Ren. Masked difusion models are fast distribution learners. arXiv preprint arXiv:2306.11363, 2023.

Jiachen Lei, Keli Liu, Julius Berner, Yu Haiming, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. There is no VAE: End-to-end pixel-space generative modeling via self-supervised pre-training. In The Fourteenth International Conference on Learning Representations, 2026. https://openreview.net/forum?id=HbUoKPIZmp.

Xuan Li et al. OpenHumanVid: A large-scale high-quality dataset for enhancing human-centric video generation. arXiv preprint arXiv:2412.00115, 2024. https://arxiv.org/abs/2412.00115.

Jingyun Liang, Jiezhang Cao, Yuchen Fan, Kai Zhang, Rakesh Ranjan, Yawei Li, Radu Timofte, and Luc Van Gool. VRT: A video restoration transformer. IEEE Transactions on Image Processing, 2024. https://arxiv.org/abs/2201.12288.

Lightricks. LTX-2.3: Open audio-video generation model. Oficial model card, 2026a. https://huggingface.co/Lightri cks/LTX-2.3.

Lightricks. LTX-2.3: Production-quality video at up to 4k with local inference. Oficial release announcement, 2026b. https://ltx.io/newsroom/ltx-2-3-ofers-creative-intelligence-without-the-cloud-costs.

Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. OmniHuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. arXiv preprint arXiv:2502.01061, 2025. https://arxiv.org/abs/25 02.01061.

Xinran Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. Vmbench: A benchmark for perception-aligned video motion generation, 2025. https: //arxiv.org/abs/2503.10076.

LingBot Team. LingBot-Video: Scaling mixture-of-experts video pretraining for embodied intelligence. Oficial model repository, 2026. https://github.com/Robbyant/lingbot-video.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. https://arxiv.org/abs/2210.02747.

Haohe Liu et al. SyncFlow: Toward temporally aligned joint audio-video generation from text. arXiv preprint arXiv:2412.15220, 2024. https://arxiv.org/abs/2412.15220.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanl Ouyang. Flow-grpo: Training flow matching models via online rl, 2025a. https://arxiv.org/abs/2505.05470.

Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Jiebo Luo, Ziwei Liu, Hao Fei, and Tat-Seng Chua. JavisDiT: Joint audio-video difusion transformer with hierarchical spatio-temporal prior synchronization. arXiv preprint arXiv:2503.23377, 2025b. https://arxiv.org/abs/2503.23377.

Kai Liu, Yanhao Zheng, Kai Wang, Shengqiong Wu, Rongjunchen Zhang, Jiebo Luo, Dimitrios Hatzinakos, Ziwei Liu, Hao Fei, and Tat-Seng Chua. JavisDiT++: Unified modeling and optimization for joint audio-video generation. arXiv preprint arXiv:2602.19163, 2026. https://arxiv.org/abs/2602.19163.

Runtao Liu, Haoyu Wu, Ziqiang Zheng, Chen Wei, Yingqing He, Renjie Pi, and Qifeng Chen. VideoDPO: Omnipreference alignment for video difusion generation. arXiv preprint arXiv:2412.14167, 2025c. https://arxiv.org/abs/ 2412.14167.

Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin backbone cross-modal fusion for audio-video generation. arXiv preprint arXiv:2510.01284, 2025. https://arxiv.org/abs/2510.01284.

Simian Luo, Chuanhao Yan, Chenxu Hu, and Hang Zhao. Dif-foley: Synchronized video-to-audio synthesis with latent difusion models. In Advances in Neural Information Processing Systems, 2023. https://arxiv.org/abs/2306.17203.

Luxury, Jie Huang, Zihao Fan, Xiaoxiao Ma, Jun hao Zhuang, Yuming Li, Zeyue Xue, Siming Fu, Haoran Li, Mingchen Zhong, Guohui Zhang, Shichen Ma, Yijun Liu, Jiaqi Shi, Yanwen Ma, Yaofeng Su, Haoyu Wang, Yaowei Li, Songchun Zhang, Weiyang Jin, Yuxuan Bian, Shiyi Zhang, Haojun Xu, Shuai Lu, Xin Han, Wei Tang, Haoyang Huang, and Nan Duan. Ultra flash: Scaling real-time streaming video generation to high resolutions, 2026. https://arxiv.org/abs/2606.09150.

Zhengyao Lv, Menghan Xia, Xintao Wang, and Kwan-Yee K. Wong. Duo-vsr: Dual-stream distillation for one-step video super-resolution, 2026. https://arxiv.org/abs/2603.22271.

MiniMax. MiniMax H3: An open omni-modal generation model. Oficial model card, 2026. https://huggingface.co /MiniMaxAI/MiniMax-H3.

Mihir Prabhudesai et al. Video difusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024. https://arxiv.org/abs/2407.08737.

Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free. In Advances in Neural Information Processing Systems, 2025. https://arxiv.org/abs/2505.06708.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. https://qwen.ai/blog?id=qwen3. 6-27b.

Claudio Rota, Marco Buzzelli, and Joost van de Weijer. Enhancing perceptual quality in video super-resolution through temporally-consistent detail synthesis using difusion models, 2024. https://arxiv.org/abs/2311.15908.

Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. MM-Difusion: Learning multi-modal difusion models for joint audio and video generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. https://arxiv.org/abs/2212.09478.

Sand AI. MAGI-2 Preview: Scaling video generation models eficiently. Oficial model release, 2026. https: //github.com/SandAI-org/MAGI-2-preview.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial difusion distillation. ACM Transactions on Graphics, 2024. https://arxiv.org/abs/2403.12015.

Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, et al. Qwen3-asr technical report. arXiv preprint arXiv:2601.21337, 2026

SII-OpenMOSS Team. MOVA: Towards scalable and synchronized video-audio generation. arXiv preprint arXiv:2602.08794, 2026. https://arxiv.org/abs/2602.08794.

Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

DreamX Team, Yancheng Bai, Rui Chen, Xiangxiang Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu, Siyu Hong, Jiachen Lei, et al. Dreamx-world 1.0: A general-purpose interactive world model. arXiv preprint arXiv:2606.16993, 2026a.

DreamX Team, Rui Chen, Xiangxiang Chu, Geng Li, Jifan Li, Qingfeng Shi, Datao Tang, Jing Tang, Jun Wang, and Pengfei Zhang. Dreamx-phi 1.0: Action-conditioned video world model for robotic manipulation, 2026b. https://arxiv.org/abs/2608.13489.

Linrui Tian et al. EMO: Emote portrait alive—generating expressive portrait videos with audio2video difusion model under weak conditions. arXiv preprint arXiv:2402.17485, 2024. https://arxiv.org/abs/2402.17485.

Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hofman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025.

Bram Wallace, Meihua Gokul, Stefano Ermon, and Nikhil Naik. Difusion model alignment using direct preference optimization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. https://arxiv.org/ab s/2311.12908.

Boyang Wang, Guangyi Xu, Jiahui Zhang, Zhipeng Tang, and Zezhou Cheng. Omnishotcut: Holistic relational shot boundary detection with shot-query transformer. arXiv preprint arXiv:2604.24762, 2026. https://arxiv.org/abs/26 04.24762.

Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. UniVerse-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025a. https://arxiv.org/abs/2509.06155.

Jianyi Wang, Zhijie Lin, Meng Wei, Yang Zhao, Ceyuan Yang, Fei Xiao, Chen Change Loy, and Lu Jiang. Seedvr: Seeding infinity in difusion transformer towards generic video restoration, 2025b. https://arxiv.org/abs/2501.01320.

Kai Wang, Shijian Deng, Jing Shi, Dimitrios Hatzinakos, and Yapeng Tian. AV-DiT: Eficient audio-visual difusion transformer for joint audio and video generation. arXiv preprint arXiv:2406.07686, 2024. https://arxiv.org/abs/24 06.07686.

Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8428–8437. IEEE, 2025c.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023.

Yazhou Xing, Yingqing He, Zhenyu Tian, Xintao Wang, and Qifeng Chen. Seeing and hearing: Open-domain visual-audio generation with difusion latent aligners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. https://arxiv.org/abs/2402.17723.

Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. Dancegrpo: Unleashing grpo on visual generation, 2025. https://arxiv.org/abs/2505.07818.

Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Qimai Li, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune difusion models without any reward model, 2024. https://arxiv.org/abs/2311.13231.

Kaixing Yang, Jiashu Zhu, Xulong Tang, Ziqiao Peng, Xiangyue Zhang, Chubin Chen, Puwei Wang, Jiahong Wu, Xiangxiang Chu, Hongyan Liu, et al. Omnidance: Multimodal driven dance video generation with large-scale internet data. arXiv preprint arXiv:2606.30019, 2026a.

Kaixing Yang, Jiashu Zhu, Xulong Tang, Ziqiao Peng, Xiangyue Zhang, Puwei Wang, Jiahong Wu, Xiangxiang Chu, Hongyan Liu, and Jun He. Mace-dance: Motion-appearance cascaded experts for music-driven dance video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, SIGGRAPH Conference Papers ’26, New York, NY, USA, 2026b. Association for Computing Machinery. ISBN 9798400725548. doi: 10.1145/3799902.3811202. https://doi.org/10.1145/3799902.3811202.

Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1190–1199. IEEE, 2022.

Zhen Ye, Xu Tan, Aoxiong Yin, Hongzhan Lin, Guangyan Zhang, Peiwen Sun, Yiming Li, Chi-Min Chan, Wei Ye, Shikun Zhang, et al. Talker-t2av: Joint talking audio-video generation with autoregressive difusion modeling. arXiv preprint arXiv:2604.23586, 2026.

Tianwei Yin, Micha"el Gharbi, Fr’edo Durand, William T. Freeman, Eli Shechtman, and Taesung Park. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, 2024a. https://arxiv.org/abs/2405.14867.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step difusion with distribution matching distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b. https://arxiv.org/abs/2311.18828.

Jiongze Yu, Xiangbo Gao, Pooja Verlani, Akshay Gadde, Yilin Wang, Balu Adsumilli, and Zhengzhong Tu. Sparkvsr: Interactive video super-resolution via sparse keyframe propagation, 2026a. https://arxiv.org/abs/2603.16864.

Meng Yu, Lei Sun, Jianhao Zeng, Xiangxiang Chu, and Kun Zhan. Elucidating the snr-t bias of difusion probabilistic models, 2026b. https://arxiv.org/abs/2604.16044.

Ailing Zeng, Casper Yang, Chauncey Ge, Eddie Zhang, Garvey Xu, Gavin Lin, Gilbert Gu, Jeremy Pi, Leo Li, Mingyi Shi, Shawn Wang, Sheng Bi, Steven Tang, Thorn Hang, Tobey Guo, Vincent Li, Xin Tong, Yikang Li, Yuchen Sun, Yue Zhao, Yuhan Lu, Yuwei Li, Zane Zhang, Zeshi Yang, and Zi Ye. Lpm 1.0: Video-based character performance model, 2026. https://arxiv.org/abs/2604.07823.

Guohui Zhang, XiaoXiao Ma, Jie Huang, Hang Xu, Hu Yu, Siming Fu, Yuming Li, Zeyue Xue, Lin Song, Haoyang Huang, Nan Duan, and Feng Zhao. Omninft: Modality-wise omni difusion reinforcement for joint audio-video generation, 2026a. https://arxiv.org/abs/2605.12480.

Guozhen Zhang, Zixiang Zhou, Teng Hu, Ziqiao Peng, Youliang Zhang, Yi Chen, Yuan Zhou, Qinglin Lu, and Limin Wang. Uniavgen: Unified audio and video generation with asymmetric cross-modal interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1950–1960, 2026b.

Xuaner Zhang et al. VideoGigaGAN: Towards detail-rich video super-resolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. https://arxiv.org/abs/2404.12388.

Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, and Kai Chen. Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. arXiv preprint arXiv:2407.01494, 2024. https: //arxiv.org/abs/2407.01494.

Youliang Zhang, Zhaoyang Li, Duomin Wang, Deyu Zhou, Zixin Yin, Xili Dai, Gang Yu, Xiu Li, et al. Speakervid-5m: A large-scale high-quality dataset for audio-visual dyadic interactive human generation. In International Conference on Learning Representations, volume 2026, pages 117896–117926, 2026c.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Difusionnft: Online difusion reinforcement with forward process, 2025. https: //arxiv.org/abs/2509.16117.

Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-a-video: Temporal-consistent difusion model for real-world video super-resolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. https://arxiv.org/abs/2312.06640.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive difusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026. https://arxiv.org/abs/2602.02214.

Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, and Tianfan Xue. Flashvsr: Towards real-time difusion-based streaming video super-resolution, 2025. https://arxiv.org/abs/2510.12747.