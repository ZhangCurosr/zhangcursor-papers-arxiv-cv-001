# LynnReal-Omni: Native multi-modal Video Generation for Agentic Visual Workflows

LynnReal AI\*

<sup>∗</sup>Lynnreal-Omni contributors are listed at the end of the report.

Video difusion models are stochastic and hard to control: precise content often requires repeated sampling without guaranteed success, and long-horizon scenes drift in appearance, interactions, and temporal coherence. Agentic visual creation provides explicit references, editable 3D scenes, or executable game states for stable control, but does not by itself guarantee high object or character fidelity. Combining the two can enable stable, high-quality generation. To realize this combination, we present LynnReal-Omni, a native multimodal video generation framework built on a 32B shared multimodal difusion transformer that unifies text-to-video, image-conditioned generation, referenceguided generation, structural control, editing, degraded video restoration, and long-video generation. It accepts heterogeneous visual inputs—appearance references, editable 3D renders, and game recordings—allowing agents to compose visual conditions within a unified model. We also train a dedicated 27B Flash shared multimodal difusion transformer for real-time rendering. We build a systematic data pipeline for video cleaning, subject association, multimodal annotation, and aligned control construction, yielding a curated corpus of multi-shot audiovisual segments, and introduce MSAVP, a 100-prompt, 20-metric evaluation design that separates instruction following, generating plausibility, visual quality, temporal behavior, and audio coordination. LynnReal-Omni-Flash further reduces inference cost through model and decoding acceleration, including a lightweight VAE decoder; on one H100, warm generation and decoding of a 22-frame 540p video take 843 ms with LynnReal-Omni and 377 ms with Flash. These results provide a foundation for real-time streaming video generation, making LynnReal-Omni a unified, controllable, and eficient basis for agentic visual creation.

Date: September 2026   
Version: Technical report; September 2026   
Code: https://github.com/LynnReal-AI/LynnReal-Omni   
Demo: https://www.youtube.com/watch?v=P5Bl2mriEmk   
Flash: https://huggingface.co/stdstu123/LynnReal-Onmi-flash-beta-0.1   
Standard: https://huggingface.co/stdstu123/LynnReal-Onmi-beta-0.1   
Light-vae: https://huggingface.co/stdstu123/LynnReal-Onmi-light-vae

## Contents

1 Introduction 2   
2 Related Work   
2.1 Native multi-modal generation.   
2.2 Distribution matching distillation.   
2.3 Physical and audiovisual evaluation.   
2.4 Agentic visual creation.   
3 Data Processing   
3.1 Overview: from public videos to high-quality multishot data .   
3.2 Video Preprocessing 6   
3.3 Coarse annotation 6   
3.4 Retrieval and curation 7   
3.5 Refinement and assets 7   
3.5.1 Subject and background references. 7   
3.5.2 Pose tracking and asset states. 7   
3.5.3 Audio alignment. 7   
3.6 High-quality multishot data 8   
3.7 Video Editing Dataset 8   
4 Method 8   
4.1 Overview 8   
4.2 Native Multimodal Backbone and Task representation 8   
4.3 Multitask Training and Few-Step Distillation 9   
4.3.1 Flash variant and inference acceleration. 10   
4.4 Long Video Generation with Bounded History 11   
4.4.1 Chunk-wise video generation 11   
4.4.2 Compact temporal context 11   
4.5 Lightweight Decoder Distillation 12   
4.5.1 Architecture and initialization . 12   
4.5.2 Training distribution and supervision . 13   
4.5.3 Distillation objectives 13   
4.5.4 Optimization and deployment . 14   
4.6 Temporal repair of generated-video artifacts 15   
4.7 Agent-generated 3D scene and executable game controls 16   
5 Experiments 17   
5.1 Experimental questions and settings 17   
5.2 MSAVP: complex audiovisual and physical evaluation 17   
5.2.1 Benchmark overview 17   
5.2.2 Evaluation workflow 18   
5.2.3 Benchmark results 21   
5.3 Lightweight VAE Evaluation 22   
5.4 Inference Latency . . 23   
5.5 Reference-guided weather and appearance editing 25   
6 Conclusion 26   
7 Contributors 26

## 1 Introduction

Video difusion models (Ho et al., 2022; Blattmann et al., 2023; Polyak et al., 2024; Wan et al., 2025; Valevski et al., 2024; Chen et al., 2025, 2024; Ceylan et al., 2023; Harvey et al., 2022; Wang et al., 2024; He et al., 2025, 2024) have achieved remarkable visual fidelity, but they remain dificult to control. Generation is stochastic (Ho et al., 2020), precise content often requires repeated sampling without any guarantee of success, and long-horizon scenes tend to drift in appearance (Lu et al., 2026b; Cui et al., 2025; Huang et al., 2025), object interactions, and temporal coherence. These limitations make it hard to use video difusion models as reliable rendering and generation engines in workflows that demand explicit and repeatable control. Agentic visual creation (Chen et al., 2026; Ye et al., 2026; OpenAI, 2026) ofers a complementary source of control: an agent can produce reference images, construct editable 3D scenes with camera trajectories, or write executable games with controllable objects and collision rules, thereby making scene geometry and motion explicit. Such conditions stabilize the generation process and reduce the need for repeated sampling. However, agentic control alone does not guarantee high-fidelity object or character appearance. Combining agentic visual creation with video difusion therefore provides a promising path toward stable, high-quality generation.

![](images/998ea2c2011e86e0304cddd04b0ec5abe6727db80b89cd2779276fd92dfe8ec9.jpg)  
Figure 1 Agent workflows. Image-based scene reconstruction and agent-written low-poly games produce distinct visual-control streams. Prompt refinement, image generation, and first-frame editing provide appearance controls. The video model receives these controls through its native reference interface.

Realizing this combination, however, requires general-purpose video models that can understand and combine diverse multimodal references and conditions while maintaining coherent appearance, motion, interactions, audio, and long-term consistency. Existing approaches fall short in several important respects:

(1) Fragmented task-specific pipelines. Task-specific pipelines for text-to-video, video-to-video, imageconditioned generation, reference-guided generation, structural control, editing, and long-video generation have advanced largely in isolation, so agents must compose multiple models and ad hoc interfaces, which increases engineering cost and often produces inconsistent behavior across tasks;

(2) Long-horizon inconsistency. Long-video methods often sufer from appearance drift (Lu et al., 2026b;   
Cui et al., 2025; Huang et al., 2025), identity changes, and photometric artifacts;

(3) Incomplete evaluation protocols. Existing evaluation protocols such as VBench (Huang et al., 2024) and VideoPhy (Bansal et al., 2025) separate some dimensions of video quality, but they do not comprehensively cover multi-shot continuity, action binding, physical plausibility, controllability, and audio quality in a single protocol, and metrics are often aggregated across incompatible scales, hiding failures in specific dimensions;

(4) Decoding bottleneck. Video difusion models are often assumed to be dominated by the denoising transformer, but few-step distillation sharply reduces denoising cost and shifts the bottleneck to the VAE decoder, which becomes the dominant inference cost. Decoder acceleration is therefore essential, yet it must be evaluated jointly with denoising, since reducing decoder compute can afect temporal phase or reconstruction quality.

To address these limitations, we present LynnReal-Omni, a native multimodal video generation framework built on a shared multimodal difusion transformer. Its design addresses the above gaps in several ways:

• Rather than treating each task as a separate pipeline, LynnReal-Omni unifies text-to-video, imageconditioned generation, reference-guided generation, structural control, editing, and long-video generation within a single model, replacing fragmented task-specific stacks with a shared denoiser and task-specific input layouts.

• A native task representation explicitly encodes modality identity, temporal coordinates, noise levels, and output targets, allowing appearance references, frame-aligned controls, editable 3D renders, game recordings, and causal history to retain their distinct roles within a shared multimodal transformer.

• A systematic data pipeline for video cleaning, subject association, multimodal annotation, and aligned control construction yields a curated corpus of multi-shot audiovisual segments, providing source-linked training units with verified conditioning assets.

• We introduce MSAVP, a 100-prompt, 25-metric benchmark for evaluating semantic alignment, visual quality, temporal consistency, physical plausibility, controllability, and audio quality, which separates semantic compliance from observed physics and retains distinct visual, temporal, and audio measures.

• LynnReal-Omni is optimized for practical deployment, and LynnReal-Omni-Flash reduces inference cost through model and decoding acceleration, including a lightweight VAE decoder, with warm generation and decoding of a 22-frame 540p video taking 909 ms on one H100 for LynnReal-Omni and 591 ms for Flash.

## 2 Related Work

## 2.1 Native multi-modal generation.

Video foundation models such as CogVideoX, HunyuanVideo, and Wan combine spatiotemporal latent compression with scalable difusion transformers (Yang et al., 2024; Kong et al., 2024; Wan et al., 2025). LTX-2 (HaCohen et al., 2026) extend generation to synchronized audio and video through cross-modal interaction, while VACE unifies reference-conditioned generation and video editing through a shared conditioning interface (Jiang et al., 2025). Our implementation builds directly on MiniMax-H3 (MiniMax, 2026), which provides a joint video–audio transformer, modality-specific codecs, and native keyframe and reference interfaces. Building on this backbone, we integrate image references, motion controls, editing, and long video generation while preserving task-specific token ordering, modality labels, and temporal positions. We distinguish video-only continuation from joint audiovisual generation and evaluate these interfaces together with few-step inference.

## 2.2 Distribution matching distillation.

Distribution Matching Distillation (DMD) trains one-step generators by matching teacher and student output distributions (Yin et al., 2024b). DMD2 removes the paired regression requirement and improves training through two-time-scale updates, adversarial supervision, and inference-matched multi-step training (Yin et al., 2024a). Subsequent work extends distribution matching along complementary directions: TDM aligns intermediate trajectory distributions for flexible few-step sampling (Luo et al., 2025); Self Forcing trains on autoregressive student rollouts to reduce exposure bias (Huang et al., 2025); and Reward Forcing introduces rewarded distribution matching to improve motion dynamics (Lu et al., 2026a). More recently, Salt combines self-consistent denoising updates with cache-aware training to improve low-step video generation (Ge et al., 2026). Our work applies few-step distillation to both standard and Flash variants, emphasizing multimoda control preservation and measured end-to-end eficiency under their respective deployment configurations.

Trajectory distribution matching (Luo et al., 2025) further motivates supervising short student transitions against the teacher distribution. Our standard and Flash paths have diferent deployment topologies and are evaluated with their respective trained configurations. Depth reduction, token reduction, quantization, operator fusion, and decoder replacement change diferent parts of the cost. Their efects require separate ablations: a reduction in parameter count does not by itself predict whole-pipeline latency, and a numerically exact kernel improvement difers from a quality-sensitive approximation.

## 2.3 Physical and audiovisual evaluation.

VBench separates several dimensions of video quality (Huang et al., 2024). VideoPhy explicitly tests caption adherence and physical commonsense (Bansal et al., 2025). Our MSAVP protocol similarly keeps semantic, physical, visual, and temporal judgments separate and adds structured accounting for multishot action, binding, sound semantics, and event timing. Specialist measurements provide evidence for cut locations and appearance, including TransNetV2 (Soucek and Lokoc, 2024) and MUSIQ (Ke et al., 2021); they cannot substitute for observing an interaction. We retain metric-level applicability counts and original scoring units alongside a hierarchical six-family aggregate, so the overall score does not replace the underlying evidence.

## 2.4 Agentic visual creation.

Early LLM-based agents orchestrated image generation and editing through prompts and tool calls (Wu et al., 2023). Multimodal backbones such as GPT-4o and Gemini 1.5 subsequently enabled visual inspection and reasoning over reference images and extended video contexts, supporting more elaborate image editing and video planning workflows (OpenAI, 2024; Reid et al., 2024). Claude 4 further strengthened sustained coding and tool use, extending agentic creation toward executable games and interactive applications (Anthropic, 2025). More recently, GPT-6 Astra supports complex workflows combining reasoning, coding, and computer use (OpenAI, 2026). However, producing detailed animated scenes still requires substantial downstream work in geometry, materials, rigging, animation, and rendering. Stronger agent backbones do not eliminate these production costs, and the cited advances do not establish real-time, end-to-end creation of finely modeled animated content. This motivates a complementary workflow in which agents construct lightweight scenes or playable game prototypes, while a fast video difusion model supplies detailed visual appearance.

## 3 Data Processing

## 3.1 Overview: from public videos to high-quality multishot data

Our goal is to turn diverse public videos into clean and well-described multishot audiovisual clips. The collected videos cover story and action, sports and performance, animation and computer graphics, nature and aerial views, everyday activities and machines, and commercial or other content. We estimate the mixture shown in Figure 2 by combining the categories used during collection with the content types found during quality screening. After all processing stages, approximately 0.6% of the original storage footprint remains as high-quality data, and every retained clip is no longer than one minute.

Pipeline structure. The pipeline has five main components. We first detect shot boundaries, segment the videos into shots, and remove visible contamination; then group shots by scene and identify recurring subjects; describe each same-scene multishot clip of at most one minute with both a clip-level caption and detailed pershot captions, and select clips with clear motion, coherent events, and good visual quality; prepare subject, background, pose, and audio information; and finally produce detailed captions with automatic consistency checks.

![](images/e2c4001f1be55feff0ffd4a3a46969067e345669a1ca161f98efc1dd7d0427b1.jpg)  
Figure 2 The data processing and annotation pipeline. The pie chart shows an approximate content distribution obtained from collection categories and quality-screening labels; percentages are rounded.

## 3.2 Video Preprocessing

Shot Segmentation. This stage converts raw videos into physically reliable shots. TransNetV2 (Soucek and Lokoc, 2024) proposes possible boundaries with a low threshold of 0.1, which keeps recall high at this stage. We then compare the frames on both sides of each candidate using color distribution, brightness, white-pixel ratio, and edge structure. A candidate is rejected when the change can be explained by a flash, camera shake, a fast pan, or a moving object that briefly covers the frame. A cut is retained when the visual evidence supports a real change of shot, including a cut made during a continuous action. Short shots are kept when the evidence supports them, while long videos without cuts are divided into continuous windows of at most 60 seconds without dropping frames or audio.

Text and border cleaning. PP-OCR (Du et al., 2020) detects and recognizes text in sampled frames. The text is used both to identify text-heavy advertisements and to propose watermark and subtitle regions. Black borders are estimated from brightness and variation across several frames, which prevents a dark scene from being mistaken for a border. We choose the smallest crop that removes the detected regions and reject any crop that would remove more than 25% of the image. If a safe crop does not exist, the clip is kept for later review or rejection instead of being altered by unconstrained image generation. Static watermarks are identified by repeated text at a stable edge position across the full video, whereas subtitles are identified by repeated text within a shot, usually near the top or bottom. After cropping, we finally retain cleaned shots with no visible contamination.

## 3.3 Coarse annotation

Scene-level shot grouping. This stage organizes cleaned shots into scenes and assigns stable identities to important subjects. We use Qwen3.5-122B-A10B (Qwen Team, 2026) to inspect mosaics of representative frames from overlapping windows of 48 consecutive shots in a long video, with 12 shots shared between neighboring windows. Shots are grouped only when they share a physical place and a continuous event or narrative context. The presence of the same person or a similar color palette is not enough by itself. Although the large mosaics provide the model with the narrative context of the full video, we find that some local shots, particularly montage inserts, can still be assigned incorrectly. We therefore refine the initial scene plan in local windows of up to eight numbered frames; this correction pass moves shots only when the visual evidence is clear. A final rule requires every shot in a clip of at most one minute to belong to exactly one scene.

Subject discovery and linking. We first identify the principal subjects in each shot and then match them across all shots in the same clip, enabling subject-consistent multishot captions. In our comparisons, directly using a multimodal model with explicit object-localization capabilities was more accurate for this task than assembling a complex pipeline of specialist models, such as the annotation pipeline used in MultiShotMas ter (Wang et al., 2026). We therefore use Qwen3.5-122B-A10B for both subject discovery and cross-shot linking. For each shot, the model identifies up to six reusable subjects, including people, animals, vehicles, machines, and important objects, and returns a bounding box in the most representative frame for each subject. We describe the two-stage localization procedure in the next paragraph. The model then jointly examines the full scene-grouped clip, representative frames from every shot, and representative crops of the discovered subjects. The full clip provides evidence about actions and narrative continuity, while the selected frames provide evidence about appearance. Cross-shot matching relies on stable cues such as the face, clothing, shape, color, and material, rather than on a subject’s temporary action or location. These stable subject identifiers connect the scene-level description with the description of each shot. We retain up to six reusable subjects for each clip. Events involving other visible subjects discovered at the shot level remain in the captions even when those subjects are not selected as reusable references.

Subject localization. We localize each subject in two stages. A first pass uses the full scene context of a shot to choose a clear frame and a rough region. A second pass examines that exact frame and refines the bounding box.

## 3.4 Retrieval and curation

This stage selects clips that contain sustained, understandable events and removes duplicates. We retrieve candidates from coarse action captions and physical metadata, apply a fast story and motion screen with Qwen v4 Flash, and then review the complete video with Qwen3.5-122B-A10B. Editing cuts, flashing lights, subtitles, and camera shake can all produce high frame diferences without useful subject motion, so optical flow, sharpness, brightness, and frame-change statistics only prioritize candidates and never decide quality on their own.

Content acceptance. A strong candidate has recognizable subjects, clear visual quality, sustained motion, and an event with observable development, such as preparation, action, and outcome. A short attractive moment cannot compensate for a clip that is otherwise static, blurred, corrupted, or dificult to understand. The catalog records strong action, weaker but usable action, reviewed rejection, and not-yet-screened content as separate states; unscreened content is never counted as rejected.

Deduplication and balance. We use the global clip captions produced in the preceding stage to support content review. We compare source-video identifiers, media signatures, time ranges, and caption signatures to detect duplicates. In most cases, only one clip is retained from a source video. A second clip is allowed for a rare topic only when its time range and shots do not overlap the first. We balance live action and animation and retain varied examples of human activity, machines and vehicles, groups, nonhuman creatures, and efects-rich environments. Once selected, a clip keeps its source mapping and selection reason, so a diferent file with the same name cannot silently replace it.

## 3.5 Refinement and assets

This stage converts the selected clips and annotations into visual, pose, and audio references for training our conditional generation model and improving its video-editing capabilities.

## 3.5.1 Subject and background references.

We also use Qwen3.5-122B-A10B to perform a second screening of the subject and background assets. A subject reference should show stable identity features with little occlusion and enough of the subject visible to recognize it. If no suitable frame exists, the reference is marked unavailable rather than replaced with a poor crop. A background reference instead aims to show the layout of the scene. Foreground boxes and, when needed, object masks are combined before Big-LaMA (Suvorov et al., 2022) fills the covered region. The completed background is accepted only after checking for remaining foreground content, damaged structure, and obvious texture artifacts.

## 3.5.2 Pose tracking and asset states.

We use Detectron2 (Wu et al., 2019) with a ViTDet backbone (Li et al., 2022) to detect people and track their poses. People are detected in each frame and linked over time using box overlap, center movement, and changes in scale. Whole-body pose estimation then produces body and hand keypoints for the corresponding frames, while relative positions are preserved when several people appear together. A confirmed absence of people is recorded separately from a failed detector. Every asset is marked as available, not applicable, or blocked, which prevents missing outputs from being mistaken for valid ones.

## 3.5.3 Audio alignment.

We use MOSS-Transcribe-Diarize 0.9B (Yu et al., 2026) to transcribe speech, separate speakers, and estimate utterance times. A speaker cluster is linked to a visible person only when presence, mouth movement, and timing support the match. Dialogue keeps its original language and word order. We then use Qwen3-Omni-Thinking (Xu et al., 2025) to identify environmental sounds, contact sounds, nonverbal vocalizations, music within the scene, and background music, which are stored separately. This separation prevents the system from adding an expected sound merely because the corresponding action is visible.

## 3.6 High-quality multishot data

This final stage turns the selected clip and its coarse annotations into a detailed, evidence-based audiovisual description. The annotator reviews the video of no more than one minute together with its shot timeline, coarse captions, stable subject identities, and audio evidence. It describes composition, appearance, position, visible actions, state changes, camera motion, dialogue, and other sounds.

Dense audiovisual captions and validation. The output contains subject definitions, an overall summary, ordered shot descriptions, the soundscape, background music, and links to the prepared assets. Automatic checks require continuous shot numbering, increasing cut times within the video duration, valid boxes, defined subject references, ordered audio events, nonempty required fields, and matching input signatures. A truncated structured response may be repaired only at the formatting level; the repair step is not allowed to invent visual or audio facts. Any factual, temporal, or identity conflict returns the sample for review.

Condition assets and division of labor. Overall, our data pipeline follows a simple division of labor. Specialized models provide measurable evidence for cuts, text regions, speech times, and human poses. The multimodal model uses the full context to organize scenes, identities, events, and captions. Deterministic checks then ensure that all outputs refer to the same video, timeline, and subjects. This division makes the pipeline easier to audit and limits the spread of errors from one stage to the next.

## 3.7 Video Editing Dataset

Beyond general video data, we further construct and collect a large-scale video editing dataset. Training on this dataset substantially improves the model’s ability to follow textual instructions and perform text conditioned control.

## 4 Method

## 4.1 Overview

LynnReal-Omni is a native multimodal video generation framework built on a shared multimodal difusion transformer. It unifies text-to-video, image-conditioned generation, reference-guided generation, structural control, editing, and long-video generation within a single model, and accepts heterogeneous visual inputs such as appearance references, editable 3D renders, and game recordings. The method consists of five main components. First, a native multimodal backbone with task-specific representation provides a shared denoiser and unambiguous conditioning interfaces. Second, multitask flow training and few-step distillation enable eficient generation across tasks, with a Flash variant that reduces denoiser depth and token count. Third, long-video generation is supported through a fixed head-overlap interface and bounded history storage. Fourth, a lightweight decoder is distilled to shift the inference bottleneck away from decoding. Fifth, agentgenerated geometry and executable game controls supply inspectable spatial and motion guidance. Together, these components enable LynnReal-Omni to jointly generate video from heterogeneous multimodal conditions, producing coherent appearance, motion, interactions, and, when applicable, synchronized audio within a unified model. We describe each component below.

## 4.2 Native Multimodal Backbone and Task representation

The standard backbone contains 50 transformer blocks. Its residual width is 5,376, with 56 attention heads of dimension 128 and a feed-forward width of 14,336. Video latents have 24 channels and use $1 \times 2 \times 2$ patches. The audio stream has 32 input channels; text conditioning has dimension 5,120. The input/output projections and normalization-sensitive computations preserve their native precision conventions.

Each task packs modality tags, modality-specific row indices, 3D rotary positions and per-row noise times. Text and visual context are encoded before denoising. The denoiser then predicts the video and, when present, audio velocity in one forward. The video decoder reconstructs RGB frames from the denoised video rows; the audio decoder reconstructs the soundtrack separately.

![](images/e2108ae892ed74a9412b8f97614fe834cfe6b12bfe49d872079a2d8c79fba75e.jpg)  
Figure 3 Native layouts and execution. (a) Task-specific representation. (b) A shared denoiser preserves a positive temporal ofset between aligned controls and target rows. (c) Video and audio have diferent clean-time coordinates within the same four step. (d) long video generation jointly decodes one fixed head and five future latents, then removes the repeated boundary frame to deliver 17 new RGB frames. Audio participation follows the task’s training contract.

Image conditioning has two explicit interfaces. In the unified reference layout, supplied images occupy ordered picture-reference segments and an opening-frame instruction is semantic. The native keyframe layout instead encodes first and optional last images into the target’s keyframe partition. The release selects this layout through the model bundle’s conditioning contract; it is available to the full standard DiT as well as Flash. A native 768p standard-model check animates a harbor photograph while retaining the main scene layout. This interface distinction is not a guarantee of pixel-exact image reconstruction, and prompt formatting follows the selected contract.

A semantic video reference may retain its own resolution and provide appearance or motion cues. A pose sequence, editing source, depth sequence, or game render instead describes the target timeline frame by frame. These controls are sampled on the 24-fps model clock, resized onto the target canvas, and extended with their last valid frame if shorter than the requested target. Source frame rate, selected duration and any terminal padding must therefore be recorded; a padded tail is not new observed motion. The reference and target then have equal latent geometry and identical spatial grids. Their temporal positions satisfy

$$
\tau _ { i } ^ { \mathrm { t a r g e t } } - \tau _ { i } ^ { \mathrm { c o n t r o l } } = \Delta , \qquad \Delta > 0 ,\tag{1}
$$

with a constant ofset for every corresponding latent frame. The positive ofset keeps reference and generated tokens in separate domains; setting their absolute rotary coordinates equal changes the model’s conditioning semantics.

## 4.3 Multitask Training and Few-Step Distillation

Let z be a clean target latent and $\epsilon \sim \mathcal { N } ( 0 , I )$ . We use the clean-time convention

$$
z _ { t } = t z + ( 1 - t ) \epsilon , \qquad v ^ { \star } = z - \epsilon .\tag{2}
$$

The denoiser learns the velocity on the task’s valid target rows. Conditioning rows and padded target rows are excluded from the corresponding target loss. Task sampling supplies text generation, image conditioning,

reference controls, editing, and long video generation to the shared model. Dataset windows, caption text, modality masks, and temporal alignment are carried together so that visual controls cannot silently refer to a diferent interval from the target.

The video noise level is drawn from a shifted logistic-normal distribution. For $g \sim \mathcal { N } ( 0 , 1 )$ and $u =$ Sigmoid(g), we define the video noise level

$$
s _ { v } = \frac { 3 u } { 1 + 2 u } \in [ 0 , 1 ] , \qquad t _ { v } = 1 - s _ { v } ,\tag{3}
$$

where $t _ { v }$ denotes the corresponding clean time. This reparameterizes the time coordinate from noise level to clean time; the velocity target $v ^ { \star } = z - \epsilon$ remains unchanged. When valid target audio exists, its noise level follows the backbone’s aligned audio clock. We first map $s _ { v }$ to an intermediate level $s _ { b } = s _ { v } / ( 1 2 - 1 1 s _ { v } )$ , and then obtain the audio noise level

$$
s _ { a } = \frac { 3 s _ { b } } { 1 + 2 s _ { b } } , \qquad t _ { a } = 1 - s _ { a } .\tag{4}
$$

This aligned clock ensures that video and audio are noised according to their respective schedules while sharing the same random draw $g .$ Video and audio each use independently sampled Gaussian noise. The objective averages the squared velocity error over valid entries in each modality,

$$
\mathcal { L } = \mathrm { M S E } _ { \mathcal { V } } ( \widetilde { v } _ { v } , z _ { v } - \epsilon _ { v } ) + 0 . 1 I _ { \mathrm { v a l i d ~ a u d i o } } \mathrm { M S E } _ { A } ( \widetilde { v } _ { a } , z _ { a } - \epsilon _ { a } ) .\tag{5}
$$

Missing audio is not treated as a supervised silent recording.

The joint model also uses guidance-aware fitting. The same denoiser supplies a captionless, stop-gradient prediction $v _ { \emptyset }$ while retaining the supplied visual controls. For the recorded training scale $w = 3$ , the loss above uses

$$
\widetilde { v } = \frac { v _ { c } + ( w - 1 ) \mathrm { s g } ( v _ { \infty } ) } { w } .\tag{6}
$$

Only the conditional branch receives gradients. At its regression optimum this encourages $v _ { c } \approx w v ^ { \star } - ( w -$ $1 ) v _ { \emptyset }$ . Inference uses the conditional prediction directly; it does not add an unconditional forward to each denoising step. Cached target video latents use the posterior mode, while reference encoding retains its native posterior convention.

The standard model retains the full backbone and adopts low-rank trajectory distribution matching (Luo et al., 2025) for four-evaluation inference. Training proceeds over four non-overlapping transition intervals, diferentiating through one selected student transition. A frozen teacher and a separately trained full-depth fake-score critic evaluate noise conditioned on the student endpoint. The critic is trained with importanceweighted clean-target regression under clipped signal-to-noise weighting, while the student uses the normalized pseudo-Huber surrogate described in Section 4.3.1. To improve the dynamics of distribution matching distillation (DMD), we incorporate the dynamic reward mechanism from Reward Forcing (Lu et al., 2026b). This visual reward reweights the video term of the combined video–audio surrogate, whereas the audio term receives a fixed weight because the reward model is not audio-aware. Deployment uses the student’s exponential moving average and requires neither the teacher nor the critic.

Although video and audio share the same four denoising evaluations, they are conditioned on distinct modality-specific noise schedules. Under the native 768P configuration, the video branch is evaluated at clean-time timesteps of approximately (0, 0.0270, 0.0769, 0.2000), whereas the corresponding audio timesteps are (0, 0.1000, 0.2500, 0.5000). These timestep values are determined by the modality-specific row-to-time mappings, rather than by directly indexing a shared timestep vector, since the same denoising evaluation can correspond to diferent noise levels for the two modalities. After the fourth denoising update, both modalities advance to their clean endpoints without requiring an additional network evaluation. Figure 3 illustrates how these four shared denoising evaluations are realized within the packed multimodal layout.

## 4.3.1 Flash variant and inference acceleration.

LynnReal-Omni-Flash accelerates text-to-video and native keyframe-conditioned generation through transformer depth reduction, spatial token compression, and three-step distillation. The model retains 42 of the 50 transformer blocks and requires only three denoising evaluations per sample.

To reduce intermediate computation while preserving multimodal conditioning, the first two transformer blocks operate on the full token sequence, enabling early integration of video, text, and audio features. The subsequent 26 blocks employ frame-wise spatial token compression: video tokens are subsampled with a stride of two along both spatial dimensions, reducing the spatial token count to approximately one quarter, while retaining boundary rows and columns to preserve edge information. Text and audio tokens remain uncompressed, and all retained video tokens preserve their original spatiotemporal rotary positional coordinates. This design reduces the dominant spatial computation without sacrificing temporal resolution or altering the multimodal token structure.

We preserve the full-resolution features through a residual connection. Let H denote the features entering the compressed stack, $P$ the token-selection operator, and $F _ { \mathrm { m i d } }$ the middle blocks. Before the final fourteen full-resolution blocks, we reconstruct

$$
H _ { \mathrm { o u t } } = H + U ( F _ { \mathrm { m i d } } ( P H ) - P H ) ,\tag{7}
$$

where $U$ assigns each omitted video token the feature update of its nearest retained spatial token in the same frame; retained tokens receive their own updates. This preserves the original full-resolution features while allowing the compressed stack to supply contextual updates. The final blocks then refine all tokens jointly. For $N _ { v } , N _ { t }$ , and $N _ { a }$ video, text, and audio tokens, the middle-stack sequence length decreases from $N _ { v } + N _ { t } + N _ { a }$ to approximately $N _ { v } / 4 + N _ { t } + N _ { a }$ . This reduces both token-wise projection and feed-forward computation, as well as the sequence-length-dependent attention cost.

We train the reduced student using trajectory distribution matching (Luo et al., 2025), with a frozen fulldepth teacher and a trainable full-depth fake-score critic. Each worker backpropagates through one selected transition of the student’s three-step trajectory. Teacher and critic predictions are evaluated at the same noisy state, sampled within that transition’s interval conditional on the student endpoint. The critic learns to predict generated clean targets using importance-weighted regression with clipped signal-to-noise weighting.

For the generator objective, let $z _ { G }$ be the student clean prediction and $\hat { z } _ { F } , \hat { z } _ { T }$ the corresponding critic and teacher predictions. Define $d = \hat { z } _ { F } - \hat { z } _ { T }$ and $a = \operatorname* { m a x } ( \operatorname* { m e a n } | z _ { G } - \hat { z } _ { T } | , 1 0 ^ { - 6 } )$ . We use

$$
\mathcal { L } _ { \mathrm { D M } } = \frac { \mathrm { m e a n } [ \rho _ { c } ( z _ { G } - \mathrm { s g } ( z _ { G } - d ) ) ] } { \mathrm { s g } ( a ) } , \qquad \rho _ { c } ( e ) = \sqrt { e ^ { 2 } + c ^ { 2 } } - c , \quad c = 1 0 ^ { - 3 } .\tag{8}
$$

Here sg denotes stop-gradient, and normalization is applied outside the robust penalty. Visual reward reweights only the video loss; the audio loss has a fixed weight. We additionally use real-data flow regression on aligned video and valid audio. Audio interval supervision matches the student’s update to the teacher’s finer integration over the same interval, using an RMS-normalized smooth- $. L _ { 1 }$ loss with a normal ization floor and separate validity masks for the stereo channels. These auxiliary computations are used only during training; inference requires three student evaluations followed by codec decoding.

The final Flash DiT adopts W4A8 quantization—INT4 weights with FP8 activations—to further reduce memory trafic and accelerate matrix multiplications, while precision-sensitive modules remain at higher precision. Operator fusion lowers intermediate memory accesses and kernel-launch overhead. We evaluate these optimizations independently of depth reduction and token compression to isolate their efects on latency and generation quality.

## 4.4 Long Video Generation with Bounded History

## 4.4.1 Chunk-wise video generation

Long videos are generated in fixed-length chunks. At each step, the model conditions on the preceding video context and generates 17 new frames. To maintain temporal coherence across adjacent chunks, the final RGB frame of the preceding chunk is reused as a shared boundary frame. Its latent remains fixed while the model predicts 5 future latent frames, which are jointly decoded with the boundary latent; the duplicated boundary frame is then removed. This design provides a consistent transition across chunks while preserving a fixed generation window. Each chunk requires 4 denoiser evaluations, without frame interpolation or exposure correction.

## 4.4.2 Compact temporal context

We construct a fixed-budget temporal context that preserves fine-grained recent dynamics while retaining coarse long-range information. Specifically, the earliest latent frame is kept at full spatial resolution, 2 recent latent frames are sampled with spatial stride 2, and up to 8 earlier frames are sampled with stride 4. The shared boundary frame is excluded to avoid redundant conditioning. Across the evaluated resolutions, the resulting context contains no more tokens than 2 full-resolution latent frames, keeping attention cost independent of video length.

The temporal context is also bounded in storage: once the capacity is reached, the initial frame and the most recent frames are retained while intermediate entries are discarded. Together, bounded storage and fixed-budget tokenization enable long-form generation without increasing the temporal conditioning cost.

## 4.5 Lightweight Decoder Distillation

We distill the original video decoder into a shallower student to reduce decoding cost while preserving the latent interface of the pretrained denoiser. Our approach combines teacher reconstruction, spatial and temporal supervision, and consistency through the frozen encoder. Paired source pixels provide additional detail supervision when a valid correspondence is available.

![](images/db688d84c2d3ba094f68dce7ea43ead22ca04b256bc0d493589ccd88a567a034.jpg)  
Figure 4 Lightweight decoder distillation. The teacher and student decode the same latent. Pixel, temporal, and feature losses transfer the teacher’s reconstruction behavior, while a frozen encoder constrains the distributions of their re-encoded outputs. Genuine paired source pixels provide additional detail supervision.

## 4.5.1 Architecture and initialization

The student decoder $D _ { S }$ is obtained by reducing the depth of the 36-block teacher decoder $D _ { T }$ to 26 transformer blocks while preserving the hidden dimension of 2,048 and an MLP expansion ratio of 4. Each retained student block is initialized from its corresponding teacher block, which also establishes the block-wise correspondence used for feature distillation. The retained teacher blocks are

$$
\begin{array} { r } { \mathcal { T } = \{ 0 , 1 , 3 , 4 , 6 , 7 , 8 , 1 0 , 1 1 , 1 3 , 1 4 , 1 5 , 1 7 , \phantom { - } } \\ { 1 8 , 2 0 , 2 1 , 2 2 , 2 4 , 2 5 , 2 7 , 2 8 , 2 9 , 3 1 , 3 2 , 3 4 , 3 5 \} . } \end{array}\tag{9}
$$

The latent representation is left unchanged: the encoder, latent dimensionality, spatial scaling, and temporal structure are identical to those of the teacher model, and no spatial or temporal latent tokens are removed during decoding. The distilled decoder can therefore replace $D _ { T }$ directly without modifying the upstream generative model or re-encoding its latent outputs. The final decoder is obtained by further refinement of this initialized student.

## 4.5.2 Training distribution and supervision

Let E denote the frozen encoder. To expose $D _ { S }$ to both encoded data latents and the latent distribution encountered during generation, training samples are drawn from 2 sources. With probability 0.6, we use latents $z = E ( x )$ encoded from the training data; with probability 0.4, we use latents produced by the generative model. Within the data-encoded branch, native-HD and high-motion samples are each selected with probability 0.2. Native-HD samples have a short-side resolution of at least 720 pixels, while videos recorded $\mathrm { a t } \geq 2 0$ fps are densely sampled with probability 0.7. Each clip contains 2–33 frames under a budget of 4 million pixel-frames.

The supervision target is determined by the latent source. For a latent z, the default target is the frozen teacher reconstruction

$$
y _ { T } = D _ { T } ( z ) .\tag{10}
$$

Model-generated latents are supervised exclusively by $y _ { T }$ , as they have no paired pixel-space target. We further improve robustness to deviations from the encoder latent distribution by perturbing latents with probability 0.25 using Gaussian noise with a relative standard deviation of 0.03; these perturbed latents are likewise supervised by $D _ { T }$ . Single-frame latent slices are sampled with probability 0.1 to strengthen image decoding.

For image samples, we additionally use a frozen image-specialized decoder $D _ { I }$ . Generated image latents are supervised by $D _ { I }$ . For encoded training images, the teacher is selected according to reconstruction fidelity: $D _ { I }$ replaces $D _ { T }$ when its clamped reconstruction yields a lower pixel-space $L _ { 1 }$ error with respect to the input image. This adaptive teacher selection exploits the stronger image reconstruction capability of $D _ { I }$ without sacrificing fidelity to the original data.

## 4.5.3 Distillation objectives

Given a latent $z ,$ the student reconstruction is $y _ { S } ~ = ~ D _ { S } ( z )$ . Training combines teacher reconstruction, structural regularization, feature distillation, perceptual supervision, latent consistency, and, when paired data are available, direct pixel-space reconstruction:

$$
\begin{array} { r l } & { \mathcal { L } = \alpha \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { r e g } } + \lambda _ { \mathrm { f e a t } } \mathcal { L } _ { \mathrm { f e a t } } + \lambda _ { \mathrm { p e r c } } \mathcal { L } _ { \mathrm { p e r c } } } \\ & { \qquad + m _ { \mathrm { c y c } } \lambda _ { \mathrm { c y c } } \mathcal { L } _ { \mathrm { c y c l e } } + \lambda _ { \mathrm { s r c } } \mathcal { L } _ { \mathrm { s r c } } + m _ { \mathrm { i m g } } \lambda _ { \mathrm { L P I P S } } \mathcal { L } _ { \mathrm { L P I P S } } . } \end{array}\tag{11}
$$

The primary distillation term matches the student output to the selected teacher reconstruction:

$$
\mathcal { L } _ { \mathrm { r e c } } = \mathrm { m e a n } \left| y _ { S } - y _ { T } \right| + 5 \mathrm { m e a n } \left( y _ { S } - y _ { T } \right) ^ { 2 } .\tag{12}
$$

The $L _ { 1 }$ component preserves reconstruction accuracy, while the quadratic term places stronger emphasis on large pixel deviations. The regularization term

$$
\mathcal { L } _ { \mathrm { r e g } } = \sum _ { k } \lambda _ { k } \mathcal { R } _ { k }\tag{13}
$$

captures complementary spatial and temporal reconstruction properties. Multiscale, gradient, high-pass, wavelet, and SSIM terms constrain appearance, edges, and local detail; patch- and tile-boundary losses suppress decoding seams; temporal first- and second-order diferences regularize frame-to-frame motion and acceleration; and spatial- and temporal-phase terms reduce periodic reconstruction artifacts. Their default weights are summarized in Table 1.

Feature distillation further aligns the internal representations of the student and teacher. We apply an $L _ { 1 }$ loss to the final 3 student blocks and their corresponding teacher blocks, assigning a 4× weight to the final block. To reduce training cost, this loss is evaluated on a reduced latent region with spatial size 16 and at most 4 temporal tokens. In parallel, frozen DINOv2 ViT-S/14 features (Oquab et al., 2023), extracted at 224-pixel resolution, provide the perceptual objective $\mathcal { L } _ { \mathrm { p e r c } }$

Table 1 Default auxiliary loss weights. Source-repair samples use the target and coeficient adjustments described in the text.
<table><tr><td>Spatial loss</td><td>Weight</td><td>Temporal or feature loss</td><td>Weight</td></tr><tr><td>Multiscale pyramid</td><td>0.25</td><td>Temporal difference</td><td>0.20</td></tr><tr><td>Spatial gradient</td><td>0.10</td><td>Temporal acceleration</td><td>0.05</td></tr><tr><td>High-pass residual</td><td>0.15</td><td>Temporal phase</td><td>0.10</td></tr><tr><td>Haar wavelet bands</td><td>0.10</td><td>Decoder features</td><td>0.20</td></tr><tr><td>Wavelet energy</td><td>0.02</td><td>DINOv2 features</td><td>0.05</td></tr><tr><td>Patch boundary</td><td>0.20</td><td>Encoder consistency</td><td>0.01</td></tr><tr><td>Tile boundary</td><td>1.50</td><td></td><td></td></tr><tr><td>Spatial phase</td><td>0.05</td><td></td><td></td></tr><tr><td>SSIM</td><td>0.10</td><td></td><td></td></tr></table>

When paired training pixels are available, we additionally supervise the student directly with the input x using a Charbonnier loss:

$$
\mathcal { L } _ { \mathrm { s r c } } = \operatorname* { m e a n } \sqrt { \left( y _ { S } - x \right) ^ { 2 } + \epsilon ^ { 2 } } .\tag{14}
$$

This source-reconstruction branch is applied to eligible image and low-resolution video samples. For these samples, we use $\alpha = 0 . 2 5$ and $\lambda _ { \mathrm { s r c } } = 2$ , and the original pixels also serve as targets for the applicable detail losses. Image reconstruction additionally enables VGG LPIPS (Zhang et al., 2018) with weight 0.1 while disabling the wavelet-energy term. Outside this branch, $\alpha = 1$ . For encoded images assigned to the specialized image teacher, $\lambda _ { \mathrm { s r c } } = 1 ;$ otherwise, $\lambda _ { \mathrm { s r c } } = 0$ . Consequently, model-generated and perturbed latents receive teacher supervision without direct pixel-space reconstruction.

We further regularize the student in latent space by requiring teacher and student reconstructions to produce consistent encoder posteriors. A shared contiguous window of at most 5 frames is sampled from $y _ { T }$ and $y _ { S } .$ , clamped, resized to 256 pixels, and re-encoded by E. Let

$$
q _ { T } = \mathcal { N } \left( \mu _ { T } , \mathrm { d i a g } \sigma _ { T } ^ { 2 } \right) , \qquad q _ { S } = \mathcal { N } \left( \mu _ { S } , \mathrm { d i a g } \sigma _ { S } ^ { 2 } \right) .\tag{15}
$$

For each posterior coordinate, we compute the KL divergence

$$
k _ { i } = \frac { 1 } { 2 } \left[ \frac { \left( \mu _ { T , i } - \mu _ { S , i } \right) ^ { 2 } + \sigma _ { T , i } ^ { 2 } } { \sigma _ { S , i } ^ { 2 } } - 1 + \log \frac { \sigma _ { S , i } ^ { 2 } } { \sigma _ { T , i } ^ { 2 } } \right] ,\tag{16}
$$

and define

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c y c l e } } = \operatorname* { m e a n } _ { i } \log ( 1 + \operatorname* { m a x } ( k _ { i } , 0 ) ) . } \end{array}\tag{17}
$$

The logarithmic transformation limits the contribution of large divergences while preserving their gradients. Log variances are clamped to [−20, 10] for numerical stability. The teacher posterior is detached, whereas gradients are propagated through the frozen encoder to update the student reconstruction.

Importantly, $\mathcal { L } _ { \mathrm { c y c l e } }$ aligns the posterior of the student reconstruction with that of the teacher reconstruction, rather than directly matching the original input latent. The loss is activated after 800 training steps, evaluated every 8 updates, and disabled for source-reconstruction samples, for which the original pixels already provide direct supervision. “ ‘

## 4.5.4 Optimization and deployment

The refinement stage is optimized with AdamW using a learning rate of $3 \times 1 0 ^ { - 6 }$ , betas $( 0 . 9 , 0 . 9 5 ) , \ \epsilon \ =$ $1 0 ^ { - 8 }$ , zero weight decay, and gradient clipping at norm 1. Each worker accumulates gradients over 2 singlesample microbatches. Model parameters and accumulated gradients are maintained in FP32, while forward computation uses BF16 autocast. Training resumes with a 100-step learning-rate warmup.

To control memory usage during decoder training, videos are processed with rectangular tiles of $2 7 2 \times 2 0 8$ pixels, using overlaps of 0 and 16 pixels along the respective axes and a tile batch size of 2. Images use

256 × 256 tiles with a repeated 5-latent context and output phase 3. At inference time, we retain the original decoder tiling configuration by default, while adaptive tiling is optionally enabled for diferent memory– latency trade-ofs. Both settings preserve the native temporal padding and output trimming.

Checkpoints are selected using a fixed reconstruction benchmark that jointly evaluates spatial fidelity and temporal consistency. Decoder depth, numerical precision, and tiling strategy are additionally ablated independently to isolate their efects on reconstruction quality and inference latency.

## 4.6 Temporal repair of generated-video artifacts

LynnReal-Omni supports instruction-guided repair of generated-video artifacts through its shared imageediting interface. Given a corrupted video and a repair instruction, the model processes each source frame and reassembles the repaired images in their original temporal order. The instruction specifies which artifacts to suppress and which scene content to retain. This procedure uses the same standard DiT across frames and does not require a scene-specific restoration network. The implementation accepts an external video and prompt; the candle sequence below is an illustrative example rather than a restriction of the repair interface.

General framewise repair. For a source sequence {x} and instruction p, we independently edit each $x _ { t }$ with the shared model, using the same seed across frames, and retain one generated image per source timestamp. The current implementation produces a short internal clip for each image edit and selects a configurable frame from that clip. This accommodates the video decoder without adding frames to the source timeline. Optional appearance references or temporally aligned guide videos can also be supplied. Each source-frame edit uses four denoiser evaluations, giving a total sampling cost of $4 T$ evaluations for T source frames. Temporal repair here denotes restoration across the complete sequence with its frame order and playback timing retained; the frame edits are independent and impose no explicit cross-frame consistency constraint.

![](images/5c05bf85e3a882939e8e093ebd6249ab88078127ee1e2cb7e8dfe6cd8d727588.jpg)  
Figure 5 An example of generated-video artifact repair with LynnReal-Omni. Top: corrupted candle video. Bottom: independently repaired frames at identical timestamps, including both endpoints. Full frames are displayed at the same scale. Each source-frame edit uses four denoiser evaluations, for 480 evaluations across this 120-frame example. The accompanying demo plays both complete videos synchronously.

Illustrative result. We apply this general procedure to a five-second candle video containing bright speckles, background grid patterns, and vertical rendering artifacts. The example uses the standard BF16 DiT, seed 7, and a 1344×768 output canvas. Its prompt requests artifact removal while retaining the observed candle and flame; its configuration selects the final frame (index 21) of each 22-frame internal clip. All 120 source frames are independently repaired, requiring 480 denoiser evaluations in total. This selected run uses the raw source frames without an appearance reference, filtered guide, generated-frame feedback, temporal interpolation, or subsequent restoration filter.

Full-sequence assessment. Inspection of all 120 repaired frames shows substantially fewer background grid artifacts and speckles than the corrupted source and the preceding joint video-edit trial, while the flame continues changing over time. Small brightness and shape fluctuations, an occasional residual point, and reduced flame-height variation remain. Complete synchronized videos and fixed-mask flame-tip and backgroundbright-pixel traces accompany the example. These observations provide qualitative evidence for this unpaired case, without clean ground truth; they do not establish pixel-exact recovery or restoration quality across arbitrary scenes. Temporal consistency remains a limitation of independent frame editing.

## 4.7 Agent-generated 3D scene and executable game controls

We use agents to construct executable 3D scenes from reference images and task descriptions, providing spatial and motion guidance for subsequent video generation. The scene representation combines geometry, materials, object transforms, lighting, and cameras with explicit state updates and control inputs. Code execution, rendered observations, and iterative refinement connect these components, allowing appearance, motion, and interaction to be specified and revised within a shared editable representation.

Image Analysis and Scene Construction. Scene construction begins with an analysis of the reference image and task description. The image provides evidence for object silhouettes, relative scales, occlusions, and support relationships, while the task description specifies the required actions and interactions. These observations inform the environment, object models, appearance, and control logic. Shared coordinate conventions and interfaces support independent development and revision of each component. Objects and their parts are represented by meshes and hierarchies, with transforms specifying position, orientation, and scale. Materials, procedural textures, and lighting define surface appearance. The agent implements the main structures and basic behavior before refining object shape and environmental detail, using execution feedback to identify and correct implementation errors. Absolute scale and hidden geometry are resolved through modeling assumptions consistent with the visible evidence.

Pose and Camera Refinement. Rendered views from the reference viewpoint guide the refinement of camera parameters and object transforms. Comparisons with the input image focus on framing, silhouettes, perspective, and occlusion, with each update evaluated through a new render. Correspondence fitting can assist parameter estimation within this process, while image feedback guides refinement across scenes. Refinement proceeds from the overall viewpoint and object arrangement to part proportions, materials, and lighting, reducing the risk that local geometry changes compensate for an incorrect perspective. Assembly transformations preserve internal relationships, and pose updates are accompanied by renewed checks of contact and occlusion. Code or configuration snapshots and rendered views are retained to support comparison and revision. These comparisons provide construction feedback rather than an independent measure of recovered 3D accuracy.

Contact Correction and Scene Validation. Scene validation combines appearance assessment with checks of spatial consistency. Reference-view renders support comparisons of composition, silhouettes, and materials, while additional viewpoints reveal hidden geometry, support relationships, and intersections. Geometric measurements and collision checks localize gaps and overlaps for correction through object height, assembly transforms, or local geometry. Corrections are followed by checks of nearby objects to account for their efects on surrounding contacts and occlusions. Runtime checks accompany visual refinement to verify that changes preserve object hierarchies, animation, and rendering performance. Shared geometry, instancing, and static merging reduce rendering cost where applicable. The resulting scene retains editable objects, materials, and camera settings and is validated through loading and inspection from multiple viewpoints. Runtime behavior, geometric validity, and visual fidelity are recorded as separate aspects of validation.

Game Logic and Temporal Control. Game logic defines motion and interaction through explicit state updates. The state includes object position, velocity, orientation, and action phase, while control inputs specify movement, turning, and action triggers. Update rules account for acceleration, gravity, ground contact, and collisions and produce the corresponding interaction events. Object poses, part animations, and cameras are updated from this state to generate visible motion. Automated control selects inputs according to the current state and task goals through the same update logic used for manual control. Initial conditions, contro policies, and behavior parameters therefore provide direct means of revising the resulting motion.

The output sequence is determined jointly by the simulation and camera behavior. Cameras follow configured trajectories or tracking rules, and a recording script advances the simulation at fixed output time intervals, capturing each frame after rendering completes. States and interaction events are stored alongside the frames, linking visible motion to program execution and decoupling the time required to record the sequence from its playback speed. The encoded video is reviewed for appearance, motion continuity, and the intended actions, with the corresponding program version, control settings, and execution records retained. This rendered sequence supplies layout and motion references for subsequent video generation, while the underlying program supports further edits to appearance and behavior.

## 5 Experiments

## 5.1 Experimental questions and settings

Our experiments address three distinct questions: whether the released sampler preserves its reference computation, which decoder and execution changes improve measured eficiency, and how visual control behaves over time. Table 2 specifies the unit of evidence for each study. Within a paired comparison, we hold source media, prompt, seed and the relevant model artifact fixed. Between studies, changes in output geometry, decoder context or arithmetic are stated explicitly. This organization avoids treating a reconstruction measurement as a generated-video score, or transferring a small-canvas timing to a larger-canvas demonstration.

<table><tr><td>Study</td><td>Geometry and execution</td><td>Controlled comparison and interpretation</td></tr><tr><td>Native inference</td><td>1344×768; 124 frames; four evaluations; full decoder</td><td>Fixed prompts, seeds, checkpoint and adapter bytes. Tests release/reference fidelity and visible event structure.</td></tr><tr><td>Decoder reconstruction</td><td>1344×768; one image, 22- and 124-frame clips</td><td>Shared official-encoder latent; matched image context. Isolates decoder depth and tile scheduling; metrics precede encoding.</td></tr><tr><td>Warm latency</td><td>960×544 native canvas, cropped to 540p; 22 frames</td><td>Synchronized repeated timings, two warmups, five retained calls. Codec, attention and precision are stated per row.</td></tr><tr><td>Continuation</td><td>720 new frames; 43 chunks; four evaluations per chunk</td><td>Bounded stored history and observed memory; paired scene/seed controls. Visual drift is assessed separately from memory growth.</td></tr><tr><td>MSAVP benchmark</td><td>100 prompts; 20 reported metrics; 720p, 15-s outputs</td><td>Cross-model scores are reported separately for T2V and I2V.</td></tr></table>

Table 2 Evaluation settings and the question supported by each experiment. Individual artifacts retain the full invocation and hashes. Development cases are reused for diagnosis and are not a held-out estimate of general quality.

## 5.2 MSAVP: complex audiovisual and physical evaluation

## 5.2.1 Benchmark overview

MultiShot-AV-Physics Bench. We introduce the MultiShot-AV-Physics Bench (MSAVP) for multishot audiovisual generation, a setting covered by relatively few existing benchmarks. MSAVP emphasizes instruction following, cross-shot consistency, physical plausibility, and audiovisual coordination. Its 20 reported metrics are summarized in Table 5; the quality criteria draw on WBench (Ying et al., 2026), while the physics criteria are informed by VideoPhy (Bansal et al., 2025) and PhyGDPO (Cai et al., 2026). Unlike conventional evaluation pipelines, MSAVP employs an agentic VLM to plan and execute the assessment, enabling structured semantic reasoning over complex requests. A fixed checklist for each prompt constrains this reasoning to explicit, verifiable conditions, making the evaluation more consistent and auditable.

Prompt construction. The benchmark tests dense events and cross-shot consistency. We do not adapt prompts to the outputs of any evaluated model. We constructed 100 prompts through text review and deterministic coverage search, using a common format for all systems. Each prompt contains an integrated audiovisual description, an overall soundscape, and an optional description of non-diegetic music. Twentytwo prompts request music, while the remaining 78 do not. The set includes 54 single-shot prompts and 46 multi-shot prompts: 10 request two shots, 7 request three, 9 request four, 16 request five, and 4 request six.

The prompts also cover a broad range of content rather than concentrating on a single failure mode. They span 99 overlapping topic or capability tags and 95 distinct primary topics, including daily activities, sports, machinery, material deformation, natural events, fantasy, science fiction, and demanding camera work.

## 5.2.2 Evaluation workflow

Inputs and generation settings. T2V and I2V receive identical text prompts. In I2V, every model also receives the same 1280×720 RGB opening image for a given prompt. Each system produces a 15-second, 16:9 video at 1280×720 resolution and 24 fps. Seedance 2.0 is evaluated only in T2V because the oficial API’s strict moderation of photorealistic human input images prevented completion of a comparable 100-prompt I2V run.

For the locally executed systems, LynnReal-Omni uses our FL2VA checkpoint for both T2V and I2V. LTX-2.5 uses the oficial DFR pipeline with BF16 weights, runtime FP8 casting, CPU ofloading, chunkedeager DifVAE decoding, and one spatial upsampling stage on one 80-GB H100. It generates 361 frames at 1280×704 and 24 fps, with 8-pixel vertical padding on each side for delivery at 1280×720. Cosmos3-Super uses the complete non-distilled BF16 I2V model for 35 steps with CFG 6 and shift 10, following the oficial four-H100 Ray/FSDP configuration with difusion caching, torch.compile, and CUDA Graphs. MiniMax-H3 uses the complete unquantized SGLang fl2va model without Turbo LoRA, running 50 steps with flow shift 12, audio flow shift 3, tensor parallelism 2, Ulysses parallelism 2, and the speed performance mode on four 80-GB H100s. Its 15-second 16:9 output is generated at a short side of 768 pixels and proportionally resized or padded to the common 1280×720, 24-fps delivery format.

Table 3 MSAVP family and overall scores. Metrics are averaged within each capability family, followed by an equal average of the six family scores.
<table><tr><td>Mode</td><td>Model</td><td>Prompt</td><td>Event</td><td>Visual</td><td>Multi-shot</td><td>World</td><td>Audio</td><td>Overall</td></tr><tr><td>T2V</td><td>Seedance 2.0</td><td>83.79</td><td>74.62</td><td>61.53</td><td>84.50</td><td>87.87</td><td>84.01</td><td>79.39</td></tr><tr><td></td><td>MiniMax-H3-FL2V</td><td>82.06</td><td>73.52</td><td>59.13</td><td>86.75</td><td>88.79</td><td>85.35</td><td>79.27</td></tr><tr><td></td><td>LynnReal-Omni</td><td>83.69</td><td>72.32</td><td>59.01</td><td>83.25</td><td>86.00</td><td>82.26</td><td>77.76</td></tr><tr><td></td><td>LTX-2.5</td><td>72.02</td><td>59.51</td><td>54.52</td><td>30.08</td><td>74.13</td><td>73.07</td><td>60.56</td></tr><tr><td></td><td>Cosmos3-Super</td><td>65.42</td><td>52.64</td><td>55.79</td><td>5.76</td><td>74.72</td><td>71.25</td><td>54.26</td></tr><tr><td>I2V</td><td>MiniMax-H3-FL2V</td><td>85.16</td><td>74.44</td><td>61.53</td><td>87.36</td><td>91.32</td><td>84.11</td><td>80.65</td></tr><tr><td></td><td>LynnReal-Omni</td><td>86.55</td><td>74.73</td><td>61.16</td><td>81.81</td><td>89.52</td><td>81.44</td><td>79.20</td></tr><tr><td></td><td>LTX-2.5</td><td>77.31</td><td>61.32</td><td>56.00</td><td>37.29</td><td>77.24</td><td>75.47</td><td>64.11</td></tr><tr><td></td><td>Cosmos3-Super</td><td>71.07</td><td>54.77</td><td>56.62</td><td>8.96</td><td>74.51</td><td>72.00</td><td>56.32</td></tr></table>

Step 1: freeze the scoring checklist. The first step converts each prompt into explicit, reusable scoring items. We use GPT-6 Astra through Codex with medium reasoning for the model-based evaluation stages because its visual reasoning supports more general and reliable semantic judgments than narrowly designed specialist models. To construct the base checklist, Astra reads the T2V prompt without seeing any generated video and returns a structured set of conditions. For I2V, it adds only conditions that are directly visible in the shared opening image. This image-derived overlay may introduce opening-state constraints for entities and attributes, cross-shot identity, and spatial or world state, but it cannot redefine action bindings, events, temporal relations, materials, or audio requirements.

Freezing the checklist ensures that every model receives the same test. All candidate models evaluated under the same prompt and generation mode use this exact set. After viewing a candidate video, the evaluator may judge whether a condition is satisfied, but it cannot merge, delete, rewrite, or invent conditions.

![](images/7a8ca8237bf79bd48ce55d65fd08eacff1d5f5c3154d83a283d45ac68da8cdd8.jpg)  
Figure 6 Per-metric MSAVP profiles for T2V and I2V. All 20 reported metrics are displayed. The radial scale begins at 20 to reveal diferences in the dense upper range; values below 20 are clipped to the center. Seedance 2.0 appears only in the T2V panel.

Step 2: collect specialist evidence. The second step extracts measurements that are more reliable when produced by dedicated tools. TransNetV2 (Soucek and Lokoc, 2024) proposes shot boundaries at a threshold of 0.1, after which Codex examines the video and rejects flashes, occlusions, and rapid camera motion that resemble cuts.

The audio specialist reports what is actually audible and when each sound occurs. Qwen3-Omni-30B-A3B-Thinking (Xu et al., 2025) reads the original soundtrack with 4-fps visual sampling. It identifies dialogue, interaction sounds, ambience, and music without treating a visible event as evidence that the corresponding sound is present. Following WBench (Ying et al., 2026), we report Aesthetic Quality using CLIP ViT-L/14 (Radford et al., 2021) with the LAION linear aesthetic head (LAION-AI, 2022), and Imaging Quality using MUSIQ (Ke et al., 2021); both are averaged over frames sampled at 2 fps. We additionally apply MSS, the VMBench Motion Smoothness Score (Ling et al., 2025), to each detected shot and aggregate the resulting scores by evaluated frames. Facial-100 is our percentage-scale name for the AVGen-Bench Facial Consistency metric (?); following that benchmark, it uses InsightFace features with face tracking and identity clustering to measure the stability and expected count of detectable primary faces. Videos without an applicable detected face are excluded from this metric rather than treated as failures.

Step 3: run the prompt-blind judge. The third step isolates judgments that should depend only on the generated video. A separate Codex context receives the candidate video and proposed cut locations, but not the prompt, opening image, model name, audio analysis, or other scores. It evaluates action continuity across the observed cuts. Consequently, a requested action that never appears remains a semantic failure rather than being counted again as a physical violation.

Step 4: run the main judge. The fourth step scores the 15 checklist-based metrics against the frozen checklist and specialist evidence. All condition-based metrics are expressed on a 0–100 scale. This judge context receives the candidate video, original prompt, applicable checklist, and specialist outputs, but not the promptblind judgment. The judge can inspect the native video, revisit short intervals, and examine adjacent frames when small objects, physical contact, material response, or cut boundaries require closer observation. Its output contains one decision and a short supporting evidence interval for every applicable condition.

Table 4 Per-metric MSAVP results. Action Binding averages its component and complete-binding subscores. Shot Structure, Cross-shot Identity, and Cross-cut Action use only the 46 multi-shot prompts; Material Behavior pools applicable conditions within each video before averaging the resulting per-video scores.
<table><tr><td rowspan="2">Metric</td><td colspan="5">T2V</td><td colspan="4">I2V</td></tr><tr><td>Seed.</td><td>H3</td><td>Lynn</td><td>LTX</td><td>Cosmos</td><td>H3</td><td>Lynn</td><td>LTX</td><td>Cosmos</td></tr><tr><td colspan="10">Prompt Fidelity</td></tr><tr><td>Camera Control</td><td>87.06</td><td>83.99</td><td>84.78</td><td>72.00</td><td>61.35</td><td>86.46</td><td>87.89</td><td>76.92</td><td>67.66</td></tr><tr><td>Event Completion</td><td>72.60</td><td>69.90</td><td>72.11</td><td>57.17</td><td>47.28</td><td>72.39</td><td>74.71</td><td>60.03</td><td>51.19</td></tr><tr><td>Entity Fidelity</td><td>91.70</td><td>92.30</td><td>94.19</td><td>86.90</td><td>87.63</td><td>96.62</td><td>97.06</td><td>94.99</td><td>94.37</td></tr><tr><td colspan="10">Event Execution</td></tr><tr><td>Temporal Order</td><td>54.80</td><td>50.45</td><td>50.25</td><td>33.06</td><td>21.50</td><td>52.43</td><td>53.60</td><td>36.05</td><td>26.28</td></tr><tr><td>Action Binding</td><td>74.77</td><td>73.82</td><td>73.94</td><td>60.81</td><td>53.51</td><td>75.93</td><td>77.30</td><td>65.19</td><td>57.64</td></tr><tr><td>Action Logic</td><td>94.30</td><td>96.28</td><td>92.78</td><td>84.65</td><td>82.90</td><td>94.97</td><td>93.30</td><td>82.71</td><td>80.38</td></tr><tr><td colspan="10">Visual Quality</td></tr><tr><td>Motion Smoothness</td><td>86.10</td><td>81.30</td><td>84.23</td><td>82.90</td><td>78.49</td><td>85.96</td><td>83.71</td><td>85.50</td><td>77.46</td></tr><tr><td>Aesthetic Quality</td><td>58.07</td><td>54.77</td><td>57.28</td><td>55.87</td><td>52.83</td><td>56.32</td><td>57.72</td><td>55.36</td><td>55.01</td></tr><tr><td>Facial Consistency</td><td>35.27</td><td>32.40</td><td>23.18</td><td>16.06</td><td>24.68</td><td>35.64</td><td>31.78</td><td>18.53</td><td>28.49</td></tr><tr><td>Imaging Quality</td><td>66.67</td><td>68.07</td><td>71.35</td><td>63.25</td><td>67.16</td><td>68.18</td><td>71.42</td><td>64.60</td><td>65.53</td></tr><tr><td colspan="10">Multi-shot Coherence</td></tr><tr><td>Shot Structure</td><td>66.94</td><td>67.71</td><td>64.00</td><td>26.33</td><td>14.03</td><td>70.72</td><td>65.53</td><td>29.24</td><td>12.15</td></tr><tr><td>Cross-cut Action</td><td>93.62</td><td>93.41</td><td>91.97</td><td>32.32</td><td>1.09</td><td>94.96</td><td>88.22</td><td>43.39</td><td>8.89</td></tr><tr><td>Cross-shot Identity</td><td>92.93</td><td>99.15</td><td>93.77</td><td>31.60</td><td>2.17</td><td>96.39</td><td>91.67</td><td>39.25</td><td>5.82</td></tr><tr><td colspan="10">World Plausibility</td></tr><tr><td>World-state Consistency</td><td>71.65</td><td>73.82</td><td>74.47</td><td>54.07</td><td>53.86</td><td>80.68</td><td>81.65</td><td>65.97</td><td>64.70</td></tr><tr><td>Contact Response</td><td>96.64</td><td>96.47</td><td>92.04</td><td>87.07</td><td>85.27</td><td>96.23</td><td>92.92</td><td>84.53</td><td>80.76</td></tr><tr><td>Material Behavior</td><td>95.33</td><td>96.09</td><td>91.49</td><td>81.24</td><td>85.03</td><td>97.07</td><td>94.00</td><td>81.23</td><td>78.06</td></tr><tr><td colspan="10">Audio Alignment</td></tr><tr><td>Sound Semantics</td><td>79.69</td><td>85.11</td><td>79.38</td><td>75.82</td><td>70.67</td><td>83.30</td><td>77.50</td><td>77.87</td><td>68.99</td></tr><tr><td>Speaker-lip Sync</td><td>99.47</td><td>98.44</td><td>99.00</td><td>99.08</td><td>96.83</td><td>99.50</td><td>98.50</td><td>98.92</td><td>95.94</td></tr><tr><td>Action-sound Timing</td><td>57.56</td><td>57.86</td><td>51.85</td><td>46.34</td><td>61.96</td><td>54.14</td><td>51.43</td><td>46.83</td><td>60.56</td></tr><tr><td>Cross-cut Audio Continuity</td><td>99.30</td><td>100.00</td><td>98.83</td><td>71.05</td><td>55.56</td><td>99.50</td><td>98.33</td><td>78.25</td><td>62.50</td></tr></table>

Step 5: validate and aggregate. The final step makes the model output mechanically checkable. After validating the item-level decisions, the pipeline computes all aggregate scores directly rather than accepting totals supplied by the judge.

Scoring and reporting. Condition-based metrics use three observable outcomes. For condition k of metric m on video i, $c _ { i m k }$ is 1 for satisfied, $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ for substantively partially satisfied, and 0 for failed. The metric score is

$$
s _ { i m } = \frac { 1 0 0 } { | \mathcal { C } _ { i m } | } \sum _ { k \in \mathcal { C } _ { i m } } c _ { i m k } .\tag{18}
$$

Partial credit requires visible partial success and cannot be used merely to represent uncertainty. A requested event that never appears remains a failed semantic condition. Shot structure, cross-shot identity, and crosscut action continuity are evaluated only on the 46 prompts that explicitly request multiple shots. If such a prompt produces a single-shot video, cross-shot identity and cross-cut action continuity receive zero, while shot structure is scored against its requested shot conditions. If the generated video has multiple shots but no action spans a cut, action continuity receives 100 because no cross-cut action discontinuity is present. The 54 single-shot prompts do not enter the model-level denominators of these three multi-shot metrics. A predeclared case without dialogue receives 100 for speaker–lip synchronization and remains distinct from an evaluator failure. For cross-cut audio continuity, a single-shot caption receives 100, whereas a caption that requests multiple shots receives 0 if the generated video remains single-shot. Component compliance and complete actor–action–target binding compliance are averaged into one binding metric. For material behavior, all applicable material conditions within each video are pooled into a per-video score, and the model-level metric is the mean of these per-video scores.

Table 5 The 20 reported MSAVP metrics, grouped into six equally weighted capability families. Condition-based metrics use a 0–100% completion score, while specialist metrics retain their native outputs on a 0–100 scale.
<table><tr><td>Metric</td><td>What is checked</td></tr><tr><td colspan="2">Prompt Fidelity</td></tr><tr><td>Camera Control</td><td>Requested viewpoint, camera motion, framing, focus, and time effects.</td></tr><tr><td>Event Completion</td><td>Whether each requested event reaches its visible outcome instead of stopping at prepa- ration.</td></tr><tr><td>Entity Fidelity</td><td>Requested objects, counts, visible attributes, text, symbols, and applicable first-frame constraints.</td></tr><tr><td colspan="2">Event Execution</td></tr><tr><td>Temporal Order</td><td>Explicit before, after, simultaneous, and during relations between events.</td></tr><tr><td>Action Binding</td><td>Correct components and complete actor-action-target tuples, including a specified body part.</td></tr><tr><td colspan="2">Action Logic</td></tr><tr><td>Visual Quality</td><td>Whether observed actions start, respond, progress, and end coherently.</td></tr><tr><td>Motion Smoothness</td><td>Frame-weighted VMBench MSS over the detected shots (Ling et al., 2025).</td></tr><tr><td>Aesthetic Quality</td><td>WBench Aesthetic Quality: mean CLIP-LAION aesthetic prediction from frames sam- pled at 2 fps (Ying et al., 2026)</td></tr><tr><td>Facial Consistency</td><td>Stability and expected count of detectable primary faces under Facial-100 (?).</td></tr><tr><td>Imaging Quality</td><td>WBench Imaging Quality: mean MUSIQ perceptual-quality prediction from frames sampled at 2 fps (Ying et al., 2026).</td></tr><tr><td colspan="2">Multi-shot Coherence</td></tr><tr><td>Shot Structure</td><td>Requested shot count, content allocation, order, and event-triggered cuts.</td></tr><tr><td>Cross-cut Action</td><td>Action phase, contact, held objects, and direction across each applicable observed cut.</td></tr><tr><td>Cross-shot Identity</td><td>Identity of people, animals, and key objects across shots.</td></tr><tr><td colspan="2">World Plausibility</td></tr><tr><td>World-state Consistency</td><td>Object permanence, lasting interaction results, layout, and connected space.</td></tr><tr><td>Contact Response</td><td>Contact timing, force response, collision direction, penetration, and action at a dis- tance.</td></tr><tr><td colspan="2">Material Behavior</td></tr><tr><td>Audio Alignment Sound Semantics</td><td></td></tr><tr><td>Speaker-lip Sync</td><td>Requested dialogue, interaction sounds, ambience, music, and silence. Correct speaker, absence of false lip motion, and visible speech alignment.</td></tr><tr><td>Action-sound Timing</td><td>Whether a sound belongs to the same visible event and occurs at the expected time.</td></tr><tr><td>Cross-cut Audio Continuity</td><td></td></tr><tr><td></td><td>Dialogue, voice, and acoustic continuity or justified change at each observed cut.</td></tr></table>

We first average metrics within six capability families: Prompt Fidelity (three metrics), Event Execution (three), Visual Quality (four), Multi-shot Coherence (three), World Plausibility (three), and Audio Alignment (four), exactly as grouped in Table 5. Prompt Fidelity measures requested entities, completed events, and camera control. Event Execution combines actor–action–target binding, temporal relations, and the internal logic of observed action chains. Visual Quality combines frame-level aesthetics and imaging quality with motion smoothness and facial consistency. Multi-shot Coherence is restricted to explicitly multi-shot prompts and measures requested shot structure, entity identity, and action continuity across cuts. World Plausibility measures persistent world state, contact response, and material behavior. If $\mathcal { M } _ { f }$ denotes the metrics in family $f ,$ its score is $g _ { f } = | \mathcal { M } _ { f } | ^ { - 1 } \sum _ { m \in \mathcal { M } _ { f } } \bar { s } _ { m }$ . The overall score gives every family equal weight,

$$
\mathrm { M S A V P } = \frac { 1 } { 6 } \sum _ { f \in \mathcal { F } } g _ { f } , \qquad | \mathcal { F } | = 6 .\tag{19}
$$

This hierarchy prevents a family from receiving more weight merely because it contains more metrics.

We report MSAVP scores for our real-time world model LynnReal-Omni, Seedance 2.0 (ByteDance Seed Team, 2026), MiniMax-H3 (MiniMax, 2026), LTX-2.5 (Lightricks, 2026), and Cosmos3-Super (NVIDIA, 2026), keeping T2V and I2V results separate. Aesthetic quality and imaging quality retain their native specialist scales. GPT-6 Astra through Codex constructs the frozen checklists and supplies the model-based judgments.

## 5.2.3 Benchmark results

Table 3 reports the six family scores and their equal-weight average. In T2V, Seedance 2.0 obtains the highest overall score at 79.39, narrowly ahead of MiniMax-H3 at 79.27. LynnReal-Omni reaches 77.76 in T2V and 79.20 in I2V, 1.63 points behind the best T2V system and 1.45 points behind MiniMax-H3 in I2V. Unlike the ofline generators in the comparison, LynnReal-Omni is designed as a real-time world model. It leads I2V Prompt Fidelity and is particularly strong in entity fidelity, world-state consistency, aesthetic quality, and imaging quality. These results show that its low-latency interactive design retains competitive instruction and visual-world fidelity rather than trading them away wholesale for speed. Seedance leads Prompt Fidelity,

Event Execution, and Visual Quality in T2V while MiniMax-H3 remains strongest in overall I2V, Multi-shot Coherence, World Plausibility, and Audio Alignment. LTX-2.5 and Cosmos3-Super underperform across most instruction-intensive and multi-shot measures. The per-metric profiles are visualized in Figure 6; Table 4 gives every reported metric.

## 5.3 Lightweight VAE Evaluation

We evaluate native-768P reconstruction quality using the oficial decoder and the selected EMA lightweight decoder with native or adaptive tiling. Visual comparisons use matched source frames and crop coordinates, while quantitative metrics are computed from uncompressed RGB outputs before file encoding.

Source pixels  
Official decoder  
Light / native tiles  
![](images/e95cecd5e1573168ed1704c4146a74b0695be5cdbf1698a668b5ce7714e7b5ea.jpg)  
Video panels are decoded from the saved MP4 files; all reported numerical errors were measured before file encoding.

Figure 7 Native-768P reconstruction comparisons. The urban image and its enlarged crop use the same five-token image context across methods. The clay-video crop uses identical source frames and spatial coordinates. No visual enhancement is applied. Video panels are extracted from saved MP4 files; quantitative metrics are computed before file encoding.

Spatial fidelity. Figure 7 shows that the lightweight decoder preserves the overall scene layout, clothing patterns, and table textures, although fine details can be softened. Its higher reconstruction PSNR on the urban image indicates lower pixel error for that example, but does not establish better perceptual quality on generated videos. The enlarged crops complement aggregate metrics by revealing local diferences in texture and edge sharpness.

Temporal fidelity. Figure 8 exposes framewise variations that clip-averaged scores can obscure. On the motion clip, temporal-diference MAE is nearly identical for the oficial and native-tile lightweight decoders: 0.006015 and 0.006012, respectively. On the clay clip, the oficial decoder achieves lower error, 0.005816 versus 0.005926. Adaptive lightweight decoding yields 0.006021 and 0.006059 on the two clips, respectively, slightly above the corresponding native-tile results.

![](images/1fc452a313d75cfe258752c12f93cadfb0572229293934424c4f10ef7d793792.jpg)  
Same vertical scale in both panels. Lower framewise error does not by itself establish perceptual or physical consistency.

Figure 8 Per-frame PSNR and temporal-diference MAE on two native-768P video inputs. All decoded frames are included, with consistent vertical ranges across clips for each metric. Curves and aggregate values are computed from uncompressed RGB reconstructions against the same source videos.

The framewise curves follow broadly similar trends, including a late quality drop on the clay clip shared by all three configurations. This behavior suggests that the observed fluctuations cannot be attributed solely to reduced decoder depth. Overall, native-tile lightweight decoding maintains temporal reconstruction errors close to the oficial decoder on these inputs, with remaining diferences in fine-detail preservation. These results characterize reconstruction fidelity;

## 5.4 Inference Latency

We measure warm inference on single H100 80GB GPUs, using two warmup calls and five measured calls per configuration, repeated on a second GPU. Prompt and seed are fixed within each setting. Standard and Flash use four and three denoiser evaluations, respectively. CUDA events measure the DiT and video decoder; synchronized generation wall time additionally includes scheduling, audio decoding, RGB conversion, and output transfer. Conditioning, loading, file encoding, and warmup compilation or tuning are excluded. Tables report pooled medians of ten calls.

Component ablations. Table 6 isolates successive execution changes at 540p and 22 frames. Each model’s complete sequence of configurations runs in one process on the same GPU. The standard baseline uses BF16; Flash retains its trained INT8 checkpoint throughout. QKV concatenation is measured separately before standard-model quantization. Subsequent rows add one component to the preceding configuration. The lightweight decoder row includes its batched implementation; tile geometry and compilation are then varied separately.

Unfused INT8 reduces memory but increases standard-model latency. Operator fusion reduces generation wall time by 43.6% for standard and 31.6% for Flash, computed from within-GPU comparisons. Triton GEMM, time-modulation caching, and GPU RGB conversion retain exact paired video latents and RGB in these tests. Quantization, operator fusion, attention changes, and decoder replacement change numerical outputs; latency gains alone do not establish equivalent visual quality. Decoder compilation changes RGB by at most one 8-bit intensity level.

Two further controls separate architectural and scheduling efects. With the same Flash weights and three evaluations, disabling token compression increases DiT time from 263 to 453 ms; this control is evaluated without retraining. With identical latents, native tile geometry, batched scheduling, precision, and eager execution, replacing the 36-block decoder with the 26-block student reduces decoding from 469 to 340 ms.

<table><tr><td rowspan="2">Component added</td><td colspan="3">Standard (4 NFE)</td><td colspan="3">Flash (3 NFE)</td></tr><tr><td>DiT+dec.</td><td>Generate</td><td>GiB</td><td>DiT+dec.</td><td>Generate</td><td>GiB</td></tr><tr><td>Baseline</td><td>2345</td><td>2682</td><td>77.2</td><td>1592</td><td>1920</td><td>52.4</td></tr><tr><td>+ QKV concatenation</td><td>2355</td><td>2684</td><td>77.2</td><td></td><td></td><td></td></tr><tr><td>+ W8A8 projections</td><td>3292</td><td>3627</td><td>59.9</td><td></td><td></td><td></td></tr><tr><td>+ operator fusion</td><td>1719</td><td>2045</td><td>59.9</td><td>976</td><td>1309</td><td>52.4</td></tr><tr><td>+ autotuned Triton INT8 GEMM</td><td>1432</td><td>1767</td><td>59.9</td><td>869</td><td>1206</td><td>52.4</td></tr><tr><td>+ FlashAttention 3</td><td>1300</td><td>1633</td><td>59.9</td><td>823</td><td>1155</td><td>52.4</td></tr><tr><td>+ time-modulation cache</td><td>1266</td><td>1600</td><td>60.0</td><td>801</td><td>1140</td><td>52.5</td></tr><tr><td>+ GPU RGB conversion</td><td>1268</td><td>1370</td><td>60.0</td><td>801</td><td>903</td><td>52.5</td></tr><tr><td>+ light decoder (native tiles)</td><td>1081</td><td>1175</td><td>54.9</td><td>599</td><td>679</td><td>47.4</td></tr><tr><td>+ adaptive decoder tiles</td><td>935</td><td>1029</td><td>53.9</td><td>460</td><td>557</td><td>46.4</td></tr><tr><td>+ decoder compilation</td><td>843</td><td>959</td><td>53.9</td><td>377</td><td>479</td><td>46.4</td></tr></table>

Table 6 Cumulative component ablation on H100, 22 frames at 540p. Latencies are in milliseconds; memory is peak allocated GiB. Flash already uses INT8 and concatenated QKV in its baseline. FA3 changes attention in both DiT and VAE. DiT+dec. is the median of per-call sums. Component efects are compared within each model, without attributing the diference between separately trained models to a single design choice.

Resolution and duration. Table 7 reports the complete accelerated path: W8A8, FA3, Triton GEMM, timemodulation caching, GPU RGB conversion, and the compiled lightweight decoder with adaptive tiles. The 540p rows reuse the final component configurations; 768p results are separate measured runs. All outputs use 24 fps. The 540p canvas is cropped from 960 × 544 to 960 × 540; 768p uses 1344 × 768. Native temporal padding is included in computation. The 15 s test explicitly permits 362 computed frames and retains the first 360; 64-bit kernel indices support tensors exceeding $2 ^ { 3 1 }$ elements. Each video is generated as one clip.
<table><tr><td>Output</td><td>Model</td><td>NFE</td><td>DiT</td><td>Decoder</td><td>DiT+dec.</td><td>Generate</td><td>Peak</td></tr><tr><td>540p, 22 frames</td><td>Standard</td><td>4</td><td>0.724</td><td>0.119</td><td>0.843</td><td>0.959</td><td>53.9</td></tr><tr><td>540p, 22 frames</td><td>Flash</td><td>3</td><td>0.262</td><td>0.115</td><td>0.377</td><td>0.479</td><td>46.4</td></tr><tr><td>768p, 5 s</td><td>Standard</td><td>4</td><td>18.293</td><td>1.723</td><td>20.018</td><td>20.514</td><td>58.0</td></tr><tr><td>768p, 5 s</td><td>Flash</td><td>3</td><td>5.474</td><td>1.724</td><td>7.198</td><td>7.661</td><td>49.9</td></tr><tr><td>768p, 10 s</td><td>Standard</td><td>4</td><td>56.969</td><td>3.428</td><td>60.393</td><td>61.490</td><td>63.0</td></tr><tr><td>768p, 10 s</td><td>Flash</td><td>3</td><td>16.322</td><td>3.394</td><td>19.716</td><td>20.763</td><td>54.4</td></tr><tr><td>768p, 15 s</td><td>Standard</td><td>4</td><td>116.207</td><td>5.112</td><td>121.315</td><td>122.920</td><td>68.1</td></tr><tr><td>768p, 15 s</td><td>Flash</td><td>3</td><td>33.505</td><td>5.163</td><td>38.665</td><td>40.362</td><td>58.9</td></tr></table>

Table 7 Measured resolution and duration scaling. Latencies are in seconds; peak allocated memory is in GiB. Each row aggregates five measured calls on each of two GPUs. DiT+dec. is the median of per-call sums, which can dife from the sum of component medians.

These measurements characterize warm execution rather than cold-start or request-to-display latency. Original videos and numerical comparisons are retained for inspection; long-clip latencies are measured directly.

Packaged inference validation. A separate single-H100 check uses the public INT8 configuration with the compiled lightweight decoder and adaptive tiles. After two preparation calls, we report medians of five measured calls (Table 8). The installer prepares common shapes and persists kernel caches; first-use compilation is recorded separately from measured inference. Grouped INT8 GEMM and residual–normalization fusion preserve paired latents, audio, and RGB within each tested attention configuration. Changing attention or decoder tile geometry is a separate numerical change. These checks do not establish bitwise equivalence across backends.

The FA3 path takes approximately 0.86 s for Standard and 0.38 s for Flash for denoising plus video decoding. Independent public-command runs give 855 and 381 ms, respectively (940 and 458 ms generation wall time), with five measured calls after two warmups. Public FA2 runs give 986 and 420 ms for DiT plus decoder. Without FA3, the measured Standard path remains above 0.8 s; a successful fallback is not evidence of equal latency. Generation wall time is reported separately and includes additional pipeline work. Installation, conditioning, warmup, and file encoding are excluded.

<table><tr><td>Model</td><td>Attention</td><td>DiT+dec.</td><td>Generate</td></tr><tr><td>Standard</td><td>FA3</td><td>857</td><td>957</td></tr><tr><td>Flash</td><td>FA3</td><td>383</td><td>477</td></tr><tr><td>Standard</td><td>FA2</td><td>990</td><td>1088</td></tr><tr><td>Flash</td><td>FA2</td><td>422</td><td>518</td></tr><tr><td>Standard</td><td>cuDNN</td><td>950</td><td>1057</td></tr><tr><td>Flash</td><td>cuDNN</td><td>416</td><td>516</td></tr></table>

Table 8 Installed inference on one H100, 540p and 22 frames. Times are in milliseconds, with four Standard or three Flash evaluations. The light decoder and tile layout are fixed. Each backend is measured in a separate process; these are deployment checks rather than isolated attention-kernel speedups.

A Input photograph  
![](images/897c7fcdba9d2b56d62e9cb310f6c14392bf1ac7e01d028340991564327768cf.jpg)  
Original 960 × 540 input, shown in full; fog and overcast illumination.

B Preserve overcast illumination  
![](images/c8ff2dc59c10eb1858241ce081969a9fe3a7172cb4450900cfa4089abe6f7365.jpg)  
“Dehaze” while retaining the original illumination: distant haze remains.

C Clear weather + warm sunlight  
![](images/27d77c297a733d50dc0d1dc860f46d29ea0bd2341341595ea904181d1ec330fe.jpg)  
Explicit clear atmosphere and warm illumination; red hull retained.

D Clear weather + yellow hull  
![](images/3494b146331b763a34cfaf3d0da8bb70f2db3c3bc862fe2cb567b448e81cfd8b.jpg)  
Explicit yellow hull and clear weather; visible appearance transfer.

Figure 9 Fixed-model prompt ablation.Clear-weather edits change illumination and synthesi

## 5.5 Reference-guided weather and appearance editing

Figure 9 isolates the efect of specifying an edited visual state. The original lake photograph is a 960×540 picture reference; every generated output is natively 1344×768. We fix the reference-capable base, active EMA adapter, seed, attention backend, four denoiser evaluations, and 22-frame request. Each prompt uses the same ordered picture-reference layout and asks for the completed edit from the opening frame with a locked camera. The exported image is frame zero, and the complete generated clip is retained for temporal inspection.

A conservative instruction to remove haze while preserving overcast illumination leaves substantial distant haze. Describing transparent air, distinct wooded hills, warm sunlight, and a pale blue sky produces a visibly clearer scene while retaining the red kayak. A separate prompt explicitly changes the hull to yellow and also requests clear weather. The black cords and main viewpoint remain recognizable in this example. Sampled frames across the short companion clips preserve the principal composition and edited hull color, although water ripples continue to change. These are controlled prompt examples, not a benchmark-wide estimate of editing reliability.

The broader edits change illumination and synthesize detail hidden by the source fog. They therefore demonstrate generative weather and appearance editing, rather than recovery of a paired clear target. We do not report restoration PSNR or infer improved video physics from a successful color change. A separate native768p nighttime video test retains visible rain near the lamp under the conservative removal instruction.

## 6 Conclusion

LynnReal-Omni couples native multi-modal video difusion with explicit controls over appearance, geometry, motion, and history. Its data pipeline separates physical timing from semantic grouping while verifying every conditioning asset against its source; a shared standard denoiser supports multiple input layouts, and bounded-history long-video generation maintains a fixed memory interface. Agent-produced scenes and executable games further provide inspectable controls whose visual realization can be altered through referenceguided generation. The experiments disentangle several questions that might otherwise be conflated—exact reload fidelity, quality under compression, decoder context, warm throughput, long-rollout continuity, and responsiveness to edits—showing that current warm 540p measurements demonstrate useful acceleration but do not establish equivalent native 768p speed or live-interaction latency. Likewise, a plausible image edit does not prove repaired video physics, and catalog or benchmark design counts are no substitute for measured training exposure or completed evaluations; native 768p development comparisons and multi-modal demonstrations therefore require their own source-linked generation and temporal review. These distinctions make the report’s claims testable as the release and evaluations develop. Future evaluation should complete independent cross-model MSAVP scoring, measure human agreement with automated judges, and test longer interactive trajectories with recorded action-to-display timing, while improving temporal stability under strong style transfer and preserving physical contacts remain central challenges for using a fast video generator as a controllable visual renderer.

## 7 Contributors

The authors are listed in descending order of their actual contributions as follows:

Xiaofeng $\mathrm { M a o ^ { 1 , 2 , 3 , ^ { * } , \ddagger } }$

Peijia Lin<sup>1,2,\*</sup>

Shaohao $\mathrm { R u i ^ { 1 , 2 , 3 , ^ { * } , \ddagger } }$

Yibo $\mathrm { Z h a n g ^ { 1 , 2 , * } }$

Haibin $\mathrm { W a n ^ { 1 , 2 } }$

Weijie $\mathrm { M a ^ { 1 , 2 , 4 , \dagger } }$

<sup>1</sup>LynnReal Lab <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Shanghai Jiao Tong University <sup>4</sup>Fudan University

<sup>\*</sup>Equal contribution. <sup>‡</sup>Project Lead. <sup>†</sup>Corresponding Author.

![](images/b3012d1d3480e10890ab9a8ebcd4772e59d858fea9f9c12d178bb4d9ec15294c.jpg)  
Yellow jacket, helmet and board remain visible through the ramp approach and jump.

Figure 10 Recorded game control and anime-conditioned generation at matched timestamps. The source recording supplies the motion timeline; a locally edited first frame supplies appearance. This development example uses fou denoiser evaluations and does not measure live game input-to-display latency.  
![](images/44c624092d13b6ce2e51b68dd5ba419aec42747c3a5694ac19b437be4c57dd90.jpg)  
The tabletop orbit reveals the cup, vase and sculpture behind the foreground cloth.

Figure 11 An existing mesh camera tour and its clay-conditioned video output at matched timestamps. Foreground occlusion and parallax expose objects behind the cloth. Changes to surface appearance do not imply exact geometric reconstruction; the displayed frames are qualitative development evidence.

## References

Anthropic. Introducing claude 4. Oficial model announcement, 2025. URL https://www.anthropic.com/news/ claude-4.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 102075–102121, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ fce2d8a485746f76aac7b5650db2679d-Paper-Conference.pdf.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023.

ByteDance Seed Team. Seedance 2.0 oficial launch. Oficial model release, 2026. URL https://seed.bytedance.com/ en/blog/seedance-2-0-oficial-launch.

Yuanhao Cai, Kunpeng Li, Menglin Jia, Jialiang Wang, Junzhe Sun, Feng Liang, Weifeng Chen, Felix Juefei-Xu, Chu Wang, Ali Thabet, Xiaoliang Dai, Xuan Ju, Alan Yuille, and Ji Hou. PhyGDPO: Physics-aware groupwise direct preference optimization for physically consistent text-to-video generation. In Proceedings of the European Conference on Computer Vision, 2026. URL https://arxiv.org/abs/2512.24551.

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image difusion. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23206–23217, 2023.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7310–7320, 2024.

Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Eficient video generation with block linear difusion transformer. arXiv preprint arXiv:2509.24695, 2025.

Yiwen Chen, Guosheng Lin, and Chi Zhang. Code world model: Coding agent as world brain. arXiv preprint arXiv:2608.25927, 2026.

Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.

Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Weiwei Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua Yang, Qingqing Dang, and Haoshuang Wang. PP-OCR: A practical ultra lightweight OCR system, 2020. URL https://arxiv.org/ abs/2009.09941.

Xingtong Ge, Yi Zhang, Yushi Huang, Dailan He, Xiahong Wang, Bingqi Ma, Guanglu Song, Yu Liu, and Jun Zhang. Salt: Self-consistent distribution matching with cache-aware training for fast video generation. arXiv preprint arXiv:2604.03118, 2026.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Eficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible difusion modeling of long videos. Advances in neural information processing systems, 35:27953–27965, 2022.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for text-to-video generation, 2024.

Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng Wei, Liangke Gui, Qi Zhao, Gordon Wetzstein, Lu Jiang, and Hongsheng Li. Cameractrl ii: Dynamic scene exploration via camera-controlled video difusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13416–13426, 2025.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video difusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. arXiv preprint arXiv:2506.08009, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models, June 2024.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 17191–17202. IEEE, 2025.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021. URL https://arxiv.org/ abs/2108.05997.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

LAION-AI. LAION Aesthetic Predictor V1. Software, 2022. URL https://github.com/LAION-AI/ aesthetic-predictor.

Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In Proceedings of the European Conference on Computer Vision, pages 280–296, 2022. URL https: //arxiv.org/abs/2203.16527.

Lightricks. LTX-2.5. Oficial model documentation, 2026. URL https://docs.ltx.io/models/ltx-2-5.

Xinran Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. VMBench: A benchmark for perception-aligned video motion generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. URL https://arxiv.org/abs/2503.10076.

Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, Yujun Shen, and Min Zhang. Reward forcing: Eficient streaming video generation with rewarded distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 34385–34397, June 2026a.

Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Eficient streaming video generation with rewarded distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 34385–34397, 2026b.

Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step difusion models by trajectory distribution matching, 2025. URL https://arxiv.org/abs/2503.06674.

MiniMax. Minimax-h3. Oficial model release and inference implementation, 2026. URL https://github.com/ MiniMax-AI/MiniMax-H3.

NVIDIA. Cosmos 3: Omnimodal world models for physical ai, 2026. URL https://arxiv.org/abs/2606.02800.

OpenAI. GPT-4o. Oficial model documentation, 2024. URL https://developers.openai.com/api/docs/models/ gpt-4o. Accessed September 10, 2026.

OpenAI. GPT-6 Astra System Card. System card, September 2026. URL https://deploymentsafety.openai.com/ gpt-6-astra. Accessed: 2026-09-10.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. URL https://arxiv.org/abs/2304.07193.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Qwen Team. Qwen3.5-122B-A10B. Oficial model release, 2026. URL https://huggingface.co/Qwen/Qwen3. 5-122B-A10B.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763, 2021. URL https://proceedings.mlr. press/v139/radford21a.html.

Machel Reid, Nikolay Savinov, Denis Teplyashin, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. URL https://arxiv.org/abs/2403.05530.

Tomás Soucek and Jakub Lokoc. Transnet v2: An efective deep network architecture for fast shot transition detection. In Proceedings of the 32nd ACM International Conference on Multimedia, MM ’24, page 11218–11221, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400706868. doi: 10.1145/3664647.3685517. URL https://doi.org/10.1145/3664647.3685517.

Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2149–2159, 2022. URL https://openaccess.thecvf.com/content/WACV2022/html/Suvorov\_ Resolution-Robust\_Large\_Mask\_Inpainting\_With\_Fourier\_Convolutions\_WACV\_2022\_paper.html.

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Difusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Qinghe Wang, Xiaoyu Shi, Baolu Li, Weikang Bian, Quande Liu, Huchuan Lu, Xintao Wang, Pengfei Wan, Kun Gai, and Xu Jia. Multishotmaster: A controllable multi-shot video generation framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. URL https://arxiv.org/abs/2512. 03041.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models, 2023. URL https://arxiv.org/abs/2303.04671.

Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. Software, 2019. URL https://github.com/facebookresearch/detectron2.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, et al. Qwen3-Omni technical report, 2025. URL https://arxiv.org/abs/2509.17765.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Junyan Ye, Jun He, Zilong Huang, Dongzhi Jiang, Xuan Yang, Rui Chen, and Weijia Li. Genclaw: Code-driven agentic image generation. arXiv preprint arXiv:2605.30248, 2026.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis, 2024a. URL https://arxiv.org/abs/2405.14867.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step difusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024b. URL https://openaccess.thecvf.com/content/CVPR2024/html/Yin\_One-step\_Difusion\_with\_Distribution\_ Matching\_Distillation\_CVPR\_2024\_paper.html.

Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. WBench: A comprehensive multi-turn benchmark for interactive video world model evaluation, 2026. URL https://arxiv.org/abs/2605.25874.

Donghua Yu, Zhengyuan Lin, Chen Yang, Yiyang Zhang, Zhaoye Fei, Hanfu Chen, Jingqi Chen, Ke Chen, Qinyuan Cheng, Liwei Fan, Yi Jiang, Jie Zhu, Muchen Li, Shimin Li, Wenxuan Wang, Yang Wang, Zhe Xu, Yitian Gong, and Yuqian Zhang. MOSS transcribe diarize: Accurate transcription with speaker diarization, 2026. URL https: //arxiv.org/abs/2601.01554.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. URL https://arxiv.org/abs/1801.03924.