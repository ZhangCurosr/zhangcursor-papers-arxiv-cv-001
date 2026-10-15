# BluePixel: Toward Native 4K Agentic Image Editing

Shanyan Guan<sup>1</sup>, Shiyu Li<sup>1</sup>, Xin Li<sup>1</sup>, Zhizhou Chen<sup>2</sup>, Yunzhe Xu<sup>2</sup>, Yuehao Liu<sup>3</sup>, Ruixue Shen<sup>1</sup>, Shiju Zhang<sup>1</sup>, Min Jin<sup>1</sup>, Qiang Liu<sup>1</sup>, Junbo Liu<sup>1</sup>, Penggang Shi<sup>1</sup>, Sheng Sun<sup>1</sup>, Xiao Wang<sup>1</sup>, Yong Zhang<sup>1</sup>, Zhongang Qi<sup>1</sup>, Wei Li<sup>1</sup>, Chao Ma<sup>3</sup>, Ying Tai<sup>2</sup>, Yanhao Ge<sup>1</sup>

<sup>1</sup>vivo Mobile Communication Co., Ltd., <sup>2</sup>Nanjing University, <sup>3</sup>Shanghai Jiao Tong University

## Abstract

Instruction-driven image editing has reached native 4K, yet how such editors are trained remains undisclosed, and pretraining one from scratch is beyond most teams. The task itself compounds the difficulty: a single end-to-end editor cannot make precise parametric adjustments, match specialized experts, or interpret the vague instructions users actually write. We argue that 4K editing is a system problem rather than a single-model problem, and present BluePixel, which couples a native-4K editing model with an agent harness; the complete training recipe is disclosed in this paper. The model makes 4K training practical: it turns a 1K-pretrained editor into a native-4K one through progressive VAE adaptation that keeps 4K training as affordable as 1K, absorbs a long tail of vertical editing skills through continual learning, and refines the aesthetic–fidelity trade-off with reinforcement learning. Distillation via source-anchored target projection (SATP) then compresses 50 denoising steps to 4, completing a 4K edit in 8.7 s on a single Kunlunxin P800 XPU while preserving task success rate and background consistency. The harness, BluePixel Agent, makes editing controllable: it decomposes user instructions into specialist and general operations, verifies every result with the same reward model that guided training, and retries within bounded attempts. On VINS-4KEval++ (3,305 native-4K cases), BluePixel ranks second among seven models, behind only GPT Image 2 in non-edited-region consistency.

Project Page:  https://tech.vivo.com/BluePixel   
Code: § https://github.com/vivo/BluePixel   
Model: Æ https://huggingface.co/openvivo ó https://modelscope.cn/organization/vivo

## 1 Introduction

Diffusion models [15, 41, 45] have brought photorealistic image synthesis and instruction-driven editing [7, 26] to production scale, and output resolution keeps rising: closed-source models went from 1K to 2K and then to 4K, and editing has now reached native 4K as well. A straightforward idea is to cascade a 1K editor with a super-resolution or repainting model [31, 56]; the cascade, however, introduces structural artifacts at the upsampling boundary (ghosting, texture repetition, and loss of fine-grained control) that professional workflows cannot accept, as shown in [11]. VINS-120K [11] first proposed post-training a 1K-pretrained editor directly into a native-4K editing model, and introduced an ultra-high-resolution editing dataset to make this possible. More recently, open-source releases such as FLUX.3 Image [5] and SenseNova-U1.5 [48] have followed the move to 4K, but their training recipes remain undisclosed. This paper starts from two questions. How can an efficient native-4K editing model be trained with limited resources? And is editing fundamentally a single-model problem, or should it be solved by an agent framework?

Replace the large bold heading 'GENESIS' on the right page with the word 'EXODUS'  
![](images/f6c33190e87114caf6ba7b227d9f7688fe6fc613b00ac3473d92bc0e223761d6.jpg)

Make the woman on the left tilt her head to rest gently on the right woman's shoulder  
![](images/8cb87f2278b35b0f570c150dd78da101a2571e93cb209edba4d341eeb7e3d95d.jpg)  
Convert the scene into a dramatic action-movie poster composition, positioning the man in the Iron Man mask as the central hero against a stylized backdrop.

![](images/7cf50848641eae5ec76b7c6de6fb49ec934b0525b5f5aa670a5a3bd42aa063b2.jpg)

Change the woman's expression to an angry look with furrowed brows  
Remove the strong sunlight flare and haze effect in the top right corner to reveal the clear window and wall details behind it  
![](images/d08fa3ebd0dfa7d01b2183103316756da48075662824967d5d88ed7006d10d81.jpg)  
Figure 1 BluePixel editing showcase. Diverse 4K editing capabilities including text editing, pose editing, scene recomposition, anime-style expression editing, and low-level image editing. The model preserves fine details faithfully: even small text remains crisp and legible.

The second question goes deeper than resolution. The dominant paradigm treats editing as a single end-toend model: one network takes an image and an instruction and returns the edited result [7, 38, 50]. This formulation conflates heterogeneous capabilities into one set of weights, and in practice it falls short in three ways. It cannot make precise, parametric adjustments: an exact brightness step or a specific color grade comes back only as an approximation. On task types such as text rendering, teeth retouching, and hairstyle editing, it trails dedicated expert models. And it offers no principled way to act on the short, vague instructions that users actually write (“make this photo look more professional”).

We answer both questions with BluePixel, whose complete training recipe is disclosed in this paper; Figure 1 previews its editing capabilities. First, the training framework is efficient: it builds on a 1K-pretrained editor and never pretrains at 4K. A progressive VAE adaptation recipe completes the jump to native 4K (4K-in, 4K-out, no cascaded super-resolution) in four days on 32 Kunlunxin P800 XPUs, continual learning with ACE-LoRA [36] and MOPD [29] absorbs a long tail of vertical editing skills, and reinforcement learning refines the aesthetic–fidelity trade-off. Second, the resulting model is fast: the high-compression VAE keeps the DiT-stage token count of a native-4K edit at the level of a 1K edit, and distillation via source-anchored target projection (SATP) then compresses 50 denoising steps to 4, enough for interactive multi-turn editing. Third, we treat editing as an agent framework rather than a single model: BluePixel Agent separates intent analysis from editing, composes specialist and general-purpose tools, and uses reward-based feedback to retry or select among candidate images; persisted image provenance supports multi-turn refinement. A data pipeline (a VLM-based quality gate and a four-dimensional reward model) holds all three to the same quality standard.

The remainder of this paper is organized as follows. Section 2 surveys related work. Section 3 presents BluePixel, covering both the native 4K training recipe and the agentic editing harness. Section 4 presents data construction and quality assessment. Section 5 presents experiments, and Section 6 concludes.

## 2 Related Work

Instruction-driven image editing. InstructPix2Pix [7] established instruction-conditioned editing, and subsequent work improved control through test-time optimization [26, 38], larger training data [50], posterior sampling [53], hybrid prompt inversion [18], and semantic intent understanding [12, 68]. Industry models such as HiDream-I1 [9] have since carried instruction editing to production scale, and their resolution is climbing accordingly: Seedream 4.0/5.0 [8, 47], Qwen-Image [51], and ChatGPT-Image [39] already gener ate 4K images; recent open-source releases such as FLUX.3 Image [5] and SenseNova-U1.5 [48] have followed suit, though their training recipes remain undisclosed. The academic line, meanwhile, still operates at 1024 × 1024 or below, and all of these editors keep the same shape: a single end-to-end network that takes an image and an instruction and returns the result. BluePixel keeps this interface and lifts it to native 4K, with its training recipe fully described in Section 3.1.

Training at native 4K. Scaling training to native 4K poses two resource problems. On the data side, UltraHR-100K [70] and VINS-120K [11] provide large-scale high-resolution editing pairs, and learnable posterior sampling [2] advances high-resolution restoration. On the architecture side, a standard f=8 VAE turns a 4096 × 4096 frame into a latent sequence that makes training intractable, and the common alternative, cascading a 1K editor with super-resolution [31, 56], avoids that training cost but introduces structural artifacts (ghosting, texture repetition, and loss of fine-grained control). BluePixel resolves the architecture half: it replaces the standard VAE with DCAE-F64 [10] (f=64 vs. f=8), cutting the 4K token count by 16×, to the level of a 1K edit, so that fine-tuning becomes tractable, and it targets editing rather than generation, which additionally demands instruction fidelity and non-edited-region consistency.

Agentic image editing. Agentic systems delegate task steps to tools under a language-model orchestrator [25], a paradigm established in text and code [40, 46, 49, 55, 61] and recently extended to visual creation and editing [19–21, 30, 32, 42, 43, 54, 57, 63, 66, 72]. BluePixel Agent follows this line but changes what the loop verifies and what the tools are: turn-level routing and persistent image provenance are coupled to a structured editing pipeline that delegates text, tone, character effects, and general edits to different tools, and a reward model judges every candidate, driving local retries and final selection rather than serving only as a post hoc quality report.

## 3 BluePixel

In this paper, we aim to address two challenges: making native-4K editing practical (efficient training and fast inference with limited resources), and making editing more controllable than a single model can be (structured tool use and feedback). BluePixel is a native 4K agentic image editing model built around these two goals. Its training recipe (Section 3.1) turns a 1K-pretrained editor into a native-4K editor with 4-step inference, and its agentic editing harness (Section 3.2) combines routing, composable editing tools, rewardguided retries, and candidate selection.

## 3.1 Eficient Native 4K Training

Overview. Native 4K image editing poses two structural challenges. First, token explosion: a standard f=8 VAE compresses a 4096×4096 frame into a latent sequence so long that training and inference become prohibitively expensive for diffusion transformers (DiTs). Second, data scarcity: high-quality 4K editing pairs are rare and costly to produce, so the data volume needed to train a 4K model from scratch is difficult to assemble. BluePixel addresses both with a three-phase training recipe. The recipe is general: it upgrades any pretrained low-resolution editor into a native 4K editor and does not assume a particular base model. For concreteness, we present it on FLUX.2 [klein] [4], the 1K-pretrained editor we build on:

• Phase 1: Preparation. Replace the f=8 VAE with DCAE-F64 [10] $( f { = } 6 4 )$ and adapt the DiT via representation alignment plus light fine-tuning to obtain an initial 4K model at low cost (∼200K samples, 32 Kunlunxin P800 XPUs for 4 days).

• Phase 2: Enhancement. Progressively improve the model through (i) progressive quality supervised fine-tuning (SFT) at 1024×1024 then 4096×4096, (ii) benchmark-driven fine-tuning with data rebalancing and model merging, (iii) continual learning [36] for vertical editing skills, and (iv) reinforcement learning with DiffusionNFT [73] for aesthetics and instruction fidelity.

• Phase 3: Acceleration. Compress the 50-step teacher into a 4-step student via source-anchored target projection (SATP) step distillation for production latency.

## 3.1.1 Phase 1: Preparation — Lightweight 4K Initialization

The first bottleneck at 4K is the VAE. The native VAE of FLUX.2 [klein] has a down-sampling factor of $f { = } 8$ with a $2 \times 2$ patchify; for a 4K frame this produces a latent sequence far beyond what a diffusion transformer can attend over in bf16 precision. Inspired by DC-Gen [22], which accelerates diffusion models by posttraining in a deeply compressed latent space, we replace the native VAE with DCAE-F64 [10] $( f = 6 4 , 1 \times 1$ patchify). The token count for a H×W frame is

$$
N _ { \mathrm { t o k e n s } } = \frac { H } { f \cdot p } \times \frac { W } { f \cdot p } ,\tag{1}
$$

so, per Equation (1), moving from f=8 to $f { = } 6 4$ reduces the token count of a 4096×4096 frame by $1 6 \times \mathrm { \Omega }$ to exactly the token count of a 1024×1024 edit under the native VAE — 4K fine-tuning thus fits the same memory footprint as 1K training.

However, the public DCAE-F64 latent has 128 channels (vs. 32 in the native VAE), so the DiT’s input projection is incompatible. The natural remedy is to retrain this projection for the 128-channel latent and continue training, but we find that adapting directly from f=8 to f=64 is infeasible: the compression ratio is too aggressive for a single step. Instead, we adopt a progressive adaptation through $f \colon \ 8  3 2  6 4 .$ , where each step bridges only a moderate representation gap. At each compression factor, we perform a three-step adaptation: (i) input projection distillation: distill a new input projection layer via dual-VAE alignment,

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \mathbb { E } \left[ \left\| g _ { \boldsymbol { \theta } } \big ( E _ { \mathrm { n e w } } ( \boldsymbol { \mathbf { x } } ) , t \big ) - g _ { \boldsymbol { \phi } } \big ( E _ { \mathrm { o l d } } ( \boldsymbol { \mathbf { x } } ) , t \big ) \right\| _ { 2 } ^ { 2 } \right] ,\tag{2}
$$

Replace the white studio background with a sunny beach scene with fine white sand  
Replace the central giraffe with a large African elephant standing in the same pose and location  
![](images/2bdff1f71519db8df61481ca9aa4a44a47abd1e38a1e4b36f027054c05c9ff02.jpg)  
Add the word 'GREVE' in a black stencil font on the empty yellow space to the right of the poster, mimicking street art style.

![](images/f49a21c303f621f1473898c3ee7422c276a4c639b1d17cb5d8a35b75f3dc2bd2.jpg)

![](images/2a5432309929f98c41c844a4f91034814b455cd12ef3ab5485342430c5f9c663.jpg)  
Figure 2 Phase 1 vs. final model. The Phase 1 checkpoint produces usable but limited results; after Phase 2 enhancement (progressive SFT, benchmark-driven fine-tuning, continual learning, and reinforcement learning), the final model achieves substantially higher editing quality and detail fidelity.

where $g _ { \theta }$ and $g _ { \phi }$ map latents from the new and original VAEs (with the timestep t) into the DiT’s input space; $g _ { \phi }$ is frozen and only $g _ { \theta }$ is trained, so the DiT receives consistent inputs from either latent space; (ii) interface calibration: train only the input and output projection layers with the transformer backbone frozen; (iii) backbone adaptation: lightly fine-tune the transformer on 1K editing data; steps (ii) and (iii) are both trained with the flow matching loss [33], while the distillation loss in Equation (2) applies only to step (i). The three-step cycle is repeated at f=32 and f=64, with the new projection warm-started from the previous factor. The entire Phase 1 completes in ∼4 days on 32 Kunlunxin P800 XPUs with ∼200K samples; the result is an initial 4K model with usable but limited quality, which Phase 2 then enhances. Figure 2 compares Phase 1 output against the final model, illustrating the quality gap that motivates the enhancement phase.

## 3.1.2 Phase 2: Enhancement — Quality and Capability Refinement

Progressive quality SFT. With the initial 4K model from Phase 1, we now train the model further on high quality editing data. Although Phase 1 aligns the representations as closely as possible, the adaptation process inevitably causes some capability degradation or forgetting in the original model, so the Phase 1 checkpoint provides a good initialization but limited editing quality.

We adopt a progressive resolution schedule: training first continues at 1024 × 1024, where data is abundant and training is efficient; then transitions to native 4096 × 4096 for the main fine-tuning phase, which is necessary because the high compression ratio causes a quality ceiling at 1K (e.g. persistently degraded face rendering) that only native 4K fine-tuning can break through. At each resolution, the input image is resized to a target area while preserving aspect ratio and aligned to a multiple of the VAE scale factor. This curriculum enables the DiT to gradually recover and exceed its pre-adaptation quality. During the early stage of training, we keep the data composition across task types approximately balanced to ensure broad coverage. Once the model approaches convergence, we begin dynamically adjusting the data ratio based on model performance: after each training epoch, the model is evaluated on an internal benchmark, failure modes are identified, and the data composition is rebalanced to target them. Finally, we merge checkpoints from multiple epochs via model merging to obtain a single stable model that is more robust to data distribution shifts than any individual checkpoint. This merged model is the base for downstream specialization.

Continual learning with ACE-LoRA and MOPD. Real-world editing involves a long tail of specialized needs. These needs share two characteristics: task-specific data is scarce and hard to collect, and the needs themselves are numerous. Training on such data jointly with the main SFT stage risks data dilution, which makes the specialized skills hard to learn well; learning each need separately with plain LoRA [23] adapta tion, in turn, invites task interference that drives the optimization toward inferior solutions and catastrophic forgetting that erodes the base model’s general capabilities. We therefore adopt continual learning with ACE-LoRA [36], which applies the gradient-orthogonal principle [35] to image editing, followed by multi teacher on-policy distillation (MOPD) [29] to consolidate the acquired knowledge back into the base model; we refer to this pair as ACE-MOPD, which makes long-sequence learning practical.

Specifically, the vertical skills are defined as many fine-grained templates, each template constituting one editing task; the templates are organized into groups of seven to nine, and each group is learned with ACE-LoRA as a two-stage gradient-orthogonal LoRA procedure: stage 1 fine-tunes the model on the group’s templates, and stage 2 continues training with gradients projected onto the orthogonal complement of the subspace spanned by earlier groups’ LoRAs, suppressing interference with previously learned groups. The learned LoRAs of each group then serve as teachers for one round of MOPD, which transfers the group’s skills back into the base model before training moves on to the next group; the base model therefore accumulates skills group by group while the adapter count stays bounded. We assess final-task performance on held-out suites rather than assuming that the orthogonality constraint alone prevents forgetting. In principle, the adaptation–consolidation cycle can run for an unbounded number of rounds, continuously absorbing new task groups into the base model.

RLfor improving aesthetics and instructionfidelity. We further refine the model with reinforcement learning in two stages, targeting aesthetics and editing fidelity respectively. Both stages build on Diffusion-NFT [73], a likelihood-free policy optimization method that operates directly on the flow matching forward process. In the first stage, we apply DiffusionNFT with PickScore [27], a text-to-image preference reward, on text-to-image (T2I) tasks: aesthetics improve, but editing capability degrades, because the T2I reward favors visual appeal without accounting for instruction adherence or edit locality. We therefore continue training with BluePixReward R(·), our four-dimensional edit quality evaluator (detailed in Section 4.3), on image-to-image (I2I) editing tasks, which improves editing performance. To improve training efficiency and effectiveness, we filter out prompts whose sampled group has low reward variance, as these provide negligible gradient signal for policy improvement.

## 3.1.3 Phase 3: Acceleration — Few-step Distillation

The consolidated 4K model requires 50 denoising steps at inference, which is impractical for production. Recent work has shown that reinforcement learning and distillation can be combined to produce few-step generators with improved quality [16, 24]. We attempted to apply this idea to our 4K editor, but encountered a fundamental obstacle: subtle background changes that are hardly noticeable at 1024 × 1024 become highly salient at 4K, so the reward model is heavily biased by even minute pixel-level drifts in non-edited regions, which makes RL training unstable and the model unable to improve. The mismatch is sharpest for a few-step student: reinforcement learning in Phase 2 operates on the 50-step policy, where later steps can compensate a background drift, whereas a few-step student offers no such tolerance.

Table 1 The 14 vertical editing tasks used for ACE-LoRA adaptation, spanning portrait beauty and three capability categories. Teeth retouching and clothing-wrinkle removal have no held-out test set.
<table><tr><td>Group</td><td>Task</td><td>Sub-templates</td><td>#Templates</td><td>#Samples</td></tr><tr><td rowspan="7">Portrait Beauty</td><td>Change Hairstyle</td><td>side-part, braids, bangs, curly, ... (see Table 9)</td><td>26</td><td>420</td></tr><tr><td>Adjust Expression</td><td>wink, cool, closed mouth, laugh, smile,</td><td>7</td><td>350</td></tr><tr><td>Retouch Teeth</td><td>smirk, open eyes remove braces, remove stains, align, whiten</td><td>4</td><td>305</td></tr><tr><td>Adjust Gesture / Pose</td><td>stretch, heart-hands, cheek-poke, face-</td><td>5</td><td>250</td></tr><tr><td>Apply Lens Defect</td><td>hold, pockets reflection, smudge</td><td>2</td><td>168</td></tr><tr><td>Adjust Hair Volume Apply Photo Effect</td><td>hairline, density boost</td><td>2</td><td>84</td></tr><tr><td></td><td>DV cam, color noise, vignette, disper- sion, grain, fisheye</td><td>6</td><td>300</td></tr><tr><td>Apply Blur Effect</td><td>radial blur, radial burst</td><td>2</td><td>100</td></tr><tr><td>Physical</td><td>Remove Clothing Wrinkle</td><td>wrinkle removal</td><td>1</td><td>191</td></tr><tr><td>Perception</td><td>Refocus Adjust Lighting Direction</td><td></td><td></td><td>300</td></tr><tr><td>Entity</td><td></td><td></td><td></td><td>300</td></tr><tr><td></td><td>Edit Text</td><td></td><td></td><td>259</td></tr><tr><td>Manipulation</td><td>Virtual Try-on</td><td></td><td></td><td>294</td></tr><tr><td>Cognitive Reasoning</td><td>Edit Causal Relation</td><td></td><td></td><td>239</td></tr></table>

This observation indicates that the core challenge in 4K editing distillation is not merely matching the output distribution of the 50-step teacher, but preserving source fidelity in non-edited regions. We therefore build the distillation loss on source-anchored target projection (SATP)<sup>1</sup>, which introduces source-image information directly into the distillation process. The key idea is to use the teacher’s prediction to distinguish edited from non-edited regions, and construct a soft target that anchors the latter to the source latent:

$$
{ \pmb x } _ { \mathrm { t a r g e t } } = { \bf M } \odot \hat { { \bf x } } _ { 0 } + ( 1 - { \bf M } ) \odot { \bf x } _ { \mathrm { s r c } } ,\tag{3}
$$

where $\hat { \mathbf { x } } _ { 0 }$ is the teacher’s clean-image estimate, $\pmb { x } _ { \mathrm { s r c } }$ is the source latent, and M is an edit mask derived from the per-position edit-intensity ratio $r = \| \hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { \mathrm { s r c } } \| ^ { 2 } / \left( \| \mathbf { x } _ { \mathrm { s r c } } \| ^ { 2 } + \epsilon \right)$ , averaged over latent channels: $\mathbf { M } = 1 - \mathbf { \bar { e } } \mathbf { x p } \bar { ( } - \gamma r \mathbf { ) }$ with sensitivity $\gamma = 3 ,$ so positions with $r \approx 0$ are anchored to the source latent while strongly edited positions retain the teacher’s estimate; in practice exp(−γr) is binarized at 0.5 to yield a hard keep/edit partition. The student is then fine-tuned by combining the standard distillation loss with a fidelity term on this anchored target (Equation (3)). By explicitly enforcing source fidelity during distillation, SATP compensates for the sensitivity that made RL infeasible.

The resulting 4-step student preserves task success rate and background consistency relative to the 50-step teacher, while maintaining comparable editing quality; it also achieves a ∼ 4× inference speedup: with bf16 inference and no other optimization, a 4K edit completes in 8.7 s on a single Kunlunxin P800 XPU. The distilled 4K model serves as the base editor for the agentic harness described in Section 3.2, where it is exposed as the primary editing skill.

## 3.2 Agentic Editing Harness

Overview. The 4K diffusion model from Section 3.1 provides the visual substrate, but it remains a single model: one network takes an image and an instruction and returns one edited result. As editing tasks grow more compositional (multi-turn refinement, region-disjoint edits, and reference-guided stylization), a single model reaches a capability ceiling that additional training data alone cannot resolve. BluePixel addresses this by wrapping the 4K model inside an agentic editing harness (BluePixel Agent) that decomposes an editing request into a routed sequence of composable skills with closed-loop verification. The design follows two principles: (i) composability: editing is performed by a registry of specialized, independently replaceable skills rather than a single monolithic model; and (ii) closed-loop verification: every edit is evaluated by a quality module before being returned, and failures trigger automatic re-editing instead of silent acceptance.

## 3.2.1 System Architecture

BluePixel Agent organizes editing around three complementary functions:

1. Request interpretation. A router considers the user’s instruction together with the current image and conversation context. It determines whether there is enough information to proceed with an edit or whether the turn instead calls for clarification, suggestions, or assessment of an existing result. For an actionable edit, an intent analyst formulates the operations to be performed and may gather supporting information for underspecified references. For example, when a user names a visual style without describing its defining features, the analyst may use web search to identify relevant visual attributes before specifying the edit.

2. Skill execution. The harness applies the relevant editing skills to these operations (Table 2). Specialized tools handle constrained tasks such as color grading and template-guided editing, while the general-purpose editor handles open-ended changes to image content.

3. Feedback and selection. The resulting images are assessed against the instruction and source image. This feedback drives further editing when needed and the final candidate selection (Section 3.2.3).

Some edit requests depend on facts beyond the source image, such as attributes of a named character or a time-sensitive reference. In these cases, the intent analyst can use web search to gather relevant information and incorporate it into the editing instruction. Figure 3 illustrates this capability with character-specific visual effects and dialogue, and with a zodiac-themed toy requested by reference to the current year.

These functions separate decisions about what should be edited, how the edit is carried out, and how its outcome is assessed. The harness also retains conversational constraints and image provenance across turns, allowing users to refine a selected result or branch from an earlier version. This continuity connects hetero geneous editing skills within a service rather than treating each request as an isolated model call.

The advantage of the skill-based approach is illustrated in Figure 4. When a user requests a specific filter look (a task with strong professional aesthetic orientation), a pure diffusion model tends to generate a plausible but imprecise approximation. BluePixel Agent, by contrast, can use LUT-based color grading to apply a calibrated color transform that matches the intended look. This is a direct consequence of the composable skill design: tasks that have well-defined technical solutions are handled by dedicated tools, while the primary editing model is reserved for generative edits that truly require it. We next describe the skill-based editing repertoire and the feedback process used to assess its outputs.

## 3.2.2 Skill-Based Editing

Production-grade image editing spans a wider range of operations than a single image-to-image transformation. A request may call for global stylization, localized changes, a calibrated color treatment, or a templateguided transformation; these operations place different demands on how an image is modified. BluePixel Agent therefore treats the 4K diffusion editor as one capability within a broader skill repertoire rather than the sole mechanism for every edit.

The repertoire has three functional tiers, as shown in Table 2. The primary editing tier contains the BluePixel 4K diffusion editor for open-ended changes to image content. Specialist editing skills encapsulate operations tailored to particular editing domains, with task-specific controls that can be used independently when a request falls within their scope. When an instruction combines such an operation with open-ended content changes, the harness can coordinate the specialist skill with the primary editor. Auxiliary capabilities interpret intent, resolve references, refine instructions, and assess results; they guide the editing process without directly modifying the image.

添加特效和台词 (Add visual effects and dialogue.)  
![](images/57e2987676cebe1a1c9ffdbb9e9fc5d68eba2cf6b513ce2b2e381eff92135744.jpg)  
Figure 3 Agent-assisted editing with supplementary factual context. Top: adding visual effects and dialogue associated with the depicted character. Bottom: adding a plush toy corresponding to the Chinese zodiac year specified in the instruction. Each row shows, left to right: the input image, BluePixel Agent, GPT Image 2, Nano Banana Pro, and Seedream 5.0 Pro.

## 3.2.3 Closed-Loop Editing

Editing in BluePixel Agent is not an open-loop chain that emits a single result and trusts it; it is a closed-loop process: edit, verify, and re-edit if necessary. An edited image is treated as a candidate whose assessment informs both subsequent actions and the final result.

Once the requested edit and its constraints have been specified, the image editor executes the relevant skill and produces a candidate. The quality evaluator receives the source image, candidate, and editing instruction, and uses BluePixReward, our VLM-based judge (Section 4.3), to assess instruction following, non-edited region consistency, and visual quality. It returns scores and diagnostic feedback that the harness can use to decide what to do next.

The evaluation is actionable rather than merely descriptive. If a specialist edit leaves part of the request unmet, the harness can retry the same skill and, if needed, carry the unmet requirement into a general edit. For a generative edit, the diagnostic feedback can inform a revised instruction for another attempt. Recovery attempts are bounded, and the harness retains intermediate candidates rather than assuming that the latest edit is the best. It selects the final image by assessing the candidates against the complete request, so feedback governs both whether editing continues and which result is returned.

Give it a clear, translucent look with crisp greens and a bright, natural skin tone.  
![](images/16d5416ced700dc91e1c0b5c79012f7586aba5fc93a1627406621fcb65c4327d.jpg)

Figure 4 Filter-style editing comparison. BluePixel Agent vs. GPT Image 2, Nano Banana Pro, and Seedream 5.0 Pro on a filter task requiring professional aesthetic judgment. The three compared baselines produce plausible but aesthetically inconsistent results. BluePixel Agent routes the request to LUT-based (look-up-table) color grading, a specialist editing capability in Table 2, rather than relying solely on the diffusion model; this produces more accurate and professionally consistent filter effects.  
![](images/b65fad4b98694631d30be8304ca5ab0b55e59d8aa72bdfa4ebe5063681de96d4.jpg)  
Figure 5 The 4K set concentrates on three task types, while the 1K set is more evenly distributed: task type proportions for the 1K and 4K training datasets across the eight unified task types.

## 4 Data Construction and Quality Assessment

## 4.1 Overview

Our training data combines publicly available editing datasets, including UnicEdit-10M [62], TextEdit [52], X2Edit [37], and SpatialEdit [59], with in-house annotated pairs. Because Phase 1 yields a good initialization, we adopt a quality-over-quantity strategy: rather than scaling data volume, we focus on filtering for visual quality, aesthetics, information density, and edit accuracy. The pipeline applies two quality gates: an image quality filter that discards pairs failing visual quality checks, and an edit evaluation reward model (BluePixReward, detailed in Section 4.3) that scores each surviving triple (source, edited, instruction) on instruction following, non-edited region consistency, visual quality, and overall. The same quality standard governs both training-time filtering and evaluation. We construct 1K and 4K datasets, reorganized into eight task types: Spatial Editing, Text Editing, Portrait Editing, Object Modification, Background & Style Editing, Object Addition, Object Removal, and Multi-step Editing (Figure 5).

Table 2 Skill-based editing capabilities. Representative editing and supporting capabilities of BluePixel Agent, organized by functional tier.
<table><tr><td>Tier</td><td>Capability</td><td>Role</td></tr><tr><td>Primary editing</td><td>BluePixel 4K diffusion editor</td><td>Native-4K, open-ended edits to objects, backgrounds, and scene composition</td></tr><tr><td rowspan="5">Specialist editing</td><td>LUT-based color grading</td><td>Global color transformations for filter-style edits</td></tr><tr><td>Tone and lighting effects</td><td>Preset-based adjustments to illumination and visual atmosphere</td></tr><tr><td>Template-guided editing</td><td>Application of predefined styles or scene treatments</td></tr><tr><td>Mask-guided local editing</td><td>Changes restricted to a selected image region</td></tr><tr><td>Hairstyle editing Reference-guided prop editing</td><td>Hair-region edits guided by a user-provided mask</td></tr><tr><td rowspan="5">Auxiliary</td><td>Intent and reference grounding</td><td>Placement of a reference object within a designated region Interprets the request and resolves underspecified</td></tr><tr><td>In-house prompt refinement model</td><td>references Reformulates instructions before editing or using</td></tr><tr><td>BluePixReward evaluator</td><td>evaluation feedback Assesses instruction following, content preservation, and</td></tr><tr><td></td><td>visual quality</td></tr><tr><td>Version tracking</td><td>Retains image history for multi-turn refinement</td></tr></table>

## 4.2 Image Quality Filtering

VINS-120K [11] establishes a multi-stage filtering paradigm for high-resolution editing data: preliminary file checks, image quality filtering via hand-crafted perceptual operators (sharpness, exposure, color, and texture statistics), instruction-following filtering via VLM parsing combined with object detection and segmentation, and aesthetic assessment via dedicated predictors; each stage relies on a manually tuned threshold, and in VINS-120K only the top 20% of samples survive the full pipeline. We adopt the same joint filtering principle (if either the source or the edited image fails any quality check, the pair is discarded), but observe that per-operator threshold tuning is inherently subjective, does not generalize across data distributions, and requires maintaining a large stack of independent operators with their own failure modes.

To eliminate this subjectivity, we train a single VLM based on Qwen3-VL-8B [44] that takes the image as input and directly outputs scores for multiple quality metrics along with a reasoned analysis, producing its own retain-or-discard judgment. The model evaluates five axes that capture complementary aspects of image quality: aesthetics (composition, lighting, and color harmony), visual clarity (image sharpness and arti facts), saturation (color vividness and richness), information density (texture, structural details, and semantic complexity), and watermark presence. We then filter images based solely on the model’s judgment, replacing the brittle stack of hand-tuned thresholds with a single, learned quality gate. We pay particular attention to information density: high-density data, comprising images rich in texture, structure, and semantic content, can accelerate convergence, as each sample exposes the model to more diverse visual patterns.

## 4.3 Edit Evaluation Reward Model

Beyond general image quality, we need to verify that each edit pair is correct: the edited image must faithfully follow the instruction without introducing artifacts or damaging non-edited regions. To this end, we train a dedicated edit evaluation reward model, BluePixReward. Expert models are first trained on different editing tasks, and multi-teacher on-policy distillation (MOPD) [29] then distills them into a single judge built on Qwen3.5-9B; it takes the source image, edited image, and instruction as a triple and outputs four-dimensional scores<sup>2</sup>:

• Instruction Following (IF): whether the edit accurately executes the instruction, penalizing wrong edit type, missing or incomplete edits, extraneous modifications, and person-related errors.

• Non-Edited Region Consistency (NC): requires a reasoning step that first localizes the edit region,

then compares the complement; camera-motion edits are treated as a special case.

• Visual Quality (VQ): naturalness and fusion quality of the edited region and overall image, penalizing fusion seams, material/lighting mismatches, inpainting artifacts, person defects, and severe distortion across the whole image.

• Overall: a composite quality score reported alongside the three component scores. The evaluator also provides a binary success decision: an output counts as successful when $\mathrm { I F } \geq 9 , \mathrm { N C } \geq 8 ,$ , and ${ \mathrm { V Q } } \geq 6 ,$ based on its assessment of instruction following, non-edited-region consistency, and visual quality. This decision is reported as the Success column in Table 8.

The scores are calibrated against human judgment: Pearson correlations with expert Likert ratings on a 150- pair validation set are $r = \bar { 0 . 8 2 } \left( \mathrm { I F } \right) , r = 0 . 7 8 \left( \mathrm { N C } \right) , r = 0 . 7 6 \left( \mathrm { V Q } \right)$ , and r = 0.84 (Overall). This reward model serves as the unified quality signal throughout the pipeline: low-scoring pairs are discarded from training data; multiple candidates for the same instruction are scored jointly as a group, providing the reward signal for DiffusionNFT; the same model also serves as the benchmark evaluator for VINS-4KEval++ (Table 8, the BluePixReward protocol), whose four-dimensional scores serve as the primary evaluation metrics, closing the quality loop.

## 4.4 4K Data Construction

Similar to VINS-120K [11], we source 4K pairs from two complementary streams, with the majority coming from video mining. The first uses a dedicated pipeline that mines video footage for frame pairs that can serve as editing data, selecting frames where the visual difference between two shots constitutes a plausible edit, and generating corresponding instructions via prompt rewriting. The second applies superresolution to a subset of 1K source images. Before upscaling, each candidate is screened by MANIQA [60], a no-reference image-quality model used here for sharpness screening: the image is cropped into 224 × 224 patches, each patch is scored with a Vision Transformer (ViT), and the patch scores are aggregated; upscaling a blurry 1K image produces artifact-laden 4K results, so only sufficiently sharp images proceed. All 4K pair pass through the same image quality filtering and edit evaluation reward model as their 1K counterparts.

## 5 Experiments

Benchmarks. We evaluate BluePixel on four benchmarks. ImgEdit-Bench [64] provides broad task coverage across nine editing categories at standard resolutions; for fair comparison, we resize our results to 1024 × 1024 on this benchmark. Next, VINS-4KEval [11] is the first benchmark dedicated to 4K image editing, and we use it to evaluate the performance at native 4K resolution. Third, we extend VINS-4KEval with more diverse scenes and instructions, termed VINS-4KEval++, comprising 3,305 evaluation cases. Finally, we use CIE-Bench [36] to evaluate the continual learning performance (as introduced in Section 3.1.2).

## 5.1 Main Results

## 5.1.1 Qualitative Results

Comparison on VINS-4KEval++. Figure 6 compares BluePixel with six representative baselines (GPT Image 2 [39], Nano Banana Pro [17], Nano Banana 2.0 [17], Seedream 5.0 Lite [8], Wan 2.7 Pro [1], and Kling [28]) on three examples drawn from VINS-4KEval++ at native 4K. The instructions require compound attribute edits, facial-expression manipulation, and typographic restyling without changing the text content. In the attribute-editing example, all six baselines render the dark brown tunic, but GPT Image 2 alters the fairground background and restyles the whole saddle instead of only making the pad red leather, whereas BluePixel makes exactly the requested changes and preserves the rest of the scene. In the expression example, the strongest baselines reach a similar bored look, but GPT Image 2 also alters the man beside her (eyes closed, smile gone) although the instruction targets only the woman; BluePixel edits her alone. The textediting example shows the widest gap: Wan 2.7 Pro retypes the sign as “UNED SHOP” and Kling misspells it as “USED VINL SHOP”, whereas BluePixel renders the requested bold white lettering with the wording and the storefront layout untouched, sharp at full resolution.

Zoom in on the man and horse, change the white shirt to a dark brown tunic, and swap the saddle pad for red leather.  
![](images/23b82d1dc8e26b0d57b7e35ecb6bdee8c298edbd5dee4fdfe49916174fe313d3.jpg)  
Figure 6 Main qualitative results on VINS-4KEval++. Each two-row block shows one instruction: compound attribute editing, facial-expression editing, or text restyling. Columns, left to right: the input image, BluePixel, GPT Image 2, Nano Banana Pro, Nano Banana 2.0, Seedream 5.0 Lite, Wan 2.7 Pro, and Kling.

Effect of vertical-template learning. On the specialized-skill side, Figure 7 compares the BluePixel model with Seedream 5.0 [8] on three vertical tasks: a causal-relation edit, a refocus edit, and a virtual try-on. BluePixel realizes the requested causal outcomes (a bite taken out of the potato, and the requested words rendered legibly on the refocused spine) and transfers the reference garment while preserving the original framing and posture, whereas Seedream 5.0 renders the badger in contact with an intact potato, leaves the spine text at its original size, and reframes the try-on result as a full-body shot. These tasks are precisely the vertical skills added by continual learning, which we analyze next.

![](images/209e76145ff932681b88db2116c540e8cd4e3e15dd38d70fe8630d80320b79c1.jpg)  
Figure 7 Qualitative comparison on CIE-Bench [36]. BluePixel vs. Seedream 5.0 on a causal-relation edit and a refocus edit (left), and a virtual try-on (right). BluePixel renders each requested outcome — the badger’s bite, the legible refocused spine text, and the garment transfer at the original framing — whereas Seedream 5.0 shows an intact potato, unchanged spine text, and a full-body shot.

## 5.1.2 Model Analysis

Beyond the end-to-end comparison, we analyze the two components that distinguish BluePixel from a plain editing model: the agent harness at inference and continual learning with ACE-LoRA and MOPD in training.

Is the agent harness effective? Figure 8 isolates the contribution of the agent harness by contrasting the backbone editor with BluePixel Agent on the same instructions. In the facial-expression example, the backbone changes the expression in the right direction but overshoots, producing a stiff, grinning look rather than the requested relaxed and serene one; the agent’s verification loop catches the mismatch, retries, and returns a calm expression that matches the instruction. In the text example, the backbone ignores half of the request: the word is rendered as garbled letterforms (“COEFAE”) in the wrong color and font, whereas the agent produces “Decaf” in bold red sans-serif without disturbing the surrounding sign. In the mask-repair example, the backbone leaves the crack clearly visible across the mask’s forehead, whereas the agent repairs the fracture and restores a continuous metallic texture. Across all three cases the backbone delivers an ap proximate one-shot result, and the agent converges to an output that actually satisfies the instruction, which is precisely what the verification-and-retry loop is for.

Can ACE-MOPD absorb data-scarce yet practically valuable templates? The fine-grained templates of Table 1 are exactly that: data-scarce yet practically valuable. Each is learned from only about 15 to 200 samples, yet covers recurring creative-workflow needs (hairstyle change, expression control, virtual try-on, causal-relation editing), where the instruction specifies the target attribute or effect and everything outside it (identity, apparel, scene content, viewpoint) is a preservation constraint. Figures 10 to 13 present 22 representative outputs of the ACE-MOPD continual-learning model, spanning local semantic control (hairstyle and expression) and broader geometric or photometric transformations (pose and global effects). They show that BluePixel learns a large repertoire of such templates and retains strong few-shot learning capability over a long sequence of tasks.

On the five CIE-Bench capability tasks, scored under the benchmark’s native VLM-judge protocol (instruction adherence + naturalness, 10-point total), continual learning improves every capability task (Table 3): the lighting task gains most (+4.69), followed by virtual try-on (+4.45), while the portrait-beauty suite shows

Modify the man's facial expression to appear more relaxed and serene. Soften his gaze by relaxing the muscles around his eyes so he looks less intense and more at ease. Flatten the corners of his mouth to remove the slight smile, giving him a neutral and calm expression. Ensure that the woman standing next to him and the solid blue wall in the background remain completely unchanged.

![](images/69069e7c78f29a74242caf5efb7347d8f737b108d43453dadde5c477407dbe5c.jpg)

![](images/ccc471f58117ed44e897a0854c535cb53a1096875da4f9e1d55802b78265bab8.jpg)

![](images/d0c5fc2cea6282d027c1dea5324bc2929f6f53a0021bc6df985361971d85340b.jpg)  
Change the visual style of the word 'Decaf' on the sign to a bold red sans-serif font.

![](images/129b535c72275d7b08438a7e4c03b80f26b16c9b744485223661cd9a03b282d7.jpg)

![](images/ad638d4505104dd4a1b187bedbf9d1fb73fe54253deb9000975f9b53910f0c0b.jpg)

![](images/0683a4de5feb7d0483b59417888a0fb6ce247611ffc97e492d6d055cb11cad2b.jpg)

Locate the silver skull mask worn by the figure and identify the vertical crack running down the center of its forehead. Repair this crack so that the forehead area appears smooth, intact, and free of any visible seams or damage, maintaining the consistent metallic texture and appearance of the surrounding mask surface.  
![](images/971e8f885825e3c7a8ed657c369138d7e76c1bf88ef6a05ab027d0b9e460626e.jpg)  
Input Image

![](images/440ee7f9bd73d7d355f48da90610e92ec833d77c26b11d1ea5b6e378af26eb45.jpg)  
BluePixel

![](images/ddd8d7c9e77ee478b2a58f2a1da45d311cff2e9063b5efc914d5358881a16fef.jpg)  
BluePixel Agent  
Figure 8 Efect of the agent harness. Three editing instructions (facial expression, text restyling, and mask repair) executed by the BluePixel backbone alone (middle) and by BluePixel Agent (right). The agent produces the requested expression, renders the target word correctly, and fully repairs the cracked mask, while the backbone under-modifies or corrupts these details.

no regression. Across the training stages, the CIE total rises from 6.24 in the base model to 8.87 after continual learning with ACE-LoRA and MOPD, dips to 8.52 after aesthetic RL, and partially recovers to 8.57 after reward RL with BluePixReward (Figure 9); general-editing quality on VINS-4KEval++ follows the opposite trajectory across the same stages.

Seed stability and per-task-type breakdown. Table 4 lists the eight seeds behind the mean in Table 8: the per-seed Overall spans 0.053 (noise level); no seed falls below Nano Banana 2.0 (8.561), and every seed except one matches or exceeds Nano Banana Pro (8.601), so the 0.02-point margin over Nano Banana Pro is within seed noise. Table 5 breaks the same runs down by the benchmark’s nine task types: adding objects is the strongest category (Overall 9.20), changing camera viewpoint the weakest (7.55), largely because instruction following conflicts with non-edited-region preservation.

<table><tr><td>Task</td><td>SFT↑ + ACE-MOPD↑</td><td></td><td>Δ</td></tr><tr><td>Refocus</td><td>7.53</td><td>8.71</td><td>+1.19</td></tr><tr><td>Light</td><td>3.19</td><td>7.87</td><td>+4.69</td></tr><tr><td>Text editing</td><td>8.37</td><td>9.28</td><td>+0.91</td></tr><tr><td>Causal-relation editing</td><td>7.50</td><td>9.42</td><td>+1.92</td></tr><tr><td>Virtual try-on</td><td>4.60</td><td>9.05</td><td>+4.45</td></tr><tr><td>Average</td><td>6.24</td><td>8.87</td><td>+2.63</td></tr></table>

Table 3 CIE-Bench capability tasks before and after continual learning (benchmark-native VLM-judge protocol; scores are the equal-weight average over the five tasks).

![](images/098967901a572df8582a8fb743bc10fe396c8ac25f2d483a2f4a73f4b4579897.jpg)  
Figure 9 Stage-wise CIE total across the training recipe (SFT, + ACE-MOPD, + aesthetic RL, + reward RL; single run per stage, benchmark VLM judge).

Table 4 Per-seed results on the 2,513 shared VINS-4KEval++ cases (BluePixReward judge, prompt rewriting), ranked by Overall. ∆ is the Overall margin over Nano Banana Pro.
<table><tr><td>Seed</td><td>Overall↑</td><td>Success (%)↑</td><td>∆ vs. Nano Banana Pro</td></tr><tr><td>2026</td><td>8.653</td><td>63.8</td><td>+0.052</td></tr><tr><td>42</td><td>8.651</td><td>64.0</td><td>+0.050</td></tr><tr><td>1337</td><td>8.621</td><td>63.2</td><td>+0.020</td></tr><tr><td>1</td><td>8.619</td><td>63.4</td><td>+0.018</td></tr><tr><td>3407</td><td>8.619</td><td>62.4</td><td>+0.018</td></tr><tr><td>777</td><td>8.615</td><td>63.0</td><td>+0.014</td></tr><tr><td>1234</td><td>8.612</td><td>62.9</td><td>+0.011</td></tr><tr><td>0</td><td>8.600</td><td>62.9</td><td>-0.001</td></tr><tr><td>8-seed mean</td><td>8.624</td><td>63.2</td><td>+0.023</td></tr></table>

![](images/a40061f3c129cc64a79adc2d1a06c4693afbd659a0aff00ae9a1fe5f9516706a.jpg)  
Figure 10 ACE-MOPD qualitative visualization: hairstyle manipulation. The leftmost column shows the source image; subsequent columns correspond to six target hairstyles: wind-swept long hair under warm backlight, wind-swept long hair, braids, a short pixie cut, loose waves, and a high ponytail. The edits change hair structure and appearance while retaining subject identity, apparel, pose, and scene layout.

## 5.1.3 Quantitative Results

ImgEdit-Bench results. Table 6 reports the per-category and overall scores of BluePixel on ImgEdit-Bench, compared against a broad set of open-source and proprietary editing methods [3, 4, 6–8, 13, 17, 39, 47, 50, 51, 67, 71], ranging from early instruction-editing approaches to recent production-level models. BluePixel reaches an Overall score of 4.32, competitive with recent production-level models: it ranks first on Hybrid (4.59) and Action (4.93), tied for first on Add (4.93), and second on Adjust (4.76); the remaining gap to the strongest methods concentrates on Extract and Style.

VINS-4KEval results. We further evaluate the final model on the original VINS-4KEval benchmark at 4K resolution under the benchmark’s default evaluation protocol (Table 7). This comparison includes FLUX.1 Kontext [Dev] + VINS, the Kontext-dev editor [6] post-adapted on VINS-4KEval [11] and released alongside the benchmark, together with the open-source editors FLUX.1 Kontext [Dev], OmniGen2 [58], Step1X-

Table 5 Per-task-type results on the full VINS-4KEval++ benchmark (8-seed mean, BluePixReward judge).
<table><tr><td>Task type</td><td>Overall↑</td><td>Success (%)↑</td></tr><tr><td>Add object or element</td><td>9.203</td><td>79.3</td></tr><tr><td>Remove object or element</td><td>8.994</td><td>72.9</td></tr><tr><td>Edit people/faces/character</td><td>8.931</td><td>74.0</td></tr><tr><td>Multi-step (compound)</td><td>8.769</td><td>62.6</td></tr><tr><td>Modify existing object</td><td>8.663</td><td>62.0</td></tr><tr><td>Alter background/texture/style</td><td>8.579</td><td>56.9</td></tr><tr><td>Cosplay special edit</td><td>8.541</td><td>58.3</td></tr><tr><td>Edit text or document content</td><td>7.867</td><td>46.1</td></tr><tr><td>Change camera view/perspective</td><td>7.549</td><td>47.5</td></tr></table>

![](images/c5f55c0914c3d9c3020a07cb005f6c5f15dbcd424a8d82e96736f01d386f5757.jpg)

Figure 11 ACE-MOPD qualitative visualization: global image-efect rendering. The leftmost column shows the source image; subsequent columns apply chromatic aberration, directional motion blur, radial zoom blur, vignetting, fisheye distortion, a DV-camera effect, and chromatic noise, respectively. The source image’s semantic content and scene structure are retained under these global appearance transformations.  
![](images/84824b7d0ec6580761946c4e7772499c347518acf72004d148de5acecfc5d209.jpg)  
Figure 12 ACE-MOPD qualitative visualization: facial-expression manipulation. The leftmost column shows the source image; subsequent columns depict a surprised expression, a subtle smile, a wink, and a broad smile. The transformation is localized to the face while preserving identity cues, hairstyle, apparel, viewpoint, and background.

Edit [34], BAGEL [14], ICEdit [69], and AnyEdit [65]; our model matches the Kontext+VINS naturalness (4.70) and exceeds it on instruction adherence and detail.

VINS-4KEval++ results. BluePixel achieves the second-highest Overall score (8.62) and success rate (63.2%) among the seven models on VINS-4KEval++ (Table 8). All models are evaluated on the 2,513 shared cases under the BluePixReward protocol. Per-metric, BluePixel ranks second in non-edited-region consistency (NC, 9.06), behind only GPT Image 2 (9.34), while instruction-following (IF, 8.74) and visual quality (VQ, 8.90) fall below four external models. Two caveats: (i) the 0.02-point Overall margin over Nano Banana Pro is within noise given the absence of uncertainty estimates; (ii) BluePixel applies prompt rewriting, whereas each external model produces a single output per case.

![](images/b1330d3c31cfa2515bc84978a289206e5625f24865b054d78120cd30d16e612d.jpg)  
Figure 13 ACE-MOPD qualitative visualization: human pose manipulation. The leftmost column shows the source image; subsequent columns show a relaxed standing pose, a cheek-heart gesture, a head tilt, a peace sign, and a raisedarms pose. The model changes the articulated body configuration while maintaining subject identity, clothing, full body framing, and the outdoor scene.

Table 6 BluePixel ranks first on Hybrid and Action, ties for first on Add, and remains competitive overall. Main results on ImgEdit-Bench: each cell is the average score (1–5); Overall is the average across the nine task categories. Superscripts distinguish the reporting source for models with multiple reported scores: <sup>a</sup>FireRed-Image-Edit, <sup>b</sup>LongCat Image, <sup>c</sup>HiDream-O1-Image, <sup>d</sup>Boogu-Image-0.1.
<table><tr><td>Method</td><td>Add</td><td>Adjust</td><td>Extract</td><td>Replace</td><td>Remove</td><td>Background</td><td>Style</td><td>Hybrid (Compose)</td><td>Action</td><td>Overall</td></tr><tr><td>GPT Image 2</td><td>4.85</td><td>4.82</td><td>4.07</td><td>4.92</td><td>4.75</td><td>4.88</td><td>4.96</td><td>4.47</td><td>4.85</td><td>4.73</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>4.54</td><td>4.57</td><td>4.13</td><td>4.70</td><td>4.46</td><td>4.36</td><td>4.89</td><td>4.16</td><td>4.81</td><td>4.51</td></tr><tr><td>Nano Banana 2.0</td><td>4.56</td><td>4.41</td><td>4.04</td><td>4.62</td><td>4.74</td><td>4.54</td><td>4.87</td><td>4.01</td><td>4.81</td><td>4.51</td></tr><tr><td>Seedream 5.0 Lite</td><td>4.93</td><td>4.69</td><td>3.01</td><td>4.41</td><td>4.45</td><td>4.65</td><td>4.93</td><td>3.91</td><td>4.82</td><td>4.42</td></tr><tr><td>Emu3.5</td><td>4.61</td><td>4.32</td><td>3.96</td><td>4.84</td><td>4.58</td><td>4.35</td><td>4.79</td><td>3.69</td><td>4.57</td><td>4.41</td></tr><tr><td>Nano Banana Pro</td><td>4.44</td><td>4.62</td><td>3.42</td><td>4.60</td><td>4.63</td><td>4.32</td><td>4.97</td><td>3.64</td><td>4.69</td><td>4.37</td></tr><tr><td>FLUX.2-Dev</td><td>4.50</td><td>4.18</td><td>3.83</td><td>4.65</td><td>4.65</td><td>4.31</td><td>4.88</td><td>3.46</td><td>4.70</td><td>4.35</td></tr><tr><td>Qwen-Image-Edit-2509b</td><td>4.32</td><td>4.36</td><td>4.04</td><td>4.64</td><td>4.52</td><td>4.37</td><td>4.84</td><td>3.39</td><td>4.71</td><td>4.35</td></tr><tr><td>Seedream 4.5</td><td>4.57</td><td>4.65</td><td>2.97</td><td>4.66</td><td>4.46</td><td>4.37</td><td>4.92</td><td>3.71</td><td>4.56</td><td>4.32</td></tr><tr><td>Seedream 4.0ª,d</td><td>4.33</td><td>4.38</td><td>3.89</td><td>4.65</td><td>4.57</td><td>4.35</td><td>4.22</td><td>3.71</td><td>4.61</td><td>4.30</td></tr><tr><td>Nano Bananaª,d</td><td>4.62</td><td>4.41</td><td>3.68</td><td>4.34</td><td>4.39</td><td>4.40</td><td>4.18</td><td>3.72</td><td>4.83</td><td>4.29</td></tr><tr><td>Qwen-Image-Edit</td><td>4.38</td><td>4.16</td><td>3.43</td><td>4.66</td><td>4.14</td><td>4.38</td><td>4.81</td><td>3.82</td><td>4.69</td><td>4.27</td></tr><tr><td>GPT Image 1 [High]</td><td>4.61</td><td>4.33</td><td>2.90</td><td>4.35</td><td>3.66</td><td>4.57</td><td>4.93</td><td>3.96</td><td>4.89</td><td>4.20</td></tr><tr><td>FLUX.1 Kontext [Pro]b,c</td><td>4.25</td><td>4.15</td><td>2.35</td><td>4.56</td><td>3.57</td><td>4.26</td><td>4.57</td><td>3.68</td><td>4.63</td><td>4.00</td></tr><tr><td>FLUX.1 Kontext [Dev]ª</td><td>3.99</td><td>3.88</td><td>2.19</td><td>4.27</td><td>3.13</td><td>3.98</td><td>4.51</td><td>3.23</td><td>4.18</td><td>3.71</td></tr><tr><td>UltraEdit</td><td>3.44</td><td>2.81</td><td>2.13</td><td>2.96</td><td>1.45</td><td>2.83</td><td>3.76</td><td>1.91</td><td>2.98</td><td>2.70</td></tr><tr><td>MagicBrush</td><td>2.84</td><td>1.58</td><td>1.51</td><td>1.97</td><td>1.58</td><td>1.75</td><td>2.38</td><td>1.62</td><td>1.22</td><td>1.90</td></tr><tr><td>Instruct-Pix2Pix</td><td>2.45</td><td>1.83</td><td>1.44</td><td>2.01</td><td>1.50</td><td>1.44</td><td>3.55</td><td>1.20</td><td>1.46</td><td>1.88</td></tr><tr><td>BluePixel</td><td>4.93</td><td>4.76</td><td>3.45</td><td>4.53</td><td>4.21</td><td>4.43</td><td>3.85</td><td>4.59</td><td>4.93</td><td>4.32</td></tr></table>

## 6 Conclusion

We have presented BluePixel, a native 4K image editing model with an agent framework, and a training recipe that makes native-4K editing practical on limited resources: progressive VAE adaptation through f: 8 → 32 → 64 yields an efficient 4K initialization, group-wise ACE-LoRA continual learning with MOPD consolidation absorbs a long tail of vertical editing skills with limited forgetting, and reinforcement learn ing with the purpose-trained BluePixReward refines the aesthetic–fidelity trade-off. Distillation via sourceanchored target projection (SATP) then compresses the 50-step teacher to 4 steps, completing a 4K edit in 8.7 s on a single Kunlunxin P800 XPU while preserving task success rate and background consistency. BluePixel Agent wraps the model in a closed loop that decomposes requests across specialist and general editing tools, grounds underspecified instructions through retrieval and prompt rewriting, and verifies every candidate with bounded recovery. A unified data pipeline (a VLM-based quality gate and a four-dimensional reward model) provides one assessment basis from training-data filtering to final evaluation. On VINS-4KEval++, BluePixel ranks second only to GPT Image 2 in overall score among seven models, and second in non-edited region consistency. Together, these results suggest a path beyond the single-model paradigm that currently dominates image editing research.

Table 7 Quantitative comparison on the VINS-4KEval benchmark [11] (509 tasks, 4K). BluePixel matches the naturalness of FLUX.1 Kontext [Dev] + VINS while exceeding it on instruction adherence and detail. ImgEditJudge scores each row on instruction following (Instr. Adh.) and naturalness (Nat.); Detail is their average. BluePixel is the final model after the full training recipe. Per-column best in bold, second best underlined.
<table><tr><td>Method</td><td>Instr. Adh.↑</td><td>Nat.↑</td><td>Detail↑</td></tr><tr><td>Nano Banana 2.0</td><td>4.68</td><td>4.79</td><td>4.74</td></tr><tr><td>Seedream 4.0</td><td>4.60</td><td>4.79</td><td>4.70</td></tr><tr><td>Seedream 5.0 Lite</td><td>4.57</td><td>4.65</td><td>4.61</td></tr><tr><td>FLUX.1 Kontext [Dev] + VINS [11]</td><td>4.23</td><td>4.70</td><td>4.47</td></tr><tr><td>FLUX.1 Kontext [Dev]</td><td>4.22</td><td>4.60</td><td>4.41</td></tr><tr><td>OmniGen2</td><td>4.14</td><td>4.54</td><td>4.34</td></tr><tr><td>Step1X-Edit</td><td>4.06</td><td>4.50</td><td>4.28</td></tr><tr><td>BAGEL</td><td>4.15</td><td>4.39</td><td>4.27</td></tr><tr><td>ICEdit</td><td>3.76</td><td>4.42</td><td>4.09</td></tr><tr><td>AnyEdit</td><td>3.24</td><td>3.89</td><td>3.57</td></tr><tr><td>BluePixel</td><td>4.52</td><td>4.70</td><td>4.61</td></tr></table>

Table 8 BluePixel ranks second on VINS-4KEval++ behind GPT Image 2, with the second-best non-edited-region consistency: BluePixReward scores on the 2,513 cases shared by all models. BluePixel reports the mean of an eight-seed run with prompt rewriting (per-seed detail in Table 4); each external model has one output per case. Scores are on a 0–10 scale, with higher values indicating better performance. Success is the percentage of outputs judged successful by the same evaluator.
<table><tr><td>Method</td><td>Overall↑</td><td>IF↑</td><td>NC↑</td><td>VQ↑ Success (%)↑</td></tr><tr><td>GPT Image 2 [39]</td><td>9.15</td><td>9.22</td><td>9.34</td><td>9.42 74.8</td></tr><tr><td>Nano Banana Pro [17]</td><td>8.60</td><td>8.80</td><td>8.72</td><td>9.12 59.4</td></tr><tr><td>Nano Banana 2.0 [17]</td><td>8.56</td><td>8.81</td><td>8.63 9.08</td><td>57.9</td></tr><tr><td>Wan 2.7 Pro [1]</td><td>8.56</td><td>8.72</td><td>8.77</td><td>9.04 59.5</td></tr><tr><td>Seedream 5.0 Lite [8]</td><td>8.44</td><td>8.95</td><td>8.53</td><td>8.81 51.8</td></tr><tr><td>Kling [28]</td><td>7.90</td><td>8.01</td><td>8.37</td><td>8.55 45.4</td></tr><tr><td>BluePixel</td><td>8.62</td><td>8.74</td><td>9.06</td><td>8.90 63.2</td></tr></table>

## References

[1] Alibaba. Tongyi wanxiang (wan 2.7 pro): Text-to-image and image editing. https://tongyi.aliyun.com/ wanxiang/, 2026.

[2] Yihang Bao, Zhen Huang, Shanyan Guan, Songlin Yang, Yanhao Ge, Wei Li, Bukun Huang, and Zengmin Xu. Learnir: Learnable posterior sampling for real-world image restoration. In International Conference on Learning Representations (ICLR), 2026.

[3] Black Forest Labs. Flux.2: Frontier visual intelligence. https://bfl.ai/blog/flux-2, 2025.

[4] Black Forest Labs. FLUX.2 [klein]: Towards interactive visual intelligence. https://bfl.ai/blog/ flux2-klein-towards-interactive-visual-intelligence, 2026.

[5] Black Forest Labs. Flux 3 image. https://bfl.ai/blog/flux-3, 2026.

[6] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. FLUX.1 Kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

[7] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[8] ByteDance Seed. Deeper thinking, more accurate generation: Introducing seedream 5.0 lite. https://seed. bytedance.com/en/blog/deeper-thinking-more-accurate-generation-introducing-seedream-5-0-lite, 2026.

[9] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.

[10] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, and Haotian Tang. Deep compression autoencoder for efficient high-resolution diffusion models. In International Conference on Learning Representations (ICLR), 2025.

[11] Zhizhou Chen, Shanyan Guan, Zhanxin Gao, En Ci, Yanhao Ge, Wei Li, Zhenyu Zhang, Jian Yang, and Ying Tai. Vins-120k: Ultra high-resolution image editing with a large-scale dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[12] En Ci, Shanyan Guan, Yanhao Ge, Yilin Zhang, Wei Li, Jian Yang, Zhenyu Zhang, and Ying Tai. Describe, don’t dictate: Semantic image editing with natural language intent. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[13] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

[14] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

[15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

[16] Linqian Fan, Peiqin Sun, Tiancheng Wen, Shun Lu, and Chengru Song. $r _ { \mathrm { d m } } \colon$ Re-conceptualizing distribution matching as a reward for diffusion distillation. arXiv preprint arXiv:2603.28460, 2026.

[17] Google. Nano banana: Ai image generation and photo editing. https://gemini.google/overview/ image-generation/, 2025. Gemini AI image generator & photo editor.

[18] Shanyan Guan, Yanhao Ge, Ying Tai, Jian Yang, Wei Li, and Mingyu You. Hybridbooth: Hybrid prompt inversion for efficient subject-driven generation. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[19] Advait Gupta, NandaKiran Velaga, Dang Nguyen, and Tianyi Zhou. CoSTA<sup>∗</sup>: Cost-sensitive toolpath agent for multi-turn image editing. arXiv preprint arXiv:2503.10613, 2025.

[20] Advait Gupta, Rishie Raj, Dang Nguyen, and Tianyi Zhou. FaSTA<sup>∗</sup>: Fast-slow toolpath agent with subroutine mining for efficient multi-turn image editing. In International Conference on Learning Representations (ICLR), 2026.

[21] Tiankai Hang, Shuyang Gu, Dong Chen, Xin Geng, and Baining Guo. CCA: Collaborative competitive agents for image editing. Frontiers of Computer Science, 2025.

[22] Wenkun He, Yuchao Gu, Junyu Chen, Dongyun Zou, Yujun Lin, Zhekai Zhang, Haocheng Xi, Muyang Li, Ligeng

Zhu, Jincheng Yu, Junsong Chen, Enze Xie, Song Han, and Han Cai. DC-Gen: Post-training diffusion acceleration with deeply compressed latent space. In Proceedings of the European Conference on Computer Vision (ECCV), 2026.

[23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

[24] Yushi Huang, Xiangxin Zhou, Ruoyu Wang, Chi Zhang, Jun Zhang, and Tianyu Pang. Reinforcing few-step generators via reward-tilted distribution matching. arXiv preprint arXiv:2605.26108, 2026.

[25] Eser Kandogan, Sajjadur Rahman, Nikita Bhutani, Dan Zhang, Rafael Li Chen, and Kushan Mitra. A blueprint architecture of compound ai systems for enterprise. arXiv preprint arXiv:2406.00584, 2024.

[26] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[27] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[28] Kuaishou. Kling ai: Image generation and editing. https://klingai.com/, 2025.

[29] Quanhao Li, Junqiu Yu, Kaixun Jiang, Yujie Wei, Zhen Xing, Pandeng Li, Ruihang Chu, Shiwei Zhang, Yu Liu, and Zuxuan Wu. DiffusionOPD: A unified perspective of on-policy distillation in diffusion models. arXiv preprint arXiv:2605.15055, 2026.

[30] Sen Li, Ruochen Wang, Cho-Jui Hsieh, Minhao Cheng, and Tianyi Zhou. MuLan: Multimodal-llm agent for progressive and interactive multi-object diffusion. arXiv preprint arXiv:2402.12741, 2024.

[31] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. arXiv preprint arXiv:2108.10257, 2021.

[32] Zihan Liang, Jiahao Sun, and Haoran Ma. An llm-lvlm driven agent for iterative and fine-grained image editing. arXiv preprint arXiv:2508.17435, 2025.

[33] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

[34] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

[35] Yuehao Liu, Shanyan Guan, Weijia Zhang, Xuanming Shang, Yanhao Ge, Wei Li, and Chao Ma. Octopus: Historyfree gradient orthogonalization for continual learning in multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[36] Yuehao Liu, Weijia Zhang, Xuanming Shang, Zhizhou Chen, Yanhao Ge, Shanyan Guan, and Chao Ma. ACE-LoRA: Adaptive orthogonal decoupling for continual image editing. arXiv preprint arXiv:2605.14948, 2026.

[37] Jian Ma, Xujie Zhu, Zihao Pan, Qirong Peng, Xu Guo, Chen Chen, and Haonan Lu. X2edit: Revisiting arbitraryinstruction image editing through self-constructed data and task-aware representation learning. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026.

[38] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. International Conference on Learning Representa tions (ICLR), 2022

[39] OpenAI. GPT-Image-2. https://developers.openai.com/api/docs/models/gpt-image-2, 2026.

[40] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST), 2023.

[41] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations (ICLR), 2024.

[42] Yuhan Pu, Hao Zheng, Ziqian Mo, Zirui Pang, Hill Zhang, and Tianyi Fan. CAMEO: A conditional and quality aware multi-agent image editing orchestrator. arXiv preprint arXiv:2604.03156, 2026.

[43] Jie Qin, Jie Wu, Weifeng Chen, and Yueming Lyu. Diffusionagent: Navigating expert models for agentic image generation. arXiv preprint arXiv:2401.10061, 2024.

[44] Qwen Team. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[46] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems (NeurIPS), 2023.

[47] ByteDance Seed Team. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

[48] SenseTime. Sensenova-u1.5: Towards native unified visual intelligence. arXiv preprint arXiv:2609.11929, 2026.

[49] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with ChatGPT and its friends in Hugging Face. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[50] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089, 2023.

[51] Qwen Team. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

[52] Changyao Tian, Danni Yang, Guanzhou Chen, Erfei Cui, Zhaokai Wang, Yuchen Duan, Penghao Yin, Sitao Chen, Ganlin Yang, Mingxin Liu, et al. Internvl-u: Democratizing unified multimodal models for understanding, reasoning, generation and editing. arXiv preprint arXiv:2603.09877, 2026. Introduces the TextEdit benchmark.

[53] Feng Tian, Yixuan Li, Yichao Yan, Shanyan Guan, Yanhao Ge, and Xiaokang Yang. Postedit: Posterior sampling for efficient zero-shot image editing. In International Conference on Learning Representations (ICLR), 2025.

[54] Kavana Venkatesh, Connor Dunlop, and Pinar Yanardag. CREA: A collaborative multi-agent framework for cre ative image editing and generation. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[55] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. In Transactions on Machine Learning Research (TMLR), 2024.

[56] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Work shops, 2021.

[57] Zhenyu Wang, Enze Xie, Aoxue Li, Zhongdao Wang, Xihui Liu, and Zhenguo Li. Divide and conquer: Language models can plan and self-correct for compositional text-to-image generation. arXiv preprint arXiv:2401.15688, 2024.

[58] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

[59] Yicheng Xiao, Wenhu Zhang, Lin Song, Yukang Chen, Wenbo Li, Nan Jiang, Tianhe Ren, Haokun Lin, Wei Huang, Haoyang Huang, Xiu Li, Nan Duan, and Xiaojuan Qi. Spatialedit: Benchmarking fine-grained image spatial editing. arXiv preprint arXiv:2604.04911, 2026.

[60] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022.

[61] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. International Conference on Learning Representations (ICLR), 2023.

[62] Keming Ye, Zhipeng Huang, Canmiao Fu, Qingyang Liu, Jiani Cai, Zheqi Lv, Chen Li, Jing Lyu, Zhou Zhao, and Shengyu Zhang. Unicedit-10m: A dataset and benchmark breaking the scale-quality barrier via unified verification for reasoning-enriched edits. arXiv preprint arXiv:2512.02790, 2025.

[63] Ruijie Ye, Jiayi Zhang, Zhuoxin Liu, Zihao Zhu, Siyuan Yang, Li Li, Tianfu Fu, and Franck Dernoncourt. Agent banana: High-fidelity image editing with agentic thinking and tooling. arXiv preprint arXiv:2602.09084, 2026.

[64] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Yuan Li. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

[65] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[66] Ziyun Zeng, Hang Hua, and Jiebo Luo. MIRA: Multimodal iterative reasoning agent for image editing. arXiv preprint arXiv:2511.21087, 2025.

[67] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[68] Weijia Zhang, Yuehao Liu, Shanyan Guan, Wu Ran, Yanhao Ge, Wei Li, and Chao Ma. Guiding a diffusion model by swapping its tokens. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[69] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[70] Chen Zhao, En Ci, Yunzhe Xu, Tiehan Fan, Shanyan Guan, Yanhao Ge, Jian Yang, and Ying Tai. Ultrahr-100k: Enhancing uhr image synthesis with a large-scale high-quality dataset. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[71] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems (NeurIPS), 2024.

[72] Yiran Zhao, Yaoqi Ye, Xiang Liu, Michael Qizhe Shieh, and Trung Bui. Imageedit-R1: Boosting multi-agent image editing via reinforcement learning. arXiv preprint arXiv:2603.08059, 2026.

[73] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. In International Conference on Learning Representations (ICLR), 2026.

## Appendix

## The BluePixel Team

BluePixel was developed through the collective efforts of the following team members and collaborators.

Research & Algorithm. Shanyan Guan led the BluePixel project and designed the overall framework, core algorithms, data construction, and evaluation pipeline. Shiyu Li developed the BluePixel Agent harness. Xin Li contributed to data construction, training instruction refinement, and instruction recommendation, and developed the reward model for editing evaluation. Zhizhou Chen validated the initial training algorithm. Yuehao Liu developed the continual learning algorithm for acquiring multiple editing capabilities. Yunzhe Xu developed the distillation pipeline. Ruixue Shen developed the data construction pipeline and explored DPO-based post-training.

Chao Ma (Shanghai Jiao Tong University) and Ying Tai (Nanjing University) contributed as long-term research collaborators. Chao Ma, Ying Tai, Zhongang Qi, Wei Li, and Yanhao Ge provided valuable guidance and support throughout the project.

Engineering. Min Jin, Junbo Liu, Qiang Liu, Penggang Shi, Sheng Sun, Xiao Wang, and Yong Zhang developed and maintained the training and deployment infrastructure. Shiju Zhang contributed to training data construction. Names are listed alphabetically by surname.

## A Hairstyle Sub-templates

Table 9 lists the 26 hairstyle sub-templates used in the continual learning experiments.

Table 9 Full list of hairstyle sub-templates for the Change Hairstyle task.
<table><tr><td>#</td><td>Template</td><td>#</td><td>Template</td><td>#</td><td>Template</td></tr><tr><td>1</td><td>side-part</td><td>2</td><td>braids</td><td>3</td><td>bangs</td></tr><tr><td>4</td><td>half-up princess</td><td>5</td><td>half-up bun</td><td>6</td><td>curly</td></tr><tr><td>7</td><td>glowing strands</td><td>8</td><td>flying hair</td><td>9</td><td>short wavy</td></tr><tr><td>10</td><td>layered perm</td><td>11</td><td>long wavy</td><td>12</td><td>latin curl</td></tr><tr><td>13</td><td>side bangs</td><td>14</td><td>japanese short curly</td><td>15</td><td>japanese bangs</td></tr><tr><td>16</td><td>straight</td><td>17</td><td>sakura</td><td>18</td><td>french bangs</td></tr><tr><td>19</td><td>above-brown bangs</td><td>20</td><td>air bangs</td><td>21</td><td>elf short</td></tr><tr><td>22</td><td>textured middle</td><td>23</td><td>wool roll</td><td>24</td><td>collar-length</td></tr><tr><td>25</td><td>flowing ponytail</td><td>26</td><td>fishtail perm</td><td></td><td></td></tr></table>