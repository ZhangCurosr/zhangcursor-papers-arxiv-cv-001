# CamPilot: A Multi-Agent Cinematic Assistant for Camera-Controlled Movie Generation

Yang Wu<sup>♣∗</sup> Stefano Petrangeli<sup>♡</sup> Ishita Dasgupta<sup>♡</sup> Yu Shen<sup>♡†</sup>

<sup>♣</sup>Worcester Polytechnic Institute, Worcester, MA, USA

<sup>♡</sup>Adobe Research, San Jose, CA, USA

ywu19@wpi.edu {petrange, idasgupt, shenyu}@adobe.com

## Abstract

The integration of large language models (LLMs) into video generation has enabled rapid text-to-video creation and improved visual quality. However, it still falls short of professional filmmaking, where cinematographic language is less refined than human-crafted camera work and multi-shot continuity remains challenging. To address these limitations, we introduce CamPilot, a multi-agent framework that integrates cinematographic planning and camera-work control to produce more coher ent, logically structured, and human-aesthetic movies. CamPilot adopts a GRPO-based learning paradigm to learn camera work planning from 14K real-world professional movies, internalizing motion patterns and composition principles that support reasoning over shooting techniques (e.g., camera angle, motion, and focal behavior) and cross-shot relationships for controllable camera-viewpoint generation. Multiple agents further collaborate and evolve to improve overall output quality. To support this work and further studies in this domain, we establish CamEval, a benchmark for evaluating camera work quality and cinematic engagement. Empirical results show that CamPilot outperforms state-of-the-art text-to-movie generation methods on cinematographic control and quality, highlighting the impact of professional camera design on movie generation.

## 1 Introduction

Large language models (LLMs) are increasingly integrated into text-conditioned video generation systems, enabling rapid text-to-video creation and improved visual quality (Kondratyuk et al., 2023; Bar-Tal et al., 2024). These advances are reshaping creative workflows by allowing creators to iterate from a textual concept to a rendered clip with reduced manual effort (Bar-Tal et al., 2024; Xie et al.,

![](images/73cf4d24dd74f443fbb58a91daa23e8fb69d13bedaac80dfb238617629afcc2b.jpg)  
Figure 1: Comparison of a classic text-to-video gen-1eration model and CamPilot for text-to-movie generation. The classic model (left) directly generates a video with generic, flat shots that often miss the user’s intent, whereas CamPilot (right) first performs reasoning to plan a professional shooting script and then generates a camera-controlled movie that better expresses the intended scene and emotion.

2024). They are particularly impactful for movielike content creation, where users often expect not only photorealistic frames but also expressive camera work that conveys intent and emotion (Wu et al., 2025a; He et al., 2024). However, existing methods still fall short of professional filmmaking, where cinematographic language remains less refined than human-crafted shooting and multi-shot continuity is particularly difficult to maintain (Xie et al., 2024; Wu et al., 2025a; Li et al., 2024a). This limitation reflects a mismatch between how current textto-video models generate content and how professional filmmakers design camera work for coherent movies (He et al., 2024; Kuang et al., 2024; Li et al., 2024a). Once the story intent is known, human filmmakers typically prepare in advance by drafting a shooting script or shot list that specifies camera angle, motion, and focal behavior for each shot and coordinates how shots connect across a sequence (Courant et al., 2024; Xing et al., 2025). We argue that text-to-movie generation should follow a similar planning-first principle, achieving stronger cinematographic expressiveness while preserving coherent multi-shot structure (Lin et al., 2023; Xie et al., 2024; Wu et al., 2025a).

Motivated by this insight, we introduce CamPilot, a multi-agent framework for text-to-movie generation that integrates professional cinematographic planning and camera-work control to produce more coherent, logically structured, and human-aesthetic movies (Hu et al., 2024). As illustrated in Figure 1, a classic text-to-video generation model directly produces a video with generic, flat shots, which can miss the user’s intent even when the visual content is plausible. In contrast, CamPilot first performs reasoning to plan a professional shooting script, including shot-specific choices of camera angle, motion, and focal behavior, and then generates a camera-controlled movie that better expresses the intended scene and emotion (Geng et al., 2025; Feng et al., 2024). CamPilot adopts a GRPO-based (Shao et al., 2024) learning paradigm to learn camera grammar from 14K real-world professional movies (Bain et al., 2020), internalizing motion patterns and composition principles that support reasoning over shooting techniques and cross-shot relationships for controllable cameraviewpoint generation. Multiple agents further collaborate and evolve to improve overall output quality. For example, given a prompt that requires conveying a clear emotional intent, CamPilot can plan a time-ordered sequence with complementary shot types and transitions, then execute the plan under explicit camera control to maintain continuity and strengthen expression (Ling et al., 2025; Chen et al., 2025).

To support this work and further studies in this domain, we establish CamEval, a benchmark for evaluating camera work quality and cinematic engagement in text-to-movie generation. CamEval is designed to assess both shot-level camera execution and multi-shot consistency, aligning evaluation with the challenges of cinematographic language and continuity (Babu et al., 2025). Using CamEval, we conduct extensive experiments against stateof-the-art text-to-movie generation methods. The results demonstrate that CamPilot improves cinematographic control and overall quality over strong baselines, highlighting the effectiveness of professional camera design for movie generation.

Our contributions in this paper are threefold and can be summarized as follows:

• We propose CamPilot, a multi-agent textto-movie framework that unifies cinematographic planning with explicit camera-work control. It learns camera grammar from 14k real-world professional movies via a GRPO-based training paradigm, enabling shot-level technique reasoning and crossshot relationship modeling for controllable cameraviewpoint generation.

• We introduce CamEval, a benchmark for evaluating camera work quality and cinematic engagement, and we use it to compare CamPilot with state-of-the-art text-to-movie generation methods.

• Empirical experiments validate that camPilot not only improves professional camera-work quality, but also yields strong video generation quality, enhancing overall visual fidelity and multi-shot coherence.

## 2 Related Work

## 2.1 Video Generation

Recent progress in text-to-video generation is largely driven by diffusion-based models, which extend image diffusion to the spatiotemporal setting and improve realism and temporal coherence. Early work such as Video Diffusion Models (Ho et al., 2022) and Make-A-Video (Singer et al., 2022) established foundational architectures for modeling motion and appearance, while latent-space formulations improve efficiency and scalability (Blattmann et al., 2023b). Building on large-scale text-toimage pretraining, recent systems further enhance quality via stronger temporal modules and data scaling, including ModelScopeT2V (Wang et al., 2023b), LaVie (Wang et al., 2025), and Stable Video Diffusion (Blattmann et al., 2023a). In parallel, transformer-style video language models tokenize visual content and learn long-range dependencies with autoregressive or decoder-only objectives, enabling flexible conditioning and multimodal generation (Yan et al., 2021; Kondratyuk et al., 2023). Despite these advances, most text-to-video backbones focus on short clips and prioritize visual fidelity over cinematographic intent, leaving professional camera work and multi-shot continuity under-specified at generation time. This motivates frameworks that explicitly model camera language and cross-shot structure on top of strong video generators.

## 2.2 LLM-based Agents

Large language models have become increasingly effective as agentic planners that decompose tasks, reason over intermediate states, and invoke tools (Brown et al., 2020; Wei et al., 2022). Toolaugmented paradigms improve grounded decisionmaking by integrating external actions, including learning to call tools from supervision or selfgenerated traces (Nakano et al., 2021; Schick et al., 2023; Shen et al., 2023; Qin et al., 2023). To improve robustness, iterative refinement and selfcritique loops have been explored for language agents, where agents revise outputs based on feedback or reflections (Shinn et al., 2023; Yao et al., 2023). Beyond single-agent setups, multi-agent systems coordinate specialized roles to solve complex tasks through communication and division of labor (Park et al., 2023; Hong et al., 2023; Wang et al., 2023a; Li et al., 2023; Qian et al., 2024). These agentic abstractions are particularly relevant to long-horizon generation problems, where planning, verification, and revision naturally map to distinct roles in a production pipeline.

## 2.3 LLMs for Movie Generation

Motivated by the need for long-form and multiscene consistency, recent work uses LLMs to expand a user prompt into structured scripts, storyboards, or scene plans, and then conditions downstream generators on these intermediate representations. VideoDirectorGPT (Lin et al., 2023) demonstrates LLM-guided multi-scene planning with explicit layouts, and VideoStudio (Long et al., 2024) similarly leverages LLMs to produce multi-scene scripts to improve content consistency. Several multi-agent pipelines further decompose the process into specialized roles and incorporate iterative refinement to improve long-video coherence, including DreamFactory (Xie et al., 2024), Mora (Yuan et al., 2024), and StoryAgent (Hu et al., 2024). More explicitly film-oriented systems simulate filmmaking roles such as director, screenwriter, and cinematographer, for example MovieAgent (Wu et al., 2025a) and FilmAgent (Xu et al., 2025), while Anim-Director targets controllable animation production via an agentic workflow (Li et al., 2024b). Additionally, spatio-temporal event modeling frameworks (Cen et al., 2025) demonstrate the benefit of hierarchical reasoning over complex dynamic processes. While these systems improve narrative structure and multi-scene consistency, they typically treat camera work as a prompted attribute or a heuristic control signal, rather than learning camera grammar from real professional footage with a trainable objective. Our work complements this line by focusing on learning camera language from real movies and using it to support controllable, cross-shot cinematographic planning within a multi-agent text-to-movie pipeline.

## 3 Methodology

## 3.1 Problem Definition

Given a textual prompt S, the goal of cameracontrolled long-form movie generation is to produce a multi-scene, multi-shot movie $\hat { V }$ that is coherent in narrative and cinematic style. We aim to learn a mapping function

$$
F : S  { \hat { V } } .\tag{1}
$$

The output $\hat { V }$ is a sequence of shot clips organized by scenes:

$$
{ \hat { V } } = \{ { \hat { v } } _ { j } ^ { i } \ | \ i = 1 , \ldots , N , \ j = 1 , \ldots , M _ { i } \} ,\tag{2}
$$

where $\hat { v } _ { j } ^ { i }$ denotes the j-th shot video in the i-th scene, N is the number of scenes, and $M _ { i }$ is the number of shots in scene i. Function $F ( \cdot )$ instantiates a hierarchical pipeline that (i) expands $S$ into a structured story with scene and shot plans, and (ii) generates each shot under explicit camera-work control, so that all shots can be concatenated into the final movie.

## 3.2 CamPilot Overview

We present CamPilot, a multi-agent framework for text-to-movie generation, as illustrated in Figure 2. The key idea is to separate story planning from cinematographic execution. CamPilot first expands the user prompt into a scene-level storyline and a shot-level script, then invokes a trainable Camera Work Planner to perform reasoning over the current shot and its context to produce structured camera work. A video generator (any off-the-shelf text-to-video backbone) then synthesizes the shot conditioned on the planned camera work. Finally, an evaluator–reviser loop iteratively refines the shot description, camera work, or generation prompt until the generated shot passes quality checks. A character bank and frame-to-frame conditioning are used to improve character consistency and crossshot continuity for long-form movie synthesis.

## 3.2.1 Scene Planner

The Scene Planner expands the user prompt $S$ into a scene outline

$$
{ \mathcal { C } } = \{ c _ { 1 } , \ldots , c _ { N } \} = \Pi _ { \mathrm { s c e n e } } ( S ) ,\tag{3}
$$

![](images/d873dd6d9fc7c6e02dbc002da93ee087009acaaad8c5217d394f058c1e69b847.jpg)  
Figure 2: Overall framework of CamPilot. Given a user prompt, CamPilot first plans a scene-level storyline and a shot-level script. A trainable camera work planner then performs reasoning over the current shot and its context to produce structured camera work, which conditions a backbone video generator. An evaluator–reviser loop iteratively refines the shot description, camera work, or generation conditions until quality checks are satisfied. A character bank and frame-to-frame conditioning support character consistency and cross-shot continuity for long-form movie generation.

where each $c _ { i }$ is a scene-level description that specifies the high-level narrative intent, location, time, and overall mood of the scene. This stage provides a global scaffold that constrains later shot-level planning and helps maintain narrative coherence across scenes.

## 3.2.2 Shot Planner

Given a scene description $c _ { i } .$ , the Shot Planner decomposes the scene into an ordered shot list:

$$
{ \mathcal { D } } _ { i } = \{ d _ { i , 1 } , \ldots , d _ { i , M _ { i } } \} = \Pi _ { \mathrm { s h o t } } ( c _ { i } , S ) ,\tag{4}
$$

where each shot description $d _ { i , j }$ is a structured script containing fine-grained elements, including Title, Location, Time and Lighting, Characters, Action, Background and Environment, Mood and Tone, and Intention. This representation makes the shot intent explicit and provides sufficient context for downstream camera-work reasoning.

## 3.2.3 Camera Work Planner

The Camera Work Planner is the core trainable agent in CamPilot. For the j-th shot in scene i, we construct its planning context from three sources: (i) the scene-level description $c _ { i } .$ , (ii) the preceding shots in the same scene $\mathcal { D } _ { i , < j } = \{ d _ { i , 1 } , \dotsc , d _ { i , j - 1 } \}$ and (iii) the current shot description $d _ { i , j }$ . We denote the input context as

$$
x _ { i , j } = ( c _ { i } , \mathcal { D } _ { i , < j } , d _ { i , j } ) .\tag{5}
$$

The planner then outputs structured camera work

$$
w _ { i , j } = \Pi _ { \mathrm { c a m } } ( x _ { i , j } ) ,\tag{6}
$$

where $w _ { i , j }$ consists of three main components:

$$
w _ { i , j } = ( a _ { i , j } , \ s _ { i , j } , \ m _ { i , j } ) ,\tag{7}
$$

with camera angle $a _ { i , j }$ , shot size $s _ { i , j }$ , and camera motion $m _ { i , j }$ . We further represent motion as a 5- tuple

$$
m _ { i , j } = ( \tau _ { i , j } , \ f _ { i , j } , \ u _ { i , j } , \ \rho _ { i , j } , \ \delta _ { i , j } ) ,\tag{8}
$$

corresponding to motion type $\tau ,$ focal length $f ,$ motion speed $u ,$ rotation intensity $\rho ,$ and translation intensity δ. In Section 4, we describe how we construct supervision from real-world movies and train $\Pi _ { \mathrm { c a m } }$ with a GRPO-based learning paradigm using a structured reward that factorizes over these attributes.

## 3.2.4 Training the Camera Work Planner with GRPO

Reinforcement learning has shown strong potential in optimizing specialized LLM policies (Wu et al., 2024; Yao et al., 2025). We train the Camera Work Planner with a GRPO-based learning paradigm, which optimizes a policy to maximize task-specific rewards under a stable, regularized update. Let π<sub>θ</sub> denote the planner policy with parameters θ, which maps the planning context $x _ { i , j }$ (Equation 5) to a camera-work output $w _ { i , j }$ . We treat each camerawork attribute as a discrete decision and define the training objective over samples drawn from the current policy:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = - \mathbb { E } _ { w \sim \pi _ { \theta } ( \cdot | x ) } \big [ R ( x , w ) \big ] } \\ & { \phantom { { = } } + \beta \mathrm { K L } ( \pi _ { \theta } ( \cdot \mid x ) \parallel \pi _ { \mathrm { r e f } } ( \cdot \mid x ) ) . } \end{array}\tag{9}
$$

where x denotes a planner input context, $R ( x , w )$ is the reward for predicting camera work w under context $x , \pi _ { \mathrm { r e f } }$ is a fixed reference policy, $\operatorname { K L } ( \cdot \| \cdot )$ is the Kullback–Leibler divergence, and $\beta$ controls the strength of regularization.

Reward design. Given the ground-truth camera work label $\begin{array} { c c l } { { w ^ { \star } } } & { { = } } & { { ( a ^ { \star } , s ^ { \star } , m ^ { \star } ) } } \end{array}$ from our dataset (Section 4) and the planner prediction $\boldsymbol { w } = ( a , s , m )$ , we compute a structured reward that factorizes over camera-work attributes:

$$
\begin{array} { l } { { R ( x , w ) = \displaystyle \frac { 1 } { 3 } r _ { \mathrm { a n g l e } } ( a , a ^ { \star } ) + \displaystyle \frac { 1 } { 3 } r _ { \mathrm { s i z e } } ( s , s ^ { \star } ) } } \\ { { + \displaystyle \frac { 1 } { 3 } r _ { \mathrm { m o t i o n } } ( m , m ^ { \star } ) . } } \end{array}\tag{10}
$$

Here $r _ { \mathrm { a n g l e } }$ and $r _ { \mathrm { s i z e } }$ measure whether the predicted camera angle and shot size match the corresponding labels. For camera motion, we use the same factorization as our output space, where $m = ( \tau , f , u , \rho , \delta )$ and $m ^ { \star } = ( \tau ^ { \star } , f ^ { \star } , u ^ { \star } , \rho ^ { \star } , \delta ^ { \star } )$ denote the predicted and labeled motion attributes, respectively. We define the motion reward as an average over the five motion attributes:

$$
\begin{array} { l } { { \displaystyle r _ { \mathrm { m o t i o n } } ( m , m ^ { \star } ) = \frac { 1 } { 5 } r _ { \mathrm { t y p e } } ( \tau , \tau ^ { \star } ) + \frac { 1 } { 5 } r _ { \mathrm { f o c a l } } ( f , f ^ { \star } ) } } \\ { { ~ + ~ \frac { 1 } { 5 } r _ { \mathrm { s p e e d } } ( u , u ^ { \star } ) + \frac { 1 } { 5 } r _ { \mathrm { r o t } } ( \rho , \rho ^ { \star } ) } } \\ { { ~ + ~ \frac { 1 } { 5 } r _ { \mathrm { t r a n s } } ( \delta , \delta ^ { \star } ) . } } \end{array}\tag{11}
$$

Each component reward $r ( \cdot , \cdot )$ compares the predicted class against the labeled class for the corresponding attribute. This design encourages the planner to learn camera grammar at both the coarse level (angle, size, motion) and the fine-grained level (motion sub-attributes), while producing a complete, structured camera-work specification.

## 3.2.5 Agent Evolution via Evaluation and Revision

Long-form generation is sensitive to accumulated errors, such as inconsistent characters, abrupt camera changes, or visually unstable motion. CamPilot addresses this with an evolution loop that iteratively evaluates and revises each shot until it meets quality requirements.

Evaluator. After generating $\hat { v } _ { j } ^ { i }$ , an evaluator E inspects the shot and returns a binary decision and textual feedback:

$$
( \mathrm { o k } _ { i , j } , g _ { i , j } ) = E ( \hat { v } _ { j } ^ { i } ) ,\tag{12}
$$

where $\mathrm { o k } _ { i , j } \in \{ 0 , 1 \}$ indicates whether the shot passes checks, and $g _ { i , j }$ describes failure reasons (e.g., inconsistency or lack of smoothness).

Reviser. If $\mathrm { o k } _ { i , j } = 0 $ , a reviser agent R takes the feedback $g _ { i , j }$ and produces a refinement by editing one of three targets: the shot description $d _ { i , j }$ , the planned camera work $w _ { i , j } .$ , or the generator-side prompt/conditions used by G:

$$
( d _ { i , j } ^ { \prime } , w _ { i , j } ^ { \prime } , \mathrm { c o n d } ^ { \prime } ) = R ( d _ { i , j } , w _ { i , j } , g _ { i , j } ) .\tag{13}
$$

We then re-generate the shot with the refined inputs and repeat the evaluate–revise cycle up to a maximum number of iterations. The accepted shot is appended to the movie sequence, and its ending frame is used to condition the next shot, enabling long-form synthesis with improved cinematic continuity.

## 3.2.6 Character Bank and Video Generation

To improve identity consistency, inspired by memory modeling in specialized LLM agents (Wu et al., 2025b), CamPilot maintains a Character Bank B that stores a reference item for each character, such as a character identifier and a reference image. For shot $( i , j )$ , we retrieve relevant character references $B _ { i , j }$ based on $d _ { i , j }$ and incorporate them into generation conditions. We then construct the generation input for a backbone video generator G as

$$
\hat { v } _ { j } ^ { i } = G \big ( d _ { i , j } , ~ w _ { i , j } , ~ \mathcal { B } _ { i , j } , ~ \hat { e } _ { j - 1 } ^ { i } \big ) ,\tag{14}
$$

where $\hat { e } _ { j - 1 } ^ { i }$ denotes the ending frame of the previous shot in the same scene. We use $\hat { e } _ { j - 1 } ^ { i }$ as the start-frame condition for the current shot whenever available, which encourages smoother transitions and better cross-shot continuity for long-form movie generation. Backbone G can be instantiated by any existing text-to-video model, and CamPilot focuses on improving controllability and cinematic structure through planning and refinement.

## 4 Dataset Construction

Learning professional camera grammar requires scalable supervision that links real-world movie footage to structured camera-work attributes. To this end, we construct CamEval, a camera-work dataset distilled from 14K real-world professional movies, where each instance pairs a clip-level visual observation with a structured shooting script and camera-work labels. Starting from the condensed movie collection, we first segment each source movie into a set of short clips.

<table><tr><td># Videos</td><td>14,581</td></tr><tr><td># Shots</td><td>99,975</td></tr><tr><td>Movie years</td><td>2019-2020</td></tr><tr><td>Movie Genres</td><td>general</td></tr><tr><td># Train videos</td><td>12,641</td></tr><tr><td># Train shots</td><td>85,048</td></tr><tr><td># Validation videos</td><td>1,000</td></tr><tr><td># Validation shots</td><td>6,663</td></tr><tr><td># Test videos</td><td>1,000</td></tr><tr><td># Test shots</td><td>6,578</td></tr></table>

Table 1: Data statistics of CamEval.

For each clip, we uniformly sample a sequence of frames to capture both appearance and motion cues while keeping annotation cost manageable. We then prompt a vision–language model, Qwen/Qwen3-VL-32B-Instruct (Bai et al., 2025), to summarize the sampled frames into a structured shooting script, including (i) a compact description of the scene and shot (e.g., location, characters, action, mood, and intention) and (ii) the corresponding camera work. The camera-work labels follow the same schema as our Camera Work Planner output space: camera angle, shot size, and camera motion, where motion is further decomposed into motion type, focal length, motion speed, rotation intensity, and translation intensity. These VLMproduced shooting scripts provide training supervision for our camera-work planner under the same contextual inputs used in CamPilot, including the scene-level description, preceding shots as crossshot context, and the current shot description. The detailed data statistics are provided in Table 1. We use these labels to train the Camera Work Planner with a GRPO-based objective (Section 3.2.4).

## 5 Experiments

## 5.1 Experimental Settings

Our CamPilot text-to-movie pipeline contains a scene planner, a shot planner, a trainable Camera Work Planner, and an evaluator–reviser loop for iterative refinement. Among these components, only the Camera Work Planner is trained, while the other agents operate via prompting and tool orchestration. The Camera Work Planner is trained on CamEval with GRPO using TRL, where we optimize the planner policy to maximize the structured camera-work reward defined in Section 3.2.4. We use AdamW with learning rate $2 \times 1 0 ^ { - 5 }$ and batch size 8, and train for 4 epochs. Unless otherwise specified, we set temperature to 0.8 and top\_p to 1.0 during generation. Our default backbone video generator is Adobe Firefly, and we run three random seeds and report mean and standard deviation. Experiments are implemented using TRL (von Werra et al., 2022), Transformers (Wolf et al., 2020), and PyTorch (Paszke et al., 2019) on a 64-core CPU and eight 80GB A100 GPUs.

## 5.2 Baselines

We compare CamPilot with four representative baselines: Standard, Vanilla-SFT, DreamFactory (Xie et al., 2024), and MovieAgent (Wu et al., 2025a) (Table 2). Standard predicts structured camera work from the same planner input context without training. Vanilla-SFT fine-tunes the Camera Work Planner with supervised learning on CamEval to predict camera-work labels. DreamFactory is an in-context learning baseline that uses demonstrations to prompt the planner backbone to output camera work. MovieAgent is a chain-of-thought baseline that performs multistep reasoning for camera planning. We conduct experiments on three open-source planner backbones: Llama-3.1-8B-Instruct (Meta, 2024), Gemma-3-12B-it (Google, 2025), and Qwen2.5- 7B-Instruct (Hui et al., 2024). We also report Standard results with closed-source planner backbones, including DeepSeek-R1 (DeepSeek AI, 2025), Claude Sonnet 4.5 (Anthropic, 2025), and GPT-5 (OpenAI, 2025). For all methods, we use the same planner input context (Equation 5) and the same camera-work label schema (Section 3.2.3). Unless otherwise specified, we fix the downstream video generator to the same Firefly setting to isolate the effect of camera-work planning.

## 5.3 Tasks and Metrics

We evaluate all methods on three tasks with automatic metrics (Table 2). Camera Work Classification We formulate camera-work planning as a multi-attribute classification problem over camera angle, shot size, and camera motion. Camera motion is further factorized into motion type, focal length, motion speed, rotation intensity, and translation intensity. We report Macro-Acc, Macro-Rec, Macro-Prec, and Macro-F1, computed over the discrete label space defined by our camerawork schema. Keyframe Generation We evaluate the visual quality of keyframes using CLIPScore based on CLIP (Radford et al., 2021) and Inception Score (Salimans et al., 2016). Movie Generation We evaluate long-form generation quality using VBench (Huang et al., 2024). Specifically, we report Subject Consistency (Sub\_Cons), which measures subject identity stability across consecutive shots, and Aesthetic, which evaluates the overall aesthetic quality of temporal transitions between shots.

<table><tr><td rowspan="2">CameraWorkPlanner LLM Backbone</td><td rowspan="2">Method</td><td colspan="4">Camera Work Classification (%) ↑</td><td colspan="2">Keyframe Generation (%) ↑</td><td colspan="2">Movie Generation (%) ↑</td></tr><tr><td>Macro-Acc</td><td>Macro-Rec</td><td>Macro-Prec</td><td>Macro-F1</td><td>CLIP</td><td>Inception</td><td>Sub_Cons</td><td>Aesthetic</td></tr><tr><td colspan="9">Baselines with Closed-Source LLMs</td><td></td></tr><tr><td>Deepseek-R1</td><td>Standard</td><td>52.7 (0.1)</td><td>40.5 (0.8)</td><td>45.7 (2.0)</td><td>36.0 (0.8)</td><td>20.9 (0.4)</td><td>9.4 (0.3)</td><td>91.2 (0.5)</td><td>55.8 (0.6)</td></tr><tr><td>Claude-Sonnet-4.5 GPT-5</td><td>Standard</td><td>60.1 (0.1)</td><td>51.6 (1.3)</td><td>55.0 (2.2)</td><td>44.8 (1.1)</td><td>21.8 (0.4)</td><td>9.9 (0.3)</td><td>93.4 (0.4)</td><td>57.9 (0.6)</td></tr><tr><td></td><td>Standard</td><td>53.2 (0.1)</td><td>41.9 (1.7)</td><td>47.0 (3.9)</td><td>35.4 (1.9)</td><td>22.0 (0.4)</td><td>10.1 (0.3)</td><td>94.0 (0.4)</td><td>58.6 (0.6)</td></tr><tr><td colspan="9">Baselines and CamPilot with the same Open-Source LLM Backbones</td><td></td></tr><tr><td rowspan="5">Llama-3.1-8B-Instruct</td><td>Standard</td><td>45.5 (0.0)</td><td>12.2 (0.6)</td><td>13.1 (0.1)</td><td>9.8 (0.3)</td><td>18.9 (0.5)</td><td>8.6 (0.4)</td><td>87.6 (0.6)</td><td>51.2 (0.8)</td></tr><tr><td>Vanilla-SFT</td><td>61.4 (0.2)</td><td>15.1 (1.2)</td><td>14.9 (1.6)</td><td>13.7 (1.1)</td><td>20.8 (0.4)</td><td>9.3 (0.3)</td><td>92.1 (0.5)</td><td>56.0 (0.7)</td></tr><tr><td>DreamFactory</td><td>58.0 (0.2)</td><td>20.0 (1.3)</td><td>19.0 (1.2)</td><td>18.5 (1.1)</td><td>20.2 (0.4)</td><td>9.1 (0.3)</td><td>91.0 (0.5)</td><td>54.8 (0.7)</td></tr><tr><td>MovieAgent</td><td>59.5 (0.2)</td><td>22.0 (1.2)</td><td>21.0 (1.1)</td><td>20.0 (1.0)</td><td>20.6 (0.4)</td><td>9.2 (0.3)</td><td>91.4 (0.5)</td><td>55.4 (0.7)</td></tr><tr><td>CamPilot (Ours)</td><td>63.7 (0.2)</td><td>28.5 (1.0)</td><td>27.1 (0.8)</td><td>27.2 (0.9)</td><td>22.0 (0.3)</td><td>9.9 (0.3)</td><td>95.0 (0.4)</td><td>59.0 (0.6)</td></tr><tr><td rowspan="5">Gemma-3-12b-it</td><td>Standard</td><td>56.7 (0.2)</td><td>38.8 (2.7)</td><td>34.7 (2.5)</td><td>34.2 (2.5)</td><td>20.4 (0.5)</td><td>9.4 (0.4)</td><td>90.6 (0.6)</td><td>54.7 (0.8)</td></tr><tr><td>Vanilla-SFT</td><td>62.6 (0.0)</td><td>37.3 (1.9)</td><td>36.5 (1.2)</td><td>36.1 (1.4)</td><td>21.2 (0.4)</td><td>9.8 (0.3)</td><td>93.0 (0.5)</td><td>56.8 (0.7)</td></tr><tr><td>DreamFactory</td><td>60.8 (0.1)</td><td>35.2 (2.2)</td><td>33.9 (2.0)</td><td>33.0 (2.0)</td><td>20.8 (0.4)</td><td>9.6 (0.3)</td><td>92.1 (0.5)</td><td>55.8 (0.7)</td></tr><tr><td>MovieAgent</td><td>61.9 (0.1)</td><td>36.5 (2.1)</td><td>37.2 (1.9)</td><td>34.5 (1.9)</td><td>21.0 (0.4)</td><td>9.7 (0.3)</td><td>92.5 (0.5)</td><td>56.2 (0.7)</td></tr><tr><td>CamPilot (Ours)</td><td>66.0 (0.2)</td><td>40.3 (0.8)</td><td>38.6 (3.0)</td><td>33.6 (0.6)</td><td>22.3 (0.3)</td><td>10.2 (0.3)</td><td>95.6 (0.4)</td><td>59.4 (0.6)</td></tr><tr><td rowspan="5">Qwen2.5-7B-Instruct</td><td>Standard</td><td>46.4 (0.1)</td><td>18.1 (0.1)</td><td>19.8 (0.2)</td><td>14.0 (0.1)</td><td>19.2 (0.5)</td><td>8.8 (0.4)</td><td>88.4 (0.6)</td><td>52.0 (0.8)</td></tr><tr><td>Vanilla-SFT</td><td>60.9 (0.1)</td><td>23.3 (0.2)</td><td>21.0 (0.2)</td><td>20.6 (0.2)</td><td>20.9 (0.4)</td><td>9.5 (0.3)</td><td>92.4 (0.5)</td><td>56.1 (0.7)</td></tr><tr><td>DreamFactory</td><td>57.6 (0.1)</td><td>26.0 (0.9)</td><td>24.0 (0.8)</td><td>23.5 (0.8)</td><td>20.2 (0.4)</td><td>9.2 (0.3)</td><td>91.0 (0.5)</td><td>54.3 (0.7)</td></tr><tr><td>MovieAgent</td><td>58.8 (0.1)</td><td>27.5 (0.8)</td><td>25.6 (0.8)</td><td>25.0 (0.8)</td><td>20.5 (0.4)</td><td>9.3 (0.3)</td><td>91.6 (0.5)</td><td>55.0 (0.7)</td></tr><tr><td>CamPilot (Ours)</td><td>62.7 (0.0)</td><td>30.7 (0.3)</td><td>29.8 (4.0)</td><td>29.6 (0.4)</td><td>21.7 (0.3)</td><td>10.0 (0.3)</td><td>95.1 (0.4)</td><td>58.9 (0.6)</td></tr></table>

Table 2: Comparative results with Automatic metrics on camera work classification and keyframe/movie generation. The mean (std) over three random runs is reported. The best results are shown in bold, and the second-best are underlined.

## 5.4 Experimental Results

Table 2 demonstrates that CamPilot consistently achieves the best camera-work classification performance when controlling for the same open-source backbone, indicating that the gains come from our training and planning framework rather than model size. For Llama-3.1-8B-Instruct, CamPilot improves Macro-Acc from 45.5 (Standard) to 63.7, and also yields clear gains over prompting baselines such as DreamFactory (58.0) and MovieAgent (59.5). Beyond classification, CamPilot also delivers the strongest downstream generation quality within each backbone family, achieving the best CLIP and Inception scores and improving crossshot subject stability and aesthetics, as reflected by Sub\_Cons around 95 and Aesthetic around 59 across backbones (e.g., Llama: 95.0/59.0, Gemma:

95.6/59.4, Qwen: 95.1/58.9), which is consistent with our goal of learning camera grammar for coherent multi-shot planning. Importantly, CamPilot remains competitive against closed-source commercial LLMs despite the substantial parameter gap. In terms of Macro-Acc, CamPilot exceeds the strongest commercial baseline in our comparison, achieving 63.7 versus 60.1 from Claude-Sonnet-4.5, while also outperforming GPT-5 (53.2) and Deepseek-R1 (52.7). Although commercial models can be stronger on certain precision or recall dimensions, these differences are expected given their much larger capacity. Overall, the results confirm that CamPilot provides consistent and reliable accuracy improvements under fixed open-source backbones, while narrowing the performance gap to proprietary systems.

## 5.5 Qualitative Results

Figure 3 presents qualitative comparisons between CamPilot and baseline methods. Overall, CamPilot produces more dynamic and expressive camera work, with clearer motion intent and more coherent shot progression across scenes. Compared to baselines that often generate static or weakly varied camera behaviors, our results exhibit richer camera movements and more deliberate changes in shot composition, leading to videos that appear more cinematic and engaging. These qualitative examples align with the quantitative gains in camera work classification and generation metrics, and further demonstrate CamPilot’s ability to translate structured camera planning into visually compelling multi-shot videos.

Prompt: A time- lapse of seasons changing in a mystical forest  
![](images/d69fb66bbc3d7f41c0a5a3bd20ddcb8935ddd0bdb4907ff27fa37ffcd803fd2a.jpg)

Figure 3: Qualitative comparison between MovieAgent and CamPilot.
<table><tr><td rowspan="2">LLM Backbone</td><td rowspan="2">Method</td><td colspan="4">Camera Work Classification (%) ↑</td><td colspan="2">Keyframe Generation (%) ↑</td><td colspan="2">Movie Generation (%) ↑</td></tr><tr><td>Macro-Acc</td><td>Macro-Rec</td><td>Macro-Prec</td><td>Macro-F1</td><td>CLIP</td><td>Inception</td><td>Sub_Cons</td><td>Aesthetic</td></tr><tr><td rowspan="5">Llama-3.1-8B-Instruct</td><td>Standard</td><td>44.3 (0.1)</td><td>11.6 (0.6)</td><td>12.5 (0.2)</td><td>9.1 (0.3)</td><td>18.9 (0.5)</td><td>8.6 (0.4)</td><td>87.6 (0.6)</td><td>51.2 (0.8)</td></tr><tr><td>Vanilla-SFT</td><td>59.8 (0.2)</td><td>26.7 (2.0)</td><td>25.4 (1.8)</td><td>25.7 (1.9)</td><td>20.8 (0.4)</td><td>9.3 (0.3)</td><td>92.1 (0.5)</td><td>56.0 (0.7)</td></tr><tr><td>DreamFactory</td><td>56.8 (0.2)</td><td>19.1 (1.3)</td><td>18.2 (1.2)</td><td>17.8 (1.1)</td><td>20.2 (0.4)</td><td>9.1 (0.3)</td><td>91.0 (0.5)</td><td>54.8 (0.7)</td></tr><tr><td>MovieAgent</td><td>58.2 (0.2)</td><td>21.1 (1.2)</td><td>20.0 (1.1)</td><td>19.2 (1.0)</td><td>20.6 (0.4)</td><td>9.2 (0.3)</td><td>91.4 (0.5)</td><td>55.4 (0.7)</td></tr><tr><td>CamPilot (Ours)</td><td>62.1 (0.2)</td><td>27.6 (1.1)</td><td>26.3 (0.8)</td><td>26.5 (0.9)</td><td>22.0 (0.3)</td><td>9.9 (0.3)</td><td>95.0 (0.4)</td><td>59.0 (0.6)</td></tr></table>

Table 3: Single-shot results with Automatic metrics on camera work classification and keyframe/movie generation. The mean (std) over three random runs is reported. The best results are shown in bold, and the second-best are underlined.

## 5.6 Ablation Study

Table 3 shows that CamPilot remains best in the single-shot setting, improving Macro-Acc from 44.3 to 62.1 and Macro-F1 from 9.1 to 26.5, and outperforming Vanilla-SFT (59.8/25.7). It also achieves the strongest generation quality (CLIP/Inception 22.0/9.9) and higher Sub\_Cons/Aesthetic (95.0/59.0) than Standard (87.6/51.2).

## 5.7 Discussion

CamPilot frames camera-work planning as a context-aware, structured decision process. By jointly considering scene-level intent, precedingshot context, and the current shot description, the Camera Work Planner produces camera specifications that are aligned not only with individual shot content but also with the progression of a multishot movie. The GRPO objective further supports this formulation by optimizing a factorized reward over complementary camera attributes, including angle, shot size, motion, focal behavior, speed, and motion intensity. This design is particularly beneficial for long-form generation, where effective cinematography requires coordinated transitions and shot relationships instead of independently selected camera tags.

CamEval provides scalable supervision for this task through a constrained and reproducible camera-work schema. Using Qwen3-VL-32B-Instruct enables consistent annotation of a large movie-derived corpus while keeping the label space grounded in standard cinematographic attributes. The evaluator–reviser loop further complements planning by providing an explicit mechanism to identify and refine shots that do not sufficiently satisfy the intended camera work or continuity requirements. Although this iterative procedure introduces additional planning steps, the refinement process is bounded and is designed to improve the reliability of generated multi-shot outputs. The consistent gains across camera-work classification, subject consistency, and aesthetic-quality metrics indicate that structured camera planning transfers effectively to downstream video generation. Future work can further extend this framework through larger-scale human studies, additional video backbones, and broader validation of camera-work annotations across diverse cinematic styles.

## 6 Conclusion and Future Work

We introduce CamPilot, a multi-agent framework for camera-controlled text-to-movie generation. CamPilot separates story planning from cinematographic execution by first constructing scene and shot scripts, then training a Camera Work Planner with GRPO on CamEval to learn camera grammar from real-world professional movies. This design enables structured, controllable camera work that better supports multi-shot continuity and improves overall cinematic quality. Experiments on CamEval show that CamPilot consistently improves camera-work classification and downstream generation quality across multiple planner backbones compared with baselines. In future work, we will extend CamPilot to stronger video backbones and richer evaluation protocols, and explore more adaptive planning and revision strategies to improve long-form coherence, character consistency, and user-controllable cinematic styles.

## Limitations

While CamPilot demonstrates promising results in improving camera-work planning and controllable text-to-movie generation, we acknowledge several limitations of the current work:

Coverage of Cinematographic Language. CamPilot learns camera grammar from a finite set of professional movies and a fixed label schema. As a result, the planner may not fully capture rare or highly stylized cinematographic patterns, such as unconventional lens behavior, complex blocking, or genre-specific shot conventions. In addition, our discrete camera-work labels can underrepresent continuous creative variations in camera movement and composition. Future work can expand coverage by enlarging the source corpus, refining the taxonomy, and exploring discrete-continuous representations that better reflect real cinematography.

• Planning Faithfulness and Error Propagation. CamPilot relies on a planning-first pipeline, where the shooting script and camera-work plan guide downstream generation. If the planner misinterprets the prompt intent or outputs an implausible plan, the resulting video can deviate from the desired narrative or exhibit inconsistent motion, and such errors can propagate across shots. While we design the planner context to provide scene information, planning faithfulness remains a key bottleneck for multi-shot generation. Future work will investigate stronger plan validation, self-consistency checks, and constrained decoding to reduce invalid or contradictory camera plans before execution.

• Generator Dependence and Limited Control Authority. Our evaluation fixes the downstream video generator to a single Firefly setting to isolate the effect of camera-work planning. This choice improves experimental control, but it limits conclusions about generalization across generators and about the absolute controllability achievable in different systems. Moreover, even with explicit camera controls, the generator may not perfectly follow the planned camera work due to model limitations or conflicts between content realization and motion constraints. Future work can broaden evaluation to additional generators and study control-aware training or feedback mechanisms that align generation behavior with planned camera trajectories.

• Benchmark Scope and Human Preference Alignment. CamEval focuses on camera work quality and cinematic engagement, but it does not exhaustively cover all factors that influence user satisfaction, such as storytelling coherence, visual realism, and aesthetic preference. In addition, automatic metrics may not fully reflect human judgments of cinematic quality, especially for subtle transitions or creative shot choices. Future work can extend CamEval with richer human evaluation protocols, preference modeling, and task-specific rubrics that better capture the end-to-end movie creation experience.

## Ethics Statement

After reviewing the ACL Ethics Policy, we confirm that this work complies with the relevant ethical guidelines. All data used in this study are derived from publicly available sources, and we do not use private or personally identifying information.

## Acknowledgment

We gratefully acknowledge the support and collaboration from Worcester Polytechnic Institute and Adobe Research. We also thank the reviewers and colleagues for their helpful comments and feedback.

## References

Anthropic. 2025. Claude sonnet 4.5. https://www. anthropic.com/news/claude-sonnet-4-5. Accessed: 2026-01-06.

Nithin C Babu, Aniruddha Mahapatra, Harsh Rangwani, Rajiv Soundararajan, and Kuldeep Kulkarni. 2025. Dynamiceval: Rethinking evaluation for dynamic text-to-video synthesis. arXiv preprint arXiv:2510.07441.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3- vl technical report. arXiv preprint arXiv:2511.21631.

Max Bain, Arsha Nagrani, Andrew Brown, and Andrew Zisserman. 2020. Condensed movies: Story based retrieval with contextual embeddings. In Asian Conference on Computer Vision, pages 460–479. Springer.

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. 2024. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023a. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023b. Align your latents: Highresolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Mengqi Cen, Xuejing Meng, X Joan Hu, Juxin Liu, and Jianhong Wu. 2025. Pde-based bayesian hierarchical modeling for event spread, with application to covid-19 infection. arXiv preprint arXiv:2509.13174.

Yubin Chen, Xuyang Guo, Zhenmei Shi, Zhao Song, and Jiahao Zhang. 2025. T2vworldbench: A benchmark for evaluating world knowledge in text-to-video generation. arXiv preprint arXiv:2507.18107.

Robin Courant, Nicolas Dufour, Xi Wang, Marc Christie, and Vicky Kalogeiton. 2024. Et the exceptional trajectories: Text-to-camera-trajectory generation with character awareness. In European Conference on Computer Vision, pages 464–480. Springer.

DeepSeek AI. 2025. deepseek-ai/deepseek-r1 model card. https://huggingface.co/deepseek-ai/ DeepSeek-R1. Accessed: 2026-01-06.

Wanquan Feng, Jiawei Liu, Pengqi Tu, Tianhao Qi, Mingzhen Sun, Tianxiang Ma, Songtao Zhao, Siyu Zhou, and Qian He. 2024. I2vcontrol-camera: Precise video camera control with adjustable motion strength. arXiv preprint arXiv:2411.06525.

Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Yusuf Aytar, Michael Rubinstein, Chen Sun, et al. 2025. Motion prompting: Controlling video generation with motion trajectories. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1–12.

Google. 2025. google/gemma-3-12b-it model card. https://huggingface.co/google/ gemma-3-12b-it. Accessed: 2026-01-06.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. 2024. Cameractrl: Enabling camera control for text-tovideo generation. arXiv preprint arXiv:2404.02101.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. 2022. Video diffusion models. Advances in neural information processing systems, 35:8633–8646.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al.

2023. Metagpt: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations.

Panwen Hu, Jin Jiang, Jianqi Chen, Mingfei Han, Shengcai Liao, Xiaojun Chang, and Xiaodan Liang. 2024. Storyagent: Customized storytelling video generation via multi-agent collaboration. arXiv preprint arXiv:2411.04925.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. 2024. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. 2024. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. 2023. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125.

Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas J Guibas, and Gordon Wetzstein. 2024. Collaborative video diffusion: Consistent multi-video generation with camera control. Advances in Neural Information Processing Systems, 37:16240–16271.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel: Communicative agents for" mind" exploration of large language model society. Advances in Neural Information Processing Systems, 36:51991–52008.

Xiaozhe Li, Kai Wu, Siyi Yang, YiZhan Qu, Guohua Zhang, Zhiyu Chen, Jiayao Li, Jiangchuan Mu, Xiaobin Hu, Wen Fang, et al. 2024a. Can video generation replace cinematographers? research on the cinematic language of generated video. arXiv preprint arXiv:2412.12223.

Yunxin Li, Haoyuan Shi, Baotian Hu, Longyue Wang, Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang. 2024b. Anim-director: A large multimodal model powered agent for controllable animation video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11.

Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. 2023. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091.

Xinran Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. 2025. Vmbench: A benchmark for perception-aligned video motion

generation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 13087– 13098.

Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. 2024. Videostudio: Generating consistent-content and multi-scene videos. In European Conference on Computer Vision, pages 468–485. Springer.

Meta. 2024. meta-llama/llama-3.1-8b-instruct model card. https://huggingface.co/meta-llama/ Llama-3.1-8B-Instruct. Accessed: 2026-01-06.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted questionanswering with human feedback. arXiv preprint arXiv:2112.09332.

OpenAI. 2025. Introducing gpt-5. https:// openai.com/index/introducing-gpt-5/. Accessed: 2026-01-06.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. 2024. Chatdev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15174–15186.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2023. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. 2016. Improved techniques for training gans. Advances in neural information processing systems, 29.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Textto-video generation without text-video data. arXiv preprint arXiv:2209.14792.

Leandro von Werra, Younes Belkada, Luke Tunstall, Edward Beeching, Thomas Thrush, Nathan Lambert, Shixiang Huang, Kashif Rasul, and Quentin Gallouëdec. 2022. Trl: Transformer reinforcement learning library. https://github.com/huggingface/ trl. Accessed: 2025-11.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023a. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023b. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. 2025. Lavie: Highquality video generation with cascaded latent diffusion models. International Journal of Computer Vision, 133(5):3059–3078.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural

language processing. In Proceedings ofthe 2020 conference on empirical methods in natural language processing: system demonstrations, pages 38–45.

Weijia Wu, Zeyu Zhu, and Mike Zheng Shou. 2025a. Automated movie generation via multi-agent cot planning. arXiv preprint arXiv:2503.07314.

Yang Wu, Chenghao Wang, Ece Gumusel, and Xiaozhong Liu. 2024. Knowledge-infused legal wisdom: Navigating llm consultation through the lens of diagnostics and positive-unlabeled reinforcement learning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 15542–15555.

Yang Wu, Rujing Yao, Tong Zhang, Yufei Shi, Zhuoren Jiang, Zhushan Li, and Xiaozhong Liu. 2025b. Teaching according to students’ aptitude: Personalized mathematics tutoring via persona-, memory-, and forgetting-aware llms. arXiv preprint arXiv:2511.15163.

Zhifei Xie, Daniel Tang, Dingwei Tan, Jacques Klein, Tegawend F Bissyand, and Saad Ezzini. 2024. Dreamfactory: Pioneering multi-scene long video generation with a multi-agent framework. arXiv preprint arXiv:2408.11788.

Jinbo Xing, Long Mai, Cusuh Ham, Jiahui Huang, Aniruddha Mahapatra, Chi-Wing Fu, Tien-Tsin Wong, and Feng Liu. 2025. Motioncanvas: Cinematic shot design with controllable image-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11.

Zhenran Xu, Longyue Wang, Jifang Wang, Zhouyi Li, Senbao Shi, Xue Yang, Yiyu Wang, Baotian Hu, Jun Yu, and Min Zhang. 2025. Filmagent: A multi-agent framework for end-to-end film automation in virtual 3d spaces. arXiv preprint arXiv:2501.12909.

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. 2021. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157.

Rujing Yao, Yang Wu, Chenghao Wang, Jingwei Xiong, Fang Wang, and Xiaozhong Liu. 2025. Elevating legal llm responses: harnessing trainable logical structures and semantic knowledge with legal reasoning. In Proceedings of the 2025 Conference of the Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5630–5642.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822.

Zhengqing Yuan, Yixin Liu, Yihan Cao, Weixiang Sun, Haolong Jia, Ruoxi Chen, Zhaoxu Li, Bin Lin, Li Yuan, Lifang He, et al. 2024. Mora: Enabling generalist video generation via a multi-agent framework. arXiv preprint arXiv:2403.13248.

## A Performance Comparison on Subtasks

Table 4 compares subtask-level camera work classification across camera angle, shot size, and multiple motion attributes. CamPilot consistently achieves the best Macro-ACC on all subtasks, for example improving Camera Angle Macro-ACC to 92.7 (vs. 84.0 for MovieAgent and 80.4 for DreamFactory) and Shot Size Macro-ACC to 83.7 (vs. 78.3 for Standard). The gains are especially clear on motion-related subtasks: on Camera Motion (Translation), CamPilot reaches 61.2 Macro-ACC with 31.0 Macro-Prec, 30.4 Macro-Rec, and 28.3 Macro-F1, outperforming the strongest baseline (MovieAgent) at 51.8/25.3/24.4/24.2. Similarly, on Camera Motion (Rotation), CamPilot achieves 86.1 Macro-ACC and leads Macro-Prec/Macro-Rec/Macro-F1 at 25.5/28.9/26.2, while MovieAgent attains 70.4 Macro-ACC. On Focal Length, CamPilot further raises Macro-ACC to 92.2 and recall to 41.0, compared to 87.1 Macro-ACC and 31.7 recall for MovieAgent. Overall, these results show that CamPilot improves both correctness and robustness across diverse subtasks, with the largest margins appearing on motion translation, rotation, and focal length where cross-shot camera reasoning is most needed.

## B Prompt

In this section, we present the prompt used for shooting script generation. The prompt frames the model as a professional cinematographer and camera operator for camera-work planning in textto-movie generation. Given a sequence of timeordered sampled frames from a short clip, the model is instructed to produce a structured and reproducible annotation that (i) describes the shot content at a high level using only observable visual evidence, (ii) predicts camera-work attributes under a fixed label schema, and (iii) provides a brief intent-oriented rationale grounded in what is visible. The prompt further specifies the annotation scope and temporal assumption: multiple frames are treated as consecutive moments within the same shot unless a cut is evident, in which case the model annotates the dominant shot and records a short cut note. To improve robustness and reproducibility, we include reliability constraints that encourage conservative judgments for ambiguous cases, enforce internal consistency across fields, and avoid subjective or speculative descriptions.

## Shooting Script Generation

## <|im\_start|>system

You are a professional cinematographer and camera operator for camera-work planning in text-tomovie generation. You will receive a sequence of time-ordered frames from a short clip, and you must produce a structured, reproducible annotation that captures cinematic shot design and camera behavior.

Your responsibilities:

\- Describe the shot content at a high level using only observable visual evidence.

\- Infer camera-work attributes using standard cinematography language and a fixed label schema.

\- Provide a brief intent-oriented rationale grounded in what is visible.

\- Output strictly valid JSON with the required fields and no extra text.

## Annotation scope:

\- Focus on camera-related attributes: shot size, shot angle, camera motion, focal behavior, motion speed, and motion intensity.

\- If multiple frames are provided, treat them as consecutive moments from the same shot unless a cut is evident.

\- If a cut is evident, annotate the dominant shot and record a short cut note.

Safety and privacy constraints:

\- Do not infer or reveal identities, personal attributes, or private information.

\- Do not name real persons, brands, or copyrighted titles.

\- Do not speculate about unseen events, ofscreen objects, or backstory.

Reliability guidelines:

\- Use conservative judgments. If an attribute is ambiguous, select the closest neutral label (e.g., eye-level, medium, static, none, fixed focal length).

\- Keep the description concrete and technical. Avoid subjective praise or vague cinematic adjectives.

\- Ensure internal consistency across fields (e.g., a static shot should not have fast translation).

Label schema (use exactly one option per field): - camera\_angle: {eye-level, high-angle, lowangle, overhead, dutch-angle}

\- shot\_size: {extreme close-up, close-up, medium, medium-long, long, extreme long}

\- camera\_motion\_type: {static, pan, tilt, zoom, dolly, tracking, handheld}

\- focal\_behavior: {fixed focal length, zoom-in, zoom-out}

\- motion\_speed: {slow, medium, fast}

\- rotation\_intensity: {none, subtle, strong}

\- translation\_intensity: {none, subtle, strong}

Output format (JSON only):

<|im\_end|>

<|im\_start|>user   
Visual input: <sampled frames from a clip> Task: Create one JSON annotation following the schema above.   
<|im\_end|>

<table><tr><td>Task</td><td>Method</td><td>Macro-ACC</td><td>Macro-Prec</td><td>Macro-Rec</td><td>Macro-F1</td></tr><tr><td rowspan="5">Camera Angle</td><td>Standard</td><td>66.9 (0.4)</td><td>2.8 (0.2)</td><td>2.9 (0.2)</td><td>2.5 (0.2)</td></tr><tr><td>Vanilla-SFT</td><td>86.0 (0.3)</td><td>11.3 (0.5)</td><td>11.4 (0.5)</td><td>11.3 (0.5)</td></tr><tr><td>DreamFactory</td><td>80.4 (0.4)</td><td>34.3 (0.7)</td><td>29.9 (0.6)</td><td>25.0 (0.6)</td></tr><tr><td>MovieAgent</td><td>84.0 (0.3)</td><td>34.8 (0.7)</td><td>28.5 (0.6)</td><td>24.9 (0.6)</td></tr><tr><td>CamPilot (Ours)</td><td>92.7 (0.2)</td><td>41.3 (0.6)</td><td>22.0 (0.6)</td><td>23.0 (0.6)</td></tr><tr><td rowspan="5">Shot Size</td><td>Standard</td><td>78.3 (0.4)</td><td>9.9 (0.5)</td><td>7.4 (0.4)</td><td>7.7 (0.4)</td></tr><tr><td>Vanilla-SFT</td><td>75.8 (0.4)</td><td>11.7 (0.5)</td><td>11.3 (0.5)</td><td>11.4 (0.5)</td></tr><tr><td>DreamFactory</td><td>66.2 (0.5)</td><td>37.0 (0.7)</td><td>41.0 (0.8)</td><td>33.5 (0.7)</td></tr><tr><td>MovieAgent</td><td>76.1 (0.4)</td><td>31.2 (0.7)</td><td>25.6 (0.7)</td><td>26.6 (0.7)</td></tr><tr><td>CamPilot (Ours)</td><td>83.7 (0.3)</td><td>37.1 (0.7)</td><td>35.7 (0.7)</td><td>34.8 (0.7)</td></tr><tr><td rowspan="5">Camera Motion (Type)</td><td>Standard</td><td>42.0 (0.5)</td><td>1.8 (0.2)</td><td>1.5 (0.2)</td><td>1.4 (0.2)</td></tr><tr><td>Vanilla-SFT</td><td>43.5 (0.5)</td><td>3.4 (0.3)</td><td>3.1 (0.3)</td><td>3.2 (0.3)</td></tr><tr><td>DreamFactory</td><td>38.3 (0.5)</td><td>8.5 (0.5)</td><td>9.3 (0.5)</td><td>7.7 (0.5)</td></tr><tr><td>MovieAgent</td><td>49.9 (0.4)</td><td>7.6 (0.5)</td><td>6.0 (0.5)</td><td>5.9 (0.5)</td></tr><tr><td>CamPilot (Ours)</td><td>59.4 (0.3)</td><td>11.1 (0.5)</td><td>8.2 (0.5)</td><td>8.7 (0.5)</td></tr><tr><td rowspan="5">Camera Motion (Translation)</td><td>Standard</td><td>48.3 (0.5)</td><td>13.9 (0.6)</td><td>12.8 (0.6)</td><td>12.3 (0.6)</td></tr><tr><td>Vanilla-SFT</td><td>45.7 (0.5)</td><td>17.4 (0.6)</td><td>17.0 (0.6)</td><td>17.1 (0.6)</td></tr><tr><td>DreamFactory</td><td>48.5 (0.5)</td><td>22.5 (0.7)</td><td>21.6 (0.7)</td><td>21.3 (0.7)</td></tr><tr><td>MovieAgent</td><td>51.8 (0.4)</td><td>25.3 (0.7)</td><td>24.4 (0.7)</td><td>24.2 (0.7)</td></tr><tr><td>CamPilot (Ours)</td><td>61.2 (0.3)</td><td>31.0 (0.7)</td><td>30.4 (0.7)</td><td>28.3 (0.7)</td></tr><tr><td rowspan="5">Camera Motion (Rotation)</td><td>Standard</td><td>80.0 (0.4)</td><td>11.2 (0.6)</td><td>10.8 (0.6)</td><td>10.8 (0.6)</td></tr><tr><td>Vanilla-SFT</td><td>77.6 (0.4)</td><td>15.6 (0.6)</td><td>15.5 (0.6)</td><td>15.5 (0.6)</td></tr><tr><td>DreamFactory</td><td>65.7 (0.5)</td><td>23.4 (0.7)</td><td>26.3 (0.7)</td><td>23.4 (0.7)</td></tr><tr><td>MovieAgent</td><td>70.4 (0.5)</td><td>17.2 (0.6)</td><td>20.0 (0.6)</td><td>18.5 (0.6)</td></tr><tr><td>CamPilot (Ours)</td><td>86.1 (0.3)</td><td>25.5 (0.7)</td><td>28.9 (0.7)</td><td>26.2 (0.7)</td></tr><tr><td rowspan="5">Camera Motion (Speed)</td><td>Standard</td><td>49.6 (0.5)</td><td>12.3 (0.6)</td><td>15.6 (0.6)</td><td>11.3 (0.6)</td></tr><tr><td>Vanilla-SFT</td><td>53.1 (0.5)</td><td>25.5 (0.7)</td><td>24.6 (0.7)</td><td>24.9 (0.7)</td></tr><tr><td>DreamFactory</td><td>52.7 (0.5)</td><td>33.4 (0.7)</td><td>42.6 (0.8)</td><td>34.2 (0.7)</td></tr><tr><td>MovieAgent</td><td>58.2 (0.4)</td><td>34.5 (0.7)</td><td>39.8 (0.8)</td><td>34.4 (0.7)</td></tr><tr><td>CamPilot (Ours)</td><td>63.7 (0.3)</td><td>36.5 (0.7)</td><td>35.4 (0.7)</td><td>35.2 (0.7)</td></tr><tr><td rowspan="5">Camera Motion (Focal Length)</td><td>Standard</td><td>74.4 (0.5)</td><td>11.4 (0.6)</td><td>12.7 (0.6)</td><td>10.9 (0.6)</td></tr><tr><td>Vanilla-SFT</td><td>86.1 (0.4)</td><td>23.6 (0.7)</td><td>23.7 (0.7)</td><td>23.6 (0.7)</td></tr><tr><td>DreamFactory</td><td>57.1 (0.6)</td><td>23.1 (0.7)</td><td>25.0 (0.7)</td><td>24.0 (0.7)</td></tr><tr><td>MovieAgent</td><td>87.1 (0.4)</td><td>32.1 (0.7)</td><td>31.7 (0.7)</td><td>31.6 (0.7)</td></tr><tr><td>CamPilot (Ours)</td><td>92.2 (0.3)</td><td>28.8 (0.7)</td><td>41.0 (0.8)</td><td>26.2 (0.7)</td></tr></table>

Table 4: Sub-task performance (%) comparison of our method against baselines for multi-shot video generation. The best result is highlighted in bold, and the second-best is underlined. Average represents the average scores across all tasks.