# EvoWorld: Towards Out-of-Sight World Evolution in Generative Video Models

Yequan Bie<sup>1</sup>, Dengyang Jiang<sup>1</sup>, Zhixuan Chen<sup>1</sup>, Xiaozhuan Liang<sup>2</sup>, Haoxuan Che<sup>1</sup>, Quande Liu<sup>3,✉</sup>, Xi Chen<sup>2</sup>, Hao Chen<sup>1,✉</sup> <sup>1</sup>The Hong Kong University of Science and Technology <sup>2</sup>Tencent <sup>3</sup>The Chinese University of Hong Kong https://tommy-bie.github.io/projects/evoworld/

## Abstract

Generative video models are increasingly used as visual world simulators, yet their primary strength lies in rendering plausible pixels rather than reasoning about how a world should evolve. Coherent world evolution requires more than visual fidelity: when an event occurs, a model must infer its consequences for the entities in the world and then render the resulting state. For example, after a wild beast appears, the model should not only depict the wild beast, but also infer whether a nearby person should flee, take cover, or fight back based on the realized scene. Existing diffusion-based video models lack an explicit mechanism for maintaining world state and reasoning over such event-conditioned consequences, often producing visually plausible but behaviorally static or causally inconsistent outputs. To make this reasoning gap concrete and measurable, we propose a new problem, Out-of-Sight Causal Evolution, which requires a model to render an event, reason about its consequences for the evolving world state, and reveal a causally plausible outcome as the world continues to evolve, even out of sight. As a baseline reference solution, we introduce EvoWorld, a brain–renderer collaboration framework in which a vision-language model serves as the world brain: it observes generated segments, maintains an evolving account of the world, and reasons about plausible state transitions, while a segment-autoregressive video renderer realizes them visually. We further introduce EvoWorld-Data and EvoWorld-Bench for training and evaluating event realization, causal reactions, and revisit consistency. Experiments show that EvoWorld substantially improves event-conditioned causal evolution over recent models while preserving visual quality and consistency, establishing a reference point for reasoning-guided, autonomous world evolution beyond the current field of view.

## 1 INTRODUCTION

Building models that can simulate evolving environments has become an important research direction in generative AI. By learning world dynamics, a world model uses observed history and input actions to predict plausible future states (Team et al., 2026b; Gao et al., 2026; Zhang et al., 2026b; Assran et al., 2025). Such predictive representations have enabled progress in decision making and autonomous driving (Feng et al., 2025; Zhou et al., 2025; Liang et al., 2025; Gao et al., 2024b;a), agent training and embodied intelligence (Li et al., 2025; Wang et al., 2025; Hafner et al., 2025), and synthetic environment generation (Bruce et al., 2024; Brooks et al., 2024).

Generative video models (Wan et al., 2025; HaCohen et al., 2026; Yang et al., 2025c; Wu et al., 2025) provide a direct approach to visual world modeling (Team et al., 2026b; He et al., 2025; Team et al., 2026a): their predictions are immediately observable, while text and camera controls offer intuitive interaction. Yet locally plausible frames alone do not yield a coherent long-horizon world. A model must preserve the environments it has traversed and the events that have occurred; otherwise, later states become disconnected from the history that should have produced them. Accordingly, recent video world models have explored various mechanisms for long-horizon memory and consistency, including recurrent state-space memory (Zhu et al., 2026), memory retrieval or dynamic frame eviction (Yu et al., 2025; Yang et al., 2026), and geometry-grounded spatial memory (e.g., 3D point clouds) (Wu et al., 2026; Team et al., 2026a; Ma et al., 2026a).

![](images/48cb894d53e637a70e0814b3d07e2c9b0b2af020dee65ba947326073d7bb4037.jpg)  
Figure 1: Challenges in Out-of-Sight Causal Evolution. The camera first observes a subject (a vendor and his handcart), moves away to reveal a newly requested event (a collapsing building facade), and then moves back to the original scene. Previous methods (rows 1–4) exhibit various failures. In contrast, EvoWorld (row 5) realizes the event, revisits the original subject, and renders a causally plausible reaction while preserving subject and scene consistency.

Most existing approaches nevertheless focus on preserving or retrieving previously observed visual states. Such persistence is necessary but insufficient: coherent world evolution also requires reasoning about how entities should change as new events unfold. This highlights a key limitation of current video models: they primarily rely on diffusion-based generation to render visual observations, without an explicit mechanism for reasoning over world states and event consequences. For example, when a monster appears, a model should not only render the monster faithfully, but also infer whether a nearby character should flee, seek cover, or fight back according to the realized world state. The challenge becomes especially pronounced when the affected subject is outside the current field of view: its resulting state cannot be determined simply by restoring its last observed appearance, but must also incorporate changes inferred from the intervening event. To make this reasoning gap concrete and measurable, we formulate a new problem, Out-of-Sight Causal Evolution, which requires a model to realize an event, reason about how the event changes the relevant world state, including its consequences for an out-of-sight subject, and render the resulting state when the subject is revisited.

Out-of-Sight Causal Evolution couples three challenges that existing models rarely address simultaneously: preserving subject and scene consistency across camera motion; realizing an event introduced by a mid-rollout prompt switch; and reasoning about, then rendering, a causally plausible world-state transition, revealed through the subject’s response when the camera returns. As illustrated in Fig. 1, HY-World 1.5 (HunyuanWorld, 2025), Echo-WM (Zhang et al., 2026b), LingBot World (Team et al., 2026b), and SANA-WM (Zhu et al., 2026) exhibit various failures: the event or reaction may be absent, the subject may follow the camera or become duplicated, appearance and background may drift, or the subject may exhibit implausible behavior. Appearance persistence, instruction following, and event-conditioned behavior must therefore be measured jointly. Therefore, we introduce EvoWorld-Data, a controlled generative dataset of around 12k long videos (≥ 10 sec onds). The training set spans 19 event families and 65 fine-grained event types, 16 scene families and 63 fine-grained scene types, and 14 reaction types across photorealistic and game-world domains. Segment-level prompts and temporal annotations provide direct supervision for event realization, long-range visual continuity, and behavioral change.

As a reference solution, we develop EvoWorld, a reasoning-and-retrieval-augmented framework that separates world-state reasoning from visual rendering. A VLM serves as a world brain that continuously perceives and records the evolving world state, while the renderer generates the next segment based on the reasoning from the world brain. We train the renderer on EvoWorld-Data with Identity-Anchored Attention and a two-stage strategy that first learns reactive rendering from clean data and then reduces exposure bias through online autoregressive rollout training. We further introduce EvoWorld-Bench, which evaluates whether the camera revisits the original region, the switched prompt realizes the requested event, the subject responds plausibly, and the subject and background remain consistent. These coupled capabilities are not captured by existing video benchmarks such as VBench (Huang et al., 2024), which primarily emphasize perceptual quality and short-range temporal behavior.

Our contributions are threefold: (i) We formulate Out-of-Sight Causal Evolution, an eventconditioned world-evolution problem that requires revisit consistency, mid-rollout event realization, and causal reactive evolution. (ii) We introduce EvoWorld, a world-brain–renderer collaboration framework that separates world-state reasoning from visual rendering to continuously perceive, reason about, and render out-of-sight world dynamics. (iii) We propose EvoWorld-Data, which contains around 12k videos spanning diverse scenes, events, and reactions, and introduce EvoWorld-Bench to measure the Out-of-Sight Causal Evolution capability. We hope this work holds the potential to advance the research of out-of-sight dynamics and causal evolution in video world modeling.

## 2 RELATED WORK

Video world models & autoregressive video generation. Generative video backbones (Wan et al., 2025; HaCohen et al., 2026; MiniMax, 2026) provide the visual synthesis foundation for interactive world simulation, including recent models and systems such as XGEN (Cai et al., 2026), Zing-0.5 (Chen et al., 2026b), and WorldPlay2 (Zhang et al., 2026a). LingBot-World (Team et al., 2026b) and LingBot-World-v2 (Gao et al., 2026) support long-horizon, real-time interaction, with the latter combining bidirectional and autoregressive attention. Echo-WM (Zhang et al., 2026b) targets open and enterable omnimodal world simulation, while Alaya-EVOKE (Yin et al., 2026) combines persistent world-state memory, mid-rollout reprompting, and few-step generation. Segment-autoregressive methods preserve full attention within each generated segment to reduce the quality degradation associated with fully causal adaptation. YUME-1.5 (Mao et al., 2026) combines long-context generation with bidirectional-attention distillation. Matrix-Game 3.0 (Wang et al., 2026) uses memoryaugmented multi-segment autoregressive distillation. BiWM (Rui et al., 2026) provides a general recipe for adapting bidirectional video backbones into controllable segment-autoregressive world models. We build on this formulation and focus on whether the generated world continues to evolve beyond the current field of view.

Out-of-sight memory and dynamics. Long-horizon consistency does not ensure that an unobserved world continues to evolve. Recent evaluations expose this limitation through camera lookaway, occlusion, or illumination changes (Ma et al., 2026b; Lu et al., 2026). ReMind (Xu et al., 2026) trains a causal video generator to recover evolving states across interruptions. LiveWorld (Duan et al., 2026) maintains a persistent global state by separating a static 3D environment from dy namic entities advanced by an evolution engine. Hybrid Memory (Chen et al., 2026a) distinguishes static-background preservation from dynamic-subject tracking across exit–reentry intervals. These works primarily examine whether an active process or motion continues through an observation gap. Our setting instead focuses on introducing an event after the subject leaves view and inferring its consequences when the subject is revisited, combining mid-rollout instruction following (Yang et al., 2025b), long-range entity consistency, and reasoning over the realized visual history.

## 3 EVOWORLD: DATA AND METHOD

## 3.1 PROBLEM & TASK SETTING

A world evolves continuously rather than only where and when it is being observed. Its entities should therefore remain active participants in that evolution: as new events occur and environmental conditions change, they should autonomously interact with those changes and exhibit appropriate responses, including while they are outside the current field of view. Leaving the frame should not freeze an entity or remove it from the unfolding dynamics—it is out of sight, but not out of the world. We therefore formulate Out-of-Sight Causal Evolution as a task for measuring whether a video world model or generation system can maintain a coherent account of off-screen entities and the relevant world state, infer how intervening events should alter that state and affect those entities, and render causally plausible outcomes, e.g., the entities’ responses, when the scene is revisited. Here, causal evolution means that the revealed post-event state must follow plausibly from the even actually realized, rather than merely remain visually consistent with the preceding frames.

![](images/9154fbf563630f502c07922cb99dedd11a0517b90bb254af3b4a0e925bdee6b9.jpg)  
Figure 2: Overview of the data engine for EvoWorld. Structured scenarios are rendered into singletake out-of-sight interactions, verified against visual evidence, temporally annotated, and converted into a training supervision dataset and evaluation benchmark.

Specifically, as shown in Fig. 1, an Out-of-Sight Causal Evolution rollout can be abstracted into at least three semantic parts (each corresponding to at least one prompt): (1) $\mathcal { P } _ { \mathrm { i n i t } }$ (initial): the part that covers the initial state of the scenario and the main subjects. Condition $p _ { i n i t }$ specifies the subjects and scene, and the camera remains static or begins to move away from the subject. (2) $\mathcal { P } _ { \mathrm { e v e n t } }$ (event): the part that introduces an event after the initial state, where the subject has left the frame because the camera view has moved away. Condition $p _ { e v e n t }$ introduces an event $( \mathrm { e . g . }$ , “something is on $\mathrm { f i r e } ^ { \prime \prime } )$ , and the camera may remain static, continue moving away to show the event, or begin moving back toward the original region. (3) $\mathcal { P } _ { \mathrm { r e v i s i t } }$ (revisit): the part in which the camera returns to the initial region and reveals the resulting post-event world state, including the reappearance of the subject and its causally plausible response to the event. The ground-truth condition $p _ { r e v i s i t }$ describes this resulting state and response, whereas the benchmark provides a blind revisit instruction without prescribing the concrete outcome. These parts describe semantic roles rather than a fixed number of generated segments. Each part may span one or more autoregressive chunks/segments. We use this three-part abstraction to isolate the minimal causal chain while allowing longer rollouts and additional intermediate observations in real-world scenarios. A system may internally infer a detailed control condition from the generated history, as our brain does instead of pre-specifying the eventual reaction in the initial condition. The evaluation targets the complete event-to-reaction causal chain: (1) the switch from $p _ { i n i t }$ to $p _ { e v e n t }$ actually instantiates the requested event; (2) the camera leaves and later revisits the original subject and its corresponding region (not necessarily the exact initial frame or viewpoint, since the subject may have moved during the out-of-sight interval); (3) at revisit, the subject preserves its identity and core visual appearance without severe drift, and the surrounding scene and background remain reasonably consistent; (4) the revealed post-event world state is causally plausible given the realized event, including an appropriate change in the subject’s state or behavior, while maintaining natural motion and visual quality.

## 3.2 EVOWORLD-DATA & BENCH CONSTRUCTION

Fig. 2 shows the overall data construction pipeline of EvoWorld. We here introduce the main components of the pipeline. Additional construction details are provided in Appendix Section A.

![](images/85f5fb06a81f28df9015105ba1f765f486bd3d9130608774ee21360ddc4f0996.jpg)  
Figure 3: Examples and statistics of EvoWorld-Data and EvoWorld-Bench. The upper part illustrates representative training and benchmark examples, while the lower part summarizes their scale and scenario diversity.

EvoWorld-Data. (1) Scenario design and video data generation. Existing real-world video datasets rarely contain samples that align with our experimental setting. We therefore construct a controlled generative dataset. The curated data aims to make event-conditioned world-state evolution observable, with the event-to-reaction causal chain separately auditable at each stage. Scenarios are represented by structured tuples containing subjects and their appearance, scenes, events, compatible reactions, and camera movements. We compose tuples under semantic compatibility constraints—for example, the selected events must admit a visually observable response from the chosen subjects—and render them into a single-take generation script with five beats: initial scenario with the main subject → move the camera view away → show event → move the camera view back → revisit and reveal the resulting state and response. The full scripts are written by a VLM (Gemini3.1-Pro (Team & DeepMind, 2026)). Videos are then generated by the video foundation models (Seedance et al., 2026; Team, 2026) based on the designed script. (2) Verification and annotation. Generated videos do not necessarily obey their scripts, so we verify visual evidence after generation. Specifically, we use Gemini3.1-Pro (Team & DeepMind, 2026) to check that each clip is a continuous shot, the designated subject remains visually consistent, the subject leaves and reappears, the event occurs in the intended view, the causal reaction is visible, and no severe hallucination invalidates the example. We further estimate global camera motion from Lucas–Kanade feature tracks using a RANSAC-based affine fit (Fischler & Bolles, 1981). To ensure the camera motion correctness, we reject full-circle camera motions, missing returns, single-direction motion, and unreliable direction estimates. For segment-wise supervision, we partition each video into multiple intervals and describe them jointly so that adjacent captions form a coherent progression. We audit these captions for future-event spoilers, references to an absent subject, malformed camera instructions, and excessive adjacent-caption overlap.

EvoWorld-Bench. (1) Benchmark Construction. We construct EvoWorld-Bench as a 120-case evaluation set spanning photorealistic (69 cases) and game/virtual (51 cases) domains. Its initial frames have two complementary sources: 66 are extracted from held-out videos disjoint from the training set, while 54 are generated with GPT-Image-2 (OpenAI, 2026). Each case is organized into three semantic parts: an initial state, an event, and a revisit. After semantic deduplication, the benchmark covers 50+ fine-grained scene types and sub-event types, as summarized in Fig. 3. Moreover, we conduct dedicated similarity screening against all training videos. For every benchmark case, we retrieve potentially similar training samples using both frame visual similarity and the lexical and embedding-based semantic similarity of the initial-event prompt pair. We then inspect retrieved candidate pairs with a VLM-based two-pass audit and human adjudication of the ambiguous cases. We remove any candidate judged highly similar, leaving no source-clip overlap, initial-event prompt-pair overlap, or human-confirmed high-similarity pair in the final benchmark. (2) Evaluation Metrics. We evaluate the complete rollout using Event Success, Revisit Success, Causal Plausibility, Reaction Quality, Visual Integrity, and revisit subject and background consistency. Detailed inputs, scoring criteria, and aggregation rules are provided in Appendix Section A.

![](images/8924f1cb6b9f92b206cd038a269a3fcb5b600f56625c5b0b74830e2673da6b06.jpg)  
Figure 4: Overview of the EvoWorld framework. (a) Stage 1 learns out-of-sight reactive rendering. (b) Stage 2 refines the renderer through online autoregressive adaptation. (c) Inference uses brain– renderer collaboration for Out-of-Sight Causal Evolution.

## 3.3 METHOD

## 3.3.1 OVERVIEW

Most existing video world models provide little explicit support for interpreting what occurred dur ing generation, maintaining a persistent account of entities outside the current view, or inferring how those entities should evolve before they reappear. We therefore introduce EvoWorld, a framework for Out-of-Sight Causal Evolution that explicitly separates world reasoning from visual rendering (Fig. 4). A VLM brain observes every generated segment and records the realized entities, scene, and events. After an event occurs, it reasons over this history and predicts how the relevant world state, including the off-screen subject, should have evolved. A segment-autoregressive visual renderer then generates the next segment from the inferred instruction, retrieved entity evidence, and recent generated latents. We equip the renderer with complementary entity and temporal anchors for long-range identity and short-range continuity. Identity-Anchored Attention (IAA) strengthens in formation flow from the entity anchor when the subject reappears. Our two-stage training first learns subject preservation and reactive rendering from clean ground-truth windows, and then performs online autoregressive rollout training from self-generated predecessor context. This Self-Forcing-style adaptation (Huang et al., 2026) narrows the train-inference gap and refines model performance.

## 3.3.2 ENTITY AND TEMPORAL ANCHORS

The entity anchor carries subject appearance across a long out-of-view interval. During training, it is selected from ground-truth frames in which the target subject is clearly visible. At inference time, the VLM inspects sampled frames from the generated history, identifies a clear and unoccluded observation of the target subject, and retrieves the corresponding latents. This helps the renderer maintain subject identity and appearance when the camera revisits. For temporal anchor, we adopt the four latents immediately preceding the target window, which participate in within-window full attention. This anchor is selected deterministically since recency is precisely the desired signal for view and motion continuity. Stage 1 takes these latents from the ground-truth videos, whereas stage 2 replaces them with the tail of the autoregressive rollout. At inference time they are always taken from the latest generated segment. The two anchors are therefore complementary: entity retrieval spans the long-term identity gap, while the temporal anchor bridges the local autoregressive boundary.

## 3.3.3 IDENTITY-ANCHORED ATTENTION

Although an appended entity reference participates in self-attention, it may not receive sufficient attention when the subject reappears: its contribution can be diluted by the much larger target and temporal context. We therefore introduce Identity-Anchored Attention (IAA), which makes the retrieved entity tokens a soft attention sink only for revisit samples. Let R denote target-query tokens in a revisit window and $\mathcal { E }$ the key tokens belonging to the entity anchor. For an attention head, IAA changes the pre-softmax score from $A$ to

Table 1: Comparison results (%) on the EvoWorld-Bench. Best results are shown in bold and secondbest results are underlined. Core denotes the proportion of cases concurrently achieving successful revisit, event realization, and full causal plausibility.
<table><tr><td></td><td colspan="5">Task metrics</td><td colspan="4">Visual quality, consistency, and motion</td></tr><tr><td>Method</td><td>Core</td><td>Revisit</td><td>Event</td><td>Causal</td><td>React.</td><td>Visual</td><td>Subj. Cons.</td><td> $\mathrm { B g . }$  Cons.</td><td>Motion</td></tr><tr><td>LingBot-World (Team et al., 2026b)</td><td>15.8</td><td>92.5</td><td>77.5</td><td>29.2</td><td>31.3</td><td>68.8</td><td>87.6</td><td>94.6</td><td>98.0</td></tr><tr><td>LingBot-World-v2 (Gao et al., 2026)</td><td>20.0</td><td>76.7</td><td>68.3</td><td>27.1</td><td>25.4</td><td>76.7</td><td>82.6</td><td>92.5</td><td>98.4</td></tr><tr><td>YUME-1.5 (Mao et al., 2026)</td><td>15.8</td><td>62.5</td><td>86.7</td><td>22.5</td><td>20.0</td><td>65.4</td><td>72.9</td><td>88.7</td><td></td></tr><tr><td>SANA-WM (Zhu et al., 2026)</td><td>13.3</td><td>88.3</td><td>67.5</td><td>26.7</td><td>24.2</td><td>53.8</td><td>67.7</td><td>90.8</td><td>98.7</td></tr><tr><td>Echo-WM (Zhang et al., 2026b)</td><td>16.7</td><td>60.8</td><td>66.7</td><td>22.5</td><td>21.7</td><td>82.9</td><td>97.6</td><td>89.2</td><td>98.7</td></tr><tr><td>Evoke (Yin et al., 2026)</td><td>20.0</td><td>95.8</td><td>80.0</td><td>32.9</td><td>36.3</td><td>78.3</td><td>93.5</td><td>95.4</td><td>99.0</td></tr><tr><td>Evoke-Turbo (Yin et al., 2026)</td><td>21.7</td><td>98.3</td><td>89.2</td><td>32.5</td><td>29.6</td><td>65.4</td><td>97.0</td><td>96.2</td><td>99.1</td></tr><tr><td>EvoWorld (ours)</td><td>90.8</td><td>100.0</td><td>97.5</td><td>94.2</td><td>85.0</td><td>83.8</td><td>99.2</td><td>98.3</td><td>99.5</td></tr></table>

$$
\widetilde { A } _ { q k } = \frac { Q _ { q } K _ { k } ^ { \top } } { \sqrt { d } } + M _ { q k } + \beta \Im \{ q \in \mathcal { R } \} \mathbb { 1 } \{ k \in \mathcal { E } \} ,\tag{1}
$$

where $M _ { q k }$ is the base additive self-attention mask, $\beta > 0$ controls identity preference, and $\mathbb { 1 } \{ \cdot \}$ denotes the indicator function. In full-attention video sequence, $M _ { q k } = 0$ for valid token pairs and takes a large negative value only for masked or padded pairs. The IAA bias is directional: revisit tokens read from the entity anchor, while unrelated query-key pairs are unchanged. In implementation, we multiply the mask entries corresponding to the target-query–entity-key block by $\cdot e ^ { \beta }$ before LTX converts the mask to log space, which is exactly equivalent to adding $+ \beta$ in Eq. (1). IAA is enabled only for samples marked as revisit during training and for the revisit chunk during inference.

## 3.3.4 TWO-STAGE AUTOREGRESSIVE TRAINING

Stage 1: Learning causally grounded reactive rendering with clean anchors. For each sliding training window, the loader supplies both the entity anchor and the temporal anchor as conditioning. The training sequence is organized as [entity anchor | temporal anchor | target]. Both anchors remain clean at timestep zero and carry zero loss weight; only the target latents are noised and supervised by the flow-matching objective. IAA is activated on revisit windows. This stage teaches the renderer core capabilities such as event execution, event-conditioned reactions, and the roles of long-range identity and short-range temporal conditioning.

Formally, let $\mathbf { e } , \mathbf { c } ,$ and z denote the clean entity-anchor, temporal-anchor, and target latents, respectively; let $p$ be the target prompt and $r \in \{ 0 , 1 \}$ the revisit indicator. For $\epsilon \sim \mathcal { N } ( 0 , I )$ and a sampled noise level $\sigma ,$ , we construct

$$
{ \bf x } ^ { \sigma } = \left[ { \bf e } \mid { \bf c } \mid ( 1 - \sigma ) { \bf z } + \sigma \epsilon \right] , \qquad { \bf s } ^ { \sigma } = \left[ { \bf 0 } _ { \mid { \bf e } \mid } \mid { \bf 0 } _ { \mid { \bf c } \mid } \mid \sigma { \bf 1 } _ { \mid { \bf z } \mid } \right] ,\tag{2}
$$

where $\mathbf { s } ^ { \sigma }$ is the per-token timestep input, and ${ \bf 0 } _ { n }$ and ${ \bf 1 } _ { n }$ denote length-n all-zero and all-one vectors, respectively. Thus the two anchors remain clean at timestep zero, whereas every target token receives timestep σ and follows the flow-matching interpolation. The renderer v<sub>θ</sub> predicts the target velocity ϵ − z, and stage 1 training minimizes

$$
\mathcal { L } _ { \mathrm { S 1 } } = \mathbb { E } _ { ( \mathbf { e } , \mathbf { c } , \mathbf { z } , p , r ) , \sigma , \epsilon } \left[ \frac { 1 } { | \Omega | D } \sum _ { i \in \Omega } \| v _ { \boldsymbol { \theta } } ( \mathbf { x } ^ { \sigma } , \mathbf { s } ^ { \sigma } , p , r ) _ { i } - ( \epsilon _ { i } - \mathbf { z } _ { i } ) \| _ { 2 } ^ { 2 } \right] ,\tag{3}
$$

where Ω contains target tokens and $D$ is the latent channel dimension. The indicator r controls whether the IAA bias in Eq. (1) is enabled.

Stage 2: Online autoregressive adaptation. Stage 1 is trained with a clean ground-truth temporal anchor, but deployment conditions each new segment on latents produced by the model itself. Small errors in appearance, motion, and geometry can therefore accumulate at precisely the event-to-revisit transition. We reduce this exposure gap with a lightweight second stage initialized from stage 1. For every selected event-to-revisit pair, the currently updated renderer performs an online one-hop rollout of the immediately preceding segment. This predecessor rollout is also conditioned on a clean entity anchor, its ground-truth temporal anchor, and the predecessor’s conditioning prompt.

Table 2: Component ablation of EvoWorld. Cumulative effects of IAA and Stage-2 online rollout training on EvoWorld-Bench. All values are percentages, and column-wise best results are bold.
<table><tr><td>Setting</td><td>Core</td><td>Revisit</td><td>Event</td><td>Causal</td><td>React.</td></tr><tr><td>Stage 1 w/o IAA</td><td>82.5</td><td>100.0</td><td>96.7</td><td>88.8</td><td>80.8</td></tr><tr><td>Stage 1 + IAA</td><td>84.2</td><td>99.2</td><td>98.3</td><td>88.7</td><td>83.9</td></tr><tr><td>Stage 1 + Stage 2</td><td>90.8</td><td>100.0</td><td>97.5</td><td>94.2</td><td>85.0</td></tr></table>

![](images/fb14297f32d641a0dd0ed7ebf5d3fab6d5968969d25aa89e6c6dd88bfdde0bea.jpg)  
Figure 5: Effect of event-conditioned reasoning prompts on Out-of-Sight Causal Evolution.

We take the final four latents of this self-generated predecessor as the temporal anchor $\widetilde { \mathbf { c } } _ { t }$ for the revisit target, replacing its ground-truth temporal anchor. The resulting Stage-2 input is

$$
\widetilde { \mathbf { x } } _ { t } ^ { \sigma } = \left[ \mathbf { e } _ { t } \mid \widetilde { \mathbf { c } } _ { t } \mid ( 1 - \sigma ) \mathbf { z } _ { t } + \sigma \epsilon \right] ,\tag{4}
$$

with the per-token timestep input $\mathbf { s } _ { t } ^ { \sigma }$ defined as in Eq. (2). We optimize the same target-only velocitymatching objective as in Stage 1, now conditioned on the self-generated temporal anchor:

$$
\mathcal { L } _ { \mathrm { S 2 } } = \mathbb { E } _ { t , \sigma , \epsilon } \left[ \frac { 1 } { | \Omega _ { t } | D } \sum _ { i \in \Omega _ { t } } \big \lVert v _ { \theta } \big ( \widetilde { \mathbf { x } } _ { t } ^ { \sigma } , \mathbf { s } _ { t } ^ { \sigma } , p _ { t } , r _ { t } \big ) _ { i } - \big ( \epsilon _ { i } - \mathbf { z } _ { t , i } \big ) \big \rVert _ { 2 } ^ { 2 } \right] ,\tag{5}
$$

where $\Omega _ { t }$ contains only target tokens, $p _ { t }$ is the target prompt, and $r _ { t }$ is the revisit indicator controlling IAA. The entity and temporal anchors remain loss-free conditioning. Additional training details for the two-stage construction are provided in Appendix Section C.

## 3.3.5 VLM BRAIN: PERCEPTION AND REASONING

At inference time, our framework adopts a VLM as brain to perceive and reason over the world evolution continuously. After every completed segment, the renderer decodes sampled frames for the brain, which identifies and records the visible entities, appearances, object states, scenes, and any realized changes. These observations accumulate into a persistent world memory. Once an injected event has been rendered and observed, the brain records what actually occurred, retrieves the identity evidence, and reasons over the accumulated context to infer a causally plausible evolution of the relevant world state, including how the off-screen subject should respond. It then expresses the inferred state as a concrete generation prompt $\hat { p }$ for the relevant subsequent segment. Additional details are provided in Appendix Section D.

## 4 EXPERIMENTS

## 4.1 EXPERIMENT SETUP

We compare EvoWorld with recent interactive video world models that support autoregressive generation: LingBot-World (Team et al., 2026b), LingBot-World-v2 (Gao et al., 2026), YUME-1.5 (Mao et al., 2026), SANA-WM (Zhu et al., 2026), Echo-WM (Zhang et al., 2026b), and the standard and Turbo variants of Evoke (Yin et al., 2026). All baselines receive the same initial, event, and blind revisit prompts without a prescribed reaction, switched at matched semantic boundaries through their native autoregressive interfaces. Implementation details and additional general video models with one-shot results under a different, composed-prompt setting are provided in Appendix Section F. We evaluate Out-of-Sight Causal Evolution on EvoWorld-Bench with the following metrics: Event Success, Revisit Success, Causal Plausibility, Reaction Quality, and Visual Integrity. We additionally assess subject and background consistency between the opening and revisited views while allowing causally appropriate changes in pose, state, and viewpoint, and report Motion Smoothness using VBench (Huang et al., 2024). Our primary task-level measure, Core Task Success, requires Event Success, Revisit Success, and full Causal Plausibility (i.e., causal plausibility score=1) to hold simultaneously. Appendix Section A provides the complete evaluation protocol.

Table 3: Transfer of the VLMbrain reasoning prompt (%).
<table><tr><td>Method</td><td>Core</td><td>Causal</td><td>React.</td></tr><tr><td>Evoke-Turbo</td><td>21.7</td><td>32.5</td><td>29.6</td></tr><tr><td>+ Brain prompt</td><td>45.0</td><td>53.3</td><td>50.8</td></tr><tr><td>EvoWorld (ours)</td><td>90.8</td><td>94.2</td><td>85.0</td></tr></table>

Table 4: Event–prompt alignment based on CLIP score.
<table><tr><td>Method</td><td>EPA-Abs</td><td>EPA-TopK</td></tr><tr><td>LingBot-World</td><td>0.2716</td><td>0.2827</td></tr><tr><td>LingBot-World-v2</td><td>0.2740</td><td>0.2860</td></tr><tr><td>SANA-WM</td><td>0.2789</td><td>0.2872</td></tr><tr><td>Echo-WM</td><td>0.2769</td><td>0.2896</td></tr><tr><td>Evoke</td><td>0.2893</td><td>0.3043</td></tr><tr><td>Evoke-Turbo</td><td>0.2885</td><td>0.3025</td></tr><tr><td>EvoWorld (ours)</td><td>0.2942</td><td>0.3126</td></tr></table>

![](images/ca588d507601fe376b4bf08c2e0af9a2ba5ca7ce3139928ec22e414006078a55.jpg)  
Figure 6: Qualitative comparison under the same VLM-brain reasoning prompt.

## 4.2 IMPLEMENTATION DETAILS

The VLM brain is instantiated with Qwen3.8-27B (Yang et al., 2025a), served through vLLM (Kwon et al., 2023), while the visual renderer is initialized from LTX-2.5 (HaCohen et al., 2026). In Stage 1, we train a rank-128 LoRA for 8,000 steps with a learning rate of $1 \times 1 0 ^ { - 4 }$ , using a per-GPU batch size of 1 on eight GPUs and eight gradient-accumulation steps, resulting in an effective batch size of 64. We set the IAA bias to $\bar { \beta } = \bar { 3 . 0 }$ . In Stage 2, we initialize from the Stage-1 checkpoint and conduct online autoregressive rollout training for 200 steps at $2 \times 1 0 ^ { - 5 }$ . Inference uses the official distilled LoRA and eight-step sampler to generate videos at $8 9 6 \times 5 1 2$ resolution and 24 fps. Complete training and inference settings are provided in Appendix Sections B and C.

## 4.3 RESULTS OF OUT-OF-SIGHT CAUSAL EVOLUTION

Out-of-sight causal evolution. As shown in Table 1, EvoWorld achieves 90.8% Core Task Success, exceeding the second-best baseline by 69.1 percentage points. Existing world models can often execute individual parts of the rollout: for example, Evoke-Turbo reaches 98.3% Revisit Success and 89.2% Event Success. However, its Core Task Success and Causal Plausibility results reveal that successful camera control and event rendering do not by themselves produce a coherent event– reaction chain. In contrast, EvoWorld obtains promising Revisit Success, Event Success, and Causal Plausibility. Its Causal Plausibility and Reaction Quality outperform other methods by a significant margin, demonstrating a substantially stronger ability to render causally plausible interaction evolution in the out-of-sight setting. Consistency and generation quality. The improvement in task success does not come at the cost of appearance preservation or visual quality. EvoWorld achieves promising Subject Consistency, Background Consistency, Visual Integrity, and Motion Smoothness. Its consistently high VLM-based scores and VBench (Huang et al., 2024) motion evaluation indicate strong Out-of-Sight Causal Evolution performance while preserving generation quality.

## 4.4 ABLATION STUDY AND ADDITIONAL ANALYSIS

In this section, we adopt a research-question-driven analysis to analyze and validate the effectiveness of our method, including ablation studies and other analyses.

Do IAA and Stage-2 online rollout training improve out-of-sight causal evolution? As shown in Table 2, IAA increases Core Task Success from 82.5% to 84.2% and Reaction Quality from 80.8% to 83.9%, demonstrating the benefit of strengthening attention to retrieved entity evidence. Stage 2 further raises Core Task Success to 90.8% and Causal Plausibility to 94.2%. These results indicate that both IAA and stage-2 online rollout training improve the model’s capability for out-of-sight causal evolution.

How do event-conditioned reasoning prompts affect different visual renderers? We replace the brain-generated reasoning prompt with the blind revisit prompt while retaining the trained EvoWorld renderer and visual conditioning. As shown in Fig. 5, performance drops substantially without event-conditioned reasoning, confirming its importance for inferring an appropriate response from the realized event. This variant nevertheless remains stronger than the baselines, indicating that the proposed data and renderer training already provide a strong event-conditioned causal-response prior. Moreover, applying the same reasoning prompts to Evoke-Turbo consistently improves its task metrics (Table 3), demonstrating that the benefit transfers across visual renderers.

How do models compare in CLIP-based event–prompt alignment? As an additional reference to the VLM-based Event Success metric, we compare the generated event segment with its event prompt using CLIP ViT-B/32 (Radford et al., 2021). We recover each model’s actual event-prompt interval, uniformly sample eight frames, and report their mean similarity (EPA-Abs) and the mean of the two highest similarities (EPA-TopK). As shown in Table 4, EvoWorld obtains the highest scores. These results should be interpreted only as a relative comparison: event prompts contain substantial shared scene and camera descriptions, while CLIP similarity cannot precisely determine whether a fine-grained event action has been correctly executed.

Does a shared reasoning prompt alone ensure faithful causal evolution? We provide LingBot-World, Evoke, and EvoWorld with the same VLM-inferred revisit prompt to isolate renderer capability. As shown in Fig. 6, the baselines exhibit hallucinated event entities, subject duplication, or coarse revisit frames, whereas EvoWorld preserves the subject and scene while rendering a coherent defensive response. Thus, an appropriate reasoning prompt must be paired with a renderer capable of preserving history and realizing the inferred evolution.

## 5 CONCLUSION

In this paper, we formulate Out-of-Sight Causal Evolution as a concrete setting for investigating autonomous world evolution and construct EvoWorld-Data/Bench to support learning and evaluating event realization, causally plausible post-event states, revisit consistency, and visual quality. We further present the EvoWorld framework, a reasoning-augmented brain-renderer collaboration framework that continuously perceives and renders the world dynamics. The renderer is trained with identity-anchored attention and a two-stage procedure that combines supervised learning of reactive rendering with autoregressive rollout training. Experiments show that EvoWorld substantially improves Out-of-Sight Causal Evolution capabilities over recent video world models while preserving generation quality and consistency. We hope this study can provide a reference for future research in reasoning-enhanced world evolution. Limitations and future directions are discussed in Appendix Section G.

## REFERENCES

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first international conference on machine learning, 2024.

Guohao Cai, Haoxuan Che, Frank Fan, Chenran Huang, Junjia Huang, Jiayi Jiang, Yanheng Li, Leon Liu, Shaoteng Liu, Xi Liu, et al. Generative world simulation with dao and jing. arXiv preprint arXiv:2610.19726, 2026.

Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, and Xiang Bai. Out of sight but not out of mind: Hybrid memory for dynamic video world models. arXiv preprint arXiv:2603.25716, 2026a.

Mingyang Chen, Shengdong Chen, Xiaoxiao Fu, Bosheng Gong, Haoyuan Guo, Bowen Li, Jiawen Li, Kejun Li, Tianpeng Li, Yin Liu, et al. Zing-0.5: Toward playable worlds with real-time joint action and text control. arXiv preprint arXiv:2609.17909, 2026b.

Zicheng Duan, Jiatong Xia, Zeyu Zhang, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen, Xinyu Zhang, and Lingqiao Liu. Liveworld: Simulating out-of-sight dynamics in generative video world models. arXiv preprint arXiv:2603.07145, 2026.

Tuo Feng, Wenguan Wang, and Yi Yang. A survey of world models for autonomous driving. arXiv preprint arXiv:2501.11260, 2025.

Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications ofthe ACM, 24 (6):381–395, 1981.

Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. In International Conference on Learning Representations, volume 2024, pp. 22841–22860, 2024a.

Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems, 37:91560–91596, 2024b.

Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, et al. Infinite worlds with versatile interactions. arXiv preprint arXiv:2607.07534, 2026.

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training agents inside of scalable world models. arXiv preprint arXiv:2509.24527, 2025.

Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21807–21818. IEEE, 2024.

Team HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pp. 611–626, 2023.

Xinqing Li, Xin He, Le Zhang, Min Wu, Xiaoli Li, and Yun Liu. A comprehensive survey on world models for embodied ai. arXiv preprint arXiv:2510.16732, 2025.

Dingkang Liang, Dingyuan Zhang, Xin Zhou, Sifan Tu, Tianrui Feng, Xiaofan Li, Yumeng Zhang, Mingyang Du, Xiao Tan, and Xiang Bai. Unifuture: A 4d driving world model for future generation and perception. arXiv preprint arXiv:2503.13587, 2025.

Jinpeng Lu, Dexu Zhu, Haoyuan Shi, Linghan Cai, Guo Tang, Yinda Chen, Jie Cao, Duyu Tang, Yi Zhang, Yong Dai, et al. Current world models lack a persistent state core. arXiv preprint arXiv:2606.20545, 2026.

Bruce D Lucas and Takeo Kanade. An iterative image registration technique with an application to stereo vision. In IJCAI’81: 7th international joint conference on Artificial intelligence, volume 2, pp. 674–679, 1981.

Wenchao Ma, Changran Liu, Sharon X Huang, and Haomiao Jiang. Closing the loop: Training-free revisit consistency for autoregressive generative rendering. arXiv preprint arXiv:2607.21848, 2026a.

Ziqi Ma, Mengzhan Liufu, and Georgia Gkioxari. Out of sight, out of mind? evaluating state evolution in video world models. arXiv preprint arXiv:2603.13215, 2026b.

Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, and Kaipeng Zhang. Yume1. 5: A text-controlled interactive world generation model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7752–7761, 2026.

MiniMax. MiniMax H3. https://huggingface.co/MiniMaxAI/MiniMax-H3, 2026. Official model card, accessed October 13, 2026.

OpenAI. Introducing chatgpt images 2.0. https://openai.com/index/ introducing-chatgpt-images-2-0/, 2026.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Shaohao Rui, Xiaofeng Mao, Zhanyu Zhang, Peijia Lin, Yansong Zhu, Yibo Zhang, Haibin Wan, Zhangrui Zhao, and Weijie Ma. Biwm: Advancing open-source interactive video world models with bidirectional autoregression. arXiv preprint arXiv:2606.10135, 2026.

Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.

AlayaWorld Team, Kaipeng Zhang, Chuanhao Li, Yifan Zhan, Yongtao Ge, Yuanyang Yin, Jiaming Tan, Kang He, Liaoyuan Fan, Ruicong Liu, et al. Alayaworld: Long-horizon and playable video world generation. arXiv preprint arXiv:2607.06291, 2026a.

ByteDance Seed Team. One-take creation, flexible referencing: Introducing seedance 2.5. https://research.doubao.com/en/blog/ one-take-creation-flexible-referencing-introducing-seedance-2-5, July 2026.

Gemini Team and Google DeepMind. Gemini 3.1 pro model card. https://deepmind. google/models/model-cards/gemini-3-1-pro/, February 2026.

Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026b.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Xinjie Wang, Liu Liu, Yu Cao, Ruiqi Wu, Wenkang Qin, Dehui Wang, Wei Sui, and Zhizhong Su. Embodiedgen: Towards a generative 3d world engine for embodied intelligence. arXiv preprint arXiv:2506.10600, 2025.

Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, et al. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. Advances in Neural Information Processing Systems, 38:49371–49393, 2026.

Tianshuo Xu, Yichen Xie, Depu Meng, Chensheng Peng, Quentin Herau, Bo Jiang, Yihan Hu, and Wei Zhan. Teaching video generators to remember: Eliciting dynamic memory for out-of-sight state evolution. arXiv preprint arXiv:2605.25333, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025b.

Ying Yang, Zhengyao Lv, Yujia Zeng, Tianlin Pan, Haofan Wang, Yueming Lyu, Binxin Yang, Hubery Yin, Chen Li, Jing Lyu, et al. Stableworld: Towards stable and consistent long interactive video generation. arXiv preprint arXiv:2601.15281, 2026.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pp. 83048–83077, 2025c.

Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, and Feng Zhao. Alayaevoke: From linear-scaling supervision to endless world. arXiv preprint arXiv:2608.13546, 2026.

Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pp. 1–11, 2025.

Haiyu Zhang, Wenqiang Sun, Tengfei Wang, Junta Wu, Jun Zhang, Yunhong Wang, Yu Qiao, and Chunchao Guo. Worldplay2: Extending real-time interactive world models in control and horizon. arXiv preprint arXiv:2609.35560, 2026a.

Songchun Zhang, Yaowei Li, Junhao Zhuang, Weiyang Jin, Haoyu Wang, Xin Lu, Yilang Sun, Shiyi Zhang, Haoran Li, Xiaoxiao Ma, et al. Echowm: Open and enterable omnimodal world models. arXiv preprint arXiv:2608.23189, 2026b.

Xin Zhou, Dingkang Liang, Sifan Tu, Xiwu Chen, Yikang Ding, Dingyuan Zhang, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. Hermes: A unified self-driving world model for simultaneous 3d scene understanding and generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27817–27827. IEEE, 2025.

Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. Sana-wm: Efficient minute-scale world modeling with hybrid linear diffusion transformer. arXiv preprint arXiv:2605.15178, 2026.

## APPENDIX

## A EVOWORLD-DATA AND EVOWORLD-BENCH DETAILS

## A.1 EVOWORLD-DATA CONSTRUCTION

Structured scenario generation. Each training example begins from a structured tuple specifying the domain, scene, subject and appearance, event, compatible reaction, and camera trajectory. Compatibility constraints exclude combinations whose consequence would be visually ambiguous or implausible for the designated subject. Gemini3.1-Pro expands the tuple into a continuous-shot script covering the initial observation, camera move-away, event reveal, camera move-back, and reacted subject. We render these scripts with Seedance 2.0 and 2.5 (Seedance et al., 2026; Team, 2026). The retained training set contains around 12k videos, each at least ten seconds long.

Verification and temporal annotation. Generated candidates are checked for continuous-shot structure, subject identity, complete subject exit and re-entry, visible event realization, visible reaction, and severe rendering failures. Camera direction is verified independently from sparse Lucas– Kanade feature tracks (Lucas & Kanade, 1981) and a RANSAC affine fit (Fischler & Bolles, 1981); clips with a missing reversal, full-circle motion, only unidirectional motion, or unreliable estimates are rejected. For each retained video, we record the subject-disappearance and reappearance times, camera phase boundaries, event interval, reaction interval, and generation provenance. Segment captions are produced jointly across adjacent intervals and audited to remove future-event spoilers, references to an absent subject, malformed camera commands, and redundant neighboring descrip tions.

## A.2 REVIEWED SEMANTIC TAXONOMY

Raw generation labels contain aliases and style prefixes and therefore overestimate semantic diversity. We consolidate the EvoWorld-Data labels into a two-level taxonomy: a coarse family and a fine-grained semantic type. Synonyms are merged, and rendering styles such as AAA, voxel, fantasy, or photorealistic do not create a new type unless the underlying event, environment, or reaction is semantically distinct. For EvoWorld-Bench, we report only event and scene coverage: the benchmark does not define or expose target reaction categories, and any causally plausible response is accepted. Table 5 reports the resulting human-reviewed statistics.

Table 5: Reviewed semantic coverage of EvoWorld-Data and EvoWorld-Bench. Reaction categories are reported only for the training data; EvoWorld-Bench does not define target reactions.
<table><tr><td rowspan="2">Split</td><td rowspan="2">Videos / cases</td><td colspan="2">Events</td><td colspan="2">Scenes</td><td colspan="2">Reactions</td></tr><tr><td>Fam.</td><td>Types</td><td>Fam.</td><td>Types</td><td>Fam.</td><td>Types</td></tr><tr><td>EvoWorld-Data</td><td>12k</td><td>19</td><td>65</td><td>16</td><td>63</td><td>5</td><td>14</td></tr><tr><td>EvoWorld-Bench</td><td>120</td><td>17</td><td>55</td><td>16</td><td>56</td><td>一</td><td>一</td></tr></table>

The event taxonomy includes, for example, fire and combustion, structural collapse, wild-animal intrusion, supernatural emergence, flooding, atmospheric disturbance, and vehicle failure. Reaction families in EvoWorld-Data include evasive relocation and active defense, among others.

## A.3 EVOWORLD-BENCH CONSTRUCTION AND TRAIN–TEST SCREENING

The 120 benchmark cases comprise 69 photorealistic and 51 game/virtual scenarios. Sixty-six initial frames come from held-out source videos and 54 are generated with GPT-Image-2 (OpenAI, 2026). We first remove exact source overlap and then screen every benchmark case against the complete training split at both the visual and semantic levels. Visual retrieval uses the benchmark initial frame to obtain its nearest training frames, while semantic retrieval operates on the paired initial and event prompts using lexical overlap and text-embedding similarity and requires both parts to come from the same training video. The union of these candidates is passed to a first VLM comparison. Pairs with substantial similarity in both the initial state and event are re-evaluated by a second multimoda pass that jointly receives the two initial frames and all four prompt fields. Ambiguous cases are manually adjudicated, and any benchmark candidate with a confirmed identical or near-identical training counterpart is removed or revised. A high first-frame similarity alone, or membership in the same scene or event category, is not treated as evidence of duplication.

## A.4 PROMPT PROTOCOL AND REACTION-INTENSITY CALIBRATION

We label every event by its required reaction intensity. A strong event requires an urgent response such as fleeing or taking cover, whereas a weak event permits either a mild or strong response. This calibration prevents a visually present but implausibly weak response to a dangerous event from being counted as correct.

At evaluation time, models receive the initial prompt, the event prompt at the appropriate switch point, and a blind revisit instruction that asks the subject to react without specifying how. The model must therefore infer an appropriate response from the event realized during its own rollout; the benchmark neither exposes nor requires a predefined reaction type, and any causally plausible response is considered valid. We audit all three prompts to ensure that the initial prompt does not reveal the event, the event prompt does not reveal the eventual response, and the blind revisit prompt contains neither a specific reaction nor future visual details.

## A.5 EVALUATION PROTOCOL

Our evaluation combines complementary sources of evidence rather than relying on a single judge. A strong VLM Qwen3.8 (Yang et al., 2025a) assesses task execution, causal plausibility, visual integrity, and revisit consistency. Established tool-based metrics provide independent diagnostics: VBench (Huang et al., 2024) measures motion smoothness, while CLIP ViT-B/32 (Radford et al., 2021) supplies the auxiliary event–prompt alignment comparison in Table 4.

For the VLM-based metrics, we temporally stratify each generated rollout into its beginning, middle, and ending portions and uniformly sample 30 frames across the three portions. The judge receives these frames, the initial and event prompts, and the strong/weak event-intensity label used to calibrate reaction strength, but not a target reaction or ground-truth revisit description. It jointly inspects whether the camera returns, whether the requested event is realized, what response the subject exhibits, whether that response is causally plausible at the required intensity, and whether the revisited subject and environment remain consistent. Event and revisit success are binary; the remaining perceptual and behavioral criteria use the three-level scale {0, 0.5, 1}.

Event and revisit success. Event Success is 1 only if the event described by the event prompt visibly occurs in the middle portion of the generated rollout; otherwise it is 0. Revisit Success is 1 only if the camera first leaves the original subject and region and subsequently returns to the same location with the subject visible again. A reaction may cause the subject to leave the frame after being revisited without invalidating the revisit. A cut to an unrelated scene, or a return without the original subject, receives 0.

Causal plausibility. This metric evaluates whether the subject’s changed behavior or state is a plausible consequence of the event that was actually rendered. The judge may use changes relative to stable landmarks—for example, retreating behind a doorway, moving behind cover, or relocating to higher ground—as evidence of an off-screen response. A score of 0 indicates no reaction when one is required, or a response that contradicts the event’s causal logic; 0.5 indicates a plausibly related but insufficiently strong response to a strong event; and 1 indicates a causally appropriate response at the expected intensity. We do not require one prescribed action: escape, cover, safe relocation, or defensive readiness may all receive full credit when appropriate. If either Event Success or Revisit Success is 0, Causal Plausibility is set to 0 because the generated video provides no valid event– reaction chain to assess.

Reaction quality and visual integrity. Reaction Quality isolates the visual realization of the response from its semantic appropriateness. It receives 0 for broken, distorted, or incoherent reaction motion; 0.5 for a visible but stiff or mildly artifacted response; and 1 for a natural and coherent action or changed state. If no changed reaction state is visible, the per-case value is null and is treated as 0 in the all-case mean reported for cross-model comparison. Visual Integrity evaluates the complete rollout independently of task correctness: 0 denotes severe structural breakdown, hallucination, or melting; 0.5 denotes a watchable video with visible local artifacts, instability, or flicker; and 1 denotes a clean and coherent rollout without obvious breakdown, with particular attention paid to prompt-switch boundaries.

Revisit consistency. Subject Consistency compares the opening subject with the revisited subject while explicitly allowing its pose, behavior, and state to change. A score of 0 denotes a different or severely drifted identity, 0.5 a recognizable subject with noticeable identity or appearance inconsistencies, and 1 a clearly preserved identity. If the subject does not reappear and therefore cannot be compared, the case receives a score of 0. The reported Subject Consistency is averaged over all 120 benchmark cases. Background Consistency compares the opening and revisited locations while allowing modest viewpoint changes: 0 indicates a different scene, 0.5 the same general place with a major object or layout inconsistency, and 1 the same location with its principal structures and objects preserved. Minor viewpoint changes or incidental small-object differences do not reduce the score.

Motion smoothness. We report the Motion Smoothness dimension of VBench (Huang et al., 2024) over all 120 generated videos. This metric is independent of the VLM judge and measures local temporal smoothness rather than completion of the event–revisit–reaction chain. Because it is highly saturated for modern video generators, we treat it as a secondary quality diagnostic rathe than a primary task metric.

Core success and aggregation. Let $r _ { i }$ and $e _ { i }$ denote binary Revisit and Event Success for case $i ,$ and let $c _ { i } \in \{ 0 , 0 . 5 , 1 \}$ denote Causal Plausibility. We define

$$
{ \mathrm { C o r e } } = { \frac { 1 0 0 } { N } } \sum _ { i = 1 } ^ { N } \mathbb { 1 } [ r _ { i } = 1 \ \wedge \ e _ { i } = 1 \ \wedge \ c _ { i } = 1 ] .\tag{6}
$$

Thus Core Success requires all three task-critical conditions in the same case. Reaction Quality and Visual Integrity are reported separately and are intentionally excluded from Core, allowing task completion and rendering quality to be analyzed independently. Unless otherwise stated, three-level metrics are averaged over all 120 benchmark cases and reported as percentages.

## B TRAINING HYPERPARAMETERS

Table 6: Training and inference settings for EvoWorld. Stage 2 inherits the Stage-1 LoRA architecture and IAA configuration.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Visual renderer VLM brain</td><td>LTX-2.5 (HaCohen et al., 2026) Qwen3.8-27B (Yang et al., 2025a)</td></tr><tr><td>Output resolution / frame rate Entity-reference length (V2V train / I2V benchmark)</td><td>896 × 512 / 24 fps 2 / 1 temporal latent steps</td></tr><tr><td>Temporal-anchor length</td><td>4 temporal latent steps</td></tr><tr><td>LoRA rank</td><td>128</td></tr><tr><td>IAA bias  $\beta$ </td><td>3.0</td></tr><tr><td></td><td></td></tr><tr><td>Stage-1 optimizer / precision</td><td>AdamW / bfloat16</td></tr><tr><td>Stage-1 learning rate / steps</td><td> $1 \times 1 0 ^ { - 4 } / 8 , 0 0 0$ </td></tr><tr><td>Stage-1 effective batch size</td><td>64 (8 GPUs, accumulation 8)</td></tr><tr><td>Stage-1 gradient clipping</td><td>1.0</td></tr><tr><td>Stage-2 learning rate / steps</td><td> $2 \times 1 0 ^ { - 5 } / 2 0 0$ </td></tr><tr><td>Stage-2 online / GT-anchor ratio</td><td> $7 5 \% / 2 5 \%$ </td></tr><tr><td>Online and test-time sampler</td><td>official distilled adapter, 8 steps</td></tr></table>

Training and benchmark inference modes. Both training stages optimize the renderer in a videoto-video (V2V) setting, using the first two temporal latent steps of the training clip as the entity reference. For benchmark evaluation, we use the same learned renderer directly in image-to-video (I2V) mode: the provided initial frame is encoded into one entity-reference latent step, after which generation proceeds autoregressively. Each later segment is conditioned on the final four latent steps of its generated predecessor. We also experimented with training the renderer directly in I2V mode and observed comparable benchmark performance; we therefore retain the unified V2V training configuration for the reported model.

## C TWO-STAGE TRAINING PROCEDURE

Stage 2 uses a one-hop rollout rather than regenerating the entire history chain: it exposes the renderer to direct predecessor errors while keeping training tractable. Because the predecessor is regenerated by the current model as training progresses, the conditioning distribution co-evolves with the renderer rather than remaining fixed to a single offline snapshot. To prevent this focused adaptation from overwriting the initial, event, and clean-revisit behavior learned in Stage 1, we mix 25% deterministic samples with ground-truth temporal anchors drawn across all phases and 75% samples with online self-generated temporal anchors. At deployment, the resulting Stage-2 LoRA is used together with the same official distilled LoRA and eight-step sampler employed during online rollout.

```python
Algorithm 1 Two-stage anchor and online-rollout construction
def stage1_sample(clip, target):
entity = opening_entity_anchor(clip)
prefix = gt_predecessor_tail(target, count=4)
return entity, prefix, target.is_revisit
def stage2_sample(clip, target, use_online):
if not use_online: # 25% Stage-1 rehearsal
clip, target = sample_stage1_window()
return stage1_sample(clip, target)
assert target.is_revisit and prompt_changed(target)
entity = opening_entity_anchor(clip)
temporal_anchor = gt_temporal_anchor(target.prev, count=4)
with no_grad():
prev = rollout_one_hop(current_model,
entity, temporal_anchor, target.prev.prompt,
sampler="distilled-8step")
prefix = stop_grad(prev.latents[-4:])
return entity, prefix, True
```

## D VLM BRAIN PROTOCOL

The VLM brain is called after every generated segment rather than only at the final revisit. Each call receives temporally sampled frames from the newly generated segment together with the accumulated textual world record. Its first responsibility is observational: it records visible entities and their attributes, persistent scene landmarks, object states, camera movement, and any newly realized event. Before an event has been observed, the brain updates this record but does not propose a reaction. This separation prevents the initial observation from leaking future event or reaction information.

After the event segment, the brain reasons from the event that was actually rendered, not merely from the requested event prompt. It identifies the affected off-screen subject, infers one causally plausible changed state or behavior, and converts that inference into a concrete renderer prompt. The prompt specifies the camera return, preserved identity cues, the subject’s post-event state, and visually realizable behavior, while avoiding hidden mental states or explanations that cannot be rendered. The benchmark remains open-ended: this inferred behavior need not match an exact reaction if it is causally plausible for the realized event.

For long-range identity conditioning, the framework can select a clear prior observation of the target entity and map it to the corresponding latent span. In the reported controlled I2V evaluation, the

![](images/c21326925f5e1f99f4a9ddf92de9f10f165484229cb49edcb60aff00fe9f4f98.jpg)  
“The camera pans left across the wet stone floor and rubble of the ruined nave, returning to the armored man. He has already raised the same round shield upright in front of his torso, gripping it with his left hand, his right hand resting on the sword hilt at his hip, both feet planted wide in a braced stance as he faces the gate.” Verify the rendered response  
Brain verification: defensive response matches the prompt, with consistent identity, equipment and scene.

Figure 7: Case study summarizing the VLM brain’s observation, event-conditioned reasoning, prompt generation, and verification over an out-of-sight rollout.

opening frame is deliberately constructed to show the target clearly and is used as the persistent one-step entity reference. Content-based retrieval is retained for longer or less constrained rollouts in which the clearest observation need not be the first frame. In either case, retrieval determines visual evidence, whereas the VLM-generated prompt determines the subsequent semantic state; the four most recent generated latent steps provide short-range temporal continuity independently of this decision.

Reasoning case study. Fig. 7 summarizes one reference brain-renderer interaction. The brain first records the knight’s identity, equipment, and cathedral environment before the camera moves away. It then observes the event actually realized by the renderer: although the requested gargoyle was intended to remain near the gate, the generated creature begins advancing into the nave. Based on this observed threat and the accumulated subject memory, the brain produces a revisit prompt in which the knight has raised the same shield and adopted a braced defensive stance. The final observation verifies that the rendered response is causally appropriate while preserving the subject, equipment, and scene. This example illustrates how the brain conditions its reasoning on the realized visual history rather than relying only on the requested event prompt.

## E ANCHOR LAYOUT AND IAA IMPLEMENTATION

Entity-reference and target-window tokens are combined through the native LTX conditioning path. Here, the target window denotes the complete generation-side latent sequence, including the clean temporal anchor and the target latents being denoised. Their physical order differs across code paths: training forms [entity reference | target window], whereas inference appends the reference and forms [target window | entity reference]. IAA therefore derives the entity-key range from the actual reference length and layout instead of assuming a fixed prefix position. Runtime checks verify the reference length, the contiguous denoised-query span, and the target–reference boundary.

IAA is enabled only for revisit windows, which are oversampled by a factor of three in Stage 1. If an entity reference contains $n _ { e }$ temporal latent steps and each step contains $H _ { \ell } W _ { \ell }$ spatial tokens, its key block contains $n _ { e } H _ { \ell } W _ { \ell }$ tokens. For each denoised target query, we multiply the corresponding entity-key entries of the LTX attention mask by exp(β). LTX subsequently converts this mask to an additive log-space bias, producing the $+ \beta$ term in Eq. (1). We use $\beta = 3 . 0 ;$ other valid pairs are unchanged, and originally masked pairs remain masked.

## F BASELINE REPRODUCTION DETAILS

Common prompt protocol. For every benchmark case, all autoregressive models receive the same three user-facing conditions: (i) an initial prompt describing the starting subject and scene, (ii) an event prompt introduced only after the camera has moved away, and (iii) a blind revisit prompt that asks the camera to return and the subject to react without revealing a particular reaction. The corresponding camera schedule moves the view away from the subject, exposes the event region, and then moves the view back. We inject each prompt at the nearest native chunk boundary and preserve the model’s recurrent state, latent history, or memory cache across switches. Thus, no model is reset at the event or revisit boundary. In the main comparison, baselines directly execute the blind revisit prompt and receive neither VLM-generated reaction reasoning nor entity retrieval; EvoWorld internally interprets the same blind request using its VLM brain. For a more detailed ablation, the prompt-transfer and shared-prompt analyses in Table 3 and Fig. 6 also provide baseline renderers with the brain-generated reasoning prompt, allowing us to separate the contribution of event-conditioned reasoning from that of the proposed data and renderer training.

Compatibility of additional recent world models. We also conducted preliminary protocolcompatibility trials with HY-World 1.5 (HunyuanWorld, 2025), Matrix-Game 3.0 (Wang et al., 2026), and AlayaWorld (Team et al., 2026a), but do not include them in the quantitative comparison because their released models did not reliably support the complete evaluation protocol. In particular, HY-World 1.5 frequently continued the preceding content while weakly realizing or entirely ignoring the event introduced by the mid-rollout prompt switch. Matrix-Game 3.0 and AlayaWorld likewise did not consistently complete the combined move-away, newly injected event, blind revisit, and event-conditioned reaction sequence under their available inference interfaces. This mismatch is unsurprising: these systems are trained primarily for their respective action-conditioned interaction, spatial persistence, or general world-generation objectives, rather than on data that explicitly couples mid-rollout event injection with off-screen subject evolution. Their behavior in our setting should therefore not be interpreted as a general assessment of their visual quality or intended capabilities. We leave systematic adaptation and controlled comparison of these models to future work.

Evaluated baselines. For LingBot-World v1 (Team et al., 2026b) and v2 (Gao et al., 2026), we use the official interactive interfaces, translate the benchmark trajectory into their native camera controls, and schedule all prompts within one persistent rollout. We evaluate the released YUME-1.5 5B checkpoint (Mao et al., 2026) at 720p through its long-context autoregressive interface. For SANA-WM (Zhu et al., 2026), we use the released streaming checkpoint and assign the three prompts to a 3/2/3 schedule over its eight autoregressive blocks, with camera motion supplied through its native trajectory interface. Echo-WM (Zhang et al., 2026b) is evaluated with sequential prompt switching while retaining its persistent autoregressive world state. Evoke and Evoke-Turbo (Yin et al., 2026) use the official image-to-video segment mode with matched inputs and prompt boundaries, differing only in their released variants and native sampling configurations.

One-shot references. We additionally evaluate Wan2.2 TI2V-5B (Wan et al., 2025), LTX-2.3, and LTX-2.5 (HaCohen et al., 2026) as general one-shot video generators. Each model receives a single composed prompt containing the initial, event, and blind revisit instructions and generates the complete video in one pass. These models therefore observe the intended event and camera-return instruction before generation begins, unlike the main autoregressive setting in which the event is introduced only during rollout. We report their results in Table 7 for completeness but do not include them in the main comparison.

Table 7: One-shot general video generators on EvoWorld-Bench, with EvoWorld included as an autoregressive reference below the separator. Best results among the one-shot models are bold (%).
<table><tr><td></td><td colspan="5">Task metrics</td><td colspan="4">Visual quality, consistency, and motion</td></tr><tr><td>Method</td><td>Core</td><td>Revisit</td><td>Event</td><td>Causal</td><td>React.</td><td>Visual</td><td>Subj. Cons.</td><td>Bg. Cons.</td><td>Motion</td></tr><tr><td>Wan2.2 TI2V-5B (Wan et al., 2025)</td><td>8.3</td><td>40.0</td><td>87.5</td><td>17.1</td><td>18.3</td><td>77.9</td><td>81.7</td><td>95.4</td><td>98.8</td></tr><tr><td>LTX-2.3 (HaCohen et al., 2026)</td><td>27.5</td><td>97.5</td><td>94.2</td><td>43.3</td><td>42.1</td><td>88.8</td><td>96.7</td><td>99.2</td><td>99.2</td></tr><tr><td>LTX-2.5 (HaCohen et al., 2026)</td><td>34.2</td><td>95.0</td><td>90.8</td><td>50.8</td><td>42.9</td><td>90.4</td><td>95.8</td><td>98.8</td><td>99.4</td></tr><tr><td>EvoWorld (ours; autoregressive)</td><td>90.8</td><td>100.0</td><td>97.5</td><td>94.2</td><td>85.0</td><td>83.8</td><td>99.2</td><td>98.3</td><td>99.5</td></tr></table>

We further evaluate MiniMax-H3 (Mini-Max, 2026) (∼33B) on EvoWorld-Bench based on the one-shot setting. As shown in Table 8, its high Event and Revisit scores do not translate into comparable Core task performance; successful components alone do not guarantee eventconditioned causal evolution.

Table 8: Task metrics of one-shot MiniMax-H3 on EvoWorld-Bench.
<table><tr><td>Method</td><td>Core</td><td>Revisit</td><td>Event</td><td>Causal</td><td>React.</td></tr><tr><td>MiniMax-H3 (MiniMax, 2026)</td><td>21.7</td><td>100.0</td><td>99.2</td><td>44.2</td><td>47.9</td></tr><tr><td>EvoWorld (ours; autoregressive)</td><td>90.8</td><td>100.0</td><td>97.5</td><td>94.2</td><td>85.0</td></tr></table>

## G LIMITATIONS AND FUTURE WORK

Although the segment-autoregressive design does not impose a fixed semantic horizon and can autoregressively extend the rollout, the present work focuses on a single event followed by a single subject reaction. This controlled scope isolates the minimal out-of-sight event–reaction chain, but does not yet capture long-horizon worlds in which multiple events interact, earlier reactions alter later conditions, or several entities respond over time. Extending the data, reasoning memory, training procedure, and evaluation protocol to multi-event and multi-reaction evolution is therefore an important direction for future work. In addition, our current scenarios primarily employ horizon tal move-away and move-back camera trajectories because they represent the most common and directly interpretable camera motion for revisit scenarios. We do not systematically evaluate more complex camera motion, such as large vertical displacement, depth-wise movement, orbiting, or compound six-degree-of-freedom trajectories, which may make scene correspondence, entity retrieval, and revisit consistency substantially more difficult. Future versions of EvoWorld-Data and EvoWorld-Bench should incorporate these trajectories and evaluate whether the framework remains robust under more general interactive camera control.

Acknowledgements. We thank all collaborators for their support and contributions throughout this project. We are also grateful to Su Guo, Jingyang Jia, and Shengrong Gu for their helpful suggestions on this work.