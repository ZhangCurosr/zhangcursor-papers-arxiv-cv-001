# Do Video Generators Track the World Across Segments? A Benchmark and Method for World-State Reasoning in Video Continuation

![](images/9aed5d3bdaab623d6f32fb6a9a5f42ca87c560781add3084e021f4ca7c878b83.jpg)

Yingmao Miao<sup>1,2,†</sup>, Pengfei Zhang<sup>2,‡</sup>, Chaoran Xu<sup>2,†</sup>, Meng Yu<sup>2,3,†</sup>, Jing Tang<sup>2,∗</sup>, Xiangxiang Chu<sup>2</sup>, Chao Shen<sup>1</sup>, Chenhao Lin<sup>1,∗</sup>

<sup>1</sup>Xi’an Jiaotong University, <sup>2</sup>DreamX Team, Alibaba Group, <sup>3</sup>Shanghai Jiaotong University <sup>†</sup>Work done during an internship at DreamX Team, Alibaba Group, <sup>‡</sup>Project leader, <sup>∗</sup>Corresponding authors.

Video generators build long videos by composing shorter parts, either by generating segments one after another or by autoregressively extending chunks. Each new part usually depends on memories of historical observations, such as recent frames, selected key frames, memory banks, or cached features. These memories preserve visible evidence from the past, but current generators do not reliably turn such evidence into a world-state interface: what holds in the video world after previous actions and how it should change under the next prompt. A past frame remains valid history, but it may not describe the state needed by the next segment; some states must instead be inferred from occluded or implicit changes rather than copied from a directly observed frame. This creates a simple but overlooked question for video continuation: given a previous video, its prompt, and a new prompt, can a model generate a continuation that reflects the state determined by both the historical video and the new prompt? To answer this question, we introduce StateBench, a benchmark that targets this gap by testing continuations over three state categories: past-visible states, occluded-process states, and complex-transition states. We further propose StateAgent, which explicitly maintains an entity-state representation, updates it under the new prompt, grounds the predicted post-action state as a future end frame, and renders the next video. Experiments show that our method improves controlled video continuation by raising the all-case state score (SCS-All) from 45.2 to 69.3, and also benefits story generation at the one-minute scale. Code is avaliable at https://github.com/AMAP-ML/StateAgent.

Date: September 2026

## 1 Introduction

As video generation aims for longer outputs and coherent visual stories, current systems face two practical choices. One is to generate a very long video in a generation process, which is computationally expensive and remains dificult for current models. The other is to generate shorter segments or chunks sequentially, where each new unit must be conditioned on the generated history. Providing the complete history to every new unit is also costly and often impractical, so existing continuation systems usually compress or select the history through recent frames, selected key frames (Zhou et al., 2024; Zhang et al., 2025), visual memory banks (Zhou et al., 2026; Xiao et al., 2026; Li et al., 2025; Wu et al., 2026c), or cached features (Chen et al., 2026b; Meng et al., 2026; Team et al., 2026). These signals form an observation memory of the generated history, useful for preserving appearance, identity, etc.

Observation memory may contain evidence for the world state, but current continuation methods expose it as frames or features rather than as an updated state interface. Historical frames record what was visible at particular moments, while the next segment needs what should hold after the previous video and how it should change under the new prompt. A past frame is valid history, but not always current-state evidence: the relevant state may be hidden at the boundary, changed after an earlier observation, or inferred from an occluded process. We view this evidence-to-state conversion as a capability future video world models should internalize; under current model capabilities, we make it explicit through state reasoning.

![](images/3e81b5093d8ff401c5cfee62d4115c33250c6b64e22024b8ecafeee285ff382e.jpg)

![](images/191b396080302cc2d61fd6e11b53ac73652f3c7d2a017349a9b99a95ee0c8cb3.jpg)  
Figure 1 Cross-segment state reasoning. Observation memory stores visible evidence from previous frames, while state reasoning updates what should now be true after the historical action. For example, after an object is placed into a container and becomes invisible, the next segment should know the object is still inside, even if the boundary frame only shows the container.

Several recent works have started to study hidden state, out-of-sight content, or long-horizon world dynamics, but they address diferent settings from video continuation. StEvo-Bench studies hidden state evolution within a single generated video, but it requires the model to establish, hide, evolve, and reveal the state in one generation, making failures hard to attribute (Ma et al., 2026). PAN uses LLM latent features as world states for long-horizon dynamics. This is useful for abstract simulation, but its state lacks explicit appearance binding, making it dificult to preserve the visual identity of entities in the next generated segment (Xiang et al., 2025). LiveWorld stores persistent 4D spatiotemporal memory for out-of-sight content, which can preserve rich scene evidence but is costly and dificult to scale. Crucially for our setting, it still stores past observations in a richer space, rather than explicitly converting historical events into an updated state for the next video segment (Duan et al., 2026). These diferences leave open how to evaluate and support state update in video continuation.

We study the basic unit of state-aware video continuation. Given a previous video $V _ { t } ,$ its source prompt $p _ { t } .$ and a continuation prompt $p _ { t + 1 } .$ , a model should infer the world state at time t + 1 by updating the historical state according to the continuation prompt, rather than merely maintaining visual continuity with previous frames. We introduce StateBench to operationalize this setting. StateBench contains 200 tasks across three state categories: past-visible states, occluded-process states, and complex-transition states. For each task, we provide a task-specific checklist. Evaluation first checks whether the generated continuation completes the state-establishing action requested by the prompt, and then checks whether the resulting state is correct.

We further propose StateAgent as a practical solution under current model capabilities. Our key insight is that state-consistent continuation should decouple state reasoning from video rendering: an explicit statereasoning role should decide the next state, while the video generator should render that predicted state into pixels. StateAgent implements this idea by building an entity-state representation from the previous video, updating it under the continuation prompt to obtain the post-action state, grounding that state as a future end frame using an image editor, and rendering the continuation. Experiments show that stronger observation memory does not make current generators reliably capture the required world state. Recent-frame, key-frame, memory-bank, and autoregressive baselines often preserve visible appearance, but still fail when the current state is hidden, changed after an earlier observation or produced by an unseen change. Overall, our contributions are:

• We identify the gap between observation memory and state reasoning in video continuation and introduce StateBench, a unit benchmark that tests whether generated continuations follow the state implied by the previous video and the new prompt.

• We propose StateAgent, which decouples state updating from video rendering by leveraging a VLM to parse historical states and predict future states, grounding them into future frames via an image editor, and rendering the video continuation through keyframe-to-video generation.

• We show that our method outperforms observation-based memory methods, not only improving controlled video continuation by raising the StateBench score from 45.2 to 69.3, but also benefiting long story generation.

## 2 Problem Formulation

## 2.1 Cross-Segment State Handof

Let $V _ { t }$ be a previous video, $p _ { t }$ its source prompt, and $p _ { t + 1 }$ the prompt for the next segment. Current continuation interfaces usually pass history forward through an observation-memory interface $O _ { t } = M _ { \mathrm { o b s } } ( V _ { t } )$ such as the last frame, recent frames, retrieved key frames, memory-bank entries, or cached temporal features. Here $M _ { \mathrm { o b s } }$ denotes the history interface actually exposed to the next generator, rather than the full information content of $V _ { t } .$ . This interface exposes visual evidence in sparse, compressed, or potentially stale forms, making it dificult for current generators to infer a state that can be queried, updated, and grounded for rendering. The next segment is then generated as:

$$
V _ { t + 1 } \sim G ( O _ { t } , p _ { t + 1 } ) .\tag{1}
$$

The problem is therefore an interface gap. At the segment boundary, the observation exposed to the next generator may not make the post-history state explicit. In our setting, this boundary is the tail frame of the previous video: the historical event has established a task-relevant state, but the tail frame hides or ambiguates the relation. A state-aware continuation process should therefore preserve both visual evidence and an explicit post-history state belief:

$$
S _ { t } = E ( V _ { t } , p _ { t } ) , \qquad V _ { t + 1 } \sim G ( O _ { t } , \ S _ { t } , \ p _ { t + 1 } ) ,\tag{2}
$$

where $O _ { t }$ preserves visual evidence such as appearance, identity, layout, and style, while $S _ { t }$ denotes what should be physically true after the previous video. The central cross-segment state handof problem occurs when the observation memory interface maps diferent histories (a) and (b) to similar exposed evidence, while their post-history states difer:

$$
M _ { \mathrm { o b s } } ( V _ { t } ^ { ( a ) } ) \approx M _ { \mathrm { o b s } } ( V _ { t } ^ { ( b ) } ) , \qquad S _ { t } ^ { ( a ) } \neq S _ { t } ^ { ( b ) } .\tag{3}
$$

A generator that relies only on the exposed observation memory interface can struggle to distinguish these cases under current model capabilities, while an explicit state representation can condition the next segment on what is currently true.

## 2.2 State as Causal Belief

We define a world state as a set of task-relevant physical relations that may persist beyond visibility. The previous video and its prompt establish a post-history state $S _ { t }$ . The continuation prompt then induces a state

![](images/1c6cfaefdac36e8bd89776a850a7cf497a058cd6c93030a6fdd5e8329a7ca405.jpg)  
Figure 2 StateBench overview. Given a previous video $V _ { t } ,$ its source prompt $p _ { t } ,$ and a continuation prompt p<sub>t+1</sub>, a method must generate $V _ { t + 1 }$ that reveals or uses the state established in the history. Left: task protocol. The previous segment contains the state-establishing event, while its tail frame, i.e., the generation boundary between two segments, hides or ambiguates the task-relevant relation. Middle: state-evidence categories. StateBench evaluates continuations across diferent ways in which the target state is evidenced by the history. Right: State-aware evaluation. SES first checks whether the requested reveal/revisit action is completed, and SCS then measures whether the exposed state is consistent with the previous video.

update that may reveal, use, or further transform this state. We write this state reasoning step as:

$$
S _ { t + 1 } = \tau ( S _ { t } , \ a _ { t + 1 } ) ,\tag{4}
$$

where $a _ { t + 1 }$ is the physical event described or implied by $p _ { t + 1 }$ . Here, $S _ { t + 1 }$ denotes the state that should be reflected in the continuation, and τ abstracts how the continuation prompt uses or updates the state carried from the previous segment. Cross-segment generation therefore requires inferring $S _ { t }$ from the historical event and generating $V _ { t + 1 }$ according to the state implied by both $S _ { t }$ and $p _ { t + 1 }$

In StateBench, we instantiate $S _ { t }$ as a set of task-relevant physical relations that expose the gap between observation and state:

$$
S _ { t } = \{ r _ { i } \} , \qquad r _ { i } \in \mathcal { F } ,\tag{5}
$$

where $\mathcal { F }$ denotes the state types evaluated in StateBench, including visibility, location, containment, transformation, surface and material state, quantity, and agent-object relations. These variables provide controlled tests of whether a continuation preserves the physical state implied by the previous segment.

## 2.3 Observation Memory Versus State Reasoning

Observation memory asks whether generated content can reuse or resemble previous visible evidence; state reasoning asks whether current generators can infer and maintain the physical consequences of previous actions for the next segment. As illustrated in Figure 1, appearance continuity and state continuity are therefore complementary but distinct. A visual memory mechanism can preserve how entities looked, where they appeared, and how the scene was framed, but it is not a reliable state interface for deciding what should now be true after an action. This distinction motivates StateBench, where continuations are evaluated not only by whether they follow the next prompt, but also by whether the revealed state is consistent with the previous video.

## 3 StateBench: Measuring State Discontinuity

StateBench is a compact unit test for state-aware video continuation. Each task provides a 5–10s previous video $V _ { t } ,$ its source prompt $p _ { t } ,$ , and a continuation prompt $p _ { t + 1 }$ . The previous segment establishes a physical state, but its tail frame hides or ambiguates the task-relevant relation. The continuation must therefore use the state established before the segment boundary, rather than only the terminal appearance, to complete the requested state update.

## 3.1 Task and Dataset

StateBench contains 200 cross-segment continuation cases across three state-evidence categories: 85 pastvisible cases, 65 occluded-process cases, and 50 complex-transition cases. In every case, the task-relevant state is hidden or ambiguous at the generation boundary, and the next prompt induces an action that exposes, uses, or transforms this state without naming the target answer. Past-visible cases contain a frame that directly shows the target state before it becomes hidden. Occluded-process cases involve state changes inside a container, behind an obstruction, or otherwise out of direct view. Complex-transition cases involve harder physical or relational state transitions, such as spatial transfer, quantity change, color mixing, dissolution, or material and surface transformations. Figure 2 summarizes the task protocol, category split, and evaluation design. Detailed schemas, fine-grained categories, and prompt construction rules are provided in the appendix.

## 3.2 Evaluation

Each generation method receives the same triplet $( V _ { t } , p _ { t } , p _ { t + 1 } )$ and generates one continuation $\hat { V } _ { t + 1 }$ . The evaluator receives $( V _ { t } , p _ { t } , p _ { t + 1 } , \hat { V } _ { t + 1 } )$ , a task-specific checklist, and reference frames selected from $V _ { t }$ for appearance-sensitive entities. These reference frames are used only for evaluation, not as additional generation inputs.

We use a two-stage checklist evaluation. State Exposure Score (SES) first measures whether the continuation completes the requested state-checking or state-updating action. For SES-positive samples, the state checklist contains state-correctness questions and anti-hallucination questions. Let $e _ { i } \in \{ 0 , 1 \}$ indicate successful exposure, and let $y _ { i q } \in \{ 0 , 1 \}$ denote whether the judge answers “yes” to checklist question q for sample i:

$$
\mathrm { S E S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e _ { i } .\tag{6}
$$

For SES-positive samples, State Correctness (SC) is the yes rate over state-correctness questions, while Hallucination Rate (HR) is one minus the yes rate over anti-hallucination questions:

$$
\mathrm { S C } = \frac { \sum _ { i } e _ { i } \sum _ { q \in \mathcal { C } _ { i } } y _ { i q } } { \sum _ { i } e _ { i } | \mathcal { C } _ { i } | } , \qquad \mathrm { H R } = 1 - \frac { \sum _ { i } e _ { i } \sum _ { q \in \mathcal { H } _ { i } } y _ { i q } } { \sum _ { i } e _ { i } | \mathcal { H } _ { i } | } .\tag{7}
$$

The conditional state consistency score is the yes rate over all state-checklist questions on SES-positive samples:

$$
\mathrm { S C S } _ { \mathrm { c o n d } } = \frac { \sum _ { i } e _ { i } \sum _ { q \in \mathcal { C } _ { i } \cup \mathcal { H } _ { i } } y _ { i q } } { \sum _ { i } e _ { i } \lvert \mathcal { C } _ { i } \cup \mathcal { H } _ { i } \rvert } .\tag{8}
$$

We also report an all-case score, SCS-All, which treats exposure failure as an end-to-end state failure:

$$
\mathrm { S C S } _ { \mathrm { A l l } } = \frac { 1 } { N } \sum _ { i } e _ { i } c _ { i } , \qquad c _ { i } = \frac { 1 } { \vert \mathcal { C } _ { i } \cup \mathcal { H } _ { i } \vert } \sum _ { q \in \mathcal { C } _ { i } \cup \mathcal { H } _ { i } } y _ { i q } .\tag{9}
$$

In the tables, SCS denotes $\mathrm { S C S } _ { \mathrm { c o n d } }$ for readability.

Following criterion-specific VLM evaluation protocols used in other benchmarks (Bansal et al., 2024; Meng et al., 2024; Ma et al., 2026), scores are produced by a checklist judge. The judge is given the previous video, prompts, generated continuation, task checklist, and entity reference frames when appearance matching is required.

## 4 StateAgent: Future-Frame State Carrier

StateBench defines a capability that future video generation models should eventually internalize: carrying a world state across generation boundaries and updating it under a new prompt. Current video generators, however, do not reliably expose such a state interface. We therefore present StateAgent as a practical solution for the current stage of model capability. The method decouples state prediction from state rendering:

![](images/b489e519a8356104df153d4af2e212f6753155d8e9e2d3d6a7e20818535130a2.jpg)  
Figure 3 StateAgent pipeline. StateAgent decouples state prediction from video rendering. Given $V _ { t } ,$ $p _ { t } ,$ and $p _ { t + 1 }$ it parses the historical event into a structured state graph, predicts the target post-action state, grounds it into a future end frame through image editing, and renders the continuation with first–last-frame video generation.

an explicit predictor first infers what should be true after the continuation prompt, and a video generator then renders a segment constrained by that predicted state. StateAgent instantiates this state handof formulation by making the intermediate state update explicit as shown in Figure 3. The parsed state graph represents $S _ { t } ,$ , the state after the previous segment. Given the continuation prompt, StateAgent predicts a target continuation state $\hat { S } _ { t + 1 }$ , grounds it as a future end frame, and then renders the next segment toward that predicted state.

This decoupling raises a representation question. Purely textual states are easy to update but too abstract to preserve appearance bindings; implicit latent states may support dynamics, but are dificult to inspect, costly to train, and can discard visual details needed for faithful continuation. StateAgent instead uses a future end frame as a visualized state carrier. A multimodal agent predicts the structured post-action state, an image-editing model grounds that state into pixels with historical appearance references, and a keyframe-to-video generator renders the motion between the historical end frame and the predicted future end frame. This follows a similar intuition to ImageWAM, which uses image editing as a compact target-state transformation prior for world-action modeling (Zhang et al., 2026).

## 4.1 Structured State Carrier

Given $( V _ { t } , p _ { t } )$ , StateAgent builds a lightweight state graph $S _ { t }$ rather than retrieved frames. Each entity node stores an identity, references to frames where the entity is visible, and a short state description that summarizes its current condition, such as whether it is open, broken, hidden, held, or stained. Relations record visibility, containment, location and so on. The graph is deliberately sparse: it stores only the state variables needed for future continuation, while leaving texture, pose, and motion synthesis to the renderer. Unlike key-frame memory, it can mark a relation as true even when not visible at the boundary, and it can mark an earlier visual observation as stale after a later action changes the state. When constructing $S _ { t } ,$ StateAgent uses $p _ { t }$ only for entity binding between names and visual entities, while state parsing is grounded in the observed video evidence.

## 4.2 Prompt-Conditioned State Update

The continuation prompt is treated as an action condition that induces a state update. Given the parsed post-history state $S _ { t }$ and $p _ { t + 1 }$ , a state updater U predicts the target continuation state:

$$
\hat { S } _ { t + 1 } = \mathcal { U } ( S _ { t } , ~ p _ { t + 1 } ) .\tag{10}
$$

Here, $\hat { S } _ { t + 1 }$ is the method’s explicit estimate of the state that should be reflected in $V _ { t + 1 }$ . It may reveal a hidden relation, use an existing relation, or further transform the physical state. The output specifies the target state for the continuation. This directly addresses the interface bottleneck of observation memory: hidden objects can remain in $S _ { t } .$ , while stale appearances from earlier frames are not treated as the current state after later actions.

## 4.3 Future-Frame Grounding

A structured state alone is not enough to condition a video generator, because it lacks concrete appearance binding. StateAgent therefore grounds $\hat { S } _ { t + 1 }$ into a future end frame $\hat { F } _ { t + 1 } ^ { \mathrm { e n d } }$ . This step converts the symbolic state prediction into a visual condition that current video generators can follow. The historical end frame $F _ { t } ^ { \mathrm { e n d } }$ provides camera, layout, and visible entities; reference frames $R _ { t }$ provide appearances for entities that are not visible at the boundary but are needed in the predicted state. A frame-grounding module $\mathcal { G } _ { \mathrm { f r a m e } }$ receives the historical end frame, selected references, the predicted state, and the continuation prompt:

$$
\hat { F } _ { t + 1 } ^ { \mathrm { e n d } } = \mathcal { G } _ { \mathrm { f r a m e } } ( F _ { t } ^ { \mathrm { e n d } } , R _ { t } , \hat { S } _ { t + 1 } , p _ { t + 1 } ) .\tag{11}
$$

The editing instruction is generated from the predicted state change and the selected visual references, so that the edited frame reflects the target state while preserving appearance and layout. The target frame is thus not a memory frame; it is a visualized prediction of the future state.

## 4.4 Video Rendering and Diagnostics

Finally, a keyframe-to-video generator $\mathcal { G } _ { \mathrm { v i d e o } }$ renders the continuation:

$$
\begin{array} { r } { \hat { V } _ { t + 1 } = \mathcal G _ { \mathrm { v i d e o } } ( F _ { t } ^ { \mathrm { e n d } } , \hat { F } _ { t + 1 } ^ { \mathrm { e n d } } , p _ { t + 1 } ) , } \end{array}\tag{12}
$$

where we instantiate $\mathcal { G } _ { \mathrm { v i d e o } }$ with the keyframe-to-video interface of Wan (Wan et al., 2025). The historical end frame keeps temporal continuity with the previous segment, while the future end frame constrains the state that must be reflected by the continuation. The renderer is therefore responsible for motion and visual realism, rather than rediscovering hidden physical state from scratch.

The modular structure makes failures inspectable. A wrong continuation can be attributed to history parsing, state update, future-frame grounding, or video rendering. This is useful because black-box video generation failures often look identical in the final video, while StateAgent exposes whether the system failed to carry the state, update it, bind it to appearance, or animate toward it.

## 5 Experiments

## 5.1 Experimental Setup

StateAgent uses Qwen3.5-Plus (Yang et al., 2025) as the parser for historical states, the state updater, and the editing-instruction generator, Wan2.7-Image (Mao et al., 2026) as the future-frame generator and Wan2.2- KF2V-Flash as the state-to-video renderer. We evaluate all methods under the same input $( V _ { t } , p _ { t } , p _ { t + 1 } )$ . Each method generates only the next segment; the previous video is used only as conditioning evidence. Unless otherwise specified, output length, resolution, prompt template, sampling settings, and random seeds are matched within each model family. We report SES and SCS across the three StateBench categories. For diagnosis, we also report State Consistency (SC) and Hallucination Rate (HR) within each category; the Overall columns report aggregate SES, conditional SCS, and SCS-All.

Main baselines. The main comparison covers three families of continuation interfaces. Frame-conditioned baselines include I2V from the final frame and a recent-frame (Team et al., 2025) variant that conditions on the last few frames. Key-frame and memory baselines include StoryMem (Zhang et al., 2025) and StoryMem supplied with full key frames covering the complete state-establishing process. Autoregressive baselines include MAGI (Teng et al., 2025).

## 5.2 Main Results on StateBench

Table 1 evaluates whether stronger visual history interfaces let current generators recover the state needed for cross-segment handof. Although observation-memory methods often complete the action described by the prompt, they frequently expose a state inconsistent with the previous segment. StateAgent improves conditional SCS from 58.5 to 74.9 over the strongest baseline, and improves the all-case SCS-All from 45.2 to

![](images/6521bc1d4a208f578190242ef1ccd4a2bf644a584b3cc79668ee76d594f9db7d.jpg)

Figure 4 Example on StateBench. Setup prompt: a woman drops a green apple into an open cardboard box while a banana remains on the kitchen counter. Continuation prompt: “A single continuous realistic video on a kitchen counter. The woman tilts the cardboard box toward the camera to show what is inside. Fixed camera, medium shot, bright indoor lighting.”
<table><tr><td>Method</td><td colspan="4">Past-visible</td><td colspan="4">Occluded-process</td><td colspan="4">Complex-transition</td><td colspan="3">Overall</td></tr><tr><td></td><td>SES↑</td><td></td><td>SC↑ HR↓</td><td>SCS↑</td><td>SES↑</td><td>SC↑ HR↓</td><td></td><td>SCS↑</td><td>SES↑</td><td>SC↑</td><td>HR↓</td><td>SCS↑</td><td>SES↑</td><td>SCS↑</td><td>SCS-All↑</td></tr><tr><td>Wan-I2V</td><td>69.4</td><td>26.0</td><td>40.7</td><td>42.9</td><td>90.8</td><td>22.6</td><td>49.2</td><td>36.1</td><td>76.0</td><td>17.1</td><td>18.4</td><td>48.2</td><td>78.0</td><td>41.6</td><td>32.5</td></tr><tr><td>LongCat-Video (recent frames)</td><td>57.6</td><td>27.9</td><td>22.4</td><td>51.8</td><td>70.8</td><td>23.9</td><td>27.2</td><td>46.0</td><td>60.0</td><td>10.6</td><td>10.0</td><td>49.2</td><td>62.5</td><td>49.0</td><td>30.6</td></tr><tr><td>StoryMem (Zhang et al., 2025)</td><td>80.0</td><td>50.5</td><td>15.4</td><td>66.9</td><td>75.4</td><td>34.7</td><td>20.4</td><td>54.8</td><td>66.0</td><td>27.8</td><td>34.8</td><td>46.5</td><td>75.0</td><td>58.5</td><td>43.9</td></tr><tr><td>StoryMem + full KFs</td><td>91.2</td><td>51.9</td><td>13.5</td><td>68.3</td><td>86.0</td><td>36.1</td><td>19.4</td><td>55.5</td><td>56.0</td><td>19.0</td><td>37.5</td><td>41.1</td><td>78.5</td><td>57.5</td><td>45.2</td></tr><tr><td>MAGI-1 (Teng et al., 2025)</td><td>40.0</td><td>19.6</td><td>23.5</td><td>47.1</td><td>40.0</td><td>21.8</td><td>51.9</td><td>33.5</td><td>12.0</td><td>44.4</td><td>25.0</td><td>59.2</td><td>33.0</td><td>42.8</td><td>14.1</td></tr><tr><td>STATEAGENT</td><td>89.4</td><td>61.0</td><td>11.2</td><td>74.1</td><td>96.9</td><td>68.8</td><td>10.3</td><td>78.5</td><td>92.0</td><td>61.2</td><td>17.4</td><td>71.4</td><td>92.5</td><td>74.9</td><td>69.3</td></tr></table>

Table 1 Main StateBench results. SES measures completion of the requested state-exposing or state-updating action; SCS denotes conditional state consistency on SES-positive samples. SC is the yes rate over state-correctness questions, while HR is the no rate over anti-hallucination questions. SCS-All is the all-case state score and treats exposure failure as failure.

69.3, showing that the bottleneck is not merely access to historical evidence, but converting that evidence into the state required by the continuation.

The category-wise results reveal three failure modes. In Past-visible cases, retrieved frames help, but the model must still decide whether a past observation remains current. In Occluded-process cases, the target state is produced by a hidden update and never appears as a reusable target frame. In Complex-transition cases, similarity to past observations is a weak cue for spatial transfer, quantity change, color mixing, or physical-state change. Figure 4 makes these failures concrete: some baselines fail to tilt the box toward the camera, so the state-checking action is incomplete; others open the box but show an empty interior despite the apple being placed inside in the previous segment; still others hallucinate an ambiguous or mismatched object. StateAgent avoids these failures by predicting that the apple should remain inside the box and grounding that target state as a future end frame before rendering.

## 5.3 Ablation on State Conditioning

Table 2 shows that the gain is not from stronger hints alone. State-based prompt enhancement improves SES to 90.0, confirming that textual state hints help execute the requested action, but its SCS remains 61.9 because text lacks the visual grounding needed to bind the predicted state to entity appearance and so

![](images/70556a47defa18ea37187de013104ab418ddece067d3ec0b763ae18dbeebc0e0.jpg)  
Figure 5 Long story generation example of ST-Bench. The same story script is given to StoryMem and StateAgent.

<table><tr><td>Variant</td><td>SES↑</td><td>SCS↑</td><td>SCS-All↑</td></tr><tr><td>Base I2V</td><td>78.0</td><td>41.6</td><td>32.5</td></tr><tr><td>Direct end-frame Editing</td><td>79.0</td><td>56.6</td><td>44.7</td></tr><tr><td>State-based Prompt Enhancement</td><td>90.0</td><td>61.9</td><td>55.8</td></tr><tr><td>Full STATEAGENT</td><td>92.5</td><td>74.9</td><td>69.3</td></tr></table>

<table><tr><td>Method</td><td>Aesthetic</td><td>PF</td><td>Consistency</td></tr><tr><td>StoryMem</td><td>5.81</td><td>0.222</td><td>0.586</td></tr><tr><td>STATEAGENT</td><td>6.35</td><td>0.230</td><td>0.594</td></tr></table>

Table 2 Ablation on state conditioning.  
Table 3 ST-Bench results for multi-shot story generation.

on. Direct end-frame editing provides such visual grounding and improves over several observation-memory baselines, but without the full state-update pipeline it still trails StateAgent. The full pipeline achieves the best SCS and SCS-All by coupling state prediction, future-frame grounding, and video rendering.

## 5.4 State Consistency in Multi-Shot Story Generation

To test whether our method remains useful beyond the controlled unit setting of StateBench, we evaluate multi-shot story generation under the ST-Bench protocol from StoryMem (Zhang et al., 2025). ST-Bench consists of 1-minute long story scripts, shot-level video prompts, and scene-cut indicators. Following this protocol, each method generates the story shot by shot from the same script-level inputs. We evaluate aesthetic quality, prompt following, and cross-shot consistency following StoryMem.

Table 3 reports the ST-Bench metrics, while Figure 5 provides a qualitative comparison. StateAgent achieves improvements across all metrics. The qualitative example further shows that explicit state update helps preserve entity states and object appearances while following scene-level prompt changes.

The qualitative example reveals the limitation of a fixed visual memory bank. StoryMem stores only a limited number of memory frames, with early key frames acting as a memory sink and recent frames capturing short-term dynamics. This can make early characters and scenes repeatedly dominate later generations, causing repeated backgrounds, incorrect binding of new characters to earlier appearances, and loss of small but state-relevant objects. For example, a palm-leaf fan may drift into a paper fan when its appearance is diluted in memory. StateAgent instead maintains explicit entity states and updates them under each prompt, allowing it to follow scene changes, preserve object appearance and state, and avoid incorrect character binding.

## 6 Related Work

Video continuation and visual memory. Video generators achieve strong short-term quality with DiT backbones (Ho et al., 2022b,a; Singer et al., 2022; Peebles and Xie, 2023; Ma et al., 2024; Wan et al., 2025; Kong et al., 2024). Long-video systems extend them through chunkwise continuation (Henschel et al., 2024), selected keyframes (Zhou et al., 2024; Zhang et al., 2025), entity or spatial memories (Zhou et al., 2026; Xiao et al., 2026; Li et al., 2025; Wu et al., 2026c), and KV caches (Chen et al., 2026b; Zhao et al., 2026; Meng et al., 2026). These interfaces preserve useful observations such as appearance, identity and so on. Our focus is complementary: converting historical observations into the current world state required by the next segment.

State evolution and video world models. Recent systems have started to explore evolving unseen states. PAN models long-horizon dynamics with LLM latent features and a video decoder (Xiang et al., 2025), but its state lacks explicit appearance binding for continuation. LiveWorld maintains a persistent 4D spatiotemporal memory for out-of-sight dynamics (Duan et al., 2026), but this powerful scene-level representation is costly. ReMind trains video generators to use KV-cache mechanisms as dynamic memory for hidden-state evolution (Xu et al., 2026), showing that specialized training can improve this ability. Our work instead evaluates this capability in current continuation systems and tests an explicit state carrier without retraining the video generator.

Benchmarks for generation, memory, and state reasoning. Video-generation benchmarks initially focused on visual quality, text alignment and so on (Huang et al., 2024; Liu et al., 2024; Sun et al., 2025; Bansal et al., 2024; Meng et al., 2024; Upadhyay et al., 2026; Feng et al., 2026; Wu et al., 2026b). Recent benchmarks have begun to shift toward memory, state reasoning, and out-of-sight state evolution. WorldReasonBench evaluates whether video generators can predict future world states from an initial condition and action (Wu et al., 2026a). MIND studies memory consistency and action control in video world models (Ye et al., 2026). STEVO-Bench studies whether natural state evolution can continue when observation is interrupted by occluders, lights-of intervals, or look-away trajectories, but state setup, hiding, evolution, and reveal are coupled within one generation, making failures hard to attribute (Ma et al., 2026). MemoBench studies revisiting behavior in dynamically changing environments, but its focus is camera-controlled video generation (Chen et al., 2026a). Despite this progress, world-state handof in cross-segment video continuation remains underexplored: a historical video establishes a world state, and a later prompt asks the next generated segment to use or change it.

## 7 Conclusion

We introduced cross-segment state handof as a requirement distinct from observation memory in video continuation. We proposed StateBench to test whether continuations preserve and update states established by previous segments across past-visible, occluded-process, and complex-transition cases. We also presented StateAgent, which explicitly predicts the continuation state, grounds it as a future end frame, and renders the next segment. Experiments show that stronger visual memory alone does not make current generators reliably infer world state; coherent continuation benefits from an explicit state belief that can be updated, grounded to appearance, and carried across generation boundaries, with supporting evidence from ST-Bench story generation.

## References

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation, 2024.

Haoyu Chen, Kaichen Zhou, Hang Hua, Kaile Zhang, Jingwen Qian, Wufei Ma, Haonan Chen, Chunjiang Liu, Yizhou Zhao, Xiaoyuan Wang, et al. Memobench: Benchmarking world modeling in dynamically changing environments, 2026a.

Shuo Chen, Cong Wei, Sun Sun, Tiancheng Shen, Ping Nie, Kai Zou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. In Forty-third International Conference on Machine Learning, 2026b.

Zicheng Duan, Jiatong Xia, Zeyu Zhang, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen, Xinyu Zhang, and Lingqiao Liu. Liveworld: Simulating out-of-sight dynamics in generative video world models, 2026.

Xiaokun Feng, Haiming Yu, Meiqi Wu, Shiyu Hu, Jintao Chen, Chen Zhu, Jiahong Wu, Xiangxiang Chu, and Kaiqi Huang. Narrlv: Towards a comprehensive narrative-centric evaluation for long video generation. In International Conference on Learning Representations, volume 2026, pages 20966–21000, 2026.

Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text, 2024.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with difusion models, 2022a.

Jonathan Ho, Tim Salimans, Alexey A Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video difusion models. In ICLR workshop on deep generative models for highly structured data, 2022b.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models, 2024.

Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 25690–25699, 2025.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22139–22149, 2024.

Xin Ma, Yaohui Wang, Xinyuan Chen, Gengyun Jia, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent difusion transformer for video generation, 2024.

Ziqi Ma, Mengzhan Liufu, and Georgia Gkioxari. Out of sight, out of mind? evaluating state evolution in video world models, 2026.

Chaojie Mao, Chen-Wei Xie, Chongyang Zhong, Haoyou Deng, Jiaxing Zhao, Jie Xiao, Jinbo Xing, Jingfeng Zhang, Jingren Zhou, Jingyi Zhang, Jun Dan, Kai Zhu, Kang Zhao, Keyu Yan, Minghui Chen, Pandeng Li, Shuangle Chen, Tong Shen, Yu Liu, Yue Jiang, Yulin Pan, Yuxiang Tuo, Zeyinzi Jiang, Zhen Han, Ang Wang, Bang Zhang, Baole Ai, Bin Wen, Boang Feng, Feiwu Yu, Gang Wang, Haiming Zhao, He Kang, Jianjing Xiang, Jianyuan Zeng, Jinkai Wang, Junjie Zhou, Ke Sun, Linqian Wu, Pei Gong, Pingyu Wu, Ruiwen Wu, Tongtong Su, Wenmeng Zhou, Wenting Shen, Wenyuan Yu, Xianjun Xu, Xiaoming Huang, Xiejie Shen, Xin Xu, Yan Kou, Yangyu Lv, Yifan Zhai, Yitong Huang, Yun Zheng, Yuntao Hong, Zhe Zhang, and Zhicheng Zhang. Wan-image: Pushing the boundaries of generative visual intelligence, 2026. URL https://arxiv.org/abs/2604.19858.

Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation, 2024.

Yihao Meng, Zichen Liu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Yue Yu, Hanlin Wang, Haobo Li, Jiapeng Zhu, Yanhong Zeng, et al. Causalcine: Real-time autoregressive generation for multi-shot video narratives, 2026.

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data, 2022.

Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8406–8416, 2025.

DreamX Team, Yancheng Bai, Rui Chen, Xiangxiang Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu, Siyu Hong, Jiachen Lei, et al. Dreamx-world 1.0: A general-purpose interactive world model. arXiv preprint arXiv:2606.16993, 2026.

Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, and Tong Zhang. Longcat-video technical report, 2025. URL https://arxiv.org/abs/25 10.22200.

Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale, 2025.

Rishi Upadhyay, Howard Zhang, Jim Solomon, Ayush Agrawal, Pranay Boreddy, Shruti Satya Narayana, Yunhao Ba, Alex Wong, Celso M de Melo, and Achuta Kadambi. Worldbench: Disambiguating physics for diagnostic evaluation of world models, 2026.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models, 2025.

Keming Wu, Yijing Cui, Wenhan Xue, Qijie Wang, Xuan Luo, Zhiyuan Feng, Zuhao Yang, Sudong Wang, Sicong Jiang, Haowei Zhu, et al. Worldreasonbench: Human-aligned stress testing of video generators as future world-state predictors, 2026a.

Meiqi Wu, Zhixin Cai, Fufangchen Zhao, Xiaokun Feng, Rujing Dang, Bingze Song, Ruitian Tian, Jiashu Zhu, Jiachen Lei, Hao Dou, et al. Omni-worldbench: Towards a comprehensive interaction-centric evaluation for world models. arXiv preprint arXiv:2603.22212, 2026b.

Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. Advances in Neural Information Processing Systems, 38:49371–49393, 2026c.

Jiannan Xiang, Yi Gu, Zihan Liu, Zeyu Feng, Qiyue Gao, Yiyan Hu, Benhao Huang, Guangyi Liu, Yichi Yang, Kun Zhou, et al. Pan: A world model for general, interactable, and long-horizon world simulation, 2025.

Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. Advances in Neural Information Processing Systems, 38: 49632–49652, 2026.

Tianshuo Xu, Yichen Xie, Depu Meng, Chensheng Peng, Quentin Herau, Bo Jiang, Yihan Hu, and Wei Zhan. Teaching video generators to remember: Eliciting dynamic memory for out-of-sight state evolution, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Yixuan Ye, Xuanyu Lu, Yuxin Jiang, Yuchao Gu, Rui Zhao, Qiwei Liang, Jiachun Pan, Fengda Zhang, Weijia Wu, and Alex Jinpeng Wang. Mind: Benchmarking memory consistency and action control in world models, 2026.

Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory, 2025.

Yuyang Zhang, Wenyao Zhang, Zekun Qi, He Zhang, Haitao Lin, Jingbo Zhang, Yao Mu, Xiaokang Yang, Wenjun Zeng, and Xin Jin. Imagewam: Do world action models really need video generation, or just image editing?, 2026.

Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive difusion distillation for real-time interactive video generation, 2026.

Jinsong Zhou, Yihua Du, Xinli Xu, Luozhou Wang, Zijie Zhuang, Yehang Zhang, Shuaibo Li, Xiaojun Hu, Bolan Su, and Ying-cong Chen. Videomemory: Toward consistent video generation via memory integration, 2026.

Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydifusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems, 37:110315–110340, 2024.

## A Experimental Details

This section provides more details: task metadata, implementation details of StateBench and StateAgent, settings of baselines.

## A.1 StateBench Data Construction

We construct StateBench by first defining fine-grained state types and then generating paired prompts for the previous segment and the continuation segment. The previous prompt creates a state-establishing event, while the continuation prompt asks for an action that exposes, uses, or transforms the resulting state without naming the expected answer. Previous videos are generated with Wan2.2-T2V/I2V-plus at 480p resolution.

<table><tr><td>State family</td><td>Fine-grained types</td><td>Examples of state reasoning</td></tr><tr><td>Existence</td><td>exit_frame, departure</td><td>containment, occlusion, coverage, Object remains inside a container; object is hidden, covered, or leaves the scene.</td></tr><tr><td>Quantity/Degree</td><td>tying</td><td>consumption, addition, division, emp- Amount increases, decreases, is consumed, divided, or emptied.</td></tr><tr><td>Spatial</td><td>displacement, swap, flip</td><td>Object moves, swaps position, flips, or is tracked under occlusion.</td></tr><tr><td>Physical state</td><td>cooking</td><td>open_close, fold, breakage, melting, Object opens or closes, folds, breaks, melts, or changes through heating/cooking.</td></tr><tr><td></td><td>Surface/Appearance writing, staining, labeling, wearing</td><td>Text, stains, labels, or worn objects change visible appearance.</td></tr><tr><td>Agent state</td><td>change</td><td>held_object, posture, costume_- Agent holds a different object, changes posture, or changes visible outfit or appearance.</td></tr></table>

Table 4 Fine-grained state types used in StateBench.
<table><tr><td>Component / Baseline</td><td>Model / Interface</td><td>Notes</td></tr><tr><td>Previous-video generation</td><td>Wan2.2-T2V/I2V-plus</td><td>480p, 16 FPS, manually filtered</td></tr><tr><td>Final-frame I2V baseline</td><td>Wan2.2-I2V-A14B</td><td>Conditions on the historical end frame</td></tr><tr><td>Recent-frame baseline</td><td>LongCat-Video</td><td>Video-prefix continuation interface</td></tr><tr><td>Memory baseline</td><td>StoryMem (Zhang et al., 2025)</td><td>Official memory-based continuation interface</td></tr><tr><td>Full key-frame baseline</td><td>StoryMem + full KFs</td><td>Uniform historical KFs: 6 frames per 5 seconds; max 10 frames in the StoryMem memory bank</td></tr><tr><td>Autoregressive baseline</td><td>MAGI-1 (Teng et al., 2025)</td><td>Native video continuation baseline</td></tr><tr><td>VLM components and judge</td><td>Qwen3.5-Plus</td><td>State parsing, update, prompting, and evaluation</td></tr><tr><td>Future-frame grounding</td><td>Wan2.7-Image</td><td>Image generation/editing API</td></tr><tr><td>Keyframe-to-video rendering</td><td>Wan2.2-KF2V-Flash</td><td>Historical end frame + future end frame</td></tr></table>

Table 5 Base models and interfaces used in our experiments.

For tasks requiring multi-step setup, intermediate setup segments are generated sequentially by conditioning on the previous segment boundary.

All generated histories are manually filtered before evaluation. A retained case must satisfy four conditions: (1) the state-changing event is completed, (2) the boundary frame does not directly reveal the task-relevant state, (3) the continuation prompt does not leak the target answer, and (4) the case can be evaluated by concrete exposure/action and state-consistency checklists. Cases that fail these checks are regenerated or removed. The final benchmark contains 85 past-visible, 65 occluded-process, and 50 complex-transition cases.

## A.2 Case Schema and State Types

Each StateBench item is stored as a structured record containing task metadata, prompts, state annotations, and evaluation checklists:

id: unique task identifier.

category: one of past-visible, occluded-process, or complex-transition.

state\_type: coarse state family, such as existence, quantity/degree, spatial, physical state, surface/appearance, or agent state.

fine\_type: fine-grained state type.

previous\_video: previous segment containing the state-establishing event.

previous\_prompt: source prompt used to generate the previous segment.

state\_after\_previous: annotated state at the historical end frame.

continuation\_prompt: prompt for the next segment.

target\_state: state that should be reflected in the continuation.

evaluation: exposure/action checklist and state-consistency checklist.

Table 4 summarizes the fine-grained state types. These types are controlled probes rather than a complete ontology of world state.

<table><tr><td>Dimension</td><td>Judgments</td><td>Score</td><td>Ours win</td><td>Baseline win</td><td>Tie</td><td>Win w/o ties</td><td>Maj. ours</td><td>Maj. agree</td></tr><tr><td>Aesthetic quality</td><td>450</td><td>74.7</td><td>61.3</td><td>12.0</td><td>26.7</td><td>83.6</td><td>68.8</td><td>74.0</td></tr><tr><td>State reveal success</td><td>450</td><td>81.0</td><td>71.8</td><td>9.8</td><td>18.4</td><td>88.0</td><td>80.0</td><td>81.0</td></tr><tr><td>State consistency</td><td>450</td><td>85.6</td><td>81.3</td><td>10.2</td><td>8.4</td><td>88.8</td><td>82.5</td><td>90.0</td></tr><tr><td>All</td><td>1,350</td><td>80.4</td><td>71.5</td><td>10.7</td><td>17.9</td><td>87.0</td><td>77.1</td><td>81.7</td></tr></table>

Table 6 Pairwise human evaluation by dimension. Score uses 1/0.5/0 for ours win/tie/baseline win. Win rates, tie rates, majority preference, and majority agreement are reported in percentages.
<table><tr><td>Baseline</td><td></td><td></td><td>N Score Ours win Base win</td><td></td><td>Tie</td></tr><tr><td>Final-frame I2V</td><td>J375</td><td>77.5</td><td>66.1</td><td>11.222.7</td><td></td></tr><tr><td>LongCat-Video</td><td>318</td><td>80.0</td><td>71.4</td><td>11.3 17.3</td><td></td></tr><tr><td>StoryMem</td><td>369</td><td>72.1</td><td>61.0</td><td>16.822.2</td><td></td></tr><tr><td>MAGI-1</td><td>288</td><td>95.3</td><td>92.0</td><td></td><td>1.4 6.6</td></tr></table>

Table 7 Pairwise human evaluation by baseline. Scores and rates are aggregated over all three dimensions and reported in percentages.

## A.3 Checklist Evaluation and VLM Judge

In this section, we describe the checklist instantiation. Each task contains three question groups: reveal achieved, state correct, and no violation. The first group contains one action-completion question. The second group checks whether the exposed or used state matches the annotated target state. The third group checks that the reveal does not introduce unsupported objects, duplicated targets, identity changes, contradictory attributes, or impossible contents.

Some checklist questions require appearance or identity comparison against the history. For these cases, we provide the judge with task-specific reference frames in which the relevant entity appearance can be clearly bound to the historical evidence. These references help the evaluator decide whether the revealed entity in the continuation matches the entity established in the previous video.

All VLM-based evaluation uses Qwen3.5-Plus. The judge receives the previous video, prompts, generated continuation, task-specific questions, and any referenced historical frames. We use a compact JSON-format instruction:

VLM Judge

You are a video state evaluator. Watch the generated continuation together with the historical video and any referenced historical frames. Answer each checklist question with yes or no based only on visible evidence and the provided historical context. For appearance-sensitive questions, compare the revealed entity with the referenced historical frames. Return a JSON object with an answer and a brief reason for each question. Do not add extra text outside JSON.

## A.4 StateAgent Implementation Details

All VLM-based components in StateAgent use Qwen3.5-Plus. The image generation/editing module uses Wan2.7-Image, and the video renderer uses Wan2.2-KF2V-Flash. The source prompt p<sub>t</sub> is used to bind textual names to visual entities; state parsing is grounded in observed video evidence.

State graph fields. Entity nodes store identity and type, together with open-ended appearance description, visibility, location, physical condition, and concise state description inferred by the VLM from the observed video. Relation edges are also open-ended and may describe containment, location, attachment, surface/material state, quantity, or interaction. Each entity is assigned a clear reference frame from the previous video when available, so that future-frame grounding can preserve appearance.

Core prompts. We use structured prompts for the VLM modules. Below we report only the core logic of the prompts. Engineering-oriented instructions, such as exact output schemas, formatting constraints, retry rules, and implementation-specific checks, are omitted for readability and will be included in the released code.

Entity Binding

You are an entity extractor. Extract only entity names, entity types, and explicitly mentioned appearance attributes from the source prompt. Do not infer hidden state, future state, or task answers. Use the extracted entities only to bind textual names to visual entities observed in the video.

State Parsing

You are a video state observer. Analyze the observed video evidence and return the actual state of each entity at the current boundary. Report visible entities, hidden entities, locations, container open or closed states, physical conditions, and relations such as inside, holding, wearing, covered by, behind, or occluded by. The source prompt may help bind names to visual entities, but state parsing must follow what is observed in the video. If the prompt and video disagree, report the observed video state.

State Update

You are a video state predictor. Given the current world state and the next-shot prompt, predict the complete world state at the end of the next shot. Treat the prompt as an action condition. Update each entity’s visibility, location, physical condition, appearance description, concise state description, and open-ended relations such as containment, attachment, surface/material state, quantity, and agent-object interaction. Carefully distinguish physical existence from camera visibility: an entity can exist while remaining hidden from the current viewpoint.

Editing Prompt Writer

You are an expert at writing image-editing prompts. The last numbered image is the current frame to edit, and earlier numbered images are appearance references. Describe the expected result image in natural photographic language. State what should be visible after the action, refer to earlier images when matching entity appearance is needed, and preserve background, lighting, camera, and unchanged objects when the viewpoint does not change. If the prompt changes viewpoint, describe the full scene from the new viewpoint.

Reference selection. For future-frame grounding, StateAgent includes references only for entities involved in the predicted update or entities that are hidden at the boundary but should become visible in the continuation.

## A.5 Base Models and Generation Settings

Table 5 lists the model interfaces used in our experiments. All methods are evaluated under matched continuation prompts and comparable output settings within each model family. Default API or oficial implementation settings are used unless otherwise specified.

The StoryMem + full key-frame baseline uniformly samples key frames from the historical video, using six frames per five seconds while respecting the StoryMem memory bank limit of 10 input frames. It tests whether dense visual evidence alone is suficient for state update under the current continuation interface.

## B User Study

We also conduct a double-blind pairwise human evaluation. The study focuses on the generated continuation rather than the historical context. Each trial compares StateAgent with one baseline; the display position is randomized and method names are hidden from participants. The preceding historical video is shown as reference, and participants are instructed to judge only the final continuation segment.

We compare against four baselines: final-frame I2V, LongCat-Video, StoryMem, and MAGI-1. Each participant is randomly assigned 30 comparisons. We collect 15 complete submissions, resulting in 450 pairwise answers and 1,350 dimension-level judgments. For each comparison, participants choose which result is better, or whether the two are tied, along three dimensions: aesthetic quality, state reveal success, and state consistency.

We decode each preferred-side/tie response using the hidden display order. For each dimension-level judgment, StateAgent receives score 1 if it is preferred, 0.5 for a tie, and 0 if the baseline is preferred. We report the resulting score rate, raw win/tie rates, and win rate excluding ties. We also aggregate votes by scenario, baseline, and dimension to compute majority preference and majority agreement. Majority agreement is the fraction of votes assigned to the most selected option in each aggregated cell.

Table 6 shows that humans prefer StateAgent most strongly on the state-related dimensions. The gain is largest for state consistency, where StateAgent obtains an 85.6 score rate, an 81.3 raw win rate, and an 88.8 win rate after excluding ties. Majority agreement is also highest on state consistency, suggesting that participants form clearer consensus when judging whether the revealed state matches the historical event. Aesthetic quality shows a smaller margin and more ties, which is expected because it is more subjective and because several methods share similar visual generation priors.

Table 7 breaks down the results by baseline. The advantage is consistent across all comparisons, with the smallest margin against StoryMem and the largest margin against MAGI-1. This pattern matches the automatic evaluation: memory-based baselines can preserve some visual evidence, but participants still prefer StateAgent when the continuation must reveal or use the state established by the history.

## C Additional Results

## C.1 Additional StateBench Qualitative Results

Figures 6–13 provide additional qualitative comparisons on StateBench. Each result uses the same layout as Figure 4: the top row shows six uniformly sampled frames from the historical segment, and each method row shows three frames sampled from the generated final five seconds.

## D Limitation

One limitation is that StateAgent currently renders the continuation from the historical end frame to a predicted future end frame. When these two frames difer substantially, the generated video may contain shot transitions. Richer temporal constraints or intermediate state anchors may help address this limitation.

![](images/f85a062884aadd36d443be7e74f9711ce53b0e78d99ba40a2092795fd88d21e6.jpg)  
Figure 6 Additional qualitative comparison. History prompt: A single continuous realistic video on a kitchen counter. A man stands behind the counter with a slice of white bread and a toaster, all clearly visible. He places the white bread vertically into the toaster slot, then presses the toaster lever down; the bread fully enters the machine. Fixed camera, medium shot, bright indoor lighting. Continuation prompt: A single continuous realistic video on a kitchen counter, same toaster. Time is up, the toaster pops. The man takes the bread slice out and holds it up to the camera Fixed camera, medium shot, bright indoor lighting.

![](images/4ce485cd7ed110e771fc1f9e3f06e38b31c2083b359dad4a31ad3f4b20033548.jpg)  
Figure 7 Additional qualitative comparison. History prompt: A single continuous realistic video. The camera faces a fitting room with a curtain. A woman in a plain gray tank top holds up a hanger with a yellow floral dress to show it to the camera, then carries it into the fitting room and pulls the curtain closed. Fixed camera, medium shot, bright indoor lighting. Continuation prompt: A single continuous realistic video. The curtain of the fitting room is pulled open. She tried on her clothes and walked out. Fixed camera, medium shot, bright indoor lighting.

![](images/ec446d6cb41a9482999599a652009af34fb33fed87f03f2a2e492d25d967dad1.jpg)  
Figure 8 Additional qualitative comparison. History prompt: A single continuous realistic video in a gym hallway with a closed door. A blue water bottle and an orange towel rest on a bench. A man picks up the blue water bottle, leaving the towel, walks through the doorway into the locker room, and closes the door behind him. Fixed camera, medium shot, bright indoor lighting. Continuation prompt: A single continuous realistic video in a gym hallway. The door opens, revealing the man standing inside. Fixed camera, medium shot, bright indoor lighting.

History segment  
![](images/fce4a1b9c4cafcf4f2d0c148592f00a660fef892d26817f8771ea5cebe65afad.jpg)  
Figure 9 Additional qualitative comparison. History prompt: A single continuous realistic video on a table. A woman sits at the table with an empty transparent glass cup and an opaque teapot already filled with clear water, both clearly visible. She scoops a spoonful of red fruit-punch drink powder and drops it into the teapot; the liquid inside the opaque teapot stays hidden from the camera. Fixed camera, medium shot, bright indoor lighting. Continuation prompt: A single continuous realistic video on a table. The woman lifts the teapot and pours its liquid into the transparent glass cup. Fixed camera, medium shot, bright indoor lighting.

![](images/5c60d2fa7cf79d0e33fd4698ff1f30084e2a03855fdedb55007ea1734e3eb0f6.jpg)  
Figure 10 Additional qualitative comparison. History prompt: A single continuous realistic video on a table. A man stands at the table with a can of blue paint, a can of yellow paint, a sheet of white paper, a wooden stirring stick, and a completely opaque black mixing cup. He pours blue paint into the opaque cup, then pours yellow paint into the same cup, and stirs thoroughly with the stick until mixing is complete; the cup remains fully opaque. Fixed camera, medium shot, bright indoor lighting. Continuation prompt: A single continuous realistic video on the same table. The man pulls the stirring stick out of the opaque cup and smears the paint from the stick onto the white paper, revealing the mixed paint color on the paper. Fixed camera, medium shot, bright indoor lighting.

![](images/a58ab69b6e366a5d24e2b18a762e46a9a8b754a6b1ac20f36cec5fea400b456e.jpg)  
Figure 11 Additional qualitative comparison. History prompt: A single continuous realistic video, fixed camera inside a room facing a closed front door with a window beside it. Through the window, a woman in a denim jacket holding a grocery bag is seen walking up from outside toward the door. She steps up to the door and stops, now hidden behind the closed door and no longer visible through the window. Fixed camera, medium shot, bright indoor lighting. Continuation prompt: A single continuous realistic video, fixed camera inside the room. The front door opens. Fixed camera, medium shot, bright indoor lighting.

History segment  
![](images/30f2104548fded65f9d188e23dff487b9a811a8e2e18e5fc5335d08ee2d1f1fc.jpg)  
Figure 12 Additional qualitative comparison. History prompt: A single continuous realistic video on a table. A man stands behind the table with an open-top opaque cardboard box with tall side walls. He tilts the box toward the camera to show nine wrapped candies inside, then places the box upright so the camera cannot see inside. He then reaches in from above, takes out five candies, and places them on the table beside the box; the camera never sees the remaining candies. Fixed camera, medium shot, bright indoor lighting. Continuation prompt: A single continuous realistic video on the same table. The man lifts the open-top cardboard box and tilts it toward the camera to show exactly what remains inside the box. Fixed camera, medium shot, bright indoor lighting.

History segment  
![](images/faa33efe9b7b65a3472a060a5a82c641ae0cc92db7e2fb1142e255f675a2d125.jpg)  
Figure 13 Additional qualitative comparison. History prompt: A single continuous realistic video. On a wooden table, two opaque cups are placed upside down side by side. A man lifts the left cup, revealing a green glass marble underneath, then places it back. Then he slides the left cup in a semicircular path around the right cup to the far right; the cup slides on the table without leaving it, the right cup stays still. Fixed camera, medium shot, bright indoor lighting. Continuation prompt: A single continuous realistic video, same wooden table. The man lifts both the left and right cups simultaneously, showing what is under each. Fixed camera, medium shot, bright indoor lighting.