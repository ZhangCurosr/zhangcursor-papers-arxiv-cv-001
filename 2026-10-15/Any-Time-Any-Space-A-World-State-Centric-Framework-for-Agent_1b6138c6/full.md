# Any Time, Any Space: A World-State-Centric Framework for Agentic Video Generation

Weijie Wang<sup>1,∗</sup> Haomin Zhang<sup>1,∗</sup> Ye Li<sup>1,∗</sup> Akide Liu<sup>2</sup> Zeyu Zhang<sup>3</sup> Haoyu Zhao<sup>1</sup> Yefei He<sup>1</sup> Donny Y. Chen<sup>2</sup> Feng Chen<sup>4</sup> Bohan Zhuang<sup>1,†</sup>

<sup>1</sup> Zhejiang University <sup>2</sup> Monash University <sup>3</sup> BAIR, UC Berkeley <sup>4</sup> Adelaide University

Abstract. Long-form agentic video generation must turn a screenplay into shots that remain spatially and temporally consistent across nonconsecutive requests. Passing prompts or recent frames between shots does not provide a shared spatial frame for revisiting a scene or a time-indexed record for replaying an earlier event. We present Any Time, Any Space (ATAS), which compiles a screenplay into shot requests with explicit location, camera, and event-time references. Before generation, it retrieves relevant evidence from a scene memory of registered static geometry and an event memory of dynamic observations indexed by scene, event, and time. After generation, ATAS reconstructs and registers the clip. It updates memory only when review and state checks accept the observation, while film delivery remains independent of memory admission. Across four video generators, ATAS improves all reported cross-shot consistency measures and achieves the highest aggregate benchmark score for each backbone. It also outperforms the compared agents and 3D-guided methods on target-image spatial revisit, leads three event-recapture criteria, and receives the highest human ratings on all six criteria.

Date: October 13, 2026 Correspondence: bohan.zhuang@zju.edu.cn Project Page: http://lhmd.top/atas Code: https://github.com/ziplab/ATAS Keywords: Long Video Generation, Agentic Framework, Multi-shot Video Generation

## 1 Introduction

Agentic video generation translates a screenplay into a sequence of video shots by coordinating shot planning, reference selection, and video synthesis. As a story unfolds across locations, times, and viewpoints, its coherence depends on preserving the spatial and temporal relationships established in earlier shots. For example, a story may depict an event in a room, shift to another location, and later return to the room or replay the earlier event from a diferent viewpoint. The return shot should preserve the established scene layout, while the replay should reproduce the same event, including its actions and temporal progression, under the new camera view. These requirements extend beyond the visual quality of individual clips [1]: they require scenes and events to remain consistently accessible across intervening shots, so that later generations remain grounded in a persistent underlying world, as illustrated in Fig. 1.

Existing agentic video systems typically preserve continuity through recent frames, text summaries, or retrieved reference images. While efective for locally consecutive shots, these cues neither organize past observations in a shared spatial and temporal frame nor support non-local requests to revisit an earlier scene or event. This motivates a persistent world state, but maintaining such a state introduces two challenges. Past observations must remain addressable in both space and time, and the state must evolve without blindly absorbing generated content, whose visual, geometric, or temporal errors could otherwise propagate to later shots. A practical system must therefore resolve each request into a concrete shot specification, retrieve the relevant spatial and temporal evidence, and incorporate the resulting observation into the persistent world state only when it is suficiently reliable.

![](images/64bb5e28a1e7d082d4bae91365438e318cb408cd7c183e14bfc80aed773740da.jpg)  
Fig. 1 World-state-centric agentic video generation. ATAS treats a multi-shot film as a sequence whose spatial and temporal information can be reused. Shared scene geometry anchors consistent return views, while time-indexed event evidence supports replay from a new camera. The generated shots form the visible film, and the resulting scene and event information guides subsequent requests.

To address these challenges, we introduce Any Time, Any Space (ATAS), a world-state-centric framework for agentic video generation. ATAS maintains two complementary forms of persistent evidence: a scene state that preserves spatial structure in a shared coordinate frame, and an event state that retains dynamic observations with their temporal context. For each screenplay request, a world-state compiler grounds the requested shot in this world state, retrieves the relevant spatial and temporal evidence, and conditions a short-video generator on the resulting context.

Each generated clip is then treated as a candidate observation rather than trusted state. ATAS validates the reconstructed observation before incorporating it into the persistent world state, preventing unreliable generated evidence from propagating to subsequent shots. Importantly, whether a clip is suitable for the final video is separated from whether its observation is reliable enough to update the world state. This closed loop allows the world state to evolve across generation while remaining addressable for later spatial revisits and historical event recapture.

Our contributions are threefold:

• We introduce ATAS, a world-state-centric framework that organizes screenplay-driven video generation around persistent scene and event memories, enabling later shots to query evidence from earlier observations in any space and any time.

• We develop complementary spatially registered scene and time-indexed event memories, together with a validation-controlled update mechanism that separates film acceptance from the admission of new evidence into persistent state.

• Experiments against five agent baselines across four generators show that ATAS improves every reported cross-shot consistency measure and attains the highest aggregate score in each backbone block. It also outperforms the evaluated agent and 3D-guided methods on all three spatial-revisit metrics, leads three of five event-recapture criteria, and receives the highest mean human rating on all six criteria.

## 2 Related Work

Video Generation. Text-to-video and image-to-video models provide the renderer for long-video systems, improving appearance, motion, identity, and temporal coverage [2–11]. Long-video methods extend temporal context through co-denoising, distributed generation, or frame-level direction [12–14], while camera-, sparse-control-, and geometry-guided methods constrain view or motion [15–21]. These models improve local rendering and short-term control, but do not specify how observations from multiple clips are registered, retained, or queried in a later shot.

Agentic Video Generation. Agentic video systems add script parsing, shot planning, reference selection, and review to the generation loop [22–31]. Their inter-shot state is usually an editorial context of prompts, reference images, or generated histories. Such context can preserve local appearance without defining a shared coordinate frame, an event-time index, or a rule for admitting a generated observation into future state. Active view selection also supports spatial reasoning from 3D scene observations [32]. ATAS separates request compilation, memory retrieval, shot generation, and state update within the plan-act-observe loop of language agents [33].

Explicit Memory for Video Generation. Point-cloud and explicit-memory methods condition future generation on spatial evidence. Neural rendering and feed-forward reconstruction recover scene structure from sparse or multi-view observations [34–39], and 3D Gaussian Splatting provides an eficient explicit scene representation [40]. Multi-view compression improves reconstruction scalability [41]. Spatial-memory methods accumulate or query scene geometry [42–45]. Dynamic view-synthesis methods study camera-controlled reshooting [46–48]. World-generation methods explore geometry-consistent or navigable scene construction [49–53]. Related 3D generation research studies compact shape representations and eficient decoding [54, 55]. These methods provide geometric memory, dynamic view synthesis, or world assets, but do not expose the same shot-local observation layer, event-time addressing, and state-update decision interface. ATAS separates static scene memory from event-local dynamic evidence and evaluates an observation before incorporating it into persistent state.

## 3 Method

ATAS turns a high-level screenplay into short video shots while keeping the spatial and temporal evidence needed by later shots. The central object is a persistent world state, queried before each shot and updated by observations that pass validation.

## 3.1 Terms and Workflow

The terminology is as follows. A source observation $o ^ { \mathrm { s r c } }$ is the input frame or clip that starts a scene. Registering this observation establishes the scene coordinate frame. The observation and the coordinate frame are diferent objects. The scene memory $S _ { t }$ stores the accepted static geometry in that frame, while the event memory $\mathcal { E } _ { t }$ stores accepted dynamic observations together with their scene, event, and time information. The world state is simply the pair of memories at a particular step,

$$
\begin{array} { r } { \mathcal { W } _ { t } = ( \mathcal { S } _ { t } , \mathcal { E } _ { t } ) . } \end{array}\tag{1}
$$

![](images/a2aefae0c940327dcb18e1d7b93e20fe59dbaa12bc4689f2ff85e3541286da3d.jpg)  
Fig. 2 Workflow of ATAS. A high-level prompt is parsed into multi-shot requests and bounded shot requests. Each request contains a description, scene, spatial anchor, camera, time interval, and optional event. The shot prompt and the context-dependent prompt condition the video generator. Scene memory supplies static point-cloud references for the requested scene, while event memory supplies dynamic references for the requested event. Shot validation either requests another candidate or sends the accepted observation to the state update. The generated shot remains available as film independently of whether the observation is added to persistent state.

Equation 1 defines the persistent state used in the method. Scene and event refer to the contents of the two memories, while world state denotes their current pair after the first t requests.

The input is a high-level screenplay u, rather than a list of fully specified video prompts. At step t, the compiler produces a bounded shot request $q _ { t }$ , the memory query produces a conditioning context, and the generator produces a clip $V _ { t } .$ . The clip is then converted into a candidate observation and either admitted to the next state or retained as film without a state update:

$$
( u , \mathcal { W } _ { t } ) \xrightarrow { \mathrm { C o m p i l e } } q _ { t } \xrightarrow { \mathrm { Q u e r y } } C _ { t } \xrightarrow { G } V _ { t } \xrightarrow { \mathrm { R e c o n s t r u c t / R e g i s t e r / V a l i d a t e } } ( V _ { t } , \mathcal { W } _ { t + 1 } ) .\tag{2}
$$

The next request uses the resulting state. Fig. 2 summarizes the workflow.

## 3.2 World-state Compilation

Before generation, the compiler recursively splits the screenplay by scene, spatial and temporal relations, event references, and camera requirements until each request fits the short-video budget.

We write a renderable request as

$$
q _ { t , k } = ( d _ { t , k } , s _ { t , k } , \xi _ { t , k } , c _ { t , k } , \tau _ { t , k } , e _ { t , k } ) ,\tag{3}
$$

where $d _ { t , k }$ is the local shot description, $s _ { t , k }$ identifies the scene, $\xi _ { t , k }$ is the spatial anchor of the request in the scene coordinate frame, $\boldsymbol { c } _ { t , k }$ is the target camera, $\tau _ { t , k }$ is the requested time interval, and $e _ { t , k }$ is an optional event identifier.

For a screenplay segment, the compiler produces a request schedule

$$
\mathcal { Q } _ { t } = \operatorname { C o m p i l e } ( u , \mathcal { W } _ { t } ) = \{ q _ { t , 1 } , \cdot \cdot \cdot , q _ { t , K _ { t } } \} ,\tag{4}
$$

where $K _ { t }$ is the number of requests currently emitted. Requests are processed in order, and a validated $\mathcal { W } _ { t + 1 }$ conditions the next request. A rejected observation remains in the film but is not state evidence.

## 3.3 Two-Stage Memory: Scene and Event Memory

Scene memory. The registered source observation $o ^ { \mathrm { s r c } }$ establishes the scene coordinate frame and initializes $ { \boldsymbol { S } } _ { 0 }$ . Later accepted observations are registered to that frame and merged as static

point-cloud evidence with dynamic regions excluded. A request recovers a previously observed space by rendering these coordinates at its spatial anchor $\xi _ { t }$ and requested camera:

$$
C _ { t } ^ { \mathrm { s c e n e } } = \operatorname { R e n d e r } ( S _ { t } , c _ { t } ) ,\tag{5}
$$

where $C _ { t } ^ { \mathrm { s c e n e } }$ is the re-rendered scene reference supplied to the generator. Reconstruction happens only after generation, when the clip is considered for a future state update [20, 42, 43, 45].

Event memory. The event memory keeps dynamic evidence separate from static geometry. It is a set of records,

$$
\begin{array} { r } { \mathcal { E } _ { t } = \{ \epsilon _ { i } \} _ { i = 1 } ^ { N _ { t } } , \qquad \epsilon _ { i } = ( s _ { i } , e _ { i } , b _ { i } , \tau _ { i } , \mathcal { D } _ { i } ) , } \end{array}\tag{6}
$$

where $s _ { i }$ is the scene identifier, $e _ { i }$ is the event identifier, $b _ { i }$ identifies the source observation, $\tau _ { i }$ is the source time interval, and $\mathcal { D } _ { i }$ contains the registered frame-level RGB-D, mask, and dynamic point-cloud evidence. If a request refers to event $e _ { t }$ during interval $\tau _ { t }$ , the event context is retrieved directly from $\mathcal { E } _ { t } ^ { { \mathrm { ~ ~ } } }$

$$
C _ { t } ^ { \mathrm { e v e n t } } = \left\{ \begin{array} { l l } { \mathrm { R e t r i e v e } ( \mathcal { E } _ { t } , s _ { t } , e _ { t } , \tau _ { t } ) , } & { e _ { t } \neq \emptyset , } \\ { \emptyset , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

The result is empty when no compatible record exists. Equation $7$ is the event branch of the query that produces $C _ { t } ^ { \mathrm { e v e n t } }$ . The memory retains frame-local dynamic evidence rather than an identity-resolved motion model [47, 48]. Scene and event references are combined into the context-dependent prompt, while reconstruction remains a post-generation operation.

## 3.4 Shot Generation from Queried State

The local description in $q _ { t }$ forms the shot prompt, while the scene and event references form the context-dependent prompt shown in Fig. 2:

$$
C _ { t } = ( C _ { t } ^ { \mathrm { s c e n e } } , C _ { t } ^ { \mathrm { e v e n t } } ) , \qquad V _ { t } = G ( q _ { t } , C _ { t } ) ,\tag{8}
$$

where $G$ is the short-video generator. The shot prompt supplies local intent and its place and time, while $C _ { t }$ supplies scene and, when needed, event references. This interface is kept fixed across comparisons [15, 16, 56].

## 3.5 Observation Registration and State Update

The generated clip enters shot validation, which produces two separate decisions. Film review returns $f _ { t } = 1$ when the clip is accepted for delivery. If it is accepted, reconstruction and registration produce a candidate observation:

$$
\widetilde { \mathcal { O } } _ { t } = \mathrm { R e c o n s t r u c t } ( V _ { t } ) , \qquad \mathcal { O } _ { t } = \mathrm { R e g i s t e r } ( \widetilde { \mathcal { O } } _ { t } , \mathcal { S } _ { t } ) .\tag{9}
$$

The registration stage returns $r _ { t } = 1$ when the reconstruction aligns to the scene coordinate frame. For the source observation, a valid registration establishes that frame. If film review fails, the system retries or rejects the candidate. If registration fails, the clip remains usable as film but cannot update state. The decisions come from the respective film-review and registration stages [48, 57–59].

The state update check returns $a _ { t } = 1$ after the film and registration checks pass and the candidate has valid provenance, scene scope, and parent-state version. For a candidate that passes this check, the proposed next state is

$$
\widehat { \mathcal { W } } _ { t + 1 } = \mathrm { U p d a t e } ( \mathcal { W } _ { t } , \mathcal { O } _ { t } ) .\tag{10}
$$

![](images/310ded088cd708bea010f06678b7c2a4538f2d5fc942270a4506e04bfcc6eb4d.jpg)  
Fig. 3 Qualitative world-state rollout. Film frames and registered world views are shown for two story sequences. The examples illustrate how accepted observations extend the scene and event state across intervening shots.

The actual transition is

$$
\mathcal { W } _ { t + 1 } = \left\{ \begin{array} { l l } { \widehat { \mathcal { W } } _ { t + 1 } , } & { f _ { t } = r _ { t } = a _ { t } = 1 , } \\ { \mathcal { W } _ { t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{11}
$$

Equation 11 makes the decision rule explicit: $f _ { t }$ controls film delivery, $r _ { t }$ registration, and $a _ { t }$ admission to persistent state. Film and state update are independent outputs, so $V _ { t }$ remains in the film when $f _ { t } = 1$ even if $r _ { t } = 0$ or $a _ { t } = 0$ . In that case, the clip is unavailable as evidence for later requests.

## 3.6 Two Persistence Queries

The same workflow supports two queries. For spatial revisit, the compiler emits the earlier scene identifier and the requested return anchor and camera. The system renders $S _ { t }$ and compares the generated return image with the target image. For historical event recapture, the request includes an earlier event identifier and time interval. The system retrieves the matching records from $\mathcal { E } _ { t }$ combines them with the scene context, and generates the event from the requested camera. The first query tests recovery of static spatial evidence, and the second tests whether a time-local event remains addressable after intervening shots. The event protocol is source-grounded and does not assume a synchronized target video.

## 4 Experiments

## 4.1 Experimental Setup

Standard benchmarks assess complementary aspects of long-form video generation. We use the oficial inputs and evaluators for VBench [60], ST-Bench [61], and ViStory-Self and ViStory-Cross [62]. VBench measures aesthetics, subject and background consistency, temporal flicker, motion smoothness, and dynamic degree. ST-Bench evaluates aesthetics, global and single-shot prompt following, and cross-shot consistency with C-All and C-Top10. ViStory-Self measures identity, appearance, and temporal consistency within a story, while ViStory-Cross evaluates identity, appearance, and scene consistency across shots. Methods within each backbone block share the generator, prompt package, duration budget, and aggregation protocol. Per-metric scores appear in Appendix Tabs. A–D.

Tab. 1 Agent-level quantitative evaluation on standard benchmarks. Models are grouped by source type and rendering backbone. Methods within a block share the generator, prompt package, duration budget, and evaluation protocol. Bold and underlined entries mark the best and second-best values within each backbone block.
<table><tr><td>Model</td><td>Method</td><td>VBench</td><td>ST-Bench</td><td>ViStory- Self</td><td>ViStory- Cross</td><td>Avg.</td></tr><tr><td colspan="7">Open-Source Models</td></tr><tr><td rowspan="6">MINIMAX-H3</td><td>I2V Extension</td><td> $0 . 8 2 8 \pm 0 . 0 1 3$ </td><td> $0 . 5 7 2 \pm 0 . 0 2 0$ </td><td> $0 . 6 0 1 \pm 0 . 0 2 8$ </td><td> $0 . 4 1 8 \pm 0 . 0 2 7$ </td><td>0.605</td></tr><tr><td>VGoT</td><td> $0 . 8 3 4 \pm 0 . 0 1 4$ </td><td> $0 . 6 1 2 \pm 0 . 0 1 7$ </td><td> $0 . 6 3 8 \pm 0 . 0 2 3$ </td><td> $0 . 4 6 8 \pm 0 . 0 3 0$ </td><td>0.638</td></tr><tr><td>Mora</td><td> $0 . 8 3 6 \pm 0 . 0 1 2$ </td><td> $0 . 5 9 8 \pm 0 . 0 1 9$ </td><td> $0 . 6 2 8 \pm 0 . 0 2 6$ </td><td> $0 . 4 5 2 \pm 0 . 0 2 4$ </td><td>0.628</td></tr><tr><td>MovieAgent</td><td> $0 . 8 5 2 \pm 0 . 0 1 2$ </td><td> $0 . 6 0 8 \pm 0 . 0 1 6$ </td><td> $0 . 6 4 2 \pm 0 . 0 2 5$ </td><td>0.476 ± 0.023 0.644</td><td></td></tr><tr><td>ReCA</td><td> $0 . 8 4 4 \pm 0 . 0 1 1$ </td><td> $0 . 6 2 8 \pm 0 . 0 1 8$ </td><td> $\underline { { 0 . 6 5 8 \pm 0 . 0 2 1 } }$ </td><td> $0 . 5 0 8 \pm 0 . 0 2 2$ </td><td>0.659</td></tr><tr><td>ATAS</td><td> $\mathbf { 0 . 8 6 0 \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 6 9 1 \pm 0 . 0 1 4 }$ </td><td> $\mathbf { 0 . 7 1 0 \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 5 4 7 \pm 0 . 0 2 1 }$ </td><td>0.702</td></tr><tr><td colspan="7">Proprietary Models</td></tr><tr><td rowspan="6">WAn 2.7</td><td>I2V Extension</td><td> $0 . 7 7 8 \pm 0 . 0 1 4$ </td><td></td><td></td><td>0.486 ± 0.022 0.512 ± 0.029 0.348 ± 0.031 0.531</td><td></td></tr><tr><td>VGoT</td><td> $0 . 7 8 6 \pm 0 . 0 1 1$ </td><td> $0 . 5 3 6 \pm 0 . 0 1 9$ </td><td> $0 . 5 6 4 \pm 0 . 0 2 6$ </td><td>0.412± 0.028 0.575</td><td></td></tr><tr><td>Mora</td><td> $0 . 7 9 1 \pm 0 . 0 1 5$ </td><td> $0 . 5 2 8 \pm 0 . 0 1 7$ </td><td> $0 . 5 5 8 \pm 0 . 0 3 1$ </td><td>0.388 ± 0.033 0.566</td><td></td></tr><tr><td>MovieAgent</td><td> $0 . 8 0 8 \pm 0 . 0 1 3$ </td><td> $0 . 5 4 2 \pm 0 . 0 2 1$ </td><td> $0 . 5 7 2 \pm 0 . 0 2 4$ </td><td> $0 . 4 0 2 \pm 0 . 0 2 6$ </td><td>0.581</td></tr><tr><td>ReCA</td><td> $\mathbf { 0 . 8 5 4 \pm 0 . 0 0 9 }$ </td><td> $0 . 5 6 8 \pm 0 . 0 1 8$ </td><td> $\underline { { 0 . 5 9 8 \pm 0 . 0 2 2 } }$ </td><td> $\underline { { 0 . 4 2 8 \pm 0 . 0 2 9 } }$ </td><td>0.612</td></tr><tr><td>ATAS</td><td> $\underline { { 0 . 8 4 6 \pm 0 . 0 1 2 } }$ </td><td> $\mathbf { 0 . 6 2 5 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 6 6 6 \pm 0 . 0 2 1 }$ </td><td> $\mathbf { 0 . 4 8 6 \pm 0 . 0 3 2 }$ </td><td>0.656</td></tr><tr><td rowspan="6">HAPPYHORSE 1.0</td><td>I2V Extension</td><td> $0 . 8 3 6 \pm 0 . 0 1 2$ </td><td> $0 . 5 5 8 \pm 0 . 0 2 3$ </td><td> $0 . 5 9 2 \pm 0 . 0 2 9$ </td><td> $0 . 4 0 8 \pm 0 . 0 3 1$ </td><td>0.599</td></tr><tr><td>VGoT</td><td> $0 . 8 2 2 \pm 0 . 0 1 4$ </td><td> $0 . 5 8 8 \pm 0 . 0 1 8$ </td><td> $0 . 6 2 8 \pm 0 . 0 2 5$ </td><td> $0 . 4 5 8 \pm 0 . 0 2 7$ </td><td>0.624</td></tr><tr><td>Mora</td><td> $0 . 8 2 4 \pm 0 . 0 1 1$ </td><td> $0 . 5 7 4 \pm 0 . 0 2 0$ </td><td> $0 . 6 1 8 \pm 0 . 0 2 7$ </td><td> $0 . 4 4 2 \pm 0 . 0 2 6$ </td><td>0.614</td></tr><tr><td>MovieAgent</td><td> $0 . 8 5 6 \pm 0 . 0 1 3$ </td><td> $0 . 5 9 8 \pm 0 . 0 1 6$ </td><td>0.648 ± 0.021 0.472 ± 0.024 0.643</td><td></td><td></td></tr><tr><td>ReCA</td><td> $0 . 8 4 1 \pm 0 . 0 1 0$ </td><td> $\underline { { 0 . 6 2 1 \pm 0 . 0 1 9 } }$ </td><td> $\underline { { 0 . 6 6 8 \pm 0 . 0 2 4 } }$ </td><td> $0 . 4 9 6 \pm 0 . 0 2 5$ </td><td>0.656</td></tr><tr><td>ATAS</td><td> $\mathbf { 0 . 8 6 2 \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 6 6 8 \pm 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 7 0 6 \pm 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 5 4 0 \pm 0 . 0 3 0 }$ </td><td>0.694</td></tr><tr><td rowspan="6">WAN 3.0</td><td>I2V Extension</td><td> $0 . 8 3 6 \pm 0 . 0 1 3$ </td><td> $0 . 5 4 8 \pm 0 . 0 2 1$ </td><td> $0 . 5 8 2 \pm 0 . 0 2 7$ </td><td> $0 . 4 0 1 \pm 0 . 0 3 0$ </td><td>0.592</td></tr><tr><td>VGoT</td><td> $0 . 8 3 4 \pm 0 . 0 1 0$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Mora</td><td></td><td> $0 . 5 7 1 \pm 0 . 0 1 9$ </td><td> $0 . 6 2 2 \pm 0 . 0 2 4$ </td><td> $0 . 4 6 1 \pm 0 . 0 2 6$ </td><td>0.622 0.613</td></tr><tr><td>MovieAgent</td><td> $0 . 8 3 8 \pm 0 . 0 1 2$   $\underline { { 0 . 8 5 8 \pm 0 . 0 1 1 } }$ </td><td> $0 . 5 6 2 \pm 0 . 0 2 3$   $0 . 5 7 8 \pm 0 . 0 1 8$ </td><td> $0 . 6 1 4 \pm 0 . 0 2 2$   $0 . 6 2 8 \pm 0 . 0 2 5$ </td><td> $0 . 4 3 8 \pm 0 . 0 3 1$   $0 . 4 5 2 \pm 0 . 0 2 4$ </td><td>0.629</td></tr><tr><td>ReCA</td><td> $0 . 8 4 8 \pm 0 . 0 0 9$ </td><td> $0 . 6 0 2 \pm 0 . 0 1 7$ </td><td> $\underline { { 0 . 6 6 1 \pm 0 . 0 2 0 } }$ </td><td> $\underline { { 0 . 4 8 1 \pm 0 . 0 2 7 } }$ </td><td>0.648</td></tr><tr><td>ATAS</td><td> $\mathbf { 0 . 8 7 7 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 7 1 9 \pm 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 7 5 6 \pm 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 5 8 4 \pm 0 . 0 2 8 }$ </td><td>0.734</td></tr></table>

Spatial revisit and event recapture use task-specific evidence. Spatial revisit compares a generated return view with a target image under a common-visibility mask using ${ \mathrm { P S N R } } _ { \mathrm { C } }$ $\mathrm { S S I M _ { C } } ,$ , and LPIPS<sub>C</sub>. The comparison includes multi-shot agent-based and 3D-guided view-synthesis methods. Event recapture uses a source-grounded VLM judge given sampled source-event frames, an event description and interval, the requested camera, and the generated clip. Its rubric scores event identity, action and state, temporal order, character and object consistency, and viewpoint realization. Binary decisions map to zero or one, ordinal ratings from one to five are normalized to zero through one, and scores are averaged across axes and then across event requests. No synchronized target video is used.

Controlled analyses measure memory cost, component contributions, and human judgments. Memory ablation fixes the ATAS agent and Wan 3.0 while varying stored evidence. Component ablation adds scene and event resolution, recursive compilation, observation validation, and stateupdate checks under matched requests. Eficiency measurements report per-story runtime, real-time factor, candidate-review outcomes, retries per shot, retry-adjusted delivery time, throughput, and memory overhead across six memory settings. Further analyses are in Appendix Tabs. E–H and Figs. C–E. Human raters compare anonymized whole-video outputs from six methods on six criteria using a five-point scale.

(a) Closed-loop spatial revisit  
(b) Historical event recapture  
![](images/a803518449dccb287feecd17aad561c9aee765ee1df7fe363186fba4435bbe33.jpg)  
Fig. 4 Spatial revisit and historical event recapture. The top row uses a target image for spatial comparison. The bottom row uses source-grounded event evidence under a requested camera.

Tab. 2 Spatial revisit. Agent-based and 3Dguided methods are evaluated against target images under a common-visibility mask, with ATAS shown separately.
<table><tr><td>Method</td><td>PSNRc</td><td>SSIMc</td><td>LPIPSC</td></tr><tr><td>I2V Extension</td><td>13.601</td><td>0.293</td><td>0.339</td></tr><tr><td>VGoT</td><td>14.444</td><td>0.313</td><td>0.300</td></tr><tr><td>Mora</td><td>13.796</td><td>0.318</td><td>0.288</td></tr><tr><td>MovieAgent</td><td>14.543</td><td>0.352</td><td>0.296</td></tr><tr><td>ReCA</td><td>13.870</td><td>0.268</td><td>0.306</td></tr><tr><td>FlexWorld</td><td>14.740</td><td>0.343</td><td>0.315</td></tr><tr><td>ViewCrafter</td><td>15.255</td><td>0.375</td><td>0.293</td></tr><tr><td>ATAS</td><td>15.968</td><td>0.385 </td><td>0.284</td></tr></table>

Tab. 3 Source-grounded event recapture. Five criteria assess event identity, action and state, temporal order, character and object consistency, and viewpoint realization against source-event evidence and the requested camera. No synchronized target video is used. Scores are averaged across criteria and event requests.
<table><tr><td>Method</td><td>Event Action id.</td><td>state</td><td>Temporal order</td><td>Char.</td><td>View object realization</td></tr><tr><td>I2V Extension 0.450</td><td></td><td>0.433</td><td>0.433</td><td>0.817</td><td>0.200</td></tr><tr><td>VGoT</td><td>0.517</td><td>0.533</td><td>0.400</td><td>0.850</td><td>0.400</td></tr><tr><td>Mora</td><td>0.286</td><td>0.339</td><td>0.357</td><td>0.768</td><td>0.250</td></tr><tr><td>MovieAgent</td><td>0.500</td><td>0.450</td><td>0.500</td><td>0.783</td><td>0.350</td></tr><tr><td>ReCA</td><td>0.733</td><td>0.700</td><td>0.617</td><td>0.917</td><td>0.500</td></tr><tr><td>ATAS</td><td>0.750</td><td>0.688</td><td>0.688</td><td>0.917</td><td>0.650</td></tr></table>

## 4.2 Main Results

Qualitative rollouts reveal continuity across nonconsecutive shots. In both stories, the character returns to a previously visited place after intervening shots. The generated views retain evidence of the earlier scene layout and object state, while the reconstructed world views connect these returns to the spatial evidence accumulated during generation (Fig. 3). Further examples appear in Appendix Figs. I–K and Appendix G.

ATAS improves cross-shot consistency across generators while retaining competitive local quality. In Tab. 1, it leads every reported aggregate on MiniMax-H3, Wan 3.0, and HappyHorse 1.0. With Wan 2.7 it remains ahead on ST-Bench and both ViStory consistency measures despite trailing on VBench. This pattern across open-source and proprietary backbones suggests that the gain comes from persistent multi-shot state rather than a single renderer. We compare with I2V Extension, VGoT [24], Mora [63], MovieAgent [25], and ReCA [64] under matched backbone blocks.

ATAS improves both target-view recovery and source-grounded event recapture. The examples in Fig. 4 show return views and event replays from diferent cameras. ATAS reaches 15.968 PSNR , 0.385 SSIM<sub>C</sub>, and 0.284 LPIPS<sub>C</sub>, outperforming the compared multi-shot and 3D-guided methods (Tab. 2). In Tab. 3, it leads event identity (0.750), temporal order (0.688), and viewpoint realization (0.650), ties ReCA on character and object consistency (0.917), and remains slightly lower on action-state preservation (0.688 versus 0.700). The remaining dificulty is exact action-state reproduction rather than retrieval of the requested occurrence or view.

![](images/1e777dc099564a224ce3fd22047df7fe3258966cedb54276ebfb8f1260856b2a.jpg)  
Fig. 5 Human evaluation of agent methods. Grouped bars show mean ratings and standard-error whiskers for six methods across six criteria.

Tab. 4 Agent-component ablation. All rows use Wan 3.0 and the same requests, cameras, candidate budget, and retry policy. Components are added cumulatively from shot rendering through memory resolution, recursive compilation, observation validation, and state-update checking.
<table><tr><td colspan="3">ViStory</td><td colspan="3">Event Recapture</td></tr><tr><td>Configuration</td><td></td><td>average ↑</td><td>C-Top10 ↑</td><td>LPIPSc ↓</td><td>Score ↑</td></tr><tr><td>ATAS w/ shot rendering</td><td>0.478</td><td>0.528</td><td></td><td>0.341</td><td>0.441</td></tr><tr><td>ATAS w/ scene and event resolution</td><td>0.612</td><td>0.568</td><td></td><td>0.334</td><td>0.508</td></tr><tr><td>ATAS w/ recursive compilation</td><td>0.638</td><td></td><td>0.684</td><td>0.288</td><td>0.711</td></tr><tr><td>ATAS w/ observation validation</td><td>0.651</td><td>0.702</td><td></td><td>0.286</td><td>0.726</td></tr><tr><td>ATAS w/ state update check</td><td></td><td>0.670</td><td>0.719</td><td>0.284</td><td>0.739</td></tr></table>

Human raters give ATAS the highest mean score on all six criteria. Fig. 5 reports ratings for visual appeal, script faithfulness, character and background consistency, physical law, and narrative coherence under the anonymized protocol and criteria of Liu et al. [64]. Ratings are highest for script faithfulness (4.40), character consistency (4.22), and visual appeal (4.10), while background consistency (3.62) and narrative coherence (3.77) remain lower. Participants therefore most clearly perceive faithful shot content and character carryover. Environment-wide consistency and whole-story coherence remain less resolved. Further details appear in Appendix E.

Persistent memory reduces retry demand at a measurable compute cost. In Fig. 6, retries fall from 1.00 to 0.54 per shot and retry-adjusted delivery time declines from 359 to 339 seconds per shot. Per-story runtime rises from 51.9 to 64.9 minutes, memory overhead reaches 98 seconds per shot, and throughput falls from 7.28 to 6.21 shots per hour. The benefit is fewer retries and faster retry-adjusted delivery, rather than lower total compute.

Film-review rejection decreases as memory evidence accumulates. The candidate-lifecycle panel shows a monotonic decline from 52.6% with no memory to 36.8% with full state, with each intermediate setting following the same direction. This trend indicates that the improvement is not confined to the final configuration. It also measures whether candidates can be delivered, complementing scores that evaluate the selected outputs.

w/o memory (a) Runtime profile

![](images/92b5c9db24dc7ce415c2be5f536254e4e0a6ba5ad687e90abf72c05ed02e79e1.jpg)

![](images/0920e2d6b6f53be96f74e59d9e1474c233c78461f371e20c58138b7a2973ea6a.jpg)

![](images/e08b5f0e9166df72cbe856cb595255607bb65328ef8cdb0f17190a317ead674a.jpg)

![](images/2e0c41e53bda49f01c5857a6f25992d8582e6ab06c761bad47326eee88f9e150.jpg)

![](images/624fb0ba72525b7a6354cc17e3a6b8ac32801ac60b7598888f5097300f93cfc5.jpg)

![](images/5a3f4a372394432d5c46bba10abae258a0cbdc27eafdb1108af330950ecea067.jpg)

![](images/2e0d16dc4d06503e0d63f7a59da705d076035d6ed3157082a9a1d3e21a0f6642.jpg)  
Fig. 6 Eficiency comparison across memory settings. Panels (a), (b), and (c) show runtime, candidate lifecycle, and retry and delivery measures across six memory settings.

## 4.3 Ablation Study and Experimental Analysis

Scene and event memory improve diferent forms of continuity. Frame history improves all three measures over no memory. Adding static scene evidence lowers LPIPS<sub>C</sub> by 0.030, and updating that scene raises C-Top10 by 0.041 with little change in event score. Event memory gives the largest recapture gain with a smaller spatial change. Across the full ablation, C-Top10 rises from 0.536 to 0.719, event score from 0.452 to 0.739, and LPIPS falls from 0.341 to 0.284.

Tab. 5 Memory analysis. ATAS with Wan 3.0 is fixed. Rows vary stored evidence and state checks.
<table><tr><td>Config. C-Top10↑ LPIPSc ↓ Recap.↑</td></tr><tr><td>ATAS w/o memory</td><td>0.341 0.452</td></tr><tr><td>ATAS w/ frame history</td><td>0.536 0.598 0.319</td></tr><tr><td></td><td>0.548 0.617</td></tr><tr><td>ATAS w/ static scene ATAS w/</td><td>0.289 0.583</td></tr><tr><td>updated scene 0.658</td><td>0.286 0.585</td></tr><tr><td>ATAS w scene + event 0.703 ATAS w/ full state</td><td>0.285 0.728 0.719 0.284 0.739</td></tr></table>

The modest final increment from scene-plus-event to full state suggests that validation preserves the main capabilities rather than adding another retrieval efect (Tab. 5).

Scene and event memory establish gains that recursive compilation amplifies. Adding memory raises ViStory average from 0.478 to 0.612, C-Top10 from 0.528 to 0.568, and event score from 0.441 to 0.508. Recursive compilation then raises the event score to 0.711 and C-Top10 to 0.684 (Tab. 4).

Film delivery and state admission. Some clips are retained in the film even when registration or the state update is rejected. This lets the sequence continue without using an unaligned observation to condition later requests. The status breakdown separates these delivered clips from observations admitted to memory. Read with the runtime profile, it shows that persistent evidence reduces repeated review cycles while adding processing overhead, rather than making total generation faster.

## 5 Conclusion

We presented ATAS, which compiles screenplays into shot requests grounded in explicit scene, camera, and event-time references, then updates persistent state only from validated observations. Across four video generators, ATAS improves cross-shot consistency, spatial revisit, event recapture, and human ratings. The ablations show complementary contributions from scene memory, event memory, and recursive compilation, while the eficiency analysis shows fewer retries at added compute cost. By keeping locations and prior events addressable throughout a story, ATAS provides a foundation for agentic video tools in interactive filmmaking, episodic content creation, and previsualization.

## Contribution and Acknowledgements

Author contribution. Weijie Wang, Haomin Zhang, and Ye Li contributed equally to this work. Weijie Wang led the project and manuscript writing, contributed to code implementation, and conducted the initial experimental validation. Haomin Zhang implemented the code and contributed to experimental validation. Ye Li contributed to experimental validation and developed the demo and project page. Akide Liu contributed to the experiments and eficiency improvements. Zeyu Zhang, Haoyu Zhao, and Yefei He contributed to screenplay design and manuscript writing. Donny Y. Chen and Feng Chen contributed ideas to the project and guided the experimental design. Bohan Zhuang provided ongoing feedback throughout the project and participated in weekly project meetings.

Acknowledgements. We thank the participants in our human evaluation for their time and feedback. We also thank the researchers and developers who released the models, benchmarks, and open-source tools used in this work.

## References

[1] Yuke Zhao, Wangbo Zhao, Weijie Wang, Zeyu Zhang, Dakai An, Akide Liu, Yinghao Yu, Jiasheng Tang, Fan Wang, Wei Wang, and Bohan Zhuang. WorldOlympiad: Can Your World Model Survive a Triathlon? arXiv preprint arXiv:2606.11129, 2026. URL https://arxiv.org/abs/2606.11129.

[2] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video difusion models. In Neural Information Processing Systems, volume 35, pages 8633–8646. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2022. doi: 10.48550/arXiv.2204.03458. URL https://arxiv.org/abs/2204.03458.

[3] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data, 2022. URL https://arxiv.org/abs/2209.14792.

[4] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, et al. Imagen video: High definition video generation with difusion models, 2022. URL https://arxiv.org/abs/2210.02303.

[5] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In International Conference on Learning Representations, 2023. doi: 10.48550/arXiv.2205.15868. URL https://arxiv.org/abs/2205.15868.

[6] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, T. Michaeli, et al. Lumiere: A space-time difusion model for video generation, 2024. URL https://arxiv.org/abs/2401.12945.

[7] D. Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Y. Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. In International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 25105–25124. PMLR, 2024. doi: 10.48550/arXiv.2312.14125. URL https://arxiv.org/abs/2312.14125.

[8] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video difusion models with an expert transformer, 2024. URL https://arxiv.org/abs/2408.06072.

[9] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models, 2024. URL https://arxiv.org/abs/2410.13720.

[10] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video difusion models. In Computer Vision and Pattern Recognition, pages 7310–7320. IEEE, 2024. doi: 10.1109/CVPR52733.2024.00698. URL https://arxiv.org/abs/2401.09047.

[11] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent difusion models. In Computer Vision and Pattern Recognition, pages 22563–22575. IEEE, 2023. doi: 10.1109/CVPR52729.2023.02161. URL https://arxiv.org/abs/2304.08818.

[12] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023. URL https://arxiv.org/abs/2305.18264.

[13] Zhenxiong Tan, Xingyi Yang, Songhua Liu, and Xinchao Wang. Video-infinity: Distributed long video generation, 2024. URL https://arxiv.org/abs/2406.16260.

[14] Susung Hong, Junyoung Seo, Heeseong Shin, Sunghwan Hong, and Seungryong Kim. Direct2v: Large language models are frame-level directors for zero-shot text-to-video generation, 2023. URL https://arxiv.org/abs/2305.14330.

[15] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. StoryDifusion: Consistent self-attention for long-range image and video generation. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ c7138635035501eb71b0adf6ddc319d6-Abstract-Conference.html.

[16] Canyu Zhao, Mingyu Liu, Wen Wang, Weihua Chen, Fan Wang, Hao Chen, Bo Zhang, and Chunhua Shen. MovieDreamer: Hierarchical generation for coherent long visual sequence. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=QHj2LL958o.

[17] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image difusion models are zero-shot video generators. In IEEE International Conference on Computer Vision, pages 15908–15918. IEEE, 2023. doi: 10.1109/ICCV51070.2023.01462. URL https://arxiv.org/abs/2303.13439.

[18] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Gongye Liu, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video difusion priors. In European Conference on Computer Vision, 2024. doi: 10.48550/arXiv.2310.12190. URL https://arxiv.org/abs/2310.12190.

[19] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video difusion models. In European Conference on Computer Vision, pages 330–348, 2024. doi: 10.48550/arXiv.2311.16933. URL https://arxiv.org/abs/2311.16933.

[20] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video difusion and 3d representation for consistent world modeling. arXiv preprint arXiv:2507.07982, 2025. URL https://arxiv.org/abs/2507.07982.

[21] Weijie Wang, Xiaoxuan He, Youping Gu, Yifan Yang, Zeyu Zhang, Yefei He, Yanbo Ding, Xirui Hu, Donny Y. Chen, Zhiyuan He, Yuqing Yang, and Bohan Zhuang. World-R1: Reinforcing 3D Constraints for Text-to-Video Generation. arXiv preprint arXiv:2604.24764, 2026. URL https://arxiv.org/abs/2604.24764.

[22] Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. VideoStudio: Generating consistent-content and multi-scene videos. In European Conference on Computer Vision, 2024. URL https://arxiv.org/abs/2401.01256.

[23] Yunxin Li, Haoyuan Shi, Baotian Hu, Longyue Wang, Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang. Anim-Director: A large multimodal model powered agent for controllable animation video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. URL https://arxiv.org/abs/2408.09787.

[24] Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng Chen, Harry Yang, and Ser-Nam Lim. VideoGen-of-Thought: Step-by-step generating multi-shot video with minimal manual intervention. arXiv preprint arXiv:2412.02259, 2025. URL https://arxiv.org/abs/2412.02259.

[25] Weijia Wu, Zeyu Zhu, and Mike Zheng Shou. Automated movie generation via multi-agent CoT planning. arXiv preprint arXiv:2503.07314, 2025. URL https://arxiv.org/abs/2503.07314.

[26] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. VideoDirectorGPT: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091, 2023. doi: 10.48550/arXiv.2309.15091. URL https://arxiv.org/abs/2309.15091.

[27] Zhifei Xie, Daniel Tang, Dingwei Tan, Jacques Klein, Tegawend F. Bissyand, and Saad Ezzini. DreamFactory: Pioneering multi-scene long video generation with a multi-agent framework. arXiv preprint arXiv:2408.11788, 2024. doi: 10.48550/arXiv.2408.11788. URL https://arxiv.org/abs/2408.11788.

[28] Yuan Gong, Youxin Pang, Xiaodong Cun, Menghan Xia, Yingqing He, Haoxin Chen, Longyue Wang, Yong Zhang, Xintao Wang, Ying Shan, et al. Interactive story visualization with multiple characters. In SIGGRAPH Asia 2023 Conference Papers, pages 1–10. ACM, 2023. doi: 10.1145/3610548.3618184. URL https://arxiv.org/abs/2305.18247.

[29] Zhaochong An, Menglin Jia, Haonan Qiu, Zijian Zhou, Xiaoke Huang, Zhiheng Liu, Weiming Ren, Kumara Kahatapitiya, Ding Liu, Sen He, et al. OneStory: Coherent multi-shot video generation with adaptive memory. arXiv preprint arXiv:2512.07802, 2025. doi: 10.48550/arXiv.2512.07802. URL https://arxiv.org/abs/2512.07802.

[30] Jinsong Zhou, Yihua Du, Xinli Xu, Luozhou Wang, Zijie Zhuang, Yehang Zhang, Shuaibo Li, Xiaojun Hu, Bolan Su, and Ying-Cong Chen. VideoMemory: Toward consistent video generation via memory integration. arXiv preprint arXiv:2601.03655, 2026. doi: 10.48550/arXiv.2601.03655. URL https://arxiv.org/abs/2601.03655.

[31] Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, Jie Zhang, et al. Video-bench: Human-aligned video generation benchmark, 2025. URL https://arxiv.org/abs/2504.04907.

[32] Haoyu Zhao, Akide Liu, Zeyu Zhang, Weijie Wang, Feng Chen, Ruihan Zhu, Gholamreza Hafari, and Bohan Zhuang. CoV: Chain-of-View Prompting for Spatial Reasoning. arXiv preprint arXiv:2601.05172, 2026. URL https://arxiv.org/abs/2601.05172.

[33] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.03629. arXiv:2210.03629.

[34] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4578–4587, 2021. URL https://arxiv.org/abs/2012.02190.

[35] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In IEEE/CVF International Conference on Computer Vision, pages 14124–14133, 2021. URL https://arxiv.org/abs/2103.15595.

[36] Weijie Wang, Yeqing Chen, Zeyu Zhang, Hengyu Liu, Haoxiao Wang, Zhiyuan Feng, Wenkang Qin, Feng Chen, Jia-Wang Bian, Zheng Zhu, Donny Y. Chen, and Bohan Zhuang. VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction. arXiv preprint arXiv:2509.19297, 2025. URL https://arxiv.org/abs/2509.19297.

[37] Duochao Shi, Weijie Wang, Donny Y. Chen, Zeyu Zhang, Jia-Wang Bian, Bohan Zhuang, and Chunhua Shen. Revisiting Depth Representations for Feed-Forward 3D Gaussian Splatting. arXiv preprint arXiv:2506.05327, 2025. URL https://arxiv.org/abs/2506.05327.

[38] Weijie Wang, Zimu Li, Jinchuan Shi, Zeyu Zhang, Botao Ye, Marc Pollefeys, Donny Y. Chen, and Bohan Zhuang. TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction. arXiv preprint arXiv:2605.26115, 2026. URL https://arxiv.org/abs/2605.26115.

[39] Weijie Wang, Qihang Cao, Sensen Gao, Donny Y. Chen, Haofei Xu, Wenjing Bian, Songyou Peng, Tat-Jen Cham, Chuanxia Zheng, Andreas Geiger, Jianfei Cai, Jia-Wang Bian, and Bohan Zhuang. Feed-Forward 3D Scene Modeling: A Problem-Driven Perspective. arXiv preprint arXiv:2604.14025, 2026. URL https://arxiv.org/abs/2604.14025.

[40] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139–1, 2023. URL https://arxiv.org/abs/2308.04079.

[41] Weijie Wang, Donny Y. Chen, Zeyu Zhang, Duochao Shi, Akide Liu, and Bohan Zhuang. ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS. arXiv preprint arXiv:2505.23734, 2025. URL https://arxiv.org/abs/2505.23734.

[42] Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory. arXiv preprint arXiv:2512.15716, 2025. URL https://arxiv.org/abs/2512.15716.

[43] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. arXiv preprint arXiv:2506.05284, 2025. URL https://arxiv.org/abs/2506.05284.

[44] Weijie Wang, Haoyu Zhao, Yifan Yang, Feng Chen, Zeyu Zhang, Yefei He, Zicheng Duan, Donny Y. Chen, Yuqing Yang, and Bohan Zhuang. Latent spatial memory for video world models. arXiv preprint arXiv:2606.09828, 2026. URL https://arxiv.org/abs/2606.09828.

[45] Jiahao Wang, Luoxin Ye, TaiMing Lu, Junfei Xiao, Jiahan Zhang, Yuxiang Guo, Xijun Liu, Rama Chellappa, Cheng Peng, Alan Yuille, et al. Evoworld: Evolving panoramic world generation with explicit 3d memory. arXiv preprint arXiv:2510.01183, 2025. URL https://arxiv.org/abs/2510.01183.

[46] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video difusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. URL https://arxiv.org/abs/2409.02048.

[47] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6121–6132, 2025. URL https://arxiv.org/abs/2503.03751.

[48] Kuan Heng Lin, Zhizheng Liu, Pablo Salamanca, Yash Kant, Ryan Burgert, Yuancheng Xu, Koichi Namekata, Yiwei Zhao, Bolei Zhou, Micah Goldblum, Paul Debevec, and Ning Yu. Vista4D: Video reshooting with 4D point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 32671–32682, 2026. URL https://arxiv.org/abs/2604.21915.

[49] Yixiang Dai, Fan Jiang, Chiyu Wang, Mu Xu, and Yonggang Qi. Fantasyworld: Geometry-consistent world modeling via unified video and 3d prediction. arXiv preprint arXiv:2509.21657, 2025. URL https://arxiv.org/abs/2509.21657.

[50] Ziyue Zhu, Zhanqian Wu, Zhenxin Zhu, Lijun Zhou, Haiyang Sun, Bing Wan, Kun Ma, Guang Chen, Hangjun Ye, Jin Xie, et al. Worldsplat: Gaussian-centric feed-forward 4d scene generation for autonomous driving. arXiv preprint arXiv:2509.23402, 2025. URL https://arxiv.org/abs/2509.23402.

[51] Chenxi Song, Yanming Yang, Tong Zhao, Ruibo Li, and Chi Zhang. Taming video models for 3d and 4d generation via zero-shot camera control. arXiv preprint arXiv:2509.15130, 2025. URL https://arxiv.org/abs/2509.15130.

[52] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5916–5926, 2025. URL https://arxiv.org/abs/2406.09394.

[53] Team HY-World, Chenjie Cao, Xuhui Zuo, Zhenwei Wang, Yisu Zhang, Junta Wu, Zhenyang Liu, Yuning Gong, Yang Liu, Bo Yuan, Chao Zhang, et al. Hy-world 2.0: A multi-modal world model for reconstructing, generating, and simulating 3d worlds. arXiv preprint arXiv:2604.14268, 2026. URL https://arxiv.org/abs/2604.14268.

[54] Mingda Lin, Weijie Wang, Zeyu Zhang, Bowen Cui, Yefei He, Haoyu Zhao, Yuanyu He, Donny Y. Chen, Feng Chen, and Bohan Zhuang. ZipTok3D: High-Fidelity 3D Tokenization with Compact Token Prefixes. arXiv preprint arXiv:2609.01740, 2026. URL https://arxiv.org/abs/2609.01740.

[55] Bowen Cui, Weijie Wang, Zeyu Zhang, Yefei He, Mingda Lin, Haoyu Zhao, Yuanyu He, Donny Y. Chen, Feng Chen, and Bohan Zhuang. Block3D: Eficient Text-to-3D Generation via Block-Wise Difusion. arXiv preprint arXiv:2608.19567, 2026. URL https://arxiv.org/abs/2608.19567.

[56] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In Computer Vision and Pattern Recognition, pages 2568–2577, 2025. doi: 10.1109/CVPR52734.2025.00245. URL https://arxiv.org/abs/2403.14773.

[57] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(4):376–380, 1991. doi: 10.1109/34.88573. URL https://doi.org/10.1109/34.88573.

[58] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651, 2025. URL https://arxiv.org/abs/2503.11651.

[59] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10510–10522, 2025. URL https://arxiv.org/abs/2501.12387.

[60] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. VBench: Comprehensive benchmark suite for video generative models. In Computer Vision and Pattern Recognition, 2024. doi: 10.1109/CVPR52733.2024.02060. URL https://arxiv.org/abs/2311.17982.

[61] Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory. arXiv preprint arXiv:2512.19539, 2025. URL https://arxiv.org/abs/2512.19539.

[62] Cailin Zhuang, Ailin Huang, Wei Cheng, Jingwei Wu, Yaoqi Hu, Jiaqi Liao, Zhewei Huang, Hongyuan Wang, Xinyao Liao, Weiwei Cai, et al. ViStoryBench: Comprehensive benchmark suite for story visualization. 2025. doi: 10.48550/arXiv.2505.24862. URL https://arxiv.org/abs/2505.24862. arXiv:2505.24862.

[63] Zhengqing Yuan, Ruoxi Chen, Zhaoxu Li, Weixiang Sun, Haolong Jia, Lifang He, Chi Wang, and Lichao Sun. Mora: Enabling generalist video generation via a multi-agent framework. arXiv preprint arXiv:2403.13248, 2024. doi: 10.48550/arXiv.2403.13248. URL https://arxiv.org/abs/2403.13248.

[64] Akide Liu, Jinbo Xing, Chaojie Mao, Ye Li, Zeyu Zhang, Yefei He, Weijie Wang, Zihan Wang, Yu Liu, Gholamreza Hafari, and Bohan Zhuang. ReCA: Multi-shot long video extrapolation via recursive context allocation. arXiv preprint arXiv:2605.26525, 2026. URL https://arxiv.org/abs/2605.26525.

## Appendix Contents

A Evaluation Protocols and Benchmark Details . . . . . . . . . . . . . . . . 18   
A.1 Source-grounded Diagnostic Evaluator . . . . . . . . . . . . . . . . . . . . . . . . . . . 18   
A.2 VBench Dimensions . . . . . . . . . . . . 19   
A.3 ST-Bench Metrics . . . . . . . . . . . . . 19   
A.4 ViStory Metrics . . . . . . . . . . . . . . . . . . . . 20   
B World-state compiler and bounded shot requests . . . . . . . . . . . . . . 20   
C Memory Construction and Query . . . . . . . . . . . . . . . 21   
D Sensitivity and Mechanism Analysis . . . . . . . . . . . . . . . 22   
E Human Evaluation. . . . . . . . . . . . . . . 23   
F Qualitative Results . . . . . . . . . . . . . . 25   
G Long-Horizon Result Gallery . . . . . . . . . . . 27   
G.1 Saturday Laundry in the Flat . . . . . . . . . . . . 27   
G.2 Winter Fire in the Cabin . . . . . . . . . . . . . . . . 31   
G.3 Running a Bath on a Winter Evening . . . . . . . . . . . . 34

This appendix provides implementation details and supporting analyses for reproducing and interpreting ATAS. It covers screenplay compilation, scene and event memory, benchmark evaluation, controlled analyses, human judgments, and extended story rollouts.

## A Evaluation Protocols and Benchmark Details

For each standard benchmark, the aggregate score is accompanied by metric-level details rather than story- or sequence-level listings. All benchmark-specific tables use the same backbone and method grouping as Tab. 1. The entries are interpreted with the evaluator definitions given in the corresponding subsection.

These breakdowns serve two purposes. First, they show which aspects of generation change when the rendering backbone changes, so that a single aggregate cannot be mistaken for a universal improvement. Second, they separate local rendering quality from the cross-shot behavior that the persistent state is intended to support. The dimensions are therefore read within a benchmark and within a backbone block. They are not combined into a new training objective or used to tune the reported agent.

## A.1 Source-grounded Diagnostic Evaluator

The source-grounded diagnostic evaluator uses one fixed VLM judge and one versioned rubric for every method. The evaluator receives sampled source-event frames, the event description and interval, the target-camera request, an optional target-camera memory render, and sampled frames from the generated clip. It does not receive a synchronized target video. The memory render is treated as a requested-view condition rather than ground truth.

For each axis, the judge returns either a binary decision or an ordinal rating from 1 to 5. We map a binary decision to 0 or 1 and map an ordinal rating r to (r − 1)/4. The normalized score for axis a is the mean over the rubric items for that axis. The Event Recapture Score is the arithmetic mean of the five normalized axes in Tab. 3: event identity, action and state preservation, temporal phase and order, character and object consistency, and viewpoint realization. We first average within each event request and then average across requests so that longer clips do not receive more weight.

The same rubric is used for all methods and memory configurations. A request without sourceevent evidence is excluded before aggregation. A request without a generated clip receives zero on the applicable axes. Human judgments on matched clips are used as a separate agreement analysis rather than as hidden calibration data.

The five axes intentionally distinguish what is recovered from the source event. Event identity asks whether the requested occurrence is the one being reproduced. Action and state evaluate what the participants and objects are doing. Temporal phase and order evaluate where the recapture lies within the source interval. Character and object consistency check persistent entities, and viewpoint realization checks the requested camera relation. A high score on one axis therefore does not imply a high score on the others. A clip can show the correct characters while depicting the wrong phase of the event, or preserve the action while failing to realize the requested view. This decomposition is why the aggregate is reported together with the per-axis values in Tab. 3.

The evaluator is source-grounded rather than pixel-grounded. It can judge whether the generated clip agrees with the supplied event evidence and camera request, but it cannot establish pixel-level agreement with an unobserved target video. The resulting score is appropriate for historical event access and view-conditioned recapture. It should not be read as a substitute for synchronized reconstruction when such a target is available.

Tab. A VBench dimension breakdown. Each row is one reported VBench dimension, and each method/backbone column uses the oficial VBench evaluator and prompt suite. Higher is better unless the oficial dimension defines the opposite direction.
<table><tr><td>Dimension</td><td>Wan 2.7 ATAS</td><td>Wan 3.0 ATAS</td><td>MiniMax-H3 ATAS</td><td>HappyHorse ATAS</td></tr><tr><td>Aesthetic quality</td><td>0.6387</td><td>0.6731</td><td>0.6945</td><td>0.6792</td></tr><tr><td>Subject consistency</td><td>0.9481</td><td>0.9643</td><td>0.9576</td><td>0.9418</td></tr><tr><td>Background consistency</td><td>0.9554</td><td>0.9717</td><td>0.9680</td><td>0.9522</td></tr><tr><td>Temporal flickering</td><td>0.9932</td><td>0.9863</td><td>0.9847</td><td>0.9884</td></tr><tr><td>Motion smoothness</td><td>0.9888</td><td>0.9766</td><td>0.9714</td><td>0.9795</td></tr><tr><td>Dynamic degree</td><td>0.5488</td><td>0.6882</td><td>0.5842</td><td>0.6314</td></tr></table>

Tab. B ST-Bench metric breakdown. The table reports oficial scores for aesthetic quality, global and single-shot prompt following, all-pair cross-shot consistency, and top-ten-pair cross-shot consistency.
<table><tr><td>Metric</td><td>Wan 2.7 ATAS</td><td>Wan 3.0 ATAS</td><td>MiniMax-H3 ATAS</td><td>HappyHorse ATAS</td><td>ReCA reference</td></tr><tr><td>Aesthetic</td><td>0.6387</td><td>0.6731</td><td>0.6945</td><td>0.6792</td><td>0.6517</td></tr><tr><td>PF-G</td><td>0.2174</td><td>0.2415</td><td>0.2351</td><td>0.2289</td><td>0.2216</td></tr><tr><td>PF-S</td><td>0.2154</td><td>0.2362</td><td>0.2308</td><td>0.2243</td><td>0.2192</td></tr><tr><td>C-All</td><td>0.5977</td><td>0.6844</td><td>0.6529</td><td>0.6615</td><td>0.5751</td></tr><tr><td>C-Top10</td><td>0.6253</td><td>0.7187</td><td>0.6908</td><td>0.6684</td><td>0.6015</td></tr></table>

## A.2 VBench Dimensions

Tab. A shows that the rendering backbones trade diferent intrinsic properties. Wan 3.0 is strongest on subject and background consistency and dynamic degree, while Wan 2.7 is stronger on flicker and motion smoothness and MiniMax-H3 is strongest on aesthetic quality. The spread is largest for dynamic degree (0.5488–0.6882), whereas the temporal-flickering and motion-smoothness values remain high for every backbone. Thus the advantage in the standard multi-shot results cannot be explained by one uniformly better renderer: the backbones begin with diferent local strengths. This breakdown also gives a useful control for the later analyses. A change in cross-shot consistency should be interpreted alongside these local properties, rather than as evidence that ATAS improves every frame-level dimension equally. The table therefore separates backbone-dependent rendering behavior from the cross-shot gains measured by ST-Bench and ViStory.

## A.3 ST-Bench Metrics

Tab. B is consistent with the aggregate comparison: the strongest ATAS columns are the cross-shot consistency measures, especially C-Top10. Prompt-following changes are smaller and more uniform than the consistency changes: for example, Wan 3.0 reaches 0.2415 on PF-G and 0.2362 on PF-S, while its C-All and C-Top10 values are 0.6844 and 0.7187. The gap between C-Top10 and C-All is positive for every ATAS backbone (0.0069–0.0379), so the two consistency summaries provide complementary checks rather than duplicate columns. Wan 3.0 gives the strongest overall cross-shot behavior, with MiniMax-H3 and HappyHorse retaining substantial gains over the reference column. The pattern is therefore more specific than a general prompt-following gain: it is consistent with state being queried at shot boundaries and helping preserve information across the sequence.

Tab. C ViStory-Self metric breakdown. Rows report the supplied within-story consistency dimensions and aggregate.
<table><tr><td>Metric</td><td>Wan 2.7 ATAS</td><td>Wan 3.0 ATAS</td><td>MiniMax-H3 ATAS</td><td>HappyHorse ATAS</td><td>ReCA reference</td></tr><tr><td>Identity consistency</td><td>0.5494</td><td>0.6617</td><td>0.6421</td><td>0.5974</td><td>0.5608</td></tr><tr><td>Appearance consistency</td><td>0.6836</td><td>0.8139</td><td>0.7476</td><td>0.7772</td><td>0.7034</td></tr><tr><td>Temporal self-consistency</td><td>0.7648</td><td>0.7908</td><td>0.7392</td><td>0.7425</td><td>0.7186</td></tr><tr><td>Aggregate</td><td>0.6659</td><td>0.7555</td><td>0.7096</td><td>0.7057</td><td>0.6609</td></tr></table>

Tab. D ViStory-Cross metric breakdown. Rows report the supplied cross-shot consistency dimensions and aggregate.
<table><tr><td>Metric</td><td>Wan 2.7 ATAS</td><td>Wan 3.0 ATAS</td><td>MiniMax-H3 ATAS</td><td>HappyHorse ATAS</td><td>ReCA reference</td></tr><tr><td>Cross-shot identity</td><td>0.5073</td><td>0.5992</td><td>0.5736</td><td>0.5428</td><td>0.4934</td></tr><tr><td>Cross-shot appearance</td><td>0.4871</td><td>0.5806</td><td>0.5392</td><td>0.5618</td><td>0.5013</td></tr><tr><td>Cross-shot scene consistency</td><td>0.4621</td><td>0.5729</td><td>0.5285</td><td>0.5138</td><td>0.4484</td></tr><tr><td>Aggregate</td><td>0.4855</td><td>0.5842</td><td>0.5471</td><td>0.5395</td><td>0.4810</td></tr></table>

## A.4 ViStory Metrics

Tab. C shows that the gains are not confined to one identity score. Wan 3.0 has the strongest appearance and temporal self-consistency and the highest aggregate, while the other ATAS backbones remain above the reference aggregate. Temporal self-consistency is the highest-scoring dimension for Wan 2.7, whereas appearance consistency is highest for the other three ATAS columns. Identity and temporal consistency move in the same direction, with Wan 3.0 reaching 0.6617, 0.8139, and 0.7908, respectively. A method that merely improves texture or color would not be expected to improve all three within-story dimensions together. The result indicates that the queried state supports both within-story identity and appearance continuity, while leaving room for future improvements in identity preservation.

Tab. D follows the same ordering as the main cross-shot results. Wan 3.0 leads identity, appearance, and scene consistency, and every ATAS backbone exceeds the reference aggregate in this table. Scene consistency is the most demanding of the three columns for each ATAS backbone, which is expected because it depends on preserving the shared environment rather than only the subject. The relative ordering is stable across identity, appearance, and scene consistency. No backbone obtains its aggregate advantage from a single isolated submetric. These dimensions provide a direct check that the behavior persists across diferent shots rather than only within individual clips.

## B World-state compiler and bounded shot requests

The world-state compiler receives a high-level screenplay request and the current scene and event evidence. It recursively resolves scene transitions, temporal relations, historical-event references, and camera requirements until each bounded shot request fits the short-video generation budget. Each request provides a local shot description, an explicit scene coordinate and target camera, and a requested time interval with optional event information. The compiler does not render a video while an internal request is being expanded. It emits a bounded shot request and waits for its validated observation before deciding which state a subsequent request may use.

Fig. A gives a concrete compiler trace from screenplay request to bounded shot requests and validated state updates.

![](images/22ac9c7dadaf080ac94cdf704625452f7a27ad2f22532d5853c72f3df2e1f365.jpg)  
Fig. A Recursive screenplay compilation. The compiler turns a high-level request into bounded shot requests and uses validated state updates to condition subsequent requests.

The trace makes the unit of generation explicit. The screenplay is resolved into requests that carry a local description together with scene, camera, time, and optional event fields. These fields determine which evidence can be queried before generation and which evidence can be attached to the resulting observation afterward. A request that changes location can therefore establish a new scene relation, while a request that revisits an earlier location can reuse its scene coordinates. An event request additionally carries the time interval needed to select dynamic evidence. The figure checks that decomposition precedes rendering and that a later request consumes the validated state produced by an earlier one.

## C Memory Construction and Query

Each generated clip is retained as a shot-local observation with its request, source provenance, timestamps, scene and event identifiers, target camera, reconstruction artifacts, and independent film, registration, and state-update statuses. The scene memory stores registered static evidence after dynamic regions are excluded. The event memory stores time-local dynamic evidence bound to a source shot, an event identifier, and a source interval. The two memories are queried separately and combined when a request refers to a historical event.

Fig. B connects each shot-local observation to the static and event query paths.

The separation is useful because the two evidence types have diferent persistence requirements. Static geometry is registered once to a scene coordinate frame and can be rendered at a new camera, whereas dynamic evidence must remain bound to an event and a time interval. Merging them into a single undiferentiated history would make a visually similar frame eligible even when it belongs to the wrong occurrence. The figure also clarifies the direction of information flow: reconstruction and registration create a candidate observation after generation, while rendering and event retrieval create conditioning references before the next generation. This distinction is central to interpreting the controlled memory rows in Tab. 5.

![](images/8c4a70361e6b5d8e1679d47864ac32eb438749503f205f83bf1ebe37b57d5399.jpg)  
Fig. B Scene and event memory. The figure follows one accepted shot-local observation as static evidence is written to scene memory and dynamic evidence is indexed in event memory, then shows how both references condition a later camera request.

Tab. E Memory statistics. This analysis table reports memory growth and query cost at rollout checkpoints. It is not a sample manifest.
<table><tr><td>Checkpoint</td><td>Scene size (×106)</td><td>Event records</td><td>Storage (GB)</td><td>Query and update time (s)</td><td>Processed shots</td></tr><tr><td>Initial state</td><td>33.54</td><td>1.0</td><td>0.95</td><td>93.0</td><td>1.2</td></tr><tr><td>After accepted update</td><td>40.92</td><td>2.1</td><td>1.38</td><td>97.8</td><td>2.8</td></tr><tr><td>End of rollout</td><td>49.60</td><td>4.8</td><td>2.44</td><td>103.5</td><td>6.9</td></tr></table>

Tab. E shows increasing scene and event coverage over the rollout while query and update time changes only modestly. From the initial state to the end of the rollout, scene size grows from 33.54 to 49.60 million units and event records from 1.0 to 4.8, while query and update time rises from 93.0 to 103.5. Storage increases from 0.95 to 2.44 in the reported units, so the state is not free. The cost grows more slowly than the amount of processed evidence in this short rollout. The processed-shot count also increases from 1.2 to 6.9, which makes the checkpoint comparison a measure of accumulation rather than a comparison of independent runs. These values should be read as an operating profile for the reported configuration, not as a general scaling law. They support treating the state as a growing evidence store whose cost is measured separately from the video generator.

## D Sensitivity and Mechanism Analysis

The sensitivity analysis keeps the prompt package, camera requests, generator, candidate budget, and retry policy fixed while varying memory capacity, update frequency, memory freshness, and state update check strictness. The figures diagnose how state quality changes. They do not introduce a second benchmark.

Tab. F shows that sparse updates and reduced scene capacity lower both cross-shot consistency and event quality, while shortening the event history mainly afects the event score. Relative to full update, sparse update reduces C-Top10 from 0.719 to 0.685 and the event score from 0.739 to 0.732. Reducing capacity has a larger spatial efect (C-Top10 0.638 and LPIPS<sub>C</sub> 0.301), while shortening event history reduces the event score to 0.608 with almost unchanged LPIPS<sub>C</sub>. Spatial quality therefore changes less than event quality when the event memory is shortened, which is consistent with the separation between static scene evidence and time-local event evidence. The comparison also suggests that update frequency and memory capacity are distinct controls: more frequent updates cannot fully compensate for a substantially smaller scene store, and a large scene store cannot replace missing event history.

![](images/94310ccff1ee669bf9df0a5f0973227210d025db3dc33d3434c56db2989b7b5e.jpg)  
Capacity: scene-memory size (×10<sup>6</sup>); k: update interval in shots. Full update is the default ATAS setting  
Fig. C Memory capacity and update sensitivity. The figure shows whether quality gains require additional state size or frequent updates.

Tab. F Memory capacity and update sensitivity. The table supplies the values used by Fig. C.
<table><tr><td>Configuration</td><td>Capacity  $( \times 1 0 ^ { 6 } )$ </td><td>Update interval</td><td>C-Top10</td><td> $\mathrm { L P I P S _ { C } }$ </td><td>Event score</td></tr><tr><td>Full update</td><td>49.60</td><td>1</td><td>0.719</td><td>0.284</td><td>0.739</td></tr><tr><td>Sparse update</td><td>49.60</td><td>3</td><td>0.685</td><td>0.286</td><td>0.732</td></tr><tr><td>Small capacity</td><td>12.00</td><td>1</td><td>0.638</td><td>0.301</td><td>0.701</td></tr><tr><td>Short event history</td><td>49.60</td><td>1</td><td>0.712</td><td>0.284</td><td>0.608</td></tr></table>

Tab. G shows that quality declines as more shots are processed after the last accepted update. C-Top10 decreases from 0.736 at zero intervening shots to 0.651 after five or more, while LPIPS increases from 0.276 to 0.306. The event score follows the same direction, from 0.762 to 0.678. The gradual decline across all three measures is more informative than any single endpoint. It suggests that stale state gradually loses usefulness rather than at one fixed horizon. These bins do not identify whether the loss comes from scene changes, event density, or accumulated registration error. They motivate the separate state-update analysis below.

Fig. E visualizes the quality and state-coverage trade-of of the state update check.

Tab. H shows the quality-coverage trade-of of the state update check. A permissive setting admits 90.0% of candidate updates but produces lower downstream quality (C-Top10 0.696 and event score 0.708), whereas a strict setting admits 36.0% and also loses quality because fewer observations remain available (0.668 and 0.682). The default setting accepts 67.4% and obtains the strongest downstream scores, 0.719 and 0.739, together with the lowest LPIPS<sub>C</sub> of the three settings. This result does not imply that the default threshold is universally optimal. It shows that both indiscriminate admission and excessive rejection can reduce the evidence available to future requests. The check is therefore useful as a controlled admission mechanism, not as a replacement for improving reconstruction or retrieval.

## E Human Evaluation

The human study follows the grouped-bar comparison in Fig. 5. Raters view anonymized clips in randomized order and score visual appeal, script faithfulness, character consistency, background consistency, physical law, and narrative coherence on a fixed five-point scale. The grouped bars report the six criteria, while this section reports score distributions and inter-rater agreement.

Shots since last accepted update

![](images/97d8db1218e323883e5b919efa6a7640a405ce2f8b0901287df8766412e10511.jpg)  
Shots since last accepted update

![](images/adc446dc248b348093c08765dcda7867a1fccdae988a95f46e4811dbf3fe35c3.jpg)  
Shots since last accepted update

![](images/fbe10870bd732cfe47e3b102edf6439662ae3fa20c1ddcf0e6e852d2580cb83e.jpg)  
Fig. D Memory freshness and staleness. The figure tests whether quality degrades as future requests operate farther from the most recent accepted observation.

![](images/fd899327034dbc5abde7918ba10d40484ea557f01f958a0999f6c427cfcb1b0f.jpg)  
Fig. E State update check trade-of. The figure shows how stricter admission changes state coverage and downstream quality.

Tab. G Memory freshness data. The table supplies the values used by Fig. D.
<table><tr><td>Shots since update</td><td>C-Top10  $\mathrm { L P I P S _ { C } }$ </td><td></td><td>Event Recapture Score</td></tr><tr><td>0</td><td>0.736</td><td>0.276</td><td>0.762</td></tr><tr><td>1 to 2</td><td>0.722</td><td>0.283</td><td>0.744</td></tr><tr><td>3 to 4</td><td>0.691</td><td>0.292</td><td>0.712</td></tr><tr><td>5+</td><td>0.651</td><td>0.306</td><td>0.678</td></tr></table>

Fig. F shows the rating interface. Each video is rated as a whole rather than shot by shot, so the consistency criteria reflect the complete film.

Fig. G shows score spread, and Fig. H compares the automatic diagnostics with human judgments.

The human-rating heatmaps show a consistent method ordering across criteria. ATAS has the highest mean rating in all six rows, with means from 3.62 to 4.40. ReCA is the strongest baseline, with means from 2.62 to 3.67. The fraction of ratings at least four points makes the same distinction: ATAS ranges from 57% to 95%, whereas ReCA ranges from 14% to 64%. The agreement coeficients shown beside the heatmap range from 0.58 to 0.72, so the comparison is not based on a single criterion with unusually unstable ratings. These values support the qualitative conclusion that the benefit is visible to raters. The spread across criteria shows that narrative coherence and physical-law judgments remain harder than script faithfulness and character consistency.

![](images/c98904ec56b1da2a776a91bd3e9b098761a97209c020409b20757f854062d340.jpg)  
Fig. F Human evaluation interface. Each page shows one anonymized long video with its story title and synopsis. Raters watch the whole video and score it on the six criteria from 1 (very poor) to 5 (excellent). The method identity is hidden, and all six criteria must be rated before moving to the next video.

Tab. H State update check sensitivity. The table supplies the values used by Fig. E. It reports no independent benchmark.
<table><tr><td rowspan="2">State update setting</td><td rowspan="2">State update rate</td><td colspan="3"></td></tr><tr><td>C-Top10</td><td>LPIPSC</td><td>Event score</td></tr><tr><td>Permissive</td><td>90.0</td><td>0.696</td><td>0.293</td><td>0.708</td></tr><tr><td>Default</td><td>67.4</td><td>0.719</td><td>0.284</td><td>0.739</td></tr><tr><td>Strict</td><td>36.0</td><td>0.668</td><td>0.288</td><td>0.682</td></tr></table>

The agreement plot provides a second check on the automatic diagnostics. Correlation with human ratings is 0.61, 0.54, 0.49, 0.41, and 0.57 for event identity, action and state, temporal phase, character and object, and viewpoint, respectively. The negated spatial distance, −LPIPS<sub>C</sub>, correlates at 0.44; negation aligns its direction with higher human ratings. The corresponding human–human ceilings are higher (0.67–0.80), leaving room for evaluator noise and for aspects of perception that are not captured by the automatic scores. The result supports using the automatic metrics as structured diagnostics rather than treating them as complete substitutes for human judgment.

## F Qualitative Results

Figures I–K provide the qualitative audit, rollout timeline, and multi-story grid.

The baseline comparison highlights diferences in shot-to-shot continuity. In the record-shop and bakery sequences, methods vary in whether room layout, object placement, and character state remain consistent after intervening shots. The ATAS row provides a reference for this behavior. A clip can remain plausible locally while drifting in layout or entity state, which is not captured by a single global similarity score. The rollout timeline shows when a scene state is created, when a later request queries it, and when a returned observation is written back. The six-story grid shows that these behaviors span diferent settings, people, objects, and camera patterns.

![](images/e14a00dcbba1cf55e65124653be18296c90a38bd6bd367361e80da9844fc60d2.jpg)

(a) Mean rating (1–5)  
![](images/06d3cd828e2c7be0f2e4198b0c083e1cdeded5a80d7f2da9beadb448ca5dbcf5.jpg)

(b) Rated ≥ 4 (%)  
![](images/997d2838233c5f67f289793a3d42f4b1479c587f4f5da689f45865d0298992cd.jpg)  
Fig. G Distribution of human judgments. The heatmaps report mean ratings and the fraction of ratings at least four for each criterion and method. The side panel summarizes inter-rater agreement.

![](images/8c80858b1de9c094378ad1ba61acf172f5d70c4a62e2ebcd3373baa989970146.jpg)  
Fig. H Automatic-to-human agreement. Spearman correlations compare the event-recapture and spatialrevisit diagnostics with human ratings on matched clips, alongside the human–human agreement ceiling.  
(a) Record shop  
Shot 2  
Shot 5  
Shot 7  
Shot 10  
(b) Bakery  
Shot 1  
Shot 3  
Shot 7  
Shot 9  
Fig. I Qualitative baseline audit. Matched frames from the record-shop and bakery stories compare methods across shot continuation, scene layout, object state, and character consistency.

![](images/d6d6422275d720bf8be1aab44f1941865b5e8d81e7c9a88b66f628e56f2f7113.jpg)  
Fig. J Representative rollout timeline. The timeline follows the generated film and memory condition through scene changes, a spatial revisit, and an event recapture request.

![](images/eab19da4721c5fe06c4220a81d98e3a19ee4156b65dcb80fb5063416435d51db.jpg)  
Fig. K Representative multi-story results. The grid shows timestamped frames from six stories, illustrating the range of scenes, actions, and camera requests used in the qualitative evaluation.

## G Long-Horizon Result Gallery

The following three full-length ATAS films each visit three locations before returning to two of them. We examine whether the later views retain layouts, objects, and changes established before the intervening scenes. Each sequence includes an overview, five individual visits, and first-visit/return pairs. Frames come directly from the generated films, with timestamps in minutes and seconds. These examples show generated continuity rather than physical simulation of unobserved intervals.

## G.1 Saturday Laundry in the Flat

A young man does his weekend laundry. The 1:50 film follows him from the living room to the laundry room and the bedroom, and then back to the living room and the laundry room. The story leaves two pieces of state behind: a mug set down on the cofee table, and a washer that is loaded and started before he leaves the laundry room.

Visit 1: living room (0:00–0:15). The film opens on a quiet Saturday morning. He sits on the dark blue sofa with a mug of cofee, facing the TV, with a gray throw over the arm of the sofa and a low wooden cofee table in front of him. Before leaving, he sets the mug down on the table. This is the first observation of the living room, and the room is reconstructed from it together with the position of the mug.

![](images/17f7b09d9513797f24004af66383d13fec01b106a925683a3d532078c872b03d.jpg)  
0:12

![](images/041273ce67f5eac4363506c3aa24946922c87df9a699ec6d04f0207ab271cbf8.jpg)  
0:27

![](images/642c249e982a9418a24506479bd51d6bd3c302cf2eee7bdd3abdb79e891e16c8.jpg)  
0:32

![](images/633124dee70df65bb9ddf58551974c7850999e894caa7f4dfcd9418064071010.jpg)

![](images/25ef55bd43ce2ac05c8c719cb30b0c075ecaaa1d23e0fb6f93c6f4e9d9cdb3c3.jpg)  
0:57

0:50  
![](images/2a3c80dcbc3e01683ecec85855042f2f6dbe8d19c53256c0015ccbe89cd594b1.jpg)  
1:05

![](images/951fd749c517224991ab2b6a8c30142815e5b2bc8ecefc159dae90303aa5c7b0.jpg)  
1:32

![](images/568cf0d40918b35714ce90c28d5bc8f14619ce27aea4f6282f2e5afc0747f1ee.jpg)  
1:37

Fig. G.1 Saturday Laundry in the Flat: overview. He leaves the living room for the laundry room, adds detergent and closes the washer, sits and then lies down on the bed, opens the curtains, and later gets up from the sofa to walk back to the laundry room.  
![](images/5e4923ed55a88f825d13bca9e51ec94d378b244b7b94be5925559f118e5c5679.jpg)  
Fig. G.2 Saturday Laundry, visit 1: living room (0:02). The first view of the living room, with the sofa, the gray throw, the cofee table, and the TV unit.

![](images/fc3a45d156699be19de28a599b5f901236496ac1a2d42d6e8683db3292b7b1e0.jpg)  
Fig. G.3 Saturday Laundry, visit 2: laundry room (0:22). He loads the clothes from the basket into the washer.

![](images/b790e59bb44c2d3b4d2d18a162b62a7dce3a6791f057f4c503db4e54528c989f.jpg)  
Fig. G.4 Saturday Laundry, visit 3: bedroom (1:07). He sits in the armchair after opening the curtains.

![](images/ab1ae20dfe3b60d3bbafd0451c003d3746d86923649dabac54c311c7001faec2.jpg)  
Fig. G.5 Saturday Laundry, visit 4: living room, return (1:25). Back on the sofa, with the mug still on the cofee table where he set it down.

Visit 2: laundry room (0:15–0:45). He carries the basket into the small tiled laundry room, loads the clothes into the front-loading washer, adds detergent from the blue bottle on the shelf, and starts the machine. The shelf with the detergent, the plant, and the jar, the towel on the wall, and the folding drying rack by the door are established here. When he leaves, the washer is running and the basket is empty.

Visit 3: bedroom (0:45–1:15). While the wash runs, he goes to the bedroom, lies down on the unmade bed, gets up to open the curtains, and sits in the green armchair by the window. The bedroom is seen only once in the film, so this visit only adds a new room to the world state. It also separates the two returns from the rooms in which the state was left.

![](images/c6cd8b6db042d1f012f6b0dcf119f311a06b9f9b1872106d8c168a3569e2f29d.jpg)  
Fig. G.6 Saturday Laundry, visit 5: laundry room, return (1:45). He opens the washer he started more than a minute earlier.

![](images/c4e9f63776331f3e1a2c884229cfdab5fd4b537459662558b64106f5c1dfd14b.jpg)  
Living room, first visit · 0:10

![](images/345455bf82afca479d3e03a17d331d39157ebc8f79b68b1feb4858bf290aef1b.jpg)  
Living room, return · 1:27

![](images/2cfcc3e6159523c5fb8726725daaa5f80e89491668cc62ef33784c9a874d64b1.jpg)  
Laundry room, first visit · 0:42

![](images/dc4174d6a04862bf44099d10618355ccfc19eaaf965959458c7358924a75eded.jpg)  
Laundry room, return · 1:40  
Fig. G.7 Saturday Laundry in the Flat: returns. Top: the mug set down on the cofee table at the start is still there more than a minute later, next to the same sofa and throw. Bottom: the laundry room keeps its shelf, towel, basket, and drying rack, and the washer he started is still shut with the laundry inside.

Visit 4: living room, return (1:15–1:40). He returns to the living room about a minute after leaving it. The stored living room conditions this visit, so the sofa, the throw, the cofee table, and the TV unit reappear in the same arrangement, and the mug he left behind is still standing on the table when he sits down again.

Visit 5: laundry room, return (1:40–1:50). He goes back to collect the wash. The laundry room is rendered from its stored state after the wash was started: the shelf, the towel, the basket, and the drying rack are where they were, and the washer is still shut with the laundry inside. He opens the door of the finished machine.

![](images/76ed19f840b88be40d5d76397979f8d825b2c05b54380e07526a218c3a517e24.jpg)  
Fig. G.8 Winter Fire in the Cabin: overview. She builds the fire from the log basket and watches it catch, stands by the frosted kitchen window with the heart she drew, rests in the bedroom, and returns to tend the fire, which has grown while she was away.

![](images/897181fe8ef8719b44cb3e6e9efd1d89c10ce015a98c3c3bbac7617edd8b04e7.jpg)  
Fig. G.9 Winter Fire, visit 1: living room (0:12). She lights the fire in the stone hearth.

## G.2 Winter Fire in the Cabin

A woman lights the fire in a cold log cabin. The 2:10 film moves from the living room to the kitchen and the bedroom, and then back through the kitchen to the living room. Both returns test state that she created herself: a heart drawn in the frost on the kitchen window, and a fire lit in the hearth before she leaves the living room.

Visit 1: living room (0:00–0:30). She enters a cold living room with a stone fireplace, a log basket on the hearth, and a brown sofa with a red plaid throw. She kneels at the hearth, builds the fire from the logs in the basket, and lights it. When she leaves, a small fire is burning in a hearth that was cold at the start of the film.

![](images/07c9a51fb75b4b9cd9ae3a5000e6b907e6dbd6cb885a40e949332257cce13184.jpg)  
Fig. G.10 Winter Fire, visit 2: kitchen (0:35). She draws in the frost on the kitchen window.

![](images/bad29ee8c114fd199947821c95e348d2f3fcfdf84458eb6644433ce7b9d2e106.jpg)  
Fig. G.11 Winter Fire, visit 3: bedroom (1:10). She lies on the bed under the lamp.

Visit 2: kitchen (0:30–0:55). In the kitchen, frost covers the window above the counter. She draws a heart in the frost with her finger, then turns to the jar and the oil lamp on the counter beside the dresser of blue-and-white plates. The heart is a small, specific mark that the kitchen must still carry when she comes back.

Visit 3: bedroom (0:55–1:25). She rests in the bedroom, sitting on and then lying across the bed with the patchwork quilt, lit by the bedside lamp. The bedroom is seen only once. It adds a third room to the cabin and places half a minute of other content between the kitchen and its return.

![](images/2a124dc4c983169bde5f0995acfe654af2cc031450269fdf53212957ae96db16.jpg)  
Fig. G.12 Winter Fire, visit 4: kitchen, return (1:30). The kitchen as she left it, with the heart still in the frost on the window.

![](images/f407f5e5dcca3df0ec73e2f78a1d88cd60337b304011322febd672ac136e27f1.jpg)  
Fig. G.13 Winter Fire, visit 5: living room, return (2:05). She sits by the fire she lit before leaving the room.

Visit 4: kitchen, return (1:25–1:40). On her way back she passes through the kitchen. The stored kitchen conditions this shot: the stove, the counter with the oil lamp, the dresser, and the doorways are in the same places, and the heart she drew is still visible in the lower right pane of the frosted window.

Visit 5: living room, return (1:40–2:10). She returns to the living room about seventy seconds after lighting the fire. The fireplace, the log basket, the lamp, and the sofa with the red plaid throw match the first visit, and the fire she lit has kept burning and now fills the hearth. She settles into the armchair under a plaid blanket.

![](images/336d3e44a182bacade8e29f12f052ce74cc34614116a0896b5ae6db5d2077f0c.jpg)  
Kitchen, first visit · 0:40

![](images/52efd5458e92e286fa12109ddc75dbc2ecd57df35ea682baf0c61ec93f3fbe96.jpg)  
Kitchen, return · 1:27

![](images/0c85252c4649e11af5142a2e3a800e873f74452252993e9d2b1ee29323809e5f.jpg)  
Living room, first visit · 0:00

![](images/60539ab27528d0c3ced1722ceb895cbcbd13591628d9de02e7dd175b1dea7577.jpg)  
Living room, return · 1:42  
Fig. G.14 Winter Fire in the Cabin: returns. Top: the heart she drew in the frost is still in the lower right pane of the same kitchen window. Bottom: the living room keeps its stone fireplace, log basket, lamp, and sofa with the red plaid throw, while the hearth that was cold at the start now holds the fire she lit.

![](images/9ba57c952a01c310bb921a9bcff0814f11abde7f461b93f55938846dffa53b97.jpg)  
0:12

![](images/dfa39f7aaac0271b6acd9f69bb3614b4870342697ee09e65c334399211a623a8.jpg)  
0:22

![](images/8cd0fcf1ceefc1d89e755e12f364f9ab0c4aa5276b84774193648d66852e2c0e.jpg)  
0:37

![](images/f7f1b1b60f858fb920a5fc92112b79e4270de5e2ccc3b7efdc2d48e99d51e05c.jpg)  
0:47

![](images/4dbb0504042061ee29252c8a49580ca4e050c2d3578d79f9d1435886342f8c3f.jpg)  
0:57

![](images/cd88db71389d9dbddcba85dde27abb4043ef6c14c15b635c6850634a4c07afa9.jpg)  
1:22

![](images/b27ca188980992ac5b0cc5293615c04133fd44971f1f5c611189224390443ce1.jpg)  
1:37

![](images/4b318a77c93f2a1c570f6c17e016a94f02532b568614486264d383a96db7706e.jpg)  
1:52  
Fig. G.15 Running a Bath on a Winter Evening: overview. She leaves the laptop on the bed, turns on the tap, lights a candle as the tub fills with pink water, browses the bookshelf in the living room, opens the wardrobe back in the bedroom, and tests the water in the filled tub.

## G.3 Running a Bath on a Winter Evening

A young woman, tired from working on her laptop, runs herself a bath. The 2:10 film moves from the bedroom to the bathroom and the living room, and then back to the bedroom and the bathroom. The two returns test a static object left behind, the closed laptop on the bed, and a process that continues while she is away, the bath filling from the running tap.

![](images/1b14faecdc187bd3ce83d50190408cad2ce55c61db5161c4569bd4d78c2b9e79.jpg)  
Fig. G.16 Running a Bath, visit 1: bedroom (0:07). She closes the laptop on the bed.

![](images/69de3b20dcefa3a9f089ae66f834eb10507f26e6268f35bd8a48d881e36d79b3.jpg)  
Fig. G.17 Running a Bath, visit 2: bathroom (0:30). The tub begins to fill with pink water.

Visit 1: bedroom (0:00–0:15). She sits cross-legged on the bed in a warmly lit bedroom with pink curtains, a white wardrobe, and a lamp on the nightstand, working on her laptop. She closes the laptop, leaves it on the bed, and walks out through the door to the bathroom.

Visit 2: bathroom (0:15–0:45). In the tiled bathroom she turns on the tap over the empty tub, pours in bath salts so that the water turns pink, and lights a candle on the glass shelf below the round mirror. She leaves with the tap running and the tub only partly filled.

![](images/34f9af7efaa0708e12119f0ee320a3a95c97f0c40eb7b08199a6d4b936767107.jpg)  
Fig. G.18 Running a Bath, visit 3: living room (1:07). She rests on the sofa while the bath runs.

![](images/ed7d2d330bbdfd9a1d0c75cbb015c7380fe42c639afcd799d5a76d181f059389.jpg)  
Fig. G.19 Running a Bath, visit 4: bedroom, return (1:30). She sits next to the laptop she left on the bed.

Visit 3: living room (0:45–1:15). While the bath runs, she waits in the living room, looking out over the city lights, browsing the bookshelf, and resting on the sofa beside a small candle. The living room is seen only once, and it separates both returns from the rooms in which the state was left.

Visit 4: bedroom, return (1:15–1:40). She comes back to the bedroom a minute after leaving it. From the same view as the opening shot, the bed, the nightstand lamp, the wardrobe, and the pink curtains match the first visit, and the laptop she closed is still lying on the bed. She sits beside it before heading to the bath.

![](images/9d1046365d6ee791aec3d7f43350f8798a21bdecbbc620362a35f529bbf83a91.jpg)  
Fig. G.20 Running a Bath, visit 5: bathroom, return (2:05). She kneels by the tub, which has filled while she was away.

![](images/f832c50f8aa499cfd2625bc51604f123aa9ec98ed03b53ffb510bf5ff4bd31db.jpg)  
Bedroom, first visit · 0:00

![](images/2db197ff3fdbbfd3f2613f8d9482a69d533b4d78f43847d5a0444a70ba7bec9c.jpg)  
Bedroom, return · 1:15

![](images/a78056cec7578cc3e223f3a78c7e08206e17c4c7c79be991c4f6f28a5eeccecb.jpg)  
Bathroom, first visit · 0:15

![](images/0092eb305497206318879f8dd4859596ea712652370a022d856da80510ea22c9.jpg)  
Bathroom, return · 1:42  
Fig. G.21 Running a Bath on a Winter Evening: returns. Top: from the same view, the bedroom matches the first visit, with the bed, nightstand lamp, wardrobe, and pink curtains, and the laptop she closed is still lying on the bed. Bottom: the tiles, glass shelf, mirror, and towel rail match her first entrance, while the empty tub has filled with pink water and the candle she lit is still burning.

Visit 5: bathroom, return (1:40–2:10). Back in the bathroom, the tiles, the glass shelf, the mirror, and the towel rail match her first entrance. The tub that was only partly filled when she left is now full of pink water, and the candle she lit is still burning. She kneels by the tub with a towel.