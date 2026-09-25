# HelloWorld: Towards Practical Applications of Generative Driving World Models

HelloWorld Team

Driving world models provide a promising route toward scalable counterfactual data generation and interactive simulation beyond recorded driving logs. Realizing this potential requires a system that can generalize across diverse scenes, respond faithfully to prescribed controls, generate coherent multi-sensor observations, and operate efficiently under repeated inference. We present HelloWorld, a 2B driving world model system designed around these requirements. HelloWorld progressively specializes broad visual and motion priors from heterogeneous video data into controllable driving generation using ego pose, HD maps, and 3D boxes. A block-causal generation interface, together with adaptation to self-generated context, aligns the model with sequential simulation. The system further supports synchronized seven-camera RGB generation and conditional LiDAR synthesis, and is distilled toward few-step inference for efficient deployment. Experiments evaluate visual quality, control fidelity, cross-view consistency, robustness under repeated generation, inference efficiency, and LiDAR synthesis. Together, HelloWorld provides a unified framework for scalable driving data generation and interactive simulation.

Website: https://helloworld-4d.github.io

## 1 Introduction

Driving logs provide rich observations of the physical world, but they are inherently limited to trajectories and traffic configurations that were actually encountered. They cannot directly reveal how the same scene would evolve under a different ego trajectory, how observations would change under an alternative traffic arrangement, or what rare situation might look like before sufficient real-world examples have been collected. Generative driving world models offer a complementary capability: they can synthesize counterfactual observations beyond recorded data for scalable data generation and, more broadly, provide generative environments for interactive driving simulation. Realizing this potential requires moving beyond isolated video generation toward a system that can generalize across diverse scenes, respond faithfully to prescribed controls, produce coherent observations of the evolving world, and operate efficiently enough for repeated interaction.

Turning such a generative model into a practical driving world model is therefore not a matter of improving visual fidelity alone. The model must represent sufficiently broad driving scenarios, expose interfaces through which the simulated world can be intentionally altered, and support repeated generation under the information flow available during simulation. It must also produce the observations required by downstream driving systems at a tractable computational cost. We view these requirements jointly as a system-level problem, encompassing generalization, controllability, multi-sensor observation generation, and efficient inference, with temporal causality providing the generation interface that connects them to sequential simulation.

![](images/7170b560aa85df234ff6a332aff89abc7f7e6c5ba80748b83072999ce459f70c.jpg)  
Figure 1. Overview of HelloWorld. Top: A block-causal RGB generator is progressively specialized from poseconditioned single-view modeling to synchronized seven-view synthesis and geometry-aware scene conditioning with ego pose, HD maps, and 3D boxes. The causal observation interface is retained across all stages. Bottom: A 20-step causal teacher is distilled into a four-step student, a decoupled super-resolution module enhances spatial fidelity, and an RGB-conditioned LiDAR branch synthesizes range and return intensity from synchronized seven-view observations.

To address this problem, we present HelloWorld, a 2B driving world model system for scalable data generation and interactive simulation. HelloWorld is built around a progressive learning strategy that first acquires broad visual and motion priors from heterogeneous video data and then specializes these priors into controllable driving dynamics using increasingly structured supervision. Explicit trajectory and scene conditions allow the model to synthesize observations beyond recorded driving logs, while causal generation and adaptation to self-generated context align its temporal behavior with sequential simulation. The framework further extends the generated world state across synchronized camera views and geometric sensing, and employs few-step generation to reduce the cost of repeated inference. Together, these designs turn a general video prior into a controllable and efficient generative model of driving observations.

A central challenge is to combine broad visual knowledge with the structured supervision required for controllable driving generation. Training only on richly annotated driving data limits scene diversity, whereas general video provides much broader appearance, motion, and interaction patterns but lacks the geometric and semantic interfaces needed for driving simulation. HelloWorld therefore adopts a progressive specialization strategy: it first learns pose-conditioned visual dynamics from heterogeneous general and fleet video, while accounting for their different pose-scale conventions, and subsequently introduces increasingly structured driving supervision. Calibrated multi-view data establish consistent surround observations, while HD maps and 3D boxes provide explicit scene-level conditions. Together with egopose control, these structured signals allow the model to move beyond replaying recorded trajectories and generate counterfactual observations under altered ego motion and scene configurations. This progression forms the upper path of Figure 1, while the lower path connects the resulting RGB model to few-step generation and conditional LiDAR synthesis.

Once the model can be actively controlled, its temporal formulation must also match the way it is used during simulation. In an interactive setting, future observations are unavailable: each new observation must be generated from the history produced so far together with the supplied conditions. We therefore adopt block-causal observation modeling from the beginning of driving adaptation, generating frames jointly within each block while restricting later blocks to previously available observations. This causal interface aligns the model with sequential simulation, but it does not eliminate the discrepancy between training on recorded histories and inference on model-generated ones. We therefore further expose the model to imperfect context through context corruption and self rollout. These mechanisms improve robustness to the self-generated histories encountered during repeated simulation.

With the spatial controls and temporal interface established, HelloWorld further broadens the observation space produced at each simulation step. Calibrated cross-view communication enables synchronized seven-camera generation that reflects a shared scene state across viewpoints, while a conditional LiDAR branch extends the system from appearance synthesis to geometric sensing. The LiDAR component uses geometry-aware tokenization and cross-modal interaction with RGB features to preserve valid returns and metric structure without forcing the two modalities into a shared native representation. Finally, because both large-scale data synthesis and interactive simulation repeatedly invoke the generative model, inference cost becomes part of the system design rather than a standalone acceleration problem. We therefore distill HelloWorld toward few-step generation using consistency training and distribution matching, reducing repeated sampling cost while preserving the same control and sequential-generation interfaces.

Taken together, these components form a unified driving world model system that connects broad data priors, explicit control, sequential generation, multi-sensor observation synthesis, and efficient inference. Rather than treating these capabilities as independent extensions, HelloWorld organizes them around a common objective: generating controllable driving observations beyond recorded trajectories and scene configurations, while retaining the temporal and computational properties required for repeated simulation.

Our contributions are summarized as follows:

• We present HelloWorld, a scalable and controllable driving world model system for data generation and interactive simulation, integrating broad video priors with structured driving supervision, synchronized multi-view generation, multi-sensor observation synthesis, and efficient inference.

• We develop a progressive specialization strategy that transfers general visual and motion knowledge into controllable driving generation. Ego pose, HD maps, and 3D boxes provide complementary interfaces for modifying camera motion and scene configuration, enabling counterfactual synthesis beyond recorded driving logs.

• We introduce a temporally causal generation interface together with adaptation to self-generated context, reducing the gap between offline training and sequential simulation.

• We extend the system to synchronized seven-camera and conditional LiDAR generation, and further distill the generative process to few-step inference using consistency training and distribution matching, improving it suitability for repeated data synthesis and interactive use.

## 2 Related Work

## 2.1 Video Foundation and Interactive World Models

Large-scale video foundation models established the architectural and data-scaling basis for modern world generation. Wan [1] combines a causal video autoencoder, a diffusion-transformer backbone, flow matching, large-scale image– video curation, and task-specific post-training in an open model suite. Cosmos [2] extends the foundation-model view toward physical AI, organizing tokenization, pretraining, conditional post-training, deployment, and safety tooling as one platform. These systems show that reusable visual priors can support many downstream generators more efficiently than training each domain from scratch. HelloWorld follows this foundation-to-specialization principle: its main 2B configuration starts from Cosmos-Predict2.5, but introduces a curriculum designed around the supervision and camera geometry of autonomous-driving data rather than immediately fitting a driving-only adapter.

A related line turns video models into interactive environments controlled by camera motion or user actions. Genie 3 [3] emphasizes promptable real-time worlds, while HY-World 1.5 / WorldPlay [4] describes a pipeline spanning pretraining, interaction-oriented post-training, and streaming distillation. LingBot-World [5] and Matrix-Game 2.0 [6] emphasize long-horizon interaction and open deployment. Publicly described systems in this family mainly expose a first-person visual stream and scale-free motion or learned action spaces. They are not designed around the fixed extrinsics, heterogeneous fields of view, and simultaneous outputs of a production vehicle rig. HelloWorld uses these broad-domain data priors in Stage 1, then explicitly transfers them to metric pose control and synchronized seven-view generation. We distinguish interactive observation generation from a validated driving closed loop: the latter additionally requires action-to-dynamics propagation, reactive traffic, and policy-in-the-loop metrics.

## 2.2 Driving Video Generation and Conditioning

Controllable driving generation has developed along two overlapping routes. Layout-driven methods condition synthesis on BEV sketches, HD maps, occupancy, or projected 3D boxes. BEVGen [7] and BEVControl [8] establish layoutto-street-view generation with multi-perspective constraints. DriveDreamer [9], MagicDrive [10], Panacea [11], and UniScene [12] extend structured conditioning to driving video and multi-view settings. MagicDrive-V2 [13] combines a multi-view DiT block with spatial-temporal encoders for camera, trajectory, map, and box conditions, and uses progressive training to increase resolution and duration. DiVE [14] likewise studies stronger control within a DiT video generator. Cosmos-Transfer [15] generalizes this idea to adaptive combinations of control modalities, including depth, segmentation, LiDAR, and HD maps.

Layout conditioning is effective because it supplies dense geometric evidence at every frame. It also creates an important modeling choice: when layouts are reprojected under the target camera trajectory, ego motion is partly represented by the layout sequence itself. Such a model can edit geometry precisely, but using it for pose-only pretraining requires dense annotations, and independently changing “where the vehicle moves” and “what exists in the scene” becomes less direct. HelloWorld retains dense layout control while assigning ego motion to a separate metric 6-DoF Plücker pathway. The distinction is architectural rather than semantic: pose and layout may describe the same future, but neither must pass through the other’s encoder.

Action-driven and predictive models expose motion more directly. GAIA-1 [16] formulates driving as autoregressive token prediction conditioned on text and vehicle actions. GAIA-2 [17] moves to latent flow matching, generates consistent multi-camera video, accepts ego dynamics, agent state, road semantics, and scene metadata, and supports slidingwindow autoregressive prediction from previous latent context. Vista [18] studies a generalizable single-view world model with several driving controls. GenAD [19] learns a generalized predictive representation. DrivingWorld [20] returns to a Video-GPT formulation, and Drive-WM [21] connects multi-view visual forecasting to planning. These works demonstrate that explicit actions and multi-view generation are compatible. HelloWorld’s narrower distinction is the simultaneous use of dense metric camera pose, a separately switchable layout tower, a calibrated seven-camera rig, and one block-causal interface across all training stages.

## 2.3 Multi-View Consistency and Single-to-Multi-View Transfer

Synchronized driving cameras introduce constraints absent from ordinary video generation: views have fixed but different intrinsics and extrinsics, only selected pairs overlap, and an object may leave one view while entering another. Existing systems address these constraints through joint generation, camera-aware embeddings, shared BEV or projected conditions, and explicit cross-view attention [10, 11, 13, 17]. These mechanisms improve consistency, but most methods train the multi-view generator as a target architecture from the beginning or initialize it from a generic video model before driving-specific training. This makes synchronized data the main route by which both appearance and cross-view correspondence are learned.

HelloWorld separates those learning problems. Stage 1 learns appearance, motion, and pose response from the union of single-view internet video and flattened fleet cameras. Stage 2 then inserts adjacency-restricted cross-view attention as a zero-output residual. Copying its query, key, value, and normalization weights from self-attention provides a non-random projection, while the zero output projection makes the added residual contribution zero at initialization. The intended benefit is a controlled initialization for learning cross-view interactions from synchronized data. Zero initialization preserves the added branch’s input at insertion. Whole-model parity and retention after full-model fine-tuning are separate empirical questions.

## 2.4 Causal and Long-Horizon Video Generation

Long-horizon generation is not synonymous with causality. A model may synthesize a long clip with bidirectional attention over the full sequence, recurrently extend a fixed denoising window, generate sparse anchors and interpolate between them, or maintain a causal state that can be updated online. These choices have different implications for latency, memory, and train–test mismatch. GAIA-1 and DrivingWorld use autoregressive token models [16, 20]. GAIA-2 predicts future latent windows from a sliding context [17], and MiLA [22] combines low-frame-rate anchor generation with a coarse-to-refine process to produce videos up to one minute. Recent driving work continues to investigate causal rollout and temporal error correction [23].

For diffusion models, a central difficulty is the distribution of the history. Teacher-forced training observes real or clean previous frames, whereas autoregressive inference conditions on samples containing the model’s own errors. Diffusion Forcing [24] assigns noise levels across a sequence so generation and prediction tasks can share a diffusion formulation. Self Forcing [25] instead trains with self-generated autoregressive context to address exposure bias more directly. HelloWorld combines these ideas at chunk granularity: per-chunk noise schedules, context corruption, and Self-Rollout Alignment train the same block-causal network that is executed sequentially with a KV cache. Unlike overlap-and-regenerate extension, completed chunks remain causal context rather than being jointly denoised again. Our distinction is not the causal mask alone, but the role of the causal interface across learning: it is established on broad Stage-1 data, preserved while multi-view and structured-control capabilities are attached under scarcer supervision, and explicitly aligned to generated history.

LingBot-World-Infinity [26] begins from the observation that an interactive world is a temporal state process: each state is generated from past observations and currently available inputs, rather than from future frames or instructions. Its reason for causal pretraining is therefore the temporal factorization of deployment, not the availability of a particular supervision subset. Training the deployable transition first produces a causal world-model teacher whose dynamics can subsequently be accelerated without changing the state definition. Its MoBA mask is not purely autoregressive: a bidirectional component regularizes teacher forcing to preserve visual fidelity and flexible-length behavior, while a lower-triangular text mask prevents future-prompt leakage on the autoregressive component. Consistency distillation then reduces denoising cost, and self-rollout DMD targets the state distribution induced by the student itself. HelloWorld adopts the same high-level sequence of causal state definition, teacher training, deployment-state alignment, and fewstep distillation. It instantiates this sequence with chunk-causal driving observations, nested geometric supervision, and explicit cache-boundary training. The benefits of LingBot’s complete recipe do not by themselves establish that HelloWorld’s early causal adaptation is superior to a matched bidirectional-to-causal retrofit.

## 2.5 Structured Control Architectures

ControlNet [27] popularized a trainable side branch with zero-initialized connections for adding spatial control while protecting a pretrained diffusion backbone. VACE [28] extends adapter-style conditioning to a broad family of video creation and editing tasks. Driving generators instantiate these ideas with maps, occupancy, boxes, depth, or LiDAR [8, 13, 15]. The main design axes are where controls are encoded, whether multiple modalities share a branch, and how control strength varies across space and time.

HelloWorld adopts a VACE-style video control tower for rendered HD maps and 3D boxes, but does not ask that tower to represent ego pose. Pose modulates the main network through Plücker features and AdaLN, while structured control enters through VAE latents and periodic tower hints. Stage 3 explicitly trains joint, pose-only, and control-only batches. Consequently, dropping either input at inference is a trained operating mode rather than an architectural modification or an assumption that the remaining condition reconstructs the missing one.

## 2.6 Few-Step Generation

Consistency models [29] and their continuous-time scaling variants [30] learn mappings that reduce the number of numerical sampling steps. Distribution matching distillation [31] and DMD2 [32] instead train a fast generator against distribution-level score differences. Most formulations are evaluated on images or self-contained clips. In an autoregressive world model, however, a small per-window distribution shift can become the next window’s context and accumulate over time. HelloWorld combines packed teacher-forced consistency training with self-forcing distribution matching. The distinction is the role of these objectives within a progressively trained driving generator: capability acquisition, generated-history alignment and compression share the same causal observation interface. Teacher/student comparisons assess retained geometry and control together with execution cost.

Table 1. Capability comparison with representative world models. “Pose form” distinguishes independent metric 6-DoF ego pose, low-dimensional action, and ego motion implicit in layout conditioning. ✓ = supported, △ = partial/preliminary, ✗ = not reported. Entries reflect public reports as of Aug. 2026.
<table><tr><td></td><td>Pose form</td><td>Multi- view</td><td>Structured control</td><td>Causal AR</td><td>Few- step</td></tr><tr><td>MagicDrive-V2 [13]</td><td>layout</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>GAIA-2 [17]</td><td>action</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Cosmos-Transfer [15]</td><td>layout</td><td>△</td><td>√</td><td>△</td><td>△</td></tr><tr><td>Cosmos WFM [2]</td><td>action</td><td>x</td><td>△</td><td>√</td><td>√</td></tr><tr><td>LingBot-World [5]</td><td>action</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>HY-World 1.5 [4]</td><td>action</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Genie 3 [3]</td><td>action</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>HelloWorld (ours)</td><td>6-DoF</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 2.7 LiDAR Representation, Conditional Generation, and Rectification

Learning a geometric video representation. Video autoencoders such as Wan [1] provide a temporally compressed latent interface, but range images are not ordinary color images. Their values represent metric surfaces, while absent returns require a separate interpretation. We retain a pretrained video-VAE architecture and adapt its representation and objectives: explicit return validity separates existence from range, and noman and Haar constraints address unsupported intermediate depths and high-frequency structure. The contribution is this geometry-specific training design, not a new video-VAE backbone or a claim that BCE and Haar transforms are new.

From visual conditions to sensor observations. Cosmos [2] provides a foundation-model setting for physical-AI generation; its RGB-to-LiDAR generator serves as a quantitative reference here. UniScene [12] organizes driving-scene generation around occupancy. UniDriveDreamer [33] introduces modality-specific autoencoders and Unified Latent Anchoring for multimodal diffusion. Sensor2Sensor [34] translates monocular dashcam video into a multi-camera and LiDAR sensor suite. Our task is narrower: given synchronized RGB windows, synthesize only LiDAR. We combine native-grid token packing, azimuth/coverage ray features, latent anchoring, and geometric supervision through a frozen decoder. Packing and latent alignment adapt the pretrained interface; the key training link is that generated latents must also satisfy output-space geometric criteria.

Geometry after range-image decoding. L3DR [35] addresses depth bleeding and wavy surfaces with threedimensional residual regression and robust Welsch supervision. We adopt this rectification principle with radial updates that retain each original ray, and study single-frame and motion-compensated temporal variants. Because both VAE reconstruction and DiT generation end in the same range representation, the rectifier interface is reusable; this does not imply that a gain measured on reconstruction transfers unchanged to generation

## 2.8 Capability Positioning

Table 1 summarizes the capability gap discussed above. It focuses on the properties of the core generator. LiDAR representation and conditional generation are developed in Sections 4.7.1 and 4.7.2, with generated-video applications in Section 5.5.4. The relevant distinction is not whether a model accepts any motion input, but whether metric ego pose is independently controllable rather than represented by a low-dimensional action or embedded in a projected layout sequence.

## 3 Data Construction

HelloWorld combines broad visual generalization with driving-specific control and synchronized observation generation, but these capabilities rely on data with different levels of supervision. General and fleet video provide large-scale appearance and motion coverage, synchronized camera rigs introduce calibrated multi-view geometry, and a smaller annotated subset provides HD maps and 3D object layouts for explicit scene control. Figure 2 traces how these supervision levels feed the three training stages through source-specific preparation and quality checks. We describe the sources, conversion and quality control, semantic annotation and balancing, pose normalization, and data governance below.

![](images/b0a46273fef5acd6fdb44a995a6df2bc2b1d1d68db046ca5eb6601222115ffed.jpg)  
Figure 2. Data construction under nested supervision. Pose-supervised video provides the broad pool for learning causal visual dynamics. Synchronized camera observations add multi-view geometry, and the structured-control subset supplies rendered maps and object annotations. Source-specific preparation and quality validation connect these supervision levels to the three training stages. The flows illustrate processing and supervision relationships rather than measured data proportions.

## 3.1 Data Sources and Supervision Availability

The data inventory distinguishes training sources from the external evaluation benchmark:

1. Fleet data: real 7-camera vehicle logs (front-wide, front-tele, left/right-front, left/right-back, rear, 10Hz, mixed FOVs, fixed extrinsics) with per-frame metric ego pose and scene tags. An annotated subset carries HD map and 3D bounding-box labels. The curated pool snapshot contains 1,915,569 quality-gated 40s clips, from which the balanced selection of Section 3.4 retains 497,773.

2. General-purpose corpora: DL3DV, RealEstate10K, SpatialVID, Sekai (real-walking and game subsets), OmniWorld-Game, ScanNet, and MatrixCity. These corpora provide video with recoverable but scale-free camera trajectories and no driving-specific labels

3. nuScenes [36]: the public benchmark used for external comparison (Section 5.2).

Table 2 quantifies the Stage-1 mixture: fleet data contribute 35.1% of the 29-frame training windows, with the remainder drawn from general-purpose video. The key observation is an asymmetry: camera pose is available or recoverable for all training sources, whereas synchronized calibration and structured scene annotations are available only for progressively smaller subsets offleet data. Consequently, pose-conditioned learning can use the broadest mixture of fleet and general-purpose video, while multi-view specialization and structured control rely on increasingly selective driving supervision. This supervision hierarchy motivates the progressive learning strategy described in Section 4: broad pose-conditioned modeling first, followed by synchronized multi-view specialization and structured scene-control adaptation.

Table 2. Stage-1 training asset inventory (shares are pre-flatten window proportions, not effective sampler exposure): 29-frame training windows per source (754,629 sequences, 3,662,212 windows in total). Fleet windows are counted after caption-QA filtering of the 1,510,454 pose-balanced windows of Section 3.4. With 7-View Flatten each fleet window yields 7 single-view samples.
<table><tr><td>Source</td><td>Windows (w29)</td><td>Share</td><td>Views</td><td>Metric</td><td>HDMap/BBox</td></tr><tr><td>Fleet (7-camera)</td><td>1,283,914</td><td>35.1%</td><td>7</td><td>√</td><td>subset</td></tr><tr><td>SpatialVID</td><td>991,073</td><td>27.1%</td><td>1</td><td>x</td><td>x</td></tr><tr><td>Sekai (real-walking)</td><td>746,509</td><td>20.4%</td><td>1</td><td>x</td><td>x</td></tr><tr><td>DL3DV</td><td>211,432</td><td>5.8%</td><td>1</td><td>x</td><td>x</td></tr><tr><td>ScanNet</td><td>185,548</td><td>5.1%</td><td>1</td><td>x</td><td>x</td></tr><tr><td>RealEstate10K</td><td>127,536</td><td>3.5%</td><td>1</td><td>x</td><td>x</td></tr><tr><td>Sekai (game)</td><td>71,930</td><td>2.0%</td><td>1</td><td>x</td><td>x</td></tr><tr><td>OmniWorld-Game</td><td>36,837</td><td>1.0%</td><td>1</td><td>x</td><td>x</td></tr><tr><td>MatrixCity</td><td>7,433</td><td>0.2%</td><td>1</td><td>x</td><td>x</td></tr></table>

Nested supervision and progressive learning. Figure 2 organizes the training data by available supervision. Let $\mathcal { D } _ { \mathrm { p o s e } }$ denote observations with camera trajectories, $\mathcal { D } _ { \mathrm { m v } }$ the subset with synchronized and calibrated camera views, and $\mathcal { D } _ { \mathrm { c t r l } }$ the subset that additionally carries structured scene annotations. At the level of source observations, these requirements define

$$
\mathcal { D } _ { \mathrm { c t r l } } \subset \mathcal { D } _ { \mathrm { m v } } \subset \mathcal { D } _ { \mathrm { p o s e } } .\tag{1}
$$

The branches are therefore overlapping supervision pools rather than disjoint partitions of raw data. A synchronized fleet observation can contribute independent camera streams to Stage 1, a grouped surround-view example to Stage 2, and, when scene labels are available, a controlled example to Stage 3. This organization allows broad video coverage to support appearance and motion learning before more selective geometric and layout supervision is introduced.

## 3.2 Conversion Pipeline

The preparation paths in Figure 2 enforce the observation requirements of each supervision pool. Pose-supervised video is partitioned into temporal windows and paired with captions and camera trajectories. Synchronized multi-view data additionally requires consistent camera calibration, ordering, and timestamps. Structured-control data further requires validated scene annotations rendered into the corresponding camera views. The illustrated 29-frame window is the standard training unit, with longer windows described in Section 3.4.

Quality validation follows these requirements. Duplicate removal, visual quality filtering, and pose and motion validation support the common video interface. Multi-view and structured-control examples additionally require synchronization and overlap validation so their streams describe compatible observations of the same scene. Control annotation checks and rendering alignment ensure that the supplied geometry refers to the associated RGB frames. These checks establish sample validity, while the balancing procedure in Section 3.4 addresses the frequency of valid scenarios and maneuvers.

For fleet data, a common intermediate representation retains synchronized RGB, calibration, poses, and available scene annotations. HD maps and boxes are rasterized into per-camera control videos depicting lane lines, curbs, road marks, traffic signs and lights, and category-colored 3D boxes. Cross-frame lane-divider fusion reduces annotation flicker, and image-boundary clipping retains the visible portions of projected elements. The resulting segments preserve camera ordering and temporal alignment for pose encoding and condition retrieval during training.

## 3.3 Captioning and Semantic Annotation

Each segment carries VLM-generated captions in six stored variants (short/medium/long × with/without ego-behavior description), produced by qwen3.6-plus, plus structured VRI tags and scene tags (road environment, turn bin, weather, time of day). Captioning is performed per segment on 3 sampled frames at 720p, with the six caption variants and a 7-dimension scene-tag set produced in a single VLM call. Captions overlapping previously annotated clips are reused. The released Stage-1 training configuration consumes only the three caption\_wo\_behavior\_{short,medium,long} fields. The other variants remain data assets rather than training inputs. Training samples select the annotation window nearest to the segment center, avoiding stale captions that we have observed to cause semantic drift later in long generated videos. In synchronized multi-view training, front-wide receives the complete caption while the other cameras receive only a view-specific prefix. This prevents one scene description from being replicated as if it described every field of view and avoids duplicating the same directional description across views.

## 3.4 Selection, Balancing, and Quality Gates

Driving data is heavily biased toward straight, uneventful cruising: in the raw fleet pool, 82.8% of clips are tagged straight road, 69.9% daytime, 84.1% medium speed, while curves, night, rain, intersections, and pedestrian interactions each fall below a few percent. We rebalance in two orthogonal layers: a clip-level tag balance decides which clips enter the pool (appearance/semantic diversity), and a segment-level pose balance decides which windows within each clip are trained on (geometric/maneuver ratios). The two layers are deliberately separated because clip-level tags are 40s-window unions. In the pool, 94.7% of high-curvature-turn clips also carry the straight tag (a 40s clip is typically 25s approach + 8s turn + 7s exit). Consequently, clip-level balance alone would not translate into window-level balance.

Layer 1: clip-level tag balancing. Starting from a quality-gated pool of 1,915,569 clips (snapshot 2026-06-11), low-image-quality clips (fog on lens or glare, 53,441) are removed, leaving 1,862,128. The 445 raw scene tags are folded into a balancing vocabulary of 29 categories / 198 nodes (deepest-node deduplication per category, pipeline-QC tags excluded). Selection then applies a three-tier policy:

• Hard-lock true long tail: all clips carrying any of the 114 tags with $n _ { t } \le 1 0 \mathrm { k }$ (tunnels, construction, roundabouts, hard braking, standing water, animals, . . . ) are retained at 100%.

• Hard-lock safety/interaction whitelist: 14 mid-frequency whitelist tags $( 1 0 \mathrm { k } < n _ { t } \le 1 0 0 \mathrm { k } \colon$ high-curvature turns, unprotected left turns, rain, cut-ins, U-turns, ramps, . . . ) are retained at 100%. Four high-frequency whitelist tags (night, fog, low/mid-curvature turns, dawn/dusk) are exempt from capping and survive via co-occurrence.

• Cap-and-trim the head: remaining candidates are greedily removed in ascending order of a rarity weight $w _ { i } =$ ma $\mathbf { x } _ { t \in T _ { i } } \left( N / \operatorname* { m a x } ( n _ { t } , 5 0 ) \right) ^ { 0 . 5 }$ , stratified over 23,405 (city, vehicle×day) strata so that redundant clips from the same collection session are removed first, until head-tag counts reach their caps.

The result is a selection of 497,773 clips (26.7% of the pool): every rare tag is preserved verbatim, long-tail share rises uniformly by 3.74× (e.g., high-curvature turns 4.9%→18.4%, unprotected left turns 4.6%→17.1%, rain 3.7%→13.9%), while head tags lose 72% of absolute volume. Residual head-share shaping (e.g., pushing the effective straight-road share to a target) is done by per-tag repeat weights in the training cache, which is reversible and tunable, rather than by destroying locked long-tail clips.

Layer 2: segment-level pose balancing. Within the selected clips (493,500 unique sequences), non-overlapping 29-frame (≈2.9s) and 61-frame (≈6.1s) windows are cut under a per-frame validity mask requiring all 7 cameras, metric pose, and sane timestamps for every frame in the window. A sampled audit found that 63% of clips contain mid-clip invalid frames, which fixed head/tail margins would miss. Each window is characterized purely from pose: cumulative yaw turn angle (5 bins), median speed (5 buckets, using the median to retain sensitivity to hard stops), and maximum longitudinal acceleration. Quotas allocate 30% to straight windows (uniform over the 5 speed buckets) and 25/20/15/10% to increasing turn-angle bins, with the largest bin $( \geq 4 0 ^ { \circ } )$ taken exhaustively as the bottleneck. Windows with max $| a | \geq 3  { \mathrm { m / s ^ { 2 } } }$ (≈0.5% of supply, the empirically rare regime) bypass quotas entirely. From 5,436,368 candidate 29-frame windows this yields 1,510,454 balanced training windows (and 1,121,947 61-frame windows), with straight-speed buckets exactly equalized at 88,919 each.

Before training, a Data QA contract is enforced per segment: correct camera ordering across all views, box-drift bounds, and caption field alignment.

## 3.5 Pose Representation and Scale Unification

For fleet data, per-frame ground-truth metric camera pose is composed as vcs2boot ◦ camera2vcs. For general corpora, trajectories are recovered up to scale. A fixed $\mathrm { O p e n C V }$ camera → vehicle rotation bridges the OpenCV convention (x right, y down, z forward) and the vehicle-control convention (x forward, y left, z up), so commanded yaw and pitch retain their physical meaning. Unless stated otherwise, conditioning uses frame-to-frame relative transforms rather than expressing every frame only against the clip’s first frame.

Each transform is converted online into a dense six-channel Plücker raymap. For camera center o and unit ray direction d, each pixel stores the ray moment $m = o \times d$ followed by the direction d. For non-metric sequences, translation is normalized by the median magnitude of frame-to-frame displacement over the complete sequence, with degenerate static sequences falling back to unit scale. The same translation normalization is applied consistently during training and inference. This canonicalization preserves within-sequence motion ratios while reducing arbitrary scale variation across reconstructed corpora.

To retain the distinction between metric fleet trajectories and scale-free reconstructed trajectories, each pose sequence is additionally associated with a binary metric-scale indicator. Together with the normalized camera trajectory, this forms the common pose-conditioning contract that allows metric driving logs and scale-ambiguous general video to coexist in the same training mixture. The model-side Plücker encoding, metric-scale conditioning, and pose injection are described in Section 4.4.

## 3.6 Data Governance

Fleet data undergoes face and license-plate anonymization. Third-party corpora and pretrained weights remain subject to their respective licenses. Their use does not confer redistribution rights. Training and evaluation sequences are strictly disjoint. Geographically sensitive information is reported only as auditable aggregate statistics.

LiDAR representation and paired generation data. The LiDAR tokenizer dataset comprises 50,000 29-frame segments (1,450,000 frames), separate from the RGB foundation inventory. It uses main LiDAR sensors 0–3 and excludes blind-spot sensors 4–7. Full 360<sup>◦</sup> range observations are re-binned from 3600 to 1792 columns. The Plan-B representation uses a minimum range of 1 m and an ego-body box $x \in [ - 1 . 1 , 4 . 0 ] , y \in [ - 1 . 1 , 1 . 1 ] \mathrm { ~ m ~ }$ . Return validity and geometric compression are described in Section 4.7. VAE training observes LiDAR alone; conditional DiT training additionally requires synchronized RGB and calibrated sensor geometry. Rectifier training uses paired VAE reconstructions and measured LiDAR, with validation windows withheld from rectifier training. Reconstruction validation, the 100-clip generation evaluation, and the eight-road-clip structural comparison are separate populations; scores are not pooled across them. Generated-input application examples have no matched physical target scan.

Calibration and condition provenance. The V5 adapter reads camera calibration from external calibration files when camera metadata is absent, and inverts vehicle-to-camera extrinsics into the camera-to-vehicle contract. Layout-only preprocessing can instead consume self-contained maps, per-frame objects, ego poses and seven-camera calibration without fetching source imagery. Training and deployment must use the same object classes, lane styles and traffic-light rendering semantics. Calibration, rendering semantics and adapter versions form part of each experiment’s condition manifest.

## 4 Method

HelloWorld progressively transforms broad video priors into a controllable driving world model for data generation and interactive simulation. The RGB core first learns pose-conditioned visual dynamics from heterogeneous video and is then specialized using synchronized multi-view observations and structured scene supervision. After capability learning, we align the model with repeated sequential generation by exposing it to imperfect and self-generated observation histories, and finally distill the multi-step generator into a few-step student for efficient inference. Figure 3 locates these mechanisms in the final architecture, from pose and scene conditioning in the causal RGB backbone to the distilled RGB generator and RGB-conditioned LiDAR branch. We first define the observation-generation task and common generative backbone, then describe broad pose-conditioned learning, structured driving specialization, sequential simulation alignment, and few-step generation. The extension from RGB observations to conditional LiDAR synthesis is presented separately at the end of this section.

![](images/e31308a9c594c248f473d26aa0a5161ddffd5660ff4129e916a7022cdef28a7d.jpg)  
Figure 3. Overview of the HelloWorld generation framework. A causal teacher is distilled into a few-step student for seven-view RGB generation. A separate super-resolution module refines the generated views before conditional LiDAR synthesis. (a) The causal multi-view backbone incorporates ego-pose geometry through a ray encoder and AdaLN, scene layout through residuals from a control tower, and text through cross-attention, while cross-view attention exchanges information between neighboring cameras. (b) The RGB-to-LiDAR backbone conditions a LiDAR diffusion on encoded multi-view RGB latents while denoising LiDAR latents, which are decoded into range and intensity.

## 4.1 Overview and Problem Formulation

Observation-generation task. Let $Y _ { k } = \{ Y _ { k } ^ { ( v ) } \} _ { v = 1 } ^ { V }$ denote the RGB observations generated for temporal chunk k, where V is the number of camera views. Let $\ddot { C } _ { k } ^ { \mathrm { s u p } }$ collect the externally supplied conditions, including text, camera poses, and structured scene controls, and let $S _ { k - 1 }$ denote the committed observation history before generating the current chunk. HelloWorld models the conditional observation transition

$$
Y _ { k } \sim p _ { \theta } \left( \cdot \mid S _ { k - 1 } , C _ { k } ^ { \mathrm { s u p } } \right) , \qquad S _ { k } = \mathcal { U } ( S _ { k - 1 } , Y _ { k } , C _ { k } ^ { \mathrm { s u p } } ) ,\tag{2}
$$

where U denotes history commitment and context management after a chunk has been generated. This formulation separates observation generation from vehicle dynamics: ego trajectories and scene configurations are prescribed conditions, while HelloWorld predicts the corresponding sensor observations. The model does not infer low-level vehicle actions or propagate an explicit dynamics model from control commands.

Chunk-sequential generation. Video is encoded by a causal video VAE and partitioned into temporal latent chunks. Frames within the current chunk are generated jointly, while previously completed chunks form fixed visual context for subsequent predictions. The chunk therefore serves as the common unit for latent generation, history commitment, causal attention, and cache management. The causal claim in this work concerns observation history: future visual observations are inaccessible when predicting the current chunk. Supplied condition sequences may cover a larger temporal window, and strictly online condition causality would additionally require restricting every conditioning pathway to information available at the current simulation step.

Training-stage terminology. The method is organized below by capability rather than by training chronology. For consistency with the experiments, however, we retain three checkpoint names. Stage 1 denotes the model after broad pose-conditioned learning, Stage 2 denotes the model after synchronized multi-view specialization, and Stage 3 denotes the model after structured scene-control adaptation. Subsequent sequential-alignment and distillation stages operate on the resulting controllable multi-view model.

## 4.2 Generative Backbone and Factorized Conditioning

Latent rectified-flow backbone. HelloWorld performs generation in the latent space of a causal video VAE using a rectified-flow objective. Given clean video latents $x _ { 0 }$ and Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , we construct

$$
x _ { \sigma } = ( 1 - \sigma ) x _ { 0 } + \sigma \epsilon , \qquad \sigma \in [ 0 , 1 ] ,\tag{3}
$$

and train a velocity network v<sub>θ</sub> to predict the constant transport direction:

$$
\mathcal { L } _ { \mathrm { R F } } = \mathbb { E } _ { \boldsymbol { x } _ { 0 } , \boldsymbol { \epsilon } , \boldsymbol { \sigma } } \left[ \lVert \boldsymbol { v } _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { \sigma } , \boldsymbol { \sigma } , c ) - ( \boldsymbol { \epsilon } - \boldsymbol { x } _ { 0 } ) \rVert _ { 2 } ^ { 2 } \right] ,\tag{4}
$$

where c contains the available text, pose, view, and scene-layout conditions. Operating in latent space reduces the spatial and temporal cost of video generation while retaining the common generative interface used throughout all RGB training stages.

Block-causal multi-view DiT. The denoising backbone uses full spatiotemporal attention within the chunk currently being generated and causal access across temporal chunks. This factorization allows all latent tokens of the current observation to be jointly refined while preventing future observations from modifying already committed history. Block causality therefore defines the temporal interface of the model rather than a separate prediction objective. At multi-view stages, temporal self-attention remains organized per camera stream, while dedicated cross-view pathways exchange information among synchronized cameras at the same latent time.

Factorized conditioning. We separate conditioning according to the physical quantity that each signal constrains. Camera identity specifies the observation view through a learned camera embedding. Ego pose specifies observer geometry and is encoded from dense Plücker raymaps before modulating the backbone through adaptive normalization. Scene layout, represented by rendered HD maps and 3D boxes, is processed by a dedicated control tower that injects residual features at multiple backbone depths. Text provides complementary semantic information through crossattention. This factorization avoids forcing geometrically distinct controls into a single token pathway and allows observer motion and scene configuration to be manipulated independently.

## 4.3 Causal Observation Modeling

Block-causal latent modeling. To align the temporal information flow during training with sequential generation, HelloWorld adopts block-causal observation modeling from the beginning of driving adaptation. The video latent sequence is partitioned into temporal chunks $\{ x ^ { ( 1 ) } , \ldots , x ^ { ( K ) } \}$ . Tokens within the same chunk interact bidirectionally, allowing each local observation to be modeled jointly, whereas attention across chunks is causal: a query in chunk k can access observation tokens from chunks $j \le k$ but not from future chunks $j > k .$ . The corresponding attention mask is

$$
M _ { i j } = \left\{ { 1 , \mathrm { c h u n k } ( j ) \leq \mathrm { c h u n k } ( i ) , } \right.\tag{5}
$$

This observation-access pattern is retained when synchronized multi-view interaction and structured scene control are introduced in later specialization stages.

Chunk-wise causal training. The block-causal mask determines which observations can be accessed, while chunkwise rectified-flow times control the quality of the available history. For each chunk $k ,$ we construct

$$
\begin{array} { r } { x _ { \sigma _ { k } } ^ { ( k ) } = ( 1 - \sigma _ { k } ) x _ { 0 } ^ { ( k ) } + \sigma _ { k } \epsilon ^ { ( k ) } , \qquad \epsilon ^ { ( k ) } \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{6}
$$

where the noise level $\sigma _ { k }$ may vary across temporal chunks. The model predicts the corresponding velocity targets under the same block-causal observation mask:

$$
\mathcal { L } _ { \mathrm { c a u s a l - R F } } = \mathbb { E } \left[ \sum _ { k \in { \cal K } _ { \mathrm { t g t } } } \Big \Vert v _ { \theta } ^ { ( k ) } - \left( \epsilon ^ { ( k ) } - x _ { 0 } ^ { ( k ) } \right) \Big \Vert _ { 2 } ^ { 2 } \right] .\tag{7}
$$

By varying the noise configuration across chunks, the same causal training interface exposes the predictor to histories of different quality. We use independently sampled, progressively cleaner, and clean-context configurations depending on the training phase.

Context corruption and Self-Rollout Alignment. Chunk-wise noise training varies the quality of the available history, but standard causal training still conditions predominantly on observations derived from recorded data. During sequential generation, later predictions increasingly depend on observations generated by the model itself, creating a distribution gap between recorded and generated histories. We address this discrepancy at two levels. First, context corruption perturbs preceding latent chunks with noise, appearance degradation, and spatial distortions, broadening the local neighborhood of histories encountered during training. Second, Self-Rollout Alignment directly exposes the model to its own prediction distribution. It first generates a variable-length history with gradients disabled and then evaluates the rectified-flow objective for subsequent chunks while conditioning on that generated history. Rollout generation and optimization are therefore separated, avoiding backpropagation through the complete generated trajectory. Context corruption primarily models local degradation around recorded histories, whereas Self-Rollout Alignment additionally captures structured errors induced by the model’s own generation process.

Causal execution and cache management. At inference, HelloWorld follows the same chunk-sequential observation interface. A generated chunk is not exposed to subsequent predictions until its denoising trajectory reaches the clean endpoint. The completed chunk is then committed to the history state and written into a bounded KV cache, so future predictions condition only on completed observations rather than intermediate denoising states.

To maintain consistent relative positions as the active temporal window moves, cached keys are stored without positional rotation, and positional offsets are reapplied according to their positions in the current window. This separates persistent observation content from its temporary location within the active cache.

## 4.4 Broad Pose-Conditioned World Modeling

The first specialization step aims to preserve the breadth of heterogeneous video while introducing an explicit camera motion interface. Because camera trajectories are either directly available or recoverable for substantially more video than synchronized multi-view or scene-layout annotations, pose-conditioned learning can exploit the broades supervision pool. General-purpose video contributes diverse appearance, object, and motion statistics, while fleet video introduces driving-specific visual dynamics and metric motion. Stage 1 therefore learns a shared pose-conditioned RGB generator before more selective driving supervision is introduced.

7-View Flatten. Synchronized fleet clips contain multiple camera streams, whereas most general video contains only a single moving camera. To place these sources under the same learning interface, we flatten each fleet camera stream into an independent pose-conditioned training sample during Stage 1. This preserves all fleet observations while allowing them to be mixed directly with general video under a common single-stream objective. Cross-view correspondence is intentionally deferred to later specialization, where synchronized observations are explicitly grouped again.

Pose representation. Each camera trajectory is converted into dense Plücker raymaps aligned with the video latent grid. For camera center o and unit ray direction d, each pixel stores the ray moment m = o × d together with d. Fleet trajectories retain metric translation, whereas reconstructed trajectories from general video are normalized by their sequence-level motion scale. We append the binary Metric Scale Flag as an additional dense pose channel to distinguish metric and scale-free regimes within the same pose encoder. Its newly introduced input weights are initialized to zero so that adding the flag does not perturb the pretrained pathway at initialization. A separately learned null-pose representation distinguishes missing pose conditioning from a physically stationary camera. The encoded pose features are temporally aligned with the latent video features and injected into the backbone through adaptive layer normalization.

## 4.5 Structured Driving Specialization

Broad pose-conditioned learning provides general visual priors and camera-motion control, but driving simulation additionally benefits from structured supervision that ties multiple views and scene-level conditions to the same underlying world state. We therefore specialize the Stage 1 model in two steps. Synchronized camera observations first introduce explicit cross-view interaction, after which HD maps and 3D boxes provide a separate interface for manipulating scene configuration.

## 4.5.1 Synchronized Multi-View Modeling

Zero-Init Cross-View Transfer. Cross-view interaction is introduced through a residual attention branch inserted after per-view self-attention. Each camera communicates with its calibrated neighboring cameras at the same latent time, allowing correspondence to emerge without assuming pixel-level alignment. Query, key, value, and normalization parameters are initialized from the corresponding self-attention layer, while the output projection of the new branch is initialized to zero. For hidden features h, the added branch has the form

$$
h ^ { \prime } = h + W _ { o } \mathrm { A t t n } _ { \mathrm { n e i g h b o r } } ( h ) , \qquad W _ { o } = 0 \quad \Longrightarrow \quad h ^ { \prime } = h .\tag{8}
$$

The inserted branch is therefore initially inactive and learns cross-view interaction progressively during optimization. Restricting communication to neighboring cameras encodes the rig topology and avoids dense interaction among every view pair.

Multi-view adaptation. Zero initialization guarantees an identity mapping only for the newly inserted residual branch when its output bias is also zero; it does not imply exact equivalence of the complete architecture once camera embeddings and multi-view organization are introduced. We therefore fine-tune the full Stage 2 network, allowing appearance, pose response, and cross-view communication to adapt jointly. The Stage 2 checkpoint used in our experiments corresponds to the model after this synchronized multi-view specialization and before structured scene control is introduced.

## 4.5.2 Explicit Scene Control

Multi-view specialization constrains how the scene is observed, but counterfactual generation additionally requires an explicit interface for modifying the scene itself. We distinguish observer motion, specified by ego pose, from scene configuration, specified by HD maps and 3D object boxes. Keeping these factors separate enables interventions on camera motion and scene layout through independent conditioning pathways.

Structured-control tower. HD maps and 3D boxes are rendered into camera-aligned control videos and encoded into latent features. A VACE-style control tower transforms these features into residual signals that are injected at multiple depths of the RGB backbone. Cross-view branches within the control pathway allow layout information to propagate across synchronized cameras, while a learnable scalar gate modulates the overall contribution of structured control. Pose conditioning remains in the geometric modulation pathway and is not replaced by the layout representation. Consequently, either pose or layout conditioning can be disabled without modifying the model architecture.

Progressive control adaptation. We introduce the structured-control pathway in two phases. We first freeze the established multi-view backbone and optimize only the newly added control tower on examples with active scene-layout conditioning. This allows the control pathway to acquire a meaningful influence before interacting with the pretrained visual representation. We then fine-tune the complete model using a mixture of joint, pose-only, and control-only examples. The mixture exposes the same network to different subsets of conditions and encourages it to retain the corresponding interfaces after joint adaptation. When ego pose is changed while world-coordinate map and object geometry are held fixed, the camera-projected controls are re-rendered for the new viewpoint.

## 4.6 Efficient Few-Step Generation

Repeated use of the world model makes sampling cost part of the system design. The multi-step rectified-flow model provides a strong teacher but requires multiple network evaluations for every generated chunk. We therefore distill the same observation and control interface into a few-step student. The student is first initialized using discretized consistency training, then optimized with continuous-time consistency under recorded history and selfforcing distribution matching under its own generated samples.

## 4.6.1 Consistency Initialization and Training

Packed teacher forcing. For each view, clean and noisy latent sequences are concatenated as $[ x _ { 0 } \mid x _ { \sigma } ]$ . Clean queries follow the standard block-causal mask. A noisy query belonging to chunk k may attend to clean chunks strictly before k and to its own noisy chunk, but not to the corresponding clean target or to future clean chunks. Conditions follow the same view and temporal ordering, and the consistency objective is evaluated only on noisy target tokens. This packed formulation allows all target chunks to be trained in a single forward pass while preserving the history available under teacher forcing.

Discretized consistency initialization. We first initialize the few-step student with discretized consistency training. Adjacent noise levels are sampled independently for each target chunk, and a frozen guided teacher advances the noisy state by one Euler step under the teacher flow. The student is trained so that predictions from the original and teacher-advanced states map to a consistent clean estimate, with the latter branch stop-gradient. This initialization compresses local segments of the teacher trajectory before continuous-time consistency and distribution matching are introduced.

Continuous-time consistency. Continuous-time consistency extends the same objective beyond neighboring discretization levels. Let $v _ { \theta }$ denote the student velocity, $v _ { T }$ the guided teacher velocity, and M the noisy-target mask. The directional derivative of the student along the teacher flow is

$$
D _ { T } v _ { \theta } = J _ { ( x , \sigma ) } v _ { \theta } \left[ M v _ { T } , M \right] .\tag{9}
$$

We construct the surrogate direction

$$
g = M \big ( v _ { T } - v _ { \theta } - r \sigma D _ { T } v _ { \theta } \big ) , \qquad \widehat { g } = \frac { g } { \| g \| _ { 2 } + \varepsilon _ { \mathrm { s } } } ,\tag{10}
$$

and optimize

$$
\mathcal { L } _ { \mathrm { s C M } } = \mathbb { E } \left[ \| M \left( v _ { \theta } - \mathrm { s g } ( v _ { \theta } ) - \mathrm { s g } ( \widehat { g } ) \right) \| _ { 2 } ^ { 2 } \right] ,\tag{11}
$$

where sg denotes stop-gradient and r gradually introduces the tangent term during warmup. Clean history is held fixed while the consistency constraint is applied to the current noisy prediction. The directional derivative is evaluated with an exact Jacobian-vector product.

## 4.6.2 Self-Forcing Distribution Matching

Consistency training compresses the teacher trajectory under observed history, but the deployed student generates both its outputs and subsequent context through its own few-step execution. We therefore additionally optimize the student on its induced sample distribution. The student first produces a sample G by chunk-sequential few-step generation through the same KV-cache interface used at inference. The generated sample is then re-noised as

$$
x _ { \sigma } = ( 1 - \sigma ) G + \sigma \epsilon .\tag{12}
$$

A frozen rectified-flow teacher and a trainable fake-score network predict corresponding clean estimates $\widehat { x } _ { T }$ and $\widehat { x } _ { F }$ The fake-score network is initialized from the teacher and trained with a separate optimizer on the student’s generated distribution.

The difference between teacher and fake-score predictions defines a normalized correction direction,

$$
\Delta = \frac { \widehat { x } _ { F } - \widehat { x } _ { T } } { \operatorname* { m a x } ( \operatorname* { m e a n } _ { M } | G - \widehat { x } _ { T } | , \varepsilon _ { \mathrm { d } } ) } ,\tag{13}
$$

and the student receives the surrogate distribution-matching objective

$$
\mathcal { L } _ { \mathrm { D M D } } = \mathbb { E } \left[ \left. M \left( G - \mathrm { s g } ( G - \Delta ) \right) \right. _ { 2 } ^ { 2 } \right] .\tag{14}
$$

The combined student objective is

$$
\mathcal { L } _ { \mathrm { s t u d e n t } } = \lambda _ { \mathrm { c } } \mathcal { L } _ { \mathrm { s C M } } + \lambda _ { \mathrm { d } } \mathcal { L } _ { \mathrm { D M D } } .\tag{15}
$$

Consistency training begins before distribution matching is activated, providing a stable few-step initialization before optimization on the student’s self-generated distribution.

## 4.6.3 Few-Step Sequential Inference

Few-step inference follows a descending rectified-flow noise schedule. At each denoising step, the student predicts a clean latent estimate and re-noises it to the next scheduled level. After the final step, the resulting chunk is committed to the KV cache and becomes context for the subsequent prediction. The distilled model retains the same text, pose, scenecontrol, and cross-view interfaces as the teacher, so acceleration does not require a separate deployment architecture. The same procedure is used for synchronized multi-view generation, with cross-view communication evaluated at every student step.

## 4.7 Multi-Sensor Extension: Conditional LiDAR Generation

We extend RGB observations to LiDAR through two complementary tasks: learning a geometry-preserving video representation and generating LiDAR conditioned on multi-camera RGB. The stages share output-space geometric supervision, while only the LiDAR stream is synthesized during conditional generation.

## 4.7.1 Geometry-preserving LiDAR VAE

We adapt the Wan 2.1 causal video VAE [1], initialized from the tokenizer distributed with Cosmos-Predict2.5, to LiDAR sequences using weights separate from the RGB tokenizer. The three-channel input and reconstruction target are [ <sup>¯</sup>d, <sup>¯</sup>d, 2m − 1]: the first two channels duplicate normalized range, and the third encodes binary return validity m. There is no intensity channel. Representing missing returns with a fixed depth value forces regression to learn whether a surface exists and how far away it is through the same signal. Instead, the validity channel records measured returns independently of the continuous depth canvas. This separates missing-return prediction from depth reconstruction and avoids treating interpolated holes as observed surfaces.

Validity alone cannot eliminate flying points: a ray with a genuine return can still be reconstructed between foreground and background. To reduce flying-point artifacts at depth discontinuities, we augment range reconstruction and validity classification with complementary Noman and Haar losses. Noman (no-man’s-land) identifies depth gaps at foreground– background boundaries in the reference and penalizes predictions inside their empty interior, excluding locations where the reference itself contains an intermediate surface. It discourages floating points between real surfaces, while range reconstruction determines which surface to recover. Haar matches multiscale high-frequency Haar-wavelet coefficients of predicted and reference depth maps. These coefficients describe local depth changes, so matching them preserves sharp boundaries and penalizes misplaced edges without indiscriminately smoothing genuine detail. Together, Noman suppresses unsupported intermediate depths, while Haar constrains boundary transitions to reduce geometric artifacts associated with blurred or displaced edges. The complete objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { V A E } } = \mathcal { L } _ { \mathrm { r a n g e } } + \lambda _ { v } \mathcal { L } _ { \mathrm { v a l i d } } + \lambda _ { n } \mathcal { L } _ { \mathrm { n o m a n } } + \lambda _ { h } \mathcal { L } _ { \mathrm { H a a r } } } \\ { + \lambda _ { s } \mathcal { L } _ { \mathrm { n o r m a l } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p e r c e p t u a l } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } . } \end{array}\tag{16}
$$

Range reconstruction supervises the two depth channels, with greater weight on measured returns; noman, Haar, normal and depth-perceptual terms supervise their reconstructed geometry. Validity uses binary cross-entropy on the third channel’s raw logits against m, while KL regularizes the latent distribution.

## 4.7.2 RGB-conditioned LiDAR diffusion

A multiview diffusion transformer couples clean camera latents with noisy LiDAR latents. RGB observations remain fixed during sampling, while attention exchanges information across views, modalities and time to generate the LiDAR sequence. Native-grid packed attention concatenates separately patchified camera and LiDAR grids, avoiding the spatial padding required by a shared canvas. Ray features encode azimuth and camera coverage, alongside view identifiers, to distinguish sensor directions. Following Unified Latent Anchoring [33], we align LiDAR latent statistics to the RGB-pretrained scale and invert this transform before decoding.

Geometry supervision through a frozen decoder. Latent prediction error does not directly capture whether decoded points lie on plausible surfaces. We therefore supervise the generator through the frozen LiDAR decoder, transferring the four geometric criteria used in representation learning to conditional generation:

$$
\mathcal { L } _ { \mathrm { g e n } } = \mathcal { L } _ { \mathrm { R F } } + w ( \sigma ) \sum _ { k \in \{ \mathrm { r a n g e , v a l i d } , \mathrm { n o m a n } , \mathrm { H a a r } \} } \alpha _ { k } \mathcal { L } _ { k } \bigl ( D _ { \phi } ( A ^ { - 1 } ( \hat { z } _ { 0 } ) ) , x \bigr ) .\tag{17}
$$

Here $\mathcal { L } _ { \mathrm { R F } }$ is the rectified-flow loss on LiDAR latents, $\hat { z } _ { 0 }$ is the generator’s clean-latent estimate, $A ^ { - 1 }$ reverses latent alignment, and x is the target range/validity map. The weights $\alpha _ { k }$ balance geometric terms, while $w ( \sigma )$ controls their contribution across noise levels. Decoder parameters $\phi$ remain fixed, but gradients pass through decoding to update the DiT. Thus the decoder cannot adapt to absorb generation errors: the generator must produce latents whose decoded geometry better matches the target. Auxiliary supervision is used only during training and adds no loss evaluation at inference.

Radial refinement. Following the 3D residual-regression approach of L3DR [35], a separately trained radial rectification network (RRN) uses point-cloud neighborhoods to adjust decoded points along their original sensor rays. Its temporal variant also incorporates neighboring predictions after ego-motion compensation; it is an offline refinement, not a causal streaming component. RRN adjusts existing returns rather than creating missing ones and requires no GT point cloud at inference. The interface can refine both VAE reconstructions and generated LiDAR, with results evaluated separately for each pipeline.

## 5 Experiments

We evaluate HelloWorld in terms of generation quality, geometric consistency, and controllability. We first examine single-view trajectory-conditioned video generation at Stage 1, followed by multi-view generation at Stage 2 and Stage 3. Additional experiments evaluate LiDAR generation, few-step distillation, and controllability under different conditioning inputs.

## 5.1 Single-View Video Generation

Experimental setup. We compare HelloWorld with HY-WorldPlay and Lingbot World 2.0 on the 29-frame and 100-frame benchmarks. The evaluation measures whether generated videos follow the prescribed camera trajectories while maintaining perceptual quality.

The short-video benchmark contains 100 clips, comprising 60 general-domain clips from DL3DV, Sekai, SpatialVid, RealEstate10K, and OmniWorld, and 40 real-world driving clips. Clips are selected through deterministic, scenestratified sampling. The driving subset covers highways, turning maneuvers, nighttime, adverse weather, parking and residential areas, and daytime urban roads.

Table 3. Pose comparison against ground truth on the 29-frame single view benchmark. Rot and Trans use Sim(3) alignment. Trans<sup>∗</sup> denotes metric-scale translation error in meters, reported only for HelloWorld on real-world on-road vehicle cases. Lower is better for all metrics.
<table><tr><td>Method</td><td>Rot↓</td><td>Trans ↓</td><td>Trans*↓</td></tr><tr><td>HY-WorldPlay</td><td>8.6305</td><td>1.0669</td><td>一</td></tr><tr><td>Lingbot World 2.0</td><td>3.2310</td><td>0.2853</td><td></td></tr><tr><td>HelloWorld</td><td>2.0620</td><td>0.2570</td><td>0.6830</td></tr></table>

Table 4. Pose comparison against ground truth on the 100-frame single view benchmark. Rot and Trans use Sim(3) alignment. Trans<sup>∗</sup> denotes metric-scale translation error in meters, reported only for HelloWorld on real-world on-road vehicle cases. Lower is better for all metrics.
<table><tr><td>Method</td><td>Rot↓</td><td>Trans ↓</td><td>Trans* ↓</td></tr><tr><td>HY-WorldPlay</td><td>15.0835</td><td>5.4504</td><td>一</td></tr><tr><td>Lingbot World 2.0</td><td>5.4169</td><td>1.4988</td><td></td></tr><tr><td>HelloWorld</td><td>2.7056</td><td>1.3997</td><td>5.5857</td></tr></table>

We estimate camera trajectories from generated videos using a fixed Pi3X estimator and compare them with the dataset reference trajectories. Estimated rotation matrices are projected onto SO(3) before evaluation. We report first-frame-relative geodesic rotation RMSE in degrees (Rot) and translation RMSE after Umeyama Sim(3) alignment (Trans). Both metrics are computed per clip and then averaged across clips. On the real-world on-road subset, we additionally report HelloWorld’s metric-scale translation RMSE (Trans<sup>∗</sup>), which aligns only the initial coordinate frame without fitting a translation scale. All pose errors are lower-is-better.

For perceptual evaluation, we report eight VBench dimensions: subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, imaging quality, and image-to-video subject and background consistency. The two image-to-video dimensions measure consistency with the reference image. The reported Mean is the unweighted arithmetic mean of these eight raw scores, rather than the full VBench benchmark score.

Trajectory adherence. Tables 3 and 4 show that HelloWorld achieves the lowest rotation and Sim(3)-aligned translation errors among the compared methods on both benchmarks. On the 29-frame benchmark, HelloWorld achieves a rotation error of 2.0620<sup>◦</sup> and a translation error of 0.2570, reducing the corresponding errors of Lingbot World 2.0 by 36.2% and 9.9%, respectively. On the 100-frame benchmark, HelloWorld obtains 2.7056<sup>◦</sup> rotation error and 1.3997 translation error, corresponding to reductions of 50.1% and 6.6%. These results demonstrate more accurate adherence to the prescribed camera trajectories under the shared evaluation pipeline.

HelloWorld’s metric-scale translation errors on the real-world driving cases are 0.6830 m and 5.5857 m on the two benchmarks. These measurements complement the scale-aligned errors by retaining sensitivity to displacement scale. Since corresponding baseline measurements are not reported, they do not establish comparative superiority in metric-scale translation accuracy.

Perceptual quality. Tables 5 and 6 show that HelloWorld’s improved trajectory adherence coexists with competitive, but not uniformly superior, perceptual scores. HelloWorld achieves eight-dimension means of 0.8328 and 0.8229, compared with 0.8411 and 0.8416 for Lingbot World 2.0. HelloWorld obtains higher motion smoothness on both benchmarks and higher dynamic degree on the 29-frame benchmark. On the 100-frame benchmark, it also achieves slightly higher subject and background consistency.

HY-WorldPlay obtains strong consistency and motion-smoothness scores, but substantially lower dynamic-degree scores of 0.2800 and 0.1400. This combination highlights the need to interpret consistency jointly with motion and trajectory adherence. HelloWorld maintains a dynamic-degree score of 0.9200 on both benchmarks while achieving the best reported pose errors. Its lower imaging-quality scores relative to Lingbot World 2.0 identify visual fidelity as a remaining limitation.

Table 5. VBench comparison on the 29-frame single view benchmark. SC: subject consistency; BC: background consistency; MS: motion smoothness; DD: dynamic degree; AQ: aesthetic quality; IQ: imaging quality; I2V-S/B: image-to-video subject/background consistency. Mean is the unweighted arithmetic mean of the eight raw scores.
<table><tr><td>Method</td><td>SC</td><td>BC</td><td>MS</td><td>DD</td><td>AQ</td><td>IQ</td><td>I2V-S</td><td>I2V-B</td><td>Mean</td></tr><tr><td>HY-WorldPlay</td><td>0.9669</td><td>0.9543</td><td>0.9933</td><td>0.2800</td><td>0.4762</td><td>0.5774</td><td>0.9829</td><td>0.9847</td><td>0.7770</td></tr><tr><td>Lingbot World 2.0</td><td>0.9150</td><td>0.9424</td><td>0.9760</td><td>0.8900</td><td>0.4819</td><td>0.5933</td><td>0.9623</td><td>0.9676</td><td>0.8411</td></tr><tr><td>HelloWorld</td><td>0.9148</td><td>0.9364</td><td>0.9807</td><td>0.9200</td><td>0.4521</td><td>0.5392</td><td>0.9563</td><td>0.9632</td><td>0.8328</td></tr></table>

Table 6. VBench comparison on the 100-frame single view benchmark.
<table><tr><td>Method</td><td>SC</td><td>BC</td><td>MS</td><td>DD</td><td>AQ</td><td>IQ</td><td>I2V-S</td><td>I2V-B</td><td>Mean</td></tr><tr><td>HY-WorldPlay</td><td>0.9354</td><td>0.9325</td><td>0.9951</td><td>0.1400</td><td>0.4685</td><td>0.5584</td><td>0.9844</td><td>0.9855</td><td>0.7500</td></tr><tr><td>Lingbot World 2.0</td><td>0.8841</td><td>0.9270</td><td>0.9814</td><td>0.9900</td><td>0.4489</td><td>0.5816</td><td>0.9573</td><td>0.9624</td><td>0.8416</td></tr><tr><td>HelloWorld</td><td>0.8876</td><td>0.9288</td><td>0.9842</td><td>0.9200</td><td>0.4453</td><td>0.4976</td><td>0.9571</td><td>0.9627</td><td>0.8229</td></tr></table>

## 5.2 Multi-View Video Generation

Experimental setup. We evaluate two training stages of HelloWorld on synchronized multi-camera generation. The mid-trained model uses pose conditioning without additional scene-control inputs, whereas the post-trained model jointly uses pose and scene-control inputs. For each model, we evaluate generation with and without first-frame image conditioning, denoted by w/ff and w/off. Removing first-frame conditioning preserves the other conditioning inputs associated with each model. We compare against Cosmos transfer 2.5 in the first-frame-conditioned setting.

We use the pose metrics defined in Section 5.1. For VBench, we report the six dimensions that do not require a reference image: subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, and imaging quality. Their unweighted arithmetic mean is reported as Mean.

Cross-view consistency metric. We measure cross-view geometric consistency using Cross-camera Sampson Error (CSE). For synchronized images from cameras a and $b ,$ we extract correspondences using a frozen outdoor LoFTR matcher. The fundamental matrix $F _ { a b }$ is computed from clip-specific camera intrinsics and extrinsics, with intrinsics adjusted to the evaluation resolution. For homogeneous matched points $\mathbf { x } _ { a }$ and $\mathbf { x } _ { b } ,$ the pixel-valued Sampson distance is

$$
d = \frac { \big | \mathbf { x } _ { b } ^ { \top } F _ { a b } \mathbf { x } _ { a } \big | } { \sqrt { \| ( F _ { a b } \mathbf { x } _ { a } ) _ { 1 : 2 } \| _ { 2 } ^ { 2 } + \| ( F _ { a b } ^ { \top } \mathbf { x } _ { b } ) _ { 1 : 2 } \| _ { 2 } ^ { 2 } } } .\tag{18}
$$

We restrict correspondences to calibrated overlap regions with five-pixel boundary erosion and retain matches with confidence at least 0.5. For each camera-pair/frame combination, we compute the confidence-weighted mean of distances truncated at eight pixels. Combinations with fewer than 20 valid matches receive an error of eight pixels rather than being discarded. Scores are averaged with equal weights over frames, camera pairs, and clips. We additionally report $\Delta \mathrm { C S E } = \mathrm { C S E } _ { \mathrm { g e n } } - \mathrm { C S E } _ { \mathrm { G T } }$ , using real videos evaluated with the same pipeline as the reference.

CSE evaluates epipolar agreement rather than complete 3D consistency: depth discrepancies that preserve the epipolar constraint may remain undetected.

Trajectory control. Table 7 shows that the first-frame-conditioned post-trained model achieves the lowest rotation and Sim(3)-aligned translation errors among the evaluated settings. Compared with Cosmos transfer 2.5, it reduces rotation error from 5.534<sup>◦</sup> to 2.027<sup>◦</sup> and translation error from 1.880 to 1.352, improvements of 63.4% and 28.1%, respectively. Metric-scale translation error decreases from 8.886 m to 4.904 m, a 44.8% reduction.

Relative to the first-frame-conditioned mid-trained model, the post-trained model reduces rotation and aligned translation errors by 38.9% and 11.3%. However, metric-scale translation error increases from 4.486 m to 4.904 m. The same pattern holds without first-frame conditioning: rotation and aligned translation errors improve, while metric-scale translation error increases from 7.972 m to 8.485 m. Thus, improved orientation and scale-aligned trajectory accuracy do not imply uniformly improved metric-scale displacement accuracy.

Table 7. Pose comparison against ground truth on multiview benchmark.
<table><tr><td>Setting</td><td>Rot (°)↓</td><td>Trans↓</td><td>Trans* ↓</td></tr><tr><td>Cosmos Transfer 2.5 (w/ ff)</td><td>5.534</td><td>1.880</td><td>8.886</td></tr><tr><td>Stage 2 (w/ ff)</td><td>3.315</td><td>1.525</td><td>4.486</td></tr><tr><td>Stage 2 (w/o ff)</td><td>5.163</td><td>2.189</td><td>7.972</td></tr><tr><td>Stage 3 (w/ ff)</td><td>2.027</td><td>1.352</td><td>4.904</td></tr><tr><td>Stage 3 (w/o ff)</td><td>3.334</td><td>1.844</td><td>8.485</td></tr></table>

Table 8. Multiview CSE.
<table><tr><td>Setting</td><td>CSE↓</td><td>∆CSE↓</td></tr><tr><td>GT video</td><td>1.730</td><td>0.000</td></tr><tr><td>Cosmos Transfer 2.5 (w/ ff)</td><td>6.549</td><td>+4.819</td></tr><tr><td>Stage 2 (w/ ff)</td><td>6.192</td><td>+4.462</td></tr><tr><td>Stage 2 (w/o ff)</td><td>7.509</td><td>+5.779</td></tr><tr><td>Stage 3 (w/ ff)</td><td>4.180</td><td>+2.450</td></tr><tr><td>Stage 3 (w/o ff)</td><td>4.892</td><td>+3.162</td></tr></table>

Cross-view consistency. Table 8 shows improved cross-view epipolar consistency for the post-trained model under both initialization settings. With first-frame conditioning, CSE decreases from 6.192 to 4.180, a 32.5% reduction relative to the mid-trained model. Without first-frame conditioning, it decreases from 7.509 to 4.892, a 34.9% reduction.

Compared with Cosmos transfer 2.5, the first-frame-conditioned post-trained model reduces CSE from 6.549 to 4.180, an improvement of 36.2%. Its excess error over the real-video reference decreases from 4.819 to 2.450, a 49.2% reduction. Even without first-frame conditioning, the post-trained model obtains lower CSE than the first-frameconditioned Cosmos baseline. All generated settings nevertheless remain above the real-video reference of 1.730, indicating a remaining gap in cross-view geometric consistency.

Perceptual quality. Table 9 shows that all four HelloWorld settings obtain higher six-dimension means than Cosmos transfer 2.5. The post-trained model without first-frame conditioning achieves the highest mean of 0.7901, compared with 0.7722 for Cosmos. Under matched first-frame conditioning, the post-trained model improves subject consistency, background consistency, and imaging quality over Cosmos, while aesthetic quality and dynamic degree are lower and motion smoothness is nearly unchanged. These results indicate a mixed perceptual-quality trade-off alongside the improvements in trajectory control and cross-view consistency.

Effect of first-frame conditioning. First-frame conditioning improves all three pose metrics and CSE for both training stages. For the post-trained model, it reduces rotation error from 3.334<sup>◦</sup> to 2.027<sup>◦</sup>, aligned translation error from 1.844 to 1.352, metric-scale translation error from 8.485 m to 4.904 m, and CSE from 4.892 to 4.180. These results support the role of image initialization in constraining generated geometry.

Its effect on perceptual quality is less uniform. First-frame conditioning increases the mid-trained model’s VBench mean from 0.7791 to 0.7882, but decreases the post-trained model’s mean from 0.7901 to 0.7754. The latter difference includes lower aesthetic and imaging-quality scores despite improved geometric accuracy. This discrepancy demonstrates that the perceptual mean alone does not capture trajectory adherence or cross-view consistency.

Public driving benchmark. Table 10 compares driving generation at short and long temporal horizons using FID and FVD. FID measures frame-level visual fidelity, while FVD evaluates the distributional similarity of generated and real videos in a spatiotemporal feature space. Lower values indicate better distributional fidelity for both metrics. The capability columns identify autoregressive generation, multi-view output, video generation, and whether dense supervision is required, distinguishing methods with different output formats and supervision requirements.

For short-horizon generation, HelloWorld obtains an FID of 7.75 and an FVD of 41.08. Among the autoregressive multi-view methods listed in the table, FAR-Drive provides a reference with an FID of 11.92 and an FVD of 82.78. Relative to FAR-Drive, HelloWorld reduces FID by 35.0% and FVD by 50.4%. Considering both metrics helps distinguish improvements in individual-frame appearance from changes in overall video quality. Image-generation baselines are included for FID comparison only, since FVD is not applicable to their image-only outputs.

Table 9. VBench comparison on the multiview benchmark.
<table><tr><td>Setting</td><td>Subject</td><td>Background</td><td>Motion</td><td>Dynamic</td><td>Aesthetic</td><td>Imaging</td><td>Mean</td></tr><tr><td>Cosmos Transfer 2.5 (w/ ff)</td><td>0.8916</td><td>0.9376</td><td>0.9878</td><td>0.9555</td><td>0.4366</td><td>0.4239</td><td>0.7722</td></tr><tr><td>Stage 2 (w/ ff)</td><td>0.8999</td><td>0.9324</td><td>0.9843</td><td>0.9800</td><td>0.4374</td><td>0.4953</td><td>0.7882</td></tr><tr><td>Stage 2 (w/o ff)</td><td>0.8716</td><td>0.9199</td><td>0.9822</td><td>0.9700</td><td>0.4466</td><td>0.4844</td><td>0.7791</td></tr><tr><td>Stage 3 (w/ ff)</td><td>0.9089</td><td>0.9462</td><td>0.9876</td><td>0.9500</td><td>0.4138</td><td>0.4456</td><td>0.7754</td></tr><tr><td>Stage 3 (w/o ff)</td><td>0.9111</td><td>0.9472</td><td>0.9788</td><td>0.9700</td><td>0.4259</td><td>0.5074</td><td>0.7901</td></tr></table>

Table 10. Driving generation on nuScenes [36] at short and long horizons. AR: autoregressive generation; MV: multi-view; DSF: dense-supervision-free. Lower FID and FVD are better.
<table><tr><td>Method</td><td>AR</td><td>MV</td><td>Video</td><td>DSF</td><td>FID↓</td><td>FVD↓</td></tr><tr><td>Short-horizon generation</td></tr><tr><td>DrivingGPT [37]</td><td>x</td><td>X</td><td>√ √</td><td>12.78</td><td>142.61</td></tr><tr><td>DrivingWorld [20]</td><td>x</td><td>x</td><td>√</td><td>√ 7.40</td><td>90.90</td></tr><tr><td>Vista [18]</td><td>x</td><td>X</td><td>√ √</td><td>6.90</td><td>89.40</td></tr><tr><td>Epona [38]</td><td>x</td><td>x</td><td>√ √</td><td>7.50</td><td>82.80</td></tr><tr><td>BEVControl [8]</td><td>x</td><td>√</td><td>x √</td><td>24.85</td><td>一</td></tr><tr><td>BEVGen [39]</td><td>x</td><td>√</td><td>x √</td><td>25.54</td><td>一</td></tr><tr><td>MagicDrive [40]</td><td>x</td><td>√</td><td>x √</td><td>16.20</td><td>一</td></tr><tr><td>UniScene [41]</td><td>x</td><td>√</td><td>√ x</td><td>6.45</td><td>71.94</td></tr><tr><td>DiST-4D [42]</td><td>x</td><td>√</td><td>√ x</td><td>7.40</td><td>25.55</td></tr><tr><td>OmniNWM [43]</td><td>X</td><td>√</td><td>√ x</td><td>5.45</td><td>23.63</td></tr><tr><td>MagicDrive [40]</td><td>x</td><td>√</td><td>√ √</td><td>18.75</td><td>218.12</td></tr><tr><td>GenAD [19]</td><td>x</td><td>√</td><td>√ √</td><td>15.40</td><td>184.00</td></tr><tr><td>Panacea [11]</td><td>x</td><td>√</td><td>√ √</td><td>16.96</td><td>139.00</td></tr><tr><td>Drive-WM [44]</td><td>x</td><td>√</td><td>√ √</td><td>15.80</td><td>122.70</td></tr><tr><td>DriveDreamer-2 [45]</td><td>x</td><td>√</td><td>√ √</td><td>25.00</td><td>105.10</td></tr><tr><td>FAR-Drive [46]</td><td>√</td><td>√</td><td>√ √</td><td>11.92</td><td>82.78</td></tr><tr><td>HelloWorld (Ours)</td><td>√</td><td>√</td><td>√ √</td><td>7.75</td><td>41.08</td></tr><tr><td colspan="8">Long-horizon generation</td></tr><tr><td>MagicDrive-V2 [47]</td><td></td><td></td><td></td><td></td><td>20.91 94.84</td></tr><tr><td>HorizonDrive [48]</td><td></td><td></td><td></td><td>13.82</td><td>92.99</td></tr><tr><td>HelloWorld (Ours)</td><td></td><td></td><td></td><td>17.81</td><td>88.10</td></tr></table>

For long-horizon generation, HelloWorld achieves an FID of 17.81 and an FVD of 88.10, compared with 20.91/94.84 for MagicDrive-V2 and 13.82/92.99 for HorizonDrive. Relative to HorizonDrive, HelloWorld has a 28.9% higher FID and a 5.3% lower FVD, indicating a trade-off between frame-level and video-level distributional quality.

## 5.3 Controllability Analysis

Pose Control. We assess pose controllability by conditioning the model on counterfactual future ego trajectories. Given the same scene prefix, we specify alternative pose sequences for left-turn and right-turn maneuvers that differ from the nominal future trajectory of the original sequence. As illustrated in Figure 4, the generated videos exhibit viewpoint transitions that are consistent with the prescribed pose conditions, while preserving realistic scene geometry and traffic dynamics. Importantly, these trajectories are not standard continuations of the observed motion, but counterfactual interventions on ego motion. The results therefore show that our model can go beyond deterministic future prediction and instead synthesize geometrically coherent counterfactual futures, enabling explicit simulation of alternative driving behaviors.

Left turn  
Right turn  
![](images/291251849b2ced072a882919196ef831b9e8752f73636606f46dcfee9c7df4b6.jpg)  
Figure 4. Counterfactual pose control results. Given the same scene context, our model generates plausible future observations conditioned on counterfactual left-turn and right-turn ego-pose trajectories, demonstrating flexible and geometrically coherent control over ego motion.

In addition to regular left-turn and right-turn controls, we further evaluate two extreme counterfactual lane-change trajectories. In particular, the rightward trajectory crosses the roadside green belt into the non-motorized lane, wherea the leftward trajectory crosses the road barrier into the opposite-direction lane. The generated results in Figure 5 show that the model can still follow these unconventional pose interventions and produce corresponding viewpoint changes while preserving coherent scene structure. This further verifies the flexibility of the proposed pose control under highly unusual ego-motion conditions

Lane change left  
Lane change right  
![](images/15dcabc91cc6b6d1fd4c7c338246fc76e67ebd1b0485072fe8bc4e546c6e1218.jpg)  
Figure 5. Unconventional lane-change control. We condition the model on two unconventional lane-change trajectories: a rightward lane change that crosses the roadside green belt into the non-motorized lane, and a leftward lane change that crosses the barrier into the opposite-direction lane. The generated results remain consistent with the prescribed pose conditions while preserving coherent scene structure.

Scale Control with Pose Conditioning. We study whether explicit pose input can improve geometric consistency and scale controllability in video generation. Our framework supports two input modes: control video only and control video together with pose. We compare these two variants with Cosmos transfer 2.5, which does not take pose as input. As illustrated in Figure 6, the control-video-only setting already captures the rough structure of the target scene, showing that the control video provides useful layout guidance. However, its estimated trajectory still departs from the ground truth, indicating limited control over ego-motion scale. By contrast, when pose is additionally provided, the generated sequence exhibits substantially better alignment with the ground-truth observations, both visually and in the recovered trajectory. In particular, the Layout + Pose trajectory remains consistently closer to the ground truth than the other two baselines, demonstrating that explicit pose conditioning effectively reduces scale drift and improves geometric fidelity during generation.

![](images/51c31b740add14e86661e242457235fcdcafce1277e649676c1d3c835b776c33.jpg)  
Figure 6. Comparison of scale control under different conditioning inputs. Our method supports both control video only and control video + pose. Compared with Cosmos transfer 2.5 and the control-video-only variant, adding pose leads to more accurate geometry, better scale consistency, and trajectories closer to the ground truth. Camera poses are estimated using Pi3X.

Environment Control. We further evaluate the model’s controllability over diverse environmental conditions, including weather and illumination. As shown in Figures 7 and 8, given similar driving contexts, the model can synthesize visually consistent scenes under a wide range of conditions, including snowy, rainy, night, night-and-rainy, golden-hour, and blue-hour scenarios. The generated results exhibit condition-specific appearance changes, such as snow coverage, wet road surfaces and reflections, low-light illumination, and characteristic color tones at different times of day, while largely preserving the underlying road structure, traffic layout, and scene semantics. Notably, the model also supports compositional control, as demonstrated by the night-and-rainy condition, where both nighttime illumination and rainy-weather effects are simultaneously reflected in the generated observations. These results demonstrate that the model can generate diverse and plausible driving environments while maintaining scene-level consistency.

Special Scene Control. We further evaluate the model on special traffic scenes, which correspond to rare or long-tail scenarios that are less frequently observed in standard driving data. As shown in Figure 9, the model can generate plausible future observations conditioned on scene-level semantic descriptions of unusual events, such as traffic accidents, abnormal road occupancy, or uncommon traffic participants. Compared with regular driving scenes, these cases require the model to capture more distinctive semantic cues and maintain consistency between the specified event and the surrounding road context. The generated results show that the model is able to synthesize semantically aligned special scenes while preserving coherent scene geometry, traffic layout, and visual realism. This suggests that the model is not limited to frequent driving patterns, but can also support controllable generation of long-tail and safety-critica scenarios, which is important for stress-testing autonomous driving systems.

![](images/616a4eae762f0cc27d9f4da766d64120e0d632a91b4766a9195c30c8581bdb8d.jpg)  
Figure 7. Given the same underlying driving scenes, our model generates diverse weather conditions, including snowy, rainy, foggy, night-and-rainy, while preserving consistent scene structure and traffic semantics.

![](images/ed12b5a77cc77419e453561c785f54da086677e7a86d6632b279c475b4a5b7d1.jpg)  
Figure 8. Given the same underlying driving scenes, our model generates diverse weather conditions, including daylight, night, golden-hour, and blue-hour scenarios, while preserving consistent scene structure and traffic semantics.

![](images/adff6ba78e232cc95a5e7a814b577430a35b3181d86f4ce2baad0badb946a2fd.jpg)  
Figure 9. Special scene control results. Our model generates plausible long-tail traffic scenarios conditioned on special scene descriptions, demonstrating controllable synthesis of rare and safety-critical events while maintaining coheren scene structure and traffic context.

## 5.4 Few-step Distillation

The post-trained seven-view generator is a 20-step rectified-flow (RF) teacher with classifier-free guidance CFG=3. We distill it into a 4-step student with CFG=1 in two stages. dCM first trains a consistency student from the RF teacher. DMD then continues from that dCM student: the RF model remains the frozen teacher and fake-score network, and the student is trained with score-matching under the same causal KV rollout used at inference. At test time both student use the distilled 4-step schedule and a single conditional forward (no CFG branch).

Quality is measured on a frozen 100-clip set of 100-frame videos, with and without the first-frame condition (w/ ff, w/o ff). Pose (Pi3X) and VBench use the front-wide camera; cross-view consistency (CSE) uses all seven cameras. Inference cost is measured on eleven 720p corner clips (1280×720, 61 frames, 10 fps, first-frame condition) on 8 GPUs with context-parallel size 8. The first clip is warmup and is excluded from the means.

Table 11 and Table 12 separate rotation and cross-view error from translation. With a first frame, dCM increases rotation from 2.027<sup>◦</sup> to 2.568<sup>◦</sup> and CSE from 4.180 to 5.201. Without a first frame both gaps widen $( 3 . 3 3 4 ^ { \circ }  4 . 6 1 9 ^ { \circ }$ 4.892 → 6.399). Translation does not follow rotation: dCM improves Trans<sup>∗</sup> in both settings (4.904 → 2.862, 8.485 → 5.311) and improves Trans with a first frame (1.352 → 1.263), while Trans without a first frame gets worse (1.844→2.907). DMD with a first frame is the best row on Rot, Trans, Trans<sup>∗</sup>, and CSE. Without a first frame it still beats the teacher on the same condition (2.311<sup>◦</sup>, 1.034, 3.193, CSE 4.223).

Table 11. Multiview generation pose errors of few-step distilled models.
<table><tr><td>Setting</td><td>Rot (°)↓</td><td>Trans↓</td><td> ${ \mathrm { T r a n s } } ^ { * } \downarrow$ </td></tr><tr><td>Baseline (w/ ff)</td><td>2.027</td><td>1.352</td><td>4.904</td></tr><tr><td>Baseline (w/o ff)</td><td>3.334</td><td>1.844</td><td>8.485</td></tr><tr><td>dCM (w/ ff)</td><td>2.568</td><td>1.263</td><td>2.862</td></tr><tr><td>dCM (w/o ff)</td><td>4.619</td><td>2.907</td><td>5.311</td></tr><tr><td>DMD (w/ ff)</td><td>1.661</td><td>0.688</td><td>1.885</td></tr><tr><td>DMD (w/o ff)</td><td>2.311</td><td>1.034</td><td>3.193</td></tr></table>

Table 12. Multiview CSE of few-step distilled models.
<table><tr><td>Setting</td><td>CSE↓</td><td>∆CSE↓</td></tr><tr><td>GT video</td><td>1.730</td><td>0.000</td></tr><tr><td>Baseline (w/ ff)</td><td>4.180</td><td>+2.450</td></tr><tr><td>Baseline (w/o ff)</td><td>4.892</td><td>+3.162</td></tr><tr><td>dCM (w/ ff) dCM (w/o ff)</td><td>5.201 6.399</td><td>+3.471 +4.669</td></tr><tr><td>DMD (w/ ff)</td><td>3.510</td><td>+1.780</td></tr><tr><td>DMD (w/o ff)</td><td>4.223</td><td>+2.493</td></tr></table>

Table 13 separates appearance from geometry. dCM lowers dynamic degree (0.9500→0.8800, 0.9700→0.7600) and imaging quality (0.4456→0.4281, 0.5074→0.3824), and the mean falls to 0.7580 / 0.7316. Motion smoothness stays near 0.98 for every row; the teacher with a first frame remains highest at 0.9876. DMD brings the mean back above the teacher (0.7906 / 0.7938 vs. 0.7754 / 0.7901) and is best on dynamic degree (0.9900 / 1.0000). Subject and background improve only without a first frame (0.9209, 0.9488). Aesthetic and imaging without a first frame stay with the teacher (0.4259, 0.5074).

Table 13. VBench metrics of few-step distilled models on multiview benchmark.
<table><tr><td>Setting</td><td>Subject</td><td>Background</td><td>Motion</td><td>Dynamic</td><td>Aesthetic</td><td>Imaging</td><td>Mean</td></tr><tr><td>Baseline (w/ ff)</td><td>0.9089</td><td>0.9462</td><td>0.9876</td><td>0.9500</td><td>0.4138</td><td>0.4456</td><td>0.7754</td></tr><tr><td>Baseline (w/o ff)</td><td>0.9111</td><td>0.9472</td><td>0.9788</td><td>0.9700</td><td>0.4259</td><td>0.5074</td><td>0.7901</td></tr><tr><td>dCM (w/ ff)</td><td>0.8961</td><td>0.9409</td><td>0.9855</td><td>0.8800</td><td>0.4176</td><td>0.4281</td><td>0.7580</td></tr><tr><td>dCM (w/o ff)</td><td>0.9051</td><td>0.9469</td><td>0.9866</td><td>0.7600</td><td>0.4086</td><td>0.3824</td><td>0.7316</td></tr><tr><td>DMD (w/ ff)</td><td>0.9029</td><td>0.9430</td><td>0.9829</td><td>0.9900</td><td>0.4210</td><td>0.5039</td><td>0.7906</td></tr><tr><td>DMD (w/o ff)</td><td>0.9209</td><td>0.9488</td><td>0.9826</td><td>1.0000</td><td>0.4171</td><td>0.4935</td><td>0.7938</td></tr></table>

Table 14 compares the RF teacher (20-step, CFG=3) with the DMD student (4-step, CFG=1) at 720p. DiT ODE time drops from 305.3 s to 30.8 s (9.91×), in line with the tenfold reduction in ODE forwards. Sampling time falls only from 357.7 s to 77.8 s (4.60×), because VAE encoding and decoding take about 32 s on both sides. End-to-end time improves by 2.98×, with essentially matched memory. dCM uses the same 4-step, CFG=1 sampler as DMD, so we report cost on the DMD student.

Table 14. Inference cost of the RF teacher and the DMD student.
<table><tr><td>Phase</td><td>Teacher</td><td>DMD student</td><td>T/S</td></tr><tr><td>Text encode (s)</td><td>2.5</td><td>2.5</td><td>1.0×</td></tr><tr><td>VAE encode RGB (s)</td><td>6.7</td><td>7.1</td><td>0.95×</td></tr><tr><td>VAE encode control (s)</td><td>13.6</td><td>13.6</td><td>1.00×</td></tr><tr><td>VAE decode (s)</td><td>11.9</td><td>11.9</td><td>1.00×</td></tr><tr><td>DiT prefix (s)</td><td>1.1</td><td>1.7</td><td>0.68×</td></tr><tr><td>DiT ODE (s)</td><td>305.3</td><td>30.8</td><td>9.91×</td></tr><tr><td>DiT KV refresh (s)</td><td>12.1</td><td>6.0</td><td>2.02×</td></tr><tr><td>Cache flush (s)</td><td>3.7</td><td>2.3</td><td></td></tr><tr><td>Sampling wall (s)</td><td>357.7</td><td>77.8</td><td>4.60×</td></tr><tr><td>Video write (s)</td><td>68.1</td><td>64.5</td><td></td></tr><tr><td>End-to-end (s)</td><td>426.7</td><td>143.2</td><td>2.98×</td></tr><tr><td>Peak allocated (GiB)</td><td>61.6</td><td>61.9</td><td></td></tr><tr><td>nvidia-smi peak (GiB)</td><td>73.9</td><td>73.3</td><td>一</td></tr></table>

In short: dCM reaches a 4-step sampler but increases rotation and CSE; DMD keeps that sampler and improves pose and CSE beyond the teacher on the matched condition, with a slightly higher VBench mean, matched memory, a 9.9× faster DiT ODE, and 4.60× faster sampling.

## 5.5 LiDAR Reconstruction and Conditional Generation

Reconstruction metrics. Range MAE measures absolute depth error on rays valid in both prediction and reference. Chamfer distance (CD) sums the two directional mean nearest-neighbor distances, using unsquared Euclidean distances in meters [49]. Edge Pred→GT applies the prediction-to-reference distance to a GT-derived boundary band, adapting edge-aware evaluation [50]: range jumps above max(2 m, $0 . 1 r _ { \mathrm { n e a r } } )$ are dilated by two pixels before intersecting the scoring region. Warp Chamfer applies CD to consecutive predicted clouds after ego-motion compensation. Lower is better for these distances; common-valid MAE and one-way edge distance do not measure missing-return coverage.

Ray-Align and Static-Align. We introduce two complementary structural diagnostics:

$$
\mathrm { R a y A l i g n } = \frac { 1 } { | \mathcal { N } | } \sum _ { i \in \mathcal { N } } \mathbf { 1 } \{ | n _ { i } ^ { \top } u _ { i } | < 0 . 2 \} , \qquad \mathrm { S t a t i c a l i g n } _ { t } = \frac { 1 } { | \mathcal { Q } _ { t } | } \sum _ { i \in \mathcal { Q } _ { t } } \mathbf { 1 } \{ | r ( T p _ { t , i } ) - \hat { r } _ { t + 1 } ( \pi ( T p _ { t , i } ) ) | < 0 . 5 \operatorname { m a x } \} .\tag{19}
$$

For $\mathrm { R a y - A l i g n } , u _ { i }$ is the unit sensor ray and $n _ { i }$ is the unit normal from right/down point neighbors. The set $\mathcal { N }$ retains valid triangles with both neighbor distances below 3 m and nondegenerate normals. Lower scores indicate fewer surfaces stretched along sensor rays, although real grazing surfaces also contribute. For Static-Align, $T$ compensates ego motion, r measures range, and π projects into the next range image. The set $\mathcal { Q } _ { t }$ retains in-bounds projections with valid target returns; no occlusion filtering is applied. Higher scores indicate stronger temporal agreement, not necessarily correct geometry or semantic alignment with RGB.

Structural scores are averaged within each clip and then equally across clips. VAE evaluation uses recorded poses; generation evaluation uses the same GT-derived ICP transforms for every method, not independent alignment of predictions. The three-camera comparison restricts both diagnostics to the common GT-derived camera region, whereas the full-panorama results do not. Thus Static-Align is motion-assisted, and neither diagnostic alone establishes camera–LiDAR correspondence.

## 5.5.1 Comparison with Cosmos1

We compare the fleet-adapted Cosmos1 three-camera, single-frame generator with our auxiliary-supervised generator, before and after temporal RRN, on the same 100 recorded clips (2,900 frames). All methods use one shared reference and are scored within the union of the front-wide, left-rear and right-rear camera frusta. Our model retains seven-camera video conditioning; Cosmos1 generates each frame from three cameras. Equalizing the output scoring region does not equalize input information, training budget or architecture.

Table 15. Common three-camera-region evaluation on 100 clips, frames 0–28. All five metrics use the same GT-derived output region; MAE uses pixels valid in all three predictions (82.7671% of regional valid GT). Static-Align uses shared reference-ICP transforms. Distances are in meters and alignment scores are fractions.
<table><tr><td>Metric</td><td>Cosmos1</td><td>Ours</td><td>+ Temporal RRN</td></tr><tr><td>Range MAE (m) ↓</td><td>6.1724</td><td>2.2199</td><td>2.1629</td></tr><tr><td>Chamfer Distance (m) ↓</td><td>2.5709</td><td>1.0134</td><td>1.0426</td></tr><tr><td>Edge Pred→GT (m) ↓</td><td>1.5802</td><td>0.7976</td><td>0.7115</td></tr><tr><td>Ray-Align↓</td><td>0.3776</td><td>0.4245</td><td>0.2757</td></tr><tr><td>Static-Align ↑</td><td>0.4503</td><td>0.6928</td><td>0.8463</td></tr></table>

Our temporally rectified generator improves all five reported measures relative to Cosmos1 under this output-region protocol. Cosmos1 nevertheless has less radial streaking than our unrectified generator according to Ray-Align, despite higher geometric errors and weaker temporal agreement. Within our pipeline, temporal RRN reduces MAE and predicted-boundary error and improves structural agreement, but increases Chamfer from 1.0134 to 1.0426 m; rectification therefore does not improve every geometric measure. All 2,800 reference-ICP pairs pass the fitnes threshold of 0.6, with none discarded.

## 5.5.2 LiDAR VAE reconstruction and rectification

We assess the first training stage independently of conditional generation. Table 16 evaluates one frozen video VAE on 100 clips of 29 frames, without rectification, with single-frame RRN, and with motion-compensated temporal RRN. All three arms use the same windows, reference data and evaluation protocol. Range MAE uses the intersection of predicted and reference return validity with native range gating (common-valid(pred\_validity)), pooled over valid pixels; it is not an error over the interpolated target canvas. Point-cloud and structural metrics are computed from each branch’s gated range maps, independently of the choice of depth-error mask.

Table 16. LiDAR VAE reconstruction on 100 clips (2,900 frames). Range MAE uses common-valid returns; temporal scores use recorded poses. Distances are in meters and alignment scores are fractions.
<table><tr><td>Metric</td><td>VAE</td><td>+ Single-frame RRN</td><td>+ Temporal RRN</td></tr><tr><td>Range MAE (m) ↓</td><td>0.3981</td><td>0.3839</td><td>0.3775</td></tr><tr><td>Chamfer Distance (m) ↓ [49]</td><td>0.3978</td><td>0.3815</td><td>0.3695</td></tr><tr><td>Edge Pred→GT (m) ↓ [50]</td><td>0.2997</td><td>0.2331</td><td>0.2106</td></tr><tr><td>Ray-Align ↓</td><td>0.2967</td><td>0.2610</td><td>0.2589</td></tr><tr><td>Static-Align ↑</td><td>0.8215</td><td>0.8510</td><td>0.8974</td></tr><tr><td>Warp Chamfer (m) ↓</td><td>0.4668</td><td>0.4422</td><td>0.3668</td></tr></table>

Temporal RRN reduces range MAE by 5.2%, Chamfer distance by 7.1%, and edge prediction-to-reference distance by 29.7% relative to the unrectified VAE. Static-Align rises from 82.15% to 89.74% (+7.59 percentage points), while pose-compensated Warp Chamfer falls from 0.4668 to 0.3668 m. These gains do not imply improved return coverage. RRN leaves the validity head unchanged, although corrected depths can cross the range gates. The single-frame and temporal models are separately trained recipes, so their difference is not a controlled ablation of temporal input alone.

## 5.5.3 RGB-conditioned LiDAR generation

Conditional generation with frozen-decoder supervision. Our final DiT results use the selected auxiliary-supervised seven-camera generator, evaluated on 100 clips of 29 frames with 35 sampling steps. Table 17 compares the same saved generator outputs without rectification, with single-frame RRN, and with temporal RRN. This generator uses threechannel azimuth/coverage ray features and retains its original frozen tokenizer, distinct from the final reconstruction tokenizer, preserving the latent/decoder pairing used in training. Both rectifiers are matched to this generation pipeline rather than substituted from the final VAE reconstruction experiment. The four decoded auxiliary terms are described in Section 4.7. This comparison evaluates postprocessing of the selected auxiliary-supervised model; it does not isolate auxiliary supervision from RF-only training

Table 17. Full-panorama RGB-to-LiDAR generation on 100 clips (2,900 frames), without the three-camera-region restriction of Table 15. All arms use the same saved generator outputs and reference data; rectification does not rerun the DiT. Distances are in meters and alignment scores are fractions.
<table><tr><td>Metric</td><td>Generator</td><td>+ Single-frame RRN</td><td>+ Temporal RRN</td></tr><tr><td>Range MAE (m) ↓</td><td>2.2797</td><td>2.2532</td><td>2.2194</td></tr><tr><td>Chamfer Distance (m) ↓</td><td>1.0738</td><td>1.1038</td><td>1.0920</td></tr><tr><td>Edge Pred→GT (m) ↓</td><td>0.8814</td><td>0.8339</td><td>0.7990</td></tr><tr><td>Ray-Align ↓</td><td>0.4366</td><td>0.3238</td><td>0.2822</td></tr><tr><td>Static-Align ↑</td><td>0.6741</td><td>0.7081</td><td>0.8310</td></tr></table>

Ray-Align and Static-Align in this table cover all 29 frames (indices 0–28), comprising 2,900 ray-alignment frames and 2,800 adjacent frame pairs across 100 clips. All ICP pairs pass the minimum fitness threshold of 0.6, with none discarded. Static-Align uses shared transforms estimated from GT LiDAR for all three arms; scores are averaged within each clip and then across clips. These reference-motion-assisted scores use panoramic outputs rather than the three-camera region above. The MAE validity support and point filtering also differ, so values should not be compared directly across the two tables.

Temporal RRN reduces range MAE and edge prediction-to-reference distance. Ray-Align falls from 0.4366 to 0.2822 relative to the generator, while Static-Align rises from 67.41% to 83.10% (+15.69 percentage points). Chamfer improves over single-frame RRN but remains worse than the generator, so these gains do not establish uniformly improved geometry. These are measured generation results, not transferred VAE reconstruction scores.

## 5.5.4 Qualitative results

Figure 10 presents paired reconstructions in four scenes selected for scene diversity rather than reconstruction scores. The camera rows provide visual context only; the VAE reconstructs LiDAR input and is not conditioned on RGB. Each point-cloud row compares the same frame and viewing direction across GT, VAE, single-frame RRN and temporal RRN. Still images illustrate geometry but do not establish temporal stability by themselves.

Figure 11 shows four recorded-road examples from the selected generator with synchronized camera observations, generated range images, and point clouds rectified by temporal RRN. All four examples use the same temporal RRN checkpoint as the temporal arm of Table 17, but are qualitative road examples rather than the 100-clip aggregate evaluation itself. The auxiliary intensity panels come from a separate prediction head, not the range/validity tokenizer.

(a) Overcast daytime | divided road | light traffic  
![](images/0b2e06cb58845a32b1dfc0b6cfc86ba2bb6ce865a4ddbdc8d3a50d4ebe38d23d.jpg)

(b) Daytime | urban traffic | bus and close vehicles  
![](images/eee51adf7ff9af3204344a8c26c25e9677744d2f561292b59cc6d75cb631effd.jpg)

(c) Artificial lighting | tunnel | sparse traffic  
![](images/f110814a01f0476049184497f27243f3db991d3918cfab12a96e42fb3eef502e.jpg)

(d) Night | wet roadway | intersection traffic  
![](images/f13bb95b1d325b5b70369e1080fad1f4710d61b88105cf8c801d6f8e7ef19cea.jpg)  
Figure 10. LiDAR VAE reconstruction across four recorded scenes: (a) overcast divided road, (b) daytime urban traffic, (c) illuminated tunnel, and (d) a wet roadway at night. For each scene, seven synchronized RGB views are followed by point clouds of GT, VAE reconstruction, VAE + single-frame RRN, and VAE + temporal RRN, from left to right. All examples use frame 14 and the same VAE. Point colors encode range with a shared 5–65 m color scale; this scale is not the evaluation range gate. RGB is shown only for scene context.

(a) Temporal RRN, t = 2.8 s  
(b) Temporal RRN, t = 1.4 s  
![](images/415d3131de8a07b926834c261e08b7630bb14eb19a921a0e9d305f24bd13d8db.jpg)  
Figure 11. RGB-to-LiDAR generation with temporal RRN on four recorded-road scenes. Each panel shows six displayed camera views, predicted range, auxiliary intensity, and a range-colored point cloud. Seven views condition generation; forward-far is omitted only from display. All four panels use the same temporal rectifier. Intensity is predicted by a separate head, not GT. These are generated and rectified point clouds, not VAE reconstructions.

LiDAR synthesis from generated videos. Figure 12 shows clear-morning and light-fog examples conditioned on generated RGB, both at t = 2.0 s. Each panel pairs a camera mosaic with the synchronized RRN-refined point cloud. The scenes use different generation seeds; these stills demonstrate application beyond recorded RGB, not a controlled weather ablation or proof of long-horizon consistency. No paired target scan is available for the altered scenes, so the recorded-input scores above are not assigned to these examples.

(a) Clear morning, t = 2.0 s  
![](images/6f60d2e3042bbabf106cd539784be88a1a2b453bf9e712421d7f7b6c4ca40829.jpg)

(b) Light fog, t = 2.0 s  
![](images/0809e21b81dd726fb1f64ead67deb08f9cb0b6df8711577e8fc380a57a702a78.jpg)  
Figure 12. LiDAR synthesis conditioned on generated RGB: selected clear-morning and light-fog frames at t = 2.0 s. Each panel pairs the displayed camera video mosaic with its synchronized rectified point cloud. These examples illustrate use beyond recorded RGB, not measured physical simulation of fog or a controlled weather ablation. Selected stills do not establish long-horizon consistency or temporal-RRN performance.

## 5.6 Super-Resolution Refinement

To further improve the visual fidelity of the generated videos, we employ a super-resolution module on the generated frames. Compared with bicubic interpolation, the proposed super-resolution stage better restores high-frequency details and sharp structural boundaries. As illustrated in Figure 13, fine-scale elements such as lane markings, vehicle fronts, headlights, and license-plate regions become noticeably clearer after super-resolution. Importantly, the enhancemen preserves the global scene layout and object structure while improving local appearance quality, making the generated videos more suitable for high-resolution visualization and downstream analysis.

![](images/5e7c725a55a3a83af478944d66d86a6360d9102f0e24dda150d669a697b8235c.jpg)  
Figure 13. Super-resolution enhancement. Our super-resolution module produces sharper boundaries and richer local details than bicubic upsampling, as highlighted by the zoomed-in lane-marking and vehicle regions.

## 6 Applications

The preceding experiments evaluate the individual capabilities of HelloWorld in terms of generation quality, control fidelity, multi-view consistency, sequential generation, and inference efficiency. We further examine how these capabilities can be combined in downstream driving applications. We focus on two representative settings: specialscene editing, where recorded driving contexts are transformed into counterfactual scenarios through controllable generation, closed-loop simulation, where HelloWorld serves as an observation generator under iteratively updated ego motion. These applications illustrate two complementary uses of the same world model: controllable data generation and interactive observation synthesis beyond recorded driving logs.

## 6.1 Special-Scene Editing

We further evaluate HelloWorld on special-scene editing, where rare or safety-critical events are inserted into recorded driving scenes in a controllable manner. Rather than generating such cases from scratch, we preserve the original road geometry, ego trajectory, and structured scene context, and modify only the semantic or object-level conditions to introduce a target event. This setting tests whether the model can perform scene-consistent counterfactual editing while maintaining geometric and contextual coherence.

As illustrated in Figure 14, HelloWorld can insert challenging traffic events such as a pedestrian stepping out from in front of a parked bus, pedestrians and a motorbike crossing the road ahead, and pedestrian–motorbike conflicts near a right-turn exit. The edited results remain aligned with the original scene layout and multi-view observations, while clearly reflecting the inserted agents and their intended motions. These examples show that HelloWorld is not limited to replaying common driving patterns, but can also support controllable generation of long-tail scenarios for safety-oriented data augmentation and stress testing.

## 6.2 Closed-Loop Simulation

HelloWorld can serve as the observation-generation component of a closed-loop driving simulation. In our implementation, we couple HelloWorld with the open-source Alpamayo 1.5 action model. At each interaction step, Alpamayo 1.5 takes the current multi-view observations generated by HelloWorld and predicts the ego trajectory for the next simulation interval. Conditioned on the committed observation history, the updated ego trajectory, and the available scene conditions, HelloWorld then generates the corresponding multi-view observations for the next temporal chunk. These observations are fed back to Alpamayo 1.5 for subsequent trajectory prediction, forming an iterative action–observation loop. As the predicted trajectory deviates from the recorded trajectory, the simulator correspondingly generates observations along the newly selected ego motion rather than replaying the original driving sequence. Representative closed-loop rollouts are shown in Figures 15 and 16.

A central challenge in this setting is that both components progressively operate outside the recorded trajectory: the action model makes decisions from generated observations, while the world model conditions on its own previously generated history and newly predicted ego motion. The block-causal formulation and self-generated-context adaptation described in Section 4.3 are designed to support this repeated execution regime. Moreover, the closed-loop system is run with thefew-step student model, which substantially reduces the sampling cost of each interaction step and make iterative action–observation rollout practical in simulation.

## 7 Limitations

The model’s scope is bounded by its observation interface, available supervision and finite history.

Observation causality and external dynamics. The trunk generates observations block-causally, but some controltower paths use bidirectional attention over the supplied condition sequence. Strict online operation requires causal access in every conditioning pathway. Ego trajectories are prescribed. Low-level vehicle dynamics, policy feedback and reactive traffic behavior are outside the modeled transition. Generated observations do not constitute a validated closed-loop simulator.

Finite memory and conflicting conditions. A bounded KV cache limits accessible history and does not provide persistent scene memory for loop closure or revisited streets. Text, pose and layout can also conflict, without a guaranteed priority rule. Long-horizon evaluation must therefore include condition adherence, visibility changes and memory-boundary behavior in addition to visual smoothness.

Compression and observation coverage. Few-step inference can alter both per-chunk quality and the distribution of subsequent history. A favorable short-window trade-off does not guarantee equivalent long-horizon behavior. Native RGB resolution is 480 × 832. Image refinement adds a separate quality and latency trade-off. Layout response is limited by annotation coverage, particularly in unlabeled or occluded regions.

Geometric extension. The conditional LiDAR generator depends on the quality and calibration of its RGB inputs. A fleet-mean conditioning rig is not equivalent to per-clip calibration. Validity and range compression can lose thin structures, boundaries or rare returns. Recorded-camera and generated-camera conditioning expose different input distributions. Success with recorded RGB does not establish equivalent quality with generated RGB.

## 8 Conclusion

HelloWorld organizes driving observation generation around a common block-causal transition. A supervision-driven curriculum progressively adds calibrated multi-view communication and structured scene control to a pose-conditioned video model. Generated-history alignment addresses the states encountered during repeated execution, while consistency and distribution-matching distillation target few-step sampling of the same transition.

The framework makes capability retention across these transformations a central evaluation question: appearance coverage, cross-view geometry and condition response must be assessed together with horizon and execution cost. Conditional LiDAR synthesis extends the observation interface through explicit return validity, geometric compression and native-grid multiview conditioning. The resulting formulation connects progressive capability learning with efficient scene continuation and counterfactual data synthesis. Persistent memory, fully causal conditioning and policy-coupled dynamics remain directions for extending the observation model.

## References

[1] Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[2] NVIDIA. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

[3] Google DeepMind. Genie 3: A new frontier for world models, 2025. https://deepmind.google/discover/ blog/genie-3-a-new-frontier-for-world-models/.

[4] Tencent Hunyuan. Hunyuanworld 1.5 (worldplay): A streaming interactive world model, 2025. Technical report, December 2025.

[5] LingBot Team. Lingbot-world: An open real-time interactive world model. arXiv preprint arXiv:2601.20540, 2026.

[6] Skywork AI. Matrix-game 2.0: An open-source real-time interactive world model, 2025. Technical report.

[7] Alexander Swerdlow, Runsheng Xu, and Bolei Zhou. Street-view image generation from a bird’s-eye view layout. IEEE Robotics and Automation Letters, 2024.

[8] Kairui Yang, Enhui Ma, Jibin Peng, Qing Guo, Di Lin, and Kaicheng Yu. Bevcontrol: Accurately controlling street-view elements with multi-perspective consistency via bev sketch layout. arXiv preprint arXiv:2308.01661, 2023.

[9] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023.

[10] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. In ICLR, 2024.

[11] Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. In CVPR, 2024.

[12] Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, et al. Uniscene: Unified occupancy-centric driving scene generation. In CVPR, 2025.

[13] Ruiyuan Gao, Kai Chen, Bo Xiao, Lanqing Hong, Zhenguo Li, and Qiang Xu. Magicdrive-v2: High-resolution long video generation for autonomous driving with adaptive control. arXiv preprint arXiv:2411.13807, 2024.

[14] Junpeng Jiang, Gangyi Hong, Lijun Zhou, Enhui Ma, Hengtong Hu, et al. Dive: Dit-based video generation with enhanced control. arXiv preprint arXiv:2409.01595, 2024.

[15] NVIDIA. Cosmos-transfer1: Conditional world generation with adaptive multimodal control. arXiv preprint arXiv:2503.14492, 2025.

[16] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

[17] Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.

[18] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. In NeurIPS, 2024.

[19] Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, et al. Generalized predictive model for autonomous driving. In CVPR, 2024.

[20] Xiaotao Hu, Wei Yin, Mingkai Jia, Junyuan Deng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Ping Tan. Drivingworld: Constructing world model for autonomous driving via video gpt. arXiv preprint arXiv:2412.19505, 2024.

[21] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In CVPR, 2024.

[22] Haiguang Wang, Daqi Liu, Hongwei Xie, Haisong Liu, Enhui Ma, Kaicheng Yu, Limin Wang, and Bing Wang. Mila: Multi-view intensive-fidelity long-term video generation world model for autonomous driving. arXiv preprint arXiv:2503.15875, 2025.

[23] Conglang Zhang, Yifan Zhan, Qingjie Wang, Zhanpeng Ouyang, Yu Li, Zihao Yang, Xiaoyang Guo, Weiqiang Ren, Qian Zhang, Zhen Dong, et al. Horizondrive: Self-corrective autoregressive world model for long-horizon driving simulation, 2026. Technical report, 2026.

[24] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024.

[25] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

[26] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, et al. Infinite worlds with versatile interactions. arXiv preprint arXiv:2607.07534, 2026.

[27] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.

[28] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025.

[29] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023.

[30] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.

[31] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024.

[32] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.

[33] Guosheng Zhao, Yaozeng Wang, Xiaofeng Wang, Zheng Zhu, et al. Unidrivedreamer: A single-stage multimodal world model for autonomous driving. arXiv preprint arXiv:2602.02002, 2026. URL https://arxiv.org/abs/ 2602.02002.

[34] Jiahao Wang, Bo Sun, Yijing Bai, Vincent Casser, et al. Sensor2sensor: Cross-embodiment sensor conversion for autonomous driving. arXiv preprint arXiv:2605.22809, 2026. URL https://arxiv.org/abs/2605.22809.

[35] Quan Liu, Xiaoqin Zhang, Ling Shao, and Shijian Lu. L3DR: 3d-aware LiDAR diffusion and rectification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17153–17163, June 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Liu\_L3DR\_ 3D-aware\_LiDAR\_Diffusion\_and\_Rectification\_CVPR\_2026\_paper.html.

[36] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020.

[37] Yuntao Chen, Yuqi Wang, and Zhaoxiang Zhang. Drivinggpt: Unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 26890–26900, 2025.

[38] Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, et al. Epona: Autoregressive diffusion world model for autonomous driving. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 27220–27230, 2025.

[39] Alexander Swerdlow, Runsheng Xu, and Bolei Zhou. Street-view image generation from a bird’s-eye view layout. IEEE Robotics and Automation Letters, 9(4):3578–3585, 2024.

[40] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. In International Conference on Learning Representations, volume 2024, pages 22841–22860, 2024.

[41] Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang, Tiancai Wang, et al. Uniscene: Unified occupancy-centric driving scene generation. In Proceedings ofthe computer vision and pattern recognition conference, pages 11971–11981, 2025.

[42] Jiazhe Guo, Yikang Ding, Xiwu Chen, Shuo Chen, Bohan Li, Yingshuang Zou, Xiaoyang Lyu, Feiyang Tan, Xiaojuan Qi, Zhiheng Li, et al. Dist-4d: Disentangled spatiotemporal diffusion with metric depth for 4d driving scene generation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 27231– 27241, 2025.

[43] Bohan Li, Zhuang Ma, Dalong Du, Baorui Peng, Zhujin Liang, Zhenqiang Liu, Chao Ma, Yueming Jin, Hao Zhao, Wenjun Zeng, et al. Omninwm: Omniscient driving navigation world models. arXiv preprint arXiv:2510.18313, 2025.

[44] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024.

[45] Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer-2: LLM-enhanced world models for diverse driving video generation. Proceedings ofthe AAAI Conference on Artificial Intelligence, 39(10):10412–10420, 2025. doi: 10.1609/aaai.v39i10.33130.

[46] Yaoru Li, Federico Landi, Marco Godi, Xin Jin, Ruiju Fu, Yufei Ma, Muyang Sun, Heyu Si, and Qi Guo. Far-drive: Frame-autoregressive video generation in closed-loop autonomous driving. arXiv preprint arXiv:2603.14938, 2026.

[47] Ruiyuan Gao, Kai Chen, Bo Xiao, Lanqing Hong, Zhenguo Li, and Qiang Xu. Magicdrive-v2: High-resolution long video generation for autonomous driving with adaptive control. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 28135–28144, 2025.

[48] Conglang Zhang, Yifan Zhan, Qingjie Wang, Zhanpeng Ouyang, Yu Li, Zihao Yang, Xiaoyang Guo, Weiqiang Ren, Qian Zhang, Zhen Dong, et al. Horizondrive: Self-corrective autoregressive world model for long-horizon driving simulation. arXiv preprint arXiv:2605.11596, 2026.

[49] Haoqiang Fan, Hao Su, and Leonidas J. Guibas. A point set generation network for 3D object reconstruction from a single image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 605–613, 2017. URL https://openaccess.thecvf.com/content\_cvpr\_2017/html/Fan\_A\_Point\_Set\_ CVPR\_2017\_paper.html.

[50] Gangwei Xu, Haotong Lin, Hongcheng Luo, Xianqi Wang, Jingfeng Yao, Lianghui Zhu, Yuechuan Pu, Cheng Chi, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Sida Peng, and Xin Yang. Pixel-perfect depth with semantics-prompted diffusion transformers. arXiv preprint arXiv:2510.07316, 2025. URL https://arxiv. org/abs/2510.07316.

![](images/0abfeca380004f61210b67c85e58a772e3d99468d33dca3c973a979f45401bdd.jpg)

![](images/43b1f239b1ef6b9f0cd42817f54db943cdbfc69ce70dd0b5735d3c698ae8b1d2.jpg)

![](images/59a93118e8a8d85785353de04dd13aca5227d6f02893abe2712b7099e402f072.jpg)

![](images/738867483a00cde78e93077a542fb216457705b3b000eff70081f103640b0b0a.jpg)

![](images/68f00c3c29c303b374ad9198bc4ae2b7d20121849a44c23337fd604dbbbe8325.jpg)

![](images/0da37e6c31fcca9a6497a063a73ca687efdae48bfefe692325d08c9e595830a3.jpg)

![](images/f76d983788b8993c90f86b2a5441034bb6cc2e16180c56705edcc66fb3b8423c.jpg)

![](images/72fdb7d3567cb078bc2c43b2701e9908c05cce844690f17d9586deb386d4942e.jpg)  
Figure 14. Special-scene editing results. Starting from recorded driving scenes, HelloWorld preserves the original road geometry, ego trajectory, and surrounding scene context while introducing rare traffic agents or events through edited semantic and object-level conditions. Each example shows the edited top-down BEV together with the corresponding synchronized multi-view observations. The edited scenarios include vehicle trajectory changes, object insertion, and safety-critical interactions involving pedestrians, motorbikes, and surrounding vehicles, demonstrating scene-consistent counterfactual editing under realistic driving contexts.

![](images/d1f09a1fbe6b9b15bbace3432d96cf9eaf098f20579c1fdc0d9fd7cce055aabd.jpg)  
Figure 15. Closed-loop simulation with Alpamayo 1.5 under trajectory adjustment. The left side shows the recorded road-test replay, while the right side shows the HelloWorld simulation driven by Alpamayo 1.5. At each step, Alpamayo 1.5 predicts an updated ego trajectory from HelloWorld-generated multi-view observations, and HelloWorld synthesizes the corresponding next observations. The rollout is executed with the few-step HelloWorld student model, enabling efficient repeated simulation. In this example, the simulated trajectory gradually deviates from the recorded path while remaining consistent with the surrounding road structure and scene context.

![](images/3e63cfb79e4a1eaf71fad0b42f8a3940b7e904b6dc50556d3a090a062214d627.jpg)  
Figure 16. Closed-loop simulation with Alpamayo 1.5 for obstacle avoidance. The left side shows the recorded road-test replay, while the right side shows the corresponding closed-loop simulation. Alpamayo 1.5 observes the multi-view frames generated by HelloWorld and predicts a lateral avoidance trajectory around a stopped vehicle. HelloWorld then generates the subsequent observations conditioned on the updated ego motion. The rollout uses the few-step HelloWorld student model and demonstrates interactive world-model simulation under action-model feedback.

## A HelloWorld Team

## Core Contributors

Fan Lu, Hanshi Wang, Zijing Wang, Quan Feng, Zhi Wang, Shijie Chen, Xianming Zeng

## Contributors

YuJian Zhang, Jiazhe Wang, Xin Zha, Kai Wang, Zhijie Zhao, Lin Zhu, Tianyi Yang, Yucheng Xu, Tao Ji, Haodong Zhang, ZhiPeng Zhang, Peixi Peng, Guang Chen

## Project Leader

Jianyun Xu

## Corresponding Authors

Jianyun Xu, Xingliang Liu, Lei Yang