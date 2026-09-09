# From Where to How: Continuous 4D Interaction Forecasting from Egocentric Video

Qiaohui Chu, Haoyu Zhang, Meng Liu, Member, IEEE, Haoxiang Shi, Dongmei Jiang, and Liqiang Nie, Senior Member, IEEE

Abstract—Egocentric 4D interaction forecasting aims to anticipate both where future interactions will occur in 3D and how the human body will move to realize them, providing an important capability for assistive robotics and human-computer interaction. Existing methods struggle to translate semantic understanding into precise continuous 3D localization and to balance motion diversity with structural consistency in pose forecasting. More fundamentally, these tasks are often modeled separately, leaving the continuous geometric and temporal correspondence between interaction locations and body motion insufficiently captured. To address these challenges, we introduce Coherent4D, a largescale egocentric dataset for continuous 4D interaction forecasting, comprising approximately 233K samples across three domains. Each sample pairs a sequence of future 3D interaction locations with corresponding full-body poses, aligned in time and expressed in a shared coordinate system. We also provide evaluation metrics in continuous space. Building on this formulation, we propose HIGFlow, a Hand Interaction Guided Residual Flow framework that models forecasting as a cascaded where-to-how process. HIGFlow first forecasts continuous future interaction locations by combining semantic grounding with short-horizon visual dynamics, and then uses the predicted location sequence to condition a deterministic motion anchor and residual Flow Matching for diverse yet structurally consistent full-body motion forecasting. Extensive experiments across all three domains demonstrate consistent improvements over representative baselines on both location and pose forecasting, while ablations validate the contributions of the proposed components. The project page is available at https://corrineqiu.github.io/from-where-to-how/.

Index Terms—4D interaction forecasting, egocentric video, interaction location forecasting, full-body pose forecasting.

## I. INTRODUCTION

E <sup>GOCENTRIC</sup> <sup>4D</sup> <sup>interaction</sup> <sup>forecasting</sup> <sup>aims</sup> <sup>to</sup> <sup>jointly</sup>anticipate where future interactions will occur in 3D anticipate where future interactions will occur in 3D space and how the human body will move to execute them. This capability is fundamental to proactive embodied intelligence, where effective anticipation, planning, and assistance require a temporally continuous and geometrically consistent understanding of future locations and their associated body motions. It can support applications such as task assistance [1], risk warning [2], and human-robot collaboration [3].

Despite the inherently coupled nature of interaction location and body motion, existing research has largely approached them as two separate forecasting problems. For interaction location forecasting, existing methods predict the spatial targets or trajectories of upcoming interactions from egocentric observations, including continuous hand motion regression using temporal or state-space models [4], generative modeling of multiple plausible hand trajectories [5], and semantic or language-guided forecasting for object grounding and procedural reasoning [6]–[9]. In parallel, full-body pose forecasting predicts temporally evolving human poses from motion history and contextual cues. Existing methods leverage pose history, egocentric observations, scene context, robot-view cues, or map-aware representations [10]–[14], while stochastic diffusion- or flow-based methods model multiple feasible future motions [15]–[17] and skeleton-aware approaches incorporate structural or kinematic priors to preserve valid body configurations [18]. Together, these two lines of research provide complementary capabilities for predicting where future interactions occur and how the body moves to realize them, but do not explicitly model their temporal and geometric correspondence.

FIction [19] represents an early effort to bridge this gap by connecting interaction localization and pose forecasting within a unified dataset and task formulation. It represents future interaction locations as discrete voxel occupancy and predicts poses conditioned on candidate locations, thereby introducing an explicit dependency between the two predictions. However, interaction locations and full-body poses are not modeled as continuous, temporally aligned sequences in a shared metric space. As a result, the stepwise correspondence between the evolving interaction target and the body motion that realizes it remains unresolved.

This limitation exposes a more fundamental gap: existing studies have not yet established a continuous and jointly grounded 4D forecasting formulation that preserves the temporal and geometric correspondence between future interaction locations and full-body motion. Addressing this gap involves four key challenges: ❶ Decoupled task formulation. Even when location and motion cues are jointly available, existing predictors typically treat interaction localization and pose forecasting as independent or loosely connected objectives. Consequently, continuous predictions of interaction locations are not explicitly propagated as geometric constraints for subsequent motion forecasting, limiting coordination between where an interaction develops and how it is physically realized. ❷ Spatiotemporal pairing gap. Existing datasets rarely provide ordered, continuous 3D hand interaction locations paired with full-body poses at matched future timestamps in a shared metric coordinate system. Without such temporally synchronized and spatially co-registered targets, models cannot directly learn or evaluate how interaction locations and body motion co-evolve over time. ❸ Semantic-dynamic localization gap. Accurate location forecasting requires both task-level semantic grounding and precise continuous 3D localization over future time steps. However, representations from vision-language models (VLMs) are primarily optimized for semantic reasoning rather than metric coordinate regression and provide limited modeling of short-horizon visual dynamics. This mismatch can produce semantically plausible yet spatially inaccurate location predictions. ❹ Pose diversitystructure trade-off. Deterministic pose forecasting tends to preserve structural consistency but suppresses alternative feasible executions, while stochastic generation captures multiple feasible futures but may compromise skeletal structure and joint consistency. Effective forecasting therefore requires motion diversity to be modeled without sacrificing structural plausibility.

![](images/890ab3b8c16b45a70e746fa997ca6387b719e2ee0d56d3269317c87001acd0b6.jpg)  
Fig. 1. Task distribution of Coherent4D. Sequences are associated with the corresponding task annotations to summarize the procedural coverage of the dataset.

To address these challenges, we introduce Coherent4D, a large-scale egocentric dataset for continuous 4D interaction forecasting. As shown in Fig. 1, Coherent4D contains approximately 233K samples spanning Cooking, Health, and Bike Repair. Unlike existing datasets that provide location and motion supervision separately or through discrete spatial representations, each sample in Coherent4D contains two temporally synchronized targets in a shared 3D coordinate system: an ordered sequence of continuous future interaction locations describing where the interaction evolves over time, and a corresponding full-body pose sequence describing how the body realizes it. This paired formulation establishes explicit temporal and geometric correspondence between interaction localization and motion realization. We further introduce continuous-space evaluation metrics to assess both forecasts at corresponding future time steps. Based on this coupled formulation, we propose HIGFlow, a cascaded framework for continuous 4D interaction forecasting that follows a structured where-to-how process. HIGFlow first forecasts the continuous spatial progression of future locations and then uses these forecasts to condition the corresponding full-body motion, thereby explicitly coupling interaction localization with motion realization. For location forecasting, we develop Semantic-Dynamic Location Forecasting, which decouples task-level semantic grounding from continuous metric localization while incorporating short-horizon latent visual dynamics to improve future location prediction. For full-body pose forecasting, we introduce Hand-Conditioned Residual Flow Matching, which first constructs a deterministic motion anchor conditioned on the predicted future interaction locations and then models bounded stochastic residuals around this anchor. This design combines a geometrically grounded and structurally stable motion estimate with stochastic variation, enabling diverse future pose predictions while preserving skeletal structure and joint consistency. Extensive experiments validate the soundness of the Coherent4D dataset and the effectiveness of HIGFlow in jointly modeling interaction localization and fullbody motion realization.

Our main contributions are summarized as follows:

• We introduce Coherent4D, a large-scale egocentric dataset with approximately 233K samples, pairing ordered future 3D interaction locations with temporally aligned full-body poses in a shared coordinate system.

• We propose HIGFlow, a cascaded where-to-how framework that uses predicted interaction locations as geometric conditions for full-body pose forecasting, explicitly coupling interaction localization with motion realization.

• We develop Semantic-Dynamic Location Forecasting for accurate continuous interaction localization and Hand-Conditioned Residual Flow Matching for diverse yet structurally consistent full-body motion forecasting.

• Extensive experiments demonstrate the effectiveness of Coherent4D and HIGFlow, as well as the benefit of jointly modeling future interaction locations and fullbody motion.

## II. RELATED WORK

## A. Datasets for 4D Interaction Forecasting

Large-scale egocentric datasets such as EPIC-KITCHENS and Ego4D provide broad supervision for action understanding and anticipation [20], [21], but do not pair ordered continuous 3D interaction locations with temporally synchronized fullbody poses. Hand-centric datasets support forecasting future hand motion or interaction targets from egocentric observations. EgoPAT3D [22] and EgoPAT3Dv2 [23] focus on 3D action target forecasting, while EgoHandTrajPred, introduced with USST [4], provides ordered annotations for future 3D hand trajectory prediction. More recent datasets incorporate richer semantic and state information: EgoH4 [5] supports forecasting future 3D trajectories and poses of both hands, EgoHaFL [6] provides language-guided annotations of future hand states, poses, and trajectories, and EgoMAN [7] supports 6-DoF hand trajectory forecasting with location awareness, using structured semantic, spatial, and motion cues. In contrast, pose-centric datasets focus on future full-body motion. MoGaze [24] and GIMO [10] incorporate workspace geometry, scene scans, egocentric observations, and gaze, while HARPER [11] and Real-IM [12] extend pose forecasting to robot-centric and map-aware settings. However, hand-centric datasets generally lack temporally aligned full-body pose sequences, whereas pose-centric datasets rarely annotate future interaction moments. Consequently, existing datasets cannot directly associate future interaction locations with the corresponding full-body motion or evaluate whether the predicted body reaches the intended interaction target at the correct time.

![](images/5945095a9607890094b6accbce17614e0c6d80758cce08df23a30fe8ab8470b8.jpg)  
Fig. 2. Representative subset of interaction events from a Coherent4D sample. Synchronized exocentric source frames on both sides provide visual references for the selected events. The center shows the scene geometry, object bounding boxes, and temporally ordered SMPL body states in the shared sample-local coordinate frame. The action timeline summarizes the corresponding interaction objects, while colors associate each event with its rendered body state and reference frame.

FIction [19] is the closest prior dataset linking future interaction localization with body pose forecasting. However, it represents locations as voxel occupancy and conditions pose prediction on individual candidate locations, rather than providing continuous, temporally aligned sequences of interaction locations and full-body poses. Coherent4D fills this gap by pairing ordered 3D interaction locations with synchronized full-body poses parameterized by the Skinned Multi-Person Linear (SMPL) [25] model in a shared coordinate system, enabling joint evaluation of interaction localization and motion execution.

## B. 3D Interaction Location Forecasting

3D interaction location forecasting predicts ordered metric hand interaction locations from egocentric observations, specifying where upcoming hand-environment interactions will occur. Related egocentric studies address long-term anticipation and next active object prediction [26]–[28]. Early hand trajectory forecasting methods mainly operate in the 2D image space. OCT predicts future 2D hand trajectories with interaction hotspots, while Diff-IP2D and MADiff use diffusion- or dynamics-aware modeling to capture future uncertainty and ego-motion effects [29]–[31]. Recent works extend hand forecasting to metric 3D space: USST establishes egocentric 3D hand trajectory forecasting from RGB observations, MMTwin extends the MADiff-style diffusion paradigm to multimodal 3D hand trajectory forecasting, and Uni-Hand further unifies 2D/3D hand waypoint forecasting with richer hand motion and interaction targets [4], [32], [33]. Although these methods advance hand motion forecasting from image-space trajectories to metric 3D waypoints, they still primarily model future hand motion itself, without sufficiently integrating task-level semantic grounding, continuous metric regression, and shorthorizon visual dynamics for ordered future interaction location forecasting.

Semantic and VLM-assisted methods introduce objectcentric grounding, action semantics, and procedural context [34]–[37]. HandsOnVLM shows that directly serializing coordinates as language is insufficient, and instead uses dedicated hand tokens with a trajectory decoder [38]. Predictive video models and latent world representations further capture short-horizon spatiotemporal dynamics [39], [40]. However, semantic reasoning, continuous metric regression, and predictive visual dynamics are still rarely integrated in a unified interaction location predictor. To address this limitation, our Semantic-Dynamic Location Forecasting separates task-level VLM grounding from coordinate decoding and augments the future position representation with short-horizon visual dynamics, improving goal consistency while preserving deterministic continuous 3D regression.

## C. Full-Body Pose Forecasting

Full-body pose forecasting predicts future global motion and articulated body configurations from observed pose histories and contextual cues [41], [42]. Prior methods improve motion stability using gaze, workspace context, spatiotemporal anchors, global trajectories, or affordance cues. MoGaze incorporates gaze and scene context, STARS separates deterministic anchors from within-mode variation, T2P conditions local pose forecasting on predicted global trajectories, and GAP3DS uses gaze-informed affordance cues in 3D scenes [24], [43]– [45]. These methods improve scene consistency and stable coarse motion, but deterministic or anchor-based forecasting often suppresses alternative feasible executions when the same interaction can be realized by different body motions.

Other methods improve motion diversity through latent variables, diffusion, flow matching, motion fields, skeletonaware generation, or residual refinement [46]–[48]. SLD-HMP learns controllable semantic latent directions, BeLFusion and CoMusion improve diverse motion generation and history consistency, SkeletonDiffusion incorporates structural priors for anatomical plausibility, and PrediFlow refines coarse motion forecasts with Flow Matching residuals [18], [49]–[52]. While these methods enhance diversity or realism, noise-initialized generation can still introduce inefficient sampling, temporal instability, or implausible articulation when stochastic variation is not sufficiently constrained. Our Hand-Conditioned Residual Flow Matching addresses this diversity-stability trade-off by first establishing a hand-conditioned deterministic SMPL motion anchor and then modeling bounded stochastic residuals around it.

## III. COHERENT4D DATASET

Coherent4D is constructed from Ego-Exo4D [53] to support continuous 4D interaction forecasting from egocentric video. Here, continuous refers to both the temporal continuity of motion sequences and the use of continuous 3D coordinates rather than discrete spatial grids. It pairs ordered 3D interaction locations with SMPL-parameterized full-body pose sequences at matched future timestamps in a shared sample-local coordinate frame, jointly capturing the spatial progression of future locations and the corresponding body motion. Each sample contains 30 egocentric frames uniformly sampled from a 30- second observation window, structured object and environment descriptors, histories of observed interaction locations and SMPL poses, and the corresponding future interaction location and pose targets. Fig. 2 illustrates representative interaction events from a Coherent4D sample. Each event associates a continuous 3D interaction location with the corresponding SMPL body state at the same timestamp, explicitly preserving their temporal and spatial correspondence.

## A. Annotation Pipeline

We construct Coherent4D from synchronized procedural takes in Ego-Exo4D [53]. The Aria egocentric stream serves as the forecasting input, while synchronized exocentric views are used only for offline annotation construction and refinement.

![](images/3bb03ae6deb21e38794ddfa071dfcfbdf8436485ccfedae625a1fe1d62bf88e9.jpg)

![](images/5fdc20e1bf1b2f6af5d794e2ec243aa20553fec178bd053aa50c4def9e310b4d.jpg)  
Fig. 3. Overview of the Coherent4D data construction pipeline. We (1) ground scene objects with semantic labels and 3D bounding boxes, (2) transform scene, location, and body annotations into a shared sample-local coordinate frame, (3) construct temporally ordered 3D interaction location sequences, (4) attach time-aligned SMPL states, and (5) organize them into forecasting samples with location-pose histories, future targets, and relative timestamps.

To establish temporally and geometrically coupled supervision, we organize the source annotations into ordered interaction location and SMPL pose sequences in a shared samplelocal coordinate frame. As illustrated in Fig. 3, the pipeline consists of five stages: scene object grounding, shared coordinate construction, location sequence construction, SMPL state attachment, and forecast sample generation.

1) Scene Object Grounding: We construct object-level semantic and geometric context for interaction location forecasting. We apply Detic [54] with the LVIS vocabulary [55] to detect candidate objects in the egocentric frames. The detected regions are lifted into 3D using the Ego-Exo4D SLAM reconstruction and subsequently clustered to obtain oriented 3D bounding boxes. For each object, we retain its semantic category together with continuous 3D bounding box attributes, including center, size, and orientation. The spatial attributes are subsequently expressed in the shared samplelocal coordinate frame defined below, while the box dimensions retain their metric scale. This representation preserves both object semantics and continuous scene geometry without discretizing the environment into voxels.

2) Shared Coordinate Construction: Geometric coupling between interaction location, body motion, and scene geometry requires all spatial quantities to be represented in a common coordinate frame. For each sample $n ,$ we define a stable egocentric reference pose $( { \bf R } _ { n } ^ { \mathrm { r e f } } , { \bf t } _ { n } ^ { \mathrm { r e f } } )$ . Given a point $\mathbf { p } ^ { \mathrm { w } }$ in the world coordinate frame, its representation in the shared sample-local coordinate frame is defined as:

$$
\mathbf { p } _ { n } ^ { \mathrm { l o c } } = \left( \mathbf { R } _ { n } ^ { \mathrm { r e f } } \right) ^ { \top } \left( \mathbf { p } ^ { \mathrm { w } } - \mathbf { t } _ { n } ^ { \mathrm { r e f } } \right) .\tag{1}
$$

We apply this transformation to interaction locations, object centers, SMPL root translations, and 3D body joints. Global orientations, including the SMPL root orientation and object orientations, are rotated by the same reference rotation, whereas the 23 SMPL body joint rotations remain defined relative to their parent joints. For model input and supervision, positional quantities are divided by $s = 5$ m and then clipped to [−1, 1] for each coordinate. Division by s rescales the coordinates without changing the reference frame, whereas clipping limits values outside this range to the corresponding boundary.

For model input and supervision, coordinates are normalized by dividing by $s ~ = ~ 5 \mathrm { m }$ and clipping to [−1, 1], which rescales their values without changing the reference frame. Root translations and translation residuals remain in meters for pose residual computation, clipping, and composition.

3) Location Sequence Construction: We convert sparse hand-object interaction annotations into temporally ordered continuous 3D interaction location sequences. Following FIction [19], candidate interaction events are identified by combining narration timestamps, Llama 3-based object matching [56], and geometric consistency between the hand and the corresponding object. Each valid event retains its timestamp, interaction object when available, and continuous 3D interaction location recovered from the corresponding hand mesh. Interactions involving the right hand or both hands are mapped into a unified interaction stream. Temporally adjacent events with redundant spatial locations are merged according to their timestamps and normalized spatial displacement. The resulting sequence therefore contains distinct interaction locations while preserving their chronological order and continuous spatial evolution.

4) SMPL State Attachment: We then associate each interaction location with the full-body state at the corresponding timestamp. We reconstruct human motion using WHAM [57], select the primary actor track, and align its SMPL trajectory with the Ego-Exo4D scene coordinate system. For every observed and future interaction timestamp, we retrieve the nearest valid SMPL state and transform its global components into the sample-local frame. Each SMPL state contains the root translation, root orientation, body joint rotations, and 3D joint positions. The root translation and 3D joints are thus spatially co-registered with the corresponding interaction location, while body joint rotations remain relative to their parent joints. This step yields temporally paired interaction location and full-body pose sequences for subsequent forecasting.

5) Forecast Sample Generation: Finally, we convert the aligned annotation streams into forecasting samples while retaining the nonuniform timing of interaction events. The ordered sequences are partitioned according to the domainspecific forecasting horizons reported in Table I. For each sample n, we take the first future interaction timestamp as the time origin and denote it by $t _ { n , 1 }$ . The preceding 30 seconds constitute the observation window and provide the egocentric frames, location history, and pose history. For each future step $k \in \{ 1 , \ldots , K \}$ , we retain its absolute timestamp $t _ { n , k }$ , relative timestamp $\Delta t _ { n , k } = t _ { n , k } - t _ { n , 1 } ,$ continuous interaction location, and temporally aligned SMPL state. The relative timestamps preserve the nonuniform intervals between interaction events, while a fixed number of target steps provides a consistent sequence interface for training and evaluation. Incomplete tail sequences are retained using padding and validity masks. Samples with invalid timestamp alignment, object grounding, coordinate transformation, interaction locations, or SMPL attachment are discarded. We split the dataset at the take level to prevent overlap of environments and procedural sequences across the training, validation, and test sets.

TABLE I  
STATISTICS OF COHERENT4D. TRAIN, VAL, TEST, AND TOTAL REPORT THE NUMBERS OF FORECASTING SAMPLES. TARGETS DENOTE VALID FUTURE INTERACTION LOCATIONS, EXCLUDING PADDED STEPS. OBJ. DENOTES INTERACTION OBJECT LABELS, WHILE THE TOTAL ENTRY FOR OBJ. REPORTS THE NUMBER OF UNIQUE LABELS ACROSS THE DATASET.
<table><tr><td>Domain</td><td>Horizon Takes</td><td></td><td>Train</td><td>Val</td><td>Test</td><td>Total</td><td>Targets</td><td>Obj.</td></tr><tr><td>Cooking</td><td>10</td><td>331</td><td>136,079</td><td>15,435</td><td>14,527</td><td>166,041</td><td>1,337,589</td><td>428</td></tr><tr><td>Health</td><td>5</td><td>205</td><td>21,532</td><td>1,899</td><td>4,167</td><td>27,598</td><td>114,964</td><td>170</td></tr><tr><td>Bike Repair</td><td>4</td><td>251</td><td>35,987</td><td>3,150</td><td>1,052</td><td>40,189</td><td>141,633</td><td>197</td></tr><tr><td>Total</td><td></td><td>787</td><td>193,598</td><td>20,484</td><td>19,746</td><td>233,828</td><td>1,594,186</td><td>535</td></tr></table>

![](images/a83c618981ef027fd83df33402e26605f0054f7689f0478ea368b476bc2a8b1f.jpg)  
Fig. 4. Narration verb distribution over future interaction locations. Bars show the share of all future interaction targets for each normalized first narration verb, and colors indicate the contribution from each domain.

## B. Dataset Statistics

As summarized in Table I, Coherent4D contains 233,828 samples derived from 787 unique takes across three domains: Cooking, Health, and Bike Repair. The corresponding forecasting horizons are 10, 5, and 4 interaction steps, respectively. The dataset further covers 535 interaction object categories, providing diverse object-centric procedural activities across the three domains.

To characterize action-level diversity, we associate each valid future interaction target with its corresponding narration. All 1,594,186 valid future targets are aligned with narration annotations. Fig. 4 shows the normalized distribution of the first verb in each narration, covering common manipulation primitives such as pick, place, hold, drop, move, pour, pass, and turn. Cooking accounts for most high-frequency manipulation actions, while Bike Repair and Health contribute complementary domain-specific interaction patterns.

![](images/b0b436b2e16fa17f2c3a05a50f5be5b49b71f6a8360185a5f29d03afb2cc2c27.jpg)  
Fig. 5. Overview of HIGFlow. The first stage forecasts the locations of future interactions from egocentric context, and the second stage forecasts temporall aligned full-body poses conditioned on the predicted location sequence.

## C. Continuous Metrics

Voxel-based accuracy measures whether a discrete spatial cell is correctly activated, but does not quantify continuous 3D localization or motion errors. We therefore introduce continuous-space metrics for both interaction location and fullbody pose forecasting. For evaluation, normalized interaction locations are converted back to metric coordinates in the shared sample-local frame using the position scale s. SMPL root translations and 3D joints are evaluated in the same metric frame. Positional and rotational errors are reported in millimeters and degrees, respectively, with lower values indicating better performance.

Let D denote the evaluation set and K the forecasting horizon. For sample n and future step k, let $m _ { n , k } \in \{ 0 , 1 \}$ indicate whether the paired interaction location and pose targets are valid.

1) Interaction Location Forecasting Metrics: To characterize overall sequence accuracy, upper-tail localization error, and endpoint accuracy, we report average displacement error (ADE), $\mathrm { \ A D E _ { 9 0 } }$ , and final displacement error (FDE), respectively. For sample n and future step $k ,$ let $\mathbf { y } _ { n , k } \in \mathbb { R } ^ { 3 }$ and $\hat { \mathbf { y } } _ { n , k } \in \mathbb { R } ^ { 3 }$ denote the ground-truth and predicted interaction locations, respectively. The localization error at each future step is $d _ { n , k } ^ { \mathrm { l o c } } = \| \hat { \mathbf { y } } _ { n , k } - \mathbf { y } _ { n , k } \| _ { 2 }$ . We compute the ADE by first averaging over the valid future steps of each sample and then across samples:

$$
\mathrm { A D E } = \frac { 1 } { | \mathcal { D } | } \sum _ { n \in \mathcal { D } } \frac { \sum _ { k = 1 } ^ { K } m _ { n , k } d _ { n , k } ^ { \mathrm { l o c } } } { \sum _ { k = 1 } ^ { K } m _ { n , k } } .\tag{2}
$$

We additionally report $\mathrm { \ A D E _ { 9 0 } }$ , defined as the 90th percentile of the per-sample ADE values, to characterize upper-tail localization error. FDE measures localization accuracy at the last valid future step. Let $\kappa _ { n } = \operatorname* { m a x } \{ k : m _ { n , k } = 1 \}$ denote the last valid step of sample n. FDE is then defined as:

$$
\mathrm { F D E } = \frac { 1 } { | \mathscr { D } | } \sum _ { n \in \mathscr { D } } d _ { n , \kappa _ { n } } ^ { \mathrm { l o c } } .\tag{3}
$$

2) Full-body Pose Forecasting Metrics: To assess absolute joint position accuracy, articulated pose accuracy after similarity alignment, global body displacement, and local body rotation, we report mean per-joint position error (MPJPE), Procrustes-aligned mean per-joint position error (PA-MPJPE), root translation error (Root Trans.), and body geodesic error (Body Geo.), respectively. Following FIction [19], we evaluate each SMPL state using the same set of $J _ { \mathrm { p o s } } ~ = ~ 1 9$ body joints. Let $\mathbf { P } _ { n , k } , \hat { \mathbf { P } } _ { n , k } \ \in \ \mathbb { R } ^ { 3 \times J _ { \mathrm { p o s } } }$ denote the ground-truth and predicted joint coordinate matrices for sample n at future step k, with $\mathbf { p } _ { n , k , j } , \hat { \mathbf { p } } _ { n , k , j } \in \mathbb { R } ^ { 3 }$ denoting the corresponding j-th joint coordinates. For compact notation, let $N _ { \mathrm { v a l i d } } ~ =$ $\textstyle \sum _ { n \in { \mathcal { D } } } \sum _ { k = 1 } ^ { K } m _ { n , k }$ denote the total number of valid future steps.

MPJPE measures the average Euclidean distance between corresponding predicted and ground-truth joints over all valid future poses:

$$
\mathrm { M P J P E } = \frac { 1 } { J _ { \mathrm { p o s } } N _ { \mathrm { v a l i d } } } \sum _ { n \in \mathcal { D } } \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { J _ { \mathrm { p o s } } } m _ { n , k } \left\| \hat { \mathbf { p } } _ { n , k , j } - \mathbf { p } _ { n , k , j } \right\| _ { 2 } .\tag{4}
$$

PA-MPJPE evaluates pose accuracy after removing global similarity misalignment. For each valid pose of sample n at future step k, we align the predicted joints to the ground truth

using the optimal similarity transformation:

$$
\begin{array} { r l r } & { } & { \left( s _ { n , k } ^ { \star } , \mathbf { Q } _ { n , k } ^ { \star } , \mathbf { b } _ { n , k } ^ { \star } \right) = \mathrm { P r o c } \mathrm { A l i g n } \left( \hat { \mathbf { P } } _ { n , k } , \mathbf { P } _ { n , k } \right) , } \\ & { } & { \tilde { \mathbf { P } } _ { n , k } = s _ { n , k } ^ { \star } \mathbf { Q } _ { n , k } ^ { \star } \hat { \mathbf { P } } _ { n , k } + \mathbf { b } _ { n , k } ^ { \star } \mathbf { 1 } ^ { \top } , \quad \quad } \end{array}\tag{5}
$$

where $s _ { n , k } ^ { \star } \in \mathbb { R } _ { + } , { \bf Q } _ { n , k } ^ { \star } \in \mathrm { S O } ( 3 )$ , and $\mathbf { b } _ { n , k } ^ { \star } \in \mathbb { R } ^ { 3 }$ denote the optimal scale, alignment rotation, and translation, respectively, and $\tilde { \mathbf { P } } _ { n , k }$ denotes the aligned prediction. The vector $\mathbf { 1 } \in \mathbb { R } ^ { J _ { \mathrm { p o s } } }$ broadcasts the translation to all joints. Let $\tilde { \bf p } _ { n , k , j }$ denote the j-th column of $\tilde { \mathbf { P } } _ { n , k }$ . PA-MPJPE is then defined as:

$$
\mathrm { P A \mathrm { - } M P J P E } = \frac { \sum _ { n \in \mathcal { D } } \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { J _ { \mathrm { p o s } } } m _ { n , k } \left\| \tilde { \mathbf { p } } _ { n , k , j } - \mathbf { p } _ { n , k , j } \right\| _ { 2 } } { J _ { \mathrm { p o s } } N _ { \mathrm { v a l i d } } } .\tag{6}
$$

Root Trans. measures the Euclidean distance between predicted and ground-truth SMPL root translations, reflecting the accuracy of global body displacement. Let $\mathbf { t } _ { n , k } , \hat { \mathbf { t } } _ { n , k } \in \mathbb { R } ^ { 3 }$ denote the ground-truth and predicted SMPL root translations for sample n at future step k, respectively. It is defined as:

$$
\mathrm { R o o t \ T r a n s . } = \frac { 1 } { N _ { \mathrm { v a l i d } } } \sum _ { n \in \mathcal { D } } \sum _ { k = 1 } ^ { K } m _ { n , k } \left\| \hat { \mathbf { t } } _ { n , k } - \mathbf { t } _ { n , k } \right\| _ { 2 } .\tag{7}
$$

Body Geo. measures the rotational discrepancy of the $J _ { \mathrm { r o t } } =$ 23 non-root body joints. Let ${ \bf R } _ { n , k , j } , \hat { \bf R } _ { n , k , j } \in \mathrm { S O } ( 3 )$ denote the ground-truth and predicted local rotation matrices of body joint j, respectively. Their geodesic angular distance is defined as:

$$
d _ { n , k , j } ^ { \mathrm { r o t } } = \operatorname { a r c c o s } \left( \frac { \mathrm { t r } \left( \hat { \mathbf { R } } _ { n , k , j } ^ { \top } \mathbf { R } _ { n , k , j } \right) - 1 } { 2 } \right) ,\tag{8}
$$

where the argument of arccos(·) is clipped to $[ - 1 , 1 ]$ for numerical stability. The mean body geodesic error, excluding the root joint, is

$$
\mathrm { B o d y ~ G e o . } = \frac { 1 8 0 } { \pi J _ { \mathrm { r o t } } N _ { \mathrm { v a l i d } } } \sum _ { n \in \mathcal { D } } \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { J _ { \mathrm { r o t } } } m _ { n , k } d _ { n , k , j } ^ { \mathrm { r o t } } .\tag{9}
$$

To account for the inherent multimodality of future body motion, the model generates multiple plausible pose sequences for each observation. We therefore report Single and Best-5 to evaluate the first candidate and the best candidate among five forecasts, respectively. For the q-th candidate, let $\hat { \mathcal { X } } _ { n } ^ { ( q ) } =$ $\{ \hat { \mathbf { X } } _ { n , k } ^ { ( q ) } \} _ { k = 1 } ^ { K }$ denote the predicted SMPL sequence. Let $\hat { \mathbf { P } } _ { n , k } ^ { ( q ) } \in$ $\mathbb { R } ^ { 3 \times J _ { \mathrm { p o s } } }$ and $\hat { \mathbf { p } } _ { n , k , j } ^ { ( q ) } \in \mathbb { R } ^ { 3 }$ denote its joint coordinate matrix and j-th joint coordinate, respectively. For Best-5, we select for each sample the candidate with the lowest joint position error over all valid future steps:

$$
q _ { n } ^ { \star } = \arg \operatorname* { m i n } _ { q \in \{ 1 , \dots , 5 \} } \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { J _ { \mathrm { p o s } } } m _ { n , k } \left\| \hat { \mathbf { p } } _ { n , k , j } ^ { ( q ) } - \mathbf { p } _ { n , k , j } \right\| _ { 2 } .\tag{10}
$$

All Best-5 pose metrics are computed from the same selected SMPL sequence $\hat { \mathcal { X } } _ { n } ^ { ( q _ { n } ^ { \star } ) }$ . Its joint coordinates are used for MPJPE and PA-MPJPE, while its root translations and body joint rotations are used for Root Trans. and Body Geo., respectively. This ensures that all Best-5 metrics evaluate a consistent motion forecast selected according to joint position accuracy.

Algorithm 1 Training and inference of HIGFlow   
Input: Observed contexts $( \mathcal { C } _ { \mathrm { l o c } } , \mathcal { H } _ { \mathrm { p o s e } } )$ , ground-truth sequences $( { \mathcal { P } } , { \mathcal { X } } ) .$   
validity masks $( \mathcal { M } _ { Y } , \mathcal { M } _ { X } )$ , forecasting horizon $K ,$ Heun steps N<sub>ODE</sub>,   
pose candidates S.   
Output: Interaction location sequence $\hat { y }$ and pose sequences $\{ \hat { \mathcal X } ^ { ( q ) } \} _ { q = 1 } ^ { S }$   
// Training   
Stage 1: Interaction location training   
1: Encode $\mathcal { C } _ { \mathrm { l o c } }$ with Qwen3-VL and extract $\{ \mathbf { h } _ { k } ^ { \mathrm { Q } } \} _ { k = 1 } ^ { K }$ at the indexed   
<hand\_traj\_k> positions.   
2: Decode the interaction locations and train the location predictor by   
minimizing $\mathcal { L } _ { \mathrm { l o c } }$ in Eq. (16) using M<sub>Y</sub> .   
3: Encode the observed frames with V-JEPA. Fuse the dynamic features with   
$\{ \mathbf { h } _ { k } ^ { \mathrm { Q } } \} _ { k = 1 } ^ { K }$ and update the V-JEPA adapter by minimizing λ<sub>ADE</sub>L<sub>ADE</sub> +   
$\lambda _ { \mathrm { S 1 } } \mathcal { L } _ { \mathrm { S 1 } }$ using $\dot { \mathcal { M } } _ { Y }$   
Stage 2: Full-body pose training   
4: Encode $( \mathcal { H } _ { \mathrm { p o s e } } , \dot { \mathcal { V } } )$ and obtain the anchor sequence $\{ \hat { \mathbf { X } } _ { k } ^ { \mathrm { a } } \} _ { k = 1 } ^ { K } .$ 1 by   
Eq. (18).   
5: Train the anchor predictor by minimizing $\mathcal { L } _ { \mathrm { a n c h o r } }$ in Eq. (19) using $\mathcal { M } _ { X }$   
6: Compute the residual targets $\{ \mathbf { r } _ { k } ^ { * } \} _ { k = 1 } ^ { K }$ by Eq. (21) and construct   
$\{ \mathbf { c } _ { k } ^ { \mathrm { { f i o w } } } , \mathcal { R } _ { k } \} _ { k = 1 } ^ { K }$ by Eqs. (22) and (23).   
7: for $k = 1$ to K do   
8: Sample τ ∼ U(0, 1) and $\mathbf { z } _ { k } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { r } ^ { 2 } \mathbf { I } _ { 7 5 } ) .$   
9: Compute the flow state $\mathbf { x } _ { \tau , k }$ as $( 1 - \tau ) \mathbf { z } _ { k } + \tau B _ { \rho } ( \mathbf { r } _ { k } ^ { * } ) .$   
10: Accumulate $\mathcal { L } _ { \mathrm { F M } }$ according to Eq. (25) and compute the direct and   
Heun rollout residuals.   
11: end for   
12: Compose the residuals with the anchors and train the flow model by   
minimizing $\mathcal { L } _ { \mathrm { p o s e } }$ in Eq. (30) using $\mathcal { M } _ { X } .$   
// Inference   
13: Apply the trained location predictor to $\mathcal { C } _ { \mathrm { l o c } }$ and obtain $\hat { \mathcal { Y } } = \{ \hat { \mathbf { y } } _ { k } \} _ { k = 1 } ^ { K } .$   
14: Condition on $( \mathcal { H } _ { \mathrm { p o s e } } , \hat { \mathcal { V } } )$ to obtain the anchor sequence and residual flow   
conditions $\{ \hat { \mathbf { X } } _ { k } ^ { \mathrm { a } } , \mathbf { c } _ { k } ^ { \mathrm { f l o w } } , \mathcal { R } _ { k } \} _ { k = 1 } ^ { K } .$   
15: for $q = 1$ to S do   
16: Independently sample the residual priors $\{ \mathbf { z } _ { k } ^ { ( q ) } \} _ { k = 1 } ^ { K } .$   
17: Integrate the residual flows with N Heun steps and compose them   
with the anchors to obtain $\{ \hat { \mathbf { X } } _ { k } ^ { ( q ) } \} _ { k = 1 } ^ { K } .$   
18: Collect the K predicted body states to form $\hat { \mathcal { X } } ^ { ( q ) } = \{ \hat { \mathbf { X } } _ { k } ^ { ( q ) } \} _ { k = 1 } ^ { K } .$   
19: end for   
20: return $\hat { y }$ and $\{ \hat { \mathcal X } ^ { ( q ) } \} _ { q = 1 } ^ { S } .$

## IV. METHOD

## A. Problem Formulation

Given an observed egocentric context, we formulate continuous 4D interaction forecasting as a coupled where-to-how prediction problem: first forecasting an ordered sequence of future interaction locations and then predicting the temporally aligned full-body motion conditioned on these locations.

For interaction location forecasting, the observed context is $\mathcal { C } _ { \mathrm { l o c } } ~ = ~ ( \mathcal { V } _ { \mathrm { o b s } } , \mathcal { E } , \mathcal { O } _ { \mathrm { o b s } } , \mathcal { T } )$ , where $\mathcal { V } _ { \mathrm { o b s } } , \ \mathcal { E } , \ \mathcal { O } _ { \mathrm { o b s } } ,$ , and $\tau$ denote the observed egocentric frames, structured environment descriptors, observed location history, and task prompt, respectively. The future location sequence is predicted as

$$
\begin{array} { r } { \hat { \mathcal { Y } } = f _ { \mathrm { l o c } } ( \mathcal { C } _ { \mathrm { l o c } } ) = \{ \hat { \mathbf { y } } _ { k } \} _ { k = 1 } ^ { K } , } \end{array}\tag{11}
$$

where K is the forecasting horizon and $\hat { \mathbf { y } } _ { k } \in \mathbb { R } ^ { 3 }$ denotes the predicted interaction location at future step k in the normalized coordinate representation. The corresponding ground-truth location sequence is denoted by $\mathcal { Y } = \{ \mathbf { y } _ { k } \} _ { k = 1 } ^ { K }$

For full-body pose forecasting, the model takes the observed pose history ${ \mathcal { H } } _ { \mathrm { p o s e } }$ together with a future interaction location sequence Y<sup>˜</sup>:

$$
\hat { \mathcal { X } } = f _ { \mathrm { p o s e } } \left( \mathcal { H } _ { \mathrm { p o s e } } , \tilde { \mathcal { V } } \right) = \{ \hat { \mathbf { X } } _ { k } \} _ { k = 1 } ^ { K } .\tag{12}
$$

During training, $\tilde { \mathcal { V } } ~ = ~ \mathcal { V }$ is the ground-truth location sequence, whereas during inference, $\tilde { \mathcal { V } } = \hat { \mathcal { V } }$ is predicted by the first stage. The corresponding ground-truth pose sequence is denoted by $\boldsymbol { \mathcal { X } } = \{ { \mathbf { X } } _ { k } \} _ { k = 1 } ^ { K } ,$ , with the same validity mask as the interaction location targets, i.e., $\mathcal { M } _ { X } = \mathcal { M } _ { Y }$ . Each $\hat { \mathbf { X } } _ { k } \in \mathbb { R } ^ { 1 4 7 }$ represents an SMPL body state comprising a 6D root orientation, a 3D root translation, and 23 local body joint rotations represented in continuous 6D form.

## B. HIGFlow Framework

HIGFlow instantiates the above where-to-how formulation with the cascaded architecture shown in Fig. 5. It consists of two specialized components: Semantic-Dynamic Location Forecasting for predicting continuous future interaction locations, and Hand-Conditioned Residual Flow Matching for generating the corresponding full-body motion. The predicted location sequence serves as an explicit geometric condition for motion forecasting, establishing a direct dependency between location progression and body motion realization. Algorithm 1 summarizes the training and inference procedures.

1) Semantic-Dynamic Location Forecasting: The interaction location stage forecasts an ordered sequence of continuous 3D interaction locations by combining high-level semantic grounding with short-horizon visual dynamics. Specifically, Qwen3-VL [58] encodes the observed egocentric context and provides semantic representations for future steps, while a frozen V-JEPA [59] encoder supplies complementary motionsensitive features for dynamic refinement.

Semantic context encoding. Qwen3-VL receives the sampled egocentric frames, task prompt, observed location history, and structured environment descriptors. The location and environment encoders transform $\mathcal { O } _ { \mathrm { o b s } }$ and E into dense features, which replace their corresponding placeholder embeddings before the Qwen3-VL forward pass. For each future step $k ,$ we extract the hidden state at the indexed placeholder <hand $\_ \tt t r a j \_ k >$ as the step-specific semantic representation $\mathbf { h } _ { k } ^ { \mathrm { Q } }$

Dynamic feature augmentation. Semantic representations provide task- and object-level grounding but may not sufficiently capture short-term hand-object dynamics that are important for precise spatial forecasting. We therefore encode the observed video with V-JEPA [59] and project its latent features into the Qwen hidden space, followed by resampling into M dynamic memory tokens. Each future step representation $\mathbf { h } _ { k } ^ { \mathrm { Q } }$ augmented with a learned step embedding $\mathbf { s } _ { k } ^ { \mathrm { l o c } }$ , attends to this dynamic memory to obtain the motion context $\mathbf { c } _ { k }$ . We then inject the dynamic information through a gated residual adapter:

$$
\left\{ \begin{array} { l l } { \mathbf { h } _ { k } ^ { \mathrm { F } } = \mathbf { h } _ { k } ^ { \mathrm { Q } } + g _ { k } \Delta \mathbf { h } _ { k } , } \\ { \Delta \mathbf { h } _ { k } = \alpha \operatorname { t a n h } \left( D \left( [ \mathbf { h } _ { k } ^ { \mathrm { Q } } , \mathbf { c } _ { k } , \mathbf { s } _ { k } ^ { \mathrm { l o c } } ] \right) \right) , , } \\ { g _ { k } = \mathrm { s i g m o i d } \left( G \left( [ \mathbf { h } _ { k } ^ { \mathrm { Q } } , \mathbf { c } _ { k } , \mathbf { s } _ { k } ^ { \mathrm { l o c } } ] \right) \right) } \end{array} \right.\tag{13}
$$

where D and G denote lightweight residual and gating heads, respectively. The scalar $g _ { k } ~ \in ~ ( 0 , 1 )$ adaptively controls the contribution of the dynamic residual, while $\alpha > 0$ bounds its magnitude.

Continuous coordinate decoding. A coordinate decoder maps each fused representation $\mathbf { h } _ { k } ^ { \mathrm { F } }$ directly to the normalized continuous 3D interaction location $\hat { \mathbf { y } } _ { k }$ . This explicit regression head separates continuous spatial prediction from the language generation interface, avoiding the need to represent metric coordinates as text tokens. Let B denote the mini-batch size and $b \in \{ 1 , \ldots , B \}$ index the samples. Let $( \mathcal { M } _ { Y } ) _ { b , k } \in \{ 0 , 1 \}$ indicate whether the interaction target ${ \bf y } _ { b , k }$ is valid. We define the mask-aware ADE loss as:

$$
\mathcal { L } _ { \mathrm { A D E } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \frac { \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { Y } ) _ { b , k } \left. \hat { \mathbf { y } } _ { b , k } - \mathbf { y } _ { b , k } \right. _ { 2 } } { \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { Y } ) _ { b , k } } .\tag{14}
$$

The Smooth L1 location loss is defined as:

$$
\mathcal { L } _ { \mathrm { S 1 } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \frac { \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { Y } ) _ { b , k } \mathrm { S m o o t h L 1 } \left( \hat { \mathbf { y } } _ { b , k } , \mathbf { y } _ { b , k } \right) } { \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { Y } ) _ { b , k } } .\tag{15}
$$

The overall location objective is:

$$
\mathcal { L } _ { \mathrm { l o c } } = \lambda _ { \mathrm { A D E } } \mathcal { L } _ { \mathrm { A D E } } + \lambda _ { \mathrm { S 1 } } \mathcal { L } _ { \mathrm { S 1 } } + \lambda _ { \mathrm { C E } } \mathcal { L } _ { \mathrm { C E } } ,\tag{16}
$$

where $\mathcal { L } _ { \mathrm { C E } }$ denotes the auxiliary language modeling loss. The SmoothL1 loss is computed coordinate-wise and averaged over the three spatial dimensions. Both location regression losses are evaluated in the normalized coordinate space.

2) Hand-Conditioned Residual Flow Matching: Given the observed pose history and ordered future interaction locations, the pose stage first forecasts a location-conditioned deterministic anchor and then models stochastic residuals around it using conditional flow matching.

Spatiotemporal conditioning and anchoring. To preserve the kinematic structure of the human body, each observed SMPL state is represented as a 24-node graph consisting of one root joint and 23 articulated body joints, with skeletal connections defining the graph edges. Graph propagation captures dependencies among physically connected body parts while preserving the SMPL topology. The resulting node features are pooled into frame-level pose tokens and processed by a pose history Transformer, whose final classification token state h summarizes the observed motion history.

Given the interaction location sequence $\tilde { \mathcal { V } }$ defined in the problem formulation, we set $\Delta \tilde { \mathbf { y } } _ { 1 } = \mathbf { 0 }$ and $\Delta \tilde { \mathbf { y } } _ { k } = \tilde { \mathbf { y } } _ { k } - \tilde { \mathbf { y } } _ { k - 1 }$ for $k = 2 , \ldots , K$ . The future condition encoder represents each future step as:

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { ( \mathbf { f } _ { k } ) _ { k = 1 } ^ { K } = \mathrm { T r a n s f o r m e r } _ { \mathrm { f u t } } \left( ( \eta _ { k } ) _ { k = 1 } ^ { K } \right) , } \\ { \eta _ { k } = \mathrm { M L P } _ { \mathrm { c } } \left( \left[ \tilde { \mathbf { y } } _ { k } , \Delta \tilde { \mathbf { y } } _ { k } , k / K \right] \right) } \end{array} \right. , } \end{array}\tag{17}
$$

where $\eta _ { k }$ encodes the interaction location, displacement, and relative future step, and $\mathbf { f } _ { k }$ denotes the corresponding contextaware representation.

For each future step, the deterministic anchor is predicted by combining the global pose history representation with the corresponding future location representation:

$$
\hat { \mathbf { X } } _ { k } ^ { \mathrm { a } } = \mathrm { M L P } _ { \mathrm { a } } \left( \left[ \mathbf { h } , \mathbf { f } _ { k } \right] \right) .\tag{18}
$$

The anchor estimates the root translation, root orientation, and articulated body configuration, providing a reference grounded in the interaction location for subsequent residual motion generation rather than serving as the final prediction.

TABLE II  
INTERACTION LOCATION FORECASTING RESULTS ON THE COHERENT4D DATASET. HIGFLOW AND ALL BASELINES ARE EVALUATED UNDER THE SAME 3D SETTING, FOLLOWING THE MMTWIN 3D SETTING. ALL VALUES ARE REPORTED IN MILLIMETERS, WITH LOWER VALUES BEING BETTER. BOLD AND UNDERLINE DENOTE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td>Model</td><td colspan="3">Health</td><td colspan="3">Bike Repair</td><td colspan="3">Cooking</td></tr><tr><td></td><td>ADE↓</td><td> $\mathrm { \ A D E _ { 9 0 } \downarrow }$ </td><td>FDE↓</td><td>ADE↓</td><td> $\mathrm { \ A D E _ { 9 0 } \downarrow }$ </td><td>FDE↓</td><td>ADE↓</td><td> $\mathrm { \ A D E _ { 9 0 } \downarrow }$ </td><td>FDE↓</td></tr><tr><td>FIction</td><td>40.30</td><td>62.62</td><td>40.58</td><td>88.55</td><td>143.21</td><td>98.61</td><td>100.61</td><td>184.63</td><td>107.11</td></tr><tr><td>Qwen3-VL</td><td>57.53</td><td>89.54</td><td>82.04</td><td>90.71</td><td>156.55</td><td>103.81</td><td>103.95</td><td>199.20</td><td>112.22</td></tr><tr><td>V-JEPA</td><td>45.18</td><td>69.49</td><td>46.26</td><td>86.50</td><td>142.43</td><td>96.33</td><td>102.33</td><td>194.80</td><td>107.35</td></tr><tr><td>Diff-IP3D</td><td>292.84</td><td>433.26</td><td>294.25</td><td>525.63</td><td>842.93</td><td>561.19</td><td>647.23</td><td>1268.28</td><td>662.71</td></tr><tr><td>MMTwin</td><td>290.45</td><td>437.88</td><td>285.21</td><td>429.71</td><td>690.80</td><td>480.29</td><td>513.75</td><td>957.17</td><td>556.40</td></tr><tr><td>HIGFlow</td><td>40.21</td><td>59.52</td><td>41.44</td><td>80.91</td><td>131.87</td><td>93.78</td><td>93.46</td><td>178.20</td><td>104.26</td></tr></table>

The anchor predictor is optimized with

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a n c h o r } } = \lambda _ { \mathrm {6 D } } \mathcal { L } _ { \mathrm { 6 D } } + \lambda _ { \mathrm { g e o } } \mathcal { L } _ { \mathrm { g e o } } + \lambda _ { \mathrm { r o o t g e o } } \mathcal { L } _ { \mathrm { r o o t g e o } } } \\ & { ~ + \lambda _ { \mathrm { b o d y g e o } } \mathcal { L } _ { \mathrm { b o d y g e o } } + \lambda _ { \mathrm { t r a n s } } \mathcal { L } _ { \mathrm { t r a n s } } } \\ & { ~ + \lambda _ { \mathrm { v e l } } \mathcal { L } _ { \mathrm { v e l } } + \lambda _ { \mathrm { j o i n t } } \mathcal { L } _ { \mathrm { j o i n t } } , } \end{array}\tag{19}
$$

where $\mathcal { L } _ { \mathrm { 6 D } }$ is the coordinate-wise $\ell _ { 1 }$ loss over the 24 continuous 6D rotations. $\mathcal { L } _ { \mathrm { g e o } }$ measures the mean geodesic error over all 24 rotations, while $\mathcal { L } _ { \mathrm { r o o t g e o } }$ and ${ \mathcal { L } } _ { \mathrm { b o d y g e o } }$ separately supervise the root and 23 non-root body rotations. All framewise losses are evaluated only at future steps marked valid by $\mathcal { M } _ { X }$ Let $\hat { \mathbf { t } } _ { b , k } ^ { \mathrm { a } }$ and $\mathbf { t } _ { b , k }$ denote the predicted and ground-truth root translations, and let $\hat { \mathbf { p } } _ { b , k , j } ^ { \mathrm { a } }$ and $\mathbf { p } _ { b , k , j }$ denote the corresponding 3D joint positions. The translation and joint position losses are

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r a n s } } = \frac { \sum _ { b = 1 } ^ { B } \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { X } ) _ { b , k } \left\| \hat { \mathbf { t } } _ { b , k } ^ { \mathrm { a } } - \mathbf { t } _ { b , k } \right\| _ { 1 } } { 3 \sum _ { b = 1 } ^ { B } \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { X } ) _ { b , k } } , } \\ { \mathcal { L } _ { \mathrm { j o i n t } } = \frac { \sum _ { b = 1 } ^ { B } \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { X } ) _ { b , k } \sum _ { j = 1 } ^ { J _ { \mathrm { p o s } } } \left\| \hat { \mathbf { p } } _ { b , k , j } ^ { \mathrm { a } } - \mathbf { p } _ { b , k , j } \right\| _ { 2 } } { J _ { \mathrm { p o s } } \sum _ { b = 1 } ^ { B } \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { X } ) _ { b , k } } . } \end{array}\tag{20}
$$

The velocity loss $\mathcal { L } _ { \mathrm { v e l } }$ further penalizes the masked coordinatewise $\ell _ { 1 }$ error between consecutive differences of the predicted and ground-truth 147-dimensional SMPL states, using only valid consecutive-step pairs.

After anchor pretraining, the pose history encoder, future condition encoder, and anchor predictor are frozen.

Residual Flow Matching. To capture multiple feasible motions without deviating excessively from the deterministic anchor, we model stochastic variations in an anchor-relative residual space using conditional Flow Matching [17]. For each future step k, the target residual is defined as:

$$
\mathbf { r } _ { k } ^ { * } = \operatorname { R e s i d u a l } \left( \hat { \mathbf { X } } _ { k } ^ { \mathrm { a } } , \mathbf { X } _ { k } \right) ,\tag{21}
$$

where Residual computes rotational corrections for the root and body joints via the logarithmic map of relative rotations, and the root translation correction by subtraction. Concatenating the 3D root rotation correction, 3D root translation correction, and $2 3 \times 3$ body rotation corrections yields a 75-dimensional residual vector. Root translation residuals are computed and clipped in meters, and are added to the anchor root translations in meters during pose composition.

To regulate the magnitude of stochastic motion variations, we derive a step-specific residual gate from the pose history, future location condition, and deterministic anchor:

$$
\begin{array} { r } { \mathbf { c } _ { k } ^ { \mathrm { f i o w } } = \left[ \mathbf { h } , \mathbf { f } _ { k } , P _ { \mathrm { a } } \left( \hat { \mathbf { X } } _ { k } ^ { \mathrm { a } } \right) \right] , \quad } \\ { \gamma _ { k } = \gamma _ { \mathrm { m a x } } \odot \mathrm { s i g m o i d } \left( \Gamma \left( \mathbf { c } _ { k } ^ { \mathrm { f i o w } } \right) \right) . } \end{array}\tag{22}
$$

Here, $P _ { \mathrm { a } }$ projects the anchor state and Γ predicts three gate values corresponding to root rotation, root translation, and body rotation. The vector $\gamma _ { \mathrm { m a x } } \in \mathbb { R } _ { + } ^ { 3 }$ sets their maximum strengths. We further impose component-specific residual bounds. Let $\rho = \mathrm { c o n c a t } ( \rho _ { \mathrm { R } } \mathbf { 1 } _ { 3 } , \rho _ { \mathrm { t } } \mathbf { 1 } _ { 3 } , \rho _ { \mathrm { B } } \mathbf { 1 } _ { 6 9 } )$ , where $\rho _ { \mathrm { R } } , \rho _ { \mathrm { t } } .$ , and $\rho _ { \mathrm { B } }$ bound the root rotation, root translation, and body rotation residuals, respectively. Denoting element-wise clipping to $[ - \rho , \rho ]$ by $B _ { \rho } ,$ we define the step-specific regulation operator as:

$$
\begin{array} { r } { \mathcal { R } _ { k } ( \mathbf { x } ) = \mathcal { B } _ { \rho } \left( \mathrm { b c a s t } ( \gamma _ { k } ) \odot \mathbf { x } \right) , } \end{array}\tag{23}
$$

where bcast expands the three gate values over the corresponding 3, 3, and 69 residual dimensions. The training target is bounded as $\bar { \mathbf { r } } _ { k } ^ { * } = { \cal B } _ { \rho } ( \mathbf { r } _ { k } ^ { * } )$ , while generated residuals are additionally modulated by $\mathcal { R } _ { k }$ . This regulation constrains stochastic deviations from the anchor while allowing their magnitude to adapt to each future interaction step.

For $\tau \sim \mathcal { U } ( 0 , 1 )$ and $\mathbf { z } _ { k } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { r } ^ { 2 } \mathbf { I } _ { 7 5 } )$ , we construct the linear probability path:

$$
\begin{array} { r } { { \bf x } _ { \tau , k } = ( 1 - \tau ) { \bf z } _ { k } + \tau \bar { \bf r } _ { k } ^ { * } . } \end{array}\tag{24}
$$

The conditional velocity field is trained with

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \left[ \left\| v \left( \mathbf { x } _ { \tau , k } , \tau , \mathbf { c } _ { k } ^ { \mathrm { f o w } } \right) - \left( \bar { \mathbf { r } } _ { k } ^ { * } - \mathbf { z } _ { k } \right) \right\| _ { 2 } ^ { 2 } \right] ,\tag{25}
$$

where v is the conditional residual velocity field. The expectation is taken over valid future steps, flow times, and Gaussian prior samples.

For training-time endpoint supervision, we obtain a direct residual estimate from an intermediate flow state as:

$$
\begin{array} { r } { \tilde { \mathbf { r } } _ { k } ^ { \mathrm { d i r e c t } } = \mathcal { R } _ { k } \left( \mathbf { x } _ { \tau , k } + ( 1 - \tau ) v \left( \mathbf { x } _ { \tau , k } , \tau , \mathbf { c } _ { k } ^ { \mathrm { f o w } } \right) \right) . } \end{array}\tag{26}
$$

We additionally perform a differentiable $N _ { \mathrm { O D E ^ { - S t e p } } }$ Heun rollout from the Gaussian prior and denote the regulated terminal residual by $\tilde { \mathbf { r } } _ { k } ^ { \mathrm { r o l l } }$

Finally, kinematic pose reconstruction composes the direct and rollout residuals with the deterministic anchor through

TABLE III  
FULL-BODY POSE FORECASTING RESULTS ON THE COHERENT4D DATASET. THE LEFT SIDE REPORTS POSE FORECASTING CONDITIONED ON GROUND-TRUTH FUTURE INTERACTION LOCATIONS, WHILE THE RIGHT SIDE REPORTS POSE FORECASTING CONDITIONED ON PREDICTED FUTURE INTERACTION LOCATIONS. MPJPE. PA-MPJPE. AND ROOT TRANS. ARE REPORTED IN MILLIMETERS, WHILE BODY GEO. IS REPORTED IN DEGREES. SINGLE USES THE FIRST CANDIDATE, WHEREAS BEST-5 REPORTS ALL METRICS FOR THE CANDIDATE SELECTED BY SAMPLE-LEVEL MPJPE. LOWER VALUES ARE BETTER. BOLD AND UNDERLINE DENOTE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td rowspan="2">Model</td><td colspan="2">MPJPE↓</td><td colspan="2">PA-MPJPE↓</td><td colspan="2">Root Trans.↓</td><td colspan="2">Body Geo.↓</td><td colspan="2">MPJPE↓</td><td colspan="2">PA-MPJPE↓</td><td colspan="2">Root Trans.↓</td><td colspan="2">Body Geo.↓</td></tr><tr><td>Single</td><td>Best-5</td><td></td><td>Single Best-5</td><td>Single</td><td>Best-5</td><td>Single Best-5</td><td></td><td>Single</td><td>Best-5</td><td>Single</td><td>Best-5</td><td>Single Best-5</td><td></td><td>Single Best-5</td></tr><tr><td colspan="10">Health</td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>FIction</td><td>117.17</td><td>115.91</td><td>42.12</td><td>42.01</td><td>104.09</td><td>102.91</td><td>9.03 9.02</td><td></td><td>124.59</td><td>123.43</td><td>50.82</td><td>49.73</td><td>108.67</td><td>107.52 9.01</td><td>9.00</td><td></td></tr><tr><td>SkeletonDiffusion</td><td>166.32</td><td>137.39</td><td>53.18</td><td>52.19</td><td>158.85</td><td>138.47</td><td>10.55 10.45</td><td></td><td>184.97</td><td>155.55</td><td>55.01</td><td>54.20</td><td>172.37</td><td>151.19 10.89</td><td>10.81</td><td></td></tr><tr><td>SLD-HMP</td><td>421.56</td><td>88.38</td><td>44.79</td><td>32.50</td><td>403.30</td><td>74.40</td><td>8.76</td><td>5.63</td><td>402.76</td><td>92.15</td><td>48.66</td><td>33.22</td><td>381.66</td><td>89.37</td><td>8.56</td><td>5.72</td></tr><tr><td>HIGFlow</td><td>95.25</td><td>93.45</td><td>43.79</td><td>43.66</td><td>80.93</td><td>79.06</td><td>8.80</td><td>8.79</td><td>119.96</td><td>118.14</td><td>47.06</td><td>46.99</td><td>94.19</td><td>92.29</td><td>9.33</td><td>9.32</td></tr><tr><td colspan="14"></td></tr><tr><td>FIction</td><td>295.34 284.52</td><td></td><td>78.95</td><td>78.24</td><td>292.07</td><td>279.21</td><td>Bike Repair 13.60</td><td>13.50</td><td>351.64</td><td>341.06</td><td>86.75</td><td>85.85</td><td>325.06 312.04</td><td></td><td>14.14</td><td>14.03</td></tr><tr><td>SkeletonDiffusion</td><td>453.52</td><td>370.26</td><td>113.22</td><td>109.63</td><td>405.49</td><td>331.24</td><td>17.69</td><td>17.44</td><td>469.82</td><td>398.82</td><td>115.09</td><td>111.77</td><td>408.17</td><td>343.07</td><td></td><td>17.74</td></tr><tr><td>SLD-HMP</td><td>269.06</td><td>234.95</td><td>74.34</td><td>70.03</td><td>261.00</td><td>218.63</td><td>11.97</td><td>11.12</td><td>358.20</td><td>314.88</td><td>83.82</td><td>81.76</td><td>311.54</td><td>296.01</td><td>17.89 12.39</td><td>11.76</td></tr><tr><td>HIGFlow</td><td>213.65</td><td>210.09</td><td>70.07</td><td>70.03</td><td>224.19</td><td>220.18</td><td>15.05</td><td>15.05</td><td>306.92</td><td>303.07</td><td>82.26</td><td>82.25</td><td>287.24</td><td>282.98</td><td>16.07</td><td>16.07</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Cooking</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="10"></td><td colspan="3"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>FIction SkeletonDiffusion</td><td>237.86 310.39</td><td>226.62</td><td>43.81</td><td>43.76</td><td>227.23</td><td>214.75</td><td>8.34</td><td>8.34</td><td>374.89</td><td>364.15</td><td></td><td>46.30</td><td>46.28</td><td>348.70336.89</td><td></td><td>8.51</td><td>8.51</td></tr><tr><td>SLD-HMP</td><td>184.42</td><td>260.00 167.19</td><td>51.43</td><td>51.25</td><td>278.05</td><td>229.97</td><td>11.42</td><td>11.41</td><td>421.75</td><td>382.35</td><td>53.09</td><td>52.91</td><td></td><td>377.64 339.03 313.26</td><td>11.89</td><td></td><td>11.88</td></tr><tr><td></td><td>143.98</td><td>143.41</td><td>43.41</td><td>41.21</td><td>174.33</td><td>156.25</td><td>7.67</td><td>7.25</td><td>358.92 340.72</td><td>342.11 340.24</td><td>44.86 45.65</td><td>43.13 45.64</td><td></td><td>332.74 312.56</td><td>312.11</td><td>7.83</td><td>7.53</td></tr><tr><td>HIGFlow</td><td></td><td></td><td>39.45</td><td>39.42</td><td>139.26</td><td>138.73</td><td>7.53</td><td>7.53</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8.50</td><td>8.50</td></tr></table>

Compose, which applies rotational corrections via the exponential map and adds the root translation correction. For $u \in \{ \mathrm { d i r e c t , r o l l } \}$ , we reconstruct

$$
\hat { \mathbf { X } } _ { k } ^ { u } = \mathrm { C o m p o s e } \left( \hat { \mathbf { X } } _ { k } ^ { \mathrm { a } } , \tilde { \mathbf { r } } _ { k } ^ { u } \right) , \qquad \hat { \mathcal { X } } ^ { u } = \{ \hat { \mathbf { X } } _ { k } ^ { u } \} _ { k = 1 } ^ { K } .\tag{27}
$$

The corresponding endpoint pose loss is:

$$
\mathcal { L } _ { u } = \mathcal { E } _ { \mathrm { p o s e } } \left( \hat { \mathcal { X } } ^ { u } , \mathcal { X } \right) , \qquad u \in \{ \mathrm { d i r e c t , r o l l } \} ,\tag{28}
$$

where $\mathcal { E } _ { \mathrm { p o s e } }$ uses the same masked 6D rotation, geodesic rotation, root translation, temporal velocity, and 3D joint position terms as the anchor objective.

We further impose a residual alignment loss that encourages the direct endpoint residual to match the clipped target residual:

$$
\mathcal { L } _ { \mathrm { r e s } } = \frac { \sum _ { b } \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { X } ) _ { b , k } \left\| \tilde { \mathbf { r } } _ { b , k } ^ { \mathrm { d i r e c t } } - \bar { \mathbf { r } } _ { b , k } ^ { * } \right\| _ { 2 } ^ { 2 } } { 7 5 \sum _ { b } \sum _ { k = 1 } ^ { K } ( \mathcal { M } _ { X } ) _ { b , k } } .\tag{29}
$$

The overall pose objective is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p o s e } } = \lambda _ { \mathrm { F M } } \mathcal { L } _ { \mathrm { F M } } + \lambda _ { \mathrm { d i r e c t } } \mathcal { L } _ { \mathrm { d i r e c t } } + \lambda _ { \mathrm { r o l l } } \mathcal { L } _ { \mathrm { r o l l } } + \lambda _ { \mathrm { r e s } } \mathcal { L } _ { \mathrm { r e s } } . } \end{array}\tag{30}
$$

At inference, we independently sample S residual priors and integrate each using $N _ { \mathrm { O D E } }$ Heun steps. The resulting regulated terminal residuals are composed with the deterministic anchor to produce $S$ plausible future motion sequences, $\{ \hat { \mathcal X } ^ { ( q ) } \} _ { q = 1 } ^ { S }$

## V. EXPERIMENTS

## A. Experimental Settings

1) Baselines: We compare HIGFlow with two groups of baselines corresponding to the two forecasting stages.

For interaction location forecasting, we consider FIction [19], Qwen3-VL-2B [58], V-JEPA 2.1 [59], Diff-IP3D [30], and MMTwin [32]. FIction is the closest prior method coupling interaction localization with pose forecasting, while Qwen3-VL and V-JEPA 2.1 provide semanticand dynamics-oriented baselines, respectively. Following MMTwin [32], which extends Diff-IP2D [30] and MADiff [31] from 2D to 3D, we extend Diff-IP2D to predict continuous 3D interaction locations and refer to this variant as Diff-IP3D. MMTwin is reproduced in its 3D configuration with its prediction targets matched to Coherent4D. All methods are evaluated in the shared sample-local coordinate frame using the corrected USST evaluation implementation [4].<sup>1</sup>

For full-body pose forecasting, we compare HIGFlow with FIction, SkeletonDiffusion, and SLD-HMP under the same interaction location conditioning protocol. During training, all methods receive ground-truth future interaction locations. FIction follows its original location-conditioned formulation, whereas SkeletonDiffusion and SLD-HMP retain their original forecasting architectures with only the input and output interfaces adapted to our SMPL representation. We report both Single and Best-5 results following the evaluation protocol defined above.

2) Implementation Details: Throughout this paper, V-JEPA refers to V-JEPA 2.1 ViT-Giant/384, and Qwen3-VL to Qwen3-VL-2B. For interaction location forecasting, we use M = 64 resampled V-JEPA memory tokens, set $\alpha = 0 . 1$ , and employ a three-layer coordinate decoder. The loss weights are $( \lambda _ { \mathrm { A D E } } , \lambda _ { \mathrm { S 1 } } , \lambda _ { \mathrm { C E } } ) = ( 1 . 5 , 0 . 6 , 0 . 0 1 )$ ). The base predictor and JEPA adapter are trained sequentially with AdamW and cosine learning rate schedules, using learning rate/warmup ratio pairs of $( 1 \times 1 0 ^ { - 5 } , 0 . 1 0 )$ and $( 3 \times 1 0 ^ { - 6 } , 0 . 0 5 )$ , respectively, while the base predictor, structured encoders, and coordinate decoder remain frozen during adapter training.

The pose history and future condition Transformers contain four and two layers, respectively, while $\mathrm { M L P _ { c } }$ and $\mathrm { M L P _ { a } }$ use two and three linear layers. The anchor predictor is pretrained with AdamW for up to 60 epochs, using a learning rate of $8 \times 1 0 ^ { - 5 }$ , weight decay $1 0 ^ { - 4 }$ , gradient clipping at 1.0, and an early stopping patience of six epochs. For Health and Cooking, we set $( \lambda _ { \mathrm { 6 D } } , \lambda _ { \mathrm { g e o } } , \lambda _ { \mathrm { r o o t g e o } } , \lambda _ { \mathrm { b o d y g e o } } , \lambda _ { \mathrm { v e l } } , \lambda _ { \mathrm { j o i n t } } ) =$ $( 1 , 0 . 1 , 0 . 5 , 1 , 1 , 1 0 )$ , with $\lambda _ { \mathrm { t r a n s } } ~ = ~ 1 2$ , batch size 1, and dropout 0.10. For Bike Repair, we set $\lambda _ { \mathrm { r o o t g e o } } = \lambda _ { \mathrm { b o d y g e o } } = 0$ and $\lambda _ { \mathrm { t r a n s } } = 1 0$ , with the remaining loss weights unchanged, a batch size of 12, and dropout 0.15.

TABLE IV  
ABLATION OF V-JEPA RESIDUAL FUSION FOR INTERACTION LOCATION FORECASTING. LOCENC, ENVENC, COORDDEC, AND SAMPLED VIDEO FRAMES ARE FIXED. ALL VALUES ARE REPORTED IN MILLIMETERS, AND LOWER VALUES ARE BETTER.
<table><tr><td>Domain</td><td>V-JEPA injection | ADE↓</td><td></td><td> $\mathrm { \bf A D E _ { 9 0 } }$ </td><td>↓ FDE↓</td></tr><tr><td rowspan="2">Health</td><td></td><td>40.28</td><td>59.34</td><td>41.47</td></tr><tr><td> $\surd$ </td><td>40.21</td><td>59.52</td><td>41.44</td></tr><tr><td rowspan="2">Bike Repair</td><td> $^ -$ </td><td>81.18</td><td>134.10</td><td>94.03</td></tr><tr><td> $\surd$ </td><td>80.91</td><td>131.87</td><td>93.78</td></tr><tr><td rowspan="2">Cooking</td><td> $^ -$ </td><td>94.20</td><td>178.21</td><td>104.83</td></tr><tr><td> $\surd$ </td><td>93.46</td><td>178.20</td><td>104.26</td></tr></table>

Residual flow matching freezes the pose history encoder, future condition encoder, and anchor predictor. We use $\sigma _ { r } =$ 0.003, batch size 16, and AdamW with weight decay $1 0 ^ { - 4 } .$ The learning rates are $8 \times 1 0 ^ { - 5 }$ for Health and Cooking and $5 \times 1 0 ^ { - 5 }$ for Bike Repair. The loss weights are set to $\lambda _ { \mathrm { F M } } ~ = ~ \lambda _ { \mathrm { d i r e c t } } ~ = ~ \lambda _ { \mathrm { r o l l } } ~ = ~ 1 . 0$ and $\lambda _ { \mathrm { r e s } } ~ = ~ 0 . 5 .$ . For Health and Cooking, we set $\gamma _ { \mathrm { m a x } } ~ = ~ ( 1 . 0 , 1 . 0 , 0 . 7 )$ and $( \rho _ { \mathrm { R } } , \rho _ { \mathrm { t } } , \rho _ { \mathrm { B } } ) = ( 0 . 1 7 4 5 3 3 \mathrm { r a d } , 0 . 1 2 \mathrm { m } , 0 . 1 2 \mathrm { r a d } ) $ . For Bike Repair, the corresponding settings are $\gamma _ { \mathrm { m a x } } = ( 0 . 8 , 1 . 0 , 0 . 3 5 )$ and $( \rho _ { \mathrm { R } } , \rho _ { \mathrm { t } } , \rho _ { \mathrm { B } } ) = ( 0 . 2 5 \mathrm { r a d } , 0 . 2 5 \mathrm { m } , 0 . 0 8 \mathrm { r a d } )$ . The bound $\rho _ { t }$ is applied directly to each coordinate of the root translation residual in meters, without division by s. Both training and inference use $N _ { \mathrm { O D E } } = 4$ Heun steps. At inference, we generate $S = 5$ candidates for Best-5 evaluation. All experiments are conducted on six NVIDIA A100 GPUs.

## B. Quantitative Analysis

We organized the quantitative analysis around the following questions.

1) Can HIGFlowjointlyforecast interaction locations and full-body poses while outperforming specialized baselines?

We evaluated the two stages of HIGFlow against methods designed specifically for the corresponding forecasting tasks. Table II reports interaction location forecasting results, while Table III reports full-body pose forecasting results under two location-conditioning settings: the left part uses ground-truth future interaction locations, whereas the right part uses predicted future interaction locations and represents the standard forecasting setting. For each task and conditioning setting, HIGFlow and all corresponding baselines are evaluated under the same evaluation protocol, enabling direct comparison of their forecasting performance. HIGFlow achieves leading performance in both interaction location forecasting and full-body pose forecasting, with particularly strong results on Cooking and Bike Repair and competitive performance on Health. Overall, HIGFlow demonstrates consistent performance on the interaction forecasting task.

TABLE V  
DOMAIN-WISE MOTION STATISTICS IN THE SHARED SAMPLE-LOCAL COORDINATE FRAME. LOC. DISP. AND ROOT DISP. MEASURE THE MEAN DISPLACEMENT BETWEEN CONSECUTIVE VALID FUTURE STEPS FOR INTERACTION LOCATIONS AND SMPL ROOT TRANSLATIONS, RESPECTIVELY. VALUES ARE REPORTED IN MILLIMETERS.
<table><tr><td>Domain</td><td>Loc. Disp.</td><td>Root Disp.</td><td>Spatial pattern</td></tr><tr><td>Health</td><td>153.86</td><td>59.45</td><td>Compact spatial movement</td></tr><tr><td>Bike Repair</td><td>293.53</td><td>195.71</td><td>Tool-centric reaching and repair</td></tr><tr><td>Cooking</td><td>318.69</td><td>171.48</td><td>Broad object-centric movement</td></tr></table>

TABLE VI

EFFECT OF THE V-JEPA MEMORY TOKEN BUDGET ON INTERACTION LOCATION FORECASTING. M DENOTES THE NUMBER OF RESAMPLED V-JEPA MEMORY TOKENS. THE SHADED ENTRIES INDICATE THE TOKEN BUDGET SELECTED FOR THE FINAL MODEL. ALL VALUES ARE REPORTED IN MILLIMETERS, AND LOWER VALUES ARE BETTER. BOLD DENOTES THE BEST RESULT FOR EACH DOMAIN AND METRIC.
<table><tr><td>Domain</td><td>M</td><td>ADE↓</td><td> $\mathrm { \ A D E _ { 9 0 } ~ . }$  →</td><td>FDE↓</td></tr><tr><td rowspan="4">Health</td><td>16</td><td>40.23</td><td>59.61</td><td>41.46</td></tr><tr><td>32</td><td>40.23</td><td>59.49</td><td>41.46</td></tr><tr><td>64</td><td>40.21</td><td>59.52</td><td>41.44</td></tr><tr><td>128</td><td>40.23</td><td>59.44</td><td>41.42</td></tr><tr><td rowspan="4">Bike Repair</td><td>16</td><td>88.47</td><td>138.75</td><td>98.59</td></tr><tr><td>32</td><td>87.08</td><td>136.32</td><td>97.56</td></tr><tr><td>64</td><td>80.91</td><td>131.87</td><td>93.78</td></tr><tr><td>128</td><td>86.94</td><td>135.53</td><td>97.58</td></tr><tr><td rowspan="4">Cooking</td><td>16</td><td>93.36</td><td>178.45</td><td>104.20</td></tr><tr><td>32</td><td>93.04</td><td>177.63</td><td>104.33</td></tr><tr><td>64</td><td>93.46</td><td>178.20</td><td>104.26</td></tr><tr><td>128</td><td>93.36</td><td>178.03</td><td>104.22</td></tr></table>

## 2) What is the effect of V-JEPA residual fusion?

We fixed the location encoder, environment encoder, coordinate decoder, frame input, and training configuration, and varied only the JEPA-guided residual adapter. When enabled, the resampled V-JEPA memory tokens are fused with the hidden states at the future step readout positions after the Qwen3-VL decoder forward pass. Table IV shows that V-JEPA residual fusion generally improves interaction location forecasting, with the most pronounced gains on Bike Repair. These results indicate that V-JEPA provides complementary motion-sensitive visual cues that help localize future locations beyond the semantic and contextual information captured by the base predictor.

## 3) Why do HIGFlow’s gains vary across domains?

HIGFlow achieves larger improvements on Cooking and Bike Repair than on Health in both interaction location and pose forecasting. As shown in Table V, Cooking and Bike Repair exhibit substantially greater location and root displacements, providing richer geometric and dynamic cues for modeling future locations and full-body motion. In contrast, Health involves more compact spatial movements, for which the competing methods already perform relatively well, leaving less room for improvement. These observations suggest that HIGFlow benefits particularly from domains involving larger and more complex motion variations.

TABLE VII  
ABLATION OF INPUT REPRESENTATIONS AND OUTPUT DECODING FOR INTERACTION LOCATION FORECASTING. LOCENC, ENVENC, AND COORDDEC DENOTE THE LOCATION ENCODER, ENVIRONMENT ENCODER, AND COORDINATE DECODER, RESPECTIVELY. A DASH IN THE LOCENC OR ENVENC COLUMN INDICATES THAT THE CORRESPONDING INPUT IS REPRESENTED AS TEXT RATHER THAN PROCESSED BY A DEDICATED ENCODER. A DASH IN THE COORDDEC COLUMN INDICATES THAT INTERACTION LOCATIONS ARE GENERATED AS TEXT RATHER THAN DECODED BY A DEDICATED COORDINATE DECODER. GRAY ROWS INCLUDE V-JEPA RESIDUAL FUSION. ALL VALUES ARE REPORTED IN MILLIMETERS, WITH LOWER VALUES INDICATING BETTER PERFORMANCE AND THE BEST RESULTS HIGHLIGHTED IN BOLD.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">LocEnc</td><td rowspan="2">EnvEnc</td><td rowspan="2">CoordDec</td><td colspan="3">Health</td><td colspan="3">Bike Repair</td><td colspan="3">Cooking</td></tr><tr><td>ADE↓</td><td> $\mathrm { \ A D E _ { 9 0 } \downarrow }$ </td><td>FDE↓</td><td>ADE↓</td><td>ADE90 ↓</td><td>FDE↓</td><td>ADE↓</td><td>ADE90↓</td><td>FDE↓</td></tr><tr><td>Qwen3-VL</td><td>一</td><td>1</td><td>1</td><td>57.53</td><td>89.54</td><td>82.04</td><td>90.71</td><td>156.55</td><td>103.81</td><td>103.95</td><td>199.20</td><td>112.22</td></tr><tr><td>w/o CoordDec</td><td>√</td><td>√</td><td>1</td><td>57.11</td><td>89.31</td><td>59.71</td><td>99.64</td><td>161.61</td><td>113.42</td><td>113.87</td><td>222.51</td><td>119.68</td></tr><tr><td>w/o LocEnc</td><td>一</td><td>√</td><td>√</td><td>42.57</td><td>63.41</td><td>44.88</td><td>87.78</td><td>143.79</td><td>98.04</td><td>97.29</td><td>182.26</td><td>105.15</td></tr><tr><td>w/o LocEnc + V-JEPA</td><td>一</td><td>√</td><td>√</td><td>42.18</td><td>63.11</td><td>44.20</td><td>86.40</td><td>139.00</td><td>96.76</td><td>92.14</td><td>179.32</td><td>101.24</td></tr><tr><td>w/o EnvEnc</td><td>小</td><td>一</td><td>√</td><td>42.17</td><td>65.26</td><td>43.32</td><td>87.65</td><td>137.12</td><td>97.31</td><td>100.13</td><td>184.76</td><td>106.65</td></tr><tr><td>w/o EnvEnc + V-JEPA</td><td>√</td><td>一</td><td>√</td><td>41.86</td><td>63.68</td><td>42.83</td><td>85.94</td><td>133.97</td><td>95.19</td><td>96.24</td><td>182.77</td><td>103.39</td></tr><tr><td>HIGFlow (Full model)</td><td></td><td></td><td></td><td>40.21</td><td>59.52</td><td>41.44</td><td>80.91</td><td>131.87</td><td>93.78</td><td>93.46</td><td>178.20</td><td>104.26</td></tr></table>

TABLE VIII

INPUT ABLATION FOR INTERACTION LOCATION FORECASTING ACROSS THE THREE DOMAINS. LOC. AND ENV. DENOTE LOCATION HISTORY AND ENVIRONMENT CONTEXT, RESPECTIVELY. FRAMES DENOTES SAMPLED VIDEO FRAMES PROVIDED TO QWEN3-VL. GRAY ROWS INCLUDE V-JEPA RESIDUAL FUSION, WHICH RETAINS FEATURES EXTRACTED BY V-JEPA FROM THE OBSERVED VIDEO. ALL VALUES ARE REPORTED IN MILLIMETERS, WITH LOWER VALUES INDICATING BETTER PERFORMANCE AND THE BEST RESULTS HIGHLIGHTED IN BOLD.
<table><tr><td rowspan="2">Input</td><td rowspan="2">Loc.</td><td rowspan="2">Env.</td><td rowspan="2">Frames</td><td colspan="2">Health</td><td colspan="3">Bike Repair</td><td colspan="3">Cooking</td></tr><tr><td>ADE↓</td><td> $\mathrm { \ A D E _ { 9 0 } \downarrow }$ </td><td>FDE↓</td><td>ADE↓  $\mathrm { \ A D E _ { 9 0 } \downarrow }$ </td><td>FDE↓</td><td>ADE↓</td><td> $\mathrm { \ A D E _ { 9 0 } \downarrow }$ </td><td>FDE↓</td></tr><tr><td>w/o Env.</td><td>√</td><td>√</td><td></td><td>37.40 58.16</td><td>38.65</td><td>114.51</td><td>161.50</td><td>124.01</td><td>97.92</td><td>189.73</td><td>106.63</td></tr><tr><td>w/o Env. + V-JEPA</td><td>√</td><td>√</td><td></td><td>37.32 58.27</td><td>38.68</td><td>91.79</td><td>141.30</td><td>101.61</td><td>94.60</td><td>186.08</td><td>104.71</td></tr><tr><td>w/o Loc.</td><td></td><td>√ √</td><td></td><td>69.48 103.51</td><td>67.04</td><td>135.71</td><td>199.00</td><td>144.62</td><td>143.82</td><td>267.09</td><td>148.47</td></tr><tr><td>w/o Loc. + V-JEPA</td><td></td><td>√ √</td><td></td><td>55.59 85.30</td><td>55.53</td><td>162.59</td><td>250.96</td><td>168.90</td><td>127.97</td><td>246.60</td><td>130.92</td></tr><tr><td>w/o Frames</td><td>√</td><td>√ 一</td><td></td><td>51.91 75.49</td><td>51.98</td><td>143.90</td><td>224.01</td><td>143.32</td><td>145.86</td><td>221.17</td><td>143.07</td></tr><tr><td>w/o Frames + V-JEPA</td><td>√</td><td>√ 一</td><td></td><td>50.61 75.93</td><td></td><td>51.78 117.27</td><td>175.46</td><td>125.37</td><td>119.60</td><td>225.82</td><td>123.65</td></tr><tr><td>HIGFlow (Full model)</td><td></td><td></td><td></td><td>40.21</td><td>59.52</td><td>41.44 80.91</td><td>131.87</td><td>93.78</td><td>93.46</td><td>178.20</td><td>104.26</td></tr></table>

4) How do location stage design choices affect interaction location forecasting?

We examined three design aspects of the location stage: structured representation and coordinate decoding, input composition, and V-JEPA memory capacity. The results collectively support the use of structured multimodal cues, explicit continuous coordinate regression, and a moderate dynamic memory budget.

a) Representation and coordinate decoding. We ablated the location encoder, environment encoder, and coordinate decoder while keeping the remaining components fixed. Without LocEnc or EnvEnc, the corresponding structured inputs are serialized as text. Without CoordDec, interaction locations are generated through the language interface rather than continuous coordinate regression. We also included Qwen3-VL as a baseline without these three components. As shown in Table VII, CoordDec provides the clearest and most consistent gains, highlighting the benefit of directly regressing continuous 3D coordinates. LocEnc and EnvEnc generally improve crossdomain consistency by preserving structured location and scene information, although Cooking shows mixed sensitivity. Overall, the full model achieves the most balanced performance across domains.

b) Input composition. We separately removed location history, environment context, or the sampled video frames provided to Qwen3-VL, while retaining the remaining inputs and coordinate decoding architecture. Each configuration was evaluated with and without V-JEPA residual fusion, which still provided features extracted from the observed video when Qwen3-VL received no frames. As shown in Table VIII, removing location history or frame input substantially degrades forecasting performance, while environment context is particularly important for Bike Repair. V-JEPA mitigates the degradation in most cases, suggesting that the visual context encoded by Qwen3-VL and the dynamic features provided by V-JEPA offer complementary information for future location forecasting.

c) V-JEPA token budget. We varied the number of resampled V-JEPA memory tokens over $M \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$ while keeping all other settings fixed. Table VI shows that performance does not improve monotonically with increasing memory size. M = 64 achieves the best results on Bike Repair and remains competitive on Health and Cooking, providing the strongest overall trade-off across domains. We therefore adopted $M = 6 4$ as the default setting.

5) How do pose stage design choices and inference settings affect pose forecasting?

We examined the contributions of location conditioning, the deterministic anchor, and residual flow matching, together with the effect of ODE integration steps.

a) Location source ablation. The left part of Table III provides a controlled evaluation using ground-truth future interaction locations while keeping the same trained pose models, remaining inputs, and inference settings. Compared with the predicted-location setting on the right, ground-truth location conditioning generally improves pose forecasting accuracy, indicating that localization quality directly affects downstream pose prediction. HIGFlow consistently benefits from more accurate location guidance while maintaining strong overall performance under both location settings.

Past Observation  
![](images/3cd8599407a1715c95f697bf832715b00b39b91e51c678df09cc26989fd82666.jpg)  
(b) Cooking  
Fig. 6. Qualitative pose forecasting comparison on Bike Repair and Cooking samples. For each example, the upper panel shows the observed SMPL pose history with synchronized exocentric source frames, while the lower panel presents future exocentric reference frames together with the corresponding groundtruth, HIGFlow, and FIction pose sequences. The exocentric frames are included only for visualization and are not used as inputs to the pose models.

![](images/a0079eecd58a5c7a7c5b236b46d13e8531ef32c3d4ee094895f28094ba67ef47.jpg)

Fig. 7. Pose stage component ablation across Health, Bike Repair, and Cooking. We compared HIGFlow with variants without future interaction location conditioning, without residual flow matching, and without the deterministic anchor.  
![](images/151d47e678dc5191bd3a663b8b4f4cd4c8eb9c6cbd4cb100dcdc61449c140ec3.jpg)  
Fig. 8. Effect of ODE integration steps on Best-5 pose forecasting. Regret is computed relative to the best tested step count for each domain and metric. The shaded band marks the selected setting of four integration steps. Lower values are better.

b) Component ablation. We removed future interaction location conditioning, residual flow matching, or the deterministic anchor while keeping the observed pose history and Best-5 protocol fixed, yielding location-independent, anchoronly, and Flow-Matching-only variants, respectively. As shown in Fig. 7, removing future interaction location conditioning causes the largest overall degradation, demonstrating the importance of interaction locations as geometric guidance for full-body motion forecasting. The deterministic anchor improves structural stability, while residual flow matching captures additional motion variation and improves forecasting accuracy. Their combination achieves the strongest and most balanced performance across domains and metrics.

c) Integration step analysis. We evaluated $N _ { \mathrm { O D E } } \in$ {1, 2, 4, 8, 16} under the same model configuration and Best-5 protocol. Because the pose metrics have different units and scales, we computed the percentage increase in error relative to the best tested result for each domain-metric pair and averaged it over all 12 pairs. As shown in Fig. 8, increasing the number of integration steps does not consistently reduce forecasting error. Four steps achieve the best result in 9 of the 12 comparisons and the lowest mean regret, whereas fewer steps provide insufficient integration accuracy and additional steps increase inference cost without consistent performance gains. We therefore used $N _ { \mathrm { O D E } } = 4$ by default.

## C. Qualitative Analysis

Fig. 6 compares HIGFlow with FIction in two representative domains, Bike Repair and Cooking. For each example, the upper part shows the observed pose history with synchronized exocentric frames, while the lower part presents future reference frames together with the ground truth and the predictions of both methods. The exocentric frames are used only for visualization. In Bike Repair, HIGFlow preserves the bent working pose during the early forecast and transitions to standing at a time closer to the ground truth, whereas FIction becomes upright too early. In Cooking, HIGFlow more often places the predicted interaction location in the correct workspace and produces body displacement and reaching poses that are closer to the reference motion. Together with the ablation results, these examples support the role of future interaction locations as geometric guidance, while the deterministic anchor and residual flow matching improve structural stability and refine articulated motion.

The examples also reveal two limitations. In the later Bike Repair frames, interaction location errors may grow over the forecasting horizon and propagate through cascaded inference to the pose stage. As a result, the predicted body remains too upright instead of following the reference leaning and reaching motion. In Cooking, HIGFlow often identifies the correct interaction region, but the contacting hand, contact height, arm configuration, or torso orientation may still differ from the ground truth. These cases suggest that interaction location conditioning improves coarse spatial alignment but does not fully capture human intent, object affordances, or detailed contact constraints.

## VI. CONCLUSION

We introduced the Coherent4D dataset, which provides temporally aligned interaction location and full-body pose sequences in a shared coordinate system for continuous supervision and evaluation. Building on this formulation, we proposed the HIGFlow framework, a cascaded where-tohow model that couples future interaction location and fullbody pose forecasting by using future interaction locations as geometric conditions for pose prediction. Within HIGFlow,

Semantic-Dynamic Location Forecasting improves continuous interaction localization, while Hand-Conditioned Residual Flow Matching produces diverse yet structurally consistent poses. Experiments validate HIGFlow on both forecasting tasks, while ablations confirm the contributions of its key components. Future work will explore human intent modeling and contact-aware conditioning to enable more physically consistent forecasts over longer horizons.

## REFERENCES

[1] C. Li, G. Wu, G. Y.-Y. Chan, D. G. Turakhia, S. Castelo Quispe, D. Li, L. Welch, C. Silva, and J. Qian, “Satori: Towards proactive ar assistant with belief-desire-intention user modeling,” in Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, 2025, pp. 1–24.

[2] Y. Pei, R. Huang, M. Zha, G. Wang, P. Wang, Q. Kang, Y. Yang, and H. T. Shen, “AttentionAR: Ar adaptation and warning for realworld safety via attention modeling and mllm reasoning,” in Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology, 2025, pp. 1–19.

[3] A. Noormohammadi-Asl, S. L. Smith, and K. Dautenhahn, “To lead or to follow? adaptive robot task planning in human–robot collaboration,” IEEE Transactions on Robotics, vol. 41, pp. 4215–4235, 2025.

[4] W. Bao, L. Chen, L. Zeng, Z. Li, Y. Xu, J. Yuan, and Y. Kong, “Uncertainty-aware state space transformer for egocentric 3D hand trajectory forecasting,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 13 656–13 665.

[5] M. Hatano, Z. Zhu, H. Saito, and D. Damen, “The invisible EgoHand: 3D hand forecasting through egobody pose estimation,” arXiv preprint arXiv:2504.08654, 2025.

[6] R. Liu, Y. Huang, L. Ouyang, C. Kang, and Y. Sato, “Sfhand: A streaming framework for language-guided 3d hand forecasting and embodied manipulation,” arXiv preprint arXiv:2511.18127, 2025.

[7] M. Chen, Y. Wang, Z. Li, H. Bharadhwaj, Y. Chen, C. Qin, Z. Kou, Y. Tian, E. Whitmire, R. Sodhi et al., “Flowing from reasoning to motion: Learning 3D hand trajectory prediction from egocentric human interaction videos,” arXiv preprint arXiv:2512.16907, 2025.

[8] L. Seminara, G. M. Farinella, and A. Furnari, “Task graph maximum likelihood estimation for procedural activity understanding in egocentric videos,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 9, pp. 10 518–10 534, 2026.

[9] T. Liu and B.-K. Bao, “Goal-guided prompting with adaptive modality selection for efficient assembly activity anticipation in egocentric videos,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 5, pp. 5945–5962, 2026.

[10] Y. Zheng, Y. Yang, K. Mo, J. Li, T. Yu, Y. Liu, C. K. Liu, and L. J. Guibas, “GIMO: Gaze-informed human motion prediction in context,” in European Conference on Computer Vision. Springer, 2022, pp. 676– 694.

[11] A. Avogaro, A. Toaiari, F. Cunico, X. Xu, H. Dafas, A. Vinciarelli, E. Li, and M. Cristani, “Exploring 3d human pose estimation and forecasting from the robot’s perspective: The HARPER dataset,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 5828–5835.

[12] Q. Jiang, B. Susam, J.-J. Chao, and V. Isler, “Map-aware human pose prediction for robot follow-ahead,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 13 031–13 038.

[13] Z. Liu, S. Wu, S. Jin, S. Ji, Q. Liu, S. Lu, and L. Cheng, “Investigating pose representations and motion contexts modeling for 3d motion prediction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 681–697, 2022.

[14] X. Shu, L. Zhang, G.-J. Qi, W. Liu, and J. Tang, “Spatiotemporal coattention recurrent neural networks for human-skeleton motion prediction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 6, pp. 3300–3315, 2021.

[15] D. Wei, H. Sun, B. Li, J. Lu, W. Li, X. Sun, and S. Hu, “Human joint kinematics diffusion-refinement for stochastic motion prediction,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 37, no. 5, 2023, pp. 6110–6118.

[16] L.-H. Chen, J. Zhang, Y. Li, Y. Pang, X. Xia, and T. Liu, “Human-MAC: Masked motion completion for human motion prediction,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 9510–9521.

[17] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[18] C. Curreli, D. Muhle, A. Saroha, Z. Ye, R. Marin, and D. Cremers, “Nonisotropic Gaussian diffusion for realistic 3D human motion prediction,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 1871–1882.

[19] K. Ashutosh, G. Pavlakos, and K. Grauman, “FICTION: 4D future interaction prediction from video,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 17 613–17 625.

[20] D. Damen, H. Doughty, G. M. Farinella, S. Fidler, A. Furnari, E. Kazakos, D. Moltisanti, J. Munro, T. Perrett, W. Price et al., “The EPIC-KITCHENS dataset: Collection, challenges and baselines,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 11, pp. 4125–4141, 2020.

[21] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu et al., “Ego4D: Around the world in 3,000 hours of egocentric video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 18 995–19 012.

[22] Y. Li, Z. Cao, A. Liang, B. Liang, L. Chen, H. Zhao, and C. Feng, “Egocentric prediction of action target in 3D,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 20 971–20 980.

[23] I. Fang, Y. Chen, Y. Wang, J. Zhang, Q. Zhang, J. Xu, X. He, W. Gao, H. Su, Y. Li et al., “EgoPAT3Dv2: Predicting 3D action target from 2D egocentric vision for human-robot interaction,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 3036–3043.

[24] P. Kratzer, S. Bihlmaier, N. B. Midlagajni, R. Prakash, M. Toussaint, and J. Mainprice, “MoGaze: A dataset of full-body motions that includes workspace geometry and eye-gaze,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 367–373, 2020.

[25] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, “SMPL: A skinned multi-person linear model,” in Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2023, pp. 851–866.

[26] Z. Qi, S. Wang, W. Zhang, and Q. Huang, “Uncertainty-boosted robust video activity anticipation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 7775–7792, 2024.

[27] S. A. Peirone, F. Pistilli, A. Alliegro, T. Tommasi, and G. Averta, “Hier-egopack: Hierarchical egocentric video understanding with diverse task perspectives,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 2, pp. 1917–1931, 2026.

[28] L. Mur-Labadia, R. Martinez-Cantin, J. J. Guerrero, G. M. Farinella, and A. Furnari, “Integrating affordances and attention models for short-term object interaction anticipation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 5, pp. 5425–5441, 2026.

[29] S. Liu, S. Tripathi, S. Majumdar, and X. Wang, “Joint hand motion and interaction hotspots prediction from egocentric videos,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 3272–3282.

[30] J. Ma, X. Chen, J. Xu, and H. Wang, “Diff-IP2D: Diffusion-based handobject interaction prediction on egocentric videos,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 4291–4298.

[31] J. Ma, X. Chen, W. Bao, J. Xu, and H. Wang, “MADiff: Motion-aware mamba diffusion models for hand trajectory prediction on egocentric videos,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 3, pp. 3250–3267, 2026.

[32] J. Ma, W. Bao, J. Xu, G. Sun, X. Chen, and H. Wang, “Novel diffusion models for multimodal 3D hand trajectory prediction,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 2408–2415.

[33] J. Ma, W. Bao, J. Xu, G. Sun, Y. Zheng, E. Zhang, X. Chen, and H. Wang, “Uni-Hand: Universal hand motion forecasting in egocentric views,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026, Early Access.

[34] Y. Zeng, X. Zhang, H. Li, J. Wang, J. Zhang, and W. Zhou, “X<sup>2</sup>- VLM: All-in-one pre-trained model for vision-language tasks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 3156–3168, 2023.

[35] S. Tian, R. Wang, H. Guo, P. Wu, Y. Dong, X. Wang, J. Yang, H. Zhang, H. Zhu, and Z. Liu, “Ego-r1: Agentic chain-of-tool-thought for ultralong egocentric video reasoning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026, Early Access.

[36] L.-H. Chen, S. Lu, A. Zeng, H. Zhang, B. Wang, R. Zhang, and L. Zhang, “MotionLLM: Understanding human behaviors from human motions and videos,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025, Early Access.

[37] Y. Liu, D. Yang, M. Zheng, and M.-H. Yang, “Anticipating object interactions via aggregation and distillation of spatio-temporal knowledge from vision language models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026, Early Access.

[38] C. Bao, J. Xu, X. Wang, A. Gupta, and H. Bharadhwaj, “HandsOnVLM: Vision-language models for hand-object interaction prediction,” arXiv preprint arXiv:2412.13187, 2024.

[39] Z. Chang, X. Zhang, S. Wang, S. Ma, and W. Gao, “Stau: a spatiotemporal-aware unit for video prediction and beyond,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 9, pp. 7916–7929, 2025.

[40] L. Maes, Q. L. Lidec, D. Scieur, Y. LeCun, and R. Balestriero, “LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels,” arXiv preprint arXiv:2603.19312, 2026.

[41] T. Fernando, H. Gammulle, S. Sridharan, S. Denman, and C. Fookes, “Remembering what is important: a factorised multi-head retrieval and auxiliary memory stabilisation scheme for human motion prediction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 3, pp. 1941–1957, 2024.

[42] J. Tang, J.-F. Hu, T. Liang, X. Lin, J. Sun, W.-S. Zheng, and J. Lai, “Human motion prediction via continual prior compensation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 5, pp. 5131–5146, 2026.

[43] S. Xu, Y.-X. Wang, and L.-Y. Gui, “Diverse human motion prediction guided by multi-level spatial-temporal anchors,” in European Conference on Computer Vision. Springer, 2022, pp. 251–269.

[44] J. Jeong, D. Park, and K.-J. Yoon, “Multi-agent long-term 3D human pose forecasting via interaction-aware trajectory conditioning,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 16 975–16 984.

[45] T. Yu, Y. Lin, J. Yu, Z. Lou, and Q. Cui, “Vision-guided action: Enhancing 3D human motion prediction with gaze-informed affordance in 3D scenes,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 12 335–12 346.

[46] W. Zhu, X. Ma, D. Ro, H. Ci, J. Zhang, J. Shi, F. Gao, Q. Tian, and Y. Wang, “Human motion generation: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 4, pp. 2430– 2449, 2023.

[47] M. Zhang, Z. Cai, L. Pan, F. Hong, X. Guo, L. Yang, and Z. Liu, “MotionDiffuse: Text-driven human motion generation with diffusion model,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 6, pp. 4115–4128, 2024.

[48] Y. Yang, Z. Huang, C. Xu, and S. He, “Lagrangian motion fields for long-term motion generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 2, pp. 1171–1184, 2026.

[49] G. Xu, J. Tao, W. Li, and L. Duan, “Learning semantic latent directions for accurate and controllable human motion prediction,” in European Conference on Computer Vision. Springer, 2024, pp. 56–73.

[50] G. Barquero, S. Escalera, and C. Palmero, “BeLFusion: Latent diffusion for behavior-driven human motion prediction,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 2317–2327.

[51] J. Sun and G. Chowdhary, “CoMusion: Towards consistent stochastic human motion prediction via motion diffusion,” in European conference on computer vision. Springer, 2024, pp. 18–36.

[52] S. Tian, M. Zheng, and X. Liang, “PrediFlow: A flow-based predictionrefinement framework for real-time human motion prediction in humanrobot collaboration,” arXiv preprint arXiv:2512.13903, 2025.

[53] K. Grauman, A. Westbury, L. Torresani, K. Kitani, J. Malik, T. Afouras, K. Ashutosh, V. Baiyya, S. Bansal, B. Boote et al., “Ego-Exo4D: Understanding skilled human activity from first -and third-person perspectives,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 19 383–19 400.

[54] X. Zhou, R. Girdhar, A. Joulin, P. Krahenb ¨ uhl, and I. Misra, “Detecting¨ twenty-thousand classes using image-level supervision,” in European conference on computer vision. Springer, 2022, pp. 350–368.

[55] A. Gupta, P. Dollar, and R. Girshick, “LVIS: A dataset for large vocabulary instance segmentation,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019, pp. 5351–5359.

[56] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[57] S. Shin, J. Kim, E. Halilaj, and M. J. Black, “WHAM: Reconstructing world-grounded humans with accurate 3D motion,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 2070–2080.

[58] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[59] L. Mur-Labadia, M. Muckley, A. Bar, M. Assran, K. Sinha, M. Rabbat, Y. LeCun, N. Ballas, and A. Bardes, “V-JEPA 2.1: Unlocking dense features in video self-supervised learning,” arXiv preprint arXiv:2603.14482, 2026.