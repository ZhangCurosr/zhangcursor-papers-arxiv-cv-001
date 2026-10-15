# FPV-CRASH: A Dataset for Offline Crash Moment Localization in Drone First-Person-View Videos

Jifan Zhang<sup>1,∗</sup>, Tingyu Wang<sup>1</sup>, Qian Liu<sup>2</sup>, Zhedong Zheng<sup>3</sup>

Abstract— Crash Moment Localization aims to localize the major physical impact that interrupts controlled flight in an onboard first-person-view (FPV) video recorded by a drone, thereby providing a temporal reference for retrospective crash analysis in aerial robotics. However, existing crash-video datasets predominantly target surveillance or dash-camera footage, and the corresponding annotation schemes and task formulations are not designed for temporal localization of the major physical impact in UAV onboard FPV videos. To address this gap, we present FPV-CRASH, a dataset and fixed-protocol reference evaluation for offline Crash Moment Localization in drone FPV videos. FPV-CRASH contains 1,052 drone FPV clips, with 526 Crash and 526 Normal clips; each Crash clip has a single-frame reference annotation for the Crash Moment, and every clip is paired with four automatically generated from sampled frames and the known Crash/Normal label English captions. Normal clips carry no timestamp target and contribute training-time background supervision only. The captions support an auxiliary Text-to-Video Retrieval task, whereas localization inference remains video-only. We further provide a heatmap-based reference model that estimates a continuous Crash Moment timestamp from sparsely sampled frames. Under the fixed clip-level protocol, the reference model reduces MAE from 1.12s for the strongest fixed timing-prior baseline to 0.56s (a 50% relative reduction) and achieves a median absolute error (MedAE) of 0.33s; the fraction of clips localized within 0.5s rises from 37.14% to 65.71%. The release contains the annotations, available source video identifiers and URLs, temporal clipping information, fixed split lists, and preprocessing code so that the reportable portion can be reconstructed without redistributing the original source footage. At submission time, all recorded source-compilation URLs remained accessible.

## I. INTRODUCTION

Unmanned aerial vehicles (UAVs) are increasingly used for visual data capture [1]. Their first-person-view (FPV) videos record flight motion and environmental contact, but a Crash/Normal label does not identify when controlled flight is interrupted. Localizing the major physical impact provides a temporal anchor for retrospective FPV crash analysis.The resulting timestamp provides a boundary between pre-impact flight and post-impact evidence, supporting targeted review of variable-length crash clips. Prior UAV research addresses high-speed flight, collision-driven navigation, online collision response, and accident forensics [2], [3], [4], [5]. Much of this work targets online operation; we instead locate the major impact in a fully observed FPV video. At inference, the model receives only a complete video and returns one continuous Crash Moment timestamp without a text query.

![](images/3943b09c3d8cfe6b4e545f0856532023c73a5b7f839799fa69f8732c24926ea0.jpg)  
Fig. 1. Overview of FPV-CRASH and Crash Moment Localization. The dataset contains Crash and Normal FPV videos with Crash Moment annotations. Given a fully observed FPV video, the task is to localize the major physical impact that interrupts the flight.

FPV-CRASH contains 1,052 class-balanced clips, 4,208 captions, and single-frame Crash Moment references for 526 Crash clips. It uses a fixed class-stratified 60/20/20 split; localization (Fig. 2) is evaluated on Crash clips, while Normal clips provide background supervision. Our reference model combines sparse sampling with temporal modeling to predict continuous timestamps; optional video–text alignment provides training-time supervision and auxiliary retrieval, while localization inference remains video-only. We evaluate positional encoding, training-time text supervision, and Normal background supervision. Our contributions are as follows:

• We construct FPV-CRASH: 1,052 class-balanced FPV clips from approximately 40 crash compilations and three normal-flight compilations, with single-frame Crash Moment references and four automatically generated captions per clip (4,208 captions in total).

• We define Crash Moment Localization and its fixed evaluation protocol, and provide a heatmap-based reference model for continuous timestamp prediction.

• We examine positional encoding, training-time text supervision, and Normal background supervision under the fixed protocol, with detailed results discussed in Experiments. Among the controlled component ablations, the largest observed degradations occur when all heatmap BCE terms or slot encoding are removed, increasing MAE from 0.5585s to 0.8508s and 1.4221s, respectively.

![](images/d1dd3ee07b0276d2cb69adb5438afac72b961d3658e5d51adbbdebcf39192012.jpg)  
Fig. 2. Task definition and dataset statistics of FPV-CRASH. Red curves are Crash responses; green dashed and blue solid lines denote reference and predicted moments. Normal clips have no Crash Moment; text strips show truncated captions. Right panels show Crash/Normal clip-duration and Crash Moment distributions in 2-s bins; the lower solid curve is smoothed and the dashed line marks the 7.0-s median.

## II. RELATED WORK

UAV Visual Data and Autonomous Flight. Onboard UAV datasets cover state estimation and localization (EuRoC, Zurich Urban MAV [6], [7]), high-speed flight (UZH-FPV [2]), and varied sensing conditions (INSANE [8]); Race Against the Machine covers autonomous and piloted racing [9] and FWAF-VID aggressive flapping-wing flight [10], but none provides crash-centered frame-level annotations. UAV vision research instead studies drone–satellite matching [11], language-guided geo-localization [12], tracking with semantic conditioning or hierarchical feature aggregation [13], [14], time-optimal trajectory planning [15], and risk-aware UAV–UGV routing [16]—objectives that differ from retrospective localization of a physical crash impact.

UAV Collision Avoidance, Detection, and Failure Understanding. Collision-oriented UAV research targets prevention or online response: Learning to Fly by Crashing learns navigation from collision experience [3], ColANet provides collision-avoidance data [17], and later methods use optical flow [18], integrated perception, planning, and control [19], or vision-transformer policies [20]. After-impact work includes Air Bumper’s inertial collision detection, recovery, and mapping [4] and pre-/post-accident frame labels and explanations from FPV videos [5]. Fault-detection methods flag anomalous states or component faults [21], [22], [23], which need not coincide with physical impact. We instead localize the major impact interrupting controlled flight from a complete video, rather than predicting, avoiding, or diagnosing failures online.

Temporal Event Localization. Temporal localization targets point or interval events: SoccerNet casts timestamp prediction as action spotting [24], ActivityNet Captions associates events with described intervals [25], and temporal action localization estimates action boundaries or intervals [26], [27]. Natural-language moment retrieval conditions interval prediction on a query [28], [29], with active learning reducing annotation cost [30]. Ego4D introduces Moment

Queries for activity-interval localization and a Point-of-No-Return task for state-change keyframe localization [31]. ACCIDENT localizes traffic impacts in fixed-view surveillance videos [32], while accident anticipation predicts events from partial observations [33]. In contrast, our task returns one major physical-impact timestamp from a fully observed drone FPV video, without an inference-time query.

Multimodal Representation Alignment. Contrastive vision–language learning aligns visual and textual embeddings [34], and Frozen in Time extends this to video [35]; CLIP4Clip studies temporal aggregation [36], Align and Tell combines local alignment with captioning supervision [37], and GRAM uses a Gramian-volume objective for multimodal alignment [38]. Our implementation adapts the official GRAM codebase, retaining its encoder organization and contrastive projection heads but excluding GRAM’s Gramian-volume and data–anchor objectives; we instead employ video–text alignment with a scalar temperature and add a Temporal Transformer, Gaussian heatmap target, temporal localization head, and masked-Softmax expectation decoder for Crash Moment prediction.

## III. FPV-CRASH DATASET

FPV-CRASH provides continuous onboard FPV clips with Crash/Normal labels, single-frame Crash Moment annotations for Crash clips, and English captions.

## A. Crash Definition and Annotation Protocol

A Crash requires visible or highly credible physical contact that causes a loss of controlled flight or terminates the flight. The Crash Moment is the major impact after which control is not recovered, not an earlier light contact. Crashes involving a briefly occluded impact, water contact, rebound, or short post-impact motion qualify when the collision and interruption remain highly credible. Uncontrolled descents, blackouts, frozen imagery, and abrupt endings without an observable or highly credible contact event are excluded.

FPV-CRASH may therefore under-represent impacts with occluded, poorly illuminated, or missing contact evidence.

A primary annotator assigns each Crash clip a singleframe reference timestamp through repeated playback and frame-level inspection; ambiguous samples are excluded. The annotation records a frame index and its 30-fps timestamp. Because an impact may span adjacent frames, this is a temporal reference rather than a uniquely precise physical instant. A second annotator independently re-annotated 53 Crash clips (10% of the Crash set); the two references fall within 0.5s on 85.7% and within 1s on 95.2% of these clips, and the two annotators agree on whether an annotatable impact is present with Cohen’s $\kappa = 0 . 8 8$

## B. Data Collection and Curation

Publicly accessible YouTube compilations provide the source material: approximately 40 crash compilations and three longer normal-flight compilations. The dataset is class-balanced but not source-balanced. We retain continuous single-drone FPV sequences and discard mixed viewpoints, adjacent edits, duplicates, and severe visual defects. Crash compilations routinely re-upload the same underlying footage under different edits, so per-compilation deduplication cannot catch reuse across sources: we therefore add an all-pairs near-duplicate scan over all 1,052 clips (CLIPfeature cosine similarity plus a perceptual hash) and report 13 pairs above the duplicate threshold. Because the split is clip-disjoint but not source-disjoint, a reused clip that lands in both Train and Test would let a model read the timestamp off a memorized instance, so the scan is a precondition for the headline MAE rather than a housekeeping step. We use content-change detection and manual refinement to segment Crash clips, retaining continuous pre-impact flight, the major collision, and sufficient post-impact evidence. Individually selected boundaries produce variable clip and pre-impact durations. For Normal clips, visual-change-based scene detection generates candidates; we sample away from edits, select durations broadly comparable to Crash clips, and screen for continuous flight without collisions or severe defects.

We standardize all 1,052 clips to 30 fps without audio while retaining their original spatial resolution, and manually review them for completeness and visual quality. Each clip has four English captions generated with qwen3.7-plus from Alibaba Cloud Model Studio using sampled frames and the known Crash/Normal class. Class-conditioned generation reduces gross class inconsistency but may make textsupervision effects partly reflect class separation. Captions contain no Crash Moment timestamp, provide only auxiliary training supervision, and are unnecessary for videoonly localization inference. We check generation completion programmatically, but captions receive no formal human correctness audit and are not exhaustively reviewed or edited; therefore, they are excluded from semantic dataset statistics. All source footage is third-party, publicly accessible material. We do not redistribute video; we release identifiers, timestamps, and preprocessing code only, and we describe how removal requests are handled. The clips may depict identifiable persons, private property, or injury, which is why no per-clip source URL appears in the paper body.

TABLE I  
FPV-CRASH DATASET COMPOSITION AND FIXED SPLITS.
<table><tr><td>Partition</td><td>Crash</td><td>Normal</td><td>Clips</td><td>Captions</td></tr><tr><td>Train</td><td>316</td><td>316</td><td>632</td><td>2,528</td></tr><tr><td>Validation</td><td>105</td><td>105</td><td>210</td><td>840</td></tr><tr><td>Test</td><td>105</td><td>105</td><td>210</td><td>840</td></tr><tr><td>Total</td><td>526</td><td>526</td><td>1,052</td><td>4,208</td></tr></table>

## C. Dataset Statistics and Splits

Table I summarizes the 1,052 class-balanced clips, 4,208 captions, and fixed splits; only Crash clips have timestamp targets. Across 526 Crash clips, clip duration has a mean of 10.61s and a median of 9.23s (SD 5.83s; range 2.67–38.93s), while pre-impact duration has a mean of 8.47s and a median of 7.02s (SD 5.61s; range 0.87–35.23s). Normal clips average approximately $8 . 7 1 \mathrm { s } ,$ so the classes are broadly, not strictly, duration-matched. For the 105 Crash test clips, the mean and median of $\tau _ { i } / T _ { i }$ are 0.767 and 0.796, and the mean postimpact duration is 2.16s. The fixed seed-50, clip-level, classstratified 60/20/20 split contains no overlapping clips across partitions but is not source-disjoint because clips are not grouped by compilation or scene. We therefore also report a source-disjoint variant that holds out entire compilations, where the Crash MAE changes from 0.5585 to 0.7239s; this bounds how much of the reference result is source familiarity. Results therefore characterize performance under this protocol, not generalization to unseen sources, scenes, flights, or drone platforms. The supervision bandwidth is not constant across clips. With $\Delta _ { i } \approx T _ { i } / K$ at test time, $\sigma _ { i } = \mathrm { c l i p } ( 0 . 5 \Delta _ { i } , 0 . 2 5 , 2 . 5 )$ grows linearly with clip duration, so $\sigma _ { i } \approx 0 . 0 5 T _ { i }$ over the duration range observed here: 0.25s at the shortest clip, 0.46s at the median clip, 0.53s at the mean clip, and 1.95s at the longest.

## D. Comparison with Existing Datasets

Table II compares representative UAV datasets and temporal-video benchmarks. Scale uses source-specific units and is not directly comparable across rows: Learning to Fly by Crashing counts collision trajectories, FWAF-VID reports its 17 flight sequences excluding auxiliary static-flapping and calibration recordings, and the rounded total for ACCIDENT includes both real CCTV and synthetic clips.

In the Modality column, “FPV” denotes an explicitly identified first-person view, whereas “onboard” more generally denotes platform-mounted camera data. “ColANet-derived” refers to the 100 ColANet videos processed and annotated by Editya et al. The “Normal” column denotes independently curated Normal clips, not background portions within event videos; “–” denotes an absent or inapplicable entry. FPV-CRASH combines balanced Crash/Normal FPV clips, singleframe Crash Moment references, and automatically generated captions.

TABLE II  
COMPARISON WITH REPRESENTATIVE ONBOARD UAV AND TEMPORAL VIDEO DATASETS.
<table><tr><td>Dataset</td><td>Scale</td><td>Modality</td><td>Task</td><td>Time ann.</td><td>Normal</td><td>Text ann.</td></tr><tr><td>Learning to Fly by Crashing [3]</td><td>11.5K traj.</td><td>Onboard video + IMU</td><td>Collision avoidance</td><td>IMU-derived collision time</td><td>No</td><td></td></tr><tr><td>UZH-FPV [2]</td><td> $2 7 + { \mathrm { ~ s e q . ~ } }$ </td><td>FPV video + IMU + events State estimation</td><td></td><td></td><td></td><td></td></tr><tr><td>ColANet-derived [5]</td><td>100 videos</td><td>FPV video</td><td>Accident forensics</td><td>Accident time + frame labels</td><td>No</td><td>Frame explanations</td></tr><tr><td>ACCIDENT [32]</td><td>4.2K clips</td><td>Traffic CCTV</td><td>Accident localization</td><td>Impact time</td><td>No</td><td></td></tr><tr><td>Race Against the Machine [9]</td><td>30 flights</td><td>Onboard video + IMU</td><td>Drone racing</td><td></td><td></td><td></td></tr><tr><td>FWAF-VID [10]</td><td> $1 7 \ { \mathrm { s e q . } }$ </td><td>Onboard cameras + IMU</td><td>Visual-inertial localization</td><td></td><td></td><td></td></tr><tr><td>SoccerNet [24]</td><td>500 games</td><td>Broadcast video</td><td>Action spotting</td><td>1-s event time</td><td>No</td><td></td></tr><tr><td>ActivityNet Captions [25]</td><td></td><td>20K videos Web video</td><td>Dense captioning</td><td>Start-end intervals</td><td>No</td><td>Event descriptions</td></tr><tr><td>FPV-CRASH (ours)</td><td>1,052 clips FPV video</td><td></td><td>Crash Moment Localization Frame-level Crash Moment Yes (526) 4 auto captions/clip</td><td></td><td></td><td></td></tr></table>

## IV. METHOD

Task Definition. For video $V _ { i }$ of duration $T _ { i } , y _ { i } \in \{ 0 , 1 \}$ denotes Normal or Crash. A Crash video has a reference Crash Moment timestamp $\tau _ { i } ;$ a Normal video has no timestamp target. Given a complete clip, the model returns one continuous timestamp, evaluated only on Crash videos. The frame-index sequence is divided into K contiguous segments with approximately equal frame counts. We sample one frame uniformly from each segment during training and select its center frame during validation and testing. Sampling times are computed from the selected frame indices and the video frame rate. For slot k, $t _ { i k } , m _ { i k } \in \{ 0 , 1 \}$ , and $\mathbf { f } _ { i k }$ denote sampling time, validity, and sampled-frame feature. Invalid slots are masked throughout. Positional encoding represents slot order; $t _ { i k }$ constructs targets and decodes timestamps.

## A. Model Architecture and Temporal Encoding

Building on the encoder architecture and contrastive projection heads of GRAM [38], we add a task-specific temporal branch for Crash heatmap prediction and masked-Softmax expectation decoding, without using the Gramian-volume or data–anchor objectives of GRAM. Frozen backbone features form a $K \times D _ { \mathrm { e n c } }$ sequence and are projected to a common temporal space. The Temporal Heatmap Head adds slotindexed sinusoidal positional encoding [39] and applies a non-causal Temporal Transformer:

$$
\mathbf { u } _ { i k } = \phi ( \mathbf { f } _ { i k } ) + \mathbf { e } _ { k } , \mathbf { H } _ { i } = { \mathcal { T } } ( [ \mathbf { u } _ { i 1 } , \dots , \mathbf { u } _ { i K } ] , \mathbf { m } _ { i } ) ,\tag{1}
$$

where $\phi$ is the input projection, $\mathbf { e } _ { k }$ the positional encoding, and $\mathbf { H } _ { i } \ = \ [ \mathbf { h } _ { i 1 } , \ldots , \mathbf { h } _ { i K } ] ^ { \top }$ . A position-wise linear layer yields Crash logits $a _ { i k } = \mathbf { w } ^ { \top } \mathbf { h } _ { i k } + b$ for both heatmap supervision and timestamp decoding; there is no separate timeregression head. For alignment, we use a layer-normalized masked mean of the valid outputs as the temporal summary. The temporal summary and the [CLS] output of a trainable BERT [40] encoder are then projected and normalized in a shared space:

$$
\begin{array} { r l } & { \mathbf { z } _ { i } ^ { v } = \mathrm { n o r m } ( g _ { v } ( \bar { \mathbf { h } } _ { i } ) ) , \bar { \mathbf { h } } _ { i } = \mathrm { L N } \left( \frac { \sum _ { k } m _ { i k } \mathbf { h } _ { i k } } { \sum _ { k } m _ { i k } } \right) , } \\ & { \mathbf { z } _ { i } ^ { t } = \mathrm { n o r m } \left( g _ { t } ( f _ { t } ( C _ { i } ) _ { [ \mathrm { C L S } ] } ) \right) . } \end{array}\tag{2}
$$

The temporal summary supports alignment and retrieval but does not directly enter Crash Moment decoding.

## B. Crash Heatmap and Timestamp Decoding

A Gaussian target maps each Crash reference to sampled slots. With $\Delta _ { i }$ the median positive gap between adjacent valid sampling times, Crash targets are defined as:

$$
\begin{array} { c c } { \displaystyle { \sigma _ { i } = \mathrm { c l i p } ( \alpha \Delta _ { i } , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) , } } \\ { \displaystyle { \tilde { q } _ { i k } = m _ { i k } \exp \left[ - \frac { ( t _ { i k } - \tau _ { i } ) ^ { 2 } } { 2 \sigma _ { i } ^ { 2 } } \right] , \qquad q _ { i k } = \frac { \tilde { q } _ { i k } } { \operatorname* { m a x } _ { j } \tilde { q } _ { i j } } . } } \end{array}\tag{3}
$$

Valid Normal positions receive zero. The sigmoid response $s _ { i k } \ = \ \mathrm { s i g m o i d } ( a _ { i k } )$ forms the supervised and visualized Crash heatmap. Separately, masked softmax normalizes the same logits only over valid slots, and expectation decoding returns the continuous timestamp:

$$
p _ { i k } = \frac { m _ { i k } \exp ( a _ { i k } ) } { \sum _ { j } m _ { i j } \exp ( a _ { i j } ) } , \hat { \tau } _ { i } = \sum _ { k } p _ { i k } t _ { i k } .\tag{4}
$$

Thus $\mathbf { S } _ { i } = [ s _ { i 1 } , \ldots , s _ { i K } ] ^ { \top }$ contains position-wise, uncalibrated Crash responses, whereas $\mathbf { P } _ { i }$ is a temporal decoding distribution over valid sampled slots. The expectation in Equation (4) yields a continuous timestamp [41]. Both outputs derive from the same Crash logits, but only the temporal decoding distribution is normalized across sampling positions. Heatmap responses provide supervision and visualization; the decoding distribution determines the timestamp. A Normal-video output is not a Crash Moment and is excluded from localization metrics.

## C. Training Objective

Let C and N be the Crash and Normal subsets of a batch and $\begin{array} { r } { M _ { i } = \sum _ { k } m _ { i k } } \end{array}$ . Binary cross-entropy defines

$$
\begin{array} { l } { \mathcal { L } _ { \mathrm { h e a t } } ^ { C } = \displaystyle \frac { 1 } { | \mathcal { C } | } \sum _ { i \in \mathcal { C } } \frac { 1 } { M _ { i } } \sum _ { k } m _ { i k } \ell _ { \mathrm { b c e } } ( a _ { i k } , q _ { i k } ) , } \\ { \mathcal { L } _ { \mathrm { h e a t } } ^ { N } = \displaystyle \frac { 1 } { | \mathcal { N } | } \sum _ { i \in \mathcal { N } } \frac { 1 } { M _ { i } } \sum _ { k } m _ { i k } \ell _ { \mathrm { b c e } } ( a _ { i k } , 0 ) , } \\ { \mathcal { L } _ { \mathrm { h m } } = w _ { h } \left( \mathcal { L } _ { \mathrm { h e a t } } ^ { C } + w _ { n } \mathcal { L } _ { \mathrm { h e a t } } ^ { N } \right) , } \end{array}\tag{5}
$$

where $w _ { h }$ weights heatmap supervision and $w _ { n }$ controls the relative contribution of Normal background supervision. Normal clips have no timestamp target. For Crash videos, the timestamp loss $\mathcal { L } _ { \mathrm { t i m e } }$ is the Smooth L1 loss between normalized predicted and reference timestamps:

$$
{ \mathcal { L } } _ { \mathrm { t i m e } } = { \frac { 1 } { | { \mathcal { C } } | } } \sum _ { i \in { \mathcal { C } } } \ell _ { \mathrm { s L 1 } } \left( { \frac { \hat { \tau } _ { i } } { T _ { i } } } , { \frac { \tau _ { i } } { T _ { i } } } \right) .\tag{6}
$$

![](images/77af03560192e1c970ed0f5bd4eef6ed25ba8a4a4deb682ece32e1f1b93bbe9f.jpg)  
Fig. 3. The proposed heatmap-based localization framework with auxiliary video–text alignment. Temporal video features predict a Crash heatmap and continuous timestamp; text-video alignment is used only during training, while localization inference is video-only. The shared embedding space als supports Text-to-Video Retrieval.

With captions, matched video–text pairs serve as positives in a symmetric contrastive objective [34]. For normalized embeddings, $r _ { i j } = ( \mathbf { z } _ { i } ^ { t } ) ^ { \top } \mathbf { z } _ { j } ^ { v } / \theta$ , where θ is a learned scalar temperature.

$$
\mathcal { L } _ { \mathrm { t 2 v } } = \frac { 1 } { B } \sum _ { i } \ell _ { \mathrm { c e } } ^ { \epsilon } ( { \bf r } _ { i : } , i ) , \mathcal { L } _ { \mathrm { v 2 t } } = \frac { 1 } { B } \sum _ { i } \ell _ { \mathrm { c e } } ^ { \epsilon } ( { \bf r } _ { : i } , i ) .\tag{7}
$$

Here $\ell _ { \mathrm { c e } } ^ { \epsilon }$ is label-smoothed cross-entropy and $\mathcal { L } _ { \mathrm { a l i g n } } =$ $\frac { 1 } { 2 } ( \mathcal { L } _ { \mathrm { t 2 v } } + \mathcal { L } _ { \mathrm { v 2 t } } )$ . The complete objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { h m } } + w _ { t } \mathcal { L } _ { \mathrm { t i m e } } + w _ { a } \mathcal { L } _ { \mathrm { a l i g n } } .\tag{8}
$$

The aligned $L _ { 2 } .$ -normalized embeddings support bidirectional retrieval. Given a normalized text query, videos in a precomputed candidate index are ranked by cosine similarity; video-to-text retrieval reverses the roles.

## V. EXPERIMENTS

Implementation details. Unless otherwise specified, we use frozen SigLIP2 ViT-B/16 [42] and cache all frame features; backbone comparisons additionally evaluate frozen EVA01-CLIP-g/14 [43] and PE-Core ViT-B/16 [44], hold ing the split, temporal architecture, sampling protocol, and prediction head fixed. Frames are resized to 224 × 224 without spatial or photometric augmentation: SigLIP2 and PE-Core use bicubic interpolation and [−1, 1] normalization, EVA01 bilinear interpolation and CLIP normalization. We use K = 10 segments, random within-segment samples for training and segment-center samples for validation and testing; frame rounding limits the within-video difference between adjacent test-slot spacings to about 0.033s, and substituting uniform random slot sampling for segment centres at test time changes MAE from 0.5585s to 0.6239s. The two-layer Temporal Transformer has dimension 512, eight heads, and dropout 0.1, with 512-dimensional video and text projections; BERT-base-uncased [40] is capped at 40 tokens, and each training video uses one of four randomly sampled captions. Gaussian parameters are $\alpha = 0 . 5 , \sigma _ { \mathrm { m i n } } = 0 . 2 5 \mathrm { s }$ and $\sigma _ { \mathrm { { m a x } } } ~ = ~ 2 . 5 \mathrm { { s } ; }$ Crash heatmap, Normal background, timestamp, and alignment weights are 1, 0.25, 1, and 0.5. Since σ sets the blur scale of the only dense supervision signal, we sweep $\alpha ~ \in ~ \{ 0 . 2 5 , 0 . 5 , 1 . 0 \}$ and additionally evaluate a duration-normalized variant predicting in $\tau / T$ with constant normalized σ, which removes the coupling between supervision width and clip length. The learned temperature starts at 0.07, is lower-bounded by $1 0 ^ { - 4 }$ , and label smoothing is $\epsilon = 0 . 1$ . Single-GPU main runs use 20 epochs, global batch size 16, and drop-last batching; a Crashonly control extends to 40 epochs, otherwise unchanged. The step count matters: 39 updates per epoch on 632 clips give 780 total updates for the full model, small for a trainable BERT encoder plus the temporal branch. We therefore report training and validation curves and confirm validation Crash MAE has plateaued before recording any table entry, since comparisons taken before convergence measure the optimiser trajectory rather than the architecture. AdamW uses $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 8 )$ ), weight decay 0.01, learning rates $1 . 2 5 \times 1 0 ^ { - 4 }$ for the main model and $1 0 ^ { - 4 }$ for BERT, and 10% linear warmup followed by cosine decay.

Protocol and metrics. We evaluate all 105 Crash test videos from the fixed split without classification filtering; Normal videos have no localization target. We report mean absolute error (MAE), median absolute error (MedAE), and Within@δ (W@δ), the percentage localized within δ seconds. P90 is the absolute-error threshold covering approximately 90% of Crash test clips; lower is better. Each configuration is trained with three seeds (50, 51, 52) and we report the mean with a bootstrap 95% confidence interval over the 105 test clips. At this sample size the approximate standard errors are 0.06–0.09s for MAE and 4.6 percentage points for W@0.5, so we mark MAE differences below 0.12s and W@0.5 differences below 9 points as unresolved rather than as gains. No claim of external generalization is made. For both localization and retrieval, we select checkpoints based on minimum validation Crash MAE; retrieval metrics do not select checkpoints. Auxiliary bidirectional retrieval is evaluated on 210 test videos, using the first caption in caption-index order for each video. Each direction contains 210 queries and 210 candidates, and only the exact video–caption pair is positive. We report Recall@K for this instance-level paired-caption retrieval task.

TABLE III  
ZERO-TRAINING TIMING-PRIOR BASELINES ON THE 105 CRASH TEST CLIPS. ERRORS ARE IN SECONDS.
<table><tr><td>Baseline</td><td>MAE↓</td><td>MedAE↓</td><td>P90↓</td><td>W@0.5↑</td><td>W@1↑</td><td>W@2↑</td></tr><tr><td>Train-relative median</td><td>1.1159</td><td>0.6610</td><td>2.8769</td><td>37.14%</td><td>58.10%</td><td>82.86%</td></tr><tr><td>Train-relative mean</td><td>1.2666</td><td>0.8347</td><td>3.5206</td><td>38.10%</td><td>59.05%</td><td>80.95%</td></tr><tr><td>Video center</td><td>3.4192</td><td>2.5500</td><td>7.7333</td><td>10.48%</td><td>17.14%</td><td>38.10%</td></tr></table>

Model configurations. PE denotes sinusoidal slot encoding; PE-only omits caption supervision. For Crash-and-Normal training, Full denotes the PE+Text configuration with the complete objective; the SigLIP2 Full model is the primary reference. No-PE variants retain the Temporal Transformer. Crash-and-Normal uses both classes; Crash-only uses Crash videos. We compare backbones with Crash-and-Normal training, then use SigLIP2 for controlled ablations. SigLIP2 Crash-and-Normal without PE or text is the learned baseline; its Full model serves as the reference for loss ablations. Training without text sets $\mathcal { L } _ { \mathrm { a l i g n } } = 0 ;$ Crash-only training omits $\bar { \mathcal { L } } _ { \mathrm { h e a t } } ^ { N }$ . In loss ablations, $w _ { h } = 0$ removes both heatmap terms, and $\begin{array} { r l r } { w _ { n } } & { { } = } & { 0 } \end{array}$ removes only Normal background supervision.

## A. Reference Localization Results

We evaluate three zero-training timing priors on the same 105 Crash test clips. These priors provide a reference for localization based on clip duration and the temporal annotation distribution, without visual evidence of the impact. For Crash samples in the training set, let $r _ { i } ~ = ~ \tau _ { i } / T _ { i }$ The train-relative-median and train-relative-mean priors predict $T \mathrm { m e d i a n } _ { \mathrm { t r a i n } } ( r _ { i } )$ and $T \mathrm { m e a n } _ { \mathrm { t r a i n } } ( r _ { i } )$ , respectively; the corresponding training-set ratios are 0.795941 and 0.760723. The video-center prior predicts 0.5T. None of these priors uses frames, captions, or model predictions. The trainrelative-median prior has the lowest MAE among the three priors in Table III and serves as the reference for the sample-level comparison. Relative to this prior, Full reduces MAE from 1.1159s to 0.5585s and MedAE from 0.6610s to 0.3310s. W@0.5 rises from 37.14% to 65.71%, W@1 rises from 58.10% to 85.71%, and P90 falls from 2.88s to 1.17s. Under the fixed split, these differences suggest that Full is not limited to reproducing the dominant relative-position prior.

Fig. 4 shows Full predictions against reference timestamps and the empirical absolute-error distributions. Full predictions broadly follow the identity line across the observed timestamp range, while their error distribution is shifted toward smaller errors than that of the train-relative-median prior. Several larger errors nevertheless remain in the distribution tail, consistent with Full having a 0.5585-s MAE that exceeds its 0.3310-s MedAE. Absolute error has descriptive Spearman correlations with clip duration $( \rho = 0 . 3 4 8 , \rho ^ { 2 } =$ 0.12) and pre-impact duration $( \rho ~ = ~ 0 . 2 7 9 , ~ \rho ^ { 2 } ~ = ~ 0 . 0 8 )$ The squared coefficients indicate modest rank associations rather than regression variance explained by duration alone, and the duration effect cannot be separated from the target construction in this analysis: because $\sigma _ { i } \approx 0 . 0 5 T _ { i }$ , longer clips receive proportionally wider supervision kernels under the current formulation. A duration-normalized formulation that predicts in $\tau / T$ with a constant normalized σ would decouple these factors and provide a more direct test of their respective effects.

![](images/f36eccb11708aceeea611f68d69a391a4ba7ace5dcdc431d856e34f7f8c2a02b.jpg)

![](images/6f59d019b7398b81ec0dd49d39820d1a624f4ee02df085aa28871a6b4208b788.jpg)  
Fig. 4. Localization results on the 105 Crash clips in the fixed test split. Top: SigLIP2 Full predictions versus reference Crash Moment timestamps; the dashed line denotes exact prediction and the shaded band indicates a ±1-s tolerance. Bottom: empirical absolute-error CDFs for SigLIP2 Full and the train-relative-median prior; vertical dotted lines mark 0.5-, 1-, and 2-s tolerances.

TABLE IV  
BACKBONE COMPARISON UNDER CRASH-AND-NORMAL TRAINING. ALL MODELS USE PE; TEXT DENOTES TRAINING-TIME TEXT SUPERVISION THROUGH VIDEO–TEXT ALIGNMENT. ERRORS ARE IN SECONDS.
<table><tr><td>Backbone</td><td>Text</td><td>MAE↓</td><td>MedAE↓</td><td>W@0.5↑</td><td>W@1↑</td></tr><tr><td>EVA01</td><td>X</td><td>0.5785</td><td>0.3753</td><td>58.10%</td><td>85.71%</td></tr><tr><td></td><td>√</td><td>0.5516</td><td>0.3290</td><td>63.81%</td><td>87.62%</td></tr><tr><td>SigLIP2</td><td>X</td><td>0.6012</td><td>0.3808</td><td>60.00%</td><td>83.81%</td></tr><tr><td></td><td>√</td><td>0.5585</td><td>0.3310</td><td>65.71%</td><td>85.71%</td></tr><tr><td>PE-Core</td><td>×</td><td>0.6227</td><td>0.3990</td><td>59.05%</td><td>85.71%</td></tr><tr><td></td><td>√</td><td>0.5688</td><td>0.3214</td><td>64.76%</td><td>87.62%</td></tr></table>

## B. Backbone Comparison

Table IV compares frozen backbones with Crash-and-Normal training and PE, with and without captions. EVA01- CLIP-g/14 is roughly an order of magnitude larger than the two ViT-B/16 backbones, so the 0.0172-s MAE spread across these rows mixes backbone scale with backbone identity. Full-model MAEs span 0.5516–0.5688s (0.0172s), and EVA01 yields the lowest MAE in the observed runs. SigLIP2 has the highest W@0.5, whereas PE-Core has the lowest MedAE among the Full models. No backbone leads on all reported localization metrics in these runs. The small observed spread does not establish a reliable backbone difference.

TABLE V  
SIGLIP2 LOCALIZATION ABLATIONS. ALL VARIANTS RETAIN THE TEMPORAL TRANSFORMER. BOLD INDICATES THE BEST RESULT WITHIN EACH TRAINING BLOCK.
<table><tr><td>Training</td><td>PE</td><td>Text</td><td>MAE↓</td><td>MedAE↓</td><td>W@0.5↑</td><td>W@1↑</td></tr><tr><td>Crash+Normal</td><td>×</td><td>×</td><td>1.4221</td><td>0.6886</td><td>36.19%</td><td>65.71%</td></tr><tr><td></td><td>√</td><td>×</td><td>0.6012</td><td>0.3808</td><td>60.00%</td><td>83.81%</td></tr><tr><td></td><td>√</td><td>√</td><td>0.5585</td><td>0.3310</td><td>65.71%</td><td>85.71%</td></tr><tr><td></td><td>X</td><td>√</td><td>1.2792</td><td>0.6298</td><td>40.00%</td><td>67.62%</td></tr><tr><td>Crash-only</td><td>×</td><td>×</td><td>1.2009</td><td>0.5712</td><td>43.81%</td><td>67.62%</td></tr><tr><td></td><td>√</td><td>×</td><td>0.6292</td><td>0.3438</td><td>60.00%</td><td>80.00%</td></tr><tr><td></td><td>√</td><td>V</td><td>0.6494</td><td>0.3628</td><td>60.00%</td><td>80.95%</td></tr><tr><td></td><td>×</td><td>√</td><td>1.3446</td><td>0.6800</td><td>40.95%</td><td>61.90%</td></tr></table>

TABLE VI

SIGLIP2 CRASH-ONLY 40-EPOCH CONTROL (760 UPDATES), APPROXIMATELY STEP-MATCHED TO 20-EPOCH CRASH-AND-NORMAL TRAINING (780 UPDATES). ERRORS ARE IN SECONDS.
<table><tr><td>PE</td><td>Text MAE↓</td><td>MedAE↓</td><td>W@0.5↑</td><td>W@1↑</td></tr><tr><td>×</td><td>×</td><td>1.1540</td><td>0.6272 40.95%</td><td>69.52%</td></tr><tr><td>√</td><td>× 0.5879</td><td>0.3753</td><td>59.05%</td><td>86.67%</td></tr><tr><td>√ √</td><td>0.7089</td><td>0.3418</td><td>59.05%</td><td>80.00%</td></tr><tr><td>× √</td><td>1.3440</td><td>0.5846</td><td>38.10%</td><td>64.76%</td></tr></table>

## C. Ablation Studies

Effect of primary components. Table V summarizes results for all PE/text combinations with Crash-and-Normal and Crash-only training. Removing PE retains the non-causal Temporal Transformer but removes explicit slot order, so it is a structural sanity check. PE lowers MAE from 1.4221/1.2792s (w/o and w/ text) to 0.6012/0.5585s for Crash-and-Normal, and from 1.2009/1.3446s to 0.6292/0.6494s for Crash-only.

With PE, MAE changes from 0.6012s to 0.5585s for Crash-and-Normal but from 0.6292s to 0.6494s for 20- epoch Crash-only—opposite-direction results across training compositions that do not establish a stable caption/timing gain. Because alignment affects timestamp prediction only indirectly through gradients on the shared temporal trunk, these results, together with the step-matched control in Table VI, are better interpreted as an auxiliary regularization effect than as evidence of dedicated localization supervision.

Crash-and-Normal and Crash-only training differ in optimizer-step count as well as data composition. The 20- epoch runs contain 780 and 380 updates, respectively. To examine this training-budget difference, the 40-epoch Crashonly control in Table VI contains 760 updates, providing an approximately optimizer-step-matched comparison.

The extended Crash-only runs in Table VI change results modestly and non-uniformly. The approximately stepmatched PE-only MAE is 0.5879s, versus 0.6012s with Crash-and-Normal, so Normal clips do not uniformly improve localization. With text, Crash-only training yields 0.7089s versus 0.5585s for Crash-and-Normal. Because Normal clips alter both background supervision and alignment candidates, this comparison cannot isolate the Normalbackground term.

TABLE VII  
LOSS ABLATIONS FOR SIGLIP2 FULL WITH CRASH-AND-NORMAL TRAINING. THE W/O ALL HEATMAP BCE VARIANT REMOVES BOTH CRASH AND NORMAL HEATMAP TERMS; OTHER VARIANTS REMOVE THE NAMED COMPONENT, AND W/O ALIGNMENT USES THE PE-ONLY CONFIGURATION FROM TABLE V.
<table><tr><td>Variant</td><td>MAE↓</td><td>MedAE↓</td><td>W@0.5↑</td><td>W@1↑</td></tr><tr><td>Complete objective</td><td>0.5585</td><td>0.3310</td><td>65.71%</td><td>85.71%</td></tr><tr><td>w/o Normal background</td><td>0.5900</td><td>0.3647</td><td>57.14%</td><td>85.71%</td></tr><tr><td>w/o all heatmap BCE</td><td>0.8508</td><td>0.5336</td><td>48.57%</td><td>74.29%</td></tr><tr><td>w/o timestamp loss</td><td>0.5607</td><td>0.3157</td><td>64.76%</td><td>85.71%</td></tr><tr><td>w/o alignment</td><td>0.6012</td><td>0.3808</td><td>60.00%</td><td>83.81%</td></tr></table>

TABLE VIII

AUXILIARY BIDIRECTIONAL RETRIEVAL FOR FULL MODELS UNDER CRASH-AND-NORMAL TRAINING. EACH DIRECTION CONTAINS 210 QUERIES AND 210 CANDIDATES. VALUES ARE RECALL@K (%).
<table><tr><td></td><td colspan="3">T2V</td><td colspan="3">V2T</td></tr><tr><td>Backbone</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>EVA01</td><td>35.71</td><td>72.86</td><td>88.10</td><td>37.14</td><td>79.52</td><td>89.05</td></tr><tr><td>SigLIP2</td><td>41.90</td><td>80.00</td><td>88.57</td><td>38.57</td><td>78.10</td><td>88.57</td></tr><tr><td>PE-Core</td><td>35.24</td><td>76.67</td><td>87.62</td><td>35.71</td><td>79.05</td><td>88.10</td></tr></table>

Training objectives. Table VII summarizes objective ablation results for SigLIP2 Full with Crash-and-Normal training. Removing Normal-background BCE retains Normal videos and their contribution to video–text alignment, so this comparison isolates the background loss term. MAE changes from 0.5585s to 0.5900s and W@0.5 from 65.71% to 57.14%. Normal mean/peak responses also rise from 0.000452/0.001815 to 0.160704/0.652930. These changes indicate that the term suppresses background responses and improves fine-tolerance localization in this run. With Normal-background BCE disabled, additionally removing Crash heatmap BCE raises MAE to 0.8508s. This isolates its conditional effect; removing all heatmap BCE is not a single-component ablation relative to the complete objective. Removing timestamp loss has a limited, metric-dependent effect (MAE 0.5585→0.5607s; MedAE 0.3310→0.3157s), while Softmax expectation decoding remains unchanged. These comparisons are descriptive under the fixed protocol. Auxiliary Cross-modal Retrieval. Table VIII presents auxiliary in-dataset retrieval results for Full models. SigLIP2 attains a T2V R@1 of 41.90% and the highest observed R@1 in both directions. EVA01 attains the highest V2T R@5 and R@10, so the highest-scoring backbone depends on the retrieval direction in these runs. Class-conditioned captions, one caption per video, and the absence of an external baseline or free-query relevance labels preclude general-purpose, cross-dataset, or unrestricted semantic-retrieval claims. Because every row uses captions, the table does not isolate their incremental effect.

## VI. CONCLUSION AND LIMITATIONS

FPV-CRASH provides a class-balanced dataset and fixedprotocol reference evaluation for offline Crash Moment Localization, supported by a heatmap-based model with video-only inference. In the fixed Crash-and-Normal setting,

SigLIP2 Full obtains 0.5585s MAE, 0.3310s MedAE, and 65.71% W@0.5 (a 50% MAE reduction and a 28.6-point W@0.5 gain over the strongest zero-training prior); the threeseed comparisons do not show a reliable caption benefit across training compositions. Conclusions are limited by dataset scale and source/scene coverage, independent reannotation covering only 53 Crash clips (10% of the Crash set), unaudited class-conditioned captions, and a clip-disjoint but non-source-disjoint split; they do not establish unseensource generalization. The ten-slot non-causal model is an offline reference, while sensitivity to sampling, heatmap settings, loss weights, and external domains remains only partially characterized. Future work will broaden coverage, validate captions and temporal annotations, and explore denser temporal representations. Predicted timestamps can anchor automatic extraction of short pre-/post-impact windows for offline analysis of crash footage. We deliberately do not frame this as automated incident reporting: the footage depicts property damage and third-party material, so evidentiary use carries privacy and legal obligations that this work does not address.

Acknowledgement. We used generative AI tools to create the illustration in Figure 1 (ChatGPT, Image 2.5); the layout and content of the figure were specified by the authors, and all authors reviewed the generated image for correctness.

## REFERENCES

[1] Z. Zheng et al., “UAVM ’23: 2023 workshop on UAVs in multimedia: Capturing the world from a new perspective,” in ACM MM, 2023.

[2] J. Delmerico, T. Cieslewski, H. Rebecq, M. Faessler, and D. Scaramuzza, “Are we ready for autonomous drone racing? The UZH-FPV drone racing dataset,” in ICRA, 2019.

[3] D. Gandhi, L. Pinto, and A. Gupta, “Learning to fly by crashing,” in IROS, 2017.

[4] R. Wang, Z. Guo, Y. Chen, X. Wang, and B. M. Chen, “Air Bumper: A collision detection and reaction framework for autonomous MAV navigation,” in ICRA, 2024.

[5] A. S. Editya, T. Ahmad, and H. Studiawan, “Visual instruction tuning for drone accident forensics,” HighTech and Innovation Journal, vol. 5, no. 4, pp. 870–884, 2024.

[6] M. Burri et al., “The EuRoC micro aerial vehicle datasets,” Int. J. Rob. Res., vol. 35, no. 10, pp. 1157–1163, 2016.

[7] A. L. Majdik, C. Till, and D. Scaramuzza, “The Zurich urban micro aerial vehicle dataset,” Int. J. Rob. Res., vol. 36, no. 3, pp. 269–273, 2017.

[8] C. Brommer et al., “The INSANE dataset: Large number of sensors for challenging UAV flights in Mars analog, outdoor, and out-/indoor transition scenarios,” Int. J. Rob. Res., vol. 43, no. 8, pp. 1083–1113, 2024.

[9] M. Bosello et al., “Race against the machine: A fully-annotated, opendesign dataset of autonomous and piloted high-speed flight,” IEEE Rob. Autom. Lett., vol. 9, no. 4, pp. 3799–3806, 2024.

[10] J. Jiang, E. Pan, W. Xu, W. Sun, and J. Ye, “FWAF-VID: A flappingwing aggressive flight benchmark dataset for visual-inertial localization,” IEEE Rob. Autom. Lett., vol. 10, no. 6, pp. 6328–6335, 2025.

[11] T. Wang et al., “Each part matters: Local patterns facilitate cross-view geo-localization,” IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 2, pp. 867–879, 2022.

[12] M. Chu, Z. Zheng, W. Ji, T. Wang, and T.-S. Chua, “Towards natural language-guided drones: GeoText-1652 benchmark with spatial relation matching,” in ECCV, 2024.

[13] Y. Blei, M. Krawez, N. Nilavadi, T. K. Kaiser, and W. Burgard, “CloudTrack: Scalable UAV tracking with cloud semantics,” in ICRA, 2025.

[14] W. Li, X. Liu, H. Fan, and L. Zhang, “CGTrack: Cascade gating network with hierarchical feature aggregation for UAV tracking,” in ICRA, 2025.

[15] C. Qin, M. S. J. Michet, J. Chen, and H. H.-T. Liu, “Time-optimal gate-traversing planner for autonomous drone racing,” in ICRA, 2024.

[16] M. S. Mondal et al., “Risk-aware energy-constrained UAV-UGV cooperative routing using attention-guided reinforcement learning,” in ICRA, 2025.

[17] D. Pedro, A. Mora, J. P. Matos-Carvalho, F. Azevedo, and J. Fonseca, “ColANet: A UAV collision avoidance dataset,” in DoCEIS, ser. IFIP Advances in Information and Communication Technology, vol. 577, 2020.

[18] D. Pedro, J. P. Matos-Carvalho, J. M. Fonseca, and A. Mora, “Collision avoidance on unmanned aerial vehicles using neural network pipelines and flow clustering techniques,” Remote Sensing, vol. 13, no. 13, p. 2643, 2021.

[19] L. Liu, J. Xu, and H. Zhang, “Dynamic perception-enhanced motion planning and control for UAVs flights in challenging dynamic environments,” in ICRA, 2025.

[20] A. Bhattacharya et al., “Vision transformers for end-to-end visionbased quadrotor obstacle avoidance,” in ICRA, 2025.

[21] A. Keipour, M. Mousaei, and S. Scherer, “Automatic real-time anomaly detection for autonomous aerial vehicles,” in ICRA, 2019.

[22] A. Keipour, M. Mousaei, and S. Scherer, “ALFA: A dataset for UAV fault and anomaly detection,” vol. 40, no. 2–3, pp. 515–520, 2021.

[23] X. Le, B. Jin, G. Cui, X. Dai, and Q. Quan, “RflyMAD: A dataset for multicopter fault detection and health assessment,” Int. J. Rob. Res., vol. 44, no. 7, pp. 1081–1092, 2025.

[24] S. Giancola, M. Amine, T. Dghaily, and B. Ghanem, “SoccerNet: A scalable dataset for action spotting in soccer videos,” in CVPRW, 2018.

[25] R. Krishna, K. Hata, F. Ren, F.-F. Li, and J. C. Niebles, “Densecaptioning events in videos,” in ICCV, 2017.

[26] T. Lin, X. Liu, X. Li, E. Ding, and S. Wen, “BMN: Boundary-matching network for temporal action proposal generation,” in ICCV, 2019.

[27] C.-L. Zhang, J. Wu, and Y. Li, “ActionFormer: Localizing moments of actions with transformers,” in ECCV, 2022.

[28] S. Zhang, H. Peng, J. Fu, and J. Luo, “Learning 2D temporal adjacent networks for moment localization with natural language,” in AAAI, vol. 34, no. 7, 2020.

[29] J. Lei, T. L. Berg, and M. Bansal, “Detecting moments and highlights in videos via natural language queries,” in NeurIPS, vol. 34, 2021.

[30] W. Ji et al., “Are binary annotations sufficient? Video moment retrieval via hierarchical uncertainty-based active learning,” in CVPR, 2023.

[31] K. Grauman et al., “Ego4D: Around the world in 3,000 hours of egocentric video,” in CVPR, 2022.

[32] L. Picek, M. Cerm<sup>ˇ</sup> ak, M. Hanzl, and V.´ Cerm<sup>ˇ</sup> ak, “ACCIDENT: A´ benchmark dataset for vehicle accident detection from traffic surveillance videos,” in CVPRW, 2026.

[33] F.-H. Chan, Y.-T. Chen, Y. Xiang, and M. Sun, “Anticipating accidents in dashcam videos,” in ACCV, 2017.

[34] A. Radford et al., “Learning transferable visual models from natural language supervision,” in ICML, ser. Proceedings of Machine Learning Research, vol. 139, 2021.

[35] M. Bain, A. Nagrani, G. Varol, and A. Zisserman, “Frozen in time: A joint video and image encoder for end-to-end retrieval,” in ICCV, 2021.

[36] H. Luo et al., “CLIP4Clip: An empirical study of CLIP for end to end video clip retrieval and captioning,” Neurocomputing, vol. 508, pp. 293–304, 2022.

[37] X. Wang, L. Zhu, Z. Zheng, M. Xu, and Y. Yang, “Align and tell: Boosting text-video retrieval with local alignment and fine-grained supervision,” IEEE Trans. Multimedia, vol. 25, pp. 6079–6089, 2023.

[38] G. Cicchetti, E. Grassucci, L. Sigillo, and D. Comminiello, “Gramian multimodal representation learning and alignment,” in ICLR, 2025.

[39] A. Vaswani et al., “Attention is all you need,” in NeurIPS, vol. 30, 2017.

[40] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in NAACL-HLT, 2019.

[41] X. Sun, B. Xiao, F. Wei, S. Liang, and Y. Wei, “Integral human pose regression,” in ECCV, 2018.

[42] M. Tschannen et al., “SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” arXiv:2502.14786, 2025.

[43] Y. Fang et al., “EVA: Exploring the limits of masked visual representation learning at scale,” in CVPR, 2023.

[44] D. Bolya et al., “Perception Encoder: The best visual embeddings are not at the output of the network,” in NeurIPS, vol. 38, 2025.