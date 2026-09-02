# Beyond the Image Plane: World-Grounded Queries for Multi-Object Tracking

Orcun Cetintas<sup>1,2</sup> Guillem Braso´<sup>1</sup> Tim Meinhardt<sup>1</sup> Laura Leal-Taixe´<sup>1</sup> <sup>1</sup>NVIDIA <sup>2</sup>Technical University of Munich

## Abstract

Monocular videos record 3D scenes as sequences of 2D image-plane projections, obscuring depth and spatial relationships. Multi-object trackers localize and associate objects primarily using appearance and geometry observed only in the image plane, inheriting these ambiguities. To address this limitation, we introduce PLANET, an end-toend multi-object tracker designed to move beyond the image plane. As an enabling step, we lift existing 2D tracking datasets into 3D. We thenform world-grounded queries by embedding reconstructed 3D scene geometry into the features and positional encodings used during query formation. An auxiliary 3D location prediction task further encourages the queries to encode object positions during training. A complementary dual-resolution temporal memory preserves this evidence across longer temporal gaps. As a result, PLANET achieves state-of-the-art performance across three diverse benchmarks.

## 1. Introduction

Multi-object tracking aims to localize multiple objects in a video while preserving their identities over time. Trackingby-detection has long served as the dominant paradigm for multi-object tracking, decomposing the problem into object detection followed by a separately designed dataassociation stage [4, 6, 21, 48, 51]. More recently, end-toend trackers have begun to replace this separation, primarily through Transformer-based architectures that connect detection and identity reasoning with object queries [11, 23, 32, 46]. Although these models commonly operate with limited temporal context [11, 23, 28, 46], their querybased formulation brings the two tasks into a shared trainable model and provides a flexible basis for improving the representations used by both. Despite their different formulations, both families face a common set of challenges, including occlusion, camera motion, object motion, viewpoint changes, and similar object appearances [33].

A central source of these difficulties is the mismatch between the physical scene and its visual observation: objects occupy and move through a 3D world, while a camera records their projection onto a 2D image plane. This projection obscures depth and spatial configuration, allowing distinct objects to overlap in the image and making physical motion ambiguous in 2D. Trackers reason primarily from appearance and image-plane position or motion, and therefore inherit these geometric ambiguities. 3D information offers a natural way to reduce these ambiguities. A line of prior work shows that 3D cues can aid occlusion reasoning [38], camera-motion compensation [14], trajectory forecasting [8], and association [24]. However, these approaches are generally tied to category-specific 3D models or handcrafted downstream association, leaving the integration of 3D information into multi-object trackers underexplored.

Until recently, recovering sufficiently complete scene geometry and camera information from monocular video relied largely on costly optimization-based pipelines that could struggle in challenging conditions. Recent feedforward geometry models, including VGGT [35] and Depth Anything 3 [17], make this process more efficient and robust by estimating dense 3D scene structure and camera parameters from image sequences with a fully learnable model. These new capabilities create an opportunity to embed estimated 3D scene information directly into 2D tracking models, enriching the internal representations used for detection and association. This geometric grounding provides an opportunity for trackers to move beyond the image plane and reason about the underlying scene.

Motivated by moving beyond the image PLANE for Tracking, we introduce PLANET , an end-to-end multiobject tracking model that incorporates 3D scene information. Our approach proceeds in two stages. First, using Depth Anything 3 [17], we lift three existing 2D tracking datasets into 3D: DanceTrack [33], SportsMOT [7], and BFT [50]. In doing so, we recover dense world-coordinate point maps and associate each annotated 2D bounding box with a pseudo-3D object center. Second, PLANET incorporates this lifted scene information into the tracker through 3D embeddings and auxiliary 3D location prediction. The 3D embeddings expose detector queries to the reconstructed scene geometry, while auxiliary 3D location prediction encourages each matched query to encode its object’s 3D location. Together, these components form world-grounded queries that enrich the representation shared by detection and association with both scene-level geometry and objectlevel 3D supervision.

World-grounded queries can provide stronger spatial evidence, but this evidence can support association after an occlusion only if it remains available to the tracker. However, existing end-to-end trackers commonly operate within a limited temporal horizon because retaining and processing a growing query history is computationally costly [11, 23, 28, 46]. To address this, PLANET introduces dualresolution temporal memory, which preserves recent observations densely while sampling older observations more sparsely. Under the same context budget, PLANET expands the accessible temporal horizon from 30 frames in MOTIP [12] to approximately 200 frames.

We evaluate PLANET on DanceTrack [33], SportsMOT [7], and BFT [50], which collectively cover challenging human motion, sports scenes, and dense non-human motion. Across all three benchmarks, PLANET achieves state-of-the-art performance. Controlled ablations further examine how 3D embeddings, auxiliary 3D location prediction, and dual-resolution temporal memory contribute to tracking performance. Together, these results highlight 3D information as a valuable source of evidence for multi-object tracking. Upon acceptance, we will release our code, trained models, and the 3D-enriched versions of the datasets.

## 2. Related Work

Multi-Object Tracking. Tracking-by-detection decomposes multi-object tracking into per-frame object detection followed by a separately designed data-association stage. This stage commonly links detections using appearance descriptors [39], image-plane motion and spatial overlap [3, 5], detection confidence [48], or learned motion models [21].

End-to-end approaches instead optimize detection and identity association jointly. Early Transformer-based trackers such as TransTrack [32] and TrackFormer [23] connect the two tasks through learned object and track queries; subsequent methods follow and improve this paradigm [11, 28, 42, 46]. MOTIP provides a particularly direct formulation in which detection and ID prediction operate on the same detector-query representations through their respective prediction heads [12].

Despite this progress, end-to-end trackers also face a temporal-context bottleneck. Extending the attention horizon requires retaining and processing more query observations, which increases computational cost and distributes attention over an increasingly long and redundant history. Consequently, identity-relevant evidence from distant frames can become diluted. Reflecting this trade-off, many recent end-to-end Transformer trackers use short training contexts of only two to ten frames [11, 23, 28, 46, 49]. This creates a need to extend temporal coverage while retaining the most useful recent and historical evidence.

Building on MOTIP’s shared-query formulation, PLANET grounds queries in reconstructed 3D geometry, allowing the same representation to inform detection and association. A complementary dual-resolution memory preserves dense short-term and sparse long-term context, extending the temporal support available for identity prediction under a fixed context budget.

3D Cues for Monocular Tracking. Previous work incorporates 3D cues through category-specific representations, including pedestrian visibility reasoning [38], world-space trajectories [14], and 3D human models [26]. Such representations aid occlusion reasoning and camera-motion compensation, but rely on category-specific models or assumptions about ego-motion.

Occlusion-focused approaches instead pursue object permanence through recurrent invisible-object supervision [34], depth-constrained Kalman forecasting [15], or bird’s-eye-view trajectory prediction [8]. Here, geometry is handled by separate forecasting, search, or re-identification modules and remains independent of the tracker’s interna object representation.

Geometry has also been incorporated into conventional 2D tracking pipelines during downstream association, through handcrafted geometric and appearance costs [30], coupled 2D–3D filtering [24], depth and motion heuristics [25], or learned depth–motion–appearance likelihoods [22]. Despite their different fusion strategies, these methods introduce geometry only after the object representations have been formed, leaving the overall tracking system as a multi-stage pipeline rather than a unified model.

In contrast, PLANET remains an end-to-end monocular tracker but introduces reconstructed scene geometry directly into query formation. Dense point maps and auxiliary 3D center localization produce world-grounded queries before downstream association, allowing the same representation to benefit both detection and identity reasoning. This avoids specialized category-specific models, external forecasting, and handcrafted cues.

## 3. Methodology

## 3.1. Preliminaries

We build PLANET on MOTIP [12], which connects a querybased DETR detector to an ID prediction module within a single end-to-end tracking model. MOTIP casts association as (K + 1)-way in-context classification over a learnable ID dictionary, and its prediction head operates directly on the detector queries. This formulation makes detection and identity prediction jointly trainable within a single model.

![](images/c8a9908807fac8c49291751f8131004144e628d8ffd6030c3cc64cbfd13dad03.jpg)  
Figure 1. Lifting 2D tracking benchmarks into 3D. We enrich existing 2D tracking benchmarks with reconstructed scene geometry and 3D object labels.

Formally, given a frame $I _ { t } ,$ the detector produces a set of object-level query embeddings $\mathcal { Q } _ { t } = \{ \mathbf { q } _ { t } ^ { i } \} _ { i = 1 } ^ { \bar { N } _ { t } }$ , with $\mathbf { q } _ { t } ^ { i } \in$ $\mathbb { R } ^ { d }$ . Each query predicts a 2D bounding box $\hat { \mathbf b } _ { t } ^ { i } \in \mathbb { R } ^ { 4 }$ and an identity label $\hat { y } _ { t } ^ { i } \in \mathcal { V }$ , yielding the frame-level output $\widehat { \mathcal { O } } _ { t } = \{ ( \hat { \mathbf { b } } _ { t } ^ { i } , \hat { y } _ { t } ^ { i } ) \} _ { i = 1 } ^ { N _ { t } }$ . Predictions with the same identity are linked across frames to form a trajectory.

With this formulation, bounding-box prediction and identity association operate on the same queries and therefore meet in a shared object representation. Improving what the queries encode can therefore benefit both tasks simultaneously. PLANET acts at precisely this interface by grounding the queries in estimated 3D scene structure. The resulting queries describe not only how an object appears in the image, but also where it lies within the reconstructed scene, giving the tracker a richer basis for localization and identity reasoning.

## 3.2. Method Overview

Building on the observation that image-plane trajectories arise from objects moving through a 3D world, we bring estimated scene geometry into the query representations of a 2D tracker. As an enabling first stage, we use the geometry foundation model Depth Anything 3 [17] to lift existing monocular tracking datasets into 3D by recovering dense world-coordinate point maps. Since these maps do not directly provide object-level information, we introduce a dedicated assignment procedure that associates each annotated

2D box with a pseudo-3D label representing its estimated center in the reconstructed scene (Fig. 1).

The point maps and pseudo-3D labels serve complementary roles in PLANET. Through geometric query lifting, dense point maps introduce scene-level geometric evidence into the model’s queries. Through anchored 3D localization, matched queries predict 3D object centers as an auxiliary task and learn object-level 3D information. Geometric query lifting and anchored 3D localization address complementary aspects of the problem and together lead to the learning of our world-grounded queries (Fig. 2).

However, a recurring constraint in transformer-based end-to-end tracking is the limited temporal context available to the identity model. PLANET therefore introduces dual-resolution temporal memory, which pairs complementary training and inference strategies to expand the temporal horizon available to the identity model. Under this formulation, temporally varied training exposes the decoder to observations separated by different intervals. At inference, the memory retains short-term context densely while sampling long-term history more sparsely within a fixed context budget. This design extends the available temporal horizon from 30 frames in MOTIP to approximately 200 frames in PLANET, allowing world-grounded evidence to inform association across substantially longer gaps.

## 3.3. Lifting 2D Tracking Datasets into 3D

Existing tracking benchmarks provide monocular videos with 2D boxes and identities. To lift these datasets into 3D, we apply DA3-Streaming, a long-sequence extension of Depth Anything 3 [17] inspired by VGGT-Long [9]. At a high level, long sequences are processed with overlapping windows to obtain a sequence-level coordinate system, while loop closure limits the drift accumulated over time. The aligned depth, camera intrinsics, and camera poses are then back-projected into dense world-coordinate point maps.

To complement the dense scene geometry with objectlevel labels, we assign each annotated 2D box an estimated 3D location. For each box, we form a candidate annotation by averaging the local point cloud. Candidates from fully visible boxes are accepted directly, while occluded boxes are handled conservatively using temporal depth, currentframe depth, and image-plane position to identify the visible foreground object. We also improve consistency along each identity trajectory using its trajectory history. Estimates that remain uncertain are marked invalid and excluded from 3D supervision. The resulting 3D annotations associate each 2D box with an estimated 3D center location, without specifying its dimensions or orientation. We also obtain camera parameters, including intrinsics and poses, although PLANET does not use them directly. We intend to release the lifted datasets, including all additional annotations and reconstructed scene information, to support future research within and beyond tracking.

![](images/6f3830e3e4a561f669a1982e32720482af8d2dd88a5a6eed61bee3293442fdee.jpg)  
Figure 2. PLANET overview. Geometric query lifting conditions query formation on dense scene geometry, while anchored 3D localization encourages matched queries to encode their 3D locations. Together, they form world-grounded queries. Dual-resolution temporal memory extends the context used for identity prediction.

## 3.4. World-Grounded Queries

PLANET forms world-grounded queries through two complementary mechanisms: geometric query lifting and anchored 3D localization. Geometric query lifting conditions query formation on dense scene geometry, while anchored 3D localization encourages the resulting queries to encode the 3D positions of their matched objects.

## 3.4.1. Geometric Query Lifting

The aligned point map P describes the reconstructed scene geometry. We process P using two separate learnable heads: a geometric feature head $L _ { F }$ and a geometric $p o -$ sition head $L _ { P }$ . These lifting heads produce geometric feature embeddings $\mathbf { F } _ { \mathrm { 3 D } } ~ = ~ L _ { F } ( \mathbf { P } )$ and geometric position embeddings $\mathbf { E } _ { \mathrm { 3 D } } = L _ { P } ( \mathbf { P } )$ , respectively. The geometric feature embeddings are added residually to the visual features $\mathbf { F } _ { \mathrm { 2 D } }$ produced by the backbone, $\widetilde { \bf F } = { \bf F } _ { \mathrm { 2 D } } + { \bf F } _ { \mathrm { 3 D } }$ allowing local 3D structure to complement the appearance representation inherited from the pretrained backbone. In parallel, the geometric position embeddings are combined with the sine-cosine 2D positional encoding $\mathrm { \bf E _ { \mathrm { 2 D } } }$ through a learned fusion head, $\widetilde { \bf E } = \mathcal { M } ( [ { \bf E } _ { \mathrm { 2 D } } \Vert { \bf E } _ { \mathrm { 3 D } } ] )$ ). This augments image-plane position with information about the reconstructed 3D scene. Deformable attention uses the geometryconditioned features $\widetilde { \mathbf { F } }$ and positional embeddings Ee when updating the object queries. The resulting queries therefore encode both visual evidence and estimated 3D scene information. We refer to this process as geometric query lifting because it extends the query representation beyond the image plane while preserving the detector’s existing 2D components.

## 3.4.2. Anchored 3D Localization

In Deformable DETR, each object query is associated with a learned 2D reference point that guides its localization. Bounding-box prediction proceeds through iterative refinements of this reference point across the transformer decoder, resulting in the final 2D localization.

A direct extension to 3D would learn an independent 3D reference point from scratch. Yet the backbone and detector have already acquired strong localization priors through large-scale 2D pretraining. Learning an isolated 3D reference would leave these priors underused and separate 3D localization from the detector’s established spatial representation. We therefore extend the motivation behind geometric query lifting to the reference points themselves by lifting the detector’s existing 2D reference point $\mathbf { r } _ { \mathrm { 2 D } }$ into a corresponding 3D reference point ${ \bf r } _ { \mathrm { 3 D } }$ using the aligned scene geometry. Given ${ \bf r } _ { \mathrm { 3 D } }$ , an additional lightweight 3D prediction head estimates a residual from the corresponding query embedding $\mathbf { q }$ and refines the reference point to predict the object’s 3D center. This anchor-relative formulation builds on the detector’s pretrained 2D localization mechanism instead of requiring absolute 3D positions to be inferred independently. After the detector performs its standard matching, valid 3D labels supervise the corresponding predicted centers through an additional loss $\mathcal { L } _ { \ell _ { 1 } } ^ { \mathrm { 3 D } }$ During training, this term is added to the existing detection and identity objectives.

Table 1. Cumulative contributions of geometric query lifting (GQL), anchored 3D localization (A3DL), and dual-resolution tempora memory (DRTM). GQL and A3DL jointly form world-grounded queries. Components are enabled cumulatively from top to bottom.
<table><tr><td rowspan="2">Variant</td><td colspan="2">World-Grounded Queries</td><td rowspan="2">Memory</td><td colspan="5">Metrics</td></tr><tr><td>GQL</td><td>A3DL</td><td>DRTM HOTA↑</td><td>AssA↑</td><td>DetA ↑</td><td>IDF1↑</td><td>MOTA ↑</td></tr><tr><td>Base</td><td>X</td><td>X</td><td>X</td><td>63.0</td><td>53.2</td><td>75.0</td><td>66.6</td><td>84.7</td></tr><tr><td>+ GQL</td><td>√</td><td>X</td><td>X</td><td>64.0</td><td>54.3</td><td>75.9</td><td>67.6</td><td>85.5</td></tr><tr><td>+ A3DL</td><td>√</td><td>√</td><td>X</td><td>64.4</td><td>54.9</td><td>76.1</td><td>67.8</td><td>85.5</td></tr><tr><td>+ DRTM</td><td>√</td><td>√</td><td>√</td><td>65.6</td><td>56.9</td><td>76.1</td><td>70.3</td><td>85.9</td></tr></table>

Anchored 3D localization complements geometric query lifting: lifting makes dense geometric evidence available during query formation, while localization encourages matched queries to retain object-specific 3D position information.

## 3.5. Dual-Resolution Temporal Memory

Transformer-based end-to-end trackers incur increasing computational cost as their identity models process and attend to a growing history of query representations. Their temporal context is therefore typically restricted, making older identity evidence inaccessible across extended oc clusions. PLANET addresses this limitation with dualresolution temporal memory, which extends the accessible temporal horizon within a fixed context budget. Specifically, we divide the fixed identity-decoder context into complementary short-term and long-term regions. The dense short-term region preserves the latest evolution of the scene, supporting consistent association through brief occlusions, missed detections, and rapid motion. The sparsely sampled long-term region allows the decoder to revisit identity evidence beyond its original consecutive-frame horizon, supporting trajectory continuity across extended disappearances without increasing the number of processed representations. To make the identity model robust to these different temporal densities, we also vary the temporal sampling rate during training. This prepares the decoder to associate evidence across the short-term and long-term gaps selected by dual-resolution memory.

Formally, given a context budget of M observations, we allocate approximately M/2 slots to the most recent consecutive frames and the remaining M/2 slots to earlier frames sampled with stride S. Because the trajectory buffer stores the detector’s final query embeddings, this extended context preserves world-grounded evidence across longer gaps without modifying the detector outputs or association architecture. This extension also does not require additional training time.

## 4. Experiments

## 4.1. Datasets and Metrics

## 4.1.1. Datasets

DanceTrack. DanceTrack contains 100 videos, primarily of group dancing, in which people with similar appearances undergo complex motion, articulation, crossovers, and occlusions [33]. Because appearance is often weakly discriminative, the benchmark places particular emphasis on motion-aware association.

SportsMOT. SportsMOT comprises 240 sequences from basketball, volleyball, and football, totaling more than 150,000 frames and 1.6 million annotated bounding boxes [7]. Its fast, variable-speed player motion makes association challenging, while similar player appearances limit the reliability of visual identity cues.

BFT. The Bird Flock Tracking (BFT) benchmark contains 106 clips spanning 22 bird species and 14 natural and human-made scene types [50]. It targets highly dynamic open-world tracking, with rapid motion, deformation, frequent occlusion, and visually similar objects in dense flocks.

## 4.1.2. Metrics

We report Higher Order Tracking Accuracy (HOTA), Identification F1 (IDF1), and Multiple Object Tracking Accuracy (MOTA), with higher values indicating better performance.

HOTA [20]. At each localization threshold, HOTA computes the geometric mean of detection accuracy (DetA) and association accuracy (AssA), and the final score averages over thresholds. Its explicit combination of detection and association provides a balanced summary of these two aspects of tracking performance.

IDF1 [27]. IDF1 is the harmonic mean of identification precision and identification recall after a global one-to-one assignment between predicted and ground-truth identities. It therefore emphasizes identity preservation over the sequence and is particularly informative about association consistency.

MOTA [2]. MOTA combines false negatives, false positives, and identity mismatches into an error count normalized by the number of ground-truth detections. Because false positives and missed detections contribute directly to this count, MOTA tends to emphasize detection coverage more than long-term identity preservation.

## 4.2. Implementation Details

To make our comparisons with prior tracking-by-query methods as direct as possible, we use the same established detector family [11, 12, 28, 45]: Deformable DETR [53] with a ResNet-50 backbone [13]. Following the initialization practice of prior works [11, 42, 45], we initialize the detector from COCO-pretrained weights. Together, these choices keep our experimental setting aligned with prior work.

We follow the training setup of MOTIP [12] and train with AdamW [19] using an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of $5 \times 1 0 ^ { - 4 }$ . Training also includes random resizing, cropping, horizontal flipping, trajectory occlusion, and identity switching. We train each ablation configuration five times and report the mean result to reduce sensitivity to run-to-run variation.

The added geometric components are lightweight, and PLANET retains a runtime comparable to MOTIP, processing video at 12 FPS in FP32 and 20 FPS in FP16.

## 4.3. Ablations

## 4.3.1. Main Components

Impact of World-Grounded Queries and Temporal Memory. Table 1 evaluates the cumulative contribution of geometric query lifting, anchored 3D localization, and dualresolution temporal memory, with the first two components jointly forming world-grounded queries. Geometric query lifting improves every reported metric over the base model, including a 1.0-point gain in HOTA. The comparable gains in association and detection accuracy indicate that incorporating estimated 3D position information during query formation benefits both aspects of tracking.

Adding anchored 3D localization on top of geometric query lifting yields a further 0.4-point improvement in HOTA, with a larger gain in association accuracy than in detection accuracy. This result shows that training matched queries to localize objects in 3D through anchored refinement helps the end-to-end tracker resolve association ambiguities caused by occlusion, rather than merely improving detection.

Finally, dual-resolution temporal memory improves HOTA by another 1.2 points and produces clear association gains while detection accuracy remains unchanged. These gains highlight the importance of extending the model context from short-term observations to long-term temporal information, with stable 3D spatial cues supporting correspondences over longer time horizons.

Table 2. Interaction between 3D awareness and dual-resolution temporal memory (DRTM). Each cell reports performance with basic memory, with DRTM, and the corresponding change in parentheses. Highlighted values indicate the larger DRTM gain for each metric.
<table><tr><td rowspan="2">Model</td><td colspan="2">Basic Memory → DRTM (Gain)</td></tr><tr><td>HOTA ↑ IDF1↑</td><td>MOTA ↑</td></tr><tr><td>2D-aware</td><td>63.0 → 63.8 66.6 → 68.1 84.7 → 84.7 (+0.8) (+1.5)</td><td>(+0.0)</td></tr><tr><td>3D-aware</td><td>+1.2 +2.5)</td><td>64.4 → 65.6 67.8 → 70.3 85.5 → 85.9 +0.4)</td></tr></table>

Table 3. Direct versus anchored 3D localization. D3DL denotes direct 3D localization, while A3DL denotes anchored 3D localization.
<table><tr><td colspan="4">Model Variant HOTA ↑ AssA ↑ DetA ↑ IDF1 ↑ MOTA ↑</td></tr><tr><td>Base</td><td>64.0 54.3</td><td>75.9 67.6</td><td>85.5</td></tr><tr><td>D3DL (Naive)</td><td>63.8 54.2</td><td>75.7 67.4</td><td>85.4</td></tr><tr><td>A3DL (Ours)</td><td>64.4 54.9</td><td>76.1 67.8</td><td>85.5</td></tr></table>

Interaction Between 3D Awareness and Temporal Memory. Having established the importance of temporal memory, we next ask how its effectiveness changes when the tracker has access to 3D information. Table 2 addresses this question by measuring the gain from adding dual-resolution temporal memory to two otherwise corresponding models: one that reasons only from 2D information and one equipped with world-grounded queries. For the 2D-aware model, dual-resolution temporal memory improves HOTA by 0.8 points and IDF1 by 1.5 points, while MOTA remains unchanged. This result confirms that extending the temporal horizon is beneficial even without explicit 3D information. For the 3D-aware model, the same memory mechanism produces larger gains across all reported metrics, improving HOTA by 1.2 points, IDF1 by 2.5 points, and MOTA by 0.4 points. Thus, temporal memory benefits both models, but its stronger effect in the 3D-aware setting further highlights the synergy between 3D information and extended temporal context. Overall, world-grounded representations provide stable spatial evidence, while dualresolution temporal memory makes this evidence available for identity reasoning across longer gaps.

## 4.3.2. How to Embed 3D Cues into a 2D Model?

Direct vs. Anchored 3D Localization. Table 3 compares two models trained with the same 3D supervision but different 3D localization paradigms. Direct 3D Localization (Naive) adds a 3D prediction head in parallel with the existing 2D head and learns an independent reference point in

3D space. Anchored 3D Localization (Ours) instead lifts the detector’s existing 2D reference point into 3D and predicts a refinement relative to this anchor. The direct version reduces HOTA from 64.0 to 63.8 and produces small declines across all reported metrics. In contrast, our anchored version outperforms the direct version across all metrics, improving HOTA by 0.6 points and exceeding the baseline by 0.4 points. By building on the pretrained 2D localization capability, our version avoids learning 3D localization entirely from scratch. This coupling allows 2D and 3D localization to shape a shared representation, making the additional 3D task easier to learn while retaining useful 2D information. These results show that although 3D supervision provides valuable geometric information, its benefit depends critically on how it is integrated into the model, supporting the refinement design used in our anchored 3D localization.

3D Foundation Features vs. World-Grounded Queries. A straightforward way to introduce 3D cues into a 2D tracker is to inject features extracted by a pretrained 3D foundation model. We use this strategy as a baseline to our world-grounded queries by extracting 3D Foundation Features from Depth Anything 3 [17] and fusing them into the tracker during training. As shown in Tab. 4, these features improve HOTA only marginally, from 63.0 to 63.1, while leaving association accuracy and IDF1 unchanged or slightly reduced. Although the injected features contain strong 3D cues, they are not assigned an explicit role in query formation or the tracking objective. Moreover, the foundation model is trained for geometry estimation rather than tracking, so its features must be adapted to the tracking domain and objectives before the model can fully capitalize on them. The limited gains suggest that tracking supervision through generic feature fusion alone does not provide a sufficiently effective interface for this adaptation.

World-grounded queries instead introduce 3D information through geometric query lifting and anchored 3D local ization, allowing dense scene geometry to condition query formation while explicitly encouraging matched queries to encode object-level 3D locations. This structured formulation improves HOTA by 1.4 points over the baseline and yields gains in both association and detection accuracy. Overall, these results show that incorporating pretrained 3D cues into a 2D tracker is not straightforward and requires design decisions that align geometric information with the model’s query representations and objectives.

## 4.4. Benchmark Results

Following the evaluation protocol of MOTIP [12], we evaluate PLANET on the same three benchmarks: DanceTrack, SportsMOT, and BFT. We also adopt MOTIP’s datasetspecific training scheme, training a model exclusively on each corresponding benchmark without pretraining or joint training on related tracking or detection datasets.

Table 4. 3D foundation features versus the world-grounded query formulation. The latter combines geometric query lifting with an chored 3D localization.
<table><tr><td>3D Cues</td><td>HOTA ↑ AssA ↑ DetA ↑</td><td></td><td></td><td>IDF1↑ MOTA↑</td><td></td></tr><tr><td>Base</td><td>63.0</td><td>53.2</td><td>75.0</td><td>66.6</td><td>84.7</td></tr><tr><td>3D Foundation Features</td><td>63.1</td><td>53.1</td><td>75.3</td><td>66.6</td><td>84.9</td></tr><tr><td>WGQ (Ours)</td><td>64.4</td><td>54.9</td><td>76.1</td><td>67.8</td><td>85.5</td></tr></table>

Table 5. Benchmark results on the DanceTrack test set.
<table><tr><td colspan="6">DanceTrack</td></tr><tr><td>Method</td><td>HOTA ↑ AssA ↑1</td><td></td><td>DetA ↑</td><td>IDF1↑</td><td>MOTA↑</td></tr><tr><td colspan="6">Tracking-by-Detection</td></tr><tr><td>FairMOT [47]</td><td>39.7</td><td>23.8</td><td>66.7</td><td>40.8</td><td>82.2</td></tr><tr><td>CenterTrack [51]</td><td>41.8</td><td>22.6</td><td>78.1</td><td>35.7</td><td>86.8</td></tr><tr><td>TraDeS [40]</td><td>43.3</td><td>25.4</td><td>74.5</td><td>41.2</td><td>86.2</td></tr><tr><td>DeepSORT [39]</td><td>45.6</td><td>29.7</td><td>71.0</td><td>47.9</td><td>87.8</td></tr><tr><td>ByteTrack [48]</td><td>47.7</td><td>32.1</td><td>71.0</td><td>53.9</td><td>89.6</td></tr><tr><td>SORT [3]</td><td>47.9</td><td>31.2</td><td>72.0</td><td>50.8</td><td>91.8</td></tr><tr><td>GTR [52]</td><td>48.0</td><td>31.9</td><td>72.5</td><td>50.3</td><td>84.7</td></tr><tr><td>QDTrack [25]</td><td>54.2</td><td>36.8</td><td>80.1</td><td>50.4</td><td>87.7</td></tr><tr><td>OC-SORT [5]</td><td>55.1</td><td>38.3</td><td>80.3</td><td>54.6</td><td>92.0</td></tr><tr><td>StrongSORT [10]</td><td>55.6</td><td>38.6</td><td>80.7</td><td>55.2</td><td>91.1</td></tr><tr><td>SparseTrack [18]</td><td>55.7</td><td>39.3</td><td>79.2</td><td>58.1</td><td>91.3</td></tr><tr><td>C-BIoU [43]</td><td>60.6</td><td>45.4</td><td>81.3</td><td>61.6</td><td>91.6</td></tr><tr><td>Hybrid-SORT [44]</td><td>62.2</td><td>一</td><td></td><td>63.0</td><td>91.6</td></tr><tr><td>DiffMOT [21]</td><td>62.3</td><td>47.2</td><td>82.5</td><td>63.0</td><td>92.8</td></tr><tr><td>TrackTrack [31]</td><td>66.5</td><td>52.9</td><td></td><td>67.8</td><td>93.6</td></tr><tr><td colspan="6">End-to-End</td></tr><tr><td>TransTrack [32]</td><td>45.5</td><td>27.5</td><td>75.9</td><td>45.2</td><td>88.4</td></tr><tr><td>MOTR [46]</td><td>54.2</td><td>40.2</td><td>73.5</td><td>51.5</td><td>79.7</td></tr><tr><td>MeMOTR [11]</td><td>63.4</td><td>52.3</td><td>77.0</td><td>65.5</td><td>85.4</td></tr><tr><td>CO-MOT [42]</td><td>65.3</td><td>53.5</td><td>80.1</td><td>66.5</td><td>89.3</td></tr><tr><td>LA-MOTR [36]</td><td>66.5</td><td>53.9</td><td>80.4</td><td>69.7</td><td>92.5</td></tr><tr><td>SambaMOTR [28]</td><td>67.2</td><td>57.5</td><td>78.8</td><td>70.5</td><td>88.1</td></tr><tr><td>MOTIP [12]</td><td>69.6</td><td>60.4</td><td>80.4</td><td>74.7</td><td>90.6</td></tr><tr><td>PLANET 0 (Ours)</td><td>72.1</td><td>63.9</td><td>81.4</td><td>78.0</td><td>91.6</td></tr></table>

Many multi-object tracking methods instead pretrain on additional datasets or augment benchmark training data with external datasets [5, 21, 42, 45, 48, 49]. A common choice is CrowdHuman [29], whose static images can be transformed into pseudo-video sequences. MOTIP argues that this practice can produce models that generalize poorly across tracking settings and complicate fair comparison because methods may benefit from different sources and amounts of additional supervision [12]. For further discussion of this evaluation choice, we refer to the MOTIP paper [12].

Consequently, our reported results do not benefit from external training data, whereas several competing methods are trained with such additional supervision. Despite this more restrictive protocol, PLANET reaches state-of-theart performance on all three benchmarks and outperforms methods both with and without external training data.

DanceTrack. DanceTrack evaluates association under complex motion, frequent crossovers and occlusions, and limited appearance diversity among the tracked dancers. In Table 5, PLANET achieves 72.1 HOTA, exceeding MOTIP by 2.5 points, and raises IDF1 from 74.7 to 78.0 while improving every reported metric over this strongest end-toend baseline. The gains across detection and association demonstrate the effectiveness of the complete model under ambiguous appearance and motion patterns.

SportsMOT. SportsMOT tests tracking under fast, variable player motion and similar appearances across diverse sports scenes. In Table 6, PLANET achieves state-of-the-art performance across the reported metrics, reaching 73.7 HOTA, 1.1 points above MOTIP, while raising DetA from 83.5 to 86.0 and MOTA from 92.4 to 95.2. These results show that the method remains effective in large-scale sequences with rapid and irregular motion.

BFT. BFT evaluates the method in dense bird flocks characterized by rapid motion, deformation, frequent occlusion, and visually similar individuals. In Table 7, PLANET achieves state-of-the-art performance across all five metrics, including 72.6 HOTA, exceeding the previous state of the art by 2.1 points. The gains across detection and association demonstrate the effectiveness of world-grounded queries and dual-resolution temporal memory in highly dynamic, densely interacting scenes.

## 5. Conclusion

We have introduced PLANET, an end-to-end multi-object tracking framework that moves beyond image-plane reasoning by grounding object queries in reconstructed 3D scene geometry and retaining this information across longer temporal horizons. Through the careful design of geometric query lifting and anchored 3D localization, PLANET incorporates 3D information directly into a 2D tracker. Our ablation studies validate the contribution of each component and show that world-grounded queries and dualresolution temporal memory provide complementary gains. Our benchmark results demonstrate state-of-the-art performance across diverse tracking scenarios, supporting the generality of the approach. We believe this work represents a step toward using 3D knowledge to improve tasks traditionally formulated in 2D. We hope PLANET encourages future research to look beyond image-plane evidence and pursue richer forms of 3D reasoning as a central direction for addressing the ambiguities that continue to limit tracking and visual understanding.

Table 6. Benchmark results on the SportsMOT test set.
<table><tr><td colspan="5">SportsMOT</td></tr><tr><td colspan="5">Method HOTA ↑ AssA ↑ DetA↑ IDF1 ↑ MOTA ↑</td></tr><tr><td colspan="5">Tracking-by-Detection</td></tr><tr><td>FairMOT [47] GTR [52]</td><td>49.3 54.5</td><td>34.7 45.9 47.2</td><td>70.2 53.5 64.8 55.8</td><td>86.4 67.9</td></tr><tr><td>QDTrack [25]</td><td>60.4</td><td>77.5</td><td>62.3</td><td>90.1</td></tr><tr><td>CenterTrack [51]</td><td>62.7 48.0</td><td>82.1</td><td>60.0</td><td>90.8</td></tr><tr><td>ByteTrack [48]</td><td>62.8 51.2</td><td>77.1</td><td>69.8</td><td>94.1</td></tr><tr><td>OC-SORT [5]</td><td>68.1 54.8</td><td>84.8</td><td>68.0</td><td>93.4</td></tr><tr><td>BoT-SORT [1]</td><td>68.7 55.9</td><td>84.4</td><td>70.0</td><td>94.5</td></tr><tr><td>DiffMOT [21] 72.1 End-to-End</td><td>60.5</td><td>86.0</td><td>72.8</td><td>94.5</td></tr><tr><td colspan="5">MeMOTR [11]</td></tr><tr><td>TransTrack [32] LA-MOTR [36]</td><td>68.8 57.8 68.9 57.5 69.5 57.9</td><td>82.0 82.7 84.4 82.2</td><td>69.9 71.5 70.0</td><td>90.2 92.6 93.5</td></tr><tr><td>SambaMOTR [28] MOTIP [12]</td><td>69.8 72.6</td><td>59.4 63.2 83.5</td><td>71.9 77.1</td><td>90.3 92.4</td></tr><tr><td>PLANET 0 (Ours)</td><td>73.7</td><td>63.2 86.0</td><td>78.7</td><td>95.2</td></tr></table>

Table 7. Benchmark results on the BFT test set.
<table><tr><td colspan="5">BFT</td></tr><tr><td>Method</td><td colspan="4">HOTA ↑ AssA↑ DetA ↑ IDF1 ↑ MOTA ↑</td></tr><tr><td colspan="5">Tracking-by-Detection</td></tr><tr><td>JDE [37] CSTrack [16]</td><td>30.7</td><td>23.4 40.9 47.0</td><td>37.4 34.5</td><td>35.4 46.7</td></tr><tr><td>FairMOT [47]</td><td>33.2 40.2</td><td>23.7 28.2 62.3</td><td>53.3 41.8</td><td>56.0</td></tr><tr><td>SORT [3]</td><td>61.2</td><td>60.6 64.1</td><td>77.2</td><td>75.5</td></tr><tr><td>ByteTrack [48]</td><td>62.5</td><td>61.2</td><td>82.3</td><td>77.2</td></tr><tr><td>CenterTrack [51]</td><td>65.0 54.0</td><td>58.5</td><td>61.0</td><td>60.2</td></tr><tr><td>OC-SORT [5]</td><td>66.8 68.7</td><td>65.4</td><td>79.3</td><td>77.1</td></tr><tr><td colspan="5">End-to-End</td></tr><tr><td>TransCenter [41]</td><td>60.0</td><td>61.1</td><td>66.0 72.4 64.2</td><td>74.1</td></tr><tr><td>TransTrack [32] TrackFormer [23]</td><td>62.1 63.3</td><td>60.3 61.1 66.0</td><td>71.4 72.4</td><td>71.4 74.1</td></tr><tr><td>SambaMOTR [28]</td><td>69.6</td><td>73.6</td><td>66.0 81.9</td><td>72.0</td></tr><tr><td>MOTIP [12]</td><td>70.5</td><td>71.8 69.6</td><td>82.1</td><td>77.1</td></tr><tr><td>PLANET 0 (Ours)</td><td>72.6</td><td>73.9</td><td>71.6 84.6</td><td>80.4</td></tr></table>

## References

[1] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. BoT-SORT: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.

[2] Keni Bernardin and Rainer Stiefelhagen. Evaluating multiple object tracking performance: The CLEAR MOT metrics. EURASIP Journal on Image and Video Processing, 2008:1– 10, 2008.

[3] Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft. Simple online and realtime tracking. In ICIP, pages 3464–3468, 2016.

[4] Guillem Braso and Laura Leal-Taix´ e. Learning a neural´ solver for multiple object tracking. In CVPR, pages 6247– 6257, 2020.

[5] Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric SORT: Rethinking SORT for robust multi-object tracking. In CVPR, pages 9686–9696, 2023.

[6] Orcun Cetintas, Guillem Braso, and Laura Leal-Taix ´ e. Uni-´ fying short and long-term tracking with graph hierarchies. In CVPR, pages 22877–22887, 2023.

[7] Yutao Cui, Chenkai Zeng, Xiaoyu Zhao, Yichun Yang, Gangshan Wu, and Limin Wang. SportsMOT: A large multiobject tracking dataset in multiple sports scenes. In ICCV, pages 9921–9931, 2023.

[8] Patrick Dendorfer, Vladimir Yugay, Aljosa O ˇ sep, and Lauraˇ Leal-Taixe. Quo vadis: Is trajectory forecasting the key to-´ wards long-term multi-object tracking? In NeurIPS, pages 15657–15671, 2022.

[9] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. VGGT-Long: Chunk it, loop it, align it–pushing VGGT’s limits on kilometer-scale long RGB sequences. arXiv preprint arXiv:2507.16443, 2025.

[10] Yunhao Du, Zhicheng Zhao, Yang Song, Yanyun Zhao, Fei Su, Tao Gong, and Hongying Meng. StrongSORT: Make DeepSORT great again. IEEE TMM, 25:8725–8737, 2023.

[11] Ruopeng Gao and Limin Wang. MeMOTR: Long-term memory-augmented transformer for multi-object tracking. In ICCV, pages 9901–9910, 2023.

[12] Ruopeng Gao, Ji Qi, and Limin Wang. Multiple object tracking as ID prediction. In CVPR, pages 27883–27893, 2025.

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.

[14] Hou-Ning Hu, Qi-Zhi Cai, Dequan Wang, Ji Lin, Min Sun, Philipp Krahenb¨ uhl, Trevor Darrell, and Fisher Yu. Joint¨ monocular 3D vehicle detection and tracking. In ICCV, pages 5390–5399, 2019.

[15] Tarasha Khurana, Achal Dave, and Deva Ramanan. Detecting invisible people. In ICCV, pages 3174–3184, 2021.

[16] Chao Liang, Zhipeng Zhang, Xue Zhou, Bing Li, Shuyuan Zhu, and Weiming Hu. Rethinking the competition between detection and ReID in multi-object tracking. IEEE TIP, 31: 3182–3196, 2022.

[17] Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Yang Zhao, Sida Peng, Hengkai Guo, Xiaowei

Zhou, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the visual space from any views. In ICLR, 2026.

[18] Zelin Liu, Xinggang Wang, Cheng Wang, Wenyu Liu, and Xiang Bai. SparseTrack: Multi-object tracking by per forming scene decomposition based on pseudo-depth. IEEE TCSVT, 35(5):4870–4882, 2025.

[19] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

[20] Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip H. S. Torr, Andreas Geiger, Laura Leal-Taixe, and Bastian´ Leibe. HOTA: A higher order metric for evaluating multiobject tracking. IJCV, 129(2):548–578, 2021.

[21] Weiyi Lv, Yuhang Huang, Ning Zhang, Ruei-Sung Lin, Mei Han, and Dan Zeng. DiffMOT: A real-time diffusion-based multiple object tracker with non-linear prediction. In CVPR, pages 19321–19330, 2024.

[22] Gianluca Mancusi, Aniello Panariello, Angelo Porrello, Matteo Fabbri, Simone Calderara, and Rita Cucchiara. TrackFlow: Multi-object tracking with normalizing flows. In ICCV, pages 9531–9543, 2023.

[23] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and´ Christoph Feichtenhofer. TrackFormer: Multi-object track ing with transformers. In CVPR, pages 8844–8854, 2022.

[24] Aljosa Oˇ sep, Wolfgang Mehner, Markus Mathias, and Bas-ˇ tian Leibe. Combined image- and world-space tracking in traffic scenes. In ICRA, pages 1988–1995, 2017.

[25] Jiangmiao Pang, Linlu Qiu, Xia Li, Haofeng Chen, Qi Li, Trevor Darrell, and Fisher Yu. Quasi-dense similarity learning for multiple object tracking. In CVPR, pages 164–173, 2021.

[26] Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, and Jitendra Malik. Tracking people by predicting 3D appearance, location and pose. In CVPR, pages 2740–2749, 2022.

[27] Ergys Ristani, Francesco Solera, Roger S. Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In Computer Vi sion – ECCV 2016 Workshops, pages 17–35, 2016.

[28] Mattia Segu, Luigi Piccinelli, Siyuan Li, Yung-Hsu Yang, Luc Van Gool, and Bernt Schiele. Samba: Synchronized set-of-sequences modeling for multiple object tracking. In ICLR, 2025.

[29] Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, and Jian Sun. CrowdHuman: A benchmark for detecting human in a crowd. arXiv preprint arXiv:1805.00123, 2018.

[30] Sarthak Sharma, Junaid Ahmed Ansari, J. Krishna Murthy, and K. Madhava Krishna. Beyond pixels: Leveraging geometry and shape cues for online multi-object tracking. In ICRA, pages 3508–3515, 2018.

[31] Kyujin Shim, Kangwook Ko, Yujin Yang, and Changick Kim. Focusing on tracks for online multi-object tracking. In CVPR, pages 11687–11696, 2025.

[32] Peize Sun, Jinkun Cao, Yi Jiang, Rufeng Zhang, Enze Xie, Zehuan Yuan, Changhu Wang, and Ping Luo. TransTrack: Multiple object tracking with transformer. arXiv preprint arXiv:2012.15460, 2020.

[33] Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. DanceTrack: Multi-object tracking in uniform appearance and diverse motion. In CVPR, pages 20993–21002, 2022.

[34] Pavel Tokmakov, Jie Li, Wolfram Burgard, and Adrien Gaidon. Learning to track with object permanence. In ICCV, pages 10860–10869, 2021.

[35] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In CVPR, pages 5294–5306, 2025.

[36] Peng Wang, Yongcai Wang, Hualong Cao, Wang Chen, and Deying Li. LA-MOTR: End-to-end multi-object tracking by learnable association. In ICCV, pages 12438–12448, 2025.

[37] Zhongdao Wang, Liang Zheng, Yixuan Liu, Yali Li, and Shengjin Wang. Towards real-time multi-object tracking. In ECCV, pages 107–122, 2020.

[38] Christian Wojek, Stefan Walk, Stefan Roth, and Bernt Schiele. Monocular 3D scene understanding with explicit occlusion reasoning. In CVPR, pages 1993–2000, 2011.

[39] Nicolai Wojke, Alex Bewley, and Dietrich Paulus. Simple online and realtime tracking with a deep association metric. In ICIP, pages 3645–3649, 2017.

[40] Jialian Wu, Jiale Cao, Liangchen Song, Yu Wang, Ming Yang, and Junsong Yuan. TraDeS: Track to detect and segment for multi-object tracking. In CVPR, pages 12352– 12361, 2021.

[41] Yihong Xu, Yutong Ban, Guillaume Delorme, Chuang Gan, Daniela Rus, and Xavier Alameda-Pineda. TransCenter: Transformers with dense representations for multiple-object tracking. arXiv preprint arXiv:2103.15145, 2021.

[42] Feng Yan, Weixin Luo, Yujie Zhong, Yiyang Gan, and Lin Ma. Bridging the gap between end-to-end and non-end-toend multi-object tracking. In ICLR, 2025.

[43] Fan Yang, Shigeyuki Odashima, Shoichi Masui, and Shan Jiang. Hard to track objects with irregular motions and similar appearances? make it easier by buffering the matching space. In WACV, pages 4799–4808, 2023.

[44] Mingzhan Yang, Guangxin Han, Bin Yan, Wenhua Zhang, Jinqing Qi, Huchuan Lu, and Dong Wang. Hybrid-SORT: Weak cues matter for online multi-object tracking. In AAAI, pages 6504–6512, 2024.

[45] En Yu, Tiancai Wang, Zhuoling Li, Yuang Zhang, Xiangyu Zhang, and Wenbing Tao. MOTRv3: Release-fetch supervision for end-to-end multi-object tracking. arXiv preprint arXiv:2305.14298, 2023.

[46] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. MOTR: End-to-end multipleobject tracking with transformer. In ECCV, pages 659–675, 2022.

[47] Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. FairMOT: On the fairness of detection and re-identification in multiple object tracking. IJCV, 129(11): 3069–3087, 2021.

[48] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. ByteTrack: Multi-object tracking by associating every detection box. In ECCV, pages 1–18, 2022.

[49] Yuang Zhang, Tiancai Wang, and Xiangyu Zhang. MOTRv2: Bootstrapping end-to-end multi-object tracking by pretrained object detectors. In CVPR, pages 22056–22065, 2023.

[50] Guangze Zheng, Shijie Lin, Haobo Zuo, Changhong Fu, and Jia Pan. NetTrack: Tracking highly dynamic objects with a net. In CVPR, pages 19145–19155, 2024.

[51] Xingyi Zhou, Vladlen Koltun, and Philipp Krahenb¨ uhl.¨ Tracking objects as points. In ECCV, pages 474–490, 2020.

[52] Xingyi Zhou, Tianwei Yin, Vladlen Koltun, and Philipp Krahenb¨ uhl. Global tracking transformers. In¨ CVPR, pages 8771–8780, 2022.

[53] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable transform ers for end-to-end object detection. In ICLR, 2021.