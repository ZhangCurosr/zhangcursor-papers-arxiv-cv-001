# A Top-Down Framework for Metric-Scale Athlete Localization from Single Broadcast Frames

Thanh-Khoi Nguyen<sup>a,b</sup>, Hoang-Phuc Nguyen<sup>a,b</sup>, Linh-Huynh<sup>a,b</sup>, and Minh-Triet Tran <sup>∗a,b</sup>

<sup>a</sup>University of Science, VNU-HCM, Ho Chi Minh City, Vietnam <sup>b</sup>Vietnam National University, Ho Chi Minh City, Vietnam

## ABSTRACT

Accurate world-coordinate localization of athletes from single-frame broadcast footage is inherently challenging due to extreme scale disparities in ultra-high-resolution imagery. In this paper, we propose a top-down framework for metric-scale athlete localization from a single calibrated frame. Our approach centers on three key contributions. First, we propose Boundary-Aware Adaptive Tiling, a semantics-guided extension of standard sliced inference. By iteratively expanding tile boundaries based on coarse bounding-box predictions, it systematically ensures full object containment, efectively mitigating boundary-splitting artifacts through a lightweight pipeline adaptation without architectural modifications. By substantially mitigating recall degradation under extreme scale variance, Boundary-Aware Adaptive Tiling enables us to isolate perspective distortion as the primary source of residual localization error. Second, we adapt the RTMPose-X architecture into a specialized two-keypoint estimator (pelvis and ground projection), employing a reformulated Gated Attention Unit optimized for this geometrically coupled point pair, and then deterministically lift the 2D ground projections into world coordinates via camera-calibrated ray casting. On the public test set, our method achieves a LocSim score of 97.44 and an mAP of 0.9128, outperforming the baseline by over 21 % and establishing a robust solution for high-resolution scale variance.

## 1. INTRODUCTION

Understanding the physical positions of athletes on a soccer pitch from camera imagery is a fundamental requirement for modern sports analytics [1, 2]. Applications such as heatmap generation, ball-possession analysis, tactical understanding, and Game State Recognition (GSR) all rely on accurate metric-scale localization of every athlete visible in a scene [2, 3]. Motivated by this need, the recently introduced Spiideo SoccerNet SynLoc dataset [1, 4, 5] formalizes the problem as a single-frame world-coordinate athlete detection and localization task. In this setting, a model receives a 4K image captured by a static, calibrated camera covering half of a soccer pitch and is required to detect all athletes while estimating the 3D world coordinates (in meters) corresponding to the orthogonal projection of each athlete’s pelvis onto the ground plane. Importantly, the evaluation metric, mAP-LocSim, is defined entirely in world space, emphasizing the importance of geometrically consistent localization rather than purely image-level representations [4].

Building upon this formulation, the primary objective of our work is to develop a robust top-down framework for metric-scale athlete localization from a single calibrated broadcast frame. The proposed system is designed to detect athletes across a wide range of apparent scales and accurately estimate the 3D world coordinates (in meters) of their orthogonal projections onto the pitch plane. To ensure a standardized and challenging evaluation protocol, we adopt the Spiideo SoccerNet SynLoc dataset as the benchmark for training and assessment of the proposed approach.

Two dificulties make this task non-trivial. First, detection at scale: because a single camera covers half of a full-sized pitch, athletes near the far touchline occupy only a handful of pixels even in 4K imagery. Ofthe-shelf person detectors trained on standard benchmarks fail to reliably localize such small targets when the full image is downsampled to a manageable inference resolution. Second, localization under perspective distortion: mapping a 2D image observation to a metric pitch coordinate requires accurately identifying a specific anatomical landmark, the pelvis, rather than a coarse bounding-box surrogate such as the bottom-edge center, which the organizer baseline has shown to be a poor approximation of the true physical location [4]. Standard multi-person pose estimators are ill-equipped for this strict geometric requirement. By dedicating capacity to modeling full-body skeletal dependencies, they over-parameterize the task, wasting computational resources on anatomically irrelevant joints. Consequently, they fail to exploit the camera’s geometric properties, treating joints as independent visual targets rather than as points physically coupled by 3D projection.

We propose a two-stage top-down pipeline that addresses each bottleneck in turn. Rather than processing the full high-resolution frame at a reduced resolution, we first run a lightweight coarse detector to obtain approximate object locations, then use these as semantic constraints to guide tile boundary placement. This ensures every object is wholly enclosed within at least one crop at a consistent apparent scale, regardless of its distance from the camera. In the second stage, we observe that the two quantities required for world-coordinate localization - an anatomical reference point and its projection onto the ground plane - are not independent 2D targets but are physically coupled by the camera’s perspective geometry: their relative displacement in the image encodes the subject’s physical height under foreshortening. By strictly adapting the architecture to this physically coupled point pair, we propose a two-keypoint formulation that implicitly captures this geometric coupling, without auxiliary supervision or explicit camera-parameter input. Finally, the predicted 2D ground-projection keypoint is directly lifted to world coordinates via deterministic ray casting against the per-image camera calibration.

Extensive evaluations on the Spiideo SoccerNet SynLoc benchmark demonstrate the robustness of our approach. Our analysis reveals that our boundary-aware adaptive tiling detection strategy achieves the best performance across all metrics, isolating pose estimation as the primary challenge for metric-scale localization. On the private test set, this configuration further achieves a LocSim score of 97.67, significantly outperforming the organizer baseline of 77.3. Furthermore, an ablation study in which we substitute predicted detections with ground-truth bounding boxes shows only a ∼1.4% gap in LocSim, confirming that pose estimation, not detection, is the primary bottleneck for further improvement.

The main contributions of this work are as follows:

• A Geometrically Coupled Localization framework that reformulates pose estimation into a specialized two-keypoint task (pelvis and ground projection). This allows the model to implicitly internalize perspective distortion and camera-projection physics within a 2D attention mechanism, thereby avoiding the over-parameterization of full-body pose estimators.

• Boundary-Aware Adaptive Tiling (BAAT), a novel, semantics-guided cropping strategy. By using a coarse detection pass to iteratively expand tile boundaries, BAAT guarantees full object containment and eliminates boundary-splitting artifacts common in standard regular-grid sliced inference, without requiring additional annotations.

## 2. RELATED WORK

## 2.1 Small-Object Detection and Metric-Scale Athlete Localization

Single-frame metric-scale athlete localization in high-resolution broadcast footage is inherently challenged by extreme scale variance and perspective distortion, as formalized by the SoccerNet SynLoc benchmark [3, 4]. Recovering these distant players is a specific instance of the broader small-object detection problem, canonically evaluated on datasets like VisDrone [6]. While tiled inference methods such as Slicing Aided Hyper Inference (SAHI) improve small-object recall, their regular-grid approach introduces boundary-splitting artifacts, and existing adaptive variants often rely on geometric heuristics rather than semantic cues [7]. Although recent methods like ESOD [8] improve computational eficiency via feature-level slicing, they do not resolve boundary severing. Complementary to these eforts, our proposed Boundary-Aware Adaptive Tiling (BAAT) explicitly eliminates boundary-splitting by repurposing coarse semantic detections to iteratively expand tile boundaries, structurally guaranteeing full object containment.

## 2.2 Geometry-Aware Pose Estimation Architectures

Multi-person pose estimation is typically addressed via top-down or bottom-up paradigms. While bottomup methods ofer favorable scalability, top-down approaches decouple detection and keypoint estimation, enabling the modular integration of specialized components for higher accuracy. Recent top-down architectures, such as ViTPose [9] and the RTMPose family [10], have substantially advanced state-of-the-art performance using transformer-based backbones and the SimCC one-dimensional coordinate classification representation [10,11]. However, these models are designed for full-body skeletons (e.g., 17 COCO joints), fundamentally over-parameterizing the SynLoc task, which evaluates performance exclusively in world space and requires only two geometrically coupled keypoints: the pelvis and its ground projection [4].

While the SynLoc baseline adopts this reduced two-keypoint representation, its reliance on a bottom-up formulation degrades both detection recall and projection accuracy compared to alternative paradigms. To address this, we adopt a top-down framework integrating a state-of-the-art YOLO26 detector [12] with an adapted RTMPose-X model [10]. By contracting the output space explicitly to the two-point formulation, our approach enables the network’s Gated Attention Unit to focus exclusively on the physically coupled pelvis– ground-projection pair. This strictly constrained architecture allows the model to implicitly capture perspective foreshortening without requiring auxiliary geometric supervision.

## 2.3 Sports Player Localization and Camera Calibration

Localizing athletes in world coordinates requires tightly integrating visual detection with camera geometry. Prior works have addressed this within the SoccerNet framework by developing teacher-student calibration pipelines [1], optimizing camera parameter estimation via keypoint and field-line correspondences [13], and formalizing holistic Game State Reconstruction (GSR) for video sequences [14]. However, these approaches predominantly rely on temporal context or continuous tracking across frames. In contrast, our framework targets the strictly harder single-frame setting defined by the Spiideo SoccerNet SynLoc benchmark [4]. By operating without temporal priors, our method must rely entirely on intra-frame geometric reasoning to achieve accurate metricscale localization.

## 3. METHOD

We formulate the task as a two-stage top-down pipeline: (1) high-resolution player detection via adaptive tiling, (2) two-point keypoint estimation using a modified RTMPose-X model. The final 2D predictions are subsequently projected into world coordinates via deterministic ray casting. Our pipeline is demonstrated in Figure 1

![](images/ab35c6636e96ccd7bf184f2a97e26014345f2b4b6572a6c61d0d03e3e4d4a21d.jpg)  
Figure 1. Overview of the proposed pipeline: boundary-aware tiled detection, two-keypoint pose estimation, and geometrybased world-coordinate projection.

## 3.1 Stage 1: High-Resolution Player Detection via Adaptive Tiling

To eliminate boundary-splitting artifacts inherent in standard sliced inference, we propose Boundary-Aware Adaptive Tiling (BAAT), a semantics-guided cropping strategy. First, a YOLO model processes the full 4K frame downsampled to 1280 × 1280 pixels to generate a set of coarse bounding boxes $\mathcal { D } = \{ b _ { i } \} _ { i = 1 } ^ { N }$ . Simultaneously, we initialize a standard SAHI grid W utilizing a 1280 × 1280 pixel slice size and a 0.2 spatial overlap.

Rather than relying purely on fixed geometry, BAAT utilizes the coarse detections D to iteratively expand the initial windows W. For a given window $w = ( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ , let $B ( w ) \subseteq { \mathcal { D } }$ denote the set of coarse boxes that intersect w but are not fully contained within it. At each iteration, the window edges are expanded to enclose these intersecting boxes alongside a safety margin of $\delta = 5$ pixels, clamped to the image width W and height H:

$$
\begin{array} { r l } & { x _ { 1 } \gets \operatorname* { m a x } \biggl ( 0 , \displaystyle \operatorname* { m i n } _ { b \in \mathcal { B } ( w ) } b _ { x _ { 1 } } - \delta \biggr ) , \qquad y _ { 1 } \gets \operatorname* { m a x } \biggl ( 0 , \displaystyle \operatorname* { m i n } _ { b \in \mathcal { B } ( w ) } b _ { y _ { 1 } } - \delta \biggr ) , } \\ & { x _ { 2 } \gets \operatorname* { m i n } \biggl ( W , \displaystyle \operatorname* { m a x } _ { b \in \mathcal { B } ( w ) } b _ { x _ { 2 } } + \delta \biggr ) , \qquad y _ { 2 } \gets \operatorname* { m i n } \biggl ( H , \displaystyle \operatorname* { m a x } _ { b \in \mathcal { B } ( w ) } b _ { y _ { 2 } } + \delta \biggr ) . } \end{array}\tag{1}
$$

This expansion systematically ensures full object containment. The process repeats until no partially contained coarse box remains, bounded by a maximum of $T = 1 0 0$ iterations, and converges empirically within three to five iterations.

## 3.2 Stage 2: Two-Point Keypoint Estimation via Adapted RTMPose

Standard top-down pose estimators are designed for full-body skeletons (e.g., 17 COCO joints), over-parameterizing our metric-localization task which requires only two geometrically coupled keypoints. To address this, we adapt the RTMPose-X [10] architecture into a specialized two-point estimator by strictly contracting the output space of its RTMCCHead from $C _ { \mathrm { o u t } } = 1 7 \mathrm { t o } C _ { \mathrm { o u t } } = 2$ , targeting exactly the pelvis and its orthogonal ground projection.

The RTMCCHead employs the SimCC one-dimensional coordinate classification [11] and a Gated Attention Unit (GAU) [15] to refine keypoint representations. The GAU integrates a gated linear unit with self-attention, computing the output O as:

$$
A = \frac { 1 } { n } \mathrm { r e l u } ^ { 2 } \left( \frac { Q ( X ) K ( Z ) ^ { \top } } { \sqrt { s } } \right) , \quad O = ( U \odot A V ) W _ { o }
$$

where $U , V ,$ and Z are linear projections of the input X, s is the scaling dimension, and A is the attention matrix. By contracting the output dimension to $C _ { \mathrm { o u t } } = 2$ , the GAU sequence length n naturally contracts to 2. Consequently, A becomes a dense $2 \times 2$ attention matrix that exclusively models the relationship between the pelvis and the ground-projection keypoint. Because these two points are physically coupled by perspective geometry—their relative image displacement encodes the athlete’s height under foreshortening—this contracted attention mechanism structurally incentivizes the network to implicitly internalize camera-projection physics without auxiliary supervision.

Finally, the predicted 2D ground-projection keypoint is deterministically mapped to metric world coordinates by un-projecting the point via the provided camera calibration and intersecting the resulting ray with the pitch plane $( Z = 0 )$ .

## 4. EXPERIMENTS

## 4.1 Experimental Setup

Dataset. We evaluate on the Spiideo SoccerNet SynLoc benchmark [4], comprising 4K broadcast frames of synthetic athletes composited onto real-world backgrounds across 17 arenas. To rigorously test generalization, the public test (9.3K images) and private challenge (11.4K images) sets feature strictly held-out arenas unseen in the training split (42.5K images). The dataset provides image-space keypoints (pelvis and ground-projection), 3D metric ground truths, and explicit camera calibration parameters $( P = [ R | t ]$ and radial distortion polynomials) essential for world-coordinate ray casting.

Metric and Baseline. The primary evaluation metric is mAP-LocSim [4], which evaluates pure worldcoordinate precision. It replaces bounding-box IoU with a metric-scale localization similarity function:

$$
L o c S i m ( d ) = \exp \left( { \frac { - \ln ( 0 . 0 5 ) \cdot d ^ { 2 } } { \tau ^ { 2 } } } \right)\tag{2}
$$

Table 1. Comparison between the organizer baseline and the proposed method on the public-test and challenge sets. Green entries indicate improvements over the baseline. AP is unavailable for the challenge set.
<table><tr><td>Set</td><td>Method</td><td>AP@0.50:0.95</td><td>LocSim@t=1</td><td>LocSim@t=5</td></tr><tr><td>Public test</td><td>Baseline</td><td>0.6958</td><td>76.17</td><td>96.40</td></tr><tr><td>Public test</td><td>Ours</td><td>0.9128</td><td>97.44</td><td>98.94</td></tr><tr><td>Challenge</td><td>Baseline</td><td></td><td>77.30</td><td>96.63</td></tr><tr><td>Challenge</td><td>Ours</td><td></td><td>96.67</td><td>98.98</td></tr></table>

where d is the ground-plane Euclidean distance in meters between the predicted and true athlete position, and $\tau = 1 \mathrm { m }$ . We compare against the benchmark’s baseline, a bottom-up YOLOX-pose framework that regresses the same two keypoints, achieving a LocSim of 76.17 on the public test set. Notably, standard 17-keypoint estimators cannot serve as valid baselines; canonical anatomical foot joints do not reliably lie on the pitch surface (Z = 0) during dynamic actions, rendering them geometrically incompatible with deterministic ground-plane ray casting.

## 4.2 Implementation Details

Detection: We fine-tune YOLO26-Large [12] on fixed SAHI 1280 × 1280 crops for 20 epochs. During inference, BAAT initializes with a 1280 × 1280 slice size and 0.2 spatial overlap. The coarse pass employs the same detector on the full 4K frame downsampled to $1 2 8 0 \times 1 2 8 0$ , utilizing a deliberate 0.001 confidence threshold to maximize baseline recall prior to downstream pose estimation.

Pose Estimation: The adapted RTMPose-X [10] is initialized from Body-7 pre-trained weights. Training utilizes the AdamW optimizer with a base learning rate of $1 0 ^ { - 4 }$ and cosine annealing down to $5 \times 1 0 ^ { - 6 }$ over 700 epochs (converging at epoch 62) with a batch size of 128 on a single GPU. Spatial augmentations include random horizontal flip, scaling, rotation, and CoarseDropout; Random HalfBody is explicitly excluded to prevent degenerate pelvis targets.

## 4.3 Main Results

As reported in Table 1, our framework significantly outperforms the organizer baseline across all metrics on the public test set, achieving a LocSim@t = 1 of 97.44 (+21.27) and an AP@0.50:0.95 of 0.9128 (+21.70). This robust performance generalizes to the strictly held-out arenas of the private challenge set, attaining a LocSim of 96.67 compared to the baseline’s 77.30. These substantial gains validate our sequential pipeline design: BAAT efectively saturates small-object detection recall across extreme scale disparities, isolating metric-scale precision under perspective distortion as the primary remaining error source. This distortion is subsequently mitigated by our geometrically coupled two-point pose formulation, demonstrating superior robustness over coarse boundingbox surrogates. Qualitative comparisons (Figure 2) further confirm these capabilities, illustrating our method’s ability to accurately recover distant athletes and cleanly resolve dense, overlapping cases where the baseline fails.

## 5. ABLATION STUDY

Detection variants. To understand the impact of the tiling strategy and detector training domain, we evaluate five configurations: C uses full-image training and full-image inference; B1 uses SAHI-crop training and fixed SAHI inference; B2 uses full-image-trained weights with fixed SAHI inference; A1 uses SAHI-trained weights with Boundary-Aware Adaptive Tiling; and A2 uses full-image-trained weights with Boundary-Aware Adaptive Tiling. In all five variants, the same adapted RTMPose-X pose estimator and ray-casting stage are applied downstream, so any performance diferences are attributable solely to the detection stage.

## 5.1 Efect of Tiling Strategy on Detection

To isolate the impact of our tiling strategy and detector training domain, we evaluate five configurations (Table 2). Full-image inference (Approach C) establishes a weak baseline for small objects $( \mathrm { A P _ { S } }$ of 0.7668) due to resolution collapse at $1 2 8 0 \times 1 2 8 0$ . While fixed SAHI tiling (Approach B2) substantially recovers these missed instances $( \mathrm { A P _ { S } }$ of 0.8220), boundary-splitting artifacts impose a strict recall ceiling. Approach A1, combining a SAHItrained detector with Boundary-Aware Adaptive Tiling, eliminates these artifacts by construction, achieving the best overall performance (AP@0.50:0.95 of 0.9128, $\mathrm { { A R _ { S } } }$ of 0.8916). The marginal drop in $\mathrm { A P _ { L } }$ compared to Approach B2 is an acceptable trade-of attributed to modest scale shifts during boundary expansion for large, nearby athletes. Finally, the consistent performance gap between Approaches A1 and A2 confirms that detectors must be trained on crops structurally aligned with their inference distribution. Qualitative comparisons are provided in Figure 3.

![](images/2a84b608401855b624548768354fc0c4f6c6f98e5d5e421021a06c560d004bd7.jpg)

![](images/49f29ec666ba05af5e9687399114ab2eca8b0288023112c6e49a504ac2a69ff6.jpg)

![](images/aaf3d5a61cc6ad4a44d314d793eec4d8eb9fb062280affdf40513c1597a3f83a.jpg)

![](images/53c92c56d3a0f6220a0aecb09b97a19e34b19b626de1cc83914c8bf1130bf4f6.jpg)

![](images/aa36b2c68c483fc8f9dc7335f2186cb35ef3d5c5c26b24c27ba740fb0261b4cb.jpg)

![](images/c9ce6ea045413f4e09c45aaadad6d80f1e768970811c864c0eea44301d4fc6b8.jpg)

![](images/dc2c5f620a3fdff95f05a48d2199fe876721ee003a7477704328c1b088bc70dd.jpg)  
GT

![](images/7d50a48031c725a32c349a36c7bea6c87db64932ebe38b832fe019de4e382f38.jpg)  
Baseline

![](images/d314da6b24c03baf2e284f068fc7ca6679a38d0f45d0fc15756069d2f1672899.jpg)  
Ours  
Figure 2. Qualitative comparison of athlete localization. Ours is more robust to scale variation and overlap, accurately predicting pelvis and ground-projection keypoints.

![](images/73a4a081ffab55132a4328b011364faba28b59eb857e537051efb94632174508.jpg)  
GT

![](images/fc40ed94d7a1b2ba6b47c69048e0d8dfcaa0b3728b2eb588d0de12a8e7f1d57b.jpg)  
Baseline

![](images/ad2f1c261f15bcb9c413b522b4b082acddc732eb5a28ff9640b02ab4e73e5111.jpg)  
Full image

![](images/20055c67e189f313ab386da6d55083cf19ce64b19601f7aa999c330e0856b686.jpg)  
SAHI

![](images/a3734736f42769709b3ce73dfb55e5e50004e9d03a42b10b0cebb7051bcb223e.jpg)  
Ours  
Figure 3. Qualitative detection outputs on a far-touchline crop under Approach C (full-image), B2 (fixed SAHI), and A1 (Boundary-Aware Adaptive Tiling). Missed athletes are progressively recovered and boundary-splitting artifacts are eliminated under A1.

## 5.2 Implicit Geometric Learning in the Two-Keypoint Formulation

To evaluate whether the two-keypoint formulation implicitly captures perspective geometry without receiving camera parameters as input, we compare the model’s predicted pelvis-to-ground pixel displacement $( \Delta \hat { p } \ =$ $\| \hat { p } _ { \mathrm { p e l v i s } } - \hat { p } _ { \mathrm { g r o u n d } } \| _ { 2 } \Big )$ against a reference geometric curve. This physically correct reference is constructed by

Table 2. Detection ablation on the public test set. Approach A1 achieves the best performance across nearly all metrics. The exception is $\mathrm { A P _ { L } } .$ , where Approach B2 marginally leads; see text for discussion. Best values per column are highlighted.
<table><tr><td>Method</td><td>AP@0.50</td><td>AP@0.50:0.95</td><td>AP@0.75</td><td> $\mathbf { A P _ { L } }$ </td><td> $\mathbf { A P _ { M } }$ </td><td> ${ \bf A P s }$ </td><td>AR@1</td><td>AR@10</td><td>AR@100</td><td> $\mathbf { A R } _ { \mathbf { L } }$ </td><td> $\mathbf { A R } _ { \mathbf { M } }$ </td><td>ARs</td></tr><tr><td>B1</td><td>0.9654</td><td>0.7895</td><td>0.8771</td><td>0.1808</td><td>0.8755</td><td>0.7521</td><td>0.0590</td><td>0.4451</td><td>0.8216</td><td>0.4232</td><td>0.9102</td><td>0.7664</td></tr><tr><td>C</td><td>0.9764</td><td>0.8133</td><td>0.9002</td><td>0.3809</td><td>0.9033</td><td>0.7668</td><td>0.0595</td><td>0.4514</td><td>0.8382</td><td>0.5856</td><td>0.9285</td><td>0.7790</td></tr><tr><td>B2</td><td>0.9792</td><td>0.8756</td><td>0.9363</td><td>0.9743</td><td>0.9581</td><td>0.8220</td><td>0.0620</td><td>0.4791</td><td>0.8897</td><td>0.9927</td><td>0.9707</td><td>0.8306</td></tr><tr><td>A2 A1</td><td>0.9787</td><td>0.8811</td><td>0.9365</td><td>0.8862</td><td>0.9444</td><td>0.8496</td><td>0.0616</td><td>0.4825</td><td>0.9035</td><td>0.9872</td><td>0.9643</td><td>0.8590</td></tr><tr><td></td><td>0.9895</td><td>0.9128</td><td>0.9669</td><td>0.9728</td><td>0.9652</td><td>0.8822</td><td>0.0623</td><td>0.4884</td><td>0.9286</td><td>0.9955</td><td>0.9791</td><td>0.8916</td></tr><tr><td>YOLOX-pose baseline</td><td>0.9543</td><td>0.6958</td><td>0.7869</td><td>0.8020</td><td>0.8065</td><td>0.6277</td><td>0.0556</td><td>0.4121</td><td>0.7420</td><td>0.9158</td><td>0.8699</td><td>0.6485</td></tr></table>

Table 3. Geometric consistency between predicted and reference displacement curves. Higher correlation and lower MAE indicate closer alignment with true perspective geometry.
<table><tr><td>Metric Value</td></tr><tr><td>Curve-level MAE (px) ↓ 0.333</td></tr><tr><td>Correlation (pred vs. reference) ↑ 0.8963</td></tr></table>

projecting 3D ground-truth annotations into image space using the provided camera calibration. Both sequences are then summarized as distance-binned medians and directly compared (Figure 4).

The predicted and reference curves align closely, following the same monotonic decrease expected from perspective foreshortening. As reported in Table 3, the predictions achieve a curve-level MAE of 0.333px and a Pearson correlation coeficient of 0.8963 across the pitch distance range. This high consistency establishes that restricting the model’s capacity exclusively to this physically coupled point pair provides a structural incentive to preserve their geometric relationship in the predictions-a characteristic unlikely to emerge in full-body pose estimators burdened with anatomically irrelevant joints. The mean signed bias of -1.214px reflects a margina tendency to underestimate displacement, predominantly localized to the sparsely sampled near-field range (7- 12m, $\mathrm { M A E } = 2 4 . 8 3 0 \mathrm { p x } )$ . This data-driven bias identifies a concrete direction for future improvement through targeted data augmentation or geometry-aware loss functions.

![](images/530f4eb22c39fcb0c98dd1d68cf5b35c1d430bc2d65c668ee907d331e44046e3.jpg)  
Figure 4. Predicted pelvis-to-ground displacement closely matches the reference geometric curve across camera-to-athlete distances (MAE 0.333px, corr. 0.8963). This confirms the two-keypoint model produces geometrically consistent outputs without camera parameter input.

## 5.3 Detection Upper Bound

To disentangle the contributions of the detection and pose estimation stages, we establish an upper bound by replacing the predicted bounding boxes from Approach A1 with ground-truth annotations, holding the RTMPose-X estimator and ray-casting stages fixed. As reported in Table 4, this perfect-detection scenario yields a LocSim of 98.86, representing a marginal absolute gain of 1.42 points over our predicted boxes (97.44). This narrow margin validates the efectiveness of Boundary-Aware Adaptive Tiling, confirming that the detection stage performs near-optimally and contributes minimally to residual localization error. Consequently, further improvements to the detection architecture or tiling strategy would yield diminishing returns. The primary bottleneck for end-to-end metric-scale localization lies squarely in the pose estimation stage-specifically, the spatial precision of the ground-projection keypoint under perspective distortion, occlusion, and extreme scale variance near the far touchline.

Table 4. Detection upper-bound analysis on the public test set. Ground-truth bounding boxes replace predicted detections while all downstream stages remain fixed, isolating the contribution of the detection stage to the final LocSim.
<table><tr><td>Detection Input</td><td>LocSim</td><td>∆ vs. Predicted</td></tr><tr><td>Predicted Boxes (Approach A1)</td><td>97.44</td><td></td></tr><tr><td>Ground-Truth Boxes (Upper Bound)</td><td>98.86</td><td>+1.42 ↑</td></tr></table>

## 6. CONCLUSION

We propose a two-stage top-down pipeline for metric-scale athlete localization from single-frame broadcast imagery. Boundary-Aware Adaptive Tiling eliminates boundary-splitting by adaptively expanding tiles using coarse detections, ensuring full object containment without annotations or architectural changes. A two-keypoint RTMPose-X formulation predicts pelvis and ground projection, enabling implicit perspective learning, with world coordinates recovered via ray casting. The method achieves 97.44 LocSim and 0.9128 AP, outperforming the baseline by over 21 points and generalizing to the private set.

Future work targets eficiency and geometric modeling. The sequential pipeline incurs latency due to perinstance pose inference, motivating batched inference, lightweight models, distillation, and hybrid designs. Incorporating geometry-aware supervision, such as calibration-based constraints, and leveraging monocular depth or geometric priors may improve robustness under occlusion, foreshortening, and scale variation, enhancing physical consistency.

## REFERENCES

[1] Cioppa, A., Giancola, S., Deli\`ege, A., Seikavandi, M., Ghanem, B., and Van Droogenbroeck, M., “Camera calibration and player localization in soccernet-v2 and investigation of their representations for action spotting,” in [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops], 4532–4541 (2021).

[2] Firstauthor, F., Secondauthor, F., and Thirdauthor, F., “Deep learning-based detection of players and teams in soccer,” Applied Computing and Informatics (2025).

[3] Somers, V., Cioppa, A., Giancola, S., Deli\`ege, A., Van Droogenbroeck, M., Ghanem, B., et al., “Soccernet 2024: A benchmark for holistic understanding of soccer video,” arXiv preprint arXiv:2409.10587 (2024).

[4] Ard¨o, H., Nilsson, M. G., Cioppa, A., Magera, F., Giancola, S., Liu, H., Ghanem, B., and Van Droogenbroeck, M., “Spiideo soccernet synloc: Single frame world coordinate athlete detection and localization with synthetic data,” in [Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISAPP)], 278–285, SciTePress (2025).

[5] Puwal, F. and Others, F., “SPL-BEV: Soccer player localization and birds-eye-view estimation,” in [Proceedings of the European Conference on Computer Vision Workshops], (2025).

[6] Zhu, P., Wen, L., Du, D., Bian, X., Fan, H., Hu, Q., and Ling, H., “Detection and tracking meet drones challenge,” IEEE Transactions on Pattern Analysis and Machine Intelligence (2021).

[7] Akyon, F. C., Altinuc, S. O., and Temizel, A., “Slicing aided hyper inference and fine-tuning for small object detection,” in [2022 IEEE International Conference on Image Processing (ICIP)], 966–970, IEEE (2022).

[8] Liu, K., Fu, Z., Jin, S., Chen, Z., Zhou, F., Jiang, R., Chen, Y., and Ye, J., “ESOD: Eficient small object detection on high-resolution images,” IEEE Transactions on Image Processing (2024).

[9] Xu, Y., Zhang, J., Zhang, Q., and Tao, D., “ViTPose: Simple vision transformer baselines for human pose estimation,” in [Advances in Neural Information Processing Systems (NeurIPS)], 35 (2022).

[10] Jiang, T., Lu, P., Zhang, L., Ma, N., Han, R., Lyu, C., Li, Y., and Chen, K., “Rtmpose: Real-time multi-person pose estimation based on MMPose,” arXiv preprint arXiv:2303.07399 (2023).

[11] Li, Y., Yang, S., Liu, P., Zhang, S., Wang, Y., Wang, Z., Yang, W., and Xia, S.-T., “Simcc: A simple coordinate classification perspective for human pose estimation,” in [European Conference on Computer Vision (ECCV)], 502– 518, Springer (2022).

[12] Sapkota, R., Cheppally, R. H., Sharda, A., and Karkee, M., “Yolo26: key architectural enhancements and performance benchmarking for real-time object detection,” arXiv preprint arXiv:2509.25164 (2025).

[13] Guti´errez-P´erez, M. and Agudo, A., “PnLCalib: Sports field registration via points and lines optimization,” Computer Vision and Image Understanding 267, 104712 (2026).

[14] Somers, V., Joos, V., Cioppa, A., Giancola, S., Ghasemzadeh, S. A., Magera, F., Standaert, B., Mansourian, A. M., Zhou, X., Kasaei, S., Ghanem, B., Alahi, A., Van Droogenbroeck, M., and De Vleeschouwer, C., “SoccerNet game state reconstruction: End-to-end athlete tracking and identification on a minimap,” in [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)], (2024).

[15] Hua, W., Dai, Z., Liu, H., and Le, Q. V., “Transformer quality in linear time,” in [Proceedings of the 39th International Conference on Machine Learning (ICML)], Proceedings of Machine Learning Research 162, 9099–9117, PMLR (2022).