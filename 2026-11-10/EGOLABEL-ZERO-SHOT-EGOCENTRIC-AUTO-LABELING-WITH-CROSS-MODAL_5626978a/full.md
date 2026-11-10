# EGOLABEL: ZERO-SHOT EGOCENTRIC AUTO-LABELING WITH CROSS-MODAL CONSISTENCY

Xiaozhi Zhang<sup>1,\*</sup>, Changjia Xu<sup>1</sup>, Yaoyuan Yan<sup>2</sup>, Haoxiang Jie<sup>2</sup>, Bingjie Fu<sup>3,4</sup>

<sup>1</sup>Robotics Lab, Country Garden Services

<sup>2</sup>AI Lab, Country Garden Services

<sup>3</sup>School of Computer Science and Engineering, Northeastern University, Shenyang

zhangxiaozhi07@bgyfw.com

## ABSTRACT

Egocentric auto-labeling requires depth, camera pose, object, and human-pose estimates to agree across independently developed models. Differences in units, coordinate conventions, and scale can make individually plausible predictions geometrically inconsistent. We present EgoLabel, a zero-shot pipeline that combines off-the-shelf models without target-domain labeled training and provides explicit cross-modal registration and consistency diagnostics. The pipeline integrates MonST3R for geometry, LocateAnything with GMC-BoT-SORT and SAM 2 for objects, and V-HPOT for hands; FisheyeViT provides a separate whole-body compatibility demonstration. Our main hand evaluation uses H2O sensor depth. Across 20 sequences, per-hand depth anchoring reduces the mean of persequence median hand–depth residuals by 23% relative to a single global scale, yielding a pooled median of 84.6 mm against a 20.5 mm ground-truth reference. Hand PA-MPJPE is 14.53 ± 1.29 mm, whereas absolute MPJPE is 119.1 ± 31.3 mm, revealing a substantial gap between aligned and metric accuracy. On three RGB-only sequences, camera ATE is 1.40 ± 0.29 cm after Sim(3) alignment. We also identify a sliding-window implementation issue that allows sensor-initialized depth to change during optimization. Because hand anchoring and its residual share the same depth map, the residual measures internal consistency rather than independent accuracy. These results characterize both the utility and the current limits of the registration pipeline.

Keywords Egocentric vision · Auto-labeling · Cross-modal consistency · Embodied intelligence

## 1 Introduction

Egocentric (first-person) vision supports activity understanding, hand–object interaction analysis, and embodied learning. Benchmarks such as Ego4D [1], EPIC-KITCHENS [2], H2O [3], AssemblyHands [4], HOT3D [5], and SceneEgo [6] have enabled progress in these areas, but producing their annotations remains costly. Depth, camera trajectories, hands, objects, and body motion typically require separate capture or processing pipelines, adding work as the number of modalities increases. Automatically generating jointly registered annotations from a single recording could help scale the supply of first-person interaction data for embodied models and robot policies [7, 8].

Recent foundation models provide many of the required components: DUSt3R [9] and MonST3R [10] for geometry, HaMeR [11] and V-HPOT [12] for hands, FisheyeViT [13] for the body, and open-vocabulary detectors, trackers, and segmenters [14, 15, 16, 17] for objects. These models can be deployed without target-domain labeled training. In particular, monocular geometry models estimate depth and 6DoF camera trajectories from RGB, although their global scale remains ambiguous.

Combining these outputs requires more than running the models in sequence. Independently developed models use different units (millimeters, meters, or focal-length-normalized scales), axis conventions, and reference frames. A hand skeleton in virtual-camera units cannot be directly combined with a camera trajectory in meters; similarly, lifting object detections into a shared 3D scene requires compatible depth, poses, and intrinsics. Such mismatches can remain unnoticed because each model returns plausible predictions and is evaluated on its own task. Per-module accuracy therefore does not establish agreement between annotation streams.

We present EgoLabel, a zero-shot toolkit for registering and inspecting heterogeneous egocentric annotations. Here, zero-shot means that no target-domain labeled training is performed. The pipeline converts V-HPOT outputs using the recording camera’s intrinsics, anchors each hand to depth, and lifts object detections using depth and estimated camera poses. We evaluate cross-modal agreement through a hand–depth residual and an object-trajectory smoothness score, with ground-truth references, a constant-scale control, and a synthetic depth-scale response analysis. The main hand study uses sensor depth, while a smaller RGB-only study examines camera pose and hand–depth consistency. The registration procedure and diagnostics make the assumptions and failure modes of model composition explicit; their scope and dependence on shared inputs are specified in Section 3.4.

Our contributions are therefore:

• A zero-shot egocentric auto-labeling pipeline. EgoLabel integrates MonST3R, LocateAnything with GMC-BoT-SORT and SAM 2, and V-HPOT into a pipeline for geometry, object, and hand annotations. A separate FisheyeViT demonstration examines whole-body compatibility, and a structured-captioning application illustrates downstream use (Sections 3 and 3.5).

• Explicit registration and consistency diagnostics. We document unit conversion, coordinate alignment, and per-hand depth anchoring, and assess their outputs using hand–depth residuals, object-trajectory smoothness, and control experiments. Across 20 H2O sequences, anchoring reduces the mean of per-sequence median residuals by 23% relative to a single global scale, with improvements on 17 of 20 sequences. The residual is an internal diagnostic because it shares depth with calibration; wrist-excluded, holdout, and perturbation analyses further characterize this dependence (Section 4.3).

• An evaluation with explicit scope and failure analysis. We report hand accuracy and consistency on 20 H2O sequences (12,032 frames and 21,876 hand samples), camera ATE on three sequences, and an object-trajectory case study on one sequence. Paired camera experiments show that the implemented sensor-depth initialization does not keep depth fixed during sliding-window optimization and yields worse ATE than RGB-only estimation. We distinguish component accuracy, cross-modal diagnostics, and qualitative demonstrations throughout the evaluation (Section 4).

Figure 1 summarizes the pipeline. Section 2 reviews related work; Section 3 describes the models, registration procedure, and consistency measures; Section 4 presents the evaluation, computational cost, and failure analysis; and Section 5 concludes.

## 2 Related Work

Egocentric perception and datasets. Egocentric datasets provide complementary forms of supervision. Ego4D [1] and EPIC-KITCHENS [2] offer large-scale activity narration and action labels; AssemblyHands [4] and HOT3D [5] provide 3D hand annotations from multi-view or headset capture; and SceneEgo [6] targets whole-body motion. H2O [3] provides sensor depth, 6DoF camera poses, hand joints, and object poses in the same recording, making it suitable for evaluating agreement across modalities. We use H2O for quantitative evaluation and the in-house ML-EgoDex dataset for compatibility checks (Section 4).

Geometry foundation models. DUSt3R [9] predicts dense pointmaps for image pairs, providing per-pixel 3D coordinates from which depth, intrinsics, and relative camera poses can be recovered. MonST3R [10] extends this representation to dynamic video through fine-tuning and sliding-window global alignment of poses, pointmaps, and focal length. Its sequence-level alignment provides the geometric backbone of our pipeline. Monocular geometry remains scale-ambiguous, however, so we handle metric anchoring explicitly (Section 3.3). The behavior of sensor-depth initialization in the configuration we use is detailed in Section 3.2.1.

Hand and whole-body pose estimation. MANO-based regressors such as HaMeR [11] predict parametric hand meshes from image crops. Their crop-normalized camera models require conversion to the recording camera’s intrinsics before integration with scene geometry. V-HPOT [12] instead predicts joint depths in a virtual camera normalized by focal length and image height. It also supports label-free test-time optimization using a 3D-consistency objective over random crops, for which the source paper reports reductions in mean 3D keypoint error of 71% on H2O and 41% on AssemblyHands. We use the frozen checkpoint without test-time optimization and explicitly convert and anchor its virtual-camera outputs (Sections 3.2.3 and 3.3). For whole-body pose, FisheyeViT [13] models head-mounted fisheye projection directly with a distortion-aware head. We transform its camera-space keypoints for visualization in a separate compatibility demonstration.

Open-vocabulary detection, tracking, and segmentation. Detection has moved from closed-set detectors such as Faster R-CNN [18] to open-vocabulary grounding models such as GroundingDINO [14] and LocateAnything [15], which take a natural-language prompt and return boxes for categories named only by that prompt. Tracking has moved from SORT [19] and ByteTrack [20] to BoT-SORT [16], whose global motion compensation corrects predicted boxes by an affine warp estimated from sparse optical flow before association—a correction that matters because head motion alone can displace the image by hundreds of pixels per frame. Segmentation has moved from per-frame prompting to SAM 2 [17], which propagates masks through a video and yields temporally coherent instances (our implementation uses per-frame box prompts with the image predictor rather than video memory propagation). Chained together, these models give identity-preserving, open-vocabulary object masks with no category-specific training. Two properties bear on our scheme: the stream is 2D, so lifting it to 3D requires the depth and pose from the geometry module; and it is identity-preserving across frames, which is what makes the object-trajectory smoothness score of Section 3.4 meaningful.

Auto-labeling and cross-modal consistency. Auto-labeling with foundation models is commonly evaluated per task. Such evaluations establish component accuracy but do not directly test whether independently produced annotations describe a consistent 3D scene. EgoLabel focuses on this cross-modal question by making scale and coordinate conversions explicit and measuring agreement against ground-truth references. To our knowledge, comparable evaluation of integrated egocentric auto-labeling pipelines remains limited. We therefore use internal controls—a ground-truth reference residual, constant-scale anchoring, and synthetic depth-scale perturbations—while acknowledging the absence of a head-to-head system comparison.

## 3 Methodology

## 3.1 Problem Formulation

Given an egocentric monocular RGB video $\mathcal { V } = \{ I _ { t } \} _ { t = 1 } ^ { T }$ captured by a head-mounted camera with known intrinsics $\mathbf { K } \in \mathbb { R } ^ { 3 \times 3 } .$ —and, optionally, a per-frame sensor depth map—we aim to produce, for every frame $t ,$ a consistent set of annotations:

$$
\mathbf { D } _ { t } \in \mathbb { R } ^ { H \times W }
$$

$$
\mathrm { d e p t h \ m a p } ,\tag{1}
$$

$$
\mathbf { T } _ { t } \in \mathrm { S E } ( 3 ) \qquad \quad \quad 6 \mathrm { D o F } \mathrm { c a m e r a } \mathrm { p o s e } ( \mathrm { C A M 2 W O R L D } ) ,\tag{2}
$$

$$
\{ \mathbf { b } _ { i } , \mathbf { m } _ { i } , c _ { i } \} _ { i }\tag{3}
$$

$$
\{ \mathbf { J } _ { t } ^ { h } \} _ { h \in \{ \mathrm { l e f t , r i g h t } \} } \in \mathbb { R } ^ { 2 \times 2 1 \times 3 } \qquad \quad \mathrm { 3 D ~ h a n d ~ k e y p o i n t s ~ ( p e r ~ h a n d ) , }\tag{4}
$$

$$
\mathbf { S } _ { t }\tag{5}
$$

Depth and hand joints are represented in the per-frame camera coordinate system, and 3D annotations can be transformed to the first-frame camera coordinate system for joint visualization. Sensor depth provides metric scale; RGB-only depth and the hands anchored to it remain scale-ambiguous. Object world trajectories additionally depend on predicted camera translations, whose scale may differ from that of the depth, and are therefore treated as visualizations rather than calibrated world coordinates. $\mathbf { T } _ { t }$ denotes the world-from-camera transform (CAM2WORLD), $\mathbf { X } _ { w } = \mathbf { T } _ { t } \mathbf { X } _ { c _ { t } } ,$ with $\mathbf { T } _ { t } ^ { - 1 }$ mapping world points into the current camera frame. The camera convention is +X right, +Y down, and $+ Z$ forward (Section 3.2.1).

The pipeline uses explicit conversions to reconcile the scales and conventions of independently estimated annotations. In sensor-assisted mode, sensor depth initializes MonST3R and is also read directly for hand anchoring and object lifting. In RGB-only mode, these operations use MonST3R’s predicted depth. Both modes use the same registration procedure and consistency measures; Table 1 identifies the depth and pose sources used by each reported quantity.

## 3.2 Foundation Models

We use off-the-shelf foundation models without target-domain labeled training or fine-tuning. The following sections describe each module and the output conventions relevant to integration.

![](images/58446e568870a48285fa57c0a50a3b59710b905875efe6feb0e743cd74a37c99.jpg)  
Figure 1: Overall architecture of EgoLabel. Off-the-shelf models produce geometry (MonST3R), object annotations (LocateAnything + GMC-BoT-SORT + SAM 2), and hand poses (V-HPOT). Unit conversion, coordinate alignment, and per-hand depth anchoring connect these streams for joint inspection. Hand–depth residuals, object-trajectory smoothness, and control experiments assess their consistency. FisheyeViT is demonstrated on separate fisheye recordings, and structured manipulation captioning is a downstream application. Metric-scale limitations are detailed in Section 3.4.

## 3.2.1 Depth Estimation and Camera Pose Estimation

MonST3R [10] extends DUSt3R’s pointmap representation [9] to dynamic video (ViT-Large encoder, Asymmetric-CroCo3DStereo decoder [21]). For a pair of frames it predicts a dense pointmap $\bar { \mathbf { P } } _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ per view—the 3D coordinates of every pixel, in a common reference frame—plus a confidence map $\mathbf { C } _ { t } ,$ , and is trained with a DUSt3R-style robust regression term,

$$
\mathcal { L } _ { \mathrm { p a i r } } = \sum _ { ( i , j ) } \sum _ { t \in \{ i , j \} } \left. \mathbf { C } _ { t } \odot \left( \mathbf { P } _ { t } - \mathbf { X } _ { t } \right) \right. _ { 1 } ,\tag{6}
$$

where $\mathbf { X } _ { t }$ is the target pointmap for view t (the source objective also normalizes the pointmaps and regularizes the confidence map, which we omit here: we do not retrain the model). The two views are related by the relative pose $( \mathbf { R } _ { i j } , \mathbf { t } _ { i j } )$ through their shared referenceframe, not pixel-by-pixel—the same pixel in two frames is generally not the same scene point, and a dynamic scene is not a single rigid motion. Because monocular pointmaps are scale-ambiguous, the absolute scale is fixed later by the global optimizer or by known depth.

Global alignment and known-depth initialization. A sliding-window global alignment jointly refines poses, perframe pointmaps and focal length under a temporal-smoothing prior and an optional flow loss. We use resolution 320×176, window W=30, overlap r=1/3, 300 iterations, flow-loss weight 1.0 and threshold 30 px. Where H2O’s 16-bit millimeter depth is available we convert it to meters and write it into the depth parameters,

$$
\mathtt { s c e n e . p r e s e t \_ d e p t h m a p } \big ( \{ \mathbf { D } _ { t } \} , \mathtt { r e q u i r e s \_ g r a d } \mathtt { = F a l s e } \big ) ,\tag{7}
$$

which in the batchified single-window path also disables the depth gradients. In the sliding-window path used for every H2O sequence (more than 30 frames) it is only a strong initialization: the window loader re-creates the window’s depth and focal length as optimizable parameters (opt\_im\_depthmaps, opt\_im\_focals, requires\_grad\_(True)

for the first window), re-enabling gradients on the parameters just set, so the injected map is the starting point of the optimization rather than a constraint. Section 4 measures the drift (96% of pixels end more than 1 cm away, median relative change 25%) and its effect on camera pose. We therefore call this mode known-depth initialization and keep the input sensor depth, the window-optimized depth and the RGB-only predicted depth distinct in every table and caption.

MonST3R outputs CAM2WORLD poses in a right-handed camera frame (+X right, $+ Y$ down, +Z forward), the frame H2O’s stored poses already use, so no axisflip is applied for H2O (OpenCV’s camera convention is the same). Data stored with the opposite up-axis (the in-house EgoDex-style files) is converted first by right-multiplying each pose with $\bar { \mathbf { F } } = \mathrm { d i a g } ( 1 , - 1 , \bar { - 1 } , 1 )$ , i.e. $T ^ { \prime } = T \bar { \bf F }$ , a 180<sup>◦</sup> rotation about the X axis that negates the $Y$ and $\dot { Z }$ directions in the camera frame while preserving right-handedness; we state the direction because applying it the other way round silently rotates the scene.

## 3.2.2 Object Segmentation and Tracking

Egocentric manipulation involves dense hand–object interactions where a small set of relevant objects (cups, lids, tools, utensils) appear, disappear, and re-enter the field of view under severe occlusion and camera motion. We therefore require an annotation pipeline that is open-vocabulary (not limited to a fixed category set) and identity-preserving across frames. Our pipeline chains three models: LocateAnything for detection, BoT-SORT for tracking, and SAM 2 for segmentation.

LocateAnything. We adopt LocateAnything [15], a vision-language grounding model that unifies open-vocabulary detection and fine-grained spatial localization in a single 3B-parameter VLM. Given an input frame $I _ { t }$ resized to a maximum side length of 800 px, LocateAnything takes a natural-language prompt listing the target categories and returns a set of detections $\mathcal { D } _ { t } = \{ ( b _ { i } , c _ { i } , s _ { i } ) \} _ { i = 1 } ^ { N _ { t } }$ with bounding boxes, semantic labels, and confidence scores in a single forward pass. We adopt LocateAnything as the primary detector and retain GroundingDINO [14] as a complementary option; both are prompted with the target-category list, and we select LocateAnything for its grounding quality on cluttered scenes (we make no claim that either model was trained for first-person imagery). We apply NMS with an IoU threshold of 0.5 to suppress duplicate proposals, and discard detections with confidence below 0.35.

BoT-SORT. For identity-preserving tracking we adopt BoT-SORT [16], which augments the two-stage ByteTrack [20] association with camera-motion compensation (GMC): each track keeps a constant-velocity Kalman state and, before association, its predicted box is corrected by an affine warp estimated from sparse Lucas-Kanade optical flow [22] via RANSAC. The correction matters for head-mounted video, where ego motion alone can displace the image by hundreds of pixels per frame and would otherwise fragment identities. Association solves a confidence-modulated IoU assignment with a gate $\tau _ { \mathrm { m a t c h } } = 0 . 8 ,$ , matching high-confidence detections first and then low-confidence ones (those surviving NMS) to recover occluded objects; we disable the appearance ReID head, which is less reliable than spatial overlap in dense, hand-occluded scenes [23], and terminate tracks after 30 unmatched frames.

SAM 2. With track identities established, we generate per-instance segmentation masks using SAM 2 [17] (Hiera-L checkpoint, sam2.1\_hiera\_large.pt). Each Kalman-corrected bounding box serves as a box prompt to the image predictor, producing a binary mask $\mathbf { \dot { m } } _ { t , i } \in \{ 0 , 1 \} ^ { H \times W }$ for object i at frame t. To avoid GPU memory contention between the 3B VLM detector and the 224M-parameter SAM 2 encoder, we explicitly release the detector’s VRAM before loading the segmentation model. The final output per frame is an object record {track\_id, label, bbox, mask\_path}, written incrementally into a structured tracks.json that is consumed by downstream modules (3D lifting and overlay rendering).

![](images/e8833ae7f84cf0cc2880e7e27de5aa220e1b6c9bf5d88ce742cd99087768b9a9.jpg)

![](images/57ab3310ddb578db1a01513c26868928c2436e94d42c12c032c768e290445765.jpg)

![](images/4ebf1e105402a5455ad84f2151ad36fc1450021d5753ba4e403a93982e89985a.jpg)  
Figure 2: Object tracking and segmentation on H2O h1/0 (frames 0, 225, 449). LocateAnything detects the prompted objects, GMC-BoT-SORT assigns identity-preserving track IDs (colored boxes), and SAM 2 produces per-instance masks overlaid on the frames. The identities are maintained across head motion and partial hand occlusion by the objects.

## 3.2.3 Egocentric 3D Hand Pose Estimation

For 3D hand pose estimation we adopt V-HPOT [12], which is camera-agnostic by construction: it predicts keypoint depths in a virtual camera normalized by focal length and image height, so it does not overfit the recording camera’s intrinsics. With focal length f (we use $\bar { f _ { x } ) }$ , image height H (the network input height) and virtual parameters $( f _ { v } , H _ { v } )$

$$
\alpha = \frac { f _ { v } } { f } \cdot \frac { H } { H _ { v } } , \qquad z _ { \mathrm { v i r t u a l } } = \alpha z _ { \mathrm { m e t r i c } } , \qquad z _ { \mathrm { m e t r i c } } = z _ { \mathrm { v i r t u a l } } / \alpha ,\tag{8}
$$

i.e. the model’s depth is the metric depth scaled by $\alpha .$ We follow the source paper’s direction and record in the supplementary material which direction our released checkpoint uses: applying the factor the other way round would only rescale the whole skeleton by a constant, so scale-invariant metrics would not reveal the error. The formula uses f at the resolution where the keypoint pixel coordinates are expressed; if the input image is resized to H the focal length must be scaled accordingly. Keypoints are unprojected with the dataset intrinsics,

$$
\mathbf { X } = \mathbf { K } ^ { - 1 } \left[ \mathbf { \Lambda } _ { 1 } ^ { \mathbf { X } } \right] \cdot \boldsymbol { z } ,\tag{9}
$$

which gives $X = ( u - c _ { x } ) z / f _ { x }$ and $Y = ( v - c _ { y } ) z / f _ { y } ,$ , with +Y down in the image, consistent with the camera frame defined above. The backbone is an EfficientNet-V2-S with a depth head and a hand head that predicts 21 heatmaps and per-joint depths per hand plus a handedness score; on two-hand data it returns 42 keypoints in each frame’s camera frame. The model also offers a self-supervised test-time optimization (TTO) that refines it without labels; our released configuration does not use TTO—the checkpoint runs frozen, one forward pass per batch—so every hand number here is without TTO and we make no claim about its reported gains (Section 2).

Integration. For each frame V-HPOT consumes the RGB image resized to 224×224 and returns 42 keypoints in the virtual camera space, which require conversion and scale correction before combination with depth. We register them in two steps: divide the virtual depth by α (Eq. (8)) using the dataset intrinsics, then correct the residual absolute scale by anchoring the wrist to the depth (Section 3.3). Because V-HPOT outputs camera-space coordinates (unlike MANO-space predictors), no left/right X-mirroring is needed. Over the 20 H2O sequences it attains a PA-MPJPE of 14.53 ± 1.29 mm (21,876 hand samples); we do not directly compare this result with HaMeR because the quoted 18.27 mm comes from a different protocol.

## 3.2.4 Whole-Body Pose Estimation

We include the wearer’s body $\mathbf { S } _ { t }$ (Eq. (5)) for completeness, using FisheyeViT [13], a ViT-based estimator for headmounted fisheye cameras that models the fisheye projection explicitly instead of undistorting the input; we run its 256×256 checkpoint (EgoBody/EgoWholeBody). H2O uses a pinhole RealSense camera, so this is an independent compatibility demonstration on SceneEgo’s fisheye sequences, not a component of the H2O results, and we do not re-evaluate it (the source paper reports $\mathrm { M P J P E _ { b o d y } } \mathrm { \dot { = } 7 2 . \dot { 3 } }$ mm and $\mathbf { P A } \mathbf { - } \mathbf { M P J P E } _ { \mathrm { b o d y } } = 4 8 . 7$ mm on SceneEgo\_test). Its keypoints are mapped into the shared frame by the camera-to-world transform, $\mathbf { J } _ { j } ^ { w } = \mathbf { T } _ { \mathrm { c a m } } \mathbf { J } _ { j } ^ { \mathrm { c a m } }$ ; because the recording differs, this stream is not fused with the H2O streams and we claim only that the same registration procedure can place a body skeleton in the shared frame. Figure 3 shows representative outputs.

## 3.3 Pipeline Integration and Engineering Considerations

Integrating the models requires explicit handling of units, camera conventions, and scale. This section describes the data flow and the transformations used to connect their outputs.

End-to-end data flow. The integrated pipeline (implemented by h2o\_adapter.py) proceeds as follows:

1. Frames and intrinsics. Raw H2O RGB frames (1280×720) are normalized to the canonical naming {index:05d}.jpg; the camera intrinsics K are read from the dataset, and every downstream resolution change is an explicit resize (the MonST3R input is 320×176 and the V-HPOT input 224×224, neither of which is a proportional scaling of 1280×720, so 2D quantities are mapped through the corresponding scale factors and depths are sampled nearest-neighbor on the resized grid).

2. Depth and camera pose (MonST3R). MonST3R runs at 320×176 with sliding-window global alignment; the known H2O depth initializes the depth parameters (Eq. (7)), producing the depth map $\mathbf { D } _ { t }$ and poses $\mathbf { T } _ { t }$ stored in pred\_poses.hdf5.

![](images/9a917f1db99ad47eb070ac196cca17207b80f61b0f92d841ed3467911f85c49a.jpg)  
Figure 3: Qualitative whole-body results from the FisheyeViT stream. Each panel pairs a head-mounted fisheye frame (left) with the reconstructed 3D skeleton in the shared metric frame (right); predicted skeletons are orange and, where available, ground-truth skeletons are red. (a) Prediction only; (b)–(d) prediction overlaid with ground truth. The skeleton is mapped into the shared frame by the camera-to-world transform (Section 3.2.4); this is a compatibility demonstration on separate recordings, not a joint H2O result.

3. Object detection, tracking, and segmentation. LocateAnything performs open-vocabulary detection, GMC-BoT-SORT associates detections into identity-preserving tracks, and SAM 2 produces per-instance masks (Section 3.2.2).

4. 3D hand pose (V-HPOT). Each frame is resized to 224×224 and passed to V-HPOT, producing 42 keypoints (21 per hand) with handedness scores, stored as hand\_tracks.json.

5. Whole-body pose (FisheyeViT). The wearer’s body is estimated separately and mapped into the same frame via $\mathbf { J } ^ { w } = \dot { \mathbf { T } _ { \mathrm { { c a m } } } } \mathbf { J } ^ { \mathrm { { c a m } } }$ (Section 3.2.4).

6. Overlay and 3D visualization. An annotated overlay video is rendered and all entities are projected into the shared world frame to produce the final RRD visualization.

Unit conversion. The modules output different units, reconciled before fusion. H2O depth is 16-bit millimeters, divided by 1000 to meters before the known-depth initialization; V-HPOT keypoints are converted with Eq. (8) using the dataset intrinsics before scale correction; and the evaluation-only trajectory alignment is a 7-DoF similarity solved in closed form (Umeyama): with covariance $\begin{array} { r } { \mathbf { H } = \sum _ { t } ( \mathbf { q } _ { t } - { \pmb { \mu } } _ { q } ) ( \mathbf { p } _ { t } ^ { \star } - \acute { \pmb { \mu } } _ { p } ) ^ { \top } \dot { = } \mathbf { U } \dot { \Sigma } \mathbf { V } ^ { \top } } \end{array}$

$$
\mathbf { R } = \mathbf { U } \operatorname { d i a g } \big ( 1 , 1 , \operatorname* { d e t } ( \mathbf { U } \mathbf { V } ^ { \top } ) \big ) { \mathbf { V } ^ { \top } } , \quad s = \frac { \operatorname { t r } \big ( \Sigma \operatorname { d i a g } ( 1 , 1 , \operatorname* { d e t } ( \mathbf { U } \mathbf { V } ^ { \top } ) ) \big ) } { \sum _ { t } \| \mathbf { p } _ { t } - \mu _ { p } \| ^ { 2 } } , \quad \mathbf { t } = \mu _ { q } - s \mathbf { R } \mu _ { p } ,\tag{10}
$$

the Sim(3) solution minimizing $\begin{array} { r } { \sum _ { t } \| s \mathbf { R } \mathbf { p } _ { t } + \mathbf { t } - \mathbf { q } _ { t } \| ^ { 2 } ; } \end{array}$ ; it is applied only to report ATE. Registration uses no ground-truth poses: all streams are normalized to the first-frame camera space with $\mathbf { R } = \mathbf { I } ( \mathbf { T } _ { t }  \mathbf { T } _ { 0 } ^ { - 1 } \mathbf { T } _ { t } ; \mathbf { \Lambda } $ the reference-referenced alignment of Eq. (10) is evaluation-only). In the sensor-assisted mode, hand anchoring (Eq. (11)) and object lifting read $\mathbf { D } _ { \mathrm { s e n s o r } } ;$ the known-depth initialization (Section 3.2.1) is part of this mode. The RGB-only variant uses $\mathbf { D } _ { \mathrm { r g b } }$

Coordinate-system conventions. MonST3R and H2O share one right-handed frame (+X right, +Y down, +Z forward, CAM2WORLD; also OpenCV’s camera convention), so no flip is applied for H2O; the in-house EgoDex-style files store the opposite up-axis and are converted first by $\bar { \mathbf { F } } = \mathrm { d i a g } ( \bar { 1 } , \bar { - 1 } , \bar { - 1 } , 1 )$ , i.e. $T ^ { \prime } = T \bar { \bf F } ( { \bf a } 1 8 0 ^ { \circ }$ rotation about X). MANO-based regressors output the left hand mirrored, which V-HPOT avoids by working directly in camera space.

Metric-scale correction of the hand skeleton. V-HPOT keypoints keep a residual absolute-scale ambiguity, which we remove by anchoring the wrist to the depth map—sampling the median depth over the hand box and scaling the

![](images/6a76ac4d4c0c934590466b734536177ed91803a23603512c382350ab42e9e0a7.jpg)  
Figure 4: Final end-to-end output of EgoLabel on H2O h1/0, as rendered in the Rerun viewer at four stages of the manipulation: (a) lifting the box to examine it, (b) peeling the top flaps, (c) removing the flaps, and (d) pulling the box sides. Each panel is one screenshot of the integrated viewer and simultaneously shows the object stream (LocateAnything detections, GMC-BoT-SORT track IDs, and SAM 2 masks, drawn as colored boxes on the overlay video), the 3D hand prediction (left/right V-HPOT skeletons and the camera+wrist trajectory in 3D), and the ASID task caption (bottom panel: hand usage, manipulated object, and action). Object, hand, and camera annotations are produced by independent zero-shot models; the ASID caption (bottom panel) is a GT-assisted downstream consumer (Section 3.5).

skeleton,

$$
s _ { \mathrm { d e p t h } } = \frac { \mathrm { m e d i a n } \big \{ { \bf D } _ { t } ( u , v ) : ( u , v ) \in \mathrm { b b o x } _ { t } ^ { \mathrm { h a n d } } \big \} } { z _ { t } ^ { \mathrm { w r i s t } } } , \qquad \tilde { \bf { j } } _ { t } = s _ { \mathrm { d e p t h } } { \bf J } _ { t } .\tag{11}
$$

The depth ratio can differ substantially from one, so we use a wide clamp, $[ 1 0 ^ { - 6 } , 1 0 ^ { 6 } ]$ , to avoid unintended truncation. The pipeline applies Gaussian temporal smoothing with $\sigma = 2$ frames; the resulting scale factor therefore approximates the per-frame ratio. Under occlusion, the box median may sample the background or a manipulated object rather than the hand, motivating a hand-mask-based alternative (Section 4.5).

Operational details. The known-depth initialization requires meters: the values go into log-depth parameters, so a millimeter map is silently mis-scaled. Depth correction is unnecessary for the scale-invariant hand metrics but required for fusion, so that hands, objects and trajectory share one frame; all entities are logged under distinct paths (world/hands, world/gt\_hand, world/objects, world/trajectory\_\*) for frame-by-frame inspection.

## 3.4 Cross-Modal Consistency Measures

We assess cross-modal agreement using a hand–depth residual and an object-trajectory smoothness score. The residual reuses the depth map that calibrates hand scale, so it measures internal consistency rather than independently validating the anchoring. Most hand and object results are sensor-assisted; RGB-only results are labeled separately. Table 1 specifies the depth source, pose source, and coordinate frame for each quantity, distinguishing input sensor depth from depth refined by MonST3R.

The object smoothness score is computed in world coordinates using predicted poses, whereas the single-object trajectory case study is evaluated in camera coordinates without predicted poses. Their scale assumptions differ, so we report them separately.

We distinguish four depth maps: $D _ { \mathrm { s e n s o r } } ,$ the sensor measurement converted to meters; $D _ { \mathrm { i n i t } } .$ , the same map written into MonST3R; $D _ { \mathrm { o p t } }$ , the depth after window optimization; and $D _ { \mathrm { r g b } }$ , the depth predicted from RGB alone. Optimization changes $D _ { \mathrm { i n i t } }$ , so it must not be conflated with $D _ { \mathrm { o p t } }$ (Section 4). The 20-sequence hand study uses $D _ { \mathrm { s e n s o r } }$ directly for both anchoring and evaluation, independently of MonST3R. World-frame object smoothness combines sensor-depth points with predicted camera translations that may have a different scale; we consequently interpret it as a relative stability diagnostic.

Table 1: Provenance of the geometry behind each reported quantity. “Sensor depth” is the dataset’s depth map converted to meters and resampled to $3 2 0 \times 1 { \dot { 7 } } 6$ (nearest neighbor); it never passes through MonST3R. “Window-optimized” is the depth after the sliding-window alignment of Section 3.2.1.
<table><tr><td>Quantity</td><td>Depth source</td><td>Pose source</td><td>Frame</td></tr><tr><td>Hand scale anchoring, Eq. (11)</td><td>sensor depth, box median</td><td></td><td>camera</td></tr><tr><td>Hand–depth residual, Eq. (12)</td><td>the same map, nearest sample</td><td></td><td>camera</td></tr><tr><td>Object-trajectory smoothness, Eq. (13)</td><td>sensor depth, box median</td><td>predicted  $\mathbf { T } _ { t }$ </td><td>world</td></tr><tr><td>Object single-object case study</td><td>sensor depth, box center</td><td></td><td>camera</td></tr><tr><td>Camera ATE, main (RGB-only)</td><td>MonST3R predicted</td><td>MonST3R predicted</td><td>first-frame camera</td></tr><tr><td>Camera ATE, ablation</td><td>window-optimized (initialized)</td><td>MonST3R predicted</td><td>first-frame camera</td></tr></table>

Hand–depth residual. After metric-scale correction (Eq. (11)), each V-HPOT joint $\tilde { \mathbf { J } } _ { j }$ lies in metric camera space. We reproject it to the image plane using the intrinsics K and compare its depth with the depth map $\mathbf { D } _ { t } \left( \mathbf { D } _ { \mathrm { s e n s o r } } \right.$ in the 20-sequence study, $\mathbf { D } _ { \mathrm { r g b } }$ in the RGB-only variant):

$$
r _ { j } = \left| \tilde { z } _ { j } - \mathbf { D } _ { t } \Big ( \pi ( \tilde { \mathbf { J } } _ { j } ) \Big ) \ \right| ,\tag{12}
$$

where $\pi ( \cdot )$ denotes perspective projection and $\mathbf { D } _ { t }$ is sampled with nearest-neighbor interpolation at the MonST3R resolution. A small $r _ { j }$ indicates agreement between the estimated joint depth and the sampled surface depth at its projected location.

Object-trajectory smoothness. For each tracked object we back-project its 2D box center using $\mathbf { D } _ { t }$ and the per-frame pose $\mathbf { T } _ { t }$ to obtain a 3D box-center proxy $\mathbf { c } _ { t } .$ , and per frame compute the second difference,

$$
a _ { t } = \frac { \left\| \mathbf { c } _ { t + \Delta t } - 2 \mathbf { c } _ { t } + \mathbf { c } _ { t - \Delta t } \right\| } { \Delta t ^ { 2 } } , \quad \Delta t = 1 \mathrm { f r a m e } ,\tag{13}
$$

aggregated as the median over valid object-frame samples. Because egocentric camera motion would otherwise inject large spurious accelerations, this score provides a temporal-stability diagnostic affected by tracking, depth, and camera estimation; a large value on a static object can indicate a registration error, though a constant-bias trajectory can also produce a low score. At f frames per second the per-frame second difference corresponds to an acceleration of $f ^ { 2 } a _ { t }$ (for H2O’s 30 fps, 900 $a _ { t } )$ , so we report both the raw per-frame value and mm $\mathrm { / s ^ { 2 } }$ where relevant.

Hand–object proximity (defined, not evaluated). A minimum distance between fingertip joints and the backprojected object mask would test whether the hand and object streams agree during a grasp; we define it for completeness but do not evaluate it, so it is not part of the validated protocol of this paper.

Naive stacking vs. consistency scheme. Naive stacking combines model outputs in their native units, leaving V-HPOT in virtual-camera units and depth in meters. This unit mismatch makes the resulting distances uninterpretable. We report it only to illustrate the need for conversion; the meaningful control for per-hand anchoring is a single global scale, evaluated in Section 4.3.

## 3.5 Downstream Application: Structured Manipulation Captioning

We illustrate downstream use through structured manipulation captioning for robot policy learning. ASID-Captioner, a video MLLM based on Qwen2.5-Omni-3B [24] and trained on attribute-structured audiovisual instructions, receives the assembled RGB video and produces timestamped manipulation steps. To improve left/right attribution, we use hand motion to distinguish a holder hand that stabilizes the object from a mover hand that performs the action. On H2O, this attribution uses ground-truth hand poses, making the demonstration GT-assisted. Predicted timestamps are linearly rescaled to the video duration, and references to audio and hallucinated content are removed. Figure 4(a)–(d) shows the resulting captions alongside the geometry annotations. Captioning is a qualitative downstream application and is not quantitatively evaluated.

![](images/390d69d1893962174d3f1da0c168b0b2aa14b90767fe5201bd939a92a1616d0f.jpg)

![](images/76030258d30d66a44d990fdd505422109d8fee24e56bc2b644ab07786c43b226.jpg)  
Figure 5: Cross-modal consistency of the V-HPOT hand stream against the depth map on H2O h1/0. (a) Distribution of the per-joint hand–depth residual; the solid line is the median. (b) Median residual for a single global scale, our per-hand anchoring, and the reprojected ground-truth hands. The figure shows a representative sequence; across the 20 sequences the per-sequence median ranges from 64.2 to 119.7 mm, and the pooled medians are in Table 6. The residual is a consistency check (does the fused skeleton lie on the reconstructed surface?), not an independent accuracy metric

Prompt design: from scene description to manipulation steps. Generic video captions tend to emphasize scene appearance rather than individual manipulation steps. Our robot-oriented prompt specifies three fields: Hand (left/right identity, grasp type, and contact points), Object (identity, appearance, location, and state), and Action (e.g., reach, grasp, lift, rotate, peel, open, insert, or place). It requests separate descriptions of the two hands and a new step when the object state changes, while excluding background, camera, atmosphere, and audio descriptions. This format organizes captions around manipulation events for use alongside the 3D annotations.

## 4 Evaluation

## 4.1 Dataset Selection

We use H2O for quantitative evaluation and additional datasets for qualitative compatibility checks. The evaluation separates hand accuracy and consistency, camera trajectories, and object-stream case studies.

H2O (Hands and Objects in Egocentric action). H2O [3] records two-hand manipulation with a head-mounted RealSense camera at 30 fps and 1280×720 resolution. It provides synchronized RGB and 16-bit depth images (in millimeters), camera intrinsics and 6DoF poses, 21 keypoints per hand, MANO [25] parameters, and object poses. We evaluate hand accuracy and hand–depth consistency on 20 subject1\_ego sequences: h1/0–h1/7, h2/0–h2/7, and o1/0–o1/3, totaling 12,032 frames and 21,876 hand samples. These runs use sensor depth directly and evaluate hand registration independently of camera estimation and object lifting. Camera ATE is evaluated on three sequences (h1/0, k1/0, and o1/0; 1,233 frames), while object smoothness and the single-object case study use h1/0, the only sequence with full-pipeline object tracks. Table 3 summarizes the coverage. We report camera results with and without sensor-depth initialization, taking RGB-only as the main camera configuration, and provide an RGB-only hand–depth diagnostic on two sequences.

Compatibility datasets. ML-EgoDex (in-house; .mp4 plus .hdf5 intrinsics and hand/object keypoints; no sensor depth) and HOT3D [5] (Project Aria, 3D hand annotations and poses; the source of V-HPOT’s pre-training) are used only to check compatibility of the pipeline without sensor depth and of the hand module across cameras. We report no quantitative numbers on either.

Table 2 summarizes the available annotations and the role of each dataset. H2O supplies the joint depth, camera, hand, and object annotations needed for our quantitative study; ML-EgoDex and HOT3D support compatibility checks, and SceneEgo supplies the separate whole-body demonstration.

Table 2: Annotation coverage and role of the datasets. ✓ = ground truth available; — = not available. “Quant.” indicates whether the dataset yields quantitative results in this work.
<table><tr><td>Dataset</td><td>Depth</td><td>Cam. pose</td><td>Hand</td><td>Object</td><td>Body</td><td>Quant.</td></tr><tr><td>H2O [3]</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>ML-EgoDex (in-house)</td><td></td><td></td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>HOT3D [5]</td><td></td><td>√</td><td>√</td><td>vt</td><td></td><td></td></tr><tr><td>SceneEgo [6]</td><td></td><td></td><td></td><td></td><td>√</td><td>一</td></tr></table>

<sup>†</sup>HOT3D provides object pose and mesh annotations [5]; our compatibility check does not use them.

Table 3: Coverage of the quantitative results in Section 4. Hand accuracy and hand–depth consistency are measured on 20 sequences; camera ATE on three; object-trajectory smoothness requires the full tracking run and is validated on h1/0. RGB-only variants are listed separately.
<table><tr><td>Result</td><td>Sequences</td><td>Frames / samples</td></tr><tr><td>Hand pose (PA-MPJPE, scale-aligned)</td><td>20</td><td>12,032 frames / 21,876 samples</td></tr><tr><td>Hand-depth residual</td><td>20</td><td>12,032 frames / 429,896 joints</td></tr><tr><td>Camera ATE (Sim(3), SE(3))</td><td>3 (h1/0, k1/0, o1/0)</td><td>1,233 frames</td></tr><tr><td>Camera ATE, no injection (RGB-only)</td><td>3 (h1/0, k1/0, o1/0)</td><td>1,233 frames</td></tr><tr><td>Hand-depth residual, no injection</td><td>2 (h1/0, o1/0)</td><td></td></tr><tr><td>Object-trajectory smoothness / 1-object case study</td><td>h1/0</td><td>450 frames / 1,925 detections</td></tr><tr><td>Object detection/tracking/segmentation</td><td>h1/0 (stability)</td><td>450 frames / 9 tracks</td></tr><tr><td>Whole-body</td><td>fisheye sequences (qual.)</td><td></td></tr><tr><td>Captioning</td><td>h1/0 (qual.)</td><td></td></tr></table>

## 4.2 Component Evaluation and Sanity Checks

We first evaluate the pretrained components on the selected H2O subsets to establish a baseline for the cross-modal analysis. These results characterize the models within our pipeline; differences in datasets and protocols prevent direct claims of reproducing their published benchmarks. Table 4 summarizes camera and hand accuracy. Object-stream stability and whole-body compatibility are discussed separately.

Table 4: Per-module accuracy measured with the pipeline on H2O with off-the-shelf models, reported as a component sanity check rather than a new benchmark claim. Camera ATE is averaged over three sequences (h1/0, k1/0, o1/0; 1,233 frames); hand metrics over 20 sequences (h1/0–h1/7, h2/0–h2/7, o1/0–o1/3; 12,032 frames, 21,876 hand samples). Hand metrics are means over hand samples within a sequence, then averaged across sequences; ± denotes sample standard deviation across sequences, not across samples. V-HPOT is run frozen (no test-time optimization); the source paper’s 71% figure is for its TTO variant and is not reproduced here.
<table><tr><td>Module</td><td>Metric</td><td>Reported in paper</td><td>EgoLabel (H2O)</td></tr><tr><td>MonST3R</td><td>Camera ATE, Sim(3)</td><td>Competitive vs. DUSt3R/CasualSAM</td><td> $1 . 4 0 \pm 0 . 2 9 \mathrm { c m } ^ { \ S }$ </td></tr><tr><td>MonST3R</td><td>Camera ATE, SE(3)</td><td></td><td> $1 . 6 0 \pm 0 . 5 1 \mathrm { c m } ^ { \ S }$ </td></tr><tr><td>V-HPOT (frozen)</td><td>PA-MPJPE</td><td>cross-domain gains with TTO</td><td> $1 4 . 5 3 \pm 1 . 2 9 \mathrm { { m m } }$ </td></tr><tr><td>V-HPOT (frozen)</td><td>Scale-aligned MPJPE</td><td></td><td> $2 6 . 4 6 \pm 4 . 4 0 \mathrm { m m }$ </td></tr><tr><td>V-HPOT (frozen)</td><td>Absolute (metric) MPJPE</td><td></td><td> $1 1 9 . 1 \pm 3 1 . 3 \mathrm { m m } ^ { \ddagger }$ </td></tr></table>

<sup>‡</sup>Absolute MPJPE compares the anchored (metric) hands with GT without any Procrustes alignment; see Section 4. <sup>§</sup>RGB-only (no injected depth). The sensor-depth-injection ablation is worse $( 2 . 1 1 \pm 1 . 0 3 / \dot { 3 } . 8 1 \pm 0 . 6 2 $ cm) in a configuration where depth is not kept fixed during window-wise alignment (Section 4). Separately, a MANO-based regressor (HaMeR [11]) is reported at 18.27 mm PA-MPJPE on h1/0 under a different protocol; we do not treat it as a controlled comparison.

Camera pose. In the MonST3R paper [10], camera pose is evaluated on TUM-dynamics and ScanNet. Within EgoLabel, MonST3R runs in the two modes of Section 3 at 320×176, window 30, overlap 1/3, 300 iterations; Table 5 reports both on the three H2O ATE sequences (h1/0, k1/0, o1/0; 1,233 frames), paired on the same frames and computed with a common evaluation implementation using similarity or rigid alignment of camera translations. We take the RGB-only mode as the main camera result: 1.40 ± 0.29 cm (Sim(3)) and 1.60 ± 0.51 cm (SE(3)), with fitted scales 0.85/1.34/1.78. Since H2O’s poses are metric, the SE(3) value is the stricter measure, and the small gap indicates a limited effect of scale alignment on ATE for these sequences; the fitted scales still vary across sequences, so

RGB-only estimation yields lower trajectory error than the initialization ablation without establishing reliable metric geometry. For reference, the unaligned trajectory differs from GT by 21.54 cm on h1/0, an unaligned discrepancy rather than an aligned ATE; Figure 6 shows the injected trajectory and its per-frame error for k1/0.

Depth changes during sliding-window optimization. The sensor-assisted mode writes sensor depth with requires\_grad=False (Eq. (7)), but the sliding-window loader re-creates depth and focal length as optimizable parameters. The sensor map therefore serves as an initialization: on h1/0, 96% of pixels differ from the input by more than 1 cm after optimization, with a median relative change of 25%. The fitted geometry scales range from 0.17 to 0.45, and camera ATE is worse than RGB-only on all three sequences $( 2 . 1 1 \pm 1 . 0 \dot { 3 } / 3 . 8 \dot { 1 } \pm 0 . 6 2 $ cm for Sim(3) / SE(3)). Initializing focal length from dataset intrinsics does not resolve the discrepancy. We retain this configuration as an ablation. Hand–depth residuals and the single-object case study use sensor depth directly in camera coordinates and are unaffected by this optimization; world-frame object smoothness depends on the predicted poses and inherits their scale limitations (Table 1).

Table 5: Camera ATE (cm) on the three H2O ATE sequences, paired on the same sequences and frame sets and computed with the same evaluation script (scripts/eval\_ate.py), using similarity or rigid alignment of camera translations. $^ { \mathrm { \bullet } } \mathrm { I n j . } ^ { \mathrm { \bullet } }$ is the sensor-depth initialization (the released sensor-assisted mode) with the focal either optimized (free) or initialized from the dataset intrinsics (focal-init, not pinned); “RGB-only” uses MonST3R’s predicted depth and is the main camera result. Fitted Sim(3) scale in parentheses.
<table><tr><td>Seq.</td><td>RGB Sim(3) (scale)</td><td>RGB SE(3)</td><td>inj. free Sim(3) (scale)</td><td>inj. SE(3)</td><td>inj. focal-init Sim(3)</td></tr><tr><td>h1/0</td><td>1.08 (0.85)</td><td>1.09</td><td>1.29 (0.17)</td><td>4.00</td><td>1.34</td></tr><tr><td>k1/0</td><td>1.51 (1.34)</td><td>1.62</td><td>1.76 (0.45)</td><td>3.12</td><td>1.76</td></tr><tr><td>o1/0</td><td>1.62 (1.78)</td><td>2.10</td><td>3.27 (0.28)</td><td>4.32</td><td>3.33</td></tr><tr><td>mean</td><td> $1 . 4 0 \pm 0 . 2 9$ </td><td> $1 . 6 0 \pm 0 . 5 1$ </td><td> $2 . 1 1 \pm 1 . 0 3$ </td><td> $3 . 8 1 \pm 0 . 6 2$ </td><td> $2 . 1 4 \pm 1 . 0 5$ </td></tr></table>

![](images/f82a548d39ce8064f477a7d64b0373bcf6c18ec95644360d5f1e24b4b3de9fe9.jpg)

![](images/225f72c1a7130fdc1aaec42f70c477de26bb2f02ef093b5e51e802894a07981d.jpg)  
Figure 6: Camera trajectory on H2O k1/0 for the sensor-depth-injection ablation (the RGB-only trajectory, which we take as the main camera result, is summarized in Table 5), in the first-frame camera frame. (a) Top view: ground-truth (gray) and MonST3R trajectory after 7-DoF Sim(3) alignment (blue). (b) Per-frame ATE (blue) with the mean as a dashed line. The injected run recovers the trajectory shape well, with the residual global-scale error left by the injection acting only as an initialization rather than a constraint (recomputed with the released estimator: Sim(3) ATE 1.76 cm, fitted scale 0.45, strict SE(3) 3.12 cm).

Hand pose. Running the pretrained V-HPOTfrozen (no TTO, matching the release) on all 20 H2O sequences (12,032 frames, 21,876 samples) gives a PA-MPJPE of 14.53 ± 1.29 mm and a scale-aligned MPJPE of 26.46 ± 4.40 mm against GT; on h1/0 the PA-MPJPE is 14.13 mm and the per-frame curve is stable (Figure 7). Because the anchor is metric we also report the absolute (unaligned) MPJPE: 63.7 mm on h1/0 and $1 1 9 . 1 \pm 3 1 . 3$ mm over the 20 sequences (pooled 117.9), with a Procrustes scale of 0.73 $( 0 . 7 1 \pm 0 . 0 7 ) $ —the anchored skeleton is systematically about 37% too large, the cost of the box-median anchor. The aligned metrics characterize hand configuration, while absolute MPJPE exposes the scale and localization errors that remain after anchoring. All results use the pretrained checkpoint without target-domain training.

![](images/e67bfc7189624a2e350b060319ae0caff1398b35eaccd54598eba558c16114c3.jpg)

![](images/48b3b3872e945fbc7810298c72fc2cb13647115b8fd074eb483989e8e913201f.jpg)  
Figure 7: V-HPOT hand accuracy on H2O h1/0. (a) Summary bars: scale-invariant PA-MPJPE (14.13 mm), scalealigned MPJPE (22.0 mm), and absolute (metric) MPJPE (63.7 mm) over 900 hand samples. (b) Per-frame PA-MPJPE over time (light) with a 15-frame smoothed curve and the sequence mean (dashed). The per-frame PA-MPJPE is broadly stable over the sequence. 20 sequences: 14.53±1.29 PA-MPJPE, 26.46±4.40 scale-aligned (mean±std).

Object and body streams. The detection/tracking/segmentation stream is built from LocateAnything-3B [15], GMC-BoT-SORT [16] and SAM 2.1 (Hiera-L) [17]. H2O provides no instance masks for our categories, so we do not re-benchmark it on H2O and instead rely on the checkpoints’ published benchmarks (LocateAnything-3B on COCO/LVIS in its model card; we use BoT-SORT with its ReID head disabled; SAM 2.1’s 79.5 J&F on SA-V test). On h1/0 the composed stream yields nine object tracks, seven persisting more than 30 frames (median coverage 0.75); the three named instances (bottle, cup, box) are detected in 450, 450 and 420 frames, while open-vocabulary “object” tracks are more fragmented (0.32–0.75), the expected weak point of zero-shot detection on small or occluded instances. Whole-body pose is qualitative and domain-separate (Section 3.2.4).

## 4.3 Cross-Modal Consistency Validation

We next examine agreement across the registered annotation streams. Figure 5 illustrates the hand–depth residual on a representative sequence, and Table 6 summarizes the 20-sequence hand study and the object smoothness case study. The constant-scale control isolates the effect of per-hand anchoring beyond basic unit conversion.

Table 6: Cross-modal consistency on H2O. Hand–depth residual: pooled median over 20 sequences (429,896 joints), with the per-sequence mean±std of the medians below it. Object-trajectory smoothness: median on $\mathrm { h } 1 / 0 . \ { \overset {  } { \sim } } \mathrm { N a i v e } ^ { \mathrm { , } }$ applies no unit conversion; “Const-scale” applies one global scale; “Per-hand” is our anchoring (Eq. (11)); “GT” reprojects the dataset’s ground-truth hands onto the same depth map as a GT reference residual. Aggregation differs by row: the residual is a pooled median over all joint samples, with the distribution of per-sequence medians underneath, and the smoothness is a median over object-frame samples at H2O’s fixed 30 fps. Lower is better.
<table><tr><td>Measure</td><td>Naive</td><td>Const-scale</td><td>Per-hand</td><td>GT ref.</td></tr><tr><td>Hand–depth residual, median (20 seq.)</td><td>unusableª</td><td> $1 0 1 . 1 \mathrm { m m }$ </td><td> $8 4 . 6 \mathrm { m m }$ </td><td> $2 0 . 5 \mathrm { m m }$ </td></tr><tr><td>per-sequence median</td><td></td><td> $1 1 1 . 7 \pm 4 3 . 9$ </td><td> $8 6 . 4 \pm 1 6 . 9$ </td><td> $2 2 . 2 \pm 6 . 0$ </td></tr><tr><td>Object-trajectory smoothness, median (h1/0)</td><td></td><td></td><td> $3 . 6 7 \mathrm { m m / f r a m e } ^ { 2 }$ </td><td></td></tr></table>

<sup>a</sup>V-HPOT’s raw output is in virtual-camera units (residual ≈ $7 . 4 \times 1 0 ^ { 5 }$ mm); this is a unit mismatch, not a meaningful distance.

Hand–depth residual. Across 20 sequences, the pooled median hand–depth residual is 84.6 mm (per-sequence $8 6 . 4 \pm 1 6 . 9$ , range 64.2–119.7). The distribution has a heavy tail: $p _ { 7 5 } = 1 3 6 . 1 , p _ { 9 0 } = 1 8 5 . 8$ , and $p _ { 9 9 } = 3 4 4 . 3$ mm, with 42.0% of joints exceeding 100 mm, consistent with the occlusion failures discussed in Section 4.5. Reprojected ground-truth hands yield a median of 20.5 mm, approximately one quarter of the predicted residual. This reference is not a strict lower bound: annotated joints lie inside the hand, whereas depth measures the visible surface, and occlusion, projection error, and invalid depth affect both measurements.

A wrist-excluded diagnostic anchors the hand using wrist-pixel depth and evaluates the remaining 20 joints. It yields 64.6 mm (per-sequence $6 4 . 8 \pm 1 1 . 2 )$ , compared with 90.0 mm for box anchoring on the same joints. This comparison characterizes sensitivity to the anchoring region, but it does not remove dependence on the calibration depth: at 320×176 resolution, different joints may sample the same nearest-neighbor cell. Moreover, a camera-frame residual cannot detect errors confined to the camera-to-world transformation.

The RGB-only diagnostic shows the same ordering between per-hand and constant-scale anchoring: 9.8% versus 12.7% on h1/0, and 11.0% versus 15.5% on o1/0 (Table 7). These residuals are normalized by wrist depth because MonST3R’s RGB-only depth is defined only up to scale.

Table 7: Sensor-depth-free (RGB-only) hand–depth residual on the two sequences that have V-HPOT tracks. MonST3R’s predicted depth is defined only up to a scale, so the residual is normalized by the wrist depth (dimensionless); “per-hand” uses the per-frame box-median ratio and “const” a single per-sequence ratio. These values must not be compared with the millimeter residuals of Table 6.
<table><tr><td>Sequence</td><td>joints</td><td>per-hand</td><td>const-scale</td></tr><tr><td>h1/0</td><td>18,900</td><td>9.8%</td><td>12.7%</td></tr><tr><td>01/0</td><td>18,144</td><td>11.0%</td><td>15.5%</td></tr></table>

Value of per-hand anchoring. On h1/0, virtual-camera conversion already yields a near-metric scale (fitted global scale ≈ 1.002), and per-hand anchoring gives nearly the same residual as a single global scale (66.6 versus 66.5 mm). Across 20 sequences, however, the pooled median decreases from 101.1 to 84.6 mm, a 16.3% reduction. The corresponding per-sequence medians are $1 1 1 . 7 \pm 4 3 . 9$ and $8 6 . 4 \pm 1 6 . 9$ mm: anchoring reduces their mean by 23% and improves 17 of 20 sequences.

To examine sensitivity to depth-scale drift, we apply a smooth per-frame factor g<sub>t</sub> to the depth map, $D _ { t } ^ { \prime } = g _ { t } D _ { t } .$ Before temporal smoothing, whole-hand anchoring propagates this perturbation as $\bar { J _ { t } ^ { \prime } } = g _ { t } J _ { t }$ , leaving pixel projections unchanged and scaling the residual as $\boldsymbol { r } _ { t } ^ { \prime } = \boldsymbol { g } _ { t } \boldsymbol { r } _ { t }$ . Depth normalization can therefore cancel the common factor even when the metric scale is wrong; temporal smoothing further couples neighboring frames. The experiment measures how the two estimators respond to a shared perturbation, rather than whether they recover true scale. With a 30% perturbation amplitude on ten sequences, the constant-scale residual increases by 19% (105.5 → 125.5 mm), while the per-hand residual changes little (86.9 → 86.4 mm).

We also restrict the data used to estimate the global scale. Fitting it on the first half of each sequence yields a pooled median of 107.9 mm (per-sequence $1 2 3 . 7 \pm \mathrm { \bar { 6 } 2 . 8 ) }$ ; fitting on ten sequences and applying it to the other ten yields 106.9 mm. Both exceed the frame-local anchor’s residual (85.0 mm; per-sequence $8 7 . 0 \pm 1 7 . 1 )$ . These controls compare estimation ranges: the per-hand anchor still uses the current frame’s depth, so the holdouts do not provide an independent validation of that anchor.

Object-trajectory smoothness. On h1/0, median object-trajectory acceleration is 3.67 $\mathrm { m m / f r a m e ^ { 2 } }$ , equivalent to $3 . 3 \times 1 0 ^ { 3 } \mathrm { m m / s ^ { 2 } }$ at 30 fps. The static bottle and cup yield 4.68 and $1 . 2 9 \mathrm { m m / f r a m e } ^ { 2 }$ , respectively. The manipulated box moves 142 mm while maintaining a median acceleration of $2 . 1 8 \mathrm { { m m } / \mathrm { { f r a m e } ^ { 2 } } }$ . These values describe temporal stability, subject to the world-frame scale limitations in Table 1. A low acceleration may also result from a stable tracking bias or excessive smoothing, so it does not establish object accuracy. Evaluation against object-pose ground truth remains future work.

GT-assisted single-object trajectory case study. The single annotated object in h1/0 is present in all 450 frames. We fix its track association using only the first frame: the ground-truth object center is projected with the camera intrinsics, and the track whose box contains that projection is selected. Of the five tracks present at frame 0, exactly one satisfies this condition—the box track, with 420 common frames. This is a GT-assisted association, but it avoids selecting tracks independently by ground-truth proximity at every frame.

In camera coordinates, the absolute centroid-to-center error has a median of 54.4 mm and a mean of 57.7 mm. The error is dominated by the offset between the back-projected visible surface and the annotated object origin, so it is not an object-pose metric. Estimating a fixed offset on the first half of the frames and evaluating on the second yields a held-out trajectory-shape error of 12.7 mm (mean 14.0; the whole-sequence-offset result is 11.9). Total displacement is 105.9 mm, compared with 111.0 mm for ground truth. The result supports motion consistency for this object and sequence. Unlike world-frame smoothness, this camera-frame comparison does not use predicted poses (Table 1); ground-truth-free association and multi-object evaluation remain open tasks.

## 4.4 Cost Analysis

MonST3R dominates the computational cost. On a single NVIDIA GPU with 24 GB of memory, it processes a 351-frame sequence in approximately 10 minutes and peaks near 23 GB of GPU memory. V-HPOT processes 450 frames in approximately 14 seconds; object detection, tracking, segmentation, and bookkeeping add a few minutes. End-to-end processing takes roughly 10–15 minutes per sequence, with observed per-frame times of approximately 1.7–2.6 seconds across the 351- and 450-frame runs. These timings come from separate runs and are not pooled into a single throughput estimate. The pipeline requires no per-frame human supervision. Reducing the MonST3R resolution (320→256) or iteration count could lower runtime, although the accuracy trade-off has not been evaluated. We do not estimate manual time savings because a comparable annotation rate for this combination of 3D modalities is unavailable.

## 4.5 Failure Analysis

We identify four failure modes in the H2O runs and the additional compatibility checks. (i) Hand–depth disagreement under occlusion. Across the 20 sequences, 42.2 ± 9.4% of per-joint samples exhibit a hand–depth residual above 100 mm (h1/0: 29.1%; worst sequence: 61.2%). These concentrate where the hand is partially occluded by an object so that the sampled sensor depth falls on the occluder rather than the hand, or where the hand is in fast motion and the sensor depth is noisy or missing. Scale-invariant PA-MPJPE can remain low when the relative joint configuration is preserved despite errors in absolute scale or position. (ii) Scale drift without sensor depth. On recordings without sensor depth, such as ML-EgoDex, MonST3R’s scale ambiguity propagates to the depth map and the anchored hands. The response analysis in Section 4.3 shows that local anchoring follows changes in depth scale. A stable normalized residual can therefore coexist with incorrect metric geometry. (iii) Reflective and specular surfaces. Specular highlights on bottles and metallic objects produce depth spikes that bias both object back-projection and the object-trajectory smoothness score. (iv) Missed hands under extreme occlusion. V-HPOT predicts a per-hand presence score; on h1/0 both hands are detected in all 450 frames (900 of 900 slots), and the rare misses in our broader trials correspond to the hand being fully outside the field of view or completely hidden behind an object. The presence score flags these cases, and the corresponding predictions are omitted.

## 4.6 Qualitative Results

Figures 4 and 2 illustrate the integrated viewer and the object tracking and segmentation outputs. The viewer places predicted hands, object points, camera trajectories, and ground-truth references under distinct entity paths, allowing frame-by-frame inspection of agreement and failure. Figure 8 shows a representative 3D scene with anchored hands, lifted object trajectories, and camera trajectories. This visualization supports qualitative inspection subject to the scale assumptions in Section 3.4; the predicted camera trajectory shown in the figure is Sim(3)-aligned to ground truth. Structured captions provide an additional qualitative downstream artifact (Section 3.5).

## 4.7 Limitations

The evaluation establishes component accuracy and several consistency diagnostics, with the following limitations. Calibration dependence and metric scale. The main hand and object studies use sensor depth. Because hand anchoring and residual evaluation share that depth, the residual cannot independently establish registration accuracy. Wrist-excluded and holdout analyses characterize this dependence without eliminating it. RGB-only depth remains scale-ambiguous, and the implemented sensor-depth initialization does not preserve depth during window optimization. In particular, combining metric sensor-depth points with camera translations at a different scale limits the interpretation of world-frame object trajectories. Independent geometric validation is needed to establish metric accuracy. Coverage. Quantitative evaluation is limited to H2O and one subject. Hand results cover 20 sequences across three action groups, camera ATE covers three sequences, and the RGB-only hand diagnostic covers two. Object results comprise a single-sequence stability analysis and a GT-assisted single-object trajectory study; the 20-sequence hand runs contain no object detections. Detection and segmentation lack mask-based evaluation, and whole-body pose and captioning are qualitative demonstrations. Broader subject, scene, and modality coverage is needed to assess generalization. Statistics and baselines. Pooled medians, means of per-sequence statistics, and per-frame curves summarize different distributions and should not be used interchangeably. Our controls are internal; no competing end-to-end auto-labeling system is evaluated. The cited HaMeR result uses a different protocol and provides context only.

![](images/9e3d51d92262d209d4f9240605602ea06ef7034d24ccb25591b4e1c60e5d5f9d.jpg)

![](images/bd85082403cbaf73805877bf10c8aa3e99101fec3b330d9daec86e2a0a43de6c.jpg)  
Figure 8: Fused 3D scene on H2O (frame 225), in the first-frame camera frame. Gray: ground-truth camera trajectory; blue: predicted camera trajectory (Sim(3)-aligned); green: the V-HPOT hand skeleton after metric-scale anchoring; colored lines: 3D trajectories of the tracked objects lifted through the depth map and the camera pose (start points marked). The independently estimated hand, object, and camera streams are drawn in one coordinate system for inspection.

## 5 Conclusion

We presented EgoLabel, a zero-shot pipeline that integrates pretrained geometry, object, and hand models through explicit coordinate and scale conversions. Its contribution is an auditable registration procedure accompanied by diagnostics that expose cross-modal disagreement and distinguish relative consistency from metric accuracy. On 20 sensor-assisted H2O sequences, per-hand anchoring reduces the mean of per-sequence median hand–depth residuals by 23% relative to a single global scale. The pooled residual remains 84.6 mm against a 20.5 mm ground-truth reference, and the gap between PA-MPJPE (14.53 ± 1.29 mm) and absolute MPJPE (119.1 ± 31.3 mm) highlights the remaining scale error. On three RGB-only sequences, camera ATE is 1.40 ± 0.29 cm with Sim(3) alignment and 1.60 ± 0.51 cm with SE(3) alignment. The sensor-depth initialization ablation further shows that implementation-level assumptions about fixed depth must be checked explicitly. Future work will prioritize independent validation of hand anchoring, consistent metric geometry, broader RGB-only evaluation, and quantitative assessment of object and body annotations. Code, configurations, and evaluation scripts will be released upon publication; a one-command reproduction pipeline is in preparation.

## References

[1] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Dima Damen, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[2] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Jian Ma, et al. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal ofComputer Vision (IJCV), 130:33–55, 2022.

[3] Taein Kwon, Bugra Tekin, Jan Stühmer, Federica Bogo, and Marc Pollefeys. H2o: Two hands manipulating objects for first person interaction recognition. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[4] Takehiko Ohkawa, Kun He, Fadime Sener, Tomas Hodan, Luan Tran, and Cem Keskin. Assemblyhands: Towards egocentric activity understanding via 3d hand pose estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12999–13008, 2023.

[5] Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, and Tomas Hodan. Hot3d: Hand and object tracking in 3d from egocentric multi-view videos. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7061–7071, 2025.

[6] Jian Wang, Diogo Luvizon, Weipeng Xu, Lingjie Liu, Kripasindhu Sarkar, and Christian Theobalt. Scene-aware egocentric 3d human pose estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13031–13040, 2023.

[7] Kailin Wang, Haoxiang Jie, Yaoyuan Yan, Jiacheng Zhou, and Zhiyou Heng. Ev-wm: Event-verified world models for long-horizon robotic manipulation. arXiv preprint arXiv:2606.13053, 2026.

[8] Haoxiang Jie, Yaoyuan Yan, Xiangyu Wei, Kailin Wang, Hongjie Yan, Zhiyou Heng, and Daocheng Chen. Omnivla-rl: A vision-language-action model with spatial understanding and online rl. arXiv preprint arXiv:2604.17706, 2026.

[9] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jérôme Revaud. Dust3r: Geometric 3d vision made easy. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[10] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. In International Conference on Learning Representations (ICLR), 2025.

[11] Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing hands in 3d with transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[12] Wiktor Mucha, Michael Wray, and Martin Kampel. Towards egocentric 3d hand pose estimation in unseen domains. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026.

[13] Jian Wang, Zhe Cao, Diogo Luvizon, Lingjie Liu, Kripasindhu Sarkar, Danhang Tang, Thabo Beeler, and Christian Theobalt. Egocentric whole-body motion capture with fisheyevit and diffusion-based motion refinement. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 777–787, 2024.

[14] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianfeng Gao, Jianwei Yang, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In Computer Vision–ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XV, pages 350–367. Springer, 2024.

[15] Shihao Wang, Shilong Liu, Yuanguo Kuang, Xinyu Wei, Yangzhou Liu, Zhiqi Li, Yunze Man, Guo Chen, Andrew Tao, Guilin Liu, Jan Kautz, Lei Zhang, and Zhiding Yu. Locateanything: Fast and high-quality vision-language grounding with parallel box decoding. arXiv preprint arXiv:2605.27365, 2026. NVIDIA; model checkpoint LocateAnything-3B.

[16] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Bot-sort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.

[17] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junxiao Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. We use the SAM 2.1 Hiera-L checkpoint (sam2.1\_hiera\_large); the 79.5 J&F figure is from the SAM 2.1 model table.

[18] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In Advances in Neural Information Processing Systems 28 (NeurIPS 2015), pages 91–99, 2015.

[19] Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft. Simple online and realtime tracking. In 2016 IEEE International Conference on Image Processing (ICIP), pages 2480–2484. IEEE, 2016.

[20] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXIV, pages 383–400. Springer, 2022.

[21] Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Romain Brégier, Yohann Cabon, Vaibhav Arora, Leonid Antsfeld, Boris Chidlovskii, Gabriela Csurka, and Jérôme Revaud. Croco: Self-supervised pre-training for 3d vision tasks by cross-view completion. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[22] Bruce D Lucas and Takeo Kanade. An iterative image registration technique with an application to stereo vision. In Proceedings ofthe 7th International Joint Conference on Artificial Intelligence (IJCAI), pages 674–679, 1981.

[23] Nicolai Wojke, Alex Bewley, and Dietrich Paulus. Simple online and realtime tracking with a deep association metric. In 2017 IEEE International Conference on Image Processing (ICIP), pages 3645–3649. IEEE, 2017.

[24] Qwen Team. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.

[25] Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics (SIGGRAPH Asia), 2017.