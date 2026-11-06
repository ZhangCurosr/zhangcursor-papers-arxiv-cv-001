# Fast and Accurate Hand Estimation in Pixels, Interaction and World Space

Jinkun Cao<sup>\*</sup>

jinkuncao@gmail.com

## Abstract

Monocular hand mesh estimators commonly predict rig parameters and project the mesh to obtain 2D keypoints. We find that directly predicted 3D joints agree better with 2D predictions, and fitting a mesh to predicted points reduces mesh error. We present FAH (Fast and Accurate Hand Estimation), afamily ofmodels built on SAM 3D Body’s hand branch through fine-tuning and feature distillation. The models predict corresponding 2D and 3D joints and surface points, fit MANO or MHR-Hand, and optionally estimate anatomical parts and contact without object geometry at inference. Causal refinement reduces acceleration error, and camera estimation places the hands in world space. Experiments show improvements in image-space alignment, hand geometry, contact estimation and world-space tracking over the evaluated baselines. The default ViT-L model runs at 65fps with meshfitting on an RTX 4070 Ti SUPER, or 49fps with parts and contact enabled. The smallest ViT-S model reaches 125fps without meshfitting, parts or contact. Code: https://github.com/noahcao/FAH.

## 1. Introduction

Augmented reality and robotic data pipelines from human video [1–3] require hand estimates that align with visible fingers, recover 3D geometry and contact, remain stable over time and run in real time. Most mesh estimators regress MANO [4] or MHR [5] parameters and project the reconstructed mesh to obtain 2D keypoints [6–11]. Errors in articulation, shape and camera placement propagate to the 2D landmarks; rig constraints and linear blend skinning can further limit pixel alignment.

Our controlled comparison shows that the output representation contributes to this error: within one network, directly predicted 3D points agree better with the 2D predictions, and fitting a mesh to them is more accurate than predicting MANO parameters (Sec. 4.4).

FAH builds on SAM 3D Body’s hand branch [10], replacing its decoder and rig head with a heatmap head for corresponding 2D and 3D joints and surface points. A separate fit to MANO, MHR-Hand or both regularizes the 3D surface while preserving the direct 2D predictions. SAM masks [12, 13] supervise the surface, part labels and detector masks, which also guide fingertip alignment at inference. Optional heads predict anatomical parts and contact without object geometry. Causal refinement extends the model to video, and camera estimation places the hands in world space. Training also requires consistent labels: kinematic MANO joints, mesh-regressed joints, ARKit skeletons and fingertip vertices define different target locations [14–18]. We harmonize these conventions before training. To present FAH, we make three major contributions:

Point prediction then mesh fitting. We predict corresponding 2D and 3D hand joints and surface points, then fit a mesh separately. In a controlled comparison, this improves mesh accuracy over direct rig-parameter prediction.

Contact, video and world-space estimation. Predicted hand geometry improves contact estimation without object geometry at inference. Causal refinement reduces pose acceleration error and improves contact timing; camera estimation supports causal and offline world-space tracking.

Hand estimation at interactive rates. The model family runs at 27–101 fps with mesh fitting on an RTX 4070 Ti SUPER. The smallest model also runs above 30 fps on an M3 Pro MacBook, supporting interactive applications and large-scale video processing.

## 2. Related Work

Hand mesh recovery. Single-image methods regress MANO [4] parameters from a hand crop: HaMeR [6] trains a large ViT on a mixture of 2D and 3D data, WiLoR [7] adds a detector and in-the-wild data, Hamba [8] uses graphguided Mamba blocks, and AnyHand [9] adds synthetic data. SAM 3D Body [10] jointly recovers the body and hands using a promptable decoder and the MHR rig [5]. We build on its hand branch. Other methods regress mesh vertices directly [19, 20], while feedback on image features [21] and silhouette supervision [22] improve meshimage alignment. FAH predicts corresponding 2D and 3D surface points before fitting a mesh.

![](images/3399a6f10c1689964d7f86d2a8e8fec0d345a07e20a7c0dbd6e7ba6549267a46.jpg)  
Figure 1. Five outputs of FAH-H (FAH with ViT-H backbone) on ten scenes. Left to right: 2D keypoints, visible dense points at predicted 2D locations, a MANO mesh fitted to the 3D points, optional part labels within the detector mask, and contact on the fitted mesh (gray to violet, probability 0–1; contact-side view). Outlines: right hand blue, left orange. Parts: palm gray; thumb to pinky red, orange, green, blue and magenta; darker shades are more distal. Image credits: Sec. D.

Egocentric video and world space. Egocentric hand reconstruction captures manipulation, while world-space estimates recover trajectories beyond the moving camera frame. HaWoR [11] combines a temporal model with masked DROID-SLAM [23] and metric depth [24] to recover world-frame trajectories. ViDiHand [25] uses a video diffusion model for detector-free clip inference and introduces coverage-aware metrics. Temporal filters [26, 27] and video hand models [28] address motion stability. FAH adds causal pose refinement and camera estimation to a fast perframe model while preserving its direct 2D stream.

Hand contact and interaction. Hand-object contact identifies where the hand touches an object, providing cues about manipulation and physical interaction. HACO [29] estimates dense per-vertex hand contact from a single image, following body-scene contact estimators [30, 31]. Contact4D [32] provides synchronized whole-body and hand motion with ground-truth finger contact during dexterous operations. Distance rules on reconstructed hand and object meshes [33] require object shape and pose. FAH predicts contact from hand features and geometry. Joint Diffusion [34] models hands and objects jointly for grasp generation, while Omnigrasp [35] learns simulated humanoid control for grasping and transporting diverse objects.

## 3. Method

Figure 2 shows the prediction paths. FAH builds on existing architectures, using SAM [12, 13] masks and SAM 3D Body [10] supervision. Separate training stages exploit

datasets with different annotations (Sec. A.1).

## 3.1. Hand detection

A one-stage YOLOv8m segmentation network [36] predicts hand boxes, handedness and instance masks at a 640 px input in FP16. The detector trains on about 96k qualitychecked images, including two-hand frames from HOT3D and OakInk2 training sequences (Sec. A.7). Target boxes enclose both SAM 3 [13] hand masks and visible keypoints. The pose network re-scores each box with presence and handedness-consistency logits and re-runs doubtful sides mirrored; identity NMS suppresses duplicate hypotheses using predicted wrist distances, and hands touching the crop border are re-cropped once.

## 3.2. Pose network

Teacher and student. We extract SAM 3D Body’s hand branch (DINOv3-H+ backbone, promptable decoder and MHR head) and fine-tune it with the benchmark sequences held out. Feature distillation initializes DINOv3 ViT-S/16, ViT-B/16 and ViT-L/16 students, which then train with the supervision below. FAH-L uses ViT-L/16 on 320 px crops; FAH-S and FAH-B use the smaller students, and FAH-H retains the teacher’s ViT-H+/16 at 448 px.

Decoder-free dense head. The pose network uses a heatmap head directly on the encoder tokens, without the teacher’s promptable decoder or rig head. It predicts 21 MANO-convention joints, two points inside each of the 20 bones and 400 farthest-point samples of the 778 MANO vertices, each with crop-normalized 2D (soft-argmax) and root-relative 3D coordinates. A root-depth output predicts $\log ( z s / f )$ for a crop of size s and focal length $f ,$ and the x, y translation follows in closed form from the 2D and 3D points at that depth. The full 778-vertex surface is reconstructed from the 400 points through the lowest 200 eigenfunctions of the template’s cotangent Laplacian.

![](images/503467803e54b63f757b86fa410ca98d14b61200770f2d51657a45c14cbe3b00.jpg)  
Figure 2. FAH on a training-disjoint HOT3D frame. The detector supplies boxes, handedness, masks and crops. The heatmap head predicts 2D and root-relative 3D joints and surface points (inset). Dashed arrows: MHR-Hand and MANO fitting; inset: side view, MANO outlined. The part head labels mask pixels; contact is shown on fitted MANO (gray to violet, probability 0–1; inset: palm side). Bottom: causal joint refinement, frames four apart with thumb/index fingertip trails, followed by world-space placement (camera gray, wrists colored). Blue: right hand; orange: left.

Supervision. Joints and bone points use ground-truth labels where available, with teacher targets on sources lacking 3D labels. Surface points use ground-truth meshes or SAM 2.1 pseudo-masks with a weak teacher-mesh prior. A reprojection-consistency loss supervises projected 3D points on 2D-labeled data, and incorrectly mirrored crops train the verification logits. Let $\hat { \mathbf { u } } _ { i }$ and $\hat { \mathbf { X } } _ { i }$ denote cropnormalized 2D and root-relative 3D predictions, with targets $\mathbf { u } _ { i } ^ { * }$ and $\mathbf { X } _ { i } ^ { * }$ . The point-supervision terms have the form

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { p t s } } = \displaystyle \sum _ { i } a _ { i } ^ { 2 D } \rho _ { 0 . 0 2 } \bigl ( \hat { \mathbf { u } } _ { i } - \mathbf { u } _ { i } ^ { * } \bigr ) } & { } \\ { + \lambda _ { 3 D } \displaystyle \sum _ { i } a _ { i } ^ { 3 D } \rho _ { 0 . 0 1 } \bigl ( \hat { \mathbf { X } } _ { i } - \mathbf { X } _ { i } ^ { * } \bigr ) . } \end{array}\tag{1}
$$

ρ sums coordinate-wise smooth-L1 penalties; 3D coordinates are in meters. Weights $a _ { i } ^ { 2 D } , a _ { i } ^ { 3 D }$ encode validity, confidence and source normalization. Ground-truth and teacher terms are accumulated separately; joint and bone losses use $\lambda _ { 3 D } ~ = ~ 3$ and teacher weight 0.5. Root-depth, visibility, mask and verification losses complete the supervision (Sec. A.3). For reprojection, let $\pi _ { K , A }$ project camera-space points through intrinsics $K$ and the crop transform $A ,$ returning crop-normalized coordinates. We minimize

$$
\mathcal { L } _ { \mathrm { r e p } } = \sum _ { i } r _ { i } \rho _ { 0 . 0 2 } \biggl ( \pi _ { K , A } ( \hat { \mathbf { X } } _ { i } + \bar { \mathbf { t } } ) - \bar { \mathbf { u } } _ { i } \biggr ) .\tag{2}
$$

Joint targets $\bar { \mathbf { u } } _ { i }$ use 2D labels where available and detached direct predictions otherwise. Translation <sup>¯</sup>t is fitted at ground-truth or teacher depth and held fixed during differentiation. Visible surface targets are detached direct 2D predictions; weights $r _ { i }$ mask invalid targets and downweight unlabeled joints.

Label conventions. We harmonize 3D sources to kinematic MANO joints and correct per-dataset 2D offsets. For EgoDex’s ARKit skeleton [18], the conversion uses a reference model trained without EgoDex (Sec. A.4).

## 3.3. Mesh fitter

The mesh stage fits MANO [4], MHR-Hand (the hand rig of MHR [5], as used by SAM 3D Body), or both, returning parameters and a mesh for each. MANO fitting has three steps. First, Horn’s quaternion method [37] aligns each template part to the corresponding predicted points. A learned fitter then predicts MANO pose and shape from this initialization and the points. One Gauss–Newton step refines the parameters against the predicted geometry, expressed relative to the predicted wrist so that the central-difference Jacobian is stable; it is evaluated in one batched forward pass. The fit runs as one CUDA graph. MANO-convention fingertips can stop short of the visible finger end. Where the detector mask resolves a fingertip, we extend its 2D prediction along the distal segment to the mask boundary, by at most 0.6 segment lengths. The mesh’s distal phalanges stretch accordingly; the benchmark 3D joints retain the MANO convention (Secs. C.5 and 4.4). The MHR-Hand fit uses the same points with its own learned initializer and Gauss–Newton step, at comparable throughput (Sec. A.9).

## 3.4. Parts and contact on the hand surface

Part segmentation. The optional part head predicts background, palm and three segments per finger using its own queries on the shared features and image-guided refinement at half the crop resolution. It trains with the pose network frozen, using cross-entropy and a loss that matches the part union to the hand mask. Reference labels are MANO part renders clipped to quality-checked SAM 2.1 masks, from ground-truth meshes where available and our mesh on 2Donly data (Sec. A.5). For pixel x, let $p _ { x , c }$ be the predicted class probability, y<sub>x</sub> its part label and $q _ { x } = 1 - p _ { x , 0 }$ the foreground union. With hand-mask target M, the objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p a r t } } = \mathrm { ~ - ~ } \frac { \sum _ { x \in \Omega } w _ { y _ { x } } \log p _ { x , y _ { x } } } { \sum _ { x \in \Omega } w _ { y _ { x } } } } \\ & { \quad \quad \quad + \lambda _ { \mathrm { u n i o n } } \big [ \mathrm { B C E } ( q , M ) + \lambda _ { D } \mathcal { D } ( q , M ) \big ] , } \\ & { \mathcal { D } ( q , M ) = 1 - \frac { 2 \sum _ { x } q _ { x } M _ { x } + 1 } { \sum _ { x } q _ { x } + \sum _ { x } M _ { x } + 1 } . } \end{array}\tag{3}
$$

Ω contains labeled pixels; $w _ { c } = 3$ for finger classes and 1 otherwise. Union BCE uses crops with a valid hand mask and negative crops (zero target); Dice is averaged only over hand-mask crops. Terms without valid labels are omitted. The pose network remains frozen.

At inference, each detector-mask pixel takes its most likely foreground part; the part union therefore equals the detector mask. This head needs no mesh.

Contact. A contact head on the frozen pose network estimates, for each of the 400 surface points, the probability that it touches an object. Each point query combines heatmap-pooled image features, predicted root-relative 3D position and visibility, and a learned point embedding. Together with the hand’s pose token, four transformer layers (256-d) attend over the points and cross-attend to pooled heatmap features and contact-branch tokens. The contact branch is a trainable copy of the encoder’s last four ViT blocks, operating on frozen encoder features; pose, points and parts are unchanged. At inference, the mesh’s spectral basis maps per-point probabilities to the 778 MANO vertices, clipped to [0, 1]. Finger and palm summaries take the maximum vertex probability, and a separate hand-level output flags contact. Training labels mark MANO vertices within 5, 10, 20 and 35 mm of or inside a groundtruth object mesh on HOT3D, DexYCB, HO-3D, OakInk2, egocentric ARCTIC [38] and HOI4D [39] training sequences, with InterHand2.6M hands as contact-free examples (699k crops, 102k of them from ARCTIC and HOI4D); MOW [33] supervises only the 35 mm output. All four sizes use 24k steps of class-balanced binary cross-entropy for the head and its branch. For tolerance $k \in \{ 5 , 1 0 , 2 0 , 3 5 \}$ mm, let $c _ { i k }$ and $y _ { i k } \in \{ 0 , 1 \}$ be the predicted point-contact probability and label. Over valid point labels in a minibatch, the balanced objective is

$$
\begin{array} { r } { \mathcal { B } _ { k } = \mathrm { ~ - ~ } \displaystyle \frac { \sum _ { i \in \mathcal { V } _ { k } } y _ { i k } \log c _ { i k } } { 2 \operatorname* { m a x } ( N _ { k } ^ { + } , 1 ) } \qquad } \\ { \displaystyle - \frac { \sum _ { i \in \mathcal { V } _ { k } } ( 1 - y _ { i k } ) \log ( 1 - c _ { i k } ) } { 2 \operatorname* { m a x } ( N _ { k } ^ { - } , 1 ) } , \ } \\ { \mathcal { L } _ { \mathrm { c o n t a c t } } = \frac { 1 } { 4 } \displaystyle \sum _ { k } \left[ \mathcal { B } _ { k } + \frac { 1 } { 2 } \mathrm { B C E } ( h _ { k } , y _ { k } ^ { h } ) \right] . } \end{array}\tag{4}
$$

$\nu _ { k }$ contains the valid labels at tolerance k, with positive and negative counts $N _ { k } ^ { + } , N _ { k } ^ { - }$ The hand-level probability $h _ { k }$ uses the corresponding binary target $y _ { k } ^ { h }$ , averaged over labeled hands. Missing tolerances contribute zero, so MOW supervises only the 35 mm terms.

A validation-fitted logit offset maps the selected 10 mm threshold to 0.5. No object geometry is used at inference. In video, a causal 4-layer transformer over the last 32 frames of each track reads the per-frame contact logits, hand logit, joint motion and hand token and predicts residuals to them; its map is blended with the per-frame map (weight 0.25) and its hand flag uses hysteresis. It trains on out-of-fold predictions: each of two image heads predicts the recordings excluded from its own training fold (Sec. A.6).

## 3.5. Video model

The video model smooths hand trajectories with causal refinement. Per-frame outputs are linked into one track per handedness, and a 4-layer, 256-d causal transformer over the last 32 frames of a track predicts residuals to the rootrelative joints and the root, and complete joint and root predictions on frames without a detection. It trains on a ViT-B image model’s predictions on HOT3D and DexYCB training clips at strides 1–3, so it also sees fast motion (Sec. A.10). At inference we blend refined and per-frame articulation with weights 0.75 and 0.25 and apply a causal One Euro filter [26]. The refined root replaces the per-frame estimate unless it would move the hand by more than 5 cm (Sec. A.10). The direct 2D outputs use their own speedadaptive filter, while a handedness memory suppresses side flips. Track proposals cover per-frame misses: a confirmed track without a confident detector box receives a box at its constant-velocity prediction, which the presence logit must accept (Sec. A.10).

## 3.6. Camera model

Our causal camera model, FAH-camera, estimates each frame’s pose on arrival from past and current frames only, so it runs online alongside hand estimation. It combines DPVO [40] odometry on hand-masked frames with a running median of background MoGe-2 [41] depth estimates for metric scale. A learned motion head uses frozen RAFTsmall [42] flow and DINOv3 ViT-S tokens to predict relative camera motion and its uncertainty. It corrects scale where odometry sees little parallax and supplies motion where odometry fails or the rotation or translation estimates disagree beyond the predicted uncertainty, for example when hands fill the frame and odometry tracks hand edges and moving objects (Sec. A.11). FAH-World (“FAH-L world” in tables) places the video model’s hands in world space with these poses. An offline mode keeps the trajectory of DROID-SLAM [23], masked with our hands, and fuses its per-clip metric scale with FAH-camera’s.

## 3.7. Efficient inference

The GPU pipeline shares one frame upload across detection and cropping, uses FP16 TensorRT detector and encoder engines, and captures the encoder, heads and mesh stage as CUDA graphs. Pose heads use TF32 and the numerically sensitive mesh stage fp32. Unless stated otherwise, fps include detection, pose, the mesh fit and the fingertip step, and exclude the optional part and contact heads.

## 4. Experiments

Benchmarks and metrics. We compare FAH with the relevant baselines in five settings:

• Egocentric 3D: HOT3D v1 (98 five-second clips from HaWoR’s validation split of HOT3D’s train\_aria recordings [17], excluded from training) and EgoStandard [43] (120 thirty-second test clips; aggregate metrics only). Metrics cover articulation (PA-MPJPE), mesh error (PA-MPVPE), placement (root error, MRRPE) and temporal quality (acceleration error, jitter). PA-MPJPE-p [25] also penalizes missed hands (Sec. B).

• Image-space pose: bench2d v2 (8.6k hands from HOT3D, CMU Panoptic, COCO-WholeBody and HInt [6]) and three small, model-assisted stress tests for challenging poses, low light and shadows (Sec. B). We report matched-hand PCK@0.1 and PCK with misses counted as failures (“incl.”).

• Contact: pooled vertex F1/AP and hand-level F1 on HOT3D and DexYCB s0 test, with additional evaluation on HOI4D and ARCTIC. Labels use ground-truth hand and object meshes; intervals are 95% paired sequencebootstrap intervals.

• World space: HOT3D v1 and EgoStandard, using W-MPJPE100, WA-MPJPE100 and RTE following

WHAM [44] and SLAHMR [45] (Sec. B).

• Throughput: fps is the inverse of median end-to-end frame time on an RTX 4070 Ti SUPER, using the same pre-decoded streams (Secs. A.12 and 3.7).

Tables use FAH-S, FAH-B, FAH-L (default) and FAH-H; video rows add causal refinement (FAH-L video for FAH-L).

## 4.1. Hand pose and mesh prediction

Images. FAH-L improves HOT3D articulation and mesh accuracy over the evaluated per-frame baselines (Tab. 1). On EgoStandard, the largest improvement over WiLoR is placement: root error falls from 64.18 to 30.21 mm, with lower articulation and mesh errors.

In 2D, FAH-L reaches 0.83 PCK@0.1 against 0.72 for the strongest baseline; it is 6.5× faster than SAM 3D Body’s hand branch (65 vs 10 fps) and more accurate on these test hands (0.83 vs 0.61, Tab. 2 and Fig. 5). It leads every subset except COCO (0.74 vs 0.75), and improves accuracy including misses on overlapping hands (0.64 vs 0.56 for RTMW-l). Recall on the three stress tests is 0.90, 0.96 and 1.00, against at most 0.84, 0.85 and 0.98 for the baselines (RTMW-l on low light and shadows).

Adding parts and then contact lowers FAH-L throughput from 65 to 54 and 49 fps. Across the released family, FAH-H has the lowest per-frame HOT3D articulation error at 27 fps and the lowest EgoStandard root error (28.53 mm), but not a lower EgoStandard PA-MPJPE (7.14 vs 7.10 mm for FAH-L).

Video. The main gain from video refinement is temporal accuracy: on HOT3D, acceleration error falls from 3.83 to 1.65 (HaWoR, the evaluated video baseline: 1.94) and PA-MPJPE from 4.31 to 4.16 mm (HaWoR: 5.89; Tab. 1). All four sizes show this trade-off: refinement lowers acceleration error on both datasets and HOT3D PA-MPJPE, but slightly raises EgoStandard PA-MPJPE. On EgoStandard, acceleration error falls from 3.99 to 1.67, while PA-MPJPE changes from 7.10 to 7.17 mm and root error from 30.21 to 30.29 mm.

## 4.2. Contact

Without object geometry at inference, FAH-L outperforms HACO on HOT3D and DexYCB (Tab. 4). The advantage also holds under HACO’s per-hand averaging protocol: 0.76 vs 0.60 on HOT3D and 0.71 vs 0.66 on DexYCB. Figure 3 includes six examples, one favoring HACO and one FAH failure.

Accuracy grows from FAH-S to FAH-H on HOT3D, DexYCB and ARCTIC; on in-domain HOI4D, FAH-H is not above FAH-L (Tab. 29).

On MOW [33] under HACO’s protocol and labels (35 mm, per image), with both trained on MOW, FAH-S and

Table 1. 3D accuracy on HOT3D v1 and EgoStandard (mm; accel and jitter in mm/frame<sup>2</sup>; PA-p: coverage-aware PA-MPJPE [25]) fps on HOT3D frames on an NVIDIA RTX 4070 Ti SUPER (16 GB) desktop GPU under Windows 11 + WSL2, slower of two rounds (MediaPipe: GPU delegate unavailable under WSL2 / Windows); <sup>§</sup>our CUDA-graph implementation; crop-based baselines use WiLoR’s detector. FAH-S, FAH-B, FAH-L and FAH-H (ViT-S, ViT-B, ViT-L, ViT-H+ at 448 px) share data, losses and schedule and differ only in encoder initialization and input size (Sec. C.1); their fps is end to end (detection, pose, mesh fit, fingertips; optional parts excluded), all four timed in one session. Evaluation scope: Secs. A.6, A.11 and B; full tables: Sec. C.
<table><tr><td rowspan="2">method</td><td rowspan="2"></td><td colspan="7">HOT3D v1</td><td colspan="6">EgoStandard</td></tr><tr><td>PA</td><td>PA-p</td><td>PVE</td><td></td><td>root MRRPE</td><td>accel</td><td>jitter</td><td></td><td>PA PA-p</td><td>PVE</td><td></td><td>root accel jitter</td><td></td></tr><tr><td>per frame</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MediaPipe Hands [46]</td><td></td><td>19.61</td><td>67.01</td><td></td><td>87.72</td><td>119.79</td><td>20.59</td><td>20.61</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HaMeR [6]</td><td>4.6</td><td>9.46</td><td>14.48</td><td>9.24</td><td>41.62</td><td>49.26</td><td>6.75</td><td>6.84</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hamba [8]</td><td>5.0</td><td>9.28</td><td>14.31</td><td>9.02</td><td>40.66</td><td>47.83</td><td>7.22</td><td>7.31</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WiLoR [7]</td><td>4.7</td><td>6.72</td><td>11.85</td><td>6.49</td><td>43.89</td><td>48.57</td><td>8.30</td><td>8.48</td><td>7.26</td><td>8.59</td><td>7.52</td><td>64.18</td><td>6.64</td><td>6.71</td></tr><tr><td>SAM 3D Body hand [10]</td><td>10§</td><td>8.30</td><td></td><td>8.49</td><td>40.00</td><td>52.36</td><td>5.62</td><td>5.81</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FAH-S</td><td>101</td><td>5.25</td><td>5.51</td><td>4.89</td><td>11.94</td><td>16.20</td><td>5.60</td><td>5.87</td><td>7.31</td><td>7.60</td><td>7.28</td><td>30.66</td><td>5.55</td><td>5.61</td></tr><tr><td>FAH-B</td><td>89</td><td>4.57</td><td>4.84</td><td>4.28</td><td>9.76</td><td>13.19</td><td>4.58</td><td>4.90</td><td>7.11</td><td>7.40</td><td>7.12</td><td>30.18</td><td>4.77</td><td>4.84</td></tr><tr><td>FAH-L</td><td>65</td><td>4.31</td><td>4.58</td><td>4.02</td><td>8.89</td><td>12.13</td><td>3.83</td><td>4.19</td><td>7.10</td><td>7.39</td><td>7.12</td><td>30.21</td><td>3.99</td><td>4.06</td></tr><tr><td>FAH-H</td><td>27</td><td>4.12</td><td>4.39</td><td>3.85</td><td>9.12</td><td>11.65</td><td>3.95</td><td>4.31</td><td>7.14</td><td>7.44</td><td>7.15</td><td>28.53</td><td>3.95</td><td>4.01</td></tr><tr><td>video</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HaWoR [11]</td><td>11</td><td>5.89</td><td>14.78</td><td>5.5729.51</td><td></td><td>27.67</td><td>1.94</td><td>2.03</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FAH-S video</td><td>88</td><td>5.08</td><td>5.28</td><td>4.80</td><td>11.59</td><td>15.70</td><td>2.24</td><td>2.45</td><td>7.35</td><td>7.50</td><td>7.30</td><td>30.75</td><td>2.14</td><td>2.14</td></tr><tr><td>FAH-B video</td><td>79</td><td>4.42</td><td>4.65</td><td>4.19</td><td>9.42</td><td>12.69</td><td>1.89</td><td>2.15</td><td>7.16</td><td>7.30</td><td>7.15</td><td>30.25</td><td>1.91</td><td>1.91</td></tr><tr><td>FAH-L video</td><td>59</td><td>4.16</td><td>4.38</td><td>3.95</td><td>8.55</td><td>11.58</td><td>1.65</td><td>1.92</td><td>7.17</td><td>7.31</td><td>7.15</td><td>30.29</td><td>1.67</td><td>1.66</td></tr><tr><td>FAH-H video</td><td>26</td><td>3.98</td><td>4.19</td><td>3.77</td><td>8.34</td><td>10.72</td><td>1.66</td><td>1.94</td><td>7.20</td><td>7.35</td><td>7.18</td><td>28.84</td><td>1.65</td><td>1.63</td></tr></table>

Table 2. 2D accuracy on bench2d v2 and the challenging-pose, low-light and shadow sets. PCK@0.1 on matched hands; “incl.”: missed hands count as failures; “3D fit”: 3D joints projected at the best-fitting translation; “overlap”: PCK@0.1 incl. misses on 387 overlapping hands; shadows: the 278 scored hands of shadow\_v1. Full metric set: Sec. C; fps as in Tab. 1.
<table><tr><td rowspan="2">method</td><td rowspan="2">fps</td><td colspan="4">bench2d v2</td><td colspan="3">challenging poses</td><td colspan="3">low light</td><td colspan="3">shadows</td></tr><tr><td>PCK</td><td></td><td>incl. 3D fit</td><td>overlap</td><td>recall</td><td>PCK</td><td>incl.</td><td>recall</td><td>PCK</td><td>incl.</td><td>recall</td><td>PCK</td><td>incl.</td></tr><tr><td>MediaPipe Hands [46]</td><td></td><td>0.65</td><td>0.34</td><td>0.55</td><td>0.28</td><td>0.05</td><td>0.94</td><td>0.05</td><td>0.33</td><td>0.65</td><td>0.22</td><td>0.58</td><td>0.89</td><td>0.51</td></tr><tr><td>RTMPose-m [47]</td><td>89</td><td>0.71</td><td>0.40</td><td></td><td>0.23</td><td>0.07</td><td>0.82</td><td>0.06</td><td>0.35</td><td>0.70</td><td>0.24</td><td>0.45</td><td>0.94</td><td>0.42</td></tr><tr><td>RTMW-1 [48]</td><td>32</td><td>0.68</td><td>0.65</td><td></td><td>0.56</td><td>0.13</td><td>0.84</td><td>0.11</td><td>0.85</td><td>0.65</td><td>0.56</td><td>0.98</td><td>0.89</td><td>0.87</td></tr><tr><td>HaMeR [6]</td><td>4.6</td><td>0.71</td><td>0.66</td><td>0.65</td><td>0.55</td><td>0.84</td><td>0.93</td><td>0.78</td><td>0.82</td><td>0.70</td><td>0.57</td><td>0.95</td><td>0.97</td><td>0.92</td></tr><tr><td>Hamba [8]</td><td>5.0</td><td>0.72</td><td>0.66</td><td>0.65</td><td>0.55</td><td>0.84</td><td>0.93</td><td>0.78</td><td>0.81</td><td>0.70</td><td>0.57</td><td>0.95</td><td>0.97</td><td>0.92</td></tr><tr><td>WiLoR [7]</td><td>4.7</td><td>0.69</td><td>0.64</td><td>0.77</td><td>0.52</td><td>0.84</td><td>0.94</td><td>0.78</td><td>0.81</td><td>0.67</td><td>0.55</td><td>0.95</td><td>0.97</td><td>0.92</td></tr><tr><td>FAH-S</td><td>101</td><td>0.80</td><td>0.78</td><td>0.79</td><td>0.60</td><td>0.90</td><td>0.99</td><td>0.89</td><td>0.96</td><td>0.76</td><td>0.73</td><td>1.00</td><td>0.98</td><td>0.97</td></tr><tr><td>FAH-B</td><td>89</td><td>0.82</td><td>0.80</td><td>0.82</td><td>0.62</td><td>0.90</td><td>0.99</td><td>0.89</td><td>0.96</td><td>0.79</td><td>0.76</td><td>1.00</td><td>0.99</td><td>0.99</td></tr><tr><td>FAH-L</td><td>65</td><td>0.83</td><td>0.81</td><td>0.83</td><td>0.64</td><td>0.90</td><td>0.99</td><td>0.90</td><td>0.96</td><td>0.81</td><td>0.78</td><td>1.00</td><td>0.99</td><td>0.99</td></tr><tr><td>FAH-H</td><td>27</td><td>0.83</td><td>0.81</td><td>0.83</td><td>0.64</td><td>0.90</td><td>0.99</td><td>0.90</td><td>0.97</td><td>0.81</td><td>0.78</td><td>1.00</td><td>0.99</td><td>0.98</td></tr></table>

FAH-B improve on HACO (F1 0.58 and 0.57 vs 0.52), while neither FAH-L nor FAH-H, the weakest of the four (0.53), is significantly above it. On the pipeline’s own detections, FAH-L video mainly improves contact timing: HOT3D onset and release vertex F1 rise by 0.053 and 0.055. False contact on hovering HOT3D hands falls from 1.50 to 0.82% of vertices, and DexYCB vertex F1 rises by 0.013.

## 4.3. Hands in world space

FAH-World places the video hands in world space with FAH-camera, our strictly causal camera model (Sec. 3.6 and Tab. 3). Errors are lower than HaWoR’s offline pipeline on all reported metrics. The 95% sequence-bootstrap intervals exclude zero on HOT3D, EgoStandard chunks and

RTE, but include zero for EgoStandard whole tracks; on frames detected by both methods, the whole-track intervals also exclude zero (Tab. 20). To separate hand and camera effects, we also use HaWoR’s DROID-SLAM camera with our hands. This improves on HaWoR’s hands except on EgoStandard whole tracks (W 260.56 vs 210.46 mm; interval includes zero). With our hands fixed, FAH-camera has higher HOT3D chunk W error than HaWoR’s DROID-SLAM camera; its EgoStandard difference is unresolved. The offline mode (DROID-SLAM with our hand masks and FAH-camera’s scale) lowers EgoStandard error relative to HaWoR’s DROID-SLAM camera, with no statistically resolved difference on HOT3D (Sec. 3.6).

Table 3. Hands in world space (W / WA-MPJPE in mm, RTE in %; chunks of 100 frames; mean of 3 runs, SDs in Tab. 21; each method’s own true positives). HOT3D scores guarded camera choices; earlier configurations were also scored on EgoStandard test (Sec. A.11). DROID-SLAM: HaWoR’s camera stage (DROID-SLAM masked with HaWoR’s hands, Metric3D scale, no infiller), one trajectory per clip shared by WiLoR, HaWoR and FAH-L video; with FAH-L world, our offline mode: DROID-SLAM masked with our hands, scale fused with FAH-camera’s. FAH-camera: our causal camera model (Sec. 3.6). causal: every stage uses only past and current frames. fps: al stages, desktop RTX 4070 Ti SUPER, 4 EgoStandard test clips, slower of 2 rounds.
<table><tr><td rowspan="5"></td><td rowspan="5"></td><td rowspan="5"></td><td colspan="6">HOT3D v1 (5 s clips)</td><td colspan="6">EgoStandard (30 s clips)</td></tr><tr><td colspan="2">chunks</td><td colspan="3">whole tracks</td><td colspan="2">chunks</td><td colspan="2">whole tracks</td><td colspan="2"></td></tr><tr><td>causal</td><td>W</td><td>WA</td><td>W</td><td>WA</td><td>RTE</td><td>W</td><td>WA</td><td>W</td><td>WA</td><td>RTE</td><td>fps</td></tr><tr><td>WiLoR [7]</td><td>DROID-SLAM</td><td>x</td><td>52.49</td><td>20.64</td><td>60.05</td><td>23.06</td><td>10.58</td><td>83.31</td><td>26.87</td><td>267.54</td><td>56.99</td><td>1.73</td><td>2.5</td></tr><tr><td>HaWoR [11]</td><td>DROID-SLAM</td><td>x</td><td>31.96</td><td>13.79</td><td>36.45</td><td>14.91</td><td>7.52</td><td>76.44</td><td>28.55</td><td>210.46</td><td>53.62</td><td>2.22</td><td>4.9</td></tr><tr><td>FAH-L video</td><td>DROID-SLAM</td><td>x</td><td>22.69</td><td>8.81</td><td>27.17</td><td>9.81</td><td>3.68</td><td>68.20</td><td>23.53</td><td>260.56</td><td>55.02</td><td>1.35</td><td>4.4</td></tr><tr><td>FAH-L world</td><td>FAH-camera</td><td>√</td><td>26.60</td><td>10.87</td><td>32.24</td><td>12.62</td><td>4.73</td><td>55.90</td><td>21.26</td><td>177.65</td><td>45.76</td><td>1.30</td><td>38</td></tr><tr><td>FAH-L world</td><td>DROID-SLAM</td><td>X</td><td>21.74</td><td>8.49</td><td>25.92</td><td>9.44</td><td>3.23</td><td>55.20</td><td>21.73</td><td>152.64</td><td>40.60</td><td>1.18</td><td>7.1</td></tr><tr><td>WiLoR [7]</td><td>GT</td><td></td><td>50.58</td><td>19.93</td><td>57.82</td><td>22.10</td><td>9.27</td><td>53.91</td><td>18.48</td><td>130.34</td><td>23.80</td><td>0.78</td><td></td></tr><tr><td>HaWoR [11]</td><td>GT</td><td></td><td>30.68</td><td>13.17</td><td>34.62</td><td>14.03</td><td>7.00</td><td>46.73</td><td>20.81</td><td>90.78</td><td>25.50</td><td>1.19</td><td></td></tr><tr><td>FAH-L video</td><td>GT</td><td></td><td>19.91</td><td>7.59</td><td>23.57</td><td>8.04</td><td>2.50</td><td>33.45</td><td>14.45</td><td>77.19</td><td>17.20</td><td>0.43</td><td></td></tr></table>

![](images/2e9cd81f07cc1f2d7a068e111ffc3f434e5aebf0263ae3d5d43f51aa167fb96e.jpg)  
Figure 3. Contact on held-out HOT3D and DexYCB hands. Each example shows the input, ground truth, HACO [29] and FAH-L on the same ground-truth MANO mesh. Gray to violet indicates contact probability, with each method’s validation threshold mapped to 0.5; scores are per-hand vertex F1. HACO wins on the fourth example; the last shows a missed fingertip contact. Selection: Sec. C.7; credits: Sec. D.

## 4.4. Ablations and analysis

Dense points and mesh fitting (Tab. 5). We compare a mesh fitted to dense points with a MANO-parameter head trained in the same network. The parameter head disagrees more with the image (its joints lie 3.07% of hand size from the direct 2D prediction, against 1.43% for the dense 3D joints and 2.28% for the fitted mesh’s joints) and is less accurate (PA-MPVPE 6.79 vs 4.09 mm). Against SAM 3D Body’s MHR decoder with a direct keypoint head, the dense head is better on HOT3D joints (4.38 vs 4.85 mm), 2D (0.84 vs 0.83 PCK@0.1), root placement (8.67 vs 11.59 mm) and silhouette agreement, at 1.7× the speed (Sec. C.5).

Backbone and resolution (Tab. 6). Larger encoders and inputs improve HOT3D accuracy in the controlled ablation; input resolution changes in-the-wild 2D PCK by at most 0.004. The released FAH-L uses ViT-L at 320 px; on the desktop, speed falls from 117 fps (ViT-S, 256 px) to 28 fps (ViT-H+, 448 px).

2D versus projected 3D. Projecting camera-frame 3D transfers errors in geometry and placement to 2D: in our ViT-B comparison the direct 2D head beats projected 3D and mesh joints (PCK@0.05 0.83 / 0.79 / 0.77; Tab. 8) and the dense 3D joints beat the mesh’s regressor joints (4.41 vs 5.14 mm), so we keep separate outputs despite their 1.42– 1.78% disagreement. Further image/video ablations and data-mix studies appear in Tabs. 13, 15 and 17.

## 5. Limitations and Conclusion

Limitations. FAH has the following limitations:

• Camera estimation dominates world-space error. Better camera trajectories remain an important direction (Sec. A.11).

Table 4. Hand contact on held-out HOT3D and DexYCB s0 test. Pooled vertex F1/AP and hand-level F1 (↑), with 10 mm contact labels. Ground-truth boxes unless marked <sup>†</sup> (full pipeline); distance rules use known object geometry. HACO is zero-shot on HOT3D. Protocol and selection details: Secs. A.6 and B.
<table><tr><td rowspan="3">method</td><td colspan="2">HOT3D</td><td colspan="3">DexYCB s0</td></tr><tr><td>F1</td><td>AP flag</td><td>F1</td><td>AP</td><td>flag</td></tr><tr><td></td><td>0.47</td><td>0.82</td><td>0.66</td><td>0.61</td></tr><tr><td>HACO [29] FAH-S</td><td>0.58 0.77</td><td>0.83 0.98</td><td>0.75</td><td>0.81</td><td>0.86 0.98</td></tr><tr><td>FAH-B</td><td>0.79</td><td>0.85</td><td>0.98</td><td>0.76 0.82</td><td>0.98</td></tr><tr><td>FAH-L</td><td>0.80</td><td>0.85</td><td>0.98</td><td>0.76 0.82</td><td>0.98</td></tr><tr><td>FAH-H</td><td>0.80</td><td>0.86 0.98</td><td>0.77</td><td>0.82</td><td>0.98</td></tr><tr><td>FAH-L, per frame† FAH-L video</td><td>0.80 0.80</td><td>0.85 0.98 0.86 0.97</td><td>0.76 0.77</td><td>0.81 0.81</td><td>0.98 0.97</td></tr><tr><td>rule: dense points rule: fitted MANO</td><td>0.72 0.65</td><td></td><td>0.98 0.96</td><td>0.56 0.56</td><td>0.95 0.94</td></tr></table>

Table 5. Dense points versus MANO parameters in one network on HOT3D (3D joints 4.41 mm PA-MPJPE): mesh fitted to the dense points (dense) vs the network’s own MANO-parameter head (param.). 2D: median per hand vs the direct 2D joints, % of hand size. Last rows: mesh vs the dense prediction, rigidly aligned (a diagnostic of the fit, not of accuracy).
<table><tr><td></td><td>dense</td><td>param.</td></tr><tr><td>3D joints vs 2D (%)</td><td>1.43</td><td>3.07</td></tr><tr><td>mesh joints vs 2D (%)</td><td>2.28</td><td>3.07</td></tr><tr><td>mesh PA-MPVPE (mm)</td><td>4.09</td><td>6.79</td></tr><tr><td>mesh vs dense joints (mm)</td><td>2.52</td><td>4.15</td></tr><tr><td>mesh vs dense surface (mm)</td><td>1.64</td><td>3.51</td></tr></table>

Table 6. Backbone and input resolution. Controlled 12k-step ablation with one recipe and no external distillation: HOT3D PA-MPJPE (mm) / bench2d v2 PCK@0.1 / desktop fps, without the fingertip step.
<table><tr><td></td><td>256 px</td><td>320px</td><td>448 px</td></tr><tr><td>ViT-S</td><td>5.50/0.80/117</td><td>5.40/0.80/109</td><td>5.35/0.80/91</td></tr><tr><td>ViT-B</td><td>4.82/0.82/104</td><td>4.75/0.82/93</td><td>4.70/0.82/73</td></tr><tr><td>ViT-L</td><td>4.53/0.83/80</td><td>4.46/0.83/68</td><td>4.41/0.83/47</td></tr><tr><td>ViT-H+</td><td></td><td></td><td>4.34/0.83/28</td></tr></table>

• Contact without object geometry remains ambiguous, and most contact vertices are hidden on 96% of HOT3D contact hands; the head also fires on hand-face and hand-hand contact in our in-the-wild videos, although training labels cover only hand-object contact.

• FAH prioritizes real-time inference over maximum accuracy. Stronger models could improve its supervision and camera estimates: SAM 3 [13] or vision-language-guided segmentation could replace SAM 2.1 pseudo-masks, and a stronger optical-flow model could replace RAFT-small in the camera motion head. These changes require evaluating accuracy gains against additional computation.

Table 7. Part-supervision ablation. Matched 2k-step joint training: PA-MPJPE (mm) on HOT3D and DexYCB for ViT-B; HOT3D PA-MPJPE and mined-set PCK@0.05 for continued FAH-L training. Union loss combines part-union/mask and densepart terms; ×0.3 scales pose gradients.
<table><tr><td>ViT-B</td><td colspan="2">HOT3D DexYCB</td></tr><tr><td>control</td><td>4.43</td><td>5.69</td></tr><tr><td>+ dense part terms</td><td>4.43</td><td>5.68</td></tr><tr><td>+ part head, union loss</td><td>4.47</td><td>5.72</td></tr><tr><td>FAH-L, continued</td><td>HOT3D</td><td>mined</td></tr><tr><td>without parts</td><td>4.30</td><td>0.80</td></tr><tr><td>+ joint part head (×0.3)</td><td>4.61</td><td>0.81</td></tr></table>

Table 8. Direct versus projected FAH outputs. Internal comparison using a ViT-B development model on matched HOT3D hands, with true intrinsics. PCK@0.05 (↑) compares direct 2D, projected camera-frame joints and projected mesh joints; PA-MPJPE (↓, mm) compares direct 3D with mesh-regressed MANO joints plus tips.
<table><tr><td></td><td colspan="2">2D PCK@0.05 ↑</td><td colspan="2">PA-MPJPE↓</td></tr><tr><td>method</td><td></td><td>2D 3D proj. mesh proj.</td><td>3D</td><td>mesh</td></tr><tr><td>FAH, image 0.83</td><td></td><td>0.79</td><td>0.774.41</td><td>5.14</td></tr><tr><td>FAH, video 0.83</td><td></td><td>0.79</td><td>0.77 4.24</td><td>5.04</td></tr></table>

Conclusion. We present FAH, a fast yet accurate hand estimation method. It predicts corresponding 2D and 3D hand points and fits MANO or MHR-Hand in a separate stage, improving mesh accuracy over a parameter head in the same network. The predicted surface also supports parts and contact; causal refinement and camera estimation extend it to video and world space. FAH achieves state-of-the-art performance on different hand estimation tasks while maintaining real-time speed on a consumer GPU.

## References

[1] Yuzhe Qin, Wei Yang, Binghao Huang, Karl Van Wyk, Hao Su, Xiaolong Wang, Yu-Wei Chao, and Dieter Fox. AnyTeleop: a general vision-based dexterous robot armhand teleoperation system. In Robotics: Science and Systems (RSS). 2023.

[2] Chen Wang, Haochen Shi, Weizhuo Wang, Ruohan Zhang, Li Fei-Fei, and C. Karen Liu. DexCap: scalable and portable mocap data collection system for dexterous manipulation. In Robotics: Science and Systems (RSS). 2024.

[3] Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang, and Xiaolong Wang. Open-TeleVision: teleoperation with immersive active visual feedback. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Pro ceedings of Machine Learning Research, 2729–2749. 2025. URL: https://proceedings.mlr.press/v270/ cheng25b.html.

[4] Javier Romero, Dimitrios Tzionas, and Michael J. Black.

Embodied hands: modeling and capturing hands and bodies together. ACM TOG, 2017.

[5] Aaron Ferguson, Ahmed A. A. Osman, Berta Bescos, Carsten Stoll, Chris Twigg, Christoph Lassner, David Otte, Eric Vignola, Fabian Prada, Federica Bogo, Igor Santesteban, Javier Romero, Jenna Zarate, Jeongseok Lee, Jinhyung Park, Jinlong Yang, John Doublestein, Kishore Venkateshan, Kris Kitani, Ladislav Kavan, Marco Dal Farra, Matthew Hu, Matthew Cioffi, Michael Fabris, Michael Ranieri, Mohammad Modarres, Petr Kadlecek, Rawal Khirodkar, Rinat Abdrashitov, Romain Prévost, Roman Rajbhandari, Ronald Mallet, Russell Pearsall, Sandy Kao, Sanjeev Kumar, Scott Parrish, Shoou-I Yu, Shunsuke Saito, Takaaki Shiratori, Te-Li Wang, Tony Tung, Yichen Xu, Yuan Dong, Yuhua Chen, Yuanlu Xu, Yuting Ye, and Zhongshi Jiang. MHR: momentum human rig. arXiv preprint arXiv:2511.15586, 2025. URL: https : / / arxiv . org / abs / 2511 . 15586, doi:10.48550/arXiv.2511.15586.

[6] Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing hands in 3D with transformers. In CVPR. 2024.

[7] Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos Zafeiriou. WiLoR: end-to-end 3D hand localization and reconstruction in-the-wild. In CVPR. 2025.

[8] Haoye Dong, Aviral Chharia, Wenbo Gou, Francisco Vicente Carrasco, and Fernando De la Torre. Hamba: singleview 3D hand reconstruction with graph-guided bi-scanning Mamba. In NeurIPS. 2024.

[9] Chen Si, Yulin Liu, Bo Ai, Jianwen Xie, Rolandos Alexandros Potamias, Chuanxia Zheng, and Hao Su. AnyHand: a large-scale synthetic dataset for RGB(-D) hand pose estimation. arXiv preprint arXiv:2603.25726, 2026.

[10] Xitong Yang, Devansh Kukreja, Don Pinkus, Taosha Fan, Jinhyung Park, Soyong Shin, Jinkun Cao, Jia-Wei Liu, Nicolás Ugrinovic, Anushka Sagar, Jitendra Malik, Matt Feiszli, Piotr Dollár, and Kris Kitani. SAM 3D Body: robust full-body human mesh recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7209–7219. 2026. URL: https://openaccess.thecvf.com/content/ CVPR2026/html/Yang\_SAM\_3D\_Body\_Robust\_ Full-Body\_Human\_Mesh\_Recovery\_CVPR\_2026\_ paper.html.

[11] Jinglei Zhang, Jiankang Deng, Chao Ma, and Rolandos Alexandros Potamias. HaWoR: world-space hand motion reconstruction from egocentric videos. In CVPR. 2025.

[12] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. SAM 2: segment anything in images and videos. In International Conference on Learning Representations. 2025. URL: https://proceedings. iclr . cc / paper \_ files / paper / 2025 / hash / 45c1f6a8cbf2da59ebf2c802b4f742cd Abstract-Conference.html.

[13] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: segment anything with concepts. In International Conference on Learning Representations. 2026. URL: https://proceedings. iclr . cc / paper \_ files / paper / 2026 / hash / e0982cbc81401df3430ee1ff780dc7a2 Abstract-Conference.html.

[14] Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, and Thomas Brox. FreiHAND: a dataset for markerless capture of hand pose and shape from single RGB images. In ICCV. 2019.

[15] Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. HOnnotate: a method for 3D annotation of hand and object poses. In CVPR. 2020.

[16] Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, and Dieter Fox. DexYCB: a benchmark for capturing hand grasping of objects. In CVPR. 2021.

[17] Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, and Tomas Hodan. HOT3D: hand and object tracking in 3D from ego centric multi-view videos. In CVPR, 7061–7071. 2025.

[18] Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: learning dexterous manipulation from large-scale egocentric video. In International Conference on Learning Representations. 2026. URL: https://arxiv.org/abs/2505.11709.

[19] Kevin Lin, Lijuan Wang, and Zicheng Liu. Mesh Graphormer. In ICCV, 12939–12948. 2021.

[20] Zhishan Zhou, Shihao Zhou, Zhi Lv, Minqiang Zou, Yao Tang, and Jiajun Liang. A simple baseline for efficient hand mesh reconstruction. In CVPR. 2024.

[21] Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, and Zhenan Sun. PyMAF: 3D hu man pose and shape regression with pyramidal mesh align ment feedback loop. In ICCV, 11446–11456. 2021.

[22] Aditya Prakash, Ruisen Tu, Matthew Chang, and Saurabh Gupta. 3D hand pose estimation in everyday egocentric images. In ECCV. 2024.

[23] Zachary Teed and Jia Deng. DROID-SLAM: deep visual SLAM for monocular, stereo, and RGB-D cameras. In NeurIPS. 2021.

[24] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3D v2: a versatile monocular geomet-

ric foundation model for zero-shot metric depth and surface normal estimation. IEEE TPAMI, 2024.

[25] Yuxi Wang, Chengkai Jin, Yufei Liu, Wenqi Ouyang, Tianyi Wei, Zhiwei Zeng, Siyuan Huang, Zhiqi Shen, and Xingang Pan. The surprising effectiveness of video diffusion models for hand motion reconstruction. arXiv preprint arXiv:2606.30308, 2026.

[26] Géry Casiez, Nicolas Roussel, and Daniel Vogel. 1 euro filter: a simple speed-based low-pass filter for noisy input in interactive systems. In CHI. 2012.

[27] Ailing Zeng, Lei Yang, Xuan Ju, Jiefeng Li, Jianyi Wang, and Qiang Xu. SmoothNet: a plug-and-play network for refining human poses in videos. In ECCV, 625–642. 2022.

[28] Qichen Fu, Xingyu Liu, Ran Xu, Juan Carlos Niebles, and Kris M. Kitani. Deformer: dynamic fusion transformer for robust hand pose estimation. In ICCV, 23600–23611. 2023.

[29] Daniel Sungho Jung and Kyoung Mu Lee. Learning dense hand contact estimation from imbalanced data. In NeurIPS. 2025.

[30] Chun-Hao P. Huang, Hongwei Yi, Markus Höschle, Matvey Safroshkin, Tsvetelina Alexiadis, Senya Polikovsky, Daniel Scharstein, and Michael J. Black. Capturing and inferring dense full-body human-scene contact. In CVPR. 2022.

[31] Shashank Tripathi, Agniv Chatterjee, Jean-Claude Passy, Hongwei Yi, Dimitrios Tzionas, and Michael J. Black. DECO: dense estimation of 3D human-scene contact in the wild. In ICCV. 2023.

[32] Jyun-Ting Song, JungEun Kim, Jinkun Cao, Yu Lei, Takuma Yagi, and Kris Kitani. Contact4D: a video dataset for wholebody human motion and finger contact in dexterous operations. In International Conference on 3D Vision (3DV), 904–914. 2026. URL: https://jyuntins.github. io/Contact4D/.

[33] Zhe Cao, Ilija Radosavovic, Angjoo Kanazawa, and Jitendra Malik. Reconstructing hand-object interactions in the wild. In ICCV. 2021.

[34] Jinkun Cao, Jingyuan Liu, Kris Kitani, and Yi Zhou. Joint diffusion for universal hand-object grasp generation. Transactions on Machine Learning Research, 2025. URL: https : / / openreview . net / forum ? id = TZ0ztsYR6x.

[35] Zhengyi Luo, Jinkun Cao, Sammy Christen, Alexander Winkler, Kris Kitani, and Weipeng Xu. Omnigrasp: grasping diverse objects with simulated humanoids. In Advances in Neural Information Processing Systems, volume 37. 2024. URL: https://proceedings. neurips . cc / paper \_ files / paper / 2024 / hash / 040ace837dd270a87055bb10dd7c0392 Abstract-Conference.html, doi:10.52202/079017- 0070.

[36] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLOv8. https://github.com/ultralytics/ ultralytics, 2023. Software.

[37] Berthold K. P. Horn. Closed-form solution of absolute orientation using unit quaternions. Journal of the Optical Society ofAmerica A, 4(4):629–642, 1987.

[38] Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, and Otmar

Hilliges. ARCTIC: a dataset for dexterous bimanual handobject manipulation. In CVPR. 2023.

[39] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. HOI4D: a 4D egocentric dataset for category-level humanobject interaction. In CVPR. 2022.

[40] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch vi sual odometry. In NeurIPS. 2023.

[41] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. MoGe-2: accurate monocular geometry with metric scale and sharp details. In NeurIPS. 2025.

[42] Zachary Teed and Jia Deng. RAFT: recurrent all-pairs field transforms for optical flow. In ECCV. 2020.

[43] Lightwheel. EgoSuite-Open100K: a fully annotated open egocentric human dataset. https://huggingface. co / collections / LightwheelAI / egosuite - open100k, 2026. Dataset; EgoStandard subset. URL: https : / / huggingface . co / datasets / LightwheelAI/EgoStandard.

[44] Soyong Shin, Juyong Kim, Eni Halilaj, and Michael J. Black. WHAM: reconstructing world-grounded humans with accurate 3D motion. In CVPR. 2024.

[45] Vickie Ye, Georgios Pavlakos, Jitendra Malik, and Angjoo Kanazawa. Decoupling human and camera motion from videos in the wild. In CVPR. 2023.

[46] Fan Zhang, Valentin Bazarevsky, Andrey Vakunov, Andrei Tkachenka, George Sung, Chuo-Ling Chang, and Matthias Grundmann. MediaPipe Hands: on-device real-time hand tracking. arXiv preprint arXiv:2006.10214, 2020.

[47] Tao Jiang, Peng Lu, Li Zhang, Ningsheng Ma, Rui Han, Chengqi Lyu, Yining Li, and Kai Chen. RTMPose: realtime multi-person pose estimation based on MMPose. arXiv preprint arXiv:2303.07399, 2023.

[48] Tao Jiang, Xinchen Xie, and Yining Li. RTMW: real-time multi-person 2D and 3D whole-body pose estimation. arXiv preprint arXiv:2407.08634, 2024.

[49] Sheng Jin, Lumin Xu, Jin Xu, Can Wang, Wentao Liu, Chen Qian, Wanli Ouyang, and Ping Luo. Whole-body human pose estimation in the wild. In ECCV. 2020.

[50] Tomas Simon, Hanbyul Joo, Iain Matthews, and Yaser Sheikh. Hand keypoint detection in single images using multiview bootstrapping. In CVPR. 2017.

[51] Hao-Shu Fang, Jiefeng Li, Hongyang Tang, Chao Xu, Haoyi Zhu, Yuliang Xiu, Yong-Lu Li, and Cewu Lu. AlphaPose: whole-body regional multi-person pose estimation and track ing in real-time. IEEE TPAMI, 45(6):7157–7173, 2023. doi:10.1109/TPAMI.2022.3222784.

[52] Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar, Richard Higgins, Sanja Fidler, David Fouhey, and Dima Damen. EPIC-KITCHENS VISOR benchmark: VIdeo segmentations and object relations. In NeurIPS. 2022.

[53] Gyeongsik Moon, Shoou-I Yu, He Wen, Takaaki Shiratori, and Kyoung Mu Lee. InterHand2.6M: a dataset and baseline for 3D interacting hand pose estimation from a single RGB image. In ECCV. 2020.

[54] Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, and Cewu Lu. OakInk2: a dataset of bimanual hands-object manipulation in complex task completion. In CVPR. 2024.

[55] Christian Zimmermann, Max Argus, and Thomas Brox. Contrastive representation learning for hand shape estima tion. In GCPR. 2021.

[56] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025. URL: https://arxiv.org/ abs/2508.10104.

[57] Zhihao Li, Jianzhuang Liu, Zhensong Zhang, Songcen Xu, and Youliang Yan. CLIFF: carrying location information in full frames into human pose and shape estimation. In ECCV. 2022.

[58] Alexander Kapitanov, Karina Kvanchiani, Alexander Nagaev, Roman Kraynov, and Andrei Makhliarchuk. HaGRID – HAnd gesture recognition image dataset. In WACV. 2024.

[59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML. 2021.

[60] Binzhu Xie, Shi Qiu, Sicheng Zhang, Yinqiao Wang, Hao Xu, Muzammal Naseer, Chi-Wing Fu, and Pheng-Ann Heng. EgoHandICL: egocentric 3D hand reconstruction with incontext learning. In International Conference on Learning Representations. 2026. URL: https://proceedings. iclr . cc / paper \_ files / paper / 2026 / hash / 3a7cb31c8f4fa170aaa38f0e28d6042e Abstract-Conference.html.

[61] Zhengdi Yu, Stefanos Zafeiriou, and Tolga Birdal. Dyn-HaMR: recovering 4D interacting hand motion from a dynamic camera. In CVPR. 2025.

[62] Tianyi Cheng, Dandan Shan, Ayda Sultan Hassen, Richard Ely Locke Higgins, and David Fouhey. Towards a richer 2D understanding of hands at scale. In NeurIPS. 2023.

[63] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: collection, pipeline and challenges for EPIC-KITCHENS-100. International Journal ofComputer Vision, 130:33–55, 2022.

# Fast and Accurate Hand Estimation in Pixels, Interaction and World Space

Supplementary Material

## A. Models and training

This section describes how we train the released models. Development ablations are in Tabs. 15 and 17. MHR-ref is the 320 px ViT-B student with SAM 3D Body’s decoder, MHR head and direct keypoints (Sec. A.2). Unless noted, development ablations use the decoder-free ViT-B at the last milestone with YOLOv8m-seg, verification, tall-box crops, one re-crop and MANO fitting; FAH-World runs with mesh fitting off (joints unchanged).

## A.1. Training setup

We train pose networks with AdamW and weight decay 0.01, using a separate learning rate for the encoder. The schedule uses linear warm-up and cosine decay. We clip gradients at 1.0 and skip updates whose pre-clip gradient norm exceeds a threshold. Training sources are sampled with fixed weights independent of dataset size (Tab. 9). We sample square windows from a 640 px crop with context factor 1.25. Scale is sampled as exp(U(log 0.85, log 1.15) · j) and shift as $\mathcal { U } ( \pm 0 . 0 8 ) \cdot j ,$ , with jitter $j = 1$ unless noted. We do not rotate these crops.

Photometric augmentation changes contrast, brightness, saturation and white balance with probability 0.8. Their ranges are 0.6–1.4, ±30, 0.5–1.5 and 0.9–1.1, respectively. Grayscale conversion, blur and JPEG compression have probabilities 0.05, 0.15 and 0.15; JPEG quality ranges from 30 to 80.

The student also receives low-light and egocentric degradations, while the teacher sees a clean 384 px view. Lowlight synthesis reduces exposure to 0.015–0.25 of the original and adds shot and read noise. Egocentric augmentation adds motion blur, sensor noise and 1–3 occluders, and reduces resolution by a factor of two to four. Hand-specific occluders cover the wrist or fingertips.

## A.2. Teacher, students and the MHR keypoint model

Hand-only SAM 3D Body teacher. We extract SAM 3D Body’s [10] hand branch and run it independently with bitexact outputs. It contains a DINOv3-H+ encoder [56], promptable decoder, hand-mode MHR head and camera head, retaining 899M of the original 953M parameters. A fast rig operates on the hand sub-skeleton and its 2,445 vertices. We map these vertices to MANO joints with a sparse convex regressor with 169 weights, fitted on 4,070 Frei-HAND training images, and mirror left hands.

We fine-tune the teacher for 10k steps on 384 px crops, with 500 warm-up steps. The decoder and heads use a learning rate of $2 \cdot 1 0 ^ { - 5 }$ ; the last eight encoder blocks and final normalization layer use one quarter of that rate. All other encoder layers remain frozen. We sample HOT3D and COCO-WholeBody with weight 1, Halpe with 0.8, HInt and the EgoDex test split with 0.6, FreiHAND with 0.5, and CMU with 0.3. The EgoDex 3D supervision weight is 0.3. The training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { 2 D } } + 3 \mathcal { L } _ { \mathrm { 3 D } } + 0 . 3 \mathcal { L } _ { \mathrm { r o o t } } + 0 . 5 \mathcal { L } _ { \mathrm { K D } } .\tag{5}
$$

We use smooth-L1 penalties with $\beta ~ = ~ 0 . 0 2$ for cropnormalized 2D coordinates and $\beta = 0 . 0 1$ m for 3D coordinates. The distillation targets come from the frozen offthe-shelf model. Results are in Tab. 15.

Feature distillation. The 840M-parameter backbone accounts for most of the computation. We distill it into DI-NOv3 ViT-S/16, ViT-B/16 and ViT-L/16 students through a two-layer adapter to the teacher’s 1280-dimensional feature space. The students match teacher tokens from 384 px views, area-pooled to $1 6 \times 1 6$ , using variance-normalized MSE plus cosine distance. Distillation initializes the students; it is not their final training objective. We train for 20k steps with AdamW, weight decay 0.05 and a learning rate of $2 \cdot 1 0 ^ { - 4 } .$ , multiplied by five for the adapter. The schedule uses 1k warm-up steps followed by cosine decay. We rotate crops by up to $2 5 ^ { \circ }$ with probability 0.7 and disable flips. Checkpoints are selected using the frozen decoder on FreiHAND evaluation images. The ViT-B student then trains for 12k steps at 256 px and 8k at 320 px with SAM 3D Body’s decoder, mask and verification heads.

Direct keypoints (MHR-ref). A 1.6M-parameter head on the decoder’s final tokens predicts 21 joints and 64 MHR surface points (4× upsampling, stride-4 RGB stem, softargmax 2D and heatmap-pooled 3D). It improves 2D alignment, particularly at fingertips (Tab. 15). Fitting the rig to these 2D predictions worsens 3D accuracy, so MHR-ref uses the head’s 2D and the mesh’s 3D (Sec. C.5).

## A.3. Dense pose network

The stride-4 heatmap head also carries a pooled token with a CLIFF-style crop condition [57]; bone points lie at 1/3 and 2/3 of each bone, and the first dense models predicted all 778 vertices. Least-squares translation is less stable in root depth than the MHR camera head. We therefore predict root depth explicitly, using both teacher-camera targets and ground-truth supervision.

Table 9. Pose training data at the last development milestone: crops, labels and sampling shares. 2D/3D: joints; mesh: MANO vertices; mask: SAM 2.1; KD: teacher targets. Released stages add HaGRID pseudo-labels (4.9%) and 885 hard crops (2.9%; Sec. A.8). <sup>†</sup>Converted to the MANO convention (Sec. A.4); benchmark sequences excluded.
<table><tr><td>source</td><td>crops</td><td>labels</td><td>share (%)</td></tr><tr><td>HOT3D train, every 2nd frame* [17]</td><td>150k</td><td>2D, 3D, mesh</td><td>13.8</td></tr><tr><td>COCO-WholeBody [49]†</td><td>58k</td><td>2D, mask, KD</td><td>10.6</td></tr><tr><td>CMU Panoptic hands [50]†</td><td>1.9k</td><td>2D, KD</td><td>3.2</td></tr><tr><td>HInt train [6]†</td><td>12.4k</td><td>2D, KD</td><td>6.3</td></tr><tr><td>Halpe-FullBody hands [51]†</td><td>21.7k</td><td>2D, KD</td><td>8.5</td></tr><tr><td>FreiHAND [14]†</td><td>130k</td><td>2D, 3D, mesh</td><td>5.3</td></tr><tr><td>EgoDex test split [18]†</td><td>148k</td><td>2D, 3D (MANO convention)</td><td>6.3</td></tr><tr><td>EgoDex training part 1, filtered [18]†</td><td>565k</td><td>2D, 3D (MANO convention), mask</td><td>6.3</td></tr><tr><td>VISOR [52]</td><td>17k</td><td>masks</td><td>8.5</td></tr><tr><td>background negatives</td><td>84k</td><td>no hand</td><td>5.3</td></tr><tr><td>DexYCB train [16]</td><td>48.7k</td><td>2D, 3D, mesh</td><td>5.3</td></tr><tr><td>InterHand2.6M / OakInk2 / HO-3D / HanCo [15, 53–55]</td><td>334k</td><td>2D, 3D, mesh</td><td>5.3 / 4.2 / 3.2 / 2.1</td></tr><tr><td>OakInk2 egocentric, contact-first</td><td>138.9k</td><td>2D, 3D, mesh</td><td>5.3</td></tr><tr><td>hard examples (data engine, Sec. A.8)</td><td>873</td><td>2D</td><td>0.5</td></tr></table>

Supervision. The point losses follow Eq. (1); KD sources are listed in Tab. 9. We supervise surface points with ground-truth MANO vertices where available. Otherwise, SAM 2.1 pseudo-masks [12] at 96×96 pull projected points inside the blurred mask, excluding points that the teacher mesh identifies as occluded. A one-sided Chamfer term encourages coverage of the mask boundary. Teacher-mesh, edge-length and bone-anchor priors receive weights of 0.3, 0.3 and 0.5, respectively. We cut masks a quarter hand length behind the wrist. We discard a mask if fewer than half the keypoints fall inside it or its IoU with the projected mesh is below 0.2. For the reprojection loss in Eq. (2), unlabeled joints receive half weight.

Surface basis and EgoDex filter. We project the 778- point predictions onto the lowest 256 Laplacian eigenfunctions to remove spikes. This basis reconstructs ground-truth HOT3D meshes to 0.61 mm. The point-count ablation is in Tab. 26. EgoDex’s ARKit labels [18] are tracker outputs. We retain the 58% of 975k first-part crops whose 2D disagreement with MHR-ref is below 0.12 hand sizes.

Milestone recipe (Tab. 15). Each stage resumes the previous checkpoint and uses a learning rate of $1 0 ^ { - 4 }$ unless noted.

1. We first train the dense-point and root heads on MHR features for 20k steps at 2 · 10<sup>−4</sup>, using ten sources and a weight of 0.6 for filtered EgoDex. We then train root depth for another 4k steps.

2. We add mesh and mask supervision, sampling every second HOT3D frame at 2.5× weight. We train with EgoDex SAM masks for 7.1k steps, then add FreiHAND meshes for 8k steps.

3. We add reprojection consistency with weight 1 for 8k steps. Another 8k steps add DexYCB and visibility supervision, each at weight 0.5, plus spectral smoothness beyond 256 modes and a teacher-surface prior, each at weight 1. To train verification, we present 15% of crops under the incorrect handedness hypothesis.

4. For label harmonization, we sample InterHand2.6M, OakInk2, HO-3D and HanCo at weights 0.5, 0.4, 0.3 and 0.2, respectively, and hard examples at 0.03. Egocentric degradation and occluder probabilities are 0.25 and 0.2; box jitter is 1.6, and surface reprojection has weight 0.5. This stage runs for 8k steps from the intermediate checkpoint with higher EgoDex 3D weight.

5. We initialize the released heads with 400 points and 200 spectral modes and train for 8k steps. Sampling weights are 1.3 for HOT3D, 0.05 for hard examples, and 0.5 each for egocentric OakInk2 and EgoDex with WiLoR shape targets.

## A.4. Label conventions

EgoDex’s ARKit supervision biases finger lengths and wrist position in egocentric images, including for the H+ teacher. Post-hoc bone and wrist corrections trade Ego-Standard gains for HOT3D losses, so we correct the training labels. We re-convert EgoDex against a kinematic WiLoR fit that never saw EgoDex; an EgoDex-trained reference would retain the ARKit proportions. We use kinematic MANO joints throughout and refit the teacher’s joint regressor. FreiHAND’s mesh-regressed joints differ from this convention by 3.3 mm on average and 6.9 mm at the pinky MCP. For 2D labels, we apply fixed corrections of −10.8% to Halpe knuckles, −6.5% to the COCO-WholeBody wrist and +5.6% to its thumb MCP. On 2Donly data, WiLoR’s MANO-native 3D provides the shape target, while the teacher’s image-aligned mesh supplies the 2D surface targets. We also add 334k MANO-labeled crops from InterHand2.6M [53], OakInk2 [54], HO-3D [15] and HanCo [55], together with student-side egocentric augmentation and a surface reprojection loss. Label conventions alone changed EgoStandard PA-MPJPE by 1.06 mm.

## A.5. Part segmentation

The 17 classes follow MANO’s skinning joints; the head adds 2×2 pooling and 3×3 convolutions to the design of Sec. 3.4. We render part labels from ground-truth MANO meshes on HOT3D, DexYCB, egocentric OakInk2, HanCo and FreiHAND. We clip them to SAM 2.1 masks, retaining masks with silhouette IoU of at least 0.7. On COCO-WholeBody, CMU and HInt, we render our predicted mesh instead and ignore fingers that disagree with the 2D labels. These rendered labels agree better with HOT3D reference labels than nearest-bone assignment. The head trains for 16k steps with Eq. (3). Joint-training ablations are in Tab. 7.

Labels clipped to SAM 3 [13] masks change part mIoU by at most 0.009 per dataset and lower boundary F against the SAM masks by 0.015–0.021, so the released heads keep the SAM 2.1 labels.

Parts, points and masks. On five held-out datasets, FAH-L achieves 0.46–0.54 mIoU over 16 parts; counting misses as background lowers COCO by 0.03–0.04 and the other sets by at most 0.003. The foreground union matches SAM masks at IoU 0.77–0.85. Joint training slightly worsens pose accuracy (Tab. 7), so we train the released part head on frozen pose features.

Dense-point finger assignment. We compare matched ViT-B runs on ground-truth-mesh/SAM-labeled crops from HOT3D, DexYCB and OakInk2. Each run uses 300 crops per dataset and 2k training steps. Part losses reduce wrongfinger assignments from 6.7% to 4.2–4.5%, without a clear pose-accuracy gain. We stop gradients through validity weights and omit targets for absent parts; controls use one seed and corrected runs use two. Coarse finger-region losses improve label agreement further but cover fewer points, so they are not released.

Reference-label limitations. Review of 168 selected difficult HOT3D crops found 18 occluder-label errors and 3 misalignments; this sample cannot estimate population error. Agreement with these labels does not establish anatomical accuracy.

## A.6. Contact head

Architecture and training. The contact branch pools heatmap features by a factor of four and copies the encoder’s final normalization layer. Winding numbers identify vertices inside the object for the contact labels.

We sample ARCTIC and HOI4D at weight 0.5 each and MOW at 0.05, holding out 16.5k crops for validation. HOI4D’s Chair category is excluded because its ground truth is unreliable and often penetrates the object. All HOI4D left hands belong to this category, leaving only right hands. We also reject crops with hand penetration beyond 15 mm on HOI4D or 25 mm on other sources.

The contact head uses a learning rate of $3 \cdot 1 0 ^ { - 4 }$

Selection. For each size, we compare branch learning rates of $3 \cdot 1 0 ^ { - 5 }$ and $1 0 ^ { - 4 }$ . We reject settings with a statistically significant vertex-F1 loss on any test set, then choose the best validation F1. Every size selects $1 0 ^ { - 4 }$ . Test scores therefore inform model selection.

Video model. In addition to the inputs in Sec. 3.4, the video model receives detection confidence, presence and side-normalized root position, with causal velocity and acceleration. It uses pre-norm blocks and four attention heads. Zero-initialized residuals initially reproduce the image model, and the hand flag resets after 32 absent frames.

We train for 20k AdamW steps at a learning rate of $5 \cdot 1 0 ^ { - 4 } $ , using warm-up and cosine decay. Batches contain 256 windows sampled at strides 1–3. Balanced point BCE and half-weight flag BCE form the loss. Detection dropout and noise on logits and motion inputs provide augmentation.

To obtain out-of-fold inputs, we train two image heads for 8k steps each. Each excludes half the HOT3D, DexYCB, HOI4D and ARCTIC recordings and predicts tracks for that held-out half. Inference uses the released image head. Separate validation recordings select the map blend and vertex recalibration. Flag hysteresis uses on/off thresholds of 0.90/0.10, except that FAH-S turns on at 0.55.

Training-source selection also uses test scores: we first require no significant vertex-F1 loss on HOT3D or DexYCB, then minimize loss on HOI4D and ARCTIC. This selects HOT3D and DexYCB for FAH-B, FAH-L and FAH-H, with HOI4D and ARCTIC added for FAH-S.

## A.7. Detectors

Detection recall limits every downstream pose estimator. With WiLoR’s detector, bench2d recall is 0.92 overall and 0.84 on COCO. We train our detectors with Ultralytics [36] and a cosine schedule. Horizontal flips are disabled to preserve handedness, and mosaic augmentation stops for the last three epochs.

Development pose detectors and verification. We finetune WiLoR’s YOLOv8m-pose on hand annotations from COCO-WholeBody, VISOR, HOT3D, HInt and CMU. VI-SOR contributes hand-only boxes from frames where every hand is matched. Training uses 68.4k frames with 124.4k hands, 15 epochs at 640 px and batches of 128. Inference uses 800 px to retain small hands. Retraining substantially improves held-out detection accuracy (Tab. 16). Adding synthetic low-light copies to YOLOv8s-pose preserves recall.

Verification combines detection confidence and hand presence by their geometric mean. Identity NMS then multiplies this score by the handedness score and ranks all hypotheses, including both sides of re-tried boxes. It suppresses a hypothesis if its wrist lies within half a palm length of an accepted wrist, where palm length runs from wrist to middle-finger MCP. This retains distinct hands in overlapping boxes without additional pose crops.

One-stage segmentation detectors. We initialize YOLOv8s-seg and YOLOv8m-seg from the pose detectors and train for 12 epochs at 800 px on 90.2k frames. SAM 2.1-L supplies pseudo-masks, prompted by a box and up to eight visible keypoints. We clip masks to the box and replace failed masks with an inscribed ellipse.

We then fine-tune YOLOv8m-seg for six epochs using boxes that enclose both the mask and keypoint skeleton. Hard-example training images are oversampled fourfold. This improves finger coverage and pose accuracy, with a small loss in precision (Tab. 16).

The released detector uses SAM 3 [13] masks, with SAM 2.1 or ellipse fallbacks. Text prompts add 15.9k previously unlabeled hands. To correct merged overlapping hands, we split masks along the hand skeletons and recover missing second hands. We also add two-hand training frames from HaGRID (2,938) and HOT3D/OakInk2 (4,003). We exclude 15 HOT3D training sequences and all subjects used in our test sets.

These changes improve masks and overlapping-hand recall, with some loss on small, difficult hands. Segmentation results are in Tab. 28.

## A.8. Hard-example data engine

We mine Open Images and HaGRID for cases where models disagree. The candidate pool includes our model under both handedness hypotheses and multiple rotations, MHRref, WiLoR, RTMPose-m and MediaPipe. Reviewers inspect candidate sheets under fixed rules and retain hands where our error exceeds 0.1 hand sizes. The review protocol includes repeat checks of retained hands or a random 10% sample. Holdouts are separated by person on HaGRID and by image on Open Images.

The released training pool contains 885 crops with 2D labels and no teacher targets. It is sampled at weight 0.3, accounting for 2.9% of training examples. The released hard\_mined\_v2 benchmark retains the original 169 heldout hands. This training improves the stress test without changing public-benchmark accuracy.

HaGRID pseudo-labels. We also use consensus 2D labels on 165,118 HaGRID crops covering 33 gestures. They account for 4.9% of training samples and receive teacher targets at weight 0.5.

An earlier dense model, MHR-ref’s keypoint head and RTMPose-m provide candidates. A joint is labeled only when at least two agree within 0.05 hand sizes; the label is their median. We retain sharp, uncropped hands with at least 12 labeled joints, boxes of at least 100 px, and handedness agreed by two models. People in the hard-example benchmark are excluded. Neither training pool is released.

## A.9. Mesh stage

Closed-form initialization. We split the template into 16 parts by largest skinning weight. Horn’s method [37] estimates each part’s rotation from the centered template and predicted points. Parent-relative rotations $R _ { \mathrm { p a r e n t } } ^ { \top } R _ { \mathrm { c h i l d } }$ give the initial 45-dimensional pose (Tab. 25).

Learned fitter. An MLP predicts pose, shape and root corrections from the initial pose, 195 surface points and 21 joints. Points and joints are wrist-centered in the root frame. The MLP has three 1024-wide GELU layers and a zero-initialized output.

We train it on the development model’s predictions for HOT3D and DexYCB training clips, using L1 losses on vertices, joints and centered vertices. Training uses AdamW at 10<sup>−3</sup>, batches of 512 and 20 epochs over 124.9k hands. Input noise of 3 mm is necessary for transfer (Tab. 25); this setting was fixed before testing.

Gauss–Newton step. One damped Gauss–Newton step refines the 61 parameters in fp32. It uses a Huber penalty with δ = 1 cm, weighting the 778 vertices by 0.3 and the 21 joints by 1. We compute the central-difference Jacobian with step $\mathrm { { i 0 ^ { - 2 } } }$ , batching all perturbed parameters in one compiled MANO pass. This matches the original mesh to within 0.004 mm.

Initialization, the learned fitter and refinement account for 37%, 42% and 21% of mesh-stage runtime, respectively. Rigid PnP aligns overlay meshes to the 2D joints.

MHR-Hand fit. The same dense points can drive an MHR-Hand fit using a learned initializer and one damped Gauss–Newton step with joint limits. Our hand-mode rig matches SAM 3D Body’s TorchScript MHR to within 0.001 mm. The joint regressor is fitted on training splits only, and the fit exports native SAM 3D Body/MHR parameters (Tab. 25). On HOT3D, FAH-L runs at 65 fps with either fit or 59 fps with both.

## A.10. Video refiner

Model and training. The causal pre-norm transformer has four attention heads. Each frame contributes a 110- dimensional vector containing presence, detection confidence, root-relative 3D joints, 2D joints and root position as $( x / z , y / z , \log z )$ . Missing inputs are zero-filled.

Training uses 600 HOT3D and 1,600 DexYCB clips. The loss is $e _ { \mathrm { r e l } } + e _ { \mathrm { P A } } + 3 e _ { \mathrm { a c c } } + 0 . 1 e _ { \mathrm { r o o t } }$ , with errors in mm. Track augmentation includes mirroring, in-plane rotation, image shift, log-depth offset, detection dropout and noise. AdamW training runs for 20k steps at $5 \cdot 1 0 ^ { - 4 } $ with batches of 256 windows. All released models share this refiner.

Inference. We fill gaps of at most two frames. The One Euro filter uses a 12 Hz cutoff and β = 1; handedness memory is 0.8. A separate filter on direct 2D predictions avoids errors from projecting refined depths. It uses 30 Hz and β = 0.05 per px/s, selected on HOT3D and DexYCB. The mesh follows the refined joints.

Track proposals. After ten confirmed detections, a track can propose a constant-velocity box. Presence and pose continuity must support the proposal, and detector boxes always take priority. We disable blind gap filling when no hand is found or less than 60% of the box lies inside the image. Track proposals reduce missed hands and false positives on EgoStandard with little change in pose accuracy or throughput.

## A.11. Camera model

World-space error combines hand error with camera rotation, trajectory and metric-scale errors, whose contributions depend on the dataset and camera mode (Tab. 19 and Sec. C.3). The reference is HaWoR’s offline pipeline [11]: masked DROID-SLAM [23] at 440 px on every frame with Metric3D [24] scale on every keyframe.

Causal mode. We estimate each arriving frame’s camera pose by combining DPVO, metric depth and a learned motion head.

DPVO [40] runs at 384 px on alternate frames, masking the previous hand boxes. We reinitialize from eight buffered frames after a rotation step above 14.2<sup>◦</sup> or a twofold scale jump. The rotation threshold is the 99.9th percentile on HOT3D training clips. Translation is disabled for pure rotations.

MoGe-2 [41] estimates depth in a background process every 30 processed frames, or every 15 frames per DPVO instance. Each call takes about 0.1 s and is applied five frames later. We correct depth scale by 1.034, fitted on 60 HOT3D training clips, and aggregate estimates with a running median in log space to reduce noise.

The motion head is a six-layer transformer. It combines two frames’ DINOv3 ViT-S/16 tokens at 192×256 resolution with pooled RAFT-small [42] flow, intrinsics, frameslot embeddings and hand-box embeddings. Outputs are rotation, metric translation and their log-variances. Training uses HOT3D, EgoDex part 2, egocentric ARCTIC [38] and HOI4D [39], excluding subject H3. We sample strides 1–6 and lags up to 80 frames, with Laplace likelihood and a chained-prediction loss.

Fusion normally retains DPVO’s rotation and corrects its translation scale using depth and the motion head’s 30-step path length. The path-length correction weakens as DPVO’s translation evidence increases. We measure this evidence as the median difference between rotation-only and fullpose reprojection residuals over 30 steps. The correction exponent falls from 0.5 to zero as evidence rises from 0.5 to 1.5 px.

We use the motion head’s step if DPVO loses tracking or their rotations differ by more than three times the predicted rotation error. We can also replace translation alone while keeping DPVO’s rotation. This happens when translations differ by more than ten times the head’s predicted error. The threshold drops to three times when hand boxes leave less than 70% of the patch grid unmasked, because the remaining patches often lie on hand edges or moving objects.

Offline mode. Masked DROID-SLAM uses our hand masks to estimate the trajectory. We take the geometric mean of its Metric3D scale and the causal scale estimate. If their Sim3 disagreement exceeds 5% of path length, we fall back to the causal estimate. We report three runs, pairing SLAM realizations with causal seeds. SLAM scale varies by 10–20% between runs. The full offline pipeline runs at 7.1 fps (Tab. 3).

Selection and test exposure. We develop camera settings on disjoint EgoStandard development clips and HOT3D training clips. We also check the 98 HOT3D benchmark clips and EgoDex part 2. Offline development uses nonbenchmark HOT3D clips.

The pre-fixed acceptance rule requires at least a 2% mean whole-track W and WA gain across domains, no worsening on chunks, and no domain loss above two seed standard deviations. The motion head was adopted for its development-set WA gain despite failing the W criterion.

HOT3D benchmark scores informed camera choices. Earlier configurations were also evaluated on EgoStandard test before the development split existed, so these are not untouched-test estimates. Later configurations were committed before testing. In particular, the translation rule was selected on development data and evaluated once on test.

Near-static cameras remain difficult: odometry can drift and maps can collapse. A few such tracks can reverse camera rankings between development and test (Tab. 21).

## A.12. Inference pipeline and timing

The eager decoder is limited by kernel launches and hostdevice synchronization. We reduce deployment overhead through TensorRT inference, shared frame uploads, GPU crop transforms and packed copies (Tab. 15). PyTorch remains available as a fallback.

Detection uses confidence 0.05, a fused threshold of 0.25 and wrist-based identity NMS. We train at 800 px and infer at 640 px in FP16. Pose crops span 1.1 times the box’s long side, with one re-crop when a hand reaches the border.

Timing. Image and video timing uses the desktop (i7- 14700F, WSL2, 8 threads) on 2,400 pre-decoded frames: the first 300 of four EgoStandard clips (1920×1456) and four HOT3D streams (1408×1408). Loading, decoding and 48 warm-up frames per resolution are excluded; uninstrumented runs synchronize once per frame, and separate perstage synchronization gives the shares. fps is 1000 divided by the slower of two median frame times (ms); datasetspecific tables use their own frames, and no older-GPU timing is substituted. World timing includes the camera stages on four EgoStandard clips, excludes loading, decoding and 30 warm-up frames, and uses the slower swap-free round. Detector and encoder ONNX exports use mixed FP16 with fp32 I/O, built per GPU for TensorRT 11.4; the ViT-L engine is 637 MB.

## B. Benchmarks and baselines

Crop-based baselines use their official fp32 implementations; our models use the optimized runtime (timing: Sec. A.12).

HOT3D v1 (3D). We evaluate on 98 five-second clips from the 27 validation sequences in HaWoR’s split of HOT3D train\_aria [17]. These clips are disjoint from our training data and contain 14.7k frames with 25.4k MANO hands. Aria RGB images are rectified to an upright pinhole camera.

We match detections at box IoU ≥ 0.3. Pose, mesh, root, relative-hand-position and acceleration errors are computed on matched hands; recall covers all ground-truth hands. Tables abbreviate PA-MPJPE as PA and PA-MPVPE as PVE. We normalize 2D acceleration error and jitter by hand size and compare consecutive matched frames with projected ground truth.

Following ViDiHand [25], PA-MPJPE-p (PA-p) penalizes missed hands. Each miss receives the unaligned error of a canonical, wrist-anchored, mean-shape MANO hand, approximately 130 mm.

Hands in world space. We transform camera-frame joints into world space using predicted camera poses. For each ground-truth hand, we split its matched detections into contiguous runs and discard runs shorter than 16 frames. Each method is evaluated on its own matched frames, so detection misses and run filtering can change the scored population.

For chunk metrics, we divide runs into 100-frame windows and retain the remainder. W-MPJPE100 uses a similarity alignment fitted on the first two frames; WA-MPJPE100 fits the whole window. RTE uses a rigid alignment of the wrist path and normalizes its error by groundtruth path length. All three metrics are frame-averaged. Whole-track metrics use the same definitions over each complete run.

The shared reference camera comes from HaWoR’s masked DROID-SLAM [23] pipeline with Metric3D [24] scale. W-MPJPE grows with alignment length, whereas path-normalized RTE tends to decrease with run length. We therefore compare RTE only within a dataset.

bench2d v2 (2D). This benchmark contains 6.4k images and 8.6k hands from HOT3D, CMU Panoptic, COCO-WholeBody validation and the HInt test splits. We report matched-hand PCK@0.1, fingertip PCK, recall and PCK including misses. Precision is scored only on fully annotated HOT3D; COCO annotations are not exhaustive.

The 3D fit metric measures projected 3D compatibility with visible 2D landmarks. We fit translation to the visible ground-truth joints, then compute PCK@0.1 using dataset intrinsics or an image-diagonal focal length. This is not a measure of metric depth or 3D accuracy. Crop-based mesh baselines use their own crop-space 2D predictions, which are more accurate than their reprojected meshes.

Custom sets. Joint and fingertip conventions differ across public datasets (Sec. A.4). We use a common modelassisted review procedure for the challenging-pose, mined, low-light and shadow sets. Reviewers inspect all methods’ candidates on original and brightened images. If none is usable, they prompt our model with a box under both handedness hypotheses. Doubtful cases receive further review, and decisions are stored with the labels.

These sets informed development and may favor methods in the candidate pool. We report recall, PCK@0.05 and PCK@0.1, fingertip PCK@0.1, PCK including misses and handedness accuracy. Matching requires mean visible-joint distance within half a hand size.

Challenging poses (hard\_pose\_v1). The challengingpose set contains 175 hands on six stock-photo sheets covering tool grips, gestures, fists and sign-language poses. Labels start from a five-method consensus. We re-label contested or rarely detected hands using box prompts and review every label.

![](images/ce2773234839791709d90069e6db5128c98044c1ad161941b9136dea2ad539b9.jpg)  
Figure 4. Custom benchmark examples. Images 1–4 show shadows across the hand or beside it; 5–8 show synthetic low light with the original image inset. Ground-truth keypoints are overlaid. Challenging-pose stock photos are omitted because their licenses do not permi redistribution. All shown photos are CC BY 2.0; sources, authors and modifications: Sec. D.

Hard in-the-wild examples (hard\_mined\_v2). This set contains 169 hands from held-out data-engine images (Sec. A.8) where an earlier model erred by more than 0.1 hand sizes. Most failures involve truncated fingers or incorrect handedness. It is a failure-mined stress test.

Low light (dark\_v1). We evaluate on 19 real dark photos with 50 hands (one unlabeled), plus a synthetic low-light version of bench2d v2. Synthesis preserves the original labels with exposure reduced to 0.05–0.15 of the original in linear light. We also add sensor noise, color shifts, gamma changes, blur and JPEG compression (Fig. 4). The released set contains only the synthetic images because the real photos have unknown licenses.

Shadows (shadow\_v1). We collect images with shadows across the hand or beside it from Open Images, HaGRID [58], Openverse and Wikimedia Commons. CLIP [59] ranking and manual review yield 170 images with 278 scored hands. Labels are per-joint medians of reviewer-selected predictions from our model, MHR-ref, WiLoR, HaMeR and RTMPose.

EgoStandard. EgoStandard [43] provides head-camera manipulation videos with hardware-assisted tracking of both hands and camera poses. The test set takes one 30 s clip per test task (tasks chosen by a hash of the task name) from the left head camera: 120 clips, 105.9k frames. Targets are MANO fits to the tracked keypoints, boxes are projected meshes, and world-space metrics use each method’s own true positives; no frames are shown. A disjoint dev split (80 clips from 182 other tasks) selects camera-model and track-proposal settings. Earlier pose milestones, labelconvention studies and camera configurations were scored on the test set, so it is not an untouched test set (Sec. A.11); each final released pose model was scored once, without selection among the four models on test scores.

DexYCB (third person). We use the seen-subjects (s0) test split of DexYCB [16]. Each sequence is evaluated once, with cameras chosen to balance camera and object frequencies. The resulting set has 160 clips and 9.3k scored hands.

We detect and match hands as on HOT3D rather than using ground-truth boxes.

Contact. Table 4 evaluates contact with ground-truth boxes. It uses 5,172 HOT3D hands sampled every fifth frame and 9,506 DexYCB s0 test hands. HACO is zeroshot on HOT3D and trained on DexYCB s0 train.

Full-pipeline evaluation includes 25,431 HOT3D and 9,319 DexYCB hand-frames. Object-distance rules use the detected subsets, with 25,378 and 9,225 hands, respectively. Each FAH size has one contact model with validation-calibrated thresholds. These evaluation sets informed model selection (Secs. A.6 and C.7).

Ground-truth contact marks MANO vertices within 10 mm of or inside the ground-truth object mesh (winding number; any penetration depth); thresholds use separate validation data (Sec. A.6). Pooled vertex F1 aggregates all hands, HACO’s per-hand vertex F1 averages over contact-positive hands, and hand F1 scores the separate flag. Onset and release windows lie within three frames of a switch, with five stable frames on each side; hovering means no contact at 10–30 mm object distance; flips count flag switches.

EgoDex part 2 (model selection) and VISOR. We use 4,000 crops from EgoDex’s second training part [18] for pose-model selection. No pose network trains on these crops. We convert labels to kinematic MANO joints and score PA-MPJPE with ground-truth boxes. VISOR [52] validation frames with hand-plus-arm masks evaluate segmentation.

Baselines. WiLoR [7], HaMeR [6], Hamba [8] (on its initial-regressor joints) and the AnyHand [9] checkpoints of WiLoR and HaMeR (released configurations) run per frame with WiLoR’s detector (confidence 0.3, NMS 0.5) and the true intrinsics; HaWoR [11] runs on whole clips. MediaPipe HandLandmarker [46] (MediaPipe Hands in the tables) runs in image mode with up to four hands; its metric hand-centered landmarks are rotated to the viewing ray and placed by least squares onto its 2D. The RTMPose-m hand model [47] (RTMDet-nano at 320 px, ONNX Runtime) and the whole-body RTMW-l [48] (YOLOX-m person detector)

are 2D only. EgoHandICL [60] reports P-MPJPE 4.0 mm on its own ARCTIC split, on detected hands only, with no released model or test lists, so it is not in our tables.

## C. Extended results

This section expands the main results with development ablations and design studies. A dash indicates an unavailable result. Unless noted, throughput is measured on HOT3D with an RTX 4070 Ti SUPER, using the slower of two rounds (Sec. A.12). For SAM 3D Body, <sup>§</sup>marks our CUDAgraph implementation; eager inference runs at 5.9 fps.

## C.1. Image model

Table 10. Additional HOT3D metrics. Detection recall and rootrelative MPJPE complement the primary errors in Tab. 1.

<table><tr><td>method</td><td></td><td>recall ↑ rr (mm) ↓</td></tr><tr><td>per frame</td><td></td><td></td></tr><tr><td>MediaPipe Hands [46]</td><td>0.58</td><td>54.43</td></tr><tr><td>HaMeR [6]</td><td>0.96</td><td>57.03</td></tr><tr><td>HaMeR + AnyHand [9]</td><td>0.96</td><td>49.56</td></tr><tr><td>Hamba [8]</td><td>0.96</td><td>56.49</td></tr><tr><td>WiLoR [7]</td><td>0.96</td><td>17.80</td></tr><tr><td>WiLoR + AnyHand [9]</td><td>0.96</td><td>19.81</td></tr><tr><td>SAM 3D Body hand [10]</td><td>0.96</td><td>28.30</td></tr><tr><td>FAH-L</td><td>1.00</td><td>7.20</td></tr><tr><td>video</td><td></td><td></td></tr><tr><td>MediaPipe Hands (video mode)</td><td>0.82</td><td></td></tr><tr><td>HaWoR [11]</td><td>0.93</td><td>14.21</td></tr><tr><td>FAH-L video</td><td>1.00</td><td>7.38</td></tr></table>

Accuracy and coverage (Tabs. 10 to 12). AnyHand checkpoints give HOT3D PA-MPJPE of 8.19 mm for HaMeR and 6.70 mm for WiLoR. Their per-dataset and stress-test results are included here; the original checkpoints’ primary scores are in Tabs. 1 and 2.

Data-mix transfer (Tab. 13). More third-person data improves public 3D and challenging-pose results but hurts egocentric tracking-label benchmarks. Increasing Frei-HAND weight drives this trade-off; hard-example and Ha-GRID additions alone do not. We therefore retain the original mixture and use held-out EgoDex part 2 for egocentric selection.

The released pose models omit MOW, ARCTIC and HOI4D pose labels; these datasets are used for contact supervision as described in Sec. A.6.

Released models. The encoders and input sizes are defined in Sec. 3.2. All initialize from the ViT-B development heads and share two 12k-step blocks followed by 6k steps with WiLoR targets, at learning rates $1 0 ^ { - 4 } / 5 \cdot 1 \dot { 0 } ^ { - 5 } /$ $3 \cdot 1 0 ^ { - 5 }$ . The part head then trains on frozen features.

Backbone and resolution (Tab. 14). DexYCB gains from higher resolution are smaller than HOT3D gains. The supplementary comparison adds mesh errors and stress-test scores to Tab. 6.

Image milestones (Tab. 15). Mesh and mask supervision, reprojection consistency and label harmonization improve complementary aspects of accuracy. The learned MANO fit provides the largest mesh gain; runtime changes nearly double throughput without changing accuracy.

Hard in-the-wild examples. Detection, re-cropping and mask-derived boxes account for most gains on this failuremined set. Person-disjoint HaGRID gestures still share pose patterns with training, so these results are a stress test rather than neutral generalization evidence. Edge-on fingers remain difficult.

Detectors (Tab. 16). Retraining, verification and higherresolution training improve recall. Mask-derived boxes further improve coverage; a higher fused threshold trades recall for precision. Detector settings are detailed in Sec. A.7.

## C.2. Video model

Refiner design (Tab. 18). Joint and root losses alone can worsen articulation, motivating the Procrustes term and partial blend. Geometric augmentation improves position invariance; in-plane rotation reduces the third-person rootcorrection bias from 13.4 to 1.7 cm. The released refiner uses both augmentations.

## C.3. Camera model and hands in world space

Where the world-space error comes from (Tab. 19). Metric scale explains much of the gap to ground-truth cameras; replacing rotations alone does not help. Translation under-scaling in the motion head motivates the parallax gate. These diagnostics separate camera limitations from hand-reconstruction error.

Track coverage (Tab. 20). We report median and 90thpercentile run lengths. Top-10 measures the fraction of W error from the ten worst runs. Common-set W/WA averages three runs on hand-frames detected by both FAH-L and Ha-WoR. On this shared set, paired 95% bootstrap intervals for causal and offline gains exclude zero.

Dyn-HaMR. Dyn-HaMR [61] stages I–II worsened its initializer on HOT3D pilots despite camera corrections. Stage III requires fixed 128-frame tracks, so we exclude it from the variable-length world comparison.

Table 11. Additional bench2d v2 metrics. We report fingertip and per-dataset PCK@0.1, overall recall and recall on overlapping hands. Aggregate PCK and accuracy including misses are in Tab. 2.
<table><tr><td>method</td><td>tips</td><td>HOT3D</td><td>CMU</td><td>COCO</td><td>HInt-ND</td><td>HInt-EPIC</td><td>recall</td><td>overlap recall</td></tr><tr><td>MediaPipe Hands [46]</td><td>0.59</td><td>0.61</td><td>0.73</td><td>0.64</td><td>0.66</td><td>0.68</td><td>0.51</td><td>0.43</td></tr><tr><td>RTMPose-m [47]</td><td>0.67</td><td>0.69</td><td>0.73</td><td>0.75</td><td>0.72</td><td>0.72</td><td>0.56</td><td>0.31</td></tr><tr><td>RTMW-1 [48]</td><td>0.60</td><td>0.64</td><td>0.66</td><td>0.69</td><td>0.71</td><td>0.72</td><td>0.94</td><td>0.94</td></tr><tr><td>HaMeR [6]</td><td>0.64</td><td>0.72</td><td>0.73</td><td>0.75</td><td>0.72</td><td>0.66</td><td>0.92</td><td>0.80</td></tr><tr><td>HaMeR + AnyHand [9]</td><td>0.60</td><td>0.72</td><td>0.67</td><td>0.71</td><td>0.64</td><td>0.58</td><td>0.92</td><td>0.80</td></tr><tr><td>Hamba [8]</td><td>0.64</td><td>0.72</td><td>0.72</td><td>0.75</td><td>0.72</td><td>0.69</td><td>0.92</td><td>0.80</td></tr><tr><td>WiLoR [7]</td><td>0.64</td><td>0.65</td><td>0.72</td><td>0.75</td><td>0.70</td><td>0.69</td><td>0.92</td><td>0.80</td></tr><tr><td>WiLoR + AnyHand [9]</td><td>0.65</td><td>0.66</td><td>0.72</td><td>0.75</td><td>0.70</td><td>0.70</td><td>0.92</td><td>0.80</td></tr><tr><td>SAM 3D Body hand [10]</td><td>0.61</td><td>0.69</td><td>0.53</td><td>0.56</td><td>0.56</td><td>0.61</td><td>0.92</td><td></td></tr><tr><td>FAH-L</td><td>0.77</td><td>0.95</td><td>0.77</td><td>0.74</td><td>0.79</td><td>0.80</td><td>0.98</td><td>0.84</td></tr></table>

Table 12. Additional stress-test metrics. PCK@0.05 and fingertip PCK@0.1 (tips) complement the results in Tab. 2.
<table><tr><td rowspan="2">method</td><td colspan="2">challenging poses</td><td colspan="2">low light</td><td colspan="2">shadows</td></tr><tr><td>PCK@0.05</td><td>tips</td><td>PCK@0.05</td><td>tips</td><td>PCK@0.05</td><td>tips</td></tr><tr><td>MediaPipe Hands [46]</td><td>0.75</td><td>0.91</td><td>0.34</td><td>0.57</td><td>0.70</td><td>0.88</td></tr><tr><td>RTMPose-m [47]</td><td>0.60</td><td>0.85</td><td>0.40</td><td>0.64</td><td>0.83</td><td>0.92</td></tr><tr><td>RTMW-1 [48]</td><td>0.72</td><td>0.76</td><td>0.36</td><td>0.57</td><td>0.71</td><td>0.83</td></tr><tr><td>HaMeR [6]</td><td>0.77</td><td>0.92</td><td>0.36</td><td>0.62</td><td>0.90</td><td>0.95</td></tr><tr><td>HaMeR + AnyHand [9]</td><td>0.71</td><td>0.87</td><td>0.26</td><td>0.55</td><td>0.82</td><td>0.91</td></tr><tr><td>Hamba [8]</td><td>0.79</td><td>0.92</td><td>0.37</td><td>0.62</td><td>0.89</td><td>0.94</td></tr><tr><td>WiLoR [7]</td><td>0.79</td><td>0.92</td><td>0.34</td><td>0.62</td><td>0.91</td><td>0.94</td></tr><tr><td>WiLoR + AnyHand [9]</td><td>0.82</td><td>0.92</td><td>0.32</td><td>0.59</td><td>0.88</td><td>0.94</td></tr><tr><td>FAH-L</td><td>0.92</td><td>0.98</td><td>0.56</td><td>0.74</td><td>0.95</td><td>0.98</td></tr></table>

Table 13. Third-person data against tracking ground truth (ViT-B development setup; PA-MPJPE in mm; hard poses: PCK@0.05). EgoStandard exposure: Sec. 5.
<table><tr><td>Dex- model YCB HOT3D</td><td>poses</td><td>hard EgoDex Ego- p. 2 Std.</td></tr><tr><td>original mix 5.68</td><td>4.41 0.91</td><td>6.27 7.08</td></tr><tr><td>+ 3rd person 5.61</td><td>4.39 0.92</td><td>6.997.42</td></tr></table>

Table 14. Additional backbone and resolution metrics. The configurations are those of Tab. 6. Errors are in mm; challengingpose and low-light scores use PCK@0.1.
<table><tr><td colspan="3">HOT3D DexYCB hard dark</td></tr><tr><td>enc.</td><td>px</td><td>PVE PA PVE PCK</td></tr><tr><td>ViT-S</td><td>256 320 448</td><td>5.09 6.45 6.11 0.98 0.76 4.99 6.42 6.09 0.98 0.76 4.94 6.39 6.06 0.98 0.76</td></tr><tr><td>ViT-B</td><td>256 320 448</td><td>4.45 5.84 5.52 0.99 0.79 4.385.82 5.51 0.99 0.79 4.34 5.79 5.47 0.99 0.79</td></tr><tr><td>ViT-L</td><td>256 320 448</td><td>4.17 5.55 5.25 0.99 0.81 4.11 5.54 5.24 0.99 0.81 4.065.52 5.22 0.99 0.80</td></tr><tr><td>ViT-H+ 448</td><td></td><td>3.99 5.33 5.05 0.99 0.80</td></tr></table>

## C.4. Held-out video benchmarks

DexYCB (Tab. 22). WiLoR retains the best articulation and mesh accuracy, while FAH-L video has the best temporal metrics. Precision is omitted because the second hand is unannotated.

## C.5. Points versus parameters

Against an MHR head (Tab. 24). Reprojection consistency narrows the gap between direct 2D and fitted 3D projections. The supplementary comparison also reports root placement, relative hand position and runtime for the development models.

Mesh fitting (Tab. 25). Input-noise augmentation is essential for transfer to EgoStandard (Tab. 25).

Surface points (Tab. 26). The 400-point model gives the best mesh accuracy at similar throughput. Fewer points increase the representation floor and lose surface detail.

Agreement with the image (Tabs. 5 and 8). Distances are normalized by ground-truth hand size; mesh joints use the MANO regressor and fingertip vertices. Rigid PnP improves joint alignment but worsens surface alignment. Adding surface points to PnP can shrink close-up fingers, so it is not the default.

Table 15. Image-model milestones. Each row adds to the preceding configuration (Sec. A). Errors are in mm; 2D scores use PCK@0.1. Top rows use YOLOv8m-pose at 800 px, except mined and EgoStandard evaluations, which use the one-stage detector. Bottom rows use the one-stage detector throughout. Dense-row PVE evaluates the raw surface. Timing uses CUDA graphs; <sup>†</sup>marks compiled 384 px inference.
<table><tr><td rowspan="2">milestone</td><td colspan="4">HOT3D v1</td><td colspan="2">bench2d v2</td><td rowspan="2">hard</td><td rowspan="2">dark</td><td rowspan="2">mined</td><td colspan="2">EgoStd.</td><td rowspan="2"></td></tr><tr><td></td><td>PA PA-P</td><td>PVE</td><td>root</td><td>PCK 3D fit</td><td>PCK</td><td>PCK</td><td>PCK PA</td></tr><tr><td>pose network</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SAM 3D Body hand, off the shelf</td><td>8.30</td><td></td><td>8.49</td><td>40.00</td><td>0.61</td><td></td><td></td><td></td><td></td><td></td><td></td><td>10§</td></tr><tr><td>fine-tuned hand-only teacher (DINOv3-H+)</td><td>4.97</td><td></td><td>5.67</td><td>11.25</td><td>0.81</td><td>0.84</td><td></td><td></td><td></td><td></td><td></td><td>18†</td></tr><tr><td>ViT-B student, MHR decoder and keypoint head</td><td>4.85</td><td>5.44</td><td>5.70</td><td>11.59</td><td>0.83</td><td>0.85</td><td>1.00</td><td>0.80</td><td></td><td>7.59</td><td>8.09</td><td>31</td></tr><tr><td>decoder-free dense head and root depth</td><td>4.79</td><td>5.37</td><td>5.69</td><td>9.81</td><td>0.84</td><td>0.76</td><td>0.99</td><td>0.82</td><td></td><td></td><td></td><td>54</td></tr><tr><td>+ meshes and masks</td><td>4.57</td><td>5.16</td><td>5.62</td><td>8.84</td><td>0.84</td><td></td><td>0.98</td><td>0.82</td><td></td><td>8.35</td><td>10.66</td><td></td></tr><tr><td>+ reprojection consistency</td><td>4.38</td><td>4.96</td><td>5.59</td><td>8.67</td><td>0.84</td><td>0.82</td><td>0.98</td><td>0.82</td><td>0.47</td><td>8.15</td><td>10.70</td><td></td></tr><tr><td>+ label conventions</td><td>4.53</td><td>5.11</td><td>5.80</td><td>8.79</td><td>0.84</td><td>0.82</td><td>0.99</td><td>0.82</td><td></td><td></td><td></td><td></td></tr><tr><td>+ 400 surface points and data</td><td>4.40</td><td>4.99</td><td>5.66</td><td>8.73</td><td>0.84</td><td>0.82</td><td>0.99</td><td>0.82</td><td></td><td></td><td></td><td></td></tr><tr><td>image model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ one-stage detector, verification, crops</td><td>4.40</td><td>4.67</td><td>5.66</td><td>9.08</td><td>0.83</td><td>0.82</td><td>0.99</td><td>0.80</td><td></td><td>0.98 7.08</td><td>9.08</td><td>46</td></tr><tr><td>+ learned MANO fit</td><td>4.40</td><td>4.68</td><td>4.08</td><td>9.07</td><td>0.83</td><td>0.82</td><td>0.99</td><td>0.80</td><td></td><td>0.98 7.08</td><td>7.28</td><td>46</td></tr><tr><td>+ fast inference</td><td>4.41</td><td>4.65</td><td>4.08</td><td>9.10</td><td>0.83</td><td>0.82</td><td>0.99</td><td>0.81</td><td></td><td>0.98 7.08</td><td>7.28</td><td>85</td></tr><tr><td>+ ViT-L student and fingertips: FAH-L</td><td>4.31</td><td>4.58</td><td>4.02</td><td>8.89</td><td>0.83</td><td>0.83</td><td>0.99</td><td>0.81</td><td></td><td>0.98 7.10</td><td>7.12</td><td>65</td></tr></table>

![](images/d7a33226c5bfaaaa6c2d94c7f07195a03c26595c09290fa6d484f46b9649ec7c.jpg)  
Figure 5. SAM 3D Body hand branch versus FAH-L on bench2d v2 (HOT3D, COCO, HInt; ninth: synthetic low light). Scores: per hand PCK@0.1, projected HOT3D GT/human labels elsewhere, averaged for two hands. Each stratum shows the first correctly sided hand found by both in fixed random order; outdoor/blur additionally require ≥0.1 FAH gain. FAH overlays use dense surfaces without fitting; the timed pipeline includes the fit. Failures: Fig. 7; sources: Sec. D.

Table 16. Detector milestones. Recall with an early dense pose model, verification and 800 px input; s/m: small/medium, pose/seg: keypoint/segmentation detector. Mask-bone boxes combine masks and keypoint bones (Sec. A.7). fps measures box-only detection on 1408<sup>2</sup> frames; WiLoR uses 640 px.
<table><tr><td>detector</td><td></td><td>fps HOT3D</td><td>2D</td><td>hard</td><td>dark</td></tr><tr><td>WiLoR&#x27;s v8m-pose @640</td><td>123</td><td>0.96</td><td>0.92</td><td>0.84</td><td>0.81</td></tr><tr><td>v8m-pose + verification</td><td>100</td><td>1.00</td><td>0.95</td><td>0.93</td><td>0.89</td></tr><tr><td>v8s-pose, low light</td><td>158</td><td>1.00</td><td>0.97</td><td>0.93</td><td>0.95</td></tr><tr><td>v8s-seg</td><td>143</td><td>1.00</td><td>0.97</td><td>0.92</td><td>0.95</td></tr><tr><td>v8m-seg</td><td>87</td><td>1.00</td><td>0.97</td><td>0.94</td><td>0.95</td></tr><tr><td>v8m-seg, mask-bone boxes</td><td>93</td><td>1.00</td><td>0.97</td><td>0.95</td><td>0.95</td></tr></table>

Fingertips (Tab. 27). The fitted MANO mesh has little median fingertip shortfall. The mask-guided step improves direct 2D and dense-surface extent at a small runtime and HOT3D mesh-accuracy cost, with little change on EgoStandard. It is independent of the part head and applies only to distal segments spanning at least six mask cells.

## C.6. Detection and instance masks in one stage

The detector masks cover the hand rather than the forearm; Tab. 28 compares the released pipeline with segmentation baselines.

Timing and prompts. We time one of four shards from 3,682 decoded 1920×1080 frames. Synchronized mask calls include image encoding and full-resolution output, but exclude loading and three warm-up calls. We compute fps from the reciprocal of the median of chunk-median runtimes.

Table 17. Video-model milestones. Errors are in mm; acceleration and jitter are in mm/frame<sup>2</sup>. Top rows add components in order, using the image model’s hands; MANO fitting begins at the fourth row. Bottom rows replace one design choice at a time. “Blend 1” uses fully refined articulation, and “det. 2nd frame” runs detection on alternate frames. <sup>p</sup>Results use stored per-frame predictions.
<table><tr><td></td><td colspan="5">HOT3D v1</td><td colspan="4">EgoStandard</td><td colspan="4">DexYCB</td></tr><tr><td>milestone</td><td>PA</td><td>PA-p</td><td>PVE</td><td>accel</td><td>jitter</td><td>PA</td><td>PVE</td><td>accel</td><td>jitter</td><td>PA</td><td>PVE</td><td>accel fps</td><td></td></tr><tr><td>per-frame hands</td><td>4.40</td><td>4.67</td><td>5.66</td><td>4.13</td><td>4.46</td><td>7.08</td><td>9.08</td><td>4.53</td><td>4.60</td><td>5.72</td><td>6.82</td><td>6.94</td><td>46</td></tr><tr><td>+ online refiner</td><td>4.25</td><td>4.43</td><td>5.67</td><td>2.37</td><td>2.71</td><td>7.10</td><td>9.08</td><td>2.28</td><td>2.31</td><td>5.58</td><td>6.82</td><td>5.07</td><td></td></tr><tr><td>+ One Euro</td><td>4.25</td><td>4.45</td><td>5.64</td><td>1.73</td><td>1.95</td><td>7.12</td><td>9.08</td><td>1.81</td><td>1.77</td><td>5.59</td><td>6.81</td><td>4.38</td><td>42</td></tr><tr><td>+ MANO fit on every frame</td><td>4.25</td><td>4.46</td><td>4.00</td><td>1.72</td><td>1.95</td><td>7.12</td><td>7.30</td><td>1.81</td><td>1.77</td><td>5.60</td><td>5.39</td><td>4.43</td><td>43</td></tr><tr><td>+ fast-motion, 2D filter</td><td>4.23</td><td>4.39</td><td>3.99</td><td>1.75</td><td>2.01</td><td>7.13</td><td>7.30</td><td>1.80</td><td>1.77</td><td>5.53</td><td>5.32</td><td>4.29</td><td>80</td></tr><tr><td>+ ViT-L hands: FAH-L video</td><td>4.16</td><td>4.38</td><td>3.95</td><td>1.65</td><td>1.92</td><td>7.17</td><td>7.15</td><td>1.67</td><td>1.66</td><td>5.29</td><td>5.10</td><td>3.93</td><td>59</td></tr><tr><td>earlier-model refinerP</td><td>4.29</td><td>4.50</td><td></td><td>2.65</td><td></td><td>7.20</td><td></td><td>2.19</td><td></td><td>5.87</td><td></td><td>4.95</td><td></td></tr><tr><td>offline refinerP</td><td>4.33</td><td>4.51</td><td></td><td>3.85</td><td></td><td>7.08</td><td></td><td>4.29</td><td></td><td>5.66</td><td></td><td>6.41</td><td></td></tr><tr><td>blend 1</td><td>4.27</td><td>4.46</td><td>5.65</td><td>1.63</td><td>1.79</td><td>7.17</td><td>9.08</td><td>1.81</td><td>1.74</td><td>5.60</td><td>6.82</td><td>4.23</td><td></td></tr><tr><td>det. 2nd frame</td><td>4.27</td><td>4.77</td><td>5.69</td><td>3.06</td><td>3.37</td><td>7.12</td><td>9.09</td><td>2.91</td><td>2.95</td><td>5.57</td><td>6.80</td><td>5.65</td><td></td></tr></table>

Table 18. Video module design on an early dense ViT-B model (HOT3D v1 PA-MPJPE / PA-MPJPE-p in mm, accel in mm/frame<sup>2</sup>). 3rd p.: root correction (cm) applied to a downwardpointing hand in a third-person video.
<table><tr><td rowspan="2">training</td><td colspan="3">online</td><td colspan="3">offline</td><td rowspan="2">3rd p.</td></tr><tr><td></td><td>PA PA-p</td><td>accel</td><td>PA</td><td>PA-p</td><td>accel</td></tr><tr><td>per frame</td><td>4.79</td><td>5.37</td><td>4.23</td><td>4.79</td><td>5.37</td><td>4.23</td><td></td></tr><tr><td>refiner</td><td>4.34</td><td>4.71</td><td>2.11</td><td>4.24</td><td>4.54</td><td>2.47</td><td>13-16</td></tr><tr><td>+ geometric aug.</td><td>4.23</td><td>4.60</td><td>2.18</td><td>4.14</td><td>4.45</td><td>2.46</td><td>13.4</td></tr><tr><td>+ rotation aug.</td><td>4.35</td><td>4.74</td><td>2.22</td><td></td><td></td><td></td><td>1.7</td></tr></table>

Table 19. Camera error diagnostics (W / WA, mm; EgoStandard dev, seed 0; an earlier FAH-L video model supplies hand masks and scored hands). Each row substitutes a ground-truth component into the reference camera; a diagnostic, not a development progression, so no best marks.
<table><tr><td>camera (EgoStandard dev)</td><td>100-frame chunks</td><td>whole tracks</td></tr><tr><td>causal camera, 5 scale calls</td><td>60.30 / 21.81</td><td>189.38 / 53.58</td></tr><tr><td>+ GT rotations</td><td></td><td>64.33 / 22.79195.52 / 54.85</td></tr><tr><td>+ GT metric scale per clip</td><td></td><td>48.27 / 19.47121.71 / 46.36</td></tr><tr><td>+ GT scale per 81-frame segment</td><td></td><td>42.87 / 18.60108.69 / 38.99</td></tr><tr><td>GT cameras</td><td></td><td>32.50 / 13.96 64.64 / 16.37</td></tr></table>

Table 20. World-space tracks on EgoStandard. Run lengths are in frames; common-set W/WA errors are in mm. Protocol: Sec. B.
<table><tr><td>hands</td><td>camera</td><td></td><td>runs med. / p90 top-10</td><td></td><td>common</td></tr><tr><td>FAH-L</td><td>DROID-SLAM 326</td><td></td><td>900/900</td><td></td><td>0.55 203.00 / 46.13</td></tr><tr><td>FAH-L</td><td>FAH-camera</td><td>326</td><td>900 / 900</td><td></td><td>0.24 141.31 / 38.67</td></tr><tr><td>FAH-L</td><td>offline</td><td>326</td><td>900/900</td><td></td><td>0.30124.32/35.36</td></tr><tr><td></td><td>HaWoR [11] DROID-SLAM</td><td>677</td><td>137 / 900</td><td></td><td>0.45 210.46 / 53.59</td></tr><tr><td>WiLoR [7]</td><td>DROID-SLAM</td><td>461</td><td>343 / 900</td><td>0.45</td><td></td></tr></table>

SAM 2.1/3 use eager PyTorch in bf16; VISOR-HOS and Hands23 use their full Detectron2 pipelines in fp32. SAM box prompts come from the released detector, whose runtime is excluded; the text prompt is “hand.” For FAH-L, timing includes pose estimation and mask export on every frame. The detector alone uses TensorRT FP16 at 640 px.

Against segmentation models (Tab. 28). The pipeline exports masks with pose faster than the segmentation baselines, but most achieve higher VISOR accuracy. VISOR-HOS also benefits from training on VISOR and predicting the forearm.

## C.7. Contact

HOI4D evaluation excludes Chair and holds out subject H3, giving 7,280 crops from 149 sequences. ARCTIC validation contains only 612 hands from four sequences, so uncertainty is high. MOW uses a 35 mm tolerance and HACO threshold 0.52.

Qualitative selection (Fig. 3). Examples use groundtruth boxes and the same ground-truth MANO mesh, viewed from the contact side; contact labels use the 10 mm rule of Tab. 4. From a fixed random order, we select the first hand with at least 30 contact vertices for each grasp type (HOT3D pinch, power, fingers; DexYCB pinch, power), without repeating objects, followed by the first FAH-L failure with vertex F1 below 0.3.

Training data (Tab. 30). Adding ARCTIC and HOI4D improves their held-out domains and MOW without a resolved pooled vertex-F1 change on HOT3D or DexYCB. These are in-domain training gains, not zero-shot transfer.

Table 21. Standard deviations of world-space errors over three runs. Means are in Tab. 3. Runs vary camera seeds, SLAM realizations or both; the extra mask-control row uses our hand masks in HaWoR’s camera pipeline. Units match the main table.
<table><tr><td colspan="4">HOT3D v1</td><td colspan="3">EgoStandard</td></tr><tr><td></td><td>camera</td><td>chunks W WA</td><td>whole</td><td>chunks</td><td>whole</td><td></td></tr><tr><td>hands</td><td></td><td></td><td>W WA RTE</td><td>W WA</td><td></td><td>W WA RTE</td></tr><tr><td>WiLoR [7]</td><td>DROID-SLAM</td><td>0.19 0.050.25 0.070.02 4.26 0.23 31.751.00 0.06</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>HaWoR [11] DROID-SLAM</td><td>0.21 0.05 0.26 0.07 0.02 4.11 0.25 28.90 1.21 0.08</td><td></td><td></td><td></td><td></td></tr><tr><td>FAH-L video DROID-SLAM</td><td></td><td>0.21 0.07 0.41 0.12 0.02 4.79 0.23 41.97 1.08 0.06</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>FAH-L video DROID-SLAM, our masks 0.34 0.09 0.40 0.12 0.04 1.66 0.41 18.97 1.42 0.02 0.43 0.31 0.69 0.38 0.30 0.18 0.08 3.71 1.64 0.02</td><td></td><td></td><td></td><td></td></tr><tr><td>FAH-L world FAH-camera</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>FAH-L world DROID-SLAM</td><td>0.29 0.18 0.58 0.24 0.17 0.92 0.61 3.57 1.98 0.02</td><td></td><td></td><td></td><td></td></tr></table>

![](images/b5c52bd08bf9ec8942a059f24d4bb0c11f54796e5112f795d8c8cb42e34172ec.jpg)

Figure 6. Selected FAH-L contact examples. Panels 1–5 and 10–12 are from training-disjoint HOT3D; 6–9 are from COCO. Each shows fitted-mesh contact, a palm view and ground truth where available. Gray to violet indicates increasing probability, with a calibrated threshold of 0.5. Failures include false contact above a vase (10), missed occluded fingertip contact (11) and missed stick contact without a ground-truth object pose (12). Credits are in Sec. D.  
![](images/4ef9a5cda37c8fc148e8730f4ebb6841c73893daa68b184a041b9f39f295c6f8.jpg)  
Figure 7. FAH-H failures, with the five outputs of Fig. 1. Piano: flattened left mesh/missed key contact; guitar: tangled fretting-hand joints; soldering: coarse meshes; phone: parts spill onto the screen and label occluded fingers; dough: missed right-hand contact; knife: coarse mesh. Sources: Wikimedia Commons and, bottom row, EPIC-KITCHENS-100 via HInt; public test splits, full credits in Sec. D.

## D. Image credits

Wikimedia Commons photos, left to right and top to bottom. Figure 1: keyboard, Cangjie6 (“Keyboard with fingers”, CC BY-SA 4.0); knitting, Sadarama (“Oma strickt Strümpfe”, CC BY-SA 4.0); pottery, Earthen Potter (“Earthen Potter Hands”, CC BY-SA 4.0); Lego, Nenad Stojkovic (“Playing with Lego bricks”, CC BY 2.0); Rubik’s cube, theilr (“Rubik cube (345056969)”, CC BY-SA 2.0). Figure 7: piano, Quincena Musical (“Mikhail Shehtman (hands playing a piano)”, CC BY 2.0); guitar, Nenad

Table 22. DexYCB (s0 test, 160 clips, 9.3k hands; mm, accel and jitter in mm/frame<sup>2</sup>). <sup>†</sup>trained on DexYCB; <sup>§</sup>with DexYCB training crops; <sup>‡</sup>refiner also trained on DexYCB training tracks.
<table><tr><td>method</td><td>recall PA PA-p PVE accel jitter root</td></tr><tr><td>WiLoR† [7]</td><td>1.00 4.875.14 4.665.634.52 52.05</td></tr><tr><td>FAH-L §</td><td>0.99 5.406.49 5.13 6.04 4.95 25.92</td></tr><tr><td>FAH-L video §‡</td><td>0.99 5.29 5.91 5.10 3.93 2.25 23.87</td></tr></table>

Table 23. Additional EgoStandard camera-frame metrics. Root-relative MPJPE and relative hand position (mm), plus detection coverage, complement the primary results in Tab. 1.
<table><tr><td>method</td><td>rr</td><td>MRRPE</td><td>recall</td><td>prec.</td></tr><tr><td>WiLoR [7]</td><td>28.10</td><td>44.38</td><td>0.99</td><td>0.98</td></tr><tr><td>FAH-L</td><td>17.05</td><td>32.97</td><td>1.00</td><td>0.98</td></tr><tr><td>FAH-L video</td><td>17.15</td><td>33.12</td><td>1.00</td><td>0.98</td></tr></table>

Table 24. Points versus parameters. We compare WiLoR’s MANO output, MHR-ref, the dense ViT-B model after reprojection consistency, and the final development model with MANO fitting. HOT3D errors are in mm; bench2d scores are PCK@0.1 for direct 2D and fitted projections. Dense PVE uses the raw surface.
<table><tr><td>model</td><td>PA 2D 3D fit</td><td>root MRRPE PVE fps</td></tr><tr><td>WiLoR</td><td>6.72 0.69 0.77 43.89</td><td>48.576.49 4.7</td></tr><tr><td>MHR-ref</td><td>4.85 0.83 0.85 11.59</td><td>14.655.7031</td></tr><tr><td>dense ViT-B</td><td>4.38 0.84 0.82 8.67</td><td>11.67 5.5953</td></tr><tr><td>last milestone + fit 4.41 0.83</td><td>0.82 9.10</td><td>12.48 4.08 85</td></tr></table>

Table 25. Mesh fitting. PA-MPVPE (mm) on development image predictions. The learned parameter head shares the network of Tab. 5; GN: Gauss–Newton; fitter input noise: 3 mm unless noted. Bold row: default fitting stage.
<table><tr><td>mesh</td><td>HOT3D</td><td>EgoStandard</td></tr><tr><td>WiLoR (MANO)</td><td>6.49</td><td>7.52</td></tr><tr><td>dense surface</td><td>5.66</td><td>9.08</td></tr><tr><td>learned MANO-parameter head</td><td>6.79</td><td>12.05</td></tr><tr><td>closed form alone</td><td>6.51</td><td></td></tr><tr><td>closed form + 3 GN steps</td><td>4.49</td><td>7.42</td></tr><tr><td>learned fitter, no input noise</td><td>4.03</td><td>8.40</td></tr><tr><td>learned fitter + 1 GN step</td><td>4.08</td><td>7.28</td></tr></table>

Stojkovic (“Eko Rio Bravo 12 – A-mol chord”, CC BY 2.0); soldering, U.S. Air Force photo by Airman Jessica Struble (“Desoldering”, public domain); smartphone, Mictlancihuatl (“Hand holding Smartphone”, CC BY-SA 4.0). The overlays are ours; the other frames come from our heldout HOT3D [17] benchmark clips (HOT3D’s train\_aria recordings, disjoint from our training; Figs. 1 to 3; CC BY-SA 4.0, hand annotations CC BY-NC-SA 4.0, which give the ground-truth meshes and contact in Fig. 3), the DexYCB [16] s0 test split (Fig. 3; CC BY-NC 4.0) and the HInt [6] test split (Figs. 1 and 7; EPIC-KITCHENS-100 frames [63] via HInt, CC BY-NC 4.0). Figure 5 (bench2d v2 test images), left to right: 1 and 9 frames of the held-out HOT3D [17] benchmark clips (CC BY-SA 4.0, hand annotations CC BY-NC-SA 4.0; 9 with our synthetic low light); 2, 8 and 10 EPIC-KITCHENS-100 [63] frames via HInt [6] (CC BY-NC 4.0); 3, 5 and 6 New Days of Hands [62] frames via HInt (public test split; no separate frame license stated); 4 “Luggage and all” by Sally Taylor (Flickr, CC BY-NC 2.0) and 7 Flickr photo 7173995615 (CC BY-NC 2.0; author page no longer online), both via COCO val2017 and COCO-WholeBody [49]. Figure 6: frames of the held-out HOT3D [17] benchmark clips (CC BY-SA 4.0, hand annotations CC BY-NC-SA 4.0) (panels 1–5, 10–12) and COCO train2017 Flickr photos (panels 6–9, in order): “Snack Break” by Eden, Janine and Jim (CC BY 2.0), “Nadya and Jeff making cellphones” by David Mellis (CC BY 2.0), “vincent’s last night in town” by brooke (CC BY-SA 2.0) and “La Creación del Donut®” by ..::Fluckr You::.. (CC BY-SA 2.0); all cropped, with our renders. Figure 4, Flickr authors in panel order: Maggie Brauer, Zepfanman.com, Nagarjun Kandukuru, super awesome, Maryland GovPics, rachaelvoorhees, Becky Wetherington and Brad Greenlee. Photo titles: 1 “Delta\_100\_Ring\_Park073”, 2 “Zebra hand”, 3 “Kinu as Capsicum”, 4 “bratmobile” (CC BY 2.0); 5 “20111005-first lady host wine tasting for nga staff-jb”, 6 “wii bowling”, 7 “68/365 – I’m too sexy to match.”, 8 “Jelly Doughnut Incident” (CC BY 2.0); 1, 3 and 4 via Open Images V7, 2 via Openverse, 5–8 via COCO val2017; all cropped, 5–8 darkened by our low-light synthesis.

Table 26. Number of dense surface points. All models share data, initialization and an 8k-step label-convention fine-tune. Reduced-point models reconstruct 778 vertices with $K = N / 2$ spectral modes; the 778-point row is unsmoothed. HOT3D errors are in mm, and 2D scores use PCK@0.1. Speed differences are below run-to-run variation.
<table><tr><td>points</td><td>PA</td><td> $\mathrm { P A } { \cdot } { \mathsf { p } }$ </td><td>PVE</td><td> $\mathrm { P V E - p }$ </td><td>F@5</td><td>root</td><td>2D</td><td>3D fit</td><td>hard</td><td>dark</td><td>fps</td></tr><tr><td>778</td><td>4.40</td><td>4.99</td><td>5.66</td><td>6.21</td><td>0.80</td><td>8.54</td><td>0.84</td><td>0.83</td><td>0.98</td><td>0.82</td><td>55</td></tr><tr><td>400</td><td>4.30</td><td>4.88</td><td>5.40</td><td>5.95</td><td>0.78</td><td>8.64</td><td>0.84</td><td>0.83</td><td>0.98</td><td>0.83</td><td>53</td></tr><tr><td>200</td><td>4.27</td><td>4.85</td><td>5.48</td><td>6.03</td><td>0.76</td><td>8.80</td><td>0.84</td><td>0.83</td><td>0.98</td><td>0.83</td><td>56</td></tr></table>

Table 27. Fingertip extent on bench2d v2 matched hands. Signed tip error is measured along the labeled distal segment in segment lengths; negative values indicate shortfall. Mesh extent uses the furthest tip-region vertex. Length is the median meshto-labeled finger-length ratio. Ideal values are zero error and unit length.
<table><tr><td></td><td>joints</td><td>mesh</td><td>HOT3D</td><td>length</td></tr><tr><td>SAM 3D Body hand</td><td>-0.18</td><td>-0.14</td><td>-0.19</td><td>0.81</td></tr><tr><td>FAH-L, no tip step</td><td>-0.06</td><td>+0.01</td><td>0.00</td><td>0.99</td></tr><tr><td>FAH-L</td><td>-0.05</td><td>+0.02</td><td>+0.02</td><td>1.00</td></tr><tr><td>WiLoR [7]</td><td>+0.05</td><td>+0.12</td><td>+0.47</td><td>0.97</td></tr></table>

Table 28. VISOR validation segmentation. We report mIoU for hand-and-arm (h+a) and wrist-cut hand (h) masks, and inference speed. <sup>o</sup>Ground-truth box oracle. VISOR-HOS is trained on VI-SOR.
<table><tr><td></td><td colspan="2">VISOR</td></tr><tr><td>method</td><td>h+a</td><td>h fps</td></tr><tr><td>FAH-L FAH detector</td><td>0.54 0.78 0.53 0.77</td><td>54 253</td></tr><tr><td>SAM 2.1-B [12], boxes SAM 2.1-L, boxes</td><td>0.55 0.77</td><td>23</td></tr><tr><td>SAM 3 [13], boxes</td><td>0.56 0.78 0.56 0.81</td><td>15 8.4</td></tr><tr><td>SAM 3, text</td><td>0.59 0.81</td><td>6.3</td></tr><tr><td>VISOR-HOS [52]</td><td>0.90 0.79</td><td>21</td></tr><tr><td>Hands23 [62]</td><td>0.50 0.72</td><td>11</td></tr><tr><td>SAM 2.1-L, GT boxesº</td><td>0.94 0.83</td><td></td></tr></table>

Table 29. Additional contact metrics. HOT3D and DexYCB use mean per-hand vertex F1; HOI4D and ARCTIC also report pooled F1/AP, and MOW uses per-image F1. Primary HOT3D/DexYCB results are in Tab. 4. HACO is excluded from best-value marks.
<table><tr><td rowspan="2">model</td><td colspan="2">HOT3D DexYCB</td><td colspan="3">HOI4D (no Chair)</td><td colspan="3">ARCTIC val</td><td rowspan="2">MOW F1</td></tr><tr><td>hand</td><td>hand</td><td>F1</td><td>AP</td><td>hand</td><td>F1</td><td>AP hand</td><td></td></tr><tr><td>HACO [29]</td><td>0.60</td><td>0.66</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.52</td></tr><tr><td>FAH-S</td><td>0.74</td><td>0.69</td><td>0.62 0.61</td><td></td><td></td><td>0.54 0.74 0.79</td><td></td><td>0.69</td><td>0.58</td></tr><tr><td>FAH-B</td><td>0.76</td><td>0.71</td><td>0.63</td><td>0.62</td><td>0.56</td><td>0.76</td><td>0.80</td><td>0.70</td><td>0.57</td></tr><tr><td>FAH-L</td><td>0.76</td><td>0.71</td><td>0.64</td><td>0.64</td><td>0.56</td><td>0.76</td><td>0.81</td><td>0.72</td><td>0.56</td></tr><tr><td>FAH-H</td><td>0.77</td><td>0.72</td><td>0.63 0.62</td><td></td><td>0.55</td><td>0.77</td><td>0.81</td><td>0.72</td><td>0.53</td></tr></table>

Table 30. Contact data ablation. Two-seed means for FAH-L; vertex F1/AP and MOW per-image F1 (HACO protocol). All recipes include HOT3D, DexYCB and MOW. Added ARC-TIC/HOI4D data make those evaluations in-domain; HOI4D excludes Chair and holds out the test subject.
<table><tr><td rowspan="2">data</td><td colspan="3">HOT3D DexYCB MOW</td><td>HOI4D</td><td>ARCTIC</td></tr><tr><td>F1 AP</td><td>F1 AP</td><td>F1</td><td>F1 AP</td><td>F1 AP</td></tr><tr><td>original</td><td>0.80 0.85 0.76 0.81</td><td></td><td></td><td></td><td>0.51 0.52 0.52 0.680.70</td></tr><tr><td>+ ARCTIC</td><td>0.80 0.85 0.76 0.82</td><td></td><td></td><td>0.53 0.540.53 0.76 0.81</td><td></td></tr><tr><td>+ ARCTIC, HOI4D 0.80 0.85 0.76 0.82</td><td></td><td></td><td></td><td>0.54 0.64 0.63 0.76 0.81</td><td></td></tr></table>