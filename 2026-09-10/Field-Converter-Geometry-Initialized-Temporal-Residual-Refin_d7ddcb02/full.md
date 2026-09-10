# Field Converter: Geometry-Initialized Temporal Residual Refinement for World-Grounded Player Pose Estimation from Soccer Broadcasts

Simon Khan <sup>1</sup> <sup>2</sup> Laurent Gajny <sup>1</sup> Jennyfer Lecompte <sup>2</sup> Sébastien Laporte <sup>1</sup>

## Abstract

Recovering 3D human pose from monocular sports broadcasts remains challenging when players must be localized in a shared metric world coordinate system rather than only reconstructed relative to their own body. We introduce Field Converter, a geometry-initialized temporal residual framework for world-grounded 3D player pose estimation from calibrated soccer broadcasts. Our method first uses camera and pitch geometry to initialize the player root through ray–ground intersection, then predicts a temporal residual correction from pose, image, camera, and geometric cues. On match-disjoint evaluation sequences, residual refinement reduces root error from 49 cm with geometry alone to 14 cm with a frame-wise MLP and 10 cm with a TCN, while a Transformer achieves a comparable 11 cm. The resulting world-space MPJPE reaches 13.2 cm, and ablations show that residual prediction clearly outperforms direct globalroot regression while temporal context matters more than the specific temporal backbone. Failure analysis further identifies airborne motion as the main limitation of the ground-based geometric initialization. The code is available at https://github.com/KhanSimon/field\_converter.

## 1. Introduction

Recovering the 3D motion of athletes from standard broadcast videos could make large sports video archives much more useful for analysis. Compared with 2D measurements or player-centered 3D pose estimates, reconstructing players in a common metric field coordinate system makes it possible to study how all players move together, measure their relative positions, and relate their motion to the geometry of the pitch. It also provides a basis for combining player motion with ball trajectories and game context. Such representations are relevant to tactical analysis (Memmert et al., 2019), augmented coaching (Wen et al., 2024) , biomechanical studies (Dos’Santos et al., 2021) , injury-risk assessment, officiating (Wang et al., 2025), and immersive or free-viewpoint replay (Hilton et al., 2010).

Recent advances in monocular 3D human pose estimation and human mesh recovery have substantially improved the estimation of articulated body pose from individual images and videos. Parametric body models such as SMPL (Loper et al., 2015) established a compact representation of human pose and shape, while methods such as HybrIK (Li et al., 2021), 4DHumans (Goel et al., 2023), Multi-HMR (Baradel et al., 2024), and SAM 3D Body (Yang et al., 2026) have progressively improved robustness to challenging viewpoints, uncommon poses, and multi-person scenes. Temporal approaches such as VideoPose3D (Pavllo et al., 2019), MixSTE (Zhang et al., 2022), and MotionBERT (Zhu et al., 2023) further exploit motion context to reduce monocular ambiguities and improve temporal consistency.

However, an accurate body-relative skeleton does not imply accurate localization in the surrounding scene. This distinction is particularly important in team sports. Two players may exhibit very similar local body configurations while occupying completely different regions of the pitch. For tactical or biomechanical applications, the desired representation is therefore not only the articulation of each player relative to the pelvis, but the complete 3D body expressed in a shared and metrically meaningful world coordinate frame.

Several recent methods explicitly address metric 3D localization beyond body-relative pose. Ray3D (Zhan et al., 2022) incorporates calibrated camera rays to estimate both 3D body configuration and its metric position with respect to the camera. GLAMR (Yuan et al., 2022), SLAHMR (Ye et al., 2023), and PACE (Kocabas et al., 2024) recover human trajectories while accounting for camera motion. WHAM (Shin et al., 2024), TRAM (Wang et al., 2024), GVHMR (Shen et al., 2024), and ProxyCap (Zhang et al., 2024) further introduce temporal, geometric, cameramotion, gravity, and contact priors for reconstruction in a global coordinate system.

![](images/5f185d3d2f742adee6b245a635fb650f5644b43ac5518d89c8e6bee0c606c9d4.jpg)  
Figure 1. From monocular broadcast video to world-grounded 3D player pose. Left: three actions observed in the broadcast and their corresponding camera-relative, self-centered 3D poses. Right: the same reconstructed poses localized in a common metric field coordinate system, where their positions can be directly related to the pitch and to one another.

The challenge becomes even more pronounced in broadcast soccer. Players are frequently small in the image, undergo partial occlusion, move rapidly over a large metric area, and are observed through a camera undergoing pan, tilt, and zoom. Generic methods for recovering camera motion by tracking static visual features across frames, such as visual odometry and simultaneous localization and mapping (SLAM) (Mur-Artal et al., 2015), are poorly matched to this setting: the pitch contains large weakly textured regions, players and spectators create substantial dynamic content, and focal length can vary during a broadcast sequence. WorldPose (Jiang et al., 2024), a large-scale global 3D pose dataset captured during the FIFA World Cup, directly highlights this local-to-global gap and reports large scene-level localization errors for several state-of-the-art global reconstruction approaches.

At the same time, soccer broadcasts contain an unusually strong geometric prior: the pitch has known metric dimensions and standardized line markings. This structure has motivated sports-specific camera-calibration and fieldregistration methods such as TVCalib (Theiner & Ewerth, 2023), No Bells, Just Whistles (Gutiérrez-Pérez & Agudo, 2024), PnLCalib (Gutiérrez-Pérez & Agudo, 2026), and BroadTrack (Magera et al., 2025). Related work has also shown that known sports geometry can support monocular athlete pose recovery when ordinary scene calibration is difficult (Baumgartner & Klatt, 2023).

With Field Converter, we exploit the complementary strengths of modern monocular pose estimation and explicit sports geometry. Our starting observation is empirical but important: for the soccer sequences considered here, the relative 3D pose produced by a strong monocular body model is already close to the corresponding camera-relative ground truth after self-centering. The dominant remaining error is therefore not the articulated pose itself, but the global camera-space translation of the player. This motivates a decomposition in which local pose is estimated upstream and our method focuses specifically on world grounding.

This leads to the central question addressed in this work: given monocular broadcast video and calibrated field geometry, how should global player translation be recovered reliably? In particular, is it more effective to refine a geometrybased initialization temporally than to regress global translation directly? To address this question, we first estimate a geometrically grounded root position using ray–ground intersection. We then refine this initialization with a temporal residual network conditioned on pose, image, camera, and field cues. Finally, the refined root anchors the relative skeleton in a common world coordinate system.

## 2. Related Work

## 2.1. Camera-Relative 3D Human Pose and Mesh Recovery

Modern monocular 3D human reconstruction has progressed from direct joint estimation toward articulated parametric body recovery. SMPL (Loper et al., 2015) introduced a widely adopted low-dimensional skinned body model that represents body shape and articulated pose. HybrIK (Li et al., 2021) combines accurate 3D joint prediction with analytical inverse kinematics to obtain body rotations, segment orientations, and human meshes. 4DHumans (Goel et al., 2023) improves image-based reconstruction through a transformer-based HMR architecture and extends the system to tracking in video. Multi-HMR (Baradel et al., 2024) performs multi-person whole-body mesh recovery in a single shot, including 3D localization cues. More recently, SAM 3D Body (Yang et al., 2026) introduced a promptable singleimage full-body human mesh recovery model designed for robust reconstruction across a wide range of images and poses.

Temporal information is particularly effective for monocular 3D pose estimation because motion provides constraints that are absent in a single image. VideoPose3D (Pavllo et al., 2019) demonstrated that dilated temporal convolutions can effectively lift 2D keypoint sequences into 3D. Transformerbased approaches such as MixSTE (Zhang et al., 2022) model spatial and temporal relations over complete pose sequences, while MotionBERT (Zhu et al., 2023) learns transferable motion representations through large-scale motion pretraining. On the standard Human3.6M benchmark, MotionAGFormer-B (Mehraban et al., 2024), for example, reports a Protocol-1 error of 38.4 mm, illustrating the accuracy achievable by recent temporal 3D pose lifting methods under conventional benchmark conditions.

These approaches mainly target body articulation, camerarelative pose, or camera-centered mesh recovery. Even when temporally smooth and locally accurate, such predictions can still follow an incorrect metric trajectory in the surrounding scene. Our method treats these local estimates as strong upstream observations and focuses specifically on the missing global translation.

## 2.2. Absolute and World-Grounded Human Motion Recovery

A growing body of work seeks to move beyond root-relative reconstruction. Ray3D (Zhan et al., 2022) transforms 2D observations into normalized 3D rays and conditions the prediction on calibrated camera extrinsics, demonstrating the importance of explicit camera geometry for absolute 3D localization.

For dynamic-camera video, GLAMR (Yuan et al., 2022) reconstructs global human meshes while accounting for long-term occlusions and camera motion. SLAHMR (Ye et al., 2023) jointly optimizes camera and human trajectories in a shared scene, while PACE (Kocabas et al., 2024) couples human and camera estimation in a global reconstruction framework. These methods address the ambiguity between observed image motion, human motion, and camera motion.

More recent approaches introduce stronger priors and more efficient formulations. WHAM (Shin et al., 2024) combines visual and motion features with camera angular velocity and contact-aware trajectory refinement. TRAM (Wang et al., 2024) estimates camera motion and metric scale before reconstructing a global human trajectory. GVHMR (Shen et al., 2024) introduces gravity-view coordinates to reduce world-coordinate ambiguity and stabilize world-grounded reconstruction. ProxyCap (Zhang et al., 2024) learns worldspace motion from human-centric proxy representations and explicitly targets plausible ground contact.

The difficulty of global localization is particularly apparent on WorldPose (Jiang et al., 2024). Under its global evaluation, in which a single shared Procrustes transformation aligns the predicted player trajectories to the ground truth, GLAMR and SLAHMR obtain G-MPJPE values of 18,888.9 and 8,334.1 mm, with per-meter drifts of 53.3 and 17.6 cm/m, respectively. The same benchmark reports perplayer, per-frame Procrustes-aligned PA-MPJPE values of 85.2 and 163.9 mm for the two methods. Although these metrics use different alignment protocols and are therefore not directly comparable, they highlight that accurate framewise body alignment does not by itself ensure an accurate trajectory in a common scene coordinate system.

Our setting differs from generic in-the-wild reconstruction in an important way: the metric geometry of the soccer pitch is known and the broadcast camera can be calibrated from field markings. We therefore avoid asking the temporal network to infer global translation without structure. Instead, known scene geometry supplies a strong metric initialization, and learning is reserved for the residual error.

## 2.3. Sports-Specific 3D Pose Estimation

Sports video is a challenging domain for monocular pose estimation because athletic motion includes unusual configurations, high accelerations, motion blur, self-occlusion, and subjects that can occupy relatively few image pixels. Sports-specific work has consequently explored domain priors, biomechanical validation, and scene geometry.

Baumgartner and Klatt (Baumgartner & Klatt, 2023) combine 2D pose estimation with partial sports-field registration and jointly reason about athlete pose and camera calibration. Their work shows that line markings and known sports geometry can provide valuable constraints when complete camera calibration is otherwise unavailable.

AutoSoccerPose (Yeung et al., 2024) proposes a semiautomated pipeline for extracting 2D and 3D posture sequences from soccer shooting videos. AthletePose3D (Yeung et al., 2025) further quantifies the domain shift associated with athletic motion: on its validation set, using ground-truth 2D poses as input and hip-centered MPJPE, TCPFormer trained only on Human3.6M obtains 234.2 mm, whereas training on Human3.6M together with AthletePose3D reduces the error to 98.3 mm. Oštrek et al. (Oštrek et al., 2019) compare monocular vision-based motion capture with reference measurements in alpine skiing, highlighting the importance of validation in physically meaningful units rather than relying only on visually plausible motion.

WorldPose (Jiang et al., 2024) is the most directly relevant benchmark to our setting. It provides global player trajectories and body poses from FIFA World Cup footage, together with broadcast-camera information, and is explicitly designed to study multi-person global pose estimation in soccer. In contrast to sports datasets centered primarily on individual athletic motions, WorldPose captures multiple interacting players moving over an entire pitch and therefore directly exposes the local-to-global localization problem addressed in this work.

## 2.4. Sports-Field Registration and Broadcast-Camera Estimation

Metric world reconstruction requires an accurate camera model. In soccer, generic feature-based calibration is difficult because the visible field contains limited texture and the broadcast camera undergoes significant pan, tilt, and zoom. Field markings, however, provide a standardized geometric calibration object.

TVCalib (Theiner & Ewerth, 2023) formulates sports-field registration as camera calibration and optimizes camera parameters using field-segment reprojection. No Bells, Just Whistles (Gutiérrez-Pérez & Agudo, 2024) exploits geometric properties of the sports field to improve calibration robustness. PnLCalib (Gutiérrez-Pérez & Agudo, 2026) further refines sports-field registration through joint pointand-line optimization. BroadTrack (Magera et al., 2025) extends the problem temporally and introduces a broadcastcamera tracking system tailored to soccer.

These approaches are complementary to our method. We do not treat camera calibration as the primary contribution. Instead, we assume that camera intrinsics and extrinsics are available from annotations or from an upstream fieldregistration system, and we use them to construct metric player rays, ground intersections, camera descriptors, and camera-to-world transformations.

## 3. Materials and Method

## 3.1. Dataset

We evaluate our method on the FIFA Skeletal Tracking Light 2026 dataset, which contains calibrated monocular soccerbroadcast sequences. Our processed subset comprises 89 clips from eight different matches, totaling approximately 2.41 million valid player–frame observations. The annotations provide time-varying camera intrinsics, extrinsics, and radial-distortion parameters, as well as player bounding boxes, validity masks, and 25-joint 3D skeletons expressed in a common metric field coordinate system.

![](images/3f804924a0853a60419456680cb99fd75e157bff6094756a63eafbfb71a9fc80.jpg)  
Figure 2. Conceptual pipeline. Player tracks, camera calibration, 2D and relative 3D poses are obtained from the broadcast video and provided to our field converter.

For each valid player bounding box, we apply SAM 3D Body to obtain 2D keypoints and a 3D body estimate. The resulting skeletons are mapped to the common 25-joint convention, and the sign of the z coordinate is inverted to match the camera-coordinate convention of the dataset. The pelvis is defined as the midpoint of the left and right hip joints, and the resulting 3D skeletons are centered around this point. The camera-space root targets are defined using the same hip midpoint, while the corresponding root-relative ground-truth poses are derived from the annotated worldspace skeletons and calibrated camera transformations. The geometry-based root initialization described in Eq. (18) is computed offline for every valid player–frame observation.

To assess generalization to unseen games, we adopt a strict match-disjoint evaluation protocol. The training set contains 62 clips from six matches, the validation set contains 12 clips from the BRA\_KOR match, and the test set contains 15 clips from the held-out ENG\_FRA match. Consequently, no match appears in more than one split, preventing the model from exploiting match-specific camera trajectories, stadium geometry, or broadcast patterns.

## 3.2. Overview of the pipeline

Given a monocular soccer-broadcast sequence, our objective is to recover the 3D pose of every tracked player in a common metric field coordinate system. The complete conceptual pipeline is illustrated in Fig. 2.

The proposed method corresponds to the final worldgrounding stage. For each visible player and frame, we assume access to an upstream bounding box, 2D keypoints, a self-centered camera-oriented 3D pose, and calibrated

camera parameters. The soccer pitch geometry is known.   
Figure 3 details the proposed field-converter method.

Our method has three stages. First, a lower-limb observation is back-projected through the calibrated camera and intersected with the field plane, yielding a geometry-based root initialization. Second, a temporal network predicts a residual correction to this initialization. Third, the refined camera-space root anchors the relative skeleton, which is transformed to the common world coordinate system using the calibrated camera extrinsics.

## 3.3. Notation and Problem Formulation

Throughout the paper, the first superscript indicates the coordinate representation (e.g., c for camera, w for world, and rel for pelvis-centered relative coordinates), while a second superscript, when present, indicates the status of the quantity (e.g., gt for ground truth and init for geometry-based initialization). Subscripts $i , t ,$ and j refer to the player, frame, and skeletal joint indices, respectively. We use column-vector notation for individual 3D points. A lowercase $\mathbf { x } _ { i , t , j } \in \mathbb { R } ^ { 3 }$ denotes one skeletal joint, while $\mathbf { X } _ { i , t } \in \mathbb { R } ^ { J \times 3 }$ denotes the corresponding collection of J joints.

The camera at frame t is represented by the intrinsic matrix ${ \bf K } _ { t } \in \mathbb { R } ^ { 3 \times 3 }$ , a world-to-camera rotation $\mathbf { R } _ { t } \in S O ( 3 )$ , a translation $\mathbf { t } _ { t } \in \mathbb { R } ^ { 3 }$ , and radial-distortion parameters $\mathbf { k } _ { t } \in$ $\mathbb { R } ^ { 2 }$

A world-space joint $\mathbf { x } _ { i , t , j } ^ { w }$ is mapped to camera coordinates by

$$
\mathbf { x } _ { i , t , j } ^ { c } = \mathbf { R } _ { t } \mathbf { x } _ { i , t , j } ^ { w } + \mathbf { t } _ { t } ,\tag{1}
$$

and the inverse transformation is

$$
\begin{array} { r } { \mathbf { x } _ { i , t , j } ^ { w } = \mathbf { R } _ { t } ^ { \top } \left( \mathbf { x } _ { i , t , j } ^ { c } - \mathbf { t } _ { t } \right) . } \end{array}\tag{2}
$$

These transformations are applied independently to all J joints.

An upstream monocular body estimator provides a selfcentered relative pose

$$
\widehat { \mathbf { X } } _ { i , t } ^ { \mathrm { r e l } } \in \mathbb { R } ^ { J \times 3 } ,\tag{3}
$$

expressed in the camera coordinate frame. In our implementation, the relative skeleton is derived from SAM 3D Body (Yang et al., 2026), mapped to the target joint convention.

The corresponding global camera-space position of each joint is obtained by adding the same root translation to every relative joint:

$$
\widehat { \mathbf { x } } _ { i , t , j } ^ { c } = \widehat { \mathbf { x } } _ { i , t , j } ^ { \mathrm { r e l } } + \widehat { \mathbf { r } } _ { i , t } ^ { c } , \qquad j = 1 , \ldots , J ,\tag{4}
$$

where $\widehat { \mathbf { r } } _ { i , t } ^ { c } \in \mathbb { R } ^ { 3 }$ is the camera-space root translation. Consequently, our learning problem is to recover the global root translation rather than to regenerate the complete articulated body.

For supervised training, each ground-truth world-space joint is transformed to camera coordinates using Eq. (1). Let $j _ { L }$ and $j _ { R }$ denote the left- and right-hip joints, respectively. The ground-truth root is defined as their midpoint:

$$
\mathbf { r } _ { i , t } ^ { c , \mathrm { g t } } = \frac { 1 } { 2 } \left( \mathbf { x } _ { i , t , j _ { L } } ^ { c , \mathrm { g t } } + \mathbf { x } _ { i , t , j _ { R } } ^ { c , \mathrm { g t } } \right) .\tag{5}
$$

The corresponding ground-truth relative pose is defined joint-wise as

$$
\begin{array} { r } { { \bf x } _ { i , t , j } ^ { \mathrm { r e l , g t } } = { \bf x } _ { i , t , j } ^ { c , \mathrm { g t } } - { \bf r } _ { i , t } ^ { c , \mathrm { g t } } , \qquad j = 1 , \ldots , J . } \end{array}\tag{6}
$$

## 3.4. Geometry-Based Root Initialization

The playing surface is modeled as a plane

$$
\boldsymbol { \pi } : \mathbf { n } ^ { \top } \mathbf { x } + b = 0 ,\tag{7}
$$

where n is the unit field normal vector and b is the plane offset. In the standard soccer-field coordinate system, this can be chosen as $z = 0$

For player i at frame t, we select a candidate lower-limb keypoint

$$
j _ { i , t } ^ { * } = \arg \operatorname* { m a x } _ { j \in \mathcal { T } _ { \mathrm { c a n d } } } v _ { i , t , j } ,\tag{8}
$$

where image coordinate v increases downward and $\mathcal { I } _ { \mathrm { c a n d } }$ contains valid candidate joints. In practice, the candidate set can be restricted to ankles, heels, toes, or other lower-limb landmarks.

Let the selected 2D point be

$$
\widetilde { \mathbf { p } } _ { i , t } = \left[ \begin{array} { l } { u _ { i , t , j ^ { * } } } \\ { v _ { i , t , j ^ { * } } } \\ { 1 } \end{array} \right] .\tag{9}
$$

After inverting radial distortion when required, its cameraspace ray direction is

$$
{ \bf d } _ { i , t } ^ { c } = \frac { { \bf K } _ { t } ^ { - 1 } \widetilde { \bf p } _ { i , t } } { { \left\| { \bf K } _ { t } ^ { - 1 } \widetilde { \bf p } _ { i , t } \right\| } _ { 2 } } .\tag{10}
$$

The camera center in world coordinates is

$$
{ \bf C } _ { t } = - { \bf R } _ { t } ^ { \top } { \bf t } _ { t } .\tag{11}
$$

The camera viewing direction expressed in world coordinates is obtained by transforming the positive camera z-axis into the world frame:

$$
{ \bf d } _ { t } ^ { \mathrm { f o r w a r d } } = { \bf R } _ { t } ^ { \top } { \bf e } _ { z } , \qquad { \bf e } _ { z } = \left[ 0 \quad 0 \quad 1 \right] ^ { \top } ,\tag{12}
$$

and the world-space ray direction is

$$
{ \bf d } _ { i , t } ^ { w } = { \bf R } _ { t } ^ { \top } { \bf d } _ { i , t } ^ { c } .\tag{13}
$$

![](images/8574951cee5dac0328e18d9d0024191623f5c8100687b0a457e601f0fd7de7c5.jpg)  
Figure 3. Overview of the proposed Field Converter method. First, a geometry-based initialization estimates the player root from the calibrated camera, the relative 3D pose, and a ray–ground intersection. Second, a temporal model predicts a residual correction using pose, image, camera, and geometric cues. Finally, the refined root anchors the relative skeleton in camera coordinates before transformation to the common world coordinate system.

Points along the ray are

$$
\mathbf { q } _ { i , t } ^ { w } ( \lambda ) = \mathbf { C } _ { t } + \lambda \mathbf { d } _ { i , t } ^ { w } .\tag{14}
$$

Intersecting the ray with the pitch plane gives

$$
\lambda _ { i , t } ^ { * } = - \frac { \mathbf { n } ^ { \top } \mathbf { C } _ { t } + b } { \mathbf { n } ^ { \top } \mathbf { d } _ { i , t } ^ { w } } ,\tag{15}
$$

and

$$
\mathbf { q } _ { i , t } ^ { w } = \mathbf { C } _ { t } + \lambda _ { i , t } ^ { * } \mathbf { d } _ { i , t } ^ { w } .\tag{16}
$$

The intersection is accepted only when the ray is not parallel to the plane, $\lambda _ { i , t } ^ { * } > 0$ , and the required observations are valid.

The intersection is transformed back to camera coordinates:

$$
{ \bf q } _ { i , t } ^ { c } = { \bf R } _ { t } { \bf q } _ { i , t } ^ { w } + { \bf t } _ { t } .\tag{17}
$$

Assuming that the selected joint is in contact with the pitch, the initial root translation is

$$
\mathbf { r } _ { i , t } ^ { c , \mathrm { i n i t } } = \mathbf { q } _ { i , t } ^ { c } - \widehat { \mathbf { x } } _ { i , t , j ^ { * } } ^ { \mathrm { r e l } } .\tag{18}
$$

This initialization is metric and directly combines 2D localization, camera geometry, field scale, and the estimated 3D body configuration. It is nevertheless only approximate. If the selected foot is airborne, the ray intersects the pitch at a point behind the true joint location. Occlusion, motion blur, imperfect keypoint localization, and relative-pose error can produce similar biases. These failure modes motivate a learned residual correction.

## 3.5. Input Representation

For each player and frame, the temporal model receives a feature vector

$$
\mathbf { f } _ { i , t } = \left[ \mathbf { f } _ { i , t } ^ { 3 D } , \mathbf { f } _ { i , t } ^ { 2 D } , \mathbf { f } _ { i , t } ^ { \mathrm { b o x } } , \mathbf { f } _ { t } ^ { \mathrm { c a m } } , \mathbf { q } _ { i , t } ^ { \mathrm { w } } , \mathbf { m } _ { i , t } \right] .\tag{19}
$$

Relative 3D pose. The relative-pose descriptor is the flattened pelvis-centered skeleton:

$$
{ \bf f } _ { i , t } ^ { 3 D } = \mathrm { v e c } \left( \widehat { \bf X } _ { i , t } ^ { \mathrm { r e l } } \right) .\tag{20}
$$

2D pose. We include both image-normalized and bounding-box-normalized 2D keypoints. For an image of width W and height H,

$$
\mathbf { p } _ { i , t , j } ^ { \mathrm { i m g } } = \left[ \frac { u _ { i , t , j } } { W } , \frac { v _ { i , t , j } } { H } \right] .\tag{21}
$$

Given box center $( c _ { i , t } ^ { x } , c _ { i , t } ^ { y } )$ , width $w _ { i , t }$ , and height $h _ { i , t }$ ,

$$
\mathbf { p } _ { i , t , j } ^ { \mathrm { b o x } } = \left[ \frac { u _ { i , t , j } - c _ { i , t } ^ { x } } { w _ { i , t } } , \frac { v _ { i , t , j } - c _ { i , t } ^ { y } } { h _ { i , t } } \right] .\tag{22}
$$

We then define the 2D pose descriptor as the concatenation of both representations over all joints:

$$
\mathbf { f } _ { i , t } ^ { 2 D } = \mathrm { v e c } \left( \left[ \mathbf { p } _ { i , t , j } ^ { \mathrm { i m g } } , \mathbf { p } _ { i , t , j } ^ { \mathrm { b o x } } \right] _ { j = 1 } ^ { J } \right) .\tag{23}
$$

Bounding-box geometry. The normalized box descriptor is

$$
\mathbf { f } _ { i , t } ^ { \mathrm { b o x } } = \left[ \frac { c _ { i , t } ^ { x } } { W } , \frac { c _ { i , t } ^ { y } } { H } , \frac { w _ { i , t } } { W } , \frac { h _ { i , t } } { H } , \log \frac { w _ { i , t } } { h _ { i , t } } \right] .\tag{24}
$$

Although the body estimator itself may already use the bounding box, box scale and image position remain useful cues for metric depth and perspective.

Camera features. The camera descriptor contains normalized intrinsic parameters, lens distortion, camera position, and viewing direction:

$$
\mathbf { f } _ { t } ^ { \mathrm { { c a m } } } = \left[ \frac { f _ { x , t } } { W } , \frac { f _ { y , t } } { H } , \frac { c _ { x , t } } { W } , \frac { c _ { y , t } } { H } , k _ { 1 , t } , k _ { 2 , t } , \mathbf { C } _ { t } , \mathbf { d } _ { t } ^ { \mathrm { { f o r w a r d } } } \right]\tag{25}
$$

Ground-intersection feature. The geometric feature provided to the temporal model is the world-space ray–ground intersection

$$
\mathbf { q } _ { i , t } ^ { \mathrm { w } } \in \mathbb { R } ^ { 3 } ,\tag{26}
$$

obtained from the selected lower-body keypoint as described in Sec. 3.4. This feature directly provides the network with the metric ground location implied by the current image observation and camera geometry.

Validity masks. Joint validity is encoded for each player and frame by

$$
\mathbf { m } _ { i , t } = [ m _ { i , t , 1 } , \dots , m _ { i , t , J } ] \in \{ 0 , 1 \} ^ { J } ,\tag{27}
$$

where $m _ { i , t , j } = 1$ indicates that joint j is valid and $m _ { i , t , j } =$ 0 otherwise. This joint-level mask is provided explicitly as an input feature.

We separately denote by $v _ { i , t } \in \{ 0 , 1 \}$ the player–frame validity indicator used to mask the training objectives.

## 3.6. Temporal Residual Refinement

We define the ground-truth residual as

$$
\Delta \mathbf { r } _ { i , t } ^ { c , \mathrm { g t } } = \mathbf { r } _ { i , t } ^ { c , \mathrm { g t } } - \mathbf { r } _ { i , t } ^ { c , \mathrm { i n i t } } .\tag{28}
$$

The temporal refinement model $F _ { \theta }$ predicts this residual from the sequence of input features:

$$
\begin{array} { r } { \Delta \widehat { \mathbf { r } } _ { i , 1 : T } ^ { c } = F _ { \theta } \left( \mathbf { f } _ { i , 1 : T } \right) , } \end{array}\tag{29}
$$

where $\Delta \widehat { \mathbf { r } } _ { i , t } ^ { c }$ denotes the predicted root correction in camera coordinates.

A per-frame encoder first maps the input vector into a latent representation:

$$
\mathbf { h } _ { i , t } = E _ { \theta } \left( \mathbf { f } _ { i , t } \right) .\tag{30}
$$

Table 1. Architecture of the temporal residual refinement models.
<table><tr><td>Component</td><td>TCN</td><td>Transformer</td></tr><tr><td>Input dimension</td><td>370</td><td>345</td></tr><tr><td>Frame encoder</td><td>370-192-192-192</td><td>345-256-256</td></tr><tr><td>Latent dimension</td><td>192</td><td>256</td></tr><tr><td>Temporal layers</td><td>5 residual blocks</td><td>2 encoder layers</td></tr><tr><td>Temporal operator</td><td>2× Conv1D/block</td><td>Self-attention</td></tr><tr><td>Kernel / heads</td><td>k = 3</td><td>4 heads</td></tr><tr><td>Dilations / FFN width</td><td>(1, 2, 4, 8, 16)</td><td>512</td></tr><tr><td>Normalization</td><td>None</td><td>Pre-LayerNorm</td></tr><tr><td>Positional encoding</td><td>Not required</td><td>Learned</td></tr><tr><td>Prediction head</td><td>192-128-3</td><td>256-128-3</td></tr><tr><td>Activation</td><td>GELU</td><td>GELU</td></tr><tr><td>Dropout</td><td>0.14</td><td></td></tr><tr><td>Window / stride</td><td></td><td>0.10</td></tr><tr><td></td><td>41 / 8 frames</td><td>41 / 8 frames</td></tr><tr><td>Parameters</td><td>1.278M</td><td>1.252M</td></tr></table>

A temporal backbone then aggregates information across the sequence:

$$
{ \bf z } _ { i , 1 : T } = G _ { \theta } \left( { \bf h } _ { i , 1 : T } \right) .\tag{31}
$$

Finally, a lightweight prediction head estimates the residual:

$$
\Delta \widehat { \mathbf { r } } _ { i , t } ^ { c } = H _ { \theta } \left( \mathbf { z } _ { i , t } \right) .\tag{32}
$$

Thus, $F _ { \theta }$ denotes the complete refinement model, i.e. the composition of $E _ { \theta } , G _ { \theta }$ , and $H _ { \theta }$ . The refined root is

$$
\begin{array} { r } { \widehat { \mathbf { r } } _ { i , t } ^ { c } = \mathbf { r } _ { i , t } ^ { c , \mathrm { i n i t } } + \Delta \widehat { \mathbf { r } } _ { i , t } ^ { c } . } \end{array}\tag{33}
$$

The temporal backbone can be instantiated with a temporal convolutional network or a Transformer. A TCN uses stacked one-dimensional convolutions, residual connections, and dilation to obtain a large temporal receptive field with low computational cost. A Transformer instead uses positional information and bidirectional self-attention to model long-range dependencies. The residual formulation is independent of this architectural choice. Table 1 describes the architecture used.

Training and inference use overlapping temporal windows. If a frame belongs to several windows, the corresponding root estimates are aggregated:

$$
\widehat { \mathbf { r } } _ { i , t } ^ { c } = \frac { 1 } { | \mathcal { W } _ { i , t } | } \sum _ { w \in \mathcal { W } _ { i , t } } \widehat { \mathbf { r } } _ { i , t } ^ { c , ( w ) } ,\tag{34}
$$

where $\mathcal { W } _ { i , t }$ denotes the set of windows containing frame t.

## 3.7. Training Objectives

The primary objective supervises the refined camera-space root. Let $\begin{array} { r } { N _ { r } = \sum _ { i , t } v _ { i , i } } \end{array}$ denote the number of valid player– frame observations and $\begin{array} { r } { \Omega = \sum _ { a \in \{ x , y , z \} } \omega _ { a } } \end{array}$ the sum of the

axis weights. We use the axis-weighted Smooth L1 loss

$$
\mathcal { L } _ { \mathrm { r o o t } } = \frac { 1 } { N _ { r } \Omega } \sum _ { i , t } v _ { i , t } \sum _ { a \in \{ x , y , z \} } \omega _ { a } \rho _ { \beta } \left( \widehat { r } _ { i , t , a } ^ { c } - r _ { i , t , a } ^ { c , \mathrm { g t } } \right) .\tag{35}
$$

Here, $v _ { i , t }$ is the player–frame validity mask and $\omega _ { a }$ controls the contribution of axis a. The scalar Smooth L1 penalty is

$$
\rho _ { \beta } ( e ) = \left\{ \begin{array} { l l } { \displaystyle \frac { e ^ { 2 } } { 2 \beta } , } & { | e | < \beta , } \\ { \displaystyle | e | - \frac { \beta } { 2 } , } & { | e | \ge \beta , } \end{array} \right. \quad \beta = 1 .\tag{36}
$$

The penalty is applied independently to each coordinate. Root, velocity, and acceleration losses are evaluated in normalized root coordinates. The camera-space consistency loss is evaluated in metres.

To supervise temporal dynamics, we define the predicted and ground-truth first-order root differences as

$$
\widehat { \mathbf { v } } _ { i , t } = \widehat { \mathbf { r } } _ { i , t } ^ { c } - \widehat { \mathbf { r } } _ { i , t - 1 } ^ { c } ,\tag{37}
$$

and

$$
\mathbf { v } _ { i , t } ^ { \mathrm { g t } } = \mathbf { r } _ { i , t } ^ { c , \mathrm { g t } } - \mathbf { r } _ { i , t - 1 } ^ { c , \mathrm { g t } } .\tag{38}
$$

The first order temporal loss is then

$$
\mathcal { L } _ { \mathrm { v e l } } = \frac { 1 } { 3 N _ { v } } \sum _ { i , t } m _ { i , t } ^ { v } \sum _ { a \in \{ x , y , z \} } \rho _ { \beta } \bigl ( \widehat { v } _ { i , t , a } - v _ { i , t , a } ^ { \mathrm { g t } } \bigr ) ,\tag{39}
$$

where $m _ { i , t } ^ { v }$ is one only when both consecutive frames are valid, and $\begin{array} { r } { N _ { v } = \sum _ { i , t } m _ { i , t } ^ { v } } \end{array}$ is the corresponding number of valid frame pairs.

A second-order temporal difference is defined as

$$
\widehat { \mathbf { a } } _ { i , t } = \widehat { \mathbf { r } } _ { i , t } ^ { c } - 2 \widehat { \mathbf { r } } _ { i , t - 1 } ^ { c } + \widehat { \mathbf { r } } _ { i , t - 2 } ^ { c } ,\tag{40}
$$

with the corresponding ground-truth quantity $\mathbf { a } _ { i , t } ^ { \mathrm { g t } }$ . The second order temporal loss is

$$
\mathcal { L } _ { \mathrm { a c c } } = \frac { 1 } { 3 N _ { a } } \sum _ { i , t } m _ { i , t } ^ { a } \sum _ { a \in \{ x , y , z \} } \rho _ { \beta } \bigl ( \widehat { a } _ { i , t , a } - a _ { i , t , a } ^ { \mathrm { g t } } \bigr ) ,\tag{41}
$$

where $m _ { i , t } ^ { a }$ requires the three consecutive frames to be valid, and $\begin{array} { r } { N _ { a } = \sum _ { i , t } m _ { i , t } ^ { a } } \end{array}$

The reconstructed camera-space pose is

$$
\widehat { \mathbf { X } } _ { i , t , j } ^ { c } = \widehat { \mathbf { X } } _ { i , t , j } ^ { \mathrm { r e l } } + \widehat { \mathbf { r } } _ { i , t } ^ { c } .\tag{42}
$$

A masked 3D consistency loss compares the reconstructed joints with the full camera-space ground truth. Let

$$
N _ { j } = \sum _ { i , t , j } v _ { i , t } m _ { i , t , j }
$$

Table 2. Training configuration of the selected temporal refinement models.
<table><tr><td>Training setting</td><td>TCN</td><td>Transformer</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Initial learning rate</td><td> $1 . 6 7 8 8 \times 1 0 ^ { - 4 }$ </td><td> $1 . 6 7 8 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $3 . 2 0 0 1 \times 1 0 ^ { - 4 }$ </td><td> $3 . 2 0 0 1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning-rate scheduler</td><td>None</td><td>None</td></tr><tr><td>Batch size</td><td>64 windows</td><td>64 windows</td></tr><tr><td>Maximum epochs</td><td>60</td><td>60</td></tr><tr><td>Early-stopping patience</td><td>10 epochs</td><td>10 epochs</td></tr><tr><td>Selection criterion</td><td>Validation root error</td><td>Validation root error</td></tr><tr><td>Gradient clipping</td><td>l2 norm, max. 1.0</td><td> $\ell _ { 2 }$  norm, max. 1.0</td></tr><tr><td>Best epoch</td><td>41</td><td>23</td></tr><tr><td>Last training epoch</td><td>51</td><td>33</td></tr></table>

denote the number of valid player–frame–joint observations. We use

$$
\begin{array} { c l } { \displaystyle \mathcal { L } _ { \mathrm { c a m 3 D } } = \frac { 1 } { 3 N _ { j } } \sum _ { i , t , j } v _ { i , t } m _ { i , t , j } } \\ { \displaystyle \times \sum _ { a \in \{ x , y , z \} } \rho _ { 1 } \left( \widehat { X } _ { i , t , j , a } ^ { c } - X _ { i , t , j , a } ^ { c , \mathrm { g t } } \right) . } \end{array}\tag{43}
$$

Here, $m _ { i , t , j }$ is the joint-validity mask and $v _ { i , t }$ is the player– frame validity mask. Since the relative skeleton is not refined, this term provides only auxiliary supervision.

The overall objective is

$$
\mathcal { L } = \lambda _ { \mathrm { { r o o t } } } \mathcal { L } _ { \mathrm { { r o o t } } } + \lambda _ { \mathrm { { v e l } } } \mathcal { L } _ { \mathrm { { v e l } } } + \lambda _ { \mathrm { { a c c } } } \mathcal { L } _ { \mathrm { { a c c } } } + \lambda _ { \mathrm { { c a m 3 D } } } \mathcal { L } _ { \mathrm { { c a m 3 D } } } .\tag{44}
$$

Both models are optimized with AdamW using an initial learning rate of $1 . 7 \times 1 0 ^ { - 4 }$ and a weight decay of $3 . 2 \times 1 0 ^ { - 4 }$ No learning-rate scheduler is used. Training uses batches of 64 temporal windows for at most 60 epochs. Gradients are clipped to a maximum global $\ell _ { 2 }$ norm of 1. Model selection is based on the lowest mean root error on the validation set, with early stopping after 10 consecutive epochs without improvement.Table 2 describes the training configuration that we used.

## 3.8. Hyperparameter Selection

Hyperparameter selection was first conducted using the TCN. We performed a 12-trial random search over architectural, temporal, optimization, and loss-related parameters. Each trial was ranked using the mean root translation error on the validation set. Starting from the best configuration, focused grid searches evaluated six window–stride combinations and 16 combinations of auxiliary loss weights.

The Transformer was optimized independently using a multifidelity random-search procedure. We evaluated 16 configurations, comprising the reference configuration and 15 randomly sampled alternatives. The search varied the temporal window and stride, encoder and latent dimensions, number of self-attention layers and heads, feed-forward dimension, dropout rate, positional encoding, prediction-head dimension, learning rate, and weight decay. The loss weights, batch size, gradient-clipping threshold, and early-stopping criterion were kept fixed during this search.

## 3.9. World-Space Reconstruction

Once the root has been refined and denormalized, each global camera-space joint is obtained using Eq. (4). The final world-space position of joint j is

$$
\widehat { \mathbf { x } } _ { i , t , j } ^ { w } = \mathbf { R } _ { t } ^ { \top } \left( \widehat { \mathbf { x } } _ { i , t , j } ^ { c } - \mathbf { t } _ { t } \right) , \qquad j = 1 , \ldots , J .\tag{45}
$$

All players are therefore represented in the same metric soccer-field coordinate system.

If an upstream body mesh is available, the same translation can be applied to every relative mesh vertex. For vertex k,

$$
\widehat { \mathbf { v } } _ { i , t , k } ^ { c } = \widehat { \mathbf { v } } _ { i , t , k } ^ { \mathrm { r e l } } + \widehat { \mathbf { r } } _ { i , t } ^ { c } ,\tag{46}
$$

followed by

$$
\widehat { \mathbf { v } } _ { i , t , k } ^ { w } = \mathbf { R } _ { t } ^ { \top } \left( \widehat { \mathbf { v } } _ { i , t , k } ^ { c } - \mathbf { t } _ { t } \right) .\tag{47}
$$

This provides world-grounded meshes for visualization or free-viewpoint rendering while leaving the original relative body shape and articulation unchanged.

## 3.10. Evaluation Protocol

We evaluate global localization using the Euclidean error of the predicted camera-space root (Root error) and the mean per-joint position error after reconstruction in the common field coordinate system (World MPJPE). We additionally report Local MPJPE, obtained after removing the global root translation, to separate errors originating from the upstream relative pose estimator from errors due to global localization. Finally, reprojection error measures the mean 2D distance, in pixels, between the reconstructed 3D joints projected through the calibrated camera and their reference 2D locations. All metrics are computed on valid player–frame and joint observations only.

Since all compared root-refinement models use the same upstream relative 3D pose, Local MPJPE is identical by construction across the geometry, MLP, TCN, and Transformer variants. This metric is therefore primarily reported to quantify the remaining local-pose error independently of the global localization stage.

## 4. Results

## 4.1. Main Quantitative Results

Results. Table 3 summarizes the main quantitative results. The geometry-based initialization yields a root error of 49 cm. Learning a residual correction substantially improves this estimate: the frame-wise MLP reduces the root error to 14 cm. Adding temporal context further improves global localization, with root errors of 10 cm for the TCN and 11 cm for the Transformer. The corresponding World MPJPE is 13 cm for both temporal models, compared with 16 cm for the frame-wise MLP. The Local MPJPE remains 8 cm for all variants.

Interpretation. Residual learning reduces the root error by approximately 72% relative to the geometry-based initialization even without temporal modeling. The additional improvement obtained by the TCN and Transformer shows that temporal information is beneficial for global player localization. However, the nearly identical World MPJPE obtained by the two temporal architectures suggests that the availability of temporal context is more important than the specific choice between convolutional and attention-based modeling. The unchanged Local MPJPE further confirms that the performance gains originate from improved global localization rather than changes to the relative body pose.

## 4.2. Ablation Studies

## 4.2.1. ABSOLUTE VERSUS RESIDUAL ROOT PREDICTION

Results. Table 4 compares direct absolute-root prediction with the proposed residual formulation. Direct root prediction results in errors of 2.56 m for the frame-wise MLP and 63 cm for the TCN. The geometry-only initialization reaches 49 cm. When the same learning models instead predict a residual correction with respect to this initialization, the errors decrease to 14 cm for the MLP and 10 cm for the TCN.

Interpretation. The large performance gap between direct and residual prediction supports the proposed decomposition of the problem. Camera and field geometry provide a strong metric prior, considerably reducing the range of translations that must be inferred from visual observations. The learning model can therefore focus on estimating the context-dependent error of this initialization rather than recovering the complete global player position from scratch.

## 4.2.2. INPUT CUES

Results. Table 5 reports the contribution of each input modality for the TCN model. Removing the valid-joint mask increases root error from 10 to 11 cm. Removing the ray– ground intersection also results in an error of approximately 11 cm. Without the relative 3D pose, root error increases to 12 cm, while removing the camera descriptor increases it to 13 cm. The strongest degradation is observed when removing the 2D pose and bounding-box cues, yielding a root error of 16 cm and a World MPJPE of 18 cm.

Table 3. Main quantitative results. All learning-based models predict a residual correction to the geometry-based root initialization. The relative 3D pose is shared across all methods and is therefore unchanged by root refinement.
<table><tr><td>Method</td><td>Root error ↓ (cm)</td><td>World MPJPE ↓ (cm)</td><td>Local MPJPE↓(cm)</td><td>Reproj. ↓ (px)</td><td>Params. (M)</td></tr><tr><td>Geometry initialization</td><td>48.58</td><td>48.36</td><td>7.74</td><td>5.39</td><td>0</td></tr><tr><td>MLP residual</td><td>13.63</td><td>15.76</td><td>7.74</td><td>3.69</td><td>0.194</td></tr><tr><td>TCN residual (41 frames)</td><td>10.12</td><td>13.20</td><td>7.74</td><td>3.49</td><td>1.278</td></tr><tr><td>Transformer residual (41 frames)</td><td>11.04</td><td>13.21</td><td>7.74</td><td>3.43</td><td>1.252</td></tr></table>

Table 4. Effect of the prediction target. Direct global-root regression is compared with geometry-based initialization and residual (∆ root) refinement.
<table><tr><td>Initialization</td><td>Target</td><td>Temporal</td><td>Root error ↓ (cm)</td></tr><tr><td></td><td>absolute root</td><td>MLP</td><td>256.00</td></tr><tr><td></td><td>absolute root</td><td>TCN</td><td>63.05</td></tr><tr><td>Geometry</td><td></td><td></td><td>48.58</td></tr><tr><td>Geometry</td><td>∆ root</td><td>MLP</td><td>13.63</td></tr><tr><td>Geometry</td><td>∆ root</td><td>TCN</td><td>10.12</td></tr></table>

Interpretation. No single input modality fully determines global player localization. The explicit ray–ground intersection remains useful even though it is already involved in constructing the initial root, indicating that the network benefits from direct access to the underlying geometric cue. Relative 3D pose and camera information provide complementary information about body configuration and scene geometry. The particularly large degradation observed without 2D pose and bounding-box features shows that image-space position and apparent player scale remain important cues for correcting metric depth and global translation.

## 4.3. Analysis of the Geometric Refinement

## 4.3.1. ROBUSTNESS TO INITIALIZATION ERROR

Results. Figure 4 reports final root error as a function of the geometry-based initialization error. For already accurate geometric estimates, the learned correction provides little improvement and can slightly perturb an initialization that is already close to the ground truth. As initialization error increases, however, the temporal models remain substantially more stable. The TCN maintains a root error of approximately 8–11 cm over a broad range of initialization errors, before increasing to approximately 19 cm in the most difficult bin, where the geometric initialization exceeds one meter on average.

Interpretation. The residual model is therefore not limited to producing small local adjustments around the geometric estimate. When the initialization becomes strongly biased, the network can use pose, image, camera, and temporal information to recover a substantially more accurate global position. Conversely, the small gain obtained when the initialization is already accurate suggests that most of the benefit of learning is concentrated on geometrically ambiguous or incorrectly grounded configurations.

![](images/9d3857de34a8e93f88218375e908cf6b17624ae31a3990c566cae6fc811f393a.jpg)  
Figure 4. Root error as a function of the geometry-based initialization error. Temporal residual refinement remains substantially more stable than the raw geometric estimate as initialization quality deteriorates.

## 4.3.2. AIRBORNE PLAYERS AND FAILURE CASES

Results. Figure 5a reports root error as a function of the clearance of the lower foot from the playing surface. When the lower foot is close to the ground, the TCN achieves approximately 10 cm root error. The error progressively increases with foot clearance, reaching approximately 15 cm at 10–13 cm clearance, 29 cm at 15–18 cm, and 37 cm in the highest-clearance interval. The same trend is considerably stronger for the geometry-based initialization, whose error increases from approximately 46 cm for grounded configurations to almost one meter for the largest foot clearances. Figure 5b illustrates an example during a heading action, where the predicted pose remains locally plausible but is globally displaced relative to the ground truth.

Interpretation. These results expose a direct limitation of the geometric initialization. The ray–ground construction implicitly assumes that the selected lower-limb joint lies close to the playing surface. When the player becomes airborne, the corresponding image ray intersects the pitch behind the true 3D joint location, producing a systematically biased root estimate. Temporal refinement compensates for a substantial part of this error, but does not completely remove the underlying ambiguity. Airborne motion therefore remains one of the main failure modes of the current approach and motivates future refinements that explicitly model ground contact or airborne states.

Table 5. Input ablation for the TCN residual model. ∆ Root denotes the increase in root error relative to the full model.
<table><tr><td>Model</td><td>Input configuration</td><td>Root↓(cm)</td><td>∆ Root (cm)</td><td>World MPJPE↓ (cm)</td><td>Reproj. ↓ (px)</td></tr><tr><td>TCN</td><td>Full</td><td>10.12</td><td>0.00</td><td>13.20</td><td>3.49</td></tr><tr><td>TCN</td><td>w/o valid-joint mask  $\mathbf { m } _ { i , t }$ </td><td>10.75</td><td>0.63</td><td>13.53</td><td>3.41</td></tr><tr><td>TCN</td><td>w/o ground intersection  $\mathbf { q } _ { i , t } ^ { w }$ </td><td>11.17</td><td>1.05</td><td>13.87</td><td>3.43</td></tr><tr><td>TCN</td><td>w/o relative 3D pose  ${ \bf f } ^ { 3 D }$ </td><td>12.40</td><td>2.28</td><td>14.93</td><td>3.70</td></tr><tr><td>TCN</td><td>w/o 2D pose and box cues  $\mathbf { f } ^ { 2 D }$  fbox</td><td>15.93</td><td>5.81</td><td>18.20</td><td>5.54</td></tr><tr><td>TCN</td><td>w/o camera descriptor  $\mathbf { f } ^ { c a m }$ </td><td>13.02</td><td>2.90</td><td>15.39</td><td>3.50</td></tr></table>

## 4.4. Additional Analyses

Results. Additional quantitative analyses are provided in the supplementary material. They include comparisons of model complexity and temporal consistency, performance across different motion regimes, sensitivity to image location, player scale and speed, and a decomposition of root localization error along the camera axes.

Interpretation. These complementary analyses further characterize the behavior of the proposed refinement model without overloading the main paper. In particular, they show that temporal modeling is especially beneficial in difficult motion regimes and that the dominant remaining localization uncertainty is associated with camera depth.

## 5. Conclusion

We introduced Field Converter, a geometry-initialized temporal residual framework for world-grounded 3D player pose estimation from monocular soccer broadcasts. Rather than regressing global player position directly, the proposed approach first exploits calibrated camera and pitch geometry to obtain a metric root initialization and then learns only its temporal residual correction. This decomposition substantially improves global localization while preserving the relative 3D pose estimated upstream. Our experiments show that residual prediction is markedly more effective than direct root regression, that temporal modeling further improves both localization accuracy and trajectory consistency, and that TCN and Transformer backbones achieve comparable performance. The ablation study further confirms that image-space pose, bounding-box, camera, relative-pose, and ray–ground cues provide complementary information. Overall, these results support the central premise of this work: in this broadcast-soccer setting, accurate world-grounded reconstruction depends less on re-estimating local body articulation than on robustly recovering the global translation that anchors this articulation to the field.

The current approach nevertheless relies on several assumptions that leave room for improvement. In particular, the geometry-based initialization assumes that the selected lower-limb joint lies close to the playing surface, which becomes invalid during airborne actions such as jumps and headers and leads to substantially larger localization errors. The method also depends on upstream player detections, pose estimates, and camera calibration, so errors in these components may propagate to the final reconstruction. Future work will therefore investigate explicit modeling of ground-contact and airborne states, improved geometric initialization when no reliable ground contact is available, and training strategies that better represent rare high-clearance motions.

## References

Baradel, F., Armando, M., Galaaoui, S., Brégier, R., Weinzaepfel, P., and Rogez, G. Multi-hmr: Multi-person whole-body human mesh recovery in a single shot. In European Conference on Computer Vision, 2024.

Baumgartner, T. and Klatt, S. Monocular 3d human pose estimation for sports broadcasts using partial sports field registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 5109–5118, 2023.

Dos’Santos, T., Thomas, C., McBurnie, A., Comfort, P., and Jones, P. A. Biomechanical determinants of performance and injury risk during cutting: A performance-injury conflict? Sports Medicine, 51(9):1983–1998, 2021. doi: 10.1007/s40279-021-01448-3.

Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A., and Malik, J. Humans in 4d: Reconstructing and tracking humans with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14783– 14794, 2023.

![](images/62b06797c56e0ecbe775c8ad76bdd0dfa2784980f40a58979f7c7e66042598ac.jpg)  
(a) Root error as a function of lower-foot clearance above the pitch.

![](images/046673acb318c4792f63cba92dd9bc7de267a107f47ff834cf7976094ddf41ba.jpg)  
(b) Qualitative airborne failure case during a header. Ground truth is shown in red and the reconstructed pose in orange.  
Figure 5. Airborne-player analysis. The geometric assumption is increasingly violated as both feet move away from the playing surface, producing larger initialization and final localization errors. Temporal refinement mitigates, but does not entirely remove, this failure mode.

Gutiérrez-Pérez, M. and Agudo, A. No bells, just whistles: Sports field registration by leveraging geometric properties. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024.

Gutiérrez-Pérez, M. and Agudo, A. Pnlcalib: Sports field registration via points and lines optimization. Computer Vision and Image Understanding, 2026.

Hilton, A., Guillemaut, J.-Y., Kilner, J., Grau, O., and Thomas, G. Free-viewpoint video for tv sport production. In Ronfard, R. and Taubin, G. (eds.), Image and Geometry Processing for 3-D Cinematography, volume 5 of Geometry and Computing, pp. 77–106. Springer, 2010. doi: 10.1007/978-3-642-12392-4\_4.

Jiang, T., Billingham, J., Müksch, S., Zarate, J. J., Evans, N., Oswald, M. R., Pollefeys, M., Hilliges, O., Kaufmann, M., and Song, J. Worldpose: A world cup dataset for global 3d human pose estimation. In European Conference on Computer Vision, 2024.

Kocabas, M., Huang, C.-H. P., Tesch, J., Müller, L., Hilliges, O., Black, M. J., Kautz, J., and Iqbal, U. Pace: Human and camera motion estimation from in-the-wild videos. In International Conference on 3D Vision, 2024.

Li, J., Xu, C., Chen, Z., Bian, S., Yang, L., and Lu, C. Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation. In

Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3383–3393, 2021.

Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., and Black, M. J. Smpl: A skinned multi-person linear model. ACM Transactions on Graphics, 34(6):248:1–248:16, 2015.

Magera, F., Hoyoux, T., Barnich, O., and Van Droogenbroeck, M. Broadtrack: Broadcast camera tracking for soccer. In Proceedings ofthe Winter Conference on Applications ofComputer Vision, pp. 6177–6187, 2025.

Mehraban, S., Adeli, V., and Taati, B. Motionagformer: Enhancing 3d human pose estimation with a transformergcnformer network. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 6920–6930, 2024.

Memmert, D., Raabe, D., Schwab, S., and Rein, R. A tactical comparison of the 4-2-3-1 and 3-5-2 formation in soccer: A theory-oriented, experimental approach based on positional data in an 11 vs. 11 game set-up. PLOS ONE, 14(1):e0210191, 2019. doi: 10.1371/journal.pone. 0210191.

Mur-Artal, R., Montiel, J. M. M., and Tardós, J. D. ORB-SLAM: A versatile and accurate monocular SLAM system. IEEE Transactions on Robotics, 31(5):1147–1163, 2015. doi: 10.1109/TRO.2015.2463671.

Oštrek, M., Rhodin, H., Spörri, J., Müller, E., and Seidel, T. Are existing monocular computer vision-based 3d motion capture approaches ready for deployment? a methodological study on the example of alpine skiing. Sensors, 19 (19):4323, 2019.

Pavllo, D., Feichtenhofer, C., Grangier, D., and Auli, M. 3d human pose estimation in video with temporal convolutions and semi-supervised training. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7753–7762, 2019.

Shen, Z., Pi, H., Xia, Y., Cen, Z., Peng, S., Bao, H., and Zhou, X. World-grounded human motion recovery via gravity-view coordinates. ACM Transactions on Graphics, 2024.

Shin, S., Kim, J., Halilaj, E., and Black, M. J. Wham: Reconstructing world-grounded humans with accurate 3d motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2070– 2080, 2024.

Theiner, J. and Ewerth, R. Tvcalib: Camera calibration for sports field registration in soccer. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 1166–1175, 2023.

Wang, H., Mills, K., Billingham, J., Robertson, S., and Hosoi, A. E. Semi-automated last touch detection for outof-bounds possession decisions in football. Sports Engineering, 28:36, 2025. doi: 10.1007/s12283-025-00518-3.

Wang, Y., Wang, Z., Liu, L., and Daniilidis, K. Tram: Global trajectory and motion of 3d humans from in-thewild videos. In European Conference on Computer Vision, 2024.

Wen, J., Gold, L., Ma, Q., and LiKamWa, R. Augmented coach: Volumetric motion annotation and visualization for immersive sports coaching. In 2024 IEEE Conference on Virtual Reality and 3D User Interfaces (VR), pp. 137– 146, 2024. doi: 10.1109/VR58804.2024.00037.

Yang, X. et al. Sam 3d body: Robust full-body human mesh recovery. arXiv preprint arXiv:2602.15989, 2026.

Ye, V., Pavlakos, G., Malik, J., and Kanazawa, A. Decoupling human and camera motion from videos in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Yeung, C. et al. Autosoccerpose: Automated 3d posture analysis of soccer shot movements. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024.

Yeung, C. et al. Athletepose3d: A benchmark dataset for 3d human pose estimation and kinematic validation in athletic movements. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2025.

Yuan, Y., Iqbal, U., Molchanov, P., Kitani, K., and Kautz, J. Glamr: Global occlusion-aware human mesh recovery with dynamic cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11038–11049, 2022.

Zhan, Y., Li, F., Weng, R., and Choi, W. Ray3d: Ray-based 3d human pose estimation for monocular absolute 3d localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13116– 13125, 2022.

Zhang, J., Tu, Z., Yang, J., Chen, Y., and Yuan, J. Mixste: Seq2seq mixed spatio-temporal encoder for 3d human pose estimation in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13232–13242, 2022.

Zhang, Y. et al. Proxycap: Real-time monocular fullbody capture in world space via human-centric proxyto-motion learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., and Wang, Y. Motionbert: A unified perspective on learning human motion representations. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 15085– 15099, 2023.

Table 6. Comparison of the frame-wise and temporal architectures. The number of trainable parameters is reported together with the main localization metrics.
<table><tr><td>Model</td><td>Root error ↓ (cm)</td><td>World MPJPE ↓ (cm)</td><td>Params. (M)</td></tr><tr><td>MLP</td><td>13.63</td><td>15.76</td><td>0.194</td></tr><tr><td>TCN (41 f)</td><td>10.12</td><td>13.20</td><td>1.278</td></tr><tr><td>Transformer (41 f)</td><td>11.04</td><td>13.21</td><td>1.252</td></tr></table>

## A. Supplementary Material

This supplementary material provides additional quantitative analyses of the proposed root-refinement framework. We report complementary comparisons of model architecture and temporal consistency, extend the input ablation to both temporal backbones, and analyze performance as a function of motion and observation conditions.

## A.1. Architecture and Model Complexity

Table 6 compares the frame-wise MLP, TCN, and Transformer residual models in terms of localization accuracy and number of trainable parameters.

The MLP contains substantially fewer parameters than the two temporal models, but also exhibits a higher root error. Introducing temporal context provides a clear improvement in global localization. Despite relying on different temporal mechanisms, the TCN and Transformer have similar model sizes and achieve comparable World MPJPE. The TCN obtains the lowest root error, whereas the Transformer provides a slightly lower reprojection error.

These results reinforce the observation that exploiting temporal information is more important in our setting than the specific choice between convolutional and attention-based temporal modeling.

## A.2. Extended Input Ablation

The main paper reports the input ablation using the TCN. Table 7 provides the corresponding results for both the TCN and Transformer.

Overall, the two architectures exhibit similar trends. Removing image-space pose and bounding-box cues produces the strongest degradation for both models. Camera information is also important, particularly for the Transformer. Removing the explicit ray–ground intersection deteriorates performance for both temporal backbones, supporting its use as an informative geometric cue in addition to its role in constructing the initial root estimate.

The relative 3D pose provides complementary information about the player’s body configuration, while the valid-joint mask improves robustness to incomplete or unreliable observations. The consistency of these trends across two different temporal architectures suggests that the observed gains primarily originate from the information contained in the features rather than from architecture-specific behavior.

## A.3. Temporal Consistency

In addition to positional accuracy, we evaluate the consistency of the predicted root trajectories through first- and second-order temporal differences.

Table 8 reports root-position, velocity, and acceleration errors for the frame-wise MLP and both temporal models. The MLP predicts each frame independently and consequently exhibits larger temporal errors. Both the TCN and Transformer considerably reduce velocity and acceleration errors while simultaneously improving root localization.

The TCN achieves the lowest root-position and velocity errors, whereas the Transformer obtains a marginally lower acceleration error. The proximity of their results again indicates that most of the gain comes from temporal context itself rather than from a specific temporal architecture.

## A.4. Temporal Gain Across Motion Regimes

Figure 6 analyzes the benefit of temporal modeling under different motion conditions. The improvement is reported relative to the frame-wise MLP.

Temporal modeling is beneficial across all evaluated regimes, but the improvement is larger in difficult situations. In particular, the TCN provides larger relative gains for airborne players, high-speed motion, and frames associated with large geometry-initialization errors than for grounded configurations.

This observation suggests that temporal context becomes particularly useful when instantaneous geometric and imagespace observations are ambiguous. Neighboring frames can then provide information about the player’s trajectory that is unavailable from a single frame.

## A.5. Sensitivity to Image-Space Position

Figure 7 reports root error as a function of the distance between the player and the image center.

No severe degradation is observed as players move away from the center of the broadcast image. This indicates that the model does not rely exclusively on a narrow image region and can exploit the camera and geometric descriptors to accommodate different viewing configurations.

## A.6. Sensitivity to Player Image Scale

Figure 8 evaluates root localization as a function of player bounding-box height.

Localization is more difficult for players occupying fewer pixels in the image, as expected in broadcast footage. Smaller players provide less accurate image-space body observations and weaker perspective cues. Performance progressively improves as the apparent player size increases.

Table 7. Extended input ablation for the TCN and Transformer residual models. ∆ Root denotes the increase in root error with respect to the corresponding full model.
<table><tr><td>Model</td><td>Input configuration</td><td>Root↓(cm)</td><td>∆ Root (cm)</td><td>World MPJPE↓(cm)</td><td>Reproj. ↓ (px)</td></tr><tr><td>TCN</td><td>Full</td><td>10.12</td><td>0.00</td><td>13.20</td><td>3.49</td></tr><tr><td>TCN</td><td>w/o valid-joint mask  $\mathbf { m } _ { i , t }$ </td><td>10.75</td><td>0.63</td><td>13.53</td><td>3.41</td></tr><tr><td>TCN</td><td>w/o ground intersection  $\mathbf { q } _ { i , t } ^ { w }$ </td><td>11.17</td><td>1.05</td><td>13.87</td><td>3.43</td></tr><tr><td>TCN</td><td>w/o relative 3D pose  ${ \bf f } ^ { 3 D }$ </td><td>12.40</td><td>2.28</td><td>14.93</td><td>3.70</td></tr><tr><td>TCN</td><td>w/o 2D pose and box cues  $\mathbf { f } ^ { 2 D }$  , fbox</td><td>15.93</td><td>5.81</td><td>18.20</td><td>5.54</td></tr><tr><td>TCN</td><td>w/o camera descriptor  $\mathbf { f } ^ { c a m }$ </td><td>13.02</td><td>2.90</td><td>15.39</td><td>3.50</td></tr><tr><td>Transformer</td><td>Full</td><td>11.04</td><td>0.00</td><td>13.21</td><td>3.43</td></tr><tr><td>Transformer</td><td>w/o valid-joint mask  $\mathbf { m } _ { i , t }$ </td><td>13.42</td><td>2.39</td><td>15.03</td><td>3.69</td></tr><tr><td>Transformer</td><td>w/o ground intersection  $\mathbf { q } _ { i , t } ^ { w }$ </td><td>12.02</td><td>0.98</td><td>13.91</td><td>3.50</td></tr><tr><td>Transformer</td><td>w/o relative 3D pose  ${ \bf f } ^ { 3 D }$ </td><td>12.28</td><td>1.24</td><td>14.34</td><td>3.68</td></tr><tr><td>Transformer</td><td>w/o 2D pose and box cues  $\mathbf { f } ^ { 2 D }$  , fbox</td><td>15.81</td><td>4.77</td><td>17.41</td><td>5.20</td></tr><tr><td>Transformer</td><td>w/o camera descriptor  $\mathbf { f } ^ { c a m }$ </td><td>15.46</td><td>4.42</td><td>17.25</td><td>4.11</td></tr></table>

Table 8. Temporal consistency of the predicted root trajectories. Lower values indicate more accurate root position, velocity, and acceleration.
<table><tr><td>Model</td><td>Root error ↓ (cm)</td><td>Root velocity error ↓ (cm s−1)</td><td>Root acceleration error ↓ (m s−2)</td></tr><tr><td>MLP</td><td>13.63 [12.50, 15.08]</td><td>131.9 [124.1, 142.2]</td><td>48.3 [45.6, 51.7]</td></tr><tr><td>TCN (41 f)</td><td>10.12 [9.20, 11.33]</td><td>100.6 [95.3, 107.6]</td><td>39.6 [37.7, 42.2]</td></tr><tr><td>Transformer (41 f)</td><td>11.04 [9.96, 12.42]</td><td>101.9 [95.7, 109.8]</td><td>39.0 [36.7, 42.1]</td></tr></table>

![](images/a5604f6bbd0d15db7d0f9647210ed28e698f9fbc8070cfcc4b346108ed1baf07.jpg)  
Figure 6. Relative improvement of temporal models over the frame-wise MLP across different motion regimes.

This result highlights image resolution as an important source of uncertainty for world-grounded reconstruction, particularly for players located far from the broadcast camera.

## A.7. Sensitivity to Player Motion

Figure 9 analyzes root error as a function of player speed.

![](images/6edd9a49ad5a9155af0421b5dfded993403dc86c8ffa71e5dabc1e90f4adcce6.jpg)  
Figure 7. Root error as a function of player distance from the image center.

Localization error increases for the fastest motions. Highspeed actions are associated with larger frame-to-frame displacement, stronger motion blur, rapidly changing body configurations, and potentially less reliable instantaneous keypoints. Nevertheless, the temporal models remain more accurate than the frame-wise baseline in this regime, consistent with the results of Sec. A.4.

## A.8. Root Error Along Camera Axes

To better understand the source of global localization error, Fig. 10 decomposes the root error along the camera coordinate axes.

The geometry-based initialization is substantially more accurate in the lateral image-plane directions than along camera depth. Temporal residual refinement strongly reduces all three components, but depth remains the dominant source of localization error.

![](images/69cfc0131cc235d6f5bb32ddc451dc88c964744c20f2356a60f660d3b2cf5c89.jpg)

Figure 8. Root error as a function of player bounding-box height in the image.  
![](images/b2a9dabf2cfc2777035db21852056e4a2fd01e4ddd20e875010f84fe98f56e6f.jpg)  
Figure 9. Root error as a function of player speed.

This behavior is consistent with the inherent ambiguity of monocular reconstruction: small image-space errors can correspond to large metric displacements along the viewing direction. The result further motivates the use of temporal and scene-geometric cues for global player localization.

## A.9. Pelvis Height Analysis

For completeness, Fig. 11 reports root error as a function of pelvis height above the pitch.

Unlike lower-foot clearance, pelvis height does not provide a direct indicator of ground contact. It varies with player morphology and body configuration, including flexion and extension during otherwise grounded movements. Consequently, its relationship with localization error is less direct than the airborne analysis presented in the main paper.

We therefore use lower-foot clearance as the primary measure for analyzing violations of the ground-contact assumption and provide pelvis-height results only as a complementary diagnostic.

![](images/2755449e8bd8bfb403fce4fe1be0a90629600bc5768ef5bd5a5eaf3297353a44.jpg)

Figure 10. Root localization error decomposed along the camera coordinate axes.  
![](images/26e2048bbda5c49866b1797c3014619d35e09187b30d5a0578b004abbd78162b.jpg)  
Figure 11. Root error as a function of pelvis height above the playing surface.

## A.10. Measured Training and Evaluation Throughput.

Runtime was measured on a single NVIDIA L40S GPU (46,068 MiB visible memory) using PyTorch 2.6 and CUDA 12.4. We considered all 29 successfully completed full-data runs from the unseen-match protocol: 2 MLP, 13 TCN, and 14 Transformer runs. $\mathbf { A } \mathbf { s }$ shown in Fig. 12, mean training throughput was $( 2 . 9 5 \pm 0 . 0 3 ) \times 1 0 ^ { 3 }$ $( 2 5 . 5 \pm 2 0 . 8 ) \times 1 0 ^ { 3 }$ , and $( 6 1 . 4 \pm 2 5 . 5 ) \times 1 0 ^ { 3 }$ input playerframes $\mathrm { s } ^ { - 1 }$ for the MLP, TCN, and Transformer, respectively. End-to-end batched evaluation, including data loading, temporal-window aggregation where applicable, metric computation, and prediction serialization, reached 530±290, $4 6 7 \pm 1 0 7$ , and $5 5 1 \pm 1 4 1$ video frames $\mathrm { s } ^ { - 1 }$ . Values denote mean ± standard deviation across successful configurations; the dispersion therefore reflects variations in input features and temporal context rather than repeated-run confidence intervals. These measurements exclude upstream player tracking, camera calibration, and SAM3DBody inference.

![](images/b45073491357454f304e5264cf8ed1c6c7f453db140ba8d0ee83c982c9b274b1.jpg)

![](images/288b9f79805272098580c7a41166c45f09f16ef493fec8f86dd226d7ad8b3b9b.jpg)  
Figure 12. Computational throughput of the MLP, TCN, and Transformer architectures on an NVIDIA L40S GPU. (a) Training throughput in input player-frames per second. (b) End-to-end inference throughput in video frames per second. Each hollow marker represents one successfully completed run, while filled markers and error bars indicate the mean and standard deviation across runs. Training throughput counts the player-frame tokens processed by the optimization loop, including overlapping temporal windows.