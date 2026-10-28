# Camera-to-Quadruped Robot Pose Estimation from Sequential Images

Seobin Hwang, Ji-Seong Beom, and Yeong-Jun Cho<sup>⋆</sup>

Department of Artificial Intelligence Convergence, Chonnam National University, Gwangju, Republic of Korea {cnu.cvl.hsb, haa0901}@gmail.com, {yj.cho}@jnu.ac.kr

![](images/e8e0f683faa064bad69b57a52f7dfe333f0321b38ee999d3df738bb767fe8f3f.jpg)  
① Articulated system base 6-DoF + joints 12-DoF = 18-DoF

![](images/317d0f235f21c3f3cf59f3fc665e3536c1ca5b6c3d0116ad27c9dd9d2d75bfd8.jpg)  
(a) Problem setup and challenges

![](images/b151f3486101215480a5b0f1aea11867a97a80cdaf0fe48fffede831b6784057.jpg)  
(b) Proposed method: QuadPose  
Fig. 1: Overview of camera-to-quadruped robot pose estimation. (a) Problem setup and challenges: an articulated quadruped model (6-DoF base + 12-DoF joints) and 2D keypoint ambiguities caused by self-occlusion and leg symmetry. (b) Overview of the proposed QuadPose framework and the QuadSyn dataset.

Abstract. Quadruped robots are increasingly deployed in hazardous environments such as disaster response and facility inspection, where external-camera monitoring is essential for safe operation. This setting is challenging because quadrupeds are articulated systems with time-varying leg joints, leg symmetry, and frequent self-occlusions, which make keypoint correspondence ambiguous. Moreover, no public benchmark exists, making quantitative comparison dificult. To address these challenges, we propose QuadPose, which estimates the full-body 18-DoF pose (6-DoF base pose + 12-DoF joint angles) from monocular video sequences, and introduce QuadSyn, the first camera-to-quadruped robot pose benchmark.

On QuadSyn, QuadPose achieves 92.18% PCK@2.5 px, 95.54% PCK-AUC, 85.94% ADD@20 cm, 66.98% ADD-AUC, and 3.65<sup>◦</sup> mean joint-angle error at approximately 33 FPS. Code and dataset will be publicly available.

Keywords: Quadruped Robot · Camera-to-Robot Pose Estimation · Robot Pose Benchmark

## 1 Introduction

Quadruped robots are increasingly deployed for patrol, inspection, and monitoring in hazardous environments [4], where real-time external-camera monitoring is essential for safe operation. However, 2D tracking only provides the robot’s location in the image plane, which is insuficient to assess whether it is stable, moving as intended, or approaching dangerous states. Efective monitoring therefore requires estimating the robot’s full-body 3D pose, including the 6-DoF base pose (position and orientation) and the articulated joint configuration.

In practice, such monitoring often relies on fixed external cameras due to cost, maintenance, and reliability constraints. As a result, only monocular video is available, without robot-side sensors (e.g., an IMU) or robot–camera extrinsic calibration. Under these conditions, depth and 3D orientation remain inherently ambiguous from 2D boxes or keypoints, making it difficult to consistently relate image observations to the robot’s kinematic state over time. Estimating the robot’s full-body 3D pose from a monocular camera remains a key challenge.

This problem can be formulated as camera-to-robot pose estimation from external videos. Existing approaches broadly fall into two categories: mesh-based and keypoint-based. Mesh-based methods estimate pose by aligning the input image with rendered views of a CAD/mesh model [12]. Keypoint-based methods detect 2D keypoints and recover the 6-DoF base pose by solving a Perspective-n-Point (PnP) problem with known 3D keypoints [5], or by learning a direct mapping from images to pose [13,16,22,6]. Most existing approaches target manipulators or drones, benefiting from fixed-base setups or rigid-body assumptions that simplify pose estimation.

In contrast, as shown in Fig. 1(a), a quadruped has a 6-DoF base pose and time-varying leg joints, so a single 6-DoF pose is insufficient. We estimate an 18-DoF state (6-DoF base + four 3-DoF legs), requiring 12 keypoints per frame (Fig. 1(a)– ○1 ). Second, left–right and front–rear symmetry introduces ambiguity in keypoint correspondence, and self-occlusion during gait reduces the visual cues (Fig. 1(a)–○2 ). In addition, no public benchmark exists for camera-to-quadruped pose estimation, making fair comparison difficult. Much of the prior work focuses on the robot’s onboard sensing, while external-camera 3D pose estimation remains less explored.

To address this challenge, we propose QuadPose, which estimates the 18-DoF full-body pose of a quadruped robot from monocular video sequences (Fig. 1(b)). Our key idea is to improve 2D keypoint quality, as accurate keypoints are critical for reliable 3D pose recovery. Specifically, a keypoint-focused token selector mitigates leg appearance ambiguity caused by symmetric limbs. In addition, skeleton-aware masked attention exploits kinematic connectivity to improve robustness under self-occlusion during gait (Fig. 1(a)–○2 ). Given stable 2D keypoints, QuadPose first predicts the 12-DoF joint angles and estimates the 6-DoF base pose, reconstructs 3D keypoints via forward kinematics, and then corrects residual errors using a refinement head. Additionally, we introduce a new dataset QuadSyn to enable reproducible evaluation.

We evaluate QuadPose on the proposed QuadSyn benchmark and demonstrate consistent improvements over existing baselines. Experimental results show that our keypoint-focused modeling improves 3D pose accuracy under symmetry and self-occlusion, while the refinement stage further reduces reconstruction errors. QuadPose achieves 95.54% PCK-AUC and 66.98% ADD-AUC while running at about 33 FPS. In summary, our main contributions are as follows:

• Full-Body Pose Estimation for Quadruped Robots. To our knowledge, we present the first camera-to-quadruped full-body pose estimation framework that recovers an 18-DoF state, including the 6-DoF base pose and 12 joints.

• Robust 2D-to-3D Pose Recovery. We propose selector tokens and skeletonaware masked attention to mitigate 2D keypoint ambiguity, while a refinement head further reduces 3D reconstruction errors.

• New Quadruped Pose Dataset. To our knowledge, we build and release the first public camera-to-quadruped pose benchmark dataset.

## 2 Related Works

Camera-to-Robot Pose Estimation. Pose estimation aims to recover an object’s position and orientation from visual observations and has been widely studied for humans and robots [13,1,18]. In 3D human pose estimation, body size varies across people, and monocular RGB sufers from scale ambiguity, so many methods predict root-relative or scale-normalized 3D pose [18,21]. In contrast, robots have fixed dimensions and a known kinematic structure, making camerato-robot pose estimation closer to recovering the state of a specific articulated robot in the camera frame.

Camera-to-robot pose estimation recovers a robot’s position and orientation in the camera frame from monocular RGB images [13]. For rigid platforms (e.g., drones and ground vehicles), the goal is to estimate a 6-DoF base transform, whereas articulated robots additionally require joint configurations. For example, a quadruped requires estimating the 6-DoF base pose and the joint angles of its 12 leg joints. Prior work can be grouped into mesh-based and keypoint-based methods, depending on whether they rely on 3D mesh models or keypoints as geometric priors (Table 1).

Table 1: Overview of camera-to-robot pose estimation methods. Existing methods are categorized into mesh-based and keypoint-based approaches, and we summarize their target robot types, input modalities, and outputs.
<table><tr><td>Category Method</td><td></td><td>Robot type</td><td>Input</td><td>Output*</td></tr><tr><td rowspan="2">Mesh</td><td>RoboPose (CVPR&#x27;21) [12]</td><td>Manipulator</td><td>Image</td><td>6DoF + q(7)</td></tr><tr><td>DronePose (ECCVW&#x27;20) [1]</td><td>Drone</td><td>Image</td><td>6DoF</td></tr><tr><td rowspan="5">Keypoints</td><td>DroneKey++ (ICRA&#x27;26)</td><td>[9] Drone</td><td>Image seq.</td><td>2D kpt + 6DoF</td></tr><tr><td>RoboPEPP (CVPR&#x27;25) [6]</td><td>Manipulator</td><td>Image</td><td> $\mathrm { 2 D / 3 D ~ k p i + 6 D o F + } q ( 7 )$ </td></tr><tr><td>DroneKey (IROS&#x27;25) [8]</td><td>Drone</td><td>Image seq.</td><td> $\mathrm { 2 D \bar { \ k p t } + 6 D o F }$ </td></tr><tr><td>SGTAPose (CVPR&#x27;23) [22]</td><td>Manipulator</td><td>Image seq.</td><td> $\mathrm { 2 D / 3 D \bar { \ k p t } + 6 D o F }$ </td></tr><tr><td>DREAM (ICRA&#x27;20) [13] QuadPose (Ours)</td><td>Manipulator</td><td>Image Quadruped Image seq. 2D/3D kpt + 6DoF + q(12)</td><td>2D/3D kpt + 6DoF</td></tr></table>

\* 2D/3D kpt: 2D/3D keypoint coordinates, 6DoF: R(3) + t(3) base pose, q(J): joint angles.

• Mesh-based methods: Mesh-based approaches estimate pose by aligning the observed image with renderings of a robot CAD (3D mesh) model. RoboPose [12] iteratively compares rendered views with the input image to optimize the camerato-robot pose and joint angles, while DronePose [1] leverages a 3D mesh prior for 6-DoF drone pose estimation. These methods leverage strong geometric constraints but require iterative optimization at test time, resulting in slow inference.

• Keypoint-based methods: Keypoint-based approaches detect 2D keypoints on robot joints, infer 3D joint locations using a kinematic model, and estimate pose from 2D–3D correspondences. DREAM [13] learns 2D keypoints from synthetic data with domain randomization [23] and estimates pose via PnP [5]. RoboPEPP [6] leverages pretraining to encode robot structure and improve robustness to occlusion. Sequence-based methods also exploit temporal cues: DroneKey [8] and DroneKey++ [9] target drones, while SGTAPose [22] focuses on manipulators. However, methods that estimate joint angles typically target fixed-base manipulators, whereas drones are treated as rigid bodies with no joints to estimate. To our knowledge, no prior work jointly estimates base motion and joint articulation.

Datasets for Camera-to-Robot Pose Estimation. Existing datasets and evaluation protocols have mainly targeted manipulators and drones. Manipulators benefit from fixed-base setups that simplify annotation [13], while drones can be treated as rigid bodies with straightforward 6-DoF labels [1,8,9]. These datasets rely on synthetic data with background compositing for scalable ground-truth generation [13,1,8,9]. In contrast, to our knowledge, no public dataset or evaluation protocol exists for external-camera full-body pose estimation of quadruped robots, whose articulated structure and frequent self-occlusion make annotation substantially more challenging. We address this gap in Sec. 4.

## 3 Proposed Methods

QuadPose consists of two main components: a 2D keypoint detector and a 3D pose estimator, as illustrated in Fig. 2. Given a cropped robot bounding box, keypoint features are first extracted using a transformer-based [26] encoder. These features are refined by a 2D keypoint head that incorporates cross-attention with selector tokens and a skeleton-aware masked attention module, enabling robust 2D keypoint detection. The 3D pose estimator then predicts joint angles and the 6-DoF base pose from the detected 2D keypoints, reconstructs 3D keypoints via forward kinematics, and corrects residual errors through a refinement head.

Given an RGB image sequence {I<sub>i</sub>} and camera intrinsics K, the goal is to estimate the full-body 3D state of a quadruped robot in the camera frame at each time step i. The robot is modeled as an articulated system with a rigid base and four multi-DoF legs comprising 12 joints, as shown in Fig. $\mathrm { 1 ( a ) - \bar { ( 1 ) } }$ . The state <sup>Y</sup>comprises 18 degrees of freedom: the 6-DoF base pose ${ \bf B } _ { i } = ( { \bf R } _ { i } \in \mathbb { R } ^ { 3 \times 3 } , ~ { \bf t } _ { i } \in \mathbb { R } ^ { 3 } )$ and the 12-DoF leg joint angles $\mathbf { q } _ { i } \in \mathbb { R } ^ { 1 2 }$ . We assume that the URDF-based forward kinematics model [3] and per-frame robot 2D bounding boxes are given.

![](images/04b25b2cc3c5bf1cc4db2f67554d1c95cc12a3a429d98af0f02220f33552961c.jpg)  
Fig. 2: The overall framework of QuadPose. (a) 2D keypoint detector with selector tokens and skeleton-aware masked attention. (b) 3D pose estimator that predicts joint angles and base pose, with FK reconstruction and refinement. (c) Loss functions.

## 3.1 2D Keypoint Detector

The 2D keypoint detector estimates 12 keypoints corresponding to the quadruped robot’s joints from an $H \times W$ input RGB image frame $\mathbf { \bar { I } } _ { i } \in \mathbb { R } ^ { \breve { H } \times W \times 3 }$ . It consists of a Transformer-based encoder [26] and a keypoint head as shown in Fig. 2(a). The encoder extracts global contextual features, while the keypoint head predicts 2D keypoint coordinates. To obtain a fixed-size input for the network, the robot bounding box is cropped and resized to a fixed resolution.

Encoder. Global contextual information improves robustness to occlusion and background variations and enables efective feature sharing across the four structurally similar legs. To capture such global spatial context over the entire image, we employ a ResNet backbone and a Transformer-based encoder. We first extract a feature map from the cropped robot image using a ResNet backbone [7]. The resulting feature map is then linearly projected into a set of N visual tokens of dimension d, yielding $\mathbf { F } \in \mathbb { R } ^ { N \times d }$ . Positional encodings $\mathbf { P E } \in \mathbb { R } ^ { N \times d }$ are added to the token features, which are then fed into the Transformer encoder:

$$
\mathbf { F } ^ { \prime } = \operatorname { E n c o d e r } ( \mathbf { F } + \mathbf { P E } ) .\tag{1}
$$

Here, $\mathbf { F ^ { \prime } }$ represents context-aware visual token features that encode global spatial relationships across the image.

![](images/55596660d42e185f55a4fab40e43f6da5dcbe0fc9eea58549ff0121e9e4b9bd1.jpg)  
Fig. 3: Details of the 2D keypoint head. It takes encoded visual tokens $\mathbf { F ^ { \prime } }$ and predicts 2D keypoints via (a) selector-token cross-attention, (b) skeleton-aware masked self-attention, and (c) SimCC-based [14] 1D heatmap regression.

![](images/8683eba00d351a9aee33a590d31b85efd49dab1f6532be54903afc2103d4869b.jpg)  
Fig. 4: Details of the 3D pose estimator. ○1 Given rays $\mathcal { R } _ { i }$ and predicted joint angles $\hat { \mathbf { q } } _ { i }$ over $T$ frames, it estimates the $\textcircled{2}$ 6-DoF base pose $\hat { \mathbf { B } } _ { i }$ via temporal convolution and weighted pooling. $\textcircled{3}$ FK yields 3D keypoints $\hat { \mathbf { y } } _ { i } ^ { 3 D }$ from $\hat { \mathbf { B } } _ { i }$ and $\hat { \mathbf { q } } _ { i }$

2D Keypoint Head. The 2D keypoint head localizes each of the 12 robot joints from the encoded visual features, as shown in Fig. 3.

• Selector tokens: Although the four legs of a quadruped robot share a similar kinematic structure, each joint occupies a distinct spatial location and plays a diferent functional role. Relying solely on shared global features $\mathbf { F ^ { \prime } }$ may therefore lead to ambiguity when localizing individual keypoints. To address this issue, we introduce a set of learnable selector tokens that explicitly guide the network to attend to joint-specific regions.

Specifically, we define 12 selector tokens ${ \bf S } = \{ { \bf s } _ { 1 } , \ldots , { \bf s } _ { 1 2 } \} \in \mathbb { R } ^ { 1 2 \times d }$ , each corresponding to a distinct joint of the quadruped robot. These selector tokens interact with the encoded visual tokens $\mathbf { F ^ { \prime } }$ via a cross-attention mechanism $\left( \mathrm { F i g . \ 3 ( a ) } \right)$ ), where S serves as the queries and the encoded feature tokens $\mathbf { F ^ { \prime } }$ act as the keys and values as follows:

$$
{ \bf Z } = \mathrm { C r o s s A t t n } ( Q = { \bf S } , { \cal K } = { \bf F } ^ { \prime } , { \cal V } = { \bf F } ^ { \prime } ) ,\tag{2}
$$

The output $\mathbf { Z } \in \mathbb { R } ^ { 1 2 \times d }$ aggregates joint-specific information, enabling each selector token to focus on its corresponding keypoint while leveraging global context.

• Skeleton-aware masked attention: While the proposed selector tokens encourage joint-specific attention, challenges remain due to the highly similar appearance of the legs and frequent self-occlusion, as illustrated in Fig. 1(a)–○2 . To better handle these, we leverage the quadruped robot’s kinematic topology as a structural prior. This topology is represented by a skeleton adjacency matrix $\mathbf { A } \in \{ 0 , 1 \} ^ { 1 2 \times 1 2 }$ over the 12 keypoints, corresponding to the shoulder–knee–foot kinematic chain of each leg. We use A as an attention mask and apply masked self-attention to $\mathbf { Z } \mathbf { : }$

$$
\begin{array} { r } { \mathbf { Z } ^ { \prime } = \mathrm { M a s k e d S e l f A t t n } ( \mathbf { Z } , \mathbf { \phi } _ { \mathrm { m a s k } } = \mathbf { A } ) \in \mathbb { R } ^ { 1 2 \times d } . } \end{array}\tag{3}
$$

Here, information exchange is restricted to directly adjacent joints along the kinematic chain $( \mathrm { e . g . }$ , front-left shoulder ↔ front-left knee ↔ front-left foot), as illustrated in Fig. 3(b). Non-adjacent joints (e.g., front-left knee and frontright foot) do not directly exchange information. This skeleton-aware masked self-attention enables topology-consistent information propagation, improving robustness to leg ambiguity and joint self-occlusion.

• 2D keypoints prediction: Many existing methods [24,17] perform 2D keypoint prediction via direct coordinate regression; however, this approach is sensitive to noise and does not model prediction uncertainty. Instead of direct regression, we predict independent 1D distributions along the x and y axes to jointly estimate keypoint coordinates and confidence, as shown in $\mathrm { F i g . \ 3 ( c ) }$ . To this end, we adopt the SimCC formulation [14] to implement this design. Specifically, for each d-dimensional keypoint token $\mathbf { z } _ { k } ^ { \prime } \in \bar { \mathbb { R } } ^ { d }$ (the k-th token in $\mathbf { Z ^ { \prime } } )$ , we predict independent 1D probability distributions along the x- and y-axes as follows:

$$
p _ { k } ^ { x } = \mathrm { S o f t m a x } \left( \mathrm { M L P } _ { x } ( \mathbf { z } _ { k } ^ { \prime } ) \right) \in \mathbb { R } ^ { W } , \quad p _ { k } ^ { y } = \mathrm { S o f t m a x } \left( \mathrm { M L P } _ { y } ( \mathbf { z } _ { k } ^ { \prime } ) \right) \in \mathbb { R } ^ { H } ,\tag{4}
$$

where $\mathrm { M L P } _ { x }$ and $\mathrm { M L P } _ { y }$ denote multi-layer perceptrons used to predict the 1D distributions along the x and y axes, respectively. We then compute the $( x , y ) \ 2 \mathrm { D }$ coordinates as the expectation of each 1D distribution: $\begin{array} { r } { \hat { x } _ { k } = \sum _ { u = 1 } ^ { W } u \cdot p _ { k } ^ { x } ( u ) } \end{array}$ , and $\hat { y } _ { k } =$ $\textstyle \sum _ { v = 1 } ^ { H } v \cdot p _ { k } ^ { y } ( v )$ . Finally, the detector outputs the 2D keypoints $\hat { \mathbf { y } } ^ { 2 D } = \{ ( \hat { x } _ { k } , \hat { y } _ { k } ) \} _ { k = 1 } ^ { 1 2 } .$

## 3.2 3D Pose Estimator

The 3D pose estimator infers the quadruped’s 3D pose in the camera coordinate frame from the detected 2D keypoints $\hat { \mathbf { y } } ^ { 2 D }$ and the camera intrinsics matrix K. The estimator consists of four components: $\textcircled{1}$ JointNet, $\textcircled{2}$ Base Pose Head, ○3 3D Keypoint Head, and $\textcircled{4}$ Refinement module, as illustrated in Fig. 2(b).

JointNet. A lightweight MLP (JointNet) predicts the 12-DoF leg joint angles $\hat { \mathbf { q } } _ { i } \in \mathbb { R } ^ { 1 2 }$ from the detected 2D keypoints $\hat { \mathbf { y } } _ { i } ^ { 2 D }$ . This module is pretrained independently and kept frozen during end-to-end training of the full pipeline. This JointNet design prevents early-stage errors from propagating to downstream modules and simplifies the learning problem. With JointNet kept frozen, the forward kinematics (FK) chain becomes a deterministic function of the base pose given the fixed joint angles $\hat { \mathbf { q } } _ { i }$ . Specifically, FK computes the 3D joint positions by chaining rigid transformations along the robot’s kinematic chain (e.g., shoulder→knee→foot) given the joint angles and base pose.

Base Pose Head. The base pose head estimates the 6-DoF base pose $\hat { \bf B } _ { i } =$ $( \hat { \bf R } _ { i } , \hat { \bf t } _ { i } )$ from the detected 2D keypoints $\hat { \mathbf { y } } _ { i } ^ { 2 D } = \{ ( \hat { x } _ { k } , \hat { y } _ { k } ) \} _ { k = 1 } ^ { 1 2 }$ and the predicted joint angles $\hat { \mathbf { q } } _ { i }$ . The overall structure of the base pose head is shown in Fig. 4. To convert the detected keypoints to camera coordinates, we first remap $\hat { \mathbf { y } } _ { i } ^ { 2 D }$ to the original image coordinates and normalize them using the inverse intrinsic matrix:

$$
\mathbf { r } _ { k } = \mathbf { K } ^ { - 1 } [ \hat { x } _ { k } , \hat { y } _ { k } , 1 ] ^ { \top } \in \mathbb { R } ^ { 3 } ,\tag{5}
$$

where $\mathbf { r } _ { k }$ denotes the 3D ray from the camera center toward the k-th keypoint. At frame i, we denote the set of ray vectors as $\mathcal { R } _ { i } = [ \mathbf { r } _ { 1 } ; \ldots ; \mathbf { r } _ { 1 2 } ] \in \mathbb { R } ^ { 1 2 \times 3 }$ . We then concatenate $\mathcal { R } _ { i }$ with the predicted joint angles $\hat { \mathbf { q } } _ { i } \in \mathbb { R } ^ { 1 2 }$ to form the per-frame input $[ \mathcal { R } _ { i } , \hat { \mathbf { q } } _ { i } ] \in \mathbb { R } ^ { 1 2 \times 4 }$ to the base pose head. A single frame provides only a 2D projection and thus cannot determine the robot’s absolute distance (scale) from the camera. In contrast, temporal cues across frames, such as consistent scale trends, provide additional evidence about depth changes. We therefore incorporate $T$ consecutive frames to help disambiguate this depth-scale.

However, not all frames are equally informative. When the robot moves, changes in its apparent scale and viewpoint provide stronger depth and translation cues, whereas near-stationary frames tend to be less informative for resolving 3D ambiguity. To account for this, we employ learnable weighted pooling for temporal aggregation. At each frame $i , \mathcal { R } _ { i }$ and $\hat { \mathbf { q } } _ { i }$ are concatenated and independently passed through a shared MLP, producing $T$ per-frame features $\{ { \bf f } _ { i } \} _ { i = 1 } ^ { \bar { T } }$ . These features are fed into 1D temporal convolutions to capture how features change across neighboring frames [21], yielding $\mathbf { g } _ { i }$ . A residual connection yields $\hat { \mathbf { f } } _ { i } = \mathbf { g } _ { i } + \mathbf { f } _ { i }$ preserving per-frame cues that may be smoothed out by the convolution. To prioritize informative frames, a Score MLP predicts a scalar score for each $\hat { \mathbf { f } } _ { i \cdot }$ normalized via softmax to obtain per-frame weights. Weighted pooling produces a fused feature, which two MLP heads decode into the base translation $\hat { \mathbf { t } } _ { i } \in \mathbb { R } ^ { 3 }$ and a 6D rotation representation [27], converted to a $3 \times 3$ rotation matrix $\hat { \mathbf { R } } _ { i }$

3D Keypoint Head. Given the predicted base pose $\hat { \bf B } _ { i } = ( \hat { \bf R } _ { i } , \hat { \bf t } _ { i } )$ and joint angles $\hat { \mathbf { q } } _ { i }$ , we compute the 3D keypoints $\hat { \mathbf { y } } _ { i } ^ { 3 D } \in \mathbb { R } ^ { 1 2 \times 3 }$ via URDF-based forward kinematics (FK) [3]. FK applies rigid transformations sequentially along each kinematic chain: starting from the base frame, it computes the shoulder, knee, and foot positions for each leg using the URDF-defined kinematic parameters and the joint angles. We refer to the FK output as the initial 3D keypoints $\hat { \mathbf { y } } _ { i } ^ { 3 D }$ Refinement Module. While FK enforces kinematic consistency, errors from upstream modules can propagate to the initial 3D keypoints. To correct these, we concatenate the coarse 3D keypoints $\hat { \mathbf { y } } _ { i } ^ { 3 D }$ , the 2D keypoints $\hat { \mathbf { y } } _ { i } ^ { 2 D }$ , and the predicted joint angles $\hat { \mathbf { q } } _ { i }$ into a single input vector, and pass it through a Refine MLP that predicts a per-keypoint residual ofset $\varDelta \hat { \mathbf { y } } _ { i } ^ { 3 D }$ . The refined 3D keypoints are then obtained by $\hat { \mathbf { y } } _ { i } ^ { \mathrm { r e f i n e d } } = \hat { \mathbf { y } } _ { i } ^ { 3 D } + \varDelta \hat { \mathbf { y } } _ { i } ^ { 3 D }$

## 3.3 Loss Functions

We train the 2D keypoint detector and 3D pose estimator independently. Within the 3D pose estimator, JointNet is pretrained and frozen before training the remaining components.

2D keypoint detector loss. The 2D keypoint detector follows the SimCC [14] formulation and uses KL divergence between the predicted 1D distributions and Gaussian-smoothed GT distributions along the x and y axes as the loss $ { \mathcal { L } } _ { \mathrm { 2 D } }$

• Kinematic loss: $\mathcal { L } _ { \mathrm { j o i n t } }$ is the $\ell _ { 1 }$ loss on the predicted joint angles $\hat { \mathbf { q } } _ { i } .$ , and $ { \mathcal { L } } _ { \mathrm { 3 D } }$ is the $\ell _ { 1 }$ loss on the coarse 3D keypoints $\hat { \mathbf { y } } _ { i } ^ { 3 D }$ , both derived from the FK chain:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { k i n } } = \lambda _ { \mathrm { j o i n t } } \mathcal { L } _ { \mathrm { j o i n t } } + \lambda _ { \mathrm { 3 D } } \mathcal { L } _ { \mathrm { 3 D } } . } \end{array}\tag{6}
$$

• Base pose loss: $\mathcal { L } _ { \mathrm { t r a n s } }$ is the $\ell _ { 1 }$ loss on base translation $\hat { \mathbf { t } } _ { i } .$ , and ${ \mathcal L } _ { \mathrm { r o t } }$ is the geodesic loss [27] on base rotation $\hat { \bf { R } } _ { i } \mathrm { : }$

$$
\mathcal { L } _ { \mathrm { b a s e } } = \lambda _ { \mathrm { t r a n s } } \mathcal { L } _ { \mathrm { t r a n s } } + \lambda _ { \mathrm { r o t } } \mathcal { L } _ { \mathrm { r o t } } .\tag{7}
$$

• Refinement loss: $\mathcal { L } _ { \mathrm { r e f i n e d } }$ is the $\ell _ { 1 }$ loss on the refined 3D keypoints $\hat { \mathbf { y } } _ { i } ^ { \mathrm { r e f i n e d } }$ $\mathcal { L } _ { \mathrm { r e p r o j } }$ is the $\ell _ { 2 }$ reprojection error between the projected refined 3D keypoints and the detected 2D keypoints. $\mathcal { L } _ { \mathrm { r e g } }$ is the $\ell _ { 2 }$ penalty on the residual offset magnitude $\| \boldsymbol { \varDelta } \hat { \mathbf { y } } _ { i } ^ { 3 D } \|$ preventing the refinement module from making unnecessarily large corrections:

$$
\mathcal { L } _ { \mathrm { r e f i n e } } = \lambda _ { \mathrm { r e f i n e d } } \mathcal { L } _ { \mathrm { r e f i n e d } } + \lambda _ { \mathrm { r e p r o j } } \mathcal { L } _ { \mathrm { r e p r o j } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } .\tag{8}
$$

The final 3D pose estimator loss is: ${ \mathcal { L } } _ { \mathrm { 3 D - p o s e } } = { \mathcal { L } } _ { \mathrm { k i n } } + { \mathcal { L } } _ { \mathrm { b a s e } } + { \mathcal { L } } _ { \mathrm { r e f i n e } } .$

## 4 QuadSyn: A New Benchmark for Quadruped Robots

We introduce QuadSyn, to our knowledge, the first benchmark dataset for 18-DoF camera-to-quadruped robot pose estimation. Quadruped robots are increasingly deployed in real-world scenarios such as disaster response and facility inspection, where images are captured under diverse and challenging conditions. However, there is no publicly available dataset that provides real camera images of quadruped robots together with the key labels needed for pose estimation (e.g., 2D/3D joint annotations and camera setup information). Attaching markers alters the robot’s appearance, and synchronizing 3D labels from external cameras during free locomotion remains dificult. To fill this gap, we build QuadSyn. Although QuadSyn is synthetic, we design its data pipeline to make rendered images closely match real camera observations, thereby reducing the domain gap.

![](images/abc016bc61b9d36bd0e674c5ad9714bf2f78953f17dd76bca5be132770b00c3f.jpg)

![](images/ee1314a29f6f07c1264dad6643ee65a5089ddb57c9d45456e0f9fa6cb7011fbc.jpg)

![](images/5cd678c0232e312aadcd65a52653e3b3e17edb279ae7b0878e96f12fa9218b77.jpg)

![](images/0bc05e1b9fdf1d252d5dfaa7ad2251a3b42159d4937ac77b22f11cb996ca12c3.jpg)

(a) 2D annotations. Bbox and 2D keypoints.  
![](images/6264720784f5bfe31f4f9e2ec66cb3f5636eb9987e86fdf59f06ecd0a2f7b1df.jpg)

![](images/fb9f024d146ea52cdc282f68ba214e8a8a557d96cd5d1fb1a635ea29ef7cb846.jpg)

![](images/ebfc63b0c3615c490312d429184865521288ce42bf189d0a87a1d8e39ddec9ba.jpg)

![](images/c7a5eb6befa597a69838e00687ecea01564d6eec10d8c25a59e88b88a5d2a06e.jpg)

(b) 3D annotations. Sequential 3D keypoints, joint angles, and 6-DoF base.  
![](images/73c764a2ebbea4b632bb8b79cef7d761a56b09b0993b020e65cce79b347cd233.jpg)

![](images/32b11a6efb7360b639699f77eda85c58a6fac45be12ef175c405c99dc37d0267.jpg)

![](images/89d13102a605c7e41989e426a6cb2f062b9a1f75b13f1b35078681c427f3dbac.jpg)  
(c) QuadSyn dataset examples. Overlay of leg keypoints (30 sampled frames) on a 360° panoramic background with robot trajectory (cyan).  
Fig. 5: QuadSyn dataset annotations and examples. (a) 2D bounding box and keypoint labels. (b) 3D keypoints, joint angles, and 6-DoF base pose in the camera frame. (c) Scene examples with overlaid leg keypoint trajectories (30 frames) and robot paths.

QuadSyn contains 51,697 synthetic RGB images at a resolution of 1280 × 720, generated using a single quadruped robot (Boston Dynamics Spot) rendered over 52 background scenes. Each scene forms a video sequence of about 1,000 frames recorded at 30 FPS, resulting in 52 sequences in total. We generate QuadSyn using NVIDIA Isaac Sim [19]. To mimic real-world visual contexts, we build 360<sup>◦</sup> panoramic textures from collected real-world images and apply them as environment backgrounds in the simulator. This reduces the synthetic-to-real domain gap; Section 5.4 confirms that our model transfers to real data without domain adaptation. We further simulate walking motions with the physics engine and vary robot trajectories (including turns), gait patterns, and camera–robot relative poses across scenes to cover a wide range of motions and viewpoints. Additional analysis of the domain gap and dataset is provided in the supplementary material.

For each frame, we derive the following annotations from the simulatorprovided robot states (base pose and joint angles) and camera parameters (Fig. 5):

• 2D/3D keypoints for 12 leg joints (shoulder–knee–foot for all 4 legs),

• the base 6-DoF pose in the camera frame (R, t) and 12-DoF joint angles,

• camera intrinsics and extrinsics,

• the 2D bounding box of the robot in each frame.

Figure 5(a) shows the 2D bounding box and 2D keypoint annotations, and Fig. 5(b) visualizes the 3D keypoints along with the robot base pose (6-DoF) and joint angles in the camera and world frames. In particular, we compute 2D keypoints by transforming 3D keypoints into the camera frame and projecting them using the camera projection model, ensuring 2D–3D correspondence. Ex ample scenes and 3D keypoint trajectories are shown in Fig. 5(c), and additional details of the data generation process are included in the supplementary video.

## 5 Experimental Results

## 5.1 Settings

Datasets. We conduct our main quantitative experiments on QuadSyn, a synthetic benchmark dataset. As described in Sec. 4, QuadSyn contains outdoor scenes with diverse backgrounds, motions, and pose variations. It provides annotations for 2D/3D keypoints, the robot base pose (6-DoF), and joint angles. For evaluation in real-world settings, we additionally use Harper [2], which is captured from a fixed external camera, to evaluate cross-domain generalization qualitatively. Harper is an HRI dataset primarily collected from the robot’s viewpoint for human pose analysis. It also provides videos captured from an external camera, where the quadruped robot is visible. However, the external-view videos do not include camera intrinsics or robot annotations. Therefore, we manually annotate 2D keypoints and restrict our evaluation to 2D keypoint accuracy.

Implementation details. All experiments are implemented using PyTorch [20] and trained on an A100 GPU using the Adam optimizer [11] for 100 epochs. The full pipeline runs at ∼33 FPS with 34.7M parameters. The batch size and other hyperparameters are empirically determined to balance accuracy and computational cost. For the pose estimation network, the loss weights are set as $\lambda _ { \mathrm { j o i n t } } = 0 . 1$ $\lambda _ { \mathrm { 3 D } } = 5 0 . 0 , \lambda _ { \mathrm { t r a n s } } = 1 0 . 0 , \lambda _ { \mathrm { r o t } } = 1 . 0 , \lambda _ { \mathrm { r e f i n e d } } = 1 0 0 . 0 , \lambda _ { \mathrm { r e p r o j } } = 0 . 1$ , and $\lambda _ { \mathrm { r e g } } = 0 . 5$ Evaluation metrics. We evaluate our method in three aspects: 2D keypoint localization, 3D base pose estimation, and joint angle prediction. For 2D keypoint localization, we use the percentage of correct keypoints (PCK) [25] with pixel thresholds of 2.5, 5, and 10. We also report the area under the PCK curve (AUC) and the median pixel error. Base pose accuracy is evaluated using the average distance metric (ADD) [13]. ADD computes the mean Euclidean distance between ground-truth 3D keypoints and those transformed by the predicted camera-to-robot pose. Joint configuration accuracy is measured by the mean absolute error (MAE) in degrees, averaged over all samples and 12 joints.

For real-world evaluation on Harper [2], no ground-truth 2D keypoints are available for the external view; therefore, we manually annotate keypoints. To account for potential annotation uncertainty, we adopt OKS-based mAP [15], which provides a soft, tolerance-aware accuracy measure. We additionally measure temporal jitter, defined as frame-to-frame variations in keypoint positions, to assess temporal stability.

## 5.2 Results on QuadSyn Dataset

In Table 2, QuadPose achieves the best PCK@2.5/5/10 and AUC, as well as the lowest median error for 2D keypoint localization. The gap to DREAM [13] is particularly large at PCK@2.5 (24.45 vs. 92.18), indicating that methods without skeleton-aware modeling are highly sensitive to self-occlusion and leg appearance ambiguity. SGTAPose [22] and YOLOv8-pose [10] perform well under relaxed

Table 2: PCK and ADD comparison on the QuadSyn benchmark. We compare with state-of-the-art methods [13,12,10,22,6] and report PCK at $2 . 5 / 5 / 1 0$ px and ADD at 10/15/20 cm, each with AUC and median error (↑: higher is better, ↓: lower is better). Missing entries (–) indicate that the method does not produce the corresponding output.
<table><tr><td rowspan="2">Method</td><td colspan="6">PCK @ (pix)</td><td colspan="5">ADD @ (cm)</td></tr><tr><td>2.5↑</td><td>5↑</td><td>10↑</td><td>AÚC↑</td><td></td><td>Med↓</td><td>10↑</td><td>15↑</td><td>20↑</td><td>AÚC↑</td><td>Med↓</td></tr><tr><td>DREAM (ICRA&#x27;20) [13</td><td>24.45</td><td>75.62</td><td>98.52</td><td>61.13</td><td></td><td>3.720</td><td>19.10</td><td>28.20</td><td>35.70</td><td>8.400</td><td>31.93</td></tr><tr><td>RoboPose(CVPR&#x27;21)</td><td>[i2]</td><td></td><td></td><td></td><td></td><td></td><td>0.120</td><td>0.450</td><td>0.720</td><td>0.004</td><td>732.6</td></tr><tr><td>YOLOv8-pose (&#x27;23)</td><td>[10] 70.44</td><td>92.54</td><td>98.97</td><td></td><td>88.90</td><td>1.683</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SGTAPose (CVPR&#x27;23)</td><td>[22] 86.58</td><td>96.55</td><td>99.15</td><td></td><td>87.04</td><td>0.740</td><td>52.08</td><td>63.93</td><td>72.12</td><td>56.97</td><td>9.270</td></tr><tr><td>RoboPEPP (CVPR&#x27;25)</td><td>[6] 70.53</td><td>86.81</td><td>96.06</td><td></td><td>75.99</td><td>1.410</td><td>23.58</td><td>55.18</td><td>85.87</td><td>51.63</td><td>14.03</td></tr><tr><td>QuadPose (Ours)</td><td>92.18</td><td>97.57</td><td>99.58</td><td></td><td>95.54</td><td>0.407</td><td>69.49</td><td>81.12</td><td>85.94</td><td>66.98</td><td>6.740</td></tr><tr><td>Improvement*</td><td>+5.60</td><td>+1.02</td><td>+0.43</td><td></td><td>+6.64</td><td>-0.33</td><td>+17.4</td><td>+17.2</td><td>+0.07</td><td>+10.0</td><td>-2.53</td></tr></table>

\* Improvement is computed against the best baseline (excluding ours).

![](images/4d01412735241f8dbcada8434b36e596d41fc8e56eb34d3cdbba95ccacfe41ba.jpg)  
Fig. 6: Qualitative comparison of 2D keypoint detection on QuadSyn.

PCK thresholds (5/10 px) but degrade at the stricter 2.5 px threshold, whereas QuadPose maintains high precision by combining keypoint-specific features with skeleton-aware kinematic constraints.

Figure 06QuadPose achieves the highest ADD accuracy across all thresholds, demonstrating robust 3D base pose estimation. DREAM [13] remains weak at strict thresholds (e.g., 19.1% at ADD@10cm), SGTAPose [22] benefits from temporal context (52.08%) but is still afected by limb ambiguity, RoboPose [12] struggles with complex articulations, and RoboPEPP [6] degrades under stricter criteria. Overall, QuadPose maintains stable 3D pose estimation by coupling precise 2D keypoint localization with temporal modeling.

Table 3 reports the joint-angle error for all 12 joints; QuadPose achieves the lowest error on 11 out of 12 joints, with a mean error of 3.65<sup>◦</sup>. RoboPose [12] exhibits errors above 8<sup>◦</sup> across all joints, indicating that mesh-alignment methods struggle to handle articulated motion. RoboPEPP [6] shows a slight advantage at J3 (front-left foot), which may be related to larger motion at the end-effector joint. Overall, QuadPose delivers the best performance, indicating that skeleton-aware masked self-attention helps mitigate error propagation along the kinematic chain.

Figure 6 compares 2D keypoint predictions of QuadPose and the strongest baseline, SGTAPose [22], on QuadSyn. QuadPose localizes keypoints more precisely, especially under self-occlusion and symmetric leg configurations where SGTAPose tends to confuse leg identities. Figure 7 compares 3D pose estimation results of QuadPose and RoboPEPP [6], showing that QuadPose consistently achieves lower MPJPE across diverse viewpoints and gait patterns.

Table 3: Joint-angle error comparison on the QuadSyn benchmark. We compare with RoboPose [12] and RoboPEPP [6] and report per-joint angle error (↓: lower is better).
<table><tr><td rowspan="3">Method</td><td colspan="10">Joint Error</td></tr><tr><td colspan="4">Front-Left</td><td colspan="2">Front-Right</td><td colspan="2">(degree) ↓ Rear-Left</td><td colspan="4">Rear-Right</td></tr><tr><td>J1</td><td>J2</td><td>J3</td><td>J4</td><td>J5</td><td>J6</td><td>J7</td><td>J8</td><td>J9</td><td>J10</td><td>J11</td><td>J12</td></tr><tr><td>RoboPose (CVPR&#x27;21) [12]</td><td>10.6</td><td>10.6</td><td>9.56</td><td>9.45</td><td>19.9</td><td>14.5</td><td>13.1</td><td>19.2</td><td>12.7</td><td>25.0</td><td>13.2</td><td>8.98</td></tr><tr><td>RoboPEPP (CVPR&#x27;25) [6]</td><td>4.11</td><td>4.46</td><td>4.01</td><td>3.85</td><td>3.85</td><td>3.99</td><td>4.14</td><td>3.61</td><td>4.09</td><td>4.40</td><td>3.97</td><td>4.33</td></tr><tr><td>QuadPose (Ours)</td><td>3.45</td><td>4.07</td><td>4.30</td><td>3.42</td><td>3.22</td><td>3.50</td><td>3.54</td><td>3.14</td><td>3.83</td><td>3.75</td><td>3.59</td><td>3.99</td></tr><tr><td>Improvement*</td><td>-0.7</td><td>-0.4</td><td>+0.3</td><td>-0.4</td><td>-0.6</td><td>-0.5</td><td>-0.6</td><td>-0.5</td><td>-0.3</td><td>-0.7</td><td>-0.4</td><td>-0.3</td></tr></table>

\* Improvement is computed against the best baseline (excluding ours).

![](images/9d75af1932b76126aff1368ea7427c2cb5d3a7c500a8e2f090c0c80bfd121665.jpg)  
(a) Scene #08 (800-900 frames)

![](images/ba517aaad82a8d60cacb45a7cf30aafa8b089f9bc5e350fd324599c7b3b65cec.jpg)

![](images/34fdb41f68e9e93dac17be176bafb07d551ef7627c7a161a9626c050c5d2d18c.jpg)  
(c) Scene #18 (300-400 frames)

(b) Scene #16 (800-900 frames)  
![](images/c0e8f4733256817759f0a16fc7f3ee1f26939b61d747e24a6317710bfe1f4fdb.jpg)  
(d) Scene #41 (300-400 frames)

![](images/27fa12a5321f7599a6c94ff515f5466b5e8d009c7eed43a09d308bc4fdeb54d8.jpg)  
bottom: 800–900 frames)  
Fig. 7: Visualization of 3D pose estimation results on QuadSyn. QuadPose (ours) vs. RoboPEPP [6]. Colors indicate error. MPJPE (Mean Per Joint Position Error) in cm.

Table 4: Ablation study on the 2D keypoint detector with different configurations.
<table><tr><td rowspan="2">Settings</td><td colspan="5">PCK @ (pix)</td></tr><tr><td>2.5↑</td><td>5个</td><td>10↑</td><td>AÚC↑</td><td>Med↓</td></tr><tr><td>Encoder</td><td>80.99</td><td>95.03</td><td>99.29</td><td>91.37</td><td>1.200</td></tr><tr><td>+ SimCC</td><td>88.98</td><td>96.82</td><td>99.35</td><td>93.78</td><td>0.736</td></tr><tr><td>+ ST</td><td>91.77</td><td>97.83</td><td>99.68</td><td>95.11</td><td>0.540</td></tr><tr><td>+ SA</td><td>92.18</td><td>97.57</td><td>99.58</td><td>95.54</td><td>0.407</td></tr></table>

Table 5: Ablation study on the 3D pose estimator with diferent configurations.
<table><tr><td rowspan="2">Settings</td><td colspan="5">ADD @ (cm)</td></tr><tr><td>10↑</td><td>15↑</td><td>20↑</td><td>AUC↑</td><td>Med↓</td></tr><tr><td>Base</td><td>45.40</td><td>69.23</td><td>79.87</td><td>54.78</td><td>10.76</td></tr><tr><td>+ Joint Frozen</td><td>53.80</td><td>74.55</td><td>82.35</td><td>59.33</td><td>9.360</td></tr><tr><td>+ Temporal</td><td>67.47</td><td>81.12</td><td>85.94</td><td>65.90</td><td>7.230</td></tr><tr><td>+ Refinement</td><td>69.49 81.12</td><td></td><td>85.94</td><td>66.98</td><td>6.740</td></tr></table>

## 5.3 Ablation Study

To analyze the contribution of each component in the keypoint detector, we conduct an ablation study. Table 4 summarizes the results for four configurations:

• Encoder: A baseline model that predicts 2D keypoints directly from encoder features using a simple MLP head.

• + SimCC: Add a SimCC head that predicts independent 1D distributions for the x and y axes, replacing direct coordinate regression.

• + ST (selector token): Introduce learnable selector tokens that attend to encoder tokens to extract keypoint-specific representations.

• + SA (skeleton-aware masked attention): Apply skeleton-aware masked self-attention that constrains attention to physically connected joints.

SimCC yields the largest improvement over the encoder-only baseline (PCK@2.5: +7.99, median error: −0.46 px), enabling more robust localization than direct regression. The selector tokens (+ST) further improve accuracy (PCK@2.5: 91.77, AUC: 95.11), while skeleton-aware masked self-attention (+SA) improves temporal stability (AUC: 95.54, median error: 0.41 px). The slight drop in PCK@5/10 (97.83→97.57, 99.68→99.58) occurs near saturation, as masked attention shifts borderline predictions toward skeleton-consistent locations.

Table 5 ablates the 3D pose estimator across four configurations:

Table 6: Overall accuracy and temporal stability on real-world Harper [2] test set. We report OKS mAP and temporal jitter (↑: higher is better, ↓: lower is better).

<table><tr><td>Method</td><td>OKS mAP↑ (Avg.)</td><td>Temporal Jitter↓ (Avg.)</td></tr><tr><td>DREAM [13</td><td>14.79</td><td>12.53</td></tr><tr><td rowspan="3">YOLOv8 [10] SGTAPose [22]</td><td>60.77</td><td></td></tr><tr><td></td><td>171.5</td></tr><tr><td>0.000</td><td>334.1</td></tr><tr><td rowspan="2">QuadPose (Ours) 64.13 Improvement*</td><td></td><td>24.32</td></tr><tr><td>+3.36</td><td>+11.8</td></tr></table>

\*Improvement is relative to the best baseline (excluding ours).

![](images/f4dd40ea4f047f70295bf970efa7585e85975387094e9674881d56602f701217.jpg)  
Fig. 8: Real-world visualization of 2D keypoint detection on Harper. Zoomed-in robot views with • manual 2D references and • Quad-Pose predictions.

• Base: Jointly trains the JointNet and Base Pose Head from a single frame, without temporal modeling or refinement.

• + Joint Frozen: Freezes JointNet and trains only the Base Pose Head.

• + Temporal: Adds 1D temporal convolutions over T consecutive frames to leverage inter-frame motion cues.

• + Refinement: Adds an MLP-based refinement module that predicts residual ofsets to correct accumulated upstream errors.

Freezing the joint predictor (+Joint Frozen) is the most critical design choice (ADD@10: 45.40→53.80, AUC: 54.78→59.33), confirming that joint training causes error coupling that destabilizes the FK chain; the supplementary material shows that freezing reduces the mean joint error from 11.55<sup>◦</sup> to 3.65<sup>◦</sup>. Temporal modeling (+Temporal) yields the largest gain (ADD@10: 53.80→67.47, AUC: 59.33→65.90) by resolving monocular depth ambiguity through multi-frame context. The refinement head (+Refinement) adds gains at strict thresholds (ADD@10: 67.47→69.49, AUC: 65.90→66.98) and lowers the median error to 6.74 cm.

## 5.4 Results on Real-world Data: Harper

To evaluate sim-to-real transfer, we test on Harper [2], a real-world dataset captured with a fixed external camera. Unlike QuadSyn, Harper provides neither camera intrinsics nor robot 3D annotations, so we limit evaluation to 2D keypoint accuracy and temporal stability. We therefore manually annotate 2D keypoints as references and visualize them as Gaussian heatmaps to account for pixel-level labeling ambiguity.

Table 6 reports OKS mAP and temporal jitter on the real-world test set. QuadPose outperforms YOLOv8 by +3.36 in OKS mAP (64.13) and significantly reduces temporal jitter from 171.5 to 24.32, showing temporally stable predictions.

Note that low jitter alone does not imply accuracy: DREAM [13] has the lowest jitter (12.53) yet poor mAP (14.79), indicating inaccurate predictions, while SGTA-Pose [22] exhibits near-zero mAP with high jitter, suggesting overfitting to synthetic appearances. QuadPose is the only method that achieves both high accuracy and low jitter.

Figure 8 presents qualitative results. Despite training solely on synthetic data without explicit domain adaptation, predicted keypoints consistently fall within or near the manual reference regions, confirming that the learned representation generalizes across the sim-to-real domain gap. Combined with the known robot kinematics, accurate 2D keypoints directly enable reliable 3D pose recovery. Failure case analysis is provided in the supplementary material.

## 6 Conclusions

We presented QuadPose, a camera-to-quadruped full-body pose estimation framework that recovers the 18-DoF state from external monocular RGB video sequences, and introduced QuadSyn, the first benchmark dataset for this task with full-body annotations. By improving 2D keypoint localization under leg symmetry and self-occlusion and leveraging temporal modeling, QuadPose achieves the best performance on QuadSyn. Our keypoint-focused design (selector tokens and skeleton-aware masked attention) improves robustness under challenging test scenarios, enabling reliable 3D pose recovery. Qualitative results on Harper suggest promising sim-to-real transfer in external-camera monitoring scenarios. Future work includes extending QuadSyn to more diverse motions and robot platforms (e.g., Go1, ANYmal), and collecting real-world 3D ground truth.

## References

1. Albanis, G., Zioulis, N., Dimou, A., Zarpalas, D., Daras, P.: Dronepose: photoreal istic uav-assistant dataset synthesis for 3d pose estimation via a smooth silhouette loss. In: European Conference on Computer Vision Workshop. pp. 663–681. Springer (2020) 3, 4

2. Avogaro, A., Toaiari, A., Cunico, F., Xu, X., Dafas, H., Vinciarelli, A., Li, E., Cristani, M.: Exploring 3d human pose estimation and forecasting from the robot’s perspective: The harper dataset (2024) 10, 11, 14

3. Boston Dynamics: Spot base URDF (armless) (2025), spot SDK supplemental URDF file; accessed 2026-02-04 5, 8

4. Fankhauser, P., Bjelonic, M., Bellicoso, C.D., Miki, T., Hutter, M.: Robust rough terrain locomotion with a quadrupedal robot. In: 2018 IEEE International Confer ence on Robotics and Automation (ICRA). pp. 5761–5768. IEEE (2018) 2

5. Gao, X.S., Hou, X.R., Tang, J., Cheng, H.F.: Complete solution classification for the perspective-three-point problem. IEEE transactions on pattern analysis and machine intelligence 25(8), 930–943 (2003) 2, 4

6. Goswami, R.G., Krishnamurthy, P., LeCun, Y., Khorrami, F.: Robopepp: Visionbased robot pose and joint angle estimation through embedding predictive pretraining. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6930–6939 (2025) 2, 3, 4, 11, 12, 13

7. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016) 5

8. Hwang, S.B., Cho, Y.J.: Dronekey: Drone 3d pose estimation in image sequences using gated key-representation and pose-adaptive learning. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 708–715 (2025) 3, 4

9. Hwang, S.B., Cho, Y.J.: Dronekey++: A size prior-free method and new benchmark for drone 3d pose estimation from sequential images (2026) 3, 4

10. Jocher, G., Qiu, J., Chaurasia, A.: Ultralytics yolo (Jan 2023) 11, 14

11. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014) 11

12. Labbé, Y., Carpentier, J., Aubry, M., Sivic, J.: Single-view robot pose and joint angle estimation via render & compare. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1654–1663 (2021) 2, 3, 4, 11, 12

13. Lee, T.E., Tremblay, J., To, T., Cheng, J., Mosier, T., Kroemer, O., Fox, D., Birchfield, S.: Camera-to-robot pose estimation from a single image. In: 2020 IEEE International Conference on Robotics and Automation (ICRA). pp. 9426–9432. IEEE (2020) 2, 3, 4, 11, 12, 14, 15

14. Li, Y., Yang, S., Liu, P., Zhang, S., Wang, Y., Wang, Z., Yang, W., Xia, S.T.: Simcc: A simple coordinate classification perspective for human pose estimation. In: European Conference on Computer Vision. pp. 89–106. Springer (2022) 6, 7, 8

15. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014) 11

16. Lu, J., Richter, F., Yip, M.C.: Markerless camera-to-robot pose estimation via self-supervised sim-to-real transfer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21296–21306 (2023) 2

17. Mao, W., Ge, Y., Shen, C., Tian, Z., Wang, X., Wang, Z., den Hengel, A.v.: Poseur: Direct human pose regression with transformers. In: European conference on computer vision. pp. 72–88. Springer (2022) 7

18. Martinez, J., Hossain, R., Romero, J., Little, J.J.: A simple yet efective baseline for 3d human pose estimation. In: Proceedings of the IEEE international conference on computer vision. pp. 2640–2649 (2017) 3

19. NVIDIA: NVIDIA Isaac Sim (2026), accessed: 2026-01-31 10

20. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems 32 (2019) 11

21. Pavllo, D., Feichtenhofer, C., Grangier, D., Auli, M.: 3d human pose estimation in video with temporal convolutions and semi-supervised training. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7753–7762 (2019) 3, 8

22. Tian, Y., Zhang, J., Yin, Z., Dong, H.: Robot structure prior guided temporal attention for camera-to-robot pose estimation from image sequence. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8917–8926 (2023) 2, 3, 4, 11, 12, 14, 15

23. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., Abbeel, P.: Domain randomization for transferring deep neural networks from simulation to the real world. In: 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS). pp. 23–30. IEEE (2017) 4

24. Toshev, A., Szegedy, C.: Deeppose: Human pose estimation via deep neural networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1653–1660 (2014) 7

25. Tremblay, J., To, T., Molchanov, A., Tyree, S., Kautz, J., Birchfield, S.: Synthetically trained neural networks for learning human-readable plans from real-world demon strations. In: 2018 IEEE International Conference on Robotics and Automation (ICRA). pp. 5659–5666. IEEE (2018) 11

26. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017) 4, 5

27. Zhou, Y., Barnes, C., Lu, J., Yang, J., Li, H.: On the continuity of rotation representations in neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5745–5753 (2019) 8, 9