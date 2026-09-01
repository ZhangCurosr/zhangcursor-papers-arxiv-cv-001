# DARP: A Calibrated Dual-Arm RGB-D-IR Dataset for Multi-View Robotic Perception

Manish Kansana\*, Mohammed Yusuf Mujawar\*, Sudip Mittal, Shahram Rahimi, Noorbakhsh Amiri Golilarz\* Department of Computer Science, The University of Alabama, Tuscaloosa, AL, USA

Abstract—Robotic perception from a single viewpoint is often limited by self-occlusion and incomplete surface visibility. This paper presents DARP(Dual-Arm Robotic Perception)<sup>1</sup>, a calibrated dual-arm RGB-D-IR dataset for object-centered robotic perception using two independently moving eye-in-hand manipulators positioned on opposite sides of a shared tabletop workspace. Each arm carries an Intel RealSense sensor that continuously records RGB, depth, and stereo infrared data while synchronized robot joint states are logged for pose recovery. Objects are placed without fixed poses or marked locations, and the acquisition procedure performs automatic localization, crossarm confirmation, adaptive viewpoint generation, and continuous multimodal recording. DARP contains ten unique tabletop objects and preserves the original sensor recordings, robot-state logs, object-level metadata, and calibration information required to reconstruct camera trajectories in a shared metric frame. To evaluate the geometric consistency of the acquisition, we implement a deterministic multi-view fusion pipeline that converts calibrated RGB-D observations into complementary partial point clouds and measured surface meshes without using learned or generative completion methods. Evaluation on 224 held-out RGB-D keyframes comprising 1,563,466 three-dimensional query points yields a median point-to-mesh distance of 2.13 mm and an RMSE of 4.04 mm, with 96.56% of points within 10 mm of the measured-surface mesh. DARP is intended as a reusable resource for multi-view reconstruction, collaborative robotic perception, multimodal fusion, active perception, and future learning-based reasoning over partial object observations.

Index Terms—dual-arm robotics, RGB-D dataset, eye-in-hand perception, multi-view fusion, collaborative perception, geometric calibration, point-cloud fusion, active perception

## I. INTRODUCTION

Robotic perception is often limited by partial visibility. A wrist-mounted camera may observe only one side of an object because of self-occlusion, object geometry, or manipulator workspace constraints. This limitation becomes important during manipulation, where information from surfaces outside the current field of view may still be useful for object understanding, grasp planning, and interaction. Multiple robotic viewpoints can reduce this limitation, but their observations must be temporally associated and expressed in a common spatial frame before they can be combined reliably.

Several datasets have supported RGB-D object recognition, multi-view perception, manipulation, and 3D reconstruction. The Washington RGB-D Object Dataset, BigBIRD, and the YCB Object and Model Set provide multi-view observations of physical objects and have been widely used for objectcentric perception research [1]–[3]. Robotics-oriented datasets such as GraspNet and ROBI further connect RGB-D sensing with grasping and multi-view robotic perception [4], [5]. Collaborative perception studies have also shown the benefit of combining observations from multiple robotic agents rather than relying on a single viewpoint [6]–[8]. Our work focuses on a close-range object-centric setting in which two independently moving eye-in-hand manipulators observe the same physical object from complementary sides while robot motion and multimodal sensor data remain geometrically linked.

We present DARP (Dual-Arm Robotic Perception), a calibrated dual-arm robotic perception dataset collected using two Unitree Z1 Pro manipulators positioned on opposite sides of a shared tabletop workspace [9]. Each arm carries a wrist-mounted Intel RealSense camera that continuously records RGB, depth, and stereo infrared observations while both robot’s joint configurations are logged with synchronized timestamps. Objects are placed without a fixed pose, marked location, or turntable. The system automatically localizes the object, confirms its position across both arms, generates object-centered viewpoints, and records complementary observations throughout the survey.

A key feature of DARP is the preservation of the geometric relationship between sensing and robot motion. ChArUcobased calibration is used to establish the required rigid transformations [10]. Camera-to-wrist hand-eye calibration relates each sensor to its corresponding manipulator, while baseto-base calibration places both robot systems in a common world frame. Recorded joint states, joint-zero corrections, and forward kinematics are then used to recover the worldframe pose of each selected camera observation. The dataset therefore retains the information needed to reconstruct metric camera trajectories and combine observations from both arms without requiring the calibration target during normal object collection.

To demonstrate the geometric consistency and practical use of DARP, we implement a deterministic dual-arm fusion pipeline. Calibrated RGB-D observations are converted to metric point clouds, transformed into the shared world frame, filtered using object support and depth consistency, and accumulated across multiple viewpoints. Repeatedly supported measurements are retained and converted into a measured surface mesh using multi-scale ball pivoting. No trained model, pretrained network, or generative method is used, and geometry that is not observed by either arm is not inferred. DARP is intentionally preserved as a task-flexible acquisition resource rather than being tied to a single reconstructed mesh or semantic benchmark. Its calibrated RGB, depth, infrared, and robot-pose information can support visible-surface reconstruction, segmentation, classification, collaborative perception, sensor-fusion studies, and next-best-view planning. The complementary partial observations may also support future learning-based shape completion approaches such as PCN and PoinTr when an appropriate training target or self-supervised objective is available [11], [12].

The main contributions of this work are:

• A calibrated dual-arm perception dataset containing synchronized RGB, metric depth, stereo infrared, and robotstate observations from two independently moving eyein-hand manipulators.

• A shared-world geometric framework combining ChArUco-based hand-eye and base-to-base calibration with recorded joint states, joint-zero corrections, and robot forward kinematics.

• An autonomous object-centered acquisition procedure that supports unconstrained object placement, cross-arm confirmation, adaptive viewpoint generation, and continuous multimodal recording.

• A deterministic measured-surface fusion pipeline with held-out RGB-D evaluation, achieving a median point-tomesh distance of 2.13 mm over more than 1.56 million unused 3D query points.

• A task-flexible dataset design that supports both geometric processing and future learning-based applications, including segmentation, classification, collaborative perception, active perception, and partial-shape completion.

The remainder of this paper is organized as follows. Section II reviews related RGB-D datasets, partial-view reconstruction, collaborative robotic perception, and calibrationbased geometric processing. Section III presents the dual-arm dataset generation and geometric processing framework, including the robotic platform, calibration, autonomous acquisition, dataset organization, and deterministic fusion procedure. Section IV reports dataset characteristics and quantitative and qualitative results. Section V discusses supported applications and compares DARP with existing resources. Section VI summarizes limitations and practical considerations, and Section VII concludes the paper.

## II. RELATED WORK

## A. RGB-D and Multi-View Object Datasets

RGB-D datasets have been widely used for object recognition, manipulation, and 3D perception. The Washington RGB-D Object Dataset provides multi-view RGB-D observations of physical objects and established an early benchmark for object-centric RGB-D recognition [1]. BigBIRD further increased viewpoint coverage for real object instances through dense RGB-D acquisition [2], while the YCB Object and Model Set introduced a standardized collection of physical objects and geometric models for robotic manipulation research [3]. Large-scale resources such as ShapeNet and ScanNet address complementary settings through digital 3D models and reconstructed indoor RGB-D scenes, respectively [13], [14].

More robotics-oriented datasets directly connect visual sensing with manipulation. GraspNet-1Billion provides RGB-D observations and large-scale grasp annotations for general object grasping [4]. ROBI focuses on multi-view sensing of reflective objects in robotic bin-picking scenarios and highlights the difficulty of acquiring reliable geometry from challenging surfaces [5]. These datasets provide valuable resources for robotic perception, but their acquisition goals differ from continuously recording complementary observations from two independently moving eye-in-hand manipulators together with synchronized robot states.

## B. Partial-View Reconstruction and Shape Completion

Partial observations naturally arise when objects are viewed from limited directions or contain self-occluded surfaces. Learning-based methods have therefore explored how incomplete point clouds can be mapped to more complete shape representations. PCN introduced an encoder–decoder architecture for point-cloud completion [11], while TopNet proposed a hierarchical structural decoder [15]. More recent methods such as PoinTr and SnowflakeNet use Transformer-based or progressive point-generation mechanisms to model missing geometry from partial observations [12], [16]. ProxyFormer and AnchorFormer further develop Transformer-based completion using missing-part reasoning and discriminative geometric structures [17], [18].

Generative approaches have also been investigated for pointcloud completion, including diffusion-based methods such as PCDreamer and SuperPC [19], [20]. Recent work has additionally emphasized that performance on synthetic or controlled completion benchmarks does not necessarily translate directly to real sensor observations [21]. These methods motivate future learning applications of the proposed dataset; however, the reconstruction presented in this paper is strictly geometric and does not infer unseen object surfaces.

## C. Collaborative Robotic Perception

Collaborative perception uses observations from multiple agents to reduce the limitations of an individual viewpoint. Reily and Zhang studied joint view and feature selection for collaborative multi-robot perception [6], while Zhou et al. explored graph neural networks for combining perceptual information across robots [7]. Multi-Robot Scene Completion further considered collaborative perception as a task-agnostic representation problem in which information from multiple robots contributes to a more complete scene description [8]. Recent datasets such as CU-Multi also provide synchronized observations for multi-robot collaborative perception research [22].

The setting considered in this work is more object-centered and close-range. Two fixed-base manipulators independently move eye-in-hand cameras around the same tabletop object, producing complementary partial observations while robot kinematics provide the pose of each sensor measurement. This makes the dataset suitable for studying how embodied viewpoints can be combined at the level of object geometry as well as for later feature-level collaborative perception.

## D. Calibration and Geometric Fusion

Reliable fusion of observations from moving robot-mounted cameras requires accurate spatial calibration. The hand-eye calibration problem estimates the rigid transformation between the robot end-effector and the attached camera, with classical solutions proposed by Tsai and Lenz and later by Daniilidis [23], [24]. ChArUco patterns combine coded ArUco markers with chessboard corners and provide feature correspondences that can be used for intrinsic and extrinsic calibration [10]. In the proposed system, ChArUco-based calibration is used to establish the camera-to-wrist and inter-base relationships required to place observations from both arms in a common metric frame.

Multi-view RGB-D reconstruction methods such as Kinect-Fusion and BundleFusion demonstrate how repeated depth observations can be integrated into coherent geometry when reliable camera poses are available [25], [26]. Our geometric pipeline follows the same general principle of transforming repeated measurements into a shared frame, but the camera poses are derived primarily from robot calibration, synchronized joint states, and forward kinematics rather than estimated through visual tracking.

Overall, existing work provides strong resources for RGB-D perception, robotic manipulation, point-cloud completion, collaborative perception, and multi-view reconstruction. The proposed dataset connects these areas through a dual-arm eyein-hand acquisition setting that preserves multimodal observations, robot motion, and calibration information in a shared metric coordinate system.

## III. DUAL-ARM DATASET GENERATION FRAMEWORK

The dataset is created using a calibrated dual-arm robotic acquisition system designed to collect complementary RGB-D-IR observations of tabletop objects. Two fixed-base manipulators observe the same object from opposite sides of a shared workspace, while the corresponding robot joint states and calibration parameters are retained so that each sensor observation can later be placed in a common metric coordinate frame. The complete framework consists of four main components namely, the robotic sensing platform, spatial calibration and coordinate alignment, autonomous object-centered acquisition, and preservation of the resulting multimodal data and robot metadata.

## A. Robotic Platform and Sensors

The acquisition platform consists of two 6-DoF Unitree Z1 Pro manipulators, denoted as arm110 (one on the left) and arm111 (one on the right), mounted on opposite sides of a shared tabletop workspace, as shown in Fig. 2. The two robot bases remain fixed throughout data collection and are oriented approximately toward one another, creating an overlapping manipulation region in which both arms can observe the same physical object from different directions.

![](images/7cebe96aea6c5f7a548c65bf61704b5060c0ba4c8612b95ef5677c4d74131a38.jpg)  
Fig. 1: Intel RealSense RGB-D-IR sensor mounted near the gripper flange of each Unitree Z1 Pro manipulator.

Each manipulator carries an Intel RealSense RGB-D camera rigidly attached near the gripper flange in an eye-in-hand configuration, as shown in Fig. 1. Camera serial 348522076490 is associated with arm110, while camera serial 405622074504 is associated with arm111. This mapping is fixed because every camera observation must later be transformed through the kinematic chain of its corresponding arm. The cameras continuously record color, depth, and stereo infrared information at 640 × 480 resolution and 30 Hz. Color is stored in BGR8 format, depth in Z16 format, and the left and right infrared channels in Y8 format. Accelerometer and gyroscope streams are also retained in the native recordings, although they are not used by the geometric reconstruction presented in this work. Camera intrinsics, inter-stream calibration, and depth scale remain embedded in the RealSense recordings and are read from each device or bag file during processing rather than being hard-coded.

The two manipulators are separated by approximately 1.3986 m. Their relative orientation is close to antiparallel, with an approximately 178.4<sup>◦</sup> rotation about the world z axis. This arrangement allows the two eye-in-hand cameras to observe complementary regions of the object surface while remaining within their respective reachable workspaces.

## B. Calibration and Coordinate Alignment

Because both cameras move with their corresponding manipulators, multi-view observations can only be combined reliably when the relationship between the camera, robot wrist, robot base, and shared world frame is known. We define the world frame as $\mathcal { F } _ { W }$ , the base frame of arm i as $\mathcal { F } _ { B _ { i } }$ the wrist or gripper frame as $\mathcal { F } _ { G _ { i } }$ , and the camera optical frame as $\mathcal { F } _ { C _ { i } }$ . A ChArUco board is used as the common calibration target, as shown in Fig. 3. ChArUco patterns combine coded markers with accurately localized chessboard corners and provide feature correspondences for geometric calibration [10]. Multiple observations of the calibration target are collected from different robot configurations to estimate the rigid camera-to-wrist transformation for each eye-in-hand sensor. The same calibration framework is used to establish the spatial relationship between the two fixed robot bases.

![](images/6c5698723ba785e1959c3ce2d5960ea1223613a03ac0c1be3847219c448f2146.jpg)  
Fig. 2: Dual-arm acquisition platform. Two Unitree Z1 Pro manipulators are mounted on opposite sides of the tabletop workspace.

For each arm, two static transforms are retained, the wristto-camera transform ${ G _ { i } } _ { \mathbf { T } _ { C _ { i } } }$ and the base-to-world transform ${ { \mathbf { \mathit { W } } } _ { \mathbf { T } _ { B _ { i } } } }$ . Robot joint-zero corrections are also applied before forward kinematics to compensate for static joint bias. The world-frame pose of camera i at time t is therefore recovered as

$$
{ } ^ { W } \mathbf { T } _ { C _ { i } } ( t ) = { } ^ { W } \mathbf { T } _ { B _ { i } } \mathbf { \Sigma } ^ { B _ { i } } \mathbf { T } _ { G _ { i } } \left( \mathbf { q } _ { i } ( t ) + \Delta \mathbf { q } _ { i } \right) \mathbf { \Sigma } ^ { G _ { i } } \mathbf { T } _ { C _ { i } } ,\tag{1}
$$

where $\mathbf { q } _ { i } ( t )$ denotes the interpolated joint configuration and $\Delta \mathbf q _ { i }$ contains the static joint-zero correction for arm i.

The robot-state logger operates at approximately 33 Hz, while the cameras record at 30 Hz. For each camera timestamp, the surrounding robot states are linearly interpolated to obtain the corresponding joint configuration. The robot log begins before camera acquisition and ends after recording is complete so that the usable camera interval remains inside the valid robot-pose interval and pose extrapolation is not required.

The final hand-eye calibration reports held-out geometric errors of approximately 3.58 mm and 3.93 mm for the two arms. These values characterize the internal calibration consistency and are later complemented by cross-arm and heldout RGB-D evaluation. The calibration is treated as epochspecific: if a camera is remounted, a robot base is moved, the sensor bracket changes, or joint-zero behavior changes, the corresponding calibration should be repeated.

## C. Autonomous Data Collection

Each object survey begins when the operator places a single object somewhere within the shared reachable workspace and starts the acquisition procedure. The object position and orientation are not manually measured, no marked placement location is required, and no turntable is used. The system therefore estimates the object location during each survey before generating the viewing trajectories.

![](images/eb5f4c2ed51a5f81db2ad2100e613e2acba71d2af5ba9d631883a10887942157.jpg)  
Fig. 3: ChArUco-based calibration setup used to establish the eye-in-hand camera transformations and the geometric relationship between the two fixed robot bases.

The complete autonomous acquisition sequence is summarized in Fig. 4. Following object placement, the system performs object localization, cross-arm confirmation, adaptive viewpoint planning, synchronized multimodal recording, and post-capture verification before the episode is accepted into the dataset.

Object localization begins with a structured pan-based search using the eye-in-hand camera. The searching manipulator directs the camera toward candidate regions of the tabletop until a valid depth-based object estimate is obtained. The esti mated object center and spatial extent are treated as an initial hypothesis rather than being used immediately for trajectory generation. The object location is then independently checked from both manipulators. During this cross-arm confirmation stage, one arm observes the object while the other remains parked. Estimates that fall outside the valid workspace or exhibit unreliable depth support are rejected. When both arms produce consistent estimates, their object-center estimates are combined, while the larger estimated radius and hten are retained to form a conservative object envelope. Conflicting detections are not blindly averaged, the system instead falls back to a valid reference estimate or terminates the automated survey.

Using the confirmed object center and extent, an independent viewing trajectory is generated for each arm. The orbit adapts to object size, with larger objects receiving broader or denser angular coverage. Candidate camera poses are oriented toward the estimated object center and checked for inversekinematics feasibility, workspace limits, table clearance, excessive joint motion, and interference with the parked manipulator. Invalid candidates are removed, and the trajectory can be regenerated if too few feasible views remain. Only one manipulator moves at a time, while the other stays in a parked configuration. Both cameras record continuously during the survey, producing the representative RGB, metric depth, and stereo infrared observations shown in Fig. 5. Robot motion is executed using non-blocking commands so that joint-state logging continues during movement, and a short settling interval is applied after each accepted viewpoint.

![](images/a356b3be6dcfc68f3e3a8989f44f60f01061189b88a6df66183798948b677b4b.jpg)  
Fig. 4: Overview of the dual-arm data collection pipeline.

This collection strategy is designed to produce complementary partial observations rather than identical views from both arms. Each camera therefore records a sequence of changing eye-in-hand viewpoints from its accessible side of the object, providing the geometric diversity later used for cross-arm fusion and evaluation.

## D. Dataset Organization

Fig. 6 shows the physical tabletop objects included in DARP. The released dataset contains ten unique physical objects. Each object survey is preserved in acquisition form and contains the two native RealSense recordings, the synchronized robot-state log, and the object-centered acquisition metadata:

<object>/

cam348522076490.bag

cam405622074504.bag

arm\_poses.csv

manip\_envelope.json

The two .bag files are continuous multi-view recordings, not single images or isolated viewpoints. Each contains the RGB, depth, left infrared, and right infrared streams from one eye-in-hand camera together with the stream profiles, intrinsics, and depth scale required to decode the measurements. The arm\_poses.csv file stores timestamped joint states for both manipulators, including the arm identifier and six recorded joint angles. The manip\_envelope.json file stores the estimated object center, radius, height, nominal camera standoff, and robot configuration information used during collection.

![](images/238c9ba888854a4382d17cc44792347ccbb7280d82cde94001633ce726f39722.jpg)  
Fig. 5: Representative multimodal observations of several dataset objects recorded by arm110 and arm111. Each observation contains RGB, metric depth, left infrared, and right infrared measurements from the corresponding eye-in-hand sensor.

For efficient downstream processing, selected observations can also be exported as compressed keyframes containing aligned RGB, raw depth, stereo infrared, a conservative geometric object-support mask, camera intrinsics, depth scale, and the corresponding 4×4 world-from-camera pose. The raw acquisition files remain the primary dataset so that alternative frame-selection, calibration, segmentation, fusion, or learning pipelines can be applied without repeating the physical experiment.

The dataset is intentionally not tied to one fixed complete object mesh or one semantic task. Instead, it preserves the multimodal sensor observations, robot motion, and calibration information required to regenerate camera trajectories, partial point clouds, fused geometric representations, or learning inputs.

## E. Geometric Fusion and Surface Reconstruction

The geometric pipeline is used to demonstrate that observations from the two eye-in-hand sensors can be combined consistently in the shared world frame. For each selected RGB-D frame, raw depth is converted to metric depth using the devicespecific scale and back-projected using the recorded camera intrinsics. The resulting camera-frame points are transformed into the world frame using Eq. (1).

![](images/55ad8a346ebfaecf56e393f565b6dceef5b993c5fdc7411fa9aa044918251b81.jpg)  
(a) Clock

![](images/788e8ca41aac21d1a966a771cda2e05387c0c83f4000bcfcbd77783f892b2f6d.jpg)  
(b) Weight

![](images/1f389ae90da2f804ef535e5b296795171b30975acb708242068a3e0e507b99b7.jpg)  
(c) Cube

![](images/3a8a1213f57234f9f83f3c7ad268ed7d9e6d67f9813c6f203bfa509135758286.jpg)  
(e) Dumbbell

![](images/8e969503fa0cc9b6dfdc6d9b337ff25c4f7222bb75dd2e2cab97818db7fb1e6c.jpg)  
(f) Game

(d) Cup  
![](images/f02aa6f34bc26ffded986341a9529fffa58a838e9974c18e33e150c3e796e1ca.jpg)  
(g) Ketchup

![](images/71bd940445a2c1b899329ffe2e2640e6c20592965818ad1764dd106748bb82ee.jpg)

![](images/5eae9c9b1d647d016f11c83c462bb739516711a770431e3ff1cbb1798253fe30.jpg)  
(h) Mug

![](images/e93d0f874d83ebb4f218721d59973bf7d9dd7b5cf5c97a521cf422ce079be49f.jpg)  
(i) Red Bull

![](images/7871d2c6a325a9cfcd0ece77b72a540d4ecd9b877f00fab1d35581b6daa0bb92.jpg)  
Fig. 6: Pictures of objects used in the experiment.  
(j) Frame

A conservative object envelope derived during acquisition removes table and background measurements. View-diverse observations from both arms are then accumulated in the shared frame. The final production fusion uses the calibrated robot-camera poses directly, local depth discontinuities and weakly supported voxels are rejected, followed by statistical outlier removal.

A measured surface mesh is generated from the fused object cloud using multi-scale ball pivoting. The procedure does not predict unobserved geometry and therefore preserves missing or occluded regions rather than artificially closing them. A subset of RGB-D keyframes is excluded from fusion and retained for the held-out point-to-mesh evaluation reported in the following section.

## IV. DATASET CHARACTERISTICS AND RESULTS

This section summarizes the collected data and evaluates whether observations acquired from the two independently moving eye-in-hand cameras remain geometrically consistent after calibration and world-frame transformation. The analysis focuses on the properties that are most relevant to reuse of the dataset mainly, complementary viewpoint coverage, cross-arm alignment, and agreement between reconstructed surfaces and RGB-D observations that were not used during fusion.

## A. Dataset Characteristics

The released dataset contains eight unique physical tabletop objects. Each object is observed from both sides of the workspace using the two eye-in-hand RealSense cameras described in Section III. For every object survey, the two cameras record continuous RGB, depth, and stereo infrared streams while the joint configurations of both manipulators are logged simultaneously. The resulting data therefore contain multiple changing viewpoints from each arm rather than a single image pair.

Table I summarizes the physical object envelopes and recording statistics of the released acquisitions. Across the retained recordings, the dataset contains 61,012 robot jointstate rows from the two manipulators and approximately 18.4 min of synchronized dual-arm acquisition. The native recordings contain more than 63,000 depth frames from the two sensors and occupy approximately 137.94 GB. These statistics reflect the continuous multi-view nature of DARP, in which each episode preserves a temporal sequence of changing eye-in-hand observations rather than a small set of isolated viewpoints.

The dataset can consequently be examined at several levels, individual RGB-D observations, single-arm multi-view sequences, complementary partial point clouds, or dual-arm fused representations. The raw recordings remain the primary released data, while the reconstructed point clouds and meshes presented below are derived products used to demonstrate geometric consistency.

## B. Cross-Arm Geometric Consistency

A central requirement of the acquisition framework is that independently recorded observations from arm110 and arm111 can be transformed into the same metric coordinate system without requiring visual re-registration of every frame. For each selected observation, the camera pose is obtained from the calibrated base and hand-eye transforms together with the synchronized robot configuration, as described in Eq. (1).

The two arms naturally produce different partial surface measurements because each sensor approaches the object from a different side. In regions observed by both cameras, however, corresponding measurements should occupy similar locations after transformation into the world frame. We therefore use overlap between the independently reconstructed arm-specific point clouds as an additional check of calibration consistency.

Fig. 7 further illustrates the geometric relationship between the two arm-specific observations. In the shared world frame, the partial point clouds occupy complementary regions of the object surface, while the co-observed portions remain reasonably aligned after calibration and pose recovery.

![](images/8f1c64419fd3c27aca85ce24167c05b4c1284551686a1c90f1b7856afe784f3e.jpg)

![](images/538b42c8c116ea8729f47bf34f062ed306f8e9cc1a577f6cee595df493dc4dbf.jpg)

![](images/681eab651012dd83ab071f6e161b784b8b570256028a9b442dfa88726805523d.jpg)  
Fig. 7: Cross-arm alignment of the arm110 and arm111 partial point clouds in the shared world frame for Ketchup object. The figure shows orthographic projections in the world XY , XZ, and Y Z planes. Red points correspond to arm110 and blue points correspond to arm111.

TABLE I: Release composition and recording statistics for DARP.
<table><tr><td>Object</td><td>Radius [cm]</td><td>Height [cm]</td><td>Standoff [cm]</td><td>Rows/Arm</td><td>Duration [s]</td><td>Rate [Hz]</td><td>Depth110</td><td>Depth111</td><td>Size [GB]</td></tr><tr><td>Red Bull</td><td></td><td></td><td></td><td>1150</td><td>118.6</td><td>9.7</td><td>3412</td><td>3436</td><td>14.85</td></tr><tr><td>Dumbbell</td><td>6.90</td><td>3.66</td><td>36.9</td><td>3297</td><td>110.6</td><td>29.8</td><td>3160</td><td>3194</td><td>13.78</td></tr><tr><td>Controller</td><td>5.78</td><td>2.64</td><td>35.8</td><td>3093</td><td>103.6</td><td>29.9</td><td>2950</td><td>2986</td><td>12.87</td></tr><tr><td>Cube</td><td>4.31</td><td>1.60</td><td>34.3</td><td>2534</td><td>88.4</td><td>28.6</td><td>2522</td><td>2535</td><td>10.96</td></tr><tr><td>Ketchup</td><td>4.41</td><td>14.93</td><td>34.4</td><td>4119</td><td>135.4</td><td>30.4</td><td>3921</td><td>3944</td><td>17.05</td></tr><tr><td>Picture Frame</td><td>11.60</td><td>19.07</td><td>52.2</td><td>2923</td><td>99.4</td><td>29.4</td><td>2843</td><td>2867</td><td>12.37</td></tr><tr><td>Mug</td><td>7.22</td><td>6.30</td><td>37.2</td><td>2954</td><td>99.4</td><td>29.7</td><td>2830</td><td>2865</td><td>12.35</td></tr><tr><td>Paperweight</td><td>4.85</td><td>7.44</td><td>34.8</td><td>3465</td><td>115.4</td><td>30.0</td><td>3315</td><td>3314</td><td>14.40</td></tr><tr><td>Clock</td><td>5.83</td><td>9.95</td><td>35.8</td><td>3820</td><td>126.6</td><td>30.2</td><td>3656</td><td>3677</td><td>15.90</td></tr><tr><td>Cup</td><td>4.11</td><td>10.16</td><td>34.1</td><td>3151</td><td>107.6</td><td>29.3</td><td>3088</td><td>3107</td><td>13.43</td></tr><tr><td>Total / Range</td><td>3.5–11.6</td><td>1.6–19.1</td><td>34.1–52.2</td><td>30,506</td><td>1105.1</td><td></td><td>31,697</td><td>31,925</td><td>137.94</td></tr></table>

The experiments show that the two partial observations align sufficiently well to form coherent combined object surfaces while still preserving complementary geometry. Importantly, the final production fusion does not apply unrestricted ICP or centroid alignment to force the two arms into agreement. The reported fused surfaces are generated primarily from the calibrated camera poses and repeated multi-frame support. As a result, visible agreement between the two arm-specific measurements reflects the quality of the robot-camera calibration and temporal pose association rather than an unconstrained post-hoc registration step. Fig. 8 illustrates the complementary nature of the two robotic viewpoints. The arm-specific point clouds contain different visible regions of the object, while their calibrated fusion combines these measurements into a more complete observed-surface representation in the shared world frame.

## C. Held-Out Surface Evaluation

To evaluate geometric repeatability without comparing the reconstruction against the same observations used to generate it, a subset of RGB-D keyframes is excluded from fusion. These held-out observations are independently back-projected into 3D and transformed into the world frame using their corresponding robot poses. The resulting query points are then compared with the measured surface mesh using point-to-mesh distance.

Across 224 held-out RGB-D keyframes, the evaluation contains 1,563,466 three-dimensional query points. The aggregate results are summarized in Table II. The median point-to-mesh distance is 2.13 mm, while the RMSE is 4.04 mm. The 90thpercentile distance is 6.38 mm. In addition, 83.82% of the held-out points lie within 5 mm of the reconstructed surface and 96.56% lie within 10 mm.

The values in Table II should be interpreted as held-out observation agreement. They do not represent absolute error against a complete object model obtained from an independent high-accuracy scanner. This distinction is important because the reconstructed mesh contains only surfaces supported by the recorded observations.

## D. Qualitative Reconstruction Results

The qualitative results show the complementary role of the two robotic viewpoints. A point cloud generated from only one arm typically contains the object surfaces visible from that side of the workspace, while self-occluded or oppositely facing regions remain missing. The second arm contributes measurements from a substantially different viewing direction, increasing visible-surface coverage when the observations are combined.

TABLE II: Held-Out RGB-D Geometric Evaluation
<table><tr><td>Metric</td><td>Result</td></tr><tr><td>Median distance</td><td>2.13 mm</td></tr><tr><td>RMSE</td><td>4.04 mm</td></tr><tr><td>90th-percentile distance</td><td>6.38 mm</td></tr><tr><td>Points within 10 mm</td><td>96.56%</td></tr></table>

The final fused representation retains repeatedly supported RGB-D measurements and removes isolated observations before surface generation. Multi-scale ball-pivoting produces a mesh over these measured regions without deliberately closing surfaces that were not observed. Together, the heldout evaluation and qualitative reconstructions show that the released calibration, robot-state information, and multimodal measurements are sufficiently consistent to support metric dual-arm fusion.

## V. APPLICATIONS AND COMPARISON WITH EXISTING DATASETS

## A. Supported Applications

The dataset is designed as a reusable robotic perception resource rather than for a single reconstruction task. Because RGB, depth, stereo infrared, robot joint states, and calibration information are preserved together, the same acquisition can be represented at the image, point-cloud, trajectory, or fusedsurface level.

A direct application is multi-view 3D reconstruction. Individual observations from either manipulator provide only partial object geometry, while calibrated observations from both arms can be transformed into the shared world frame to obtain a more complete measured surface. The arm-specific observations also make it possible to study single-arm versus dual-arm perception and quantify the benefit of complementary viewpoints.

The dataset can further support multimodal robotic perception. RGB, metric depth, and stereo infrared measurements may be used independently or jointly for object segmentation, recognition, feature fusion, and sensor-ablation studies. Since the corresponding robot configuration and camera pose are retained, perception models can additionally incorporate viewpoint or pose information rather than treating each image as an isolated observation.

Another relevant direction is active perception and nextbest-view planning. The recorded camera trajectories and partial observations provide examples of how object visibility changes as an eye-in-hand camera moves around the workspace. Future methods could use this information to predict viewpoints that reduce occlusion or provide additional geometric coverage.

![](images/2ca81b3a589d4e227fb821dcd2f1a21a45996c373dad15992c7ea53b0cf524b9.jpg)  
Fig. 8: Complementary geometric observations from the two robotic arms: (a) partial point cloud from arm110, (b) partial point cloud from arm111, and (c) fused dual-arm point cloud in the shared world frame.

The complementary partial observations may also support future learning-based shape completion. Existing methods such as PCN, PoinTr, and related point-cloud completion approaches predict geometry that is absent from the input observation [11], [12]. In contrast, the geometric pipeline in this paper reconstructs only surfaces supported by measured RGB-D observations. Shape completion should therefore be treated as a downstream learning task requiring an appropriate complete-shape target or self-supervised training objective rather than as an output already provided by the dataset.

From a robotics perspective, the shared metric representation can also support collaborative manipulation studies. For example, one manipulator may observe surfaces that are occluded from the other, while both observations remain related through a common world frame. Such data can be useful for future systems in which an embodied robot reasons jointly about viewpoint, object geometry, and manipulation actions.

## B. Comparison with Existing Datasets

Table III summarizes the position of the proposed dataset relative to representative object-centric RGB-D and robotic perception resources discussed in Section II. The comparison is intended to highlight differences in acquisition design rather than to suggest that the datasets address identical research problems.

The Washington RGB-D Object Dataset and BigBIRD provide multi-view RGB-D observations of physical objects for object recognition and 3D perception [1], [2]. The YCB Object and Model Set combines physical manipulation objects with associated visual and geometric models and has become widely used in robotics research [3]. GraspNet-1Billion connects RGB-D observations with large-scale grasp annotations [4], while ROBI focuses on multi-view observations of reflective objects in robotic bin-picking environments [5].

TABLE III: Position of the proposed dataset relative to representative RGB-D and robotic object datasets. The comparison emphasizes acquisition characteristics relevant to this work.
<table><tr><td>Dataset</td><td>Physical Objects</td><td>RGB-D / Multi-View</td><td>Robotics-Oriented</td><td>Dual Eye-in-Hand Views</td><td>Synchronized Robot State</td></tr><tr><td>Washington RGB-D [1]</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>BigBIRD [2]</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>YCB [3]</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>GraspNet-1Billion [4]</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>ROBI [5]</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>DARP (Ours)</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

The proposed dataset differs primarily in the way observations are acquired and geometrically linked. Two independently moving eye-in-hand cameras observe the same object from opposite sides of a shared manipulation workspace, while synchronized robot joint states and calibration parameters are retained with the original sensor recordings. This allows the pose of each selected observation to be reconstructed through robot kinematics and enables direct analysis of complementary arm-specific and dual-arm measurements.

The purpose of this comparison is not to claim that existing datasets lack multi-view or robotic information. Instead, the distinction of the proposed dataset is the combination of two independently moving eye-in-hand sensors, continuous multimodal acquisition, synchronized manipulator states, and a calibration chain that expresses both camera streams in the same metric world frame. This combination supports direct study of complementary robotic viewpoints while preserving the raw acquisition data for alternative processing and learning methods.

## VI. DISCUSSION AND LIMITATIONS

The dataset demonstrates that complementary observations from two independently moving eye-in-hand cameras can be aligned in a shared metric frame using robot calibration, synchronized joint states, and forward kinematics. However, several limitations should be considered.

First, the reconstructed meshes represent only observed surface geometry. Regions that are not visible to either camera, such as undersides or strongly occluded surfaces, may remain incomplete. The current geometric pipeline does not infer or artificially close these missing regions. Second, reconstruction quality depends on the calibration state of the robotic system. Changes in camera mounting, robot-base position, sensor brackets, or joint-zero behavior may reduce cross-arm consistency and require recalibration. Third, the current release contains ten unique tabletop objects collected in a single laboratory setup. Although the objects provide geometric variation, the dataset does not cover the full diversity of materials, shapes, and environmental conditions encountered in broader robotic applications.

Finally, the held-out point-to-mesh evaluation measures agreement with unused RGB-D observations from the same acquisition rather than absolute error against an independently scanned complete-object model. Future extensions may include additional objects, broader sensing conditions, reference 3D models, and learning-based studies such as shape completion and next-best-view selection.

## VII. CONCLUSION

This paper presented a calibrated dual-arm RGB-D-IR dataset for object-centered robotic perception using two independently moving eye-in-hand manipulators. The dataset preserves continuous multimodal sensor recordings, synchronized robot joint states, object-level metadata, and calibration information so that observations from both arms can be reconstructed in a shared metric frame. To demonstrate the geometric consistency of the data, we used a deterministic multi-view fusion pipeline that transforms calibrated RGB-D measurements into complementary partial point clouds and measured surface meshes without using learned completion or generative methods. Evaluation with held-out RGB-D observations produced a median point-to-mesh distance of 2.13 mm, with 96.56% query points within 10 mm of the reconstructed surface. The dataset currently contains ten unique tabletop objects and is intended as a reusable acquisition resource rather than a task-specific benchmark. Future work will extend the object and sensing diversity and investigate downstream applications including multimodal perception, next-best-view planning, collaborative manipulation, and learning-based reasoning over partially observed object geometry.

## ACKNOWLEDGMENT

The authors acknowledge the support and resources provided by the Bioinspired Robotics, AI, Imaging and Neurocognitive Systems (BRAINS) Laboratory at The University of Alabama.

## REFERENCES

[1] K. Lai, L. Bo, X. Ren, and D. Fox, “A large-scale hierarchical multiview rgb-d object dataset,” in 2011 IEEE International Conference on Robotics and Automation (ICRA), 2011.

[2] A. Singh, J. Sha, K. S. Narayan, T. Achim, and P. Abbeel, “BigBIRD: A large-scale 3d database of object instances,” in 2014 IEEE International Conference on Robotics and Automation (ICRA), 2014.

[3] B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, and A. M. Dollar, “The YCB object and model set: Towards common benchmarks for manipulation research,” in 2015 International Conference on Advanced Robotics (ICAR), 2015.

[4] H.-S. Fang, C. Wang, M. Gou, and C. Lu, “GraspNet-1Billion: A largescale benchmark for general object grasping,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[5] J. Yang, Y. Gao, D. Li, and S. L. Waslander, “ROBI: A multi-view dataset for reflective objects in robotic bin-picking,” arXiv preprint arXiv:2105.04112, 2021.

[6] B. Reily and H. Zhang, “Simultaneous view and feature selection for collaborative multi-robot perception,” arXiv preprint arXiv:2012.09328, 2021.

[7] Y. Zhou, J. Xiao, Y. Zhou, and G. Loianno, “Multi-robot collaborative perception with graph neural networks,” IEEE Robotics and Automation Letters, 2022.

[8] Y. Li, J. Zhang, D. Ma, Y. Wang, and C. Feng, “Multi-robot scene completion: Towards task-agnostic collaborative perception,” in Proceedings of the 6th Conference on Robot Learning (CoRL), 2022.

[9] M. Kansana, M. Y. Mujawar, S. Mittal, S. Rahimi, and N. A. Golilarz, “Darp: A calibrated dual-arm rgb-d-ir dataset for multi-view robotic perception,” 2026. [Online]. Available: https://doi.org/10.21227/ rmv3-be47

[10] G. H. An, S. Lee, M.-W. Seo, K. Yun, W.-S. Cheong, and S.-J. Kang, “Charuco board-based omnidirectional camera calibration method,” Electronics, vol. 7, no. 12, p. 421, 2018.

[11] W. Yuan, T. Khot, D. Held, C. Mertz, and M. Hebert, “PCN: Point completion network,” in 2018 International Conference on 3D Vision (3DV), 2018.

[12] X. Yu, Y. Rao, Z. Wang, Z. Liu, J. Lu, and J. Zhou, “PoinTr: Diverse point cloud completion with geometry-aware transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[13] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, J. Xiao, L. Yi, and F. Yu, “ShapeNet: An information-rich 3d model repository,” arXiv preprint arXiv:1512.03012, 2015.

[14] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “ScanNet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[15] L. P. Tchapmi, V. Kosaraju, H. Rezatofighi, I. Reid, and S. Savarese, “TopNet: Structural point cloud decoder,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[16] P. Xiang, X. Wen, Y.-S. Liu, Y.-P. Cao, P. Wan, W. Zheng, and Z. Han, “SnowflakeNet: Point cloud completion by snowflake point deconvolution with skip-transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[17] S. Li, P. Gao, X. Tan, and M. Wei, “ProxyFormer: Proxy alignment assisted point cloud completion with missing part sensitive transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[18] Z. Chen, F. Long, Z. Qiu, T. Yao, W. Zhou, J. Luo, and T. Mei, “AnchorFormer: Point cloud completion from discriminative nodes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[19] G. Wei, Y. Feng, L. Ma, C. Wang, Y. Zhou, and C. Li, “PCDreamer: Point cloud completion through multi-view diffusion priors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[20] Y. Du, Z. Zhao, S. Su, S. Golluri, H. Zheng, R. Yao, and C. Wang, “SuperPC: A single diffusion model for point cloud completion, upsampling, denoising, and colorization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[21] S. Pathak, P. Kumar, D. Baiju, N. Mboga, G. Steenackers, and R. Penne, “Revisiting point cloud completion: Are we ready for the real-world?” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[22] D. Albin, D. McGann, M. Mena, A. Thomas, H. Biggie, X. Sun, S. McGuire, J. P. How, and C. Heckman, “CU-Multi: A dataset for multi-robot collaborative perception,” arXiv preprint arXiv:2509.19463, 2026.

[23] R. Y. Tsai and R. K. Lenz, “A new technique for fully autonomous and efficient 3d robotics hand/eye calibration,” IEEE Transactions on Robotics and Automation, vol. 5, no. 3, 1989.

[24] K. Daniilidis, “Hand-eye calibration using dual quaternions,” The International Journal of Robotics Research, vol. 18, no. 3, pp. 286–298, 1999.

[25] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. Fitzgibbon, “KinectFusion: Real-time dense surface mapping and tracking,” in 2011 10th IEEE International Symposium on Mixed and Augmented Reality (ISMAR), 2011.

[26] A. Dai, M. Nießner, M. Zollhofer, S. Izadi, and C. Theobalt, “Bundle-¨ Fusion: Real-time globally consistent 3d reconstruction using on-the-fly surface re-integration,” ACM Transactions on Graphics, 2017.