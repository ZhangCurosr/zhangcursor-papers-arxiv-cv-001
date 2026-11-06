# FREE-LIVO: Physically Decoupled Yet Tightly Coupled LiDAR-Inertial-Visual Odometry

Reon Tabata<sup>1</sup>, Kenji Koide<sup>2</sup>, Shuji Oishi<sup>2</sup>, Taku Okawara<sup>2</sup>, Mizuho Aoki<sup>2</sup>, Aoki Takanose<sup>2</sup>, Masashi Yokozuka<sup>2</sup>, and Jun Miura<sup>1</sup>

Abstract— LiDAR-Inertial-Visual Odometry (LIVO) integrates complementary LiDAR, camera, and IMU measurements to achieve robust state estimation. However, most existing methods assume a fixed relative pose and accurate time synchronization between sensors. Toward modular and heterogeneous robotic systems, this paper presents FREE-LIVO, a tightly coupled odometry method that requires neither a rigid LiDAR– camera rig nor LiDAR–camera time synchronization. The proposed method uses a pretrained image feature matcher to establish correspondences between a camera image and a dense LiDAR intensity image generated from a single LiDAR scan using a generative model. From the resulting 3D point-to-pixel correspondences, we construct reprojection factors that directly constrain LiDAR and camera poses at different acquisition times. These factors are jointly optimized in a single factor graph together with constraints from point cloud registration, visual feature tracking, and the IMU measurements associated with each sensor. On six sequences from the public M3DGR dataset, FREE-LIVO achieves an average absolute trajectory error (ATE) RMSE of 0.45 m without provided LiDAR–camera extrinsic parameters and exhibits minimal accuracy degradation under time synchronization errors of up to 0.5 s. Furthermore, experiments with the LiDAR and camera mounted on separate robots demonstrate that the method can jointly estimate the trajectories of independently moving sensors.

## I. INTRODUCTION

Robust localization and mapping are important for autonomous systems. LiDARs and cameras are among the most commonly used sensors for this purpose. LiDAR directly measures 3D geometry, but geometric degeneracy can limit localization accuracy. Cameras provide rich appearance information and long-range visual observations, but visual localization is sensitive to illumination changes and low-texture conditions. LiDAR-Inertial-Visual Odometry (LIVO) integrates complementary LiDAR, camera, and IMU measurements to achieve robust state estimation in various environments.

However, most existing LIVO systems assume that the LiDAR, camera, and IMU are rigidly mounted on a common platform with known extrinsic parameters and accurate time synchronization. As robotic systems become increasingly modular and heterogeneous, sensors can instead be mounted on independently moving actuators or mobile sensing units, or even distributed across separate robots. In these configurations, the LiDAR–camera relative pose changes over time and strict synchronization may not be available, invalidating the assumptions of conventional LIVO. Although several methods use cross-modal LiDAR–camera matching for global localization or place recognition [1]–[4], using such matching to continuously connect independently moving sensor trajectories for odometry remains challenging.

![](images/db5ec73ec1aa97dde1918e0771f181f4eb87235506d5607bfc3de43c99212f85.jpg)  
Fig. 1. FREE-LIVO estimation results with a physically decoupled LiDAR and camera. (a) The LiDAR and camera are held in separate hands. (b) A colored point cloud generated using the estimated sensor poses. (c) Jointly estimated trajectories of the independently moving LiDAR and camera.

In this work, we propose FREE-LIVO, a robust, tightly coupled LIVO method that does not require a fixed LiDAR– camera rig or LiDAR–camera time synchronization. To make LiDAR–camera matching easier, we use the generativemodel-based LiDAR intensity image upsampling method [5]. The proposed method introduces a LiDAR–camera reprojection factor that computes the reprojection error between a projected 3D LiDAR point and its corresponding pixel in the camera image. This factor jointly constrains the LiDAR and camera poses directly from point-to-pixel correspondences. While the system allows the LiDAR and camera to be physically decoupled and unsynchronized, the proposed system is based on a tight coupling of LiDAR, camera, and IMU constraints, meaning that the raw observations from these sensors are fused in a single factor graph.

Through experiments, we show that the proposed method can flexibly integrate observations from physically decoupled and unsynchronized LiDAR and camera setups to achieve stable trajectory estimation. On public datasets, we show that the proposed method achieves accuracy comparable to state-of-the-art LIVO methods without using known LiDAR– camera transformations and does not require LiDAR–camera time synchronization. We also demonstrate that the method can estimate the sensor trajectories even when the LiDAR and camera are mounted on separate robots and their relative pose changes continuously.

The main contributions of this work are three-fold:

• We propose a tightly coupled LiDAR-inertial-visual odometry method that does not require a fixed rig or LiDAR–camera time synchronization. This enables operation without accurate LiDAR–camera extrinsic calibration and even with only software-level time synchronization, allowing the LiDAR and camera to be placed and moved independently, including on separate robots.

• We propose a factor graph architecture that uses LiDAR–camera correspondences obtained by matching an upsampled dense intensity image from a single LiDAR scan with an RGB image using a pretrained feature matcher.

• We demonstrate through experiments that the proposed method achieves accuracy comparable to that of other methods with externally provided parameters, suffers minimal degradation even when LiDAR–camera time synchronization is not available, and provides stable trajectory estimation even when the LiDAR and camera are mounted on completely independent robots.

## II. RELATED WORK

## A. LiDAR-Inertial and Visual-Inertial Odometry

Most LiDAR-Inertial Odometry (LIO) and Visual-Inertial Odometry (VIO) methods are either filtering-based or optimization-based. Filtering-based methods sequentially update the state using sensor measurements. Representative examples include FAST-LIO2 [6] for LIO and OpenVINS [7] for VIO. Optimization-based methods jointly estimate multiple states in a sliding window or factor graph. FORM [8] and GLIM [9] use this approach for LIO, while VINS-Mono [10] and Kimera [11] use it for VIO. Both LIO and VIO depend on constraints obtained from their main sensor. LIO can lose accuracy in environments with limited geometric constraints, such as long corridors and tunnels. VIO can become unstable in dark or low-texture environments, under lighting changes, or during motion blur. Either system can fail when its main sensor measurements provide insufficient constraints, and thus the complementary fusion of different sensor modalities has been studied to enhance the robustness of these systems.

## B. LiDAR-Inertial-Visual Odometry

LIVO combines LiDAR, camera, and IMU measurements to improve robustness when LiDAR geometry or image texture is insufficient. LIC-Fusion 2.0 [12] tracks LiDAR plane features in a sliding window and combines them with camera and IMU measurements. LVI-SAM [13] combines LIO and VIO subsystems through a factor graph, while FAST-LIVO2 [14], R3LIVE [15], and LVIO-Fusion [16] fuse LiDAR, camera, and IMU measurements for robust state estimation. Recent methods also use Gaussian Splatting [17], [18] or adjust sensor weights according to the environment [19], [20]. CLIC [21] uses a continuous-time trajectory to combine LiDAR, camera, and IMU measurements with different sampling rates.

Although these LIVO methods achieve robust state estimation across diverse environments and situations, most of them assume accurate calibration of LiDAR–camera extrinsic parameters and time synchronization. Even small errors in these parameters can misalign LiDAR measurements and image pixels, weakening the cross-modal constraints and degrading state estimation accuracy [14], [21]. However, these assumptions do not hold for systems that use only approximate extrinsic calibration and software-level time synchronization. R3LIVE [15] estimates the relative pose and time offset between the LiDAR and camera but assumes that both remain constant. In contrast, our method introduces robust cross-modal LiDAR–camera data association based on generative sensor modality conversion and does not assume a fixed sensor rig or accurate time synchronization.

## C. Cross-Modal LiDAR-Visual Matching

Global Localization and Place Recognition LiDAR point cloud and camera image matching has been studied for global localization and place recognition. These tasks estimate the current location by matching a camera image with a prebuilt LiDAR map. Because camera images and LiDAR point clouds have very different data representations, many methods use deep learning to map both into a common feature space. Cattaneo et al. [1] use separate networks for 2D images and 3D point clouds. They learn a shared embedding in which features from the same place are close to each other. $( L C ) ^ { 2 }$ [2] generates a disparity image from a camera image and converts a LiDAR scan into a range image. It extracts localization descriptors from both images and adds the best match to a pose graph as a loop factor. LIP-Loc [3] trains image and LiDAR encoders using contrastive learning. It retrieves the place in a LiDAR map that corresponds to a camera image. C2L-PR [4] generates a semantic point cloud from a camera image to reduce the modality gap. It then retrieves the corresponding place from a prebuilt LiDAR database.

These methods compare global features from images and point clouds to retrieve corresponding places or detect loop closure candidates. Therefore, they do not perform continuous pose estimation based on point or pixel-level correspondences. In contrast, our method finds fine-grained correspondences between individual LiDAR and camera features and estimates the LiDAR and camera poses through factor graph optimization, enabling accurate odometry estimation.

Image-to-Point Cloud Matching DeepI2P [22], CorrI2P [23], and 2D3D-MATR [24] match LiDAR point clouds with camera images. Using deep learning, these methods can establish correspondences between LiDAR point clouds and camera images. These methods require training on large datasets with ground-truth LiDAR–camera correspondences.

Tabata et al. [5] proposed a method for matching LiDAR point clouds with camera images using LiDAR intensity upsampling. This method establishes correspondences between LiDAR point clouds and camera images using a pretrained image feature matcher. Because it employs self-supervised pretraining on camera image datasets, this method does not require training on large LiDAR datasets.

![](images/b9c1998e78d2f9c8b0ebe1d86f744a5861889d4ab3931879e37d67f9ebd4610f.jpg)  
Fig. 2. Overview of FREE-LIVO. Upsampled LiDAR intensity images are matched with camera images to obtain 2D–3D correspondences. These are integrated with LiDAR, visual, and IMU constraints in a single factor graph to jointly estimate sensor trajectories without a fixed LiDAR–camera rig or time synchronization.

We propose a method that uses the correspondences obtained from Tabata et al. [5] in a tightly coupled LIVO system. This enables sensor fusion based on accurate LiDAR– camera matching without requiring large amounts of costlyto-collect LiDAR training data.

## III. METHODOLOGY

## A. Proposed Pipeline

The pipeline of the proposed FREE-LIVO framework is shown in Fig. 2. The pipeline mainly consists of LiDAR-IMU preprocessing, camera-IMU preprocessing, image-to-pointcloud matching, and factor graph optimization. The LiDAR frame L and the camera frame C do not need to be fixed to the same rigid body. They can move independently, and they do not need to be synchronized.

In LiDAR-IMU preprocessing, the LiDAR scan $\begin{array} { r l } { \mathcal { P } _ { i } } & { { } = } \end{array}$ $\{ ( ^ { L } p _ { i k } , \ell _ { i k } ) \} _ { k = 1 } ^ { N _ { i } }$ is downsampled and deskewed using IMU data, where $\bar { \boldsymbol { \mathbf { \mathit { z } } } } _ { \pmb { \mathbf { \mathit { p } } } _ { i k } } ^ { - \ \mathbf { \bar { \mathbf { \Lambda } } } } \in \mathbb { R } ^ { 3 }$ and $\ell _ { i k } \in \mathbb { R }$ are a 3D point in L and its intensity, respectively. In camera-IMU preprocessing, the motion of image features in the camera image $\mathcal { T } _ { j }$ is estimated using optical flow. The LiDAR and camera preprocessing modules respectively generate LiDAR-to-LiDAR and camera-to-camera correspondences, which are later fused on a factor graph. Image-to-point-cloud matching first projects the sparse LiDAR scan $\mathcal { P } _ { i }$ into a sparse LiDAR intensity image $\mathcal { I } _ { i } ^ { L }$ , then upsamples it to a dense LiDAR intensity image $\hat { \mathcal { T } } _ { i } ^ { L }$ , and finally matches the dense image with the camera image $\mathcal { T } _ { j }$ . The resulting 2D–3D LiDAR–camera correspondences $\mathsf { \bar { M } } _ { i j } = \{ ( ^ { L } { \pmb p } _ { i k } , z _ { j k } ) \} _ { k = 1 } ^ { K _ { i j } }$ , where $z _ { j k } \in \mathbb { R } ^ { 2 }$ is a pixel in $\mathcal { T } _ { j }$ , are incorporated into the same factor graph. The factor graph optimization jointly minimizes errors from LiDAR-to-LiDAR, camera-to-camera, camera-to-LiDAR and IMU constraints to estimate the LiDAR pose $w _ { \pmb { T } _ { L } }$ and camera pose $^ W T _ { C _ { j } }$ in the common world frame W.

## B. Image-to-Point Cloud Matching

The LiDAR intensity image $\mathcal { I } _ { i } ^ { L }$ and the camera image $\mathcal { T } _ { j }$ exhibit different appearance patterns, but they share geometric structures such as edges and corners. These structures allow us to find correspondences using image feature matching [25]. However, direct matching is difficult because the intensity image $\mathcal { I } _ { i } ^ { L }$ from a single LiDAR scan is sparse. Therefore, we first apply a generative-model-based upsampling method [5] to the sparse LiDAR intensity image $\mathcal { T } _ { i } ^ { L }$ shown in Fig. 3(a) and obtain the dense intensity image $\mathcal { \hat { T } } _ { i } ^ { L }$ shown in Fig. 3(b). As shown in Fig. 3(c), we then match the dense intensity image $\hat { \mathcal { T } } _ { i } ^ { L }$ with the camera image $\mathcal { T } _ { j }$ using a pretrained feature matcher. We obtain the 2D–3D correspondences $\mathcal { M } _ { i j }$ by associating the matched pixels with the available LiDAR depth.

![](images/9c785d1ea36ad3598a9ad08181da76bee18f2bfe073f25dd04e08991d0f6c672.jpg)

![](images/c2339b9c41c4fc19be9bcf8ae6b2cae1208a97de8c2673d47b254f5d84f15ba6.jpg)

(a) Sparse LiDAR intensity  
(b) Dense LiDAR intensity  
![](images/41fbd74e59ec14678aabce63201186f1bce2e74e930a88104cf227a83a256c79.jpg)  
(c) Matching result  
Fig. 3. LiDAR–camera matching. (a) Sparse intensity image from a single LiDAR scan. (b) Dense intensity image generated by Conditional Rectified Flow. (c) Feature correspondences between the dense intensity image and the camera image.

LiDAR Intensity Image Upsampling We use Conditional Rectified Flow [26] to generate the dense intensity image $\hat { \mathcal { T } } _ { i } ^ { L }$ from the sparse intensity image $\mathcal { I } _ { i } ^ { L }$ . Conditional Rectified Flow represents the process from Gaussian noise to the dense intensity image $\hat { \mathcal { T } } _ { i } ^ { L }$ as a velocity field generated by a neural network. We project the LiDAR scan $\mathcal { P } _ { i }$ into the sparse intensity image $\mathcal { I } _ { i } ^ { L }$ using virtual camera parameters, and use $\mathcal { I } _ { i } ^ { L }$ as the condition. The neural network $V _ { \theta }$ with parameters θ predicts the velocity from the current image state $X _ { k }$ , the sparse condition image $\mathcal { I } _ { i } ^ { L }$ , and the time $t _ { k }$ . We integrate the predicted velocity over time with the Euler method at each generation step to generate the dense intensity image $\hat { \mathcal { T } } _ { i } ^ { L }$

$$
\begin{array} { r } { \boldsymbol { X } _ { k + 1 } = \boldsymbol { X } _ { k } + \boldsymbol { V } _ { \boldsymbol { \theta } } \big ( \boldsymbol { X } _ { k } , \boldsymbol { { \mathcal { I } } } _ { i } ^ { L } , t _ { k } \big ) \Delta t , \quad t _ { k } = k \Delta t . } \end{array}\tag{1}
$$

![](images/fa192ef1b8909fd60af0f0f231ef6fdff1e46e2c895e95d566687fafea1e53cc.jpg)  
Fig. 4. Sliding-window factor graph of FREE-LIVO. LiDAR–camera reprojection factors connect the LiDAR and camera state sequences. States outside the optimization window are marginalized. For clarity, only factors involving the latest LiDAR or camera state are shown.

Training the network requires pairs of intensity images from one LiDAR scan and dense LiDAR intensity images. Collecting such paired data at scale is challenging due to their limited availability. Therefore, we train the network in two stages: pretraining and fine-tuning. In pretraining, we apply online masking to a large RGB image dataset and train the network to restore the original images. This pretraining gives the network a general ability to reconstruct missing image content from sparse inputs. We then fine-tune the network with a small number of paired sparse and dense LiDAR intensity images. This allows it to learn the features of LiDAR intensity images. As a result, the network can generate dense LiDAR intensity images without overfitting, even with very little real LiDAR data. The pretraining dataset contains 2.4 million images from GSV-Cities [27], Places365 [28], and MIT Indoor67 [29]. The in-house finetuning dataset consists of 80 outdoor and indoor point clouds collected using a Livox AVIA.

Feature Matching We estimate 2D-2D correspondences between the dense intensity image $\hat { \mathcal { T } } _ { i } ^ { L }$ and the camera image $\mathcal { T } _ { j }$ using a pretrained feature matcher. Our method is agnostic to image matching algorithms, so any matching algorithm can be used. In this work, we employ SuperPoint [30] and LightGlue [31] to balance matching accuracy and computational efficiency. For LightGlue, we use the MIN-IMA weights [32], which were fine-tuned for multi-modal matching. We found that this setting performed better at matching RGB camera images with LiDAR intensity images than the other tested matchers, including LightGlue with its standard weights, SuperGlue [33], and XoFTR [34]. We use the resulting 2D-2D correspondences to estimate the 2D– 3D correspondences $\mathcal { M } _ { i j }$ between the LiDAR scan $\mathcal { P } _ { i }$ and the camera image $\mathcal { T } _ { j }$ . The correspondences $\mathcal { M } _ { i j }$ are obtained from the 2D-2D correspondences and the LiDAR depth data. However, only the intensity image $\mathcal { I } _ { i } ^ { L }$ is upsampled, so depth data is not available at every matched pixel. Therefore, we use only matches with depth data within 5 pixels and discard the other matches.

![](images/20f5a84801ab10fa1efb450a1c96466d164ba0eeb5156dd6a89a8d127a602183.jpg)  
Fig. 5. Factor graph during operation. LiDAR–camera reprojection factors connect the LiDAR and camera trajectories. The colored point clouds show the reprojection points used by these factors.

## C. Factor Graph Optimization

1) Factor Graph Structure: Figs. 4 and 5 show the overall factor graph structure and the factor graph during operation, respectively. The factor graph consists of three parts: one for standard LiDAR–IMU odometry, one for visual–IMU odometry, and one connecting the two. The consecutive LiDAR states are connected by point cloud registration factors and IMU preintegration factors [35] from LiDAR– inertial odometry. The consecutive camera states are connected by structure-less visual reprojection factors (smart factors) [36] and IMU preintegration factors from visual– inertial odometry. The LiDAR and camera state sequences are created independently, but they are connected by LiDAR– camera reprojection factors based on the correspondences $\mathcal { M } _ { i j }$ from image-to-point cloud matching. These factors couple the matched poses $w _ { \pmb { T } _ { L } }$ and $^ W T _ { C _ { j } }$ even when they have different timestamps, and one pose can have multiple matches. Therefore, the method does not need to include a fixed LiDAR–camera relative pose or a constant time offset as a state.

2) LiDAR-Inertial Odometry: We use the odometry estimation module of GLIM [9] as the LIO part. For each LiDAR frame, we estimate the pose, velocity, and IMU bias. In preprocessing, we integrate the IMU measurements between two LiDAR frames and compensate for the sensor motion during the LiDAR scan. We add an IMU preintegration factor between consecutive LiDAR states and a random-walk factor between their IMU bias states [35]. We add point cloud registration factors [37] that directly evaluate the VGICP matching cost [38] between the current LiDAR scan $\mathcal { P } _ { i }$ and previous LiDAR scans as a function of their poses. For the current scan $\mathcal { P } _ { i }$ and a previous scan $\mathcal { P } _ { i ^ { \prime } }$ , the error is written as

$$
\begin{array} { r } { { \cal E } _ { i i ^ { \prime } } ^ { \mathrm { p c } } = { \cal E } _ { \mathrm { V G I C P } } \left( \mathcal { P } _ { i ^ { \prime } } , \mathcal { P } _ { i } , ( { ^ { W } } { \mathbf { T } } _ { L _ { i ^ { \prime } } } ) ^ { - 1 W } { \mathbf { T } } _ { L _ { i } } \right) , } \end{array}\tag{2}
$$

![](images/6a70ed1a82601772ba1b4b619ea7524693a47729fcd7bb4491afd2c42ba1384a.jpg)  
Fig. 6. LiDAR–camera reprojection factor. The reprojection error between a projected 3D LiDAR point and its corresponding image feature jointly constrains the LiDAR and camera poses.

where $E _ { \mathrm { V G I C P } }$ denotes the VGICP matching cost for aligning the current scan $\mathcal { P } _ { i }$ to the previous scan $\mathcal { P } _ { i ^ { \prime } }$ . We denote the sums of the point cloud registration errors $E _ { i i ^ { \prime } } ^ { \mathrm { p c } }$ , IMU preintegration errors, and bias random-walk errors in the optimization window by $E _ { \mathrm { p c } } ^ { \mathrm { L I O } } , E _ { \mathrm { i m u } } ^ { \mathrm { L I O } }$ , and $E _ { \mathrm { b i a s } } ^ { \mathrm { L I O } }$ , respectively. The deskewed LiDAR scan $\mathcal { P } _ { i }$ is also used to generate the sparse intensity image $\mathcal { I } _ { i } ^ { L }$

3) Visual-Inertial Odometry: Our VIO module follows the general system design of VINS-Mono [10] and Kimera-VIO [11]. For each camera image $\mathcal { T } _ { j }$ , we estimate the camera pose $^ { W } { \pmb { T } } _ { C _ { i } }$ , velocity, and IMU bias. We track image features in the camera image $\mathcal { T } _ { j }$ using the KLT tracker [39], [40] and check the tracks with a forward–backward check. The camera rotation predicted from the IMU is used as the initial estimate for KLT tracking. This improves feature tracking when the camera rotates greatly. We remove incorrect tracks using RANSAC with the rotation from the IMU. We select keyframes based on the image parallax, elapsed time, and number of tracked features. We add smart visual projection factors [36] from observations of the same features across camera poses $^ W T _ { C _ { j } }$ . We also add IMU preintegration factors between consecutive camera states. Smart visual projection factors constrain multiple camera poses using image feature observations. We also add random-walk factors between consecutive camera bias states. We denote the sums of the visual reprojection, IMU preintegration, and bias randomwalk errors in the optimization window by $E _ { \mathrm { v i s } } ^ { \mathrm { V I O } } , ~ E _ { \mathrm { i m u } } ^ { \mathrm { V I O } }$ and $E _ { \mathrm { b i a s } } ^ { \mathrm { V I O } }$ , respectively.

4) LiDAR–Camera Reprojection Factor: The LiDAR– camera reprojection factor uses the 2D–3D correspondences $\mathcal { M } _ { i j }$ obtained by image-to-point cloud matching. It directly constrains the LiDAR pose ${ ^ W { \pmb T } } _ { L _ { i } }$ and the camera pose $^ { W } { \pmb { T } } _ { C _ { i } }$ without assuming time synchronization or a fixed LiDAR–camera extrinsic transformation. Its inputs are the correspondences $\mathcal { M } _ { i j }$ , the LiDAR pose ${ ^ W { \pmb T } } _ { L _ { i } }$ , the camera pose $\dot { \bar { w } } _ { T _ { C _ { i } } }$ , and the known camera intrinsics. For efficient computation, all correspondences in $\mathcal { M } _ { i j }$ are grouped into one LiDAR–camera reprojection factor.

For a correspondence $( { } ^ { L } p _ { i k } , z _ { j k } ) \ \in \ { \mathcal { M } } _ { i j }$ , the relative transformation from L at scan i to C at image $j$ is $( ^ { W } { \pmb T } _ { C _ { j } } ) ^ { - 1 W } { \pmb T } _ { L _ { i } }$ . We denote the camera projection function using the known camera intrinsics by $\pi ( \cdot )$ . The pixel $z _ { j k }$ in $\mathcal { T } _ { j }$ corresponds to the 3D point ${ } ^ { L } { p } _ { i k }$ in $\mathcal { P } _ { i }$ . The reprojection error is defined as the difference between the projected pixel

and the observed pixel:

$$
{ \pmb r } _ { i j k } = \pi \big ( ( { \pmb { \operatorname { \cal W } } } { \pmb T } _ { C _ { j } } ) ^ { - 1 W } { \pmb T } _ { L _ { i } } { \pmb \operatorname { \sp L } { \pmb \operatorname { p } } } _ { i k } \big ) - z _ { j k } .\tag{3}
$$

Minimizing this error constrains the relative position and orientation of the LiDAR and camera poses. The LiDAR scan $\mathcal { P } _ { i }$ and the camera image $\mathcal { T } _ { j }$ may be acquired at different times. Thus, the factor can connect any frame pair without time synchronization.

For robustness, we apply the Huber loss to each correspondence to reduce the effect of a small number of incorrect matches. We also apply Dynamic Covariance Scaling (DCS) [41] to all correspondences in each LiDAR–camera frame pair. If many correspondences are incorrect, DCS smoothly reduces the weight of the whole factor. The combination of point-level Huber loss and factor-level DCS handles both individual outliers and incorrect frame pairs. We denote the sum of these robust LiDAR–camera reprojection errors in the optimization window by $E _ { \mathrm { L C } }$

5) Joint Factor Graph Optimization: Let $\mathcal { X }$ denote all LiDAR and camera poses, velocities, and IMU biases in the current optimization window. The complete objective is

$$
\begin{array} { r } { \mathcal { X } ^ { * } = \arg \underset { \mathcal { X } } { \mathrm { m i n } } \ E _ { \mathrm { p c } } ^ { \mathrm { L I O } } + E _ { \mathrm { v i s } } ^ { \mathrm { V I O } } + E _ { \mathrm { L C } } + E _ { \mathrm { p r i o r } } } \\ { + \sum _ { s \in \{ \mathrm { L I O } , \mathrm { V I O } \} } \left( E _ { \mathrm { i m u } } ^ { s } + E _ { \mathrm { b i a s } } ^ { s } \right) , } \end{array}\tag{4}
$$

where $E _ { \mathrm { p r i o r } }$ collects initialization priors, marginalization priors, and numerical regularization terms. Thus, all constraints from the LiDAR, camera, and IMU measurements and the LiDAR–camera matches are optimized together in one tightly coupled factor graph. We solve $\operatorname { E q . }$ (4) incrementally using iSAM2 [42] implemented in GTSAM [43]. To bound the computation cost, we use fixed-lag smoothing with a 5-second window and marginalize states that leave the window.

## D. Implementation Details

In all experiments described below, the system runs in real time on a laptop with an AMD Ryzen 9 7940HS CPU, an NVIDIA GeForce RTX 4080 GPU, and 32 GB of memory. Both Conditional Rectified Flow for LiDAR intensity image upsampling and SuperPoint with LightGlue for image feature matching are accelerated using TensorRT with FP16 precision. Matches with a LightGlue confidence score below 0.01 are removed, and a LiDAR–camera reprojection factor is added only when at least 50 valid 2D–3D correspondences remain.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Dataset: We use the M3DGR dataset [44], the FAST-LIVO2 dataset [14], and an in-house dataset for evaluation. The FAST-LIVO2 dataset and in-house dataset do not provide ground truth and are therefore used only for qualitative evaluation. In the M3DGR dataset, Outdoor01, Outdoor04, Dynamic03, and Dynamic04 are outdoor sequences, whereas Dynamic01 and Dynamic02 are indoor sequences.

TABLE I  
EVALUATION RESULTS ON M3DGR.
<table><tr><td>Sequence</td><td>Metric</td><td>VINS-Mono</td><td>GLIM</td><td>LVI-SAM</td><td>R3LIVE</td><td>FAST-LIVO2</td><td>FREE-LIVO</td><td>FREE-LIVO (Loose)</td><td>FREE-LIVO (NN)</td></tr><tr><td>Outdoor01</td><td>ATE [m]</td><td>3.23</td><td>0.21</td><td>4.94</td><td>0.85</td><td>0.15</td><td>0.18</td><td>1.28</td><td>0.20</td></tr><tr><td>Outdoor04</td><td>ATE [m]</td><td>5.04</td><td>0.39</td><td>15.74</td><td>5.10</td><td>0.58</td><td>0.40</td><td>5.13</td><td>0.54</td></tr><tr><td>Dynamic01</td><td>ATE [m]</td><td>0.33</td><td>1.72</td><td>0.50</td><td>4.43</td><td>7018.72†</td><td>0.98</td><td>1.21</td><td>1.19</td></tr><tr><td>Dynamic02</td><td>ATE [m]</td><td>0.34</td><td>1.31</td><td>0.27</td><td>1.21</td><td>1746.95†</td><td>0.56</td><td>0.41</td><td>0.56</td></tr><tr><td>Dynamic03</td><td>ATE [m]</td><td>1.34</td><td>0.26</td><td>4.94</td><td>0.39</td><td>0.14</td><td>0.25</td><td>2.22</td><td>0.33</td></tr><tr><td>Dynamic04</td><td>ATE [m]</td><td>2.53</td><td>0.39</td><td>10.44</td><td>2.02</td><td>0.22</td><td>0.32</td><td>2.37</td><td>0.29</td></tr></table>

All methods were evaluated using Livox AVIA and Intel RealSense D435i measurements with the parameter settings provided by M3DGR.

TABLE II  
ATE [M] ON M3DGR FOR SELECTED DELAY OFFSETS (PAIR-AVERAGED BY SEQUENCE GROUPS).
<table><tr><td rowspan="2">Sequence</td><td colspan="4">LVI-SAM</td><td colspan="4">R3LIVE</td><td colspan="4">FAST-LIVO2</td><td colspan="4">FREE-LIVO</td></tr><tr><td>0.0 s</td><td>0.1 s</td><td>0.3 s</td><td>0.5 s</td><td>0.0 s</td><td>0.1 s</td><td>0.3 s</td><td>0.5 s</td><td>0.0 s</td><td>0.1 s</td><td>0.3 s</td><td>0.5 s</td><td>0.0 s</td><td>0.1 s</td><td>0.3 s</td><td>0.5 s</td></tr><tr><td>Outdoor01+04</td><td>10.34</td><td>10.80</td><td>11.72</td><td>28.18</td><td>2.98</td><td>1.73</td><td>8.72</td><td>2.28</td><td>0.36</td><td>0.39</td><td>0.36</td><td>0.40</td><td>0.29</td><td>0.24</td><td>0.29</td><td>0.29</td></tr><tr><td>Dynamic01+02</td><td>0.38</td><td>1.06</td><td>2.04</td><td>2.53</td><td>2.82</td><td>9.80</td><td>4.92</td><td>1.37</td><td></td><td></td><td></td><td></td><td>0.77</td><td>1.23</td><td>1.39</td><td>1.83</td></tr><tr><td>Dynamic03+04</td><td>7.69</td><td>8.32</td><td>15.47</td><td>25.80</td><td>1.20</td><td>1.34</td><td>1.83</td><td>0.89</td><td>0.18</td><td>0.17</td><td>0.20</td><td>0.28</td><td>0.29</td><td>0.29</td><td>0.29</td><td>0.31</td></tr></table>

2) Baseline Methods: We compare the proposed method with the following LIO, VIO, and LIVO baselines. We use the following baselines: GLIM [9] as the LIO, VINS-Mono [10] as the VIO, and LVI-SAM [13], R3LIVE [15], and FAST-LIVO2 [14] as the LIVO baselines. Although GLIM, VINS-Mono, and LVI-SAM provide loop closure, we disable it and evaluate them only as odometry estimation methods. The LIVO baselines use the LiDAR–camera extrinsic parameters and time synchronization provided by each dataset. The M3DGR dataset uses software-level rather than strict hardware time synchronization, which poses challenges for accurate LiDAR–camera fusion. We evaluate the proposed method without explicit LiDAR–camera extrinsic parameters.

We evaluate trajectory accuracy using the evo toolkit and report the root mean square error (RMSE) of the translational absolute trajectory error (ATE) after SE(3) alignment with the ground-truth trajectory.

## B. Trajectory Estimation Accuracy on M3DGR

We compare the proposed method with VINS-Mono, GLIM, LVI-SAM, R3LIVE, and FAST-LIVO2 using ATE on six M3DGR sequences. Table I shows the errors for all sequences. Although the proposed method does not use explicit LiDAR–camera extrinsic parameters, it remains competitive with the baseline methods. Across the six sequences, the average errors of GLIM and the proposed method are 0.71 m and 0.45 m, respectively. This result shows that adding camera information improves the accuracy of the GLIM odometry module used in the proposed method. Dynamic01 and Dynamic02 are narrow indoor sequences with moving objects. FAST-LIVO2 fails on these sequences, as indicated by the dagger symbols. These failures may be caused by errors in the software-level time synchronization and LiDAR– camera extrinsic calibration provided by M3DGR.

## C. Robustness to LiDAR–Camera Synchronization Errors

To evaluate robustness to inaccurate LiDAR–camera time synchronization, we add delays of 0.1, 0.3, and 0.5 s to the camera data and compare LVI-SAM, R3LIVE, FAST-LIVO2, and the proposed method. Table II shows the ATE for each time offset on M3DGR, and Fig. 7 qualitatively compares colored point clouds generated using the estimated trajectories on the FAST-LIVO2 dataset. For LVI-SAM, the synchronization errors degrade the estimated trajectories and increase the ATE, while the proposed method maintains relatively stable trajectory estimation. For R3LIVE and FAST-LIVO2, the ATE shows no consistent trend with increasing time offsets, making the effects of synchronization errors difficult to assess from Table II alone. Their LIO components may strongly constrain the trajectories on M3DGR, potentially masking the effects of inaccurate camera data fusion. In the colored point clouds from R3LIVE and FAST-LIVO2, the colors are misaligned with object boundaries and become blurred when a time offset is added. In contrast, the proposed method maintains the correspondence between the camera image and LiDAR point cloud and generates a well-aligned colored point cloud even with a 0.3 s offset. These results show that the proposed method not only maintains trajectory accuracy but also correctly integrates LiDAR and camera observations despite synchronization errors.

## D. Ablation Study

To evaluate the effects of tightly coupling LiDAR, camera, and IMU data and upsampling LiDAR intensity images with Conditional Rectified Flow, we compare FREE-LIVO, FREE-LIVO (Loose), and FREE-LIVO (NN). FREE-LIVO (Loose) is a loosely coupled version of the proposed method. It optimizes LIO and VIO in separate factor graphs, connects the resulting poses using LiDAR–camera reprojection factors, and estimates the fused states in a sliding-window factor graph. To validate the contribution of the generativemodel-based LiDAR upsampling, FREE-LIVO (NN) replaces Conditional Rectified Flow-based upsampling with nearest-neighbor interpolation while keeping the other parts unchanged. As shown in Table I, the average ATE over the six sequences is 0.45 m for FREE-LIVO, 2.10 m for FREE-LIVO (Loose), and 0.52 m for FREE-LIVO (NN). FREE-LIVO (Loose) is less accurate than the proposed method on all outdoor sequences, possibly due to accumulated odometry drift in the loosely coupled architecture. This effect may become more pronounced over longer sequences. Improving dense intensity image quality with Conditional Rectified Flow may be more beneficial in challenging scenarios, such as independently moving sensors, although the gains on the M3DGR dataset are modest.

R�LIVE  
![](images/7d4ffc1c39711162c0a1e95e71f1f8321765f99689ff80060007568f85e859ba.jpg)

FAST-LIVO�  
![](images/e5e5d2761b49774433b2f0573ffe6f24268e3f8b6138e9ba9966dcbab04c812a.jpg)

FREE-LIVO  
![](images/5ebb0ad61314a08dba6b5ff78a4a6a08d558a37c81d1e4066b4d7350d198007a.jpg)

Fig. 7. Colored point clouds generated by R3LIVE, FAST-LIVO2, and FREE-LIVO on the FAST-LIVO2 dataset, with and without an added 0.3 s camera delay. FREE-LIVO maintains alignment between image colors and LiDAR geometry under the delay.  
![](images/e8f1eed72977917c3eb06a56334ce7c1b7971308f91a857025a79a9bd233f057.jpg)  
(a)

![](images/5a685db59e1b41827af12acbd98418473dffb2b06a26d84fb817844a7522c1b2.jpg)  
(b)

![](images/08fe0b391387e44287c891fa03be619acc012f8fee1d6973f41c72ae3cdbcd1e.jpg)  
(c)

Fig. 8. Colored point clouds generated by FREE-LIVO with the LiDAR and camera held in separate hands, as in Fig. 1(a).  
![](images/36943abea5a5a003d81d761e00cec4ac2f6a41fb6acac058171308574dc9e7ce.jpg)  
(a)

![](images/83c93a19392b5f0989aeeb3eab549af4da6c07cc7dc7bb62241f687ff0ba2ab0.jpg)  
(b)

![](images/aab3b7171938b20ddc6980e862de05eb2f13526377855ed38e1ea59bcec799f6.jpg)  
(c)  
Fig. 9. FREE-LIVO estimation with two independently moving robots. (a) LiDAR and camera units mounted on separate robots. (b) Online estimation during operation. (c) Final estimated trajectories.

## E. Evaluation with Physically Decoupled Sensors

We evaluate FREE-LIVO using data collected with the LiDAR and camera held in separate hands, as shown in Fig. 1(a). Figure 8 shows the colored point clouds and corresponding estimated sensor trajectories obtained with this handheld setup. In both outdoor and indoor environments, FREE-LIVO jointly estimates the separate LiDAR and camera trajectories, coupled by LiDAR–camera reprojection factors, and produces consistently colored point clouds.

We also conducted an experiment with the LiDAR and camera mounted on separate mobile robots moving independently, as shown in Fig. 9. The LiDAR and camera are mounted on a rover and a legged robot, and FREE-LIVO runs in real time on a laptop carried by the rover. The results show that both trajectories can be estimated despite continuous changes in the relative pose between the sensors and different vibration characteristics of the two robots.

## V. CONCLUSION

This paper presented FREE-LIVO, a tightly coupled LIVO method that does not require a fixed LiDAR–camera rig or time synchronization. The method constructs LiDAR– camera reprojection factors and optimizes them with all other constraints in a single factor graph. Our experiments demonstrated high estimation accuracy and robustness to synchronization errors and also confirmed that the method can operate with sensors mounted on separate robots. Future work will extend the proposed framework to long-term state estimation that fuses LiDAR and camera data collected across multiple sessions and to multi-robot SLAM.

## REFERENCES

[1] D. Cattaneo, M. Vaghi, S. Fontana, A. L. Ballardini, and D. G. Sorrenti, “Global visual localization in lidar-maps through shared 2d-3d embedding space,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 4365–4371.

[2] A. J. Lee, S. Song, H. Lim, W. Lee, and H. Myung, “(LC)<sup>2</sup>: Lidarcamera loop constraints for cross-modal place recognition,” IEEE Robotics and Automation Letters, vol. 8, no. 6, pp. 3589–3596, 2023.

[3] S. S. Puligilla, M. Omama, H. Zaidi, U. S. Parihar, and M. Krishna, “LIP-Loc: Lidar image pretraining for cross-modal localization,” in 2024 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW). IEEE, 2024, pp. 939–948.

[4] H. Xu, H. Liu, S. Huang, and Y. Sun, “C2L-PR: Cross-modal camerato-lidar place recognition via modality alignment and orientation voting,” IEEE Transactions on Intelligent Vehicles, vol. 10, no. 2, pp. 1128–1144, 2024.

[5] R. Tabata, K. Koide, S. Oishi, M. Yokozuka, T. Okawara, A. Takanose, and J. Miura, “Image-to-point cloud registration made easy with rectified flow-based lidar upsampling,” in IEEE International Conference on Intelligent Robots and Systems (IROS2026), 2026.

[6] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, “FAST-LIO2: Fast direct lidar-inertial odometry,” IEEE Transactions on Robotics, vol. 38, no. 4, pp. 2053–2073, 2022.

[7] P. Geneva, K. Eckenhoff, W. Lee, Y. Yang, and G. Huang, “OpenVINS: A research platform for visual-inertial estimation,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 4666–4672.

[8] E. R. Potokar, T. Pool, D. McGann, and M. Kaess, “FORM: Fixedlag odometry with reparative mapping utilizing rotating lidar sensors,” arXiv preprint arXiv:2510.09966, 2025.

[9] K. Koide, M. Yokozuka, S. Oishi, and A. Banno, “GLIM: 3d rangeinertial localization and mapping with gpu-accelerated scan matching factors,” Robotics and Autonomous Systems, vol. 179, p. 104750, 2024.

[10] T. Qin, P. Li, and S. Shen, “VINS-Mono: A robust and versatile monocular visual-inertial state estimator,” IEEE transactions on robotics, vol. 34, no. 4, pp. 1004–1020, 2018.

[11] A. Rosinol, M. Abate, Y. Chang, and L. Carlone, “Kimera: an opensource library for real-time metric-semantic localization and mapping,” in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 1689–1696.

[12] X. Zuo, Y. Yang, P. Geneva, J. Lv, Y. Liu, G. Huang, and M. Pollefeys, “LIC-Fusion 2.0: Lidar-inertial-camera odometry with sliding-window plane-feature tracking,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020, pp. 5112–5119.

[13] T. Shan, B. Englot, C. Ratti, and D. Rus, “LVI-SAM: Tightly-coupled lidar-visual-inertial odometry via smoothing and mapping,” in 2021 IEEE international conference on robotics and automation (ICRA). IEEE, 2021, pp. 5692–5698.

[14] C. Zheng, W. Xu, Z. Zou, T. Hua, C. Yuan, D. He, B. Zhou, Z. Liu, J. Lin, F. Zhu et al., “FAST-LIVO2: Fast, direct lidar–inertial–visual odometry,” IEEE Transactions on Robotics, vol. 41, pp. 326–346, 2024.

[15] J. Lin and F. Zhang, “R3LIVE: A robust, real-time, rgb-colored, lidarinertial-visual tightly-coupled state estimation and mapping package,” in 2022 international conference on robotics and automation (ICRA). IEEE, 2022, pp. 10 672–10 678.

[16] H. Zhang, L. Du, S. Bao, J. Yuan, and S. Ma, “LVIO-Fusion: Tightlycoupled lidar-visual-inertial odometry and mapping in degenerate environments,” IEEE robotics and automation letters, vol. 9, no. 4, pp. 3783–3790, 2024.

[17] H. Zhao, W. Guan, and P. Lu, “LVI-GS: Tightly coupled lidar–visual– inertial slam using 3-d gaussian splatting,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–10, 2025.

[18] S. Hong, C. Zheng, Y. Shen, C. Li, F. Zhang, T. Qin, and S. Shen, “GS-LIVO: Real-time lidar, inertial, and visual multisensor fused odometry with gaussian mapping,” IEEE Transactions on Robotics, vol. 41, pp. 4253–4268, 2025.

[19] V. Hulchuk, J. Bayer, and J. Faigl, “Lidar-visual-inertial tightlycoupled odometry with adaptive learnable fusion weights,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 5641–5647.

[20] T. Wen, Y. Fang, B. Lu, X. Zhang, and C. Tang, “LIVER: A tightly coupled lidar-inertial-visual state estimator with high robustness for underground environments,” IEEE Robotics and Automation Letters, vol. 9, no. 3, pp. 2399–2406, 2024.

[21] J. Lv, X. Lang, J. Xu, M. Wang, Y. Liu, and X. Zuo, “Continuoustime fixed-lag smoothing for lidar-inertial-camera slam,” IEEE/ASME Transactions on Mechatronics, vol. 28, no. 4, pp. 2259–2270, 2023.

[22] J. Li and G. H. Lee, “DeepI2P: Image-to-point cloud registration via deep classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 15 960–15 969.

[23] S. Ren, Y. Zeng, J. Hou, and X. Chen, “CorrI2P: Deep image-to-point cloud registration via dense correspondence,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 3, pp. 1198– 1208, 2022.

[24] M. Li, Z. Qin, Z. Gao, R. Yi, C. Zhu, Y. Guo, and K. Xu, “2D3D-MATR: 2d-3d matching transformer for detection-free registration

between images and point clouds,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 14 128– 14 138.

[25] K. Koide, S. Oishi, M. Yokozuka, and A. Banno, “General, single-shot, target-less, and automatic lidar-camera extrinsic calibration toolbox,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 11 301–11 307.

[26] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=XVjTT1nw5z

[27] A. Ali-Bey, B. Chaib-Draa, and P. Giguere, “GSV-Cities: Toward appropriate supervised visual place recognition,” Neurocomputing, vol. 513, pp. 194–203, 2022.

[28] B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba, “Places: A 10 million image database for scene recognition,” IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 6, pp. 1452– 1464, 2017.

[29] A. Quattoni and A. Torralba, “Recognizing indoor scenes,” in 2009 IEEE conference on computer vision and pattern recognition. IEEE, 2009, pp. 413–420.

[30] D. DeTone, T. Malisiewicz, and A. Rabinovich, “SuperPoint: Selfsupervised interest point detection and description,” in 2018 IEEE/CVF conference on computer vision and pattern recognition workshops (CVPRW). IEEE, 2018, pp. 337–33 712.

[31] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “LightGlue: Local feature matching at light speed,” in 2023 IEEE/CVF international conference on computer vision (ICCV). IEEE, 2023, pp. 17 581– 17 592.

[32] J. Ren, X. Jiang, Z. Li, D. Liang, X. Zhou, and X. Bai, “MINIMA: Modality invariant image matching,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 23 059–23 068.

[33] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “Super-Glue: Learning feature matching with graph neural networks,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 4937–4946.

[34] O. Tuzcuo<sup>¨</sup> glu, A. K˘ oksal, B. Sofu, S. Kalkan, and A. A. Alatan,¨ “XoFTR: Cross-modal feature matching transformer,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 4275–4286.

[35] C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza, “Imu preintegration on manifold for efficient visual-inertial maximum-a-posteriori estimation,” in Robotics: Science and Systems XI, 2015.

[36] L. Carlone, Z. Kira, C. Beall, V. Indelman, and F. Dellaert, “Eliminating conditionally independent sets in factor graphs: A unifying perspective based on smart factors,” in 2014 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2014, pp. 4290–4297.

[37] K. Koide, M. Yokozuka, S. Oishi, and A. Banno, “Globally consistent 3d lidar mapping with gpu-accelerated gicp matching cost factors,” IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 8591–8598, 2021.

[38] ——, “Voxelized GICP for fast and accurate 3d point cloud registration,” in 2021 IEEE international conference on robotics and automation (ICRA). IEEE, 2021, pp. 11 054–11 059.

[39] B. D. Lucas and T. Kanade, “An iterative image registration technique with an application to stereo vision,” in IJCAI’81: 7th international joint conference on Artificial intelligence, vol. 2, 1981, pp. 674–679.

[40] J. Shi et al., “Good features to track,” in 1994 Proceedings of IEEE conference on computer vision and pattern recognition. IEEE, 1994, pp. 593–600.

[41] P. Agarwal, G. D. Tipaldi, L. Spinello, C. Stachniss, and W. Burgard, “Robust map optimization using dynamic covariance scaling,” in 2013 IEEE international conference on robotics and automation. Ieee, 2013, pp. 62–69.

[42] M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. J. Leonard, and F. Dellaert, “iSAM2: Incremental smoothing and mapping using the bayes tree,” The International Journal of Robotics Research, vol. 31, no. 2, pp. 216–235, 2012.

[43] F. Dellaert, “Factor graphs and gtsam: A hands-on introduction,” 2012.

[44] D. Zhang, J. Zhang, Y. Sun, T. Li, H. Yin, H. Xie, and J. Yin, “Towards robust sensor-fusion ground slam: A comprehensive benchmark and a resilient framework,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 8894–8901.