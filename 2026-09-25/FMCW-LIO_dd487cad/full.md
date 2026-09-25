# FMCW-LIO

# A Doppler LiDAR-Inertial Odometry

Mingle Zhao, Jiahao Wang, Tianxiao Gao, Chengzhong Xu, and Hui Kong

Abstract—Conventional LiDAR-inertial odometry (LIO) or simultaneous localization and mapping (SLAM) methods heavily rely on geometric features of environments, as LiDARs primarily provide range measurements instead of motion measurements. From now on, however, the situation changes thanks to the novel Frequency Modulated Continuous Wave (FMCW) Doppler LiDARs. FMCW Doppler LiDARs not only offer the point range with high resolution but also capture the instant point Doppler velocity through the Doppler effect. In the letter, we propose FMCW-LIO, a novel and robust LIO, leveraging intrinsic Doppler measurements from FMCW Doppler LiDARs. To correctly exploit Doppler velocities, a motion compensation method is designed, and a Doppler-aided observation model is applied for on-manifold state estimation. Then, dynamic points can be effectively removed by the Doppler criteria, deriving more consistent geometric observations. FMCW-LIO eventually achieves accurate state estimation and static mapping, even in structure-degenerated environments. Extensive experiments in diverse scenes are performed and FMCW-LIO outperforms other algorithms on both accuracy and robustness.

Index Terms—Sensor Fusion, Localization, SLAM, State Estimation, FMCW Doppler LiDAR.

## RESOURCES

IEEE Xplore Link : FMCW-LIO arXiv Paper Link : P FMCW-LIO Code & Sequence : <sup>§</sup> FMCW-LIO Experiment Video : <sup>Å</sup> FMCW-LIO

## I. INTRODUCTION

L <sup>ASER</sup> <sup>Light</sup> <sup>Detection</sup> <sup>And</sup> <sup>Ranging</sup> <sup>(LiDAR)</sup> <sup>sensors</sup>are indispensable and play a crucial role in robotics and are indispensable and play a crucial role in robotics and automation. In recent years, LiDAR sensors have undergone dramatic innovations with more advanced features. Among them, a remarkable milestone is the technology of Frequency Modulated Continuous Wave (FMCW) Doppler LiDAR that can capture both the range and radial Doppler velocity of points based on the modulation and demodulation of laser waves in frequency domain [1] [2]. Additionally, the FMCW Doppler LiDAR shows promise of miniaturization, thus holding significant potential in robotic applications.

The new dimension of Doppler velocity in the FMCW Doppler LiDAR implicitly conveys the motion information of the LiDAR, which opens up new pathways for robotic perception and state estimation. Consequently, numerous unexplored topics arise in the integration of conventional methods, previously dominated by geometric feature-based methods, with the intrinsic Doppler velocity. Besides, FMCW Doppler LiDARs can output precise range sensing of massive points with significantly high temporal-spatial resolution. On the other hand, FMCW Doppler LiDARs possess the inherent capability to measure the Doppler velocity. More importantly, Doppler measurements can provide a direct observation of the LiDAR motion state (LiDAR velocity), which has a correspondence-free nature. The correspondence-free observation is irrelevant to the locally geometric features (e.g., line or plane features) of the surrounding environment. As a result, FMCW Doppler LiDARs, representing the next generation of laser sensing technology, hold the potential to address various challenging issues in the conventional LiDAR field, including the degeneracy and dynamic problem [2].

![](images/f2dc6e9d6b5452cf817d9b01c88e0de67c883014def2b0bd9ad753c8418006df.jpg)  
Fig. 1. Online mapping comparison of the whole S-shaped tunnel. (a) and (c) show the drifted and inconsistent maps of FAST-LIO2. (b) and (d) show the neat and consistent maps of FMCW-LIO.

In this work, harnessing the advantages of FMCW Doppler LiDAR and Inertial Measurement Unit (IMU), we propose a Doppler LiDAR-inertial odometry and static mapping framework. The contributions can be summarized as follows:

1) A robust Doppler-aided LIO, FMCW-LIO, is proposed for FMCW Doppler LiDAR systems. To our knowledge, FMCW-LIO is the first framework for FMCW Doppler LiDAR-inertial state estimation and static mapping.

2) To precisely utilize Doppler velocities in an accumulated LiDAR scan, a novel motion compensation method for Doppler measurements is designed by fusing with IMU. Notably, beyond motion compensation, the proposed Doppler relationship essentially reveals the interrelation between Doppler velocities of a static point with respect to different LiDAR frames.

3) A Doppler-aided observation, integrated in the onmanifold Iterated Error-State Extended Kalman Filter (IESEKF), and an efficient dynamic point removal method are exploited to obtain consistent geometric observations and static maps.

4) Comprehensive experiments on multiple mobile platforms equipped with an FMCW Doppler LiDAR are conducted to validate and demonstrate the effectiveness, accuracy, and robustness of the proposed framework.

5) The source code of FMCW-LIO, the data sequences, the integrated framework of FMCW-LIO with Free-Init, and the experiment video are made publicly available (see the Resources section for details).

## II. RELATED WORK

In this section, we review the most related works focusing on LiDAR-inertial odometry, FMCW Doppler LiDAR-based algorithms and datasets, and recent works on automotive radarinertial odometry.

## A. LiDAR-Inertial Odometry

To handle dynamic motions of robots, the tightly-coupled LiDAR-inertial fusion has attracted significant research attention. LIO-SAM [3] employs a factor-graph framework, and it can also fuse Global Navigation Satellite System (GNSS) and loop closure factors. LIO-mapping [4] uses the IMU pre-integration in a constrained refinement on estimation and mapping. LINS [5] introduces a framework based on the IESEKF to recursively update the state, but it heavily relies on the ground assumption. In [6], authors apply the IESEKF and a novel Kalman gain formulation for efficient computing. FAST-LIO2 [7] further enhances the performance by using a direct scan-to-map observation model and the incremental k-dimensional tree (ikd-Tree) for the online map. Faster-LIO [8] uses an incremental voxel map (iVox) to achieve higher search speed. Point-LIO [9] uses a point-wise framework and a novel IMU modeling method that allows a high-frequency odometry output. A hierarchical geometrical observer and a higher-order motion correction are applied in [10] which can efficiently improve the estimation accuracy and map quality.

## B. FMCW Doppler LiDAR-based Algorithms and Datasets

Recently, DICP [2] introduces a variant of conventional geometric Iterative Closest Point (ICP) methods by including a Doppler objective function in the registration problem, revealing significant improvements in accuracy even in featuredenied scenes. Guo et al. [11] propose a velocity estimation and clustering framework with Doppler velocities in the CARLA simulation [12]. Gu et al. [13] present a movingobject tracking method using an FMCW Doppler LiDAR. In [14] and [15], authors present a continuous-time LiDAR odometry (LO) and a velocity estimator supporting Doppler measurements. In [16], authors use FMCW Doppler LiDARs to simultaneously estimate the kinematic state and the shape of moving objects. More recently, Jung et al. [17] release a heterogeneous LiDAR dataset (HeLiPR) for place recognition tasks, including an FMCW Doppler LiDAR. However, He-LiPR is not appropriate for online state estimation because the velocity of points is the post-processed absolute velocity using external sensors, instead of the raw Doppler velocity measurement.

![](images/94075cda798c711f35df6e7fc2ef3912cc62e283dbbd8917bc3ff325ed770f50.jpg)  
Fig. 2. System overview of FMCW-LIO.

## C. Automotive Radar-Inertial Odometry

3D or 4D imaging radars can measure the position and Doppler velocity of points similar to FMCW Doppler LiDARs, but the point density is far less than that of FMCW Doppler Li-DARs. In [18] [19], authors propose EKF-based radar-inertial odometry (RIO) systems, with ego-velocity estimates. In [20], authors use the radar intensity in the RIO. Ng et al. [21] propose a radar-inertial odometry which uses a continuoustime framework. Lately, 4D iRIOM [22] shows the impressive localization and mapping accuracy of a RIO, integrated with the scan-to-submap matching and loop closure modules.

## III. SYSTEM OVERVIEW

The system overview is illustrated in Fig. 2. FMCW-LIO takes 6-axis IMU signals as system inputs. LiDAR scans from an FMCW Doppler LiDAR are fed into the motion compensation to correct the distortion of position and Doppler measurements. The undistorted scans are first used in the LiDAR velocity observation model. Adopting the updated velocity, potential dynamic points in the scan can be removed, according to raw Doppler measurements. Then, the remaining static points are sent to the LiDAR geometric observation, resulting in more accurate and consistent associations for the second state update and static mapping. Overall, the Doppler-aided FMCW-LIO can achieve accurate and robust state estimation and output a consistent, neat, and static map.

For notations, the inertial frame is set as the world frame w and the body frame b is the IMU frame. $^ { w } ( \cdot ) , ^ { \ b } ( \cdot )$ , and $^ l ( \cdot )$ respectively represent a 3D vector projected in the world, body, and LiDAR frame. The gravity $\mathbf { \omega } ^ { w } \mathbf { g }$ is constant in the world frame under the steady and flat earth assumption. A 3D vector p from point A to B projected in the frame C is ${ } ^ { C } { \bf p } _ { A B } .$ . Similarly, a 3D linear velocity of the frame B with respect to the frame A expressed in the frame $C { \bf \ i s } ^ { C } { \bf v } _ { A B }$ . We use $\mathbf { R } \in S O ( 3 )$ to represent rotations and a rotation matrix rotating a vector in the frame B to A is $\mathbf { R } _ { A B } . \mathbf { T } _ { A B } \in S E ( 3 )$ physically transforms a vector in the frame B to the frame $A ,$ considering the lever arm. 0 is the $3 \times 1$ zero vector, ${ \bf 0 } _ { n \times m }$ is the $n \times m$ zero matrix, I is the $3 \times 3$ identity matrix, and ${ \mathbf I } _ { n }$ is the $n \times n$ identity matrix. $\widetilde { ( \cdot ) } , \widehat { ( \cdot ) }$ , and $\overline { { ( \cdot ) } }$ are respectively the measurement, the propagated state, and the updated state.

## IV. METHODOLOGY

## A. System State Description

The state $\mathbf { x } \in \mathcal { M }$ evolves on the 24-dimensional manifold $\mathcal { M } _ { : }$ , including the rotation $\mathbf { R } _ { w b }$ and position ${ } ^ { w } { \bf p } _ { w b }$ of body frame with respect to the world frame (the first body frame), velocity ${ w } _ { \mathbf { v } _ { w b } }$ , gyroscope and accelerometer biases ${ } ^ { b } \mathbf { b } _ { g }$ and ${ } ^ { b } \mathbf { b } _ { a } .$ , and LiDAR-IMU extrinsic parameters $\mathbf { R } _ { b l }$ and ${ } ^ { b } \mathbf { p } _ { b l } \colon$

$$
\mathcal { M } \triangleq S O ( 3 ) \times \mathbb { R } ^ { 1 5 } \times S O ( 3 ) \times \mathbb { R } ^ { 3 } , \quad \dim ( \mathcal { M } ) = 2 4 ,
$$

$$
\mathbf { x } \triangleq \left[ \mathbf { R } _ { w b } ^ { \top } \quad ^ { w } \mathbf { p } _ { w b } ^ { \top } \quad ^ { w } \mathbf { v } _ { w b } ^ { \top } \quad ^ { b } \mathbf { b } _ { g } ^ { \top } \quad ^ { b } \mathbf { b } _ { a } ^ { \top } \quad ^ { w } \mathbf { g } ^ { \top } \quad \mathbf { R } _ { b l } ^ { \top } \quad ^ { b } \mathbf { p } _ { b l } ^ { \top } \right] ^ { \top }
$$

To parameterize the error state on the tangent space of the locally homeomorphic manifold, two operators (“boxplus”: ⊞ and “boxminus”: ⊟) are utilized [7] [23]. Hence the errorstate $\delta \mathbf { x } \triangleq \mathbf { x } \ominus \mathbf { x } _ { e s t i } \in \mathbb { R } ^ { 2 4 }$ between the true state x and the nominal estimate ${ \bf x } _ { e s t i } ~ ( { \bf x } _ { e s t i } = \widehat { { \bf x } } ~ \mathrm { o r } ~ { \bf x } _ { e s t i } = \overline { { { \bf x } } } )$ is:

$$
\delta { \bf x } = \left[ \delta \pmb { \theta } _ { w b } ^ { \top } ~ ^ { w } \delta { \bf p } _ { w b } ^ { \top } ~ ^ { w } \delta { \bf v } _ { w b } ^ { \top } ~ ^ { b } \delta { \bf b } _ { g } ^ { \top } ~ ^ { b } \delta { \bf b } _ { a } ^ { \top } ~ ^ { w } \delta { \bf g } ^ { \top } ~ \delta \pmb { \theta } _ { b l } ^ { \top } ~ ^ { b } \delta { \bf p } _ { b l } ^ { \top } \right] ^ { \top } .
$$

## B. State Propagation

1) System Input: The system input is the IMU measurement (angular velocity $^ { b } \widetilde { \omega }$ and specific force $^ b \widetilde { \mathbf { a } } )$ . The input is denoted as u, interrupted by zero-mean Gaussian noises $\mathbf { n } _ { g }$ and $\mathbf { n } _ { a } . \ \mathbf { n } _ { b g }$ and ${ \bf n } _ { b a }$ are Gaussian noises of IMU biases, modeled as random walks. The input u and the noise w are: u ≜ <sup>b</sup>ae<sup>⊤</sup> ∈ R<sup>6</sup>, w ≜ -n<sup>⊤</sup><sub>g</sub> n<sup>⊤</sup><sub>a</sub> n<sup>⊤</sup><sub>bg</sub> n<sup>⊤</sup><sub>ba</sub> ∈ R<sup>12</sup>.

2) Continuous-Time Model: Considering the input signal and the rigid connection between LiDAR-IMU frames, the state dynamics model in continuous time can be derived as:

$$
\begin{array} { r l r } { \dot { \mathbf { R } } _ { w b } = \mathbf { R } _ { w b } \left[ ^ { b } \widetilde { \omega } - ^ { b } \mathbf { b } _ { g } - \mathbf { n } _ { g } \right] _ { \times } , } & { { } ^ { w } \dot { \mathbf { p } } _ { w b } = ^ { w } \mathbf { v } _ { w b } } & { } \\ { ^ { w } \dot { \mathbf { v } } _ { w b } = \mathbf { R } _ { w b } \left( ^ { b } \widetilde { \mathbf { a } } - ^ { b } \mathbf { b } _ { a } - \mathbf { n } _ { a } \right) + ^ { w } \mathbf { g } , ^ { b } \dot { \mathbf { b } } _ { g } = \mathbf { n } _ { b g } , ^ { b } \dot { \mathbf { b } } _ { a } = \mathbf { n } _ { b a } } & { } \\ { ^ { w } \dot { \mathbf { g } } = \mathbf { 0 } , } & { { } \dot { \mathbf { R } } _ { b l } = \mathbf { 0 } _ { 3 \times 3 } , } & { { } ^ { b } \dot { \mathbf { p } } _ { b l } = \mathbf { 0 } , } & { ( 1 ) } \end{array}
$$

where $[ \mathbf { t } ] _ { \times }$ is the skew-symmetric matrix of a 3D vector t.

3) Discrete-Time Model: Using discrete IMU measurements, we can propagate the nominal state from (1) when every IMU measurement (denoting i the index) arrives [7]:

$$
\begin{array} { r } { \mathbf { x } _ { i + 1 } = \mathbf { x } _ { i } \boxplus \left( \mathbf { f } \left( \mathbf { x } _ { i } , \mathbf { u } _ { i } , \mathbf { w } _ { i } \right) \Delta t \right) , } \end{array}\tag{2}
$$

where $\Delta t$ is the sampling period between two adjacent IMU measurements. f is the derivative of the discrete dynamics model [7]. The estimated nominal state $\widehat { \mathbf { x } }$ is propagated using the above function where the noise is set to zero:

$$
\widehat { \mathbf { x } } _ { i + 1 } = \widehat { \mathbf { x } } _ { i } \boxplus \left( \mathbf { f } \left( \widehat { \mathbf { x } } _ { i } , \mathbf { u } _ { i } , \mathbf { 0 } _ { 1 2 \times 1 } \right) \Delta t \right) .\tag{3}
$$

Retaining first-order terms from (2) and (3), the error state $\delta \mathbf { x }$ and the covariance $\hat { \mathbf { P } }$ follow the linear dynamics model:

$$
\begin{array} { r l } & { \delta \mathbf { x } _ { i + 1 } = \mathbf { F } _ { \delta \mathbf { x } _ { i } } \delta \mathbf { x } _ { i } + \mathbf { F } _ { \mathbf { w } _ { i } } \mathbf { w } _ { i } , } \\ & { \widehat { \mathbf { P } } _ { i + 1 } = \mathbf { F } _ { \delta \mathbf { x } _ { i } } \widehat { \mathbf { P } } _ { i } \mathbf { F } _ { \delta \mathbf { x } _ { i } } ^ { \top } + \mathbf { F } _ { \mathbf { w } _ { i } } \mathbf { Q } _ { i } \mathbf { F } _ { \mathbf { w } _ { i } } ^ { \top } , } \end{array}\tag{4}
$$

where $\mathbf { F } _ { \delta \mathbf { x } _ { i } }$ and $\mathbf { F } _ { \mathbf { w } _ { i } }$ are respectively the transition matrix and noise Jacobian matrix linearized at $\widehat { \mathbf { x } } _ { i } [ 6 ] . \mathbf { Q } _ { i }$ is the covariance of the noise $\mathbf { w } _ { i } .$ , which can be obtained from the offline IMU calibration [24]. The initial state $\widehat { \mathbf { x } } _ { 0 }$ and covariance $\widehat { \mathbf { P } } _ { 0 }$ of the current propagation are the updated state $\overline { { \mathbf { x } } } _ { k - 1 }$ and covariance $\overline { { \mathbf { P } } } _ { k - 1 }$ at the last scan-end time $t _ { k - 1 }$

## C. Motion Compensation and State Update

1) Motion Compensation for Doppler Measurements: A generic LiDAR usually returns accumulated scans at a fixed frequency (e.g., 10 Hz), instead of returning each individual LiDAR point separately. The system is propagated from the last update at the last scan-end time $t _ { k - 1 }$ until the current scan arrives at time $t _ { k }$ . However, although a LiDAR scan arrives at a certain time, each LiDAR point in this scan is sampled at its individual discrete time. Hence, for a moving LiDAR, each point is sampled with respect to the LiDAR position at its sampling time, instead of the LiDAR position at the scan-end time. To exactly use LiDAR points in state update, we need to correct point positions in a scan by motion compensation.

For a LiDAR point $p _ { j }$ sampled at $t _ { j }$ between two adjacent IMU sampling moments $t _ { i - 1 }$ and $t _ { i }$ , we can perform backward propagation [6] with IMU to correct its position measurements, obtaining the corrected position $l _ { k } \widetilde { \mathbf { p } } _ { l _ { k } p _ { j } }$ of $p _ { j }$ with respect to the LiDAR position at the scan-end time $t _ { k }$ . We denote the raw position measurement of $p _ { j }$ as ${ l } _ { j } \widetilde { \mathbf { p } } _ { { l } _ { j } { p } _ { j } }$ , which is with respect to the LiDAR position at its sampling time $t _ { j } . ~ ( \check { \cdot } )$ represents the backward propagated state at the point sampling time. The motion compensation for the point position measurement is:

$$
\begin{array} { r l } & { { \mathbf \Lambda } ^ { l _ { k } } \widetilde { \mathbf p } _ { l _ { k } p _ { j } } = \widehat { \mathbf R } _ { b l } ^ { \top } ( \widehat { \mathbf R } _ { w b _ { k } } ^ { \top } ( \check { \mathbf R } _ { w b _ { j } } ( \check { \mathbf R } _ { b l } { } ^ { l _ { j } } \widetilde { \mathbf p } _ { l _ { j } p _ { j } } + { } ^ { b } \check { \mathbf p } _ { b l } )   } \\ & { \qquad +   { } ^ { w } \check { \mathbf p } _ { w b _ { j } } - { } ^ { w } \widehat { \mathbf p } _ { w b _ { k } } ) - { } ^ { b } \widehat { \mathbf p } _ { b l } ) , } \end{array}\tag{5}
$$

where $( \widehat { \mathbf { R } } _ { w b _ { k } } , \ w _ { \widehat { \mathbf { p } } _ { w b _ { k } } } )$ and $( \check { \mathbf { R } } _ { w b _ { j } } , { } ^ { w } \check { \mathbf { p } } _ { w b _ { j } } )$ are respectively system poses at the scan-end time $t _ { k }$ and the sampling time $t _ { j }$ of point $p _ { j }$ . This step can achieve the motion compensation for positions except for Doppler measurements. Next, we design an accurate motion compensation method for Doppler measurements, enabling the state update at $t _ { k }$

Under the implicit assumption in the above position compensation that LiDAR points are static in the world frame $( \mathrm { i . e . } ,$ $w _ { \mathbf { V } _ { w p _ { i } } } \equiv \mathbf { 0 } )$ , we can derive a motion compensation method for Doppler measurements. For a static point $p _ { j }$ , its Doppler velocity $\widetilde { v } _ { i } ^ { d _ { j } }$ is measured with respect to the LiDAR position and LiDAR velocity at its sampling time $t _ { j }$ [2]:

$$
\widetilde { v } _ { j } ^ { d _ { j } } = - \frac { l _ { j } \widetilde { \mathbf { p } } _ { l _ { j } p _ { j } } ^ { \intercal } } { \left\| l _ { j } \widetilde { \mathbf { p } } _ { l _ { j } p _ { j } } \right\| } l _ { j } \check { \mathbf { v } } _ { w l _ { j } } ,\tag{6}
$$

where $l _ { j }  { \bf { \check { v } } } _ { w l _ { j } }$ is the LiDAR velocity with respect to the world frame and projected in the LiDAR frame at $t _ { j } .$ For $p _ { j }$ , its Doppler velocity $\widetilde { v } _ { j } ^ { d _ { k } }$ , with respect to the LiDAR position and LiDAR velocity at the scan-end time $t _ { k } ,$ is unequal to the raw measurement $\bar { \widetilde { v } } _ { j } ^ { d _ { j } }$ at $t _ { j }$ . Similar to (6), $\widetilde { v } _ { j } ^ { d _ { k } }$ can be derived as:

$$
\widetilde { v } _ { j } ^ { d _ { k } } = - \frac { l _ { k } \widetilde { \mathbf { p } } _ { l _ { k } p _ { j } } ^ { \top } } { \left\| l _ { k } \widetilde { \mathbf { p } } _ { l _ { k } p _ { j } } \right\| } { l _ { k } } \widehat { \mathbf { v } } _ { w l _ { k } } ,\tag{7}
$$

where $l _ { k } \widehat { \mathbf { v } } _ { w l _ { k } }$ is the LiDAR velocity with respect to the world frame and projected in the LiDAR frame at $t _ { k }$ . To compensate the Doppler measurement from $\widetilde { v } _ { j } ^ { d _ { j } }$ to $\widetilde { v } _ { j } ^ { d _ { k } }$ , we can use the forward and backward propagated states, LiDAR-IMU extrinsic parameters, and IMU measurements with the transformation between $\boldsymbol { l } _ { j } \check { \mathbf { v } } _ { w l _ { i } }$ and $l _ { k } \widehat { \mathbf { v } } _ { w l _ { k } }$ . Then substituting (5), (6), and $\begin{array} { r } { \iota _ { \mathbf { v } _ { w l } } = \mathbf { R } _ { b l } ^ { \top } ( \mathbf { R } _ { w b } ^ { \top } \mathbf { \Phi } ^ { w } \mathbf { v } _ { w b } + \mathbf { \Phi } ^ { b } \omega _ { w b } \times \mathbf { \Phi } ^ { b } \mathbf { p } _ { b l } ) } \end{array}$ into (7):

$$
\begin{array} { r l r } {  { \widetilde { v } _ { j } ^ { d _ { k } } = \frac { \| { } ^ { l _ { j } } \widetilde { \mathbf { p } } _ { l _ { j } p _ { j } } \| } { \| { } ^ { l _ { k } } \widetilde { \mathbf { p } } _ { l _ { k } p _ { j } } \| } \widetilde { v } _ { j } ^ { d _ { j } } - \frac { 1 } { \| { } ^ { l _ { k } } \widetilde { \mathbf { p } } _ { l _ { k } p _ { j } } \| } [ { } ^ { w } \check { \mathbf { p } } _ { l _ { j } p _ { j } } ^ { \top } ( { } ^ { w } \widehat { \mathbf { v } } _ { w l _ { k } } - { } ^ { w } \check { \mathbf { v } } _ { w l _ { j } } )  } } \\ & { } & { +  ( { } ^ { w } \check { \mathbf { p } } _ { w l _ { j } } - { } ^ { w } \widehat { \mathbf { p } } _ { w l _ { k } } ) ^ { \top } { } ^ { w } \widehat { \mathbf { v } } _ { w l _ { k } } ] , } \end{array}
$$

where

$$
\begin{array} { r l } & { ^ w \check { \mathbf { p } } _ { l j p j } = \hat { \mathbf { R } } _ { w b _ { j } } \check { \mathbf { R } } _ { b l } { } ^ { l } \dot { \mathbf { p } } _ { l j p j } , } \\ & { ^ w \check { \mathbf { p } } _ { w l _ { j } } = { } ^ { w } \check { \mathbf { p } } _ { w b _ { j } } + \hat { \mathbf { R } } _ { w b _ { j } } { } ^ { b } \check { \mathbf { p } } _ { b l } , } \\ & { ^ w \widehat { \mathbf { p } } _ { w l _ { k } } = { } ^ { w } \widehat { \mathbf { p } } _ { w b _ { k } } + \hat { \mathbf { R } } _ { w b _ { k } } { } ^ { b } \widehat { \mathbf { p } } _ { b l } , } \\ & { ^ w \check { \mathbf { v } } _ { w l _ { j } } = { } ^ { w } \check { \mathbf { v } } _ { w b _ { j } } + \hat { \mathbf { R } } _ { w b _ { j } } \left( { } ^ { b _ { j } } \omega _ { w b _ { j } } \times { } ^ { b } \check { \mathbf { p } } _ { b l } \right) , } \\ & { ^ w \widehat { \mathbf { v } } _ { w l _ { k } } = { } ^ { w } \widehat { \mathbf { v } } _ { w b _ { k } } + \hat { \mathbf { R } } _ { w b _ { k } } \left( { } ^ { b _ { k } } \widehat { \omega } _ { w b _ { k } } \times { } ^ { b } \widehat { \mathbf { p } } _ { b l } \right) . } \end{array}\tag{9}
$$

In (9), $( \widehat { \mathbf { R } } _ { w b _ { k } } , { } ^ { w } \widehat { \mathbf { p } } _ { w b _ { k } } , { } ^ { w } \widehat { \mathbf { v } } _ { w b _ { k } } )$ are forward propagated states at $t _ { k } , ( \check { \mathbf { R } } _ { w b _ { j } } , { w }  \check { \mathbf { p } } _ { w b _ { j } } , { w }  \check { \mathbf { v } } _ { w b _ { j } } )$ are backward propagated states at $t _ { j } . \stackrel { b _ { j } } { \omega } _ { w b _ { j } }$ is the unbiased gyroscope measurement $( \mathrm { i . e . } ,$ ${ } ^ { b _ { j } } \tilde { \omega } _ { w b _ { j } } = { } ^ { b _ { j } } \tilde { \widetilde { \omega } } _ { j } - { } ^ { b _ { j } } \check { \mathbf { b } } _ { g } )$ at $t _ { j }$ from the IMU measurement interpolation, and $b _ { k } \widehat { \omega } _ { w b _ { k } }$ is the unbiased gyroscope measurement at $t _ { k } .$ . Notably, beyond motion compensation, (8) essentially reveals the interrelation between the Doppler velocities of a static point with respect to arbitrary different LiDAR frames. Finally, we can align all points in a scan to the scan-end time $t _ { k }$ under a unified temporal-spatial frame, by the motion compensation for both position (5) and Doppler (8) measurements.

2) LiDAR Velocity Observation: In a motion-compensated scan at $t _ { k }$ , each point can provide the radial information of the LiDAR velocity in the LiDAR frame. As in (7), the Doppler velocity of a static point can be predicted using the point direction and the LiDAR velocity. The normalized direction vector $l _ { k } \widetilde { \mathbf { d } } _ { l _ { k } p _ { j } }$ of point $p _ { j }$ after motion compensation can be computed through: $\begin{array} { r } { l _ { k } \widetilde { \mathbf { d } } _ { l _ { k } p _ { j } } = \frac { { { \mathbf { \ell } } ^ { l _ { k } } } \widetilde { \mathbf { p } } { { \mathbf { } } _ { l _ { k } p _ { j } } } } { \left\| { { \mathbf { \ell } } ^ { l _ { k } } } \widetilde { \mathbf { p } } { { \mathbf { } } _ { l _ { k } p _ { j } } } \right\| } } \end{array}$ . For n points, a matrix formulation can be yielded by stacking all points:

$$
\begin{array} { r } { \widetilde { \mathbf { D } } _ { k } { } ^ { l _ { k } } \mathbf { v } _ { w l _ { k } } = \widetilde { \mathbf { V } } _ { k } , } \end{array}\tag{10}
$$

where $\widetilde { \mathbf { D } } _ { k } = \left[ \ldots \quad - \mathbf { \widetilde { k } } \widetilde { \mathbf { d } } _ { l _ { k } p _ { j } } \quad \ldots \right] ^ { \top }$ is a $n \times 3$ matrix and $\widetilde { \mathbf { V } } _ { k } = \left\lceil \dots \quad \widetilde { v } _ { j } ^ { d _ { k } } \quad \dots \right\rceil ^ { \top }$ is a $n \times 1$ vector. Accordingly, we can estimate a LiDAR velocity measurement at the scan-end time $t _ { k }$ by solving a least squares problem [18]:

$$
{ ^ { l _ { k } } \widetilde { \mathbf { v } } _ { w l _ { k } } } = \mathop { \arg \operatorname* { m i n } } _ { { ^ { l _ { k } } \mathbf { v } _ { w l _ { k } } } } \left\| \widetilde { \mathbf { D } } _ { k } { ^ { l _ { k } } \mathbf { v } _ { w l _ { k } } } - \widetilde { \mathbf { V } } _ { k } \right\| ^ { 2 } = \left( \widetilde { \mathbf { D } } _ { k } ^ { \top } \widetilde { \mathbf { D } } _ { k } \right) ^ { - 1 } \widetilde { \mathbf { D } } _ { k } ^ { \top } \widetilde { \mathbf { V } } _ { k } .
$$

Meanwhile, the covariance of the estimated LiDAR velocity, $\mathbf { R } _ { \mathbf { v } } .$ , can be computed from the estimation residual [18], serving as the observation covariance in the state update.

For an FMCW Doppler LiDAR scan containing tens of thousands of points, directly solving the above problem is computationally burdensome and error-prone, due to potential dynamic points. Thus, a 3-point Random Sample and Consensus (RANSAC) method [18] is applied to efficiently obtain a robust solution. For example, setting the outlier probability to 30% and the desired success probability to 99%, the iteration number is 11, which leads to a high efficiency. Besides, to prevent the solution from falling into the null-space of (10)

![](images/5b9b6d247dfb3bf89f661afa773686de44bd5ad567f756cf618d3add37c32bd0.jpg)  
Fig. 3. Illustration of the feedback loop between the state estimate and the online map.

(i.e., $\widetilde { \mathbf { D } } _ { k } { } ^ { l _ { k } } \mathbf { v } _ { w l _ { k } } \ = \ \mathbf { 0 } _ { n \times 1 } )$ and improve the ratio of valid samples, the sampling probability is weighted by the absolute cosine of the angle $\phi _ { \widetilde { \mathbf { v } } _ { l } \widetilde { \mathbf { d } } _ { i } }$ between the point direction and the forward propagated LiDAR velocity in the LiDAR frame:

$$
\begin{array} { r } { \left\| \cos \left( \phi _ { \widehat { \mathbf { v } } _ { l } \widetilde { \mathbf { d } } _ { j } } \right) \right\| = \frac { \left\| \big \langle { } ^ { l _ { k } } \widetilde { \mathbf { d } } _ { l _ { k } p _ { j } } , { } ^ { l _ { k } } \widehat { \mathbf { v } } _ { w l _ { k } } \big \rangle \right\| } { \left\| { } ^ { l _ { k } } \widetilde { \mathbf { d } } _ { l _ { k } p _ { j } } \right\| \left\| { } ^ { l _ { k } } \widehat { \mathbf { v } } _ { w l _ { k } } \right\| } , } \end{array}\tag{11}
$$

where ${ l _ { k } } _ { \widehat { \mathbf { v } } _ { w l _ { k } } } \ = \ \Big ( \widehat { \mathbf { R } } _ { w b _ { k } } \widehat { \mathbf { R } } _ { b l } \Big ) ^ { \top } w _ { \widehat { \mathbf { v } } _ { w l _ { k } } }$ is the forward propagated LiDAR velocity in the LiDAR frame at the scan-end time $t _ { k }$ , and $w _ { \widehat { \mathbf { v } } _ { w l _ { k } } }$ can be computed from (9).

The above estimated measurement of LiDAR velocity provides a correspondence-free observation for the system state. The LiDAR velocity observation model $\mathbf { h } _ { \mathbf { v } } \left( \mathbf { x } , \mathbf { \xi } ^ { l } \mathbf { n } _ { \mathbf { v } } \right)$ is:

$$
\begin{array} { r l } & { l _ { \widetilde { \mathbf { v } } _ { w l } } ^ { \sim } = \mathbf { h } _ { \mathbf { v } } \left( \mathbf { x } , \mathbf { \xi } ^ { l } \mathbf { n } _ { \mathbf { v } } \right) } \\ & { \qquad \triangleq \mathbf { R } _ { b l } ^ { \top } \left( \mathbf { R } _ { w b } ^ { \top } \mathbf { \Phi } \mathbf { v } _ { w b } + \left( ^ { b } \widetilde { \omega } - ^ { b } \mathbf { b } _ { g } - \mathbf { n } _ { g } \right) \times ^ { b } \mathbf { p } _ { b l } \right) + ^ { l } \mathbf { n } _ { \mathbf { v } } , } \end{array}
$$

where the covariance of ${ l } _ { { \bf n } _ { \bf v } }$ is the aforementioned $\mathbf { R } _ { \mathbf { v } }$ . To reach the optimal estimate at $t _ { k }$ , the LiDAR velocity residual $\mathbf { r } _ { \mathbf { v } _ { k } } = { l _ { k } }  _ { \widetilde { \mathbf { v } } _ { w l _ { k } } } - { l _ { k } } _ { \widehat { \mathbf { v } } _ { w l _ { k } } }$ can be linearized as:

$$
\mathbf { r } _ { \mathbf { v } _ { k } } = \mathbf { h } _ { \mathbf { v } } ( \mathbf { x } _ { k } , \mathbf { n } _ { \mathbf { v } } ) - \mathbf { h } _ { \mathbf { v } } ( \widehat { \mathbf { x } } _ { k } , \mathbf { 0 } ) \approx \mathbf { H } _ { \mathbf { v } _ { k } } \delta \mathbf { x } _ { k } + { } ^ { l } \mathbf { n } _ { \mathbf { v } } .\tag{12}
$$

The observation Jacobian matrix $\mathbf { H } _ { \mathbf { v } _ { k } } \in \mathbb { R } ^ { 3 \times 2 4 }$ is:

$$
\mathbf { H } _ { \mathbf { v } _ { k } } = \left[ \mathbf { H } _ { \delta \pmb { \theta } _ { k } } \quad \mathbf { 0 } _ { 3 \times 3 } \quad \mathbf { H } _ { \delta \mathbf { v } _ { k } } \quad \mathbf { H } _ { \delta \mathbf { b } _ { g k } } \quad \mathbf { 0 } _ { 3 \times 6 } \quad \mathbf { H } _ { \delta \pmb { \theta } _ { b l } } \quad \mathbf { H } _ { \delta \mathbf { p } _ { b l } } \right] .
$$

The different block components of $\mathbf { H } _ { \mathbf { v } _ { k } }$ are:

$$
\begin{array} { r l } & { \mathbf H _ { \delta \pmb \theta _ { k } } = \widehat { \mathbf R } _ { b l } ^ { \top } [ \widehat { \mathbf R } _ { w b _ { k } } ^ { \top } ^ { \top } ^ { \top } ^ { \top } \widehat { \mathbf v } _ { w b _ { k } } ] _ { \times } , \ \mathbf H _ { \delta \mathbf v _ { k } } = ( \widehat { \mathbf R } _ { w b _ { k } } \widehat { \mathbf R } _ { b l } ) ^ { \top } } \\ & { \mathbf H _ { \delta \mathbf b _ { g } _ { k } } = \widehat { \mathbf R } _ { b l } ^ { \top } [ ^ { b } \widehat { \mathbf p } _ { b l } ] _ { \times } , \ \mathbf H _ { \delta \mathbf p _ { b l } } = \widehat { \mathbf R } _ { b l } ^ { \top } [ ^ { b _ { k } } \widetilde \omega _ { k } - ^ { b _ { k } } \widehat { \mathbf p } _ { g } ] _ { \times } } \\ & { \mathbf H _ { \delta \pmb \theta _ { b l } } = [ \widehat { \mathbf R } _ { b l } ^ { \top } ( \widehat { \mathbf R } _ { w b _ { k } } ^ { \top } ^ { \top } ^ { \top } \^ { w } \widehat { \mathbf v } _ { w b _ { k } } + \binom { b _ { k } } { \alpha } _ { k } - ^ { b _ { k } } \widehat { \mathbf b } _ { g } ) \times ^ { b } \widehat { \mathbf p } _ { b l } ) ] _ { \times } . } \end{array}
$$

On account of the low dimension of 3D LiDAR velocity observation $( \dim ( \mathbf { r } _ { \mathbf { v } _ { k } } ) \ll \dim ( \mathcal { M } ) )$ , the conventional formulation of Kalman gain instead of (15) is applied:

$$
\mathbf { K } _ { \mathbf { v } _ { k } } = \widehat { \mathbf { P } } _ { k } \mathbf { H } _ { \mathbf { v } _ { k } } ^ { \top } \Big ( \mathbf { H } _ { \mathbf { v } _ { k } } \widehat { \mathbf { P } } _ { k } \mathbf { H } _ { \mathbf { v } _ { k } } ^ { \top } + \mathbf { R } _ { \mathbf { v } } \Big ) ^ { - 1 } .
$$

The updated state $\overline { { \mathbf { x } } } _ { \mathbf { v } _ { k } }$ and covariance $\overline { { \mathbf { P } } } _ { \mathbf { v } _ { k } }$ using the LiDAR velocity observation are respectively:

$$
\begin{array} { r l } & { \overline { { \mathbf { x } } } _ { \mathbf { v } _ { k } } = \widehat { \mathbf { x } } _ { k } \boxplus \overline { { \delta \mathbf { x } } } _ { \mathbf { v } _ { k } } = \widehat { \mathbf { x } } _ { k } \boxplus \big ( \mathbf { K } _ { \mathbf { v } _ { k } } \mathbf { r } _ { \mathbf { v } _ { k } } \big ) , } \\ & { \overline { { \mathbf { P } } } _ { \mathbf { v } _ { k } } = \mathbf { L } _ { \mathbf { v } _ { k } } \big ( \mathbf { I } _ { 2 4 \times 2 4 } - \mathbf { K } _ { \mathbf { v } _ { k } } \mathbf { H } _ { \mathbf { v } _ { k } } \big ) \widehat { \mathbf { P } } _ { k } \mathbf { L } _ { \mathbf { v } _ { k } } ^ { \top } , } \end{array}\tag{13}
$$

where $\mathbf { L } _ { \mathbf { v } _ { k } }$ is the projection matrix between tangent spaces arising from the on-manifold increment $\overline { { \delta \mathbf { x } } } _ { \mathbf { v } _ { k } }$ [23].

3) LiDAR Geometric Observation: Apart from the instant correspondence-free Doppler-aided observation, the geometric association from LiDAR points can also provide observations. Here we use the point-plane distance model [7] [25]:

$$
\begin{array} { r l } & { 0 = h _ { g _ { j } } \left( \mathbf { x } , \mathbf { \xi } ^ { l } \mathbf { n } _ { p _ { j } } \right) } \\ & { \quad \triangleq ^ { w } \mathbf { n } _ { p l a n e _ { j } } ^ { \top } \left( \mathbf { T } _ { w b } \mathbf { T } _ { b l } \left( \mathbf { \xi } ^ { l } \widetilde { \mathbf { p } } _ { l p _ { j } } - \mathbf { \xi } ^ { l } \mathbf { n } _ { p _ { j } } \right) - \mathbf { \xi } ^ { w } \widetilde { \mathbf { q } } _ { w q _ { j } } \right) , } \end{array}\tag{14}
$$

where $\mathbf { T } _ { w b } , \mathbf { T } _ { b l } \in S E ( 3 )$ are respectively the system pose and extrinsic parameters, $\boldsymbol { l } _ { \mathbf { n } _ { p _ { j } } }$ is the point position noise, $w _ { \mathbf { n } _ { p l a n e _ { j } } }$ is the unit normal vector of the nearest neighbor plane of point $p _ { j }$ in the world frame, and $w _ { \widetilde { \mathbf { q } } _ { w q _ { j } } }$ is a point in that plane projected in the world frame.

In IESEKF, the state updated by the LiDAR velocity observation give a prior Gaussian distribution in the tangent space at $\overline { { \mathbf { x } } } _ { \mathbf { v } _ { k } }$ , but the optimal estimate is derived in the tangent space at $\widehat { \mathbf { x } } _ { k } ^ { \kappa }$ , i.e., the state of κ-th update iteration (equals the updated state after (κ − 1)-th update iteration):

$$
\widehat { \mathbf { x } } _ { k } ^ { \kappa } = \overline { { \mathbf { x } } } _ { k } ^ { \kappa - 1 } = \widehat { \mathbf { x } } _ { k } ^ { \kappa - 1 } \boxplus \overline { { \delta \mathbf { x } } } _ { k } ^ { \kappa - 1 } , \ \widehat { \mathbf { x } } _ { k } ^ { 0 } = \overline { { \mathbf { x } } } _ { \mathbf { v } _ { k } } .
$$

With $\delta \mathbf { x } _ { k } = \mathbf { x } _ { k } \ominus \bar { \mathbf { x } } _ { \mathbf { v } _ { k } } \sim \mathcal { N } \left( \mathbf { 0 } _ { 2 4 \times 1 } , \mathbf { \overline { { P } } } _ { \mathbf { v } _ { k } } \right)$ , the prior distribution projected in the tangent space at $\widehat { \mathbf { x } } _ { k } ^ { \kappa }$ imposing a prior error-state distribution in the κ-th update iteration:

$$
\delta \mathbf { x } _ { k } ^ { \kappa } \sim \mathcal { N } \left( - \mathbf { J } _ { k } ^ { \kappa } \left( \widehat { \mathbf { x } } _ { k } ^ { \kappa } \boxdot \nabla \overline { { \mathbf { x } } } _ { \mathbf { v } _ { k } } \right) , \mathbf { J } _ { k } ^ { \kappa } \mathbf { \overline { { P } } } _ { \mathbf { v } _ { k } } \mathbf { J } _ { k } ^ { \kappa \top } \right) ,
$$

where $\mathbf { J } _ { k } ^ { \kappa }$ is the tangent projection matrix because of the onmanifold update step $\big ( \widehat { \mathbf { x } } _ { k } ^ { \kappa } \boxminus \overline { { \mathbf { x } } } _ { \mathbf { v } _ { k } } \big )$ , similar to (13). The LiDAR geometric observation residual of point $p _ { j }$ at $t _ { k }$ is:

$$
\mathbf { r } _ { g _ { k _ { j } } } ^ { \kappa } = 0 - h _ { g _ { j } } ( \widehat { \mathbf { x } } _ { k } ^ { \kappa } , \mathbf { 0 } ) = - h _ { g _ { j } } ( \widehat { \mathbf { x } } _ { k } ^ { \kappa } , \mathbf { 0 } ) .
$$

Then, fusing the prior distribution and the likelihood observation distribution from all observations, an equivalent maximum a-posteriori estimate (MAP) can be obtained [23].

Therefore, the iterated state update follows:

$$
\begin{array} { r l } & { { \bf K } _ { g _ { k } } ^ { \kappa } = \left( { \bf H } _ { g _ { k } } ^ { \kappa } { \bf R } _ { g _ { k } } ^ { - 1 } { \bf H } _ { g _ { k } } ^ { \kappa } + \left( { \bf J } _ { k } ^ { \kappa } { \bf \overline { { P } } } _ { { \bf v } _ { k } } { \bf J } _ { k } ^ { \kappa \top } \right) ^ { - 1 } \right) ^ { - 1 } { \bf H } _ { g _ { k } } ^ { \kappa \top } { \bf R } _ { g _ { k } } ^ { - 1 } , } \\ & { \overline { { \delta } } { { \bf x } } _ { k } ^ { \kappa } = { \bf K } _ { g _ { k } } ^ { \kappa } { \bf r } _ { g _ { k } } ^ { \kappa } + \left( { \bf K } _ { g _ { k } } ^ { \kappa } { \bf H } _ { g _ { k } } ^ { \kappa } - { \bf I } _ { 2 4 \times 2 4 } \right) { \bf J } _ { k } ^ { \kappa } \left( \widehat { \bf x } _ { k } ^ { \kappa } \boxplus \overline { { \bf x } } _ { { \bf v } _ { k } } \right) , } \\ & { \widehat { \bf x } _ { k } ^ { \kappa + 1 } = \widehat { \bf x } _ { k } ^ { \kappa } \boxplus \overline { { \delta } } { \bf x } _ { k } ^ { \kappa } , \qquad ( 1 \ddagger ) } \end{array}\tag{5}
$$

where $\mathbf { r } _ { g _ { k } } ^ { \kappa } , \mathbf { H } _ { g _ { k } } ^ { \kappa }$ , and $\mathbf { R } _ { g _ { k } }$ are formulated by stacking matrices of all valid points. Finally, after the convergence of IESEKF, the optimally updated state and covariance are:

$$
\begin{array} { r l } & { \overline { { \mathbf { x } } } _ { k } = \overline { { \mathbf { x } } } _ { g _ { k } } = \widehat { \mathbf { x } } _ { k } ^ { \kappa + 1 } , } \\ & { \overline { { \mathbf { P } } } _ { k } = \overline { { \mathbf { P } } } _ { g _ { k } } = \mathbf { L } _ { g _ { k } } \left( \mathbf { I } _ { 2 4 \times 2 4 } - \mathbf { K } _ { g _ { k } } ^ { \kappa } \mathbf { H } _ { g _ { k } } ^ { \kappa } \right) \big ( \mathbf { J } _ { k } ^ { \kappa } \overline { { \mathbf { P } } } _ { \mathbf { v } _ { k } } \mathbf { J } _ { k } ^ { \kappa \top } \big ) \mathbf { L } _ { g _ { k } } ^ { \top } . } \end{array}\tag{16}
$$

In (16), $\overline { { \mathbf { x } } } _ { g _ { k } }$ and $\overline { { \mathbf { P } } } _ { g _ { k } }$ are the updated state and covariance using the LiDAR geometric observation, $\overline { { \mathbf { x } } } _ { k }$ and $\overline { { \mathbf { P } } } _ { k }$ are the optimal state estimate and covariance after both LiDAR velocity and geometric update at the scan-end time $t _ { k }$

## D. Dynamic Point Removal and Static Mapping

The LiDAR geometric observation in (14) depends on the online established map. Dynamic points in the LiDAR scan can break the static world assumption and introduce significant errors. Importantly, dynamic points can contaminate the online map, which is prone to provide erroneous geometric correspondences. This dependence creates a loop between the state estimate and the map that can destabilize the system, as shown in the left of Fig. 3. Specifically, if the state estimate is disturbed by dynamic points, the online map will be inaccurate and chaotic, and the erroneous correspondences from dynamic points will amplify the estimation error. This positive feedback loop of the estimation error may lead to divergence even failure of the state estimator.

![](images/e29b3eabd19853c6d334771c1de5e968b0cf05ed51385b96ef533d631753fb69.jpg)  
Fig. 4. The sensor suite on the (a) handheld and the (b) wheeled platforms.

Fortunately, the above problem in traditional LiDAR systems can be solved directly and efficiently thanks to the inherent Doppler measurements from FMCW Doppler LiDARs. After the state update in (13), we can predict a Doppler velocity $\widetilde { v } _ { j , p r e d } ^ { d _ { k } }$ for point $p _ { j }$ . If the discrepancy between the prediction and the measurement exceeds a given threshold, the point is removed to prevent it from being added to the LiDAR geometric observation and the online map. The method is similar to the outlier rejection mentioned in [2] but the threshold $\delta v _ { t h r } ^ { d }$ is adjustable according to the angle $\phi _ { \overline { { \mathbf { v } } } _ { l } \widetilde { \mathbf { d } } _ { i } }$ between the normalized direction of a point and the LiDAR velocity from the updated state in (13). The condition that the point $p _ { j }$ is regarded as a dynamic point is:

$$
\left\| \widetilde { v } _ { j , p r e d } ^ { d _ { k } } - \widetilde { v } _ { j } ^ { d _ { k } } \right\| > \delta v _ { t h r } ^ { d } \left\| \cos \left( \phi _ { \overline { { \mathbf { v } } } _ { l } \widetilde { \mathbf { d } } _ { j } } \right) \right\| ,\tag{17}
$$

where the cosine is in (11) but uses the updated state $l _ { k } \overline { { \mathbf { v } } } _ { w l _ { k } }$

Significantly, as shown in (11) and (17), this cosine reveals a crucial physical insight: the effectiveness and the signal-tonoise ratio of a LiDAR point in estimating the 3D linear Li-DAR velocity depend on the angle (relative direction) between the point direction and the LiDAR velocity projected in the LiDAR frame $( \mathrm { i } . \mathrm { e } . , \ ^ { l } \mathbf { v } _ { w l } )$ , rather than directly on the LiDAR velocity projected in the world frame $( \mathrm { i } . \mathrm { e } . , \ ^ { w } \mathbf { v } _ { w l } )$ or the Fieldof-View (FoV) of an FMCW Doppler LiDAR. Meanwhile, this cosine represents the radial nature of the Doppler effect.

Consequently, only almost static points are fed into the LiDAR geometric observation and the map, which can further improve the accuracy and robustness of the online system. FMCW-LIO can eventually maintain a neat and static map.

Meanwhile, the Doppler-aided correspondence-free observations can partially alleviate the drawback of the aforementioned positive error feedback loop. This observation model reduces the reliance on historical associations from the online map, as shown in the right of Fig. 3.

## V. EXPERIMENTAL RESULTS

## A. FMCW Doppler LiDAR-Inertial Data Collection

It is necessary to collect the FMCW Doppler LiDAR-inertial data in the real world to validate the proposed method, due to the absence of public FMCW Doppler LiDAR-inertial datasets including raw Doppler measurements, as mentioned in Section II-B. We use an Aeva Aeries II [26] FMCW Doppler LiDAR and an Xsens MTi-G-710 IMU to assemble our sensor suite. This sensor suite can be mounted on handheld or wheeled platforms, as in Fig. 4. The LiDAR FoV is $1 2 0 ^ { \circ } \times 2 8 . 8 ^ { \circ }$ The LiDAR returns 10 Hz scans including position and raw Doppler measurements. The frequency of IMU measurements is 200 Hz. The computing device is the Intel NUC with an i7-1165G7 CPU.

The data description is shown in Table I. For the handheld platform, the sensor suite is held by a walking person with persistent shaking and swing. The speed of the handheld platform is slow (about 1.5 m/s). For the wheeled platform, the sensor suite is mounted on a vehicle and the moving speed is relatively fast (up to 7.0 m/s). To comprehensively evaluate different algorithms, diverse conditions (structured vs. structure-degenerated scenes, wide vs. narrow scenes, slow vs. fast speeds) are considered, as shown in Fig. 5. In campus 01 and campus 02, we traverse the campus of the University of Macau and the scene is wide enough to extract and match geometric features for general LIO systems. On the contrary, street 01 and street 02 are collected in narrow streets in downtown Macao, thus the scene is considerably narrow leading to significant challenges for LIO systems. Furthermore, to validate the Doppler-aided observation, four structure-degenerated sequences are collected in a subaqueous pedestrian tunnel. The tunnel is S-shaped and the one-side length is about 850 m. Since the dataset is challenging and the GNSS accuracy is highly unreliable in these narrow, amongbuilding, or indoor scenes, we choose the trajectory optimized by offline post-processing as the ground-truth reference. We utilize a state-of-the-art loop closure [27] and the pose graph optimization [28] to generate the ground truth. Notably, all post-processed ground-truth trajectories are meticulously examined to ensure alignment with the real world. Besides, we use two metrics, absolute translational error (RMSE) and endto-end error (E2E) [7], to evaluate algorithms. Therefore, the starting and ending positions of all sequences are confirmed to be exactly coincident to enable the end-to-end evaluation.

For the thorough evaluation and ablation study of different algorithms and modules, we compare FMCW-LIO with conventional mainstream LIO systems under diverse framework designs, i.e., LINS [5], LIO-SAM [3], DLIO [10], and FAST-LIO2 [7]. We also evaluate the recently released Doppleraided LO, STEAM-DICP [14], which can serve as an ablation study to validate the role of IMU on handheld or small and medium-scale robot platforms with narrow sensing FoV, narrow workspaces, and high-frequency dynamic motions. We tune the parameters of all algorithms to achieve their optimal accuracy and use these tuned parameters in all experiments. In FMCW-LIO, we use ikd-Tree [7] as the map structure, and the local map size is set to 1000 m. We also test two variants of FMCW-LIO to perform ablation studies and validate the effectiveness of two modules, including 1) the variant without the Doppler compensation (w/o DC) and 2) the variant without the dynamic point removal (w/o DR).

TABLE I  
DESCRIPTION OF THE COLLECTED DATASET
<table><tr><td>Sequence</td><td>Platform</td><td>Speed</td><td>Scene Type</td><td>Length (m)</td><td>Duration (s)</td></tr><tr><td>campus_01</td><td>Handheld</td><td>Slow</td><td>Wide</td><td>425</td><td>403</td></tr><tr><td>campus_02</td><td>Wheeled</td><td>Fast</td><td>Wide</td><td>454</td><td>172</td></tr><tr><td>street_01</td><td>Handheld</td><td>Slow</td><td>Narrow</td><td>182</td><td>204</td></tr><tr><td>street_02</td><td>Handheld</td><td>Slow</td><td>Narrow</td><td>186</td><td>223</td></tr><tr><td>tunnel_01</td><td>Handheld</td><td>Slow</td><td>Degenerated</td><td>1129</td><td>740</td></tr><tr><td>tunnel_02</td><td>Wheeled</td><td>Fast</td><td>Degenerated</td><td>232</td><td>90</td></tr><tr><td>tunnel_03</td><td>Handheld</td><td>Slow</td><td>Degenerated</td><td>283</td><td>179</td></tr><tr><td>tunnel_04</td><td>Wheeled</td><td>Fast</td><td>Degenerated</td><td>1712</td><td>368</td></tr></table>

![](images/817ff74ec20d01e202cffcb728130e52b27d895114481e86dbe18576c2bd0471.jpg)  
Fig. 5. The scenes of the collected dataset. (a) and (b) are campus scenes. (c) and (d) are narrow street scenes. (e) is the tunnel scene.

## B. Experiments in Structured Environments

The results in four structured scenes are in the left four columns of Table II and the loop closure of LIO-SAM is disabled for fair comparisons. FMCW-LIO achieves accuracy comparable to FAST-LIO2 in campus scenes. However, in street 01, the narrow scene and intense on-stairs motions (as shown in Fig. 5(c)) result in failures or large drifts of other algorithms, but FMCW-LIO can achieve remarkably low errors. The reason is that, in street 01, the sensor suite faces a feature-less wall within an extremely narrow scene which leads to erroneous or invalid geometric observations (as in Fig. 5(d)). Nevertheless, FMCW-LIO can use the Doppleraided LiDAR velocity observation that is independent of the geometric features of environments. Meanwhile, the LiDAR velocity observation can perform the Zero-Velocity Update (ZUPT) when the platform is steady when facing the wall or offer z-axis velocity when a large motion in the z direction occurs. As a result, the estimated trajectory of FMCW-LIO can exactly return to the starting position.

In addition, due to the presence of numerous moving pedestrians and cars in street sequences, the variant (w/o DR)

TABLE II  
ABSOLUTE TRANSLATIONAL ERRORS (RMSE, METERS) AND END-TO-END ERRORS (E2E, METERS) ON THE COLLECTED DATASET
<table><tr><td rowspan="2">Method/Sequence</td><td colspan="2">campus_01</td><td colspan="2">campus_02</td><td colspan="2">street_01</td><td colspan="2">street_02</td><td colspan="2">tunnel_01</td><td colspan="2">tunnel_02</td><td colspan="2">tunnel_03</td><td colspan="2">tunnel_04</td></tr><tr><td>RMSE</td><td>E2E</td><td>RMSE</td><td>E2E</td><td>RMSE</td><td>E2E</td><td>RMSE</td><td>E2E</td><td>RMSE</td><td>E2E</td><td>RMSE</td><td>E2E</td><td>RMSE</td><td>E2E</td><td>RMSE</td><td>E2E</td></tr><tr><td>LINS</td><td>3.18</td><td>34.36</td><td>4.24</td><td>38.98</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LIO-SAM</td><td>1.24</td><td>4.29</td><td>4.89</td><td>3.16</td><td>X</td><td>X</td><td>0.23</td><td>0.18</td><td>X</td><td>X</td><td>2.11</td><td>0.10</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>DLIO</td><td>1.11</td><td>1.51</td><td>3.44</td><td>2.53</td><td>X</td><td>X</td><td>2.00</td><td>7.96</td><td>X</td><td>X</td><td>5.61</td><td>11.87</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FAST-LIO2</td><td>0.84</td><td>1.08</td><td>0.54</td><td>0.07</td><td>11.74</td><td>4.33</td><td>0.19</td><td>0.05</td><td>X</td><td>X</td><td>14.99</td><td>17.50</td><td>46.11</td><td>99.83</td><td>60.09</td><td>23.84</td></tr><tr><td>STEAM-DICP</td><td>1.23</td><td>1.49</td><td>1.76</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>35.16</td><td>X</td><td>4.39</td><td>X</td><td>0.81</td><td>X</td><td>49.32</td><td>X</td></tr><tr><td>FMCW-LIO (w/o DC)</td><td>0.11</td><td>0.14</td><td>0.87</td><td>0.31</td><td>1.14</td><td>0.23</td><td>0.20</td><td>0.08</td><td>0.96</td><td>0.52</td><td>1.94</td><td>1.65</td><td>0.59</td><td>0.32</td><td>2.04</td><td>0.86</td></tr><tr><td>FMCW-LIO (w/o DR)</td><td>0.13</td><td>0.13</td><td>0.16</td><td>0.09</td><td>2.73</td><td>0.72</td><td>1.20</td><td>0.35</td><td>0.48</td><td>0.20</td><td>0.12</td><td>0.06</td><td>0.28</td><td>0.05</td><td>0.11</td><td>0.06</td></tr><tr><td>FMCW-LIO</td><td>0.08</td><td>0.12</td><td>0.13</td><td>0.07</td><td>0.19</td><td>0.11</td><td>0.18</td><td>0.04</td><td>0.47</td><td>0.12</td><td>0.07</td><td>0.03</td><td>0.27</td><td>0.05</td><td>0.13</td><td>0.02</td></tr></table>

− As LINS is designed for spinning LiDARs and ground vehicles, its applicability on our LiDAR and platforms is highly challenging. × The algorithm failed or severely drifted in the corresponding sequence.

![](images/3f27bd45643802c4dc3f06d5b6aac63fc7ccbeb6573be704aefc6ab15ceacc8e.jpg)  
Fig. 6. Mapping in structured scenes. (a) Campus scene. (b) Street scene.

exhibits larger localization errors. Conversely, the complete FMCW-LIO and the variant (w/o DC) achieve accurate estimation performance. In structured sequences, the Doppler compensation effect is not significant due to the slow-moving speeds of both handheld and wheeled platforms operating within the campus or street blocks. Hence, the variant (w/o DC) achieves comparable accuracy with the complete FMCW-LIO. The mapping results of FMCW-LIO in structured scenes are shown in Fig. 6 and the average running time per scan of algorithms is listed in Table III. FMCW-LIO can achieve realtime and efficient performance that is comparable to that of FAST-LIO2. Online demonstrations can be found in the video.

Furthermore, we observe that STEAM-DICP demonstrates estimation accuracy comparable to other LIO methods in campus 01. However, in other sequences, it fails when encountering dynamic motions, fast rotations, or narrow scenes due to the lack of keypoints. In campus 02, STEAM-DICP can successfully run up to the halfway point, until rapid rotations occur. Thus, we only calculate the RMSE of the segments where it can successfully run without system failures. Therefore, it can be observed that IMU is essential for handheld or robotic platforms that frequently encounter narrow workspaces and highly dynamic motions.

## C. Experiments in Structure-Degenerated Environments

To further validate the Doppler-aided observation model, we evaluate algorithms on four tunnel sequences targeting geometric structure-degenerated scenes. These sequences have diverse distributions in length and moving speed. The results are shown in the right four columns of Table II. As expected, all conventional LIO systems that depend on geometric observations fail in most tunnel sequences. FMCW-LIO, however, accomplishes incredibly low errors, accurately returning to starting points. In the shortest sequence tunnel 02, LIO-SAM can achieve accurate localization with a factor graph framework which can maintain relatively long-term constraints in this short length. Although DLIO and FAST-LIO2 do not fail, they fail to achieve accurate localization, generating chaotic, inconsistent and drifted tunnel maps, as in Fig. 1(a)(c). In the longest sequence tunnel 04, FMCW-LIO can perfectly achieve precise localization (2 cm end-to-end error over 1.7 km traversed round-trip distance) through the whole tunnel. The Doppler-aided observation provides a direct observation of the LiDAR velocity state. However, in long tunnel sequences, the factor graph framework in LIO-SAM that only relies on geometric features also proves ineffective in handling incorrect correspondences. Finally, FMCW-LIO can generate a neat, complete, and consistent map of the whole tunnel (as shown in Fig. 1(d)), and the estimated tunnel length is highly close to the real data. Trajectories of algorithms overlaid with a satellite map are shown in Fig. 7. It can be found that FMCW-LIO outputs a precious trajectory accurately aligned with the tunnel ends while other algorithms suffer from severe drifts in such structure-degenerated scenes.

Similarly, STEAM-DICP can only achieve successful running up to the halfway point of each tunnel sequence. When the platform executes a U-turn to return, STEAM-DICP fails due to the narrow scene, fast rotations, and a scarcity of keypoints. Despite STEAM-DICP achieving a halfway success in tunnel sequences compared to conventional LIO methods, it still exhibits significant drifted errors, especially in the zaxis. We only compute the RMSE of the successful segments for STEAM-DICP. An insight from the results is that while the LiDAR-only continuous-time model of STEAM-DICP can partially cope with dynamic motions, it may encounter challenges or fail to handle the narrow workspaces and intense motion on handheld or robot platforms, without the assistance of exteroception-irrelevant IMU sensors.

In tunnel 02 and tunnel 04, due to high speeds (about 7.0 m/s) of the wheeled platform in the tunnel, the variant (w/o DC) exhibits a significant accuracy decrease which shows the effectiveness of the Doppler compensation in a lowfrequency scan pattern. Additionally, in tunnel 01 and tunnel 03, the presence of significant vibrations during walking on the handheld platform may lead to a further deterioration of IMU estimation performance when the geometric degeneration occurs, resulting in failures of other algorithms.

TABLE III  
AVERAGE RUNNING TIME (MILLISECONDS) PER LIDAR SCAN
<table><tr><td></td><td>LIO-SAM</td><td>DLIO</td><td>FAST-LIO2</td><td>STEAM-DICP</td><td>FMCW-LIO</td></tr><tr><td>Structured</td><td>40.08</td><td>45.93</td><td>33.53</td><td>178.25</td><td>39.71</td></tr><tr><td>Degenerated</td><td>28.76</td><td>36.00</td><td>20.54</td><td>85.46</td><td>21.33</td></tr></table>

![](images/1d6c079cd7ec42a6de531ca0859935bd386fb50848e74dc2c9a5a4cd52055bb6.jpg)  
Fig. 7. Estimated trajectories overlaid with the satellite map.

## D. Dynamic Point Removal and Static Mapping Results

As described in Section IV-D, FMCW-LIO is capable of removing dynamic points and generating static maps according to the Doppler residual and the updated LiDAR velocity. The method operates at the point level rather than the semantic or object level, resulting in direct and efficient performance. The results and comparisons of dynamic point detection and static mapping are shown in Fig. 8. It is obvious that, for conventional methods without removal, ghost points and residual blur of moving persons and buses may persist in the online map. Contrarily, FMCW-LIO can detect dynamic points (red points in Fig. 8(c)(g)) in the current scan and finally retain points that are nearly static in the map. As a result, ghost points and blur artifacts do not manifest in the map generated by FMCW-LIO. More online results can be found in the video.

## VI. CONCLUSION

In this letter, the Doppler-aided FMCW-LIO is presented for robotic state estimation and mapping. To achieve accurate and robust state estimation, an on-manifold IESEKF is formulated, fusing IMU and FMCW Doppler LiDAR measurements. The motion compensation for Doppler measurements is addressed by the IMU-driven backward propagation. FMCW-LIO utilizes the inherent Doppler-aided LiDAR velocity observation, enabling robust state estimation and static mapping, even in severely narrow or structure-degenerated scenes. Diverse experiments on multiple platforms validate the effectiveness and show the superior performance of FMCW-LIO.

![](images/950a6442687c4bebc56c38d8df128fc9266548929afa1c41892c5eb0a07c535e.jpg)  
Fig. 8. Dynamic point removal and static mapping. (a), (e) Moving persons and bus (orange ellipsoids). (b), (f) Ghost points (red ellipsoids) in the maps of FAST-LIO2. (c), (g) Online dynamic point detection (red points) of FMCW-LIO. (d), (h) Dynamic point removal (no ghost point in red ellipsoids) of FMCW-LIO.

## REFERENCES

[1] J. Anderson, R. Massaro, J. Curry, R. Reibel, J. Nelson, and J. Edwards, “Ladar: frequency-modulated, continuous wave laser detection and ranging,” Photogrammetric Engineering & Remote Sensing, vol. 83, no. 11, pp. 721–727, 2017.

[2] B. Hexsel, H. Vhavle, and Y. Chen, “DICP: Doppler Iterative Closest Point Algorithm,” in Proceedings ofRobotics: Science and Systems, New York City, NY, USA, June 2022.

[3] T. Shan, B. Englot, D. Meyers, W. Wang, C. Ratti, and D. Rus, “Lio-sam: Tightly-coupled lidar inertial odometry via smoothing and mapping,” in 2020 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2020, pp. 5135–5142.

[4] H. Ye, Y. Chen, and M. Liu, “Tightly coupled 3d lidar inertial odometry and mapping,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 3144–3150.

[5] C. Qin, H. Ye, C. E. Pranata, J. Han, S. Zhang, and M. Liu, “Lins: A lidar-inertial state estimator for robust and efficient navigation,” in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 8899–8906.

[6] W. Xu and F. Zhang, “Fast-lio: A fast, robust lidar-inertial odometry package by tightly-coupled iterated kalman filter,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 3317–3324, 2021.

[7] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, “Fast-lio2: Fast direct lidarinertial odometry,” IEEE Transactions on Robotics, vol. 38, no. 4, pp. 2053–2073, 2022.

[8] C. Bai, T. Xiao, Y. Chen, H. Wang, F. Zhang, and X. Gao, “Faster-lio: Lightweight tightly coupled lidar-inertial odometry using parallel sparse incremental voxels,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4861–4868, 2022.

[9] D. He, W. Xu, N. Chen, F. Kong, C. Yuan, and F. Zhang, “Point-lio: Robust high-bandwidth light detection and ranging inertial odometry,” Advanced Intelligent Systems, p. 2200459, 2023.

[10] K. Chen, R. Nemiroff, and B. T. Lopez, “Direct lidar-inertial odometry: Lightweight lio with continuous-time motion correction,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 3983–3989.

[11] M. Guo, K. Zhong, and X. Wang, “Doppler velocity-based algorithm for clustering and velocity estimation of moving objects,” in 2022 7th International Conference on Automation, Control and Robotics Engineering (CACRE). IEEE, 2022, pp. 216–222.

[12] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “Carla: An open urban driving simulator,” in Conference on robot learning. PMLR, 2017, pp. 1–16.

[13] Y. Gu, H. Cheng, K. Wang, D. Dou, C. Xu, and H. Kong, “Learning moving-object tracking with fmcw lidar,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 3747–3753.

[14] Y. Wu, D. J. Yoon, K. Burnett, S. Kammel, Y. Chen, H. Vhavle, and T. D. Barfoot, “Picking up speed: Continuous-time lidar-only odometry using doppler velocity measurements,” IEEE Robotics and Automation Letters, vol. 8, no. 1, pp. 264–271, 2022.

[15] D. J. Yoon, K. Burnett, J. Laconte, Y. Chen, H. Vhavle, S. Kammel, J. Reuther, and T. D. Barfoot, “Need for speed: Fast correspondencefree lidar-inertial odometry using doppler velocity,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 5304–5310.

[16] M. Yoneda, K.-M. Dahlen, and T. Ogawa, “Extended object tracking´ with doppler velocity-based point registration,” in 2023 IEEE Symposium Sensor Data Fusion and International Conference on Multisensor Fusion and Integration (SDF-MFI). IEEE, 2023, pp. 1–8.

[17] M. Jung, W. Yang, D. Lee, H. Gil, G. Kim, and A. Kim, “Helipr: Heterogeneous lidar dataset for inter-lidar place recognition under spatial and temporal variations,” 2023.

[18] C. Doer and G. F. Trommer, “An ekf based approach to radar inertial odometry,” in 2020 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI). IEEE, 2020, pp. 152–159.

[19] J. Michalczyk, R. Jung, and S. Weiss, “Tightly-coupled ekf-based radar-inertial odometry,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 12 336–12 343.

[20] Y. Almalioglu, M. Turan, C. X. Lu, N. Trigoni, and A. Markham, “Millirio: Ego-motion estimation with low-cost millimetre-wave radar,” IEEE Sensors Journal, vol. 21, no. 3, pp. 3314–3323, 2020.

[21] Y. Z. Ng, B. Choi, R. Tan, and L. Heng, “Continuous-time radar-inertial odometry for automotive radars,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 323–330.

[22] Y. Zhuang, B. Wang, J. Huai, and M. Li, “4d iriom: 4d imaging radar inertial odometry and mapping,” IEEE Robotics and Automation Letters, 2023.

[23] D. He, W. Xu, and F. Zhang, “Kalman filters on differentiable manifolds,” arXiv preprint arXiv:2102.03804, 2021.

[24] N. El-Sheimy, H. Hou, and X. Niu, “Analysis and modeling of inertial sensors using allan variance,” IEEE Transactions on instrumentation and measurement, vol. 57, no. 1, pp. 140–149, 2007.

[25] J. Zhang and S. Singh, “Loam: Lidar odometry and mapping in realtime.” in Robotics: Science and systems, vol. 2, no. 9. Berkeley, CA, 2014, pp. 1–9.

[26] Aeva Inc. Aeries II, Accessed Aug. 8, 2023. [Online]. Available: https://www.aeva.com/aeries-ii/

[27] Y. Wang, Z. Sun, C.-Z. Xu, S. E. Sarma, J. Yang, and H. Kong, “Lidar iris for loop-closure detection,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020, pp. 5769–5775.

[28] F. Dellaert, “Factor graphs and gtsam: A hands-on introduction,” Georgia Institute of Technology, Tech. Rep, vol. 2, p. 4, 2012.

## SUPPLEMENTARY FRAMEWORK DOCUMENTATION

FMCW-LIO is a unified and extensible LiDAR-inertial odometry framework that supports both 4D Doppler LiDARs and conventional 3D LiDARs. Built on a modular architecture design, the key framework components can be configured, extended, or replaced independently. Beyond the Doppler-aided estimation, FMCW-LIO also offers the following features:

• Doppler Velocity Compensation: A Doppler compensation scheme aligns Doppler velocity measurements into a unified spatiotemporal frame. Beyond motion compensation, Doppler compensation essentially reveals the interrelation between the Doppler velocities of a static point with respect to arbitrary different frames.

• Flexible Initialization: Multiple initialization algorithm and strategies are supported, including conventional stationary initialization and Free-Init, which delivers accurate initial states even in the presence of aggressive motion, moving objects, and structure degeneracy.

• Multiple Integration and Multiple Motion Compensation Schemes: Various integration and motion compensation methods are provided, including second-order Runge-Kutta method (RK2), third-order Runge-Kutta method (RK3), fourth-order Runge-Kutta method (RK4), mechanization, and ACI2.

• Robust Velocity Estimation: Multiple algorithms are integrated to improve the resilience and accuracy of velocity estimation, including least squares with RANSAC, covariance, and robust kernels.

• Multiple Map Structures: Different map structures are supported, including ikd-Tree and iVox, enabling flexible and efficient mapping under varying conditions.

• Multiple LiDAR Types: The framework is compatible with both 4D Doppler LiDARs and conventional 3D LiDARs, facilitating evaluation and deployment across different LiDAR sensing configurations.

FMCW-LIO is tested on numerous typical LiDAR-inertial datasets covering different LiDAR sensors, platform types, motion patterns, and environmental conditions, including:

1) Boreas-RT Dataset<sup>1</sup>;

2) HeRCULES Dataset<sup>2</sup>;

3) Newer College Dataset<sup>3</sup>;

4) Oxford Spires Dataset<sup>4</sup>;

5) Multi-Campus Dataset (MCD)<sup>5</sup>;

6) NTU VIRAL Dataset<sup>6</sup>;

7) Hilti SLAM Challenge Dataset 2022<sup>7</sup>;

8) UrbanNav Dataset<sup>8</sup>;

9) FAST-LIO2 Dataset<sup>9</sup>;

10) FAST-LIVO2 Dataset<sup>10</sup>.

Extensive test experiments are performed to validate the framework across heterogeneous sensing configurations and environmental conditions. FMCW-LIO demonstrates consistently robust and accurate odometry and mapping results with extremely efficient performance, within both well-structured and structure-degenerated scenes. Representative mapping results on selected sequences are presented in Fig. 9.

In addition, the typical 4D FMCW Doppler LiDAR-inertial sequences collected in this work are also released as the FMCW-LIO Dataset and the Free-Init Dataset. FMCW-LIO Dataset provides five handheld sequences across campus, street, and tunnel scenes, covering both well-structured and structure-degenerated environments. Free-Init Dataset provides four handheld and vehicular sequences recorded under dynamic initial motions for the qualitative evaluation of dynamic initialization. Characteristics and download links of the sequences are listed and detailed on the respective project pages.

![](images/9c573a0eb43a92dd0eee43f536ecbcf6070c7aa34dd772ea22edbb38d82c968c.jpg)  
Fig. 9. Representative mapping results of FMCW-LIO on different typical datasets. (a), (b) Boreas-RT Dataset. (c) HeRCULES Dataset. (e), (f) Newer College Dataset. (d) FMCW-LIO Dataset. (g)–(j) MCD. (k) FAST-LIO2 Dataset. (l) FAST-LIVO2 Dataset.