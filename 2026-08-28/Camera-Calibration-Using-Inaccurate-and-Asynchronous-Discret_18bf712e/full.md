# Camera Calibration Using Inaccurate and Asynchronous Discrete GPS Trajectory from Drones

R. Yang

Y. Bar-Shalom

H.A.J. Huang

August 28, 2026

Abstract—This paper considers a stationary camera calibration problem, which estimates the camera orientation angles yaw, pitch and roll, using a drone trajectory recorded by a GPS. There are three challenges in using a GPS trajectory as ground truth for camera calibration. One, the altitude of GPS data is inaccurate with an unknown bias. Two, the GPS receiver and camera are not time synchronized, and there is an unknown time ofset between the two systems. Three, the GPS trajectory is time-discrete and accurate interpolation is needed. This is actually an estimation problem since velocity is also needed. To address the first two challenges, we formulate the problem as a parameter estimation problem to estimate a vector consisting of the GPS altitude bias and time ofset in addition to the camera yaw, pitch and roll biases. We then develop a special maximum likelihood estimator using the Iterated Least Squares algorithm which can work with a non-synchronized time-discrete GPS trajectory for the third challenge. Since the camera measurement errors are usually small, this requires a high calibration accuracy so that the residual bias error following the calibration should not be significant compared to the measurement error standard deviation. The calibration accuracy depends highly on the drone<sup>[</sup> trajectory. This paper also recommends an appropriate drone trajectory which can yield a good calibration accuracy, namely, 14% of the measurement error standard deviation. Simulation tests are conducted to demonstrate the algorithm performance. The estimation results meet the Cramer-Rao Lower Bound (CRLB) since the Normalized Estimation Error Squared w.r.t. the CRLB is statistically acceptable.

Index Terms—Camera calibration, inaccurate GPS altitude, time ofset, maximum likelihood estimator.

## I. Introduction

This paper presents a camera calibration approach for a stationary camera which looks at air targets. We assume the camera is of “pinhole” type without radial and tangential distortion. The position of the camera is assumed known. The calibration computes the camera orientation, which is defined by three rotation angles, yaw, pitch and roll. Since the camera is looking for air targets, there is no fixed object with known position in its Field of View (FOV). The calibration is based on the trajectory of a drone instrumented with a GPS receiver, which, however, usually has a significant altitude bias error<sup>1</sup>. Also the camera and GPS receiver are not time synchronized. This introduces an unknown fixed time ofset between the GPS and camera time-stamps. Furthermore, the drone trajectory is a sequence of discrete points with a certain time interval and there is no analytical expression for the trajectory. This paper will develop a practical approach for the problem of estimating the camera orientation, the GPS altitude bias and the time ofset.

Camera calibration is not a new problem. Numerous works have been done before, and they can be categorized into two areas: computer vision related applications and estimation theory based approaches. The camera calibration in computer vision is developed from the Perspectiven-Point (PnP) problem [4][13]. The original PnP problem is described as follows: Given the relative spatial locations of n control points $P _ { i }$ with $i = 1 , \ldots , n .$ , and given the angle to every pair of these points from an additional point, called the center of perspective C, find the lengths of the line segments joining C to each of the control points. The camera calibration is based on the matching of n 3D control points and their corresponding points on the 2D image space. They share the same angles of arrival with reference to the camera center of perspective C. A number of solutions have been developed with this approach [4][10][7][8][13]. Some focused on the solution of the minimum number of control points required (n = 3) as P3P problem [10][7][8][13], and some deal with a large number of points consisting of outliers and inaccurate points. The RANSAC [4] scheme can be applied to select good samples. Some extensions on the camera calibration take unknown focal length and radial distortion into consideration [9][21].

If we apply the PnP approach to our problem, a 3D GPS-instrumented drone trajectory needs to match the camera-measured 2D trajectory. Since there is an unknown time ofset between GPS-based 3D and camera 2D trajectories and an unknown altitude bias on the 3D trajectory, it is not practical to apply point-to-point 3D-

2D matching. We then seek a solution from the estimation theory.

Unlike the computer vision approach, which devel ops the camera calibration as a particular geometric problem, the estimation theory approach formulates the camera calibration as a parameter estimation problem with stochastic models. It defines the unknown parameter to be estimated as θ, and builds a relationship between θ and measurements that include noise. If the problem is observable (i.e., with a unique solution), optimization algorithms, such as Gradient Descent, Newton’s algorithm or Iterated Least Squares (ILS) [1], can be applied in a systematic manner. A number of works along this line have been carried out to estimate the sensor position/orientation and measurement biases. This is often referred to as the sensor registration problem. It can be solved ofline using either ILS or Maximum Likelihood (ML) estimator from a batch of data [3][22][23][19][5][27], or online (estimating the sensor biases and target trajectories simultaneously) using a Kalman Filter (KF) type dynamic estimator or Recursive Least Squares (RLS) approach [2][17][25][24]. The online approaches (also referred as to auto-calibration) sound more attractive. However, they estimate a large augmented state consisting of all target states and sensor biases. This large state may create computational infeasibility for real time when the number of targets is large. Furthermore, sensor bias estimation accuracy is not always guaranteed as arbitrary target trajectories do not reduce the bias error compared to a dedicated special trajectory. The calibration accuracy (or sensor bias estimation accuracy) is paramount in our problem as camera orientation must be accurately estimated so that the residual bias error should not be significant compared to the camera measurement error. We therefore prefer an ofline approach which allows a GPS-equipped drone to fly in a special predefined path dedicated to accurate camera calibration. Such a path will be discussed in the sequel. The previous work on ofline sensor registration mainly dealt with radar pose and measurement biases [22][23][19][5]. Camera calibration was conducted in [3][27]. The yaw, pitch and roll biases of multiple cameras and target locations are estimated simul taneously using the ILS method in [3] for a satellite-based camera observing an exoatmospheric target of opportunity. In [27], a camera was calibrated through observing a planar pattern shown at several diferent orientations, and camera intrinsic and extrinsic parameters were estimated using a closed-form solution. Neither of them deals with unknown time ofset among diferent systems, for example, sensor and ground truth systems — the diferent sensors are assumed time synchronized.

Online and ofline calibration with an unknown time ofset have been discussed in various applications [14][11][18][6][15][16][20]. We focus on the ofline solutions [11][18][6][20]. In [11][18] the time ofset and spatial calibration were conducted separately in sequence. The time ofset was estimated first, and then spatial calibration was conducted. A more robust approach [6][20] estimated the time ofset and spatial biases simultaneously. This was a robotics application with camera, Inertial Measurement Unit (IMU) and laser rangefinder. It estimated the time ofset among sensors and measurement transformation. However, the camera was assumed well calibrated. The Levenberg-Marquardt (LM) algorithm was used to minimize an objective function based on the maximum likelihood criterion, using stationary objects detected by the camera and rangefinder on a moving platform. Although the approach included unknown time ofsets into its estimation parameter, camera calibration was not conducted.

In this paper, we develop our approach based on estimation theory which will include the GPS altitude bias and time ofset in the estimation of the parameter vector θ. Another challenge is that the GPS 3D trajectory is given in numerical form. The preliminary version of the present study, [26], conducted calibration assuming an accurate GPS without altitude bias. An ILS algorithm was developed to perform calibration based on a stochastic model dealing with a GPS trajectory expressed by a sequence of discrete-time points. In the present paper, inaccurate GPS with unknown altitude bias is used. The calibration accuracy drops significantly with this additional unknown unless it is part of the estimated parameter vector. If the residual bias error (following the calibration) is not small enough compared to the camera measurement error, the calibration is not meaningful. We will develop an enhanced ILS algorithm to improve the estimation accuracy, and recommend a practical drone path to achieve good calibration accuracy.

The rest of paper is structured as follows. Section II describes the three coordinate systems used in this paper. Section III describes the problem formulation, namely, the stochastic model for estimation. Section IV presents the estimation algorithm based on the stochastic model dealing with numeric GPS trajectories. Section V presents simulation results on calibration error, and recommends a suitable drone path. Section VI draws the conclusions.

## II. Coordinate systems

The following three coordinate systems are used in this paper:

Common coordinate system with x-y-z as East, North and Up (ENU).

Camera coordinate system with $x ^ { \mathrm { C } } - y ^ { \mathrm { C } } - z ^ { \mathrm { C } }$ centered at the camera position $( x ^ { \mathrm { s } } , y ^ { \mathrm { s } } , z ^ { \mathrm { s } } )$ , shown in Fig. 1.

Image coordinate system with $x ^ { \mathrm { I } } - y ^ { \mathrm { I } }$ shown in Fig. 1. The notations used in the paper are listed in Table I. The conversion of x to $\mathbf { x } ^ { \mathrm { { C } } }$ is given by

$$
\begin{array} { l c l } { \mathbf { x } ^ { \mathrm { { C } } } } & { = } & { { \bf { T } } ( \alpha , \epsilon , \rho ) ( { \bf { x } } - { \bf { x } } ^ { \mathrm { { s } } } ) } \\ & { = } & { { \bf { T } } ^ { z } ( \rho ) { \bf { T } } ^ { x } ( \epsilon - 9 0 ^ { \mathrm { { o } } } ) { \bf { T } } ^ { z } ( - \alpha ) ( { \bf { x } } - { \bf { x } } ^ { \mathrm { { s } } } ) } \end{array}\tag{1}
$$

where we use the following mnemonic notations for rotations between 3D Cartesian systems:

![](images/de02e816350f09ae0f4bcb1706e32faaa0f7e296b2640d121e8c01ba78dd1bc3.jpg)  
Fig. 1. Camera and image coordinate systems.

<table><tr><td>X  $[ x \ y \ z ] ^ { \prime } , \mathrm { a }$   $\mathbf { x } ^ { \mathrm { { C } } }$   $[ x ^ { \mathrm { C } } \ y ^ { \mathrm { C } } \ z ^ { \mathrm { C } } ] ^ { \prime }$   $\mathbf { x } ^ { \mathrm { { I } } }$   $[ x ^ { \mathrm { I } } y ^ { \mathrm { I } } ] ^ { \prime }$   $\mathbf { x } ^ { \mathrm { { s } } }$  [xs α €  $\rho$ </td><td>point in the common (ENU) coordinate system. , a point in the camera coordinate system. , a point in the image coordinate system.  $\bar { y } ^ { \mathrm { s } } z ^ { \mathrm { s } } ] ^ { \prime }$  sensor (camera) position. camera pointing azimuth or yaw (clockwise from N). camera pointing elevation or pitch (up from horizontal). camera roll (ideally zero), clockwise around the center of the Focal Plane Array (FPA). GPS altitude bias. The GPS provided altitude is higher thar the true value when h is positive, otherwise, h is negative. time offset between the drone GPS and the camera. The GP</td></tr></table>

$$
\mathbf { T } ^ { x } ( \phi ) = \left[ \begin{array} { c c c } { 1 } & { 0 } & { 0 } \\ { 0 } & { \cos \phi } & { \sin \phi } \\ { 0 } & { - \sin \phi } & { \cos \phi } \end{array} \right]\tag{2}
$$

for a rotation around the x-axis by $\phi$ from y toward $z ,$

$$
\begin{array} { r } { \mathbf { T } ^ { z } ( \phi ) = \left[ \begin{array} { c c c } { \cos \phi } & { \sin \phi } & { 0 } \\ { - \sin \phi } & { \cos \phi } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} \right] } \end{array}\tag{3}
$$

for a rotation around the z-axis by ϕ from x toward y. The rotation around the y-axis is not necessary as $\mathbf { T } ^ { x } ( 9 0 ^ { \mathrm { o } } - \epsilon )$ replaces the y-axis by the z-axis, so that rotation around the z-axis occurs twice. The combined rotation in (1) is

$$
\mathbf { T } ( \alpha , \epsilon , \rho ) = \left[ \begin{array} { c c c } { c _ { \alpha } c _ { \rho } + s _ { \alpha } s _ { \epsilon } s _ { \rho } } & { c _ { \alpha } s _ { \epsilon } s _ { \rho } - s _ { \alpha } c _ { \rho } } & { - c _ { \epsilon } s _ { \rho } } \\ { s _ { \alpha } s _ { \epsilon } c _ { \rho } - c _ { \alpha } s _ { \rho } } & { s _ { \alpha } s _ { \rho } + c _ { \alpha } s _ { \epsilon } c _ { \rho } } & { - c _ { \epsilon } c _ { \rho } } \\ { s _ { \alpha } c _ { \epsilon } } & { c _ { \alpha } c _ { \epsilon } } & { s _ { \epsilon } } \end{array} \right]
$$

where

(4)

$$
s _ { \alpha } = \sin \alpha , s _ { \epsilon } = \sin \epsilon , s _ { \rho } = \sin \rho\tag{5}
$$

$$
c _ { \alpha } = \cos \alpha , ~ c _ { \epsilon } = \cos \epsilon , ~ c _ { \rho } = \cos \rho\tag{6}
$$

The conversion of $\mathbf { x } ^ { \mathrm { { C } } }$ to $\mathbf { x } ^ { \mathrm { { I } } }$ is

$$
\mathbf { x } ^ { \mathrm { I } } = \mathbf { f } ( \mathbf { x } ^ { \mathrm { C } } ) = \left[ \begin{array} { c } { \frac { P _ { \mathrm { x } } } { 2 } + \frac { x ^ { \mathrm { C } } f } { z ^ { \mathrm { C } } } } \\ { \frac { P _ { \mathrm { y } } } { 2 } + \frac { y ^ { \mathrm { C } } f } { z ^ { \mathrm { C } } } } \end{array} \right]\tag{7}
$$

where $f$ is the focal length with units of pixel (assumed square)

$$
f = \frac { P _ { \mathrm { x } } } { 2 \tan ( \Theta _ { \mathrm { x } } / 2 ) } = \frac { P _ { \mathrm { y } } } { 2 \tan ( \Theta _ { \mathrm { y } } / 2 ) }\tag{8}
$$

and $P _ { \mathrm { { x } } }$ and $P _ { \mathrm { y } }$ are the numbers of pixels in $x ^ { \mathrm { I } }$ and $y ^ { \mathrm { I } }$ coordinates, respectively; $\Theta _ { \mathbf { x } }$ and $\Theta _ { \mathrm { y } }$ are the fields of view (FOV) — angular spans — in $x ^ { \mathrm { I } }$ and $y ^ { \mathrm { I } }$ , respectively.

## III. Problem Formulation

This section formulates the estimation problem in a stochastic model. The parameter to estimate is

$$
\theta = [ \alpha \ \epsilon \ \rho \ \hbar \ \tau ] ^ { \prime }\tag{9}
$$

which consists of three camera orientation angles α, ϵ and $\rho ,$ GPS altitude bias $\hbar ,$ and the time ofset between the drone GPS and camera systems $\tau ,$ which are estimated simultaneously. The stochastic model for estimating θ is

$$
\mathbf { Z } = \mathbf { H } ( \theta , \mathbf { X } ) + \mathbf { w }\tag{10}
$$

where $\mathbf { H } ( \cdot )$ is defined in (17), Z is the camera measurement vector consisting of n discrete-time points in the image coordinates as

$$
\begin{array} { r c l } { \mathbf { Z } } & { = } & { [ \mathbf { z } ( t _ { 1 } ) ^ { \prime } \dots \mathbf { z } ( t _ { n } ) ^ { \prime } ] ^ { \prime } } \\ & { = } & { [ \mathbf { x } ^ { \mathrm { I } } ( t _ { 1 } ) ^ { \prime } \dots \mathbf { x } ^ { \mathrm { I } } ( t _ { n } ) ^ { \prime } ] ^ { \prime } + \mathbf { w } } \end{array}\tag{11}
$$

with measurement times $t _ { 1 } , \ldots , t _ { n } ,$ w is a 2n zero-mean Gaussian measurement noise vector with covariance

$$
{ \bf R } = { \bf I } _ { 2 n \times 2 n } \sigma _ { \mathrm { F } } ^ { 2 }\tag{12}
$$

and $\sigma _ { \mathrm { F } } ^ { 2 }$ is the variance of the measurement noise in the FPA. For details of how this is obtained based on the optics’ Point Spread Function (PSF) and pixel size, see [12]. X is the GPS drone trajectory (with unknown altitude bias and time ofset) represented by a set of discrete-time points in the common coordinate system (ENU) at times corresponding to the camera measurement times, corrected by the (unknown) time ofset. It is defined as

![](images/02a786996004efcc7494e89328a7a19bd372f00789ab03c654b313f405208738.jpg)  
• discrete-time points on true trajectory ---- X

o provided discrete-time points on GPS trajectory ---- X

▲ corresponding points (toX) on GPS trajectory ---- X

Fig. 2. True and GPS trajectories.

$$
\mathbf { X } = [ \mathbf { x } ( t _ { 1 } + \tau ) ^ { \prime } \ \dots \ \mathbf { x } ( t _ { n } + \tau ) ^ { \prime } ] ^ { \prime }\tag{13}
$$

However, X is not known exactly. The available information on the GPS trajectory is

$$
\overline { { \mathbf { X } } } = [ \mathbf { x } ( \bar { t } _ { 1 } ) ^ { \prime } \ \dots \ \mathbf { x } ( \bar { t } _ { m } ) ^ { \prime } ] ^ { \prime }\tag{14}
$$

where $\bar { t } _ { 1 } , \ldots , \bar { t } _ { m }$ do not correspond to the times in $\mathbf { X } ,$ and X and X intervals can difer. We need to find the relationship between X and X, so that the model in (10) can be utilized for estimation. This will be solved in next section.

Fig. 2 shows the relationship of the true trajectory X<sup>˘</sup> and GPS trajectory of the drone, where

$$
\breve { { \bf X } } = [ \breve { { \bf x } } ( t _ { 1 } ) ^ { \prime } ~ . ~ . ~ \breve { { \bf x } } ( t _ { n } ) ^ { \prime } ] ^ { \prime }\tag{15}
$$

Each discrete-time point (•) on the true trajectory has a corresponding point (▲) on the GPS trajectory. The relationship of the ith points of X<sup>˘</sup> and X is

$$
\Breve { \mathbf { x } } ( t _ { i } ) = \mathbf { x } ( t _ { i } + \tau ) - [ 0 \mathrm { ~ } 0 ~ \hbar ] ^ { \prime }\tag{16}
$$

The measurement function H in (10) is then

$$
\begin{array} { r l r } { \mathbf { H } } & { = } & { \left[ \begin{array} { c } { \mathbf { h } _ { 1 } [ \alpha , \epsilon , \rho , \Breve { \mathbf { x } } ( t _ { 1 } ) ] } \\ { \vdots } \\ { \mathbf { h } _ { n } [ \alpha , \epsilon , \rho , \Breve { \mathbf { x } } ( t _ { n } ) ] } \end{array} \right] } \end{array}\tag{17}
$$

with

$$
\begin{array} { r l } & { \mathbf { h } _ { i } ( \cdot ) = \mathbf { f } \left\{ \mathbf { T } ( \alpha , \epsilon , \rho ) [ \overline { { \mathbf { x } } } ( t _ { i } + \tau ) - [ 0 \mathrm { ~ } 0 \mathrm { ~ } \hbar ] ^ { \prime } - \mathbf { x } ^ { \mathrm { s } } ] \right\} } \\ & { \quad \quad \quad = \mathbf { f } ( \mathbf { x } _ { i } ^ { \mathrm { C } } ) = \mathbf { x } _ { i } ^ { \mathrm { I } } } \end{array}\tag{18}
$$

The above converts a position $\mathbf { \ddot { \rho } } \mathbf { A } ^ { \mathsf { \prime } } \mathbf { \vec { x } } _ { i }$ to a position $\cdots$ $\breve { \mathbf { x } } _ { i } ,$ then converts to camera coordinates as $\mathbf { x } _ { i } ^ { \mathrm { { C } } }$ using (1), and finally converts to the image space as ${ \bf x } _ { i } ^ { \mathrm { I } }$ using (7).

## IV. Estimation Algorithm

This section solves the problem described in Section III using a unique Iterated Least Squares (ILS) algorithm which is illustrated in Fig. 3. Its uniqueness lies in the fact that Z and X are used to estimate X and X<sup>˙</sup> and then Z and X are used in the iterative estimation of $\theta ,$ defined in (9).

Given the camera measurement Z, the GPS trajectory X and the initial value of the parameter $\widehat { \theta } _ { 0 }$ , the algorithm finds $\hat { \theta }$ through iteration, indexed by j, based on the nonlinear model given in (10). We will describe the algorithm with the following three steps:

(A) Estimation of $\mathbf { X } _ { j }$ and its velocity ${ \dot { \bf X } } _ { j }$ from $\overline { { \mathbf { X } } }$ and $\widehat { \theta } _ { j }$ in the jth iteration. $\mathbf { X } _ { j }$ and $\dot { \mathbf { X } } _ { j }$ are needed in the θ estimation in (B);

(B) Updating of ${ \widehat { \theta } } _ { j }$ to $\hat { \theta } _ { j + 1 }$ using an optimization algorithm based on the model (10) with estimated $\mathbf { X } _ { j }$ and ${ \dot { \bf X } } _ { j } ;$

(C) Stop the iteration when a satisfactory $\hat { \theta }$ is obtained.

![](images/35d881195b127f194d597eb9bb3a3e71f92135430ac4e9f6aac06e69fa331a56.jpg)  
Fig. 3. The ILS estimation algorithm.

## A. Estimate X and its Velocities $\dot { \bf X }$

To estimate θ we need to know X and its velocities X<sup>˙</sup> from the positions $\overline { { \mathbf { X } } }$ (namely, to estimate the GPS trajectory $" \Delta ^ { \prime \prime }$ points from $^ { 6 6 } \mathrm { O } ^ { 9 9 }$ points in Fig. 2), so that the discrete-time points on the GPS trajectory are at times $[ t _ { 1 } + \tau , . . . , t _ { n } + \tau ]$ corresponding to the camera measurements at times ${ \bar { [ t _ { 1 } , \dots , t _ { n } ] } } . ^ { 2 }$ The Least Squares (LS) fitting algorithm developed in [26] used a sliding window containing the neighbouring $^ { 6 6 } \circ ^ { 9 9 }$ points before and after a particular ${ \bf \ddot { \boldsymbol { \omega } } } _ { \bf A } , { \bf \ddot { \boldsymbol { \omega } } }$ to estimate its position and velocity. However, this will not perform well when a maneuver happens within the window. We therefore enhance it as a two-step LS fitting approach in this paper. Fig. 4 illustrates the two steps. We can see the one-step LS approach in (a) has a large error when there is a maneuver. The two-step LS fitting approach shown in (b) uses two LS estimators on the neighbouring points before and after the $" \pmb { \Delta } ^ { \prime \prime }$ . They obtain two estimates “b” and $\mathrm { ^ { 6 6 } a } ^ { \prime \prime }$ , respectively. The final estimate $\mathrm { ^ { 6 } c } ^ { \mathrm { 7 } }$ is a combination of “b” and $\mathrm { ^ { 6 } a \mathrm { ^ { 9 } } }$ The estimation error of the two-step LS fitting is therefore reduced significantly.

In the two-step LS fitting approach, we illustrate LS1 (applied to the neighbors before the $\mathbf { \ddot { \delta } } \mathbf { \Delta } ^ {  } \mathbf { \Phi } ^ {  } \mathbf { \Phi } ^ {  } \mathbf { \Phi } ^ {  } )$ to obtain point $\ddot { \mathfrak { s o } } ^ { , \vec { \mathfrak { n } } }$ in detail in the following. The estimation of point $\mathrm { ^ { 6 6 } a } ^ { \mathrm { 7 } }$ is similar. First of all, we estimate the velocities and accelerations of the nearest $^ { 6 6 } \mathrm { O } ^ { 9 9 }$ [assuming $\mathbf { x } ( \bar { t } _ { i } ) ]$ before the $^ { 6 6 } { \pmb { \triangle } } ^ { \prime 5 } .$ . Its velocity and acceleration are

$$
\begin{array} { l l l } { \dot { { \bf x } } ( \bar { t } _ { i } ) } & { = } & { [ \dot { x } ( \bar { t } _ { i } ) \dot { y } ( \bar { t } _ { i } ) \dot { z } ( \bar { t } _ { i } ) ] ^ { \prime } } \end{array}\tag{19}
$$

$$
\begin{array} { l l l } { \ddot { { \bf x } } ( \bar { t } _ { i } ) } & { = } & { [ \ddot { x } ( \bar { t } _ { i } ) \ddot { y } ( \bar { t } _ { i } ) \ddot { z } ( \bar { t } _ { i } ) ] ^ { \prime } } \end{array}\tag{20}
$$

The vectors consisting of velocities and accelerations in

![](images/5ccb2c85b699b68dec1823178f7a21fe4892b841b55cb1409ec6c70be68e4df6.jpg)  
Fig. 4. The LS fitting algorithms. (a) One-step LS fitting approach developed in [26]. (b) Two-step LS fitting approach used in the present paper.

x,y and z coordinates are defined as

$$
\begin{array} { r l r } { { \bf d } _ { x } ^ { i } } & { { } = } & { [ \dot { x } ( \bar { t } _ { i } ) \ddot { x } ( \bar { t } _ { i } ) ] ^ { \prime } } \end{array}\tag{21}
$$

$$
\begin{array} { r c l } { \mathbf { d } _ { y } ^ { i } } & { = } & { [ \dot { y } ( \bar { t } _ { i } ) \ddot { y } ( \bar { t } _ { i } ) ] ^ { \prime } } \end{array}\tag{22}
$$

$$
\begin{array} { r c l } { { { \bf d } _ { z } ^ { i } } } & { { = } } & { { [ \dot { z } ( \bar { t } _ { i } ) \ddot { z } ( \bar { t } _ { i } ) ] ^ { \prime } } } \end{array}\tag{23}
$$

They are estimated separately. The model to estimate $\mathbf { d } _ { x } ^ { i }$ from its neighbors is

$$
\Delta _ { x } ^ { i } = \mathbf { D } ^ { i } \mathbf { d } _ { x } ^ { i }\tag{24}
$$

where

$$
\begin{array} { r l r } { \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } } & { = } & { \left[ \begin{array} { c c } { x ( \bar { t } _ { i - \delta } ) - x ( \bar { t } _ { i } ) } \\ { \vdots } \\ { x ( \bar { t } _ { i - 1 } ) - x ( \bar { t } _ { i } ) } \end{array} \right] } \\ { \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Sigma } \mathbf { \Delta } } & { = } & { \left[ \begin{array} { c c } { \overline { { T } } _ { i - \delta } } & { - 0 . 5 \overline { { T } } _ { i - \delta } ^ { 2 } } \\ { \vdots } & { \vdots } \\ { \overline { { T } } _ { i - 1 } } & { - 0 . 5 \overline { { T } } _ { i - 1 } ^ { 2 } } \end{array} \right] } \end{array}\tag{25}
$$

(26)

and

$$
\begin{array} { r c l } { { { \overline { { T } } } _ { i - k } } } & { { = } } & { { { \bar { t } } _ { i - k } - { \bar { t } } _ { i } } } \\ { { } } & { { } } & { { k = [ \delta , \ldots , 1 ] } } \end{array}\tag{27}
$$

The number of neighboring points used in (24) is δ=3. LS is applied to estimate $\mathbf { d } _ { x } ^ { i }$ as

$$
\hat { \mathbf { d } } _ { x } ^ { i } = [ ( \mathbf { D } ^ { i } ) ^ { \prime } \mathbf { D } ^ { i } ] ^ { - 1 } ( \mathbf { D } ^ { i } ) ^ { \prime } \Delta _ { x } ^ { i }\tag{28}
$$

and $\mathbf { d } _ { y } ^ { i }$ and $\mathbf { d } _ { z } ^ { i }$ are estimated similarly as

$$
\begin{array} { r l r } { \hat { { \bf d } } _ { y } ^ { i } } & { = } & { [ ( { \bf D } ^ { i } ) ^ { \prime } { \bf D } ^ { i } ] ^ { - 1 } ( { \bf D } ^ { i } ) ^ { \prime } { \pmb { \Delta } } _ { y } ^ { i } } \end{array}\tag{29}
$$

$$
\begin{array} { r l r } { \hat { \mathbf { d } } _ { z } ^ { i } } & { { } = } & { [ ( \mathbf { D } ^ { i } ) ^ { \prime } \mathbf { D } ^ { i } ] ^ { - 1 } ( \mathbf { D } ^ { i } ) ^ { \prime } \Delta _ { z } ^ { i } } \end{array}\tag{30}
$$

Next, we compute positions and velocities of $" \Delta ^ { \prime \prime }$ , namely, $^ { 6 6 }$ point in Fig. 4(b). We assume the $" \pmb { \Delta } ^ { \prime \prime }$ is the kth point in X. The positions and velocities are computed by

$$
\left[ \begin{array} { l } { x _ { b } ( t _ { k } + \tau _ { j } ) } \\ { \dot { x } _ { b } ( t _ { k } + \tau _ { j } ) } \end{array} \right] = \left[ \begin{array} { c c c } { 1 } & { T _ { k } } & { \frac { T _ { k } ^ { 2 } } { 2 } } \\ { 0 } & { 1 } & { T _ { k } } \end{array} \right] \left[ \begin{array} { c } { x ( \bar { t } _ { i } ) } \\ { \dot { x } ( \bar { t } _ { i } ) } \\ { \ddot { x } ( \bar { t } _ { i } ) } \end{array} \right]\tag{31}
$$

$$
\left[ \begin{array} { l } { y _ { b } ( t _ { k } + \tau _ { j } ) } \\ { \dot { y } _ { b } ( t _ { k } + \tau _ { j } ) } \end{array} \right] = \left[ \begin{array} { c c c } { 1 } & { T _ { k } } & { \frac { T _ { k } ^ { 2 } } { 2 } } \\ { 0 } & { 1 } & { T _ { k } } \end{array} \right] \left[ \begin{array} { l } { y ( \bar { t } _ { i } ) } \\ { \dot { y } ( \bar { t } _ { i } ) } \\ { \ddot { y } ( \bar { t } _ { i } ) } \end{array} \right]\tag{32}
$$

$$
\left[ \begin{array} { l } { z _ { b } ( t _ { k } + \tau _ { j } ) } \\ { \dot { z } _ { b } ( t _ { k } + \tau _ { j } ) } \end{array} \right] = \left[ \begin{array} { c c c } { 1 } & { T _ { k } } & { \frac { T _ { k } ^ { 2 } } { 2 } } \\ { 0 } & { 1 } & { T _ { k } } \end{array} \right] \left[ \begin{array} { l } { z ( \bar { t } _ { i } ) } \\ { \dot { z } ( \bar { t } _ { i } ) } \\ { \ddot { z } ( \bar { t } _ { i } ) } \end{array} \right]\tag{33}
$$

where $T _ { k } = t _ { k } + \tau _ { j } - { \bar { t } } _ { i } ,$ and $\tau _ { j }$ is from $\theta _ { j }$ in the jth iteration. The likelihood of point $^ { 6 6 } { } ^ { , 5 }$ [see in Fig. 4(b)] is computed using the measurement residual

$$
\mathbf { v } _ { b } = [ ( \Delta _ { x } ^ { i } - \mathbf { D } ^ { i } \mathbf { d } _ { x } ^ { i } ) ^ { \prime } ~ ( \Delta _ { y } ^ { i } - \mathbf { D } ^ { i } \mathbf { d } _ { y } ^ { i } ) ^ { \prime } ~ ( \Delta _ { z } ^ { i } - \mathbf { D } ^ { i } \mathbf { d } _ { z } ^ { i } ) ^ { \prime } ] ^ { \prime }\tag{34}
$$

according to

$$
\mathcal { L } _ { b } = \mathcal { N } ( \mathbf { v } _ { b } ; \mathbf { 0 } , \mathbf { I } )\tag{35}
$$

where $\mathcal { N } ( \cdot )$ is the standard 3δ-multivariate Gaussian pdf. The second step LS is computed in a similar manner to obtain the positions, velocities and likelihood of point ${ \bf \ddot { a } } _ { \bf \Phi } ^ { \ast } .$ . The final estimate, for point $\mathrm { ^ 6 c } ^ { \mathrm { 5 } }$ in Fig. 4(b) is based on a weighted average as follows:

$$
\begin{array} { r } { \left\lceil \begin{array} { l } { \hat { x } ( t _ { k } + \tau _ { j } ) } \\ { \hat { \dot { x } } ( t _ { k } + \tau _ { j } ) } \end{array} \right\rceil \qquad \left\lceil \begin{array} { l } { x _ { a } ( t _ { k } + \tau _ { j } ) } \\ { \dot { x } _ { a } ( t _ { k } + \tau _ { j } ) } \end{array} \right\rceil } \end{array}
$$

$$
\left| \begin{array} { l } { \hat { x } ( t _ { k } + \tau _ { j } ) } \\ { \hat { y } ( t _ { k } + \tau _ { j } ) } \\ { \hat { y } ( t _ { k } + \tau _ { j } ) } \\ { \hat { z } ( t _ { k } + \tau _ { j } ) } \\ { \hat { z } ( t _ { k } + \tau _ { j } ) } \end{array} \right| = \frac { \mathcal { L } _ { a } } { \mathcal { L } _ { a } + \mathcal { L } _ { b } } \left| \begin{array} { l } { \dot { x } _ { a } ( t _ { k } + \tau _ { j } ) } \\ { y _ { a } ( t _ { k } + \tau _ { j } ) } \\ { \dot { y } _ { a } ( t _ { k } + \tau _ { j } ) } \\ { z _ { a } ( t _ { k } + \tau _ { j } ) } \\ { \dot { z } _ { a } ( t _ { k } + \tau _ { j } ) } \end{array} \right|
$$

$$
+ \frac { \mathcal { L } _ { b } } { \mathcal { L } _ { a } + \mathcal { L } _ { b } } \left[ \begin{array} { l } { x _ { b } ( t _ { k } + \tau _ { j } ) } \\ { \dot { x } _ { b } ( t _ { k } + \tau _ { j } ) } \\ { y _ { b } ( t _ { k } + \tau _ { j } ) } \\ { \dot { y } _ { b } ( t _ { k } + \tau _ { j } ) } \\ { z _ { b } ( t _ { k } + \tau _ { j } ) } \\ { \dot { z } _ { b } ( t _ { k } + \tau _ { j } ) } \end{array} \right]\tag{36}
$$

B. Update the estimate of θ

The parameter given in (9) is estimated based on

$$
\boldsymbol { \hat { \theta } } = \arg \operatorname* { m i n } _ { \boldsymbol { \theta } } | | \mathbf { Z } - \mathbf { H } ( \boldsymbol { \theta } , \mathbf { X } ) | | _ { \mathbf { R } ^ { - 1 } } ^ { 2 }\tag{37}
$$

Using the ILS [1] to solve the above optimization<sup>3</sup>, one has

$$
\begin{array} { r l r } { \hat { \theta } _ { j + 1 } } & { = } & { \hat { \theta } _ { j } + { \bf P } _ { j } { \bf J } _ { j } ^ { \prime } { \bf R } ^ { - 1 } [ { \bf Z } - { \bf H } ( \theta _ { j } , { \bf X } _ { j } ) ] } \end{array}\tag{38}
$$

$$
\begin{array} { c c l } { \mathbf { P } _ { j } } & { = } & { ( \mathbf { J } _ { j } ^ { \prime } \mathbf { R } ^ { - 1 } \mathbf { J } _ { j } ) ^ { - 1 } } \end{array}\tag{39}
$$

$$
j = 1 , \dotsc , n _ { j }
$$

with the Jacobian

$$
\mathbf { J } _ { j } = [ \nabla _ { \theta _ { j } } \mathbf { H } ( \theta _ { j } , \mathbf { X } ) ^ { \prime } ] ^ { \prime } = [ \nabla _ { \theta _ { j } } \mathbf { h } _ { 1 } ( \cdot ) ^ { \prime } ~ \dots ~ \nabla _ { \theta _ { j } } \mathbf { h } _ { n } ( \cdot ) ^ { \prime } ] ^ { \prime }\tag{40}
$$

<sup>3</sup>The ILS is the numerical algorithm to solve for the ML Estimate under Gaussian assumption.

where $j$ is the iteration index. The final estimate $\hat { \theta }$ is the value to which the iteration (38) converged using a stopping criterion. The derivatives needed for (40) are given in Appendix B.

## C. Stopping Criterion

To obtain a good calibration result, we set a tight stopping criterion. First, we normalize the measurement residual squared element by element in iteration j

$$
\mathbf { V } _ { j } = [ \mathbf { Z } - \mathbf { H } ( \mathbf { X } _ { j } , \hat { \theta } _ { j } ) ] \otimes [ \mathbf { Z } - \mathbf { H } ( \mathbf { X } _ { j } , \hat { \theta } _ { j } ) ] \sigma _ { \mathrm { F } } ^ { - 2 }\tag{41}
$$

Then, we check every element $v _ { j , i }$ with $( i = 1 \ldots 2 n )$ in $\mathbf { V } _ { j }$ , where $2 n$ is the number of measurements times the measurement dimension 2. All $v _ { j , i }$ must be below the $^ { 6 6 } 3$ sigma” limit

$$
v _ { j , i } \leq 3 ^ { 2 }\tag{42}
$$

This element-wise checking criterion can prevent a few large measurement residuals being smoothed by a large number of small residuals. Also, to prevent a run that cannot meet the stopping criterion, the maximum number of iterations is set to 20.

## V. Simulation Results

This section evaluates the performance of the algorithm described in Section IV. We simulate two test scenarios. Scenario 1 shown in Fig. 5 has a drone (quadcopter) moving in a vertical rectangular trajectory 1,2,3,4 with two cycles. Points 1 and 4 are at near range 200m with altitudes 284m and 98m, respectively. Points 2 and 3 are at farther range 500m with altitudes 284m and 98m, respectively. The drone moves with a nearly constant speed of 12.5m/s between the four edges. When reaching a corner, it decelerates to $0 \mathrm { m / s }$ , then accelerates to 12.5m/s on the new direction. The total duration is 109.2s with 546 measurements. The design principle of the trajectory for this scenario is to span the entire FOV (with near and far motion, i.e., also in depth). Scenario 2 uses the recommended drone path in [26]. This is shown in Fig. 6 with the drone moving with speed of $\mathrm { 1 2 . 8 m / s }$ at altitude 100m, and then it makes a 180<sup>o</sup> turn, and flies back with the same speed and altitude. The total duration is 36s with 126 measurements. The inaccurate GPS trajectories are discretized with a time interval of 0.1s. Camera measurements sampling interval is 0.2s. The camera to calibrate has a field of view of $1 0 ^ { \mathrm { o } }$ and $1 7 . 8 ^ { \mathrm { o } }$ horizontal and vertical, respectively. The nominal orientation $\mathrm { \ a n g l e s ^ { 4 } }$ are set as $\alpha = 3 0 ^ { \mathrm { o } } , \epsilon = 2 ^ { \mathrm { o } }$ and $\rho = 0 ^ { \mathrm { o } }$ . However their actual values (to be estimated) are $\alpha = 3 2 ^ { \mathrm { o } } , \epsilon = 4 . 1 ^ { \mathrm { o } }$ and $\rho = 2 . 3 ^ { \mathrm { o } }$ . The camera provided the measurements only when the target is in its field of view with measurement error standard deviation $\sigma _ { \mathrm { F } }$ =1pixel. The time and altitude ofsets are $\tau \ : = \ : 1 . 3 5 \mathrm { s }$ and $\hbar \ : = \ : 1 0 \mathrm { m }$ in both scenarios, respectively. We set the time ofset precision lower than the GPS discretized precision<sup>5</sup> on purpose to observe the algorithm estimation accuracy better. We will study the estimation accuracy, the statistical eficiency through Normalized Estimation Error Squared (NEES) w.r.t. the CRLB [2] and the real impact of the results next.

![](images/1cbbb415d2ca77aa8a527713d3a432e865a32a580124549766a805ca10ff1d59.jpg)  
(a)

![](images/d60ac4794f422d8e1e51872926ed39017501ef1a19fe0e6169c8613f552afa1e.jpg)  
(b)

Fig. 5. Test scenario 1. Target moves in constant velocity in a vertical rectangle $^ { ( 1 , 2 , 3 , 4 ) }$ twice. The higher and lower horizontal edges are at altitudes 284m and 84m, respectively, and near and far vertical edges are at ranges 200m and 500m, respectively. The target speed is 12.5m/s. (a) Top view in the 3D common coordinates. (b) Trajectory in image coordinates.  
![](images/758c234ea067c06bc33f8e794143b87d6ae8baeeabbd0e992b12dc04cbc5dcca.jpg)  
(a)

![](images/68783af5a1eec9e650ee0375b1088999d49b84040d5008c73075259cd90a05c6.jpg)  
(b)  
Fig. 6. Test scenario 2. Target moves in constant velocity, makes a 180<sup>o</sup> turn, and flies back in constant velocity. The target speed is 12.5m/s. (a) Top view in the 3D common coordinates. (b) Trajectory in image coordinates.

## A. Estimation Accuracy

We conducted 100 Monte Carlo runs for each scenario, and recorded the Root Mean Square Error (RMSE). The CRLB-based covariance matrix is also computed as a benchmark, and is given by

$$
{ \bf P } = ( { \bf J ^ { \prime } } { \bf R } ^ { - 1 } { \bf J } ) ^ { - 1 }\tag{43}
$$

where J is computed by (38), but θ used in (38) is the true value, namely,

$$
\theta = [ 3 2 ^ { \mathrm { o } } ~ 4 . 1 ^ { \mathrm { o } } ~ 2 . 3 ^ { \mathrm { o } } ~ 1 0 \mathrm { m } ~ 1 . 3 5 \mathrm { s } ] ^ { \prime }\tag{44}
$$

Note the three angles in θ should be converted to radians as the unit of measurement in both CRLB and ILS

computing, as discussed before. The CRLB standard deviations of the estimated parameters are

$$
\begin{array} { r l r } { \alpha ^ { \mathrm { C R L B } } } & { { } = } & { \sqrt { { \bf P } ( 1 , 1 ) } } \end{array}\tag{45}
$$

$$
\begin{array} { r l r } { \epsilon ^ { \mathrm { C R L B } } } & { { } = } & { \sqrt { { \bf P } ( 2 , 2 ) } } \end{array}\tag{46}
$$

$$
\begin{array} { r l r } { \rho ^ { \mathrm { C R L B } } } & { { } = } & { \sqrt { { \bf P } ( 3 , 3 ) } } \end{array}\tag{47}
$$

$$
\begin{array} { r l r } { \hbar ^ { \mathrm { C R L B } } } & { { } = } & { \sqrt { { \bf P } ( 4 , 4 ) } } \end{array}\tag{48}
$$

$$
\begin{array} { r l r } { \tau ^ { \mathrm { C R L B } } } & { { } = } & { \sqrt { { \bf P } ( 5 , 5 ) } } \end{array}\tag{49}
$$

Table II gives the RMSE and CRLB for scenarios 1 and 2. It also lists the results of the scenario 2 (under $2 ^ { * } )$ obtained in [26], where the same drone path was used, but GPS altitude was assumed perfect without bias. It can be seen that the estimate RMSEs of scenario 1 are close to their CRLBs. The algorithm is statistically eficient in this scenario, as shown in the next subsection. However, the results of scenario 2 are significantly less accurate. The CRLBs are significantly larger than those of scenario 1, especially for ϵ and ℏ with values 32.21mdeg and 475mm, respectively. This indicates the observabilities of ϵ and ℏ are marginal in this scenario.

We plot the drone trajectories as seen by the camera and the GPS converted positions in the image space for scenarios 1 and 2 in Figs. 7 and 8, respectively. The parameters used for GPS conversion are set the same as the true values, except for the two marginally observable parameters ϵ and ℏ. The true values are $\epsilon = 4 . 1 ^ { \mathrm { o } }$ and $\hbar =$ 10m, respectively. The values used in the GPS conversion are $\epsilon = 2 ^ { \mathrm { o } }$ and $\hbar = 0 \mathrm { m }$ , respectively. It can be seen that the true and the GPS converted trajectories for scenario 1 (Fig. 7) are quite diferent. This is mainly because the longer vertical edge is at near range 200m, and the shorter vertical edge is at farther range 500m. One cannot match them without correct values on both ϵ and ℏ. However, the trajectories for scenario 2 (Fig. 8) are almost parallel. Since the two legs on the drone path are at similar range, one can change either GPS altitude or pitch to match the two trajectories. Furthermore, we can also observe that the diference between the RMSE and $\sigma _ { \mathrm { C R L B } }$ for ϵ and ℏ are also significantly larger in scenario 2 than those of scenario 1. The algorithm does not perform well when the problem observability is marginal, as in scenario 2 which does not meet the design principle of Scenario 1.

Comparing scenarios 2 and $2 ^ { * }$ we can see that including GPS altitude bias (which is generally present) significantly increases the estimation error using the drone path recommended in [26]. The path is not practical for camera calibration when GPS altitude bias is taken into consideration. The reason is that the trajectory of scenario 2 has poor observerbility when both GPS altitude and camera pitch are unknown.

Another interesting observation is the estimation accuracy of τ is smaller than the GPS time discretization of 100ms. The best RMSE reaches 0.27ms in test scenario 1. This shows that the trajectory estimation algorithm described in Sec. IV overcomes the discretization of the GPS trajectory problem efectively.

TABLE II  
CRLB and RMSE from 100 runs
<table><tr><td rowspan=1 colspan=1>Scenario</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>RMSE  σCRLB</td></tr><tr><td rowspan=2 colspan=1>1</td><td rowspan=2 colspan=1> $\overline { { \alpha - \mathrm { y a w } \ ( \mathrm { m d e g } ) } }$  $\epsilon { \mathrm { - } } \mathrm { p i t c h ~ } ( \mathrm { m d e g } )$  $\rho { \mathrm { - } } \mathrm { r o l l ~ } ( \mathrm { m d e g } )$  $\hbar - \ ( \mathrm { m m } )$  $\tau \ ( \mathrm { m s } )$ </td><td rowspan=1 colspan=1>0.23     0.21</td></tr><tr><td rowspan=1 colspan=1>0.87    0.822.90     2.814.69     4.350.27    0.22</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $\overline { { \alpha - \mathrm { y a w } \ ( \mathrm { m d e g } ) } }$  $\epsilon { \mathrm { - } } \mathrm { p i t c h ~ } ( \mathrm { m d e g } )$  $\rho { \mathrm { - } } \mathrm { r o l l ~ } \left( \mathrm { m d e g { \mathrm { - } } y a w } \right)$  $\hbar - \ ( \mathrm { m m } )$  $\tau \ ( \mathrm { m s } )$ </td><td rowspan=1 colspan=1>1.34    1.0536.08    32.2114.63   14.25531     4750.80    0.75</td></tr><tr><td rowspan=1 colspan=1> $2 ^ { * }$ </td><td rowspan=1 colspan=1> $\overline { { \alpha - \mathrm { y a w } \ ( \mathrm { m d e g } ) } }$  $\epsilon { \mathrm { - } } \mathrm { p i t c h ~ } ( \mathrm { m d e g } )$  $\rho { \mathrm { - } } \mathrm { r o l l ~ } ( \mathrm { m d e g } )$  $\tau \ ( \mathrm { m s } )$ </td><td rowspan=1 colspan=1>0.45     0.430.47     0.438.89     8.420.57     0.50</td></tr></table>

![](images/9e72d9abc0d7f3f9dbb30e87b74b1b0af5d682b1380a048388aef4d67281ebed.jpg)  
Fig. 7. Measured and GPS converted trajectories of scenario 1, where the actual and nominal yaw, roll and time diference are set to the same values as $\alpha = 3 2 ^ { \mathrm { o } } , \rho = 2 . 3 ^ { \mathrm { o } }$ and τ = 1.35s, respectively. The actual and nominal pitch are $\epsilon = 4 . 1 ^ { \mathrm { o } }$ and $2 ^ { \mathrm { o } }$ , respectively. The actual and nominal GPS altitude bias are 10m and 0m, respectively.

## B. Statistical Eficiency

The statistical eficiency analysis was conducted using the Normalized Estimation Error Squared (NEES) [2] computed w.r.t. the CRLB, namely

$$
\epsilon ^ { i } ( t _ { k } ) = ( \theta - \hat { \theta } ) ^ { \prime } \mathrm { \bf P } ^ { - 1 } ( \theta - \hat { \theta } )\tag{50}
$$

where $\hat { \theta }$ and θ are the parameter estimate and true value, respectively. The NEESs of $N { = } 1 0 0$ runs were recorded and the analysis is carried out for each run, as well as using the average. The NEES of the parameter (with dimension 5) is a 5 degrees of freedom chi-square random variable if the errors are Gaussian. Its two-sided $\scriptstyle p = 9 5 \%$ probability region is [0.8, 12.8]. The estimation is statistically eficient, if 95% of NEESs are within this region. Figs. 9–10 show the NEES for the two test scenarios, and the number of NEES out of the region [0.8, 12.8] for scenario 1 is 0 (vs. the expected value of 5), i.e., the algorithm produced statistically eficient estimates — consistent with equality in the CRLB. However, the number of NEES out of the this interval from 100 runs is scenario 2 is 9. This shows that the estimation algorithm for a marginally observable scenario is marginally statistically eficient. This is because the standard deviation of the number of exceedances of the 95% probability interval is $\sqrt { N p ( 1 - p ) } \approx 2$ , thus the borderline eficiency.

![](images/c60efaca8c1478d309d82550f9abdd5e2d3018f785e649d52e4b19aa66efed0b.jpg)  
Fig. 8. Measured and GPS converted trajectories of scenario $^ { 2 , }$ where the actual and nominal yaw, roll and time diference are set to the same values as $\alpha = 3 2 ^ { \mathrm { o } }$ $\rho = 2 . 3 ^ { \mathrm { o } }$ and $\tau = 1$ .35s, respectively. The actual and nominal pitch are $\epsilon = 4 . 1 ^ { \mathrm { o } }$ and $2 ^ { \mathrm { o } }$ , respectively. The actual and nominal GPS altitude bias are 10m and 0m, respectively.

![](images/55f26afea3241dfb1016077761d6ba04652f02e67b81b8603873723c9d114685.jpg)  
Fig. 9. NEES of 100 runs of the scenario 1.

For the average NEES over 100 runs, the 95% probability region, based on $\chi _ { 5 0 0 } ^ { 2 } / 1 0 0$ , is the interval [4.1 5.63]. For scenario 1 the average NEES is 4.69 while for scenario 2 it is 5.86 Thus, the same conclusions can be drawn: for scenario 1 the algorithm is eficient while for scenario 2 it is borderline.

## C. Impact of the Residual Biases

The real impact is further discussed based on the calibration result of scenario 1 which yields a good calibration result. The pixel bias error in the image space caused by the residual calibration errors should be much lower than the measurement error, so that the residual calibration errors are negligible. We compute the pixel bias error based on the calibration RMSE of yaw, pitch, roll and their combination. The residual bias error impact is obtained from the shifted distances (the unit of measure is pixel) for uniformly distributed $5 \times 5$ pixel grid elements covering the whole image space<sup>6</sup> (1–2160 in $\bar { x ^ { \mathrm { I } } }$ , 1–3840 in $y ^ { \mathrm { I } } )$ when the residual yaw, pitch and roll errors are introduced. The residual bias error of the kth grid is

![](images/ace785440eee7bc5c31d92d10032857cc47547c7d7f06b0d3dc0a74ad3a3e89f.jpg)  
Fig. 10. NEES of 100 runs of the scenario 2.

$$
b _ { k } = | ( \breve { x } _ { k } ^ { \mathrm { I } } , \breve { y } _ { k } ^ { \mathrm { I } } ) - ( x _ { k } ^ { \mathrm { I } } , y _ { k } ^ { \mathrm { I } } ) |\tag{51}
$$

where $( x _ { k } ^ { \mathrm { I } } , y _ { k } ^ { \mathrm { I } } )$ is the center of the kth grid element in pixel units and $( \breve { x } _ { k } ^ { \mathrm { I } } , \breve { y } _ { k } ^ { \mathrm { I } } )$ is the shifted grid center when the residual calibration errors are added to the nominal yaw, pitch and roll; $b _ { k }$ is the distance in pixel units between these two grids. We recorded the residual errors $b _ { k }$ of all the grids and plot them in Fig. 11 for three cases. Case (a) has 0.23mdeg calibration error added to yaw only. Cases (b) and (c) have 0.87mdeg error added to pitch and 2.9mdeg error added to roll, respectively. Fig. 12 shows the efect of the combination of yaw, pitch and roll errors. Case (a) increases the yaw, pitch and roll by 0.23mdeg, 0.87mdeg and 2.9mdeg, respectively. Case (b) reduces the yaw, pitch and roll by 0.23mdeg, 0.87mdeg and 2.9mdeg, respectively. The statistics of the grid biases are summarized in Table III. It shows the bias min., max., mean, standard deviation and Root Mean Square (RMS) value for the five cases in Figs. 11 and 12. From these results we observe the following:

The residual bias error is negligible compared to the measurement error. The highest RMSE due to the residual bias is 0.20pixel. The measurement RMSE in one coordinate (either $x ^ { \mathrm { I } }$ or $y ^ { \mathrm { I } } )$ is 1pixel. Assuming they are uncorrelated between the coordinates, the total measurement error standard deviation is 1.41pixel. The highest RMSE due to residual bias is 7.2 times smaller than the measurement RMSE. Thus the calibration using the scenario 1 drone trajectory achieves negligible bias error.

![](images/d0cdee26786fe34960c8e73025948d1ff7e38a7c2937382505517a8f250d0e45.jpg)  
(a)

![](images/ac0d70e01840406153c22c635f8109a23c0c597ff9726f0b9f1dfdf072d31df6.jpg)  
(b)

![](images/ea30fdb1db7e69c39129971be725ddf19ece605dc5b7ffbcdda82b23c58378c8.jpg)  
(c)  
Fig. 11. Biased errors on all 5 5 grids. (a) The biased error caused by calibration error on yaw of 0.23mdeg. (b) The biased error caused by calibration error on pitch of 0.87mdeg. (c) The biased error caused by calibration error on roll of 2.9mdeg.

A yaw error creates higher bias on the two vertical edges, and pitch error creates higher bias on the two horizontal edges, as shown in Fig. 11 (a) and (b). The diferences between the edges and the center are, however, very small.

A roll error creates higher bias at the four corners, the furthest distance to the center, and the center has zero bias in Fig. 11 (c). Nevertheless, the bias at the corners is negligible.

A combined yaw, pitch and roll error creates the highest bias at one of the corners from Fig. 12. However, the max. 0.29pixels is still negligible compared to the measurement RMSE of 1.41pixel. The max. combined RMSE (measured and bias) is $\sqrt { 1 . 4 1 ^ { 2 } + 0 . 2 9 ^ { 2 } }$ = 1.44pixel.

## VI. Conclusions

In this paper, we developed a camera calibration algorithm using drone trajectories recorded by a GPS receiver. However, the recorded GPS data has an unknown altitude bias and an unknown time ofset between the GPS and camera systems. The GPS trajectories are discretized with time interval 0.1s. The paper developed a special ${ \mathrm { M L } } / { \mathrm { I L S } }$ algorithm dealing with discretized GPS trajectories to estimate camera orientation angles (yaw, pitch and roll), GPS altitude bias and time ofset simultaneously. The simulation tests were conducted and an appropriate drone trajectory is recommended whose estimation results met the CRLB and NEES requirements. The time ofset estimation error was much smaller than the discretization of the GPS reference trajectory (0.27ms vs. 100ms). The recommended drone trajectory is suitable for practical use. Its residual calibration bias RMSE was 14% of the measurement error standard deviation, which is negligible.

![](images/3b6b97070aa3214ff640da07fbd36be35a24014375883c31edba8e4151079c9e.jpg)  
(a)

![](images/4bbd2f28ae36034eb6d30d96f240f261c8db5393f0f9c27c8da23d1acd072783.jpg)  
(b)  
Fig. 12. Biased errors on all 5 5 grids. (a) Increases the yaw, pitch and roll by 0.23mdeg, 0.87mdeg and 2.9mdeg, respectively. (b) Reduces the yaw, pitch and roll by 0.23mdeg, 0.87mdeg and 2.9mdeg, respectively.

TABLE III  
Biased error in pixel caused by the calibration error
<table><tr><td colspan="3">Calibration error (mdeg)</td><td colspan="5">Bias (pixel)</td></tr><tr><td>α</td><td>€</td><td>ρ</td><td>Min.</td><td>Max.</td><td>Mean</td><td>Stdv.</td><td>RMS</td></tr><tr><td>0.23</td><td>0</td><td>0</td><td>0.050</td><td>0.050</td><td>0.050</td><td>0.000</td><td>0.050</td></tr><tr><td>0</td><td>0.87</td><td>0</td><td>0.187</td><td>0.192</td><td>0.189</td><td>0.001</td><td>0.189</td></tr><tr><td>0</td><td>0</td><td>2.90</td><td>0.000</td><td>0.110</td><td>0.064</td><td>0.025</td><td>0.059</td></tr><tr><td>0.23</td><td>0.87</td><td>2.90</td><td>0.134</td><td>0.285</td><td>0.210</td><td>0.033</td><td>0.200</td></tr><tr><td>-0.23</td><td>-0.87</td><td>-2.90</td><td>0.134</td><td>0.285</td><td>0.210</td><td>0.033</td><td>0.200</td></tr></table>

In our real camera setup and calibration experiments, we realized that more work needs to be done along this research. First, the camera focal length cannot be fixed beforehand accurately. It needs to be adjusted during setup based on the real situation. Due to lack of accurate equipment to measure a camera focal length, it should be an additional camera parameter included in the estimation. Second, the GPS equipment usually has quantization error in latitude and longitude. This error cannot be ignored when a target is in a near range (with 10 pixel quantization). The ILS algorithm proposed in this paper need to be further developed to handle these types

of errors.

## Appendix A

The Importance of Being Earnest about Radians

When trigonometric functions are expressed as Taylor expansion, one has to use radians as the unit of measure. This can be illustrated using the following simple example, using the first order Taylor expansion to compute sin(30.01<sup>o</sup>). The answer should be 0.50015. If we use degrees as the unit of measure we will have wrong result as

$$
{ \begin{array} { r l } & { \sin ( 3 0 . 0 1 ^ { \circ } ) = \sin ( 3 0 ^ { \circ } + 0 . 0 1 ^ { \circ } ) } \\ & { ~ \approx \sin ( 3 0 ^ { \circ } ) + 0 . 0 1 ^ { \circ } \times [ \sin ( 3 0 ^ { \circ } ) ] ^ { \prime } } \\ & { ~ \approx \sin ( 3 0 ^ { \circ } ) + 0 . 0 1 ^ { \circ } \times \cos ( 3 0 ^ { \circ } ) } \\ & { ~ \approx 0 . 5 + 0 . 0 1 \times 0 . 8 6 6 } \\ & { ~ \approx 0 . 5 0 8 6 6 } \end{array} }\tag{52}
$$

If we use radians, the correct result is

$$
\begin{array} { c } { { \sin \left( \displaystyle \frac { 3 0 . 0 1 \times \pi } { 1 8 0 } \right) \approx \sin \left( \displaystyle \frac { 3 0 \times \pi } { 1 8 0 } \right) + \displaystyle \frac { 0 . 0 1 \times \pi } { 1 8 0 } \cos \left( \displaystyle \frac { 3 0 \times \pi } { 1 8 0 } \right) } } \\ { { \approx 0 . 5 + 0 . 0 0 1 7 5 \times 0 . 8 6 6 } } \\ { { \approx 0 . 5 0 0 1 5 } } \end{array}
$$

Although sin(·) and cos(·) should give the same values whether the units are degrees or radians, the small diference 0.01<sup>o</sup> in front of cos(·) in (52) leads to wrong result in (53). Thus angles must be converted to radians when using series expansions.

## Appendix B Derivatives for (40)

The iteration index $j$ is omitted for simplicity. The gradients needed are

$$
[ \nabla _ { \theta } \mathbf { h } _ { k } ( \cdot ) ^ { \prime } ] ^ { \prime } = \frac { \partial \mathbf { x } _ { k } ^ { \mathrm { I } } } { \partial \mathbf { x } _ { k } ^ { \mathrm { C } } } \frac { \partial \mathbf { x } _ { k } ^ { \mathrm { C } } } { \partial \theta } \qquad k = 1 \dots n\tag{54}
$$

$$
\begin{array} { r l r } { \frac { \partial \mathbf { x } _ { k } ^ { \mathrm { I } } } { \partial \mathbf { x } _ { k } ^ { \mathrm { C } } } } & { = } & { \left[ \begin{array} { c c c } { \displaystyle \frac { f } { z _ { k } ^ { \mathrm { C } } } } & { 0 } & { - \displaystyle \frac { f x _ { k } ^ { \mathrm { C } } } { ( z _ { k } ^ { \mathrm { C } } ) ^ { 2 } } } \\ { 0 } & { \displaystyle \frac { f } { z _ { k } ^ { \mathrm { C } } } } & { - \displaystyle \frac { f y _ { k } ^ { \mathrm { C } } } { ( z _ { k } ^ { \mathrm { C } } ) ^ { 2 } } } \end{array} \right] } \end{array}\tag{55}
$$

$$
\begin{array} { r l r } { \frac { \partial { \bf x } _ { k } ^ { \mathrm { C } } } { \partial \theta } } & { = } & { \left[ \begin{array} { l l l l l } { \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \alpha } } & { \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \epsilon } } & { \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \rho } } & { \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \hbar } } & { \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \tau } } \\ { \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \alpha } } & { \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \epsilon } } & { \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \rho } } & { \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \hbar } } & { \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \tau } } \\ { \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \alpha } } & { \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \epsilon } } & { \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \rho } } & { \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \hbar } } & { \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \tau } } \end{array} \right] } \end{array}\tag{56}
$$

and

$$
\begin{array} { r c l } { \displaystyle \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \alpha } } & { = } & { \displaystyle \Delta x _ { k } ( c _ { \alpha } s _ { \epsilon } s _ { \rho } - s _ { \alpha } c _ { \rho } ) - \Delta y _ { k } ( s _ { \alpha } s _ { \epsilon } s _ { \rho } + c _ { \alpha } c _ { \rho } ) } \end{array}\tag{57}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \epsilon } } & { = } & { \Delta x _ { k } s _ { \alpha } c _ { \epsilon } s _ { \rho } + \Delta y _ { k } c _ { \alpha } c _ { \epsilon } s _ { \rho } + \Delta z _ { k } s _ { \epsilon } c _ { \rho } } \end{array}\tag{58}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \rho } } & { = } & { \Delta x _ { k } \big ( s _ { \alpha } s _ { \epsilon } c _ { \rho } - c _ { \alpha } s _ { \rho } \big ) } \end{array}
$$

$$
+ \Delta y _ { k } ( c _ { \alpha } s _ { \epsilon } c _ { \rho } + s _ { \alpha } s _ { \rho } ) - \Delta z _ { k } c _ { \epsilon } c _ { \rho }\tag{59}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \hbar } } & { = } & { c _ { \epsilon } s _ { \rho } } \end{array}\tag{60}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \tau } } & { = } & { \hat { \dot { x } } ( t _ { k } + \tau ) ( c _ { \alpha } c _ { \rho } + s _ { \alpha } s _ { \epsilon } s _ { \rho } ) } \\ & & { + \hat { y } ( t _ { k } + \tau ) ( c _ { \alpha } s _ { \epsilon } s _ { \rho } - s _ { \alpha } c _ { \rho } ) - \hat { \dot { z } } ( t _ { k } + \tau ) c _ { \epsilon } s _ { \rho } } \end{array}\tag{61}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \alpha } } & { = } & { \Delta x _ { k } ( c _ { \alpha } s _ { \epsilon } c _ { \rho } + s _ { \alpha } s _ { \rho } ) + \Delta y _ { k } ( c _ { \alpha } s _ { \rho } - s _ { \alpha } s _ { \epsilon } c _ { \rho } ) } \end{array}\tag{62}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \epsilon } } & { = } & { \Delta x _ { k } s _ { \alpha } c _ { \epsilon } c _ { \rho } + \Delta y _ { k } c _ { \alpha } c _ { \epsilon } c _ { \rho } + \Delta z _ { k } s _ { \epsilon } c _ { \rho } } \end{array}\tag{63}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \rho } } & { = } & { \displaystyle - \Delta x _ { k } ( s _ { \alpha } s _ { \epsilon } s _ { \rho } + c _ { \alpha } c _ { \rho } ) } \end{array}
$$

$$
\begin{array} { c c l } { \displaystyle \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \hbar } } & { = } & { c _ { \epsilon } c _ { \rho } } \end{array}\tag{64}
$$

(65)

$$
\begin{array} { r c l } { \displaystyle \frac { \partial y _ { k } ^ { \mathrm { C } } } { \partial \tau } } & { = } & { \hat { \dot { x } } ( t _ { k } + \tau ) \big ( s _ { \alpha } s _ { \epsilon } c _ { \rho } - c _ { \alpha } s _ { \rho } \big ) } \\ & & { + \hat { y } ( t _ { k } + \tau ) \big ( s _ { \alpha } s _ { \rho } + c _ { \alpha } s _ { \epsilon } c _ { \rho } \big ) - \hat { \dot { z } } ( t _ { k } + \tau ) c _ { \epsilon } c _ { \rho } } \end{array}\tag{66}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \alpha } } & { = } & { \Delta x _ { k } c _ { \alpha } c _ { \epsilon } - \Delta y _ { k } s _ { \alpha } c _ { \epsilon } } \end{array}\tag{67}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \epsilon } } & { = } & { \displaystyle - \Delta x _ { k } s _ { \alpha } s _ { \epsilon } - \Delta y _ { k } c _ { \alpha } s _ { \epsilon } + \Delta z _ { k } c _ { \epsilon } } \end{array}\tag{68}
$$

$$
\begin{array} { l c l } { { \displaystyle { \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \rho } } } } & { { = } } & { { 0 } } \end{array}
$$

$$
\begin{array} { l c l } { \frac { \partial x _ { k } ^ { \mathrm { C } } } { \partial \hbar } } & { = } & { - s _ { \epsilon } } \end{array}\tag{69}
$$

(70)

$$
\begin{array} { r c l } { \displaystyle \frac { \partial z _ { k } ^ { \mathrm { C } } } { \partial \tau } } & { = } & { \hat { \dot { x } } ( t _ { k } + \tau ) s _ { \alpha } c _ { \epsilon } + \hat { \dot { y } } ( t _ { k } + \tau ) c _ { \alpha } c _ { \epsilon } + \hat { \dot { z } } ( t _ { k } + \tau ) s _ { \epsilon } } \end{array}\tag{71}
$$

where

$$
\begin{array} { r c l } { \Delta x _ { k } } & { = } & { \hat { x } ( t _ { k } + \tau ) - x ^ { \mathrm { s } } } \end{array}
$$

$$
\begin{array} { r c l } { \Delta y _ { k } } & { = } & { \hat { y } ( t _ { k } + \tau ) - y ^ { \mathrm { s } } } \end{array}\tag{72}
$$

(73)

$$
\begin{array} { r c l } { \Delta z _ { k } } & { = } & { \hat { z } ( t _ { k } + \tau ) - \hbar - z ^ { \mathrm { s } } } \end{array}\tag{74}
$$

The point $[ \hat { x } ( t _ { k } + \tau ) , \hat { y } ( t _ { k } + \tau ) , \hat { z } ( t _ { k } + \tau ) ]$ in $( 7 2 ) \mathrm { - } ( 7 4 )$ on the drone trajectory and its velocity $[ \ddot { x } ( t _ { k } + \tau ) , \ddot { y } ( t _ { k } +$ τ ), $\hat { \dot { z } } ( t _ { k } + \tau ) ]$ in (61), (66) and (71) have been estimated in $\mathrm { { I V - A } }$

The unit of measure for the three angles $\alpha , \epsilon$ and $\rho$ has to be radians — see Appendix A.

[12] Q. Lu, Y. Bar-Shalom and P. Willett

[6] P. Furgale, J. Rehder and R. Siegwart

## References

[1] Y. Bar-Shalom, X. R. Li and T. Kirubarajan Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software. Wiley, 2001.

[2] Y. Bar-Shalom, P. K. Willet, and X. Tian Tracking and Data Fusion: A Handbook of Algorithms. YBS Publishing, 2011.

[3] D. Belfadel, R. W. Osborne, III and Y. Bar-Shalom “Bias estimation and observability for optical sensors with targets of opportunity”.

Journal of Advances in Information Fusion, 9,2 (Dec. 2014), 59–74.

[4] 1.A. “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography”. Communications of the ACM, 24,6 (June 1981), 381–395.

[5] S. Fortunati, A. Farina, F. Gini, A. Graziano, M. S. Greco, and S. Giompapa “Least squares estimation and Cramér–Rao type lower bounds for relative sensor registration process”. IEEE Transactions on Signal Processing, 59,3 (Mar. 2011), 1075–1087.

“Unified temporal and spatial calibration for multi-sensor systems”.

In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems 2013, Nov. 2013, 1280–1286.

[7] R. Haralick, C. Lee, K. Ottenberg and M. Nolle

“Analysis and solutions of the three point perspective pose estimation problem”.

In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 1991, June 1991, 592–598.

[8] R. Haralick, C.-N. Lee, K. Ottenberg and M. Nolle

“Review and analysis of solutions of the three point perspective pose estimation problem”.

International Journal of Computer Vision, 13,3 (Dec. 1994). 331–356.

”Pose estimation with radial distortion and unknown focal length”.

[10] S. Linnainmaa, D. Harwood and L. Davis

In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2009, June 2009, 2419–2426.

“Pose estimation of a three-dimensional object using triangle pairs”.

IEEE Transactions on Pattern Analysis and Machine Intelligence, 10,5 (Sep. 1988), 634–647.

In Proceedings of the 12th International Symposium on Experimental Robotics, Dec. 2010, 195–209.

“Measurement extraction for a Point target from an optical sensor”.

IEEE Transactions on Aerospace and Electronic Systems, 54,6 (Dec. 2018), 2735–2745.

[13] L. Kneip, D. Scaramuzza and R. Siegwart

“A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation”.

In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2011, June 2011, 2969–2976.

[14] W. Li, H. Leung and Yifeng Zhou “Space-time registration of radar and ESM using unscented Kalman filter”.

IEEE Transactions on Aerospace and Electronic Systems, 40,3 (July 2004) 824–836.

The International Journal of Robotics Research, 33,7 (May 2014), 947–964.

[16] S. Li, Y. Cheng, D. Brown, R. Tharmarasa, G. Zhou and T. Kirubarajan

“Comprehensive time-ofset estimation for multisensor target tracking”.

IEEE Transactions on Aerospace and Electronic Systems, 56,3 (June 2020), 2351–2373.

[17] X. D. Lin, Y. Bar-Shalom and T. Kirubarajan

“Exact multisensor dynamic bias estimation with local tracks”. IEEE Transactions on Aerospace and Electronic Systems, 40,2 (Apr. 2004), 576–590.

[18] E. Mair, M. Fleps, M. Suppa and D. Burschka [18] E. Mair, M. Fleps, M. Suppa and D. Burschka

“Spatio-temporal initialization for IMU to camera registration”. In Proceedings of the IEEE International Conference on Robotics and Biomimetics (ROBIO) 2011, Dec. 2011, 557–564.

“Maximum likelihood registration for multiple dissimilar sensors”.

IEEE Transactions on Aerospace and Electronic Systems, 39, 3 (July 2003) 1074–1083.

[20] J. Rehder, R. Siegwart and P. Furgale “A general approach to spatiotemporal calibration in multisensor Systems”.

IEEE Transactions on Robotics, 32,2 (Apr. 2016) 383–398. [21] CH

“P3.5P: pose estimation with unknown focal length”. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2015, June 2015, 2440–2448.

[22] Y. Zhou, H. Leung and P. C. Yip

“An exact maximum likelihood registration algorithm for data fusion”.

IEEE Transactions on Signal Processing, 45,6 (June 1997), 1560–1572.

[23] Y. Zhou, H. Leung and M. Blanchette “Sensor alignment with earth-centered earth-fixed (ECEF) coordinate systems”.

IEEE Transactions on Aerospace and Electronic Systems, 35,2 (Apr. 1999) 410–417.

“Compensation of navigation uncertainty for target tracking on a moving platform”.

In Proceedings of the 19th International Conference on Information Fusion (FUSION) 2016, July. 2016.

[25] C. Yang, E. Blasch and P. Douville

“Design of Schmidt-Kalman filter for target tracking with navigation errors”.

In Proceedings of the IEEE Aerospace Conference 2010, Mar. 2010,

[26] R. Yang, Y. Bar-Shalom and H.A.J. Huang

“Camera calibration with unknown time ofset between the camera and drone GPS Systems”.

In Proceedings of the 25th International Conference on Information Fusion 2022, July 2022.

[27] Z. Zhang “A flexible new technique for camera calibration”. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22,11 (Nov. 2000), 1330–1334.

![](images/b493050467a248a986dae4c2868aae3be64bf2cc0866b1bd6fe849f18a902ad7.jpg)

Rong Yang received her B.E. degree in information and control from Xi’an Jiao Tong University, China in 1986, M.Sc. degree in electrical engineering from National University of Singapore in 2000, and Ph.D. degree in electrical engineering from Nanyang Technological University, Singapore in 2012. She is currently a Principal Member of Technical Staf at DSO National Laboratories, Singapore. Her research interests include passive tracking, low observable target tracking, GMTI tracking,

hybrid dynamic estimation and data fusion. She was Publicity and Publication Chair of FUSION 2012 and received the FUSION 2014 Best Paper Award (First runner up).

![](images/812b7dc5fef87b5d0d6e5b5453efbc1321275a52b1e5de0be32341a58808eb88.jpg)

Yaakov Bar-Shalom (F’84) received the B.S. and M.S. degrees from the Technion in 1963 and 1967 and the Ph.D. degree from Princeton University in 1970, all in EE. Currently he is Board of Trustees Distinguished Professor in the ECE Dept. and Marianne E. Klewin Professor at the University of Connecticut. His current research interests are in estimation theory, target tracking and data fusion. He has published over 650 papers and book chapters. He coauthored/edited 8 books, including

Tracking and Data Fusion (YBS Publishing, 2011). He has been elected Fellow of IEEE for ”contributions to the theory of stochastic systems and of multitarget tracking”. He served as Associate Editor of the IEEE Transactions on Automatic Control and Automatica. He was General Chairman of the 1985 ACC, General Chairman of FUSION 2000, President of ISIF in 2000 and 2002 and Vice President for Publications during 2004-13. Since 1995 he is a Distinguished Lecturer of the IEEE AESS. He is corecipient of the M. Barry Carlton Award for the best paper in the IEEE TAE Systems in 1995 and 2000. In 2002 he received the J. Mignona Data Fusion Award from the DoD JDL Data Fusion Group. He is a member of the Connecticut Academy of Science and Engineering. In 2008 he was awarded the IEEE Dennis J. Picard Medal for Radar Technologies and Applications, and in 2012 the Connecticut Medal of Technology. He has been listed by academic.research.microsoft (top authors in engineering) as #1 among the researchers in Aerospace Engineering based on the citations of his work. He is the recipient of the 2015 ISIF Award for a Lifetime of Excellence in Information Fusion. This award has been renamed in 2016 as the Yaakov Bar-Shalom Award for a Lifetime of Excellence in Information Fusion. He has the following Wikipedia page: https://en.wikipedia.org/wiki/Yaakov Bar-Shalom.

![](images/72bd04a535df9d465561f043fe6c547adb1b80c8f0ad4f32a52c1d80e4f14611.jpg)

Huang Hong’An Jack was born in Singapore in 1983. He received the B.E. from National University of Singapore (NUS) in 2008. He is currently a Senior Member of Technical Staf at DSO National Laboratories. His research interests in target tracking including GMTI tracking, passive tracking and image tracking. He received the FUSION 2014 Best Paper Award (First runner up).