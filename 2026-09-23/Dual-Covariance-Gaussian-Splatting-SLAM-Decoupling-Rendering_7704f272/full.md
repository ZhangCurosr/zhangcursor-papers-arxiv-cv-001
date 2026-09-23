![](images/dad9a59129019a557cd889205deb469dd594302c1abaf21a90d182e46479996f.jpg)

# Dual Covariance Gaussian Splatting SLAM: Decoupling Rendering and Registration for Robust Real-Time Tracking

Edward Beng Wai Tan<sup>1,∗</sup>, Siew-Kei Lam<sup>1</sup>

![](images/50936e9aa25ffef99798f8a48bbc4d8d51307110290b48787cc7ea18cf226894.jpg)

![](images/60454db53e05732114baf416849f889cf386396831660052819b260946f52b9b.jpg)  
Fig. 1: Our method tracks robustly in various challenging conditions, such as indoor loops in (a) compared to other ICP methods, and in structure-less scenes rivaling MonoGS and Photo-SLAM in (b), and in outdoor scenes (d) shows the lowest drift among evaluated 3DGS-based methods. We do this by maintaining two covariances per Gaussian: one for rendering, and one for tracking, as shown in (c), colored by the size of the respective covariances.

Abstract— ICP-based 3D Gaussian Splatting (3DGS) SLAM tracks in real time by registering incoming frames against map Gaussians, using each primitive’s covariance for both rendering and registration. These two uses place conflicting demands on one covariance. The mapper shapes it to minimize photometric error, often flattening it against surfaces, while robust registration typically benefits from measurement uncertainty. We propose a dual-covariance parameterization. Each Gaussian keeps a single mean but holds two covariances: a rendering covariance optimized by the mapper, and a tracking covariance derived from an RGB-D sensor noise model. We further use the tracking covariances as Gaussian anchors for image corners, providing constraints in directions where depth geometry is weak. We evaluate on TUM RGB-D, ScanNet, Replica, and two outdoor sequences recorded with a RealSense D435i on wheeled and handheld platforms. We achieve robust tracking performance across multiple scenes and reduced odometry drift, while tracking at ∼ 60 FPS.

## I. INTRODUCTION

Recently, 3D Gaussian Splatting (3DGS) [1] has been incorporated into SLAM for photorealistic mapping. Initially, the pose was determined by dense photometric error [2], [3] due to the differentiable properties of 3DGS primitives. These methods were effective but proved to be computationally heavy, usually running at single digit FPS on desktop GPUs.

To allow for online, real-time execution of 3DGS SLAM, various hybrid approaches were proposed, such as using sparse feature matching [4] or the aligning point clouds [5] using ICP. The ICP methods recognized that the explicit Gaussian parameterization compared to earlier neural implicit methods, could be used as both mapping primitives and as point clouds for alignment.

In sharing the primitive, these methods also reused the Gaussian’s covariance, optimized by the mapper for minimizing photometric error. This method proved to be sufficient for tracking in many scenes, yet [6] demonstrated that it failed under specific adverse conditions, such as geometrically ill conditioned scenes, due to lack of observability. We additionally find that the primitive’s covariance itself does not encode sensor uncertainties accurately, due to its shared use with the mapper, often recording confidence in the incorrect direction with respect to the camera ray.

To mitigate this, we propose a system where each Gaussian receives two covariances: one for tracking and one for mapping, while sharing the mean. This decoupling allows for the Gaussian to act as both a metric uncertainty model, and as a 3DGS primitive. We show that our method results in general tracking improvement across multiple datasets and camera types, reduced odometry drift, and good mapping quality, while maintaining real-time tracking performance. Finally, we evaluate the scene-level impact our method has on observability compared to other ICP-based 3DGS SLAM methods. To that end, we list our contributions:

• We introduce a dual-covariance parameterization using 3D Gaussians for SLAM tracking, where one covariance stores the sensor uncertainty measurements, and the other is used for mapping.

• We propose a joint tracking strategy which uses both the sensor-derived covariances for ICP tracking, and stable landmark Gaussians as Kanade-Lucas-Tomasi (KLT) anchors to improve observability.

• We show that our method is robust to various challenging conditions with reduced odometry drift.

## II. RELATED WORK

Tracking in 3DGS SLAM was initially based on photometric methods [3], [2], [7] which used the rasterized Gaussian appearance to optimize for pose. These methods produced high quality maps but often had relatively low FPS, making them largely unsuitable for real-time tracking. Methods such as Photo-SLAM [4] and GS-ICP SLAM [5] introduced the use of classical tracking methods for high speed tracking. The ICP based methods were significantly faster compared to the earlier photometric tracking, but various works noted issues with geometry [8], observability [6] and performance under RGB-D sensor noise [5]. We argue that these issues are partly rooted in the use of the render-optimized Gaussian shape as the tracking covariance.

Uncertainty weighted registration proposed by G-ICP [9] allows for arbitrary covariances to weight the point-cloud registration. These covariances can be derived from RGB-D sensor uncertainty estimates [10], [11], yet the question remains on how the primitive should be modeled in the 3DGS case, when it is also optimized by photometric error.

Observability and degeneracy are long-standing problems in LiDAR SLAM for geometrically self-similar environments where some pose directions are left underconstrained, leading to degeneracy and tracking failure [12], [13]. This is typically diagnosed from the registration Hessian [14]. Recently, [6] showed that ICP-based 3DGS SLAM fails similarly in structure-less scenes, attributing this to the mapper flattening the shared Gaussian covariances, and proposed detecting and mitigating the resulting degeneracy.

We distinguish two sources of this ill-conditioning. The first is weighting: mapper flattened covariances assign spurious precision along surface normals, which a covariance reflecting actual sensor precision can correct. The second is geometric: when the scene itself lacks structure in some direction, as on a single plane, no reweighting of depth residuals can recover it, and so additional information is required. Classical RGB-D tracking systems supply this by combining geometric alignment with visual information, either as sparse feature matches [15] or dense photometric error [16]. We therefore address the first problem with the sensor-derived tracking covariance, the second with KLT anchors to Gaussian primitives.

## III. METHOD

## A. Preliminaries

Let $a _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ be the frame’s points in camera coordinates and $T = ( R , t ) \in \mathrm { S E } ( 3 )$ the camera-to-world pose.

Generalized-ICP [9] minimizes the residual

$$
E ( T ) = \sum _ { i } d _ { i } ^ { \top } \big ( C _ { \pi ( i ) } ^ { B } + R C _ { i } ^ { A } R ^ { \top } \big ) ^ { - 1 } d _ { i } ,\tag{1}
$$

where $d _ { i } \ : = \ : b _ { \pi ( i ) } \ : - \ : ( R a _ { i } \ : + \ : t )$ , and $C ^ { A }$ and $C ^ { B }$ are the metric covariances of the point positions and π is the nearestneighbour association.

3DGS primitives $\mathcal { G } _ { j } ~ = ~ \{ \mu _ { j } , \Sigma _ { r , j } , c _ { j } , \sigma _ { j } \}$ are used by GS-ICP SLAM [5] and related methods [8], [6], [17] by performing registration between the map Gaussians $b _ { j } = \mu _ { j }$ $C _ { j } ^ { B } ~ = ~ \mathcal { Z } ( \Sigma _ { r , j } )$ and $C _ { i } ^ { A } ~ = ~ \mathcal { Z } ( Q _ { i } )$ , where $\Sigma _ { r , j }$ is the rendering covariance, $Q _ { i }$ is the scatter of the k nearest frame points and a regularizer $\mathcal { Z } ( \Sigma ) = \Sigma / \lambda _ { 2 } ( \Sigma )$ . Although $Q _ { i }$ encodes the local geometric surface distribution, the map covariances are also initialized from the KNN scatter. In practice, we find that the mapper’s modification of the Gaussian mean $\mu _ { j }$ sits well below depth characteristic noise for more than 97.5% of cases when measured on the TUM RGB-D dataset. In contrast, prior work [6] has found that the mapper tends to flatten $\Sigma _ { r , j }$ against the surface $( \lambda _ { 3 } / \lambda _ { 2 } \to 0 )$ thus the target term assigns a small scale along the surface normal, despite the absence of corresponding depth precision in the 3DGS rendering model.

## B. Tracking covariance

We introduce a second covariance $\Sigma _ { t , j }$ associated with each $\mathcal { G } _ { j }$ alongside $\Sigma _ { r , j }$ . The mapper optimises $\Sigma _ { r , j }$ for rendering, and $\Sigma _ { t , j }$ is derived from the sensor uncertainty at the point of observation. A point measured along the unit ray v at range z with lateral precision of s pixels has approximate covariance, in $\mathrm { m ^ { 2 } }$

$$
\begin{array} { r } { \mathcal { S } ( v , z ; s ) = \sigma _ { \perp } ^ { 2 } \left( I - v v ^ { \top } \right) + \sigma _ { \parallel } ^ { 2 } v v ^ { \top } , } \end{array}\tag{2}
$$

$$
\sigma _ { \perp } = \frac { z s } { f } , \qquad \sigma _ { \parallel } ^ { 2 } = \sigma _ { z } ^ { 2 } ( z ) ,\tag{3}
$$

with $f$ the focal length and $\sigma _ { z } ( z ) = A + B z ^ { 2 }$ the structuredlight range law [10]. At each keyframe (or at spawn), the in-view Gaussians with optical center $o _ { k }$ have their tracking covariance defined as

$$
\Sigma _ { t , j } = S ( u _ { j } , z _ { j } ; p ) , \qquad z _ { j } = \| \mu _ { j } - o _ { k } \| , \quad u _ { j } = ( \mu _ { j } - o _ { k } ) / z _ { j } ,\tag{4}
$$

with $p = 0 . 5 \mathrm { p x }$ , so the anchor is the noise of the most recent observation of the primitive.

## C. Depth residual

On the observation side, we set the uncertainty to encode the measurement uncertainty plus the ambiguity of the ICP max. association radius, which lies in the local tangent plane, thus its covariance is defined as:

$$
D _ { i } = \mathcal { S } \big ( \boldsymbol { v } _ { i } , \| \boldsymbol { a } _ { i } \| ; p _ { d } \big ) + r _ { i } ^ { 2 } \big ( \boldsymbol { I } - \boldsymbol { n } _ { i } \boldsymbol { n } _ { i } ^ { \top } \big ) , \qquad \boldsymbol { v } _ { i } = \boldsymbol { a } _ { i } / \| \boldsymbol { a } _ { i } \| ,\tag{5}
$$

where $p _ { d }$ is the pixel footprint of the depth sampling cell, $n _ { i }$ is the eigenvector of $Q _ { i }$ with the smallest eigenvalue, and the association radius r is the max. nearest-neighbour search distance. The depth pair is therefore defined as $d _ { i } =$ $\mu _ { \pi ( i ) } - T a _ { i }$ with $\Omega _ { i } = \Sigma _ { t , \pi ( i ) } + R D _ { i } R ^ { \top }$

![](images/9c0727038f62c74fa28321ce8300776c07b5a40288666acfb1f19f6bf5faa0db.jpg)  
Fig. 2: System Overview. (a) We define each Gaussian to consist of a photometrically optimized covariance $\Sigma _ { r }$ and a sensor measurement uncertainty covariance $\Sigma _ { t } .$ . (b) The depth residuals are defined by the standard G-ICP cost with the source covariance $D _ { i }$ defined as the observation’s uncertainty. (c) The image residuals are defined as KLT corners anchored to nearby Gaussians, with uncertainty defined by the anchor’s covariance.

## D. Image residual

Although the proposed second covariance models sensor uncertainty, it does not introduce constraints along directions lacking observability. Similar to classical RGB-D systems [15], [16], we utilize image information to supply the tangent-plane constraint that the depth observations lack. At each keyframe, a KLT corner back-projected to the world point $x ^ { w }$ near a primitive $j$ is placed on that primitive’s tangent plane,

$$
\ell = x ^ { w } + n _ { j } n _ { j } ^ { \top } \big ( \mu _ { j } - x ^ { w } \big ) ,\tag{6}
$$

where $n _ { j }$ is the minimum-scale axis of $\Sigma _ { r , j }$ , roughly corresponding to the surface normal. The landmark keeps its tangential position from the image, takes its normal coordinate from the map, and inherits the anchor’s covariance $\Sigma _ { t , j }$ . In frame $f$ its KLT-tracked pixel is back-projected at the measured depth $\hat { z } _ { \ell }$ to $^ { a \ell } ,$ with covariance

$$
\begin{array} { r } { S _ { \ell } = \mathcal { S } \big ( \boldsymbol { v } _ { \ell } , \hat { \boldsymbol { z } } _ { \ell } ; \boldsymbol { \sigma } _ { m } \big ) , \qquad \boldsymbol { v } _ { \ell } = \boldsymbol { a } _ { \ell } / \| \boldsymbol { a } _ { \ell } \| , } \end{array}\tag{7}
$$

where $\sigma _ { m } \ = \ 2 \mathsf { p x }$ is the matcher’s precision. This is (5) without the association term, because the track fixes the correspondence. The image pair is $\begin{array} { r } { e _ { \ell } ~ = ~ \ell - T a _ { \ell } } \end{array}$ with $\Omega _ { \ell } \ = \ \Sigma _ { t , j ( \ell ) } + R S _ { \ell } R ^ { \intercal }$ . In practice, as the matched set is slowly accumulated (and lost) over frames, this acts as a “best effort” frame-to-map tracker, where landmarks are retained until KLT loses the corner due to sufficient viewpoint, illumination change, or other image artifacts. This is sufficient because its main purpose is to improve observability, rather than act as a primary tracker. In practice, our ablations show this additional residual contributes to a general increase in tracking robustness when applied in conjunction with the tracking covariance.

## E. Estimation

The pose is found by minimizing the term

$$
E ( T ) = \sum _ { i } d _ { i } ^ { \top } \Omega _ { i } ^ { - 1 } d _ { i } + \sum _ { \ell } \rho _ { \delta } \left( e _ { \ell } ^ { \top } \Omega _ { \ell } ^ { - 1 } e _ { \ell } \right)\tag{8}
$$

which represents the joint ICP and image residual with $\rho _ { \delta }$ the Huber kernel, solved by Levenberg-Marquardt on SE(3).

## IV. EXPERIMENTS

## A. Implementation and Experimental Setup

Datasets and Metrics. We use TUM RGB-D [18], Scan-Net [19], Replica [20] datasets to evaluate our work. We evaluate with Umeyama SE(3) aligned RMSE Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) for measuring tracking accuracy, and map quality with PSNR, SSIM [21] and LPIPS [22]. We report all FPS as the average number of input frames processed over the entire sequence during SLAM tracking, excluding the input processing or post-hoc refinement time that some methods have.

Implementation Details. All of our evaluations were performed on a computer equipped with an i9-13900KF CPU and an RTX 4090 GPU. For reproduction, we evaluated GS-ICP SLAM [5], SGAD-SLAM [23], Spectral GS-SLAM [6], Photo-SLAM [4], MonoGS [2], SplaTAM [7] and ORB-SLAM3 [24] using their publicly released implementations and default configurations.

## B. Tracking Performance

We report our method’s tracking results on the standard TUM sequences reported by most 3DGS SLAM methods in Table I. Compared to other ‘Coupled’ methods, ours achieves the lowest ATE on average and lower than other methods on the fast-motion desk scene. Furthermore, ours has a significantly higher average FPS than the other two methods FeatureSLAM and SGAD-SLAM, which have comparable tracking performance to ours. Similar to ours, all the methods in the ‘Coupled’ section are based on ICP tracking. The two ‘Decoupled’ methods listed also have Bundle Adjustment (BA), while ours does not. We cap our method and GS-ICP SLAM at 30 FPS to evaluate ATE. We report the FPS, uncapped, as a separate run. G2S-ICP SLAM does not report their uncapped FPS.

Additionally, we report extended results covering a more diverse scene set from TUM RGB-D in Table II. These scenes cover a varying set of environments, including similar ones (desk2, fr1/xyz) to the ones in Table I, and more challenging ones (room, 360), testing fast motion/rotation.

TABLE I: ATE [cm] ↓, on the standard TUM RGB-D scenes. Methods indicated with † are capped at 30 FPS.
<table><tr><td>Type</td><td>Method</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>avg</td><td>FPS ↑</td></tr><tr><td rowspan="2">Decoupled</td><td>ORB-SLAM3 [24]</td><td>1.7</td><td>0.4</td><td>1.7</td><td>1.3</td><td>86.7</td></tr><tr><td>Photo-SLAM [4]</td><td>2.6</td><td>0.3</td><td>1.0</td><td>1.3</td><td>51.6</td></tr><tr><td rowspan="6">Coupled</td><td>GS-ICP SLAM [5] †</td><td>2.7</td><td>1.8</td><td>2.7</td><td>2.4</td><td>62.7</td></tr><tr><td>G2S-ICP SLAM [8] †</td><td>2.74</td><td>1.59</td><td>2.78</td><td>2.37</td><td></td></tr><tr><td>FeatureSLAM [17]</td><td></td><td></td><td></td><td>2.05</td><td>~5</td></tr><tr><td>SGAD-SLAM [23]</td><td>2.2</td><td>1.7</td><td>2.0</td><td>2.0</td><td>20.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours †</td><td>1.94</td><td>1.55</td><td>2.23</td><td>1.91</td><td>59.1</td></tr></table>

TABLE II: Extended TUM RGB-D evaluation, ATE [cm] ↓.
<table><tr><td>Type</td><td>Method</td><td>desk2</td><td>fr1/xyz</td><td>room</td><td>360</td><td>rpy</td><td>s_tex</td><td>avg</td></tr><tr><td>Decoupled</td><td>ORB-SLAM3 [24] Photo-SLAM [4]</td><td>2.23 2.77</td><td>1.06 1.00</td><td>6.47 6.95</td><td>20.93 21.6</td><td>0.43 0.31</td><td>0.98 0.99</td><td>5.35 5.60</td></tr><tr><td></td><td>GS-ICP SLAM [5]</td><td>13.34</td><td>1.43</td><td>18.08</td><td>42.30</td><td>2.51</td><td>2.14</td><td>13.30</td></tr><tr><td>Coupled</td><td>Spectral GS-SLAM [6]</td><td>12.18</td><td>1.44</td><td>17.35</td><td>29.24</td><td>2.34</td><td>3.80</td><td>11.06</td></tr><tr><td></td><td>SGAD-SLAM [23]</td><td>7.21</td><td>1.40</td><td>27.87</td><td></td><td></td><td>2.13</td><td>22.33</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>80.14</td><td>15.19</td><td></td><td></td></tr><tr><td></td><td>Ours</td><td>3.83</td><td>1.47</td><td>16.23</td><td>17.13</td><td>2.24</td><td>1.82</td><td>7.12</td></tr></table>

We also test s tex as a reference scene containing both rich visual texture and geometry. As not all methods in Table I have released implementation, we report results on the available ones.

Our method is robust across challenging scenes containing a lack of visual features (notexture) and a lack of geometric structure (nostructure). We partitioned the results in Table III into methods that use depth for tracking only (top), and methods which have image-level signals (bottom). Ours maintains tracking robustly across both conditions, without any detection mechanism like [6]. ‘X’ denotes failed or partial tracking due to late initialization.

TABLE III: TUM RGB-D no-structure/no-texture sequences, ATE [cm] ↓.
<table><tr><td>Method</td><td>nos_tex_near</td><td>nos_tex_far</td><td>s_notex_near s_notex_far</td><td></td></tr><tr><td>GS-ICP SLAM [5]</td><td>194.94</td><td>116.16</td><td>1.55</td><td>6.07</td></tr><tr><td>SGAD-SLAM [23]</td><td>195.08</td><td>116.17</td><td>1.35</td><td>1.94</td></tr><tr><td>Photo-SLAM [4]</td><td>2.27</td><td>5.36</td><td>X</td><td>X</td></tr><tr><td>ORB-SLAM3 [24]</td><td>2.03</td><td>10.07</td><td>X</td><td>X</td></tr><tr><td>Spectral GS-SLAM [6]</td><td>7.79</td><td>16.85</td><td>1.55</td><td>2.19</td></tr><tr><td>Ours</td><td>1.67</td><td>9.00</td><td>1.86</td><td>4.93</td></tr></table>

Our results on a subset of ScanNet are reported in Table IV, where we achieve stable tracking performance compared to other methods, performing on par with methods like ORB-SLAM3 with both BA and loop closure, and dense photometric methods like SplaTAM and Gaussian-SLAM. Multiple scenes, e.g. 0169 have opportunities for loop closure. For SGAD-SLAM, we reproduced the results from the open-sourced code, using the authors’ default perscene tuning of the depth truncation and KNN initialization.

We report our results on Replica in Table V, a simulated indoor scene without depth sensor noise, a near ideal case for ICP. As our covariance model described in Section III-B requires a small non-zero depth noise floor for numerical stability, and this additional noise unsurprisingly results in marginally worse performance than other ICP methods, though it still outperforms other non-ICP 3DGS methods.

TABLE IV: ScanNet evaluation, ATE [cm] ↓.
<table><tr><td>Method</td><td>0000</td><td>0059</td><td>0106</td><td>0169</td><td>0181</td><td>0207</td><td>avg</td></tr><tr><td>ORB-SLAM3 [24]</td><td>8.38</td><td>7.11</td><td>9.57</td><td>8.45</td><td>67.68</td><td>7.74</td><td>18.2</td></tr><tr><td>SplaTAM [7]</td><td>12.8</td><td>10.1</td><td>17.7</td><td>12.1</td><td>11.1</td><td>7.5</td><td>11.9</td></tr><tr><td>Gaussian-SLAM [25]</td><td>24.8</td><td>8.6</td><td>11.3</td><td>14.6</td><td>18.7</td><td>14.4</td><td>15.4</td></tr><tr><td>SGAD-SLAM [23] (reported)</td><td>11.9</td><td>6.4</td><td>5.3</td><td>8.5</td><td>10.3</td><td>4.7</td><td>7.9</td></tr><tr><td>GS-ICP SLAM [5]</td><td>36.5</td><td>22.17</td><td>6.04</td><td>31.26</td><td>15.8</td><td>10.47</td><td>20.4</td></tr><tr><td>SGAD-SLAM [23] </td><td>177.8</td><td>55.1</td><td>11.1</td><td>19.1</td><td>19.6</td><td>9.3</td><td>48.7</td></tr><tr><td>Ours</td><td>15.63</td><td>8.67</td><td>5.59</td><td>6.89</td><td>13.06</td><td>7.83</td><td>9.61</td></tr></table>

‡ indicates reproduced results by running the official code.

TABLE V: Replica evaluation, ATE [cm] ↓.
<table><tr><td>Method</td><td>r0</td><td>rl</td><td>r2</td><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>avg</td></tr><tr><td>ORB-SLAM3 [24]</td><td>0.56</td><td>0.43</td><td>0.40</td><td>0.90</td><td>0.67</td><td>1.55</td><td>0.96</td><td>2.29</td><td>0.97</td></tr><tr><td>SplaTAM [7]</td><td>0.31</td><td>0.40</td><td>0.29</td><td>0.47</td><td>0.27</td><td>0.29</td><td>0.32</td><td>0.55</td><td>0.36</td></tr><tr><td>GS-SLAM [3]</td><td>0.48</td><td>0.53</td><td>0.34</td><td>0.52</td><td>0.41</td><td>0.59</td><td>0.46</td><td>0.70</td><td>0.50</td></tr><tr><td>MonoGS [2]</td><td>0.48</td><td>0.36</td><td>0.34</td><td>0.44</td><td>0.52</td><td>0.23</td><td>0.16</td><td>2.53</td><td>0.63</td></tr><tr><td>G2S-ICP SLAM [8]</td><td>0.14</td><td>0.16</td><td>0.10</td><td>0.19</td><td>0.12</td><td>0.16</td><td>0.17</td><td>0.20</td><td>0.15</td></tr><tr><td>FeatureSLAM [17]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.15</td></tr><tr><td>GS-ICP SLAM [5] ‡</td><td>0.14</td><td>0.16</td><td>0.11</td><td>0.18</td><td>0.12</td><td>0.17</td><td>0.18</td><td>0.20</td><td>0.16</td></tr><tr><td>SGAD-SLAM [23] ↓</td><td>0.15</td><td>0.15</td><td>0.09</td><td>0.16</td><td>0.12</td><td>0.93</td><td>0.27</td><td>0.19</td><td>0.26</td></tr><tr><td>Ours</td><td>0.16</td><td>0.17</td><td>0.36</td><td>0.16</td><td>0.13</td><td>0.18</td><td>0.20</td><td>0.22</td><td>0.20</td></tr></table>

‡ indicates reproduced results by running the official code.

We report our method’s relative pose error in Table VI, which has the lowest overall error among the ICP-based methods, with drift characteristics more similar to ORB-SLAM3 than to the other ICP methods in both relative translation and relative rotation error over a 1 second window. This highlights our method’s robustness to drift, due to the dedicated covariance used for tracking rather than reusing the mapping covariance.

TABLE VI: Extended TUM RGB-D evaluation, relative pose error over 1 s without trajectory alignment: translation RMSE [cm] ↓ / rotation RMSE [deg] ↓.
<table><tr><td>Type</td><td>Method</td><td>desk2</td><td>fr1/xyz</td><td>room</td><td>360</td><td>rpy</td><td>s_tex</td><td>avg</td></tr><tr><td rowspan="2">Decoupled</td><td>ORB-SLAM3 [24]</td><td>3.59/2.04</td><td>1.72/1.15</td><td>5.32/2.13</td><td>18.12/4.27</td><td>0.36/0.34</td><td>1.03/0.63</td><td>5.02/1.76</td></tr><tr><td>Photo-SLAM [4]</td><td>3.35/1.88</td><td>1.54/1.00</td><td>4.30/1.80</td><td>14.41/3.40</td><td>0.29/0.32</td><td>0.96/0.59</td><td>4.14/1.50</td></tr><tr><td rowspan="3">Coupled</td><td>GS-ICP SLAM [5]</td><td>9.64/6.36</td><td>2.51/1.78</td><td>7.66/3.94</td><td>41.99/8.55</td><td>1.34/0.74</td><td>2.39/1.00</td><td>10.92/3.73</td></tr><tr><td>SGAD-SLAM [23]</td><td>10.96/6.46</td><td>2.50/1.74</td><td>13.23/4.37</td><td>62.94/18.68</td><td>5.17/1.25</td><td>1.32/1.00</td><td>16.02/5.58</td></tr><tr><td>Ours</td><td>5.36/3.67</td><td>2.61/1.54</td><td>5.74/2.94</td><td>8.67/3.67</td><td>0.84/0.50</td><td>1.11/0.89</td><td>4.05/2.20</td></tr></table>

To evaluate our method’s robustness to drift over long sequences, we recorded two sequences in an outdoor scene pictured in Figure 3. These sequences were recorded on a wheeled platform and handheld respectively, using a D435i RGB-D camera. The pseudo-GT trajectories were generated offline using COLMAP [26] with scale from the RGB-D sensor. We run our method, with the other ICP-based baselines (GS-ICP SLAM, SGAD-SLAM), dense photometric methods (SplaTAM, MonoGS), and ORB-SLAM3 as reference. Despite having no bundle adjustment, our method reports the lowest drift of all evaluated 3DGS SLAM methods outperformed only by ORB-SLAM3, while simultaneously constructing a 3DGS map in real-time. We attempted a few other methods (e.g., Photo-SLAM) but there was insufficient memory to complete the sequence.

TABLE VII: Rendering performance of real-time 3DGS SLAM systems on TUM RGB-D
<table><tr><td></td><td>Metric</td><td>desk</td><td>fr2/xyz</td><td>office</td><td>desk2</td><td>fr1/xyz</td><td>room</td><td>360</td><td>rpy</td><td>s_tex</td><td>nos_tex</td><td>avg</td></tr><tr><td rowspan="3">GS-ICP SLAM</td><td>PSNR↑</td><td>17.99</td><td>23.60</td><td>20.97</td><td>15.33</td><td>19.57</td><td>17.73</td><td>18.05</td><td>22.25</td><td>22.55</td><td>17.50</td><td>19.55</td></tr><tr><td>SSIM↑</td><td>0.710</td><td>0.836</td><td>0.764</td><td>0.674</td><td>0.739</td><td>0.697</td><td>0.727</td><td>0.815</td><td>0.788</td><td>0.744</td><td>0.749</td></tr><tr><td>LPIPS↓</td><td>0.293</td><td>0.136</td><td>0.222</td><td>0.363</td><td>0.252</td><td>0.316</td><td>0.349</td><td>0.170</td><td>0.227</td><td>0.516</td><td>0.284</td></tr><tr><td rowspan="3">Spectral GS-SLAM</td><td>PSNR↑</td><td>18.52</td><td>22.13</td><td>18.18</td><td>15.66</td><td>19.60</td><td>17.42</td><td>16.38</td><td>19.74</td><td>19.67</td><td>24.98</td><td>19.23</td></tr><tr><td>SSIM↑</td><td>0.724</td><td>0.819</td><td>0.714</td><td>0.680</td><td>0.740</td><td>0.689</td><td>0.703</td><td>0.778</td><td>0.751</td><td>0.849</td><td>0.745</td></tr><tr><td>LPIPS↓</td><td>0.279</td><td>0.163</td><td>0.303</td><td>0.356</td><td>0.250</td><td>0.323</td><td>0.383</td><td>0.239</td><td>0.276</td><td>0.182</td><td>0.275</td></tr><tr><td rowspan="3">Photo-SLAM</td><td>PSNR↑</td><td>21.22</td><td>21.85</td><td>22.64</td><td>18.26</td><td>23.29</td><td>18.14</td><td>15.44</td><td>20.53</td><td>25.47</td><td>26.95</td><td>21.38</td></tr><tr><td>SSIM↑</td><td>0.752</td><td>0.760</td><td>0.774</td><td>0.694</td><td>0.815</td><td>0.657</td><td>0.610</td><td>0.719</td><td>0.857</td><td>0.865</td><td>0.750</td></tr><tr><td>LPIPS↓</td><td>0.225</td><td>0.174</td><td>0.155</td><td>0.325</td><td>0.167</td><td>0.316</td><td>0.431</td><td>0.203</td><td>0.144</td><td>0.143</td><td>0.228</td></tr><tr><td rowspan="3">Ours</td><td>PSNR↑</td><td>18.87</td><td>24.52</td><td>21.51</td><td>18.04</td><td>20.50</td><td>18.24</td><td>18.51</td><td>22.75</td><td>24.47</td><td>26.51</td><td>21.39</td></tr><tr><td>SSIM↑</td><td>0.731</td><td>0.859</td><td>0.781</td><td>0.725</td><td>0.762</td><td>0.711</td><td>0.755</td><td>0.829</td><td>0.827</td><td>0.865</td><td>0.784</td></tr><tr><td>LPIPS↓</td><td>0.267</td><td>0.114</td><td>0.201</td><td>0.286</td><td>0.216</td><td>0.300</td><td>0.321</td><td>0.151</td><td>0.181</td><td>0.157</td><td>0.219</td></tr></table>

![](images/127cba6e8b2148d0614a55baa8343a0d887f9790e2b34fefac1add06c9c44bcd.jpg)  
Fig. 3: Performance on outdoor scenes. Ours has the lowest drift of all evaluated 3DGS SLAM methods in both the wheeled-platform (left) and handheld settings (right) on up to 70m+ length trajectories.

## C. Mapping Performance

We report mapping performance by scene in Table VII. Our method performs similarly to Photo-SLAM in PSNR and has the best SSIM and LPIPS of all real-time methods evaluated, in part due to the robust tracking, which allows for the mapping to take place from accurate viewpoints. We also provide qualitative renders of some of the methods in Figure 4.

![](images/385051eb50998ef013e7deb0d390c11824e0962f22dca3fb9c0044e989aa7f05.jpg)  
Fig. 4: Qualitative render quality of our method compared to some baselines. We render scenes from TUM RGB-D, ScanNet, and from our custom sequences.

## D. Observability analysis

The proposed covariance is only locally derived from the RGB-D measurement model, whereas pose observability emerges only after aggregating information from all visible primitives. We therefore examine the spectrum of the Hessian $\begin{array} { r } { \mathbf { \bar {  { H } } } = \sum { J ^ { \top } \boldsymbol { \Omega } ^ { - 1 } J } } \end{array}$ using its condition number [14] as a proxy for observability. Replacing the rendering covariance with $\Sigma _ { t }$ changes the uncertainty weighting of individual geometric observations, while the image constraints provide complementary information. Figure 5 shows that $\Sigma _ { t }$ substantially improves Hessian conditioning even without image constraints, indicating that the improvement cannot be attributed solely to KLT. In str notex, the condition number changes little because the scene already provides strong geometric constraints (from the perpendicular walls and floors). In scenes with weaker geometric constraints, adding KLT constraints further improves the conditioning by contributing complementary image-based information.

![](images/1605ef29cedf665f093450f9f8799baa85e659111ae277f65079cd0968d2a9d7.jpg)  
Fig. 5: Comparison of condition number across ICP methods. Ours reduces the condition number in general, indicating that the optimization is more well-constrained.

## E. Ablation studies.

In Table VIII, we ablate our two main contributions, the tracking covariance $\Sigma _ { t }$ in Section III-B and the KLT image residuals in Section III-D. “Shared $\operatorname { c o v } ^ { \mathfrak { N } }$ refers to the baseline’s method where $\Sigma _ { t } = \Sigma _ { r }$ . We show that the image residuals are inconsistently helpful, as fusing them with the shared covariance gives worse tracking in some cases, even with multiple attempted weighting values. On the other hand, $\Sigma _ { t }$ alone improves ATE across real scenes, showing that the representation change does indeed contribute to better tracking. Lastly, our full method achieves the overall lowest ATE, and in particular helps the nos tex scene since the scene is otherwise unobservable from depth constraints alone.

TABLE VIII: Ablation of our method, ATE [cm] ↓.
<table><tr><td>Configuration</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>fr3/nos_tex</td><td>0106</td><td>0169</td></tr><tr><td>Shared cov.</td><td>2.66</td><td>1.78</td><td>2.70</td><td>194.96</td><td>6.04</td><td>19.32</td></tr><tr><td>Shared  $\mathrm { c o v . + K L T } \ ( w { = } 1 )$ </td><td>3.13</td><td>1.82</td><td>11.73</td><td>4.08</td><td>5.73</td><td>20.68</td></tr><tr><td>Shared  $\mathrm { c o v . + K L T } \ ( w { = } . 1 )$ </td><td>2.56</td><td>1.58</td><td>4.20</td><td>3.76</td><td>8.45</td><td>9.21</td></tr><tr><td>Shared  $\mathrm { c o v . } + \mathrm { K L T } \ ( w { = } 1 0 )$ </td><td>2.98</td><td>2.16</td><td>11.91</td><td>6.07</td><td>17.32</td><td>42.09</td></tr><tr><td>Σt only</td><td>2.47</td><td>1.60</td><td>2.22</td><td>192.38</td><td>5.68</td><td>14.78</td></tr><tr><td>Full</td><td>1.94</td><td>1.55</td><td>2.23</td><td>1.67</td><td>5.59</td><td>6.89</td></tr></table>

## V. LIMITATIONS

Although our method is able to robustly track across diverse scene types, limitations exist in terms of large-scale scenes, where our method lacks bundle adjustment to ensure long-term consistency. Like other ICP methods, our work is also sensitive to high depth error, which for many sensors increases with distance. This means that for outdoor scenes or where the majority of the image content is far away, tracking would be less accurate. Future work may consider using the landmarks or covariances for some form of bundle adjustment without trading off too much speed.

## VI. CONCLUSION

We proposed Dual Covariance Gaussian Splatting SLAM, a method in which each Gaussian primitive for mapping carries a second covariance used to store the measurement uncertainty. We showed that our method results in improved tracking performance across the board, including in difficult scenes with fast handheld motion, lack of structure or visual features, and reduced odometry drift. This is done while maintaining real-time performance, enabling deployment on robot platforms for online tracking and mapping.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, “3D Gaussian Splatting for Real-Time Radiance Field Rendering,” ACM Trans. Graph., vol. 42, no. 4, pp. 139:1–139:14, Jul. 2023.

[2] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, “Gaussian Splatting SLAM,” 2024, pp. 18 039–18 048.

[3] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting,” 2024, pp. 19 595–19 604.

[4] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular Stereo and RGB-D Cameras,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2024, pp. 21 584–21 593.

[5] S. Ha, J. Yeon, and H. Yu, “RGBD GS-ICP SLAM,” in Computer Vision – ECCV 2024, A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds., vol. 15094. Cham: Springer Nature Switzerland, 2025, pp. 180–197.

[6] E. B. W. Tan, S.-K. Lam, and D. Zhang, “Spectral GS-SLAM: Observability-Aware, Degeneracy-Robust Tracking for Real-Time 3D Gaussian Splatting SLAM,” Jun. 2026, arXiv:2606.21258 [cs.RO]. [Online]. Available: http://arxiv.org/abs/2606.21258

[7] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, “SplaTAM: Splat Track & Map 3D Gaussians for Dense RGB-D SLAM,” 2024, pp. 21 357–21 366.

[8] G. Pak, H. M. Cho, and E. Kim, “G2S-ICP SLAM: Geometry-aware Gaussian Splatting ICP SLAM,” Jul. 2025, arXiv:2507.18344 [cs]. [Online]. Available: http://arxiv.org/abs/2507.18344

[9] A. V. Segal, D. Hahnel, and S. Thrun, “Generalized-ICP,” in¨ Robotics: Science and Systems, 2009.

[10] C. V. Nguyen, S. Izadi, and D. Lovell, “Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking,” in Visualization & Transmission 2012 Second International Conference on 3D Imaging, Modeling, Processing, Oct. 2012, pp. 524–530, iSSN: 1550-6185. [Online]. Available: https://ieeexplore.ieee.org/document/6375037

[11] K. Khoshelham and S. O. Elberink, “Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications,” Sensors, vol. 12, no. 2, pp. 1437–1454, Feb. 2012. [Online]. Available: https://www.mdpi.com/1424-8220/12/2/1437

[12] T. Tuna, J. Nubert, Y. Nava, S. Khattak, and M. Hutter, “X-ICP: Localizability-Aware LiDAR Registration for Robust Localization in Extreme Environments,” IEEE Transactions on Robotics, vol. 40, pp. 452–471, 2024.

[13] H. Yue, Q. Xu, F. Chen, J. Pan, and W. Chen, “LP-ICP: General Localizability-Aware Point Cloud Registration for Robust Localization in Extreme Unstructured Environments,” 2025, version Number: 3.

[14] J. Zhang, M. Kaess, and S. Singh, “On degeneracy of optimizationbased state estimation problems,” in 2016 IEEE International Conference on Robotics and Automation (ICRA), May 2016, pp. 809–816.

[15] P. Henry, M. Krainin, E. Herbst, X. Ren, and D. Fox, “RGB-D Mapping: Using Depth Cameras for Dense 3D Modeling of Indoor Environments,” in Experimental Robotics: The 12th International Symposium on Experimental Robotics, O. Khatib, V. Kumar, and G. Sukhatme, Eds. Berlin, Heidelberg: Springer, 2014, pp. 477–491.

[16] T. Whelan, R. F. Salas-Moreno, B. Glocker, A. J. Davison, and S. Leutenegger, “ElasticFusion,” International Journal of Robotics Research, vol. 35, no. 14, pp. 1697–1716, Dec. 2016. [Online]. Available: https://doi.org/10.1177/0278364916669237

[17] C. Thirgood, O. Mendez, E. Ling, J. Storey, and S. Hadfield, “FeatureSLAM: Feature-enriched 3D gaussian splatting SLAM in real time,” Mar. 2026, arXiv:2601.05738 [cs.CV]. [Online]. Available: http://arxiv.org/abs/2601.05738

[18] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A Benchmark for the Evaluation of RGB-D SLAM Systems,” in Proc of the International Conference on Intelligent Robot Systems (IROS), Oct. 2012.

[19] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Niessner, “ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Honolulu, HI: IEEE, Jul. 2017, pp. 2432–2443. [Online]. Available: https://ieeexplore.ieee.org/document/8099744/

[20] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, A. Clarkson, M. Yan, B. Budge, Y. Yan, X. Pan, J. Yon, Y. Zou, K. Leon, N. Carter, J. Briales, T. Gillingham, E. Mueggler, L. Pesqueira, M. Savva, D. Batra, H. M. Strasdat, R. D. Nardi, M. Goesele, S. Lovegrove, and R. Newcombe, “The Replica Dataset: A Digital Replica of Indoor Spaces,” arXiv preprint arXiv:1906.05797, 2019.

[21] Zhou Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, Apr. 2004.

[22] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2018.

[23] P. Hu and Z. Han, “SGAD-SLAM: Splatting gaussians at adjusted depth for better radiance fields in RGBD SLAM,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2026.

[24] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM3: An Accurate Open-Source Library for´ Visual, Visual–Inertial, and Multimap SLAM,” IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874–1890, Dec. 2021.

[25] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting,”

[26] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”¨ in Conference on computer vision and pattern recognition (CVPR), 2016.

Mar. 2024, arXiv:2312.10070 [cs.CV]. [Online]. Available: http://arxiv.org/abs/2312.10070