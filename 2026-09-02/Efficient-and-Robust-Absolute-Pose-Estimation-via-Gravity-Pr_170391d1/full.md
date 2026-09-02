# Efficient and Robust Absolute Pose Estimation via Gravity-Prior-Driven Transformation Decoupling and Pose Refinement

Hu Cao <sup>1†</sup>, Qianyi Yang <sup>2†</sup>, Xinyi Li <sup>2</sup>, Jiong Liu <sup>2</sup>, Yinlong Liu <sup>3∗</sup>, Alois Knoll <sup>2</sup>, Fellow, IEEE

Abstract—Estimation of the absolute pose of an object is an essential task for various robotic applications. Recently, incorporating gravity direction as prior information has emerged as a popular approach to simplify absolute pose estimation. However, developing a robust and efficient algorithm to solve this challenging problem remains a difficult question due to large amounts of mismatches. In addition, obtaining an accurate pose solution from selected inlier correspondences with gravity prior is still a research gap. In this paper, we propose a novel transformation strategy that exploits geometric relations derived from the gravity prior. Through transformation decoupling, the original 6 degrees of freedom (DoF) absolute pose estimation problem is simplified into a 4-DoFs problem: 1-DoF for the rotation angle and 3-DoFs for translation, significantly improving the efficiency. For the 1-DoF rotation angle, we apply a onedimensional global voting algorithm for optimal estimation. Once the optimal rotation is obtained, the mismatched correspondences are preliminarily filtered, and translation estimation, a linear problem, can be easily solved. Furthermore, to obtain accurate pose results, we introduce a novel pose refinement algorithm to enhance the accuracy of both rotation and translation. Extensive experiments on synthetic data and three publicly available realworld datasets (TUM RGB-D, ETH3D, and RobotCar) demonstrate that the proposed method achieves stronger performance compared to existing state-of-the-art (SOTA) approaches. To further validate our method, we integrated it into ORB-SLAM2. The results on the KITTI dataset show it effectively reduces drift and improves trajectory alignment during relocalization. The source code will be released upon acceptance.

Index Terms—Absolute pose estimation, transformation decoupling, global voting, pose refinement, robust estimation.

## I. INTRODUCTION

BSOLUTE pose estimation is regarding calculating the camera pose from 2D image features and their 3D coun  
terparts, which is a fundamental task for many applications of   
computer vision and robotics, such as augmented reality [1],   
[2], visual odometry [3], [4], and visual localization [5], [6].

Technologically speaking, the task of estimating absolute pose involves determining the camera’s orientation and position relative to a 3D reference world through the matching of 2D-3D features [7], [8] (see Fig. 1). It must be acknowledged that the pose can be estimated if the 2D and 3D correspondences are perfectly obtained, which is the wellknown Perspective-n-Point (PnP) problem [9]. However, it is almost impossible to have perfect 2D-3D feature matches in real applications [10], and the input feature correspondences are typically contaminated by mismatches, which are also known as outliers [11]. Unfortunately, precisely estimating the parameters from outlier-contaminated inputs is extremely hard (specifically, NP-hard [12]), and even a small percentage of outliers might lead to wildly incorrect results [13].

![](images/3baa28c92d73cfc8640b965d1fec4f6b494a1c0c9a7e6772e015c018bd2d5248.jpg)  
Fig. 1: The illustration of the absolute pose estimation problem. Technically, the objective is to estimate the rigid pose, comprising the rotation matrix R and the translation vector t, from 2D-3D correspondences.

Notably, in modern robotic sensor systems, inertial measurement units (IMUs) are typically integrated and are capable of providing prior knowledge of the gravity direction [14]. Understandably, the prior gravity can reduce the difficulty and help estimate the absolute pose of the camera [15], [16]. Additionally, the direction of gravity can be estimated by identifying vertical vanishing points or directions [17], [18], particularly in specific scenarios such as structured environments [19]. Nonetheless, researchers utilize this prior directional information to enhance various tasks, including absolute pose estimation [10], [20], relative pose estimation [21], [22], simultaneous localization and mapping (SLAM) [23], [24], and panoramic image stitching [25]. As a matter of fact, leveraging the gravity prior, certain parameters in camera orientation estimation can be simplified by applying geometric constraints [20]. Technically, the full rigid pose estimation problem is reduced from 6-DoFs (three for rotation and three for translation) to 4-DoFs, as the rotation matrix is constrained to a single DoF for rotation about the specified axis [26].

Accordingly, we focus on 4-DoFs absolute pose estimation under the assumption that gravity directions are predetermined in this paper. Unfortunately, as far as we know, there is no effective and efficient way to remove outliers in advance [12]. Therefore, we target solving the absolute pose estimation with

outlier-contaminated inputs.

More specifically, as illustrated in Fig. 2, the absolute pose estimation problem is commonly referred to as the well-known PnP problem [27], [28], and to estimate the absolute pose, the 2D-3D feature correspondences should be constructed [29]. However, due to the limited performance (mismatches) of current feature matching methods [30], [31] and challenges such as partial overlap and noisy data, the obtained 2D-3D correspondences often contain a significant proportion of outliers, which lead to serious incorrect results [31]. In the practical applications, the de facto standard method to suppress the outlier is RANdom SAmple Consensus (RANSAC) method [11]. However, as a non-deterministic algorithm, RANSAC only produces satisfactory results with a certain probability [32]. In other words, RANSAC method may provide an unsatisfactory pose solution occasionally [33]. Besides, its runtime grows exponentially with the outlier rate [34].

For robotic applications, achieving an optimal solution with extreme robustness is essential, particularly in safety-critical scenarios [35]–[38]. To overcome this limitation, globally optimal and deterministic algorithms such as global voting [39], enumeration [23], and branch-and-bound (BnB) [40], [41] have been employed. However, to search for the optimal solution, the time complexity of these highly robust algorithms increases substantially with the dimensionality of the optimization problem [10]. It should be mentioned that full rigid pose estimation, which is 6-DoFs, can not be considered a low-dimensional problem. In contrast, estimating the 4DOFsgravity-known pose will be more efficient.

It is worth noting that to accelerate the globally optimal solution, transformation decoupling [10], [26], [42]–[44] has been recently proposed. Specifically, the strategy uses the geometric constraints of the rigid motion to decouple the original full estimation problem into sequential sub-problems, which can be solved more efficiently in lower-dimensional parameter spaces. Consequently, the final solution can be assembled by the optimal solutions to the sub-problems [26]. This smart strategy has been verified to be effective in many pose estimation problems [26]. However, for the gravity-known pose estimation problem, the existing decoupling-based method [10] still has several shortcomings. Specifically, (1) it solves the problem by sequentially solving translation estimation (3- DoF) and rotation angle estimation (1-DoF). Unfortunately, estimating the 3-DoF translation is still not an easily solved low-dimensional problem, which retains high computational complexity. (2) In addition, the existing approach [10] only returns the inlier 2D-3D point correspondences, and it lacks a dedicated non-minimal refinement solution for the 4DoF gravity-constrained pose estimation problem, which still leaves a research gap [45].

Broadly speaking, existing gravity-aware 4-DoFs pose solvers have made important advances in reducing the dimensionality of the PnP problem, but two critical limitations remain unaddressed. First, mainstream methods adopt a translation-first decoupling strategy, which solves the 3-DoFs translation sub-problem with unfiltered outlier-contaminated correspondences, failing to fully exploit the low-dimensional advantage of the gravity-constrained rotation space and suffering from severe performance degradation under high outlier rates. Second, there is a lack of dedicated non-minimal refinement schemes tailored for the 4-DoFs gravity-constrained space, with most works relying on general PnP refinement that cannot fully leverage the gravity prior for further accuracy improvement. In this work, we address these limitations with a holistic rotation-first decoupling framework, consisting of three core original components: (1) a novel rotation-prioritized transformation decoupling strategy; (2) a 1D global voting scheme for robust rotation estimation and simultaneous outlier filtering; (3) a dedicated non-minimal refinement algorithm based on hidden variable resultant for high-precision 4-DoFs pose optimization. The following experiments systematically validate the independent contribution of each component, as well as the synergistic performance gain of our holistic framework.

In conclusion, our main contributions can be summarized as follows:

• We establish a novel rotation-prioritized geometric relationship based on the gravity prior, which reduces the absolute pose estimation problem from 6-DoFs to 4- DoFs, and fundamentally improves the solving efficiency by reversing the conventional translation-first decoupling order.

• We design a 1D global voting algorithm tailored for the gravity-constrained 1DoF rotation space, which achieves deterministic, efficient rotation estimation and simultaneous outlier filtering, significantly enhancing the robustness of the method under high outlier rates.

• We propose a novel non-minimal refinement algorithm based on hidden variable resultant for the 4-DoFs gravityaware pose estimation problem, which fills the research gap of high-precision optimization for this specific scenario and further improves the final pose accuracy.

• Extensive experiments on synthetic data, three real-world datasets, and a SLAM system integration validate that our method outperforms existing SOTA approaches, and can effectively reduce trajectory drift and improve relocalization performance in practical applications.

## II. RELATED WORK

## A. Outlier-free Absolute Pose Estimation

Given 2D-3D correspondences without mismatches, numerous algorithms have been extensively studied [46]. The earliest known solution was introduced by Grunert [47] in 1841, which addressed the minimal case of three projective point feature correspondences (P3P) [48]. Since then, various closed-form solutions have been proposed, often formulating the problem as finding the roots of a polynomial equation [49], [50]. For more than minimal inputs, i.e., PnP problem, a straightforward approach is to ignore the non-linear constraints and apply direct linear transformation (DLT) to estimate the absolute pose by solving a linear system [51]. Alternatively, methods that fully consider non-linear constraints have been discussed in [7], where the core idea is to construct a system of polynomial equations derived from the original PnP problem and solve it using algebraic techniques [52]. In addition, it should be mentioned that Lepetit et al. [53] proposed the EPnP algorithm, which directly estimates the pose from four virtual control points [1], offering a more efficient solution for the PnP problem.

More recently, learning-based approaches have also been explored for absolute pose estimation [29], [54], [55]. Among these, transformer-based architectures have shown remarkable capability in modeling geometric dependencies. The Visual Geometry Grounded Transformer (VGGT) [56] embeds explicit geometric constraints into transformer layers for end-toend pose learning and achieves SOTA accuracy on standard benchmarks via geometry-aware attention. In addition, RF-Mamba [57] designs a frequency-aware state space model for RF-based perception, whose efficient frequency-domain noise processing provides insights for visual pose estimation with noisy correspondences. However, these learning-based methods [29], [54]–[57] require large-scale training data along with accurate ground truth poses to achieve good performance [58]– [60], which limits their deployment in scenarios where data annotation is scarce or expensive.

## B. Outlier-Robust Absolute Pose Estimation

In practical applications, obtaining perfect 2D-3D correspondences is challenging due to many factors such as low image resolution and variations in point density. Consequently, many practical algorithms have been developed to suppress outlier correspondences and achieve robust pose estimation [11]. Among these, RANSAC-based methods and their variants are widely employed when dealing with outliercontaminated data [6]. However, these algorithms are inherently non-deterministic and cannot guarantee reliable results as the outlier rate increases [32]. To address this limitation, deterministic global optimization methods have been proposed [10]. Svarm et al. [ ¨ 23] introduced a deterministic approach for city-scale localization with a known vertical direction. Aiger et al. [39] proposed a voting-based method to find the optimal absolute pose under the same prior. More recently, Liu et al. [10] employed a branch-and-bound (BnB) framework to determine the optimal translation, leveraging its strength in solving NP-hard and non-convex problems [33]. As a matter of fact, even without 2D-3D correspondences, the BnB-based methods can still solve the absolute pose, although they will take a lot of time [8]. Despite their robustness, most deterministic methods are computationally intensive and too slow for real-time applications [10].

## C. Transformation Decoupling

In essence, the computational burden of robust pose estimation comes from the high dimensional space, which has to be searched globally, of the full rigid pose [33]. One recently popular way is decoupling the rotation and translation components and searching the separated spaces [26], [61]. The key idea of the decoupling strategy is to explore the geometric constraints of the specific problem, and to construct sub-problems, which can be solved more efficiently. The most straightforward geometric constraints are pairwise constraints. For example, the angle pairwise constraints are applied to solve 3D point cloud registration [62]. Similarly, the point-topoint pairwise constraints are also used to solve 3D point cloud registration [63]. Furthermore, the point-to-point pairwise constraints are used to solve the PnP problem [64]. Besides, more complicated geometric constraints are explored in 3D point cloud registration [26], [44], [61], which significantly inspire our work.

![](images/07d414c264f23dbe0d69b4aed61a88a03e74073b395a1128eb7fb19a52691779.jpg)  
Fig. 2: The PnP problem involves determining the absolute camera pose from a set of given 2D-3D point pairs.

## D. Known Gravity Direction

With the widespread use of IMU sensors, the gravity direction can now be readily determined in advance, enabling numerous pose estimation techniques that leverage this prior information. Kukelova et al. [65] proposed a closed-form solution for absolute camera pose estimation with a known gravity direction. Sweeney et al. [2] introduced another method that utilizes the gravity prior by projecting points in both the world and camera frames onto the gravity vector to form a constraint, ensuring that the distances between corresponding points remain consistent across both coordinate systems. Chandrasekhar [45] addressed the absolute pose estimation problem with an axis prior by enumerating the intersection of two conic curves. Several other methods also exploit the known vertical direction in other pose estimation [26]. For instance, the Perspective-n-Line (PnL) problem has been explored in [20], [66], while relative pose estimation techniques with a gravity prior are discussed in [16], [21]. The core idea behind these approaches is to reduce the problems dimensionality with the help of known gravity direction, thereby simplifying the computation.

## III. PROBLEM FORMULATION

The Perspective-n-Point (PnP) problem is a fundamental geometric problem in computer vision. Its goal is to estimate the position and orientation (i.e., pose) of the calibrated camera coordinate system relative to the world coordinate system, given a set of 2D image points (pixel points) and their corresponding 3D world points. In other words, the PnP problem aims to find the cameras external parameters (the rotation matrix R and translation vector t).

With calibrated camera intrinsic parameters, the feature points’ locations on the image can be normalized to the unitsphere surface $( \mathrm { i . e . , \mathbb { S } ^ { 2 } } )$ in the camera coordinate, where they are referred to as bearing vectors [8]. Specifically, for the i-th

![](images/110dab6a7de0f191c831ce09c94ee691066e0814526b401ae761bca200d5419e.jpg)  
Fig. 3: The overview of our proposed absolute pose estimation method. It consists of four stages: decoupling rotation with gravity prior, estimation of rotation with global voting, estimation of translation with RANSAC from outlier-filtered inputs, and pose refinement.

3D observation $\pmb { x } _ { i } \in \mathbb { R } ^ { 3 }$ and its corresponding bearing vector $\ b { y } _ { i } \in \mathbb { S } ^ { 2 }$ , if they are perfectly aligned (see Fig. 2),

$$
\lambda _ { i } y _ { i } = \mathbf { R } x _ { i } + t , \quad i = 1 \cdot \cdot \cdot n\tag{1}
$$

where $\lambda _ { i }$ is the i-th scale factor, and n denotes the number of point pairs. The variables $\mathbf { R } \in \mathbb { S O } ( 3 )$ and $\pmb { t } \in \mathbb { R } ^ { 3 }$ represent the rigid motion to be determined. The PnP problem focuses on estimating the absolute camera pose, specifically R and t, from a set of given 2D-3D point pairs.

## IV. METHOD

## A. Overview

As illustrated in Fig. 3, our proposed absolute pose estimation method comprises four stages: decoupling rotation using a gravity prior, estimating rotation through global voting, estimating translation with RANSAC from outlier-filtered inputs, and pose refinement. Specifically, by leveraging the known gravity direction, the original PnP problem is reduced to a 4-DoFs problem: 1 DoF for the rotation angle and 3 DoFs for translation. For rotation angle estimation, we propose a novel geometric relation that transforms the problem into solving a corresponding trigonometric function. Global voting is then employed to identify the optimal rotation angle θ. Once the optimal camera rotation R is determined, mismatched points are also significantly eliminated. The translation vector t in the 2D-3D correspondence problem is then efficiently computed using RANSAC. To further improve the accuracy of the rotation R and translation vector t, pose refinement is performed. Notably, a hidden variable resultant is introduced to reduce computational complexity and ensure certifiable results.

## B. Decoupling Rotation with Gravity Prior

Decoupling rotation using a gravity prior reduces the problem from 6-DoFs to 4-DoFs. By obtaining the gravity directions $\mathbf { \nabla } _ { \mathbf { \boldsymbol { { g } } } _ { c } }$ in the camera coordinate system and $\mathbf { \nabla } _ { \mathbf { \boldsymbol { \mathbf { \mathit { g } } } } \omega }$ in the world coordinate system, an additional constraint on the rotational motion can be established,

$$
\mathbf { \mathit { g } } _ { c } = \mathbf { \mathbf { R } } \mathbf { \mathit { g } } _ { w }\tag{2}
$$

![](images/a9eca48909ed5814ae17e8e8fcaf3018cc18b8519ab96fb362f0759e5d1e71be.jpg)  
Fig. 4: The rotation solution that is constrained by prior gravity, i.e., Eq. (3). Specifically, all the rotation solutions that meet the prior gravity constraint can be composed of two steps: (I) and (II). Please consult the main body of the text to understand the implications of all variables.

The solution for the desired rotation R can be calculated explicitly [34](see Fig. 4),

$$
\mathbf { R } = \mathbf { R } \left( \theta , \pmb { g } _ { c } \right) \mathbf { R } _ { 0 }\tag{3}
$$

where $\mathbf { R } ( \theta , g _ { c } )$ represents a rotation around the axis $\mathbf { \nabla } _ { \mathbf { \boldsymbol { { g } } } _ { c } }$ by an angle θ. This rotation can be computed using Rodrigues rotation formula [40], [51],

$$
\begin{array} { r l } & { \mathbf { R } \left( \theta , { { g } _ { c } } \right) = \exp \left( \theta \left[ { { g } _ { c } } \right] _ { \times } \right) } \\ & { \quad \quad = \mathbf { I } + \sin ( \theta ) [ \mathbf { { g } _ { c } } ] _ { \times } + \left( 1 - \cos ( \theta ) \right) [ \mathbf { { g } _ { c } } ] _ { \times } ^ { 2 } } \end{array}\tag{4}
$$

where $[ { \pmb g } _ { c } ] _ { \times }$ denotes the skew-symmetric matrix used for the cross product. Formally,

$$
[ { \pmb g } _ { c } ] _ { \times } = \left[ \begin{array} { c } { g _ { c } ( 1 ) } \\ { g _ { c } ( 2 ) } \\ { g _ { c } ( 3 ) } \end{array} \right] _ { \times } = \left[ \begin{array} { c c c } { 0 } & { - g _ { c } ( 3 ) } & { g _ { c } ( 2 ) } \\ { g _ { c } ( 3 ) } & { 0 } & { - g _ { c } ( 1 ) } \\ { - g _ { c } ( 2 ) } & { g _ { c } ( 1 ) } & { 0 } \end{array} \right]\tag{5}
$$

Similarly, $\mathbf { R } _ { 0 }$ is a rotation that can align $\mathbf { \nabla } _ { \mathbf { \boldsymbol { \mathbf { \mathit { g } } } } _ { w } }$ with $\mathbf { \eta } _ { \mathbf { { \sigma } } _ { \mathbf { { \sigma } } _ { \mathbf { { \sigma } } _ { \mathbf { { \sigma } } } } } } \mathbf { { \sigma } } _ { \mathbf { { \sigma } } _ { \mathbf { { \sigma } } _ { \mathbf { { \sigma } } _ { \mathbf { { \sigma } } } } } }$ . The easy way is via the minimal/maximal geodesic motion, and it can be computed as [10]:

$$
\begin{array} { r l } & { \mathbf { R } _ { 0 } = \exp \left( \omega \left[ \pmb { g } _ { m } \right] _ { \times } \right) } \\ & { \quad \quad = \mathbf { I } + \sin ( \omega ) [ \pmb { g } _ { m } ] _ { \times } + \left( 1 - \cos ( \omega ) \right) [ \pmb { g } _ { m } ] _ { \times } ^ { 2 } } \end{array}\tag{6}
$$

where $\omega = \operatorname { a r c c o s } ( g _ { c } ^ { T } g _ { w } )$ and $\begin{array} { r } { { \pmb g } _ { m } = \frac { { \pmb g } _ { w } \times { \pmb g } _ { c } } { | | { \pmb g } _ { w } \times { \pmb g } _ { c } | | } } \end{array}$ . By substituting the Eq. (3) back into the Eq. (1), the original problem can be reformulated as:

$$
\lambda _ { i } { \pmb y } _ { i } = { \bf R } \left( \theta , { \pmb g } _ { c } \right) { \bf R } _ { 0 } { \pmb x } _ { i } + { \pmb t } , \quad i = 1 \cdots n\tag{7}
$$

Using Eq. (6), $\mathbf { R } _ { 0 }$ can be computed efficiently, leaving only the rotation angle θ of $\mathbf { R } ( \theta , g _ { c } )$ to be determined. As a result, with the known gravity direction, the PnP problem is reduced to a 4-DoFs problem: 1 DoF for the rotation angle $\theta \in [ - \pi , \pi ]$ and 3 DoFs for the translation $\pm \in \mathbb { R } ^ { 3 }$

## C. The Estimation of Rotation Angle

By considering pairwise constraints (2D-3D pairs), we can derive a relational equation based on Eq. (1). Specifically, given two 2D-3D correspondences, we have

$$
\left\{ \begin{array} { c } { \lambda _ { i } y _ { i } = \mathbf { R } x _ { i } + t } \\ { \lambda _ { i + 1 } y _ { i + 1 } = \mathbf { R } x _ { i + 1 } + t } \end{array} \right.\tag{8}
$$

Eliminating λ<sub>i</sub> and $\lambda _ { i + 1 }$

$$
\left( { \pmb y } _ { i } \times { \pmb y } _ { i + 1 } \right) ^ { T } { \bf R } \left( { \pmb x } _ { i } - { \pmb x } _ { i + 1 } \right) = 0 \Leftrightarrow m _ { k } ^ { T } { \bf R } { \pmb n } _ { k } = 0\tag{9}
$$

where × denotes the cross product. The terms $\mathbf { \nabla } m _ { k }$ and $\mathbf { \mathit { n } } _ { k }$ correspond to $\left( { \pmb y } _ { i } \times { \pmb y } _ { i + 1 } \right)$ and $\left( \pmb { x } _ { i } - \pmb { x } _ { i + 1 } \right)$ , respectively. Notably, only the rotation R needs to be determined in Eq. (9). Furthermore, with known gravity directions in both the camera and world coordinate systems, only the rotation angle θ remains to be solved. This can be accomplished using Rodrigues’ rotation formula [40], [51]. According to Eq. (3) and Eq. (9),

$$
m _ { k } ^ { T } \mathbf { R } \left( \theta , { \pmb g } _ { c } \right) \mathbf { R } _ { 0 } { \pmb n } _ { k } = 0\tag{10}
$$

The equation can be reformulated as follows, based on Eq. (4):

$$
m _ { k } ^ { T } \left( { \bf I } + \sin ( \theta ) [ g _ { c } ] _ { \times } + \left( 1 - \cos ( \theta ) \right) [ g _ { c } ] _ { \times } ^ { 2 } \right) { \bf R } _ { 0 } n _ { k } = 0\tag{11}
$$

Then, it can be simplified to

$$
\left\{ \begin{array} { r } { A _ { k } \sin ( \theta ) + B _ { k } \cos ( \theta ) + C _ { k } = 0 } \\ { \sin ^ { 2 } \left( \theta \right) + \cos ^ { 2 } \left( \theta \right) = 1 } \end{array} \right.\tag{12}
$$

where $A _ { k } = { \pmb { m } } _ { k } ^ { T } [ { \pmb { g } } _ { c } ] _ { \times } { \bf R } _ { 0 } { \pmb { n } } _ { k } , B _ { k } = - { \pmb { m } } _ { k } ^ { T } [ { \pmb { g } } _ { c } ] _ { \times } ^ { 2 } { \bf R } _ { 0 } { \pmb { n } } _ { k }$ and $C _ { k } \ = \ m _ { k } ^ { T } { \bf R } _ { 0 } n _ { k } ( 1 + [ g _ { c } ] _ { \times } ^ { 2 } )$ . The rotation angle θ can be computed by solving the corresponding trigonometric function. Specifically, global voting is employed to determine the optimal rotation angle $\theta ^ { * }$ . For N correspondences, the calculation is performed $\overset { \circ } { \underline { { N } } } ( N - 1 )$ times. To identify the most likely 2 correct $\theta ^ { * }$ , the range $[ - \pi , \pi ]$ is divided into 360 intervals, and the occurrences of θ in each interval are counted. The optimal rotation angle $\theta ^ { * }$ can be selected within the bin with the largest votes. If $\mathbf { \nabla } m _ { k }$ and $\mathbfit { n } _ { k }$ cannot produce a θ in the bin with the largest votes, there should be at least an outlier for the 2D-3D correspondences. In contrast, if they produce a satisfactory θ, the two correspondences are likely to be inliers. However, due to noise and randomness, we cannot guarantee that all correspondences are destined to be correct. Therefore, we count the number of each 2D-3D correspondence that produces θ in the interval where $\theta ^ { * }$ is located. The more often they appear, the more likely they are inliers [67]. According to the frequency of occurrence, we can select the top-ranked the data as the inner 2D-3D correspondences according to the application scenarios [68], [69]. Notably, although this method cannot guarantee that all the remaining data are inliers, it can filter out a large part of outliers, which facilitates the next step of calculation.

Algorithm 1 RANSAC for Estimating Translation Vector t   
(Given R)   
Input: 2D bearing vectors $Y = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { i } \}$ , 3D world   
points $\pmb { X } = \{ \pmb { x } _ { 1 } , \pmb { x } _ { 2 } , \ldots , \pmb { x } _ { i } \}$ , calculated rotation matrix   
R, max iterations N, error threshold threshold.   
Output: Translation vector $\pmb { t } _ { \mathrm { b e s t } } .$   
1: Initialize $t _ { \mathrm { b e s t } }  \mathrm { N u l l } ,$ max inliers $ 0 ;$   
2: for each iteration $j = 0$ to $N - 1$ do   
3: Randomly select two 2D-3D point pairs $( { \pmb x } _ { i } , { \pmb y } _ { i } ) ;$   
4: Construct the overdetermined equation $\boldsymbol { A } \boldsymbol { x } = \boldsymbol { b }$ using   
Eq. (8);   
5: Compute translation vector t by solving an overdeter  
mined equation using least squares;   
6: Initialize inliers count $ 0 ;$   
7: for each 2D ${ \mathbf { } } _ { \mathbf { } ^ { 3 k } }$ and corresponding 3D $\scriptstyle { \mathbf { { \mathit { x } } } } _ { k }$ do   
8: Compute bearing vector: ${ \pmb y } _ { k } ^ { \mathrm { t r } }$ ansformed $ \mathbf { R } \cdot \pmb { x } _ { k } + \pmb { t } ;$   
9: Compute angle between ${ \pmb y } _ { k }$ and ${ \pmb y } _ { k } ^ { \mathrm { t r } }$ ansformed   
10: if angle < threshold then   
11: inliers count ← inliers count + 1;   
12: end if   
13: end for   
14: if inliers count > max inliers then   
15: max inliers ← inliers count;   
16: $t _ { \mathrm { b e s t } }  t .$   
17: end if   
18: end for   
19: return $\scriptstyle t _ { \mathrm { b e s t } }$

## D. Translation Estimation

After obtaining the optimal camera rotation R, solving the translation then becomes a linear model fitting problem [33]. Besides, many mismatches can be removed by rotational constraints. Therefore, solving the sequential translation becomes quite easy. In this paper, the RANSAC algorithm is applied to estimate translation vector t. The task is to determine t such that the transformed 3D points, after applying R, align closely with their corresponding 2D bearing vectors. The key steps of this calculation process are shown in Algorithm 1.

## E. Pose Refinement

To achieve more accurate estimates of the rotation R and the translation vector t, pose refinement is employed in this work. The inliers derived from the initial estimates of R and t are used to refine and re-compute these values. Specifically, the pose estimation problem is formulated using linear projective constraints as follows:

$$
\left[ \pmb { y } _ { i } \right] _ { \times } \left( \mathbf { R } \pmb { x } _ { i } + \pmb { t } \right) = 0 , \quad i = 1 \cdots n\tag{13}
$$

For a larger number of 2D-3D correspondences, the left and right sides of Eq. (13) may not be exactly equal. In such cases, the goal is to estimate the optimal R and t in the least squares sense. Therefore, the optimizable objective function can be defined to minimize the errors as follows:

$$
\operatorname* { m i n } _ { R , t } \sum _ { i } ( [ { \pmb y } _ { i } ] _ { \times } \left( { \bf R } { \pmb x } _ { i } + t \right) ) ^ { T } ( [ { \pmb y } _ { i } ] _ { \times } \left( { \bf R } { \pmb x } _ { i } + t \right) )\tag{14}
$$

Constrained least squares. Note that Rx, the rotation of a vector x, can be expressed as:

$$
\left[ \begin{array} { c c c c c c c c c } { x _ { 1 } } & { x _ { 2 } } & { x _ { 3 } } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { x _ { 1 } } & { x _ { 2 } } & { x _ { 3 } } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { x _ { 1 } } & { x _ { 2 } } & { x _ { 3 } } \end{array} \right] \mathrm { v e c } ( \mathbf { R } )\tag{15}
$$

where $\operatorname { v e c } ( \mathbf { R } )$ represents the row-flattened vector form of R. In this work, we align $\mathbf { \nabla } _ { \mathbf { \boldsymbol { { g } } } _ { c } }$ with the z-axis, the rotation matrix contains only three unique elements, allowing the expression to be simplified as follows:

$$
\mathbf { R } { \pmb x } = \left[ \begin{array} { c c c } { \cos ( \theta ) } & { - \sin ( \theta ) } & { 0 } \\ { \sin ( \theta ) } & { \cos ( \theta ) } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} \right] \left[ \begin{array} { c c c } { x _ { 1 } } \\ { x _ { 2 } } \\ { x _ { 3 } } \end{array} \right]\tag{16}
$$

where $x = \cos ( \theta )$ and $y = \sin ( \theta )$ . It can be transformed into the following form:

$$
\mathbf { R } { \pmb x } = { \pmb X } { \pmb r } = \left[ \begin{array} { c c c } { x _ { 1 } } & { - x _ { 2 } } & { 0 } \\ { x _ { 2 } } & { x _ { 1 } } & { 0 } \\ { 0 } & { 0 } & { x _ { 3 } } \end{array} \right] \left[ \begin{array} { l } { x } \\ { y } \\ { 1 } \end{array} \right]\tag{17}
$$

where X represents the matrix form of the vector x. Consequently, the optimizable objective function of Eq. (14) can be rewritten as follows:

$$
\operatorname* { m i n } _ { { \boldsymbol { r } } , t } \sum _ { i } ( X _ { i } { \boldsymbol { r } } + { \boldsymbol { t } } ) ^ { T } Q _ { i } ( X _ { i } { \boldsymbol { r } } + { \boldsymbol { t } } )\tag{18}
$$

where $Q _ { i }$ represents $\left[ { \pmb y } _ { i } \right] _ { \times } ^ { T } \left[ { \pmb y } _ { i } \right] _ { \times }$ . This sum of squared linear equations is non-negative and convex concerning the unconstrained t. By taking the derivative of the objective function for t and setting it to zero, the globally minimizing value of t can be computed given r:

$$
\begin{array} { l } { { \displaystyle t = - S r } } \\ { { \displaystyle \quad = - ( \sum _ { i } Q _ { i } ) ^ { - 1 } ( \sum _ { i } Q _ { i } X _ { i } ) r } } \end{array}\tag{19}
$$

By substituting the value of t back into the objective function (Eq. 18) and consolidating the terms, the simplified constrained quadratic minimization problem can be expressed as follows:

$$
\begin{array} { l } { \underset { \pmb { r } , \pmb { t } } { \operatorname* { m i n } } \sum _ { i } ( X _ { i } \pmb { r } - \pmb { S } \pmb { r } ) ^ { T } Q _ { i } ( X _ { i } \pmb { r } - \pmb { S } \pmb { r } ) } \\ { = \underset { \pmb { r } } { \operatorname* { m i n } } \pmb { r } ^ { T } \Omega \pmb { r } \quad \mathrm { s . t . } \quad x ^ { 2 } + y ^ { 2 } = 1 . } \end{array}\tag{20}
$$

where Ω is represented as follows:

$$
\pmb { \Omega } = \sum _ { i } ( \pmb { X } _ { i } - \pmb { S } ) ^ { T } \pmb { Q } _ { i } ( \pmb { X } _ { i } - \pmb { S } )\tag{21}
$$

Solution with hidden variable resultant. Clearly, Eq. (20) represents a typical equality-constrained optimization problem [70]. The corresponding Lagrangian formulation is:

$$
\operatorname* { m i n } _ { x , y } f ( x , y ) + \beta ( x ^ { 2 } + y ^ { 2 } - 1 )\tag{22}
$$

where $\beta$ is the Lagrangian multiplier, and $f ( x , y )$ represents the expanded polynomial form of $r ^ { T } \Omega r$ . By taking the partial derivatives with respect to x, y, and $\beta \colon$

$$
\frac { d f } { d x } + 2 \beta x = 0\tag{23}
$$

$$
{ \frac { d f } { d y } } + 2 \beta y = 0\tag{24}
$$

$$
x ^ { 2 } + y ^ { 2 } = 1\tag{25}
$$

The term $\beta$ can be eliminated by multiplying Eq. (23) by y and subtracting Eq. (24) multiplied by x:

$$
y { \frac { d f } { d x } } - x { \frac { d f } { d y } } = 0\tag{26}
$$

More specifically, Eq. (26) can be expressed in a conic form, referred to as the derivative conic:

$$
\begin{array} { r l } & { \left[ \boldsymbol { x } ^ { } \right] ^ { T } \left[ \begin{array} { c c c } { - 2 \Omega _ { 0 , 1 } } & { \Omega _ { 0 , 0 } - \Omega _ { 1 , 1 } } & { - \Omega _ { 1 , 2 } } \\ { \Omega _ { 0 , 0 } - \Omega _ { 1 , 1 } } & { 2 \Omega _ { 0 , 1 } } & { \Omega _ { 0 , 2 } } \\ { - \Omega _ { 1 , 2 } } & { \Omega _ { 0 , 2 } } & { 0 } \end{array} \right] \left[ \begin{array} { c } { x } \\ { y } \\ { 1 } \end{array} \right] } \\ & { = \boldsymbol { r } ^ { T } \boldsymbol { \Lambda } \boldsymbol { r } = 0 } \end{array}\tag{27}
$$

This conic generally represents a hyperbola. The Eq. (25) defines the equation of a circle, which can also be expressed in conic form. Specifically, solving Eq. (25) and Eq. (27) can be framed as finding the intersection of two conic curves, a problem comprehensively discussed by Prof. Jrgen Richter-Gebert in [71]. The core idea involves constructing a new degenerate conic, which can consist of two possibly coincident lines. The points of intersection between this degenerate conic and either of the original conics yield the desired points. In [45], the authors addressed this problem using this geometric approach. However, a drawback of this method is the need to identify all possible intersections, which increases computational complexity and susceptibility to calculation errors. In this work, we introduce the use of a hidden variable resultant [72] to address this problem. Specifically, from Eq. (25) and Eq. (27), we derive:

$$
\left\{ \begin{array} { r l l } { a x ^ { 2 } + b y ^ { 2 } + c x y + d x + e y } & { = 0 } \\ { x ^ { 2 } + y ^ { 2 } - 1 } & { = 0 } \\ { a x ^ { 3 } + b x y ^ { 2 } + c x ^ { 2 } y + d x ^ { 2 } + e x y } & { = 0 } \\ { x ^ { 3 } + x y ^ { 2 } - x } & { = 0 } \end{array} \right.\tag{28}
$$

where $a = \Omega _ { 0 , 1 } , b = - \Omega _ { 0 , 1 } , c = \Omega _ { 1 , 1 } - \Omega _ { 0 , 0 } , d = \Omega _ { 1 , 2 } ,$ $e = - \Omega _ { 0 , 2 }$ . The equations can then be rewritten as follows:

$$
{ \left[ \begin{array} { l l l l l } { 0 } & { a } & { c y + d } & { b y ^ { 2 } + e y } \\ { 0 } & { 1 } & { 0 } & { y ^ { 2 } - 1 } \\ { a } & { c y + d } & { b y ^ { 2 } + e y } & { 0 } \\ { 1 } & { 0 } & { y ^ { 2 } - 1 } & { 0 } \end{array} \right] } { \left[ \begin{array} { l } { x ^ { 3 } } \\ { x ^ { 2 } } \\ { x } \\ { 1 } \end{array} \right] } = { \left[ \begin{array} { l } { 0 } \\ { 0 } \\ { 0 } \\ { 0 } \end{array} \right] }\tag{29}
$$

To obtain the non-trivial solutions, the determinant of the coefficient matrix should be zero.

$$
\left| { \begin{array} { c c c c } { 0 } & { a } & { c y + d } & { b y ^ { 2 } + e y } \\ { 0 } & { 1 } & { 0 } & { y ^ { 2 } - 1 } \\ { a } & { c y + d } & { b y ^ { 2 } + e y } & { 0 } \\ { 1 } & { 0 } & { y ^ { 2 } - 1 } & { 0 } \end{array} } \right| = 0\tag{30}
$$

This leads to a univariate polynomial in terms of the hidden variable,

$$
m _ { 4 } y ^ { 4 } + m _ { 3 } y ^ { 3 } + m 2 y ^ { 2 } + m _ { 1 } y + m _ { 0 } = 0\tag{31}
$$

where $m _ { 4 } = ( a - b ) ^ { 2 } + c ^ { 2 } , m _ { 3 } = 2 c d - 2 e ( a - b )$ $m _ { 2 } =$ $- c ^ { 2 } + d ^ { 2 } + e ^ { 2 } - 2 a ( a - b ) , m _ { 1 } = 2 a e - 2 c d , m _ { 0 } = a ^ { 2 } - d ^ { 2 }$ By solving the univariate polynomial system, we can obtain y. Then, x can be solved as follows:

$$
x = \pm { \sqrt { 1 - y ^ { 2 } } }\tag{32}
$$

Finally, we select the $( x , y )$ that minimizes $\mathbf { r } ^ { T } \pmb { \updownarrow }$ Ωr as the result.

## V. EXPERIMENTAL SETTING

## A. Compared Methods

In our experiments, we compare our proposed pose estimation method against several existing approaches on both synthetic and real-world datasets. Specifically, the following algorithms are included in the comparison:

• BnB: The 4-DoF algorithm based on the Branch-and-Bound framework proposed in [10].

• Gao: The P3P algorithm proposed by Gao et al. [73], embedded within RANSAC.

• Ap3: The AP3P algorithm introduced by Ke et al. [74], incorporated into the RANSAC framework.

• Swee: The P2P algorithm proposed by Sweeney et al. [75], encapsulated within the RANSAC framework.

• Kuke: The P2P algorithm introduced by Kukelova et al. [65], integrated with RANSAC.

• Sqpnp: The SqPnP algorithm proposed by Terzakis et al. [76], embedded within RANSAC.

• Epnp: The EPnP algorithm proposed by Lepetit et al. [53], integrated with RANSAC.

• SupeRANSAC: The PnP algorithm proposed by Barath [77], which represents the SOTA in RANSAClike methods.

• AaPnP: The axis-aligned PnP algorithm proposed by Roch et al. [78], which is a method for camera pose estimation that exploits specific real-world constraints.

## B. Evaluation Metrics

To compare and evaluate the performance of each method, we employ two evaluation metrics: rotation error and translation error. The rotation error, measured by angle distance [79], is defined as follows:

$$
e _ { r o t } = \operatorname { a r c c o s } \left( \frac { T r ( \mathbf { R } _ { g t } ^ { T } \mathbf { R } _ { c a l } ) - 1 } { 2 } \right)\tag{33}
$$

where $T r ( \cdot )$ denotes the trace of a square matrix, and $\mathbf { R } _ { c a l }$ represents the rotation result obtained from the implemented algorithm. Furthermore, the translation error is defined as follows:

$$
e _ { t r a n s } = \| \pmb { t } _ { g t } - \pmb { t } _ { c a l } \|\tag{34}
$$

where $\mathbf { \delta } _ { t _ { c a l } }$ denotes the translation result obtained from the corresponding algorithms.

## C. Hardware Setting

To ensure a fair comparison, all experiments are conducted under the same hardware settings, and all methods are implemented in C++, and the experiments are performed on a personal laptop equipped with an R7-6800H CPU @ 3.20 GHz and 16 GB of RAM.

## VI. EXPERIMENTAL RESULTS

## A. Synthetic Data Experiments

Data generation. In our simulated experiments, we first randomly generate the pose of the truth of the ground, with the rotation matrix $\mathbf { R _ { g t } } \in \mathbb { S O } ( 3 )$ and the translation vector $t _ { g t } \in [ - 1 0 0 , 1 0 0 ] ^ { 3 }$ . To obtain 2D-3D correspondences, 3D points are randomly generated within the range $[ - 1 0 0 , 1 0 0 ] ^ { 3 }$ and their corresponding bearing vectors are computed by applying a rigid transformation and normalizing them to unit length. Outliers are introduced by randomly generating new 3D points to replace the original ones, with the outlier rate denoted as $\omega ~ = ~ N _ { o u t l i e r } / N$ . To simulate noise, we apply a standard normal distribution with a deviation parameter σ. Noise can be added in two ways: either by perturbing the 3D points directly or by adding noise to pixel coordinates, which requires setting the focal length and intrinsic matrix.

Controlled experiments. To evaluate the performance of different methods, we conduct controlled experiments on synthetic data, focusing on two key aspects: outlier rate and noise level. To clearly illustrate the results, we use the success rate as the evaluation metric. Specifically, Er $\mathrm { ~ R ~ } < 1 ^ { \circ }$ means a successful case for rotation calculation that $e _ { r o t }$ is less than 1 degree. Similarly, Er $_ - \mathrm { ~ T ~ } < 1 . 5$ represents a good case for translation calculation that $e _ { t r a n s }$ is less than 1.5.

First, we evaluate the robustness of each method against outliers. The input data are tested with different outlier rates, specifically $\omega = \{ 0 . 1 , 0 . 2 , \cdot \cdot \cdot , 0 . 8 \}$ . In these experiments, the number of 2D-3D correspondences is set to $N = 1 0 0 0 .$ the noise level is fixed at $\sigma \ = \ 2 . 0 .$ , and each scenario is repeated 100 times. The results are presented in Fig. 5(a) and Fig. 5(b). It can be observed that, under the same noise level, our proposed method exhibits significantly higher robustness in both rotation and translation estimation as the outlier rate increases compared to the other methods.

Second, we evaluate the robustness against noise. The input data is tested under various noise levels, i.e., $\sigma =$ $\{ 0 . 5 , 1 , \cdots , 4 \}$ . Specifically, we set the number of correspondences to $N \ = \ 1 0 0 0$ and fix the outlier rate at $\omega \ : = \ : 0 . 5$ Each experiment is repeated 100 times. The results are shown in Fig. 5(c) and Fig. 5(d). We observe that, compared to the outlier rate, the noise level has a more significant impact on the performance. However, within a reasonable noise range, our method maintains an almost 100% success rate in both rotation and translation estimations.

![](images/ede18b2ada796f0ac7daecb0007546bb01329a5148b8f6f2c037223c27f050b1.jpg)  
(a)

![](images/a5303892f81686e3ed01cbee3733effef94b8d346fd4f575a181acbfb004ed06.jpg)  
(b)

![](images/d88bd624d81ad452ae888cbfda9351bd4796f55143d388b82f7aa55028c8aa66.jpg)  
(c)

![](images/40aa2914f7e19e892cacf81359d90da4f4be0b7634fbdae78454881f4ecf124d.jpg)  
(d)  
Fig. 5: Controlled Experiments. (a) and (b) illustrate the success rates for input data under varying outlier rates. (c) and (d) depict the success rates for input data with different noise levels.

![](images/f1d585d99bc994d7dc47c4b5171bae3a166ed3d5f8369fb44a46b532558dc386.jpg)  
(a)

![](images/d05029bca85e02578c9743b3cfc5b18c488618ba68d7263ab92943d53f21775f.jpg)  
(b)  
Fig. 6: Accuracy level-success rate curves under heavily corrupted data conditions, showing the performance of each method in scenarios where both outlier rates and noise levels are high.

Finaly, to further demonstrate the robustness of the proposed method, we conduct experiments in extreme scenarios where the input data is highly corrupted with both a large outlier rate and a high noise level. Specifically, we set the number of correspondences to $N = 1 0 0 0$ , the outlier rate to $\omega = 0 . 5$ and the noise level to $\sigma = 1 0 . 0$ . Each experiment is repeated 100 times. As illustrated in Fig. 6, our proposed method demonstrates significantly greater robustness to both noise and outliers compared to other methods, maintaining a higher success rate even under severe conditions.

![](images/42c828d276b2e157f705d058abb831fd5b20589682dd2ff653b39cc3463ad27f.jpg)  
(a)

![](images/2fb01f43321fa482e3df61fdacdf03472bdf9e2d8e38c1b41a3240a1bcf70c2c.jpg)

Gravity direction bias. It is also important to consider the vertical gravity direction bias, as IMU sensors may occasionally return values with deviations. To simulate such scenarios, we introduce a vertical direction bias with the parameters $N = 1 0 0 0 , \sigma = 1$ , and $\omega = 0 . 5$ . For the gravity bias, we randomly rotate the gravity vector in arbitrary directions, using varying rotation angles $\theta _ { g } = \{ 0 . 5 ^ { \circ } , 1 ^ { \circ } , \cdot \cdot \cdot , 5 ^ { \circ } \}$ . Each trial is repeated 100 times, and the resulting error distributions are presented in Fig. 7. The results indicate that our proposed method can still achieve acceptable absolute camera pose solutions, even with biased vertical directions. However, as the bias angle increases, both translation and rotation estimation errors grow accordingly, with a higher likelihood of large errors. This behavior is expected, as greater bias naturally leads to more significant deviations in pose estimation.

(b)  
Fig. 7: The error distribution with various gravity bias. (a) Rotation error (b) translation error.  
![](images/2b844fa75e805d36e4b7c72d1be4bf4f5e520a69da63d29a0ae918fd4bdff411.jpg)

![](images/d485d527793b323eacd752757f10d6b6ef96e639096a851522a89820cfdfdab0.jpg)  
Fig. 8: Performance comparison of the proposed method with or without pose refinement.

Effect of pose refinement. To further validate the effectiveness of the proposed pose refinement module, we conduct ablation experiments comparing the method with and without pose refinement. The results, summarized in Fig. 8, demonstrate that incorporating the pose refinement module leads to improved performance.

![](images/0eff2b7806693b02d09b49210fd2e7ccc774dd067116fa884549ac86d20f300c.jpg)  
(a)

![](images/d483186415a24a5671df3d1f86e47be1a9d2c70e4444d217570d89cb94420361.jpg)  
(b)  
Fig. 9: Average runtime for different numbers of input pairs. (a) for $N \in [ 1 0 0 , 1 5 0 0 ]$ , (b) for N ∈ [3000, 10000].

TABLE I: Average runtime and number of input pairs for the three components.
<table><tr><td>Part</td><td></td><td></td><td>| R estimation | t estimation | pose refinement</td></tr><tr><td>Time(sec)|</td><td>0.70</td><td>0.21</td><td>0.004</td></tr><tr><td>N</td><td>1000</td><td>100</td><td>91</td></tr></table>

TABLE II: Average runtime of our method under different parallelization settings.
<table><tr><td>Threads</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td>Time(sec)</td><td>8.05</td><td>4.09</td><td>2.15</td><td>1.24</td><td>0.91</td></tr></table>

Time complexity analysis. In this part, we analyze the time complexity of our proposed method. As previously discussed, the runtime primarily consists of three components: R estimation, t estimation, and pose refinement. In Tab. I, we present the average runtime for each component, along with the corresponding number of processed pairs, based on 100 experiments with the settings $N ~ = ~ 1 0 0 0 , ~ \sigma ~ = ~ 1$ , and $\omega = 0 . 5$ . We also compare the runtime across different input sizes, specifically $N _ { 1 } ~ = ~ \{ 1 0 0 , 3 0 0 , \cdot \cdot \cdot , 1 5 0 0 \}$ and $N _ { 2 } \ =$ $\{ 3 0 0 0 , 4 0 0 0 , \cdots , 1 0 0 0 0 \}$ . The average runtime results, computed over 100 trials, are illustrated in Fig. 9(a) and Fig. 9(b). The results indicate that R estimation is the most timeconsuming step due to its need to process the largest number of pairs, N. Specifically, it requires $N ( N { - } 1 ) / 2$ computations for each pair combination, resulting in a quadratic growth trend in runtime. Both curves in the figures clearly exhibit this trend, further confirming that R estimation dominates the overall time complexity. Notably, unlike noise-sensitive algorithms such as BnB, the runtime of our method remains unaffected by increases in noise level (σ) and outlier rate (ω).

Furthermore, our method processes only two pairs at a time during R estimation, meaning that additional pairs can be handled concurrently. This characteristic makes our approach well-suited for parallel acceleration techniques, such as multithreading or GPU-based computation. In Tab. II, we compare the average runtime for 100 trials using varying numbers of threads, with $N = 1 0 0 0 , \sigma = 1$ , and $\omega = 0 . 5$ . The results show a significant reduction in runtime as the number of threads increases. Therefore, our method can be easily accelerated through parallel computing, making it feasible for real-time applications to some extent.

![](images/72b792ade5677d938f0a7f085bf5d0e69463aa09a1255fc6ce46a85237bf26f7.jpg)  
Fig. 10: SIFT matching example from the freiburg1 desk scene in the TUM RGB-D dataset.

The above synthetic experimental results further validate the complementary advantages of our two-stage framework: the coarse estimation stage ensures stable and robust pose recovery under extreme outlier and noise conditions, while the refinement stage delivers consistent accuracy improvement on the basis of the coarse initial pose. The performance degradation of all compared baselines under high outlier rates also confirms the superiority of our proposed method in robust estimation for the gravity-constrained 1DoF rotation space.

## B. Real-World Data Experiments

To evaluate the feasibility of our proposed method in real-world applications, we conduct experiments using realworld data. We utilize typical images and their corresponding 3D data (depth maps or 3D point sets) from three publicly available datasets: the TUM RGB-D dataset [80], the ETH3D dataset [81], and the RobotCar dataset [82], all of which provide ground-truth trajectories.

In our experiments, we select two images at a time to create 2D-3D pairs. For the first image, we obtain the bearing vectors y by using its 2D information, including pixel coordinates and the intrinsic matrix. For the second image, we acquire the corresponding 3D points x by using depth maps or Lidar point clouds. The 2D-3D pairs are generated by matching image pairs through feature descriptors, such as ORB [83] or SIFT [84]. Since feature matching is not perfectly accurate, noise and outliers are inevitably introduced during the matching process. We then estimate the absolute camera pose for the first image and calculate the error relative to the ground-truth pose to assess the performance of our method. To further highlight the robustness of our proposed method, we select non-adjacent frames for evaluation, as such pairs tend to produce more outliers during 2D-2D feature matching. Additionally, we intentionally avoid fine-tuning the featurematching parameters to achieve ideal results. Examples from the three datasets are shown in Fig. 10, Fig. 11, and Fig. 12.

When retrieving the 3D points corresponding to the matched pixels from the first image in the second image, we expand the search criteria to include pixels within a 10- to 20-pixel radius of the exact match to account for potential inaccuracies. For each dataset, we select approximately 100 image pairs of varying types and present the accuracy results in Fig. 13, Fig. 14, and Fig. 15. From the figures, we observe that our proposed method is effective in estimating the absolute camera pose when the vertical gravity direction is known. Additionally, it demonstrates greater robustness compared to many existing methods, particularly in scenarios with significant outlier contamination. However, a notable drawback of our approach is its longer runtime compared to RANSACtype methods, especially when handling large input sizes (N). Fortunately, the method can be easily accelerated through parallel computing techniques, such as GPU processing or multithreading, making it theoretically viable for real-time applications. We also find that the compared method AaPnP [78] performs significantly worse on the real-world datasets. This is mainly due to the fact that the real-world data contains mainly 6-DoF transformations, while the AaPnP [78] is only focusing on 3-DoF transformations. Furthermore, we compare the time consumption of eight different methods in Tab. III. Due to the variability in the number of pairs (N) generated from different image pairs, we report the average runtime by taking the middle 10% of all outcomes for each dataset to provide a more representative comparison.

![](images/c3de51a6f3285128d3048e3d3e7e861ea2f5267e2eedf27e782a80a666c1d5bb.jpg)  
Fig. 11: SIFT matching example from the exhibition scene in the ETH3D dataset. (For better visualization, the shown matches are downsampled).

![](images/ec81951b39b8ca8e86b8b8da4de61d74f7cb5f8fa6e630e55fee39595a1e05ac.jpg)  
Fig. 12: SIFT matching example from the overcast scene in the RobotCar dataset. (For better visualization, the shown matches are downsampled).

![](images/0e77adedfc62dac1bdf14f298c90cedc7d4ac10d843ff221e2c6a0aa49624b01.jpg)  
(a)

![](images/860dc15426c9579b1d0cbdb6c608a7c378577bddf7e63860dadb5f9cb4ddc9f7.jpg)  
(b)

Fig. 13: Experimental results on selected images from the TUM RGB-D dataset (higher is better).  
![](images/6b0c09044fd8e6222408a022309abd41257559f8b658813f77c4e99400bf8315.jpg)  
(a)

![](images/6cebbce79e530c439d7b780ed0ef36ad4ad43ca37c38eb2e8ae3380bbbf0cdad.jpg)  
(b)  
Fig. 14: Experimental results on selected images from the ETH3D dataset (higher is better).

![](images/8498e66743710b82ccff5e5e2493c83a44ea611a33c67baf0a1dcfb813bcbd20.jpg)  
(a)

![](images/f6b9f8e21d822b7d23836a15a04e8dda04ad88f73b6e8bfb09ccedc6845ac649.jpg)  
(b)

Fig. 15: Experimental results on selected images from the RobotCar dataset (higher is better).  
![](images/5cee90a3f391c4862624fa405b2d8134ada86277e890d6be8ff87462fa9531f6.jpg)  
Fig. 16: Error visualization of our method on a real-world trajectory from KITTI 01. The estimated trajectory (solid line) is compared against the reference trajectory (dashed line), with colors indicating the magnitude of relative pose error (RPE). The predominantly blue coloring demonstrates our method’s ability to maintain low errors across most of the trajectory, with only occasional higher errors (green/red) in limited segments.

TABLE III: Runtime comparison of different methods on the real-world datasets.
<table><tr><td>Time(sec) Method</td><td rowspan="2">Swee</td><td rowspan="2">Kuke</td><td rowspan="2">Epnp</td><td rowspan="2">Gao</td><td rowspan="2">Sqpnp</td><td rowspan="2">Ap3</td><td rowspan="2">BnB</td><td rowspan="2">SupeRANSAC</td><td rowspan="2">AaPnP</td><td rowspan="2">Our</td></tr><tr><td>Dataset</td></tr><tr><td>TUM RGBD</td><td>0.53</td><td>0.54</td><td>0.03</td><td>0.005</td><td>0.03</td><td>0.01</td><td>10</td><td>0.06</td><td>0.08</td><td>0.60</td></tr><tr><td>ETH3D</td><td>0.81</td><td>0.83</td><td>0.08</td><td>0.04</td><td>0.08</td><td>0.05</td><td>30</td><td>0.07</td><td>0.09</td><td>1.47</td></tr><tr><td>RobotCar</td><td>0.66</td><td>0.67</td><td>0.06</td><td>0.02</td><td>0.06</td><td>0.04</td><td>20</td><td>0.05</td><td>0.08</td><td>0.87</td></tr></table>

TABLE IV: Comparison between our method and the original ORB SLAM2 [85] system. The evaluation metrics include Root Mean Square Error (RMSE), Sum of Squared Errors (SSE), minimum error (min), maximum error (max), and standard deviation (std). The metrics are computed based on the absolute pose error (APE) of every timestamp.
<table><tr><td>Method</td><td>RMSE</td><td>SSE</td><td>min</td><td>max</td><td>std</td></tr><tr><td>ORB_SLAM2</td><td>16.40</td><td>361053.9</td><td>1.22</td><td>32.15</td><td>6.5687</td></tr><tr><td>ORB_SLAM2 + Ours</td><td>9.36</td><td>115607.3</td><td>1.63</td><td>14.93</td><td>2.8524</td></tr></table>

TABLE V: Comparison between our method and the original ORB SLAM2 [85] system. The evaluation metrics include Root Mean Square Error (RMSE), Sum of Squared Errors (SSE), minimum error (min), maximum error (max), and standard deviation (std). The metrics are computed based on the relative pose error (RPE) of every timestamp.
<table><tr><td>Method</td><td>RMSE</td><td>SSE</td><td>min</td><td>max</td><td>std</td></tr><tr><td>ORB_SLAM2</td><td>1.029</td><td>1121.2</td><td>0.0101</td><td>31.36</td><td>0.9910</td></tr><tr><td>ORB  $\_ { \mathrm { L A M 2 } } + \mathrm { O u r s }$ </td><td>0.539</td><td>311.7</td><td>0.0058</td><td>16.70</td><td>0.5157</td></tr></table>

The real-world experimental results are consistent with the synthetic data, demonstrating that our method maintains strong robustness in complex real scenarios. Compared with the compared baselines, our method’s performance advantage comes from the joint effect of the global voting module’s outlier filtering capability and the refinement module’s accuracy optimization, which makes it more suitable for practical applications.

## C. Validation of ORB-SLAM2 integrated with our method on the KITTI dataset.

To validate our method, we conducted comprehensive experiments by integrating our absolute pose estimation approach into ORB-SLAM2 [85]. Specifically, the original PnP solver of ORB-SLAM2 is replaced by our proposed solution. The experiments were conducted on the KITTI [14] dataset. We assessed the system’s performance using Absolute Pose Error (APE) and Relative Pose Error (RPE) as the primary evaluation metric. APE and RPE were selected because they effectively measure the system’s ability to maintain consistent pose estimation between consecutive frames while respecting the gravity constrainta critical requirement for real-world applications where local accuracy and stability are essential. The experimental results confirm the effectiveness of our method.

The visualization in Fig. 16 shows that the estimated trajectory (solid blue line) closely aligns with the reference trajectory (dashed line) across a complex path. The color-coded error map indicates that our method maintains consistently low errors (predominantly blue regions) throughout most of the trajectory, with only minor deviations (green and red regions) observed in challenging segments. Quantitative comparisons in Tab. IV and Tab. V further validate these findings. For APE, our method achieves around 50% less RMSE (9.36 vs. 16.40), SSE (115607.3 vs. 361053.9), max error (14.93 vs. 32.15), and std (2.8524 vs. 6.5687). For RPE, our method achieves an RMSE of 0.159, significantly lower than the original ORB-SLAM2’s 0.176. Additionally, our approach yields a lower SSE (57.63 vs. 70.74), reduced maximum error (3.64 vs. 4.58), and a smaller standard deviation (0.1344 vs. 0.1550). The results from APE and RPE demonstrate the improved stability and reliability. These results indicate that our method not only enhances accuracy but also increases robustness in pose estimation.

More graphically, as illustrated in Fig. 17, our experimental results further demonstrate the effectiveness of our method through multiple relocalization test cases on the KITTI [14] dataset. Each test case presents paired visualizations showing system performance before and after relocalization events. The visualizations depict the estimated trajectories (shown in blue) alongside the constructed point cloud maps (in black), with current key points highlighted in red within the map structure. These results highlight the ability of our method to correct significant drift and improve trajectory alignment during relocalization. In the pre-relocalization images, trajectory misalignments are consistently observed, where the estimated paths exhibit noticeable drift relative to the mapped environment. Such misalignments are common challenges in visual SLAM systems, particularly after prolonged operation or during challenging sequences. The post-relocalization images, however, clearly demonstrate our system’s robust recovery capabilities across various scenarios. Following each relocalization event, our gravity-prior-driven approach successfully corrects trajectory alignments, effectively eliminating the previously accumulated drift. The corrected trajectories exhibit improved consistency with the surrounding environmental structure, validating our method’s ability to ensure long-term reliability through effective relocalization.

The system-level experimental results fully validate the practical value of our proposed method. By replacing the original PnP solver of ORB-SLAM2 with our full two-stage framework, we achieve significant improvements in both trajectory accuracy and relocalization performance. This further confirms that our method’s robust coarse estimation and high-precision refinement modules can work synergistically to address the practical challenges of drift accumulation and relocalization failure in real SLAM systems.

## VII. DISCUSSION AND ANALYSIS

In this section, we systematically analyze the independent contribution of each core module and the synergistic effect of our two-stage cascaded framework, based entirely on the experimental results presented in previous sections. Our method is built on two tightly coupled stages with clear, complementary design goals: a coarse estimation stage for robust outlier filtering and reliable initial pose recovery, and a fine refinement stage for high-precision pose optimization. The contribution of each component is validated as follows.

![](images/91ae615c9b7589bdb6b27b181ae38d3393a69972bcd15cdc7ac2c3664a92decd.jpg)  
(a)

![](images/f4a3fa072ca8f34873384565f18014550b0c8a3c55f5099b6075348908258d79.jpg)  
(b)

![](images/30f65ffbeaba2dda66ad8b0424c0caacb186ef30879cec1cee469374e149480d.jpg)  
(c)

![](images/9a71a4f74e5603e4ae4e4c57e5e177204b5cd18baee38737ddc610f7fed6416b.jpg)  
(d)  
Fig. 17: Visualization of our system’s relocalization performance on Sequence 00 of the KITTI [14] dataset. (a) and (c) show the estimated trajectory (blue) before relocalization, where misalignment with the point cloud map (black) is evident due to accumulated drift. (b) and (d) illustrate the trajectory after relocalization, demonstrating that our gravity-prior-driven approach successfully corrects the alignment during the relocalization event.

## A. Contribution of the Coarse Estimation Stage

The core of our coarse estimation stage is the proposed rotation-first decoupling strategy and 1D global voting module, which is the primary source of our methods outlier robustness and computational efficiency. We conduct head-to-head comparisons with the SOTA gravity-aware 4DoF baseline [10], which shares the identical problem setting and gravity prior with our work and differs only in its core robust estimation module and decoupling strategy. The consistent performance superiority of our method directly verifies the fundamental improvements brought by our proposed module, which overcomes the limitations of existing frameworks in both robustness and computational efficiency. We also compare our method with mainstream de facto standard robust PnP baselines, and the results further confirm that our voting scheme, which fully leverages the low-dimensional nature of the gravity-constrained rotation space, delivers stronger robustness than traditional random sampling-based robust estimation frameworks.

## B. Contribution of the Fine Refinement Stage

The core of our fine refinement stage is the proposed nonminimal refinement algorithm tailored for the 4-DoFs gravityconstrained space, which is the primary source of the final pose accuracy improvement. We have completed dedicated module-level ablation experiments for this component, with a controlled variable setting that isolates the impact of the refinement module while keeping all other experimental conditions consistent. The ablation results clearly demonstrate that the proposed refinement module delivers a significant and consistent improvement in both rotation and translation accuracy, fully validating its independent contribution to the final performance, and filling the research gap of dedicated high-precision optimization for gravity-aware 4-DoFs pose estimation.

## C. Synergistic Effect of the Two-Stage Framework

Our two-stage framework is designed with tight coupling and mutual reinforcement between the two stages. The coarse estimation stage filters the vast majority of outliers and provides a reliable initial pose, which ensures the stability and convergence of the subsequent refinement module. In turn, the fine refinement stage fully unlocks the accuracy potential of the high-quality inlier correspondences and initial pose provided by the coarse stage. The system-level integration experiments in the SLAM framework further validate this synergistic effect. The full two-stage pipeline delivers significant improvements in trajectory accuracy and relocalization performance, which cannot be achieved by either stage alone.

In summary, the existing complete set of experiments in our manuscript has rigorously validated the independent contribution of each core module, the superiority of our core design over strong baselines, and the rationality of our holistic cascaded framework, fully establishing the contribution closure of our work.

## VIII. CONCLUSION

In this paper, we introduce a novel, efficient, and robust algorithm for absolute pose estimation. Our approach is based on a new transformation decoupling strategy that leverages geometric relations derived from the known gravity prior. By applying this strategy, the original 6-DoFs absolute pose estimation problem is reduced to a 4-DoFs problem: 1-DoF for the rotation angle and 3-DoFs for translation, significantly enhancing computational efficiency. To solve these subproblems, we employ a one-dimensional global voting algorithm for rotation estimation and use RANSAC to determine the translation. Additionally, we propose a novel pose refinement algorithm to further improve the accuracy of both rotation and translation estimates. Extensive experiments on synthetic data and real-world datasets (TUM RGB-D [80], ETH3D [81], and RobotCar [82]) demonstrate that our method outperforms existing SOTA approaches. To further validate the effectiveness of our method, we integrated it into ORB-SLAM2 [85] and conducted comprehensive experiments on the KITTI [14] dataset. The results demonstrate that our method effectively corrects substantial drift and significantly improves trajectory alignment during relocalization.

The module contribution analysis further confirms that the performance superiority of our method comes from the tight coupling of our three core components: the rotation-first decoupling strategy unlocks the low-dimensional advantage of the gravity-constrained space, the 1D global voting module ensures robust outlier filtering and initial pose recovery, and the dedicated refinement module delivers high-precision pose optimization. Our holistic framework addresses the key limitations of existing gravity-aware 4-DoFs pose solvers, and provides an efficient, robust, and high-precision solution for practical robotic and visual SLAM applications.

## ACKNOWLEDGMENTS

This work is supported in part by Macao Science and Technology Development Fund (Grant No. 0022/2025/ITP1); and in part by the MANNHEIM-CeCaS (Central Car Server-Supercomputing for Automotive, No. 16ME0820).

## REFERENCES

[1] E. Marchand, H. Uchiyama, and F. Spindler, “Pose estimation for augmented reality: a hands-on survey,” IEEE Transactions on Visualization and Computer Graphics, vol. 22, no. 12, pp. 2633–2651, 2015. 1, 3

[2] C. Sweeney, J. Flynn, B. Nuernberger, M. Turk, and T. Hollerer,¨ “Efficient computation of absolute pose for gravity-aware augmented reality,” in 2015 IEEE International Symposium on Mixed and Augmented Reality. IEEE, 2015, pp. 19–24. 1, 3

[3] J. Rehder, K. Gupta, S. Nuske, and S. Singh, “Global pose estimation with limited gps and long range visual odometry,” in ICRA, 2012. 1

[4] H. Li, W. Chen, J. Zhao, J.-C. Bazin, L. Luo, Z. Liu, and Y.-H. Liu, “Robust and efficient estimation of absolute camera pose for monocular visual odometry,” in ICRA, 2020. 1

[5] S. Lynen, B. Zeisl, D. Aiger, M. Bosse, J. Hesch, M. Pollefeys, R. Siegwart, and T. Sattler, “Large-scale, real-time visual–inertial localization revisited,” The International Journal ofRobotics Research, vol. 39, no. 9, pp. 1061–1084, 2020. 1

[6] Y. Jiao, Y. Wang, X. Ding, B. Fu, S. Huang, and R. Xiong, “2-entity random sample consensus for robust visual localization: Framework, methods, and verifications,” IEEE Transactions on Industrial Electron ics, vol. 68, no. 5, pp. 4519–4528, 2020. 1, 3

[7] L. Kneip, H. Li, and Y. Seo, “Upnp: An optimal o (n) solution to the absolute pose problem with universal applicability,” in ECCV, 2014. 1, 2

[8] D. Campbell, L. Petersson, L. Kneip, and H. Li, “Globally-optimal inlier set maximisation for camera pose and correspondence estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 2, pp. 328–342, 2020. 1, 3

[9] M. Garcia-Salguero, E. Dima, A. Mateus, and J. Gonzalez-Jimenez, “Fast certifiable algorithm for the absolute pose estimation of a camera,” SIAM Journal on Imaging Sciences, vol. 17, no. 3, pp. 1415–1432, 2024. 1

[10] Y. Liu, G. Chen, and A. Knoll, “Absolute pose estimation with a known direction by motion decoupling,” IEEE Transactions on Circuits and Systems for Video Technology, 2023. 1, 2, 3, 4, 7, 12

[11] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981. 1, 2, 3

[12] P. Antonante, V. Tzoumas, H. Yang, and L. Carlone, “Outlier-robust estimation: Hardness, minimally tuned algorithms, and applications,” IEEE Transactions on Robotics, vol. 38, no. 1, pp. 281–301, 2022. 1

[13] Y. Liu, Y. Wang, M. Wang, G. Chen, A. Knoll, and Z. Song, “Globally optimal linear model fitting with unit-norm constraint,” International Journal of Computer Vision, vol. 130, no. 4, pp. 933–946, 2022. 1

[14] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” The International Journal of Robotics Research, vol. 32, no. 11, pp. 1231–1237, 2013. 1, 11, 12, 13

[15] X. Shao, L. Zhang, T. Zhang, Y. Shen, and Y. Zhou, “Mofis slam: A multi-object semantic slam system with front-view, inertial, and surround-view sensors for indoor parking,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 7, pp. 4788–4803, 2021. 1

[16] Y. Ding, D. Barath, J. Yang, H. Kong, and Z. Kukelova, “Globally optimal relative pose estimation with gravity prior,” in CVPR, 2021. 1, 3

[17] Y. Salaun, R. Marlet, and P. Monasse, “Robust and accurate line-and/or¨ point-based pose estimation without manhattan assumptions,” in ECCV, 2016. 1

[18] A. Almansa, A. Desolneux, and S. Vamech, “Vanishing point detection without any a priori information,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 25, no. 4, pp. 502–507, 2003. 1

[19] B. Micusik and H. Wildenauer, “Minimal solution for uncalibrated absolute pose problem with a known vanishing point,” in 3DV, 2013. 1

[20] L. Lecrosnier, R. Boutteau, P. Vasseur, X. Savatier, and F. Fraundorfer, “Camera pose estimation based on pnl with a known vertical direction,” IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3852–3859, 2019. 1, 3

[21] Y. Ding, J. Yang, J. Ponce, and H. Kong, “Homography-based minimalcase relative pose estimation with known gravity direction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 1, pp. 196–210, 2020. 1, 3

[22] Y. Liu, G. Chen, R. Gu, and A. Knoll, “Globally optimal consensus maximization for relative pose estimation with known gravity direction,” IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 5905–5912, 2021. 1

[23] L. Svarm, O. Enqvist, F. Kahl, and M. Oskarsson, “City-scale localiza-¨ tion for cameras with known vertical direction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 7, pp. 1455– 1461, 2016. 1, 2, 3

[24] M. V. Ornhag, P. Persson, M. Wadenb<sup>¨</sup> ack, K.¨ Astr<sup>˚</sup> om, and A. Heyden,¨ “Trust your imu: Consequences of ignoring the imu drift,” in CVPR, 2022. 1

[25] Y. Ding, D. Barath, and Z. Kukelova, “Minimal solutions for panoramic stitching given gravity prior,” in ICCV, 2021. 1

[26] X. Li, Z. Ma, Y. Liu, W. Zimmer, H. Cao, F. Zhang, and A. Knoll, “Transformation decoupling strategy based on screw theory for deterministic point cloud registration with gravity prior,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 1, 2, 3

[27] V. Lepetit, F. Moreno-Noguer, and P. Fua, “Ep n p: An accurate o (n) solution to the p n p problem,” International Journal ofComputer Vision, vol. 81, pp. 155–166, 2009. 2

[28] X.-S. Gao, X.-R. Hou, J. Tang, and H.-F. Cheng, “Complete solution classification for the perspective-three-point problem,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 25, no. 8, pp. 930–943, 2003. 2

[29] D. Campbell, L. Liu, and S. Gould, “Solving the blind perspective-npoint problem end-to-end with robust differentiable geometric optimization,” in ECCV, 2020. 2, 3

[30] S. Hadfield, K. Lebeda, and R. Bowden, “Hard-pnp: Pnp optimization using a hybrid approximate representation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 3, pp. 768–774, 2018. 2

[31] H. Zhou, T. Zhang, and J. Jagadeesan, “Re-weighting and 1-point ransacbased p n n p solution to handle outliers,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 12, pp. 3022–3033, 2018. 2

[32] H. Le, T.-J. Chin, A. Eriksson, T.-T. Do, and D. Suter, “Deterministic approximate methods for maximum consensus robust fitting,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 3, pp. 842–857, 2019. 2, 3

[33] T.-J. Chin and D. Suter, The maximum consensus problem: recent algorithmic advances. Morgan & Claypool Publishers, 2017. 2, 3, 5

[34] A. P. Bustos and T.-J. Chin, “Guaranteed outlier removal for point cloud registration with correspondences,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 12, pp. 2868–2882, 2017. 2, 4

[35] H. Cao, G. Chen, J. Xia, G. Zhuang, and A. Knoll, “Fusion-based feature attention gate component for vehicle detection based on event camera,” IEEE Sensors Journal, vol. 21, no. 21, pp. 24 540–24 548, 2021. 2

[36] Y. Cheng, A. Knoll, and H. Cao, “Urnet: uncertainty-aware refinement network for event-based stereo depth estimation,” Visual Intelligence, vol. 3, no. 1, p. 18, 2025. 2

[37] H. Cao, G. Chen, H. Zhao, D. Jiang, X. Zhang, Q. Tian, and A. Knoll, “Sdpt: Semantic-aware dimension-pooling transformer for image segmentation,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 11, pp. 15 934–15 946, 2024. 2

[38] P. Antonante, V. Tzoumas, H. Yang, and L. Carlone, “Outlier-robust estimation: Hardness, minimally tuned algorithms, and applications,” IEEE Transactions on Robotics, vol. 38, no. 1, pp. 281–301, 2021. 2

[39] D. Aiger, H. Kaplan, E. Kokiopoulou, M. Sharir, and B. Zeisl, “General techniques for approximate incidences and their application to the camera posing problem,” arXiv preprint, 2019. 2, 3

[40] R. I. Hartley and F. Kahl, “Global optimization through rotation space search,” International Journal of Computer Vision, vol. 82, no. 1, pp. 64–79, 2009. 2, 4, 5

[41] D. Campbell and L. Petersson, “Gogma: Globally-optimal gaussian mixture alignment,” in CVPR, 2016. 2

[42] Y. Jiao, Y. Wang, X. Ding, M. Wang, and R. Xiong, “Deterministic optimality for robust vehicle localization using visual measurements,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 6, pp. 5397–5410, 2021. 2

[43] X. Li, Y. Liu, Y. Xia, V. Lakshminarasimhan, H. Cao, F. Zhang, U. Stilla, and A. Knoll, “Fast and deterministic (3+ 1) dof point set registration with gravity prior,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 199, pp. 118–132, 2023. 2

[44] X. Li, H. Cao, Y. Liu, X. Liu, F. Zhang, and A. Knoll, “Efficient and deterministic search strategy based on residual projections for point cloud registration with correspondences,” IEEE Transactions on Intelligent Vehicles, 2024. 2, 3

[45] A. Chandrasekhar, “Posegravity: Pose estimation from points and lines with axis prior,” arXiv preprint, 2024. 2, 3, 6

[46] G. Marullo, L. Tanzi, P. Piazzolla, and E. Vezzetti, “6d object position estimation from 2d images: a literature review,” Multim. Tools Appl., vol. 82, no. 16, pp. 24 605–24 643, 2023. 2

[47] J. A. Grunert, “Das pothenotische problem in erweiterter gestalt nebst bber seine anwendungen in der geodasie,” Grunerts Archiv fur Mathematik und Physik, pp. 238–248, 1841. 2

[48] Y. Ding, J. Yang, V. Larsson, C. Olsson, and K. Astr <sup>˚</sup> om, “Revisiting the¨ P3P problem,” in CVPR, 2023. 2

[49] L. Kneip, D. Scaramuzza, and R. Siegwart, “A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation,” in CVPR, 2011. 2

[50] T. Ke and S. I. Roumeliotis, “An efficient algebraic solution to the perspective-three-point problem,” in CVPR, 2017. 2

[51] R. Hartley and A. Zisserman, Multiple view geometry in computer vision. Cambridge university press, 2003. 2, 4, 5

[52] R. Hartshorne, Algebraic geometry. Springer Science & Business Media, 2013, vol. 52. 2

[53] V. Lepetit, F. Moreno-Noguer, and P. Fua, “Epnp: An accurate o(n) solution to the pnp problem,” International Journal of Computer Vision, vol. 81, no. 2, p. 155166, 2009. 3, 7

[54] J. Liu, Z. Cao, Y. Tang, X. Liu, and M. Tan, “Category-level 6d object pose estimation with structure encoder and reasoning attention,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 10, pp. 6728–6740, 2022. 3

[55] G. Zhou, D. Wang, Y. Yan, H. Chen, and Q. Chen, “Semi-supervised 6d object pose estimation without using real annotations,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 8, pp. 5163–5174, 2021. 3

[56] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in CVPR, 2025. 3

[57] R. Zhang, R. Geng, Y. Li, R. Song, H. Gong, D. Zhang, Y. Hu, and Y. Chen, “Rfmamba: Frequency-aware state space model for rf-based human-centric perception,” in ICLR, 2025. 3

[58] X. Zhou, K. Larintzakis, H. Guo, W. Zimmer, M. Liu, H. Cao, J. Zhang, V. Lakshminarasimhan, L. Strand, and A. Knoll, “TUMTraf videoQA: Dataset and benchmark for unified spatio-temporal video understanding in traffic scenes,” in ICML, 2025. 3

[59] P.-E. Sarlin, A. Unagar, M. Larsson, H. Germain, C. Toft, V. Larsson, M. Pollefeys, V. Lepetit, L. Hammarstrand, F. Kahl et al., “Back to the feature: Learning robust camera localization from pixels to pose,” in CVPR, 2021. 3

[60] H. Cao, Z. Zhang, Y. Xia, X. Li, J. Xia, G. Chen, and A. Knoll, “Embracing events and frames with hierarchical feature refinement network for object detection,” in ECCV, 2024. 3

[61] W. Chen, H. Li, Q. Nie, and Y.-H. Liu, “Deterministic point cloud registration via novel transformation decomposition,” in CVPR, 2022. 3

[62] Y. Liu, C. Wang, Z. Song, and M. Wang, “Efficient global point cloud registration by matching rotation invariant features through translation search,” in ECCV, 2018. 3

[63] H. Yang, J. Shi, and L. Carlone, “Teaser: Fast and certifiable point cloud registration,” IEEE Transactions on Robotics, vol. 37, no. 2, pp. 314– 333, 2021. 3

[64] Y. Liu, X. Li, M. Wang, A. Knoll, G. Chen, and Z. Song, “A novel method for the absolute pose problem with pairwise constraints,” Remote Sensing, vol. 11, no. 24, p. 3007, 2019. 3

[65] Z. Kukelov´ a, M. Bujnak, and T. Pajdla, “Closed-form solutions to´ minimal absolute pose problems with known vertical direction,” in ACCV, 2010. 3, 7

[66] H. Abdellali and Z. Kato, “Absolute and relative pose estimation of a multi-view camera system using 2d-3d line pairs and vertical direction,” in 2018 Digital Image Computing: Techniques and Applications (DICTA). IEEE, 2018, pp. 1–8. 3

[67] Y. Wu, J. Jiang, Y. Yuan, M. Gong, Q. Miao, H. Li, M. Zhang, and wenping ma, “Pointtruss: K-truss for point cloud registration,” in NeurIPS, 2025. 5

[68] J. L. Moratalla, P. S. S. Carrillo, and D. A. S <sup>´</sup> anchez, “Clireg: Clique-´ based robust point cloud registration,” IEEE Transactions on Robotics, vol. 41, pp. 1898–1917, 2025. 5

[69] S. Yan, P. Shi, Z. Zhao, K. Wang, K. Cao, J. Wu, and J. Li, “Turboreg: Turboclique for robust and efficient point cloud registration,” in ICCV, 2025. 5

[70] D. P. Bertsekas, Constrained optimization and Lagrange multiplier methods. Academic press, 2014. 6

[71] J. Richter-Gebert, Perspectives on projective geometry: a guided tour through real and complex geometry. Springer, 2011. 6

[72] R. Hartley and H. Li, “An efficient hidden variable approach to minimalcase camera motion estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 12, pp. 2303–2314, 2012. 6

[73] X.-S. Gao, X.-R. Hou, J. Tang, and H.-F. Cheng, “Complete solution classification for the perspective-three-point problem,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 25, no. 8, pp. 930–943, 2003. 7

[74] T. Ke and S. I. Roumeliotis, “An efficient algebraic solution to the perspective-three-point problem,” in CVPR, 2017. 7

[75] C. Sweeney, J. Flynn, B. Nuernberger, M. Turk, and T. Hllerer, “Efficient computation of absolute pose for gravity-aware augmented reality,” in IEEE International Symposium on Mixed and Augmented Reality, 2015. 7

[76] G. Terzakis and M. Lourakis, “A consistently fast and globally optimal solution to the perspective-n-point problem,” in ECCV, 2020. 7

[77] D. Barath, “Superansac: One ransac to rule them all,” arXiv preprint arXiv:2506.04803, 2025. 7

[78] P. Roch, B. Shahbaz Nejad, M. Handte, and P. J. Marron, “Axes-aligned´ non-linear optimized pnp algorithm,” Machine Vision and Applications, vol. 35, no. 6, p. 137, 2024. 7, 10

[79] R. Hartley, J. Trumpf, Y. Dai, and H. Li, “Rotation averaging,” International Journal of Computer Vision, vol. 103, pp. 267–305, 2013. 7

[80] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of rgb-d slam systems,” in IROS, 2012. 9, 13

[81] T. Schops, T. Sattler, and M. Pollefeys, “BAD SLAM: Bundle adjusted¨ direct RGB-D SLAM,” in CVPR, 2019. 9, 13

[82] W. Maddern, G. Pascoe, C. Linegar, and P. Newman, “1 year, 1000 km: The oxford robotcar dataset,” The International Journal of Robotics Research, vol. 36, no. 1, pp. 3–15, 2017. 9, 13

[83] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “Orb: An efficient alternative to sift or surf,” in ICCV, 2011. 9

[84] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International Journal of Computer Vision, vol. 60, pp. 91–110, 2004. 9

[85] R. Mur-Artal and J. D. Tardos, “ORB-SLAM2: an open-source SLAM´ system for monocular, stereo and RGB-D cameras,” IEEE Transactions on Robotics, vol. 33, no. 5, pp. 1255–1262, 2017. 11, 13