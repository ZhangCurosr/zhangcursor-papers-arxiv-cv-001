# DLOSAC: Joint Spatial-Geometric Image Estimation via Delaunay-Guided Local Optimization RANSAC

Yu Liu<sup>1</sup>, Meiqi Zhou<sup>2</sup>, Zipei Fan<sup>1,†</sup>, Guanyuan Feng<sup>2</sup>, and Xuan Song<sup>1</sup>

Abstract— Reliable image geometric estimation from noisy matching points is typically addressed using RANSAC, often combined with local optimization to refine inliers and model parameters. However, existing paradigms are predominantly residual-centric; even heuristic attempts to enforce spatial uniformity suffer from a sequential, two-stage nature that inherently decouples distribution optimization from geometric refinement, preventing an effective trade-off. In this paper, we propose DLOSAC, a local optimization framework that jointly incorporates both geometric consistency and spatial distribution information. We begin by constructing a Delaunay triangulation over candidate inliers to explicitly encode neighborhood topology. The geometric module introduces an edge-based consistency metric to evaluate geometric errors, while the spatial module derives distribution descriptors from triangle areas and edge attributes to quantify the clustering of inliers. Finally, we combine edge-based geometric constraints, spatial distribution regularization, and traditional point-level geometric constraints to improve the quality of geometric parameters. Extensive experiments on six publicly available realworld datasets, covering three types of geometric estimation tasks, achieve state-of-the-art performance.

## I. INTRODUCTION

Reliable image geometric estimation from noisy matching points is a key component of many robotic perception systems, including visual localization [1], [2], autonomous navigation [3], [4], and SLAM [4], [5], [6]. The problem is typically formulated as estimating optimal motion parameters from two or more frames by solving an optimization problem [7]. A standard pipeline comprises a front end and a back end: the front end extracts and matches feature points from raw sensor data (e.g., camera images), whereas the back end estimates geometric quantities (e.g., homography, fundamental and essential matrices, and relative poses). However, imaging noise and errors in feature extraction and matching often cause front-end data to deviate from ideal geometric assumptions, thereby complicating back-end estimation [8], [9], [10].

The Random Sample Consensus (RANSAC) algorithm [11] robustly estimates geometric model parameters under high outlier rates by repeatedly randomly sampling the minimal dataset, generating model hypotheses, and filtering out outliers based on consistency scores. Additionally, the algorithm can improve model fitting accuracy through local optimization (LO) strategies [12], such as point-based geometric consistency assumptions [12], [13] and spatial neighborhood consistency assumptions [14], [15], [16], [17], which refine the geometric error of inliers. However, despite the efficacy of LO, current paradigms remain fundamentally tethered to point-level geometric residual minimization. While bucketing-based heuristic approaches [18], [6], [19] attempt to impose spatial constraints, their sequential, two-stage nature decouples spatial distribution optimization from residual minimization, precluding an effective trade-off between the two. This motivates a central question: How to formulate a unified optimization framework that transcends the limitations of point-based geometric verification, harmonizing the trade-off between spatial uniformity and geometric consistency.

We propose DLOSAC, a local optimization framework that jointly leverages geometric consistency and spatial distribution. DLOSAC first constructs a topological graph over the inlier set via Delaunay triangulation, explicitly encoding neighborhood relations. Next, an edge-level geometric consistency metric is introduced and coupled with the conventional point-level hypothesis to refine inlier quality assessment. To capture spatial structure, distribution descriptors derived from triangle areas and edge attributes are designed to quantify inlier clustering. Finally, these cues are fused into an inlier weighting function for precise inlier selection, thereby improving the robustness and accuracy of model estimation.

In summary, our main contributions are as follows:

• We propose DLOSAC, a novel local optimization RANSAC framework that jointly models geometric consistency and spatial distribution to refine robust geometric estimation.

• We define a unified geometric consistency model, which couples a novel edge-based geometric measure with traditional point-based metrics to quantify geometric error, and a spatial distribution model to assess inlier clustering, thereby jointly improving model estimation.

• Extensive experiments on six public real-world datasets across three geometric estimation tasks validate that DLOSAC significantly outperforms state-of-the-art methods in both accuracy and robustness.

## II. RELATED WORK

## A. Image Geometric Estimation

Geometric estimation is a fundamental problem in computer vision, serving as a cornerstone for key applications such as visual localization [1], [2], autonomous navigation [3], [4], and SLAM [4], [5], [6]. A typical pipeline begins with raw sensor data (e.g., camera images), from which local features are extracted and matched across views using algorithms (e.g., SIFT [20], SURF [21], ORB [22]) to establish a set of correspondences. Subsequently, these correspondences are used to estimate multi-view geometric parameters (e.g., the fundamental matrix, homography matrix, essential matrix, or relative pose) using methods like the eight-point [23], normalized eight-point [24], or five-point algorithms [25]. However, due to imaging noise and feature matching errors, this correspondence set often contains both inaccuracies (inliers with noise) and gross outliers [14]. Therefore, practical systems typically employ robust estimation algorithms (e.g., RANSAC [11] and its variants) to reject outliers, thereby ensuring a consistent and accurate estimation of the geometric parameters.

Moving beyond the standard RANSAC pipeline, we aim to explore a new local optimization RANSAC framework that jointly optimizes geometric consistency and the spatial distribution of inliers to improve geometric estimation accuracy and robustness.

## B. Local Optimization RANSAC

Random Sample Consensus (RANSAC) [11] is primarily used to fit models to data containing a large proportion of outliers. Its core idea is to repeatedly draw minimal sample sets to generate candidate models, and then evaluate each model by the size of its inlier consensus set or by a residual metric, thereby obtaining parameter estimates that are robust to outliers. Within this framework, accuracy and efficiency can be improved along five dimensions: optimizing the sampling strategy [10], [26], [9], refining modelquality evaluation [8], [7], [27], designing adaptive termination criteria [28], [8], [9], [26], introducing pre-emptive verification [28], and applying local optimization [12], [13], [14], [15]. This paper primarily focuses on local optimization strategies, wherein parameters are re-estimated from inliers identified by an initial model (e.g., via nonlinear refinement or reweighted least squares) to further improve fitting accuracy [12]. To balance robustness and efficiency, a truncated quadratic cost function can be employed, and inliers can be selectively admitted to the optimization, maintaining fit quality while reducing computation [13]. Building on this idea, subsequent work incorporates spatial neighborhood consistency into inlier selection either encouraging adjacent points to share the same labels [14] or to exhibit identical pixel displacement transformations [15] thereby improving outlier partitioning and enhancing the coherence and overall robustness of the inlier set.

Prior local optimization has predominantly focused on point-level geometry. We propose an edge patch collaborative strategy that enforces geometric consistency at the edge level and models spatial distributions at the patch level, providing complementary cues that improve geometric estimation accuracy.

## III. PROBLEM FORMULATION

Given N matched keypoints $\mathcal { M } = \{ ( \mathbf { x } _ { i } , \mathbf { x } _ { i } ^ { \prime } ) \} _ { i = 1 } ^ { N }$ , where $\mathbf { x } _ { i } ~ = ~ [ x _ { i } , y _ { i } , 1 ] ^ { \top } ~ \in ~ \mathbb { P } ^ { 2 }$ and $\mathbf { x } _ { i } ^ { \prime } = [ x _ { i } ^ { \prime } , y _ { i } ^ { \prime } , 1 ] ^ { \top } \in \mathbb { P } ^ { 2 }$ are the homogeneous coordinates of corresponding points in the first (query) and second (reference) images, the goal of geometric parameter estimation is to find θ that best satisfies the geometric constraints induced by these correspondences. We formulate the problem as the weighted least-squares minimization:

$$
\underset { \pmb { \theta } } { \operatorname* { m i n } } \sum _ { i = 1 } ^ { N } w _ { i } \left\| f ( \mathbf { x } _ { i } ^ { \prime } , \pmb { \theta } , \mathbf { x } _ { i } ) \right\| _ { 2 } ^ { 2 } ,\tag{1}
$$

where $f ( \cdot )$ denotes the constraint residual associated with the chosen geometric model. For example, if the model is the fundamental matrix F, then $f ( \mathbf { x } _ { i } ^ { \prime } , \mathbf { F } , \mathbf { x } _ { i } ) = { \mathbf { x } _ { i } ^ { \prime } } ^ { \top } \mathbf { F } \mathbf { x } _ { i }$ (epipolar constraint); if the model is a homography H, then $f ( \cdot ) = \mathbf { x } _ { i } ^ { \prime } \times \left( \mathbf { H } \mathbf { x } _ { i } \right)$ (projective constraint); and if the model is the relative pose [R, t], then $f ( \cdot ) = { \mathbf { x } _ { i } ^ { \prime } } ^ { \top } { \mathbf { E x } _ { i } }$ with ${ \bf E } =$ $[ \mathbf { t } ] _ { \times } \mathbf { R }$ (rigid-motion constraint). The weights $w _ { i }$ modulate the influence of each correspondence to improve robustness.

Our goal is to introduce Delaunay triangulation to redefine the optimization weight matrix $\mathbf { W } = \{ w _ { i } \} _ { i = 1 } ^ { N }$ , taking into account both geometric consistency and spatial distribution.

## IV. METHODOLOGY

## A. Preliminaries

1) Delaunay triangulation: Delaunay triangulation (DT) is a widely used spatial discretization method that partitions a given planar point set $\mathcal { P } = \{ { \bf p } _ { i } \} _ { i = 1 } ^ { n } \subset \mathbb { R } ^ { 2 }$ into a collection of non-overlapping triangular elements. Its fundamental principle is the empty circumcircle property: for any triangle in the triangulation, the circumcircle contains no other points from the set except the triangle’s three vertices [29].

DT maximizes the minimum angle among all triangles, effectively preventing the formation of highly acute or degenerate elements and ensuring a more uniform and numerically stable mesh. Moreover, Delaunay triangulation naturally induces a graph structure $\mathcal { G } = ( \nu , \mathcal { E } )$ , where $\nu = \mathcal { P }$ denotes the set of nodes and $\mathcal { E } = \{ ( \mathbf { p } _ { i } , \mathbf { p } _ { j } ) \mid \mathbf { p } _ { i } , \mathbf { p } _ { j } \in \mathcal { P } \}$ defines the adjacency relationships among them. This geometry-induced graph captures the intrinsic local topology of the point set and has been widely applied in medical imaging, computer vision, and robotic downstream tasks [30].

2) Local Optimization RANSAC: Image geometric estimation seeks to recover scene or camera motion parameters from feature correspondences across multiple views. Let $\mathcal { M } = \{ ( \mathbf { x } _ { i } , \mathbf { x } _ { i } ^ { \prime } ) \} _ { i = 1 } ^ { N }$ denote the set of putative correspondences, θ the model parameters (e.g., a fundamental or essential matrix), and $\varepsilon ( \mathbf { x } _ { i } , \mathbf { x } _ { i } ^ { \prime } ; \mathbf { \boldsymbol { \theta } } )$ a geometric residual (e.g., the symmetric epipolar distance). Given an inlier threshold $\tau > 0 ,$ , a correspondence $( \mathbf { x } _ { i } , \mathbf { x } _ { i } ^ { \prime } )$ is considered an inlier if $\varepsilon ( \mathbf { x } _ { i } , \mathbf { x } _ { i } ^ { \prime } ; \pmb { \theta } ) < \tau$ . RANSAC [11] formulates estimation as a maximum-consensus (maximum inlier count) problem:

$$
\hat { \pmb { \theta } } = \underset { \pmb { \theta } \in \Theta } { \arg \operatorname* { m a x } } I ( \pmb { \theta } ) , \quad I ( \pmb { \theta } ) = \sum _ { i = 1 } ^ { N } \mathbf { 1 } \big [ \varepsilon \big ( \mathbf { x } _ { i } , \mathbf { x } _ { i } ^ { \prime } ; \pmb { \theta } \big ) < \tau \big ] ,\tag{2}
$$

![](images/6c5c8db2c1ecd9b7742e16bf3e711c95a37d42e6e8481d7e22dca9ce145c54d3.jpg)  
Fig. 1: Overview of our framework. Given RGB images from different viewpoints, initial matches are obtained through ORB feature extraction and refined using RANSAC. A Delaunay triangulation is then built over the inliers to capture neighborhood relationships, followed by the computation of spatial distribution and geometric consistency losses. Finally, a weighted matrix is used to optimize the geometric parameters.

![](images/24e1013fe61256fc3b0503d821ed9a4463e62e8c9fab010e9a62588b91417932.jpg)  
Fig. 2: Edge-based geometric consistency models: (a) rotation and (b) translation.

where Θ is the feasible parameter space and $\mathbf { 1 } [ \cdot ]$ denotes the indicator function, which equals 1 if the condition holds and 0 otherwise.

The algorithm proceeds iteratively by (i) sampling a minimal subset from M to generate a hypothesis $\theta _ { k } ,$ (ii) scoring $\pmb { \theta } _ { k }$ by evaluating $I ( \pmb \theta _ { k } )$ over all matches, and (iii) retaining the highest-scoring hypothesis; iterations terminate once a confidence-driven stopping criterion is met, at which point the hypothesis with the largest consensus set is selected. To further enhance accuracy and stability, LO-RANSAC [12] then performs localized optimization when a sufficiently large consensus set has been identified: letting $\pmb { \theta } ^ { * }$ denote the current best model with inliers $I ^ { * } .$ the parameters are re-estimated from all inliers (or a filtered subset $\tilde { I } ^ { * } \subseteq I ^ { * } )$ which reduces threshold τ induced bias, improves numerical conditioning, and delivers superior performance under measurement noise and non-uniform outlier distributions.

## B. Proposed Local Optimization RANSAC

1) Edge-Based Geometric Consistency Loss: For clarity, we first analyze edge-based geometric consistency under a simplified camera motion model and then generalize the conclusions to general camera movement. As shown in Fig. 2(a), the matched points $\mathbf { p } _ { a }$ and $\mathbf { p } _ { b }$ in the query image are transformed into the corresponding points $\mathbf { p } _ { a ^ { \prime } }$ and $\mathbf { p } _ { b ^ { \prime } }$ in the reference image after the camera undergoes rotation. The adjacency relationships in the query image (such as $\mathbf { p } _ { a }$ and $\mathbf { p } _ { b } )$ are represented by the line segment $\ell ( \mathbf { p } _ { a } , \mathbf { p } _ { b } )$ via Delaunay triangulation, while the adjacency relationships in the reference image (such as $\ell \big ( \mathbf { p } _ { a ^ { \prime } } , \mathbf { p } _ { b ^ { \prime } } \big ) \big )$ are determined through point correspondences between the query and reference images. From the figure, it can be observed that after the camera undergoes a rotational transformation, the lengths of the adjacency segments $\ell ( \mathbf { p } _ { a } , \mathbf { p } _ { b } )$ and $\ell \left( \mathbf { p } _ { a ^ { \prime } } , \mathbf { p } _ { b ^ { \prime } } \right)$ remain consistent. However, when an incorrect match occurs, such as when $\mathbf { p } _ { b }$ is erroneously matched to $\mathbf { p } _ { c ^ { \prime } }$ , the distance between adjacent points (such as $\ell ( \mathbf { p } _ { a } , \mathbf { p } _ { b } )$ and $\ell \big ( \mathbf { p } _ { a ^ { \prime } } , \mathbf { p } _ { c ^ { \prime } } \big ) \big )$ will deviate. Accordingly, we introduce a regularization constraint: under pure rotation, edges connecting neighboring points produced by correct correspondences exhibit smaller length variations between the query and reference images than those produced by mismatches.

Fig. 2(b) illustrates the correspondences between an adjacent feature point pair, $\mathbf { p } _ { a ^ { \prime } }$ and $\mathbf { p } _ { b ^ { \prime } }$ , in the query image and their respective matches, ${ \bf p } _ { a ^ { \prime \prime } }$ and $\mathbf { p } _ { b ^ { \prime \prime } }$ , in the reference image following camera translation. According to the principle of triangulation [31], an object’s depth is inversely proportional to its parallax. Specifically, when the depth difference between neighboring points $\mathbf { p } _ { a ^ { \prime } }$ and $\mathbf { p } _ { b ^ { \prime } }$ in the query image is small, their pixel displacements across viewpoints are also small. Consequently, the connecting edges $\ell \left( \mathbf { p } _ { a ^ { \prime } } , \mathbf { p } _ { b ^ { \prime } } \right)$ and $\ell ( \mathbf { p } _ { a ^ { \prime \prime } } , \mathbf { p } _ { b ^ { \prime \prime } } )$ in the query and reference images, respectively, are more likely to maintain consistent lengths. Conversely, a large depth difference may lead to correspondences such as $( { \bf p } _ { a ^ { \prime } } \to { \mathrm { ~ \bf ~ p } } _ { c ^ { \prime } } )$ and $( \mathbf { p } _ { b ^ { \prime } }   \mathbf { p } _ { b ^ { \prime \prime } } )$ . Even when these correspondences are correct, the edge lengths $\ell \left( \mathbf { p } _ { a ^ { \prime } } , \mathbf { p } _ { b ^ { \prime } } \right)$ and $\ell \left( \mathbf { p } _ { c ^ { \prime } } , \mathbf { p } _ { b ^ { \prime \prime } } \right)$ can still be inconsistent between the query and reference images. Based on the preceding analysis, we define the edge-based geometric consistency loss $( \mathcal { L } _ { \mathrm { e d g e } } )$ as:

$$
{ \mathcal { L } } _ { \mathrm { e d g e } } = \left( 1 - d _ { a b } \right) \cdot \left( \ell ( \mathbf { p } _ { a ^ { \prime } } , \mathbf { p } _ { b ^ { \prime } } ) - \ell ( \mathbf { p } _ { a ^ { \prime \prime } } , \mathbf { p } _ { b ^ { \prime \prime } } ) \right) ^ { 2 } ,\tag{3}
$$

where $d _ { a b }$ denotes the pixel distance between neighboring points $\mathbf { p } _ { a ^ { \prime } }$ and $\mathbf { p } _ { b ^ { \prime } }$ , normalized by the length of the image diagonal. Inspired by [15], we introduce the weight $\left( 1 - d _ { a b } \right)$ points that are closer in the image (small $d _ { a b } )$ are more likely to have similar depths and thus to satisfy the geometric consistency constraint on adjacent edges. Conversely, points that are farther apart (large $d _ { a b } )$ are more likely to exhibit significant depth differences, increasing the probability that the edge connecting them becomes inconsistent across views.

![](images/6d2eef836e9172cf6497b7c7b190c3ffae654a06832837417824d7f7940dc93c.jpg)  
Fig. 3: Illustration of an instance of edge-based geometric consistency.

This assumption extends to general camera models, as distant points are more susceptible to perspective-induced geometric distortions and therefore deviate more from the imposed constraint.

Note that the $\mathcal { L } _ { \mathrm { e d g e } }$ is a discriminative prior derived under a simplified camera model rather than a physical law. In practice, mismatches often result in substantial pixel-level deviations. Consequently, even under general camera motion model, this prior is also more likely to be violated (see Fig. 3). More importantly, we use $\mathcal { L } _ { \mathrm { e d g e } }$ solely as a regularizer (details in Sec. IV-B.3), while the primary constraints are imposed by point-based geometric errors (e.g., epipolar constraint or homography residuals).

2) Spatial Distribution Loss: As shown in Fig. 3, after Delaunay triangulation, each matched point serves as a shared vertex for multiple adjacent triangles. To quantify each point’s spatial distribution, we utilize the areas and largest internal angles of its adjacent triangles. These geometric properties are then used to define the Area Descriptor $( D _ { A } ^ { ( i ) } )$ and the Shape Descriptor $( D _ { S } ^ { ( i ) } )$ as follows:

$$
D _ { A } ^ { ( i ) } = \sqrt { \frac { 1 } { m _ { i } - 1 } \sum _ { j = 1 } ^ { m _ { i } } { \left( \frac { A ( T _ { i j } ) \cdot n } { \sum _ { k = 1 } ^ { n } A ( T _ { k } ) } - 1 \right) ^ { 2 } } } ,\tag{4}
$$

$$
D _ { S } ^ { ( i ) } = \sqrt { \frac { 1 } { m _ { i } - 1 } \sum _ { j = 1 } ^ { m _ { i } } \left( \frac { 3 \cdot \operatorname* { m a x } ( J _ { i j } ) } { \pi } - 1 \right) ^ { 2 } } ,\tag{5}
$$

where $m _ { i }$ is the number of triangles adjacent to the i-th matched keypoint, and n is the total triangle count in the triangulation. $A ( T _ { i j } )$ denotes the area of the j-th triangle adjacent to keypoint $i , \sum _ { k = 1 } ^ { n } A ( T _ { k } )$ is the total area of all n triangles, and max $( J _ { i j } )$ is the largest internal angle (in radians) of triangle $T _ { i j }$

Smaller values for the Area Descriptor $( D _ { A } ^ { ( i ) } )$ and the Shape Descriptor $( D _ { S } ^ { ( i ) } )$ indicate that the point’s adjacent triangles are closer to equilateral and their areas are nearer the global mean, respectively. This implies a more uniform spatial distribution for the matched point. Since clustered points can introduce local errors, we aim to promote uniformity by minimizing these descriptors. We thus define the spatial distribution loss for the i-th point, $L ^ { ( i ) }$ , as:

Algorithm 1 Delaunay-Guided Local Optimization   
Require: Initial match set M; Balancing factor $\lambda _ { \mathrm { s p a t i a l } } ;$   
Scaling parameter $\sigma ^ { 2 }$   
Ensure: Estimated geometric parameters θ   
1: $\mathcal { P } _ { \mathrm { i n l i e r } }  \mathrm { R A N S A C } ( \mathcal { M } )$   
2: $\mathcal G ( \nu , \mathcal { E } )$ ← DelaunayTriangulation $\left( \mathcal { P } _ { \mathrm { i n l i e r } } \right)$   
3: $W  \emptyset$ ▷ Initialize an empty set for weights   
4: for each node $\mathbf { p } \in \mathcal { V }$ do   
5: $\bar { \mathcal { L } } _ { \mathrm { e d g e } }  \bf M$ eanEdgeLoss(p, G) ▷ per Eq. (3)   
6: L<sup>¯</sup><sub>spatial</sub> ← MeanSpatialL $\mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { 0 } \mathbf { S } \mathbf { S } } } ( \mathbf { p } , \mathcal { G } )$ ▷ per Eq. (6)   
7: $\mathcal { L } _ { p }  \bar { \mathcal { L } } _ { \mathrm { e d g e } } + \lambda _ { \mathrm { s p a t i a l } } \bar { \mathcal { L } } _ { \mathrm { s p a t i a l } }$   
8: $w _ { p } \gets \exp ( - \mathcal { L } _ { p } / \sigma ^ { 2 } )$   
9: $\mathbf { W }  \mathbf { W } \cup \{ w _ { p } \}$   
10: end for   
11: $\pmb \theta \gets \mathbf W \mathbf L \mathbf S ( \gamma , \mathbf W )$ ▷ per Eq. (1)   
12: return θ

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s p a t i a l } } ^ { ( i ) } = D _ { A } ^ { ( i ) } \cdot D _ { S } ^ { ( i ) } . } \end{array}\tag{6}
$$

3) Delaunay-Guided Local Optimization: An overview of the proposed Delaunay-Guided Local Optimization RANSAC (DLOSAC) framework is shown in Fig. 1. The pipeline begins by constructing a Delaunay triangulation on the inlier set $\mathcal { P } _ { \mathrm { i n l i e r } } \ \in \ \mathcal { M }$ , thereby establishing topological neighborhood relationships among the matches. Subsequently, for each node (keypoint) $\mathbf { p } \in \mathcal { P } _ { \mathrm { i n l i e r } }$ within the triangulation, we define two structured losses: (i) the mean edge geometric consistency loss $\bar { \mathcal { L } } _ { \mathrm { e d g e } }$ over all its adjacent edges (Eq. 3), and (ii) the mean spatial distribution loss $\bar { \mathcal { L } } _ { \mathrm { s p a t i a l } }$ over all its incident triangles (Eq. 6).

To derive a robust weight $w _ { p }$ for each node, we first compute a unified loss $\mathcal { L } _ { p } = \bar { \mathcal { L } } _ { \mathrm { e d g e } } + \lambda _ { \mathrm { s p a t i a l } } \bar { \mathcal { L } } _ { \mathrm { s p a t i a l } } .$ , where the $\lambda _ { \mathit { s p a t i a l } }$ coefficients act as balancing factors. This unified loss is then transformed into a weight via an exponential decay function, $w _ { p } = \exp ( - \mathcal { L } _ { p } / \sigma ^ { 2 } )$ , where $\sigma ^ { 2 }$ is a scaling parameter. This weighting scheme ensures that nodes with higher geometric consistency and better spatial distribution (i.e., lower $\mathcal { L } _ { p } )$ receive exponentially higher weights, while outliers or poorly-structured nodes are effectively downweighted. Finally, these node-specific weights $w _ { p }$ are employed to solve for the geometric parameters θ using the Weighted Least-Squares (WLS) formulation (Eq. 1).

## V. EXPERIMENTS

The experimental section is organized to demonstrate the effectiveness and robustness of our method. Specifically, Sec. V-B provides a comparative study against state-of-theart geometric estimation techniques. Sec. V-C presents an ablation study to quantify the benefits of the edge–geometry consistency and spatial-distribution losses. Sec. V-D evaluates the influence of critical hyperparameters on the final estimation quality.

## A. Experimental Setup

1) Benchmark: Following the benchmark of [14], we evaluate three tasks: fundamental matrix estimation, essential matrix (relative pose) estimation, and homography estimation. For the first two tasks, we conduct experiments on four real-world datasets: TUM [32], KITTI [33], Tanks and Temples (T&T) [34], and Community Photo Collection (CPC) [35]. TUM consists of indoor video sequences with a frame resolution of $6 4 0 \times 4 8 0$ pixels; KITTI comprises consecutive frames captured by a vehicle-mounted camera with an image resolution of $1 2 2 6 \times 3 7 0$ pixels. In both datasets, most image pairs have short baselines, making epipolar-geometry estimation relatively easy, with a high inlier ratio. T&T provides real-object images for reconstruction, mostly wide-baseline pairs, with resolutions ranging from 1080 × 1920 to $1 0 8 0 \times 2 0 4 8$ pixels. CPC collects landmark photos from Flickr with varying resolutions and is likewise dominated by wide-baseline pairs. For homography estimation, we use two datasets, Homo (16 image pairs) and EVD (15 image pairs) [13]. They provide point correspondences, and image sizes span from 329×278 to $1 7 1 2 \times 1 7 1 2$ pixels. To construct the evaluation benchmark, we uniformly sample 1,000 image pairs from each dataset with more than 1,000 images, as shown in Fig. 4.

2) Baselines: RANSAC [11], with local optimization disabled, is used as the baseline to evaluate the accuracy and stability of robust estimation. LO-RANSAC [12] performs local optimization (e.g., iterative least squares) on the inlier set of the current best model to refine both the model parameters and the inlier membership. LO<sup>+</sup>-RANSAC [13] extends the LO stage with a truncated-quadratic loss and an upper bound on the inlier count. GC-RANSAC [14] employs graph-cut energy minimization to jointly exploit residual magnitudes and spatial coherence for inlier/outlier labeling and subsequent model refinement. PSC-RANSAC [15] clusters correspondences in displacement space to remove mismatches, followed by model refinement.

3) Evaluation Metrics: We evaluate the accuracy of fundamental matrix and homography estimation using the mean $\epsilon _ { \mathrm { a v g } }$ (in pixel) and standard deviation $\boldsymbol { \epsilon } _ { \mathrm { s t d } }$ (in pixel) of the pixel error. The improvement rate $f _ { \mathrm { i m p } }$ quantifies the performance of a competing method relative to ours and is defined as:

$$
f _ { \mathrm { i m p } } ( \epsilon _ { \mathrm { a v g } } ) = \frac { \epsilon _ { \mathrm { a v g } } ^ { \mathrm { b a s e l i n e } } - \epsilon _ { \mathrm { a v g } } ^ { \mathrm { o u r s } } } { \epsilon _ { \mathrm { a v g } } ^ { \mathrm { b a s e l i n e } } } \times 1 0 0 \%\tag{7}
$$

For relative pose estimation, we report the rotation error $\epsilon _ { \mathrm { R } }$ (in degree) and the translation error $\epsilon _ { \mathrm { t } }$ (in degree), and define $f _ { \mathrm { i m p } } ( \epsilon _ { \mathrm { a v g } } )$ , where $\epsilon _ { \mathrm { a v g } } = \frac { 1 } { 2 } \left( \epsilon _ { \mathrm { R } } + \epsilon _ { \mathrm { t } } \right)$ , to characterize the overall error level.

## B. Comparison with state-of-the-art methods

1) Quantitative Geometry Estimation Evaluation: Table I summarizes the results of three image-geometry estimation tasks (1st column) evaluated on six benchmark datasets and compares them with the strongest baselines (1st row). For each task, we additionally report the cross-dataset average (Avg.). Under a unified evaluation protocol, DLOSAC achieves state-of-the-art performance across all three tasks, improving over the strongest baseline to date by (+12.85%) on fundamental-matrix estimation, (+13.85%) on relativepose estimation, and (+15.88%) on homography estimation.

![](images/4773dba742214cfdc46848fda3a1ec86c8b37e682784472576d3be41dd17f621.jpg)

![](images/d57edffa3d6c2567684669eb3e29a34c7c3d7dc0f4a89b0c4aec4b4dc00d8d02.jpg)

(a)  
![](images/9be178c48e5e486732a368f652a9962ff14b9fdbb051a8e1493c263fe47978ce.jpg)

(b)  
![](images/da41cd7564f0d791fb35d13a4cbc94d97c6ec3e10d503b96e2accdca34eb5986.jpg)

![](images/a62bf83beef9f7a311e7d507b3464e38d4605e875130995d7f2ca66cec2e014a.jpg)

(c)  
![](images/fbad4d21d490e4522e490243fab06ed317b650a864eda57aeceedc747606c728.jpg)  
(d)

![](images/92be1ca0a212319781aa510609a463fa7f94aba6a2b480dffc1bc5c1554cc6ce.jpg)

![](images/ae8585c73abb54f13cde3c5a7d417351a3546ea80b6eace0c26a21c410d3bc13.jpg)  
(e)  
Fig. 4: Delaunay triangulation results and corresponding matching instances for image evaluation benchmarks: (a) TUM, (b) KITTI, (c) T&T, (d) CPC, (e) EVD.

The performance of each method is analyzed as follows: RANSAC [11] lacks local optimization and is therefore highly sensitive to threshold selection and noise, yielding the weakest overall performance. LO-RANSAC [12] and $\mathrm { L O ^ { + } { \mathrm { - R A N S A C } } }$ [13] introduce local optimization on pointwise geometric errors; the latter further refines the inlier set through a more efficient sampling strategy, thereby improving model fit. However, these local refinements and repeated evaluations increase computational cost, and the methods can become trapped in suboptimal solutions when inliers are unevenly distributed or local minima are present. As strong baselines, GC-RANSAC [14] and PSC-RANSAC [15] explicitly incorporate a neighborhood consistency prior in addition to point-wise geometric errors where nearby points tend to share inlier/outlier labels or exhibit similar pixel displacements, which improves accuracy and robustness. By contrast, DLOSAC explicitly models both spatial geometric consistency and the distribution of correspondences during local optimization, coupling them with the standard point-wise geometric constraint. This enhances the geometric consistency of correspondences while regularizing their spatial distribution to mitigate cluster-induced overfitting. Notably, T&T [34] and CPC [35] are long-baseline datasets, whereas the others are primarily short baseline. We observe larger gains on the short-baseline datasets; under long baselines, stronger perspective changes and scale/deformation effects undermine the assumptions underlying our geometricconsistency term, leading to more limited improvements. This behavior aligns with the method’s intended operating regime.

TABLE I: The error and improvement rates for all methods (1st row) across tasks (1st column) and datasets (2nd column) are presented, along with the average performance $( \operatorname { A v g } . )$ of each task across benchmarks. The optimal results are indicated in bold, and the second-best results are marked with underline.
<table><tr><td>TTUUM</td><td></td><td>RANSAC [11]</td><td>LO-RANSAC [12]</td><td>LO+-RANSAC [13]</td><td>GC-RANSAC [14]</td><td>PSC-RANSAC [15]</td><td>DLOSAC (ours)</td></tr><tr><td rowspan="5"></td><td> $\epsilon _ { \mathrm { a v g } } \ ( \mathrm { p x } )$ </td><td>0.482</td><td>0.294</td><td>0.247</td><td>0.139</td><td>0.148</td><td>0.116</td></tr><tr><td> $\epsilon _ { \mathrm { s t d } } ~ \mathrm { ( p x ) }$ </td><td>0.478</td><td>0.230</td><td>0.225 53.04</td><td>0.193</td><td>0.190 21.62</td><td>0.130</td></tr><tr><td> $f _ { i m p } \ ( \% )$ </td><td>75.93</td><td>60.54</td><td></td><td>16.55</td><td></td><td>0.00</td></tr><tr><td> $\epsilon _ { \mathrm { a v g } } \ ( \mathrm { p x } )$ </td><td>0.538</td><td>0.402</td><td>0.329</td><td>0.229</td><td>0.204</td><td>0.197</td></tr><tr><td>Estd (px)</td><td>0.501</td><td>0.398</td><td>0.394</td><td>0.263</td><td>0.250</td><td>0.214</td></tr><tr><td>nar Marix T&amp;T</td><td> $f _ { i m p } \ ( \% )$ </td><td>63.38</td><td>51.00</td><td>40.12</td><td>13.97</td><td>3.43</td><td>0.00</td></tr><tr><td>KITI</td><td>€avg (px)</td><td>0.493</td><td>0.338</td><td>0.310</td><td>0.134</td><td>0.154</td><td>0.115</td></tr><tr><td></td><td> $\epsilon _ { \mathrm { s t d } } ~ \mathrm { ( p x ) }$ </td><td>0.582</td><td>0.351</td><td>0.307</td><td>0.201</td><td>0.193</td><td>0.203</td></tr><tr><td></td><td> $f _ { i m p } \ ( \% )$ </td><td>76.67</td><td>65.98</td><td>62.90</td><td>14.18</td><td>25.32</td><td>0.00</td></tr><tr><td>CPC</td><td> $\epsilon _ { \mathrm { a v g } } \ ( \mathrm { p x } )$ </td><td>0.777</td><td>0.573</td><td>0.489</td><td>0.225</td><td>0.210</td><td>0.194</td></tr><tr><td rowspan="4">AS</td><td></td><td>0.757</td><td>0.501</td><td>0.511</td><td>0.313</td><td>0.235</td><td>0.216</td></tr><tr><td> $\epsilon _ { \mathrm { s t d } } ~ \mathrm { ( p x ) }$   $f _ { i m p } \ ( \% )$ </td><td>75.03</td><td>66.14</td><td>60.33</td><td>13.78</td><td>7.62</td><td>0.00</td></tr><tr><td></td><td></td><td>0.402</td><td></td><td></td><td></td><td></td></tr><tr><td> $\epsilon _ { \mathrm { a v g } } \ ( \mathrm { p x } )$   $\epsilon _ { \mathrm { s t d } } ~ \mathrm { ( p x ) }$ </td><td>0.573 0.580</td><td>0.370</td><td>0.344 0.359</td><td>0.182 0.243</td><td>0.179 0.217</td><td>0.156</td></tr><tr><td>TUM</td><td> $f _ { i m p } \ ( \% )$ </td><td>72.77</td><td>61.19</td><td>54.65</td><td>14.29</td><td>12.85</td><td>0.191 0.00</td></tr><tr><td rowspan="10">Ree ose</td><td></td><td>1.247</td><td>0.924</td><td>0.881</td><td>0.498</td><td>0.563</td><td></td></tr><tr><td> $\epsilon _ { R } \ ( ^ { \circ } )$   $\epsilon _ { t } \ ( ^ { \circ } )$ </td><td>14.396</td><td>13.964</td><td>13.151</td><td>13.149</td><td>13.217</td><td>0.471</td></tr><tr><td> $f _ { \mathrm { i m p } } \ ( \% )$ </td><td>34.62</td><td>31.30</td><td>27.11</td><td>25.05</td><td>25.78</td><td>9.757</td></tr><tr><td> $\epsilon _ { R } \ ( ^ { \circ } )$ </td><td>5.195</td><td>4.786</td><td>3.634</td><td></td><td></td><td>0.00</td></tr><tr><td>T&amp;T</td><td></td><td>11.478</td><td></td><td>3.218</td><td>3.089</td><td>2.910</td></tr><tr><td> $\epsilon _ { t } \ ( ^ { \circ } )$   $f _ { \mathrm { i m p } } \ ( \% )$ </td><td>42.78</td><td>9.646 33.90</td><td>9.064 24.87</td><td>8.605 19.31</td><td>7.338 8.51</td><td>6.630</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.00</td></tr><tr><td>KITI  $\epsilon _ { R } \ ( ^ { \circ } )$ </td><td>1.104 7.522</td><td>0.979</td><td>0.905</td><td>0.643</td><td>0.752</td><td>0.590</td></tr><tr><td> $\epsilon _ { t } \ ( ^ { \circ } )$   $f _ { \mathrm { i m p } } \ ( \% )$ </td><td>45.06</td><td>5.546 27.37</td><td>5.515 26.18</td><td>4.956 15.36</td><td>4.806</td><td>4.149</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>14.74</td><td>0.00</td></tr><tr><td>CPC</td><td> $\epsilon _ { R } \ ( ^ { \circ } )$ </td><td>8.985 18.846</td><td>6.642 13.690</td><td>6.192</td><td>5.541</td><td>5.216</td><td>5.307</td></tr><tr><td rowspan="5">AV</td><td> $\epsilon _ { t } \ ( ^ { \circ } )$ </td><td>46.91</td><td>27.33</td><td>13.715</td><td>13.406</td><td>10.618</td><td>9.469</td></tr><tr><td> $f _ { \mathrm { i m p } } \ ( \% )$ </td><td></td><td></td><td>25.77</td><td>22.01</td><td>6.68</td><td>0.00</td></tr><tr><td> $\epsilon _ { R } \ ( ^ { \circ } )$ </td><td>4.133</td><td>3.333</td><td>2.903</td><td>2.475</td><td>2.405</td><td>2.320</td></tr><tr><td> $\epsilon _ { t } \ ( ^ { \circ } )$ </td><td>13.061</td><td>10.712</td><td>10.361</td><td>10.029</td><td>8.995</td><td>7.501</td></tr><tr><td> $f _ { \mathrm { i m p } } \ ( \% )$ </td><td>42.88</td><td>30.07</td><td>25.96</td><td>21.46</td><td>13.85</td><td>0.00</td></tr><tr><td rowspan="8">Ho Hoorgphy EVΛVD</td><td> $\epsilon _ { \mathrm { a v g } } \ ( \mathrm { p x } )$ </td><td>0.920</td><td>0.521</td><td>0.503</td><td>0.349</td><td>0.287</td><td>0.246</td></tr><tr><td>€std (px)</td><td>0.810</td><td>0.833</td><td>0.784</td><td>0.631</td><td>0.569</td><td>0.537</td></tr><tr><td> $f _ { \mathrm { i m p } } \ ( \% )$ </td><td>73.26</td><td>52.78</td><td>51.09</td><td>29.51</td><td>14.29</td><td>0.00</td></tr><tr><td> $\epsilon _ { \mathrm { a v g } } \ ( \mathrm { p x } )$ </td><td>1.973</td><td>1.496</td><td>1.358</td><td>1.049</td><td>1.187</td><td>0.929</td></tr><tr><td> $\epsilon _ { \mathrm { s t d } } ~ \mathrm { ( p x ) }$ </td><td>2.012</td><td>1.611</td><td>1.629</td><td>1.367</td><td>1.480</td><td>1.266</td></tr><tr><td> $f _ { \mathrm { i m p } } \ ( \% )$ </td><td>52.91</td><td>37.90</td><td>31.59</td><td>11.44</td><td>21.74</td><td>0.00</td></tr><tr><td> $\epsilon _ { \mathrm { a v g } } \ ( \mathrm { p x } )$ </td><td>1.447</td><td>1.009</td><td>0.931</td><td>0.699</td><td>0.737</td><td>0.588</td></tr><tr><td>Av Estd (px)</td><td></td><td>1.411</td><td></td><td></td><td>1.025</td><td>0.902</td></tr><tr><td rowspan="4"></td><td></td><td></td><td>1.222</td><td>1.207</td><td>0.999 15.88</td><td>20.22</td><td>0.00</td></tr><tr><td> $f _ { \mathrm { i m p } } \ ( \% )$ </td><td>59.36</td><td>41.72</td><td>36.84</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

2) Runtime Analysis: Table II summarizes the average runtime efficiency on the Fundamental Matrix (F-Matrix), Relative Pose (Rel. Pose), and Homography (Homo.) tasks. Although standard RANSAC [11] and LO-RANSAC [12] exhibit lower latency due to their algorithmic simplicity, this comes at the cost of reduced estimation accuracy, as evidenced in Table I. In comparison to the state-of-theart GC-RANSAC [14], which necessitates computationally intensive graph-cut optimization, our DLOSAC achieves an approximate 39% reduction in runtime. While the primary computational overhead in DLOSAC stems from Delaunay triangulation, this process is performed solely on the subset of candidate inliers rather than the entire correspondence set. Consequently, the computational cost remains tractable, supporting the method’s applicability to real-time robotic systems.

TABLE II: Comparison of average runtime (ms). Our method is highlighted in blue . Bold and underlined values indicate the best and second-best performance, respectively.
<table><tr><td>Method</td><td>F-Matrix</td><td>Rel. Pose</td><td>Homo.</td><td>Avg.</td></tr><tr><td>RANSAC</td><td>18.24</td><td>24.62</td><td>8.26</td><td>17.04</td></tr><tr><td>LO-RANSAC</td><td>29.66</td><td>41.18</td><td>16.41</td><td>29.08</td></tr><tr><td>LO+-RANSAC</td><td>26.43</td><td>38.08</td><td>14.24</td><td>26.25</td></tr><tr><td>GC-RANSAC</td><td>65.32</td><td>92.56</td><td>45.11</td><td>67.66</td></tr><tr><td>PSC-RANSAC</td><td>72.17</td><td>98.32</td><td>48.22</td><td>72.90</td></tr><tr><td>DLOSAC (Ours)</td><td>42.65</td><td>58.90</td><td>22.73</td><td>41.43</td></tr></table>

TABLE III: Ablation Study. Mean errors and standard deviations across all datasets for three geometric estimation tasks. For relative-pose estimation, rotation and translation errors are reported separately. The best results are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">Fund. Matrix</td><td colspan="2">Rel. Pose</td><td colspan="2">Homo.</td></tr><tr><td>(px)  $\epsilon _ { \mathrm { a v g } }$ </td><td> $\epsilon _ { \mathrm { s t d } }$ </td><td>(px) ∈R (°)</td><td> $\epsilon _ { \mathrm { t } } ~ ( ^ { \circ } )$ </td><td> $\epsilon _ { \mathrm { a v g } }$  (px)</td><td> $\epsilon _ { \mathrm { s t d } } ~ \mathrm { ( p x ) }$ </td></tr><tr><td>w/o  $( \mathcal { L } _ { \mathrm { e d g e } } + \mathcal { L } _ { \mathrm { s p a t i a l } } )$ </td><td>0.573</td><td>0.580</td><td>4.133</td><td>13.061</td><td>1.447</td><td>1.411</td></tr><tr><td>w/o  $( \mathcal { L } _ { \mathrm { e d g e } } )$ </td><td>0.295</td><td>0.371</td><td>2.755</td><td>9.122</td><td>0.935</td><td>1.173</td></tr><tr><td>w/o  $( \mathcal { L } _ { \mathrm { s p a t i a l } } )$ </td><td>0.181</td><td>0.212</td><td>2.643</td><td>10.90</td><td>0.736</td><td>0.928</td></tr><tr><td>DLOSAC (Ours)</td><td>0.156</td><td>0.191</td><td></td><td>2.3207.501</td><td>0.588</td><td>0.902</td></tr></table>

## C. Ablation study

DLOSAC consists of two regularization terms: an edgebased geometric consistency term $( { \mathcal { L } } _ { \mathrm { e d g e } } )$ and a spatial distribution term $( \mathcal { L } _ { \mathrm { s p a t i a l } } )$ . The ablation study aims to assess the contribution of each of these terms. Table III presents the results of the ablation experiments across three tasks (1st row). By progressively removing the regularization terms of DLOSAC, we evaluate the cross-dataset performance of each variant across the three tasks. The last row shows the performance of the full DLOSAC model, with the effectiveness of each term validated by separately removing the spatial distribution term (w/o $\mathcal { L } _ { \mathrm { s p a t i a l } } )$ and the edge-based geometric consistency term (w/o $\mathcal { L } _ { \mathrm { e d g e } } )$ . Finally, we also remove both regularization terms (w/o ${ \mathcal { L } } _ { \mathrm { e d g e } } + { \mathcal { L } } _ { \mathrm { s p a t i a l } } )$ to demonstrate the performance of the baseline method.

## D. Sensitivity Analyses

DLOSAC controls the contribution of the spatialdistribution loss via the hyperparameter $\lambda _ { \mathrm { s p a t i a l } }$ . Setting this weight too large can weaken geometric consistency, while setting it too small tends to produce clustered keypoints and local overfitting. To select an appropriate value, we performed a parameter sweep over [0, 1] and evaluated crossdataset average errors on three tasks. Fig. 5 summarizes the results: the left axis reports the average pixel error $\varepsilon _ { \mathrm { a v g } }$ for fundamental matrix and homography estimation, while the right axis reports the average rotation and translation errors for relative-pose estimation. The curves reach a lowerror region around 0.6–0.8; therefore, we set $\lambda _ { \mathrm { s p a t i a l } } ~ =$ 0.6 for all experiments. Moreover, the error curves vary smoothly within this range, indicating low sensitivity to this hyperparameter.

![](images/eebcfcfeb27be73bde450ba4f364d59a024bbc91873bd739f866088e24ff336b.jpg)  
(a)

![](images/6535d5a53480a30d00000596a4680ba81d41b36b2c62d3c86cb2a2e442ee8ef6.jpg)  
(b)  
Fig. 5: Parameter analysis: (a) Weighting coefficient for the spatial distribution loss $\lambda _ { \mathrm { s p a t i a l } } ;$ (b) Regularization weight σ.

The parameter σ controls the sharpness of the exponential weighting function. Smaller σ amplifies differences in the weights and penalizes high-loss points more strongly, whereas larger σ smooths the weights and weakens the regularization effect. To identify an appropriate range, Fig. 5(b) reports cross-dataset average errors for three geometric estimation tasks under varying $\sigma .$ . The left axis shows the average pixel error for fundamental-matrix and homography estimation, while the right axis shows the average rotation and translation errors for relative-pose estimation. The curves remain smooth and reach a stable low-error region around $\sigma = 0 . 8$ ; therefore, we adopt $\sigma = 0 . 8$ as the default value, balancing stability and performance across all three tasks.

## VI. CONCLUSION

In this paper, we introduced DLOSAC, a novel local optimization framework that integrates both geometric consistency and spatial distribution information to improve geometric transformation estimation. Unlike existing methods that mainly focus on point-level geometric constraints, DLOSAC takes advantage of the spatial distribution of inliers by incorporating Delaunay triangulation, edge-level consistency metrics, and distribution descriptors. These innovations lead to more robust inlier selection and model refinement. Extensive experiments across six public datasets and three geometric estimation tasks demonstrate that our method consistently achieves superior performance compared to existing stateof-the-art methods. Ablation studies and sensitivity analysis further confirm the effectiveness and robustness of the joint geometric consistency and spatial distribution optimization strategy.

Although DLOSAC outperforms baseline methods across six public benchmarks, its gains are relatively modest in wide-baseline settings (e.g., CPC and T&T). The primary cause of this limitation is that the current geometric regularizers rely heavily on the assumption of approximate Euclidean invariance. When significant viewpoint or scale changes occur, especially with strong parallax, pronounced depth variation, and non-coplanar structures, this assumption breaks down. As a result, true-inlier residuals are systematically inflated, reducing inlier–outlier separability and weakening mismatch discrimination. Notably, this issue is not unique to our approach; algorithms incorporating spatial-level constraints, such as GC-RANSAC [14] and PSC-RANSAC [15], are similarly affected.

In future work, we will (a) develop projective- and affineinvariant regularizers with epipolar-based reweighting to maintain stability under large viewpoint changes and strong parallax; (b) employ multi-scale triangulation and distributional matching using a coarse-to-fine schedule with adaptive weighting of node- and distribution-level terms to mitigate bias arising from unstable local statistics; and (c) integrate DLOSAC’s local optimization into a complete, state-of-theart RANSAC framework (e.g., USACv20 [36]).

## REFERENCES

[1] S. Yan, Y. Liu, L. Wang, Z. Shen, Z. Peng, H. Liu, M. Zhang, G. Zhang, and X. Zhou, “Long-term visual localization with mobile sensors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 245–17 255.

[2] J. Miao, K. Jiang, T. Wen, Y. Wang, P. Jia, B. Wijaya, X. Zhao, Q. Cheng, Z. Xiao, J. Huang et al., “A survey on monocular relocalization: From the perspective of scene map representation,” IEEE Transactions on Intelligent Vehicles, 2024.

[3] H.-W. Chae, J.-H. Choi, and J.-B. Song, “Robust and autonomous stereo visual-inertial navigation for non-holonomic mobile robots,” IEEE Transactions on Vehicular Technology, vol. 69, no. 9, pp. 9613– 9623, 2020.

[4] B. Zhao, Y. Zhang, J. Huang, X. Zhang, Z. Long, and Y. Li, “L-VIWO: Visual-inertial-wheel odometry based on lane lines,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 18 079–18 085.

[5] A. R. Sahili, S. Hassan, S. M. Sakhrieh, J. Mounsef, N. Maalouf, B. Arain, and T. Taha, “A survey of visual SLAM methods,” IEEE Access, vol. 11, pp. 139 643–139 677, 2023.

[6] R. Mur-Artal and J. D. Tardos, “Orb-slam2: An open-source slam´ system for monocular, stereo, and rgb-d cameras,” IEEE Transactions on Robotics, vol. 33, no. 5, pp. 1255–1262, 2017.

[7] D. Barath, J. Noskova, M. Ivashechkin, and J. Matas, “MAGSAC++, a fast, reliable and accurate robust estimator,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 1304–1312.

[8] D. Barath, J. Matas, and J. Noskova, “MAGSAC: marginalizing sample consensus,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 10 197–10 205.

[9] D. Barath, M. Ivashechkin, and J. Matas, “Progressive NAPSAC: sampling from gradually growing neighborhoods,” arXiv preprint arXiv:1906.02295, 2019.

[10] J. Matas, O. Chum, M. Urban, and T. Pajdla, “Robust wide-baseline stereo from maximally stable extremal regions,” Image and vision computing, vol. 22, no. 10, pp. 761–767, 2004.

[11] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981.

[12] O. Chum, J. Matas, and J. Kittler, “Locally optimized RANSAC,” in Joint pattern recognition symposium. Springer, 2003, pp. 236–243.

[13] K. Lebeda, J. Matas, and O. Chum, “Fixing the locally optimized ransac–full experimental evaluation,” in British machine vision conference, vol. 2. Citeseer Princeton, NJ, USA, 2012.

[14] D. Barath and J. Matas, “Graph-cut RANSAC: Local optimization on spatially coherent structures,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 9, pp. 4961–4974, 2021.

[15] S. Ma, P. Guo, H. You, P. He, G. Li, and H. Li, “An image matching optimization algorithm based on pixel shift clustering RANSAC,” Information Sciences, vol. 562, pp. 452–474, 2021.

[16] D. Barath and J. Matas, “Graph-cut ransac,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6733–6741.

[17] J. Bian, W.-Y. Lin, Y. Matsushita, S.-K. Yeung, T.-D. Nguyen, and M.- M. Cheng, “Gms: Grid-based motion statistics for fast, ultra-robust feature correspondence,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 4181–4190.

[18] Z. Zhang, “Parameter estimation techniques: A tutorial with application to conic fitting,” Image and vision Computing, vol. 15, no. 1, pp. 59–76, 1997.

[19] O. Bailo, F. Rameau, K. Joo, J. Park, O. Bogdan, and I. S. Kweon, “Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution,” Pattern Recognition Letters, vol. 106, pp. 53–60, 2018.

[20] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International journal of computer vision, vol. 60, no. 2, pp. 91–110, 2004.

[21] H. Bay, T. Tuytelaars, and L. Van Gool, “Surf: Speeded up robust features,” in European conference on computer vision. Springer, 2006, pp. 404–417.

[22] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efficient alternative to SIFT or SURF,” in 2011 International conference on computer vision. IEEE, 2011, pp. 2564–2571.

[23] H. C. Longuet-Higgins, “A computer algorithm for reconstructing a scene from two projections,” Nature, vol. 293, no. 5828, pp. 133–135, 1981.

[24] R. I. Hartley, “In defense of the eight-point algorithm,” IEEE Transactions on pattern analysis and machine intelligence, vol. 19, no. 6, pp. 580–593, 1997.

[25] D. Nister, “An efficient solution to the five-point relative pose prob-´ lem,” IEEE transactions on pattern analysis and machine intelligence, vol. 26, no. 6, pp. 756–770, 2004.

[26] O. Chum and J. Matas, “Matching with PROSAC-progressive sample consensus,” in 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), vol. 1. IEEE, 2005, pp. 220–226.

[27] P. H. Torr and A. Zisserman, “MLESAC: A new robust estimator with application to estimating image geometry,” Computer vision and image understanding, vol. 78, no. 1, pp. 138–156, 2000.

[28] O. Chum and J. Matas, “Optimal randomized RANSAC,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 8, pp. 1472–1482, 2008.

[29] F. P. Preparata and M. I. Shamos, Computational geometry: an introduction. Springer Science & Business Media, 2012.

[30] Y. S. Elshakhs, K. M. Deliparaschos, T. Charalambous, G. Oliva, and A. Zolotas, “A comprehensive survey on Delaunay triangulation: applications, algorithms, and implementations over CPUs, GPUs, and FPGAs,” IEEE Access, vol. 12, pp. 12 562–12 585, 2024.

[31] R. I. Hartley and P. Sturm, “Triangulation,” Computer vision and image understanding, vol. 68, no. 2, pp. 146–157, 1997.

[32] Sturm, Jurgen and Engelhard, Nikolas and Endres, Felix and Burgard,¨ Wolfram and Cremers, Daniel, “A benchmark for the evaluation of rgb-d slam systems,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 573–580.

[33] Geiger, Andreas and Lenz, Philip and Urtasun, Raquel, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in 2012 IEEE Conference on Computer Vision and Pattern Recognition, 2012, pp. 3354–3361.

[34] Knapitsch, Arno and Park, Jaesik and Zhou, Qian-Yi and Koltun, Vladlen, “Tanks and temples: benchmarking large-scale scene reconstruction,” ACM Trans. Graph., vol. 36, no. 4, Jul. 2017.

[35] K. Wilson and N. Snavely, “Robust global translations with 1dsfm,” in Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part III 13. Springer, 2014, pp. 61–75.

[36] M. Ivashechkin, D. Barath, and J. Matas, “USACv20: robust essential, fundamental and homography matrix estimation,” CoRR, vol. abs/2104.05044, 2021.