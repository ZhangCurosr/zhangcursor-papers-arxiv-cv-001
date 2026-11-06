# GEKey: Learning Geodesic Eccentric 3D Keypoints via Self-Supervision

Pierre Onghena<sup>∗</sup> Santiago Velasco-Forero Beatriz Marcotegui

Mines Paris – PSL University Centre for Statistics and Images (STIM)

{pierre.onghena, santiago.velasco, beatriz.marcotegui}@minesparis.psl.eu

![](images/99cf0dce83d1836c417fc6b5954467560e9d0a3c460fb6dc6fcaf78637c14626.jpg)  
Figure 1. Overview of the geometric analysis in GEKey, illustrated with the Stanford dragon [6]. In this module, the geodesic eccentricity function reveals a set of salient, object-specific points from the 3D point cloud to guide our self-supervised method.

## Abstract

Point clouds represent 3D object geometry as sets of spatial coordinates. A fundamental task in their analysis is keypoint detection, i.e., identifying a sparse subset of points that captures the most salient features. While supervised methods rely on manual annotations, unsupervised approaches leverage intrinsic geometric properties to localize keypoints without human input. We introduce GEKey, a self-supervised framework that uses the geodesic eccentricity transform, an SE(3)-invariant geometric measure, to identify instance-specific salient points. These points serve as pseudo-labels to train a model that aligns its predictions with the object’s intrinsic structure. GEKey is simple, efficient, and robust, preserving intraclass variation by balancing instance-specific and categorylevel structure. Extensive experiments on ShapeNetCoreV2, KeypointNet, CAPE, ITOP, and DeformingThings4D show state-of-the-art performance in both keypoint detection and alignment, with strong resilience to noise, downsampling, deformations, and real-world conditions. The code and pretrained models are available at https://github.com/Pierreoo/GEKey.

## 1. Introduction

Unsupervised learning for point clouds has proven particularly difficult due to the complexity of geometric processing [33]. Under limited data conditions, self-supervised learning remains fundamental to advancing 3D understanding and representation learning [24, 35, 42]. A common approach to geometric representation is to derive a set of distinctive points. This keypoint set offers a compact, interpretable summary of an object’s geometry, capturing the salient structure that can be shared consistently across instances of a category.

Recent methods for unsupervised keypoint detection seek to reconstruct the complete point cloud from a small set of predicted keypoints [11, 28, 32, 41, 43, 50]. Whenever full geometry is approximated from predicted points alone, a model is forced to encode fine-grained surface details into a sparse set of points. The limited capacity often restricts the model to a coarse level, leading to similar or even identical keypoint placements across different shapes. This inability to account for subtle variations forms a barrier to practical applications and scenarios with intra-class variation. Predicted points that capture global geometry, however, remain essential to generalize the main features of an object class. Therefore, it is crucial to strike a balance between keypoints that highlight instance-specific attributes and those that reflect the category-level structure.

In this work, we present a formal definition based on geodesic eccentricity to identify local maxima as points that characterize the object. One of the workflows proposed by GEKey is shown in Fig. 1, illustrating how GEKey-PL first performs a geometric analysis to extract pseudo-labels. This initial module is lightweight, shape-agnostic, and designed to operate on 3D point clouds to retrieve the local maxima set of the geodesic eccentricity function. Then, as part of the second module, the predicted coordinates are optimized toward these pseudo-labels within a self-supervised framework. The central concepts that define GEKey are simple, yet effective and robust in practice to provide meaningful keypoint detection.

The primary contributions of this paper are summarized as follows:

• We generate pseudo-labels for self-supervised keypoint localization by detecting local maxima of the geodesic eccentricity function on the point cloud, leveraging its SE(3)-invariance and robustness to perturbations.

• Our method combines instance-specific and categoryspecific keypoints, enabling a balance between finegrained and generalized feature detection.

• We evaluate our approach on ShapeNetCoreV2 [3] and KeypointNet [40] and demonstrate state-of-the-art performance, surpassing existing methods in keypoint detection and alignment.

• We demonstrate robustness under perturbations, deformations, and real-world conditions, with experiments on CAPE [22, 26], ITOP [10], and DeformingThings4D [19].

## 2. Related Work

Keypoint detection methods can be broadly organized by their source of supervision. We review each group in turn, with an emphasis on how geometric structure is exploited.

Supervised Detection. Keypoint datasets originate from human annotations that require manual verification [1]. This includes KeypointNet [40], a large-scale semantic dataset intended to benchmark keypoint localization across a variety of state-of-the-art methods. Common supervised approaches formulate localization as a classification problem to output probability maps, where confidence thresholding leads to accurate but clustered keypoint predictions [4, 39, 40]. To avoid the need for post-processing with non-maximum suppression, KeypointDETR [14] generates Gaussian multi-heatmaps, where each heatmap’s intensity represents the proximity to the corresponding keypoint. Sampling-argmax [18] addresses the limitations of detection-based localization methods that rely on the argmax operation, proposing a Gumbel-softmax reparameterization to sample from the probability map for end-to-end differentiable training.

Hand-crafted Detection. Before the widespread use of deep learning, keypoint detectors relied on hand-crafted geometric features to identify salient points. Harris3D [29] and MeshDoG [44] are mesh-based detectors that respectively adapt the Harris corner response and the Differenceof-Gaussians operator to triangulated surfaces. Both require explicit mesh connectivity and yield dense interest points that lack semantic consistency. HKS [31] similarly operates on meshes, using extrema of the heat kernel signature to detect interest points, but captures only intrinsic geometry with no semantic discriminability. In contrast, ISS [48] works directly on raw point clouds, selecting salient keypoints by thresholding the eigenvalues of a local weighted scatter matrix. Our proposed geometric approach is simple, lightweight, and also operates directly on point clouds, with semantic consistency achieved through self-supervision.

Unsupervised Detection. In recent years, researchers have sought to improve keypoint detection to close the gap between automatically generated predictions and human annotations. One line of work focuses on detecting geometrically stable keypoints that are robust to transformations, for point cloud registration and matching. USIP [17] is among the first learning-based methods to detect such keypoints in point clouds under arbitrary transformations. Following this, UKPGAN [41] adopts an autoencoder with a reconstruction-based decoder, using GAN-based sparsity control and salient information distillation to produce keypoints that are robust to both rigid and non-rigid transformations. Similarly, SNAKE [47] jointly reconstructs the object and represents keypoints as a neural field. These methods produce unordered keypoint sets, with no index binding a keypoint to a fixed semantic slot across instances.

A distinct line of work discovers a sparse set of semantically consistent keypoints across category instances, for shape understanding and correspondence. ULCS [8] uses a prior based on symmetric linear basis shapes to learn such category-specific keypoints from misaligned point clouds. Skeleton Merger [28] implements an autoencoder whose decoder uses the predicted keypoints to form a skeletonbased reconstruction of the input point cloud. KeypointDeformer [13] aligns a source to a target object of the same category to discover category-specific keypoints and drive shape deformation in a controllable manner. SC3K [49] processes two rotated versions of an object and applies an individual loss per view for desired geometric properties, while a mutual dependency loss aligns the two predicted keypoint sets for semantic coherence. More recently, this line of work has been extended to deformable objects. Self-Geo [50] enforces keypoints to move with the deforming object by minimizing a geodesic loss that preserves constant pairwise distances between them. Key-Grid [11] uses an autoencoder framework, introducing a grid heatmap in the decoder that encodes distances from points uniformly sampled in a normalized cubic space to the skeleton formed by keypoint pairs. Despite these advancements, existing approaches rely primarily on geometric loss functions that drive keypoints toward category-level structure. In contrast, GEKey includes instance-specific cues to complement the category-level keypoints.

![](images/80fe5e3fa47777c842af9563b3a8679d04d176bf771d72b26f40adc466ac51b6.jpg)  
Figure 2. The geometric analysis of GEKey-PL, outlining the process from geodesic eccentricity to the derivation of pseudo-labels.

## 3. Method

We first present a geometric analysis that derives salient points from the geodesic eccentricity of a point cloud, and then describe how these points serve as pseudo-labels to supervise our learning framework.

## 3.1. Geometric Analysis

k-NN Graph. A 3D point cloud ${ \mathcal { X } } = \{ \mathbf { x } _ { 1 } , \dotsc , \mathbf { x } _ { n } \} \subseteq$ $\mathbb { R } ^ { 3 }$ , with matrix form $\boldsymbol { X } ~ \in ~ \mathbb { R } ^ { n \times 3 }$ , can be represented as a directed graph $\mathcal { G } _ { 0 } ~ = ~ ( \nu , \mathcal { E } _ { 0 } )$ , where the vertex set $\mathcal { V } = \{ 1 , \ldots , n \}$ indexes the points in $x ,$ and ${ \mathcal { E } } _ { 0 }$ denotes the set of directed edges. We identify each vertex u with its point $\mathbf { x } _ { u } .$ To construct this graph representation, we consider the k-Nearest Neighbor (k-NN) approach that connects each vertex $i \in \mathcal V$ to its k nearest neighbors $\mathcal { N } _ { k } ( i )$ in Euclidean space. This results in a directed graph that is generally asymmetric, which can be made undirected by symmetrizing the adjacency matrix $A \in \{ 0 , 1 \} ^ { n \times n }$ of $\mathcal { G } _ { 0 }$ via the element-wise maximum with its transpose:

$$
A ^ { \mathrm { s y m } } = \operatorname* { m a x } ( A , A ^ { \top } ) .\tag{1}
$$

The symmetric relation induces the edge set $\mathcal { E } ^ { \mathrm { s y m } } = \{ ( i , j )$ $A _ { i j } ^ { \mathrm { s y m } } = 1 \}$ , so that an edge $( i , j )$ is included in the graph if either $j \in \mathcal { N } _ { k } ( i ) \ \mathrm { o r } \ i \in \mathcal { N } _ { k } ( j )$ , thereby making the connectivity bidirectional. In addition, to achieve full connectivity, any disconnected components $\{ { \mathcal { C } } _ { 1 } , \ldots , { \mathcal { C } } _ { m } \}$ are linked. For two components $\scriptstyle { { \mathcal { C } } _ { a } }$ and ${ \mathcal { C } } _ { b } ,$ , we select the closest pair of vertices:

$$
( p ^ { * } , q ^ { * } ) \in \underset { \underset { q \in \mathcal { C } _ { b } } { p \in \mathcal { C } _ { a } } } { \arg \operatorname* { m i n } } \ : \| \mathbf { x } _ { p } - \mathbf { x } _ { q } \| _ { 2 } ,\tag{2}
$$

and take $\| \mathbf { x } _ { p ^ { \ast } } - \mathbf { x } _ { q ^ { \ast } } \| _ { 2 }$ as the distance between the two components. The pair of components with the smallest such distance is then merged by adding the edges $( p ^ { * } , q ^ { * } )$ and $( \boldsymbol { q } ^ { * } , \boldsymbol { p } ^ { * } )$ to the current edge set. Repeating this until one component remains adds $m - 1$ undirected connections and yields the graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , whose edge set E extends ${ \mathcal { E } } ^ { \mathrm { s y m } }$ with those added for connectivity.

Geodesic Eccentricity. With connectivity established, geodesic distances between vertices in the point cloud can now be meaningfully approximated. Specifically, the Euclidean distances between connected vertices are assigned as edge weights to define the weighted graph $\mathcal { G } _ { \omega } =$ $( \nu , \mathcal { E } , \omega )$ , with $\omega _ { i j } = \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 }$ for each $( i , j ) \in \mathcal { E }$ . These weights determine the lengths of the shortest paths in the graph, enabling the approximation of geodesic distances between all vertex pairs. Formally, the geodesic distance between vertices u and v is defined by:

$$
d _ { \mathcal { G } _ { \omega } } ( u , v ) = \operatorname* { m i n } _ { P \in \mathcal { P } _ { u v } } \sum _ { ( i , j ) \in P } \omega _ { i j } ,\tag{3}
$$

where $d \boldsymbol { g } _ { \omega } \left( \boldsymbol { u } , \boldsymbol { v } \right)$ represents the minimum total Euclidean distance among all paths $P \in \mathcal { P } _ { u v }$ from vertex u to vertex v. Given these distances, we consider the geodesic eccentricity transform [12, 25] as the mapping $u \mapsto \operatorname { e c c } ( u )$ where:

$$
\mathcal { \mathrm { e c c } } ( u ) = \operatorname* { m a x } _ { v \in \mathcal { V } } d \mathcal { G } _ { \omega } ( u , v ) .\tag{4}
$$

This transform assigns to each vertex its geodesic eccentricity as the greatest geodesic distance to any other vertex in the graph. An example of a normalized eccentricity function is presented on the left in Fig. 2.

Local Maxima. Based on the geodesic eccentricity, the objective is to outline a technique that characterizes a set of keypoints. In particular, we observe that such keypoints often correspond to local maxima of the geodesic eccentricity function. More precisely, these are the vertices whose eccentricity values are greater than or equal to those of their immediate neighbors. To formalize this, we apply a local dilation operator δ to each vertex $u \in \mathcal V$ over its closed neighborhood $\mathcal { N } _ { \mathcal { G } } [ u ]$ . The dilated eccentricity is defined as:

![](images/c2d5e953abf120cfdfc42367686efeddc924143e93a1a73d51aa417a0f7dc0ac.jpg)  
Figure 3. Overview of self-supervised GEKey to generalize keypoint prediction. The blue workflow shows a backbone model predicting keypoints Y<sup>ˆ</sup> , which are compared, via the bipartite matching loss, to pseudo-labels ${ \mathcal { V } } ^ { * }$ generated by the green pipeline.

$$
\delta ( \operatorname { e c c } ( u ) ) = \operatorname* { m a x } _ { v \in \mathcal { N } _ { \mathcal { G } } [ u ] } \operatorname { e c c } ( v ) ,\tag{5}
$$

which propagates the maximum eccentricity value within each local neighborhood. We identify local maxima of the eccentricity function as vertices where the original and the dilated eccentricities are equal, that is:

$$
{ \mathcal { M } } = \{ u \in \mathcal { V } : \sec ( u ) - \delta ( \sec ( u ) ) = 0 \} ,\tag{6}
$$

to indicate that each vertex $u \in \mathcal { M }$ attains a locally maximal geodesic eccentricity within its neighborhood [23]. An example is illustrated in the middle of Fig. 2 to provide an interpretation of the difference between the original and dilated eccentricity. On the right side of Fig. 2, the local maxima in set M, whose eccentricity values remain unchanged after the dilation operation, are highlighted. The proposed approach of GEKey-PL is shape-agnostic to detect salient points specific to each object. Note that our geometric analysis is also SE(3)-invariant, as geodesic distances and eccentricities are preserved under rigid transformations of the point cloud [15]. The local maxima set thus remains unchanged under rotations and translations, which encourages the keypoints to be highly repeatable and robust.

## 3.2. Self-Supervision

We propose a novel self-supervised framework, named GEKey, to enable models to generalize across objects and learn generic keypoints representative of each category. The framework trains a network that aligns its ordered sequence of predicted keypoints $\hat { Y } ~ = ~ \left( \hat { \bf y } _ { 1 } , \ldots , \hat { \bf y } _ { K } \right)$ with the unordered set of pseudo-labels $\mathcal { V } ^ { * } = \{ \mathbf { y } _ { 1 } ^ { * } , \ldots , \mathbf { y } _ { K } ^ { * } \}$ }, both of cardinality K.

Pseudo-Labels. The local maxima M of the geodesic eccentricity function are leveraged as pseudo-labels to guide the model in the localization of keypoints. As shown on the right side of Fig. 2, the number of local maxima $N _ { \mathrm { m a x } }$ varies with each object due to differences in geometric features. To standardize downstream processing, a consistent number of pseudo-labels K is derived from M using the farthest point sampling (FPS) algorithm. If $N _ { \mathrm { m a x } } < K$ FPS is initialized with M and iteratively samples additional points from the point cloud to reach K pseudo-labels. If $N _ { \mathrm { m a x } } \geq K$ , FPS selects a representative subset ${ \mathcal { V } } ^ { * } \subseteq { \mathcal { M } }$ of size $K , \mathbf { A }$ pseudo-label set ${ \mathcal { V } } ^ { * }$ of fixed cardinality is formed from the local maxima M to allow one-to-one matching with predictions under the loss function.

Keypoint Head. The underlying idea is intuitive in that the pointwise features of a network are decoded to produce a proposal of keypoint coordinates. Given input $\boldsymbol { X } \in \mathbb { R } ^ { n \times 3 }$ the backbone produces pointwise features $\begin{array} { r } { F ~ \in ~ \mathbb { R } ^ { n \times C } } \end{array}$ which are then projected by a keypoint attention head into attention scores:

$$
W = \mathrm { M L P } ( F ) , \quad W \in \mathbb { R } ^ { n \times K } ,\tag{7}
$$

where K denotes the desired number of keypoints. Each row $\mathbf { w } _ { i } \in \mathbb { R } ^ { K }$ is the attention score vector for point i, while each column corresponds to one fixed MLP output channel. A softmax is applied row-wise to $W ^ { \top } \in \mathbb { R } ^ { K \times \hat { n } }$ , normalizing the contribution of all n points for each keypoint. The predicted keypoints are then computed as a weighted aggregation of the input point coordinates:

$$
\hat { Y } = \operatorname { S o f t m a x } ( \boldsymbol { W } ^ { \top } ) \boldsymbol { X } , \quad \hat { Y } \in \mathbb { R } ^ { K \times 3 } ,\tag{8}
$$

where each of the $K$ rows of $\hat { Y }$ is a dedicated output slot for one keypoint. This approach, depicted by the blue workflow in Fig. 3, enables the model to localize keypoints through a soft attention mechanism, where the learned features $F$ determine how input points are aggregated into each slot.

Loss Function. Since the pseudo-labels ${ \mathcal { V } } ^ { * }$ are unordered, we supervise the predicted keypoints $\hat { Y }$ with a permutation-invariant bipartite matching loss based on the Hungarian algorithm [2, 16]. As presented in Fig. 3, the matching measures the optimal assignment cost between the ordered sequence of predicted keypoints and the set of pseudo-labels. Let $\pi \in$ Π be a bijection that pairs each predicted keypoint $\hat { \mathbf { y } } _ { i }$ with exactly one pseudo-label $\mathbf { y } _ { \pi ( i ) } ^ { * }$ . The matching loss is defined as the minimal average squared Euclidean distance over all permutations:

$$
\mathcal { L } _ { \mathrm { m a t c h } } ( \hat { Y } , \mathcal { V } ^ { * } ) = \operatorname* { m i n } _ { \pi \in \Pi } \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \left\| \hat { \mathbf { y } } _ { i } - { \mathbf { y } ^ { * } } _ { \pi ( i ) } \right\| _ { 2 } ^ { 2 } .\tag{9}
$$

The loss aligns each prediction with its matched pseudolabel and forms the complete training objective. Because the matching permutes the pseudo-labels while the slot indices remain fixed, each slot receives a coherent training signal across instances and can specialize to a single semantic region. This slot-to-region assignment emerges during training and may differ across runs, but stays consistent across instances within a model.

## 4. Experiments

## 4.1. Experimental Setup

Architecture. Our framework is backbone-agnostic, and we instantiate the backbone with Point Transformer V2 (PTv2) [37] to produce the pointwise representations consumed by the soft attention mechanism. PTv2 incorporates a more precise k-NN attention compared to PTv3’s [38] serialized mapping strategy. This improves spatial awareness in local neighborhoods and makes it effective for tasks requiring fine-grained precision such as keypoint detection.

Datasets. Unsupervised methods frequently employ ShapeNetCoreV2 [3], with 51300 single-object point clouds across 55 categories, and KeypointNet [40], with 8329 keypoint-annotated shapes across 16 categories. We train on ShapeNetCoreV2 and evaluate on KeypointNet, focusing on 13 categories that the two datasets share, with one model trained per category. Given that KeypointNet is a curated subset of ShapeNetCoreV2 that excludes objects deviating from the majority, training on ShapeNetCoreV2 demonstrates the model’s ability to generalize to more diverse or atypical instances. Pseudo-labels are derived for each training object to drive the learning process in our self-supervised framework. The KeypointNet dataset then serves as ground truth for a quantitative comparison with the predicted keypoints. The evaluation for each category includes the annotated training, validation, and test shapes of KeypointNet to align the benchmark with existing methods. This setting is used to assess pseudo-label quality, keypoint detection, semantic alignment, and robustness under perturbations.

To further evaluate robustness under non-rigid deformations and real-world conditions, we incorporate three additional datasets, following the data splitting protocol described by SelfGeo [50]. The Clothed Auto Person Encoding (CAPE) [22, 26] dataset consists of clothed human bodies registered to the SMPL mesh topology [20], captured under diverse poses and clothing types to provide a direct test of keypoint stability under combined body motion and clothing-induced surface deformation. We consider 51 temporal sequences, with 40 used for training (12432 frames) and 11 for testing (3311 frames). The Invariant-Top View Dataset (ITOP) [10] contains human subjects recorded from multiple camera viewpoints. We use the side-view set of ITOP and select four scenes (4626 frames), where human body points are segmented from the background but are incomplete due to self-occlusion and sensor noise. The DeformingThings4D [19] dataset provides animated sequences of deformable animals, from which we select nine categories for a total of 714 temporal sequences, with 555 for training (23494 frames) and 159 for testing (7887 frames). Together, these datasets enable the evaluation of keypoint stability across diverse deformation types, object categories, and real-world capture conditions.

Metrics. We follow standard metric evaluation practices used in unsupervised methods to ensure a fair comparison. We first evaluate pseudo-label quality using inclusivity and coverage, which assess how well keypoints are placed on and across the shape. The inclusivity quantifies the proportion of keypoints that are within a distance threshold of 0.10 from the nearest point in the normalized point cloud. The coverage measures the ratio of the keypoint bounding box diagonal to that of the point cloud, reflecting how well the keypoints span the full spatial extent of the shape. When applied to dynamic sequences, inclusivity and coverage are again assessed, with inclusivity computed using a stricter distance threshold of 0.05. In addition, temporal consistency $( T _ { c o n } )$ is evaluated to assess whether each predicted keypoint maintains a consistent semantic identity across consecutive frames.

Table 1. Keypoint inclusivity on KeypointNet, comparing unsupervised learning-based methods.
<table><tr><td>Method</td><td>Airplane</td><td>Bed</td><td>Bottle</td><td>Cap</td><td>Car</td><td>Chair</td><td>Guitar</td><td>Helmet</td><td>Knife</td><td>Motorcycle</td><td>Mug</td><td>Table</td><td>Vessel</td><td>Avg.</td></tr><tr><td>ULCS [8]</td><td>71.0</td><td>67.0</td><td>75.4</td><td>57.5</td><td>71.3</td><td>68.5</td><td>50.1</td><td>64.1</td><td>52.1</td><td>78.4</td><td>47.4</td><td>60.1</td><td>76.9</td><td>64.6</td></tr><tr><td>Skeleton Merger [28]</td><td>72.1</td><td>71.9</td><td>72.8</td><td>59.5</td><td>72.0</td><td>69.7</td><td>69.3</td><td>72.4</td><td>92.0</td><td>95.3</td><td>65.9</td><td>79.1</td><td>94.2</td><td>75.9</td></tr><tr><td>SC3K [49]</td><td>87.2</td><td>80.0</td><td>77.4</td><td>56.3</td><td>76.1</td><td>56.7</td><td>96.5</td><td>55.0</td><td>98.3</td><td>85.0</td><td>46.3</td><td>79.2</td><td>92.9</td><td>75.9</td></tr><tr><td>GEKey</td><td>98.5</td><td>87.0</td><td>61.3</td><td>65.0</td><td>92.7</td><td>90.5</td><td>99.7</td><td>51.3</td><td>99.8</td><td>98.2</td><td>90.9</td><td>91.2</td><td>96.3</td><td>86.3</td></tr></table>

Table 2. Keypoint coverage on KeypointNet, comparing hand-crafted (top) detectors and unsupervised learning-based (bottom) methods.
<table><tr><td>Method</td><td>Airplane</td><td>Bed</td><td>Bottle</td><td>Cap</td><td>Car</td><td>Chair</td><td>Guitar</td><td>Helmet</td><td>Knife</td><td>Motorcycle</td><td>Mug</td><td>Table</td><td>Vessel</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>FPS [27]</td><td>97.1</td><td>98.3</td><td>97.1</td><td>93.5</td><td>94.0</td><td>97.1</td><td>97.8</td><td>89.7</td><td>98.3</td><td>94.2</td><td>95.2</td><td>98.0</td><td>97.1</td><td>95.9</td></tr><tr><td>ISS [48]</td><td>77.0</td><td>95.6</td><td>96.0</td><td>91.2</td><td>92.4</td><td>93.7</td><td>87.6</td><td>88.0</td><td>85.2</td><td>89.1</td><td>93.4</td><td>95.4</td><td>89.9</td><td>90.3</td></tr><tr><td>GEKey-PL</td><td>99.2</td><td>98.7</td><td>96.7</td><td>93.6</td><td>95.5</td><td>97.6</td><td>98.7</td><td>88.6</td><td>99.6</td><td>94.8</td><td>96.0</td><td>98.5</td><td>98.2</td><td>96.6</td></tr><tr><td>ULCS [8]</td><td>88.6</td><td>94.2</td><td>80.9</td><td>60.8</td><td>83.7</td><td>83.9</td><td>79.8</td><td>79.9</td><td>76.8</td><td>78.9</td><td>89.6</td><td>83.0</td><td>78.8</td><td>81.5</td></tr><tr><td>Skeleton Merger [28]</td><td>92.6</td><td>84.3</td><td>91.4</td><td>85.0</td><td>90.7</td><td>85.9</td><td>85.7</td><td>82.1</td><td>77.4</td><td>86.1</td><td>83.2</td><td>91.3</td><td>85.3</td><td>86.2</td></tr><tr><td>SC3K [49]</td><td>96.3</td><td>98.2</td><td>98.0</td><td>94.6</td><td>89.8</td><td>95.3</td><td>97.6</td><td>90.5</td><td>98.8</td><td>94.3</td><td>95.2</td><td>97.4</td><td>97.2</td><td>95.6</td></tr><tr><td>GEKey</td><td>95.8</td><td>95.4</td><td>90.4</td><td>88.2</td><td>90.9</td><td>93.7</td><td>96.6</td><td>77.1</td><td>97.6</td><td>87.0</td><td>92.1</td><td>95.5</td><td>94.7</td><td>91.9</td></tr></table>

For keypoint detection, the mean Intersection over Union (mIoU) is reported as the primary spatial accuracy metric. A predicted keypoint is considered correct if its Euclidean distance to the ground truth is within a threshold of 0.10. Following common practice, the mIoU is computed globally by aggregating intersection and union across all samples within a category. While mIoU captures spatial accuracy, the Dual Alignment Score (DAS) evaluates whether the predicted keypoint set corresponds semantically across different shapes. The DAS score reflects whether the predicted keypoint at index i in the sequence $\hat { Y } = ( \hat { \mathbf { y } } _ { 1 } , \dotsc , \hat { \mathbf { y } } _ { K } )$ consistently corresponds to the same semantic region. Originally introduced by Skeleton Merger [28], $\mathrm { D A S } _ { \mathrm { S M } }$ assesses the semantic consistency of predicted keypoints through a bidirectional matching with the annotated labels from KeypointNet. In contrast, SC3K [49] implements a variant of the metric that omits alignment with the KeypointNet labels. Rather than using annotated labels, $\mathrm { D A S } _ { \mathrm { S C } 3 \mathrm { K } }$ designates the first test sample as a reference shape and evaluates whether predicted keypoints in other samples correspond to the same positions. Notably, the DAS score is most meaningful when accompanied by a strong mIoU score, as semantic alignment is only reliable when the predicted keypoints are also spatially accurate.

Implementation Details. We set the number of predicted keypoints to ten, in line with state-of-the-art keypoint benchmarks. Based on previous studies [36, 46], the kparameter is set to 20 in the k-NN graph. The geodesic map computation and pseudo-labeling are efficient in practice and scale well to large datasets. We build our proposed framework in Pointcept [5], a codebase that is designed for point cloud perception. All models are trained for 100 epochs with a batch size of 8, except for models trained on the dynamic sequence datasets, which use a batch size of 32 and predict twelve keypoints to match the setting of Self-Geo [50]. The network employs an AdamW optimizer [21] with a learning rate of $1 0 ^ { - 3 }$ and a one-cycle scheduler [30].

## 4.2. Pseudo-Label Quality

The initial step is to identify a set of pseudo-labels from the eccentricity function, as outlined by the green workflow of GEKey-PL in Fig. 3. This geometric operation is shape-agnostic, SE(3)-invariant, and anchored directly to the point cloud. Therefore, GEKey-PL achieves perfect inclusivity scores across all categories and thresholds by construction, as pseudo-labels lie exactly on the point cloud. Furthermore, self-supervising GEKey with these pseudolabels pulls the predicted keypoints toward the point cloud. In Tab. 1, with a threshold of 0.10, GEKey achieves the highest average inclusivity of 86.3% among all learningbased methods. Complementing this, Tab. 2 shows that GEKey-PL achieves the highest average coverage of 96.6%, confirming that our pseudo-labels are not only on the point cloud but also well distributed across it. Since the pseudolabels are well distributed, the matching loss inherently promotes spatial diversity among the predicted keypoints, so no explicit separation loss is required. We also report the detection accuracy of GEKey-PL in the upper half of Tab. 3, where it achieves the leading average mIoU of 46.7% among hand-crafted detectors, surpassing even the learning-based Skeleton Merger.

## 4.3. Keypoint Detection

The rationale for training a model to retrieve the pseudolabels is to induce generalization in the presence of intraclass variation. We notice that prior works tend to overfit to a fixed keypoint pattern, frequently overlooking instancespecific keypoints despite their annotation in the ground truth. As shown in Fig. 4, training over a broad range of shapes leads GEKey to balance instance-specific with category-typical keypoints. For example, GEKey marks each engine in the airplane with a keypoint to illustrate the instance-specific structure. The learned components of GEKey generalize the pseudo-labels of GEKey-PL to improve mIoU in every category of Tab. 3, raising the average from 46.7% to 54.5%. In the lower half of the table, we compare GEKey against other learning-based methods. We observe that GEKey achieves the highest mIoU scores over most categories, with an average of 54.5%. Overall, GEKey advances the state of the art in unsupervised keypoint detection, improving the average mIoU by +5.5 points over the next-best method. The gains are most significant in categories of high intra-class variation, such as airplane (+11.6) and motorcycle (+18.0).

Table 3. Keypoint detection on KeypointNet using mIoU metric, comparing hand-crafted (top) detectors and unsupervised learning-based (bottom) methods.
<table><tr><td>Method</td><td>Airplane</td><td>Bed</td><td>Bottle</td><td>Cap</td><td>Car</td><td>Chair</td><td>Guitar</td><td>Helmet</td><td>Knife</td><td>Motorcycle</td><td>Mug</td><td>Table</td><td>Vessel</td><td>Avg.</td></tr><tr><td>FPS [27]</td><td>67.6</td><td>27.9</td><td>38.5</td><td>17.4</td><td>40.4</td><td>41.1</td><td>64.6</td><td>22.9</td><td>45.2</td><td>37.1</td><td>23.0</td><td>38.0</td><td>48.5</td><td>39.4</td></tr><tr><td>ISS [48]</td><td>64.0</td><td>29.4</td><td>38.8</td><td>14.3</td><td>39.3</td><td>38.8</td><td>63.7</td><td>20.8</td><td>38.0</td><td>41.2</td><td>22.6</td><td>32.4</td><td>49.7</td><td>37.9</td></tr><tr><td>GEKey-PL</td><td>80.5</td><td>45.2</td><td>39.6</td><td>27.4</td><td>31.8</td><td>57.1</td><td>67.1</td><td>23.7</td><td>50.0</td><td>46.8</td><td>26.2</td><td>62.6</td><td>49.6</td><td>46.7</td></tr><tr><td>KeypointDeformer [13]</td><td>53.3</td><td>40.7</td><td>34.0</td><td>9.3</td><td>53.8</td><td>62.0</td><td>52.6</td><td>13.5</td><td>38.1</td><td>45.7</td><td>12.6</td><td>62.0</td><td>51.2</td><td>40.7</td></tr><tr><td>Skeleton Merger [28]</td><td>73.5</td><td>59.9</td><td>36.5</td><td>15.5</td><td>53.3</td><td>60.0</td><td>49.7</td><td>16.7</td><td>45.0</td><td>39.7</td><td>23.6</td><td>60.9</td><td>58.9</td><td>45.6</td></tr><tr><td>SC3K [49]</td><td>74.9</td><td>19.9</td><td>38.0</td><td>11.3</td><td>33.2</td><td>39.8</td><td>68.0</td><td>17.6</td><td>47.9</td><td>39.7</td><td>18.4</td><td>46.4</td><td>47.5</td><td>38.7</td></tr><tr><td>Key-Grid [11]</td><td>74.4</td><td>51.2</td><td>38.7</td><td>17.6</td><td>56.1</td><td>65.4</td><td>56.2</td><td>39.2</td><td>46.8</td><td>48.2</td><td>24.3</td><td>66.5</td><td>52.4</td><td>49.0</td></tr><tr><td>GEKey</td><td>86.5</td><td>51.9</td><td>46.1</td><td>32.1</td><td>43.9</td><td>65.6</td><td>71.6</td><td>36.5</td><td>53.0</td><td>66.2</td><td>28.6</td><td>70.7</td><td>55.8</td><td>54.5</td></tr></table>

![](images/a77b681fa0d7f394fd8f099a06a4487b6640ae591943fda519fa46e8aca7fa8e.jpg)  
Figure 4. Keypoint detection by GEKey for an example in each category, showing accurate localization at salient regions.

## 4.4. Keypoint Alignment

This benchmark measures whether predicted keypoints are consistently ordered, i.e., whether a given index corresponds to the same semantic part across instances. While hand-crafted methods can select accurate keypoints, they are unaware of semantics and thus fail to align them. In particular, these detectors produce an unordered set of keypoints, so any correspondence must be imposed externally. This misalignment motivates the application of selfsupervision in GEKey to learn and establish a semantic keypoint ordering between instances. A qualitative visualization of our method is presented in Fig. 5 for four categories, with the keypoints marked by color to represent their index in the prediction. In addition to precise detection, the examples demonstrate how predictions across varying objects align with one another to match semantic parts. The figure also includes irregular or atypical airplanes to further illustrate that our method preserves semantic consistency under geometric variation.

Building on the quality (Tabs. 1 and 2) and detection (Tab. 3) results, we evaluate the same predicted keypoints in terms of alignment. As reported in Tabs. 3 and 4, GEKey achieves the highest average mIoU of 54.5% and DAS<sub>SC3K</sub> of 73.0% among the unsupervised methods, the strongest joint performance across the two metrics. This pairing matters, since DAS is only informative when accompanied by a high mIoU. The two metrics are nonetheless in tension. As discussed in the supplementary material, intra-class variation means that instances contain different subsets of semantic labels, so predictions that adapt to instance-specific detail are matched to ground-truth labels that differ across instances, which lowers the alignment score. The same trade-off appears over the course of training, where detection improves while alignment declines, suggesting that further detection gains come from features the alignment metric does not reward. Where prior work operates at a coarse category level, blind to intra-class variation, GEKey works at a finer one, adapting to each shape to establish correspondence among instances that share the same semantic subset.

Table 4. Keypoint alignment on KeypointNet using DAS<sub>SC3K</sub> metric, comparing unsupervised learning-based methods.
<table><tr><td>Method</td><td>Airplane</td><td>Bed</td><td>Bottle</td><td>Cap</td><td>Car</td><td>Chair</td><td>Guitar</td><td>Helmet</td><td>Knife</td><td>Motorcycle</td><td>Mug</td><td>Table</td><td>Vessel</td><td>Avg.</td></tr><tr><td>KeypointDeformer [13]</td><td>73.4</td><td>69.5</td><td>60.3</td><td>56.7</td><td>80.2</td><td>84.3</td><td>62.9</td><td>55.3</td><td>59.8</td><td>61.5</td><td>69.8</td><td>72.6</td><td>73.9</td><td>67.7</td></tr><tr><td>Skeleton Merger [28]</td><td>77.8</td><td>61.3</td><td>59.9</td><td>53.1</td><td>79.5</td><td>76.8</td><td>63.2</td><td>57.0</td><td>60.4</td><td>57.8</td><td>67.2</td><td>70.0</td><td>72.4</td><td>65.9</td></tr><tr><td>SC3K [49]</td><td>82.9</td><td>64.9</td><td>62.7</td><td>59.7</td><td>75.2</td><td>87.0</td><td>65.7</td><td>58.6</td><td>63.0</td><td>59.4</td><td>75.3</td><td>76.0</td><td>76.0</td><td>69.7</td></tr><tr><td>Key-Grid [11]</td><td>84.7</td><td>70.6</td><td>66.2</td><td>58.5</td><td>84.5</td><td>93.4</td><td>69.1</td><td>62.5</td><td>60.9</td><td>63.7</td><td>79.4</td><td>78.6</td><td>74.6</td><td>72.8</td></tr><tr><td>GEKey</td><td>87.6</td><td>72.8</td><td>59.6</td><td>63.5</td><td>90.3</td><td>97.1</td><td>43.4</td><td>72.9</td><td>58.0</td><td>83.8</td><td>73.6</td><td>91.8</td><td>54.1</td><td>73.0</td></tr></table>

![](images/1485bcce929a8e6b56923d91aad701358ed4f3586002d68b63f227c51c874100.jpg)  
Figure 5. Keypoint alignment by GEKey, with keypoints mapped to consistent regions, including highly varied geometries.

## 4.5. Keypoint Robustness

Gaussian Noise. In point clouds, noise refers to random deviations of individual point positions. A common variant is jitter that applies Gaussian displacements to each coordinate to simulate sensor imprecision. In this noisy setting, the geodesic eccentricity function is established from the jittered object to determine its set of local maxima. The top graph in Fig. 6 shows how detection accuracy varies for GEKey across different noise scales and thresholds. This is compared against the baseline score of 86.5% in Tab. 3, for the airplane category at a threshold of 0.10 without jitter. The impact of training with different noise scales is quantitatively evaluated on the unperturbed test shapes to preserve ground-truth integrity and keep the annotations intact. The curves consistently show stable detection over different noise scales, with a notable drop only at the highest scale. To also illustrate the effect of noise at inference time, keypoints are visualized on jittered test objects in Fig. 6 for two noise scales, demonstrating the robustness of the geodesic eccentricity transform under positional disturbances.

Point Downsampling. By default, the self-supervised framework is optimized with a local maxima set derived from a connected graph of 2048 surface points. Reducing point density through farthest point sampling lowers the spatial resolution. We evaluate the geodesic eccentricity function and its local maxima set at sampling ratios of 2, 4, 8, 16, and 32, which reduce the original 2048 points to 1024, 512, 256, 128, and 64 points, respectively. To accommodate the sparser point clouds, the number of neighbors k in the k-NN graph is kept at its default value of 20 for a ratio of 2 and lowered to 15, 10, 10, and 5 for ratios of 4, 8, 16, and 32. The bottom graph in Fig. 6 reports for the airplane category how positional accuracy is affected by different downsampling ratios and thresholds. We evaluate objects at full resolution to preserve ground-truth integrity and annotated keypoint positions. With these values of k, accuracy remains stable up to a ratio of 8, drops at a ratio of 16, and partially recovers at a ratio of 32. Overall, the plot indicates that our method is largely robust to downsampling and generalizes effectively to complete inputs. To also illustrate the effect of downsampling at inference time, predicted keypoints are visualized on downsampled test objects in Fig. 6 for two downsampling ratios to demonstrate the robustness under reduced point density.

![](images/b7e5c26a3d0b51a4ce1a47660aa416f1dd0b1c39d5865dcead1e8bed99b1b48f.jpg)  
Figure 6. Robustness of GEKey to noise (top) and downsampling (bottom) perturbations, visualized at noise scales of 0.10 and 0.20 and downsampling ratios of 16 and 32.

Table 5. Keypoint quality under non-rigid deformations on the CAPE, ITOP, and DeformingThings4D datasets.
<table><tr><td>Dataset</td><td>Method</td><td>Inclusivity↑</td><td>Coverage↑</td><td> $T _ { c o n } \uparrow$ </td></tr><tr><td rowspan="4">CAPE [22, 26]</td><td>ULCS [8]</td><td>47.8</td><td>67.6</td><td>88.1</td></tr><tr><td>SC3K [49]</td><td>79.3</td><td>86.5</td><td>70.7</td></tr><tr><td>SelfGeo [50]</td><td>85.9</td><td>91.9</td><td>90.4</td></tr><tr><td>GEKey</td><td>99.5</td><td>96.7</td><td>93.3</td></tr><tr><td rowspan="4">ITOP [10]</td><td>ULCS [8]</td><td>42.1</td><td>64.3</td><td>76.5</td></tr><tr><td>SC3K [49]</td><td>76.4</td><td>83.1</td><td>42.9</td></tr><tr><td>SelfGeo [50]</td><td>84.2</td><td>91.1</td><td>80.6</td></tr><tr><td>GEKey</td><td>99.5</td><td>91.3</td><td>75.7</td></tr><tr><td rowspan="4">DeformingThings4D [19]</td><td>ULCS [8]</td><td>48.1</td><td>82.9</td><td>41.8</td></tr><tr><td>SC3K [49]</td><td>81.7</td><td>84.9</td><td>77.4</td></tr><tr><td>SelfGeo [50]</td><td>85.2</td><td>88.1</td><td>80.5</td></tr><tr><td>GEKey</td><td>97.1</td><td>95.0</td><td>85.3</td></tr></table>

## 4.6. Non-Rigid Deformations

Beyond perturbations of the input, we evaluate GEKey on shapes that deform. Geodesic distances and eccentricities are invariant under rigid transformations and approximately preserved under near-isometric deformations, including the articulated pose changes captured by the SMPL body model. Because eccentricity is determined by the global structure of the shape, even local non-isometric deformations such as the clothing folds in CAPE have only a limited effect on the locations of the local maxima. We assess keypoint quality on the CAPE, ITOP, and DeformingThings4D datasets, comparing against unsupervised learning-based methods in Tab. 5. Unlike SelfGeo, we train a model on sequences using only our geometric pseudo-labels, without imposing a temporal smoothness loss between consecutive frames.

We first evaluate on the CAPE dataset, which presents non-rigid deformations of the human body under diverse poses and clothing. In Tab. 5, GEKey outperforms the baselines on CAPE for all three metrics, with a significant margin in inclusivity and coverage. This confirms that, even as the point cloud undergoes non-rigid deformations, GEKey is able to retain keypoints that lie close to the surface. The temporal consistency $( T _ { c o n } )$ of 93.3% demonstrates that, based on geometric cues alone, the keypoints remain semantically stable across frames. The top of Fig. 7 qualitatively illustrates this stability, showing consistent keypoint placement throughout a sequence of human poses. To assess transfer from clean registered meshes to raw depth data, the model trained on CAPE is applied directly to the real-world setting of ITOP without any additional training or fine-tuning. As the data in ITOP is captured from a side-view perspective, the resulting scans are often both incomplete due to self-occlusion and cluttered by background noise. Despite these challenges, Tab. 5 shows that GEKey maintains the highest inclusivity and coverage on ITOP. The predicted keypoints are visualized at the bottom of Fig. 7, illustrating GEKey’s generalization from the clean registered meshes of CAPE to the raw single-view depth data of ITOP.

![](images/17451652ecac61c3e9a7d4d46c15689d078b6099c7b0bf717279ed2424dea748.jpg)  
Figure 7. Without temporal supervision in training, GEKey detects keypoints that remain stable under non-rigid human deformations in CAPE (top), and transfers to the real-world ITOP dataset without fine-tuning (bottom), despite self-occlusion and sensor noise.

![](images/67612f8d248339215d76552a11884a9b052d65d0602a4aeeaf231c6e5fb4480b.jpg)  
Figure 8. Keypoints detected by GEKey on animal shapes from DeformingThings4D, obtained from geometry alone without temporal supervision. Placements are precise, stable across frames, and consistent across species.

Finally, we evaluate on the DeformingThings4D dataset, which contains frames of animal sequences, and train a single model on all nine categories without a temporal loss. Nonetheless, the visual results in Fig. 8 show that GEKey produces temporally and cross-species consistent keypoints. The quantitative results in Tab. 5 confirm this, with our method outperforming the baselines across all three metrics. We attribute the high inclusivity of 97.1%, strong coverage of 95.0%, and temporal consistency of 85.3% to the precise identification of all anatomical parts. Together, these results confirm that GEKey generalizes across animal species and their deformations using geometry alone, without the need for temporal coherence or interframe constraints.

## 5. Conclusion

We introduced GEKey, a novel self-supervised framework that combines eccentricity-based attributes with a learningdriven approach to advance 3D keypoint detection. In particular, we addressed the limitation of being restricted to category-specific keypoints by incorporating instancespecific interest points in the learning process. The bipartite matching loss aligns each prediction with its matched pseudo-label, yielding keypoints that are both spatially accurate and consistently ordered across instances. The geometry-aware supervision further induces robustness to noise, downsampling, non-rigid deformations, and realworld conditions. The experimental results demonstrate that GEKey achieves the best performance, surpassing prior approaches to set a new state of the art in unsupervised keypoint detection. As such, a promising direction would be to further incorporate object-specific geometry into our learning-based framework.

## Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocation 2026-AD011014750R2 made by GENCI, on the V100, A100, and H100 partitions of the Jean Zay supercomputer. This research was partially supported by a government grant administered by the French National Research Agency (ANR) under the France 2030 program, grant PR[AI]RIE-PSAI (ANR-23-IACL-0008).

## References

[1] Matteo Bastico, Pierre Onghena, David Ryckelynck, Beatriz Marcotegui, Santiago Velasco-Forero, Laurent Corte, Caro-´ line Robine-Decourcelle, and Etienne Decenciere. LmPT:\` Conditional point transformer for anatomical landmark detection on 3D point clouds. In 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI), pages 1–5, 2026. 2

[2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In Computer Vision – ECCV 2020, pages 213–229, Cham, 2020. Springer International Publishing. 5

[3] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. ShapeNet: An Information-Rich 3D Model Repository. Technical Report arXiv:1512.03012 [cs.GR], Stanford University — Princeton University — Toyota Technological Institute at Chicago, 2015. 2, 5

[4] Silin Cheng, Xiwu Chen, Xinwei He, Zhe Liu, and Xiang Bai. PRA-Net: Point relation-aware network for 3D point cloud analysis. IEEE Transactions on Image Processing, 30: 4436–4448, 2021. 2

[5] Pointcept Contributors. Pointcept: A codebase for point cloud perception research, 2023. 6

[6] Brian Curless and Marc Levoy. A volumetric method for building complex models from range images. In Proceed ings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques, pages 303–312, New York, NY, USA, 1996. Association for Computing Machinery. 1

[7] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Pro cessing Systems. Curran Associates, Inc., 2013. 13

[8] Clara Fernandez-Labrador, Ajad Chhatkuli, Danda Pani Paudel, Jose J. Guerrero, Cedric Demonceaux, and Luc Van´ Gool. Unsupervised learning of category-specific symmetric 3D keypoints from point sets. In Computer Vision – ECCV 2020, pages 546–563, Cham, 2020. Springer International Publishing. 2, 6, 9

[9] Jean Feydy, Thibault Sejourn ´ e, Franc¸ois-Xavier Vialard,´ Shun-ichi Amari, Alain Trouve, and Gabriel Peyr´ e. Interpo-´ lating between optimal transport and MMD using Sinkhorn divergences. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 2681–2690, 2019. 13

[10] Albert Haque, Boya Peng, Zelun Luo, Alexandre Alahi, Serena Yeung, and Li Fei-Fei. Towards viewpoint invariant 3D human pose estimation. In Computer Vision – ECCV 2016, pages 160–177, Cham, 2016. Springer International Publishing. 2, 5, 9

[11] Chengkai Hou, Zhengrong Xue, Bingyang Zhou, Jinghan Ke, Lin Shao, and Huazhe Xu. Key-Grid: Unsupervised 3D keypoints detection using grid heatmap features. In Advances in Neural Information Processing Systems, pages 49154–49179. Curran Associates, Inc., 2024. 1, 2, 7, 8

[12] Adrian Ion, Gabriel Peyre, Yll Haxhimusa, Samuel Peltier,´ Walter G. Kropatsch, and Laurent D. Cohen. Shape Match ing Using the Geodesic Eccentricity Transform - A Study. In Proceedings of the 31st annual workshop of the Austrian Association for Pattern Recognition (OAGM/AAPR), pages 97–104, Krumbach, Austria, 2007. 3

[13] Tomas Jakab, Richard Tucker, Ameesh Makadia, Jiajun Wu, Noah Snavely, and Angjoo Kanazawa. KeypointDeformer: Unsupervised 3D keypoint discovery for shape control. In Proceedings of the IEEE/CVF Conference on Computer Vi sion and Pattern Recognition (CVPR), pages 12783–12792, 2021. 2, 7, 8

[14] Hairong Jin, Yuefan Shen, Jianwen Lou, Kun Zhou, and Youyi Zheng. KeypointDETR: An end-to-end 3D keypoint detector. In Computer Vision – ECCV 2024, pages 374–390, Cham, 2025. Springer Nature Switzerland. 2

[15] Nathan Krislock and Henry Wolkowicz. Euclidean distance matrices and applications. In Handbook on semidef inite, conic and polynomial optimization, pages 879–914. Springer, 2012. 4

[16] Harold W. Kuhn. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1-2):83–97, 1955. 5, 13

[17] Jiaxin Li and Gim Hee Lee. USIP: Unsupervised stable interest point detection from 3D point clouds. In Proceedings of

the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. 2

[18] Jiefeng Li, Tong Chen, Ruiqi Shi, Yujing Lou, Yong-Lu Li, and Cewu Lu. Localization with Sampling-Argmax. In Advances in Neural Information Processing Systems, pages 27236–27248. Curran Associates, Inc., 2021. 2

[19] Yang Li, Hikari Takehara, Takafumi Taketomi, Bo Zheng, and Matthias Nießner. 4DComplete: Non-rigid motion estimation beyond the observable surface. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12706–12716, 2021. 2, 5, 9

[20] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multiperson linear model. ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, 2015. 5, 17

[21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6

[22] Qianli Ma, Jinlong Yang, Anurag Ranjan, Sergi Pujades, Gerard Pons-Moll, Siyu Tang, and Michael J. Black. Learning to dress 3D people in generative clothing. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2, 5, 9

[23] Pierre Onghena, Santiago Velasco-Forero, and Beatriz Marcotegui. MorphoSkel3D: Morphological skeletonization of 3D point clouds for informed sampling in object classification and retrieval. In 2025 International Conference on 3D Vision (3DV), pages 1350–1359, 2025. 4

[24] Yatian Pang, Wenxiao Wang, Francis E. H. Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning. In Computer Vision – ECCV 2022, pages 604–621, Cham, 2022. Springer Nature Switzerland. 1

[25] Gabriel Peyre and Laurent D. Cohen. Geodesic methods for´ shape and surface processing. Advances in computational vision and medical image processing: methods and applications, pages 29–56, 2009. 3

[26] Gerard Pons-Moll, Sergi Pujades, Sonny Hu, and Michael Black. ClothCap: Seamless 4D clothing capture and retargeting. ACM Transactions on Graphics (Proc. SIGGRAPH), 36(4), 2017. 2, 5, 9

[27] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. Point-Net++: Deep hierarchical feature learning on point sets in a metric space. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pages 5105–5114, Red Hook, NY, USA, 2017. Curran Associates Inc. 6, 7, 13, 14, 18

[28] Ruoxi Shi, Zhengrong Xue, Yang You, and Cewu Lu. Skeleton Merger: An unsupervised aligned keypoint detector. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 43–52, 2021. 1, 2, 6, 7, 8, 15

[29] Ivan Sipiran and Benjamin Bustos. Harris 3D: A robust extension of the Harris operator for interest point detection on 3D meshes. The Visual Computer, 27(11):963–976, 2011. 2

[30] Leslie N. Smith and Nicholay Topin. Super-convergence: Very fast training of neural networks using large learning

rates. In Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, page 1100612. In ternational Society for Optics and Photonics, SPIE, 2019. 6

[31] Jian Sun, Maks Ovsjanikov, and Leonidas Guibas. A concise and provably informative multi-scale signature based on heat diffusion. In Computer Graphics Forum, pages 1383–1392. Wiley Online Library, 2009. 2

[32] Junshu Tang, Zhijun Gong, Ran Yi, Yuan Xie, and Lizhuang Ma. LAKe-Net: Topology-aware point cloud completion by localizing aligned keypoints. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1726–1735, 2022. 1

[33] Federico Tombari, Samuele Salti, and Luigi Di Stefano. Performance evaluation of 3D keypoint detectors. International Journal ofComputer Vision, 102(1):198–220, 2013. 1

[34] Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019. 18

[35] Hanchen Wang, Qi Liu, Xiangyu Yue, Joan Lasenby, and Matt J. Kusner. Unsupervised point cloud pre-training via occlusion completion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9782–9792, 2021. 1

[36] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. Dynamic graph CNN for learning on point clouds. ACM Transactions on Graphics, 38(5), 2019. 6, 17

[37] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point Transformer V2: Grouped vector atten tion and partition-based pooling. In Advances in Neural Information Processing Systems, pages 33330–33342. Curran Associates, Inc., 2022. 5, 13, 14

[38] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point Transformer V3: Simpler faster stronger. In Pro ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4840–4851, 2024. 5

[39] Li Yi, Hao Su, Xingwen Guo, and Leonidas J. Guibas. Sync SpecCNN: Synchronized spectral CNN for 3D shape segmentation. In Proceedings ofthe IEEE Conference on Com puter Vision and Pattern Recognition (CVPR), 2017. 2

[40] Yang You, Yujing Lou, Chengkun Li, Zhoujun Cheng, Liangwei Li, Lizhuang Ma, Cewu Lu, and Weiming Wang. KeypointNet: A large-scale 3D keypoint dataset aggregated from numerous human annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2, 5, 16, 17

[41] Yang You, Wenhai Liu, Yanjie Ze, Yong-Lu Li, Weiming Wang, and Cewu Lu. UKPGAN: A general self-supervised keypoint detector. In Proceedings ofthe IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 17042–17051, 2022. 1, 2

[42] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-BERT: Pre-training 3D point

cloud transformers with masked point modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19313–19322, 2022. 1

[43] Haocheng Yuan, Chen Zhao, Shichao Fan, Jiaxi Jiang, and Jiaqi Yang. Unsupervised learning of 3D semantic keypoints with mutual reconstruction. In Computer Vision – ECCV 2022, pages 534–549, Cham, 2022. Springer Nature Switzerland. 1

[44] Andrei Zaharescu, Edmond Boyer, and Radu Horaud. Keypoints and local descriptors of scalar functions on 2D manifolds. International Journal of Computer Vision, 100(1): 78–98, 2012. 2

[45] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip H.S. Torr, and Vladlen Koltun. Point Transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 16259–16268, 2021. 13, 14

[46] Puning Zhao and Lifeng Lai. Analysis of KNN density estimation. IEEE Transactions on Information Theory, 68(12): 7971–7995, 2022. 6, 17

[47] Chengliang Zhong, Peixing You, Xiaoxue Chen, Hao Zhao, Fuchun Sun, Guyue Zhou, Xiaodong Mu, Chuang Gan, and Wenbing Huang. SNAKE: Shape-aware neural 3D keypoint field. In Advances in Neural Information Processing Systems, pages 7052–7064. Curran Associates, Inc., 2022. 2

[48] Yu Zhong. Intrinsic shape signatures: A shape descriptor for 3D object recognition. In 2009 IEEE 12th International Conference on Computer Vision Workshops, ICCV Workshops, pages 689–696, 2009. 2, 6, 7, 14, 18

[49] Mohammad Zohaib and Alessio Del Bue. SC3K: Selfsupervised and coherent 3D keypoints estimation from rotated, noisy, and decimated point cloud data. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 22509–22519, 2023. 2, 6, 7, 8, 9, 15

[50] Mohammad Zohaib, Luca Cosmo, and Alessio Del Bue. SelfGeo: Self-supervised and geodesic-consistent estimation of keypoints on deformable shapes. In Computer Vision – ECCV 2024, pages 71–88, Cham, 2025. Springer Nature Switzerland. 1, 2, 5, 6, 9

# GEKey: Learning Geodesic Eccentric 3D Keypoints via Self-Supervision

Supplementary Material

## Overview

In this supplementary material, we provide additional details and experimental results for the main paper, including:

• Comparison between Hungarian matching and Sinkhorn algorithm (Sec. A).

• Impact of the backbone architecture on keypoint prediction (Sec. B).

• Evaluation of keypoint detection across error thresholds below 0.10 (Sec. C).

• Correlation between the metrics of keypoint detection and alignment (Sec. D).

• Discussion of the Dual Alignment Score (DAS) metric in prior works (Sec. E).

• Runtime analysis of the components in our geometric workflow (Sec. F).

• Effect of Delaunay triangulation on geodesics and local maxima (Sec. G).

• Influence of the k-parameter in the graph on pseudo-label extraction (Sec. H).

• Stability of the extracted keypoints under near-isometric deformations (Sec. I).

• Generalization to uncontrolled real-world scans without fine-tuning (Sec. J).

## A. Matching Loss Functions

The bipartite matching loss of our method performs a hard selection by determining an optimal one-to-one assignment between predictions and targets using the Hungarian algorithm [16]. A relaxed approximation of this matching can be obtained via the Sinkhorn algorithm [7, 9], which yields a differentiable loss function suitable for optimization. In this relaxation, the hard permutation is replaced by a soft transport plan $P \in \mathbb { R } _ { + } ^ { K \times K }$ , where $P _ { i j }$ represents the probability mass transported from prediction i to target j. The Sinkhorn loss is defined as the optimal transport cost between two distributions, which is smoothed with an entropic regularization term controlled by the parameter ε, and can be computed as:

$$
\mathcal { L } _ { \mathrm { S i n k h o r n } } = \operatorname* { m i n } _ { P \in \mathcal { U } ( r , c ) } \sum _ { i = 1 } ^ { K } \sum _ { j = 1 } ^ { K } P _ { i j } \left\| \hat { \mathbf { y } } _ { i } - \mathbf { y } _ { j } ^ { * } \right\| _ { 2 } ^ { 2 } - \varepsilon H ( P ) ,\tag{10}
$$

where $\mathcal { U } ( r , c )$ is the set of transport plans with uniform marginals $\boldsymbol { r } = \boldsymbol { c } = \frac { 1 } { K } \mathbf { 1 } _ { K }$ . The term $H ( P )$ denotes the entropy of P, given by:

$$
H ( P ) = - \sum _ { i = 1 } ^ { K } \sum _ { j = 1 } ^ { K } P _ { i j } \log P _ { i j } .\tag{11}
$$

![](images/e8311db0a63d0ccd7e51022f310ba672306556ad5e4fdce0c754f13e06728680.jpg)  
Figure 9. Comparison of ${ \mathcal { L } } _ { \mathrm { m a t c h } }$ and $\mathcal { L } _ { \mathrm { { S i n k h o r n } } }$ for the airplane and chair categories.

When $\varepsilon = 0 ,$ , the Sinkhorn loss reduces to the exact optimal transport cost, i.e., the matching loss without any regularization. We observe that the Sinkhorn formulation provides a smooth objective that can facilitate early training stability. A comparison between the soft assignments produced by the Sinkhorn loss and the hard assignments induced by the bipartite matching loss is shown in Fig. 9. While the Sinkhorn loss offers a more stable optimization, the bipartite matching loss enables faster mIoU convergence during the early epochs. Moreover, a gap in keypoint alignment is observed, as GEKey models trained with the bipartite matching achieve higher DAS<sub>SC3K</sub> scores. Therefore, the bipartite matching loss computed via the Hungarian algorithm is preferred in the main paper.

## B. Backbone Architecture

Our framework only requires pointwise features $F \in \mathbb { R } ^ { n \times C }$ from the backbone, so it is agnostic to the architecture that produces them. To verify that the improvements stem from our pseudo-labels and not from a particular network, we replace the PTv2 [37] backbone with PTv1 [45] and Point-Net++ [27]. All variants share the same keypoint head, pseudo-labels, and training protocol, so that any differences are attributable to the backbone alone. As reported in Tabs. 6 and 7, the three backbones lie within 1.9 points in average mIoU and 2.6 points in average DAS<sub>SC3K</sub>, indicating that GEKey is largely independent of the feature extractor. PTv2 is best on both averages and is therefore adopted as the default, though GEKey transfers to other architectures without modification.

Table 6. Keypoint detection on KeypointNet using mIoU metric, comparing different backbone architectures with GEKey.
<table><tr><td>Method</td><td>Airplane</td><td>Bed</td><td>Bottle</td><td>Cap</td><td>Car</td><td>Chair</td><td>Guitar</td><td>Helmet</td><td>Knife</td><td>Motorcycle</td><td>Mug</td><td>Table</td><td>Vessel</td><td>Avg.</td></tr><tr><td>PointNet++ [27]</td><td>86.2</td><td>53.0</td><td>52.8</td><td>27.7</td><td>42.6</td><td>66.9</td><td>63.9</td><td>31.2</td><td>53.3</td><td>66.4</td><td>24.7</td><td>71.1</td><td>57.2</td><td>53.6</td></tr><tr><td>PTv1 [45]</td><td>84.4</td><td>48.3</td><td>46.8</td><td>29.6</td><td>46.1</td><td>65.7</td><td>69.0</td><td>31.1</td><td>52.0</td><td>60.6</td><td>25.6</td><td>69.4</td><td>55.5</td><td>52.6</td></tr><tr><td>PTv2 [37]</td><td>86.5</td><td>51.9</td><td>46.1</td><td>32.1</td><td>43.9</td><td>65.6</td><td>71.6</td><td>36.5</td><td>53.0</td><td>66.2</td><td>28.6</td><td>70.7</td><td>55.8</td><td>54.5</td></tr></table>

Table 7. Keypoint alignment on KeypointNet using DAS<sub>SC3K</sub> metric, comparing different backbone architectures with GEKey.
<table><tr><td>Method</td><td>Airplane</td><td>Bed</td><td>Bottle</td><td>Cap</td><td>Car</td><td>Chair</td><td>Guitar</td><td>Helmet</td><td>Knife</td><td>Motorcycle</td><td>Mug</td><td>Table</td><td>Vessel</td><td>Avg.</td></tr><tr><td>PointNet++ [27]</td><td>83.7</td><td>70.8</td><td>52.2</td><td>62.4</td><td>92.8</td><td>94.0</td><td>48.1</td><td>79.6</td><td>59.8</td><td>82.2</td><td>55.2</td><td>91.6</td><td>53.5</td><td>71.2</td></tr><tr><td>PTv1 [45]</td><td>81.0</td><td>69.9</td><td>56.1</td><td>60.8</td><td>90.3</td><td>96.9</td><td>37.0</td><td>76.3</td><td>60.7</td><td>74.4</td><td>63.9</td><td>91.6</td><td>56.0</td><td>70.4</td></tr><tr><td>PTv2 [37]</td><td>87.6</td><td>72.8</td><td>59.6</td><td>63.5</td><td>90.3</td><td>97.1</td><td>43.4</td><td>72.9</td><td>58.0</td><td>83.8</td><td>73.6</td><td>91.8</td><td>54.1</td><td>73.0</td></tr></table>

Table 8. Keypoint detection using mIoU metric at a distance threshold of 0.01.
<table><tr><td>Method</td><td>Airplane</td><td>Bed</td><td>Bottle</td><td>Cap</td><td>Car</td><td>Chair</td><td>Guitar</td><td>Helmet</td><td>Knife</td><td>Motorcycle</td><td>Mug</td><td>Table</td><td>Vessel</td><td>Avg.</td></tr><tr><td>FPS [27]</td><td>13.2</td><td>10.9</td><td>2.8</td><td>1.0</td><td>2.7</td><td>7.4</td><td>7.1</td><td>2.4</td><td>10.7</td><td>6.2</td><td>2.3</td><td>10.7</td><td>6.7</td><td>6.5</td></tr><tr><td>ISS [48]</td><td>1.7</td><td>1.3</td><td>1.2</td><td>1.2</td><td>1.5</td><td>1.7</td><td>1.2</td><td>1.4</td><td>3.0</td><td>1.4</td><td>1.1</td><td>1.8</td><td>1.1</td><td>1.5</td></tr><tr><td>GEKey-PL</td><td>20.0</td><td>19.3</td><td>2.5</td><td>2.5</td><td>1.4</td><td>13.6</td><td>10.8</td><td>1.8</td><td>14.9</td><td>7.8</td><td>2.9</td><td>21.0</td><td>6.3</td><td>9.6</td></tr><tr><td>GEKey</td><td>15.0</td><td>12.4</td><td>1.0</td><td>1.3</td><td>0.2</td><td>8.7</td><td>9.7</td><td>1.1</td><td>10.9</td><td>5.4</td><td>1.3</td><td>12.0</td><td>3.7</td><td>6.4</td></tr></table>

![](images/408ec9c468dbde96d2b600a453b9b69bd2037be21b5e11f1eacb47f79ae26ea5.jpg)  
Figure 10. Per-category mIoU (%) curves of GEKey over Euclidean distance thresholds.

## C. Error Thresholds

In the main paper, we report mIoU with a threshold of 0.10 to follow common practice in unsupervised keypoint evaluation. To complement the detection results at the 0.10 threshold, we provide per-category mIoU over smaller Euclidean distance thresholds in Fig. 10. At the more restrictive threshold of 0.01, we compare our method against two traditional geometric keypoint detectors in Tab. 8. We observe that GEKey-PL is on average more precise than both hand-crafted detectors at this stricter threshold. It is also more precise than GEKey, since the attention head aggregates coordinates into slightly off-surface predictions. Even so, GEKey stays close to FPS on average and well ahead of

ISS, so the offset introduced by the attention head is small relative to the detection gains at the 0.10 threshold.

## D. Metric Correlation

To further contextualize the DAS in relation to the mIoU, their scores are jointly reported over the course of training to assess whether improvements in semantic alignment reflect precise keypoint localization. In particular, high DAS values in the absence of strong mIoU indicate semantically coherent but inaccurate predictions, which in turn limit their practical utility. In addition to using a Euclidean error threshold of 0.10 to determine the overlap between prediction and ground truth, a stricter threshold of 0.01 is also taken into account to assess finer localization accuracy.

In the graphs of Fig. 11, three metrics are visualized for the validation sets of the airplane and chair categories. The bipartite matching loss rapidly pulls the predictions toward the pseudo-labels, which lie close to the ground-truth keypoints, as reflected in the early convergence of mIoU-0.10. The steady increase in mIoU-0.01 indicates that the model continues to refine fine-grained localization throughout training. In contrast, the $\mathrm { D A S } _ { \mathrm { S C } 3 \mathrm { K } }$ score decreases after an initial peak, showing a negative correlation with the mIoU metrics. This trend suggests that as the model shifts toward instance-specific keypoints for more precise detection, it simultaneously drifts away from the fixed patterns of category-level alignment. In the following section, we further discuss how this shift affects the interpretation and reliability of the DAS metric.

![](images/2cde4c58b23150493a32980f2133f9ebe85bacf07b1e1065fbb01123758a5ddc.jpg)  
Figure 11. Evolution of mIoU-0.10, mIoU-0.01, and DAS<sub>SC3K</sub> over training for the airplane and chair categories, illustrating the negative correlation between detection and alignment.

## E. Dual Alignment Score

The Dual Alignment Score (DAS) measures whether predicted keypoints are both spatially aligned and semantically consistent across categorical shapes. The DAS metric, as implemented by Skeleton Merger [28], applies bidirectional matching between the predicted keypoints and the annotated ground-truth labels of KeypointNet. This process evaluates whether each predicted keypoint, when matched to its closest ground-truth keypoint, consistently corresponds to the same semantic ID across instances in a category. For example, KeypointNet’s expert annotators identified twentyone distinct semantic label IDs in the airplane category. However, intra-class variability means that each airplane instance contains only a subset of the full semantic label set. Example subsets are illustrated in Fig. 12, where two airplanes and chairs are shown to visualize their ground-truth keypoints and associated, color-coded semantic IDs. The two cases highlight that different instances yield different subsets, including instance-specific IDs circled in black.

In Tab. 9, we report the results with the DAS metric proposed by Skeleton Merger [28] to compare our method GEKey against their baseline on ten keypoints. The DAS<sub>SM</sub> metric reaches a perfect value when each of the ten predicted keypoints consistently matches the same groundtruth semantic label. However, as seen in Fig. 12, not all objects share the same semantic ID set. Predictions therefore naturally align with different valid semantic subsets across instances. Specifically, when predictions vary to capture instance-specific details, assignments to unique semantic labels occur, reducing the $\mathrm { D A S } _ { \mathrm { S M } }$ score. This significant limitation of the alignment metric motivates unsupervised methods to produce a fixed keypoint pattern. Yet, instancespecific adaptation by GEKey improves keypoint detection accuracy, advancing the state of the art in unsupervised keypoint detection.

Table 9. Keypoint alignment on KeypointNet using DAS<sub>SM</sub> metric.
<table><tr><td rowspan="2"></td><td colspan="2">Airplane</td><td colspan="2">Chair</td></tr><tr><td>DAS</td><td>mIoU</td><td>DAS</td><td>mIoU</td></tr><tr><td>Skeleton Merger [28]</td><td>77.8</td><td>73.5</td><td>76.8</td><td>60.0</td></tr><tr><td>GEKey</td><td>73.3</td><td>86.5</td><td>74.9</td><td>65.6</td></tr></table>

![](images/74a6e538420bbb54269ed47b7656daa3a2b15b318b6be516056ab99687567d49.jpg)  
Figure 12. Ground-truth keypoints color-coded by semantic ID, with instance-specific circled to highlight intra-class variability: not all instances share the same semantic label subset, constraining the $\mathrm { D A S } _ { \mathrm { S M } }$ metric when predictions adapt to per-shape geometry.

SC3K [49] proposes a variant of the DAS metric. Rather than matching predictions to KeypointNet’s annotated labels, it designates the first test sample as a reference shape and evaluates whether predicted keypoints at a given index fall in the same position as on that reference. Although this removes the dependence on annotated semantic labels, the approach still favors fixed keypoint patterns, since instancespecific predictions tend to diverge in position from the reference shape. This explains the negative correlation observed between alignment and detection in Fig. 11. Overall, GEKey’s more precise detection of instance-specific keypoints logically trades off against the static alignment of fixed patterns.

Table 10. Processing times (in seconds) for the k-NN graph, geodesic eccentricity, and pseudo-label computation stages of GEKey-PL.
<table><tr><td rowspan="2"></td><td rowspan="2">#Obj</td><td rowspan="2">k-NN</td><td colspan="2"> $\mathrm { e c c } _ { 1 0 0 } ( u )$ </td><td colspan="2"> $\ker \mathrm { l } _ { 1 0 } ( u )$ </td><td> $\operatorname { e c c } ( u ) - \delta ( \operatorname { e c c } ( u ) )$ </td></tr><tr><td> $T _ { \mathrm { T o t } }$   $T _ { \mathrm { T o t } }$ </td><td>mIoU</td><td> $T _ { \mathrm { T o t } }$ </td><td>mIoU</td><td> $T _ { \mathrm { T o t } }$ </td></tr><tr><td>Airplane</td><td>1022</td><td>7.6</td><td>79.3</td><td>80.5</td><td>9.1</td><td>80.3</td><td>11.6</td></tr><tr><td>Bed</td><td>146</td><td>1.0</td><td>11.7</td><td>45.2</td><td>1.4</td><td>42.3</td><td>2.4</td></tr><tr><td>Bottle</td><td>380</td><td>2.5</td><td>30.2</td><td>39.6</td><td>3.5</td><td>39.1</td><td>14.6</td></tr><tr><td>Cap</td><td>38</td><td>0.3</td><td>3.0</td><td>27.4</td><td>0.4</td><td>24.9</td><td>0.9</td></tr><tr><td>Car</td><td>1002</td><td>7.7</td><td>84.8</td><td>31.8</td><td>10.1</td><td>34.8</td><td>39.6</td></tr><tr><td>Chair</td><td>999</td><td>7.4</td><td>82.6</td><td>57.1</td><td>9.7</td><td>56.3</td><td>18.8</td></tr><tr><td>Guitar</td><td>697</td><td>4.6</td><td>56.9</td><td>67.1</td><td>6.6</td><td>66.9</td><td>28.3</td></tr><tr><td>Helmet</td><td>90</td><td>0.6</td><td>7.6</td><td>23.7</td><td>0.9</td><td>20.6</td><td>0.6</td></tr><tr><td>Knife</td><td>270</td><td>1.8</td><td>21.2</td><td>50.0</td><td>2.5</td><td>50.0</td><td>11.0</td></tr><tr><td>Motorcycle</td><td>298</td><td>2.2</td><td>25.1</td><td>46.8</td><td>2.9</td><td>46.6</td><td>8.0</td></tr><tr><td>Mug</td><td>186</td><td>1.3</td><td>15.4</td><td>26.2</td><td>1.8</td><td>24.8</td><td>1.5</td></tr><tr><td>Table</td><td>1124</td><td>8.2</td><td>91.6</td><td>62.6</td><td>10.4</td><td>62.3</td><td>27.8</td></tr><tr><td>Vessel</td><td>910</td><td>6.7</td><td>75.5</td><td>49.6</td><td>8.5</td><td>49.7</td><td>32.8</td></tr><tr><td>Avg.</td><td>551</td><td>4.0</td><td>45.0</td><td>46.7</td><td>5.2</td><td>46.0</td><td>15.2</td></tr></table>

## F. Runtime Complexity

We provide a detailed analysis of the computational efficiency of the k-NN graph construction, geodesic eccentricity transform computation, and pseudo-label extraction that form the core of GEKey-PL. The runtime complexity is first analyzed theoretically, and a strategy to accelerate the geodesic eccentricity transform is proposed. In addition, the three processes are evaluated empirically on all KeypointNet [40] splits to measure their processing times and scalability to large datasets.

Let $\mathcal { G } _ { \omega } = ( \nu , \mathcal { E } , \omega )$ denote the weighted k-NN graph of the main paper, where |V| and |E| represent the cardinalities of the vertex and edge sets. A single Dijkstra traversal with an adjacency list and a binary heap from a source vertex in a weighted graph with non-negative weights has a time complexity of

$$
O ( ( | \mathcal { V } | + | \mathcal { E } | ) \log | \mathcal { V } | ) .\tag{12}
$$

When geodesic distances are computed from every vertex in the point cloud, there are $| \nu |$ such traversals, and the total complexity becomes

$$
O ( | \mathcal { V } | ( | \mathcal { V } | + | \mathcal { E } | ) \log | \mathcal { V } | ) .\tag{13}
$$

In the particular case of a k-NN graph, the number of edges satisfies $| \mathcal { E } | = O ( k | \mathcal { V } | )$ . Substituting this gives

$$
O ( | \mathcal { V } | ( | \mathcal { V } | + k | \mathcal { V } | ) \log | \mathcal { V } | ) = O ( k | \mathcal { V } | ^ { 2 } \log | \mathcal { V } | ) .\tag{14}
$$

The eccentricity transform is therefore the main bottleneck of GEKey-PL, as its cost scales linearly with the number of Dijkstra traversals and hence quadratically with |V| when every vertex is taken as a source, whereas the k-NN graph construction and the pseudo-label extraction remain comparatively cheap.

To enhance scalability, we introduce an acceleration strategy based on the farthest point sampling (FPS) algorithm. Rather than computing geodesic distances from every vertex, we select a maximally spaced subset of FPS samples as sources, since traversals from central vertices have a negligible effect on the eccentricity. The geodesic eccentricity for each vertex $u \in \mathcal V$ is then determined as the maximum distance from this subset of source vertices $s \subset \nu$ , defining the mapping $u \mapsto \csc s ( u )$

$$
\mathop { \mathrm { e c c } } _ { \cal S } ( u ) = \operatorname* { m a x } _ { v \in { \cal S } } d _ { { \mathcal { G } } _ { \omega } } ( u , v ) , \quad u \in \mathscr { V } .\tag{15}
$$

This approach reduces the number of Dijkstra traversals from $| \nu |$ to |S| while ensuring good coverage of the point cloud, so that the complexity of the transform becomes

$$
O ( | S | ( | \mathcal { V } | + k | \mathcal { V } | ) \log | \mathcal { V } | ) = O ( | S | k | \mathcal { V } | \log | \mathcal { V } | ) .\tag{16}
$$

For a fixed seed budget |S|, with $| S | \ll | \nu |$ in practice, the transform is thus linear in |V| up to the logarithmic factor, which keeps it tractable on larger point clouds. In the next paragraph, we verify empirically that reducing the number of FPS sources preserves the keypoint detection accuracy.

In Tab. 10, the total processing time $T _ { \mathrm { T o t } }$ per category for the three stages of GEKey-PL is reported on an Intel Core i7-12800H CPU, where a category contains 551 objects on average. The left column group shows the k-NN graph construction times, which take four seconds on average per category. The two middle column groups compare the subset sizes $| S | = 1 0 0 { \mathrm { ~ a n d ~ } } | S | = 1 0$ , where reducing the number of FPS sources cuts the eccentricity computation from 45.0 to 5.2 seconds, while the mIoU only drops from 46.7% to 46.0%. This highlights the effective approximation of the eccentricity function from FPS source points. The extraction times for the pseudo-labels defined by Eq. (6) of the main paper, shown in the right column group, also remain negligible with 15.2 seconds on average per category. In summary, the empirical evaluation confirms that GEKey-PL is computationally efficient and scalable to large datasets, making it suitable for practical applications in keypoint detection.

![](images/d500fb7d0a4922b3f0ac17a4eaa7df4997649c6ad57ab460d75401edea8f30f9.jpg)  
Figure 13. Geodesic distances to the wingtip point on the Delaunay triangulation, alongside geodesic distances on the k-NN graph fo comparison.

![](images/ace7502892241d1e867809d07fa923470a621e41e05e761d7defd990f39f36a5.jpg)  
Figure 14. The local maxima of the geodesic eccentricity transform on the Delaunay triangulation and the k-NN graph.

## G. Delaunay Triangulation

Given the non-uniform sampling of our point clouds, we employ a k-NN graph rather than Delaunay triangulation to avoid overlong edges. As shown in Fig. 13, Delaunay triangulation connects antipodal points across the surface, introducing shortcut edges that bypass the underlying surface geometry and distort geodesic distance estimates to the wingtip source point. These shortcuts also directly link antipodal local maxima, corrupting the local maxima detection in Fig. 14. In contrast, the k-NN graph enforces geometric proximity between connected neighbors, preserving local topology for accurate geodesic estimation and reliable local maxima detection.

## H. k-Parameter

Based on previous studies [36, 46], the k-parameter was set to 20 to connect the k-NN graph for the geometric analysis in preceding experiments. In this section, to assess the influence of the k-parameter on pseudo-label extraction, we vary its value from ten to forty in increments of ten and compare the resulting keypoint detection performance on all KeypointNet [40] splits.

Table 11. Comparison of the k-parameter in the k-NN graph of GEKey-PL on pseudo-label Coverage (Cov., %) and mIoU (%), computed with ecc<sub>10</sub>(u).
<table><tr><td rowspan="2"></td><td colspan="2">10-NN</td><td colspan="2">20-NN</td><td colspan="2">30-NN</td><td colspan="2">40-NN</td></tr><tr><td>Cov.</td><td>mIoU</td><td>Cov.</td><td>mIoU</td><td>Cov.</td><td>mIoU</td><td>Cov.</td><td>mIoU</td></tr><tr><td>Airplane</td><td>98.7</td><td>79.4</td><td>99.2</td><td>80.3</td><td>99.3</td><td>78.3</td><td>99.3</td><td>77.1</td></tr><tr><td>Bed</td><td>98.5</td><td>37.5</td><td>98.7</td><td>42.3</td><td>98.6</td><td>39.6</td><td>98.6</td><td>38.6</td></tr><tr><td>Bottle</td><td>96.4</td><td>44.4</td><td>97.0</td><td>39.1</td><td>97.1</td><td>38.0</td><td>97.2</td><td>37.5</td></tr><tr><td>Cap</td><td>93.2</td><td>22.0</td><td>93.9</td><td>24.9</td><td>95.0</td><td>23.6</td><td>95.4</td><td>25.1</td></tr><tr><td>Car</td><td>94.2</td><td>42.2</td><td>95.4</td><td>34.8</td><td>95.8</td><td>35.5</td><td>95.9</td><td>36.0</td></tr><tr><td>Chair</td><td>97.3</td><td>52.6</td><td>97.6</td><td>56.3</td><td>97.7</td><td>53.3</td><td>97.7</td><td>52.8</td></tr><tr><td>Guitar</td><td>98.2</td><td>69.8</td><td>98.7</td><td>66.9</td><td>98.8</td><td>65.7</td><td>98.9</td><td>66.1</td></tr><tr><td>Helmet</td><td>88.3</td><td>19.8</td><td>87.3</td><td>20.6</td><td>87.6</td><td>22.0</td><td>87.7</td><td>21.9</td></tr><tr><td>Knife</td><td>99.5</td><td>58.0</td><td>99.6</td><td>50.0</td><td>99.6</td><td>47.7</td><td>99.6</td><td>47.0</td></tr><tr><td>Motorcycle</td><td>93.7</td><td>43.5</td><td>94.8</td><td>46.6</td><td>95.4</td><td>43.8</td><td>95.9</td><td>42.3</td></tr><tr><td>Mug</td><td>95.2</td><td>23.0</td><td>95.4</td><td>24.8</td><td>95.8</td><td>24.2</td><td>96.1</td><td>24.1</td></tr><tr><td>Table</td><td>98.5</td><td>56.8</td><td>98.5</td><td>62.3</td><td>98.5</td><td>57.0</td><td>98.4</td><td>55.0</td></tr><tr><td>Vessel</td><td>97.3</td><td>52.5</td><td>98.2</td><td>49.7</td><td>98.4</td><td>48.1</td><td>98.5</td><td>48.0</td></tr><tr><td>Avg.</td><td>96.1</td><td>46.3</td><td>96.5</td><td>46.0</td><td>96.7</td><td>44.4</td><td>96.9</td><td>44.0</td></tr></table>

For these experiments, after the construction of the k-NN graph for 2048 points, the geodesic eccentricity transform map is computed with the use of ten FPS source points ecc<sub>10</sub>(u) as described in Tab. 10. The effect of the kparameter on the coverage and mIoU metrics is summarized in Tab. 11 to highlight the variations in pseudo-label quality. From the results, we observe that the four k-parameters generally lead to a similar coverage, while slight decreases in mIoU occur as the parameter increases. In conclusion, the choice of k has a limited impact on pseudo-label quality, and the selected value of 20 provides a good balance between graph connectivity and computational efficiency.

## I. Near-Isometric Deformations

We generate near-isometric human body variations using the SMPL [20] deformation basis and evaluate keypoint stability against other geometric detectors on two metrics. The repeatability measures keypoint stability between a pair of during training. The results illustrate GEKey’s synthetic-toreal generalization that preserves accurate keypoint detection and consistent alignment on real-world, cluttered scans.

![](images/0c2da021e786db444f696184b85f65d7c80fb429a0dfc889b3dccf49f669c8e2.jpg)  
Figure 16. GEKey exhibits strong synthetic-to-real generalization on real-world examples from the OBJ-BG variant of ScanObjectNN, despite challenges such as background clutter, partial occlusions, sensor noise, and varying point density.

Table 12. Repeatability and consistency error between SMPLdeformed shapes for ten keypoints.
<table><tr><td>Method</td><td>Repeatability (%)↑</td><td>Consistency (×10−2)↓</td></tr><tr><td>FPS [27]</td><td>11.6</td><td>12.0</td></tr><tr><td>ISS [48]</td><td>9.0</td><td>11.0</td></tr><tr><td>GEKey-PL</td><td>32.3</td><td>4.4</td></tr></table>

![](images/10e4f13938dd997c9bbce4ad17aa00f2f736d062996178a79a140f1ba6b73941.jpg)  
Figure 15. GEKey-PL with stable keypoints across deformations of the SMPL human model.

deformed shapes by checking whether a keypoint detected on one model has an exactly corresponding point detected on the other. The consistency error complements this by computing the average distance between each detected keypoint and the nearest keypoint on the other model. As shown in Fig. 15 and Tab. 12, GEKey-PL outperforms both baselines on both metrics, confirming the stability and spatial accuracy of the detected keypoints across deformations.

## J. Uncontrolled Real-World Scans

We further demonstrate the generalizability of our method to the OBJ-BG variant of ScanObjectNN [34], a challenging dataset designed for object classification. It contains point clouds captured directly from indoor scene scans, introducing real-world noise and clutter. Despite these challenges, we show that our keypoint detector trained on synthetic data generalizes well to the real-world domain. In particular, the model of the chair category trained on ShapeNetCoreV2 for keypoint detection and alignment in Tabs. 3 and 4 is applied directly to the chairs of ScanObjectNN, without any additional training or fine-tuning. We visualize in Fig. 16 the predicted keypoints on chair scans from a domain unseen