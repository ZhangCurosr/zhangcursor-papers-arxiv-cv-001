# A Systematic Point Cloud Edge Detection Framework for Automatic Aircraft Skin Milling

Zijie Wu, Yaonan Wang, He Xie, Mingtao Feng, Haotian Wu, Chao Ding, and Ajmal Mian

Abstract—The edge detection technique is an essential step for aircraft skin milling in aviation manufacturing. Most of the current detection methods focus on traditionally defined edge extraction tasks but disregard the crucial systematic requirement of edge milling. In this paper, we proposed a novel edge detection framework for automatic edge milling of aircraft skins. First, an edge probability detector is proposed by the spatial tangent continuity (STC) to provide the essential reference. Second, we propose a hierarchical branch searching (HBS) method to hierarchically strip the desired milling edges from the raw point cloud, which consists of three graded progressive steps: branch backbone generation, branch extension, and branch pruning. We demonstrate the performance of the proposed method on both synthetic models and aircraft skin workpieces. The proposed method outperforms the other baselines and shows accurate edges for the edge milling task.

Index Terms—Edge detection, aircraft skin, automatic edge milling, accurate 3D measurement

## I. Introduction

A <sup>IRCRAFT</sup> <sup>skin</sup> <sup>[1]</sup> <sup>has</sup> <sup>a</sup> <sup>significant</sup> <sup>role</sup> <sup>in</sup> <sup>aviation</sup><sub>manufacturing and is usually assembled from multi-</sub> manufacturing and is usually assembled from multi ple parts to cover much of the aircraft. The skin parts are riveted to the beams and trusses of the aircraft skeleton. Then, the clearances of adjacent skins are controlled by the edge machining procedure. In the assembly, the edge of the skin part is the critical reference for machining the baseline of the edge milling and the related drilling riveting. More importantly, the milling accuracy of the aircraft skin involves the aircraft aerodynamic performance and stealth performance and is essential for the internal structure safety of the aircraft system. The traditional method of aircraft skin machining, including manually comparing and marking the assembly location and removing the edge allowance of the skin part, depends on manual intervention. However, limited by the performance of manual machining, problems including low eficiency, poor consistency, and poor accuracy make it challenging to achieve eficient and accurate aircraft skin assembly.

Motivated by eficient and accurate measurement, the vision-based, 3D measurement system [2] [3] is usually more consistent, accurate, and eficient than the traditional manual method and hence applicable to provide a better solution to detect and locate the edge of the skin part. Diferent from 2D images, point cloud data directly reflect the spatial location of the measured workpiece and are easily utilized for the machining path planning of intelligent manufacturing systems. Nonetheless, a visionbased milling system demands an eficient and accurate edge detection method to separate the desired milling edge from the raw data. Moreover, the thickness of aircraft skins creates dificulties when multiple edge types exist in the current 3D milling edge detection task. Sequentially, it is crucial for robotic edge milling to construct an eficient and accurate milling edge detection framework.

Fig. 1 shows a common illustration of the assembled aircraft skin. The diferent viewpoint results in the visible side of thickness and invisible thickness. We define two types of edges in the second part of Fig. 1, which constitute the desired upper milling edges: 1) A closed edge is usually the intersecting line between two adjacent surfaces. These edges usually have severe curvature variance. 2) An open edge is usually the intersecting line between the measured surface and the blank space. The aircraft skin is scanned from the measured surface to the invisible side, and there is no curvature variance but a distinct diferent point distribution. Because of the complex industrial measurement environment, a complete point cloud is dificult or impossible to obtain in practice, which leads to vague detection caused by the simultaneous appearance of two types of edges. In this paper, we proposed a novel milling edge detection framework based on 3D unorganized point clouds to obtain an accurate milling edge.

Over the past few decades, many algorithms have been proposed to detect crucial edges for variant applications. Some works [4] [5] focused on diferential operators to detect the gradient variance in 2D pixel images. The Laplacian operator [6] detects the image edges by the changes in the second derivative. These methods usually work in most cases and give a fine performance of detected 2D edges. However, these pixel-based, 2D edge detectors [7] highly depend on the texture of the gray level image, and the accuracy is limited by the image resolution. Therefore, subpixel-based detection techniques [8] are proposed to improve the accuracy of edge detection. Zhang et al. [9] applied the pseudo-Zernike moment to detect the subpixel edges as a preprocessing step. Xiong et al. [10] combined cubic spline interpolation with mathematical morphology to enhance the reliability of edge detection. Fitting the edge curves [11] was proposed to directly improve the accuracy by moving least squares. The subpixel-based detection methods showed better accuracy than pixelbased methods. However, these methods still limit 2D images and cannot achieve the precise edge detection applied for industrial aircraft skin assembly.

Hence, numerous 3D-based detection techniques have been proposed to address this problem. Open Edge Detection: Open Edge detection is usually applied in border extraction between the object and the background. One type of solution [12] [13] [14] in 3D is often an extension of 2D edge detection techniques. Xi et al. [12] converted point cloud patches that were divided by the coplanar condition to a 2D range image and applied the Laplacian operator to detect the edges. Bao et al. [13] used the 2D Canny operator to detect the edges on the range image and included the 3D points corresponding to the detected 2D pixels as the 3D edges. This approach takes advantage of the 2D Canny, which is well applied to 3D areas and shows more robust edge detection performance than other 3D extensions from 2D areas. Steder et al. [14] proposed the normal aligned radial feature (NARF) descriptor to extract the boundaries (edges). This descriptor explicitly considers the borders of the objects and the backgrounds, which can obtain an accurate result for object feature description. Ahmed et al. [15] employed the nonparametric estimation of the mean-shift algorithm by computing the moving distance to classify the edges. Closed Edge Detection: Closed Edge Detection is often utilized in model contour description to simplify the object shapes. Several works focus on the direct detection of 3D point cloud data. Bazazian et al. [16] computed the curvature of the local surface by eigenvalue filtering. Although it gives a coarse edge detection result, eigenvalue filtering eficiently utilizes singular value decomposition. Demarsin et al. [17] proposed a graph theory to refine the detected edges from normal estimation. The graph-based method [18] is well applied to closed edge detection. Furthermore, the methods using curvature information such as normal are usually time-consuming with low productivity and cannot simultaneously detect all types of edges such as open edges and closed edges. It is not applicable in aviation manufacturing with the demand for high accuracy.

![](images/0b033c40dbcdf45b5ec8639f8c96d56a95cbb4882e1e211e50e6f1eb133a1cc3.jpg)  
Fig. 1. Illustration of the significance of milling edge detection. The point cloud is scanned by the camera and then employed for milling edge detection. The edge information is then applied in robotic edge milling by a machining tool to remove the allowance for final aircraft skin assembly.

The following two key conditions are significant for achieving accurate and eficient edge detection in industrial manufacturing of aircraft skin assemblies:

1) The detected edge point should have high accuracy: The edge milling for aircraft skin assembly requires an accuracy of ±0.5 mm, a thickness of approximately 2-6 mm [19], and a longitudinal deflection of ≤0.3 mm. The edge information is an important reference for controlling the clearances of adjacent skins and measuring the thickness, so it is necessary for the detection algorithm to output the precise point set of edges. Nonetheless, the features of diferent types of edges make accurate edge detection challenging.

2) Only the desired milling edges need to be detected In industrial measurement, the closed edges brought by the thickness of the aircraft skins are mixed with the open edges. However, the desired milling edge consists of both types, and hence, the greatest challenge is to accurately detect the milling edge branches to direct robotic milling.

In this paper, we proposed a novel edge detection framework that consists of an edge probability detector using spatial tangent continuity (STC) for edge point description and the hierarchical branch searching (HBS) method to perform eficient milling edge detection on aircraft skins. This method provides a hierarchical branch searching framework for upper milling edge detection on aircraft skins, which solves the three main problems of edge detection described above in aviation manufacturing. In summary, our contributions are given as follows:

1) A simple continuous geometric feature is proposed to precisely detect edges: We propose a spatial tangent continuity detector method around the local identified point neighbors to describe the edge probability for each point. We give the common definition of edge on the measured point cloud. The proposed detector can detect the desired edges used for aircraft skin milling.

2) The hierarchical branch searching framework is proposed for accurate edge detection We propose a unified edge detection framework that accurately detects the milling edges and approaches a smooth machining path for robotic measurement. This framework is an essential support for the robotic assembly of aircraft skins.

## II. Systematic assumptions

Aircraft skin edge detection is the problem of detecting the complete milling edges within various perturbations. We consider the point cloud acquisition process from the measured aircraft skin for illustration. As shown in Figure. 2, multiple types of points are acquired with the structure light camera. The milling edge (Solid red line) is defined as the complete and end-to-end boundary for the use of robotic machining requirements. The purpose of our edge detection method is to strip the desired milling edge from the scanned raw point cloud. However, various perturbations to the desired milling edge are challenging to achieve accurate detection.

Several assumptions are considered in terms of the workpiece, point cloud, and systematic process in our detection system: Workpiece assumptions: 1) The aircraft skin is assumed as the curved workpiece with thickness of δ, which results in multiple types of perturbation to desired milling edge detection. 2) The reflectivity of each measured point on skin is considered to be 100% scanned. We do not consider the occlusion brought by the low reflectivity of overexposure. Point cloud assumptions: 3) The scanned point cloud of skins is assumed to be acquired with uniform density. The unusual density points are preprocessed by spatial filtering. 4) Each point cloud contains the location of a crucial physical boundary. Otherwise, the detected edge is not reliable. For instance, there is no edge to detect on the measured point cloud of the metal ball. System assumption: 5) The process of the milling edge is assumed to be conducted after the scanning and before the skin machining. Edge detection and point cloud acquisition processes are performed separately.

![](images/643d6262a37ed35a971b6ceff049d6a089e201017a21c6f5c0bd935c4c5f812c.jpg)  
Fig. 2. Illustration of the systematic problem formulation: the scanned point cloud from the measured aircraft skin consists of multiple types of points.

Workpiece assumptions show that we focus on the problem of perturbations that are resulted from thickness. Point cloud assumptions define the basic parameters of data with uniform density to ensure the eficiency of parallel computing in each sub-area. System assumption is given to illustrate the application in industrial measurement.

As shown in Figure 2, the numerous nonedge points make up the vast majority of the total point cloud. It is crucial to split the edge points from the numerous points of the skin point cloud. Furthermore, the desired milling edge, which is defined as an end-to-end edge on the same surface, is still mixed with a large number of undesired edge points due to the thickness of workpiece assumption. In fact, the undesired and desired edges are dificult to distinguish because of their similar properties. Milling edge can be directly applied as the aircraft skin machining path and hence demands high accuracy. The smoothness of the milling path is a necessary requirement for robotic machining. Therefore, detecting a smooth and accurate milling edge, as shown in the solid red lines of Figure 2, is the purpose of our edge detection framework.

## III. Approach

In this section, we introduce our edge detection framework, which is shown in Fig. 3. The proposed framework consists of a STC descriptor and a HBS method. STC is computed as the essential reference to instruct edge detection in Section. III-A. Based on the proposed STC, we propose the HBS method to hierarchically and accurately detect the desired milling edge in Section. III-B. First, the edge detector for the initial backbone of edge branch construction is proposed to eficiently establish the general edge structure in Section. III-B1. Second, based on the features of aircraft skins, we also present a linear evaluation algorithm for more feasible edge point selection upon initial edge selection. This module is designed to smoothly extend the branches from the initial branch backbone. We describe the cross-validation method with the two edge criteria in Section. III-B2. This process is meant to include the edge points based on the initial structure as much as possible but also introduces the incorrect edge branches that are hard to distinguish. Hence, we proposed a graph optimization for edge refinement in Section III-B3 to prune these incorrect point branches and to obtain an accurate and smooth edge for robotic aircraft skin milling. The main advantages of the hierarchical branch searching method can be concluded as:

![](images/e47107f6c7ea0cb6fafddabc1bc720f4921058e7a5d8364f74dd8e6ab1f28f79.jpg)  
Fig. 3. Framework of the proposed milling edge detection.

1) HBS employs a local edge-aware STC descriptor and designs a hierarchical framework to precisely select the edge point in 3D space.

2) Hierarchical framework is tolerant to the multiple types of perturbations. For instance, the uncompleted bottom edge lines can be recognized and pruned, which is crucial for automatic aircraft skin measurement.

3) HBS detects an accurate milling edge that can be directly applied for robotic machining.

## A. Spatial Tangent Continuity for Edge Point Selection

Diferent from the existing method that relies on the point distribution, we rethink the definition of the edge and nonedge in 3D. Here, the spatial tangent continuity is proposed to distinguish the nonedge point from edge point. We define the tangent continuity at a point condition:

Definition 1: (discrete tangent continuity for a spatial nonedge point) If point p does not belong to the edge, there exists a constant $\delta \ > \ 0$ such that all points $\| { \pmb p } _ { s } - { \pmb p } \| \ < \delta$ of the measured surface constitute a neighbor plane P . The plane P is

![](images/97e88dc222532446919ef1b6d6570dc6b7a19a6c364b4115e4740795a715e6da.jpg)

![](images/7de66432d8a8fc43b02561a21a15a9591a14e6562ef295447aaef5019529e14c.jpg)  
(a) Strictly continuity  
Fig. 4. Illustration for the definition of spatial tangent continuity.

(i) strictly tangent continuous at $p , \mathrm { i f } \exists p _ { s } ^ { - } \in P$ and there exists a constant $\varepsilon > 0$ 2

$$
\left. { p - p _ { s } ^ { - } , p _ { s } - p } \right. < \varepsilon\tag{1}
$$

for all $p _ { s } \in P , \langle , \rangle$ is the angle between two vectors,

(ii) strongly tangent continuous at $p { \mathrm { ~ i f ~ } } \exists p _ { s } ^ { - } \in P$

$$
\left. { p - p _ { s } ^ { - } , p _ { s } - p } \right. = 0\tag{2}
$$

for all $\pmb { p } _ { s } \in \mathcal { P }$

Definition 1 is illustrated in Fig. 4. If P is strongly tangent continuous at p, then P is strictly tangent continuous at p. Definition 1(ii) is considered the tangent continuity of all points in the continuous surface, and Definition 1(i) is the generalized solution upon the point cloud. We apply the pairwise vector to approach the tangent and compute whether a collinear and opposite tangent exists for all tangents that go through the point p. With the above definition, we obtain the following discrete computation to obtain a probability of the point p that belongs to an edge.

Discrete computation of tangent continuity on the point cloud: The 3D point cloud is considered the observation point set from the measured continuous surface. However, due to the essence of the observation sample, the point cloud is inherently sampled with gaps. We only consider a strict continuity on discrete point cloud data to extract the edge point.

For a point p, we apply discrete computation for continuity estimation. First, we consider the neighbors ${ \pmb p } _ { s = 1 } ^ { n } = \{ { \pmb p } _ { 1 } , { \pmb p } _ { 2 } , . . . , { \pmb p } _ { N } \}$ for which the Euclidean distance is less than $\delta ,$ where N is the number of neighbors. Second, for each neighbor point $\mathbf { \nabla } _ { \pmb { p } _ { s } } .$ , we estimate if it meets the condition of strict continuity. The greater the number of $\pmb { p } _ { s }$ that is not strictly continuous, the higher the probability that it belongs to an edge. The probability E is computed as:

$$
E ( \pmb { p } ) = \sum _ { s = 1 } ^ { N } C ( \langle \pmb { p } - \pmb { p } _ { s } ^ { - } , \pmb { p } _ { s } - \pmb { p } \rangle ) / N ,\tag{3}
$$

where C is the counting function for the number of $\pmb { p } _ { s }$ when there is no $\pmb { p } _ { s } ^ { - } \in \pmb { p } _ { s = 1 } ^ { n }$ meeting the condition that

$$
\begin{array} { r l } & { \langle { \pmb p } - { \pmb p } _ { s } ^ { - } , { \pmb p } _ { s } - { \pmb p } \rangle < \varepsilon . } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}\tag{4}
$$

In practice, the neighbor points are not always distributed within a neighbor plane. Against the numerical sampling radius $\textstyle R _ { c } ,$ we apply the adaptive continuity threshold:

$$
\varepsilon = \frac { 2 \pi } { N } \exp ^ { \frac { \| p - { \pmb p } _ { s } \| } { R _ { c } } } ,\tag{5}
$$

This term is applied by considering more tangent bias tolerance as the distance increases. Based on the edge probability of each point, point p is selected as an edge candidate by the proposed HBS method.

B. Hierarchical Branch Searching for Milling Edge Detection

In this section, we propose a hierarchical branch searching method to detect the desired milling edge.

1) Branch Backbone Generation: For aircraft skins, the edge points of complex edges show much divergence from each other. For instance, points in corners, lines, arcs, etc., have diferent point distributions and result in diferent edge probabilities.

The edge candidates generated by probability selection contain many suboptimum edge points, which yields a high probability because their locations are near the real edge. To determine the optimal edges, the problem that we faced is to choose the threshold of the probability score for edge point identification. However, due to the various edges, such as convex and concave edges, the confidence probability score of optimal edge points is diferent for each edge in a point cloud. A single threshold value is dificult to distinguish for eficient edge detection, and a dynamic and flexible filter scheme is needed.

Hence, we divide the measured points with a threshold tuple $\{ E _ { m a x } , E _ { m i n } \}$ . We generate a rough branch backbone using high edge probability points by selecting the point p when $E ( \pmb { p } )$ is larger than the threshold $E _ { m a x }$ and discard the point p when $E ( \pmb { p } )$ is lower than the threshold $E _ { m i n }$

2) Branch Extending by Cross Validation: Based on the edge probability $E ( \pmb { p } )$ , the edges can be coarsely selected by threshold filtering. However, as shown in most edge detection methods, the single threshold filter hardly addresses the complex edge variation, especially on aircraft skins. For point p with $E _ { m i n } ~ \le ~ E ( \pmb { p } ) ~ \le ~ E _ { m a x }$ , it is dificult to judge only with the edge probability criterion. We introduce an auxiliary criterion, Linear Degree, for better branch extension as follows:

Hence, the linear degree is applied to provide extra information, which is defined as:

Definition 2: (Linear Branches Degree) For an ordered point set $\{ p | p _ { 1 } \to p _ { 2 } \to p _ { 3 } \to p _ { 4 } \to \cdot \cdot \cdot \to p _ { M } \}$ , this branch is assumed to have a linear degree if all $\left\{ < p _ { i } - \right.$ $p _ { i - 1 } \to p _ { i + 1 } - p _ { i } > \} \leq \theta .$

The value $\theta = 0$ usually determines a straight line. For better tolerance of measurement noise, θ is usually set as a suficiently small constant.

Linear Clustering: A clustering algorithm is proposed to separate the line out; for instance, the lines that belong to edges contain a larger number of points. Here, P is the candidate set of all selected edge points and $\mathbf { \Delta } _ { \mathbf { \mathcal { p } } _ { i } } \in \mathcal { P }$ is the i-th point. We propose the progressive accessible clustering algorithm (PAC) to fulfill the linear degree estimation and implement it based on the backbone of the DBSCAN [20] algorithm.

The PAC is proposed to segment edges from local neighbors. This algorithm is based on the concept of Progressive and Accessible, which is defined as follows: if the neighbor point $\| \pmb { p } _ { A } - \pmb { p } _ { i } \| < \epsilon _ { a } , \pmb { p } _ { A }$ is accessible from $\mathbf { \nabla } _ { \mathbf { p } _ { i } , \mathrm { ~ } }$ each time of accessible propagation, the depth from measured point p to $\pmb { p } _ { i }$ must be increasing. A cluster consists of all segments that are accessible to each other. Furthermore, the clusters whose number is less than min\_pts would be labeled as noise points and deleted from linear estimation. The whole accessible clustering algorithm is detailed in Algorithm. 1. The number of linear clusters provides extra information to detect the edge point when it has a relatively smaller edge probability.

Algorithm 1: Progressive Accessible Clustering Al  
gorithm   
Input: Clustering Candidates P, Measured point p,   
Least cluster number min\_pts,   
Accessible radius $\epsilon _ { a } ;$   
Output: Linear Degree L;   
1 Init: Current cluster label $L = 0 ,$ depth $D { = } 0 ;$   
2 $\mathcal { P }$ is set to the origin: $\mathcal { P } _ { i } = \mathcal { P } _ { i } - p ;$   
3 $\mathcal { P }$ is projected to the unit ball: $\mathcal { P } = \mathcal { P } / l e n g t h ( \mathcal { P } )$   
4 foreach unlabeled point $\mathbf { \Delta } _ { \mathbf { \mathcal { p } } _ { i } } \in \mathcal { P }$ do   
5 The current depth is updated:   
$D _ { l a s t } = D , D = l e n g t h ( \pmb { p } _ { i } )$ ;   
6 $\mathcal { Q } = R a n g e S e a r c h ( \pmb { p } _ { i } , \epsilon _ { a } ) ;$   
7 while $\mathcal { Q } \neq \mathcal { Q }$ do   
8 if $\left( D - D _ { l a s t } \right) \cdot \left( \| \pmb { p } _ { i } \| - \left\| \pmb { p } _ { i - 1 } \right\| \right)$ then   
9 Label $\mathbf { \mathscr { p } } _ { i } \in \mathcal { Q }$ as $L ;$   
10 end   
11 $\mathcal { Q } = R a n g e S e a r c h ( \mathcal { Q } , \epsilon _ { a } ) ;$   
12 end   
13 $L = L + 1 ;$   
14 end   
15 for $j \in L$ do   
16 if |Cluster<sub>j</sub>| ≤ min\_pts then   
17 Cluster<sub>j</sub> is labelled as noise;   
18 end   
19 end

Cross Validation by Auxiliary Linear Degree: To improve the smoothness of edges, we follow the works of [4], and the linear degree of the edge point is evaluated to ensure the quality of the generated edges. We employ two thresholds $E _ { m i n }$ and $E _ { m a x } \ \left( E _ { m i n } \ < \ E _ { m a x } \right)$ in the cross-validation process, where a value lower than $E _ { m i n }$ represents a low confidence of the nonedge point to discard these points and a value higher than $E _ { m a x }$ to include the high confidence points as edges. The backbone of the initial edges is extracted by selecting the high confidence points with the edge probability detector.

![](images/c3da682a9ca212b3c1d98250fbc1aa2891b8de0cb5a4ecb2a4119f2f87af3507.jpg)  
Fig. 5. Branch Refinement by Graph Optimization. Every row proceeds in parallel, and all linear branches are merged at the end to constitute the final milling edge.

As defined in Definition. 2, all points of the branch backbone are computed to obtain their related linear branches. These edge points of aircraft skins should have their branch points, which also belong to the edges. Hence, we evaluate both criteria to approach the accurate edges, namely, the cross validation:

1) The linear branches of all points in the initial edges that $E ( \pmb { p } ) > E _ { m a x }$ are computed, and all points that $E ( \pmb { p } ) < E _ { m i n }$ are discarded.

2) The points in the initial edges that have no linear branches are discarded and labelled as noise.

3) These points in the linear branches that $E ( p ) \ >$ $E _ { m i n }$ are included as edges.

Based on this cross-validation process, the edges consisting of points and branches are detected.

3) Branch Pruning by Graph Optimization Refinement: Traditional methods for edge detection end with the edge candidates that are selected in Section III-B1. However, industrial manufacturing, such as measuring and machining aircraft skin, demands an accurate 3D case. Vague information usually leads to measurement bias and machining jitter. We propose a branch extension scheme to improve the precision. In this section, we propose a graph optimization theory to guarantee smoothness for better milling edge detection to support the robotic system.

Graph reconstruction: Based on the branches extending after backbone establishment, the milling edge is roughly detected. However, for milling operations, the edge is desired with smoothness and accuracy. Hence, the graph reconstruction method is applied for error branch pruning. As shown in Fig 5, we consider the coarse edges after clustering as the backbone of the graph structure.

(i) All points in a cluster are locally connected to the neighborhood graph G to prevent local perturbations. In practice, the perturbations distort the points around the real edges and cause the incorrect connection if a graph is established with few connected neighbors. To solve this problem, previous works usually connect all points on a large scale, which would be computationally intensive to prune the numerous unwanted connections. Furthermore, the diferent lines of aircraft skin edges are not always far from each other. The scheme of large-scale connections would produce unexpected connections. In our solution, we give a related feasible receptive field to produce the local connections to enable path searching across the local perturbations. These connections with the local receptive field of radius $R _ { l }$ are constructed on the backbone of the clustered coarse edges to a global graph.

(ii) The local optimal edge of point p is estimated by all its neighborhood connections with graph radius $R _ { g }$ $( R _ { g } > R _ { l } )$ . The local graph of point p is taken out to search for an ordered optimal path by the shortest path method. Formally, given the chains of local adjacent edges, there is only one path of the shortest path searching by the defined path cost from $\pmb { p }$ to each neighborhood. The path cost is defined in two parts. We obtain the probability cost:

$$
\mathcal { C } _ { p } = 1 - \frac { E ( \pmb { p } _ { i } ) + E ( \pmb { p } _ { j } ) } { 2 } ,\tag{6}
$$

The cost forces on the connections are obtained from the points with high edge probability. In addition, a Euclidean distance cost is applied to connect the neighbors:

$$
\mathcal { C } _ { e } = 1 - \frac { d _ { \# } } { \left\| \pmb { p } _ { i } - \pmb { p } _ { j } \right\| } ,\tag{7}
$$

where the Euclidean cost is to force edges from the nearer neighbors. $d _ { \# }$ is the average distance of the measured point cloud.

The overall cost is ${ \mathcal C } \ = \ { \mathcal C } _ { p } + { \mathcal C } _ { e }$ . Fig. 5 shows the diagram of the graph optimization. We construct our local optimal edge on the connected cost function by running the minimum span tree algorithm [21] and by searching the shortest paths [22]. Each neighborhood point generates an ordered path from p that is represented by the chains of adjacent edges. All paths from p are evaluated for their linear degree, and hence, the top two nonoverlapping edges are selected based on the linear degree to constitute the upper milling edges. The purpose of the linear degree evaluation on the edge graph instead of directly on the edge points is to better preserve the corner points and improve the accuracy.

## C. Time Complexity Analysis

Assume the point cloud of the measured point cloud contains the m number of total points and the n number of edge points. The proposed approach consists of branch backbone generation, branch extending, and branch pruning. In branch backbone generation, the computation of STC for every point consists of a k nearest neighbor (knn) searching for all points and a top-1 nearest neighbor searching in the knn set. The construction of the searching tree takes O(m log m) time. The top-1 nearest neighbor searching of STC takes ${ \cal O } ( \log \mathcal { R } )$ time, in which R is the number of neighbors.

![](images/817231ff39b04b6898bb048561f85a3661ec652416899264ce0c742cdab560de.jpg)

![](images/9da0172967774f04e486b13636095d01fd03af6e590d872ea7bf23ff056b36e3.jpg)  
(a) Model 1  
Fig. 6. Skin models  
(b) Model 2

![](images/04471e2c95ee276636809252640e248cb4547dbebe9fd9e15cb4032835c98993.jpg)  
(c) Model 3

In branch extending, the time cost of the branches extending process is mainly from the linear clustering scheme using the PAC algorithm and takes $O ( n ^ { 2 } )$ time. Using the k-dimension tree building, the time complexity can be reduced to $O ( n \log n )$

Last, for branch pruning, the total time cost consists of minimum span tree construction around the desired edge points and shortest path searching. Minimum span tree construction takes $( \bar { O } ( n ) \cdot O ( R _ { g } ^ { 2 } \log { R _ { t } ^ { 2 } } )$ time, where the $R _ { g } , R _ { t }$ are the knn searching radius take $2 \cdot O ( n \log n )$ time. Shortest path searching take nearly $O ( n ) \cdot O ( R _ { g } ^ { 2 } \cdot R _ { t } ^ { 2 } )$ time.

Furthermore, the parallel technique can be used to speed up HBS, since the algorithm is inherently taskparallel in all three processes due to the uniform density of point cloud assumption. Especially when dealing with low canning density, the time complexity of branch pruning can be reduced to $O ( \log n ) \cdot ( O ( R _ { q } ^ { 2 } \log R _ { t } ^ { 2 } ) + O ( R _ { q } ^ { 2 } \cdot R _ { t } ^ { 2 } ) )$ . We abbreviate $O ( \log \mathcal { R } ) , ( O ( R _ { g } ^ { 2 } \log R _ { t } ^ { 2 } ) + O ( R _ { g } ^ { 2 } \cdot \mathbf { \check { R } } _ { t } ^ { 2 } ) )$ as $K _ { 1 }$ and $K _ { 2 }$ respectively because they can be regarded as a constant when dealing with the uniform density of point clouds.

As a result, the whole edge detection consumes time T to be accomplished, in which $T$ is computed as follows:

$$
T = K _ { 1 } O ( m \log m ) + 3 \cdot O ( n \log n ) + K _ { 2 } O ( \log n )\tag{8}
$$

As the usual condition, the proportion of $n / m$ decreases as the point density increase. For instance, m $\gg n \gg \mathcal { R } \gg$ $R _ { g } > R _ { t }$ when dealing with a large number of points, the time cost is dominated by the STC construction. $T \approx$ $K _ { 1 } O ( m \log m )$

## IV. Experimental Results

In this section, three parts are designed and dedicated to testing the performance of an aircraft skin model with a complex surface. The first part tests the performance with four metrics to show the efectiveness of our edge detection method in Section A. The second part is designed to show the perturbation resistance ability with respect to measurement noise and outliers in Section B. These previous two parts are conducted based on the synthetic dataset for better evaluation. Hence, we evaluate the edge detection experiment above the standard workpieces on our robotic measurement system in the third part. The third part further verifies the advantages of our proposed method in a practical industrial environment. In addition, an aircraft skin with a complex free-form surface is also included to visualize the extracted edge. All aircraft skins are designed with a thickness of 5mm.

![](images/6a6102b38e41931987bd02fafbc975fe252251bde9e247dfdf2449d8d40cd340.jpg)

(c) Recall ( 100%)/Model 1  
![](images/1aa1fe21e3be06ff5ee6d4356b76a4765b5e4dec0a1c687580755a304d5dcf74.jpg)  
(b) Recall ( 100%)/Model 2

![](images/b5a8999caf4cbf62f2ae9330604ebcc001babad36566b55a43d57951f6643130.jpg)  
Fig. 7. Precision/Recall evaluation of diferent edge detectors.

Baselines: We compare our method HBS with two typical edge detection methods: ED represents detection via eigenvalue variations, and MS[15] is a mean-shift edge detector that relies on the neighbor point distribution.

Evaluation Metrics: Here, we employ four metrics [23] to evaluate the quality of edge extraction methods: accuracy, precision, recall, and error. Note that, the error is defined as the distance to the nearest reference points.

## A. Evaluation on the performance

In this section, we model the aircraft skin measurement with limited views to obtain point cloud data that simultaneously contain the open edges and corner edges.

1) Robustness and efectiveness evaluation of the proposed algorithm: Here, we focus on the performance comparisons to show the robustness and efectiveness of our proposed method using three complex surface models shown in Fig. 6. These models are designed with complex free-form surfaces. In addition, both the bottom and two sides, which indicate the thickness of the model, are removed from the surface model to simultaneously create the structure with closed edges and open edges. These models are measured with fewer views, which improves the eficiency and accuracy to extract the edge information.

Edge Comparison of Performance Evaluation: In this experiment, we evaluate the quality of the extracted edges upon three complex surface models. Given the CAD models, the model shapes consist of two parts: the nonedge points (90k points), which are directly sampled on the surface and the edge points (3k points) sampled from the edge curves. Because few surface sampling points are close enough to the ground-truth edges, we label these points $\| P - P _ { g t } \| ~ < ~ \frac { 1 } { 1 0 } ~ * ~ d _ { r e s }$ as edge points, even if they do not belong to the edge samples. Here, $d _ { r e s }$ is the average point distance of the current point cloud. We conduct experiments to show the performance of the spatial tangent continuity detector with two 3D edge detectors, which are described in the Baselines. Note that for a fair comparison, we only evaluate the original detectors here by disabling the cross-validation and refinement of HBS. Quantitative analysis of the performance results is shown in Fig. 7, using the precision to recall estimation on the three models. As shown in Fig. 7, Figures (a)-(c) reflect the performance of Models 1-3, respectively. It is obvious that our detector of HBS has consistently higher precision and recall, which outperforms the other algorithms. Furthermore, ED represents the curve variation detection using eigenvalues and achieves high precision when the recall is low, but it quickly drops when out of range. This situation worsens, especially as the surfaces become more complex, such as when the precision directly falls when the recall is larger than 0.4 on Models 2 and 3 with complex free-form surfaces. The MS has a stepped performance on all figures, which shows that the points are detected with several common features and easily classified as diferent clusters. The complex shapes on Models 2 and 3 cause more divergence of MS features. Furthermore, as the recall increases, the MS cannot achieve precise edge detection on these complex models.

In addition, we also show a more quantitative comparison that is detailed in Table I. Here, we select one node of the above experiments that follows the max F1-score [24] from each method of model comparison. We employ the F1-score to evaluate the suitable performance of the diferent detectors and conduct a comparison between the proposed HBS and baselines.

TABLE I  
Quantitative Comparison on edges
<table><tr><td>Metrics</td><td>Accuracy(%) Precision(%) Recall(%)</td><td></td><td>[Error(mm)</td></tr><tr><td colspan="4">Model 1</td></tr><tr><td>ED</td><td>98.79 71.36</td><td>67.93</td><td>0.3088</td></tr><tr><td>MS</td><td>99.20 86.08</td><td>73.53</td><td>0.1233</td></tr><tr><td>HBS</td><td>99.92 98.84</td><td>99.17</td><td>0.0072</td></tr><tr><td colspan="4">Model 2</td></tr><tr><td>ED</td><td>98.38 57.88</td><td>39.04</td><td>0.5972</td></tr><tr><td>MS</td><td>98.94 74.07</td><td>74.56</td><td>0.2784</td></tr><tr><td>HBS</td><td>99.87 98.07</td><td>97.17</td><td>0.0105</td></tr><tr><td colspan="4">Model 3</td></tr><tr><td>ED</td><td>98.47</td><td>61.69</td><td>34.80</td><td>0.5203</td></tr><tr><td>MS</td><td>98.96</td><td>74.08</td><td>73.50</td><td>0.2673</td></tr><tr><td>HBS</td><td>99.85</td><td>98.61</td><td>98.19</td><td>0.0098</td></tr></table>

As shown in Table I, it is obvious that our baseline HBS outperforms the other baselines. Notably, the complex shapes are challenging to edge detection methods, which verifies the robustness and efectiveness of our proposed method HBS . The error that is introduced by the complex shapes of Models 2 and 3 increases with {45.83%, 36.11%}, while the increasing ratios are {93.39%, 68.49%} for ED and {125.79%, 116.79%} for MS. Furthermore, the average error of our methodHBS is approximately 0.01 mm, which meets the edge mailing of the accuracy condition in industrial manufacturing for aircraft skins.

![](images/a45b391e6b89322aa1975cbceda10fbbab37912612e98e8e6c9738e6fdc152b3.jpg)  
(a) Sampled point cloud  
(b) ED  
(c) MS  
(d) HBS  
Fig. 8. Error chromatogram of detected edges. The first column shows the sampled point clouds from the three models. The remaining columns show the error chromatogram corresponding to the sampled point clouds for three methods, namely, ED, MS, and HBS. The local details are enlarged on the right of each chromatogram map. These error maps share the same enlarged areas that are indicated with blue arrows.

An overall visual illustration of the edge comparison of the three models is presented in Fig. 8. The first column denotes the three point clouds that are sampled from Models 1-3. Based on this information, the last three columns from left to right denote the error chromatogram of the edge detection methods. For better visualization, we only present the color variation range from 0.00 mm to 1.00 mm and show the error that is out of the range as ’black’. The ’black’ points can be regarded as points that cannot be correctly recognized by the tested edge detection method. As shown in Fig. 8(b), the edges with large eigenvalue variation are correctly detected. Most points of these edges are green, which denotes a low deviation. However, the open edges that do not have eigenvalue variation are missed by the ED method. Another line of the MS detector is shown in Fig. 8(c). The open edges are detected with low errors. However, the closed edges show a high error that even influences the performance of neared open edges. These two typical edge detection methods cannot meet the accuracy requirements of aircraft skin milling and assembly. The error chromatograms of edges that are extracted by HBS are shown in Fig. 8(d). In conclusion, these tests show that our proposed method outperforms and achieves fine performance in terms of accuracy, precision, recall, and error evaluation.

Robustness to Noise: This experiment is designed to test the noise resistance ability of the proposed framework.

We add Gaussian noise with standard deviation of 2-20 $\times 1 0 ^ { - 3 } m m$ to each point of the measured workpieces and report our results in Table II. As the standard deviation increases, the radius of the local search field needs to be enlarged to compensate for the uncertainty due to noise and the unavoidable loss of local details in the extracted edges. We note that even with the addition of noise, our HBS method still outperforms ED and MS.

In Table II, we report our results for outlier experiments. We add 2-20% number of outliers to the measured point cloud. The overall results do not change even when 20% outliers are added. This is because our branch pruning efectively removes the outliers with the help of our graph optimization refinement and gives an accurate milling edge for practical workpiece machining.

We also consider non-uniform sampling of the parts and examine the influence of sparse or dense scanning. We ran domly choose ten seed points and take their neighborhood area through a down-sampling or up-sampling process to change the local density of the point cloud. The results are reported in Table II along with the diferent noise levels. We can see the performance of our method is consistent. The x˙ represent the experiments with irregularly sampling, where $x ~ \in ~ \{ 2 , 4 , . . . , 2 0 \}$ . We also give the systematic assumption (3) in Section II to pre-process the point cloud by spatial filtering and ensure the eficiency of parallel computing. The experiments show that the HBS method can deal with the non-uniform sampled points.

2) Ablation Study: In this test, we explore the contributions with diferent module combinations of our proposed method, which consists of the probability detector (PD), cross validation (CV), and graph refinements (GR). These tests are designed to verify the efectiveness of these modules. The testing data are still sampled from the synthesized models, and the nonedge points (90k points) and edge points (3k points) are identical to the configurations above. Furthermore, we also add 1k points of uniform noise around the cubes that cover the whole model to increase the randomness of the ablation study. All tests were repeated ten times, and then the averages were taken as the final results. The accuracy, precision, recall, and error metrics are included for performance evaluation and are shown in Table III. The symbol $^ \prime \checkmark ^ { \prime }$ denotes the module selected in these combinations, and $^ \prime - \prime$ represents not. First, the detection only runs with the probability detector, which shows great performance but still needs to be improved. Hence, we add CV and GR to constitute the combinations on the backbone of the detector. From Line 2 of Table. III, the PD+CV combination includes more edge points and decreases the false negatives to improve the recall ratios and contribute fewer improvements in decreasing error. This strategy indicates that our proposed method makes progress in edge point recognition to guarantee the continuity of edge lines. Diferent from the CV module, GR are designed to prune undesired false positives. The performance of PD+GR is shown on Line 3 of Table. III. This module combination greatly contributes to the precision and error because the GR is the purely trimming process and does not involve new uncertainty.

TABLE II  
Noise Resistance with irregularly sampled
<table><tr><td>Noise  $\overline { { ( \times 1 0 ^ { - 3 } m m ) } }$ </td><td>2</td><td>4</td><td>6</td><td> $\overline { { 8 } }$ </td><td>10</td><td>12</td><td>14</td><td>16</td><td>18</td><td>20</td></tr><tr><td>Error (mm)</td><td>0.0093</td><td>0.0094</td><td>0.0096</td><td>0.0101</td><td>0.0104</td><td>0.0108</td><td>0.0117</td><td>0.0121</td><td>0.0126</td><td>0.0130</td></tr><tr><td>Outlier  $( \times \% \# \mathrm { p t s } )$ </td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td><td>14</td><td>16</td><td>18</td><td>20</td></tr><tr><td>Error (mm)</td><td>0.0092</td><td>0.0093</td><td>0.0092</td><td>0.0092</td><td>0.0092</td><td>0.0094</td><td>0.0095</td><td>0.0094</td><td>0.0095</td><td>0.0093</td></tr></table>

As shown in the last line of Table. III, the HBS $\mathrm { ( P D + C V + G R ) }$ algorithm shows a great performance for edge detection of complex surfaces, which is often applied to aircraft skins. This combination is regarded as the standard paradigm for the robotic edge detection of our aircraft skin workpieces.

Impact of hyper-parameters: We perform ablation study to analyze the efect of hyper-parameters. We use precision and recall as the evaluation metrics. We also report the computational cost to highlight its tradeof with detection accuracy. First, we study the efect of edge probability threshold $E _ { m i n } , E _ { m a x }$ . Table IV shows our results. Higher $E _ { m i n }$ discards too many edge points, making the establishment of the backbone hard. The subsequent extending and pruning processes are unable to recover the complete desired edges due to sparse backbones. Moreover, the large range of possibilities increases the computation time and does not improve the performance. Next, we study the efect of local graph radius $R _ { g } , R _ { l }$ values. We combine diferent integer multiples of the average distance $d _ { \# }$ to show the impact of $R _ { g } , R _ { l }$ tuple. $R _ { g }$ denotes the larger local field of graph optimization and $R _ { l }$ is the value associated with local perturbation pruning. We report our results in Table IV. Setting the values of $R _ { g } .$ , R to 5,3 and ${ 1 0 , 3 \ d _ { \# } }$ , the value of $R _ { g }$ covers the edge refinement, and a larger value increases the confidence of edge bridging between the highly confident edge patches. Setting the values to 10,3 and 10,5 show that $R _ { l }$ determines the precision. This experiment shows the efectiveness of the local perceptive field module. Increasing $R _ { l }$ makes the graph connect more neighborhood points and inspect more local details to prune perturbations. Overall, our ablation study shows that our method is not sensitive to the selection of hyper-parameters for edge detection.

Time complexity: We also perform an empirical study to evaluate the time complexity on diferent point numbers of both three models. The results are reported in Table. V to examine the influence of the point density on our HBS method. As illustrated for the time complexity in Section III-C, the point number of sampling for the measured workpiece results in the diferent ratios of the edge point. The number of point cloud is sampled with $1 \times 1 0 ^ { 4 } , 5 \times 1 0 ^ { 4 } , 1 0 \times 1 0 ^ { 4 } , 5 0 \times 1 0 ^ { 4 }$ ,and $1 0 0 \times 1 0 ^ { 4 }$ respectively. As shown in Table. V, the overall running time increasing from $1 \times 1 0 ^ { 4 } p t s / 0 . 1 3 s$ to $1 0 0 \times 1 0 ^ { 4 } p t s / 0 . 8 2 s$ has shown the efectiveness of the parallel hardware and knn construction accelerating. The errors are also reported as the point number variant. The low point density results in a high error. The HBS needs to enlarge the search radius and cause vague edge classification because the sparse point distribution can not support accurate STC computing. On the contrary, the edge errors achieve fine performance as the point number increases. Note that, simply increasing the point density/number is not always helpful to the desired edge detection, as shown in the last two rows of Table. V. This experiment inspires us that the suitable sample density to reduce the time complexity but not performance in practice.

## B. Edge measurement of aircraft skin workpieces

In this section, we perform the edge measurement with our aircraft skin measurement system, which is introduced in Fig. 9. The whole system mainly consists of three parts: 1) structured light camera, 2) robotic manipulator, 3) the measuring platform. The structured light camera that is equipped at the end of a robotic manipulator measures the aircraft skins on the measuring platform. The manipulator provides diferent measurement positions for the view scanning of the camera. All measured aircraft skins are manufactured with the thickness of $\delta \ : = \ : 5 m m$ . In this experiment, the controlling parameters of the manipulator are given and not considered in this article.

TABLE IV Impact of hyper-parameters  
TABLE III  
Ablation study for diferent module combinations
<table><tr><td>PD</td><td>CV</td><td>GR</td><td>Accuracy(%)</td><td>Precision(%)</td><td>Recall(%)</td><td>Error(mm)</td></tr><tr><td>V</td><td>1</td><td>一</td><td>99.83</td><td>96.22</td><td>96.65</td><td>0.0214</td></tr><tr><td>√</td><td>V</td><td>1</td><td>99.87</td><td>96.57</td><td>98.10</td><td>0.0187</td></tr><tr><td>√</td><td>一</td><td>V</td><td>99.84</td><td>98.31</td><td>97.13</td><td>0.0113</td></tr><tr><td>√</td><td>√</td><td>V</td><td>99.88</td><td>98.51</td><td>98.18</td><td>0.0092</td></tr></table>

<table><tr><td>Free parameters</td><td>Value</td><td>Precision(%) Recall(%)</td><td></td><td>Time(s)</td></tr><tr><td rowspan="6"> $E _ { m i n } , E _ { m a x }$ </td><td>0.8,0.9|</td><td>92.87</td><td>93.56</td><td>0.46</td></tr><tr><td>0.6,0.9</td><td>96.16</td><td>94.78</td><td>0.59</td></tr><tr><td>0.6,0.8|</td><td>98.49</td><td>98.17</td><td>0.72</td></tr><tr><td>0.4,0.8</td><td>98.51</td><td>98.18</td><td>0.82</td></tr><tr><td>0.2,0.8</td><td>98.51</td><td>98.18</td><td>3.58</td></tr><tr><td>0.4,0.6|</td><td>98.51</td><td>98.18</td><td>1.24</td></tr><tr><td rowspan="6"> $R _ { g } , R _ { l } \ ( ^ { * } d _ { \# } )$ </td><td>0.2,0.6</td><td>98.51</td><td>98.18</td><td>3.27</td></tr><tr><td>5,3</td><td>94.71</td><td>91.10</td><td>0.37</td></tr><tr><td>10,3</td><td>96.94</td><td>98.11</td><td>0.69</td></tr><tr><td>10,5</td><td>98.30</td><td>98.07</td><td>0.73</td></tr><tr><td>20,5</td><td>98.51</td><td>98.18</td><td>0.82</td></tr><tr><td>40,10</td><td>98.51</td><td>98.18</td><td>3.73</td></tr></table>

TABLE V

Time cost of our HBS method to diferent point numbers
<table><tr><td>Point Number(#pts)</td><td>Running Time(s)</td><td>Error(mm)</td></tr><tr><td> $1 \times 1 0 ^ { 4 }$ </td><td>0.13</td><td>0.0209</td></tr><tr><td> $5 \times 1 0 ^ { 4 }$ </td><td>0.22</td><td>0.0099</td></tr><tr><td> $1 0 \times 1 0 ^ { 4 }$ </td><td>0.28</td><td>0.0093</td></tr><tr><td> $5 0 \times 1 0 ^ { 4 }$ </td><td>0.51</td><td>0.0092</td></tr><tr><td> $1 0 0 \times 1 0 ^ { 4 }$ </td><td>0.82</td><td>0.0092</td></tr></table>

![](images/93de83517a484457e2128250c556701dfb1e804b7e6f2f23e4e4c0e6fd397f88.jpg)  
Fig. 9. Robotic measurement system.

In this experiment, we perform the edge measurements of four aircraft skin workpieces that consist of two parts: the measurement of the standard workpieces via the detected edges and edges of nonstandard workpieces visualization.

1) Standard workpieces measurement: Here, we apply our proposed HBS method to measure the parameters via two standard workpieces. These two workpieces (skin 1, 2) are designed as a standard circle with a radius of 0.25m (skin a) and a square with a size of $0 . 5 m { \times } 0 . 5 m$ (skin b). The thickness is designed as 5 mm to approach the size of aircraft skins. The point clouds of the skins are scanned with our robotic measurement system, and the path of the scanning manipulator is pregiven.

![](images/c9fc8bfb549af690fb655347fc7c1885b651a7ffbe1b0779cd5a3d2538942041.jpg)  
Fig. 10. Edge Detection using HBS on Skin Models.

We apply the coordinate average distance error, distance $d _ { z }$ along the z-axis that indicating flatness, and standard deviation Std as the evaluation metric. The point sets are manually aligned to the standard parameters. These performance evaluation details are shown in Table VI. Here, the comparisons are presented in this test. Because the EV cannot simultaneously detect the open edge and closed edge, we increase the scanning times (abbreviated as S.t.) to include the surface thickness and ensure that all desired milling edges are scanned as closed edges. As shown in the circle comparison of Table VI, the EV method demands nearly twice the other methods to approach the desired edges but has greater performance than the MS method. It can be concluded that the curvature and eigenvalue of all upper edges are similar in circle skins. Unlike the performance in Section. IV-A, the performance of the MS method is limited by the multiple edge types that were introduced during scanning. Our HBS outperforms the baselines and achieves fine milling edge detection. In the square comparison of Table VI, there is edge variation, such as the corners and lines. EV shows a significant error increase but still outperforms the MS method due to the greater number of scans. In contrast, MS and HBS still require few scans. HBS is indicated to be suitable for edge detection for automatic machining. Diferent from the current edge detection methods, HBS is designed to detect the upper milling edges and is experimentally shown to be superior.

TABLE VI  
Performance comparison on skins
<table><tr><td>Circle</td><td>Error(mm)</td><td> $d _ { z } ( m m )$ </td><td>Std(mm)</td><td>S.t.</td></tr><tr><td>EV</td><td>0.1124</td><td>0.0678</td><td>0.1059</td><td>23 times</td></tr><tr><td>MS</td><td>0.4912</td><td>2.7123</td><td>0.4219</td><td>11 times</td></tr><tr><td>HBS</td><td>0.0112</td><td>0.0096</td><td>0.0173</td><td>11 times</td></tr><tr><td>Square</td><td>Error(mm)</td><td>dz(mm)</td><td>Std(mm)</td><td>S.t.</td></tr><tr><td>EV</td><td>0.2015</td><td>0.0893</td><td>0.2876</td><td>25 times</td></tr><tr><td>MS</td><td>0.5789</td><td>3.2310</td><td>0.5134</td><td>13 times</td></tr><tr><td>HBS</td><td>0.0104</td><td>0.0086</td><td>0.0141</td><td>13 times</td></tr></table>

2) Visualization of Measured Workpieces: Here, we apply our HBS method to detect the edge on two nonstandard aircraft skins and then visualize the detected edges, including open and closed edges, which are detailed in Fig. 10 to show the efectiveness of our method. The overall illustration shows that HBS is suitable for performing edge detection for further milling allowance tasks.

3) Extended Applications: We show the generalization ability of our HBS method and test it in various extended applications as shown in Fig. 11, including blade profile inspection, hole measurement, and assembly gap detection. Given the CAD models for measurement, we simulate the view scan process using Blender Sensor Simulation (BlenSor [25]). We report the measurement results using HBS in Table VII.

1) Profile inspection: We sample the profile of the blades that can be used as the quality inspection for aviation manufacturing. The edge of the profile is extracted by HBS and then evaluated with the metric, same as the skin performance comparison. Furthermore, we evaluate the flatness $d _ { z }$ of profile surface using the detected edge. The details of the profile edge are reported in Table VII. The overall performance meets the condition of profile measurement.

2) Hole diameter measurement can be used in many diversified scenarios, such as drilling and riveting, and surface hole measurement. We first apply our view sample simulation to get the surface point cloud around the target holes. Note that for multiple circle detection, the linear degree θ is increased to allow distortions of circle edge and hence diferent to our skin milling edge detection. θ = 1◦here. The detailed diameter computing is the same as our circle milling edge detection in Table. VI. The measured average diameter is 1.5014 which achieves an accurate hole diameter measurement compared to the groundtruth (GT) 1.5000.

3) Assembly gap detection is also an important application using edge detection. Assembly quality can be computed by the detected edges of the adjacent workpieces. We get point clouds from the assembly position to get the two edges from the adjacent car parts respectively. The distance to the nearest neighbor point (1-NN) searched from another edge is regarded as the coarse gap and the average gap is reported in Table VII. We use the same metrics as the former edge point evaluation. This example of car assembly gap measurement shows that the edge detection technique is an essential process for automatic measurement.

![](images/48998b9b5a73cfdf3725cf2379536cb9b2a414eac1659f9ca7056dc3fc7af1c2.jpg)  
Fig. 11. Edge detection in diversified scenarios using HBS.

TABLE VII  
Three Extended Applications using HBS
<table><tr><td>Metric</td><td>Error(mm)Std(mm)</td><td>Flatness  $\overline { { d _ { z } ( m m ) } }$ </td></tr><tr><td>Profile inspection</td><td>0.0013 0.0016</td><td>0.0009</td></tr><tr><td>Metric</td><td></td><td>Error(mm)Std(mm)GT/Avg.Diameter(mm)</td></tr><tr><td>Hole diameter measurement</td><td>0.0016 0.0007</td><td>1.5000/1.5014</td></tr><tr><td>Metric</td><td>Error(mm)Std(mm)</td><td>1-NN Gap(mm)</td></tr><tr><td>Assembly gap detection</td><td>0.0011 0.0014</td><td>4.0010</td></tr></table>

## V. Discussion

The current edge milling scheme is only the tip of the iceberg for industrial measurement. It is crucial to fuse diferent requirements like skin machining and planning to integrate our proposed techniques into the whole automatic industrial manufacturing. Our method has several limitations. First, it is not designed for the separate detection of feature lines, so the detailed measurement would be miss out even if the edges have been welldetected. In the future, we plan to investigate how to enhance the measurement framework and the edge graph for more detailed measurements of aircraft skins, such as size inspection and quality evaluation. Besides, with insuficient points, the measurement framework needs to enlarge the neighbor searching radius and hence ignore some key points. Moreover, the current implementation considers point patches with a fixed radius of the percep tive field instead of adaptive structures with varying scales. We plan to design an adaptive structures and employ the novel theories to achieve more accurate aircraft skin measurement in our future works.

## VI. Conclusion

In this paper, we proposed a novel edge detection method for the edge machining of aircraft skins, which was introduced for accurate upper edge extraction with the scanning of complex skin structures. Our proposed method, which consists of the initial backbone establishment, edge extension, and refinement modules, accurately realizes milling edge detection. The detected upper edge is employed in automatic aircraft skin assembly and aviation manufacturing. According to the comparisons between our proposed method and other baselines, our approach performed well and presented a novel paradigm for the milling edge detection task.

## References

[1] G. Wang, W.-l. Li, C. Jiang, and H. Ding, “Machining allowance calculation for robotic edge milling an aircraft skin considering the deformation of assembly process,” IEEE/ASME Transactions on Mechatronics, pp. 1–12, 2022.

[2] X. Liu, W. Chen, H. Madhusudanan, J. Ge, C. Ru, and Y. Sun, “Optical measurement of highly reflective surfaces from a single exposure,” IEEE Transactions on Industrial Informatics, vol. 17, no. 3, pp. 1882–1891, 2021.

[3] M. Feng, L. Zhang, X. Lin, S. Z. Gilani, and A. Mian, “Point attention network for semantic segmentation of 3d point clouds,” Pattern Recognition, vol. 107, p. 107446, 2020.

[4] J. Canny, “A computational approach to edge detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-8, no. 6, pp. 679–698, 1986.

[5] R. Tian, G. Sun, X. Liu, and B. Zheng, “Sobel edge detection based on weighted nuclear norm minimization image denoising,” Electronics, vol. 10, no. 6, p. 655, 2021.

[6] I. Bouganssa, M. Sbihi, and M. Zaim, “Laplacian edge detection algorithm for road signal images and fpga implementation,” International Journal of Machine Learning and Computing, vol. 9, no. 1, pp. 57–61, 2019.

[7] X. Zhou, Y. Wang, Q. Zhu, J. Mao, C. Xiao, X. Lu, and H. Zhang, “A surface defect detection framework for glass bottle bottom using visual attention model and wavelet transform,” IEEE Transactions on Industrial Informatics, vol. 16, no. 4, pp. 2189–2201, 2020.

[8] W. Liu, Y. Zhang, and X. Yu, “A novel subpixel industrial chip detection method based on the dual-edge model for surface mount equipment,” IEEE Transactions on Industrial Informatics, pp. 1–1, 2022.

[9] Y.-D. Zhang, Y. Jiang, W. Zhu, S. Lu, and G. Zhao, “Exploring a smart pathological brain detection method on pseudo zernike moment,” Multimedia Tools and Applications, vol. 77, no. 17, pp. 22 589–22 604, 2018.

[10] S. Xiong, X. Wu, H. Chen, L. Qing, T. Chen, and X. He, “Bidirectional skip connection feature pyramid network and subpixel convolution for high-quality object detection,” Neurocomputing, vol. 440, pp. 185–196, 2021.

[11] H. Zhang, C. Meng, X. Bai, and Z. Li, “Rock-ring detection accuracy improvement in infrared satellite image with sub-pixel edge detection,” IET Image Processing, vol. 13, no. 5, pp. 729– 735, 2019.

[12] X. Xi, Y. Wan, and C. Wang, “Building boundaries extraction from points cloud using an image edge detection method,” in 2016 IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2016, pp. 1270–1273.

[13] T. Bao, J. Zhao, and M. Xu, “Step edge detection method for 3d point clouds based on 2d range images,” Optik, vol. 126, no. 20, pp. 2706–2710, 2015.

[14] B. Steder, R. B. Rusu, K. Konolige, and W. Burgard, “Point feature extraction on 3d range scans taking into account object boundaries,” in 2011 IEEE International Conference on Robotics and Automation. IEEE, 2011, pp. 2601–2608.

[15] S. M. Ahmed, Y. Z. Tan, C. M. Chew, A. A. Mamun, and F. S. Wong, “Edge and corner detection for unorganized 3d point clouds with application to robotic welding,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 7350–7355.

[16] D. Bazazian, J. R. Casas, and J. Ruiz-Hidalgo, “Fast and robust edge extraction in unorganized point clouds,” in 2015 International Conference on Digital Image Computing: Techniques and Applications (DICTA), 2015, pp. 1–8.

[17] K. Demarsin, D. Vanderstraeten, T. Volodine, and D. Roose, “Detection of closed sharp edges in point clouds using normal estimation and graph theory,” Computer-Aided Design, vol. 39, no. 4, pp. 276–283, 2007.

[18] N. Ganganath, C.-T. Cheng, T. Fernando, H. H. C. Iu, and C. K. Tse, “Shortest path planning for energy-constrained mobile platforms navigating on uneven terrains,” IEEE Transactions on Industrial Informatics, vol. 14, no. 9, pp. 4264–4272, 2018.

[19] G. Wang, W. Li, C. Jiang, D. Zhu, Z. Li, W. Xu, H. Zhao, and H. Ding, “Trajectory planning and optimization for robotic machining based on measured point cloud,” IEEE Transactions on Robotics, pp. 1–17, 2021.

[20] Y. Chen, S. Tang, N. Bouguila, C. Wang, J. Du, and H. Li, “A fast clustering algorithm based on pruning unnecessary distance computations in dbscan for high-dimensional data,” Pattern Recognition, vol. 83, pp. 375–387, 2018.

[21] W.-C. Tu, S. He, Q. Yang, and S.-Y. Chien, “Real-time salient object detection with a minimum spanning tree,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 2334–2342.

[22] S. Broumi, A. Bakal, M. Talea, F. Smarandache, and L. Vladareanu, “Applying dijkstra algorithm for solving neutrosophic shortest path problem,” in 2016 International conference on advanced mechatronic systems (ICAMechS). IEEE, 2016, pp. 412–416.

[23] B. Juba and H. S. Le, “Precision-recall versus accuracy and the role of large data sets,” in Proceedings of the AAAI conference on artificial intelligence, vol. 33, no. 01, 2019, pp. 4039–4048.

[24] C. J. Van Rijsbergen, “Information retrieval. 2nd. newton, ma,” 1979.

[25] M. Gschwandtner, Support framework for obstacle detection on autonomous trains. Dissertation, University of Salzburg, 2013.