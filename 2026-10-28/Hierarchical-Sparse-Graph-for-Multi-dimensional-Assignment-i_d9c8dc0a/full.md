# Hierarchical Sparse Graph for Multi-dimensional Assignment in Event-based Tracking

Muxi Zha, Yang Shang, Wenlong Zhang, Qifeng Yu, and Banglei Guan

Abstract—Existing event-based tracking methods typically operate in an online manner, associating tracklets across adjacent batches. However, this limited temporal window often leads to failures in challenging scenarios, such as long-term occlusions or missed detections, where broader temporal context should be incorporated for high-precision tracking. To achieve robust and accurate tracking, this paper concentrates on the event-based multi-dimensional assignment that associates tracklets across multiple batches simultaneously. Unlike image-based detectors, event-based detectors can generate multiple correct yet spatially distinct tracklets for a single object per batch. This violates the one-to-one rule in classic data association, making event-based association an underdetermined many-to-many matching problem. To the best of our knowledge, this is the first work to tackle this NP-hard problem. To address its underdetermined nature, we formulate the assignment problem via fuzzy logic, where pairwise tracklet correlations are quantified through multiple long-range kinematic metrics. This fuzzy design eliminates the need for hard tracking thresholds and enables multiple heterogeneous metrics to be integrated into a unified correlation. By treating tracklets as nodes and their fuzzy correlations as edges, we construct a Hierarchical Sparse Graph (HSG). Building upon the HSG, we pose assignment as a graph partition task, and employ a heuristic solution that measures vertex affinity for partitioning adaptively. The entire pipeline is training-free. Comprehensive experiments on two challenging event datasets, eTram and Ev-UAV, show that our method outperforms existing event-based and most imagebased association models. Particularly, it achieves over 2× the performance of OC-SORT across all metrics in the many-to-many association scenario.

Index Terms—Data association, event-based vision, multiple object tracking.

## I. INTRODUCTION

D <sup>ATA</sup> <sup>assignment</sup> <sup>is</sup> <sup>a</sup> <sup>critical</sup> <sup>step</sup> <sup>in</sup> <sup>Multiple</sup> <sup>Object</sup>Tracking (MOT) [1]–[4], especially in the tracking-by- Tracking (MOT) [1]-[4], especially in the tracking-bydetection paradigm. It is a classic combinatorial optimization problem aimed at identifying matching pairs among detections (or tracklets). Depending on the number of frames considered, image-based data association can be roughly categorized into two types: linear assignment for online tracking and multidimensional assignment for offline tracking. Although linear assignment methods [5]–[7] have shown strong performance, certain challenges, such as long-term occlusions or missed detections [8], [9], can lead to a performance drop. To address this problem, a few works [10], [11] incorporate multiple frames into the association process. These approaches are termed multi-dimensional assignment.

![](images/d28f17f4885656677423828d957fa72fba89b7a478c0502ad9bd482eacc05a4a.jpg)  
(c)  
Fig. 1. Illustration of event-based detections and tracklet association. (a) Detection comparison between ground-truth (GT) and the state-of-the-art (SOTA) event-based detector RVT [12] on eTram [13]. RVT produces detection boxes at a fixed rate, emulating a frame-like output. (b) 2D and 3D visualizations of tracklets within a 300 ms interval. Distinct colors represent different tracklets. These tracklets are obtained by the SOTA event-based segmentation network EV-SpSegNet [14], followed by region growing to cluster spatially connected events for association. (c) Comparisons of two types of event-based data association: image-like one-to-one association, as shown in (a), and eventbased many-to-many association, as shown in (b).

Event cameras are a notable class of bio-inspired sensors that work differently from conventional image-based cameras. Rather than outputting images at a fixed rate, they asynchronously capture per-pixel brightness changes as a stream of binary events. This unique working principle offers distinctive properties compared to image-based cameras, including high temporal resolution (1 µs), high dynamic range (>120 dB), and high sensitivity to dynamic stimuli [15], [16]. These characteristics make event cameras particularly suitable for object tracking [17]–[19] in challenging conditions, like fast motion, extreme illumination, or low-texture targets, where image-based cameras often fail. In such scenarios, achieving high tracking accuracy and robust performance becomes the priority. This motivates a rethinking of the prevailing eventbased efficiency-oriented tracking design and calls for more accurate event-based tracking approaches.

Event-based association is a critical tracking component but has received limited attention. In this paper, we study eventbased multi-dimensional assignment to improve the overall tracking performance. As illustrated in Fig. 1(a) and (b), upstream detections lead to two distinct association scenarios. The former resembles image-based association, where each object corresponds to exactly one detection per batch. We refer to this as one-to-one association. But for the other scenario, this one-to-one characteristic does not hold since asynchronous events can yield multiple spatially distinct yet valid detections for a single object. While image-based detectors can also produce multiple detections per object, this matching ambiguity can be eliminated via Non-Maximum Suppression (NMS) or other strategies. In event-based vision, multiple detections for an object can all be correct and cannot be suppressed, as they may be different temporal segments of the event trajectory. In this case, event-based association becomes an underdetermined many-to-many matching problem. As the association period increases, the processing difficulty grows substantially.

Unlike the one-to-one bipartite matching with binary outcomes, event-based association is an inherently fuzzy problem due to the non-unique correspondence between objects and tracklets. Therefore, we construct a fuzzy modeling pipeline that takes two motion-based metrics as input and yields a fuzzy correlation value in [0, 1] as output. Since no explicit mathematical mapping exists between these inputs and the output, we develop a hierarchical quantization method to bridge them, and subsequently propose a heuristic solution to seek the optimal assignment. The overall framework is shown in Fig. 2.

The assignment problem is underdetermined, so we design a heuristic search-merge approach to find the optimal solution, which iteratively groups tracklets into cohesive partitions of HSG. Each partition corresponds to a set of tracklets of the same object. Central to this approach is a connection similarity measure, which leverages topological and weight information to assess the vertex affinity for merging. As this measure depends on multiple factors, manually setting a merging threshold is impractical. We derive an adaptive threshold under a null model to make the whole optimization automatic. Our method is evaluated by comparing with multiple event-based and image-based association methods on eTram [13] and Ev-UAV [14] datasets for the above two association cases. Contributions are summarized as follows:

This is the first work on event-based multi-dimensional assignment. We model the problem via fuzzy logic to integrate multiple kinematic metrics into unified tracklet correlations and construct a HSG for assignment optimization.

• To model unpredictable and non-linear long-range event trajectories, we devise two kinematic metrics, Bezier curve-based position smoothness and direction-´ displacement-based variation consistency. These metrics capture the spatio-temporal coherence of tracklets in positional and directional domains, respectively.

• A hyperparameter-free graph partition solution is proposed. It quantifies vertex affinities and partitions the graph adaptively for optimal assignment.

• Despite being training-free, our method consistently outperforms existing event-based and motion-based methods, while remaining competitive with leading imagebased association networks. Notably, it also demonstrates robust performance in tracking tiny objects $( 6 . 8 ~ \times ~ 5 . 4$ pixels) under large spatio-temporal gaps, surpassing OC-SORT [7] by over 2× across all metrics.

The remainder is as follows. Section II reviews related work. Section III presents our fuzzy modeling pipeline and HSG construction, followed by the graph partition solution in Section IV. Experimental results and analysis are reported in Section V, and Section VI concludes the paper.

## II. RELATED WORKS

To date, event-based multi-dimensional assignment remains largely unexplored. Existing event-based tracking methods often associate detections from adjacent batches or continuous timestamps for efficiency. Accordingly, we review both eventbased association methods and relevant image-based multidimensional assignment techniques.

## A. Event-based association

Existing event-based association methods can be broadly grouped into two paradigms, direct association and trackingby-detection. Direct association methods [20]–[22] usually assign individual events to existing clusters, which is a unique paradigm benefiting from asynchronously triggered events. Tracking-by-detection is the classic paradigm in the imagebased tracking domain. Therefore, most existing event-based association methods are derived from the image-based counterparts. These methods typically employ Kalman Filter with the Hungarian algorithm [23]–[26], neighbor searching [27]– [31], or Intersection over Union (IoU)-based matching [32], [33]. For instance, Alzugaray et al. [29] use a multi-hypothesis framework where the alignment score of each hypothesis is updated incrementally to track features in adjacent batches. Hu et al. [30] generate head and tail descriptors from each tracklet, then match head descriptors of the previous batch with tail descriptors of the current batch. Jiang et al. [24] use an offline-trained detector and an online-trained tracker, where a Kalman fusion scheme has been proposed to incorporate measurements from the detector and the tracker for real-time intervention-free operations. Barranco et al. [26] utilize Mean Shift to cluster event tracklets in X-Y-T space and employ Kalman Filter with Hungarian algorithm to associate them in adjacent batches. Chen et al. [32] detect objects by bounding boxes in event frames and use IoU to evaluate association candidates. Despite their diversity in implementation, all these methods operate on adjacent batches, making them unable to associate fragmented tracklets with large spatio-temporal gaps.

![](images/177dbc4ce28a63b07fe7ff5250fd5ee51ddcf704073f9a26b579b93bb8469cfa.jpg)  
Fig. 2. Flowchart of the event-based multi-dimensional assignment optimization. Batches of tracklets from event-based detectors are fed into the fuzzy modeling pipeline. Positional and directional affinities are first computed to evaluate pairwise tracklet relationships. These metrics are integrated via hierarchical and sparse quantization to produce unified correlations, based on which the HSG is constructed. Finally, graph partition is performed on the HSG to identif cohesive partitions as optimal assignments.

## B. Image-based multi-dimensional assignment

A fundamental constraint in image-based association is the one-to-one constraint. It states that each object must correspond to exactly one detection and one detection must belong to only one object. This results in a bijective mapping between detections and objects. In image-based multidimensional assignment, although the increasing number of candidates elevates the complexity of the problem, this constraint renders this NP-hard problem well-posed.

Some works have formulated multi-dimensional assignments as a network flow problem [8], [34]–[37], with nodes as detections and edge weights based on appearance or motion. Butt et al. [10] treat tracking as a binary linear program and apply Lagrangian Relaxation to decompose the assignment into a series of simpler min-cost network flow subproblems. Collins et al. [36] break the problem into several two-frame linear assignments solved with the Hungarian algorithm. While these approaches yield good results, some works explore diverse modeling methods for the multi-dimensional assignment problem. Poschmann et al. [37] avoid explicit data association. They model association as a Gaussian Mixture Model (GMM) within a factor graph. Ochs et al. [11] build a hypergraph and use spectral clustering to obtain trajectories. Roshan et al. [38] formulate data association as a Generalized Minimum Clique Problem (GMCP) on a graph. This method addresses one object’s data association at a time and employs Tabu Search to find the clique that minimizes a global cost combining appearance similarity and motion consistency iteratively. Based on this work, Dehghan et al. [39] model a Generalized Maximum Multi Clique Problem (GMMCP) and use Binary Integer

Programming (BIP) to find K-cliques that maximize the sum of edge weights with dummy nodes to handle occlusions.

While image-based methods work for the one-to-one association, they still fail in the many-to-many association. Besides, these methods typically utilize appearance cues for association, but such information is largely unavailable in events. We therefore rethink the modeling of event-based multi-dimensional assignment and leverage only kinematic features to capture relationships between tracklet pairs.

## III. FUZZY-BASED OPTIMIZATION FORMULATION

This section first presents the optimization objective for event-based multi-dimensional assignment. We then detail our fuzzy modeling pipeline, including the definition and computation of two input metrics, the hierarchical and sparse quantization, and the resulting HSG.

## A. Event-based Multi-dimensional Assignment Optimization

We formulate event-based multi-dimensional assignment as a graph-based partitioning problem. Let G be the constructed HSG, where nodes are tracklets and edge weights are their correlations. The goal is to obtain a partition $C = \{ c _ { i } \} _ { i = 1 } ^ { K }$ of G that maximizes the sum of intra-partition weights $\varphi _ { c }$ while minimizing inter-partition weights $\psi ( c _ { i } , c _ { j } )$ . Main difficulties of the optimization lie in the unknown number of partitions $K .$ which denotes the number of targets, and the non-deterministic relationship between K and the number of tracklets.

$$
\begin{array} { r l } { \underset { C } { a r g m a x } } & { \displaystyle \sum _ { c \in C } \varphi _ { c } - \sum _ { c _ { i } , c _ { j } \in C , i \neq j } \psi ( c _ { i } , c _ { j } ) , } \\ { \mathrm { s . t . } } & { \displaystyle \bigcup _ { c \in C } c = G . } \end{array}\tag{1}
$$

![](images/87fa40ad33238c17be0031e9f60ad63c1e26028ca84e77cf796530d3949fb518.jpg)  
Fig. 3. Comparison of quadratic and cubic Bezier curve fitting. Trackets from´ (a) one object and (b) different objects.

## B. Bezier curve-based Position Smoothness´

Straight line motion is a common and effective assumption in event-based vision [23], [40], but it only holds over short time intervals. Over long durations, object trajectories often exhibit complex patterns due to unpredictable acceleration. To effectively associate long-range tracklets, we additionally introduce a position smoothness metric based on Bezier curves.´

Bezier curve [41], [42] is a flexible polynomial representa-´ tion capable of expressing complex curves with relatively few control points. The expressive power of a Bezier curve depends´ on its order, i.e., the number of control points. Higher-order curves can fit complex trajectories more accurately. However, they do not always improve matching reliability, since they tend to overfit the data with overly intricate curvature. A comparison between the quadratic and cubic Bezier curve´ fitting, as shown in Fig. 3, illustrates this problem. For quantitative results, please refer to Sec. V-F. Though the cubic Bezier curve fits trajectory as accurately as the quadratic one,´ it also fits incorrect matching pairs well. This overfitting risk is further amplified by the high temporal resolution of event cameras, which provide finer-grained temporal units for curve evolution. Consequently, we adopt the quadratic Bezier curve´ as the position fitting polynomial. As in Eq. 2, the curve has three control points, $P _ { 0 } , P _ { 1 }$ , and $P _ { 2 }$ . The independent variable t describes the progression of curve generation.

$$
B ( t ) = ( 1 - t ) ^ { 2 } P _ { 0 } + 2 t ( 1 - t ) P _ { 1 } + t ^ { 2 } P _ { 2 } .\tag{2}
$$

Given two tracklets, $L _ { i }$ and $L _ { j }$ , control points are computed from a combined set of their centerline samples, denoted as $\{ p _ { k } \} _ { k = 1 } ^ { 2 M }$ , where each tracklet provides M samples. These points can be obtained either by extracting centroids at uniform time intervals of tracklets, or by using the point-normal representation of the centerline. Assuming that $p _ { 1 }$ and p<sub>2M</sub> are the temporally earliest and latest samples, the start and end control points are fixed as $P _ { 0 } = p _ { 1 }$ and $P _ { 2 } = p _ { 2 M } . \ P _ { 1 }$ is computed via least squares:

$$
P _ { 1 } = \left( A ^ { T } A \right) ^ { - 1 } A ^ { T } b ,\tag{3}
$$

where,

$$
A = \left[ \begin{array} { c } { 2 t _ { 1 } ( 1 - t _ { 1 } ) } \\ { \vdots } \\ { 2 t _ { 2 M } ( 1 - t _ { 2 M } ) } \end{array} \right] ,\tag{4}
$$

![](images/85d58491e5464e3661f66d35bd8d7e90b35f7a20e5562e0efc69f374e0a17156.jpg)  
Fig. 4. Two association cases with both low $E _ { p } .$ (a) Correct and (b) incorrect matching pairs.

$$
b = \left[ \begin{array} { c } { p _ { 1 } - ( 1 - t _ { 1 } ) ^ { 2 } P _ { 0 } - t _ { 1 } ^ { 2 } P _ { 2 } } \\ { \vdots } \\ { p _ { 2 M } - ( 1 - t _ { 2 M } ) ^ { 2 } P _ { 0 } - t _ { 2 M } ^ { 2 } P _ { 2 } } \end{array} \right] .\tag{5}
$$

t consists of 2M values to fit all elements in $\{ p _ { k } \} _ { k = 1 } ^ { 2 M }$ It naturally corresponds to the timestamps of these samples, reflecting how tracklet positions evolve over time. Bezier´ curve-based position smoothness error is defined as the Mean Square Error (MSE) between the fitted points and samples, as given in Eq. 6.

$$
E _ { p 2 } \triangleq \frac { 1 } { 2 M - 2 } \sum _ { k = 1 } ^ { 2 M } \bigl ( B ( t _ { k } ) - p _ { k } \bigr ) ^ { 2 } .\tag{6}
$$

Unlike standard MSE, its denominator is $2 M - 2$ as fitting errors at endpoints are zero. Combining with straight linebased error $E _ { p 1 }$ , the final position smoothness error $E _ { p }$ is

$$
E _ { p } \triangleq \left\{ \begin{array} { l l } { E _ { p 1 } , } & { \quad E _ { p 1 } \leq \mathrm { m i n } ( w _ { i } , w _ { j } ) } \\ { E _ { p 2 } , } & { \quad E _ { p 1 } > \mathrm { m i n } ( w _ { i } , w _ { j } ) } \end{array} \right. ,\tag{7}
$$

$$
E _ { p 1 } \triangleq \operatorname* { m i n } \big ( \mathcal { D } ( c _ { i } , c _ { j } , \mathbf { d _ { i } } ) , \mathcal { D } ( c _ { j } , c _ { i } , \mathbf { d _ { j } } ) \big ) ,\tag{8}
$$

$$
\mathcal { D } ( c _ { i } , c _ { j } , \mathbf { d _ { i } } ) \triangleq \big | \big | ( \mathbf { c _ { i } } \mathbf { c _ { j } } \times \mathrm { n o r m a l i z e } ( \mathbf { d _ { i } } ) ) \big | \big | _ { 2 } ,\tag{9}
$$

where $w _ { i }$ and $w _ { j }$ are spatial widths of $L _ { i }$ and $L _ { j } .$ , calculated from detection boxes or estimated geometrically from event tracklets. $\mathbf { d _ { i } }$ and $\mathbf { d _ { j } }$ are unit direction vectors of $L _ { i }$ and $L _ { j } .$ obtained similarly to the widths w. $\mathbf { c _ { i } c _ { j } }$ denotes the vector from centroid $c _ { i }$ of $L _ { i }$ to centroid $c _ { j }$ of $\bar { L _ { j } } . \mathcal { D } ( \cdot )$ is the distance function in X-Y-T space. $\| \cdot \| _ { 2 }$ and normalize(·) are L2-norm and L2 normalization, respectively.

## C. Direction-displacement-based Variation Consistency

In the long-range tracklet association, Bezier curve-based´ position smoothness is an effective cue for continuous motion. However, it only serves to exclude tracklet pairs with implausible positional changes. When two tracklets are spatially close but exhibit significantly different velocities, this metric yields a small value, as illustrated in Fig. 4. Similarly, our line-based error considers only positional changes. To further constrain tracklets of the same trajectory, we introduce a second cue for continuous motion: velocity consistency.

Velocity can be decomposed into speed and direction. Since it is difficult to accurately estimate the speed of asynchronous events, we focus on the direction. When two tracklets originate from one object, the variation between their moving directions and positional displacement should be small. We quantify this variation as the average cosine similarity between the direction vectors of tracklets and the displacement vector:

![](images/ad129b1540c4443c1fb2619f61d4de78b1d6f2218a87444dfd58fa54bbef2829.jpg)  
Fig. 5. Workflow of our fuzzy modeling pipeline. Compute two kinematic metrics for each pair of tracklets, and an intermediate FCG is generated for each metric. Since these metrics differ in orders and units, we perform hierarchical and sparse quantization on them to generate a unified correlation metric. Finally, the correlations and tracklets are represented as an HSG for subsequent processing.

![](images/d21dccf44b4fb8d54f54d62ec8e07daeffdf325d6bd4290c44de0940edd28d9e.jpg)  
Fig. 6. Two association cases with direction continuity. (a) Correct matching pairs with big $E _ { \theta }$ , and (b) incorrect matching pairs with small $E _ { \theta }$

$$
E _ { \theta } \triangleq { \frac { \mathbf { d _ { i } } \cdot \operatorname { n o r m a l i z e } ( \mathbf { c _ { i } } \mathbf { c _ { j } } ) + \operatorname { n o r m a l i z e } ( \mathbf { c _ { i } } \mathbf { c _ { j } } ) \cdot \mathbf { d _ { j } } } { 2 } } ,\tag{10}
$$

where $L _ { i }$ occurs earlier than $L _ { j } .$ . Though $E _ { p }$ is insufficient to distinguish correct associations shown in Fig. 4, $E _ { \theta }$ effectively disambiguates them by yielding higher values for correct matches, as illustrated in Fig. 6.

## D. The Computation of Metrics

The computation of two aforementioned metrics starts from temporally proximal candidates to distant ones. By representing tracklets as nodes and metric values as edge weights, we obtain two intermediate fully connected graphs (FCGs), one for each metric, as shown in Fig. 5. However, the high temporal resolution of events introduces two challenges during computation. First, $E _ { p }$ becomes unreliable as temporal intervals increase. When the interval is large, two tracklets act like fixed anchors at the start and end of a long quadratic Bezier´ curve. This always yields a low $E _ { p }$ even for incorrect matching pairs. Second, $E _ { \theta }$ may lose its effectiveness in distinguishing correct associations. Event cameras offer microsecond precision, but a detection batch often spans hundreds of milliseconds. This duration is several orders of magnitude larger than the spatial scale $( \mathbf { e . g . , 1 2 8 0 \times 7 2 0 } )$ . As a result, direction vectors are dominated by the temporal component, making spatial variation negligible. Hence, $E _ { \theta }$ remains large for all candidate pairs and may fail to differentiate correct matches from the rest.

Temporal scaling is a common strategy in event processing, typically implemented as exponential decay [43] to compress the temporal dimension. This strategy also balances the disparity in temporal and spatial scales, thereby helping to mitigate the second issue. However, it is not effective for the first one. Temporal scaling essentially samples the event stream along the t-axis. When a large scaling factor s is applied, tracklets may degrade into points, further amplifying the first problem. To address it, we introduce a temporal weight factor $\lambda > 1$ to $E _ { p }$ to adjust the association reliability with temporal intervals.

## E. Temporal Weight Analysis

λ is introduced to reduce erroneous associations across large temporal gaps when computing positional error $E _ { p } .$ Theoretically, tracklets should be primarily associated with their temporal proximate neighbors. Only in the absence of such neighbors should the association be relaxed to include more temporally distant candidates. This association principle involves two factors, association interval $\Delta t$ and the historical match count α. $\Delta t$ denotes the temporal gap between tracklets, which is an intuitive indicator of association reliability. If two tracklets are temporally distant, they are less likely to be associated. α reflects the number of times a tracklet has been successfully associated in the past. In our framework, $E _ { p }$ computation proceeds from temporally proximal candidate pairs to more distant ones. For a tracklet with a high historical match count, further associations with remaining candidates should be treated with caution. Conversely, when a tracklet has low $\alpha ,$ its association reliability should remain high even as $\Delta t$ increases, allowing it to connect with more candidates and recover from fragmentation.

![](images/7bb23f363c781ece0a3aa2b5a7afe542288bff31407936b957355de6b1a1a78b.jpg)

![](images/13a0f29f19d5bcf582f50b6275a45bd0d6d046462b674cf65d8fca5eb9fdea38.jpg)  
(a)  
(b)  
Fig. 7. Distribution of two metric values. (a) Bezier curve-based position´ smoothness error $E _ { p } ,$ where lower values indicate stronger correlation. (b) Direction-displacement-based variation $E _ { \theta }$ , where higher values correspond to stronger correlation.

Besides, we incorporate object size w to adjust the temporal weight for objects of different scales. In dense or frequently overlapping scenarios, the positional errors of small objects tend to be small, while those of large objects tend to be large even after normalization. This is because small objects (e.g., pedestrians walking side by side) can be extremely close, resulting in negligible positional deviation. In contrast, large objects (e.g., vehicles) maintain larger separations, leading to larger normalized errors. To resolve the association ambiguity in such cases, we incorporate a scale-aware adjustment into the temporal weight $\lambda ,$ assigning higher weights to small objects and lower weights to large ones. The detailed determination of λ and the effectiveness of each factor are reported in our supplementary material.

## F. Hierarchical and Sparse Quantization

Since no deterministic mapping exists between the input metrics and correlations, directly deriving the latter from the former is challenging. Furthermore, these metrics are heterogeneous in range and unit, where $E _ { p } \in [ 0 , \infty )$ is measured in pixels and $E _ { \theta } ~ \in ~ [ - 1 , 1 ]$ is unitless. This heterogeneity complicates their integration into a unified output.

Facing these issues, we construct the transformation between inputs and outputs based on an established principle that stronger correlation corresponds to lower $E _ { p }$ and higher $E _ { \theta }$ . As illustrated in Fig. 7, distributions of $E _ { p }$ and $E _ { \theta }$ reveal clear gaps. These gaps typically arise from inherent correlation strengths or the temporal weight λ. We leverage these gaps to quantify continuous metrics into discrete correlations through hierarchical modeling. Sparse modeling is coupled to prune irrelevant edges, thereby reducing the computational burden.

Before hierarchical modeling, it is crucial to address the dependency of $E _ { p }$ on object scale, since the same $E _ { p }$ value implies different correlation strengths for objects of different scales. To mitigate this effect, we normalize $E _ { p }$ by the object width w. Concurrently, we perform initial sparsification by filtering out pairs if their normalized positional error exceeds the object size $( E _ { p } ^ { \prime } > 1 )$ or direction-displacement variation angle exceeds $\pi / 2$ radians $( E _ { \theta } < 0 )$ . The remaining $E _ { p } ^ { \prime }$ and $E _ { \theta }$ share the same range [0, 1].

After initial sparsification, node i has $j$ association candidates. We denote their positional errors as the set $\mathcal { E } _ { p } ^ { ' i } =$ $\{ E _ { p } ^ { ' i 1 } , \ldots , E _ { p } ^ { ' i j } \}$ and their direction-displacement variations as the set $\mathcal { E } _ { \theta } ^ { i } \stackrel { \cdot } { = } \{ E _ { \theta } ^ { i 1 } , \ldots , E _ { \theta } ^ { i j } \}$ . Hierarchical modeling assigns $N + 1$ levels to these values, where levels 1 to N represent valid correlation levels, and level 0 is reserved for outliers. We employ K-means++ with $N + 1$ initial clusters. The cluster results are then sorted by centroids, descending for $E _ { p }$ values and ascending for $E _ { \theta }$ values, to assign levels from the lowest 0 to $N$ . Our method is not sensitive to clustering configurations, as evidenced by experimental results in the supplementary material. In positional errors $E _ { p } ^ { ' } ,$ there are two fitting models. Since the line model imposes a stronger constraint than the Bezier curve model, we introduce a bias´ $\beta$ to differentiate their contributions in outputs. The final correlation $r ^ { i j } = r ^ { j i }$ between i and $j$ is defined in Eq. 11. It is the sum of directional correlation levels $l _ { 1 }$ (from i to j) and $l _ { 2 }$ (from j to i).

$$
\begin{array} { r l } & { r ^ { i j } = \left\{ l _ { 1 } + l _ { 2 } + \beta , ~ \mathrm { i f } ~ E _ { p } ^ { ' i j } ~ \mathrm { i s ~ l i n e - b a s e d } ~ E _ { p 1 } ^ { ' i j } \right. } \\ & { ~ \left. l _ { 1 } + l _ { 2 } , ~ \mathrm { i f } ~ E _ { p } ^ { ' i j } ~ \mathrm { i s ~ B \in \mathrm { z i e r ~ c u r v e - b a s e d } } ~ E _ { p 2 } ^ { ' i j } \right. } \\ & { ~ l _ { 1 } = \operatorname* { m i n } ( N _ { p } ^ { i j } , N _ { \theta } ^ { i j } ) \in [ 0 , N ] , } \\ & { ~ l _ { 2 } = \operatorname* { m i n } ( N _ { p } ^ { j i } , N _ { \theta } ^ { j i } ) \in [ 0 , N ] , } \end{array}\tag{11}
$$

where $N _ { p } ^ { i j }$ and $N _ { p } ^ { j i }$ denote the levels of $E _ { p } ^ { ' i j }$ and $E _ { p } ^ { ' j i }$ respectively. Similarly, $N _ { \theta } ^ { i j }$ and $N _ { \theta } ^ { j i }$ correspond to the levels of $E _ { \theta } ^ { i j }$ and $E _ { \theta } ^ { j i }$ . The final output HSG is constructed on correlations. Detailed pseudocode of our fuzzy modeling pipeline is provided in Algorithm 1.

## IV. CONNECTION SIMILARITY BASED GRAPH PARTITION

Our optimization problem is an underdetermined combinatorial problem with no closed-form solution. Inspired by community detection in social network analysis [44], [45], we adopt a heuristic search-merge solution. Based on the topological structure and edge weights within HSG, we design a connection similarity with an adaptive threshold to automatically guide the vertex merging for graph partitioning.

## A. Connection Similarity

Connection similarity evaluates the vertex affinity for merging. Consider two merged vertices $u _ { i } = ( V _ { i } , R _ { i } )$ and $u _ { j } =$ $( V _ { j } , R _ { j } )$ , where a vertex consists of a few nodes V and edges R. Their connection similarities are two asymmetric scores $S _ { i } ^ { j }$ and $S _ { j } ^ { i }$ , as defined in $\operatorname { E q } .$ . 12. $S _ { i } ^ { j }$ is the connection similarity from $u _ { j }$ to $u _ { i }$ . It computes the proportion of edge weights shared between $u _ { i }$ and $u _ { j }$ via common neighbors relative to $u _ { i } \mathrm { ^ { * } s }$ total weights. $S _ { j } ^ { i }$ is defined analogously. This measure focuses on connectivity through shared neighbors rather than direct connections. Consequently, it allows merging vertices that share common neighbors even if they lack direct links.

Algorithm 1 Fuzzy Modeling Pipeline   
Require: Tracklet set T, hierarchical levels N.   
1: $G $ EmptyDirectedGraph()   
2: HSG ← EmptyUndirectedGraph()   
/\* Calculate two metrics $^ { * }$   
3: for $i = 1$ to NumberOfNodes(G) do   
4: for $j = i + 1$ to NumberOfNodes(G) do   
5: $\bar { E } _ { p 1 } ^ { i j } , E _ { p 2 } ^ { i j }$ ← PosSmoothError $[ T _ { i } , T _ { j } )$ using Eq. 7   
6: $\dot { E _ { p 1 } ^ { i j } } , \dot { E _ { p 2 } ^ { i j } }  E _ { p 1 } ^ { i j } \cdot \lambda _ { 1 } , E _ { p 2 } ^ { i j } \cdot \lambda _ { 2 }$   
7: $E _ { \theta } ^ { i j } \gets$ DirectDisplaceVariation ${ _ 1 } ( T _ { i } , T _ { j } )$ using Eq. 10   
$/ * ^ { \mathrm { ~ \biggr ~ } }$ Hierarchical & sparse quantization: initial sparsification   
$^ { * }$   
8: if $E _ { p 1 } ^ { i j } < \operatorname* { m i n } ( w _ { i } , w _ { j } )$ then   
9: $\bar { G }  \mathrm { A d d E d g e } ( i , j , E _ { p 1 } ^ { i j } / w _ { i } , E _ { \theta } ^ { i j } )$   
10: $G \gets \mathrm { A d d E d g e } ( j , i , E _ { p 1 } ^ { \dot { i } j } / w _ { j } , E _ { \theta } ^ { i j } )$   
11: else if $E _ { p 2 } ^ { i j } <$ min $( w _ { i } , w _ { j } )$ then   
12: $G \gets \dot { \mathrm { A d d E d g e } } ( i , j , E _ { p 2 } ^ { i j } / w _ { i } , E _ { \theta } ^ { i j } )$   
13: $G \gets \mathrm { A d d E d g e } ( j , i , E _ { p 2 } ^ { \dot { i } j } / w _ { j } , E _ { \theta } ^ { i j } )$   
14: end if   
15: end for   
16: end for   
/\* Hierarchical & sparse quantization: hierarchical & sparse   
modeling \*/   
17: for i in Nodes $( G )$ do   
18: $\mathcal { E } _ { p } ^ { ' i } , \mathcal { E } _ { \theta } ^ { i } \gets \mathrm { G r a p h W e i g h t } ( G , ~ i )$   
19: $N _ { p } ^ { i } , N _ { \theta } ^ { i } \ \cdot$ ← HierarchicalSparseMode $( \mathcal { E } _ { p } ^ { ' i } , \mathcal { E } _ { \theta } ^ { i } , N + 1 )$   
20: $\dot { H S G }  \mathrm { A d d E d g e } ( i , N _ { p } ^ { i } , N _ { \theta } ^ { i } , \beta )$ using Eq. 11   
21: end for   
22: return HSG

$$
\begin{array} { r }  \left\{ \begin{array} { l l } { \displaystyle \chi _ { i j } = \sum _ { \tiny \begin{array} { c } { o \in V _ { i } , p \in V _ { j } } \\ { k \in \mathrm { N e i g h b o r } ( V _ { i } ) \prod \mathrm { N e i g h b o r } ( V _ { j } ) } \\ { S _ { i } ^ { j } = \frac { \displaystyle \chi _ { i j } } { \sum _ { \tiny \begin{array} { c } { o \in V _ { i } } \\ { k \in \mathrm { N e i g h b o r } ( V _ { i } ) } \end{array} } } = \frac { \displaystyle \chi _ { i j } } { \displaystyle \chi _ { i } } \in [ 0 , 1 ] } \\ { S _ { j } ^ { i } = \frac { \displaystyle \chi _ { i j } } { \sum _ { \tiny \begin{array} { c } { h \in \mathrm { N e i g h b o r } ( V _ { i } ) } \\ { k \in \mathrm { N e i g h b o r } ( V _ { j } ) } \end{array} } \displaystyle \qquad \chi _ { j } ^ { n } \in [ 0 , 1 ] } \end{array} } } \end{array} \right. , } \end{array}\tag{12}
$$

where Neighbor(·) denotes the neighboring node set, $\chi _ { i j }$ is the total edge weight shared between $u _ { i }$ and $u _ { j }$ via common neighbors, and $S _ { i }$ and $S _ { j }$ are the total edge weights of $u _ { i }$ and $u _ { j }$ respectively.

## B. Threshold for Connection Similarity under a Null Model

Connection similarity depends on multiple factors, such as topological structures and edge weights, making it impractical to set a reliable threshold manually for merging decisions. Here, we provide a threshold for automatic decisions. Inspired by the null model used in modularity [46], we derive an adaptive threshold τ . This threshold approximates the expected connection similarity in a randomly connected graph, and serves as the lower bound for our vertex merging. Specifically, $u _ { i }$ and $u _ { j }$ are merged if $S _ { i } ^ { j } \ge \tau _ { i } ~ \mathrm { o r } ~ S _ { j } ^ { i } \ge \tau _ { j }$

To derive $\tau _ { i } ,$ we first estimate the expected common edges shared between $u _ { i }$ and $u _ { j }$ . For vertex $u _ { i } ,$ let $n _ { v } ^ { i }$ denote its number of nodes, $n _ { r } ^ { i }$ the number of intra-vertex edges, $\zeta _ { i }$ the average number of edges per node, and $r _ { i }$ the average edge weight. The internal graph density of $u _ { i }$ is $D _ { i } = 2 n _ { r } ^ { i } \big / n _ { v } ^ { i } ( n _ { v } ^ { i } - 1 )$

and similarly for $u _ { j } , D _ { j } = 2 n _ { r } ^ { j } \big / n _ { v } ^ { j } ( n _ { v } ^ { j } - 1 )$ . Suppose $u _ { i }$ has m merging candidates that are similar to $u _ { j } .$ , and the graph density of the null model consisting of $u _ { i }$ together with m candidates is estimated as

$$
D _ { i } ^ { j } = \frac { n _ { r } ^ { i } + m n _ { r } ^ { j } } { \left( n _ { v } ^ { i } + m n _ { v } ^ { j } \right) \cdot \left( n _ { v } ^ { i } + m n _ { v } ^ { j } - 1 \right) } .\tag{13}
$$

The unknown m is estimated based on the nodes in $u _ { j }$ and the neighbors of $u _ { i } .$ . Let $\eta _ { i }$ be the number of neighboring nodes of $u _ { i }$ . Under a null model, we employ $n _ { v } ^ { j } / \eta _ { i }$ to approximate the probability that $u _ { j }$ appears among the neighboring vertices of $u _ { i } .$ The expected number of such vertices $m = \eta _ { i } \big / n _ { v } ^ { j }$ is the reciprocal of the probability.

The above expectation is effective when $\begin{array} { r } { \eta _ { i } ~ > ~ n _ { v } ^ { j } } \end{array}$ , i.e., $m \geq 1$ . This constraint arises from the fundamental property of $m$ that the number of mergeable candidates should be larger than 1 if $u _ { i }$ has at least one connection outside itself. However, $\begin{array} { r l } { \eta _ { i } } & { { } < \ , \ n _ { v } ^ { j } } \end{array}$ may occur. If $m \ < \ 1$ , according to Eq. 13 where $D _ { i } ^ { j } \propto { ^ { 1 } } / { m ^ { 2 } }$ , the expected graph density $D _ { i } ^ { j }$ will inflate quadratically as m decreases. This effect is particularly severe when the connectivity within each vertex is relatively dense (i.e., small $\eta _ { i }$ and large $n _ { v } ^ { j } )$ , causing $m \ll 1$ and the merging condition to be overly strict. As a result, vertices from the same partition may fail to merge. We therefore propose an alternative formulation in Eq. 14. It replaces $m$ in the denominator of $\operatorname { E q . }$ . 13 with $m + 1$ , while m in the numerator is retained to preserve the true expected edge connection. In our experiments, we adopt this alternative formulation, and a comparison between Eq. 13 and Eq. 14 is provided in the ablation study.

$$
D _ { i } ^ { j } = \frac { n _ { r } ^ { i } + m n _ { r } ^ { j } } { \left( n _ { v } ^ { i } + n _ { v } ^ { j } + m n _ { v } ^ { j } \right) \cdot \left( n _ { v } ^ { i } + n _ { v } ^ { j } + m n _ { v } ^ { j } - 1 \right) } .\tag{14}
$$

In this null model, the expected number of common edges shared between $u _ { i }$ and a single $u _ { j }$ is $E [ \zeta _ { i } ^ { j } ] = D _ { i } ^ { j } \cdot n _ { v } ^ { i } \cdot n _ { v } ^ { j }$ The threshold $\tau _ { i }$ for merging is

$$
\tau _ { i } = \frac { \gamma _ { i } \cdot E [ \zeta _ { i } ^ { j } ] } { S _ { i } } = \frac { \gamma _ { i } \cdot D _ { i } ^ { j } \cdot n _ { v } ^ { i } \cdot n _ { v } ^ { j } } { S _ { i } } .\tag{15}
$$

where $\gamma _ { i } = \operatorname* { m i n } ( \zeta _ { i } / \zeta _ { j } , 1 )$ is a balancing factor.

Strictly speaking, the numerator of $\tau _ { i }$ represents the expected direct connections, which naturally serves as a lower bound for our connection similarity that captures both direct and indirect connections. However, the null model assumption may be invalid when two vertices exhibit significantly different sizes, as vertices under random connections should exhibit comparable sizes. Therefore, we introduce $\gamma$ in Eq. 15 to mitigate the impact of such size disparities between vertices. An ablation on this factor is also provided in the subsequent experiment.

## C. Two-stage graph partition

Edge weights in the HSG denote correlation strengths, allowing disconnected subgraphs to serve as initial partitions, as shown in Fig. 2. Hence, graph partition is performed independently on each subgraph, which reduces the computational burden compared to performing it on the full graph.

Algorithm 2 Connection Similarity based Graph Partition   
Require: A subgraph $G _ { k } = \{ u _ { i } \} _ { i = 1 } ^ { N _ { k } } = \{ ( V _ { i } , R _ { i } ) \} _ { i = 1 } ^ { N _ { k } }$   
1: while True do   
2: Partitions $C \gets \{ \} , f l a g \gets$ False   
$/ { * }$ Vertex initialization $^ { * }$   
3: $U _ { k } \gets \mathrm { ~ R ~ }$ andomShuffleVertex $\left( G _ { k } \right)$   
4: for $u _ { i }$ in $U _ { k }$ do   
5: $\mathcal { N } \gets \mathrm { N e i g h b o r } ( u _ { i } , G _ { k } )$   
6: for $u _ { j }$ in $\mathcal { N }$ do   
7: Calculate connection similarity $S _ { i } ^ { j }$ and $S _ { j } ^ { i }$ using Eq. 12   
$/ { * }$ Adaptive threshold calculation $^ { * }$   
8: Calculate graph density $D _ { i } ^ { j }$ and $D _ { j } ^ { i }$ using $\operatorname { E q . }$ 14   
9: Calculate threshold $\tau _ { i }$ and $\tau _ { j }$ using Eq. 15   
$/ { * }$ Merge decision $^ { * }$   
10: if $S _ { i } ^ { j } \geq \tau _ { i }$ or $S _ { j } ^ { i } \geq \tau _ { j }$ then   
11: flag ← True   
12: C(u<sub>i</sub>) ← MergeAndLabel $( C ( u _ { i } ) , C ( u _ { j } ) )$   
13: end if   
14: end for   
15: end for   
16: if $f l a g$ is False then   
17: break   
18: end if   
/\* Merge nodes and reconstruct graph $^ { * }$   
19: $G _ { k } , \bar { U _ { k } } \gets$ MergeAndReconstruc $\dot { ( C ) }$   
20: end while   
21: return $C$

This solution operates through an iterative two-stage process of search and merge. Initially, each node is treated as a singleton vertex. In the search stage, connection similarities are evaluated between neighboring vertex pairs. Any pair satisfying the merging criterion is assigned a common label. This continues until all vertices are examined. In the subsequent merge stage, vertices sharing the same label are integrated into a new vertex. A new graph is reconstructed with updated vertex set. The entire procedure repeats until no mergeable vertices remain, and the final vertex set is the optimal partitioning result. A summary of the procedure is provided in Algorithm 2.

## V. EXPERIMENTS

This section evaluates the effectiveness of our method for event-based association. We conduct experiments on eTram [13] and Ev-UAV [14]. eTram is a novel event-based traffic perception dataset for MOT. It is curated using high spatial resolution (1280 × 720) event cameras across challenging lighting conditions, with object sizes ranging from approximately $8 \times 8$ to 200 × 200 pixels. Ev-UAV is the first event-based tiny object detection dataset, which contains tiny targets with an average size of $6 . 8 \times 5 . 4$ pixels and provides event-wise labels. It covers challenging scenarios like highbrightness and low-light conditions.

On eTram, where the association scenario often resembles the image-based one-to-one matching, we compare our method against both event-based and image-based association methods under various detectors. In contrast, the association on the Ev-UAV is many-to-many. We evaluate our method’s ability to solve this underdetermined problem and to achieve robust tracking of tiny objects. Finally, we conduct ablation studies and sensitivity experiments to assess the contribution of key components and the robustness of our method under various parameter configurations. Detailed parameter experiments and analyses are provided in the supplementary material.

## A. Evaluation metrics and model configurations

On eTram, we employ MT (Mostly Tracked), ML (Mostly Lost), MOTA [47] (Multiple Object Tracking Accuracy), and IDF1 (Identification F-Score) [48]. For completeness, we also report the detection F1 score which is related to MOTA. On Ev-UAV, we still report the IDF1, while adopting the evaluation metrics from its original work [14], IoU and Acc (Accuracy). To assess tracking performance, we extend them to identity-aware versions by matching predicted identities with GT identities, referred to as Track IoU and Track Acc.

While our partition solution has no manually set hyperparameters, the fuzzy modeling pipeline has five parameters. They are the number of fitting samples $M ,$ hierarchical modeling levels $N ,$ correlation bias $\beta ,$ temporal scaling factor s and temporal weight factor $\lambda .$ We set $N = 4 , \beta = 1$ , and $s ~ = ~ 0 . 0 0 1$ . M is not an important parameter, as it only affects the fitting accuracy of the quadratic Bezier curve,´ which only requires a minimum of three samples to fit. We set $M = 7$ on eTram and $M = 3$ on Ev-UAV since their batch periods are different. The key temporal weight λ is determined empirically. Specifically, $\lambda _ { p 1 } ~ = ~ ( 1 . 0 0 1 + \alpha ~ \cdot$ $( 0 . 0 7 - 0 . 0 0 8 \log _ { 1 0 } w ) ) ^ { \Delta t }$ for line-based positional errors, and $\lambda _ { p 2 } = ( 1 . 0 0 1 + ( \alpha + 1 )$ $\operatorname { t a n h } ^ { 2 } ( 0 . 0 2 w ) \bar { ) } ^ { \Delta t }$ for Bezier curve-´ based positional errors.

## B. Comparison methods

Event-based association methods. To the best of our knowledge, existing event-based tracking methods operate in an online manner and lack dedicated multi-dimensional assignment methods. To demonstrate the performance improvement enabled by our method, we compare it against three prevailing event-based association methods: (1) Kalman Filter (KF) with Hungarian algorithm (HA) [26], [49]–[51], (2) neighbor searching (NS) [29], [30], [52], [53], and (3) IoUbased matching [32], [33]. Besides these, we also compare with two event-based single object tracking (SOT) models, HDETrack [54] and TSETrack [55].

Image-based association methods. Given the limited number of event-based association methods, we further compare our method with association modules from SOTA image-based MOT trackers. It is important to note that these methods benefit from the one-to-one constraint and often utilize rich appearance features, while our method does not. Despite these differences, we conduct a cross-modality comparison to thoroughly assess our tracking performance. This is done by substituting the association module in image-based trackers with our method. The comparative models are Permatrack [5], UAVMOT [6], OC-SORT [7], LTTrack [9], TOPIC [56], along with a multi-dimensional assignment based model LPT [57]. Since OC-SORT is a purely motion-based method that effectively combines and improves upon KF+HA and IoU-based matching, we adopt it as the primary baseline on Ev-UAV, rather than comparing against its individual components. All models are evaluated under their standard training and testing protocols, and no extra post-processing (such as filtering) is applied to ensure a fair comparison.

TABLE I  
MOT PERFORMANCE OF DIFFERENT EVENT-BASED ASSOCIATIONMETHODS ON ETRAM. THE BEST VALUES ARE HIGHLIGHTED IN BOLD.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>DatasetSplit</td><td rowspan=1 colspan=1>F1(%)</td><td rowspan=1 colspan=1>MT(%)</td><td rowspan=1 colspan=1>ML(%)</td><td rowspan=1 colspan=1>MOTA(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=2 colspan=1>KF+HA</td><td rowspan=2 colspan=1>No</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>99.3</td><td rowspan=1 colspan=1>98.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>3.7</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>99.2</td><td rowspan=1 colspan=1>98.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>2.9</td></tr><tr><td rowspan=2 colspan=1>NS</td><td rowspan=2 colspan=1>No</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>99.3</td><td rowspan=1 colspan=1>99.4</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>57.7</td><td rowspan=1 colspan=1>30.9</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>99.2</td><td rowspan=1 colspan=1>98.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>65.0</td><td rowspan=1 colspan=1>38.8</td></tr><tr><td rowspan=2 colspan=1>IoU</td><td rowspan=2 colspan=1>No</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>99.3</td><td rowspan=1 colspan=1>99.4</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>87.2</td><td rowspan=1 colspan=1>30.7</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>99.2</td><td rowspan=1 colspan=1>98.9</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>90.6</td><td rowspan=1 colspan=1>22.8</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=2 colspan=1>No</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>99.3</td><td rowspan=1 colspan=1>99.4</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>98.5</td><td rowspan=1 colspan=1>95.7</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>99.2</td><td rowspan=1 colspan=1>99.1</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>98.4</td><td rowspan=1 colspan=1>95.3</td></tr></table>

## C. Tracking on eTram

Comparisons with event-based association methods. Experimental results of our method and event-based association methods under GT detections are presented in Table I. Our method achieves the best tracking performance, attaining 98.5 MOTA and 95.7 IDF1 on the validation set and 98.4 MOTA and 95.3 IDF1 on the test set. It surpasses the second-best method by a significant margin (8% in MOTA and 50% in IDF1). Though all methods are provided with GT detections, the reported F1 scores are not 1.0. We further analyze sequences with low F1. As visualized in Fig. 8, some GT boxes contain very few or even zero events, making association infeasible. This leads to a reduction in true positives, which in turn lowers F1 and MOTA.

Comparisons with image-based association methods. We compare our method against SOTA image-based MOT association models and event-based SOT models under five different detectors, three image-based and two event-based. As reported in Table II, our method consistently outperforms association baselines under three detectors, Permatrack, UAVMOT, and RVT. Notably, under the event-based detector RVT, our HSG surpasses OC-SORT with notable tracking performance improvements, especially in MT (>5%), ML (>5%) and IDF1 (>11%).

When evaluated with the YOLOX detector [58], HSG shows slightly lower MOTA (0.6%) and IDF1 (0.5%) than OC-SORT on the validation dataset but achieves a clear lead (>3%) in the other three metrics. We attribute OC-SORT’s slight advantage in MOTA and IDF1 to OC-SORT’s filtering of fragmented tracklets, which reduces false positives in detections and identity switches. Within the same YOLOX setting, HSG also outperforms TOPIC by over 10% but falls slightly behind LTTrack (0.2%-1.4%). When sharing the same detection results as TSETrack, our method consistently outperforms other methods such as OC-SORT and LPT, but shows a marginal performance drop of 0.9% compared to TSETrack on the validation set. It is worth noting that the narrow margins against both LTTrack and TSETrack are within expectations. Given that our method operates without the oneto-one matching constraint, is completely training-free, and relies solely on kinematic cues, these minor performance drops represent a highly reasonable trade-off. Collectively, these results demonstrate the effectiveness of our approach in eventbased association.

![](images/8cf0fa41a25168ed04fc90f94d25b7b0342c26327224feb704fe87d1a898557b.jpg)  
Fig. 8. Visualization of GT tracking annotations on eTram. Sequences (from left to right)are val night 008 and val night 011, showing GT boxes with few events that limit tracking performance during evaluation.

![](images/af137309f28a49e1f11453bcc7f4c72d83d4f66f42a3ecc03f29ac7a0b179e72.jpg)  
Fig. 9. Qualitative comparison of OC-SORT and our HSG on two hard eTram sequences under YOLOX detections. Shown case (a) represents small objects with missed detections, and (b) represents dense object distribution and occlusions. Red dashed ellipses highlight the challenging scenarios and result differences between two methods. Two sequences (from left to right) are val night 001 and val night 005.

Case Study. Fig. 9 and Fig. 10 visualize several challenging tracking scenarios comparing baseline OC-SORT and our HSG under YOLOX and RVT detections, respectively. In Fig. 9, two representative cases are shown, (a) small objects with missed detections, and (b) missed detections in dense distributions. Tracking differences between OC-SORT and our method are highlighted in red dashed ellipses. In case (a), some detections of small objects are temporarily missing in frames 132 and 133. In case (b), some pedestrian detections are also missing in frames 344 and 346. Our method successfully associates fragmented tracklets in both scenarios, while OC-SORT fails and drops them due to insufficient tracking hits over time.

In Fig. 10, we also present two cases, (a) objects partially occluded by background, and (b) dense distribution with object occlusions. In these cases, our method still successfully associates the tracklets, while OC-SORT fails. These results demonstrate that our method achieves higher tracking accuracy than a strong motion-based baseline. Additionally, we observe that in case (a), our method fails to associate a tracklet on the left side of frame 249 where OC-SORT succeeds. This is because the corresponding detection bounding box contains nearly zero events, making our event-based association infeasible. A similar issue appears in frame 311 of case (b) where OC-SORT produces identities for three overlapping detections.

TABLE II  
TRACKING PERFORMANCE: OUR METHOD VS. SOTA IMAGE-BASED TRACKERS UNDER MULTIPLE DETECTORS ON ETRAM. METHODS SHARING THE SAME DETECTION RESULTS ARE GROUPED IN IDENTICALLY COLORED BLOCKS. IN TRACKING TYPE, -L MEANS LINEAR ASSIGNMENT-BASED AND -M MEANS MULTI-DIMENSIONAL ASSIGNMENT-BASED. IN ASSOCIATION FEATURE TYPE, M- MEANS MOTION-BASED, A- MEANS APPEARANCE-BASED, AND POST MEANS POST-PROCESSING. ‘TRAIN’ COLUMN INDICATES WHETHER THE ASSOCIATION METHOD UTILIZES ADDITIONAL CUES BEYOND BOUNDING-BOX COORDINATES FROM A TRAINED NETWORK. REGARDING PERFORMANCE COMPARISONS, PURPLE INDICATES THE POSITIVE PERFORMANCE GAP OF OUR METHOD COMPARED TO THE BASELINE, WHILE DARK GREEN INDICATES A NEGATIVE PERFORMANCE GAP.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Trackingtype</td><td rowspan=1 colspan=1>Associationfeature type</td><td rowspan=1 colspan=1>Associationtechniques</td><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>One-to-oneconstraint</td><td rowspan=1 colspan=1>Year</td><td rowspan=1 colspan=1>DatasetSplit</td><td rowspan=1 colspan=1>F1(%)</td><td rowspan=1 colspan=1>MT(%)</td><td rowspan=1 colspan=1>ML(%)</td><td rowspan=1 colspan=1>MOTA(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=1 colspan=1>Permatrack</td><td rowspan=1 colspan=1>MOT-L</td><td rowspan=1 colspan=1>M-+A-</td><td rowspan=1 colspan=1>HA+ConvGRU [59]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>2021</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>54.747.9</td><td rowspan=1 colspan=1>30.613.4</td><td rowspan=1 colspan=1>18.021.5</td><td rowspan=1 colspan=1>16.517.1</td><td rowspan=1 colspan=1>49.041.6</td></tr><tr><td rowspan=1 colspan=1>HSG</td><td rowspan=1 colspan=1>MOT-M</td><td rowspan=1 colspan=1>M-</td><td rowspan=1 colspan=1>Graph-based optim.</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>54.7(0.0)47.9(0.0)</td><td rowspan=1 colspan=1>30.6(0.0)13.4(0.0)</td><td rowspan=1 colspan=1>18.0(0.0)21.5(0.0)</td><td rowspan=1 colspan=1>16.9(+0.4)17.6(+0.5)</td><td rowspan=1 colspan=1>50.4(+1.4)43.7(+2.1)</td></tr><tr><td rowspan=1 colspan=1>UAVMOT</td><td rowspan=1 colspan=1>MOT-L</td><td rowspan=1 colspan=1>M-+A-</td><td rowspan=1 colspan=1>KF+HA+IoU+NS</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>2022</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>65.668.6</td><td rowspan=1 colspan=1>73.051.9</td><td rowspan=1 colspan=1>2.54.3</td><td rowspan=1 colspan=1>19.834.1</td><td rowspan=1 colspan=1>57.759.4</td></tr><tr><td rowspan=1 colspan=1>HSG</td><td rowspan=1 colspan=1>MOT-L</td><td rowspan=1 colspan=1>M-</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>_</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>65.9(+0.3)68.9(+0.3)</td><td rowspan=1 colspan=1>73.8(+0.8)52.9(+1.0)</td><td rowspan=1 colspan=1>2.4(-0.1)4.3(0.0)</td><td rowspan=1 colspan=1>21.5(+1.7)36.3(+2.2)</td><td rowspan=1 colspan=1>59.1(+1.4)60.7(+1.3)</td></tr><tr><td rowspan=1 colspan=1>YOLOX-X</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>(</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>2023</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>57.251.1</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>–</td></tr><tr><td rowspan=1 colspan=1>OC-SORT</td><td rowspan=1 colspan=1>MOT-L</td><td rowspan=1 colspan=1>M-</td><td rowspan=1 colspan=1>KF+HA+IoU</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>2023</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>55.146.5</td><td rowspan=1 colspan=1>31.29.2</td><td rowspan=1 colspan=1>28.245.3</td><td rowspan=1 colspan=1>27.925.4</td><td rowspan=1 colspan=1>52.944.8</td></tr><tr><td rowspan=3 colspan=1>LTTrack</td><td rowspan=3 colspan=1>MOT-L</td><td rowspan=3 colspan=1>M-+A-M-+A-+Post</td><td rowspan=3 colspan=1>HA+IoU+NS+FastReID [60]</td><td rowspan=3 colspan=1>Yes</td><td rowspan=3 colspan=1>Yes</td><td rowspan=3 colspan=1>2024</td><td rowspan=3 colspan=1>ValTestValTest</td><td rowspan=1 colspan=1>58.753.0</td><td rowspan=1 colspan=1>44.615.4</td><td rowspan=1 colspan=1>23.439.1</td><td rowspan=1 colspan=1>27.529.3</td><td rowspan=1 colspan=1>53.847.6</td></tr><tr><td rowspan=2 colspan=1>59.255.0</td><td rowspan=2 colspan=1>47.718.6</td><td rowspan=1 colspan=1>23.1</td><td rowspan=1 colspan=1>26.3</td><td rowspan=1 colspan=1>54.2</td></tr><tr><td rowspan=1 colspan=1>38.7</td><td rowspan=1 colspan=1>30.5</td><td rowspan=1 colspan=1>49.3</td></tr><tr><td rowspan=3 colspan=1>TOPIC</td><td rowspan=3 colspan=1>MOT-L</td><td rowspan=3 colspan=1>M-+A-M-+A-+Post</td><td rowspan=3 colspan=1>OC-SORT+FastReID</td><td rowspan=3 colspan=1>Yes</td><td rowspan=3 colspan=1>Yes</td><td rowspan=3 colspan=1>2025</td><td rowspan=3 colspan=1>ValTestValTest</td><td rowspan=1 colspan=1>48.237.7</td><td rowspan=1 colspan=1>10.51.9</td><td rowspan=1 colspan=1>28.648.8</td><td rowspan=1 colspan=1>17.413.5</td><td rowspan=1 colspan=1>20.815.7</td></tr><tr><td rowspan=1 colspan=1>48.9</td><td rowspan=1 colspan=1>21.7</td><td rowspan=1 colspan=1>26.8</td><td rowspan=1 colspan=1>8.3</td><td rowspan=1 colspan=1>22.2</td></tr><tr><td rowspan=1 colspan=1>40.2</td><td rowspan=1 colspan=1>4.9</td><td rowspan=1 colspan=1>45.2</td><td rowspan=1 colspan=1>9.2</td><td rowspan=1 colspan=1>18.0</td></tr><tr><td rowspan=2 colspan=1>LPT</td><td rowspan=2 colspan=1>MOT-M</td><td rowspan=2 colspan=1>M-+A-</td><td rowspan=2 colspan=1>Graph-based optim.</td><td rowspan=2 colspan=1>Yes</td><td rowspan=2 colspan=1>Yes</td><td rowspan=2 colspan=1>2022</td><td rowspan=2 colspan=1>ValTest</td><td rowspan=2 colspan=1>56.852.3</td><td rowspan=2 colspan=1>44.816.5</td><td rowspan=2 colspan=1>22.839.4</td><td rowspan=1 colspan=1>19.5</td><td rowspan=1 colspan=1>36.5</td></tr><tr><td rowspan=1 colspan=1>24.8</td><td rowspan=1 colspan=1>34.2</td></tr><tr><td rowspan=3 colspan=1>HSG</td><td rowspan=3 colspan=1>MOT-M</td><td rowspan=3 colspan=1>M-</td><td rowspan=3 colspan=1>Graph-based optim.</td><td rowspan=3 colspan=1>No</td><td rowspan=3 colspan=1>No</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>ValTest</td><td rowspan=3 colspan=1>58.1(+3.0)52.1(+5.6)</td><td rowspan=3 colspan=1>42.3(+11.1)16.7(+7.5)</td><td rowspan=1 colspan=1>23.2(-5.0)</td><td rowspan=1 colspan=1>27.3(-0.6)</td><td rowspan=1 colspan=1>52.6(-0.3)</td></tr><tr><td rowspan=2 colspan=1>39.4(-5.9)</td><td rowspan=2 colspan=1>29.0(+3.6)</td><td rowspan=1 colspan=1>47.2</td></tr><tr><td rowspan=1 colspan=1>(+2.4)</td></tr><tr><td rowspan=1 colspan=1>RVT-b</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2023</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>40.739.1</td><td rowspan=1 colspan=1>1一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>OC-SORT</td><td rowspan=1 colspan=1>MOT-L</td><td rowspan=1 colspan=1>M-</td><td rowspan=1 colspan=1>KF+HA+IoU</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>2023</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>43.743.5</td><td rowspan=1 colspan=1>10.87.2</td><td rowspan=1 colspan=1>18.826.5</td><td rowspan=1 colspan=1>-11.71.7</td><td rowspan=1 colspan=1>29.428.0</td></tr><tr><td rowspan=1 colspan=1>LPT</td><td rowspan=1 colspan=1>MOT-M</td><td rowspan=1 colspan=1>M-+A-</td><td rowspan=1 colspan=1>Graph-based optim.</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>2022</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>36.936.3</td><td rowspan=1 colspan=1>84.457.1</td><td rowspan=1 colspan=1>1.53.7</td><td rowspan=1 colspan=1>-178.5-146.3</td><td rowspan=1 colspan=1>18.817.1</td></tr><tr><td rowspan=4 colspan=1>HSG</td><td rowspan=4 colspan=1>MOT-M</td><td rowspan=4 colspan=1>M-</td><td rowspan=4 colspan=1>Graph-based optim.</td><td rowspan=4 colspan=1>No</td><td rowspan=4 colspan=1>No</td><td rowspan=4 colspan=1>1</td><td rowspan=4 colspan=1>ValTest</td><td rowspan=4 colspan=1>45.9(+2.2)46.6(+3.1)</td><td rowspan=4 colspan=1>15.8(+5.0)15.5(+8.3)</td><td rowspan=2 colspan=1>13.1(-5.7)</td><td rowspan=2 colspan=1>-10.4(+1.3)</td><td rowspan=1 colspan=1>40.9</td></tr><tr><td rowspan=1 colspan=1>(+11.5)</td></tr><tr><td rowspan=2 colspan=1>16.1(-10.4)</td><td rowspan=2 colspan=1>2.1(+0.4)</td><td rowspan=1 colspan=1>40.4</td></tr><tr><td rowspan=1 colspan=1>(+12.4)</td></tr><tr><td rowspan=1 colspan=1>HDETrack</td><td rowspan=1 colspan=1>SOT</td><td rowspan=1 colspan=1>M-+A-</td><td rowspan=1 colspan=1>E2E Transformer</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>2024</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>68.759.2</td><td rowspan=1 colspan=1>69.948.8</td><td rowspan=1 colspan=1>3.07.2</td><td rowspan=1 colspan=1>36.917.8</td><td rowspan=1 colspan=1>66.556.0</td></tr><tr><td rowspan=1 colspan=1>TSETrack</td><td rowspan=1 colspan=1>SOT</td><td rowspan=1 colspan=1>M-+A-</td><td rowspan=1 colspan=1>E2E Transformer</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>2026</td><td rowspan=1 colspan=1>ValTest</td><td rowspan=1 colspan=1>74.565.4</td><td rowspan=1 colspan=1>83.361.3</td><td rowspan=1 colspan=1>3.07.0</td><td rowspan=1 colspan=1>48.229.9</td><td rowspan=1 colspan=1>72.362.1</td></tr><tr><td rowspan=2 colspan=1>OC-SORT</td><td rowspan=2 colspan=1>MOT-L</td><td rowspan=2 colspan=1>M-</td><td rowspan=2 colspan=1>KF+HA+I₀U</td><td rowspan=2 colspan=1>Yes</td><td rowspan=2 colspan=1>Yes</td><td rowspan=2 colspan=1>2023</td><td rowspan=2 colspan=1>ValTest</td><td rowspan=2 colspan=1>72.863.9</td><td rowspan=2 colspan=1>76.652.3</td><td rowspan=2 colspan=1>4.09.2</td><td rowspan=1 colspan=1>47.3</td><td rowspan=1 colspan=1>67.0</td></tr><tr><td rowspan=1 colspan=1>30.8</td><td rowspan=1 colspan=1>56.5</td></tr><tr><td rowspan=2 colspan=1>LPT</td><td rowspan=2 colspan=1>MOT-M</td><td rowspan=2 colspan=1>M-+A-</td><td rowspan=2 colspan=1>Graph-based optim.</td><td rowspan=2 colspan=1>Yes</td><td rowspan=2 colspan=1>Yes</td><td rowspan=2 colspan=1>2022</td><td rowspan=2 colspan=1>ValTest</td><td rowspan=1 colspan=1>72.1</td><td rowspan=1 colspan=1>82.6</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>41.3</td><td rowspan=1 colspan=1>53.0</td></tr><tr><td rowspan=1 colspan=1>64.1</td><td rowspan=1 colspan=1>39.9</td><td rowspan=1 colspan=1>12.2</td><td rowspan=1 colspan=1>37.2</td><td rowspan=1 colspan=1>39.8</td></tr><tr><td rowspan=3 colspan=1>HSG</td><td rowspan=3 colspan=1>MOT-M</td><td rowspan=3 colspan=1>M-</td><td rowspan=3 colspan=1>Graph-based optim.</td><td rowspan=3 colspan=1>No</td><td rowspan=3 colspan=1>No</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>ValTest</td><td rowspan=1 colspan=1>75.0(+0.5)</td><td rowspan=1 colspan=1>83.2(-0.1)</td><td rowspan=1 colspan=1>3.0(0.0)</td><td rowspan=1 colspan=1>49.9(+1.7)</td><td rowspan=1 colspan=1>71.4(-0.9)</td></tr><tr><td rowspan=1 colspan=1>66.3</td><td rowspan=2 colspan=1>61.1(-0.2)</td><td rowspan=2 colspan=1>7.0(0.0)</td><td rowspan=1 colspan=1>32.9</td><td rowspan=1 colspan=1>63.6</td></tr><tr><td rowspan=1 colspan=1>(+0.9)</td><td rowspan=1 colspan=1>(+3.0)</td><td rowspan=1 colspan=1>(+1.5)</td></tr></table>

![](images/412d0e20fa0f8888202049e675118d4fdb7e2dc93ea2ba093462e98ac581f15e.jpg)  
Fig. 10. Qualitative comparison of OC-SORT and our HSG on two hard eTram sequences under RVT detections. Shown case (a) represents objects with partial occlusions, and (b) represents dense object distribution and occlusions. Red dashed ellipses highlight the challenging scenarios and result differences between two methods. Two sequences (from left to right) are test day 005 and test day 006.

![](images/b73f4db3eb102c4fdba872e19b97b2892906f6672e7f01c15ad469b23964ef96.jpg)  
Fig. 11. Visualization of the GT tracking labels and our corrected tracking labels on Ev-UAV. Sequences (from top to bottom): val 16 and test 9.

## D. Tracking on Ev-UAV

As Ev-UAV is a detection benchmark, the originally provided tracking labels contain some inaccuracies, we corrected and refined the tracking labels on test and validation sets. This was done by manually comparing UAV motion patterns in 2D plane and 3D space to ensure reliable tracking annotations, as illustrated in Fig. 11. More tracking label correction details please refer to the supplementary material, and the corrected dataset will be published in the future <sup>1</sup>. The total number of sequences in test and validation sets is 48. Among these, 20 sequences were refined, including 14 in the test set and 6 in the validation set. To ensure transparency, we report evaluation results both on the remaining 28 unmodified sequences and on the full set with corrected sequences in Table III. The rectified dataset with updated tracking annotations will be made publicly available in our repository.

TABLE III  
TRACKING PERFORMANCE ON EV-UAV. ∗ DENOTES THE FULL SET WITH CORRECTED TRACKING ANNOTATIONS.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Track IoU(%)</td><td rowspan=1 colspan=1>Track Acc(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=2 colspan=1>OC-SORT</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>21.2</td><td rowspan=1 colspan=1>13.1</td><td rowspan=1 colspan=1>9.6</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>23.3</td><td rowspan=1 colspan=1>16.5</td><td rowspan=1 colspan=1>8.8</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>51.1 (+29.9)</td><td rowspan=1 colspan=1>62.3 (+49.2)</td><td rowspan=1 colspan=1>66.2 (+56.6)</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>47.8 (+24.5)</td><td rowspan=1 colspan=1>49.6 (+33.1)</td><td rowspan=1 colspan=1>57.3 (+48.5)</td></tr><tr><td rowspan=2 colspan=1>OC-SORT</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>15.3</td><td rowspan=1 colspan=1>11.7</td><td rowspan=1 colspan=1>4.7</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>15.3</td><td rowspan=1 colspan=1>13.3</td><td rowspan=1 colspan=1>3.9</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>49.9 (+34.6)</td><td rowspan=1 colspan=1>63.7 (+52.0)</td><td rowspan=1 colspan=1>81.3 (+76.6)</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>47.2 (+31.9)</td><td rowspan=1 colspan=1>52.7 (+39.4)</td><td rowspan=1 colspan=1>74.0 (+70.1)</td></tr></table>

The upstream detector is EV-SpSegNet [14], a segmentation network that extracts object events from raw event data. Since it performs binary segmentation, we apply a region growing algorithm with a searching radius of $r = 7$ pixels within every 300 ms interval to cluster the segmented events into tracklets. These tracklets are then fed into the association pipeline.

As shown in Table III, our approach outperforms OC-SORT by over 2× across all metrics on both original and corrected Ev-UAV. Fig. 12 shows several challenging sequences containing tracklets with large spatio-temporal gaps and multiple motion patterns. Additional experiments with varying region growing radius r are provided in the supplementary material. All these results demonstrate that our method successfully associates long-range fragmented tracklets into complete trajectories.

The Ev-UAV dataset presents a classic and challenging tiny object tracking scenario. Different from image-based small object tracking, events here offer no appearance cues and the association problem is further complicated by its many-to-many nature. Despite these challenges, our method demonstrates robust and accurate associations, reliably linking isolated tracklets and effectively addressing the inherent difficulties of tiny object tracking.

## E. Computational Efficiency

Although multi-dimensional assignment sacrifices some computational efficiency for accuracy gains, we rigorously analyze the time complexity of our proposed method. Our method consists of two stages, graph modeling and graph partitioning. Let N denote the number of tracklets to be associated, and $\zeta ~ ( \zeta \ll \mathcal { N } )$ denote the candidates of each tracklet after initial sparsification. In the modeling stage, metric calculations operate in $O ( \mathcal { N } ^ { 2 } )$ , and K-means with I iterations and N + 1 centers operates in $O ( \mathcal { N } \cdot I \cdot \zeta \cdot ( N + 1 ) )$ . Consequently, the whole time complexity of modeling is $O ( N \cdot \zeta + \ N ^ { 2 } )$ which is dominated by $O ( \mathcal { N } ^ { 2 } )$ since our implementation is in serial. After modeling, the HSG consists of $\mathcal { N }$ nodes and E ≈ $( { \mathcal { N } } { \cdot } { \zeta } ) / { 2 }$ edges. In the first partitioning iteration, computing the connectivity similarity for all vertices costs $O ( \mathcal { N } \cdot \zeta ^ { 2 } )$ Since merging costs $O ( E )$ which is negligible compared to $O ( \mathcal { N } \cdot \zeta ^ { 2 } )$ , the time complexity of a single search-merge is $O ( \mathcal { N } \cdot \zeta ^ { 2 } )$ . Each iteration reduces the number of vertices by at least half, so partitioning requires at most $O ( \log _ { 2 } \mathcal { N } )$

![](images/f983e79d6acac8358bc078d70b7d5d12766cb055ada732852c7c2782e22b246e.jpg)  
Fig. 12. Visualization of the object trajectories of GT, proposed HSG, and OC-SORT on Ev-UAV. Sequences feature the challenge of associating tracklets under large spatio-temporal gaps and multiple motion patterns. Sequences (from top to bottom): val 11#, val 20#, test 6∗, test 10∗, and test 13∗. # denotes the unmodified sequence, while ∗ denotes the corrected sequence.

TABLE IV  
RUNNING EFFICIENCY COMPARING WITH ASSOCIATION BASELINES UNDER YOLOX DETECTIONS ON ETRAM. THE BEST VALUES ARE IN “BOLD”.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>FPS</td><td rowspan=1 colspan=1> $\mathrm { E P S } ( \times 1 0 ^ { 6 } )$ </td><td rowspan=1 colspan=1>Hardware</td></tr><tr><td rowspan=2 colspan=1>LTTrack</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>3.31</td><td rowspan=1 colspan=1>一</td><td rowspan=6 colspan=1>GPU</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>3.68</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=2 colspan=1>TOPIC</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>27.56</td><td rowspan=1 colspan=1>二</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>33.07</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=2 colspan=1>LPT</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>38.62</td><td rowspan=1 colspan=1>二</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>46.48</td><td rowspan=1 colspan=1>二</td></tr><tr><td rowspan=2 colspan=1>OCSORT</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>439.84</td><td rowspan=1 colspan=1>一</td><td rowspan=4 colspan=1>CPU</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>522.78</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>9.99</td><td rowspan=1 colspan=1>1.95</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>12.07</td><td rowspan=1 colspan=1>1.75</td></tr></table>

iterations. The time complexity of the whole graph partitioning is $O ( \mathcal { N } \cdot \zeta ^ { 2 } \cdot \log _ { 2 } \mathcal { N } )$ . Therefore, the overall time complexity of our method is $O ( \mathcal { N } ^ { 2 } + \mathcal { N } \cdot \zeta ^ { 2 } \cdot \log _ { 2 } \mathcal { N } )$

Beyond qualitative analysis, we quantitatively evaluate the computational efficiency of our method against some association methods in Table IV. Since comparisons are frame-based whereas our method is event-driven, we report two metrics, frames per second (FPS) and events per second (EPS). All evaluations are conducted on a platform equipped with an Intel i7-7800X CPU @3.50GHz and an NVIDIA RTX4090 GPU. The fastest method is OC-SORT. It achieves a significantly higher speed (over 10× faster) than the others. Although our method is slower than LPT, the other global optimization assignment method, it still processes over 1M EPS and requires no GPU resources. Given that the modeling stage with $O ( \mathcal { N } ^ { 2 } )$ is the primary computational bottleneck, our method presents potential for further acceleration via parallelization.

TABLE V  
ABLATION OF BEZIER CURVE ´ -BASED TERM (➀), AND DIRECTION-DISPLACEMENT-BASED TERM (➁) UNDER YOLOX DETECTIONS ON ETRAM. “W/O” MEANS WITHOUT THE CORRESPONDING COMPONENT.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>MT(%)</td><td rowspan=1 colspan=1>ML(%)</td><td rowspan=1 colspan=1>MOTA(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=2 colspan=1>w/o ①</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.2</td><td rowspan=1 colspan=1>23.2</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>52.5</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>39.4</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>46.7</td></tr><tr><td rowspan=2 colspan=1>w/o ②</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.3</td><td rowspan=1 colspan=1>23.2</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>52.7</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>39.4</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>46.8</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.3</td><td rowspan=1 colspan=1>23.2</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>52.6</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>39.4</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>47.2</td></tr></table>

TABLE VI  
ABLATION OF BEZIER CURVE ´ -BASED TERM (➀) AND DIRECTION-DISPLACEMENT-BASED TERM (➁) ON EV-UAV.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Track IoU(%)</td><td rowspan=1 colspan=1>Track Acc(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=2 colspan=1>w/o ①</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>41.6</td><td rowspan=1 colspan=1>34.9</td><td rowspan=1 colspan=1>52.7</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>51.6</td><td rowspan=1 colspan=1>43.0</td><td rowspan=1 colspan=1>59.6</td></tr><tr><td rowspan=2 colspan=1>w/o ②</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>61.6</td><td rowspan=1 colspan=1>62.5</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>46.8</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>56.3</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>51.1</td><td rowspan=1 colspan=1>62.3</td><td rowspan=1 colspan=1>66.2</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>47.8</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>57.3</td></tr><tr><td rowspan=2 colspan=1>w/o ①</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>33.5</td><td rowspan=1 colspan=1>32.9</td><td rowspan=1 colspan=1>37.5</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>33.6</td><td rowspan=1 colspan=1>32.1</td><td rowspan=1 colspan=1>37.2</td></tr><tr><td rowspan=2 colspan=1>w/o ②</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>49.1</td><td rowspan=1 colspan=1>63.3</td><td rowspan=1 colspan=1>83.0</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>45.5</td><td rowspan=1 colspan=1>52.6</td><td rowspan=1 colspan=1>75.6</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>49.9</td><td rowspan=1 colspan=1>63.7</td><td rowspan=1 colspan=1>81.3</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>52.7</td><td rowspan=1 colspan=1>74.0</td></tr></table>

TABLE VII  
ABLATION OF LINEAR (➀-L), QUADRATIC (➀), AND CUBIC (➀-C) BEZIER´ CURVE-BASED TERM UNDER YOLOX DETECTIONS ON ETRAM.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>MT(%)</td><td rowspan=1 colspan=1>ML(%)</td><td rowspan=1 colspan=1>MOTA(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=2 colspan=1>w ①-1</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.3</td><td rowspan=1 colspan=1>23.2</td><td rowspan=1 colspan=1>27.2</td><td rowspan=1 colspan=1>52.1</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>39.4</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>46.9</td></tr><tr><td rowspan=2 colspan=1>w ①-c</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.3</td><td rowspan=1 colspan=1>23.2</td><td rowspan=1 colspan=1>27.2</td><td rowspan=1 colspan=1>52.1</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>39.4</td><td rowspan=1 colspan=1>28.9</td><td rowspan=1 colspan=1>46.7</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.3</td><td rowspan=1 colspan=1>23.2</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>52.6</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>39.4</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>47.2</td></tr></table>

## F. Ablation Studies

Our ablation studies focus on two modeling terms of our fuzzy modeling pipeline and the adaptive threshold τ in the graph partitioning. Experimental settings remain identical to before. Ablation results on modeling terms are presented in Table V and Table VI. Besides, we conduct a quantitative analysis of the Bezier curve order, comparing linear, quadratic,´ and cubic fitting, as shown in Table VII and Table VIII.

(1) Analysis of Bezier curve-based position smoothness´ . On eTram, the improvement brought by the Bezier curve-´ based position smoothness is modest, as object motion is predominantly linear, making the line-based error metric the primary distinguishing factor. In contrast, on Ev-UAV, where tracklets undergo large spatio-temporal variations, the Bezier´ curve model becomes the preferred choice. On the original set, the performance of our method is lower than that of the variant without ➀. This occurs because the incorrect GT tracking labels divide an object trajectory into a few segments, further penalizing our metrics as one predicted trajectory can only match one GT trajectory. However, on the corrected set, our method yields nearly 2× improvements over the variant without this term, clearly demonstrating its effectiveness in handling complex motion patterns and long-range association.

(2) Analysis of the Bezier curve order´ . On eTram, our selected quadratic Bezier curve consistently outperforms the´ linear and cubic counterparts by 0.1%-0.3% in terms of MOTA and IDF1. On Ev-UAV, the quadratic curve model achieves the best trade-off. This experiment validates our analysis of Bezier´ curves in Sec. III-B.

TABLE VIII  
ABLATION OF THE LINEAR (➀-L), QUADRATIC (➀), AND CUBIC (➀-C) BEZIER CURVE´ -BASED TERM ON EV-UAV.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Track IoU(%)</td><td rowspan=1 colspan=1>Track Acc(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=2 colspan=1>w ①-1</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>54.9</td><td rowspan=1 colspan=1>64.9</td><td rowspan=1 colspan=1>68.1</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>42.4</td><td rowspan=1 colspan=1>47.6</td><td rowspan=1 colspan=1>59.7</td></tr><tr><td rowspan=2 colspan=1>w ①-c</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>51.4</td><td rowspan=1 colspan=1>67.4</td><td rowspan=1 colspan=1>60.5</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>40.1</td><td rowspan=1 colspan=1>48.3</td><td rowspan=1 colspan=1>59.9</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>51.1</td><td rowspan=1 colspan=1>62.3</td><td rowspan=1 colspan=1>66.2</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>47.8</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>57.3</td></tr><tr><td rowspan=2 colspan=1>w ①-1</td><td rowspan=1 colspan=1>Val *</td><td rowspan=1 colspan=1>54.5</td><td rowspan=1 colspan=1>67.1</td><td rowspan=1 colspan=1>81.2</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>46.9</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>69.1</td></tr><tr><td rowspan=2 colspan=1>w ①-c</td><td rowspan=1 colspan=1>Val *</td><td rowspan=1 colspan=1>48.7</td><td rowspan=1 colspan=1>62.2</td><td rowspan=1 colspan=1>77.4</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>40.6</td><td rowspan=1 colspan=1>50.5</td><td rowspan=1 colspan=1>75.4</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>49.9</td><td rowspan=1 colspan=1>63.7</td><td rowspan=1 colspan=1>81.3</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>52.7</td><td rowspan=1 colspan=1>74.0</td></tr></table>

![](images/6b26460ae0d615ebf9a0dbbae428c5cdd1d755ce606a0d1e114a28b821633cae.jpg)  
Fig. 13. Ablation of the direction-displacement-based variation consistency term (➁) on eTram. Red dashed ellipses highlight cases where multiple tracklets have close positions but different moving directions. Two sequences are presented, separated by a black line: test day 001 (top) and test day 005 (bottom).

(3) Analysis of direction-displacement-based variation consistency. On eTram, the variant without this term achieves a slightly higher IDF1 than HSG by 0.1%. This marginal gain stems from a reduction in tracklet fragmentation, but as analyzed in Sec. III-C, it also reflects over-association in some cases. To further analyze this, we visualize several test cases in Fig. 13, where tracklets needed to be distinguished are highlighted by red dashed ellipses. HSG correctly associates them, whereas the variant without this term incorrectly associates them, confirming that this term is essential for distinguishing tracklets in such cases. The results on Ev-UAV also validate the importance of this term since the full model performs better than the variant without this term most of the time.

TABLE IX  
ABLATIONS OF λ ON ETRAM.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Detections</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>MOTA (%)</td><td rowspan=1 colspan=1>IDF1 (%)</td></tr><tr><td rowspan=2 colspan=1>w/o λ</td><td rowspan=4 colspan=1>GT</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>98.2</td><td rowspan=1 colspan=1>80.1</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>98.0</td><td rowspan=1 colspan=1>77.1</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>98.5</td><td rowspan=1 colspan=1>95.7</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>98.4</td><td rowspan=1 colspan=1>95.3</td></tr><tr><td rowspan=2 colspan=1>w/o λ</td><td rowspan=4 colspan=1>YOLOX</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>98.2</td><td rowspan=1 colspan=1>80.1</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>98.0</td><td rowspan=1 colspan=1>77.1</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>98.5</td><td rowspan=1 colspan=1>95.7</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>98.4</td><td rowspan=1 colspan=1>95.3</td></tr></table>

TABLE X

ABLATION OF FACTORS IN DERIVING τ , INCLUDING NODE SIZE BALANCING FACTOR γ (➂) AND DENOMINATOR m + 1 (➃) UNDER YOLOX DETECTIONS ON ETRAM. ∗ AND <sup>‡</sup> DENOTE STATISTICALLY SIGNIFICANT IMPROVEMENT (p < 0.05) OVER ➂ AND ➃ (PAIRED T-TEST, n=25 FOR VAL AND n=30 FOR TEST), RESPECTIVELY.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>MT(%)/p</td><td rowspan=1 colspan=1>ML(%)/p</td><td rowspan=1 colspan=1>MOTA(%)/p</td><td rowspan=1 colspan=1>IDF1(%)/p</td></tr><tr><td rowspan=2 colspan=1>w/o ③</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.3/-</td><td rowspan=1 colspan=1>23.2/-</td><td rowspan=1 colspan=1>27.2/1.21E-2</td><td rowspan=1 colspan=1>52.0/3.99E-2</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7/-</td><td rowspan=1 colspan=1>39.4/-</td><td rowspan=1 colspan=1>29.0/3.56E-4</td><td rowspan=1 colspan=1>46.9/2.18E-2</td></tr><tr><td rowspan=2 colspan=1>w/o ④</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.3/-</td><td rowspan=1 colspan=1>23.2/-</td><td rowspan=1 colspan=1>27.0/4.01E-9</td><td rowspan=1 colspan=1>47.2/1.77E-5</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7/-</td><td rowspan=1 colspan=1>39.4/-</td><td rowspan=1 colspan=1>28.8/7.24E-8</td><td rowspan=1 colspan=1>42.0/6.25E-6</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>42.3</td><td rowspan=1 colspan=1>23.2</td><td rowspan=1 colspan=1> $2 7 . 3 ^ { * \ddagger }$ </td><td rowspan=1 colspan=1> ${ \bf 5 2 . 6 ^ { * \ddagger } }$ </td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>39.4</td><td rowspan=1 colspan=1> $2 9 . 0 ^ { * \ddagger }$ </td><td rowspan=1 colspan=1> $4 7 . 2 ^ { * \ddagger }$ </td></tr></table>

(4) Analysis of temporal weight λ. As shown in Table IX, omitting λ drops IDF1 by over 15% under GT detections and 10% under YOLOX detections. This degradation confirms that λ effectively mitigates the association unreliability introduced by long time intervals, thereby ensuring accurate tracking performance. Detailed factor ablations and sensitivity analyses are deferred to the supplementary material.

(5) Analysis of τ. We evaluate the effectiveness of τ and the necessity of its factor γ and denominator m + 1. Fig. 14 compares our adaptive threshold τ against a series of fixed thresholds ranging from 0.1 to 0.9. The performance using $\tau$ is represented by horizontal dashed lines. Evidently, using τ achieves more robust and accurate tracking than the fixed thresholds. Table X and Table XI report the performance of our method without $\gamma \mathrm { ~ } \mathrm { ~ o r ~ }$ the denominator $m + 1$ from the τ calculation. On eTram, our full method consistently outperforms these variants. To validate the improvements, we also conducted paired t-tests on per-sequence metrics. As shown in Table X, HSG yields statistically significant improvements over the two variants with $p < 0 . 0 5$ . In contrast to this, two variants perform on par with or even better than our full method on Ev-UAV. We attribute this to the differing graph sparsity across datasets. Specifically, γ and $m + 1$ correction are designed for dense graphs, addressing size imbalance and overly strict merging conditions, respectively. When the HSG is sparse, the connectivity within a vertex degenerates into a path, and their effect diminishes or even introduces few incorrect associations. These results confirm that adapting the method’s configuration to the target scenario can improve performance across different applications.

TABLE XI  
ABLATION OF FACTORS IN DERIVING τ, INCLUDING THE BALANCING FACTOR γ (➂) AND THE DENOMINATOR m + 1 (➃) ON EV-UAV.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Track IoU(%)</td><td rowspan=1 colspan=1>Track Acc(%)</td><td rowspan=1 colspan=1>IDF1(%)</td></tr><tr><td rowspan=2 colspan=1>w/o ③</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>51.4</td><td rowspan=1 colspan=1>62.5</td><td rowspan=1 colspan=1>67.2</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>47.8</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>57.3</td></tr><tr><td rowspan=2 colspan=1>w/o ④</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>53.1</td><td rowspan=1 colspan=1>64.2</td><td rowspan=1 colspan=1>68.7</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>47.1</td><td rowspan=1 colspan=1>50.3</td><td rowspan=1 colspan=1>58.8</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val</td><td rowspan=1 colspan=1>51.1</td><td rowspan=1 colspan=1>62.3</td><td rowspan=1 colspan=1>66.2</td></tr><tr><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>47.8</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>57.3</td></tr><tr><td rowspan=2 colspan=1>w/o ③</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>50.3</td><td rowspan=1 colspan=1>64.0</td><td rowspan=1 colspan=1>82.5</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>52.7</td><td rowspan=1 colspan=1>74.0</td></tr><tr><td rowspan=2 colspan=1>w/o ④</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1>63.4</td><td rowspan=1 colspan=1>86.6</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>48.3</td><td rowspan=1 colspan=1>53.6</td><td rowspan=1 colspan=1>72.9</td></tr><tr><td rowspan=2 colspan=1>HSG</td><td rowspan=1 colspan=1>Val*</td><td rowspan=1 colspan=1>49.9</td><td rowspan=1 colspan=1>63.7</td><td rowspan=1 colspan=1>81.3</td></tr><tr><td rowspan=1 colspan=1>Test*</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>52.7</td><td rowspan=1 colspan=1>74.0</td></tr></table>

![](images/a87a038d171ac9e1269230c2840f8fa5b2556c58ff53c753f8bd451ae4fd45f3.jpg)

![](images/70fc28809b05a60fce871a9524633ebec9afb5a48424c569b64cc12aa070cb90.jpg)

![](images/245bd71188196a8f0fb0255778d265dde3f02b09ceff53b0fa18069a0911f403.jpg)

![](images/acdd3c4992670dac22423cfa7143f95e113ba3131d9426ad4d0ec7861d757235.jpg)  
Fig. 14. Comparisons of adaptive merging threshold τ with a series of fixed thresholds. The first row is the results on eTram, and the second row is the results on Ev-UAV.

## G. Discussion

Experiments demonstrate the effectiveness and superior ity of our method. However, our approach still possesses certain limitations. Fundamentally, our graph partitioning is akin to a voting system, assigning each tracklet to the most likely trajectory without enforcing global continuity within a partition. This can lead to erroneous associations when voting evidence is insufficient or ambiguous. For example, under sparse evidence, an isolated tracklet may be incorrectly absorbed into a continuous trajectory merely due to spatial proximity and a similar motion direction. Conversely, when evidence is ambiguous, such as in extremely dense crowds, the limitation becomes equally pronounced. In such scenarios, event trajectories of adjacent targets heavily overlap and exhibit similar motion patterns, introducing severe ambiguity into the voting mechanism. Addressing these limitations opens up several future directions, including examining a sliding window of tracklets within each partition after each round of graph partitioning, or designing incremental metrics that capture motion continuity across multiple tracklets.

## VI. CONCLUSION

In this work, we formulate the event-based multidimensional assignment problem as a graph partition task. Unlike image-based tracking, event-based tracking inherently involves the many-to-many association, which is an underdetermined and NP-hard task arising from the asynchronous nature of events. Therefore, we propose a fuzzy modeling method to construct a Hierarchical Sparse Graph (HSG) that captures relationships between candidate pairs. Subsequently, an automatic graph partitioning algorithm is also developed based on the HSG to obtain optimal assignments. The overall method is training-free and plug-and-play. Experiments on both one-to-one and many-to-many matching scenarios show the strong performance of our method across diverse challenging conditions, particularly in long-range and tiny object tracking.

## REFERENCES

[1] S. H. Bae and K. J. Yoon, “Robust Online Multiobject Tracking With Data Association and Track Management,” IEEE Trans. Image Process., vol. 23, no. 7, pp. 2820–2833, Jul. 2014.

[2] J. Gao, B. Zhong, and Y. Chen, “Robust Tracking via Learning Model Update With Unsupervised Anomaly Detection Philosophy,” IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 5, pp. 2330– 2341, May 2023.

[3] L. Rakai, H. Song, S. Sun, W. Zhang, and Y. Yang, “Data association in multiple object tracking: A survey of recent techniques,” Expert Syst. Appl., vol. 192, p. 116300, Apr. 2022.

[4] X. Wang, C. Wang, S. Wang, X. Wang, Z. Zhao, L. Zhu, and B. Jiang, “MambaEVT: Event Stream-Based Visual Object Tracking Using State Space Model,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, no. 1, pp. 278–291, Jan. 2026.

[5] P. Tokmakov, J. Li, W. Burgard, and A. Gaidon, “Learning to Track with Object Permanence,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Montreal, QC, Canada, Oct. 2021, pp. 10 840–10 849.

[6] S. Liu, X. Li, H. Lu, and Y. He, “Multi-Object Tracking Meets Moving UAV,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., New Orleans, Louisiana, USA, 2022.

[7] J. Cao, J. Pang, X. Weng, R. Khirodkar, and K. Kitani, “Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Vancouver, Canada, 2023, pp. 9686–9696.

[8] L. Zhang, Y. Li, and R. Nevatia, “Global data association for multi-object tracking using network flows,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Anchorage, Alaska, Jun. 2008, pp. 1–8.

[9] J. Lin, G. Liang, and R. Zhang, “LTTrack: Rethinking the Tracking Framework for Long-Term Multi-Object Tracking,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 10, pp. 9866–9881, Oct. 2024.

[10] A. A. Butt and R. T. Collins, “Multi-target Tracking by Lagrangian Relaxation to Min-cost Network Flow,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Jun. 2013, pp. 1846–1853.

[11] P. Ochs and T. Brox, “Higher order motion models and spectral clustering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Providence, RI, USA, Jun. 2012, pp. 614–621.

[12] M. Gehrig and D. Scaramuzza, “Recurrent Vision Transformers for Object Detection With Event Cameras,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Vancouver, Canada, 2023, pp. 13 884– 13 893.

[13] A. A. Verma, B. Chakravarthi, A. Vaghela, H. Wei, and Y. Yang, “eTraM: Event-based Traffic Monitoring Dataset,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Seattle WA, USA, 2024, pp. 22 637– 22 646.

[14] N. Chen, C. Xiao, Y. Dai, S. He, M. Li, and W. An, “Event-based Tiny Object Detection: A Benchmark Dataset and Baseline,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Honolulu, Hawaii, Jun. 2025.

[15] G. Gallego, T. Delbruck, G. Orchard, C. Bartolozzi, B. Taba, A. Censi, S. Leutenegger, A. Davison, J. Conradt, K. Daniilidis, and D. Scaramuzza, “Event-based Vision: A Survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 1, pp. 154–180, Jan. 2022.

[16] J. Huang, S. Wang, M. Guo, and S. Chen, “Event-Guided Structured Output Tracking of Fast-Moving Objects Using a CeleX Sensor,” IEEE Trans. Circuits Syst. Video Technol., vol. 28, no. 9, pp. 2413–2417, Sep. 2018.

[17] M. Wang, F. Shi, X. Cheng, and S. Chen, “Prior Knowledge-Driven Hybrid Prompter Learning for RGB-Event Tracking,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 9, pp. 8679–8691, Sep. 2025.

[18] C. Liu, Z. Guan, S. Lai, Y. Liu, H. Lu, and D. Wang, “EMTrack: Efficient Multimodal Object Tracking,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 3, pp. 2202–2214, Mar. 2025.

[19] Q. Meng, A. Lu, and Z. Jin, “BR-MoE: Blind Multi-Modal Tracking With Route-Dynamic Mixture of Experts,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 12, pp. 11 908–11 921, Dec. 2025.

[20] A. Dietsche, G. Cioffi, J. Hidalgo-Carrio, and D. Scaramuzza, “Powerline Tracking with Event Cameras,” in Proc. IEEE/RSJ Int. Conf. Intell. Rob. Syst. Prague, Czech Republic: IEEE, Sep. 2021, pp. 6990–6997.

[21] L. Everding and J. Conradt, “Low-Latency Line Tracking Using Event-Based Dynamic Vision Sensors,” Front. Neurorobot., vol. 12, 2018.

[22] C. Le Gentil, F. Tschopp, I. Alzugaray, T. Vidal-Calleja, R. Siegwart, and J. Nieto, “IDOL: A Framework for IMU-DVS Odometry using Lines,” in Proc. IEEE/RSJ Int. Conf. Intell. Rob. Syst. Las Vegas, NV, USA: IEEE, Oct. 2020, pp. 5863–5870.

[23] A. Z. Zhu, N. Atanasov, and K. Daniilidis, “Event-Based Visual Inertial Odometry,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Honolulu, Hawaii: IEEE, Jul. 2017, pp. 5816–5824.

[24] R. Jiang, X. Mou, S. Shi, Y. Zhou, Q. Wang, M. Dong, and S. Chen, “Object tracking on event cameras with offline-online learning,” CAAI Trans. Intell. Technol., vol. 5, no. 3, pp. 165–171, 2020.

[25] Z. Wang, T. Molloy, P. van Goor, and R. Mahony, “Event Blob Tracking: An Asynchronous Real-Time Algorithm,” arXiv preprint arXiv:2307.10593, 2023.

[26] F. Barranco, C. Fermuller, and E. Ros, “Real-time clustering and multitarget tracking using event-based sensors,” in IEEE/RSJ Int. Conf. Intell. Robots Syst., 2018, pp. 5764–5769.

[27] I. Alzugaray and M. Chli, “Asynchronous Corner Detection and Tracking for Event Cameras in Real Time,” IEEE Robot. Autom. Lett., vol. 3, no. 4, pp. 3177–3184, Oct. 2018.

[28] X. Lagorce, C. Meyer, S.-H. Ieng, D. Filliat, and R. Benosman, “Asynchronous Event-Based Multikernel Algorithm for High-Speed Visual Features Tracking,” IEEE Trans. Neural Netw. Learn. Syst., vol. 26, no. 8, 2015.

[29] I. Alzugaray, “HASTE: Multi-Hypothesis Asynchronous Speeded-up Tracking of Events,” in Proc. Brit. Mach. Vis. Conf., UK, 2020.

[30] S. Hu, Y. Kim, H. Lim, A. J. Lee, and H. Myung, “ecdt: Event clustering for simultaneous feature detection and tracking,” in IEEE/RSJ Int. Conf. Intell. Robots Syst., 2022, pp. 3808–3815.

[31] B. Ramesh, S. Zhang, H. Yang, A. Ussa, M. Ong, G. Orchard, and C. Xiang, “E-TLD: Event-based Framework for Dynamic Object Tracking,” IEEE Trans. Circuits Syst. Video Technol., vol. 30, no. 10, pp. 3996–4006, Sep. 2020.

[32] H. Chen, Q. Wu, Y. Liang, X. Gao, and H. Wang, “Asynchronous Tracking-by-Detection on Adaptive Time Surfaces for Event-based Object Tracking,” in Proc. ACM Int. Conf. Multimedia, France, 2019, pp. 473–481.

[33] H. Chen, D. Suter, Q. Wu, and H. Wang, “End-to-End Learning of Object Motion Estimation from Retinal Events for Event-Based Object Tracking,” vol. 34, no. 07, Apr. 2020, pp. 10 534–10 541.

[34] J. Berclaz, F. Fleuret, E. Turetken, and P. Fua, “Multiple Object Tracking Using K-Shortest Paths Optimization,” IEEE Trans. Pattern Anal. Mach Intell., vol. 33, no. 9, pp. 1806–1819, Sep. 2011.

[35] H. Pirsiavash, D. Ramanan, and C. C. Fowlkes, “Globally-optimal greedy algorithms for tracking a variable number of objects,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Colorado Springs, CO, USA, Jun. 2011, pp. 1201–1208.

[36] R. T. Collins, “Multitarget data association with higher-order motion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Providence, RI, USA, Jun. 2012, pp. 1744–1751.

[37] J. Poschmann, T. Pfeifer, and P. Protzel, “Factor Graph based 3D Multi-¨ Object Tracking in Point Clouds,” in Proc. IEEE/RSJ Int. Conf. Intell. Rob. Syst., Las Vegas, NV, USA, Oct. 2020, pp. 10 343–10 350.

[38] A. Roshan Zamir, A. Dehghan, and M. Shah, “GMCP-Tracker: Global Multi-object Tracking Using Generalized Minimum Clique Graphs,” in Proc. Eur. Conf. Comput. Vis. Firenze, Italy: Springer, 2012, pp. 343– 356.

[39] A. Dehghan, S. Modiri Assari, and M. Shah, “GMMCP Tracker: Globally Optimal Generalized Maximum Multi Clique Problem for Multiple Object Tracking,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Boston, Massachusetts, USA, 2015, pp. 4091–4099.

[40] A. Z. Zhu, N. Atanasov, and K. Daniilidis, “Event-based feature tracking with probabilistic data association,” in Proc. IEEE Int. Conf. Robot. Autom., Shanghai, China, 2017, pp. 4465–4470.

[41] H. Seok and J. Lim, “Robust Feature Tracking in DVS Event Stream using Bezier Mapping,” in´ Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., CO, U.S.A., Mar. 2020, pp. 1647–1656.

[42] Y. Chen, Y. Cai, J. Zheng, and D. Thalmann, “Accurate and Efficient Approximation of Clothoids Using Bezier Curves for Path Planning,”´ IEEE Trans. Robot., vol. 33, no. 5, pp. 1242–1247, Oct. 2017.

[43] X. Lagorce, G. Orchard, F. Galluppi, B. E. Shi, and R. B. Benosman, “HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 7, pp. 1346–1359, Jul. 2016.

[44] V. D. Blondel, J. L. Guillaume, R. Lambiotte, and E. Lefebvre, “Fast unfolding of communities in large networks,” J. Stat. Mech., vol. 2008, no. 10, p. P10008, Oct. 2008.

[45] X. Xu, N. Yuruk, Z. Feng, and T. A. J. Schweiger, “SCAN: A structural clustering algorithm for networks,” in Proc. 13th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. NY, USA: Association for Computing Machinery, Aug. 2007, pp. 824–833.

[46] M. E. J. Newman, “Modularity and community structure in networks,” Proc. Natl. Acad. Sci. USA, vol. 103, no. 23, pp. 8577–8582, Jun. 2006.

[47] B. Keni and S. Rainer, “Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics,” EURASIP J. Image Video Process., vol. 2008, no. 1, p. 246309, Jan. 2008.

[48] E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi, “Performance Measures and a Data Set for Multi-target, Multi-camera Tracking,” in in Proc. Eur. Conf. Comput. Vis. Workshops, G. Hua and H. Jegou, Eds.´ Cham: Springer International Publishing, 2016, pp. 17–35.

[49] S. Mentasti, A. W. Kambale, and M. Matteucci, “Event-Based Object Detection and Tracking - A Traffic Monitoring Use Case -,” in Int. Conf. Image Anal. Process. Workshops. Cham: Springer International Publishing, 2022, vol. 13374, pp. 95–106.

[50] A. Apps, Z. Wang, V. Perejogin, T. Molloy, and R. Mahony, “Asynchronous Multi-Object Tracking with an Event Camera,” in Proc. IEEE Int. Conf. Robot. Autom., Atlanta, GA, USA, 2025.

[51] Z. Wang, T. Molloy, P. van Goor, and R. Mahony, “Asynchronous Blob Tracker for Event Cameras,” IEEE Trans. Robot., vol. 40, pp. 4750– 4767, 2024.

[52] A. Rusch and T. Roesgen, “TrackAER: Real-Time Event-Based Particle Tracking,” in Proc. Int. Symp. Part. Image Velocim., vol. 1, no. 1, Aug. 2021.

[53] F. Hamann, D. Gehrig, F. Febryanto, K. Daniilidis, and G. Gallego, “ETAP: Event-based Tracking of Any Point,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Nashville, TN, USA, 2025.

[54] X. Wang, S. Wang, C. Tang, L. Zhu, B. Jiang, Y. Tian, and J. Tang, “Event Stream-based Visual Object Tracking: A High-Resolution Benchmark Dataset and A Novel Baseline,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Seattle, WA, USA, 2024, pp. 19 248–19 257.

[55] J. Zhang, X. Yang, H. Tang, Y. Wang, B. Yin, H. Wang, and X. Fu, “Efficient Vision Transformer with Token Sparsification for Event-Based Object Tracking,” Int. J. Comput. Vis., vol. 134, no. 2, p. 75, Feb. 2026.

[56] X. Cao, Y. Zheng, Y. Yao, H. Qin, X. Cao, and S. Guo, “TOPIC: A Parallel Association Paradigm for Multi-Object Tracking Under Complex Motions and Diverse Scenes,” IEEE Trans. Image Process., vol. 34, pp. 743–758, 2025.

[57] S. Li, Y. Kong, and H. Rezatofighi, “Learning of Global Objective for Network Flow in Multi-Object Tracking,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 8855–8865.

[58] Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, “YOLOX: Exceeding YOLO Series in 2021,” arXiv preprint arXiv:2107.08430, 2021.

[59] N. Ballas, L. Yao, C. Pal, and A. Courville, “Delving Deeper into Convolutional Networks for Learning Video Representations,” arXiv preprint arXiv:1511.06432, 2016.

[60] L. He, X. Liao, W. Liu, X. Liu, P. Cheng, and T. Mei, “Fastreid: A pytorch toolbox for general instance re-identification,” arXiv preprint arXiv:2006.02631, 2020.

![](images/182f15aeb901008be6fee070d78272654f7a81cf0c171326fc243e444045f65a.jpg)

Muxi Zha received the B.E. degree from University of Science & Technology Beijing, Beijing, China in 2020, and the M.E. degree from the College of Electrical and Information Engineering, Hunan University, Changsha, China, in 2023. She is currently pursuing the Ph.D. degree with the College of Aerospace Science and Engineering, National University of Defense Technology, Changsha, China. Her research interests include data processing and object tracking in both image and event domains.

![](images/edb28923a1f0f00df04a444d40f4e8a00e44cd18ff67eb88eca3b847090b6370.jpg)

Yang Shang received the B.S. degree in aerodynamics, and the Ph.D. degree in aeronautical and astronautical science and technology from the National University of Defense Technology, Changsha, China, in 2000 and 2006, respectively. He is currently a Professor with the College of Aerospace Science and Engineering, National University of Defense Technology. His main research interests include photogrammetry, machine vision, and navigation systems.

![](images/01a97f6f4759dcd450e021f8c3a0c5cd4c7576e52d81318e00b5706316c89223.jpg)

Wenlong Zhang received the B.E., M.S., and Ph.D. degrees from the National University of Defense Technology, Changsha, China, in 2014, 2016, and 2020, respectively. He is currently a Lecturer with the College of Aerospace Science and Engineering, National University of Defense Technology. His research interests include image processing and compute vision.

![](images/54a6359499b18073b97d6a646f46c9f2b6366511ed634cebe67ec194dd22eb08.jpg)

Qifeng Yu received the B.S. degree from Northwestern Polytechnic University, Xian, China, in 1981, the M.S. degree from the National University of Defense Technology, Changsha, China, in 1984, and the Ph.D. degree from Bremen University, Bremen, Germany, in 1996. Currently, he is a Professor at the National University of Defense Technology. He is a Member of the Chinese Academy of Sciences. His current research fields are image measurement and vision navigation.

![](images/9a4107d8773f5f4e65fa595effaef53e845faa93bd23a5f0ae72a044d3e4f5ad.jpg)

Banglei Guan received a B.S. degree in Geomatics Engineering from Wuhan University, Wuhan, China, in 2012 and a PhD degree in Aeronautical and Astronautical Science and Technology from National University of Defense Technology, Changsha, China, in 2018. From 2016 to 2017, he was an Exchange Student at Graz University of Technology, Graz, Austria. He is currently a professor with the National University of Defense Technology. His research interests include photomechanics and videometrics.