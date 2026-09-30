# CurvSpec: Adaptive Multi-Curvature Learning for Partial Relevant Video Retrieval

Zhen Liu<sup>∗</sup>   
Tongji University   
Shanghai, China   
sincetodayz@gmail.com   
Letian Li<sup>∗</sup>   
SIGS, Tsinghua University   
Shenzhen, China   
lilt24@mails.tsinghua.edu.cn   
Yuzhi Huang   
SIGS, Tsinghua University   
Shenzhen, China   
yz\_huang13@163.com

Jinpeng Wang Harbin Institute of Technology, Shenzhen Shenzhen, China wangjp26@gmail.com

Jingyan Jiang<sup>†</sup>   
SIGS, Tsinghua University   
Shenzhen, China   
jiangjingyanjlu@gmail.com   
Shuzhao Xie   
SIGS, Tsinghua University   
Shenzhen, China   
xsz24@mails.tsinghua.edu.cn   
Zhi Wang<sup>†</sup>   
SIGS, Tsinghua University   
Shenzhen, China   
wangzhi@sz.tsinghua.edu.cn

## Abstract

Partially Relevant Video Retrieval (PRVR) seeks to retrieve untrimmed videos containing a moment that matches a text query, without temporal annotations. The relevant moment may last only seconds within a video spanning several minutes, creating an extremely low signal-to-noise ratio that makes PRVR more challenging than standard full-video retrieval. This task presents two intertwined challenges: (1) signal dilution, where coarse global representations blur the brief relevant signal into the dominant irrelevant surround ings; (2) curvature rigidity, where embedding all videos in the same fixed-geometry space distorts representations for videos that range from flat atomic events to deep compositional hierarchies. Existing PRVR methods have improved moment selection and cross-modal matching, but they still typically encode all videos in a single fixedcurvature retrieval space, limiting their ability to model diverse video structures. To address both challenges, we propose Curv-Spec, a framework that learns content-adaptive curvature for video retrieval representations rather than imposing a fixed geometric prior. CurvSpec processes features through parallel Euclidean and hyperbolic attention layers, with independently learned curvatures assigned to the hyperbolic layers, and a content-aware fusion mech anism routes each input to its most suitable geometric regime. To further suppress signal dilution, CurvSpec represents each video with semantic centroids whose number is determined by the video’s content complexity, projects them onto the learned manifold, and matches each query against its nearest centroid by geodesic distance. Experiments on ActivityNet Captions, TVR, and Charades STA demonstrate state-of-the-art retrieval performance.

## CCS Concepts

• Information systems → Multimedia information systems.

Keywords   
Video Retrieval, Multimodal Learning, Hyperbolic Attention, Hy  
perbolic Geometry, Semantic Centroids ACM Reference Format:   
Zhen Liu, Letian Li,Jinpeng Wang, Shuzhao Xie, Yuzhi Huang,JingyanJiang, and Zhi Wang. 2026. CurvSpec: Adaptive Multi-Curvature Learning for Partial Relevant Video Retrieval. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3767308. 3835167

## 1 Introduction

The rapid growth of online video has driven extensive research on text-to-video retrieval [8, 14, 20, 21, 24, 34, 36, 40, 41], a task that seeks to bridge the semantic gap between natural-language descriptions and visual content. Standard methods assume that a query describes the entire video, yet real-world untrimmed videos often contain content only partially relevant to a given query. Partially Relevant Video Retrieval (PRVR) [5] formalizes this more realistic setting: given a text query, retrieve untrimmed videos that contain at least one semantically matching moment. The relevant moment may last only seconds within a video spanning several minutes, creating an extremely low signal-to-noise ratio that distinguishes PRVR from standard full-video retrieval.

An important challenge in PRVR is signal dilution: coarse global representations blur the brief relevant signal into the domi nant irrelevant surroundings, making fine-grained, segment-aware matching essential. Existing methods [5, 37, 38] have made progress through improved moment selection and cross-modal matching strategies, yet they share a common assumption: the embedding space is geometrically adequate for all videos. The quality of any matching function, however, is bounded by the fidelity of its input representations, which depends not only on the encoder architecture but also on the geometric properties of the embedding space.

Among these geometric properties, we identify curvature rigidity, the practice of forcing all videos into the same fixed-geometry space regardless of their structural depth, as an underappreciated source of degradation. Real-world videos range from flat atomic events to deep multi-level hierarchies (Fig. 1a), and these structures have intrinsically diferent geometries: flat relationships are naturally accommodated in Euclidean space, while deep hierarchies are more faithfully captured by hyperbolic manifolds, whose exponential volume growth mirrors the branching structure of trees [29]. Hyperbolic geometry provides the volume needed for hierarchical videos but imposes spurious separation on flat ones; Euclidean geometry preserves uniform distances but compresses hierarchical depth. No single curvature resolves this tension, committing to one can distort the representations of videos it does not suit.

![](images/a3754f44d6851519ceda17a8bd3a74daf1451a2dabc81550d7b0aa9dce2d0476.jpg)  
Figure 1: Motivation and key ideas. (a) Videos exhibit diverse structures (Video 1 to 3, from flat to deep hierarchies). Neither Euclidean space nor a single fixed hyperbolic curvature accommodates all of them; CurvSpec routes each video to its optimal geometry. (b) Global pooling dilutes relevant signals into an ambiguous single vector; semantic centroids decompose the video into coherent regions, allowing the query to match the nearest relevant centroid free from irrelevant interference.

In PRVR, curvature rigidity can further exacerbate signal dilution. If the embedding geometry distorts distances around the relevant moment, that moment may rank lower than irrelevant segments that happen to sit in a geometrically favorable region (Fig. 1b). Recent hyperbolic methods [22, 39] show the value of non-Euclidean geometry for video retrieval, yet impose a single fixed curvature on all inputs, biasing every video toward the same geometric regime regardless of actual structural depth. Meanwhile, most methods still compress each video into a single global vector, conflating relevant moments with surrounding irrelevant content and leaving retrieval further exposed to segments sharing only surface-level similarity with the query.

These observations call for a framework that (i) adapts its embedding geometry to each video’s structural depth, and (ii) explicitly isolates relevant content from irrelevant interference. We address both in CurvSpec, built on two designs. (1) To overcome curvature rigidity (Fig. 1a), we introduce a learnable curvature spectrum: � Euclidean and � hyperbolic attention layers operating in parallel, with each hyperbolic layer using an independently learned curvature. A content-aware fusion mechanism weights each layer’s contribution according to the input, routing structurally simple videos toward Euclidean layers and hierarchically complex ones toward hyperbolic layers. The hyperbolic curvatures are optimized end-to-end, eliminating the need for manual geometric tuning.

(2) To combat signal dilution (Fig. 1b), we decompose each video into multiple semantic centroids, each capturing a coherent thematic region. The centroids are refined through cross-attention over frame-level and clip-level temporal representations, so each centroid reflects a semantically distinct segment rather than a blend of unrelated content. By projecting the refined centroids onto the learned hyperbolic manifold, retrieval reduces to matching a query against the nearest centroid, naturally suppressing interference from irrelevant content.

Extensive experiments on ActivityNet Captions, TVR, and Charades-STA demonstrate that CurvSpec achieves competitive performance at computational cost comparable to single-geometry baselines. Further analysis confirms that the learned curvatures diversify from identical initialization and allocate geometric capacity in proportion to structural complexity, with retrieval gains disproportionately concentrated on structurally complex videos.

Our contributions are as follows:

• We identify curvature rigidity as a distinct bottleneck in PRVR and address it with a learnable curvature spectrum of parallel Euclidean and hyperbolic attention layers, where each hyperbolic layer uses an independently optimized curvature and a content-aware mechanism routes each video to its most suitable geometric regime.

• We address signal dilution through semantic centroid decomposition, which represents video as a set of coherent thematic regions refined through cross-attention, reducing partial relevance matching to nearest-centroid retrieval that isolates relevant moments from irrelevant content.

• Extensive experiments on three benchmarks establish stateof-the-art retrieval performance, and analysis confirms that the learned curvatures allocate geometric capacity in proportion to structural complexity, with gains disproportionately concentrated on structurally complex videos.

## 2 Related Work

## 2.1 Partially Relevant Video Retrieval

Text-to-video retrieval (T2VR) [1, 10, 24, 36, 41, 45] typically assumes global video–query alignment, which is inadequate for untrim med videos containing only partially relevant content. Dong et al. [5] formalized this setting as Partially Relevant Video Retrieval (PRVR), where a query may match only a temporal segment rather than the entire video. Early PRVR methods alleviate this mismatch by generating multi-scale segment representations [5] or modeling temporal context with Gaussian-weighted attention [37, 38]. Recent work further improves partial matching through CLIP-based distillation [6], event-level alignment [15], uncertainty modeling [44], ambiguity-aware contrastive learning [3], and active or uneven event modeling [33, 46]. These methods have substantially advanced PRVR by locating or emphasizing more informative temporal regions. However, most still rely on global or fixed-geometry representations, which can dilute short relevant moments and fail to accommodate heterogeneous video structures. CurvSpec addresses these limitations by combining adaptive retrieval geometry with semantic centroid decomposition.

## 2.2 Hyperbolic Representation Learning

Hyperbolic spaces are well suited to hierarchical and tree-structured data because their exponential volume growth naturally preserves branching relations [12, 16, 26, 29, 32]. Several formulations have been developed for stable and discriminative hyperbolic learning, including Lorentzian embeddings and distance learning [18, 30]. These ideas have also been extended to multimodal representation learning, such as image–text hierarchy modeling [4], hierarchical video understanding [39], and compositional CLIP representa tions [31]. For video retrieval, hyperbolic geometry is attractive because videos often contain nested event structures, from atomic actions to multi-step activities. In PRVR, HLFormer [22] introduces hyperbolic geometry to encode such hierarchical video structures, but uses a single fixed curvature for all inputs. This fixed choice cannot adapt to videos with diferent structural depths. Our work instead learns a curvature spectrum and adaptively fuses multiple geometric regimes according to video content.

## 3 Method

## 3.1 Preliminaries

Problem formulation. Given a dataset $\mathcal { D } \ = \ \{ ( V _ { i } , T _ { i } ) \}$ , each untrimmed video�<sub>�</sub> is associated with text descriptions $T _ { i } = \{ t _ { i } ^ { 1 } , \ldots ,$ $t _ { i } ^ { m _ { i } } \}$ , where each $t _ { i } ^ { j }$ corresponds to a specific temporal moment within $V _ { i } .$ Temporal boundaries are not annotated. The goal of PRVR is to retrieve, for a given text query, videos containing a semantically matching moment, despite the query being relevant to only a fraction of the video content.

The Lorentz model. Our framework operates in a mixed geometry that combines Euclidean and hyperbolic spaces. We adopt the Lorentz model, in which the hyperboloid $\mathbb { H } _ { \kappa } ^ { \bar { d } } = \{ x \in \mathbb { R } ^ { d + 1 }$ $\langle x , x \rangle _ { \mathcal { L } } ~ = ~ - \frac { 1 } { \kappa } , ~ x _ { 0 } ~ > ~ 0 \}$ is equipped with the Minkowski inner product:

$$
\begin{array} { r } { \langle x , y \rangle _ { \mathcal { L } } ~ = ~ - x _ { 0 } y _ { 0 } + \sum _ { i = 1 } ^ { d } x _ { i } y _ { i } . } \end{array}\tag{1}
$$

The geodesic distance between two points on the manifold is:

$$
\begin{array} { r } { d _ { \kappa } ( x , y ) ~ = ~ \frac { 1 } { \sqrt { \kappa } } ~ \mathrm { a r c o s h } ~ \bigl ( - \kappa ~ \langle x , y \rangle _ { \mathcal { L } } \bigr ) . } \end{array}\tag{2}
$$

Transitioning between the tangent space $T _ { o } \mathbb { H } _ { \kappa } ^ { d }$ at the origin $o \ =$ $\textstyle { \bigl ( } { \frac { 1 } { \sqrt { \kappa } } } , 0 { \bigr ) }$ and the manifold is achieved via the exponential and logarithmic maps. The exponential map traces the geodesic from � in the direction of a tangent vector $v \in T _ { o } \mathbb { H } _ { \kappa } ^ { d }$ :

$$
\begin{array} { r } { \exp _ { o } ^ { \kappa } ( v ) \ = \ \cosh \Big ( \sqrt { \kappa } \ \lVert v \rVert _ { 2 } \Big ) \ o \ + \ \frac { 1 } { \sqrt { \kappa } } \ \sinh \Big ( \sqrt { \kappa } \ \lVert v \rVert _ { 2 } \Big ) \ \frac { v } { \lVert v \rVert _ { 2 } } , } \end{array}\tag{3}
$$

and its inverse recovers the tangent representation of a manifold point $x \in \mathbb { H } _ { \kappa } ^ { d }$

$$
\begin{array} { r l r } & { } & { \log _ { o } ^ { \kappa } ( x ) = d _ { \kappa } ( o , x ) \frac { x _ { \perp } } { \| x _ { \perp } \| _ { 2 } } , } \\ & { } & { x _ { \perp } = x + \kappa \langle o , x \rangle _ { \mathcal { L } } o , } \end{array}\tag{4}
$$

where $x _ { \perp }$ denotes the component of � orthogonal to � under the Minkowski metric. Curvature � governs the rate of exponential volume growth: larger � allocates more representational capacity to deep hierarchies, while � → 0 recovers flat Euclidean geometry. This geometric sensitivity motivates learning a distinct � per attention layer, allowing the model to discover a curvature spectrum that reflects the varied geometric demands ofthe data rather than relying on a single prescribed geometry.

## 3.2 Overall Architecture

As illustrated in Fig. 2a, CurvSpec follows a dual-stream encoding paradigm [22, 33, 37, 38]. Text queries and video content are independently encoded, then compared via multi-granularity similarity. Text encoding. A text query $T = \{ w _ { 1 } , \dots , w _ { S } \}$ is encoded by a frozen RoBERTa [23] to obtain $X _ { q } ~ \in ~ \mathbb { R } ^ { S \times d _ { q } }$ . These representations are linearly projected, augmented with learnable positional embeddings $P _ { \mathrm { t e x t } }$ , and refined by Euclidean self-attention [35]:

$$
Z _ { q } = \mathrm { A t t n } _ { \mathrm { E } } \big ( \mathrm { F C } ( X _ { q } ) + P _ { \mathrm { t e x t } } \big ) ,\tag{5}
$$

followed by attention-weighted pooling into a single query vector:

$$
\begin{array} { r } { q = \sum _ { s = 1 } ^ { S } a _ { s } z _ { q , s } , } \end{array}\tag{6}
$$

where $a _ { s } =$ softmax<sub>�</sub> $( w _ { q } ^ { \top } z _ { q , s } ) , w _ { q } \in \mathbb { R } ^ { d }$ is a learnable projection, and $\{ z _ { q , s } \}$ are the refined token representations.

Video encoding. An untrimmed video is represented at two temporal granularities: � frame-level features $\bar { X _ { f } } \in \mathbb { R } ^ { L \times d _ { f } }$ and � clip-level features $X _ { c } \in \mathbb { R } ^ { M \times d _ { c } }$ , both from a pre-trained visual encoder. Each branch is projected to a shared dimension �, augmented with positional embeddings, and processed by our CurvSpec block:

$$
\begin{array} { l } { { Z _ { f } = \mathbf { C u r v S p e c } \big ( \mathrm { F C } ( X _ { f } ) + P _ { f } \big ) , } } \\ { { Z _ { c } = \mathbf { C u r v S p e c } \big ( \mathrm { F C } ( X _ { c } ) + P _ { c } \big ) . } } \end{array}\tag{7}
$$

Similarity computation. The retrieval score combines maxpooled cosine similarities at both granularities:

$$
S ( q , V ) = \alpha \ \operatorname * { m a x } _ { i } \ \cos ( q , z _ { f , i } ) \ + \ ( 1 - \alpha ) \ \operatorname * { m a x } _ { i } \ \cos ( q , z _ { c , j } ) ,\tag{8}
$$

where $\alpha \in [ 0 , 1 ]$ is a balancing hyperparameter. This max-pooling strategy is particularly suited to the partial relevance setting, as it isolates the most semantically aligned moment in the video at each granularity rather than averaging over the full content, which would dilute the matching signal from the relevant segment.

![](images/0ce3e61c22572050588175127f2208cc733ce08ecfdadad8ec9456d3f8c95e71.jpg)  
Figure 2: Overview of CurvSpec. (a) Text queries and video frames/clips are encoded and processed by the CurvSpec block to yield enriched representations $\mathbf { Z } _ { c }$ and $\mathbf { { Z } } _ { f } .$ . Clip features are further decomposed into semantic centroids, which, together with the query, are projected onto the Lorentz manifold for fine-grained partial relevance matching. (b) The CurvSpec block processes input through parallel Euclidean and Lorentz attention branches, each at a distinct geometry. A content-aware fusion mechanism adaptively combines the multi-geometry representations.

## 3.3 CurvSpec Block

The CurvSpec block (Fig. 2b) is our core component. It processes input features through � Euclidean attention layers and � Lorentz attention layers, with each Lorentz layer using an independently learnable curvature, and fuses their outputs via a content-aware mechanism to capture flat relational structures and hierarchical semantics at multiple geometric scales.

Euclidean attention. Each Euclidean layer applies Gaussianweighted self-attention [38], with query, key, and value matrices obtained by learned projections from input $\dot { X } \in \mathbb { R } ^ { L \times d }$ . A proximity mask $G _ { \sigma } [ i , j ] = N ( j ; i , \sigma ^ { 2 } )$ gates the temporal receptive field:

$$
H _ { \sigma } = \mathrm { s o f t m a x } \left( G _ { \sigma } \odot { \frac { Q K ^ { \top } } { \sqrt { d } } } \right) V .\tag{9}
$$

The first Euclidean layer uses global attention without Gaussian mask, the remaining $N { - } 1$ layers use Gaussian widths $\sigma _ { n } = 2 ^ { n }$ for $n = 1 , \ldots , N { - } 1$ , yielding a multi-scale hierarchy of receptive fields. Lorentz attention. Each Lorentz layer operates on $\mathbb { H } _ { \kappa _ { n } } ^ { d }$ with its own learnable curvature $\kappa _ { n } .$ . Input features are linearly projected, scaled by $\gamma \leq 1$ for numerical stability, and lifted onto $\mathbb { H } _ { \kappa _ { n } } ^ { d }$ via the exponential map. Hyperbolic queries, keys, and values are computed via the Lorentz linear map, which maps to the tangent space, applies an afine transformation, and projects back:

$$
\mathrm { L L } ( x ) = \exp _ { o } ^ { \kappa } ( W \log _ { o } ^ { \kappa } ( x ) + b ) .\tag{10}
$$

Denoting the resulting row vectors as $q _ { i } , k _ { j } , v _ { i } ^ { h }$ , the attention logits, weights, and Einstein midpoint aggregation [18] yield:

$$
\begin{array} { r l r } & { a _ { i j } = \frac { \beta + \langle q _ { i } , k _ { j } \rangle _ { \mathcal { L } } } { \tau } , \quad } & { w _ { i j } = \frac { e ^ { a _ { i j } } } { \sum _ { j ^ { \prime } } e ^ { a _ { i j ^ { \prime } } } } , } \\ & { } & \\ & { \bar { v } _ { i } = \sum _ { j = 1 } ^ { L } w _ { i j } v _ { j } ^ { h } , \quad [ H _ { n } ] _ { i } = \frac { 1 } { \gamma } \log _ { o } ^ { \kappa _ { n } } \left( \frac { \bar { v } _ { i } } { \sqrt { | \langle \bar { v } _ { i } , \bar { v } _ { i } \rangle _ { \mathcal { L } } | } } \right) , } \end{array}\tag{11}
$$

where $[ H _ { n } ] _ { i }$ denotes the �-th row, � a learnable temperature, and � a constant bias centered at zero for coincident query–key pairs (�=1).

All curvatures are initialized identically $\left( \kappa _ { n } = 1 . 0 \right)$ and optimized independently. During training, each layer converges to a distinct stable curvature spanning a wide geometric range: layers handling shallow structures settle near lower curvatures, while those capturing deep hierarchies gravitate toward higher values. This emergent diversification, driven by the task objective, constitutes the learned curvature spectrum.

Content-aware fusion. The 2� branch outputs $\{ H _ { n } \} _ { n = 1 } ^ { 2 N }$ are combined via learned, input-dependent weights. A global content token first queries each geometry branch to obtain branch-aware descriptors, which are then mapped to token-wise fusion weights:

$$
\begin{array} { r l r } { g = \displaystyle \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \frac { 1 } { 2 N } \sum _ { n = 1 } ^ { 2 N } h _ { i , n } , } & { { } } & { r _ { n } = \mathrm { A t t n } ( g , H _ { n } , H _ { n } ) , } \\ { \pi _ { i , n } = \displaystyle \mathrm { s o f t m a x } _ { n } \left( \left[ W _ { 2 } \phi ( W _ { 1 } r _ { n } ) \right] _ { i } / \tau _ { f } \right) , } & { { } } & { o _ { i } = \sum _ { n = 1 } ^ { 2 N } \pi _ { i , n } h _ { i , n } . } \end{array}\tag{12}
$$

where $h _ { i , n }$ is the �-th token from branch $H _ { n } , g$ is the global content token, and $r _ { n }$ is the branch descriptor obtained by attending to $H _ { n }$ The attention weights $\pi _ { i , n }$ route each token toward its most suitable geometric representation. Since diferent segments within a partially relevant video may exhibit fundamentally diferent semantic structures, this token-level selection allows the model to apply the most appropriate geometry to each position independently, rather than committing the entire sequence to a single fixed geometry.

## 3.4 Semantic Centroid Decomposition

A central challenge in PRVR is that a query matches only a fraction of the video. Representing a video as a single global vector conflates relevant and irrelevant segments. We address this by decomposing each video into a set ofsemantic centroids, each capturing a coherent thematic region, with the number ofcentroids adapted to the video’s content complexity.

Content-adaptive centroid count. We quantify the semantic complexity of a video via the efective rank of its clip feature matrix $X _ { c } \in \mathbb { R } ^ { M \times { \bar { d } } }$

$$
r _ { \mathrm { e f f } } ( X _ { c } ) = \exp \left( - \sum _ { j } \bar { \sigma } _ { j } \log \bar { \sigma } _ { j } \right) , \quad \bar { \sigma } _ { j } = \frac { \sigma _ { j } ^ { 2 } } { \sum _ { l } \sigma _ { l } ^ { 2 } } ,\tag{13}
$$

where $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdots$ are the singular values of $X _ { c }$ in descending or der. $r _ { \mathrm { e f f } }$ measures the efective number of semantically independent directions in the clip feature space: small values indicate structurally simple videos dominated by a single theme, large values reflect multi-topic content with richer temporal diversity. We select $K _ { i }$ as the smallest value in $\{ 2 , 4 , 6 \}$ for which the top-�<sub>�</sub> singular values collectively explain at least a fraction � of the total variance:

$$
K _ { i } = \operatorname* { m i n } \left. K \in \left. 2 , 4 , 6 \right. \left| \ \frac { \sum _ { j = 1 } ^ { K } \sigma _ { j } ^ { 2 } } { \sum _ { j } \sigma _ { j } ^ { 2 } } \geq \theta \right. \right.\tag{14}
$$

If no $K \in \{ 2 , 4 , 6 \}$ satisfies the condition (rare in practice), $K _ { i }$ defaults to 6. The criterion is computed ofline from pre-extracted features, incurring no inference overhead. Crucially, � is a single dataset-agnostic threshold: the same value applied to diferent corpora naturally yields diferent � distributions, reflecting each corpus’s intrinsic content diversity without per-dataset calibration (Fig. 3).

Given clip features $Z _ { c }$ , K-means clustering [25] with $K _ { i }$ clusters yields initial prototypes $\left\{ \mu _ { k } \right\} _ { k = 1 } ^ { K _ { i } }$ . Clip features are used for initialization because they provide coarser, semantically stable summaries at the scene level; frame-level details are then incorporated during the cross-attention refinement step. Each prototype is then refined by cross-attending over both frame and clip features and fusing the results through a learned gate:

$$
c _ { k } = \sigma ( g _ { k } ) \odot \mathrm { A t t n } ( \mu _ { k } , Z _ { f } ) ~ + ~ \left( 1 - \sigma ( g _ { k } ) \right) \odot \mathrm { A t t n } ( \mu _ { k } , Z _ { c } ) ,\tag{15}
$$

where $\sigma ( g _ { k } )$ is a sigmoid gate that adaptively modulates the contribution of each granularity:

$$
g _ { k } \ = \ \mathrm { F C } \big ( [ \mathrm { A t t n } ( \mu _ { k } , Z _ { f } ) \| \mathrm { A t t n } ( \mu _ { k } , Z _ { c } ) ] \big ) ,\tag{16}
$$

with $[ \cdot | | \cdot ]$ denoting concatenation. The refined centroids are projected onto $\mathbb { H } _ { \kappa } ^ { d }$ via $\exp _ { o } ^ { \kappa }$ , and partial relevance is captured by the minimum geodesic distance to any centroid:

$$
d _ { \operatorname* { m i n } } ( q , V ) = \operatorname* { m i n } _ { k \in [ K _ { i } ] } \ d _ { \kappa } \bigl ( q _ { \mathcal { L } } , c _ { k , \mathcal { L } } \bigr ) ,\tag{17}
$$

where $q \_ c$ and $c _ { k , \mathscr { L } }$ denote the query and centroid projected onto the hyperboloid. This allows the model to match a query against only the most relevant semantic region of a video, directly addressing the partial relevance challenge without requiring any temporal boundary annotations.

For videos with multiple queries, we further encourage diverse query–centroid correspondences by minimizing the average nearestcentroid distance:

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { \vert \mathcal { V } \vert } \sum _ { V \in \mathcal { V } } \frac { 1 } { N _ { V } } \sum _ { i = 1 } ^ { N _ { V } } d _ { \mathrm { m i n } } ( q _ { i } , V ) ,\tag{18}
$$

where $N _ { V }$ is the number of queries associated with video � and $_ \textmd { ‰}$ denotes the subset of videos with at least two queries. Combined with the diversity loss ${ \mathcal { L } } _ { \mathrm { d i v } }$ (Section 3.5), this encourages each centroid to specialize in a distinct semantic region.

## 3.5 Training Objective

The overall objective unifies cross-modal matching in Euclidean space, centroid-based alignment in hyperbolic space, and geometric regularization:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { m a t c h } } + \lambda _ { h } \mathcal { L } _ { \mathrm { h y p } } + \lambda _ { r } \mathcal { L } _ { \mathrm { r e g } } .\tag{19}
$$

$\mathcal { L } _ { \mathrm { m a t c h } }$ aggregates bilateral NCE and triplet margin losses at both frame and clip granularities, encouraging the learned representations to rank positive query–video pairs above negatives. $\mathcal { L } _ { \mathrm { h y p } }$ operates on the semantic centroids. Its retrieval component follows an InfoNCE formulation over geodesic distances:

$$
\mathcal { L } _ { \mathrm { r e t r } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( - s \cdot d _ { \operatorname* { m i n } } ( q _ { i } , V _ { i } ^ { + } ) \right) } { \sum _ { j = 1 } ^ { B } \exp \left( - s \cdot d _ { \operatorname* { m i n } } ( q _ { i } , V _ { j } ) \right) } ,\tag{20}
$$

where $V _ { i } ^ { + }$ is the ground-truth video for query �<sub>�</sub> and � is a learnable scaling factor. The full term is ${ \mathcal { L } } _ { \mathrm { h y p } } = { \mathcal { L } } _ { \mathrm { r e t r } } + { \mathcal { L } } _ { \mathrm { a l i g n } } . { \mathcal { L } } _ { \mathrm { r e g } }$ combines a diversity term ${ \mathcal { L } } _ { \mathrm { d i v } }$ that prevents query embeddings from collapsing to nearby representations within a batch, and a curvature regularizer ${ \mathcal { L } } _ { \mathrm { c u r v } }$ that confines all pairwise gaps $\lvert \kappa _ { i } - \kappa _ { j } \rvert$ within a target band $[ \Delta _ { \mathrm { m i n } } , \Delta _ { \mathrm { m a x } } ]$ , preventing both geometric collapse and uncontrolled divergence of the curvature spectrum.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate our model on three widely used large-scale video datasets that span diverse content domains: ActivityNet Captions [17], TVR [19], and Charades-STA [11]. ActivityNet Captions was originally introduced for dense video captioning and has since become a standard benchmark for partially relevant video retrieval. It consists of approximately 20K YouTube videos with an average duration of around 118 seconds. TVR comprises around 21.8K video clips collected from six diferent TV shows, with an average duration of roughly 76 seconds per video. Each clip is associated with five natural language sentences that describe distinct moments within the video. Charades-STA includes $^ { 6 , 6 7 0 }$ videos and 16,128 sentence-level annotations. For all datasets, we follow the standard training and testing splits adopted in previous works.

Evaluation Strategy. Following prior work [22, 38, 44], we adopt rank-based evaluation metrics, including Recall at K (R@K, $\textrm { K } =$ 1,5,10,100). R@K measures the proportion of test queries whose corresponding items appear within the top K retrieved results. For comprehensive comparison, we also calculate the Sum of Recalls (SumR), which aggregates all R@K scores.

Implementation Details. Following standard practice, we use preextracted visual features: ResNet-152+I3D (3072-dim) for TVR [19], and I3D for ActivityNet and Charades [28, 43]. Text features are RoBERTa embeddings [23]: 768-dim for TVR, 1024-dim for ActivityNet and Charades [5].

Each CurvSpec block contains 2�=8 parallel attention layers: $N { = } 4$ Euclidean layers and �=4 Lorentz layers, with each Lorentz layer using an independently learnable curvature. Following the design strategy of GMMFormer [38], we employ multiple Gaussian widths to capture contextual dependencies at diferent scales, using one global layer and widths ranging from $2 ^ { 1 }$ to $2 ^ { N - 1 }$

<table><tr><td>Method</td><td colspan="5">ActivityNet Captions</td><td colspan="5">TVR</td><td colspan="5">Charades-STA</td></tr><tr><td></td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@100</td><td>SumR</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@100</td><td>SumR</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@100</td><td>SumR</td></tr><tr><td colspan="10">Text-to-Video Retrieval</td><td colspan="7"></td></tr><tr><td>HGR [1]</td><td>4.0</td><td>15.0</td><td>24.8</td><td>63.2</td><td>107.0</td><td>1.7</td><td>4.9</td><td>8.3</td><td>35.2</td><td>50.1</td><td>1.2</td><td>3.8</td><td>7.3</td><td>33.4</td><td>45.7</td></tr><tr><td> $\mathrm { D E } { + } { + } \left[ \bar { 7 } \right]$ </td><td>5.3</td><td>18.4</td><td>29.2</td><td>68.0</td><td>121.0</td><td>8.8</td><td>21.9</td><td>30.2</td><td>67.4</td><td>128.3</td><td>1.7</td><td>5.6</td><td>9.6</td><td>37.1</td><td>54.1</td></tr><tr><td>RIVRL [8]</td><td>5.2</td><td>18.0</td><td>28.2</td><td>66.4</td><td>117.8</td><td>9.4</td><td>23.4</td><td>32.2</td><td>70.6</td><td>135.6</td><td>1.6</td><td>5.6</td><td>9.4</td><td>37.7</td><td>54.3</td></tr><tr><td>CLIP4Clip [24]</td><td>5.9</td><td>19.3</td><td>30.4</td><td>71.6</td><td>127.3</td><td>9.9</td><td>24.3</td><td>34.3</td><td>72.5</td><td>141.0</td><td>1.8</td><td>6.5</td><td>10.9</td><td>44.2</td><td>63.4</td></tr><tr><td>Cap4Video [40]</td><td>6.3</td><td>20.4</td><td>30.9</td><td>72.6</td><td>130.2</td><td>10.3</td><td>26.4</td><td>36.8</td><td>74.0</td><td>147.5</td><td>1.9</td><td>6.7</td><td>11.3</td><td>45.0</td><td>65.0</td></tr><tr><td colspan="10">Video Corpus Moment Retrieval (w/o moment localization)</td><td colspan="7"></td></tr><tr><td>XML [19]</td><td>5.3</td><td>19.4</td><td>30.6</td><td>73.1</td><td></td><td></td><td></td><td>38.1</td><td>80.3</td><td>157.1</td><td>1.6</td><td></td><td></td><td></td><td>64.6</td></tr><tr><td>ReLoCLNet [43]</td><td>5.7</td><td>18.9</td><td>30.0</td><td>72.0</td><td>128.4 126.6</td><td>10.7 10.0</td><td>28.1 26.5</td><td>37.3</td><td>81.3</td><td>155.1</td><td>1.2</td><td>6.0 5.4</td><td>10.1 10.0</td><td>46.9 45.6</td><td>62.3</td></tr><tr><td>CONQUER [13]</td><td>6.5</td><td>20.4</td><td>31.8</td><td>74.3</td><td>133.1</td><td>11.0</td><td>28.9</td><td>39.6</td><td>81.3</td><td>160.8</td><td>1.8</td><td>6.3</td><td>10.3</td><td>47.5</td><td>66.0</td></tr><tr><td>JSG [2]</td><td>6.8</td><td>22.7</td><td>34.8</td><td>76.1</td><td>140.5</td><td></td><td></td><td></td><td></td><td></td><td>2.4</td><td>7.7</td><td>12.8</td><td>49.8</td><td>72.7</td></tr><tr><td colspan="10">Partially Relevant Video Retrieval</td><td colspan="7"></td></tr><tr><td>MS-SL [5]</td><td>7.1</td><td>22.5</td><td>34.7</td><td>75.8</td><td>140.1</td><td>13.5</td><td>32.1</td><td>43.4</td><td>83.4</td><td>172.4</td><td>1.8</td><td>7.1</td><td>11.8</td><td>47.7</td><td>68.4</td></tr><tr><td>PEAN [15]</td><td>7.4</td><td>23.0</td><td>35.5</td><td>75.9</td><td>141.8</td><td>13.5</td><td>32.8</td><td>44.1</td><td>83.9</td><td>174.2</td><td>2.7</td><td>8.1</td><td>13.5</td><td>50.3</td><td>74.7</td></tr><tr><td>LH [9]</td><td>7.4</td><td>23.5</td><td>35.8</td><td>75.8</td><td>142.4</td><td>13.2</td><td>33.2</td><td>44.4</td><td>85.5</td><td>176.3</td><td>2.1</td><td>7.5</td><td>12.9</td><td>50.1</td><td>72.7</td></tr><tr><td>GMMFormer [38]</td><td>8.3</td><td>24.9</td><td>36.7</td><td>76.1</td><td>146.0</td><td>13.9</td><td>33.3</td><td>44.5</td><td>84.9</td><td>176.6</td><td>2.1</td><td>7.8</td><td>12.5</td><td>50.6</td><td>72.9</td></tr><tr><td>ARL [3]</td><td>8.3</td><td>24.6</td><td>37.4</td><td>78.0</td><td>148.3</td><td>15.6</td><td>36.3</td><td>47.7</td><td>86.3</td><td>185.9</td><td>一</td><td>一</td><td>一</td><td>一</td><td>1</td></tr><tr><td>MamFusion [42]</td><td>8.0</td><td>25.4</td><td>37.2</td><td>76.8</td><td>147.4</td><td>14.2</td><td>33.9</td><td>44.9</td><td>84.5</td><td>177.5</td><td>一</td><td>一</td><td>1</td><td>一</td><td>一</td></tr><tr><td>ProtoPRVR [27]</td><td>7.9</td><td>24.9</td><td>37.2</td><td>77.4</td><td>147.4</td><td>15.4</td><td>35.9</td><td>47.5</td><td>86.3</td><td>185.1</td><td>1</td><td>1</td><td>一</td><td></td><td>一</td></tr><tr><td>HLFormer [22]]</td><td>8.7</td><td>27.1</td><td>40.1</td><td>79.0</td><td>154.9</td><td>15.7</td><td>37.1</td><td>48.5</td><td>86.4</td><td>187.7</td><td>2.6</td><td>8.5</td><td>13.7</td><td>54.0</td><td>78.7</td></tr><tr><td>CurvSpec (Ours)</td><td>9.0</td><td>27.6</td><td>39.9</td><td>79.1</td><td>155.6</td><td>15.9</td><td>37.5</td><td>49.0</td><td>86.7</td><td>189.1</td><td>2.6</td><td>8.9</td><td>14.3</td><td>53.3</td><td>79.1</td></tr></table>

Table 1: Retrieval performance on ActivityNet Captions, TVR, and Charades-STA. Bold and underline denote the best and second-best results, respectively. “–” indicates unavailable results.

![](images/655b365485187bc8f51b481bf27d5b3c4a39ee1a7aac13ef551dbf9b0542f99b.jpg)

![](images/ede29662961835a5fb34ed45acd2bc4b85f8866a8e3ecc6afc3eb3fcb9551bd7.jpg)  
Assigned K = 2 Assigned K = 4 Assigned K = 6

![](images/2a6fc25ce0d6455d17146019509ea5fc76ba4581e0ead6d4a79a28f9449ef60a.jpg)  
Figure 3: Adaptive centroid count $K _ { i }$ across datasets. Stacked histograms of per-video efective rank $r _ { \mathrm { e f f } }$ (Eq. (13)), coloured by the assigned $K _ { i } \in \{ 2 , 4 , 6 \}$

For semantic centroid initialization, we employ K-means with the content-adaptive �<sub>�</sub> defined in Section 3.4. The variance-explanation threshold is set to �=0.85, selected on the Charades-STA validation set. As shown in Fig. 3, this single threshold produces naturally diferent � distributions across datasets, with structurally simpler corpora (e.g., TVR) predominantly assigned �=2 and more diverse ones (ActivityNet, Charades-STA) receiving higher values, confirming that the criterion captures intrinsic content diversity without per-dataset tuning.

For the training configurations, we train the model using the Adam optimizer with a batch size of 128. All experiments are implemented in PyTorch and conducted on NVIDIA RTX 3090 GPU.

<table><tr><td>Method</td><td>Params</td><td>FLOPs</td><td>Latency†</td><td>R@1</td><td>SumR</td></tr><tr><td>GMMFormer</td><td>11.0M</td><td>1.63G</td><td>2.58ms</td><td>13.9</td><td>176.6</td></tr><tr><td>HLFormer</td><td>28.0M</td><td>4.73G</td><td>2.90ms</td><td>15.7</td><td>187.7</td></tr><tr><td>CurvSpec</td><td>28.8M</td><td>4.74G</td><td>2.96ms</td><td>15.9</td><td>189.1</td></tr></table>

Table 2: Eficiency-accuracy tradeof on TVR (RTX 3090). <sup>†</sup>Online query latency (ms/query).

## 4.2 Comparison with Prior Works

Baselines. For the PRVR task, we compare with eight representative methods: MS-SL [5], PEAN [15], LH [9], GMMFormer [38], ARL [3], MamFusion [42], ProtoPRVR [27], and HLFormer [22]. We further compare with models from related tasks: for T2VR, HGR [1], DE++ [7], RIVRL [8], CLIP4Clip [24], and Cap4Video [40]; for VCMR, XML [19], ReLoCLNet [43], CONQUER [13], and JSG [2].

Performance Comparison. As shown in Table 1, CurvSpec achieves the best performance on the majority of metrics across all three benchmarks, with especially consistent gains in aggregate SumR. CurvSpec also outperforms recent PRVR methods such as MamFusion and ProtoPRVR under the same I3D-based protocol. In particular, CurvSpec ranks first in SumR on all datasets, demonstrating the combined efectiveness of our learnable curvature spectrum and semantic centroid decomposition for partial relevance modeling.

Eficiency Analysis. Table 2 reports the eficiency-accuracy tradeof on TVR. Despite introducing mixed-geometry attention and semantic centroid decomposition, CurvSpec maintains comparable FLOPs and online query latency to existing methods, while achieving the best retrieval accuracy. The adaptive-� selection and K-means initialization are performed ofline, so they do not afect online retrieval latency. This confirms the geometric operations in CurvSpec add negligible computational overhead, making the approach deployable in real-time retrieval scenarios.

<table><tr><td rowspan="2">Configuration</td><td colspan="6">ActivityNet Captions</td><td colspan="4">TVR</td><td colspan="4">Charades-STA</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@100</td><td>SumR</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@100</td><td>SumR R@1</td><td>R@5</td><td>R@10</td><td>R@100</td><td>SumR</td></tr><tr><td colspan="10">Geometric Modeling Strategy</td><td colspan="7"></td></tr><tr><td>Euclidean Only</td><td>8.3</td><td>26.2</td><td>38.5</td><td>78.5</td><td>151.5</td><td>14.9</td><td>36.4</td><td>47.7</td><td>86.2</td><td>185.2</td><td>2.4</td><td>8.1</td><td>13.5</td><td>52.1</td><td>76.1</td></tr><tr><td>Fixed Curvature</td><td>8.4</td><td>26.9</td><td>39.6</td><td>78.6</td><td>153.5</td><td>15.4</td><td>37.1</td><td>48.4</td><td>86.2</td><td>186.9</td><td>2.4</td><td>8.3</td><td>13.7</td><td>52.8</td><td>77.2</td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td>Aggregation Strategy Tangent Space Sum</td><td>8.6</td><td>26.9</td><td>39.2</td><td>78.1</td><td>152.8</td><td>15.3</td><td>37.0</td><td>48.5</td><td>86.0</td><td>186.8</td><td>2.3</td><td>8.4</td><td>13.7</td><td>52.9</td><td>77.3</td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td>Partial Relevance Modeling</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">w/o Semantic Centroids</td><td colspan="7">184.1 2.3 8.5 13.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Feature Granularity w/o Frame Features</td><td>8.0</td><td>24.8</td><td>37.1</td><td>76.9</td><td>146.8</td><td>14.1</td><td>34.0</td><td>44.9</td><td>85.1</td><td>178.1</td><td>2.1</td><td>8.3</td><td>14.0</td><td>52.1</td><td>76.5</td></tr><tr><td colspan="10">w/o Clip Features</td><td colspan="7">1.9 8.0</td></tr><tr><td></td><td></td><td>23.0</td><td></td><td>75.4</td><td>141.3</td><td>12.6</td><td>31.8</td><td>43.3</td><td>83.4</td><td>171.1</td><td></td><td></td><td>13.5</td><td>50.1</td><td>73.5</td></tr><tr><td colspan="10">Auxiliary Losses</td><td colspan="7"></td></tr><tr><td>w/o  $\underline { { \tilde { \mathcal { L } } } } _ { \mathrm { r e g } }$  w/o</td><td>8.7</td><td>27.4</td><td>39.5</td><td>78.3 77.9</td><td>153.9</td><td>15.4</td><td>35.8</td><td>47.6</td><td>86.5</td><td>185.3</td><td>2.4</td><td>8.5</td><td>14.0</td><td>52.5</td><td>77.4</td></tr><tr><td> $\mathcal { L } _ { \mathrm { d i v } }$ </td><td>8.6</td><td>27.0</td><td>39.4</td><td></td><td>152.9</td><td>15.4</td><td>36.0</td><td>47.9</td><td>86.1</td><td>185.4</td><td>2.2</td><td>8.5</td><td>14.0</td><td>52.2</td><td>76.9</td></tr></table>

Table 3: Ablation study on ActivityNet Captions, TVR, and Charades-STA. Each group isolates one design dimension.

![](images/0708df25458754b8c456baf33390886392e937ca7ceaa79472a95a27ccde03b8.jpg)  
(a) Geometry usage distribution

![](images/040243298c3fe80493648ba9acaae5219b17e8ffae6a418d1d54c216d4438ddc.jpg)  
(b) Curvature distribution  
Figure 4: Adaptive geometric selection. (a) Videos with higher complexity utilize more hyperbolic geometry. (b) The efective curvature $\kappa _ { \mathrm { e f f } }$ increases with video complexity.

## 4.3 Ablation Study

We conduct systematic ablation studies on all three benchmarks to isolate and evaluate the contribution of each individual component. Results are presented in Table 3.

Geometric Modeling Strategy. We compare three variants: (i) replacing all Lorentz attention layers with standard Euclidean attention, (ii) using a single fixed curvature across all hyperbolic layers, and (iii) our learnable curvature spectrum. The learnable spectrum consistently outperforms both alternatives, confirming that contentadaptive geometric biases provide richer representational capacity than any single fixed geometry.

Aggregation Strategy. In our Lorentz attention, values are aggregated via the Einstein midpoint directly on the hyperboloid. We compare this against an alternative that maps values to the tangent space, performs weighted summation, and projects back. The Einstein midpoint yields consistent improvements across all datasets, as tangent space operations introduce geometric distortions that accumulate in high-curvature regions, progressively undermining the fidelity of the resulting hyperbolic representations.

![](images/65cf176c1b25ae49be4edadb7f6382c5ea78defb9ef0fb6f2e49d48c77339d06.jpg)  
Figure 5: Evolution of learned curvatures during training. Diferent attention layers converge to distinct stable values, demonstrating the emergence of a geometric spectrum.

Partial Relevance Modeling. Removing semantic centroids and reverting to globally pooled video representations leads to consistent and notable performance drops across all three datasets. This validates that decomposing videos into multiple semantic regions enables more precise query-to-segment matching.

Feature Granularity. Both frame-level and clip-level features contribute to the full model, with clip features proving more critical overall, as removing them causes noticeably larger degradation. This supports our coarse-to-fine design where clip features capture global semantic context and frame features encode fine-grained temporal dynamics.

Auxiliary Losses. Both $\mathcal { L } _ { \mathrm { r e g } }$ and ${ \mathcal { L } } _ { \mathrm { d i v } }$ contribute to the final performance. While their individual metric gains appear modest, $\mathcal { L } _ { \mathrm { r e g } }$ plays a crucial role in preventing curvature collapse during training, and ${ \mathcal { L } } _ { \mathrm { d i v } }$ encourages discriminative query embeddings.

![](images/28f767bc8433d98fe061c8a91bf994ff497b177a898f2b00d5673a27617ef03d.jpg)  
Figure 6: t-SNE visualization of query and video embeddings for three video examples from TVR.

<table><tr><td colspan="3">Simple</td><td colspan="2">Medium</td><td colspan="2">Complex</td></tr><tr><td>Variant</td><td>| R@1 SumR |</td><td></td><td>|R@1 SumR</td><td></td><td></td><td>|R@1 SumR</td></tr><tr><td>Euclidean Only</td><td>17.2</td><td>191.5</td><td>15.1</td><td>186.0</td><td>11.8</td><td>175.8</td></tr><tr><td>Fixed Curvature</td><td>17.4</td><td>192.0</td><td>15.6</td><td>187.8</td><td>12.5</td><td>178.4</td></tr><tr><td>CurvSpec</td><td>17.6</td><td>192.8</td><td>16.0</td><td>189.6</td><td>13.6</td><td>183.2</td></tr><tr><td>∆ (vs. Euclidean)</td><td>+0.4</td><td>+1.3</td><td>+0.9</td><td>+3.6</td><td>+1.8</td><td>+7.4</td></tr></table>

Table 4: Per-complexity retrieval performance on TVR.

## 4.4 Analysis

Learned Curvature Diversity. Fig. 5 tracks the curvature parameters of each Lorentz attention layer across training epochs. Starting from identical initialization, all layers converge to distinct and stable values that span a wide geometric range, showing that the model discovers a heterogeneous curvature spectrum instead of collapsing to a single geometry. This diversification emerges from the retrieval objective without explicit geometric supervision, suggesting that diferent types of semantic relationships inherently favor diferent curvature regimes.

Content-Aware Geometric Adaptation. We examine how the learned spectrum is applied to videos of varying structural complexity. As shown in Fig. 4(a), the adaptive fusion mechanism assigns progressively greater weight to hyperbolic layers as video complexity increases, while simpler videos are predominantly processed in near-Euclidean geometry. Correspondingly, Fig. 4(b) reveals that the efective curvature $\kappa _ { \mathrm { e f f } }$ exhibits a clear positive correlation with complexity, consistent with the principle of content-aware geometric allocation. This behavior aligns with the theoretical motivation that hyperbolic spaces, through exponential volume growth, are better suited for encoding deep hierarchies, while flat Euclidean geometry sufices for shallow semantic structures.

Per-Complexity Performance Breakdown. To validate that this adaptive behavior translates to tangible retrieval gains, we stratify the TVR test set by video complexity and compare geometric modeling variants in Table 4. We define a video complexity score C by combining three clip-level statistics: temporal feature variance $\sigma _ { t } ^ { 2 } ,$ pairwise cosine distance diversity $\sigma _ { d } ,$ and efective rank $r _ { \mathrm { e f f } }$ (Eq. (13)). Videos are stratified into Simple $( C \le \mathrm { { P } } 2 5 )$ , Medium, and Complex (C > P75) groups. While CurvSpec yields consistent improvements across all groups, the gains are disproportionately concentrated on structurally complex videos, where the SumR improvement is over 5× larger than on simple ones. This provides direct evidence that learnable curvature allocates stronger geometric biases precisely where hierarchical depth demands richer modeling capacity.

Semantic Centroid Visualization. Fig. 6 visualizes the embedding space via t-SNE. With semantic centroids (right panel), query embeddings cluster tightly around their matched video regions, forming well-separated and compact groups that enable fine-grained partial relevance matching. In contrast, the global-pooling baseline (left panel) produces scattered query embeddings far from the coarse video representations, reflecting the inherent dificulty of aligning a specific text description against an entire untrimmed video. Fig. 3 further confirms that the adaptive criterion assigns richer decompositions to structurally complex videos and fewer centroids to uniform ones, achieving fine-grained matching with negligible additional overhead.

## 5 Conclusion

We presented CurvSpec, a framework for partially relevant video retrieval that combines a learnable curvature spectrum with semantic centroid decomposition. The curvature spectrum processes features across multiple Euclidean and hyperbolic attention layers whose curvatures are optimized end-to-end, while the semantic centroids decompose each video into coherent regions for fine-grained matching. Experiments on three benchmarks show state-of-theart performance with comparable eficiency to single-geometry baselines. Our work highlights a broader insight for video retrieval that partial relevance should not be treated only as a problem of finding the right temporal evidence, it also depends on whether the retrieval space has the right geometry for the video’s semantic structure. Learning curvature from content ofers a principled way to align representation geometry with the heterogeneous structures of real-world videos.

## Acknowledgments

This study was conducted entirely during Zhen Liu’s internship at Tsinghua University and was supported by the National Natural Science Foundation of China (Grant Nos. 92467204 and 62472249), the Shenzhen Science and Technology Program (Grant Nos. KJZD 20240903102300001 and JCYJ20250604145014018), the Natural Science Foundation for Top Talents of Shenzhen Technology University (Grant No. GDRC202413), and the National Natural Science Foundation of China under Grant 624B2088.

## References

[1] Shizhe Chen, Yida Zhao, Qin Jin, and Qi Wu. 2020. Fine-Grained Video-Text Retrieval With Hierarchical Graph Reasoning. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020), 10635–10644.

[2] Zhiguo Chen, Xun Jiang, Xing Xu, Zuo Cao, Yijun Mo, and Heng Tao Shen. 2023. Joint Searching and Grounding: Multi-Granularity Video Content Retrieval. In Proceedings ofthe 31st ACM International Conference on Multimedia. Association for Computing Machinery, New York, NY, USA.

[3] Cheol-Ho Cho, WonJun Moon, Woojin Jun, MinSeok Jung, and Jae-Pil Heo. 2025. Ambiguity-Restrained Text-Video Representation Learning for Partially Relevant Video Retrieval. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 2500–2508.

[4] Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Ramakr ishna Vedantam. 2023. Hyperbolic Image-Text Representations. In Proceedings of the International Conference on Machine Learning.

[5] Jianfeng Dong, Xianke Chen, Minsong Zhang, Xun Yang, Shujie Chen, Xirong Li, and Xun Wang. 2022. Partially Relevant Video Retrieval. In Proceedings ofthe 30th ACM International Conference on Multimedia.

[6] Jianfeng Dong, Lei Huang, Daizong Liu, Xianke Chen, Xun Yang, Changting Lin, Xun Wang, and Meng Wang. 2025. Dual Learning with Dynamic Knowledge Distillation and Soft Alignment for Partially Relevant Video Retrieval. arXiv:2510.12283 [cs.CV]

[7] Jianfeng Dong, Xirong Li, Chaoxi Xu, Xun Yang, Gang Yang, Xun Wang, and Meng Wang. 2022. Dual Encoding for Video Retrieval by Text. IEEE Trans. Pattern Anal. Mach. Intell. (2022).

[8] Jianfeng Dong, Yabing Wang, Xianke Chen, Xiaoye Qu, Xirong Li, Yuan He, and Xun Wang. 2022. Reading-Strategy Inspired Visual Representation Learning for Text-to-Video Retrieval. IEEE Transactions on Circuits and Systems for Video Technology 32 (08 2022), 5680–5694. doi:10.1109/TCSVT.2022.3150959

[9] Sheng Fang, Tiantian Dang, and Shuhui Wang. 2024. Linguistic Hallucination for Text-Based Video Retrieval. IEEE Transactions on Circuits and Systems for Video Technology PP (10 2024), 1–1.

[10] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. 2020. Multimodal Transformer for Video Retrieval. In Computer Vision – ECCV2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV. Springer Verlag, Berlin, Heidelberg.

[11] J. Gao, Chen Sun, Zhenheng Yang, and Ramakant Nevatia. 2017. TALL: Temporal Activity Localization via Language Query. 2017 IEEE International Conference on Computer Vision (ICCV) (2017), 5277–5285.

[12] Neil He, Hiren Madhu, Ngoc Bui, Menglin Yang, and Rex Ying. 2025. Hyperbolic Deep Learning for Foundation Models: A Survey. Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (2025)

[13] Zhijian Hou, Chong-Wah Ngo, and W. K. Chan. 2021. CONQUER: Contextual Query-aware Ranking for Video Corpus Moment Retrieval. In Proceedings ofthe 29th ACM International Conference on Multimedia. Association for Computing Machinery.

[14] Sarah Ibrahimi, Xiaohang Sun, Pichao Wang, Amanmeet Garg, Ashutosh Sanan, and Mohamed Omar. 2023. Audio-enhanced text-to-video retrieval using textconditioned feature alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12054–12064.

[15] Xun Jiang, Zhiguo Chen, Xing Xu, Fumin Shen, Zuo Cao, and Xunliang Cai. 2023. Progressive Event Alignment Network for Partial Relevant Video Retrieval. 2023 IEEE International Conference on Multimedia and Expo (ICME) (2023).

[16] Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky. 2020. Hyperbolic Image Embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

[17] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. 2017. Dense-Captioning Events in Videos . In 2017 IEEE International Conference on Computer Vision (ICCV). IEEE Computer Society, Los Alamitos, CA, USA, 706–715.

[18] Marc Law, Renjie Liao, Jake Snell, and Richard Zemel. 2019. Lorentzian Distance Learning for Hyperbolic Representations. In Proceedings ofthe 36th International Conference on Machine Learning. 3672–3681.

[19] Jie Lei, Licheng Yu, Tamara L. Berg, and Mohit Bansal. 2020. TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval. In Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXI (Glasgow, United Kingdom). Springer-Verlag, Berlin, Heidelberg

[20] Jun Li, Peifeng Lai, Xuhang Lou, Jinpeng Wang, Yuting Wang, Ke Chen, Yaowei Wang, and Shu-Tao Xia. 2026. Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval. In Forty-third International Conference on Machine Learning.

[21] Jun Li, Xuhang Lou, Jinpeng Wang, Yuting Wang, Yaowei Wang, Shu-Tao Xia, and Bin Chen. 2026. Imagine Before Concentration: Difusion-Guided Registers Enhance Partially Relevant Video Retrieval. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9710–9721.

[22] Jun Li, Jinpeng Wang, Chaolei Tan, Niu Lian, Long Chen, Yaowei Wang, Min Zhang, Shu-Tao Xia, and Bin Chen. 2025. Enhancing Partially Relevant Video Retrieval with Hyperbolic Learning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV).

[23] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692 (2019).

[24] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. 2022. CLIP4Clip: An empirical study of CLIP for end to end video clip retrieval and captioning. Neurocomputing 508 (2022), 293–304. doi:10.1016/j.neucom.2022. 07.028

[25] J. MacQueen. 1967. Some Methods for Classification and Analysis of Multivariate Observations. In Proceedings ofthe Fifth Berkeley Symposium on Mathematical Statistics and Probability, Vol. 1. University of California Press, 281–297.

[26] Pascal Mettes, Mina Atigh, Martin Keller-Ressel, Jefrey Gu, and Serena Yeung. 2024. Hyperbolic Deep Learning in Computer Vision: A Survey. International Journal ofComputer Vision 132 (03 2024), 1–25.

[27] WonJun Moon, Cheol-Ho Cho, Woojin Jun, Minho Shim, Taeoh Kim, Inwoong Lee, Dongyoon Wee, and Jae-Pil Heo. 2025. Prototypes are Balanced Units for Eficient and Efective Partially Relevant Video Retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 21789–21799.

[28] Jonghwan Mun, Minsu Cho, and Bohyung Han. 2020. Local-Global Video-Text Interactions for Temporal Grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 10807–10816. doi:10.1109/ CVPR42600.2020.01082

[29] Maximilian Nickel and Douwe Kiela. 2017. Poincaré embeddings for learning hierarchical representations. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems (Long Beach, California, USA) (NIPS’17). Curran Associates Inc., Red Hook, NY, USA, 6341–6350.

[30] Maximilian Nickel and Douwe Kiela. 2018. Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry. In Proceedings ofthe 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, Jennifer G. Dy and Andreas Krause (Eds.). PMLR.

[31] Avik Pal, Max van Spengler, Guido Maria D’Amely di Melendugno, Alessandro Flaborea, Fabio Galasso, and Pascal Mettes. 2025. Compositional Entailment Learning for Hyperbolic Vision-Language Models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025.

[32] Wei Peng, Tuomas Varanka, Abdelrahman Mostafa, Henglin Shi, and Guoying Zhao. 2022. Hyperbolic Deep Neural Networks: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 12 (2022), 10023–10044.

[33] Peipei Song, Long Zhang, Long Lan, Weidong Chen, Dan Guo, Xun Yang, and Meng Wang. 2025. Towards Eficient Partially Relevant Video Retrieval With Active Moment Discovering. IEEE Transactions on Multimedia PP (01 2025), 1–12.

[34] Kaibin Tian, Ruixiang Zhao, Zijie Xin, Bangxiang Lan, and Xirong Li. 2024. Holistic features are almost suficient for text-to-video retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 17138–17147.

[35] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA.

[36] Xiaohan Wang, Linchao Zhu, Zhedong Zheng, Mingliang Xu, and Yi Yang. 2023. Align and Tell: Boosting Text-Video Retrieval With Local Alignment and Fine Grained Supervision. Trans. Multi. 25 (2023).

[37] Yuting Wang, Jinpeng Wang, Bin Chen, Tao Dai, Ruisheng Luo, and Shu-Tao Xia. 2024. GMMFormer v2: An Uncertainty-aware Framework for Partially Relevant Video Retrieval. arXiv:2405.13824

[38] Yuting Wang, Jinpeng Wang, Bin Chen, Ziyun Zeng, and Shu-Tao Xia. 2024. GMMFormer: Gaussian-Mixture-Model Based Transformer for Eficient Partially Relevant Video Retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence.

[39] Jun Wen, Yufeng Chen, Ruiqi Shi, Wei Ji, Menglin Yang, Difei Gao, Junsong Yuan, and Roger Zimmermann. 2025. HOVER: Hyperbolic Video-Text Retrieval. IEEE Transactions on Image Processing (2025), 6192–6203.

[40] Wenhao Wu, Haipeng Luo, Bo Fang, Jingdong Wang, and Wanli Ouyang. 2023. Cap4Video: What Can Auxiliary Captions Do for Text-Video Retrieval?. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10704–10713. doi:10.1109/CVPR52729.2023.01031

[41] Wenhao Wu, Xiaohan Wang, Haipeng Luo, Jingdong Wang, Yi Yang, and Wanli Ouyang. 2024. Cap4Video++: Enhancing Video Understanding with Auxiliary Captions. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

[42] Xinru Ying, Jiaqi Mo, Jingyang Lin, Canghong Jin, Fangfang Wang, and Lina Wei. 2025. MamFusion: Multi-Mamba with Temporal Fusion for Partially Relevant Video Retrieval. In 2025 IEEE International Conference on Multimedia and Expo (ICME). 1–6. doi:10.1109/ICME59968.2025.11210166

[43] Hao Zhang, Aixin Sun, Wei Jing, Guoshun Nan, Liangli Zhen, Joey Tianyi Zhou, and Rick Siow Mong Goh. 2021. Video Corpus Moment Retrieval with Con trastive Learning. In Proceedings ofthe 44th International ACM SIGIR Conference

on Research and Development in Information Retrieval. Association for Computing Machinery, New York, NY, USA, 685–695. doi:10.1145/3404835.3462874

[44] Long Zhang, Peipei Song, Jianfeng Dong, Kun Li, and Xun Yang. 2025. Enhancing Partially Relevant Video Retrieval with Robust Alignment Learning. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.).

[45] Shuai Zhao, Linchao Zhu, Xiaohan Wang, and Yi Yang. 2022. CenterCLIP: Token Clustering for Eficient Text-Video Retrieval. In Proceedings ofthe 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 970–981. doi:10.1145/3477495.3531950

[46] Sa Zhu, Huashan Chen, Wanqian Zhang, Jinchao Zhang, Zexian Yang, Xiaoshuai Hao, and Bo Li. 2025. Uneven Event Modeling for Partially Relevant Video Retrieval. In 2025 IEEE International Conference on Multimedia and Expo (ICME). 1–6.