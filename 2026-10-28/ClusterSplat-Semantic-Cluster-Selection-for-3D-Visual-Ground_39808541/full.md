# ClusterSplat: Semantic Cluster Selection for 3D Visual Grounding in Gaussian Splatting

Jaesung Lee Hyeontaek Hwang Gabriel Manalu Daeyoung Kim School of Computing, KAIST {ljs6017, hwanght, gabrieldavin23, kimd}@kaist.ac.kr

![](images/50e8809deb78ccae58d66bb2001094bf76d54baf0d731173a56c53495559daf3.jpg)  
Figure 1: Primitive-level filtering versus cluster-level selection. Given a referring expression, primitive score fields can be unstable, and top-ratio filtering may scatter the 3D support across the target and nearby distractors. ClusterSplat instead performs scene-adaptive clustering followed by query-conditioned cluster scoring to select one coherent 3D target.

## Abstract

We study 3D visual grounding in 3D Gaussian Splatting (3DGS), where a referring expression should identify a target object as both a rendered 2D mask and a set of explicit 3D Gaussians. Prior methods typically extract 3D targets by applying heuristic thresholding or top-ratio filtering to dense primitive-level relevance scores. However, these scores often vary substantially across scenes and prompts, making such heuristics sensitive and leading to unstable targeting. ClusterSplat addresses this by reframing 3D target selection as cluster-level selection over scene-adaptive candidates. The method first learns instance-aware Gaussian features, forms featureaware seed voxels, and merges neighboring seed voxels when a description-length criterion decreases, producing a scene-specific set of cluster candidates. Given a referring expression, a query-conditioned cluster scorer ranks the cluster candidates with a lightweight MLP, and the selected cluster directly defines the explicit 3D target while its cluster score is rendered for the 2D mask. On ScanRefer, ClusterSplat achieves the best rendered 2D segmentation and explicit 3D target metrics among the compared 3DGS baselines. It also obtains the best average Ref-LERF scores, supporting the same trend under fine-grained referring descriptions. On ScanNet v2 class name queries, ClusterSplat achieves the strongest rendered 2D results and the highest 3D matching accuracy, while maintaining competitive 3D IoU. Our project page is available at https://jaesung8.github.io/clustersplat/.

## 1 Introduction

3D Gaussian Splatting (3DGS) has recently emerged as a compelling explicit alternative to neural radiance fields [1]. It offers photorealistic novel-view synthesis with real-time rendering capabilities [2]. Beyond visual rendering, 3DGS is rapidly evolving into a general-purpose 3D representation capable of structural modeling [3, 4] and semantic understanding [5–8]. A pivotal direction in this evolution is language grounding. This capability enables natural-language querying of 3D scenes through referring expressions. While recent approaches have successfully lifted vision-language embeddings into 3D space for open-vocabulary querying [5, 9], explicitly extracting the precise 3D extent of a target object remains a significant challenge.

The core difficulty lies in the continuous nature of the relevance fields used in prior works. Existing methods typically compute dense, primitive-level relevance scores and isolate the target by applying heuristic discretization, such as global thresholding or top-k filtering [6, 10, 11]. However, these heuristics are highly sensitive in open-vocabulary settings. Relevance score distributions often vary substantially with the scene context and the complexity of the prompt (e.g., compositional attributes or spatial relations). Consequently, relying on scale-sensitive hyperparameters often leads to unstable targeting. Figure 1 illustrates this failure mode: top-ratio filtering can scatter selected primitives across the referred object and nearby distractors rather than producing one coherent 3D target. As shown by the top-ratio analysis in Appendix B.1, dense relevance fields often lack clear decision boundaries, causing the target to be missed or merged with background under a fixed selection ratio. Moreover, indoor scenes vary substantially in object count, object scale, clutter, and same-class distractors, so cluster candidates should adapt their granularity to the scene structure. Section 4.3 analyzes the candidate-quality trade-off behind this design choice.

To overcome this instability, we design ClusterSplat as a cluster-level targeting framework (Figure 2) that shifts the selection granularity from individual primitives to scene-adaptive cluster candidates. We view the cluster index as the coarse decision variable for a single referred target and construct these candidates with a scene-adaptive merge procedure guided by description length. These observations, together with the scattered primitive selections in Figure 1, motivate formulating 3D target selection as a discrete selection problem over coherent object-level candidates rather than as a filtering task over millions of unstructured primitives. The framework can be viewed as two stages that align with the offline scene preparation and query-conditioned grounding blocks in Figure 2. First, we optimize instance-aware Gaussian features and construct feature-aware seed voxels. We then build a candidate graph over seed-voxel descriptors and merge adjacent components only when the description length decreases, producing a scene-specific number of clusters. Second, during target selection, a queryconditioned cluster scorer ranks the candidates with a lightweight MLP to select the target cluster. The selected cluster directly defines the explicit 3D target, and its cluster-level scores are rendered to produce the 2D mask.

Furthermore, we observe that the advancement of 3DGS grounding is hindered by the lack of evaluation protocols that jointly assess rendered 2D masks and explicit 3D target sets. Current benchmarks like Ref-LERF [12] are limited in scene diversity, while standard semantic segmentation metrics fail to capture the nuances of compositional language grounding. Class name prompts provide category-level grounding, but they cannot uniquely specify one target when multiple objects from the same category appear in a scene. Referring expressions address this ambiguity through attributes and spatial relations, making ScanRefer a more demanding test of object-level targeting. To bridge this gap, we define a reproducible evaluation protocol that combines official ScanNet v2 annotations with ScanRefer expressions [13, 14]. Our main comparison uses the subset of 10 scenes adopted by recent 3DGS open-vocabulary baselines for reproducibility. This enables class name grounding and referring expression segmentation to be reported on the same ScanNet v2 scenes.

In summary, our contributions are as follows:

• We introduce a scale- and granularity-aware formulation for 3D visual grounding in 3DGS. By transforming primitive-level score filtering into cluster-level selection over scene-adaptive candidates, we avoid per-query score thresholds while allowing the cluster granularity to follow the scene structure.

• We propose ClusterSplat, a two-stage framework that first learns instance-aware Gaussian features and constructs scene-adaptive clusters with a description-length criterion, followed by query-conditioned cluster scoring for explicit target selection and 2D mask rendering.

• We evaluate ClusterSplat with a reproducible protocol combining ScanNet v2 annotations and ScanRefer expressions that reports class name grounding and referring expression segmentation on the same ScanNet v2 scenes.

• On the ScanRefer benchmark, experiments show state-of-the-art referring segmentation among the compared 3DGS baselines. Ref-LERF further confirms consistent gains in fine-grained scenes, while ScanNet v2 class name evaluation shows improved rendered 2D masks and competitive 3D object matching.

## 2 Related Work

## 2.1 3D Gaussian Splatting and Language Grounding

Neural rendering first achieved high-quality novel-view synthesis with implicit radiance fields [1], while recent explicit representations such as 3DGS provide real-time rasterization-based rendering [2]. Explicit primitives also support faster optimization and direct geometric manipulation compared with fully implicit scene encodings. Subsequent work improves 3DGS through structural regularity, filtering, surface-aligned primitives, and alternative optimization strategies [3, 15–17].

Beyond rendering, explicit Gaussian primitives have become a substrate for semantic and languageaware feature fields learned through language embeddings, 2D foundation-model distillation, or cross-modal rasterization [5, 7, 8, 18–20]. However, view-consistent semantic rendering does not by itself define a reliable single-object 3D target. Dr. Splat directly registers language embeddings to Gaussians [10], interaction-oriented methods lift 2D prompts into Gaussian space [21, 22], and instance-level Gaussian representations recover object-level subsets [23, 24]. Yet many pipelines still extract targets from continuous relevance scores using thresholding, top-k selection, or related discretization rules [6, 10, 11]. These rules remain sensitive to scene- and prompt-dependent score scales, which makes explicit 3D targeting unstable. This sensitivity has also motivated alternatives to fixed similarity cutoffs, such as hyperplane-based selection of Gaussians of Interest [11].

Referring tasks in 3D Gaussian scenes make this limitation more apparent. ReferSplat introduces the Ref-LERF benchmark and obtains supervision from reliable Grounded-SAM candidates [12, 25], but view-wise mask cues can still under-specify whether a prediction forms one coherent 3D instance when the Gaussian representation admits ambiguous groupings [6]. Related 3DGS methods use grouped or instance-aware structure for point-level open-vocabulary understanding, view-dependent semantic aggregation, or dense open-vocabulary 3D segmentation [6, 26, 27]. These objectives are complementary to our setting, where a referring expression should select one referred target. This motivates grouping primitives into stable candidates and performing discrete selection in the grouped space. In ClusterSplat, clusters serve as the decision domain for single-target referring expressions, and the selected cluster directly determines the explicit 3D target.

## 2.2 3D Visual Grounding

3D visual grounding aims to localize a language-specified target within a 3D scene. It is commonly separated into comprehension tasks, which predict a coarse target location such as a box, and segmentation tasks, which predict a precise point- or mask-level region [14, 28]. ScanNet v2 scenes and ScanRefer expressions have been instrumental in this area by providing RGB-D reconstructions with rich, compositional language descriptions [13, 14]. Related 3D referring-expression datasets further study language grounding beyond a single benchmark family [29]. Building on these datasets, point-cloud methods have improved language-geometry alignment and relational reasoning [28, 30, 31].

Open-vocabulary 3D scene understanding has also been explored with CLIP-aligned point features and class-agnostic 3D mask proposals before the recent shift toward 3DGS representations [32, 33]. Foundation-model supervision further improves scalability, with AugRefer generating diverse synthetic supervision [34] and SeeGround enabling zero-shot grounding by bridging 3D scenes with 2D vision-language models [35].

Adapting this progress to 3DGS requires care because 3DGS optimizes unstructured primitives primarily for rendering. Unlike explicit point clouds, Gaussian primitives are not directly organized as semantic units, so pixel-level parsing can hide failures in the underlying 3D target set. OpenGaussian highlights this 2D-3D discrepancy when improving feature association and discretized 3D selection [6]. Since relevance scores in open-vocabulary 3DGS vary across scenes and prompts, robust evaluation should report both rendered 2D segmentation and discrete 3D target selection for referring expressions. This need motivates our cluster-level formulation, which avoids scale-dependent score discretization while preserving 3D structural consistency and 2D renderability.

![](images/639e6ae2f9d889968e1fcc2cce84fbc813b9d5a2f5a5897a77c85bfa909e80fe.jpg)  
Figure 2: ClusterSplat overview. The pipeline first performs Instance-Aware Gaussian Feature Learning and Scene-Adaptive Clustering: it learns instance-aware Gaussian features, forms featureaware seed voxels, and merges them into scene-adaptive cluster candidates. It then performs Query-Conditioned Cluster Scoring: the referring expression conditions the cluster tokens, and the MLP scorer ranks the candidates to select the Top-1 cluster as the explicit 3D target. Finally, Cluster-Level Mask Rendering assigns the selected cluster scores to Gaussians and renders a 2D mask supervised by the mask loss.

## 3 Method

## 3.1 Problem Definition and Target Selection

Problem Definition. We study 3D visual grounding in a 3DGS scene representation. A scene is modeled as a set of Gaussian primitives $\mathsf { \bar { g } } = \{ g _ { i } \} _ { i = 1 } ^ { N }$ . Each primitive $g _ { i }$ has geometric and appearance parameters (mean $\mu _ { i } \in { \mathbb { R } } ^ { 3 }$ , covariance $\Sigma _ { i } .$ , opacity $\alpha _ { i }$ , and spherical harmonics $\mathbf { S H } _ { i } )$ We augment each primitive with a learnable instance-aware feature $\mathbf { f } _ { i } \in \mathbb { R } ^ { D _ { f } }$ . Given a query $q$ with referring expression text $T _ { q } ,$ the model predicts two outputs. The first is a 2D segmentation mask for an arbitrary camera pose π, denoted by $\dot { M } _ { 2 D } ( \pi )$ . The second is a 3D target represented as a subset of Gaussians, $S _ { \mathrm { t a r g e t } } \subset \mathcal { G }$ . We focus on the single-target referring setting and introduce a query-specific target cluster index $k _ { q } ^ { \star } \in \{ 1 , \ldots , K _ { s } \}$ . Here, $K _ { s }$ is the scene-specific number of cluster candidates produced for scene s.

3D Targeting via Scene-Adaptive Clusters. Prior 3DGS language grounding pipelines typically compute continuous relevance scores over primitives and extract the target using heuristic filtering rules. However, such primitive-level filtering is inherently unstable as score distributions vary substantially across scenes. Cluster-level selection provides a more structured decision space, but the cluster candidates must remain faithful to scene-dependent object scale and clutter. We therefore formulate 3D visual grounding as selection among scene-adaptive cluster candidates. By aggregating Gaussians into clusters, we reformulate the challenging dense prediction task into a tractable discrete selection problem. Crucially, this simplification relies on clusters representing coherent object-level candidates rather than arbitrary geometric fragments. Our cluster candidates are generated by a scene-adaptive merge procedure guided by description length over feature-aware seed voxels defined in Section 3.2. This allows the number of cluster candidates to vary across scenes. For scene s, we partition the Gaussians into $K _ { s } \ll N$ clusters $\{ \mathcal { G } _ { k } \} _ { k = 1 } ^ { K _ { s } }$ . For each query-cluster pair, the scorer in Section 3.3 produces a query-conditioned score $S _ { q , k } .$ . The selected index $k _ { q } ^ { \star }$ is the cluster index with the maximum score. The corresponding set $\mathcal { G } _ { k _ { q } ^ { \star } }$ contains all Gaussians assigned to that selected cluster, and we define the predicted 3D target set as $S _ { \mathrm { t a r g e t } } = \mathcal { G } _ { k _ { a } ^ { \star } }$ . The selected cluster directly defines the explicit 3D target, and the cluster-level scores provide the renderable evidence for the 2D mask.

$$
k _ { q } ^ { \star } = \operatorname * { a r g m a x } _ { k \in \{ 1 , \dots , K _ { s } \} } S _ { q , k } , \qquad S _ { \mathrm { t a r g e t } } = \mathcal { G } _ { k _ { q } ^ { \star } } .\tag{1}
$$

## 3.2 Instance-Aware Gaussian Feature Learning and Scene-Adaptive Clustering

Instance-aware Gaussian feature learning. Stage 1 adapts an instance-aware Gaussian feature objective related to OpenGaussian [6], using training-view instance masks to learn geometry-aligned Gaussian features. Each Gaussian has a learnable feature $\mathbf { f } _ { i } \in \mathbb { R } ^ { D _ { f } }$ . These features are rendered to feature maps with the same opacity-aware compositing used by 3DGS. Within each training mask, rendered features are pulled toward a mask prototype. Prototypes from different masks in the same view are separated to discourage feature collapse. The resulting geometry-aligned feature field is used for scene-adaptive cluster construction. Appendix A.2 provides the detailed loss definitions.

Scene-adaptive clustering. After feature learning, we construct cluster candidates from featureaware seed voxels. Each Gaussian is assigned using its scene-normalized spatial voxel index and the dominant signed dimension of its $\ell _ { 2 }$ -normalized instance feature. This initialization over-segments the scene so Gaussians with different feature directions can remain separate even within the same spatial voxel. For each seed voxel, we compute a normalized feature prototype $\phi _ { a }$ and a 3D centroid. A minimum spanning tree (MST) in the whitened feature-position descriptor space restricts the candidate merges.

During greedy merging, a component $S$ is a connected set of seed-voxel prototypes. We evaluate each component with an MDL-inspired description-length cost and a BIC-style component penalty [36, 37]:

$$
{ \mathcal { D } } _ { \mathrm { g m d l } } ( S ) = \frac { \mathrm { R S S } _ { \mathrm { f e a t } } ( S ) } { 2 \sigma _ { f } ^ { 2 } } + \lambda _ { \mathrm { g m d l } } \frac { 1 } { 2 } ( D _ { f } + 1 ) \log N _ { \mathrm { e f f } } ,\tag{2}
$$

where $\begin{array} { r } { \mathrm { R S S } _ { \mathrm { f e a t } } ( S ) = | S | - \| \sum _ { a \in S } \phi _ { a } \| . } \end{array}$ <sub>2</sub> measures inconsistency among the feature directions, with each seed prototype contributing unit weight. The feature variance $\sigma _ { f } ^ { 2 }$ and effective prototype count $N _ { \mathrm { e f f } }$ are computed once per scene. Merging components with inconsistent features increases the first term, while the second term is paid once per component and favors fewer clusters. The coefficient $\lambda _ { \mathrm { g m d l } }$ controls this penalty. We merge MST-adjacent components A and B only when the total description length decreases:

$$
\Delta _ { \mathrm { g m d l } } ( A , B ) = { \mathcal { D } } _ { \mathrm { g m d l } } ( A ) + { \mathcal { D } } _ { \mathrm { g m d l } } ( B ) - { \mathcal { D } } _ { \mathrm { g m d l } } ( A \cup B ) > 0 .\tag{3}
$$

When no candidate merge reduces the cost, the remaining components define the Gaussian assignment $a ( i ) \in \{ 1 , \dots , K _ { s } \}$ . The cluster count $K _ { s }$ adapts to each scene, and the assignment remains fixed during Stage 2 training. Appendix A.3 gives the seed construction, candidate graph, and sufficientstatistic definitions.

## 3.3 Query-Conditioned Cluster Scoring

From Stage 1 (Section 3.2), each Gaussian $g _ { i }$ has an adaptive cluster assignment $a ( i ) \in \{ 1 , \dots , K _ { s } \}$ The k-th cluster is ${ \mathcal { G } } _ { k } = \{ g _ { i } \mid a ( i ) = k \}$ . Given query $q$ with referring expression text $T _ { q } ,$ , we score the $K _ { s }$ scene clusters and select a 3D target without score-threshold calibration. We encode $T _ { q }$ with a frozen CLIP text encoder [38]. The token embeddings are projected to the ClusterSplat hidden dimension. This yields the token matrix $\mathbf { E } _ { q } = [ \mathbf { t } _ { q , 1 } , . . . , \mathbf { t } _ { q , L _ { q } } ] ^ { \top } \in \mathbb { R } ^ { L _ { q } \times d _ { p } }$ , where each $\mathbf { \bar { t } } _ { q , \ell } \in \mathbb { R } ^ { d _ { p } }$

We summarize cluster geometry by the centroid

$$
\bar { \mu } _ { k } = \frac { 1 } { \vert \mathcal { G } _ { k } \vert } \sum _ { g _ { i } \in \mathcal { G } _ { k } } \mu _ { i } \in \mathbb { R } ^ { 3 } .\tag{4}
$$

We associate each cluster with a learnable cluster token $\mathbf { c } _ { k } \in \mathbb { R } ^ { d _ { \mathtt { F } } }$ . We inject 3D position using sinusoidal encoding PE(·) [39]:

$$
\begin{array} { r l } & { \mathbf { p } _ { k } ^ { ( 0 ) } = \mathbf { c } _ { k } + \mathrm { P E } ( \bar { \mu } _ { k } ) , } \\ & { \mathbf { P } ^ { ( 0 ) } = [ \mathbf { p } _ { 1 } ^ { ( 0 ) } , \dots , \mathbf { p } _ { K _ { s } } ^ { ( 0 ) } ] ^ { \top } \in \mathbb { R } ^ { K _ { s } \times d _ { p } } . } \end{array}\tag{5}
$$

We update the initialized cluster tokens $\mathbf { P } ^ { ( 0 ) }$ using a Transformer-decoder architecture [39, 40]. First, we apply self-attention to model the global spatial and semantic context:

$$
\mathbf { P } ^ { ( 1 ) } = \mathrm { L N } ( \mathbf { P } ^ { ( 0 ) } + \mathrm { S A } ( \mathbf { P } ^ { ( 0 ) } ) ) .\tag{6}
$$

A 3D target is often distinguished by its relationship with surrounding distractors. Self-attention therefore contextualizes each cluster token with the scene layout. This context reduces false-positive clusters that share local appearance cues but lack the correct relational support.

Subsequently, we employ cross-attention to align the contextualized cluster tokens with CLIP token features $\mathbf { E } _ { q } \mathbf { . }$

$$
{ \bf H } _ { q } = \mathrm { L N } ( { \bf P } ^ { ( 1 ) } + \mathrm { C A } ( { \bf P } ^ { ( 1 ) } , { \bf E } _ { q } ) ) .\tag{7}
$$

Here, $\mathbf { H } _ { q }$ is the query-conditioned token matrix for all scene candidates, with one row per cluster and hidden dimension $d _ { p }$ . Unlike simple global pooling, this cross-attention step allows each cluster representation to access the query tokens that are relevant to that candidate. These tokens may encode object nouns, attributes, or spatial prepositions.

MLP cluster scorer. Let $\mathbf { h } _ { q , k }$ denote the query-conditioned token for the k-th cluster candidate, corresponding to the k-th row of $\mathbf { H } _ { q }$ . To preserve token-level query information, the scorer forms a cluster-specific query summary with cosine-similarity token attention,

$$
\beta _ { q , k , \ell } = \mathrm { s o f t m a x } _ { \ell } \left( \frac { \cos ( \mathbf { h } _ { q , k } , \mathbf { t } _ { q , \ell } ) } { \tau } \right) , \qquad \mathbf { y } _ { q , k } = \sum _ { \ell = 1 } ^ { L _ { q } } \beta _ { q , k , \ell } \mathbf { t } _ { q , \ell } .\tag{8}
$$

Here, $\cos ( \mathbf { a } , \mathbf { b } ) = \langle \mathbf { a } / \| \mathbf { a } \| _ { 2 } , \mathbf { b } / \| \mathbf { b } \| _ { 2 } \rangle$ , and $\tau$ is the token-attention temperature. A two-layer MLP with a 128-dimensional hidden layer predicts the score from the cluster token, its query summary, and their elementwise product (⊙) and absolute difference:

$$
S _ { q , k } = \mathrm { M L P } _ { \mathrm { s c o r e } } \left( \left[ \mathbf { h } _ { q , k } ; \mathbf { y } _ { q , k } ; \mathbf { h } _ { q , k } \odot \mathbf { y } _ { q , k } ; \left| \mathbf { h } _ { q , k } - \mathbf { y } _ { q , k } \right| \right] \right) .\tag{9}
$$

This scorer preserves the final decision as a relative ranking over cluster candidates rather than a threshold on primitive-level scores. We use these scores as the cluster-level evidence for the queryspecific target cluster index introduced in Section 3.1. We select the top-scoring cluster and define the explicit 3D target set:

$$
\begin{array} { r l } & { k _ { q } ^ { \star } = \underset { k \in \{ 1 , \ldots , K _ { s } \} } { \mathrm { a r g m a x } } S _ { q , k } , } \\ & { S _ { \mathrm { t a r g e t } } = \{ g _ { i } \in \mathcal { G } \mid a ( i ) = k _ { q } ^ { \star } \} . } \end{array}\tag{10}
$$

This argmax is the decision rule of the single-target cluster formulation. It relies on the relative ranking of structured cluster candidates. This avoids per-query or per-scene thresholds on primitive-level score magnitudes.

## 3.4 Cluster-Level Mask Rendering

After query-conditioned scoring, each Gaussian inherits the score of its assigned cluster,

$$
\ell _ { q , i } = S _ { q , a ( i ) } .\tag{11}
$$

The explicit 3D output is the Top-1 cluster set in Eq. (10). For the 2D output at camera pose π, we render the Gaussian logits $\ell _ { q , i }$ by the same opacity-aware Gaussian rasterization used for 3DGS rendering. For pixel $u ,$

$$
R _ { q } ( \pi , u ) = \sum _ { i = 1 } ^ { N } \ell _ { q , i } \alpha _ { \pi , i } ( u ) \prod _ { j = 1 } ^ { i - 1 } \bigl ( 1 - \alpha _ { \pi , j } ( u ) \bigr ) , \qquad P _ { q } ( \pi , u ) = \sigma ( R _ { q } ( \pi , u ) ) .\tag{12}
$$

Given the training-view ground-truth mask $M _ { q } ( \pi )$ , Stage 2 is supervised by the same focal and Dice objective used in our implementation,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m a s k } } = ( 1 - \lambda _ { D } ) \mathcal { L } _ { \mathrm { f o c a l } } \big ( P _ { q } ( \pi ) , M _ { q } ( \pi ) \big ) + \lambda _ { D } \mathcal { L } _ { \mathrm { d i c e } } \big ( P _ { q } ( \pi ) , M _ { q } ( \pi ) \big ) , \qquad \lambda _ { D } = 0 . 5 . } \end{array}\tag{13}
$$

Thus, cluster scores are learned through differentiable 2D mask supervision while the explicit 3D prediction remains the selected Gaussian cluster [41, 42].

Table 1: ScanRefer and ScanNet v2 quantitative results on the shared evaluation scenes. All scores are percentages. Bold and underlined values indicate the best and second-best distinct scores in each column.
<table><tr><td rowspan="3">Methods</td><td colspan="4">ScanRefer</td><td colspan="4">ScanNet v2 class name</td></tr><tr><td colspan="2">2D</td><td colspan="2">3D</td><td colspan="2">2D</td><td colspan="2">3D</td></tr><tr><td>mIoU↑</td><td>mAcc@0.25↑</td><td>mIoU↑</td><td>mAcc@0.25↑</td><td>mIoU↑</td><td>mAcc@0.25↑</td><td>mIoU↑</td><td>mAcc↑</td></tr><tr><td>LangSplat [5]</td><td>11.00</td><td>13.41</td><td>5.81</td><td>4.95</td><td>27.67</td><td>40.82</td><td>2.01</td><td>8.13</td></tr><tr><td>OpenGaussian [6]</td><td>18.48</td><td>5.53</td><td>1.06</td><td>0.10</td><td>19.55</td><td>26.11</td><td>31.05</td><td>46.39</td></tr><tr><td>Dr.Splat [10]</td><td>19.01</td><td>27.76</td><td>9.48</td><td>11.17</td><td>19.94</td><td>32.64</td><td>30.67</td><td>46.12</td></tr><tr><td>ReferSplat [12]</td><td>31.30</td><td>45.07</td><td>4.05</td><td>0.10</td><td>28.70</td><td>40.12</td><td>6.35</td><td>17.22</td></tr><tr><td>ReferSplat (w/ our loss)</td><td>31.55</td><td>45.99</td><td>3.85</td><td>0.00</td><td>32.70</td><td>47.32</td><td>4.52</td><td>12.89</td></tr><tr><td>ClusterSplat (Ours)</td><td>39.72</td><td>54.35</td><td>23.33</td><td>43.38</td><td>42.20</td><td>58.30</td><td>29.21</td><td>46.60</td></tr></table>

## 4 Experiments

## 4.1 Experimental Setup

Datasets. To evaluate 3D visual grounding in 3DGS, we use two complementary benchmarks. We first pair ScanNet v2 [13] RGB-D scenes and official annotations with ScanRefer [14] referring expressions, using the scene subset and preprocessing convention adopted by recent 3DGS openvocabulary evaluations. ScanRefer provides referring expressions and target object IDs for objects in the ScanNet v2 scenes, enabling evaluation of both class name grounding and referring expression segmentation within the same reconstructed scenes. We also use Ref-LERF from ReferSplat [12] to assess fine-grained spatial reasoning in small-scale scenes; Appendix A.4 gives the ReferSplat confidence-weighted IoU pseudo-mask rule used for Ref-LERF supervision.

Evaluation Protocol. For ScanNet v2 and ScanRefer, we evaluate on the common 10-scene subset adopted by OpenGaussian [6]. Table 1 reports two complementary settings on the same reconstructed scenes: ScanRefer referring segmentation and ScanNet v2 class name grounding. Both settings evaluate rendered 2D masks and explicit 3D Gaussian target sets. Appendix A.4 and Appendix A.5 provide the held-out view split, annotation sources, and object ID propagation details.

Baselines. We compare ClusterSplat against language-embedded 3DGS pipelines, including LangSplat [5], OpenGaussian [6], Dr.Splat [10], and ReferSplat [12]. Prior works typically output continuous primitive relevance scores, so they require thresholding or top-ratio discretization to define a 3D target set. We therefore sweep the top-ratio in {0.01, 0.02, . . . , 0.10} for each score-based baseline and report the best per-method oracle result; ClusterSplat instead uses Top-1 cluster selection and does not tune a per-query or per-scene score threshold.

## 4.2 Benchmark results

ScanRefer referring segmentation. ScanRefer directly evaluates the referred-object setting because each referring expression is tied to a specific object ID in a ScanNet v2 scene. Table 1 therefore reports ScanRefer first. Compared with ScanNet v2 class name evaluation, ScanRefer is more demanding because the model must identify one referred object among category distractors using attributes, spatial relations, and instance-specific language. Under this setting, ClusterSplat improves both rendered 2D segmentation and explicit 3D target selection over the compared 3DGS baselines. The comparison with ReferSplat trained under the same supervision indicates that the gain is not solely attributable to pseudo-caption or mask supervision, but also to the cluster-level selection architecture. The larger improvement in 3D target metrics further suggests that this architecture is especially useful when the query contains attributes or spatial relations that make primitive-level score filtering unstable.

ScanNet v2 class name evaluation. The ScanNet v2 class name results in Table 1 provide a complementary diagnostic on the same evaluation scenes. ClusterSplat substantially improves rendered 2D segmentation, indicating that the selected clusters provide stable image-space evidence across held-out views. We include this evaluation because prior language-embedded 3DGS studies commonly report 3D evaluation based on class name queries, and the results show that ClusterSplat remains competitive under this established protocol. In 3D, ClusterSplat attains the highest matching accuracy and remains close to the strongest 3D IoU baselines. Since class name prompts describe categories rather than a unique referred instance, this protocol has limited ability to test targeting of a specific object in scenes with multiple objects from the same category. This limitation motivates the ScanRefer referring segmentation evaluation above.

![](images/95284ddf6e2c69c56584440dc658f4d620bb0835efd169c7e5bcf4c1fc2f1631.jpg)  
scene0400\_00  
there is a light brown working desk placed next to the window.  
Figure 3: Qualitative examples on ScanRefer. Each row shows a referring expression with rendered masks and top-down 3D selections. ClusterSplat localizes the referred object more coherently and reduces distractor support compared with score-discretized baselines.

Ref-LERF. On Ref-LERF, ClusterSplat reaches 31.14% average mIoU and 41.70% mAcc@0.25, compared with 26.59% and 36.35% for ReferSplat with our loss. Using the same human expressions and pseudo-masks, this gain supports cluster-level selection under matched supervision and loss. Appendix B.5 reports the per-scene results in Table 8.

Qualitative analysis. Figure 3 supports the ScanRefer trends with queries that require object identity and relational cues, such as a trash can behind a door, a picture above a couch, a toilet in a room corner, and a desk next to a window. Score-discretized baselines often leak support to nearby structures or distractor regions in the rendered view and the top-down 3D selection. ClusterSplat keeps the selected support concentrated on the referred object and yields masks aligned with the target.

Efficiency. On an RTX A6000, ClusterSplat takes 19.55 ms per query after 125.3 min of methodspecific scene preparation. Preparation is incurred once per scene, whereas queries are repeated during interaction. Appendix B.6 reports the comparison and timing scope.

## 4.3 Design analysis

Table 2 shows why purity alone is insufficient for single-cluster grounding. Increasing K in fixed KMeans raises purity but lowers Coverage@1 and Oracle Top-1 IoU as objects split across more clusters. Even perfect selection then recovers less of the target. Scene-adaptive clustering offers more complete candidates, achieving the highest Coverage@1 and Oracle Top-1 IoU despite lower purity than fixed KMeans. This tradeoff matters because Stage 2 can rank candidates but cannot repair their boundaries. Appendix B.3 defines the metrics.

Table 2: Cluster candidate quality before Stage 2 on ScanRefer. Purity, Coverage@1, and Oracle Top-1 IoU are fractions in [0, 1]; Frag@0.9 is a cluster count. Metric definitions are in Appendix B.3. Bold values indicate the best score in each column.
<table><tr><td colspan="4">Oracle Top-1</td></tr><tr><td>Method</td><td>Purity ↑</td><td>Coverage@1 ↑</td><td>IoU↑ Frag@0.9↓</td></tr><tr><td>Fixed KMeans (K = 128)</td><td>0.7825</td><td>0.5649</td><td>0.4017 5.95</td></tr><tr><td>Fixed KMeans (K = 256)</td><td>0.8094 0.4840</td><td>0.3734</td><td>8.90</td></tr><tr><td>Fixed KMeans (K = 512)</td><td>0.8269 0.3935</td><td>0.3200</td><td>14.15</td></tr><tr><td>Fixed KMeans + pair merge</td><td>0.7436</td><td>0.6693</td><td>4.71</td></tr><tr><td>Scene-adaptive clustering (ours)</td><td>0.7635</td><td>0.6933</td><td>0.4615 0.4798 3.86</td></tr><tr><td>Scene-adaptive clustering + xyz cost</td><td>0.7745</td><td>0.6720</td><td>0.4682 3.85</td></tr></table>

Table 3 tests whether multi-cluster selection can compensate for this fragmentation. We retrain Stage 2 under the same protocol for each construction and form cumulative Top-k unions from each model. Fixed KMeans with K = 256 and K = 512 benefits from multiple selections, consistent with fragment recovery. Scene-adaptive clustering instead falls from 23.33% 3D mIoU at Top-1 to 14.96% at Top-5, suggesting that the cost of false positives outweighs the benefit of fragment recovery. This agrees with the focal and Dice supervision in Section 3.4, which encourages query-associated foreground responses and penalizes distractors through rendered cluster logits. No tested union exceeds scene-adaptive Top-1, supporting this combination for the evaluated single-target queries.

Table 3: Effect of cluster construction and cumulative Top-k selection on ScanRefer 3D mIoU (%). Within each construction, Top-k predictions combine the k highest-scoring clusters from the same trained model. Bold indicates the highest value in the table.
<table><tr><td>Construction</td><td>Top-1</td><td>Top-2</td><td>Top-3</td><td>Top-4</td><td>Top-5</td></tr><tr><td>Fixed KMeans (K = 128)</td><td>18.31</td><td>17.66</td><td>16.42</td><td>15.81</td><td>14.47</td></tr><tr><td>Fixed KMeans (K = 256)</td><td>15.44</td><td>16.34</td><td>16.27</td><td>16.32</td><td>16.21</td></tr><tr><td>Fixed KMeans (K = 512)</td><td>10.97</td><td>13.29</td><td>14.56</td><td>14.60</td><td>14.62</td></tr><tr><td>Fixed KMeans + pair merge</td><td>21.23</td><td>19.66</td><td>18.67</td><td>17.47</td><td>16.46</td></tr><tr><td>Scene-adaptive clustering</td><td>23.33</td><td>19.78</td><td>17.42</td><td>16.09</td><td>14.96</td></tr><tr><td>Scene-adaptive + xyz cost</td><td>21.63</td><td>19.64</td><td>17.41</td><td>16.09</td><td>14.56</td></tr></table>

Text encoder and caption source. Appendix B.2 compares frozen CLIP and BERT [43] encoders under the same ClusterSplat configuration. CLIP achieves 23.33% 3D mIoU, compared with 14.62% for BERT. This supports CLIP for our cluster scorer, consistent with a benefit from its image-text pretraining. The caption-source comparison in Appendix B.4 gives 23.33% for GPT-4.1, 21.23% for InternVL3.5-8B [44], and 20.01% for Qwen3-VL-8B [45]. Both local caption generators outperform the compared ScanRefer baselines. The gains persist across the tested generators, although their magnitude depends on the caption source.

## 5 Discussion and Limitations

ClusterSplat reduces dependence on primitive-level score filtering by selecting a scene-adaptive cluster as the explicit target. Its main limitation is therefore the candidate pool. Over-merging nearby similar objects or splitting one object across several clusters can limit the conservative Top-1 selection. This explains why ScanNet v2 class name matching accuracy is strong while exact 3D IoU remains stricter: the selected cluster can overlap the correct object enough for matching, yet still miss parts of the object extent or include nearby fragments. Future work can handle fragmented candidates with uncertainty-aware multi-cluster selection.

Our formulation handles expressions that refer to a single object. Separate queries can select several objects, but grounding a plural expression that refers to a variable number of instances remains future work.

The experiments cover ten indoor ScanNet v2 scenes and four Ref-LERF scenes, with each scene prepared separately. Broader scene coverage and transfer to unseen scenes remain to be evaluated. Training relies on annotated instance masks and generated descriptions, and each new scene requires offline preparation before queries can be processed.

## 6 Conclusion

We presented ClusterSplat, which converts 3DGS visual grounding from primitive-level score filtering into discrete selection over scene-adaptive Gaussian clusters. The method learns instance-aware Gaussian features, builds scene-adaptive clusters with a description-length criterion, ranks them with query-conditioned scoring, and renders cluster-level scores for 2D mask prediction. Results show state-of-the-art referring segmentation on ScanRefer among the compared 3DGS baselines, with Ref LERF providing complementary fine-grained validation. The method also produces strong rendered masks and competitive 3D object matching on ScanNet v2. These results support scene-adaptive cluster selection for referring expression grounding in 3DGS. Remaining errors suggest future work on uncertainty-aware multi-cluster selection while preserving explicit target sets.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00573160); the Technology Innovation Program (RS-2025- 02222776, Development and Demonstration of AI and Lightweight Technology-Based Automated E-Waste Sorting and Retrieval System) funded by the Ministry of Trade, Industry and Resources (MOTIR, Korea); and the “Advanced GPU Utilization Support Program” funded by the Government of the Republic of Korea (Ministry of Science and ICT).

## References

[1] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision, pages 405–421, 2020.

[2] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, July 2023. doi: 10.1145/3592433. URL https://doi.org/10.1145/3592433. Article 139.

[3] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-GS: Structured 3D Gaussians for view-adaptive rendering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654–20664, 2024.

[4] Antoine Guédon and Vincent Lepetit. SuGaR: Surface-aligned Gaussian splatting for efficient 3D mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5354–5363, 2024.

[5] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. LangSplat: 3D language Gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051–20060, 2024.

[6] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, and Jian Zhang. OpenGaussian: Towards point-level 3D Gaussian-based open vocabulary understanding. In Advances in Neural Information Processing Systems, volume 37, pages 19114–19138, 2024.

[7] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin Ren, Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo Gevers, Luc Van Gool, Martin R. Oswald, and Danda Pani Paudel. SceneSplat: Gaussian splatting-based scene understanding with vision-language pretraining. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4961–4972, 2025.

[8] Ruijie Zhu, Mulin Yu, Linning Xu, Lihan Jiang, Yixuan Li, Tianzhu Zhang, Jiangmiao Pang, and Bo Dai. ObjectGS: Object-aware scene reconstruction and scene understanding via Gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8350–8360, 2025.

[9] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. LERF: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19729–19739, 2023.

[10] Jun-Seong Kim, GeonU Kim, Yu-Ji Kim, Yu-Chiang Frank Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. Splat: Directly referring 3D Gaussian splatting via direct language embedding registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14137–14146, June 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Jun-Seong\_Dr. \_Splat\_Directly\_Referring\_3D\_Gaussian\_Splatting\_via\_Direct\_Language\_CVPR\_2025\_ paper.html.

[11] Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Liujuan Cao, Shengchuan Zhang, and Rongrong Ji. GOI: Find 3D Gaussians of interest with an optimizable open-vocabulary semantic-space hyperplane. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pages 5328–5337, 2024. doi: 10.1145/3664647.3680852.

[12] Shuting He, Guangquan Jie, Changshuo Wang, Yun Zhou, Shuming Hu, Guanbin Li, and Henghui Ding. ReferSplat: Referring segmentation in 3D Gaussian splatting. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 22456–22467. PMLR, 2025. URL https://proceedings.mlr.press/v267/he25h.html.

[13] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5828–5839, July 2017. URL https://openaccess.thecvf.com/content\_cvpr\_2017/html/Dai\_ScanNet\_ Richly-Annotated\_3D\_CVPR\_2017\_paper.html.

[14] Dave Zhenyu Chen, Angel X. Chang, and Matthias Nießner. ScanRefer: 3D object localization in RGB-D scans using natural language. In Computer Vision – ECCV 2020, volume 12365 of Lecture Notes in Computer Science, pages 202–221, Cham, 2020. Springer. doi: 10.1007/978-3-030-58565-5\_13. URL https://doi.org/10.1007/978-3-030-58565-5\_13.

[15] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-Splatting: Alias-free 3D Gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19447–19456, 2024.

[16] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D Gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024 Conference Papers, pages 1–11. Association for Computing Machinery, 2024. doi: 10.1145/3641519.3657428. URL https://doi.org/10.1145/ 3641519.3657428. Article 32.

[17] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3D Gaussian splatting as Markov chain Monte Carlo. In Advances in Neural Information Processing Systems, volume 37, pages 80965–80986, 2024.

[18] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3D Gaussians for open-vocabulary scene understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5333–5343, June 2024.

[19] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3DGS: Supercharging 3D Gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676–21685, 2024. doi: 10.1109/CVPR52733.2024.02048.

[20] Qucheng Peng, Benjamin Planche, Zhongpai Gao, Meng Zheng, Anwesa Choudhuri, Terrence Chen, Chen Chen, and Ziyan Wu. 3D vision-language Gaussian splatting. In The Thirteenth International Conference on Learning Representations (ICLR) 2025, pages 61008–61031. OpenReview.net, 2025.

[21] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian Grouping: Segment and edit anything in 3D scenes. In Computer Vision – ECCV 2024, pages 162–179, 2025.

[22] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3D Gaussians. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1971–1979, 2025.

[23] Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao Zhang, Ronggang Wang, and Jian Zhang. InstanceGaussian: Appearance-semantic joint Gaussian representation for 3D instance-level perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14078– 14088, 2025.

[24] Jens Piekenbrinck, Christian Schmidt, Alexander Hermans, Narunas Vaskevicius, Timm Linder, and Bastian Leibe. OpenSplat3D: Open-vocabulary 3D instance segmentation using Gaussian splatting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 5285–5294, June 2025. URL https://openaccess.thecvf.com/content/CVPR2025W/OpenSUN3D/ html/Piekenbrinck\_OpenSplat3D\_Open-Vocabulary\_3D\_Instance\_Segmentation\_using\_ Gaussian\_Splatting\_CVPRW\_2025\_paper.html.

[25] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded SAM: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.

[26] Jiazhong Cen, Xudong Zhou, Jiemin Fang, Changsong Wen, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Tackling view-dependent semantics in 3D language Gaussian splatting. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 7013–7034. PMLR, 2025.

[27] Runsong Zhu, Ka-Hei Hui, Zhengzhe Liu, Qianyi Wu, Weiliang Tang, Shi Qiu, Pheng-Ann Heng, and Chi-Wing Fu. COS3D: Collaborative open-vocabulary 3D segmentation. In Advances in Neural Information Processing Systems, volume 38, pages 11351–11371. Curran Associates, Inc., 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 10a6bdcabbd5a3d36b760daa295f63c1-Abstract-Conference.html.

[28] Shuting He and Henghui Ding. RefMask3D: Language-guided transformer for 3D referring segmentation. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 8316–8325, 2024.

[29] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. ReferIt3D: Neural listeners for fine-grained 3D object identification in real-world scenes. In European Conference on Computer Vision, pages 422–440. Springer, 2020.

[30] Lichen Zhao, Daigang Cai, Lu Sheng, and Dong Xu. 3DVG-Transformer: Relation modeling for visual grounding on point clouds. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2928–2937, 2021.

[31] Yanmin Wu, Xinhua Cheng, Renrui Zhang, Zesen Cheng, and Jian Zhang. EDA: Explicit textdecoupling and dense alignment for 3D visual grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19231–19242, June 2023. URL https: //openaccess.thecvf.com/content/CVPR2023/html/Wu\_EDA\_Explicit\_Text-Decoupling\_ and\_Dense\_Alignment\_for\_3D\_Visual\_Grounding\_CVPR\_2023\_paper.html.

[32] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, and Thomas Funkhouser. OpenScene: 3D Scene Understanding with Open Vocabularies. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 815–824, Vancouver, BC, Canada, June 2023. IEEE. ISBN 9798350301298. doi: 10.1109/CVPR52729.2023.00085.

[33] Ayça Takmaz, Elisabetta Fedele, Robert W. Sumner, Marc Pollefeys, Federico Tombari, and Francis Engelmann. OpenMask3D: Open-vocabulary 3D instance segmentation. In Advances in Neural Information Processing Systems, volume 36, pages 68367–68390. Curran Associates, Inc., 2023. doi: 10.52202/075280-2989. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/d77b5482e38339a8068791d939126be2-Abstract-Conference.html.

[34] Xinyi Wang, Na Zhao, Zhiyuan Han, Dan Guo, and Xun Yang. AugRefer: Advancing 3D visual grounding via cross-modal augmentation and spatial relation-based referring. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 8006–8014, 2025.

[35] Rong Li, Shijie Li, Lingdong Kong, Xulei Yang, and Junwei Liang. SeeGround: See and ground for zero-shot open-vocabulary 3D visual grounding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3707–3717, 2025.

[36] Jorma Rissanen. Modeling by shortest data description. Automatica, 14(5):465–471, 1978.

[37] Gideon Schwarz. Estimating the dimension of a model. The Annals of Statistics, 6(2):461–464, 1978.

[38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021. URL https://proceedings.mlr.press/v139/radford21a.html.

[39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008, 2017.

[40] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European Conference on Computer Vision, pages 213–229. Springer, 2020.

[41] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 2980–2988, 2017.

[42] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 Fourth International Conference on 3D Vision (3DV), pages 565–571. IEEE, 2016.

[43] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, 2019.

[44] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. URL https://arxiv.org/abs/2508.18265.

[45] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025. URL https://arxiv.org/abs/2511.21631.

[46] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

[47] OpenAI. Introducing GPT-4.1 in the API, April 2025. URL https://openai.com/index/gpt-4-1/. Published April 14, 2025; accessed October 12, 2026.

## A Implementation Details

A.1 System Setup and Hyperparameters

A.2 Instance-Aware Gaussian Feature Learning Details

A.3 Scene-Adaptive Clustering and Mask Rendering Details

A.4 Data Preparation

A.5 Automated Pseudo Label Pipeline

A.6 Baseline Implementation Details

## B Additional Results

B.1 Sensitivity Analysis of Selection Heuristics

B.2 Design Choices

B.3 Cluster Evaluation Protocols

B.4 Caption Source and Evaluation-Language Analysis

B.5 Ref-LERF Quantitative Results

B.6 Efficiency Analysis

B.7 Additional Qualitative Results

## Supplementary Material

In this supplementary material, we provide additional details omitted from the manuscript. Section A covers implementation details and the evaluated 3D tasks. Section B includes analysis of baseline methods, ablation studies, and qualitative results.

## A Implementation Details

## A.1 System Setup and Hyperparameters

Unless otherwise specified, experiments are conducted on a single NVIDIA RTX 5090 GPU. The online efficiency measurements in Appendix B.6 are measured on an NVIDIA RTX A6000 server. We implement our framework using PyTorch 2.8.0+cu128 and CUDA 12.8. To ensure training stability and precision, mixed-precision training is not employed.

For the model architecture, we use a frozen CLIP text encoder [38] and project token embeddings to the ClusterSplat cluster/text hidden dimension $d _ { p } = 6 4$ . Cluster assignments are loaded from the offline scene-adaptive feature-clustering step and used as fixed assignments during Stage 2. The resulting scene-specific cluster counts on the 10-scene ScanNet v2 subset are 53-143, with an average of 87.3 clusters per scene.

The token-conditioned cluster scorer is a two-layer MLP with one hidden layer of dimension 128, separate from the prototype dimension $d _ { p }$ . The same cluster scores are used for Top-1 3D target selection and rendered 2D mask prediction.

The model is optimized using the Adam optimizer with a fixed learning rate of $1 \times 1 0 ^ { - 4 }$ for all experiments. The total number of training iterations is set to 10,000. We use the focal and Dice segmentation objective with equal weights and do not use cluster-entropy regularization in the current full setting.

## A.2 Instance-Aware Gaussian Feature Learning Details

Stage 1 learns instance-aware Gaussian features from mask supervision to build the feature field used by scene-adaptive clustering. For this feature-learning branch, we follow the OpenGaussian-style setup and obtain mask-level supervision from multi-view SAM instance masks [6, 46], separate from the ScanNet v2 masks used for pseudo-caption and evaluation-label construction. For a training view v and pixel $u \in \Omega$ , the per-Gaussian feature $\mathbf { f } _ { i }$ is rendered with opacity-aware compositing:

$$
\mathbf { F } _ { v } ( u ) = \sum _ { i = 1 } ^ { N } \mathbf { f } _ { i } \alpha _ { v , i } ( u ) \prod _ { j = 1 } ^ { i - 1 } \bigl ( 1 - \alpha _ { v , j } ( u ) \bigr ) ,\tag{14}
$$

where $\alpha _ { v , i } ( u )$ is the per-pixel opacity contribution of Gaussian i after front-to-back sorting. We use the normalized rendered feature $\tilde { \mathbf { F } } _ { v } ( u ) = \mathbf { F } _ { v } ( u ) / \| \mathbf { F } _ { v } ( u ) \| _ { 2 }$ for the mask-level losses.

Let $\mathcal { M } _ { v } = \{ M _ { v , m } \} _ { m = } ^ { M _ { v } }$ denote training-view instance masks and $\Omega _ { v , m } = \{ u \in \Omega \ | \ M _ { v , m } ( u ) = 1 \}$ denote the pixel support of mask m. The normalized prototype of mask m i

$$
\begin{array} { l } { { \displaystyle \bar { \psi } _ { v , m } = \frac { 1 } { | \Omega _ { v , m } | } \sum _ { u \in \Omega _ { v , m } } \tilde { \mathbf { F } } _ { v } ( u ) } } \\ { { \displaystyle \psi _ { v , m } = \frac { \bar { \psi } _ { v , m } } { | | \bar { \psi } _ { v , m } | | _ { 2 } } . } } \end{array}\tag{15}
$$

Following OpenGaussian-style mask supervision [6], the objective combines intra-mask cohesion and inter-mask separation:

$$
\mathcal { L } _ { \mathrm { c o h } } = \frac { 1 } { \sum _ { v } M _ { v } } \sum _ { v } \sum _ { m = 1 } ^ { M _ { v } } \frac { 1 } { \left| \Omega _ { v , m } \right| } \sum _ { u \in \Omega _ { v , m } } \left\| \tilde { \mathbf { F } } _ { v } ( u ) - \boldsymbol { \psi } _ { v , m } \right\| _ { 2 } ^ { 2 } ,\tag{16}
$$

$$
\mathcal { L } _ { \mathrm { s e p } } = \frac { 1 } { \sum _ { v } M _ { v } ( M _ { v } - 1 ) } \sum _ { v } \sum _ { m \neq n } \frac { w _ { v , m n } } { \| \psi _ { v , m } - \psi _ { v , n } \| _ { 2 } ^ { 2 } + \epsilon } ,\tag{17}
$$

where $w _ { v , m n }$ is a distance-rank weight for nearby or easily confused mask pairs. The Stage 1 instance-feature loss is

$$
\mathcal { L } _ { \mathrm { i n s t } } = \lambda _ { \mathrm { c o h } } \mathcal { L } _ { \mathrm { c o h } } + \lambda _ { \mathrm { s e p } } \mathcal { L } _ { \mathrm { s e p } } .\tag{18}
$$

## A.3 Scene-Adaptive Clustering and Mask Rendering Details

This subsection expands the mathematical details of the scene-adaptive clustering stage used by the reported model. The clustering stage is performed once per scene before query-conditioned training. It uses Gaussian centers and learned instance features, but not the evaluation query, CLIP text features, 3D object IDs, or a manually fixed target number of clusters.

Feature-aware seed voxels. Let $\tilde { \mu } _ { i } \in [ 0 , 1 ] ^ { 3 }$ denote the per-scene normalized center of Gaussian $g _ { i }$ , and let $\hat { \mathbf { f } } _ { i } = \mathbf { f } _ { i } / \lVert \mathbf { f } _ { i } \rVert _ { 2 }$ denote its normalized instance feature. Given voxel size $\delta ,$ the spatial voxel index is

$$
\nu _ { i } = \lfloor \tilde { \mu } _ { i } / \delta \rfloor .\tag{19}
$$

We further split each occupied voxel by the dominant signed feature direction,

$$
d _ { i } = \underset { j } { \operatorname { a r g m a x } } | \hat { \mathbf { f } } _ { i , j } | , \qquad s _ { i } = \mathrm { s i g n } ( \hat { \mathbf { f } } _ { i , d _ { i } } ) , \qquad z _ { i } = ( \nu _ { i } , d _ { i } , s _ { i } ) .\tag{20}
$$

The key $z _ { i }$ defines the initial seed voxel assignment. This construction intentionally over-segments the scene so that small objects and local boundaries are less likely to be absorbed before merging.

Seed-voxel prototypes and candidate graph. For each seed voxel a, we compute a normalized feature prototype and a spatial centroid,

$$
\phi _ { a } = \frac { \sum _ { i : z _ { i } = a } \hat { \mathbf { f } } _ { i } } { \left\| \sum _ { i : z _ { i } = a } \hat { \mathbf { f } } _ { i } \right\| _ { 2 } } , \qquad \bar { \mu } _ { a } = \frac { 1 } { n _ { a } } \sum _ { i : z _ { i } = a } \tilde { \mu } _ { i } ,\tag{21}
$$

where $n _ { a } = | \{ i : z _ { i } = a \} |$ |. In the final setting, each seed voxel contributes one prototype observation with unit weight. Candidate merge edges are then built from a whitened feature-position descriptor,

$$
\mathbf { y } _ { a } = [ \phi _ { a } ; \bar { \mu } _ { a } ] , \qquad \tilde { \mathbf { y } } _ { a } = ( \mathbf { y } _ { a } - m _ { y } ) \oslash s _ { y } , \qquad E _ { \mathrm { c a n d } } = \mathrm { M S T } ( \{ \tilde { \mathbf { y } } _ { a } \} _ { a } ) ,\tag{22}
$$

where $\mathbf { \nabla } m _ { y }$ and $s _ { y }$ are the per-dimension mean and standard deviation, and ⊘ denotes element-wise division. This graph restricts which neighboring prototypes can be tested for merging, while the description-length cost below decides whether an allowed pair should merge.

Description-length merge. We adopt an MDL-inspired description-length objective for components built from seed-voxel prototypes; its complexity term adapts a BIC-style asymptotic penalty to this component representation [36, 37]. For a component S containing seed-voxel prototypes, let

$$
W _ { S } = \sum _ { a \in S } w _ { a } , \qquad \mathbf { G } _ { S } = \sum _ { a \in S } w _ { a } \phi _ { a } , \qquad \mathrm { R S S } _ { \mathrm { f e a t } } ( S ) = W _ { S } - \| \mathbf { G } _ { S } \| _ { 2 } .\tag{23}
$$

With unit prototype weights in the reported model, $w _ { a } = 1$ . The scene-global feature variance and effective count are

$$
\sigma _ { f } ^ { 2 } = \operatorname* { m a x } \left( \frac { W _ { \mathrm { a l l } } - \| \mathbf { G } _ { \mathrm { a l l } } \| _ { 2 } } { W _ { \mathrm { a l l } } D _ { f } } , \epsilon \right) , \qquad N _ { \mathrm { e f f } } = \operatorname* { m a x } \left( \frac { W _ { \mathrm { a l l } } ^ { 2 } } { \sum _ { a } w _ { a } ^ { 2 } } , 1 \right) .\tag{24}
$$

The component cost used for merging is

$$
{ \mathcal { D } } _ { \mathrm { g m d l } } ( S ) = \frac { \mathrm { R S S } _ { \mathrm { f e a t } } ( S ) } { 2 \sigma _ { f } ^ { 2 } } + \lambda _ { \mathrm { g m d l } } \frac { 1 } { 2 } ( D _ { f } + 1 ) \log N _ { \mathrm { e f f } } ,\tag{25}
$$

up to constants that cancel in pairwise merge decisions. The second term has the standard $\textstyle { \frac { p } { 2 } }$ log $N$ information-criterion form with $p = D _ { f } + 1$ and $N = N _ { \mathrm { e f f } }$ , but we use it as an MDL-inspired penalty rather than an exact BIC score. The second term is paid once per component, so a candidate edge $( A , B )$ is accepted only when

$$
\Delta _ { \mathrm { g m d l } } ( A , B ) = { \mathcal { D } } _ { \mathrm { g m d l } } ( A ) + { \mathcal { D } } _ { \mathrm { g m d l } } ( B ) - { \mathcal { D } } _ { \mathrm { g m d l } } ( A \cup B ) > 0 .\tag{26}
$$

Accepted merges add the sufficient statistics of the two components and update the candidate graph. The final labels are compacted into an assignment $a ( i ) \in \{ \bar { 1 } , \dots , K _ { s } \}$ , where $K _ { s }$ is determined by the accepted merges and can vary across scenes.

Cluster scores to mask supervision. During Stage 2, the assignment is fixed. Given cluster scores $S _ { q , k } ,$ , Gaussian $g _ { i }$ receives logit $\ell _ { q , i } = S _ { q , a ( i ) }$ . For camera pose π, opacity-aware rendering produces a logit map

$$
R _ { q } ( \pi ) = \mathrm { R e n d e r } _ { \pi } \bigl ( \{ \ell _ { q , i } \} _ { i = 1 } ^ { N } \bigr ) , \qquad P _ { q } ( \pi ) = \sigma ( R _ { q } ( \pi ) ) .\tag{27}
$$

The Stage 2 mask objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m a s k } } = ( 1 - \lambda _ { D } ) \mathcal { L } _ { \mathrm { f o c a l } } ( P _ { q } , M _ { q } ) + \lambda _ { D } \mathcal { L } _ { \mathrm { d i c e } } ( P _ { q } , M _ { q } ) , \qquad \lambda _ { D } = 0 . 5 , } \end{array}\tag{28}
$$

where $M _ { q }$ is the training-view target mask. Thus, the query-conditioned scorer is trained through rendered 2D mask supervision while the explicit 3D output remains the selected cluster.

## A.4 Data Preparation

ScanNet v2 Data Formatting. We preprocess the ScanNet v2 sequences by decoding the raw .sens files to extract RGB images and camera poses using the official ScanNet toolkit. For fair comparison and reproducibility, we use the 10-scene evaluation subset introduced by OpenGaussian and follow the standard data processing pipeline for 3DGS-based methods.

Interleaved Train/Evaluation Split. To rigorously evaluate 3D generalization on unseen viewpoints, we adopt an interleaved evaluation split. We define the evaluation set $\mathcal { V } _ { e v a l }$ as every 20th frame with an offset of 10. The remaining frames constitute the training set $\nu _ { t r a i n }$ . All pseudo-caption generation and feature supervision construction are performed after this split and only on $\mathcal { V } _ { t r a i n }$ , so the held-out evaluation frames are never used to construct training supervision.

Evaluation Protocol Details. The main quantitative table for ScanNet v2 and ScanRefer reports two related but distinct evaluations on the same reconstructed scenes. The ScanRefer columns use referring expressions tied to target object instances, while the ScanNet v2 columns use class name queries derived from official scene annotations. For 2D evaluation, predicted masks are rendered on held-out frames and compared with official ScanNet v2 instance annotations projected to those frames [13]. For 3D evaluation, object instance IDs are propagated to Gaussians using the Mahalanobis-distance assignment described below, and IoU is computed over the resulting object ID Gaussian set [10]. ScanRefer target object IDs define the foreground instance for referring segmentation queries, preventing same-category distractors from being counted as target foreground.

Ground Truth Generation. We employ distinct supervision strategies for training and evaluation:

• Training set $( \ V _ { t r a i n } )$ : In ScanNet v2 and ScanRefer experiments, we use the official ScanNet v2 instance masks on training frames as visual anchors and generate pseudo-captions for those masks using the Automated Supervision Generation Pipeline detailed in Sec. A.5.

• Evaluation set $( \mathcal { V } _ { e v a l } ) \colon$ To strictly evaluate rendered 2D segmentation quality, we utilize the official ScanNet v2 2D instance annotation projections (\*\_2d-instance-filt.zip). For 3D referring segmentation, we use ScanNet v2 and ScanRefer object instance IDs to construct object-level Gaussian pseudo-GT labels, as detailed in Sec. A.5.

Ref-LERF pseudo masks. For Ref-LERF, we follow ReferSplat’s Grounded-SAM pseudo-mask selection protocol [12, 25]. Given K candidate masks $\{ \hat { M } _ { k } \} _ { k = 1 } ^ { K }$ and confidences $\{ \gamma _ { k } \} _ { k = 1 } ^ { K }$ for an image-expression pair, define the confidence-weighted mask $A _ { k } ( u ) = \gamma _ { k } \hat { M } _ { k } ( u )$ . The ReferSplat confidence-weighted IoU score is

$$
p _ { k } = \sum _ { j = 1 } ^ { K } \frac { \sum _ { u } \operatorname* { m i n } \big ( A _ { k } ( u ) , A _ { j } ( u ) \big ) } { \sum _ { u } \operatorname* { m a x } \big ( A _ { k } ( u ) , A _ { j } ( u ) \big ) } , \qquad \hat { M } = \hat { M } _ { \mathrm { a r g m a x } _ { k } p _ { k } } .\tag{29}
$$

Equivalently, this is the weighted-overlap form $\textstyle \sum _ { j } ( \gamma _ { k } { \hat { M } } _ { k } \cap \gamma _ { j } { \hat { M } } _ { j } ) / ( \gamma _ { k } { \hat { M } } _ { k } \cup \gamma _ { j } { \hat { M } } _ { j } )$ proposed in ReferSplat.

## A.5 Automated Pseudo Label Pipeline

We generate language supervision for the training views $( \mathcal { V } _ { t r a i n } )$ using training frame ScanNet v2 instance masks and Vision-Language Models (VLMs). The masks are official training frame annotations, while the associated captions are pseudo-captions generated for language supervision. This setting is therefore annotation-assisted rather than annotation-free. Evaluation frame masks, official ScanRefer referring expressions, and evaluation target object IDs are not provided to the pseudo-caption generator or the optimization objective.

Step 1: Training-frame Instance Mask Extraction. For each training frame, we extract the official ScanNet v2 2D instance masks and compute the tight 2D bounding box for each visible instance. These training frame masks, rather than SAM proposals, serve as the visual anchors for the subsequent caption generation process.

Step 2: Extensive Pseudo-Caption Generation. Previous works have explored the use of VLMs to generate comprehensive 3D object and scene descriptions at scale [34]. Similarly, we employ GPT-4.1 [47] to generate five diverse pseudo-captions for each training frame instance mask. The GPT-4.1 input consists of the training RGB image, the instance mask, and its bounding box; it does not include official ScanRefer expressions, evaluation target object IDs, evaluation-frame masks, or 3D object annotations. Unlike 3D visual grounding methods that rely on a single description per object, these multiple pseudo-captions provide richer linguistic variation while remaining independent of the evaluation queries. We prompt GPT-4.1 to describe visual attributes, spatial context, and object distinctiveness visible within the masked region, following the types of attributes emphasized in 3D visual grounding [35]. As shown by the generated captions in Figure 4, this produces pseudo-captions that describe the training frame instance without exposing the held-out target answer.

Leakage-safe supervision set. Using ScanNet $\mathbf { v } 2$ instance masks on training frames is leakage-safe with respect to our held-out evaluation protocol as long as the split is applied before supervision is constructed. Specifically, for a training view $v \in \mathcal { V } _ { t r a i n } ,$ , let $\bar { \mathcal { M } } _ { v } ^ { i n s t } = \bar { \{ M _ { v , m } ^ { i n s t } \} } _ { m = 1 } ^ { M _ { v } }$ be the official instance masks visible in that training frame and let $c _ { v , m } ^ { ( r ) }$ denote the r-th GPT pseudo-caption for mask $m .$ . The training caption-supervision set is

$$
\mathcal { D } _ { t r a i n } ^ { c a p } = \{ ( v , M _ { v , m } ^ { i n s t } , c _ { v , m } ^ { ( r ) } ) : v \in \mathcal { V } _ { t r a i n } , m \in [ 1 , M _ { v } ] , r \in [ 1 , R ] \} .
$$

For a ScanRefer evaluation query $q$ with expression $T _ { q }$ and target object ID $o _ { q } ,$ the official expression $T _ { q }$ and held-out evaluation masks are not included in $\mathcal { D } _ { t r a i n } ^ { c a p } \dot { } .$ . The object ID $o _ { q }$ may correspond to the same physical instance visible in training frames, which is expected in a scene-level view split protocol; however, it is not supplied to GPT-4.1 as the query answer or referring expression during caption generation. The 2D evaluation mask is constructed only on held-out frames from the official instance projection $A _ { v } ^ { 2 D }$ as $M _ { q , v } ^ { 2 D } ( u ) = \mathbf { 1 } [ A _ { v } ^ { 2 D } ( u ) = o _ { q } ]$ , for $v \in \mathcal { V } _ { e v a l }$ . Thus, the protocol is not annotation-free, but it avoids leakage from evaluation views and official referring expressions into training.

![](images/00826790320f58a68de365733744542652bbdf2dddc17cd55069a66c27625dac.jpg)  
Figure 4: Illustration of how we prompt GPT-4.1 to obtain five diverse pseudo-captions for a trainingframe instance mask. The specific view shows two tables, and the mask/bounding-box cue enables GPT-4.1 to generate captions that are descriptive and localized to the selected instance.

Object ID Gaussian pseudo-GT for 3D evaluation. For 3D referring segmentation, the target should be the referred object instance rather than every object of the same semantic class. We therefore follow the Gaussian label propagation protocol based on Mahalanobis distance used in Dr. Splat [10], but replace the semantic label space with ScanNet v2 and ScanRefer object IDs. Let $\mathcal { P } _ { s } = \{ p _ { k } \} _ { k = 1 } ^ { Q }$ be annotated points in scene $s ,$ each with object ID $o ^ { p _ { k } } \in \mathcal { O } _ { s } ,$ , and let $g _ { i }$ be a Gaussian with center $\mu _ { i }$ and covariance $\Sigma _ { i }$ . We compute

$$
\begin{array} { r } { d ^ { \mathrm { m a h a l } } ( p _ { k } , g _ { i } ) = ( p _ { k } - \mu _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( p _ { k } - \mu _ { i } ) , \qquad \eta ( p _ { k } , g _ { i } ) = \exp ( - \frac { 1 } { 2 } d ^ { \mathrm { m a h a l } } ( p _ { k } , g _ { i } ) ) . } \end{array}
$$

The object ID pseudo-GT label of Gaussian $g _ { i }$ is assigned by aggregating this local affinity over annotated points with the same object ID,

$$
o _ { i } ^ { g } = \underset { o \in \mathcal { O } _ { s } } { \mathrm { a r g m a x } } \sum _ { p _ { k } \in \mathcal { P } _ { s } } \mathbf { 1 } [ o ^ { p _ { k } } = o ] \eta ( p _ { k } , g _ { i } ) .
$$

Given a ScanRefer query with expression $T _ { q }$ and target ID $o _ { q } ,$ the 3D pseudo-GT foreground is ${ \mathcal { V } } _ { q } ^ { \mathrm { o b j } } = \{ i : o _ { i } ^ { g } = o _ { q } \}$ . This object ID definition prevents same-class distractors from being counted as foreground and is therefore aligned with referring expression segmentation. For qualitative inspection, Figure 9 visualizes the resulting Gaussian pseudo labels alongside the annotated ScanNet v2 point cloud labels after mapping object IDs to the 19 semantic classes.

Template-based Augmentation for ScanNet v2. While GPT-generated captions provide linguistic diversity, short category names in ScanNet v2 (e.g., "chair", "table") can be too terse as language prompts. To provide the CLIP text encoder with natural sentence context, we wrap the class nouns into simple prompt templates:

• "There is a {} in the scene."

• "The scene contains a {}."

• "This visual element is part of a $\{ \} . "$

• "This local feature corresponds to a $\{ \} . "$

This augmentation preserves class name supervision in a sentence-like form and improves the stability of text-cluster alignment without introducing evaluation queries into training.

## A.6 Baseline Implementation Details

We evaluate our framework against four strong language-embedded 3DGS pipelines: LangSplat [5], OpenGaussian [6], Dr.Splat [10], and ReferSplat [12]. To ensure a fair comparison, all baselines are reproduced using their official implementations with the following specific configurations:

• LangSplat: We employ the CLIP-based auto-encoder with a latent dimension of 3.

• OpenGaussian: We use the official implementation with the SAM-based feature setting and evaluate its continuous relevance scores under the same held-out view split and top-ratio sweep described below.

• Dr.Splat: We use the official implementation with the released configuration on the same reconstructed scenes. Its continuous relevance scores are converted to 3D target sets using the same top-ratio sweep, while the Mahalanobis label propagation protocol is used only to construct object ID pseudo-GT for evaluation.

• ReferSplat: We train this model using the same training-frame mask and pseudo-caption supervision as our method. This ensures that any performance gains are attributed to our architectural contributions rather than differences in training data quality.

3D Target Extraction Strategy. A critical challenge in benchmarking is that prior works typically output continuous relevance scores, requiring a discretization step to define the 3D target set $S _ { \mathrm { t a r g e t } } .$ . As shown in Fig. 5, a fixed global threshold is often suboptimal due to scale variations across scenes. To prevent underestimating the baselines, we adopt an oracle evaluation protocol. Specifically, for each baseline method, we perform a grid search for the top-ratio parameter $\rho \in \{ 0 . 0 1 , 0 . 0 2 , \ldots , 0 . 1 0 \}$ We report the best per-method oracle performance achieved across this range. This strategy provides a per-method oracle within the swept range, so the comparison is less affected by under-tuned discretization choices for competing score-based methods.

## B Additional Results

This section provides further analysis of the experiments. We examine selection heuristics, model design, cluster candidates, caption sources, and lexical overlap, then give per-scene Ref-LERF results and additional qualitative comparisons.

## B.1 Sensitivity Analysis of Selection Heuristics

We investigated the stability of primitive-level score discretization in existing baselines. To this end, we conducted a grid search on the top-ratio hyperparameter for OpenGaussian [6] and Dr.Splat [10]. We evaluated the 3D targeting performance across 10 ScanNet v2 scenes by sweeping the top-ratio from 0.01 to 0.10.

Figure 5 visualizes the results. We observe that the optimal selection ratio depends heavily on the scene. For example, Dr.Splat peaks at a ratio of 0.04 in scene0070. However, scene0645 shows a decreasing trend. This variation indicates that using a fixed global ratio is ineffective for open-vocabulary scenarios. Additionally, the performance curves show sharp peaks. Small deviations from the optimal ratio cause significant drops in 3D IoU. These findings confirm that primitivelevel relevance fields lack clear decision boundaries in 3D space. Our method uses a Top-1 cluster extraction rule that removes per-query and per-scene score-threshold calibration during 3D target extraction.

## B.2 Design Choices

We examine the text encoder, cluster candidates, scorer modules, and prototype dimension.

Comparison settings. The encoder, caption-source, and cluster-construction comparisons use $d _ { p } = 6 4$ on the ten ScanRefer scenes. Cumulative Top-k predictions combine score-ranked clusters from the same trained model.

![](images/74d8d0fe025a4179135ea80932052293cb36cbd5194fbcae5277633385485dbb.jpg)  
Figure 5: Sensitivity of primitive-level target extraction to the top-ratio hyperparameter. We evaluate the 3D targeting performance (mIoU) of OpenGaussian [6] and Dr.Splat [10] across 10 ScanNet v2 scenes by sweeping the selection ratio $\rho \in [ 0 . 0 1 , 0 . 1 0 ]$ . The optimal ratio is highly scenedependent, and the performance often drops sharply away from the peak, indicating that primitivelevel score discretization requires careful calibration for robust open-vocabulary 3D targeting. This motivates cluster-level Top-1 (argmax) selection, which relies on relative cluster ranking rather than absolute score thresholds.

Text backbone scope. ClusterSplat uses CLIP token features at the cluster level. Adaptive cluster tokens attend to query tokens and are ranked by a lightweight MLP scorer. Table 4 compares frozen CLIP ViT-B/32 and BERT-large-uncased with the same scene features, cluster assignments, scorer, loss, and training settings. Both encoders provide last-layer token features projected to $d _ { p } = 6 4$ and use their respective tokenization. CLIP yields higher scores across all reported metrics. Its image-text pretraining [38] may make its tokens better suited to alignment with visual cluster representations through projection, cross-attention, and cosine token attention. The paired comparison supports CLIP for this scorer; the preferred encoder should be evaluated within each grounding architecture.

Table 4: Frozen-encoder comparison on ScanRefer. Bold values indicate the best score in each column.
<table><tr><td rowspan="2">Encoder</td><td colspan="2">2D</td><td colspan="2">3D</td></tr><tr><td>mIoU↑</td><td>mAcc@0.25↑</td><td>mIoU↑</td><td>mAcc@0.25↑</td></tr><tr><td>BERT</td><td>31.47</td><td>43.48</td><td>14.62</td><td>25.40</td></tr><tr><td>CLIP</td><td>39.72</td><td>54.35</td><td>23.33</td><td>43.38</td></tr></table>

Adaptive cluster candidates. The cluster assignment is generated before Stage 2 and then kept fixed as cluster labels during query-conditioned training. This design keeps the online target selection rule simple: the model ranks a scene-specific set of renderable cluster candidates and selects the Top-1 cluster. Table 2 evaluates this cluster candidate pool before query-conditioned training; Appendix B.3 defines the metrics.

Cluster-level rendering. The cluster scorer ranks scene-adaptive candidates, the Top-1 cluster defines the explicit 3D target, and the corresponding cluster-level scores are rendered for 2D mask prediction. This keeps the inference rule aligned with the central cluster-level formulation.

Module ablation. Table 5 reports a ScanRefer ablation of the CLIP-based cluster scorer while keeping the prototype dimension fixed at $d _ { p } = 6 4$ for all variants. The token scorer baseline already gives a strong 3D mIoU, so the gains are not simply due to adding more layers. Cross-attention alone does not improve the 3D metrics, but the full scorer with position encoding, cluster self-attention, cross-attention, and token scoring gives the best 2D accuracy, 3D mIoU, 3D mAcc@0.25, and Hit@1.

Overall, the components are most useful as a joint scorer for scene-level cluster ranking rather than as independent monotonic additions. We analyze the effect of $d _ { p }$ separately in Table 6.

Table 5: Module ablation on ScanRefer with $d _ { p } = 6 4$ for all variants. Bold values indicate the best score in each column.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">Pos.</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Cluster SA Cross-attn Token scorer</td><td colspan="2">2D</td><td colspan="3">3D</td></tr><tr><td>mIoU↑</td><td>mAcc@0.25↑</td><td>mIoU↑</td><td>mAcc@0.25↑</td><td>Hit@1↑</td></tr><tr><td>Pooled scorer baseline</td><td></td><td></td><td></td><td>-</td><td>37.93</td><td>51.59</td><td>22.20</td><td>41.29</td><td>43.52</td></tr><tr><td>Token scorer baseline</td><td></td><td></td><td></td><td>√</td><td>37.86</td><td>51.66</td><td>23.17</td><td>41.96</td><td>42.29</td></tr><tr><td>+ Cross-attention</td><td></td><td></td><td></td><td>√</td><td>37.77</td><td>52.18</td><td>21.69</td><td>40.75</td><td>42.67</td></tr><tr><td>+ Cluster self-attention</td><td></td><td>√</td><td>√</td><td>√</td><td>37.81</td><td>51.69</td><td>22.55</td><td>41.34</td><td>44.05</td></tr><tr><td>Full scorer</td><td>√</td><td>√</td><td>√</td><td>√</td><td>39.72</td><td>54.35</td><td>23.33</td><td>43.38</td><td>46.03</td></tr></table>

Prototype dimension. Table 6 varies the prototype dimension $d _ { p }$ while keeping the full scorer. The 128-dimensional setting gives the highest rendered 2D scores, but its explicit 3D target metrics are lower than those of $d _ { p } = 6 4$ . Increasing the dimension to 256 further reduces 3D mIoU, mAcc@0.25, mAcc, and Hit@1. We therefore use $d _ { p } = 6 4$ , which provides the best 3D target selection while retaining competitive 2D segmentation.

Table 6: Prototype dimension ablation on ScanRefer with the full scorer. Bold values indicate the best score in each column.
<table><tr><td rowspan="2"> $d _ { p }$ </td><td colspan="2">2D</td><td colspan="4">3D</td></tr><tr><td>mIoU↑</td><td>mAcc@0.25↑</td><td>mIoU↑</td><td>mAcc@0.25↑</td><td>mAcc↑</td><td>Hit@1↑</td></tr><tr><td>32</td><td>38.75</td><td>52.59</td><td>22.06</td><td>40.92</td><td>33.66</td><td>43.88</td></tr><tr><td>64</td><td>39.72</td><td>54.35</td><td>23.33</td><td>43.38</td><td>35.86</td><td>46.03</td></tr><tr><td>128</td><td>40.41</td><td>55.00</td><td>21.42</td><td>40.32</td><td>32.09</td><td>41.71</td></tr><tr><td>256</td><td>39.77</td><td>54.78</td><td>20.52</td><td>36.69</td><td>30.30</td><td>39.42</td></tr></table>

## B.3 Cluster Evaluation Protocols

We further analyze the cluster candidates before query-conditioned Stage 2 training. This analysis uses object ID labels assigned to 3D Gaussians and therefore measures the quality of the cluster candidates themselves, not the final query-conditioned segmentation performance. We report four complementary metrics. Purity measures whether a cluster is dominated by one object, Coverage@1 measures how much of a ground-truth object is covered by its best single cluster, Oracle Top-1 IoU measures the best possible single-cluster IoU under perfect cluster selection, and Frag@0.9 measures how many clusters are needed to cover 90% of an object.

Section 4.3 relates these candidate metrics in Table 2 to downstream grounding in Table 3.

Cumulative selection protocol. We also evaluate downstream grounding under the training settings in Appendix B.2. Let $k _ { q , 1 } , \ldots , k _ { q , K _ { s } }$ denote the cluster indices ranked by query scores. The cumulative Top-k prediction is

$$
S _ { q } ^ { ( k ) } = \bigcup _ { j = 1 } ^ { k } \mathcal { G } _ { k _ { q , j } } .
$$

Thus, Top-3 evaluates the union of three ranked clusters, not the third cluster alone. Table 3 reports mean ScanRefer 3D mIoU over the ten scenes. The same trained model is used for all five unions within each construction.

## B.4 Caption Source and Evaluation-Language Analysis

Caption-generator sensitivity. We compare captions from GPT-4.1, InternVL3.5-8B, and Qwen3- VL-8B using the same scene features, cluster candidates, CLIP encoder, and training settings described in Appendix B.2. Table 7 shows that GPT-4.1 gives the highest scores. The lower scores with InternVL3.5-8B and Qwen3-VL-8B show sensitivity to the caption source. Both local models still outperform the compared ScanRefer baselines in Table 1, so the gains persist with local caption generation.

Table 7: Caption-source comparison on ScanRefer. Bold values indicate the best score in each column.
<table><tr><td rowspan="2">Caption generator</td><td colspan="2">2D</td><td colspan="2">3D</td></tr><tr><td>mIoU↑</td><td>mAcc@0.25↑</td><td>mIoU↑</td><td>mAcc@0.25↑</td></tr><tr><td>GPT-4.1</td><td>39.72</td><td>54.35</td><td>23.33</td><td>43.38</td></tr><tr><td>InternVL3.5-8B</td><td>38.60</td><td>52.82</td><td>21.23</td><td>39.00</td></tr><tr><td>Qwen3-VL-8B</td><td>34.22</td><td>46.63</td><td>20.01</td><td>36.82</td></tr></table>

Scene-local lexical comparison. We compare 20,290 training captions with 958 official ScanRefer evaluation expressions from the ten scenes. Each expression is compared with all training captions from the same scene. After Unicode-NFKC normalization, lowercasing, and punctuation removal, we compute token-set Jaccard similarity:

$$
J ( A _ { q } , A _ { c } ) = \frac { \left| { A } _ { q } \cap { A } _ { c } \right| } { \left| { A } _ { q } \cup { A } _ { c } \right| } , \qquad J _ { q } ^ { \mathrm { n e a r e s t } } = \operatorname* { m a x } _ { \mathrm { s a m e - s c e n e ~ } c } J ( A _ { q } , A _ { c } ) ,
$$

where $A _ { q }$ and $A _ { c }$ are the distinct token sets of an expression and a caption. No stopword removal or stemming is applied for this metric. The median nearest-caption Jaccard similarity is 0.391.

In a separate analysis, we remove stopwords and stem alphabetic content words before comparing each expression with the training-caption vocabulary of its scene. Of the 958 evaluation expressions, 286 (29.85%) contain a content word absent from that vocabulary. The evaluation set thus includes expressions with wording and content words beyond the scene’s training captions. These lexical statistics do not measure grounding accuracy on that subset separately.

## B.5 Ref-LERF Quantitative Results

Table 8 gives the per-scene results summarized in Section 4.2. Ref-LERF complements the roomscale ScanRefer evaluation with four small-scale scenes and fine-grained referring expressions. ClusterSplat achieves the highest average mIoU and mAcc@0.25. The gains on scenes such as teatime suggest that selecting a coherent cluster provides more stable object support under detailed descriptions.

Table 8: Ref-LERF quantitative results in percentages. Bold and underlined values indicate the best and second-best distinct scores in each column; ties receive the same marking. ClusterSplat improves the average referring segmentation scores across the evaluated scenes.
<table><tr><td rowspan="2">Methods</td><td colspan="5">mIoU↑</td><td colspan="5">mAcc@0.25↑</td></tr><tr><td>ramen</td><td>figurines</td><td>teatime</td><td>kitchen</td><td>Avg.</td><td>ramen</td><td>figurines</td><td>teatime</td><td>kitchen</td><td>Avg.</td></tr><tr><td>LangSplat [5]</td><td>8.87</td><td>21.78</td><td>3.04</td><td>11.39</td><td>11.27</td><td>7.89</td><td>38.10</td><td>3.39</td><td>13.79</td><td>15.79</td></tr><tr><td>OpenGaussian [6]</td><td>3.96</td><td>24.42</td><td>8.49</td><td>10.43</td><td>11.83</td><td>2.63</td><td>39.68</td><td>10.17</td><td>20.69</td><td>18.29</td></tr><tr><td>Dr.Splat [10]</td><td>6.21</td><td>7.25</td><td>2.90</td><td>6.76</td><td>5.78</td><td>9.21</td><td>9.52</td><td>0.00</td><td>6.90</td><td>6.41</td></tr><tr><td>ReferSplat [12]</td><td>28.81</td><td>14.14</td><td>35.54</td><td>23.44</td><td>25.48</td><td>39.47</td><td>17.46</td><td>55.93</td><td>31.03</td><td>35.97</td></tr><tr><td>ReferSplat (w/ our loss)</td><td>33.06</td><td>16.82</td><td>36.26</td><td>20.21</td><td>26.59</td><td>38.16</td><td>25.40</td><td>54.24</td><td>27.59</td><td>36.35</td></tr><tr><td>ClusterSplat (Ours)</td><td>35.27</td><td>26.18</td><td>41.39</td><td>21.71</td><td>31.14</td><td>43.20</td><td>34.92</td><td>57.63</td><td>31.03</td><td>41.70</td></tr></table>

## B.6 Efficiency Analysis

Table 9 reports preparation costs and online efficiency on an NVIDIA RTX A6000. ClusterSplat requires 125.3 min of method-specific preparation per scene, 10.2% more than ReferSplat. This cost is incurred once per scene, whereas online targeting is repeated for every expression. ClusterSplat takes 19.55 ms per query, about 6.1 times faster than ReferSplat, while achieving higher grounding accuracy. The trade-off is therefore more preparation for lower latency during repeated interaction with a prepared scene.

Online targeting includes text encoding, cluster scoring, and target Gaussian selection; 2D rendering is measured separately. Offline costs average ten scenes and exclude shared RGB reconstruction and cache preparation. ClusterSplat’s value sums separately timed instance-feature learning, cluster construction, and Stage 2 training with different scene representations.

Table 9: Offline preparation costs and online efficiency on an NVIDIA RTX A6000. Offline costs cover method-specific stages, excluding RGB reconstruction and cache preparation. Accuracy is in percentages. Bold and underlined values indicate the best and second-best distinct scores in each column.
<table><tr><td>Method</td><td>Offline Preparation Time (min/scene)↓</td><td>Online Targeting Time (ms/query)↓</td><td>Rendering FPS↑</td><td>ScanRefer Accuracy 2D mIoU↑</td><td>3D mIoU↑</td></tr><tr><td>LangSplat</td><td>21.6</td><td>20.45</td><td>0.28</td><td>11.00</td><td>5.81</td></tr><tr><td>OpenGaussian</td><td>25.8</td><td>43.74</td><td>13.10</td><td>18.48</td><td>1.06</td></tr><tr><td>Dr.Splat</td><td>2.0</td><td>13.20</td><td>68.00</td><td>19.01</td><td>9.48</td></tr><tr><td>ReferSplat</td><td>113.7</td><td>119.71</td><td>1.13</td><td>31.30</td><td>4.05</td></tr><tr><td>ClusterSplat (Ours)</td><td>125.3</td><td>19.55</td><td>51.12</td><td>39.72</td><td>23.33</td></tr></table>

## B.7 Additional Qualitative Results

We provide visual comparisons to examine model behavior under complex queries.

Figure 6 complements the Ref-LERF quantitative comparison with fine-grained descriptions that include attributes, surface cues, and spatial relations. These examples test whether a method can localize targets described by detailed appearance cues, such as a green fruit-like object, writing on an object’s surface, or a silver baking machine with a sleek exterior, as well as spatial prepositions such as the slices above the ramen noodles. Baseline masks often merge the target with nearby background, miss object parts, or spread support to distractor regions, whereas ClusterSplat keeps the predicted masks more concentrated on the referred objects across these examples.

![](images/b7896d90e45ab12cfb26183b9d97893f6e31e20c85977de2ed8c95a494fdd584.jpg)  
Slices cut evenly in half down the center, above the ramen noodles.

Figure 6: Ref-LERF qualitative results. Each row shows a fine-grained referring description with attributes, surface cues, or spatial relations. ClusterSplat produces compact masks that better follow the referred object across the shown examples.

Figures 7 and 8 visualize ScanRefer qualitative results for all 10 evaluation scenes. These examples are intended to show qualitative coverage across the evaluation split, including attribute-based descriptions and spatial relations. The rows compare rendered 2D masks with explicit top-down 3D target selections, illustrating that ClusterSplat generally keeps support concentrated on the referred object while reducing distractor responses.

Figure 9 provides an additional ScanNet v2 19-class visualization across all 10 evaluation scenes. The annotated point-cloud labels and Gaussian pseudo labels illustrate the evaluation label construction, while the method columns compare class-level scene labeling quality across 3DGS baselines.

Ground Truth

LangSplat

OpenGaussian

Dr.Splat

ReferSplat

Ours  
![](images/c0b468ac516e411af7b22f798f6f87190054492ab1d8b0ee2df7ee8967ddb658.jpg)  
scene0140\_00  
the office chair is underneath the blue painting. the office chair has a slightly bent backside and five legs.

Figure 7: ScanRefer qualitative results on the first five evaluation scenes. Each row shows the groundtruth mask, rendered mask predictions, and top-down 3D target selections for a referring expression. The examples illustrate compact, object-aligned support with reduced distractor responses.

![](images/741b94e0e2e713feccff3dbaa8d53d62a5d5c76b6ccb4fdc917e9d3c0d2e526e.jpg)  
scene0645\_00  
the chair is facing the table to the left. the chair is slightly pulled away from the table.

Figure 8: ScanRefer qualitative results on the remaining five evaluation scenes. The examples cover attribute-based and spatial referring expressions, showing consistent 2D mask concentration and top-down 3D target localization across the evaluation scenes.

GT Point cloud with labels

3D Gaussians with pseudo labels

OpenGaussian

Dr. Splat

ReferSplat

Ours

![](images/fcff166628947039965605e8eda1c19fb0465a87d00e17e7ceec629553033770.jpg)  
Figure 9: Top-down ScanNet v2 19-class visualization across all 10 evaluation scenes. The first two columns compare annotated ScanNet v2 point-cloud labels with Gaussian pseudo labels, followed by class-level predictions from 3DGS baselines and ClusterSplat.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the 3D visual grounding setting, the cluster-level selection formulation, and the ScanNet v2/ScanRefer compatible evaluation protocol; these claims are supported by the empirical results in Sections 4.2 and 4.3, and Appendices B.6 and B.7.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 5 discusses cluster merging and fragmentation, single-target grounding, limited scene coverage, reliance on annotated supervision, and preparation costs for each new scene.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not present theoretical theorems or formal proof claims. The mathematical expressions define the model, clustering objective, and evaluation protocol.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Sections 3–4 and Appendix A describe the model architecture, data preparation, held-out split, pseudo-label construction, baseline configurations, hyperparameters, and evaluation protocol. The supplementary material will include code and reproduction instructions.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: An anonymized supplementary code package will be submitted with training and evaluation scripts, configuration files, and data-preparation instructions. The paper uses public ScanNet v2, ScanRefer, and Ref-LERF resources, with preprocessing details described in Appendix A.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Appendix A.1 reports the software environment, optimizer, learning rate, training iterations, and architecture details. Appendix A.4–A.6 describe the data split, supervision construction, evaluation protocol, and baseline settings.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experiments report quantitative benchmark statistics, including mIoU, mAcc, Top-1/Top-3 accuracy, ablation metrics, and efficiency measurements on fixed evaluation sets. All compared methods are evaluated under the same splits and metric definitions, so these reported statistics provide the basis for assessing the experimental differences.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Appendix A.1 specifies the RTX 5090 and RTX A6000 hardware and software environment used for the experiments. Appendix B.6 and Table 9 report method-specific offline preparation time, online query latency, and rendering throughput.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The work uses established public research datasets and does not collect private data or conduct new human-subject studies. The submitted paper and supplementary material are anonymized for review.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: The work is a technical study of 3D visual grounding on established public indoor-scene benchmarks. It does not collect new personal data, release a deployed user-facing system, or target safety-critical decision making, so we do not identify direct societal impacts specific to this method beyond the general research context of 3D scene understanding.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release a high-risk pre-trained language model, image generator, or scraped dataset. The supplementary release is code for reproducing the reported 3D visual grounding experiments.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper cites the datasets, baselines, and models used in the experiments, including ScanNet v2, ScanRefer, Ref-LERF, CLIP, and the compared 3DGS baselines. The supplementary code package will include attribution, license notices where applicable, and instructions requiring users to follow the original dataset terms.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The paper does not introduce a new dataset, but it will submit an anonymized supplementary code asset. The supplementary package will document installation, data preparation, training, and evaluation steps for reproducing the reported experiments.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The work does not involve new crowdsourcing experiments or human-subject studies. It uses existing public datasets and annotations.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not conduct new research with human subjects, so IRB approval or an equivalent review is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: Appendix A.5 describes the use of GPT-4.1 as a VLM to generate pseudocaptions from training-frame instance masks. The appendix specifies the inputs provided to the model and states that official ScanRefer expressions, evaluation masks, and evaluation target IDs are not used in this generation process.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.