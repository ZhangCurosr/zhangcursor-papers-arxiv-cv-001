# End-to-End Cell Detection via Instance-aware Graph Modeling

Ruochen Liu<sup>a</sup>, Yalin Zheng<sup>b</sup>, Jingxin Liu<sup>c</sup>, Jianfeng Zhang<sup>d</sup>, Shoujun Huang<sup>d</sup>, Dexing Kong<sup>e</sup>, Haofeng Li<sup>f,∗</sup> and Wei Lou<sup>d,∗</sup>

<sup>a</sup>Faculty ofScience and Engineering, University ofLiverpool, Liverpool, L69 3GJ, U.K.

<sup>b</sup>Department of Eye and Vision Sciences, University of Liverpool, Liverpool, L7 8TX, U.K.

<sup>c</sup>School ofAI and Advanced Computing, Xi’an Jiaotong-Liverpool University, Suzhou, 215400, China

<sup>d</sup>College of Mathematical Medicine, Zhejiang Normal University, Jinhua, 321004, China

<sup>e</sup>School ofMathematical Sciences, Zhejiang University, Hangzhou, 310027, China

<sup>f</sup>School of Systems Science and Engineering, Sun Yat-sen University, Guangzhou, 510275, China

## A R T I C L E I N F O

<sub>Keywords:</sub>0 Cell and Nucleus Detection   
Graph Learningp   
End-to-End Learninge   
Query-based DetectionS   
Selective State Space Model

## A BS T R AC T

Accurate cell detection and classification are crucial for pathological analysis, directly afecting diagnostic accuracy and treatment planning. To capture complex cellular interactions beyond visual appearance within the tumor microenvironment, several approaches have employed graph neural networks to model spatial and relational patterns among cell nuclei, yielding promising results. However, these methods typically adopt a two-stage paradigm of visual extraction followed by relational modeling, which necessitates separate tuning for each stage, thereby increasing pipeline complexity and hindering end-to-end joint optimization. In this paper, we propose an end-to-end framework for cell detection and classification that jointly models patchlevel visual representations and instance-level interactions, which incorporates a dynamic graph construction module and an instance-aware graph network. Specifically, the graph construction module dynamically builds the graph structure using learnable queries derived from patch-level features as cell instance representations, with adjacency defined by integrating feature similarity and spatial distances. The instance-aware graph network performs adaptive instance filtering and feature reorganization, aggregating them over the cell graph into a topological latent state for a selective state-space transition driven by visual cues, fusing appearance and relational evidence. When evaluated on multiple datasets with diferent staining protocols for cell and nucleus detection, our method significantly outperforms existing approaches in both detection and classification performance. The code will be released at https://github.com/RuochenLiu23/IGM.

## 1. Introduction

Histopathological images are the gold standard for diagnosing and grading complex diseases such as cancer, playing a critical role in clinical diagnostics and public health [4]. Cell detection and classification, which involve identifying cell or nucleus locations and predicting their types, provide essential quantitative and qualitative support for diagnosis and prognosis [43]. However, manual analysis is labor-intensive and costly due to the thousands of cells typically present in whole slide images (WSIs) [3, 40]. Although computer-aided methods enable rapid and accurate cell identification [40], their performance is challenged by dense clustering, extensive overlap, and adhesion that make adjacent instances dificult to separate [41, 16, 50, 29], ambiguous boundaries caused by uneven staining and imaging noise [41, 25, 47, 29], and high cellular heterogeneity that manifests as intra-sample variability in morphology and staining intensity as well as diferences across laboratory protocols [16, 54, 38, 36, 33].

Extensive research has been dedicated to this challenging task, with cell and nucleus detection approaches broadly categorized into three groups: point-based, instance segmentation, and instance detection methods [33]. Pointbased methods [1, 52, 53, 51] predict only centroids and their categories to reduce computational complexity and annotation costs. However, they lack explicit regional information, making it challenging to recover cell shape or boundary details. This limits their ability to support downstream morphological analyses such as chromatin pattern characterization [61, 68] or mitotic figure quantification [2, 19]. In contrast, instance segmentation methods [41, 16,

![](images/9d8e62e7f989f0392fa43f7bfc87d8972a12f408fd4c601740faa13154d41267.jpg)  
Figure 1: Comparison of diferent cell detection and classification approaches incorporating graph models: (A) Graph construction is initialized using the output of a pre-trained detector. (B) Graph construction is initialized based on the feature map. (C) Ours: Dynamic graphs are initialized using learnable queries. The Instance-aware Graph Network (IGN) performs adaptive noise filtering and feature reorganization, enabling efective capture of local-global relationships.

50, 29, 54, 38, 36, 7, 26] ofer pixel-level delineation of individual cells but demand extensive annotations and high computational resources [55]. Instance detection [19, 11, 67, 33] strikes a favorable balance: it provides explicit regional localization via bounding boxes, with lower annotation and computational overhead than segmentation, while ofering richer spatial detail than point-based approaches, making it well-suited for clinical deployment. Accordingly, we aim to develop an accurate and eficient framework for cell detection and classification.

Within the instance detection paradigm, query-based Transformer models [5, 62, 64, 44] excel by using learnable queries to represent objects. However, in cell detection, their reliance solely on visual and positional cues leads to ambiguity in dense, overlapping, or morphologically heterogeneous regions with high inter-class similarity. Pathologyspecific methods [41, 16, 50, 7, 26] mitigate this by incorporating geometric priors (distance maps or contour fitting) but require pixel-level annotations and are incompatible with detection-only settings. With the rise of pathology foundation models, recent methods counter such ambiguity by enriching instance representation with pre-trained features, yet the objective gap between representation learning and coordinate regression degrades these features under joint optimization, necessitating decoupled multi-stage training [60]. In general, these approaches follow the conventional computer vision paradigm, modeling cells largely in isolation, prioritizing local appearance over interinstance contextual relationships. This limits detection performance in complex tissue architectures where cellular identity is inherently relational.

To address these challenges in pathology, previous studies have focused on intercellular interactions to provide additional cues, leading to the exploration of graph-based methods to model such contextual relationships in the tissue microenvironment. Two-stage approaches [28, 37, 20, 35, 13, 63] (Figure 1(A)) first detect or segment cells and then build graphs from cropped regions, but sufer from error propagation from imperfect proposals and increased training complexity due to separate per-stage optimization. In contrast, pixel- or superpixel-based methods [39, 14] (Figure 1(B)) construct graphs directly from feature maps with pixels as nodes, yet struggle to model meaningful inter-instance relationships while producing overly dense and noisy structures that obscure instance-level information.

Inspired by these observations, we propose an end-to-end framework to enhance instance awareness in query-based cell detection and classification. As shown in Figure 1(C), our method treats learnable queries as graph nodes which represent candidate cell instances and construct a context-aware graph. The graph connectivity is determined by both spatial proximity and feature similarity, enabling dynamic adaptation to diverse cell distributions and morphologies. However, since the number of queries typically far exceeds the true cell count, the graph contains redundant and spurious nodes, introducing significant structural noise. This poses a major challenge for conventional GNNs [31], which rely solely on local message passing and thus struggle to capture long-range dependencies in such large, noisy graphs. Although Transformer-based GNNs [58, 30] can model global relationships, their dense attention mechanisms incur substantial computational overhead, especially when applied to noisy graphs with excessive nodes. Irrelevant nodes in such cases not only inflate computation cost but also distort attention, undermining inter-instance relationship modeling. To address this, we leverage the selective State Space Model (SSM) [17], which efectively suppresses noise while preserving global structural patterns, thereby enabling robust relational modeling at linear computational cost. We embed the graph topology into the SSM, jointly modeling inter-cellular relations and visual representations.

In this paper, our framework comprises three key components: a query feature learning network, a dynamic graph construction module (DGC), and an instance-aware graph network (IGN). The query feature learning network fuses initial query features with tissue contextual and pathological features extracted from patches, producing enriched query embeddings that encode instance-level semantics, bridging patch-level features to instance-level representations. The DGC module takes the queries as graph nodes and dynamically constructs the graph structure based on both spatial distance and feature similarity among queries. This adaptive construction strategy allows the graph to better capture varying cell distributions and morphologies, enhancing the model’s robustness during training. The IGN integrates two synergistic components: an Instance-aware Graph Learning (IGL) module, in which Selective Feature Reorganization (SFR) gates out redundant queries and reorganizes the retained ones into a directionally aligned latent space, whereupon graph aggregation superposes neighbors into near-orthogonal subspaces, yielding a noise-resistant topological representation; and a Topology-structured State Space (TSS) layer, where the state is transitioned along the cell-graph topology and query’s visual feature drives state update, so that relational and visual evidence accumulate within a unified state. To account for the high semantic complexity of histopathological images, both IGL and TSS are augmented with a Context-Guided State Encoding module, which injects context-aware priors into dynamic state transitions. Finally, the refined node features from the TSS are decoded into cell instance predictions, including bounding box regression and class labels.

Our main contributions are summarized as follows:

1. We propose a new end-to-end cell detection framework that leverages learnable queries for graph construction and relational modeling, aiming to enhance instance awareness and detection performance.

2. We propose an instance-aware graph network that recasts graph learning as a topology-structured state space model, in which selective reorganization and orthogonal aggregation yield a noise-resistant topological structural representation as the latent state driven by visual evidence, fusing relational and appearance cues within the state transition.

3. We propose a dynamic graph construction method that integrates both the spatial relationships and feature similarities of queries to build graphs and in turn adaptively adjusts their structure in response to dynamic updates in query content.

4. Experimental results demonstrate that our framework achieves state-of-the-art performance on multiple public benchmarks for cell and nucleus detection across diferent staining protocols.

## 2. Related Work

## 2.1. Instance Detection for Cells and Nuclei

Cell detection and classification typically rely on instance segmentation or point detection. Segmentation methods [50, 16, 54, 7, 26, 56] like Hover-Net [16] and StarDist [50] provide detailed masks but require costly pixelwise labels and are computationally expensive. Point-detection approaches [1, 52, 53, 51], while eficient, only predict centroids and lack geometric information needed for morphometric analysis. Bounding-box-based instance detection [19, 11, 67, 33] ofers a balanced alternative, providing richer spatial localization than point detection while avoiding the overhead of full segmentation. Recent query-based detectors [5, 62, 64, 44, 66, 34, 69, 8, 46, 45] such as Cell-DETR [45] and DINO [62, 44] enable end-to-end, anchor-free instance detection with strong global context modeling. However, these models have seen limited exploration in the field of pathology, as they primarily involve the adaptation of general-purpose models, leaving a need for more pathology-specific approaches. These limitations underscore the necessity of a cell instance detection framework.

## 2.2. Graph-Based Learning in Pathology

In computational pathology, graph-based cell modeling typically involves two stages: graph construction and graph representation learning. For graph construction, two strategies dominate: (1) Two-stage methods [28, 37, 20, 35, 13] first detect or segment cells to form semantically meaningful nodes and connect them via spatial proximity, but they sufer from error propagation and pipeline complexity. (2) End-to-end pixel- or patch-level approaches [14] build graphs directly from feature maps using �-nearest neighbors (KNN), but often produce redundant or biologically irrelevant edges. For graph representation learning, Graph Convolutional Networks (GCNs) [31, 15, 6, 10] and Graph Attention Networks (GATs) [58] are widely adopted. GCNs perform fixed-weight neighborhood aggregation but sufer from limited receptive fields and over-smoothing in deep architectures. GATs improve adaptivity by assigning attention-based weights to neighbors, but standard attention mechanisms incur quadratic computational complexity with respect to the number of nodes [57], which hinders scalability in dense cellular graphs containing thousands of instances. Recently, Mamba has been introduced into graph learning [12], ofering global modeling at linear complexity. Motivated by these observations, in this work, we propose an eficient dynamic graph learning approach that reformulates the state transition of a selective SSM over the cell graph topology, thereby achieving global contextual modeling of relational and visual representations at low complexity.

![](images/06fade2e94db1f10a6a8e2aac515cf42b7c7a41cdf7a01d9df1a2a2036bca352.jpg)  
Figure 2: Overview of the whole framework. The proposed end-to-end learning framework comprises three key components: a query feature learning network for initializing query embeddings with patch-to-instance-level mapping, a dynamic graph construction (DGC) module for building the cell graph, and an instance-aware graph network (IGN) for efective graph modeling of inter-cell interactions, all jointly optimized during training.

## 3. Method

In this section, we introduce the proposed end-to-end cell detection and classification framework, which is shown in Figure 2. The framework consists of three key components: (1) a query feature learning network that learns tissue and pathological features, extracts high-quality query representations from patches, achieving the mapping from patchlevel to instance-level features; (2) a dynamic graph construction strategy that adaptively builds the cell graph by jointly leveraging spatial proximity and feature similarity among learnable queries; and (3) an instance-aware graph network that learns cell-level features by modeling local and global relational contexts with the visual evidence to enhance the discriminative ability of query features for both instance separation and class prediction.

## 3.1. Query Feature Learning Network

Given a pathological image of size $H \times W \times 3$ , it is fed into a backbone network to obtain multi-scale visual features $f _ { \mathrm { m u l t i } } = \{ f _ { 0 } , f _ { 1 } , \ldots , f _ { l } \}$ . These features are flattened and concatenated into an initial global feature $f \in \mathbb { R } ^ { N _ { \mathrm { e n c } } \times C }$ where $N _ { \mathrm { e n c } }$ is the total number of spatial positions across all scales. A 2D sinusoidal positional encoding is generated for each spatial location, flattened, and added to $f ,$ , resulting in $f _ { \mathrm { e n c } } \in \mathbb { R } ^ { N _ { \mathrm { e n c } } \times C }$ . The features are then processed by a Transformer encoder using multi-scale deformable attention [66], which captures fine-grained pathological details and aggregates contextual information from the tissue microenvironment to produce enhanced visual representations $f _ { \mathrm { e n c } } ^ { \prime } \in \mathbb { R } ^ { \mathbf { \breve { N } } _ { \mathrm { e n c } } \times C }$ . From $f _ { \mathrm { e n c } } ^ { \prime } { \mathrm { . } }$ , we generate initial object proposals. Specifically, each spatial position corresponds to a candidate box $\pmb { a } = [ c _ { x } , c _ { y } , w , h ]$ , where $( c _ { x } , c _ { y } )$ denote the center coordinates and (�, ℎ) are randomly initialized dimensions. After applying class-agnostic non-maximum suppression (NMS), the top-� high-scoring proposals are selected. Their features are extracted from $f _ { \mathrm { e n c } } ^ { \prime }$ at $( c _ { x } , c _ { y } )$ and linearly transformed to form the anchor features $a _ { s } \in \mathbb { R } ^ { N \times C }$

Similar to common detection pipelines [66, 62, 64], for each anchor, a lightweight prediction head takes its associated feature to compute a score and a bounding-box regression ofset, facilitating the selection of highquality detection proposals: we first apply class-agnostic non-maximum suppression (NMS) on their boxes to remove duplicates, then keep the top-� anchors by score (� ≫ the actual object count). To improve the quality of query embeddings, the selected anchor features $\bar { a _ { s } } \doteq \mathbb { R } ^ { N \times C }$ are added to a set of learnable query vectors $q ^ { \prime } \in \dot { \mathbb { R } } ^ { N \times \dot { C } }$ , yielding the final query features $Q _ { \mathrm { e n c } } \in \mathbb { R } ^ { N \times C }$ . These learnable query vectors serve as instance-level priors for potential objects (e.g., cells), which are randomly initialized and optimized during training but remain fixed during inference. The query features $Q _ { \mathrm { e n c } }$ , together with 4D reference points $( c _ { x } , c _ { y } , w , h )$ normalized to [0, 1], are fed into the Transformer decoder.

In the Transformer decoder, each decoder layer iteratively updates the query features. First, positional embeddings are generated from the current reference points using sine-based encoding followed by a MLP layer. Each query is updated through self-attention and multi-scale deformable cross-attention, enabling it to gather visual evidence from relevant regions across encoder feature scales. Owing to the dense sampling of initial queries, the receptive fields of queries overlap with each other. To further reduce redundancy and ensure the distinctness of queries for accelerating model convergence, between decoder layers, the model selects a subset of high-quality, non-overlapping queries using class-agnostic NMS with an IoU threshold of 0.8, based on the current 4D reference points and their classification scores.

Following the design of [64], a subset of queries determined by the configuration is treated as dense and routed to the auxiliary head for refinement to ensure the integrity of feature representations, while they are not used for final predictions at inference. In addition, 100 denoising (DN) queries are added only during training as noisy copies of ground-truth boxes and labels to provide direct supervision and speed up convergence. Finally, the Transformer decoder outputs the updated query features � and predicts the ofset of the selected anchor positions. Combining these ofsets with the original anchor allows us to calculate the updated reference points, as well as the extraction of the center coordinates $P \in \bar { \mathbb { R } ^ { N \times 2 } }$ , providing key spatial cues for subsequent cell graph construction and relationship modeling.

## 3.2. Dynamic Graph Construction

Existing graph-based cell classification methods typically use K-Nearest Neighbor (KNN) based on spatial proximity to construct cell graphs [28, 37, 20, 14]. However, in our end-to-end framework, the 2D coordinates of reference points are not always reliable. Furthermore, solely spatial information has limited capacity to model complex intercellular distributions and interactions, rendering such constructions sensitive to noise and harmful to convergence. To address this, we propose a Dynamic Graph Construction (DGC) strategy that combines both query embeddings and positional information to define a more robust query distance function in KNN, enabling accurate relationship modeling. Notably, by constructing the graph over queries, whose features and reference points are dynamically updated during training, the cell graph structure is refined via gradient propagation through query features. This dynamic refinement enhances flexibility and avoids the limitations of fixed graph topologies.

Based on the query embeddings � and reference points � output by the query feature learning network, we design a query distance function � for KNN computation, which consists of two components: spatial distance $s ^ { p }$ and feature distance $s ^ { f }$ . For spatial distance, the distance is defined via an exponential decay of the Euclidean distance between any two reference points $p _ { i }$ and $p _ { j }$

$$
s _ { i j } ^ { p } = \exp \left( - \frac { \| p _ { i } - p _ { j } \| _ { 2 } } { 2 \bar { d } } \right)\tag{1}
$$

where $\| p _ { i } - p _ { j } \| _ { 2 }$ is the Euclidean distance between $p _ { i }$ and $p _ { j }$ , and $\bar { d }$ denotes the mean distance between all pairs of reference points. A large $s ^ { p }$ indicates stronger spatial correlation.

For feature distance, we first perform L2 normalization on the query embeddings $Q$ to eliminate the influence of feature scale. Subsequently, we measure feature distance $s _ { i j } ^ { f }$ using cosine similarity between any two queries $q _ { i }$ and $q _ { j }$ The formula is as follows:

$$
s _ { i j } ^ { f } = \frac { q _ { i } \cdot q _ { j } ^ { T } } { \Vert q _ { i } \Vert _ { 2 } \Vert q _ { j } \Vert _ { 2 } }\tag{2}
$$

In the end, the query distance function can be described as:

$$
s _ { i j } = \alpha \cdot ( 1 - s _ { i j } ^ { p } ) + ( 1 - \alpha ) \cdot ( 1 - s _ { i j } ^ { f } )\tag{3}
$$

where $\alpha$ is a hyper-parameter that controls the contribution weights of the two distances. For dynamic balance in this specific setting, � is set to 0.5. Then, we apply the KNN algorithm to establish node connectivity based on the query

![](images/7c6f45bd2b8557052fd17cb029d398b4ecd912fe670e55158f82f9ded47b9d9a.jpg)  
Figure 3: The proposed Dynamic Graph Construction (DGC) strategy and Instance-aware Graph Network (IGN) are employed to construct and model the query-based graph for simulating inter-cell interactions.

distance $s _ { i j } ,$ , resulting in an adjacency matrix $M \in \mathbb { R } ^ { N \times N }$ , where � is the number of queries. The constructed graph is defined as $G = ( V , E )$ , where each node $v _ { i } \in V$ corresponds to a query embedding �, and edges � are determined by the KNN connections

## 3.3. Instance-aware Graph Network

We propose an Instance-aware Graph Network (IGN) for dynamic query-based cell detection, as illustrated in Figure 3. IGN comprises two tightly coupled components: the Instance-aware Graph Learning (IGL) and the Topologystructured State Space (TSS) layer. In IGL, Selective Feature Reorganization (SFR) applies a selective mechanism to suppress redundant query features and reorganizes the retained features into a directionally structured latent space. The resulting representations are then aggregated over the dynamic cell graph, where near-orthogonal directions suppress interference from semantically inconsistent neighbors, yielding a noise-resistant topological representation of semantic associations. Within TSS, the state is updated over this representation under the drive of instance-level query features, integrating contextual information with global relational dependencies. A Context-Guided State Encoding (CGSE) module abstracts all queries into an instance-level context that steers the state-transition parameters of both SFR and TSS, guiding adaptation to instance-specific feature distributions.

## 3.3.1. Instance-aware Graph Learning

Within this module, selective feature reorganization is coupled with graph aggregation. SFR first gates out redundant queries and lifts the retained ones into a directionally structured latent space, rendering dissimilar cells nearorthogonal. Aggregating these over the dynamic cell graph then consolidates the orthogonalized node representations with graph connectivity into a unified topological structural representation, wherein inconsistent neighbors reside in mutually orthogonal subspaces and remain non-interfering. Similarity-driven selective aggregation therefore arises from the state geometry itself, obviating explicit edge pruning and rendering the structure robust to redundant nodes. Selective Feature Reorganization. Given a set of nodes �, we flatten their node features into a sequence $x \in \mathbb { R } ^ { N \times d }$ and feed it into the Selective Feature Reorganization (SFR) module. Inspired by the selective state space model [17], we apply a selective projection to adaptively filter and reconstruct each node feature $x _ { i } \in \mathbb { R } ^ { d }$ , where the projection matrices $\Delta \in \mathbb { R } ^ { N \times d }$ and $\mathbf { B } \in \mathbb { R } ^ { N \times n }$ are dynamically determined by Context-Guided State Encoding, yielding the

hidden state $h _ { i } \in \mathbb { R } ^ { d \times n }$

$$
h _ { i } = \left( \mathbf { \Delta } \mathbf { \delta } \otimes x _ { i } \right) \otimes \mathbf { \delta } \mathbf { B } _ { i }\tag{4}
$$

where ⊗ denotes the outer product.

$\Delta _ { i } \in \mathbb { R } ^ { d }$ serves as an element-wise amplitude gate that controls the amount of information written from each node feature into the state space. By implicitly encoding the node information through CGSE, Δ achieves content-adaptive modulation, enabling efective filtering and compression of noisy node features in a fully data-driven manner.

$\mathbf { B } _ { i } \in \mathbb { R } ^ { n }$ expands each scalar component of $x _ { i } \in \mathbb { R } ^ { d }$ onto the direction it spans, yielding $h _ { i } \in \mathbb { R } ^ { d \times n }$ . Lifting node features transforms the entangled space into a directionally discriminative one, mapping similar node vectors $h _ { i }$ to proximal directions and dissimilar ones to near-orthogonal directions, whereby the afinity between two states factorizes as cos $( h _ { i } , h _ { j } ) = \cos ( \tilde { x } _ { i } , \tilde { x } _ { j } ) \cdot \cos ( \mathbf { B } _ { i } , \mathbf { B } _ { j } )$ with $\widetilde { \boldsymbol { x } } _ { i } = \boldsymbol { \Delta } _ { i } \odot \boldsymbol { x } _ { i }$ , sharpening the contrast between related and unrelated node pairs.

Directional Graph Aggregation. Building on these near-orthogonal features, we aggregate them over the cell graph, injecting graph connectivity to form a topological representation. We remap all $h _ { i } \in \mathbb { R } ^ { d \times n }$ as node features $h \in$ $\mathbb { R } ^ { \bar { N } \times ( d \cdot n ) }$ into a cell graph $G ,$ , whose connectivity is established using the adjacency matrix $M \in \mathbb { R } ^ { N \times N }$ derived from the dynamic graph construction strategy. To balance node degree diferences and enhance stability, we first perform symmetric normalization on it [31], followed by aggregating adjacent features based on structural information. The relevant formulas are as follows:

$$
\hat { h } = D ^ { - 1 / 2 } \cdot ( M + I ) \cdot D ^ { - 1 / 2 } \cdot h\tag{5}
$$

where � denotes the identity matrix for adding self-loops,  represents the degree matrix, and the two $\scriptstyle { \mathcal { D } } ^ { - 1 / 2 }$ terms perform symmetric normalization on the target and neighbor nodes respectively. Node-wise, each node takes a degreenormalized weighted sum of its own state and those of its neighbors, whereby the graph connectivity determines which neighbors are included and the normalization determines how much each contributes.

Since each node’s state is confined to the direction assigned by its own $\mathbf { B } _ { i } .$ , this weighted sum acts as a superposition along distinct directions rather than a blending of values. Semantically inconsistent neighbors, whose states point in near-orthogonal directions, therefore enter the aggregated state without contaminating the target node’s own state and remain separable at readout. Selectivity thus resides in the geometry rather than in the edges, requiring no explicit pruning, and the aggregated ℎ<sup>̂</sup> constitutes a noise-resistant topological structural representation consolidating node features with graph connectivity.

## 3.3.2. Topology-structured State Space

Due to the complexity and global nature of graph structures, we employ Mamba [17] to model global features. Specifically, we propose the Topology-structured State Space (TSS) layer, which adopts the topological structure as the latent state and drives its update with the visual cues of the queries, thereby constructing a topology-structured state transition that yields the output �:

$$
H = \bar { \mathbf { A } } \hat { h } + \bar { \mathbf { B } } x , \quad y = \mathbf { C } H\tag{6}
$$

In contrast to a canonical state space update, driven by the state of the preceding token, here the cell topology $\hat { h }$ obtained by graph aggregation serves as the incoming state, with the visual features � continually re-injected to drive each update. Instance-level appearance is thereby assimilated into the topological structure, unifying relational and appearance evidence within a single state �.

Since Mamba is defined on the continuous time, Zero-Order Hold (ZOH) discretization is applied to adapt the continuous-time model to discrete node sequence inputs and facilitate training:

$$
\bar { \mathbf { A } } = \exp ( \Delta \mathbf { A } ) , \quad \bar { \mathbf { B } } = ( \Delta \mathbf { A } ) ^ { - 1 } \left( \exp ( \Delta \mathbf { A } ) - \mathbf { I } \right) \cdot \Delta \mathbf { B }\tag{7}
$$

Herein, � and � are all parameter matrices, which are discretized based on the parameter Δ and their specific parameters are determined by CGSE.

Unlike Transformer, which explicitly computes pairwise similarities at quadratic complexity to obtain the global state, Mamba implicitly encodes global instance and topological context into a compact hidden state �. Transformer attention weights, by contrast, tend to disperse over complex graph structures, leaving irrelevant node signals dificult to suppress. Within TSS, �<sup>̄</sup> adaptively determines, conditioned on the representation of the corresponding instance, how much of the accumulated topological structure survives each update, so that the contextual topology aggregated onto visually substantiated cells persists. �<sup>̄</sup> writes the visual feature of each query into the very state that carries the topology, along the direction spanned by $\mathbf { B } _ { i }$ and scaled to the salience of that instance, thereby driving the state forward at every step. � performs the inverse operation, projecting the fused state back onto content-dependent directions to retrieve for each node its relevant global context, which keeps the readout selective at linear complexity. Meanwhile, cells admit no canonical ordering within tissue and thus form an unordered set rather than a sequence. We process nodes in parallel and keep the topological propagation permutation-equivariant, thereby avoiding the ordering bias inherent to sequential formulations.

To enhance feature propagation and enrich representational capacity, a learnable feedforward matrix � is introduced to form a residual-style connection, yielding the IGN output $Q _ { f } = y +  { \mathbf Ḋ x Ḍ } _ { 0 }$

The fused features $Q _ { f }$ are fed into the classification head to predict the object categories. Simultaneously, $Q _ { f }$ and its corresponding reference point are used by the regression head to predict the bounding box coordinates. Finally, predictions are matched to ground truths via the Hungarian algorithm, and the overall loss is computed accordingly.

$$
\mathcal { L } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { b b o x } } \mathcal { L } _ { \mathrm { b b o x } } + \lambda _ { \mathrm { i o u } } \mathcal { L } _ { \mathrm { i o u } }\tag{8}
$$

where $\mathcal { L } _ { b b o x }$ is the L1 Loss, $\mathcal { L } _ { i o u }$ is the GIoU Loss [48], and $\mathcal { L } _ { c l s }$ employs Focal Loss [32] to address the positivenegative sample imbalance in pathological images.

## 3.3.3. Context-Guided State Encoding.

In SSM, �, �, �, and � are the key matrices governing the state transition. We propose a Context-Guided State Encoding (CGSE) module to generate these matrices from instance features, enabling adaptive, knowledge-guided node control.

The state transition matrix � is initialized via HiPPO-LegS [18], providing a stable multi-scale decay spectrum for task-relevant retention and noise suppression during hidden state propagation. As � encodes task-level global decay structure, it is shared globally across all nodes as a single learnable parameter. For matrices � and � [17, 22], we employ a linear transformation to learn knowledge $\mathbf { W } _ { x }$ from the cell instance information. Following the tensor generation mechanism used in self-attention [57], this cellular knowledge, encompassing the morphological appearance and spatial distribution of cell instances, is transformed through learned mappings into corresponding matrices, enabling adaptive processing of nodes guided by all cell instance features. The calculation process is as follows:

$$
\mathbf { B } = f ( x , \mathbf { W } _ { x } ) \mathbf { W } _ { B } , \quad \mathbf { C } = f ( x , \mathbf { W } _ { x } ) \mathbf { W } _ { C }\tag{9}
$$

For the $\Delta ,$ , we use an activation function Softplus [17] to adapt the characteristics of the modulation matrix:

$$
\mathbf { \Delta } \Delta = \log \left( 1 + \exp ( f ( x , \mathbf { W } _ { x } ) \cdot \mathbf { W } _ { \delta } \cdot \mathbf { W } _ { d t } ) \right)\tag{10}
$$

Due to the large output dimensionality of Δ, a low-rank structure $\mathbf { W } _ { \delta } \cdot \mathbf { W } _ { d t }$ with $\mathbf { W } _ { \delta } \in \mathbb { R } ^ { d \times r }$ and $\mathbf { W } _ { d t } \in \mathbb { R } ^ { r \times d } \left( r \ll d \right)$ is additionally employed to control the parameter count.

## 4. Experiment

## 4.1. Datasets

We evaluated our method on three public histopathological datasets: CoNSeP [16], CytoDArk0 [56], and OCELOT [49], with annotations converted to bounding box labels. CoNSeP comprises 41 H&E-stained 1000×1000- pixel images of colorectal adenocarcinoma at 40× magnification with nucleus-level instance segmentation masks. CytoDArk0 includes Nissl-stained mammalian brain tissue images at 40× magnification with cell-level masks for neurons and glial cells. OCELOT is derived from H&E-stained WSIs in the TCGA database, with cell nuclei annotated. To balance eficiency and resolution, 128×128 patches are extracted from CoNSeP due to its denser cell distribution, while 256×256 patches are used for CytoDArk0 and OCELOT. Evaluations on nuclear detection, cell detection, and multi-type datasets with diverse staining methods confirm the model’s adaptability across pathological detection tasks.

## 4.2. Implementation Details

All experiments were conducted on an NVIDIA A40 GPU. We adopt ResNet-50 as the backbone, ensuring that the performance advantage stems from the detection architecture design itself rather than from backbone capacity. For training, we employed the AdamW optimizer with an initial learning rate of 2e-4 and a weight decay coeficient of 0.05. For the hyperparameter � in KNN adopted in Dynamic Graph Construction, experimental validation demonstrates that setting � = 8 on CoNSeP, � = 3 on CytoDArk0, and � = 4 on OCELOT achieves the optimal performance. To ensure fairness in the comparison, we unified the number of queries to 900 for general DETR-based methods.

Performance comparison of diferent methods on instance detection tasks across three datasets.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>AR</td><td rowspan=1 colspan=1>mAP</td><td rowspan=1 colspan=1> $\boldsymbol { \mathsf { m } } \mathsf { A } \mathsf { P } _ { 5 0 }$ </td><td rowspan=1 colspan=1> $\mathtt { m A P } _ { 7 5 }$ </td></tr><tr><td rowspan=10 colspan=1>CoNSeP</td><td rowspan=10 colspan=1>Deformable-DETR (ICLR&#x27;21)DAB-DETR (ICLR&#x27;22)DINO (ICLR&#x27;23)CO-DINO (ICCV&#x27;23)DDQ-DETR (CVPR&#x27;23)Relation-DETR (ECCV&#x27;24)Mamba-YOLO (AAAI&#x27;24)DECO (ICLR&#x27;25)CellMamba (BMVC&#x27;25)Ours</td><td rowspan=1 colspan=1>49.3</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>50.3</td><td rowspan=1 colspan=1>25.2</td></tr><tr><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>24.9</td><td rowspan=1 colspan=1>50.5</td><td rowspan=1 colspan=1>25.5</td></tr><tr><td rowspan=1 colspan=1>54.8</td><td rowspan=1 colspan=1>28.5</td><td rowspan=1 colspan=1>52.4</td><td rowspan=1 colspan=1>28.9</td></tr><tr><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>29.8</td><td rowspan=1 colspan=1>54.6</td><td rowspan=1 colspan=1>29.2</td></tr><tr><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>54.4</td><td rowspan=1 colspan=1>31.4</td></tr><tr><td rowspan=1 colspan=1>55.1</td><td rowspan=1 colspan=1>29.7</td><td rowspan=1 colspan=1>54.3</td><td rowspan=1 colspan=1>30.3</td></tr><tr><td rowspan=1 colspan=1>51.3</td><td rowspan=1 colspan=1>25.2</td><td rowspan=1 colspan=1>50.7</td><td rowspan=1 colspan=1>23.6</td></tr><tr><td rowspan=1 colspan=1>53.5</td><td rowspan=1 colspan=1>26.9</td><td rowspan=1 colspan=1>51.4</td><td rowspan=1 colspan=1>25.7</td></tr><tr><td rowspan=1 colspan=1>53.2</td><td rowspan=1 colspan=1>25.7</td><td rowspan=1 colspan=1>51.1</td><td rowspan=1 colspan=1>23.8</td></tr><tr><td rowspan=1 colspan=1>58.7</td><td rowspan=1 colspan=1>32.0</td><td rowspan=1 colspan=1>57.0</td><td rowspan=1 colspan=1>33.5</td></tr><tr><td rowspan=10 colspan=1>CytoDArk0</td><td rowspan=10 colspan=1>Deformable-DETR (ICLR&#x27;21)DAB-DETR (ICLR&#x27;22)DINO (ICLR&#x27;23)CO-DINO (ICCV&#x27;23)DDQ-DETR (CVPR&#x27;23)Relation-DETR (ECCV&#x27;24)Mamba-YOLO (AAAI&#x27;24)DECO (ICLR&#x27;25)CellMamba (BMVC&#x27;25)Ours</td><td rowspan=1 colspan=1>67.5</td><td rowspan=1 colspan=1>52.1</td><td rowspan=1 colspan=1>80.1</td><td rowspan=1 colspan=1>59.3</td></tr><tr><td rowspan=1 colspan=1>67.8</td><td rowspan=1 colspan=1>52.6</td><td rowspan=1 colspan=1>80.5</td><td rowspan=1 colspan=1>59.4</td></tr><tr><td rowspan=1 colspan=1>69.9</td><td rowspan=1 colspan=1>58.2</td><td rowspan=1 colspan=1>84.7</td><td rowspan=1 colspan=1>67.1</td></tr><tr><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>85.6</td><td rowspan=1 colspan=1>68.7</td></tr><tr><td rowspan=1 colspan=1>70.3</td><td rowspan=1 colspan=1>59.6</td><td rowspan=1 colspan=1>86.3</td><td rowspan=1 colspan=1>69.5</td></tr><tr><td rowspan=1 colspan=1>70.1</td><td rowspan=1 colspan=1>58.8</td><td rowspan=1 colspan=1>85.9</td><td rowspan=1 colspan=1>68.2</td></tr><tr><td rowspan=1 colspan=1>67.4</td><td rowspan=1 colspan=1>52.3</td><td rowspan=1 colspan=1>81.2</td><td rowspan=1 colspan=1>56.6</td></tr><tr><td rowspan=1 colspan=1>68.9</td><td rowspan=1 colspan=1>54.6</td><td rowspan=1 colspan=1>81.9</td><td rowspan=1 colspan=1>63.1</td></tr><tr><td rowspan=1 colspan=1>68.2</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>83.5</td><td rowspan=1 colspan=1>59.8</td></tr><tr><td rowspan=1 colspan=1>71.7</td><td rowspan=1 colspan=1>61.7</td><td rowspan=1 colspan=1>88.5</td><td rowspan=1 colspan=1>71.8</td></tr><tr><td rowspan=10 colspan=1>OCELOT</td><td rowspan=10 colspan=1>Deformable-DETR (ICLR&#x27;21)DAB-DETR (ICLR&#x27;22)DINO (ICLR&#x27;23)CO-DINO (ICCV&#x27;23)DDQ-DETR (CVPR&#x27;23)Relation-DETR (ECCV&#x27;24)Mamba-YOLO (AAAI&#x27;24)DECO (ICLR&#x27;25)CellMamba (BMVC&#x27;25)Ours</td><td rowspan=1 colspan=1>54.5</td><td rowspan=1 colspan=1>31.5</td><td rowspan=1 colspan=1>68.4</td><td rowspan=1 colspan=1>24.8</td></tr><tr><td rowspan=1 colspan=1>54.7</td><td rowspan=1 colspan=1>31.2</td><td rowspan=1 colspan=1>68.9</td><td rowspan=1 colspan=1>24.4</td></tr><tr><td rowspan=1 colspan=1>55.2</td><td rowspan=1 colspan=1>33.3</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=1>27.3</td></tr><tr><td rowspan=1 colspan=1>54.9</td><td rowspan=1 colspan=1>33.5</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=1>27.5</td></tr><tr><td rowspan=1 colspan=1>55.5</td><td rowspan=1 colspan=1>33.5</td><td rowspan=1 colspan=1>71.1</td><td rowspan=1 colspan=1>27.2</td></tr><tr><td rowspan=1 colspan=1>55.1</td><td rowspan=1 colspan=1>32.8</td><td rowspan=1 colspan=1>70.6</td><td rowspan=1 colspan=1>26.0</td></tr><tr><td rowspan=1 colspan=1>54.7</td><td rowspan=1 colspan=1>30.9</td><td rowspan=1 colspan=1>68.2</td><td rowspan=1 colspan=1>23.6</td></tr><tr><td rowspan=1 colspan=1>55.0</td><td rowspan=1 colspan=1>32.1</td><td rowspan=1 colspan=1>68.8</td><td rowspan=1 colspan=1>25.9</td></tr><tr><td rowspan=1 colspan=1>54.3</td><td rowspan=1 colspan=1>31.7</td><td rowspan=1 colspan=1>69.0</td><td rowspan=1 colspan=1>25.4</td></tr><tr><td rowspan=1 colspan=1>56.8</td><td rowspan=1 colspan=1>34.9</td><td rowspan=1 colspan=1>72.3</td><td rowspan=1 colspan=1>29.3</td></tr></table>

## 4.3. Evaluation Setup and Metrics

To comprehensively evaluate our model, we compare against both detection and classification methods. The comparison with detection methods spans general-purpose and pathology-specific models: general-purpose detectors, though not designed for histopathology, have shown strong capability on pathological tasks, while pathology-specific models are largely adapted from such frameworks, and both are therefore included for a thorough validation. We then compare with classification methods on the multi-class dataset, allowing us to assess our classification performance against pathology-specific models. In terms of metrics, we adopt standard detection metrics including Average Precision (���), �� $P _ { 5 0 } ,$ �� $P _ { 7 5 }$ , and Average Recall (��) to assess detection and classification performance for detection methods across confidence thresholds. For comparison with cell-specific classification methods, which are mostly supervised with masks or points and difer in output granularity, we follow [45] and adopt the F1-score, the standard metric for classification, as a consistent measure across methods.

## 4.4. Comparison with the Existing Methods

## 4.4.1. Comparison with Detection Methods

We evaluate our framework on CoNSeP, CytoDArk0, and OCELOT against state-of-the-art detectors. Specifically, Deformable-DETR [66] enhances feature learning; DAB-DETR [34], DINO [62], and DDQ-DETR [64] improve query design; CO-DINO [69] introduces more comprehensive supervision; Relation-DETR [23] exploits attentionbased query-feature interaction; while Mamba-YOLO [59], DECO [8], and CellMamba [33] prioritize computational eficiency by leveraging the strengths of Mamba and convolution. As reported in Table 1, our method achieves top performance across all three benchmarks. On CoNSeP, it surpasses DDQ-DETR [64] by 2.0% in ��� and 2.2% in ��, the most notable improvement among the three benchmarks. CoNSeP is also the most challenging, with dense nuclei and high inter-class similarity hindering both instance separation and type discrimination, explicitly modeling inter-cell relations showing a clear advantage over appearance-driven detection here. On CytoDArk0, the margin over DDQ-DETR [64] is 2.1% in ��� and 1.4% in ��. On OCELOT, consistent superiority is maintained across all metrics. All improvements are statistically significant $( p < 0 . 0 5 )$ . These results demonstrate that modeling inter-cell relationships via our instance-aware graph network provides critical relational cues absent in general-purpose detectors, validating its eficacy for pathological cell instance detection.

![](images/fff68c1e6a5f415971a15e3b06f4a3a7ad6c0ba1ba8e48a88cff409c6fb5cc3e.jpg)  
Figure 4: Visualization of cell detection results.

We also visualize detection results from DINO, CO-DINO, DDQ-DETR, and our model in Figure 4 across diferent staining protocols, where (A) and (B) are from Nissl staining, and (C)–(E) are from H&E staining. The results confirm that our method detects more cells while reducing false positives, particularly for densely packed and adherent cells.

## 4.4.2. Comparison with Classification Methods

Since most existing nucleus classification methods rely on segmentation masks or point annotations, we further compare our detection-based approach against representative end-to-end mask-based and point-based cell instance classification models. As shown in Table 2, we select a set of representative models for comparison, including DIST [41], Micro-Net [47], Hover-Net [16], Triple-UNet [65], MCSpatNet [1], TSFD-Net [27], Mask2Former [9], PGT [24], ACFormer [25], SMILE [42], CellViT [26], Cell-DETR [45] and PathContext [51]. Our method achieves state-of-the-art performance across all metrics, outperforming mask-based methods with richer pixel-level supervision and point-based methods with inherently lower task complexity. These results validate the efectiveness of our detection-based framework for accurate cell localization and classification, particularly under severe class imbalance in pathological images. Notably, $F _ { c } ^ { ~ m }$ varies most substantially across methods, with several failing to identify this class entirely, as its scarcity and atypical morphology aford few reliable appearance cues. Our method attains the highest $F _ { c } ^ { ~ m }$ , suggesting that contextual dependencies among neighboring cells yield discriminative evidence beyond individual appearance.

Table 3  
Performance comparison of nuclei classification methods and our approach on the CoNSeP dataset, reporting detection, mean and per-class F1 scores. <sup>†</sup> Mask-based methods. <sup>‡</sup> Point-based methods. <sup>⋆</sup> Bounding Box-based methods.
<table><tr><td>Model</td><td> $F _ { d }$ </td><td> $F _ { a v g }$ </td><td></td><td> ${ F _ { c } } ^ { m }$ </td><td> $F _ { c } ^ { ~ i }$ </td><td> $F _ { c } ^ { ~ e }$   $F _ { c } ^ { ~ s }$ </td><td></td><td>Model</td><td> $F _ { d }$ </td><td> $F _ { a v g }$ </td><td> $F _ { c } ^ { ~ m }$ </td><td> $F _ { c } ^ { ~ i }$ </td><td> $F _ { c } ^ { ~ e }$ </td><td> $F _ { c } ^ { ~ s }$ </td></tr><tr><td>DIST† (TMI&#x27;18)</td><td></td><td>0.71</td><td>0.42</td><td>0.00</td><td>0.53</td><td>0.62</td><td>0.51</td><td>PGT(MICCAI&#x27;23)</td><td>0.74</td><td></td><td></td><td>0.62</td><td>0.64</td><td></td></tr><tr><td>Micro-Net{†</td><td>(MIA&#x27;19)</td><td>0.74</td><td>0.47</td><td>0.12</td><td>0.59</td><td>0.62</td><td>0.53</td><td>ACFormer (ICCV&#x27;23)</td><td>0.74</td><td></td><td></td><td>0.64</td><td>0.64</td><td></td></tr><tr><td>Hover-Net{</td><td>(MIA&#x27;19)</td><td>0.75</td><td>0.57</td><td>0.43</td><td>0.63</td><td>0.64</td><td>0.57</td><td>SMILE† (MIA&#x27;23)</td><td>0.76</td><td>0.56</td><td>0.38</td><td>0.62</td><td>0.67</td><td>0.58</td></tr><tr><td>Triple-UNet†</td><td>(MIA&#x27;20)</td><td>0.66</td><td>0.38</td><td>0.10</td><td>0.57</td><td>0.42</td><td>0.44</td><td>CellViT† (MIA&#x27;24)</td><td>0.76</td><td>0.58</td><td>0.46</td><td>0.64</td><td>0.65</td><td>0.57</td></tr><tr><td>MCSpatNet</td><td>(iCCV&#x27;21)</td><td>0.73</td><td>0.51</td><td>0.40</td><td>0.54</td><td>0.58</td><td>0.54</td><td>Cell-DETR†(MIDL&#x27;24)</td><td>0.74</td><td>0.49</td><td>0.21</td><td>0.63</td><td>0.61</td><td>0.51</td></tr><tr><td>TSFD-Net{†</td><td>(NN&#x27;22)</td><td>0.68</td><td>0.44</td><td>0.12</td><td>0.57</td><td>0.56</td><td>0.51</td><td>PathContext‡(AAAI&#x27;26)</td><td>0.76</td><td>0.59</td><td>0.49</td><td>0.65</td><td>0.65</td><td>0.57</td></tr><tr><td>Mask2Former†</td><td>(CVPR&#x27;22)</td><td>0.66</td><td>0.41</td><td>0.33</td><td>0.46</td><td>0.46</td><td>0.41</td><td>Ours*</td><td>0.77</td><td>0.61</td><td>0.52</td><td>0.66</td><td>0.67</td><td>0.59</td></tr></table>

Ablation study on CoNSeP dataset.
<table><tr><td>Model Variant</td><td>AR</td><td>mAP |</td><td> ${ \mathsf { m A P } } _ { 5 0 }$ </td><td> ${ \mathsf { m A P } } _ { 7 5 }$ </td></tr><tr><td colspan="5">Ablation of Dynamic Graph Construction</td></tr><tr><td>w/o Feat. Sim.</td><td>56.6</td><td>30.6</td><td>54.9</td><td>31.8</td></tr><tr><td>w/o Pos. Corr.</td><td>57.3</td><td>31.4</td><td>55.5</td><td>32.9</td></tr><tr><td>Ours</td><td>58.7</td><td>32.0</td><td>57.0</td><td>33.5</td></tr><tr><td colspan="5">Ablation of Instance-aware Graph Network</td></tr><tr><td>Baseline (Query Feature Learning Network)</td><td>56.5</td><td>30.0</td><td>54.4</td><td>31.4</td></tr><tr><td>+ IGL</td><td>57.4</td><td>30.8</td><td>56.1</td><td>32.3</td></tr><tr><td>+ TSS (Ours)</td><td>58.7 56.3</td><td>32.0 30.2</td><td>57.0 54.6</td><td>33.5</td></tr><tr><td>GCN (ICLR&#x27;17) GAT (ICLR&#x27;19)</td><td>56.7</td><td>30.5</td><td>55.0</td><td>31.7</td></tr><tr><td>TokenGT (NeurIPS&#x27;22)</td><td>56.9</td><td>31.1</td><td>55.7</td><td>32.0</td></tr><tr><td>DMbaGCN (AAAI&#x27;26)</td><td></td><td>30.9</td><td></td><td>32.6</td></tr><tr><td></td><td>57.2</td><td></td><td>55.5</td><td>32.1</td></tr></table>

## 4.5. Ablation Study

We conduct ablation studies on the CoNSeP dataset to validate the design choices of our framework, as reported in Table 3. We ablate two key components: the Dynamic Graph Construction and the Instance-aware Graph Network (IGN). First, we evaluate the importance of dynamic graph construction by ablating either feature similarity or positional similarity, using only one cue for graph building while keeping all other components fixed. Removing positional similarity reduces ��� from 32.0% to 31.4% (-0.6%) and �� from 58.7% to 57.3% (-1.4%). Removing feature similarity causes a larger drop: ��� falls to 30.6% (-1.4%) and �� to 56.6% (-2.1%), confirming that both cues are essential, where feature similarity contributes more substantially to overall performance. Second, we validate the efectiveness of each component within IGN. Starting from the baseline (���: 30.0%, ��: 56.5%), integrating the Instance-aware Graph Learning (IGL) module improves ��� by 0.8%, and further incorporating the Topologystructured State Space (TSS) yields an additional 1.2% gain. To further verify the overall efectiveness of IGN, we replace it with alternative graph learners: the convolution-based GCN [31], the Transformer-based GAT [58] and TokenGT [30], and the Mamba-based DMbaGCN [21]. Our IGN outperforms all variants by clear margins, demonstrating its superiority.

## 4.6. Eficiency analysis

We compare the computational complexity of our method with representative models on the CoNSeP dataset, as summarized in Table 4. Our model (49.57M parameters, 276 GFLOPs) is substantially lighter than instance segmentation methods and comparable to point-based models, yet accomplishes a more complex task by providing

Computational complexity comparison. <sup>†</sup> Mask-based methods. <sup>‡</sup> Point-based methods. <sup>⋆</sup> Bounding Box-based methods.
<table><tr><td>Method</td><td>Para. (M)</td><td>FLOP (G)</td></tr><tr><td>Hover-Net† [16]</td><td>34.76</td><td>1992</td></tr><tr><td>Mask2Former† [9]</td><td>298.21</td><td>896</td></tr><tr><td>CellViT† [26]</td><td>142.85</td><td>3567</td></tr><tr><td>DINO* [62]</td><td>47.55</td><td>279</td></tr><tr><td>CO-DINO* [69]</td><td>65.49</td><td></td></tr><tr><td>DDQ-DETR* [64]</td><td>48.32</td><td>270</td></tr><tr><td>PathContext [51]</td><td>48.08</td><td>186</td></tr><tr><td>Ours*</td><td>49.57</td><td>276</td></tr></table>

![](images/e56fd2814e3a4991e0c0c16d54b38cde09b2cecc3e0e61a4b853a093cfa7f72a.jpg)  
(a) Diferent �-values.

![](images/154667dec61b68c711add5f0bbc2d39a6989a09960bb15f04c650cb495e78c0f.jpg)  
(b) Diferent query numbers.  
Figure 5: Hyper-parameter studies on CoNSeP.

instance-level representations approximating segmentation quality via bounding box detection. The marginal overhead over DDQ-DETR (+1.25M parameters, + 6 GFLOPs) confirms that our graph-based refinement module introduces minimal additional cost, achieving a favorable trade-of between eficiency and detection performance.

## 4.7. Investigation of Hyper-parameters

## 4.7.1. Investigation of �-values

In constructing the query distance function, we incorporate both spatial distance and feature distance, balanced by a weighting factor �. Therefore, we perform a hyperparameter study on � to determine its optimal setting. As shown in Figure 5a, the optimal performance is achieved at � = 0.5, indicating that spatial distance and feature distance contribute equally and complementarily to the query distance function. Accordingly, $\alpha = 0 . 5$ is adopted in our model.

## 4.7.2. Investigation of Query Numbers

The number of queries plays a critical role in query-based detectors, as it directly determines the upper bound on the number of detectable instances. Therefore, we conduct a dedicated hyperparameter study on query numbers. As shown in Figure 5b, increasing � consistently improves performance. Raising � from 100 to 300 boosts �� by 2.3% and �� $P _ { 5 0 }$ by 1.9%; further increasing to 900 yields additional gains of 3.8% and 2.3%, respectively, indicating better candidate coverage and feature querying. When the number of queries is further increased, while AR improves, precision metrics decrease. We thus adopt 900 as the number of queries for our model to achieve the optimal balance between recall and precision. Larger values were not tested due to GPU memory constraints.

Performance comparison of diferent K values on detection tasks across three datasets.
<table><tr><td>Dataset</td><td>K values</td><td>AR (%)</td><td>mAP (%)</td><td> ${ \bf m } { \bf A } { \bf P } _ { 5 0 }$  (%)</td><td> $\boldsymbol { \mathsf { m } } \mathbf { A } \mathsf { P } _ { 7 5 } \ \left( \% \right)$ </td></tr><tr><td rowspan="4">CoNSeP</td><td>5</td><td>54.7</td><td>29.5</td><td>54.5</td><td>30.4</td></tr><tr><td>7</td><td>56.5</td><td>31.3</td><td>56.7</td><td>32.3</td></tr><tr><td>8</td><td>58.7</td><td>32.0</td><td>57.0</td><td>33.5</td></tr><tr><td>9</td><td>55.8</td><td>30.9</td><td>55.8</td><td>32.0</td></tr><tr><td rowspan="4">CytoDArk0</td><td>3</td><td>71.7</td><td>61.7</td><td>88.5</td><td>71.8</td></tr><tr><td>4</td><td>71.3</td><td>61.1</td><td>88.1</td><td>70.8</td></tr><tr><td>5</td><td>70.4</td><td>60.8</td><td>87.7</td><td>70.7</td></tr><tr><td>8</td><td>68.5</td><td>57.6</td><td>86.7</td><td>67.9</td></tr><tr><td rowspan="4">OCELOT</td><td>3</td><td>56.5</td><td>34.8</td><td>72.1</td><td>29.1</td></tr><tr><td>4</td><td>56.8</td><td>34.9</td><td>72.3</td><td>29.3</td></tr><tr><td>5</td><td>56.4</td><td>34.1</td><td>71.8</td><td>28.3</td></tr><tr><td>8</td><td>56.6</td><td>33.5</td><td>70.9</td><td>27.1</td></tr></table>

## 4.7.3. Investigation of K-values

In handling complex graphs, the graph construction step is of critical importance: the graph topology directly governs the neighborhood aggregation, which in turn conditions the subsequent global modeling. Herein, we utilize KNN to establish node connectivity, and the setting of K value is particularly vital to graph construction. Therefore, we conduct additional experiments on this hyperparameter K.

Based on empirical observations, the selection of � can be broadly categorized into two regimes: sparse graph connectivity $( K \in [ 3 , 5 ] )$ and dense graph connectivity $( K \in [ 7 , 9 ] )$ . We first conduct pilot experiments with $K = 5$ and $K = 8$ as representative values of each regime to determine which connectivity pattern is more suitable for each dataset. Based on these results, we further evaluate adjacent � values within the identified regime to pinpoint the optimal setting.

As summarized in Table 5, on CoNSeP where nuclei are densely packed and highly overlapping, the best performance is achieved at $K = 8 .$ , outperforming sparser graphs (e.g., $K = 5 )$ by up to 2.5% in $m A P$ . In contrast, CytoDArk0 contains well-separated cells, and its peak performance occurs at $K = 3$ , while OCELOT achieves the best results at $K = 4$

## 5. Discussion

The central insight of this work is that cellular identity in histopathology is inherently relational. Intercellular interactions and signaling constitute the biophysical basis of structural integration and functional coordination in multicellular systems, such that the state of a cell is shaped not merely by its own morphology but largely by its interactions within the surrounding microenvironment. Existing detectors, however, model cells largely in isolation and prioritize local appearance, an approach that falters in dense, overlapping, or morphologically heterogeneous regions where visual cues alone cannot resolve instance identity. Certain pathology-specific methods instead introduce relational context through graph structures, yet predominantly follow a two-stage paradigm that decouples graph construction from detection: the graph is built upon the outputs of an independently trained detector or segmentor, serving only as post-hoc refinement, propagating upstream errors and precluding optimization under a unified objective. Neither line of work treats relational structure as something to be learned jointly with detection, which motivates a framework in which relational reasoning is not appended to detection but intrinsic to it.

To realize an end-to-end graph-based detection framework, we reconceive what constitutes a graph node. Rather than deriving nodes from the outputs of an upstream detector or from pixels on a feature map, we take learnable queries as candidate cell instances and construct the graph directly over them. Through decoding, each query aggregates multiscale visual evidence from the tissue microenvironment, thereby transforming patch-level contextual representations into instance-level semantic descriptions upon which graph construction and relational reasoning operate. A single representation thus spans the entire path from patch-level feature extraction, through instantiation, to inter-instance relational modeling, allowing the graph to focus on instances themselves and all three stages to be optimized under a unified objective. Connectivity is determined jointly by spatial proximity and feature similarity, enabling the graph to encode organizational regularities that neither cue expresses alone. The resulting structure is genuinely dynamic: as query embeddings evolve through learning, the induced adjacency shifts accordingly; conversely, gradients propagated through graph learning continually refine the query features themselves, so that representation and topology shape each other over training and erroneous connections arising from unreliable reference points or immature features are not permanently entrenched. The topology is thereby shaped dynamically by the data rather than fixed a priori by spatial heuristics.

We next turn to how such a graph ought to be processed. As the number of queries far exceeds that of true cells, the graph abounds in redundant nodes, over which dense attention merely disperses irrelevant signals at quadratic cost. The state transition of a selective state space model, by contrast, is inherently endowed with content-adaptive selectivity, suppressing irrelevant signals while preserving global structural patterns at a complexity that scales linearly with the number of nodes. Departing from the denoise-then-model paradigm, we let graph learning itself yield a structure that the state space model can carry: once selectively reorganized, node states unfold along their respective directions, whereby aggregation becomes a superposition across distinct directions and semantically inconsistent neighbors, being near-orthogonal, remain mutually non-interfering. Conventional denoising schemes must assess neighbor relevance through attention weights or edge pruning, whereas filtering here follows directly from directional relations: irrelevant neighbors require no separate identification or removal, their contribution having attenuated substantially over end-to-end training. Building upon this, we recast the state transition of the SSM, taking the resulting topological representation as the latent state while the visual features of the queries are re-injected to drive its evolution, so that relational and appearance evidence accumulate within a single state. CGSE further abstracts all queries into an instance-level context from which the transition parameters are generated, enabling the dynamics to adapt to the feature distribution of each individual instance rather than conforming to a uniformly imposed prior.

Within the overall architecture, neither NMS nor KNN, despite being non-diferentiable, impedes end-to-end training. Both act as parameter-free structural selectors conditioned on the current query states: NMS deterministically retains high-quality, non-redundant candidates, while KNN establishes connectivity from the prevailing query features. Neither introduces learnable parameters nor interrupts any gradient path, and the features of the retained queries participate fully in backpropagation. The dynamic character of the graph arises not from any adjustment within KNN itself, but from the continual learning of the query embeddings upon which it operates, thereby establishing an implicit feedback loop between feature learning and graph topology that constitutes the core mechanism of end-to-end joint optimization throughout the framework.

## 6. Conclusion

In this paper, we have presented a novel end-to-end framework for cell detection and classification, designed to enhance the instance discrimination capability of query-based methods through structured reasoning. By treating learnable queries as graph nodes, our approach dynamically constructs a context-aware topology using both spatial and feature cues, enabling adaptive modeling of complex and varying cell distributions. To address noise and longrange dependency challenges in large query sets, we introduce an instance-aware graph network (IGN) that formulates a Topology-structured State Space (TSS) model with graph topology as the latent state, driving global modeling. Extensive experiments demonstrate the efectiveness of our method, achieving state-of-the-art performance across multiple benchmarks. This work highlights the potential of combining query-based detection with structured graph learning, ofering a promising direction for precise and robust cell instance analysis in digital pathology.

## References

[1] Abousamra, S., Belinsky, D., Arnam, J.V., Allard, F., Yee, E., Gupta, R., Kurc, T., Samaras, D., Saltz, J., Chen, C., 2021. Multi-class cell detection using spatial context representation, in: IEEE/CVF International Conference on Computer Vision, pp. 4005–4014.

[2] Aubreville, M., Stathonikos, N., Bertram, C.A., et al., 2023. Mitosis domain generalization in histopathology images—the midog challenge. Medical Image Analysis 84, 102699.

[3] Ayyad, S.M., Shehata, M., Shalaby, A., El-Ghar, M.A., Ghazal, M., El-Melegy, M., Abdel-Hamid, N.B., Labib, L.M., Ali, H.A., El-Baz, A., 2021. Role of ai and histopathological images in detecting prostate cancer: A survey. Sensors 21, 2586.

[4] Baxi, V., Edwards, R., Montalto, M., Saha, S., 2022. Digital pathology and artificial intelligence in translational medicine and clinical practice. Modern Pathology 35, 23–32.

[5] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S., 2020. End-to-end object detection with transformers, in: European Conference on Computer Vision, pp. 213–229.

[6] Chen, M., Wei, Z., Huang, Z., Ding, B., Li, Y., 2020. Simple and deep graph convolutional networks, in: International conference on machine learning, pp. 1725–1735.

[7] Chen, S., Ding, C., Liu, M., Cheng, J., Tao, D., 2023. Cpp-net: Context-aware polygon proposal network for nucleus segmentation. IEEE Transactions on Image Processing 32, 980–994.

[8] Chen, X., Li, S., Yang, Y., Wang, Y., 2025. Deco: Unleashing the potential of convnets for query-based detection and segmentation, in: International Conference on Learning Representations.

[9] Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R., 2021. Masked-attention mask transformer for universal image segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

[10] Chien, E., Peng, J., Li, P., Milenkovic, O., 2021. Adaptive universal generalized pagerank graph neural network, in: International Conference on Learning Representations.

[11] Da, Q., Huang, X., Li, Z., et al., 2022. Digestpath: A benchmark dataset with challenge review for the pathological detection and segmentation of digestive-system. Medical Image Analysis 80, 102485.

[12] Duan, M., Yang, Z., Ma, Y., Wang, M., Song, Z., 2025. Knowledge-guided multi-scale graph mamba for whole slide image classification, in: International Conference on Medical Image Computing and Computer-Assisted Intervention.

[13] Frei, A.L., Garcia-Baroja, J., Rau, T., Neppl, C., Lugli, A., Solass, W., Wartenberg, M., Fischer, A., Zlobec, I., 2026. Grep: Graph-based epithelial cell classification refinement in histopathology h&e images. Pattern Recognition 171, 112197.

[14] Gao, Z., Lu, Z., Wang, J., Ying, S., Shi, J., 2022. A convolutional neural network and graph convolutional network based framework for classification of breast histopathological images. IEEE Journal of Biomedical and Health Informatics 26, 3163–3173.

[15] Gasteiger, J., Bojchevski, A., Günnemann, S., 2019. Predict then propagate: Graph neural networks meet personalized pagerank, in: International Conference on Learning Representations.

[16] Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T., Rajpoot, N., 2019. Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. Medical Image Analysis 58, 101563.

[17] Gu, A., Dao, T., 2024. Mamba: Linear-time sequence modeling with selective state spaces, in: Conference On Language Modeling.

[18] Gu, A., Dao, T., Ermon, S., Rudra, A., Ré, C., 2020. Hippo: Recurrent memory with optimal polynomial projections, in: Advances in Neural Information Processing Systems, pp. 1474–1487.

[19] Gu, H., Haeri, M., Ni, S., Williams, C.K., Zarrin-Khameh, N., Magaki, S., Chen, X.A., 2022. Detecting mitoses with a convolutional neural network for midog 2022 challenge, in: MICCAI Challenge on Mitosis Domain Generalization.

[20] Hassan, T., Javed, S., Mahmood, A., Qaiser, T., Werghi, N., Rajpoot, N., 2022. Nucleus classification in histology images using message passing network. Medical Image Analysis 79, 102480.

[21] He, X., Wang, Y., Dai, Y., Wang, X., 2026. Dual mamba for node-specific representation learning: Tackling over-smoothing with selective state space modeling, in: Proceedings of the AAAI Conference on Artificial Intelligence.

[22] He, X., Wang, Y., Fan, W., Shen, X., Juan, X., Miao, R., Wang, X., 2025. Mamba-based graph convolutional networks: Tackling over-smoothing with selective state space, in: arXiv:2501.15461.

[23] Hou, X., Liu, M., Zhang, S., Wei, P., Chen, B., Lan, X., 2024. Relation detr: Exploring explicit position relation prior for object detection, in: European conference on computer vision.

[24] Huang, J., Li, H., Sun, W., Wan, X., Li, G., 2023a. Prompt-based grouping transformer for nucleus detection and classification, in: Internationa Conference on Medical Image Computing and Computer-Assisted Intervention.

[25] Huang, J., Li, H., Wan, X., Li, G., 2023b. Afine-consistent transformer for multi-class cell nuclei detection, in: IEEE/CVF International Conference on Computer Vision, pp. 21384–21393.

[26] Hörst, F., Rempe, M., Heine, L., Seibold, C., Keyl, J., Baldini, G., Ugurel, S., Siveke, J., Grünwald, B., Egger, J., Kleesiek, J., 2024. Cellvit: Vision transformers for precise cell segmentation and classification. Medical Image Analysis 94, 103143.

[27] Ilyas, T., Mannan, Z.I., Khan, A., Azam, S., Kim, H., Boer, F.D., 2022. Tsfd-net: Tissue specific feature distillation network for nuclei segmentation and classification. Neural Networks 151, 1–15.

[28] Jaume, G., Pati, P., Bozorgtabar, B., Foncubierta, A., Anniciello, A.M., Feroce, F., Rau, T., Thiran, J.P., Gabrani, M., Goksel, O., 2021. Quantifying explainers of graph neural networks in computational pathology, in: Proceedings of the IEEE/CVF conference on compute vision and pattern recognition, pp. 8106–8116.

[29] Jiang, H., Zhang, R., Zhou, Y., Wang, Y., Chen, H., 2023. Donet: Deep de-overlapping network for cytology instance segmentation, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15641–15650.

[30] Kim, J., Nguyen, D., Min, S., Cho, S., Lee, M., Lee, H., Hong, S., 2022. Pure transformers are powerful graph learners, in: Advances in Neural Information Processing Systems, pp. 14582–14595.

[31] Kipf, T.N., Welling, M., 2017. Semi-supervised classification with graph convolutional networks, in: International conference on learning representations.

[32] Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollar, P., 2017. Focal loss for dense object detection, in: Proceedings of the IEEE international conference on computer vision.

[33] Liu, R., Tian, Y., Wang, J., Liu, H., Hou, X., Liu, J., 2025. Cellmamba: Adaptive mamba for accurate and eficient cell detection, in: arXiv:2512.21803.

[34] Liu, S., Li, F., Zhang, H., Yang, X., Qi, X., Su, H., Zhu, J., Zhang, L., 2022. Dab-detr: Dynamic anchor boxes are better queries for detr, in: International Conference on Learning Representations.

[35] Lou, W., Li, G., Wan, X., Li, H., 2024a. Cell graph transformer for nuclei classification, in: Proceedings of the AAAI Conference on Artificia Intelligence, pp. 3873–3881.

[36] Lou, W., Li, H., Li, G., Han, X., Wan, X., 2022. Which pixel to annotate: A label-eficient nuclei segmentation framework. IEEE Transactions on Medical Imaging 42, 947–958.

[37] Lou, W., Wan, X., Li, G., Lou, X., Li, C., Gao, F., Li, H., 2024b. Structure embedded nucleus classification for histopathology images. IEEE Transactions on Medical Imaging 43, 3149–3160.

[38] Ma, J., Xie, R., Ayyadhury, S., et al., 2024. The multimodality cell segmentation challenge: toward universal solutions. Nature Methods 21, 1103–1113.

[39] Meng, Y., Zhang, H., Zhao, Y., Yang, X., Qiao, Y., MacCormick, I.J.C., Huang, X., Zheng, Y., 2021. Graph-based region and boundary aggregation for biomedical image segmentation. IEEE Transactions on Medical Imaging 41, 690–701.

[40] Miao Cui, D.Y.Z., 2021. Artificial intelligence and computational pathology. Laboratory Investigation 101, 412–422.

[41] Naylor, P., Laé, M., Reyal, F., Walter, T., 2018. Segmentation of nuclei in histopathology images by deep regression of the distance map. IEEE Transactions on Medical Imaging 38, 448–459.

[42] Pan, X., Cheng, J., Hou, F., Lan, R., Lu, C., Li, L., Feng, Z., Wang, H., Liang, C., Liu, Z., Chen, X., Han, C., Liu, Z., 2023. Smile: Cost-sensitive multi-task learning for nuclear segmentation and classification with imbalanced annotations. Medical Image Analysis 88, 102867.

[43] Pan, X., Yang, D., Li, L., Liu, Z., Yang, H., Cao, Z., He, Y., Ma, Z., Chen, Y., 2018. Cell detection in pathology and microscopy images with multi-scale fully convolutional neural networks. World Wide Web 21, 1721–1743.

[44] Pang, M., Roy, T.K., Wu, X., Tan, K., 2025. Cellotype: a unified model for segmentation and classification of tissue images. Nature Methods 22, 348–357.

[45] Pina, O., Dorca, E., Vilaplana, V., 2024. Cell-detr: Eficient cell detection and classification in wsis with transformers, in: Medical Imaging with Deep Learning.

[46] Prangemeier, T., Reich, C., Koeppl, H., 2020. Attention-based transformers for instance segmentation of cells in microstructures, in: IEEE International Conference on Bioinformatics and Biomedicine, pp. 700–707.

[47] Raza, S.E.A., Cheung, L., Shaban, M., Graham, S., Epstein, D., Pelengaris, S., Khan, M., Rajpoot, N.M., 2019. Micro-net: A unified model for segmentation of various objects in microscopy images. Medical Image Analysis 52, 160–173.

[48] Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S., 2019. Generalized intersection over union: A metric and a loss for bounding box regression, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

[49] Ryu, J., Puche, A.V., Shin, J., Park, S., Brattoli, B., Lee, J., Jung, W., Cho, S.I., Paeng, K., Ock, C.Y., Yoo, D., Sérgio, 2023. Ocelot: Overlapped cell on tissue dataset for histopathology, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23902–23912.

[50] Schmidt, U., Weigert, M., Broaddus, C., Myers, G., 2018. Cell detection with star-convex polygons, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 265–273.

[51] Shui, Z., Guo, R., Li, H., Sun, Y., Zhang, Y., Zhu, C., Cai, J., Chen, P., Su, Y., Yang, L., 2026. Towards efective and eficient context-aware nucleus detection in histopathology whole slide images, in: Proceedings of the AAAI Conference on Artificial Intelligence.

[52] Shui, Z., Zhang, S., Zhu, C., Wang, B., Chen, P., Zheng, S., Yang, L., 2022. End-to-end cell recognition by point annotation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 109–118.

[53] Shui, Z., Zheng, S., Zhu, C., Zhang, S., Yu, X., Li, H., Li, J., Chen, P., Yang, L., 2024. Dpa-p2pnet: Deformable proposal-aware p2pnet for accurate point-based cell detection, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 4864–4872.

[54] Stringer, C., Wang, T., Michaelos, M., Pachitariu, M., 2021. Cellpose: a generalist algorithm for cellular segmentation. Nature Methods 18, 100–106.

[55] Tajbakhsh, N., Jeyaseelan, L., Li, Q., Chiang, J.N., Wu, Z., Ding, X., 2020. Embracing imperfect datasets: A review of deep learning solutions for medical image segmentation. Medical Image Analysis 63, 101693.

[56] Vadori, V., Graïc, J.M., Perufo, A., Vadori, G., Finos, L., Grisan, E., 2025. Cisca and cytodark0: a cell instance segmentation and classification method for histo(patho)logical image analyses and a new, open, nissl-stained dataset for brain cytoarchitecture studies. Computers in Biology and Medicine 197, 111018.

[57] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Łukasz Kaiser, Polosukhin, I., 2017. Attention is all you need, in: Advances in Neural Information Processing Systems.

[58] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y., 2018. Graph attention networks, in: International conference on learning representations.

[59] Wang, Z., Li, C., Xu, H., Zhu, X., Li, H., 2025. Mamba yolo: A simple baseline for object detection with state space model, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 8205–8213.

[60] Yang, Z., Kuang, C., Fu, D., 2026. Denuc: Decoupling nuclei detection and classification in histopathology, in: arXiv:2603.04240.

[61] Yoon, S., Chandra, A., Vahedi, G., 2022. Stripenn detects architectural stripes from chromatin conformation data using computer vision. Nature Communications 13, 1602.

[62] Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y., 2023a. Dino: Detr with improved denoising anchor boxes for end-to-end object detection, in: The eleventh international conference on learning representations.

[63] Zhang, L., Zhang, Y., Cai, L., Guan, X., Zhang, K., Zhang, Y., 2025. Relation-aware graph attention network for nuclei classification, in: 2025 IEEE International Conference on Multimedia and Expo (ICME).

[64] Zhang, S., Wang, X., Wang, J., Pang, J., Lyu, C., Zhang, W., Luo, P., Chen, K., 2023b. Dense distinct query for end-to-end object detection, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7329–7338.

[65] Zhao, B., Chen, X., Li, Z., Yu, Z., Yao, S., Yan, L., Wang, Y., Liu, Z., Liang, C., Han, C., 2020. Triple u-net: Hematoxylin-aware nuclei segmentation with progressive dense feature aggregation. Medical Image Analysis 65, 101786.

[66] Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J., 2021. Deformable detr: Deformable transformers for end-to-end object detection, in: International Conference on Learning Representations.

[67] Zhu, Y., Yang, Q., Xu, L., 2024. Active learning enabled low-cost cell image segmentation using bounding box annotation, in: arXiv:2405.01701.

[68] Zink, D., Fischer, A.H., Nickerson, J.A., 2004. Nuclear structure in cancer cells. Nature Reviews Cancer 4, 677–687.

[69] Zong, Z., Song, G., Liu, Y., 2023. Detrs with collaborative hybrid assignments training, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6748–6758.