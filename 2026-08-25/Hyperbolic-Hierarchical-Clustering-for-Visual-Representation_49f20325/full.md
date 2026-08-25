# Hyperbolic Hierarchical Clustering for Visual Representation Learning

Jianan Wei<sup>1</sup>, Guikun Chen<sup>1</sup>, Zhiyuan Weng<sup>1</sup>, Chunchao Guo<sup>2</sup>, Yujia Wang<sup>3†</sup>, and Wenguan Wang<sup>1</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>Tencent Hunyuan <sup>3</sup>Zhejiang Sci-Tech University https://github.com/weijianan1/HCFormer

Abstract. We investigate the token mixer in vision backbones by revisit ing clustering, one of the most classic approaches in machine learning. An efective token mixer is a fundamental component of modern vision back bones like vision Transformers, facilitating information exchange between image patches. Mainstream token mixers, which rely on convolution, attention, MLP, or their hybrids, primarily focus on navigating the trade-of between accuracy and computational cost. However, a significant drawback of these methods is their black-box nature; their encoding process is opaque and lacks interpretability. Diverging from these opaque designs, we introduce ClusterMixer, a transparent token mixer that is grounded in a clustering paradigm and interpretable by design. ClusterMixer explicitly formulates the token mixing process through a hierarchical clustering mechanism. To model the natural, tree-like relationships inherent in visual data, the clustering is performed in hyperbolic space, which is well-suited for embedding hierarchies with low distortion. Building on this innovation, we present HCFormer, a new backbone architecture that integrates ClusterMixer with a series of meticulously designed clustering strategies to ensure robust performance across tasks. Extensive experiments demonstrate that HCFormer consistently outperforms its counterparts across diverse tasks, including image classification, object detection, instance segmentation, and semantic segmentation. Considering its transparency and eficacy, we hope HCFormer can facilitate a paradigm shift toward interpretable backbones.

Keywords: Representation learning · Hyperbolic geometry · Clustering

## 1 Introduction

Convolutional Networks (ConvNets) and Vision Transformers (ViTs) are the dominant paradigms in computer vision. ConvNets serve as the de facto standard since the pioneering success of AlexNet [31], owing to their strong inductive bias (i.e., locality and translation equivariance). Marking a paradigm shift, ViTs [16] introduce self-attention mechanisms with fewer inductive biases, enabling efective scaling and superior generalization over ConvNets. Building on this, MLP-Mixer [59] conceptualizes token mixing and proposes a convolution- and attention-free alternative, which replaces the self-attention layers with spatial MLPs. This work spurs research into alternative token mixers in the vision community. For instance, VAN [21] employs convolution as the token mixer, while Uniformer [32] integrates both convolution and self-attention. Despite their incremental performance gains, the reasoning process of these models remains opaque to humans. Recently, there has been growing research interest in clustering-based vision backbones [9, 46], which can provide enhanced interpretability. Albeit promising, it has not yet fully exploited the capabilities of clustering algorithms for visual representation learning, as evidenced by their suboptimal performance, indicating a notable gap that merits further investigation.

![](images/4d17f823bac02a3482b8ef346a90b4ce68a28b1d7693ace6ee9561f1580f07b0.jpg)  
Fig. 1: Top-1 accuracy of HCFormer and other SOTAs on ImageNet-1K [13].

In this work, we introduce an eficient token mixer based on clustering algorithms, dubbed ClusterMixer, which operates via: i) initializing data points and cluster centers based on token representations, ii) assigning data points to centers to form distinct clusters, and iii) performing token mixing based on the established clusters. Nevertheless, the direct application of ClusterMixer presents a dual challenge in terms of computational eficiency and model performance. First, it faces analogous challenges of computational complexity as self-attention and spatial MLPs, stemming from data point and cluster center counts. This eficiency limitation becomes more pronounced for vision tasks that require extensive token sets to accomplish dense prediction or high-resolution image representation. A straightforward approach is to reduce the number of cluster centers; however, it leads to an undesirable tradeof by degrading model performance. Second, the similarity estimation of clustering algorithms is primarily conducted in Euclidean geometry [46, 52]. However, the flat geometry of Euclidean space limits its ability in modeling hierarchical relation [34], leading to distortions in the semantic distance and suboptimal performance. In other words, even if two embeddings are close in Euclidean geometry, they may be distant in the semantic hierarchy.

To bridge these gaps, we propose HCFormer, a framework for visual representation learning that enhances ClusterMixer via two meticulously designed strategies: i) hierarchical clustering. Here input patches are partitioned into several non-overlapping windows, with token mixing restricted to local windows to reduce computational complexity in clustering operations. To capture global contextual information, we further apply a mixing operation over these windows, which introduces only minimal computational overhead. ii) hyperbolic hierarchical clustering. Unlike Euclidean geometry, which excels in flat, simple structures, hyperbolic geometry naturally models hierarchical relationships as a continuous analog of trees [29], thereby better capturing abstract and complex semantics.

To leverage these complementary properties, we extend the ClusterMixer to use Euclidean space for fine-grained patch-level clustering and hyperbolic space for abstract window-level clustering.

Despite its attractive properties (e.g., grasping abstract and complex semantic relationships), hyperbolic geometry is numerically unstable, whereas Euclidean geometry demonstrates superior performance and stability in simple scenarios. Besides, the operations defined in hyperbolic geometry incur greater computational overhead compared to their Euclidean counterparts. To address these trade-ofs, we extend the ClusterMixer to perform clustering across dual geometries: Euclidean similarity is computed at the finer-grained and computationally demanding patch-level, while hyperbolic similarity is estimated at the abstract and computationally eficient window-level.

HCFormer enjoys several desirable virtues: First, eficiency. Based on the proposed clustering strategies, HCFormer can reduce the quadratic computational complexity associated with increasing data points and cluster centers to a linear one. This reduction in complexity is especially beneficial for downstream tasks, e.g., semantic segmentation and object detection, which facilitates the use of a greater number of cluster centers for improved performance. Second, holistic hierarchy. HCFormer conducts ClusterMixer in both Euclidean and Hyperbolic geometries, enabling the model to handle similarity estimation across diverse scenarios, from simple (e.g., flat relation) to complex (e.g., tree-like structure). Third, flexibility. Owing to its non-parametric and clustering-based design, ClusterMixer enables HCFormer to be efectively applied to various downstream tasks (see §4).

For comprehensive evaluation, HCFormer is benchmarked on three datasets covering diverse application scenarios, including ImageNet-1K [13] for image classification, ADE20K [76] for semantic segmentation, COCO [38] for instance detection and segmentation. In §4.1, by training from scratch, HCFormer outperforms mainstream counterparts with similar parameter counts, e.g., exceeding ResNet-50 by 2.6% and Swin-Tiny by 1.1% in top-1 accuracy. Notably, HC-Former also improves over prior clustering-based backbones across model scales, with gains of 1.2–3.3% over CoC [46] and 0.3–2.4% over FEC [9]. In §4.2, as the backbone with Semantic FPN [30], HCFormer achieves performance gains of 0.7%\~2.8% mIoU for semantic segmentation. In §4.3, HCFormer integrated with Mask R-CNN [23] demonstrates competitive performance for instance detection and segmentation. These results are impressive for a clustering-based backbone like HCFormer, which also enjoys built-in interpretability.

## 2 Related Work

Clustering in Vision. The objective of clustering is to partition finite, unlabeled data into discrete subsets of inherent groupings or clusters using a similarity measure (e.g., Euclidean distance) [47, 68], operating as an essential tool in pattern recognition. Traditional clustering approaches are adopted in computer vision as an efective image preprocessing method [1, 35, 53], which groups pixels into perceptually meaningful atomic regions. Modern clustering-based methods are further developed for downstream vision tasks, such as semantic segmentation [15, 33, 36, 37, 70, 71], visual relationship understanding [25, 66] and trajectory prediction [58, 67]. Due to its flexibility and interpretability, clustering methods are increasingly utilized for modeling complex data types, e.g., point clouds [17, 45, 50, 69] and protein [2, 51], and employed to elucidate the decisions derived from the neural network [26, 64, 77]. Motivated by these compelling virtues, our work endeavors to explore the adaptation of clustering methods for basic visual representation learning.

Learning in Hyperbolic Geometry. Hyperbolic geometry has attracted significant attention owing to its efectiveness in modeling data with tree-like structures [6, 48]. These desirable properties render hyperbolic embeddings a superior alternative in diverse modalities, e.g., graphs [4, 7], images [3, 18] and videos [28, 42]. Another important line of research involves leveraging hyperbolic geometry for representation learning. For instance, Hyperbolic Neural Networks [19] introduce a set of corresponding hyperbolic neural network layers and demonstrates superior performance over their Euclidean counterparts in various downstream tasks, such as text entailment and noisy prefix prediction. Subsequent works further expand this framework: Hyperbolic Neural Networks++ [56] proposes hyperbolic convolutional layers, while Hyperbolic attention networks [40] present a Graph Neural Network architecture that operates in hyperbolic space. In contrast to these works, this paper incorporates hyperbolic geometry into clustering algorithms for visual representation learning, aiming to enhance token mixing by better capturing hierarchical relationships between features.

Generic Vision Backbone. Revolutions in computer vision can be characterized as a paradigm shift in feature extraction. In early research [11, 44, 55], feature extraction typically relies on manually crafted features (e.g., geometric structure or color statistics) derived from the predefined rules. Beginning with AlexNet [31], Convolutional Networks (ConvNets) [24, 57] evolve this paradigm from hand-crafted feature engineering to data-driven feature learning, which ofers greater versatility and performance. ConvNets utilize sliding windows to partition the entire image into a set of rectangular patches, where convolutional kernels conduct pixel mixing via weighted summations in the local region. In contrast, Vision Transformers (ViTs) [16] provide an attention-based alternative, which reduces the image-specific inductive biases in ConvNets for feature extraction. ViTs split images into a collection of non-overlapping patches and conduct token mixing across them in a global range, facilitating model scalability. Beyond these two paradigms, Context Cluster (CoC) [46] proposes an innovative idea to group the points into clusters, where pixel features are aggregated and then dispatched within a cluster. This simple design is convolution- and attention-free, which only relies on a clustering algorithm to provide interpretability. FEC [9] extends this by introducing clustering-based feature pooling for downsampling, thereby achieving a fully clustering-based feature extraction.

Nevertheless, the potential of clustering methodologies for representation learning remains underexplored within the research community. Our work achieves strong performance on various vision tasks, and we hope it will contribute to a paradigm shift toward clustering-based vision backbones.

Hyperbolic Hierarchical Clustering for Visual Representation Learning  
![](images/4d6ac2ee276c4249ffce37f142d1b428cfbc5d73ed1e9c8f9c18697ee8ce7879.jpg)  
Fig. 2: (a) Overall pipeline of HCFormer. Following the MetaFormer paradigm [72], HCFormer adopts a hierarchical architecture with 4 stages. (b) Hierarchical clustering. ClusterMixer conducts patch-level clustering in Euclidean space and window-level clustering in hyperbolic space.

## 3 Method

## 3.1 Overall Architecture

An overview of the HCFormer is presented in Figure 2, which is built upon the MetaFormer paradigm [72]. The input image I is first split into non-overlapping patches, following the patch splitting module of ViT [16]. Given the fundamental role of spatial proximity in visual clustering [27], these patches are equipped with their coordinates and processed into embedding tokens. Then, a series of residual blocks incorporating token mixers are applied on these tokens. Instead of attention mechanism (e.g., Transformer [63]), spatial MLP (e.g., MLP-Mixer [59]), or average pooling (e.g., PoolFormer [72]), we implement interpretable clustering algorithms as the token mixer, dubbed ClusterMixer. To produce a hierarchical representation akin to ConvNets [24, 57], the number of tokens is reduced by concatenating the spatially adjacent tokens, which is achieved by a convolutional operation. This design enables seamless adaptation to downstream tasks such as semantic segmentation and object detection. The following section first delineates the implementation of ClusterMixer (§3.2) and then describes the two strategies developed to enhance it (§3.3 & §3.4).

## 3.2 ClusterMixer

Considering data points $\pmb { X } \in \mathbb { R } ^ { N \times C }$ , the ClusterMixer is performed as following: Center Estimation. The estimation of cluster centers from data points can be implemented through various methods, such as the K-means [53] or the Sinkhorn-Knopp algorithm [43, 64]. However, these methods are computationally intensive and thus not suitable for token mixing. In this paper, we initialize the data points $\pmb { X } \in \mathbb { R } ^ { N \times C }$ with feature embeddings and employ average pooling over neighboring patches to estimate cluster centers $\mathbf { \boldsymbol { C } } \in \mathbb { R } ^ { \mathbf { \tilde { M } } \times \mathbf { \check { C } } }$ . Inspired by [72], the cluster centers $C \in \mathbb { R } ^ { M \times C }$ are estimated via a pooling operator, defined as:

$$
C _ { i } = \frac { 1 } { K ^ { 2 } } \sum _ { p = 1 } ^ { K } \sum _ { q = 1 } ^ { K } F _ { [ : , r \cdot K + p , c \cdot K + q ] } , s . t . r = \lfloor i / M _ { w } \rfloor , c = i \bmod M _ { w } ,\tag{1}
$$

where K is the pooling size, $M _ { w }$ denotes the number of cluster centers in a row, $i . e . , M _ { w } = W / K$ . This approach incurs a computational complexity linear in the sequence length while involving no learnable parameters.

Cluster Assignment. We first partition all data points into distinct clusters by assigning them to diferent cluster centers $c _ { j } \in C$ based on the feature similarity. Notably, in contrast to hard clustering (i.e., each token is exclusively assigned to a single cluster), we employ a soft clustering approach, where data points can belong to multiple clusters with varying degrees of probability. This process resembles the Expectation-step (E-step) in Expectation-Maximization (EM) clustering [12], with the distinction that our cluster centers are precomputed via average pooling over neighboring patches, avoiding iterative updates. The i-th row $a _ { i } \in \mathbb { R } ^ { 1 \times M }$ of the assignment matrix $\overset { \cdot } { A } \in \mathbb { R } ^ { N \times \smile }$ for each token $x _ { i } \in X$ is computed as:

$$
\begin{array} { r } { a _ { i } = \mathrm { S o f t m a x } _ { c _ { j } \in C , b _ { i , j } \in { \bf B } } ( \alpha \cdot D ( x _ { i } , c _ { j } ) + b _ { i , j } ) , } \end{array}\tag{2}
$$

where D denotes the similarity metric (see §3.4), α is a learnable scalar for feature similarity, and $B \in \mathbb { R } ^ { N \times M }$ parameterizes the learnable relative positional bias. Notably, D is the only component evaluated in Euclidean or hyperbolic geometry; the cluster centers $c _ { j }$ and token features $x _ { i }$ used by center estimation and token mixing remain Euclidean feature vectors.

Token Mixing. We perform token mixing in two steps: i) feature aggregation from data points and ii) feature propagation back to data points. Specifically, it adaptively gathers contextual information from data points based on the assignment matrix to compute the aggregated feature $g _ { i } \in \mathbb { R } ^ { 1 \times C }$ :

$$
g _ { i } = \sum _ { j = 1 } ^ { M } A _ { i , j } \cdot c _ { j } ^ { \prime } , ~ s . t . ~ c _ { j } ^ { \prime } = ( c _ { j } + \sum _ { i = 1 } ^ { N } A _ { i , j } \cdot x _ { i } ) / ( 1 + \sum _ { i = 1 } ^ { N } A _ { i , j } ) .\tag{3}
$$

The aggregated feature $g _ { i }$ is then employed to update the corresponding data point via:

$$
x _ { i } ^ { \prime } = x _ { i } + \operatorname { F C } ( g _ { i } ) ,\tag{4}
$$

where FC denotes a fully-connected layer to maintain the feature dimension of token embeddings.

Our ClusterMixer ofers several advantages: i) the soft clustering (vs. CoC’s hard clustering [46]) enables our method to benefit from constructing more clusters across various tasks (see §4.4). ii) the relative positional bias efectively assists the clustering algorithm in establishing relational dependencies.

## 3.3 Hierarchical Clustering

To tackle the challenge of computational complexity, we perform clustering operations within non-overlapping local windows, where each input is partitioned into S windows, each containing K patches. However, such disconnected windows are recognized to limit the acquisition of global contextual information for representation learning. While this limitation can be mitigated via shifted windows in Swin Transformer [41], our clustering-based token mixer ofers an alternative approach by performing mixing at both the patch-level and the window-level. These two strategies difer solely in how data points and cluster centers are formulated in the ClusterMixer:

Patch-level clustering. Each patch serves as a data point, while cluster centers are estimated within each localized window, and the ClusterMixer is performed in parallel across all windows.

– Window-level clustering. Each window is modeled as a data point, represented by the mean feature embedding of its patches, while cluster centers are estimated globally over the entire input.

This formulation transforms the quadratic complexity of ClusterMixer into linear complexity. Empirically, this hierarchical clustering mechanism is better suited for the proposed ClusterMixer compared to the shifted windowing scheme (§4.4).

## 3.4 Hyperbolic Hierarchical Clustering

The choice of similarity metric in Equation 2 is critical for cluster assignment. Currently, similarity estimation is primarily conducted in Euclidean geometry [46, 52]: given data points $\boldsymbol { x } _ { \mathbb { E } } \in \mathbb { R } ^ { \bar { C } }$ and a cluster center $c _ { \mathbb { E } } \in \mathbb { R } ^ { C }$ , the feature distance is estimated by pair-wise cosine similarity:

$$
D _ { \mathbb { E } } ( x _ { \mathbb { E } } , c _ { \mathbb { E } } ) : = \sin ( x _ { \mathbb { E } } , c _ { \mathbb { E } } ) = { \frac { x _ { \mathbb { E } } \cdot c _ { \mathbb { E } } } { \| x _ { \mathbb { E } } \| \| c _ { \mathbb { E } } \| } } ,\tag{5}
$$

where ∥ · ∥ denotes the L2 norm. To better model abstract and complex semantic relationships, we next perform similarity estimation in hyperbolic space.

Defining Hyperboloid. Hyperbolic spaces are Riemannian manifolds characterized by a constant negative curvature, whereas Euclidean spaces exhibit zero-curvature (i.e., flat) geometry. There are five well-known isometric models of hyperbolic geometry, including the Lorentz model, the Poincaré ball model, the Poincaré half-space model, the Klein model, and the Hemisphere model [5]. Among these, the Lorentz model is widely adopted due to its numerical stability and computational eficiency.

The Lorentz model is an n-dimensional hyperbolic space on the upper half of a two-sheeted hyperboloid in n+1-dimensional Minkowski space. In the Lorentz space, every vector $x \in \mathbb { R } ^ { n + 1 }$ can be written as $[ x _ { \mathrm { t i m e } } , x _ { \mathrm { s p a c e } } ]$ , where $x _ { \mathrm { t i m e } } \in \mathbb { R }$ denotes the first dimension as the time dimension and $x _ { \mathrm { s p a c e } } \in \mathbb { R } ^ { n }$ denotes the remaining n dimensions as the space dimension. The model is described as:

$$
\mathbb { L } ^ { n } = \{ x \in \mathbb { R } ^ { n + 1 } : \langle x , x \rangle _ { \mathbb { L } } = - 1 / \kappa , \kappa > 0 \} , \quad s . t . \ x _ { \mathrm { t i m e } } = \sqrt { 1 / \kappa + \| x _ { \mathrm { s p a c e } } \| ^ { 2 } } ,\tag{6}
$$

where $- \kappa \in \mathbb { R }$ is the curvature of the space, typically set to $\kappa = 1$ for simplicity, and $\langle \cdot , \cdot \rangle _ { \mathbb { L } }$ denotes the Lorentzian inner product, defined as:

$$
\langle x , y \rangle _ { \mathbb { L } } : = - x _ { \mathrm { t i m e } } y _ { \mathrm { t i m e } } + x _ { \mathrm { s p a c e } } ^ { \top } y _ { \mathrm { s p a c e } } .\tag{7}
$$

Lifting onto Hyperboloid. To project a vector from Euclidean space to hyperbolic space, we first define a mapping from the tangent space $T _ { z } \mathbb { L } ^ { n }$ onto the hyperbolic manifold $\mathbb { L } ^ { n }$ , where $T _ { z } \mathbb { L } ^ { n }$ is a Euclidean space of vectors that are orthogonal to some points $z \in \mathbb { L } ^ { n }$ on the hyperboloid. Given a tangent vector $v \in T _ { z } \mathbb { L } ^ { n }$ , the exponential mapping $T _ { z } \mathbb { L } ^ { n } \to \mathbb { L } ^ { n }$ can be defined as:

$$
\exp _ { z } ^ { \kappa } ( v ) = \cosh ( \sqrt { \kappa } \| v \| _ { \mathbb { L } } ) z + \sinh ( \sqrt { \kappa } \| v \| _ { \mathbb { L } } ) \frac { v } { \sqrt { \kappa } \| v \| _ { \mathbb { L } } } , s . t . \| v \| _ { \mathbb { L } } = \sqrt { \langle v , v \rangle _ { \mathbb { L } } } ,\tag{8}
$$

By treating Euclidean vectors as tangent vectors at the origin $\mathbf { 0 } = ( \sqrt { 1 / \kappa } , 0 , \ldots , 0 ) ^ { \top }$ of hyperbolic space, one can map vectors from Euclidean space to hyperbolic space via $\mathrm { e x p m } _ { 0 } ( \cdot )$

Given a data point $x _ { \mathbb { E } } \in \mathbb { R } ^ { C }$ and a cluster center $c _ { \mathbb { E } } \in \mathbb { R } ^ { C }$ , we first project both onto the Lorentz hyperboloid $\mathbb { L } ^ { n }$ , yielding embeddings $x _ { \mathbb { L } } \in \mathbb { L } ^ { n }$ and $c _ { \mathbb { L } } \in { \mathbb { L } } ^ { n }$ in the hyperbolic space:

$$
x _ { \mathbb { L } } = \mathrm { e x p m } _ { 0 } ^ { \kappa } ( x _ { \mathbb { E } } ) , ~ c _ { \mathbb { L } } = \mathrm { e x p m } _ { 0 } ^ { \kappa } ( c _ { \mathbb { E } } ) .\tag{9}
$$

Estimating Hyperbolic Similarity. Given two hyperbolic embeddings $x _ { \mathbb { L } } , c _ { \mathbb { L } } \in$ $\mathbb { L } ^ { n }$ , we estimate hyperbolic similarity via the Lorentzian distance, as:

$$
D _ { \mathbb { L } } ^ { \kappa } ( x _ { \mathbb { L } } , c _ { \mathbb { L } } ) : = \sqrt { 1 / \kappa } \cdot \cosh ^ { - 1 } ( - \kappa \langle x _ { \mathbb { L } } , c _ { \mathbb { L } } \rangle _ { \mathbb { L } } ) .\tag{10}
$$

Reformulation of ClusterMixer. Despite its attractive properties $( e . g .$ ., grasping abstract and complex semantic relationships), hyperbolic geometry is numerically unstable, whereas Euclidean geometry demonstrates superior performance and stability in simple scenarios. Besides, the operations defined in hyperbolic geometry incur greater computational overhead compared to their Euclidean counterparts. To address these trade-ofs, we extend the ClusterMixer to perform clustering across dual geometries: Euclidean similarity is computed at the finergrained and computationally demanding patch-level, while hyperbolic similarity is estimated at the abstract and computationally eficient window-level. Drawing upon this principle, Equation 4 is rewritten as:

$$
x ^ { \prime } = x + \operatorname { F C } ( \operatorname { N o r m } ( \left[ g _ { W } , g _ { P } \right] ) ) ,\tag{11}
$$

where $[ \cdot , \cdot ]$ denotes the concatenation operation. Here, g<sub>W</sub> represents features aggregated from windows via hyperbolic similarity, while $g _ { P }$ does so for patches using Euclidean similarity. Prior to the FC layer, g<sub>W</sub> is upsampled to match the dimensions of $g _ { P }$ . The computations for both are performed in parallel and remain decoupled from one another.

## 3.5 Network Configuration

Following conventional protocols [16, 72], the network input for image classification is set to $2 2 4 \times 2 2 4$ , yielding $2 2 4 / 4 \times 2 2 4 / 4 = 5 6 \times 5 6$ patches. The network architecture comprises four stages, with the number of patches reduced by a factor of four, while each cluster center remains derived from 4 data points throughout the forward process. In this way, global patch-level mixing would impose a significant computational burden on the clustering algorithm. For instance, at stage 1, there would be $5 6 / 2 \times 5 6 / 2$ cluster centers and $5 6 \times 5 6$ data points, resulting in a $7 8 4 \times 3 1 3 6$ assignment matrix. Given $S = 4 9$ partitioned windows at this stage, our hyperbolic hierarchical clustering strategy reduces the patch-level complexity to $4 9 \times 1 6 \times 6 4$ , at the cost of an additional window-level complexity $1 6 \times 4 9$ . For downstream tasks such as segmentation and detection with variable-sized inputs, reflect padding is applied to maintain this configuration. Additionally, the relative positional bias is linearly interpolated before use to accommodate variable input sizes. See Supplementary for more details of network configuration.

## 4 Experiment

## 4.1 Image Classification

Dataset. The evaluation for image classification is carried out on ImageNet-1K [13], which contains 1.3M training and 50K validation images across 1K classes. Setup. We adopt Timm as the codebase and the experiments are run on 4 A100 GPUs with a batch size of 256. Following [46, 72], all of our models are trained for 310 epochs using a momentum of 0.9 and a weight decay of 0.05. The learning rate is set to 1e-3 and adjusted via a cosine schedule with 5 warmup epochs. For data augmentation, we use Mixup [74], CutMix [73], CutOut [75], and RandAugment [10]. Top-1 classification accuracy is reported.

Results. Table 1 compares HCFormer with other widely-used baselines on image classification. HCFormer achieves superior results over counterparts with similar parameter counts, demonstrating the efectiveness of the proposed method. For example, with 16M parameters, HCFormer outperforms ResMLP-12 by 3.2%, PVT-Tiny by 4.7%, and ConvMixer-1024/12 by 2.0% in top-1 accuracy. HCFormer also demonstrates superior performance compared to other models adhering to the MetaFormer paradigm, as evidenced by HCFormer-Medium outperforming Swin-Tiny (82.4% vs. 81.3%) and MLP-Mixer-B/16 (82.4% vs. 76.4%). These results are impressive, considering the transparent, clustering-based interpretable nature of HCFormer. With a marginal increase in parameters and FLOPs, HCFormer yields a notable improvement compared to existing cluster-based approaches; for instance, it achieves 2.4%/1.7%/1.2% gains in top-1 accuracy across three configurations of model size relative to FEC [9]. To further substantiate the performance of our method, we introduce a smaller variant of HCFormer with only 5.1M parameters, i.e., HCFormer-Nano, which has the fewest parameters among all methods listed in Table 1. Despite its compact size, HCFormer-Nano achieves 73.0% top-1 accuracy, exceeding CoC-Tiny by 1.2% with 0.2M fewer parameters and FEC-Small by 0.3% with 0.4M fewer parameters. Notably, our 16M model exhibits performance comparable to ConvMixer-768/32 [62] (21.1M) and DeiT-Small/16 [61] (22.1M). In conclusion, these promising results manifest the efectiveness and wide benefit of our algorithm.

Table 1: Classification top-1 accuracy on ImageNet-1K [13] val. Throughput (images/s) measured on a single V100 GPU at batch size 128, averaged over last 500 iterations.
<table><tr><td colspan="3">Method</td><td>Param. (M)</td><td>GFLOPs (G)</td><td>Top-1(%) ↑</td><td>Throughput (images/s)</td></tr><tr><td rowspan="6">CIP</td><td>MERU-S/16 MERU-B/16</td><td>[14][1CML23]</td><td>一</td><td></td><td>34.3</td><td></td></tr><tr><td></td><td>[14][1CML23]</td><td></td><td></td><td>37.5</td><td></td></tr><tr><td>MERU-L/16</td><td>[14][1CML23]</td><td></td><td></td><td>38.8</td><td></td></tr><tr><td>HyCoCLIP-S/16</td><td>[49][ICLR24]</td><td></td><td></td><td>41.7</td><td></td></tr><tr><td>HyCoCLIP-B/16</td><td>[49][1CLR24]</td><td></td><td></td><td>45.8</td><td></td></tr><tr><td>HCL</td><td>[20][CVPR23]</td><td>一</td><td>1</td><td>58.5</td><td></td></tr><tr><td rowspan="7">MP</td><td>ResMLP-12</td><td>[60][TPAMI22]</td><td>15.0</td><td>3.0</td><td>76.6</td><td></td></tr><tr><td>ResMLP-24</td><td>[60][TPAMI22]</td><td>30.0</td><td>6.0</td><td>79.4</td><td></td></tr><tr><td>ResMLP-36</td><td>[60][TPAMI22]</td><td>45.0</td><td>8.9</td><td>79.7</td><td></td></tr><tr><td>MLP-Mixer-B/16</td><td>[59][NeurIPS21]</td><td>59.0</td><td>12.7</td><td>76.4</td><td></td></tr><tr><td>MLP-Mixer-L/16</td><td>[59][NeurIPS21]</td><td>207.0</td><td>44.8</td><td>71.8</td><td></td></tr><tr><td>gMLP-Ti</td><td>[39][NeurIPS21]</td><td>6.0</td><td>1.4</td><td>72.3</td><td></td></tr><tr><td>gMLP-S</td><td>[39][NeurIPS21]</td><td>20.0</td><td>4.5</td><td>79.6</td><td></td></tr><tr><td rowspan="8">Atnion</td><td>ViT-B/16</td><td>[16][1CLR20]</td><td>86.0</td><td>55.5</td><td>77.9</td><td></td></tr><tr><td>ViT-L/16</td><td>[16][ICLR20]</td><td>307</td><td>190.7</td><td>76.5</td><td></td></tr><tr><td>PVT-Tiny</td><td>[65][1CCV21]</td><td>13.2</td><td>1.9</td><td>75.1</td><td></td></tr><tr><td>PVT-Small</td><td>[65][1CCV21]</td><td>24.5</td><td>3.8</td><td>79.8</td><td></td></tr><tr><td>DeiT-Tiny/16</td><td>[61][1CML21]</td><td>5.7</td><td>1.3</td><td>72.2</td><td></td></tr><tr><td>DeiT-Small/16</td><td>[61][1CML21]</td><td>22.1</td><td>4.6</td><td>79.8</td><td></td></tr><tr><td>Swin-Tiny</td><td>[41][1CCV21]</td><td>29</td><td>4.5</td><td>81.3</td><td></td></tr><tr><td>Swin-Small</td><td>[41][1CCV21]</td><td>50</td><td>8.7</td><td>83.0</td><td></td></tr><tr><td rowspan="5">Connv·</td><td>ResNet-18</td><td>[24][CVPR16]</td><td>12</td><td>1.8</td><td>69.8</td><td></td></tr><tr><td>ResNet-50</td><td>[24][CVPR16]</td><td>26</td><td>4.1</td><td>79.8</td><td></td></tr><tr><td>ConvMixer-512/16</td><td>[62][TMLR23]</td><td>5.4</td><td>-</td><td>73.8</td><td></td></tr><tr><td>ConvMixer-1024/12</td><td>[62][TMLR23]</td><td>14.6</td><td>-</td><td>77.8</td><td></td></tr><tr><td>ConvMixer-768/32</td><td>[62][TMLR23]</td><td>21.1</td><td>-</td><td>80.2</td><td></td></tr><tr><td rowspan="13">CIusster</td><td>CoC-Tiny</td><td>[46][ICLR23]</td><td>5.3</td><td>1.0</td><td>71.8</td><td>792.5</td></tr><tr><td>CoC-Small</td><td>[46][ICLR23]</td><td>14.0</td><td>2.6</td><td>77.5</td><td>581.8</td></tr><tr><td>CoC-Medium</td><td>[46][ICLR23]</td><td>27.9</td><td>5.5</td><td>81.0</td><td>473.8</td></tr><tr><td>FEC-Small</td><td>[9][CVPR24]</td><td>5.5</td><td>1.4</td><td>72.7</td><td>742.1</td></tr><tr><td>FEC-Base</td><td>[9][CVPR24]</td><td>14.4</td><td>3.4</td><td>78.1</td><td>532.1</td></tr><tr><td>FEC-Large</td><td>[9][CVPR24]</td><td>28.3</td><td>6.5</td><td>81.2</td><td>478.2</td></tr><tr><td>HCFormer-Nano</td><td>(ours)</td><td>5.1</td><td>0.9</td><td>73.0±0.29</td><td>719.0</td></tr><tr><td>HCFormer-Tiny</td><td>(ours)</td><td>7.0</td><td>1.0</td><td>75.1±0.14</td><td>536.0</td></tr><tr><td>HCFormer-Small</td><td>(ours)</td><td>16.1</td><td>2.9</td><td>79.8±0.06</td><td>324.7</td></tr><tr><td>HCFormer-Medium</td><td>(ours)</td><td>33.7</td><td>6.4</td><td>82.4±0.03</td><td>235.7</td></tr></table>

## 4.2 Semantic Segmentation

Dataset. The evaluation for semantic segmentation is carried out on ADE20K [76], which includes 20K training and 2K validation images across 150 classes. Setup. We adopt mmsegmentation as the codebase and the experiments are run on 4 A100 GPUs with a batch size of 16. HCFormer serves as the backbone equipped with Semantic FPN [30]. For a fair comparison, all models are trained for 80k iterations using AdamW. The learning rate starts at 2e-4, with polynomial decay (power=0.9). The backbones are initialized with ImageNet pre-trained weights and the added layers employ Xavier initialization. During training, we use random scale jittering with a factor in [0.5, 2.0] and a crop size of $5 1 2 \times 5 1 2$ for training. During inference, we use one input image scale with shorter side as 512 pixels. Mean intersection-over-union (mIoU) is reported.

Results. Table 2 illustrates our compelling results over semantic segmentation. In terms of mIoU, our HCFormer exceeds CoC [46] and FEC [9] by significant improvements with comparable parameter counts: 40.4% vs. 36.6% vs. 37.7% at 19M parameters, 43.3% vs. 40.2% vs. 40.5% at 35M parameters. Notably,

HCFormer-Small achieves performance comparable to CoC-Medium/4 and FEC-Large, with 6.3M and 13.0M fewer parameters, respectively. Furthermore, our HCFormer-Nano outperforms FEC-Small by 1.5% with 0.3M fewer parameters, while even surpassing CoC-Small (17.6M parameters). These benchmarking results are significant, which provide solid evidence that our method serves as an efective backbone architecture for semantic segmentation. We attribute this advancement to the adopted soft clustering

Table 2: Segmentation mIoU score of diferent backbones with Semantic FPN [30] on ADE20K [76] val.
<table><tr><td rowspan=1 colspan=2>Backbone</td><td rowspan=1 colspan=1>Param.(M)</td><td rowspan=1 colspan=1>mIoU(%) ↑</td></tr><tr><td rowspan=1 colspan=1>ResNet-18          [</td><td rowspan=1 colspan=1>24][CVPR16]</td><td rowspan=2 colspan=1>15.528.5</td><td rowspan=4 colspan=1>32.936.735.739.8</td></tr><tr><td rowspan=1 colspan=1>ResNet-50          [</td><td rowspan=1 colspan=1>24][CVPR16]</td></tr><tr><td rowspan=1 colspan=1>PVT-Tiny</td><td rowspan=1 colspan=1>[65][1CCV21]</td><td rowspan=1 colspan=1>17.0</td></tr><tr><td rowspan=1 colspan=1>PVT-Small</td><td rowspan=1 colspan=1>[65][1CCV21]</td><td rowspan=1 colspan=1>28.2</td></tr><tr><td rowspan=1 colspan=1>CoC-Small/4</td><td rowspan=1 colspan=1>[46][ICLR23]</td><td rowspan=1 colspan=1>17.6</td><td rowspan=1 colspan=1>36.6</td></tr><tr><td rowspan=1 colspan=1>CoC-Medium/4</td><td rowspan=1 colspan=1>[46][ICLR23]</td><td rowspan=1 colspan=1>25.2</td><td rowspan=1 colspan=1>40.2</td></tr><tr><td rowspan=1 colspan=1>FEC-Small</td><td rowspan=1 colspan=1>[9][CVPR24]</td><td rowspan=1 colspan=1>9.1</td><td rowspan=3 colspan=1>35.337.740.5</td></tr><tr><td rowspan=1 colspan=1>FEC-Base</td><td rowspan=1 colspan=1>[9][CVPR24]</td><td rowspan=1 colspan=1>18.0</td></tr><tr><td rowspan=1 colspan=1>FEC-Large</td><td rowspan=1 colspan=1>[9][CVPR24]</td><td rowspan=1 colspan=1>31.9</td></tr><tr><td rowspan=1 colspan=2>HCFormer-Nano   (ours)</td><td rowspan=1 colspan=1>8.8</td><td rowspan=1 colspan=1>36.8</td></tr><tr><td rowspan=1 colspan=2>HCFormer-Tiny   (ours)</td><td rowspan=1 colspan=1>10.3</td><td rowspan=1 colspan=1>37.3</td></tr><tr><td rowspan=1 colspan=2>HCFormer-Small  (ours)HCFormer-Medium (ours)</td><td rowspan=1 colspan=1>18.935.7</td><td rowspan=1 colspan=1>40.443.3</td></tr></table>

mechanism and hierarchical clustering strategies, which enable HCFormer to capture multi-scale features essential for semantic segmentation.

## 4.3 Object Detection and Instance Segmentation

Dataset. The evaluation for object detection and instance segmentation is carried out on MS COCO 2017 [38], which has 118K training and 5K validation images. Setup. We adopt mmdetection as the codebase and the experiments are run on 4 A100 GPUs with a batch size of 16. HCFormer is adopted as the backbone of Mask R-CNN [23] for both object detection and instance segmentation tasks. The backbones are initialized with ImageNet pre-trained weights and the added layers employ Xavier initialization. All models are trained for 12 epochs (1× schedule) using AdamW optimizer with an initial learning rate of 1e-4. During training, images are resized with the shorter side at 800 pixels and the longer side ≤ 1,333 pixels. During inference, the shorter side is also scaled to 800 pixels. Mean Average Precision (mAP) is adopted for evaluation.

Table 3: Object detection and instance segmentation results using Mask R-CNN [23] on COCO [38] val2017. $\mathrm { A P ^ { b o x } }$ and $\mathrm { \ A P ^ { m a s \bar { k } } }$ denote bounding box AP and mask AP.
<table><tr><td>Method</td><td></td><td>[Param. (M)</td><td> $\mathrm { A P ^ { b o x } }$ </td><td> $\mathrm { A P _ { 5 0 } ^ { b o x } }$   $\mathrm { A P } _ { 7 5 } ^ { \mathrm { b o x } }$ </td><td> $\mathrm { { A P } ^ { m a s k } }$ </td><td> $\mathrm { A P _ { 5 0 } ^ { m a s k } }$ </td><td> $\mathrm { A P _ { 7 5 } ^ { m a s k } }$ </td></tr><tr><td>ResNet-18</td><td>[24][CVPR16]</td><td>31.2</td><td>34.0</td><td>54.0</td><td>36.7</td><td>31.2 51.0</td><td>32.7</td></tr><tr><td>ResNet-50</td><td>[24][CVPR16]</td><td>44.2</td><td>38.0</td><td>58.6 41.4</td><td>34.4</td><td>55.1</td><td>36.7</td></tr><tr><td>PVT-Tiny</td><td>[65][1CCV21]</td><td>32.9</td><td>36.7</td><td>59.2</td><td>39.3 35.1</td><td>56.7</td><td>37.3</td></tr><tr><td>PVT-Small</td><td>[65][1CCV21]</td><td>44.1</td><td>40.4</td><td>62.9</td><td>43.8 37.8</td><td>60.1</td><td>40.3</td></tr><tr><td>CoC-Small/4</td><td>[46][ICLR23]</td><td>33.6</td><td>35.9</td><td>58.3</td><td>38.3</td><td>33.8 55.3</td><td>35.8</td></tr><tr><td>CoC-Small/25</td><td>[46][ICLR23]</td><td>33.6</td><td>37.5</td><td>60.1</td><td>40.0 35.4</td><td>57.1</td><td>37.9</td></tr><tr><td>CoC-Small/49</td><td>[46][ICLR23]</td><td>33.6</td><td>37.2</td><td>59.8</td><td>39.7 34.9</td><td>56.7</td><td>37.0</td></tr><tr><td>CoC-Medium/4</td><td>[46][ICLR23]</td><td>42.1</td><td>38.6</td><td>61.1</td><td>41.5 36.1</td><td>58.2</td><td>38.0</td></tr><tr><td>CoC-Medium/25</td><td>[46][ICLR23]</td><td>42.1</td><td>40.1</td><td>62.8</td><td>43.6</td><td>37.4 59.9</td><td>40.0</td></tr><tr><td>CoC-Medium/49</td><td>[46][ICLR23]</td><td>42.1</td><td>40.6</td><td>63.3</td><td>43.9</td><td>37.6 60.1</td><td>39.9</td></tr><tr><td>FEC-Small</td><td>[9][CVPR24]</td><td>24.3</td><td>35.6</td><td>57.5</td><td>38.2</td><td>33.6 54.7</td><td>35.7</td></tr><tr><td>FEC-Base</td><td>[9][CVPR24]</td><td>33.1</td><td>37.9</td><td>60.1</td><td>40.8</td><td>35.5 57.5</td><td>37.8</td></tr><tr><td>FEC-Large</td><td>[9][CVPR24]</td><td>47.1</td><td>39.9</td><td>62.5</td><td>43.2 37.3</td><td>59.5</td><td>39.5</td></tr><tr><td>HCFormer-Nano</td><td>(ours)</td><td>24.0</td><td>36.0</td><td>57.8</td><td>38.4</td><td>34.0 55.1</td><td>36.0</td></tr><tr><td>HCFormer-Tiny</td><td>(ours)</td><td>25.5</td><td>36.4</td><td>58.5</td><td>38.6 34.5</td><td>55.8</td><td>36.5</td></tr><tr><td>HCFormer-Small</td><td>(ours)</td><td>34.1</td><td>38.7</td><td>60.9</td><td>41.8</td><td>36.0 58.0</td><td>38.3</td></tr><tr><td>HCFormer-Medium</td><td> $\mathbf { \tau } ( \mathbf { o u r s } )$ </td><td>50.8</td><td>40.9</td><td>63.0</td><td>44.0</td><td>37.6 59.8</td><td>40.0</td></tr></table>

Results. Table 3 reports the numerical results for both object detection and instance segmentation. Empirically, our method achieves superior performance over other competitors under identical network sizes across all evaluation metrics. For instance, at the 34M model scale, HCFormer outperforms recent clusteringbased advancements (i.e., CoC [46] and FEC [9]), achieving 38.7% vs. 37.2% vs. 37.9% $\mathrm { A P ^ { b o x } }$ for object detection and 36.0% vs. 35.4% vs. 35.5% $\mathrm { { A P } ^ { m a s k } }$ for instance segmentation, respectively. With fewer parameters, HCFormer-Nano outperforms FEC-Small by 0.4% $\mathrm { A \dot { P } ^ { b o x } }$ and 0.4% $\mathrm { \bar { A } P ^ { m a s k } }$ . Moreover, HCFormer-Medium achieves 40.9% $\mathrm { A P ^ { b o x } }$ in object detection and 37.6% $\mathrm { { A P } ^ { m a s k } }$ in instance segmentation, surpassing all other methods except PVT-Small, to which it is only marginally inferior (by 0.1% $\mathrm { A P ^ { m a s k } ) }$ in instance segmentation. However, its advantage narrows when compared to CoC-Medium/49. We posit that this is because object detection performance benefits from the number of cluster centers, which enables CoC-Medium/49 to achieve results competitive with ours. Crucially, this accuracy comes at the expense of eficiency; HCFormer-Medium achieves a 37% higher throughput (14.0 vs. 10.2) than CoC-Medium/49 (as shown in Supplementary Table S2), underscoring its superior computational economy.

(d) Cluster number for window-level clustering.

Table 4: A set of ablative experiments on ImageNet [54] val. Hier. Clus. indicates hierarchical clustering; Shift. implies shifted windows; Hyp. Geo. represents hyperbolic geometry; Rel. Pos. denotes relative position; Euc. and Hyp. are Euclidean and hyperbolic geometry, respectively.
<table><tr><td>Strategy</td><td>top-1(%)↑</td><td>top-5(%)↑</td></tr><tr><td>Shift.</td><td>72.7</td><td>91.1</td></tr><tr><td>Hier. Clus.</td><td>73.4</td><td>91.4</td></tr></table>

(a) Hierarchical clustering for ClusterMixer.

<table><tr><td>Window</td><td>Patch</td><td></td><td>top-1(%)↑ top-5(%)↑</td></tr><tr><td>Euc.</td><td>Euc.</td><td>73.4</td><td>91.4</td></tr><tr><td>Hyp.</td><td>Hyp.</td><td>74.7</td><td>92.3</td></tr><tr><td>Hyp.</td><td>Euc.</td><td>75.1</td><td>92.5</td></tr></table>

<table><tr><td>curvature κ</td><td>top-1(%)↑</td><td>top-5(%)↑</td></tr><tr><td>0.1</td><td>74.9</td><td>92.4</td></tr><tr><td>1.0</td><td>75.1</td><td>92.5</td></tr><tr><td>10.0</td><td>74.8</td><td>92.4</td></tr></table>

(b) Geometry for clustering.
<table><tr><td>cluster number</td><td>top-1(%)↑ top-5(%)↑</td><td></td></tr><tr><td>9</td><td>74.9</td><td>92.4</td></tr><tr><td>16</td><td>75.1</td><td>92.5</td></tr><tr><td>25</td><td>74.6</td><td>92.3</td></tr></table>

<table><tr><td>Hier. Clus.</td><td>Hyp. Geo.</td><td>Rel. Pos.</td><td>top-1(%)↑ top-5(%)↑</td><td></td><td>mIoU(%) ↑|</td><td>APbox</td><td>APmask</td></tr><tr><td></td><td>√</td><td></td><td>71.9 72.7</td><td>90.8</td><td>35.1</td><td>34.3</td><td>32.7</td></tr><tr><td>√</td><td></td><td>√ √</td><td>73.4</td><td>91.1 91.4</td><td>34.6 35.2</td><td>34.4</td><td>32.8 33.1</td></tr><tr><td>√</td><td>√</td><td></td><td>74.4</td><td>92.2</td><td>36.4</td><td>34.7 35.9</td><td>34.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>75.1</td><td>92.5</td><td>37.3</td><td>36.4</td><td>34.5</td></tr></table>

(e) Key Component Analysis.

## 4.4 Diagnostic Experiment

For thorough evaluation, we perform a series of ablative studies on ImageNet-1K [13] val for image classification to investigate the following aspects. HCFormer-Tiny is utilized as the baseline.

Shifted Windows or Hierarchical Clustering. We first investigate the efectiveness of the proposed hierarchical clustering strategies for acquiring global contextual information in representation learning, compared with the shifted window mechanism [41]. As outlined in Table 4a, while the shifted window mechanism remains efective, our hierarchical clustering strategies are better suited for ClusterMixer (i.e., 72.7% → 73.4%). This may arise because the shifted operation only enables localized interaction between adjacent windows, thereby remaining constrained by the limited receptive field.

Hyperbolic or Euclidean Distance. We next examine the impact of hyperbolic versus Euclidean geometry on window-level clustering. As summarized in Table 4b, we find that the use of hyperbolic geometry yields a notable performance gain over Euclidean geometry by 1.6% top-1 accuracy. When clustering is performed entirely in hyperbolic space, a marginal performance degradation (0.3% top-1) is observed, accompanied by a substantial reduction in computational eficiency (throughput decreases from 536.0 to 319.2). We posit this may be because local visual neighborhoods are usually nearly flat, making Euclidean space more suitable for preserving local geometry.

Curvature in Hyperbolic Space. As shown in Table 4c, it can be seen that the variation in curvature exhibits minimal impact. We hypothesize that this may be attributed to the fact that the current head dimension (24 for Tiny, 32 for Small and Medium) is suficient to capture the relational structure in the embedding space across varying curvature regimes. We attribute the numerical safeguards of our method from two aspects: i) In our assignment logits, the learnable scale α Eq. 2 helps absorb curvature-scale changes. This aligns with prior work demonstrating that representations across diferent curvatures can be related via scaling transformations [8]. ii) We clip all features to a bounded norm before hyperbolic mapping, which limits the input magnitude of the cosh / sinh terms, following [22].

![](images/e9821cf71c89b74a8867ccb55031ef8a4a3f71aca1c21898c33f13451a2c1c8e.jpg)  
Fig. 3: Visualization of clustering maps for our HCFormer-Tiny on ImageNet [54] val. Diferent colored masks indicate diferent clusters, ranging from 2 to 5.

Cluster Number for Window-level Clustering. Table 4d presents that a higher cluster count in the hyperbolic space yields performance gains, peaking at 16 clusters. This may be because 25 cluster centers are excessive for the number of windows, given the 49 windows defined in our experimental setup.

Key Component Analysis. We finally ablate the key design elements in the proposed HCFormer. As shown in Table 4e, the baseline of our model achieves only 71.9% without the proposed components. Empirical results demonstrate that all components provide complementary benefits, as the absence of either leads to performance degradation: -2.3% without hierarchical clustering, -1.6% without hyperbolic geometry, and -0.6% without relative position.

## 4.5 Visualization of Clustering

Our configuration ensures that the feature representation at each center of the network architecture (except the final stage) is derived from a 2×2 token grid, while tokens are progressively merged to reduce their count by a factor of 2×2 by downsampling operator (See Supplementary Table S3). This mechanism indicates that HCFormer develops gradually expanding clusters during feature extraction, ultimately forming 16 clusters at the final stage. Following FEC’s visualization protocol [9], we also use K-means to reduce the number of clusters for better visualization. As illustrated in Figure 3, the clustering results demonstrate that our HCFormer is able to capture intrinsic relational patterns among tokens.

## 5 Conclusion

Clustering represents a promising yet underexplored direction in architectural design, lacking an efective and eficient framework that enables it to emerge as a competitive alternative to mainstream architectures. This paper proposes a new token-mixer design, termed ClusterMixer, which leverages clustering algorithms to enhance token aggregation. To leverage this design efectively, we advocate a universal vision framework, designated as HCFormer, along with a set of training strategies tailored for ClusterMixer. Empirical results demonstrate that this framework improves interpretability over conventional convolution- and attention-based methods while achieving superior performance to existing clusterbased approaches. Given its favorable balance of interpretability and performance, we expect that this approach will potentially benefit a wider range of visual tasks.

[1] Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., Süsstrunk, S.: Slic superpixels compared to state-of-the-art superpixel methods. IEEE TPAMI (2012)

[2] Alley, E.C., Khimulya, G., Biswas, S., AlQuraishi, M., Church, G.M.: Unified rational protein engineering with sequence-based deep representation learning. Nature methods (2019)

[3] Atigh, M.G., Schoep, J., Acar, E., Van Noord, N., Mettes, P.: Hyperbolic image segmentation. In: CVPR (2022)

[4] Bai, Y., Ying, Z., Ren, H., Leskovec, J.: Modeling heterogeneous hierarchies with relation-specific hyperbolic cones. NeurIPS (2021)

[5] Cannon, J.W., Floyd, W.J., Kenyon, R., Parry, W.R., et al.: Hyperbolic geometry. Flavors of geometry (1997)

[6] Chamberlain, B.P., Clough, J., Deisenroth, M.P.: Neural embeddings of graphs in hyperbolic space. arXiv preprint arXiv:1705.10359 (2017)

[7] Chami, I., Wolf, A., Juan, D.C., Sala, F., Ravi, S., Ré, C.: Low-dimensional hyperbolic knowledge graph embeddings. arXiv preprint arXiv:2005.00545 (2020)

[8] Chami, I., Ying, Z., Ré, C., Leskovec, J.: Hyperbolic graph convolutional neural networks. NeurIPS (2019)

[9] Chen, G., Li, X., Yang, Y., Wang, W.: Neural clustering based visual representation learning. In: CVPR (2024)

[10] Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: Randaugment: Practical automated data augmentation with a reduced search space. In: CVPRW (2020)

[11] Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: CVPR (2005)

[12] Dempster, A.P., Laird, N.M., Rubin, D.B.: Maximum likelihood from incomplete data via the em algorithm. Journal of the royal statistical society: series B (methodological) 39(1), 1–22 (1977)

[13] Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: CVPR (2009)

[14] Desai, K., Nickel, M., Rajpurohit, T., Johnson, J., Vedantam, S.R.: Hyperbolic image-text representations. In: ICML (2023)

[15] Ding, Y., Li, L., Wang, W., Yang, Y.: Clustering propagation for universal medical image segmentation. In: CVPR (2024)

[16] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. ICLR (2020)

[17] Feng, T., Quan, R., Wang, X., Wang, W., Yang, Y.: Interpretable3d: An ad-hoc interpretable classifier for 3d point clouds. In: AAAI (2024)

[18] Franco, L., Mandica, P., Kallidromitis, K., Guillory, D., Li, Y.T., Darrell, T., Galasso, F.: Hyperbolic active learning for semantic segmentation under domain shift. arXiv preprint arXiv:2306.11180 (2023)

[19] Ganea, O., Bécigneul, G., Hofmann, T.: Hyperbolic neural networks. NeurIPS (2018)

[20] Ge, S., Mishra, S., Kornblith, S., Li, C.L., Jacobs, D.: Hyperbolic contrastive learning for visual representations beyond objects. In: CVPR (2023)

[21] Guo, M.H., Lu, C.Z., Liu, Z.N., Cheng, M.M., Hu, S.M.: Visual attention network. Computational visual media (2023)

[22] Guo, Y., Wang, X., Chen, Y., Yu, S.X.: Clipped hyperbolic classifiers are super-hyperbolic classifiers. In: CVPR (2022)

[23] He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: ICCV (2017)

[24] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016)

[25] Hong, J., Wei, J., Wang, W.: Learning human-object interaction as groups. NeurIPS (2025)

[26] Huang, Z., Li, Y.: Interpretable and accurate fine-grained recognition via region grouping. In: CVPR (2020)

[27] Jolion, J.M., Meer, P., Bataouche, S.: Robust clustering with applications in computer vision. IEEE TPAMI (1991)

[28] Jun, L., Jinpeng, W., Chaolei, T., Niu, L., Long, C., Min, Z., Yaowei, W., Shu-Tao, X., Bin, C.: Hlformer: Enhancing partially relevant video retrieval with hyperbolic learning. ICCV (2025)

[29] Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., Lempitsky, V.: Hyperbolic image embeddings. In: CVPR (2020)

[30] Kirillov, A., Girshick, R., He, K., Dollár, P.: Panoptic feature pyramid networks. In: CVPR (2019)

[31] Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. NeurIPS (2012)

[32] Li, K., Wang, Y., Gao, P., Song, G., Liu, Y., Li, H., Qiao, Y.: Uniformer: Unified transformer for eficient spatiotemporal representation learning. ICLR (2022)

[33] Li, L., Wang, W., Zhou, T., Li, J., Yang, Y.: Unified mask embedding and correspondence learning for self-supervised video segmentation. In: CVPR (2023)

[34] Li, L., Zhou, T., Wang, W., Li, J., Yang, Y.: Deep hierarchical semantic segmentation. In: CVPR (2022)

[35] Li, Z., Chen, J.: Superpixel segmentation using linear spectral clustering. In: CVPR (2015)

[36] Liang, C., Wang, W., Miao, J., Yang, Y.: Gmmseg: Gaussian mixture based generative semantic segmentation models. NeurIPS (2022)

[37] Liang, J.C., Zhou, T., Liu, D., Wang, W.: Clustseg: Clustering for universal segmentation. In: ICML (2023)

[38] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV (2014)

[39] Liu, H., Dai, Z., So, D., Le, Q.V.: Pay attention to mlps. NeurIPS (2021)

[40] Liu, Q., Nickel, M., Kiela, D.: Hyperbolic graph neural networks. NeurIPS (2019)

[41] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV (2021)

[42] Long, T., Mettes, P., Shen, H.T., Snoek, C.G.: Searching for actions on the hyperbole. In: CVPR (2020)

[43] Long, T., van Noord, N.: Cross-modal scalable hyperbolic hierarchical clustering. In: ICCV (2023)

[44] Lowe, D.G.: Distinctive image features from scale-invariant keypoints. IJCV (2004)

[45] Ma, X., Qin, C., You, H., Ran, H., Fu, Y.: Rethinking network design and local geometry in point cloud: A simple residual mlp framework. ICLR (2022)

[46] Ma, X., Zhou, Y., Wang, H., Qin, C., Sun, B., Liu, C., Fu, Y.: Image as set of points. ICLR (2023)

[47] Madhulatha, T.S.: An overview on clustering methods. arXiv preprint arXiv:1205.1117 (2012)

[48] Nickel, M., Kiela, D.: Poincaré embeddings for learning hierarchical representations. NeurIPS 30 (2017)

[49] Pal, A., van Spengler, M., di Melendugno, G.M.D., Flaborea, A., Galasso, F., Mettes, P.: Compositional entailment learning for hyperbolic vision-language models. In: ICLR (2024)

[50] Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. NeurIPS (2017)

[51] Quan, R., Wang, W., Ma, F., Fan, H., Yang, Y.: Clustering for protein representation learning. In: CVPR (2024)

[52] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)

[53] Ren, Malik: Learning a classification model for segmentation. In: ICCV (2003)

[54] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recognition challenge. IJCV (2015)

[55] Schmid, C., Mohr, R.: Local grayvalue invariants for image retrieval. IEEE TPAMI (2002)

[56] Shimizu, R., Mukuta, Y., Harada, T.: Hyperbolic neural networks++. arXiv preprint arXiv:2006.08210 (2020)

[57] Simonyan, K., Zisserman, A.: Very deep convolutional networks for largescale image recognition. ICLR (2015)

[58] Sun, J., Li, Y., Fang, H.S., Lu, C.: Three steps to multimodal trajectory prediction: Modality clustering, classification and synthesis. In: ICCV (2021)

[59] Tolstikhin, I.O., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J., et al.: Mlp-mixer: An all-mlp architecture for vision. NeurIPS (2021)

[60] Touvron, H., Bojanowski, P., Caron, M., Cord, M., El-Nouby, A., Grave, E., Izacard, G., Joulin, A., Synnaeve, G., Verbeek, J., et al.: Resmlp: Feedforward

networks for image classification with data-eficient training. IEEE TPAMI (2022)

[61] Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jégou, H.: Training data-eficient image transformers & distillation through attention. In: ICML (2021)

[62] Trockman, A., Kolter, J.Z.: Patches are all you need? arXiv preprint arXiv:2201.09792 (2022)

[63] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. NeurIPS (2017)

[64] Wang, W., Han, C., Zhou, T., Liu, D.: Visual recognition with deep nearest centroids. ICLR (2022)

[65] Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In: ICCV (2021)

[66] Wei, J., Zhou, T., Yang, Y., Wang, W.: Nonverbal interaction detection. In: ECCV (2024)

[67] Xu, C., Mao, W., Zhang, W., Chen, S.: Remember intentions: Retrospectivememory-based trajectory prediction. In: CVPR (2022)

[68] Xu, R., Wunsch, D.: Survey of clustering algorithms. IEEE TNNLS (2005)

[69] Yin, J., Zhou, D., Zhang, L., Fang, J., Xu, C.Z., Shen, J., Wang, W.: Proposalcontrast: Unsupervised pre-training for lidar-based 3d object detection. In: ECCV (2022)

[70] Yu, Q., Wang, H., Kim, D., Qiao, S., Collins, M., Zhu, Y., Adam, H., Yuille, A., Chen, L.C.: Cmt-deeplab: Clustering mask transformers for panoptic segmentation. In: CVPR (2022)

[71] Yu, Q., Wang, H., Qiao, S., Collins, M., Zhu, Y., Adam, H., Yuille, A., Chen, L.C.: k-means mask transformer. In: ECCV (2022)

[72] Yu, W., Luo, M., Zhou, P., Si, C., Zhou, Y., Wang, X., Feng, J., Yan, S.: Metaformer is actually what you need for vision. In: CVPR (2022)

[73] Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization strategy to train strong classifiers with localizable features. In: ICCV (2019)

[74] Zhang, H., Cisse, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. ICLR (2018)

[75] Zhong, Z., Zheng, L., Kang, G., Li, S., Yang, Y.: Random erasing data augmentation. In: AAAI (2020)

[76] Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ade20k dataset. In: CVPR (2017)

[77] Zhou, T., Wang, W., Konukoglu, E., Van Gool, L.: Rethinking semantic segmentation: A prototype view. In: CVPR (2022)