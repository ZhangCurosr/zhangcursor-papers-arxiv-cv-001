# CoViG: Coarsened Vision GNN for Image Classification

Martial Guidez, Stefan Duffner, Christophe Garcia

INSA Lyon, CNRS, Universite Claude Bernard Lyon 1, LIRIS, UMR5205, 69621 Villeurbanne, France´

martial.guidez@gmail.com

## Abstract

Although Graph Neural Networks are widely used in various domains, they remained mostly unexplored in computer vision. Convolutional Neural Networks and Vision Transformers were leading the field, until the quite recent introduction of Vision Graphs (ViG) that treat images as a set of nodes and apply graph convolution on their k-NN graphs. ViGs show impressive accuracy on image classification but their computational cost remains relatively high. Inspired by optimization methods for GNNs, we present CoViG, a novel approach combining graph coarsening and ViGs. Our method introduces an additional MinCUTpooling step within the Grapher component ofintermediate ViG blocks. By simplifying the intermediate graphs and thus decreasing the number of nodes of specific blocks, our approach considerably reduces the computational cost of ViG during inference. We evaluated our approach on ImageNet with several ViG architectures, and we obtained accuracies equivalent to the original ViG models with only about 51% of FLOPs.

## 1. Introduction

Although Convolutional Neural Networks (CNNs) and Vision Transformer (ViTs) are the predominant neural network architectures of state-of-the-art computer vision models, a new type of architecture called Vision Graphs (ViGs) has been introduced recently [14] showing excellent performances on the standard ImageNet image classification benchmark. Similar to ViT, ViGs divide the input image into non-overlapping patches, and then compute embeddings that are processed further by the backbone. But, instead of the Transformer blocks that implicitly integrate the contextual information using attention, ViGs model these spatial relations and visual similarities more explicitly using graphs encoded with a Graph Neural Network (GNN) based architecture.

Although the costly attention mechanism is avoided in ViGs, computational complexity remains still relatively high due to the graph generation process and GCN at each block of the model. Much effort has been spent in the last decade to increase the efficiency of CNNs and ViTs (and Transformers in general), both in terms of computation and memory. However, only a few recent works have specifically tackled the reduction of ViG complexity [29–31]. The main computational bottleneck is the k-Nearest-Neighborbased graph construction in each ViG block, where the similarity between each pair of patch embeddings needs to be computed, and a Graph Convolutional Network (GCN) is applied on this graph.

![](images/7fa2563b30e2473d074ef35fca36e281c6cabe4c14f1c042ec2f0495d47c02d0.jpg)  
Figure 1. Overview of the CoViG architecture: graph coarsening reduces the node set from l nodes to j super-nodes between two stages of ViG blocks. In this illustration, only a single coarsening step is applied.

Our proposed method directly addresses this by introducing a graph coarsening approach at pre-determined stages (blocks) of the model. More specifically, we replace one or several of the Grapher (GN) components with, what we call a Coarsened Grapher that simplifies the graph into a reduced version with much fewer nodes. This process is represented in Figure 1. This reduction then propagates to all subsequent blocks of the model and has a direct impact on the computational efficiency. Our coarsening approach is based on the MinCUT Pooling algorithm [2] that learns the clustering jointly with the rest of the model and a taskspecific loss, as opposed to other GNN or ViG optimisation methods where this clustering is performed independently [5, 31].

Our contributions can be summarised as follows:

• We propose a new Vision Graph-based architecture with reduced computational complexity thanks to an integrated and differentiable graph coarsening approach based on MinCUT Pooling.

• By introducing a new Coarsened Grapher module, the proposed approach is flexible and adaptable to different (pre-trained) ViG models. Before training or fine-tuning, one simply has to choose one or several blocks that should be reduced and determine the desired compression rate for each coarsening step.

• We extensively evaluated our approach on the ImageNet benchmark, with different coarsening configurations and rates, and with different ViG models, and we were able to reduce the FLOPs up to a factor of 2 without decreasing classification accuracy.

## 2. Related Work

This section provides a brief overview of the different architectures used for image classification and the presentation of the compression methods that are applied to them.

## 2.1. CNN and ViT

Convolutional Neural Networks (CNNs) have been the dominant paradigm in computer vision, driving remarkable advances in image classification [22], object detection [34], and semantic segmentation [37]. LeNet [24] established the fundamental principles of local receptive fields, weight sharing, and hierarchical feature extraction. It was followed by AlexNet [22], a breakthrough on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). This work paved the way for the well known VGGNet [38], ResNet [19], and MobileNet [21].

The generalized use of CNNs led to the need of compressing the networks to alleviate their memory footprint and computational complexity [4, 16]. Many methods have been developed such as Pruning [15, 39], Knowledge Distillation [1, 20, 35], Early-Exit [32, 40], and Quantization [11, 13]. CNNs have long dominated the field of computer vision, primarily due to their ability to capture spatial hierarchies through convolutional layers compared to classic non-convolutional Deep Neural Networks (DNN).

After the rise of attention in the field of NLP [42], Dosovitsky et al. [6] have developed ViT for image recognition. Compared to CNNs, ViTs capture global image context more effectively especially where long-range dependencies are crucial. However, it usually needs to be trained on very large-scale datasets. ViTs are also computationally expensive due to the self-attention mechanism, which scales quadratically with the number of image patches, leading to high memory and computation resource requirements. These computational costs have motivated the research of compression techniques for ViT.

Some CNN compression techniques have been successfully applied to ViT. DeiT [41] have applied knowledge distillation to ViT by adding a specific token. Liu et al. [27] also achieved post-training quantization on ViT.

Other methods also try to reduce the complexity of the self-attention mechanism, which scales quadratically with the patch number. For example, Wang et al. proposed pyramid architecture [43] to improve ViT performances while reducing the network cost. Liu et al. proposed the Swin Transformer [26], which has a linear computational complexity with respect to the image size.

Finally, some compression methods leverage ViTspecific architecture. For example, Rao et al. introduce DynamicViT [33], to prune unnecessary tokens. Bolya et al. explore token merging in ToMe [3].

## 2.2. From GNN to Vision GNN

In many applications, objects and relationships between examples in a dataset can be represented as a graph. However, common machine learning algorithms rely on the assumption that instances are independent of each other. This is invalid for graph data. This leads to ever-increasing use of graph-based methods in different fields. Neural networks that operate on graphs (GNN) have been introduced by Gori et al. [12]. Convolution-like propagation in GNN was first introduced by Duvenaud et al. [7] and later improved to became the modern GCN by Micheli [28]. GCN have been used widely in computer vision applications and specific visual tasks with naturally constructed graphs, such as point cloud classification and segmentation [23, 44], scene graph generation [45], and action recognition [46].

The idea behind Vision Graphs (ViG) [14] is to model the semantic spatial relationship of image parts (i.e. patches) together with their content in a graph. This approach introduces a GCN-based backbone that directly processes image data. It uses patches as nodes and builds a graph with the top first-order similarity as neighbors. This model outperformed state-of-the-art CNN and ViT-based architectures on standard image classification benchmarks. They proposed two versions of their network: an isotropic version and a pyramid version (PViG). Inspired by this work, Munir et al. have proposed an hybrid CNN-GNN architecture MobileViG [29] based on PViG, that uses Sparse Vision Graph Attention instead of the previously used k-Nearest Neighbors (k-NN) graph attention. Munir et al. also proposed GreedyViG [30] which uses Dynamic Axial Graph to replace k-NN graph construction on PViG. Parikh et al. have proposed ClusterViG [31], introducing Dynamic Efficient Graph Convolution. It reduces the computational expense of image graph construction by partitioning the image patch feature grid into several partitions and performing graph construction in parallel within each partition. K-NN is performed locally within each partition in parallel, reducing the cost from $O \left( N ^ { 2 } \right)$ to $O \left( N ^ { 2 } / k ^ { 2 } \right)$ . The number of nodes N never changes, they are just grouped into partitions. The used clustering method is purely geometric and cannot adapt to learn semantically meaningful groupings.

![](images/165f7df9da40138b6efd128c1fb340e1828e2164e37aa626b4815bb2847078e5.jpg)  
Figure 2. Comparison of the baseline Grapher block (a) and the proposed coarsened Grapher block (b).

## 2.3. Graph compression

Classic compression methods like pruning [51], quantization [50] or knowledge distillation [47] have been successfully applied to GNN. Graphs can be quite complex and this has led to the exploration of graph condensation to replace wide graphs with smaller versions that are easier to manipulate in term of computations [18]. There are three main categories of graph condensations: graph sparsification, coarsening and condensation. Graph sparsification can be seen as a form of pruning, it aims to selects existing nodes or edges and remove the rest of the graph. Graph coarsening is the surjective mapping of a graph that results in a smaller version of the graph made of ”super nodes” that aggregate original nodes. Finally, graph condensation is the creation of a small synthetic graph that aims to provide similar performances as the original graph.

There are different graph coarsening techniques. We can cite DiffPool [48], that learns a soft assignment matrix S via a full GNN pass, costing $O ( N ^ { 2 } )$ memory for the assignment matrix. The Top-K method [10] select a subset of existing nodes and drops the rest, retaining the subgraph induced by selected nodes. This works for sparse graphs but in ViG’s dense k-NN setting it creates coverage gaps. SAG-Pool [25] is quite similar to Top-K but the score is computed differently using a GCN meaning that this method in contrast to the other is graph-aware. Graclus [5] is a deterministic graph algorithm that matches neighboring nodes. For each node, it selects the node that maximizes a criterion and matches those nodes together. This process can be applied iteratively to coarsen even more.

## 3. Method

In this section, we will first describe the basic ViG architecture and the MinCUT Pooling algorithm. Then, we will introduce our proposed CoViG approach.

## 3.1. Vision GNN

Han et al. [14] proposed the first ViG; this method can be considered a replacement for the self-attention mechanism of a Vision Transformer, using graph message passing instead of attention. The image is first divided into visual nodes. Each node corresponds to the receptive field of one feature map location.

![](images/4cbab8252ddb5266fe98948fdf185d97ebd66c0e9319836ebea016128cc8edaf.jpg)  
Figure 3. Overview of the Vision Graph Neural Network (ViG) architecture.

The network is composed of several blocks that repeatedly follow the following path shown in Figs. 2a and 3. For each node, ViG computes the k nearest neighbors, producing a graph. This graph is based on feature similarity, not spatial adjacency. It is then passed through a GCN, which gathers neighbour features, aggregates the information and updates the node representation. Each node is then processed by a feed-forward network (FFN).

After the final block, global pooling is applied to all nodes, and the resulting feature vector is fed to a linear classifier.

In terms of computational complexity, consider the ViG-Ti (tiny) architecture: the number of nodes is $N = 1 9 6$ , the number of feature channels $C = 7 6 8$ , and the number of neighbors k = 9. The FLOPs of a single GRAPHER+FFN block scale as:

$$
\underbrace { N C ^ { 2 } } _ { f c 1 } + \underbrace { 2 N ^ { 2 } C } _ { d i s t a n c e } + \underbrace { 8 N C ^ { 2 } } _ { M R C o n v } + \underbrace { 2 N C ^ { 2 } } _ { f c 2 }\tag{1}
$$

Figure 4 visualizes the k-NN graph structure and node feature representations at two depths of the network: the first block and the last block. In the first block, node features exhibit rich spatial diversity and the corresponding k-NN graph is unstructured with edges connecting spatially distant nodes at low similarity scores, encoding meaningful long-range semantic relationships across the image. This confirms that the full node resolution is justified at early depth, where spatial heterogeneity is high and every node contributes discriminative information. At the final block, the picture changes, the feature map reveals that the vast majority of nodes have collapsed to a near-uniform representation. Only a handful of patches retains distinct features. Similarly, the k-NN graph degenerates into a uniform high edge similarity graph.

![](images/f262f4b202ce26ee0a45ef912c5b7ade605a48ab833f93cdb2b196fe2d3ac6db.jpg)  
Figure 4. Visualization of the graph representation learned by the ViG [14]. From left to right: the input image overlaid with the 14×14 graph nodes, the dynamically constructed k-NN graph based on feature similarity, and a PCA projection of the learned node features into RGB space. Subfigures a) and b) correspond to different backbone blocks, illustrating the evolution of graph connectivity and feature representations throughout the network.

At the deepest blocks, the expensive computation cost of the Grapher and FFN blocks are not justified because the graph does not carries much structural information. These observations reveal the feature diversity collapse of the ViG and justifies the introduction of graph coarsening to not maintain redundant nodes through the entire network. In our approach, we have chosen to apply a learned pooling operation (MinCUT) that can merge near identical nodes into a set of $K < < N$ super nodes.

## 3.2. MinCUT Pooling

Bianchi et al. [2] proposed a pooling / coarsening method that produces a soft assignment matrix S learned via backpropagation. In contrast to ClusterViG [31] or Graclus [5] that are not differentiable, MinCUT allows model to learn which patch to merge based on the usual task-specific loss - i.e. cross entropy (CE) for classification - and the unsupervised loss $\mathcal { L } _ { u }$ (Equation 2):

$$
\mathcal { L } _ { u } = \mathcal { L } _ { c } + \mathcal { L } _ { o } = \underbrace { - \frac { T r \left( \mathbf { S } ^ { T } \mathbf { A } \mathbf { S } \right) } { T r \left( \mathbf { S } ^ { T } \mathbf { D } \mathbf { S } \right) } } _ { \mathcal { L } _ { c } } + \underbrace { \left\| \frac { \mathbf { S } ^ { T } \mathbf { S } } { \| \mathbf { S } ^ { T } \mathbf { S } \| _ { F } } - \frac { \mathbf { I } _ { K } } { \sqrt { K } } \right\| _ { F } } _ { \mathcal { L } _ { o } } ,\tag{2}
$$

where A is the normalized adjacency matrix of the graph, $\mathbf { D } = \mathrm { d i a g } \left( \mathbf { A } \mathbf { 1 } _ { N } \right)$ is the degree matrix. ${ \mathbf { I } } _ { K }$ refers to the identity matrix, $\mathbf { 1 } _ { N }$ refers to the matrix of ones, and $\| . \| _ { F } ,$ to the Frobenius norm defined as:

$$
\| A \| _ { F } : = \sqrt { T r ( A ^ { * } A ) } .\tag{3}
$$

The unsupervised loss is composed of the cut loss $\mathcal { L } _ { c }$ and the orthogonality loss $\mathcal { L } _ { o }$

The cut loss explicitly encourages the assignment to respect graph connectivity. It means that strongly connected nodes (nodes sharing similar features, close in feature space) are encouraged to be merged together. This loss prevents degenerate solutions where all nodes collapse to the same cluster. However, when all nodes are assigned to the same cluster, the numerator and denominator are equal and $\mathcal { L } _ { c }$ is minimized.

To avoid such local minimum, $\mathcal { L } _ { o }$ encourages the MLP to find clusters that are orthogonal between each other (i.e. avoiding assigning all nodes to all clusters).

Ensuring that each node is assigned to at least one supernode avoids any coverage gaps that can occur with Top-K or SAGPool.

The fact that MinCUT performs soft assignment - contrary to Top-K, SAGPool, Graclus, and ClusterViG - allows patches at boundaries between semantic regions to be assigned to several super-nodes. It preserves boundary information that would be lost by hard methods.

Comparing to DiffPool, MinCUT computes S via an MLP on node features costing $O ( N \times K )$ memory for the assignment matrix. DiffPool also requires a separate GNN to compute S, it effectively doubles the graph computation at the coarsening block.

For all these reasons, we have chosen to use MinCUT over other coarsening methods to coarsen ViG.

## 3.3. CoViG architecture

We will now describe how the CoViG architecture integrates MinCUT Pooling in ViG.

While the original ViG keeps the same number of nodes through all blocks, CoViG adds a coarsened Grapher to replace the original Grapher. This new Grapher reduces the number of nodes according to different parameters. Several coarsened Grapher can be placed along the model to reduce multiple time the nodes number. The position of the different pooling blocks is defined using two parameters: coarsen start $( C _ { s t a r t } )$ is the index of the first block where coarsening is applied; coarsen step $( C _ { s t e p } )$ is the distance between two coarsening blocks. The coarsening strength (the number of coarsened nodes) is defined by two parameters: coarsening ratio $( C _ { r } )$ is the fraction of nodes kept after each coarsening blocks; node threshold $( N _ { T } )$ is the minimum number of nodes.

For example, using a ViG-Ti as base model (with 196 nodes and 12 blocks) if the coarsening start on the third block $( C _ { s t a r t } ~ = ~ 3 )$ with coarsening applied each sixth blocks $( C _ { s t e p } = 6 )$ , 50% of the nodes pruned each time $( C _ { r } ~ = ~ 0 . 5 )$ and a minimum of 49 nodes $( N _ { T } ~ = ~ 4 9 )$ The final model will apply coarsening on the third block (196 → 98 nodes) and the ninth block (98 → 49 nodes).

MinCUT is applied to the graph using the existing k-NN graph and computing the soft assignment matrix S using a trained MLP. The node features X is pooled using $\mathbf { X } _ { c o a r s e } = \mathbf { X } \mathbf { S }$ . The graph convolutions are then applied to the newly created $\mathbf { X } _ { c o a r s e }$ . To train the MinCUT Pooling, the auxiliary loss $\mathcal { L } _ { u }$ is added to the cross-entropy:

$$
\mathcal { L } _ { t o t } = \mathcal { L } _ { \mathrm { C E } } + \omega _ { u } \mathcal { L } _ { u }\tag{4}
$$

The value of $\omega _ { u }$ will be discussed in the following sections. As the number of nodes changed inside the coarsened Grapher, the residual connection from the original Grapher block has been removed in the coarsened version.

## 4. Experiments

This section provides details on the setup used to perform the experiments.

Dataset: We choose to use ImageNet [36] a widely used image classification dataset, for our experiments. It contains 1.2M training images and 50K validation images, distributed across 1000 classes.

Experimental settings: Using the pre-trained models from the original paper as base, we trained the coarsened versions for 150 epochs using a batch size of 128. We kept most of the original parameters, the depth and dimension D remain the same for the 2 models. We keep a k k-NN in range 9 to 18, the graph convolution used is Max-Relative. We used a learning rate of $1 0 ^ { - 4 }$ with AdamW as an optimizer and cosine (with restarts) as learning rate scheduler.

Metrics: For all experiments, we report top-1 accuracy as the primary evaluation metric (noted accuracy in the following section). Top-1 accuracy corresponds to the percentage of samples for which the model’s highest-confidence prediction matches the ground-truth class.

The computational cost of a model is calculated using fvcore FlopCountAnalysis. This library provides a computational cost in MACs but since most works [14] used 1 $\mathrm { { F L O P s } = 1 \ M A C s }$ , we have chosen to present our results in FLOPs with the same conversion ratio instead of the classic $2 { \mathrm { F L O P s } } = 1 { \mathrm { M A C s } }$

The model latency is measured using a batch size of 1. A warm-up of 20 images is used. Then, the latency is averaged over 200 images. The latency is calculated on a single NVIDIA A100 GPU.

Ablations: We tested several positions $C _ { s t a r t }$ for the coarsening block with a fixed coarsening ratio $C _ { r }$ and changed the coarsening ratio while keeping the coarsening block on the same position. We also explored the possibility to coarsen multiple times the same model with different gaps between two coarsenings and changing the coarsening rate.

We also explore the impact of different $\omega _ { u }$ values in range 0 to 2. For all other experiments, we used $\omega _ { u } = 0 . 1$ Hardware: Our experiments have been performed on 3 different hardware settings. To train the CoViG-Ti, we used 8 NVIDIA A100 GPU. For the CoViG-s training, we used 16 NVIDIA A100 GPU on 4 nodes. Finally, we evaluated our models on a single NVIDIA A100 GPU.

## 5. Results

In this section, CoViG-Ti-X-Y model refers to a CoViG-Ti model with a coarsening starting on the X-th block and a compression ratio of Y.

<table><tr><td>Model</td><td> $C _ { s t a r t }$ </td><td> $C _ { r }$ </td><td>Params (M)</td><td>FLOPs (G)</td><td>Latency (ms)</td><td>Acc (%)</td></tr><tr><td>ViG-Ti</td><td>-</td><td>-</td><td>7.18</td><td>1.31</td><td>15.22</td><td>74.4</td></tr><tr><td rowspan="10">CoViG-Ti</td><td>6</td><td>0.5</td><td>7.21</td><td>1.03</td><td>15.19</td><td>75.5</td></tr><tr><td>3</td><td>0.5</td><td>7.21</td><td>0.88</td><td>14.64</td><td>75.0</td></tr><tr><td>6</td><td>0.25</td><td>7.21</td><td>0.89</td><td>15.18</td><td>75.1</td></tr><tr><td>5</td><td>0.25</td><td>7.21</td><td>0.82</td><td>15.23</td><td>74.9</td></tr><tr><td>4</td><td>0.25</td><td>7.21</td><td>0.74</td><td>15.08</td><td>74.9</td></tr><tr><td>3</td><td>0.25</td><td>7.21</td><td>0.67</td><td>14.54</td><td>74.5</td></tr><tr><td>2</td><td>0.25</td><td>7.21</td><td>0.60</td><td>14.87</td><td>73.8</td></tr><tr><td>1</td><td>0.25</td><td>7.21</td><td>0.53</td><td>15.34</td><td>72.7</td></tr><tr><td>0</td><td>0.25</td><td>7.21</td><td>0.45</td><td>15.31</td><td>70.3</td></tr><tr><td>3 3</td><td>0.1 0.05</td><td>7.21 7.20</td><td>0.55 0.51</td><td>14.84 14.82</td><td>73.5 72.5</td></tr></table>

Table 1. CoViG-Ti performances under different single coarsening configurations

We have first tested single coarsening applied on different blocks and with different coarsening ratios on ViG-Ti. These models are presented in Table 1. We can note that most of the tested configurations outperform or match the baseline accuracy. For example, the $C _ { s t a r t } = 6 $ and $C _ { r } = 0 . 5$ obtain a one point higher accuracy while only using 78% of the reference computational cost. $C _ { s t a r t } = 3$ and $C _ { r } = 0 . 2 5$ matches the baseline accuracy while reducing the computational cost by half. We can conclude that starting the coarsening after the third block and having a coarsening ratio over 25% gives more accurate models than the original ViG-Ti. We can also note that the number of parameters increases due to the introduction of a MinCUT Pooling block, but this only adds 0.5% parameters, which is negligible.

![](images/210ec314a0506a83f94d0f689ced8cc791d7cffa322b821c51969743ef67f95a.jpg)

![](images/839585c9870e389f3808247ffdee10232e809e69fa7748fcd8a3cde08a4521cf.jpg)

![](images/72329b34cd1d76a492adf6388d49edc6d02b76a28f55ccdebed891c262611e1e.jpg)  
(a) Block 0  
(b) Block 6  
Figure 5. Effect of coarsening depth on MinCUT assignment quality: near-hard spatial assignments at block 0 vs. soft semantic aggregation at block 6.

We changed $\omega _ { u }$ to see its impact on the CoViG accuracy. Testing values of 0.1, 1, and 2 we did not find any statistically significant difference in term of accuracy so we kept $\omega _ { u } = 0 . 1$ 1 for all our experiments.

To qualitatively assess the behavior of the MinCUT Pooling module and validate the choice of coarsening position, we visualize the soft assignment matrix $\mathbf { S } \in \mathcal { R } ^ { \bar { 1 } \bar { 9 6 } \times 4 \mathfrak { L } }$ and the adjacency matrix $\mathbf { A } \in \mathcal { R } ^ { 1 9 6 \times 1 9 6 }$ for two coarsening configurations: CoViG-0-0.25 and CoViG-6-0.25. These matrices are visualized in Figure 5. Several observations emerge from these visualizations:

First, the adjacency matrices of both configurations exhibit a clear block-diagonal structure, confirming that the MinCUT loss successfully encourages the pooling to preserve intra-cluster connectivity. Clusters are well-separated and super-nodes are connected primarily to other supernodes within the same semantic group, as theoretically prescribed by the MinCUT objective. However, the adjacency matrices exhibit an important difference: at Block 0, the connectivity is strongly concentrated around the diagonal and a few nearby bands, whereas at Block 6 the connectivity is more dispersed across the graph, with substantially more off-diagonal structure. This indicates that deeper ViG representations contain richer non-local relationships between visual nodes.

Second, the soft assignment matrices reveal a marked qualitative difference between the two configurations. At block 0, assignments are near-hard and high-contrast, with individual nodes assigning almost their entire probability mass to a single cluster with confidence values reaching 0.9. This behavior indicates that early-block features are spatially discriminative enough for the MLP to produce confident, near-deterministic cluster memberships. However, such near-hard assignments provide little benefit over hard partitioning methods and produce super-nodes that aggregate only a handful of original nodes, limiting the semantic richness of the coarsened representation. In contrast, at block 6, assignments are considerably softer and more diffuse, with probability mass distributed across multiple clusters and peak values limited to 0.35. This diffuse structure reflects the feature homogenization that occurs at deeper network layers. As node representations converge toward more uniform embeddings, the assignment MLP naturally distributes contributions across clusters, producing supernodes that are genuine weighted aggregations of semantically related patches rather than near-singleton partitions. The corresponding pooled adjacency at block 6 is denser and exhibits richer inter-cluster connectivity, indicating that the coarsened graph retains meaningful global structure for the downstream Grapher blocks to operate on.

To conclude, coarsening at Block 6 can aggregate nodes after the network has established more meaningful global relationships, whereas coarsening at Block 0 operates on a less developed representation. This qualitative observation is consistent with the quantitative results: delaying coarsening generally provides a better accuracy–efficiency trade-off than applying it at the beginning of the network.

<table><tr><td>Model</td><td> $C _ { s t a r t }$ </td><td> $C _ { r }$ </td><td> $C _ { s t e p }$ </td><td> $N _ { T }$ </td><td>Params (M)</td><td>s FLOPs Acc (G)</td><td>(%)</td></tr><tr><td>ViG-Ti</td><td>-</td><td>-</td><td>-</td><td>-</td><td>7.18</td><td>1.31</td><td>74.4</td></tr><tr><td rowspan="4">CoViG-Ti</td><td>3</td><td>0.9</td><td>1</td><td>49</td><td>7.45</td><td>1.03</td><td>7.0</td></tr><tr><td>6</td><td>0.9</td><td>1</td><td>49</td><td>7.38</td><td>1.19</td><td>32.5</td></tr><tr><td>4</td><td>0.5</td><td>4</td><td>49</td><td>7.24</td><td>0.84</td><td>69.7</td></tr><tr><td>6</td><td>0.5</td><td>3</td><td>49</td><td>7.24</td><td>0.96</td><td>71.7</td></tr></table>

Table 2. CoViG-Ti performances under different multiple coarsening configurations

Table 2 shows that applying graph coarsening multiple times is less effective than performing a single coarsening operation. While repeated coarsening further reduces the number of visual nodes and consequently the computational cost, it leads to a significant degradation in accuracy. This indicates that successive reductions of the visual graph progressively discard discriminative information. In contrast, a single coarsening stage allows the network to perform

graph reasoning at full resolution during the earlier layers before reducing the graph size, providing a better trade-off between computational efficiency and accuracy. For example the first model in Table 2 has the same computational cost as the CoViG-Ti-6-0.5 model but achieves only 7% of accuracy comparing to the 75.5% accuracy of the single coarsened model.
<table><tr><td>Model</td><td> $C _ { s t a r t }$ </td><td> $C _ { r }$ </td><td>Params (M)</td><td>FLOPs Latency (G) (ms)</td><td>Acc (%)</td></tr><tr><td>ViG-S</td><td>-</td><td>-</td><td>22.7</td><td>4.54 20.31</td><td>80.6</td></tr><tr><td rowspan="3">CoViG-S</td><td>8</td><td>0.5</td><td>22.81</td><td>3.52 19.93</td><td>78.7</td></tr><tr><td>6</td><td>0.5</td><td>22.81</td><td>3.26</td><td>20.65 78.3</td></tr><tr><td>4</td><td>0.5</td><td>22.81</td><td>3.0</td><td>20.37 77.8</td></tr></table>

Table 3. CoViG-S performances under different single coarsening configurations

The results presented in Table 3 show that the trends observed for CoViG-Ti also extend to the larger ViG-S model. Although the accuracy-efficiency trade-off is less favorable. The tested models reduces the FLOPs from 22.5 to 33.9% while decreasing accuracy by 1.9 to 2.8 accuracy points. These results suggest that graph coarsening is particularly effective for the smaller ViG-Ti model, where a substantial reduction in computational cost can be achieved with limited impact on accuracy, whereas the larger ViG-S model appears more sensitive to the reduction of visual nodes. Nevertheless, CoViG-S still provides a meaningful computational reduction.

<table><tr><td>Model</td><td>Params (M)</td><td>FLOPs (G)</td><td>Acc (%)</td></tr><tr><td>ViG-Ti [14]</td><td>7.18</td><td>1.31</td><td>74.4</td></tr><tr><td>SVIG-Ti [8]</td><td>7.2</td><td>1.3</td><td>74.6</td></tr><tr><td>ViHGNN-Ti [17]</td><td>8.2</td><td>1.8</td><td>74.3</td></tr><tr><td>Isotropic ViDR-GNN-Ti [49]</td><td>7.3</td><td>1.4</td><td>74.6</td></tr><tr><td>ViG-LRGC [9]</td><td>8.1</td><td>1.5</td><td>74.3</td></tr><tr><td>CoViG-Ti-6-0.5 (ours)</td><td>7.21</td><td>1.03</td><td>75.5</td></tr><tr><td>CoViG-Ti-3-0.25 (ours)</td><td>7.21</td><td>0.67</td><td>74.5</td></tr></table>

Table 4. Comparison of our CoViG method with different isotropic ViG-based models.

We have compared our method against other isotropic ViG based methods (see Table 4). Our CoViG-Ti-6-0.5 achieves the best accuracy among all models while requiring less computations than the other methods. CoViG-3- 0.25 is half the size of the other models while keeping the same accuracy. These results demonstrate that graph coarsening can substantially reduce the computational cost of an isotropic ViG architecture while preserving, and in the less aggressive setting even improving, its classification performances.

## 6. Conclusion

In this work, we introduce CoViG, a graph coarsening approach to reduce the computational cost of isotropic Vision GNNs while preserving their classification performances. CoViG integrates MinCUT Pooling into the ViG architecture to progressively reduce the number of visual tokens processed by subsequent graph convolutional blocks. This reduces the computational cost of graph reasoning while keeping the original depth and feature dimensionality of the ViG architecture.

Our experiments on ImageNet demonstrate that the effectiveness of coarsening strongly depends on the coarsened Grapher position in the network. Applying coarsening too early can discard spatial information before sufficiently high-level representations have been formed, whereas delaying coarsening allows the network to perform graph reasoning at full resolution before aggregating visual tokens. In particular, for ViG-Ti, CoViG can reduce the computational cost by approximately half while maintaining the baseline accuracy, and a less aggressive configuration even improves the accuracy from 74.4% to 75.5% while reducing FLOPs from 1.31G to 1.03G. These results indicate that graph coarsening can provide computational savings without necessarily requiring a proportional loss in recognition performance.

We further showed that repeatedly coarsening the graph is less effective than applying a single coarsening operation at an appropriate depth. Although multiple coarsening stages can further reduce the number of visual nodes, they can progressively discard discriminative information and result in substantial accuracy degradation. Experiments with the larger ViG-S model confirm the generality of the approach, while also showing that larger models can be more sensitive to aggressive node reduction.

Overall, these results demonstrate that visual tokens in graph-based vision architectures do not necessarily need to be processed at full resolution throughout the entire network. Carefully positioned graph coarsening provides a simple mechanism for controlling the computational cost of graph reasoning while preserving, and in some configurations improving, recognition accuracy. Future work could investigate adaptive coarsening strategies that determine the number of visual nodes dynamically based on the input or the semantic content of intermediate representations, as well as extending CoViG to other graph-based vision architectures (i.e. pyramid ViG).

## References

[1] Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? In Neural Information Processing Systems, 2013. 2

[2] Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. Mincut pooling in graph neural networks. ArXiv, abs/1907.00481, 2019. 1, 4

[3] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. ArXiv, abs/2210.09461, 2022. 2

[4] Yu Cheng, Duo Wang, Pan Zhou, and Zhang Tao. A survey of model compression and acceleration for deep neural networks. ArXiv, abs/1710.09282, 2017. 2

[5] Inderjit S. Dhillon, Yuqiang Guan, and Brian Kulis. Weighted graph cuts without eigenvectors a multilevel approach. IEEE Transactions on Pattern Analysis and Machine Intelligence, 29, 2007. 1, 3, 4

[6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ArXiv, abs/2010.11929, 2020. 2

[7] David Kristjanson Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Gomez-Bombarelli, Timo-´ thy D. Hirzel, Alan Aspuru-Guzik, and Ryan P. Adams. Con-´ volutional networks on graphs for learning molecular fingerprints. ArXiv, abs/1509.09292, 2015. 2

[8] Ismael Elsharkawi, Hossam Sharara, and Ahmed A. Rafea. Svig: A similarity-thresholded approach for vision graph neural networks. IEEE Access, 13:19379–19387, 2025. 7

[9] Ismael Elsharkawi, Hossam Sharara, and Ahmed A. Rafea. Vig-lrgc: Vision graph neural networks with learnable reparameterized graph construction. Pattern Recognit. Lett., 207: 36–41, 2025. 7

[10] Hongyang Gao and Shuiwang Ji. Graph u-nets. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44: 4948–4960, 2019. 3

[11] Yunchao Gong, L. Liu, Ming Yang, and Lubomir D. Bourdev. Compressing deep convolutional networks using vector quantization. ArXiv, abs/1412.6115, 2014. 2

[12] Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005., 2:729–734 vol. 2, 2005. 2

[13] Suyog Gupta, Ankur Agrawal, K. Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. In International Conference on Machine Learning, 2015. 2

[14] Kai Han, Yunhe Wang, Jianyuan Guo, Yehui Tang, and Enhua Wu. Vision GNN: An Image is Worth Graph of Nodes. ArXiv, abs/2206.00272, 2022. 1, 2, 3, 4, 5, 7

[15] Song Han, Jeff Pool, John Tran, and William J. Dally. Learning both weights and connections for efficient neural network. In Neural Information Processing Systems, 2015. 2

[16] Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, and Yulin Wang. Dynamic neural networks: A sur-

vey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44:7436–7456, 2021. 2

[17] Yan Han, Peihao Wang, Souvik Kundu, Ying Ding, and Zhangyang Wang. Vision hgnn: An image is more than a graph of nodes. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 19821–19831, 2023. 7

[18] Mohammad Hashemi, Shengbo Gong, Juntong Ni, Wenqi Fan, B. Aditya Prakash, and Wei Jin. A comprehensive survey on graph reduction: Sparsification, coarsening, and con densation. In International Joint Conference on Artificial Intelligence, 2024. 3

[19] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2015. 2

[20] Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. ArXiv, abs/1503.02531, 2015. 2

[21] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco An dreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. ArXiv, abs/1704.04861, 2017. 2

[22] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. Communications ofthe ACM, 60:84 – 90, 2012. 2

[23] Loic Landrieu and Martin Simonovsky. Large-scale point cloud semantic segmentation with superpoint graphs. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4558–4567, 2017. 2

[24] Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick´ Haffner. Gradient-based learning applied to document recognition. Proc. IEEE, 86:2278–2324, 1998. 2

[25] Junhyun Lee, Inyeop Lee, and Jaewoo Kang. Self-attention graph pooling. ArXiv, abs/1904.08082, 2019. 3

[26] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9992–10002, 2021. 2

[27] Zhenhua Liu, Yunhe Wang, Kai Han, Siwei Ma, and Wen Gao. Post-training quantization for vision transformer. ArXiv, abs/2106.14156, 2021. 2

[28] Alessio Micheli. Neural network for graphs: A contextual constructive approach. IEEE Transactions on Neural Networks, 20:498–511, 2009. 2

[29] Mustafa Munir, William Avery, and Radu Marculescu. Mobilevig: Graph-based sparse attention for mobile vision applications. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 2211– 2219, 2023. 1, 2

[30] Mustafa Munir, William Avery, Md Mostafijur Rahman, and Radu Marculescu. Greedyvig: Dynamic axial graph construction for efficient vision gnns. 2024 IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 6118–6127, 2024. 2

[31] Dhruv Parikh, Jacob Fein-Ashley, Tian Ye, Rajgopal Kannan, and Viktor K. Prasanna. Clustervig: Efficient globally aware vision gnns via image partitioning. 2025 International Conference on Multimedia Computing, Networking and Applications (MCNA), pages 136–143, 2025. 1, 2, 4

[32] Eunhyeok Park, Dongyoung Kim, Soobeom Kim, Yong-Deok Kim, Gunhee Kim, Sungroh Yoon, and Sungjoo Yoo. Big/little deep neural network for ultra low power inference. 2015 International Conference on Hardware/Software Codesign and System Synthesis (CODES+ISSS), pages 124–132, 2015. 2

[33] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. ArXiv, abs/2106.02034, 2021. 2

[34] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39:1137–1149, 2015. 2

[35] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. CoRR, abs/1412.6550, 2014. 2

[36] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211 – 252, 2014. 5

[37] Evan Shelhamer, Jonathan Long, and Trevor Darrell. Fully convolutional networks for semantic segmentation. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3431–3440, 2014. 2

[38] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014. 2

[39] Suraj Srinivas and R. Venkatesh Babu. Data-free parameter pruning for deep neural networks. In British Machine Vision Conference, 2015. 2

[40] Surat Teerapittayanon, Bradley McDanel, and H. T. Kung. Branchynet: Fast inference via early exiting from deep neural networks. 2016 23rd International Conference on Pattern Recognition (ICPR), pages 2464–2469, 2016. 2

[41] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J´ egou. Training´ data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, 2020. 2

[42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems, 2017. 2

[43] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 548–558, 2021. 2

[44] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (TOG), 38:1 – 12, 2018. 2

[45] Danfei Xu, Yuke Zhu, Christopher Bongsoo Choy, and Li Fei-Fei. Scene graph generation by iterative message passing. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3097–3106, 2017. 2

[46] Sijie Yan, Yuanjun Xiong, and Dahua Lin. Spatial tempo ral graph convolutional networks for skeleton-based action recognition. In AAAI Conference on Artificial Intelligence, 2018. 2

[47] Yiding Yang, Jiayan Qiu, Mingli Song, Dacheng Tao, and Xinchao Wang. Distilling knowledge from graph convolutional networks. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7072–7081, 2020. 3

[48] Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, and Jure Leskovec. Hierarchical graph representation learning with differentiable pooling. In Neural Information Processing Systems, 2018. 3

[49] Zeyang Zhang, Xiao Cao, Xiang Zhang, Li Peng, Lijia Ma, and Jielong Yang. Vidr-gnn: Vision implicit discriminative reorganization graph neural networks. IEEE Transactions on Multimedia, 28:2172–2185, 2026. 7

[50] Yiren Zhao, Duo Wang, Daniel Bates, Robert D. Mullins, Mateja Jamnik, and Pietro Lio. Learned low precision graph\` neural networks. ArXiv, abs/2009.09232, 2020. 3

[51] Hongkuan Zhou, Ajitesh Srivastava, Hanqing Zeng, Rajgopal Kannan, and Viktor K. Prasanna. Accelerating large scale real-time gnn inference using channel pruning. ArXiv, abs/2105.04528, 2021. 3