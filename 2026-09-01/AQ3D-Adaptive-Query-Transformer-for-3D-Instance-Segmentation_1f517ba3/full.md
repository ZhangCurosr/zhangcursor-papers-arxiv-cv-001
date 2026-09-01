# AQ3D: Adaptive Query Transformer for 3D Instance Segmentation

Keno Moenck<sup>\*</sup> Thorsten Schuppstuhl¨ Hamburg University of Technology

## Abstract

Transformer-based decodersfor 3D instance segmentation typically commit to afixed number ofqueries and positional modeling calibrated on the training distribution rather than on the scene at hand. Indoor scans vary widely in spatial extent and object count, so afixed query set over-initializes small scenes and under-initializes large ones, while learned absolute and relative encodings are bound to the training scenes’ extents and can saturate. We present AQ3D, which is designed to handle scenes of various sizes during training and inference. Queries are instantiated at a fixed ratio of the scene’s superpoints,forming an overcomplete set whose background rejection is entirely left to the decoder. Positional information is encoded using 3D RoPE over quantized metric coordinates, replacing learned bounded lookup tables ofprior decoders. Further, we improve the decoder itself by using attribution-based superpoint pooling, a mask refinement branch, and a cosine classifierfor background rejection. Experiments show our method sets a new stateof-the-art on validation and hidden test splits across the datasets ScanNetV2, ScanNet200, and ScanNet++V2 among decoder methods trained without additional data augmentation. Code is available at github.com/kenomo/aq3d.

## 1. Introduction

Perception in real-world environments from threedimensional sensing is a long-standing task in computer vision. Applications range from robotic manipulation [38, 47] and navigation [13, 27] to building as-built verification/as-is model generation [1, 2] and retrofit [8, 24]. Point clouds are a native representation, and 3D semantic and instance segmentation, assigning a semantic label to every point and additionally separating individual object instances, are the perception tasks at hand.

3D instance segmentation is dominated by methods using transformer-based decoders that follow the end-to-end setprediction paradigm of DETR [3] and the mask-classification formulation of Mask2Former [5]. Starting with Mask3D [29] and SPFormer [32], instance queries represent objects that are refined against the scene representation in stacked crossand self-attention layers and decoded into binary masks and class labels. Subsequent work has improved different parts: stronger and more efficient backbones [43], contextbased training data augmentation [39], better intra- and interinstance feature discrimination [22, 40, 41], and richer relational modeling in cross-attention or among queries in self-attention [17, 22, 35].

![](images/24530a873a6e08f2a79982926e64d4066fc0d63da625b48ae4e347897a4f56bb.jpg)  
Figure 1. ScanNetV2 scene scale variance. Top: small and large scenes with corresponding superpoint $N _ { S }$ and instance $N _ { I }$ count. Bottom: Per-scene instance versus superpoint count.

What has received far less attention is how the samples’ dimensions shape two coupled design choices: the query set and the modeling of positional information. Indoor scans in datasets differ in spatial extent and number of foreground objects (cf. Fig. 1), both within a dataset and across datasets (cf. Tab. 1). ScanNetV2 [6] and ScanNet200 [28] are based on the same point cloud data but differ in label taxonomy. ScanNet++V2 [42] consists of high-resolution, dense-annotated scans of varying sizes, resulting in more superpoints and greater variance across samples. Within most related methods, e.g., [17, 22, 23, 32, 35], the decoder attends to superpoints, clustering the point cloud into geometrically similar regions.

Most decoders commit to a fixed number of queries, a parameter shared across every training and inference sample, regardless of the scene content [17, 22, 29, 32]. SGI-Former [40] and LaSSM [41] do take scene context into account when selecting an initial pool of scene-derived queries. SGIFormer uses a semantic-confidence threshold, and LaSSM uses semantic-guided spatial ranking. However, both collapse this pool back to a fixed number. Only OneFormer3D [15] proposes a superpoint-dependent query count by turning every superpoint into a query and, during training, retaining a random fraction between 0.5 and 1.0. A scene-agnostic query budget can fall short in both directions. In small scenes, the decoder is over-initialized, leading to many queries competing for the same object, with only one retained by the bipartite matching. In large scenes, the decoder is under-initialized, resulting in low assignment coverage.

Table 1. Dataset splits and scene statistics. Train splits are cropped/chunked as used during training; val splits are not chunked. Values are reported as mean ± standard deviation for $\bar { N } _ { \mathcal { S } } =$ number of superpoints, ${ \bar { N } } _ { I } = \mathrm { i n s t a n c e s }$ (ScanNet200 N<sup>¯</sup><sub>I,200</sub>), and $\bar { D } _ { S } = \mathrm { { d i a g o n a l } }$ of xy-axis-aligned scenes; $N _ { S , m a x }$ is the maximum number of superpoints across all scenes.
<table><tr><td rowspan="2">Metric</td><td colspan="2">ScanNetV2 / ScanNet200</td><td colspan="2">ScanNet++V2</td></tr><tr><td>Train</td><td>Val</td><td>Train</td><td>Val</td></tr><tr><td> $\bar { N } _ { \mathcal { S } }$ </td><td> $9 1 0 \pm 4 6 4$ </td><td> $\begin{array} { c } { { 1 0 6 8 \pm 6 3 2 } } \\ { { 3 9 0 9 } } \end{array}$ </td><td> $1 3 4 3 \pm 6 0 2$ </td><td>2433±1918</td></tr><tr><td> $N _ { S , m a x }$ </td><td>2195</td><td></td><td>4127</td><td>10656</td></tr><tr><td> $\bar { N } _ { I }$ </td><td> $1 2 . 8 \pm 7 . 0$ </td><td> $1 4 . 0 \pm 8 . 1$ </td><td> $2 6 . 1 \pm 1 6 . 1$ </td><td> $5 3 . 7 \pm 4 2 . 7$ </td></tr><tr><td> $\underset { - } { \bar { N } } _ { I , 2 0 0 }$ </td><td> $2 4 . 1 \pm 1 3 . 2$ </td><td> $2 5 . 5 \pm 1 4 . 3$ </td><td></td><td></td></tr><tr><td> $\bar { D } _ { S } \ [ \mathrm { m } ]$ </td><td> $7 . 4 \pm 2 . 0$ </td><td> $7 . 6 \pm 2 . 0$ </td><td> $7 . 1 \pm 1 . 1$ </td><td> $8 . 1 \pm 3 . 3$ </td></tr></table>

The query budget is compounded by how queries obtain their location. A parametric query learns where and what to look for from the training distribution. MAFT [17] makes this explicit by storing position queries as learnable coordinates in a normalized $[ 0 , 1 ] ^ { 3 }$ cube. Such normalization ties the encoding to the scene bounding box rather than to metric space. Relative position modeling in current approaches inherits a similar defect. MAFT [17] introduced it as a learned context-dependent lookup table in cross-attention indexed by quantized relative offsets, whose quantization step and table length are calibrated on the training scene statistics and are, by construction, saturated by offsets that exceed them. CompetitorFormer [35] adapts the same mechanism in self-attention. In short, existing methods condition the geometry they reason about on the scale of the training scenes, neglecting the variance in sample scale. All but [15] fix the number of queries independently of the input scene.

In this work, we propose the Adaptive Query Transformer for 3D Instance Segmentation (AQ3D) that circumvents scene-dimension ambiguity in query initialization and positional modeling, where these are functions of the input scene. Within a dataset, the number of foreground objects $N _ { I }$ grows with the number of superpoints $N _ { S }$ (cf. Fig.

1; per-scene plots for all three datasets in the supplement). This rules out a global parametric, learnable query set, whose cardinality is fixed at training time [32] and cannot track $N _ { S }$ Instead, queries must emerge from the scene. Rather than learning which superpoints deserve a query, we instantiate an $N _ { S }$ -relative (adaptive) but overcomplete set and leave background rejection to the decoder layers, requiring no auxiliary supervision for query proposal, unlike [40, 41]. In contrast to OneFormer3D [15], which subsamples only during training and decodes the full superpoint set at inference, we apply the same ratio at both. On ScanNetV2, the two are on par (cf. Tab. 5), but at $r = 0 . 6$ our query set is 40% smaller at inference, where the cost of self-attention grows quadratically with $N _ { \mathcal { Q } }$ . Further, an adaptive query set is only useful if the decoder’s spatial reasoning is scale-consistent. We therefore remove the absolute and learned (contextual) relative position encodings of prior decoders and replace them with a 3D extension of Rotary Position Embedding (RoPE) [31] following the protocol as developed for 3D backbones [43, 44]. Our contributions are as follows:

• We identify scene-dimension ambiguity as a systematic weakness of query-based 3D instance segmentation decoders and show that it manifests in the scene-agnostic query budget and the learned positional encodings.

• We show that the query count should be a function of the scene, and that decoder capacity should be allocated in proportion to scene complexity. A ratio well below one suffices, keeping the set smaller during inference than a full superpoint budget would.

• We show that the learned absolute and relative position encodings used by existing decoders can be replaced with metrically consistent, parameter-free 3D RoPE across all attention modules.

## 2. Related Works

## 2.1. 3D Instance Segmentation

Recent 3D instance segmentation methods have increasingly moved from proposal-based/coarse-to-fine (top-down) [12, 16, 30], grouping-based (bottom-up) [4, 14, 18, 34], and convolution-based [26, 37] paradigms toward transformerbased decoders [15, 17, 22, 23, 29, 32, 35, 40, 41] that represent each object as an instance query, following DETR’s endto-end set-prediction paradigm [3] and Mask2Former’s [5] mask and class prediction approach, removing, e.g., handtuned voting or clustering. Initially, two concurrent works transfer the Mask2Former architecture to the 3D domain. In Mask3D [29], instance queries iteratively attend to multiscale point features through stacked transformer decoder layers. In contrast, SPFormer [32] pools point features into superpoints, enabling the decoder to attend to a single level of scene representation. Reducing the scene to a few hundred superpoint tokens substantially lowers the cost of crossattention and permits a lighter backbone than Mask3D’s, establishing the efficient backbone-plus-decoder template adopted by most subsequent work.

Building on SPFormer, several works address the decoder’s observed weaknesses. For example, One-Former3D [15] passes semantic queries alongside instance queries through a shared decoder, thereby solving semantic, instance, and panoptic segmentation with a single set of weights. MAFT [17] attributes the slow convergence of the mask-attention scheme to the low recall of the initial instance masks, and consequently replaces mask attention with an auxiliary center-regression task. CompetitorFormer [35] proposes competition-oriented designs that mitigate interquery competition by amplifying the score disparity between queries and accelerating the emergence of a dominant one.

## 2.2. Query Initializing

DETR [3] represents each candidate object by a learnable query embedding, randomly initialized and shared across all images. Subsequent work made queries explicit and data-dependent. Deformable DETR [46] introduces a twostage variant in which region proposals predicted by the encoder are selected as queries, grounding them in the actual image content. DAB-DETR [19] reinterprets each query as an anchor box that is refined layer by layer, giving its positional part an explicit geometric meaning. DINO [45] adds mixed query selection, where the positional part of the query is selected from encoder features while the content part remains learnable. Evolving from DETR, a query must carry content and position information.

Methods of 3D instance segmentation have followed this evolution, especially since the 3D sparsity and irregularity of point clouds in datasets like ScanNet have made query initialization and downstream improvements necessary. SP-Former [32] relies entirely on parametric, learnable queries, whereas Mask3D [29] samples the position part from the scene at a fixed FPS and initializes the queries itself nonparametrically with zeros. In contrast, MAFT [17] directly learns the position part in the form of 3D coordinates paired with zero-initialized content queries.

Rather than learning queries’ position or content parts, they can emerge from the scene representation, e.g., [15]. Other works condition queries based on semantics [40, 41]. With the exception of [15], prior work fixes the query count independently of the input scene and innovates on the content and position parts instead, leaving the count scene-agnostic even when the initialization is scene-derived. OneFormer3D [15] does tie the count to $N _ { S }$ , but only at inference. The ratio is a training-time augmentation and is not applied at test time, so the decoder always processes the full superpoint set.

## 2.3. Positional Modeling

Attention is permutation-invariant, meaning positional information must be explicitly injected into the queries and keys. The vanilla Transformer [33] and DETR [3] add fixed sinusoidal absolute position encodings to the queries and keys, whereas the vanilla Vision Transformer (ViT) [7] learns them. Beyond absolute position encoding or embeddings, Swin [20] adds a learned relative position bias to the attention logits. Rather than adding an encoding to the token sequence, Rotary Position Embedding (RoPE) [31] rotates queries and keys as a function of their absolute positions. RoPE requires no learned parameters, supports variable sequence lengths, and, under its standard frequency schedule, induces a decay of inter-token dependency with increasing relative distance. It has since been extended to vision [11].

Transformer-based approaches for 3D tasks have also explored different ways to inject positional information. PointTransformerV3 [36] orders tokens for attention through space-filling curves and adds learned conditional positional encoding. LitePT [44] replaces the heavy conditional positional encoding in PTv3 with PointROPE, a parameter-free 3D RoPE variant. Volt [43] injects positional information exclusively through a 3D extension of RoPE, developed concurrently with LitePT’s PointROPE. Within 3D instance segmentation, SPFormer does not use extra positional encoding at all. Mask3D uses Fourier positional encodings of voxel coordinates, added to queries and keys. MAFT [17] introduces two complementary position-aware designs. For absolute position modeling, it pairs each content query with a learnable position query stored in a scene-normalized cube so that the encoded location is expressed relative to the scene bounding box. For relative position modeling, MAFT [17] quantizes the offsets between position queries and superpoints into discrete bins and indexes a learned encoding table that content-dependently reweights the cross-attention. Relation3D [22] further introduces relative positional modeling between queries, improving inter-query interactions in the self-attention mechanism. Prior instance decoders rely on learned, short-horizon formulas for positional modeling, which can saturate or lose resolution in different-scale scenes.

## 3. Method

Figure 2 shows an overview of the overall model architecture. Given an input point cloud $\mathcal { P } \in \mathbb { R } ^ { N \times 3 }$ with N points, assigned color $\bar { \mathcal { F } _ { r g b } } \in \mathbb { R } ^ { N \times 3 }$ , and normal values $\mathbf { \bar { \mathcal { F } } } _ { n } \in \mathbb { R } ^ { N \times \mathbf { \breve { 3 } } }$ , the 3D instance segmentation task is to predict binary masks $\hat { M } \in \left\{ 0 , 1 \right\} ^ { K \times \mathbb { N } }$ and corresponding class labels $\hat { L } \in \mathcal { C } ^ { K }$ for the K foreground objects in the scene, where ${ \mathcal { C } } = \{ 1 , \ldots , C \}$ is the set of semantic classes. Following previous transformer-based decoder methods [17, 32], we split the task into feature extraction and parallel decoding of the instance predictions through stacked decoder layers. The backbone extracts point features $\mathcal { F } _ { \mathcal { P } } \in \mathbb { R } ^ { N \times d _ { B } }$ , which we pool into superpoint features $\mathcal { F } _ { p o o l } \in \mathbb { R } ^ { N _ { S } \times d _ { B } }$ using a set of precomputed superpoints $\bar { \pmb { S } } = \{ S _ { 1 } , \ldots , S _ { N _ { S } } \}$ that partitions the point indices.

![](images/95dd37165721d6a48910e9cb433243ee598d0294953faf28db3a0e8d5252c981.jpg)  
Figure 2. Overview of AQ3D. Top: the backbone maps the input point cloud P with colors $\mathcal { F } _ { r g b }$ and normals ${ \mathcal { F } } _ { n }$ to point features, which are pooled into superpoint features $\mathcal { F } _ { S }$ and mask features $\hat { \mathcal { M } _ { S } } ^ { ( 0 ) }$ . The query set ${ \mathcal Q } ^ { ( 0 ) }$ is FPS-instantiated per scene with $N _ { \mathscr { Q } } = \lfloor r \cdot N _ { \mathscr { S } } \rfloor$ L decoder layers refine $\mathcal { Q } ^ { ( l ) } , \mathcal { M } _ { S } ^ { ( l ) }$ , and $\mathcal { P } _ { \mathcal { Q } ^ { ( l ) } }$ . Bottom: one decoder layer, consisting of self-attention, cross-attention to $\mathcal { F } _ { S }$ , and an FFN. An MLP $\phi _ { \triangle }$ predicts a per-query offset that updates $\mathcal { P } _ { \mathcal { Q } ^ { ( l ) } }$ . Every $\rho = 2$ layers, the mask refinement branch updates $\boldsymbol { \mathcal { M } } _ { \mathcal { S } } ^ { ( l ) }$ by cross-attending to the queries. Positional information enters through 3D RoPE on the query and key projections.

The decoder layers then iteratively refine a set of queries $Q ^ { ( l ) } \in \mathbb { R } ^ { N _ { Q } \times d _ { M } }$ and the mask features $\mathcal { M } _ { \mathcal { S } } ^ { ( l ) } \in \mathbb { R } ^ { \dot { N } _ { \mathcal { S } } \times d _ { M } }$ over L layers. Queries attend to the sample’s superpoint features $\dot { \mathcal { F } } _ { S } = \phi _ { S } ( \mathcal { F } _ { p o o l } ) \in \mathbb { R } ^ { N _ { S } \times d _ { M } }$ . Similar, $\dot { \mathcal { M } } _ { \mathcal { S } } ^ { \ ( 0 ) } =$ $\phi _ { \mathcal { M } } ( \mathcal { F } _ { p o o l } )$ , where $\phi _ { \mathcal { S } } , \phi _ { \mathcal { M } }$ are small MLPs. $d _ { M }$ and $d _ { B }$ are the decoder model and bottleneck dimensions, respectively. A class head $H _ { c l s } : \mathbb { R } ^ { d _ { M } }  \mathbb { R } ^ { C + 1 }$ maps each refined query to logits over ${ \mathcal { C } } \cup \{ \emptyset \}$ , i.e., the C semantic classes and an additional non-object label. Rather than a linear projection, we use a cosine classifier with a set of learned class prototypes $W \in \mathbb { R } ^ { ( C + 1 ) \times d _ { M } }$

$$
H _ { c l s } \big ( \mathcal { Q } _ { q } ^ { ( l ) } \big ) = s \cdot \frac { \mathcal { Q } _ { q } ^ { ( l ) } } { \Vert \mathcal { Q } _ { q } ^ { ( l ) } \Vert _ { 2 } } \left( \frac { W } { \Vert W \Vert _ { 2 } } \right) ^ { \top } .\tag{1}
$$

Mask scores are obtained as the dot product between the queries $\mathcal { Q } ^ { ( l ) }$ and the mask features $\dot { \mathcal { M } } _ { \mathcal { S } } ( l )$ , which we finally lift to point resolution. In addition, a score head $H _ { s c o r e } : \mathbb { R } ^ { d _ { M } }  [ 0 , 1 ]$ predicts the IoU between a query and its matched ground-truth mask. The final predictions are the top-k scored ones from all foreground class-mask combinations.

Backbone and Pooling We follow previous methods and use a sparse U-Net [10] and a transformer-based backbone

Volt [43] for feature $\mathcal { F } _ { \mathcal { P } }$ extraction. Volt is a “vanilla Transformer” for 3D using cubic patches of voxels as tokens, full global self-attention, and 3D RoPE. Common choices for pooling are mean [17, 32] and adaptive (softattention guided) pooling [22]. We follow the adaptive pooling from [22] but simplify it to increase computational efficiency. A small $\mathbf { M L P } \phi _ { a }$ predicts a scalar attribution score $a _ { i } = \overset { \cdot } { \phi _ { a } } ( \mathcal { F } _ { \mathcal { P } } ^ { \ ( i ) } )$ for every point i. Superpoint features are then the attribution-weighted sum of their constituent point features:

$$
\mathcal { F } _ { p o o l } ^ { ( j ) } = \sum _ { i \in \mathcal { S } _ { j } } \operatorname * { s o f t m a x } _ { S _ { j } } ( a _ { i } ) \mathcal { F } _ { \mathcal { P } } { } ^ { ( i ) } , \quad j = 1 , \ldots , N _ { \mathcal { S } } .\tag{2}
$$

In contrast to [22], the scores depend only on the point’s own features, so no pairwise interaction within a superpoint is required.

Adaptive Queries We scale the number of queries with the number of superpoints $N _ { S }$ of the given scene, using a fixed ratio $r , \mathrm { i . e . , } N _ { \mathscr { Q } } = \lfloor r \cdot N _ { \mathscr { S } } \rfloor$ obtained by FPS on the superpoint centroids, so that the initial queries $\boldsymbol { \mathcal { Q } } ^ { ( 0 ) }$ are spread across the scene. However, dense superpoint regions will have a lower seed density, which is why we use a higher $r ~ = ~ 0 . 6$ with the idea that more than half of the superpoints are seeded. Our experiments show that performance degrades below and above $r ~ = ~ 0 . 6$ on ScanNetV2. The set is overcomplete rather than roughly matched to, e.g., a predicted instance count. The corresponding query content features are initialized by projection $\phi _ { \mathcal { Q } } : \mathbb { R } ^ { d _ { B } }  \mathbb { R } ^ { d _ { M } }$ from the pooled superpoint features ${ \mathcal { F } } _ { p o o l }$ , rather than being learned or set to zero, which grounds every query in an actual, local region of the scene. Since $N _ { S }$ differs across samples, we pad each batch to its maximum $N _ { \mathcal { Q } }$ and mask padded queries in all attention blocks. Padded queries are likewise excluded from the bipartite matching and from the final top-k selection.

Decoder Each of the L decoder layers consists of selfattention among the queries, a cross-attention between queries and superpoint features $\mathcal { F } _ { \mathcal { S } }$ , and an FFN. Unlike [32], which places cross-attention first because its queries are parametric, we retain the standard self-attention-first order, since our queries already carry scene content. Following MAFT [17], we adopt the auxiliary center regression task, in which an ML ${ \bf P } \phi _ { \triangle }$ predicts a positional offset per query and updates its position $P _ { \mathcal { Q } ^ { ( l ) } }$ accordingly. As in [22, 40], we additionally introduce a reversed cross-attention block; however, instead of refining the superpoint features $\mathcal { F } _ { \mathcal { S } }$ , we only refine the mask features $\mathcal { M } _ { \mathcal { S } } ^ { ( l ) }$ every $\rho = 2$ layers by cross-attending to the queries $\mathcal { Q } ^ { ( l ) }$ . We empirically find that refining only the mask features $\mathcal { M } _ { \mathcal { S } }$ rather than refining the superpoint features $\mathcal { F } _ { \mathcal { S } }$ and omitting supervision, e.g., contrastive loss [22], does not degrade performance.

Positional Modeling We initially adopted the absolute positional modeling of MAFT [17], where superpoint and query positions are normalized to the unit cube per sample. This normalization is at odds with our scene-scale adaptivity. We therefore rely on relative positional modeling within all attention operations. Following [43, 44], we apply a 3D RoPE to the query and key projections, split across the x, y, and z axes, which injects the relative offset between two tokens directly into the dot-product attention. Position indices are obtained by quantizing metric coordinates on a fixed grid of size $\delta ,$ anchored at the minimum corner of the sample. Consistent with this reasoning, the absolute encoding yields only a small improvement on ScanNetV2 (cf. Sec. 4.4), whose scenes are uniform single rooms and contain low-density instance occurrences.

Matching Since every query is initialized from a superpoint, and every superpoint belongs to at most one groundtruth instance, the correspondence between queries and instances is determined at initialization. We therefore adopt the disentangled matching of [15].

Loss The overall objective is a weighted sum applied after every decoder layer $l = 1 , \ldots , L$ for deep supervision:

$$
\begin{array} { r l } & { \mathcal { L } = \displaystyle \sum _ { l = 1 } ^ { L } \big ( \lambda _ { c l s } \mathcal { L } _ { c l s } ^ { ( l ) } + \lambda _ { b c e } \mathcal { L } _ { b c e } ^ { ( l ) } + \lambda _ { d i c e } \mathcal { L } _ { d i c e } ^ { ( l ) } } \\ & { ~ + \lambda _ { s } \mathcal { L } _ { s } ^ { ( l ) } + \lambda _ { c } \mathcal { L } _ { c } ^ { ( l ) } \big ) , } \end{array}\tag{3}
$$

where $\mathcal { L } _ { c l s }$ is cross-entropy, with unmatched queries supervised towards the non-object label at weight $\lambda _ { \mathcal { O } } ; \mathcal { L } _ { b c e }$ and $\mathcal { L } _ { d i c e }$ are the binary cross-entropy and dice loss between the predicted and ground-truth superpoint masks of a matched pair; $\mathcal { L } _ { s }$ is the binary cross-entropy loss between the predicted score and the IoU of the query’s mask with its matched ground truth; and $\mathcal { L } _ { c }$ is the $\ell _ { 1 }$ distance between the query position ${ \mathcal { P } } _ { \mathcal { Q } ^ { ( l ) } }$ and the matched instance centroid in metric coordinates. All terms except $\mathcal { L } _ { c l s }$ are computed on matched pairs only. Padded queries are excluded throughout.

## 4. Experiments

## 4.1. Experimental Setting

Datasets and Metrics We evaluate our method on three common indoor benchmarks: ScanNetV2 [6], Scan-Net200 [28], and ScanNet++V2 [42]. ScanNetV2 consists of 1,613 richly annotated RGB-D reconstructions of indoor environments, split into 1,201 training, 312 validation, and 100 hidden-test scans. ScanNet200 reuses the same reconstructions as ScanNetV2 but replaces its coarse label set (20 classes) with a substantially finer taxonomy of 200 categories. ScanNet++V2 comprises 856 training, 50 validation, and 50 test scans, offering sub-millimeter-resolution geometry and densely labeled scenes drawn from an instance vocabulary of 84 classes. Because its scenes are far larger and denser, each training scene is commonly partitioned into 6m × 6m chunks with a 3m × 3m stride. For both ScanNetV2 and ScanNet200, we generate superpoints with the standard graph-based segmentator [9] under its defaul configuration (cutoff value $k _ { t } = 0 . 0 1$ and minimum number of vertices $n _ { v } = 2 0 )$ . For ScanNet++V2, we generate superpoints with $k _ { t } = 0 . 2$ and $n _ { v } = 1 0 0$ to account for the higher point density. Following the standard 3D instance segmentation protocol, we report mean Average Precision (mAP) and AP@50. mAP is averaged over IoU threshold from 50% (AP@50) to 95% (AP@95) in 5% steps.

Implementation Details The smallest GPU we used for experiments are a 4090 for ScanNetV2/ScanNet200 and a Pro 6000 for ScanNet++V2. All models are trained under a single, identical seed. We use neither test-time augmentation nor out-of-context or context-based mixing, as in [25, 39]. We use a batch size of 4 and train for 512 and 384 epochs when using the sparse U-Net and Volt [43], respectively. We use AdamW [21] with a learning rate of $2 \times 1 0 ^ { - 4 }$ , weight decay of 0.05, and polynomial learning rate scheduler. The decoder comprises $L = 6$ layers. For ScanNetV2, we retain MAFT’s absolute positional modeling (cf. Tab. 8), whereas for ScanNet200 and ScanNet++V2 we rely solely on RoPE. For the backbones, we quantize coordinates on a 0.02 m grid. For RoPE, we quantize coordinates on a 0.05 m grid and use a base frequency of $\theta = 1 0 0 ^ { \circ }$ . As in [15, 39, 43], we apply NMS during inference. The loss weights of Eq. (3) are $\lambda _ { c l s } = 0 . 5 , \lambda _ { b c e } = 1 . 0 , \lambda _ { s } = 0 . 1$ , and $\lambda _ { c } = 1 . 0 ; \lambda _ { d i c e }$ increases over the decoder layers from 1.0 to 4.0, following previous works that do not batch-normalize the dice loss in the last layer. With the sparse U-Net, we use $d _ { B } = 6 4$ and $d _ { M } = 2 5 6$ with 8 attention heads. This does not admit a symmetric three-way split for the 3D RoPE, so we partition the per-head channels asymmetrically as $1 2 / 1 2 / 8$ for the x, y, and z axes. With Volt [43] as the backbone, we use $d _ { B } =$ 128 and $d _ { M } = 3 8 4$ , keeping 8 attention heads, yielding a symmetric 16/16/16 RoPE split. The other methods using Volt-B, as reported in Tab. 3 and Tab. 4, use a larger Voltdecoder $( d _ { B } = 2 5 6 )$ , which is why our variant has 6M fewer parameters.

## 4.2. Instance Segmentation

ScanNetV2 Table 2 compares AQ3D against common and benchmark-leading methods. With the SPFormer-scale backbone (11M parameters), AQ3D improves over the strongest model of identical backbone capacity, Competitor-Former [35]. Scaling the sparse backbone to 44M parameters results in +2.2 mAP over the best previously reported validation result. The mAP drop using the 11M backbone is smaller than the margin by which it outperforms all other 11M methods in Tab. 2, indicating that the improvement lies in the decoder rather than in the backbone capacity. On the hidden test split, AQ3D ranks first in both metrics among all listed methods. It surpasses SPFormer + Volt-B as a backbone [43] while using half its parameters (44M vs. 88M).

ScanNet200 Table 3 reports results on ScanNet’s finer class taxonomy. With the 44M sparse backbone, AQ3D improves over the strongest sparse-backbone competitor, CompetitorFormer [35], and matches SPFormer + Volt-B [43] with ACGP [39] on mAP while using half the parameters (44M vs. 88M) and neither context-based augmentation nor test-time augmentation. Replacing the sparse backbone with Volt-B [43] adds further gains, yielding the best reported validation results. We note that [39] is an unpublished work, with only the code available.

ScanNet++V2 Table 4 reports results on ScanNet++V2. Within the sparse backbone variants, AQ3D improves over CompetitorFormer [35] (+2.8 mAP) and LaSSM [41] (+7.8 mAP). Replacing the sparse backbone with Volt-B [43] yields the best reported validation result. On the hidden test split, AQ3D is ahead of SPFormer + Volt-B [43] by +2.0 mAP and +0.8 AP@50, while remaining within 0.8 mAP of the ACGP-augmented variant, which additionally relies on context-based training data augmentation, which is orthogonal to our contribution.

Table 2. ScanNetV2 validation and hidden test set results (<sup>†</sup>denotes that we recreated the method; top-3 reported results per column are highlighted as first , second , and third ; dash (-) denotes no, incomplete, or ambiguous available information; hidden test set results are from August 26, 2026).
<table><tr><td>Method</td><td>Backbone size</td><td>With  ${ \mathcal { F } } _ { n }$ </td><td>Validation mAP AP@50</td><td></td><td>Test mAP AP@50</td></tr><tr><td colspan="6">Non transformer-based</td></tr><tr><td>PointGroup [14]</td><td>1</td><td>1</td><td>34.8</td><td>56.9</td><td>40.7 63.6</td></tr><tr><td>SSTNet [18]</td><td>1</td><td>1</td><td>49.4</td><td>64.3</td><td>50.6 69.8</td></tr><tr><td>ISBNet [26]</td><td>1</td><td>/</td><td>54.5</td><td>73.1</td><td>55.9 75.7</td></tr><tr><td>SphericalMask [30]</td><td>1</td><td>1</td><td>62.3</td><td>79.9</td><td>61.6 81.2</td></tr><tr><td colspan="6">Transformer-based decoder</td></tr><tr><td>Mask3D [29]</td><td>38M</td><td></td><td>55.2</td><td>73.7</td><td>56.6 78.0</td></tr><tr><td>SPFormer [32]</td><td>11M</td><td></td><td>56.3</td><td>73.9</td><td>54.9 77.0</td></tr><tr><td>QueryFormer [23]</td><td>38M</td><td></td><td>56.5</td><td>74.2 58.3</td><td>78.7</td></tr><tr><td>MAFT [17]</td><td>11M</td><td></td><td>58.4</td><td>75.9 59.6</td><td>78.6</td></tr><tr><td>SGIFormer [40]</td><td>11M</td><td></td><td>58.9</td><td>78.4 58.6</td><td>79.9</td></tr><tr><td>LaSSM [41]</td><td>11M</td><td></td><td>58.4</td><td>78.1 57.9</td><td></td></tr><tr><td>SPFormer† [32]</td><td>11M</td><td></td><td>59.1 77.4</td><td></td><td></td></tr><tr><td>OneFormer3D [15]</td><td>11M</td><td></td><td>59.3 78.1</td><td>56.6</td><td>80.1</td></tr><tr><td>MAFT [17]</td><td>11M</td><td></td><td>59.9 76.5</td><td></td><td></td></tr><tr><td>Relation3D [22]</td><td>11M</td><td></td><td>62.5 80.2</td><td>62.2</td><td>81.6</td></tr><tr><td>CompetitorFormer [35]</td><td>11M</td><td></td><td>63.4</td><td>81.6 62.9</td><td>81.1</td></tr><tr><td>SPFormer [32] + Volt-B [43]</td><td>88M</td><td></td><td>78.3</td><td>64.0</td><td>82.7</td></tr><tr><td>AQ3D (Ours)</td><td>11M</td><td></td><td>64.8 81.6</td><td></td><td></td></tr><tr><td>AQ3D (Ours)</td><td>44M</td><td></td><td>65.6</td><td>82.1</td><td>65.6 83.4</td></tr></table>

Table 3. ScanNet200 validation and hidden test set results (Notation as in Tab. 2; we list only point cloud-based methods; approaches additionally using posed RGB-D frames are excluded; hidden test set results are from August 26, 2026).
<table><tr><td rowspan=1 colspan=9>BackboneWith Validation     TestMethodsize    ${ \mathcal { F } } _ { n }$ mAP AP@50mAP AP@50</td></tr><tr><td rowspan=10 colspan=3>SPFormer [32] results from [17]Mask3D [29]SPFormer [32]†MAFT [17]SGIFormer [40]LaSSM [41]Relation3D [22]CompetitorFormer [35]SPFormer [32] + Volt-B [43]+ ACGP [39]</td><td rowspan=1 colspan=1>11M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>25.2 33.8</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>27.4 37.0</td><td rowspan=6 colspan=2>27.8 38.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>11M</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=2>28.4 37.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>11M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>29.2 38.2</td></tr><tr><td rowspan=1 colspan=1>38M</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=2>39.429.3 39.2</td></tr><tr><td rowspan=1 colspan=1>38M</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>11M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>31.6 41.2</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>11M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>34.1 44.1</td><td rowspan=1 colspan=1>32.8</td><td rowspan=1 colspan=1>41.5</td></tr><tr><td rowspan=1 colspan=1>88M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>48.4</td><td rowspan=1 colspan=1>36.7</td><td rowspan=1 colspan=1>47.5</td></tr><tr><td rowspan=1 colspan=1>88M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>39.6 50.2</td><td rowspan=1 colspan=1>38.1</td><td rowspan=1 colspan=1>49.4</td></tr><tr><td rowspan=2 colspan=3>AQ3D (Ours)AQ3D (Ours) + Volt-B [43]</td><td rowspan=1 colspan=1>44M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>39.6 49.7</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>88M</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>44.1 55.1</td><td rowspan=1 colspan=1>38.5</td><td rowspan=1 colspan=1>49.1</td></tr></table>

## 4.3. Scene-Adaptive Queries

Query Budget What query ratio saturates the model’s disentangling and rejection capabilities? Fig. 3 shows mAP,

Table 4. ScanNet++V2 validation and hidden test set results (Notation as in Tab. 2; hidden test set results are from August 26, 2026).
<table><tr><td>Method</td><td>Backbone size</td><td>With  ${ \mathcal { F } } _ { n }$ </td><td>Validation mAP AP@50</td><td></td><td>Test mAP AP@50</td></tr><tr><td>OneFormer3D [15] MAFT [17]†</td><td>11M</td><td></td><td>26.0 39.2</td><td>28.2</td><td>43.3</td></tr><tr><td>SGIFormer [40] resuls from [42]</td><td></td><td></td><td>27.7</td><td>29.9</td><td>45.7</td></tr><tr><td>LaSSM [41]</td><td>11M</td><td></td><td>42.1 29.1</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>43.5</td><td>32.4</td><td>48.0</td></tr><tr><td>CompetitorFormer [35]</td><td>11M</td><td>√</td><td>34.1 48.5</td><td>33.5</td><td>48.0</td></tr><tr><td>SPFormer [32] + Volt-B [43]</td><td>94M</td><td>√</td><td></td><td>36.0</td><td>54.9</td></tr><tr><td>+ ACGP [39]</td><td>94M</td><td>√</td><td>37.2 56.3</td><td>38.8</td><td>56.3</td></tr><tr><td>AQ3D (Ours)</td><td>44M</td><td>√</td><td>36.9</td><td>53.2</td><td></td></tr><tr><td> $\mathrm { A Q 3 D ( O u r s ) } + \mathrm { V o l t } \mathrm { - } \mathrm { B } \ [ 4 3 ]$ </td><td>88M</td><td>」</td><td>38.3</td><td>57.1</td><td>38.0 55.7</td></tr></table>

AP@50, and mRC (mean Recall) on the ScanNetV2 validation split. The sweep shows saturation for $r > 0 . 6 .$ , except for mRC, which continues to rise. AP and RC diverge. Recall only requires that some query covers an instance, so a denser initialization keeps discovering objects that a sparser one misses, but the additional coverage comes at the cost of harder query discrimination. Below $r = 0 . 4 .$ , mRC and mAP fall together, i.e., instances are lost that no amount of decoding can recover. On the ScanNetV2 training split, $r = 0 . 6$ corresponds to 657 k total queries per training epoch. Related approaches [17, 22, 32] use $N _ { \mathcal { Q } } = 4 0 0$ per sample, which totals 480 k queries per epoch. Is the gain then simply a matter of a larger budget? $\mathbf { A t } \ r = 0 . 4$ (438 k, below the fixed-budget total), AQ3D achieves 64.1 mAP and outperforms the strongest fixed-query budget competitor. Further, Tab. 5 compares the adaptive initialization strategy with using a fixed $N _ { \mathcal { Q } }$ and feature-based or zero initialization of queries. On ScanNet200, the commonly used $N _ { \mathcal { Q } } = 8 0 0$ requires a third more total queries than our adaptive approach. On ScanNet++V2, $r = 0 . 6$ is on average on par with using $N _ { \mathcal { Q } } = 8 0 0$ . However, across all three datasets, fixing $N _ { \mathcal { Q } }$ degrades performance. On ScanNetV2, a fixed $N _ { \mathcal { Q } } = 4 0 0$ sits just below the average of using $r = 0 . 4$ (roughly the same total query budget). So, using adaptive queries only slightly increases performance on ScanNetV2, while having a much larger impact on ScanNet200, even though it uses fewer total queries. Zero-initialization is indistinguishable from feature-based initialization on ScanNetV2 but incurs an additional −1.6 mAP on ScanNet200. The last row of Tab. 5 replaces our fixed ratio with the OneFormer3D-like scheme [15], which subsamples only during training and decodes every superpoint at inference. It matches our result on ScanNetV2 (−0.1 mAP), so performance comes from tying $N _ { Q }$ to $N _ { S }$ rather than from the particular sampler, while $r = 0 . 6$ reaches it with 40% fewer queries at inference.

Scene Scale The consequence of the adaptive query budget is that the decoder no longer inherits the training scene’s scales. We can measure it on the ScanNet++V2 dataset. Training scenes have an average xy-axis-aligned scene diagonal of $7 . 1 \pm 1 . 1$ [m] when following the standard procedure of chunking and data augmentation. When we do not chunk the validation split, the average xy-axis-aligned scene diagonal is $8 . 1 \pm 3 . 3$ [m]. Scenes are much more varied in scale. When evaluating MAFT and AQ3D on full-scale scenes, MAFT loses ≈ 5 points on mAP and AP@50, whereas AQ3D only loses ≈ 1 mAP/AP@50. Although 800 queries should be sufficient even for the largest validation scenes, the learned query position distribution is calibrated to the training scene chunks. Further, MAFT fails due to learned positional modeling, which only scales to the scenes seen during training. The queries’ positional parts lose resolution, while relative positional modeling lacks a sufficiently large horizon in large scenes.

![](images/e3167db5b16a9c5b1cb7713ce16b5004c3b0b497fd7e238b0ebabd36bad09de9.jpg)  
Figure 3. Query ratio sweep (r) results on ScanNetV2 validation split, reporting mAP, AP@50, and mRC (mean Recall, derived over the same IoU bins as mAP).

Table 5. Query initialization strategies; values are differences w.r.t. the baselines. We evaluate ScanNet++V2 on the non-chunked validation split. Query budget is Ada. = adaptive, Fix. = fixed with $N _ { \mathcal { Q } } = 4 0 0$ (ScanNetV2), $N _ { \mathcal { Q } } = 8 0 0$ (ScanNet200), and $N _ { \mathcal { Q } } = 8 0 0 \ ( \mathrm { S c a n N e t } { + + } \mathrm { V } 2 )$ . Query positions are FPS-sampled. Query content is Fea. = feature-initialized or Zer. = zero-initialized. Last row is OneFormer3D-like [15] query initialization: random $r \in [ 0 . 5 , 1 ]$ with $r = 1 . 0$ at inference.
<table><tr><td>Query initialization Ada. Fix. Fea. Zer.</td><td></td><td></td><td></td><td>ScanNetV2 mAP AP@50</td><td></td><td>ScanNet200 mAP AP@50</td><td></td><td>ScanNet++V2 mAP AP@50</td></tr><tr><td>√</td><td></td><td></td><td></td><td>65.6</td><td>82.1</td><td>39.6</td><td>49.7</td><td>36.1</td></tr><tr><td></td><td></td><td>√</td><td>一</td><td>-1.6</td><td>-1.4</td><td>-1.9</td><td>-2.5</td><td>-5.2 -6.7</td></tr><tr><td></td><td> $\checkmark$ </td><td></td><td> $\checkmark$ </td><td>-1.7</td><td>-1.4</td><td>-3.5</td><td>-5.0</td><td></td></tr><tr><td>r</td><td></td><td> $\checkmark$ </td><td></td><td>-0.1</td><td>-0.1</td><td>=</td><td>=</td><td></td></tr></table>

Query Rejection Figure 4 makes query rejection visible on a single ScanNetV2 validation scene. Within the mask of the top-15 scored predictions, 418 queries were initialized (all balls), of which 23 were not rejected by non-object labeling, scoring, and NMS. The masks of the 15 highest-scoring predictions (green balls) lie on or inside the objects they predict, even though several dozen queries were initialized nearby, e.g., on the table surface and the chair cluster. The majority (blue) are rejected. The 8 predictions, which are outside of the top-15 are marked red.

Table 6. ScanNet++V2 validation results on full-scale scenes. MAFT was trained with $N _ { \mathcal { Q } } ~ = ~ 8 0 0$ queries and on the same superpoint partitions as AQ3D.
<table><tr><td></td><td colspan="2">With chunking</td><td colspan="2">Without chunking</td></tr><tr><td>Method</td><td>mAP</td><td>AP@50</td><td>mAP(∆)</td><td> $\mathrm { A P } @ \bar { \infty } 5 0 \left( \Delta \right)$ </td></tr><tr><td>MAFT [17]</td><td>26.0</td><td>39.2</td><td>-4.7</td><td>-5.0</td></tr><tr><td>AQ3D (44M)</td><td>36.9</td><td>53.2</td><td>-0.8</td><td>-0.9</td></tr><tr><td>AQ3D (88M)</td><td>38.3</td><td>57.1</td><td>-0.9</td><td>-1.1</td></tr></table>

![](images/25b57b52846c36d2f69ae94708e8e18c2df50682c1e9f54ef26d1356a365293f.jpg)  
Figure 4. Query rejection on a ScanNetV2 validation scene. Left: top-15 highest-scoring predictions. Right: the initial position of every query that lies within the prediction masks.

Table 7. Ablation on the ScanNetV2 validation set, starting from MAFT [17]. All deltas are w.r.t. to the baseline.
<table><tr><td></td><td>|mAP AP@50</td></tr><tr><td>MAFT (baseline) [17] w/ Fn</td><td>59.9 76.5</td></tr><tr><td>+ Pooling and  $\lambda _ { \mathcal { O } } = 0 . 5$ </td><td>+1.6 +2.2</td></tr><tr><td>+ Relative positional modeling through RoPE</td><td>+2.1 +4.1</td></tr><tr><td>+ Adaptive query set</td><td>+3.4 +4.9</td></tr><tr><td>+ Scaling backbone (11M → 44M)</td><td>+4.2 +5.3</td></tr><tr><td>+ Mask refinement</td><td>+4.6 +4.6</td></tr><tr><td>+ Dropout</td><td>+5.0+5.3</td></tr><tr><td>+ Cosine classifier</td><td>+5.7 +5.6</td></tr></table>

## 4.4. Ablation Study

Table 7 traces the path from the MAFT baseline to AQ3D, each row adding one component on top of the preceding ones. Replacing MAFT’s relative encodings with 3D RoPE is the largest single step on AP@50 (+1.9). ScanNetV2 is the least favorable benchmark for this component, since its scenes are single rooms of uniform extent and low instance density, consistent with the much larger effect on ScanNet++V2 (cf. Tab. 6). Besides pooling and increasing $\lambda _ { \mathcal { O } }$ , using the adaptive query set is the largest single-step improvement in mAP (+1.3). The remaining rows are scaling, decoder- and training-level refinements (further ablations in the supplement).

Tab. 8 disentangles MAFT-like absolute and our relative positional signals. On ScanNetV2, both contribute. We still use absolute positional modeling for ScanNetV2; however, even without it, our mAP exceeds all previously reported validation results. On ScanNet200, the higher instance density requires a different approach. The best configuration uses RoPE alone, and adding absolute encoding degrades accuracy, while removing both costs −3.1 mAP and −3.8 AP@50. RoPE is the component that carries positional information in both settings, and its contribution grows with instance density.

Table 8. Positional modeling ablation (AP values are differences w.r.t. the ScanNetV2 (I) and ScanNet200 (V) baselines).
<table><tr><td>Absolute RoPE</td><td></td><td>ScanNetV2 mAP AP@50</td><td>ScanNet200 mAP AP@50</td><td></td></tr><tr><td>√</td><td>√</td><td>I 65.6 82.1</td><td>|IV-0.8</td><td>-1.3</td></tr><tr><td></td><td>√</td><td>ⅡI-0.6 -1.1</td><td>V 39.6</td><td>49.7</td></tr><tr><td></td><td>1</td><td>-1.5 -0.7</td><td>VI -3.1</td><td>-3.8</td></tr></table>

## 5. Conclusion

We presented AQ3D, a decoder for 3D instance segmentation. Queries are instantiated at a fixed ratio of the scene’s superpoints, without proposal supervision. 3D RoPE over quantized metric coordinates keeps spatial reasoning scale-consistent. We removed the bounded lookup tables [17, 22, 35], the relational priors of Relation3D [22] in self-attention, and the semantic proposal prefix of [40, 41], leaving close to a plain decoder. This swaps expressiveness for scale-consistency. RoPE is content-agnostic, so unlike MAFT’s contextual encoding, it cannot learn querydependent distance preferences. On full-scale ScanNet++V2 scenes, MAFT loses roughly 5 mAP while AQ3D loses about 1 mAP, although only certain applications motivate this task, where scene extent and object count are not known in advance. Both contributions are orthogonal to context-based augmentation [39] and to backbone capacity [43], so some combinations remain unexplored across all three datasets. The highest costs are the quadratic growth of self-attention with $N _ { S }$ , depending on the superpoint segmentator, which is unlearned, must be recalibrated per dataset, and upperbounds recall through instances lost to superpoint majority vote. AQ3D trades scene scale for a dependency on a geometric partition. Deriving the budget from a learned measure of complexity may remove it.

## References

[1] Fred´ eric Bosch´ e. Automated recognition of 3D CAD model´ objects in laser scans and calculation of as-built dimensions for dimensional compliance control in construction. Advanced Engineering Informatics, 24(1):107–118, 2010. 1

[2] Fred´ eric Bosch´ e, Mahmoud Ahmed, Yelda Turkan, Carl T.´

Haas, and Ralph Haas. The value of integrating Scan-to-BIM and Scan-vs-BIM techniques for construction monitoring using laser scanning and BIM: The case of cylindrical MEP components. Automation in Construction, 49:201–213, 2015. 1

[3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-End Object Detection with Transformers. In ECCV, pages 213–229, 2020. 1, 2, 3

[4] Shaoyu Chen, Jiemin Fang, Qian Zhang, Wenyu Liu, and Xinggang Wang. Hierarchical Aggregation for 3D Instance Segmentation. In ICCV, pages 15447–15456, 2021. 2

[5] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention Mask Transformer for Universal Image Segmentation. In CVPR, pages 1280–1289, 2022. 1, 2

[6] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes. In CVPR, pages 2432–2443, 2017. 1, 5

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, 2021. 3

[8] Gabor Erd ´ os, Takahiro Nakano, Gergely Horv ˝ ath, Youichi´ Nonaka, and Jozsef V ´ ancza. Recognition of complex engi-´ neering objects from large-scale point clouds. CIRP Annals, 64(1):165–168, 2015. 1

[9] Pedro F. Felzenszwalb and Daniel P. Huttenlocher. Efficient Graph-Based Image Segmentation. IJCV, 59(2):167–181, 2004. 5, 3

[10] Benjamin Graham, Martin Engelcke, and Laurens van der Maaten. 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks. In CVPR, pages 9224–9232, 2018. 4

[11] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary Position Embedding for Vision Transformer. In ECCV, pages 289–305, 2024. 3

[12] Ji Hou, Angela Dai, and Matthias Nießner. 3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans. In CVPR, pages 4416–4425, 2019. 2

[13] Nathan Hughes, Yun Chang, and Luca Carlone. Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization, 2022. 1

[14] Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, Chi-Wing Fu, and Jiaya Jia. PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation. In CVPR, pages 4866–4875, 2020. 2, 6

[15] Maxim Kolodiazhnyi, Anna Vorontsova, Anton Konushin, and Danila Rukhovich. OneFormer3D: One Transformer for Unified Point Cloud Segmentation. In CVPR, pages 20943– 20953, 2024. 2, 3, 5, 6, 7

[16] Maksim Kolodiazhnyi, Anna Vorontsova, Anton Konushin, and Danila Rukhovich. Top-Down Beats Bottom-Up in 3D Instance Segmentation. In WACV, pages 3554–3562, 2024. 2

[17] Xin Lai, Yuhui Yuan, Ruihang Chu, Yukang Chen, Han Hu, and Jiaya Jia. Mask-Attention-Free Transformer for 3D Instance Segmentation. In ICCV, pages 3670–3680, 2023. 1, 2, 3, 4, 5, 6, 7, 8

[18] Zhihao Liang, Zhihao Li, Songcen Xu, Mingkui Tan, and Kui Jia. Instance Segmentation in 3D Scenes using Semantic Superpoint Tree Networks. In ICCV, pages 2763–2772, 2021. 2, 6

[19] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR, 2022. 3

[20] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. In ICCV, pages 9992–10002, 2021. 3

[21] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization, 2019. 5

[22] Jiahao Lu and Jiacheng Deng. Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation. In CVPR, pages 8889–8899, 2025. 1, 2, 3, 4, 5, 6, 7, 8

[23] Jiahao Lu, Jiacheng Deng, Chuxin Wang, Jianfeng He, and Tianzhu Zhang. Query Refinement Transformer for 3D Instance Segmentation. In ICCV, pages 18470–18480, 2023. 1, 2, 6

[24] Keno Moenck and Thorsten Schuppstuhl. Geometric digital ¨ twins of long-living assets: Uncertainty-aware 3D images from measurement and CAD data. Procedia CIRP, 126:975– 980, 2024. 1

[25] Alexey Nekrasov, Jonas Schult, Or Litany, Bastian Leibe, and Francis Engelmann. Mix3D: Out-of-Context Data Augmentation for 3D Scenes. In 3DV, pages 116–125, 2021. 5

[26] Tuan Duc Ngo, Binh-Son Hua, and Khoi Nguyen. ISB-Net: A 3D Point Cloud Instance Segmentation Network with Instance-aware Sampling and Box-aware Dynamic Convolution. In CVPR, pages 13550–13559, 2023. 2, 6

[27] Antoni Rosinol, Andrew Violette, Marcus Abate, Nathan Hughes, Yun Chang, Jingnan Shi, Arjun Gupta, and Luca Carlone. Kimera: From SLAM to Spatial Perception with 3D Dynamic Scene Graphs, 2021. 1

[28] David Rozenberszki, Or Litany, and Angela Dai. Language-Grounded Indoor 3D Semantic Segmentation in the Wild. In ECCV, pages 125–141, 2022. 1, 5

[29] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3D: Mask Transformer for 3D Semantic Instance Segmentation. In ICRA, pages 8216–8223, 2023. 1, 2, 3, 6

[30] Sangyun Shin, Kaichen Zhou, Madhu Vankadari, Andrew Markham, and Niki Trigoni. Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation. In CVPR, pages 4060–4069, 2024. 2, 6

[31] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568(C), 2024. 2, 3

[32] Jiahao Sun, Chunmei Qing, Junpeng Tan, and Xiangmin Xu. Superpoint transformer for 3D scene instance segmentation. In AAAI, pages 2393–2401, 2023. 1, 2, 3, 4, 5, 6, 7

[33] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 6000–6010, 2017. 3

[34] Thang Vu, Kookhoi Kim, Thanh Nguyen, Tung M. Luu, Junyeong Kim, and Chang D. Yoo. Scalable SoftGroup for 3D Instance Segmentation on Point Clouds. IEEE TPAMI, 46(4): 1981–1995, 2024. 2

[35] Duanchu Wang, Jing Liu, Haoran Gong, Yinghui Quan, and Di Wang. CompetitorFormer: Competitor Transformer for 3D Instance Segmentation, 2025. 1, 2, 3, 6, 7, 8

[36] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point Transformer V3: Simpler, Faster, Stronger. In CVPR, pages 4840–4851, 2024. 3

[37] Yizheng Wu, Min Shi, Shuaiyuan Du, Hao Lu, Zhiguo Cao, and Weicai Zhong. 3D Instances as 1D Kernels. In ECCV, pages 235–252, 2022. 2

[38] Christopher Xie, Yu Xiang, Arsalan Mousavian, and Dieter Fox. Unseen Object Instance Segmentation for Robotic Environments. IEEE Transactions on Robotics, 37(5):1343–1359, 2021. 1

[39] Rongkun Yang, Ye Zhang, Longguang Wang, Zhiheng Fu, Lian Xu, and Yulan Guo. Beyond context bias: Adaptive instance placement for robust 3d instance segmentation, 2026. 1, 5, 6, 7, 8

[40] Lei Yao, Yi Wang, Moyun Liu, and Lap-Pui Chau. SGI-Former: Semantic-guided and Geometric-enhanced Interleaving Transformer for 3D Instance Segmentation. IEEE TCSVT, 35(3):2276–2288, 2025. 1, 2, 3, 5, 6, 7, 8

[41] Lei Yao, Yi Wang, Yawen Cui, Moyun Liu, and Lap-Pui Chau. LaSSM: Efficient Semantic-Spatial Query Decoding via Local Aggregation and State Space Models for 3D Instance Segmentation. IEEE TCSVT, 36(6):7513–7525, 2026. 1, 2, 3, 6, 7, 8

[42] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes. In ICCV, pages 12–22, 2023. 1, 5, 7

[43] Kadir Yilmaz, Adrian Kruse, Tristan Hofer, Daan de Geus,¨ and Bastian Leibe. Volume Transformer: Revisiting Vanilla Transformers for 3D Scene Understanding, 2026. 1, 2, 3, 4, 5, 6, 7, 8

[44] Yuanwen Yue, Damien Robert, Jianyuan Wang, Sunghwan Hong, Jan Dirk Wegner, Christian Rupprecht, and Konrad Schindler. LitePT: Lighter Yet Stronger Point Transformer, 2026. 2, 3, 5

[45] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection, 2022. 3

[46] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable Transformers for End-to-End Object Detection, 2021. 3

[47] Chungang Zhuang, Shaofei Li, and Han Ding. Instance segmentation based 6D pose estimation of industrial objects using point clouds for robotic bin-picking. Robotics and Computer-Integrated Manufacturing, 82:102541, 2023. 1

# AQ3D: Adaptive Query Transformer for 3D Instance Segmentation

Supplementary Material

Table 9. Class head and mask refinement ablation on the Scan-NetV2 and ScanNet200 validation sets. Rows are independent single-factor changes to the baselines.
<table><tr><td></td><td>ScanNetV2 mAP AP@50</td><td></td><td>ScanNet200 mAP AP@50</td></tr><tr><td>Baseline</td><td>65.6</td><td>82.1</td><td>39.6 49.7</td></tr><tr><td>Class head: cosine → linear</td><td>-0.7</td><td>-0.3</td><td>-1.9 -2.5</td></tr><tr><td>Mask refinement → none</td><td>-0.9</td><td>-0.7</td><td>-4.1 -3.4</td></tr><tr><td>Query initialization with zeros</td><td>-0.7</td><td>-0.9</td><td>=</td></tr></table>

## 6. Additional Ablations

Class Head Table 9 replaces the cosine classifier of Eq. (1) with a linear projection. The cost is 0.7 mAP on ScanNetV2 but 1.9 mAP and 2.5 AP@50 on ScanNet200. The adaptive query set is overcomplete, so most queries must be routed to the non-object label. A linear head can express this through feature magnitude alone, whereas normalizing the query and prototype forces the decision to rely on direction, keeping background rejection comparable across queries with differing norms.

Mask Refinement Removing the mask refinement branch, in which superpoint features attend back to the queries before the mask logits are recomputed, costs 0.9 mAP and 0.7 AP@50 on ScanNetV2. On ScanNet200, the same removal costs 4.1 mAP and 3.4 AP@50 (Tab. 9), a factor of more than four. A one-way decoder updates queries against a scene representation that is frozen after the backbone, so all instances of a superpoint region compete for the same static mask features. Especially in the ScanNet200 case, mask tokens that can absorb query information help object discrimination.

Training Recipe Table 10 varies the non-object weight $\lambda _ { \mathcal { O } }$ and dropout. $\lambda _ { \mathcal { O } } { = } 0 . 1$ is the standard value in related methods [5, 22, 32]. To strengthen the class head’s nonobject prediction capabilities, we varied it while adding dropout. The two factors are not separable: without dropout, the smaller weight $\lambda _ { \mathcal { O } } { = } 0 .$ 1 is preferable, whereas a higher $\lambda _ { \mathcal { O } } { = } 0 . 5$ needs more dropout. Together, increasing $\lambda _ { \mathcal { O } }$ and adding dropout adds 1.4 mAP. $\mathbf { A } \lambda _ { \mathcal { O } }$ that is too low leaves the background queries under-penalized. Increasing $\lambda _ { \mathcal { O } }$ increases the training signal of background predictions. However, the objective overfits easily, which the layer dropout schedule (from 0.2 in the first layer to 0 in the last) and the head dropout of 0.1 counteract.

Table 10. Training recipe on the ScanNetV2 validation set (varying λ<sub>∅</sub> = non-object weight and dropout). AP values are differences w.r.t. the baseline IV.
<table><tr><td></td><td> $\lambda _ { \mathcal { O } } = 0 . 1$ </td><td> $\lambda _ { \mathcal { O } } = 0 . 5$ </td><td>Dropout</td><td>mAP AP@50</td><td></td></tr><tr><td>I</td><td>√</td><td></td><td></td><td>-1.4</td><td>-0.7</td></tr><tr><td>ⅡI</td><td>√</td><td></td><td>√</td><td>-0.8</td><td>-0.5</td></tr><tr><td>III</td><td></td><td>√</td><td>=</td><td>-1.6</td><td>-1.0</td></tr><tr><td>IV</td><td></td><td>√</td><td>√</td><td>65.6</td><td>82.1</td></tr></table>

Table 11. RoPE hyperparameters on the ScanNetV2 validation set (varying θ and grid size δ). AP values are differences w.r.t. the baseline (θ = 100 and grid size 0.05).
<table><tr><td>θ</td><td>δ</td><td>mAP</td><td>AP@50</td></tr><tr><td>50</td><td>0.05</td><td>+0.2</td><td>+0.1</td></tr><tr><td>50</td><td>0.1</td><td>-1.4</td><td>-1.1</td></tr><tr><td>100 100</td><td>0.05 0.1</td><td>65.6 -0.8</td><td>82.1 -0.7</td></tr><tr><td></td><td></td><td></td><td>-0.5</td></tr><tr><td>1000 1000</td><td>0.05 0.1</td><td>-0.5 -0.6</td><td>-0.6</td></tr></table>

RoPE Hyperparameters Table 11 varies the two hyperparameters of the 3D extension. The grid size is the dominating factor. At every value of θ, coarsening the quantization from 0.05 m to 0.1 m degrades performance. The base frequency θ is comparatively benign at the finer grid (+0.2 at θ=50, −0.5 at θ=1000), and we retain θ=100, matching the value used for RoPE in [43].

## 7. Compute

Protocol We profile the three decoder modules that scale with the query count and are present in MAFT [17] and AQ3D: self-attention, cross-attention, and the FFN, accumulating forward-pass Multiply-Accumulate (MAC) operations over one epoch of each training split using the applied training augmentation. MAFT’s and AQ3D’s total compute exceeds the given numbers; however, we compare only the interesting parts that are scaled in AQ3D. For the attention modules, we only count the attention operation itself. Input projections, normalization, the backbone, and the mask refinement branch, which has no counterpart in MAFT, are excluded; the numbers are therefore not our total decoder cost but the cost of the modules shared by the two decoders. MAFT’s relative position encoding is included in the profiled operation and is counted. Backward-pass arithmetic is excluded; it would roughly add twice the forward cost for these modules. MAFT is measured at its published budget, $N _ { \mathcal { Q } } = 4 0 0$ on ScanNetV2 and 800 on ScanNet++V2; AQ3D at $r = 0 . 6$ . Both use $L = 6$ layers, $d _ { M } = 2 5 6$ , and the sparse U-Net backbone. We report GMACs per scene.

![](images/ae38c369f759f3bfe183cabb95e8d0773f88f236a88161eefca2982228d2f50a.jpg)  
Figure 5. Decoder compute on ScanNetV2. Forward-pass MACs per scene of the modules shared by both decoders, split into selfattention, cross-attention, and FFN, for MAFT [17] $( N _ { \mathscr { Q } } = 4 0 0 )$ and AQ3D (solid, $r = 0 . 6 )$ . Totals are given above each bar. The difference between the two batch sizes is the arithmetic spent on padded entries.

Decoder Cost Figure 5 and Fig. 6 break the per-scene cost into its components. At batch size 1, AQ3D requires 6.8 GMACs per scene on ScanNetV2 against MAFT’s 4.6, a factor of $\approx 1 . 5$ . On ScanNet++V2, the ordering reverses, 10.3 against 11.5, so our decoder is a little bit cheaper than the fixed-budget baseline while significantly improving over it (Tab. 4). The FFN isolates the query budget, being linear in $N _ { \mathcal { Q } }$ and roughly identical in both methods. Cross-attention departs from it. On ScanNet++V2 we rely on RoPE alone, so queries and keys carry no concatenated positional channel and the query-key product is taken at width $d _ { M }$ rather than $2 \times d _ { M }$ , resulting in fewer GMACs at an equal budget. Selfattention is the component the adaptive scheme makes more expensive.

Batching Overhead Since $N _ { \mathscr { Q } }$ varies per sample, each batch is padded to its maximum and masked (cf. Sec. 3); the masked entries are still computed. Fig. 7 compares perscene cost at batch size 4 against the unpadded batch-size-1 measurement. A batch of four raises our cost by 56% and 50%, against 19% and 16% for MAFT, whose fixed budget admits no query-side padding at all. Grouping samples of similar $N _ { S }$ would reduce padding, at the price of correlating batch composition with scene size.

## 8. Scene Statistics

The design of AQ3D rests on the premise (Sec. 1) that the number of foreground objects in a scene grows with its geometric complexity and spatial extent, for which we use $N _ { S }$ as a proxy. Fig. 8 to Fig. 10 show the number of ground-truth instances against $N _ { S }$ , separately for the training split (including the training data augmentation used, e.g., cropping) and the (unchunked) validation split.

![](images/48e1c454ed4d5b2881aab4761c1c9ba57ed6e3027bed6f4f609b29ad282d02c2.jpg)  
Figure 6. Decoder compute on ScanNet++V2, plotted as in Fig. 5. MAFT uses $N _ { \mathcal { Q } } = 8 0 0$ . At an average query budget equal to the baseline’s, the adaptive decoder is cheaper overall. Relying on RoPE alone removes the concatenated positional channel from the query-key product, which offsets the higher self-attention cost from a variable $N _ { \mathfrak { Q } }$

![](images/d92d99c677c27f8543df5c4fb4c5860fe1424b08d42d37bbbfbd875b25c0aaf4.jpg)  
Figure 7. Cost of padding a batch to its largest query set. Solid bars are the unpadded per-scene cost measured at batch size 1; hatched bars are the additional arithmetic resulting from batch size 4, annotated as a percentage of the unpadded cost. A fixed budget admits no query-side padding, so MAFT’s residual growth comes entirely from the varying superpoint count $N _ { S }$ , which both methods share.

The relationship is positive across all three datasets and both splits. On ScanNetV2 (cf. Fig. 8) instance counts are modest, with training chunks reaching roughly 2 200 superpoints and 40 instances. ScanNet200 (cf. Fig. 9) reuses the identical reconstructions but replaces the label set with a finer 200-class taxonomy, resulting in more instances. Scan-Net++V2 (cf. Fig. 10) spans by far the widest range. Fullscale validation scenes extend beyond 10 000 superpoints and 200 instances, with markedly larger variance than either ScanNet variant. This is precisely the regime in which a scene-agnostic query budget is most mismatched, and in which conditioning the budget on $N _ { S }$ pays off.

![](images/fc5ce6985e0a4ddb52de61bfe735aad9630a68bb4ad03022b82f9c23b280b0b5.jpg)  
Figure 8. Per-scene instance count versus superpoint count $N _ { S }$ on ScanNetV2.

![](images/e6d27a90410fe8c78b15b64c3cca186f32a78e515a2f267c3247c17e31403a57.jpg)  
Figure 9. ScanNet200 shares the reconstructions of ScanNetV2 but uses a finer class taxonomy; the instance count per scene roughly doubles, while $N _ { S }$ is unchanged.

## 9. ScanNet++V2 Segmentator Configuration

For ScanNetV2 and ScanNet200, we obtain superpoints from the graph-based segmentator of [9] in its default configuration $( k _ { t } = 0 . 0 1 , n _ { v } = 2 0 )$ ). ScanNet++V2 is reconstructed at sub-millimeter resolution, so this configuration produces an excessive number of superpoints, inflating both the decoder’s cross-attention cost and, through $N _ { \mathscr { Q } } = \lfloor r \cdot N _ { \mathscr { S } } \rfloor$ , the query budget and self-attention cost. We therefore recalibrate the two segmentator parameters for ScanNet++V2: the merging cutoff $k _ { t }$ and the minimum segment size $n _ { v }$

Two target values should be minimized: The number of superpoints $N _ { S }$ (average across all samples is $\bar { N } _ { \mathcal { S } } )$ and the number of lost instances during training due to point-instance majority vote. A superpoint is only assigned an instance label if at least 50% of its points belong to an instance. During training, after grid sampling and superpoint pooling, only instances with at least one superpoint are supervised. Coarser superpoints reduce the average $\bar { N } _ { S }$ but merge small instances into their neighbors, raising this loss.

![](images/09a2d37c7c69dfabe5d9fc06fd4fdeb252246c142292f99c87dcc48a6970e831.jpg)  
Figure 10. ScanNet++V2 covers a far wider range of scene sizes. Full-scale validation scenes reach beyond 10 000 superpoints and 200 instances, with substantially larger variance, the setting in which a fixed query budget is most severely miscalibrated.

Figure 11 varies the minimum segment size at fixed cutoff. Instance loss grows monotonically with $n _ { v } ,$ from ≈ 1.6 at $n _ { v } ~ = ~ 5 0 ~ $ to 8.9 at $n _ { v } ~ = ~ 3 0 0$ , as larger minimum segments swallow small instances. Based on Fig. 11, we choose $n _ { v } = 1 0 0$ . Fig. 12 varies the cutoff at $n _ { v } = 1 0 0$ and exhibits a minimum around $k _ { t } \in [ 0 . 1 , 0 . 2 ]$ (3.15 and 3.16 lost instances on average per sample), rising on both sides. Fig. 13 shows that $\bar { N } _ { S }$ decreases monotonically with $k _ { t } .$ , from 2 676 at $k _ { t } = 0 . 0 2$ to 2 358 at $k _ { t } = 0 . 5$ , as coarser merging yields fewer, larger superpoints. Around 3 lost instances per sample are few, giving an instance average of $\bar { N } _ { I } = 2 6 . 1 \pm 1 6 . 1$ (cf. Tab. 1; with training data augmentation) compared to increasing $n _ { v }$ and ending up losing around 9 instances per sample on average. We adopt $k _ { t } = 0 . 2 , n _ { v } = 1 0 0$

## 10. Qualitative Results on ScanNet++V2

Figure 14 to Fig. 17 compare the predictions of MAFT [17] (trained with 800 queries) and AQ3D (44M backbone variant) on full-scale ScanNet++V2 validation scenes. Both models are evaluated without chunking, so the scenes are considerably larger than the crops seen during training. The observations below are consistent with the quantitative gap

![](images/e72e6f075fcb4aa63d232935d910999494cc88ebd129a9626e439e7ef5d0a9b3.jpg)

Figure 11. Average number of ground-truth instances lost (train split) over the minimum segment size $n _ { v }$ , for two cutoff values $k _ { t } .$ The loss grows monotonically as larger minimum segments absorb small instances.  
![](images/c5aeb0ba67984126c169f65b1fa0e4b2de85c9dfe394d3f06ae3fb7a32793c05.jpg)  
Figure 12. Average instances lost (train split) over the merging cutoff k<sub>t</sub> at $n _ { v } = 1 0 0$ . The average lost instances is minimized in a plateau around $k _ { t } \in [ 0 . 1 , 0 . 2 ]$

![](images/8d396d6f00184364c46f1a5f3012e12d650e6aa41f1214398dc0f67549cd2e25.jpg)  
Figure 13. Mean superpoints per scene $\bar { N } _ { S }$ over $k _ { t }$ at $n _ { v } = 1 0 0$ (train split). $\bar { N } _ { S }$ decreases monotonically as coarser merging yields fewer, larger superpoints.

## reported in Tab. 6.

Fine-grained Objects in Dense Regions The clearest difference appears in regions of high object density. On the large desks in Fig. 14 and on the desks in the bottom-left and top-left of Fig. 15, AQ3D recovers small objects placed on the surfaces, whereas MAFT predominantly predicts the supporting furniture and absorbs the clutter on top of it into a few large masks. A fixed query budget is allocated uniformly over the scene, so densely populated areas receive no more decoder capacity than empty floor. Since $N _ { S }$ is itself elevated in geometrically detailed regions, the adaptive budget places more queries there.

Foreground Object Retrieval At an equal number of retained predictions, AQ3D spends fewer of them on noninstance classes. In Fig. 14, some share of MAFT’s top-100 masks covers structural surfaces (windows, walls, ceiling parts), which are not in AQ3D’s top-100 ranks.

Separation of Neighboring Instances Objects of the same class that are spatially adjacent are separated more reliably, while the instance masks are more complete. The stool cluster in the lower middle of Fig. 14 is merged into a single mask by MAFT, whereas AQ3D assigns a distinct instance to each stool. Every query is initialized from a superpoint and reasons about its neighbors through metrically consistent relative offsets, so queries seeded on adjacent but distinct objects remain distinguishable instead of collapsing onto the dominant one.

Small Scenes Performance is not restricted to large or cluttered scans. Fig. 17 shows the top-10 predictions on a small scene, where AQ3D retrieves more of the few present objects than MAFT. Even when the adaptive scheme instantiates fewer queries than the fixed budget of the baseline, the queries that are instantiated are grounded in the scene and are not competing against a large pool of slots calibrated to the average training scene.

![](images/d3b0ec547ed1de6fc85162defe184571d4815812e31e3f3f6a40e57dffebae1c.jpg)

![](images/03df86ef8e59ae01cc8333a9f1269db03324a84004911e75ce701a060a0a7452.jpg)  
(b) Ours (44M backbone)  
Figure 14. Top-100 predictions in ScanNet++V2 validation scene 578511c8a9.

![](images/7ad8a07f57c32edb50e212cc221c7143c67bf2c7defd907c4cb0d7037568c27b.jpg)  
(a) MAFT

![](images/169a3c2abb9e05f981a5201dd3f23da668ee795283b4d7bc66b17cb22243de15.jpg)  
(b) Ours (44M backbone)  
Figure 15. Top-100 predictions in ScanNet++V2 validation scene ac48a9b736.

![](images/ae8c1518d065675c78fc8c04fbc59aad0627bb300addb04284040c9f937765e0.jpg)  
Figure 16. Top-100 predictions in ScanNet++V2 validation scene 09c1414f1b.

![](images/e5f00251574fcb90b806e47ab238d45dcf2ad1ffc6b5282152bfa56c8666df7c.jpg)

![](images/03ad371daa331262635226cd1165a5a7e431c6f0f85afa48ab1fb2042d0e877d.jpg)  
Figure 17. Top-10 predictions in ScanNet++V2 validation scene 09c1414f1b.