# CrossWeave: Cross-Modal Scene and Instance Retrieval from Sparse 2D-3D Alignment

Aadith Warrier<sup>1</sup> Gnana Prakash Punnavajhala<sup>1</sup> Siddharth Tourani<sup>2</sup> Muhammad Haris Khan<sup>3</sup> Avinash Sharma<sup>4</sup> Madhava Krishna<sup>1</sup>

<sup>1</sup>IIIT Hyderabad <sup>2</sup>University of Heidelberg <sup>3</sup>MBZUAI <sup>4</sup>IIT Bombay

<sup></sup> crossweave3d.github.io <sup></sup> github.com/crossweave3d/crossweave

## Abstract

Cross-modal scene and instance retrieval aims to retrieve a corresponding 3D environment or object given query 2D images, serving as a foundational mechanism for spatial reasoning in robotics and AR/VR. Existing methods achieve this by learning alignment only at the level of whole scenes or objects, which provides no supervision for the fine-grained local structure within them. In this work, we show that supervision on a set of sparse 2D-3D correspondences, without scene- or instance-level contrastive objectives, is sufficient to train a single model for both scene- and instance-level retrieval. Specifically, we propose CrossWeave, a novel framework for cross-modal retrieval that learns sparse patch-point alignment, which alone outperforms existing methods. An optimal transport-based aggregator over the aligned features gives further gains by clustering structurally redundant regions, freeing descriptor capacity for semantically distinct local features. CrossWeave achieves state-of-the-art performance on both scene- and instance-level retrieval on ScanNet and 3RScan with a single model, whereas existing methods require separate models for each task. Our results suggest that whole-scene encoding is unnecessary for cross-modal retrieval and that sparse local correspondence supervision is a more effective and flexible alternative.

## 1 Introduction

Within autonomous navigation and mapping, processing continuous streams of image and point cloud data is a major computational bottleneck. Scene and instance retrieval provide a first-pass spatial reasoning step, enabling location- and object-specific downstream tasks without exhaustive search over large databases or 3D maps. Such coarse reasoning is often sufficient for identifying object presence, recognizing the current environment, or initializing fine-grained localization and loop closure in SLAM. Cross-modal retrieval is especially valuable when the map is built using a richer sensor modality than the query device, such as a camera-equipped robot localizing against a LiDAR-built map, or an AR/VR device anchoring content to a pre-scanned environment.

Formally, Cross-Modal Scene Retrieval (CMSR) is defined as retrieving, from a database of aligned 3D point cloud sub-maps, the sub-map corresponding to query RGB image(s). The related task of Cross-Modal Instance Retrieval (CMIR) is defined analogously: given RGB image(s) of an object, the system retrieves the corresponding 3D point cloud instance from a database of segmented object point clouds. CMSR remains an emerging problem, with few established methods for bridging the modality gap between 2D images and 3D maps. Existing approaches such as CrossOver [27] and SceneGraphLoc [19] rely on scene- or object-level supervision: CrossOver aligns modalities with separate scene- and instance-level contrastive objectives and requires a separate model for each task, while SceneGraphLoc matches image patches to the object nodes of a multimodal scene graph. In both cases, descriptors are learned using contrastive objectives similar to those used in visual place recognition [1, 11]. However, both methods supervise alignment at the level of whole scenes or objects rather than individual 3D points.

To address this, we propose CrossWeave, a cross-modal retrieval framework that learns from sparse local scene regions during training instead of aligning full scenes. CrossWeave introduces an image-patch-to-3D-point contrastive objective that preserves local cross-modal correspondences. It further aligns corresponding image-point cloud pairs using modality-specific descriptors obtained by aggregating local features, without requiring full-scene alignment. Using this architecture, we show that a single model trained with these losses, without any scene- or instance-level supervision, supports both scene- and object-level retrieval on the 3RScan and ScanNet benchmarks. We further demonstrate that patch-point alignment is the primary driver of instance retrieval performance, and improves scene retrieval when combined with learned aggregation.

## Contributions:

• We propose CrossWeave, a novel multi-modal framework that performs both CMSR and CMIR with a single model, using a dual-contrastive loss at both the local patch-point level and the image-point cloud level, but not a scene/instance-level contrastive loss. This allows improved retrieval when compared to methods that encode the whole 3D map either as global point cloud features or scene graphs.

• We show that adding local patch-point alignment consistently improves both CMSR and CMIR over training on aggregated descriptors alone, with the largest gains on fine-grained instance retrieval.

• We achieve state-of-the-art performance on 3RScan and ScanNet, outperforming competing baselines by 21 points on scene retrieval and 8 points on instance retrieval on ScanNet.

## 2 Related Work

## 2.1 Multi-modal Representation Learning

Multi-modal representation learning aims to align heterogeneous data sources within a shared embedding space. In this setting, contrastive learning has become the dominant paradigm. CLIP [25] demonstrated that large-scale contrastive training on paired image-text data yields rich representations capable of cross-modal understanding and strong zero-shot generalization to various downstream tasks. Building on this foundation, a series of works extended cross-modal alignment to the imagepoint cloud [40, 44] and text-image-point cloud modalities [35, 36, 15] with Uni3D [43] scaling model and dataset sizes used for training. Subsequently, ImageBind [9] performed alignment between six modalities by leveraging images as a hub and demonstrated emergent relationships between untrained pairs. Similarly, PointBind [10] incorporated the 3D modality and performed various downstream tasks by augmenting with an LLM. However, these methods remained restricted to object-level point clouds. CLIP<sup>2</sup> [38] extended multi-modal representation learning to scene-level point clouds, yet remained focused on semantic understanding rather than retrieval. Overall, prior work focuses on cross-modal alignment for semantic understanding, with comparatively less emphasis on retrieval-oriented representation design for 3D scenes.

## 2.2 Uni-modal and Cross-modal Retrieval

Retrieval methods aim to identify relevant database entries given a query, typically represented by a global descriptor. While early methods aggregated local features via hand-crafted clustering [13], later works introduced learned aggregation like NetVLAD [1] to preserve distributional structure. Unimodal visual [1, 14] and LiDAR-based [30, 31, 37, 24, 42] place recognition are widely studied along these lines. In the cross-modal domain, image-text retrieval is well-established [3], heavily driven by contrastive joint-embedding alignment [25, 17, 28]. Extending beyond 2D, recent works bridge textual queries with 3D point clouds via part-to-word matching [29] and multi-modal consistencies [26, 8]. Concurrently, scene graphs are used to capture explicit structural relations for cross-modal text-toimage retrieval, evolving from basic query parsing [12] to advanced message passing and attribute alignment [6, 33]. Despite these advances, cross-modal scene and instance retrieval remains largely underexplored, particularly for indoor environments mapped between images and 3D point clouds.

![](images/ab7dd772de0cf6639b4b8a3b43882f768b5d2d4afb1dc024e5373f612536ade7.jpg)  
Figure 1: Overview of CrossWeave. CrossWeave learns cross-modal retrieval representations from sparse image patch-3D point correspondences rather than scene- or instance-level labels. Scene Sampling: Given calibrated RGB-D observations and a reconstructed 3D map, visible patchpoint pairs are obtained by projecting 3D points into the image plane, filtering occluded points using rendered depth, quantizing valid pixels into DINOv2 patch tokens, and selecting a compact semantically diverse subset using feature-space farthest point sampling. Feature Extraction: Frozen encoders extract local patch and point features, which are projected by learnable projection matrices $W ^ { \mathrm { p a t c h } }$ and $W ^ { \mathrm { p o i n t } }$ into a shared embedding space. Feature Aggregation: A shared SALAD aggregator then maps variable-sized sets of local features into fixed-dimensional descriptors. Training combines a local patch-point $\mathcal { L } _ { p 2 p }$ contrastive loss with an aggregated image-point-cloud contrastive loss ${ \mathcal { L } } _ { i 2 p c } ,$ preserving fine-grained alignment while organizing local evidence into retrieval-ready descriptors. At inference, the same encoders and projection heads serve both tasks: SALAD descriptors are used for scene retrieval and mean-pooled local features for instance retrieval.

The most closely related work to ours is CrossOver [27], which trains for cross-modal alignment across text, images, floorplans, CAD models, and point clouds at both scene and instance levels, enabling retrieval across any pair of these modalities at multiple levels of granularity. We highlight two design aspects of prior work most relevant to our setting. First, instance- and scene-level retrieval rely on separate task-specific checkpoints, requiring distinct models for different granularities and limiting flexibility at inference time. Second, scene-level representations are constructed by averaging per-frame features (CLS and mean patch tokens) across frames for images and mean pooling point features for point clouds, which implicitly assigns equal weight to all local features, potentially losing fine-grained structural information critical for distinguishing visually similar scenes. In contrast, our method uses a unified model that learns fine-grained patch-to-point correspondences, which can be flexibly aggregated at inference time to support both instance- and scene-level retrieval. Furthermore, our aggregation assigns local features to shared cluster centers, capturing their distributional structure rather than averaging them away.

## 3 Methodology

The design philosophy behind CrossWeave is to both: effectively utilize the redundancy in the 3D scene structure and not lose fine-grained details due to the aggregation process over the local features to obtain modality-specific descriptors. Figure 1 provides an overview of CrossWeave.

## 3.1 Architecture and Training

In keeping with the convention in previous CMSR literature, we assume the 3D structure consists of individual RGB-D scans aligned into a single 3D map and not the individual scans themselves.

Scene Sampling for Patch-Point Pair Generation To exploit the redundancy present in both image and point-cloud observations, we avoid encoding all 3D points visible in an image and instead construct a compact set of informative patch-point correspondences. Given a point cloud P, we first project each 3D point into the image plane using the camera intrinsics K and camera pose $[ R \mid t ]$ Since projection alone does not guarantee that a point is actually visible from the camera viewpoint, we apply an explicit visibility constraint using the depth map rendered from the reconstructed mesh. A projected point is retained only if its depth lies within a small tolerance ϵ of the observed depth at the corresponding pixel. This removes occluded points and ensures that the resulting correspondences reflect physically visible image-geometry associations. These correspondences are the only supervision used during training, and no scene or instance labels are required.

The remaining valid pixel-point pairs are then quantized into image patch indices using a patch size of 14, following the patchification scheme of DINOv2 [22]. For each image patch, we randomly sample a single visible 3D point, producing a one-to-one patch-point mapping. This step reduces the strong local redundancy caused by multiple 3D points projecting to the same visual token, while preserving the spatial correspondence required for patch-level cross-modal supervision.

Finally, we further downsample these patch-point pairs using Farthest Point Sampling (FPS) [7] in feature space rather than Euclidean space. Specifically, FPS is applied to the point features extracted by our frozen 3D encoder, Utonia [41], a recent point-cloud feature extractor based on the Point Transformer v3 architecture [34]. Unlike geometric FPS, which primarily encourages spatial coverage, feature-space FPS encourages semantic diversity by selecting points that are maximally separated in the learned embedding space. This prevents the sampled set from being dominated by large or frequently observed objects in the field of view, yielding a more balanced set of local correspondences for contrastive learning. The output of this scene sampling is a set of point-patch pairs $\{ p _ { i } \in \mathbb { R } ^ { 3 } , q _ { i } \in \mathbb { R } ^ { 1 4 \times 1 4 } \} , i = 1 \to \mathbf { \bar { N } }$

Feature Extraction The sampled point-patch pairs are passed on to pre-trained modality-specific encoders. For the image patch $q _ { i }$ , we use the DINOv2-L variant for generating per-patch features $f _ { i } ^ { \mathrm { p a t c h } }$ . For the 3D point $p _ { i } ,$ we use Utonia to extract per-point features $f _ { i } ^ { \mathrm { p o i n t } }$ from the point cloud. Subsequently, both features are $\ell _ { 2 }$ normalized and projected into a shared embedding space using learnable projection matrices $W _ { \mathrm { p a t c h } }$ and $W _ { \mathrm { p o i n t } }$ . This results in per modality aligned features $H ^ { p a t c h } = \{ h _ { i } ^ { \mathrm { p a t c h } } \} _ { i = 1 } ^ { N }$ and $H ^ { p o i n t } = \{ h _ { i } ^ { \mathrm { p o i n t } } \} _ { i = 1 } ^ { N }$ per image.

Feature Aggregation via Optimal Transport For each image-point-cloud sample, we aggregate the N sampled patch-point correspondences into fixed-dimensional descriptors using a $\mathrm { S A L A D } \left[ 1 1 \right]$ aggregation layer. Here, N denotes the number of retained patch-point pairs after visibility filtering and sampling, while K denotes the number of latent aggregation clusters used by SALAD. The goal of this aggregation is to summarize a compact set of local cross-modal correspondences into a single descriptor, while capturing discriminative and recurring patterns in the cross-modal data. Importantly, aggregation is performed only over the $N$ sampled correspondences from one image-point-cloud pair, rather than over all points in a full scene or features of an object instance.

To complete the mapping from patch-point level local features $( H ^ { p o i n t } , H ^ { p a t c h } )$ to image-point cloud level features, SALAD uses two MLPs shared across both modalities. The first MLP, $\varphi ^ { \mathrm { { \bar { c l u s t e r } } } }$ , maps each local feature to a cluster feature, while the second MLP, $\varphi ^ { \mathrm { s c o r e } }$ , predicts its assignment scores to the K latent clusters:

$$
C = \varphi ^ { \mathrm { c l u s t e r } } ( H ) \in \mathbb { R } ^ { N \times d _ { \mathrm { c l } } } , \qquad S = \varphi ^ { \mathrm { s c o r e } } ( H ) \in \mathbb { R } ^ { N \times K } .
$$

Here, $d _ { \mathrm { c l } }$ is the dimensionality of the transformed local feature used inside the aggregation layer. Thus, $C _ { i } ,$ the i-th row of $C ,$ is the feature content contributed by the i-th local correspondence, while $S _ { i , k }$ measures how strongly it should be assigned to the k-th latent cluster.

Following SALAD [11], the score matrix S is augmented with a learnable dustbin parameter ϕ and passed through the Sinkhorn algorithm [4] for three iterations. This produces a balanced soft-assignment matrix $\mathcal { P } \in \mathbb { R } ^ { K \times N }$ , after removing the dustbin assignments. The dustbin allows low-confidence or uninformative local features to be absorbed rather than being forced into one of the K clusters, which improves robustness when some sampled correspondences are noisy or weakly discriminative. Each cluster descriptor $V _ { k }$ is then computed as a weighted sum of the transformed local features assigned to the k-th cluster. Thus, each cluster summarizes the subset of local evidence associated with the k-th latent concept. The vectors $V _ { k }$ are individually $\ell _ { 2 } \cdot$ -normalized, concatenated, and then globally $\ell _ { 2 }$ -normalized to obtain the final descriptor.

$$
G = \operatorname { n o r m } \left( \operatorname { c o n c a t } \left( \operatorname { n o r m } ( V _ { 1 } ) , \dots , \operatorname { n o r m } ( V _ { K } ) \right) \right) \in \mathbb { R } ^ { K d _ { \mathrm { c l } } } .
$$

Applying this aggregation independently to $H ^ { \mathrm { p a t c h } }$ and $H ^ { \mathrm { p o i n t } }$ yields the modality-specific descriptors $G ^ { \mathrm { p a t c h } }$ and $G ^ { \mathrm { p o i n t } }$ , which are later used for cross-modal contrastive supervision.

Dual-Objective Contrastive Learning CrossWeave is trained with two complementary contrastive objectives, both based on InfoNCE [21, 20]. The first objective, the local patch-point loss $\mathcal { L } _ { \mathrm { p 2 p } }$ is computed over all $M = B \times N$ patch-point correspondences in a batch, where B is the batch size and N is the number of sampled correspondences per image. This loss directly aligns local visual and geometric features, encouraging each image patch to be close to its corresponding 3D point while treating other patch-point pairs in the batch as negatives. Since it is defined only at the correspondence level, it does not require scene-level or instance-level annotations.

The second objective, the image-point-cloud loss ${ \mathcal { L } } _ { \mathrm { i 2 p c } }$ realized as the InfoNCE loss as well, is applied to the SALAD-aggregated descriptors of each image-point-cloud pair in the batch. This loss encourages the aggregated image descriptor and the corresponding aggregated point-cloud descriptor to occupy a shared retrieval space. Unlike full-scene supervision, this alignment is derived from descriptors built only from the sampled local correspondences, preserving the focus on sparse but discriminative cross-modal evidence.

The final training objective combines the local and aggregated contrastive losses, $\mathcal { L } = \mathcal { L } _ { \mathrm { p 2 p } } + \mathcal { L } _ { \mathrm { i 2 p c } }$ Together, these objectives allow CrossWeave to learn retrieval-oriented representations from sparse patch-point correspondences alone. The local loss preserves fine-grained cross-modal alignment, while the aggregated loss organizes these local cues into image-level and point-cloud-level descriptors.

## 3.2 Inference

Although CrossWeave is trained only on sparse local patch-point correspondences, it can be applied directly to both scene-level and instance-level retrieval at inference. This follows from the inputagnostic design of the model. The projection layers are applied independently to each local image or point feature, and the SALAD aggregator [11] maps any variable-sized set of local features to a fixed-dimensional descriptor. Thus, the same network can aggregate a small set of training correspondences, a complete scene point cloud, or a segmented object instance without architectural modification. SALAD produces a descriptor $G \in \mathbb { R } ^ { K \widetilde { d _ { \mathrm { c l } } } }$ , where K is the number of learned clusters and $d _ { \mathrm { c l } }$ is the dimensionality of each cluster vector.

Scene Retrieval For scene-level retrieval, the full scene point cloud is encoded using Utonia, projected into the shared embedding space, and aggregated with SALAD to obtain a point-cloud descriptor G<sup>point</sup>. Let the scene be observed by V images, where the v-th image produces $\dot { N } _ { v }$ DINOv2 patch tokens. We concatenate the patch tokens from all views before aggregation:

$$
H ^ { \mathrm { p a t c h } } = \operatorname { c o n c a t } \Big ( H _ { 1 } ^ { \mathrm { p a t c h } } , \ldots , H _ { V } ^ { \mathrm { p a t c h } } \Big ) \in \mathbb { R } ^ { ( \sum _ { v = 1 } ^ { V } N _ { v } ) \times d } .
$$

These concatenated multi-view patch features are passed through the image projection layer and SALAD aggregator to produce the image-side scene descriptor $G ^ { \mathrm { { p a t c h } } }$ . Retrieval is then performed by ranking database descriptors according to their cosine similarity with the query.

Instance Retrieval Instance-level retrieval restricts the inputs to a segmented object instance. The scene point cloud is replaced by the instance point cloud, and each query view is an RGB image together with the instance mask, of which only the patch tokens covered by the mask are used. For instance retrieval we use the local features directly instead of the SALAD descriptor: the projected patch features from all query views and the projected instance point features are each mean-pooled, and the two resulting descriptors are compared using cosine similarity. We use local features because they give better instance retrieval than the SALAD descriptor on both datasets (Table 4).

## 3.3 Implementation Details

CrossWeave is implemented in PyTorch [23] and trained on a single NVIDIA A6000 GPU using mixed-precision bfloat16 training. Training requires 17.1 GB of VRAM and about 21 GPU hours. We use the AdamW optimizer [16] with a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 2 }$ , and train for 30 epochs with a batch size of 16. During training, each sample contains 32 patch-point correspondences. DINOv2-L produces 1024-dimensional image patch features, and Utonia produces 1224-dimensional point features. These are projected into a shared 1024-dimensional embedding space using modalityspecific projection matrices $\bar { W } _ { \mathrm { p a t c h } }$ and $W _ { \mathrm { p o i n t } }$ . The SALAD aggregator uses K = 24 cluster vectors with $d _ { \mathrm { c l } } = 1 2 8$ , producing a final descriptor of dimension $K \times { \bar { d } } _ { \mathrm { c l } } = 2 4 \times 1 2 8 = 3 0 7 2$ . We measure inference time on a single NVIDIA A6000 GPU using 10 query images of resolution $3 \times 5 1 8 \times 5 1 8$ and a point cloud of 50,000 points. The RGB forward pass (DINOv2 + SALAD) runs in $1 1 8 . 6 9 { \pm } 1 . 8 7 $ ms and the point cloud forward pass (Utonia + SALAD) runs in $3 3 2 . 1 5 \pm 4 . 5 2$ ms, with a combined peak memory usage of 5.24 GB. In practice, the point cloud database descriptors are computed offline and cached, so only the RGB forward pass is required at query time.

## 4 Experiments

## 4.1 Datasets

ScanNet [5] ScanNet is a large-scale indoor dataset containing posed RGB-D frames from 707 scenes and 1513 scans, with multiple runs per scene. The dataset provides dense 3D meshes, with semantic and instance annotations for RGB frames and reconstructed meshes. We use the standard splits of 565 scenes / 1201 scans for training and 142 scenes / 312 scans for validation.

3RScan [32] 3RScan contains 1482 scans of 478 indoor scenes, with posed RGB-D frames, dense   
3D meshes, and semantic and instance annotations. We use 384 scenes / 1177 scans for training and   
46 scenes / 157 scans from the validation split for evaluation following [17].

Additional dataset details are provided in Appendix A.

## 4.2 Tasks and Metrics

Metrics For both scene-level and instance-level retrieval, we report Top-k accuracy, defined as the percentage of queries for which the correct scene or instance appears among the top-k retrieved results. For instance retrieval, we additionally report Recall@k% (R@k%), which measures the percentage of scenes for which at least k% of the query instances are correctly retrieved.

Scene Retrieval In the scene retrieval task, the query consists of 10 RGB views of a scene, selected following the protocol of CrossOver [27], and the database consists of 3D point clouds reconstructed from scanned scenes. A retrieval is considered correct if the model retrieves the point cloud corresponding to the queried scene. The database includes both original scans and re-scans, where the same environment may be captured again after objects have been displaced.

Instance Retrieval In the instance retrieval task, the query consists of multiple RGB views of an object instance, each with its instance mask. The model must retrieve the corresponding 3D object point cloud from the instance point clouds of the scene the query was taken from. The task therefore evaluates fine-grained object matching within a known environment. Top-k accuracy is computed within each scene and averaged over all validation scenes.

## 4.3 Baselines

Our primary baseline is CrossOver [27]. We evaluate two publicly released CrossOver checkpoints for each task: one trained on the same datasets as our method, and another checkpoint of the same model trained on additional data, which we denote CrossOver<sup>†</sup>. To ensure that performance differences are not caused by the choice of 3D encoder, we also retrain CrossOver on its original preprocessed training data after replacing its 3D backbone with Utonia [41]; we refer to this variant as CrossOver (Updated). All methods are evaluated with CrossOver’s evaluation protocol for a fair comparison.

## 5 Results

## 5.1 Cross-Modal Scene Retrieval

Scene retrieval results are reported in Table 1. CrossWeave consistently outperforms all baselines on both ScanNet and 3RScan across all retrieval ranks. On ScanNet, CrossWeave achieves a Top-1 recall of 61.54, improving over CrossOver (Updated) by 21.16 points. The gains remain strong at higher ranks, where CrossWeave reaches 99.36 Top-5 recall and perfect Top-10 recall. These results indicate that the correct scene is almost always ranked among the first few retrieved candidates.

Table 1: Comparison of scene retrieval performance on ScanNet and 3RScan. <sup>†</sup> Trained on additional datasets (ArkitScenes [2] + MultiScan [18]). Best and Second-best results.
<table><tr><td rowspan="3"></td><td colspan="3">ScanNet</td><td colspan="3">3RScan</td></tr><tr><td colspan="3">Scene Recall</td><td colspan="3">Scene Recall</td></tr><tr><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-1</td><td>Top-5</td><td>Top-10</td></tr><tr><td>Scene CrossOver</td><td>22.44</td><td>61.86</td><td>75.32</td><td>19.11</td><td>58.60</td><td>77.07</td></tr><tr><td>Scene CrossOver†</td><td>19.87</td><td>58.01</td><td>74.36</td><td>17.83</td><td>51.59</td><td>73.25</td></tr><tr><td>Scene CrossOver (Updated)</td><td>40.38</td><td>90.06</td><td>96.47</td><td>23.57</td><td>66.88</td><td>84.71</td></tr><tr><td>Ours</td><td>61.54</td><td>99.36</td><td>100.00</td><td>46.50</td><td>85.99</td><td>92.36</td></tr></table>

Table 2: Comparison of within-scene instance retrieval performance on ScanNet and 3RScan. <sup>†</sup> Trained on additional datasets (ArkitScenes [2] + MultiScan [18]). Best and Second-best results.

<table><tr><td></td><td colspan="2">Instance Retrieval</td><td colspan="3">Scene-Level Recall</td></tr><tr><td>Method</td><td>Top-1</td><td>Top-3</td><td>R@25%</td><td>R@50%</td><td>R@75%</td></tr><tr><td>Instance CrossOver</td><td>63.58</td><td>88.71</td><td>96.79</td><td>80.77</td><td>31.09</td></tr><tr><td>Instance CrossOver†</td><td>63.42</td><td>88.55</td><td>96.79</td><td>80.77</td><td>31.09</td></tr><tr><td>Instance CrossOver (Updated)</td><td>70.94</td><td>92.39</td><td>99.36</td><td>88.14</td><td>49.68</td></tr><tr><td>Ours</td><td>78.77</td><td>94.05</td><td>98.40</td><td>88.78</td><td>67.95</td></tr></table>

<table><tr><td></td><td colspan="2">Instance Retrieval</td><td colspan="3">Scene-Level Recall</td></tr><tr><td>Method</td><td>Top-1</td><td>Top-3</td><td>R@25%</td><td>R@50%</td><td>R@75%</td></tr><tr><td>Instance CrossOver</td><td>58.88</td><td>84.24</td><td>90.45</td><td>71.97</td><td>28.03</td></tr><tr><td>Instance CrossOver†</td><td>57.09</td><td>84.58</td><td>89.81</td><td>63.06</td><td>24.84</td></tr><tr><td>Instance CrossOver (Updated)</td><td>66.80</td><td>90.85</td><td>97.45</td><td>87.90</td><td>35.03</td></tr><tr><td>Ours</td><td>73.50</td><td>92.73</td><td>98.09</td><td>92.99</td><td>49.68</td></tr></table>

The improvement is particularly meaningful because CrossOver (Updated) uses the same Utonia backbone. This suggests that the performance gain does not simply come from stronger 3D encoding, but from our training strategy based on sparse local patch-point supervision. By aligning local image patches with corresponding 3D points, CrossWeave learns descriptors that preserve fine-grained geometric and visual correspondences. This helps disambiguate scenes with similar global layouts, such as corridors, bedrooms, or repeated furniture arrangements, where purely global descriptors can collapse visually similar environments.

On 3RScan, CrossWeave shows the same trend. It achieves a Top-1 recall of 46.50, compared to 23.57 for CrossOver (Updated), corresponding to a 22.93 point improvement. CrossWeave also improves Top-5 recall from 66.88 to 85.99 and Top-10 recall from 84.71 to 92.36. These results demonstrate that local patch-point alignment generalizes across datasets and remains effective under the more challenging scene variations present in 3RScan. We also observe that CrossOver<sup>†</sup> does not improve over the base CrossOver model, suggesting that additional training data alone is insufficient without a training objective that explicitly encourages cross-modal local alignment.

Figure 2 shows a representative qualitative example on ScanNet. Given a set of query images, CrossWeave retrieves the correct 3D scene at rank 1, whereas both CrossOver baselines fail to retrieve the ground-truth scene within the top five results. This behavior is consistent across different scene categories: even when the query images cover only partial or localized views, CrossWeave produces descriptors that remain discriminative enough to separate visually similar scenes. Additional qualitative examples are provided in Appendix F, and failure cases are analyzed in Appendix G.

## 5.2 Cross-Modal Instance Retrieval

Instance retrieval results are reported in Table 2. Each query instance is retrieved from the instances of its own scene.

On ScanNet, CrossWeave achieves the best instance retrieval performance, with a Top-1 recall of 78.77 compared to 70.94 for CrossOver (Updated), an improvement of 7.83 points. CrossWeave also obtains the best Top-3 recall, improving from 92.39 to 94.05. Local patch-point supervision therefore improves fine-grained instance discrimination as well as scene matching.

Across scenes on ScanNet, the difference between methods becomes more pronounced at stricter recall thresholds. At R@25%, CrossWeave and CrossOver (Updated) are both near saturation, achieving 98.40 and 99.36, respectively. At R@50%, CrossWeave obtains the best result with 88.78, slightly outperforming CrossOver (Updated). The largest gain appears at R@75%, where CrossWeave reaches 67.95 compared to 49.68 for CrossOver (Updated), an improvement of 18.27 points. This suggests that CrossWeave is especially effective when the task requires retrieving a larger fraction of all correct instances, rather than only identifying the easiest matches.

![](images/068d0671e7a14694ddde1c7dd92ac651d8679dfda953d2968b015915eb6616b0.jpg)  
Figure 2: Qualitative comparison of cross-modal scene retrieval on ScanNet. Given a set of query RGB images, each model retrieves the most similar 3D scenes from the database. The groundtruth scene is shown on the left, and retrieved scenes are ordered by similarity from left to right. CrossOver and CrossOver (Updated) fail to retrieve the correct scene within the top five results, while CrossWeave retrieves it at rank 1. The remaining retrieved scenes are also semantically consistent, indicating that CrossWeave learns descriptors that capture both local cross-modal correspondences and global scene context.

The same pattern is observed on 3RScan. CrossWeave achieves the best instance retrieval performance, with a Top-1 recall of 73.50 and a Top-3 recall of 92.73, outperforming CrossOver (Updated) by 6.70 and 1.88 points, respectively. CrossWeave also obtains the best scene-level recall at all thresholds, reaching 98.09 at R@25%, 92.99 at R@50%, and 49.68 at R@75%. The improvement at R@75% is especially notable, where CrossWeave improves over CrossOver (Updated) by 14.65 points.

Overall, the instance retrieval results support the same conclusion as the scene retrieval experiments: sparse local patch-point alignment provides a stronger cross-modal training signal than instance-level supervision alone. While the baselines are often sufficient to retrieve a small number of easy matches, CrossWeave is more reliable under stricter retrieval criteria, where the model must recover a broader and more complete set of correct instances. Figure 3 shows two examples on ScanNet, where CrossWeave ranks the correct instance first among 16 and 26 candidates in the scene. Results with noisy 2D and 3D instance masks are reported in Appendix D.

## 5.3 Single View Scene Retrieval

Table 3 compares CrossWeave with SceneGraphLoc [19] and CrossOver [27] on 3RScan, where each query image is matched against N candidate scenes, following the evaluation protocol of SceneGraphLoc. SceneGraphLoc uses ground-truth scene graphs at inference, whereas CrossWeave uses only a single query image and the raw point cloud of each candidate scene. With 10 candidates, CrossWeave matches SceneGraphLoc at Top-1 (53.60) and improves Top-5 recall from 81.90 to 93.10 and Top-10 recall from 92.80 to 100.00. Recall drops for all methods when the pool grows to 50 candidates. In this setting, CrossWeave is 2.70 points behind SceneGraphLoc at Top-1 (27.50 vs. 30.20), but leads by 10.00 points at Top-5 and 15.30 points at Top-10. CrossWeave also outperforms CrossOver in every setting, by between 7.60 and 24.88 points. Since CrossWeave needs no scene graph annotations at inference, it can be deployed where such annotations are not available.

## 5.4 Ablation Studies

We ablate the main design choices of CrossWeave in Table 4. Specifically, we study the effect of the local patch-point alignment loss ${ \mathcal { L } } _ { p 2 p } ,$ , the image-point-cloud aggregation loss $\mathcal { L } _ { i 2 p c }$ , the aggregation used during training (mean pooling or SALAD), and the choice of retrieval features. We compare three feature variants: SALAD, which uses the aggregated descriptor produced by the SALAD module; LOCAL, which mean-pools the patch and point features; and HYBRID, which uses SALAD features for scene retrieval and LOCAL features for instance retrieval.

![](images/26c9096a4bea451346432c401f26283a9105081b001507c1c4dd714458232837.jpg)  
Figure 3: Qualitative instance retrieval on ScanNet. For each query view with its instance mask (left), CrossWeave ranks the instances of the same scene (middle) by cosine similarity. The Top-3 retrievals are shown with their similarity scores, and the correct match is outlined in green.

Table 3: Scene retrieval on 3RScan against SceneGraphLoc and CrossOver. Each query image is ranked against N candidate scenes; Top-k: percentage of queries with the correct scene in the top k. <sup>‡</sup> Uses ground-truth scene graphs at inference. Best and second-best results are highlighted.
<table><tr><td></td><td colspan="3">10 candidates</td><td colspan="3">50 candidates</td></tr><tr><td>Method</td><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-1</td><td>Top-5</td><td>Top-10</td></tr><tr><td>SceneGraphLoc‡ [19]</td><td>53.60</td><td>81.90</td><td>92.80</td><td>30.20</td><td>50.20</td><td>61.20</td></tr><tr><td>CrossOver [27]</td><td>46.00</td><td>77.97</td><td>90.58</td><td>18.69</td><td>39.16</td><td>51.62</td></tr><tr><td>Ours</td><td>53.60</td><td>93.10</td><td>100.00</td><td>27.50</td><td>60.20</td><td>76.50</td></tr></table>

Local patch-point alignment improves both scene and instance retrieval Comparing the SALAD baseline trained only with $\mathcal { L } _ { i 2 p c }$ against CrossWeave trained with both $\mathcal { L } _ { i 2 p c }$ and $\mathcal { L } _ { p 2 p }$ shows the benefit of local cross-modal supervision. On ScanNet, using SALAD features, CrossWeave improves scene Top-1 recall from 57.37 to 61.54, instance Top-1 recall from 73.33 to 77.04, and instance R@75% from 56.09 to 66.67. A similar trend is observed on 3RScan, where scene Top-1 recall improves from 43.95 to 46.50, instance Top-1 recall from 66.98 to 70.01, and instance R@75% from 38.22 to 43.31. These gains indicate that sparse local patch-point correspondences provide a stronger training signal than global image-point-cloud supervision alone, especially for fine-grained instance retrieval.

Local alignment alone is not sufficient for scene retrieval Training with only $\mathcal { L } _ { p 2 p }$ and meanpooled LOCAL features already gives strong instance retrieval, with 78.96 and 73.30 instance Top-1 recall on ScanNet and 3RScan. Scene retrieval, however, drops to 47.12 and 27.39 Top-1 recall, compared to 61.54 and 46.50 for the main model. Local alignment therefore learns discriminative object-level features, but does not by itself produce descriptors that separate scenes. Adding $\mathcal { L } _ { i 2 p c }$ while keeping mean pooling improves scene Top-1 recall from 47.12 to 52.24 on ScanNet and from 27.39 to 30.57 on 3RScan, so the aggregated objective provides supervision that the local loss does not. Replacing mean pooling with SALAD further improves scene Top-1 recall to 61.54 and 46.50. Hence, both the aggregation objective and the aggregation module contribute to scene retrieval. Mean pooling with both losses gives the highest instance Top-1 recall (79.63 and 73.62), slightly above the main model (78.77 and 73.50), but its scene Top-1 recall is 9.30 and 15.93 points lower.

Table 4: Ablation of training losses, aggregation and retrieval features. Agg.: aggregation used during training. Feats: SALAD (aggregated descriptors), LOCAL (mean-pooled patch and point features), or HYBRID (SALAD for scene, LOCAL for instance retrieval). Best and second-best results are highlighted; the main model is shaded and marked with <sup>⋆</sup>.
<table><tr><td colspan="5"></td><td colspan="3">ScanNet</td><td colspan="3">3RScan</td></tr><tr><td></td><td></td><td></td><td></td><td>Scene</td><td colspan="2">Instance</td><td>Scene</td><td colspan="2">Instance</td></tr><tr><td>Agg.</td><td> $\mathcal { L } _ { i 2 p c }$ </td><td> $\mathcal { L } _ { p 2 p }$ </td><td>Feats</td><td>Top-1</td><td>Top-1</td><td>R@75%</td><td>Top-1</td><td>Top-1</td><td>R@75%</td></tr><tr><td>Mean</td><td>x</td><td>√</td><td>LOCAL</td><td>47.12</td><td>78.96</td><td>70.51</td><td>27.39</td><td>73.30</td><td>47.13</td></tr><tr><td>Mean</td><td>√</td><td>√</td><td>LOCAL</td><td>52.24</td><td>79.63</td><td>71.15</td><td>30.57</td><td>73.62</td><td>49.04</td></tr><tr><td>SALAD</td><td>√</td><td>x</td><td>SALAD</td><td>57.37</td><td>73.33</td><td>56.09</td><td>43.95</td><td>66.98</td><td>38.22</td></tr><tr><td>SALAD</td><td>√</td><td>x</td><td>LOCAL</td><td>38.14</td><td>73.31</td><td>52.88</td><td>19.75</td><td>67.43</td><td>36.31</td></tr><tr><td>SALAD</td><td>√</td><td>√</td><td>SALAD</td><td>61.54</td><td>77.04</td><td>66.67</td><td>46.50</td><td>70.01</td><td>43.31</td></tr><tr><td>SALAD</td><td>√</td><td>√</td><td>LOCAL</td><td>48.40</td><td>78.77</td><td>67.95</td><td>28.66</td><td>73.50</td><td>49.68</td></tr><tr><td>SALAD</td><td>√</td><td>√</td><td>HYBRID*</td><td>61.54</td><td>78.77</td><td>67.95</td><td>46.50</td><td>73.50</td><td>49.68</td></tr></table>

Aggregated features are better suited for scene retrieval For scene-level retrieval, SALAD features consistently outperform LOCAL features. Without ${ \mathcal { L } } _ { p 2 p } ,$ the SALAD baseline achieves 57.37 Top-1 scene recall on ScanNet, compared to 38.14 when using local features. On 3RScan, the gap is even larger, with SALAD achieving 43.95 compared to 19.75 for LOCAL. The same pattern holds for CrossWeave: SALAD features achieve 61.54 scene Top-1 recall on ScanNet and 46.50 on 3RScan, whereas LOCAL features achieve only 48.40 and 28.66, respectively. This confirms that scene retrieval benefits from a parametric aggregation module over mean-pooled local features.

Local features are better suited for instance retrieval In contrast, instance retrieval benefits more from preserving local feature structure. For CrossWeave on ScanNet, using LOCAL features improves instance Top-1 recall from 77.04 to 78.77 and R@75% from 66.67 to 67.95 compared to SALAD features. On 3RScan, the improvement is larger: instance Top-1 recall increases from 70.01 to 73.50, and R@75% improves from 43.31 to 49.68. This suggests that aggregation can discard fine-grained spatial and geometric cues that are useful for distinguishing object instances, whereas local descriptors retain the detailed correspondence information required for accurate instance matching.

The hybrid feature strategy gives the best overall trade-off The HYBRID configuration combines the strengths of both feature types by using SALAD descriptors for scene retrieval and LOCAL for instance retrieval. This yields the best trade-off between scene and instance retrieval across both datasets, and we therefore use this configuration as our main model.

## 6 Conclusion

In this work, we introduced CrossWeave, a novel framework for cross-modal scene and instance retrieval that demonstrates the superiority of sparse, local feature alignment over traditional global scene encoding. By pairing a local patch-to-point contrastive objective with optimal transportbased aggregation, CrossWeave achieves state-of-the-art scene and instance retrieval on ScanNet and 3RScan with a single model, trained without scene- or instance-level contrastive objectives. CrossWeave provides a flexible alternative for bridging the modality gap in spatial reasoning tasks.

## Acknowledgments and Disclosure of Funding

Aadith Warrier is supported by the Reliance Foundation Postgraduate Scholarship.

## References

[1] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef Sivic. Netvlad: Cnn architecture for weakly supervised place recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5297–5307, 2016.

[2] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021.

[3] Min Cao, Shiping Li, Juntao Li, Liqiang Nie, and Min Zhang. Image-text retrieval: A survey on recent research and development. arXiv preprint arXiv:2203.14713, 2022.

[4] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.

[5] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017.

[6] Yu Duan, Yun Xiong, Yao Zhang, Yuwei Fu, and Yangyong Zhu. Hsgmp: Heterogeneous scene graph message passing for cross-modal retrieval. In Proceedings of the 2021 International Conference on Multimedia Retrieval, pages 82–91, 2021.

[7] Y. Eldar, M. Lindenbaum, M. Porat, and Y.Y. Zeevi. The farthest point strategy for progressive image sampling. IEEE Transactions on Image Processing, 6(9):1305–1315, 1997. doi: 10.1109/83.623193.

[8] Yanglin Feng, Yang Qin, Dezhong Peng, Hongyuan Zhu, Xi Peng, and Peng Hu. Pointcloud-text matching: Benchmark dataset and baseline. IEEE Transactions on Multimedia, 2025.

[9] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15180–15190, June 2023.

[10] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, and Pheng-Ann Heng. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following, 2023. URL https://arxiv.org/abs/2309.00615.

[11] Sergio Izquierdo and Javier Civera. Optimal transport aggregation for visual place recognition. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 17658–17668, 2024.

[12] Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael Bernstein, and Li Fei-Fei. Image retrieval using scene graphs. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3668–3678, 2015.

[13] Hervé Jégou, Matthijs Douze, Cordelia Schmid, and Patrick Pérez. Aggregating local descriptors into a compact image representation. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 3304–3311, 2010. doi: 10.1109/CVPR.2010.5540039.

[14] Nikhil Keetha, Avneesh Mishra, Jay Karhade, Krishna Murthy Jatavallabhula, Sebastian Scherer, Madhava Krishna, and Sourav Garg. Anyloc: Towards universal visual place recognition. IEEE Robotics and Automation Letters, 9(2):1286–1293, 2023.

[15] Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao Su. Openshape: Scaling up 3d shape representation towards open-world understanding. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 44860–44879. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 8c7304e77c832ddc70075dfee081ca6c-Paper-Conference.pdf.

[16] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[17] Haoyu Lu, Yuqi Huo, Mingyu Ding, Nanyi Fei, and Zhiwu Lu. Cross-modal contrastive learning for generalizable and efficient image-text retrieval. Machine Intelligence Research, 20(4):569–582, 2023.

[18] Yongsen Mao, Yiming Zhang, Hanxiao Jiang, Angel Chang, and Manolis Savva. Multiscan: Scalable rgbd scanning for 3d environments with articulated objects. Advances in neural information processing systems, 35:9058–9071, 2022.

[19] Yang Miao, Francis Engelmann, Olga Vysotska, Federico Tombari, Marc Pollefeys, and Dániel Béla Baráth. Scenegraphloc: Cross-modal coarse visual localization on 3d scene graphs. In European Conference on Computer Vision, pages 127–150. Springer, 2024.

[20] Kevin Musgrave, Serge J. Belongie, and Ser-Nam Lim. Pytorch metric learning. ArXiv, abs/2008.09164, 2020.

[21] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[22] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[23] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

[24] Sai Shubodh Puligilla, Mohammad Omama, Husain Zaidi, Udit Singh Parihar, and Madhava Krishna. Lip-loc: Lidar image pretraining for cross-modal localization. In 2024 IEEE/CVF Winter Conference on Applications ofComputer Vision Workshops (WACVW), pages 939–948. IEEE, 2024.

[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 18–24 Jul 2021. URL https://proceedings. mlr.press/v139/radford21a.html.

[26] Junlong Ren and Hao Wang. Enhanced cross-modal 3d retrieval via tri-modal reconstruction. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2025.

[27] Sayan Deb Sarkar, Ondrej Miksik, Marc Pollefeys, Daniel Barath, and Iro Armeni. Crossover: 3d scene cross-modal alignment. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8985–8994, June 2025.

[28] Konstantin Schall, Kai Uwe Barthel, Nico Hezel, and Klaus Jung. Optimizing clip models for image retrieval with maintained joint-embedding alignment. In International Conference on Similarity Search and Applications, pages 97–110. Springer, 2024.

[29] Chuan Tang, Xi Yang, Bojian Wu, Zhizhong Han, and Yi Chang. Parts2words: Learning joint embedding of point clouds and texts by bidirectional matching between parts and words. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6884–6893, 2023.

[30] Mikaela Angelina Uy and Gim Hee Lee. Pointnetvlad: Deep point cloud based retrieval for large-scale place recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[31] Kavisha Vidanapathirana, Milad Ramezani, Peyman Moghadam, Sridha Sridharan, and Clinton Fookes. Logg3d-net: Locally guided global descriptor learning for 3d place recognition. In 2022 International Conference on Robotics and Automation (ICRA), pages 2215–2221, 2022. doi: 10.1109/ICRA46639.2022. 9811753.

[32] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Niessner. Rio: 3d object instance re-localization in changing indoor environments. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.

[33] Suping Wang, Fei Zhou, Ming Yang, Lei Shi, and Chaohong Tan. Sgg-mvar: Cross-modal retrieval with scene graph generation and multiview attribute relationship guidance. IEEE Transactions on Computational Social Systems, 12(5):3671–3683, 2025.

[34] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 4840–4851, 2024.

[35] Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1179–1189, June 2023.

[36] Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip-2: Towards scalable multimodal pre-training for 3d understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27091–27101, June 2024.

[37] Chongjian Yuan, Jiarong Lin, Zheng Liu, Hairuo Wei, Xiaoping Hong, and Fu Zhang. Btc: A binary and triangle combined descriptor for 3-d place recognition. IEEE Transactions on Robotics, 40:1580–1599, 2024. doi: 10.1109/TRO.2024.3353076.

[38] Yihan Zeng, Chenhan Jiang, Jiageng Mao, Jianhua Han, Chaoqiang Ye, Qingqiu Huang, Dit-Yan Yeung, Zhen Yang, Xiaodan Liang, and Hang Xu. Clip2: Contrastive language-image-point pretraining from real-world point cloud data. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15244–15253, June 2023.

[39] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong. Faster segment anything: Towards lightweight sam for mobile applications. arXiv preprint arXiv:2306.14289, 2023.

[40] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8552–8562, June 2022.

[41] Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, and Hengshuang Zhao. Utonia: Toward one encoder for all point clouds, 2026. URL https: //arxiv.org/abs/2603.03283.

[42] Shuhang Zheng, Yixuan Li, Zhu Yu, Beinan Yu, Si-Yuan Cao, Minhang Wang, Jintao Xu, Rui Ai, Weihao Gu, Lun Luo, et al. I2p-rec: Recognizing images on large-scale point cloud maps through bird’s eye view projections. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1395–1400. IEEE, 2023.

[43] Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. arXiv preprint arXiv:2310.06773, 2023.

[44] Xiangyang Zhu, Renrui Zhang, Bowei He, Ziyu Guo, Ziyao Zeng, Zipeng Qin, Shanghang Zhang, and Peng Gao. Pointclip v2: Prompting clip and gpt for powerful 3d open-world learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2639–2650, October 2023.

## Appendix

## A Data Pre-processing and Training Optimization Details

Both datasets are preprocessed by temporally subsampling the RGB-D frames: one in every 30 frames for ScanNet and one in every 5 frames for 3RScan, chosen according to the typical sequence length of each dataset. This yields 63,683 training images from ScanNet and 56,883 from 3RScan, for a total of 120,566 training images. For scene-level evaluation we use the 312 ScanNet and 157 3RScan validation scans. For instance-level evaluation we use 6,419 ScanNet and 3,274 3RScan validation instances.

To reduce dataloader overhead, projected pixel-point correspondences are precomputed and cached as .pt files, while FPS downsampling and patch-point pair generation are performed online. Utonia point features are also cached to avoid redundant forward passes over the same point clouds during training.

## B Data Scaling

We study scene retrieval performance as a function of training data volume. The results reported in the main paper use 120,566 images from ScanNet and 3RScan as described in Appendix A. To study the effect of data scaling, we subsample the training set by uniform striding. As shown in Figure 4, CrossWeave is stable across data regimes: even at 10% of the full training set, the model outperforms the previous state of the art. We attribute the data efficiency of our model to two factors: feature-space FPS during patch-point pair selection, which ensures semantic diversity within each batch, and the use of strong pre-trained encoders for each modality, which provide rich local features from the outset. Together, these properties mean that even a small number of uniformly sampled images contains sufficient supervision signal to learn a discriminative cross-modal representation.

![](images/81f52a000cadb1a264da7fd7dfe973e9b949a28887af0468f72734cb98761177.jpg)  
Figure 4: Scene retrieval performance (Top-1 recall) on ScanNet and 3RScan as a function of training data volume. Performance is stable across data regimes, with even 10% of the full training set sufficient to outperform the previous state of the art.

## C Effect of Scaling Camera Views

Figure 5 reports scene retrieval performance as a function of the number of query camera views at inference. Top-1 recall improves from 45.83 to 64.10 on ScanNet and from 28.66 to 49.68 on 3RScan as the number of views increases from 3 to 30. Most of the gain comes between 3 and 10 views; beyond 10 views, recall still improves by up to 2.56 points on ScanNet and 4.46 points on 3RScan. This behavior is a direct consequence of the architecture: because the SALAD aggregator maps a variable number of input features into a fixed-dimensional descriptor, CrossWeave naturally incorporates additional views without any retraining or architectural modification. In contrast, CrossOver [27] averages per-frame features with equal weight, and its authors report that performance stabilizes after 10 frames, with additional frames giving only slight improvements.

CrossWeave continues to improve up to 25–30 views, since SALAD assigns patch features to clusters rather than averaging them. The ability to leverage more views at inference is particularly valuable in practical deployments, where a robot or AR device may capture varying numbers of observations depending on the environment.

![](images/a6cb0b622e8bfaedca1307371a6dacc15361fa2e02fd81bce89227e8bd0d7b5c.jpg)  
Figure 5: Scene retrieval performance (Top-1 recall) as a function of the number of query camera views. The main results use 10 views.

## D Performance with Degraded Instance Masks

The instance retrieval results in Table 2 use ground-truth instance masks for both the query images and the database point clouds. In deployment, these masks come from a segmentation model and contain errors. We therefore evaluate the main model, with LOCAL features for instance retrieval, under noisy 2D masks, noisy 3D masks, and both. For 2D, we generate masks with MobileSAM [39], prompted with a single point sampled well inside the ground-truth mask to mimic a human click, and keep the predicted mask with the highest IoU to the ground truth. For 3D, we corrupt each ground-truth instance point cloud in two ways: points from neighboring background and objects bleed in at the instance boundary, and contiguous chunks of the instance are dropped. We set the corruption so that the IoU degradation is similar to that of the 2D masks. Results for each mask condition are reported in Table 5.

Table 5: Instance retrieval under noisy instance masks, using the main model with LOCAL features.
<table><tr><td colspan="2"></td><td colspan="2">Instance Retrieval</td><td colspan="3">Scene-Level Recall</td></tr><tr><td rowspan="2">Dataset</td><td>Mask condition</td><td>Top-1</td><td>Top-3</td><td>R@25%</td><td>R@50%</td><td>R@75%</td></tr><tr><td>GT</td><td>78.77</td><td>94.05</td><td>98.40</td><td>88.78</td><td>67.95</td></tr><tr><td rowspan="4">ScanNet</td><td>2D noisy</td><td>75.75</td><td>93.02</td><td>97.44</td><td>88.14</td><td>62.50</td></tr><tr><td>3D noisy</td><td>75.33</td><td>92.91</td><td>97.12</td><td>87.50</td><td>63.46</td></tr><tr><td>2D + 3D noisy</td><td>72.29</td><td>91.86</td><td>97.12</td><td>85.90</td><td>55.45</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">3RScan</td><td>GT</td><td>73.50 68.00</td><td>92.73 89.95</td><td>98.09 97.45</td><td>92.99 88.54</td><td>49.68 34.39</td></tr><tr><td>2D noisy 3D noisy</td><td>65.65</td><td>88.79</td><td>96.82</td><td>84.08</td><td>32.48</td></tr><tr><td>2D + 3D noisy</td><td>61.28</td><td>86.11</td><td>95.54</td><td>76.43</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>26.75</td></tr></table>

## FPS Sampling Visualizations

![](images/618b8e2139e822e8a2c914132dec5d3c48097125db58274642874c80623f0ec1.jpg)  
(a) ScanNet scene0665\_00

![](images/6921216af001cac6ddddce523c64554b30a77418d2742a725d4ad16998a55b8d.jpg)  
(b) ScanNet scene0435\_02

![](images/c8ec0d8a7089711f99d681ac3f5a05faaf9960f410fc2a70b7a29f044c2fc0bf.jpg)  
(c) ScanNet scene0474\_00  
Figure 6: Visualizations for the effect of FPS sampling. Images with red dots denote random sampling. Images with blue dots denote FPS sampled points. It can clearly be observed that FPS enables us to avoid over-sampling on the often visually dominant but geometrically uninformative areas in the images such as walls, floors and other flat surfaces, thereby boosting feature diversity within each training batch. Note: We have visualized a large number of pixel-point pairs in these images; in our training framework we only use 32 patch-point pairs. Refer to Section 3.1 for more details.

## F Additional Visualizations for Qualitative Results

## F.1 ScanNet Visualizations

![](images/7a451918a8e46613e2db4dcf9b4a08ccdbb86d36dcaae27cefc0192373df5fde.jpg)  
Figure 7: Comparisons of Top-5 retrievals on a ScanNet room scene. Retrievals are ordered from most to least similar moving left to right, with correct matches highlighted in green. CrossWeave successfully retrieves the correct scene at rank 1, while CrossOver (Updated) retrieves it at rank 2, and the base model of CrossOver fails within the top 5.

![](images/4ef56a5b870ec293ea9ff7fc9a0187634116c909b9fa59118563c56fa243088d.jpg)  
CrossWeave (Ours)

Figure 8: Additional qualitative results on a ScanNet office scene. CrossWeave retrieves the ground truth point cloud as its top match, whereas both CrossOver (Updated) and CrossOver only identify the correct scene at rank 5.

## F.2 3RScan Visualizations

![](images/9fe2cb95181bef71e98269634262a5c0fe4abdfec4cc1e0ab04ae8e6f899a928.jpg)  
Figure 9: Qualitative results on a 3RScan dining area scene. CrossWeave successfully retrieves the ground truth point cloud as its top match, while CrossOver (Updated) ranks it fourth and the base CrossOver model fails to retrieve it within the top 5.

Ground Truth Scene  
![](images/2a1b59c5e4b552fa8b141045a772081af4e597ca521e8d3d81c34f4650cebdc1.jpg)  
CrossWeave (Ours)

Figure 10: Retrieval results on a 3RScan kitchen and dining scene. CrossWeave demonstrates superior discriminative ability by retrieving the exact scene at rank 1 from highly localized query images, whereas both baseline models completely fail to place the correct match within their top 5 retrievals.

## Ground Truth Scene

![](images/7e283733d7a90b073a23c1febfb639ece7d74da59732fbc52bd79416f8995a90.jpg)  
CrossWeave (Ours)

Figure 11: Additional results on a 3RScan kitchen scene. CrossWeave correctly identifies the target environment at rank 1, outperforming CrossOver, which places the match at rank 4, and CrossOver (Updated), which misses it entirely.

![](images/80b618b4dcffff646aff0005ce8274c5154be6e8a570e8ad3947fe5732c10a78.jpg)  
Figure 12: Qualitative instance retrieval on 3RScan. For each query view with its instance mask (left), CrossWeave ranks the instances of the same scene (middle) by cosine similarity. The Top-3 retrievals are shown with their similarity scores, and the correct match is outlined in green.

## G Failure Case Analysis

![](images/dd87cf70586e1882cc0c78da3370b2563b762310e43b91e17b3ba7db6ba57f7a.jpg)  
Figure 13: Failure case on a 3RScan living room scene. All three models miss the ground-truth scene in the top 5, as the retrieved scenes are visually and structurally similar to the query (a brown sectional sofa with a wooden coffee table). CrossWeave’s top matches still share the furniture arrangement (sofas and central coffee tables), whereas CrossOver (Updated) retrieves structurally different scenes.

![](images/8d4cc5d51e90a4da6a664470be55e2da657e3aa96e250535a4ec4911ce145d21.jpg)  
Figure 14: Instance retrieval failures on ScanNet and 3RScan. In both cases the query is a chair and the Top-1 retrieval (red) is another chair of the same type in the same scene, while the correct instance (green) is ranked second with a similar score.

## H Limitations

Despite strong performance on most benchmarks, CrossWeave has several limitations. First, at inference, CrossWeave assumes that query images are captured within a single viewing session; queries that span multiple sessions are not explicitly handled. Second, the training correspondences are derived from the camera intrinsics, camera poses, depth rendered from the reconstructed mesh, and the reconstruction itself. Errors in calibration or pose, missing or inaccurate rendered depth, and incomplete or noisy reconstructions produce incorrect or missing patch-point pairs. We use the reconstructions and poses provided with ScanNet and 3RScan and do not study the effect of such errors. Finally, instance retrieval requires segmented instances, both as query masks and as database point clouds, making retrieval quality dependent on the upstream segmentation, which is outside the scope of this work. Appendix D shows that retrieval degrades under noisy 2D and 3D masks, most strongly at R@75%.

## I Broader Impacts

CrossWeave enhances spatial intelligence for indoor robotics, autonomous navigation, and mixedreality systems through efficient cross-modal localization. However, by relying on pre-trained foundation models (DINOv2 and Utonia), it risks inheriting and propagating their latent training biases. This could result in disparate retrieval performance across culturally or socioeconomically diverse architectural styles, causing the system to underperform in underrepresented environments. Real-world deployments must actively audit these underlying models to mitigate biased spatial reasoning.