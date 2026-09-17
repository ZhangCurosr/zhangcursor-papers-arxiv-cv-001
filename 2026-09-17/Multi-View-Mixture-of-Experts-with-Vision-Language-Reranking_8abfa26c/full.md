# Multi-View Mixture-of-Experts with Vision-Language Reranking for Cross-View Object Geo-Localization

Xuyu Fan<sup>1</sup>, Qi Ming<sup>1,∗</sup>, Zhu Han<sup>1</sup>, Liuqian Wang<sup>2</sup>, Si-Yuan Cao<sup>3</sup>, Xiaohan Zhang<sup>3</sup>, Xudong Zhao<sup>4</sup>, Mingjing Zhao<sup>5</sup>, Yuhan Zhang<sup>6</sup> <sup>1</sup>College of Computer Science, Beijing University of Technology, <sup>2</sup>Zhengzhou University, <sup>3</sup>Zhejiang University, <sup>4</sup>Beijing Institute of Technology, <sup>5</sup>Beijing Electronic Science and Technology Institute, <sup>6</sup>Intelligent Science & Technology Academy of CASIC fanxy6522@mails.jlu.edu.cn, chaser.ming@gmail.com

## Abstract

Cross-view object geo-localization (CVOGL) locates a target in satellite imagery using drone or street-view queries. Existing methods train separate detectorsfor each viewpoint, leading to parameter redundancy and impeding cross-view knowledge sharing. Moreover, top-ranked satellite candidates are often visually similar, so visual appearance and categorical labels alone are insufficient to resolve such ambiguity. To address these, we propose MVLGeo, an efficient framework designed to unify multiple viewpoints and reduce model redundancy. First, we introduce environmental contextual text from the query view as cues to distinguish visually similar candidates via Vision-Language Reranking (VL-Rerank). Second, we design a multi-view Mixture-of-Experts architecture (MV-MoE) with a shared encoder and view-specific experts to reduce redundancy and promote knowledge sharing, while crossview contrastive learning aligns their representations for consistency. Third, we introduce an adaptive elliptical prior (ESAM-Prior) as auxiliary positional encoding for anisotropic geometric perception. Extensive experiments on the CVOGL benchmarks confirm that MVLGeo, as a unified model for multiple query viewpoints, achieves stateof-the-art performance, demonstrating robustness to input degradation and generalization across viewpoints. Code and models will be available on GitHub to facilitate future work.

## 1. Introduction

Cross-view object geo-localization (CVOGL) aims to pinpoint a target in satellite imagery using UAV or SVI (street-view imagery) queries, serving as a fundamental capability for UAV navigation in GPS-denied environments, real-time urban monitoring, and post-disaster situation assessment. To bridge the geometric gap between viewpoints, existing approaches address this task through two main paradigms. Image-level methods [10, 35, 43] retrieve the most similar satellite patch but only provide coarse location estimates. Recently, DetGeo [29] pioneered object-level CVOGL with click-prompted cross-view fusion, enabling direct bounding box regression from UAV to satellite imagery. Subsequent works further improved positional encoding [12, 24], detection architectures [17, 19], and output refinement [39]. More recent efforts have explored rotated bounding boxes for orientation awareness [9] and singlemodel localization that adapts to varying orientations and FoVs within street-view queries [3].

![](images/2e42f5cd50afcbdd415af249fe296b520467a537566f76d0d63b91881f474771.jpg)  
Figure 1. Illustration of the VL-Rerank framework. A VLM generates environmental context (e.g., road topology) from query images. These descriptions rerank satellite candidates, filtering out visually similar but contextually mismatched bounding boxes and identifying the correct target.

Despite these advances, current object-level CVOGL methods face fundamental limitations: (1) Candidate ambiguity. We observe that after detection and non-maximum suppression, the top-ranked satellite candidates are often visually indistinguishable from each other. Their confidence scores cluster within a narrow margin. As Fig. 1 shows, the flowerbeds of roundabouts appear as nearly identical green circular patches in satellite imagery. Category-level text labels [5, 40] cannot resolve this ambiguity, because they carry no spatial or topological context. This suggests that richer, spatially grounded textual cues could provide the missing disambiguation signal. (2) Inherent viewpoint non-transferability. Existing methods adopt a viewpointspecific training paradigm, where a separate detector is trained for each viewpoint. Such a paradigm achieves satisfactory performance only on the trained viewpoints, but degrades drastically on unseen ones. Different configurations thus require different models, even though visual features across views are largely shared. DetGeo [29], for example (illustrated in Fig. 2), deploys two separate models, each trained independently with its own parameters to handle a specific viewpoint. This per-viewpoint paradigm leads to near-linear growth in parameters and computational cost, making multi-domain deployment increasingly expensive and inflexible as the number of viewpoints grows. (3) Insufficient input prior. Most existing encodings are direction-agnostic. For elongated or irregular objects, isotropic priors spread activation into background regions, introducing noise that impairs precise localization. This problem is specific to object-level CVOGL, as in image-level cross-view geo-localization (CVGL), background context contributes to scene-level matching and is therefore benign, but CVOGL regresses a precise bounding box, so any prior activation that leaks into the background directly pulls the regression away from the target boundary. An anisotropic shape-aware prior would confine attention to the target region, yet such a prior is missing from current methods.

To tackle these challenges, we design a unified framework, MVLGeo. (1) We observe that environmental context, including road topology, building layout, and vegetation, constitutes an instance-level spatial fingerprint that remains discriminative when matching a query to its satellite counterpart. This motivates Vision-Language Reranking (VL-Rerank), where we apply prompt engineering to extract structured environmental descriptions from the query view, then fine-tune a VLM to compare and rerank candidates, providing a disambiguation signal beyond visual features and class labels. (2) Instead of training separate models for each viewpoint, we adopt the Mixture-of-

![](images/a716d96f8091f5771d07b6c05419e2dd2a5d82143c8c951118866e87f2b8e2a5.jpg)  
Figure 2. Comparison of the proposed MV-MoE and independent models. MV-MoE shares one encoder and two lightweight experts (UAV, SVI) for four input variants (UAV Vis, UAV Gray, SVI Vis, SVI Gray), reducing parameters by 38% while maintaining Acc@50 (%).

Experts paradigm [20] and design our Multi-View Mixtureof-Experts (MV-MoE) for parameter sharing rather than capacity scaling, where a shared encoder captures cross-view generic features, while lightweight view-specific experts learn only minimal view-specific offsets. Hard routing by view identity replaces learned gating, and a cross-view contrastive loss [2] prevents expert divergence. (3) We note that SAM [13] excels at isolating objects from backgrounds via point-prompted segmentation. We adapt this capability to CVOGL by fitting oriented ellipses to SAM masks, producing anisotropic Gaussian heatmaps (ESAM-Prior) that capture instance-specific shape and orientation.

Our contributions can be summarized as follows:

• We show that cross-view invariant environmental topology resolves candidate ambiguity where visual features and category labels fail. VL-Rerank introduces language modality through environmental textual descriptions, providing discriminative cues beyond visual appearance.

• MVLGeo is a unified multi-view framework for CVOGL that explicitly shares parameters across viewpoints via Mixture-of-Experts. Our MV-MoE architecture replaces per-view detectors with a shared encoder and identity-initialized view-specific experts, reducing total parameters while maintaining competitive accuracy across all viewpoints.

• We introduce an elliptical adaptive prior (ESAM-Prior) as a shape-aware positional encoding. By fitting an oriented ellipse to the target mask, this prior provides anisotropic geometric guidance for elongated and irregular targets.

## 2. Related Work

Cross-View Object Geo-Localization. DetGeo [29] pioneered object-level CVOGL with click-prompted crossview fusion, enabling direct bounding box regression from UAV or street-view queries to satellite imagery. Subsequent works improve positional encoding [12,24], detection architectures [17,19], and output refinement [39], with recent extensions to oriented boxes [9], multi-object scenarios [21], and single-stage frameworks [31]. These methods improve localization accuracy but remain viewpoint-specific and rely on isotropic positional priors, which are ill-suited for precise localization of elongated targets. SAM [13] has been applied in CVOGL for output-side refinement (TRO-Geo [39]) and mask-driven encoding (EDGeo [11]), but using SAM-derived anisotropic priors as input-level positional encoding remains unexplored.

![](images/a518b0789c361ae5513419bafec976d46d8b87eae56a671364c8d8af60a1e5ea.jpg)  
Figure 3. Schematic diagram of the MVLGeo network architecture. (A) Input-guided features are generated using SAM and shape priors. (B) Balancing heterogeneous input channels. (C–D) Using a Mixture-of-Experts (MoE) architecture with hard gating to perform cross-view feature extraction, fusion, and detection from UAV and SVI perspectives. (E) Aligning cross-view features during the training phase via InfoNCE contrastive learning. (F) Performing multi-image comparison and reranking by combining environmental semantic descriptions and image cropping to output the Top-1 matching result.

Mixture-of-Experts in Cross-View Geo-Localization. Classic MoE [25] scales model capacity through learned gating that sparsely activates experts, extended by GShard [14] and Switch Transformer [8] to large-scale models. Subsequently, the MoE paradigm has continued to evolve, with Expert Choice [42] inverting the routing direction, DeepSeekMoE [7] introducing shared experts with fine-grained segmentation, and Soft MoE [22] replacing discrete routing with soft assignments, broadening MoE into a more general conditional computation framework. Liu [20] and Wong [32] further formalize hard-routed MoE as routing each input to a single expert, providing a theoretical basis for our view-identity-based hard routing. In CVOGL, SMGeo [37] introduces grid-level sparse MoE for capacity expansion, and HiSymGeo [1] employs querygated multi-expert feature fusion. Concurrent efforts explore platform-level expert partitioning [16] and KV routing for cross-view alignment [36]. Most existing MoE applications in CVOGL adopt learned routing for capacity scaling, while unifying multiple query viewpoints within a single detection model remains largely unexplored.

Beyond MoE-based methods, SinGeo [3] targets orientation and FoV robustness within street-view queries through curriculum learning, while SkyPart [30] addresses visual degradation via invariant texture topology. These works reduce per-condition model replication but focus on robustness within a single viewpoint.

Vision-Language Models for Geo-Localization. CLIP [23], BLIP-2 [15], and InternVL [4] have enabled cross-view spatial reasoning through vision-language alignment. Existing text-guided methods fall into two categories. Category-level supervision associates object classes with visual features (GeoText [5], MoPT [40]), but class labels lack the spatial and topological detail needed to distinguish visually similar candidates. Scene-level methods leverage hierarchical descriptions for image matching (GeoMatch [34]), semantic-anchored multi-view models (GeoBridge [28]), or VLM-based reranking (GeoVLM [6], CLIP-UG [33]), while GeoX-Bench [41] provides a benchmark for evaluating such methods. However, existing methods either rely on coarse category labels or operate at the scene level. Using VLM-based comparative reasoning with viewpoint-invariant environmental descriptions for object-level candidate disambiguation remains open.

![](images/fda2b6bdf0c88f1b1241a0228f3f47727966914002ab758673f52148d214c043.jpg)

![](images/23fab3a7ee55d43fc2a7c06ba2220f33f8e9ba5f7b98e1a906a26e4bd0995727.jpg)  
Figure 4. Recall@K reveals that the ground-truth box often appears among top candidates but is not ranked first.

## 3. Methodology

## 3.1. Overview

We build upon DetGeo [29] as our baseline. MVLGeo improves the baseline from three perspectives: instanceadaptive anisotropic positional encoding for better geometric perception, a multi-view unified architecture that eliminates per-view parameter redundancy, and language-guided candidate disambiguation. As illustrated in Fig. 3 (stages A to F), these components form an end-to-end unified framework. Both SAM priors and VLM descriptions can be precomputed offline and cached, with online generation supported for new queries.

## 3.2. VL-Rerank: Vision-Language Reranking

To resolve candidate ambiguity, we introduce a VLMbased reranking stage that leverages environmental context from the query view. As observed in Fig. 4, the groundtruth box appears among the top-3 candidates in 78.3% of cases but is ranked first only 55.1%, confirming that visual appearance alone is insufficient for disambiguation.

## 3.2.1 Environmental Description Generation.

To ensure viewpoint invariance, descriptions must avoid absolute directions (e.g., north of) or viewpoint-specific attributes, instead capturing stable spatial relations (e.g., adjacent to, surrounded by) that hold from any perspective. We design four complementary prompts for Qwen-VL-Chat, covering road topology, building layout, parking lots, vegetation, and landmarks, with a red circular marker overlaid at the click point to direct attention (used purely for VLM, not detection). The VLM outputs a few factual sentences, which are cached for reuse.

## 3.2.2 VLM-Based Comparative Reranking.

We fine-tune InternVL [4] for multi-image comparative reasoning. For each query, the top-K satellite candidates are cropped, resized, color-bordered with numeric indices, and arranged into a composite image. This composite, paired with the environmental description, forms the multi-modal input. Training data is generated offline using the frozen DetGeo detector, and for each sample we composite the top-3 candidates and select the one with highest IoU to the ground truth as the target label. Data is formatted in the LLaMA-Factory chat template for next-token prediction fine-tuning. At inference, InternVL directly outputs the best candidate index, enabling end-to-end visual comparison without intermediate scoring functions.

## 3.3. MV-MoE: Multi-View Mixture-of-Experts Architecture

DetGeo trains independent models per viewpoint. Extending to V viewpoints would require V full detectors. Yet the underlying task is shared. Only the statistical distribution differs, not the visual concepts. This suggests most parameters can be shared, with minimal view-specific adaptations.

The MV-MoE architecture (C–D) in Fig. 3 operationalizes this via a shared-residual decomposition. At each expert insertion point ℓ, a view’s function is factorized as:

$$
f _ { v i e w } ^ { ( \ell ) } ( x ) = f _ { s h a r e d } ^ { ( \ell ) } ( x ) + \Delta _ { v i e w } ^ { ( \ell ) } ( x ) ,\tag{1}
$$

where $f _ { s h a r e d } ^ { ( \ell ) }$ captures generic capabilities jointly optimized across views, and $\Delta _ { v i e w } ^ { ( \ell ) }$ compensates for viewspecific discrepancies. Each residual is initialized near zero, so the model learns only minimal offsets from a sharingoptimal state. Our MV-MoE adopts hard routing by view identity instead of learned gating. This eliminates router overhead and unstable dynamics, which is particularly beneficial for our parameter-sharing objective:

$$
\operatorname* { m i n } _ { \theta _ { s h a r e d } , \{ \theta _ { v i e w } \} } \sum _ { v \in \{ U A V , S V I \} } ( f _ { s h a r e d } ( x _ { v } ) + \Delta _ { v } ( x _ { v } ) , y _ { v } ) .\tag{2}
$$

The complete training objective additionally incorporates cross-view contrastive alignment (Eq. 6).

We insert experts at two positions. The first is ResNet18 layer4 (C), which encodes high-level semantic features where cross-view discrepancies manifest while early layers (conv1 to layer3) encode viewpoint-invariant edges and textures, making them unsuitable for expert insertion:

$$
h _ { o u t } = \mathrm { l a y e r } 4 _ { s h a r e d } ( x ) + \Delta _ { v i e w } ( x ) .\tag{3}
$$

Concretely, the view-specific expert $\Delta _ { v i e w }$ at layer4 consists of two 3×3 convolutional layers with batch normalization and ReLU, matching the residual block design of

ResNet’s layer4. The final convolution weights are zeroinitialized and the final BN is initialized with $\gamma { = } 1 , \beta { = } 0$ ensuring $\Delta _ { v i e w } ( x ) = 0$ at initialization and thus identity mapping from the pretrained backbone. The detection head expert (D) consists of a single $1 \times 1$ convolution followed by batch normalization, reducing from 512 channels to 54, the YOLOv3 output dimension, with weights drawn from a zero-mean Gaussian $( \sigma { = } 1 0 ^ { - 4 } )$ to keep its initial contribution near zero. It operates on the fused feature f:

$$
d e t _ { o u t } = \mathrm { h e a d } _ { s h a r e d } ( \mathbf { f } ) + \alpha \cdot \Delta _ { v i e w } ( \mathbf { f } ) ,\tag{4}
$$

where α is a learnable scalar initialized to 1.0. Since the shared head is randomly initialized from scratch, the head expert uses Kaiming initialization with a small-weight final projection, keeping its initial contribution near zero. This design lets the view-specific expert adapt to viewpointspecific textures (e.g., roof contours vs. facades) while the head expert adjusts detection preferences.

## 3.3.1 Contrastive Feature Alignment.

The shared-residual decomposition (Eq. 1) ensures each expert learns only offsets, but without constraint, UAV and SVI experts could drift into incompatible feature spaces. The gradients of their detection losses on the shared encoder exhibit conflicting directions, causing oscillation rather than convergence.

To address this, we apply a symmetric InfoNCE contrastive loss (E) in Fig. 3 that pulls corresponding crossview features together while pushing non-matching pairs apart:

$$
\mathcal { L } _ { C L } = \frac { 1 } { 2 } \left[ \mathrm { I n f o N C E } ( z _ { U A V } , z _ { S V I } ) + \mathrm { I n f o N C E } ( z _ { S V I } , z _ { U A V } ) \right] .\tag{5}
$$

Here, $z _ { v i e w }$ is a normalized 128-dimensional embedding from a 2-layer MLP projection head (following Sim-CLR [2]). For a batch of B aligned UAV-SVI pairs, positives are $( z _ { U } ^ { i } , z _ { S } ^ { i } )$ and negatives are $( z _ { U } ^ { i } , z _ { S } ^ { j } ) , i \neq j$ . Gradient flow from $\mathcal { L } _ { C L }$ is blocked from the detection head experts $( \alpha \cdot \Delta )$ to preserve detection-level specialization, while updating the backbone and shared parameters to ensure well-aligned updates. The overall objective balances alignment and specialization:

$$
\mathcal { L } _ { t o t a l } = 0 . 5 ( \mathcal { L } _ { d e t } ^ { U A V } + \mathcal { L } _ { d e t } ^ { S V I } ) + \lambda \cdot \mathcal { L } _ { C L } .\tag{6}
$$

## 3.4. ESAM-Prior: SAM Adaptive Elliptical Prior

We propose an instance-adaptive anisotropic prior that captures the target’s shape and orientation at runtime, confining the positional signal to the target’s geometric footprint rather than spreading into background regions.

![](images/474088e2785c5cde6122e5331341a5f3af3fc7e815783a09cb2bf6d7ed3d8288.jpg)  
Figure 5. Shape prior generation pipeline. (a) Query image, (b) SAM mask and fitted ellipse, (c) Anisotropic Gaussian heatmap.

## 3.4.1 SAM-Guided Ellipse Fitting.

Given the query image $I _ { q u e r y }$ and click $( c _ { x } , c _ { y } )$ , we first obtain a segmentation mask via SAM [13] (ViT-H, point prompt). An oriented ellipse is then fitted to this mask to extract three parameters: major axis $\sigma _ { m a j o r }$ , minor axis $\sigma _ { m i n o r } ,$ and orientation θ. Unlike fixed isotropic Gaussians that treat all targets identically, these parameters naturally adapt to each instance. Compact vehicles produce nearcircular ellipses $( \sigma _ { m a j o r } { \approx } \sigma _ { m i n o r } )$ , while elongated structures produce highly anisotropic ones $( \sigma _ { m a j o r } \gg \sigma _ { m i n o r } )$ with explicit directional alignment.

## 3.4.2 Dual-Map Input Encoding.

The extracted parameters are rendered as an anisotropic Gaussian heatmap centered at the click point (see Fig. 5)

$$
P _ { e l l i p s e } ( x , y ) = \exp \left( - \frac { 1 } { 2 } \Bigg [ \Bigg ( \frac { x _ { r o t } } { \sigma _ { m a j o r } } \Bigg ) ^ { 2 } + \Bigg ( \frac { y _ { r o t } } { \sigma _ { m i n o r } } \Bigg ) ^ { 2 } \Bigg ] \right) ,\tag{7}
$$

where $( x _ { r o t } , y _ { r o t } ) \ = \ \mathbf { R } ( \theta ) ( x - c _ { x } , y - c _ { y } )$ . Unlike isotropic priors that spread equally in all directions, this heatmap concentrates activation along the object’s major axis and decays rapidly along the minor axis, confining the positional signal to the target’s actual geometric footprint.

When SAM segmentation is unreliable, we fall back to a fixed circular Gaussian prior centered at the click point. Alongside the elliptical heatmap, we retain a binary click map to provide a precise center anchor. The two maps are concatenated with the RGB channels and compressed via a 1×1 convolution before entering the shared encoder, allowing the network to jointly leverage visual appearance, positional anchor, and shape prior.

<table><tr><td rowspan="3">Method</td><td colspan="4">Ground→Satellite</td><td colspan="4">Drone→Satellite</td><td rowspan="3">|Models</td></tr><tr><td colspan="2">Validation</td><td colspan="2">Test</td><td colspan="2">Validation</td><td colspan="2">Test</td></tr><tr><td>Acc@25%Acc@50%</td><td></td><td>Acc@25% Acc@50%</td><td></td><td>Acc@25%Acc@50%</td><td></td><td>Acc@25% Acc@50%</td><td></td></tr><tr><td>CVM-Net [10]</td><td>5.09</td><td>0.87</td><td>4.73</td><td>0.51</td><td>20.04</td><td>3.47</td><td>20.14</td><td>3.29</td><td>2</td></tr><tr><td>RK-Net [18]</td><td>8.67</td><td>0.98</td><td>7.40</td><td>0.82</td><td>19.94</td><td>3.03</td><td>19.22</td><td>2.67</td><td>2</td></tr><tr><td>L2LTR [35]</td><td>12.24</td><td>1.84</td><td>10.69</td><td>2.16</td><td>38.68</td><td>5.96</td><td>38.95</td><td>6.27</td><td>2</td></tr><tr><td>Polar-SAFA [26]</td><td>19.18</td><td>2.71</td><td>20.66</td><td>3.19</td><td>36.19</td><td>6.39</td><td>37.41</td><td>6.58</td><td>2</td></tr><tr><td>TransGeo [43]</td><td>21.67</td><td>3.25</td><td>21.17</td><td>2.88</td><td>34.78</td><td>5.42</td><td>35.05</td><td>6.47</td><td>2</td></tr><tr><td>SAFA [27]</td><td>20.59</td><td>3.25</td><td>22.20</td><td>3.08</td><td>36.19</td><td>6.39</td><td>37.41</td><td>6.58</td><td>2</td></tr><tr><td>GeoDTR+ [38]</td><td>14.08</td><td>1.95</td><td>14.19</td><td>5.14</td><td>15.71</td><td>3.68</td><td>16.03</td><td>4.73</td><td>2</td></tr><tr><td>DetGeo [29]</td><td>46.70</td><td>43.99</td><td>45.43</td><td>42.24</td><td>59.81</td><td>55.15</td><td>61.97</td><td>57.66</td><td>2</td></tr><tr><td>VAGeo [17]</td><td>47.56</td><td>44.42</td><td>48.21</td><td>45.22</td><td>64.25</td><td>59.59</td><td>66.19</td><td>61.87</td><td>2</td></tr><tr><td>OCGNet [12]</td><td>48.54</td><td>44.20</td><td>51.49</td><td>47.69</td><td>66.52</td><td>61.86</td><td>68.35</td><td>63.93</td><td>2</td></tr><tr><td>MVLGeo (Ours)</td><td>48.32</td><td>45.18</td><td>50.05</td><td>47.24</td><td>69.09</td><td>63.11</td><td>73.72</td><td>66.04</td><td>1</td></tr></table>

Table 1. Comparison of Acc@25/50 (%) and number of models on the CVOGL dataset. Bold denotes the best results.

## 4. Experiments

## 4.1. Dataset and Evaluation Metrics

Dataset. CVOGL [29] is a large-scale cross-view object geo-localization dataset. It supports two tasks: Drone→Satellite using UAV images as queries, and Street-View→Satellite (also referred to as Ground→Satellite) using street-view images as queries. The UAV query images are 256×256, the street-view query images are 256×512, and the satellite reference images are 1024×1024. Each task is split into 4,343 training, 923 validation, and 973 test pairs.

Evaluation Metrics. Following DetGeo [29], we adopt Acc@50 (%) and Acc@25 (%) as evaluation metrics, where Acc@K measures the proportion of queries whose predicted bounding box achieves an IoU greater than K with the ground truth. We report results on both validation and test sets. In addition to localization accuracy, we report the total number of model parameters to assess parameter efficiency. Unlike prior methods that train two separate detectors, one for each viewpoint, our framework uses a single unified model for both UAV and SVI, enabling a direct comparison of total parameter cost. We further introduce the average Acc@50 (%) across different modalities and viewpoints as a new metric to evaluate cross-modal and cross-view generalization.

## 4.2. Implementation Details

All experiments are conducted on two NVIDIA RTX 4090 GPUs. We adopt the DetGeo [29] framework as our base architecture. In the VL-Rerank, environmental descriptions are generated by Qwen-VL-Chat with structured prompts, and InternVL is fine-tuned as a multi-image comparative reranker using the LLaMA-Factory chat template. In the MV-MoE, we adopt ResNet18 pretrained on ImageNet-1K as the shared encoder, with view-specific experts inserted at the encoder and detection head. The model is trained with RMSProp for 25 epochs with a batch size of 12, alternating UAV and SVI samples per step. The learning rate is initialized at $1 \times 1 0 ^ { - 4 }$ and decayed by a factor of 10 every 10 epochs. The contrastive learning weight λ is set to 0.1. In the ESAM-Prior, we apply SAM (ViT-H) to generate instance-level segmentation masks, which are fitted with oriented ellipses to produce the anisotropic Gaussian prior.

<table><tr><td>Method</td><td>Params</td><td>Models</td><td>Total</td></tr><tr><td>DetGeo</td><td>73.8M</td><td>2</td><td>147.6M</td></tr><tr><td>OCGNet</td><td>74.8M</td><td>2</td><td>149.6M</td></tr><tr><td>MVLGeo (Ours)</td><td>91.0M</td><td>1</td><td>91.0M</td></tr></table>

Table 2. Comparison of model parameters and deployment costs between existing dual-detector methods and our single unified model.

## 4.3. Comparison with State-of-the-Art Methods

Tab. 1 compares MVLGeo with existing CVOGL methods. Results of prior works are taken from their original papers. Our method achieves state-of-the-art performance on the Drone→Satellite (UAV) task and competitive accuracy on Ground→Satellite (street-view, SVI). More importantly, this is accomplished with only a single unified model for both viewpoints, whereas all prior methods require training and deploying two independent detectors, one per viewpoint.

To our knowledge, MVLGeo is the first framework to unify two distinct viewpoints (UAV and SVI) within a single detection model for object-level CVOGL. This architectural advantage is reflected in parameter efficiency and model count. As shown in Tab. 2, prior methods duplicate the entire detection, requiring twice the parameters of a single model for two viewpoints. MVLGeo unifies both tasks within a single model, reducing total model size by 38%. This unified design confirms that the vast majority of visual knowledge is shared across viewpoints, with only a small fraction of parameters requiring view-specific adaptation. This paradigm opens a new direction for scalable deployment across arbitrary viewpoints without linear parameter growth, simplifying deployment and reducing memory footprint.

<table><tr><td>VL- Rerank MoE</td><td>MV- ESAM-</td><td>Prior</td><td>UAV Acc@50 (%) Acc@50 (%)</td><td>SVI</td><td>Params</td></tr><tr><td>X</td><td>×</td><td>×</td><td>57.66</td><td>42.24</td><td>73.8M×2</td></tr><tr><td>X</td><td>X</td><td>√</td><td>62.58</td><td>46.76</td><td>73.8M×2</td></tr><tr><td>×</td><td>√</td><td>X</td><td>52.14</td><td>41.35</td><td>91.0M×1</td></tr><tr><td>X</td><td>√</td><td>√</td><td>56.63</td><td>44.09</td><td>91.0M×1</td></tr><tr><td>√</td><td>√</td><td>√</td><td>66.04</td><td>47.24</td><td>91.0M×1</td></tr></table>

Table 3. Ablation study of each component in MVLGeo on the UAV and SVI test sets.

![](images/341746142b10724e341c7cb55682117d59e3bd1f59f32c4d2515f1a852eb7196.jpg)  
Figure 6. Contributions of each component to the UAV and SVI test sets, and corresponding parameter relationships.

## 4.4. Ablation Studies

## 4.4.1 Component-wise Ablations.

To evaluate the MVLGeo framework, we conduct ablation studies on each component and perform multiple runs. The experimental results are summarized in Tab. 3 and Fig. 6. First, applying ESAM-Prior to the baseline [29] yields a 4.9% gain on UAV and 4.5% on SVI over the baseline. Incorporating MV-MoE reduces parameter redundancy and eliminates dual-model deployment, at the cost of a moderate accuracy drop that is consistently observed across configurations due to parameter sharing. On this basis, integrating VL-Rerank further enhances performance by 9.4% on UAV and 3.2% on SVI, pushing both tasks beyond the baseline. The full MVLGeo integrates all three components, demonstrating their effectiveness and complementarity.

<table><tr><td>#</td><td>CL λ</td><td>BN γ init</td><td>Val Best</td><td>Test Best</td></tr><tr><td>1</td><td>0.1</td><td>1.0 (identity)</td><td>49.29</td><td>50.36</td></tr><tr><td>2</td><td>0.1</td><td>Random</td><td>44.74</td><td>43.91</td></tr><tr><td>3</td><td>0.5</td><td>1.0</td><td>47.52</td><td>46.87</td></tr><tr><td>4</td><td>0.0</td><td>1.0</td><td>49.59</td><td>49.23</td></tr></table>

Table 4. Average Acc@50 (%) across UAV and SVI on validation and test sets under MV-MoE configurations.

![](images/6aea578daae07f780b6b3a10e3cc2cad496aa9c0b6358c3c4a11e481c9054c33.jpg)  
Figure 7. Cross-dataset localization accuracy distribution on UAV and SVI scenes. The yellow value line denotes the Pareto tradeoff boundary, and points above the line achieve superior overall performance.

## 4.4.2 Analysis of MV-MoE Configuration.

We systematically test inserting expert branches at different positions of the feature extraction layers. As shown in Fig. 7, inserting experts at Layer4 achieves the best cross-view trade-off balance, lying above the Pareto boundary. Layer4 encodes high-level semantics containing prominent cross-view discrepancies, making it suitable for expert modules. By contrast, shallow layers capture view-invariant low-level features, and we observe that inserting experts at these layers yields less than 0.5% improvement, confirming their limited benefit for domain-specific adaptation. The detection head further calibrates box regression and confidence outputs, complementing backbone feature adaptation.

We perform ablations on identity initialization and contrastive loss weight, as listed in Tab. 4. Identity initialization outperforms random initialization by 5.5% on average. Random initialization injects large gradients from untrained experts through residual connections, destabilizing the pretrained shared backbone from the first iteration. For the contrastive loss, λ = 0.1 achieves the best overall performance. Removing contrastive regularization widens the validation-test gap (overfitting), while excessive regularization imposes overly strict constraints and degrades final accuracy.

(b) Acc@50(%) Across Aspect Ratio Buckets  
![](images/f9cd108198719f9364a09634a8d2ec85d86b7ea15642cfe5d8d869217724c449.jpg)  
Figure 8. ESAM-Prior ablation results on UAV and SVI datasets. (a) Validation Acc@50 (%) over training epochs. (b) Acc@50 (%) across object aspect ratio buckets partitioned into Low, Medium and High groups.

## 4.4.3 Effects of ESAM-Prior.

We explore various heatmap encoding strategies from the single-view DetGeo baseline. Full ablation studies are performed on the UAV dataset. Substituting DetGeo’s distance map with a fixed isotropic Gaussian yields moderate gains. We further adopt a fixed elliptical prior, improving UAV further and confirming that anisotropic kernels benefit elongated targets. However, a fixed ellipse cannot adapt to varying instance shapes. Inspired by this, we propose ESAM-Prior to produce instance-adaptive elliptical priors, achieving the optimal result on UAV. We then verify generalization on the SVI dataset with three representative schemes, including DetGeo, fixed circular Gaussian, and ESAM-Prior, and consistent gains are obtained.

As shown in Fig. 8, ESAM-Prior consistently outperforms baselines under all conditions. Compared to isotropic priors that spread activation into background regions for elongated objects, ESAM-Prior produces tighter heatmaps along the target’s major axis, as shown in Fig. 5. Specifically, improvements are most prominent for elongated targets with aspect ratio of 3.0 or higher. In the High bucket, ESAM-Prior achieves the largest gains. Medium-bucket samples show marginal differences among all methods.

## 4.5. Extension to Other Input Modalities

The parameter-sharing property of MV-MoE generalizes effectively to grayscale inputs without retraining, as shown in Tab. 5. By training on visible-light data and applying the same unified model directly to grayscale at test time, MV-MoE retains competitive accuracy across all four modalityviewpoint combinations with minimal degradation. These results further demonstrate the broad modality robustness of the proposed unified architecture.

<table><tr><td>Test modality</td><td>Single baseline MV-MoE unified Retention</td><td></td><td></td></tr><tr><td>UAV Vis (UV)</td><td>62.6</td><td>56.6</td><td>90.4%</td></tr><tr><td>UAV Gray (UG)</td><td>62.4</td><td>54.5</td><td>87.3%</td></tr><tr><td>SVI Vis (SV)</td><td>46.8</td><td>44.0</td><td>94.0%</td></tr><tr><td>SVI Gray (SG)</td><td>45.9</td><td>44.1</td><td>96.1%</td></tr><tr><td>4-View Avg</td><td></td><td>49.8</td><td></td></tr></table>

Table 5. Unified MV-MoE generalizes across four modalityviewpoint combinations with minimal accuracy degradation under grayscale inputs.

## 5. Conclusion

In this paper, we presented MVLGeo, a unified multiview framework for cross-view object geo-localization. MVLGeo addresses parameter redundancy through sharedresidual Mixture-of-Experts, resolves candidate ambiguity via vision-language reranking with environmental context, and provides shape-aware spatial encoding through instance-adaptive elliptical priors. Extensive experiments demonstrate that MVLGeo surpasses existing dual-detector methods with a single unified model and generalizes robustly to unseen modalities without retraining. These results open up new possibilities for scalable multi-view deployment in geo-localization.

Limitations. The MVLGeo architecture has not been extended beyond the CVOGL benchmark. Environmental descriptions generated from street-view queries suffer from perspective distortion, reducing the reranking gain on this task. Prompt engineering remains manual.

## References

[1] Cuiqun Chen, Qi Chen, Mang Ye, and Xingyi Zhang. HiSymGeo: Hierarchical context symbiosis for cross-view object-level image geo-localization. IEEE Transactions on Image Processing, 35:6401–6415, 2026. 3

[2] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. Simclr: A simple framework for contrastive learning. In Proceedings of the International Conference on Machine Learning (ICML), 2020. 2, 5

[3] Yang Chen, Xieyuanli Chen, Junxiang Li, Jie Tang, and Tao Wu. SinGeo: Unlock single model’s potential for robust cross-view geo-localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 2, 3

[4] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3, 4

[5] Meng Chu, Zhedong Zheng, Wei Ji, Tingyu Wang, and Tat-Seng Chua. Towards natural language-guided drones: GeoText-1652 benchmark with spatial relation matching. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. arXiv:2311.12751. 2, 3

[6] Barkin Dagda, Muhammad Awais, and Saber Fallah. GeoVLM: Improving automated vehicle geolocalisation using vision-language matching, 2025. 3

[7] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. DeepSeek-MoE: Towards ultimate expert specialization in mixture-ofexperts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024. 3

[8] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23:1–39, 2022. 3

[9] Chenlin Fu, Ao Gong, and Yingying Zhu. From horizontal to rotated: Cross-view object geo-localization with orientation awareness, 2026. CVPR 2026 Findings. 2, 3

[10] Sixing Hu, Mengdan Feng, Rang M. H. Nguyen, and Gim Hee Lee. Cvm-net: Cross-view matching network for image-based ground-to-aerial geo-localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7258–7267, 2018. 1, 6

[11] Shuhan Hu, Yiru Li, Yuanyuan Li, and Yingying Zhu. Seeing the unseen: Mask-driven positional encoding and stripconvolution context modeling for cross-view object geolocalization, 2025. 3

[12] Zheyang Huang, Jagannath Aryal, Saeid Nahavandi, Xuequan Lu, Chee Peng Lim, Lei Wei, and Hailing Zhou. Object-level cross-view geo-localization with location enhancement and multi-head cross attention, 2025. 1, 3, 6

[13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and´ Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2, 3, 5

[14] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling giant models with conditional computation and automatic sharding. In Proceedings of the International Conference on Learning Representations (ICLR), 2021. 3

[15] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Proceedings of the International Conference on Machine Learning (ICML), 2023. 3

[16] LinFeng Li, Jian Zhao, Zepeng Yang, Yuhang Song, Bojun Lin, Tianle Zhang, Yuchen Yuan, Chi Zhang, and Xuelong Li. A parameter-efficient mixture-of-experts framework for cross-modal geo-localization, 2025. 3

[17] Zhongyang Li, Xin Yuan, Wei Liu, and Xin Xu. Vageo: View-specific attention for cross-view object geo-localization. In Proceedings of ICASSP, 2025. arXiv:2501.07194. 1, 3, 6

[18] Jinliang Lin, Zhedong Zheng, Zhun Zhong, Zhiming Luo, Shaozi Li, Yi Yang, and Nicu Sebe. Joint representation learning and keypoint detection for cross-view geolocalization. IEEE Transactions on Image Processing (TIP), 2022. 6

[19] Xingtao Ling, Chenlin Fu, and Yingying Zhu. Anchor-free cross-view object geo-localization with gaussian position encoding and cross-view association, 2025. 1, 3

[20] Tianlin Liu, Mathieu Blondel, Carlos Riquelme Ruiz, and Joan Puigcerver. Routers in Vision Mixture of Experts: An Empirical Study. Transactions on Machine Learning Research, 2024. 2, 3

[21] Bo Lv, Qingwang Zhang, Le Wu, Yuanyuan Li, and Yingying Zhu. MOGeo: Beyond one-to-one cross-view object geolocalization, 2026. 3

[22] Joan Puigcerver, Carlos Riquelme Ruiz, Basil Mustafa, Cedric Renggli, Andr´ e Susano Pinto, Sylvain Gelly, Daniel´ Keysers, and Neil Houlsby. From sparse to soft mixtures of experts. In Proceedings of the International Conference on Learning Representations (ICLR), 2024. 3

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning (ICML), 2021. 3

[24] Zehao Sang, Jun Lu, Haitao Guo, Lei Ding, Kun Zhu, Guojun Xu, and Haoqi Wei. GHGeo: A cross-view objectlevel geo-localization method based on heterogeneous spatial contrastive loss. Journal of Geo-Information Science, 27(11):2563–2577, 2025. 1, 3

[25] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. In Proceedings of the International Conference on Learning Representations (ICLR), 2017. 3

[26] Yujiao Shi, Liu Liu, Xin Yu, and Hongdong Li. Polar-safa: Polar transformer for cross-view geo-localization. In Advances in Neural Information Processing Systems (NeurIPS), 2019. 6

[27] Yujiao Shi, Liu Liu, Xin Yu, and Hongdong Li. Spatialaware feature aggregation for image-based cross-view geolocalization. In Advances in Neural Information Processing Systems (NeurIPS), 2019. 6

[28] Zixuan Song, Jing Zhang, Di Wang, Zidie Zhou, Wenbin Liu, Haonan Guo, En Wang, and Bo Du. GeoBridge: A semanticanchored multi-view foundation model bridging images and text for geo-localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 3

[29] Yuxi Sun, Yunming Ye, Jian Kang, Ruben Fernandez-Beltran, Shanshan Feng, Xutao Li, Chuyao Luo, Puzhao

Zhang, and Antonio Plaza. Cross-view object geolocalization in a local region with satellite imagery. IEEE Transactions on Geoscience and Remote Sensing, 61:1–16, 2023. 1, 2, 4, 6, 7

[30] Chi-Nguyen Tran, Dao Sy Duy Minh, Huynh Trung Kiet, Nguyen Lam Phu Quy, Phu-Hoa Pham, and Long Tran-Thanh. SkyPart: Weather-robust cross-view geo-localization via prototype-based semantic part discovery, 2026. 3

[31] Liyao Wang, Ruipu Wu, Haojun Xu, Lei Shi, Linjiang Huang, and Si Liu. Beyond 2D matching: A unified singlestage framework for geometry-aware cross-view object geolocalization, 2026. 3

[32] Gina Wong, Drew Prinster, Suchi Saria, Rama Chellappa, and Anqi Liu. Toward calibrated mixture-of-experts under distribution shift. In Proceedings of the International Conference on Machine Learning (ICML), 2026. 3

[33] Jiayi Wu and Guorui Feng. CLIP-UG: CLIP-driven visionlanguage model for UAV-view geo-localization. IEEE Transactions on Consumer Electronics, 71(4):10096–10107, 2025. 3

[34] Yongchuang Wu, Hui Yang, Yanlan Wu, and Peng Zhang. GeoMatch: Natural language-guided cross-view object geolocalization and matching. ISPRS Journal ofPhotogrammetry and Remote Sensing, 237:668–685, 2026. 3

[35] Hongji Yang, Xiufan Lu, and Yingying Zhu. Cross-view geo-localization with layer-to-layer transformer. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 1, 6

[36] Hualin Ye, Bingxi Liu, Jixiang Du, Yu Qin, Ziyi Chen, and Hong Zhang. Learnable query aggregation with KV routing for cross-view geo-localisation, 2025. 3

[37] Fan Zhang, Haoyuan Ren, Fei Ma, Qiang Yin, and Yongsheng Zhou. Smgeo: Cross-view object geo-localization with grid-level mixture-of-experts, 2025. 3

[38] Jing Zhang, Zhaoxin Fan, Yuliang Guo, Hongyu Xu, Yaq Wu, Dongqing Zou, and Jianfeng Lu. Geodtr+: Towards generic cross-view geo-localization via geometric disentanglement and global correspondence. International Journal of Computer Vision, 2024. 6

[39] Qingwang Zhang and Yingying Zhu. Breaking rectangular shackles: Cross-view object segmentation for fine-grained object geo-localization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 1, 3

[40] Xiaohan Zhang, Zhangkai Shen, Si-Yuan Cao, Xiaokai Bai, Yiming Li, Zheheng Han, Zhe Wu, Qi Ming, and Hui-Liang Shen. Learning better uav-based cross-view object geolocalization from multi-modal prompts: MoP-UAV benchmark and MoPT framework, 2026. Accepted at AAAI 2026. 2, 3

[41] Yushuo Zheng, Jiangyong Ying, Huiyu Duan, Chunyi Li, Zicheng Zhang, Jing Liu, Xiaohong Liu, and Guangtao Zhai. GeoX-Bench: Benchmarking cross-view geo-localization and pose estimation capabilities of large multimodal models. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026. 3

[42] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M. Dai, Zhifeng Chen, Quoc V. Le,

and James Laudon. Mixture-of-experts with expert choice routing. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 3

[43] Sijie Zhu, Mubarak Shah, and Chen Chen. Transgeo: Transformer is all you need for cross-view image geo-localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 6