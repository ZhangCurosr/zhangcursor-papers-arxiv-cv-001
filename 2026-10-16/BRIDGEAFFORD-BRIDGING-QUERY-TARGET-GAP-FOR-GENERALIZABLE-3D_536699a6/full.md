# BRIDGEAFFORD: BRIDGING QUERY–TARGET GAP FOR GENERALIZABLE 3D AFFORDANCE GROUNDING

Yijie Su<sup>1</sup>, Yakun Zhu<sup>1</sup>, Yi Bin<sup>1,\*</sup>, Pengpeng Zeng<sup>1</sup>, Liang Chen<sup>1</sup>, Guoqing Wang<sup>2</sup>, Jingkuan Song<sup>1</sup>, Heng Tao Shen<sup>1</sup>

<sup>1</sup>Tongji University <sup>2</sup>University of Electronic Science and Technology of China

Corresponding author: Yi Bin (yi.bin@hotmail.com)

## ABSTRACT

Language-guided 3D affordance grounding aims to localize functional regions on 3D objects according to interaction intents. However, existing methods exhibit poor generalization to unseen objects. One challenge is the query–target gap: a semantic discrepancy between query features encoding interaction intents and target-region features, leading to inaccurate grounding. End-to-end training favors query representations tailored to seen objects, limiting their compatibility with shifted target-region features on unseen objects. To address this issue, we propose BridgeAfford, a framework that reduces the query–target semantic gap from two complementary perspectives. First, Interaction Region Alignment (IRA) promotes cross-object alignment among geometrically similar interaction regions during training. This enables structures on unseen objects that resemble previously observed ones to produce representations similar to those learned during training, facilitating target-region recognition. Second, Visual Evidence Aggregation (VEA) leverages a 2D affordance model to refine the attention distribution and guide the query toward the correct region, yielding more target-aware queries. Extensive experiments on the LASO and PIAD datasets demonstrate that BridgeAfford outperforms existing methods, particularly on unseen objects, highlighting its robust generalization ability.

## 1 INTRODUCTION

3D affordance grounding aims to localize functional regions on 3D objects according to interaction intents, which is fundamental for embodied agents. Recently, language-guided 3D affordance grounding has attracted increasing attention, as natural-language instructions can express more flexible and fine-grained interaction intents than predefined affordance labels (Li et al., 2024).

However, existing methods still suffer from limited generalization, with substantial performance degradation on objects unseen during training. Most prior approaches (Li et al., 2024; Lu et al., 2025; Li et al., 2025) follow a common LASO paradigm: given an instruction and a point cloud, the model uses the interaction intent to extract target-region information from point-wise object features and incorporate it into query features, aligning them semantically with the target region. The affordance mask is subsequently predicted by measuring the similarity between the target-aware queries and the point-wise object features.

For objects seen during training, the model can learn and memorize the correspondence between interaction intents and target regions, incorporating the correct regional features into the query to achieve semantic alignment with the target region. On unseen objects at inference time, however, it may encounter interaction-region representations that deviate from those observed during training. As a result, it cannot reliably use learned associations to incorporate the relevant information into the query, leading to a semantic gap with the target region.

Based on this observation, we aim to narrow the semantic gap between queries and target regions. Specifically, we enable the model to reliably capture target-region features and incorporate them into queries, even on unseen objects. First, we encourage geometrically similar interaction regions across training objects to have consistent representations. This allows similar structures on unseen objects to retain familiar feature patterns, making them easier for the model to recognize. Second, novel interaction structures may not be reliably associated with those observed during training through geometric similarity alone. We therefore integrate spatial evidence from a generalizable 2D affordance model to directly guide target-region feature extraction and facilitate the formation of target-aware queries.

![](images/734dd075d8e9839b02c2d574d8dfae5f5000139b9e45a1c43ce75680f4eb4e37.jpg)  
Figure 1: Motivation for BridgeAfford. (a) Accurate prediction relies on high semantic similarity between the query and the target-region representation. However, shifted local representations on unseen objects may prevent the model from extracting the target region, resulting in query features with low similarity to it. (b) Unseen objects may contain interaction regions that are geometrically similar to those of seen objects. Aligning these regions with geometrically similar regions on training objects promotes consistent representations, helping the model recognize interaction regions and incorporate their semantic information into the query. (c) Objects may contain novel interaction structures that cannot be reliably aligned with those observed during training based on geometric similarity. We therefore use a more generalizable 2D affordance model to directly guide feature extraction, facilitating the generation of target-aware queries.

To this end, we propose BridgeAfford, which narrows the query–target gap through Interaction Region Alignment (IRA) and Visual Evidence Aggregation (VEA). IRA encourages consistent representations of recurring local interaction structures across objects. We first use Diversity-Aware Initialization (DAI) to construct a compact set of geometrically diverse anchors in a Category-Agnostic Interaction Memory (CAIM). During training, Affinity-Based Contrastive Learning (ABCL) aligns interaction regions with these anchors according to geometric similarity. This encourages the model to represent similar interaction structures consistently despite differences in the objects containing them. When such structures appear on unseen objects, their interaction-region features remain close to those of geometrically similar local structures observed during training.

Complementing this region alignment, VEA provides spatial guidance for target-region feature extraction on the current object. We employ a Normal-Based Gaussian Renderer (NBGR) to generate multi-view renderings and a lightweight 2D affordance model to localize target regions in each view. Confidence-Aware Multi-View Aggregation (CAMVA) then lifts these predictions into dense pointwise 3D visual evidence. We incorporate this evidence as a bias in text-to-point cross-attention, guiding feature extraction toward the target region and yielding more reliable, target-aware queries.

We validate BridgeAfford on the LASO and PIAD datasets. Experimental results demonstrate that our model achieves leading performance, particularly on unseen categories.

In summary, our contributions are as follows:

• We propose BridgeAfford, a framework that narrows the query–target gap by enabling more reliable target-region feature extraction on unseen objects.

• We introduce Interaction Region Alignment (IRA) to promote consistent representations of geometrically similar interaction regions across objects, providing transferable cues for target recognition. Complementarily, Visual Evidence Aggregation (VEA) converts multiview 2D predictions into dense spatial evidence to guide target-region feature extraction and produce target-aware queries.

• Extensive experiments on LASO and PIAD demonstrate that BridgeAfford outperforms existing methods, particularly in unseen settings, validating its strong generalization ability.

## 2 RELATED WORK

3D Affordance Learning. 3D affordance learning localizes functional regions on 3D objects. Early work formulates it as point-wise multi-label segmentation (Deng et al., 2021), while later methods represent interaction intent using human-object interaction images (Yang et al., 2023; Shao et al., 2025) or language. For language-based grounding, OpenAD (Nguyen et al., 2023) enables open-vocabulary affordance detection with pretrained text encoders, and LASO (Li et al., 2024) extends the input to sentence-level instructions. Recent methods (Lu et al., 2025; Li et al., 2025; Chu et al., 2025; Basak & Yin, 2025) improve this setting with external models, cross-modal designs, or enhanced training strategies, yet generalization to unseen objects remains challenging.

Object Representation Learning for 3D Affordance. LASO (Li et al., 2024) integrates textual and geometric features through multi-layer upsampling, while GEAL (Lu et al., 2025) improves robustness with 2D–3D consistency alignment. GLANCE (Li et al., 2025) introduces cross-modal connectors between the 3D and text encoders, and CMAT (Huang et al., 2026) transfers knowledge from 2D foundation models via cross-modal affinity consistency. Overall, these methods enhance object features by injecting semantic information from 2D or text encoders. Additionally, Aff3DFunc (Wu et al., 2025) utilizes contrastive learning to associate point cloud features with functional semantics. Gou & He (2026) employ a set of prototypes to align features corresponding to the same affordance. These methods typically rely on explicit labels to align or distinguish local features. However, in language-guided tasks, interaction intents are implicitly expressed by sentences, making explicit labels unavailable. Extracting such labels with LLMs is costly and does not resolve a deeper issue: the same intent may correspond to geometrically diverse regions across objects, and forcibly aligning them can distort object representations. In contrast to prior works, our method does not rely on explicit affordance labels. Instead, it identifies interaction regions with similar geometric patterns across objects and enforces consistent representations over these regions.

Query Enhancement for 3D Affordance. Recent studies have explored different ways to enhance target-aware query representations. PART (Gou & He, 2026) and HAMMER (Yao et al., 2026) rely on LLMs or MLLMs to infer interaction-related semantics. However, such high-level and abstract semantic priors still remain semantically distant from features of the target region, while also introducing considerable computational overhead. In contrast, we employ a lightweight DINOv2-based 2D affordance model to directly predict target-region masks on rendered views and lift them into 3D, providing more explicit spatial guidance. GLANCE (Li et al., 2025) further introduces 2D visual cues through sparse keypoint sampling. However, sparse samples capture only partial evidence from the target mask and discard its continuous spatial structure and intra-region confidence variation. Our CAMVA instead produces dense and smooth point-wise guidance over the entire target region, enabling the query to aggregate more comprehensive regional features while preserving stronger responses at more confident points.

## 3 METHOD

## 3.1 THE BACKBONE OF BRIDGEAFFORD

Given a 3D point cloud $\mathcal { P } = \{ p _ { i } \} _ { i = } ^ { N }$ and an interaction instruction T, language-guided 3D affordance grounding aims to predict a dense affordance map $\hat { \mathbf { y } } = \{ \hat { y } _ { i } \} _ { i = 1 } ^ { N }$ , where $\hat { y } _ { i } \in [ 0 , 1 ]$ denotes the interaction likelihood of point $p _ { i }$ . We first extract geometric features G from the point cloud using a 3D encoder (Zhou et al., 2024) and fuse them with the text features T produced by a text encoder (Liu et al., 2019). The fused features are then upsampled to obtain point-wise object features F. Meanwhile, in the query generator, a set of learnable queries first extracts semantic information from the text features and subsequently aggregates object representations of the target region through the decoder, yielding target-aware queries $\mathbf { Q } ^ { f }$ . Finally, the affordance mask is obtained by computing the similarity between the target-aware queries and the point-wise object features.

![](images/2a54ec73a20172dad5d9505aaa37dc7d2f8acf1105c0f006ebba937f31d3f684.jpg)  
Figure 2: Overview of BridgeAfford. (a) BridgeAfford follows the LASO pipeline and improves generalization from two sides: IRA aligns geometrically similar interaction regions across objects, while VEA lifts multi-view 2D affordance predictions into 3D as an attention bias. (b) During initialization, new region features are directly inserted when empty slots are available; otherwise, the two most similar entries are merged to release a slot for the new entry. (c) Visual evidence is incorporated as an attention bias to guide query-to-point attention toward interaction regions.

## 3.2 INTERACTION REGION ALIGNMENT

Category-Agnostic Interaction Memory (CAIM). During training, we extract two region-level features from the annotated interaction region for each sample: a geometric region feature $\mathbf { g } ^ { r }$ and a language-aware region feature $\mathbf { f } ^ { r }$ . We obtain $\mathbf { g } ^ { r }$ and $\mathbf { f } ^ { r }$ by masked average pooling:

$$
\mathbf { g } ^ { r } = \mathrm { N o r m } \left( \frac { \sum _ { i = 1 } ^ { R } y _ { i } ^ { g } \mathbf { g } _ { i } } { \sum _ { i = 1 } ^ { R } y _ { i } ^ { g } + \epsilon } \right) , \quad \mathbf { f } ^ { r } = \mathrm { N o r m } \left( \frac { \sum _ { i = 1 } ^ { N } y _ { i } \mathbf { f } _ { i } } { \sum _ { i = 1 } ^ { N } y _ { i } + \epsilon } \right) .\tag{1}
$$

Here, $\mathbf { g } ^ { r }$ summarizes the local geometric pattern of the interaction region, while $\mathbf { f } ^ { r }$ summarizes the corresponding object features that integrate both geometric information and interaction semantics. The point-wise mask $\mathbf { y }$ is mapped to group-wise scores $\mathbf { y } ^ { g }$ according to the group centers of the 3D encoder. CAIM maintains a compact memory bank

$$
\mathcal { M } = \{ ( \mathbf { m } _ { k } ^ { g } , \mathbf { m } _ { k } ^ { f } , c _ { k } ) \} _ { k = 1 } ^ { K } ,\tag{2}
$$

where each entry stores a geometric region feature $\mathbf { m } _ { k } ^ { g } ,$ its paired fused feature $\mathbf { m } _ { k } ^ { f }$ , and a count $c _ { k }$ that records how many region features have been merged into this entry.

Diversity-Aware Initialization (DAI). To capture diverse local geometric structures in the training data, we adopt a diversity-aware strategy to initialize the CAIM. If the CAIM contains fewer than K entries, the new feature pair $\bigl ( \mathbf { g } ^ { r } , \mathbf { f } ^ { \bar { r } } \bigr )$ is directly inserted with count 1. Once the CAIM is full, we first insert the new feature into the $( K + 1 )$ -th temporary slot. CAIM then considers all $K + 1$ entries jointly and finds the two most similar entries according to their geometric region features m<sup>g</sup>:

$$
( u , v ) = \arg \operatorname* { m a x } _ { u \neq v , \ u , v \in \{ 1 , \ldots , K + 1 \} } \cos ( \mathbf { m } _ { u } ^ { g } , \mathbf { m } _ { v } ^ { g } ) .\tag{3}
$$

The selected two entries are merged according to their accumulated counts:

$$
\bar { \mathbf { m } } ^ { g } = \mathrm { N o r m } \left( \frac { c _ { u } \mathbf { m } _ { u } ^ { g } + c _ { v } \mathbf { m } _ { v } ^ { g } } { c _ { u } + c _ { v } } \right) , \quad \bar { \mathbf { m } } ^ { f } = \mathrm { N o r m } \left( \frac { c _ { u } \mathbf { m } _ { u } ^ { f } + c _ { v } \mathbf { m } _ { v } ^ { f } } { c _ { u } + c _ { v } } \right) , \quad \bar { c } = c _ { u } + c _ { v } .\tag{4}
$$

After merging, CAIM is reduced back to K entries. Repeating this procedure allows CAIM to maintain $\bar { K }$ diverse region features instead of being dominated by redundant samples.

Affinity-Based Contrastive Learning (ABCL). After initialization, we introduce Affinity-Based Contrastive Learning (ABCL) to align f<sup>r</sup> based on geometric affinity. Unlike conventional contrastive learning with label-defined positive pairs, ABCL dynamically selects geometrically similar interaction regions as positives, avoiding the forced alignment of regions that share the same interaction function but exhibit substantially different local representations. For each sample, its geometric region feature $\mathbf { g } ^ { r }$ retrieves the most similar $\mathbf { m } _ { k } ^ { g }$ in CAIM:

$$
k ^ { + } = \arg \operatorname* { m a x } _ { k } \cos ( \mathbf { g } ^ { r } , \mathbf { m } _ { k } ^ { g } ) .\tag{5}
$$

Here, $\mathbf { m } _ { k ^ { + } } ^ { g }$ defines the shared geometric pattern for the current interaction region. Its paired feature $\mathbf { m } _ { k ^ { - } } ^ { f }$ is used as the positive target to align the $\mathbf { f } ^ { r }$ of the current sample with an InfoNCE loss:

$$
\mathcal { L } _ { \mathrm { c a i m } } = - \log \frac { \exp ( \cos ( \mathbf { f } ^ { r } , \mathbf { m } _ { k ^ { + } } ^ { f } ) / \tau ) } { \sum _ { k = 1 } ^ { K } \exp ( \cos ( \mathbf { f } ^ { r } , \mathbf { m } _ { k } ^ { f } ) / \tau ) } .\tag{6}
$$

In this way, for each sample, $\mathbf { g } ^ { r }$ retrieves the most similar interaction region from CAIM through similarity matching, and $\mathbf { f } ^ { r }$ is aligned with the corresponding $\mathbf { m } _ { k ^ { + } } ^ { f }$ . This encourages the model to produce more general and cross-object-consistent region features. Finally, the selected memory entry is updated online by exponential moving average (EMA):

$$
\mathbf { m } _ { k + } ^ { g } \gets \mathrm { N o r m } \left( \alpha \mathbf { m } _ { k + } ^ { g } + ( 1 - \alpha ) \mathbf { g } ^ { r } \right) , \quad \mathbf { m } _ { k + } ^ { f } \gets \mathrm { N o r m } \left( \alpha \mathbf { m } _ { k + } ^ { f } + ( 1 - \alpha ) \mathbf { f } ^ { r } \right) .\tag{7}
$$

## 3.3 VISUAL EVIDENCE AGGREGATION

Geometric alignment may provide insufficient cues for unseen objects with novel or substantially shifted local structures, causing the model to attend to incorrect or incomplete target regions. We therefore introduce Visual Evidence Aggregation (VEA), which converts multi-view 2D affordance predictions into dense 3D visual evidence to guide target-region feature extraction.

Normal-Based Gaussian Renderer (NBGR). Renderers such as Mitsuba 3 provide high-quality visualization but are relatively heavy for repeatedly generating multi-view renderings. Following GEAL (Lu et al., 2025), we adopt Gaussian rendering for efficient multi-view point-cloud rendering. To enrich visual cues for colorless point clouds, we further estimate point normals from local neighborhoods and convert them into RGB colors. This normal-based coloring produces rendered images with clearer surface orientation, object parts, and geometric structures.

2D Affordance Model. Given a rendered view and the language instruction, we use a lightweight 2D affordance model to predict a view-level affordance map. We instantiate this model with the 2D branch of GEAL (Lu et al., 2025), which encodes rendered views with a frozen visual encoder (Oquab et al., 2023) and fuses visual and language features for affordance prediction. The model is supervised by 2D heatmaps rendered from 3D annotations.

Confidence-Aware Multi-View Aggregation (CAMVA). For point $p _ { i }$ , we project it onto the view v and obtain its 2D affordance score by bilinearly interpolating the values of neighboring pixels. However, direct projection ignores view-dependent visibility: an occluded point may still be projected to an image location, but the sampled 2D score from that view is unreliable. Inspired by D<sup>3</sup>Fields (Wang et al., 2025), we compute confidence scores to measure whether the sampled 2D evidence should be trusted. Let $d _ { i } ^ { v }$ denote the distance from point $p _ { i }$ to the camera plane in view $v ,$ and let $\tilde { d } _ { i } ^ { v }$ denote the depth value sampled from the rendered depth map. $\mathrm { I f } \ p _ { i }$ is visible in view $v ,$ $\tilde { d } _ { i } ^ { v }$ should be close to $d _ { i } ^ { v }$ . Therefore, we define the depth inconsistency and confidence score as

$$
\Delta _ { i } ^ { v } = \operatorname* { m a x } \left( 0 , \frac { d _ { i } ^ { v } - \tilde { d } _ { i } ^ { v } } { \sigma _ { d } ^ { v } + \epsilon } \right) , \quad c _ { i } ^ { v } = \mathbb { I } _ { i } ^ { v } \exp \left( - \frac { \Delta _ { i } ^ { v } } { \tau _ { d } } \right) ,\tag{8}
$$

where $\sigma _ { d } ^ { v }$ is the robust depth scale of view v, $\mathbb { I } _ { i } ^ { v }$ indicates whether $p _ { i }$ has a valid projection and valid depth in view v, and $\tau _ { d }$ controls the depth tolerance. The final 3D visual evidence score is obtained by confidence-weighted aggregation:

$$
s _ { i } = \frac { \sum _ { v = 1 } ^ { V } c _ { i } ^ { v } a _ { i } ^ { v } } { \sum _ { v = 1 } ^ { V } c _ { i } ^ { v } + \epsilon } .\tag{9}
$$

The resulting $s _ { i }$ provides dense point-wise visual evidence in 3D space.

Visual-Guided Cross-Attention. We inject the aggregated visual evidence into query-to-point attention as an additive bias. For a language-conditioned query $\mathbf { q } _ { m } ^ { t }$ and point representation $\mathbf { f } _ { i }$ , the attention logit is modified as

$$
\tilde { e } _ { m , i } = \frac { ( W _ { q } \mathbf { q } _ { m } ^ { t } ) ^ { \top } ( W _ { k } \mathbf { f } _ { i } ) } { \sqrt { C } } + \lambda _ { v } s _ { i } ,\tag{10}
$$

where $\lambda _ { v }$ controls the strength of visual guidance. In this way, VEA provides dense spatial evidence that guides language queries toward likely interaction regions.

## 3.4 TRAINING STRATEGY AND OBJECTIVE

The 2D affordance model is first trained with view-level supervision rendered from 3D point-wise annotations following GEAL (Lu et al., 2025), and then frozen to produce multi-view affordance heatmaps for VEA. For the 3D model, we optimize the point-wise affordance prediction loss together with the CAIM alignment objective:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { \mathrm { c a i m } } \mathcal { L } _ { \mathrm { c a i m } } .\tag{11}
$$

$\mathcal { L } _ { \mathrm { s e g } }$ supervises dense affordance scores using a combination of Focal loss and Dice loss, following prior works (Li et al., 2024; Lu et al., 2025; Li et al., 2025). The CAIM loss encourages geometrically similar interaction regions to share a unified and generalized representation.

Training the 3D backbone proceeds in three stages. First, we train the backbone using only $\mathcal { L } _ { \mathrm { s e g } } .$ , so that the text-geometry fusion module and upsampler learn relatively stable object representations. Second, we initialize CAIM with the Diversity-Aware Initialization described above. Third, we jointly optimize $\mathcal { L } _ { \mathrm { s e g } }$ and $\mathcal { L } _ { \mathrm { c a i m } }$ while updating the CAIM online with EMA. During inference, CAIM is omitted, and BridgeAfford operates with only VEA and the 3D backbone.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Implementation Details. Following prior work (Li et al., 2025), we use Uni3D (Zhou et al., 2024) as the point-cloud encoder and RoBERTa (Liu et al., 2019) as the text encoder. The mask decoder is implemented with a Transformer decoder (Vaswani et al., 2017). We render 6 views for each point cloud at a resolution of 224 × 224. For the 3D backbone, we train for 30 epochs using AdamW with a batch size of 16 and a learning rate of $1 \times 1 0 ^ { - 4 }$ . The CAIM contains $K = 1 6$ entries, uses 3 warm-up epochs and 1 initialization epoch, and is then updated online with EMA coefficient 0.999. In all experiments, the 2D model and the 3D backbone are trained using the same training data. All experiments were conducted on a single NVIDIA GeForce RTX 5090 GPU. Detailed hyperparameter configurations are presented in Appendix A.

Datasets. We evaluate BridgeAfford on the LASO (Li et al., 2024) and PIAD (Yang et al., 2023) datasets. LASO is a language-guided 3D affordance segmentation benchmark. To evaluate PIAD in the language-guided setting, we directly reuse the PIAD language templates provided by GEAL (Lu et al., 2025). Both datasets are evaluated under seen and unseen settings. In the seen setting, training and testing share the same object categories or object-affordance compositions. In the unseen setting, specific object-affordance pairs are held out during training and appear only during testing. Therefore, unseen performance directly reflects the model’s ability to generalize beyond memorized object-region associations, which is the main focus of our work.

Evaluation Metrics. Following prior work, we report average Intersection over Union (aIoU), Area Under the ROC Curve (AUC), Similarity (SIM), and Mean Absolute Error (MAE). Higher aIoU, AUC, and SIM indicate better performance, while lower MAE is preferred.

## 4.2 QUANTITATIVE RESULTS

Results on LASO. Table 1 compares BridgeAfford with existing methods on LASO. On seen objects, BridgeAfford achieves strong performance and remains competitive with recent approaches. More importantly, BridgeAfford achieves leading performance across all metrics in the unseen setting, demonstrating its strong generalization ability to unseen objects.

Table 1: Results on the LASO dataset. Higher is better for aIoU, AUC, and SIM, while lower is better for MAE. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">Seen</td><td colspan="4">Unseen</td></tr><tr><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td>ReferTrans (Li &amp; Sigal, 2021)</td><td>NeurIPS 2021</td><td>13.7</td><td>79.8</td><td>0.497</td><td>0.124</td><td>10.2</td><td>69.1</td><td>0.432</td><td>0.145</td></tr><tr><td>ReLA (Liu et al., 2023)</td><td>CVPR 2023</td><td>15.2</td><td>78.9</td><td>0.532</td><td>0.118</td><td>10.7</td><td>69.7</td><td>0.429</td><td>0.144</td></tr><tr><td>3D-SPS (Luo et al., 2022)</td><td>CVPR 2022</td><td>11.4</td><td>76.2</td><td>0.433</td><td>0.138</td><td>7.9</td><td>68.8</td><td>0.402</td><td>0.158</td></tr><tr><td>IAGNet (Yang et al., 2023)</td><td>ICCV 2023</td><td>17.8</td><td>82.3</td><td>0.561</td><td>0.109</td><td>12.9</td><td>77.8</td><td>0.443</td><td>0.129</td></tr><tr><td>PointRefer (Li et al., 2024)</td><td>CVPR 2024</td><td>20.8</td><td>87.3</td><td>0.629</td><td>0.093</td><td>14.6</td><td>80.2</td><td>0.507</td><td>0.119</td></tr><tr><td>3D-AffLLM (Chu et al., 2025)</td><td>ICLR 2025</td><td>18.2</td><td>84.9</td><td>0.622</td><td>0.104</td><td>15.3</td><td>78.7</td><td>0.542</td><td>0.124</td></tr><tr><td>GEAL (Lu et al., 2025)</td><td>CVPR 2025</td><td>22.0</td><td>86.7</td><td>0.634</td><td>0.092</td><td>16.7</td><td>80.9</td><td>0.567</td><td>0.106</td></tr><tr><td>ViSPLA (Basak &amp; Yin, 2025)</td><td>NeurIPS 2025</td><td>22.8</td><td>87.3</td><td>0.651</td><td>0.090</td><td>17.1</td><td>81.5</td><td>0.571</td><td>0.103</td></tr><tr><td>GLANCE (Li et al., 2025)</td><td>ICCV 2025</td><td>23.3</td><td>88.1</td><td>0.632</td><td>0.089</td><td>18.7</td><td>81.7</td><td>0.532</td><td>0.112</td></tr><tr><td>BridgeAfford (Ours)</td><td></td><td>23.7</td><td>86.8</td><td>0.646</td><td>0.086</td><td>21.8</td><td>83.0</td><td>0.597</td><td>0.099</td></tr></table>

Results on PIAD. Table 2 reports the results on PIAD. Among language-guided methods, BridgeAfford achieves the best performance under both seen and unseen settings. Compared with ViSPLA, which introduces an LLM for affordance reasoning, BridgeAfford still achieves clear gains, improving aIoU from 9.20 to 14.50. In addition, we compare BridgeAfford with current intention-driven methods, and the results show that our method still achieves the best performance on most metrics.

Table 2: Results on the PIAD dataset. The best and second-best results among all methods are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">Seen</td><td colspan="4">Unseen</td></tr><tr><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td colspan="10">Intention-Driven 3D Affordance Grounding</td></tr><tr><td>PMF (Zhuang et al., 2021)</td><td>ICCV 2021</td><td>10.13</td><td>75.05</td><td>0.425</td><td>0.141</td><td>4.67</td><td>60.25</td><td>0.330</td><td>0.211</td></tr><tr><td>ILN (Chen et al., 2022)</td><td>SIGGRAPH 2022</td><td>11.25</td><td>75.84</td><td>0.427</td><td>0.137</td><td>4.71</td><td>59.69</td><td>0.325</td><td>0.207</td></tr><tr><td>FRCNN (Xu et al., 2023)</td><td>Remote Sensing 2023</td><td>11.97</td><td>76.05</td><td>0.429</td><td>0.136</td><td>4.71</td><td>61.92</td><td>0.332</td><td>0.195</td></tr><tr><td>PFusion (Xu et al., 2018)</td><td>CVPR 2018</td><td>12.31</td><td>77.50</td><td>0.432</td><td>0.135</td><td>5.33</td><td>61.87</td><td>0.330</td><td>0.193</td></tr><tr><td>XMFNet (Aiello et al., 2022)</td><td>NeurIPS 2022</td><td>12.94</td><td>78.25</td><td>0.441</td><td>0.127</td><td>5.68</td><td>62.58</td><td>0.342</td><td>0.188</td></tr><tr><td>IAGNet (Yang et al., 2023)</td><td>ICCV 2023</td><td>20.51</td><td>84.85</td><td>0.545</td><td>0.098</td><td>7.95</td><td>71.84</td><td>0.352</td><td>0.127</td></tr><tr><td>GREAT (Shao et al., 2025)</td><td>CVPR 2025</td><td>19.61</td><td>85.22</td><td>0.569</td><td>0.093</td><td>8.32</td><td>67.46</td><td>0.330</td><td>0.121</td></tr><tr><td>HAMMER (Yao et al., 2026)</td><td>CVPR 2026</td><td>22.20</td><td>88.43</td><td>0.605</td><td>0.083</td><td>13.71</td><td>80.92</td><td>0.449</td><td>0.109</td></tr><tr><td colspan="10">Language-Guided 3D Affordance Grounding</td></tr><tr><td>PointRefer (Li et al., 2024)</td><td>CVPR 2024</td><td>19.70</td><td>84.20</td><td>0.590</td><td>0.096</td><td>8.00</td><td>69.20</td><td>0.386</td><td>0.118</td></tr><tr><td>GEAL (Lu et al., 2025)</td><td>CVPR 2025</td><td>22.50</td><td>85.00</td><td>0.600</td><td>0.092</td><td>8.70</td><td>72.50</td><td>0.390</td><td>0.102</td></tr><tr><td>ViSPLA (Basak &amp; Yin, 2025)</td><td>NeurIPS 2025</td><td>23.10</td><td>85.80</td><td>0.664</td><td>0.089</td><td>9.20</td><td>73.10</td><td>0.431</td><td>0.099</td></tr><tr><td>BridgeAfford (Ours)</td><td></td><td>25.40</td><td>87.60</td><td>0.668</td><td>0.080</td><td>14.50</td><td>78.80</td><td>0.459</td><td>0.095</td></tr></table>

Cross-Dataset Evaluation. To evaluate cross-dataset transferability, we train models on the LASO dataset and directly evaluate them on the PIAD test set. As shown in Table 3, our method achieves the best performance across all reported metrics. The results of PointRefer and GEAL are from our reproduction, while the results of 3D-AffLLM and ViSPLA are taken from ViSPLA (Basak & Yin, 2025). Note that ViSPLA does not report MAE, and therefore the corresponding entry is unavailable.

Table 3: Cross-dataset generalization.
<table><tr><td>Method</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td>PointRefer (Li et al., 2024)</td><td>17.7</td><td>83.2</td><td>0.567</td><td>0.110</td></tr><tr><td>GEAL (Lu et al., 2025)</td><td>18.5</td><td>83.8</td><td>0.582</td><td>0.096</td></tr><tr><td>3D-AffLLM (Chu et al., 2025)</td><td>17.6</td><td>82.4</td><td>0.570</td><td>一</td></tr><tr><td>ViSPLA (Basak &amp; Yin, 2025)</td><td>19.7</td><td>84.5</td><td>0.610</td><td></td></tr><tr><td>Ours</td><td>23.6</td><td>87.0</td><td>0.633</td><td>0.080</td></tr></table>

Generalization with Limited Training Data. To evaluate the data efficiency of BridgeAfford, we randomly retain {20%, 40%, 60%, 80%, 100%} of the original LASO unseen training split, while ensuring a balanced sampling distribution across different object–affordance combinations. As shown in Figure 3, the blue curves denote the performance of BridgeAfford under different training data ratios, while the horizontal lines indicate existing baselines trained with the full training set. Notably, BridgeAfford trained with only 60% of the data surpasses all baselines trained with 100% of the data, and achieves the best aIoU with only 20% of the data. These results demonstrate the superior data efficiency and strong generalization of BridgeAfford under limited training data.

![](images/e2b2cd8bd10faa23e74c96b5e0982a4e352535bf7d8f00af8b37425c79c85b4b.jpg)  
Figure 3: Data efficiency of our method with limited training data.

## 4.3 ABLATION STUDIES

Main Module Ablation. We evaluate the contribution of the two main components on the LASO dataset. As shown in Table 4, both IRA and VEA consistently improve the backbone, especially in the unseen setting. IRA improves the generalization by encouraging geometrically similar interac tion regions across objects to learn consistent representations. VEA provides dense target-region evidence to guide queries toward target regions. IRA improves the transferability of interaction-region representations, whereas VEA provides dense spatial evidence for target-region feature extraction. Combining them therefore achieves the best overall performance.

Table 4: Main module ablation on the LASO dataset.
<table><tr><td rowspan="2">Variant</td><td colspan="4">Seen</td><td colspan="4">Unseen</td></tr><tr><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td>Backbone</td><td>22.0</td><td>84.6</td><td>0.610</td><td>0.094</td><td>19.3</td><td>80.7</td><td>0.568</td><td>0.107</td></tr><tr><td>Backbone + IRA</td><td>23.4</td><td>85.7</td><td>0.635</td><td>0.090</td><td>20.7</td><td>82.5</td><td>0.585</td><td>0.100</td></tr><tr><td>Backbone + VEA</td><td>22.9</td><td>86.0</td><td>0.638</td><td>0.089</td><td>20.4</td><td>81.1</td><td>0.592</td><td>0.102</td></tr><tr><td>Backbone + IRA + VEA</td><td>23.7</td><td>86.8</td><td>0.646</td><td>0.086</td><td>21.8</td><td>83.0</td><td>0.597</td><td>0.099</td></tr></table>

Table 5: Fine-grained ablation on the LASO unseen split.
<table><tr><td colspan="5">IRA fine-grained ablation</td><td colspan="5">VEA fine-grained ablation</td></tr><tr><td>Variant</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td><td>Variant</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td>Label-based alignment</td><td>20.1</td><td>80.5</td><td>0.563</td><td>0.102</td><td>2D affordance model only</td><td>15.2</td><td>77.1</td><td>0.500</td><td>0.192</td></tr><tr><td>Random initialization</td><td>19.8</td><td>81.4</td><td>0.588</td><td>0.102</td><td>w/o normal-based coloring</td><td>19.1</td><td>81.0</td><td>0.565</td><td>0.108</td></tr><tr><td rowspan="2">Full IRA</td><td rowspan="2">21.8</td><td rowspan="2">83.0</td><td rowspan="2">0.597</td><td rowspan="2">0.099</td><td>w/o depth confidence</td><td>21.4</td><td>81.5</td><td>0.592</td><td>0.106</td></tr><tr><td>Full VEA</td><td>21.8 83.0</td><td></td><td>0.597</td><td>0.099</td></tr></table>

Label-Based Alignment vs. Geometry-Based Alignment. We evaluate an alternative strategy that aligns interaction regions across objects according to affordance labels. As shown in Table 5, label-based alignment underperforms geometry-based alignment. Since regions with the same label may exhibit substantially different geometries, enforcing their alignment can distort the original object representations.

Random Initialization vs. Diversity-Aware Initialization (DAI). To validate the role of DAI, we randomly initialize K vectors, which are then updated during training. The results show that this variant improves over the backbone, but underperforms the DAI strategy. Notably, we observe that random initialization can easily lead to training collapse, preventing the model from learning meaningful representations. This is mainly because, at the early training stage, local features may be aligned to uninformative entries, thereby corrupting the representation learning process.

2D Affordance Model Only. We also evaluate a variant that directly uses the point-wise scores obtained by aggregating multi-view 2D predictions into 3D as the final output, without using the 3D model. The results in Table 5 show that this variant performs substantially worse than the full 3D model. This is mainly because limited viewpoints and occlusions prevent 2D predictions from forming geometrically continuous and smooth masks in 3D. In Appendix D.2, we further provide empirical evidence that 2D predictions exhibit stronger generalization in localizing target regions, but remain insufficient for producing high-quality point-wise masks.

Without Normal-Based Coloring. We remove the normal-based coloring mechanism and render point clouds following the rendering strategy of GEAL (Lu et al., 2025). Results in Table 5 show that this variant leads to a performance drop. This is mainly because different surfaces and parts of the same object lack color distinction, making the object and its components harder to recognize. In Appendix D.1, we further illustrate the rendering results of our method, demonstrating that normal based coloring significantly enhances the visual quality of the rendered images.

Without Depth Confidence. We remove the confidence scores computed from depth consistency and projection validity, and directly average the sampled 2D scores of each 3D point across all views as its visual evidence. The results show that removing depth confidence leads to performance degradation because occlusion is no longer properly accounted for during multi-view aggregation. Consequently, the 2D masks cannot be accurately lifted into point-wise 3D visual evidence.

## 4.4 QUALITATIVE RESULTS

As shown in Figure 4, we visualize representative cases on the LASO unseen split and compare BridgeAfford with previous methods. Our method localizes the target interaction regions more accurately, produces predictions with higher confidence, and effectively suppresses noisy responses in irrelevant regions, resulting in cleaner and more complete affordance masks. These qualitative results further demonstrate the superior generalization ability of our method on unseen objects.

![](images/b94c4d3d4a9b017983ad34afbcac9e4dc1cb601575a771d9ab39efe620614bf3.jpg)  
Figure 4: Qualitative comparisons on the LASO dataset (unseen).

## 5 CONCLUSION

BridgeAfford improves 3D affordance grounding on unseen objects by narrowing the semantic gap between queries and target regions. Our results suggest that transferable local geometric structures learned from similar interaction regions contribute to cross-object generalization, while 2D visual evidence provides valuable guidance for query localization. Extensive experiments on LASO and PIAD validate the superiority of BridgeAfford, highlighting the potential of combining geometric structures with 2D visual priors for generalizable 3D affordance understanding.

## REFERENCES

Emanuele Aiello, Diego Valsesia, and Enrico Magli. Cross-modal learning for image-guided point cloud shape completion. Advances in Neural Information Processing Systems, 35:37349–37362, 2022.

Hritam Basak and Zhaozheng Yin. Vispla: Visual iterative self-prompting for languageguided 3d affordance learning. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 65706–65724. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-2204. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/5eee634cb9729b8bcc2ec9f2a46a74ae-Paper-Conference.pdf.

Honghua Chen, Zeyong Wei, Yabin Xu, Mingqiang Wei, and Jun Wang. Imlovenet: Misaligned image-supported registration network for low-overlap point cloud pairs. In ACM SIGGRAPH 2022 conference proceedings, pp. 1–9, 2022.

Hengshuo Chu, Xiang Deng, Qi Lv, Xiaoyang Chen, Yinchuan Li, Jianye HAO, and Liqiang Nie. 3d-affordancellm: Harnessing large language models for open-vocabulary affordance detection in 3d worlds. In The Thirteenth International Conference on Learning Representations, 2025.

Shengheng Deng, Xun Xu, Chaozheng Wu, Ke Chen, and Kui Jia. 3d affordancenet: A benchmark for visual object affordance understanding. In proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 1778–1787, 2021.

Dongqiang Gou and Xuming He. Part-aware open-vocabulary 3d affordance grounding via prototypical semantic and geometric alignment. arXiv preprint arXiv:2603.17647, 2026.

Yu Huang, Zelin Peng, Changsong Wen, Xiaokang Yang, and Wei Shen. Unlocking 3d affordance segmentation with 2d semantic knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6399–6409, 2026.

Muchen Li and Leonid Sigal. Referring transformer: A one-step approach to multi-task visual grounding. Advances in neural information processing systems, 34:19652–19664, 2021.

Yicong Li, Na Zhao, Junbin Xiao, Chun Feng, Xiang Wang, and Tat-seng Chua. Laso: Languageguided affordance segmentation on 3d object. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14251–14260, June 2024.

Yicong Li, Yiyang Chen, Zhenyuan Ma, Junbin Xiao, Xiang Wang, and Angela Yao. Intermediate connectors and geometric priors for language-guided affordance segmentation on unseen object categories. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22836–22845, 2025.

Chang Liu, Henghui Ding, and Xudong Jiang. Gres: Generalized referring expression segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 23592–23601, 2023.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Dongyue Lu, Lingdong Kong, Tianxin Huang, and Gim Hee Lee. Geal: Generalizable 3d affordance learning with cross-modal consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1680–1690, June 2025.

Junyu Luo, Jiahui Fu, Xianghao Kong, Chen Gao, Haibing Ren, Hao Shen, Huaxia Xia, and Si Liu. 3d-sps: Single-stage 3d visual grounding via referred point progressive selection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16454–16463, 2022.

Toan Nguyen, Minh Nhat Vu, An Vuong, Dzung Nguyen, Thieu Vo, Ngan Le, and Anh Nguyen. Open-vocabulary affordance detection in 3d point clouds. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5692–5698. IEEE, 2023.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Yawen Shao, Wei Zhai, Yuhang Yang, Hongchen Luo, Yang Cao, and Zheng-Jun Zha. Great: Geometry-intention collaborative inference for open-vocabulary 3d object affordance grounding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17326–17336, 2025.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Yixuan Wang, Mingtong Zhang, Zhuoran Li, Tarik Kelestemur, Katherine Rose Driggs-Campbell, Jiajun Wu, Li Fei-Fei, and Yunzhu Li. D<sup>3</sup>fields: Dynamic 3d descriptor fields for zero-shot generalizable rearrangement. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pp. 272–298. PMLR, 2025.

Lin Wu, Wei Wei, Peizhuo Yu, and Jianglin Lan. Open-vocabulary 3d affordance understanding via functional text enhancement and multilevel representation alignment. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pp. 7988–7997, 2025.

Danfei Xu, Dragomir Anguelov, and Ashesh Jain. Pointfusion: Deep sensor fusion for 3d bounding box estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

Xinli Xu, Shaocong Dong, Tingfa Xu, Lihe Ding, Jie Wang, Peng Jiang, Liqiang Song, and Jianan Li. Fusionrcnn: Lidar-camera fusion for two-stage 3d object detection. Remote Sensing, 15(7): 1839, 2023.

Yuhang Yang, Wei Zhai, Hongchen Luo, Yang Cao, Jiebo Luo, and Zheng-Jun Zha. Grounding 3d object affordance from 2d interactions in images. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10905–10915, October 2023.

Lei Yao, Yong Chen, Yuejiao Su, Yi Wang, Moyun Liu, and Lap-Pui Chau. Hammer: Harnessing mllms via cross-modal integration for intention-driven 3d affordance grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23805–23815, 2026.

Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 46766–46782, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ cc2c03ce028a1d6cbed6f98cede7f5a9-Paper-Conference.pdf.

Zhuangwei Zhuang, Rong Li, Kui Jia, Qicheng Wang, Yuanqing Li, and Mingkui Tan. Perceptionaware multi-sensor fusion for 3d lidar semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 16280–16290, 2021.

## A HYPERPARAMETER ANALYSIS

We analyze the key hyperparameters of Interaction Region Alignment and Visual Evidence Aggregation. Specifically, we investigate the memory size K, the number of warm-up epochs, the CAIM loss weight $\lambda _ { \mathrm { c a i m } }$ , the EMA coefficient, the visual attention bias coefficient $\lambda _ { v } ,$ , and the number of rendered views. All hyperparameters are selected on the LASO unseen validation split, while the main results reported in the paper are evaluated on the corresponding test split.

Table 6: Hyperparameter selection of IRA-related components on the LASO unseen split.
<table><tr><td>Hyperparameter</td><td>Value</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td>Memory size K</td><td>4</td><td>20.7</td><td>82.0</td><td>0.583</td><td>0.104</td></tr><tr><td>Memory size K</td><td>8</td><td>21.5</td><td>82.6</td><td>0.588</td><td>0.102</td></tr><tr><td>Memory size K</td><td>16</td><td>22.0</td><td>83.9</td><td>0.598</td><td>0.096</td></tr><tr><td>Memory size K</td><td>32</td><td>21.5</td><td>83.1</td><td>0.596</td><td>0.101</td></tr><tr><td>Warm-up epochs</td><td>0</td><td>20.8</td><td>82.7</td><td>0.582</td><td>0.102</td></tr><tr><td>Warm-up epochs</td><td>1</td><td>21.0</td><td>83.4</td><td>0.594</td><td>0.099</td></tr><tr><td>Warm-up epochs</td><td>3</td><td>22.0</td><td>83.9</td><td>0.598</td><td>0.096</td></tr><tr><td>Warm-up epochs</td><td>5</td><td>21.5</td><td>82.5</td><td>0.592</td><td>0.101</td></tr><tr><td>Cross-object alignment  $\lambda _ { \mathrm { c a i m } }$ </td><td>0.1</td><td>20.4</td><td>82.7</td><td>0.585</td><td>0.106</td></tr><tr><td>Cross-object alignment  $\lambda _ { \mathrm { c a i m } }$ </td><td>0.5</td><td>20.9</td><td>83.0</td><td>0.597</td><td>0.104</td></tr><tr><td>Cross-object alignment  $\lambda _ { \mathrm { c a i m } }$ </td><td>1.0</td><td>22.0</td><td>83.9</td><td>0.598</td><td>0.096</td></tr><tr><td>Cross-object alignment  $\lambda _ { \mathrm { c a i m } }$ </td><td>1.5</td><td>21.5</td><td>83.7</td><td>0.586</td><td>0.099</td></tr><tr><td>EMA coefficient</td><td>0.9</td><td>21.0</td><td>83.3</td><td>0.590</td><td>0.101</td></tr><tr><td>EMA coefficient</td><td>0.99</td><td>21.4</td><td>83.1</td><td>0.595</td><td>0.102</td></tr><tr><td>EMA coefficient</td><td>0.999</td><td>22.0</td><td>83.9</td><td>0.598</td><td>0.096</td></tr></table>

Table 7: Hyperparameter selection of VEA-related components on the LASO unseen split.
<table><tr><td>Hyperparameter</td><td>Value</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td>Attention bias  $\lambda _ { v }$  Attention bias  $\lambda _ { v }$ </td><td>0.1 0.5</td><td>20.7 21.6</td><td>81.5 82.8</td><td>0.584 0.594</td><td>0.109 0.100</td></tr><tr><td>Attention bias  $\lambda _ { v }$ </td><td>1.0</td><td>22.0</td><td>83.9</td><td>0.598</td><td>0.096</td></tr><tr><td>Attention bias  $\lambda _ { v }$ </td><td>2.0</td><td>21.3</td><td>82.1</td><td>0.589</td><td>0.103</td></tr><tr><td>Number of views</td><td>2</td><td>21.2</td><td>82.9</td><td>0.589</td><td>0.100</td></tr><tr><td>Number of views</td><td>4</td><td>21.7</td><td>83.4</td><td>0.596</td><td>0.098</td></tr><tr><td>Number of views</td><td>6</td><td>22.0</td><td>83.9</td><td>0.598</td><td>0.096</td></tr><tr><td>Number of views</td><td>8</td><td>22.0</td><td>83.7</td><td>0.601</td><td>0.098</td></tr></table>

## B QUANTITATIVE ANALYSIS OF THE QUERY–TARGET GAP

In this section, we quantify the query–target gap and examine the effects of individual modules from this perspective. As shown in Table 8, we measure the similarity between queries and target-region features under four configurations. Specifically, we first compute a target-region representation by averaging point-wise features weighted by the ground-truth affordance scores. We then calculate the cosine similarity between each query feature and this pooled representation and average the similarities across queries to obtain the query–target similarity. For comparison, we also measure the similarity between queries and non-target-region features. The results show that both IRA and VEA improve query–target similarity, with their combination achieving the highest target-region similarity.

Table 8: Query–region cosine similarity on the evaluation set. $\Delta$ denotes target similarity minus background similarity.
<table><tr><td>Method</td><td>Target ↑</td><td>Background↓</td><td> $\Delta \uparrow$ </td></tr><tr><td>Backbone</td><td>-0.009</td><td>-0.202</td><td>0.193</td></tr><tr><td> $\mathrm { B a c k b o n e + I R A }$ </td><td>0.045</td><td>-0.161</td><td>0.206</td></tr><tr><td> $\mathbf { B a c k b o n e + V E A }$ </td><td>0.021</td><td>-0.204</td><td>0.225</td></tr><tr><td> $\mathrm { B a c k b o n e + I R A + V E A }$ </td><td>0.062</td><td>-0.183</td><td>0.245</td></tr></table>

## C VISUAL ANALYSIS OF DIVERSITY-AWARE INITIALIZATION

To inspect whether DAI preserves diverse geometric patterns, we visualize geometric region features of each entry in the initialized CAIM using t-SNE. As shown in Figure 5, the red stars denote the 16 geometric features, while the gray points represent the target-region geometric features extracted from a subset of samples randomly selected from the dataset with balanced sampling across object categories. The 16 stars are widely distributed in the feature space, indicating that DAI effectively captures diverse geometric patterns.

![](images/504befa290f19a13d3faff3f2487b6cec0d0cc9e28cc950efebf350b5d107c47.jpg)  
Figure 5: t-SNE visualization of geometric region features after DAI initialization.

## D FURTHER ANALYSIS OF VISUAL EVIDENCE AGGREGATION

## D.1 ADVANTAGES OF NORMAL-BASED GAUSSIAN RENDERER

We qualitatively demonstrate that NBGR can generate more informative rendered images for the 2D affordance model. As illustrated in Figure 6, NBGR produces geometry-aware renderings, where different surface regions exhibit distinguishable appearances and object components preserve clear geometric boundaries. Such renderings provide richer spatial cues for subsequent affordance prediction.

![](images/cdf9007949ead36cc3fed95f3901524d50b3c8c71512ced1783fec0b1997bc87.jpg)  
Figure 6: Visualization of renderings generated by Normal-Based Gaussian Renderer.

## D.2 GENERALIZATION CAPABILITY OF THE 2D AFFORDANCE MODEL VERSUS THE 3D BACKBONE

As discussed in Section 4.3, directly adopting aggregated 2D masks as the final output for 3D affordance grounding leads to performance degradation. However, this does not necessarily indicate weaker generalization of 2D models. To investigate this, we project the predictions of the pure 3D backbone onto the image plane and compare them with the 2D affordance predictions. Since the projection shares the same rendering process used for data generation, the projection error is negligible, enabling a fair comparison between 2D and 3D models. As shown in Table 9, the 2D affordance model achieves better performance, highlighting its strong generalization ability and effectiveness as a visual prior for 3D affordance grounding.

Table 9: Comparison of the generalization capabilities of the 2D affordance model and the 3D backbone on the LASO unseen split.
<table><tr><td>Method</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td>3D Backbone (Projected)</td><td>21.7</td><td>91.9</td><td>0.645</td><td>0.042</td></tr><tr><td>2D Affordance Model</td><td>25.2</td><td>92.2</td><td>0.704</td><td>0.040</td></tr></table>

## D.3 DENSE POINT-WISE EVIDENCE VS. SPARSE KEYPOINT SAMPLING

In this section, we compare the sparse keypoint sampling strategy introduced by GLANCE with our CAMVA to investigate how different visual guidance strategies convert multi-view 2D affordance masks into effective 3D guidance. For a fair comparison, both variants use the same 3D backbone, 2D affordance model, multi-view predictions, and training settings, differing only in how the 2D visual evidence is aggregated and incorporated into the 3D model. The GLANCE-style strategy selects a sparse set of reliable 3D keypoints from the multi-view masks as query priors, whereas CAMVA performs confidence-weighted multi-view aggregation based on projection validity and depth consistency, producing dense point-wise visual evidence that is incorporated as a cross-attention bias.

As shown in Table 10, CAMVA consistently outperforms the sparse keypoint strategy across all metrics. These results indicate that preserving and aggregating dense point-wise confidence information provides more effective spatial guidance for target-region feature aggregation than representing the multi-view masks with a sparse set of keypoints.

Table 10: Comparison of different visual guidance strategies on the LASO unseen split. All variants use the same backbone, 2D affordance model, multi-view predictions, and training settings.
<table><tr><td>Guidance Strategy</td><td>Guidance Form</td><td>aIoU↑</td><td>AUC↑</td><td>SIM↑</td><td>MAE↓</td></tr><tr><td>GLANCE-style</td><td>Sparse keypoint guidance</td><td>21.2</td><td>81.9</td><td>0.585</td><td>0.103</td></tr><tr><td>CAMVA (Ours)</td><td>Dense point-wise evidence</td><td>21.8</td><td>83.0</td><td>0.597</td><td>0.099</td></tr></table>

## D.4 EFFECT OF VISUAL EVIDENCE AGGREGATION ON QUERY ATTENTION

To further verify that VEA helps improve the attention distribution of queries, we visualize the attention maps of queries with and without visual evidence guidance. As shown in Figure 7, without the additional 2D visual evidence, queries tend to attend to irrelevant regions on unseen objects or fail to capture the complete interaction areas. In contrast, after incorporating visual evidence through VEA, the attention distributions become more concentrated on the target regions, while suppressing responses from irrelevant background areas.

![](images/70f9467e92dab47fab416676d1dc1b98089954210fa928dcd21903c4b52cd863.jpg)  
Figure 7: Attention visualization with and without visual guidance.

## E FINE-GRAINED RESULTS BY OBJECT–AFFORDANCE COMBINATION

To provide a more detailed evaluation, Table 11 reports the performance of BridgeAfford on each object–affordance combination in the LASO unseen split across four metrics: aIoU, AUC, SIM, and MAE.

Table 11: Per-class results on the LASO unseen split.
<table><tr><td>Class</td><td>aIoU</td><td>AUC</td><td>SIM</td><td>MAE</td><td>Class</td><td>aIoU</td><td>AUC</td><td>SIM</td><td>MAE</td></tr><tr><td>bag-contain</td><td>29.40</td><td>98.21</td><td>0.796</td><td>0.092</td><td>bag-grasp</td><td>20.86</td><td>83.71</td><td>0.507</td><td>0.106</td></tr><tr><td>bag-lift</td><td>35.39</td><td>89.36</td><td>0.538</td><td>0.026</td><td>bag-open</td><td>13.90</td><td>83.21</td><td>0.523</td><td>0.077</td></tr><tr><td>bed-lay</td><td>25.38</td><td>86.16</td><td>0.625</td><td>0.103</td><td>bed-sit</td><td>13.59</td><td>85.17</td><td>0.521</td><td>0.095</td></tr><tr><td>bed-support</td><td>13.69</td><td>67.55</td><td>0.495</td><td>0.127</td><td>bottle-contain</td><td>25.42</td><td>87.56</td><td>0.570</td><td>0.122</td></tr><tr><td>bottle-grasp</td><td>26.25</td><td>81.23</td><td>0.572</td><td>0.140</td><td>bottle-open</td><td>37.19</td><td>95.83</td><td>0.517</td><td>0.047</td></tr><tr><td>bottle-pour</td><td>33.12</td><td>91.94</td><td>0.535</td><td>0.060</td><td>bottle-wrap-grasp</td><td>7.09</td><td>60.32</td><td>0.703</td><td>0.144</td></tr><tr><td>bowl-contain</td><td>29.56</td><td>95.45</td><td>0.721</td><td>0.088</td><td>bowl-pour</td><td>12.37</td><td>90.57</td><td>0.817</td><td>0.103</td></tr><tr><td>bowl-wrap-grasp</td><td>4.50</td><td>67.85</td><td>0.745</td><td>0.135</td><td>chair-move</td><td>15.86</td><td>80.17</td><td>0.559</td><td>0.123</td></tr><tr><td>chair-sit</td><td>46.93</td><td>96.48</td><td>0.751</td><td>0.063</td><td>chair-support</td><td>30.33</td><td>92.40</td><td>0.716</td><td>0.073</td></tr><tr><td>clock-display</td><td>13.74</td><td>86.73</td><td>0.513</td><td>0.071</td><td>dishwasher-contain</td><td>36.78</td><td>93.72</td><td>0.632</td><td>0.083</td></tr><tr><td>dishwasher-open</td><td>24.67</td><td>91.86</td><td>0.354</td><td>0.048</td><td>display-display</td><td>17.48</td><td>83.60</td><td>0.472</td><td>0.113</td></tr><tr><td>door-open</td><td>24.93</td><td>84.82</td><td>0.345</td><td>0.032</td><td>door-pull</td><td>27.50</td><td>92.49</td><td>0.318</td><td>0.029</td></tr><tr><td>door-push</td><td>12.30</td><td>82.82</td><td>0.555</td><td>0.066</td><td>earphone-grasp</td><td>4.05</td><td>53.95</td><td>0.199</td><td>0.197</td></tr><tr><td>earphone-listen</td><td>24.49</td><td>93.30</td><td>0.638</td><td>0.082</td><td>faucet-grasp</td><td>12.20</td><td>77.09</td><td>0.593</td><td>0.137</td></tr><tr><td>faucet-open</td><td>30.05</td><td>94.50</td><td>0.579</td><td>0.039</td><td>hat-grasp</td><td>3.93</td><td>42.74</td><td>0.508</td><td>0.157</td></tr><tr><td>hat-wear</td><td>8.39</td><td>67.53</td><td>0.582</td><td>0.131</td><td>keyboard-press</td><td>6.98</td><td>82.93</td><td>0.150</td><td>0.069</td></tr><tr><td>knife-cut</td><td>12.78</td><td>94.64</td><td>0.760</td><td>0.102</td><td>knife-grasp</td><td>21.54</td><td>87.26</td><td>0.690</td><td>0.081</td></tr><tr><td>knife-stab</td><td>20.91</td><td>99.54</td><td>0.756</td><td>0.021</td><td>laptop-display</td><td>2.48</td><td>39.88</td><td>0.067</td><td>0.171</td></tr><tr><td>laptop-press</td><td>21.16</td><td>94.42</td><td>0.559</td><td>0.050</td><td>microwave-contain</td><td>18.92</td><td>96.09</td><td>0.444</td><td>0.103</td></tr><tr><td>microwave-open</td><td>35.49</td><td>89.94</td><td>0.434</td><td>0.011</td><td>mug-contain</td><td>14.38</td><td>84.40</td><td>0.621</td><td>0.097</td></tr><tr><td>mug-grasp</td><td>42.05</td><td>93.93</td><td>0.499</td><td>0.068</td><td>mug-pour</td><td>5.16</td><td>82.91</td><td>0.620</td><td>0.082</td></tr><tr><td>mug-wrap-grasp</td><td>1.13</td><td>42.83</td><td>0.357</td><td>0.163</td><td>refrigerator-contain</td><td>26.13</td><td>92.32</td><td>0.512</td><td>0.087</td></tr><tr><td>refrigerator-open</td><td>25.16</td><td>97.39</td><td>0.427</td><td>0.029</td><td>scissors-cut</td><td>20.10</td><td>96.98</td><td>0.445</td><td>0.120</td></tr><tr><td>scissors-grasp</td><td>2.59</td><td>51.34</td><td>0.392</td><td>0.154</td><td>scissors-stab</td><td>12.66</td><td>98.64</td><td>0.308</td><td>0.045</td></tr><tr><td>storagefurniture-contain</td><td>34.88</td><td>88.03</td><td>0.605</td><td>0.075</td><td>storagefurniture-open</td><td>17.89</td><td>91.65</td><td>0.332</td><td>0.030</td></tr><tr><td>table-move</td><td>5.12</td><td>60.25</td><td>0.563</td><td>0.181</td><td>table-support</td><td>15.86</td><td>85.66</td><td>0.723</td><td>0.094</td></tr><tr><td>trashcan-contain</td><td>24.29</td><td>82.64</td><td>0.377</td><td>0.133</td><td>trashcan-open</td><td>12.09</td><td>79.58</td><td>0.280</td><td>0.074</td></tr><tr><td>trashcan-pour</td><td>2.78</td><td>40.50</td><td>0.134</td><td>0.198</td><td>vase-contain</td><td>16.90</td><td>79.18</td><td>0.503</td><td>0.122</td></tr><tr><td>vase-pour</td><td>12.73</td><td>77.57</td><td>0.584</td><td>0.109</td><td>vase-wrap-grasp</td><td>6.93</td><td>66.32</td><td>0.632</td><td>0.136</td></tr></table>

Failure Analysis. Table 11 indicates that wrap-grasping remains challenging, particularly for mugs, bowls, and bottles. One possible factor is ambiguity in the ground-truth annotations: multiple contact configurations can be physically plausible, while the annotated affordance map may capture only some of them. Consequently, a reasonable prediction may still differ from the reference in its location or coverage. Moreover, wrap-grasping involves an extended contact surface rather than a localized functional part, making it difficult to accurately match both the spatial extent and the point-wise score distribution. This is reflected in bowl-wrap-grasp and bottle-wrap-grasp, where relatively high SIM scores (0.745 and 0.703) coexist with low aIoU scores (4.50 and 7.09), suggesting partial distributional agreement without precise spatial overlap. Limited visibility across rendered views may further leave parts of these surrounding contact surfaces insufficiently supported by visual evidence. These cases suggest room for improving spatial coverage and handling ambiguity in interaction regions.