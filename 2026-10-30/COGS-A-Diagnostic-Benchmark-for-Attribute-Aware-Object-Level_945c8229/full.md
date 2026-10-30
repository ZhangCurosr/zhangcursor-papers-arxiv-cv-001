# COGS: A Diagnostic Benchmark for Attribute-Aware Object-Level 3D Gaussian Splatting Perception

Zheng Xiao<sup>1</sup> Yifan Liu<sup>2</sup> Weijian Deng<sup>3</sup> Jiashu Li<sup>2</sup> Ziqin Liu<sup>2</sup> Zhenjun Han<sup>2\*</sup> Jianbin Jiao<sup>2</sup>

<sup>1</sup>School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences, China <sup>2</sup>University of Chinese Academy of Sciences, China <sup>3</sup>Tsinghua Shenzhen International Graduate School, China

xiaozheng19@mails.ucas.ac.cn, hanzhj@ucas.ac.cn

## Abstract

3D Gaussian Splatting (3DGS) is increasingly explored as an input representation for 3D perception. Yet it remains unclear how current 3D architectures exploit heterogeneous Gaussian attributes. This ambiguity makes accuracy hard to interpret and limits principled architecture design for 3DGS perception. However, existing object-level 3DGS datasets lack the appearance variation or artifact control neededfor systematic attribute-level analysis. We introduce COGS, a large-scale 3DGS dataset with 15,675 artifactfiltered and appearance-rich objects across 40 categories. Built on COGS, we establish COGS-Benchfor object classification and attribute-usage diagnosis, complemented by external validation. Our benchmark shows that Gaussian attributes are not uniformly beneficial: selected subsets can outperform the full attribute set, and the effects of appearance, opacity, and covariance-related attribute combinations vary substantially across architectures. We further study this finding with HALO, a reference model that preserves attribute-specific evidence before hierarchical aggregation. Experiments on COGS-Bench and external 3DGS datasets demonstrate that COGS supports systematic diagnosis of 3DGS attribute usage, while HALO highlights the value of role-preserving attribute organization. Our dataset is available at https://huggingface.co/ datasets/XiaoZhengCV/COGS.

## 1. Introduction

3D Gaussian Splatting (3DGS) has become a high-fidelity and real-time 3D representation for novel-view synthesis [12]. Beyond rendering, recent studies increasingly explore 3DGS for 3D perception. Some methods [2, 6, 8, 22,

![](images/163daac8da7c856c36b1566451ca5e9e4214838f577284ba669d5a0f270ac19c.jpg)  
Figure 1. Overview of COGS and COGS-Bench. Existing object-level 3DGS datasets provide complementary but limited properties, motivating COGS as a curated, artifact-filtered, and appearance-rich dataset for controlled attribute diagnosis.

31, 33, 39] use Gaussian primitives as carriers of distilled 2D semantic or language features [2, 19, 24] through multiview feature lifting and language-field optimization. More recent works directly process Gaussian primitives for object recognition and scene understanding [15, 18, 37]. These studies demonstrate the potential of 3DGS for perception, but they also expose a representation-level ambiguity: it remains unclear whether current 3D architectures truly exploit Gaussian-specific attributes, or instead obtain their gains from positional or appearance shortcuts.

This ambiguity is important because 3DGS primitives are not ordinary point samples with extra feature channels. Each primitive is parameterized by position, opacity, scale, rotation, and appearance coefficients. These attributes are jointly optimized for rendering, but they encode different aspects. Position specifies spatial support, scale and rotation describe anisotropic extent, opacity is related to visibility, and appearance coefficients provide color-related cues. Therefore, the benefit of 3DGS for perception should not be viewed simply as adding more input channels to point coordinates. Treating these heterogeneous attributes as homogeneous point features may underuse informative cues or introduce interference. A systematic benchmark is therefore needed to disentangle architecture design and attribute usage under unified protocols.

Scene-level 3DGS datasets are essential for evaluating scalability and contextual reasoning [15], but recognition in scenes is entangled with background context, occlusion, layout, and co-occurrence, making them less suitable for controlled attribute diagnosis. Object-level datasets provide a complementary setting, yet existing datasets remain limited as shown in Fig. 1. CAD-derived datasets such as ModelNet40GS [18] provide clean geometry but limited appearance variation, while capture-based datasets such as MACGS [37] contain realistic observations but may include reconstruction artifacts. A diagnostic benchmark for 3DGS perception should therefore provide object-level controllability, rich appearance cues, artifact-filtered assets, and full Gaussian attribute records.

To this end, we introduce COGS, Common Objects of 3D Gaussian Splatting, a large-scale object-centric 3DGS dataset for controlled analysis of Gaussian attribute utilization. COGS contains 15,675 objects across 40 categories and is constructed from diverse object images through a curated generative pipeline with multi-stage filtering and manual 3DGS inspection. The resulting dataset provides rich appearance variation, artifact-filtered structures, and full Gaussian attribute records for controlled attribute-level diagnosis. Built on COGS, we establish COGS-Bench for object classification and attribute-usage diagnosis, evaluating recognition accuracy, efficiency, and attribute combinations under consistent protocols, with additional validation on other 3DGS datasets. As an additional study, we examine COGS as a pre-training source for object classification on MACGS. Beyond diagnosis, COGS-Bench also enables us to examine an attribute-aware modeling principle: models should preserve attribute-specific evidence from Gaussian attributes with different representational roles before high-level semantic aggregation. We instantiate this principle with HALO, Hierarchical Aggregation with LocallydecOupled MLP, as a reference model. HALO separately extracts local evidence from Gaussian attribute groups and fuses them through hierarchical aggregation. HALO provides a concrete example of translating attribute-usage diagnosis into architecture design for 3DGS perception.

Our contributions are summarized as follows:

• We introduce COGS, a large-scale 3DGS dataset containing 15,675 objects across 40 categories, with artifactfiltered structures and rich appearance variation.

• We establish COGS-Bench for object classification and attribute-usage diagnosis under consistent protocols, with cross-pipeline validation on external 3DGS datasets.

• We provide systematic evidence that Gaussian attributes are not uniformly beneficial across architectures and that more attributes do not always lead to better recognition.

• We provide HALO, an attribute-aware reference model that preserves attribute-specific evidence before hierarchical semantic aggregation, illustrating how attributeusage diagnosis can inform model design.

## 2. Related Work

## 2.1. Semantic 3DGS via 2D Feature Lifting

Recent studies have extended 3DGS beyond photorealistic rendering by associating semantic or language features with Gaussian primitives. Feature 3DGS [39] learns semantic feature fields on top of 3DGS, enabling semantic segmentation and language-guided editing. LangSplat [22] distills and compresses CLIP-related language features into 3D Gaussians, constructing a compact 3D language field for efficient open-vocabulary localization. Gaussian Grouping [33] introduces an identity encoding for each Gaussian and supervises it with multi-view 2D masks and 3D consistency constraints, enabling object grouping, 3D segmentation, and local editing. Segment Any 3D Gaussians [2] and Click-Gaussian [6] further leverage 2D segmentation masks or interactive cues for 3DGS segmentation.

## 2.2. Direct 3DGS Perception and Benchmarks

Beyond using 3DGS as a carrier for distilled 2D features, recent works have begun to treat Gaussian primitives themselves as inputs for 3D perception. MACGS [37] studies object classification with 3DGS. It constructs an object-level 3DGS dataset from real-world captures in MVImgNet [36] and evaluates multiple point-cloud classification architectures by comparing position-only inputs with non-color Gaussian attributes. ShapeSplat [18] builds object-level 3DGS datasets from commonly used 3D asset datasets [4, 35]. It further introduces Gaussian-MAE, which adapts masked autoencoding from point clouds to Gaussian primitives for self-supervised representation learning. SceneSplat [15] extends direct Gaussian primitive processing to scene-level understanding. It constructs 3DGS scenes from indoor datasets [1, 3, 7, 25–27, 34] and adopts a PTv3-style backbone [30] with self-supervised and vision-language pretraining, supporting both closed-set semantic segmentation and open-vocabulary scene understanding.

![](images/682e43c289385073905ec7bc8654e5cc6ddeef15f261f9f2bdcfa374b7c9d92f.jpg)  
Figure 2. Overview of the COGS construction pipeline. Starting from large-scale web image collection, we perform semantic localization, foreground-aware filtering and deduplication, manual image inspection, generative 3DGS reconstruction with TRELLIS, and viewer-based 3DGS quality inspection to obtain curated object-centric 3DGS assets.

![](images/bda717733e9ea57022afdfd3d1fc9dc6d5ba94e5504ac2301f4ab67583f95153.jpg)  
Figure 3. Discarded 3DGS samples rendered from multiple views. Row 1 shows view-inconsistent structural duplication, where drawer features expected on one face also appear on adjacent faces; row 2 shows unintended fusion between two lipstick bodies; row 3 shows incomplete chair geometry.

## 3. COGS and COGS-Bench

## 3.1. Dataset Construction

Capture-derived object assets can suffer from unobserved contact regions and mask leakage, producing incomplete geometry or floating artifacts. COGS therefore uses the scalable generative pipeline in Fig. 2 as a complementary object-centric testbed with reduced capture noise, rich appearance, and full Gaussian attributes, rather than as a replacement for captured datasets.

Image collection and pre-filtering. We construct COGS from 227,782 raw object images discovered through established image-search services using category-specific queries. During acquisition, our automated data-collection pipeline checks license fields exposed in the search results together with the associated source-page metadata. We retain only images marked as CC0 or Public Domain. Samples with missing, conflicting, or unverifiable license information, NoDerivatives restrictions, or other terms incompatible with reuse, modification, and redistribution are discarded. Generic category names and fine-grained descriptors of material, style, shape, and subtype improve intraclass coverage; their overlapping results also motivate the subsequent deduplication. Blank images and exact duplicates are removed first.

Semantic localization. Web images often contain multiple objects, cluttered backgrounds, or irrelevant content. To obtain object-centric inputs, we localize the target object using text-prompted object detection with GroundingDINO [16]. Detections with confidence < 0.7, object area < 2.25%, edge distance < 5 pixels, or redundant overlap (IoU > 0.8) are discarded; the remaining boxes are cropped with a small margin and re-centered.

Foreground-aware filtering and deduplication. U2-Net foreground masks [23] reduce background bias in similarity estimation. For incomplete or occluded foregrounds, diffusion inpainting [13] provides more complete appearances for CLIP feature extraction. These inpainted images are used only for CLIP-based deduplication. Samples with cosine similarity > 0.9 are deduplicated [24]. Manual image inspection then removes severe truncation, incorrect or ambiguous categories, heavy occlusion, low visual quality, and unsuitable viewpoints.

![](images/f13ebde64080a0f6822802132e7e97ef564d011a85d0869e5955d53579c54e5b.jpg)  
Figure 4. Category distribution comparison. Categories in each dataset are independently sorted by object count. The x-axis represents dataset-specific sorted category indices rather than aligned category identities. MACGS statistics use the authors’ publicly released 30-category dataset.

Generative 3DGS reconstruction and quality assurance. Each retained image is converted into a 3DGS asset using TRELLIS [32]. Three annotators independently inspect each sample in the SuperSplat Editor by rotating and zooming it through arbitrary viewpoints. As illustrated in Fig. 3, we flag (i) view-inconsistent structural duplication, where a feature expected on one surface is spuriously repeated on adjacent surfaces; (ii) unintended fusion between parts that should remain separate; and (iii) incomplete geometry with missing major structures. Only assets unanimously judged acceptable are retained; this conservative qualitycontrol stage reduces 23,832 candidates to 15,675 assets.

Overall, the pipeline reduces 227,782 images to 15,675 curated 3DGS objects across 40 categories, prioritizing object validity, diversity, and artifact control over scale. Figure 4 shows the distribution; category counts and additional visualizations are provided in the supplement.

## 3.2. COGS-Bench Protocol

Built on COGS, we establish COGS-Bench for object classification and attribute-usage diagnosis. The dataset is released for non-commercial research and educational purposes. Rights-holder removal requests will be supported.

Object classification. The primary task is 40-class classification, evaluated by overall accuracy (OA) and mean class accuracy (mAcc). Following the common train/test-only benchmark protocol in point-cloud classification [20, 29], we use no separate validation split. To control the longtailed distribution, we use a fixed training set of 100 instances per category (4,000 in total), with the remaining 11,675 instances reserved for testing.

Attribute usage diagnosis. COGS-Bench evaluates eight input configurations: P, PO, PC, POC, PSQ, POSQ, PSQC, and POSQC, where P, O, S, Q, and C denote position, opacity, scale, quaternion/rotation, and appearance/color. Position is retained as the common structural reference because the evaluated backbones use it for neighborhood construction and sampling; removing it would change the backbone rather than only its input attributes. Since scale and rotation jointly parameterize anisotropic covariance, SQ is treated as one functional group. The eight settings therefore cover all combinations of the optional groups O, C, and SQ, with P always included. This protocol compares how attribute combinations affect recognition across the evaluated architectures.

External validation. We repeat the eight-configuration attribute diagnosis on MACGS [37] and further examine attribute usage and attribute-specific processing in 3DGS semantic segmentation on ScanNet [15]. These experiments assess whether the diagnostic findings extend to capturederived objects and real-world scenes.

## 4. HALO: Attribute-aware Reference Model

## 4.1. Motivation and Overview

HALO is designed to examine an attribute-aware modeling principle suggested by the heterogeneous nature of 3DGS primitives: models should preserve attribute-specific evidence associated with attributes that play different representational roles. Directly treating them as a single descriptor may force one encoder to learn attribute-specific patterns and cross-attribute interactions simultaneously. Based on this principle, we introduce HALO, Hierarchical Aggregation with Locally-decOupled MLP, as an attribute-aware reference model for COGS-Bench. HALO first extracts local evidence within attribute-specific streams and then fuses the resulting descriptors through hierarchical aggregation. As shown in Fig. 5, each hierarchy level contains a Feature Extraction (FE) module and a Hierarchical Aggregation (HA) module. FE combines attribute-specific local descriptors with point-wise features, while HA aggregates neighborhood context before downsampling. The final representation is globally pooled for object-level recognition.

## 4.2. Attribute-aware Feature Extraction

The FE module is designed to avoid premature homogenization of Gaussian attributes. Let $A = \{ p , o , s , q , c \}$ denote the attribute groups, corresponding to position, opacity, scale, rotation, and color, and let $\mathbf { z } _ { i } ^ { a } \in \mathbb { R } ^ { d _ { a } }$ be the value of attribute group a for primitive i. For attribute-specific encoding in the decoupled FE module, we convert normalized quaternions to axis-angle vectors and DC sphericalharmonic coefficients to RGB, which serve as the rotation and color inputs, respectively. For each primitive, HALO first extracts local descriptors in attribute-specific streams and then fuses the resulting features.

![](images/c0d6734c2eb770a3440bf66dd9992da58dc325ddf40bda4d9b4a42af08c76b59.jpg)  
Figure 5. Overview of HALO. Given a set of Gaussian primitives, HALO progressively builds hierarchical representations through repeated Feature Extraction (FE), Hierarchical Aggregation (HA), and downsampling. FE performs multi-stream attribute encoding with Locally decOupled MLP (LOM), while HA aggregates neighborhood evidence before primitive decimation.

For primitive $i ,$ we construct a k-nearest-neighbor set $\mathcal { N } _ { k } ( i )$ in coordinate space. For the m-th component of attribute group a, we form a compact neighbor sequence

$$
\Delta \mathbf { z } _ { i , m } ^ { a } = [ z _ { j _ { 1 } , m } ^ { a } - z _ { i , m } ^ { a } , \ldots , z _ { j _ { k } , m } ^ { a } - z _ { i , m } ^ { a } ] , j _ { r } \in \mathcal { N } _ { k } ( i ) ,\tag{1}
$$

where neighbors are ordered by Euclidean distance to obtain a fixed-length sequence. The Locally-decOupled MLP (LOM) projects each component-wise sequence and concatenates the component descriptors:

$$
\begin{array} { r } { \mathbf { l } _ { i } ^ { a } = \mathrm { C a t } _ { m = 1 } ^ { d _ { a } } \rho _ { a } ( \Delta \mathbf { z } _ { i , m } ^ { a } ) , } \end{array}\tag{2}
$$

where $\rho _ { a }$ is an attribute-specific sequence projection. Unlike per-neighbor local MLPs that repeatedly process $N \times k$ neighbor features, LOM treats each component-wise neighborhood as a compact sequence. The neighbor dimension is mainly handled in the first projection layer, making multistream attribute modeling more efficient.

In parallel, HALO encodes point-wise attribute values with two mappings rather than one mapping for each attribute. Specifically, we keep position as an individual point-wise stream and concatenate all non-position attributes into a second stream:

$$
\begin{array} { r } { \mathbf { u } _ { i } ^ { p } = \eta _ { p } ( \mathbf { p } _ { i } ) , \quad \mathbf { u } _ { i } ^ { n p } = \eta _ { n p } ( \operatorname { C a t } ( o _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } , \mathbf { c } _ { i } ) ) , } \end{array}\tag{3}
$$

where $\eta _ { p }$ and $\eta _ { n p }$ are point-wise MLPs for the position and non-position streams, respectively. The FE output is obtained by fusing local and point-wise descriptors in the position and non-position streams:

$$
F _ { i } ^ { \mathrm { F E } } = \phi _ { p } \big ( \operatorname { C a t } ( 1 _ { i } ^ { p } , \mathbf { u } _ { i } ^ { p } ) \big ) + \phi _ { n p } \big ( \operatorname { C a t } ( 1 _ { i } ^ { n p } , \mathbf { u } _ { i } ^ { n p } ) \big ) ,\tag{4}
$$

where $\phi _ { p }$ and $\phi _ { n p }$ are projection functions, and $\mathbf { l } _ { i } ^ { n p } \ =$ $\mathrm { C a t } ( \mathbf { l } _ { i } ^ { o } , \dot { \mathbf { l } _ { i } ^ { s } } , \mathbf { l } _ { i } ^ { q } , \mathbf { l } _ { i } ^ { c } )$ . This formulation delays cross-attribute interaction until after attribute-specific local modeling.

## 4.3. Hierarchical Aggregation

HALO progressively reduces the number of primitives to build an efficient hierarchical encoder. However, directly downsampling Gaussian primitives may remove local evidence that is useful for recognition. The HA module therefore aggregates neighborhood context before each decimation step.

At hierarchy level l, let $F _ { l } ^ { \mathrm { F E } }$ be the output of FE. For $l > 1$ , features inherited from the previous level are aligned to the current sampled primitives and fused with current features:

$$
\begin{array} { r } { F _ { l } ^ { \mathrm { f u s e } } = U _ { l } + \theta _ { l } ( U _ { l } ) , \quad U _ { l } = F _ { l } ^ { \mathrm { F E } } + D _ { l } ( F _ { l - 1 } ) , } \end{array}\tag{5}
$$

where $D _ { l } ( \cdot )$ denotes index-based downsampling with channel projection, and $\theta _ { l }$ is a residual [9] mapping. For the first level, ${ \bar { F } } _ { 1 } ^ { \mathrm { f u s e } } = F _ { 1 } ^ { \mathrm { F E } }$

Given a feature set $\begin{array} { r c l } { H } & { = } & { \{ { \bf h } _ { i } \} } \end{array}$ , an edge-pooling block [14] retains crucial local information within each neighborhood:

$$
\Omega _ { l } ( H ) _ { i } = \operatorname* { m a x } _ { j \in \mathcal { N } _ { k } ( i ) } \big ( \widehat { \mathbf { h } } _ { j } - \widehat { \mathbf { h } } _ { i } \big ) , \quad \widehat { \mathbf { h } } _ { i } = \omega _ { l } ( \mathbf { h } _ { i } ) ,\tag{6}
$$

where $\omega _ { l }$ is a linear projection and the maximum is taken channel-wise over neighbors. Two residual edge-pooling blocks are stacked before downsampling:

$$
\begin{array} { r l } & { \widetilde { F } _ { l } = F _ { l } ^ { \mathrm { f u s e } } + \Omega _ { l , 1 } ( F _ { l } ^ { \mathrm { f u s e } } ) , } \\ & { F _ { l } = \gamma _ { l } ( \widetilde { F } _ { l } ) + \Omega _ { l , 2 } ( \widetilde { F } _ { l } ) , } \end{array}\tag{7}
$$

where $\gamma _ { l }$ aligns feature dimensions when necessary. The resulting features are passed to the next hierarchy level after downsampling. In this way, HALO follows an aggregationbefore-decimation strategy: local evidence around retained primitives is first absorbed into their features, and primitive density is then reduced for efficient high-level reasoning.

Table 1. Rendered visual quality comparison. We report MUSIQ and CLIPIQA+ scores on rendered 48-view and single-view images. Standard deviations are shown in parentheses.
<table><tr><td rowspan="3">Dataset</td><td colspan="2">48 views</td><td colspan="2">Single view</td></tr><tr><td>MUSIQ (↑)</td><td>CLIPIQA+ (↑)</td><td>MUSIQ (↑)</td><td>CLIPIQA+ (↑)</td></tr><tr><td>ModelNet40GS [18]</td><td>58.71 (±3.39)</td><td>0.5228 (±0.0158)</td><td>52.25 (±5.33)</td><td>0.4969 (±0.0162)</td></tr><tr><td>MACGS [37]</td><td>38.64 (±3.87)</td><td>0.4509 (±0.0256)</td><td>49.94 (±5.23)</td><td>0.5172 (±0.0233)</td></tr><tr><td>COGS</td><td>64.75 (±3.84)</td><td>0.6013 (±0.0278)</td><td>63.21 (±4.54)</td><td>0.5863 (±0.0343)</td></tr></table>

## 5. Experiments and Diagnostic Analysis

Our experiments cover dataset quality and characterization, classification and attribute usage, HALO component ablations, and external validation. We additionally examine COGS-to-MACGS transfer.

## 5.1. Experimental Setup

For classification, we sample 1,024 primitives from each 3DGS object using FPS [21]. Before FPS, COGS objects undergo random rotations followed by removal of a spatially contiguous region containing 5–20% of the primitives around a randomly selected primitive. Across datasets, we activate opacity and scale using sigmoid and exponential functions, respectively, center positions and normalize positions and scales by the object radius, normalize quaternions, and retain only DC appearance coefficients. All classification experiments use random seed $4 2 .$ . Unless otherwise specified, supervised models are trained from random initialization under the same protocol. We train each model for 200 epochs using AdamW [17] with a batch size of 32 on a single RTX 4090 GPU. The learning rate is initialized to $1 \times 1 0 ^ { - 3 }$ and decayed to $1 \times 1 0 ^ { - 5 }$ using a cosine annealing scheduler. Gaussian-MAE [18] follows its masked pre-training protocol. For HALO, we evaluate two variants, HALO-B as the recommended accuracy–efficiency configuration and HALO-L as a capacity-scaling variant. Both use a three-level hierarchy containing 1,024, 256, and 64 primitives, with 20 neighbors at every level. HALO-B uses feature dimensions [64, 128, 256], while HALO-L widens them to [256, 512, 1024].

## 5.2. Dataset Quality and Characterization

Rendered visual quality. We assess 3DGS asset quality using rendered RGB views, where severe geometry or appearance defects become directly observable. To reduce bias from the long-tailed category distributions, stratified sampling selects up to 80 objects per category with seed 42, yielding 3,200 COGS, 3,200 ModelNet40GS, and 2,345 MACGS objects. Scores are macro-averaged over categories. For each selected instance, we render 48 views on four spherical elevation rings (0<sup>◦</sup>, 15<sup>◦</sup>, 30<sup>◦</sup>, and 45<sup>◦</sup>), with 12 uniformly spaced azimuths per ring. Adjacent rings are offset by $1 5 ^ { \circ }$ in azimuth to improve coverage. The singleview setting uses the canonical front view at $0 ^ { \circ }$ elevation and $0 ^ { \circ }$ azimuth. We use two no-reference image quality metrics: MUSIQ [11] for perceptual quality and technical distortions, and CLIPIQA+ [28] for high-level perceptual realism. As shown in Tab. 1, COGS achieves the best results in both single-view and 48-view settings. MACGS [37] obtains a competitive single-view CLIPIQA+ score, but its 48- view scores drop substantially, suggesting that artifacts and incomplete geometry become more visible under broader viewpoint coverage. In contrast, COGS maintains strong scores across both settings. Together with the manual 3DGS inspection in Sec. 3.1, these results support COGS as a curated object-level benchmark for controlled 3DGS attribute diagnosis.

![](images/410c4f41f6daf9778a8848640fe5c362167e2789005fa838166b7fafbb8cd4c4.jpg)  
Figure 6. Gaussian attribute statistics across COGS, MACGS, and ModelNet40GS. We compare primitive count, high-opacity ratio (>0.8), normalized Gaussian volume, and color saturation.

Gaussian attribute statistics. We next compare Gaussian attribute statistics: the number of Gaussian primitives, the fraction of primitives with activated opacity $> 0 . 8$ , the mean log-normalized Gaussian volume, and the mean color saturation. For a primitive with scales $( s _ { x } , s _ { y } , s _ { z } )$ and object bounding-box diagonal $d _ { \mathrm { b b o x } } ,$ the normalized volume statistic is computed as $\log _ { 1 0 } ( s _ { x } s _ { y } s _ { z } / d _ { \mathrm { b b o x } } ^ { 3 } )$ and then averaged over primitives. For color, we convert the degree-0 spherical harmonic coefficients to RGB, compute the channel range max $( R , G , B ) - \operatorname* { m i n } ( R , G , B )$ for each primitive, and average it over the object. Fig. 6 summarizes these distributions using boxplots. The three datasets exhibit markedly different distributions. COGS contains more primitives and a higher ratio of high-opacity Gaussians. ModelNet40GS has fewer primitives and near-zero color saturation because it is derived from CAD models without rich texture information. MACGS lies between them in primitive count, but shows lower high-opacity ratios and more concentrated color statistics. These dataset-specific attribute distributions motivate the external validation in

![](images/f0c1158aecdf93790275d6200374e84c13175fd569c5de3f0af3bd439397f51c.jpg)  
Figure 7. Intra-class pairwise cosine distance distribution in the DINOv2 feature space. COGS shows the largest spread, indicating richer intra-class diversity than ModelNet40GS and MACGS.

Table 2. Classification benchmark on COGS with all Gaussian attributes as input.
<table><tr><td>Method</td><td>OA (% ↑)</td><td>mAcc (% ↑)</td><td>Mem. (MB ↓)</td><td>Time (min ↓)</td><td>Params (M ↓)</td></tr><tr><td>PointNet [20]</td><td>79.53</td><td>79.70</td><td>1112</td><td>17.40</td><td>3.47</td></tr><tr><td>DGCNN [29]</td><td>85.69</td><td>86.70</td><td>5411</td><td>61.09</td><td>1.81</td></tr><tr><td>PTv1 [38]</td><td>87.67</td><td>88.32</td><td>744</td><td>72.80</td><td>3.26</td></tr><tr><td>DeLA [5]</td><td>88.48</td><td>89.14</td><td>1771</td><td>33.06</td><td>5.34</td></tr><tr><td>PTv3 [30]</td><td>89.11</td><td>89.77</td><td>5710</td><td>212.15</td><td>38.87</td></tr><tr><td>Gaussian-MAE [18]</td><td>88.20</td><td>88.58</td><td>2876</td><td>49.31</td><td>22.10</td></tr><tr><td>HALO-B (Ours, Ref.)</td><td>94.50</td><td>94.70</td><td>750</td><td>39.60</td><td>2.68</td></tr><tr><td>HALO-L (Ours, Ref.)</td><td>94.70</td><td>94.91</td><td>1786</td><td>41.60</td><td>10.46</td></tr></table>

## Sec. 5.4.

Intra-class diversity. We further measure intra-class diversity in the rendered image feature space. For each object, we render four views, extract DINOv2 features [19], and average the view-level features into an object-level representation. We then compute intra-class pairwise cosine distances. Fig. 7 shows the kernel density estimation of the resulting distance distributions. COGS has the largest spread, with a standard deviation of 0.1993, compared with 0.1426 for ModelNet40GS and 0.1016 for MACGS. This suggests that COGS contains richer rendered visual intraclass variation.

## 5.3. COGS-Bench

Classification benchmark. Tab. 2 reports classification results on COGS using all Gaussian attributes. PTv3 [30] and Gaussian-MAE [18] reach 89.11% and 88.20% OA, while the attribute-aware HALO-B/L references obtain 94.50%/94.70%. These results show that COGS-Bench is non-trivial for existing architectures and provides a standardized testbed for studying 3DGS attribute modeling. Supplementary Table S4 further reports counted FLOPs, measured inference latency and throughput, and peak inference memory.

Attribute usage diagnosis. Beyond final classification accuracy, COGS-Bench enables a controlled diagnosis of how architectures use heterogeneous Gaussian attributes. We denote position, opacity, scale, quaternion/rotation and appearance/color as P, O, S, Q, and C, respectively. Each entry in Tab. 3 reports OA/mAcc (%) under a specific attribute subset. The results show that adding more Gaussian attributes does not uniformly improve recognition. For earlyconcatenation backbones, selected attribute subsets can outperform the full POSQC input. For example, PointNet [20] and DGCNN [29] perform best with POSQ rather than POSQC. These results indicate that simply adding more Gaussian attributes does not guarantee better recognition, and that heterogeneous attributes may introduce optimization or fusion conflicts when they are collapsed into a single feature space too early. COGS-Bench therefore diagnoses architecture-dependent attribute utilization rather than prescribing a universal best subset. HALO-B/L are reference responses to this diagnosis: after attribute-specific encoding and delayed fusion, both attain their best OA with full POSQC. HALO-B improves from 89.66%/89.88% with position only to 91.31%/91.85% with PO, 93.70%/93.95% with POSQ, and 94.50%/94.70% with POSQC. HALO-L follows the same trend. Overall, these results highlight the main purpose of COGS-Bench: it is not merely an accuracy leaderboard, but a diagnostic benchmark for studying when and how Gaussian attributes contribute to object-level 3DGS perception.

Ablation on attribute-specific processing. We ablate HALO-B by replacing its attribute-specific feature extraction with a shared stream over concatenated Gaussian attributes. As shown in Tab. 4, shared processing reduces performance from 94.50%/94.70% to 89.03%/89.67% in OA/mAcc, confirming that preserving attribute-specific evidence before fusion is critical for HALO.

## 5.4. External Validation

Object-level cross-pipeline validation. We repeat the attribute diagnosis on capture-derived MACGS. With full results in Supplementary Tab. S2, Tab. 5 summarizes the complete eight-configuration diagnosis. All six conventional backbones attain higher OA with at least one attribute subset than with POSQC, with gains of 0.28–3.05 points. In contrast, HALO-B/L attain their best OA with full POSQC, reaching 93.99%/94.55%. The COGS diagnosis therefore persists on a capture-derived distribution with substantially different Gaussian statistics. In addition Supplementary Tab. S6 reports the full-POSQC comparison on capture-derived MACGS and CAD-derived Model-Net40GS.

Real-world scene-level validation. We further test subsetover-full behavior and the benefit of attribute-specific processing in 3DGS semantic segmentation on ScanNet [15] using the official 10% Limited-Reconstruction split [10]. As summarized in Tab. 6 and detailed in Supplementary Tab. S3, the complete PTv1 diagnosis shows that PO,

Table 3. Attribute usage diagnosis on COGS-Bench. P, O, S, Q, and C denote position, opacity, scale, rotation, and appearance/color, respectively. Each entry reports OA/mAcc (%).
<table><tr><td>Method</td><td>P</td><td>PO</td><td>PC</td><td>POC</td><td>PSQ</td><td>POSQ</td><td>PSQC</td><td>POSQC</td></tr><tr><td>PointNet [20]</td><td>73.25/68.48</td><td>80.73/79.84</td><td>76.52/75.82</td><td>77.27/77.24</td><td>81.23/79.31</td><td>82.71/81.30</td><td>78.93/78.49</td><td>79.53/79.70</td></tr><tr><td>DGCNN [29]</td><td>85.87/86.15</td><td>87.04/87.31</td><td>82.79/83.84</td><td>84.17/85.20</td><td>88.57/88.42</td><td>89.18/89.36</td><td>85.29/86.08</td><td>85.69/86.70</td></tr><tr><td>PTv1 [38]</td><td>86.45/86.43</td><td>87.10/87.65</td><td>87.85/88.34</td><td>88.09/88.76</td><td>88.54/88.27</td><td>88.82/89.26</td><td>88.46/89.15</td><td>87.67/88.32</td></tr><tr><td>DeLA [5]</td><td>89.28/89.97</td><td>91.21/91.68</td><td>83.17/84.31</td><td>85.45/86.54</td><td>91.17/91.33</td><td>90.40/91.00</td><td>87.58/88.49</td><td>88.48/89.14</td></tr><tr><td>PTv3 [30]</td><td>88.53/88.80</td><td>89.50/89.89</td><td>89.16/89.16</td><td>88.58/88.89</td><td>88.74/89.01</td><td>89.68/89.82</td><td>88.44/88.68</td><td>89.11/89.77</td></tr><tr><td>Gaussian-MAE [18]</td><td>87.01/86.32</td><td>89.18/89.49</td><td>85.89/85.79</td><td>86.56/87.43</td><td>91.71/91.09</td><td>89.10/88.96</td><td>87.01/87.18</td><td>88.20/88.58</td></tr><tr><td>HALO-B (Ours, Ref.)</td><td>89.66/89.88</td><td>91.31/91.85</td><td>91.82/92.63</td><td>92.35/93.09</td><td>93.34/93.07</td><td>93.70/93.95</td><td>93.93/94.00</td><td>94.50/94.70</td></tr><tr><td>HALO-L (Ours, Ref.)</td><td>90.56/91.07</td><td>92.23/92.82</td><td>92.57/93.11</td><td>93.13/94.02</td><td>93.88/94.12</td><td>94.30/94.40</td><td>94.54/94.94</td><td>94.70/94.91</td></tr></table>

Table 4. Ablation on attribute-specific processing in HALO-B.
<table><tr><td>HALO-B</td><td>OA (% ↑)</td><td>mAcc (% ↑)</td></tr><tr><td>Shared attribute processing</td><td>89.03</td><td>89.67</td></tr><tr><td>Attribute-specific processing</td><td>94.50</td><td>94.70</td></tr></table>

Table 5. Cross-pipeline attribute diagnosis on MACGS (OA).
<table><tr><td>Models</td><td>Diagnostic result (OA, %)</td></tr><tr><td>6 conventional baselines 6/6: subset &gt; POSQC (+0.28–3.05) HALO-B / HALO-L</td><td>POSQC best: 93.99 / 94.55</td></tr></table>

Table 6. Scene-level validation on ScanNet.
<table><tr><td>Method</td><td>Variant/Input</td><td>mIoU (%)</td><td>OA (%)</td></tr><tr><td>PTv1</td><td>POSQC</td><td>48.45</td><td>80.95</td></tr><tr><td rowspan="3">HALO</td><td>POSQ</td><td>50.37</td><td>81.08</td></tr><tr><td>Shared POSQC</td><td>43.81</td><td>79.16</td></tr><tr><td>Attribute-specific POSQC</td><td>49.83</td><td>81.96</td></tr></table>

POSQ, and PSQC all outperform POSQC in mIoU, with POSQ improving from 48.45% to 50.37% (+1.92). This is composition-dependent rather than a general preference for fewer attributes, since the other subsets perform below POSQC. Under the identical full POSQC input, attributespecific HALO improves shared processing from 43.81% to 49.83% mIoU (+6.02). These results extend both diagnostic findings from COGS to captured objects and real-world scenes.

## 5.5. Transfer and Low-Shot Adaptation

As an additional study, we examine COGS as a pre-training source for object classification on MACGS. For COGS pretraining, we use a fixed train/test split that assigns 80% of the objects in each category to training and the remaining 20% to testing. Each model is then fully fine-tuned using 10%, 20%, 50%, or 100% of the MACGS training instances in each class. Scratch and pre-trained runs differ only in initialization. We consider representative backbone families, including point-wise MLPs (PointNet [20]), graph-based local aggregation (DGCNN [29]), transformer-based attention (PTv1 [38]), and HALO-B as the attribute-aware reference model. Despite the distribution gap in Fig. 6, Tab. 7 shows consistent gains from COGS pre-training. With only 10% MACGS training data, OA improves by 26.98, 21.91, 12.01, and 3.23 points for PointNet, DGCNN, PTv1, and HALO-B, respectively. The gains decrease as more target data becomes available but remain positive for every evaluated architecture. Full scratch and pre-trained OA/mAcc results are provided in Supplementary Tab. S1.

Table 7. Models are fine-tuned on varying fractions of MACGS using either random initialization or COGS pre-training. Each cell reports pre-trained OA and OA gain over training from scratch. Full results are provided in the supplementary material.
<table><tr><td>Method</td><td>10%</td><td>20%</td><td>50%</td><td>100%</td></tr><tr><td>PointNet [20]</td><td>56.65 (+26.98)</td><td>64.33 (+18.58)</td><td>76.25 (+7.03)</td><td>81.89 (+2.13)</td></tr><tr><td>DGCNN [29]</td><td>60.54 (+21.91)</td><td>71.35 (+12.20)</td><td>81.79 (+6.00)</td><td>88.82 (+1.67)</td></tr><tr><td>PTv1 [38]</td><td>62.29 (+12.01)</td><td>73.29 (+8.41)</td><td>84.94 (+4.81)</td><td>90.11 (+1.85)</td></tr><tr><td>HALO-B</td><td>69.87 (+3.23)</td><td>82.72 (+2.04)</td><td>90.29 (+1.10)</td><td>94.08 (+0.09)</td></tr></table>

## 6. Conclusion and Limitations

We presented COGS, a large-scale object-centric 3DGS dataset with 15,675 artifact-filtered, appearance-rich assets across 40 categories, and COGS-Bench for object classification and attribute-usage diagnosis. Our experiments show that the usefulness of Gaussian attributes depends on the architecture and attribute combination, and that adding more attributes does not always improve recognition. HALO demonstrates the benefits of attribute-specific local encoding before fusion and hierarchical aggregation. External validation on MACGS and ScanNet further supports these diagnostic findings.

COGS focuses on controlled object-level diagnosis and does not replace scene-level or severity-stratified robustness benchmarks. Future work can extend COGS-Bench to controlled corruption tests, compositional scenes, dense annotations, and broader 3DGS construction pipelines.

## Acknowledgements

This work was supported in part by the Key Deployment Program of the Chinese Academy of Sciences, China under Grant KGFZD-145-25-39, the National Natural Science Foundation of China under Grants 62272438, and Beijing Natural Science Foundation L25700.

## References

[1] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021.

[2] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3d gaussians. In Proceedings of the AAAI conference on artificial intelligence, pages 1971–1979, 2025.

[3] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017.

[4] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

[5] Binjie Chen, Yunzhou Xia, Yu Zang, Cheng Wang, and Jonathan Li. Decoupled local aggregation for point cloud learning. arXiv preprint arXiv:2308.16532, 2023.

[6] Seokhun Choi, Hyeonseop Song, Jaechul Kim, Taehyeong Kim, and Hoseok Do. Click-gaussian: Interactive segmentation to any 3d gaussians. In European Conference on Computer Vision, pages 289–305. Springer, 2024.

[7] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

[8] Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, and Qing Li. Semantic gaussians: Open-vocabulary scene understanding with 3d gaussian splatting. IEEE Transactions on Circuits and Systems for Video Technology, 2026.

[9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[10] Ji Hou, Benjamin Graham, Matthias Nießner, and Saining Xie. Exploring data-efficient 3d scene understanding with contrastive scene contexts. In 2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 15582–15592. IEEE, 2021.

[11] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021.

[12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 42:1 – 14, 2023.

[13] Black Forest Labs, Stephen Batifol, A. Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Muller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in context image generation and editing in latent space. ArXiv, abs/2506.15742, 2025.

[14] Guohao Li, Matthias Muller, Ali Thabet, and Bernard Ghanem. Deepgcns: Can gcns go as deep as cnns? In Proceedings of the IEEE/CVF international conference on computer vision, pages 9267–9276, 2019.

[15] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin Ren, Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo Gevers, et al. Scenesplat: Gaussian splatting-based scene understanding with vision-language pretraining. In Proceed ings of the IEEE/CVF International Conference on Computer Vision, pages 4961–4972, 2025.

[16] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.

[17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[18] Qi Ma, Yue Li, Bin Ren, Nicu Sebe, Ender Konukoglu, Theo Gevers, Luc Van Gool, and Danda Pani Paudel. Shapesplat: A large-scale dataset of gaussian splats and their self supervised pretraining. arXiv preprint arXiv:2408.10906, 2024.

[19] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[20] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017.

[21] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017.

[22] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051–20060, 2024.

[23] Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R Zaiane, and Martin Jagersand. U2-net: Going deeper with nested u-structure for salient object detec tion. Pattern recognition, 106:107404, 2020.

[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[25] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021.

[26] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019.

[27] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner. Rio: 3d object instance relocalization in changing indoor environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7658–7667, 2019.

[28] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, pages 2555–2563, 2023.

[29] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 38(5):1–12, 2019.

[30] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4840–4851, 2024.

[31] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, et al. Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. Advances in Neural Information Processing Systems, 37:19114–19138, 2024.

[32] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21469–21480, 2025.

[33] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In European conference on computer vision, pages 162–179. Springer, 2024.

[34] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023.

[35] Li Yi, Vladimir G Kim, Duygu Ceylan, I-Chao Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, and Leonidas Guibas. A scalable active framework for

region annotation in 3d shape collections. ACM Transactions on Graphics (ToG), 35(6):1–12, 2016.

[36] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9150–9161, 2023.

[37] Ruiqi Zhang, Hao Zhu, Jingyi Zhao, Qi Zhang, Xun Cao, and Zhan Ma. Mitigating ambiguities in 3d classification with gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 27275–27284, 2025.

[38] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16259–16268, 2021.

[39] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676–21685, 2024.

## Supplementary Material

## S1. Additional Dataset Details and Cross-Dataset Transfer

COGS Category Distribution. The detailed category names and per-category sample counts of COGS are illustrated in Fig. S1.

Full COGS-to-MACGS Transfer Results. We provide the full COGS-to-MACGS transfer results in Tab. S1. Compared with training from random initialization, COGS pretraining consistently improves all evaluated architectures across different target-data fractions. This result indicates that COGS provides transferable object-level 3DGS cues.

## S2. Full Attribute Diagnosis on Capture-Derived MACGS

To further examine whether the attribute-usage findings observed on COGS are specific to the TRELLIS generation pipeline, we additionally repeat the complete attribute diagnosis on MACGS. MACGS is constructed from realworld multi-view captures and exhibits substantially different Gaussian attribute statistics from COGS. We use the same eight input configurations as in the main COGS-Bench analysis: P, PO, PC, POC, PSQ, POSQ, PSQC, and POSQC. All models follow the same training and evaluation protocol used in the main paper. Tab. S2 reports the complete results. The same attribute-usage behavior observed on COGS remains evident on MACGS. For every evaluated conventional backbone, at least one attribute subset outperforms the full POSQC input. For example, PointNet improves from 79.76% OA with POSQC to 81.51% with POC, DGCNN improves from 87.15% to 90.20%, and PTv1 improves from 88.26% to 90.93% with PSQC. Similar subset-over-full behavior is also observed for DeLA, PTv3, and Gaussian-MAE. In contrast, HALO-B and HALO-L achieve their best performance with the full POSQC input, reaching 93.99/93.24% and 94.55/93.81% OA/mAcc, respectively. These results provide an external diagnostic check beyond final classification accuracy: the observation that heterogeneous Gaussian attributes are not uniformly beneficial under conventional homogeneous processing, while HALO can more consistently exploit the complete attribute set, persists on a capture-derived 3DGS distribution.

Table S1. Full cross-dataset transfer results from COGS to MACGS. Models are fine-tuned on varying fractions of MACGS using either random initialization or COGS pre-training. Gains over Scratch are shown in parentheses.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Fraction</td><td>Scratch</td><td>Pre-trained on COGS</td></tr><tr><td>OA / mAcc (% ↑)</td><td>OA / mAcc (% ↑)</td></tr><tr><td>PointNet [20]</td><td rowspan="3">10%</td><td>29.67 / 29.12</td><td>56.65 (+26.98) / 54.51 (+25.39)</td></tr><tr><td>DGCNN [29] PTv1 [38]</td><td>38.63 / 37.28</td><td>60.54 (+21.91) / 58.42 (+21.14)</td></tr><tr><td>HALO-B (Ours, Ref.)</td><td>50.28 / 49.73 66.64 / 64.95</td><td>62.29 (+12.01) / 60.99 (+11.26)  $6 9 . 8 7 \ { _ { ( + 3 . 2 3 ) } } / \ 6 8 . 5 3 \ { _ { ( + 3 . 5 8 ) } }$ </td></tr><tr><td rowspan="3">PointNet [20] DGCNN [29] PTv1 [38]</td><td rowspan="3">20%</td><td>45.75 / 45.01</td><td>64.33 (+18.58) / 62.84 (+17.83)</td></tr><tr><td>59.15 / 57.50</td><td>71.35 (+12.20) / 69.28 (+11.78)</td></tr><tr><td>64.88 / 62.99</td><td>73.29 (+8.41) / 71.88 (+8.89)</td></tr><tr><td>HALO-B (Ours, Ref.)</td><td rowspan="3">50%</td><td>80.68 / 79.89</td><td>82.72 (+2.04) / 80.99 (+1.10)</td></tr><tr><td>PointNet [20]</td><td>69.22 / 68.75</td><td>76.25 (+7.03) / 74.99 (+6.24)</td></tr><tr><td>DGCNN [29] PTv1 [38]</td><td>75.79 / 74.38</td><td>81.79 (+6.00) / 80.09 (+5.71)</td></tr><tr><td rowspan="3">HALO-B (Ours, Ref.)</td><td rowspan="3"></td><td>80.13 / 78.89</td><td>84.94 (+4.81) / 83.97 (+5.08)</td></tr><tr><td>89.19 / 88.43</td><td>90.29 (+1.10) / 89.54 (+1.11)</td></tr><tr><td>79.76 / 77.97</td><td>81.89 (+2.13) / 80.80 (+2.83)</td></tr><tr><td>PointNet [20] DGCNN [29]</td><td rowspan="3">100%</td><td>87.15 / 85.80</td><td></td></tr><tr><td>PTv1 [38]</td><td></td><td>88.82 (+1.67) / 87.70 (+1.90)</td></tr><tr><td></td><td>88.26 / 87.54</td><td>90.11 (+1.85) / 88.43 (+0.89)</td></tr><tr><td>HALO-B (Ours, Ref.)</td><td rowspan="3"></td><td>93.99 / 93.24</td><td>94.08 (+0.09) / 93.51 (+0.27)</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

![](images/18cbd0fecbce6d4c0bb71396c2fed0579d9675048e2be3742cd4db207dfcfda0.jpg)  
Figure S1. Category distribution of the COGS dataset. The figure reports the 40 category names and the corresponding number of curated 3DGS assets in each category.

Table S2. Complete attribute-usage diagnosis on MACGS. P, O, S, Q, and C denote position, opacity, scale, rotation, and appearance/color, respectively. Each entry reports OA/mAcc (%). Bold numbers indicate the best OA for each method.
<table><tr><td>Method</td><td>P</td><td>PO</td><td>PC</td><td>POC</td><td>PSQ</td><td>POSQ</td><td>PSQC</td><td>POSQC</td></tr><tr><td>PointNet [20]</td><td>76.71/75.56</td><td>78.10/77.59</td><td>78.65/77.79</td><td>81.51/80.83</td><td>77.36/76.24</td><td>76.06/75.65</td><td>78.16/77.61</td><td>79.76/77.97</td></tr><tr><td>DGCNN [29]</td><td>88.91/88.24</td><td>87.89/86.71</td><td>88.17/87.28</td><td>90.20/89.06</td><td>87.62/86.31</td><td>86.04/84.87</td><td>87.71/87.08</td><td>87.15/85.80</td></tr><tr><td>PTv1 [38]</td><td>88.91/87.79</td><td>87.80/86.68</td><td>87.99/87.33</td><td>88.82/88.46</td><td>89.00/88.11</td><td>89.37/88.85</td><td>90.93/89.53</td><td>88.26/87.54</td></tr><tr><td>DeLA [5]</td><td>91.50/90.43</td><td>91.50/90.47</td><td>92.51/91.97</td><td>92.14/90.90</td><td>92.05/90.88</td><td>91.87/90.43</td><td>92.24/91.28</td><td>91.50/90.15</td></tr><tr><td>PTv3 [30]</td><td>89.93/89.31</td><td>90.76/89.72</td><td>92.51/91.74</td><td>92.61/91.82</td><td>92.51/91.72</td><td>92.79/92.17</td><td>92.61/92.03</td><td>92.33/91.45</td></tr><tr><td>Gaussian-MAE [18]</td><td>87.62/86.59</td><td>89.37/88.76</td><td>90.30/89.08</td><td>91.50/90.61</td><td>90.11/89.61</td><td>89.74/88.71</td><td>91.40/90.45</td><td>91.22/90.49</td></tr><tr><td>HALO-B (Ours, Ref.)</td><td>91.59/90.58</td><td>92.24/91.40</td><td>93.25/92.39</td><td>93.62/92.91</td><td>93.25/92.43</td><td>93.35/92.65</td><td>93.53/92.83</td><td>93.99/93.24</td></tr><tr><td>HALO-L (Ours, Ref.)</td><td>91.77/90.85</td><td>93.35/92.73</td><td>93.53/93.01</td><td>94.09/93.25</td><td>93.81/93.10</td><td>93.99/93.33</td><td>94.36/93.44</td><td>94.55/93.81</td></tr></table>

## S3. Real-World Scene-Level Validation on ScanNet

COGS-Bench is designed primarily for controlled objectlevel attribute diagnosis. However, scene-level perception introduces additional factors, including background context, occlusion, clutter, and varying scene geometry. We therefore conduct a proof-of-concept experiment to test whether the two main attribute-usage observations also transfer to real-world scene-level 3DGS perception.

Dataset and task. We use the ScanNet 3DGS data provided by SceneSplat [15] for training and evaluation. The SceneSplat release also provides semantic labels aligned with its Gaussian primitives, which we use directly as supervision. We consider the standard 20-class ScanNet semantic segmentation task. To keep the experiment computationally tractable while following an established protocol, training uses the official ScanNet Data-Efficient 10% Limited-Reconstruction (LR) split [10]. No scenes outside the official 10% training subset are used for model training. Evaluation is performed on the complete validation split.

Scene preprocessing and sampling. We use only the native Gaussian attributes (P/O/S/Q/C). Unlike object classification, scene-level 3DGS samples contain a variable and potentially very large number of Gaussian primitives. We partition each training scene into 6 m × 6 m spatial chunks with a stride of 3 m. Within each chunk, we first apply grid sampling with a grid size of 0.02 m. We then randomly select a center and perform spherical sampling to obtain 16,384 Gaussian primitives, enabling mini-batch training while retaining substantially denser scene geometry than in object classification. Each chunk is sampled 16 times per epoch, such that repeated random spherical sampling provides broad spatial coverage of the training chunks. All scene-level models are trained for 100 epochs with batch size 8. The learning rate follows cosine annealing from

$1 \times 1 0 ^ { - 3 } \mathrm { ~ t o ~ } 1 \times 1 0 ^ { - 5 }$ During validation, the complete scene is grid-sampled at 0.02 m and all sampled primitives are processed by the network. Predictions are then propagated to all evaluation points using nearest-neighbor assignment, allowing metrics to be computed over the complete validation scene rather than only the grid-sampled inputs.

Dense prediction adaptation of HALO. HALO was originally introduced as an object-classification reference model and progressively downsamples Gaussian primitives before global pooling. For semantic segmentation, we retain its hierarchical encoder but replace the classificationonly output with a lightweight dense prediction path. At each hierarchy level, an MLP projects the per-primitive features. These features are propagated by nearest-neighbor feature propagation to the first hierarchy, allowing every primitive to collect representations from multiple semantic levels. Multi-level features are then aggregated to predict a semantic label using a simple classifier for each input primitive. We use [4, 4, 8, 4] HA blocks with [64, 128, 256, 512] feature channels at the four hierarchy levels to accommodate the substantially denser scene input. The shared-processing and attribute-specific variants use the same scene preprocessing, hierarchy, decoder, optimization, and POSQC input; they differ only in how heterogeneous Gaussian attributes are processed before fusion.

Results. Tab. S3 evaluates two complementary questions. First, we test whether the early-fusion degradation identified by COGS-Bench persists in real-world scenelevel perception by repeating the complete eight-input diagnostic protocol with PTv1. Full POSQC obtains 48.45% mIoU and 80.95% OA. Importantly, three attribute subsets outperform POSQC on both metrics: PO reaches 49.67% mIoU / 81.14% OA, PSQC reaches 49.34% / 81.23%, and POSQ achieves the best mIoU of 50.37% / 81.08%. In particular, POSQ improves over full POSQC by 1.92 mIoU points. However, this is not a general preference for using fewer attributes: PSQ, POC, PC, and P all perform below POSQC in mIoU. The full ablation therefore shows that the contribution of heterogeneous Gaussian attributes under early fusion is composition-dependent rather than monotonically increasing with the number of available attributes. This subset-over-full behavior is consistent with the diagnostic observations on COGS and capture-derived MACGS, and demonstrates that the phenomenon also persists in real-world scene-level 3DGS perception. Second, we test whether attribute-specific processing can mitigate this degradation. Using the identical full POSQC input, HALO with shared attribute processing obtains 43.81% mIoU and 79.16% OA, whereas attribute-specific processing reaches 49.83% mIoU and 81.96% OA, corresponding to gains of 6.02 mIoU and 2.80 OA points. The substantial improvement indicates that preserving attribute-specific evidence before fusion remains beneficial under real-world scene clutter, background context, and occlusion. Together, the PTv1 ablation and the controlled HALO comparison provide complementary evidence that (i) homogeneous early fusion does not consistently exploit all available Gaussian attributes, and (ii) role-preserving attribute processing can alleviate this issue beyond clean object-level classifica tion.

Table S3. Real-world scene-level proof-of-concept on the standard ScanNet 20-class semantic segmentation task. Training uses the official ScanNet Data-Efficient 10% split and evaluation uses the complete validation set.
<table><tr><td>Method</td><td>Variant/Input</td><td>mIoU (% ↑) OA (% ↑)</td></tr><tr><td>PTv1 [38] P</td><td></td><td>45.64 78.84</td></tr><tr><td>PTv1 [38] PO</td><td></td><td>49.67 81.14</td></tr><tr><td>PTv1 [38] PC</td><td></td><td>47.81 80.50</td></tr><tr><td>PTv1 [38] POC</td><td></td><td>46.70 80.49</td></tr><tr><td>PTv1 [38] PSQ</td><td></td><td>48.05 79.71</td></tr><tr><td>PTv1 [38] POSQ</td><td></td><td>50.37 81.08</td></tr><tr><td>PTv1 [38]</td><td>PSQC</td><td>49.34 81.23</td></tr><tr><td>PTv1 [38]</td><td>POSQC</td><td>48.45 80.95</td></tr><tr><td>HALO</td><td>Shared POSQC</td><td>43.81 79.16</td></tr><tr><td>HALO</td><td>Attribute-specific POSQC</td><td>49.83 81.96</td></tr></table>

## S4. Inference Efficiency

The main-paper classification benchmark reports parameter count, peak GPU memory during training, and total training-and-evaluation time. To provide a more complete characterization of the accuracy–efficiency trade-off, we further measure arithmetic complexity, actual GPU inference latency, inference throughput, and peak inference memory under the same unified classification setting.

Measurement protocol. All models are evaluated using the full POSQC input with 1,024 Gaussian primitives per object, consistent with the main classification benchmark. Measurements are performed in FP32 on the same NVIDIA RTX 4090 GPU with models in evaluation mode and gradient computation disabled. Inputs are preloaded on the GPU; therefore, disk I/O, data loading, host-to-device transfer, and the common dataset-level preprocessing are excluded. In contrast, all model-specific operations executed inside the forward pass, including neighborhood construction, grouping, and hierarchical downsampling, are included in the measured latency and memory.

Arithmetic FLOPs are measured for a single object $( B =$ 1) using fvcore. Following the fvcore convention, one fused multiply-add is counted as one FLOP. Since the default counter does not support several point-cloud-specific operators, including some KNN queries, farthest-point sampling, top-k operations, indexing, and custom CUDA kernels, we report these values as counted $F L O P s$ rather than a complete hardware-cost estimate. Unsupported operations are excluded from the FLOP totals but remain fully included in the measured inference latency and memory.

For latency, each model is first executed for 100 warm-up iterations. We then measure 1,000 forward passes using CUDA events and report the mean GPU execution time per complete batch. Measurements are conducted for both batch size 1 (B1) and batch size 32 (B32). B32 throughput is derived from the measured batch latency as $\frac { 3 2 } { l a t e n c y }$ and is reported in objects per second. Peak inference memory is measured with torch.cuda.max memory allocated() during a forward pass after resetting the CUDA peak-memory statistics. The reported value therefore includes the residen model and input together with the output and temporary tensors allocated during inference.

Tab. S4 summarizes the results. PointNet provides the lowest latency and highest B32 throughput, but at substantially lower recognition accuracy. HALO-B reaches 94.50% OA with only 2.68M parameters and 0.18G counted FLOPs. Its B1 inference latency is 5.70ms, while its B32 batch latency is only 5.81ms, corresponding to a throughput of 5,504 objects/s. This is the second-highest B32 throughput among the evaluated methods while retaining substantially higher accuracy than the lightweight baselines. HALO-B also requires only 30.58MiB peak allocated memory at B1 and 404.39MiB at B32.

The results further illustrate why arithmetic FLOPs alone are insufficient for characterizing point-cloud inference efficiency. For example, PTv1 has the lowest counted FLOPs (0.10G) but requires 10.48ms for B1 inference. Its default fvcore trace does not account for several operations, including farthest-point sampling, KNN queries, and associated neighborhood operations, all of which contribute to the measured execution time. We therefore use the measured latency and peak memory together with counted FLOPs when comparing computational efficiency.

HALO-L increases the model capacity from 2.68M to 10.46M parameters and the counted FLOPs from 0.18G to 1.51G, while improving OA by only 0.20 percentage points (94.50% to 94.70%). Its B32 throughput also decreases from 5,504 to 3,932 objects/s. These results reinforce our positioning of HALO-B as the recommended accuracy–efficiency configuration, while HALO-L primarily serves as a capacity-scaling variant.

Table S4. Inference efficiency under the unified 1,024-Gaussian classification protocol. Counted FLOPs are obtained using fvcore with one fused multiply-add counted as one FLOP; unsupported point-cloud-specific operators are excluded from FLOPs but included in latency and memory measurements. B1/B32 latency denotes the mean CUDA-event execution time of the complete batch after 100 warm-up and 1,000 measured forward passes. B32 throughput is computed from the measured B32 batch latency. Peak inference memory reports the maximum allocated GPU memory during one forward pass.
<table><tr><td>Method</td><td>OA (% ↑)</td><td>Params (M ↓)</td><td>Counted FLOPs (G ↓)</td><td>B1 / B32 Lat. (ms/batch ↓)</td><td>B32 Throughput (obj/s ↑)</td><td>B1 / B32 Peak Mem. (MiB ↓)</td></tr><tr><td>PointNet [20]</td><td>79.53</td><td>3.47</td><td>0.45</td><td>1.24 / 3.81</td><td>8394</td><td>34.24 / 303.31</td></tr><tr><td>DGCNN [29]</td><td>85.69</td><td>1.81</td><td>2.73</td><td>1.38 / 26.48</td><td>1208</td><td>76.11 / 1968.81</td></tr><tr><td>PTv1 [38]</td><td>87.67</td><td>3.26</td><td>0.10</td><td>10.48 / 18.35</td><td>1744</td><td>27.48 / 237.56</td></tr><tr><td>DeLA [5]</td><td>88.48</td><td>5.34</td><td>0.73</td><td>3.49 / 7.92</td><td>4039</td><td>46.33 / 565.85</td></tr><tr><td>PTv3 [30]</td><td>89.11</td><td>38.87</td><td>6.32</td><td>30.95 / 71.62</td><td>447</td><td>271.25 / 424.38</td></tr><tr><td>Gaussian-MAE [18]</td><td>88.20</td><td>22.10</td><td>2.44</td><td>4.01 / 8.84</td><td>3619</td><td>104.68 / 484.68</td></tr><tr><td>HALO-B (Ours, Ref.)</td><td>94.50</td><td>2.68</td><td>0.18</td><td>5.70/5.81</td><td>5504</td><td>30.58 / 404.39</td></tr><tr><td>HALO-L (Ours, Ref.)</td><td>94.70</td><td>10.46</td><td>1.51</td><td>5.75 / 8.14</td><td>3932</td><td>66.49 / 502.91</td></tr></table>

Table S5. Ablation study of hierarchical aggregation on COGS-Bench. “–” indicates that the component is not applicable. ◦ denotes that downsampling is omitted.
<table><tr><td>Cross-level Fusion</td><td>Neighborhood  $\mathbf { A g g } .$ </td><td>OA (% ↑)</td><td>mAcc (% ↑)</td><td>Mem. (MB ↓)</td></tr><tr><td>×</td><td>一</td><td>84.26</td><td>85.15</td><td>482</td></tr><tr><td>√</td><td>X</td><td>85.07</td><td>85.68</td><td>603</td></tr><tr><td>√</td><td>0</td><td>88.97</td><td>90.03</td><td>1885</td></tr><tr><td>√</td><td>√</td><td>94.50</td><td>94.70</td><td>750</td></tr></table>

## S5. Additional HALO Ablations

Although the main contribution of this work is COGS and COGS-Bench, we include component ablations of HALO to validate the attribute-aware reference model used in the benchmark. Tab. S5 evaluates the role of hierarchical aggregation. Removing hierarchical aggregation and directly pooling multi-level features leads to 84.26% OA. Adding cross-level fusion without neighborhood aggregation brings only a small improvement, reaching 85.07% OA. Omitting downsampling retains more primitives and improves accuracy to 88.97%, but increases memory consumption to 1885 MB. The complete HA module achieves 94.50% OA with only 750 MB memory. These results show that aggregating neighborhood evidence before primitive decimation is important for both accuracy and efficiency.

## S6. Full Cross-Pipeline Classification Results

For completeness, Tab. S6 reports the full-POSQC classification comparison on capture-derived MACGS and CADderived ModelNet40GS. All models are trained and evaluated within each target dataset without COGS pre-training, following the unified classification protocol in the main paper.

Table S6. Full-POSQC cross-pipeline classification results on MACGS and ModelNet40GS.
<table><tr><td rowspan="2">Method</td><td colspan="2">MACGS</td><td colspan="2">ModelNet40GS</td></tr><tr><td>OA (% ↑)</td><td>mAcc (% ↑)</td><td>OA (% ↑)</td><td>mAcc (% ↑)</td></tr><tr><td>PointNet [20]</td><td>79.76</td><td>77.97</td><td>90.98</td><td>87.79</td></tr><tr><td>DGCNN [29]</td><td>87.15</td><td>85.80</td><td>90.82</td><td>87.50</td></tr><tr><td>PTv1 [38]</td><td>88.26</td><td>87.54</td><td>91.06</td><td>88.17</td></tr><tr><td>DeLA [5]</td><td>91.50</td><td>90.15</td><td>91.99</td><td>89.60</td></tr><tr><td>PTv3 [30]</td><td>92.33</td><td>91.45</td><td>91.97</td><td>89.14</td></tr><tr><td>Gaussian-MAE [18]</td><td>91.22</td><td>90.49</td><td>92.88</td><td>90.52</td></tr><tr><td>HALO-B (Ours, Ref.)</td><td>93.99</td><td>93.24</td><td>93.24</td><td>91.14</td></tr><tr><td>HALO-L (Ours, Ref.)</td><td>94.55</td><td>93.81</td><td>93.53</td><td>91.33</td></tr></table>

## S7. Additional COGS Visualizations

We provide additional multi-view visualizations of COGS samples. Each sample is rendered from multiple viewpoints to show object-level structure, appearance variation, and view consistency. These visualizations complement the dataset examples and quality analysis in the main paper.

# E W T CI 5 G G C C D 0 0 7 7 1 1

Figure S2. Additional multi-view visualizations of COGS samples.

# 二 二QQOg

![](images/cad969bc4f5b2693741464ddbed1a487ad11aa7339459234c5b9cac1e97bc014.jpg)

Figure S3. Additional multi-view visualizations of COGS samples.

#

Figure S4. Additional multi-view visualizations of COGS samples.

# 山 一と 7T T1入 ↑IIODDO

Figure S5. Additional multi-view visualizations of COGS samples.

# IOOD DI1A 元食

Figure S6. Additional multi-view visualizations of COGS samples.

# 月两两月两两月两 TOESUBTO 凸口可口口了口口 II 年二9日一一一

Figure S7. Additional multi-view visualizations of COGS samples.