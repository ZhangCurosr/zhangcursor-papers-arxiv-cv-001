# GDBA-Net: A Gated Dual-Branch Adaptation of SAM2 for Robust Cross-Domain Nuclei Segmentation

Biplab Poudel<sup>1,2,3</sup>, Betserai Albert Magarire<sup>3</sup>, Anish Kumar Shah<sup>4</sup>, Jiacheng Xie<sup>1,2,5</sup>, Dong Xu<sup>1,2,5,∗</sup>

<sup>1</sup>Department of Electrical Engineering and Computer Science, University of Missouri, Columbia, MO 65211, USA

<sup>2</sup>Bond Life Sciences Center, University of Missouri, Columbia, MO 65211, USA

<sup>3</sup>Department of Applied Computing, Lander University, Greenwood, SC 29649, USA

<sup>4</sup>Department of Electronics and Computer Engineering, Sagarmatha Engineering College, Tribhuvan University, Nepal <sup>5</sup>Health Informatics Institute, Morsani College of Medicine, University of South Florida, Tampa, FL 33612, USA <sup>∗</sup>Corresponding author. Email: dxu1@usf.edu

Abstract— Accurate nuclei segmentation underpins quantitative computational pathology, yet remains challenging due to morphological variability, weak boundaries, nuclear crowding, and pronounced domain shifts arising from differences in tissue preparation, staining, and acquisition. While deep learning and recent foundation models have improved segmentation performance under matched distributions with the original training data, their generalization across heterogeneous histopathology domains remains limited. We introduce GDBA-Net, a parameter-efficient nuclei segmentation framework built around the proposed Gated Dual-Branch Adapter (GDBA). GDBA-Net adapts a frozen SAM2 encoder by integrating pretrained semantic representations with explicit frequency-derived structural priors. The complementary representations are independently transformed through Grid MLP branches and dynamically integrated using sample-adaptive channel-wise gating, selectively emphasizing structural cues relevant to nuclear delineation. Despite using only 4.3 million trainable parameters, GDBA-Net achieves state-of-theart in-dataset performance and consistently superior cross-dataset generalization across six diverse nuclei segmentation benchmarks under fully supervised, crossdataset, and five-shot adaptation protocols, outperforming competing CNN-, transformer-, and foundation modelbased methods. Comprehensive ablations confirm that these gains arise from the coordinated design of semanticfrequency complementarity and adaptive gating rather than increased model capacity. By uniting transferable semantic representations with explicit structural priors, GDBA-Net provides a robust, annotation-efficient solution to domain variability in nuclei segmentation. https://github.com/biplabpoudel25/GDBA\_Net

## I. INTRODUCTION

Index Terms— Nuclei segmentation, Computational pathology, Foundation models, SAM2, Parameter-efficient fine-tuning, Domain generalization

CCURATE segmentation of cell nuclei from histopathological images constitutes a fundamental analytical step in computational pathology, serving as the gateway to downstream quantitative tasks, including cell counting, cell profiling, spatial transcriptomics mapping, and diagnostic decision support [1]–[3]. The proliferation of digital pathology and whole-slide imaging (WSI) in modern clinical workflows has intensified the demand for automated, robust nuclei segmentation pipelines that alleviate the substantial annotation burden on pathologists and enable high-throughput, reproducible morphological analysis [4]. However, reliable nuclei delineation remains challenging because nuclei exhibit substantial variations in size, shape, appearance, and spatial organization, while weak boundaries, densely clustered cells, staining variability, and imaging artifacts further complicate segmentation [5]. More importantly, differences in tissue preparation, staining protocols, scanners, and acquisition conditions introduce pronounced domain shifts, causing models trained on one dataset to deteriorate when applied to unseen histopathological data.

Traditional image processing methods, including watershed segmentation, adaptive thresholding, and handcrafted morphological descriptors, provided initial promise in controlled, homogeneous tissue settings [6] but were highly sensitive to staining variability, noise, and nuclear crowding, often requiring extensive parameter tuning and failing in complex tissue environments [7]. Deep learning substantially advanced the field, beginning with U-Net [8] and subsequently incorporating attention, multi-scale representations, instance-aware modeling, and transformer-based architectures [5], [9]–[13]. Although these methods have achieved strong performance under matched training and testing distributions, robust generalization across heterogeneous datasets remains difficult. Their learned representations are largely driven by the sourcedomain appearance, making segmentation particularly vulnerable when staining characteristics, tissue morphology, or acquisition conditions differ from those encountered during

training.

Vision foundation models provide a promising alternative by transferring representations learned from large and diverse image corpora. Segment Anything Model (SAM) [14] and its successor SAM2 [15] have demonstrated strong generalpurpose segmentation capabilities, yet their direct transfer to fine-grained biomedical structures remains limited by the substantial gap between natural and histopathological images [16], [17]. Medical fine-tuning approaches like MedSAM [18], and MedSAM2 [19], and parameter-efficient adaptation methods like SAMed [20], SAM-Adapter [21], SAM2-Adapter [22], and the Medical SAM Adapter [23], have improved their applicability while reducing the need to train task-specific models from scratch. Nevertheless, efficient adaptation alone does not fully address the representational requirements of nuclei segmentation.

High-level semantic features provide valuable contextual information, whereas accurate separation of densely packed nuclei also depends on fine structural cues associated with boundaries and local morphological transitions. Frequencydomain representations can emphasize such information [24], but high-frequency components may simultaneously amplify staining irregularities, texture variations, and acquisition noise. Thus, a central challenge is not simply introducing structural information, but determining how structural and semantic representations should be selectively balanced during adaptation. Existing adapter methods have explored task-specific information, including high-frequency components [22], but the adaptive integration of separately transformed structural and semantic representations remains insufficiently explored to achieve robust nuclei segmentation across heterogeneous domains.

In this work, we present GDBA-Net, a SAM2-based framework for robust multi-domain nuclei segmentation. The framework is motivated by the complementary strengths of pretrained deep representations for semantic abstraction and frequency-derived representations for capturing fine structural information. Its central component is the proposed Gated Dual-Branch Adapter (GDBA), which uses branch-specific Grid MLP adapters to independently process semantic and frequency-derived structural features before integrating them through sample-adaptive channel-wise gating. This design selectively emphasizes structural cues relevant to nuclear delineation while preserving the pretrained knowledge of the frozen encoder. We evaluate GDBA-Net across six morphologically diverse benchmarks - Kumar [25], MoNuSeg [26], CryoNuSeg [27], CPM [28], TNBC [29], and CoNSeP [5], under both indataset and rigorous cross-dataset generalization protocols.

The principal contributions of this paper are as follows:

1) We introduce GDBA-Net, a parameter-efficient dualbranch adaptation framework for SAM2 that jointly leverages pretrained semantic representations and frequency-derived structural information for robust nuclei segmentation.

2) We develop a sample-adaptive channel-wise gating mechanism that dynamically balances semantic and structural information during adaptation. This design enables the model to selectively exploit fine structural cues while preserving transferable semantic knowledge, rather than relying on fixed or direct feature fusion.

3) We demonstrate state-of-the-art performance and strong cross-domain generalization across six heterogeneous nuclei segmentation benchmarks under in-dataset, crossdataset, and few-shot settings outperforming eleven competing CNN-, transformer-, and foundation modelbased methods.

4) We establish the effectiveness and efficiency of GDBA-Net through few-shot adaptation and systematic ablation studies. Additional experiments on retinal vessel and skin lesion segmentation further support the applicability of the proposed adaptation strategy beyond nuclei segmentation.

## II. RELATED WORK

## A. Deep Learning for Nuclei Segmentation

Deep convolutional neural networks, anchored by the U-Net architecture [8], have become the dominant paradigm for biomedical image segmentation. In the nuclei segmentation literature, architectural innovations such as attention mechanisms [30], dense skip connections [31], and multi-scale feature pyramids [32] progressively improved performance on standard benchmarks. Task-specific frameworks addressing the challenge of overlapping nuclei further advanced the field: StarDist [11] predicts object probabilities together with radial distances that define star-convex polygonal representation, while HoVer-Net [5] decomposes the segmentation task into horizontal and vertical gradient prediction branches to enable instance separation in crowded tissues. DAN-NucNet [33] incorporates spatial and channel attention to enhance nuclear representations under heterogeneous histological conditions. Transformer-based architectures, including [12], Swin-UNet [13], HiFormer [34], and TransNuSeg [35], subsequently improved the modeling of long-range dependencies and globallocal context. Although these methods substantially improve representation learning and instance separation, these architectural advances do not inherently ensure robustness to domain shifts across tissue types, staining characteristics, preparation protocols, and acquisition conditions.

## B. Domain Shift and Generalization in Computational Pathology

Domain shift remains a major challenge in computational pathology, where variations in staining, tissue preparation, scanners, and acquisition protocols can substantially alter image appearance and degrade performance on unseen data [36], [37]. Cross-domain evaluations have further demonstrated that strong performance within a single dataset does not necessarily translate across acquisition conditions or institutions [38]. Consequently, domain-generalization methods have explored appearance perturbation, representation learning, and frequency-based augmentation to reduce dependence on source-specific characteristics [39], [40].

## C. Parameter-Efficient Fine-Tuning of Foundational Models

The emergence of segmentation foundation models has created new opportunities for transferring large-scale visual representations to medical imaging. SAM [14] introduced promptable general-purpose segmentation, while SAM2 [15] extended this paradigm using a hierarchical multi-scale encoder. However, large-scale pretraining on natural images does not eliminate the domain gap to specialized medical imagery, while full-model fine-tuning can impose substantial computational and memory requirements, particularly when taskspecific annotations are limited. Parameter-efficient fine-tuning (PEFT) strategies, originally developed for natural language processing and later extended to vision models, have emerged as a principled solution. In medical imaging, approaches such as SAMed [20] applied low-rank adaptation (LoRA) to the SAM image encoder, while SAM-Adapter [21], SAM2- Adapter [22], and the Medical SAM Adapter [23] introduce lightweight mechanisms for incorporating task- or domainspecific information while retaining much of the pretrained representation. MedSAM [18] and MedSAM2 [19] instead rely on full or partial fine-tuning on curated medical datasets.

These approaches demonstrate that foundational models can be effectively specialized without developing task-specific architectures from scratch. However, they differ substantially in the number of trainable parameters, the form of task-specific information introduced, and the extent to which the pretrained backbone is modified. Moreover, most were developed for broad downstream or medical segmentation tasks rather than specifically addressing fine-grained nuclear structure and cross-dataset robustness in heterogeneous histopathology.

## D. Frequency-Domain Feature Integration

Frequency-domain representations have been increasingly investigated to improve robustness to distribution shift. Fourier Domain Adaptation and related approaches have demonstrated that manipulating frequency components can improve domain adaptation and generalization, including in medical image segmentation. Frequency-aware architectures such as FRUNet [41] have further shown the value of high-frequency information for capturing fine-grained structural details in biomedical segmentation. However, these approaches primarily employ frequency information for distribution manipulation, augmentation, or feature attention.

## III. METHODOLOGY

## A. Proposed Architecture Overview

The proposed GDBA-Net framework, as shown in Figure 1, adapts the hierarchical image encoder of SAM2 for nuclei segmentation using the Gated Dual-Branch Adapter (GDBA). Given an input histopathology image $I ~ \in ~ \hat { \mathbb { R } ^ { B \times C _ { i n } \times H \times \dot { W } } }$ where B is the batch size, $C _ { i n } = 3 $ , and $H = W = 1 0 2 4 .$ the image is first converted into visual tokens using the SAM2 patch embeddings and positional embedding:

$$
X _ { 1 } = P a t c h E m b e d ( I ) + E _ { p o s } .\tag{1}
$$

The tokens are then progressively processed through the hierarchical encoder, where query pooling at stage transitions reduces the spatial resolution and increases the channel dimension, producing a hierarchical representation as:

$$
X _ { s } \in \mathbb { R } ^ { B \times H _ { s } \times W _ { s } \times C _ { s } } , \qquad s \in \{ 1 , 2 , 3 , 4 \}\tag{2}
$$

with $C _ { s } \in \{ 1 4 4 , 2 8 8 , 5 7 6 , 1 1 5 2 \}$

To adapt these pretrained representations for nuclei segmentation, GDBA integrates semantic information from the SAM2 encoder with frequency-derived structural information from the input image. The resulting adaptation features are injected into the corresponding SAM2 tokens through residual addition before the associated transformer blocks, enabling task-specific adaptation throughout the hierarchical encoder.

## B. Semantic and Frequency-Derived Feature Construction

The two complementary inputs to GDBA are constructed at multiple resolutions corresponding to the hierarchical feature levels of the SAM2 encoder, as described below.

1) Semantic Representation: For each feature level $s ,$ the hierarchical SAM2 representation $X _ { s }$ serves as the source of semantic integration. To obtain a parameter-efficient representation compatible with GDBA, X is projected from the original encoder dimension $C _ { s }$ into a lower-dimensional feature space:

$$
S e m _ { s } = X _ { s } W _ { s } ^ { e } + b _ { s } ^ { e } , \qquad S e m _ { s } \in \mathbb { R } ^ { B \times H _ { s } \times W _ { s } \times C _ { s } ^ { \prime } }\tag{3}
$$

where $W _ { s } ^ { e }$ and $b _ { s } ^ { e }$ are learnable projection parameters. The resulting representation $S e m _ { s }$ constitutes the semantic representation supplied to the semantic branch of GDBA.

2) Frequency-Domain Handcrafted Feature Generation: To explicitly capture robust structural cues often blurred by purely spatial learned features, we employ frequency-domain filtering via 2D Fast Fourier Transform:

$$
\mathcal { F } ( I ) = \mathrm { F F T S h i f t } ( \mathrm { F F T 2 } ( I ) ) , \qquad \mathcal { F } _ { h } = \mathcal { F } ( I ) \odot ( 1 - M )\tag{4}
$$

A binary mask M suppresses the central 25% of frequency coefficients to isolate high-frequency components. The filtered representation is transformed back to the spatial domain via inverse FFT:

$$
I _ { h } = \left| \mathrm { R e } \left[ \mathrm { I F F T 2 } \left( \mathrm { I F F T S h i f t } ( \mathcal { F } _ { h } ) \right) \right] \right|\tag{5}
$$

Thus, low-frequency appearance information is suppressed while rapid spatial variations associated with boundaries, contours, and fine structural patterns are retained. Handcrafted features for each stage are then sequentially embedded:

$$
H a n d _ { s } = P a t c h E m b e d _ { s } ( H a n d _ { s - 1 } ) \in \mathbb { R } ^ { B \times H _ { s } \times W _ { s } \times C _ { s } ^ { \prime } }\tag{6}
$$

where $C _ { s } ^ { \prime } = C _ { s } / \alpha$ , with $\alpha = 3 2$ acts as a bottleneck scaling factor.

## C. Gated Dual Branch Adapter (GDBA)

The GDBA module constitutes the core adaptation mechanism, inserted at each transformer block indexed by $( s , d )$ where s denotes the stage and d denotes the depth within the stage.

![](images/845e0d77a8382e587b85dd07e1f747ac935edd267f7f65533783cc8e17fa546e.jpg)  
Fig. 1. Overall architecture of the proposed Gated Dual-Branch Adapter Network (GDBA-Net) framework for nuclei segmentation. The input H&E image is processed through multiple encoder stages, while the extracted frequency-derived structural and semantic features are integrated by the GDBA modules at the corresponding stages. The resulting representations are subsequently used to generate the predicted nuclei mask. The frequency and semantic feature extraction modules are detailed in Figure 2.

![](images/62afcfac7c5be540ebc01920497a9aba4054757bf21e1c0e1ceb9c3a3264a8d2.jpg)  
Fig. 2. Frequency and semantic feature extraction process used in GDBA framework. The upper panel illustrates the frequency branch, whereas the lower panel shows the semantic feature extraction process. These complementary features are subsequently integrated by the GDBA modules.

1) Branch-specific Grid MLP Adapter (Branch Adapter): At each transformer depth $d ,$ GDBA processes the semantic and frequency-derived representations through two independent Grid MLP Adapters. For an input $Z \in \overline { { \mathbb { R } } } ^ { B \times H _ { s } \times W _ { s } \times C _ { s } ^ { \prime } }$ , the Grid MLP Adapter is defined as:

$$
\begin{array} { r } { A ( Z ) = Z + \mathcal { D } _ { 2 } \left[ W _ { 2 } \mathcal { D } _ { 1 } \left( \mathrm { G E L U } \left( W _ { 1 } \mathrm { L N } ( Z ) + b _ { 1 } \right) \right) + b _ { 2 } \right] } \end{array}\tag{7}
$$

Here, $W _ { 1 }$ expands the channel dimension from $C _ { s } ^ { \prime }$ to $4 C _ { s } ^ { \prime } ,$ $W _ { 2 }$ projects it back to $C _ { s } ^ { \prime } .$ , and $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ denote dropout operations. A residual connection preserves the original branch representation.

The MLP operates independently at every spatial grid location; hence, it performs nonlinear channel transformation without directly mixing neighboring spatial positions. Separate Grid MLP parameters are learned for the semantic and structural branches at each transformer depth:

$$
Z _ { s , d } ^ { \mathrm { s e m } } = \mathcal { A } _ { s , d } ^ { \mathrm { s e m } } ( S e m _ { s } ) , \qquad Z _ { s , d } ^ { \mathrm { f r e q } } = \mathcal { A } _ { s , d } ^ { \mathrm { f r e q } } ( H a n d _ { s } )\tag{8}
$$

The two adapted representations are concatenated along the channel dimension and reduced to the original adapter dimension using a depth-specific fusion layer:

$$
F _ { s , d } = \mathrm { G E L U } \left( W _ { s , d } ^ { f } \left[ Z _ { s , d } ^ { \mathrm { s e m } } \Vert Z _ { s , d } ^ { \mathrm { f r e q } } \right] + b _ { s , d } ^ { f } \right)\tag{9}
$$

where $F _ { s , d } ~ \in ~ \mathbb { R } ^ { B \times H _ { s } \times W _ { s } \times C _ { s } ^ { \prime } }$ and ∥ denotes channel-wise concatenation.

The fusion operation enables the model to learn optimal weighting between handcrafted structural cues and learned semantic representations, allowing it to dynamically emphasize whichever feature type is more informative for a given spatial location or image context.

2) Adaptive Channel-Wise Gating: Because frequencyderived information may contain useful nuclear boundaries as well as undesirable texture or acquisition artifacts, the fused representation is further regulated using an input-adaptive channel gate. Global average pooling is first performed over the spatial dimensions:

$$
\bar { F } _ { s , d } = \frac { 1 } { H _ { s } W _ { s } } \sum _ { h = 1 } ^ { H _ { s } } \sum _ { w = 1 } ^ { W _ { s } } F _ { s , d } ( h , w )\tag{10}
$$

A depth-specific linear transformation followed by a sigmoid activation produces the channel gate:

$$
g _ { s , d } = \sigma \left( W _ { s , d } ^ { g } \bar { F } _ { s , d } + b _ { s , d } ^ { g } \right) , \qquad g _ { s , d } \in [ 0 , 1 ] ^ { C _ { s } ^ { \prime } }\tag{11}
$$

The gate is then broadcast across spatial locations and applied to the fused representation:

$$
\hat { F } _ { s , d } ( h , w , c ) = F _ { s , d } ( h , w , c ) g _ { s , d } ( c )\tag{12}
$$

Therefore, the implemented gate is sample-dependent and channel-wise, rather than spatially varying. Moreover, the gate operates on the fused representation rather than independently assigning weights to the semantic and frequency branches; cross-branch interaction is learned by the preceding fusion projection.

3) Prompt Projection and Integration with SAM2: The gated representation remains in the compressed adapter space $C _ { s } ^ { \prime } .$ A stage-specific linear projection maps it back to the native SAM2 embedding dimension:

$$
\begin{array} { r } { P _ { s , d } = W _ { s } ^ { p } \hat { F } _ { s , d } + b _ { s } ^ { p } , \qquad P _ { s , d } \in \mathbb { R } ^ { B \times H _ { s } \times W _ { s } \times C _ { s } } } \end{array}\tag{13}
$$

Unlike the branch adapters, fusion layers, and gating layers, which are depth-specific, the final projection $\boldsymbol { W } _ { s } ^ { p }$ is shared among the GDBA modules operating at the same feature resolution.

The generated feature is then injected through residual addition into the current SAM2 tokens immediately before the corresponding Hiera transformer block:

$$
\tilde { X } _ { s , d } = X _ { s , d } + P _ { s , d }\tag{14}
$$

followed by:

$$
X _ { s , d + 1 } = T _ { s , d } ( \tilde { X } _ { s , d } )\tag{15}
$$

where $T _ { s , d }$ represents the corresponding pretrained Hiera block. Consequently, GDBA does not replace the SAM2 representation; instead, it learns a task-specific residual modification that introduces complementary semantic and structural information into the existing feature stream.

This process repeats for each depth d within stage s, with the GDBA module progressively refining the feature representations as they propagate through the encoder hierarchy. The final segmentation mask is generated by the SAM2 Mask Decoder.

## D. Training Objective

Nuclei typically occupy a substantially smaller region than the histopathological background, resulting in considerable foreground-background imbalance. The model is therefore optimized using the Tversky loss. Given true positives TP, false positives FP, and false negatives FN, the Tversky index is defined as:

$$
T I = \frac { T P } { T P + \alpha _ { T } F P + \beta _ { T } F N } , \qquad \mathcal { L } _ { \mathrm { T v e r s k y } } = 1 - T I\tag{16}
$$

where $\alpha _ { T }$ and $\beta _ { T }$ control the relative penalties assigned to false-positive and false-negative predictions. The optimization objective is $\mathcal { L } _ { \mathrm { T v e r s k y } } = 1 - T I$

During adaptation, the pretrained SAM2 image-encoder parameters are kept fixed, while the GDBA and decoder-related trainable parameters are optimized. Training is performed using the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ for 200 epochs.

## IV. EXPERIMENTAL SETUP

## A. Datasets

We evaluate the proposed framework on six publicly available H&E-stained nuclei segmentation benchmarks: Kumar [25], MoNuSeg [26], CryoNuSeg [27], Combined CPM [28], TNBC [29], and CoNSeP [5]. These datasets encompass diverse tissue types, image resolutions, and preparation protocols, providing a comprehensive setting for evaluating both in-dataset performance and cross-dataset generalization. Table I summarizes the characteristics of each dataset along with the training, validation, and testing partitions used in our experiments.

![](images/e99b12a41d69b5c160b0642ee81f28ee90439a044b17b244629a1392aeb1aa1e.jpg)  
Fig. 3. Overview of the proposed Gated Dual-Branch Adapter (GDBA). The upper light-gray shaded region illustrates the GDBA module, which integrates independently adapted handcrafted and semantic features to generate prompt features. The lower-left inset presents the branch adapter architecture, which is applied independently to the handcrafted and semantic feature streams. The generated prompts are combined with the current SAM2 tokens and passed to the corresponding transformer block

TABLE I  
SUMMARY OF BENCHMARK DATASETS USED FOR NUCLEI SEGMENTATION EVALUATION.
<table><tr><td>Dataset</td><td>Tissue / Con- Images dition</td><td></td><td>Resolution</td><td>Nuclei</td><td>Split (Train/Val/Test)</td></tr><tr><td>Kumar [25]</td><td>Multi-organ (7 organs)</td><td>30</td><td>1000×1000</td><td>21,623</td><td>12/4/14</td></tr><tr><td>MoNuSeg [26]</td><td>Multi-organ (9 organs)</td><td>51</td><td>1000×1000</td><td>~28,500</td><td>33/7/11</td></tr><tr><td>CryoNuSeg [27]</td><td>Multi-organ (10 organs)</td><td>30</td><td>512×512</td><td>~7,596</td><td>15/5/10</td></tr><tr><td>Combined CPM [28] Brain, Head &amp;</td><td>Neck, Lung</td><td>79</td><td>Variable</td><td>~10,000</td><td>30/10/39</td></tr><tr><td>TNBC [29]</td><td>Breast (TNBC subtype)</td><td>50</td><td>512×512</td><td>4,022</td><td>32/8/10</td></tr><tr><td>CoNSeP [5]</td><td>Colorectal adenocarci- noma</td><td>41</td><td>1000×1000</td><td>24,319</td><td>22/5/14</td></tr></table>

The validation sets were separated from the available training data and used for model selection, while the test sets were reserved exclusively for final evaluation. For the Combined CPM dataset, CPM-15 and CPM-17 were merged by combining their respective training, validation, and test partitions. The same fixed partitions were used across all competing methods to ensure consistent comparison.

## B. Experimental Protocols

We evaluate GDBA-Net under two complementary settings: in-dataset evaluation and cross-dataset generalization. In the in-dataset setting, models are trained and evaluated using the training and testing partitions of the same dataset. In the cross-dataset setting, models trained on a source dataset are directly evaluated on the unseen target dataset without targetdomain fine-tuning, providing a measure of robustness under domain shift. To further examine the effect of limited targetdomain supervision, we compare direct cross-dataset transfer with 5-shot target-domain adaptation and full target-domain training. In the 5-shot setting, five target-domain images are used for training, a small subset is retained for validation, and the remaining images are used for testing. Additionally, the applicability of GDBA-Net beyond nuclei segmentation is evaluated on distinct biomedical segmentation tasks.

## C. Evaluation Metrics

Performance is quantified using Dice Similarity Coefficient (DSC), Jaccard Index (IoU), Precision, and Specificity. Using the pixel-level TP, FP, and FN quantities defined above, and denoting the number of true-negative background pixels by TN, the metrics are computed as:

$$
{ \begin{array} { l l } { { \mathrm { D S C } } = { \frac { 2 T P } { 2 T P + F P + F N } } , \ } & { { \mathrm { I o U } } = { \frac { T P } { T P + F P + F N } } , } \\ { { \mathrm { P r e c . } } = { \frac { T P } { T P + F P } } , \ } & { { \mathrm { S p e c . } } = { \frac { T N } { T N + F P } } . } \end{array} }\tag{17}
$$

Dice and Jaccard quantify the spatial overlap between the predicted and ground-truth nuclei masks. Precision reflects false-positive foreground predictions, whereas Specificity measures the model’s ability to correctly identify background pixels. Together, these metrics provide complementary measures of segmentation overlap and pixel-level classification performance.

## V. RESULTS

We evaluate GDBA-Net across heterogeneous nuclei segmentation benchmarks to examine its segmentation accuracy, robustness to domain shift, and ability to adapt under limited target-domain supervision. Comparisons with conventional CNN-, transformer-, nuclei-specific, and foundation modelbased approaches are used to assess performance across varying tissue types, preparation protocols, and imaging characteristics.

## A. In-dataset evaluation

Table II presents the in-dataset performance on MoNuSeg, Kumar, and CryoNuSeg, where models are trained and tested on the same dataset. GDBA-Net achieves the highest Dice and Jaccard scores on all three datasets, demonstrating consistently accurate segmentation across diverse tissue types, image characteristics, and preparation protocols. The comparison further shows that architectural complexity or foundation-model pretraining alone does not guarantee strong segmentation performance. While nnU-Net, HoVer-Net, and DanNucNet provide competitive results, transformer-based models and directly adapted SAM variants show less consistent performance across datasets. In contrast, GDBA-Net maintains strong performance under both FFPE and cryo-sectioned imaging conditions, supporting the effectiveness of combining SAM2 semantic representations with frequency-derived structural information. Notably, the 5-shot GDBA-Net achieves a 0.8359 Dice on CryoNuSeg, surpassing fully trained foundational models like MedSAM. Overall, these results establish the proposed method as a robust and annotation-efficient approach for in-dataset nuclei segmentation.

## B. Cross-dataset generalization

Table III presents cross-dataset generalization by training on a single source dataset and directly testing it on unseen target datasets without target-domain fine-tuning. Across the eight source-target combinations, GDBA-Net achieves the highest Dice and Jaccard scores in seven settings and ranks second in the remaining CoNSeP-to-CPM evaluation. This consistent performance across multiple source-target combinations indicates robustness to variations in tissue composition, staining appearance, image resolution, and preparation protocol.

When trained on MoNuSeg, GDBA-Net achieves the best overlap performance on all three unseen datasets, with particularly clear improvements on TNBC (Dice: 0.8146), where methods such as nnU-Net and U-Net obtained substantially lower performance. Although DanNucNet remains competitive on CPM, its Dice decreases to 0.5338 on TNBC, demonstrating that strong transfer to one target dataset does not necessarily ensure consistent cross-domain generalization. A similar pattern is observed when CryoNuSeg is used as the source domain, with GDBA-Net achieving the highest Dice and Jaccard scores across the evaluated target datasets. Several competing models retain high Precision or Specificity under this shift but exhibit considerably lower overlap scores, suggesting conservative predictions that fail to recover substantial portions of the nuclear regions. In contrast, GDBA-Net maintains a better balance between nuclear detection and background discrimination.

A more challenging pattern emerges when CoNSeP is used as the source domain. Although HoVer-Net achieves the highest overlap on CPM, GDBA-Net produces the highest precision and specificity, indicating fewer false-positive predictions. More importantly, GDBA-Net regains the leading overlap performance on the CoNSeP to TNBC transfer, whereas HoVer-Net shows a substantial decrease despite its strong performance on CPM. This contrast highlights that success on a single dataset doesn’t necessarily indicate robust cross-domain generalization.

The per-image distributions in Figure 4 provide further insight into this behavior. Under in-dataset evaluation on Cry oNuSeg, most methods exhibit relatively compact performance distributions. When transferred from CryoNuSeg to CPM, the distributions become substantially wider for several competing methods, indicating increased sensitivity to individual target images under domain shift. GDBA-Net maintains a comparatively high central tendency for Dice and Jaccard while avoiding the pronounced degradation observed for several baselines. The Precision and Specificity distributions further show that high background classification performance can coexist with substantially reduced segmentation overlap under crossdomain transfer. Overall, these results suggest that integrating pretrained semantic representations with frequency-derived structural information improves resistance to domain-specific appearance variations and enables more reliable segmentation of previously unseen histopathology data.

## C. Generalization vs target-domain adaptation

Figure 5 compares the performance of GDBA-Net under direct cross-dataset transfer, 5-shot target-domain adaptation, and full target-domain training on CPM, TNBC, and CoNSeP. On CPM and TNBC, cross-dataset transfer remains close to few-shot and fully supervised performance, indicating that GDBA-Net transfers effectively when the source and target domains have comparable nuclear characteristics. In contrast, a larger performance gap is observed on CoNSeP, where direct transfer from MoNuSeg or CryoNuSeg results in lower performance. Introducing only five labeled target-domain images substantially narrows this gap towards the fully supervised performance. These results demonstrate that GDBA-Net can maintain strong generalization when domain shift is moderate, while limited target-domain supervision provides substantial benefits under more challenging domain shifts.

TABLE II  
IN-DATASET NUCLEI SEGMENTATION PERFORMANCE COMPARISON OF GDBA-NET AND COMPETING METHODS ON THE MONUSEG, CRYONUSEG, AND KUMAR DATASETS. BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINE, RESPECTIVELY.
<table><tr><td rowspan="2">Models</td><td colspan="4">MoNuSeg</td><td colspan="4">Kumar</td><td colspan="4">CryoNuSeg</td></tr><tr><td>Dice</td><td></td><td>Jaccard Precision</td><td>Specificity</td><td>Dice</td><td>Jaccard</td><td>Precision</td><td>Specificity</td><td>Dice</td><td>Jaccard Precision</td><td></td><td>Specificity</td></tr><tr><td>U-Net [8]</td><td>0.8021</td><td>0.6712</td><td>0.7929</td><td>0.9411</td><td>0.7278</td><td>0.5775</td><td>0.6332</td><td>0.8149</td><td>0.7737</td><td>0.6349</td><td>0.7400</td><td>0.8531</td></tr><tr><td>Attn. U-Net [30]</td><td>0.7903</td><td>0.6551</td><td>0.7615</td><td>0.9264</td><td>0.7609</td><td>0.6212</td><td>0.7451</td><td>0.8833</td><td>0.7596</td><td>0.6176</td><td>0.7164</td><td>0.8410</td></tr><tr><td>nnU-Net [31]</td><td>0.8218</td><td>0.7091</td><td>0.8015</td><td>0.9396</td><td>0.8199</td><td>0.6966</td><td>0.8177</td><td>0.9424</td><td>0.8108</td><td>0.6845</td><td>0.8144</td><td>0.9186</td></tr><tr><td>PSPNet [32]</td><td>0.5922</td><td>0.4311</td><td>0.4347</td><td>0.6061</td><td>0.6014</td><td>0.4348</td><td>0.4402</td><td>0.6274</td><td>0.6805</td><td>0.5204</td><td>0.5361</td><td>0.6042</td></tr><tr><td>HoVer-Net [5]</td><td>0.8208</td><td>0.7086</td><td>0.7996</td><td>0.9399</td><td>0.7981</td><td>0.6749</td><td>0.7741</td><td>0.9223</td><td>0.8284</td><td>0.7125</td><td>0.8452</td><td>0.9231</td></tr><tr><td>NucleiSegNet [42]</td><td>0.7681</td><td>0.6251</td><td>0.7195</td><td>0.9095</td><td>0.7701</td><td>0.6283</td><td>0.7541</td><td>0.9221</td><td>0.8023</td><td>0.6756</td><td>0.8478</td><td>0.9282</td></tr><tr><td>DanNucNet [33]</td><td>0.8288</td><td>0.7101</td><td>0.8244</td><td>0.9452</td><td>0.8199</td><td>0.6953</td><td>0.8169</td><td>0.9327</td><td>0.8033</td><td>0.6755</td><td>0.7904</td><td>0.9027</td></tr><tr><td>TransNuSeg [35]</td><td>0.7682</td><td>0.6257</td><td>0.7295</td><td>0.9145</td><td>0.7556</td><td>0.5975</td><td>0.7050</td><td>0.8690</td><td>0.7745</td><td>0.6355</td><td>0.7305</td><td>0.8563</td></tr><tr><td>HiFormer [34]</td><td>0.6703</td><td>0.5060</td><td>0.7308</td><td>0.9322</td><td>0.6340</td><td>0.4669</td><td>0.6410</td><td>0.8984</td><td>0.7371</td><td>0.5843</td><td>0.8056</td><td>0.9226</td></tr><tr><td>SAM [14]</td><td>0.7842</td><td>0.6470</td><td>0.7991</td><td>0.9462</td><td>0.7091</td><td>0.5627</td><td>0.6838</td><td>0.8990</td><td>0.7416</td><td>0.5949</td><td>0.7571</td><td>0.9003</td></tr><tr><td>MedSAM [18]</td><td>0.7811</td><td>0.6420</td><td>0.7552</td><td>0.9250</td><td>0.7617</td><td>0.6170</td><td>0.7746</td><td>0.7620</td><td>0.8092</td><td>0.6814</td><td>0.8063</td><td>0.9104</td></tr><tr><td>GDBA-Net5-shot</td><td>0.8262</td><td>0.7050</td><td>0.8220</td><td>0.9414</td><td>0.8180</td><td>0.6942</td><td>0.8390</td><td>0.9494</td><td>0.8359</td><td>0.7205</td><td>0.8725</td><td>0.9436</td></tr><tr><td>GDBA-Net</td><td>0.8408</td><td>0.7259</td><td>0.8388</td><td>0.9516</td><td>0.8201</td><td>0.6972</td><td>0.8104</td><td>0.9365</td><td>0.8481</td><td>0.7381</td><td>0.8519</td><td>0.9286</td></tr></table>

TABLE III

CROSS-DATASET GENERALIZATION PERFORMANCE OF GDBA-NET AND COMPETING METHODS TRAINED ON MONUSEG, CRYONUSEG, OR CONSEP AND DIRECTLY EVALUATED ON THE UNSEEN COMBINED CPM, TNBC, AND COMBINED CONSEP TEST DATASETS. FOR EACH SOURCE–TARGET DATASET PAIR AND METRIC, THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINE, RESPECTIVELY.
<table><tr><td rowspan="3"></td><td rowspan="3">Methods</td><td colspan="10">Test dataset</td></tr><tr><td colspan="4">Combined CPM</td><td colspan="3">TNBC</td><td colspan="4">Combined CoNSeP</td></tr><tr><td>Dice</td><td colspan="3">Jaccard Precision Specificity</td><td colspan="3">Dice Jaccard Precision</td><td colspan="2">Specificity Dice</td><td colspan="3">Jaccard Precision Specificity</td></tr><tr><td rowspan="6">Se 5e</td><td>U-Net</td><td>0.7329</td><td>0.5878</td><td>0.8281</td><td>0.9640</td><td>0.6732 0.5187</td><td>0.7145</td><td>0.9642</td><td>0.6236</td><td>0.4678</td><td>0.5698</td><td>0.9000</td></tr><tr><td>Attn U-Net</td><td>0.7262</td><td>0.5799</td><td>0.8170 0.9589</td><td>0.7027</td><td>0.5529</td><td>0.7438</td><td>0.9629</td><td>0.6003</td><td>0.4477</td><td>0.5147</td><td>0.8568</td></tr><tr><td>nnU-Net</td><td>0.7801</td><td>0.6604</td><td>0.8004</td><td>0.9440 0.7025</td><td>0.5712</td><td>0.6132</td><td>0.9065</td><td>0.6322</td><td>0.4854</td><td>0.5444</td><td>0.8407</td></tr><tr><td>PSPNet</td><td>0.6062</td><td>0.4565</td><td>0.4674</td><td>0.6863</td><td>0.2606 0.1634</td><td>0.1639</td><td>0.1928</td><td>0.4142</td><td>0.2844</td><td>0.2900</td><td>0.4596</td></tr><tr><td>HiFormer</td><td>0.7386</td><td>0.5886</td><td>0.7955</td><td>0.9600</td><td>0.6435 0.4870</td><td>0.6245</td><td>0.9562</td><td>0.5244</td><td>0.3646</td><td>0.5363</td><td>0.9203</td></tr><tr><td>HoVer-Net</td><td>0.8115</td><td>0.6949</td><td>0.8663</td><td>0.9659 0.6508</td><td>0.5370</td><td>0.7810</td><td>0.9777</td><td>0.6191</td><td>0.4966</td><td>0.5971</td><td>0.9072</td></tr><tr><td></td><td>NucleiSegNet 0.7686</td><td>0.6474</td><td>0.8088</td><td>0.9485</td><td>0.6773 0.5338</td><td>0.5474 0.4039</td><td>0.5965</td><td>0.8955</td><td>0.6089</td><td>0.4600</td><td>0.5326</td><td>0.8400</td></tr><tr><td rowspan="4">rned</td><td>DanNucNet</td><td>0.8399</td><td>0.7282</td><td>0.9165</td><td>0.9835</td><td></td><td>0.8824</td><td>0.9831</td><td>0.6101</td><td>0.4709</td><td>0.7424</td><td>0.8894</td></tr><tr><td>TransNuSeg</td><td>0.7466</td><td>0.6016</td><td>0.7967</td><td>0.9570</td><td>0.6578 0.5025</td><td>0.6241</td><td>0.9486</td><td>0.6175</td><td>0.4639</td><td>0.5572</td><td>0.8954</td></tr><tr><td>SAM</td><td>0.7094</td><td>0.5615</td><td>0.7248</td><td>0.9350</td><td>0.6409 0.5012</td><td>0.6190</td><td>0.9514</td><td>0.4995</td><td>0.3548</td><td>0.4926</td><td>0.8937</td></tr><tr><td>MedSAM</td><td>0.7649</td><td>0.6242</td><td>0.8260</td><td>0.9662</td><td>0.6261 0.4840</td><td>0.5803</td><td>0.9394</td><td>0.6124</td><td>0.4673</td><td>0.6065</td><td>0.9240</td></tr><tr><td>GDBA U-Net</td><td>0.8558</td><td>0.7504</td><td>0.8922</td><td>0.9741</td><td>0.8146</td><td>0.6936</td><td>0.8127</td><td>0.9715</td><td>0.6833</td><td>0.5351</td><td>0.7253</td><td>0.9565</td></tr><tr><td rowspan="4">CrSeg</td><td></td><td>0.6536</td><td>0.5177</td><td>0.8863</td><td>0.9752</td><td>0.2179</td><td>0.1311 0.8927</td><td>0.9975</td><td>0.1966</td><td>0.1184</td><td>0.7858</td><td>0.9898</td></tr><tr><td>Attn U-Net</td><td>0.6593</td><td>0.5131</td><td>0.8275</td><td>0.9607</td><td>0.3279</td><td>0.2093 0.8130</td><td>0.9938</td><td>0.3319</td><td>0.2124</td><td>0.7623</td><td>0.9841</td></tr><tr><td>nnU-Net</td><td>0.7819</td><td>0.6516</td><td>0.7476</td><td>0.9330</td><td>0.5602 0.4221</td><td>0.4599</td><td>0.8704</td><td>0.5603</td><td>0.4143</td><td>0.4916</td><td>0.8563</td></tr><tr><td>PSPNet</td><td>0.5678</td><td>0.4122</td><td>0.4630</td><td>0.7653</td><td>0.2926</td><td>0.1818 0.3527</td><td>0.9370</td><td>0.2727</td><td>0.1650</td><td>0.3194</td><td>0.8681</td></tr><tr><td></td><td>HiFormer</td><td>0.6147 0.4626</td><td>0.9001</td><td>0.9864</td><td>0.4292</td><td>0.2905</td><td>0.8616</td><td>0.9934</td><td>0.1520</td><td>0.0861</td><td>0.6602</td><td>0.9871</td></tr><tr><td></td><td>HoVer-Net</td><td>0.7227</td><td>0.5968</td><td>0.9004</td><td>0.9792</td><td>0.3344 0.2209</td><td>0.9155</td><td>0.9952</td><td>0.5561</td><td>0.3977</td><td>0.7160</td><td>0.9628</td></tr><tr><td></td><td>NucleiSegNet</td><td>0.6326</td><td>0.5133</td><td>0.9311</td><td>0.9898</td><td>0.5089</td><td>0.3749 0.8929</td><td>0.9917</td><td>0.6331</td><td>0.4823</td><td>0.6839</td><td>0.9154</td></tr><tr><td></td><td>DanNucNet</td><td>0.6256</td><td>0.4946</td><td>0.9345</td><td>0.9883</td><td>0.0928</td><td>0.0516 0.9375</td><td>0.9994</td><td>0.1980</td><td>0.1183</td><td>0.8152</td><td>0.9915</td></tr><tr><td>rron</td><td>TransNuSeg</td><td>0.7114</td><td>0.5684</td><td>0.7804</td><td>0.7114</td><td>0.4344</td><td>0.2968 0.8742</td><td>0.4344</td><td>0.5354</td><td>0.3759</td><td>0.666</td><td>0.5354</td></tr><tr><td rowspan="4"></td><td>SAM</td><td>0.7283</td><td>0.5830</td><td>0.7397</td><td>0.7283</td><td>0.6520</td><td>0.5065 0.6199</td><td>0.6520</td><td>0.4921</td><td>0.3438</td><td>0.4600</td><td>0.4921</td></tr><tr><td>MedSAM</td><td>0.7932</td><td>0.6611</td><td>0.8150</td><td>0.9584</td><td>0.6370</td><td>0.4863</td><td>0.5945</td><td>0.9439</td><td>0.5026 0.3582</td><td>0.4801</td><td>0.8632</td></tr><tr><td>GDBA</td><td>0.8266</td><td>0.7126</td><td>0.8744</td><td>0.9706</td><td>0.8045</td><td>0.6829 0.8271</td><td>0.9723</td><td>0.6557</td><td>0.5051</td><td>0.6649</td><td>0.9368</td></tr><tr><td>U-Net</td><td>0.3961</td><td>0.2722</td><td>0.8602</td><td>0.9800</td><td>0.5443</td><td>0.4008</td><td>0.7971</td><td>0.9926</td><td></td><td></td><td></td></tr><tr><td rowspan="8">Train on oNsep</td><td></td><td>0.2888</td><td>0.1977</td><td>0.7817</td><td>0.9704</td><td>0.6611 0.5100</td><td>0.8225</td><td>0.9705</td><td></td><td></td><td></td><td></td></tr><tr><td>Attn U-Net nnU-Net</td><td>0.7912</td><td></td><td></td><td></td><td></td><td>0.6512</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.6594</td><td>0.8832</td><td>0.9802</td><td>0.7823</td><td>0.7929</td><td>0.9689</td><td></td><td></td><td></td><td></td></tr><tr><td>PSPNet</td><td>0.4179 0.4441</td><td>0.2883</td><td>0.3428</td><td>0.7109</td><td>0.4542</td><td>0.3123</td><td>0.3181</td><td>0.7307</td><td></td><td></td><td></td></tr><tr><td>HiFormer HoVer-Net</td><td>0.8187</td><td>0.3067 0.6979</td><td>0.9078 0.9293</td><td>0.9840 0.9845</td><td>0.6274 0.6055</td><td>0.4758 0.4575</td><td>0.7736 0.9458</td><td>0.9744 0.9916</td><td></td><td></td><td></td></tr><tr><td>NucleiSegNet</td><td>0.6509</td><td>0.5153</td><td>0.9228</td><td>0.9781 0.7207</td></table>

## D. Qualitative segmentation analysis

Figure 6 presents a qualitative comparison of segmentation results on representative samples. U-Net, Attention U-Net, PSPNet, and SAM exhibit false-positive regions, oversegmentation, or missed nuclei, particularly in densely populated regions. Although HoVer-Net and MedSAM yield stronger masks, some nuclei remain incomplete or missed. In comparison, GDBA-Net more closely follows the ground-truth masks, with improved boundary delineation and fewer falsepositive and false-negative regions.

CryoNuSeg (Train & Test)  
![](images/48391bc5810b3e85f8390a1e6c3f597d8b137fab1939a44f05599b8cac9d3d31.jpg)

CryoNuSeg Train → CPM Test  
![](images/2c493764a089b4324f370630dfd18ab4eda4eb4ae736a613823d18a621b5ce4f.jpg)  
Fig. 4. Boxplot-based comparison of nuclei segmentation performance under in-dataset and cross-dataset evaluation. The top panel presents results when models are trained and tested on CryoNuSeg, whereas the bottom panel shows cross-dataset performance when models are trained on CryoNuSeg and evaluated on CPM dataset. Boxes represent the interquartile range, horizontal lines indicate the median, whiskers show the non-outlier range, and circular markers denote outlier samples.

![](images/788cc259999f8404eba2aaf56c8efb2086a123af2bad316ae32459ac9a862539.jpg)  
Fig. 5. Comparison of dice scores for GDBA across cross-dataset transfer, five-shot adaptation, and full target-domain fine-tuning on the CPM, TNBC, and CoNSeP datasets. Cross-dataset models are trained on MoNuSeg or CryoNuSeg and evaluated directly on each unseen target dataset.

## E. Parameter Efficiency

Despite leveraging the powerful SAM2 backbone, GDBA-Net restricts optimization to the lightweight GDBA modules while keeping the pretrained encoder frozen. As shown in Figure 7, the proposed framework requires only 4.3 million trainable parameters, fewer than several competing models while maintaining strong segmentation performance. This favorable balance between trainable model capacity and segmentation accuracy highlights the parameter-efficient nature of the proposed adaptation strategy.

## F. Applicability to Additional Biomedical Segmentation Tasks

To examine the applicability of GDBA-Net beyond nuclei segmentation, we further evaluate the framework on two distinct biomedical segmentation tasks: retinal vessel segmentation using DRIVE [43] and skin lesion segmentation using ISIC 2018 dataset [44], [45]. DRIVE contains retinal fundus images with pixel-level vessel annotations, whereas ISIC 2018 contains dermoscopic images with corresponding skin lesion masks. These datasets provide additional evaluation of GDBA-Net across structures and imaging characteristics that differ substantially from histopathological nuclei.

As shown in Table IV, GDBA-Net has better performance across both datasets and evaluation metrics. The strong performance across two structurally distinct segmentation tasks indicates that the benefits of integrating pretrained semantic representations with frequency-derived structural information are not limited to nuclei segmentation. This further supports GDBA-Net’s adaptability across diverse biomedical segmentation tasks.

## VI. ABLATION STUDIES

## A. Contribution ofGDBA components

To systematically assess the contribution of the GDBA components, we evaluate multiple architectural configurations on the MoNuSeg and CryoNuSeg datasets (Table V). Removing either the frequency or semantic branch reduces segmentation performance, demonstrating the complementary contribution of structural and semantic representations. Disabling the gating mechanism also degrades segmentation quality, confirming that adaptive weighting of the two feature streams is more effective than direct fusion. Replacing the shared projection with independent stage-specific projections increases the number of trainable parameters while reducing overlap performance. These findings demonstrate that the effectiveness of GDBA-Net stems from the coordinated contributions of its dualbranch representation, adaptive gating, and parameter-efficient shared-projection design.

TABLE IV  
PERFORMANCE COMPARISON OF GDBA-NET AND COMPETING METHODS ON ISIC 2018 AND DRIVE FOR SKIN LESION AND RETINAL VESSEL SEGMENTATION.
<table><tr><td rowspan="2">Models</td><td colspan="2">ISIC 2018 [44], [45]</td><td colspan="2">DRIVE [43]</td></tr><tr><td>Dice</td><td>Jaccard</td><td>Dice</td><td>Jaccard</td></tr><tr><td>U-Net</td><td>0.858</td><td>0.776</td><td>0.816</td><td>0.701</td></tr><tr><td>nnU-Net</td><td>0.855</td><td>0.766</td><td>0.780</td><td>0.640</td></tr><tr><td>Attn U-Net</td><td>0.867</td><td>0.785</td><td>0.808</td><td>0.681</td></tr><tr><td>NucleiSegNet</td><td>0.881</td><td>0.808</td><td>0.650</td><td>0.482</td></tr><tr><td>MedSAM</td><td>0.863</td><td>0.759</td><td>0.806</td><td>0.676</td></tr><tr><td>SAM</td><td>0.903</td><td>0.827</td><td>0.741</td><td>0.588</td></tr><tr><td>GDBA-Net</td><td>0.921</td><td>0.864</td><td>0.819</td><td>0.702</td></tr></table>

## B. Bottleneck Scaling Factor

Table VI evaluates the trade-off between adapter capacity and parameter efficiency. Empirical evaluations indicate that α = 32 strikes the optimal balance between expressiveness and regularized efficiency. Increasing capacity (α = 8) increases the parameter count to 10.23M without improving segmentation, while stronger compression (α = 64) slightly reduces both parameters and overlap performance.

## VII. DISCUSSION

The proposed framework improves nuclei segmentation by combining a frozen foundation model encoder with a lightweight domain adaptation mechanism that integrates pretrained semantic representations with frequency-derived structural information. While the gains under matched training and testing distributions were relatively modest, the advantage became considerably more evident under domain shift, where GDBA-Net achieved the highest Dice and Jaccard scores in seven of the eight evaluated source-target settings. This is particularly important in computational pathology, where variations in tissue type, preparation, staining, and acquisition can substantially alter image appearance and limit the generalizability of models optimized for a single dataset.

The cross-dataset results suggest that the complementary design of GDBA-Net is central to this robustness. SAM2 provides rich pretrained semantic representations, whereas the frequency branch emphasizes structural information associated with nuclear boundaries and local intensity transitions. Their integration allows the model to rely on information that is potentially less sensitive to source-specific appearance. This is especially evident in the strong transfer from CryoNuSeg to

![](images/0e58ba171c5d22f43ec882a49982042f93206f0a9ac18d3a08ff3db1334d85c4.jpg)  
Fig. 6. Qualitative comparison of nuclei segmentation results on representative samples. Ground-truth masks and overlap maps produced by the competing methods are shown for visual comparison. In the overlay, white denotes correctly segmented nuclei pixels, red denotes false positives predictions, and blue indicates false negatives predictions.

TABLE V  
COMPONENT-WISE ABLATION ANALYSIS OF THE PROPOSED GDBA FRAMEWORK ON THE MONUSEG AND CRYONUSEG DATASETS.
<table><tr><td rowspan="2">Frequency branch</td><td rowspan="2">Semantic branch</td><td rowspan="2">Adaptive gate</td><td rowspan="2">Shared projections</td><td rowspan="2">Params (M)</td><td colspan="4">MoNuSeg</td><td colspan="4">CryoNuSeg</td></tr><tr><td>Dice</td><td>Jacc.</td><td>Prec.</td><td>Spec.</td><td>Dice</td><td>Jacc.</td><td>Prec.</td><td>Spec.</td></tr><tr><td>×</td><td>√</td><td>√</td><td>√</td><td>4.11</td><td>0.8145</td><td>0.6893</td><td>0.8028</td><td>0.9332</td><td>0.8331</td><td>0.7156</td><td>0.8618</td><td>0.9369</td></tr><tr><td>√</td><td>×</td><td>√</td><td>√</td><td>4.07</td><td>0.8087</td><td>0.6813</td><td>0.7954</td><td>0.9292</td><td>0.8360</td><td>0.7196</td><td>0.8248</td><td>0.9123</td></tr><tr><td>√</td><td>√</td><td>×</td><td>√</td><td>4.28</td><td>0.8108</td><td>0.6841</td><td>0.7982</td><td>0.9306</td><td>0.8340</td><td>0.7173</td><td>0.8686</td><td>0.9396</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X*</td><td>4.88</td><td>0.8048</td><td>0.6758</td><td>0.7907</td><td>0.9268</td><td>0.8402</td><td>0.7260</td><td>0.8568</td><td>0.9311</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>4.30</td><td>0.8408</td><td>0.7259</td><td>0.8388</td><td>0.9516</td><td>0.8481</td><td>0.7381</td><td>0.8519</td><td>0.9286</td></tr></table>

∗ Independent stage-specific projections are used instead of the shared projection.

TABLE VI  
EFFECT ON BOTTLENECK SCALING FACTOR α ON MODEL PERFORMANCE AND PARAMETER EFFICIENCY. LARGER α CORRESPONDS TO STRONGER DIMENSIONALITY REDUCTION.
<table><tr><td rowspan="2">α</td><td rowspan="2">Params (M)</td><td colspan="4">MoNuSeg</td><td colspan="4">CryoNuSeg</td></tr><tr><td>Dice</td><td>Jacc.</td><td>Prec.</td><td>Spec.</td><td>Dice</td><td>Jacc.</td><td>Prec.</td><td>Spec.</td></tr><tr><td>8</td><td>10.23</td><td>0.8176</td><td>0.6941</td><td>0.8068</td><td>0.9347</td><td>0.8368</td><td>0.7213</td><td>0.8591</td><td>0.9332</td></tr><tr><td>16</td><td>5.54</td><td>0.8189</td><td>0.6958</td><td>0.8087</td><td>0.9359</td><td>0.8352</td><td>0.7189</td><td>0.8618</td><td>0.9250</td></tr><tr><td>32</td><td>4.30</td><td>0.8408</td><td>0.7259</td><td>0.8388</td><td>0.9516</td><td>0.8481</td><td>0.7381</td><td>0.8519</td><td>0.9286</td></tr><tr><td>64</td><td>3.97</td><td>0.8194</td><td>0.6963</td><td>0.8096</td><td>0.9361</td><td>0.8372</td><td>0.7213</td><td>0.8475</td><td>0.9276</td></tr></table>

![](images/b1462fdf3528f95c7dc32247263a1d7ce79e479022a21ebe0c32986f0a6816b9.jpg)  
Fig. 7. Comparison of trainable parameter counts for GDBA and competing segmentation methods.

FFPE datasets despite their distinct preparation characteristics. Importantly, the CoNSeP-to-CPM setting, where HoVer-Net achieved higher overlap, also demonstrates that no representation is universally optimal and highlights the importance of evaluating generalization across multiple source-target combinations rather than isolated datasets.

The adaptation experiments further reveal a practical relationship between domain shift and annotation requirements. For CPM and TNBC, direct cross-dataset transfer of GDBA-Net remained close to target-domain-supervised performance, suggesting that limited or no additional target-domain annotation may be sufficient when cross-domain is already strong. For CoNSeP, where direct cross-domain transfer exhibited a larger performance gap, only five labeled target images recovered a substantial portion of the gap towards fully supervised performance. GDBA-Net therefore provides a practical continuum between direct cross-domain transfer, limited targetdomain adaptation, and full target-domain training, which is valuable in histopathology where pixel-level annotation by experts is expensive and time-consuming.

The ablation studies reinforce that these gains originate from architectural design rather than increased trainable model capacity. Removing either feature branch or the gating mechanism consistently degraded overlap performance, confirming the benefit of complementary representations and their adaptive integration. Moreover, increasing the number of trainable parameters through independent projections or a wider bottleneck did not improve segmentation. With only approximately 4.3M trainable parameters, GDBA-Net demonstrates that foundation models can be effectively specialized for nuclei segmentation without full backbone fine-tuning. This combination of cross-domain robustness, adaptation with limited targetdomain supervision, and parameter efficiency highlights the potential of the proposed design to specialize large pretrained models for fine-grained biomedical segmentation tasks.

Despite these findings, the present study is limited to H&E nuclei segmentation using publicly available datasets. Although the additional experiments on retinal vessel and skin lesion segmentation provide initial evidence of applicability beyond histopathological nuclei, broader validation across imaging modalities, institutions, acquisition systems, and segmentation tasks are still needed to establish the generality of the approach. In addition, the current frequency extraction uses a fixed filtering strategy, and the effect of the frequency cutoff has not been systematically investigated. Learning or adapting the relevant frequency components may further improve performance across varying resolutions and noise characteristics. Future work will therefore explore adaptive frequency modeling, instance-level nuclear segmentation, and extension of the GDBA principle to other computational pathology and medical imaging tasks.

## VIII. CONCLUSION

This study introduced GDBA-Net, a parameter-efficient adaptation framework built around the Gated Dual-Branch Adapter (GDBA), which integrates frequency-domain structural priors with pretrained semantic representations for robust nuclei segmentation. Through adaptive channel-wise gating and shared stage-specific projections, GDBA effectively incorporates complementary structural and semantic information into the hierarchical encoder. Evaluations across six heterogeneous histopathology datasets demonstrated accurate indataset segmentation, robust generalization to unseen domains, and effective adaptation using only five labeled target images. Additional evaluations on retinal vessel and skin lesion segmentation further demonstrated the applicability of the proposed adaptation strategy beyond nuclei segmentation. The ablation results further confirmed that these improvements arise from the coordinated design of the semantic and frequency branches, the gating mechanism, and the shared projection rather than increased trainable model capacity.

More broadly, these findings demonstrate the value of combining task-relevant structural priors with transferable semantic representations. The strong cross-dataset performance of GDBA-Net suggests that their adaptive integration can improve robustness to domain-specific appearance variations while preserving fine structural information. GDBA-Net therefore provides a parameter- and annotation-efficient strategy for addressing two persistent challenges in computational pathology: domain variability and limited labeled data. Beyond nuclei segmentation, the results suggest that the proposed adaptation principle may provide a useful direction for specializing foundation models to fine-grained biomedical segmentation tasks where structural information and cross-domain robustness are important.

## CREDIT AUTHORSHIP CONTRIBUTION STATEMENT

Biplab Poudel (Conceptualization, Methodology, Data preparation, Formal analysis, Validation, Initial manuscript, Visualization), Betserai Albert Magarire (Data preparation, Formal analysis, Validation, Visualization), Anish Kumar Shah (Formal analysis, Validation), Jiacheng Xie (Formal analysis, Validation), Dong Xu (Conceptualization, Resources, Supervision, Funding acquisition). All authors contributed to the revision of the manuscript and have approved the final version.

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## FUNDING

This work was partially supported by the National Institutes of Health (grants R35GM126985 and R01LM015252).

## CODE AVAILABILITY

The source code is available at: https://github.com/ biplabpoudel25/GDBA\_Net.

## REFERENCES

[1] Y. Cheng, P. Qiao, H. He, G. Song, and J. Chen, “Hardboundary attention network for nuclei instance segmentation,” in ACM Multimedia Asia, Gold Coast, Australia, Dec. 2021, pp. 1–7.

[2] A. Mahbod, G. Dorffner, I. Ellinger, R. Woitek, and S. Hatamikia, “Improving generalization capability of deep learning-based nuclei instance segmentation by nondeterministic train time and deterministic test time stain normalization,” Computational and Structural Biotechnology Journal, vol. 23, pp. 669–678, 2024.

[3] N. Coudray et al., “Classification and mutation prediction from non-small cell lung cancer histopathology images using deep learning,” Nature Medicine, vol. 24, no. 10, pp. 1559–1567, Oct. 2018.

[4] A. J. Evans et al., “Validating whole slide imaging systems for diagnostic purposes in pathology,” Archives of Pathology & Laboratory Medicine, vol. 146, no. 4, pp. 440–450, Apr. 2022.

[5] S. Graham et al., “Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images,” Medical Image Analysis, vol. 58, p. 101563, Dec. 2019.

[6] H. Irshad, A. Veillard, L. Roux, and D. Racoceanu, “Methods for nuclei detection, segmentation, and classification in digital histopathology: A review—current

status and future potential,” IEEE Reviews in Biomedical Engineering, vol. 7, pp. 97–114, 2014.

[7] A. A. Chowdhury, S. M. H. Mahmud, M. P. Uddin, S. Kadry, J.-Y. Kim, and Y. Nam, “Nuclei segmentation and classification from histopathology images using federated learning for end-edge platform,” PLoS ONE, vol. 20, no. 7, p. e0322749, 2025.

[8] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, vol. 9351. Cham: Springer International Publishing, 2015, pp. 234–241.

[9] J. Schlemper et al., “Attention gated networks: Learning to leverage salient regions in medical images,” Medical Image Analysis, vol. 53, pp. 197–207, Apr. 2019.

[10] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, “UNet++: Redesigning skip connections to exploit multiscale features in image segmentation,” IEEE Transactions on Medical Imaging, vol. 39, no. 6, pp. 1856–1867, Jun. 2020.

[11] U. Schmidt, M. Weigert, C. Broaddus, and G. Myers, “Cell detection with star-convex polygons,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2018, vol. 11071. Cham: Springer International Publishing, 2018, pp. 265–273.

[12] J. Chen et al., “TransUNet: Transformers make strong encoders for medical image segmentation,” arXiv preprint arXiv:2102.04306, 2021.

[13] H. Cao et al., “Swin-Unet: Unet-like pure transformer for medical image segmentation,” in Computer Vision – ECCV 2022 Workshops, vol. 13803. Cham: Springer Nature Switzerland, 2023, pp. 205–218.

[14] A. Kirillov et al., “Segment anything,” arXiv preprint arXiv:2304.02643, 2023.

[15] N. Ravi et al., “SAM 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024.

[16] C. Cui et al., “All-in-sam: from weak annotation to pixelwise nuclei segmentation with prompt-based finetuning,” Journal of Physics: Conference Series, vol. 2722, no. 1, p. 012012, Mar. 2024.

[17] A. D. VandeLoo et al., “SAMCell: Generalized labelfree biological cell segmentation with segment anything,” PLoS ONE, vol. 20, no. 9, p. e0319532, 2025.

[18] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in medical images,” Nature Communications, vol. 15, no. 1, p. 654, Jan. 2024.

[19] J. Ma et al., “MedSAM2: Segment anything in 3d medical images and videos,” arXiv preprint arXiv:2504.03600, 2025.

[20] K. Zhang and D. Liu, “Customized segment anything model for medical image segmentation,” arXiv preprint arXiv:2304.13785, 2023.

[21] T. Chen et al., “SAM-Adapter: Adapting segment anything in underperformed scenes,” in 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), Paris, France, Oct. 2023, pp. 3359–3367.

[22] ——, “SAM2-Adapter: Evaluating & adapting segment anything 2 in downstream tasks: Camouflage, shadow,

medical image segmentation, and more,” Preprints, 2024.

[23] J. Wu et al., “Medical SAM adapter: Adapting segment anything model for medical image segmentation,” Medical Image Analysis, vol. 102, p. 103547, May 2025.

[24] P. Li, R. Zhou, J. He, S. Zhao, and Y. Tian, “A globalfrequency-domain network for medical image segmentation,” Computers in Biology and Medicine, vol. 164, p. 107290, Sep. 2023.

[25] N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane, and A. Sethi, “A dataset and a technique for generalized nuclear segmentation for computational pathology,” IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550–1560, Jul. 2017.

[26] N. Kumar et al., “A multi-organ nucleus segmentation challenge,” IEEE Transactions on Medical Imaging, vol. 39, no. 5, pp. 1380–1391, May 2020.

[27] A. Mahbod et al., “CryoNuSeg: A dataset for nuclei instance segmentation of cryosectioned H&E-stained histological images,” Computers in Biology and Medicine, vol. 132, p. 104349, May 2021.

[28] Q. D. Vu et al., “Methods for segmentation and classification of digital microscopy tissue images,” Frontiers in Bioengineering and Biotechnology, vol. 7, Apr. 2019.

[29] P. Naylor, M. Lae, F. Reyal, and T. Walter, “Segmentation´ of nuclei in histopathology images by deep regression of the distance map,” IEEE Transactions on Medical Imaging, vol. 38, no. 2, pp. 448–459, Feb. 2019.

[30] O. Oktay et al., “Attention U-Net: Learning where to look for the pancreas,” arXiv preprint arXiv:1804.03999, 2018.

[31] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203–211, Feb. 2021.

[32] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” arXiv preprint arXiv:1612.01105, 2016.

[33] I. Ahmad, Y. Xia, H. Cui, and Z. U. Islam, “DAN-NucNet: A dual attention based framework for nuclei segmentation in cancer histology images under wild clinical conditions,” Expert Systems with Applications, vol. 213, p. 118945, Mar. 2023.

[34] M. Heidari et al., “HiFormer: Hierarchical multi-scale representations using transformers for medical image segmentation,” in 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, Jan. 2023, pp. 6191–6201.

[35] Z. He, M. Unberath, J. Ke, and Y. Shen, “TransNuSeg: A lightweight multi-task transformer for nuclei segmentation,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, vol. 14223. Cham: Springer Nature Switzerland, 2023, pp. 206–215.

[36] D. Tellez et al., “Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology,” Medical Image Analysis, vol. 58, p. 101544, Dec. 2019.

[37] K. Stacke, G. Eilertsen, J. Unger, and C. Lundstrom,¨ “Measuring domain shift for deep learning in histopathol-

ogy,” IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 2, pp. 325–336, Feb. 2021.

[38] M. Aubreville et al., “Mitosis domain generalization in histopathology images—the MIDOG challenge,” Medical Image Analysis, vol. 84, p. 102699, Feb. 2023.

[39] C. Ouyang et al., “Causality-inspired single-source domain generalization for medical image segmentation,” IEEE Transactions on Medical Imaging, vol. 42, no. 4, pp. 1095–1106, Apr. 2023.

[40] H. Li et al., “Frequency-mixed single-source domain generalization for medical image segmentation,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, vol. 14225. Cham: Springer Nature Switzerland, 2023, pp. 127–136.

[41] F. Zou, Y. Liu, Z. Chen, K. Zhanghao, and D. Jin, “Fourier channel attention powered lightweight network for image segmentation,” IEEE Journal of Translational Engineering in Health and Medicine, vol. 11, pp. 252– 260, 2023.

[42] S. Lal, D. Das, K. Alabhya, A. Kanfade, A. Kumar, and J. Kini, “NucleiSegNet: Robust deep learning architecture for the nuclei segmentation of liver cancer histopathology images,” Computers in Biology and Medicine, vol. 128, p. 104075, Jan. 2021.

[43] J. Staal, M. D. Abramoff, M. Niemeijer, M. A. Viergever, and B. van Ginneken, “Ridge-based vessel segmentation in color images of the retina,” IEEE Transactions on Medical Imaging, vol. 23, no. 4, pp. 501–509, Apr. 2004.

[44] N. Codella et al., “Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC),” arXiv preprint arXiv:1902.03368, 2019.

[45] P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Scientific Data, vol. 5, no. 1, p. 180161, Aug. 2018.