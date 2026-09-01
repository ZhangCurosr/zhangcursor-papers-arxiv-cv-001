# Beyond Accuracy: Quantifying Pulmonary Attribution in Anatomy-Guided Chest X-Ray Classification Under Domain Shift

Abdullah Al Mamun<sup>a</sup>, Md. Nasif Osman Khansur<sup>a</sup>, Md Ashraful Hossen Akash<sup>a</sup>, Md. Kishor Morol<sup>b</sup>, Tze Hui Liew<sup>c,d,∗</sup>

<sup>a</sup>Department of Computer Science and Engineering, Rajshahi University of Engineering & Technology (RUET), Kazla, Rajshahi 6204, Bangladesh <sup>b</sup>Elite Research Lab LLC, New York, USA

<sup>c</sup>Faculty of Information Science and Technology, Multimedia University, Melaka, Malaysia <sup>d</sup>Centre for Intelligent Cloud Computing (CICC), COE of Advanced Cloud, Faculty of Information Science & Technology, Multimedia University, 75450 Melaka, Malaysia

## Abstract

Deep-learning models can achieve strong chest X-ray (CXR) classification performance without establishing whether their predictions predominantly rely on pulmonary image content. This study evaluates pulmonary attribution containment as an anatomy-related reliability property distinct from diagnostic performance. We propose DBCA-SegNet-MGAP, a multi-task anatomy-guided CNN–Transformer framework that combines complementary feature representations through bidirectional cross-backbone attention, predicts a soft lung mask, and incorporates this anatomical prior directly into classification through Mask-Guided Adaptive Global Average Pooling (MGAP). Pulmonary attribution containment is quantified using the Anatomical Local Energy Ratio (ALR) and high-intensity cumulative ALR (cALR@0.9). Experiments were repeated across three training seeds using the COVID-19 Radiography Database for four-class internal testing and a locked Shenzhen-to-Montgomery protocol for zero-shot external tuberculosis testing. On COVID-19, the proposed model achieved a weighted F1 of 0.9615 ± 0.0015 and macro ROC-AUC of $0 . 9 9 0 6 \pm 0 . 0 0 0 7$ . In an architecturematched dual-bridge comparison, replacing conventional GAP with MGAP

increased ALR from 0.3878 ± 0.0098 to $0 . 7 0 8 6 \pm 0 . 0 1 0 4$ and cALR@0.9 from $0 . 5 2 6 5 \pm 0 . 0 1 0 1 \mathrm { { \ t o \ } } 0 . 9 9 0 5 \pm 0 . 0 0 1 8$ , while weighted F1 remained essentially unchanged $( 0 . 9 6 1 8 \pm 0 . 0 0 1 5 ~ \mathrm { v s . ~ } 0 . 9 6 1 5 \pm 0 . 0 0 1 5 )$ . Under locked external transfer to Montgomery, ROC-AUC remained $0 . 9 0 8 0 \pm 0 . 0 0 4 3$ and pulmonary ALR remained $0 . 6 4 6 6 \pm 0 . 0 0 8 1$ , whereas weighted F1 decreased to $0 . 7 5 2 8 \pm 0 . 0 0 8 0$ and ECE increased to $0 . 1 6 8 3 \pm 0 . 0 0 5 5$ . These findings show that diagnostic discrimination, calibration, and pulmonary attribution containment are distinct model properties and support their joint evaluation under internal testing and external domain shift.

Keywords: Chest X-ray, pulmonary attribution, anatomy-guided classification, mask-guided pooling, explainable AI, domain shift

## 1. Introduction

Chest radiography is a widely used imaging modality for the assessment of pulmonary and thoracic abnormalities because it is rapid, comparatively inexpensive, and broadly accessible. Deep-learning methods have consequently been investigated extensively for automated chest X-ray (CXR) classification, including pneumonia, tuberculosis (TB), COVID-19, pulmonary opacity, and other thoracic conditions. CNN-based, Transformer-based, and more recent hybrid architectures have achieved strong diagnostic performance on public CXR datasets [1–3]. However, predictive performance alone does not establish whether a model has learned a diagnostically meaningful decision strategy.

This distinction is particularly important in chest radiography because a CXR contains substantial information outside the pulmonary field. Image borders, embedded text, acquisition protocol, positioning, scanner characteristics, preprocessing patterns, and dataset-specific signatures may correlate with diagnostic labels. A classifier optimized only for label prediction is not required to distinguish pulmonary evidence from such correlations. Previous work has shown that radiographic classifiers can exploit non-clinical shortcuts while maintaining apparently strong internal performance [4]. Consequently, high accuracy or ROC-AUC cannot by itself establish that a model’s decision-related attribution is concentrated within clinically relevant pulmonary anatomy.

Anatomical guidance provides one approach to reducing this disconnect. Existing studies have incorporated lung or anatomical part information through cropping, auxiliary segmentation, part-aware representation learning, mask-guided attention, and feature refinement [1, 5, 6]. These approaches demonstrate that anatomical priors can influence thoracic image classification, but they also expose an important distinction: learning pulmonary anatomy is not necessarily equivalent to using pulmonary anatomy during classification. An auxiliary network may predict the lung field accurately while the classification head remains free to aggregate discriminative information from non-pulmonary regions. Establishing anatomical prediction alone is therefore insuficient to demonstrate anatomically concentrated classification behavior.

The feature-aggregation stage provides a direct point at which this distinction can be investigated. Conventional Global Average Pooling (GAP) aggregates spatial features uniformly and does not explicitly distinguish pulmonary from extrapulmonary locations. A learned pulmonary prior can instead be incorporated directly into the aggregation rule. In this study, Mask-Guided Adaptive Global Average Pooling (MGAP) uses a jointly predicted soft lung mask to continuously weight the final spatial representation before classification. Unlike hard lung cropping, the complete radiograph remains available to the feature encoders; the predicted pulmonary probability modifies the relative contribution of spatial features during normalized pooling rather than removing image regions outright.

The representation supplied to such an aggregation mechanism may also influence its efect. CNNs provide strong inductive biases for local texture and morphology, whereas hierarchical vision Transformers ofer complementary contextual modeling. Hybrid CNN–Transformer systems have therefore become increasingly common, including approaches based on direct feature interaction, multiscale cross-attention, and lesion-aware Transformer representations [2, 7–9]. Accordingly, CNN–Transformer fusion or cross-attention alone is not treated here as the central novelty. Instead, the present study uses Bidirectional Cross-Backbone Attention Blocks (BCABs) to construct controlled local–global representation states and asks whether the efect of anatomical pooling changes with the preceding feature representation.

A second challenge concerns how anatomical behavior should be evaluated. Grad-CAM [10], Layer-CAM [11], and related methods are widely used to visualize class-associated image regions, but selected heatmaps provide limited evidence about model behavior across an entire test cohort. Moreover, visually plausible saliency does not necessarily imply accurate localization or faithful explanation [3]. The present study therefore separates qualitative visualization from quantitative pulmonary attribution containment. The Anatomical Local Energy Ratio (ALR) measures the fraction of total predicted-class attribution mass contained within the lung field, whereas cumulative ALR at a high attribution threshold (cALR@0.9) evaluates whether the strongest retained attribution is also pulmonary. These measures are deliberately interpreted as attribution containment rather than lesion localization, causal explanation, or diagnostic correctness.

Reliability must additionally be examined beyond the development distribution. Deep-learning systems that perform strongly during internal testing often deteriorate when evaluated on external cohorts because patient populations, acquisition equipment, disease prevalence, imaging protocols, and other data characteristics may difer [12]. Furthermore, diferent reliability properties need not deteriorate equally. ROC-AUC reflects discrimination or case ranking, whereas fixed-threshold metrics depend on the selected operating point, and probability calibration describes whether predicted confidence remains numerically meaningful [13]. A model may therefore retain useful external discrimination while showing poorer calibration or threshold-dependent classification performance. Current reporting guidance for medical-imaging AI consequently emphasizes explicit separation of internal and external testing and transparent description of data partitions and reference standards [14].

Figure 1 summarizes the reliability gap motivating the study. The potential non-pulmonary cues shown in the diagram are conceptual shortcut risks rather than confirmed features used by the trained models.

These considerations motivate a beyond-accuracy perspective in which diagnostic competence, pulmonary attribution containment, calibration, and external robustness are treated as related but non-equivalent model properties. To investigate these relationships, we develop DBCA-SegNet-MGAP, a multi-task anatomy-guided CNN–Transformer framework. ResNet-50 and Swin-Tiny provide complementary convolutional and hierarchical Transformer representations, while BCABs permit reciprocal feature exchange at intermediate resolutions. A progressive decoder jointly predicts a continuous lung probability map, which is used both for auxiliary anatomical supervision and as the spatial prior for MGAP.

The experimental design is structured to isolate these efects. Conventional CNN and Transformer classifiers provide diagnostic reference baselines, while a lightweight architecture serves as a capacity control. For the fullcapacity framework, late-fusion, single-bridge, and dual-bridge representations are each evaluated with GAP and MGAP, enabling architecture-matched GAP→MGAP comparisons without changing the preceding representation. Lung segmentation and classifier attribution are evaluated separately, and principal comparisons are repeated across three independent training seeds.

![](images/41b4ce58a4bae4add4001643fb4c518483b15d45d6ffe135fa3d2e31382c0d08.jpg)  
Figure 1: Reliability gap in chest X-ray classification: diagnostic performance versus pulmonary attribution containment and external robustness. Strong internal diagnostic performance does not establish whether class-related attribution is concentrated within the pulmonary field or whether diagnostic behavior is preserved under external cohort shift. The pulmonary anatomical prior represents the target anatomical region used for anatomy-guided feature aggregation. Potential non-pulmonary cues are shown as conceptual examples and are not presented as confirmed shortcuts used by the trained models.

External reliability is evaluated through a locked Shenzhen-to-Montgomery TB protocol. Model development and checkpoint selection are performed using Shenzhen only. The selected checkpoint is then applied to Montgomery without target-domain fine-tuning, recalibration, checkpoint reselection, or threshold optimization. This permits discrimination, fixed-threshold performance, calibration, and pulmonary attribution to be examined independently under cohort shift.

The study makes four connected contributions:

1. a jointly predicted soft pulmonary prior is incorporated directly into normalized feature aggregation through MGAP rather than remaining solely an auxiliary segmentation output or being converted into a hard crop;

2. quantitative pulmonary-attribution metrics and architecture-matched GAP/MGAP comparisons isolate the anatomical aggregation intervention from changes in the preceding representation;

3. lung-segmentation quality is explicitly separated from pulmonary classifier attribution, testing whether learning pulmonary anatomy and using it for classification are equivalent properties; and

4. diagnostic performance, calibration, and pulmonary attribution are evaluated separately under a locked external domain shift, with a lightweight capacity-control configuration testing whether strong pulmonary containment alone is suficient for robust transfer.

Accordingly, the study addresses the following research questions:

RQ1. Can classifiers with comparable diagnostic performance exhibit substantially diferent pulmonary-attribution distributions?

RQ2. Can mask-guided feature aggregation increase pulmonary attribution containment without materially compromising diagnostic competence, and does this efect depend on the preceding feature representation?

RQ3. Are diagnostic discrimination, probability calibration, and pulmonary attribution containment preserved to the same extent under external cohort shift?

The objective is not to establish universal superiority of a particular backbone, attention block, or pooling architecture. Instead, the study examines whether diagnostic competence, pulmonary attribution containment, calibration, and external robustness can vary independently, and whether explicit anatomical feature aggregation can make pulmonary attribution more directly measurable and controllable.

## 2. Related Work

2.1. Deep learning and hybrid representation learning for CXR classification

Deep-learning approaches to CXR analysis have progressed from predominantly convolutional architectures toward Transformer-based and hybrid representation learning. CNNs such as ResNet [15], DenseNet [16], Eficient-Net [17], and ConvNeXt [18] remain relevant baselines because hierarchical convolution provides an efective inductive bias for localized radiographic texture and morphology. Hierarchical vision Transformers provide a complementary mechanism for content-dependent contextual modeling; Swin Transformer [19] combines multiscale feature extraction with shifted-window self-attention.

Hybrid CNN–Transformer methods seek to exploit these complementary properties. CTransCNN [2] introduced direct cross-representation interaction for medical-image classification, while more recent CXR systems have investigated multiscale cross-attention, lesion-aware hybrid representations, and explainable hybrid Transformers [7–9]. These studies establish that hybridization and cross-attention are active and increasingly mature research directions. The present work therefore does not position cross-attention alone as the main contribution; instead, BCABs provide controlled representation states on which the efect of anatomical aggregation can be tested.

## 2.2. Anatomy-guided and segmentation-assisted classification

Anatomical information has been incorporated into thoracic classification through lung cropping, part-aware representations, auxiliary segmentation, spatial attention, and anatomy-guided feature refinement. Zhang et al. [1] introduced part-aware mask-guided attention to integrate anatomical region information into thoracic disease classification. More recent anatomy-guided approaches have incorporated anatomical priors through progressive feature refinement and segmentation-derived localization mechanisms [5, 6]. These studies establish that pulmonary priors can influence classification behavior.

However, successful lung segmentation does not necessarily establish that the final classification head relies strongly on pulmonary features. An auxiliary decoder can learn a high-quality anatomical task while the classifier continues to aggregate information from pulmonary and extrapulmonary spatial locations. The present work targets this distinction by inserting a predicted soft lung prior directly into normalized spatial aggregation and comparing GAP and MGAP within otherwise matched representations.

## 2.3. Explainability, shortcut learning, and quantitative pulmonary attribution

Grad-CAM [10] and Layer-CAM [11] are widely used to inspect classassociated spatial regions. Recent CXR studies similarly combine diagnostic models with visual explanation [9, 20, 21]. Nevertheless, qualitative plausibility is not equivalent to validated localization. Saliency benchmarking in CXR analysis has demonstrated important limitations in localization performance [3], while shortcut-learning studies have shown that radiographic classifiers may exploit unintended acquisition- or dataset-associated signals [4].

These observations motivate cohort-level quantitative containment rather than reliance on selected heatmaps alone. ALR measures the fraction of predicted-class attribution mass inside the pulmonary field, while cALR@0.9 asks whether the strongest retained attribution is also pulmonary. Neither metric is interpreted as lesion localization, explanation faithfulness, or causal reasoning.

## 2.4. Domain shift, calibration, and external reliability

Internal test performance does not guarantee preservation of model behavior across cohorts. External evaluation of radiologic deep-learning systems frequently reveals performance deterioration under changes in patient population, acquisition equipment, protocols, and other site-specific factors [12]. Reliability under domain shift is also multidimensional: ROC-AUC measures ranking discrimination, threshold-dependent metrics depend on the operating point, and calibration measures whether predicted confidence remains numerically meaningful [13].

Public Shenzhen and Montgomery CXR cohorts provide a relevant setting for TB transfer analysis [22]. Previous work has demonstrated the feasibility of deep-learning-based TB screening and cross-population evaluation [23, 24]. The present study difers in its explicitly locked source-to-target protocol: Shenzhen is used for model development and checkpoint selection, whereas Montgomery is reserved for external testing without target-domain fine-tuning, recalibration, checkpoint reselection, or threshold optimization.

## 2.5. Research gap and study positioning

Across the literature, anatomical prediction, anatomical use, diagnostic performance, and external reliability are often investigated as separate problems. The unresolved issue addressed here is not simply how to construct another accurate hybrid classifier, but how to connect a learned pulmonary prior directly to classifier aggregation, isolate the efect of that intervention from the preceding representation, quantify pulmonary attribution across a test cohort, and determine whether diagnostic and anatomy-related properties are preserved similarly under cohort shift.

DBCA-SegNet-MGAP is positioned around this experimental gap. The study therefore treats

$$
\begin{array} { r l } & { \mathrm { d i a g n o s t i c ~ c o m p e t e n c e } \neq \mathrm { p u l m o n a r y - p r i o r ~ q u a l i t y } } \\ & { \qquad \neq \mathrm { p u l m o n a r y ~ a t t r i b u t i o n ~ c o n t a i n m e n t } } \\ & { \qquad \neq \mathrm { e x t e r n a l ~ r e l i a b i l i t y } , } \end{array}
$$

and tests their relationships under controlled internal and locked external evaluation.

## 3. Methodology

## 3.1. Study design and experimental framework

This study employed a retrospective controlled computational design to evaluate CXR classification beyond diagnostic performance alone. Four complementary properties were examined: diagnostic discrimination, probability calibration, pulmonary segmentation, and pulmonary attribution containment. External robustness was additionally evaluated through a locked cross-cohort TB experiment.

Two independent task tracks were used. The first used the COVID-19 Radiography Database for four-class classification of Normal, Lung Opacity, COVID-19, and Viral Pneumonia. The second used Shenzhen for binary Normal-versus-TB development and internal testing, followed by external testing on Montgomery. The task-specific models used separate classifier outputs and checkpoints.

The experimental hierarchy comprised conventional pretrained classifiers as diagnostic baselines, a lightweight configuration as a capacity control, architecture-matched GAP/MGAP conditions for mechanism isolation, and the complete DBCA-SegNet-MGAP framework. Montgomery was treated strictly as an external test cohort and was not used for training, model selection, recalibration, normalization adaptation, or decision-threshold optimization, consistent with the distinction between internal and external testing emphasized in CLAIM [14].

The overall workflow is shown in Fig. 2. Following data validation, manifest construction, deterministic partitioning, and paired preprocessing, the COVID-19 and TB tracks were executed independently. The TB checkpoint was locked before Montgomery evaluation.

## 3.2. Datasets and reference standards

Three publicly available CXR cohorts were used. Their roles were predefined to separate internal performance assessment from external domain-shift evaluation.

## 3.2.1. COVID-19 Radiography Database

The COVID-19 Radiography Database was assembled through the public releases described by Chowdhury et al. [25] and Rahman et al. [26]. The version used here contained 21,165 radiographs: 10,192 Normal, 6,012 Lung Opacity, 3,616 COVID-19, and 1,345 Viral Pneumonia images. A paired lung mask was available for each image. The fixed partition contained 14,815 training images, 3,175 model-selection images, and 3,175 held-out internal-test images. Because reliable patient identifiers are not consistently exposed in the public release, partitioning was performed at image level; this is retained as a study limitation.

![](images/cbe626e966233da7fcdfd2776b48789c4dadaf4d593f0580d5e07b1f2cfcf7f4.jpg)  
Figure 2: Experimental workflow for four-class COVID-19 evaluation and locked Shenzhen to-Montgomery TB transfer. The COVID-19 track uses held-out internal testing, whereas the TB track uses Shenzhen for development and checkpoint selection followed by locked Montgomery zero-shot testing. Montgomery is not used for target-domain fine-tuning, recalibration, checkpoint reselection, or threshold optimization.

## 3.2.2. Shenzhen Tuberculosis Dataset

The Shenzhen dataset contains 662 frontal CXRs, comprising 326 Normal and 336 TB cases [22]. The fixed partition contained 463 training images, 99 model-selection images, and 100 held-out internal-test images. Each Shenzhen radiograph was paired with a teacher-generated pulmonary pseudo-mask for anatomy-guided training. These pseudo-masks provide auxiliary anatomical supervision but are not treated as independent expert reference standards.

Table 1: Dataset composition and experimental role.
<table><tr><td>Dataset</td><td>Task</td><td>Classes</td><td></td><td>Train Model selection Internal test External test Lung reference</td><td></td><td></td></tr><tr><td>COVID-19 Radiography Four-class</td><td></td><td>Normal, Lung Opacity, COVID-19, Viral Pneumonia</td><td>14,815</td><td>3,175</td><td>3,175</td><td>Paired lung masks</td></tr><tr><td>Shenzhen</td><td>Binary TB</td><td>Normal, TB</td><td>463</td><td>99</td><td>100</td><td>Teacher pseudo-masks</td></tr><tr><td>Montgomery</td><td></td><td>Binary TB Normal, TB</td><td></td><td></td><td></td><td>138 Manual left/right lung union</td></tr></table>

## 3.2.3. Montgomery County Dataset

The Montgomery County dataset contains 138 frontal CXRs, including 80 Normal and 58 TB cases [22]. Separate manual left- and right-lung masks are provided; the pixel-wise union of these masks was used as the pulmonary reference region. All 138 cases were reserved exclusively for external testing.

## 3.3. Data preparation and reproducibility

## 3.3.1. Manifest construction and integrity checks

Each cohort was converted into an explicit image–mask manifest containing the radiograph path, diagnostic label, class name, dataset identity, associated mask path or components, and mask provenance. COVID-19 masks were matched using identical filenames, Shenzhen pseudo-masks by filename stem, and Montgomery records required both manual lung-mask components. Missing directories, missing masks, unreadable images, or empty manifests terminated processing rather than silently removing observations.

## 3.3.2. Fixed partitioning and multi-seed training

COVID-19 and Shenzhen were partitioned using a two-stage stratified 70%/15%/15% procedure. The partition was generated once using split seed 42 and remained fixed throughout the multi-seed experiments. For each subset $D _ { s }$ , the ordered relative image identifiers were stored and hashed as

$$
h _ { s } = \mathrm { S H A 2 5 6 } \left( \mathrm { j o i n } \{ r _ { i } : i \in D _ { s } \} \right) ,\tag{1}
$$

where $s \in$ {train, selection, test}. The split fingerprints were stored with the experiment metadata and checked on resumed runs. Final training was repeated independently with seeds 12, 42, and 112 while preserving identical data membership.

## 3.3.3. Image and mask preprocessing

Radiographs were loaded in grayscale and converted to three identical channels for ImageNet-pretrained encoders. COVID-19 images underwent contrast-limited adaptive histogram equalization (CLAHE) with clip limit

2.0 and an $8 \times 8$ tile grid, following the enhancement setting investigated on the same resource [26]. Shenzhen images were consumed in their curated stored form, whereas raw Montgomery images received the predefined CLAHE operation before external inference. This dataset-specific rule was fixed before external testing and was not selected using Montgomery labels or outcomes.

Images were resized while preserving aspect ratio so that the longest dimension equaled 224 pixels and were zero-padded to $2 2 4 \times 2 2 4$ . Lung masks were resized with nearest-neighbor interpolation. ImageNet normalization was applied using

$$
\pmb { \mu } = ( 0 . 4 8 5 , 0 . 4 5 6 , 0 . 4 0 6 ) , \qquad \pmb { \sigma } = ( 0 . 2 2 9 , 0 . 2 2 4 , 0 . 2 2 5 ) .\tag{2}
$$

Reference masks were represented as binary $1 \times 2 2 4 \times 2 2 4$ tensors.

Training-only augmentation was applied synchronously to each image– mask pair using Albumentations [27]. The policy included shift and scale changes of up to $5 \% .$ , rotation up to $1 0 ^ { \circ }$ , brightness/contrast perturbation within ±0.1, and horizontal flipping, with probability 0.5 for each augmentation group. Model-selection, internal-test, and external-test images received deterministic preprocessing only.

## 3.3.4. Class balancing and deterministic execution

A weighted random sampler with replacement was used during training. For sample i belonging to class $y _ { i }$

$$
w _ { i } = \frac { 1 } { n _ { y _ { i } } } ,\tag{3}
$$

where $n _ { y _ { i } }$ denotes the number of training examples in that class. Modelselection and test loaders were not shufled. Python, NumPy, PyTorch CPU/CUDA generators, data-loader workers, and sampling generators were initialized deterministically for each training seed; cuDNN benchmarking was disabled and deterministic algorithms were requested where supported.

## 3.4. Proposed DBCA-SegNet-MGAP framework

The proposed framework integrates local convolutional representation, hierarchical Transformer context, bidirectional cross-backbone interaction, pulmonary segmentation, and mask-guided classification. ResNet-50 [15] was used as the convolutional encoder and Swin-Tiny [19] as the hierarchical Transformer encoder. Intermediate representations were exchanged through two BCABs.

![](images/f747ac973f87f69958b249b3ff172673f91ec28108b74f28412a1d7588081ec4.jpg)  
Figure 3: Overall DBCA-SegNet-MGAP architecture. The dual-encoder framework uses BCAB1 at $2 8 \times 2 8$ , BCAB2 at $1 4 \times 1 4$ , and final $7 \times 7$ multi-scale feature fusion before pulmonary segmentation and mask-guided classification.

Figures 3–5 separately summarize the overall architecture, BCAB mechanism, and progressive decoder with the MGAP classifier.

## 3.4.1. Dual feature encoders

For an input $x \in \mathbb { R } ^ { 3 \times 2 2 4 \times 2 2 4 }$ , the ResNet branch produced

$$
R _ { 1 } \in \mathbb { R } ^ { 2 5 6 \times 5 6 \times 5 6 } ,
$$

$$
R _ { 2 } \in \mathbb { R } ^ { 5 1 2 \times 2 8 \times 2 8 } ,\tag{4}
$$

$$
R _ { 3 } \in \mathbb { R } ^ { 1 0 2 4 \times 1 4 \times 1 4 } ,
$$

$$
R _ { 4 } \in \mathbb { R } ^ { 2 0 4 8 \times 7 \times 7 } ,\tag{5}
$$

while the Swin branch produced

$$
S _ { 1 } \in \mathbb { R } ^ { 9 6 \times 5 6 \times 5 6 } ,
$$

$$
S _ { 2 } \in \mathbb { R } ^ { 1 9 2 \times 2 8 \times 2 8 } ,\tag{6}
$$

$$
S _ { 3 } \in \mathbb { R } ^ { 3 8 4 \times 1 4 \times 1 4 } ,
$$

$$
S _ { 4 } \in \mathbb { R } ^ { 7 6 8 \times 7 \times 7 } .\tag{7}
$$

BCAB1 operated on $R _ { 2 }$ and $S _ { 2 }$ , whereas BCAB2 operated on $R _ { 3 }$ and $S _ { 3 }$ The BCAB2-updated ResNet representation was propagated through the final ResNet stage so that the second interaction influenced the final classifier representation.

## 3.4.2. Bidirectional Cross-Backbone Attention

For same-resolution ResNet and Swin features R and S, independent 1 × 1 projections aligned channels to an attention width d. Flattening the aligned

![](images/6ccb6e255b64c183ec56ca572784897cebc5a49a190e1fc808afab4db34db850.jpg)  
Figure 4: Internal mechanism of the Bidirectional Cross-Backbone Attention Block (BCAB). CNN context is transferred to the Swin stream, and Swin context is transferred to the CNN stream through reciprocal cross-attention and residual feature updates.

spatial features produced

$$
C = \mathrm { t o k e n s } ( \phi _ { R } ( R ) ) , \qquad T = \mathrm { t o k e n s } ( \phi _ { S } ( S ) ) , \qquad C , T \in \mathbb { R } ^ { N \times d } ,\tag{8}
$$

where $N = H W$

Swin tokens queried CNN keys and values:

$$
A _ { T  C } = \mathrm { s o f t m a x } ( { \frac { Q _ { T } ( T ) K _ { C } ( C ) ^ { \top } } { \sqrt { d } } } ) ,\tag{9}
$$

$$
U _ { T } = A _ { T  C } V _ { C } ( C ) .\tag{10}
$$

Conversely,

$$
A _ { C  T } = \mathrm { s o f t m a x } ( { \frac { Q _ { C } ( C ) K _ { T } ( T ) ^ { \top } } { \sqrt { d } } } ) ,\tag{11}
$$

$$
U _ { C } = A _ { C  T } V _ { T } ( T ) .\tag{12}
$$

![](images/61c36b538866231e4f88c629b489a58c32395e94fdbe1824fb9fea7c947cd757.jpg)  
Figure 5: Progressive lung decoder and Mask-Guided Adaptive Global Average Pooling (MGAP) classifier. The decoder reconstructs a soft pulmonary mask from fused features and multiscale ResNet skip connections; MGAP resizes this mask to classifier resolution and uses it to weight normalized spatial feature aggregation.

The context tensors were restored to spatial form, projected back to their original channel widths, and added residually:

$$
R ^ { \prime } = R + \psi _ { R } ( U _ { C } ) , \qquad S ^ { \prime } = S + \psi _ { S } ( U _ { T } ) .\tag{13}
$$

BCAB1 used $d = 2 5 6$ and BCAB2 used $d = 5 1 2$

## 3.4.3. Feature fusion and pulmonary decoder

The final encoder representations were concatenated and projected to a shared feature map:

$$
\begin{array} { r } { F _ { 0 } = \mathrm { R e L U } \left[ \mathrm { B N } \left( \mathrm { C o n v } _ { 1 \times 1 } ( [ R _ { 4 } ; S _ { 4 } ] ) \right) \right] , \qquad F _ { 0 } \in \mathbb { R } ^ { 1 0 2 4 \times 7 \times 7 } . } \end{array}\tag{14}
$$

Channel recalibration used a squeeze-and-excitation operation [28]:

$$
s = \sigma \left[ W _ { 2 } \delta \left( W _ { 1 } \mathrm { G A P } ( F _ { 0 } ) \right) \right] , \qquad F = F _ { 0 } \odot s .\tag{15}
$$

A progressive decoder reconstructed the lung field from the $7 \times 7$ fused representation using five upsampling stages. Projected ResNet skip features were incorporated at $1 4 \times 1 4 , 2 8 \times 2 8 , 5 6 \times 5 6$ , and $1 1 2 \times 1 1 2$ , following the multiscale encoder–decoder principle of U-Net [29]. A final $1 \times 1$ convolution produced

$$
G \in \mathbb { R } ^ { 1 \times 2 2 4 \times 2 2 4 }\tag{16}
$$

lung-mask logits. The continuous probability map $\sigma ( G )$ was retained for classification rather than converted into a hard crop.

## $\ 3 . 4 . 4 .$ Mask-Guided Adaptive Global Average Pooling

Let $\boldsymbol { F } \in \mathbb { R } ^ { C \times h \times w }$ denote the final classifier feature representation and

$$
\widetilde { M } = \mathrm { R e s i z e } _ { h , w } [ \sigma ( G ) ] , \qquad h = w = 7 ,\tag{17}
$$

the predicted soft lung mask at classifier resolution. MGAP computes

$$
z _ { c } = \frac { \displaystyle \sum _ { i = 1 } ^ { h } \sum _ { j = 1 } ^ { w } F _ { c i j } \widetilde { M } _ { i j } } { \displaystyle \sum _ { i = 1 } ^ { h } \sum _ { j = 1 } ^ { w } \widetilde { M } _ { i j } + \epsilon } , \qquad \epsilon = 1 0 ^ { - 6 } .\tag{18}
$$

The mask therefore acts as a continuous spatial weighting mechanism rather than a hard crop. The pooled vector is passed through dropout $( p = 0 . 2 )$ and a linear classifier:

$$
\hat { \mathbf { y } } = W _ { \mathrm { c l s } } \mathrm { D r o p o u t } ( z ) + b _ { \mathrm { c l s } } .\tag{19}
$$

The output dimension is four for the COVID-19 task and two for TB. Architecture-matched GAP controls omit the soft-mask weighting while retaining the same preceding representation.

## 3.5. Multi-task training and optimization

Disease classification was optimized using cross-entropy with label smoothing $\alpha = 0 . 0 5$ . For reference class $y .$

$$
q _ { k } = ( 1 - \alpha ) \mathbb { I } [ k = y ] + \frac { \alpha } { K } ,\tag{20}
$$

and

$$
\mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sum _ { k = 1 } ^ { K } q _ { b k } \log p _ { b k } .\tag{21}
$$

The pulmonary decoder was optimized using binary cross-entropy with logits plus soft Dice loss. For $\hat { m } _ { i } = \sigma ( g _ { i } )$ and reference pixel $m _ { i }$

$$
\mathrm { D i c e } _ { \mathrm { s o f t } } = \frac { 2 \sum _ { i } \hat { m } _ { i } m _ { i } + \epsilon } { \sum _ { i } \hat { m } _ { i } + \sum _ { i } m _ { i } + \epsilon } ,\tag{22}
$$

$$
\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { B C E } } + \left( 1 - \mathrm { D i c e } _ { \mathrm { s o f t } } \right) .\tag{23}
$$

The joint objective was

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { s e g } } \mathcal { L } _ { \mathrm { s e g } } , \qquad \lambda _ { \mathrm { s e g } } = 0 . 2 .\tag{24}
$$

Models were trained for 30 epochs using AdamW [30]. The initial learning rate was $2 \times 1 0 ^ { - 4 }$ and was cosine-annealed toward $1 0 ^ { - 6 }$ . A physical batch size of 8 with two-step gradient accumulation produced an efective batch size of 16. Gradients were clipped to a maximum norm of 1.0, training used FP32 precision, and the best checkpoint was selected by model-selection weighted F1. Experiments were implemented in PyTorch and executed on an NVIDIA Tesla T4 GPU.

Table 2: Shared training configuration.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Input</td><td> $2 2 4 \times 2 2 4 \times 3$ </td></tr><tr><td>Initialization</td><td>ImageNet-pretrained encoders</td></tr><tr><td>Epochs</td><td>30</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }  1 0 ^ { - 6 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>8 physical / 16 effective</td></tr><tr><td>Loss</td><td> $\mathcal { L } _ { c l s } + 0 . 2 \mathcal { L } _ { s e g }$ </td></tr><tr><td>Label smoothing</td><td>0.05</td></tr><tr><td>Dropout</td><td>0.20</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Checkpoint criterion</td><td>Model-selection weighted F1</td></tr><tr><td>Data-partition seed</td><td>42</td></tr><tr><td>Training seeds</td><td>12, 42, 112</td></tr></table>

Table 3: Controlled experimental configurations.
<table><tr><td>Representation</td><td>Encoders</td><td>BCAB1</td><td>BCAB2</td><td>Pooling</td><td>Experimental role</td></tr><tr><td>Late fusion</td><td>ResNet-50 + Swin-Tiny</td><td></td><td></td><td>GAP MGAP</td><td>No-bridge mechanism control</td></tr><tr><td>Single bridge</td><td>ResNet-50 + Swin-Tiny</td><td>√</td><td></td><td>GAP MGAP</td><td>Intermediate-fusion control</td></tr><tr><td>Dual bridge</td><td>ResNet-50 + Swin-Tiny</td><td>√</td><td></td><td>GAP /MGAP</td><td>Complete architecture</td></tr><tr><td>Lightweight</td><td>EfficientNet-B0 + TinyViT-5M</td><td></td><td></td><td>GAP /MGAP</td><td>Capacity control</td></tr></table>

## 3.6. Baselines and controlled experimental design

Four pretrained classifiers were retained as conventional diagnostic baselines: ResNet-50 [15], Swin-Tiny [19], DenseNet-121 [16], and ConvNeXt-Tiny [18]. Each used its pretrained feature extractor, conventional GAP, dropout, and a task-specific linear classifier.

The mechanism experiment used a factorial architecture-matched design. Three full-capacity representation conditions were defined: late fusion, singlebridge BCAB1, and dual-bridge BCAB1+BCAB2. Each representation was paired with either GAP or MGAP. The lightweight capacity-control pair used EficientNet-B0 [17] and TinyViT-5M [31] with final-stage fusion and no BCAB. It was not used to infer the BCAB efect.

## 3.7. Evaluation and statistical analysis

## 3.7.1. Diagnostic performance and calibration

Primary diagnostic metrics were accuracy, weighted F1, macro F1, and ROC-AUC. Weighted F1 accounts for the imbalanced COVID-19 class distribution, while macro F1 assigns equal contribution to each class. For binary TB testing, sensitivity and specificity at the fixed threshold 0.5 were treated as secondary threshold-dependent measures. Multiple metrics were retained because metric choice can change the interpretation and ranking of medical-imaging systems [32].

Calibration was summarized using 15-bin Expected Calibration Error (ECE):

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { 1 5 } \frac { | B _ { m } | } { N } \left| \operatorname { a c c } ( B _ { m } ) - \operatorname { c o n f } ( B _ { m } ) \right| ,\tag{25}
$$

where $B _ { m }$ is the set of samples assigned to confidence bin m.

## 3.7.2. Pulmonary segmentation

Predicted lung masks were thresholded at 0.5 for segmentation evaluation. The primary segmentation metric was Dice:

$$
\mathrm { D i c e } ( P , M ) = \frac { 2 | P \cap M | + \epsilon } { | P | + | M | + \epsilon } .\tag{26}
$$

COVID-19 predictions were evaluated against paired dataset masks, Shenzhen predictions against teacher-generated pseudo-masks, and Montgomery predictions against the union of manual left- and right-lung masks. These measurements evaluate lung-field segmentation rather than disease-lesion localization.

## 3.7.3. Pulmonary attribution containment

Grad-CAM [10] was used for the primary quantitative containment analysis. In DBCA-SegNet-MGAP, the Grad-CAM target was the 7×7 mask-influenced classifier representation immediately before MGAP; the matched GAP control used the corresponding unweighted fused representation. Layer-CAM [11] was generated from the BCAB2-updated 14 × 14 ResNet representation as a complementary higher-resolution qualitative view. Attribution maps were upsampled to image resolution and min–max normalized.

Let $A ( i ) \geq 0$ denote normalized predicted-class Grad-CAM attribution and $M ( i ) \in \{ 0 , 1 \}$ the pulmonary reference mask. ALR was defined as

$$
\mathrm { A L R } = \frac { \sum _ { i } A ( i ) M ( i ) } { \sum _ { i } A ( i ) + \epsilon } .\tag{27}
$$

High-intensity containment was summarized using

$$
\mathrm { c A L R } \ @ \tau = \frac { \sum _ { i } A ( i ) \mathbb { I } [ A ( i ) \geq \tau ] M ( i ) } { \sum _ { i } A ( i ) \mathbb { I } [ A ( i ) \geq \tau ] + \epsilon } .\tag{28}
$$

The main analysis reports cALR@0.9. It describes where attribution mass surviving the 0.9 threshold is located; it does not represent the fraction of the lung attended to and is not interpreted as lesion localization, explanation faithfulness, or causal evidence.

For selected qualitative TB cases, normalized class-specific Grad-CAM maps were probability weighted:

$$
A _ { y } ^ { w } ( x ) = P ( y \mid x ) A _ { y } ( x ) .\tag{29}
$$

The signed diference

$$
D ( x ) = A _ { \mathrm { T B } } ^ { w } ( x ) - A _ { \mathrm { N o r m a l } } ^ { w } ( x )\tag{30}
$$

was visualized to examine spatial class dominance. This analysis was descriptive and was not used as a quantitative endpoint.

## 3.7.4. Multi-seed variability

Principal experiments were independently repeated with training seeds 12, 42, and 112 on the same fixed data partition. Results are summarized as mean ± sample SD. The across-seed SD characterizes stochastic training variability; with only three runs, it is not interpreted as a formal confidence interval or evidence of statistical superiority.

## 3.7.5. Locked external zero-shot evaluation

For the external experiment, model development and checkpoint selection used Shenzhen only. The checkpoint with the highest Shenzhen modelselection weighted F1 was frozen before Montgomery evaluation. The same checkpoint was applied to all 138 Montgomery radiographs with no targetdomain training, no fine-tuning, no target-specific checkpoint selection, no probability recalibration, no threshold optimization, and a fixed TB decision threshold of 0.5. ROC-AUC characterized external discrimination, weighted F1 and threshold-dependent metrics characterized operating-point transfer, ECE characterized calibration shift, and pulmonary attribution was reassessed using the same $\mathrm { A L R / c A L R }$ definitions.

## 3.7.6. Model complexity and inference eficiency

Model complexity was summarized using trainable parameter count, serialized checkpoint size, and inference throughput:

$$
\mathrm { T h r o u g h p u t } = \frac { N _ { \mathrm { i m a g e s } } } { t _ { \mathrm { i n f e r e n c e } } } ,\tag{31}
$$

after GPU synchronization. Throughput is hardware- and implementationdependent and is therefore interpreted only as a within-study comparison.

## 4. Results

All principal internal model comparisons and COVID-19 mechanism experiments were evaluated over three training runs using seeds 12, 42, and 112 while retaining the fixed partitions described in Section 3. Unless otherwise stated, results are reported as mean ± sample SD. The SD characterizes run-to-run variability and is not interpreted as a formal test of statistical superiority.

## 4.1. Internal diagnostic performance and pulmonary attribution

DBCA-SegNet-MGAP showed stable diagnostic performance on the fourclass COVID-19 test set. Across the three runs, accuracy was $0 . 9 6 1 6 \pm 0 . 0 0 1 6 .$ weighted F1 was $0 . 9 6 1 5 \pm 0 . 0 0 1 5$ , macro F1 was $0 . 9 6 8 6 \pm 0 . 0 0 1 4$ , macro ROC-AUC was $0 . 9 9 0 6 \pm 0 . 0 0 0 7$ , and ECE was $0 . 0 2 4 7 \pm 0 . 0 0 1 3$

At class level, COVID-19 was the most consistently classified category, with $\mathrm { F 1 } = 0 . 9 9 6 3 { \pm } 0 . 0 0 0 9 .$ . Viral Pneumonia achieved $0 . 9 7 9 8 { \pm } 0 . 0 0 1 1$ , Normal $0 . 9 6 1 1 \pm 0 . 0 0 1 3$ , and Lung Opacity $0 . 9 3 7 2 \pm 0 . 0 0 2 4$ . Lung Opacity therefore remained the most dificult class. Figure 6 shows the representative Seed-42 confusion matrix and one-vs-rest ROC curves; aggregate multi-seed results are reported separately in Table 4.

The diagnostic and pulmonary-attribution metrics produced diferent model rankings. $\mathrm { { D B C A - L i g h t } + M G A P }$ achieved the highest mean accuracy and weighted F1, while Swin-Tiny achieved the highest macro AUC. DBCA-SegNet-MGAP was therefore not the highest-performing classifier on every conventional endpoint.

![](images/e92f2bb9918b2b0a75b351f1817284eab797e5df03a21274b6f06e131dc4db3b.jpg)

![](images/ee99d44068893b1d45231592e880e6b585d28a391d68902f7823370fb75c9aa2.jpg)  
Figure 6: Representative Seed-42 diagnostic behavior of DBCA-SegNet-MGAP on the held-out four-class COVID-19 test set: confusion matrix (left) and one-vs-rest ROC curves (right). The figure is shown for qualitative error-pattern inspection; multi-seed aggregate metrics are reported in Table 4.

Table 4: Multi-seed COVID-19 diagnostic performance, calibration, and pulmonaryattribution comparison.
<table><tr><td rowspan="2">Configuration</td><td colspan="4">Diagnostic performance and calibration</td><td colspan="2">Pulmonary attribution</td></tr><tr><td>Acc.</td><td> $\mathrm { F } 1 _ { w }$ </td><td>AUC</td><td>ECE</td><td>ALR</td><td>cALR@0.9</td></tr><tr><td> $\mathrm { R e s N e t - 5 0 }$ </td><td> $0 . 9 5 4 5 \pm 0 . 0 0 1 7$ </td><td> $0 . 9 5 4 6 \pm 0 . 0 0 1 7$ </td><td> $0 . 9 9 2 5 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 2 9 8 \pm 0 . 0 0 1 3$ </td><td> $0 . 4 9 8 1 \pm 0 . 0 1 0 0$ </td><td> $0 . 7 6 5 9 \pm 0 . 0 1 0 7$ </td></tr><tr><td>Swin-Tiny</td><td> $0 . 9 6 2 8 \pm 0 . 0 0 1 4$ </td><td> $0 . 9 6 2 7 \pm 0 . 0 0 1 4$ </td><td> $\mathbf { 0 . 9 9 3 4 \pm 0 . 0 0 0 7 }$ </td><td> $0 . 0 3 0 7 \pm 0 . 0 0 1 3$ </td><td> $0 . 2 4 1 5 \pm 0 . 0 0 6 1$ </td><td> $0 . 2 0 5 4 \pm 0 . 0 0 6 5$ </td></tr><tr><td> $\mathrm { D e n s e N e t - 1 2 1 }$ </td><td> $0 . 9 6 2 0 \pm 0 . 0 0 1 1$ </td><td> $0 . 9 6 2 0 \pm 0 . 0 0 1 1$ </td><td> $0 . 9 9 1 1 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 3 0 8 \pm 0 . 0 0 1 4$ </td><td> $0 . 3 2 2 8 \pm 0 . 0 0 7 8$ </td><td> $0 . 3 5 4 6 \pm 0 . 0 0 8 3$ </td></tr><tr><td> $\mathrm { C o n v N e X t - T i n y }$ </td><td> $0 . 9 5 6 0 \pm 0 . 0 0 1 2$ </td><td> $0 . 9 5 6 0 \pm 0 . 0 0 1 2$ </td><td> $0 . 9 9 1 5 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 3 1 5 \pm 0 . 0 0 1 3$ </td><td> $0 . 2 6 7 9 \pm 0 . 0 0 7 4$ </td><td> $0 . 2 3 4 2 \pm 0 . 0 0 7 2$ </td></tr><tr><td> $\mathrm { \Delta D B C A - L i g h t + G A P }$ </td><td> $0 . 9 6 1 5 \pm 0 . 0 0 1 1$ </td><td> $0 . 9 6 1 4 \pm 0 . 0 0 1 1$ </td><td> $0 . 9 9 1 1 \pm 0 . 0 0 0 6$ </td><td> $\mathbf { 0 . 0 2 4 7 \pm 0 . 0 0 1 5 }$ </td><td> $0 . 2 7 5 8 \pm 0 . 0 0 7 7$ </td><td> $0 . 3 7 5 8 \pm 0 . 0 0 9 1$ </td></tr><tr><td> $\mathrm { { D B C A - L i g h t } + M G A P }$ </td><td> $\mathbf { 0 . 9 6 6 4 \pm 0 . 0 0 1 1 }$ </td><td> $\mathbf { 0 . 9 6 6 4 \pm 0 . 0 0 1 1 }$ </td><td> $0 . 9 8 9 9 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 2 5 4 \pm 0 . 0 0 1 5$ </td><td> $0 . 4 4 8 2 \pm 0 . 0 1 0 1$ </td><td> $0 . 8 5 7 2 \pm 0 . 0 0 9 7$ </td></tr><tr><td> $\mathbf { D B C A - S e g N e t - M G A P }$ </td><td> $0 . 9 6 1 6 \pm 0 . 0 0 1 6$ </td><td> $0 . 9 6 1 5 \pm 0 . 0 0 1 5$ </td><td> $0 . 9 9 0 6 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 2 4 7 \pm 0 . 0 0 1 3$ </td><td> $\mathbf { 0 . 7 0 8 6 \pm 0 . 0 1 0 4 }$ </td><td> $\mathbf { 0 . 9 9 0 5 \pm 0 . 0 0 1 8 }$ </td></tr></table>

Values are mean ± sample SD across seeds 12, 42, and 112. Bold values denote the highest displayed mean point estimate and do not imply statistical significance.

In contrast, the complete configuration produced the strongest pulmonary containment among the displayed configurations. Its ALR of $0 . 7 0 8 6 \pm 0 . 0 1 0 4$ exceeded ResNet-50 $( 0 . 4 9 8 1 \pm 0 . 0 1 0 0 )$ DBCA-Light + MGAP $( 0 . 4 4 8 2 \pm$ 0.0101), and the remaining baselines. The contrast with Swin-Tiny is particularly informative: Swin-Tiny achieved slightly higher weighted F1 (0.9627 vs. 0.9615) and macro AUC (0.9934 vs. 0.9906), but ALR was 0.2415 rather than 0.7086. Thus, within the evaluated protocol, high diagnostic competence did not uniquely determine pulmonary-attribution behavior.

Table 5: Architecture-matched efect of replacing GAP with MGAP on the COVID-19 test set.
<table><tr><td>Representation</td><td>Metric</td><td>GAP</td><td>MGAP</td><td> $\Delta \ ( \mathrm { M G A P \mathrm { ~ - ~ } G A P } )$ </td></tr><tr><td rowspan="4">Late fusion</td><td>Weighted F1</td><td> $0 . 9 5 7 9 \pm 0 . 0 0 1 2$ </td><td> $0 . 9 6 0 9 \pm 0 . 0 0 1 4$ </td><td>+0.0029</td></tr><tr><td>Dice</td><td> $0 . 9 7 8 0 \pm 0 . 0 0 1 1$ </td><td> $0 . 9 7 6 4 \pm 0 . 0 0 1 2$ </td><td>-0.0016</td></tr><tr><td>ALR</td><td> $0 . 2 6 6 8 \pm 0 . 0 0 7 4$ </td><td> $\mathbf { 0 . 5 8 8 4 \pm 0 . 0 1 1 8 }$ </td><td>+0.3216</td></tr><tr><td>cALR@0.9</td><td> $0 . 3 6 8 5 \pm 0 . 0 1 2 2$ </td><td> $\mathbf { 0 . 9 1 1 6 \pm 0 . 0 0 7 9 }$ </td><td>+0.5431</td></tr><tr><td rowspan="4">Single bridge (BCAB1)</td><td>Weighted F1</td><td> $0 . 9 6 2 7 \pm 0 . 0 0 1 4$ </td><td> $0 . 9 6 5 0 \pm 0 . 0 0 1 2$ </td><td>+0.0023</td></tr><tr><td>Dice</td><td> $0 . 9 8 4 6 \pm 0 . 0 0 1 0$ </td><td> $0 . 9 7 9 3 \pm 0 . 0 0 1 2$ </td><td>-0.0053</td></tr><tr><td>ALR</td><td> $0 . 2 9 4 0 \pm 0 . 0 0 7 6$ </td><td> $\mathbf { 0 . 6 2 6 8 \pm 0 . 0 1 2 1 }$ </td><td>+0.3328</td></tr><tr><td>cALR@0.9</td><td> $0 . 5 2 4 4 \pm 0 . 0 1 0 4$ </td><td>0.9587 ± 0.0059</td><td>+0.4343</td></tr><tr><td rowspan="4">Dual bridge  $\mathrm { ( B C A B 1 + B C A B 2 ) }$ </td><td>Weighted F1</td><td> $0 . 9 6 1 8 \pm 0 . 0 0 1 5$ </td><td>0.9615 ± 0.0015</td><td>-0.0003</td></tr><tr><td>Dice</td><td> $0 . 9 8 4 9 \pm 0 . 0 0 1 0$ </td><td> $0 . 9 8 1 2 \pm 0 . 0 0 1 4$ </td><td>-0.0037</td></tr><tr><td>ALR</td><td> $0 . 3 8 7 8 \pm 0 . 0 0 9 8$ </td><td> $\mathbf { 0 . 7 0 8 6 \pm 0 . 0 1 0 4 }$ </td><td>+0.3209</td></tr><tr><td>cALR@0.9</td><td> $0 . 5 2 6 5 \pm 0 . 0 1 0 1$ </td><td> $\mathbf { 0 . 9 9 0 5 \pm 0 . 0 0 1 8 }$ </td><td>+0.4640</td></tr></table>

## 4.2. Controlled efect of mask-guided pooling

To isolate mask-guided aggregation, GAP and MGAP were compared within late-fusion, single-bridge, and dual-bridge representation conditions. Table 5 reports the direct GAP→MGAP efect within each matched representation.

The anatomical efect of MGAP was directionally consistent across all three representation conditions. Mean ALR increased by +0.3216 under late fusion, +0.3328 with BCAB1, and +0.3209 with the complete dual-bridge representation. cALR@0.9 increased by +0.5431, +0.4343, and +0.4640, respectively. By comparison, weighted F1 changed by only +0.0029, +0.0023, and −0.0003.

The dual-bridge comparison provides the clearest controlled result. Replacing GAP with MGAP increased ALR from $0 . 3 8 7 8 \pm 0 . 0 0 9 8$ to $0 . 7 0 8 6 \pm 0 . 0 1 0 4$ and cALR@0.9 from $0 . 5 2 6 5 \pm 0 . 0 1 0 1$ to $0 . 9 9 0 5 \pm 0 . 0 0 1 8$ , while weighted F1 remained essentially unchanged. The ALR increases were +0.3197, +0.3219, and +0.3210 for seeds 12, 42, and 112, respectively.

Segmentation and classifier attribution were also not equivalent. In the dual-bridge condition, GAP achieved slightly higher Dice than MGAP $( 0 . 9 8 4 9 \pm 0 . 0 0 1 0 \ \mathrm { v s . \ 0 . 9 8 1 2 \pm 0 . 0 0 1 4 } )$ while producing substantially lower ALR (0.3878 vs. 0.7086). High-quality lung-mask prediction therefore did not, by itself, imply that class-related attribution was concentrated within the lungs.

Representative attribution maps are shown in Fig. 7. Grad-CAM provides broad predicted-class attribution, Layer-CAM provides a finer intermediate layer view, and the overlap map separates pulmonary-contained from extrapulmonary attribution. These images complement the cohort-level ALR/cALR analysis but are not treated as lesion annotations or causal explanations.

Grad-CAM Attribution  
Original CXR  
![](images/dc17c303083dfd80c44187b29f11abc1e03ff92a026037570cd13458ea6ba9d9.jpg)  
Layer-CAM Attribution  
Lung-Overlap Map  
Figure 7: Representative pulmonary-attribution analysis of DBCA-SegNet-MGAP using Grad-CAM, Layer-CAM, and lung-overlap maps. Columns show the original CXR, predicted-class Grad-CAM attribution, Layer-CAM attribution, and the attribution–lung overlap map. Green indicates attribution within the pulmonary reference region and red indicates extrapulmonary attribution.

## 4.3. External Shenzhen-to-Montgomery zero-shot evaluation

External reliability was evaluated by training and selecting models exclusively on Shenzhen and applying the locked checkpoints to Montgomery without target-domain fine-tuning, recalibration, or threshold optimization.

For DBCA-SegNet-MGAP, Shenzhen weighted F1 was $0 . 8 8 9 8 \pm 0 . 0 0 3 4$ and ROC-AUC was $0 . 9 6 8 4 \pm 0 . 0 0 1 9$ . Under zero-shot transfer to Montgomery, weighted F1 decreased to $0 . 7 5 2 8 \pm 0 . 0 0 8 0$ , whereas ROC-AUC remained $0 . 9 0 8 0 \pm 0 . 0 0 4 3$ . ECE increased from $0 . 0 7 1 5 \pm 0 . 0 0 3 4$ to $0 . 1 6 8 3 \pm 0 . 0 0 5 5$ The mean source-to-target changes were therefore −0.1371 in weighted F1, −0.0604 in ROC-AUC, and +0.0968 in ECE. Montgomery weighted F1 ranged from 0.7449 to 0.7608 across the three runs, ROC-AUC from 0.9037 to 0.9123, and ECE from 0.1627 to 0.1736.

Table 6: Multi-seed Shenzhen internal and Montgomery zero-shot external performance and pulmonary-attribution comparison.
<table><tr><td>Configuration</td><td>Cohort</td><td> $\operatorname { F } 1 _ { w }$ </td><td>AUC</td><td>ALR</td><td>cALR@0.9</td></tr><tr><td rowspan="2">ResNet-50</td><td>Shenzhen</td><td> $0 . 7 8 8 5 \pm 0 . 0 0 3 3$ </td><td> $0 . 8 6 5 7 \pm 0 . 0 0 3 2$ </td><td></td><td></td></tr><tr><td>Montgomery</td><td> $0 . 4 4 9 6 \pm 0 . 0 0 6 8$ </td><td> $0 . 5 6 9 8 \pm 0 . 0 0 8 2$ </td><td> $0 . 4 2 9 0 \pm 0 . 0 0 7 4$ </td><td> $0 . 3 2 6 1 \pm 0 . 0 1 1 4$ </td></tr><tr><td rowspan="2">Swin-Tiny</td><td>Shenzhen</td><td> $0 . 8 4 9 3 \pm 0 . 0 0 3 3$ </td><td> $0 . 9 2 5 9 \pm 0 . 0 0 3 2$ </td><td></td><td></td></tr><tr><td>Montgomery</td><td> $0 . 7 3 4 5 \pm 0 . 0 0 6 7$ </td><td> $0 . 8 5 3 2 \pm 0 . 0 0 6 2$ </td><td> $0 . 2 4 9 8 \pm 0 . 0 0 6 8$ </td><td> $0 . 1 8 6 8 \pm 0 . 0 0 7 7$ </td></tr><tr><td>DenseNet-121</td><td>Shenzhen Montgomery</td><td> $0 . 8 7 9 7 \pm 0 . 0 0 3 2$   $0 . 7 2 9 9 \pm 0 . 0 0 6 3$ </td><td> $0 . 9 2 0 6 \pm 0 . 0 0 2 8$   $0 . 7 7 0 7 \pm 0 . 0 0 6 6$ </td><td> $0 . 4 3 7 9 \pm 0 . 0 0 7 3$ </td><td></td></tr><tr><td rowspan="2"> $\mathrm { \Delta D B C A - L i g h t + G A P }$ </td><td>Shenzhen</td><td></td><td></td><td></td><td> $0 . 4 2 4 4 \pm 0 . 0 0 8 3$ </td></tr><tr><td>Montgomery</td><td> $0 . 8 4 9 0 \pm 0 . 0 0 3 2$   $0 . 7 5 0 9 \pm 0 . 0 0 6 8$ </td><td> $0 . 9 3 0 0 \pm 0 . 0 0 3 0$   $0 . 8 6 0 1 \pm 0 . 0 0 5 9$ </td><td> $0 . 3 4 6 8 \pm 0 . 0 0 7 2$ </td><td></td></tr><tr><td rowspan="2"> $\mathrm { { D B C A - L i g h t } + M G A P }$ </td><td></td><td></td><td></td><td></td><td> $0 . 5 3 7 0 \pm 0 . 0 0 8 2$ </td></tr><tr><td>Shenzhen Montgomery</td><td> $0 . 8 7 9 1 \pm 0 . 0 0 3 3$   $0 . 5 4 4 0 \pm 0 . 0 0 6 9$ </td><td> $0 . 9 3 6 0 \pm 0 . 0 0 3 3$   $0 . 8 3 5 2 \pm 0 . 0 0 5 9$ </td><td></td><td></td></tr><tr><td rowspan="2"> $\mathbf { D B C A - S e g N e t - M G A P }$ </td><td></td><td></td><td></td><td> $0 . 6 2 9 3 \pm 0 . 0 0 7 9$ </td><td> $0 . 9 8 5 3 \pm 0 . 0 0 3 2$ </td></tr><tr><td>Shenzhen Montgomery</td><td> $\mathbf { 0 . 8 8 9 8 \pm 0 . 0 0 3 4 }$   $\mathbf { 0 . 7 5 2 8 \pm 0 . 0 0 8 0 }$ </td><td> $\mathbf { 0 . 9 6 8 4 \pm 0 . 0 0 1 9 }$   $\mathbf { 0 . 9 0 8 0 \pm 0 . 0 0 4 3 }$ </td><td> $\mathbf { 0 . 6 4 6 6 \pm 0 . 0 0 8 1 }$ </td><td> $\mathbf { 0 . 9 9 0 5 \pm 0 . 0 0 2 0 }$ </td></tr></table>

Values are mean ± sample SD across seeds 12, 42, and 112. Pulmonary-attribution metrics are shown for the external Montgomery cohort, which is the principal domain-shift attribution analysis.

The complete framework achieved the highest mean point estimates for source-domain F1 and AUC and for external AUC among the displayed configurations. On Montgomery, weighted F1 $( 0 . 7 5 2 8 \pm 0 . 0 0 8 0 )$ was close to DBCA-Light + GAP $( 0 . 7 5 0 9 \pm 0 . 0 0 6 8 )$ , while ROC-AUC was higher $( 0 . 9 0 8 0 \pm 0 . 0 0 4 3$ vs. $0 . 8 6 0 1 \pm 0 . 0 0 5 9 )$ . Pulmonary attribution containment remained high, with $\mathrm { A L R } = 0 . 6 4 6 6 \pm 0 . 0 0 8 1$ and $\mathrm { c A L R @ 0 . 9 } = 0 . 9 9 0 5 \pm 0 . 0 0 2 0$

The capacity-control pair provides an important counterexample. Replacing GAP with $\mathrm { M G A P }$ in DBCA-Light increased Montgomery ALR from $0 . 3 4 6 8 \pm 0 . 0 0 7 2$ to $0 . 6 2 9 3 \pm 0 . 0 0 7 9$ and $\mathrm { c A L R @ 0 . 9 }$ from $0 . 5 3 7 0 \pm 0 . 0 0 8 2$ to $0 . 9 8 5 3 \pm 0 . 0 0 3 2$ , yet weighted F1 decreased from $0 . 7 5 0 9 \pm 0 . 0 0 6 8$ to $0 . 5 4 4 0 \pm 0 . 0 0 6 9$ , and AUC decreased from $0 . 8 6 0 1 \pm 0 . 0 0 5 9$ to $0 . 8 3 5 2 \pm 0 . 0 0 5 9$ Strong pulmonary containment therefore did not guarantee robust external classification.

Figure 8 provides a qualitative view of probability-weighted competingclass attribution on Montgomery. Red corresponds to TB-dominant attribution and blue to Normal-dominant attribution in the signed-diference map. These maps are descriptive and are not interpreted as verified lesion localization or causal evidence.

![](images/b70b6be1f0fa966c44f447100392c7148d66519f4a133d82e03b19910d62215f.jpg)  
Figure 8: Representative probability-weighted competing-class attribution for Tuberculosis versus Normal under external Montgomery testing. Rows show a high-confidence Normal prediction, a lower-confidence Normal prediction, and a high-confidence Tuberculosis prediction. Columns show the input CXR, class probabilities, TB-weighted Grad-CAM, Normal-weighted Grad-CAM, attribution composite, and signed attribution diference.

Table 7: Model complexity and inference eficiency of the capacity-controlled DBCA configurations.
<table><tr><td>Configuration</td><td>Parameters (M)</td><td>Checkpoint (MB)</td><td>Throughput (img/s)</td></tr><tr><td>DBCA-Light + GAP</td><td>9.028</td><td>34.88</td><td>158.83</td></tr><tr><td>DBCA-Light + MGAP</td><td>9.028</td><td>34.88</td><td>89.98</td></tr><tr><td> $\mathrm { D B C A - S e g N e t + G A P }$ </td><td>68.821</td><td>262.96</td><td>58.60</td></tr><tr><td>DBCA-SegNet + MGAP</td><td>68.821</td><td>262.96</td><td>59.33</td></tr></table>

Throughput was measured using the common COVID-19 inference benchmark and is intended for within-study comparison.

## 4.4. Model complexity and inference eficiency

Model complexity and inference eficiency were evaluated to contextualize the capacity controls.

The lightweight configurations contained 9.028 million parameters, an approximately 86.9% reduction relative to the 68.821-million-parameter complete architecture, while checkpoint size decreased from 262.96 MB to 34.88 MB. Within the complete architecture, replacing GAP with MGAP did not increase parameter count or checkpoint size, and measured throughput remained similar (58.60 vs. 59.33 images/s). Thus, the increase in pulmonary attribution containment associated with MGAP was not attributable to increased parameter count.

The lightweight configurations show that substantially lower complexity is achievable, but Sections 4 and Table 6 show that reduced capacity did not reproduce the complete framework’s overall external reliability profile. Eficiency and reliability were therefore treated as separate properties.

## 5. Discussion

## 5.1. Diagnostic competence and pulmonary attribution are separable

The internal experiments directly answer RQ1: classifiers with similar diagnostic performance can exhibit substantially diferent pulmonary-attribution distributions. Swin-Tiny slightly exceeded DBCA-SegNet-MGAP in weighted F1 and macro AUC, yet its mean ALR was less than half that of the complete framework. Conversely, DBCA-Light + MGAP achieved the highest internal weighted F1 while retaining substantially lower ALR than DBCA-SegNet-MGAP. These diferences support the central premise that conventional diagnostic metrics do not determine where class-related attribution is spatially concentrated.

This result is consistent with the broader caution that high CXR performance can coexist with unintended shortcut behavior [4]. It also complements anatomy-guided classification studies [1, 5, 6]: the relevant question is not only whether anatomical information can be introduced, but whether its influence on the classifier can be isolated and measured. ALR and cALR provide one reproducible containment-oriented view of that question, while remaining deliberately narrower than lesion localization or explanation faithfulness.

## 5.2. MGAP as a controlled anatomical aggregation intervention

The matched GAP/MGAP experiments address RQ2. Across late-fusion, single-bridge, and dual-bridge representations, MGAP increased ALR by approximately 0.32 in all three conditions and produced large increases in cALR@0.9. The corresponding changes in weighted F1 were small and representation dependent. The most reproducible consequence of MGAP was therefore anatomical: it altered the spatial distribution of class-related attribution more consistently than it altered diagnostic performance.

The segmentation comparison further clarifies the mechanism. The dualbridge GAP model achieved slightly higher Dice than the MGAP counterpart but much lower ALR. Accurate prediction of pulmonary anatomy was therefore not equivalent to pulmonary concentration of the classifier’s attribution. Auxiliary segmentation can establish that a representation contains suficient information to reconstruct the lungs; it does not by itself establish that the final classifier uses that anatomy preferentially. Connecting the predicted soft mask directly to normalized aggregation makes this relationship operational rather than implicit.

The interpretation should nevertheless remain bounded. Because Grad-CAM for the proposed model is computed on the mask-influenced classifier representation, higher pulmonary containment is partly an expected consequence of the intervention being measured. The controlled GAP comparison reduces architectural confounding, but it does not convert Grad-CAM into a causal explanation. Future sensitivity analyses should therefore include matched pre-mask/pre-pooling attribution and perturbation-based faithfulness tests.

## 5.3. External domain shift reveals multidimensional reliability

RQ3 is addressed by the locked Shenzhen-to-Montgomery experiment. For DBCA-SegNet-MGAP, ROC-AUC decreased less than weighted F1, while ECE increased markedly. The source-domain probability scale and fixed threshold therefore transferred less successfully than ranking discrimination. This pattern is consistent with the broader observation that external radiologic performance often degrades under cohort shift [12] and with the distinction between discrimination and calibration emphasized in neural-network calibration research [13].

Pulmonary attribution containment showed yet another transfer pattern. The proposed model retained ALR of approximately 0.65 and cALR@0.9 near 0.99 on Montgomery even though threshold-dependent performance and calibration worsened. These results reinforce that external robustness is not a single scalar property. A model may preserve ranking ability, lose calibration, and retain anatomically concentrated attribution to diferent degrees.

## 5.4. Strong pulmonary containment is not suficient for robust transfer

The lightweight MGAP result is an important negative control. DBCA-Light + MGAP retained high Montgomery ALR and cALR@0.9 but its external weighted F1 fell sharply relative to DBCA-Light + GAP. Thus, stronger pulmonary containment alone did not guarantee that the model had learned disease features that generalized across cohorts.

This observation prevents overinterpretation of the anatomical metric. Whole-lung containment can reduce or expose extrapulmonary reliance, but the pulmonary field itself contains cohort-specific texture, acquisition efects, projection diferences, and non-pathological variation. A classifier can therefore be “inside the lungs” spatially while still relying on features that do not transfer diagnostically. Pulmonary containment should be interpreted as one reliability dimension rather than a surrogate for robustness or clinical validity.

The eficiency results add a practical trade-of. The lightweight configurations reduced parameter count by approximately 86.9%, but the full framework achieved a more favorable combination of external discrimination and pulmonary containment. Conversely, the larger architecture did not dominate every internal classification metric. No evaluated configuration simultaneously optimized diagnostic performance, calibration, anatomical containment, external transfer, and computational eficiency.

## 6. Conclusion and Future Work

This study evaluated CXR classification from a beyond-accuracy perspective by treating diagnostic performance, pulmonary attribution containment, calibration, and external robustness as related but distinct properties. Across three training seeds, $\mathrm { D B C A - S e g N e t - M G A P }$ achieved weighted F1 of $0 . 9 6 1 5 \pm 0 . 0 0 1 5$ and macro ROC-AUC of $0 . 9 9 0 6 \pm 0 . 0 0 0 7$ on the four-class COVID-19 task. More importantly, in the complete dual-BCAB representation, replacing GAP with MGAP increased ALR from $0 . 3 8 7 8 \pm 0 . 0 0 9 8$ to $0 . 7 0 8 6 \pm 0 . 0 1 0 4$ and cALR@0.9 from $0 . 5 2 6 5 \pm 0 . 0 1 0 1$ to $0 . 9 9 0 5 \pm 0 . 0 0 1 8$ while weighted F1 remained essentially unchanged. This separates accurate prediction of pulmonary anatomy from concentrating classifier attribution within that anatomy.

The locked Shenzhen-to-Montgomery experiment further showed that reliability properties do not transfer equally. On Montgomery, the proposed model retained ROC-AUC of $0 . 9 0 8 0 { \pm } 0 . 0 0 4 3$ and strong pulmonary attribution containment, whereas weighted F1 decreased to $0 . 7 5 2 8 \pm 0 . 0 0 8 0$ and ECE increased to $0 . 1 6 8 3 \pm 0 . 0 0 5 5$ . The lightweight MGAP configuration provided a complementary counterexample: strong pulmonary containment could coexist with substantial external diagnostic degradation. Pulmonary containment is therefore not suficient evidence of robust generalization.

Overall, diagnostic competence, pulmonary attribution containment, segmentation quality, calibration, and external transfer should be assessed separately. ALR and cALR remain containment metrics: they do not establish lesion localization, causal reasoning, or clinical correctness. The proposed framework should therefore be viewed as a controlled reliability-oriented research framework rather than evidence of clinical deployment readiness.

Future work should prioritize broader multi-center and prospective external evaluation, patient-level partitioning where identifiers are available, expert lesion annotations, and perturbation-based faithfulness tests that distinguish whole-lung containment from pathology-specific evidence. Independent calibration cohorts and clinically motivated operating-point selection are also needed. Finally, model compression, lower-cost cross-backbone interaction, and reliability-aware knowledge distillation should be investigated to reduce computational demand while preserving diagnostic performance, lung-mask quality, calibration, and pulmonary attribution behavior.

## Acknowledgement

During the preparation of this work, the authors used generative AI and AI-assisted technologies to improve grammar, readability, and formatting. After using these tools, the authors carefully reviewed and edited the content and take full responsibility for the contents of the published article.

## Data Availability

The COVID-19 Radiography Database, Shenzhen Tuberculosis Dataset, and Montgomery County Chest X-ray Dataset used in this study are publicly available from their respective cited repositories and original publications.

## Code Availability

All source code, training scripts, and evaluation notebooks are publicly available at https://github.com/Abdullah-229/BeyondAccuracy.

## Declaration of Competing Interest

The authors declare that there are no known conflicting interests or personal relationships which might have seemed to afect their research presented in this paper.

## References

[1] R. Zhang, F. Yang, Y. Luo, J. Liu, J. Li, C. Wang, Part-aware maskguided attention for thorax disease classification, Entropy 23 (6) (2021) 653. doi:10.3390/e23060653.

[2] X. Wu, Y. Feng, H. Xu, Z. Lin, T. Chen, S. Li, S. Qiu, Q. Liu, Y. Ma, S. Zhang, CTransCNN: Combining transformer and CNN in multilabel medical image classification, Knowledge-Based Systems 281 (2023) 111030. doi:10.1016/j.knosys.2023.111030.

[3] A. Saporta, X. Gui, A. Agrawal, A. Pareek, S. Q. H. Truong, C. D. T. Nguyen, V.-D. Ngo, J. Seekins, F. G. Blankenberg, A. Y. Ng, M. P. Lungren, P. Rajpurkar, Benchmarking saliency methods for chest X-ray interpretation, Nature Machine Intelligence 4 (2022) 867–878. doi:10.1038/s42256-022-00536-x.

[4] A. J. DeGrave, J. D. Janizek, S.-I. Lee, AI for radiographic COVID-19 detection selects shortcuts over signal, Nature Machine Intelligence 3 (2021) 610–619. doi:10.1038/s42256-021-00338-7.

[5] S. Shabina, S. Kalyani, AGFR-Net: Anatomy-guided feature refinement for robust multi-label thoracic disease classification from chest X-rays, Scientific ReportsOnline ahead of final issue assignment; published 16 July 2026 (2026). doi:10.1038/s41598-026-59302-3.

[6] Y. Wu, N. Japkowicz, S. Gilbert, R. Corizzo, Localization-aware chest X-ray classification via segmentation and gradient-based attention, Data Mining and Knowledge Discovery 40 (2026) 47. doi:10.1007/s10618-026- 01214-x.

[7] G. Shi, Z. Wang, Y. Shi, J. Pan, L. Sun, F. Fang, L. Jin, Graph guided multiscale cross attention for multilabel chest X ray classification, Scientific Reports 16 (2026) 22761. doi:10.1038/s41598-026-53115-0.

[8] H. K. Shin, D. H. Lee, H. Jung, W.-J. Nam, Refining explainability in chest X-ray diagnostics with lesion-aware hybrid transformer and local similarity of integrated re-normalized attention map, Biomedical Signal Processing and Control 119 (2026) 109766. doi:10.1016/j.bspc.2026.109766.

[9] X. Fu, R. Lin, W. Du, A. Tavares, Y. Liang, Explainable hybrid transformer for multi-classification of lung disease using chest X-rays, Scientific Reports 15 (2025) 6650. doi:10.1038/s41598-025-90607-x.

[10] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra, Grad-CAM: Visual explanations from deep networks via gradient-based localization, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626. doi:10.1109/ICCV.2017.74.

[11] P.-T. Jiang, C.-B. Zhang, Q. Hou, M.-M. Cheng, Y. Wei, LayerCAM: Exploring hierarchical class activation maps for localization, IEEE Transactions on Image Processing 30 (2021) 5875–5888. doi:10.1109/TIP.2021.3089943.

[12] A. C. Yu, B. Mohajer, J. Eng, External validation of deep learning algorithms for radiologic diagnosis: A systematic review, Radiology: Artificial Intelligence 4 (3) (2022) e210064. doi:10.1148/ryai.210064.

[13] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, On calibration of modern neural networks, in: Proceedings of the 34th International Conference on Machine Learning, Vol. 70 of Proceedings of Machine Learning Research, PMLR, 2017, pp. 1321–1330.

[14] A. S. Tejani, M. E. Klontzas, A. A. Gatti, J. T. Mongan, L. Moy, S. H. Park, C. E. J. Kahn, CLAIM 2024 Update Panel, Checklist for artificial intelligence in medical imaging (CLAIM): 2024 update, Radiology: Artificial Intelligence 6 (4) (2024) e240300. doi:10.1148/ryai.240300.

[15] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778. doi:10.1109/CVPR.2016.90.

[16] G. Huang, Z. Liu, L. Van Der Maaten, K. Q. Weinberger, Densely connected convolutional networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 4700–4708. doi:10.1109/CVPR.2017.243.

[17] M. Tan, Q. V. Le, EficientNet: Rethinking model scaling for convolutional neural networks, in: Proceedings of the 36th International

Conference on Machine Learning, Vol. 97 of Proceedings of Machine Learning Research, PMLR, 2019, pp. 6105–6114.

[18] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie, A ConvNet for the 2020s, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11976– 11986. doi:10.1109/CVPR52688.2022.01167.

[19] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10012–10022. doi:10.1109/ICCV48922.2021.00986.

[20] R. Priyanka, G. Gajendran, S. Boulaaras, S. S. Tantawy, PediaPulmoDx: Harnessing cutting edge preprocessing and explainable AI for pediatric chest X-ray classification with DenseNet121, Results in Engineering 25 (2025) 104320. doi:10.1016/j.rineng.2025.104320.

[21] P. Kaushik, E. Jain, V. Kukreja, S. Hariharan, M. Krishnamoorthy, V. Ahuja, A. Bhattacherjee, R. K. Kaushal, S.-Y. Chen, Modelling radiological features fusion and explainable AI in pneumonia detection: A graph-based deep learning and transformer approach, Results in Engineering 26 (2025) 105225. doi:10.1016/j.rineng.2025.105225.

[22] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wang, P.-X. Lu, G. Thoma, Two public chest X-ray datasets for computer-aided screening of pulmonary diseases, Quantitative Imaging in Medicine and Surgery 4 (6) (2014) 475–477. doi:10.3978/j.issn.2223-4292.2014.11.20.

[23] F. Pasa, V. Golkov, F. Pfeifer, D. Cremers, D. Pfeifer, Eficient deep network architectures for fast chest X-ray tuberculosis screening and visualization, Scientific Reports 9 (2019) 6268. doi:10.1038/s41598-019- 42557-4.

[24] D. Das, K. C. Santosh, U. Pal, Cross-population train/test deep learning model: Abnormality screening in chest X-rays, in: 2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS), 2020, pp. 514–519. doi:10.1109/CBMS49503.2020.00103.

[25] M. E. H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M. A. Kadir, Z. B. Mahbub, K. R. Islam, M. S. Khan, A. Iqbal, N. Al-Emadi, M. B. I. Reaz, T. I. Islam, Can AI help in screening viral and COVID-19 pneumonia?, IEEE Access 8 (2020) 132665–132676. doi:10.1109/ACCESS.2020.3010287.

[26] T. Rahman, A. Khandakar, Y. Qiblawey, A. Tahir, S. Kiranyaz, S. B. Abul Kashem, M. T. Islam, S. Al Maadeed, S. M. Zughaier, M. S. Khan, M. E. H. Chowdhury, Exploring the efect of image enhancement techniques on COVID-19 detection using chest Xray images, Computers in Biology and Medicine 132 (2021) 104319. doi:10.1016/j.compbiomed.2021.104319.

[27] A. Buslaev, A. P. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, A. A. Kalinin, Albumentations: Fast and flexible image augmentations, Information 11 (2) (2020) 125. doi:10.3390/info11020125.

[28] J. Hu, L. Shen, G. Sun, Squeeze-and-excitation networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132–7141. doi:10.1109/CVPR.2018.00745.

[29] O. Ronneberger, P. Fischer, T. Brox, U-Net: Convolutional networks for biomedical image segmentation, in: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, Springer, 2015, pp. 234–241. doi:10.1007/978-3-319-24574-4\_28.

[30] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, international Conference on Learning Representations (ICLR) (2019).

[31] K. Wu, J. Zhang, H. Peng, M. Liu, B. Xiao, J. Fu, L. Yuan, TinyViT: Fast pretraining distillation for small vision transformers, in: Computer Vision – ECCV 2022, Springer, 2022, pp. 68–85. doi:10.1007/978-3-031- 20083-0\_5.

[32] A. Reinke, et al., Understanding metric-related pitfalls in image analysis validation, Nature Methods 21 (2024) 182–194. doi:10.1038/s41592-023- 02150-0.