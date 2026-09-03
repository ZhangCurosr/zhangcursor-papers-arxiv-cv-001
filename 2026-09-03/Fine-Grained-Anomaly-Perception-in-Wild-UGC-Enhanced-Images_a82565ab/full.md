# Fine-Grained Anomaly Perception in Wild UGC-Enhanced Images: A Comprehensive Dataset and Diference-Fusion Framework

Yan Zhong<sup>1,2∗</sup>, Gefei Chen<sup>1,2</sup>, Qiufang Ma<sup>2</sup>, Zhen Wang<sup>2</sup>, Zhiwei Fan<sup>2</sup>, Lei Shi<sup>3</sup>, Tingting Jiang<sup>2†</sup>

<sup>1</sup> Peking University, Beijing, China

<sup>2</sup> Douyin, ByteDance Inc.

<sup>3</sup> Communication University of China, Beijing, China

{zhongyan@stu.,chengefei@stu.,ttjiang@}pku.edu.cn; {maqiufang.cv,wangzhen3560,fanzhiwei.rice}@bytedance.com

## Abstract

Image enhancement and restoration have become standard back-end operations on short-video and social media platforms to boost UGC visual experience. Yet these processes inevitably introduce visual anomalies—especially in faces, texts, and textures—that directly undermine perceptual fidelity and viewer trust. While existing IQA methods perform well on classic distortions, they target holistic quality assessment and fail to capture the specific, localized anomalies caused by enhancement algorithms in real-world UGC. To bridge this gap, we formally define a new task—quality Anomaly Perception for UGC image Enhancement (UEAP), and contribute the first UEAP benchmark dataset, named UEAP-4k, curated from the real business scenarios. It provides fine-grained annotations for anomaly categories, localization and severity levels. Furthermore, we propose a Diference-Fusion Anomaly Perception Method (DFAP-UGC) for wild UGC-enhanced images, which leverages explicit problemreference diference fusion with dense spatial querying, regional verification, and quality-aware ranking, enabling robust anomaly identification in challenging scenarios. To handle the inherent coupling of subtasks in this new task, we propose a Locality-Aware Dynamic Task Prioritization (LADTP) training strategy that enables efective end-to-end learning and eliminates multi-stage overhead. Extensive experiments show that our method outperforms baselines adapted from classical approaches for this task, validating the value of this dataset and the superior of DFAP-UGC for robust UGC-enhanced image anomaly perception. Code and data will be public.

## 1 Introduction

With the rapid growth of short-video and social media platforms, user-generated content (UGC) has become a primary carrier of visual information (Shao, Wang, and Hao 2019; Zhong et al. 2025c). To enhance visual experiences, platforms widely adopt quality enhancement and compression algorithms to automatically process UGC, with generative enhancement techniques further improving image clarity and color fidelity (Safonov et al. 2025). However, such processing inevitably introduces local visual anomalies—such as content distortion, unintended recovery of private information, color artifacts, and texture irregularities, which severely undermine user experience and pose risks of complaints and brand damage (Kushwaha et al. 2022). Therefore, there is an urgent need for an anomaly perception approach tailored to UGC enhancement scenarios, capable of accurately identifying and localizing quality anomalies, thereby providing critical support for content quality assurance and algorithmic optimization.

<table><tr><td colspan="2">Visual Quality Detection Tasks</td><td>NR-IQA</td><td>FR-IQA</td><td>VAD</td><td>UEAP</td></tr><tr><td rowspan="3">Input</td><td>Ideal Ref. Image</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>Defective Ref. Image</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Test Image</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td rowspan="3"></td><td>Localization</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Output Classification</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Regression (Score)</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td rowspan="2">Aim</td><td>Perceptual Distortion</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Semantic Anomaly</td><td>x</td><td>x</td><td>√</td><td>√</td></tr></table>

Table 1: Comparison of the defined new task UEAP with other related visual quality detection tasks. NR-IQA and FR-IQA denote no-reference and full-reference image quality assessment for real-world distortion scenarios, respectively, while VAD stands for visual anomaly detection. Localization, Classification, and Regression in the output correspond to anomaly/distortion localization, anomaly/distortion categorization, and scoring of anomaly/distortion severity. The symbol ✓ indicates Yes and ✗ indicates No. These labels are defined for the majority of scenarios of each task. The proposed UEAP task requires no high-quality references but imposes stricter output criteria.

For anomaly perception in UGC-enhanced images, currently there are two lines of related work. First, Image Quality Assessment (IQA) is an important tool for evaluating image quality, including the quality of enhanced images. According to the availability of reference images, IQA methods are categorized as Full-Reference IQA (FR-IQA), Reduced-Reference(RR-IQA), No-Reference(NR-IQA) (Yang et al. 2023). For evaluating the quality of enhanced images, since usually there are no high quality reference images, many previous work adopt NR-IQA approach (Cao et al. 2025; Zhang et al. 2024; Chahine et al. 2025; Wang et al. 2026) Second, visual anomaly detection (VAD) (Cao et al. 2024; Batzner, Heckler, and König 2024) has also been widely explored: some VAD methods localize deviations by reconstruction errors (Bergmann et al. 2019; Zavrtanik, Kristan, and Skočaj 2021), feature distribution and memory-bank methods compare test patches with normal feature statistics (Defard et al. 2021; Roth et al. 2022), and transformer or distillation-based methods improve anomaly representation and localization (Deng and Li 2022; Liu et al. 2024).

However, neither IQA nor VAD can adequately perceive local anomalies in real-world UGC image enhancement scenarios. Specifically, (1)IQA methods only output global quality scores without fine-grained localization, classification and quality rating of abnormal regions,such as the artifacts of text, texture and human face caused by enhancement algorithms. The FR setting requires the high quality reference image which is often missing in UGC scenarios.The NR setting ignores the pre-enhancement images which can provide important information for anomaly perception in UGC image enhancement. (2) VAD methods localize deviations in the single test image, but they do not compare a postenhancement image with a non-ideal pre-enhancement reference, making it dificult to distinguish newly introduced artifacts from original UGC defects or benign pairwise changes.

To fill this gap, this paper proposes a new image anomaly perception task, termed the fine-grained Anomaly Perception for image Enhancement in UGC scenarios (UEAP). As shown in Table 1, this task difers from traditional IQA methods (Mao et al. 2025) that emphasize global quality scoring, and from generic visual anomaly detection (Liu et al. 2024) tasks that identify abnormalities based solely on single-image distributions. Instead, our task targets pairwise local anomaly instance detection for generative UGC image enhancement: using the pre- and post-enhancement image pair as dual inputs, it accurately detects local unexpected bad cases introduced during enhancement, outputting bounding boxes, anomaly categories, and severity scores. It specifically focuses on high-risk local bad cases, such as text tampering, texture distortion, and abnormal portrait efect. Therefore, solving this task requires overcoming three key Challenges: (i) the lack of a real-world paired UGC enhancement benchmark with fine-grained and consistent annotations for local failure cases in short video scenarios; (ii) since the reference image is the pre-enhancement image rather than an ideal image, local anomalies are subtle and easily confounded with semantic variations, lacking clear discriminative cues. (iii) jointly localizing, classifying, and grading anomalies is difficult due to their heterogeneous output spaces and the risk of inter-task interference.

To address Challenge (i), we constructs the first dataset for this new task, drawn from public sources on video contents, termed the UGC-Enhanced Anomaly Perception (UEAP-4k) Dataset. Each sample consists of a pre-enhancement reference image, the corresponding post-enhancement image to be examined, and human-annotated labels for the latter, covering three aspects: anomaly localization, anomaly category, and severity score. To ensure more consistent and reliable annotations, we employ multiple annotators and propose a greedy merging strategy to reconcile their labels. The final dataset comprises over 4,000 samples, serving as the first benchmark for this task.

Based on UEAP-4k, we propose a Diference-Fusion Anomaly Perception method (named DFAP-UGC) for wild UGC-enhanced images, aiming to achieve robust anomaly identification in complex real-world scenarios.

To address Challenge (ii), DFAP-UGC is designed around explicit pairwise comparison. As shown in Fig. 3, it jointly uses the problem feature, the reference feature, and their diference, allowing the detector to focus on enhancementinduced changes rather than defects already present in the original UGC image. It further verifies candidate regions with local paired evidence and calibrates their ranking quality, providing compact yet discriminative cues for subtle text, texture, and portrait artifacts in complex UGC scenarios.

To address Challenge (iii), considering the anomaly localization sub-task is the most challenging, we propose a Locality-Aware Dynamic Task Prioritization (LADTP) approach during training DFAP-UGC. Its core idea is to leverage Generalized Intersection over Union (GIoU) Loss(Rezatofighi et al. 2019) as a real-time measure of localization quality, and jointly model it with the dynamical dificulty levels of sub-tasks to explicitly characterize task dependencies and adaptively adjust weights. Specifically, when localization performance is poor (high GIoU loss), the weights of classification and anomaly scoring losses are suppressed to focus the model on bounding box regression. As localization performance improves (low GIoU loss), other sub-tasks receive varying degrees of attention according to their respective dificulty levels, ultimately achieving balanced multi-task optimization. Additionally, LADTP updates weights with exponential moving average smoothing to ensure stable and robust training.

The main contributions are summarized below.

• We formulate UEAP, a new fine-grained anomaly perception task for UGC image enhancement, and build UEAP-4k, the first benchmark dataset constructed from realworld short-video platforms. UEAP-4k contains paired pre-/post-enhancement images and unified annotations of ocalizing, classifying, and grading anomalies. Extensive experiments validating the value of this task and dataset.

• Tailored for this new task, we propose the first framework DFAP-UGC, which explicitly fuses problem-reference diference cues and combines dense spatial queries, regional verification, and quality-aware ranking to reliably identify enhancement-induced local anomalies.

• A Locality-Aware Dynamic Task Prioritization (LADTP) strategy is proposed to adaptively adjust sub-task weights based on localization quality and task dificulty, enabling balanced multi-task optimization during training.

## 2 Related Works

## Image Quality Assessment

Image quality assessment (IQA) predicts perceptual image quality and is studied in full-, reduced-, and no-reference settings (Zhai and Min 2020; Mao et al. 2025). Full-reference methods compare a distorted image with a high-quality reference (Lan et al. 2025; Liao et al. 2023), while no-reference methods are more practical for in-the-wild images where pristine references are unavailable (Zhong et al. 2025a; Su et al. 2020; Feng, Li, and Hao 2021; Zhong et al. 2025b; Sun et al. 2023; Zhao et al. 2023; Zhong et al. 2024). Recent studies further exploit contrastive learning, semisupervised representation learning, and vision-language or multimodal priors to improve generalization (Saha, Mishra, and Bovik 2023; Prabhakaran and Swamy 2023; Wang, Chan, and Loy 2023; Zhang et al. 2023; Radford et al. 2021; Chen et al. 2024). Several eforts have specifically targeted enhanced images: a blind IQA model was learned from big data (Gu et al. 2018), a debiased subjective assessment protocol was proposed for real-world enhancement (Cao, Wang, and Ma 2021), and more recently, a multi-scale feature fusion BIQA method with a large-scale natural image enhancement database (NIED) was introduced (Cao et al. 2025). Meanwhile, AU-IQA (Wang et al. 2025) provides a dedicated benchmark for AI-enhanced UGC, yet its annotations remain holistic quality scores, lacking instance-level anomaly localization and severity labels. This gap further motivates our UEAP task. Overall, existing IQA methods still produce holistic quality scores and typically assume either no reference or an ideal reference. In contrast, UEAP takes a nonideal pre-enhancement UGC image as reference and requires localizing enhancement-induced anomalies, classifying their types, and estimating severity—demanding explicit pairwise anomaly modeling.

![](images/e4202091b813da3d17f14598aa15d60872ccb1e2001bc45e3e33d1593ecbcd9a.jpg)  
Figure 1: Annotation interface and example for UEAP-4k. Annotators compare the reference and enhanced image side by side, draw bounding boxes on the enhanced image, select one anomaly type, and assign severity score from −5 to −1.

## Visual Anomaly Detection

Visual anomaly detection (VAD) identifies deviations from normal visual distributions and has been widely studied for industrial inspection (Liu et al. 2024; Bergmann et al. 2019; Zou et al. 2022). Representative methods include reconstruction-based detection (Zavrtanik, Kristan, and Skočaj 2021; You et al. 2022), patch-level distribution modeling and memory retrieval with pretrained features (Defard et al. 2021; Roth et al. 2022), and knowledge distillation based on teacher-student feature discrepancies (Deng and Li 2022). While efective in one-class or unsupervised settings (Pang et al. 2021), these methods judge a single image against normality to output anomaly scores or maps. UEAP instead compares a pre-enhancement UGC image with its enhanced counterpart and requires localization, category prediction, and severity estimation for enhancement-induced local failures. To address this setting, we contribute UEAP-4k, the first real-world benchmark, and propose DFAP-UGC, a tailored framework for robust pairwise anomaly perception in UGC enhancement scenarios.

![](images/2e3f6a7db64c9f1f66934d3be352c62396228096ae5c980ae831b480c5ea933b.jpg)  
Figure 2: Statistics of UEAP-4k after label unification. The left plot shows the histogram of anomaly box counts per sample, and the right plot shows the cumulative distribution.

## 3 Task Definition and Datasets

## UEAP Task

This section first formalizes the UEAP task and then describes the construction of the UEAP-4k benchmark.

Given a pre-enhancement UGC image I<sup>r</sup> and its enhanced counterpart I<sup>e</sup>, UGC image Enhancement Anomaly Perception (UEAP) aims to identify local quality failures newly introduced by the enhancement process. Diferent from fullreference IQA, whose reference is usually an ideal image and whose output is a global quality score, UEAP uses a non-ideal original UGC image as reference and focuses on instance-level abnormal changes. Formally, for each image pair $( I ^ { r } , I ^ { e } )$ , the task outputs a set of anomaly instances $\mathcal { Y } = \{ ( \bar { b _ { i } } , c _ { i } , s _ { i } ) \} _ { i = 1 } ^ { N } ,$ , where $b _ { i }$ denotes the bounding box of an anomalous region in $I ^ { e } , c _ { i }$ is its category, and $s _ { i } \in [ - 5 , - 1 ]$ is the severity score. Following the annotation protocol, anomalies are grouped into three practical categories: portrait efect anomaly, text efect anomaly, and texture efect anomaly. The severity score measures how strongly the target image becomes worse than the reference, ranging from barely perceptible degradation to severe quality collapse or semantic corruption. The smaller the score is, the more severe the abnormality is. This formulation directly supports real UGC quality inspection by jointly requiring localization, categorization, and severity estimation.

## UEAP-4k Dataset

Data Source. UEAP-4k is constructed from the public sources. We randomly sampled a subset of samples from these datasets and constructed the original reference images for the data used in this paper by randomly extracting frames. We then simulated a short-video platform pipeline, where the back-end system performs compression, encoding, decoding, and enhancement to reduce storage and bandwidth costs while improving playback quality, to obtain the enhanced images after processing. Although these operations are beneficial in most cases, they may also introduce subtle diferences between the original uploaded content and the final displayed content, especially in human faces, text regions, and high-frequency textures. Such changes are often local and easily overlooked by global quality scores, but they directly afect user experience when they cause facial deformation, text corruption, texture hallucination, or semantic inconsistency. We therefore sample frames from the back-end video processing pipeline and collect paired images, where each pair consists of a pre-enhancement reference image and its corresponding post-enhancement target image. The goal is to annotate local bad cases that are introduced or amplified by the enhancement process.

![](images/82d878beaa077cf6e7f9e65a899f583185937763fcebf17e165fbaf25025de51.jpg)  
Figure 3: Overview of our proposed Diference-Fusion Anomaly Perception Method (DFAP-UGC).

Annotation Protocol. Following the task definition, each sample is annotated by three professional annotators using the same web-based annotation interface, as shown in Fig. 1. The reference image and enhanced target image are displayed side by side, and annotators are instructed to mark only regions in the target image that become worse than the reference. To reduce missed cases and make the criterion consistent, annotation is conducted in three rounds, focusing on portrait efect anomalies, text efect anomalies, and texture efect anomalies. For each abnormal region, the annotator draws a bounding box, selects the anomaly category, and assigns a severity score in [−5, −1]. A score of −1 indicates a barely perceptible artifact or minor semantic change, while −5 indicates severe visual degradation or semantic corruption. Before formal labeling, all annotators are trained with the same guideline and example cases, and the annotation is performed on desktop displays under a consistent interface layout. During labeling, annotators are required to inspect the paired images independently, avoid marking regions whose diferences are caused only by normal enhancement, and ensure that each box tightly covers the afected region.

Label Unification. Since diferent annotators may draw slightly diferent boxes or assign diferent scores to the same abnormal region, we design a greedy label unification strategy to merge multiple annotations into a single target label set. Firstly, the raw records are first grouped by sample identity. Then, for each sample, annotations are grouped by anomaly category, and boxes within the same category are greedily clustered according to an IoU threshold τ = 0.5. Matched boxes are merged by averaging their coordinates, severity scores are averaged, and the final textual label is determined by majority voting. Unmatched boxes are retained as individual instances, which helps preserve rare but valid local failures. The proposed greedy label unification strategy is summarized in Algorithm 1.

Dataset Statistics. After label unification, UEAP-4k contains 4,222 paired samples and 13,662 anomaly boxes from 10,470 valid raw annotation records. The dataset covers three anomaly categories: portrait efect anomaly, texture efect anomaly, and text efect anomaly, with 7,921, 3,153, and 2,588 boxes, respectively. Portrait-related artifacts account for 58.0% of all instances, indicating that face and body regions are particularly sensitive to enhancement failures. Texture and text anomalies account for 23.1% and 18.9%, respectively, reflecting complementary risks in high-frequency details and semantic symbols. Fig. 2 further shows the distribution of box counts per sample. Each sample contains 3.24 boxes on average, with a median of 3, a standard deviation of 2.48, and a range from 1 to 33. The cumulative distribution shows that 75% of samples contain no more than 4 boxes and 90% contain no more than 6 boxes, while a small number of hard cases contain many local artifacts. This long-tailed distribution makes UEAP-4k realistic and challenging for robust local anomaly perception.

```latex
Algorithm 1: Greedy label unification for UEAP-4k
Require: Raw annotation records A, IoU threshold τ
Ensure: Unified dataset D
1: for each sample group G do
2: Collect valid boxes, categories, and severity scores
from all annotators.
3: for each anomaly category c do
4: Put all boxes of category c into queue $\textit { Q } _ { c } .$
5: while $Q _ { c }$ is not empty do
6: Pop an anchor annotation a from $\textit { Q } _ { c } .$
7: Find matches $\mathcal { M } = \{ b \in Q _ { c } \ | \ \mathrm { I o U } \bar { ( a , b ) } > \tau \}$
8: Remove M from $Q _ { c } .$
9: Average box coordinates and severity scores of
$\{ a \} \cup { \bar { \mathcal { M } } } .$
10: Assign category c and the majority textual label.
11: end while
12: end for
13: Add the merged annotations of G to D.
14: end for
15: return D
```

## 4 Methodology

## Overview

Figure 3 shows our proposed Diference-Fusion Anomaly Perception Method (DFAP-UGC) for paired UGC enhancement anomaly perception. Given a pre-enhancement reference image $\bar { I ^ { r } }$ and a post-enhancement problem image I<sup>p</sup>, DFAP-UGC detects local enhancement-induced failures on the problem image and predicts their category, bounding box, and severity score. The method explicitly presents the model with problem features, reference features, and their absolute diference, preserving a 28 × 28 detection grid. DFAP-UGC follows three coupled paths. The main detection path extracts aligned features with a shared Swin-T backbone (Liu et al. 2021), constructs explicit Stage-3 diference-fusion tokens, and uses a dense Transformer encoder (Vaswani et al. 2017; Carion et al. 2020) to produce spatial detection queries. The regional path samples high-resolution Stage-2 paired features inside each predicted box to verify local evidence. The quality path estimates class-conditional localization quality from detached query and box features, which calibrates inference ranking. During training, we propose the Locality-Aware Dynamic Task Prioritization (LADTP), which can dynamically adjust the loss weights according to localization quality and task dificulty, achieving balanced multi-task optimization and robust anomaly perception.

## Shared Swin Encoder and Diference Fusion

Both images are resized to 448 × 448 and processed by the same Swin-T encoder (Liu et al. 2021). Let F<sup>p</sup> and $\mathbf { F } _ { s } ^ { r }$ denote the projected enhanced and reference image features at Stage s. Under the input resolution, Stage 2 has a $5 6 \times 5 6$ grid and Stage 3 has a $2 8 \times 2 8$ grid. DFAP-UGC uses Stage 3 and Stage 2 for the main detection path and the regional verification path, respectively.

For the main path, the model builds a triplet diference representation at Stage 3:

$$
\mathbf { D } _ { 3 } = | \mathbf { F } _ { 3 } ^ { p } - \mathbf { F } _ { 3 } ^ { r } | ,\tag{1}
$$

$$
\begin{array} { r } { \mathbf { Z } = \mathrm { L N } \left( \mathrm { G E L U } \left( W _ { f } [ \mathbf { F } _ { 3 } ^ { p } ; \mathbf { F } _ { 3 } ^ { r } ; \mathbf { D } _ { 3 } ] \right) \right) , } \end{array}\tag{2}
$$

where $[ \cdot ; \cdot ]$ denotes channel concatenation. This representation keeps the current visual content, the reference-side context, and the explicit local diference. The fused feature is then projected to the Transformer hidden dimension, producing $2 8 \times 2 8 = 7 8 4$ dense tokens.

## Dense Transformer Sequence Module

The fused dense tokens are combined with 2D Sine positional encodings (Carion et al. 2020) and fed into a six-layer Transformer encoder:

$$
\mathbf { H } = T _ { e n c } ( \mathbf { Z } + \mathbf { P } ) , \quad \mathbf { H } = \{ \mathbf { h } _ { i } \} _ { i = 1 } ^ { 7 8 4 } .\tag{3}
$$

Each output token $\mathbf { h } _ { i }$ corresponds to a fixed grid position and is treated as a dense anchor query. For the query located at row u and column v, its initial anchor center is $( ( v +$ $0 . 5 ) / 2 8 , ( u + 0 . 5 ) / 2 8 )$ , with default width and height set to 0.15. The box head predicts an anchor-relative delta:

$$
\hat { \mathbf { b } } _ { i } = \sigma ( \mathrm { l o g i t } ( \mathbf { a } _ { i } ) + \mathrm { M L P } _ { b } ( \mathbf { h } _ { i } ) ) ,\tag{4}
$$

where ${ \bf a } _ { i }$ is the fixed anchor. The category and severity predictions are

$$
\hat { \mathbf { p } } _ { i } = W _ { c } \mathbf { h } _ { i } , \quad \hat { s } _ { i } = - 5 \cdot \sigma ( W _ { s } \mathbf { h } _ { i } ) .\tag{5}
$$

Here $\hat { { \bf p } } _ { i }$ contains class logits for the three foreground anomaly types, $\hat { \mathbf { b } } _ { i }$ is the normalized box in $( c _ { x } , c _ { y } , w , h )$ format, and $\hat { s } _ { i } \in [ - 5 , 0 ]$ is the predicted severity score. The severity score is an output attribute and is not used as the detection confidence.

## Task-Aligned Assignment and Detection Losses

Because DFAP-UGC uses 784 dense queries, one-to-one Hungarian matching is too sparse for training the neighborhood around each local anomaly. We therefore use dense task-aligned assignment (TAL) (Feng et al. 2021). For query i and ground-truth box j, the alignment score is

$$
A _ { i j } = p _ { i } ( y _ { j } ) ^ { \alpha } \cdot \mathrm { I o U } ( \hat { \mathbf { b } } _ { i } , \mathbf { b } _ { j } ) ^ { \beta } ,\tag{6}
$$

where $p _ { i } ( y _ { j } )$ is the predicted probability of the ground-truth category. We use $\alpha = 1 , \beta = 6$ , and select the top-k in-box queries for each ground-truth instance with $k = 5 ,$ . If no query center lies inside a ground-truth box, the query closest to the box center is selected. This assignment produces multiple positives per anomaly and aligns classification confidence with localization quality.

The main classification loss is a dense focal-style loss (Lin et al. 2017) over the TAL-selected positives and dense background queries:

$$
\begin{array} { r l } & { \mathcal { L } _ { c l s } = \frac { 1 } { \sum _ { i , c } y _ { i c } } \sum _ { i , c } w _ { i c } \mathrm { B C E } ( p _ { i c } , y _ { i c } ) , } \\ & { w _ { i c } = y _ { i c } + ( 1 - \alpha _ { f } ) p _ { i c } ^ { \gamma } \mathbf { 1 } [ y _ { i c } = 0 ] . } \end{array}\tag{7}
$$

Here $p _ { i c } = \sigma ( \hat { p } _ { i c } ) , y _ { i c }$ denotes the TAL target, i.e., the IoU score for selected positive query-class pairs and 0 otherwise, and $\gamma = 2$ . Localization and severity are optimized by

$$
\mathcal { L } _ { b b o x } = \frac { 1 } { N } \sum _ { i } \Vert \hat { \mathbf { b } } _ { i } - \mathbf { b } _ { i } \Vert _ { 1 } , \mathcal { L } _ { g i o u } = \frac { 1 } { N } \sum _ { i } ( 1 - \mathrm { G I o U } ( \hat { \mathbf { b } } _ { i } , \mathbf { b } _ { i } ) ) ,\tag{8}
$$

$$
\mathcal { L } _ { s c o r e } = \frac { 1 } { N } \sum _ { i } | \hat { s } _ { i } - s _ { i } | .\tag{9}
$$

During LADTP training, we use category-aware weights for localization-related positives to reduce the dominance of easier categories and emphasize harder texture and text failures.

## Regional Verification Branch

Dense queries improve recall but can also create highscoring false positives. The regional verification branch checks whether each predicted box contains local paired evidence for the predicted anomaly category. It uses the higherresolution Stage-2 features. After projecting the problem and reference Stage-2 features to 64 channels, we construct

$$
{ \bf R } _ { 2 } = [ { \bf F } _ { 2 } ^ { p } ; { \bf F } _ { 2 } ^ { r } ; | { \bf F } _ { 2 } ^ { p } - { \bf F } _ { 2 } ^ { r } | ] .\tag{10}
$$

For each predicted box $\hat { \mathbf { b } } _ { i } , \mathrm { ~ a ~ 3 ~ } \times \mathrm { ~ 3 ~ }$ Region of Interest (RoI) feature is sampled from ${ \bf R } _ { 2 }$ by bilinear grid sampling. The regional classifier takes the dense query feature, the pooled regional feature, and the predicted box coordinates as input:

$$
\hat { \mathbf { r } } _ { i } = R ( [ \mathbf { h } _ { i } ; \mathrm { P o o l } ( \mathbf { R } _ { 2 } , \hat { \mathbf { b } } _ { i } ) ; \mathrm { s g } ( \hat { \mathbf { b } } _ { i } ) ] ) .\tag{11}
$$

Output $\hat { \mathbf { r } } _ { i }$ is class-conditional regional logits. $\operatorname { s g } ( \cdot )$ denotes stop-gradient. The branch is trained end-to-end with respect to the shared feature representation, while the sampling coordinates are detached so that the regional loss does not directly update the box head through the sampling operation. Hard negative mining is used in this branch to strengthen suppression of visually salient but non-anomalous regions. The regional branch is optimized by a class-conditional binary cross-entropy loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { r e g } = \frac { 1 } { N _ { r e g } ^ { + } } \sum _ { i , c } \omega _ { i c } ^ { r e g } \operatorname { B C E } ( \boldsymbol { \sigma } ( \hat { \boldsymbol { r } } _ { i c } ) , y _ { i c } ^ { r e g } ) , } \\ & { y _ { i c } ^ { r e g } = m _ { i c } ^ { r e g } \mathbf { 1 } [ m _ { i c } ^ { r e g } \geq \tau _ { r e g } ] , m _ { i c } ^ { r e g } = \underset { \mathbf { b } \in \mathcal { B } _ { c } } { \operatorname* { m a x } } \operatorname { I o U } ( \hat { \mathbf { b } } _ { i } , \mathbf { b } ) . } \end{array}\tag{12}
$$

Here $B _ { c }$ is ground-truth boxes of class $c , \tau _ { r e g } = 0 . 5 .$ , and $\omega _ { i c } ^ { r e g }$ includes category weights and hard-negative weights.

## Dense-IoU Quality Branch and Inference Ranking

The dense-IoU quality branch estimates localization reliability, which is diferent from severity prediction. It takes detached dense query features and detached predicted boxes:

$$
\hat { \mathbf { q } } _ { i } = Q ( [ \mathrm { s g } ( \mathbf { h } _ { i } ) ; \mathrm { s g } ( \hat { \mathbf { b } } _ { i } ) ] ) .\tag{13}
$$

For each class, the training target is the maximum IoU between the predicted box and ground-truth boxes of that class when the IoU is above $0 . 3 ;$ otherwise, the target is zero. This design lets the quality head learn to calibrate candidate ranking without directly reshaping the main localization head. The dense-IoU quality branch is optimized by

$$
\begin{array} { r l } & { \mathcal { L } _ { q u a l } = \frac { 1 } { N _ { q u a l } ^ { + } } \sum _ { i , c } \omega _ { i c } ^ { q u a l } \mathrm { B C E } ( \boldsymbol { \sigma } ( \boldsymbol { \hat { q } } _ { i c } ) , y _ { i c } ^ { q u a l } ) , } \\ & { y _ { i c } ^ { q u a l } = m _ { i c } ^ { q u a l } \mathbf { 1 } [ m _ { i c } ^ { q u a l } \geq \tau _ { q u a l } ] , m _ { i c } ^ { q u a l } = \underset { \mathbf { b } \in \mathcal { B } _ { c } } { \operatorname* { m a x } } \mathrm { I o U } ( \hat { \mathbf { b } } _ { i } , \mathbf { b } ) . } \end{array}\tag{14}
$$

Algorithm 2: LADTP training for DFAP-UGC   
Require: Training set D, base weights $\lambda _ { t } ^ { 0 } .$ , smoothing factor   
$\rho ,$ exponent r   
Ensure: Trained DFAP-UGC model   
1: Initialize model parameters and task KPI $K _ { t } .$   
2: for each epoch e do   
3: Clear the epoch loss bufer.   
4: for each mini-batch $( I ^ { r } , I ^ { p } , \mathcal { V } )$ do   
5: Predict boxes, categories, severity scores, regional   
logits, and quality logits.   
6: Assign dense positives by task-aligned assignment.   
7: Compute $\mathcal { L } _ { c l s } , \mathcal { L } _ { b b o x } , \mathcal { L } _ { g i o u } , \mathcal { L } _ { s c o r e } , \mathcal { L } _ { r e g } , \mathcal { L } _ { q u a l } .$   
8: Optimize $\sum _ { t } \lambda _ { t } ^ { e } { \mathcal { L } } _ { t }$ and record each raw loss.   
9: end for   
10: Convert epoch-average losses to KPI scores and up  
date EMA KPI.   
11: Compute DTP weights and the localization factor.   
12: Update task weights for the next epoch.   
13: end for

where $\tau _ { q u a l } = 0 . 3$ and $\omega _ { i c } ^ { q u a l }$ down-weights dense negative query-class pairs.

At inference time, DFAP-UGC combines the main classification probability, the regional probability, and the quality probability to compute the confidence score:

$$
C _ { i c } = \sigma ( \hat { p } _ { i c } ) \cdot \sigma ( \hat { r } _ { i c } ) \cdot \sigma ( \hat { q } _ { i c } ) ^ { 0 . 5 } .\tag{15}
$$

Each query is decoded in single-label mode by choosing the class with the largest confidence. Class-wise NMS with IoU threshold 0.5 is then applied, and the final output is $\{ \hat { \bf b } , c , \hat { s } \}$

## LADTP Training

Although DFAP-UGC relies on dense assignment and ranking branches, UEAP remains a coupled multi-task problem: localization afects whether classification, severity regression, regional verification, and quality estimation receive meaningful supervision. Inspired by Dynamic Task Prioritization (DTP) (Guo et al. 2018), we propose Locality-Aware DTP (LADTP) training strategy to balance task losses according to epoch-level task dificulty and localization quality.

Let $\mathcal { T } = \{ c l s , b b o x , g i o u , s c o r e , r e g , q u a l \}$ denote the losses. At epoch e, LADTP records the average raw loss $\bar { L } _ { t } ^ { e }$ for each task and converts it into a KPI score. For localizationrelated losses, we use

$$
\kappa _ { t } ^ { e } = \exp ( - \mathrm { c l i p } ( \bar { L } _ { t } ^ { e } , 0 , 2 ) ) , \quad t \in \{ b b o x , g i o u \} ,\tag{16}
$$

and for the remaining losses,

$$
\kappa _ { t } ^ { e } = \exp ( - \mathrm { c l i p } ( \bar { L } _ { t } ^ { e } , 0 , 3 ) ) , \quad t \in \{ c l s , s c o r e , r e g , q u a l \} .\tag{17}
$$

The KPI is smoothed by exponential moving average:

$$
K _ { t } ^ { e } = \rho K _ { t } ^ { e - 1 } + ( 1 - \rho ) \kappa _ { t } ^ { e } .\tag{18}
$$

The dynamic task weight is

$$
w _ { t } ^ { e } = - ( 1 - K _ { t } ^ { e } ) ^ { r } \log K _ { t } ^ { e } .\tag{19}
$$

<table><tr><td>Method</td><td> $\mathsf { A P } _ { 5 0 }$ </td><td> $\mathrm { m A P }$ </td><td> $\mathrm { R } _ { 5 0 }$ </td><td>MAE</td></tr><tr><td>DFAP-UGC</td><td>0.4328</td><td>0.2038</td><td>0.8106</td><td>0.3222</td></tr><tr><td>Faster R-CNN-sum</td><td>0.3758</td><td>0.1719</td><td>0.6951</td><td>0.3215</td></tr><tr><td>Faster R-CNN-single</td><td>0.3755</td><td>0.1682</td><td>0.7021</td><td>0.3303</td></tr><tr><td>Faster R-CNN-concat</td><td>0.3734</td><td>0.1720</td><td>0.6999</td><td>0.3226</td></tr><tr><td>YOLOv12n-concat</td><td>0.3058</td><td>0.1661</td><td>0.5848</td><td>2.0990</td></tr><tr><td>YOLOv12n-single</td><td>0.2813</td><td>0.1524</td><td>0.5617</td><td>2.3276</td></tr><tr><td>YOLOv12n-add</td><td>0.2783</td><td>0.1485</td><td>0.5702</td><td>2.6422</td></tr><tr><td>YOLOv1-concat</td><td>0.2044</td><td>0.0771</td><td>0.4492</td><td>0.3407</td></tr><tr><td>YOLOv1-single</td><td>0.1932</td><td>0.0738</td><td>0.4438</td><td>0.3385</td></tr><tr><td>YOLOv1-sum</td><td>0.1897</td><td>0.0710</td><td>0.4394</td><td>0.3354</td></tr></table>

Table 2: Test comparison between DFAP-UGC and the adapted baselines, including Faster R-CNN, YOLOv1/v12n.

To make the weighting locality-aware, the GIoU KPI produces a localization factor

$$
\eta ^ { e } = \frac { 1 } { 1 + \exp [ - 4 ( K _ { g i o u } ^ { e } - 1 ) ] } .\tag{20}
$$

The efective loss weight for the next epoch is

$$
\lambda _ { t } ^ { e } = \left\{ \begin{array} { l l } { \lambda _ { t } ^ { 0 } w _ { t } ^ { e } , } & { t \in \{ b b o x , g i o u \} , } \\ { \lambda _ { t } ^ { 0 } w _ { t } ^ { e } \eta ^ { e } , } & { t \in \{ c l s , s c o r e , r e g , q u a l \} . } \end{array} \right.\tag{21}
$$

The proposed LADTP training method is summarized in Algorithm 2. Thus, when localization is unreliable, LADTP prevents classification, severity, regional, and quality objectives from dominating optimization too early; as localization improves, these tasks receive stronger efective weights.

## 5 Experiments

We first present experimental settings, followed by performance results, ablation studies, and a case study.

## Experimental Setup

We evaluate all methods on UEAP-4k, which contains 4,222 paired samples. The dataset is split into 3,040 training images, 338 validation images, and an untouched test set of 844 images with 2,729 ground-truth anomaly boxes. A prediction is counted as a true positive only when its category is correct and its IoU with an unmatched ground-truth box is at least 0.5. We report AP at IoU threshold $0 . 5 \ ( \mathrm { A P _ { 5 0 } ) }$ , mAP over IoU thresholds 0.50:0.05:0.95, Recall $\left( \mathrm { R } _ { 5 0 } \right)$ , and severity score MAE on matched true positives. All reported DFAP-UGC detection metrics use the original ranked detections without confidence-threshold filtering. We train DFAP-UGC for 50 epochs using AdamW with learning rate $5 \times 1 0 ^ { - 5 }$ backbone learning rate $5 \times 1 0 ^ { - 6 }$ , weight decay $1 0 ^ { - 4 }$ , and the total batch size is set as 8 during training and inference.

## Comparison with Adapted Baselines

Since existing detectors cannot directly solve UEAP with paired images and severity scores, we adapt Faster R-CNN (Ren et al. 2015), YOLOv12n (Tian, Ye, and Doermann

<table><tr><td>Variant</td><td> $\mathrm { { A P } _ { 5 0 } }$ </td><td> $\mathrm { m A P }$ </td><td> $\mathrm { R } _ { 5 0 }$ </td><td>MAE</td></tr><tr><td>Full DFAP-UGC</td><td>0.4328</td><td>0.2038</td><td>0.8106</td><td>0.3222</td></tr><tr><td>w/o dynamic weighting</td><td>0.4370</td><td>0.1991</td><td>0.7999</td><td>0.3291</td></tr><tr><td>w/o quality head</td><td>0.4328</td><td>0.1937</td><td>0.8010</td><td>0.3317</td></tr><tr><td>w/o TAL</td><td>0.3914</td><td>0.1620</td><td>0.8080</td><td>0.3326</td></tr><tr><td>Problem-only fusion</td><td>0.4235</td><td>0.1934</td><td>0.8069</td><td>0.3142</td></tr><tr><td>w/o regional classifier</td><td>0.4119</td><td>0.1839</td><td>0.8054</td><td>0.3332</td></tr><tr><td>w/o regional HN</td><td>0.4260</td><td>0.1961</td><td>0.7893</td><td>0.3306</td></tr><tr><td>w/o class-weighted box</td><td>0.4253</td><td>0.1923</td><td>0.7948</td><td>0.3357</td></tr><tr><td>w/o box-aware quality</td><td>0.4233</td><td>0.1911</td><td>0.7963</td><td>0.3244</td></tr></table>

Table 3: Ablation results of DFAP-UGC variants on the test set. Each variant removes or weakens one component from the full model. “w/o TAL” replaces task-aligned assignment with dense focal supervision; “Problem-only fusion” removes reference and diference features; “w/o regional HN” disables regional hard-negative mining; and “w/o box-aware quality” removes predicted-box input from the quality head.

2026), and YOLOv1 (Redmon et al. 2016) to the same output protocol. For pairwise inputs, both reference and enhanced images are passed through the backbone, and their features are fused by concatenation or summation before predicting anomaly boxes, categories, and severity scores. We also report the single-image setting, which uses only the enhanced image. Table 2 compares DFAP-UGC with the adapted baselines on the test set. DFAP-UGC achieves the best $\mathrm { A P _ { 5 0 } } .$ mAP, and $\mathrm { R } _ { 5 0 }$ , reaching 0.4328 $\mathsf { A P } _ { 5 0 }$ , 0.2038 mAP, and 0.8106 $\mathrm { R } _ { 5 0 } .$ Compared with the strongest Faster R-CNN baseline, DFAP-UGC improves $\mathrm { { A P } _ { 5 0 } }$ by 5.70 points and mAP by 3.18 points, while also substantially improving recall. Faster R-CNN variants are competitive in severity MAE, but their AP and recall are consistently lower. YOLOv12n and YOLOv1 variants underperform in AP and recall, suggesting that one-stage grid prediction struggles with small and fine-grained enhancement artifacts. Overall, the comparison shows that explicitly modeling problem-reference diferences with dense DETR-style queries provides stronger detection quality than adapting standard detectors.

## Ablation Study

Table 3 reports ablation results of DFAP-UGC on the test set. All variants use the original ranked detections without confidence-threshold filtering. Starting from the full model, each variant removes or weakens one component, including dynamic loss weighting, the dense-IoU quality head, TAL assignment, paired reference/diference fusion, regional classifier, regional hard-negative mining, class-weighted box regression, and box-aware quality estimation.

Full DFAP-UGC tops $\mathsf { A P } _ { 5 0 }$ , mAP, and $\mathrm { R } _ { 5 0 }$ , ranks second in MAE, and shows the best overall balance. Dynamic weighting improves overall metrics despite a slight $\mathsf { A P } _ { 5 0 }$ drop; removing it reduces mAP, $\mathrm { R } _ { 5 0 }$ , and MAE. TAL removal causes the largest AP loss, confirming its key role in dense supervision, while single-image input lowers AP, validating paired reference and diference features. All ablations contribute positively, yet none match the full model’s mAP and recall.

![](images/db13ae048f54c09d13e00bfe7a54f290567751b38c0b0a78c05492d596c16a10.jpg)  
Figure 4: Qualitative inference cases on UEAP-4k. From left to right in each case: reference image, ground-truth annotation on the enhanced image, DFAP-UGC prediction, Faster R-CNN prediction, YOLOv1 prediction, and YOLOv12n prediction.

## Inference Case Study

Figure 4 presents some representative inference cases from UEAP-4k. Each row compares the reference image, the ground-truth annotation on the enhanced image, DFAP-UGC prediction, YOLO prediction, and Faster R-CNN prediction. Compared with the adapted detectors, DFAP-UGC produces predictions that are more consistent with the annotated local anomalies, especially for subtle enhancement-induced distortions that require comparing the enhanced image with a nonideal reference. These cases show that the proposed pairwise diference-fusion design improves local anomaly perception while reducing missed or noisy detections in UGC scenes.

## 6 Conclusion

We introduced UEAP, a new paired-image anomaly perception task for identifying local failures introduced by UGC enhancement. Based on this task, we constructed UEAP-4k and proposed DFAP-UGC, which combines explicit problemreference diference fusion with dense spatial queries, regional verification, quality-aware ranking, and locality-aware dynamic task prioritization. Experiments show that DFAP-UGC achieves stronger performances than the adapted detectors reported in our comparison. The ablation study validates the contribution of the main training and ranking designs.

## References

Batzner, K.; Heckler, L.; and König, R. 2024. Eficientad: Accurate visual anomaly detection at millisecond-level la-

tencies. In Proceedings of the IEEE/CVF winter conference on applications ofcomputer vision, 128–138.

Bergmann, P.; Fauser, M.; Sattlegger, D.; and Steger, C. 2019. MVTec AD–A comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9592–9600.

Cao, J.; Zhang, S.; Liu, Y.; Gao, F.; Gu, K.; Zhai, G.; Dong, J.; and Kwong, S. 2025. Multi-Scale Local and Global Feature Fusion for Blind Quality Assessment of Enhanced Images. IEEE Transactions on Circuits and Systems for Video Technology, 35(9): 8917–8928.

Cao, P.; Wang, Z.; and Ma, K. 2021. Debiased Subjective Assessment of Real-World Image Enhancement. In 2021 IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR), 711–721.

Cao, Y.; Xu, X.; Zhang, J.; Cheng, Y.; Huang, X.; Pang, G.; and Shen, W. 2024. A survey on visual anomaly detection: Challenge, approach, and prospect. arXiv preprint arXiv:2401.16402.

Carion, N.; Massa, F.; Synnaeve, G.; Usunier, N.; Kirillov, A.; and Zagoruyko, S. 2020. End-to-End Object Detection with Transformers. In European Conference on Computer Vision, 213–229. Springer.

Chahine, N.; Ferradans, S.; Vazquez-Corral, J.; and Ponce, J. 2025. Generalized portrait quality assessment. Pattern Recognition Letters, 189: 122–128.

Chen, Z.; Qin, H.; Wang, J.; Yuan, C.; Li, B.; Hu, W.; and Wang, L. 2024. PromptIQA: Boosting the Performance and Generalization for No-reference Image Quality Assessment via Prompts. In European Conference on Computer Vision, 247–264. Springer.

Defard, T.; Setkov, A.; Loesch, A.; and Audigier, R. 2021. PaDiM: A patch distribution modeling framework for anomaly detection and localization. In International Conference on Pattern Recognition, 475–489. Springer.

Deng, H.; and Li, X. 2022. Anomaly detection via reverse distillation from one-class embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9737–9746.

Feng, C.; Zhong, Y.; Gao, Y.; Scott, M. R.; and Huang, W. 2021. TOOD: Task-Aligned One-Stage Object Detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 3510–3519.

Feng, Y.; Li, S.; and Hao, S. 2021. An Error Self-learning Semi-supervised Method for No-reference Image Quality Assessment. In International Conference on Visual Communications and Image Processing (VCIP).

Gu, K.; Tao, D.; Qiao, J.-F.; and Lin, W. 2018. Learning a No-Reference Quality Assessment Model of Enhanced Images With Big Data. IEEE Transactions on Neural Networks and Learning Systems, 29(4): 1301–1313.

Guo, M.; Haque, A.; Huang, D.-A.; Yeung, S.; and Fei-Fei, L. 2018. Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), 270–287.

Kushwaha, R. S.; Rakhra, M.; Singh, D.; and Singh, A. 2022. An overview: super-image resolution using generative adversarial network for image enhancement. In 2022 5th International Conference on Contemporary Computing and Informatics (IC3I), 1243–1246. IEEE.

Lan, X.; Jia, F.; Zhuang, X.; Wei, X.; Luo, J.; Zhou, M.; and Kwong, S. 2025. Hierarchical degradation-aware network for full-reference image quality assessment. Information Sciences, 690: 121557.

Liao, X.; Wei, X.; Zhou, M.; and Kwong, S. 2023. Fullreference image quality assessment: Addressing content misalignment issue by comparing order statistics of deep features. IEEE Transactions on Broadcasting, 70(1): 305–315.

Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Dollar, P. 2017. Focal Loss for Dense Object Detection. In Proceedings ofthe IEEE International Conference on Computer Vision, 2980– 2988.

Liu, J.; Xie, G.; Wang, J.; Li, S.; Wang, C.; Zheng, F.; and Jin, Y. 2024. Deep industrial image anomaly detection: A survey. Machine Intelligence Research, 21(1): 104–135.

Liu, Z.; Lin, Y.; Cao, Y.; Hu, H.; Wei, Y.; Zhang, Z.; Lin, S.; and Guo, B. 2021. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10012–10022.

Mao, Q.; Liu, S.; Li, Q.; Jeon, G.; Kim, H.; and Camacho, D. 2025. No-Reference Image Quality Assessment: Past, Present, and Future. Expert Systems, 42(3): e13842.

Pang, G.; Shen, C.; Cao, L.; and Hengel, A. V. D. 2021. Deep learning for anomaly detection: A review. ACM Computing Surveys (CSUR), 54(2): 1–38.

Prabhakaran, V.; and Swamy, G. 2023. Image Quality Assessment using Semi-Supervised Representation Learning.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning Transferable Visual Models from Natural Language Supervision. In International Conference on Machine Learning, 8748–8763. PMLR.

Redmon, J.; Divvala, S.; Girshick, R.; and Farhadi, A. 2016. You Only Look Once: Unified, Real-Time Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 779–788.

Ren, S.; He, K.; Girshick, R.; and Sun, J. 2015. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In Advances in Neural Information Processing Systems, 91–99.

Rezatofighi, H.; Tsoi, N.; Gwak, J.; Sadeghian, A.; Reid, I.; and Savarese, S. 2019. Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 658–666.

Roth, K.; Pemula, L.; Zepeda, J.; Sch"olkopf, B.; Brox, T.; and Gehler, P. 2022. Towards total recall in industrial anomaly detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14318– 14328.

Safonov, N.; Bryntsev, A.; Moskalenko, A.; Kulikov, D.; Vatolin, D.; Timofte, R.; Lei, H.; Gao, Q.; Luo, Q.; Li, Y.; et al. 2025. NTIRE 2025 challenge on UGC video enhancement: Methods and results. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1503–1513.

Saha, A.; Mishra, S.; and Bovik, A. C. 2023. Re-IQA: Unsupervised learning for image quality assessment in the wild.

Shao, T.; Wang, R.; and Hao, J.-X. 2019. Visual destination images in user-generated short videos: an exploratory study on Douyin. In 2019 16th International Conference on Service Systems and Service Management (ICSSSM), 1–5. IEEE.

Su, S.; Yan, Q.; Zhu, Y.; Zhang, C.; Ge, X.; Sun, J.; and Zhang, Y. 2020. Blindly assess image quality in the wild guided by a self-adaptive hyper network.

Sun, S.; Yu, T.; Xu, J.; Zhou, W.; and Chen, Z. 2023. GraphIQA: Learning Distortion Graph Representations for Blind Image Quality Assessment.

Tian, Y.; Ye, Q.; and Doermann, D. 2026. Yolov12: Attention-centric real-time object detectors. volume 38, 78433–78457.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems, 5998–6008.

Wang, J.; Chan, K. C. K.; and Loy, C. C. 2023. Exploring CLIP for Assessing the Look and Feel of Images.

Wang, J.; Duan, H.; Zhai, G.; and Min, X. 2026. Quality assessment for AI generated images with instruction tuning. IEEE Transactions on Multimedia.

Wang, S.; Li, C.; Zhang, Z.; Zhou, H.; Dong, W.; Chen, J.; Zhai, G.; and Liu, X. 2025. AU-IQA: A Benchmark

Dataset for Perceptual Quality Assessment of AI-Enhanced User-Generated Content. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25). Dublin, Ireland: ACM. ISBN 979-8-4007-2035-2.

Yang, J.; Lyu, M.; Qi, Z.; and Shi, Y. 2023. Deep learning based image quality assessment: A survey. Procedia Computer Science, 221: 1000–1005.

You, Z.; Cui, L.; Shen, Y.; Yang, K.; Lu, X.; Zheng, Y.; and Le, X. 2022. A unified model for multi-class anomaly detection. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, 4571–4584.

Zavrtanik, V.; Kristan, M.; and Skočaj, D. 2021. DRAEM–A discriminatively trained reconstruction embedding for surface anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 8330–8339.

Zhai, G.; and Min, X. 2020. Perceptual image quality assessment: a survey. Science China Information Sciences.

Zhang, W.; Ma, K.; Zhai, G.; and Yang, X. 2024. Taskspecific normalization for continual learning of blind image quality models.

Zhang, W.; Zhai, G.; Wei, Y.; Yang, X.; and Ma, K. 2023. Blind image quality assessment via vision-language correspondence: A multitask learning perspective.

Zhao, K.; Yuan, K.; Sun, M.; Li, M.; and Wen, X. 2023. Quality-aware pre-trained models for blind image quality assessment.

Zhong, Y.; Wu, X.; Zhang, L.; Yang, C.; and Jiang, T. 2024. Causal-IQA: Towards the Generalization of Image Quality Assessment Based on Causal Inference. In ICML.

Zhong, Y.; Yang, C.; Zhao, S.; and Jiang, T. 2025a. Semi-supervised blind quality assessment with confidencequantifiable pseudo-label learning for authentic images. In Forty-second International Conference on Machine Learning.

Zhong, Y.; Zhao, X.; Zhang, L.; Song, X.; and Jiang, T. 2025b. Adaptive Prompt Learning for Blind Image Quality Assessment with Multi-modal Mixed-datasets Training. In Proceedings of the 33rd ACM International Conference on Multimedia, 7453–7462.

Zhong, Y.; Zhao, X.; Zhao, G.; Chen, B.; Hao, F.; Zhao, R.; He, J.; Shi, L.; and Zhang, L. 2025c. Ctd-inpainting: Towards the coherence of text-driven inpainting with blended difusion. Information Fusion, 122: 103163.

Zou, Y.; Jeong, J.; Pemula, L.; Zhang, D.; and Dazhou, O. 2022. Spot-the-diference self-supervised pre-training for anomaly detection and segmentation. In European Conference on Computer Vision (ECCV), 392–408. Springer.