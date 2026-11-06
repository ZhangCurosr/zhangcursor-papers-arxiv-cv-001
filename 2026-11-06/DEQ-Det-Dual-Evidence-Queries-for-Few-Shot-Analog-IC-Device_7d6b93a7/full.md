# DEQ-Det: Dual Evidence Queries for Few-Shot Analog IC Device Detection

Zidong Zhao<sup>1</sup>, Rong Huang<sup>2</sup>, Chengxing Fan<sup>1</sup>, Jie Hu<sup>1</sup>,

Malik Haris<sup>1</sup>, Ghazal A. Fahmy<sup>3</sup>, Morteza Mousazadeh<sup>4</sup>, Guoqiang Zhang<sup>5†</sup>

<sup>1</sup>Zhejiang University, China, <sup>2</sup>Donghua University, China,

<sup>3</sup>National Telecommunication Institute, Egypt, <sup>4</sup>Urmia University, Iran,

<sup>5</sup>Ningbo Tech University, China

Abstract—Detecting analog integrated circuit (IC) devices in layout images supports circuit analysis and inspection. With only a few annotated examples, variation in device geometry and visual similarity to dummy or routing structures make reliable detection difficult. Foreground similarity alone can highlight plausible targets without distinguishing them from competing layout patterns. We propose Dual Evidence Queries for Detection (DEQ-Det), which constructs complementary foreground and background references from the same support images without additional annotations. Foreground references capture target morphology, while selected background references characterize surrounding layout patterns. Both sources guide query selection and region verification, combining target appearance with layout context. Across five held-out device categories and three random seeds, DEQ-Det achieves macro AP50 of 39.41 and 53.03 at 10 and 30 shots, exceeding the strongest listed baselines by 3.97 and 9.71 points, respectively. A matched 30-shot ablation further shows that RoI background evidence reduces pooled false positives at comparable recall, supporting its role in interference suppression.

Index Terms—few-shot object detection, analog IC layouts, background evidence, query selection, localization quality

## I. INTRODUCTION

Integrated circuit (IC) device detection localizes resistors, capacitors, and transistors in layout images for circuit analysis, targeted inspection, and physical probing [1]. Layout representations can be reconstructed from optical microscopy or scanning electron microscopy (SEM) images [2]. Analog devices have customized geometries reflecting sizing and matching requirements [3], limiting the transferability of digital standard-cell templates to analog layouts. Dummy fills and repetitive routing patterns can resemble devices and cause false positives. Automated localization can focus manual inspection on candidate regions. Dense annotation for supervised detection is costly, with additional labeling often needed across fabrication processes.

Few-shot object detection (FSOD) adapts to targets defined by a small annotated support set [4], [5]. Existing approaches improve target representation through support-conditioned aggregation, contrastive learning, and prototype distillation [6]– [9]. DE-ViT and FSOD-VFM use transferable features for exemplar-based or cross-domain detection [10], [11]. For IC imagery, SSRNet combines few-shot segmentation and a region classifier for packaged ICs in optical PCB images [12]. Detecting internal devices in analog layouts requires resolving customized geometries amid repeated, target-like structures.

![](images/ad26d666f7166258e85baecec46627dd0f58c7c0b67b5106a6ea03273eee3aae.jpg)  
Fig. 1. Analog layout variability and ambiguity. Device geometries vary within categories; non-target dummy structures share outlines or repeated patterns with devices. Blue/green boxes indicate devices, and the red dashed region indicates dummies.

Figure 1 illustrates device variability and resemblance to non-target structures. Foreground similarity can highlight plausible candidates while also responding to dummy fills or routing patterns. The same support images provide context that helps resolve this ambiguity: annotated devices specify the requested morphology, while selected background regions characterize competing layout patterns. Using both sources allows candidate selection and verification to account for the target and its surroundings.

We propose Dual Evidence Queries for Detection (DEQ-Det) to integrate device morphology and support-specific layout context throughout detection. Our contributions are twofold: (1) a dual-evidence framework that derives complementary foreground and background references without additional annotations and uses them in query selection and region verification to couple target selection with interference suppression; and (2) evaluation across five held-out device categories and three random seeds, demonstrating systemlevel AP50 gains. Matched controls establish the roles of support-guided allocation and quality-aware ranking, while a RoI background control shows fewer pooled false positives at comparable recall.

![](images/8f8724d6a9c7f50bbb97c960050633a40d4fa2b64f17db06bc512e80714202c6.jpg)  
Fig. 2. DEQ-Det. Foreground references guide target selection, and filtered background references supply layout context for interference suppression. Their means guide query selection, while individual bank records enter the RoI relation head. Foreground probability and predicted localization quality jointly rank candidate detections.

## II. PROPOSED METHOD

## A. Support-Derived Foreground and Background

Given K selected boxes of one novel category, DEQ-Det constructs two evidence banks (Fig. 2). A shared DINOv2 ViT-L/14 encoder [13], with a frozen IC reconstruction adapter, produces 1,024-dimensional features. Each support box is enlarged by 20%, cropped, resized while preserving its aspect ratio, and padded to 336 × 336 before encoding in its original orientation. Masked pooling over the annotated region yields one morphology record per instance, giving $B _ { f g } = \{ f _ { k } \} _ { k = 1 } ^ { K } .$

Background extraction summarizes the surroundings of annotated devices in full support images, using spatial exclusion and residual filtering to favor stable layout interiors. Valid feature patches outside a two-cell dilation of the selected boxes form a candidate mask $C _ { s }$ . The channel-averaged absolute difference between the image and its Gaussian-blurred version (15 × 15 kernel, $\sigma \ = \ 3 )$ measures local texture variation, providing a visual heuristic for background selection. Quantiles are computed over $C _ { s }$ : patches in its lowest residual quintile are retained, excluding a three-cell neighborhood of its highest residual quintile. If this intersection is empty, the lowest-residual candidate patches provide a fallback; annotated and padded patches remain excluded.

The retained features are pooled into image-level and patch-chunk records. The background bank $B _ { b g }$ has 360 records, repeating valid pooled records when necessary. Its mean <sup>¯</sup>b summarizes support-specific layout context for query selection, while individual records remain available for region classification.

## B. Dual-Evidence Query Selection

Query selection directs the decoder toward locations that resemble the support devices while discounting similarity to support context. After trainable feature processing, let $x _ { i }$ denote a query token in the selector embedding space and ${ \bar { f } } \ = \ K ^ { - 1 } \sum _ { k } f _ { k }$ the foreground mean; hats denote $\ell _ { 2 }$ normalization. The token ranking is

$$
a _ { i } ^ { f g } = \langle \widehat { x } _ { i } , \widehat { \bar { f } } \rangle , \quad a _ { i } ^ { b g } = \langle \widehat { x } _ { i } , \widehat { \bar { b } } \rangle , \quad s _ { i } = a _ { i } ^ { f g } - \lambda _ { \mathrm { s e l } } a _ { i } ^ { b g } .\tag{1}
$$

Up to 600 highest-ranked valid tokens initialize a two-layer decoder, which predicts boxes from their spatial references. This allocates the query budget according to support relevance. A shared residual MLP adjusts selector query features with scale 0.1; $\lambda _ { \mathrm { s e l } } = 0 . 1 5$

During adaptation, a selector loss uses feature-token centers inside selected boxes as positives and other valid centers as negatives. Sets P and $\mathcal { N }$ contain up to 128 lowest-scoring positives and 512 highest-scoring negatives, respectively:

$$
\mathcal { L } _ { \mathrm { s e l } } = \frac { 1 } { | \mathcal { P } | | \mathcal { N } | } \sum _ { i \in \mathcal { P } , j \in \mathcal { N } } [ 0 . 2 - s _ { i } + s _ { j } ] _ { + } .\tag{2}
$$

## C. Region Evidence and Localization Quality

Region verification compares each decoded candidate with both evidence banks. RoIAlign [14] extracts a 14 × 14 feature grid at each decoded box. The relation head combines similarity to bank means, similarity to individual records, and learned pairwise relations with weights 0.5, 0.5, and 1.0. It uses 256 hidden dimensions, eight region tokens, and the three bestmatching support records. Foreground/background logits are normalized jointly:

$$
( p _ { f } ( r ) , p _ { b } ( r ) ) = \mathrm { s o f t m a x } ( z _ { f } ( r ) , z _ { b } ( r ) ) .\tag{3}
$$

RoI classification labels are determined by the maximum IoU between each pooled region and annotated targets: regions with $\mathrm { I o U } \geq 0 . 5$ are foreground, those with $\mathrm { I o U ~ < ~ 0 . 4 }$ are background, and the intermediate range is ignored. Multiple RoIs can therefore be positive for one target, while box regression uses one-to-one Hungarian matching [15].

An IoU-supervised quality head predicts $q _ { r } \in [ 0 , 1 ]$ from region features for localization-aware detection ranking. Binary cross entropy and Smooth L1 anchor predictions to detached IoU targets for assigned or overlapping regions, with zero targets for non-overlapping background regions. A complementary IoU-gap hinge encourages a minimum quality separation between foreground boxes on the same normalized scale. For target $^ { g , }$ let $\mathcal { E } _ { g }$ contain ordered pairs assigned to that target by maximum IoU, where $u _ { i }$ is the IoU of region i with $g , u _ { i } , u _ { j } \ge 0 . 5 .$ , and $u _ { i } - u _ { j } \geq 0 . 1$ . Then

$$
\mathcal { L } _ { \mathrm { r a n k } } = \frac { 1 } { \left| \mathcal { G } ^ { \prime } \right| } \sum _ { g \in \mathcal { G } ^ { \prime } } \frac { 1 } { \left| \mathcal { E } _ { g } \right| } \sum _ { ( i , j ) \in \mathcal { E } _ { g } } \left[ ( u _ { i } - u _ { j } ) - ( q _ { i } - q _ { j } ) \right] + ,\tag{4}
$$

where $\mathcal { G } ^ { \prime }$ includes targets with eligible pairs. Equal target weighting prevents crowded proposal groups from dominating. Box coordinates are detached when constructing targets for the ordering loss.

## D. Training Objective and Detection Ranking

The novel-stage objective combines region cross entropy, box $\ell _ { 1 } / \mathrm { G I o U }$ regression, a foreground/background logit margin, selector supervision, and quality losses:

$$
\begin{array} { r l } & { \mathcal { L } = 2 \mathcal { L } _ { \mathrm { C E } } + 5 \mathcal { L } _ { 1 } + 2 \mathcal { L } _ { \mathrm { G I o U } } + 0 . 2 5 \mathcal { L } _ { \mathrm { m a r g i n } } } \\ & { ~ + 0 . 0 5 \mathcal { L } _ { \mathrm { s e l } } + \mathcal { L } _ { q } + \lambda _ { r } \mathcal { L } _ { \mathrm { r a n k } } + \mathcal { L } _ { \mathrm { d e c } } . } \end{array}\tag{5}
$$

The margin is 0.2 on foreground/background logits, and cross entropy uses background weight 0.2. Quality supervision assigns weight 0.25 to zero-overlap background RoIs and unit weight otherwise. Intermediate decoder predictions receive corresponding Hungarian classification, margin and regression losses, denoted ${ \mathcal { L } } _ { \mathrm { d e c } }$ . The final 100-step continuation activates $\lambda _ { r } = 0 . 2 5$ to refine quality ordering after the initial adaptation.

At inference, candidates satisfy $p _ { f } > 0 . 0 5$ . The score used for NMS and reported confidence is

$$
\begin{array} { r } { s _ { r } ^ { \mathrm { d e t } } = p _ { f } ( r ) q _ { r } ^ { 1 + 0 . 2 5 h _ { r } } d _ { r } , \quad \quad } \\ { h _ { r } = { \bf 1 } [ p _ { f } ( r ) - p _ { b } ( r ) < - 0 . 0 5 ] . \quad \quad } \end{array}\tag{6}
$$

The conditional exponent further downweights regions favoring background according to their predicted localization quality. Here $d _ { r }$ discounts redundant boxes. Within sameclass overlap components at IoU 0.6, the highest- $- p _ { f }$ box is the owner. A lower-confidence box overlapping its owner by at least 0.6 is multiplied by $1 - 0 . 3 5 \mathrm { I o U } ( r , r _ { o } )$ when $q _ { r } \ \leq$ $q _ { r _ { o } } + 0 . 0 2 ;$ otherwise $d _ { r } = 1$ . NMS uses IoU 0.5 and retains at most 100 detections. All categories share these settings.

![](images/9184b1f66e515a8ac50c77ff44b0f8985195ff6ba814a3dd0768e32ce106cd19.jpg)  
Cyan dashed: ground truth in the first column Green: correct prediction Red: incorrect prediction  
Fig. 3. Qualitative comparison at 30 shots on Diode and NPN/PNP examples. Columns show ground truth, TFA, DE-ViT, FSOD-VFM, and DEQ-Det. Cyan dashed boxes denote ground truth; green/red boxes denote matched/unmatched predictions at IoU 0.5.

## III. EXPERIMENTS

## A. Data and Evaluation Protocol

Our dataset comprises image windows cropped from electron micrographs of integrated circuits. It contains 907 training and 282 test windows (3,118/760 instances) from 329/94 source images, respectively. All windows from the same source image remain in a single partition. Each of five fixed splits holds out one novel category—Capacitor, Diode, MOS, NPN/PNP, or Resistor—with the remaining categories used for base training. The novel test subsets contain 35/40/29/31/90 windows and 44/46/55/88/152 target instances, respectively. Their union has 195 distinct windows from 76 test source images; category subsets can overlap in images. All compared methods use these same test manifests.

One shot is one selected novel box, with at most one selected box per support image. At 10 and 30 shots, the novel training data and support banks use exactly those K boxes; locations of other same-class instances are not supplied as ignore regions. All compared methods share the selected support instances. Base/refinement initialization is fixed within each held-out-category split.

Following PASCAL-style few-shot detection [5], [7], our primary metric is novel-category AP50, computed with COCO evaluation [16]. We additionally report AP75 for the quality ablation. For each seed, we average the five category scores equally, then compute the mean and sample standard deviation over seeds 1/2/3. Support sampling and training randomness are paired; all results use fixed final checkpoints.

## B. Implementation and Baselines

The shared encoder includes a frozen rank-4 LoRA adapter $( \alpha = 8 )$ on query/value projections in its last six blocks [17]. The adapter is pretrained for image reconstruction without device labels and shared across splits and IC-adapted baselines. We retain each split’s initialization from 8,000 basetraining and 2,000 base-refinement steps. Novel adaptation uses AdamW, batch size two, learning rate $1 0 ^ { - 5 } ;$ , and cosine decay for 1,200 steps with 20 warmup steps. A fresh optimizer runs 100 further steps at the same initial learning rate without warmup, enabling Eq. (4). Adaptation and inference settings are common across categories and support budgets. Images are resized to a maximum side of 1,024 pixels, normalized with ImageNet statistics, and padded to multiples of 14. Training augmentation uses horizontal flips and right-angle rotations. Training uses two RTX 4090 GPUs. At 30 shots, inference with batch size one averages 432 ms per window on an RTX 4090, including model preprocessing and postprocessing; data loading and offline bank construction are excluded.

TABLE I  
FIVE-CATEGORY MACRO AP50 AT 10 AND 30 SHOTS, MEAN±SAMPLE STANDARD DEVIATION OVER SEEDS 1/2/3.
<table><tr><td>Method</td><td>10-shot</td><td>30-shot</td></tr><tr><td>Template matching</td><td> $1 3 . 2 0 { \pm } 2 . 0 0 \ $ </td><td> $1 7 . 1 7 { \pm } 2 . 7 5 $ </td></tr><tr><td>DINOv2 prototype</td><td> $1 2 . 9 2 { \pm } 1 . 4 3$ </td><td> $2 0 . 8 0 { \pm } 3 . 2 8 $ </td></tr><tr><td>TFA [5]</td><td> $1 8 . 7 1 { \pm } 1 . 5 6 $ </td><td> $2 1 . 9 3 { \pm } 1 . 0 8$ </td></tr><tr><td>Meta R-CNN [6]</td><td> $3 1 . 0 2 { \pm } 4 . 8 1$ </td><td> $4 3 . 3 2 { \pm } 5 . 7 8 $ </td></tr><tr><td>FSCE [7]</td><td> $3 2 . 3 8 { \pm } 2 . 1 7$ </td><td> $3 8 . 6 5 { \pm } 0 . 3 5 $ </td></tr><tr><td>VFA [8]</td><td> $3 2 . 8 9 { \pm } 1 . 9 8 $ </td><td> $3 8 . 0 1 { \pm } 2 . 1 8 $ </td></tr><tr><td>FPD [9]</td><td> $2 6 . 3 8 { \pm } 0 . 9 3$ </td><td> $3 1 . 7 4 \pm 3 . 3 4$ </td></tr><tr><td>DE-ViT [10]</td><td> $2 5 . 9 1 { \pm } 1 . 9 0 $ </td><td> $3 0 . 2 6 { \pm } 1 . 7 0 $ </td></tr><tr><td>DE-ViT + IC adapter</td><td> $2 3 . 9 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $2 8 . 9 8 { \pm } 5 . 4 4$ </td></tr><tr><td> $\mathrm { F S O D - V F M \ [ 1 1 ] }$ </td><td> $3 1 . 9 7 { \pm } 1 . 5 0 $ </td><td> $3 4 . 8 2 { \pm } 0 . 0 7$ </td></tr><tr><td>FSOD-VFM + IC adapter</td><td> $\underline { { 3 5 . 4 4 \pm 0 . 8 8 } }$ </td><td> $3 7 . 8 8 { \pm } 0 . 4 3 $ </td></tr><tr><td>DEQ-Det</td><td> ${ \bf 3 9 . 4 1 } { \pm 2 . 2 8 }$ </td><td> ${ \pm } 3 . 0 3 { \pm } 4 . 8 1 $ </td></tr></table>

Bold and underlined results indicate the best and second-best mean AP50 in each column, respectively. R101: TFA, Meta R-CNN, FSCE, VFA, and FPD; ViT-L/14: the remaining learned methods. IC denotes the frozen IC reconstruction adapter; FSOD-VFM also uses UPN and SAM2-L.

Original and IC-adapted DE-ViT/FSOD-VFM are reported separately. The adapted variants share the frozen IC encoder with DEQ-Det, but retain their detection pipelines. In particular, IC-adapted DE-ViT retains released background prototypes, and FSOD-VFM retains UPN and SAM2-L. Backbones, pretraining and optimization budgets differ across systems; the comparison evaluates complete detection systems under shared support and test sets.

## C. Main Results

DEQ-Det achieves macro AP50 of 39.41±2.28 and 53.03±4.81 at 10 and 30 shots, respectively (Table I). The gains over the strongest listed baselines, IC-adapted FSOD-VFM and Meta R-CNN, are 3.97 and 9.71 points, respectively. It also leads the IC-adapted foundation-model baselines sharing its frozen encoder. At 30 shots, category AP50 is 62.59/56.08/41.49/64.99/40.03 for Capacitor, Diode, MOS, NPN/PNP, and Resistor, respectively. All five category means improve from 10 to 30 shots. Figure 3 compares detections on shared Diode and NPN/PNP test images at 30 shots.

## D. Component Ablations

All component ablations use paired support sets and seeds with identical initialization and training schedules. The selector control replaces support-guided selection with featurenorm ranking and removes selector supervision, while retaining RoI evidence and quality scoring. The quality control removes quality supervision and quality-dependent scoring in the same intervention.

![](images/89059d69c07f78291c120f0d460b9156ac9aa0a7ce02cf65b7982dda58a65572.jpg)  
Fig. 4. Matched 30-shot component ablations: five-category macro AP50, mean and sample standard deviation over three seeds. Blue/orange bars denote full/ablated models.

Figure 4 evaluates the contributions of query selection and detection ranking. Support-guided selection provides an 11.12- point macro AP50 gain over its matched control. Quality supervision together with quality-dependent scoring adds 9.80 points. Both gains are positive for all three paired seeds. These controls support allocating a fixed query budget to support-relevant candidates and ranking decoded regions with localization-aware confidence. The same quality configuration also improves AP75 by 9.59 points over its matched control.

Background evidence at comparable recall. To assess interference suppression, we zero the RoI support-background features during novel adaptation and inference, while retaining selector background evidence $( \lambda _ { \mathrm { s e l } } = 0 . 1 5 )$ . Using fixed final checkpoints and score-ordered one-to-one matching at IoU $\geq ~ 0 . 5 ,$ , we pool categories within each seed and use each arm’s highest threshold reaching the prespecified recall targets of 50/60/70/80/85%. Across these comparable-recall points, retaining RoI background reduces false positives pooled over seeds by 3.5–9.8%. At 60% recall, false positives decrease from 1,004 to 906, with unchanged matched-target counts and reductions for all three paired seeds. RoI background references therefore improve device–background discrimination and reduce spurious candidates for targeted layout inspection. The same control also yields a 0.86-point mean AP50 gain.

## IV. CONCLUSION

DEQ-Det combines device morphology and layout context for few-shot analog IC detection without additional background annotations. Support examples update task-specific references, allowing new target geometries to be specified without separate labels for dummy or routing structures. Macro AP50 exceeds the strongest listed baselines by 3.97 and 9.71 points at 10 and 30 shots. Matched controls show gains from support-guided query allocation and quality-aware ranking. RoI background evidence reduces false positives at comparable recall, yielding fewer distracting candidates for manual review. With K annotated devices defining the inspection target, the predicted boxes can guide review toward localized regions in electron micrographs. These locations also provide spatial references for circuit analysis and selection of probing regions.

## REFERENCES

[1] V. K. Ravikumar, J. M. Chin, W. Lua, N. Linarto, G. Ranganathan, J. Trisno, K. L. Pey, and J. K. W. Yang, “Super-resolution laser probing of integrated circuits using algorithmic methods,” Nature Communications, vol. 13, no. 1, p. 5155, 2022.

[2] H.-C. Shao, C.-Y. Peng, J.-R. Wu, C.-W. Lin, S.-Y. Fang, P.-Y. Tsai, and Y.-H. Liu, “From IC layout to die photograph: A CNN-based data-driven approach,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 40, no. 5, pp. 957–970, 2021.

[3] J. Lienig and J. Scheible, Fundamentals of Layout Design for Electronic Circuits. Cham, Switzerland: Springer, 2020.

[4] B. Kang, Z. Liu, X. Wang, F. Yu, J. Feng, and T. Darrell, “Few-shot object detection via feature reweighting,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 8420–8429.

[5] X. Wang, T. E. Huang, J. E. Gonzalez, T. Darrell, and F. Yu, “Frustratingly simple few-shot object detection,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. Proceedings of Machine Learning Research, vol. 119, 2020, pp. 9919–9928.

[6] X. Yan, Z. Chen, A. Xu, X. Wang, X. Liang, and L. Lin, “Meta R-CNN: Towards general solver for instance-level low-shot learning,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 9577–9586.

[7] B. Sun, B. Li, S. Cai, Y. Yuan, and C. Zhang, “FSCE: Few-shot object detection via contrastive proposal encoding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 7352–7362.

[8] J. Han, Y. Ren, J. Ding, K. Yan, and G.-S. Xia, “Few-shot object

detection via variational feature aggregation,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 37, no. 1, 2023, pp. 755–763.

[9] Z. Wang, B. Yang, H. Yue, and Z. Ma, “Fine-grained prototypes distillation for few-shot object detection,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 38, no. 6, 2024, pp. 5859–5866.

[10] X. Zhang, Y. Liu, Y. Wang, and A. Boularias, “Detect everything with few examples,” in Proc. Conf. Robot Learn. (CoRL), ser. Proceedings of Machine Learning Research, vol. 270, 2025, pp. 3986–4004.

[11] C.-B. Feng, Y. Sha, L. Liu, Y. Yu, C. M. Vong, X. Yu, and X. Shen, “FSOD-VFM: Few-shot object detection with vision foundation models and graph diffusion,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[12] Y. Wang, X. Wang, D. Cheng, T. Lin, F. Ji, Y. Shi, and B.-H. Gwee, “SSRNet: Few-shot IC segmentation in automated PCB image processing,” in Proc. IEEE Int. Symp. Circuits Syst. (ISCAS), 2025.

[13] M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[14] K. He, G. Gkioxari, P. Dollar, and R. Girshick, “Mask R-CNN,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2961–2969.

[15] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 213–229.

[16] T.-Y. Lin et al., “Microsoft COCO: Common objects in context,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 740–755.

[17] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.