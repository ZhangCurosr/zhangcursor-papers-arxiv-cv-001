# Cross-Model Agreement as a Deployment-Time Reliability Signal for Automatic Polyp Segmentation

Siddharth Gupta<sup>a</sup>, Jitin Singla<sup>a,∗</sup>

<sup>a</sup>Department of Biosciences and Bioengineering, Indian Institute of Technology Roorkee, Roorkee, 247667, Uttarakhand, India

## Abstract

In real-time colonoscopy, ground-truth annotations are unavailable at inference, so polyp segmentation models can fail silently. We propose Referee-Based Quality Estimation (RBQE), a reference-free framework measuring agreement between a primary segmentation model and an independently trained referee on the same image. RBQE is evaluated on a standardized 1,223-image external benchmark drawn from four public datasets, using four referee configurations chosen to separate two design axes: referee independence and architectural diversity. Using a common Agreement Dice descriptor, a same-architecture referee difering from the primary model only in random initialization already yields a useful reliability signal (ROC-AUC = 0.923), showing that independent training alone is suficient. Cross-architecture referees improve further: SegFormer-B0 achieves the strongest performance (ROC-AUC = 0.960), significantly outperforming the same-architecture control and UNet++, and exceeding a representative Test-Time Augmentation baseline by 0.055 ROC-AUC under an identical protocol, whereas a prompt-coupled MedSAM referee underperforms despite maximal architectural diversity. Because empty-mask agreement is trivially separable, we also report a restricted evaluation excluding such cases: ROC-AUC falls to 0.876 (SegFormer-B0, 1,046 images) and 0.783 (same-architecture control, 975 images), yet RBQE’s margin over both baselines widens on this identical subset. RBQE additionally increases the mean Dice of retained predictions as low-agreement cases are progressively rejected, supporting selective prediction, and requires only one additional deterministic referee forward pass at inference. Our study therefore supports cross-model agreement as a practical, interpretable reliability framework for automated polyp segmentation.

Keywords: Segmentation quality estimation, Polyp segmentation, Reliability estimation, Model agreement, Deep learning, Colonoscopy

## 1. Introduction

Colorectal cancer (CRC) is the third most commonly diagnosed cancer and the second leading cause of cancer-related death worldwide, with an estimated 1.93 million new cases and 904,000 deaths reported globally in 2022 [1]. Timely detection and removal of precancerous polyps substantially improves patient survival, and colonoscopy remains the gold-standard technique for early polyp detection. However, colonoscopy is resource-intensive, requires specially trained endoscopists, and its diagnostic yield is highly dependent on operator experience — a meta-analysis of 43 tandem-colonoscopy studies (>15,000 procedures) reported an adenoma miss rate of 26% (95% CI: 23–30%) [2].

Deep learning has substantially advanced automatic polyp segmentation, enabling more accurate lesion delineation and supporting computer-aided diagnosis (CAD) systems [3]. Despite this progress, segmentation models continue to fail under challenging imaging conditions such as camera motion, poor illumination, specular artifacts, blur, very small polyps, and domain shift across imaging hardware. Because groundtruth annotations are unavailable at inference, such failures cannot be detected using standard metrics like the Dice Similarity Coeficient (DSC) or Intersection-over-Union (IoU), leaving clinicians and downstream decision-support systems with no reliable mechanism for flagging an unreliable segmentation before it informs diagnosis.

This gap has motivated no-reference Segmentation Quality Estimation (SQE), whose objective is to estimate segmentation reliability when ground-truth annotations are unavailable at inference. As detailed in Section 2, existing SQE approaches derive their quality signal either from a single model’s own behavior (confidence scores, predictive entropy, Monte Carlo (MC) Dropout, Test-Time Augmentation(TTA)), from disagreement among ensembles of identically architected models, or from per-image reverse-classifier fitting against a reference atlas; these families respectively inherit the deployed model’s calibration errors, a shared architectural inductive bias, and a substantial per-image retraining cost. To date, agreement between architecturally heterogeneous, independently trained segmentation models — decoupled from any single model’s calibration and requiring no per-image retraining — has not been systematically investigated as a deployment-time reliability signal for polyp segmentation.

In the present work, we propose Referee-Based Quality Estimation (RBQE), a simple and practical framework that estimates segmentation reliability using agreement between independently trained segmentation models. Rather than relying on a single model’s internal confidence, RBQE assesses the coherence between a primary segmentation model and an independently trained referee model applied to the same image. The underlying hypothesis is that when independently trained models converge to similar segmentation masks, the prediction is more likely to be reliable, whereas substantial disagreement indicates an increased likelihood of segmentation failure. We frame the choice of referee along two design axes: (i) independence — whether the referee’s training and, crucially, its inference-time output are decoupled from the primary model and (ii) architectural diversity — whether the referee embodies a diferent inductive bias. The four referee configurations evaluated in this study (Section 3.4) are chosen to separate these axes: a same-architecture, independently trained control isolates the contribution of independence; two architecturally distinct, independently trained referees add diversity; and a prompt-driven foundation model that is architecturally diverse but functionally coupled to the primary prediction isolates the role of output-level independence. RBQE is not proposed to predict the exact Dice coeficient, nor does it seek to replace supervised quality estimation models; instead, it provides a practical, deployment-time reliability estimate capable of flagging unreliable predictions when ground-truth labels are unavailable.

To evaluate this hypothesis, external validation was performed on a standardized benchmark of 1,223 colonoscopy images drawn from four independent, publicly available datasets, with all headline results additionally reported on restricted subsets that exclude trivially separable empty-mask cases. The proposed framework was compared against representative uncertainty-based (TTA) and morphology-based quality estimation baselines under a standardized evaluation protocol, using ROC-AUC, bootstrap confidence intervals, paired DeLong significance testing, and computational complexity analysis.

The principal contributions of this work are as follows:

1. Isolating the role of referee independence. Using a same-architecture control referee that difers from the primary model only in random initialization, we demonstrate that independent training alone, without architectural diversity, already provides a meaningful reliability signal (ROC-AUC = 0.923) — to the best of our knowledge, a distinction that has not been explicitly investigated in the SQE literature.

2. Cross-architecture diversity adds a statistically significant increment. Agreement between the primary model and an independently trained, architecturally distinct referee (SegFormer-B0) provides the strongest deployment-time failure-detection signal (ROC-AUC = 0.960), significantly outperforming the same-architecture control and UNet++, and outperforming representative single-model uncertainty (TTA) and morphology-based baselines.

3. Diversity without output-level independence fails. A prompt-driven foundation-model referee (MedSAM, ROC-AUC = 0.863), whose inference prompt is derived from the primary model’s own prediction, underperforms even the same-architecture control despite maximal architectural diversity — identifying genuine output-level independence, rather than architectural diference alone, as the necessary ingredient of the agreement signal.

4. A dual full/non-degenerate evaluation protocol. Because empty-mask agreement is trivially separable, all headline comparisons are reported both on the full standardized benchmark and on restricted subsets excluding degenerate cases (ROC-AUC 0.960 → 0.876 for SegFormer-B0; 0.923 → 0.783 for the same-architecture control); RBQE’s margin over the evaluated baselines widens, rather than narrows, on the restricted subset.

5. Practical, label-free, single-pass deployment. RBQE requires only one additional deterministic forward pass through a referee model at inference, with no per-image retraining, no quality-labelled training data, and no modification of the deployed primary model; five interpretable agreement descriptors are evaluated and ranked, and the continuous agreement score is shown to support selective prediction.

## 2. Related Work

In real-time colonoscopy, ground-truth annotations are unavailable during inference, making it impossible to directly assess whether an automatically generated segmentation is reliable. A substantial body of work has therefore investigated Segmentation Quality Estimation (SQE) without access to ground truth.

## 2.1. Single-model quality signals

The first family relies on the behavior of a single segmentation model. Robinson et al. [4] introduced a real-time CNN-based method for predicting segmentation quality directly from image and mask features for cardiac MRI quality control. Jungo and Reyes [5] systematically evaluated multiple uncertainty estimation methods for medical image segmentation and showed that existing approaches are reliable at the dataset level but often miscalibrated at the subject level. Eaton Rosen et al. [6] proposed a Bayesian deep learning framework converting voxel-wise uncertainty into calibrated volumetric confidence intervals. These approaches, together with MC Dropout [7] and TTA [8], share a common limitation: the quality signal is derived entirely from one model’s own behavior and therefore remains fundamentally dependent on that model’s internal calibration; when a model is overconfident or encounters out-of-distribution data, these signals become unreliable precisely when accurate quality assessment matters most.

## 2.2. Agreement-based estimation

A second line of research instead estimates segmentation quality through agreement between multiple predictions. Deep ensembles [9] train several instances of the same architecture from diferent random initializations and use their predictive disagreement as an uncertainty signal; random initialization alone can already yield substantial function-space diversity [10], yet ensemble members sharing an identical architectural inductive bias may still exhibit correlated rather than genuinely independent failure modes. Reverse Classification Accuracy (RCA) [11] instead fits a new classifier per test image using the model’s own prediction as pseudo ground truth and scores it against a reference atlas, avoiding a second trained model but at the cost of per-image retraining and atlas dependence. RBQE is positioned within this agreement-based family but difers from both: it measures agreement between a fixed primary model and a single, independently trained referee requiring no retraining at inference time, and explicitly isolates whether the agreement signal depends on architectural diversity or on independent training alone — a question not directly addressed by either deep ensembles or RCA.

## 2.3. Selective prediction

Beyond binary reliability estimation, selective classification with a reject option [12, 13, 14] formalizes the risk–coverage trade-of between prediction coverage and error rate; RBQE’s continuous agreement score is evaluated under this same framework in Section 5.6.

Table 1: Summary of RBQE agreement features.
<table><tr><td>Feature</td><td>Captures</td><td>Range</td><td>Interpretation</td></tr><tr><td>Agreement Dice</td><td>Regional overlap</td><td>[0, 1]</td><td>Higher = stronger agreement</td></tr><tr><td>Agreement IoU</td><td>Regional overlap (stricter)</td><td>[0, 1]</td><td>Monotonic in Dice; comparability only</td></tr><tr><td>Area Ratio</td><td>Size consistency</td><td>[0, 1]</td><td>Higher = similar predicted extent</td></tr><tr><td>Boundary Agreement</td><td>Contour consistency</td><td>[0, 1]</td><td>Higher = aligned boundaries</td></tr><tr><td>Centroid Distance</td><td>Spatial localization</td><td>≈ [0, 1]</td><td>Lower = closer agreement</td></tr></table>

## 3. The RBQE Framework

## 3.1. Overview

RBQE estimates reliability by measuring the agreement between two independently trained segmentation models applied to the same input image. Depending on the referee configuration, the models may share the same architecture but difer in initialization, or may additionally difer in architecture and training procedure. The underlying objective is to reduce the likelihood of systematically correlated failure modes while providing an independent prediction against which the primary model can be checked.

Let I be the input colonoscopy image and $M _ { p }$ be the primary segmentation model. Then

$$
S _ { p } = M _ { p } ( I ) ,\tag{1}
$$

where $S _ { p }$ denotes the binary segmentation mask produced by the primary model, deployed in clinical practice. An independently trained referee model, denoted $M _ { r }$ , processes the same image:

$$
S _ { r } = M _ { r } ( I ) .\tag{2}
$$

RBQE quantifies the consistency between $S _ { p }$ and $S _ { r }$ using a set of complementary agreement descriptors, described in Section 3.2. As illustrated in Fig. 1, the framework consists of four steps: (i) generate a primary segmentation using the deployed model; (ii) obtain an independent prediction from a separately trained referee model; (iii) extract agreement features from the two masks; and (iv) estimate segmentation reliability from the resulting cross-model agreement.

RBQE operates directly on the output masks $S _ { p }$ and $S _ { r } \mathbf { \mathbf { \mathbf { \otimes } } }$ it therefore requires no additional training, no architectural modification of the primary model, and no ground-truth annotations at inference. It is designed as a self-contained, post-hoc reliability module that can, in principle, be attached to any deployed segmentation pipeline producing a binary or probabilistic mask — though this study validates it specifically with the architectures described in Sections 3.3–3.4. The complete evaluation protocol, including benchmark construction, the definition of segmentation failure, and the statistical testing procedures, is described in Section 4.

## 3.2. Agreement descriptors

RBQE characterizes agreement between the primary and referee segmentation masks using descriptors that capture regional overlap, size consistency, boundary similarity, and spatial localization, summarized in Table 1.

Agreement Dice. Agreement Dice measures the overlap between the primary and referee segmentation masks,

$$
D _ { a g r } = \frac { 2 \left| S _ { p } \cap S _ { r } \right| } { \left| S _ { p } \right| + \left| S _ { r } \right| } ,\tag{3}
$$

where |S| denotes the number of foreground pixels in mask S. Larger values indicate stronger regional agreement.

Agreement IoU (remark). Agreement IoU, $I o U _ { a g r } = | S _ { p } \cap S _ { r } | / | S _ { p } \cup S _ { r } | ,$ , is reported for interpretability and comparability with prior segmentation literature. Note that $I o U _ { a g r }$ is a strictly monotonic function of $D _ { a g r }$ for any pair of masks,

$$
I o U _ { a g r } = \frac { D _ { a g r } } { 2 - D _ { a g r } } ,\tag{4}
$$

and therefore the two descriptors yield identical rank orderings, and consequently identical ROC-AUC values for failure detection and identical Spearman correlations with ground-truth segmentation quality (Section 5.2). Agreement IoU is retained alongside Agreement Dice for reader familiarity, rather than as an independent source of evidence; RBQE thus rests on four functionally distinct descriptors.

Area Ratio. To evaluate consistency in predicted lesion size, independent of spatial overlap, the Area Ratio is computed as

$$
A _ { r a t i o } = \frac { \operatorname* { m i n } ( | S _ { p } | , | S _ { r } | ) } { \operatorname* { m a x } ( | S _ { p } | , | S _ { r } | ) } .\tag{5}
$$

This feature is insensitive to small spatial shifts while capturing diferences in foreground extent.

Boundary Agreement. Boundary Agreement evaluates contour consistency between the predicted masks by computing the F1-score between their extracted object boundaries, using a distance tolerance of 3 pixels to determine correspondence between boundary points, following the boundary F-measure formulation [15]. Unlike region-based metrics, this descriptor emphasizes boundary alignment and captures fine contour discrepancies that may not substantially afect overlap-based measures.

Centroid Distance. Spatial consistency is evaluated using the Euclidean distance between the centroids of the predicted masks,

$$
d _ { c } = \sqrt { ( x _ { p } - x _ { r } ) ^ { 2 } + ( y _ { p } - y _ { r } ) ^ { 2 } } ,\tag{6}
$$

where $( x _ { p } , y _ { p } )$ and $( x _ { r } , y _ { r } )$ denote the centroids of the primary and referee segmentations, respectively. Centroid distances are normalized by the image diagonal, yielding a dimensionless quantity approximately within [0, 1] that is independent of image resolution and directly comparable across datasets. Smaller centroid distances indicate greater localization agreement.

Handling of degenerate cases. For images in which the primary or referee mask contains no foreground pixels, agreement features are assigned as follows: Agreement Dice = 0, Agreement IoU = 0, Boundary Agreement = 0, Area Ratio = 0 (1.0 if both masks are empty, reflecting identical zero extent), and Centroid Distance = 1.0 (worst case). Because such degenerate cases are trivially separable from substantive predictions, they motivate the dual full/non-degenerate evaluation protocol defined in Section 4.2.

Together, these agreement descriptors characterize complementary aspects of segmentation consistency — regional overlap, object size, boundary fidelity, and lesion localization — with the exception of the Dice– IoU pair, which are mathematically related rather than independent. Rather than estimating the true Dice score, RBQE interprets the degree of agreement between independent segmentation models as a surrogate indicator of prediction reliability.

## 3.3. Primary segmentation model

YOLOv8n-Seg [16] serves as the primary segmentation model throughout this study; its predictions constitute the candidate segmentations whose reliability is subsequently estimated using RBQE. YOLOv8n-Seg was selected owing to its favorable balance between segmentation accuracy, computational eficiency, and real-time inference capability. The model simultaneously predicts object localization and pixel-wise segmentation masks within a unified architecture, making it suitable for deployment in resource-constrained clinical environments.

The primary segmentation model was trained exclusively on the Kvasir-SEG dataset (1,000 images) [17] using the oficial Ultralytics implementation. The dataset was randomly partitioned into 700 training, 200 validation, and 100 held-out test images (70/20/10 split). Only the training and validation subsets were used during model development: the 700 training images were used for optimization and the 200 validation images were used exclusively for model selection. The 100-image held-out subset was not used for training or model selection and was not included in the external RBQE benchmark. The split was not stratified, as Kvasir-SEG is a single-class binary segmentation dataset. During training, standard data augmentation techniques such as random horizontal flipping, scaling, translation, and color-space augmentation were employed to improve generalization. Images were resized to the input resolution required by the network while preserving the corresponding binary segmentation masks. Unless otherwise specified, all training hyperparameters followed the recommended default configuration provided by the Ultralytics implementation, and the checkpoint achieving the highest validation segmentation performance was retained for all subsequent experiments.

During deployment, each external colonoscopy image was processed once by the primary segmentation model to generate the predicted binary segmentation mask. Importantly, RBQE does not modify or retrain the primary segmentation model. It functions as an independent, deployment-time quality estimation module that operates alongside existing segmentation systems without altering their training or inference procedures. This property is what allows RBQE to be evaluated against multiple referee configurations (Section 3.4) without any corresponding change to the primary model itself.

## 3.4. Referee configurations: independence and architectural diversity

To evaluate whether the proposed agreement signal depends on architectural diversity, model independence, or both, four referee configurations were investigated, arranged along the two design axes introduced in Section 1: one same-architecture, independently trained control; two cross-architecture, independently trained referees; and one cross-architecture, prompt-coupled control. The Independent YOLO Referee, SegFormer-B0, and UNet++ were independently trained on Kvasir-SEG, whereas MedSAM was used as a frozen pretrained foundation model. Because MedSAM requires a prompt derived from the primary prediction, it is treated as a prompt-dependent architectural control rather than as a fully independent referee.

Independent YOLO Referee (independent / same architecture). A second YOLOv8n-Seg model was trained using an architecture and training protocol identical to the primary segmentation model, but with an independent random weight initialization and optimization trajectory. This model shares no weights, training checkpoints, or optimization state with the primary model and was never used to produce the primary segmentation. Its role is exclusively that of a referee at deployment, providing a same-architecture control condition that isolates the contribution of model independence from that of architectural diversity.

SegFormer-B0 (independent / cross-architecture). SegFormer-B0 [18] represents a transformer-based semantic segmentation architecture employing a hierarchical Mix Transformer encoder with a lightweight decoder. Owing to its strong segmentation capability and architectural independence from the primary model, SegFormer-B0 serves as the principal cross-architecture referee throughout this study.

UNet++ (independent / cross-architecture). UNet++ [19] represents a convolutional encoder–decoder architecture with nested dense skip connections. This model is included to investigate whether RBQE remains efective when the referee belongs to a fundamentally diferent, CNN-based segmentation family, distinct from both the primary model and SegFormer-B0.

MedSAM (prompt-coupled / cross-architecture). MedSAM [20] represents a prompt-driven medical foundation model derived from the Segment Anything framework. Since MedSAM requires external prompts during inference — here, a bounding-box prompt derived from the primary model’s predicted mask — its output is not fully independent of the primary model, despite maximal architectural diversity. It is there fore employed to isolate the influence of output-level independence in prompt-based segmentation systems, rather than as a principal cross-architecture referee.

Table 2 summarizes the training configuration for each referee. The Independent YOLO Referee, SegFormer-B0, and UNet++ were each trained from a pretrained initialization on the Kvasir-SEG dataset, following the same training/validation protocol described for the primary model (Section 3.3). SegFormer-B0 and UNet++ were each trained from ImageNet-pretrained encoders using their respective oficial implementations, with hyperparameters selected according to common practice for each architecture. MedSAM was used as a frozen, oficially released pretrained checkpoint, applied directly at inference without any addi tional training or fine-tuning on Kvasir-SEG or the external evaluation datasets. All four referee models were trained or evaluated exclusively using the Kvasir-SEG dataset (for the three trainable referees) or the oficially released pretrained checkpoint (for MedSAM), and none were fine-tuned or adapted on any of the four external evaluation datasets (Section 4.1), consistent with the cross-dataset generalization protocol applied to the primary model.

Table 2: Training configuration for each referee model. MedSAM was used as a frozen pretrained checkpoint and was not trained or fine-tuned on any dataset in this study.
<table><tr><td>Referee</td><td>Architecture</td><td>Pretraining</td><td>Epochs</td><td>Optimizer</td><td>LR</td><td>Batch</td><td>Resolution</td><td>Framework</td></tr><tr><td>Independent YOLO Referee</td><td>YOLOv8n-Seg</td><td>YOLOv8n-Seg pretrained;</td><td>100</td><td>SGD (default)</td><td>default</td><td>16</td><td>640×640</td><td>Ultralytics YOLOv8</td></tr><tr><td>SegFormer-B0</td><td>SegFormer-B0</td><td>independent run ImageNet</td><td>100</td><td>AdamW</td><td>6×10−5</td><td>8</td><td>512×512</td><td>MMSegmentation</td></tr><tr><td>UNet++</td><td>UNet++ (ResNet34)</td><td>ImageNet</td><td>100</td><td>Adam</td><td>1×10-4</td><td>8</td><td>512×512</td><td>segmentation models.pytorch</td></tr><tr><td>MedSAM</td><td>SAM-based foundation model</td><td>Official pretrained checkpoint</td><td></td><td></td><td></td><td></td><td>1024×1024</td><td>Official MedSAM implementation</td></tr></table>

The objective of evaluating these four referee configurations is not to compare segmentation architectures for their own sake, but to determine whether agreement-based quality estimation depends on architectural diversity between the primary and referee models, on independent model training alone, or on a combination of both.

## 4. Experimental Protocol

This section describes the datasets, benchmark construction, evaluation metrics, baselines, and implementation details used to assess the proposed framework. Unless otherwise specified, all experiments were performed using identical preprocessing, evaluation criteria, and deployment settings to ensure fair comparison across referee architectures and competing quality estimation methods.

## 4.1. Datasets

The proposed framework was developed using the Kvasir-SEG dataset [17] and evaluated exclusively on four independent public colonoscopy datasets. Using external datasets for evaluation enables assessment of the framework under realistic domain shifts arising from diferences in imaging equipment, acquisition protocols, patient populations, bowel preparation quality, illumination conditions, and polyp characteristics. The Kvasir-SEG dataset was used solely for training the primary segmentation model and the referee segmentation models; no images from Kvasir-SEG were included in the final evaluation benchmark.

The external benchmark comprises the following publicly available datasets, summarized in Table 3:

• CVC-ClinicDB [21] contains 612 colonoscopy images with manually annotated polyp segmentation masks acquired during routine clinical examinations, exhibiting relatively high image quality and moderate variation in polyp appearance.

• CVC-ColonDB [22] consists of 380 colonoscopy images containing numerous challenging cases characterized by small polyps, irregular boundaries, specular highlights, and complex mucosal textures.

• ETIS-Larib PolypDB [23] contains 196 particularly dificult colonoscopy images with substantial variability in polyp size, shape, illumination, and image quality, widely regarded as one of the most challenging benchmarks for polyp segmentation.

• CVC-300 [24] contains 60 annotated colonoscopy images, commonly used to evaluate model generalization under limited-data conditions.

Combining these four datasets results in a diverse external benchmark containing a broad spectrum of lesion sizes, imaging conditions, anatomical variations, and segmentation dificulty levels. Since none of these datasets were used during model training, they provide an unbiased assessment of deployment-time segmentation reliability under realistic clinical conditions.

Table 3: Summary of the datasets used for training and external evaluation.
<table><tr><td>Dataset</td><td>Images</td><td>Role</td><td>Used for</td></tr><tr><td>Kvasir-SEG</td><td>1,000</td><td>Training</td><td>Primary and referee model development</td></tr><tr><td>CVC-ClinicDB</td><td>612</td><td>Testing</td><td>External benchmark</td></tr><tr><td>CVC-ColonDB</td><td>380</td><td>Testing</td><td>External benchmark</td></tr><tr><td>ETIS-Larib PolypDB</td><td>196</td><td>Testing</td><td>External benchmark</td></tr><tr><td>CVC-300</td><td>60</td><td>Testing</td><td>External benchmark</td></tr></table>

## 4.2. Benchmark construction and evaluation subsets

For standalone segmentation evaluation, all 1,248 external images (Section 4.1) were retained to provide an unbiased assessment of the primary segmentation model. RBQE, however, estimates segmentation reliability by analyzing the agreement between the primary segmentation model and an independently trained referee model, and its evaluation therefore uses two nested definitions, both fixed a priori and applied uniformly.

Full standardized benchmark (N = 1,223). During benchmark construction, 25 images were identified in which both the primary model and the SegFormer-B0 referee — the first referee evaluated — produced empty segmentation masks. Since these cases contain no segmented foreground in either mask, they represent degenerate cases that provide no meaningful information regarding the ability of RBQE to distinguish reliable from unreliable segmentations; including them would artificially inflate agreement-based quality estimation metrics while contributing no evidence regarding segmentation correctness. These 25 images were therefore excluded, resulting in a standardized benchmark of 1,223 images. This identical, fixed benchmark was subsequently used for all four referee configurations, rather than computing referee-specific exclusion sets, ensuring that all referee models were evaluated on exactly the same images and that ROC-AUC, bootstrap confidence intervals, DeLong tests, and all other evaluation metrics remain directly comparable across referees without confounding diferences in sample selection. Unless explicitly stated otherwise, all reported RBQE performance metrics correspond to this standardized 1,223-image benchmark.

Non-degenerate (restricted) subsets. Because RBQE’s agreement descriptors can be trivially informative when a mask is empty (Section 3.2), headline results are additionally reported on restricted subsets excluding degenerate cases. Empty-mask status is referee-specific, so the restricted subsets difer in composition and size across referees. For the SegFormer-B0 configuration, all images in which either the primary or the referee mask contains no more than a negligible foreground region were excluded; this removed 177 images

— all corresponding to a primary-empty prediction with a non-empty SegFormer-B0 referee mask (177 primary-empty, 0 referee-empty) — yielding a restricted subset of 1,046 images in which both masks contain substantive predicted foreground. For the Independent YOLO Referee, the corresponding restricted subset is its “Both Non-Empty” stratum of 975 images (Section 5.4). Importantly, the 130 “Both Empty” cases in the Independent YOLO stratification are specific to that referee and are not the 25 SegFormer-B0 doubleempty cases excluded during benchmark construction; they remained in the standardized benchmark because benchmark membership was determined once, using the SegFormer-B0 exclusion criterion, before evaluating the other referee configurations.

## 4.3. Failure definition and evaluation metrics

The efectiveness of RBQE was evaluated as a binary segmentation failure detection problem. Given a predicted segmentation mask and its corresponding ground-truth annotation, segmentation quality was quantified using the Dice Similarity Coeficient (DSC),

$$
D S C = \frac { 2 \left| S _ { p } \cap G \right| } { \left| S _ { p } \right| + \left| G \right| } ,\tag{7}
$$

where $S _ { p }$ denotes the predicted segmentation mask and G the corresponding ground-truth mask. This is the same overlap formulation as Agreement Dice (Eq. 3), applied here against ground truth rather than between the primary and referee predictions. Following prior segmentation quality estimation literature [11], a prediction was classified as a segmentation failure when

$$
D S C < 0 . 5 0 .\tag{8}
$$

This threshold serves as the primary operating point throughout the paper, while additional threshold sensitivity analyses (Section 5.4) assess robustness under alternative failure definitions.

Since RBQE produces a continuous reliability score rather than a binary prediction, its discriminative ability was primarily assessed using the Area Under the Receiver Operating Characteristic Curve (ROC-$\mathrm { A U C } )$ , selected because it is threshold-independent and measures the ability of the quality estimation method to distinguish reliable from unreliable segmentations across all possible operating thresholds. For thresholddependent evaluation, binary decisions were obtained using the Youden-optimal operating threshold [25] determined from the corresponding ROC curve. Performance was subsequently quantified using

$$
{ \mathrm { A c c u r a c y } } = { \frac { T P + T N } { T P + T N + F P + F N } } ,\tag{9}
$$

$$
{ \mathrm { P r e c i s i o n } } = { \frac { T P } { T P + F P } } , \qquad { \mathrm { R e c a l l } } = { \frac { T P } { T P + F N } } ,\tag{10}
$$

$$
F _ { 1 } = 2 \times \frac { \mathrm { P r e c i s i o n } \times \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } ,\tag{11}
$$

where TP, TN, FP, and FN denote the numbers of true positives, true negatives, false positives, and false negatives, respectively.

To quantify statistical uncertainty, 95% confidence intervals for ROC-AUC were estimated using 1,000 bootstrap resamples drawn with replacement at the image level from the standardized 1,223-image benchmark. Pairwise comparisons between correlated ROC curves were performed using DeLong’s test [26] to determine whether observed diferences between competing quality estimation methods were statistically significant. A significance level of $p < 0 . 0 5$ was adopted throughout the study.

## 4.4. Baselines

Test-Time Augmentation (TTA). For comparison with single-model uncertainty estimation, a TTA baseline was evaluated using the same primary YOLOv8n-Seg model and the standardized external benchmark. Nine inference conditions were considered: horizontal flipping, increased brightness, decreased brightness, increased contrast, decreased contrast, gamma correction $( \gamma = 1 . 1 )$ , CLAHE, Gaussian blur, and the unmodified input image. Each transformed image was processed by the primary segmentation model and the resulting predictions were converted to binary masks. The resulting prediction set was used to derive uncertainty descriptors, including mean predictive entropy, maximum predictive entropy, zero-prediction count, and pixel-level disagreement ratio. The zero-prediction count provided the strongest failure-detection performance among the evaluated TTA descriptors and was therefore used as the TTA reliability signal in the baseline comparison.

Morphology-based SQE. A morphology-based baseline derives a quality signal from geometric properties of the predicted mask alone; among the evaluated geometric descriptors, Circularity achieved the strongest failure-detection performance and is used as the morphology-based baseline.

Both baselines were evaluated using the same failure definition (ground-truth $D S C < 0 . 5 0 )$ , standardized benchmark, ROC-AUC analysis, and statistical evaluation protocol used for RBQE, and were additionally recomputed on the identical 1,046-image restricted subset used for the SegFormer-B0 configuration (Section 5.5).

Table 4: Implementation details.
<table><tr><td>Framework</td><td>PyTorch</td></tr><tr><td>Primary model</td><td>YOLOv8n-Seg</td></tr><tr><td>Referee models</td><td>SegFormer-B0, UNet++, MedSAM, Independent YOLO Referee</td></tr><tr><td>Training dataset</td><td>Kvasir-SEG</td></tr><tr><td>Evaluation datasets</td><td>CVC-ClinicDB, CVC-ColonDB, ETIS-Larib PolypDB, CVC-300</td></tr><tr><td>Operating system GPU</td><td>Windows 11 NVIDIA RTX 4060 (8 GB)</td></tr><tr><td>CPU</td><td>AMD Ryzen 7 7700X</td></tr><tr><td>RAM</td><td>32 GB</td></tr><tr><td>Statistical analysis</td><td>Bootstrap (1,000 resamples), DeLong test</td></tr></table>

## 4.5. Implementation details

All models were implemented using the PyTorch deep learning framework. The primary segmentation model (YOLOv8n-Seg) was trained using the oficial Ultralytics implementation, while SegFormer-B0, UNet++, MedSAM, and the Independent YOLO Referee were implemented using their respective oficial or publicly available repositories. Unless otherwise specified, the original network architectures and recommended training configurations were adopted for all models to ensure fair comparison (Table 2).

All input colonoscopy images were resized to the resolution required by the corresponding segmentation model before inference. Predicted probability maps were resampled to the original input image’s resolution prior to thresholding — using bilinear interpolation for the continuous logits (SegFormer-B0, UNet++) and nearest-neighbor resizing for already-binarized outputs (YOLOv8n-Seg) — before being converted to binary segmentation masks using the default inference settings provided by the respective implementations, and these binary masks served as the direct inputs to the proposed RBQE framework. This ensures that all agreement features (Section 3.2) were computed on a common, per-image pixel grid regardless of each model’s native inference resolution (Table 2), rather than across mismatched grids (e.g., 640×640 for the primary model and Independent YOLO Referee, 512×512 for SegFormer-B0 and UNet++, 1024×1024 for MedSAM). Agreement features were computed directly from the binary masks generated by the primary model and each referee model. For each experiment, the continuous agreement score produced by RBQE was interpreted as a segmentation reliability score, with larger values indicating greater confidence in the predicted segmentation.

To ensure reproducibility, the same external benchmark, ground-truth failure definition, agreementfeature extraction procedures, and evaluation metrics were used across all referee architectures and competing quality-estimation methods. All experiments were performed on the workstation configuration summarized in Table 4, with model training and inference accelerated using CUDA.

## 5. Results

Results are organized around four research questions. RQ1: Does cross-model agreement detect segmentation failure, and which agreement descriptors carry the signal (Section 5.2)? RQ2: Does the signal require architectural diversity, or does independent training alone sufice — and what happens when diversity is present but output-level independence is not (Section 5.3)? RQ3: How robust is the signal to the definition of segmentation failure and to degenerate empty-mask cases (Section 5.4)? RQ4: How does RBQE compare with representative existing no-reference SQE paradigms (Section 5.5)? Sections 5.6 and 5.7 then examine deployment characteristics — selective prediction and computational cost — and qualitative behavior. Section 5.1 first characterizes the primary model itself.

## 5.1. External segmentation performance of the primary model

Since RBQE estimates the reliability of predicted segmentations rather than improving segmentation accuracy itself, it must be evaluated under realistic conditions where both successful and failed predictions naturally occur. We therefore first characterize the segmentation performance of the deployed primary model on previously unseen data. The primary YOLOv8n-Seg model was trained exclusively on the Kvasir-SEG dataset and evaluated without any fine-tuning on the four independent external datasets, which exhibit considerable variation in imaging conditions, lesion appearance, polyp size, illumination, and acquisition protocols. Table 5 summarizes the segmentation performance on each external dataset.

Table 5: External segmentation performance of YOLOv8n-Seg on unseen datasets.
<table><tr><td>Dataset</td><td>Dice</td><td>IoU</td><td>Precision</td><td>Recall</td></tr><tr><td>CVC-ClinicDB</td><td>0.766</td><td>0.699</td><td>0.773</td><td>0.802</td></tr><tr><td>CVC-ColonDB</td><td>0.675</td><td>0.607</td><td>0.673</td><td>0.698</td></tr><tr><td>ETIS-Larib PolypDB</td><td>0.609</td><td>0.557</td><td>0.587</td><td>0.651</td></tr><tr><td>CVC-300</td><td>0.828</td><td>0.749</td><td>0.769</td><td>0.906</td></tr><tr><td>Weighted mean (1,248 images)</td><td>0.717</td><td>0.651</td><td>0.713</td><td>0.752</td></tr></table>

Table 6: Failure-detection performance and correlation with ground-truth quality for individual agreement descriptors, using SegFormer-B0 as the referee on the full standardized 1,223-image benchmark. Threshold-dependent metrics use the Youden-optimal operating threshold. The best-performing descriptor is highlighted in bold.
<table><tr><td>Agreement signal</td><td>ROC-AUC</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td><td>Spearman ρ</td></tr><tr><td>Agreement Dice</td><td>0.960</td><td>0.926</td><td>0.762</td><td>0.894</td><td>0.823</td><td>0.734</td></tr><tr><td>Agreement IoUª</td><td>0.960</td><td>0.926</td><td>0.762</td><td>0.894</td><td>0.823</td><td>0.734</td></tr><tr><td>Centroid Distance</td><td>0.951</td><td>0.925</td><td>0.744</td><td>0.881</td><td>0.807</td><td>-0.620</td></tr><tr><td>Boundary Agreement</td><td>0.949</td><td>0.923</td><td>0.738</td><td>0.873</td><td>0.800</td><td>0.622</td></tr><tr><td>Area Ratio</td><td>0.463</td><td>0.804</td><td>0.493</td><td>0.356</td><td>0.413</td><td>0.131</td></tr></table>

<sup>a</sup> Identical to Agreement Dice by construction (Eq. 4); reported for comparability only.

As expected, segmentation performance varies across the four datasets owing to their difering levels of dificulty. Images containing small polyps, irregular object boundaries, poor illumination, specular reflections, motion blur, and complex mucosal textures are generally more challenging than images containing well-defined lesions acquired under favorable imaging conditions. Consequently, the external benchmark contains a broad spectrum of segmentation quality, ranging from highly accurate predictions to complete segmentation failures. This diversity is essential for evaluating deployment-time quality estimation: a benchmark with uniformly accurate predictions would make quality estimation a trivial task, whereas the mix of successful and failed segmentations here provides a meaningful test bed for cross-model agreement.

## 5.2. RQ1: Agreement as a failure-detection signal, and descriptor behavior

RBQE was first evaluated using SegFormer-B0 as the referee model, selected for its strong segmentation performance, architectural independence from the YOLOv8n-Seg primary model, and favorable computational eficiency. For every image in the standardized 1,223-image benchmark, both the primary segmentation model and the independently trained SegFormer-B0 referee generated binary segmentation masks. The agreement descriptors described in Section 3.2 were computed from the two predicted masks, and failuredetection performance was assessed for each descriptor individually according to the protocol in Section 4.3, alongside the Spearman rank correlation of each descriptor with ground-truth segmentation quality (DSC). Table 6 summarizes both analyses.

As expected from Eq. 4, Agreement Dice and Agreement IoU produced identical rankings and therefore identical ROC-AUC and Spearman correlation values. Centroid Distance and Boundary Agreement each provide strong failure-detection signals close to the overlap-based descriptors. Area Ratio, by contrast, achieves a ROC-AUC of only 0.463 — near chance level — indicating that predicted lesion size alone, independent of spatial overlap or location, is a weak and unreliable indicator of segmentation failure at the primary $D S C < 0 . 5 0$ operating point; given this near-chance performance, Area Ratio is not recommended as a standalone reliability descriptor in deployment settings.

The Spearman analysis is consistent with this ranking: Agreement Dice and Agreement IoU achieve the strongest positive correlation with ground-truth DSC $\left( \rho = 0 . 7 3 4 \right)$ , followed by Boundary Agreement $( \rho = 0 . 6 2 2 )$ Centroid Distance exhibits a strong negative correlation $( \rho = - 0 . 6 2 0 )$ , consistent with its interpretation as an error measure rather than an agreement measure: lower centroid distance corresponds to higher segmentation quality. Area Ratio shows only weak positive correlation $( \rho = 0 . 1 3 1 )$ , consistent with its near-chance ROC-AUC. A segmentation can reproduce a lesion’s approximate size while still being incorrectly localized or poorly delineated at the boundary, so size consistency alone says little about where and how the lesion is delineated; overlap- and boundary-based measures therefore serve as the primary reliability signal, while size- and location-based descriptors ofer weaker but complementary information in specific failure regimes. RBQE’s descriptors are thus not interchangeable.

Based on this comparison, Agreement Dice is adopted as the primary descriptor for all subsequent analyses. Fig. 2a presents its ROC curve, and Fig. 2b the confusion matrix at the Youden-optimal operating threshold, illustrating the distribution of correctly identified successful segmentations, correctly identified failures, false alarms, and missed failures. Fig. 3 shows the association between the Agreement Dice value and the corresponding ground-truth Dice score for all images in the standardized benchmark: images with higher Agreement Dice values generally exhibit more accurate segmentations, whereas low agreement values are predominantly associated with segmentation failures. Although some overlap exists between successful and unsuccessful cases, reflecting the inherent variability of medical image segmentation, the overall trend is consistent with the strong discriminative performance in Table 6. The corresponding non-degenerate (restricted-subset) performance of Agreement Dice is reported alongside the referee comparison in Sec tion 5.3.

Overall, cross-model consistency provides an informative surrogate indicator of segmentation reliability: strong agreement between the independently trained primary and referee models generally corresponds to successful segmentation, while substantial disagreement is associated with failure. Unlike methods that derive confidence from repeated evaluation of a single model, RBQE estimates reliability from consistency between independent predictions.

## 5.3. RQ2: Independence versus architectural diversity

An open question is whether the observed agreement principle requires architectural diversity between the primary and referee models, or whether independent model training alone — without any architectural diference — is suficient to produce a useful reliability signal. RBQE was therefore evaluated using the four referee configurations of Section 3.4. For each configuration, the complete RBQE pipeline remained unchanged; only the referee segmentation model was replaced, while identical agreement features, failure definitions, evaluation metrics, and experimental settings were maintained throughout. Consequently, any observed performance diferences can be attributed solely to the characteristics of the referee model rather than to modifications of the quality estimation framework.

Table 7 presents the head-to-head comparison, using a fixed Agreement Dice descriptor applied uniformly across all referees rather than selecting the best-performing descriptor independently for each (the corresponding best-of-five descriptor results are reported later in this section as a sensitivity analysis; Table 8). The table consolidates, for each referee: full-benchmark ROC-AUC with bootstrap 95% confidence intervals (1,000 resamples), the restricted-subset ROC-AUC on the referee’s non-degenerate subset (Section 4.2), the paired DeLong comparison against SegFormer-B0 using matched Agreement Dice scores, and threshold-dependent metrics at the Youden-optimal operating point.

Independence alone sufices; diversity adds a significant increment. The Independent YOLO Referee (samearchitecture control, $\mathrm { R O C - A U C = 0 . 9 2 3 } )$ is outperformed by both SegFormer-B0 (0.960) and UNet++ (0.938), while exceeding MedSAM (0.863). The DeLong analysis confirms that SegFormer-B0 significantly outperforms both the Independent YOLO Referee $( p < 0 . 0 0 1 )$ and $\mathrm { U N e t } + + \left( p = 4 . 2 9 \times 1 0 ^ { - 3 } \right)$ , establishing that the improvement associated with cross-architecture diversity is statistically significant rather than arising from random sampling variation. UNet++ also achieves a higher ROC-AUC than the same-architecture control, although a direct DeLong comparison between UNet++ and the Independent YOLO Referee was not performed. These results support the interpretation that independent training alone provides a useful reliability signal, while architectural diversity can provide additional discriminative strength when the referee remains genuinely independent of the primary model.

Table 7: Primary head-to-head referee comparison using a fixed Agreement Dice descriptor, applied uniformly across all four referees. Full-benchmark results on the standardized 1,223-image benchmark, with bootstrap 95% confidence intervals (1,000 resamples); restricted results on each referee’s non-degenerate subset (Section 4.2); DeLong p-values from paired comparisons against SegFormer-B0 using matched Agreement Dice scores. Threshold-dependent metrics (Acc., Prec., Rec., F1) at the Youden-optimal threshold on the full benchmark.
<table><tr><td>Referee</td><td>Category</td><td>ROC-AUC full [95% CI]</td><td>ROC-AUC restricted (N)</td><td>DeLong p vs. SegFormer</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>SegFormer-B0</td><td>Cross-arch.</td><td>0.960 [0.947, 0.972]</td><td>0.876 (1,046)</td><td>ref.</td><td>0.926</td><td>0.762</td><td>0.894</td><td>0.823</td></tr><tr><td>UNet++</td><td>Cross-arch.</td><td>0.938 [0.925, 0.951]</td><td></td><td> $4 . 2 9 \times 1 0 ^ { - 3 }$ </td><td>0.863</td><td>0.590</td><td>0.941</td><td>0.725</td></tr><tr><td>Independent YOLO</td><td>Same-arch. control</td><td>0.923 [0.902, 0.942]</td><td>0.783 (975)</td><td> $6 . 3 2 \times 1 0 ^ { - 5 }$ </td><td>0.908</td><td>0.710</td><td>0.881</td><td>0.786</td></tr><tr><td>MedSAMa</td><td>Cross-arch. (prompt-coupled)</td><td>0.863 [0.826, 0.902]</td><td></td><td> $\mathrm { n / a ^ { b } }$ </td><td>0.952</td><td>1.000</td><td>0.750</td><td>0.857</td></tr></table>

<sup>a</sup> MedSAM’s Agreement Dice and Centroid Distance thresholds (≈0 and 1.0, respectively) happen to flag the identical 177 images as unreliable (TP=177, FN=59, FP=0, TN=987), despite difering ROC-AUCs (0.863 vs. 0.873)—because a badly failed prompt derived mask tends to diverge in overlap and centroid together. Area Ratio and Boundary Agreement give distinct confusion matrices under the same pipeline, ruling out a computation artifact.  
<sup>b</sup> MedSAM was excluded from the paired DeLong comparison: its prompt-dependent inference precludes an independent paired signal.

Diversity without output-level independence fails. MedSAM is a partial exception to this trend: despite being the most architecturally distinct of the four referees, it achieved the lowest ROC-AUC (0.863), underperforming even the same-architecture YOLO control. As established in Section 3.4, this is attributable to MedSAM’s prompt-driven inference paradigm: because its segmentation prompt is derived from the primary model’s own predicted mask, its output is not fully independent of the primary model, despite the substantial architectural diference between a promptable foundation model and a convolutional detector. This finding refines the central hypothesis of this work: architectural diversity is beneficial only insofar as it is accompanied by genuine output-level independence; a referee that is architecturally distinct but functionally coupled to the primary model’s predictions provides a weaker signal than a same-architecture referee with no such coupling.

Restricted-subset (non-degenerate) performance. To directly address whether the headline SegFormer-B0 result is inflated by trivially separable empty-mask cases, the failure-detection evaluation was repeated on the 1,046-image restricted subset in which both masks contain substantive predicted foreground (Section 4.2). On this subset, Agreement Dice achieves an ROC-AUC of 0.876, compared with 0.960 on the full standardized benchmark. The 177 excluded images correspond overwhelmingly to genuine segmentation failures (groundtruth DSC ≈ 0 for all excluded cases). Repeating this restricted evaluation for the Independent YOLO Referee on its own 975-image “Both Non-Empty” subset yields an ROC-AUC of 0.783, compared with 0.923 on the full standardized benchmark — a proportionally larger reduction than observed for SegFormer-B0. This indicates that the same-architecture control referee’s discriminative power is more strongly concentrated in the comparatively easy empty-mask cases than the cross-architecture SegFormer-B0 referee, consistent with the broader finding that cross-architecture diversity provides additional discriminative strength beyond model independence alone. We therefore report both full-benchmark and restricted figures explicitly for each referee: 0.960 and 0.923 as the headline results on the full standardized benchmark, and 0.876 and 0.783 as conservative, restricted estimates of RBQE’s discriminative performance when the comparatively trivial empty-mask cases are excluded.

Descriptor sensitivity: an upper-bound comparison. The head-to-head comparison above deliberately fixes Agreement Dice across all four referees to ensure a fair, uniform comparison. To assess whether refereespecific descriptor selection would alter this ranking, Table 8 reports the best-performing agreement descriptor selected retrospectively for each referee, with bootstrap validation (1,000 resamples). Because the strongest descriptor was chosen after observing performance, these best-of-five ROC-AUCs are not used as the primary head-to-head referee comparison (Table 7); they instead illustrate the upper bound achievable by referee-specific descriptor selection. SegFormer-B0 and the Independent YOLO Referee are unafected, since Agreement Dice is already their best descriptor; UNet++ improves modestly under Area Ratio (0.938 $ 0 . 9 4 3 )$ , and MedSAM improves under Centroid Distance $( 0 . 8 6 3  0 . 8 7 3 )$ . In neither case does descriptorspecific selection change the overall ranking of the four referees established above.

Table 8: Descriptor sensitivity analysis: best-performing agreement descriptor selected retrospectively for each referee, with bootstrap validation (1,000 resamples). Threshold-dependent metrics at the Youden-optimal operating threshold.
<table><tr><td rowspan="2">Referee</td><td rowspan="2">Best descriptor</td><td rowspan="2">ROC-AUC</td><td rowspan="2">Bootstrap mean</td><td rowspan="2">95% CI</td><td rowspan="2">Acc.</td><td rowspan="2">Prec.</td><td rowspan="2">Rec.</td><td rowspan="2">F1</td></tr><tr><td></td></tr><tr><td>SegFormer-B0</td><td>Agreement Dice</td><td>0.960</td><td>0.960</td><td>[0.947, 0.972]</td><td>0.926</td><td>0.762</td><td>0.894</td><td>0.823</td></tr><tr><td>UNet++</td><td>Area Ratio</td><td>0.943</td><td>0.942</td><td>[0.925, 0.956]</td><td>0.883</td><td>0.639</td><td>0.907</td><td>0.750</td></tr><tr><td>Independent YOLO Referee</td><td>Agreement Dice</td><td>0.923</td><td>0.923</td><td>[0.902, 0.942]</td><td>0.908</td><td>0.710</td><td>0.881</td><td>0.786</td></tr><tr><td>MedSAM</td><td>Centroid Distance</td><td>0.873</td><td>0.873</td><td>[0.840, 0.908]</td><td>0.952</td><td>1.000</td><td>0.750</td><td>0.857</td></tr></table>

Table 9: Performance of the proposed RBQE framework using the Independent YOLO Referee under diferent segmentation failure definitions, using the Agreement Dice descriptor.
<table><tr><td>Failure definition</td><td>ROC-AUC</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td> $D S C < 0 . 3 0$ </td><td>0.940</td><td>0.926</td><td>0.729</td><td>0.910</td><td>0.809</td></tr><tr><td> $D S C < 0 . 4 0$ </td><td>0.937</td><td>0.911</td><td>0.693</td><td>0.914</td><td>0.788</td></tr><tr><td> $D S C < 0 . 5 0$ </td><td>0.923</td><td>0.908</td><td>0.710</td><td>0.881</td><td>0.786</td></tr><tr><td> $D S C < 0 . 6 0$ </td><td>0.917</td><td>0.910</td><td>0.747</td><td>0.859</td><td>0.799</td></tr><tr><td> $D S C < 0 . 7 0$ </td><td>0.911</td><td>0.901</td><td>0.782</td><td>0.801</td><td>0.791</td></tr></table>

## 5.4. RQ3: Robustness to the failure definition and to empty-mask cases

Robustness to diferent failure definitions. The principal experiments define a segmentation failure as a prediction with $D S C < 0 . 5 0$ . Although this operating point has been widely adopted in the segmentation quality estimation literature, the practical definition of a segmentation failure may vary depending on the clinical application: more conservative deployment scenarios may require higher segmentation quality, whereas other applications may tolerate lower segmentation accuracy. To investigate whether RBQE remains efective under diferent operating conditions, additional experiments were performed using multiple Dice thresholds for defining segmentation failure: 0.30, 0.40, 0.50, 0.60, and 0.70, with all other settings held fixed. The analysis used the Independent YOLO Referee, as the same-architecture control condition, and the Agreement Dice descriptor throughout, so that only the failure threshold was varied. Table 9 summarizes the results, and Fig. 4 illustrates the variation in ROC-AUC across the evaluated failure definitions.

Overall, the proposed framework exhibits stable performance across multiple failure thresholds, indicating that the agreement-based reliability signal is not restricted to a particular operating point. Lower Dice thresholds primarily identify severe segmentation failures; higher thresholds progressively include moderatequality segmentations, making the classification problem harder as the distinction between successful and failed segmentations becomes less pronounced. Nevertheless, RBQE continues to provide meaningful discrimination across the evaluated range, confirming that its efectiveness does not depend on a single empirically selected Dice threshold and remains applicable under diferent clinical requirements.

Stratification by empty-mask status. To confirm that the framework’s overall discriminative performance is not disproportionately driven by degenerate cases, the standardized 1,223-image benchmark, evaluated with the Independent YOLO Referee, was stratified into four mutually exclusive groups based on the emptiness of the primary and referee predictions (Table 10, Fig. 5). The four groups sum to exactly 1,223 images, confirming complete coverage of the benchmark. As expected, both “Both Empty” and “Primary Empty Only” cases correspond to complete segmentation failure (100% failure rate), since an empty primary prediction cannot achieve meaningful overlap with a non-empty ground-truth lesion. The overwhelming majority of the benchmark (975 of 1,223 images, 79.7%) falls into the “Both Non-Empty” group, where both models produced a non-trivial prediction; within this group, the true segmentation failure rate is only 4.82%. Trivial empty-mask agreement is therefore not the dominant composition of the benchmark, and the restrictedsubset results already reported in Section 5.3 quantify RBQE’s discriminative performance on precisely this non-degenerate regime.

Table 10: Groupwise failure rate stratified by empty-mask status (Independent YOLO Referee, 1,223-image benchmark).
<table><tr><td>Group</td><td>Images</td><td>Failure rate</td></tr><tr><td>Both Empty</td><td>130</td><td>100.0%</td></tr><tr><td>Primary Empty Only</td><td>47</td><td>100.0%</td></tr><tr><td>Referee Empty Only</td><td>71</td><td>16.9%</td></tr><tr><td>Both Non-Empty</td><td>975</td><td>4.82%</td></tr></table>

Table 11: Comparison of RBQE (SegFormer-B0 referee) against representative no-reference quality estimation paradigms, evaluated on the full standardized 1,223-image benchmark and recomputed on the identical 1,046- image restricted (non-degenerate) subset.
<table><tr><td>Method</td><td>ROC-AUC, full (N = 1,223)</td><td>ROC-AUC, restricted (N = 1,046)</td></tr><tr><td>Morphology-based SQE</td><td>0.927</td><td>0.709</td></tr><tr><td>Test-Time Augmentation (TTA)a</td><td>0.905</td><td>0.621</td></tr><tr><td>RBQE (SegFormer-B0)</td><td>0.960</td><td>0.876</td></tr></table>

<sup>a</sup> TTA reaches ROC-AUC = 0.914 on the pre-exclusion 1,248-image benchmark (provenance only, Section 4.2); the value shown here (0.905) uses the standardized 1,223-image benchmark, matching the protocol applied to RBQE and the morphology baseline throughout.

## 5.5. RQ4: Comparison with existing no-reference SQE methods

Having established RBQE’s efectiveness across referee configurations, we position the proposed framework against representative existing paradigms for no-reference segmentation quality estimation: morphologybased descriptors, which derive a quality signal from geometric properties of the predicted mask alone, and Test-Time Augmentation, which derives an uncertainty signal from a single model’s stochastic behavior across repeated inference (Section 4.4). Both baselines were evaluated under the identical protocol, on the same standardized 1,223-image benchmark, and additionally on the identical 1,046-image restricted subset used for the SegFormer-B0 configuration. Table 11 and Fig. 6 present the comparison.

On the full standardized benchmark, RBQE with the SegFormer-B0 referee achieved a ROC-AUC of 0.960, compared with 0.905 for the TTA baseline and 0.927 for the best-performing morphology-based baseline (Circularity), exceeding the protocol-matched TTA baseline by 0.055 ROC-AUC and the morphologybased baseline by 0.033 ROC-AUC. These results are consistent with the central hypothesis of this work: single-model uncertainty signals such as TTA remain bound to the calibration and stochastic behavior of the model being interrogated, whereas RBQE derives its reliability signal from agreement between two independently trained segmentation models. The morphology-based baseline demonstrates that geometric properties of the predicted mask contain useful information about segmentation quality; however, such descriptors cannot identify failures in which a predicted mask remains geometrically plausible while being incorrectly localized or delineated relative to the true lesion.

The restricted-subset comparison is more discriminating still. Both baselines degrade substantially more than RBQE when trivially separable empty-mask cases are excluded: morphology-based SQE falls from

Table 12: Risk–coverage analysis: mean DSC of retained predictions as coverage decreases, ranked by Agreement Dice (Inde pendent YOLO Referee, 1,223-image benchmark).
<table><tr><td>Coverage</td><td>Mean DSC</td></tr><tr><td>1.00</td><td>0.731</td></tr><tr><td>0.90</td><td>0.787</td></tr><tr><td>0.80</td><td>0.861</td></tr><tr><td>0.70</td><td>0.888</td></tr><tr><td>0.60</td><td>0.902</td></tr><tr><td>0.50</td><td>0.911</td></tr><tr><td></td><td>0.917</td></tr><tr><td>0.40</td><td></td></tr><tr><td>0.30</td><td>0.927</td></tr><tr><td>0.20</td><td>0.934</td></tr><tr><td>0.10</td><td>0.945</td></tr></table>

0.927 to 0.709, and TTA falls from 0.905 to 0.621, whereas RBQE (SegFormer-B0) falls from 0.960 to 0.876. On the identical, protocol-matched restricted subset, RBQE therefore exceeds the morphology baseline by 0.167 ROC-AUC and TTA by 0.255 ROC-AUC — a substantially wider margin than on the full benchmark (0.033 and 0.055, respectively). This pattern is consistent with both baselines’ comparative reliance on the same easily detected empty-mask cases that RBQE’s restricted evaluation deliberately excludes: circularity is undefined or trivially informative for an empty mask, and TTA’s zero-prediction-count signal is, by construction, maximally informative precisely when the primary model produces no prediction at all. RBQE’s agreement-based signal, by contrast, retains substantial discriminative power even when restricted to cases where the primary model does produce a non-trivial prediction. (The Independent YOLO Referee’s restricted result, 0.783 on its own 975-image subset, was not benchmarked against the restricted baselines, since its restricted subset difers in composition from SegFormer-B0’s; the comparison in Table 11 is specific to the headline SegFormer-B0 configuration.)

These results indicate that agreement-based reliability estimation is not merely competitive with, but measurably stronger than, both a purely geometric baseline and a representative single-model uncertainty method, while requiring only one additional referee forward pass at deployment.

## 5.6. Deployment characteristics: selective prediction and computational cost

Risk–coverage analysis. The preceding sections evaluate RBQE as a binary failure detector at a fixed operating threshold. In practice, RBQE’s continuous agreement score also supports selective prediction: rather than accepting every prediction, a deployment system can reject the lowest-agreement predictions for expert review, retaining only those above a chosen confidence level. We evaluate this use case via a risk–coverage analysis, performed using the Independent YOLO Referee to assess whether the continuous agreement signal supports selective prediction under the same-architecture control condition: predictions are ranked by Agreement Dice and progressively rejected from lowest to highest agreement, reporting the mean DSC of the retained predictions at each coverage level. Fig. 7 and Table 12 present the complete risk–coverage curve.

Mean DSC increases monotonically as coverage decreases, from 0.731 at full coverage to 0.945 when only the top 10% of predictions by Agreement Dice are retained. The full-coverage mean DSC of 0.731 is higher than the overall mean DSC of 0.717 reported for the 1,248-image external benchmark in Table 5 because the risk–coverage analysis uses the standardized 1,223-image RBQE benchmark after exclusion of the 25 images removed during benchmark construction (Section 4.2). At a coverage of 0.50 — rejecting the least reliable half of predictions for review — the mean DSC of the retained predictions reaches 0.911, representing a substantial improvement over the full-coverage value. This confirms that Agreement Dice is not only an efective binary failure detector, but also a useful continuous ranking signal: predictions it deems more reliable are, on average, genuinely of higher quality, supporting its use as a practical triage mechanism in clinical deployment where a proportion of predictions can be flagged for expert review rather than accepted automatically.

Table 13: Computational and deployment comparison of no-reference segmentation quality estimation methods.
<table><tr><td>Method</td><td>Additional model</td><td>Forward passes</td><td>Quality labels required</td><td>Deployment overhead</td></tr><tr><td>Morphology-based SQE</td><td>No</td><td>1</td><td>No</td><td>Very low</td></tr><tr><td>Test-Time Augmentation</td><td>No</td><td>Multiple</td><td>No</td><td>High</td></tr><tr><td>RBQE (Independent YOLO Referee)</td><td>Yes</td><td>2</td><td>No</td><td>Moderate</td></tr><tr><td>RBQE (SegFormer-B0)</td><td>Yes</td><td>2</td><td>No</td><td>Moderate</td></tr><tr><td>RBQE (UNet++)</td><td>Yes</td><td>2</td><td>No</td><td>Moderate</td></tr><tr><td>RBQE (MedSAM)</td><td>Yes</td><td>2</td><td>No</td><td>High</td></tr></table>

Computational complexity and deployment feasibility. Practical deployment of segmentation quality estimation methods also requires consideration of computational eficiency and inference cost. Unlike uncertainty estimation methods based on TTA, which require multiple forward passes through the same segmentation model, RBQE performs only a single forward pass using the primary model and one additional forward pass using an independently trained referee model. The agreement descriptors are subsequently computed using simple, deterministic geometric operations on binary segmentation masks, introducing negligible computational cost relative to the segmentation inference itself and requiring no optimization or additional learnable parameters during deployment.

Table 13 summarizes the computational and practical deployment characteristics of the evaluated quality estimation approaches, including all four RBQE referee configurations; Fig. 8 illustrates the relationship between failure-detection performance and inference forward-pass burden. Overhead is reported qualitatively, in terms of the number and nature of forward passes required, rather than as hardware-specific latency, since absolute inference time depends on GPU, batch size, precision, and implementation details that were not held constant across methods in this study.

Among the evaluated referee models, the Independent YOLO Referee, SegFormer-B0, and UNet++ require one additional forward pass using lightweight segmentation architectures. In contrast, MedSAM operates as a frozen foundation model at a substantially higher input resolution (1024 × 1024), resulting in greater inference cost despite eliminating the need for referee training. The choice of referee architecture therefore provides flexibility in balancing accuracy and computational eficiency according to application requirements: lightweight, same-architecture referees may be preferred for real-time or resource-constrained systems; cross-architecture referees such as SegFormer-B0 ofer the strongest failure-detection performance at moderate additional cost; and foundation-model referees such as MedSAM may be suited to ofline analysis where training data or infrastructure for a dedicated referee is unavailable, despite their higher inference latency. A further practical advantage of RBQE is its modular design: since the referee model operates independently of the primary segmentation network, existing segmentation systems can be augmented with reliability estimation without modifying the original segmentation architecture or retraining the deployed model, simplifying integration into clinical workflows and facilitating compatibility with heterogeneous segmentation models developed by diferent research groups or manufacturers.

## 5.7. Qualitative analysis

While quantitative evaluation demonstrates the overall efectiveness of the proposed framework, qualitative analysis provides additional insight into how cross-model agreement relates to segmentation reliability across diferent clinical scenarios. Fig. 9 presents four representative examples, one per row, spanning the range of segmentation outcomes observed in the external benchmark. For each example: (a) the input colonoscopy image, (b) the ground-truth annotation, (c) the primary model’s predicted segmentation, (d) the SegFormer-B0 referee’s predicted segmentation, (e) a visualization of the agreement between the two masks, and (f) the resulting Agreement Dice value are shown.

In the high-quality segmentation case (Fig. 9, first row), the primary and referee models produce highly consistent segmentation masks with substantial overlap and similar boundary localization; consequently, the computed Agreement Dice value is high, correctly reflecting the successful segmentation outcome. The moderately challenging case (second row) exhibits partial disagreement between the two models: although the overall lesion location remains consistent, diferences arise along the object boundary and in the predicted lesion extent, producing an intermediate Agreement Dice value that reflects the increased uncertainty associated with this prediction. In contrast, the failed segmentation case (third row) is characterized by pronounced disagreement between the independently trained models, manifesting as incomplete lesion delineation, inaccurate boundary localization, or a missed polyp; as a result, the Agreement Dice value decreases substantially, enabling RBQE to correctly identify this segmentation as unreliable.

An important observation across these examples is that disagreement does not necessarily arise from identical error patterns. In several challenging cases, one model produces a reasonable segmentation while the other fails, leading to low agreement that serves as an efective indicator of reduced prediction reliability. This behavior demonstrates that RBQE captures complementary information from independent segmentation models rather than relying solely on the confidence of a single predictor.

The fourth row presents a case that RBQE does not handle well, included deliberately for balance. Here the primary and referee models agree closely with each other (Agreement Dice = 0.793) despite both diverging substantially from the ground truth (DSC = 0.467): the two models converge on a similar, but similarly wrong, delineation of the lesion. This is precisely the failure mode anticipated by the framework’s reliance on independent failure modes rather than independent training alone — when two models share a systematic bias toward a particular imaging condition or lesion appearance, their agreement can no longer be assumed to track correctness. We return to this limitation in Section 7.

## 6. Discussion

The experiments presented in this study demonstrate that agreement between independently trained segmentation models provides an efective, low-cost surrogate for segmentation reliability at deployment. Across the agreement descriptors, four referee configurations, and a standardized external benchmark of 1,223 colonoscopy images — with all headline results additionally verified on non-degenerate restricted subsets — cross-model agreement consistently distinguished reliable from unreliable segmentations without requiring access to ground-truth annotations at inference.

Why does cross-model agreement work?. The strongest agreement descriptors in this study — Agreement Dice and Agreement IoU (ROC-AUC = 0.960), followed by Centroid Distance (0.951) and Boundary Agreement (0.949) — are directly derived from spatial and geometric consistency between two separately parameterized segmentation models. Although the models are trained using the same development dataset, they do not share weights, checkpoints, or optimization state, and cross-architecture referees additionally introduce diferent architectural inductive biases. Consequently, agreement between their predictions can provide an additional check on the primary model’s reliability, although shared dataset exposure and systematic biases can still produce correlated errors — as illustrated by the high-agreement failure case in Section 5.7. The results of Section 5.3 give this mechanism a precise formulation: model independence is the necessary foundation of the agreement signal, architectural diversity a further — though not strictly required — source of additional discriminative strength, and output-level coupling (as in MedSAM’s prompt dependence) is suficient to undermine the signal even under maximal architectural diversity.

Positioning relative to existing no-reference approaches. Section 5.5 substantiates, empirically, the taxonomy introduced in Sections 1–2: morphology-based descriptors capture only geometric regularity and cannot detect plausible-looking but incorrectly localized predictions, while TTA, like other single-model uncertainty methods, remains fundamentally bound to the calibration of one model and is therefore vulnerable precisely when that model is confidently wrong — a vulnerability made explicit by the sharp degradation of both baselines on the non-degenerate restricted subset. Unlike RCA [11], which requires fitting a new reverse classifier for every test image against a reference atlas, RBQE requires no per-image retraining and no atlas dependency; unlike deep ensembles, which derive disagreement from same-architecture models difering only in initialization, RBQE’s strongest results arise specifically from cross-architecture referees, indicating that architectural diversity provides genuine additional signal beyond what ensemble-style disagreement alone would ofer.

Clinical relevance. During real-time colonoscopy, ground-truth segmentation is, by definition, unavailable, and learned segmentation accuracy cannot be directly measured at the point of use. RBQE addresses this gap by providing a reference-free, post-hoc reliability signal that can be computed alongside the primary model’s prediction using one additional referee forward pass, without requiring quality-labelled training data. Beyond binary failure detection, the risk–coverage analysis (Section 5.6) shows that rejecting the lowest-agreement half of predictions raises mean retained DSC from 0.731 to 0.911, and retaining only the top 10% raises it further to 0.945: RBQE can therefore operate not only as a binary alarm, but as a triage mechanism, flagging a tunable proportion of the least reliable predictions for expert review while allowing high-agreement predictions to proceed with greater confidence, ofering clinicians and system designers explicit control over the trade-of between automation coverage and diagnostic risk. This positions RBQE not as a replacement for the primary segmentation model, but as an independent safety layer capable of flagging potentially unreliable predictions before they inform diagnostic or therapeutic decisions — a role that is particularly valuable given that segmentation failures are most likely to occur under precisely the challenging imaging conditions (poor illumination, small polyps, domain shift) in which clinicians can least aford an undetected error.

## 7. Limitations

Although the proposed RBQE framework demonstrates strong and consistent performance across referee configurations, failure thresholds, and robustness analyses, several limitations should be acknowledged.

First, this study evaluates RBQE exclusively on binary polyp segmentation. While this represents an important and clinically relevant task, the generalizability of agreement-based reliability estimation to multiclass segmentation problems, or to other medical imaging modalities beyond colonoscopy, remains to be established.

Second, RBQE’s efectiveness depends on the availability of a suitable referee model. Although this study demonstrates that both same-architecture independence and cross-architecture diversity contribute to a useful signal (Section 5.3), deployment in a new clinical setting requires either training an additional referee model or identifying a suitable pretrained one, which represents a practical prerequisite not required by single-model uncertainty methods. Relatedly, the framework’s core assumption — that independently trained models fail in diferent ways — can itself fail, as demonstrated by the high-agreement failure case in Fig. 9 (fourth row). Such cases are, by construction, dificult for any agreement-based method to detect, since the signal RBQE relies on is precisely the one that fails to distinguish them.

Third, the agreement descriptors employed in this study are deterministic geometric measures computed directly from binary masks. While efective, this design does not incorporate probabilistic segmentation outputs, feature-space similarity, or topology-aware representations, any of which may further improve the sensitivity of the framework to subtler failure modes.

Fourth, this evaluation, although comprising 1,223 external images across four independent public benchmarks, is retrospective. Prospective validation on multi-center clinical data, with variability in acquisition hardware and patient populations beyond that captured by the public benchmarks used here, is necessary before routine clinical deployment.

Finally, RBQE estimates the reliability of a segmentation prediction as a whole, but does not localize which region of a prediction is unreliable. Combining agreement-based reliability estimation with localized, pixel-level uncertainty maps may provide more actionable feedback for clinicians and downstream decisionsupport systems.

## 8. Conclusion

This work introduced Referee-Based Quality Estimation (RBQE), a deployment-time, reference-free framework that estimates segmentation reliability through agreement between a primary segmentation model and an independently trained referee, requiring no ground-truth annotations, no quality-labelled training data, and no modification of the deployed segmentation model.

A central finding of this study is the explicit separation of model independence from architectural diversity as contributors to the agreement signal. The Independent YOLO Referee — identical in architecture to the primary model, difering only in initialization — achieved a ROC-AUC of 0.923, confirming that independence alone is suficient for a meaningful reliability signal. Cross-architecture referees provided a further, statistically significant improvement, with SegFormer-B0 achieving the strongest overall performance (0.960). This distinction, together with the identification of MedSAM’s prompt-dependence as a case where architectural diversity does not translate into genuine output-level independence, provides a more precise account of why agreement-based quality estimation works than has previously been established in the literature.

Across a standardized external benchmark of 1,223 colonoscopy images spanning four independent public datasets, RBQE’s performance was shown to be robust to the choice of failure threshold and to the choice of referee architecture. Discriminative performance declines when trivial empty-mask agreement cases are excluded (ROC-AUC falls from 0.960 to 0.876 for SegFormer-B0, and from 0.923 to 0.783 for the Independent YOLO Referee, when each is restricted to images with substantive foreground in both the primary and referee masks), confirming that part of the full-benchmark result is attributable to these comparatively easy cases; on the identical restricted subset, however, RBQE’s margin over the evaluated baselines widens rather than narrows. RBQE was further shown to outperform representative morphology-based and TTA-based baselines by 0.033 and 0.055 ROC-AUC respectively under an identical evaluation protocol (Section 5.5), to support efective selective prediction via risk–coverage analysis, and to require substantially lower computational overhead than TTA-based alternatives, needing only one additional deterministic forward pass at inference.

Taken together, these results support agreement between independently trained segmentation models as a practical, interpretable, and computationally eficient reliability signal for deployment-time segmentation quality estimation, ofering a potentially practical path toward integrating automated polyp segmentation reliability estimation into future colonoscopy workflows. Future work should extend this framework to multiclass and multi-modality segmentation tasks, incorporate richer agreement representations, and pursue prospective multi-center clinical validation.

## Code availability

The implementation of the proposed Referee-Based Quality Estimation (RBQE) framework, including the reproducibility notebook, evaluation scripts, and sample experimental results, is available at: https: //github.com/sidgupta307/RBQE-Referee-Based-Quality-Estimation.

## Data availability

The external evaluation datasets used in this study (CVC-ClinicDB, CVC-ColonDB, ETIS-Larib PolypDB, CVC-300, and Kvasir-SEG) are publicly available third-party datasets, cited in Section 4.1. The code, trained model configurations, and derived experimental results supporting this study are available at the repository referenced in the Code availability section.

## Ethics statement

This study did not involve new data collection from human participants. All colonoscopy image datasets used (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS-Larib PolypDB, CVC-300) are publicly available, pre-existing, de-identified datasets released by their original authors for research use; no new ethics approval was required for this work.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This work was supported by the IITR@175 Fellowship, Indian Institute of Technology Roorkee.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the author(s) used Claude (Anthropic) in order to improve the grammar, language, and readability of the manuscript text. After using this tool, the author(s) reviewed and edited the content as needed and take full responsibility for the content of the publication. All experimental results, analysis, and conclusions are the authors’ own.

## CRediT authorship contribution statement

Siddharth Gupta: Conceptualization, Methodology, Software, Formal analysis, Investigation, Writing – original draft. Jitin Singla: Conceptualization, Supervision, Writing – review and editing.

## References

[1] F. Bray, M. Laversanne, H. Sung, J. Ferlay, R. L. Siegel, I. Soerjomataram, A. Jemal, Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries, CA: A Cancer Journal for Clinicians 74 (3) (2024) 229–263. doi:10.3322/caac.21834.

[2] S. Zhao, S. Wang, P. Pan, T. Xia, X. Chang, X. Yang, L. Guo, Q. Meng, F. Yang, W. Qian, Z. Xu, Y. Wang, Z. Wang, L. Gu, R. Wang, F. Jia, J. Yao, Z. Li, Y. Bai, Magnitude, risk factors, and factors associated with adenoma miss rate of tandem colonoscopy: A systematic review and meta-analysis, Gastroenterology 156 (6) (2019) 1661–1674. doi:10.1053/j.gastro.2019.01.260.

[3] D. Jha, S. Ali, N. K. Tomar, H. D. Johansen, D. Johansen, J. Rittscher, M. A. Riegler, P. Halvorsen, Real-time polyp detection, localization and segmentation in colonoscopy using deep learning, IEEE Access 9 (2021) 40496–40510. doi:10.1109/ACCESS.2021.3063716.

[4] R. Robinson, O. Oktay, W. Bai, V. V. Valindria, M. M. Sanghvi, N. Aung, J. M. Paiva, F. Zemrak, K. Fung, E. Lukaschuk, A. M. Lee, V. Carapella, Y. J. Kim, B. Kainz, S. K. Piechnik, S. Neubauer, S. E. Petersen, C. Page, D. Rueckert, B. Glocker, Real-time prediction of segmentation quality, in: International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2018, pp. 578–585. doi:10.1007/978-3-030-00937-3\_66.

[5] A. Jungo, M. Reyes, Assessing reliability and challenges of uncertainty estimations for medical image segmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2019, pp. 48–56. doi:10.1007/978-3-030-32245-8\_6.

[6] Z. Eaton-Rosen, F. Bragman, S. Bisdas, S. Ourselin, M. J. Cardoso, Towards safe deep learning: Accurately quantifying biomarker uncertainty in neural network predictions, in: International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2018, pp. 691–699. doi:10.1007/978-3-030-00928-1\_78.

[7] Y. Gal, Z. Ghahramani, Dropout as a Bayesian approximation: Representing model uncertainty in deep learning, in: Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016, pp. 1050–1059.

[8] G. Wang, W. Li, M. Aertsen, J. Deprest, S. Ourselin, T. Vercauteren, Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks, Neurocomputing 338 (2019) 34–45. doi:10.1016/j.neucom.2019.01.103.

[9] B. Lakshminarayanan, A. Pritzel, C. Blundell, Simple and scalable predictive uncertainty estimation using deep ensembles, in: Advances in Neural Information Processing Systems (NeurIPS), 2017.

[10] S. Fort, H. Hu, B. Lakshminarayanan, Deep ensembles: A loss landscape perspective, arXiv preprint arXiv:1912.02757 (2019). doi:10.48550/arXiv.1912.02757.

[11] V. V. Valindria, I. Lavdas, W. Bai, K. Kamnitsas, E. O. Aboagye, A. G. Rockall, D. Rueckert, B. Glocker, Reverse classification accuracy: Predicting segmentation performance in the absence of ground truth, IEEE Transactions on Medical Imaging 36 (8) (2017) 1597–1606. doi:10.1109/TMI. 2017.2665165.

[12] C. Chow, On optimum recognition error and reject tradeof, IEEE Transactions on Information Theory 16 (1) (1970) 41–46. doi:10.1109/TIT.1970.1054406.

[13] R. El-Yaniv, Y. Wiener, On the foundations of noise-free selective classification, Journal of Machine Learning Research 11 (2010) 1605–1641.

[14] Y. Geifman, R. El-Yaniv, Selective classification for deep neural networks, in: Advances in Neural Information Processing Systems (NeurIPS), 2017.

[15] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, A. Sorkine-Hornung, A benchmark dataset and evaluation methodology for video object segmentation, in: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 724–732. doi:10.1109/CVPR.2016.85.

[16] G. Jocher, A. Chaurasia, J. Qiu, Ultralytics YOLOv8, https://github.com/ultralytics/ ultralytics, version 8.0.0, released 10 January 2023 (2023).

[17] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, H. D. Johansen, Kvasir-SEG: A segmented polyp dataset, in: International Conference on Multimedia Modeling (MMM), 2020, pp. 451–462. doi:10.1007/978-3-030-37734-2\_37.

[18] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, P. Luo, SegFormer: Simple and eficient design for semantic segmentation with transformers, in: Advances in Neural Information Processing Systems (NeurIPS), 2021.

[19] Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, J. Liang, UNet++: A nested U-Net architecture for medical image segmentation, in: Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (DLMIA), 2018, pp. 3–11. doi:10.1007/978-3-030-00889-5\_1.

[20] J. Ma, Y. He, F. Li, L. Han, C. You, B. Wang, Segment anything in medical images, Nature Communications 15 (2024) 654. doi:10.1038/s41467-024-44824-z.

[21] J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D. Gil, C. Rodríguez, F. Vilariño, WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians, Computerized Medical Imaging and Graphics 43 (2015) 99–111. doi:10.1016/j.compmedimag.2015. 02.007.

[22] J. Bernal, J. Sánchez, F. Vilariño, Towards automatic polyp detection with a polyp appearance model, Pattern Recognition 45 (9) (2012) 3166–3182. doi:10.1016/j.patcog.2012.03.002.

[23] J. Silva, A. Histace, O. Romain, X. Dray, B. Granado, Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer, International Journal of Computer Assisted Radiology and Surgery 9 (2014) 283–293. doi:10.1007/s11548-013-0926-3.

[24] D. Vázquez, J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, A. M. López, A. Romero, M. Drozdzal, A. Courville, A benchmark for endoluminal scene segmentation of colonoscopy images, Journal of Healthcare Engineering 2017 (2017) 4037190. doi:10.1155/2017/4037190.

[25] W. J. Youden, Index for rating diagnostic tests, Cancer 3 (1) (1950) 32–35. doi:10.1002/ 1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3.

[26] E. R. DeLong, D. M. DeLong, D. L. Clarke-Pearson, Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach, Biometrics 44 (3) (1988) 837–845. doi:10.2307/2531595.

![](images/157c8ae6886621ee579a2e34965c1beefa499300309c314b53f47c93f8830431.jpg)  
Figure 1: Overview of the Referee-Based Quality Estimation (RBQE) framework. The primary model’s prediction is compared against one of four referee configurations: an independently trained Independent YOLO Referee (same-architecture control), independently trained cross-architecture referees (SegFormer-B0 and UNet++), or a frozen prompt-driven MedSAM referee. Five agreement descriptors are then computed to obtain a deployment-time segmentation reliability estimate.

![](images/40e244c2f497a2183544cf15616170fff17ffd96f581d20578fdff125473e129.jpg)  
(a)

![](images/4a01aa0f1e0e8b7adf5fac2a2f2d8265b5fc04a2132994ce63459f0fed414584.jpg)  
(b)  
Figure 2: (a) Receiver Operating Characteristic (ROC) curve for Agreement Dice (SegFormer-B0 referee, $N = 1 { , } 2 2 3 )$ , AUC $= { \dot { 0 } } . 9 6 0 .$ The marked point indicates the Youden-optimal operating threshold used for the threshold-dependent metrics in Table 6. (b) Confusion matrix for SegFormer-B0 referee failure detection using Agreement Dice at the Youden-optimal operating threshold (N = 1,223).

![](images/f4271f2d90a12987b3b78d31db46412d5cfbf5729c9f181e13ff9562b7604acd.jpg)  
Figure 3: Spearman rank correlation between each RBQE agreement descriptor and ground- truth DSC (SegFormer-B0 referee, 1,223-image benchmark), corresponding to Table 6.

![](images/53bfe45d4b0b8e41a74b7451cfae63b7d9e7b3b560fde3f5e07ad44787f979e0.jpg)  
Figure 4: Robustness of Agreement Dice failure-detection performance to the definition of segmentation failure. The failure threshold τ is varied from $\bar { D } S C < 0 . 3 0$ to $D S C < 0 . 7 0$ using the Independent YOLO Referee, with $\tau = 0 . 5 0$ representing the primary operating point.

![](images/470b93930a5a80676669d81f5742448e5a989f4fd6bbcaad20ccbd95ce552426.jpg)  
Figure 5: Failure rate stratified by empty-mask status (Independent YOLO Referee, 1,223-image benchmark), corresponding to Table 10.

![](images/9280edd6a71311a2192f1a91cabce9bdda699ed5644a7c48bef82408e7e21873.jpg)  
Figure 6: Comparison of RBQE (SegFormer-B0 referee) against representative no-reference quality estimation paradigms — morphology-based SQE and Test-Time Augmentation (TTA) — evaluated under the identical protocol described in Section 4.3.

![](images/9318963f1caf5ebdd7dccd03d7f548cc560911bf6f47b80d85d34dad6710f6bc.jpg)  
Figure 7: Risk–coverage curve for selective prediction using Agreement Dice (Independent YOLO Referee, 1,223-image benchmark). Rejecting lower-agreement predictions monotonically increases the mean DSC of the retained predictions, from 0.731 at full coverage to 0.945 at 10% coverage.

Failure-Detection Performance vs. Computational Cost  
![](images/db6270c0f16f08729675803b1735fc360b494f7e0bdc1d28378b8283b3dbe97e.jpg)  
Figure 8: Relationship between segmentation failure-detection performance (ROC-AUC) and the total number of inference forward passes required per image by the evaluated deployment-time quality estimation methods. Morphology-based quality estimation requires one primary-model inference, TTA requires nine augmented primary-model inferences, and RBQE requires one primary-model inference plus one referee-model inference.

![](images/b00b50b8962e9e921325809c314e4cfd6452317f7a87af70f62f012a2e02b217.jpg)  
Figure 9: Representative qualitative examples illustrating the behavior of Agreement Dice across successful (image cvc\_520), moderately challenging (image cvc\_69), and failed (image cvc\_545) segmentation cases, together with one additional highagreement failure mode (bottom, image colondb\_286) in which the primary and referee models agree closely (Agreement Dice = 0.793) while both diverge substantially from the ground truth (DSC = 0.467), illustrating a genuine limitation of agreementbased reliability estimation. Cases are drawn from CVC-ClinicDB (rows 1–3) and CVC-ColonDB (row 4). (a) Input image; (b) ground truth; (c) primary model prediction; (d) SegFormer-B0 referee prediction; (e) agreement visualization (green = both models agree, blue = primary only, magenta = referee only).