# Anatomical Grounding and Leakage-Aware Multimodal Contrastive Learning for Alzheimer’s Disease Classification from Structural MRI

Paul-Gabriel Nicolae<sup>1</sup>, Irina Mocanu<sup>1</sup>, for the Alzheimer’s Disease Neuroimaging Initiative<sup>∗</sup>

<sup>1</sup>Computer Science and Engineering Department, Faculty of Automatic Control and Computers, National University of Science and Technology POLITEHNICA Bucharest, Romania

## Abstract

Deep networks trained on structural MRI for Alzheimer’s disease (AD) staging often reach reasonable accuracy while attending to anatomically irrelevant regions, and multimodal models that add clinical tables frequently rely on variables that were used to assign the diagnostic label in the first place. We study both issues with a deliberately lightweight slice-based encoder (ResNet18 with a one-layer Transformer over slices) on 1,075 baseline T1-weighted scans from ADNI-1. First, we use FastSurfer segmentations as an anatomical reference: YOLOv8 models trained on segmentation-derived labels localize Alzheimer-relevant structures with mAP above 0.96, and a Grad-CAM comparison shows that the imageonly classifier frequently attends to the skull, orbits and background. Second, we adapt a CLIP-style image–tabular contrastive framework and organize ADNIMERGE variables along a label-leakage spectrum. Fusion with cognitive scores yields 87.3% three-way accuracy, which we treat as a leakage-driven upper bound rather than an imaging result; fusion with regional volumes yields 73.0%. We observe that the choice of contrastive target changes what the image encoder learns: on MCI vs. CN, the image-only head reaches 52.4% when the encoder is aligned to cognitive scores and 73.8% when aligned to volumes, although no tabular input is used at inference. Third, restricting the input to a per-subject crop of the medial temporal lobe raises image-only three-way accuracy from 58.7% to 65.1%. All results come from single runs on a small balanced test set, and we report confidence intervals and the protocol diferences that prevent direct comparison with published numbers.

Keywords: Alzheimer’s disease, structural MRI, multimodal contrastive learning, label leakage, FastSurfer, explainability, ADNI

## 1 Introduction

Alzheimer’s disease (AD) is the most common cause of dementia, and its prodromal stage, mild cognitive impairment (MCI), is the window in which intervention is expected to be most useful. Structural T1-weighted MRI is widely available and captures the atrophy that accompanies the disease, which has made it a standard input for automatic staging into cognitively normal (CN),

MCI and AD. Separating AD from CN on MRI is comparatively easy, because late-stage atrophy is visible; separating MCI from CN is much harder, because baseline diferences are subtle [1].

Two practical problems recur in this literature. The first concerns what the image model looks at. A convolutional network that receives whole slices has no reason to restrict itself to brain tissue, and can exploit contrast at the skull, orbits or ventricle boundaries. The second concerns what multimodal models are given. Large cohorts such as ADNI [2] ship rich clinical tables, and recent image–tabular contrastive frameworks [3, 4] report strong gains when these tables are fused with MRI. However, several table columns, most notably cognitive scores such as the MMSE and CDR, are part of the criteria used to assign the ADNI diagnostic label [5]. High accuracy obtained with these inputs measures how well a model can recover the labelling rule, not how much diagnostic information the image contributes.

In this paper we examine both problems within one lightweight pipeline, a 2D ResNet18 slice encoder aggregated by a single Transformer layer, trained on ADNI-1 baseline scans. We make the following contributions:

1. Anatomical reference and diagnosis of the image model. We use FastSurfer [6] segmentations to derive slice ranges and detection/segmentation labels for Alzheimerrelevant structures, train YOLOv8 models on them, and compare their outputs with Grad-CAM maps of the image classifier (Section 5).

2. A leakage-aware multimodal protocol. We adapt CLIP-style contrastive fusion [3, 7] to our encoder and evaluate ADNIMERGE modality groups separately, ordered by how directly they encode the label (Section 6).

3. The contrastive target shapes the image encoder. Our main finding: with architecture, split and training held fixed, aligning the encoder to anatomical volumes instead of cognitive scores produces an image-only representation that is 21 points better on MCI vs. CN, even though no tabular input is used at inference (Section 8).

4. Anatomically guided input. A per-subject crop of the medial temporal lobe (MTL), derived from the same segmentations, improves image-only accuracy without changing the encoder (Section 9).

We also make explicit why our numbers should not be read as outperforming prior work: the test set is small (63 scans), results are single-run, and the evaluation protocol difers from published ones in several respects (Section 10).

## 2 Related Work

MRI-based AD classification. Convolutional networks on 2D slices or 3D volumes are the dominant approach. 3D ResNets [8] and attention-augmented 3D residual networks [9] report AD vs. CN accuracies around 90% on ADNI-1, while hybrid CNN–Transformer models such as Conv-Swinformer [10] use attention to relate regions or slices. Performance on MCI vs. CN remains considerably lower. Wen et al. [1] documented how data leakage, for example splitting at the slice rather than the subject level, inflates reported accuracies, and argued for reproducible subject-level evaluation.

Multimodal and contrastive fusion. DAFT [11] conditions 3D CNN feature maps on tabular data. Hager et al. [4] pretrain image and tabular encoders jointly with a contrastive objective. Huang [3] adapts CLIP [7] to ADNI by contrasting MR images with groups of tabular variables (biomarkers, cognitive tests, volumes, medical history) and adds a tabular attention module; the fused model reports 83.8% three-way accuracy and 95.5% on AD vs. CN. Other multimodal works combine imaging with clinical data for conversion or progression prediction [12, 13].

Table 1: ADNIMERGE modality groups ordered by their overlap with the diagnostic label. Cognitive scores are part of the ADNI diagnostic criteria [5]; volumes are correlates of atrophy but not criteria.
<table><tr><td></td><td>Group Features</td><td>Leakage risk</td></tr><tr><td>cog</td><td>CDR-SB, ADAS11, MMSE, RAVLT- High: diagnostic instruments immediate</td><td></td></tr><tr><td>bio</td><td>Aβ, tau, p-tau (CSF)</td><td>Medium: biological correlates of pathology</td></tr><tr><td>pet</td><td>FDG, AV45</td><td>Medium: physiological markers</td></tr><tr><td>vol</td><td>Hippocampus, whole brain, entorhi- Low: atrophy correlates nal, middle temporal</td><td></td></tr><tr><td>demo</td><td>APOE4, age</td><td>Low: risk factors</td></tr></table>

Anatomical priors and explainability. FreeSurfer [14] and its deep-learning reimplementation FastSurfer [6] provide subcortical segmentation and cortical parcellation according to the Desikan–Killiany–Tourville (DKT) protocol [15]. Grad-CAM [16] is widely used to inspect which regions drive MRI classifiers [9, 13]. Our work combines both: segmentation is first used to audit the classifier and then to constrain its input.

## 3 Data

Imaging. We use 1,075 baseline T1-weighted scans from the ADNI-1 Screening 1.5 T collection acquired from 982 subjects; the excess of scans over subjects is due to a small number of subjects with more than one screening acquisition. Scans are labelled CN, MCI or AD, with per-class counts of approximately CN: 289 / MCI: 507 / AD: 279. All volumes were processed with FastSurfer [6], which conforms each scan to a 256<sup>3</sup> grid of 1 mm<sup>3</sup> isotropic voxels (orig.mgz) and produces a DKT [15] cortical parcellation with subcortical labels (aparc.DKTatlas+aseg.deep.mgz). Conformed volumes are stored in LIA orientation, so voxel axes 0, 1 and 2 correspond to the sagittal, axial and coronal directions, respectively. Training-time augmentation consists of random left–right flips, small random afine transforms and random bias fields.

Splits. Splits are made at the subject level: no subject appears in more than one split. The test set is balanced with 21 scans per class (63 scans); binary tasks use the corresponding 42-scan subsets. The remaining 1,012 scans are divided into approximately 862 training and 150 validation scans.

Tabular data. Clinical variables come from the consolidated ADNIMERGE table (16,421 subject–visit rows across all ADNI phases). Each scan is matched to a row by restricting to its subject identifier, preferring baseline visit codes (bl, sc, scmri) and choosing the row whose examination date is closest to the acquisition date; all 1,075 scans were matched. Following [3], variables are grouped into five modality groups (Table 1). Features are standardized with training-set statistics; missing values in the cog and vol groups (fewer than 3% of cells in either group at baseline) are imputed with the training-set mean of the corresponding feature before standardization.

In our experiments we use cog and vol. CSF biomarkers are available for only about 14% of ADNIMERGE rows, so the bio group would require heavy imputation, which we expect to distort contrastive alignment. The cog and vol groups sit at the two ends of the leakage spectrum and therefore provide the most informative contrast.

![](images/d0e823321cb62f538e4ea3208333028c35ca8faff2a83cec694395d43b667bbd.jpg)  
Solid arrows: forward pass. Purple arrows: CLIP loss connects image emb and tab emb Both heads are trained with cross-entropy; the CLIP loss is added during pretraining (phase 1) and with weight 0.3 during joint finetuning (phase 2)

Figure 1: Multimodal architecture. Image and tabular embeddings (128-d) are aligned by a CLIP loss. The image-only head classifies from the image embedding alone; the fusion head classifies from the concatenated embeddings.

## 4 Image Encoder

The image branch (Figure 1, top) processes a stack of S contiguous 2D slices of a subject. Each slice is resized to 224 × 224 and passed through an ImageNet-pretrained ResNet18 [17] shared across slices; only the last residual stage (layer4) is fine-tuned. The resulting 512-dimensional slice features are linearly projected, summed with a learned positional encoding, and processed by a single Transformer encoder layer [18] with 8 attention heads. Mean pooling over slices gives a 512-dimensional subject descriptor, which a two-layer MLP maps to a 128-dimensional embedding. Dropout of 0.15 is applied in attention, feed-forward and classifier layers.

For the full-slice (“raw”) configuration we use S = 90 slices (indices 62–152) along voxel axis 1, i.e. axial planes. The window was chosen with the slice statistics of Section 5.1 to cover as many AD-relevant structures as possible while excluding the orbits and neck.

We chose this 2D design over 3D alternatives even though, in a preliminary comparison on an earlier split, 3D encoders reached noticeably higher accuracy (a ResNet18 3D backbone with a slice-Transformer aggregator was about 20 points ahead of the 2D counterpart). Slicewise processing makes it straightforward to compare per-slice Grad-CAM maps with per-slice detections (Section 5.3), to crop anatomically (Section 9) and to combine the encoder with the fusion architecture (Section 6), while keeping the parameter count low.

## 5 Anatomical Reference from FastSurfer Segmentations

## 5.1 Structures and slice statistics

We selected structures that are routinely inspected in AD: the hippocampus, entorhinal cortex, amygdala and parahippocampal gyrus of the MTL [19–21]; the middle and inferior temporal and fusiform gyri; inferior and superior parietal cortex, precuneus, posterior and isthmus cingulate [22]; superior and rostral middle frontal gyri; the lateral and third ventricles; and cerebellar grey and white matter, all taken bilaterally. For each structure and voxel axis we counted, over 50 subjects, in how many subjects the structure’s labels appear on each slice, and defined its slice range as the interval covering 90% of occurrences. These ranges were used to choose the input window of the encoder and to decide which structures to annotate on each slice.

Table 2: YOLOv8 detection (40 epochs) and instance segmentation (15 epochs) of AD-relevant structures on held-out slices.
<table><tr><td>Task</td><td>Model</td><td>Precision</td><td>Recall</td><td> $\mathbf { m A P } _ { 5 0 }$ </td><td> $\mathbf { m A P _ { 5 0 - 9 5 } }$ </td></tr><tr><td rowspan="3">Detection</td><td>YOLOv8n</td><td>0.935</td><td>0.917</td><td>0.951</td><td>0.764</td></tr><tr><td>YOLOv8s</td><td>0.951</td><td>0.931</td><td>0.964</td><td>0.809</td></tr><tr><td>YOLOv8m</td><td>0.960</td><td>0.939</td><td>0.969</td><td>0.838</td></tr><tr><td rowspan="3">Segmentation</td><td>YOLOv8s-seg</td><td>0.939</td><td>0.923</td><td>0.962</td><td>0.639</td></tr><tr><td>YOLOv8m-seg</td><td>0.945</td><td>0.931</td><td>0.968</td><td>0.654</td></tr><tr><td> $\mathrm { Y O L O v 8 l – s e g }$ </td><td>0.944</td><td>0.934</td><td>0.969</td><td>0.663</td></tr></table>

## 5.2 Detection and segmentation with $\mathbf { Y O L O v 8 }$

Segmentation maps were converted into YOLOv8 [23] training data. For detection, each structure’s binary mask on a slice was split into connected components; components smaller than 100 pixels (5 pixels for the amygdala and entorhinal cortex) were discarded, and the remaining ones were converted to bounding boxes. For instance segmentation, external contours were extracted, simplified with the Douglas–Peucker algorithm, filtered below 50 pixels, and capped at 100 polygon vertices. Slices were intensity-normalized and resized to $6 4 0 \times 6 4 0$ . Models were trained on slices along the same axis as the encoder input (21 structures for detection, 20 for segmentation). Table 2 shows that all variants localize structures reliably $( \mathrm { m A P } _ { 5 0 } \geq 0 . 9 5 )$ ; the stricter $\mathrm { m A P _ { 5 0 - 9 5 } }$ still grows with model size and had not saturated within the training budget.

## 5.3 What does the image classifier look at?

We applied Grad-CAM [16] to the ResNet18 backbone of the raw-slice classifier and inspected the maps slice by slice next to the YOLO detections of the same subjects. Two patterns were consistent across subjects (Figure 2). On many slices, the network does attend to diagnostically meaningful tissue: when the lateral ventricles appear, attention concentrates on the high-contrast ventricular region, and on lower slices it moves to the temporal lobes, overlapping detected hippocampus, parahippocampal, fusiform, middle and inferior temporal structures. On a substantial number of other slices, however, attention falls on the skull, neck, orbits or outside the head. Correct predictions can therefore be supported by anatomically irrelevant evidence, which motivates constraining the input (Section 9). This analysis is qualitative; a quantitative measure such as the fraction of Grad-CAM mass inside the brain or MTL mask is left for future work.

## 6 Leakage-Aware Multimodal Contrastive Learning

## 6.1 Architecture

Figure 1 shows the multimodal model. The image branch is the encoder of Section 4, unchanged. The tabular branch is a TabTransformer-style encoder [24]: each of the F features of a modality group $( F = 4$ for cog and vol) is linearly tokenized into a 32-dimensional embedding with a learned positional embedding; the token sequence passes through a 6-layer Transformer with 8 heads of dimension 16; tokens are flattened, layer-normalized and projected to 128 dimensions.

Let $z _ { i } ^ { I }$ and $z _ { i } ^ { T }$ be the ℓ<sub>2</sub>-normalized image and tabular embeddings of subject i in a batch of size $N _ { ; }$ , and $s _ { i j } = \exp ( t ) z _ { i } ^ { I ^ { \top } } z _ { j } ^ { T }$ with learnable temperature t initialized at $\log ( 1 / 0 . 0 7 )$ . The

![](images/09192d14fb8ca554b7002189be8ec2188e6bb79af993f0eef36c5071414c8d2c.jpg)

![](images/18138050c79c15c759fa2f43ee2427093fdfc9960d79f0de13180a3cea0f3804.jpg)  
(a) Slice at the level of the lateral ventricles: attention on the ventricular region.  
(b) Slice at the level of the orbits: attention spreads to non-brain regions.  
Figure 2: Grad-CAM of the raw-slice ResNet18 backbone on two input slices of AD subjects (predicted class and confidence shown above each map).

symmetric CLIP loss [7] is

$$
\mathcal { L } _ { \mathrm { C L I P } } = - \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left[ \log \frac { e ^ { s _ { i i } } } { \sum _ { j } e ^ { s _ { i j } } } + \log \frac { e ^ { s _ { i i } } } { \sum _ { j } e ^ { s _ { j i } } } \right] .\tag{1}
$$

We compute this loss in fp32: under mixed precision it remained at its chance value log N.

Two heads are trained on top of the embeddings. The image-only head is a linear classifier on $z ^ { I }$ and measures what the image encoder has learned by itself; it uses no tabular input at inference. The fusion head is a two-layer MLP with dropout on the 256-dimensional concatenation $[ z ^ { I } ; z ^ { T } ]$

## 6.2 Training

When cross-entropy and contrastive terms were optimized jointly from the start, the cross-entropy gradients dominated and the CLIP loss stayed near log N. We therefore train in two phases: (i) contrastive pretraining with ${ \mathcal { L } } _ { \mathrm { C L I P } }$ only for 8–10 epochs; (ii) joint fine-tuning for 20–25 epochs with

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { C E } } ^ { \mathrm { i m g } } + \mathcal { L } _ { \mathrm { C E } } ^ { \mathrm { f u s } } + 0 . 3 \mathcal { L } _ { \mathrm { C L I P } } .\tag{2}
$$

We use AdamW [25] with a cosine learning-rate schedule, gradient clipping, class-balanced sampling, label smoothing of 0.1, and an exponential moving average of the weights for evaluation. Each modality group is trained in a separate run.

## 6.3 Why the leakage spectrum matters

In ADNI-1, CN, MCI and AD participants were defined by MMSE ranges, the global Clinical Dementia Rating and memory testing, together with clinical judgement [5]. The cog group contains the MMSE and the CDR sum of boxes, which is closely related to the global CDR. Predicting the label from these variables therefore largely amounts to re-deriving the rule that produced it. We report cog fusion as an upper bound and a sanity check of our implementation, and treat vol fusion, whose features are atrophy measurements rather than diagnostic criteria, as the imaging-relevant result.

Table 3: Three-way (CN / MCI / AD) test accuracy, 63 scans (21 per class). The last two rows are reported by Huang [3] under a diferent protocol (882 test slices, all tabular groups fused, mean ± s.d. over 5 splits) and are listed for reference only.
<table><tr><td>Configuration</td><td>Correct</td><td>Accuracy (%)</td></tr><tr><td>Image only (raw slices)</td><td>37/63</td><td>58.7 [46.4, 70.0]</td></tr><tr><td>Multimodal, cog (upper bound, leakage)</td><td>55/63</td><td>87.3 [76.9, 93.4]</td></tr><tr><td>Multimodal, vol</td><td>46/63</td><td>73.0 [61.0, 82.4]</td></tr><tr><td>Huang [3], MR images only</td><td></td><td> $7 6 . 1 \pm 1 . 4$ </td></tr><tr><td>Huang [3], multimodal (all data)</td><td></td><td> $8 3 . 8 \pm 2 . 3$ </td></tr></table>

Table 4: Binary test accuracy (%), 42 scans per task (21 per class). “Image head (vol-aligned)” is the image-only head of the model trained jointly with the vol contrastive target; “Image head (cog-aligned)” is the image-only head of the model trained with the cog target. Reference rows come from other protocols and are not directly comparable.
<table><tr><td>Configuration</td><td>AD vs. CN</td><td>MCI vs. CN</td></tr><tr><td>Image head (vol-aligned)</td><td>85.7 [72.2, 93.3]</td><td>73.8 [58.9, 84.7]</td></tr><tr><td>Image head (cog-aligned)</td><td></td><td>52.4 [37.7, 66.6]</td></tr><tr><td>Multimodal, cog (upper bound, leakage)</td><td>95.2 [84.2, 98.7]</td><td>83.3 [69.4, 91.7]</td></tr><tr><td>Multimodal, vol</td><td>83.3 [69.4, 91.7]</td><td>78.6 [64.1, 88.3]</td></tr><tr><td>Huang [3], MR images only</td><td> $8 8 . 5 \pm 1 . 5$ </td><td></td></tr><tr><td>Huang [3], multimodal (all data)</td><td> $9 5 . 5 \pm 1 . 7$ </td><td></td></tr><tr><td>Zhang et al. [9], 3D ResAttNet (MRI, ADNI-1)</td><td> $9 1 . 3 \pm 1 . 2$ </td><td></td></tr></table>

## 7 Results

All accuracies are computed on the balanced subject-level test set: 63 scans for three-way classification and 42 for each binary task. Brackets give 95% Wilson confidence intervals [26]; on 63 scans the interval spans roughly ±12 percentage points. All results are single runs.

## 7.1 Three-way classification

Table 3 shows that fusion with cognitive scores raises accuracy from 58.7% to 87.3%, with balanced per-class performance (19/21 MCI, 17/21 CN, 19/21 AD). This confirms that the contrastive pipeline is functioning, but for the reasons in Section 6 it is not evidence of better image understanding. Fusion with volumes also improves over the image-only model (73.0%); its residual errors are concentrated in the MCI/CN distinction, where baseline hippocampal and entorhinal volumes of the two groups overlap substantially at the subject level.

## 7.2 Binary tasks

On AD vs. CN, cog fusion misclassifies only two of 42 scans (CN 21/21, AD 19/21). The vol-aligned image head reaches 85.7%, which illustrates that gross late-stage atrophy is largely visible to the image encoder; vol fusion does not improve over the image head on this easier task. On MCI vs. CN, both fusion variants improve over the image head, and the vol-aligned head substantially outperforms the cog-aligned head (Section 8). We are not aware of an MCI vs. CN result in [3] and therefore list no reference for this task.

![](images/d25e6375fedb01f3500b5d9ed306a4c5a02da54891396243003beb756f749d7f.jpg)  
Figure 3: Image-only head accuracy on MCI vs. CN for encoders aligned to cognitive scores (cog) or regional volumes (vol). Same architecture, training schedule, split and test set (42 scans); error bars are 95% Wilson intervals.

## 8 The Contrastive Target Shapes the Image Encoder

The results above are accuracies of models. This section reports what we consider the main finding of the study, which concerns the representation rather than the accuracy: changing only the tabular group that the CLIP loss aligns the image embedding to changes what the image encoder itself learns.

A controlled comparison. The image-only head takes no tabular input at inference, so its accuracy measures the visual representation alone. We compare this head across two runs that are identical in architecture, training data, subject split, two-phase schedule, hyperparameters and test set. The single diference is the contrastive target: in one run the encoder is aligned to cog, in the other to vol. On MCI vs. CN, the cog-aligned image head is near chance (52.4%, 22/42), while the vol-aligned head reaches 73.8% (31/42), a gap of 21 points obtained without the tabular branch being present at evaluation (Figure 3).

Interpretation. A plausible account is that the geometry of the target determines what the image branch must encode in order to match it. Regional volumes live in a spatial, structural feature space: an image embedding can only be pulled towards them by encoding hippocampal, entorhinal and global atrophy, and those features remain discriminative once the tabular branch is removed. Cognitive scores ofer no spatial structure to latch onto. Worse, because they nearly determine the label (Table 1), the fusion head can solve the task from the tabular branch alone, so neither the contrastive nor the classification gradient pushes the encoder to extract atrophy-related features. The fused accuracies invert the ordering of the two runs, 87.3% for cog against 73.0% for vol on the three-way task, which is exactly why the image-only head has to be reported alongside the fused one: the configuration that looks better end-to-end is the one that learned the weaker visual representation.

Consequences. Two practical points follow. For benchmarking, a fused number alone cannot tell whether a multimodal model improved its use of imaging or merely gained access to the labelling rule; reporting the unimodal head under each target separates the two. For training, the tabular side can be treated as a design choice that shapes the encoder rather than as extra input, which is the reasoning that led us to restrict the input to the same anatomical region in Section 9.

Strength of evidence. The gap corresponds to nine test scans. An unpaired two-proportion z-test gives $p \approx 0 . 0 4$ for a single run, so we regard the efect as suggestive rather than established, and its size should not be taken literally. Confirming it requires a paired test on per-scan predictions, repetition over several seeds, and ideally a third target of intermediate leakage such as pet or demo.

![](images/fbb856f3d499255ba25c303784dfb095aa43cef132d14743a784324be8e03c35.jpg)  
Figure 4: MTL crop for an AD subject. (a) Uncropped coronal slice. (b) FastSurfer labels of the eight MTL structures and the bounding box of their union, whose centroid defines the crop centre. (c) The fixed-size crop given to the encoder after resizing to 224 × 224.

Table 5: Raw-slice vs. coronal MTL-crop input. Three-way on 63 scans, binary on 42 scans. The “Image only, raw slices” binary numbers are the image head of the vol-aligned jointly trained model of Table 4; for a fair comparison the “Image only, MTL crop” rows also report the image head of a vol-aligned jointly trained model on the cropped input, rather than a cross-entropy-only model.
<table><tr><td>Configuration</td><td>3-way</td><td>AD vs. CN</td><td>MCI vs. CN</td></tr><tr><td>Image only, raw slices</td><td>58.7</td><td>85.7</td><td>73.8</td></tr><tr><td>Image only, MTL crop (coronal)</td><td>65.1</td><td>88.1</td><td>81.0</td></tr><tr><td>Multimodal vol, raw slices</td><td>73.0</td><td>83.3</td><td>78.6</td></tr><tr><td>Multimodal vol, MTL crop (coronal)</td><td>76.2</td><td>90.5</td><td>85.7</td></tr></table>

## 9 Anatomically Guided Input: MTL Crop

Motivated by the Grad-CAM analysis, we restrict the encoder input to the medial temporal lobe. For each subject, we form a binary mask from the union of eight FastSurfer labels, bilateral hippocampus (17, 53), amygdala (18, 54), entorhinal cortex (1006, 2006) and parahippocampal cortex (1016, 2016), and compute its centroid. A fixed-size box of $9 6 \times 4 8 \times 6 4$ voxels (left–right × inferior–superior × anterior–posterior) centered on this subject-specific centroid is extracted, so input size is constant while position follows each subject’s anatomy (Figure 4). The crop is sliced along the coronal axis (voxel axis 2, 64 slices of 96 × 48), each slice resized to 224 × 224, and passed to the unchanged encoder. The MTL was chosen because it carries the earliest structural changes in AD [19–21] and overlaps with the vol features, so the input and the contrastive target emphasize the same region.

Table 5 summarizes the results. With the coronal crop, image-only three-way accuracy increases from 58.7% (37/63) to 65.1% (41/63, 95% CI [52.8, 75.7]); the MCI vs. CN task benefits more than AD vs. CN, consistent with the MTL carrying the early, subtle signal while late-stage atrophy is already visible in full slices. Coronal is the standard view for the hippocampus and entorhinal cortex; we also trained axial and sagittal variants of the crop, but the numbers are omitted here pending final log extraction. Combining the crop with vol alignment helps further but sub-additively, which we attribute to both interventions targeting the same region. We stress that the three-way improvement corresponds to four scans and its confidence interval overlaps

with the baseline.

## 10 Discussion and Limitations

Comparison with published results. Our cog numbers are numerically close to those of Huang [3], but the protocols difer in ways that preclude a ranking: (i) we evaluate 63 scans at the subject level, whereas [3] evaluates 882 slices; (ii) we use ADNI-1 baseline scans only, whereas [3] uses a diferent and larger selection of ADNI images; (iii) we fuse one modality group at a time, whereas [3] fuses all groups, including a medical-history group that contains the baseline diagnosis; (iv) we report single runs, whereas [3] averages five splits. We read the proximity of the numbers as evidence that our re-implementation works, not as an improvement. Our image-only baselines remain below 3D encoders on ADNI [3, 9]; the MTL crop narrows but does not close this gap.

Leakage. Cognitive scores are legitimate clinical inputs, and a deployed system may well use them. For assessing what an imaging model contributes, however, they conflate image information with the labelling rule. Our results suggest a further efect: a leaky contrastive target can also degrade the image representation, since the image-only head of the cog-aligned model is near chance on MCI vs. CN. We recommend reporting fusion results per modality group and always reporting the image-only head alongside the fused one.

Limitations. The test set is small and every configuration is a single run, so diferences of a few points are within noise. The MTL crop requires a FastSurfer segmentation at inference, and segmentation errors propagate directly to the crop position. All data come from one cohort and field strength (ADNI-1, 1.5 T), and Grad-CAM findings are qualitative. Our compute environment imposed a 12-hour limit per job, which prevented longitudinal multi-phase training.

Future work. We plan to (i) repeat all experiments over several seeds and a larger, longitudinal ADNI-1/2/3 set with cached slice features; (ii) add the bio group where CSF coverage permits and test whether it induces a similar efect on the image encoder; (iii) quantify Grad-CAM overlap with segmented structures and analyze disagreements between explanation and prediction; and (iv) reduce the dependence on external segmentation, for example through semi-supervised training.

## 11 Conclusion

Using a lightweight slice-based encoder on ADNI-1, we showed that anatomical references derived from FastSurfer can be used both to audit and to constrain an Alzheimer’s classifier, that multimodal gains obtained with cognitive scores largely reflect label leakage, and that the choice of contrastive target influences what the image encoder learns, with volume alignment yielding a markedly better image-only representation for MCI vs. CN than alignment to cognitive scores. Restricting the input to the medial temporal lobe further improves image-only accuracy. Given the small test set and single runs, these findings should be confirmed with repeated, larger-scale experiments.

## Acknowledgements

Data collection and sharing for this project was funded by the Alzheimer’s Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012). Data used in preparation of this article were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni. loni.usc.edu/wp-content/uploads/how\_to\_apply/ADNI\_Acknowledgement\_List.pdf. Experiments were run on the FEP cluster of the National University of Science and Technology POLITEHNICA Bucharest.

## References

[1] Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz-Melo, Jorge Samper-Gonz´alez, Alexandre Routier, Simona Bottani, Didier Dormont, Stanley Durrleman, Ninon Burgos, and Olivier Colliot. Convolutional neural networks for classification of Alzheimer’s disease: Overview and reproducible evaluation. Medical Image Analysis, 63:101694, 2020.

[2] Cliford R. Jack, Matt A. Bernstein, Nick C. Fox, Paul Thompson, Gene Alexander, Danielle Harvey, Bret Borowski, Paula J. Britson, Jennifer L. Whitwell, Chadwick Ward, et al. The Alzheimer’s disease neuroimaging initiative (ADNI): MRI methods. Journal of Magnetic Resonance Imaging, 27(4):685–691, 2008.

[3] Weichen Huang. Multimodal contrastive learning and tabular attention for automated Alzheimer’s disease prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2023. arXiv:2308.15469.

[4] Paul Hager, Martin J. Menten, and Daniel Rueckert. Best of both worlds: Multimodal contrastive learning with tabular and imaging data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23924–23935, 2023.

[5] Ronald C. Petersen, Paul S. Aisen, Laurel A. Beckett, Michael C. Donohue, Anthony C. Gamst, Danielle J. Harvey, Cliford R. Jack, William J. Jagust, Leslie M. Shaw, Arthur W. Toga, John Q. Trojanowski, and Michael W. Weiner. Alzheimer’s disease neuroimaging initiative (ADNI): Clinical characterization. Neurology, 74(3):201–209, 2010.

[6] Leonie Henschel, Sailesh Conjeti, Santiago Estrada, Kersten Diers, Bruce Fischl, and Martin Reuter. FastSurfer – a fast and accurate deep learning based neuroimaging pipeline. NeuroImage, 219:117012, 2020. doi: 10.1016/j.neuroimage.2020.117012.

[7] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of PMLR, pages 8748– 8763, 2021.

[8] Amir Ebrahimi, Suhuai Luo, and Raymond Chiong. Introducing transfer learning to 3D ResNet-18 for Alzheimer’s disease detection on MRI images. In 35th International Conference on Image and Vision Computing New Zealand (IVCNZ), pages 1–6, 2020.

[9] Xin Zhang, Liangxiu Han, Wenyong Zhu, Liang Sun, and Daoqiang Zhang. An explainable 3D residual self-attention deep neural network for joint atrophy localization and Alzheimer’s disease diagnosis using structural MRI. IEEE Journal of Biomedical and Health Informatics, 26(11):5289–5297, 2022. doi: 10.1109/JBHI.2021.3066832.

[10] Zhentao Hu, Yanyang Li, Zheng Wang, Shuo Zhang, Wei Hou, et al. Conv-swinformer: Integration of CNN and shift window attention for Alzheimer’s disease classification. Computers in Biology and Medicine, 164:107304, 2023.

[11] Sebastian P¨olsterl, Tom Nuno Wolf, and Christian Wachinger. Combining 3D image and tabular data via the dynamic afine feature map transform. In Medical Image Computing and Computer Assisted Intervention (MICCAI), pages 688–698, 2021.

[12] Matthew Velazquez and Yugyung Lee. Multimodal ensemble model for Alzheimer’s disease conversion prediction from early mild cognitive impairment subjects. Computers in Biology and Medicine, 151:106201, 2022.

[13] Nasir Rahim, Shaker El-Sappagh, Sajid Ali, Khan Muhammad, Javier Del Ser, and Tamer Abuhmed. Prediction of Alzheimer’s progression based on multimodal deep-learning-based fusion and visual explainability of time-series data. Information Fusion, 92:363–388, 2023.

[14] Bruce Fischl. FreeSurfer. NeuroImage, 62(2):774–781, 2012. doi: 10.1016/j.neuroimage.2012. 01.021.

[15] Arno Klein and Jason Tourville. 101 labeled brain images and a consistent human cortical labeling protocol. Frontiers in Neuroscience, 6:171, 2012.

[16] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-CAM: Visual explanations from deep networks via gradientbased localization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 618–626, 2017.

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[18] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

[19] C. Pennanen, M. Kivipelto, S. Tuomainen, P. Hartikainen, T. H¨anninen, M. P. Laakso, et al. Hippocampus and entorhinal cortex in mild cognitive impairment and early AD. Neurobiology of Aging, 25(3):303–310, 2004.

[20] B. C. Dickerson, A. Bakkour, D. H. Salat, E. Feczko, J. Pacheco, D. N. Greve, et al. The cortical signature of Alzheimer’s disease: Regionally specific cortical thinning relates to symptom severity in very mild to mild AD dementia and is detectable in asymptomatic amyloid-positive individuals. Cerebral Cortex, 19(3):497–510, 2009.

[21] C. Ech´avarri, P. Aalten, H. B. Uylings, H. Jacobs, P. J. Visser, E. Gronenschild, et al. Atrophy in the parahippocampal gyrus as an early biomarker of Alzheimer’s disease. Brain Structure and Function, 215(3):265–271, 2011.

[22] H. I. Jacobs, M. P. Van Boxtel, J. Jolles, F. R. Verhey, and H. B. Uylings. Parietal cortex matters in Alzheimer’s disease: An overview of structural, functional and metabolic findings. Neuroscience & Biobehavioral Reviews, 36(1):297–309, 2012.

[23] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLOv8. https://github. com/ultralytics/ultralytics, 2023. Version 8.0.0, AGPL-3.0 license.

[24] Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. TabTransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678, 2020.

[25] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.

[26] Edwin B. Wilson. Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158):209–212, 1927.