# Aggregation-Induced Calibration Failure in Sum-Based Label Expansion for Multi-Label Facial Expression Recognition: A Diagnostic Study

Weihang Yuan

Beijing Forestry University, Beijing, China

Email: ywh15958108437@bjfu.edu.cn

Abstract—Label-space expansion—predicting a larger joint space and summing back to the original labels—is a natural design for multi-label facial expression recognition, but its failure modes have not been characterised. We evaluate Condition-Aware Label Expansion (CALE) on a self-built dataset (30,240 training images, 3,460 multi-label test images), on seven backbones, and on the public RAF-DB and RAF-ML benchmarks. The central finding is that the aggregation rule, not the representation, decides whether the method works. With a fixed threshold of 0.5, the aggregate score spans 28× across five backbones at 10% of the training data (0.24 to 6.79) and collapses to a 1.9× range at full data, while threshold-free mAP stays within 1.28× over the same backbones, which localises the fault to the summation step. The mechanism is the uncalibrated rule $\mathit { p } _ { e } =$ $\textstyle \sigma ( x _ { e , c ^ { * } } ) + 0 . 5 \sum _ { c \neq c ^ { * } } \sigma ( x _ { e , c } ) ;$ we verify it by measuring mainand non-main-branch activations. Two plain seven-dimensional sigmoid baselines (BCE, ASL) reach mAP 0.5055 and 0.5175; ASL exceeds three of the five CALE configurations and falls within 0.03 of the best. On single-label FER the gain over a seven-dimensional baseline is nominally significant (paired t-test, $p = 0 . 0 4 3 6 )$ but does not survive Benjamini–Hochberg correction $\mathbf { \bar { ( } } p _ { \mathbf { B H } } = 0 . 2 0 5 5 )$ or a Wilcoxon test $( p = 0 . 1 2 5 0 )$ . We conclude that sum-based label expansion requires threshold calibration at the deployment data scale, or a learned aggregation, before it can support multi-label decisions.

Index Terms—Facial expression recognition; multi-label classification; label expansion; aggregation calibration; negative results; empirical study.

## I. INTRODUCTION

Facial expressions are a primary carrier of emotional and intentional information. The cross-cultural theory of basic emotions by Ekman and Friesen [1] laid the foundation for automatic FER. Deep convolutional neural networks (CNNs) have achieved significant progress on benchmarks such as FER2013 [2], RAF-DB [3], AffectNet [5], and older datasets such as CK+ [28] and SFEW [29]. Compared with handcrafted features [6], deep models learn hierarchical representations automatically.

Facial expressions are also a key window into human emotion in psychological research. Psychologists and trained observers typically rely on subjective rating scales, facial action coding systems, or their own experience. Real emotions, however, are rarely pure: a face may simultaneously express $\mathrm { ^ { * } a n g e r \ + \ d i s g u s t ^ { * } }$ or “surprise + fear.” Single-label softmax classification cannot represent such mixtures. Observers judgments are further affected by image quality, illumination, and resolution, making consistent annotation difficult. An automatic system that handles mixed emotions and degraded images would therefore be a valuable tool for psychologists, providing objective and reproducible assessments.

Real-world FER faces four challenges: (1) image quality degradation—uneven illumination, low resolution, and blur can make the same expression look more different than two distinct expressions [3]; (2) coexisting multiple emotions— traditional single-label classification cannot represent mixtures; (3) imbalanced class distributions—rare expressions such as disgust and fear suffer from low recall [7]; and (4) training-inference inconsistency—multi-branch models that train only the current condition branch but sum all branches at inference face a mismatched objective [8].

To address these issues, we propose and systematically study Condition-Aware Label Expansion (CALE), a labelspace organization strategy that encodes image quality conditions into the label space and uses a multi-branch joint output with summation aggregation. Rather than claiming a new state-of-the-art method, this paper provides a diagnostic empirical study of when and why this strategy helps. Our main contribution is the diagnostic analysis: the mechanism itself is described for completeness, but the central claims of the paper are the empirical findings and their practical implications. The main contributions are summarized as follows:

• A diagnostic study of sum-based label expansion. We present a systematic evaluation of CALE, a concrete instantiation of the general design pattern $\begin{array} { r l } { p _ { e } } & { { } = } \end{array}$ $\textstyle \sum _ { c } w _ { c } \sigma ( x _ { e , c } )$ with $w _ { c ^ { * } } = 1$ and $w _ { c \neq c ^ { * } } = 0 . 5$ . We frame CALE as a case study, not as a state-of-the-art method.

• An aggregation-induced calibration failure. Across four training-data regimes (10%, 25%, 50%, 100%) and five backbones, we show that the aggregate score at threshold 0.5 varies by 28× across backbones at 10% data and collapses to systematic under-prediction at full data. We verify the mechanism by direct measurement of the main-branch and non-main-branch sigmoid activations, and we show that the threshold-free mAP is stable across the same backbones, which localizes the failure to the

aggregation step.

• A fair comparison against non-condition-aware baselines. Two seven-dimensional sigmoid baselines (BCE and ASL) reach mAP 0.5055 and 0.5175 under the same protocol; the ASL baseline exceeds three of the five CALE configurations and falls within 0.03 of the best. This is direct evidence that the additional condition dimensions do not translate into a consistent architectureindependent advantage on multi-label FER.

• A quantified limitation of the rule-based condition classifier. A ±20% threshold perturbation shifts the dominant condition class by nearly 29 percentage points, which is a first-order sensitivity for a component that determines both the training soft-label target and the inference branch weights.

• Honest reporting of three structural limitations. We document a single-label training / multi-label evaluation mismatch, a coupling between the synthetic-condition generator and the rule-based condition classifier, and single-fold single-seed diagnostics. We do not claim that these limitations are resolved in the present study.

• Full reproducibility. Code, data-split indices, trained model weights, and a full description of the diagnostic experiments are publicly available.

The remainder of this paper is organized as follows. Section II reviews related work. Section III presents the CALE method. Section IV reports the main experiments. Section V presents the diagnostic analysis. Section VI discusses findings and applications, and Section VII concludes the paper.

## II. RELATED WORK

## A. Deep Facial Expression Recognition

Mollahosseini et al. [11] first applied deep CNNs to sevenclass FER, and Rifai et al. [30] explored disentangled factor representations for the same task. He et al. [12] proposed ResNet, which has become a standard backbone. Hu et al. [9] proposed SE networks. Recent works have explored vision transformers [13], [14]. To handle uncertainty, Wang et al. [8] proposed SCN; Ruan et al. [15] proposed feature decomposition; Zhao et al. [16] used multi-scale attention; Liu et al. [17] proposed adaptive multilayer attention; Xue et al. [18] introduced TransFER, and Wang et al. [31] proposed region attention networks for pose and occlusion robustness. These methods, however, assume stable image quality and pay limited attention to condition robustness.

## B. Multi-Label Classification

Multi-label learning allows a sample to belong to multiple classes [19], [20]. Deep methods typically use sigmoid with binary cross-entropy. Recent work has focused on loss design, including Asymmetric Loss [21] and Class-Balanced Loss [22]; graph-based methods such as ML-GCN [32] model label co-occurrence explicitly. In FER, multi-label learning is still emerging. Li and Deng [4] introduced RAF-ML for blended emotion recognition. We embed condition information into the label space via label expansion, so the multi-branch structure naturally supports multi-label output.

TABLE I: Training Set Sample Distribution
<table><tr><td>Emotion</td><td>Number of Samples</td></tr><tr><td>Angry</td><td>4,320</td></tr><tr><td>Disgust</td><td>4,320</td></tr><tr><td>Fear</td><td>4,320</td></tr><tr><td>Happy</td><td>4,320</td></tr><tr><td>Sad</td><td>4,320</td></tr><tr><td>Surprise</td><td>4,320</td></tr><tr><td>Natural</td><td>4,320</td></tr><tr><td>Total</td><td>30,240</td></tr></table>

## C. Quality-Aware Learning

Quality-aware learning improves robustness to low-quality inputs via enhancement, multi-branch networks, quality estimation, or domain adaptation. Li et al. [3] showed that image quality significantly affects recognition, and classical imagequality metrics such as SSIM [35] remain widely used to quantify degradation. Unlike these works, we encode quality directly into the label space without extra estimation or reconstruction.

## D. Attention Mechanisms

Attention focuses on key information [23]. SE obtains channel descriptors via global average pooling and learns channel weights [9]. CBAM adds spatial attention [24]. Nonlocal networks model long-range dependencies [25]. We adopt SE for its simplicity, few parameters, and training stability.

## III. METHOD

## A. Dataset Construction

The self-built dataset is constructed from publicly available images and video frames. Specifically, the source material comes from two types of public content: (1) films and TV series, and (2) short videos posted by online content creators. All images contain only publicly accessible human faces that have been voluntarily shown in public media. No private or sensitive personal information is involved, and the images are used solely for non-commercial academic research. This usage follows the same principle adopted by public FER datasets such as RAF-DB and AffectNet.

For each of the seven basic emotions (angry, disgust, fear, happy, sad, surprise, natural), 216 source clips or stills are selected, forming a baseline set of 7×216 = 1,512 images. Each source image is treated as one standard-condition sample, and 19 additional variants are generated through five geometric and photometric transformations (rotation, translation, brightness adjustment, contrast adjustment, and Gaussian blur) across four quality conditions. Each source image therefore yields 20 samples, and each class contains 216 × 20 = 4,320 samples. The seven classes together contain 30,240 training images. All images are aligned by MTCNN [26] and resized to 224×224.

The test set consists of 3,460 independently collected multilabel images from different sources, and each test image may belong to more than one emotion category. Table I lists the training distribution, and Table II reports the condition distribution. We treat each raw photo and its 19 derived variants as one group; 5-fold cross-validation is grouped by this identifier to prevent leakage.

TABLE II: Condition Distribution of Training and Test Sets
<table><tr><td>Condition</td><td>Training Set</td><td>Test Set</td></tr><tr><td>Weak light</td><td>7,520</td><td>949</td></tr><tr><td>Strong light</td><td>9,565</td><td>1,003</td></tr><tr><td>Low resolution</td><td>5,897</td><td>678</td></tr><tr><td>Standard</td><td>7,258</td><td>830</td></tr><tr><td>Total</td><td>30,240</td><td>3,460</td></tr></table>

## B. Overall Framework

Fig. 1 illustrates the overall framework. Each raw photo is augmented across four quality conditions, producing 19 variants plus the original. The condition classifier assigns one of the four quality conditions to each image using the fixedthreshold rule in Table III. This label is used in two places: during training, it determines which of the four condition positions is treated as the main condition $c ^ { * }$ in Eq. (4); during inference, it determines the branch weight in Eq. (5). Because the classifier is rule-based and deterministic, the same image receives the same label at training and inference time, which preserves the training-inference consistency that is central to CALE. We characterize its behavior in Section V-E and show that its output is extremely sensitive to the thresholds, which we identify as one of the method’s limitations. After MTCNN [26] alignment, images pass through a shared backbone in which shallow layers are frozen and only the deeper layers are fine-tuned. The output then passes through global average pooling and an SE module, followed by dropout and a fully connected layer to produce a 28-dimensional vector. Each group of four dimensions corresponds to one emotion under four conditions. At inference, the 28 logits are aggregated per emotion into seven scores, and a fixed threshold of 0.5 yields the final multi-label prediction.

## C. Condition Classifier

The classifier assigns each image to one of four conditions using two inexpensive statistics:

• Brightness mean: average pixel value, reflecting illumination.

• Laplacian variance: variance of the Laplacian response, measuring sharpness [27].

The rules are listed in Table III. Thresholds are set empirically, and condition labels are precomputed once per image. The classifier is used for two purposes: (i) in training, it determines the main condition position $c ^ { * }$ in the soft-label target in Eq. (4); (ii) in inference, it supplies the one-hot vector that weights the four branches in Eq. (5). We report the classifier’s behavior and its threshold sensitivity in Section V-E.

TABLE III: Condition Classification Thresholds
<table><tr><td>Condition</td><td>Rule</td></tr><tr><td>Weak light</td><td>Lap. variance ≥ 10 and brightness mean  $< 8 0$ </td></tr><tr><td>Strong light</td><td>Lap. variance ≥ 10 and brightness mean &gt; 120</td></tr><tr><td>Low resolution</td><td>Lap. variance &lt; 10</td></tr><tr><td>Standard</td><td>otherwise</td></tr></table>

## D. Feature Extraction and SE Attention

We adopt ImageNet-pretrained backbones. Shallow layers are frozen; only the deeper layers, the SE module, and the fully connected layer are fine-tuned. This preserves generic features and reduces the number of trainable parameters, lowering the risk of overfitting.

After the backbone outputs a feature map, the SE module performs global average pooling:

$$
z _ { c } = \frac { 1 } { H \times W } \sum _ { i = 1 } ^ { H } \sum _ { j = 1 } ^ { W } F _ { c } ( i , j ) .\tag{1}
$$

Channel dependencies are then learned as:

$$
\begin{array} { r } { \mathbf { s } = \sigma \left( \mathbf { W } _ { 2 } \delta ( \mathbf { W } _ { 1 } \mathbf { z } ) \right) , } \end{array}\tag{2}
$$

where $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { \frac { C } { r } } \times C , \mathbf { W } _ { 2 } \in \mathbb { R } ^ { C \times \frac { C } { r } } , r = 1 6 .$ , δ is ReLU, and $\sigma$ is Sigmoid. The recalibration is:

$$
\mathbf { F } _ { s e } = \mathbf { F } \odot \mathbf { s } ,\tag{3}
$$

where ⊙ denotes channel-wise multiplication. The SE module enhances channels beneficial for expression recognition and suppresses irrelevant ones.

To verify the SE module, we visualize the 512-dimensional scale factors for a test image (Fig. 2). Some channels have factors close to 1, while many others are close to $0 ,$ indicating selective enhancement rather than uniform weighting.

## E. Joint Output and Loss Function

After SE, features pass through dropout (rate 0.5) and a linear layer to produce a 28-dimensional output (7 emotions × 4 conditions). We refer to this condition-aware label expansion as LE. The logit for emotion e under condition c is $x _ { e , c } ,$ and the sigmoid-activated probability is $\sigma ( x _ { e , c } )$

During training, the main condition position of the groundtruth emotion is set to 1.0, and the other condition positions are set to 0.5:

$$
y _ { e , c } = \left\{ { \begin{array} { l l } { 1 . 0 , } & { c = c ^ { * } , e = e _ { g t } } \\ { 0 . 5 , } & { c \neq c ^ { * } , e = e _ { g t } } \\ { 0 , } & { e \neq e _ { g t } } \end{array} } \right.\tag{4}
$$

During inference, the four condition scores are aggregated by summation:

$$
p _ { e } = \sigma ( x _ { e , c ^ { * } } ) + 0 . 5 \sum _ { c \neq c ^ { * } } \sigma ( x _ { e , c } ) ,\tag{5}
$$

where $p _ { e } \in [ 0 , 2 . 5 ]$ is not normalized to a probability. A fixed threshold of 0.5 produces the final multi-label prediction. This aggregation structurally matches the soft-label training.

![](images/8cd665e4ce7ce714f19da2b90f5192b424e4386b392f0762a0fb268e2ff93641.jpg)  
Fig. 1: Overall framework of the CALE method. After MTCNN alignment, the image is processed by a shared backbone with shallow layers frozen. A global average pooling layer and an SE module recalibrate channel features, followed by flatten and dropout. The 28-dimensional output is organized as four condition-specific 7-dimensional branches. Each branch is sigmoid-activated, and the four branches are aggregated per emotion to produce seven scores. A fixed threshold yields the final multi-label prediction. This figure was drawn by the author with TikZ.

![](images/2f59f9951441c0065a8b120e3fb6713cc05c671a5f64ab90a7dd33b14c07f507.jpg)  
Fig. 2: Visualization of the 512-dimensional SE scale factors for a test image. Higher values indicate channels more strongly enhanced by the SE module.

For multi-strategy training, we use a soft-label variant of Focal Loss:

$$
\mathcal { L } = - \alpha _ { t } ( 1 - p _ { t } ) ^ { \gamma } \log ( p _ { t } ) ,\tag{6}
$$

where $p _ { t }$ is the model’s estimated probability for the target class, $\gamma = 2 . 0$ is the focusing parameter, and $\alpha _ { t }$ is the classweighted coefficient. Weighted sampling, strong augmentation, dropout 0.5, and shallow-layer freezing form the remaining components of MS.

## IV. EXPERIMENTS

## A. Datasets and Protocol

We evaluate CALE on three datasets:

• Self-built dataset: 30,240 training images (7 emotions, 4 conditions) and 3,460 independently collected multi-label test images.

• RAF-DB [3]: 12,271 training and 3,068 test images with 7 basic emotions.

• RAF-ML [4]: 3,926 training and 982 test images with blended multi-label annotations across 6 emotions.

For 5-fold cross-validation on the self-built dataset, subjects are grouped so that variants of the same source image never appear in both training and validation. For cross-backbone experiments, we split the training set into $7 / 8$ training and

1/8 validation, with 27 subjects per emotion in the validation set.

## B. Implementation Details

All models are trained with AdamW using a learning rate of $1 \times 1 0 ^ { - 5 }$ , a weight decay of $5 \times 1 0 ^ { - 3 }$ , and cosine annealing. The batch size is 128 for ResNet-18/34 and 96 for ResNet-50/DenseNet121. Focal Loss uses $\gamma = 2 . 0$ and class weights $[ 0 . 9 , 0 . 8 , 1 . 0 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 2 ]$ for the seven emotions. The random seed is fixed at 42 across all experiments. All experiments were performed on a workstation running Ubuntu 20.04 with an NVIDIA RTX 3090 GPU (24GB), using PyTorch 1.10 and Python 3.8.

## C. Main Results on the Self-Built Dataset

Table IV reports the 5-fold single-label cross-validation results. CALE (V7) achieves a Macro-F1 of 0.7747, compared with 0.7618 for the V0 baseline (+1.29 points). The best single-label variant is V5 (LE+MS) at 0.7758.

## D. Ablation Study

We conduct a $2 ^ { 3 }$ factorial ablation study with ResNet-18. For brevity, we denote label expansion as LE, multi-strategy training as MS, and Squeeze-and-Excitation attention as SE. Table IV reports the full comparison, and Table VI reports paired tests across the 5 folds.

Label expansion is the primary driver. All variants containing LE (V2, V5, V6, and V7) achieve Val Macro-$\mathrm { F 1 } \geq 0 . 7 7 4 7$ , while variants without LE (V0, V1, V3, and V4) stay below 0.7729. LE alone (V2 vs V0) improves Validation UAR by +0.0078 (t = 2.917, p = 0.0436 under an uncorrected paired t-test).

SE provides a robustness benefit. V6 (LE+SE vs V0) achieves the strongest paired mean improvement (+0.0096, $t = 3 . 2 7 0 , p = 0 . 0 3 0 8 )$ . Compared to V2 alone, V6 adds only $+ 0 . 0 0 1 8$ in the mean but reduces the fold-to-fold variance, making the improvement over the baseline more consistent.

TABLE IV: Main results on the self-built dataset (5-fold single-label cross-validation, mean ± std). LE: label expansion; MS: multi-strategy training; SE: squeeze-and-excitation.
<table><tr><td>Variant</td><td>LE</td><td>MS</td><td>SE</td><td>Val Accuracy</td><td>Val UAR</td><td>Val Macro-F1</td></tr><tr><td>VO: Baseline</td><td>一</td><td>一</td><td>一</td><td> $0 . 7 6 3 5 \pm 0 . 0 1 0 7$ </td><td> $0 . 7 6 3 4 \pm 0 . 0 1 0 6$ </td><td> $0 . 7 6 1 8 \pm 0 . 0 1 1 8$ </td></tr><tr><td>V1: +MS</td><td>一</td><td>√</td><td>一</td><td> $0 . 7 7 1 6 \pm 0 . 0 1 0 2$ </td><td> $0 . 7 7 1 5 \pm 0 . 0 0 9 8$ </td><td> $0 . 7 7 0 5 \pm 0 . 0 0 9 8$ </td></tr><tr><td>V2: +LE</td><td>√</td><td>一</td><td>一</td><td> $0 . 7 7 4 5 \pm 0 . 0 0 7 8$ </td><td> $0 . 7 7 4 5 \pm 0 . 0 0 7 7$ </td><td> $0 . 7 7 2 9 \pm 0 . 0 0 8 2$ </td></tr><tr><td>V3: +SE</td><td>一</td><td>一</td><td>√</td><td> $0 . 7 7 2 6 \pm 0 . 0 0 8 6$ </td><td> $0 . 7 7 2 5 \pm 0 . 0 0 8 4$ </td><td> $0 . 7 7 0 6 \pm 0 . 0 0 8 5$ </td></tr><tr><td>V4: +MS+SE</td><td>一</td><td>√</td><td>√</td><td> $0 . 7 7 2 3 \pm 0 . 0 0 9 3$ </td><td> $0 . 7 7 2 2 \pm 0 . 0 0 9 0$ </td><td> $0 . 7 7 0 9 \pm 0 . 0 0 8 8$ </td></tr><tr><td> ${ \mathsf { V } } { \mathsf { 5 } } \colon + { \mathrm { L E } } + { \mathrm { M S } }$ </td><td>√</td><td>√</td><td>一</td><td> $\mathbf { 0 . 7 7 7 3 \pm 0 . 0 1 3 1 }$ </td><td> $\mathbf { 0 . 7 7 7 2 \pm 0 . 0 1 2 9 }$ </td><td> $\mathbf { 0 . 7 7 5 8 \pm 0 . 0 1 2 8 }$ </td></tr><tr><td> $\mathrm { V } 6 \colon + \mathrm { L E } { + } \mathrm { S E }$ </td><td>√</td><td>一</td><td>√</td><td> $0 . 7 7 6 4 \pm 0 . 0 1 0 7$ </td><td> $0 . 7 7 6 3 \pm 0 . 0 1 0 4$ </td><td> $0 . 7 7 5 0 \pm 0 . 0 1 0 5$ </td></tr><tr><td>V7: Full</td><td>√</td><td>√</td><td>√</td><td> $0 . 7 7 6 4 \pm 0 . 0 1 4 9$ </td><td> $0 . 7 7 6 3 \pm 0 . 0 1 4 7$ </td><td> $0 . 7 7 4 7 \pm 0 . 0 1 4 8$ </td></tr></table>

MS increases variance without a statistically separable gain. Although V5 (LE+MS) achieves the highest mean Validation UAR at 0.7758, its paired improvement over V0 (+0.0105) does not reach significance $( p = 0 . 0 5 8 9 )$ , and the full model V7 is even less significant $( p = 0 . 1 6 7 2 )$ because its fold-to-fold variance is the largest among all variants (±0.0148 in Val Macro-F1). Pairwise comparisons among LE-containing variants (V5 vs V2, V6 vs V2, V7 vs V2) are all nonsignificant $( p > 0 . 5 )$ . This suggests that MS is associated with a higher mean but also with greater fold-to-fold variability, so its net effect is not statistically separable from LE alone with 5 folds.

Non-parametric robustness check and multiplecomparison correction. Because paired t-tests assume normality and $n \ = \ 5 ,$ and because the table reports 5 computable comparisons, we apply two robustness checks. First, under the Wilcoxon signed-rank test, none of the comparisons against V0 reach significance: V2 vs V0 gives $p \ = \ 0 . 1 2 5$ , V3 vs V0 gives $p \ = \ 0 . 1 8 8$ , V4 vs V0 gives $p = 0 . 1 8 8$ , and V7 vs V0 gives $p = 0 . 1 8 8$ . Second, under a Benjamini–Hochberg FDR correction at $q = 0 . 0 5$ over the 5 computable comparisons, the paired t-test adjusted p-values are V2 vs V0 $p _ { \mathrm { B H } } = 0 . 2 0 5 5$ , V3 vs $\mathsf { V } 0 \ p _ { \mathrm { B H } } = 0 . 2 0 5 5$ , V4 vs V0 $p _ { \mathrm { B H } } = 0 . 2 0 5 5 ,$ , and V7 vs V0 $p _ { \mathrm { B H } } = 0 . 2 0 9 0 .$ , none of which is significant. The nominally significant improvement of LE therefore does not survive either a non-parametric test or a multiple-comparison correction, and the single-label advantage of CALE should be treated as exploratory rather than established.

Multi-label performance at target density. We evaluate on the multi-label test set with threshold scanning to match the true label density (2.5). Table V reports the results. MS alone $( \mathsf { V } 0 \to \mathsf { V } 1 )$ is associated with $\mathbf { a + 0 . 0 3 1 6 }$ change in multi-label Macro-F1, while LE alone is associated with +0.0154. Adding MS on top of LE (V2 → V5) is associated with +0.0178.

## E. Cross-Backbone Evaluation

We evaluate CALE on seven mainstream backbones. Table VII reports the single-label and multi-label results on the self-built dataset (1/8 validation split).

Three observations follow. First, CALE improves singlelabel performance on 6 of 7 backbones by +0.70 to +2.21 points, with EfficientNet-B0 + CALE achieving the best single-label result of 0.8099. Second, under the full-data multicondition setting with threshold scanning to match the target label density, CALE is associated with multi-label improvements of +3.28 to +6.27 points. Third, DeiT-Tiny is the only exception on single-label performance (−1.18 points). The cross-backbone picture under the threshold-scanned setting is favorable; we show below that this picture depends on threshold calibration, and that a fixed threshold produces very different behavior under different training data scales.

TABLE V: Multi-label test set results at AvgLabels ≈ 2.5.
<table><tr><td>Variant</td><td>LE</td><td>MS</td><td>SE</td><td>Macro-F1</td></tr><tr><td>V0</td><td></td><td>一</td><td>一</td><td>0.5607</td></tr><tr><td>V1</td><td></td><td>√</td><td></td><td>0.5923</td></tr><tr><td>V2</td><td>√</td><td>一</td><td>一</td><td>0.5761</td></tr><tr><td>V3</td><td>一</td><td>一</td><td>√</td><td>0.5643</td></tr><tr><td>V4</td><td>一</td><td>√</td><td>√</td><td>0.5928</td></tr><tr><td>V5</td><td>√</td><td>√</td><td>一</td><td>0.5939</td></tr><tr><td>V6</td><td>√</td><td>一</td><td>√</td><td>0.5844</td></tr><tr><td>V7</td><td>√</td><td>√</td><td>√</td><td>0.5966</td></tr></table>

TABLE VI: Paired tests across the 5 folds on Validation UAR. V0 is a reproduction with Val $\mathrm { U A R } = 0 . 7 6 6 7 .$ . Columns ${ ^ { * } t } / { p _ { t } } ^ { * }$ are a paired t-test; column $^ {  } p _ { W } ^ { \mathrm { ~  ~ } } \mathrm { ~ i s ~ } \mathrm { ~ a ~ }$ Wilcoxon signed-rank test. After Benjamini–Hochberg FDR correction at $q = 0 . 0 5$ over the 5 computable comparisons, none of the paired t-test comparisons are significant (p<sub>BH</sub> ≥ 0.2055); the same holds for the Wilcoxon comparisons $( p _ { \mathrm { B H } } \ge 0 . 2 3 4 4 )$ . We therefore do not claim a statistically established single-label advantage.
<table><tr><td>Comparison</td><td>△ mean</td><td>t</td><td>pt</td><td>pw</td></tr><tr><td>V2 (LE) vs V0</td><td> $+ 0 . 0 0 7 8$ </td><td>2.917</td><td>0.0436</td><td>0.1250</td></tr><tr><td>V3 (SE) vs VO</td><td> $+ 0 . 0 0 5 8$ </td><td>1.947</td><td>0.1233</td><td>0.1875</td></tr><tr><td>V4 (MS+SE) vs V0</td><td> $+ 0 . 0 0 5 5$ </td><td>2.046</td><td>0.1102</td><td>0.1875</td></tr><tr><td>V5 (LE+MS) vs V0</td><td> $+ 0 . 0 1 0 5$ </td><td>2.614</td><td>0.0589</td><td></td></tr><tr><td>V6 (LE+SE) vs V0</td><td> $+ 0 . 0 0 9 6$ </td><td>3.270</td><td>0.0308</td><td></td></tr><tr><td>V7 (Full) vs V0</td><td>+0.0096</td><td>1.685</td><td>0.1672</td><td>0.1875</td></tr><tr><td>V5 vs V2</td><td>+0.0027</td><td>0.663</td><td>0.5436</td><td></td></tr><tr><td>V6 vsV2</td><td>+0.0018</td><td>0.479</td><td>0.6569</td><td></td></tr><tr><td> $\mathrm { \Delta V } 7 \ \mathrm { \bf v s \ } \mathrm { \bf \ V } 2$ </td><td>+0.0018</td><td>0.238</td><td>0.8236</td><td>1.0000</td></tr><tr><td>V7 vs V5</td><td>-0.0009</td><td>-0.149</td><td>0.8893</td><td></td></tr></table>

## F. Public Benchmarks

1) RAF-DB: We evaluate CALE on RAF-DB. Table VIII reports the results. With raw RAF-DB training data (single condition), CALE drops to a Macro-F1 of 0.5226. After augmenting the training set with four-condition variants (49,084 images), CALE recovers to 0.7475, exceeding the raw baseline (0.6769) by +7.1 points but remaining below the multicondition baseline (0.7603).

TABLE VII: Cross-backbone evaluation. Single-label results use the 1/8 validation split; multi-label results use the independent test set at AvgLabels ≈ 2.5. The multi-label column uses a threshold scan that matches the target label density on the evaluation set; these numbers should be interpreted as an upper bound rather than as achievable performance (see Section V-A).
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Params (M)</td><td colspan="2">Single-label Macro-F1</td><td colspan="2">Multi-label Macro-F1</td></tr><tr><td>Baseline</td><td>+CALE</td><td>Baseline</td><td>+CALE</td></tr><tr><td>ResNet-18</td><td>11.2</td><td>0.7830</td><td>0.7951</td><td>0.5596</td><td>0.5959</td></tr><tr><td>ResNet-34</td><td>21.8</td><td>0.7829</td><td>0.8016</td><td>0.5637</td><td>0.5965</td></tr><tr><td>ResNet-50</td><td>25.6</td><td>0.7834</td><td>0.8011</td><td>0.5566</td><td>0.5956</td></tr><tr><td>MobileNetV2</td><td>3.5</td><td>0.7726</td><td>0.7859</td><td>0.5321</td><td>0.5948</td></tr><tr><td>DenseNet121</td><td>8.0</td><td>0.7756</td><td>0.7977</td><td>0.5379</td><td>0.5985</td></tr><tr><td>EfficientNet-B0</td><td>5.3</td><td>0.8029</td><td>0.8099</td><td>0.5667</td><td>0.6023</td></tr><tr><td>DeiT-Tiny</td><td>5.7</td><td>0.7749</td><td>0.7631</td><td>0.5157</td><td>0.5745</td></tr></table>

TABLE VIII: RAF-DB evaluation (1/8 split used for baseline; augmented training for multi-condition settings). The row “V1 diagnostic (raw)” corresponds to a 7-dimensional baseline trained with the multi-strategy training (MS) configuration on raw RAF-DB.
<table><tr><td>Setting</td><td>Training Data</td><td>Test Macro-F1</td></tr><tr><td>Baseline (raw)</td><td>12K</td><td>0.6769</td></tr><tr><td>V1 diagnostic (raw)</td><td>12K</td><td>0.7555</td></tr><tr><td>CALE (raw)</td><td>12K</td><td>0.5226</td></tr><tr><td>Baseline (multi-condition)</td><td>49K</td><td>0.7603</td></tr><tr><td>CALE (multi-condition)</td><td>49K</td><td>0.7475</td></tr></table>

TABLE IX: RAF-ML multi-label evaluation under the fixed decision threshold 0.5. All rows use the fixed decision threshold 0.5; threshold-calibrated variants are not reported because they require evaluation-set label density.
<table><tr><td>Setting</td><td>AvgLabels</td><td>Macro-F1</td></tr><tr><td>Prior BCE baseline [4]</td><td></td><td>0.5954</td></tr><tr><td>Prior ASL [21]</td><td></td><td>0.6481</td></tr><tr><td>Prior ASL with soft labels [21]</td><td></td><td>0.6500</td></tr><tr><td>Baseline (raw)</td><td>2.06</td><td>0.6835</td></tr><tr><td>CALE (raw, Focal)</td><td>5.61</td><td>0.5526</td></tr><tr><td>Baseline (multi-cond.)</td><td>2.08</td><td>0.7359</td></tr><tr><td>CALE (multi-cond., Focal)</td><td>4.36</td><td>0.6411</td></tr><tr><td>CALE (multi-cond., BCE)</td><td>2.45</td><td>0.7330</td></tr></table>

2) RAF-ML: Table IX reports the multi-label benchmark. With raw training data, CALE achieves a Macro-F1 of 0.6654 at the calibrated threshold (0.99), below our own baseline (0.6835). With multi-condition training, CALE reaches 0.7330 at the fixed threshold 0.5 and 0.7207 at AvgLabels = 2.20, both below the multi-condition baseline (0.7359 at AvgLabels = 2.08).

3) Threshold Sensitivity: Fig. 3 plots Macro-F1 and UAR as functions of the decision threshold for the baseline (V0) and the full CALE model (V7). The baseline softmax curve saturates at AvgLabels = 1.0 once the threshold exceeds 0.5. In contrast, the CALE aggregation score spans [0, 2.5], and the curve decreases smoothly from AvgLabels = 5.6 at 0.5 to 1.0 at 1.2. This behavior reflects the range of the aggregated score rather than a performance advantage, as we analyze in Section V.

## G. Condition-wise Analysis

Table X reports per-condition UAR on the augmented RAF-DB test set. CALE shows comparable performance across all four conditions, with weak light (0.7482) slightly exceeding standard (0.7437).

TABLE X: Per-condition UAR on augmented RAF-DB test set.
<table><tr><td>Condition</td><td>Baseline</td><td>CALE</td></tr><tr><td>Standard</td><td>0.7555</td><td>0.7437</td></tr><tr><td>Weak light</td><td>0.7514</td><td>0.7482</td></tr><tr><td>Strong light</td><td>0.7078</td><td>0.6925</td></tr><tr><td>Low resolution</td><td>0.7142</td><td>0.7190</td></tr></table>

TABLE XI: Per-emotion F1 on the RAF-DB test set (1/8 split). ∆ is CALE minus baseline.
<table><tr><td>Emotion</td><td>Baseline</td><td>CALE</td><td>∆</td></tr><tr><td>Angry</td><td>0.8070</td><td>0.7905</td><td>-0.0165</td></tr><tr><td>Disgust</td><td>0.7044</td><td>0.7295</td><td>+0.0251</td></tr><tr><td>Fear</td><td>0.8247</td><td>0.8010</td><td>-0.0237</td></tr><tr><td>Happy Sad</td><td>0.8458</td><td>0.8977</td><td>+0.0519</td></tr><tr><td>Surprise</td><td>0.7817 0.7062</td><td>0.7925 0.7233</td><td>+0.0108 +0.0171</td></tr><tr><td>Natural</td><td>0.8111</td><td>0.8311</td><td>+0.0200</td></tr><tr><td>Mean</td><td>0.7830</td><td>0.7951</td><td>+0.0121</td></tr></table>

1) Per-emotion Analysis: Table XI reports per-emotion F1 on the RAF-DB test set. CALE improves five of the seven categories, most on happy (+0.0519) and disgust (+0.0251), and loses slightly on angry (−0.0165) and fear (−0.0237). The mean gain is +0.0121, consistent with the +1.29 points reported for the full model.

## V. DIAGNOSTIC ANALYSIS: WHEN DOES CALE HELP?

The results in Sections IV-C to IV-F show a mixed pattern: CALE improves single-label FER, but its multi-label behavior is not uniform. To understand why, we conduct a set of diagnostic experiments under four training-data regimes (10%, 25%, 50%, 100% of the training set), a single fold, and five backbones (ResNet-18/34/50, MobileNetV2, EfficientNet-B0). The reduced-data regimes amplify data-scale effects and keep the diagnostic cost manageable.

## A. Scope and Limitations

We emphasize that the diagnostics are preliminary and cover a single fold, four data regimes, and a limited set of backbones. The behavior reported below may differ at additional backbones. We frame the findings as directional evidence for a working hypothesis, not as a proven causal mechanism, and identify the additional experiments needed to test the hypothesis in Section VI-J.

![](images/6fbf62d15368500dd69ea4e9a2ab7100d10d2af6b4f3ed62e6601fba90d9a523.jpg)

![](images/91c8b15c7534a5644e82ddca52fc2fe53b44e7797a9b5f2fad70e041058d6f6b.jpg)  
Fig. 3: Threshold sensitivity on the multi-label test set. (a) Macro-F1 vs threshold. (b) UAR vs threshold. The CALE curve spans a wider threshold range because the aggregation score is not normalized, not because the model predicts more labels correctly.

We additionally note three limitations of the diagnostic study that are important for interpreting the results. First, the multi-label evaluation uses an uncalibrated threshold of 0.5; a threshold tuned on a held-out multi-label validation set would change the absolute AvgLabels and may change some of the qualitative conclusions. The threshold scans used in Tables V, VII, and IX match the target label density on the evaluation set and therefore use information that would not be available at deployment time; those numbers should be interpreted as upper bounds rather than as achievable performance. Second, the diagnostics use a single random seed and a single fold; seed-to-seed variability is not characterized. Third, the fair multi-label baseline comparison uses a single ResNet-18 backbone; extending it to the other backbones would strengthen the conclusion but is left as future work.

## B. Multi-Label Behavior Depends on the Training Data Scale

Tables XII, XIII, and XIV report the diagnostic results at 10%, 100%, and intermediate data regimes, respectively.

Three observations follow. First, the range of AvgLabels shrinks drastically when the training set is enlarged. At the 10% regime, AvgLabels spans 28× (0.24 to 6.79); at 25% it collapses to a 1.5× range (0.74 to 1.14); at 50% and 100% it remains below 1.0 for all backbones. The large spread in the reduced-data regime is therefore a small-data effect, and the transition from over-prediction to under-prediction occurs between 10% and 25% of the training set. Second, the direction of the mis-calibration reverses with data scale: under limited data, most backbones over-predict (AvgLabels greater than the target of 2.5); from 25% onward, all backbones under-predict (AvgLabels below 1.0). Third, the effect of the soft-label weight is consistent across data scales: SOFT=0.5 always yields higher AvgLabels than SOFT=0.0, though the gap narrows at full data.

We trace this behavior to the aggregation formula in Eq. (5), $\begin{array} { r c l } { p _ { e } } & { = } & { \sigma ( x _ { e , c ^ { * } } ) \ + \ 0 . 5 \sum _ { c \neq c ^ { * } } \sigma ( x _ { e , c } ) } \end{array}$ . Under the intended regime, the main condition branch converges to $\sigma ( x _ { e , c ^ { * } } ) \approx 1$ and the other branches to $\sigma ( x _ { e , c } ) ~ \approx ~ 0 . 5 .$ , giving $p _ { e }$ ≈ $1 + 0 . 5 \times 3 \times 0 . 5 = 1 . 7 5$ , which is comfortably above the fixed threshold 0.5. In practice, however, the condition branches do not reach these targets under either data regime. Under limited data, the branches cannot learn condition invariance, so all four sigmoids hover around 0.5 and $p _ { e } \approx 1 . 2 5$ , still above threshold (over-prediction). Once the model is trained on at least 25% of the data, the branches learn stronger condition specificity, but the main branch does not saturate to 1, and the other branches fall below 0.5; the resulting $p _ { e }$ falls below 0.5, producing under-prediction. The aggregation formula thus inherits a systematic bias that cannot be corrected by training alone; a calibrated threshold or a learned aggregation rule is required, and we leave this as a primary direction for future work.

To verify this explanation, we measured the mean sigmoid activation of the main condition branch $\sigma ( x _ { e , c ^ { * } } )$ and of the non-main branches $\sigma ( x _ { e , c \neq c ^ { * } } )$ on the test set for the ResNet-18 SOFT=0.5 configuration under three data regimes. At 10% training data, the mean main-branch activation is 0.52 and the mean non-main activation is 0.48, giving an expected aggregation score of $0 . 5 2 \ : + \ : 0 . 5 \ : \times \ : 3 \ : \times \ : 0 . 4 8 \ : = \ : 1 . 2 4$ , well above the fixed threshold of 0.5 and consistent with the observed over-prediction. At 25% training data, the mean main-branch activation drops to 0.41 and the mean non-main activation drops to 0.22, giving an expected aggregation score of $0 . 4 1 + 0 . 5 \times 3 \times 0 . 2 2 = 0 . 7 4$ , still above the threshold but no longer by a wide margin. At full training data, the mean main-branch activation is 0.31 and the mean non-main activation is 0.14, giving an expected aggregation score of $0 . 3 1 + 0 . 5 \times 3 \times 0 . 1 4 = 0 . 5 2$ , essentially at the threshold and consistent with the observed near-zero AvgLabels. These measurements confirm that the aggregation formula carries a systematic bias that is not corrected by training alone and that its direction is determined by how far the branch activations depart from the soft-label targets of 1.0 and 0.5.

## C. Threshold-Free Evaluation Shows Stable Ranking Quality

The AvgLabels behavior documented above is a property of the decision rule (fixed threshold 0.5), not of the ranking quality of the underlying model. To separate the two, we report the threshold-free mean average precision (mAP) under the unified protocol (sigmoid output, threshold 0.5 for macro-F1). Table XV summarizes the results at 10% data across five backbones. While AvgLabels spans 28× across these configurations, mAP spans only 1.28× (0.4264 to 0.5468). This confirms that the multi-label failure mode documented above is dominated by the aggregation step rather than by any loss of per-class ranking ability.

TABLE XII: Diagnostic experiment under the 10% subject-level training subset with a single fold. Val Macro-F1 uses the single-label argmax protocol on the in-distribution validation split; Test Macro-F1 and AvgLabels use the independent multi-label test set. Val/Test are reported for the best epoch. The AvgLabels column is measured at threshold 0.5. A hyperparameter sensitivity study on EfficientNet-B0 (the configuration with the largest AvgLabels) shows that its AvgLabels is itself hyperparameter-dependent: it ranges from 1.70 (lr=1e-5, bs=64) to 7.00 (lr=5e-6, bs=128) across four configurations, with the original configuration (lr=1e-5, bs=128) yielding 6.79. This suggests that the extreme value in the table is not a stable architectural property, and that the range reported here should be interpreted as an upper bound.
<table><tr><td>Backbone</td><td>Setting</td><td>Params (M)</td><td>Val F1</td><td>Test F1</td><td>AvgLabels</td></tr><tr><td>ResNet-18</td><td>CALE, SOFT=0.5</td><td>11.2</td><td>0.4069</td><td>0.5200</td><td>5.86</td></tr><tr><td>ResNet-18</td><td>CALE, SOFT=0.0</td><td>11.2</td><td>0.3533</td><td>0.5139</td><td>5.98</td></tr><tr><td>ResNet-34</td><td>CALE, SOFT=0.5</td><td>21.8</td><td>0.4272</td><td>0.5238</td><td>3.08</td></tr><tr><td>ResNet-34</td><td>CALE, SOFT=0.0</td><td>21.8</td><td>0.3645</td><td>0.3686</td><td>1.72</td></tr><tr><td>ResNet-50</td><td>CALE, SOFT=0.5</td><td>25.6</td><td>0.4582</td><td>0.3359</td><td>0.96</td></tr><tr><td>ResNet-50</td><td>CALE, SOFT=0.0</td><td>25.6</td><td>0.3975</td><td>0.1051</td><td>0.24</td></tr><tr><td>MobileNetV2</td><td>CALE, SOFT=0.5</td><td>3.5</td><td>0.4167</td><td>0.4785</td><td>2.71</td></tr><tr><td>MobileNetV2</td><td>CALE, SOFT=0.0</td><td>3.5</td><td>0.3424</td><td>0.3819</td><td>1.68</td></tr><tr><td>EfficientNet-B0</td><td>CALE, SOFT=0.5</td><td>5.3</td><td>0.2776</td><td>0.5206</td><td>6.79</td></tr></table>

TABLE XIII: Diagnostic experiment under the full training set (30,240 images) with a single fold. Same metrics and protocol as Table XII. Under full data, all five backbones collapse to the under-prediction regime (AvgLabels < 1). The AvgLabels range narrows from 28× (Table XII) to 1.9× (this table).
<table><tr><td>Backbone</td><td>Setting</td><td>Params (M)</td><td>Val F1</td><td>Test F1</td><td>AvgLabels</td></tr><tr><td>ResNet-18</td><td>CALE, SOFT=0.5</td><td>11.2</td><td>0.7606</td><td>0.3153</td><td>0.66</td></tr><tr><td>ResNet-18</td><td>CALE, SOFT=0.0</td><td>11.2</td><td>0.7393</td><td>0.2439</td><td>0.47</td></tr><tr><td>ResNet-34</td><td>CALE, SOFT=0.5</td><td>21.8</td><td>0.7576</td><td>0.3845</td><td>0.87</td></tr><tr><td>ResNet-34</td><td>CALE, SOFT=0.0</td><td>21.8</td><td>0.7569</td><td>0.2772</td><td>0.53</td></tr><tr><td>ResNet-50</td><td>CALE, SOFT=0.5</td><td>25.6</td><td>0.7496</td><td>0.3302</td><td>0.77</td></tr><tr><td>ResNet-50</td><td>CALE, SOFT=0.0</td><td>25.6</td><td>0.7496</td><td>0.2636</td><td>0.54</td></tr><tr><td>MobileNetV2</td><td>CALE, SOFT=0.5</td><td>3.5</td><td>0.7283</td><td>0.3369</td><td>0.74</td></tr><tr><td>MobileNetV2</td><td>CALE, SOFT=0.0</td><td>3.5</td><td>0.7174</td><td>0.2719</td><td>0.54</td></tr><tr><td>EfficientNet-B0</td><td>CALE, SOFT=0.5</td><td>5.3</td><td>0.7399</td><td>0.3923</td><td>0.87</td></tr></table>

TABLE XIV: AvgLabels at threshold 0.5 as a function of the training-data fraction for two backbones. The transition from over-prediction (AvgLabels > 2) to under-prediction (AvgLabels < 1) occurs between 10% and 25%; from 25% onward the AvgLabels are stable.
<table><tr><td>Backbone</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>ResNet-18</td><td>5.86</td><td>1.14</td><td>0.87</td><td>0.66</td></tr><tr><td>ResNet-34</td><td>3.08</td><td>0.74</td><td>0.95</td><td>0.87</td></tr></table>

## D. A Fair Multi-Label Baseline

The data-scale dependence above raises a natural question: is CALE’s multi-label behavior actually better than a simple baseline that is allowed to make multi-label decisions? To answer this, we compare CALE against two seven-dimensional sigmoid baselines under the same ResNet-18 backbone, the same 10% subject-level training subset, and the same decision protocol (sigmoid output, threshold 0.5). The first baseline is trained with standard binary cross-entropy (BCE); the second is trained with Asymmetric Loss (ASL) [21] with $\gamma _ { - } = 4$ $\gamma _ { + } = 0 ;$ , and clipping 0.05. Table XVII reports the results.

Two observations follow. First, macro-F1 at threshold 0.5 is strongly coupled to the number of predicted labels: configurations with a large AvgLabels (e.g., ResNet-18 CALE at 5.86, EfficientNet-B0 CALE at 6.79) reach higher macro-

F1 simply because recall saturates, while configurations with a small AvgLabels (e.g., ResNet-50 CALE at 0.96) collapse toward low macro-F1 even when their per-class ranking is comparable. Macro-F1 at a fixed threshold is therefore not a fair comparison across configurations whose AvgLabels differ by more than an order of magnitude. Second, the thresholdfree metric mAP avoids this coupling. Under mAP, the sevendimensional BCE baseline achieves 0.5055 and the ASL baseline achieves 0.5175; the CALE configurations span 0.4264 to 0.5468. The best CALE configuration (ResNet-34, 0.5468) exceeds both baselines, but the ASL baseline exceeds three of the five CALE configurations and falls within 0.03 of the best. CALE therefore does not exhibit a consistent architectureindependent advantage in ranking quality on multi-label FER.

This result should be read together with the data-scale dependence documented above: CALE’s multi-label failure mode under reduced data is dominated by miscalibration of the aggregation formula, not by a loss of per-class ranking ability. The near-constant mAP across backbones confirms that the underlying 7-class probabilities are comparable; what differs is how they are aggregated into a decision. This reinforces the practical recommendation that any multi-label deployment of CALE must calibrate the aggregation step at the deployment data scale.

TABLE XV: Threshold-free mean average precision (mAP) and AvgLabels at threshold 0.5 for the five backbones under the 10% data regime. All values use the unified protocol. mAP is stable across backbones (1.28× range) while AvgLabels varies by 28×, localizing the failure to the aggregation step.
<table><tr><td>Backbone</td><td>AvgLabels</td><td>mAP</td><td>Test F1@0.5</td></tr><tr><td>ResNet-18</td><td>5.86</td><td>0.5005</td><td>0.5200</td></tr><tr><td>ResNet-34</td><td>3.08</td><td>0.5468</td><td>0.5237</td></tr><tr><td>ResNet-50</td><td>0.96</td><td>0.5220</td><td>0.3356</td></tr><tr><td>MobileNetV2</td><td>2.71</td><td>0.5045</td><td>0.4786</td></tr><tr><td>EfficientNet-B0</td><td>6.79</td><td>0.4264</td><td>0.5207</td></tr><tr><td>Range (max /min)</td><td>28×</td><td>1.28×</td><td>1.56×</td></tr></table>

TABLE XVI: Per-class average precision (AP) for four backbones under the 10% data regime, ResNet-18 + CALE SOFT=0.5. Rows are sorted by the ResNet-18 AP. The per-class ordering is not consistent across backbones (e.g., disgust ranks above surprise on ResNet-18 and ResNet-34, but below surprise on MobileNetV2 and EfficientNet-B0), which further supports the calibration hypothesis over a class-discriminability explanation.
<table><tr><td>Emotion</td><td>ResNet-18</td><td>ResNet-34</td><td>MobileNetV2</td><td>EffB0</td></tr><tr><td>Angry</td><td>0.347</td><td>0.322</td><td>0.363</td><td>0.330</td></tr><tr><td>Disgust</td><td>0.497</td><td>0.618</td><td>0.450</td><td>0.459</td></tr><tr><td>Fear</td><td>0.303</td><td>0.413</td><td>0.252</td><td>0.292</td></tr><tr><td>Happy</td><td>0.745</td><td>0.705</td><td>0.741</td><td>0.514</td></tr><tr><td>Sad</td><td>0.504</td><td>0.545</td><td>0.482</td><td>0.377</td></tr><tr><td>Surprise</td><td>0.478</td><td>0.539</td><td>0.640</td><td>0.374</td></tr><tr><td>Natural</td><td>0.629</td><td>0.686</td><td>0.603</td><td>0.641</td></tr><tr><td>mAP</td><td>0.501</td><td>0.547</td><td>0.505</td><td>0.426</td></tr></table>

TABLE XVII: Fair multi-label comparison under the 10% data regime. All entries are evaluated under the unified decision protocol (sigmoid output, threshold 0.5). macro-F1@0.5 is coupled to the number of predicted labels and is not comparable across configurations; mAP is threshold-free and is the more meaningful comparison. All entries use the same ResNet-18 backbone and the same 10% subject-level training subset.
<table><tr><td>Method</td><td>AvgLabels</td><td>macro-F1@0.5</td><td>mAP</td></tr><tr><td>7-dim sigmoid + BCE</td><td>1.14</td><td>0.3460</td><td>0.5055</td></tr><tr><td>7-dim sigmoid + ASL</td><td>2.53</td><td>0.4808</td><td>0.5175</td></tr><tr><td>CALE (ResNet-18)</td><td>5.86</td><td>0.5200</td><td>0.5005</td></tr><tr><td>CALE (ResNet-34)</td><td>3.08</td><td>0.5237</td><td>0.5468</td></tr><tr><td>CALE (ResNet-50)</td><td>0.96</td><td>0.3356</td><td>0.5220</td></tr><tr><td>CALE (MobileNetV2)</td><td>2.71</td><td>0.4786</td><td>0.5045</td></tr><tr><td>CALE (EfficientNet-B0)</td><td>6.79</td><td>0.5207</td><td>0.4264</td></tr></table>

## E. Condition Classifier Behavior

We characterize the rule-based condition classifier from two angles: its output distribution on the training and test sets, and its sensitivity to the thresholds.

The training set distribution of the classifier’s outputs under the default thresholds (80, 120, 10) is 24.9% weak light, 31.6% strong light, 19.5% low resolution, and 24.0% standard. On the multi-label test set, the corresponding proportions are 27.4%, 29.0%, 19.6%, and 24.0%. The distributions are roughly similar between the two sets, which is consistent with the augmentation procedure used to construct the training set.

The threshold sensitivity, however, is severe. Table XVIII reports the class distribution under five threshold scalings (0.8× to 1.2× the default values). On the training set, the strong-light proportion varies from 48.6% at scale 0.8 to 19.9% at scale 1.2, a range of 28.7 percentage points. On the test set, the range is similar (45.0% to 17.5%). The distribution is balanced only at the specific scale 1.0; small perturbations of the thresholds completely change the shape of the distribution.

This sensitivity has two consequences. First, the soft-label target in Eq. (4) depends on the main condition position $c ^ { * } ,$ , and therefore on the classifier’s output. Under a ±20% threshold perturbation, a substantial fraction of training images receive a different main condition, which changes the softlabel target. Second, the aggregation weight in Eq. (5) also depends on the classifier’s output, and shifts in the dominant class change which branch is up-weighted at inference. Both effects contribute to the data-scale dependence documented above, though we do not attempt to quantify their relative contributions in the present study.

TABLE XVIII: Condition classifier sensitivity to threshold perturbation. The default thresholds (80, 120, 10) are scaled by the factor in the first column. Numbers are the percentage of images assigned to each condition.
<table><tr><td>Scale</td><td>Weak</td><td>Strong</td><td>Low-res</td><td>Standard</td></tr><tr><td></td><td></td><td>Training set</td><td></td><td></td></tr><tr><td>0.8×</td><td>18.4</td><td>48.6</td><td>16.6</td><td>16.4</td></tr><tr><td>0.9×</td><td>21.7</td><td>39.4</td><td>18.2</td><td>20.8</td></tr><tr><td>1.0×</td><td>24.9</td><td>31.6</td><td>19.5</td><td>24.0</td></tr><tr><td>1.1×</td><td>28.0</td><td>25.6</td><td>20.9</td><td>25.5</td></tr><tr><td>1.2×</td><td>32.1</td><td>19.9</td><td>22.1</td><td>26.0</td></tr><tr><td></td><td></td><td>Test set</td><td></td><td></td></tr><tr><td>0.8×</td><td>18.3</td><td>45.0</td><td>16.4</td><td>20.2</td></tr><tr><td>0.9×</td><td>22.5</td><td>36.4</td><td>18.1</td><td>23.0</td></tr><tr><td>1.0×</td><td>27.4</td><td>29.0</td><td></td><td></td></tr><tr><td>1.1×</td><td>30.3</td><td></td><td>19.6</td><td>24.0</td></tr><tr><td></td><td></td><td>23.0</td><td>20.9</td><td>25.8</td></tr><tr><td>1.2×</td><td>35.1</td><td>17.5</td><td>22.5</td><td>24.9</td></tr></table>

## VI. DISCUSSION

## A. Design Guideline for Sum-Based Label Expansion

The diagnostic results in Section V support the following design guideline. A sum-based label expansion rule of the form $\begin{array} { r } { p _ { e } = \sum _ { c } w _ { c } \sigma ( x _ { e , c } ) } \end{array}$ with $\textstyle \sum _ { c } w _ { c } > 1$ is miscalibrated for fixed-threshold multi-label decision unless the branch activations satisfy a saturation condition: the main branch must reach $\sigma ( x _ { e , c ^ { * } } )  1$ and the non-main branches must converge to the soft-label target $w _ { c \neq c ^ { * } } / w _ { c ^ { * } }$ . We measured this condition directly for CALE and found that it is violated under every data regime we tested: at 10% data both branch groups hover near 0.5 (over-prediction), and at full data the main branch does not saturate while the non-main branches fall below the target (under-prediction). Any practitioner planning to use sum-based label expansion for multi-label FER should therefore (i) calibrate the decision threshold on held-out data at the deployment scale (post-hoc threshold calibration in the spirit of [33]), or (ii) learn the aggregation weights jointly with the backbone, or (iii) replace the summation with a normalized aggregation that is invariant to the number of conditions.

## B. Positive Findings

On the self-built dataset, the 5-fold single-label evaluation shows that CALE improves Macro-F1 by +1.29 points over a seven-dimensional baseline (Table IV). The improvement is nominally significant under an uncorrected paired t-test on the LE-only variant $( p \ : = \ : 0 . 0 4 3 6 )$ , and the combination of LE with SE achieves the strongest paired mean improvement $( p = 0 . 0 3 0 8 )$ . However, the effect size is less than one Macro-F1 point, and none of the comparisons survives a Benjamini– Hochberg FDR correction at $q = 0 . 0 5 ~ ( p _ { \mathrm { B H } } \geq 0 . 2 0 5 5 )$ or a Wilcoxon signed-rank test at $n = 5 \ ( p \ge 0 . 1 2 5 )$ . We therefore treat these single-label gains as exploratory evidence that labelspace organization can provide a useful regularization signal, not as a confirmed statistical advantage.

## C. Negative Findings: Data-Scale Dependence

CALE’s multi-label behavior is primarily a function of the training data scale rather than of the backbone architecture. At 10% training data, AvgLabels spans 28× across five backbones (0.24 to 6.79); at 25% it collapses to a 1.5× range (0.74 to 1.14); from 25% onward all backbones under-predict (AvgLabels below 1.0). A hyperparameter sensitivity study on EfficientNet-B0 shows that its extreme AvgLabels at 10% data is itself hyperparameter-dependent (1.70 to 7.00 across four configurations), so the 28× spread should be read as an upper bound rather than as a stable architectural property. We trace the systematic behavior to the aggregation formula in Eq. (5), which assumes that the main condition branch saturates to 1 and the other branches to 0.5; in practice, neither assumption holds. The formula thus carries a systematic bias that is a property of the design, not of any particular backbone. This reframes the failure mode from a per-backbone phenomenon to a per-data-scale phenomenon: CALE’s multi-label deployment requires calibration of the aggregation step regardless of the backbone, and calibration must be repeated whenever the training data scale changes. The threshold-free mAP, by contrast, is nearly invariant across backbones (1.28× range at 10% data), which indicates that the failure is localized to the aggregation step rather than to the learned representations.

## D. Negative Findings: Fair Baseline Comparison

Under the unified decision protocol (sigmoid output, threshold 0.5), the seven-dimensional BCE baseline reaches mAP

0.5055 and the ASL baseline reaches mAP 0.5175. The best CALE configuration (ResNet-34, 0.5468) exceeds both baselines, but the ASL baseline exceeds three of the five CALE configurations and falls within 0.03 of the best. This is a strong negative result for the practical value of the condition-aware label expansion on multi-label FER: the additional condition dimensions and summation aggregation do not translate into a consistent advantage over a simple multi-label baseline, and the comparison is sensitive to the choice of decision protocol.

## E. Practical Significance of the Single-Label Improvement

The paired tests in Table VI show that label expansion alone $( \mathrm { V } 2 \ \mathrm { ~ v s } \ \mathrm { V } 0 , \ + 0 . 0 0 7 8 , \ p _ { t } \ = \ 0 . 0 4 3 6 , \ p _ { \mathrm { B H } } \ = \ 0 . 2 0 5 5$ $p _ { W } = 0 . 1 2 5 )$ and label expansion combined with SE (V6 vs $\mathrm { V } 0 , + 0 . 0 0 9 6 , p _ { t } = 0 . 0 3 0 8 )$ reach nominal significance under an uncorrected paired t-test at the 0.05 level. However, the absolute effect size is less than one Macro-F1 point, and no comparison survives a Benjamini–Hochberg FDR correction or a Wilcoxon signed-rank test. Whether such a small improvement justifies the additional complexity of a 28-dimensional joint output, a rule-based condition classifier, a soft-label target, and an aggregation rule depends on the deployment scenario. In domains where individual percentage points of accuracy carry direct practical value (e.g., clinical assessment, safety-critical human-computer interaction), the improvement may be justified; in scenarios where implementation simplicity and robustness are prioritized, a standard seven-dimensional baseline may be preferable. We do not attempt to resolve this trade-off in the present study, but we emphasize that statistical significance and practical significance are not the same thing, and that the difference between the two deserves explicit consideration in applied FER.

## F. Why the Aggregation Formula Miscalibrates

The uncalibrated aggregation formula in Eq. (5) produces the observed data-scale dependence for two reasons. First, the formula assumes the soft-label targets 1.0 and 0.5 are attainable by the trained branch outputs; in practice, the sigmoid outputs of the condition branches are pulled toward a common value that depends on the training data scale. Under limited data, this common value is close to 0.5 (the soft-label target itself), producing over-prediction. Once the training set exceeds 25%, the model learns to distinguish conditions and pushes the non-main branches below 0.5, while the main branch does not saturate to 1, producing underprediction. Second, the fixed threshold 0.5 is not calibrated to the statistical range of $p _ { e }$ . The aggregation score $p _ { e } \in [ 0 , 2 . 5 ]$ has a distribution that shifts with the training data scale and with the backbone, and a single fixed threshold cannot accommodate both shifts. A natural fix is to replace the fixed aggregation with a learned module, or to calibrate the threshold on a held-out multi-label validation set; both are outside the scope of the present study and are listed as primary directions for future work.

## G. Condition Classifier Sensitivity

The rule-based condition classifier is extremely sensitive to its thresholds: $a \pm 2 0 \%$ perturbation of the default thresholds changes the proportion of the dominant class by up to 28.7 percentage points (Table XVIII). This sensitivity has two practical consequences. First, the soft-label target in Eq. (4) depends on the main condition position, which is assigned by the classifier; if the classifier’s output shifts under a different threshold, the soft-label target shifts accordingly, and the model is trained on a different objective. Second, the aggregation weight in Eq. (5) depends on the classifier’s output, and shifts in the dominant class change which branch is up-weighted at inference. This sensitivity is a limitation of the current study’s design; replacing the rule-based classifier with a learnable classifier, or using soft condition labels rather than hard assignments, would be a natural direction for future work.

## H. Applications in Psychological Research

The proposed method has direct relevance to psychological research. First, condition-aware label expansion allows the model to handle degraded images (weak light, strong light, low resolution, standard) without additional quality estimation, making it suitable for naturalistic settings such as clinical interviews or field observations. Second, the multi-label output provides a more faithful representation of blended emotions, which is essential for studying complex affective states. For example, in studies of depression or anxiety, mixed emotions such as sadness and fear frequently co-occur; our method can provide objective, reproducible emotion scores that complement human ratings. We emphasize, however, that CALE’s multi-label scores require calibration on held-out data at the deployment data scale before they can be used in a practical setting.

## I. Limitations

Several limitations remain. First, the diagnostic experiments cover a single fold and four data regimes, which reveals the data-scale dependence but does not characterize seed-to-seed variability. Second, the five backbones in the diagnostic study do not exhaust the space of possible architectures. Third, the condition classifier is rule-based and highly sensitive to its thresholds; a learnable or soft-labeled alternative would be more robust but was not evaluated here. Fourth, EfficientNet-B0’s AvgLabels at the 10% data regime is hyperparameter-dependent: across four configurations (lr $\in \{ 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } \}$ and $\mathsf { b s } \in \{ 6 4 , 1 2 8 \}$ ), AvgLabels ranges from 1.70 to 7.00, with the configuration reported in Table XII yielding 6.79. The extreme value in that table should be interpreted as an upper bound rather than as a stable architectural property. Fifth, the training set is singlelabel while the test set is multi-label, which is a structural mismatch; a fully multi-label training setup would require a separate study. Sixth, CALE’s performance on public singlelabel and multi-label benchmarks is below the corresponding baselines at the ResNet-18 scale. Seventh, the diagnostic study reports per-class metrics only under the 10% data regime. Eighth, the fair multi-label comparison uses two baselines (BCE and ASL) but only one backbone (ResNet-18); a wider suite of multi-label baselines and backbones would strengthen the comparison. Ninth, the multi-label diagnostics use an uncalibrated threshold of 0.5; a calibrated threshold or a learned aggregation module would change the absolute AvgLabels and possibly some of the qualitative conclusions. Tenth, the diagnostics use a single random seed and a single fold. Eleventh, the four quality conditions are produced by synthetic augmentations (rotation, translation, brightness, contrast, blur), and the rule-based condition classifier operates on statistics (brightness mean, Laplacian variance) that are correlated with these augmentations; the condition branches may therefore learn augmentation-type shortcuts rather than genuine quality invariance. Twelfth, the paired tests in Table VI use $n = 5 ,$ and after Benjamini–Hochberg FDR correction over the 5 computable comparisons none of the paired t-test comparisons are significant $( p _ { \mathrm { B H } } \ge 0 . 2 0 5 5 )$ ; the nominally significant comparisons in that table should therefore be read as exploratory.

## J. Future Work Directions

The diagnostic findings above point to five concrete experiments.

First, the data-scale dependence of CALE’s multi-label behavior should be mapped across all 5 folds and additional backbones (DenseNet121, DeiT-Tiny) at the intermediate ratios (25%, 50%) to confirm that the transition from overprediction to under-prediction occurs consistently between 10% and 25%.

Second, the aggregation formula should be replaced with a calibrated variant. Two natural options are (a) threshold calibration on a held-out multi-label validation set, and (b) a learned aggregation module whose weights are optimized jointly with the backbone. Both would test whether the underprediction under full data can be eliminated by proper calibration, or whether it reflects a deeper limitation of the 28- dimensional joint output.

Third, the condition classifier should be replaced with a learnable module or a soft-label scheme. The threshold sensitivity documented in Table XVIII is a fundamental limitation of the current design, and it affects both training and inference.

Fourth, the fair baseline comparison in Table XVII should be extended to the other four backbones and to additional multi-label losses (class-balanced loss, threshold-tuned softmax). A comprehensive multi-label baseline suite would clarify whether the seven-dimensional sigmoid baseline with ASL represents the true upper bound for the setting.

Fifth, a fully multi-label training setup should be evaluated. The structural mismatch between single-label training and multi-label evaluation may be a partial cause of the aggregation miscalibration; training CALE on genuinely multi-label data (e.g., RAF-ML’s multi-label annotations or an augmented version of the self-built dataset with multi-label targets) would separate the two effects.

## VII. CONCLUSION

This paper presented a diagnostic empirical study of Condition-Aware Label Expansion (CALE) for facial expression recognition. We evaluated the strategy on a selfbuilt dataset under 5-fold cross-validation, on seven backbone architectures under a full-data multi-condition setting, and on the public RAF-DB and RAF-ML benchmarks, and we conducted diagnostic experiments under four data regimes and five backbones to identify the boundary conditions under which the strategy helps or hurts.

Our results are predominantly negative. On single-label FER, CALE improves Macro-F1 by +1.29 points over a seven-dimensional baseline, but the improvement does not survive a Benjamini-Hochberg FDR correction $( p _ { \mathrm { B H } } = 0 . 2 0 5 5 )$ or a Wilcoxon signed-rank test $( p \ : = \ : 0 . 1 2 5 )$ , and the effect size is smaller than one Macro-F1 point. On multi-label FER the method’s behavior is dominated by an uncalibrated aggregation rule: the aggregate score at threshold 0.5 spans 28× across five backbones at 10% data and collapses to underprediction at full data. The threshold-free mAP is stable across the same backbones (1.28×), which localizes the failure to the aggregation step rather than to representation learning. Two simple seven-dimensional sigmoid baselines match or exceed most CALE configurations without any conditionaware machinery.

The practical guidance that follows is threefold. First, sumbased label expansion with a fixed decision threshold should not be deployed for multi-label FER without a calibration procedure; the summation rule carries a systematic bias that is not corrected by training alone. Second, CALE’s multi-label behavior under limited data and under full data differs in both magnitude and direction, so any multi-label deployment must calibrate the aggregation threshold (or replace the aggregation formula) on held-out data at the deployment data scale. Third, the condition classifier’s threshold sensitivity is a first-order limitation of the current design and should be addressed before CALE is used in safety-critical or naturalistic settings.

Future work will pursue five directions: mapping the datascale dependence across all folds and a wider range of backbones; replacing the fixed aggregation with a calibrated threshold or a learned aggregation module; replacing the rulebased condition classifier with a learnable module or a softlabel scheme; extending the fair baseline comparison to a wider suite of multi-label methods and backbones; and training CALE on genuinely multi-label data to separate the effect of the training/evaluation label structure mismatch from the effect of the aggregation formula.

## DATA AND CODE AVAILABILITY

The training set and test set are permanently archived on Zenodo: https://doi.org/10.5281/zenodo.23252243. The trained model weights are available at Figshare: https://figshare.com/ articles/dataset/CALE Model Weights for Multi-Label FER/34284447. The code and data-split indices are available at GitHub: https://github.com/yuanweihang810/CALE.

Ethical and copyright statement. The self-built dataset is constructed from publicly available images and video frames (films, TV series, short videos by online content creators) in which faces are shown voluntarily in public media. No private or sensitive personal information is involved. All images are used solely for non-commercial academic research, following the same principle adopted by public FER datasets such as RAF-DB and AffectNet. Source URLs are not redistributed; only MTCNN-aligned 224 × 224 crops are released for reproducibility. Any copyright holder who wishes to have specific samples removed may contact the author.

The RAF-DB and RAF-ML datasets are publicly available from their original sources [3], [4].

## ACKNOWLEDGMENT

The author is grateful to the providers of the RAF-DB and RAF-ML datasets for making their benchmarks publicly available, and to the open-source community for the PyTorch and torchvision libraries that facilitated this research. Special thanks are extended to colleagues who provided valuable feedback during the preparation of this work. The author used ChatGPT (OpenAI, GPT-4, 2024, https://chat.openai.com) for language polishing. The author reviewed and edited the content and takes full responsibility for the final version.

## REFERENCES

[1] P. Ekman and W. V. Friesen, “Constants across cultures in the face and emotion,” J. Pers. Soc. Psychol., vol. 17, no. 2, pp. 124–129, 1971.

[2] I. J. Goodfellow et al., “Challenges in representation learning: A report on three machine learning contests,” in Proc. Int. Conf. Neural Inf. Process. (ICONIP), 2013, pp. 117–124.

[3] S. Li, W. Deng, and J. Du, “Reliable crowdsourcing and deep localitypreserving learning for expression recognition in the wild,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2584– 2593.

[4] S. Li and W. Deng, “Blended emotion in-the-wild: Multi-label facial expression recognition using crowdsourced annotations and deep locality feature learning,” Int. J. Comput. Vis., vol. 127, no. 6–7, pp. 884–906, 2019.

[5] A. Mollahosseini, B. Hasani, and M. H. Mahoor, “AffectNet: A database for facial expression, valence, and arousal computing in the wild,” IEEE Trans. Affective Computing, vol. 10, no. 1, pp. 18–31, 2019.

[6] Y.-I. Tian, T. Kanade, and J. F. Cohn, “Recognizing action units for facial expression analysis,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 23, no. 2, pp. 97–115, 2001.

[7] Y. Zhang, C. Wang, and W. Deng, “Relative uncertainty learning for facial expression recognition,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2021, pp. 17616–17627.

[8] K. Wang, X. Peng, J. Yang, S. Lu, and Y. Qiao, “Suppressing uncertainties for large-scale facial expression recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 6896–6905.

[9] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7132– 7141.

[10] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2980–2988.

[11] A. Mollahosseini, D. Chan, and M. H. Mahoor, “Going deeper in facial expression recognition using deep neural networks,” in Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV), 2016, pp. 1–10.

[12] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.

[13] A. Vaswani et al., “Attention is all you need,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 5998–6008.

[14] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[15] D. Ruan, Y. Yan, S. Lai, Z. Chai, C. Shen, and H. Wang, “Feature decomposition and reconstruction learning for effective facial expression recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 7656–7665.

[16] Z. Zhao, Q. Liu, and S. Wang, “Learning deep global multi-scale and local attention features for facial expression recognition in the wild,” IEEE Trans. Image Process., vol. 30, pp. 6544–6556, 2021.

[17] H. Liu, H. Cai, Q. Lin, X. Li, and H. Liu, “Adaptive multilayer perceptual attention network for facial expression recognition,” IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 9, pp. 6253–6266, 2022.

[18] F. Xue, Q. Wang, and G. Guo, “TransFER: Learning relation-aware facial expression representations with transformers,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2021, pp. 3581–3590.

[19] M.-L. Zhang and Z.-H. Zhou, “A review on multi-label learning algorithms,” IEEE Trans. Knowl. Data Eng., vol. 26, no. 8, pp. 1819–1837, 2014.

[20] G. Tsoumakas, I. Katakis, and I. Vlahavas, “Mining multi-label data,” in Data Mining and Knowledge Discovery Handbook, Springer, 2010, pp. 667–685.

[21] T. Ridnik et al., “Asymmetric loss for multi-label classification,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2021, pp. 82–91.

[22] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, “Class-balanced loss based on effective number of samples,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 9268–9277.

[23] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.

[24] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional block attention module,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 3–19.

[25] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7794–7803.

[26] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment using multitask cascaded convolutional networks,” IEEE Signal Process. Lett., vol. 23, no. 10, pp. 1499–1503, 2016.

[27] J. L. Pech-Pacheco, G. Cristobal, J. Chamorro-Martinez, and J.´ Fernandez-Valdivia, “Diatom autofocusing in brightfield microscopy: A´ comparative study,” in Proc. Int. Conf. Pattern Recognit. (ICPR), 2000, pp. 314–317.

[28] P. Lucey, J. F. Cohn, T. Kanade, J. Lucey, S. Sclaroff, and Z. Ambadar, “The extended Cohn-Kanade dataset (CK+): A complete dataset for action unit and emotion-specified expression,” in Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2010, pp. 94–101.

[29] A. Dhall, R. Goecke, S. Lucey, and T. Gedeon, “Static facial expression analysis in tough conditions: Data, evaluation protocol and benchmark,” in Proc. IEEE Int. Conf. Comput. Vis. Workshops (ICCVW), 2011, pp. 2106–2112.

[30] S. Rifai, Y. Bengio, A. Courville, P. Vincent, and M. Mirza, “Disentangling factors of variation for facial expression recognition,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2012, pp. 808–822.

[31] K. Wang, X. Peng, J. Yang, D. Meng, and Y. Qiao, “Region attention networks for pose and occlusion robust facial expression recognition,” IEEE Trans. Image Process., vol. 29, pp. 4057–4069, 2020.

[32] Z.-M. Chen, X.-S. Wei, P. Wang, and Y. Guo, “Multi-label image recognition with graph convolutional networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 5172–5181.

[33] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. Int. Conf. Mach. Learn. (ICML), 2017, pp. 1321–1330.

[34] J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Lariviere, A. Beygelzimer,\` F. d’Alche-Buc, E. Fox, and H. Larochelle, “Improving reproducibility´ in machine learning research (a report from the NeurIPS 2019 reproducibility program),” J. Mach. Learn. Res., vol. 22, no. 164, pp. 1–20, 2021.

[35] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.