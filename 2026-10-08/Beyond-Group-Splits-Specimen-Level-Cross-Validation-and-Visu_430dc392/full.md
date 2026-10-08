# Beyond Group Splits: Specimen-Level Cross-Validation and Visual Attribution for Remaining-Shelf-Life Regression in Climacteric Fruit

Rovhona Mudau

Department of Computer Science

University of the Western Cape

Jean Frederic Isingizwe Nturambirwe

Cape Town, South Africa

4225313@myuwc.ac.za

eResearch Office

University of the Western Cape

Cape Town, South Africa

fisingizwe@uwc.ac.za

Clement Nthambazale Nyirenda

Department of Computer Science /

eResearch Office

University of the Western Cape

Cape Town, South Africa

cnyirenda@uwc.ac.za

Abstract—Estimating remaining shelf life (RSL) from images could provide affordable decision support for perishable produce, but evaluation protocols can substantially affect reported performance when repeated images are available from the same biological specimen. We use the Hass Avocado Ripening dataset, comprising 8,834 image–RSL pairs from 426 fruits across three storage regimes, to evaluate a frozen ImageNet-pretrained visual backbone with a lightweight regression head. Our contributions are threefold: we quantify the effect of observation-level versus specimen-disjoint evaluation, compare lightweight and heavier visual backbones under specimen-disjoint cross-validation, and examine their spatial attributions using Grad-CAM. Across ten observation-level random splits, the model achieves a mean RMSE of 2.37 days with a standard deviation of 0.03 days, whereas specimen-disjoint 5-fold cross-validation yields a mean RMSE of 3.12 days with a standard deviation of 0.11 days. The corresponding mean coefficient of determination is 0.553. A matched per-specimen comparison confirms higher error under specimen-disjoint evaluation, with a probability value below 0.001 across 426 specimens, showing that observation-level partitioning gives a substantially more optimistic estimate for this dataset and model configuration. Under specimen-disjoint evaluation, MobileNetV3-Small (0.93 million parameters) achieves accuracy comparable to ResNet-18 while providing substantially higher throughput, and Grad-CAM reveals differences in spatial attribution between the lightweight backbones. These results support specimen-disjoint evaluation and attribution analysis when assessing lightweight vision models for longitudinal shelflife prediction.

Index Terms—remaining shelf life, specimen-disjoint crossvalidation, transfer learning, lightweight CNN, Grad-CAM, explainable AI.

## I. INTRODUCTION

Post-harvest loss is a significant issue in the food system, particularly for fresh fruits and vegetables that deteriorate quickly in poor storage conditions [1]. A camera is the most cost-effective tool for predicting remaining shelf life (RSL), allowing operators to reroute, prioritise, or discount produce before deterioration occurs. The value of such a system, however, depends on a prior question: can its reported accuracy be trusted, and can a human verify that it is correct for the right reasons? The question is sharpened by the way these datasets are constructed. Monitoring deterioration requires photographing the same specimen repeatedly over time, so successive images of a single fruit are often highly similar. A standard random, image-level split can then place near-duplicate images of a test specimen into the training set, and the reported score reflects interpolation within a known distribution rather than generalisation to an unseen fruit. A recent peer-reviewed study trained convolutional networks on the same ‘Hass’ avocado imagery, showing that the visual signal is learnable [2]; however, we take a different perspective by asking when such a predictor’s performance is credible and when its logic is verifiable. The work is organised around three questions: how much reported accuracy survives when entire specimens are held out; whether a heavier backbone offers any advantage over a lightweight one under frozenfeature transfer; and whether the most accurate models attend to biologically meaningful fruit regions.

The rest of the paper is organized as follows: Section II reviews related work, Section III describes the materials and methods, Section IV presents the results, Section V discusses the findings, and Section VI concludes the paper.

## II. RELATED WORK

Visual produce assessment is now dominated by learned convolutional features [3], [4]. Ripeness classification and shelf-life regression are separated by an important distinction:

the former assigns a fruit to a discrete class [5], [6], whereas the latter must resolve fine-grained, time-varying cues into a calibrated estimate of remaining time. The most direct precedent is Xavier et al. [2], who trained CNNs on the same longitudinal ‘Hass’ avocado imagery and used sampledisjoint training, validation, and test subsets for ripening-stage prediction and subsequent shelf-life estimation. Other studies use environmental, physicochemical, or contextual variables for shelf-life prediction [7]–[10]. Mphahlele et al. [10], for example, combined IoT-monitored environmental conditions with laboratory quality attributes to evaluate machine-learning models for fresh-apple shelf-life prediction. This paper investigates not only whether the predictor attends to the fruit, but also how strongly its reported RSL accuracy depends on the unit used to partition longitudinal observations.

A major cause of poor reproducibility in applied machine learning is data leakage [11]. In agricultural vision, models can achieve high in-distribution accuracy while relying on background or acquisition artefacts rather than biological signal [4], [12]. Longitudinal shelf-life data introduce an additional repeated-measures problem because several images may originate from the same biological specimen. There are significant differences in the evaluation methods employed in related fruit studies. Hyperspectral sub-pictures from the same avocado photos were randomly assigned to training, validation, and test sets by Davur et al. [13]. Without disclosing specimen-grouped partitioning for the regression analysis, Lee et al. [14] randomly divided 1,400 smartphone avocado images into 80/10/10 training, validation, and test partitions. In contrast, Xavier et al. [2] explicitly prevented observations from the same avocado sample from showing up in distinct subgroups for training, validation, and testing. Ahmad et al. [15] supplemented internal cross-validation for mango shelf-life-stage classification with evaluation on an independent image dataset, while Brockelt et al. [16] trained a strawberry storage-age model on control-storage samples and evaluated it on samples stored under modified-atmosphere packaging. Table I summarises these differences.

The aforementioned protocols should not be used interchangeably because they address separate generalization issues. Related observations of the same biological specimen can occur in both training and test sets by means of random image-level partitioning. Instead, specimen-disjoint evaluation evaluates predictions on specimens that have not yet been seen under the specified acquisition and storage conditions. While independent external validation tests transfer to independently obtained data and hence address a wider distribution shift, condition-level holdout assesses transfer across storage regimes. Therefore, the controlled quantification of the performance optimism associated with observation-level specimen leakage and its comparison with specimen-disjoint and condition-level evaluation under an otherwise fixed prediction pipeline is the contribution of the current study rather than the introduction of specimen-disjoint partitioning itself, for which prior precedent exists.

There is also a persistent tension between predictive capacity and deployment cost. Strong accuracy is achieved by residual and dense backbones [17], [18], but heavy networks such as VGG are unsuitable for edge hardware [19]. The MobileNet family is widely used on-device [20], [21], because it trades a small accuracy margin for substantial efficiency gains through depthwise-separable convolutions [22], [23]; MobileNetV3 adds squeeze-and-excitation channel attention [24]. Importantly, the fine-tuning regime [25] provides much of the evidence that “better ImageNet models transfer better.” Under frozen transfer, representations are reused without adaptation, and a compact extractor may match a heavier one [26]. Lastly, explainability is becoming an important validation requirement [27], and Grad-CAM is widely used for spatial attribution [28], [29]. However, saliency maps can be unfaithful [30], and squeeze-and-excitation blocks reweight channels rather than spatial locations [31], motivating quantified attribution and perturbation-based corroboration such as Score-CAM [32].

## III. MATERIALS AND METHODS

## A. Dataset and RSL Target Construction

We use the peer-reviewed ‘Hass’ Avocado Ripening dataset [2], which tracks individual avocados under three storage regimes (cold storage at $1 0 ^ { \circ } \mathrm { C }$ , denoted T10; T20; and ambient) with daily photographs and a per-fruit ripening index. We construct a regression target using 14,710 photographs of 478 fruits. The first day a fruit reaches Ripening Index 4, corresponding to full spoilage, marks the end of its shelf life, and the number of days from acquisition to that day is the RSL of a photograph. We retain only images taken on or before the fruit’s end-of-shelf-life day, and we exclude fruit that never reaches Index 4 because they are rightcensored. This yields 8,834 image–RSL pairs from 426 fruit. All models are compared to a reference RMSE of 4.68 days from a constant-mean predictor.

## B. Model and Training

Each model uses a frozen ImageNet-pretrained convolutional backbone as a fixed feature extractor, followed by a lightweight multilayer-perceptron (MLP) head trained to regress RSL. We obtain features from the penultimate layer by replacing each backbone’s classifier with an identity mapping, yielding 1280-, 576-, and 512-dimensional descriptors for MobileNetV2, MobileNetV3-Small, and ResNet-18, respectively. All backbones use IMAGENET1K\_V1 weights and are frozen, so only the head is trained. The head is a two-layer MLP $( d  1 2 8  1 )$ with ReLU and dropout $( p \ : = \ : 0 . 3 )$ optimised with Adam (learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 } )$ under mean-squared error for 150 epochs at batch size 128.

The pipeline is kept leak-free by two measures. First, only training-fold statistics are used to standardise features. Second, the regression target is standardised within each training fold to stabilise optimisation. Predictions are then inverse-transformed to days before scoring, ensuring that all reported errors are in the original unit. Using a unique random seed for each fold, the head is retrained separately for each fold.

TABLE I  
EVALUATION PROTOCOLS IN REPRESENTATIVE FRUIT RIPENING AND SHELF-LIFE STUDIES.
<table><tr><td>Reference</td><td>Task</td><td>Protocol category</td><td>Evaluation design</td></tr><tr><td>[13]</td><td>Hass avocado ripeness regression</td><td>Random sub-image split</td><td>Each hyperspectral fruit image&#x27;s sub-images were assigned to training, validation, and test sets; as a result, source-image identity was not disjoint across partitions.</td></tr><tr><td>[2]</td><td>Hass avocado ripening Specimen-level split classification and shelf-life estimation</td><td></td><td>To prevent observations from the same sample from crossing partitions, avocado samples were uniquely assigned to 70/15/15 training, validation, and test subsets.</td></tr><tr><td>[14]</td><td>Hass avocado firmness pre- diction with RSL recom- mendation</td><td>Random image split</td><td>A total of 1,400 images were randomly shuffled into 80/10/10 training, validation, and test subsets; specimen grouping was not reported for the regression split.</td></tr><tr><td>[16]</td><td>Strawberry prediction</td><td>storage-age Condition-level split</td><td>Control-storage samples were used for model development and samples stored under modified-atmosphere packaging were used as the test condition.</td></tr><tr><td>[15]</td><td>Mango classification</td><td>shelf-life-stage Independent external vali- dation</td><td>Internal 10-fold cross-validation was supplemented by evaluation on a separate online mango image dataset.</td></tr><tr><td>This study</td><td>Hass avocado RSL regres- sion</td><td>Random, condition-level, and evaluation</td><td>Observation-level random splitting is used as a leakage diagnostic; storage specimen-disjoint regimes are held out for cross-condition analysis; primary within-dataset generalisation is assessed using specimen-disjoint 5-fold cross-validation.</td></tr></table>

## C. Evaluation Protocols

We compare three protocols that differ only in how images are partitioned:

1) Random observation-level split: without grouping by avocado identification, an 80/20 partition is performed at the image level. The procedure uses ten random seeds, and the regression head is retrained independently for each split. Only the corresponding training partition is used to estimate the feature and target standardization parameters. The mean and standard deviation for each of the ten test partitions are used to report performance. This approach is regarded as a leakage-prone diagnostic rather than an evaluation of performance on previously unseen specimens because images from the same avocado may appear in both training and test sets.

2) Leave-one-storage-regime-out: one storage regime (T10, T20, or ambient) is held out entirely, testing cross-condition generalisation.

3) Specimen-disjoint 5-fold cross-validation: the fruit identifiers are partitioned into five disjoint folds, each of which is held out in turn. All observations associated with a given avocado remain within the same fold, preventing specimen overlap between training and test partitions and evaluating generalisation to previously unseen specimens within the represented dataset and acquisition conditions.

For evaluation, only held-out observations are utilized. The regression head is retrained for every 80/20 split, and the mean and standard deviation of the ten held-out test sets are used to report performance. The random observation-level approach is repeated over ten independent seeds. Predictions from the three held-out storage conditions are combined over the corresponding test partitions for the leave-one-storageregime-out evaluation. All observations from a specific avocado remain within a single fold for specimen-disjoint 5- fold cross-validation. Out-of-fold predictions for all 8,834 observations are obtained by combining predictions from the five held-out specimen folds.

The specimen-disjoint out-of-fold prediction for all images in a random test partition is selected for the paired comparison between random observation-level and specimendisjoint assessment so that both protocols are assessed on identical image rows. One paired error summary per avocado is produced for statistical testing by averaging the absolute errors inside each biological specimen and then over the ten random repetitions.

## D. Attribution and the Mass-on-Fruit Measure

We calculate Grad-CAM [28] saliency for the two MobileNet backbones and summarise each heatmap by its masson-fruit: the percentage of total saliency energy that falls inside the avocado silhouette, used to assess whether predictions rely on the fruit rather than the background. The silhouette is obtained by thresholding brightness and saturation against the light background. Because the fruit occupies a nearly constant area, about 17% of the frame, a value near 0.17 suggests chance-level attention. Values substantially above 0.17 indicate spatial concentration on the fruit. For each of the four ripening stages, we compute mass-on-fruit for 40 images.

## E. Evaluation Metrics

We report image-level RMSE, MAE, and the coefficient of determination $R ^ { \bar { 2 } }$ :

$$
\begin{array} { r l } & { \mathrm { R M S E } _ { \mathrm { i m g } } = \displaystyle \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \hat { y } _ { i } - y _ { i } ) ^ { 2 } } , } \\ & { \mathrm { M A E } _ { \mathrm { i m g } } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \hat { y } _ { i } - y _ { i } \right| . } \end{array}
$$

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { N } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { \sum _ { i = 1 } ^ { N } ( y _ { i } - \bar { y } ) ^ { 2 } } ,\tag{1}
$$

(2)

where $\hat { y } _ { i }$ is the predicted RSL, $y _ { i }$ is the true RSL, $\bar { y }$ is the mean RSL, and N is the number of held-out image observations.

Because individual avocados contribute different numbers of longitudinal images, pooled image-level metrics give greater weight to specimens with longer observation sequences. We therefore also report fruit-balanced RMSE and MAE. Let S denote the number of held-out biological specimens and let specimen s contribute $n _ { s }$ observations with residuals $e _ { s i } = \hat { y } _ { s i } - y _ { s i }$ . The fruit-balanced metrics are

$$
\mathrm { R M S E } _ { \mathrm { f r u i t } } = \sqrt { \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \left( \frac { 1 } { n _ { s } } \sum _ { i = 1 } ^ { n _ { s } } e _ { s i } ^ { 2 } \right) } ,\tag{3}
$$

$$
\mathrm { M A E } _ { \mathrm { f r u i t } } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \left( \frac { 1 } { n _ { s } } \sum _ { i = 1 } ^ { n _ { s } } | e _ { s i } | \right) .\tag{4}
$$

Thus, all held-out observations are retained, but irrespective of how long an avocado’s image sequence is, each avocado adds the same overall weight. Metrics are calculated independently for each held-out test set and provided as mean ± standard deviation across seeds for the ten random observation-level divisions.

As shown in Section IV, $R ^ { 2 }$ becomes unstable when a held-out fold has little target variance. We therefore use RMSE and MAE, particularly the fruit-balanced variants, as the more interpretable measures for comparisons involving heterogeneous storage conditions.

## IV. RESULTS

## A. Evaluation Protocol

The reported performance was strongly affected by the evaluation protocol (Table II). Across ten independent observation-level random splits, the model achieved RMSE = $2 . 3 7 \pm 0 . 0 3 \mathrm { d } , \mathrm { M A E } = 1 . 7 3 \pm 0 . 0 3 \mathrm { d } ,$ and $R ^ { 2 } = 0 . 7 4 4 { \pm } 0 . 0 0 6$ Under specimen-disjoint 5-fold cross-validation, performance decreased to $\mathrm { R M S E } = 3 . 1 2 \pm 0 . 1 1 \mathrm { { c } }$ , $\mathrm { M A E } = 2 . 3 5 \pm 0 . 0 8 \mathrm { d } ,$ and $R ^ { 2 } ~ = ~ 0 . 5 5 3 \pm 0 . 0 2 3$ . For this dataset and model configuration, observation-level random partitioning produced a substantially more optimistic performance estimate, as demonstrated by the resulting difference in mean RMSE of approximately 0.76 d.

Fruits with longer observation sequences may be overweighted by pooled image-level metrics because the number of longitudinal observations varies among specimens. We therefore also report fruit-balanced RMSE and MAE, in which errors are first aggregated within each avocado and then averaged across the specimens represented in the corresponding held-out evaluation set, so that each represented fruit contributes equal total weight. Fruit-balanced RMSE/MAE under ten random observation-level splits were $2 . 4 2 \pm$ $0 . 0 4 / 1 . 7 6 \pm 0 . 0 2 \mathrm { d } ;$ specimen-disjoint 5-fold cross-validation produced 3.07/2.30 d; and leave-one-storage-regime-out evaluation produced 4.68/3.60 d. Therefore, the benefit of random observation-level partitioning cannot be explained by the fact that some specimens contribute more images than others; the protocol effect is still present even when the total weight of each fruit is identical.

Random-split and specimen-disjoint predictions were also assessed on matched held-out image observations to improve the statistical strength of the comparison. The corresponding specimen-disjoint out-of-fold prediction was selected for the same test rows for each random split. One paired error summary was produced for each biological specimen by averaging absolute errors within each avocado and across random-seed repetitions. Mean matched per-specimen MAE was 1.77 d under random observation-level splitting and 2.32 d under specimen-disjoint evaluation, corresponding to a mean paired difference of 0.55 d. The difference was significant under a Wilcoxon signed-rank test $( W ~ = ~ 6 8 2 8 , ~ p ~ < ~ 0 . 0 0 1$ n = 426 specimens). The specimen-disjoint estimate was also stable across folds, with RMSE ranging from 2.96 to 3.27 d (standard deviation 0.11 d). These results demonstrate that permitting repeated observations of the same fruit across training and test partitions can result in optimistic withindataset performance estimates and that the greater specimendisjoint error was not caused by a single held-out fold.

The leave-one-storage-regime-out result must be interpreted with care, because strongly negative $R ^ { 2 }$ values can be misleading when the held-out target variance is small. The three regimes yielded RMSE/MAE values of 5.92/4.87 d (T10), 3.82/2.85 d (T20), and 3.71/2.85 d (ambient). The fast-ripening T20 and ambient regimes had mean RSL values of approximately 3 d and relatively limited target variance; therefore, moderate absolute errors produced strongly negative $R ^ { 2 }$ values (−2.34 and −2.28, respectively). In contrast, T10 had a mean RSL of 7.75 d and produced the largest absolute errors when held out. Because $R ^ { 2 }$ depends on the variance of the target within each held-out condition, MAE is treated as the more interpretable statistic for cross-condition comparison.

## B. Backbone Comparison

Under identical specimen-disjoint 5-fold evaluation, backbone capacity does not translate into better accuracy (Table III). The smallest and fastest extractor, MobileNetV3- Small, with only 0.93 M parameters and a throughput of 76 images/s, is at least as accurate as the larger alternatives $( R ^ { 2 } ~ = ~ 0 . 5 9 6 \pm 0 . 0 2 7 )$ ; the differences among the three backbones are small relative to the observed fold-to-fold variability. The decisive point is therefore not that the lightest model wins but that the heavier ResNet-18 does not provide a corresponding improvement in predictive performance under the evaluated frozen-feature setting. This is consistent with the frozen-feature, modest-data regime: without fine-tuning, the larger network cannot exploit its extra capacity, whereas the efficient MobileNet representations are well matched to the fine-grained skin-colour signals that indicate ripening. Under the resource-constrained transfer setting targeted here, the lightweight architecture is therefore the rational default on both accuracy and compute, confirming that the fine-tuning evidence for heavier backbones [25] need not apply to frozen transfer [26].

TABLE II  
IMAGE-LEVEL AND FRUIT-BALANCED PERFORMANCE UNDER THE THREE EVALUATION PROTOCOLS.
<table><tr><td>Protocol</td><td> $\mathbf { R M S E } _ { \mathrm { i m g } } \ ( \mathbf { d } )$ </td><td> $\mathbf { M A E } _ { \mathrm { i m g } } \left( \mathbf { d } \right)$ </td><td> $R _ { \mathrm { i m g } } ^ { 2 }$ </td><td> ${ \mathbf { R M S E } } _ { \mathrm { f r u i t } }$  (d)</td><td> $\mathbf { M A E } _ { \mathrm { f r u i t } } \ ( \mathbf { d } )$ </td></tr><tr><td>Random observation-level split</td><td> $2 . 3 7 \pm 0 . 0 3$ </td><td> $1 . 7 3 \pm 0 . 0 3$ </td><td> $0 . 7 4 4 \pm 0 . 0 0 6$ </td><td> $2 . 4 2 \pm 0 . 0 4$ </td><td> $1 . 7 6 \pm 0 . 0 2$ </td></tr><tr><td>Leave-one-storage-regime-out</td><td> $5 . 1 8$ </td><td>4.07</td><td> $^ { - 0 . 2 2 7 }$ </td><td>4.68</td><td>3.60</td></tr><tr><td>Specimen-disjoint 5-fold CV</td><td> $3 . 1 2 \pm 0 . 1 1$ </td><td> $2 . 3 5 \pm 0 . 0 8$ </td><td> $0 . 5 5 3 \pm 0 . 0 2 3$ </td><td>3.07</td><td>2.30</td></tr></table>

TABLE III  
BACKBONE COMPARISON UNDER SPECIMEN-DISJOINT5-FOLD CROSS-VALIDATION
<table><tr><td>Backbone</td><td>Par. (M)</td><td> $\mathrm { T h r }$ </td><td>RMSE (d)</td><td>MAE (d)</td><td> $R ^ { 2 }$ </td></tr><tr><td>MobileNetV3-S</td><td>0.93</td><td>76/s</td><td> $\mathbf { 2 . 9 7 \pm 0 . 1 4 }$ </td><td> $\mathbf { 2 . 2 4 \pm 0 . 1 2 }$ </td><td> $\mathbf { 0 . 5 9 6 { \scriptstyle \pm 0 . 0 2 7 } }$ </td></tr><tr><td>MobileNetV2</td><td>2.22</td><td></td><td> $3 . 1 2 \pm 0 . 1 1$ </td><td> $2 . 3 5 \pm 0 . 0 8$ </td><td> $0 . 5 5 3 { \pm } 0 . 0 2 3$ </td></tr><tr><td>ResNet-18</td><td>11.18</td><td>11/s</td><td> $3 . 0 9 \pm 0 . 0 8$ </td><td> $2 . 3 5 \pm 0 . 0 7$ </td><td> $0 . 5 6 2 { \pm } 0 . 0 1 8$ </td></tr></table>

## C. Visual Attribution

Accurate prediction does not ensure that a model attends to the fruit. Table IV shows mass-on-fruit by ripening stage, where the chance baseline is 0.17. When ripening cues are clear, MobileNetV2 focuses heavily on the fruit—mass-onfruit reaches 0.44 at stage 1, 2.6× chance (Fig. 1)—and decreases monotonically to 0.12 at full spoilage. This decrease makes physiological sense: saliency disperses because a spoiled, uniformly dark avocado lacks distinguishing surface characteristics. Despite having the most accurate predictions of the three backbones, MobileNetV3-Small behaves very differently. Its mass-on-fruit remains close to chance at every stage (0.12–0.28), with substantial variance and no discernible trend, localising to the background.

This reveals a divergence between the two lightweight models’ accuracy and interpretability: MobileNetV3-Small, which is more accurate, is less spatially verifiable. We interpret this cautiously. The disparity is likely caused by MobileNetV3’s squeeze-and-excitation blocks, which reweight feature channels globally rather than spatially [31]. As a result, predictive evidence may be dispersed across channels in a way that gradient-based spatial attribution is unable to localise. Because saliency maps are not guaranteed to be faithful [30], the diffuse maps may partially reflect a limitation of Grad-CAM as computed on squeeze-and-excitation architectures rather than a property of the model itself. Corroboration with perturbation-based attribution, such as Score-CAM [32], is left to future work. Despite its slightly lower accuracy, MobileNetV2’s more localised Grad-CAM attribution may be advantageous for inspection in decision-support settings where forecasts must be checked.

## V. DISCUSSION

Three implications follow. First, the evaluation protocol has a larger effect on reported performance than backbone choice in this repeated-measures dataset. Across ten observationlevel random splits, RMSE was $2 . 3 7 \pm 0 . 0 3 { \mathrm { d } } .$ , compared with $3 . 1 2 \pm 0 . 1 1 { \mathrm { d } }$ under specimen-disjoint 5-fold crossvalidation, a difference of approximately 0.76 d. This difference is larger than that observed between any two backbones in Table III. The matched analysis supports the same conclusion: mean per-specimen MAE increased from 1.77 d under random observation-level splitting to 2.32 d under specimendisjoint evaluation, with a mean paired difference of 0.55 d $( p ~ < ~ 0 . 0 0 1 , ~ n ~ = ~ 4 2 6$ specimens). Therefore, neither a single favorable random partition nor a single highly difficult specimen fold can be attributed to the effect. Rather, allowing observations from the same avocado to occur in both training and test segments yields an optimistic estimate of performance for this dataset and model configuration. The observed difference should not be interpreted as a universal correction applicable to other shelf-life datasets.

TABLE IV  
GRAD-CAM MASS-ON-FRUIT BY STAGE
<table><tr><td>Stage</td><td>MobileNetV2</td><td>MobileNetV3-Small</td></tr><tr><td>1 (fresh)</td><td>0.44</td><td>0.12</td></tr><tr><td>2</td><td>0.29</td><td>0.23</td></tr><tr><td>3</td><td>0.16</td><td>0.28</td></tr><tr><td>4 (spoiled)</td><td>0.12</td><td>0.14</td></tr></table>

![](images/6f44b3082da100a729f37d3dec6aa41290f58933b254cbda2d4c9bc3f207f0ae.jpg)  
Fig. 1. Grad-CAM saliency at the ripening endpoints. MobileNetV2 (top) concentrates on the fruit when fresh and disperses once spoiled; the more accurate MobileNetV3-Small (bottom) localises to the background at both stages.

The dependence among repeated images also affects metric aggregation. Specimens that provide more observations are given higher weight in image-level evaluation. The same qualitative pattern persists when each avocado is instead given an identical total weight: the corresponding MAEs are 1.76 and 2.30 d, and the fruit-balanced RMSE is 2.42 d for random observation-level assessment and 3.07 d for specimen-disjoint evaluation. Therefore, varying numbers of images per specimen are not the only explanation for the protocol effect. Because the held-out regimes have significantly different target distributions, interpretation of $R ^ { 2 }$ for leave-one-storageregime-out evaluation requires extra caution. Specifically, minor absolute errors result in highly negative $R ^ { 2 }$ values due to the limited RSL variance in the T20 and ambient folds. For comparisons across heterogeneous storage regimes, MAE and RMSE are consequently easier to understand than $R ^ { 2 }$

Second, while the feature extractor is frozen, the backbone comparison indicates limited benefit from greater model capacity. Despite having far fewer parameters and higher observed throughput, MobileNetV3-Small achieves an RMSE of $2 . 9 7 \pm 0 . 1 4 \mathrm { d }$ during specimen-disjoint evaluation, compared with $3 . 0 9 \pm 0 . 0 8 \ :$ d for ResNet-18. Therefore, increasing backbone size does not result in a proportional gain in RSL prediction accuracy within the assessed transferlearning setting. This result is relevant to resource-constrained deployment, where predictive performance may be impacted by inference cost, memory footprint, and throughput. It does not mean that a different adaptation method, such as taskspecific fine-tuning, may not be advantageous for a larger backbone.

Third, predictive accuracy and spatial attribution quality do not necessarily coincide. Among the assessed backbones, MobileNetV3-Small has the best prediction performance; however, its Grad-CAM maps are less reliably localized to the avocado than those generated by MobileNetV2. It is challenging to ascertain whether this discrepancy stems from the underlying model representation, the attribution method, or both, due to architectural mechanisms such as squeeze-andexcitation channel reweighting [31] and the known limitations of gradient-based saliency methods [30]. Therefore, rather than serving as evidence of the network’s causal properties, the Grad-CAM results should be viewed as an attribution audit. Nevertheless, spatially plausible attributions may help identify obvious failure modes in decision-support settings. These findings motivate considering predictive performance, computational efficiency, and attribution behaviour jointly when selecting lightweight models.

## VI. CONCLUSION

This study makes three main contributions to image-based longitudinal remaining-shelf-life prediction. Firstly, it quantifies the effect of an evaluation protocol within a defined modeling pipeline. RMSE was $2 . 3 7 { \pm } 0 . 0 3 { \mathrm { d } }$ over ten observationlevel random splits, whereas specimen-disjoint 5-fold crossvalidation yielded $3 . 1 2 \pm 0 . 1 1 \mathrm { d }$ . The same direction of effect was confirmed by a corresponding per-specimen study, which demonstrated that for this dataset and model configuration, observation-level random partitioning yields a substantially more optimistic performance estimate. Furthermore, the conclusion holds valid under fruit-balanced aggregate, suggesting that varying numbers of observations per specimen are not the sole explanation.

Second, the backbone comparison shows that greater model capacity does not necessarily improve predictive performance under frozen-feature transfer. With 0.93 M parameters, MobileNetV3-Small produced substantially greater throughput while achieving accuracy comparable to ResNet-18. When computational efficiency is crucial and more backbone capacity does not result in improved RSL prediction, this encourages the use of lightweight visual extractors.

Thirdly, the Grad-CAM analysis uncovers variations in spatial attribution behavior that predictive metrics alone fail to capture. MobileNetV3-Small demonstrated the highest predictive efficacy across the assessed architectures, but MobileNetV2 yielded more consistently localized Grad-CAM visualizations. These results are interpreted as an attribution audit rather than evidence that one backbone is inherently more faithful than another because saliency methods are unable to demonstrate causal feature usage on their own.

Several limitations bound these findings. The backbones were used as frozen feature extractors, so full or partial fine-tuning could alter their relative performance. The masson-fruit measure depends on an estimated fruit silhouette obtained using a heuristic threshold, and RSL is defined relative to a single end-of-shelf-life stage within one produce type and acquisition setting. The present results therefore characterise within-dataset evaluation behaviour rather than external generalisation. Future work should incorporate environmental sensor streams associated with post-harvest deterioration, evaluate the approach under independent produce and imaging conditions, and corroborate the attribution findings using complementary perturbation-based methods such as Score-CAM [32].

## REFERENCES

[1] C. Cederberg, U. Sonesson, and J. Gustavsson, “Global food losses and food waste: Extent, causes and prevention,” Food and Agriculture Organization of the United Nations, Rome, Tech. Rep., 2011.

[2] P. Xavier, P. M. Rodrigues, and C. L. M. Silva, “Shelf-life management and ripening assessment of ‘Hass’ avocado using deep learning approaches,” Foods, vol. 13, no. 8, p. 1150, 2024.

[3] L. Chuquimarca Jimenez, B. Vintimilla, and S. Velastin, “A review´ of external quality inspection for fruit grading using CNN models,” Artificial Intelligence in Agriculture, vol. 14, 2024.

[4] S. P. Mohanty, D. P. Hughes, and M. Salathe, “Using deep learning´ for image-based plant disease detection,” Frontiers in Plant Science, vol. 7, p. 1419, 2016.

[5] N. Begum and M. Kumar Hazarika, “Spoilage detection of tomatoes using a convolutional neural network,” Research in Agricultural Engineering, vol. 71, no. 2, pp. 80–87, 2025.

[6] S. Palakodati, V. Ch, Y. Dasari, and S. Bulla, “Fresh and rotten fruits classification using CNN and transfer learning,” Revue d’Intelligence Artificielle, vol. 34, no. 5, pp. 617–622, 2020.

[7] D. K. V. Baskar, R. N. Sri, S. Lavanya, and V. Harisha, “Deep CNN model for predicting shelf life of fresh fruits and vegetables using temperature simulation data for optimized transport and storage,” Journal for Educators, Teachers and Trainers, vol. 15, no. 5, pp. 425– 434, 2024.

[8] K. Agarwal, “Transforming agricultural management: A multimodal deep learning approach to shelf life estimation,” in QTanalytics Publication, 2025, pp. 102–128.

[9] L. Liang, Z. Wang, K. Liu, J. Xu, C. Li, H. Liu, and M. Diao, “Research on tomato quality prediction models based on the coupling of environmental factors and appearance phenotypes,” Plants, vol. 14, no. 23, 2025.

[10] T. Mphahlele, J. F. I. Nturambirwe, and A. Klein, “Leveraging datadriven machine learning for intelligent shelf life prediction of fresh apple,” in IST-Africa 2026 Conference Proceedings, M. Cunningham and P. Cunningham, Eds. IST-Africa Institute and IIMC International Information Management Corporation Ltd, 2026.

[11] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, vol. 4, no. 9, p. 100804, 2023.

[12] M. A. Noyan, “Uncovering bias in the PlantVillage dataset,” arXiv preprint arXiv:2206.04374, 2022.

[13] Y. J. Davur, W. Kamper, K. Khoshelham, S. J. Trueman, and S. H. Bai,¨ “Estimating the ripeness of hass avocado fruit using deep learning with hyperspectral imaging,” Horticulturae, vol. 9, no. 5, p. 599, 2023.

[14] I.-H. Lee, Z. Li, and L. Ma, “Explainable ai and mobile imaging for non-destructive avocado ripeness and internal quality assessment to reduce food waste,” Current Research in Food Science, vol. 11, p. 101196, 2025.

[15] I. Ahmad, B. Siddique, M. Junaid, M. Gouda, A. Khaliq, Z. U. Haq, and Z. Qiu, “SE-DBIRNet: Squeeze-and-excitation driven dual-path residual network for mango shelf-life stages classification,” Foods, vol. 15, no. 13, p. 2279, 2026.

[16] J. Brockelt, R. Dammann, J. Griese, A. Weiss, M. Fischer, and M. Creydt, “Storage profiling: Evaluating the effect of modified atmosphere packaging on metabolomic changes of strawberries (Fragaria × ananassa),” Metabolites, vol. 15, no. 5, p. 330, 2025.

[17] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[18] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 4700–4708.

[19] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in International Conference on Learning Representations (ICLR), 2015.

[20] A. Boulahya, “Tomato fruit diseases classification using MobileNetV2,” Engineering and Technology Journal, vol. 10, no. 7, pp. 5607–5616, 2025.

[21] M. Karim, M. Goni, M. Nahiduzzaman, M. Ahsan, J. Haider, and M. Kowalski, “Enhancing agriculture through real-time grape leaf disease classification via an edge device with a lightweight CNN architecture and Grad-CAM,” Scientific Reports, vol. 14, 2024.

[22] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “MobileNets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.

[23] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520.

[24] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, “Searching for MobileNetV3,” arXiv preprint arXiv:1905.02244, 2019.

[25] S. Kornblith, J. Shlens, and Q. V. Le, “Do better ImageNet models transfer better?” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 2661– 2671.

[26] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable are features in deep neural networks?” in Advances in Neural Information Processing Systems (NeurIPS), vol. 27, 2014, pp. 3320–3328.

[27] A. Barredo Arrieta, N. D´ıaz-Rodr´ıguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garcia, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, and F. Herrera, “Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI,” Information Fusion, vol. 58, pp. 82–115, 2020.

[28] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626.

[29] C. Pal, S. Karmakar, I. Mukherjee, and P. P. Chakrabarti, “A lightweight and explainable CNN model for empowering plant disease diagnosis,” Scientific Reports, vol. 15, no. 1, p. 30720, 2025.

[30] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity checks for saliency maps,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 31, 2018.

[31] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132–7141.

[32] H. Wang, Z. Wang, M. Du, F. Yang, Z. Zhang, S. Ding, P. Mardziel, and X. Hu, “Score-CAM: Score-weighted visual explanations for convolutional neural networks,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020, pp. 24–25.