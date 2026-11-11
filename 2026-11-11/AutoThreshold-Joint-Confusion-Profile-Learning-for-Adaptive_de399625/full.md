# AutoThreshold: Joint Confusion-Profile Learning for Adaptive Mask Binarization

Shaokai Wu<sup>1,2</sup>, Xunchu Zhou<sup>1,2</sup>, Pinzhi Wang<sup>2</sup>, Bo Dong<sup>2</sup> Qi Kong<sup>2</sup>, Xiao Cui<sup>2</sup>, Yue Ding<sup>1</sup>, Hongtao Lu<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Zelos Technology

## Abstract

A single validation-tuned threshold can become brittle when dense-score distributions shift. We introduce AutoThreshold, a post-hoc policy that predicts the joint mass ofpositive and negative pixels over score bins. This target directly parameterizes threshold-indexed confusion counts, avoiding products of independently estimated prevalence and class-conditional rates. Softmax normalization and cumulative decoding yield coherent, monotone confusion profiles; a plug-in decoder then selects a threshold for any $F _ { \beta }$ objective without retraining the head. Across three independently trained Kvasir pipelines transferred to four polyp datasets, Joint exceeds continuous Otsu by 0.0080 ± 0.0046 external macro Dice, with positive gains in 3/3 runs. On the reference pipeline, three head seeds achieve 0.6324 ± 0.0027, 0.0085 above Otsu; direct-utility Ridge achieves 0.6342, with the paired difference interval spanning zero. The same Joint checkpoints outperform Otsufor $F _ { 0 . 5 } ,$ Dice, and $F _ { 2 } .$ . On developmental KITTI Road leaveone-corruption-family-out evaluation, Joint improves over scalar regression by 0.0082 [0.0064, 0.0101], while direct utility is 0.0022 higher. These results establish joint confusion prediction as a useful, metric-retargetable inductive bias, without claiming Bayes-optimal $F _ { \beta }$ decisions.

## 1. Introduction

Vision models produce continuous scores, but deployed systems ultimately act on discrete masks. The binarization threshold directly trades false positives against false negatives, yet it is commonly fixed once on source validation data. That choice ignores image- and domain-dependent changes in score calibration, score shape, and foreground prevalence—precisely the changes encountered after deployment.

Adaptive thresholding is well established in low-level image processing [23], and task-specific systems adapt training or post-processing thresholds in object detection and crowd localization [9, 19, 26, 30]. Learned mask binarization, however, presents a specific supervision choice. Regressing one tie-broken optimum discards the utility landscape: two equally distant errors can incur very different regret. Regressing every sampled utility is a principled finite-action value estimator, but it imposes no structure across neighboring thresholds and must be retrained when the target metric changes.

We seek a post-hoc target between these two extremes: richer than a scalar optimum, structured across thresholds, and independent of a particular $F _ { \beta }$ . Binary masks expose exactly such a target. The joint distribution of pixel label and score bin determines $T P ( t )$ $F P ( t )$ , and $F N ( t )$ at every threshold; these confusion profiles in turn support the full family of $F _ { \beta }$ plug-in decisions.

AutoThreshold predicts positive and negative score-bin mass as one normalized distribution. Cumulative decoding enforces monotone confusion-count curves, and a metric decoder selects a threshold for Dice, $F _ { 0 . 5 }$ , or $F _ { 2 }$ from the same prediction. An auxiliary loss aligns its score marginal with the observed, label-free histogram. The policy remains fully detached from the segmenter: labels construct offline targets from cached score maps, whereas deployment uses only score-map statistics and a lightweight MLP.

Our main contributions are:

• We formulate adaptive mask binarization as joint confusion-profile prediction. The target avoids products of separately estimated prevalence and conditional rates, while its cumulative decoder supports monotone, retraining-free $F _ { \beta }$ retargeting.

• We separate the benefit of the structured target from capacity and objective choices through scalar, hardgrid, soft-utility, expected-regret, direct value-regression, Ridge, GBRT, Otsu, and area-only controls.

• We test transfer across four polyp datasets and six heldout KITTI corruption families, with complete-pipeline repeats, two external aggregation schemes, paired uncertainty, and sensitivity and profile-error analyses.

![](images/1f46be0e16e25fd53b879bafd6cbd45182a79eabe30b09555a1223634fd247a0.jpg)  
Figure 1: Design space for threshold policies. Scored outputs admit different threshold types, spatial supports (image, patch, or dense location), and decision-unit utilities. We instantiate image-level mask binarization; other supports are valid only when both operator and utility decompose accordingly. Color denotes confidence, and size illustrates a second thresholdable attribute.

## <sub>b</sub>2. Related Work

Post-Processing Thresholds High-level vision systems use rator-Specific Thresholding Algorithmic methodsconfidence thresholds to filter predictions and IoU threshde a particular decision rule. Adaptive NMS changesolds in both post-processing and evaluation. These roles difpression according to local box density (Liu, Huang, andfer: an NMS overlap threshold controls suppression among g 2019); DIoU supplies a distance-aware NMS crite-predicted boxes [2], whereas evaluation IoU thresholds de-<sup>,</sup> <sup>while</sup> <sup>the</sup> <sup>CIoU</sup> <sup>study</sup> <sup>incorporates</sup> <sup>geometric</sup> <sup>factors</sup>termine whether detections match ground truth [17]. Fig-<sup>Cluster-NMS</sup> <sup>(Zheng</sup> <sup>et</sup> <sup>al.</sup> <sup>2020,</sup> <sup>2022).</sup> <sup>Threshold-</sup>ure 1 situates mask binarization in this broader design <sup>tive</sup> <sup>focal</sup> <sup>loss</sup> <sup>changes</sup> <sup>how</sup> <sup>target-domain</sup> <sup>samples</sup> <sup>con-</sup>space. Our joint decoder specifically exploits binary score ute during training (Yan et al. 2023), while classical im-<sub>maps; it does not extend directly to NMS or corpus-level</sub> <sup>method</sup>ranking.

Operator-Specific Thresholding Algorithmic methods encode a particular decision rule. Adaptive NMS changes suppression according to local box density [19]; DIoU supplies a distance-aware NMS criterion, while the CIoU study rned Threshold Policies Learned threshold methodsincorporates geometric factors into Cluster-NMS [31, 32]. cally attach adaptation to one task: CPSS-FAT adaptsThreshold-adaptive focal loss changes how target-domain tive-sample selection thresholds during detector train-samples contribute during training [29], while classical im-(Yang et al. 2023), Independent Instance Maps learnage methods derive binarization thresholds from intensity ge-dependent spatial binarization maps for crowd lo-statistics [23, 24]. These methods exploit operator-specific zation (Gao et al. 2020), and Crowd Hat and CTFNetstructure but do not learn a confusion representation that <sup>n</sup> <sup>region-adaptive</sup> <sup>NMS</sup> <sup>or</sup> <sup>confidence-</sup>can be decoded for multiple task metrics.

<sup>s</sup> <sup>(Wu</sup> <sup>and</sup> <sup>Yang</sup> <sup>2023;</sup> <sup>Wu</sup> <sup>et</sup> <sup>al.</sup> <sup>2022).</sup> <sup>AutoThreshold</sup>Learned Threshold Policies Learned threshold meth-<sup>ad</sup> <sup>learns</sup> <sup>a</sup> <sup>detached</sup> <sup>mask</sup> <sup>policy</sup> <sup>from</sup> <sup>cached</sup> <sup>outputs</sup>ods typically attach adaptation to one task: CPSS-FAT adapts positive-sample selection thresholds during detec-<sub>mask thresholds for a task metric. Decision-focused lear</sub>tor training [30], Independent Instance Maps learn imagetrains predictions for downstream optimization value (Ddependent spatial binarization maps for crowd localiza-Amos, and Kolter 2017; Wilder, Dilkina, and Tambe 2tion [9], and Crowd Hat and CTFNet learn region-adaptive We share that motivation but predict threshold-suficientNMS or confidence-foresight mechanisms [26, 27]. Aufusion statistics from cached outputs, requiring neither atoThreshold instead learns a detached mask policy from ferentiable optimizer nor changes to the frcached outputs of a specified frozen operator.

Contextual Decisions and Calibration Post-hoc calibration aligns confidence with empirical correctness [11], while selective prediction learns when to abstain at a desired coverage [10]. They address score reliability or risk– coverage behavior, not per-image mask thresholds for a task metric. Decision-focused learning trains predictions for downstream optimization value [4, 25]. We share that motivation but predict threshold-sufficient confusion statistics from cached outputs, requiring neither a differentiable optimizer nor changes to the frozen model.

<sup>1</sup> Given a finite threshold grid and measured rewards for every action, direct utility regression is a full-information cost-sensitive plug-in rule [6]; contextual-bandit methods address the harder partial-feedback setting [1, 5]. We include the resulting decision-consistent finite-action estimator as our strongest learned baseline. AutoThreshold instead predicts a metric-independent joint distribution over labels and score bins, sharing cumulative structure across neighboring thresholds.

Metric-Aware Binary Decisions Optimal $F _ { 1 }$ thresholding and plug-in rules for generalized classification metrics derive decisions from calibrated probabilities or estimated confusion quantities [15, 18]. We use the same dependence of $F _ { \beta }$ on confusion counts, but at a different decision unit: our head predicts an entire threshold-indexed confusion curve for each frozen score map. Because $F _ { \beta }$ is nonlinear, evaluating predicted mean confusion counts is not generally equivalent to maximizing conditional expected $F _ { \beta }$ We therefore study a structured plug-in rule, not a decisionconsistency claim.

## 3. Problem Formulation

Problem Definition Let x denote a decision unit, such as an image or a spatial region. A frozen model produces raw output $Y _ { x } = \mathcal { M } ( I ) _ { x }$ and a label-free descriptor $F _ { x }$ Given ground truth $g _ { x }$ , a thresholded operator $\mathcal { P } ( Y _ { x } , \tau , t )$ and bounded scalar utility $m ( \mathcal { P } ( Y _ { x } , \tau , t ) , g _ { x } )$ , we learn a policy $\pi ( F _ { x } )$ for threshold type τ . Its deployment objective is

$$
J ( \pi ) = \mathbb { E } _ { x } [ m ( \mathcal { P } ( Y _ { x } , \tau , \pi ( F _ { x } ) ) , g _ { x } ) ] .\tag{1}
$$

This expectation fixes the method’s scope: J must decompose over units receiving independent thresholds. Macro image Dice and mean regional utility satisfy this condition. A corpus-level ranking functional such as standard COCO AP does not, because changing one image can alter other images’ ranking contributions.

Data Protocol We split labeled data into disjoint $\mathcal { D } _ { \mathrm { t r a i n } } .$ $\mathcal { D } _ { \mathrm { v a l } }$ , and $\mathcal { D } _ { \mathrm { t e s t } }$ . The frozen model runs once on each split. Utility labels from $\mathcal { D } _ { \mathrm { t r a i n } }$ fit the policy; $\mathcal { D } _ { \mathrm { v a l } }$ labels serve only checkpoint and hyperparameter selection. Test labels score the selected policy but never fit or select it.

Overview For each training image, AutoThreshold records the joint mass of positive and negative pixels in every score bin. A lightweight, detached predictor N maps label-free score-map features $F _ { x }$ to this target. At deployment, cumulative decoding reconstructs a confusion profile over $A = \{ \alpha _ { 1 } , \ldots , \alpha _ { k } \}$ , and a chosen $F _ { \beta }$ decoder returns its predicted maximizer. This step requires neither labels nor online utility evaluation.

We detach the threshold policy from the vision model for three reasons:

• It leaves the original architecture unchanged and avoids redesigning an adaptive head for every task-model pair.

• The same cached score maps support direct decision baselines and multiple $F _ { \beta }$ decoders.

• Offline utility annotation avoids backpropagating through discrete threshold decisions.

The small policy thus amortizes a discrete decision evaluated offline.

## 4. Learning Adaptive Threshold Policies

## 4.1. Threshold Policy Design

Before learning a threshold policy, one must specify both its action and its evaluation unit. Figure 1 organizes these choices along three axes.

Threshold Type: Thresholds control distinct postprocessing decisions. For example, Crowd Hat adapts regional NMS thresholds [26], whereas IIM predicts a dense binarization-threshold map for crowd localization [9].

Decision Unit: A policy may share one image-wide threshold, predict one threshold per region, or produce a dense threshold field. “Pixel-level” in Figure 1 is the raster-output special case of the last setting. A globally coupled decision, such as suppression, cannot be localized unless its operator is regionalized as well.

Decision Utility: Different utilities induce different optimal actions. We therefore annotate the same quantity that Equation (1) averages at deployment. It is not interchangeable with a non-decomposable corpus metric, even when the two share a name or scale.

Our experiments fix these choices to one maskbinarization threshold per image, supervised and evaluated by image-level $F _ { \beta }$

## 4.2. Offline Utility Annotation

Optimizing Thresholds Because M is frozen, every threshold candidate reuses $Y _ { x }$ . We define the sampled utility profile and its deterministic scalar summary as

$$
\begin{array} { r l } & { u _ { j } ( x ) = m ( \mathcal { P } ( Y _ { x } , \tau , \alpha _ { j } ) , g _ { x } ) , } \\ & { t ^ { * } ( x ) = \alpha _ { r ( \arg \operatorname* { m a x } _ { j } u _ { j } ( x ) ) } , } \end{array}\tag{2}
$$

where r is a fixed tie rule. We retain these labels for utility, scalar, and grid-classification baselines. AutoThreshold instead uses the threshold-sufficient confusion statistics below. The same construction applies to a p×p regional policy only when both operator and utility decompose by region; coupled decisions require a joint decision unit.

Annotating Threshold Labels For each unit in $\mathcal { D } _ { \mathrm { t r a i n } }$ we evaluate a fixed uniform grid and cache $( F _ { x } , t ^ { * } ( x ) , \mathbf { u } ( x ) )$ for the baselines. For the joint head, we cache normalized positive and negative pixel mass over score bins. A separate validation cache supports model selection, but its tuples never update N. Reported test data enter neither training nor selection.

## 4.3. Joint Confusion-Profile Learning

Only cached training tuples update N; neither the segmenter nor hard binarization participates in backpropagation.

Threshold Head A descriptor may be pooled from internal features [16] or computed from the frozen score map. We use the latter: quantiles, histograms, moments, entropy, and

thresholded area ratios feed a small ReLU MLP [20]. For B score bins, its 2B logits parameterize one normalized joint distribution.

Joint Target and Decoder For N valid pixels with score $s _ { p }$ and binary label $y _ { p } .$ , define the positive and negative joint mass in score bin b as

$$
\begin{array} { l } { { \displaystyle a _ { b } = \frac { 1 } { N } \sum _ { p } \mathbf { 1 } [ y _ { p } = 1 , ~ s _ { p } \in b ] } , } \\ { { \displaystyle b _ { b } = \frac { 1 } { N } \sum _ { p } \mathbf { 1 } [ y _ { p } = 0 , ~ s _ { p } \in b ] } . } \end{array}\tag{3}
$$

The observed, label-free score marginal is $q _ { b } = a _ { b } + b _ { b }$ A single softmax maps the logits to $( \hat { \mathbf { a } } , \hat { \mathbf { b } } ) \bar { \in } \Delta ^ { 2 B - 1 }$ . Let $w _ { j b } \in [ 0 , 1 ]$ denote the fraction of bin b above threshold $\alpha _ { j }$ providing within-bin interpolation. We decode

$$
\begin{array} { l } { { \widehat { T P } _ { j } = \displaystyle \sum _ { b } w _ { j b } \hat { a } _ { b } , \quad \widehat { F P } _ { j } = \displaystyle \sum _ { b } w _ { j b } \hat { b } _ { b } , } } \\ { { \widehat { F N } _ { j } = \displaystyle \sum _ { b } \hat { a } _ { b } - \widehat { T P } _ { j } . } } \end{array}\tag{4}
$$

This target preserves prevalence–score dependence. In particular, it avoids substituting the generally unequal product $\mathbb { E } [ \rho \ | \ F ] \mathbb { E } [ R ^ { + } \ | \ F ]$ for $\mathbb { E } [ \rho R ^ { + } ~ | ~ \boldsymbol { F } ]$ . Nonnegativity and normalization keep the estimated confusion counts coherent, while cumulative decoding makes TP and FP nonincreasing in the threshold. We regularize the predicted marginal $\hat { q } _ { b } = \hat { a } _ { b } + \hat { b } _ { b }$ toward the current score histogram $q _ { b } .$ , but do not impose equality.

Joint Objective and Metric Retargeting We jointly supervise the cumulative counts, joint density, and observed marginal:

$$
\begin{array} { c } { { \displaystyle \mathcal { L } _ { \mathrm { j o i n t } } = \frac { 1 } { k } \sum _ { j } \left[ ( \widehat { T P } _ { j } - T P _ { j } ) ^ { 2 } + ( \widehat { F P } _ { j } - F P _ { j } ) ^ { 2 } \right] } } \\ { { \displaystyle + \lambda _ { a b } \mathrm { C E } ( [ { \bf a } ; { \bf b } ] , [ \hat { \bf a } ; \hat { \bf b } ] ) + \lambda _ { q } \mathrm { C E } ( { \bf q } , \hat { \bf q } ) , } } \end{array}\tag{5}
$$

where $\begin{array} { r } { \mathrm { C E } ( p , q ) \ = \ - \sum _ { i } p _ { i } } \end{array}$ log q<sub>i</sub> and $\lambda _ { a b } = \lambda _ { q } = . 0 1$ One predicted profile then supports any sampled $F _ { \beta }$ decoder:

$$
\begin{array} { c } { { \hat { U } _ { \beta , j } = \displaystyle \frac { ( 1 + \beta ^ { 2 } ) \widehat { T P } _ { j } } { ( 1 + \beta ^ { 2 } ) \widehat { T P } _ { j } + \beta ^ { 2 } \widehat { F N } _ { j } + \widehat { F P } _ { j } } , } } \\ { { \hat { t } _ { \beta } = \displaystyle \alpha _ { \mathrm { a r g } \operatorname* { m a x } _ { j } \hat { U } _ { \beta , j } } . } } \end{array}\tag{6}
$$

Changing the precision–recall preference therefore changes only the decoder, not the learned head.

This is deliberately a plug-in rule. Even perfect conditional mean confusion counts generally satisfy $F _ { \beta } ( \mathbb { E } [ C$ | $F ] ) \neq \mathbb { E } [ F _ { \beta } ( C ) \mid F ]$ . Direct utility regression has a finitegrid decision-consistency property that our joint decoder does not. The empirical hypothesis is instead that cumulative sharing supplies a useful finite-sample inductive bias while enabling metric retargeting.

Direct Decision Baselines A scalar head regresses t<sup>∗</sup> with SmoothL1, while hard classification predicts its grid index. We additionally compare soft utility classification, direct expected-regret optimization, and unconstrained utility regression:

$$
\mathcal { L } _ { \mathrm { u t i l } } = \frac { 1 } { k } \sum _ { j } ( \hat { u } _ { j } - u _ { j } ) ^ { 2 } , \quad \hat { t } = \alpha _ { \mathrm { a r g m a x } _ { j } } \hat { u } _ { j } .\tag{7}
$$

The last objective is standard finite-action value regression. Under squared loss, its population prediction is $\mathbb { E } [ { \bf u } \mid F ]$ whose largest component maximizes conditional expected sampled utility. This makes it a stringent comparator, but it provides neither cross-metric reuse nor our structured confusion representation.

Sampled Oracle Performance For a labeled evaluation set of n decision units, annotation yields

$$
J _ { \mathrm { o r a c l e } } ^ { \mathcal { A } } = \frac { 1 } { n } \sum _ { x = 1 } ^ { n } \operatorname* { m a x } _ { j } u _ { j } ( x ) .\tag{8}
$$

This is an empirical upper bound over the sampled grid for the decomposable objective in Equation (1). Although unavailable at inference because it uses ground truth, it quantifies recoverable threshold headroom. It is not an upper bound for non-decomposable corpus metrics.

Computational Characteristics Caching Y avoids repeated base-model inference. Generic annotation evaluates k choices per unit in $O ( n k C _ { \mathcal { P } , m } )$ time and stores $O ( n k )$ values. Binary masks admit cumulative counts: once score maps are available, constructing all 257-point direct-utility profiles for the 800 Kvasir training/validation images takes 0.58 seconds and 0.82 MB of raw storage in one Intel Xeon 8255C run (7.79 seconds including loading and feature extraction). With $B ~ = ~ 1 2 8$ , the joint target contains 256 scalars.

Deployment At deployment, one pass of M produces Y and F. The head predicts (aˆ, b<sup>ˆ</sup>), and Equation (6) returns the chosen metric’s threshold in place of a fixed binarization value. No ground truth, joint target, or online utility replay is required.

## 5. Experiments

## 5.1. Experimental Protocols

We ask three questions: (1) does joint confusion prediction transfer better than scalar objectives and simple adaptive rules; (2) can one checkpoint retarget precision–recall preferences; and (3) does the same decoder generalize to a corruption family excluded from policy training? Every reported utility decomposes per image as required by Equation (1).

For mask binarization, we compare fixed 0.5, a validation-selected global threshold, and per-image Otsu [21]; Otsu uses its continuous output rather than our grid. A ground-truth-selected threshold appears only as the sampled oracle. Target-global diagnostics use external labels and are unavailable to deployed methods. Per-image Dice is $( 2 T P + \epsilon ) / ( 2 T P + F P + F N + \epsilon )$ and IoU is $( T P + \epsilon ) / ( T P + F P + F N + \epsilon )$ , with $\epsilon = 1 0 ^ { - 7 } ;$ all reported values are image macro-averages. Intervals use 10,000 paired bootstrap samples. Kvasir intervals resample images within each fixed dataset because patient/video identifiers are unavailable. KITTI intervals cluster all conditions of each original image. Within every declared split or outer fold, fitting and selection exclude the reported test images.

The supplementary material provides exact runners, manifests, tests, result JSON, and full environment details; runs use Python 3.10/PyTorch 2.6 on a Tesla V100 32 GB GPU.

## 5.2. Cross-Dataset Polyp Segmentation

Protocol We first study image-level mask binarization on Kvasir-SEG [14]. Images are resized to $3 5 2 \times 3 5 2$ and split into 700/100/200 training/validation/test images; detailed tables use the seed-20260726 reference split. We train an ImageNet-initialized ResNet34 U-Net [12, 22] with BCE and Dice losses using AdamW (batch 16, learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 } )$ , independent horizontal and vertical flips with probability 0.5, and at most 50 epochs with validation patience 8. After selecting its checkpoint by validation Dice, we freeze it. Only the 700 training tuples fit threshold heads; the 100 validation tuples select the global threshold, head checkpoint, and objective.

Each score map yields 61 label-free features: mean, standard deviation, minimum, maximum, binary entropy, fractions above 0.5 and above the map mean; 13 quantiles at {.001, .01, .025, .05, .1, .25, .5, .75, .9, .95, .975, .99, .999}; a normalized 32-bin histogram on [0, 1]; and nine foreground fractions at thresholds 0.1 through 0.9. AutoThreshold uses $B = 1 2 8$ and a 61-96-48-256 ReLU MLP with dropout 0.10/0.05 (23,152 parameters), versus 23,201 for the 257-value direct utility head. Kvasir validation selects $\lambda _ { a b } ~ = ~ \lambda _ { q } ~ = ~ . 0 1$ The capacity-matched scalar baseline is $\mathrm { ~ a ~ } 6 1 - 1 8 3 - 6 4 - 1$ MLP with 23,187 parameters; the original scalar trunk has 10,657. For tied scalar labels, ascending-grid argmax selects the smallest threshold. All neural heads use AdamW (batch 128, learning rate $1 0 ^ { - 3 }$ weight decay $1 0 ^ { - 4 } )$ , at most 800 epochs, and validation patience 100. Three fixed head seeds quantify conditional variability for the reference segmenter and split.

We then freeze the complete Kvasir-trained pipeline and transfer it to CVC-300, CVC-ClinicDB, CVC-ColonDB, and ETIS-LaribPolypDB under the standard polyp segmentation protocol [7]. No images, labels, or thresholds from these four datasets enter training or model selection.

Cross-Dataset Results Table 1 contrasts source-selected policies with a label-informed target-global diagnostic. On Kvasir test, a single threshold already suffices. Under transfer, Joint is the strongest deployable method listed for every target dataset and reaches $0 . 6 3 2 4 \pm 0 . 0 0 2 7$ equaldomain macro Dice, versus 0.6080 for Global and 0.6239 for continuous Otsu. Its paired gain over Otsu is +0.0085 $\left[ + 0 . 0 0 2 8 , + 0 . 0 1 5 0 \right]$ . Pooled over images instead of domains, Joint scores 0.5778 versus 0.5718, a gain of +0.0060 $[ - 0 . 0 0 0 6 , + 0 . 0 1 2 2 ]$ . The effect is therefore positive under both aggregations, but its interval excludes zero only for equal-domain averaging.

Objective and Structure Controls Table 2 holds score maps, features, source split, and validation protocol fixed. Matching the scalar head’s capacity does not improve transfer; hard labels, soft utility classification, and expectedregret training also remain below Otsu. Source validation selects soft-target temperature from $\{ . 0 0 5 , . 0 1 , . 0 2 , . 0 5 , . 1 \}$ and tunes the Ridge and GBRT searches. Replacing scalar targets with direct utility profiles raises external Dice by 0.0242 for the MLP and 0.0328 for Ridge, showing that the supervision target, rather than parameter count, drives the gain. Joint is +0.0023 over direct utility MLP $[ - 0 . 0 0 1 7 , + 0 . 0 0 6 5 ]$ ; direct utility Ridge is +0.0018 over Joint $[ - 0 . 0 1 1 2 , + 0 . 0 0 7 3 ]$ . Both intervals span zero. Areaonly, scalar Ridge, and scalar GBRT stay near Global, arguing against a simple foreground-ratio explanation.

Decision Geometry and Metric Retargeting $\displaystyle \mathrm { A t } \epsilon = . 0 0 5 ,$ the near-optimal set occupies 0.144 of the threshold range on average, and 11.7% of external profiles contain multiple disconnected near-optimal components. Joint-minus-Scalar gain correlates negatively with near-optimal width (Spearman $\rho = - . 2 9 4 )$ : mean gains from the narrowest to widest quartile are $+ . 1 0 1 5 , + . 0 1 8 7 , - . 0 0 2 7 , + . 0 0 0 5$ Thus Joint helps most when scalar threshold error is costly. Mean external decision regret drops from 0.1174 for scalar regression to 0.0878 for both direct utility and Joint. Figure 2 visualizes high- and low-gain cases. In Table 3, changing only β in Equation (6) yields improvements over Otsu for all three utilities.

Sensitivity and Profile Error With one fixed head seed, $\begin{array} { r l r } { B } & { { } = } & { \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 \} } \end{array}$ gives external Dice {.6326, .6300, .6351, .6306}, while $\{ 6 5 , 1 2 9 , 2 5 7 \}$ grid points give {.6334, .6341, .6351}. For $\lambda _ { q } = \{ 0 , . 0 1 , . 1 , 1 \}$ results are {.6337, .6351, .6271, .6264}; over-enforcing marginal reconstruction hurts the decision. On external data, prevalence MAE is .0494, TP-curve MAE .0266, FPcurve MAE .0149, and score-survival MAE .0294. These errors quantify the approximation behind the plug-in decision: the head does not reconstruct every score map exactly. Observed-Marginal Control A stricter alternative hardwires the histogram available at inference: $\hat { a } _ { b } = q _ { b } \sigma ( r _ { b } )$ and $\hat { b } _ { b } = q _ { b } [ 1 - \sigma ( r _ { b } ) ]$ . With B = 128 and source-selected composition weight .01, this control reaches 0.6095 external Dice. Increasing to B = 256 matches Joint’s 23,152 parameters but reaches 0.6070; both trail continuous Otsu at 0.6239 and Joint at 0.6351 for the same fixed head seed. Thus exact score-marginal reconstruction is not sufficient for accurate decisions, supporting soft marginal regularization rather than a hard constraint. The supplement includes full per-dataset outputs.

<table><tr><td>Dataset</td><td>N</td><td>Fixed .5</td><td>Global</td><td>Otsu</td><td>Target global†</td><td>Joint</td><td>Oracle</td></tr><tr><td colspan="8">Held-out source-domain test</td></tr><tr><td>Kvasir-SEG</td><td>200</td><td>0.8903</td><td>0.8902</td><td>0.8912</td><td>0.8904</td><td>0.8885</td><td>0.9238</td></tr><tr><td colspan="8">External target datasets without target tuning</td></tr><tr><td>CVC-300</td><td>60</td><td>0.7873</td><td>0.7876</td><td>0.7994</td><td>0.7883</td><td>0.8106</td><td>0.8632</td></tr><tr><td>CVC-ClinicDB</td><td>62</td><td>0.6566</td><td>0.6537</td><td>0.6603</td><td>0.6655</td><td>0.6691</td><td>0.7698</td></tr><tr><td>CVC-ColonDB</td><td>380</td><td>0.5583</td><td>0.5584</td><td>0.5826</td><td>0.5595</td><td>0.5837</td><td>0.6764</td></tr><tr><td>ETIS-LaribPolypDB</td><td>196</td><td>0.4325</td><td>0.4324</td><td>0.4533</td><td>0.4329</td><td>0.4662</td><td>0.5511</td></tr></table>

Table 1: Cross-dataset Dice for the reference Kvasir pipeline. Joint averages three head seeds; Otsu uses its continuous threshold. Target global<sup>†</sup> is selected with each target’s labels and is diagnostic, not deployable.

![](images/329289aa3cb7a2527bcb7367aecafac4994bc2c41da84c83221f1a57ee57adad.jpg)  
Figure 2: True and predicted Dice profiles for representative external images. Vertical lines mark thresholds from source Global, scala regression, Joint, and the sampled Oracle. The profile supplies a maximizer rather than a calibrated utility estimate; the right case exposes exposes a remaining<sub>a remaining selection error.</sub>

Full-Pipeline Variability For seeds 20260727 and 20260728, we regenerate the split, retrain and select the interval of : clean performance<sub>segmenter,</sub> <sub>cache</sub> <sub>new</sub> <sub>source</sub> <sub>and</sub> <sub>external</sub> <sub>score</sub> <sub>maps,</sub> able, and the data do not establish an improvement.<sub>and refit the heads without external selection. Across</sub> ted by common-corruption evaluation (Hendrycks<sub>all</sub> <sub>three</sub> <sub>pipelines,</sub> <sub>Joint-minus-Otsu</sub> <sub>differences</sub> <sub>are</sub> terich 2019), we construct b+.0112, +.0027, +.0101<sub>:</sub> <sub>a</sub> <sub>mean</sub> $+ 0 . 0 0 8 0 \ \pm \ 0 . 0 0 4 6 .$ gamma darkening (γ = 1.4/2.0), contrast (fac-positive in 3/3 runs. Direct utility is likewise positive in .40), white veil (α = .20/.40), Gaussian blur3/3, by +0.0055 ± 0.0017. Joint exceeds direct utility in 2/3 pipelines. The transfer gain therefore survives complete-pipeline variation, although Joint does not uniformly dominate direct value learning.

## <sup>AutoThreshold</sup> <sup>is</sup> <sup>a</sup> <sup>post-hoc</sup> <sup>threshold</sup> <sup>polic</sup>5.3. Cross-Corruption Threshold Adaptation

Setup We next test detachment from the base model and generalization across corruptions on KITTI Road [8]. We freeze NVIDIA’s SegFormer-B0 Cityscapes checkpoint at 512 × 1024 resolution<sup>1</sup> [3, 28] and use its road-class softmax probability as the score map; the segmenter is never The representation exposes auditable invarianfine-tuned on KITTI. Because official test annotations are ˆprivate, we evaluate the 289 labeled training images by five-<sup>b</sup>  fold nested cross-validation with seed 2027, stratified by the three KITTI Road categories. Every image occurs in exactly one outer test fold.

The clean experiment uses a compact 61-32-16-1 scalar MLP with 2,529 parameters and the same explicitly deh d f h d<sup>fined</sup> <sup>score-map</sup> <sup>statistics</sup> <sup>as</sup> <sup>Kvasir.</sup> <sup>It</sup> <sup>regresses</sup> <sup>the</sup> <sup>per-</sup> image Dice-optimal threshold on a 101-point grid. Ties favor the threshold nearest the fold’s global anchor, then the smaller threshold. Inner validation selects both checkpoint and shrinkage $\alpha \in \{ 0 , . 0 5 , . . . , 1 \}$ , where $\hat { t } = t _ { g } +$ $\alpha ( \mathcal { N } ( F ) - t _ { g } )$ and $t _ { g }$ is the validation-tuned global threshold.

<table><tr><td>Method</td><td>Kvasir test</td><td>External</td></tr><tr><td>Fixed 0.5</td><td>0.8903</td><td>0.6087</td></tr><tr><td>Global threshold</td><td>0.8902</td><td>0.6080</td></tr><tr><td>Otsu [21]</td><td>0.8912</td><td>0.6239</td></tr><tr><td>Area-only linear (scalar)</td><td>0.8904</td><td>0.6087</td></tr><tr><td>Ridge (scalar, 61 features)</td><td>0.8879</td><td>0.6014</td></tr><tr><td>GBRT (scalar, 61 features)</td><td>0.8899</td><td>0.6073</td></tr><tr><td>MLP scalar regression</td><td>0.8880±0.0003 0.6059±0.0104</td><td></td></tr><tr><td>Capacity-matched scalar</td><td>0.8878±0.0005</td><td> $0 . 5 9 9 7 { \scriptstyle \pm 0 . 0 0 2 3 }$ </td></tr><tr><td>MLP grid classification</td><td>0.8868±0.0028</td><td> $0 . 6 0 3 4 { \scriptstyle \pm 0 . 0 0 6 1 }$ </td></tr><tr><td>Soft utility classification</td><td>0.8900±0.0001</td><td> $0 . 6 1 2 6 { \pm } 0 . 0 0 1 5$ </td></tr><tr><td>Expected-regret policy</td><td> $0 . 8 9 0 3 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $0 . 6 1 0 0 { \scriptstyle \pm 0 . 0 0 1 9 }$ </td></tr><tr><td>Direct utility MLP</td><td>0.8888±0.0008</td><td> $0 . 6 3 0 1 { \scriptstyle \pm 0 . 0 0 2 0 }$ </td></tr><tr><td>Direct utility Ridge</td><td>0.8890</td><td>0.6342</td></tr><tr><td>Joint confusion (ours)</td><td>0.8885±0.0000 0.6324±0.0027</td><td></td></tr><tr><td>Oracle</td><td>0.9238</td><td>0.7151</td></tr></table>

Table 2: Objective and structure controls on frozen reference outputs. “External” averages four target datasets equally. Neural entries report mean ± sample standard deviation over three head seeds.

<table><tr><td>Method</td><td> $F _ { 0 . 5 }$ </td><td> $\mathrm { D i c e } / F _ { 1 }$ </td><td> $F _ { 2 }$ </td></tr><tr><td>Global</td><td>0.6250</td><td>0.6080</td><td>0.6351</td></tr><tr><td>Otsu [21]</td><td>0.6305</td><td>0.6239</td><td>0.6418</td></tr><tr><td>Joint confusion</td><td>0.6577</td><td>0.6324</td><td>0.6485</td></tr><tr><td>Oracle</td><td>0.7328</td><td>0.7151</td><td>0.7483</td></tr></table>

Table 3: Metric retargeting on four external datasets. Each of three seeds selects one checkpoint by Kvasir validation Dice; changing β then requires no retraining or target data.
<table><tr><td>Split/model seed</td><td>Global</td><td>Otsu</td><td>Direct utility</td><td>Joint</td></tr><tr><td>20260726</td><td>0.6080</td><td>0.6239</td><td>0.6279</td><td>0.6351</td></tr><tr><td>20260727</td><td>0.5941</td><td>0.6078</td><td>0.6130</td><td>0.6105</td></tr><tr><td>20260728</td><td>0.6286</td><td>0.6458</td><td>0.6531</td><td>0.6558</td></tr></table>

Table 4: Complete Kvasir pipeline repeats. Each row uses a new split and newly trained segmenter; external Dice averages four datasets equally. The head seed is fixed.

The corruption experiment compares this scalar policy with direct utility regression and the joint decoder in Equation (4). All share the nested LOCO splits, 61-32-16 trunk, optimizer, and validation-Dice criterion. The utility MLP has 101 sigmoid outputs and 4,229 parameters; Joint uses $B = 5 0$ , 100 softmax outputs, $\lambda _ { a b } = \lambda _ { q } = . 0 1$ , and 4,212 parameters. Both profile heads select a 101-point predicted argmax without scalar-label tie breaking or shrinkage. All KITTI heads use AdamW (learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 3 } )$ , at most 600 epochs, and validation patience 100. Selection batches are 32 for clean and 128 for corruption evaluation; final refits use batch 32 on the complete outer training fold for the validation-selected epoch count. Dice and IoU are image macro-averages excluding official invalid pixels. We designed this experiment after observing the initial clean-transfer result, so it is developmental rather than an untouched confirmation benchmark; nested crossvalidation nevertheless prevents fold-level leakage.

<table><tr><td>Method</td><td>Dice</td><td>IoU</td></tr><tr><td>Clean images Fixed 0.5 Otsu [21] Validation-tuned global Scalar regression Per-image oracle</td><td>0.8795 0.8790 0.9067 0.9072</td><td>0.7985 0.7977 0.8376 0.8398</td></tr><tr><td>Leave-one-corruption-family-out (LOCO) Clean-validation global Otsu [21] Seen-family global Scalar regression 0.8856 Direct utility profile 0.8961</td><td>0.9142 0.8735 0.8721 0.8767</td><td>0.8512 0.7958 0.7874 0.7988 0.8106 0.8237</td></tr></table>

Table 5: KITTI Road grouped out-of-fold results. LOCO excludes the evaluated corruption family from both policy training and selection. The best non-oracle result per setting is bold.

Clean and Corruption Results On clean images (Table 5), scalar regression reaches 0.9072 Dice and 0.8398 IoU, versus 0.9067 and 0.8376 for the inner-validation-tuned global threshold. The paired Dice difference is +0.0006 with a 95% bootstrap interval of $[ - 0 . 0 0 1 7 , + 0 . 0 0 2 7 ]$ : clean performance is comparable, and the data do not establish an improvement.

Motivated by common-corruption evaluation [13], we construct brightness (factors .75/.50), gamma darkening $( \gamma = 1 . 4 / 2 . 0 )$ , contrast (factors .70/.40), white veil (α = .20/.40), Gaussian blur $( \sigma = 1 . 2 / 2 . 5 )$ , and JPEG (quality 50/20). Every variant of an original image stays in the same outer fold. In each of six LOCO runs, both severities of one family are absent from inner training and validation. Clean images and the other five families select the head and epoch count; scalar-only anchor and shrinkage selection use the same seen-family validation units. Evaluation uses only outer-test images under the held family. The held-familyaware global in Table 5 is diagnostic because its validation labels reveal that family.

<table><tr><td>Family</td><td>Scalar Direct</td><td></td><td>Joint</td><td>t Aware†</td><td>J-S</td><td>J-D</td></tr><tr><td>Brightness</td><td>0.9029 0.9098 0.9076</td><td></td><td></td><td>0.9073</td><td>3+0.0048</td><td>-0.0022</td></tr><tr><td>Gamma</td><td>0.8949 0.90050.8979</td><td></td><td></td><td></td><td>0.8969+0.0029-0.0026</td><td></td></tr><tr><td>Gaussian blur0.81400.85320.8493</td><td></td><td></td><td></td><td></td><td>0.8492 +0.0353 -0.0039</td><td></td></tr><tr><td>JPEG</td><td>0.9016 0.9029 0.90300.8944+0.0014+0.0001</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Low contrast</td><td>0.9013 0.9058 0.90340.9031 +0.0021 -0.0024</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>White veil</td><td>0.8991 0.9043 0.90210.9010+0.0030−0.0023</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 6: LOCO Dice by excluded family. Aware uses labels from that family and is diagnostic. J−S and J−D compare Joint with scalar and direct utility.

Under LOCO, Joint reaches 0.8939 Dice: +0.0082 over scalar regression [+0.0064,+0.0101] and +0.0218 over Otsu [+0.0179,+0.0258]. Its gain over scalar is positive in all six held families, and it exceeds the label-informed heldfamily global by 0.0019. Direct utility remains strongest at 0.8961, with a paired +0.0022 advantage over Joint [+0.0007,+0.0038]. Intervals use a paired cluster bootstrap over original-image identity. Joint marginal L1 error is 0.0447 and prevalence MAE is 0.0247. KITTI thus provides complementary evidence for profile rather than scalar supervision, while also showing that decision-consistent direct utility can be better for a fixed metric.

## 6. Discussion and Conclusion

AutoThreshold is a post-hoc threshold policy, not a replacement segmenter. Its core contribution is a thresholdsufficient joint target: positive and negative score mass yield coherent confusion curves and multiple $F _ { \beta }$ decisions. Joint prediction avoids products of conditional mean prevalence and class-conditional rates; marginal reconstruction remains soft.

The representation exposes auditable invariants. Defining $\begin{array} { r } { \widehat { T N } _ { j } = \sum _ { b } \widehat { b } _ { b } - \widehat { F P } _ { j } } \end{array}$ yields a nonnegative, unit-sum confusion vector. Each score bin contributes to every lower threshold, coupling neighboring decisions through cumulative mass. These geometric, not accuracy, guarantees distinguish Joint from scalar labels and unconstrained utilities.

The experiments identify when adaptation matters. One global threshold suffices on source Kvasir and clean KITTI. Under cross-dataset transfer, Joint exceeds Otsu in all three pipelines and helps most at narrow utility optima. Its reference-pipeline difference from direct utility Ridge has an interval spanning zero; its distinctive benefit is retraining-free metric retargeting, not uniformly higher Dice. Oracle headroom therefore diagnoses when adaptation is worthwhile.

Developmental KITTI provides a complementary boundary case: LOCO removes the evaluated corruption family from every outer split, and Joint improves over scalar regression but trails direct utility. Because these synthetic transformations informed method development, the evidence supports generalization across predefined families, not arbitrary real-world shift or independent confirmation.

Equation (1) marks a methodological boundary. Image-, patch-, and dense-level policies require operator and utility to decompose at the same level; corpus ranking metrics and coupled operators such as NMS require groupwise decisions. Within this scope, AutoThreshold offers an auditable route from frozen score maps to adaptive masks while exposing the trade-off between structured plug-in estimation and direct value learning.

## References

[1] A. Beygelzimer, J. Langford, L. Li, L. Reyzin, and R. E. Schapire. Contextual bandit algorithms with supervised learning guarantees. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 19–26. PMLR, 2011.

[2] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis. Soft-nms - improving object detection with one line of code. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, pages 5562–5570. IEEE Computer Society, 2017.

[3] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The cityscapes dataset for semantic urban scene understanding. In IEEE Conference on Computer Vision and Pattern Recognition, pages 3213–3223, 2016.

[4] P. L. Donti, B. Amos, and J. Z. Kolter. Task-based end-to-end model learning in stochastic optimization. In Advances in Neural Information Processing Systems, pages 5484–5494, 2017.

[5] M. Dud´ık, J. Langford, and L. Li. Doubly robust policy evaluation and learning. In Proceedings of the 28th International Conference on Machine Learning, pages 1097–1104, 2011.

[6] C. Elkan. The foundations of cost-sensitive learning. In Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence, pages 973–978. Morgan Kauf mann, 2001.

[7] D.-P. Fan, G.-P. Ji, T. Zhou, G. Chen, H. Fu, J. Shen, and L. Shao. Pranet: Parallel reverse attention network for polyp segmentation. In Medical Image Computing and Computer Assisted Intervention, pages 263–273, 2020.

[8] J. Fritsch, T. Kuehnl, and A. Geiger. A new performance measure and evaluation benchmark for road detection algorithms. In 16th International IEEE Conference on Intelligent Transportation Systems, pages 1693–1700, 2013.

[9] J. Gao, T. Han, Q. Wang, Y. Yuan, and X. Li. Learning independent instance maps for crowd localization. CoRR, abs/2012.04164, 2020.

[10] Y. Geifman and R. El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Pro cessing Systems, pages 4878–4887, 2017.

[11] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In Proceedings of the

34th International Conference on Machine Learning, pages 1321–1330, 2017.

[12] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[13] D. Hendrycks and T. Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019.

[14] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, and H. D. Johansen. Kvasir-seg: A segmented polyp dataset. In MultiMedia Modeling, pages 451–462, 2020.

[15] W. Kotłowski and K. Dembczynski. Surrogate regret bounds´ for generalized classification performance metrics. Machine Learning, 106(4):549–572, 2017.

[16] M. Lin, Q. Chen, and S. Yan. Network in network. In International Conference on Learning Representations, 2014.

[17] T. Lin, M. Maire, S. J. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Com-´ mon objects in context. In Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V, pages 740–755. Springer, 2014.

[18] Z. C. Lipton, C. Elkan, and B. Naryanaswamy. Optimal thresholding of classifiers to maximize f1 measure. In Machine Learning and Knowledge Discovery in Databases, pages 225–239, 2014.

[19] S. Liu, D. Huang, and Y. Wang. Adaptive nms: Refining pedestrian detection in a crowd. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6459–6468, 2019.

[20] V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), June 21-24, 2010, Haifa, Israel, pages 807–814. Omnipress, 2010.

[21] N. Otsu. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1):62–66, 1979.

[22] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pages 234–241. Springer, 2015.

[23] P. Roy, S. Dutta, N. Dey, G. Dey, S. Chakraborty, and R. Ray. Adaptive thresholding: A comparative study. In 2014 International Conference on Control, Instrumentation, Communication and Computational Technologies (ICCICCT), pages 1182–1186, 2014.

[24] J. Song, W. Jiao, K. Lankowicz, Z. Cai, and H. Bi. A two-stage adaptive thresholding segmentation for noisy lowcontrast images. Ecol. Informatics, 69:101632, 2022.

[25] B. Wilder, B. Dilkina, and M. Tambe. Melding the datadecisions pipeline: Decision-focused learning for combinatorial optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1658–1665, 2019.

[26] S. Wu and F. Yang. Boosting detection in crowd analysis via underutilized output features. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 15609– 15618. IEEE, 2023.

[27] S. Wu, Z. Liu, W. Pei, J. Hong, and Z. Li. Faster, lighter, robuster: A weakly-supervised crowd analysis enhancement network and a generic feature extraction framework. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2022, New Or leans, LA, USA, June 19-20, 2022, pages 4049–4058. IEEE, 2022.

[28] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo. Segformer: Simple and efficient design for seman tic segmentation with transformers. In Advances in Neural Information Processing Systems, pages 12077–12090, 2021.

[29] W. Yan, Y. Qian, C. Wang, and M. Yang. Threshold-adaptive unsupervised focal loss for domain adaptation of semantic segmentation. IEEE Trans. Intell. Transp. Syst., 24(1):752– 763, 2023.

[30] X. Yang, J. Wu, L. He, S. Ma, Z. Hou, and W. Sun. Cpss-fat: A consistent positive sample selection for object detection with full adaptive threshold. Pattern Recognit., 141:109627, 2023.

[31] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren. Distance-iou loss: Faster and better learning for bounding box regression. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second In novative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 12993–13000. AAAI Press, 2020.

[32] Z. Zheng, P. Wang, D. Ren, W. Liu, R. Ye, Q. Hu, and W. Zuo. Enhancing geometric factors in model learning and inference for object detection and instance segmentation. IEEE Transactions on Cybernetics, 52(8):8574–8586, 2022.