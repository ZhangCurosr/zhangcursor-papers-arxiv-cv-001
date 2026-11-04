# Bounded AUKS: A Novel Evaluation Metric for Anomaly Detection via Bounded Area Under the Kolmogorov–Smirnov Curve

Yanshuo ZHANG<sup>1</sup> and Juan DU<sup>∗1,2,3</sup>

<sup>1</sup>Smart Manufacturing Thrust, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, 100190, China

<sup>2</sup>Data Science and Analytics Thrust, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, 100190, China

<sup>3</sup>Department of Mechanical and Aerospace Engineering, The Hong Kong University of Science and Technology, Hong Kong SAR, 999077, China

## Abstract

Recent literature widely used global metrics such as AUROC and AUPRO to quantify the detection performance. AUROC and AUPRO scores are approaching 100% in some public datasets such as MVTec AD dataset and VisA dataset, which misleadingly suggests that the pixel-level anomaly detection task has already been solved. Threshold-independent metrics represented by AUROC inherently ignore the threshold selection process, which is inevitable in practical applications. Although AUPRO restricts the False Positive rates (FPRs) to an acceptable range, extremely low FPRs can lead to increased false negative rates (FNRs). We argue that there is a lack of metrics capable of quantifying the separation between the anomaly score distributions of normal and anomalous samples, which hinders advancements in anomaly detection. In response, we propose the Bounded Area Under the Kolmogorov– Smirnov curve (Bounded AUKS), a partial integral metric that integrates the diference between TPR and FPR within a data-driven bounded range. Specifically, we introduce AUKS from KS integral as a direct measure of separability between the anomaly score distributions. The bounded range is determined only by score distributions of normal test samples. Lower bound excludes the uninformative high-FPR region. Upper bound avoids evaluation in extreme regions that aggressively push FPR towards zero, which result in a very low recall. Our experiments on 9 models across MVTec AD and VisA datasets show that Bounded AUKS can address the limitations of existing metrics and has a better performance in evaluating unsupervised anomaly detection models.

Keywords: evaluation metrics, distribution separability, threshold selection

<sup>∗</sup>Email: yzhang420@connect.hkust-gz.edu.cn; juandu@ust.hk (corresponding author).

## 1 Introduction

Anomaly detection is a critical task in the field of vision-based inspection, fundamentally aiming to accurately distinguish anomalies from normals. This task is based on normal patterns, where models must identify anomalous features or behavioral patterns that deviate from normality (Yan et al. 2017, Austin et al. 2024). In visual inspection involving images, videos and 3D point-cloud data, the initial objective is typically to determine whether an anomaly is present at the instance level (Du et al. 2025). However, mere instance-level detection is often insuficient. For example, PointSGRADE combines sparse learning with a graph representation to detect surface anomalies in unstructured 3D point clouds (Tao & Du 2025). In image-based inspection, anomaly localization requires precisely identifying the spatial extent of anomalies at the pixel level (Bertoldo et al. 2024, Rafiei et al. 2023).

In most modern anomaly localization pipelines, models output a continuous anomaly score � (or a pixel-wise anomaly map), which will be converted into a hard prediction via a decision threshold � (e.g., predicting anomaly when $S > \theta )$ to obtain actionable decisions in deployment. Notably, the threshold is not an auxiliary detail: it directly determines the precision–recall trade-of, controls the false positive rate (FPR), and the shape of predicted anomaly regions. In unsupervised settings, anomaly labels are not available to calibrate � on a test set, so the selection of practical thresholds is typically based on anomalyfree statistics (i.e., independent of information on anomalies) (Wilkinghof & Imoto 2024, Bertoldo et al. 2024). Therefore, the existing evaluation protocol that involves sweeping over thresholds implicitly presumes a plausible set of candidate thresholds.

The rapid development of anomaly detection methods has also raised increasing demands on evaluation metrics. Specifically, after converting continuous anomaly scores into binary predictions through a decision threshold, model performance can be assessed using conventional metrics such as Accuracy and F1 score, as defined in Supplementary Material Section S2, Table S2. Meanwhile, anomaly detection studies also commonly adopt threshold-independent metrics to evaluate model performance from a global perspective, including the Area Under the Receiver Operating Characteristic Curve (AUROC) (Fawcett 2006, Bergmann et al. 2021) and the Area Under the Precision-Recall Curve (AUPR) (Davis & Goadrich 2006). For the task of anomaly localization, specialized evaluation metrics have been developed to assess region-level detection performance. A prominent example is the Area Under the Per-Region Overlap (AUPRO) curve (Bergmann et al. 2021), which emphasizes detection sensitivity at low false positive rates (FPR), a regime of particular importance for practical industrial deployment. Despite these advances, such evaluation metrics still sufer from three major limitations:

• Inflated Scores: The AUROC scores are approaching 100% on the open dataset MVTec Anomaly Detection (MVTec AD) (Bergmann et al. 2019, 2021) and Vision Anomaly (VisA) (Zou et al. 2022), which misleadingly suggests that the pixel-level anomaly detection task has already been solved (Bertoldo et al. 2024). This is an inflated figure.

• Lack of Separability Measure: Existing metrics primarily assess ranking quality or detection accuracy, but they do not explicitly quantify the degree of separation between the score distributions of normal and anomalous samples. In other words, they cannot indicate whether the two distributions are well separated with minimal overlap.

• Disregard for Threshold Selection: Threshold-dependent metrics such as F1- max capture only the single best operating point, ofering no insight into performance stability across thresholds. AUROC bypasses threshold selection entirely, providing no deployment guidance. AUPRO restricts evaluation to a fixed low-FPR regime, a choice that may not suit all applications.

Table 1: Recent high-performing anomaly detection models with near-perfect AUROC scores on MVTec AD and VisA benchmarks. Pixel-level scores are shown.
<table><tr><td>Model</td><td></td><td>First Publication MVTec AD AUROC VisA AUROC</td><td></td></tr><tr><td>Dinomaly (Guo et al. 2025)</td><td>2025-06</td><td>98.6</td><td>99.1</td></tr><tr><td>HETMM (Chen, Xie, Yang &amp; Lai 2025)</td><td>2025-05</td><td>99.0</td><td>99.1</td></tr><tr><td>URD (Liu et al. 2025)</td><td>2025-02</td><td>99.0</td><td>99.1</td></tr><tr><td>GLASS (Chen, Luo, Lv &amp; Zhang 2025)</td><td>2024-11</td><td>99.3</td><td>98.8</td></tr><tr><td>GLAD (Yao et al. 2025)</td><td>2024-10</td><td>98.6</td><td>98.6</td></tr></table>

Regarding the first limitation, Table 1 presents the pixel-level AUROC results of representative recent anomaly detection models on the MVTec AD and VisA datasets. As illustrated, many recent state-of-the-art (SOTA) models report near-perfect pixel-level AUROC scores on MVTec AD and VisA benchmarks. However, these high AUROC values often obscure significant shortcomings in anomaly localization performance.

![](images/c34a814aa97ff0831565caf8b06d53c52c8186ff811a5d38d1ecb878b772592d.jpg)  
(a)

![](images/0ae239eda39f564806cce9c17ca1fe048b9d553b624a44708cbe1b528aa11d53.jpg)  
(b)

![](images/98e1e0aa24c09a7b19a4faf3b4f1f882ab315b5479e5514137ac72ef5fc26c0d.jpg)  
(c)

![](images/a2928027c29a5640f3d896de6c8da4a68450753cc35ef34ef44fdac617113a66.jpg)  
(d)  
Figure 1: Comparison of ideal score distributions under diferent metrics. (a) A narrow margin under AUROC = 1.0. (b) A wide margin under $\mathrm { A U R O C } = 1 . 0$ . Both examples in (a) and (b) achieve the same AUROC, but they difer fundamentally in robustness of threshold selection. (c) The strict requirement for $\mathrm { { A U K S } = 1 . 0 }$ , in which normal-scores are 0 and anomalous-scores are 1, underscores that a distinct separation between distributions makes the threshold easier to determine. (d) Bounded AUKS = 1.0. A perfect value requires the normal distribution to collapse to a constant � and all anomalies to exceed �.

Regarding the second limitation, perfect anomaly detection fundamentally requires separating positive (anomalous) and negative (normal) distributions of anomaly scores completely.

Both Lunardi et al. (2025) and Jiang et al. (2023) emphasize such similar concepts. Without explicitly measuring score-distribution separability, one cannot determine whether a high evaluation score arises from a genuine separation between normal and anomalous samples or merely from a favorable ranking efect. Moreover, when the two score distributions overlap substantially, samples in the overlapping region become ambiguous, making reliable discrimination more dificult (Metzner et al. 2022). However, no existing metric directly quantifies such distributional separability. AUROC assesses score ranking (Maurer & Pontil 2020), while AUPR and AUPRO evaluate precision-recall trade-ofs and region overlap, respectively.

Regarding the third limitation, existing metrics largely bypass the candidate threshold selection problem. AUROC is threshold-independent because it integrates over all thresholds, ignoring the dificulty of selecting possible thresholds in practice. As shown in Fig. 1 (a–b), AUROC= 1.0 can correspond to either a negligibly small or a wide feasible threshold margin, but a larger margin clearly provides better robustness in threshold selection. AUPRO partially addresses threshold selection by restricting the evaluation to a low-FPR range (e.g. $\mathrm { F P R } \le 0 . 0 5 )$ . However, this range includes extremely low FPRs corresponding to very high thresholds, which can lead to an unacceptable increase in the False Negative Rate (FNR) or decrease in the recall. A detailed visual analysis is provided in Supplementary Material Section S6. Meanwhile, calculating FPR is a posterior process that requires access to anomaly labels, which is not possible in common unsupervised settings. Consequently, existing metrics do not capture model performance under realistic deployment conditions, where the candidate threshold range is necessarily derived from anomaly-free samples alone. To address these limitations, we propose the Bounded Area Under the Kolmogorov– Smirnov (Bounded AUKS) curve. We derive AUKS from the Kolmogorov–Smirnov(KS) integral as a direct measure of separability between normal and anomalous scores distributions, equivalent to the 1-Wasserstein distance. We define the candidate threshold interval exclusively from normal samples, consistent with the unsupervised deployment setting in which anomalous labels are unavailable for threshold selection. The lower bound $\theta _ { \mathrm { m i n } } = \mathrm { M e d i a n } + \alpha \mathrm { M A D }$ is motivated by robust statistics and filters out the low-threshold region that is dominated by high FPR yet provides limited discriminative information. The upper bound $\theta _ { \mathrm { m a x } }$ defines the upper limit of the interval and excludes excessively high thresholds, since the segmentation corresponding to the thresholds near the maximum value can be overly conservative and the recall can be markedly reduced. We use Extreme Value Theory(EVT) to determine $\theta _ { \mathrm { m a x } }$ at tail-risk level � because this upper bound is governed by rare upper-tail of normal-scores. Bounded AUKS integrates the diference between TPR and FPR strictly over the interval defined by the bounded thresholds region.

The contributions of this paper are as follows:

• We propose Bounded AUKS, a bounded KS-integral metric that quantifies anomaly score distribution separability within a threshold range, addressing the limitations of ranking-based evaluation.

• We introduce anomaly-free threshold bounds inferred solely from anomaly scores of normal test samples, including a robust lower bound and an EVT-based upper bound, to define a practically relevant evaluation interval without requirements of known anomalies.

• We provide numerical studies on MVTec AD and VisA datasets, demonstrating that Bounded AUKS has a better performance in evaluating unsupervised anomaly detection models.

The remainder of this paper is organized as follows. Section 2 reviews related work on anomaly detection evaluation metrics and threshold selection. Section 3 presents the proposed Bounded AUKS and its properties. Section 4 describes the experimental settings and reports the main results. Section 5 concludes the paper.

## 2 Related Work

The choice of anomaly-detection evaluation metrics remains actively debated. Fourure et al. (2021) showed that non-standardized protocols and varying anomaly rates can substantially inflate F1 and AUPR, undermining comparisons across datasets. At the pixel level, Rafiei et al. (2023) found that precision–recall-based metrics better reflect relative localization performance under severe class imbalance. For imbalanced data, Ma et al. (2025) recommended average precision across recall levels and the Dice similarity coeficient.

Recent work has also introduced specialized metrics. Zhang et al. (2023) proposed Instance Average Precision (IAP), which defines region-level recall by treating a region as detected when at least half of its pixels are correctly identified. IAP is also used for threshold selection, while pixel-level precision at fixed recall (precision-at-k%) supports model comparison. Nevertheless, IAP remains a ranking-based precision–recall metric and relies on a heuristic 50% overlap criterion. Bertoldo et al. (2024) proposed the Area Under the Per-Image Overlap (AUPIMO) curve, an AUROC-derived recall metric that validates models using only normal images, reducing bias toward previously observed anomalies. Its per-image recall supports both individual- and cross-image analyses. However, AUPIMO evaluates localization recall within prescribed lower and upper FPR bounds. This interval remains parameterized by FPR, an a posteriori quantity that depends on access to anomaly labels; like AUROC, AUPIMO primarily measures score ranking rather than the separability of normal- and anomalous-score distributions. Focusing instead on industrial operating points, Baitieva et al. (2025) introduced two image-level ROC points: Presorted Good at 2% (PG2), defined as TNR at $\mathrm { F N R } = 0 . 0 2$ , and Presorted Bad at 2% (PB2), defined as

TPR at FPR = 0.02 (where good denotes normal and bad denotes anomalous). Because these are point estimates at fixed thresholds, they do not summarize performance across a range of operating conditions.

Wilkinghof & Imoto (2024) proposed the F1-expected value (F1-EV) score to quantify threshold-estimation dificulty and a bounded variant for realistic operating points. Its integration interval uses the lower bound $\mu - \alpha \sigma$ and the upper bound $\theta _ { \mathrm { o p t } } + \alpha \sigma$ , where $\mu$ and � are the mean and standard deviation of normal scores and $\theta _ { \mathrm { o p t } }$ maximizes F1. Although this construction reduces the influence of outliers and correlates strongly with deployment performance, it assumes a Gaussian-like normal-score distribution and may therefore be unsuitable for heavy-tailed or skewed scores. Moreover, using the groundtruth $\theta _ { \mathrm { o p t } }$ requires anomalous-sample information and introduces hindsight bias. The lower bound is also questionable in practice: thresholds below $\mu$ would classify approximately half of the normal samples as anomalous. Supplementary Material Section S1 provides further discussion of threshold selection.

## 3 Bounded AUKS Metric

In this section, we formalize the proposed metric. We start by defining the Area Under the Kolmogorov–Smirnov curve (AUKS) as a directed separability measure derived from the gap between distributions of normal-scores and anomalous-scores. We then introduce Bounded AUKS, which refines this measure by restricting evaluation to a operational range $[ \theta _ { \mathrm { L o w e r } } , \theta _ { \mathrm { U p p e r } } ]$ derived exclusively from anomaly scores of normal samples.

## 3.1 Score distributions and TPR–FPR

Let $S$ be the continuous anomaly score output by the model. We denote the scores of normal samples as $S _ { \mathrm { n o r m } }$ and those of anomalous samples as $S _ { \mathrm { a n o m } }$ . Denote by $F _ { \mathrm { n o r m } } ( \theta )$

and $F _ { \mathrm { a n o m } } ( \theta )$ the Cumulative Distribution Function(CDF) of � for normal and anomalous samples, respectively. Using the decision rule “predict anomaly if $S > \theta ^ { \prime \prime }$ , we have:

$$
\mathrm { T P R } ( \theta ) = \mathbb { P } ( S > \theta \mid \mathrm { a n o m } ) = 1 - F _ { \mathrm { a n o m } } ( \theta ) , \quad \mathrm { F P R } ( \theta ) = \mathbb { P } ( S > \theta \mid \mathrm { n o r m } ) = 1 - F _ { \mathrm { n o r m } } ( \theta ) .
$$

Consequently, the pointwise gap between the CDFs corresponds directly to the gap between TPR and FPR:

$$
F _ { \mathrm { n o r m } } ( \theta ) - F _ { \mathrm { a n o m } } ( \theta ) = \mathrm { T P R } ( \theta ) - \mathrm { F P R } ( \theta ) .\tag{1}
$$

The standard Kolmogorov-Smirnov statistic (Frank J. Massey 1951) evaluates distribution separability by measuring the maximum absolute diference between the CDFs:

$$
\mathrm { K S ~ s t a t i s t i c } = \operatorname* { m a x } | F _ { \mathrm { n o r m } } ( \theta ) - F _ { \mathrm { a n o m } } ( \theta ) | = \operatorname* { m a x } | \mathrm { T P R } ( \theta ) - \mathrm { F P R } ( \theta ) | .\tag{2}
$$

However, the absolute value is inappropriate for anomaly detection because it removes the direction of separation. Consequently, inverted score distributions can still produce a large KS value even when normal samples are assigned higher anomaly scores than anomalous ones. We therefore use the signed gap $\mathrm { K S } ( \theta ) = \mathrm { T P R } ( \theta ) - \mathrm { F P R } ( \theta )$ , so that positive values indicate the desired ordering and negative values indicate inversion.

## 3.2 AUKS: integral separability and mathematical grounding

We define the area under the KS curve (AUKS) as the integral of KS over the entire normalized score domain [0, 1]:

$$
\mathrm { A U K S } = \int _ { 0 } ^ { 1 } \left( F _ { \mathrm { n o r m } } ( \theta ) - F _ { \mathrm { a n o m } } ( \theta ) \right) d \theta = \int _ { 0 } ^ { 1 } \left( \mathrm { T P R } ( \theta ) - \mathrm { F P R } ( \theta ) \right) d \theta .\tag{3}
$$

In an efective anomaly detection setting, models should assign consistently higher scores to anomalous samples, establishing the ideal stochastic ordering $F _ { \mathrm { n o r m } } ( \theta ) \geq F _ { \mathrm { a n o m } } ( \theta )$ for any applicable threshold �. Therefore, we purposely omit the absolute value inside the integral.

A larger AUKS thus mathematically guarantees that the anomaly score distribution is structurally separated to the right of the normal distribution, providing a robust global summary of separability.

AUKS also has an optimal-transport interpretation. For one-dimensional distributions on [0, 1], the 1-Wasserstein distance is $\begin{array} { r } { W _ { 1 } ( F _ { \mathrm { n o r m } } , F _ { \mathrm { a n o m } } ) = \int _ { 0 } ^ { 1 } | F _ { \mathrm { n o r m } } ( \theta ) - F _ { \mathrm { a n o m } } ( \theta ) | } \end{array}$ �� (Panaretos & Zemel 2019, Vallender 1974). Under the stochastic ordering assumed above, the absolute value can be removed, and AUKS therefore equals $W _ { 1 }$ . When this ordering is violated, the signed AUKS instead penalizes score inversion and need not equal the Wasserstein distance.

## 3.3 Bounded AUKS: Anomaly-free threshold bounds

While AUKS measures global separability, practical deployment requires evaluating models only within a reliable � range. We propose Bounded AUKS, which restricts the integration to a data-driven interval $[ \theta _ { \mathrm { L o w e r } } , \theta _ { \mathrm { U p p e r } } ]$ inferred solely from the data of normal samples.

The Bounded AUKS is formulated as the average signed separability over this defined interval:

$$
\mathrm { B o u n d e d } \mathrm { A U K S } = \frac { 1 } { \theta _ { \mathrm { U p p e r } } - \theta _ { \mathrm { L o w e r } } } \int _ { \theta _ { \mathrm { L o w e r } } } ^ { \theta _ { \mathrm { U p p e r } } } \left( \mathrm { T P R } ( \theta ) - \mathrm { F P R } ( \theta ) \right) d \theta .\tag{4}
$$

The proposed bounds are as follows:

$$
\theta _ { \mathrm { L o w e r } } = \mathrm { M e d i a n } ( S _ { \mathrm { n o r m } } ) + \alpha \cdot \mathrm { M A D } ( S _ { \mathrm { n o r m } } ) , \quad \theta _ { \mathrm { U p p e r } } \ = \mathrm { E V T } ( S _ { \mathrm { n o r m } } , q )\tag{5}
$$

where � and $q$ are tunable hyperparameters, MAD denote the median absolute deviation.   
We provide the detailed justification for each bound below.

Figure 2 illustrates the efect of diferent parameters for Bounded AUKS on the resulting segmentation and corresponding integration bounds, using the Pill category of MVTec AD

q= 0.1 q= 0.01 F1 max q= 0.001 q= 0.0001

![](images/8ed6a3d757ce0c0c7f835fc2401fb5e8328be70644b4d73c4dc5acb6dfa910ac.jpg)

![](images/aa76f9595700ada387a3eeb85502f7f302473d33b16f1d9f795bb5c12363ab9c.jpg)  
Original  
Zoom in  
(b) Lower Bounds of Bounded AUKS(a) Anomaly Maps of Pill (MVTec AD)  
(c) Upper Bounds of Bounded AUKS

(d) Lower and Upper Bounds Curves  
![](images/dd97e24948e23a991985e9dce5024312215cb105b6493f0e443583ec3ad3dc14.jpg)  
Figure 2: Impact of bounding parameters on segmentation bounds and metric curves for the MVTec AD Pill. (a) Corresponding anomaly map. (b) Lower bounds derived using diferent �. (c) Upper bounds configured using varying EVT risk levels �. (d) Alignment of these bounds along the corresponding curves.

dataset as an example. The multi-panel figure shows how variations in the bounds influence the spatial segmentation and map onto the Probability Density Function (PDF), CDF, and KS curves. For the upper bound, regions corresponding to excessively high thresholds are excluded, thereby preventing a substantial reduction in TPR. We observe that our chosen setting (� = 0.01) produces an appropriate segmentation and is reasonably close to the theoretically F1-max optimal point. If the allowable risk � is set too large (e.g., � = 0.1), the resulting threshold becomes excessively low, leading to a loose segmentation with many false positives and potentially causing the upper bound to approach or even fall below the lower bound. Conversely, if $q$ is excessively small $( \mathrm { e . g . , } q = 0 . 0 0 0 1 )$ , the resulting threshold becomes overly stringent, causing the segmentation to contract to such an extent that the anomaly may fail to be detected altogether. In Figure 2 (d), we observe that an excessively restrictive � provokes a steep collapse in TPR (falling below 0.7), whereas optimal lower bounds successfully sidestep the high-FPR regime (FPR remaining below about 0.15) while retaining a broad and informative integration interval. Regarding the lower bound, which is governed by the multiplier $\alpha ,$ its primary function is to filter out the high FPR region, while guaranteeing the residual integration interval under the KS curve remains suficiently wide to capture the separability of distributions.

## 3.3.1 Upper Bound: EVT-Based Bound and Recall Preservation

The upper bound $\theta _ { \mathrm { U p p e r } }$ defines a principled upper limit for the threshold interval used in Bounded AUKS. Because anomalous labels are unavailable in the unsupervised setting, we avoid threshold selection strategies that depend on FPR-indexed evaluation protocols, such as those used in AUPRO assessment. Instead, the selected threshold should be determined solely by the statistical behavior of normal data, which is also the reason that our analysis is only based on normal-scores. Since the threshold is primarily determined by the extreme upper tail of the normal-score distribution, where samples are scarce, direct estimation via empirical quantiles can be unstable in the high-confidence regime. We therefore employ EVT to model the tail, rather than imposing a strong parametric assumption on the entire distribution. Moreover, EVT enables us to specify a risk level $q$ that defines a nominal false alarm probability and thereby yielding a principled threshold. When the tail model is well specified and the sample size is suficiently large, the resulting FPR approaches the prescribed risk level $q .$

Normal-score distribution tail modeling. Let $S _ { \mathrm { n o r m } }$ denote normal-scores and select a high initial threshold � (e.g., we use the 90th percentile). We then focus on the excesses of normal-scores above �, defined as $Y = S _ { \mathrm { n o r m } } - u \vert S _ { \mathrm { n o r m } } > u$

Under the Peaks-over-Threshold (POT) framework, these excesses can be modeled by a Generalized Pareto Distribution (GPD) with shape parameter $\xi$ and scale parameter $\sigma ,$ whose cumulative distribution function is given by

$$
\mathbb { P } ( Y \le y ) = 1 - \left( 1 + \xi y / \sigma \right) ^ { - 1 / \xi } , \qquad y > 0 ,\tag{6}
$$

where $\xi > 0$ corresponds to a heavy tail (unbounded support). Let $\zeta _ { u } = \mathbb { P } ( S _ { \mathrm { n o r m } } > u )$ be the exceedance probability. Supplementary Material Section S3, Table S3 summarizes the key parameters used to instantiate this tail model and the resulting upper bound.

EVT-based upper bound definition. Let $q \in ( 0 , \zeta _ { u } ]$ denote a target tail risk level, which serves as a nominal false alarm probability under the EVT tail model. Under the POT-GPD approximation, the upper bound $\theta _ { \mathrm { U p p e r } }$ is defined as: $\mathbb { P } ( S _ { \mathrm { n o r m } } > \theta _ { \mathrm { U p p e r } } ) = q$ Using the standard POT tail approximation

$$
\begin{array} { r } { \mathbb { P } \big ( S _ { \mathrm { n o r m } } > \theta \big ) = \zeta _ { u } \big ( 1 + \xi ( \theta - u ) / \sigma \big ) ^ { - 1 / \xi } , \qquad \theta \geq u , } \end{array}\tag{7}
$$

we obtain

$$
\theta _ { \mathrm { U p p e r } } ( q ) = u + \frac { \sigma } { \xi } \left[ \left( \frac { q } { \zeta _ { u } } \right) ^ { - \xi } - 1 \right] .\tag{8}
$$

In the small-risk regime $q \ll \zeta _ { u }$

$$
\theta _ { \mathrm { U p p e r } } ( q ) \sim \frac { \sigma \zeta _ { u } ^ { \xi } } { \xi } q ^ { - \xi } , \qquad q \to 0 ,\tag{9}
$$

and hence $\theta _ { \mathrm { U p p e r } } ( q ) = \Theta ( q ^ { - \xi } )$ . With � normal samples and $N _ { u }$ exceedances above �, we use $\hat { \zeta } _ { u } = N _ { u } / N$ together with fitted $( \hat { \xi } , \hat { \sigma } )$ , yielding

$$
\hat { \theta } _ { \mathrm { U p p e r } } ( q ) = u + \frac { \hat { \sigma } } { \hat { \xi } } \left[ \left( \frac { q N } { N _ { u } } \right) ^ { - \hat { \xi } } - 1 \right] .\tag{10}
$$

Recall–risk trade-of and practical choice of $q = 0 . 0 1$ . Decreasing � makes the EVT upper bound increasingly conservative. If the bound moves beyond most anomalous scores, recall drops sharply, producing what we term the recall clif. Parameterized simulations and empirical recall–risk curves on MVTec AD Pill and VisA pcb1 consistently exhibit this low- $^ { - } q$ collapse. Together with Figure 2, these results motivate $q = 0 . 0 1$ as a stable setting in our experiments. This value avoids the collapse associated with smaller values without producing the loose upper bound observed at $q = 0 . 1$ . The complete derivation, simulation design, and all simulated and empirical curves are provided in Supplementary Material Section S3.1.

## 3.3.2 Lower Bound: Robust Statistics and Parameter Mapping

We adopt a robust formulation: $\theta _ { \mathrm { L o w e r } } = \mathrm { M e d i a n } + \alpha \cdot \mathrm { M A D }$ . We rely on Median and MAD rather than Mean and Standard Deviation for two key reasons grounded in robust statistics. First, anomaly scores typically follow a heavy-tailed, right-skewed distribution where the mean is heavily pulled towards the tail. Our median-based formulation inherently adapts to this skewness by targeting the right side of the central mass, efectively skipping the uninformative bulk. Second, Median and MAD provide superior resistance to outliers. Unlike the mean, which can be drastically distorted by even a single extreme prediction error (a 0% breakdown point), the median and MAD remain stable as long as less than half of the scores are extreme (a 50% breakdown point). This guarantees that our lower bound will not be artificially inflated by a few occasional noisy normal samples.

Proposed Mapping of � to FPR. To mathematically map � to the percentile $p ,$ which represents the percentage of anomaly scores of normal samples sorted from smallest to largest $( \mathrm { i . e . , F P R } = 1 - p )$ , we must approximate the MAD.

We model the normal-score distribution using the Fréchet distribution $( F ( x ) \ =$ exp $( - ( x / s ) ^ { - \gamma } ) _ { , } ^ { \textrm { } }$ ) (Coles 2001). We adopt this specific distribution because, in EVT, the Fréchet distribution is the standard macroscopic model for strictly positive data with heavy, polynomial-decay tails. Thus, while $\theta _ { \mathrm { U p p e r } }$ uses POT and GPD locally to rigorously constrain the extreme upper excesses, the Fréchet distribution serves as its natural global counterpart to characterize the overall right-skewed behavior of anomaly scores.

Since normal-score distributions are bounded by zero on the left and highly skewed to the right, negative deviations from the median are strictly compressed. As a result, the absolute deviations are overwhelmingly dominated by the upper half of the data. Therefore, we safely approximate MAD as the distance between the upper quartile $( 7 5 ^ { \mathrm { t h } }$ percentile, $x _ { 0 . 7 5 } )$ and the Median $( 5 0 ^ { \mathrm { t h } }$ percentile, $x _ { 0 . 5 } )$ : $\mathrm { M A D } = x _ { 0 . 7 5 } - x _ { 0 . 5 }$ . Under the Fréchet assumption, the ratio $R = x _ { 0 . 7 5 } / x _ { 0 . 5 }$ elegantly cancels out the scale parameter � and depends solely on the shape parameter $\gamma$ (where lower $\gamma$ signifies heavier right tails):

$$
R = \left( \frac { \ln ( 1 / 0 . 7 5 ) } { \ln ( 1 / 0 . 5 ) } \right) ^ { - 1 / \gamma } = \left( \frac { \ln 4 - \ln 3 } { \ln 2 } \right) ^ { - 1 / \gamma } .
$$

Substituting $\theta _ { \mathrm { L o w e r } } = x _ { 0 . 5 } [ 1 + \alpha ( R - 1 ) ]$ into the CDF yields the explicit mapping:

$$
p ( \alpha ) = F ( \theta _ { \mathrm { L o w e r } } ) = \exp \left( - \ln 2 \cdot \left[ 1 + \alpha \cdot \left( \left( \frac { \ln 4 - \ln 3 } { \ln 2 } \right) ^ { - 1 / \gamma } - 1 \right) \right] ^ { - \gamma } \right) .\tag{11}
$$

Table 2 presents the resulting FPR for �, serving as a reference for parameter selection.

Table 2: Mapping � to FPR under Fréchet Distribution.
<table><tr><td>Tail shape</td><td colspan="6">α</td></tr><tr><td>γ</td><td>0.2</td><td>1.0</td><td>2.0</td><td>3.0</td><td>4.0</td><td>5.0</td></tr><tr><td>4 (Light)</td><td>37.8%</td><td>25.0%</td><td>10.9%</td><td>5.3%</td><td>2.8%</td><td>1.6%</td></tr><tr><td>2 (Typical)</td><td>37.0%</td><td>25.0%</td><td>13.5%</td><td>9.2%</td><td>5.1%</td><td>3.5%</td></tr><tr><td>1 (Extreme)</td><td>36.6%</td><td>25.0%</td><td>15.6%</td><td>10.8%</td><td>8.0%</td><td>6.2%</td></tr></table>

## 3.4 Properties

Range and Interpretation. Bounded AUKS values range from -1 to 1. A value of 1 indicates perfect separation within the operational window. A value of 0 implies random performance $( \mathrm { T P R } = \mathrm { F P R } )$ , and negative values indicate systematic error where the model assigns higher scores to normal samples than to anomalies.

Compactness and Ideal Separation. While AUROC reaches 1.0 for any perfectly separated ranking regardless of distribution spread, AUKS and Bounded AUKS impose stricter geometrical constraints (see Figure 1). $\mathbf { A U K S @ 1 0 0 \% = 1 . 0 \ ( F i g . \ 1 ( c ) ) }$ requires extreme polarization: all normal-scores must be exactly 0 and all anomalous-scores exactly 1, which is often theoretically unattainable. In contrast, Bounded AUKS = 1.0 (Fig. 1(d)) is asymptotically achievable under a specific condition: the normal-score distribution must collapse to a single constant � (minimizing the intra-class variance), and all anomalousscores must exceed �. Since we utilize upper and lower bounds to estimate the threshold, the metric inherently incorporates threshold selection information. In (d), our estimated threshold corresponds to � (or is slightly larger than �).

It is worth noting that achieving a perfect Bounded AUKS value only requires anomalous samples to have scores exceeding �, which is consistent with the nature of anomalies, namely their diversity and varying levels of detection dificulty. Therefore, it’s unrealistic to expect all the scores in the anomalous samples to be concentrated at the perfect score of 1. Similarly, suppressing the scores of all normal samples to 0 is also extremely dificult. Instead, we believe that normal patterns should be compactly clustered and a high Bounded AUKS value only requires the normal-scores to be tightly concentrated.

## 4 Results

In this section, we present empirical results on two industrial anomaly detection benchmarks and compare Bounded AUKS with widely used evaluation metrics. After introducing the experimental settings in Section 4.1, we present quantitative comparisons, discuss representative case studies, and summarize the main findings.

## 4.1 Experimental Settings

We conduct experiments on two standard industrial benchmarks: MVTec AD and VisA. MVTec AD covers 15 categories (5 textures and 10 objects) with diverse anomaly types on high‑resolution images. Detailed per‑category statistics are provided in Supplementary Material Section S4, Table S4. VisA includes 12 object categories, over 9,000 images, and 19 anomaly types under real‑world manufacturing conditions. We follow the default VisA one‑class split used in our implementation, with per‑category counts summarized in Supplementary Material Section S4, Table S5.

The comparison covers representative unsupervised anomaly detection models: PaDiM (Defard et al. 2021), PatchCore (Roth et al. 2022), STFPM (Wang et al. 2021), Reverse Distillation (Deng & Li 2022), FastFlow (Yu et al. 2021), EficientAD-S/M (Batzner et al. 2024), SuperSimpleNet (Rolih et al. 2025), UniNet (Wei et al. 2025), and Dinomaly (Guo et al. 2025). These models span density estimation, memory-bank retrieval, student–teacher distillation, normalizing flows, and reconstruction-based paradigms. A concise overview of the evaluated counterparts is given in Supplementary Material Section S5, Table S6. Implementations use Anomalib defaults (Akcay et al. 2022). For each model and category, we compute pixel-level AUROC, AUPRO@5%, AUPRO@30%, AUKS@100%, and Bounded AUKS with � = 2 and $q = 0 . 0 1$

Anomaly Map GT + Pred Mask  
Anomaly Map GT + Pred Mask  
![](images/f0f02c97f1bd151c7817d279d5ea6c7b1ca79c59eae0ed11308e9044bfb4546e.jpg)  
(a) Cashew(VisA)

![](images/70e4e7313bd5a00f3e51f8440ff07251445cf3605b051db5ec7b2f79cf77db38.jpg)  
(b) Model: ReverseDistillation AUROC: 0.9514 Bounded AUKS: 0.7940

![](images/3ae07d6e29b447f242b568615fd0be26455945658614706bb5ee7882ae2b6405.jpg)  
(c) Model: EfficientAD-S AUROC: 0.9536 Bounded AUKS: 0.7740

Anomaly Map GT + Pred Mask  
![](images/8a213a1d14d15604c4f81cf77dc22d0e903e2476beba0d39a855c2cd7b011276.jpg)  
(d) Model: Fastflow AUROC: 0.9525 Bounded AUKS: 0.7014  
Figure 3: Case comparison on VisA (Cashew). (a) test images and ground truth masks. (b)–(d) Anomaly heatmaps and predicted mask segmentations at the F1-max optimal point overlaid on the ground-truth mask for (b) Reverse Distillation, (c) EficientAD-S, and (d) FastFlow.

## 4.2 Cases Comparison Results

Figure 3 shows a representative VisA Cashew case. AUROC ranks EficientAD-S above FastFlow and Reverse Distillation with only marginal diferences: 0.9536 for EficientAD-S, 0.9525 for FastFlow, and 0.9514 for Reverse Distillation. Visual masks show the opposite ordering. Reverse Distillation produces the most concentrated heatmaps over the anomaly and the cleanest segmentation. EficientAD-S shows occasional over- and undersegmentation, while FastFlow incorrectly flags large normal background regions as anomalous. Bounded AUKS produces the visually consistent order with scores 0.7940, 0.7740, and 0.7014 for Reverse Distillation, EficientAD-S, and FastFlow.

(c) UniNet  
![](images/7cfd577ddfe6f9d9e40bb7c3ffca8ea293d5386a82327087bb01eaf33c09a8d2.jpg)  
(a) Original  
(b) Dinomaly  
Figure 4: Comparison between Dinomaly and UniNet on the grid category of MVTec AD dataset. (a) Original RGB test images and ground truth masks. (b) (c) Anomaly maps and predicted mask segmentations overlaid on the ground truth mask for (b) Dinomaly and (c) UniNet. (d) Corresponding ROC, PRO, PDF, CDF, and KS curves for Dinomaly (left) and UniNet (right). The segmentation threshold for each model is set at the F1-max optimal point.

Figure 4 compares Dinomaly and UniNet on the MVTec AD grid category. Dinomaly produces cleaner anomaly maps and more accurate masks by suppressing background responses while preserving true anomalies, yet their AUROC values are nearly identical (0.9942 versus 0.9934), and both AUPRO@5% and AUPRO@30% incorrectly favor UniNet. Bounded AUKS instead assigns 0.9409 to Dinomaly and 0.9091 to UniNet, consistent with the visual evidence. The right-side PDF, CDF, and KS curves explain this result: Dinomaly achieves greater separation between normal- and anomalous-score distributions, whereas UniNet shows more overlap in the operative threshold region. Thus, Bounded AUKS evaluates whether score distributions support reliable localization rather than ranking alone.

The visual comparison of the threshold bounds used by Bounded AUKS and AUPRO is provided in Supplementary Material Section S6.

## 4.3 Metrics Performance Summary

Table 3 provides the summary of model rankings and mean scores across all categories on MVTec AD and VisA. Each entry is reported as average rank followed by the mean score in parentheses. Figure 5 complements this quantitative result with per-category scatter plots, so that both dataset-level averages and cross-category variations can be jointly analyzed.

We summarize the main findings as follows. First, AUROC is near ceiling for many methods and the per-category scatters cluster tightly with little diference. Second, restricting the FPR range to diferent limits produces diferent evaluation results. Comparing the AUPRO operations at range [0, 0.05] and [0, 0.30] reveals that imposing a tighter restriction on the FPR thresholds results in lower AUPRO values across all models, demonstrating that the metric is highly dependent on the chosen FPR thresholds. Third, as shown in Table 3, the model rankings produced by AUKS@100% exhibit significant discrepancies compared to other metrics, and its evaluation does not align well with actual qualitative performance. The primary reason is that AUKS@100% integrates over the entire threshold range, incorporating many undesirable intervals i.e., regions where both TPR and FPR are simultaneously high, or simultaneously low. These regions are largely uninformative and can therefore distort the evaluation, resulting in biased performance estimates.

In particular, Bounded AUKS confines the integration to a strictly bounded interval, which efectively functions as an decision-threshold selection mechanism. Meanwhile, our Bounded AUKS is a stricter metric for quantifying the separability between normal- and

Table 3: Overall average ranks and pixel-level metric values across all categories on MVTec AD and VisA.
<table><tr><td colspan="6">MVTec AD</td></tr><tr><td>Model</td><td>AUROC</td><td></td><td>AUPRO@5% AUPRO@30% AUKS@100%</td><td></td><td>Bounded AUKS</td></tr><tr><td>Dinomaly</td><td>1 (0.9845)</td><td>1 (0.7924)</td><td>1 (0.9535)</td><td>3 (0.4059)</td><td>1 (0.8329)</td></tr><tr><td>PatchCore</td><td>2 (0.9801)</td><td>6 (0.6885)</td><td>3 (0.9298)</td><td>6 (0.3525)</td><td>2 (0.7964)</td></tr><tr><td>Reverse Distillation</td><td>3 (0.9739)</td><td>3 (0.7152)</td><td>2 (0.9304)</td><td>4 (0.3828)</td><td>5 (0.7688)</td></tr><tr><td>FastFlow</td><td>4 (0.9738)</td><td>5 (0.6941)</td><td>4 (0.9138)</td><td>2 (0.4120)</td><td>7 (0.7574)</td></tr><tr><td>UniNet</td><td>5 (0.9721)</td><td>7 (0.6812)</td><td>6 (0.9022)</td><td>5 (0.3555)</td><td>4 (0.7693)</td></tr><tr><td>SuperSimpleNet</td><td>6 (0.9645)</td><td>9 (0.6062)</td><td>9 (0.8670)</td><td>1 (0.4426)</td><td>6 (0.7673)</td></tr><tr><td>PaDiM</td><td>7 (0.9632)</td><td>10 (0.5816)</td><td>7 (0.8974)</td><td>7 (0.2342)</td><td>9 (0.6371)</td></tr><tr><td>EfficientAD-M</td><td>8 (0.9579)</td><td>2 (0.7392)</td><td>5 (0.9027)</td><td>8 (0.1811)</td><td>3 (0.7708)</td></tr><tr><td>EfficientAD-S</td><td>9 (0.9497)</td><td>4 (0.7118)</td><td>8 (0.8790)</td><td>10 (0.1478)</td><td>8 (0.7465)</td></tr><tr><td>STFPM</td><td>10 (0.9473)</td><td>8 (0.6224)</td><td>10 (0.8647)</td><td>9 (0.1549)</td><td>10 (0.6345)</td></tr></table>

<table><tr><td>Model</td><td></td><td></td><td></td><td></td><td>AUROC AUPRO@5% AUPRO@30% AUKS@100% Bounded AUKS</td></tr><tr><td>Dinomaly</td><td>1 (0.9885)</td><td>1 (0.8098)</td><td>1 (0.9543)</td><td>4 (0.3575)</td><td>1 (0.8718)</td></tr><tr><td>Reverse Distillation</td><td>2 (0.9854)</td><td>2 (0.7034)</td><td>2 (0.9112)</td><td>3 (0.3735)</td><td>2 (0.8636)</td></tr><tr><td>EfficientAD-M</td><td>3 (0.9834)</td><td>3 (0.6484)</td><td>3 (0.8828)</td><td>8 (0.1729)</td><td>3 (0.8420)</td></tr><tr><td>PatchCore</td><td>4 (0.9804)</td><td>8 (0.5708)</td><td>4 (0.8616)</td><td>6 (0.3177)</td><td>4 (0.8036)</td></tr><tr><td>EfficientAD-S</td><td>5 (0.9726)</td><td>7 (0.5841)</td><td>6 (0.8240)</td><td>9 (0.1363)</td><td>6 (0.7925)</td></tr><tr><td>FastFlow</td><td>6 (0.9705)</td><td>6 (0.5890)</td><td>5 (0.8452)</td><td>2 (0.3801)</td><td>8 (0.7820)</td></tr><tr><td>UniNet</td><td>7 (0.9652)</td><td>5 (0.5896)</td><td>7 (0.8221)</td><td>5 (0.3352)</td><td>7 (0.7873)</td></tr><tr><td>PaDiM</td><td>8 (0.9618)</td><td>10 (0.3916)</td><td>9 (0.7509)</td><td>7 (0.2100)</td><td>9 (0.6603)</td></tr><tr><td>SuperSimpleNet</td><td>9 (0.9604)</td><td>4 (0.6092)</td><td>8 (0.8149)</td><td>1 (0.4018)</td><td>5 (0.8004)</td></tr><tr><td>STFPM</td><td>10 (0.8843)</td><td>9 (0.4387)</td><td>10 (0.6652)</td><td>10 (0.1346)</td><td>10 (0.5433)</td></tr></table>

anomalous-score distributions; its values span a wider range, distinctly highlighting the performance gaps between these models and thereby ranking the indeed better detection models.

![](images/7020144bb99b3011d9c1b6cd18d96dfa86e2c9805b89017103f5d8fc951d328d.jpg)

![](images/2cf8500ebb268694ad652a14055ed0da55e1d5884a4e2986e3a27d9f083a0c1c.jpg)

![](images/faa41025e81250f47ab4bcf5a1faa80cec522e2232489c094c049778d7a66d9b.jpg)

![](images/10ff33f5ba1b3e293bd9b2ee50d8caeea7a2931e64a0824ad2b2af971fe8a400.jpg)  
(d) Bounded AUKS (q=0.01, α=2)  
Figure 5: Pixel-level metric scatter summaries across categories on MVTec AD and VisA. (a) AUROC. (b) AUPRO@5% vs AUPRO@30%. (c) AUKS@100%. (d) Bounded $\mathrm { A U K S } ( q = 0 . 0 1 , \alpha = 2 )$ . Points denote per-category scores and mean markers lie on each model baseline; marker shape distinguishes datasets.

## 5 Conclusion

Bounded AUKS evaluates anomaly localization through signed score-distribution separability over an anomaly-free threshold range. Unlike AUROC, fixed FPR windows, and posthoc optimal thresholds, it directly assesses robust thresholding. Experiments on MVTec AD and VisA show that it reveals diferences hidden by saturated metrics and explains qualitative localization behavior. The same principle can extend to other scored anomalydetection settings and potentially serve as a training objective.

In application, Bounded AUKS supports model selection, threshold-range design, and deployment auditing. For detectors with similar conventional scores, it favors models with persistent score separation across a usable interval. Normal-score-derived bounds can be updated without anomalous samples, while � links the interval to a nominal false-alarm tolerance. The metric thus connects ofline comparison to practical threshold decisions.

## Data Availability Statement

The MVTec AD dataset is available from https://www.mvtec.com/research-teaching/datasets/mvtec-ad; the VisA dataset is available through https://github.com/amazon-science/spot-diff. The dataset providers’ terms of use apply.

## Supplementary Material

The Supplementary Material provides supporting analyses and additional experimental details.

## References

Akcay, S., Ameln, D., Vaidya, A., Lakshmanan, B., Ahuja, N. & Genc, U. (2022), Anomalib: A deep learning library for anomaly detection, in ‘2022 IEEE International Conference on Image Processing (ICIP)’, pp. 1706–1710.

Austin, E., Eckley, I. A. & Bardwell, L. (2024), ‘Detection of emergent anomalous structure in functional data’, Technometrics 66(4), 614–624.

Baitieva, A., Bouaouni, Y., Briot, A., Ameln, D., Khalfaoui, S. & Akcay, S. (2025), Beyond academic benchmarks: Critical analysis and best practices for visual industrial anomaly detection, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops’, pp. 4054–4064.

Batzner, K., Heckler, L. & König, R. (2024), Eficientad: Accurate visual anomaly detection at millisecondlevel latencies, in ‘Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)’, pp. 128–138.

Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D. & Steger, C. (2021), ‘The mvtec anomaly detection

dataset: A comprehensive real-world dataset for unsupervised anomaly detection’, International Journal of Computer Vision 129, 1038–1059.

Bergmann, P., Fauser, M., Sattlegger, D. & Steger, C. (2019), Mvtec ad – a comprehensive real-world dataset for unsupervised anomaly detection, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)’, pp. 9592–9600.

Bertoldo, J. P. C., Ameln, D., Vaidya, A. & Akcay, S. (2024), Aupimo: Redefining anomaly localization benchmarks with high speed and low tolerance, in ‘35th British Machine Vision Conference 2024 (BMVC 2024)’, BMVA.

Chen, Q., Luo, H., Lv, C. & Zhang, Z. (2025), A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization, in A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler & G. Varol, eds, ‘Computer Vision – ECCV 2024’, Vol. 15125 of Lecture Notes in Computer Science, Springer, pp. 37–54.

Chen, Z., Xie, X., Yang, L. & Lai, J.-H. (2025), ‘Hard-normal example-aware template mutual matching for industrial anomaly detection’, International Journal of Computer Vision 133(5), 2927–2949.

Coles, S. (2001), An Introduction to Statistical Modeling of Extreme Values, Springer Series in Statistics, Springer.

Davis, J. & Goadrich, M. (2006), The relationship between precision-recall and roc curves, in ‘Proceedings of the 23rd international conference on Machine learning’, ACM, pp. 233–240.

Defard, T., Setkov, A., Loesch, A. & Audigier, R. (2021), Padim: A patch distribution modeling framework for anomaly detection and localization, in A. Del Bimbo, R. Cucchiara, S. Sclarof, G. M. Farinella, T. Mei, M. Bertini, H. J. Escalante & R. Vezzani, eds, ‘Pattern Recognition. ICPR International Workshops and Challenges’, Vol. 12664 of Lecture Notes in Computer Science, Springer, pp. 475–489.

Deng, H. & Li, X. (2022), Anomaly detection via reverse distillation from one-class embedding, in ‘2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)’, pp. 9727–9736.

Du, J., Tao, C., Cao, X. & Tsung, F. (2025), ‘3D vision-based anomaly detection in manufacturing: A survey’, Frontiers of Engineering Management 12(2), 343–360.

Fawcett, T. (2006), ‘An introduction to ROC analysis’, Pattern Recognition Letters 27(8), 861–874.

Fourure, D., Javaid, M. U., Posocco, N. & Tihon, S. (2021), Anomaly detection: How to artificially increase your f1-score with a biased evaluation protocol, in ‘Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track’.

Frank J. Massey, J. (1951), ‘The kolmogorov-smirnov test for goodness of fit’, Journal of the American Statistical Association 46(253), 68–78.

Guo, J., Lu, S., Zhang, W., Chen, F., Li, H. & Liao, H. (2025), Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)’, pp. 20405–20415.

Jiang, M., Han, S. & Huang, H. (2023), Anomaly detection with score distribution discrimination, in ‘Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23)’, Association for Computing Machinery, pp. 984–996.

Liu, X., Wang, J., Leng, B. & Zhang, S. (2025), ‘Unlocking the potential of reverse distillation for anomaly detection’, Proceedings of the AAAI Conference on Artificial Intelligence 39(6), 5640–5648.

Lunardi, W. T., Banabila, A., Herzalla, D. & Andreoni, M. (2025), ‘Contrastive representation modeling for anomaly detection’, arXiv preprint 2501.05130.

Ma, W., Yao, Q., Zhang, X., Huang, Z., Jiang, Z. & Zhou, S. K. (2025), Towards accurate unified anomaly segmentation, in ‘2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)’, pp. 1342–1352.

Maurer, A. & Pontil, M. (2020), ‘Estimating weighted areas under the roc curve’, Advances in Neural Information Processing Systems 33, 7733–7742.

Metzner, C., Schilling, A., Traxdorf, M., Tziridis, K., Schulze, H. & Krauss, P. (2022), ‘Classification at the accuracy limit: facing the problem of data ambiguity’, Scientific Reports 12, 22121.

Panaretos, V. M. & Zemel, Y. (2019), ‘Statistical aspects of wasserstein distances’, Annual Review of Statistics and Its Application 6(1), 405–431.

Rafiei, M., Breckon, T. P. & Iosifidis, A. (2023), ‘On pixel-level performance assessment in anomaly detection’, arXiv preprint 2310.16435.

Rolih, B., Fučka, M. & Skočaj, D. (2025), SuperSimpleNet: Unifying unsupervised and supervised learning

for fast and reliable surface defect detection, in A. Antonacopoulos, S. Chaudhuri, R. Chellappa, C.-L.   
Liu, S. Bhattacharya & U. Pal, eds, ‘Pattern Recognition’, Springer Nature Switzerland, pp. 47–65.

Roth, K., Pemula, L., Zepeda, J., Schlölkopf, B., Brox, T. & Gehler, P. (2022), Towards total recal in industrial anomaly detection, in ‘2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)’, IEEE, pp. 14298–14308.

Tao, C. & Du, J. (2025), ‘PointSGRADE: Sparse learning with graph representation for anomaly detection by using unstructured 3D point cloud data’, IISE Transactions 57(2), 131–144.

Vallender, S. S. (1974), ‘Calculation of the wasserstein distance between probability distributions on the line’, Theory of Probability & Its Applications 18(4), 784–786.

Wang, G., Han, S., Ding, E. & Huang, D. (2021), Student-teacher feature pyramid matching for anomaly detection, in ‘Proceedings of the British Machine Vision Conference (BMVC)’.

Wei, S., Jiang, J. & Xu, X. (2025), Uninet: A contrastive learning-guided unified framework with feature selection for anomaly detection, in ‘Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)’, pp. 9994–10003.

Wilkinghof, K. & Imoto, K. (2024), F1-ev score: Measuring the likelihood of estimating a good decision threshold for semi-supervised anomaly detection, in ‘ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)’, pp. 256–260.

Yan, H., Paynabar, K. & Shi, J. (2017), ‘Anomaly detection in images with smooth background via smoothsparse decomposition’, Technometrics 59(1), 102–114.

Yao, H., Liu, M., Yin, Z., Yan, Z., Hong, X. & Zuo, W. (2025), Glad: Towards better reconstruction with global and local adaptive difusion models for unsupervised anomaly detection, in A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler & G. Varol, eds, ‘Computer Vision – ECCV 2024’, Vol. 15129 of Lecture Notes in Computer Science, Springer.

Yu, J., Zheng, Y., Wang, X., Li, W., Wu, Y., Zhao, R. & Wu, L. (2021), ‘Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows’, arXiv preprint 2111.07677.

Zhang, X., Li, S., Li, X., Huang, P., Shan, J. & Chen, T. (2023), Destseg: Segmentation guided denoising student-teacher for anomaly detection, in ‘2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)’, IEEE, pp. 3914–3923.

Zou, Y., Jeong, J., Pemula, L., Zhang, D. & Dabeer, O. (2022), Spot-the-diference self-supervised pretraining for anomaly detection and segmentation, in S. Avidan, G. Brostow, M. Cissé, G. M. Farinella & T. Hassner, eds, ‘Computer Vision – ECCV 2022’, Springer Nature Switzerland, pp. 392–408.