# Calibrating Multimodal Test-Time Adaptation under Reliability and Availability Modality Shifts

Jiangmeng Li, Jinglin Xu, Yi Li, Yumeng Liu, Jiahuan Zhou, Fanjiang Xu, Hui Xiong, Fellow, IEEE, Gang Hua, Fellow, IEEE

Abstract—Multimodal test-time adaptation (TTA) enables pretrained multimodal models to accommodate distribution shifts during inference using only unlabeled test data, without recourse to the original training data. Compared with unimodal adaptation, multimodal adaptation is further complicated by the coexistence and interaction of multiple representation perspectives. This complexity is particularly pronounced under modality reliability shifts, which preserve modality availability but degrade information quality, and modality availability shifts, which render the corresponding modality entirely inaccessible. Category-conditional Gaussian modeling offers a natural starting point for handling such distributional variations, having proven effective in unimodal test-time adaptation for characterizing representation distributions, yet its extension to multimodal settings requires a more elaborate formulation. To this end, we propose Adaptive Probabilistic Gaussian Calibration (AdaPGC), which extends category-conditional Gaussian modeling across modality-specific and multimodal fusion representations for probabilistic calibration under multimodal distribution shifts. AdaPGC continuously updates category-conditional Gaussian models over modality-specific and fusion representations, using the resulting probabilistic predictions to calibrate source model predictions and guide reliability based rectification, while exploiting their joint distributions to recover unavailable fusion representations. Extensive experiments under diverse modality corruption and missingness conditions demonstrate that AdaPGC effectively improves robustness under both modality reliability and availability shifts. Our source code will be released at https://github.com/XuJinglinn/AdaPGC-plus.

Index Terms—Multimodal Learning, Test-Time Adaptation, Category-Conditional Gaussian Model.

## I. INTRODUCTION

ULTIMODAL models integrate complementary inforhave demonstrated considerable promise in tasks such as action recognition [4]–[8], sentiment analysis [9]–[13], and other complex perception tasks [14]–[20]. Conventional multimodal models are often vulnerable to distribution shifts, wherein discrepancies between the source and target domains can precipitate pronounced performance degradation [21]–[23]. In response to this limitation, multimodal test-time adaptation (TTA) [24]–[26] has emerged as a viable paradigm, which dynamically adapts multimodal models using only unlabeled target data, thus facilitating real-time robustness enhancement under distribution shifts.

Recent unimodal TTA methods [27]–[29] model categoryconditional distributions of test-time representations through Gaussian discriminant analysis (GDA) [30], [31] and have demonstrated consistent improvements under distribution shifts. However, extending such distribution modeling to multimodal TTA is considerably more challenging. Compared with unimodal models, multimodal models derive multiple modality-specific representations together with a multimodal fusion representation [32]–[34], making it insufficient to treat the distribution modeled in each interrelated but distinct perspective merely as an independent basis for prediction. Crucially, the coexistence of multiple representation perspectives renders test-time distribution shifts substantially more intricate in multimodal settings [35].

Specifically, as illustrated in Figure 1, test-time distribution shifts in multimodal settings are further complicated by modality reliability shifts and modality availability shifts. Under modality reliability shifts, the affected modality remains fully observable, while its information quality deteri orates due to noise, blur, environmental interference, or other corruptions. Such degradation can distort the corresponding modality-specific representation, subsequently perturbing the multimodal fusion representation and ultimately compromising prediction performance [36]. Modality availability shifts constitute a more disruptive form of test-time variation, occurring when sensor failures, transmission interruptions, or privacy restrictions render the corresponding modality entirely unavailable for certain test instances [37]–[40]. In this case, not only is the representation of the missing modality inaccessible, but the corresponding multimodal fusion representation can no longer be directly constructed, resulting in a more severe loss of complementary information. Addressing these two forms of distribution shift therefore requires a more nuanced exploitation of the information available across different representation perspectives.

To address these challenges, we exploit the distributional information revealed by different representation perspectives in the target stream to calibrate source-model predictions under varying modality distribution shifts. More concretely, modality-specific and multimodal fusion representations characterize category structure from different information sources, and their corresponding distributional predictions therefore vary with changes in modality reliability [41]. When the reliability of a modality degrades, the resulting probabilistic predictions from different perspectives become relatively inconsistent. Such discrepancies can serve as an indicator of which modality is more strongly affected and thereby support targeted prediction calibration. Meanwhile, completemodality test instances naturally provide simultaneously observed modality-specific and fusion representations, forming online paired observations in the target stream. These pairs provide a statistical basis for recovering the unavailable fusion representation from the observed modality. Such recovery can partially compensate for the multimodal predictive information lost when direct fusion becomes impossible due to modality missingness [42], [43].

![](images/6918a6d4302bfaed16fbb1da1e1175898ea456ecf64aec9accad65191ffa0093.jpg)  
Fig. 1. Illustration of modality reliability and availability shifts in multimodal TTA. Under reliability shifts, corrupted modalities distort the corresponding modality-specific representations and consequently affect the multimodal fusion representation. Under availability shifts, missing modalities make the corresponding modality-specific representations unavailable and prevent the fusion representation from being directly constructed, motivating its recovery from the remaining modality.

Building on these considerations, we propose Adaptive Probabilistic Gaussian Calibration (AdaPGC), <sup>1</sup> a probabilistic approach for multimodal TTA under varying modality reliability and availability. AdaPGC builds category-conditional Gaussian models for different representation perspectives and continuously updates them using statistics from the target stream. These models yield probabilistic predictions that provide distribution-aware calibration signals for the source model under different forms of modality shift. Under modality reliability shifts, probabilistic predictions from different perspectives are compared to assess relative modality reliability, based on which we employ a contrastive rectification approach to correct the representation of the less reliable modality. Under modality availability shifts, AdaPGC further extends the category-conditional Gaussian modeling to the joint distribution of modality-specific and multimodal fusion representations, with the model parameters learned from modalitycomplete test instances. This extension yields a closed-form conditional estimate of the unavailable fusion representation from the observed modality. This estimate is then incorporated into the probabilistic calibration of source model predictions. Overall, our contributions can be summarized as follows:

• We study multimodal TTA under both modality reliability and availability shifts, covering scenarios where modalities remain observable but vary in reliability, as well as scenarios where part of the multimodal input becomes unavailable. This setting allows us to systematically investigate robustness to both corrupted and incomplete multimodal observations.

• We propose AdaPGC, a probabilistic approach for multimodal TTA under both modality reliability and availability shifts. AdaPGC models both category-conditional distributions and the dependencies between modalityspecific and fusion representations, enabling a unified probabilistic treatment of fully observed, corrupted, and partially observed multimodal inputs.

• We conduct extensive experiments under diverse modality corruption and missingness conditions, demonstrating the effectiveness and robustness of AdaPGC under both modality reliability and availability shifts.

## II. RELATED WORK

## A. Multimodal Test-Time Adaptation

Test-time adaptation (TTA) adapts a pretrained model to distribution shifts using only unlabeled target data. Representative unimodal approaches include entropy minimization with TENT [45], informative sample selection with EATA [46] and SAR [47], and test-time classifier adjustment with T3A [48]. These studies establish the basic adaptation mechanisms in unimodal settings, while multimodal TTA further considers the complementary information and interactions among different modalities.

Building on this setting, existing multimodal TTA methods exploit cross-modal information through diverse adaptation strategies. MM-TTA [24] generates complementary pseudolabels across modalities to facilitate source-free adaptation, although its effectiveness depends on the quality of the generated pseudo-labels. READ [25] introduces a more robust approach by enhancing the impact of confident predictions and reducing the influence of noisy predictions. Meanwhile, ABPEM [49] improves multimodal TTA by mitigating modality misalignment under distribution shift through attention bootstrapping and principal entropy minimization. More recently, SuMi [26] employs smoothing and cross-modal information sharing to improve robustness under mixed shifts, including modality corruption and missingness. TSA [50], a recent method, further explores the problem of unimodal distribution shift through a selective adaptation framework that handles shifts in the affected modality. BriMPR [35] addresses the coupled effects of unimodal feature shifts and cross-modal semantic misalignment through progressive re-alignment.

Most existing methods are predicated on full modality observability and thus do not accommodate modality missing scenarios. In contrast, AdaPGC jointly models marginal and cross-representation distributions, enabling unified adaptation to modality reliability and availability shifts.

## B. Incomplete Multimodal Learning

Incomplete multimodal learning seeks to preserve predictive robustness under partially observed inputs, where variations in modality availability can otherwise result in substantial performance degradation. Existing studies have developed various mechanisms for learning with incomplete multimodal inputs. MAP [51] introduces modality-missingaware prompts to equip pretrained multimodal transformers for diverse missing-modality conditions. DCP [52] further models the correlations between prompts and input features, as well as those across different prompt layers, to better exploit crossmodal complementary information. More recently, MoRA [53] introduces modality-common and modality-specific low-rank parameters to facilitate bidirectional cross-modal knowledge transfer while preserving modality-specific flexibility.

A limited number of studies have investigated missing modalities under the more restrictive TTA setting. MiDl [39] combines mutual information minimization with self-distillation to adapt multimodal models using incomplete test inputs. SuMi [26] handles missing modalities by smoothing distribution shifts between complete and incomplete observations.

Despite this progress, existing methods primarily enhance predictions from the available modalities, without explicitly recovering the inaccessible fusion representation. AdaPGC addresses this limitation by modeling the category-conditional distributions of modality-specific and fusion representations, enabling the missing fusion representation to be inferred from the observed modality.

## C. Distribution Modeling for TTA

Distribution modeling provides a probabilistic basis for testtime adaptation by estimating the category structure of unlabeled target representations. In particular, category-conditional Gaussian modeling enables posterior inference through Gaussian discriminant analysis (GDA), using feature statistics accumulated during inference. DOTA [28] continuously estimates category-conditional distributions and derives posterior probabilities via Bayesian inference, mitigating the forgetting associated with limited sample caches. FreeTTA [29] employs online expectation-maximization to estimate target distributions, incorporating zero-shot predictions as prior information. BayesTTA [27] further considers temporally evolving distributions, combining incremental Gaussian estimation with adaptive covariance selection to calibrate predictions and guide representation adaptation. ADAPT [54] incorporates CLIP priors and a historical knowledge bank to correct potential likelihood biases, refining GDA-based inference without backpropagation. These studies demonstrate the value of target feature statistics for informing test-time predictions.

These approaches primarily model distributions in a single visual representation space. AdaPGC extends categoryconditional Gaussian modeling to modality-specific and fusion representations, using their marginal distributions for probabilistic calibration and their joint distributions for fusion representation recovery.

## III. PRELIMINARY

## A. Gaussian Discriminant Analysis

Gaussian Discriminant Analysis (GDA) is a classical generative classification model that predicts category labels based on category-conditional feature distributions [30], [55]. Consider a labeled sample set $\boldsymbol { \mathcal { D } } = \{ ( z _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where $z _ { i } \in \mathbb { R } ^ { d }$ is a d-dimensional feature vector and $y _ { i } \in \{ 1 , 2 , \ldots , C \}$ is its category label, with N and C denoting the number of samples and categories, respectively. GDA assumes that the feature distribution conditioned on category c is Gaussian:

$$
\begin{array} { l } { p ( z _ { i } \mid c ) = \mathcal { N } ( z _ { i } ; \pmb { \mu } _ { c } , \pmb { \Sigma } _ { c } ) } \\ { \quad = \frac { \exp \left( - \frac { 1 } { 2 } ( z _ { i } - \pmb { \mu } _ { c } ) ^ { \top } \pmb { \Sigma } _ { c } ^ { - 1 } ( z _ { i } - \pmb { \mu } _ { c } ) \right) } { \sqrt { ( 2 \pi ) ^ { d } \left| \pmb { \Sigma } _ { c } \right| } } , } \end{array}\tag{1}
$$

where $\mu _ { c } \in \mathbb { R } ^ { d }$ and $\pmb { \Sigma } _ { c } \in \mathbb { R } ^ { d \times d }$ are the mean vector and covariance matrix of category c, respectively.

According to Bayes’ theorem, the posterior probability that $z _ { i }$ belongs to category c is

$$
p ( c \mid z _ { i } ) = { \frac { \pi _ { c } p ( z _ { i } \mid c ) } { \sum _ { c ^ { \prime } = 1 } ^ { C } \pi _ { c ^ { \prime } } p ( z _ { i } \mid c ^ { \prime } ) } } ,\tag{2}
$$

where $\pi _ { c } = p ( c )$ denotes the prior probability of category c.

Taking the logarithm of the posterior probability above and omitting terms shared across all categories yields the following log-posterior for category c:

$$
\begin{array} { l } { { \displaystyle g _ { c } ( z _ { i } ) = - \frac { 1 } { 2 } ( z _ { i } - \pmb { \mu } _ { c } ) ^ { \top } \pmb { \Sigma } _ { c } ^ { - 1 } ( z _ { i } - \pmb { \mu } _ { c } ) } } \\ { { \displaystyle ~ - \frac { 1 } { 2 } \log | \pmb { \Sigma } _ { c } | + \log \pi _ { c } . } } \end{array}\tag{3}
$$

The posterior probabilities can be recovered by applying the softmax function to the log-posteriors.

## B. Conditional Gaussian Estimation

Let $z _ { X }$ and $z _ { Y }$ denote two statistically correlated representation vectors. Suppose their statistical dependence is characterized by the following joint Gaussian distribution:

$$
\left[ z _ { X } \right] \sim \mathcal { N } \left( \left[ \pmb { \mu } _ { X } \right] , \left[ \pmb { \Sigma } _ { X } \quad \pmb { \Sigma } _ { X Y } \right] \right) ,\tag{4}
$$

where $\mu _ { X }$ and $\pmb { \mu } _ { Y }$ are the corresponding mean vectors, $\Sigma _ { X }$ and $\Sigma _ { Y }$ denote their covariance matrices, and $\pmb { \Sigma } _ { Y X } = \pmb { \Sigma } _ { X Y } ^ { \top }$ denotes the cross-covariance between the two representations.

A fundamental property of the joint Gaussian distribution is that the conditional distribution of one representation given the other remains Gaussian [56]. Specifically,

$$
z _ { Y } \mid z _ { X } \sim \mathcal { N } \left( \pmb { \mu } _ { Y \mid X } , \pmb { \Sigma } _ { Y \mid X } \right) ,\tag{5}
$$

with conditional mean and covariance

$$
{ \pmb \mu } _ { Y | X } = { \pmb \mu } _ { Y } + \pmb { \Sigma } _ { Y X } \pmb { \Sigma } _ { X } ^ { - 1 } \left( z _ { X } - { \pmb \mu } _ { X } \right) ,\tag{6}
$$

$$
\pmb { \Sigma } _ { Y | X } = \pmb { \Sigma } _ { Y } - \pmb { \Sigma } _ { Y X } \pmb { \Sigma } _ { X } ^ { - 1 } \pmb { \Sigma } _ { X Y } .\tag{7}
$$

The conditional mean gives the expected representation of ${ z } _ { Y }$ for an observed $z _ { X }$ . It can therefore be used as a point estimate of the unobserved representation [55], [57]. We therefore define the conditional Gaussian estimation function as

$$
\begin{array} { r l } & { \mathcal { E } _ { X  Y } ( z _ { X } ) = \operatorname { \mathbb { E } } [ z _ { Y } \mid z _ { X } ] } \\ & { \quad \quad = \mu _ { Y } + \Sigma _ { Y X } \Sigma _ { X } ^ { - 1 } ( z _ { X } - \mu _ { X } ) . } \end{array}\tag{8}
$$

The unobserved representation can thus be estimated as ${ \hat { z } } _ { Y \mid X } = { \mathcal { E } } _ { X  Y } ( z _ { X } )$ , using the statistical dependence captured by the cross-covariance.

## IV. METHOD

This section presents the proposed AdaPGC framework. We first formulate multimodal test-time adaptation under modality distribution shifts, covering modality degradation and modality missingness. We then introduce online categoryconditional Gaussian modeling for modality-specific and fusion representations, which provides the probabilistic foundation for subsequent adaptation. Based on these Gaussian distributions, we develop Gaussian probabilistic calibration to refine source-model predictions, where the joint distributions between modality-specific and fusion representations are further exploited to recover unavailable fusion representations under modality missingness. Finally, for samples affected by single-modality degradation, we introduce adaptive contrastive rectification to identify the relatively degraded modality and selectively rectify its representation while preserving the reliable modality. The overall framework is illustrated in Figure 2.

## A. Problem Formulation

Without loss of generality, we consider a multimodal classification problem with two modalities, denoted by $m _ { 1 }$ and $m _ { 2 }$ At test time, the unlabeled target data arrive sequentially as a stream of mini-batches, ${ \mathcal { D } } _ { \mathrm { t a r g e t } } = \{ { \mathcal { D } } ^ { t } \} _ { t = 1 } ^ { T }$ , where t indexes the incoming batch, T denotes the total number of test batches, and each batch contains $\boldsymbol { B }$ samples. Neither the source-domain data nor the target labels are accessible, and the model is updated sequentially without resetting its parameters between batches. Depending on modality availability, each batch is partitioned into three disjoint subsets:

$$
\begin{array} { r } { \mathcal { D } ^ { t } = \mathcal { D } _ { \mathrm { f u l l } } ^ { t } \cup \mathcal { D } _ { m _ { 1 } } ^ { t } \cup \mathcal { D } _ { m _ { 2 } } ^ { t } , } \\ { \left| \mathcal { D } _ { \mathrm { f u l l } } ^ { t } \right| + \left| \mathcal { D } _ { m _ { 1 } } ^ { t } \right| + \left| \mathcal { D } _ { m _ { 2 } } ^ { t } \right| = \mathcal { B } . } \end{array}\tag{9}
$$

Here, $\mathcal { D } _ { \mathrm { f u l l } } ^ { t }$ contains samples for which both modalities $m _ { 1 }$ and $m _ { 2 }$ are available, i.e., paired observations $( \pmb { x } _ { m _ { 1 } , i } ^ { t } , \pmb { x } _ { m _ { 2 } , i } ^ { t } )$ In contrast, $\mathcal { D } _ { m _ { 1 } } ^ { t }$ contains samples for which only modality $m _ { 1 }$ is available, while $\mathcal { D } _ { m _ { 2 } } ^ { t }$ contains samples for which only modality $m _ { 2 }$ is available.

Let $\mathcal { M } _ { \theta } = ( \phi _ { 1 } , \phi _ { 2 } , \mathcal { F } , s )$ denote the pretrained multimodal model, where $\phi _ { 1 }$ and $\phi _ { 2 }$ are the modality encoders, $\mathcal { F }$ denotes the fusion layers, and s is the shared final linear classifier. The source model supports both unimodal and multimodal inference. For a modality-complete sample, the model produces

$$
z _ { m _ { 1 } , i } ^ { t } = \mathcal { F } \big ( \phi _ { 1 } ( \pmb { x } _ { m _ { 1 } , i } ^ { t } ) \big ) ,\tag{10}
$$

$$
z _ { m _ { 2 } , i } ^ { t } = \mathcal { F } \big ( \phi _ { 2 } ( \pmb { x } _ { m _ { 2 } , i } ^ { t } ) \big ) ,\tag{11}
$$

$$
z _ { F , i } ^ { t } = \mathcal { F } \big ( \mathrm { C o n c a t } \big [ \phi _ { 1 } ( \boldsymbol { x } _ { m _ { 1 } , i } ^ { t } ) , \phi _ { 2 } ( \boldsymbol { x } _ { m _ { 2 } , i } ^ { t } ) \big ] \big ) .\tag{12}
$$

Here, $z _ { m _ { 1 } , i } ^ { t }$ and $z _ { m _ { 2 } , i } ^ { t }$ are the unimodal representations, while $z _ { F , i } ^ { t }$ is the fused representation. For a modality-incomplete sample, only the unimodal representation corresponding to the available modality can be obtained. Specifically, a sample in $\mathcal { D } _ { m _ { 1 } } ^ { t }$ produces $z _ { m _ { 1 } , i } ^ { t } ,$ while a sample in $\mathcal { D } _ { m _ { 2 } } ^ { t }$ produces $z _ { m _ { 2 } , i } ^ { t } .$ In either case, the fused representation $z _ { F , i } ^ { t }$ cannot be directly obtained because one modality is unavailable.

Since the three representations are mapped to category logits by the same linear classifier s, we regard them as three perspectives of the same sample in a shared representation space. To describe the classification of different representations in a unified manner, let $m \in \{ m _ { 1 } , m _ { 2 } , F \}$ index a representation perspective. For any available representation $z _ { m , i } ^ { t } ,$ the logit of category c is given by

$$
s _ { c } \big ( z _ { m , i } ^ { t } \big ) = { \pmb w } _ { c } ^ { \top } z _ { m , i } ^ { t } + b _ { c } , \qquad c \in \{ 1 , \ldots , C \} ,\tag{13}
$$

where ${ \pmb w } _ { c }$ and $b _ { c }$ are the classifier weight and bias for category $c ,$ respectively. Collecting the category-wise logits into $\begin{array} { r } { { \pmb { s } } ( z ) = } \end{array}$ $[ s _ { 1 } ( z ) , \ldots , s _ { C } ( z ) ] ^ { \top }$ , the source logits for each target sample are determined by its modality availability as

$$
\pmb { s } _ { i } ^ { t } = \left\{ \begin{array} { l l } { \pmb { s } \left( z _ { F , i } ^ { t } \right) , } & { \pmb { x } _ { i } ^ { t } \in \mathcal { D } _ { \mathrm { f u l l } } ^ { t } , } \\ { \pmb { s } \left( z _ { m _ { 1 } , i } ^ { t } \right) , } & { \pmb { x } _ { i } ^ { t } \in \mathcal { D } _ { m _ { 1 } } ^ { t } , } \\ { \pmb { s } \left( z _ { m _ { 2 } , i } ^ { t } \right) , } & { \pmb { x } _ { i } ^ { t } \in \mathcal { D } _ { m _ { 2 } } ^ { t } . } \end{array} \right.\tag{14}
$$

The corresponding prediction is $\hat { y } _ { i } ^ { t } = \arg \operatorname* { m a x } _ { c \in \{ 1 , \dots , C \} } s _ { c , i } ^ { t } .$ Thus, the objective is to continuously adapt the pretrained model to the incoming target stream while maintaining accurate predictions under diverse modality shifts, including sample-wise modality missingness.

## B. Test-Time Gaussian Modeling

The source model provides discriminative predictions through learned decision boundaries, but such predictions alone do not explicitly capture the underlying feature distributions under target shifts. To obtain distribution-aware evidence for test-time adaptation, we establish a category-conditional Gaussian modeling framework over target representations.

![](images/850f832557a7c99160d639b60b0f915e879974159b161fd4a5540150255d9635.jpg)  
Fig. 2. Overview of the proposed multimodal test-time adaptation framework. The input consists of paired video and audio streams affected by reliability shifts (modality corruption) or availability shifts (missing video/audio). For complete instances, the pretrained source model feeds the fused representation $z _ { F }$ into the shared classifier to obtain source logits s; for incomplete instances, it instead uses the representation of the available modality, z or $z _ { A }$ . The category-conditional Gaussian means and priors are initialized from the pretrained classifier, while the covariance and cross-covariance matrices are initialized as identity matrices. During adaptation, these parameters are updated using target-stream statistics to define the discriminants $g _ { V } , g _ { A } .$ , and $g _ { F }$ and conditional estimators $\mathcal { E } _ { V  F }$ and $\mathcal { E } _ { A  F }$ . For incomplete instances, the available modality representation is used to estimate category-conditional fused representations whose Gaussian predictions are combined with the unimodal predictions to obtain calibration logits r. For complete instances, the fused Gaussian discriminant produces the calibration logits $r = g _ { F } ( z _ { F } )$ , while discrepancies between unimodal and fused Gaussian predictions guide reliability-aware one-sided alignment

Specifically, for each perspective $m \ \in \ \{ m _ { 1 } , m _ { 2 } , F \}$ and category c, we assume a category-conditional Gaussian distribution

$$
p ( z _ { m , i } ^ { t } \mid y _ { i } = c ) = \mathcal { N } \left( z _ { m } ^ { t } ; \pmb { \mu } _ { m , c } ^ { t } , \pmb { \Sigma } _ { m , c } ^ { t } \right) ,\tag{15}
$$

together with the corresponding category prior $\pi _ { m , c } ^ { t } = p ( c )$ where $\mu _ { m , c } ^ { t } , \Sigma _ { m , c } ^ { t }$ and $\pi _ { m , c } ^ { t }$ denote the category-specific mean vector, covariance matrix, and prior probability for perspective m, respectively, estimated from the target samples observed up to the t-th batch. Applying Bayes’ rule and omitting terms shared across categories gives the log-posterior score for each perspective:

$$
\begin{array} { r l } & { g _ { m , c } ^ { t } ( z _ { m , i } ^ { t } ) = - \displaystyle \frac { 1 } { 2 } ( z _ { m , i } ^ { t } - \pmb { \mu } _ { m , c } ^ { t } ) ^ { \top } ( \pmb { \Sigma } _ { m , c } ^ { t } ) ^ { - 1 } ( z _ { m , i } ^ { t } - \pmb { \mu } _ { m , c } ^ { t } ) } \\ & { ~ - \displaystyle \frac { 1 } { 2 } \log | \pmb { \Sigma } _ { m , c } ^ { t } | + \log \pi _ { m , c } ^ { t } . } \end{array}\tag{16}
$$

Collecting the scores over all categories gives $g _ { m } ^ { t } ( z _ { m , i } ^ { t } )$ resulting in three perspective-specific probabilistic predictions for $m _ { 1 } , m _ { 2 } .$ , and $F$ after softmax.

Moreover, to model the statistical dependency between different representations, we further assume that their joint distribution conditioned on category c follows a joint Gaussian distribution. For each pair consisting of a modality-specific representation $z _ { m , i } ^ { t } ,$ with m $\in \{ m _ { 1 } , m _ { 2 } \}$ , and the corresponding fusion representation $z _ { F , i } ^ { t }$ from a modality-complete sample, we specify their joint distribution as follows:

$$
\left[ \boldsymbol { z } _ { m , i } ^ { t } \right] \left[ \boldsymbol { c } \sim \mathcal { N } \left( \left[ \boldsymbol { \mu } _ { m , c } ^ { t } \right] , \left[ \boldsymbol { \Sigma } _ { m , c } ^ { t } \quad \boldsymbol { \Sigma } _ { F m , c } ^ { t } \right] \right) \right] c \cdots { }\tag{17}
$$

Among these parameters, the mean vectors $\mu _ { m , c } ^ { t }$ and $\mu _ { F , c } ^ { t } ,$ together with the covariance terms within each perspective, $\Sigma _ { m , c } ^ { t }$ and $\Sigma _ { F , c } ^ { t } ,$ are already obtained from the Gaussian modeling above. Therefore, only the cross-covariance matrix $\Sigma _ { F m , c } ^ { t }$ needs to be additionally estimated from modality-complete samples, reflecting the statistical relationship between the modality-specific and fusion representations for each category.

Under this joint Gaussian assumption, we define the category-specific fusion estimation function as the conditional expectation of the fusion representation given the observed

modality representation:

$$
\begin{array} { r l } & { \hat { z } _ { F | m , i , c } ^ { t } = \xi _ { m  F , c } ^ { t } ( z _ { m , i } ^ { t } ) } \\ & { \quad \quad \quad = \pmb { \mu } _ { F , c } ^ { t } + \pmb { \Sigma } _ { F m , c } ^ { t } ( \pmb { \Sigma } _ { m , c } ^ { t } ) ^ { - 1 } ( z _ { m , i } ^ { t } - \pmb { \mu } _ { m , c } ^ { t } ) . } \end{array}\tag{18}
$$

Here, $\hat { z } _ { F | m , i , c } ^ { t }$ is the conditional mean of the fusion representation given $z _ { m , } ^ { t }$ and category c.

Since the category of a test instance is unknown, we apply the estimation function for each category to obtain a set of category-conditioned fusion estimates, denoted by $\mathcal { E } _ { m  F } ^ { t } ( z _ { m , i } ^ { t } ) = \{ \hat { z } _ { F | m , i , c } ^ { t } \} _ { c = 1 } ^ { C }$

1) Gaussian Parameter Initialization: A fundamental challenge in online Gaussian modeling is how to obtain informative priors for the category-conditional parameters before sufficient target samples have been observed. Since source data are unavailable during test-time adaptation, we instead exploit the category knowledge already encoded in the pretrained classifier to initialize these parameters.

For a category-conditional Gaussian distribution, the mean naturally determines the center of each category in the representation space. Meanwhile, pretrained classifier weights have been shown to encode category prototype information [48]. This correspondence allows the pretrained classifier to provide an informative initialization for the category-conditional Gaussian parameters. We further adopt an isotropic covariance initialization, avoiding the introduction of additional categoryspecific structure before target statistics become available. For each $m \ \in \ \{ m _ { 1 } , m _ { 2 } , F \}$ and category c, we initialize the Gaussian mean and covariance as

$$
{ \pmb { \mu } } _ { m , c } ^ { 0 } = { \pmb w } _ { c } , \qquad { \pmb { \Sigma } } _ { m , c } ^ { 0 } = { \pmb I } .\tag{19}
$$

The category prior required by the Gaussian model can be naturally derived from the bias term of the pretrained classifier, which already reflects category-specific preference. Accordingly, we initialize the category prior by absorbing the classifier bias term into the category prior:

$$
\log \pi _ { m , c } ^ { 0 } = b _ { c } + \frac { 1 } { 2 } \| \pmb { w } _ { c } \| _ { 2 } ^ { 2 } .\tag{20}
$$

With the above initialization, the initial log-posterior is equivalent to the original classifier score, i.e., $g _ { m , c } ^ { 0 } ( z ) ~ =$ $s _ { c } ( z )$

For each modality $m \in \{ m _ { 1 } , m _ { 2 } \}$ , the category-conditional cross-covariance between its corresponding representation and the fusion representation is initialized as

$$
\begin{array} { r } { \Sigma _ { F m , c } ^ { 0 } = { \cal I } . } \end{array}\tag{21}
$$

Since the modality-specific and fusion representations are extracted from the same sample and lie in the same representation space, the identity matrix is adopted as the initial cross-covariance prior.

2) Incremental Gaussian Statistics Update: In multimodal TTA, target samples arrive sequentially, and their distribution often differs significantly from the source-domain distribution. Consequently, the initial parameters $\pi _ { m , c } ^ { 0 } , \mu _ { m , c } ^ { 0 } , \Sigma _ { m , c } ^ { 0 }$ and $\Sigma _ { F m , c } ^ { 0 }$ obtained from the pretrained source model may no longer be reliable. To mitigate this mismatch, we progressively update these parameters with each incoming batch of target data, allowing the model to adapt to the evolving target distribution.

Since category labels are unavailable during test-time adaptation, we estimate the category assignment of each sample through soft responsibilities derived from the pretrained model predictions. Given the source model logits $\boldsymbol { s } _ { i } ^ { t } ,$ , the soft responsibility of sample i is defined as $\gamma _ { i } ^ { t } = \mathrm { S o f t m a x } ( s _ { i } ^ { t } )$

For each category c and perspective $m \in \ m _ { 1 } , m _ { 2 } , F .$ we maintain sufficient statistics weighted by the corresponding category responsibilities, including the effective sample count $N _ { m , c } ^ { t } ,$ first-order statistic $S _ { m , c } ^ { t } ~ \in ~ \mathbb { R } ^ { d }$ , and secondorder statistic $Q _ { m , c } ^ { t } \in \mathbb { R } ^ { d \times d }$ . For each unimodal perspective $m \in \ m _ { 1 } , m _ { 2 }$ , we additionally maintain the cross statistic $Q _ { F m , c } ^ { t } \in \mathbb { R } ^ { d \times d }$ . These statistics are computed as

$$
\begin{array} { r l } & { \displaystyle N _ { m , c } ^ { t } = \sum _ { i \in \mathcal { D } ^ { 1 : t } } \gamma _ { i , c } ^ { t } , } \\ & { \displaystyle S _ { m , c } ^ { t } = \sum _ { i \in \mathcal { D } ^ { 1 : t } } \gamma _ { i , c } ^ { t } z _ { m , i } ^ { t } , } \\ & { \displaystyle Q _ { m , c } ^ { t } = \sum _ { i \in \mathcal { D } ^ { 1 : t } } \gamma _ { i , c } ^ { t } z _ { m , i } ^ { t } ( z _ { m , i } ^ { t } ) ^ { \top } , } \\ & { \displaystyle Q _ { F m , c } ^ { t } = \sum _ { i \in \mathcal { D } ^ { 1 : t } } \gamma _ { i , c } ^ { t } z _ { F , i } ^ { t } ( z _ { m , i } ^ { t } ) ^ { \top } , } \end{array}
$$

where $\begin{array} { r } { \mathcal { D } ^ { 1 : t } = \bigcup _ { r = 1 } ^ { t } \mathcal { D } ^ { r } } \end{array}$ denotes all target samples observed up to batch t. To enable streaming adaptation, the statistics are updated incrementally with the newly arrived batch:

$$
\begin{array} { r l r l } & { N _ { m , c } ^ { t } = N _ { m , c } ^ { t - 1 } + \Delta N _ { m , c } ^ { t } , } & & { \Delta N _ { m , c } ^ { t } = \displaystyle \sum _ { i \in \mathcal { D } ^ { t } } \gamma _ { i c } ^ { t } , } \\ & { } & { \mathbf { S } _ { m , c } ^ { t } = \mathbf { S } _ { m , c } ^ { t - 1 } + \Delta \mathbf { S } _ { m , c } ^ { t } , } & & { \Delta \mathbf { S } _ { m , c } ^ { t } = \displaystyle \sum _ { i \in \mathcal { D } ^ { t } } \gamma _ { i c } ^ { t } \ \boldsymbol { z } _ { m , i } ^ { t } , } \\ & { } & { \mathbf { Q } _ { m , c } ^ { t } = \mathbf { Q } _ { m , c } ^ { t - 1 } + \Delta \mathbf { Q } _ { m , c } ^ { t } , } & & { \Delta \mathbf { Q } _ { m , c } ^ { t } = \displaystyle \sum _ { i \in \mathcal { D } ^ { t } } \gamma _ { i c } ^ { t } \ \boldsymbol { z } _ { m , i } ^ { t } \boldsymbol { z } _ { m , i } ^ { t \top } , } \\ & { } & { \mathbf { Q } _ { F m , c } ^ { t } = \mathbf { Q } _ { F m , c } ^ { t - 1 } + \Delta \mathbf { Q } _ { F m , c } ^ { t } , } & & { \Delta \mathbf { Q } _ { F m , c } ^ { t } = \displaystyle \sum _ { i \in \mathcal { D } ^ { t } } \gamma _ { i c } ^ { t } \ \boldsymbol { z } _ { F , i } ^ { t } \boldsymbol { z } _ { m , i } ^ { t \top } . } \end{array}
$$

Given these accumulated sufficient statistics, the maximum likelihood estimates of the category prior, mean, covariance, and cross-covariance have the following closed-form expressions:

$$
\begin{array} { r l } & { \hat { \boldsymbol { \pi } } _ { m , c } ^ { t } = \cfrac { \boldsymbol { N } _ { m , c } ^ { t } } { \sum _ { c ^ { \prime } } N _ { m , c ^ { \prime } } ^ { t } } , } \\ & { \hat { \boldsymbol { \mu } } _ { m , c } ^ { t } = \cfrac { \mathbf { S } _ { m , c } ^ { t } } { N _ { m , c } ^ { t } } , } \\ & { \hat { \mathbf { \boldsymbol { \Sigma } } } _ { m , c } ^ { t } = \cfrac { \mathbf { Q } _ { m , c } ^ { t } } { N _ { m , c } ^ { t } } - \hat { \boldsymbol { \mu } } _ { m , c } ^ { t } \hat { \boldsymbol { \mu } } _ { m , c } ^ { t \top } , } \\ & { \hat { \mathbf { \boldsymbol { \Sigma } } } _ { F _ { m , c } } ^ { t } = \cfrac { \mathbf { Q } _ { F _ { m , c } } ^ { t } } { N _ { m , c } ^ { t } } - \hat { \boldsymbol { \mu } } _ { F , c } ^ { t } \hat { \boldsymbol { \mu } } _ { m , c } ^ { t \top } . } \end{array}\tag{22}
$$

For stable online parameter estimation, we adopt exponential moving average (EMA) updates with rate α:

$$
\begin{array} { r l } { \pi _ { m , c } ^ { t } } & { = \alpha \pi _ { m , c } ^ { t - 1 } + ( 1 - \alpha ) \hat { \pi } _ { m , c } ^ { t } , } \\ { \mu _ { m , c } ^ { t } } & { = \alpha \pmb { \mu } _ { m , c } ^ { t - 1 } + ( 1 - \alpha ) \hat { \pmb { \mu } } _ { m , c } ^ { t } , } \\ { \Sigma _ { m , c } ^ { t } } & { = \alpha \Sigma _ { m , c } ^ { t - 1 } + ( 1 - \alpha ) \hat { \Sigma } _ { m , c } ^ { t } , } \\ { \Sigma _ { F m , c } ^ { t } = \alpha \Sigma _ { F m , c } ^ { t - 1 } + ( 1 - \alpha ) \hat { \Sigma } _ { F m , c } ^ { t } . } \end{array}\tag{23}
$$

Algorithm 1 Streaming Test-Time Adaptation of AdaPGC   
Require: Target stream $\overline { { \mathcal { D } _ { \mathrm { t a r g e t } } = \{ \mathcal { D } ^ { t } \} _ { t = 1 } ^ { T } } }$ ; pretrained model $\mathcal { M } _ { \theta ^ { 0 } } ;$ loss weights $w _ { \mathrm { g } } , w _ { \mathrm { c } } ;$ calibration coefficient λ.   
1: Initialize the category-conditional Gaussian parameters $\{ \pi _ { m , c } ^ { 0 } , \mu _ { m , c } ^ { 0 } , \Sigma _ { m , c } ^ { 0 ^ { - } } \}$ from the pretrained classifier according to   
Eqs. (19) and (20), for $m \in \{ m _ { 1 } , m _ { 2 } , F \}$ and $c = 1 , \ldots , C .$   
2: Initialize $N _ { m , c } ^ { 0 } \gets 0 , S _ { m , c } ^ { 0 } \gets \mathbf { 0 } ,$ and $Q _ { m , c } ^ { 0 } \gets \mathbf { 0 }$ for $m \in \{ m _ { 1 } , m _ { 2 } , F \}$ and $c = 1 , \ldots , C .$   
3: Initialize ${ Q _ { F m , c } ^ { 0 }  \bf { 0 } }$ and $\Sigma _ { F m , c } ^ { 0 } \gets I$ for $m \in \{ m _ { 1 } , m _ { 2 } \}$ and $c = 1 , \ldots , C .$   
4: Collect all initialized sufficient statistics and Gaussian parameters into the online state ${ \mathcal { S } } ^ { 0 } .$   
5: for $t = 1$ to $T$ do   
6: Partition $\mathcal { D } ^ { t }$ according to Eq. (9), and extract all available unimodal/multimodal representations and source logits $\boldsymbol { s } _ { i } ^ { t } .$   
7: Compute the soft responsibilities from $s _ { i } ^ { t } ,$ and update the online state $S ^ { t - 1 } \to \bar { S } ^ { t }$ using the incremental sufficient   
statistics and EMA updates.   
8: Construct the Gaussian calibration logits $\boldsymbol { r } _ { i } ^ { t }$ using Eq.(28).   
9: Compute the Gaussian prediction alignment loss $\mathcal { L } _ { g } ^ { t }$ using Eq. (29).   
10: For complete samples, detect the unreliable modality by symmetric KL divergence and compute the one-sided   
rectification loss $\mathcal { L } _ { c } ^ { t }$ using Eq. (33); set $\mathcal { L } _ { c } ^ { t } = 0 \mathrm { ~ i f ~ } \mathcal { D } _ { \mathrm { f u l l } } ^ { t } = \mathrm { ~ \dot { \alpha } ~ }$   
11: Update the adaptable model parameters:   
$\theta ^ { t } \gets \theta ^ { t - 1 } - \eta \nabla _ { \theta } \left( w _ { \mathrm { g } } \mathcal { L } _ { \mathrm { g } } ^ { t } + w _ { \mathrm { c } } \mathcal { L } _ { \mathrm { c } } ^ { t } \right)$   
12: Obtain the calibrated logits using Eq. (30), and predict   
$\widehat { y } _ { i } ^ { t } = \arg \operatorname* { m a x } _ { c } \widetilde { s } _ { i , c } ^ { t } .$   
13: Output $( \widehat { \pmb { y } } ^ { t } , \theta ^ { t } , S ^ { t } )$ ; pass $( \theta ^ { t } , S ^ { t } )$ to batch $t + 1 .$   
14: end for   
15: return .

Overall, this online estimation enables the Gaussian parameters of modality-specific and fusion representations, together with the cross-covariance between them, to adapt to evolving test distributions without retaining historical samples.

## C. Gaussian-Guided Prediction Calibration

Based on the online category-conditional Gaussian statistics, we construct Gaussian calibration logits to capture the evolving category structure of the target stream. For each incoming sample, we construct the Gaussian calibration logits $\boldsymbol { r } _ { i } ^ { t }$ based on the availability of its multimodal representations, distinguishing between modality-complete and modality-incomplete samples.

For a complete sample $i \in \mathcal { D } _ { \mathrm { f u l l } } ^ { t } .$ , the fusion representation $z _ { F , i } ^ { t }$ is directly observable. Since the fusion representation integrates information from all available modalities, its category-conditional Gaussian model provides a more comprehensive description of the sample’s category-specific distribution. We therefore obtain its Gaussian calibration logits from the category-conditional Gaussian model associated with the fusion perspective as

$$
{ \pmb r } _ { i } ^ { t } = { \pmb g } _ { F } \left( { \pmb z } _ { F , i } ^ { t } \right) .\tag{24}
$$

These logits characterize the category preference induced by the updated Gaussian distribution and are directly used as calibration signals for complete samples.

For an incomplete sample $i \in \mathcal { D } _ { m } ^ { t }$ , where $m \in \{ m _ { 1 } , m _ { 2 } \}$ denotes the observed modality, only the modality-specific representation $z _ { m , i } ^ { t }$ is available, while the fusion representation cannot be directly obtained. Therefore, based on the categoryconditional Gaussian modeling of the observed modality, we can only obtain the log-posterior induced by the available representation, given by $g _ { m } \left( z _ { m , i } ^ { t } \right)$ , while the prediction induced by the unavailable fusion representation, which exploits the complete multimodal representation, cannot be directly obtained.

To incorporate the unavailable fusion representation into prediction, we estimate its category-conditioned value for each candidate category k using the learned cross-perspective statistics. Since the observed modality representation and the fusion representation are jointly modeled by a category-conditional Gaussian distribution, the conditional distribution of the fusion representation given $z _ { m , i } ^ { t }$ also follows Gaussian distribution. We therefore use its conditional mean as a point estimate:

$$
\hat { z } _ { F | m , k , i } ^ { t } = \pmb { \mu } _ { F , k } ^ { t } + \pmb { \Sigma } _ { F m , k } ^ { t } \left( \pmb { \Sigma } _ { m , k } ^ { t } \right) ^ { - 1 } \left( z _ { m , i } ^ { t } - \pmb { \mu } _ { m , k } ^ { t } \right)\tag{25}
$$

This formulation expresses the latent fusion representation through a collection of category-conditioned estimates $\hat { z } _ { F | m , k , i } ^ { t } ,$ with each estimate corresponding to the conditional expectation associated with category k.

Since the category condition is unknown, we consider all candidate categories when estimating the fusion representation. The posterior prediction from the observed modality is then used to weight the fusion prediction obtained under each category condition. Specifically, the weight for candidate category k is given by Softmax $\left( g _ { m } \left( z _ { m , i } ^ { t } \right) \right) _ { k }$ . For each candidate category k, the corresponding estimate $\hat { z } _ { F | m , k , i } ^ { t }$ is evaluated by the Gaussian model of the fusion perspective, yielding the prediction score for each output category. We aggregate these prediction scores across all candidate category conditions using the posterior weights from the observed modality. Accordingly, the recovered fusion logit for category c is defined as

$$
g _ { m  F , c } ( z _ { m , i } ^ { t } ) = \sum _ { k = 1 } ^ { C } \mathrm { S o f t m a x } ( g _ { m } ( z _ { m , i } ^ { t } ) ) _ { k } \cdot g _ { F , c } ( \hat { z } _ { F | m , k , i } ^ { t } )\tag{26}
$$

Here, k specifies the category condition used to estimate the latent fusion representation, whereas c denotes the output category of the resulting recovery logits. The corresponding vector of recovery logits is denoted by $\begin{array} { r l r } { g _ { m  F } ( z _ { m , i } ^ { t } ) } & { = } & { [ g _ { m  F , 1 } ( z _ { m , i } ^ { t } ) , \ldots , g _ { m  F , C } ( z _ { m , i } ^ { t } ) ] ^ { 1 } } \end{array}$ The probabilistic formulation and the rationale for the aggregation rule are provided in Appendix A.

The modality-specific log-posterior and the recovery logits provide complementary contributions to the Gaussian calibration of an incomplete sample. The former is evaluated directly from the observed modality, whereas the latter is derived from the estimated fusion representation. We therefore combine the two terms with equal weighting to obtain the Gaussian calibration logits for incomplete samples as

$$
\pmb { r } _ { i } ^ { t } = \frac { 1 } { 2 } [ \pmb { g } _ { m } ( \pmb { z } _ { m , i } ^ { t } ) + \pmb { g } _ { m  F } ( \pmb { z } _ { m , i } ^ { t } ) ] .\tag{27}
$$

The effect of the relative weighting between the two terms is examined in Section V-D.

Combining the complete and incomplete cases, the Gaussian calibration logits for each incoming sample are defined as

$$
\begin{array} { r } { r _ { i } ^ { t } = \{ \begin{array} { l l } { g _ { F } ( z _ { F , i } ^ { t } ) , } & { i \in \mathcal { D } _ { \mathrm { f u l l } } ^ { t } , } \\ { \displaystyle \frac { 1 } { 2 } [ g _ { m } ( z _ { m , i } ^ { t } ) + { \bf g } _ { m  F } ( z _ { m , i } ^ { t } ) ] , } & { i \in \mathcal { D } _ { m } ^ { t } . } \end{array}  } \end{array}\tag{28}
$$

Accordingly, complete samples are calibrated directly from the Gaussian evidence of their fusion representations, while incomplete samples combine the evidence available from the observed modality with that inferred for the latent fusion representation.

The Gaussian calibration logits $\boldsymbol { r } _ { i } ^ { t }$ provide distributional evidence derived from the online category-conditional statistics, whereas the source logits $\mathbf { \boldsymbol { s } } _ { i } ^ { t }$ retain the discriminative knowledge learned from the source data. However, the two predictions may exhibit distributional discrepancy, which can weaken their complementarity when directly combined. We therefore align the source model prediction with the Gaussian calibration prediction before performing logit-level fusion.

Specifically, we convert the Gaussian calibration logits into a soft prediction and use it as a fixed reference to supervise the source prediction. The resulting soft-prediction alignment loss is defined as

$$
\mathcal { L } _ { \mathrm { g } } = - \frac { 1 } { \left| \mathcal { B } \right| } \sum _ { i \in \mathcal { B } } \sum _ { c = 1 } ^ { C } \mathrm { S o f t m a x } \left( \boldsymbol { r } _ { i } ^ { t } \right) _ { c } \log \big ( \mathrm { S o f t m a x } \left( \boldsymbol { s } _ { i } ^ { t } \right) _ { c } \big ) .\tag{29}
$$

After alignment, we combine the source logits and the Gaussian calibration logits as

$$
\tilde { \mathbf { s } } _ { i } ^ { t } = \mathbf { s } _ { i } ^ { t } + \lambda \boldsymbol { r } _ { i } ^ { t } ,\tag{30}
$$

where $\lambda$ controls the contribution of the Gaussian calibration logits to the final prediction.

In this way, the soft-prediction alignment progressively adapts the source model toward the category structure captured from the test stream, while the logit-level fusion directly integrates the discriminative source prediction with the online Gaussian evidence.

## D. Reliability-Aware Representation Rectification

To address single-modality reliability shifts, we selectively rectify the representation of the degraded modality using information from the relatively reliable modality. Since both unimodal and multimodal representations are required for reliability assessment and cross-modal alignment, this rectification is performed only on complete-modality test instances in $\mathcal { D } _ { \mathrm { f u l l } } ^ { t } .$ . Specifically, we first identify the degraded modality for each test sample according to the discrepancy between unimodal and multimodal predictions, and then perform onesided contrastive alignment to rectify its representation while preserving the reliable modality.

From the log-posterior, the probabilistic prediction for each representation m $\in \{ m _ { 1 } , m _ { 2 } , F \}$ is given by $p ( c \mid z _ { m , i } ^ { t } ) =$ Softmax $( g _ { m } ( z _ { m , i } ^ { t } ) ) _ { c } .$ As demonstrated in [26], the multimodal prediction $p ( c \mid z _ { F , i } ^ { t } )$ is generally more robust and stable than the unimodal predictions $p ( c \mid z _ { m _ { 1 } , i } ^ { t } )$ and $p ( c \mid z _ { m _ { 2 } , i } ^ { t } )$ . We therefore use the multimodal prediction as a reference, such that a modality whose prediction exhibits a larger distribution discrepancy from it is considered more likely to have undergone distribution shift.

Therefore, we detect modalities exhibiting distribution shift by comparing the distribution discrepancies between different unimodal predictions and multimodal predictions. Specifically, we compute the distribution discrepancy by

$$
\begin{array} { r l } & { D _ { m _ { 1 } , i } ^ { t } = { \cal D } _ { \mathrm { S K L } } \big ( p ( c \mid z _ { m _ { 1 } , i } ^ { t } ) , p ( c \mid z _ { F , i } ^ { t } ) \big ) , } \\ & { { \cal D } _ { m _ { 2 } , i } ^ { t } = { \cal D } _ { \mathrm { S K L } } \big ( p ( c \mid z _ { m _ { 2 } , i } ^ { t } ) , p ( c \mid z _ { F , i } ^ { t } ) \big ) , } \end{array}\tag{31}
$$

where $D _ { \mathrm { S K L } } ( \cdot , \cdot )$ is the symmetric KL divergence term calculated by $\begin{array} { r } { D _ { \mathrm { S K L } } ( p , q ) = \frac { 1 } { 2 } \Big ( D _ { \mathrm { K L } } ( p \| q ) + D _ { \mathrm { K L } } ( q \| p ) \Big ) } \end{array}$ . Based on these discrepancies, the complete-modality test instances are partitioned according to which modality is less reliable:

$$
\begin{array} { r l } & { \mathcal { R } _ { m _ { 1 } } ^ { t } = \left\{ i \big | i \in \mathcal { D } _ { \mathrm { f u l l } } ^ { t } , D _ { m _ { 1 } , i } ^ { t } > D _ { m _ { 2 } , i } ^ { t } \right\} , } \\ & { \mathcal { R } _ { m _ { 2 } } ^ { t } = \left\{ i \big | i \in \mathcal { D } _ { \mathrm { f u l l } } ^ { t } , D _ { m _ { 2 } , i } ^ { t } \geq D _ { m _ { 1 } , i } ^ { t } \right\} . } \end{array}\tag{32}
$$

Here, $\mathcal { R } _ { m _ { 1 } } ^ { t }$ and $\mathcal { R } _ { m _ { 2 } } ^ { t }$ denote the index sets of completemodality test instances for which modalities $m _ { 1 }$ and $m _ { 2 }$ are identified as the less reliable modality, respectively.

Therefore, we align the representation of the less reliable modality with that of the reliable modality using a onesided InfoNCE loss. Gradients propagate only through the less reliable modality representation, while gradients through the reliable modality representation are stopped to prevent the unreliable modality from altering it. Let $\bar { z } _ { m _ { 1 } , i } ^ { \bar { t } } = \bar { z } _ { m _ { 1 } , i } ^ { t } / \| z _ { m _ { 1 } , i } ^ { t } \| _ { 2 }$ and $\bar { z } _ { m _ { 2 } , i } ^ { t } = z _ { m _ { 2 } , i } ^ { t } / \| z _ { m _ { 2 } , i } ^ { t } \| _ { 2 }$ be $\ell _ { 2 }$ -normalized representations and $\tau = 0 . 0 5$ be the temperature. We denote stop-gradient by $\operatorname { s g } ( \cdot )$ and inner product similarity by $\langle \cdot , \cdot \rangle$ . For each sample i in batch t, the loss is defined as

TABLE I  
PREDICTION ACCURACIES (IN %) ON THE KINETICS50-C BENCHMARK WITH CORRUPTED VIDEO INPUTS.
<table><tr><td rowspan="2">Models</td><td colspan="3">Noise</td><td colspan="4">Blur</td><td colspan="4">Weather</td><td colspan="4">Digital</td><td rowspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>Gauss.</td><td>Shot</td><td>Impul.</td><td>Defoc.</td><td>Glass</td><td>Mot.</td><td>Zoom</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Brit.</td><td>Contr.</td><td>Elas.</td><td>Pix.</td><td>JPEG</td></tr><tr><td>Source</td><td>46.8</td><td>48.0</td><td>46.9</td><td>67.5</td><td>62.2</td><td>70.8</td><td>66.7</td><td>61.6</td><td>60.3</td><td>46.7</td><td>75.2</td><td>52.1</td><td>65.7</td><td>66.5</td><td>61.9</td><td>59.9</td></tr><tr><td>Tent[ICLR 2021]</td><td>46.3</td><td>47.0</td><td>46.3</td><td>67.2</td><td>62.5</td><td>71.0</td><td>67.6</td><td>63.1</td><td>61.1</td><td>34.9</td><td>75.4</td><td>51.6</td><td>66.8</td><td>67.2</td><td>62.7</td><td>59.4</td></tr><tr><td>EATA[ICML 2022]</td><td>46.8</td><td>47.6</td><td>47.1</td><td>67.2</td><td>62.7</td><td>70.6</td><td>67.2</td><td>62.3</td><td>60.9</td><td>46.7</td><td>75.2</td><td>52.4</td><td>65.9</td><td>66.8</td><td>62.5</td><td>60.1</td></tr><tr><td>SAR[ICLR 2023]</td><td>46.7</td><td>47.4</td><td>46.8</td><td>67.0</td><td>61.9</td><td>70.4</td><td>66.4</td><td>61.8</td><td>60.6</td><td>46.0</td><td>75.2</td><td>52.1</td><td>65.7</td><td>66.4</td><td>62.0</td><td>59.8</td></tr><tr><td>MMT[CVPR 2022]</td><td>46.2</td><td>46.6</td><td>46.1</td><td>58.8</td><td>55.7</td><td>62.4</td><td>61.7</td><td>52.6</td><td>54.4</td><td>48.5</td><td>69.3</td><td>49.3</td><td>57.6</td><td>56.4</td><td>54.5</td><td>54.5</td></tr><tr><td>READ[ICLR 2024]</td><td>49.4</td><td>49.7</td><td>49.0</td><td>68.0</td><td>65.1</td><td>71.2</td><td>69.0</td><td>64.5</td><td>64.4</td><td>57.4</td><td>75.5</td><td>53.6</td><td>68.3</td><td>68</td><td>65.1</td><td>62.5</td></tr><tr><td>ABPEM[AAAI 2025]</td><td>50.3</td><td>51.1</td><td>50.4</td><td>70.0</td><td>69.6</td><td>72.5</td><td>71.2</td><td>65.2</td><td>66.2</td><td>65.6</td><td>75.7</td><td>56.6</td><td>71.9</td><td>70.5</td><td>67.8</td><td>65.0</td></tr><tr><td>SuMi[ICLR 2025]</td><td>50.1</td><td>50.7</td><td>50.4</td><td>68.2</td><td>65.6</td><td>72.2</td><td>69.7</td><td>65.7</td><td>67.0</td><td>56.5</td><td>77.1</td><td>55.2</td><td>69.3</td><td>71.2</td><td>68.9</td><td>63.9</td></tr><tr><td>TSA[ICML 2025]</td><td>52.6</td><td>52.3</td><td>52.0</td><td>68.7</td><td>68.0</td><td>70.7</td><td>68.8</td><td>65.2</td><td>66.6</td><td>64.3</td><td>74.6</td><td>57.4</td><td>70.5</td><td>69.0</td><td>66.2</td><td>64.5</td></tr><tr><td>BriMPR[AAAI 2026]</td><td>55.3</td><td>56.1</td><td>56.7</td><td>67.8</td><td>67.9</td><td>70.6</td><td>68.8</td><td>65.9</td><td>66.2</td><td>64.1</td><td>76.2</td><td>56.3</td><td>72.0</td><td>73.7</td><td>70.5</td><td>65.9</td></tr><tr><td>AdaPGC[This Work]</td><td>52.2</td><td>54.4</td><td>53.9</td><td>70.7</td><td>70.9</td><td>72.4</td><td>72.2</td><td>67.1</td><td>68.8</td><td>67.4</td><td>76.4</td><td>58.0</td><td>72.9</td><td>73.1</td><td>70.7</td><td>66.7</td></tr></table>

TABLE II  
PREDICTION ACCURACIES (IN %) ON THE VGGSOUND-C BENCHMARK WITH CORRUPTED VIDEO INPUTS.
<table><tr><td rowspan="2">Models</td><td colspan="3">Noise</td><td colspan="4">Blur</td><td colspan="4">Weather</td><td colspan="4">Digital</td><td rowspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>Gauss.</td><td>Shot</td><td>Impul.</td><td>Defoc.</td><td>Glass</td><td>Mot.</td><td>Zoom</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Brit.</td><td>Contr.</td><td>Elas.</td><td>Pix.</td><td>JPEG</td></tr><tr><td>Source</td><td>52.8</td><td>52.7</td><td>52.7</td><td>57.2</td><td>57.2</td><td>58.7</td><td>57.6</td><td>56.4</td><td>56.6</td><td>55.6</td><td>58.9</td><td>53.7</td><td>56.9</td><td>55.8</td><td>56.9</td><td>56.0</td></tr><tr><td>Tent[ICLR 2021]</td><td>52.7</td><td>52.7</td><td>52.7</td><td>56.7</td><td>56.5</td><td>57.9</td><td>57.2</td><td>55.9</td><td>56.3</td><td>56.3</td><td>58.4</td><td>54.0</td><td>57.4</td><td>56.2</td><td>56.7</td><td>55.8</td></tr><tr><td>EATA[ICML 2022]</td><td>53.0</td><td>52.8</td><td>53.0</td><td>57.2</td><td>57.1</td><td>58.6</td><td>57.8</td><td>56.3</td><td>56.8</td><td>56.4</td><td>59.0</td><td>54.1</td><td>57.4</td><td>56.1</td><td>57.0</td><td>56.2</td></tr><tr><td>SAR[ICLR 2023]</td><td>52.9</td><td>52.8</td><td>52.9</td><td>57.2</td><td>57.1</td><td>58.6</td><td>57.6</td><td>56.3</td><td>56.7</td><td>55.9</td><td>58.9</td><td>54.0</td><td>57.0</td><td>56.0</td><td>57.0</td><td>56.1</td></tr><tr><td>MMT[CVPR 2022]</td><td>7.1</td><td>7.3</td><td>7.3</td><td>44.8</td><td>41.5</td><td>48.0</td><td>45.5</td><td>27.4</td><td>23.5</td><td>30.5</td><td>46.3</td><td>24.0</td><td>43.0</td><td>40.7</td><td>45.7</td><td>32.0</td></tr><tr><td>READ[ICLR 2024]</td><td>53.6</td><td>53.6</td><td>53.5</td><td>57.9</td><td>57.7</td><td>59.4</td><td>58.8</td><td>57.2</td><td>57.8</td><td>55.0</td><td>59.9</td><td>55.2</td><td>58.6</td><td>57.1</td><td>57.9</td><td>56.9</td></tr><tr><td>ABPEM[AAAI 2025]</td><td>54.0</td><td>53.9</td><td>54.0</td><td>58.2</td><td>58.1</td><td>59.6</td><td>59.3</td><td>57.5</td><td>58.2</td><td>58.2</td><td>60.2</td><td>56.2</td><td>59.1</td><td>57.5</td><td>58.3</td><td>57.5</td></tr><tr><td>SuMi[ICLR 2025]</td><td>54.0</td><td>54.3</td><td>53.8</td><td>58.2</td><td>58.4</td><td>59.4</td><td>58.7</td><td>57.5</td><td>58.2</td><td>57.6</td><td>59.4</td><td>54.8</td><td>59.0</td><td>57.5</td><td>58.2</td><td>57.3</td></tr><tr><td>TSA[ICML 2025]</td><td>53.9</td><td>53.9</td><td>54.0</td><td>57.6</td><td>58.0</td><td>59.0</td><td>58.6</td><td>56.9</td><td>57.0</td><td>56.6</td><td>59.8</td><td>54.7</td><td>58.6</td><td>56.7</td><td>57.9</td><td>56.9</td></tr><tr><td>BriMPR[AAAI 2026]</td><td>54.9</td><td>55.0</td><td>55.0</td><td>57.9</td><td>58.5</td><td>58.9</td><td>58.7</td><td>57.5</td><td>58.0</td><td>58.5</td><td>60.3</td><td>54.5</td><td>59.7</td><td>59.3</td><td>59.0</td><td>57.7</td></tr><tr><td>AdaPGC[This Work]</td><td>54.9</td><td>54.9</td><td>55.1</td><td>58.2</td><td>58.2</td><td>59.5</td><td>58.9</td><td>57.6</td><td>57.8</td><td>58.5</td><td>59.8</td><td>55.2</td><td>59.2</td><td>58.4</td><td>58.6</td><td>57.7</td></tr></table>

$$
\ell _ { i } = \left\{ \begin{array} { l l } { \displaystyle - \log \frac { \exp \left( \left. \bar { z } _ { m _ { 1 } , i } ^ { t } , \mathrm { s g } \left( \bar { z } _ { m _ { 2 } , i } ^ { t } \right) \right. / \tau \right) } { \displaystyle \sum _ { j \in \mathcal { D } _ { \mathrm { t u l } } ^ { t } } \exp \left( \left. \bar { z } _ { m _ { 1 } , i } ^ { t } , \mathrm { s g } \left( \bar { z } _ { m _ { 2 } , j } ^ { t } \right) \right. / \tau \right) } , } & { i \in \mathcal { R } _ { m _ { 1 } } ^ { t } , } \\ { \displaystyle - \log \frac { \exp \left( \left. \bar { z } _ { m _ { 2 } , i } ^ { t } , \mathrm { s g } \left( \bar { z } _ { m _ { 1 } , i } ^ { t } \right) \right. / \tau \right) } { \displaystyle \sum _ { j \in \mathcal { D } _ { \mathrm { t u l } } ^ { t } } \exp \left( \left. \bar { z } _ { m _ { 2 } , i } ^ { t } , \mathrm { s g } \left( \bar { z } _ { m _ { 1 } , j } ^ { t } \right) \right. / \tau \right) } , } & { i \in \mathcal { R } _ { m _ { 2 } } ^ { t } . } \end{array} \right.\tag{33}
$$

The overall alignment loss at batch t is defined as the average of the sample-wise losses over all complete-modality test instances: $\begin{array} { r } { \mathcal { L } _ { c } = \frac { 1 } { | \mathcal { D } _ { \mathrm { f u l l } } ^ { t } | } \sum _ { i \in \mathcal { D } _ { \mathrm { f u l l } } ^ { t } } \ell _ { i } } \end{array}$

The overall loss is

$$
\mathcal { L } = w _ { \mathrm { g } } \mathcal { L } _ { \mathrm { g } } + w _ { \mathrm { c } } \mathcal { L } _ { \mathrm { c } } ,\tag{34}
$$

where only $\mathcal { L } _ { \mathrm { g } }$ and $\mathcal { L } _ { \mathrm { c } }$ are weighted by hyperparameters $( w _ { \mathrm { g } } , w _ { \mathrm { c } } )$

## V. EXPERIMENTS

## A. Experimental Settings

a) Datasets: To evaluate robustness against different types of multimodal distribution shifts, we consider both modality corruption and modality missingness scenarios. Specifically, we construct corrupted datasets, Kinetics50-C and VGGSound-C, derived from Kinetics50 [58] and VG-GSound [59]. Kinetics50 contains YouTube videos of human actions across 50 categories, while VGGSound includes 309 audio-visual categories, covering a wide range of real-world audio-visual events. Following previous work [25], [60], we introduce 15 visual distortions (e.g., Gaussian noise, motion blur, brightness variations) and six audio disturbances (e.g., traffic noise, rain noise) to simulate real-world distribution shifts. These distortions resulted in the corrupted Kinetics50-C and VGGSound-C datasets, which were then used for our evaluation, providing a challenging benchmark for multimodal testtime adaptation. We further construct modality-missing test streams from the original Kinetics50 and VGGSound datasets to evaluate model robustness when one or more modalities become unavailable during inference. Following previous work [51], [52], we consider three settings: Audio Missing, where $\eta \%$ of the samples contain only the visual modality; Visual Missing, where $\eta \%$ of the samples contain only the audio modality; and Both Missing, where $\frac { \eta } { 2 } \%$ of the samples are audio-only and another $\frac { \eta } { 2 } \%$ are visual-only. The remaining (100−η)% of the samples retain both modalities. We evaluate 5 missing rates, with η ∈ {50%, 60%, 70%, 80%, 90%}.

b) Baselines: We compare the proposed method AdaPGC with a comprehensive set of representative test-time adaptation approaches. Specifically, we include both unimodal and multimodal TTA methods to evaluate their effectiveness under challenging multimodal distribution shifts. For unimodal TTA, we consider entropy minimization-based methods, including TENT [45], EATA [46], and SAR [47]. For multimodal TTA, we compare with MM-TTA [24], READ [25], ABPEM [49], SuMi [26], BriMPR [35], and TSA [50], which are designed to exploit multimodal interactions and improve adaptation under modality variations. These baselines provide a comprehensive comparison with our probabilistic calibration framework.

TABLE III  
PREDICTION ACCURACIES (IN %) ON THE KINETICS50-C (LEFT) AND VGGSOUND-C (RIGHT) BENCHMARKS WITH CORRUPTED AUDIO INPUTS.
<table><tr><td rowspan="2">Models</td><td colspan="3">Noise</td><td colspan="3">Weather</td><td rowspan="2"> $\operatorname { A v g } .$ </td><td colspan="3">Noise</td><td colspan="3">Weather</td><td rowspan="2">Avg.</td></tr><tr><td>Gauss.</td><td>Traff.</td><td>Crowd.</td><td>Rain</td><td>Thund.</td><td>Wind</td><td>Gauss.</td><td>Traff.</td><td>Crowd.</td><td>Rain</td><td>Thund.</td><td>Wind</td></tr><tr><td>Source</td><td>73.7</td><td>65.5</td><td>67.9</td><td>70.3</td><td>67.9</td><td>70.3</td><td>69.3</td><td>37.0</td><td>25.5</td><td>16.8</td><td>21.6</td><td>27.3</td><td>25.5</td><td>25.6</td></tr><tr><td>Tent[ICLR 2021]</td><td>73.9</td><td>67.4</td><td>69.2</td><td>70.4</td><td>66.5</td><td>70.5</td><td>69.6</td><td>10.6</td><td>2.6</td><td>1.8</td><td>2.8</td><td>5.3</td><td>4.1</td><td>4.5</td></tr><tr><td>EATA[ICML 2022]</td><td>73.7</td><td>66.1</td><td>68.5</td><td>70.3</td><td>67.9</td><td>70.1</td><td>69.4</td><td>39.2</td><td>26.1</td><td>22.9</td><td>26.0</td><td>31.7</td><td>30.4</td><td>29.4</td></tr><tr><td>SAR[ICLR 2023]</td><td>73.7</td><td>65.4</td><td>68.2</td><td>69.9</td><td>67.2</td><td>70.2</td><td>69.1</td><td>37.4</td><td>9.5</td><td>11.0</td><td>12.1</td><td>26.8</td><td>23.7</td><td>20.1</td></tr><tr><td>MMT[CVPR 2022]</td><td>70.8</td><td>69.2</td><td>68.5</td><td>69.0</td><td>69.8</td><td>68.5</td><td>69.4</td><td>14.1</td><td>5.2</td><td>6.4</td><td>9.8</td><td>8.6</td><td>4.5</td><td>7.6</td></tr><tr><td>READ[ICLR 2024]</td><td>74.1</td><td>69.0</td><td>69.7</td><td>71.1</td><td>71.8</td><td>70.7</td><td>71.1</td><td>40.4</td><td>28.9</td><td>26.6</td><td>30.9</td><td>36.7</td><td>30.6</td><td>32.4</td></tr><tr><td>ABPEM[AAAI 2025]</td><td>74.8</td><td>71.3</td><td>71.5</td><td>71.9</td><td>73.8</td><td>71.6</td><td>72.5</td><td>40.6</td><td>33.7</td><td>34.8</td><td>32.2</td><td>41.1</td><td>34.4</td><td>36.1</td></tr><tr><td>SuMi[ICLR 2025]</td><td>75.1</td><td>68.9</td><td>70.6</td><td>71.6</td><td>72.8</td><td>72.1</td><td>71.9</td><td>41.9</td><td>26.3</td><td>27.9</td><td>31.6</td><td>37.1</td><td>34.1</td><td>33.2</td></tr><tr><td>TSA[ICML 2025]</td><td>74.5</td><td>69.6</td><td>70.5</td><td>71.4</td><td>72.0</td><td>71.0</td><td>71.5</td><td>41.5</td><td>31.8</td><td>30.9</td><td>32.6</td><td>38.9</td><td>32.6</td><td>34.7</td></tr><tr><td>BriMPR[AAAI 2026]</td><td>74.8</td><td>69.6</td><td>71.7</td><td>71.5</td><td>72.4</td><td>72.0</td><td>72.0</td><td>39.3</td><td>35.0</td><td>36.7</td><td>32.5</td><td>41.0</td><td>34.6</td><td>36.5</td></tr><tr><td>AdaPGC[This Work]</td><td>75.3</td><td>72.1</td><td>72.7</td><td>72.2</td><td>75.4</td><td>72.4</td><td>73.3</td><td>41.1</td><td>34.3</td><td>37.2</td><td>34.1</td><td>43.7</td><td>34.4</td><td>37.5</td></tr></table>

TABLE IV  
PREDICTION ACCURACIES (IN %) ON THE KINETICS50 BENCHMARK UNDER MODALITY-MISSING SCENARIOS.
<table><tr><td rowspan="2">Method</td><td colspan="5">Missing Audio Ratio</td><td colspan="5">Missing Video Ratio</td><td colspan="5">Missing Both Ratio</td><td rowspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>50%</td><td>60%</td><td>70%</td><td>80%</td><td>90%</td><td>50%</td><td>60%</td><td>70%</td><td>80%</td><td>90%</td><td>50%</td><td>60%</td><td>70%</td><td>80%</td><td>90%</td></tr><tr><td>Source</td><td>74.9</td><td>72.6</td><td>72.3</td><td>70.0</td><td>68.8</td><td>64.9</td><td>61.0</td><td>58.3</td><td>53.8</td><td>50.5</td><td>70.0</td><td>68.3</td><td>65.5</td><td>62.8</td><td>59.5</td><td>64.9</td></tr><tr><td>Tent[ICLR 2021]</td><td>75.0</td><td>72.8</td><td>72.7</td><td>70.1</td><td>69.0</td><td>65.2</td><td>61.6</td><td>58.2</td><td>53.7</td><td>50.4</td><td>70.4</td><td>68.5</td><td>66.0</td><td>62.9</td><td>59.4</td><td>65.1</td></tr><tr><td>EATA[ICML 2022]</td><td>75.5</td><td>72.7</td><td>72.5</td><td>70.5</td><td>69.2</td><td>65.3</td><td>61.3</td><td>58.4</td><td>53.8</td><td>50.9</td><td>70.2</td><td>68.7</td><td>65.8</td><td>63.1</td><td>59.2</td><td>65.1</td></tr><tr><td>SAR[ICLR 2023]</td><td>75.1</td><td>73.0</td><td>72.5</td><td>70.3</td><td>68.9</td><td>64.9</td><td>61.1</td><td>58.2</td><td>53.6</td><td>50.4</td><td>70.2</td><td>68.5</td><td>65.4</td><td>62.9</td><td>59.5</td><td>65.0</td></tr><tr><td>MMT[CVPR 2022]</td><td>71.5</td><td>71.0</td><td>70.4</td><td>69.3</td><td>68.2</td><td>62.0</td><td>58.7</td><td>56.2</td><td>52.8</td><td>49.9</td><td>67.0</td><td>66.3</td><td>63.6</td><td>61.6</td><td>59.1</td><td>63.2</td></tr><tr><td>READ[ICLR 2024]</td><td>74.7</td><td>73.3</td><td>72.0</td><td>70.1</td><td>69.2</td><td>64.7</td><td>61.1</td><td>56.8</td><td>53.0</td><td>49.3</td><td>70.5</td><td>68.0</td><td>64.5</td><td>61.4</td><td>57.4</td><td>64.4</td></tr><tr><td>ABPEM[AAAI 2025]</td><td>75.2</td><td>73.0</td><td>72.9</td><td>71.0</td><td>69.3</td><td>65.6</td><td>61.9</td><td>58.3</td><td>54.1</td><td>50.6</td><td>70.4</td><td>68.7</td><td>65.6</td><td>62.9</td><td>59.4</td><td>65.3</td></tr><tr><td>SuMi[ICLR 2025]</td><td>75.2</td><td>72.6</td><td>72.5</td><td>70.2</td><td>69.0</td><td>65.1</td><td>61.3</td><td>58.2</td><td>53.7</td><td>50.8</td><td>70.1</td><td>68.6</td><td>65.7</td><td>62.9</td><td>59.4</td><td>65.0</td></tr><tr><td>TSA[ICML 2025]</td><td>73.9</td><td>72.5</td><td>72.3</td><td>69.8</td><td>69.4</td><td>64.4</td><td>60.7</td><td>58.7</td><td>54.5</td><td>50.6</td><td>68.8</td><td>66.1</td><td>65.4</td><td>62.0</td><td>59.0</td><td>64.5</td></tr><tr><td>BriMPR[AAAI 2026]</td><td>75.0</td><td>73.7</td><td>72.0</td><td>70.9</td><td>69.8</td><td>64.3</td><td>60.0</td><td>57.3</td><td>53.2</td><td>50.0</td><td>70.3</td><td>67.2</td><td>65.0</td><td>61.8</td><td>57.8</td><td>64.5</td></tr><tr><td>AdaPGC[This Work]</td><td>75.5</td><td>73.9</td><td>73.4</td><td>71.8</td><td>70.7</td><td>65.8</td><td>62.5</td><td>59.9</td><td>55.5</td><td>52.2</td><td>70.7</td><td>69.4</td><td>66.3</td><td>63.6</td><td>61.1</td><td>66.2</td></tr></table>

TABLE V

PREDICTION ACCURACIES (IN %) ON THE VGGSOUND BENCHMARK UNDER MODALITY-MISSING SCENARIOS.
<table><tr><td rowspan="2">Method</td><td colspan="5">Missing Audio Ratio</td><td colspan="5">Missing Video Ratio</td><td colspan="5">Missing Both Ratio</td><td rowspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>50%</td><td>60%</td><td>70%</td><td>80%</td><td>90%</td><td>50%</td><td>60%</td><td>70%</td><td>80%</td><td>90%</td><td>50%</td><td>60%</td><td>70%</td><td>80%</td><td>90%</td></tr><tr><td>Source</td><td>46.7</td><td>43.2</td><td>39.4</td><td>36.6</td><td>32.8</td><td>58.0</td><td>56.5</td><td>55.5</td><td>54.2</td><td>53.3</td><td>52.1</td><td>50.0</td><td>47.3</td><td>45.1</td><td>43.2</td><td>47.6</td></tr><tr><td>Tent[ICLR 2021]</td><td>42.9</td><td>38.1</td><td>32.2</td><td>27.1</td><td>22.1</td><td>57.0</td><td>55.7</td><td>54.9</td><td>53.6</td><td>52.7</td><td>50.9</td><td>48.2</td><td>45.6</td><td>43.1</td><td>41.0</td><td>44.3</td></tr><tr><td>EATA[ICML 2022]</td><td>45.3</td><td>42.1</td><td>38.0</td><td>35.3</td><td>31.4</td><td>57.9</td><td>56.5</td><td>55.5</td><td>54.3</td><td>53.5</td><td>51.7</td><td>49.3</td><td>46.8</td><td>44.6</td><td>42.8</td><td>47.0</td></tr><tr><td>SAR [ICLR 2023]</td><td>44.5</td><td>40.7</td><td>36.8</td><td>33.7</td><td>29.3</td><td>55.3</td><td>53.9</td><td>52.6</td><td>50.9</td><td>49.8</td><td>50.0</td><td>47.6</td><td>44.8</td><td>42.8</td><td>40.7</td><td>44.9</td></tr><tr><td>MMT[CVPR 2022]</td><td>43.4</td><td>40.5</td><td>37.5</td><td>34.8</td><td>31.8</td><td>55.1</td><td>54.5</td><td>54.0</td><td>53.5</td><td>53.1</td><td>49.1</td><td>47.5</td><td>45.7</td><td>43.7</td><td>42.6</td><td>45.8</td></tr><tr><td>READ[ICLR 2024]</td><td>43.9</td><td>40.2</td><td>35.1</td><td>30.5</td><td>18.0</td><td>56.9</td><td>55.4</td><td>54.3</td><td>53.2</td><td>51.7</td><td>50.4</td><td>47.9</td><td>45.2</td><td>42.7</td><td>40.5</td><td>44.4</td></tr><tr><td>ABPEM[AAAI 2025]</td><td>42.2</td><td>39.5</td><td>36.1</td><td>33.9</td><td>31.1</td><td>52.4</td><td>51.5</td><td>51.0</td><td>50.4</td><td>50.5</td><td>46.9</td><td>45.2</td><td>43.3</td><td>41.6</td><td>41.0</td><td>43.8</td></tr><tr><td>SuMi[ICLR 2025]</td><td>46.0</td><td>42.5</td><td>38.8</td><td>36.1</td><td>32.3</td><td>58.0</td><td>56.4</td><td>55.4</td><td>54.4</td><td>53.4</td><td>51.8</td><td>49.7</td><td>47.0</td><td>44.8</td><td>43.0</td><td>47.3</td></tr><tr><td>TSA[ICML 2025]</td><td>42.7</td><td>40.1</td><td>35.3</td><td>33.4</td><td>29.4</td><td>56.1</td><td>54.4</td><td>53.5</td><td>52.3</td><td>50.1</td><td>48.6</td><td>46.8</td><td>44.5</td><td>42.7</td><td>39.2</td><td>44.6</td></tr><tr><td>BriMPR[AAAI 2026]</td><td>44.0</td><td>40.1</td><td>36.1</td><td>33.1</td><td>29.0</td><td>58.1</td><td>56.8</td><td>55.6</td><td>54.4</td><td>53.2</td><td>50.3</td><td>47.7</td><td>44.6</td><td>42.2</td><td>40.0</td><td>45.7</td></tr><tr><td>AdaPGC[This Work]</td><td>46.3</td><td>43.2</td><td>39.6</td><td>36.8</td><td>33.6</td><td>58.5</td><td>57.2</td><td>56.2</td><td>55.5</td><td>54.7</td><td>52.0</td><td>50.2</td><td>47.7</td><td>46.1</td><td>44.3</td><td>48.1</td></tr></table>

c) Implementation Details: For the source model, we follow READ [25] and adopt the pre-trained CAV-MAE [61]. All experiments use the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 16 (Appendix B). We configure the hyperparameters using loss weights $w _ { \mathrm { c } } = 0 . 0 1$ and $w _ { \mathrm { g } } = 1$ . The fusion weight λ is set to 1 for the modality corruption setting and 0.1 for the modality missingness setting. All experiments are conducted on a single NVIDIA A100 GPU. During adaptation, we enforce a clear separation of update routes. The loss $\mathcal { L } _ { \mathrm { g } }$ updates only the fusion-layer attention parameters $W _ { \Theta _ { h } }$ and $B _ { \Theta _ { h } }$ for $h ~ \in ~ \{ Q , K , V \}$ without modifying ny modality-specific encoders. In contrast, $\mathcal { L } _ { \mathrm { c } }$ updates only the LayerNorm modules within the encoder of the modality flagged as corrupted by the reliability test.

## B. Main Results

We first evaluate the overall performance of AdaPGC under different multimodal distribution shift scenarios. As shown in Tables I-III, AdaPGC achieves the best average performance under modality corruption scenarios, where video and audio modalities are independently evaluated under different corruption types. Tables IV-V further report the results under modality missingness scenarios, where AdaPGC achieves consistent improvements across different missing patterns.

TABLE VI  
ABLATION STUDY ON THE CONTRIBUTIONS OF DIFFERENT COMPONENTS IN ADAPGC ON THE KINETICS50 DATASET. THREE REPRESENTATIVE CONDITIONS ARE SHOWN FOR EACH SETTING. AVG. IS COMPUTED OVER ALL 15 VIDEO-C CORRUPTIONS, ALL 6 AUDIO-C CORRUPTIONS, OR ALL 15 MISSING-MODALITY CONDITIONS.
<table><tr><td rowspan="2">PA</td><td rowspan="2"></td><td rowspan="2">AR</td><td colspan="4">Video-C</td><td colspan="4">Audio-C</td><td colspan="4">Missing</td></tr><tr><td>Defoc.</td><td>Snow</td><td>Pix.</td><td>Avg.</td><td>Gauss.</td><td>Crowd</td><td>Wind</td><td>Avg.</td><td>A-70%</td><td>V-70%</td><td>Both-70%</td><td>Avg.</td></tr><tr><td rowspan="4">√</td><td></td><td></td><td>67.82</td><td>61.09</td><td>68.95</td><td>60.64</td><td>73.55</td><td>67.74</td><td>70.65</td><td>69.22</td><td>72.06</td><td>58.15</td><td>65.36</td><td>64.83</td></tr><tr><td></td><td></td><td>68.15</td><td>63.23</td><td>69.07</td><td>63.09</td><td>74.11</td><td>69.48</td><td>70.81</td><td>70.70</td><td>72.86</td><td>59.11</td><td>65.85</td><td>65.56</td></tr><tr><td>√</td><td></td><td>68.95</td><td>64.68</td><td>70.93</td><td>64.40</td><td>74.31</td><td>70.65</td><td>71.81</td><td>72.17</td><td>72.90</td><td>58.99</td><td>65.77</td><td>65.49</td></tr><tr><td></td><td>√</td><td>68.23</td><td>65.32</td><td>70.32</td><td>63.42</td><td>74.52</td><td>70.16</td><td>71.49</td><td>70.77</td><td>72.38</td><td>58.27</td><td>65.56</td><td>65.02</td></tr><tr><td rowspan="3">√ √</td><td>√</td><td></td><td>69.40</td><td>65.85</td><td>71.98</td><td>65.71</td><td>75.00</td><td>71.94</td><td>72.46</td><td>73.05</td><td>73.55</td><td>60.04</td><td>66.13</td><td>66.09</td></tr><tr><td></td><td>√</td><td>69.11</td><td>66.57</td><td>71.73</td><td>65.43</td><td>74.84</td><td>71.33</td><td>71.17</td><td>71.92</td><td>72.82</td><td>59.52</td><td>66.33</td><td>65.78</td></tr><tr><td>√</td><td>√</td><td>69.72</td><td>66.17</td><td>72.38</td><td>65.67</td><td>74.92</td><td>71.41</td><td>72.26</td><td>72.64</td><td>72.98</td><td>59.15</td><td>66.09</td><td>65.58</td></tr><tr><td></td><td>V</td><td></td><td>70.69</td><td>67.10</td><td>73.10</td><td>66.73</td><td>75.28</td><td>72.66</td><td>72.42</td><td>73.35</td><td>73.39</td><td>59.88</td><td>66.21</td><td>66.13</td></tr></table>

![](images/43bc5eb08335f70650e5b8ec370254f0b8c0b157f286afdc53d912c8ec5e5221.jpg)

![](images/a16b97618961922de1cdc90f8d52c4811ddb17d26961d4341d58dec6e2868bd5.jpg)

![](images/9705c4de2f20f1e972ab5a385e12533bba41a55d8d61b5950724ba8cd20b3a95.jpg)

![](images/f15651eaf64ecaa4576bf873898955ee394069bfe87dff86d4d4eb52cd1e4237.jpg)  
Fig. 3. Sensitivity analysis of λ on Kinetics50. (a) Mean accuracy under audio corruption, video corruption, and missing-modality settings. (b–d) Accurac under representative audio corruptions, video corruptions, and missing-modality conditions, respectively.

Under modality corruption scenarios, AdaPGC demonstrates consistent robustness against diverse types of visual and audio degradations. Across the corruption benchmarks, AdaPGC achieves competitive average accuracy in both videocorrupted and audio-corrupted settings on both datasets. The improvements are consistently observed across different corruption types rather than being limited to specific corruption. For example, on Kinetics50-C with corrupted video inputs, AdaPGC improves the average accuracy from 65.9% achieved by the strongest competing method to 66.7%. For corrupted audio inputs on VGGSound-C, AdaPGC further improves the average accuracy from 36.5% to 37.5%, with consistent gains across various audio disturbances. These results indicate that modeling category-conditional representation distributions provides reliable calibration signals when modality information is affected by different types of corruption.

Under modality missingness scenarios, AdaPGC improves the adaptation performance when partial modality information becomes unavailable. Across different missing configurations and missing ratios, AdaPGC achieves the highest average accuracy on both Kinetics50 and VGGSound benchmarks. The performance remains stable as the missing ratio increases, demonstrating the effectiveness of recovering latent fusion representations from the available modality information. For example, on Kinetics50 with the video modality missing in 90% of samples, AdaPGC achieves 52.2% accuracy, exceeding the best competing result by 1.3 percentage points. Similarly, under the Both Missing setting with a 90% missing ratio, AdaPGC improves accuracy from the best baseline result of 59.5% to 61.1%. These results verify that the proposed cross-perspective representation recovery effectively preserves discriminative multimodal information even when complete multimodal inputs are unavailable.

## C. Ablation Study

We further conduct ablation studies to investigate the contribution of each component in AdaPGC. The results are summarized in Table VI. Specifically, we evaluate three components: the fused-logit calibration mechanism (FL; (30)), which integrates the calibration logits with the original model predictions to refine the final output; the prediction alignment loss (PA; (29)), which uses the calibration prediction as supervision to guide test-time optimization; and the asymmetric rectification module (AR; (33)), which mitigates the impact of inconsistent modality reliability by rectifying the representation of the less reliable modality. When all three components are included, AdaPGC achieves the highest average accuracy in each of the three evaluation settings; removing any component reduces the corresponding average accuracy.

FL consistently improves the baseline across all three settings, increasing the average accuracy by 2.45, 1.48, and 0.73 percentage points on Video-C, Audio-C, and Missing, respectively. Its contribution remains evident when PA is already enabled, where adding FL further improves the average accuracy from 64.40 to 65.71 on Video-C, from 72.17 to 73.05 on

(a) Audio-C Avg.  
![](images/36af841386824938f4c95f72c0eaf658fa8187b05d901b120ef5eeb0b80132e5.jpg)

![](images/e9ef41f05668e105001e3d14f115265610f8eae27789e2089702178e323d27bb.jpg)

![](images/994158f96f7a713fcae2d42116924828e920c6514f578fb8b7985189158ddb9a.jpg)

(d) Thund.  
![](images/4bb2ea0fb734fc2bc0027d7f63616aa999c0c66f450bf5f9914c0fcc3ccae084.jpg)

(e) Video-C Avg.  
![](images/69949891cf445ae813a481c5c13cc235ee0870eddcc421b414757bb7a1e44cdc.jpg)

(f) Shot  
![](images/bd7dd96d60e9b8b4c2121b019f48d0861aab7632dabc0d3a15b53b1e6216e37e.jpg)

(g) Defoc.  
![](images/71260e2d73db3e80d4fbdbcaafd514b72e704e69f38ab4115b04f7535f25873f.jpg)

(h) Snow  
![](images/ad775f1e7c35ddb0c776dbe70524d23b1687879d0d93042714d6d7bd35fee486.jpg)

(i) Missing Avg.  
![](images/f9718da6106410c57e6dcc8aad315e55e0b4a97930f76adb912168f215cc9b4f.jpg)

(j) A-Miss-70%  
![](images/2b0c61771785488d5d0c88f1b78d54feadf48bac3d70c831f112fa193d6c4be8.jpg)

(k) V-Miss-70%  
![](images/1b0014b3c3c3e40a5f96299d6958280e824611435d8ed28d56c4c4080ace5772.jpg)

(l) Both-Miss-70%  
![](images/6230fcd4c927acd13ce9b054684c90fb56cb0710d5ffdfc0005d8ec8a2feee30.jpg)  
Fig. 4. Sensitivity of AdaPGC to w and w on Kinetics50. Rows correspond to audio corruption (a–d), video corruption (e–h), and missing-modality settings (i–l). The first column shows the accuracy averaged over all conditions in each setting, while the remaining columns present representative conditions.

(a) Overall  
![](images/f884baaf714039795e363a4f84fc5b1e1c830a4bffa9cbeb7adf8b784e6e45fa.jpg)

![](images/3415fd44a05d7c9dd8fffbc1e27368b13e0be991fc1fa5a2d404e68612ed659c.jpg)  
Fig. 5. Sensitivity to the relative weighting of the modality-specific logposterior and the recovery logits on Kinetics50. For incomplete samples, the Gaussian calibration logits are computed as $\begin{array} { r } { \pmb { r } = ( 1 - \beta ) \pmb { g } _ { m  F } + \beta \pmb { g } _ { m } . } \end{array}$ (a) Mean accuracy over all 15 missing-modality conditions. (b) Mean accuracy under each modality-missing setting, averaged over missing rates of 50%– 90%. Horizontal dashed lines indicate the results using only $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } \_ m }$ , while red stars mark the adopted equal weighting (β = 0.5).

Audio-C, and from 65.49 to 66.09 under modality missingness. These results demonstrate that directly incorporating Gaussian calibration into the final prediction provides complementary information beyond adapting the model parameters alone.

PA yields the largest standalone improvement under modality corruption, raising the average accuracy by 3.76 percentage points on Video-C and 2.95 percentage points on Audio-C. This indicates that the probabilistic predictions derived from online Gaussian distributions provide effective supervision for adapting the source model to reliability changes at test time. Moreover, the consistent gains obtained by combining PA with FL show that Gaussian calibration serves two complementary roles: it can guide test-time model adaptation through PA while also directly calibrating the final prediction through FL.

AR provides a distinct contribution that is primarily concentrated on modality corruption. When used alone, it improves the average accuracy by 2.78 and 1.55 percentage points on Video-C and Audio-C, respectively, while producing only a marginal improvement under modality missingness. A similar pattern is observed when AR is added to FL and PA: the average accuracy increases from 65.71 to 66.73 on Video-C and from 73.05 to 73.35 on Audio-C, whereas the change under modality missingness is negligible. This behavior is consistent with the role of AR, which is specifically designed to rectify degraded modality representations rather than compensate for unavailable modalities. Its benefit is particularly evident under several corruption conditions, such as Snow, where AR alone improves the accuracy from 61.09 to 65.32.

Overall, FL, PA, and AR operate at complementary levels: FL directly calibrates the prediction, PA transfers probabilistic calibration into test-time model adaptation, and AR rectifies degraded modality representations. Combining all three components achieves the highest average accuracy on Video-C, Audio-C, and Missing, confirming their complementary contributions to robust adaptation under modality shifts.

Clean

Corrupted

## (a) Gaussian noises

GT: Baking cookies

![](images/8513f86ba24696d688fe6c9fa8bf80dcea59031331425e06c65de803e4a8b441.jpg)

![](images/c5e3530f3fedaa5d3bec4e00868576f6139dde167a6a4822831cf457f608ae9c.jpg)  
Backbone

![](images/19946255008ec6091f62f04b8375c546d792a4852a1b4b04f46fcea76bde81ab.jpg)  
Waxing legs

(c) Impulse noises GT: Unboxing

![](images/36d361361447f83c4d73118aaec5132ce0cbdc35e92712d4e0dfa71769528a88.jpg)

![](images/7c74a4a45a14e81e1f6673ed2522fc8108ffef137f547332058b1fff8005fcb5.jpg)

![](images/54c967eebc99e323320ec27d14419f41454d63e64dbec82680da711e4c8b155b.jpg)  
Baking cookies

(e) Glass blur

GT: Passing American football

![](images/2681faf0aa34c08a22d277be6987e14106fe2b3a97a8b385d4e4b1e42c9e98ea.jpg)

![](images/dfd4610f02118cd7077bbb3ec54c1f63c9b25a39ff187ad2710d40eb41419cf7.jpg)

![](images/efbd58fb77227a08aee54bae0cbe1128d3b212a46dc3423557329766b73f4473.jpg)  
Diving cliff

(g) Snow

GT: Playing flute

![](images/c8a79fdb9dc8996dfe524d2b1e61267340e35a60cecd24b67ba78d24e364a9f5.jpg)

![](images/43b2cb530c4e346d446087070ad274321f594ddc5560c6fa9280710c8f859236.jpg)

![](images/88e34ceb1fc3a7fa19c797135ec1db825b7beb5a3bc85de77e8e070f057afe0e.jpg)  
Beatboxing

![](images/df79b8d4d00f7bf93f2441016c806cb53fbd29a8f27366187c968dda51aacb81.jpg)

![](images/895534dc5700c1ac73358244cca25f60b5d69d70fc78bda01e7c99803c8ad1e3.jpg)  
AdaPGC  
Unboxing  
Baking cookies  
Passing American football

(i) Contrast

![](images/c03ed768377e75d400de549dff64f4c79409791dccd15ae82b7f1ee5184a8953.jpg)

![](images/ca7885e2158904cbff6dd29a365c2097ccb0bfbbfaa2c82f4b7221a193028448.jpg)  
Playing flute

![](images/479c8a0cc9a34aff5152e5b1eb6ddc7af6e57c436cf40d76b454d938d0457891.jpg)

![](images/dd595e5271bfb6487b884f0f571989be585eb606a127270d4b2457966b840082.jpg)

GT: Petting cat

![](images/cde3f5fdbbcbce5b199ef07021db08f24d0a62b615ee122f8be329878ee23068.jpg)  
Taking a shower  
Clean  
Corrupted

## (b) Shot noises

GT: Golf driving

![](images/7c027b68e9a624cb0849ce3ed37ce58e14d47992df70aad80ec7a86eb1c2b5a2.jpg)

![](images/b4a5ce0eaa61a788d70ac0fbfa7d3b496c29d450856af432b77a8f9013648bed.jpg)

![](images/d72744a33db44227050d69c313dc6f98438daa8e6af2e01e4e0c8eb777531e2e.jpg)  
Petting cat

Backbone

![](images/7c3d280820ba3edc64218ff1aafb4d254aa4c1f5330e17e74e08f1623f91e2ed.jpg)  
Opening bottle  
Golf driving

## (d) Defocus blur

![](images/8eb6b046d5bc2380977217468fb1e3463844067b678a8d343f778d238cda0459.jpg)

![](images/3db975328e795071b500eca6e5b24ac8dcbd9ba2878589f9022037f4c64b3c4c.jpg)

![](images/97b4b7b64677f8c79fc2f8bf877f14bb01600ec21f42e8ed71982c372728cb6f.jpg)

![](images/4a01f2165d19fa42280122eb9d039b78ea1ea7c462a538671037f7aaf6afa53a.jpg)  
Assembling computer

AdaPGC

![](images/d6da02f162702819d07457b6c8b47dbc08b191e098bb5eeda29ee66ecc07d62d.jpg)  
Extinguishing fire

GT: Dribbling basketball

![](images/a5d276095a994643304766bed83169a419190dd4a7f564dc3f7c979c7b929bc8.jpg)

![](images/9e1ac9c428f45031172557bc04629df489cfeb4796bb9074c3e2e18098d54d0d.jpg)

![](images/702f4a1e80566d89d0df053e1affc216d517086f6375dc21f7e51dbb4c4eb4d0.jpg)  
Taking a shower  
(h) Brightness GT: Pull-ups

![](images/dc6b99c3721965f051d6babc6afb59402edc0caa8b46ecb4c1803368282ce1a6.jpg)

![](images/dd7b0f0894acb662449b9c2256ca19ea75d8beb925f42e7460f7f29676f74d11.jpg)  
Dribbling basketball

![](images/dfba1ce58b21f577e609762d2dddf2e99036d6cc15c9f2c7178bfc518f176a41.jpg)  
Petting cat

![](images/b570b591987ef5863a0e942cc6ce356339b7e1fa5d91fb457c707b1b765c1e1d.jpg)

![](images/665c5360147866c28d8997c85ff7a67f3e95a19fa26712070462fc957a1c535a.jpg)  
Pull-ups

## (j) Elastic transform

![](images/5c6d1733dedd170f3a0d0f70977041fd650ebe45eca3d5b1b21d22a24fa3e725.jpg)

![](images/0de9a770b3779af7d7f4e558eb6003ed9f39dfbfad60aaab0fbc3fa10cef351f.jpg)

![](images/3ff08efb238b3d81c0a75d7681ec1f0b31a2bd4531cf580e1d90b1a6810d7513.jpg)  
Throwing ball

![](images/eceeeff51b614a73ed000a5a0f1719795c16240b57e1b5af98cee70894c4dca9.jpg)  
Push-up

(k) Pixelation

GT: Eating doughnuts

![](images/3786df2b5c10ea7912570ed375280e9688b9a636a10c84ba3d50b73da2435cb8.jpg)

![](images/03d1db6e4aed96adea6a74ed75959153da0193b4e0626e70cd1e39cd5b6904c3.jpg)

![](images/c2525751d6bfac6713b88cff6cc28ad4552db189d473d0cdaa13eb3522f7ee3e.jpg)  
Headbutting

![](images/40ba1eeaee54c966c620320a8646a26eadcc73b721314a6fda1ca73c59f47702.jpg)  
Eating doughnuts

## (l) JPEG compression

![](images/e32eada57f5ccd3a7b2a4344214ec83b952d976d09bcaf5078bd0e3f38f185ec.jpg)

![](images/9f8e0b81fb7f1819eb90b5220c954d69bd8c9f8310eb83b9047b7c51365e8fab.jpg)

![](images/6ffe363eede1767f44ec73b74f13d0a3ab1b0e2d36422ea11ac386a5b66b5030.jpg)  
Pumping fist

![](images/962e83790e39387e74829e891732fe8685537ecf7967e3f9264c12aeea952756.jpg)  
Dyeing hair

Fig. 6. Grad-CAM visualizations of the backbone and AdaPGC under twelve visual corruptions on Kinetics50-C. Each example shows the clean frame, corrupted input, and prediction-specific Grad-CAM overlays for the backbone and AdaPGC. GT denotes the ground-truth category; red and green labels indicate incorrect and correct predictions, respectively. In these selected correction cases, AdaPGC generally emphasizes action-related regions more clearly, whereas backbone responses are often dispersed or concentrated on the background. Maps are independently normalized for each branch’s predicted category, with warmer colors indicating larger relative positive contributions.

![](images/5b50b2759d9ad917dd478d9dec8dbe7725eb41ef46b3c2e6a5c25e4465b78d3f.jpg)  
Fig. 7. Cumulative accuracy during test-time adaptation on Kinetics50. Rows correspond to audio corruption (a–d), video corruption (e–h), and missingmodality settings (i–l). A-miss and V-miss denote missing audio and visual modalities, respectively, with percentages indicating the missing rates. Each curve shows the accuracy over all test samples processed so far.

## D. Further Analysis

a) Sensitivity to the Residual Calibration Weight: We investigate the sensitivity of AdaPGC to the residual calibration weight λ, which scales the Gaussian calibration term added to the source-model logits. As shown in Figure 3, the effect of λ differs between modality corruption and missingness. Under modality corruption, increasing λ beyond 1 provides limited additional improvements, although higher accuracy can be obtained on Video-C with larger values. Under modality missingness, a smaller calibration contribution is preferable, with $\lambda \ : = \ : 0 . 1$ achieving the highest mean accuracy among the evaluated values. These observations are consistent with the adopted settings of $\lambda = 1$ for modality corruption and $\lambda = 0 . 1$ for modality missingness.

b) Hyperparameter Sensitivity Analysis: We further analyze the sensitivity of AdaPGC to the loss weights $w _ { g }$ and $w _ { c } ,$ which control Gaussian prediction alignment and reliabilityaware representation rectification, respectively. Figure 4 reports the results obtained by varying these two hyperparameters over a predefined grid. The optimal combinations of $w _ { \mathrm { c } }$ and $w _ { \mathrm { g } }$ vary across different corruption types and missing-modality conditions, as highlighted by the stars in each subfigure. However, when accuracy is averaged over all conditions within each evaluation setting, the performance surfaces become relatively smooth across the grid, indicating that AdaPGC maintains stable average performance without relying on precise tuning of these two weights.

c) Relative Weighting of Calibration Terms: To examine the equal weighting adopted in (27), we introduce a mixing coefficient β and compute the Gaussian calibration logits for incomplete samples as $\pmb { r } ~ = ~ ( 1 - \beta ) \pmb { g } _ { m  F } + \beta \pmb { g } _ { m }$ , where $\beta ~ = ~ 0$ uses only the recovery logits and $\beta ~ = ~ 1$ uses only the modality-specific log-posterior. As shown in Figure 5, the preferred weighting differs across modality-missing settings: Visual Missing favors the recovery logits, whereas Both Missing achieves its highest mean accuracy with equal weighting. These results suggest that the relative utility of the two calibration terms depends on the modality availability pattern. A paired t-test further supports a statistically significant improvement in overall mean accuracy over using only $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { m } }$ at the 95% confidence level $( p = 0 . 0 1 5 6 )$ . The adopted setting $\beta = 0 . 5$ balances these setting-dependent preferences and achieves near-optimal overall accuracy, supporting its use as a common configuration across different modality-missing settings.

d) Case Study : We use Grad-CAM to examine the spatial evidence associated with the predictions of the backbone and AdaPGC under modality reliability shifts. As shown in Figure 6, twelve representative samples from Kinetics50- C are selected across different visual corruptions, covering cases where AdaPGC corrects an erroneous backbone prediction. The visualizations reveal differences in the regions contributing to the respective predictions. In the illustrated examples, the backbone exhibits responses extending beyond action-related regions, whereas AdaPGC shows comparatively stronger emphasis on regions relevant to the target action. For example, under shot noise and fog, AdaPGC emphasizes the golfer in Figure 6 (b) and the basketball player in Figure 6 (f), whereas the backbone responds more strongly to surrounding regions. In Figure 6 (g), the backbone misclassifies playing flute as beatboxing and primarily highlights the face, while

![](images/9c5f5e38cd5ae49b50fa29a19dd4301f182ef29c7b8e4ce571b4115c02a55184.jpg)  
Category A Category B Category C Category D  
Fig. 8. Visualization of fusion representation recovery under multimodal missingness. Panels (a–b) and (c–d) show the 80% and 90% missing-both settings, respectively, using the available video or audio modality as the recovery source. Colors and fill styles distinguish representation types, marker shapes denot semantic categories, and gray lines connect each recovered representation to its paired clean ground truth.

![](images/300f15e60975b2feb9bb9381a8a7a591b9f16ca14b1de0ce06be412d7303a053.jpg)

![](images/2daf6394317f7428afef2dc4920a84938b14fc072a5b173a0ebc02f5f1916f26.jpg)  
Fig. 9. Unreliable modality identification accuracy on Kinetics50-C and VGGSound-C under video and audio corruptions. The dashed horizontal line indicates the 50% chance level, and error bars represent 95% confidence intervals.

AdaPGC emphasizes the flute and arms, providing spatial evidence more consistent with the correct action. Similar patterns are observed under elastic transformation and JPEG compression, where AdaPGC highlights the person performing the push-up and the hair region, respectively. These selected cases illustrate how successful prediction calibration is accompanied by spatial attribution patterns that better correspond to recognizable action-related content, complementing the quantitative evaluation of robustness to visual corruption. Each map explains the corresponding branch’s predicted category and is independently normalized, so the comparison concerns qualitative spatial patterns rather than absolute attribution magnitudes.

e) Adaptation Performance Over the Test Stream: To examine how adaptation performance evolves as test samples arrive, we compare the cumulative accuracy of AdaPGC with the source model and representative multimodal TTA methods. Figure 7 presents results under selected audio corruptions, video corruptions, and missing-modality conditions. Despite initial fluctuations, AdaPGC achieves the highest final cumulative accuracy across all displayed conditions. Under modality corruption, AdaPGC maintains a clear advantage in most cases, particularly for thunder noise, shot noise, and JPEG compression. Some competing methods exhibit declining accuracy as adaptation proceeds, such as SuMi under rain noise, whereas AdaPGC maintains relatively stable performance. Moreover, under glass blur, elastic transformation, and JPEG compression, AdaPGC exhibits an overall upward trend in cumulative accuracy after the initial fluctuations, indicating that its predictions improve as adaptation proceeds. Under modality missingness, AdaPGC also achieves consistent improvements at both 80% and 90% missing rates. These results complement the aggregate comparisons and demonstrate the effectiveness of AdaPGC throughout adaptation under corrupted and incomplete multimodal inputs.

f) Recovered Fusion Representation Analysis: We further analyze the recovered fusion representations under modality missingness scenarios. Since the proposed recovery process estimates category-conditional fusion representations $\hat { z } _ { F | m , c , i }$ for each possible category, we aggregate them using the modality-specific posterior probabilities for visualization:

$\begin{array} { r } { \hat { z } _ { F , i } = \sum _ { c = 1 } ^ { C } \mathrm { S o f t m a x } ( g _ { m } ( z _ { m , i } ) ) _ { c } \hat { z } _ { F | m , c , i } } \end{array}$ . The visualization compares the aggregated recovered representations with the ground-truth fusion representations obtained from complete multimodal inputs. As shown in Figure 8, the recovered representations remain close to their corresponding ground-truth fusion representations and preserve the category-wise clustering structures. Moreover, this alignment is maintained under different missing ratios, indicating that the proposed categoryconditional cross-perspective recovery captures meaningful statistical relationships between available modality representations and fusion representations.

g) Unreliable Modality Identification: We evaluate the effectiveness of the proposed distribution discrepancy criterion in identifying unreliable modalities under different corruption scenarios. As shown in Figure 9, the identification accuracy remains consistently above the 50% chance level across different corruption types and datasets, demonstrating that the discrepancy between modality-specific and multimodal predictions provides informative signals for reliability assessment. The variations among corruption types indicate that different corruptions induce different levels of prediction discrepancy, while the overall effectiveness of the criterion remains stable. These results validate the reliability identification strategy used in the asymmetric rectification module.

## VI. CONCLUSION

This work presents AdaPGC, a probabilistic multimodal test-time adaptation framework for distribution shifts involving variations in both modality reliability and availability. AdaPGC continuously models category-conditional distributions over modality-specific and multimodal fusion representations, using the resulting probabilistic predictions to calibrate source model outputs throughout the target stream. Beyond prediction calibration, AdaPGC exploits cross-perspective distributional information to rectify degraded modality representations and further models the joint distributions between modality-specific and fusion representations to recover unavailable fusion information under modality missingness. Extensive experiments across diverse corruption and missingness conditions demonstrate that AdaPGC consistently improves adaptation robustness over existing approaches, validating the effectiveness of probabilistic distribution modeling for multimodal test-time adaptation under complex modality shifts.

## REFERENCES

[1] P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transformers: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 45, no. 10, pp. 12 113–12 132, 2023.

[2] X. Bruce, Y. Liu, X. Zhang, S.-h. Zhong, and K. C. Chan, “Mmnet: A model-based multimodal network for human action recognition in rgb-d videos,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 3, pp. 3522–3538, 2022.

[3] Y. Zhu, Y. Wu, N. Sebe, and Y. Yan, “Vision+ x: A survey on multimodal learning in the light of data,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9102–9122, 2024.

[4] E. Kazakos, A. Nagrani, A. Zisserman, and D. Damen, “Epic-fusion: Audio-visual temporal binding for egocentric action recognition,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2019, pp. 5491–5500.

[5] A. Nagrani, S. Yang, A. Arnab, A. Jansen, C. Schmid, and C. Sun, “Attention bottlenecks for multimodal fusion,” Advances in neural information processing systems, vol. 34, pp. 14 200–14 213, 2021.

[6] A. Das, P. Sil, P. K. Singh, V. Bhateja, and R. Sarkar, “Mmhar-ensemnet: a multi-modal human activity recognition model,” IEEE Sensors Journal, vol. 21, no. 10, pp. 11 569–11 576, 2020.

[7] J. Liu, S. Song, C. Liu, Y. Li, and Y. Hu, “A benchmark dataset and comparison study for multi-modal human action analytics,” ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), vol. 16, no. 2, pp. 1–24, 2020.

[8] Z. Sun, Q. Ke, H. Rahmani, M. Bennamoun, G. Wang, and J. Liu, “Human action recognition from various data modalities: A review,” IEEE transactions on pattern analysis and machine intelligence, vol. 45, no. 3, pp. 3200–3225, 2022.

[9] A. Zadeh, M. Chen, S. Poria, E. Cambria, and L. Morency, “Tensor fusion network for multimodal sentiment analysis,” in Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017, M. Palmer, R. Hwa, and S. Riedel, Eds. Association for Computational Linguistics, 2017, pp. 1103–1114. [Online]. Available: https://doi.org/10.18653/v1/d17-1115

[10] Y. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L. Morency, and R. Salakhutdinov, “Multimodal transformer for unaligned multimodal language sequences,” in Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, A. Korhonen, D. R. Traum, and L. Marquez, Eds. Association for Computational\` Linguistics, 2019, pp. 6558–6569. [Online]. Available: https://doi.org/ 10.18653/v1/p19-1656

[11] Y. Yu, D. Zhang, and S. Li, “Unified multi-modal pre-training for fewshot sentiment analysis with prompt-based learning,” in Proceedings of the 30th ACM international conference on multimedia, 2022, pp. 189– 198.

[12] W. Yu, H. Xu, Z. Yuan, and J. Wu, “Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 12, 2021, pp. 10 790–10 797.

[13] J. Li, X. Wang, and Z. Zeng, “Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for multimodal emotion recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 10, pp. 8786–8803, 2025.

[14] P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, and L. Zhang, “Bottom-up and top-down attention for image captioning and visual question answering,” in 2018 IEEE/CVF conference on computer vision and pattern recognition. IEEE, 2018, pp. 6077–6086.

[15] X. Bai, Z. Hu, X. Zhu, Q. Huang, Y. Chen, H. Fu, and C.-L. Tai, “Transfusion: Robust lidar-camera fusion for 3d object detection with transformers,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 1090–1099.

[16] Y. Xin, J. Du, Q. Wang, K. Yan, and S. Ding, “Mmap: Multi-modal alignment prompt for cross-domain multi-task learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 14, 2024, pp. 16 076–16 084.

[17] Z. Zhang, S. Cao, and Y.-X. Wang, “Tamm: Triadapter multi-modal learning for 3d shape understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 413–21 423.

[18] A. Kamath, M. Singh, Y. LeCun, G. Synnaeve, I. Misra, and N. Carion, “Mdetr-modulated detection for end-to-end multi-modal understanding,” in Proceedings of the IEEE/CVF International conference on computer vision, 2021, pp. 1780–1790.

[19] H. Ding, C. Liu, S. Wang, and X. Jiang, “Vlt: Vision-language transformer and query generation for referring segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 6, pp. 7900–7916, 2022.

[20] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 46, no. 8, pp. 5625–5644, 2024.

[21] M. Jaritz, T.-H. Vu, R. De Charette, E. Wirbel, and P. P <sup>´</sup> erez, “Cross-´ modal learning for domain adaptation in 3d semantic segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 2, pp. 1533–1544, 2022.

[22] S. Chen, J. Gu, Z. Han, Y. Ma, P. Torr, and V. Tresp, “Benchmarking robustness of adaptation methods on pre-trained vision-language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 51 758–51 777, 2023.

[23] H. Dong, M. Liu, K. Zhou, E. Chatzi, J. Kannala, C. Stachniss, and O. Fink, “Advances in multimodal adaptation and generalization: From traditional approaches to foundation models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[24] I. Shin, Y.-H. Tsai, B. Zhuang, S. Schulter, B. Liu, S. Garg, I. S. Kweon, and K.-J. Yoon, “Mm-tta: multi-modal test-time adaptation for 3d semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 928–16 937.

[25] M. Yang, Y. Li, C. Zhang, P. Hu, and X. Peng, “Test-time adaptation against multi-modal reliability bias,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. [Online]. Available: https://openreview.net/forum?id=TPZRq4FALB

[26] Z. Guo and T. Jin, “Smoothing the shift: Towards stable test-time adaptation under complex multimodal noises,” in The Thirteenth International Conference on Learning Representations, 2025.

[27] S. Cui, J. Xu, Y. Li, X. Tang, J. Li, J. Zhou, F. Xu, F. Sun, and H. Xiong, “Bayestta: Continual-temporal test-time adaptation for visionlanguage models via gaussian discriminant analysis,” arXiv preprint arXiv:2507.08607, 2025.

[28] Z. Han, J. Yang, G. Wang, J. Li, Q. Xu, M. Z. Shou, and C. Zhang, “Dota: Distributional test-time adaptation of vision-language models,” Advances in Neural Information Processing Systems, vol. 38, pp. 142 772–142 798, 2026.

[29] Q. Dai and S. Yang, “Free on the fly: Enhancing flexibility in test-time adaptation with online em,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9538–9548.

[30] T. Hastie, R. Tibshirani, J. Friedman et al., “The elements of statistical learning,” 2009.

[31] Z. Wang, J. Liang, L. Sheng, R. He, Z. Wang, and T. Tan, “A hardto-beat baseline for training-free clip-based adaptation,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 28 841– 28 859.

[32] J. Li, R. Selvaraju, A. Gotmare, S. Joty, C. Xiong, and S. C. H. Hoi, “Align before fuse: Vision and language representation learning with momentum distillation,” Advances in neural information processing systems, vol. 34, pp. 9694–9705, 2021.

[33] A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, and D. Kiela, “Flava: A foundational language and vision alignment model,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 15 617–15 629.

[34] Z.-Y. Dou, Y. Xu, Z. Gan, J. Wang, S. Wang, L. Wang, C. Zhu, P. Zhang, L. Yuan, N. Peng et al., “An empirical study of training end-to-end vision-and-language transformers,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 18 145–18 155.

[35] J. Li and S. Feng, “Bridging modalities via progressive re-alignment for multimodal test-time adaptation,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2026, pp. 22 931–22 939.

[36] Z. Gao, X. Jiang, X. Xu, F. Shen, Y. Li, and H. T. Shen, “Embracing unimodal aleatoric uncertainty for robust multimodal fusion,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 26 866–26 875.

[37] M. Ma, J. Ren, L. Zhao, D. Testuggine, and X. Peng, “Are multimodal transformers robust to missing modality?” in 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2022, pp. 18 156–18 165.

[38] Y. Wang, Z. Cui, and Y. Li, “Distribution-consistent modal recovering for incomplete multimodal learning,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 21 968– 21 977.

[39] M. Ramazanova, A. Pardo, B. Ghanem, and M. Alfarra, “Test-time adaptation for combating missing modalities in egocentric videos,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 82 755–82 774.

[40] M. K. Reza, A. Prater-Bennette, and M. S. Asif, “Robust multimodal learning with missing modalities via parameter-efficient adaptation,” IEEE transactions on pattern analysis and machine intelligence, vol. 47, no. 2, pp. 742–754, 2024.

[41] H. Cao, Y. Xu, J. Yang, P. Yin, S. Yuan, and L. Xie, “Multimodal continual test-time adaptation for 3d semantic segmentation,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 18 763–18 773.

[42] M. Li, D. Yang, X. Zhao, S. Wang, Y. Wang, K. Yang, M. Sun, D. Kou, Z. Qian, and L. Zhang, “Correlation-decoupled knowledge distillation for multimodal sentiment analysis with incomplete modalities,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 12 458–12 468.

[43] M. Chen, F. Zhang, Z. Zhao, J. Yao, Y. Zhang, and Y. Wang, “Probabilistic conformal distillation for enhancing missing modality robust-

ness,” Advances in Neural Information Processing Systems, vol. 37, pp. 36 218–36 242, 2024.

[44] J. Xu, Y. Li, C. Sun, X. Xu, J. Li, and F. Xu, “Multi-modal test-time adaptation via adaptive probabilistic gaussian calibration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 30 268–30 277.

[45] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, “Tent: Fully test-time adaptation by entropy minimization,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=uXl3bZLkr3c

[46] S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, and M. Tan, “Efficient test-time model adaptation without forgetting,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 17–23 Jul 2022, pp. 16 888–16 905. [Online]. Available: https://proceedings.mlr.press/v162/niu22a.html

[47] S. Niu, J. Wu, Y. Zhang, Z. Wen, Y. Chen, P. Zhao, and M. Tan, “Towards stable test-time adaptation in dynamic wild world,” in International Conference on Learning Representations, 2023.

[48] Y. Iwasawa and Y. Matsuo, “Test-time classifier adjustment module for model-agnostic domain generalization,” Advances in Neural Information Processing Systems, vol. 34, pp. 2427–2440, 2021.

[49] Y. Zhao, J. Luo, X. Luo, J. Huang, J. Yuan, Z. Xiao, and M. Zhang, “Attention bootstrapping for multi-modal test-time adaptation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 21, 2025, pp. 22 849–22 857.

[50] M. Chen, B. Zhang, Z. Han, W. Jiang, Y. Wang, S. Feng, Y. Du, and B. BAO, “Test-time selective adaptation for unimodal distribution shift in multi-modal data,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=6EZMWeV5sH

[51] Y.-L. Lee, Y.-H. Tsai, W.-C. Chiu, and C.-Y. Lee, “Multimodal prompting with missing modalities for visual recognition,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 14 943–14 952.

[52] L. Hu, T. Shi, W. Feng, F. Shang, and L. Wan, “Deep correlated prompting for visual recognition with missing modalities,” Advances in Neural Information Processing Systems, vol. 37, pp. 67 446–67 466, 2024.

[53] S. Zhao, N. A. Ahuja, T. Yu, T. Shen, and V. Narayanan, “Mora: Missing modality low-rank adaptation for visual recognition,” CoRR, vol. abs/2511.06225, 2025. [Online]. Available: https://doi.org/10. 48550/arXiv.2511.06225

[54] Y. Zhang, Y. Kim, Y.-G. Choi, H. Kim, H. Liu, and S. Hong, “Backpropagation-free test-time adaptation via probabilistic gaussian alignment,” arXiv preprint arXiv:2508.15568, 2025.

[55] K. P. Murphy, Probabilistic machine learning: an introduction. MIT press, 2022.

[56] C. K. Williams and C. E. Rasmussen, Gaussian processes for machine learning. MIT press, Cambridge, MA, 2006, vol. 2, no. 3.

[57] Z. Ghahramani and M. Jordan, “Supervised learning from incomplete data via an em approach,” Advances in neural information processing systems, vol. 6, 1993.

[58] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev et al., “The kinetics human action video dataset,” arXiv preprint arXiv:1705.06950, 2017.

[59] H. Chen, W. Xie, A. Vedaldi, and A. Zisserman, “Vggsound: A largescale audio-visual dataset,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 721–725.

[60] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” Proceedings of the International Conference on Learning Representations, 2019.

[61] Y. Gong, A. Rouditchenko, A. H. Liu, D. Harwath, L. Karlinsky, H. Kuehne, and J. R. Glass, “Contrastive audio-visual masked autoencoder,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/ forum?id=QPtMRyk5rb

![](images/9aa03a0c83ae8761f9c78114e0eb395dd6d3b8b9fa163ccba917a487433df7a9.jpg)

Jiangmeng Li received the BS degree from Xiamen University in 2016, the MS degree from New York University in 2018, and the Ph.D. degree from University of Chinese Academy of Sciences in 2023. He is currently an Associate Professor (tenuretrack) at the Institute of Software, Chinese Academy of Sciences. His research interests include multimodal mearning, self-supervised learning, and graph learning. He has published more than 70 papers in journals and conferences such as IEEE TPAMI, IEEE TKDE, IJCV, ICML, NeurIPS, and etc. 7 papers recognized as Spotlight Papers at ICML, NeurIPS, KDD, and AAAI. In academic service, he serves as Area Chair for ICLR 2027, Session Chair for AAAI 2026, Senior Program Committee for AAAI, reviewer or Program Committee for prestigious journals and conferences such as IEEE TPAMI, IJCV, NeurIPS, ICML, and ICLR. He also serves as Editorial Board Member of the international SCI journal Mathematics and International Standard Editor to the International Telecommunication Union (ITU).

![](images/e301639dc686a5601d614a4cca8667fe62ea15787a9733477209e6ac57b8e13a.jpg)

Jinglin Xu received the B.S. degree from Beihang University, Beijing, China, in 2023. She is currently pursuing the Ph.D. degree at the University of Chinese Academy of Sciences and the Institute of Software, Chinese Academy of Sciences, Beijing, China. Her research interests include multimodal representation learning, self-supervised learning, and test-time adaptation.

![](images/3b472d19f88f3cd16f7ea54142bd45614ef8e84325cf5a4e65958d10b9577365.jpg)

Yi Li received the B.Sc degree in the School of Fuzhou University, Fuzhou, China. He is currently working toward the Ph.D degree in University of Chinese Academy of Sciences, Beijing, China. His research interests lie in the domains of multimodal learning and causality, with a particular focus on vision-language representation learning and causal representation learning.

![](images/01b39638d9da89316ebb8bf3e94d9a171ad873670b46a90f5cc25ce14286036c.jpg)

Yumeng Liu received the Ph.D. degree in computer science from the University of Chinese Academy of Sciences and the M.S. degree in computer science from the University of Southern California. He is currently a Senior Engineer at the Institute of Software, Chinese Academy of Sciences. His main research fields are cloud computing, database systems, and artificial intelligence. He has published more than 21 papers in international conferences and journals.

![](images/e558b4c1b02289bf05317b684a9e54b424dd0fcfeb9ccc18c2f6d292015a66c3.jpg)

Jiahuan Zhou received his B.E. degree from the Department of Automation, Tsinghua University, in 2013, and his Ph.D. degree from the Department of Electrical Engineering and Computer Science, Northwestern University, in 2018. He is currently a Tenure-Track Assistant Professor at the Wangxuan Institute of Computer Technology, Peking University. His research interests lie at the intersection of computer vision, multimodal learning, and AI4Science. He has published more than 90 papers in leading international journals and conferences, including Nature Synthesis, Nature Communications, IEEE TPAMI, IJCV, IEEE TIP, CVPR, ICCV, ECCV, ICML, NeurIPS, ICLR, and others. He has served as a Workshop Chair for ICCV 2027, an Area Chair for CVPR, ECCV, ICML, NeurIPS, ICLR, BMVC, ICME, and ICPR, and as a Senior Program Committee Member for AAAI. He also serves as an Associate Editor for Machine Vision and Applications (MVA), and as a Youth Associate Editor for National Materials. His honors and recognitions include the National Overseas Excellent Young Scholars, the Xiaomi-Boya Young Scholar Award, and the Yang Fuqing–Wang Yangyuan Academician Faculty Award Excellence Prize.

![](images/55f0dee22d4c0d5e9bd609ee8effa416e6dd905c9ef277e3bbfe83cb6b09869a.jpg)

Fanjiang Xu received his B.Sc. and M.Sc. degrees in computer science both from National University of Defense Technology and his Ph.D. degree in computer science from Huazhong University of Science and Technology. He is currently a Professor at the Institute of Software, Chinese Academy of Sciences. His research interests include intelligent information processing and integration.

![](images/4b6b1e1a5650615c2ba04a16e76726425e5f07a48b3416647b4f38a4dd669cd4.jpg)

Hui Xiong received his Ph.D. in Computer Science from the University of Minnesota - Twin Cities, USA, in 2005, the B.E. degree in Automation from the University of Science and Technology of China (USTC), Hefei, China, and the M.S. degree in Computer Science from the National University of Singapore (NUS), Singapore. He is a chair professor at the Hong Kong University of Science and Technology (Guangzhou). He is also a Distinguished Professor at Rutgers, the State University of New Jersey, where he received the 2018 Ram Charan Management

Practice Award as the Grand Prix winner from the Harvard Business Review, RBS Dean’s Research Professorship (2016), two-year early promotion/tenure (2009), the Rutgers University Board of Trustees Research Fellowship for Scholarly Excellence (2009), the ICDM-2011 Best Research Paper Award (2011), the Junior Faculty Teaching Excellence Award (2007), Dean’s Award for Meritorious Research (2010, 2011, 2013, 2015) at Rutgers Business School, the 2017 IEEE ICDM Outstanding Service Award (2017), and the AAAI-2021 Best Paper Award (2021). Dr. Xiong is also a Distinguished Guest Professor (Grand Master Chair Professor) at the University of Science and Technology of China (USTC). For his outstanding contributions to data mining and mobile computing, he was elected an ACM Distinguished Scientist in 2014, an IEEE Fellow and an AAAS Fellow in 2020. His general area of research is data and knowledge engineering, with a focus on developing effective and efficient data analysis techniques for emerging data intensive applications. He currently serves as a co-Editor-in-Chief of Encyclopedia of GIS (Springer) and an Associate Editor of IEEE Transactions on Data and Knowledge Engineering (TKDE), IEEE Transactions on Big Data (TBD), ACM Transactions on Knowledge Discovery from Data (TKDD) and ACM Transactions on Management Information Systems (TMIS). He has served regularly on the organization and program committees of numerous conferences, including as a Program Co-Chair of the Industrial and Government Track for the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), a Program Co-Chair for the IEEE 2013 International Conference on Data Mining (ICDM), a General Co-Chair for the IEEE 2015 International Conference on Data Mining (ICDM), and a Program Co-Chair of the Research Track for the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD2018).

![](images/6c809a7a2aaa1ced34885aac5a2e391a7e668f672002e2571e728af694a1e2ba.jpg)

Gang Hua (Fellow, IEEE) received the B.S. and M.S. degrees in automatic control engineering from Xi’an Jiaotong University, Xi’an, China, in 1999 and 2002, respectively, and the Ph.D. degree in electrical engineering and computer science with Northwestern University, Evanston, IL, USA, in 2006. He is currently a Director of Applied Science at Amazon Alexa AI. He is an Associate Editor for TPAMI, IJCV and MVA. He is the General Chair of ICCV’2027, the Program Chair of CVPR’2019&2022. He is the author of more than 280 peer-reviewed publications in prestigious international journals and conferences. He holds 37 U.S. patents and has 15 more U.S. patents pending. He was the recipient of the 2015 IAPR Young Biometrics Investigator Award. He is an IEEE Fellow, IAPR Fellow and an ACM Distinguished Scientist.

## APPENDIX AGAUSSIAN FUSION ESTIMATION AND PREDICTIONAGGREGATION UNDER MODALITY MISSINGNESS

We provide the probabilistic motivation and derivation for the posterior-weighted recovery logits in Eq. (26). Without loss of generality, we consider the case where modality m<sub>2</sub> is unavailable and only the representation $z _ { m }$ 1 of modality $m _ { 1 }$ is observable. The corresponding fusion representation $z _ { F }$ cannot be directly obtained. For clarity, we omit the batch superscript t and sample index i throughout this derivation.

Given only the observable modality-specific representation $z _ { m _ { 1 } }$ , its category posterior can be directly obtained from the corresponding Gaussian mixture as

$$
p _ { m _ { 1 } } ( c | z _ { m _ { 1 } } ) = \frac { \pi _ { m _ { 1 } , c } \mathcal { N } \left( z _ { m _ { 1 } } ; \pmb { \mu } _ { m _ { 1 } , c } , \pmb { \Sigma } _ { m _ { 1 } , c } \right) } { \sum _ { c ^ { \prime } = 1 } ^ { C } \pi _ { m _ { 1 } , c ^ { \prime } } \mathcal { N } \left( z _ { m _ { 1 } } ; \pmb { \mu } _ { m _ { 1 } , c ^ { \prime } } , \pmb { \Sigma } _ { m _ { 1 } , c ^ { \prime } } \right) } .\tag{35}
$$

This posterior characterizes category discrimination solely in the modality-specific representation space. However, the fusion representation $z _ { F }$ normally incorporates multimodal information that is unavailable from $z _ { m }$ alone. To further exploit the category distribution learned in the fusion representation space, we treat $z _ { F }$ as a latent variable and marginalize over its possible values. The resulting recovery prediction is

$$
p _ { m _ { 1 }  F } ( c \mid z _ { m _ { 1 } } ) = \int p ( c \mid z _ { m _ { 1 } } , z _ { F } ) p ( z _ { F } \mid z _ { m _ { 1 } } ) d z _ { F } .\tag{36}
$$

We assume that, conditioned on the fusion representation, the observable modality does not provide additional category information, $c \perp z _ { m _ { 1 } } \mid z _ { F }$ , such that $p \left( c \mid z _ { m _ { 1 } } , z _ { F } \right) ~ =$ $p \left( c \vert z _ { F } \right)$ . Equation (36) therefore becomes

$$
p _ { m _ { 1 }  F } ( c \mid z _ { m _ { 1 } } ) = \int p _ { F } ( c \mid z _ { F } ) p ( z _ { F } \mid z _ { m _ { 1 } } ) d z _ { F } ,\tag{37}
$$

where $p ( z _ { F } \mid z _ { m _ { 1 } } )$ describes the possible fusion representations associated with the observed modality-specific representation.

We introduce the latent category variable k and decompose this distribution according to the law of total probability:

$$
p \left( \boldsymbol { z } _ { F } \mid \boldsymbol { z } _ { m _ { 1 } } \right) = \sum _ { k = 1 } ^ { C } p \left( \boldsymbol { z } _ { F } \mid \boldsymbol { z } _ { m _ { 1 } } , k \right) p _ { m _ { 1 } } \left( k \mid \boldsymbol { z } _ { m _ { 1 } } \right) .\tag{38}
$$

To obtain a tractable form for $p ( z _ { F } \mid z _ { m _ { 1 } } , k )$ , we further model the modality-specific and fusion representations as jointly Gaussian within each category:

$$
\begin{array} { r } { \left[ z _ { m _ { 1 } } \right] \bigg | k \sim { \mathcal N } \left( \left[ \begin{array} { c } { \mu _ { m _ { 1 } , k } } \\ { \mu _ { F , k } } \end{array} \right] , \left[ \begin{array} { c c } { \Sigma _ { m _ { 1 } , k } } & { \Sigma _ { m _ { 1 } F , k } } \\ { \Sigma _ { F m _ { 1 } , k } } & { \Sigma _ { F , k } } \end{array} \right] \right) . } \end{array}\tag{39}
$$

Under this category-conditional joint Gaussian assumption, the conditional distribution of the latent fusion representation is also Gaussian:

$$
\begin{array} { r } { p \left( \boldsymbol { z } _ { F } \mid \boldsymbol { z } _ { m _ { 1 } } , \boldsymbol { k } \right) = \mathcal { N } \left( \boldsymbol { z } _ { F } ; \hat { \boldsymbol { z } } _ { F | m _ { 1 } , \boldsymbol { k } } , \boldsymbol { \Sigma } _ { F | m _ { 1 } , \boldsymbol { k } } \right) , } \end{array}\tag{40}
$$

with conditional mean

$$
\hat { z } _ { F | m _ { 1 } , k } = \pmb { \mu } _ { F , k } + \pmb { \Sigma } _ { F m _ { 1 } , k } \pmb { \Sigma } _ { m _ { 1 } , k } ^ { - 1 } \left( z _ { m _ { 1 } } - \pmb { \mu } _ { m _ { 1 } , k } \right) .\tag{41}
$$

Consequently,

$$
p \left( \boldsymbol { z } _ { F } \mid \boldsymbol { z } _ { m _ { 1 } } \right) = \sum _ { k = 1 } ^ { C } p _ { m _ { 1 } } \left( k \mid \boldsymbol { z } _ { m _ { 1 } } \right) \mathcal { N } \left( \boldsymbol { z } _ { F } ; \hat { \boldsymbol { z } } _ { F | m _ { 1 } , k } , \boldsymbol { \Sigma } _ { F | m _ { 1 } , k } \right) .\tag{42}
$$

Substituting Eq. (42) into Eq. (37) yields

$$
\begin{array} { l } { \displaystyle p _ { m _ { 1 }  F } ( c | z _ { m _ { 1 } } ) } \\ { \displaystyle = \int p _ { F } ( c | z _ { F } ) \sum _ { k = 1 } ^ { C } p _ { m _ { 1 } } ( k | z _ { m _ { 1 } } ) \mathcal { N } ( z _ { F } ; \hat { z } _ { F | m _ { 1 } , k } , \Sigma _ { F | m _ { 1 } , k } ) d z _ { F } } \\ { \displaystyle = \sum _ { k = 1 } ^ { C } p _ { m _ { 1 } } ( k | z _ { m _ { 1 } } ) \int p _ { F } ( c | z _ { F } ) \mathcal { N } ( z _ { F } ; \hat { z } _ { F | m _ { 1 } , k } , \Sigma _ { F | m _ { 1 } , k } ) d z _ { F } } \\ { \displaystyle = \sum _ { k = 1 } ^ { C } p _ { m _ { 1 } } ( k | z _ { m _ { 1 } } ) \mathbb { E } _ { z _ { F } \sim \mathcal { N } ( \hat { z } _ { F | m _ { 1 } , k } , \Sigma _ { F | m _ { 1 } , k } ) } [ p _ { F } ( c \mid z _ { F } ) ] . } \end{array}\tag{43}
$$

When the conditional distribution is sufficiently concentrated around its mean, the latent fusion representation can be approximated by its conditional mean, $z _ { F } \approx \hat { z } _ { F | m _ { 1 } , k }$ , which leads to the plug-in approximation

$$
\mathbb { E } _ { z _ { F } \sim \mathcal { N } \left( \hat { z } _ { F | m _ { 1 } , k } , \pmb { \Sigma } _ { F | m _ { 1 } , k } \right) } \left[ p _ { F } \left( c \mid z _ { F } \right) \right] \approx p _ { F } \left( c \mid \hat { z } _ { F | m _ { 1 } , k } \right) .\tag{44}
$$

Therefore, the recovery prediction can be approximated as

$$
p _ { m _ { 1 }  F } ( c \mid z _ { m _ { 1 } } ) \approx \sum _ { k = 1 } ^ { C } p _ { m _ { 1 } } ( k \mid z _ { m _ { 1 } } ) p _ { F } ( c \mid \hat { z } _ { F | m _ { 1 } , k } )\tag{45}
$$

Equation (45) shows that the prediction associated with the unavailable fusion representation can be approximated by a posterior-weighted combination of category-conditional recovery branches. Specifically, $p _ { m _ { 1 } } ( k \mid z _ { m _ { 1 } } )$ measures the plausibility of category k given the observed modality, while $p _ { F } ( c \mid \hat { z } _ { F | m _ { 1 } , k } )$ gives the posterior probability of output category c for the corresponding recovered representation under the Gaussian model in the fusion space.

Using the log-posterior defined in the main text, these posterior probabilities can be written as

$$
p _ { m _ { 1 } } \left( k \mid z _ { m _ { 1 } } \right) = \frac { \exp \left( g _ { m _ { 1 } , k } \left( z _ { m _ { 1 } } \right) \right) } { \sum _ { d = 1 } ^ { C } \exp \left( g _ { m _ { 1 } , d } \left( z _ { m _ { 1 } } \right) \right) } ,\tag{46}
$$

and

$$
p _ { F } \left( c \mid \hat { z } _ { F | m _ { 1 } , k } \right) = \frac { \exp \left( g _ { F , c } \left( \hat { z } _ { F | m _ { 1 } , k } \right) \right) } { \sum _ { d = 1 } ^ { C } \exp \left( g _ { F , d } \left( \hat { z } _ { F | m _ { 1 } , k } \right) \right) } .\tag{47}
$$

Therefore, the approximated recovery prediction can be interpreted as a mixture of category-conditioned fusion predictions, where each component is weighted by the posterior probability of its corresponding category assumption:

$$
p _ { m _ { 1 }  F } ( c | z _ { m _ { 1 } } ) \approx \sum _ { k = 1 } ^ { C } \frac { \exp { ( g _ { m _ { 1 } , k } ( z _ { m _ { 1 } } ) ) } } { \sum _ { d = 1 } ^ { C } \exp { ( g _ { m _ { 1 } , d } ( z _ { m _ { 1 } } ) ) } } p _ { F } ( c | \hat { z } _ { F | m _ { 1 } , k } ) .\tag{48}
$$

This formulation indicates that the posterior obtained from the observed modality determines the contribution of each category-conditioned recovered fusion representation. Since the log-posteriors preserve the additive category evidence before normalization, we further perform the aggregation in the log-posterior space rather than directly averaging normalized probabilities. Specifically, the recovery logits are defined as

$$
\begin{array} { r l } {  { \pmb { g } _ { m _ { 1 } \to F } \big ( \pmb { z } _ { m _ { 1 } } \big ) = \sum _ { k = 1 } ^ { C } p _ { m _ { 1 } } \big ( k \big | \pmb { z } _ { m _ { 1 } } \big ) \pmb { g } _ { F } \big ( \hat { \pmb { z } } _ { F | m _ { 1 } , k } \big ) } \quad } & { } \\ & { = \sum _ { k = 1 } ^ { C } \frac { \exp \big ( g _ { m _ { 1 } , k } \big ( \pmb { z } _ { m _ { 1 } } \big ) \big ) } { \sum _ { d = 1 } ^ { C } \exp \big ( g _ { m _ { 1 } , d } \big ( \pmb { z } _ { m _ { 1 } } \big ) \big ) } \pmb { g } _ { F } \big ( \hat { \pmb { z } } _ { F | m _ { 1 } , k } \big ) . } \end{array}\tag{49}
$$

(c) Missing

Batch size  
![](images/4f647edec21895c9531989e57f21063af17e5412d06df36da4c0c819f1b40842.jpg)

![](images/6bf27c68df72e14bd4688aef5593de8f806b98995c6593b59bdf6deb57293ad6.jpg)

![](images/9e4ddb313254196d2b075d00adedc660cea33b45f6abba08989863b47192ef23.jpg)  
Fig. 10. Effect of batch size on Kinetics50. (a–c) Mean accuracy under audio corruption, video corruption, and modality-missing settings, respectively. Darker bars and bold values indicate the best performance in each setting, while stars mark the adopted batch size of 16.

This is the recovery term used in the missing-modality calibration described in the main text. The case where modality $m _ { 1 }$ is missing follows analogously by taking $m _ { 2 }$ as the observable modality.

## APPENDIX B

## SENSITIVITY ANALYSIS OF ADAPTATION BATCH SIZE

We investigate the sensitivity of AdaPGC to the adaptation batch size. As shown in Figure 10, increasing the batch size does not consistently improve performance. A batch size of 16 achieves the highest average accuracy under audio corruption and modality missing, while a batch size of 8 performs slightly better under video corruption, with a difference of 0.15 percentage points. Further increasing the batch size to 32 or 64 yields lower average accuracy across all three settings than using 16. These results support a batch size of 16 as a consistent choice across different distribution shifts, balancing performance under modality corruption and missingness. Therefore, we adopt a batch size of 16 in our experiments.