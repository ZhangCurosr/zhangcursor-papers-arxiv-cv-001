# GOOD RANKINGS, POOR CHOICES?RETHINKING LANGUAGE-ONLY VISION–LANGUAGEMODEL SELECTION

Hyuk Kwon, Jaebyeong Jeon, Hyeonseo Jang, Kibok Lee

Yonsei University

{kh12043, jaebyeong98, jhyeonseo715, kibok}@yonsei.ac.kr

## ABSTRACT

Language-Only Vision–Language Model Selection (LOVM) aims to identify the Vision–Language Model (VLM) with the highest zero-shot accuracy on a target task using only textual information. We revisit the existing LOVM evaluation protocol and argue that ranking metrics, such as top-5 recall and Kendall’s rank correlation, do not directly assess the performance of the selected model. We therefore adopt simple regret (SR), defined as the performance gap between the best and selected models, and introduce Outlier-Robust Normalized Regret (ORNR) to account for the varying performance ranges across tasks while reducing sensitivity to outliers. Building on this revised evaluation protocol, we propose Target-Conditioned Expected Source Regret (TC-ESR), a training-free method that selects the model with the lowest similarity-weighted mean regret across source tasks. Across 22 target tasks and 43 VLMs, TC-ESR achieves the lowest mean SR and ORNR among all compared methods. Separately, we identify two sources of target performance leakage in the prior ranking-based evaluation protocol and provide corrections that remove both.

## 1 INTRODUCTION

![](images/fd6a45dbf1121e998a7f5bcdeead0b6185d26cd68ce9f91d3ce4484a3ab9443e.jpg)  
High Scores on Ranking Metrics ≠ Good Model Choices.  
Figure 1: Example based on the top-5 rankings produced by SWAB and TC-ESR on GTSRB. SWAB attains higher $R _ { 5 }$ and $\tau _ { 5 }$ , whereas TC-ESR selects the best-performing model.

Large-scale image–text pretraining has produced a growing family of vision–language models (VLMs) with strong zero-shot transfer across diverse visual recognition tasks (Radford et al., 2021; Zhai et al., 2023). As this model zoo expands, selecting an appropriate VLM for each target task becomes increasingly important, since the best model can vary across tasks (Zohar et al., 2023). Conventional model selection, however, usually requires evaluating the candidates on a labeled dataset for the target task. Constructing such a dataset entails collecting and annotating target images, which can require substantial time and resources.

Language-Only Vision–Language Model Selection (LOVM) was introduced to avoid this requirement (Zohar et al., 2023). It assumes no access to target images, labels for individual target instances, or the performance of any candidate model on the target task. Using textual information of the target task and previously evaluated source tasks, together with candidate model accuracies on the source tasks, an LOVM method predicts a ranking of the candidate models. Since prior LOVM method typically induce a ranking over candidate models, they are commonly evaluated by comparing the predicted ranking with the ground-truth ranking determined by target-task accuracy. In particular, prior work primarily evaluates predicted rankings using top-5 recall and Kendall’s rank correlation (Zohar et al., 2023; Yi et al., 2024), which measure top-5 set recovery and relative ordering. However, these metrics do not directly assess the performance of the selected model on the target task. As Figure 1 illustrates, higher scores on both ranking metrics need not lead to the selection of a model with higher target accuracy.

To align evaluation with the goal of selecting the best model for the target task from a set of candidate models (Raschka, 2020; Karimi et al., 2021), we directly compare the selected model with the best candidate. Accordingly, we adopt simple regret (SR) to measure the gap between their target accuracies (Bubeck et al., 2009). However, the same SR value can have different implications across tasks because candidate accuracies vary over different ranges. Normalizing SR by the range of candidate accuracies can make regret values more comparable across tasks, but using the full range is sensitive to extreme candidates: a single unusually weak model can inflate the range and make the normalized value appear artificially small. We therefore introduce Outlier-Robust Normalized Regret (ORNR), which normalizes SR using a robust performance range computed separately for each task. ORNR complements SR with a relative measure of the accuracy gap, making regret values more comparable across tasks while reducing the influence of unusually weak candidate models.

This focus on the performance of the selected model also motivates using simple regret as a criterion for model selection. We first formulate Expected Source Regret (ESR), a simple training-free baseline that selects the model with the lowest mean simple regret across source tasks. Although ESR attains lower mean regret than prior LOVM methods in our experiments, it gives all source tasks equal weight and therefore does not account for their relevance to the target. Semantic similarity between class names alone may not fully capture this relevance, since transferability can depend on both domain and task differences (Tan et al., 2021). We therefore extend ESR to Target-Conditioned Expected Source Regret (TC-ESR), which measures source–target similarity along visual domain and label granularity in addition to label semantics, using textual task information alone, and weights source tasks accordingly. Across 22 target tasks and 43 VLMs, TC-ESR achieves the lowest mean SR and ORNR among all compared methods.

Separately, we identify and correct two sources of target performance leakage in prior evaluation protocols, both arising from the failure to separate the source and target tasks: First, for each target task, prior methods perform feature selection using all tasks, including the evaluation target. We address this by extending the leave-one-out strategy to feature selection. Second, ImageNet-1K serves as an evaluation target, even though the ImageNet-1K accuracies of candidate models are used as input features. We eliminate this leakage by excluding ImageNet-1K from the target tasks.

We summarize our contributions as follows:

• Evaluation metrics. We adopt SR to quantify the performance gap between the best and selected models, and introduce ORNR to account for the varying performance ranges across tasks while reducing sensitivity to outliers.

• Evaluation protocol corrections. We identify and correct two sources of target performance leakage in the prior protocol by performing feature selection exclusively on source tasks and excluding ImageNet-1K from the evaluation targets.

• Selection method. We propose TC-ESR, which uses only textual task information to measure the similarity between source and target tasks along three axes: visual domain, label semantics, and label granularity; it then weights source tasks by these similarities when aggregating the simple regret of each candidate model.

## 2 PRELIMINARIES

## 2.1 PROBLEM SETUP

Given a VLM zoo M comprising M candidate models, Language-Only Vision–Language Model Selection (LOVM) aims to identify the best-performing VLM on a target task t using only textual task information, without access to target images or instance-level labels. The available textual task information consists of a domain and task description $x _ { t }$ and a set of class names $\mathcal { C } _ { t }$ . In the conventional LOVM setting (Zohar et al., 2023; Yi et al., 2024), candidate VLMs are evaluated on a set of source tasks $S _ { t }$ that excludes the target task t. LOVM can then be performed by measuring the similarity between the source tasks $s \in S _ { t }$ and the target task t, and estimating the relative performance of the candidate VLMs on t by aggregating their performance across $S _ { t }$ . For evaluation, a leave-one-out strategy is applied to a set of tasks, with each task serving as t in turn and the remaining tasks as $S _ { t }$

## 2.2 EXISTING LOVM METHODS

ModelGPT is the first LOVM method introduced by Zohar et al. (2023). It constructs features for each candidate VLM using textual proxies for target images, and employs a learnable ranker model to predict the ranking of candidate VLMs on the target task. SWAB (Yi et al., 2024) is a subsequent method developed under the same protocol, which uses optimal transport to construct features that capture modality and capability gaps, and further transfers rankings from relevant source classes to rank models.

## 3 RETHINKING LOVM EVALUATION

## 3.1 RANKING-BASED EVALUATION

Prior work evaluates LOVM methods using Top-5 Recall $( R _ { 5 } )$ and Kendall’s rank correlation $\left( \tau _ { 5 } \right)$ $R _ { 5 }$ measures the proportion of ground-truth top-5 models correctly identified, whereas $\tau _ { 5 }$ assesses the relative ordering of models shared between the predicted and ground-truth top-5 rankings. Although these metrics provide useful diagnostics of LOVM results in terms of ranking quality, they do not directly assess the performance of the selected model on the target task. In practical model selection scenarios, only the top-ranked model is deployed rather than a ranked set; therefore, these metrics fail to quantify how much the selected model underperforms the best model, as illustrated in Figure 1. Table 5 shows that this discrepancy is systematic across methods and targets.

## 3.2 TARGET PERFORMANCE LEAKAGE

We first examine whether the original LOVM evaluation protocol suffers from information leakage, specifically whether the set of source tasks $S _ { t }$ and the target task t are properly separated during evaluation. First, for each t, ModelGPT and SWAB perform feature selection to construct the input features fed to their ranker models based on mean ranking performance over the set of source tasks. However, this set includes t during feature selection, i.e., $S _ { t } \cup \{ t \}$ . Consequently, the ground-truth model ranking on t influences feature selection, even though t is intended to be held out when fitting the ranker models. Second, both methods use the ImageNet-1K accuracies of the candidate models as input features to their ranker models, while ImageNet-1K itself also serves as an evaluation target under the original protocol. In this case, the input feature reveals the accuracies of the candidate models on $t ,$ and consequently, their ground-truth ranking.

To eliminate the first leakage, we extend the leave-one-out strategy to feature selection, ensuring that the feature subset for t is selected using only $S _ { t }$ , without access to t. To eliminate the second leakage, we exclude ImageNet-1K from the target task candidates, while retaining it as a source task and as a dataset for computing input features for ModelGPT and SWAB. As shown in Table $^ { 6 , }$ both corrections lower $R _ { 5 }$ and $\tau _ { 5 }$ of ModelGPT and SWAB, indicating that the original protocol overestimates ranking performance relative to the revised protocol.

## 3.3 REGRET-BASED EVALUATION

Because model selection aims to identify the candidate with the highest performance on the target task (Raschka, 2020; Karimi et al., 2021), we evaluate the selected model by comparing its performance with that of the best candidate. We use simple regret (SR) to quantify this absolute performance gap (Bubeck et al., 2009; Renggli et al., 2022). We further introduce Outlier-Robust Normalized Regret (ORNR) to normalize the gap using a robust performance range defined per task.

Simple Regret. Let $m _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { m \in \mathcal { M } } a _ { t } ( m )$ denote a best-performing model for a task $t ,$ where $a _ { t } ( m )$ is the performance of m on t. We then define the simple regret of selected model

![](images/408a5e53db0c111e2aefdf4a3e8add423a2d8d6818d2400872ead189acb1ae24.jpg)  
Figure 2: Motivation for ORNR. (a) Similar SR can occupy different fractions of the accuracy range across tasks. (b) An unusually weak model can inflate the full range and distort normalization.

m $\in \mathcal { M }$ on task t as

$$
\mathrm { S R } _ { t } ( m ) = a _ { t } ( m _ { t } ^ { \star } ) - a _ { t } ( m ) .\tag{1}
$$

Because $a _ { t } ( m ) \in [ 0 , 1 ] , \operatorname { S R } _ { t } ( m ) \in [ 0 , 1 ] ;$ zero indicates that a best-performing model is selected, while larger values indicate a greater performance gap.

Outlier-Robust Normalized Regret. Although SR directly quantifies the performance loss incurred by the selected model, it does not account for variation in the range of candidate model accuracies across tasks. As illustrated in Figure 2(a), similar SR values can have different implications across tasks, representing a small fraction of the accuracy range on one task but a large fraction on another. Consequently, SR does not provide a common scale for comparing model selection outcomes across tasks.

A straightforward approach is min–max normalization, which divides SR by the range of candidate model accuracies of task t: $\begin{array} { r } { \Delta _ { t } ^ { \mathrm { m m } } = \operatorname* { m a x } _ { u \in \mathcal { M } } a _ { t } ( u ) - \operatorname* { m i n } _ { u \in \mathcal { M } } a _ { t } ( u ) } \end{array}$ . However, min–max normalization is sensitive to a single model with unusually low accuracy. As Figure 2(b) illustrates, a model with unusually low accuracy can increase the normalization denominator and thereby reduce the normalized value, even though the performance gap between model m and the best model remains unchanged.

We therefore replace the minimum accuracy with the lower whisker of a box-and-whisker plot (Tukey, 1977). Let $Q _ { 1 , t }$ and $Q _ { 3 , t }$ denote the first and third quartiles of $\{ a _ { t } ( u ) : u \in \mathcal { M } \}$ Following Tukey’s boxplot convention, we use the 1.5 IQR rule, a standard nonparametric criterion for flagging potential outliers that requires neither a distributional assumption nor tuning for individual tasks. We define the interquartile range $\mathrm { I Q R } _ { t } = Q _ { 3 , t } - Q _ { 1 , t }$ and the lower fence $F _ { t } ^ { - } = Q _ { 1 , t } - 1 . 5 \mathrm { I Q R } _ { t } .$ . Candidate accuracies below $F _ { t } ^ { - }$ are treated as potential low outliers. We then define the lower whisker $\begin{array} { r } { \ell _ { t } \ = \ \operatorname* { m i n } _ { u \in \mathcal { M } : a _ { t } ( u ) \geq F _ { t } ^ { - } } a _ { t } ( u ) } \end{array}$ and the robust performance range $\Delta _ { t } ^ { \mathrm { r o b } } = a _ { t } ( m _ { t } ^ { \star } ) - \ell _ { t }$ . Here, $\ell _ { t }$ is the smallest observed candidate accuracy at or above the lower fence, and $\Delta _ { t } ^ { \mathrm { r o b } }$ is the resulting range between the lower whisker and the accuracy of the best model. Accordingly, we define ORNR as SR normalized by this robust range:

$$
\mathrm { O R N R } _ { t } ( m ) = \frac { \mathrm { S R } _ { t } ( m ) } { \operatorname* { m a x } \bigl ( \Delta _ { t } ^ { \mathrm { r o b } } , \varepsilon \bigr ) } ,\tag{2}
$$

where $\varepsilon > 0$ is a small numerical constant that prevents division by zero. The lower fence and lower whisker are used only to compute the normalization denominator; they do not remove any model from the candidate set or restrict which model may be selected. ORNR can exceed one when the selected model’s accuracy falls below the lower whisker, because its gap from the best model then exceeds the robust performance range. We report SR and ORNR together: SR measures the absolute accuracy gap, whereas ORNR scales this gap by the robust range of candidate model accuracies for each task, making regret values more comparable across tasks. Appendix E.4 confirms that the comparison between methods is unchanged under alternative denominators and that ORNR is substantially less sensitive than min–max normalization to weak candidates, although, like any task-relative normalization, it is not invariant to the composition of the candidate pool.

## 4 METHOD

Building on this revised evaluation, we use simple regret on source tasks as the basis for model selection. Expected Source Regret (ESR) selects the model with the lowest mean simple regret across source tasks. To account for differences in source relevance, we estimate the similarity between each source and the target from text descriptions and class name sets along three axes: visual domain, label semantics, and label granularity. Target-Conditioned Expected Source Regret (TC-ESR) uses these similarities to weight source tasks and selects the model with the lowest weighted mean simple regret. Unlike prior LOVM methods, ESR and TC-ESR do not require training a ranker model.

![](images/9ea80ea21ed90edd9b560337f6974ca22410a311231de842d8c3a79c696a1be6.jpg)  
Figure 3: TC-ESR framework. Text-only task specifications are mapped to three-axis source–target similarities using frozen vocabularies and text embeddings. $\lambda _ { t } ^ { \star }$ is selected by using cross validation over source tasks, and the resulting weights aggregate each model’s source task simple regrets. The model with the lowest TC-ESR score is selected.

## 4.1 EXPECTED SOURCE REGRET

For each source task $s \in S _ { t }$ and candidate model $m \in { \mathcal { M } }$ , let $\mathrm { S R } _ { s } ( m )$ denote the simple regret of selecting m on s. We define the ESR score of model m for target t as its mean simple regret across the available source tasks, and the ESR method selects the candidate with the lowest ESR score:

$$
\mathrm { E S R } _ { t } ( m ) = \frac { 1 } { \vert S _ { t } \vert } \sum _ { s \in S _ { t } } \mathrm { S R } _ { s } ( m ) , \qquad \widehat { m } _ { t } ^ { \mathrm { E S R } } = \arg \operatorname* { m i n } _ { m \in \mathcal { M } } \mathrm { E S R } _ { t } ( m ) .\tag{3}
$$

A lower ESR score indicates a smaller average gap from the best model on each source task. Since the best accuracy on each source task is constant across candidate models, this criterion is equivalent to selecting the model with the highest average source accuracy.

## 4.2 LANGUAGE-ONLY TASK SIMILARITY

ESR assigns equal weight to all source tasks, even though tasks more closely related to the target may provide more informative evidence about relative model performance (Zamir et al., 2018; Achille et al., 2019). SWAB uses class name similarity to identify relevant source classes when transferring model rankings (Yi et al., 2024). However, semantic similarity between class names alone may not fully capture the relevance of the corresponding source task, since transferability can depend on both domain and task differences (Tan et al., 2021). We therefore estimate the similarity of each source task to the target along three axes informed by prior work: visual domain (Cui et al., 2018; Tan et al., 2021), label semantics (Frome et al., 2013), and label granularity (Min et al., 2020). Unlike DCD (Yang et al., 2026), which derives task similarity from unlabeled source and target images, our measure uses only the text descriptions and class name sets of the source and target tasks.

Task Metadata Construction. To support the domain and label granularity components of task similarity, we represent each task using two categorical fields: a domain family for visual domain similarity and a label scale type that indicates whether label granularity can be meaningfully estimated from class name geometry. To ensure that the same set of labels is used for each field acros tasks, we use an LLM to induce a shared vocabulary for each field. This design follows prior work in which LLMs establish a shared label space before assigning labels (Wan et al., 2024; Huang & He, 2025).

Let ${ \mathcal { D } } = S _ { t } \cup \{ t \}$ denote the set containing target task t and all source tasks available for it. In the first stage, the LLM induces a vocabulary for each field, including category labels, definitions, and assignment criteria, from anonymized records $\{ ( x _ { d } , C _ { d } ) \} _ { d \in \mathcal { D } }$ , after which we fix both vocabularies.

Second, for each task, the LLM receives only its anonymized record and the two fixed vocabularies and assigns one category from each. To reduce sensitivity to stochastic outputs, we repeat the category assignment for each task and select the majority category across runs for each field. For each task d $\in { \mathcal { D } } .$ , let $z _ { d } ^ { \mathrm { d o m } }$ and $z _ { d } ^ { \mathrm { s c a l e } }$ denote the resulting domain family and label scale assignments, respectively. We keep these assignments fixed throughout evaluation and compute all similarity scores deterministically from them and the class name embeddings.

Domain Similarity. For each source task $s \in S _ { t }$ , we define its domain similarity to target task t as

$$
D _ { t } ( s ) = \mathbb { I } \big [ z _ { s } ^ { \mathrm { d o m } } = z _ { t } ^ { \mathrm { d o m } } \big ] .\tag{4}
$$

Because the closed vocabulary defines discrete families, we use exact membership rather than graded similarity between family names, as families with semantically similar names need not contain visually similar tasks.

Label Semantic Similarity. For each task $d \in \mathcal { D }$ , let $n _ { d } = | C _ { d } |$ , and let $c _ { d , i }$ denote the class name in $C _ { d }$ indexed by i. Using a fixed text encoder E (e.g., MPNet (Song et al., 2020)), we obtain an $\ell _ { 2 }$ -normalized embedding for each class name: $\mathbf { e } _ { d , i } \overset { - } { = } E ( c _ { d , i } ) / \Vert E ( \bar { c _ { d , i } } ) \Vert _ { 2 } \in \mathbb { R } ^ { p }$ , with embedding dimension $p .$

For source task $s \in S _ { t }$ and target task $t ,$ we define the pairwise class similarity between $c _ { s , i }$ and $c _ { t , j }$ as their cosine similarity rescaled to $[ 0 , 1 ] , \Gamma _ { i j } ^ { s , t } = ( { \bf e } _ { s , i } ^ { \top } { \bf e } _ { t , j } + 1 ) / 2$ . Because the two class name sets may differ in size or semantic scope, we measure coverage in both directions by matching each class name in one task to its most similar counterpart in the other:

$$
\kappa _ { s  t } = \frac { 1 } { n _ { t } } \sum _ { j = 1 } ^ { n _ { t } } \operatorname* { m a x } _ { 1 \leq i \leq n _ { s } } \Gamma _ { i j } ^ { s , t } , \kappa _ { t  s } = \frac { 1 } { n _ { s } } \sum _ { i = 1 } ^ { n _ { s } } \operatorname* { m a x } _ { 1 \leq j \leq n _ { t } } \Gamma _ { i j } ^ { s , t } .\tag{5}
$$

Here, $\kappa _ { s  t }$ averages the best source match for each target class name, while $\kappa _ { t  s }$ measures the reverse. Their mean defines the label semantic similarity:

$$
L _ { t } ( s ) = \frac { \kappa _ { s  t } + \kappa _ { t  s } } { 2 } .\tag{6}
$$

When one class name set corresponds to only a subset of the other, directional coverage may be high in one direction but low in the reverse. Averaging the two directions therefore prevents such partial correspondence from producing a high $L _ { t } ( s )$

Label Granularity Similarity. Prior work characterizes dataset granularity using distances between classes (Cui et al., 2019). Since LOVM provides no target images, we estimate granularity from distances between the class name embeddings defined above. This estimate is intended for taxonomies of semantic categories, rather than labels that denote ordered levels or bins of an underlying attribute. We therefore set $\eta _ { d } = 1$ when $z _ { d } ^ { \mathrm { s c a l e } }$ indicates such a taxonomy and $\eta _ { d } = 0$ otherwise. For a task d with $\eta _ { d } = 1$ and $n _ { d }$ class names, let $\mathcal { P } _ { d } = \{ 1 - \mathbf { e } _ { d , i } ^ { \top } \mathbf { e } _ { d , j } : 1 \leq i < j \leq n _ { d } \}$ denote the collection of cosine distances between distinct class names.

Let $\widetilde { \mu } _ { d }$ denote the median of $\mathcal { P } _ { d } .$ , which provides a robust summary of the pairwise distances between class name embeddings within task $d .$ For a target task with $\eta _ { t } = 1$ , we define its label granularity similarity to source task s by comparing their median pairwise embedding distances:

$$
G _ { t } ( s ) = \left\{ \begin{array} { l l } { 1 - \displaystyle \frac { \left| \widetilde { \mu } _ { s } - \widetilde { \mu } _ { t } \right| } { \operatorname* { m a x } \left( \left| \widetilde { \mu } _ { s } \right| , \left| \widetilde { \mu } _ { t } \right| , \varepsilon \right) } , } & { \eta _ { s } = 1 , } \\ { 0 , } & { \eta _ { s } = 0 . } \end{array} \right.\tag{7}
$$

Here, ε is the numerical constant defined above. Thus, $G _ { t } ( s )$ compares the median pairwise embedding distances of the two tasks when the source task also has a taxonomy of semantic categories, and is zero otherwise.

Aggregated Task Similarity. We define $S _ { t } ( s )$ as a target-conditioned task similarity score for source task s and target task t, where larger values indicate greater similarity. We average the components applicable to the target with equal weights, avoiding additional hyperparameters:

$$
S _ { t } ( s ) = \left\{ \begin{array} { l l } { \displaystyle \frac { D _ { t } ( s ) + L _ { t } ( s ) + G _ { t } ( s ) } { 3 } , } & { \eta _ { t } = 1 , } \\ { \displaystyle \frac { D _ { t } ( s ) + L _ { t } ( s ) } { 2 } , } & { \eta _ { t } = 0 . } \end{array} \right.\tag{8}
$$

Because the included components depend on the target, $S _ { t } ( s )$ need not equal $S _ { s } ( t )$ , consistent with asymmetric transfer relationships (Zamir et al., 2018; Daroya et al., 2024).

## 4.3 TARGET-CONDITIONED EXPECTED SOURCE REGRET

TC-ESR preserves ESR’s modelwise simple regret on source tasks but replaces uniform averaging over source tasks with target-conditioned weighting. For a temperature parameter $\lambda > 0$ , we convert the task similarities into normalized source weights and select the model with the lowest weighted source regret:

$$
w _ { t } ( s ; \lambda ) = \frac { \exp \left( S _ { t } ( s ) / \lambda \right) } { \sum _ { u \in S _ { t } } \exp \left( S _ { t } ( u ) / \lambda \right) } , \qquad \widehat { m } _ { t } ^ { \mathrm { T C - E S R } } = \arg \operatorname* { m i n } _ { m \in \mathcal { M } } \sum _ { s \in S _ { t } } w _ { t } ( s ; \lambda _ { t } ^ { \star } ) \operatorname { S R } _ { s } ( m ) .\tag{9}
$$

The temperature λ controls the concentration of the source weights and is a hyperparameter, which we select for each target from a grid H by leave-one-out validation over the source tasks. Consistent with the evaluation targets, ImageNet-1K is used only as a source and not as a validation task, so the validation tasks are $V _ { t } = S _ { t } \setminus \{ \bar { t _ { \mathrm { I N } } } \}$ , where t denotes ImageNet-1K. For each $\lambda \in \mathcal H$ and each $v \in$ $V _ { t } .$ , we treat v as the target, compute the weights $w _ { v \mid t } ( s ; \lambda )$ in Equation 9 over $S _ { t , v } = S _ { t } \setminus \{ v \}$ , and select $\begin{array} { r } { \widehat { m } _ { v | t } ( \lambda ) = \arg \operatorname* { m i n } _ { m \in \mathcal { M } } \sum _ { s \in S _ { t , v } } w _ { v | t } ( s ; \lambda ) \dot { \mathrm { S R } } _ { s } ( m ) } \end{array}$ . We then set $\lambda _ { t } ^ { \star } = \arg \operatorname* { m i n } _ { \lambda \in \mathcal { H } } J _ { t } ( \lambda )$ where $\begin{array} { r } { J _ { t } ( \lambda ) = \frac { 1 } { | V _ { t } | } \sum _ { v \in V _ { t } } \mathrm { O R N R } _ { v } \big ( \widehat { m } _ { v | t } ( \lambda ) \big ) } \end{array}$ is the mean ORNR.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Datasets and Model Zoo. Following SWAB (Yi et al., 2024), we use the same pool of 23 datasets and 43 candidate VLMs, covering diverse visual recognition tasks and model architectures. Complete lists of datasets and models are provided in Appendix B.

Baselines. We compare TC-ESR with the ImageNet baseline (INB) and ModelGPT (Zohar et al., 2023), as well as AvgRank and SWAB (Yi et al., 2024). INB ranks the candidates by their ImageNet-1K accuracies, whereas AvgRank ranks them by their mean rank across source classes. We use SWAB’s publicly available implementation for all baselines. ModelGPT and SWAB select their input features by $R _ { 5 } + \tau _ { 5 }$ , following their original protocols.

Evaluation Protocol. Following prior LOVM work (Zohar et al., 2023; Yi et al., 2024), we perform leave-one-out evaluation over the 22 targets excluding ImageNet-1K under our revised evaluation protocol. For each target, all remaining datasets, including ImageNet-1K, form the source pool. We evaluate the selected model using SR and ORNR, defined in equation 1 and equation 2, respectively, and report macro-averages across targets. For readability, both metrics are multiplied by 100 and reported in percentage points (%p) and percent (%), respectively.

Implementation Details. We use $\mathtt { g p t - 5 . 4 - n a n o }$ to induce the closed vocabularies and assign a category to each task. We use $\mathtt { a l l - m p n e t - b a s e - v 2 }$ as the fixed text encoder E. The sensitivity of TC-ESR to the LLM used for metadata construction and to the text encoder is analyzed in Appendix E.1. For each target, we select λ from $\mathcal { H } = \left. 0 . 0 3 , 0 . 0 5 , 0 . 0 7 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 , 0 . 5 \dot { 0 } , 0 . 7 0 \right.$ by cross validation over source tasks. We use $\varepsilon = 1 \stackrel { \sim } { 0 } ^ { - 1 2 }$ wherever needed to prevent division by zero. We use the zero-shot accuracies released by SWAB (Yi et al., 2024) to compute source regrets and evaluate the selected models. Additional details are provided in Appendix C.

Table 1: Mean SR and ORNR over 22 target tasks.
<table><tr><td>Method</td><td>SR (%p) ↓ ORNR (%) ↓</td></tr><tr><td>INB</td><td>7.23 24.66</td></tr><tr><td>AvgRank</td><td>7.35 25.69</td></tr><tr><td>ModelGPT</td><td>7.11 24.25</td></tr><tr><td>SWAB</td><td>7.39 25.81</td></tr><tr><td>ESR</td><td>5.83 22.56</td></tr><tr><td>TC-ESR</td><td>4.66 17.81</td></tr><tr><td>Best fixed model</td><td>5.52 21.55</td></tr></table>

Table 2: Ablation of the three task similarity axes used in TC-ESR.
<table><tr><td>Task Similarity Axes</td><td colspan="2">Performance</td></tr><tr><td colspan="3">Dt(s) Lt(s) Gt(s) SR (%p) ↓| ORNR (%) ↓</td></tr><tr><td>√ √</td><td>√ 4.66</td><td>17.81</td></tr><tr><td>x √</td><td>√ 5.91</td><td>23.05</td></tr><tr><td>√ x</td><td>√ 7.39</td><td>26.39</td></tr><tr><td>√ √</td><td>X 8.22</td><td>28.58</td></tr></table>

Table 3: Effect of source weight assignment. Permuted: averaged over 10K permutations.  
Table 4: Comparison with alternative uses of the same task information.
<table><tr><td>Weight Assignment</td><td>SR (%p) ↓</td><td>ORNR (%) ↓</td></tr><tr><td>Uniform (ESR)</td><td>5.83</td><td>22.56</td></tr><tr><td>Similarity (TC-ESR)</td><td>4.66</td><td>17.81</td></tr><tr><td>Permuted</td><td>9.08</td><td>30.64</td></tr></table>

<table><tr><td colspan="3">Method SR (%p) ↓ ORNR (%) ↓</td></tr><tr><td>Uniform (ESR)</td><td>5.83</td><td>22.56</td></tr><tr><td>Nearest source</td><td>6.43</td><td>24.75</td></tr><tr><td>Top-k sources</td><td>9.31</td><td>29.56</td></tr><tr><td>Description embedding</td><td>5.95</td><td>23.25</td></tr><tr><td>Class names only</td><td>6.52</td><td>23.26</td></tr><tr><td>Learned weights</td><td>5.36</td><td>22.70</td></tr><tr><td>TC-ESR</td><td>4.66</td><td>17.81</td></tr></table>

## 5.2 MAIN RESULTS

As shown in Table 1, ESR attains lower mean SR and ORNR than all prior methods, suggesting that simple regret on source tasks provides more useful evidence for model selection than source ranks. TC-ESR, which differs from ESR only in its source weights, further reduces both metrics, indicating that emphasizing source tasks more similar to the target improves model selection over uniform aggregation. TC-ESR and ESR remain the best and second-best methods when ModelGPT and SWAB instead select their input features by SR or ORNR, as shown in Appendix E.5. TC-ESR also attains lower mean SR and ORNR than the best fixed model, the single candidate with the highest mean target accuracy across targets, although this reference requires target accuracies unavailable in LOVM. Appendix F further analyzes where target conditioning improves model selection.

## 5.3 ABLATION STUDIES

Analysis of Task Similarity Components. Table 2 evaluates the three axes of the task similarity measure over 22 target tasks, with label granularity applied to the 13 targets for which it is applicable. Removing any component degrades both metrics, with the largest degradation observed when label granularity is removed. Using all three axes performs best, indicating that the three axes make complementary contributions within the TC-ESR procedure.

Effect of Source Weight Assignment. To test whether TC-ESR benefits from unequal weights themselves or from assigning larger weights to more similar sources, we randomly permute each target’s weights across source tasks, keeping the set of weight values unchanged. As shown in Table 3, similarity-based assignment outperforms uniform weighting, whereas random assignment falls below even uniform weighting. Thus, the improvement comes from assigning larger weights to more similar sources, supporting the premise of TC-ESR that models performing well on source tasks similar to the target also tend to perform well on the target.

Comparison with Alternative Uses of Task Information. We next compare TC-ESR with alternative uses of the same task information (Table 4; details in Appendix C). These alternatives use the TC-ESR similarity to select only a few sources (Nearest source, Top-k sources), replace it with a single task embedding (Description embedding) or with label semantic similarity alone (Class names only), or learn the weights of the three axes by source-only validation instead of averaging them (Learned weights). None of them outperforms ESR on both metrics. Restricting aggregation to the most similar sources performs worse than weighting all sources, suggesting that less similar sources also provide useful evidence, and the simpler similarities do not improve over uniform weighting. Together with Table 3, these results show that the gains of TC-ESR depend on how the similarity is defined and applied across all source tasks, not merely on similarity-based weighting.

Table 5: Agreement between ranking metrics and SR over six methods and 22 targets.
<table><tr><td>Ranking metric</td><td>Rank corr. with SR</td><td>Disagreement (%)</td><td>Mean SR loss (%p)</td><td>Median SR loss (%p)</td></tr><tr><td> $R _ { 5 }$ </td><td>-0.83</td><td>25.7</td><td>3.32</td><td>1.82</td></tr><tr><td> $\tau _ { 5 }$ </td><td>-0.47</td><td>43.1</td><td>2.46</td><td>1.05</td></tr></table>

Table 6: Effect of protocol corrections on $R _ { 5 }$ and $\tau _ { 5 }$ for ModelGPT and SWAB.
<table><tr><td rowspan="2">Leakage removed</td><td colspan="2">ModelGPT</td><td colspan="2">SWAB</td><td rowspan="2">Avg. change</td></tr><tr><td> $R _ { 5 }$  ↑</td><td> $\tau _ { 5 } \uparrow$ </td><td> $R _ { 5 }$  ↑</td><td> $\tau _ { 5 } \uparrow$ </td></tr><tr><td>None (original)</td><td>0.443</td><td>0.267</td><td>0.504</td><td>0.320</td><td></td></tr><tr><td>Feature selection</td><td>0.435</td><td>0.252</td><td>0.496</td><td>0.302</td><td>-3.7%</td></tr><tr><td>Exclude IN-1K</td><td>0.418</td><td>0.233</td><td>0.482</td><td>0.294</td><td>-7.7%</td></tr><tr><td>Both (final)</td><td>0.411</td><td>0.180</td><td>0.474</td><td>0.283</td><td>-14.3%</td></tr></table>

## 5.4 FURTHER ANALYSIS

Ranking Metrics vs. Selection Quality. Table 5 examines whether the discrepancy illustrated in Figure 1 is systematic and how much target accuracy it costs. First, we rank the six LOVM methods in Table 1 by their mean ranking metric and by their mean SR, negating SR so that higher is better for both, and compute the Kendall rank correlation between the two rankings. For $R _ { 5 } ,$ , the correlation is −0.83, so among these six methods, those with higher mean $R _ { 5 }$ selected worse models on average. Second, for each target and each pair of methods, excluding ties, we count a disagreement when the method with the higher ranking score has the higher SR, and define the SR loss as the SR gap between the two methods. $R _ { 5 }$ and $\tau _ { 5 }$ disagree with SR in 25.7% and 43.1% of the comparisons, with mean SR losses of 3.32%p and 2.46%p, so choosing between methods by their ranking scores often favors the one that selects the worse model, and $\tau _ { 5 }$ disagrees nearly as often as a random choice would. This negative relationship is consistent with how the methods are designed: prior methods are built to predict rankings and select their input features by $R _ { 5 } + \tau _ { 5 } ,$ , whereas ESR and TC-ESR directly minimize the regret of the selected model on source tasks. Retuning the prior methods by SR does not reverse this relationship, as shown in Appendix E.5. These results support evaluating LOVM methods by the regret of the selected model rather than by ranking metrics, and suggest that selection methods should target this regret directly.

Effect of the Protocol Corrections. Although SR and ORNR are our primary metrics for evaluating the final model choice, $R _ { 5 }$ and $\tau _ { 5 }$ remain useful for diagnosing how the protocol corrections affect the rankings predicted by prior methods. In Table $^ { 6 , }$ “Feature selection” denotes source only feature selection, whereas “Exclude $\mathrm { I N } { - } 1 \mathrm { K } '$ removes ImageNet-1K from the target set. Avg. change denotes the mean relative change from the original protocol, averaged over $\bar { R _ { 5 } }$ and $\tau _ { 5 }$ for both ModelGPT and SWAB. Each correction lowers both metrics for both methods, and applying both produces the largest decrease in all four cases.

## 6 CONCLUSION

We revisited LOVM evaluation and showed that ranking metrics often disagree with the performance of the selected model, which SR and ORNR measure directly, and corrected two sources of target performance leakage in the prior protocol. Under this revised protocol, our proposed TC-ESR achieves the lowest mean SR and ORNR by weighting the simple regret on source tasks with their similarity to the target, measured solely from text along three axes. These results suggest that LOVM methods should be evaluated and designed around the performance of the selected model rather than the quality of the predicted ranking.

## REFERENCES

Alessandro Achille, Michael Lam, Rahul Tewari, Avinash Ravichandran, Subhransu Maji, Charless Fowlkes, Stefano Soatto, and Pietro Perona. Task2vec: Task embedding for meta-learning. In International Conference on Computer Vision, 2019.

Sebastien Bubeck, R´ emi Munos, and Gilles Stoltz. Pure exploration in multi-armed bandits prob-´ lems. In Algorithmic Learning Theory, 2009.

Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 105(10):1865–1883, 2017. doi: 10.1109/JPROC.2017.2675998.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014.

Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, 2011.

Yin Cui, Yang Song, Chen Sun, Andrew Howard, and Serge Belongie. Large scale fine-grained categorization and domain-specific transfer learning. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

Yin Cui, Zeqi Gu, Dhruv Mahajan, Laurens van der Maaten, Serge Belongie, and Ser-Nam Lim. Measuring dataset granularity, 2019. URL https://arxiv.org/abs/1912.10154.

Rangel Daroya, Aaron Sun, and Subhransu Maji. Task2box: Box embeddings for modeling asymmetric task relationships. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In The IEEE Conference on Computer Vision and Pattern Recognition, 2009.

Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes challenge 2007 (VOC2007) results, 2007. URL http://www.pascal-network.org/challenges/VOC/voc2007/ workshop/index.html.

Andrea Frome, Greg S Corrado, Jon Shlens, Samy Bengio, Jeff Dean, Marc' Aurelio Ranzato, and Tomas Mikolov. Devise: A deep visual-semantic embedding model. In Advances in Neural Information Processing Systems, 2013.

A Geiger, P Lenz, C Stiller, and R Urtasun. Vision meets robotics: The kitti dataset. Int. J. Rob. Res., 32(11):1231–1237, September 2013. ISSN 0278-3649. doi: 10.1177/0278364913491297. URL https://doi.org/10.1177/0278364913491297.

Ian J. Goodfellow, Dumitru Erhan, Pierre Luc Carrier, Aaron Courville, Mehdi Mirza, Ben Hamner, Will Cukierski, Yichuan Tang, David Thaler, Dong-Hyun Lee, Yingbo Zhou, Chetan Ra maiah, Fangxiang Feng, Ruifan Li, Xiaojie Wang, Dimitris Athanasakis, John Shawe-Taylor, Maxim Milakov, John Park, Radu Ionescu, Marius Popescu, Cristian Grozea, James Bergstra, Jingjing Xie, Lukasz Romaszko, Bing Xu, Zhang Chuang, and Yoshua Bengio. Challenge in representation learning: A report on three machine learning contests, 2013. URL https: //arxiv.org/abs/1307.0414.

Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019. doi: 10. 1109/JSTARS.2019.2918242.

Chen Huang and Guoxiu He. Text clustering as classification with llms. In Proceedings of the 2025 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, SIGIR-AP 2025, 2025.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Kaggle and EyePACS. Diabetic retinopathy detection. Kaggle Competition, 2015. URL https: //www.kaggle.com/c/diabetic-retinopathy-detection.

Mohammad Reza Karimi, Nezihe Merve Gurel, Bojan Karla¨ s, Johannes Rausch, Ce Zhang, andˇ Andreas Krause. Online active model selection for pre-trained classifiers. In Arindam Banerjee and Kenji Fukumizu (eds.), Proceedings ofThe 24th International Conference on Artificial Intel ligence and Statistics, volume 130 of Proceedings ofMachine Learning Research, pp. 307–315. PMLR, 13–15 Apr 2021.

Leslie Kish. Survey Sampling. John Wiley & Sons, New York, 1965.

Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In 2013 IEEE International Conference on Computer Vision Workshops, 2013. doi: 10.1109/ICCVW.2013.77.

Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Tech nical report, University of Toronto, 2009.

Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. MNIST handwritten digit database. AT&T Labs, 2010. URL http://yann.lecun.com/exdb/mnist/.

Muyang Li, Yucheng Liu, Jianbo Ma, Elliot Osborne, Bo Han, and Tongliang Liu. Rethinking model selection in vlm through the lens of gromov-wasserstein distance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Shaobo Min, Hongtao Xie, Hantao Yao, Xuran Deng, Zheng-Jun Zha, and Yongdong Zhang. Hierarchical granularity transfer learning. In Advances in Neural Information Processing Systems, 2020.

Yuval Netzer, Tao Wang, Adam Coates, A. Bissacco, Bo Wu, and A. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.

Cuong V. Nguyen, Tal Hassner, Matthias Seeger, and Cedric Archambeau. Leep: A new measure to evaluate transferability of learned representations. In International Conference on Machine Learning, 2020.

Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, 2008.

Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, 2012.

Gabriel Peyre and Marco Cuturi. Computational optimal transport.´ Found. Trends Mach. Learn., 11(5–6):355–607, February 2019. ISSN 1935-8237. doi: 10.1561/2200000073. URL https: //doi.org/10.1561/2200000073.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.

Sebastian Raschka. Model evaluation, model selection, and algorithm selection in machine learning, 2020. URL https://arxiv.org/abs/1811.12808.

Cedric Renggli, Andre Susano Pinto, Luka Rimanic, Joan Puigcerver, Carlos Riquelme, Ce Zhang,´ and Mario Luciˇ c. Which model to transfer? finding the needle in the growing haystack. In´ Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mpnet: Masked and permuted pretraining for language understanding. In Advances in Neural Information Processing Systems, 2020.

Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. The german traffic sign recognition benchmark: A multi-class classification competition. In The 2011 International Joint Conference on Neural Networks, 2011.

Hao-Zhe Tan, Zhi Zhou, Yu-Feng Li, and Lan-Zhe Guo. Vision-language model selection and reuse for downstream adaptation. In International Conference on Machine Learning, 2025.

Yang Tan, Yang Li, and Shao-Lun Huang. Otce: A transferability metric for cross-domain crosstask representations. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

John W. Tukey. Exploratory Data Analysis. Addison-Wesley Publishing Company, Reading, Mass., 1977.

Bastiaan S. Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. Rotation equivariant cnns for digital pathology. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2018, 2018.

Mengting Wan, Tara Safavi, Sujay Kumar Jauhar, Yujin Kim, Scott Counts, Jennifer Neville, Siddharth Suri, Chirag Shah, Ryen W. White, Longqi Yang, Reid Andersen, Georg Buscher, Dhruv Joshi, and Nagu Rangan. Tnt-llm: Text mining at scale with large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, 2024.

Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2010.

Wei Yang, Hong Xie, Tao Tan, Xin Li, Defu Lian, and Enhong Chen. Model specific task similarity for vision language model selection via layer conductance, 2026. URL https://arxiv. org/abs/2602.01346.

Chao Yi, Yu-Hang He, De-Chuan Zhan, and Han-Jia Ye. Bridge the Modality and Capability Gaps in Vision-Language Model Selection. In Advances in Neural Information Processing Systems, 2024.

Kaichao You, Yong Liu, Jianmin Wang, and Mingsheng Long. Logme: Practical assessment of pretrained models for transfer learning. In International Conference on Machine Learning, 2021.

Amir Zamir, Alexander Sax, William Shen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, Lucas Beyer, Olivier Bachem, Michael Tschannen, Marcin Michalski, Olivier Bousquet, Sylvain Gelly, and Neil Houlsby. A large-scale study of representation learning with the visual task adaptation benchmark, 2020. URL https://arxiv.org/abs/1910.04867.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In International Conference on Computer Vision, 2023.

Orr Zohar, Shih-Cheng Huang, Kuan-Chieh Wang, and Serena Yeung. LOVM: Language-Only Vision Model Selection. In Advances in Neural Information Processing Systems, 2023.

## A RELATED WORK

## A.1 MODEL SELECTION AND LOVM

Conventional model selection evaluates candidate models on a labeled target validation set (Raschka, 2020). Transferability estimators such as LEEP and LogME instead use labeled target examples together with candidate predictions or representations to estimate transfer performance without adapting every model (Nguyen et al., 2020; You et al., 2021).

Renggli et al. (2022) formulate pretrained model search under a limited fine tuning budget and evaluate search methods using regret, but their proposed hybrid method uses labeled downstream data. These approaches therefore do not address the setting in which no target images are available.

Language-Only Vision–Language Model Selection (LOVM) selects a pretrained VLM using textual target task information and performance observed on source tasks (Zohar et al., 2023). Model-GPT (Zohar et al., 2023) constructs textual proxies for the target classes and learns a ranker model from evaluated source tasks. SWAB (Yi et al., 2024) uses optimal transport to transfer source class statistics and rankings while accounting for modality and capability gaps. Prior LOVM work evaluates the predicted ranking mainly with Top-5 Recall and Kendall’s τ . We instead measure the performance gap between the selected model and the best candidate, and use task similarity derived from text to weight simple regret on the source tasks.

## A.2 OTHER VLM SELECTION SETTINGS

MLL (Tan et al., 2025) addresses a related setting but differs from LOVM in the information it uses and how it selects models. It first constructs a semantic graph with commonly occurring visual concepts and images from the sample datasets, and performs model labeling by evaluating each candidate VLM on this graph. For a target task, MLL uses these labels to select a model separately for each class and combines the selected models’ predictions during inference. It may therefore use different candidate VLMs for different classes, whereas LOVM selects one candidate VLM for the entire target task. Thus, the output of MLL does not correspond to the single selected model evaluated in the LOVM setting.

DCD (Yang et al., 2026) directly compares with prior LOVM methods, but considers a different information setting in which unlabeled images from both the source and target tasks are available. It processes these images with each candidate visual encoder to construct model specific task representations and estimate directional source relevance. LOVM, by contrast, assumes that no target images are available and relies on textual information about the target task.

Li et al. (2026) select a vision encoder for a fixed LLM before multimodal training and estimate the performance of the resulting VLM. This differs from selecting one complete pretrained VLM for a downstream task without additional training.

## A.3 TASK SIMILARITY FOR SOURCE WEIGHTING

Task relationships have been estimated using trained task models and transfer experiments, labeled examples and probe networks, or source and target data (Zamir et al., 2018; Achille et al., 2019; Tan et al., 2021). Task2Box further models asymmetric task relationships with box embeddings (Daroya et al., 2024). However, these approaches require images, labels, probe models, or observed transfer outcomes.

Prior work suggests that visual domain proximity, label semantics, and label granularity are important factors in transferability (Cui et al., 2018; Tan et al., 2021; Frome et al., 2013; Cui et al., 2019; Min et al., 2020).

Motivated by these findings, TC-ESR estimates task similarity along these three axes using only task and domain descriptions and class name sets. The resulting target conditioned similarity is shared across candidate VLMs and weights their simple regrets on source tasks, without target images or candidate model inference.

Table 7: The task information follows the LOVM and SWAB benchmarks (Zohar et al., 2023; Yi et al., 2024).
<table><tr><td>Dataset</td><td>Classes</td><td>Task</td><td>Domain</td></tr><tr><td>Imagenet (Deng et al., 2009)</td><td>1000</td><td>classification</td><td>natural image</td></tr><tr><td>SUN397 (Xiao et al., 2010)</td><td>397</td><td>scene und.</td><td>natural image</td></tr><tr><td>Country211 (Radford et al., 2021)</td><td>211</td><td>geolocation</td><td>natural image</td></tr><tr><td>Stanford Cars (Krause et al., 2013)</td><td>196</td><td>classification</td><td>natural image</td></tr><tr><td>Flowers102 (Nilsback &amp; Zisserman, 2008)</td><td>102</td><td>classification</td><td>natural image</td></tr><tr><td>CIFAR100 (Krizhevsky &amp; Hinton, 2009)</td><td>100</td><td>classification</td><td>natural image</td></tr><tr><td>DTD (Cimpoi et al., 2014)</td><td>46</td><td>classification</td><td>texture image</td></tr><tr><td>RESISC45 (Cheng et al., 2017)</td><td>45</td><td>classification</td><td>satellite images</td></tr><tr><td>GTSRB (Stallkamp et al., 2011)</td><td>43</td><td>classification</td><td>natural image</td></tr><tr><td>Oxford Pets (Parkhi et al., 2012)</td><td>37</td><td>classification</td><td>natural image</td></tr><tr><td>VOC2007 (Everingham et al., 2007)</td><td>20</td><td>classification</td><td>natural image</td></tr><tr><td>STL10 (Coates et al., 2011)</td><td>10</td><td>classification</td><td>natural image</td></tr><tr><td>EuroSAT (Helber et al., 2019)</td><td>10</td><td>classification</td><td>satellite images</td></tr><tr><td>MNIST (LeCun et al., 2010)</td><td>10</td><td>classification</td><td>hand-writing</td></tr><tr><td>SVHN (Netzer et al., 2011)</td><td>10</td><td>OCR</td><td>natural image</td></tr><tr><td>CLEVR-C (Johnson et al., 2017)</td><td>8</td><td>object counting</td><td>natural image</td></tr><tr><td>CLEVR-D (Johnson et al., 2017)</td><td>8</td><td>distance est.</td><td>natural image</td></tr><tr><td>FER2013 (Goodfellow et al., 2013)</td><td>7</td><td>fac. exp. rec.</td><td>natural image</td></tr><tr><td>DMLab (Zhai et al., 2020)</td><td>6</td><td>distance est.</td><td>synthetic</td></tr><tr><td>Retinopathy (Kaggle &amp; EyePACS, 2015)</td><td>5</td><td>classification</td><td>retina scan</td></tr><tr><td>KITTI (Geiger et al., 2013)</td><td>4</td><td>distance est.</td><td>natural image</td></tr><tr><td>PCam (Veeling et al., 2018)</td><td>2</td><td>classification</td><td>histopathology</td></tr><tr><td>Rendered SST2 (Radford et al., 2021)</td><td>2</td><td>OCR</td><td>text image</td></tr></table>

## B LOVM BENCHMARK AND EVALUATION DETAILS

## B.1 DATASETS

Following prior LOVM work (Zohar et al., 2023; Yi et al., 2024), we use a collection D of 23 tasks covering a diverse range of visual recognition tasks and image domains. Table 7 reports the number of classes and the task and domain descriptions released with the LOVM benchmark. Following prior LOVM studies (Zohar et al., 2023; Yi et al., 2024), we assume access to these descriptions and the class name sets. We use this information to construct the task metadata for TC-ESR. Under our revised evaluation protocol, ImageNet-1K is retained in the source pool but is not used as an evaluation target. Specifically, letting $t _ { \mathrm { I N } }$ denote ImageNet-1K, the evaluation target set is $\mathcal { T } =$ $\mathcal { D } \setminus \{ t _ { \mathrm { I N } } \}$ . For each target $t \in \tau$ , all remaining tasks form the source pool, ${ \mathcal { S } } _ { t } = { \bar { \mathcal { D } } } \setminus \{ t \}$ . Thus, evaluation is performed over 22 target tasks, while ImageNet-1K remains available as a source task for every target.

## B.2 VLM ZOO

We adopt the 43 candidate VLMs evaluated by SWAB (Yi et al., 2024), comprising OpenCLIP, BLIP, and BEiT-3 checkpoints, as summarized in Table 8. We use the zero-shot accuracies released by SWAB for all 43 models and 23 datasets. These accuracies are used as source task performance by the LOVM methods and as target task performance only for final evaluation, so we do not rerun zero-shot inference.

## B.3 RANKING METRICS

In addition to SR and ORNR, we report the ranking metrics used in prior LOVM evaluation: Top-5 Recall and Kendall’s rank correlation (Zohar et al., 2023; Yi et al., 2024). These metrics measure agreement between the predicted and ground truth model rankings. They are retained as supplementary diagnostics of ranking quality, whereas SR and ORNR are our primary metrics for evaluating the model selected at rank one.

For target task t, let $\widehat { \mathbf { r } } _ { t }$ denote the predicted ranking of the candidate models and let $\mathbf { r } _ { t } ^ { \star }$ denote the ground truth ranking obtained by sorting the models by decreasing zero-shot accuracy on t.

Table 8: Detailed information on the 43 candidate models used in the LOVM benchmark. The model information follows the LOVM and SWAB benchmarks (Zohar et al., 2023; Yi et al., 2024).
<table><tr><td>ID</td><td>Model</td><td>Name</td><td>Dataset</td><td>Name</td></tr><tr><td>1</td><td>RN50</td><td>RN50</td><td>openai</td><td>WIT</td></tr><tr><td>2</td><td>RN101</td><td>RN101</td><td>openai</td><td>WIT</td></tr><tr><td>3</td><td>RN50x4</td><td>RN50x4</td><td>openai</td><td>WIT</td></tr><tr><td>4</td><td>RN50x16</td><td>RN50x16</td><td>openai</td><td>WIT</td></tr><tr><td></td><td>RN50x64</td><td>RN50x64</td><td>openai</td><td>WIT</td></tr><tr><td>56</td><td>ViT-B-32</td><td>ViT-B/32</td><td>laion400m_e31</td><td>L400m</td></tr><tr><td>7</td><td>ViT-B-32</td><td>ViT-B/32</td><td>laion400m_e32</td><td></td></tr><tr><td>8</td><td></td><td>ViT-B/32</td><td></td><td>L400m</td></tr><tr><td>9</td><td>ViT-B-32-quickgelu ViT-B-32</td><td>ViT-B/32</td><td>laion400m_e32</td><td>L400m</td></tr><tr><td>10</td><td>ViT-B-32</td><td>ViT-B/32</td><td>openai</td><td>WIT</td></tr><tr><td>11</td><td>ViT-B-32</td><td>ViT-B/32</td><td>laion2b_s34b_b79k</td><td>L2b-b</td></tr><tr><td>12</td><td>ViT-B-16</td><td>ViT-B/16</td><td>laion2b_e16</td><td>L2b-c</td></tr><tr><td>13</td><td>ViT-B-16</td><td>ViT-B/16</td><td>laion400m_e32</td><td>L400m</td></tr><tr><td>14</td><td>ViT-B-16-240</td><td>ViT-B/16-240</td><td>openai</td><td>WIT</td></tr><tr><td>15</td><td>ViT-L-14</td><td>ViT-L/14</td><td>laion400m_e32 laion400m_e31</td><td>L400m</td></tr><tr><td>16</td><td>ViT-L-14</td><td>ViT-L/14</td><td>laion400m_e32</td><td>L400m</td></tr><tr><td>17</td><td>ViT-L-14</td><td>ViT-L/14</td><td></td><td>L400m</td></tr><tr><td>18</td><td>ViT-L-14</td><td>ViT-L/14</td><td>laion2b_s32b_b82k</td><td>L2b-b</td></tr><tr><td>19</td><td>ViT-L-14-336</td><td>ViT-L/14-336</td><td>openai</td><td>WIT</td></tr><tr><td>20</td><td>ViT-G-14</td><td>ViT-G/14</td><td>openai laion2b_s12b_b42k</td><td>WIT</td></tr><tr><td>21</td><td>ViT-G-14</td><td>ViT-G/14</td><td>laion2b_s34b_b88k</td><td>L2b-a</td></tr><tr><td>22</td><td>ViT-H-14</td><td>ViT-H/14</td><td>laion2b_s32b_b79k</td><td>L2b-a</td></tr><tr><td>23</td><td>coca_ViT-B-32</td><td>CoCa-ViT-B/32</td><td>laion2b_s13b_b90k</td><td>L2b-b</td></tr><tr><td>24</td><td>coca_ViT-B-32</td><td>CoCa-ViT-B/32</td><td>mscoco_finetuned_laion2b_s13b_b90k</td><td>L2b-c</td></tr><tr><td>25</td><td>coca_ViT-L-14</td><td>CoCa-ViT-L/14</td><td>laion2b_s13b_b90k</td><td>L2b-c + coco</td></tr><tr><td>26</td><td>coca_ViT-L-14</td><td>CoCa-ViT-L/14</td><td>mscoco_finetuned_laion2b_s13b_b90k</td><td>L2b-c</td></tr><tr><td>27</td><td>convnext_base</td><td>ConvNEXT-B</td><td>laion400m_s13b_b51k</td><td>L2b-c + coco</td></tr><tr><td>28</td><td>convnext_base_w</td><td>ConvNEXT-BW</td><td>laion2b_s13b_b82k</td><td>L400m-c</td></tr><tr><td>29</td><td>convnext_base_w</td><td>ConvNEXT-BW</td><td>laion2b_s13b_b82k_augreg</td><td>L2b-d</td></tr><tr><td>30</td><td>convnext_base_w</td><td>ConvNEXT-BW</td><td>laion_aesthetic_s13b_b82k</td><td>L2b-e</td></tr><tr><td>31</td><td>convnext_base_w_320</td><td>ConvNEXT-BW-320</td><td>laion_aesthetic_s13b_b82k</td><td>L2b-f</td></tr><tr><td>32</td><td>convnext_base_w_320</td><td>ConvNEXT-BW-320</td><td>laion_aesthetic_s13b_b82k_augreg</td><td>L2b-f</td></tr><tr><td>33</td><td>convnext_large_d</td><td>ConvNEXT-LD</td><td>laion2b_s26b_b102k_augreg</td><td>L2b-g</td></tr><tr><td>34</td><td>convnext_large_d_320</td><td>ConvNEXT-LD-320</td><td>laion2b_s29b_b131k_ft</td><td>L2b-h L2b-i</td></tr><tr><td>35</td><td>convnext_large_d_320</td><td>ConvNEXT-LD-320</td><td>laion2b_s29b_b131k_ft_soup</td><td></td></tr><tr><td>36</td><td>BLIP_retrieval_base_coco</td><td>BLIP_retrieval_base_coco</td><td>COCO+VG+CC+CC12M+SBU</td><td>L2b-j BLIP-Dataset</td></tr><tr><td>37</td><td>BLIP_retrieval_base_f30k</td><td>BLIP_retrieval_base_f30k</td><td>COCO+VG+CC+CC12M+SBU+Flickr30k</td><td>BLIP-Dataset-f</td></tr><tr><td>38</td><td>BLIP_retrieval_large_coco</td><td>BLIP_retrieval_large_coco</td><td>COCO+VG+CC+CC12M+SBU</td><td>BLIP-Dataset</td></tr><tr><td>39</td><td>BLIP_retrieval_large_f30k</td><td>BLIP_retrieval_large_f30k</td><td>COCO+VG+CC+CC12M+SBU+Flickr30k</td><td>BLIP-Dataset-f</td></tr><tr><td>40</td><td>BEiT-3_retrieval_base_coco</td><td>BEiT-3_retrieval_base_coco</td><td>CC12M+CC3M+SBU+COCO+VG</td><td>BEiT-Dataset</td></tr><tr><td>41</td><td>BEiT-3_retrieval_base_f30k</td><td>BEiT-3_retrieval_base_f30k</td><td>CC12M+CC3M+SBU+COCO+VG+Flickr30k</td><td>BEiT-Dataset-f</td></tr><tr><td>42</td><td>BEiT-3_retrieval_large_coco</td><td>BEiT-3_retrieval_large_coco</td><td>CC12M+CC3M+SBU+COCO+VG</td><td>BEiT-Dataset</td></tr><tr><td>43</td><td>BEiT-3_retrieval_large_f30k</td><td>BEiT-3_retrieval_large_f30k</td><td>CC12M+CC3M+SBU+COCO+VG+Flickr30k</td><td>BEiT-Dataset-f</td></tr></table>

For both rankings, rank one denotes the most highly ranked model. For methods defined through a score to be minimized, including ESR and TC-ESR, the predicted ranking is obtained by sorting candidate models in increasing order of their scores. For any ranking r, define its top-5 model set as

$$
\mathcal { T } _ { 5 } ( \mathbf { r } ) = \left\{ m \in \mathcal { M } : r ( m ) \leq 5 \right\} .\tag{10}
$$

The models shared by the predicted and ground truth top-5 sets are

$$
\mathcal { T } _ { t } = \mathcal { T } _ { 5 } ( \widehat { \mathbf { r } } _ { t } ) \cap \mathcal { T } _ { 5 } ( \mathbf { r } _ { t } ^ { \star } ) .\tag{11}
$$

Top-5 Recall. Top-5 recall measures the proportion of the ground truth top-5 models recovered in the predicted top-5 set:

$$
R _ { 5 } ( t ) = \frac { | \mathscr { T } _ { t } | } { 5 } .\tag{12}
$$

Thus, $R _ { 5 } ( t ) \in [ 0 , 1 ]$ , with a value of one indicating that the predicted and ground truth top-5 sets contain the same models. The metric evaluates set recovery and does not depend on the ordering of the recovered models.

Kendall’s Rank Correlation. Kendall’s rank correlation measures agreement between the relative orderings of the models shared by the two top-5 sets. Following the released LOVM implementation, it is computed only over $\mathcal { T } _ { t }$

$$
\tau _ { 5 } ( t ) = \left\{ \begin{array} { l l } { \tau ( \widehat { \mathbf { r } } _ { t } | _ { \mathbb { Z } _ { t } } , \mathbf { r } _ { t } ^ { \star } | _ { \mathbb { Z } _ { t } } ) , } & { | \mathbb { Z } _ { t } | \geq 3 , } \\ { 0 , } & { | \mathbb { Z } _ { t } | < 3 , } \end{array} \right.\tag{13}
$$

Algorithm 1 Nested source-only feature selection for ModelGPT and SWAB under the revised   
LOVM protocol.   
Require: Task collection D; ImageNet task t<sub>IN</sub>; target $t \in \mathcal { D } \setminus \{ t _ { \mathrm { I N } } \}$   
Require: Candidate feature configurations $\mathcal { F } ;$ validation metric $Q ,$ , where larger is better   
Ensure: Predicted ranking $\hat { r } _ { t }$   
1: $\mathcal { T }  \mathcal { D } \setminus \{ t _ { \mathrm { I N } } \}$   
2: $S _ { t } \gets \mathcal { D } \dot { \backslash } \{ t \}$   
3: ${ \mathcal { V } } _ { t } \gets { \mathcal { T } } \dot { \backslash } \dot { \{ t \} }$   
4: for each configuration $f \in { \mathcal { F } }$ do   
5: $q _ { f } \gets 0$   
6: for each validation task $v \in \mathcal { V } _ { t }$ do   
7: $S _ { t , v } \gets \mathcal { D } \setminus \{ t , v \}$   
8: $\hat { r } _ { v } ^ { f , S _ { t , v } } \gets$ FITRANKERANDPREDICT $( f , S _ { t , v } , v )$   
9: $q _ { f } \gets q _ { f } + Q \Big ( \hat { r } _ { v } ^ { f , S _ { t , v } } , r _ { v } ^ { \star } \Big )$   
10: end for   
11: $q _ { f }  q _ { f } / | \mathcal { V } _ { t } |$   
12: end for   
13: $\hat { f } _ { t } \gets$ arg max $f \in \mathcal { F } ^ { q _ { f } }$   
14: $\hat { r } _ { t } \gets$ FITRANKERANDPREDICT $\cdot ( \hat { f } _ { t } , S _ { t } , t )$   
15: return $\hat { r } _ { t }$

where $\tau ( \cdot , \cdot )$ denotes Kendall’s rank correlation and $\mathbf { r } | _ { \mathcal { T } _ { t } }$ denotes the relative ordering restricted to the models in $\mathcal { T } _ { t }$ . The metric ranges from −1 to 1, where larger values indicate stronger agreement. We use the notation $\tau _ { 5 }$ to make explicit that the correlation is computed only among models appearing in both top-5 sets. For both metrics, we report the macro mean across the 22 evaluation targets:

$$
\overline { { R } } _ { 5 } = \frac { 1 } { | { \mathcal T } | } \sum _ { t \in { \mathcal T } } R _ { 5 } ( t ) , \qquad \overline { { \tau } } _ { 5 } = \frac { 1 } { | { \mathcal T } | } \sum _ { t \in { \mathcal T } } \tau _ { 5 } ( t ) .\tag{14}
$$

Higher values are better for both metrics.

## B.4 REVISED EVALUATION PROTOCOL

Algorithm 1 summarizes the revised evaluation protocol for ModelGPT and SWAB, which fit a ranker model using only source task data. For ModelGPT, the candidate feature configurations are $\mathcal { F } _ { \mathrm { M o d e l G P T } } = \mathsf { \bar { \{ I N B , } }  C , G \}$ , where INB denotes the ImageNet baseline and $C$ and $\bar { G }$ denote the features introduced in LOVM. For SWAB, the candidate feature configurations are $\mathcal { F } _ { \mathrm { S W A B } } =$ $\{ \mathrm { I N B } , \mathrm { G A P } _ { C } , \mathrm { G A P } _ { G } \}$ , where $\mathrm { G A P } _ { C }$ and ${ \mathrm { G A P } } _ { G }$ are obtained by incorporating the modality gap estimated by SWAB into $C$ and $G ,$ respectively. Following SWAB, we use $Q = R _ { 5 } + \tau _ { 5 }$ as the validation metric. For each target $t ,$ each candidate configuration is evaluated by treating every $v \in \mathcal V _ { t }$ as a validation task and fitting the ranker using $\mathcal { S } _ { t , v } \stackrel { \cdot } { = } \mathcal { D } \setminus \{ t , v \}$ . The configuration with the highest mean $Q$ over the validation tasks is then used to fit the final ranker model on $S _ { t } = \mathcal { D } \setminus \{ t \}$ and predict the ranking for t. Thus, candidate performance on t is not used for either configuration selection or ranker model fitting.

TC-ESR does not fit a ranker model and is training-free. To follow the same source-only principle, we apply this nested validation procedure only when selecting its softmax temperature λ. For each target t, λ is selected using the validation tasks in V<sub>t</sub> and the corresponding source sets $S _ { t , v } .$ without using candidate performance on t. Since $\mathcal { T } = \mathcal { D } \setminus \{ t _ { \mathrm { I N } } \}$ and $\mathcal { V } _ { t } \bar { = } \mathcal { T } \backslash \bar { \{ t \} }$ , ImageNet-1K is used neither as a target nor as a validation task. It remains available as a source task, including for the input features of ModelGPT and SWAB and the source regrets used by ESR and TC-ESR.

## C ADDITIONAL IMPLEMENTATION DETAILS

Computational Resources. All experiments are conducted on a server equipped with an AMD Ryzen 9 7950X CPU with 16 cores and a single NVIDIA GeForce RTX 4090 GPU with 24 GB of memory. We use PyTorch 2.10.0 with CUDA 12.8.

Table 9: Category labels in the shared vocabularies for the two task metadata fields.
<table><tr><td>Field</td><td>Category labels</td></tr><tr><td>Domain family</td><td>Object centered natural photographs; scene centered natural photographs; texture centered photographs; document or glyph images; medical images;</td></tr><tr><td></td><td>aerial or satellite images; synthetic or rendered images Label scale type Unordered semantic taxonomy; binary state; ordered level; numeric count; symbol or code identity; multi factor composition</td></tr></table>

Baseline Implementation and Random Seeds. We obtain the results for INB, AvgRank, ModelGPT (Zohar et al., 2023), and SWAB (Yi et al., 2024) using the implementation released with SWAB under our revised evaluation protocol. In the released implementation, ModelGPT introduces stochasticity by adding seed dependent Gaussian noise to the generated text embeddings used to construct its proxy features. SWAB inherits this stochastic proxy feature construction in its ModelGPT based ranker branch, while its optimal transport rank transfer and final rank aggregation are deterministic. Because the proxy feature construction produces variation across seeds, we repeat the full evaluation of ModelGPT and SWAB using random seeds 1 through 10 and report the mean over the 10 runs. In contrast, INB and AvgRank rank models directly from fixed performance statistics, while ESR and TC-ESR deterministically aggregate fixed source regrets once the task metadata and text embeddings are fixed. We therefore evaluate INB, AvgRank, ESR, and TC-ESR once with the random seed fixed to 1.

Details of the Alternative Selection Methods. All alternatives in Table 4 use the same task metadata, text encoder, and class-name template as TC-ESR, and select their hyperparameters by the same source-only validation with mean ORNR as the objective. Top-k sources ranks source tasks by the TC-ESR similarity $S _ { t } ( s )$ and averages SR uniformly over the top k sources with $k \in \{ 1 , 3 , 5 , 7 \}$ averaging over tied sources; Nearest source corresponds to $k = 1 .$ . Description embedding represents each task by the normalized sum of the embedding of its domain and task description and the mean embedding of its class names, and uses their cosine similarity rescaled to [0, 1] in place of $S _ { t } ( s )$ . Class names only uses $L _ { t } ( s )$ in place of $S _ { t } ( s )$ . Learned weights replaces the equal average in Equation 8 with $w _ { D } \dot { D } _ { t } ( s ) + \dot { w _ { L } } \dot { L } _ { t } ( \bar { s ) } + w _ { G } G _ { t } ( \dot { s } )$ , where the weights are selected jointly with λ from a simplex grid with step 0.25 augmented with equal weights; for targets without the granularity component, the weights of the remaining two axes are renormalized. Description embedding, Class names only, and Learned weights use the softmax weighting in Equation 9 with λ selected from the same grid as TC-ESR.

## D TASK METADATA CONSTRUCTION DETAILS

TC-ESR represents each task using two categorical fields: a domain family for visual domain similarity and a label scale type that indicates whether label granularity can be meaningfully estimated from class name geometry. We construct the task metadata in two steps: vocabulary induction and task category assignment. For each field, the shared vocabulary consists of category labels, definitions, and assignment criteria.

## D.1 VOCABULARY INDUCTION

For vocabulary induction, each anonymized task record contains a domain description, a task description, and all class names. The domain and task descriptions for all datasets are reported in Table 7. Dataset names, images, candidate model accuracies, model rankings, and downstream results are not provided. From the 23 task records, the LLM induces seven categories for the domain family and six categories for the label scale type, together with their definitions and assignment criteria. We then fix both vocabularies before task category assignment. The resulting category labels are listed in Table 9.

## D.2 TASK CATEGORY ASSIGNMENT

For each task and each run, the LLM receives only its anonymized task record and the two fixed vocabularies and assigns one category from each vocabulary. To reduce sensitivity to stochastic

Table 10: Task metadata for the 23 tasks. Agreement denotes the number of the five runs that assign the selected category. We set $\eta _ { d } = 1$ only when the label scale type indicates a taxonomy of semantic categories and $n _ { d } \geq 3 .$ , and set $\eta _ { d } = 0$ otherwise. The number of classes for each task is reported in Table 7. ImageNet-1K is included during metadata construction and as a source task, but is excluded from the evaluation targets.
<table><tr><td>Task</td><td>Domain family</td><td>Agreement</td><td>Label scale type</td><td>Agreement</td><td> $\eta _ { d }$ </td></tr><tr><td>Stanford Cars</td><td>Object centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>CIFAR100</td><td>Object centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>CLEVR closest distance</td><td>Synthetic or rendered images</td><td>5/5</td><td>Ordered level</td><td>5/5</td><td>0</td></tr><tr><td>CLEVR count all</td><td>Synthetic or rendered images</td><td>5/5</td><td>Numeric count</td><td>5/5</td><td>0</td></tr><tr><td>Country211</td><td>Scene centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>Diabetic Retinopathy</td><td>Medical images</td><td>5/5</td><td>Ordered level</td><td>5/5</td><td>0</td></tr><tr><td>DMLab</td><td>Synthetic or rendered images</td><td>5/5</td><td>Multi factor composition</td><td>3/5</td><td>0</td></tr><tr><td>DTD</td><td>Texture centered photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>EuroSAT</td><td>Aerial or satellite images</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>FER2013</td><td>Object centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>Flowers102</td><td>Object centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>GTSRB</td><td>Scene centered natural photographs</td><td>4/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>ImageNet-1K</td><td>Object centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>KITTI closest distance</td><td>Object centered natural photographs</td><td>5/5</td><td>Ordered level</td><td>3/5</td><td>0</td></tr><tr><td>MNIST</td><td>Document or glyph images</td><td>5/5</td><td>Symbol or code identity</td><td>5/5</td><td>0</td></tr><tr><td>PCam</td><td>Medical images</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>3/5</td><td>0</td></tr><tr><td>Oxford Pets</td><td>Object centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>Rendered SST2</td><td>Document or glyph images</td><td>5/5</td><td>Binary state</td><td>5/5</td><td>0</td></tr><tr><td>RESISC45</td><td>Aerial or satellite images</td><td>4/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>STL10</td><td>Object centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>SUN397</td><td>Scene centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr><tr><td>SVHN</td><td>Document or glyph images</td><td>5/5</td><td>Symbol or code identity</td><td>5/5</td><td>0</td></tr><tr><td>VOC2007</td><td>Object centered natural photographs</td><td>5/5</td><td>Unordered semantic taxonomy</td><td>5/5</td><td>1</td></tr></table>

variation in the LLM outputs, we repeat the category assignment five times per task and select the majority category across runs for each field. All 23 tasks yield a majority within five runs. For 21 tasks, all five runs produce the same domain family assignment; for 20 tasks, all five runs produce the same label scale assignment. Table 10 reports the resulting task metadata and agreement across runs.

## D.3 CLASS NAME EMBEDDINGS AND LABEL GRANULARITY

For all 23 tasks, we encode each class name using all-mpnet-base-v2 with the fixed template Class label: {class name} and $\ell _ { 2 }$ normalize the resulting embeddings. These embeddings are used to compute label semantic similarity for every pair of source and target tasks. For each task with $\eta _ { d } = 1$ , we additionally compute $\widetilde { \mu } _ { d }$ as the median pairwise cosine distance between distinct class name embeddings, following the definition in the main paper. Label granularity is therefore computed for 14 of the 23 tasks, including ImageNet-1K, and for 13 of the 22 evaluation targets.

The two vocabularies, the resulting task metadata, and the class name embeddings are generated once and held fixed throughout evaluation. The domain family assignment is used to compute do main similarity, while the label scale assignment determines whether the label granularity component is applicable. Label semantic similarity is computed directly from these class name embeddings, independently of the categorical metadata.

## E MORE EXPERIMENTAL RESULTS

## E.1 FURTHER ABLATION STUDIES

All design choices of TC-ESR were fixed before evaluation on the target tasks, following standard defaults and the motivations given in Section 4.2: exact matching for categorical domain families, the median as a robust summary of pairwise distances, bidirectional coverage to avoid favoring partial correspondence, and an unweighted average of the three axes. The ablations below are posthoc analyses of these choices rather than the basis on which they were selected.

Domain Similarity Measure. We encode the assigned domain family names using the fixed text encoder E. We then compare exact matching of the assigned domain families with cosine similarity

Table 11: Comparison of do- Table 12: Comparison of label se- Table 13: Comparison of label main similarity measures. mantic aggregation rules. granularity statistics.
<table><tr><td>Measure</td><td>SR (%p)↓</td><td>ORNR (%) ↓</td></tr><tr><td>Semantic sim.</td><td>6.71</td><td>24.43</td></tr><tr><td>Exact (Ours)</td><td>4.66</td><td>17.81</td></tr></table>

<table><tr><td>Aggregation rule</td><td>SR (%p) ↓</td><td>ORNR (%) ↓</td></tr><tr><td>Mean (all pairs)</td><td>4.83</td><td>19.59</td></tr><tr><td>Balanced OT</td><td>5.14</td><td>21.41</td></tr><tr><td> $\kappa _ { s  t }$  only only</td><td>5.57</td><td>22.80</td></tr><tr><td> $\kappa _ { t  s }$  Bidirectional (Ours)</td><td>5.29</td><td>21.56</td></tr><tr><td></td><td>4.66</td><td>17.81</td></tr></table>

<table><tr><td>Statistic</td><td>SR (%p) ↓</td><td>ORNR (%)↓</td></tr><tr><td>NN mean (k=1)</td><td>8.21</td><td>27.45</td></tr><tr><td>NN mean (k=3)</td><td>8.51</td><td>29.68</td></tr><tr><td>NN mean (k=6)</td><td>8.50</td><td>29.65</td></tr><tr><td>Mean dist.</td><td>7.74</td><td>24.38</td></tr><tr><td>Median dist. (Ours)</td><td>4.66</td><td>17.81</td></tr></table>

Table 14: Mean results over 22 target tasks under the revised LOVM evaluation protocol. SR is reported in percentage points, whereas ORNR, $R _ { 5 } ,$ and $\tau _ { 5 }$ are reported as percentages. For ModelGPT and SWAB, each target result is averaged over ten seeds before macro averaging. Bold and underline denote the best and second-best mean results, respectively.
<table><tr><td>Method</td><td>SR (%p) ↓</td><td>ORNR (%) ↓</td><td> $R _ { 5 } \left( \% \right) \mathrm { \uparrow }$ </td><td> $\tau _ { 5 } \left( \% \right) \mathrm { \uparrow }$ </td></tr><tr><td>INB</td><td>7.23</td><td>24.66</td><td>41.8</td><td>23.3</td></tr><tr><td>AvgRank</td><td>7.35</td><td>25.69</td><td>41.8</td><td>23.0</td></tr><tr><td>ModelGPT</td><td>7.11</td><td>24.25</td><td>41.1</td><td>18.0</td></tr><tr><td>SWAB</td><td>7.39</td><td>25.81</td><td>47.4</td><td>28.3</td></tr><tr><td>ESR</td><td>5.83</td><td>22.56</td><td>31.8</td><td>18.2</td></tr><tr><td>TC-ESR</td><td>4.66</td><td>17.81</td><td>40.0</td><td>22.7</td></tr></table>

Table 15: Effect of the regret used for source aggregation.
<table><tr><td>Method</td><td>Source Regret</td><td>SR (%p) ↓</td><td>ORNR (%) ↓</td></tr><tr><td rowspan="2">ESR</td><td>SR</td><td>5.83</td><td>22.56</td></tr><tr><td>ORNR</td><td>7.03</td><td>27.19</td></tr><tr><td rowspan="2">TC-ESR</td><td>SR</td><td>4.66</td><td>17.81</td></tr><tr><td>ORNR</td><td>4.70</td><td>18.97</td></tr></table>

between the resulting embeddings. As shown in Table 11, exact matching yields lower SR and ORNR, consistent with the motivation in Section 4.2. This may be because domain families with semantically similar labels do not necessarily contain visually similar tasks.

Label Semantic Similarity Measure. We compare five aggregation rules for deriving $L _ { t } ( s )$ from the pairwise class similarity matrix $\Gamma ^ { s , t }$ , while keeping all other components fixed. The mean over all pairs averages every entry of $\Gamma ^ { s , t } ,$ whereas balanced optimal transport couples the two class name sets under uniform marginals (Peyre & Cuturi, 2019). The score´ $\kappa _ { s  t }$ averages the strongest source match for each target class, while $\kappa _ { t  s }$ applies the same operation in the reverse direction. Bidirectional coverage averages these two directional scores. As shown in Table 12, bidirectional coverage achieves the lowest mean SR and ORNR. These results suggest that bidirectional coverage benefits from retaining each class’s strongest match in both directions and evaluating these matches independently.

Label Granularity Measure. As shown in Table 13, the median pairwise distance achieves the lowest SR and ORNR. The mean over all pairwise distances also outperforms the nearest neighbor statistics, suggesting that overall class separation is more informative than local neighborhood structure. The further improvement from the mean to the median is consistent with the benefit of reduced sensitivity to extreme pairwise distances. We test $k \in \{ 1 , 3 , 6 \}$ ; among the tasks for which label granularity is computed, the smallest label set contains seven classes, making k = 6 the largest candidate.

Regret Metrics for Source Aggregation. Table 15 compares the regret used for source aggregation and for selecting λ. For source aggregation, ORNR normalizes each source regret by the performance range of the corresponding task. This normalization alters source contributions independently of target similarity. In contrast, SR preserves absolute performance gaps and lets the similarity weights control source contributions.

Table 16: Sensitivity to the LLM used for metadata construction.
<table><tr><td>LLM</td><td>SR (%p)</td><td>ORNR (%)</td></tr><tr><td>gpt-5.4-nano</td><td>4.66</td><td>17.81</td></tr><tr><td>Qwen3-4B-Instruct</td><td>4.83</td><td>19.59</td></tr></table>

Table 17: Sensitivity to the text encoder.
<table><tr><td>Text encoder</td><td>SR (%p)</td><td>ORNR (%)</td></tr><tr><td>all-mpnet-base-v2</td><td>4.66</td><td>17.81</td></tr><tr><td>BGE-base-en-v1.5</td><td>4.83</td><td>19.59</td></tr><tr><td>all-distilroberta-v1</td><td>4.88</td><td>19.79</td></tr></table>

Table 18: Per-task results under the revised LOVM evaluation protocol. SR is reported in percentage points, whereas ORNR, $R _ { 5 } ,$ , and $\tau _ { 5 }$ are reported as percentages. For ModelGPT and SWAB, each per-task result is averaged over ten seeds. Bold and underline denote the best and second-best means in the Avg. column, respectively.
<table><tr><td></td><td>Method</td><td>CarS</td><td>clevss-dtance Cliar10</td><td>clev-count</td><td>cCouuun71</td><td>reinopthy</td><td>dqmlab</td><td>u</td><td>eurosat</td><td>Fer2013</td><td>Fowers</td><td>srb</td><td>i</td><td>mnisst</td><td>pcamm</td><td>Pets</td><td>reunest2 resic45</td><td>sII</td><td></td><td>sn397</td><td>Svhn</td><td>V207</td></tr><tr><td rowspan="7">SR (%p) ↓</td><td rowspan="7">INB AvgRank</td><td>0.00</td><td>0.10</td><td>5.37</td><td>11.68</td><td>3.56</td><td>49.81 9.06</td><td>0.00</td><td>5.87</td><td>2.70</td><td>0.49</td><td>8.73</td><td>12.52</td><td>13.03</td><td>9.09</td><td>0.41</td><td>14.39</td><td>0.00</td><td>0.49</td><td>1.27</td><td>10.56</td><td>0.00</td><td>Avg. 7.23</td></tr><tr><td>0.46</td><td>0.00</td><td>9.59</td><td>12.85</td><td>4.61</td><td>42.93</td><td>7.75</td><td>1.22 0.00</td><td>8.16</td><td>0.00</td><td>0.00</td><td>32.49</td><td>12.47</td><td>4.62</td><td>0.00</td><td>7.08</td><td>0.57</td><td>0.96</td><td>0.09</td><td>12.38</td><td>3.37</td><td>7.35</td></tr><tr><td>ModelGPT</td><td>0.00 0.10</td><td>5.79</td><td>11.68</td><td>3.56</td><td>49.81</td><td>8.93</td><td>0.00</td><td>5.87</td><td>3.24</td><td>0.49</td><td>6.11</td><td>12.52</td><td>13.03</td><td>8.64</td><td>0.37</td><td>13.64</td><td>0.00</td><td>0.53</td><td>1.27</td><td>10.56</td><td>0.34 7.11</td></tr><tr><td>SWAB</td><td>0.00 0.02</td><td>9.59</td><td>11.68</td><td>3.56</td><td>49.13</td><td>9.06</td><td>0.00</td><td>5.87</td><td>2.70</td><td>0.49</td><td>8.73</td><td>12.52</td><td>13.03</td><td>8.64</td><td>0.41 10.29</td><td>0.00</td><td>0.49</td><td>1.27</td><td></td><td>12.38 2.69</td><td>7.39</td></tr><tr><td>ESR</td><td>0.29 1.99</td><td>3.27</td><td>9.58</td><td>7.93</td><td>20.57</td><td>5.60</td><td>0.21</td><td>0.04</td><td>2.68</td><td>3.17</td><td>0.17</td><td>32.35</td><td>10.12</td><td>10.46</td><td>0.79</td><td>15.32</td><td>0.95</td><td>0.86</td><td>1.82</td><td>0.00</td><td>0.13 5.83</td></tr><tr><td>TC-ESR</td><td>0.00 0.10</td><td>3.27</td><td>15.16</td><td>4.61</td><td>2.22</td><td>5.60</td><td>0.00</td><td>5.87</td><td>2.70</td><td>0.49</td><td>0.00</td><td>12.52</td><td>10.12</td><td>5.94</td><td>0.41</td><td>14.94</td><td>0.57</td><td>0.49</td><td>0.09</td><td>17.52</td><td>0.00 4.66</td></tr><tr><td>INB</td><td>0.00</td><td>0.28 33.44</td><td></td><td>38.30 12.18</td><td>70.37</td><td>98.52</td><td>0.00</td><td>14.32 15.85</td><td></td><td>2.16</td><td>33.82</td><td>37.71</td><td>26.97</td><td>751.46</td><td>4.48</td><td>67.88</td><td>0.00</td><td>9.31</td><td>7.27</td><td>18.20 0.00</td><td>|24.66</td></tr><tr><td rowspan="4">ORNR (%) ↓</td><td>AvgRank ModelGPT</td><td>1.16</td><td>0.00 0.28</td><td>59.78 42.1415.76 36.08</td><td></td><td>38.3012.18</td><td>60.65 70.37 97.09</td><td>84.22</td><td>4.45 0.00 0.00</td><td>48.01 14.32 19.07</td><td>0.00 2.16</td><td>0.00 23.68</td><td>97.88 837.71</td><td></td><td>25.81 26.17 26.97 48.93</td><td>0.00 4.03</td><td>33.42 64.35</td><td>2.06 0.00</td><td>18.38 10.21</td><td>0.53 7.27</td><td></td><td>21.34 23.52 18.20 2.35</td><td>25.69 24.25</td></tr><tr><td>SWAB</td><td>0.00 0.00 0.06</td><td>59.78</td><td></td><td></td><td>38.30 12.18 69.40 98.52</td><td></td><td>0.00</td><td></td><td>14.32 15.85</td><td>2.16</td><td>33.82</td><td>237.71</td><td></td><td>26.97 48.93</td><td>4.48</td><td>48.55</td><td>0.00</td><td>9.31</td><td>7.27</td><td></td><td>21.34 18.81</td><td>25.81</td></tr><tr><td>ESR 0.72</td><td>5.48</td><td>20.40</td><td></td><td></td><td>31.41 27.14 29.06 60.88</td><td></td><td>0.77</td><td>0.09</td><td>15.73</td><td>14.05</td><td>0.68</td><td>97.46</td><td></td><td>20.94 59.27</td><td>8.66</td><td>72.28</td><td>3.43</td><td>16.47</td><td>10.42</td><td>0.00</td><td>0.93</td><td>22.56</td></tr><tr><td>TC-ESR</td><td>0.00 0.28</td><td></td><td>20.40 49.70 15.76</td><td></td><td>3.13</td><td>60.88</td><td>0.00</td><td></td><td>14.32 15.85</td><td>2.16</td><td>0.00</td><td>37.71</td><td></td><td>20.94 33.62</td><td>4.48</td><td>70.47</td><td>2.06</td><td>9.31</td><td>0.53</td><td>30.21</td><td>0.00</td><td>17.81</td></tr><tr><td rowspan="5">R5 (%) ↑</td><td>INB AvgRank</td><td>80.0</td><td>80.0</td><td>0.0</td><td>40.0</td><td>40.0</td><td>0.0</td><td>0.0</td><td>80.0 60.0</td><td>40.0</td><td>60.0</td><td>60.0</td><td>0.0</td><td>20.0</td><td>0.0</td><td>60.0</td><td>0.0</td><td>100.0</td><td>20.0</td><td>60.0</td><td>60.0</td><td>60.0</td><td>41.8</td></tr><tr><td></td><td>80.0 80.0 80.0</td><td>80.0</td><td>0.0</td><td>40.0</td><td>40.0</td><td>0.0</td><td>0.0 0.0</td><td>80.0</td><td>60.0 40.0</td><td></td><td>60.0</td><td>60.0</td><td>0.0</td><td>20.0 0.0</td><td>60.0</td><td>0.0</td><td>100.0</td><td></td><td>20.0</td><td>60.0</td><td>60.0</td><td>60.0 41.8</td></tr><tr><td>ModelGPT</td><td></td><td>0.0</td><td>38.0</td><td></td><td>50.0 0.0</td><td></td><td>80.0</td><td>58.0</td><td>36.0</td><td>66.0</td><td>58.0</td><td>0.0</td><td>12.0</td><td>0.0</td><td>66.0</td><td>8.0</td><td>90.0</td><td>20.0</td><td>60.0</td><td></td><td>42.0 60.0</td><td>41.1</td></tr><tr><td>SWAB</td><td>80.0 80.0</td><td>0.0</td><td>60.0</td><td></td><td>40.0 4.0</td><td>0.0</td><td>80.0</td><td>60.0</td><td>56.0</td><td>80.0</td><td>60.0</td><td>0.0</td><td>40.0</td><td>0.0</td><td>80.0</td><td>22.0</td><td>100.0</td><td>20.0</td><td>60.0</td><td>60.0</td><td>60.0</td><td>47.4</td></tr><tr><td>ESR TC-ESR</td><td>60.0</td><td>40.0 0.0</td><td>60.0</td><td></td><td>20.0 0.0</td><td></td><td>0.0 80.0</td><td>60.0</td><td>20.0</td><td>40.0</td><td>40.0</td><td>0.0</td><td>40.0</td><td>0.0</td><td>40.0</td><td>0.0</td><td>60.0</td><td>20.0</td><td>20.0</td><td>60.0</td><td>40.0</td><td>31.8</td></tr><tr><td></td><td>80.0</td><td>60.0</td><td>20.0</td><td>0.0</td><td>20.0</td><td>20.0</td><td>0.0</td><td>80.0</td><td>60.0</td><td>40.0</td><td>60.0</td><td>20.0</td><td>0.0</td><td>40.0</td><td>20.0</td><td>60.0</td><td>40.0</td><td>60.0</td><td>40.0</td><td>60.0</td><td>40.0</td><td>60.0</td><td>40.0</td></tr><tr><td rowspan="5">τ5 (%) ↑</td><td>INB</td><td>66.7</td><td>66.7</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>66.7 100.0 33.3 100.0</td><td>0.0</td><td>33.3 100.0 100.0</td><td>100.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>-33.3 33.3</td><td>0.0 0.0</td><td>80.0 40.0</td><td>0.0 0.0</td><td>-100.0 -33.3</td><td>33.3 -33.3</td><td>100.0 33.3</td><td>23.3 23.0</td></tr><tr><td>AvgRank ModelGPT</td><td>33.3 66.7 66.7</td><td>100.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></table>

Sensitivity to the LLM and Text Encoder. Tables 16 and 17 report TC-ESR with a different LLM for metadata construction and with different text encoders, keeping all other components fixed. Replacing gpt-5.4-nano with the open-source Qwen3-4B-Instruct, we rerun the entire metadata construction, including vocabulary induction and category assignment. The resulting SR of 4.83%p remains lower than those of ESR and the best prior method, ModelGPT, which are 5.83%p and 7.11%p, respectively. For the text encoder, we evaluate BGE-base-en-v1.5 and all-distilroberta-v1 with encoder-specific templates. Both encoders yield SR and ORNR comparable to MPNet.

## E.2 PER-TASK EXPERIMENTAL RESULTS

Table 14 reports the mean results over the 22 target tasks. In addition to the SR and ORNR results presented in the main paper, we report $R _ { 5 }$ and $\tau _ { 5 }$ , with TC-ESR achieving the lowest SR and ORNR and SWAB achieving the highest $R _ { 5 }$ and $\tau _ { 5 } .$ . The complete per-task results for all four metrics are reported in Table 18.

## E.3 TASK SIMILARITY ANALYSIS

Figure 4 visualizes the task similarity scores used by TC-ESR. High similarity is observed between the two CLEVR tasks and between EuroSAT and RESISC45, consistent with their shared synthetic scene and remote sensing domains, respectively. MNIST and SVHN are also highly similar despite their different visual appearances, reflecting their common digit label semantics. For Cars, the high similarity to Pets and Flowers is consistent with the combined effects of their shared domain family and similar label granularity, despite weak label semantic overlap. Together, these patterns indicate that the measured task similarity $S _ { t } ( s )$ captures complementary information from visual domain, label semantics, and label granularity.

![](images/c26ae505b3d9d71f4108c48369eaca146afc83bd61294700c8d2041b74aa3296.jpg)  
Figure 4: Task similarity matrix used by TC-ESR. Rows denote target tasks and columns denote source tasks, and each cell reports $S _ { t } ( s )$ . The score combines visual domain, label semantics, and label granularity when applicable. Crosses mark excluded pairs with $s = t ,$ while ImageNet-1K appears only as a source task under the revised evaluation protocol. Because the inclusion of label granularity depends on the label scale type of the target task, the matrix need not be symmetric.

## E.4 ROBUSTNESS OF ORNR

Alternative Denominators. Table 19 examines whether the comparison depends on the denominator used by ORNR. We consider one-sided variants, which replace only the lower end of the range, and two-sided variants, which also replace its upper end. For the one-sided variants, in addition to the full range, we consider percentile and interquartile range based alternatives. For the percentile alternative $P _ { q } ,$ , we use the qth percentile of the candidate model accuracies, which defines the cutoff for the lowest $q \%$ of accuracies, as the lower endpoint. For the interquartile range alternatives, we define $\mathrm { I Q R } _ { t } = Q _ { 3 , t } - Q _ { 1 , t }$ and, for a multiplier c, set the cutoff to $\bar { F _ { t , c } ^ { - } } = Q _ { 1 , t } \bar { - c } \mathrm { I Q R } _ { t }$ We then use the lowest candidate model accuracy that is no smaller than $F _ { t , c } ^ { - }$ as the lower endpoint of the denominator. The multipliers 1.5 and 3.0 are commonly used in exploratory data analysis, particularly in boxplots, to identify outliers and extreme outliers, respectively (Tukey, 1977). We additionally vary c from 1.0 to 5.0 to evaluate a broader range of choices. For the two-sided variants, we use the difference between the upper and lower percentiles, $P _ { 9 5 } - P _ { 5 }$ and $P _ { 9 0 } - P _ { 1 0 }$ , or the range between the upper and lower Tukey whiskers with multiplier k. TC-ESR achieves the lowest normalized regret under all 13 denominator choices, with ESR consistently ranking second. The two-sided Tukey range with $k = 3 . 0$ yields the same values as the one-sided 3.0 IQR variant, indicating that no candidate is flagged as a high outlier at this threshold. These results show that the relative advantage of TC-ESR does not depend on the denominator choice. We adopt 1.5 IQR as the primary setting because it is the most commonly used criterion, rather than selecting the multiplier based on these results.

Table 19: Mean normalized regret over 22 targets under alternative denominators. ORNR corresponds to 1.5 IQR. Lower is better; bold and underline denote the best and second-best results, respectively.
<table><tr><td>Type</td><td>Denominator</td><td>INB</td><td>AvgRank</td><td>ModelGPT</td><td>SWAB</td><td>ESR</td><td>TC-ESR</td></tr><tr><td>One-sided</td><td>Min-max</td><td>21.67</td><td>22.26</td><td>21.35</td><td>22.16</td><td>19.29</td><td>15.97</td></tr><tr><td></td><td> $P _ { 5 }$ </td><td>26.99</td><td>28.65</td><td>26.61</td><td>28.78</td><td>23.78</td><td>19.52</td></tr><tr><td></td><td> $P _ { 1 0 }$ </td><td>29.35</td><td>31.19</td><td>28.97</td><td>31.26</td><td>26.58</td><td>21.47</td></tr><tr><td></td><td>1.0IQR</td><td>26.05</td><td>28.48</td><td>25.78</td><td>28.07</td><td>23.63</td><td>19.01</td></tr><tr><td></td><td>1.5 IQR (ORNR)</td><td>24.66</td><td>25.69</td><td>24.25</td><td>25.81</td><td>22.56</td><td>17.81</td></tr><tr><td></td><td>2.0IQR</td><td>24.20</td><td>25.19</td><td>23.79</td><td>25.23</td><td>21.82</td><td>17.41</td></tr><tr><td></td><td>3.0IQR</td><td>23.53</td><td>24.81</td><td>23.12</td><td>24.55</td><td>21.21</td><td>16.83</td></tr><tr><td></td><td>4.0IQR</td><td>22.67</td><td>24.81</td><td>22.51</td><td>23.70</td><td>21.11</td><td>16.79</td></tr><tr><td></td><td>5.0IQR</td><td>22.38</td><td>23.91</td><td>22.16</td><td>23.41</td><td>20.70</td><td>16.50</td></tr><tr><td>Two-sided</td><td> $P _ { 9 5 } - P _ { 5 }$ </td><td>30.53</td><td>33.17</td><td>30.10</td><td>32.25</td><td>27.95</td><td>22.56</td></tr><tr><td></td><td> $P _ { 9 0 } - P _ { 1 0 }$ </td><td>37.68</td><td>40.33</td><td>37.17</td><td>39.53</td><td>35.42</td><td>28.66</td></tr><tr><td></td><td> $\mathrm { T u k e y } , k = 1 . 5$ </td><td>25.02</td><td>25.87</td><td>24.59</td><td>26.06</td><td>22.94</td><td>18.18</td></tr><tr><td></td><td> $\mathrm { T u k e y } , k = 3 . 0$ </td><td>23.53</td><td>24.81</td><td>23.12</td><td>24.55</td><td>21.21</td><td>16.83</td></tr></table>

Table 20: Mean absolute change in normalized regret when weak candidates are perturbed.
<table><tr><td>Perturbation</td><td>Min-max</td><td>ORNR</td><td>Reduction (%)</td></tr><tr><td>Weakest candidate → 50%</td><td>4.58</td><td>1.19</td><td>74.0</td></tr><tr><td>Weakest candidate → 0</td><td>6.54</td><td>1.17</td><td>82.0</td></tr><tr><td>Bottom 5% removed</td><td>3.48</td><td>1.82</td><td>47.7</td></tr><tr><td>Bottom 10% removed</td><td>5.36</td><td>3.62</td><td>32.5</td></tr><tr><td>Bottom 20% removed</td><td>7.84</td><td>6.00</td><td>23.5</td></tr></table>

Sensitivity to Weak Candidates. This experiment empirically examines the scenario illustrated in Figure 2b, in which a single unusually weak candidate inflates the normalization range. Since neither the best model nor the selected model changes in this scenario, an ideal normalized regret should remain unchanged. We therefore modify only the weak candidates of each target, either by lowering the accuracy of the weakest candidate to half of its value or to zero, or by removing the bottom 5%, 10%, or 20% of candidates, and measure the mean absolute change in the normalized regret of the selected models under min–max normalization and ORNR. Table 20 reports these changes together with the relative reduction achieved by ORNR. When only the weakest candidate is degraded, ORNR changes 74.0% to 82.0% less than min–max normalization and remains unchanged on 16 of the 22 targets. When multiple weak candidates are removed, ORNR remains less sensitive, but the reduction decreases from 47.7% to 23.5%, because removing many candidates shifts the quartiles that determine its lower whisker. Thus, ORNR is robust to individual weak outliers but, like any task-relative normalization, is not invariant to substantial changes in the composition of the candidate pool.

## E.5 BASELINES RETUNED FOR SELECTION QUALITY

In the main experiments, ModelGPT and SWAB select their input feature configurations by $R _ { 5 } +$ τ<sub>5</sub> on source tasks, following their original protocols, whereas TC-ESR selects its temperature by ORNR. To rule out that this asymmetry favors TC-ESR, we rerun the same source-only feature selection for ModelGPT and SWAB with SR or ORNR as the validation objective. Table 21 shows that this barely changes ModelGPT and degrades SWAB on both SR and ORNR. TC-ESR and ESR therefore remain the best and second-best methods regardless of the objective used to tune the baselines.

## F ROLE OF TARGET-CONDITIONED WEIGHTING

Prior methods and ESR select nearly the same candidate model for all target tasks. This section examines how target-conditioned weighting allows TC-ESR to depart from this choice and how this affects its gains.

Table 21: ModelGPT and SWAB with feature selection retuned by SR or ORNR.
<table><tr><td>Method</td><td>Feature selection</td><td>SR</td><td>ORNR</td><td> $R _ { 5 }$ </td><td> $\tau _ { 5 }$ </td></tr><tr><td>ModelGPT</td><td> $R _ { 5 } + \tau _ { 5 }$ </td><td>7.11</td><td>24.25</td><td>41.1</td><td>18.0</td></tr><tr><td></td><td>SR</td><td>7.14</td><td>24.16</td><td>42.1</td><td>17.1</td></tr><tr><td></td><td>ORNR</td><td>7.12</td><td>24.18</td><td>42.5</td><td>16.6</td></tr><tr><td>SWAB</td><td> $R \mathrm { 5 } + \tau \mathrm { 5 }$ </td><td>7.39</td><td>25.81</td><td>47.4</td><td>28.3</td></tr><tr><td></td><td>SR</td><td>8.58</td><td>28.40</td><td>34.3</td><td>10.2</td></tr><tr><td></td><td>ORNR</td><td>8.10</td><td>26.42</td><td>36.6</td><td>10.9</td></tr><tr><td>ESR</td><td>一</td><td>5.83</td><td>22.56</td><td>31.8</td><td>18.2</td></tr><tr><td>TC-ESR</td><td>一</td><td>4.66</td><td>17.81</td><td>40.0</td><td>22.7</td></tr></table>

Table 22: Distribution of the selected candidate models over 22 target tasks. For ModelGPT and SWAB, selections are pooled over ten seeds.
<table><tr><td>Method</td><td>Distinct candidate models</td><td>Share of the most frequent (%)</td></tr><tr><td>INB</td><td>1</td><td>100.0</td></tr><tr><td>AvgRank</td><td>1</td><td>100.0</td></tr><tr><td>ModelGPT</td><td>3</td><td>95.0</td></tr><tr><td>SWAB</td><td>5</td><td>80.9</td></tr><tr><td>ESR</td><td>2</td><td>95.5</td></tr><tr><td>TC-ESR</td><td>8</td><td>45.5</td></tr></table>

Table 23: SR and ORNR on Retinopathy with and without PCam in the source tasks.
<table><tr><td></td><td colspan="2">SR (%p) ↓</td><td colspan="2">ORNR (%) ↓</td></tr><tr><td>Method</td><td>w/ PCam</td><td>w/o PCam</td><td>w/ PCam</td><td>w/o PCam</td></tr><tr><td>INB</td><td>49.81</td><td>49.81</td><td>70.37</td><td>70.37</td></tr><tr><td>AvgRank</td><td>42.93</td><td>42.93</td><td>60.65</td><td>60.65</td></tr><tr><td>ModelGPT</td><td>49.81</td><td>49.81</td><td>70.37</td><td>70.37</td></tr><tr><td>SWAB</td><td>49.13</td><td>49.13</td><td>69.40</td><td>69.40</td></tr><tr><td>ESR</td><td>20.57</td><td>20.57</td><td>29.06</td><td>29.06</td></tr><tr><td>TC-ESR</td><td>2.22</td><td>20.57</td><td>3.13</td><td>29.06</td></tr><tr><td>Best fixed model</td><td>20.57</td><td>20.57</td><td>29.06</td><td>29.06</td></tr></table>

The best fixed model as a reference. The best fixed model selects the same candidate model for all target tasks using their target accuracies. It therefore attains the lowest mean SR among all selections that do not depend on the target task. As shown in Table 22, prior methods and ESR are close to such selections: INB and AvgRank select the same candidate model for all 22 targets, ModelGPT and SWAB select their most frequent candidate model in 95.0% and 80.9% of their selections over targets and seeds, and ESR selects the best fixed model for 21 of the 22 targets using source tasks alone, which largely explains its gain over prior methods. In contrast, TC-ESR selects eight distinct candidate models by assigning target-conditioned weights to source tasks. Such target-specific selection is necessary to attain lower SR than the best fixed model.

Case study: Retinopathy. On Retinopathy, PCam is the only source task sharing its domain family, medical images. Its similarity to the target therefore exceeds those of the other source tasks by a large margin, and it receives more than 99.99% of the source weight. TC-ESR then selects OpenAI RN50x16. As shown in Table 23, its SR is 2.22%p, whereas the best fixed model, which ESR also selects, incurs a substantially higher SR of 20.57%p. Removing PCam from the source and validation tasks and repeating the source-only validation increases the SR of TC-ESR to 20.57%p, while the selections of the other methods remain unchanged. Among the compared methods, only TC-ESR measures domain similarity between tasks, and thus only it identifies PCam as the source task most relevant to Retinopathy, which accounts for its low SR on this target.

Concentration of source weights. To examine how concentrated the source weights of TC-ESR are, we compute the effective number of sources for each target,

$$
N _ { \mathrm { e f f } , t } = \frac { 1 } { \sum _ { s \in { \cal S } _ { t } } w _ { t } ( s ; \lambda _ { t } ^ { \star } ) ^ { 2 } } ,\tag{15}
$$

Table 24: Effective number of source tasks $N _ { \mathrm { e f f } , t }$ of TC-ESR for each target, grouped by domain family. #Sources denotes the number of source tasks sharing the target’s domain family, counting ImageNet-1K. For every target with at least one such source, these sources receive more than 99.3% of the total weight.
<table><tr><td>Domain family</td><td>Target</td><td>#Sources</td><td> $N _ { \mathrm { e f f } , t }$ </td></tr><tr><td rowspan="9">Object centered natural photographs</td><td>Stanford Cars</td><td>8</td><td>4.24</td></tr><tr><td>CIFAR100</td><td>8</td><td>3.93</td></tr><tr><td>FER2013</td><td>8</td><td>3.63</td></tr><tr><td>Flowers102</td><td>8</td><td>5.25</td></tr><tr><td>KITTI closest distance</td><td>8</td><td>6.24</td></tr><tr><td>Oxford Pets</td><td>8</td><td>4.27</td></tr><tr><td>STL10</td><td>8</td><td>5.22</td></tr><tr><td>VOC2007</td><td>8</td><td>3.91</td></tr><tr><td>Country211</td><td>2</td><td>1.80</td></tr><tr><td rowspan="3"></td><td>GTSRB</td><td>2</td><td>1.77</td></tr><tr><td>SUN397</td><td>2</td><td>1.44</td></tr><tr><td>CLEVR closest distance</td><td>2</td><td>2.00</td></tr><tr><td rowspan="3"></td><td>CLEVR count all</td><td>2</td><td>1.65</td></tr><tr><td>DMLab</td><td>2</td><td>1.61</td></tr><tr><td>MNIST</td><td>2</td><td>1.24</td></tr><tr><td rowspan="3"></td><td>Rendered SST2</td><td>2</td><td>1.99</td></tr><tr><td>SVHN</td><td>2</td><td>1.21</td></tr><tr><td>EuroSAT</td><td>1</td><td>1.01</td></tr><tr><td rowspan="2">Medical images</td><td>RESISC45</td><td>1</td><td>1.00</td></tr><tr><td>Diabetic Retinopathy</td><td>1</td><td>1.00</td></tr><tr><td rowspan="2"></td><td>PCam</td><td>1</td><td>1.00</td></tr><tr><td>DTD</td><td>0</td><td>7.97</td></tr></table>

Table 25: Mean SR and ORNR over 21 target tasks excluding Retinopathy. Bold and underline denote the best and second-best LOVM methods.
<table><tr><td>Method</td><td>SR (%p) ↓</td><td>ORNR (%) ↓</td></tr><tr><td>INB</td><td>5.20</td><td>22.48</td></tr><tr><td>AvgRank</td><td>5.65</td><td>24.03</td></tr><tr><td>ModelGPT</td><td>5.08</td><td>22.06</td></tr><tr><td>SWAB</td><td>5.40</td><td>23.73</td></tr><tr><td>ESR</td><td>5.13</td><td>22.25</td></tr><tr><td>TC-ESR</td><td>4.78</td><td>18.51</td></tr><tr><td>Best fixed model</td><td>4.81</td><td>21.19</td></tr></table>

where $w _ { t } ( s ; \lambda _ { t } ^ { \star } )$ denotes the source weights in Equation 9. This quantity is the effective sample size of Kish (1965): it equals one when a single source task receives all of the weight and $\lvert S _ { t } \rvert$ when the weights are uniform, as in ESR, and can be read as the number of equally weighted source tasks that would yield the same concentration. As shown in Table 24, source tasks sharing the target’s domain family receive nearly all of the weight, and the weight is further differentiated among them by label semantics and, where applicable, label granularity. $N _ { \mathrm { e f f } , t }$ is therefore at most 1.01 for the four targets with a single such source, including Retinopathy, whereas it is at least 1.21 for all other targets and ranges from 3.63 to 6.24 for the object-centered targets. This behavior differs from the fixed Nearest-source rule, which performs worse than ESR in Table 4: TC-ESR concentrates on a single source task only when the target shares its domain family with a single source task.

Results excluding Retinopathy. We next examine whether the advantage of TC-ESR extends beyond Retinopathy. Table 25 reports the results over the remaining 21 targets, where we exclude Retinopathy only from the evaluation targets and keep the predictions of all methods and the selected $\lambda _ { t } ^ { \star }$ of TC-ESR unchanged.

The best fixed model is reselected over the 21 targets, and selecting it by either SR or ORNR yields the same candidate model as before. TC-ESR remains the best LOVM method on both metrics, but its SR margin becomes small: its SR of 4.78%p compares with 5.08%p for ModelGPT, the strongest prior method on these targets, and 5.13%p for ESR, and is comparable to the 4.81%p of the best fixed model, which requires target accuracies. The margin remains substantial in ORNR, where TC-ESR attains 18.51%, compared with 22.06% for ModelGPT, 22.25% for ESR, and 21.19% for the best fixed model.

Summary. While prior methods and ESR select nearly the same candidate model for all target tasks, TC-ESR selects target-specific models by concentrating the source weights on tasks sharing the target’s domain family and differentiating among them by label similarity. Unlike the other methods, TC-ESR identifies the closely related source task of Retinopathy through domain similarity, attaining an SR of 2.22%p compared with 20.57%p for the best fixed model. On the remaining targets, it still attains the lowest mean SR and ORNR among LOVM methods, and matches the best fixed model in SR while surpassing it in ORNR, although the best fixed model requires target accuracies. These results are consistent with the premise of TC-ESR that models performing well on source tasks similar to the target also tend to perform well on the target.