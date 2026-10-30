# DOSE: Dominance-Guided Ordinal Severity Estimation from Sparse Relative Annotations

Tushir Sahu Hideaki Hayashi Hajime Nagahara

The University of Osaka, Japan

tushir.sahu@is.ids.osaka-u.ac.jp, {hayashi, nagahara}@ids.osaka-u.ac.jp

## Abstract

Medical severity estimation commonly relies on absolute grade annotations, which arefundamentally ordinal, yet existing methods often treat the grades as independent categories. Learning-to-rank methods, which can be trained using relative annotations based on severity comparisons of image pairs, can provide continuous severity rankings but cannot directly predict discrete severity grades. We propose Dominance-guided Ordinal Severity Estimation from Sparse Relative Annotations (DOSE), a framework for learning ordinal medical severity from sparse pairwise comparisons without using absolute annotations. DOSE aggregates the observed comparisons through a global comparison graph to estimate a dominance score for each image. Differences between dominance scores set cumulative margins in the embedding space, so that pairs with larger estimated severity differences are separated more strongly, and also generate pseudo-pairs that supervise a scalar severity head alongside the observed comparisons. The learned continuous severity axis is converted into discrete grades via label-free score calibration, with tie-anchored segmentation as the main calibration method. In the experiment, DOSE outperformed pairwise-ranking baselines in most evaluated settings on ACNE04, IDRiD, and LIMUC under different pair budgets. Code is available at https://github.com/TushirSahu/DOSE

## 1. Introduction

In the medical domain, automatic image-based disease severity estimation has become an important problem in computer-aided diagnosis, as it can help clinicians assess disease progression and make treatment decisions. Many deep learning methods have been widely applied to this problem in different medical imaging domains, including retinal imaging, endoscopy, and facial photography [12, 22, 27, 40, 41]. There are several approaches to framing severity estimation using images; one method involves treating it simply as a standard classification problem, where images are assigned to one of several predefined, independent severity grades. Alternatively, to better account for the ordinal nature of disease severity, methods based on ordinal regression or ranking are also frequently considered. This is important because neighboring severity grades are usually more similar than distant grades, and an error between grades 1 and 2 should not be treated in the same way as an error between grades 1 and 4.

Most severity estimation methods rely on absolute annotations, in which a medical expert assigns a single discrete grade to each training image. This annotation process can be difficult because disease appearance often changes continuously, while the clinical grading scale divides this into a small number of discrete categories. Images near a grade boundary can therefore be ambiguous even for experienced clinicians, leading to inter-observer variability. Kalpathy-Cramer et al. [20] reported poor agreement when experts assigned absolute retinopathy-of-prematurity categories. In contrast, the agreement was considerably stronger when the same images were compared by relative disease severity. Similar observations have been reported in biomedical image assessment [17, 31].

Relative annotation provides an alternative supervision protocol in which an annotator labels an image pair according to which image is more severe, or marks the two as comparable. Such pairwise relations directly encode the ordered structure of disease severity without providing an absolute grade for either image. Previous work has shown that relative comparisons can be used to learn continuous disease severity rankings, and the annotation cost is lower than that of absolute annotation [19, 20]. However, the number of possible image pairs grows quadratically with the dataset size, making it important to learn effectively from sparse comparisons.

Learning-to-rank is the natural approach to exploiting these relative annotations, but standard pairwise ranking has two limitations for severity grading. First, a pairwise ranking loss captures only the direction of severity: given an ordered pair $( x _ { i } , x _ { j } )$ , a ranking loss encourages the score of the more severe image to be larger, but does not specify how much larger it should be. A mild–moderate pair and a mild–severe pair therefore impose essentially the same ordering constraint, although the latter should represent a larger ordinal separation. Second, the learned rank score is continuous, whereas many clinical practices require estimating discrete severity grades.

In this work, we ask whether both ordinal separation and discrete severity grades can be recovered from sparse relative comparisons without absolute-grade supervision. We answer this by proposing Dominance-guided Ordinal Severity Estimation from Sparse Relative Annotations (DOSE). To address the first limitation, DOSE aggregates the sparse comparison graph into global dominance scores, whose differences provide a label-free estimate of ordinal separation. These differences determine the cumulative margins in the embedding space so that images with similar estimated severity remain closer together, while more distant images are separated more. A scalar severity head is trained in parallel on the observed comparisons together with confident dominance-derived pseudopairs. To address the second, DOSE converts the learned continuous severity axis into discrete grades via label-free score calibration, using annotated ties as anchors that keep equally severe images within a single grade. DOSE thus yields both a continuous severity ordering and discrete ordinal predictions without training a label-dependent classifier.

The main contributions of this work are:

• We introduce a global dominance-based formulation that converts the sparse comparison structure into ordinal distance estimates, enabling cumulative multi-margin representation learning without absolute class labels.

• We propose dominance pseudo-pair supervision for the scalar severity head, which propagates transitive ordering information from the comparison graph to pairs that were not explicitly annotated.

• We propose tie-anchored segmentation (Tie-seg), a labelfree score-to-grade calibration, and compare it with GMM- and prior-based alternatives, as well as a knownprior oracle.

## 2. Related Work

## 2.1. Ordinal Learning for Medical Image Analysis

Most clinical severity scales are ordered, but the losses typically used to fit them are not. The cross-entropy loss treats grade 0 and grade 3 as two arbitrary grades, so a model pays the same penalty whether it confuses two adjacent stages or two stages at opposite ends of the scale. For disease grading, that is the wrong consideration, since mistaking mild for moderate rarely alters treatment, whereas misinterpreting mild for severe frequently does. Early works adapted neural networks to ordered categories by redesigning the output layer [11]. Niu et al. [29] proposed a K-way ordinal problem as a bank of binary subproblems solved jointly by a multiple-output CNN, and CORAL [5] sharpened the idea by forcing those subtasks to share a weight vector, which makes the predicted cumulative probabilities monotonic by construction and removes the rank inconsistencies that affect independent binary heads. CORN [37] relaxes the shared-weight constraint and models conditional rank probabilities instead, recovering consistency without tying the subtasks together. More recent work shifts the ordinal prior elsewhere entirely, with OrdinalCLIP [26] learning rank prompts in a language latent space, ensuring that the class prototypes emerge ordered by construction.

The same works are applied across different organs and modalities. Chen et al. [9] grade knee osteoarthritis with an ordinal loss on Kellgren-Lawrence scores. Wu et al. [42] jointly model acne severity and lesion count using label distribution learning, addressing overlap between neighboring grades. Polat et al. [33] proposed a loss function that takes the order of classes and the distances of the classes into consideration for estimating the ulcerative severity. MORF [24] couples a CNN with a differentiable regression forest inside a meta-learning loop. DOSE shares the same goal of preserving the ordered structure of severity grades.

These ordinal learning methods rely on absolute grades to define their training objective. In contrast, DOSE learns the ordinal structure from sparse relative comparisons and does not use absolute grades as training targets.

## 2.2. Contrastive and Ordinal Representation Learning

Contrastive learning structures the embedding space by bringing related samples closer and separating dissimilar ones [2, 8, 10, 13, 14, 21, 44], while ordinal representation methods further consider the order or distance between targets. Rank-N Contrast [45] preserves target order through rank-aware contrastive learning, and POE [25] models ordinal embeddings probabilistically to capture prediction uncertainty. SCOL [35] incorporates label-dependent ordinal distances into supervised contrastive learning, while SupReMix [43] uses ordinal information and pair hardness for contrastive regression. CLOC [32], which is particularly related to our work, uses multiple margins to separate samples according to their ordinal classes.

However, these methods derive the ordinal structure from absolute labels or target values, whereas DOSE estimates it from dominance differences recovered from sparse relative comparisons.

## 2.3. Pairwise Comparison and Dominance-Based Ranking

Pairwise comparison provides a natural way to recover an ordering when absolute scores or full rankings are difficult to obtain. Classical models such as Bradley–Terry [3] estimate a latent strength for each item from observed comparison outcomes. RankNet [4] extends this idea to neural models by learning a score function directly from a pairwise approach. Some simpler counting-based approaches aggregate the number of wins and losses associated with each item. Shah et al. [36] showed that such estimators can be both robust and statistically effective under suitable assumptions. Graph-based ranking methods make use of the global structure of the comparison set. Rank Centrality [28] represents pairwise outcomes as a Markov chain and derives item scores from its stationary distribution. This allows evidence to propagate through indirect comparison paths rather than being limited to directly observed pairs. Heckel et al. [16] studied the number of comparisons required to recover an approximate ranking, while GNNRank [15] learns global rankings from directed comparison graphs.

Also, relative supervision has been explored in computer vision. Parikh et al. [30] introduced relative attributes, where visual properties are learned from statements indicating that one category exhibits an attribute more strongly than another. Their work showed that pairwise relations can capture semantic structure that is difficult to express through independent categorical labels. Phelps et al. [31] found pairwise assessment to be more reliable than Likertstyle rating for biomedical image quality. Jang et al. [17] further showed that human-in-the-loop pairwise sorting can reduce the annotation burden associated with expert assessment. Pairwise ranking objectives have also been used directly for medical grading, including colorectal cancer grading in pathology images [23].

DOSE builds on the same pairwise supervision and global ranking principles but uses the resulting dominance scores differently. Rather than treating the graph-based rank as the final prediction, DOSE uses dominance as an intermediate source of supervision to estimate ordinal distances and generate pseudo-pairs.

## 2.4. Discrete Grades Without Labels

Learning an ordered representation and assigning a discrete grade are, although related, two separate problems. Supervised ordinal regression methods such as CORAL [5] and CORN [37] produce ordered predictions by using absolute ordinal labels during training. Pseudo-labeling and selftraining provide another route by treating confident model predictions as temporary supervision [38]. In settings where no class labels are available, deep clustering methods instead recover discrete groups from learned features. Self-supervised approaches, like DeepCluster [6], alternate between clustering visual features and using the resulting assignments to update the representation. Invariant Information Clustering [18] learns assignments by maximizing mutual information between predictions for related views of an image. SeLa [1] jointly learns representations and self-labels through an optimal transport formulation with a balanced assignment constraint. SwAV [7] performs online clustering and learns by predicting the cluster assignment of one image view from another. SCAN [39] follows a twostage formulation in which self-supervised representation learning is followed by neighbor-based semantic clustering. Together, these methods show that meaningful discrete partitions can be learned without absolute labels.

DOSE is related to these approaches in that it also obtains discrete outputs without using absolute grades as training targets. However, clustering methods produce unordered cluster identities, whereas the severity axis in DOSE already has a low-to-high direction determined by the relative comparisons.

## 3. Proposed Method

## 3.1. Overview

Fig. 1 shows the overall framework of DOSE. We consider a set of training images $\boldsymbol { \mathcal { X } } ~ = ~ \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N }$ with $K$ ordered severity levels, where supervision is available only through sparse pairwise comparisons. For a pair $( \mathbf { x } _ { i } , \mathbf { x } _ { j } )$ , we use $\mathbf { x } _ { i } \succ \mathbf { x } _ { j }$ when $\mathbf { x } _ { i }$ is more severe than $\mathbf { x } _ { j } , \mathbf { x } _ { i } \prec \mathbf { x } _ { j }$ when it is less severe, and $\mathbf { x } _ { i } \sim \mathbf { x } _ { j }$ when the two have comparable severity. The corresponding relative label is

$$
r _ { i j } = \left\{ \begin{array} { l l } { + 1 , } & { { \bf x } _ { i } \succ { \bf x } _ { j } , } \\ { - 1 , } & { { \bf x } _ { i } \prec { \bf x } _ { j } , } \\ { 0 , } & { { \bf x } _ { i } \sim { \bf x } _ { j } . } \end{array} \right.\tag{1}
$$

Our method consists of three main components. First, an image encoder produces an ordinal embedding z, which is used both for representation learning and by a scalar head to estimate a severity score. Second, the observed comparisons are aggregated through the dominance estimation module to obtain a global dominance score for each training image. The dominance differences are converted into ordinal distance levels for the cumulative multi-margin loss and are also used to construct confident pseudo-pairs for the severity head. Finally, after training, the learned severity scores are converted into K discrete grades using the score-calibration routes described in Sec. 3.5.

## 3.2. Encoder and Embedding Space

Given an input image $\mathbf { x } ,$ the encoder $f _ { \theta }$ produces a $d -$ dimensional embedding,

$$
\mathbf { z } = \frac { f _ { \theta } ( \mathbf { x } ) } { \| f _ { \theta } ( \mathbf { x } ) \| _ { 2 } } , \qquad s = \alpha _ { \psi } ( \mathbf { z } ) ,\tag{2}
$$

where $\mathbf { z } \in \mathbb { R } ^ { d }$ is the normalized embedding and $s \in \mathbb { R }$ is the continuous severity score.

![](images/5a8a9d0cfb4f936a6dd2b7811d58dfbc8429be1800803bbb5f98f4f39e8be871.jpg)  
Figure 1. The overall architecture of our proposed method DOSE

The embedding z is used by the ordinal representation loss, while the scalar head $\alpha _ { \psi }$ maps the same representation onto a one-dimensional severity axis. Here, the similarity between two embeddings is measured by $\begin{array} { r } { S _ { i j } = \frac { \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { j } } { \tau } } \end{array}$ where τ is the temperature parameter.

## 3.3. From Pairwise Signs to Ordinal Distances

Sparse pairwise annotations. Within a mini-batch of B images, we store the observed comparisons in a sparse matrix

$$
M _ { i j } = { \left\{ \begin{array} { l l } { r _ { i j } } & { { \mathrm { i f ~ o b s e r v e d , } } } \\ { \emptyset } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }\tag{3}
$$

Under a realistic annotation budget, most entries remain empty, and unobserved pairs do not contribute to the training.

Now, with B images in a batch, there are $\binom { B } { 2 } = B ( B -$ $1 ) / 2$ possible pairwise comparisons. It is computationally expensive to compute all such pairs and introduces unnecessary redundancy. Instead, during each pairwise-encoding round, we partition the mini-batch into disjoint pairs and consider only $\left\lfloor { \frac { B } { 2 } } \right\rfloor$ comparisons. Thus, each image participates in at most one comparison in a sampling round, reducing the number of encoded comparisons from $\mathcal { O } ( B ^ { 2 } )$ to $\mathcal { O } ( B )$ . However, this also limits the number of directly observed relationships. To address this, we created a dominance estimation module to mitigate this limitation.

Dominance Estimation Module. Fig. 2 illustrates the overall architecture of the dominance estimation module. A win-and-loss total calculated in a single batch B is noisy and removes transitive evidence. Even though the two images were never directly compared, $\mathbf { x } _ { a }$ should dominate $\mathbf { x } _ { c }$ if $\mathbf { x } _ { a } \succ \mathbf { x } _ { b }$ and ${ \bf x } _ { b } \succ { \bf x } _ { c }$ . We recover this by estimating dominance once over the entire comparison graph rather than batch by batch. Let $A _ { u v }$ count the observed comparisons in which $\mathbf { x } _ { u }$ is judged more severe than $\mathbf { x } _ { v }$ . We build a rowstochastic random walk toward more severe images and obtain its distribution after L iterations as a latent severity strength [28]. For observed non-tied pairs, the off-diagonal transitions are:

$$
P _ { u v } \propto \frac { A _ { v u } } { A _ { u v } + A _ { v u } } , \qquad \pi = \left( P ^ { \top } \right) ^ { L } \frac { { \bf 1 } } { N } .\tag{4}
$$

See supplementary for more details. An image that has won over other images many times in terms of severity should accumulate high probability mass, which is exactly the transitive signal that a batch-local tally misses. The resulting probabilities $\pi _ { i }$ can be highly skewed, so we operate with their ranks instead. The dominance score $q _ { i }$ is the rank of $\pi _ { i }$ normalized to [0, 1], the fraction of images with smaller stationary mass,

$$
q _ { i } = \frac { 1 } { N - 1 } \big | \big \{ u : \pi _ { u } < \pi _ { i } \big \} \big | .\tag{5}
$$

Working with ranks keeps q stable against the heavy-tailed values that π can take on in an imbalanced graph. This score aggregates transitive evidence across the whole graph and is used to define ordinal distances and pseudo-pairs, but never the final prediction.

Dominance pseudo-pairs. Only the pairs that are annotated are considered by $\mathcal { L } _ { \mathrm { r a n k } }$ . Since every image is already ordered by global dominance q, we use it to densify the pairwise signal at no additional annotation cost. A pseudo-pair with dominance label $\hat { r } _ { i j } = \mathrm { s i g n } ( q _ { i } - q _ { j } )$ ), treated as a fixed target, is formed by any batch pair with a consistent gap

![](images/a6255a4e8e17354892ba6e645e7f82e3983f610ff33cab8b46f3d865541fb578.jpg)  
Figure 2. The Dominance Estimation Module

$$
| q _ { i } - q _ { j } | > \tau _ { p p } .
$$

$$
\mathcal { L } _ { \mathrm { p p } } = \frac { 1 } { \vert \tilde { \mathcal { P } } \vert } \sum _ { ( i , j ) \in \tilde { \mathcal { P } } } \left[ m - \hat { r } _ { i j } \left( s _ { i } - s _ { j } \right) \right] _ { + } ,\tag{6}
$$

This carries the transitive ordering over to pairs that were never directly annotated, again without any absolute label.

Ordinal distance. For an observed pair we measure the dominance gap $\hat { d } _ { i j } = | q _ { i } - q _ { j } |$ and turn it into an ordinaldistance level in $\{ 1 , \ldots , K - 1 \}$ }. Rather than slice this gap on a fixed grid, we cut it at the empirical quantiles of the observed gaps, which keeps the levels balanced even when one grade dominates the data. Let $e _ { m }$ denote the $\frac { m } { K - 1 }$ empirical quantile of $\{ \hat { d } _ { \mathrm { u v } } \}$ taken over all observed non-tied pairs, so the $K - 2$ interior edges $\{ e _ { m } \} _ { m = 1 } ^ { K - 2 }$ divide the gap distribution into $K - 1$ equal-mass levels and every level holds a comparable share of pairs. The level of a pair is then the number of edges its gap exceeds,

$$
\tilde { d } _ { i j } = \mathrm { c l a m p } \Big ( 1 + \sum _ { m = 1 } ^ { K - 2 } \mathbb { I } \Big [ \hat { d } _ { i j } > e _ { m } \Big ] , 1 , K - 1 \Big ) ,\tag{7}
$$

with $\tilde { d } _ { i j } ~ = ~ 0$ for tied pairs, whereas $\tilde { d } _ { i j }$ records how strongly two images should be separated, giving the labelfree proxy for ordinal distance that the multi-margin loss consumes.

## 3.4. Learning the Embedding and the Severity Axis

Dominance-aware multi-margin loss. For each anchor $\mathbf { x } _ { i }$ we split the batch into positives $S _ { \mathrm { p o s } } ( i )$ with $M _ { i j } = 0$ and ordinal negatives $S _ { \mathrm { l o w } } ( i ) \cup S _ { \mathrm { h i g h } } ( i )$ with $M _ { i j } \neq 0$ . A single margin would push a grade-1 negative and a grade-3 negative equally far from the anchor, which is wrong for an ordinal target. We instead grow the margin with the ordinal distance. With learnable step margins $m _ { t } = \mathrm { S o f t p l u s } ( \theta _ { t } )$ a label-free proxy ordinal distance $\tilde { d } _ { i k }$ accumulates a cumulative margin $\begin{array} { r } { \Delta _ { i k } = \sum _ { t = 1 } ^ { d _ { i k } } m _ { t } . } \end{array}$ , so the required gap is monotonic in distance by construction. Over the triplet set ${ \mathcal { T } } = \{ ( i , j , k ) : j \in S _ { \mathrm { p o s } } ( i ) , k \in S _ { \mathrm { l o w } } ( i ) \cup S _ { \mathrm { h i g h } } ( i ) \}$ the loss is

![](images/44ab0ee5ca1634c7e3f713f4c722971a0ebcb5fdb24cc7013726e35f09cfd4db.jpg)  
Figure 3. Score-to-grade calibration routes. A learned scalar severity score is converted into discrete ordinal grades by placing $K - 1$ cut-offs on the score axis. Tie-seg, GMM, and Est-prior are label-free routes, while Known-prior uses the true training class proportions.

$$
\mathcal { L } _ { \mathrm { R M M } } = \frac { 1 } { | T | } \sum _ { ( i , j , k ) \in \mathcal { T } } \big [ \Delta _ { i k } - S _ { i j } + S _ { i k } \big ] _ { + } .\tag{8}
$$

We replace $\Delta _ { i k }$ with a single shared margin to recover a single-margin variant that we report only as an ablation in Sec. 5.3.

Severity ranking. In parallel, the scalar head is trained to assign higher scores to more severe images. With $\delta _ { i j } = $ $s _ { i } - s _ { j }$ over the non-tied set $\mathcal { P } _ { \neq } = \{ ( i , j ) : r _ { i j } \neq 0 \}$ ,

$$
\mathcal { L } _ { \mathrm { r a n k } } = \frac { 1 } { | \mathcal { P } _ { \neq } | } \sum _ { ( i , j ) \in \mathcal { P } _ { \neq } } \log \bigl ( 1 + \exp \bigl ( - r _ { i j } \delta _ { i j } \bigr ) \bigr ) .\tag{9}
$$

The three loss terms are optimized jointly to give us the total loss function as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { R M M } } + \lambda _ { \mathrm { r a n k } } \mathcal { L } _ { \mathrm { r a n k } } + \lambda _ { \mathrm { p p } } \mathcal { L } _ { \mathrm { p p } } . } \end{array}\tag{10}
$$

## 3.5. Reading Grades from the Severity Axis

After training, each image is represented by a continuous severity score $s _ { i } = \alpha _ { \psi } ( { \bf z } _ { i } )$ . This score provides an ordinal ranking, but clinical grading requires a discrete severity level. We obtain a grade by placing $K - 1$ ordered cut-offs $t _ { 1 } < \dots < t _ { K - 1 }$ on the score axis and counting how many cut-offs the score exceeds:

$$
\hat { y } _ { i } = \sum _ { k = 1 } ^ { K - 1 } \mathbb { I } [ s _ { i } > t _ { k } ] .\tag{11}
$$

Since no absolute grades are available during training, these cut-offs cannot be learned from labeled validation data. Instead, they are estimated from the learned score distribution and the relative global comparison structure. We study four cut-off placement routes as shown in Fig. 3. The first three are label-free. The fourth uses the true class prior and is reported only as known-prior (oracle). Let $s _ { ( 1 ) } \leq \cdots \leq s _ { ( N ) }$ denote the sorted training scores.

Tie-seg (prior-free). We split the sorted scores into $K$ contiguous bins $B _ { 1 } , \ldots , B _ { K }$ and pick the split that keeps each bin tight while keeping equally graded images together.

$$
\operatorname* { m i n } _ { B _ { 1 } , \ldots , B _ { K } } \sum _ { b = 1 } ^ { K } \sum _ { s \in B _ { b } } ( s - \mu _ { b } ) ^ { 2 } + \lambda _ { \mathrm { t i e } } \sum _ { ( i , j ) \in \mathcal { T } } \mathbb { I } [ B ( i ) \neq B ( j ) ] ,\tag{12}
$$

where $B ( i )$ is the bin of image $i , \mu _ { b }$ is the mean score of bin $B _ { b } ,$ and $\tau$ is the set of training tie pairs. The first term is the usual within-bin variance and favors compact bins. The second penalizes every tie the boundaries pull apart by $\lambda _ { t i e } .$ , so a pair annotated equally severe resists being split across a grade line. Ties act as soft anchors that hold the same-grade images within a single bin. Since the scores are one-dimensional and the cost is additive across bins, the optimal set of boundaries is obtained efficiently by sorting scores, without any threshold search. Each cut-off $t _ { k }$ sits at the midpoint of the two adjacent scores that straddle a boundary. This route assumes neither a class prior nor a density model and applies to practical conditions.

GMM. As a label-free parametric alternative, we fit a $K$ component one-dimensional Gaussian mixture to the training scores,

$$
p ( s ) = \sum _ { k = 1 } ^ { K } \pi _ { k } \mathcal { N } ( s ; \mu _ { k } , \sigma _ { k } ^ { 2 } ) .\tag{13}
$$

The mixture is fitted without using absolute labels. After fitting, we sort the components by their means, $\mu _ { 1 } < \cdots <$ $\mu _ { K }$ , so that the components follow the learned severity direction. We then place the cut-off between two adjacent grades at the midpoint of neighboring component means,

$$
t _ { k } = \frac 1 2 ( \mu _ { k } + \mu _ { k + 1 } ) , \qquad k = 1 , \ldots , K - 1 .\tag{14}
$$

This route assumes that each ordinal grade forms a distinct compact bin on the score axis. It is fully label-free, but it can become unstable when the score distribution is imbalanced or when adjacent grades overlap strongly.

Est-prior. The GMM fit also gives mixture weights $\pi _ { k } ,$ which provide a label-free estimate of the class proportions. Instead of directly using the Gaussian component locations as boundaries, this route uses only the estimated prior. After sorting the mixture components by their means, we treat the reordered weights as estimated grade frequencies and place cut-offs at the corresponding empirical score quantiles:

$$
\begin{array} { r } { t _ { k } = Q \Big ( \sum _ { l = 1 } ^ { k } \pi _ { l } \Big ) , \qquad k = 1 , \ldots , K - 1 , } \end{array}\tag{15}
$$

where $Q ( \cdot )$ is the empirical quantile function of the training scores. This separates the role of estimating the class prior from the role of assigning samples to Gaussian components. In other words, it tests whether an estimated labelfree prior is sufficient to recover useful grade boundaries on the learned severity axis.

Known-prior. Finally, we report an oracle quantile route that uses the true class proportions $p _ { k }$ from the training set:

$$
\begin{array} { r } { t _ { k } = Q \Big ( \sum _ { l = 1 } ^ { k } p _ { l } \Big ) . } \end{array}\tag{16}
$$

This route is not a label-free method. It is included only as an upper-bound reference, as it shows how well the learned severity axis can perform if the true class prior is available.

## 4. Experiments

We evaluate DOSE on three medical image datasets across domains such as endoscopy, retinal fundus imaging, and facial acne photography. The experiments examine whether sparse relative comparisons are sufficient to learn a continuous severity score. Since the benchmark datasets provide absolute grades, we use the training labels only to construct pairwise relations in the form of more severe, less severe, or comparable. The absolute labels are retained only for the evaluation, and true class distribution is used for the known prior, which is reported separately from the label-free calibration routes.

## 4.1. Datasets

LIMUC. The Labeled Images for Ulcerative Colitis (LIMUC) [33] dataset is a publicly available colonoscopy dataset used to assess ulcerative colitis severity. It contains 11,276 images collected from 564 patients and 1,043 colonoscopy procedures.

IDRiD. The Indian Diabetic Retinopathy Image Dataset (IDRiD) [34] contains retinal fundus images labeled according to the diabetic retinopathy grading scale. The scale contains five ordinal levels, from grade 0 to grade 4.

ACNE. ACNE04 [42] (simply referred to as ACNE) is a facial acne severity dataset containing 1,457 images with image-level ordinal acne grades. Although lesion-level bounding boxes are also provided, we use only the imagelevel severity grades, since our objective is ordinal severity estimation rather than lesion localization.

## 4.2. Relative Annotation Protocol

All three datasets provide absolute grades, which we use only to synthesize relative labels. For a pair budget of $p \% ,$ we sample $p \%$ of the $N ( N - 1 ) / 2$ possible training pairs from the training set to form a candidate comparison pool and label each pair as +1, −1, or 0 according to whether the first image has a higher, lower, or equal grade. The absolute grades are never given to the model as targets and they are used only to evaluate test predictions and, for the known-prior oracle, to compute the training class proportions. Each experiment is repeated with five different seeds for pair sampling and model training, and we report the mean and standard deviation over the three runs.

## 4.3. Implementation Details

The encoder uses an ImageNet-pretrained ConvNeXt-Tiny backbone followed by projection layers that produce a 128- dimensional embedding. A lightweight scalar head maps this embedding to the continuous severity score. We set $\lambda _ { \mathrm { r a n k } } = \lambda _ { \mathrm { p p } } = 0 . 5 , \tau = 0 . 0 7 , L = 3 0$ and use a margin of $m = 0 . 5$ for the pseudo-pair loss. The cumulative margin steps are parameterized using Softplus, initialized between 0.5 and 2.0 across the K − 1 classes’ ordinal distances. The pseudo-pairs are selected using $\tau _ { \mathrm { p p } } = 0 . 1$ . During training, it uses resizing, horizontal flipping, rotation, and mild color jitter. All models are trained for 150 epochs with a batch size of 32, a learning rate of $1 0 ^ { - 4 }$ , weight decay of $1 0 ^ { - 4 }$ and cosine learning-rate decay. For Tie-seg, we use $\lambda _ { \mathrm { t i e } } =$ 1.0 with a minimum bin size 1% of the training samples.

## 4.4. Evaluation Metrics

We report accuracy and quadratic weighted kappa (QWK) since they capture complementary aspects of ordinal prediction. We also report the mean absolute error (MAE) in the supplementary material. Although accuracy measures exact agreement, it does not distinguish an adjacent-grade error from an error spanning several severity levels. Here, the QWK accounts for the ordered nature of the grades by assigning a larger penalty to predictions that are farther from the reference grade.

## 5. Results

## 5.1. Comparison with Fully Supervised and Pairwise-Ranking Baselines

Table 1 compares DOSE with a fully supervised crossentropy classifier using the same ConvNeXt-Tiny backbone and a linear layer on top of the backbone. The supervised model uses every absolute label, whereas DOSE is trained only on sparse relative comparisons. As expected, full supervision yields higher accuracy on ACNE and IDRiD, with gains of 3.8% and 2.1% over the strongest DOSE result, respectively. However, DOSE remains close in QWK, differing only by 0.001 on ACNE and 0.004 on IDRiD. On LIMUC, DOSE matches or slightly exceeds the supervised classifier, reaching 78.1% accuracy and a QWK of 0.862. These results show that sparse relative supervision can retain much of the performance obtained with complete absolute labels supervision.

We further compare DOSE with MarginRank and RankNet using the ConvNeXt-Tiny backbone and the labelfree Tie-seg route. These methods use the same relative comparisons and pair budgets; therefore, the comparison isolates the effect of the proposed dominance-guided objective. DOSE achieves the highest accuracy and QWK across almost all evaluated settings, with a few exceptions. The improvement is particularly clear on IDRiD, where DOSE reaches 84.0–84.5% accuracy, compared with 81.1–81.9% for MarginRank and approximately 60% for RankNet. Also, the consistent gains on ACNE and LIMUC further show that DOSE improves both accuracy and ordinal agreement.

## 5.2. Comparison of Calibration Routes

Table 2 compares the four score-to-grade routes using DOSE with ConvNeXt-Tiny. The label-free routes perform similarly on ACNE and LIMUC, while Tie-seg is the most stable route on IDRiD across all pair budgets. Tie-seg also remains close to the known prior, showing that the learned severity axis can be converted into discrete grades without access to the true class distribution.

## 5.3. Ablation of the Margin Formulation

Table S2 compares the cumulative multi-margin objective with a single shared margin using ConvNeXt-Tiny. All other training and calibration components are unchanged. Multi-margin improves Tie-seg accuracy in eight of nine settings, with average gains of 3.1, 0.6, and 1.5 % points on ACNE, IDRiD, and LIMUC, respectively. It also improves QWK, reduces MAE, and achieves higher known-prior accuracy across every pair budget. These results indicate that distance-dependent margins yield a more clearly separated ordinal representation of severity. Fig. 4 shows the qualitative behavior of our multi-margin loss on the embedding z when compared with the single-margin-style loss. Complete results across backbones, including QWK and MAE, are provided in the supplementary material.

## 6. Conclusion

We proposed a novel framework for estimating ordinal medical severity grades using pairwise comparisons rather than relying on absolute grades. The comparison graph is used in our method to compute global dominance scores, which are then converted into ordinal distance levels. We also make use of dominance-based pseudo-pairs to provide additional supervision for a continuous-severity head. The discrete grades are then extracted from the learned severity axis through three different label-free calibration routes and a known-prior oracle. We also created a multi-margin using relative labels, which outperforms a single margin across almost all settings, demonstrating that images with larger estimated severity differences should receive greater separation. The Tie-seg route remains close to the known prior across all backbones and pair budgets (see supplementary). In case of an imbalanced score distribution, it is more stable than density-based calibration. However, the relative labels are synthesized from absolute labels, so they contain no real annotator disagreement, and at a very low pair budget, the comparison graph becomes too sparse for reliable dominance estimates. Our future work will be to evaluate DOSE on real expert comparisons with disagreement, and to reduce the number of required comparisons through active pair selection and model-comparison uncertainty.

Table 1. Main comparison on ConvNeXt-Tiny. Relative-supervision methods use sparse pairwise comparisons with Tie-seg calibration, while cross-entropy uses all absolute grades. Best relative-supervision results are in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Supervision Budget</td><td rowspan="2"></td><td colspan="2">ACNE (K=4)</td><td colspan="2">IDRiD (K=5)</td><td colspan="2">LIMUC (K=4)</td></tr><tr><td>Accuracy ↑</td><td> $\mathrm { Q W K \uparrow }$ </td><td>Accuracy ↑</td><td> $\mathrm { Q W K \uparrow }$ </td><td>Accuracy ↑</td><td>QWK↑</td></tr><tr><td>Cross-entropy</td><td>Absolute</td><td>100%</td><td> $7 8 . 6 \pm 0 . 8$ </td><td> $0 . 8 2 2 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $8 6 . 6 \pm 0 . 4$ </td><td> $0 . 9 1 4 \pm 0 . 0 0 8$ </td><td> $7 8 . 0 { \pm } 0 . 6 $ </td><td> $0 . 8 4 8 { \scriptstyle \pm 0 . 0 1 1 }$ </td></tr><tr><td rowspan="5">MarginRank</td><td></td><td>5%</td><td> $6 8 . 6 \pm 1 . 5$ </td><td> $0 . 7 4 3 \pm 0 . 0 0 9$ </td><td> $7 3 . 6 \pm 7 . 8$ </td><td> $0 . 8 1 2 \pm 0 . 1 0 7$ </td><td> $5 5 . 7 \pm 0 . 2 $ </td><td> $0 . 2 5 2 { \scriptstyle \pm 0 . 0 2 0 }$ </td></tr><tr><td>Relative</td><td>10%</td><td> $6 9 . 5 \pm 3 . 4$ </td><td> $0 . 7 5 5 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td> ${ \bf 8 0 . 5 \pm 1 . 1 }$ </td><td> $0 . 8 8 2 \pm 0 . 0 0 8$ </td><td> $7 0 . 3 \pm 1 . 1$ </td><td>0.671 ±0.046</td></tr><tr><td></td><td>30%</td><td> $7 1 . 2 \pm 0 . 3$ </td><td> $0 . 7 8 8 { \pm } 0 . 0 0 9$ </td><td> $8 1 . 2 \pm 0 . 5$ </td><td> $0 . 8 7 8 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $7 3 . 9 \pm 0 . 8$ </td><td>0.844±0.008</td></tr><tr><td></td><td>50%</td><td> $7 2 . 6 \pm 2 . 4$ </td><td> $0 . 7 9 9 { \pm } 0 . 0 2 0$ </td><td> $8 1 . 1 \pm 0 . 1$ </td><td> $0 . 8 7 8 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $7 4 . 0 { \pm } 0 . 6 $ </td><td> $0 . 8 4 7 \pm 0 . 0 0 5$ </td></tr><tr><td></td><td>70%</td><td> $7 3 . 1 \pm 1 . 5$ </td><td> $0 . 7 8 6 \pm 0 . 0 2 3$ </td><td> $8 1 . 9 \pm 1 . 5$ </td><td> $0 . 8 9 0 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $7 6 . 1 \pm 0 . 4$ </td><td> $0 . 8 3 9 \pm 0 . 0 0 4$ </td></tr><tr><td rowspan="5">RankNet</td><td rowspan="5">Relative</td><td>5%</td><td> $6 8 . 1 \pm 1 . 2$ </td><td> $0 . 5 1 4 \pm 0 . 0 0 9$ </td><td> $7 2 . 1 \pm 9 . 1$ </td><td> $\mathbf { 0 . 8 4 1 \pm 0 . 0 0 4 }$ </td><td> ${ \bf 6 9 . 4 \pm 0 . 8 }$ </td><td> $\mathbf { 0 . 5 9 9 } \pm \mathbf { 0 . 0 0 5 }$ </td></tr><tr><td>10%</td><td> $6 6 . 7 \pm 0 . 9$ </td><td> $0 . 4 8 7 \pm 0 . 0 1 1$ </td><td> $7 2 . 3 \pm 8 . 6$ </td><td> $0 . 8 4 1 \pm 0 . 0 0 6$ </td><td> $7 1 . 0 \pm 2 . 9$ </td><td> $0 . 6 7 4 \pm 0 . 1 0 5$ </td></tr><tr><td>30%</td><td> $6 5 . 5 \pm 1 . 7$ </td><td> $0 . 4 7 2 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $6 0 . 4 \pm 0 . 6$ </td><td> $0 . 8 6 1 \pm 0 . 0 0 4$ </td><td> $7 4 . 4 \pm 0 . 4$ </td><td> $0 . 8 1 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr><tr><td>50%</td><td> $6 6 . 0 { \pm } 0 . 9$ </td><td> $0 . 4 8 7 \pm 0 . 0 1 3$ </td><td> $6 1 . 0 { \pm } 0 . 5 $ </td><td> $0 . 8 6 4 \pm 0 . 0 0 2$ </td><td> $7 4 . 4 \pm 0 . 8$ </td><td> $0 . 8 1 9 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr><tr><td>70%</td><td> $6 6 . 2 \pm 2 . 2$ </td><td> $0 . 4 8 8 \pm 0 . 0 2 9$ </td><td> $5 9 . 8 \pm 0 . 0$ </td><td> $0 . 8 6 1 \pm 0 . 0 0 1$ </td><td> $7 4 . 1 \pm 0 . 6$ </td><td> $0 . 8 1 9 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td rowspan="5">DOSE</td><td rowspan="5">Relative</td><td>5%</td><td> ${ \bf 6 9 . 3 \pm 1 . 0 }$ </td><td> $\mathbf { 0 . 7 5 5 \pm 0 . 0 0 9 }$ </td><td> ${ \bf 7 8 . 4 \pm 0 . 4 }$ </td><td> $0 . 8 3 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $5 6 . 4 \pm 0 . 3$ </td><td> $0 . 4 9 8 \pm 0 . 1 4 1$ </td></tr><tr><td>10%</td><td> ${ \bf 7 0 . 2 \pm 0 . 9 }$ </td><td> $\mathbf { 0 . 7 6 2 \pm 0 . 0 0 9 }$ </td><td> $7 9 . 6 { \pm } 0 . 8 $ </td><td>0.900 ±0.006</td><td> ${ \bf 7 5 . 2 \pm 0 . 8 }$ </td><td>0.821 ±0.008</td></tr><tr><td>30%</td><td> $7 2 . 6 \pm 1 . 2$ </td><td> $\mathbf { 0 . 8 0 7 \pm 0 . 0 1 1 }$ </td><td> ${ \bf 8 4 . 5 \pm 0 . 7 }$ </td><td> $\mathbf { 0 . 9 0 9 } \pm \mathbf { 0 . 0 1 0 }$ </td><td> ${ 7 7 . 5 \pm 0 . 4 }$ </td><td> $\mathbf { 0 . 8 5 7 \pm 0 . 0 0 3 }$ </td></tr><tr><td>50%</td><td> ${ \bf 7 4 . 8 \pm 0 . 3 }$ </td><td> $\mathbf { 0 . 8 2 1 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 8 4 . 0 \pm 0 . 4 }$ </td><td> $\mathbf { 0 . 9 0 3 \pm 0 . 0 0 3 }$ </td><td> ${ \bf 7 8 . 0 \pm 0 . 6 }$ </td><td> $\mathbf { 0 . 8 6 2 \pm 0 . 0 0 4 }$ </td></tr><tr><td>70%</td><td> ${ \bf 7 4 . 1 } \pm 2 . 0$ </td><td> $\mathbf { 0 . 8 0 7 \pm 0 . 0 2 1 }$ </td><td> ${ \bf 8 4 . 0 \pm 0 . 4 }$ </td><td> $\mathbf { 0 . 9 1 0 \bot 0 . 0 0 8 }$ </td><td> ${ \bf 7 8 . 1 \pm 0 . 4 }$ </td><td> $\mathbf { 0 . 8 5 9 \pm 0 . 0 0 4 }$ </td></tr></table>

Table 2. Calibration-route comparison using ConvNeXt-Tiny. Tieseg, GMM, and Est-prior are label-free; Known-prior uses the true training class prior.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Budget</td><td colspan="3">Label-free</td><td>Oracle</td></tr><tr><td>Tie-seg</td><td>GMM</td><td>Est-prior</td><td>Known-prior</td></tr><tr><td rowspan="3">ACNE</td><td>30%</td><td> $7 2 . 6 \pm 1 . 2$ </td><td> ${ \bf 7 3 . 6 \pm 1 . 2 }$ </td><td> ${ \bf 7 3 . 6 \pm 0 . 0 }$ </td><td> $7 3 . 3 \pm 0 . 9$ </td></tr><tr><td>50%</td><td> $7 4 . 8 \pm 0 . 3$ </td><td> $7 5 . 0 \pm 1 . 0$ </td><td> ${ \bf 7 5 . 7 \pm 0 . 6 }$ </td><td> $7 5 . 5 \pm 0 . 9$ </td></tr><tr><td>70%</td><td> $7 4 . 1 \pm 2 . 0$ </td><td> ${ \bf 7 5 . 1 \pm 0 . 8 }$ </td><td> $7 4 . 4 \pm 0 . 8$ </td><td> $7 4 . 9 \pm 1 . 3$ </td></tr><tr><td rowspan="3">IDRiD</td><td>30%</td><td> ${ \bf 8 4 . 5 \pm 0 . 7 }$ </td><td> $8 4 . 2 \pm 0 . 5$ </td><td> $8 4 . 3 \pm 1 . 1$ </td><td> $8 4 . 7 \pm 0 . 3$ </td></tr><tr><td>50%</td><td> ${ \bf 8 4 . 0 \pm 0 . 4 }$ </td><td> $8 3 . 6 \pm 1 . 0$ </td><td> $8 3 . 9 \pm 0 . 7$ </td><td> $8 4 . 7 \pm 0 . 8$ </td></tr><tr><td>70%</td><td> ${ \bf 8 4 . 0 \pm 0 . 4 }$ </td><td> $8 3 . 8 \pm 0 . 7$ </td><td> $8 3 . 1 \pm 0 . 7$ </td><td> $8 4 . 8 \pm 0 . 5$ </td></tr><tr><td rowspan="3">LIMUC</td><td>30%</td><td> $7 7 . 5 \pm 0 . 4$ </td><td> ${ \bf 7 7 . 6 \pm 0 . 3 }$ </td><td> $7 6 . 2 \pm 0 . 6$ </td><td> $7 7 . 6 \pm 0 . 5$ </td></tr><tr><td>50%</td><td> ${ \bf 7 8 . 0 \pm 0 . 6 }$ </td><td> ${ \bf 7 8 . 0 \pm 0 . 6 }$ </td><td> $7 6 . 6 \pm 0 . 3$ </td><td> $7 8 . 1 \pm 0 . 6 $ </td></tr><tr><td>70%</td><td> $7 8 . 1 \pm 0 . 4$ </td><td> ${ \bf 7 8 . 2 \pm 0 . 1 }$ </td><td> $7 7 . 3 \pm 0 . 6$ </td><td> $7 8 . 4 \pm 0 . 2 $ </td></tr></table>

Table 3. Margin ablation with ConvNeXt-Tiny. Best results are in bold.
<table><tr><td rowspan="3">Dataset</td><td rowspan="3">Budget</td><td colspan="4">Tie-seg</td><td colspan="2">Known-prior</td></tr><tr><td colspan="2">Acc. ↑ (%)</td><td colspan="2">QWK↑</td><td colspan="2">Acc. ↑(%)</td></tr><tr><td>Multi</td><td>Single</td><td>Multi</td><td>Single</td><td>Multi</td><td>Single</td></tr><tr><td rowspan="3">ACNE</td><td>30%</td><td> ${ 7 2 . 6 \pm 1 . 2 }$ </td><td> $6 9 . 4 \pm 2 . 6$ </td><td>0.807</td><td>0.768</td><td> ${ \bf 7 3 . 3 \pm 0 . 9 }$ </td><td> $7 0 . 3 \pm 1 . 7$ </td></tr><tr><td>50%</td><td>74.8 ±0.3</td><td> $7 1 . 3 \pm 2 . 5$ </td><td>0.821</td><td>0.785</td><td> ${ \bf 7 5 . 5 \pm 0 . 9 }$ </td><td> $7 1 . 1 \pm 1 . 5$ </td></tr><tr><td>70%</td><td>74.1 ±2.0</td><td> $7 1 . 6 \pm 1 . 2$ </td><td>0.807</td><td>0.786</td><td> ${ \bf 7 4 . 9 \pm 1 . 3 }$ </td><td> $7 1 . 1 \pm 1 . 7$ </td></tr><tr><td rowspan="3">IDRiD</td><td>30%</td><td>84.5 ±0.7</td><td> $8 2 . 8 \pm 1 . 3$ </td><td>0.909</td><td>0.897</td><td> ${ \bf 8 4 . 7 \pm 0 . 3 }$ </td><td> $8 2 . 6 \pm 0 . 8$ </td></tr><tr><td>50%</td><td>84.0 ±0.4</td><td> $8 3 . 7 \pm 1 . 0$ </td><td>0.903</td><td>0.902</td><td>84.7 ±0.8</td><td> $8 2 . 5 \pm 1 . 5$ </td></tr><tr><td>70%</td><td> $8 4 . 0 \pm 0 . 4$ </td><td> ${ \bf 8 4 . 3 \pm 1 . 0 }$ </td><td>0.910</td><td>0.906</td><td> ${ \bf 8 4 . 8 \pm 0 . 5 }$ </td><td> $8 3 . 1 \pm 1 . 0$ </td></tr><tr><td rowspan="3">LIMUC</td><td>30%</td><td>77.5 ±0.4</td><td>76.9±0.3</td><td>0.857</td><td>0.840</td><td> ${ 7 7 . 6 \pm 0 . 5 }$ </td><td> $7 7 . 1 \pm 0 . 4$ </td></tr><tr><td>50%</td><td>78.0 ±0.6</td><td>75.7±0.1</td><td>0.862</td><td>0.826</td><td> ${ \bf 7 8 . 1 \pm 0 . 6 }$ </td><td> $7 6 . 0 \pm 0 . 0$ </td></tr><tr><td>70%</td><td>78.1 ±0.4</td><td>76.5 ±0.7</td><td>0.859</td><td>0.835</td><td> ${ \bf 7 8 . 4 \pm 0 . 2 }$ </td><td> $7 7 . 1 \pm 0 . 4$ </td></tr></table>

![](images/0cd0217abdc3589960c7316a15c3a1c9acb1f2951317d8a18d6baef6b47404ae.jpg)  
Figure 4. Comparison of the embedding. Left: our multi-margin loss. Right: single-margin variant. This comparison uses the LIMUC dataset with a 30% pair budget.

## Acknowledgments

This work was supported in part by the Japan Society for the Promotion of Science (JSPS) KAKENHI under Grant Number JP24K03010.

## References

[1] Yuki M. Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2020. 3

[2] Adrien Bardes, Jean Ponce, and Yann LeCun. VI-CReg: Variance-invariance-covariance regularization for self-supervised learning. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2022. 2

[3] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 3

[4] Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. Learning to rank using gradient descent. In Proceedings of the International Conference on Machine learning (ICML), pages 89– 96, 2005. 3

[5] Wenzhi Cao, Vahid Mirjalili, and Sebastian Raschka. Rank consistent ordinal regression for neural networks with application to age estimation. Pattern Recognition Letters, 140: 325–331, 2020. 2, 3

[6] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Proceedings of the European Conference on Computer Vision (ECCV), pages 132–149, 2018. 3

[7] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. Advances in Neural Information Processing Systems (NeurIPS), 33:9912–9924, 2020. 3

[8] Mathilde Caron, Hugo Touvron, Ishan Misra, et al. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision (ICCV), 2021. 2

[9] Pingjun Chen, Linlin Gao, Xiaoshuang Shi, Kyle Allen, and Lin Yang. Fully automatic knee osteoarthritis severity grading using deep neural networks with a novel ordinal loss. Computerized Medical Imaging and Graphics, 75:84–92, 2019. 2

[10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the International Conference on Machine Learning (ICML), 2020. 2

[11] Jianlin Cheng, Zheng Wang, and Gianluca Pollastri. A neural network approach to ordinal regression. In Proceedings of the International Joint Conference on Neural Networks (IJCNN), pages 1279–1284. IEEE, 2008. 2

[12] Bum-Joo Cho, Chang Seok Bang, Se Woo Park, Young Joo Yang, Seung In Seo, Hyun Lim, Woon Geon Shin, Ji Taek Hong, Yong Tak Yoo, Seok Hwan Hong, Jae Choi, Jun Lee,

and Gwang Baik. Automated classification of gastric neoplasms in endoscopic images using a convolutional neural network. Endoscopy, 51(12):1121–1129, 2019. 1

[13] Jean-Bastien Grill, Florian Strub, Florent Altche, et al. Boot-´ strap your own latent: A new approach to self-supervised learning. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 2

[14] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), 2020. 2

[15] Yixuan He, Quan Gan, David Wipf, Gesine D Reinert, Junchi Yan, and Mihai Cucuringu. GNNRank: Learning global rankings from pairwise comparisons via directed graph neural networks. In Proceedings of the International Conference on Machine Learning (ICML), pages 8581– 8612. PMLR, 2022. 3

[16] Reinhard Heckel, Max Simchowitz, Kannan Ramchandran, and Martin Wainwright. Approximate ranking from pairwise comparisons. In Proceedings ofthe International Conference on Artificial Intelligence and Statistics (AISTATS), pages 1057–1066. PMLR, 2018. 3

[17] Ikbeom Jang, Garrison Danley, Ken Chang, and Jayashree Kalpathy-Cramer. Decreasing annotation burden of pairwise comparisons with human-in-the-loop sorting: Application in medical image artifact rating. 2021. 1, 3

[18] Xu Ji, Joao F Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9865– 9874, 2019. 3

[19] Takeaki Kadota, Hideaki Hayashi, Ryoma Bise, Kiyohito Tanaka, and Seiichi Uchida. Deep Bayesian active learningto-rank with relative annotation for estimation of ulcerative colitis severity. Medical Image Analysis, 97:103262, 2024. 1

[20] Jayashree Kalpathy-Cramer, J Peter Campbell, Deniz Er dogmus, Peng Tian, Dharanish Kedarisetti, Chace Moleta, James D Reynolds, Kelly Hutcheson, Michael J Shapiro, Michael X Repka, et al. Plus disease in retinopathy of pre maturity: improving diagnosis by ranking disease severity and using quantitative image analysis. Ophthalmology, 123 (11):2345–2351, 2016. 1

[21] Prannay Khosla, Piotr Teterwak, Chen Wang, et al. Supervised contrastive learning. In Advances in Neural Informa tion Processing Systems (NeurIPS), 2020. 2

[22] Eyal Klang, Ana Grinman, Shelly Soffer, Reuma Margalit Yehuda, Oranit Barzilay, Michal Marianne Amitai, Eli Konen, Shomron Ben-Horin, Rami Eliakim, Yiftach Barash, et al. Automated detection of crohn’s disease intestinal strictures on capsule endoscopy images using deep neural networks. Journal ofCrohn’s and Colitis, 15(5):749–756, 2021. 1

[23] Trinh Thi Le Vuong, Kyungeun Kim, Boram Song, and Jin Tae Kwak. Ranking loss: a ranking-based deep neural network for colorectal cancer grading in pathology images. In Proceedings of the International Conference on

Medical Image Computing and Computer-Assisted Intervention (MICCAI), pages 540–549. Springer, 2021. 3

[24] Yiming Lei, Haiping Zhu, Junping Zhang, and Hongming Shan. Meta ordinal regression forest for medical image classification with ordinal labels. Journal ofAutomatica Sinica, 9(7):1233–1247, 2022. 2

[25] Wanhua Li, Xiaoke Huang, Jiwen Lu, Jianjiang Feng, and Jie Zhou. Learning probabilistic ordinal embeddings for uncertainty-aware regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13896–13905, 2021. 2

[26] Wanhua Li, Xiaoke Huang, Zheng Zhu, Yansong Tang, Xiu Li, Jiwen Lu, and Jie Zhou. Ordinalclip: Learning rank prompts for language-guided ordinal regression. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2

[27] Ziying Vanessa Lim, Farhan Akram, Cuong Phuc Ngo, Amadeus Aristo Winarto, Wei Qing Lee, Kaicheng Liang, Hazel Hweeboon Oon, Steven Tien Guan Thng, and Hwee Kuan Lee. Automated grading of acne vulgaris by deep learning with convolutional neural networks. Skin Research and Technology, 26(2):187–192, 2020. 1

[28] Sahand Negahban, Sewoong Oh, and Devavrat Shah. Rank centrality: Ranking from pairwise comparisons. Operations Research, 65(1):266–287, 2017. 3, 4

[29] Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, and Gang Hua. Ordinal regression with multiple output cnn for age estimation. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4920– 4928, 2016. 2

[30] Devi Parikh and Kristen Grauman. Relative attributes. In Proceedings of the International Conference on Computer Vision (ICCV), pages 503–510. IEEE, 2011. 3

[31] Andrew S Phelps, David M Naeger, Jesse L Courtier, Jack W Lambert, Peter A Marcovici, Javier E Villanueva-Meyer, and John D MacKenzie. Pairwise comparison versus likert scale for biomedical image assessment. American Journal of Roentgenology, 204(1):8–14, 2015. 1, 3

[32] Dileepa Pitawela, Gustavo Carneiro, and Hsiang-Ting Chen. CLOC: Contrastive learning for ordinal classification with multi-margin N-pair loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15538–15548, 2025. 2

[33] Gorkem Polat, Haluk Tarik Kani, Ilkay Ergenc, Yesim Ozen Alahdab, Alptekin Temizel, and Ozlen Atug. Improving the computer-aided estimation of ulcerative colitis severity according to mayo endoscopic score by using regressionbased deep learning. Inflammatory Bowel Diseases, 29(9): 1431–1439, 2023. 2, 6

[34] Prasanna Porwal, Samiksha Pachade, Ravi Kamble, Manesh Kokare, Girish Deshmukh, Vivek Sahasrabuddhe, and Fabrice Meriaudeau. Indian diabetic retinopathy image dataset (IDRiD): a database for diabetic retinopathy screening research. Data, 3(3):25, 2018. 6

[35] Afsah Saleem, Zaid Ilyas, David Suter, Ghulam Mubashar Hassan, Siobhan Reid, John T Schousboe, Richard Prince, William D Leslie, Joshua R Lewis, and Syed Zulqarnain Gi-

lani. SCOL: supervised contrastive ordinal loss for abdominal aortic calcification scoring on vertebral fracture assessment scans. In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Inter vention (MICCAI), pages 273–283. Springer, 2023. 2

[36] Nihar B Shah and Martin J Wainwright. Simple, robust and optimal ranking from pairwise comparisons. Journal ofMa chine Learning Research, 18(199):1–38, 2018. 3

[37] Xintong Shi, Wenzhi Cao, and Sebastian Raschka. Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. Pattern Analysis and Applica tions, 26(3):941–955, 2023. 2, 3

[38] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi supervised learning with consistency and confidence. Advances in Neural Information Processing Systems (NeurIPS), 33:596–608, 2020. 3

[39] Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. SCAN: Learning to classify images without labels. In Proceedings of the European Conference on Computer Vision (ECCV), pages 268–285. Springer, 2020. 3

[40] Yuelin Wang, Miao Yu, Bojie Hu, Xuemin Jin, Yibin Li, Xiao Zhang, Yongpeng Zhang, Di Gong, Chan Wu, Bilei Zhang, et al. Deep learning-based detection and stage grading for optimising diagnosis of diabetic retinopathy. Dia betes/Metabolism Research and Reviews, 37(4):e3445, 2021. 1

[41] Patrick Wienholt, Alexander Hermans, Firas Khader, Behrus Puladi, Bastian Leibe, Christiane Kuhl, Sven Nebelung, and Daniel Truhn. An ordinal regression framework for a deep learning based severity assessment for chest radiographs. arXiv preprint arXiv:2402.05685, 2024. 1

[42] Xiaoping Wu, Ni Wen, Jie Liang, Yu-Kun Lai, Dongyu She, Ming-Ming Cheng, and Jufeng Yang. Joint acne image grading and counting via label distribution learning. In Proceedings of the IEEE/CVF International Conference on Com puter Vision (ICCV), pages 10642–10651, 2019. 2, 6

[43] Yilei Wu, Zijian Dong, Chongyao Chen, Wangchunshu Zhou, and Juan Helen Zhou. SupReMix: Supervised contrastive learning for medical imaging regression with mixup. Medical Image Analysis, page 103909, 2025. 2

[44] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stephane Deny. Barlow twins: Self-supervised learning via´ redundancy reduction. In Proceedings of the International Conference on Machine Learning (ICML), 2021. 2

[45] Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, and Dina Katabi. Rank-n-contrast: Learning continuous representations for regression. Advances in Neural Information Processing Systems (NeurIPS), 36:17882–17903, 2023. 2

## A. Additional Experimental Results

This supplementary material provides the complete results omitted from the main paper due to space constraints. We report calibration results and the ordinal-margin ablation across ConvNeXt-Tiny, ResNet-50, and DenseNet-121, including Accuracy, QWK, and MAE.

## A.1. Calibration Results Across Backbones

Table S1 extends the calibration comparison in the main paper to all three backbones. Tie-seg, GMM, and Est-prior remain label-free, while Known-prior uses the true training class prior as an oracle reference.

## A.2. Margin Ablation Across Backbones

Table S2 extends the margin ablation in the main paper to ResNet-50 and DenseNet-121 and additionally reports MAE. All other components are kept unchanged between the multi-margin and single-margin variants.

## B. Additional Details of Dominance Estimation

We provide the details of the random-walk transition matrix used for global dominance estimation in Sec. 3.3 of the main paper.

Let $A _ { u v }$ denote the number of observed comparisons in which image u is judged more severe than image v. Tied comparisons are excluded. The maximum degree of the non-tied comparison graph is defined as

$$
d _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { u } \sum _ { v \neq u } \mathbb { I } [ A _ { u v } + A _ { v u } > 0 ] .\tag{17}
$$

Following Rank Centrality [28], we construct the transition matrix $P$ as

$$
P _ { u v } = \left\{ \begin{array} { l l } { \displaystyle \frac { A _ { v u } } { d _ { \mathrm { m a x } } \big ( A _ { u v } + A _ { v u } \big ) } , } & { u \neq v , A _ { u v } + A _ { v u } > 0 , } \\ { 1 - \displaystyle \sum _ { w \neq u } P _ { u w } , } & { u = v , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{18}
$$

The off-diagonal transitions move toward images judged more severe, while the diagonal entries provide self-loops to ensure that each row sums to one. Unobserved and tied pairs, therefore, contribute no off-diagonal transitions.

Starting from the uniform distribution $\pmb { \pi } ^ { ( 0 ) } = \mathbf { 1 } / N ,$ , we perform $L$ power iterations:

$$
\pmb { \pi } ^ { ( t + 1 ) } = P ^ { \top } \pmb { \pi } ^ { ( t ) } , \qquad t = 0 , \ldots , L - 1 .\tag{19}
$$

The probability vector is renormalized after each iteration for numerical stability. We use $\pi = \pi ^ { ( L ) }$ to derive the dominance scores. Thus, convergence to a unique stationary distribution is not required.

Table S1. Label-free calibration across backbones and pair budgets, with Accuracy, QWK, and MAE for every route. Tie-seg (prior-free), GMM, and Est-prior use no labels; Known-prior uses the true class prior. Per row, the best label-free accuracy is in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2">Budget</td><td colspan="3">Tie-seg</td><td colspan="3">GMM</td><td colspan="3">Est-prior</td><td colspan="3">Known-prior</td></tr><tr><td>Acc ↑</td><td>QWK↑</td><td>MAE↓</td><td>Acc ↑</td><td>QWK↑</td><td>MAE↓</td><td>Acc ↑</td><td>QWK↑</td><td>MAE↓</td><td>Acc ↑</td><td>QWK↑</td><td>MAE↓</td></tr><tr><td rowspan="8">ACNE (K=4)</td><td>ConvNeXt-T</td><td>30%</td><td>72.6±1.2 0.807 ±0.011</td><td>0.274±0.012</td><td>73.6 ±1.2</td><td>0.821 ±0.012</td><td>0.264±0.012</td><td>73.6 ±0.0</td><td>0.818±0.003</td><td></td><td>0.264±0.000</td><td>73.3 ±0.9</td><td>0.819±0.010</td><td>0.267 ±0.009</td></tr><tr><td></td><td>50%</td><td>74.8±0.3 0.821 ±0.001 0.807 ±0.021</td><td>0.252 ±0.003</td><td>75.0±1.0 75.1 ±0.8</td><td>0.824±0.007</td><td>0.252 ±0.009</td><td></td><td>75.7 ±0.6</td><td>0.818 ±0.003</td><td>0.248 ±0.003</td><td>75.5 ±0.9</td><td>0.826±0.008</td><td>0.248 ±0.009</td></tr><tr><td></td><td>70%</td><td>74.1 ±2.0</td><td>0.263 ±0.023</td><td></td><td>0.825 ±0.005</td><td></td><td>0.251 ±0.008</td><td>74.4±0.8</td><td>0.813±0.007</td><td>0.260±0.010</td><td>74.9±1.3</td><td>0.820±0.014</td><td>0.256±0.016</td></tr><tr><td></td><td>30%</td><td>72.7±1.8 0.785 ±0.020</td><td>0.280 ±0.020</td><td>74.0 ±2.0</td><td>0.810±0.018</td><td>0.266±0.020</td><td></td><td>73.1 ±0.9</td><td>0.793 ±0.007</td><td>0.276±0.006</td><td>73.7±1.1</td><td>0.798±0.012</td><td>0.271 ±0.012</td></tr><tr><td>ResNet-50</td><td>50%</td><td>73.9±1.7 0.804±0.012</td><td>0.266±0.017</td><td>74.3 ±1.1</td><td>0.812±0.007</td><td>0.264±0.011</td><td>74.0±1.6</td><td>0.799 ±0.015</td><td></td><td>0.266±0.014</td><td>75.0±0.9</td><td>0.812±0.011</td><td>0.257 ±0.010</td></tr><tr><td></td><td>70%</td><td>72.1 ±0.8 0.783 ±0.010</td><td>0.284 ±0.003</td><td>73.3±1.5</td><td>0.803 ±0.005</td><td>0.273 ±0.010</td><td>70.0±2.5</td><td>0.750 ±0.030</td><td></td><td>0.309 ±0.021</td><td>73.4±1.4</td><td>0.795 ±0.012</td><td>0.274±0.010</td></tr><tr><td></td><td>30%</td><td>73.6±1.9 0.805 ±0.014</td><td>0.269 ±0.017</td><td>74.1 ±2.6</td><td>0.813±0.016</td><td>0.264±0.024</td><td>72.9±2.0</td><td>0.789±0.021</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DenseNet-121</td><td>50%</td><td>73.4±1.2 0.801 ±0.006</td><td>0.271 ±0.010</td><td>73.1 ±1.4</td><td>0.809±0.013</td><td>0.274±0.016</td><td>74.0 ±1.2</td><td>0.796±0.017</td><td>0.270±0.014</td><td>0.279±0.017</td><td>73.7 ±2.1 72.7 ±0.8</td><td>0.802 ±0.013 0.801 ±0.012</td><td>0.271 ±0.017 0.280 ±0.012</td></tr><tr><td rowspan="8">IDRiD (K=5)</td><td></td><td>70%</td><td>74.1 ±1.4 0.799±0.017</td><td>0.266±0.017</td><td>73.7±1.6</td><td>0.808 ±0.019</td><td>0.269±0.018</td><td>72.9±3.0</td><td>0.780 ±0.025</td><td>0.283 ±0.028</td><td></td><td>73.6±1.9 0.799 ±0.016</td><td></td><td>0.273 ±0.019</td></tr><tr><td></td><td>30%</td><td>84.5 ±0.7 0.909±0.010</td><td>0.196±0.013</td><td>84.2 ±0.5</td><td>0.924±0.005</td><td>0.186±0.006</td><td>84.3 ±1.1</td><td>0.925 ±0.008</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ConvNeXt-T</td><td>50%</td><td>84.0 ±0.4 0.903 ±0.003</td><td>0.205 ±0.005</td><td>83.6±1.0</td><td>0.922±0.005</td><td>0.193±0.010</td><td>83.9±0.7</td><td>0.921 ±0.004</td><td></td><td>0.184±0.014 0.191 ±0.008</td><td>84.7 ±0.3 84.7 ±0.8</td><td>0.916±0.003</td><td>0.190±0.004</td></tr><tr><td></td><td>70%</td><td>84.0 ±0.4 0.910±0.008</td><td>0.198±0.008</td><td>83.8±0.7</td><td>0.923±0.006</td><td>0.191 ±0.010</td><td>83.1 ±0.7</td><td>0.920±0.007</td><td></td><td>0.198±0.011</td><td>84.8±0.5</td><td>0.919±0.005 0.918±0.002</td><td>0.187 ±0.008</td></tr><tr><td></td><td>30%</td><td>82.2 ±0.9 0.877 ±0.007</td><td>0.239±0.011</td><td>36.3 ±2.5</td><td>0.790±0.006</td><td>0.707 ±0.020</td><td>31.9±5.7</td><td>0.771 ±0.018</td><td></td><td></td><td></td><td></td><td>0.190 ±0.005</td></tr><tr><td>ResNet-50</td><td>50%</td><td>82.8 ±0.6 0.888±0.011</td><td>0.227±0.011</td><td>34.1 ±1.8</td><td>0.783±0.006</td><td>0.731 ±0.018</td><td>29.3 ±4.0</td><td>0.763±0.013</td><td></td><td>0.768 ±0.056</td><td>82.5 ±0.2</td><td>0.900 ±0.007</td><td>0.220 ±0.007</td></tr><tr><td></td><td>70%</td><td>82.5 ±0.6 0.886±0.003</td><td>0.230 ±0.006</td><td>32.5±1.9</td><td>0.779±0.010</td><td>0.745 ±0.023</td><td>24.2 ±4.5</td><td>0.745 ±0.013</td><td></td><td>0.801 ±0.041</td><td>83.1 ±0.7</td><td>0.900±0.006</td><td>0.218±0.010</td></tr><tr><td></td><td>30%</td><td>83.2 ±0.9 0.902 ±0.008</td><td>0.214±0.011</td><td>46.7±17.0</td><td>0.815±0.043</td><td>0.599 ±0.182</td><td>45.9±17.9</td><td></td><td></td><td>0.852±0.044</td><td>83.6±0.3</td><td>0.903 ±0.007</td><td>0.211 ±0.007</td></tr><tr><td rowspan="7"></td><td>DenseNet-121</td><td>50%</td><td>83.1 ±1.0 0.890 ±0.009</td><td>0.223 ±0.013</td><td>56.1 ±20.1</td><td>0.844±0.054</td><td>0.495 ±0.221</td><td>54.6±20.9</td><td>0.809 ±0.048</td><td></td><td>0.619±0.199</td><td>83.1 ±1.0</td><td>0.905 ±0.004</td><td>0.212±0.011</td></tr><tr><td></td><td>70%</td><td>83.1 ±0.9 0.891 ±0.003</td><td>0.221 ±0.007</td><td>39.2 ±2.1</td><td>0.799 ±0.007</td><td>0.683 ±0.019</td><td></td><td>0.834±0.054</td><td></td><td>0.521 ±0.232</td><td>83.6±1.0</td><td>0.904 ±0.006</td><td>0.210 ±0.011</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>37.2±1.5</td><td>0.784±0.011</td><td>0.723 ±0.024</td><td>83.8±0.9</td><td>0.909 ±0.005</td><td>0.205 ±0.010</td></tr><tr><td>30%</td><td>50%</td><td>77.5±0.4 0.857 ±0.003</td><td>0.228 ±0.003 0.223 ±0.006</td><td>77.6±0.3</td><td>0.860±0.001</td><td>0.228 ±0.002</td><td>76.2 ±0.6</td><td></td><td>0.851 ±0.003</td><td>0.241 ±0.006</td><td>77.6±0.5</td><td>0.860±0.003</td><td>0.228 ±0.004</td></tr><tr><td>ConvNeXt-T</td><td>70%</td><td>78.0 ±0.6 0.862±0.004 78.1 ±0.4 0.859±0.004</td><td>0.223 ±0.004</td><td>78.0 ±0.6</td><td>0.864±0.004</td><td>0.223 ±0.007</td><td>76.6±0.3</td><td>0.857 ±0.003</td><td></td><td>0.236±0.004</td><td>78.1 ±0.6</td><td>0.866 ±0.005</td><td>0.222±0.007</td></tr><tr><td></td><td></td><td></td><td></td><td>78.2±0.1</td><td>0.863 ±0.003</td><td>0.222 ±0.002</td><td>77.3 ±0.6</td><td>0.861 ±0.004</td><td></td><td>0.231 ±0.007</td><td>78.4±0.2</td><td>0.865 ±0.003</td><td>0.220±0.002</td></tr><tr><td>ResNet-50</td><td>30%</td><td>76.7 ±0.7 0.841 ±0.004</td><td>0.240±0.007</td><td>76.7 ±0.8</td><td>0.847 ±0.004</td><td>0.239±0.008</td><td>76.3 ±0.7</td><td>0.846±0.004</td><td>0.242±0.006</td><td></td><td>76.8±0.7</td><td>0.845 ±0.002</td><td>0.239 ±0.007</td></tr><tr><td rowspan="5">LIMUC (K=4)</td><td></td><td>50% 76.5±0.5</td><td>0.845±0.004</td><td>0.240 ±0.005</td><td>76.6 ±0.6</td><td>0.851 ±0.004</td><td>0.239±0.007</td><td>75.8±0.4</td><td>0.845±0.004</td><td>0.247 ±0.004</td><td></td><td>76.7 ±0.6 0.851 ±0.004</td><td></td><td>0.239 ±0.006</td></tr><tr><td></td><td>70%</td><td>76.7 ±0.7 0.843 ±0.008</td><td>0.239 ±0.008</td><td>76.8 ±0.6</td><td>0.848 ±0.005</td><td>0.238 ±0.006</td><td>76.1 ±0.8</td><td></td><td>0.847±0.005</td><td>0.245 ±0.007</td><td>77.0±0.7</td><td>0.850 ±0.007</td><td>0.236±0.007</td></tr><tr><td></td><td>30%</td><td>76.0±0.5 0.839±0.003</td><td>0.245 ±0.004</td><td>76.3 ±0.6</td><td>0.846 ±0.003</td><td>0.242 ±0.006</td><td>75.2±1.1</td><td>0.841 ±0.008</td><td></td><td>0.252 ±0.010</td><td>76.6±0.5</td><td>0.847 ±0.004</td><td></td></tr><tr><td>DenseNet-121</td><td>50%</td><td>76.4 ±0.5 0.844±0.003</td><td>0.241 ±0.004</td><td>76.3 ±0.8</td><td>0.849 ±0.004</td><td>0.241 ±0.007</td><td>75.8±0.9</td><td>0.849 ±0.006 0.848 ±0.003</td><td>0.247 ±0.009 0.248 ±0.006</td><td></td><td>76.4 ±0.7</td><td>0.850 ±0.004</td><td>0.239 ±0.004 0.241 ±0.007</td></tr><tr><td></td><td>70%</td><td>76.3 ±0.3 0.844±0.004</td><td>0.242 ±0.004</td><td>76.6 ±0.4</td><td>0.851 ±0.004</td><td>0.239 ±0.005</td><td>75.7 ±0.6</td></table>

Table S2. Ablation of the ordinal margin formulation. The proposed cumulative multi-margin objective is compared with a single learnable margin under identical pair budgets, backbones, dominance estimation, and calibration protocols.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2">Budget</td><td colspan="6">Tie-seg</td><td rowspan="2" colspan="2">Known-prior</td></tr><tr><td colspan="2">Accuracy ↑(%)</td><td colspan="2">QWK↑</td><td colspan="2">MAE↓</td></tr><tr><td></td><td></td><td></td><td>Multi</td><td>Single</td><td>Multi</td><td>Single</td><td>Multi</td><td>Single</td><td>Multi</td><td>Single</td></tr><tr><td rowspan="10">ACNE (K=4)</td><td rowspan="3">ConvNeXt-T</td><td>30%</td><td>72.6±1.2</td><td>69.4±2.6</td><td>0.807</td><td>0.768</td><td>0.274</td><td>0.309</td><td>73.3 ±0.9</td><td>70.3±1.7</td></tr><tr><td>50%</td><td>74.8 ±0.3</td><td>71.3 ±2.5</td><td>0.821</td><td>0.785</td><td>0.252</td><td>0.294</td><td>75.5 ±0.9</td><td>71.1 ±1.5</td></tr><tr><td>70%</td><td>74.1 ±2.0</td><td>71.6±1.2</td><td>0.807</td><td>0.786</td><td>0.263</td><td>0.289</td><td>74.9 ±1.3</td><td>71.1 ±1.7</td></tr><tr><td rowspan="3">ResNet-50</td><td>30%</td><td>72.7±1.8</td><td>68.7 ±2.3</td><td>0.785</td><td>0.742</td><td>0.280</td><td>0.317</td><td>73.7 ±1.1</td><td>70.4±1.6</td></tr><tr><td>50%</td><td>73.9±1.7</td><td>67.3 ±0.8</td><td>0.804</td><td>0.716</td><td>0.266</td><td>0.339</td><td>75.0 ±0.9</td><td>69.1 ±0.9</td></tr><tr><td>70%</td><td>72.3±1.7</td><td>67.1 ±1.6</td><td>0.783</td><td>0.744</td><td>0.286</td><td>0.331</td><td>73.6 ±2.1</td><td>69.0±1.9</td></tr><tr><td rowspan="3">DenseNet-121</td><td>30%</td><td>73.6±1.9</td><td>69.4 ±1.5</td><td>0.805</td><td>0.748</td><td>0.269</td><td>0.313</td><td>73.7 ±2.1</td><td>69.3 ±1.7</td></tr><tr><td>50%</td><td>73.4±1.2</td><td>69.3 ±2.2</td><td>0.801</td><td>0.761</td><td>0.271</td><td>0.311</td><td>72.7 ±0.8</td><td>71.0±2.7</td></tr><tr><td>70%</td><td>74.1 ±1.4</td><td>68.1 ±2.8</td><td>0.799</td><td>0.738</td><td>0.266</td><td>0.327</td><td>73.6 ±1.9</td><td>68.7 ±2.6</td></tr><tr><td rowspan="6">ConvNeXt-T</td><td rowspan="3"></td><td>30%</td><td>84.5 ±0.7</td><td>82.8 ±1.3</td><td>0.909</td><td>0.897</td><td>0.196</td><td>0.221</td><td>84.7 ±0.3</td><td>82.6±0.8</td></tr><tr><td>50%</td><td>84.0 ±0.4</td><td>83.7 ±1.0</td><td>0.903</td><td>0.902</td><td>0.205</td><td>0.213</td><td>84.7 ±0.8</td><td>82.5±1.5</td></tr><tr><td>70%</td><td>84.0±0.4</td><td>84.3 ±1.0</td><td>0.910</td><td>0.906</td><td>0.198</td><td>0.203</td><td>84.8 ±0.5</td><td>83.1 ±1.0</td></tr><tr><td rowspan="3">ResNet-50</td><td>30%</td><td>82.2±0.9</td><td>81.8 ±1.1</td><td>0.877</td><td>0.886</td><td>0.239</td><td>0.234</td><td>82.5 ±0.2</td><td>82.2±1.1</td></tr><tr><td>50%</td><td>82.8 ±0.6</td><td>81.5 ±1.1</td><td>0.888</td><td>0.884</td><td>0.227</td><td>0.239</td><td>83.1 ±0.7</td><td>82.6±0.6</td></tr><tr><td>70%</td><td>82.5 ±0.6</td><td>80.1 ±0.9</td><td>0.886</td><td>0.877</td><td>0.230</td><td>0.254</td><td>83.6 ±0.3</td><td>81.5 ±0.5</td></tr><tr><td rowspan="3"></td><td rowspan="3">DenseNet-121</td><td>30%</td><td>83.2±0.9</td><td>81.5 ±1.6</td><td>0.902</td><td>0.890</td><td>0.214</td><td>0.234</td><td>83.1 ±1.0</td><td>80.4±1.0</td></tr><tr><td>50%</td><td>83.1 ±1.0</td><td>83.1 ±2.4</td><td>0.890</td><td>0.897</td><td>0.223</td><td>0.221</td><td>83.6 ±1.0</td><td>81.6±1.0</td></tr><tr><td>70%</td><td>83.1 ±0.9</td><td>81.9±1.3</td><td>0.891</td><td>0.894</td><td>0.221</td><td>0.231</td><td>83.8 ±0.9</td><td>82.2±1.1</td></tr><tr><td rowspan="9">LIMUC (K=4)</td><td rowspan="3">ConvNeXt-T</td><td>30%</td><td>77.5 ±0.4</td><td>76.9 ±0.3</td><td>0.857</td><td>0.840</td><td>0.228</td><td>0.239</td><td>77.6 ±0.5</td><td>77.1 ±0.4</td></tr><tr><td>50%</td><td>78.0 ±0.6</td><td>75.7 ±0.1</td><td>0.862</td><td>0.826</td><td>0.223</td><td>0.251</td><td>78.1 ±0.6</td><td>76.0±0.0</td></tr><tr><td>70%</td><td>78.1 ±0.4</td><td>76.5 ±0.7</td><td>0.859</td><td>0.835</td><td>0.223</td><td>0.242</td><td>78.4 ±0.2</td><td>77.1 ±0.4</td></tr><tr><td rowspan="3">ResNet-50</td><td>30%</td><td>76.7 ±0.7</td><td>75.3 ±0.8</td><td>0.841</td><td>0.833</td><td>0.240</td><td>0.253</td><td>76.8 ±0.7</td><td>75.7±0.4</td></tr><tr><td>50%</td><td>76.5 ±0.5</td><td>76.2±0.5</td><td>0.845</td><td>0.838</td><td>0.240</td><td>0.245</td><td>76.7 ±0.6</td><td>76.2±0.4</td></tr><tr><td>70%</td><td>76.7±0.7</td><td>75.9±0.5</td><td>0.843</td><td>0.837</td><td>0.239</td><td>0.249</td><td>77.0 ±0.7</td><td>76.0±0.6</td></tr><tr><td rowspan="3">DenseNet-121</td><td>30%</td><td>76.0 ±0.5</td><td>75.2 ±0.7</td><td>0.839</td><td>0.825</td><td>0.245</td><td>0.257</td><td>76.6 ±0.5</td><td>75.4±0.8</td></tr><tr><td>50%</td><td>76.4±0.5</td><td>75.6±0.8</td><td>0.844</td><td>0.831</td><td>0.241</td><td>0.250</td><td>76.4 ±0.7</td><td>75.8±1.0</td></tr><tr><td>70%</td><td>76.3 ±0.3</td><td>75.1 ±0.7</td><td>0.844</td><td>0.830</td><td>0.242</td><td>0.256</td><td>76.5 ±0.3</td><td>75.6±0.6</td></tr></table>