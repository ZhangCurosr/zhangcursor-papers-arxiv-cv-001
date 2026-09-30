# Context without Commitment: Robust Dense Correspondence under Non-Rigid Deformation

Mannheim Institute for Intelligent Systems in Medicine (MIISM) Medical Faculty Mannheim, Heidelberg University, Mannheim, Germany

Sara Homscheid<sup>∗</sup>

Mannheim Institute for Intelligent Systems in Medicine (MIISM) Medical Faculty Mannheim, Heidelberg University, Mannheim, Germany

## Abstract

Non-rigid point-cloud registration aims to find the corresponding target point for every point on a deforming source surface. Point-level matching keeps the complete target cloud available, but correspondence can become ambiguous when different regions have similar local geometry. Regional or coarse-to-fine methods provide larger spatial context, but an incorrect regional match can exclude the correct point correspondence before the final dense matching stage. We propose CoCo-Reg, which uses regional patches to enrich dense point features without allowing patch predictions to restrict the final point-level search. CoCo-Reg constructs farthestpoint-sampled patches, exchanges geometric information within and between the source and target, supervises patch similarity using identity-corrected point overlap, and projects the resulting regional information back to the dense point features. The final registration stage still scores the complete target cloud before performing its global point-level candidate selection. On 726 held-out ModelNet10 objects across nine deformation levels, two established learning-based non-rigid registration baselines obtain mean correspondence errors of 0.1993 and 0.1921, whereas CoCo-Reg obtains 0.0547. Relative to the point-level baseline on which CoCo-Reg is built, this corresponds to a 72.6% reduction. CoCo-Reg achieves lower correspondence error on 92.3% of paired test objects and reduces the mean fraction of points with error above 0.1 from 47.3% to 17.3%. Chamfer distance and HD95 decrease in the same direction, and CoCo-Reg remains lower across all tested deformation levels. These results support the use of regional context for dense non-rigid correspondence without imposing a hard patch-level restriction on the final search. Because the evaluation uses one checkpoint per method, the reported gains characterize the complete evaluated systems rather than the isolated causal contribution of an individual component. Code will be made publicly available.

## 1 Introduction and Related Work

Non-rigid point-cloud registration estimates how a source surface moves to a target surface by predicting a spatially varying displacement for its points. Unlike rigid registration, one global transformation is not sufficient: different parts of a deformable object may move in different directions and by different amounts. Reliable correspondence is therefore difficult when deformation is large, local neighborhoods change, or several parts of the shape look geometrically similar. This problem appears in deformable-object analysis, 3D scene understanding, and soft-tissue modeling, where tissue shift can invalidate a fixed preoperative geometry [8, 9, 11–13].

Point-level learned matching. Many learning-based registration methods keep correspondence reasoning at the point level. DCP, RPM-Net, PREDATOR, PointDSC, and RegTR learn point descriptors and correspondence rules for rigid or partially overlapping point clouds [1, 4, 19, 22, 23]. For deformable geometry, FLOT learns dense motion through transport, Lepard introduces cross-cloud positional information, and Neural Deformation Pyramid models non-rigid motion hierarchically [6, 7, 16]. Recent non-rigid methods such as DefTransNet keep a global point-level candidate search, while DINE adds a prior over the complete deformation field [12, 14]. Point-level matching is flexible because every target point can remain a candidate, but the descriptor of one point or a small neighborhood may be insufficient to distinguish repeated or symmetric regions under strong deformation.

Coarse-to-fine and regional matching. A second family first reasons over larger regions and then performs finer matching. PointNet++ and DGCNN established hierarchical and neighborhood-based point representations, while CoFiNet and GeoTransformer use superpoints or coarse correspondences to guide registration [17, 18, 20, 24]. Robust-DefReg similarly uses a coarse-to-fine strategy for non-rigid registration [13]. Regional reasoning can reduce local ambiguity, but it introduces another failure mode: if a coarse source region is matched to the wrong target region and that decision is used to restrict the fine search, the correct point correspondence may no longer be reachable. This is particularly important for non-rigid deformation because source and target patches are constructed independently and one physical region can overlap several patches after deformation.

Patch representations. Patch-based representation learning provides a useful middle ground between individual points and whole objects. Point-BERT and Point-MAE use local patches for masked representation learning, and PointGPT uses ordered point patches for autoregressive pretraining [2, 15, 25]. PointGPT motivated our use of structured patch tokens, but CoCo-Reg does not use its pretrained model, tokenizer, causal ordering, or autoregressive objective. Instead, patches are constructed directly from the registration clouds and are used only to provide regional information for dense correspondence.

The remaining challenge is therefore not simply to choose between point-level and patch-level matching. Point-level search preserves flexibility but may lack enough context; hard patch-based matching provides context but can make an early mistake irreversible. CoCo-Reg addresses this trade-off by using patches to inform the dense point features without using patch predictions to define the final search space. Each cloud is partitioned into farthest-point-sampled patches, patch tokens exchange geometry-aware information within each cloud and across the source and target, and patch similarity is trained from identity-corrected point overlap. The contextual patch token is then projected back to the points of that patch as a residual feature update. The original dense registration stage still ranks candidates over the complete target cloud. We refer to this simple design as context without commitment: patches provide guidance, but they do not decide which target region a point is allowed to search. Our main contribution is therefore not a new dense registration backbone, but a way to add regional information without turning the regional prediction into hard candidate pruning.

Our contributions are:

• We introduce a patch-context extension for dense non-rigid registration that adds larger-scale regional information to point features while preserving the original global point-level correspondence search.

• We formulate identity-corrected, multi-positive patch supervision for independently constructed source and target patches, allowing one source patch to overlap more than one valid target patch after deformation.

• We evaluate the complete CoCo-Reg configuration on 726 paired held-out objects across deformation levels 0.1–0.9 using correspondence-specific, set-level, tail-error, failure-rate, and paired statistical analyses, and explicitly separate the observed system-level gains from stronger component-level causal claims.

## 2 Proposed Method

Problem definition. Let the source cloud be $S = \{ s _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { R } ^ { 3 }$ and let the identity-indexed, unshuffled target be $T ^ { \star } = \{ t _ { i } \} _ { i = 1 } ^ { N }$ . During training and evaluation the target tensor is shuffled, so the network receives $T = \{ t _ { \pi ( r ) } \} _ { r = 1 } ^ { N }$ for a permutation π. The correspondence identity is not given to the network; it is retained only for supervision and evaluation. The ground-truth displacement of source point i is

$$
d _ { i } ^ { \star } = t _ { i } - s _ { i } , \qquad { \hat { D } } = \{ { \hat { d } } _ { i } \} _ { i = 1 } ^ { N } ,\tag{1}
$$

and the primary dense objective is the mean absolute component-wise displacement error,

$$
\mathcal { L } _ { \mathrm { p o i n t } } = \frac { 1 } { 3 N } \sum _ { i = 1 } ^ { N } \Vert \hat { d } _ { i } - d _ { i } ^ { \star } \Vert _ { 1 } .\tag{2}
$$

The final implementation uses $N = 1 0 2 4$ points per cloud.

Dense backbone features. CoCo-Reg retains the DefTransNet feature and displacement backbone [12]. Source and target are mapped to dense 64-dimensional descriptors

$$
( F ^ { S } , F ^ { T } ) = \Phi _ { \mathrm { D e f } } ( S , T ) , \qquad F ^ { S } , F ^ { T } \in \mathbb { R } ^ { N \times 6 4 } .\tag{3}
$$

These same descriptors feed both the patch-context branch and the final dense registration stage.

Patch construction and token initialization. For either cloud $P ,$ farthest-point sampling selects $M = 3 2$ center indices $I = \mathrm { F P S } ( P , M )$ with coordinates $C = P [ I$ ]. Every dense point is assigned to its closest center,

$$
g ( n ) = \arg \operatorname* { m i n } _ { m \in \{ 1 , \ldots , M \} } \| p _ { n } - c _ { m } \| _ { 2 } ^ { 2 } ,\tag{4}
$$

forming independent source and target ownership sets. At most $K _ { p } = 2 4$ closest owned points are retained for overlap supervision. Importantly, the token is initialized from the FPS-center feature, not from a pooled patch descriptor. With $H = \mathbf { \dot { F } } [ I ] \in \mathbb { R } ^ { M \times 6 4 }$

$$
U = \mathrm { L N } ( H W _ { f } + b _ { f } ) , \qquad z _ { m } ^ { 0 } = \mathrm { N o r m } _ { 2 } ( u _ { m } + 0 . 1 \rho ( c _ { m } ) ) ,\tag{5}
$$

where $W _ { f }$ projects to 128 dimensions and $\rho$ is a two-layer MLP on the center coordinate.

Geometry-aware patch interaction. Within each cloud, four-head self-attention is biased by the normalized Euclidean distance between patch centers. For head r,

$$
e _ { i j } ^ { ( r ) } = \frac { ( q _ { i } ^ { ( r ) } ) ^ { \top } k _ { j } ^ { ( r ) } } { \sqrt { d _ { h } } } + \eta _ { r } \left( \frac { \| c _ { i } - c _ { j } \| _ { 2 } } { \mu } \right) ,\tag{6}
$$

where $\mu$ is the median pairwise center distance and $\eta _ { r }$ is a learned MLP. Source and target tokens then exchange information bidirectionally,

$$
\begin{array} { r } { Z ^ { S  T } = Z ^ { S } + \mathrm { M H A } ( \mathrm { L N } ( Z ^ { S } ) , Z ^ { T } , Z ^ { T } ) , } \end{array}\tag{7}
$$

$$
Z ^ { T  S } = Z ^ { T } + \mathrm { M H A } \big ( \mathrm { L N } ( Z ^ { T } ) , Z ^ { S } , Z ^ { S } \big ) ,\tag{8}
$$

followed by a residual feed-forward layer with expansion factor two. Patch similarity is a scaled cosine score

$$
a _ { i j } = \gamma \frac { ( z _ { i } ^ { S } ) ^ { \top } z _ { j } ^ { T } } { \| z _ { i } ^ { S } \| _ { 2 } \| z _ { j } ^ { T } \| _ { 2 } } ,\tag{9}
$$

with learned scale $\gamma ,$ , initialized to 8 and constrained to the interval [1, 30] during the forward pass.

Identity-corrected patch-overlap supervision. Because target tensor positions are shuffled, overlap supervision is computed in point-identity space rather than tensor-position space. Let $R _ { i } ^ { S }$ and $R _ { j } ^ { \bar { T } }$ denote the retained source and target tensor indices owned by patches i and $j ,$ , respectively. Since target position r contains point $t _ { \pi ( r ) }$ , define the corresponding identity sets as $I _ { i } ^ { S } = R _ { i } ^ { S }$ and $I _ { j } ^ { T } = \{ \pi ( r ) : r \in R _ { j } ^ { T } \}$ . The symmetric overlap is

$$
o _ { i j } = \frac { 1 } { 2 } \left( \frac { | I _ { i } ^ { S } \cap I _ { j } ^ { T } | } { | I _ { i } ^ { S } | } + \frac { | I _ { i } ^ { S } \cap I _ { j } ^ { T } | } { | I _ { j } ^ { T } | } \right) .\tag{10}
$$

![](images/60128c7b1d6f9e6e168065f478395f8e1b379ff75372db589d0dd25d2a7f89d4.jpg)  
Figure 1: Overview of CoCo-Reg. Top: The proposed method extracts dense point features from the source and target point clouds, constructs geometric patches, and exchanges information between source and target patches through geometry-aware interaction. The resulting patch information is projected back to the dense point features through a residual refinement. Final correspondence estimation is then performed at the point level using the global sLBP search, which scores the complete target cloud before retaining the Top-128 candidates. Bottom: Comparison with hard patchbased matching. In a hard coarse-to-fine strategy, an incorrect patch match can remove the correct point correspondence from the subsequent search. CoCo-Reg instead uses patch information to enrich the point representation without applying a hard patch-level restriction to the final correspondence search.

Let $V _ { T } = \{ 1 , \dots , M \}$ and $V _ { S } ^ { + } = \{ i : \operatorname* { m a x } _ { j } o _ { i j } > 0 \}$ . For each $i \in V _ { S } ^ { + }$ , pairs with $o _ { i j } \geq 0 . 1 0$ form the positive set $Q _ { i } ;$ if this set is empty despite non-zero overlap, the maximum-overlap target is used as a fallback. The source-to-target multi-positive loss is

$$
\mathcal { L } _ { \mathrm { M P } } ^ { S  T } = - \frac { 1 } { \vert V _ { S } ^ { + } \vert } \sum _ { i \in V _ { S } ^ { + } } \log \frac { \sum _ { j \in Q _ { i } } \exp ( a _ { i j } ) } { \sum _ { j \in V _ { T } } \exp ( a _ { i j } ) } ,\tag{11}
$$

and the implementation also applies the analogous target-to-source term. Writing $V _ { T } ^ { + } ~ = ~ \{ j$ max<sub>i</sub> $o _ { i j } > 0 \}$ , the bidirectional aggregate is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M P } } = \frac { 1 } { 2 } \mathopen { } \mathclose \bgroup ( \mathcal { L } _ { \mathrm { M P } } ^ { S  T } + \mathcal { L } _ { \mathrm { M P } } ^ { T  S } \aftergroup \egroup ) . } \end{array}\tag{12}
$$

A complementary best-overlap cross-entropy uses y<sub>i</sub> = arg max<sub>j</sub> o<sub>ij</sub> and $x _ { j } = \arg \operatorname* { m a x } _ { i } o _ { i j }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { b e s t } } = - \frac { 1 } { 2 } \Bigg [ \frac { 1 } { | V _ { S } ^ { + } | } \sum _ { i \in V _ { S } ^ { + } } \log \frac { \exp ( a _ { i y _ { i } } ) } { \sum _ { j \in V _ { T } } \exp ( a _ { i j } ) } } \\ { + \frac { 1 } { | V _ { T } ^ { + } | } \sum _ { j \in V _ { T } ^ { + } } \log \frac { \exp ( a _ { x _ { j } j } ) } { \sum _ { i = 1 } ^ { M } \exp ( a _ { i j } ) } \Bigg ] . } \end{array}\tag{13}
$$

Both patch terms use unit weight in the reported configuration. Thus the patch loss is

$$
\mathcal { L } _ { \mathrm { p a t c h } } = \mathcal { L } _ { \mathrm { M P } } + \mathcal { L } _ { \mathrm { b e s t } } .\tag{14}
$$

Context back to points without restricting the final search. As summarized in Fig. 1, each source and target point gathers the final token of its owned patch, $q _ { n } ^ { S } = z _ { g ^ { S } ( n ) } ^ { S }$ and $q _ { n } ^ { T } = z _ { g ^ { T } ( n ) } ^ { \breve { T } }$ , and receives a residual refinement

$$
\widetilde { f } _ { n } ^ { S } = f _ { n } ^ { S } + 0 . 1 0 \psi ( q _ { n } ^ { S } ) , \qquad \widetilde { f } _ { n } ^ { T } = f _ { n } ^ { T } + 0 . 1 0 \psi ( q _ { n } ^ { T } ) ,\tag{15}
$$

where $\psi : \mathbb { R } ^ { 1 2 8 } \to \mathbb { R } ^ { 6 4 }$ . The final displacement remains

$$
\hat { D } = \mathrm { s L B P } ( S , T , \widetilde { F } ^ { S } , \widetilde { F } ^ { T } ) , \qquad \mathscr { L } _ { \mathrm { t o t a l } } = \mathscr { L } _ { \mathrm { p o i n t } } + 0 . 1 0 \mathscr { L } _ { \mathrm { p a t c h } } .\tag{16}
$$

For each source point, sLBP scores all target points and only then retains its global Top-128 feature candidates. Thus patch context can change the dense feature ranking, but an incorrect patch relation cannot remove a target point simply because that target lies outside a selected patch. This operational definition is the paper’s context without commitment principle: the phrase refers specifically to avoiding an early hard patch-level restriction, while the downstream sLBP module still performs global point-level candidate selection.

## 3 Experimental Results

Dataset and implementation. We use ModelNet10 [21], with 1024 normalized surface points per object. Following the controlled synthetic-deformation protocols used in previous non-rigid registration studies [10, 12, 13], each source cloud is smoothly deformed at a controlled level $\ell \in \{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ . In addition to the non-rigid deformation, the evaluated pairs contain relative rigid pose variation; the applied source–target rotation angle is retained and used only for the descriptive rotation-stratified analysis in Table 5. Exact point identity is preserved by the generator, but the target order is shuffled before it is given to the network. The standard 3,991-mesh training split is used for training; the 908 standard test meshes are split deterministically and by deformation level into 182 validation objects and 726 final test objects.

The point and patch objectives are optimized jointly. Patch-specific settings are those defined in Sec. 2: 32 patches, at most 24 retained members per patch for overlap supervision, 128-dimensional patch tokens, one four-head interaction block, overlap threshold 0.10, and residual strength 0.10. DefTransNet is the closest architectural reference because CoCo-Reg keeps its dense feature and sLBP backbone; Robust-DefReg provides a second recent non-rigid baseline. All methods are evaluated on exactly the same source–target pairs and target permutations. One final checkpoint per method is used, so reported standard deviations describe variation across test objects rather than across independent training runs.

Evaluation metrics. Because the synthetic deformation preserves point identity, the primary metric is mean correspondence error (MCE). Let $\hat { S } = \{ s _ { i } + \hat { d } _ { i } \bar  \} _ { i = } ^ { N }$ be the registered source and, for source point i, let $e _ { i } = \lVert s _ { i } + \hat { d } _ { i } - t _ { i } \rVert _ { 2 }$ . The per-object MCE is

$$
E _ { \mathrm { M C E } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e _ { i } .\tag{17}
$$

This measures distance to the known corresponding target point, rather than to the nearest target point. We also report the symmetric unsquared Chamfer distance [3],

$$
E _ { \mathrm { C D } } = \frac { 1 } { 2 } \left( \frac { 1 } { N } \sum _ { \hat { s } \in \hat { S } } \operatorname* { m i n } _ { t \in T } \Vert \hat { s } - t \Vert _ { 2 } + \frac { 1 } { N } \sum _ { t \in T } \operatorname* { m i n } _ { \hat { s } \in \hat { S } } \Vert t - \hat { s } \Vert _ { 2 } \right) ,\tag{18}
$$

and the bidirectional 95th-percentile Hausdorff distance (HD95) [5]. With $d ( A , B ) = \{ \operatorname* { m i n } _ { b \in B } \| a -$ $b \| _ { 2 } : a \in A \}$ ,

$$
E _ { \mathrm { H D 9 5 } } = \operatorname * { m a x } \Big ( Q _ { 0 . 9 5 } [ d ( \hat { S } , T ) ] , ~ Q _ { 0 . 9 5 } [ d ( T , \hat { S } ) ] \Big ) .\tag{19}
$$

MCE evaluates the known correspondence, whereas Chamfer and HD95 evaluate the geometry of the registered and target point sets without using identity. We additionally report the median and 95th percentile of per-object MCE. As a complementary severe-error statistic, object failure is defined as $E _ { \mathrm { M C E } } > 0 . 1$ and point failure as the within-object fraction with $e _ { i } > 0 . 1$ . The threshold is used only for this diagnostic analysis and does not affect training or the continuous metrics.

Statistical analysis. All comparisons with the closest baseline are paired on the same 726 test objects. For the overall scalar metrics we report the mean paired difference, a 20,000-resample paired bootstrap 95% confidence interval, a two-sided Wilcoxon signed-rank test, and the paired rank-biserial effect size $r _ { r b }$ . Win-rate confidence intervals are exact binomial intervals. Object-level failure is analyzed with a 20,000-resample paired bootstrap interval for the failure-rate difference and an exact McNemar test. Deformation- and category-stratified mean-difference intervals use 10,000 paired bootstrap resamples; the nine deformation-level Wilcoxon tests are Holm-corrected. These analyses quantify consistency across the fixed paired test set; they do not estimate training-to-training uncertainty because only one checkpoint per method is available.

Table 1: Overall results on 726 held-out objects. Mean-valued quantities are reported as mean ± SD across test objects. Median and P95 are quantiles of the 726 per-object MCE values. “Objects with $\mathbf { M C E } > 0 . 1 ^ { \ ' }$ reports the number and percentage of test objects whose mean correspondence error exceeds 0.1. “Points with error $> 0 . 1 \ '$ reports the mean within-object percentage of points whose identity-specific correspondence error exceeds 0.1. Lower is better for all metrics.
<table><tr><td>Method</td><td>MCE</td><td>Median</td><td>P95</td><td>Objects with MCE &gt; 0.1</td><td>Points with error &gt; 0.1 (%)</td><td>Chamfer</td><td>HD95</td></tr><tr><td>Robust-DefReg</td><td>0.1921 ± 0.2001</td><td>0.1188</td><td>0.6287</td><td>402/726 (55.4%)</td><td>48.4 ± 35.6%</td><td>0.0178 ± 0.0100</td><td>0.1066 ± 0.0608</td></tr><tr><td>DefTransNet</td><td>0.1993 ± 0.2165</td><td>0.1180</td><td>0.6865</td><td>389/726 (53.6%)</td><td>47.3 ± 35.9%</td><td>0.0170 ± 0.0094</td><td>0.1015 ± 0.0575</td></tr><tr><td>CoCo-Reg</td><td>0.0547 ± 0.0693</td><td>0.0379</td><td>0.1529</td><td>90/726 (12.4%)</td><td>17.3 ± 21.1%</td><td>0.0108 ± 0.0061</td><td> $\mathbf { 0 . 0 6 8 2 \pm 0 . 0 3 8 1 }$ </td></tr></table>

Table 2: Paired CoCo-Reg vs. DefTransNet statistics. ∆ is CoCo-Reg−DefTransNet; negative values favor CoCo-Reg. Mean-difference CIs are 20,000-resample paired bootstrap intervals; win-rate CIs are exact binomial intervals. $r _ { r b }$ is oriented so positive values favor CoCo-Reg.
<table><tr><td>Metric</td><td>Mean ∆ [95% CI]</td><td></td><td>rrb</td><td>CoCo-Reg win rate [95% CI]</td><td>Wilcoxon p</td></tr><tr><td>MCE</td><td></td><td>-0.1447 [−0.1585, -0.1314]</td><td>0.960</td><td>92.3% [90.1, 94.1]</td><td> $2 . 8 9 \times 1 0 ^ { - 1 1 1 }$ </td></tr><tr><td>Chamfer</td><td></td><td>-0.00623 [−0.00657, -0.00588]</td><td>0.964</td><td>93.1% [91.0, 94.8]</td><td> $3 . 1 7 \times 1 0 ^ { - 1 1 2 }$ </td></tr><tr><td>HD95</td><td></td><td>-0.03328 [-0.03569, -0.03096]</td><td>0.934</td><td>89.5% [87.1, 91.7]</td><td> $1 . 8 3 \times 1 0 ^ { - 1 0 5 }$ </td></tr><tr><td>Point failure rate</td><td></td><td>-0.2994 [−0.3200, −0.2789]</td><td>0.973</td><td>88.8% [86.3, 91.0]</td><td> $1 . 1 0 \times 1 0 ^ { - 1 0 6 }$ </td></tr></table>

Overall correspondence accuracy. Table 1 gives the complete primary and complementary metrics. CoCo-Reg has the lowest MCE, lower distributional tail, fewer object- and point-level failures, and lower Chamfer and HD95 than both evaluated reference models. Relative to DefTransNet, mean MCE is 72.6% lower and CoCo-Reg is better on 670 of 726 paired objects.

Paired statistical evidence. Table 2 uses the paired design directly. For every continuous metric, the bootstrap interval lies entirely below zero and the paired rank-biserial effect is large and positive when oriented in favor of CoCo-Reg. These tests quantify consistency and effect magnitude on the common test set; they do not remove the checkpoint-training limitation discussed later.

Robustness across deformation levels. Table 3 shows the paired result across the complete tested deformation range. CoCo-Reg has lower MCE in every deformation stratum. Representative CoCo-Reg registrations at deformation levels 0.2, 0.5, and 0.8 are shown in Fig. 2, providing a qualitative view of the predicted correspondence under increasing deformation. The paired bootstrap CI for CoCo-Reg−DefTransNet remains below zero at every level; all nine paired Wilcoxon tests remain significant after Holm correction, with the largest adjusted value $3 . { \dot { 0 } } 5 \times 1 0 ^ { - 9 }$ . The intermediate means are not interpreted as a pure deformation dose-response because the fixed evaluation also contains pose variation.

Set-level geometry across deformation. The correspondence result is not specific to MCE. Table 4 reports Chamfer and HD95 at the same nine deformation levels. CoCo-Reg is lower than both reference checkpoints for both set-level metrics in every stratum. Thus, the deformation-stratified advantage is visible both when ground-truth identity is enforced by MCE and when registration quality is evaluated only through nearest-neighbor geometry.

Severe correspondencefailures. The reduction in failure rate is also strongly paired. DefTransNet fails on 389/726 objects and CoCo-Reg on 90/726, an absolute reduction of 41.18 percentage points with paired bootstrap 95% CI [37.47, 44.90]. Among the 305 discordant pairs, 302 change from DefTransNet failure to CoCo-Reg success and only 3 change in the opposite direction; exact McNemar $p = 1 . 4 5 \times 1 0 ^ { - 8 5 }$ . The within-object point-failure result in Tables 1–2 shows the same pattern at finer resolution.

Additional rotation-stratified robustness. Rotation is not a primary claim of this study, but the same 726 evaluated outputs can be grouped descriptively without new inference. Table 5 shows that CoCo-Reg remains lower in all three observed rotation groups. We do not interpret this as a controlled claim of rotation invariance or extrapolation because the available checkpoints were not produced as a matched rotation-specific study.

![](images/36536079b73c59fae985b5e411f3ec18cff6981d8f31f6f3e0874f12f2b34113.jpg)  
Figure 2: Qualitative registration under increasing deformation. Representative source, target, and CoCo-Reg registrations are shown at deformation levels 0.2, 0.5, and 0.8. Colors indicate the identityspecific correspondence error associated with each source-point identity and are propagated across the corresponding source, target, and registered points for spatial comparison.

Table 3: Deformation-stratified mean correspondence error (MCE). Means are mean $\pm \thinspace \mathrm { S D }$ across objects. ∆ is the paired CoCo-Reg−DefTransNet mean difference with bootstrap 95% CI; negative values favor CoCo-Reg.
<table><tr><td>Level</td><td>N</td><td>Robust-DefReg</td><td>DefTransNet</td><td>CoCo-Reg</td><td>∆ [95% CI]</td><td></td><td>Win</td><td>Holm p</td></tr><tr><td>0.1</td><td>63</td><td> $0 . 1 1 1 1 \pm 0 . 1 8 4 1$ </td><td> $0 . 1 2 0 5 \pm 0 . 2 0 5 5$ </td><td> $\mathbf { 0 . 0 2 9 3 \pm 0 . 0 9 3 4 }$ </td><td></td><td> $\cdot 0 . 0 9 1 3 [ - 0 . 1 3 8 1 , - 0 . 0 5 0 4 ]$ </td><td>84.1%</td><td> $3 . 0 5 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>0.2</td><td>80</td><td> $0 . 1 4 3 6 \pm 0 . 2 1 0 4$ </td><td> $0 . 1 4 8 5 \pm 0 . 2 2 1 2$ </td><td> $\mathbf { 0 . 0 3 8 2 \pm 0 . 0 9 6 7 }$ </td><td> $- 0 . 1 1 0 3 [ - 0 . 1 5 3 0 , - 0 . 0 7 1 3 ]$ </td><td></td><td>85.0%</td><td> $5 . 4 3 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>0.3</td><td>74</td><td> $0 . 1 3 4 6 \pm 0 . 2 1 2 8$ </td><td> $0 . 1 4 7 7 \pm 0 . 2 3 6 6$ </td><td> $\mathbf { 0 . 0 3 3 6 \pm 0 . 0 8 8 7 }$ </td><td> $- 0 . 1 1 4 2 [ - 0 . 1 6 4 0 , - 0 . 0 7 0 4 ]$ </td><td></td><td>90.5%</td><td> $5 . 4 3 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>0.4</td><td>92</td><td> $0 . 1 1 1 5 \pm 0 . 1 2 2 1$ </td><td> $0 . 1 1 6 7 \pm 0 . 1 4 4 0$ </td><td> $\mathbf { 0 . 0 3 0 9 \pm 0 . 0 2 8 8 }$ </td><td> $- 0 . 0 8 5 8 [ - 0 . 1 1 5 6 , - 0 . 0 6 0 0 ]$ </td><td></td><td>94.6%</td><td> $2 . 5 7 \times 1 0 ^ { - 1 4 }$ </td></tr><tr><td>0.5</td><td>88</td><td> $0 . 1 6 7 6 \pm 0 . 1 6 2 5$ </td><td> $0 . 1 7 6 8 \pm 0 . 1 7 9 8$ </td><td> $\mathbf { 0 . 0 4 3 8 \pm 0 . 0 3 1 2 }$ </td><td> $- 0 . 1 3 3 0 [ - 0 . 1 6 8 8 , - 0 . 1 0 0 4 ]$ </td><td></td><td>92.0%</td><td> $1 . 1 3 \times 1 0 ^ { - 1 4 }$ </td></tr><tr><td>0.6</td><td>80</td><td>0.2143 ± 0.1870</td><td> $0 . 2 2 4 5 \pm 0 . 2 0 9 7$ </td><td> $\mathbf { 0 . 0 5 3 7 \pm 0 . 0 3 4 1 }$ </td><td> $- 0 . 1 7 0 8 [ - 0 . 2 1 5 2 , - 0 . 1 2 9 2 ]$ </td><td></td><td>92.5%</td><td> $8 . 9 8 \times 1 0 ^ { - 1 4 }$ </td></tr><tr><td>0.7</td><td>81</td><td>0.2591 ± 0.2029</td><td> $0 . 2 6 1 0 \pm 0 . 2 2 2 4$ </td><td> $\mathbf { 0 . 0 7 2 0 \pm 0 . 0 5 0 3 }$ </td><td> $- 0 . 1 8 9 0 [ - 0 . 2 3 4 6 , - 0 . 1 4 5 5 ]$ </td><td></td><td>96.3%</td><td> $4 . 3 4 \times 1 0 ^ { - 1 4 }$ </td></tr><tr><td>0.8</td><td>82</td><td>0.2392 ± 0.1833</td><td> $0 . 2 3 6 7 \pm 0 . 1 8 3 3$ </td><td> $\mathbf { 0 . 0 7 9 8 \pm 0 . 0 5 4 2 }$ </td><td> $- 0 . 1 5 6 9 [ - 0 . 1 9 2 3 , - 0 . 1 2 4 6 ]$ </td><td></td><td>92.7%</td><td> $4 . 7 5 \times 1 0 ^ { - 1 4 }$ </td></tr><tr><td>0.9</td><td>86</td><td> $0 . 3 2 8 5 \pm 0 . 2 1 7 0$ </td><td> $0 . 3 4 3 2 \pm 0 . 2 3 8 5$ </td><td> $\mathbf { 0 . 1 0 3 9 \pm 0 . 0 7 6 0 }$ </td><td> $- 0 . 2 3 9 3 [ - 0 . 2 8 4 5 , - 0 . 1 9 6 8 ]$ </td><td></td><td>100.0%</td><td> $7 . 1 9 \times 1 0 ^ { - 1 5 }$ </td></tr></table>

Category consistency. The aggregate advantage is not dominated by one ModelNet10 class. Table 6 shows a negative CoCo-Reg−DefTransNet paired difference in all ten categories; every category bootstrap interval remains below zero and category-wise paired win rates range from 85.1% to 96.5%.

## 4 Discussion and Outlook

The results support the central design of CoCo-Reg: dense non-rigid correspondence benefits from regional geometric information, but this information does not need to be converted into a hard restriction on the final point-level search. By projecting patch-level information back to the dense descriptors, CoCo-Reg preserves the flexibility of global point matching while providing each point with a larger spatial context. Across the fixed paired evaluation, this leads to substantially lower correspondence error than the evaluated reference methods, and the same trend is observed across all tested deformation levels. The improvements in Chamfer distance and HD95 further show that the gain is not limited to identity-specific correspondence error, but is also reflected in the geometry of the registered point set.

Table 4: Deformation-stratified set-level geometry. Values are mean $\pm \mathrm { \bf S D }$ across objects. CD is symmetric unsquared Chamfer distance and HD95 is the 95th-percentile bidirectional Hausdorff distance. Lower is better.
<table><tr><td rowspan="2">Level</td><td colspan="3">Chamfer distance (CD)</td><td colspan="3">HD95</td></tr><tr><td>Robust</td><td>DefTransNet</td><td>CoCo-Reg</td><td>Robust</td><td>DefTransNet</td><td>CoCo-Reg</td></tr><tr><td>0.1</td><td> $0 . 0 0 9 7 \pm 0 . 0 0 8 7$ </td><td> $0 . 0 0 9 3 \pm 0 . 0 0 8 3$ </td><td> $\mathbf { 0 . 0 0 5 6 \pm 0 . 0 0 4 6 }$ </td><td> $0 . 0 5 5 8 \pm 0 . 0 4 8 5$ </td><td> $0 . 0 5 3 4 \pm 0 . 0 4 5 7$ </td><td> $\mathbf { 0 . 0 3 5 8 \pm 0 . 0 2 9 3 }$ </td></tr><tr><td>0.2</td><td> $0 . 0 1 1 4 \pm 0 . 0 0 8 5$ </td><td> $0 . 0 1 1 3 \pm 0 . 0 0 8 6$ </td><td> $\mathbf { 0 . 0 0 6 6 \pm 0 . 0 0 4 8 }$ </td><td> $0 . 0 6 7 0 \pm 0 . 0 5 0 8$ </td><td> $0 . 0 6 8 2 \pm 0 . 0 5 4 1$ </td><td> $\mathbf { 0 . 0 4 2 2 \pm 0 . 0 3 0 8 }$ </td></tr><tr><td>0.3</td><td> $0 . 0 1 1 2 \pm 0 . 0 0 8 4$ </td><td> $0 . 0 1 0 8 \pm 0 . 0 0 8 0$ </td><td> $\mathbf { 0 . 0 0 6 9 \pm 0 . 0 0 4 9 }$ </td><td> $0 . 0 6 7 8 \pm 0 . 0 4 9 4$ </td><td> $0 . 0 6 5 2 \pm 0 . 0 4 6 3$ </td><td> $\mathbf { 0 . 0 4 4 7 \pm 0 . 0 3 1 3 }$ </td></tr><tr><td>0.4</td><td> $0 . 0 1 4 6 \pm 0 . 0 0 7 0$ </td><td> $0 . 0 1 3 9 \pm 0 . 0 0 6 7$ </td><td> $\mathbf { 0 . 0 0 8 4 \pm 0 . 0 0 3 9 }$ </td><td> $0 . 0 8 6 3 \pm 0 . 0 3 9 8$ </td><td> $0 . 0 8 2 2 \pm 0 . 0 3 8 0$ </td><td> $\mathbf { 0 . 0 5 4 7 \pm 0 . 0 2 3 9 }$ </td></tr><tr><td>0.5</td><td> $0 . 0 1 7 8 \pm 0 . 0 0 7 7$ </td><td> $0 . 0 1 7 0 \pm 0 . 0 0 7 3$ </td><td> $\mathbf { 0 . 0 1 0 5 \pm 0 . 0 0 4 4 }$ </td><td> $0 . 1 0 7 0 \pm 0 . 0 4 4 0$ </td><td> $0 . 1 0 1 9 \pm 0 . 0 4 3 1$ </td><td> $\mathbf { 0 . 0 6 6 7 \pm 0 . 0 2 6 0 }$ </td></tr><tr><td>0.6</td><td> $0 . 0 2 0 4 \pm 0 . 0 0 9 8$ </td><td> $0 . 0 1 9 1 \pm 0 . 0 0 8 6$ </td><td> $\mathbf { 0 . 0 1 1 9 \pm 0 . 0 0 4 6 }$ </td><td> $0 . 1 2 2 5 \pm 0 . 0 6 1 0$ </td><td> $0 . 1 1 2 2 \pm 0 . 0 5 2 6$ </td><td> $\mathbf { 0 . 0 7 3 1 \pm 0 . 0 2 8 2 }$ </td></tr><tr><td>0.7</td><td> $0 . 0 2 2 7 \pm 0 . 0 0 8 5$ </td><td> $0 . 0 2 1 1 \pm 0 . 0 0 7 7$ </td><td> $\mathbf { 0 . 0 1 3 9 \pm 0 . 0 0 4 8 }$ </td><td> $0 . 1 3 3 4 \pm 0 . 0 5 2 9$ </td><td> $0 . 1 2 5 4 \pm 0 . 0 4 9 8$ </td><td> $\mathbf { 0 . 0 8 6 7 \pm 0 . 0 2 9 7 }$ </td></tr><tr><td>0.8</td><td> $0 . 0 2 3 4 \pm 0 . 0 0 8 9$ </td><td> $0 . 0 2 2 3 \pm 0 . 0 0 8 3$ </td><td> $\mathbf { 0 . 0 1 4 6 \pm 0 . 0 0 5 3 }$ </td><td> $0 . 1 4 1 1 \pm 0 . 0 5 5 9$ </td><td> $0 . 1 3 4 4 \pm 0 . 0 5 2 9$ </td><td> $\mathbf { 0 . 0 9 3 1 \pm 0 . 0 3 6 8 }$ </td></tr><tr><td>0.9</td><td> $0 . 0 2 6 6 \pm 0 . 0 0 7 2$ </td><td> $0 . 0 2 5 7 \pm 0 . 0 0 6 8$ </td><td> $\mathbf { 0 . 0 1 7 0 \pm 0 . 0 0 5 7 }$ </td><td> $0 . 1 6 2 0 \pm 0 . 0 4 7 8$ </td><td> $0 . 1 5 5 8 \pm 0 . 0 4 6 4$ </td><td> $\mathbf { 0 . 1 0 7 0 \pm 0 . 0 3 6 0 }$ </td></tr></table>

<table><tr><td>Rotation</td><td>N</td><td>Robust-DefReg</td><td>DefTransNet</td><td>CoCo-Reg</td><td>Win vs. Def.</td></tr><tr><td> $0 { - } 3 0 ^ { \circ }$ </td><td>234</td><td> $0 . 0 7 9 2 \pm 0 . 1 0 1 8$ </td><td> $0 . 0 7 7 1 \pm 0 . 1 0 3 0$ </td><td> $\mathbf { 0 . 0 3 5 4 \pm 0 . 0 4 6 5 }$ </td><td>80.3%</td></tr><tr><td> $> 3 0 { - } 6 0 ^ { \circ }$ </td><td>240</td><td> $0 . 1 1 6 0 \pm 0 . 1 1 0 1$ </td><td> $0 . 1 1 2 7 \pm 0 . 1 1 1 2$ </td><td> $\mathbf { 0 . 0 4 8 7 \pm 0 . 0 5 4 7 }$ </td><td>96.7%</td></tr><tr><td> $> 6 0 ^ { \circ }$ </td><td>252</td><td> $0 . 3 6 9 3 \pm 0 . 2 1 3 4$ </td><td> $0 . 3 9 5 4 \pm 0 . 2 3 2 4$ </td><td> $\mathbf { 0 . 0 7 8 3 \pm 0 . 0 8 9 8 }$ </td><td>99.2%</td></tr></table>

<table><tr><td>Category</td><td>DefTransNet</td><td>CoCo-Reg</td><td>∆ [95% CI] / Win</td><td></td><td>Category</td><td>DefTransNet</td><td>CoCo-Reg</td><td>∆ [95% CI] / Win</td></tr><tr><td>Bathtub</td><td>0.2807.±.0.2607</td><td>0.0863 ± 0.0947</td><td> $- 0 . 1 9 4 3 [ - 0 . 2 6 2 8 , - 0 . 1 3 2 7 ] / 9 2 . 9 \%$ </td><td></td><td>Monitor</td><td> $\overline { { 0 . 1 3 4 9 \pm 0 . 1 3 3 0 } }$ </td><td> $\overline { { { \bf 0 . 0 4 7 3 \pm 0 . 0 4 4 8 } } }$ </td><td> $\overline { { - 0 . 0 8 7 6 [ - 0 . 1 0 9 9 , - 0 . 0 6 8 1 ] / 9 2 . 6 \% } }$ </td></tr><tr><td>Bed</td><td> $0 . 2 3 3 4 \pm 0 . 2 8 0 6$ </td><td> $\mathbf { 0 . 0 5 1 8 \overset {  } { = } 0 . 0 6 6 6 }$ </td><td> $- 0 . 1 8 1 6 [ - 0 . 2 4 1 5 , - 0 . 1 2 5 7 ] / 9 3 . 7 \%$ </td><td></td><td>Night stand</td><td> $0 . 2 1 2 4 \overset { - } { \pm } 0 . 2 3 8 2$ </td><td> $\mathbf { 0 . 0 6 7 5 \overset { - } { \pm } 0 . 1 0 4 1 }$ </td><td> $- 0 . 1 4 4 9 \dot { [ - 0 . 1 9 3 9 , - 0 . 0 9 9 2 ] } / 8 5 . 1 \%$ </td></tr><tr><td>Chair</td><td> $0 . 1 7 8 5 \pm 0 . 1 7 2 0$ </td><td>0.0457 ± 0.0381</td><td> $- 0 . 1 3 2 8 [ - 0 . 1 6 5 6 , - 0 . 1 0 2 1 ] / 9 0 . 1 \%$ </td><td></td><td>Sofa</td><td> $0 . 1 9 8 9 \pm 0 . 2 2 4 0$ </td><td> $\mathbf { 0 . 0 4 1 5 \overset { - } { \pm } 0 . 0 4 6 4 }$ </td><td> $- 0 . 1 5 7 3 \dot { \left[ - 0 . 2 0 0 8 , - 0 . 1 1 7 9 \right] } / 9 6 . 5 \%$ </td></tr><tr><td>Desk</td><td>0.2502 ± 0.2476</td><td> $\mathbf { 0 . 0 6 2 0 \overset { - } { \pm } 0 . 0 5 7 8 }$ </td><td> $- 0 . 1 8 8 1 \dot { [ - 0 . 2 4 6 6 , - 0 . 1 3 6 4 ] } / 9 5 . 4 \%$ </td><td></td><td>Table</td><td> $\mathbf { 0 . 2 2 2 8 \pm 0 . 2 3 5 2 }$ </td><td> $\mathbf { 0 . 0 5 3 1 \pm 0 . 0 8 2 8 }$ </td><td> $- 0 . 1 6 9 7 \dot { [ - 0 . 2 1 9 2 , - 0 . 1 2 4 0 \dot { ] } } / 9 2 . 2 \%$ </td></tr><tr><td>Dresser</td><td> $0 . 2 0 4 8 \pm 0 . 1 9 4 3$ </td><td> $\mathbf { 0 . 0 6 7 7 \mathop { \pm } 0 . 0 9 5 1 }$ </td><td> $\begin{array} { r } { - 0 . 1 3 7 1 [ - 0 . 1 7 8 4 , - 0 . 1 0 0 2 ] / 9 4 . 1 \% } \\ { - 0 . 1 3 7 1 [ - 0 . 1 7 8 4 , - 0 . 1 0 0 2 ] / 9 4 . 1 \% } \end{array}$ </td><td></td><td>Toilet</td><td> $\mathbf { 0 . 1 3 1 3 \pm 0 . 1 2 7 9 }$ </td><td> $\mathbf { 0 . 0 4 5 3 \overset { - } { \pm } 0 . 0 4 3 6 }$ </td><td> $- 0 . 0 8 6 0 [ - 0 . 1 0 6 8 , - 0 . 0 6 6 4 ] / 9 0 . 1 \%$ </td></tr></table>

Table 5: Rotation-stratified descriptive analysis of the final test set. Values are $\mathbf { M C E } \pm \mathbf { S D }$ across objects. The last group contains observed rotations above $6 0 ^ { \circ }$ up to 91.7<sup>◦</sup>.  
Table 6: Category-stratified CoCo-Reg vs. DefTransNet MCE. $\mathbf { M e a n } \pm \mathbf { S D }$ are across objects; ∆ is CoCo-Reg−DefTransNet with bootstrap 95% CI. All ten categories are shown.

A second contribution is the patch-supervision strategy. Because source and target patches are constructed independently, a source patch may overlap with several valid target patches after deformation. Treating patch correspondence as a single-label classification problem would therefore impose an unnecessarily rigid target. The identity-corrected, multi-positive formulation instead allows several geometrically consistent patch relations to contribute to supervision while still encouraging the strongest overlap through the complementary best-overlap objective. This makes the patch branch better aligned with the non-rigid setting, where regional boundaries need not remain identical after deformation.

The paired analysis also shows that the improvement is systematic rather than being driven by a small number of favorable cases. CoCo-Reg performs better on the large majority of paired test objects, reduces the upper tail of the correspondence-error distribution, and markedly lowers both object-level and point-level severe-error rates. The same direction is observed across deformation strata and object categories. These results are important for dense deformation estimation because a small number of large correspondence errors can lead to locally incorrect displacement fields even when an average set-distance metric remains acceptable.

The present study evaluates the complete CoCo-Reg configuration on one final checkpoint per method and on controlled synthetic deformations of complete ModelNet10 shapes. The main next step is therefore to test whether the same advantage persists under more realistic conditions, including partial overlap, noise, outliers, topology changes, and real deforming surfaces such as soft tissue. A matched multi-seed study and separately trained hard-routing variants would also make it possible to isolate more precisely which parts of the patch-context design contribute most strongly to the observed gain. Beyond this, adaptive patch construction and confidence-weighted regional information could further improve the method when the reliability of patch context varies across the shape.

## 5 Conclusion

We presented CoCo-Reg, a dense non-rigid point-cloud registration method that introduces regional patch information without using patch predictions to restrict the final point-level correspondence search. The method combines independently constructed geometric patches, source–target patch interaction, identity-corrected multi-positive patch supervision, and residual projection of regional information back to dense point features. On 726 paired held-out ModelNet10 objects across deformation levels 0.1–0.9, CoCo-Reg consistently reduces correspondence error compared with the evaluated reference methods. The improvement is also reflected in Chamfer distance, HD95, tail-error statistics, and severe correspondence failures, indicating that the benefit extends beyond a reduction in average error. Overall, the results support the use of regional context as guidance for dense correspondence while preserving a globally searchable point-level matching stage. Future work will examine the same principle under partial, noisy, and real deforming observations and will further isolate the contribution of individual patch-context components through matched multi-seed and hard-routing comparisons.

## References

[1] Xuyang Bai, Zixin Luo, Lei Zhou, Hongkai Chen, Lei Li, Zeyu Hu, Hongbo Fu, and Chiew-Lan Tai. Pointdsc: Robust point cloud registration using deep spatial consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15859–15869, 2021. doi: 10.1109/CVPR46437.2021.01560.

[2] Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, and Yufeng Yue. Pointgpt: Autoregressively generative pre-training from point clouds. In Advances in Neural Information Processing Systems, volume 36, 2023.

[3] Haoqiang Fan, Hao Su, and Leonidas J. Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 605–613, 2017. doi: 10.1109/CVPR.2017.264.

[4] Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, and Konrad Schindler. Predator: Registration of 3d point clouds with low overlap. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4267–4276, 2021.

[5] Daniel P. Huttenlocher, Gregory A. Klanderman, and William J. Rucklidge. Comparing images using the hausdorff distance. IEEE Transactions on Pattern Analysis and Machine Intelligence, 15(9):850–863, 1993. doi: 10.1109/34.232073.

[6] Yang Li and Tatsuya Harada. Lepard: Learning partial point cloud matching in rigid and deformable scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5554–5564, 2022.

[7] Yang Li and Tatsuya Harada. Non-rigid point cloud registration with neural deformation pyramid. In Advances in Neural Information Processing Systems, volume 35, pages 27757–27768, 2022.

[8] David Männle, Jan Pohlmann, Sara Monji-Azad, Jürgen Hesser, Nicole Rotter, Annette Affolter, Anne Lammert, Benedikt Kramer, Sonja Ludwig, Lena Huber, and Claudia Scherl. Artificial intelligence directed development of a digital twin to measure soft tissue shift during head and neck surgery. PLOS ONE, 18(8):e0287081, 2023. doi: 10.1371/journal.pone.0287081.

[9] Sara Monji-Azad, Jürgen Hesser, and Nikolas Löw. A review of non-rigid transformations and learning-based 3d point cloud registration methods. ISPRS Journal of Photogrammetry and Remote Sensing, 196:58–72, 2023. doi: 10.1016/j.isprsjprs.2022.12.023.

[10] Sara Monji-Azad, Marvin Kinz, Claudia Scherl, David Männle, Jürgen Hesser, and Nikolas Löw. Synbench: A synthetic benchmark for non-rigid 3d point cloud registration. arXiv preprint arXiv:2409.14474, 2024. doi: 10.48550/arXiv.2409.14474.

[11] Sara Monji-Azad, David Männle, Jürgen Hesser, Jan Pohlmann, Nicole Rotter, Annette Affolter, Cleo Aron Weis, Sonja Ludwig, and Claudia Scherl. Point cloud registration for measuring shape dependence of soft tissue deformation by digital twins in head and neck surgery. Biomedicine Hub, 9(1):9–15, 2024. doi: 10.1159/000535421.

[12] Sara Monji-Azad, Marvin Kinz, Siddharth Kothari, Robin Khanna, Amrei Carla Mihan, David Männel, Claudia Scherl, and Jürgen Hesser. Deftransnet: A transformer-based method for non-rigid point cloud registration in the simulation of soft tissue deformation. Measurement Science and Technology, 36(7):076006, 2025. doi: 10.1088/1361-6501/ade613.

[13] Sara Monji-Azad, Marvin Kinz, David Männel, Claudia Scherl, and Jürgen Hesser. Robustdefreg: A robust coarse to fine non-rigid point cloud registration method based on graph convolutional neural networks. Measurement Science and Technology, 36(1):015426, 2025. doi: 10.1088/1361-6501/ad916c.

[14] Sara Monji-Azad, Rohit Beer, Marvin Kinz, Claudia Scherl, and Jürgen Hesser. Dine: Distance is not enough—learning global deformation priors for robust soft-tissue point cloud registration. arXiv preprint arXiv:2607.14946, 2026. doi: 10.48550/arXiv.2607.14946.

[15] Yatian Pang, Wenxiao Wang, Francis E. H. Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning. In Computer Vision – ECCV 2022, pages 604–621, 2022.

[16] Gilles Puy, Alexandre Boulch, and Renaud Marlet. Flot: Scene flow on point clouds guided by optimal transport. In Computer Vision – ECCV 2020, pages 527–544, 2020.

[17] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J. Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in Neural Information Processing Systems, volume 30, pages 5099–5108, 2017.

[18] Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, Slobodan Ilic, Dewen Hu, and Kai Xu. Geotransformer: Fast and robust point cloud registration with geometric transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8):9806–9821, 2023. doi: 10.1109/TPAMI.2023.3259038.

[19] Yue Wang and Justin M. Solomon. Deep closest point: Learning representations for point cloud registration. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 3523–3532, 2019.

[20] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics, 38 (5):146:1–146:12, 2019. doi: 10.1145/3326362.

[21] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1912–1920, 2015.

[22] Zi Jian Yew and Gim Hee Lee. Rpm-net: Robust point matching using learned features. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11824–11833, 2020.

[23] Zi Jian Yew and Gim Hee Lee. Regtr: End-to-end point cloud correspondences with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6677–6686, 2022. doi: 10.1109/CVPR52688.2022.00656.

[24] Hao Yu, Fu Li, Mahdi Saleh, Benjamin Busam, and Slobodan Ilic. Cofinet: Reliable coarse-tofine correspondences for robust point cloud registration. In Advances in Neural Information Processing Systems, volume 34, pages 23872–23884, 2021.

[25] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19313– 19322, 2022.