# Centering before Pruning: Lightweight Geometry Correction for Diversity-Based Visual Token Pruning in LVLMs

Shunjie Wen, Jaeyeon Lee, Dong-Wan Choi<sup>∗</sup>

Inha University

## Abstract

Large vision-language models (LVLMs) incur substantial inference costs due to their long and highly redundant visualtoken sequences. Diversity-based pruning mitigates this cost by selecting token subsets based on pairwise cosine similarity. We find, however, that similarities between raw visual tokens are strongly concentrated in the positive range, limiting their ability to distinguish non-redundant tokens. A natural way to improve this resolution is to center token features before computing cosine similarity. Centering indeed reveals a substantially richer pairwise structure, yet unexpectedly degrades pruning performance when used alone. We show that this apparent contradiction arises because the raw geometry does more than represent pairwise diversity: it also implicitly favors globally distinctive tokens, which tend to contain semantically informative content. Centering better resolves subset diversity but loses this useful token-wise preference, revealing that diversity and distinctiveness are entangled in the raw geometry. Based on this analysis, we propose the Centered Geometry Pruner (Cen-Prune), which measures subset diversity using centered cosine similarity while retaining raw-space distinctiveness as a complementary token-wise preference. This lightweight, plug-and-play correction leaves the underlying selection mechanism unchanged and incurs negligible computational overhead. Extensive experiments across multiple image- and video-understanding benchmarks and LVLM architectures demonstrate that Cen-Prune provides robust improvements in overall performance across existing diversitybased pruners.

## 1 Introduction

Large vision-language models (LVLMs) (Liu et al. 2023, 2024a; Zhang et al. 2025d; Bai et al. 2025) achieve strong multimodal understanding but incur substantial inference costs. Images are represented as dense grids of patch tokens (Dosovitskiy et al. 2021), and recent models further expand these sequences through dynamic-resolution processing for high-resolution images (Liu et al. 2024a; Zhu et al. 2025; Bai et al. 2025) and multi-frame processing for videos (Li et al. 2024b; Zhang et al. 2025d). Consequently, visual tokens can occupy a substantial portion of the sequence processed by the language model. As attention cost grows quadratically with sequence length, visual tokens become a major source of latency and memory consumption (Zhang et al. 2025c), motivating the reduction of visual tokens for eficient LVLM inference.

Visual token pruning ofers a practical way to reduce this cost (Zhang et al. 2025c). Since visual representations contain substantial redundancy, many tokens can be removed with limited performance degradation. Trainingfree methods identify the tokens to retain using diferent criteria. Attention-based methods rank tokens using attention from the vision encoder or the language model (Yang et al. 2025; Zou et al. 2025; Zhang et al. 2025a). Diversity based methods instead retain mutually dissimilar tokens, either through purely geometric (Alvar et al. 2025), instructionconditioned (Zhang et al. 2025b), or sensitivity-guided selection (Kim et al. 2026). Other methods suppress redundancy through semantic grouping (Fang et al. 2026) or multimodal coverage (Dong et al. 2026). Despite these variations, diversity-based selection fundamentally relies on pairwise cosine similarity to measure visual-token diferences.

Yet, the geometry underlying this similarity has remained largely unexplored (Lee et al. 2026). Diversity-based pruning assumes that the vision encoder provides a proper embedding space to distinguish redundant from non-redundant tokens. We find, however, that pairwise similarities among visual tokens are strongly concentrated in the positive range, mak ing most tokens appear mutually similar. Existing selectors can still suppress near-duplicates, but the resulting subsets remain composed of positively similar tokens. Interestingly, this restricted geometry is not entirely uninformative: it implicitly favors globally distinctive tokens, which are often semantically important. Thus, existing diversity-based pruning operates on a geometry that retains an implicit importance signal but provides limited resolution for subset diversity.

Feature centering appears to provide an intuitive remedy. It exposes the missing diversity by removing the shared component of visual tokens, and spreads pairwise similarities into a substantially broader range and reveals centroid-relative directions that are obscured in the raw geometry. Nevertheless, directly applying existing selectors to this centered geometry reduces downstream performance. Centering removes the implicit importance preference of the raw geometry, allowing tokens to be selected primarily because their residual directions difer from the current subset, regardless of whether those variations are informative. This reveals a fundamental tension: the raw geometry preserves token importance but poorly resolves diversity, whereas the centered geometry reveals diversity but cannot determine which diverse tokens are worth retaining.

Motivated by this observation, we propose Cen-Prune, a lightweight, plug-and-play framework that explicitly unites importance and diversity for visual token pruning. Cen-Prune assigns the two roles to complementary geometries: it constructs the pairwise similarity matrix from centered features to capture subset diversity, while assessing token importance through angular distinctiveness in the raw feature geometry. Specifically, we introduce a raw-space distinctiveness score derived from raw cosine similarities and incorporate it into the original selection rule without changing the underlying pruning procedure. Across diverse image- and videounderstanding benchmarks, Cen-Prune provides robust performance gains for existing diversity-based pruners on multiple LVLM architectures with negligible computational overhead.

## 2 Related Works

Visual token pruning (VTP) reduces the inference cost of large vision-language models by retaining a compact set of visual tokens. Existing methods can be broadly understood through two complementary selection principles: token importance and subset diversity.

Importance-Based Visual Token Pruning. Importancebased methods estimate the contribution of each visual token using signals such as attention, instruction relevance, or semantic coverage. VisPruner (Zhang et al. 2025a) estimates token importance from visual attention and subsequently removes redundancy based on feature similarity. VisionZip (Yang et al. 2025) retains dominant tokens while merging less important tokens into contextual representations according to semantic similarity. HoloV (Zou et al. 2025) preserves holistic visual evidence through crop-wise adaptive token allocation, while MMTok (Dong et al. 2026) formulates selection as a multimodal maximum-coverage problem over textual and visual tokens. These approaches primarily determine which visual evidence should be preserved, although several also incorporate redundancy reduction to avoid repeatedly retaining similar content.

Diversity-Based Visual Token Pruning. Diversity-based methods consider relationships among tokens to retain a complementary subset rather than ranking each token independently. PruneSID (Fang et al. 2026) clusters semantically related tokens using Principal Semantic Components Analysis (PSCA) and suppresses redundancy within each group through Non-Maximum Suppression (NMS). DivPrune (Alvar et al. 2025) more explicitly formulates token selection as a Max–Min Diversity Problem (MMDP), while CD-Pruner (Zhang et al. 2025b) combines determinantal diversity with image–text relevance for query-aware pruning. ZOO-Prune (Kim et al. 2026) further couples a diversity objective with zeroth-order sensitivity estimation to retain influential yet complementary tokens.

Diversity–Importance Entanglement. Diversity-based methods, however, are not independent of token importance. CDPruner (Zhang et al. 2025b) and ZOO-Prune (Kim et al. 2026) explicitly incorporate relevance or sensitivity scores. More importantly, our empirical analysis shows that even DivPrune (Alvar et al. 2025), despite having no explicit importance term, favors tokens that are distinctive relative to the remaining tokens in the image. Such tokens not only increases subset diversity but are also often visually distinctive and informative. Thus, part of the benefit attributed to diversity may arise from a token-wise preference already encoded in the raw similarity geometry. Prior work has not clearly separated this preference from subset-level diversity, which is the focus of our analysis.

## 3 Preliminaries

This section formulates visual token pruning and two representative diversity-based selection objectives that form the basis of our method.

Visual Token Pruning Problem. Given an image, the visual encoder of an LVLM produces a sequence of visual tokens $\pmb { V } = \left[ \pmb { v } _ { 1 } , \dots , \pmb { v } _ { N } \right] ^ { \top } \in \mathbb { R } ^ { N \times D }$ , which is fed into the language model together with the text tokens. Visual token pruning aims to select an index set $T \subseteq [ N ]$ with $| T | = n \ll N$ such that retaining only $V _ { T }$ minimizes the degradation in downstream task performance, while reducing the visual sequence length to be processed by the language model. For a token ${ \mathbf { } } v _ { i }$ , we define its $\ell _ { 2 }$ -normalized representation as $\hat { \pmb { v } } _ { i } = \pmb { v } _ { i } / \lVert \pmb { v } _ { i } \rVert _ { 2 }$ and its centered representation as ${ \bar { \mathbf { v } } } _ { i } = { \mathbf { v } } _ { i } - \mu $ , where $\begin{array} { r } { \pmb { \mu } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \pmb { v } _ { j } } \end{array}$ is the mean token of the corresponding image. We use the same accents for the associated token matrices: $\hat { V } = [ \hat { \pmb { v } } _ { 1 } , \dots , \hat { \pmb { v } } _ { N } ] ^ { \top }$ and $\bar { \pmb { V } } = [ \bar { \pmb { v } } _ { 1 } , \ldots , \bar { \pmb { v } } _ { N } ] ^ { \top }$

Diversity-Based Selection. To minimize the performance drop after token pruning, diversity-based methods select a subset of mutually dissimilar visual tokens, treating pairwise dissimilarity as a surrogate for preserving non-redundant visual information. More formally, let $S = \hat { V } \hat { V } ^ { \top }$ denote the cosine-similarity matrix, where $S _ { i j } = \langle \hat { \pmb { v } } _ { i } , \hat { \pmb { v } } _ { j } \rangle$ , and let $D _ { i j } = 1 - S _ { i j }$ be the corresponding distance. The pruning problem is then formulated as selecting a subset $T \subseteq [ N ]$ with $| T | = n$ that maximizes diversity under this pairwise geometry. Existing diversity-based visual token pruning methods mainly follow two representative selection mechanisms: max–min diversity and determinantal diversity.

Max–Min Diversity. The max–min diversity objective, adopted by DivPrune (Alvar et al. 2025) and ZOO-Prune (Kim et al. 2026), maximizes the minimum pairwise distance among the selected tokens:

$$
T ^ { \star } = \arg \operatorname* { m a x } _ { | T | = n } \operatorname* { m i n } _ { i , j \in T \atop i \neq j } D _ { i j } .\tag{1}
$$

This objective is typically approximated by greedy farthestpoint selection, which iteratively adds the token that maximizes its minimum distance to the already selected set.

![](images/644b16e1982f8844ab813bc37d827edabcb62aa88dd44650f6fbe7aa8937f1ab.jpg)  
Figure 1: Raw and centered visual-token geometry. (a) A shared mean direction limits angular contrast in raw space, leading max–min selection to R. (b) Centering reveals diverse residual directions. (c) Centered geometry alone may favor weakly distinctive A, whereas distinctiveness-aware selection favors B, which is both diverse and distinctive. Color indicates distinctiveness, measured by distance from the imagewise mean.

Determinantal Diversity. The determinantal diversity objective, adopted by CDPruner (Zhang et al. 2025b), favors subsets that span a large volume:

$$
T ^ { \star } = \arg \operatorname* { m a x } _ { | T | = n } ~ \operatorname* { d e t } ( L _ { T } ) , \quad L = \operatorname { d i a g } ( q ) S \operatorname { d i a g } ( q ) ,\tag{2}
$$

where $q _ { i }$ denotes the quality score of token i, instantiated in CDPruner as its relevance to the instruction.

Although these objectives encode diversity diferently, both determine pairwise non-redundancy through the cosine geometry induced by S. Max–min selection uses the cosine distance $D _ { i j } = 1 - S _ { i j }$ , whereas determinantal selection constructs its diversity kernel from S. This common reliance implicitly assumes that the visual features V produced by the encoder induce a similarity geometry suitable for diversity-based pruning. We therefore first investigate the geometry underlying visual-token similarities and how it affects diversity-based selection.

## 4 Geometry Analysis and Cen-Prune

We begin by examining the assumption raised above: whether the cosine-similarity matrix S induced by visual-token features provides a reliable geometry for diversity-based pruning. We analyze $\pmb { S }$ computed from LLaVA-1.5-7B (Liu et al. 2023) visual tokens on 1,000 COCO val2017 images (Lin et al. 2014). Through this analysis, we uncover two key observations: (1) the raw similarity matrix is highly concentrated, limiting its ability to distinguish redundant from non-redundant tokens, and (2) centering token features alleviates this concentration but can suppress token-specific information in the original feature distribution. These findings motivate Cen-Prune, which combines subset diversity measured in the centered geometry with token distinctiveness captured by a raw-space distinctiveness score.

![](images/4d7023b262da3ebb253c596715e21827650e86411825c1e8c2aa8fcdf76b3a91.jpg)  
Figure 2: Cosine similarity over visual-token pairs. Frequencies are computed with 0.01-wide bins and smoothed for display. Raw features (blue) concentrate inside the acute range, with a median angle of 72<sup>◦</sup>. Centered features (orange) shifts mass into the obtuse range, producing a negative lobe absent from the raw matrix.

<table><tr><td>Selector</td><td>Mean Cos.</td><td>Obtuse (%)</td><td>&lt; −0.5 (%)</td><td>&gt; 0.99 (%)</td></tr><tr><td>All pairs</td><td>0.37</td><td>2.3</td><td>0.0</td><td>5.3</td></tr><tr><td>Random</td><td>0.37</td><td>2.3</td><td>0.0</td><td>5.3</td></tr><tr><td>MMDP</td><td>0.26</td><td>6.9</td><td>0.0</td><td>0.0</td></tr><tr><td>DPP</td><td>0.33</td><td>2.5</td><td>0.0</td><td>0.0</td></tr></table>

Table 1: Geometry statistics of retained subsets selected by diferent diversity-based selectors (size = 32). Random is averaged over 10 samplings. MMDP with local search improves the max-min objective by only 2.6%.

## 4.1 Visual Similarity Is Highly Concentrated

A Positive-Similarity Bias. We first examine the distribution of cosine similarities between distinct visual tokens in the raw similarity matrix S. As a reference, two independently and uniformly sampled unit vectors in a 4,096- dimensional space are nearly orthogonal, with their angle concentrated around $9 0 . 0 ^ { \circ }$ and a standard deviation of approximately 0.9<sup>◦</sup>. However, as shown in Fig. 2, visual tokens deviate substantially from this reference. Their pairwise angles concentrate well below 90<sup>◦</sup>, corresponding to predominantly positive cosine similarities, with a median angle of $7 2 . 0 ^ { \circ }$ and a mean cosine similarity of $0 . 3 6 6 3 { \pm } 0 . 0 2 4 6$ . Only 2.3% of token pairs have negative cosine similarity, and none falls below −0.5 (Tab. 1). Meanwhile, 5.3% of pairs are nearduplicates with cosine similarity above 0.99.

A similar concentration is reported by (Lee et al. 2026), highlighting the geometric gap between visual and text tokens toward word-like image tokens. They report a mean cosine similarity of 0.3823 ± 0.0018 among LLaVA post-projector visual tokens, compared with 0.0378 ± 0.0002 among text tokens. Together, these results show that raw visual-token similarities occupy a narrow, predominantly positive range. As illustrated in Fig. 1(a), this shared directional bias compresses the angular contrast available for diversity-based selection. The raw geometry clearly identifies near-duplicates, but provides limited contrast for distinguishing broader degrees of non-redundancy among the remaining tokens.

Selectors Only Deduplicate. One might still expect a strong selector to recover the few dissimilar token pairs hidden in the raw geometry. Tab. 1 tests this possibility by selecting 32 tokens using MMDP from DivPrune (Alvar et al. 2025) and DPP from CDPruner (Zhang et al. 2025b), with query relevance disabled for DPP. Random sampling closely reproduces the all-pairs statistics, confirming the broad homogeneity of the matrix.

![](images/5567d2a92ba5f4a47e70b9ab8b04afea120ea52f19a4df0f7b8a39a006ebbb89.jpg)

![](images/642f76530306646a08acf3249f23a954f66378ffc0c4dd02cec1c093508bb0e8.jpg)  
Figure 3: Comparison of selection on the raw/centered matrix. (Left) Cosine similarity distributions over selected subset $( n = 3 2 )$ using DivPrune. Light shaded areas represent all pair reference. (Right) As the retention budget tightens, the fraction of selected negative-cosine pairs rises under both matrices but far more steeply for centered selection (lines), while its relative accuracy stays below raw selection and the gap widens (bars).

MMDP eliminates near-duplicates, reducing the fraction of pairs with cosine similarity above 0.99 from 5.3% to 0.0%, and increases negative-cosine pairs from 2.3% to 6.9%. Nevertheless, the selected subset still has a mean cosine similarity of 0.26, with more than 93% positive-cosine pairs. A singleswap local search changes these statistics only negligibly, indicating this limitation is not merely due to greedy search.

DPP exhibits a similar limitation. For two unit vectors, its Gram determinant is $1 - \cos ^ { 2 } \theta$ , making the objective insensitive to the sign of pairwise similarity. It therefore does not specifically favor negatively correlated pairs, yielding only $2 . 5 \% .$ , close to random sampling. Across diferent objectives, these selectors efectively remove near-duplicates but fail to construct broadly dissimilar subsets. The primary bottleneck thus lies in the underlying geometry of S.

## 4.2 Mean Centering for Pure Diversification

Centering Reveals Residual Diversity. The strong positive-similarity bias suggests removing the shared component across visual tokens. We therefore center each token feature as ${ \bar { \boldsymbol { v } } } _ { i } = { \boldsymbol { v } } _ { i } - { \boldsymbol { \mu } } .$ , where $\pmb { \mu }$ denotes the mean feature within an image. As illustrated in Fig. 1(b), centering removes the common direction and exposes the residual variation among tokens. Consequently, pairwise cosine similarities spread substantially into the negative range (Fig. 2, orange). After centering, 32% of all token pairs have cosine similarity below −0.5, with the distribution peaking near −0.72. Applying the same greedy MMDP procedure to the centered similarity matrix can therefore select tokens with strongly opposing residual directions, which are nearly absent from the raw geometry. In this sense, centering moves selection toward pure diversification: tokens are compared solely by their residual directions, allowing the selector to prioritize angular separation more directly.

Pure Diversification Is Not Enough. The broader angular range exposed by centering substantially changes the subsets produced by diversity-based selection. As shown in Fig. 3, the centered selector increasingly favors opposing residual directions as the token budget decreases. Under DivPrune, the fraction of negative-cosine pairs rises from $1 0 . 0 \%$ at $n =$ 128 to 51.7% at $n = 8 ,$ , compared with an increase from 2.1% to 18.8% in the raw geometry. $\mathrm { A t } n = 8 ,$ the centered subset even exhibits a negative mean pairwise cosine similarity.

However, this stronger diversification does not translate into better pruning. ${ \mathrm { A t ~ } } n = 3 2$ , centered selection underperforms its raw counterpart on most benchmarks, and its relative accuracy remains lower across token budgets, with the gap widening as the budget becomes more restrictive (Fig. 3). This does not imply that negative-cosine pairs are undesirable. Rather, it shows that maximizing separation among residual directions alone is insuficient to determine which tokens should be retained.

Centering Leaves Out Distinctiveness. For each centered token feature ${ \bar { \mathbf { v } } } _ { i } = { \mathbf { v } } _ { i } - \mu ,$ selection relies on the cosine similarity: $\bar { S } _ { i j } = ( \bar { \pmb { v } } _ { i } ^ { \top } \bar { \pmb { v } } _ { j } ) / \dot { ( } \| \bar { \pmb { v } } _ { i } \| \| \bar { \pmb { v } } _ { j } \| )$ . Because this similarity normalizes each residual, it captures diferences in residual direction but discards residual magnitude, $\lVert \pmb { v } _ { i } - \pmb { \mu } \rVert$ . This magnitude measures how strongly a token deviates from the common feature pattern shared across the image: tokens close to the mean largely follow this pattern and are more likely to be globally redundant, whereas tokens farther from the mean contain features less shared with the remaining tokens. Consequently, a weak residual can appear highly diverse solely because of its direction, while a token with a stronger and potentially more informative residual receives no additional preference, as illustrated in Fig. 1(c). We view this residual magnitude as a token-level distinctiveness signal, which we formalize as a distinctiveness score in Sec. 4.3. Although high distinctiveness does not necessarily imply semantic importance and may simply reflect an outlier, our empirical results show that highly distinctive tokens tend to capture semantically salient or globally non-redundant content, and that restoring this preference consistently improves centered selection. This suggests that centered directions and residual magnitudes provide complementary signals: the former captures subset diversity, while the latter provides a soft preference for individually distinctive tokens. Cen-Prune combines these signals to preserve the richer diversity of the centered geometry without discarding the useful token-wise preference retained in the original feature distribution.

## 4.3 Method: Cen-Prune

Our analysis identifies two complementary roles ofthe shared component in visual-token features. Removing it exposes the angular contrast needed for subset diversity, while each token’s relation to it provides a useful distinctiveness signal. Cen-Prune separates these roles: diversity is measured in the centered geometry, whereas distinctiveness is estimated from the uncentered geometry and incorporated as a soft selection preference. Figure 4 summarizes this procedure.

Centered Geometry for Subset Diversity. Let $\begin{array} { r l } { \boldsymbol { X } } & { { } = } \end{array}$ $[ { \pmb x } _ { 1 } , \dots , { \pmb x } _ { N } ] ^ { \top } \in \mathbb { R } ^ { \mathbf { \tilde { N } } \times D }$ denote the working token representation, which is either the original feature matrix V or its standardized variant described below. We first remove the image-wise feature mean:

![](images/ab13e40ec1c8e5121acdfc1cefc3fa4b7b148a9de04b4f32a340a6827009de3d.jpg)  
Figure 4: Overview of Cen-Prune. The traditional path feeds a diversity-based selector with the raw-feature similarity matrix (S). Cen-Prune uses the shared mean direction twice: removed to build the selection matrix (S<sup>¯</sup>), and retained as a per-token distinctiveness score $g _ { i }$ computed from S. The selector is unchanged, yielding the retained tokens $V _ { T }$

$$
\pmb { \mu } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \pmb { x } _ { j } , \qquad \bar { \pmb { x } } _ { i } = \pmb { x } _ { i } - \pmb { \mu } .\tag{3}
$$

The centered similarity matrix is then constructed from the normalized residuals:

$$
{ { \bar { S } } _ { i j } } = \left. { { \hat { \bar { x } } } _ { i } , { { \hat { \bar { x } } } _ { j } } } \right. , \qquad { { \hat { \bar { x } } } _ { i } } = \frac { { { \bar { x } } _ { i } } } { \left\| { { \bar { x } } _ { i } } \right\| _ { 2 } } .\tag{4}
$$

By removing the shared component, $\bar { S }$ exposes diferences among residual directions and provides the angular contrast used to diversify the retained subset.

Token Distinctiveness from Uncentered Geometry. Centering is used only to construct the subset-diversity geometry. To retain the token-wise signal removed by this transformation, we measure how weakly each token aligns with the remaining tokens before centering. Let $S _ { i j } = \langle \hat { { \bf x } } _ { i } , \hat { { \bf x } } _ { j } \rangle$ denote the uncentered cosine similarity. We define the distinctiveness score of token i as its negated mean similarity to the other tokens:

$$
g _ { i } = - \frac { 1 } { N - 1 } \sum _ { j \neq i } S _ { i j } .\tag{5}
$$

This score has a direct geometric interpretation. Let $\textbf { \em m } =$ $\textstyle { \frac { 1 } { N } } \sum _ { j } { \hat { \mathbf { x } } } _ { j }$ denote the mean of the normalized uncentered features. Then,

$$
g _ { i } = \frac { 1 - N \langle \hat { \pmb x } _ { i } , \pmb m \rangle } { N - 1 } ,\tag{6}
$$

which is monotonically related to $\| \hat { \pmb x } _ { i } - { \pmb m } \| _ { 2 } ^ { 2 }$ . Thus, a high $g _ { i }$ identifies a token that lies far from the feature pattern shared across the image, whereas a low $g _ { i }$ indicates that the token largely follows that pattern. We interpret this quantity as token-level distinctiveness, rather than as a standalone measure of semantic importance, and use it only as a soft preference within diversity-based selection. The scores are min–max normalized within each image to obtain $\tilde { g } _ { i } \in [ 0 , 1 ]$

Integration with Diversity-Based Selectors. Cen-Prune preserves the optimization procedure of the underlying selector while replacing its raw geometry with S<sup>¯</sup> and incorporating $\tilde { \mathbf { g } }$ as a token-wise quality signal. For the greedy MMDP selector used by DivPrune (Alvar et al. 2025), we construct the distance matrix D with entries $D _ { i j } = 1 - \bar { S } _ { i j }$ After the selector’s original initialization, each greedy step adds:

$$
i ^ { \star } = \arg \operatorname* { m a x } _ { i \notin { T } } \tilde { g } _ { i } \operatorname* { m i n } _ { s \in { T } } D _ { i s } ,\tag{7}
$$

until $| T | = n . \mathrm { A }$ token is therefore preferred when it is both directionally separated from the retained set and individually distinctive in the original geometry. For the DPP-based selector used by CDPruner (Zhang et al. 2025b), we incorporate the same two signals into its kernel. Let $\tilde { r } _ { i }$ denote the original instruction-relevance score. The corrected kernel is

$$
\begin{array} { r } { \pmb { L } ^ { \mathrm { C e n } } = \mathrm { d i a g } ( \tilde { \pmb { r } } \odot \tilde { \pmb { g } } ) \bar { S } \mathrm { d i a g } ( \tilde { \pmb { r } } \odot \tilde { \pmb { g } } ) , } \end{array}\tag{8}
$$

where $\odot$ denotes element-wise multiplication. The original determinantal selection procedure is then applied to $\check { L } ^ { \mathrm { C e n } }$ Thus, both selectors use centered directions to measure subset diversity and uncentered distinctiveness to determine which diverse tokens should receive preference.

## 5 Experiments

## 5.1 Experimental Setups

Models and Baselines. We evaluate Cen-Prune across representative image and video LVLMs, including LLaVA-1.5 (Liu et al. 2023), LLaVA-NeXT (Liu et al. 2024a), LLaVA-Video (Zhang et al. 2025d), and Qwen2.5-VL (Bai et al. 2025). We compare against recent visual-token pruning methods: VisionZip (Yang et al. 2025), HoloV (Zou et al. 2025), VisPruner (Zhang et al. 2025a), DivPrune (Alvar et al. 2025), CDPruner (Zhang et al. 2025b), PruneSID (Fang et al. 2026), MMTok (Dong et al. 2026), and ZOO-Prune (Kim et al. 2026). All methods are evaluated using their oficial implementations under a unified evaluation protocol.

Datasets. Following prior work (Kim et al. 2026; Fang et al. 2026), we evaluate image understanding on 11 benchmarks: GQA (Hudson and Manning 2019), MMBench (Liu et al. 2024b), MME (Fu et al. 2025a), POPE (Li et al. 2023), ScienceQA (Lu et al. 2022), VQAv2 (Goyal et al. 2017), TextVQA (Singh et al. 2019), OCRBench (Liu et al. 2024c), VizWiz (Gurari et al. 2018), SEED-Bench-Image (Li et al. 2024a), and MMMU (Yue et al. 2024). We further evaluate LLaVA-Video-7B on four video-understanding benchmarks: MVBench (Li et al. 2024c), Video-MME (Fu et al. 2025b), the EgoSchema subset (Mangalam, Akshulakov, and Malik 2023), and SEED-Bench-Video (Li et al. 2024a).

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>VQAV2</td><td>VQAText</td><td>OCR-B</td><td>VizWiz</td><td>SEED-I</td><td>MMMU</td><td>Avg.</td></tr><tr><td>Upper Bound (576 Tokens)</td><td>61.9</td><td>64.7</td><td>1511</td><td>85.9</td><td>69.6</td><td>78.5</td><td>58.2</td><td>31.3</td><td>54.4</td><td>66.2</td><td>36.3</td><td>100.0%</td></tr><tr><td>LLaVA-1.5-7B</td></tr><tr><td>VisionZip (CVPR25)</td><td>57.5</td><td>62.3</td><td>1439 83.1</td><td>68.6</td><td>75.6</td><td>56.9</td><td>Retain 128 Tokens (↓ 77.8%) 30.0</td><td>54.4</td><td>61.5</td><td>37.8</td><td>97.0%</td></tr><tr><td>HoloV (NeurIPS25)</td><td>57.4</td><td>63.0</td><td>1441</td><td>82.1</td><td>67.9</td><td>75.4</td><td>55.9</td><td>28.5 54.7</td><td>61.2</td><td>36.7</td><td>96.0%</td></tr><tr><td>VisPruner (ICCV25)</td><td>58.4</td><td>62.5</td><td>1421</td><td>85.0</td><td>69.1</td><td>75.8 57.2</td><td>29.5</td><td>55.3</td><td>61.6</td><td>36.4</td><td>97.0%</td></tr><tr><td>PruneSID (ICLR26)</td><td>58.2</td><td>62.2</td><td>1427</td><td>84.8 68.4</td><td>75.4</td><td>54.6</td><td>28.5</td><td>55.8</td><td>62.1</td><td>36.3</td><td>96.3%</td></tr><tr><td>DivPrune (CVPR25)</td><td>59.3</td><td>62.4</td><td>1394</td><td>86.8 68.5</td><td>76.0</td><td>56.2</td><td>28.8</td><td>56.4</td><td>62.3</td><td>35.9</td><td>96.9%</td></tr><tr><td>+ Ours</td><td>59.3</td><td>62.5</td><td>1416</td><td>86.8 68.6</td><td>76.7</td><td>57.2</td><td>29.8</td><td>55.8</td><td>63.2</td><td>36.3</td><td>97.7%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>59.5</td><td>62.1</td><td>1387</td><td>86.8</td><td>68.5 76.4</td><td>56.8</td><td>29.1</td><td>56.0</td><td>63.0</td><td>35.8</td><td>97.1%</td></tr><tr><td>+ Ours</td><td>59.7</td><td>62.5</td><td>1445</td><td>86.5</td><td>68.9</td><td>76.7 57.0</td><td>29.5</td><td>55.5</td><td>63.5</td><td>36.9</td><td>98.0%</td></tr><tr><td>MMTok* (ICLR26)</td><td>59.3</td><td>61.9</td><td>1435</td><td>86.3</td><td>69.0</td><td>76.4 56.9</td><td>29.8</td><td>55.7</td><td>63.2</td><td>35.6</td><td>97.5%</td></tr><tr><td>CDPruner (NeurIPS25)</td><td>59.6</td><td>62.4</td><td>1410</td><td>86.8</td><td>68.7</td><td>76.6 56.4</td><td>28.9</td><td>56.3</td><td>64.0</td><td>35.3</td><td></td></tr><tr><td>+ Ours</td><td>60.2</td><td>63.1</td><td>1442</td><td>86.7</td><td>68.2</td><td></td><td></td><td>55.4</td><td>63.9</td><td>36.1</td><td>97.3% 98.1%</td></tr><tr><td>LLaVA-1.5-7B</td><td></td><td></td><td></td><td></td><td></td><td>76.9</td><td>57.0</td><td>30.0</td><td></td><td></td><td></td></tr><tr><td>VisionZip (CVPR25)</td><td>55.1</td><td>60.1</td><td>1370</td><td>77.1</td><td>69.0</td><td>72.4</td><td>Retain 64 Tokens (↓ 88.9%)</td><td></td><td></td><td></td><td></td></tr><tr><td>HoloV (NeurIPS25)</td><td>55.1</td><td>60.2</td><td>1366</td><td>76.9</td><td>68.8 72.7</td><td>55.4 55.1</td><td>28.3 27.1</td><td>54.8 55.8</td><td>57.8 58.2</td><td>36.4</td><td>93.4%</td></tr><tr><td>VisPruner (ICcV25)</td><td>55.6</td><td>59.6</td><td>1357</td><td>80.4</td><td>68.5 72.6</td><td>55.9</td><td>29.2</td><td>56.6</td><td>58.0</td><td>36.1</td><td>93.1%</td></tr><tr><td>PruneSID(ICLR26)</td><td>57.7</td><td>60.1</td><td>1401</td><td>84.5</td><td>68.8</td><td>73.8 54.7</td><td>27.2</td><td>56.5</td><td>60.2</td><td>35.7</td><td>94.1%</td></tr><tr><td>DivPrune (CVPR25)</td><td>57.8</td><td>59.5</td><td>1351</td><td>85.6</td><td>68.0</td><td>74.1 54.7</td><td>27.4</td><td></td><td></td><td>37.2</td><td>95.3%</td></tr><tr><td>+ Ours</td><td>58.4</td><td>60.5</td><td>1370</td><td>85.3</td><td>68.5 75.3</td><td>56.6</td><td></td><td>57.4</td><td>60.3</td><td>35.7</td><td>94.8%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>58.4</td><td>60.1</td><td>1375</td><td>85.7</td><td>68.2 74.9</td><td>55.6</td><td>28.5 27.6</td><td>57.4 57.3</td><td>60.8</td><td>36.2</td><td>96.2%</td></tr><tr><td>+ Ours</td><td>58.2</td><td>60.7</td><td>1390</td><td>85.3</td><td>67.8</td><td>75.3 56.2</td><td>28.4</td><td>56.9</td><td>61.0 61.7</td><td>35.6 36.1</td><td>95.5%</td></tr><tr><td>MMTok* (ICLR26)</td><td>58.2</td><td>60.4</td><td>1404</td><td>85.8</td><td>68.7</td><td></td><td></td><td></td><td></td><td></td><td>96.2%</td></tr><tr><td>CDPruner*</td><td>58.8</td><td>61.3</td><td>1400</td><td>87.0</td><td>68.5</td><td>75.2 55.7</td><td>28.6</td><td>57.2</td><td>61.5</td><td>36.0</td><td>96.3%</td></tr><tr><td>(NeurIPS25) + Ours</td><td>59.0</td><td>61.9</td><td></td><td></td><td></td><td>75.4 55.4</td><td>28.0</td><td>57.0</td><td>62.6</td><td>35.4</td><td>96.3%</td></tr><tr><td>LLaVA-1.5-7B</td><td></td><td></td><td>1414</td><td>86.1</td><td>68.8</td><td>76.1</td><td>56.3</td><td>28.1</td><td>56.9</td><td>62.7</td><td>36.3 97.0%</td></tr><tr><td>VisionZip (CVPR25)</td><td>51.8</td><td>57.3</td><td>1243</td><td>69.1</td><td>68.8</td><td>67.1</td><td>Retain 32 Tokens (↓ 94.4%) 53.2</td><td>25.4 55.7</td><td>53.2</td><td>35.1</td><td></td></tr><tr><td>HoloV (NeurIPS25)</td><td>52.8</td><td>59.0</td><td>1262</td><td>70.5</td><td>68.9 68.7</td><td>53.7</td><td>25.9</td><td>56.3</td><td>55.0</td><td>36.0</td><td>88.3% 89.9%</td></tr><tr><td>VisPruner (ICcV25)</td><td>51.6</td><td>56.9</td><td>1261</td><td>73.1</td><td>68.1 67.6</td><td>53.6</td><td>25.8</td><td>55.8</td><td>53.7</td><td>36.2</td><td></td></tr><tr><td>PruneSID (ICLR26)</td><td>54.7</td><td>56.0</td><td>1297</td><td>79.9</td><td>67.2 70.5</td><td>52.4</td><td>24.9</td><td>56.7</td><td>56.9</td><td>35.3</td><td>89.3%</td></tr><tr><td>DivPrune (CVPR25)</td><td>55.0</td><td>58.3</td><td>1306</td><td>81.7</td><td>68.4 71.1</td><td>53.1</td><td>25.5</td><td>57.0</td><td>57.2</td><td></td><td>90.7%</td></tr><tr><td>+ Ours</td><td>56.0</td><td>59.5</td><td>1335</td><td>82.8</td><td>68.6 72.9</td><td>55.3</td><td>26.2</td><td>57.6</td><td>58.6</td><td>35.2</td><td>91.8%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>56.0</td><td>57.7</td><td>1313</td><td>83.0</td><td>69.1 72.3</td><td>53.8</td><td>26.1</td><td>57.2</td><td>57.9</td><td>34.9</td><td>93.4%</td></tr><tr><td>+ Ours</td><td>56.2</td><td>59.8</td><td>1354</td><td>83.1</td><td>69.2 73.1</td><td>55.0</td><td>26.2</td><td>57.1</td><td>58.6</td><td>34.9 34.9</td><td>92.6% 93.6%</td></tr><tr><td>MMTok* (ICLR26)</td><td>56.2</td><td></td><td></td><td></td><td>69.0 73.1</td><td>53.7</td><td>26.4</td><td>57.2</td><td>59.6</td><td>34.8</td><td>93.4%</td></tr><tr><td>CDPruner* (NeurIPS25)</td><td>57.2</td><td>58.8 60.1</td><td>1358 1359</td><td>82.9 87.2 69.0</td></table>

Table 2: Performance comparison on LLaVA-1.5-7B and LLaVA-NeXT-7B across 11 multimodal benchmarks. Methods marked with <sup>\*</sup> are query-aware. The best average result within each category is highlighted in bold.

Implementation Details. All experiments are conducted on NVIDIA A6000 GPUs and evaluated with lmms-eval. For LLaVA-NeXT-7B, which processes high-resolution images through up to five crops, we retain the same token proportion from each crop following VisionZip. Cen-Prune is training-free and has no tunable hyperparameters. Based on our geometry analysis, we apply the correction after the multimodal projector in LLaVA and before the visual merger in Qwen2.5-VL. Full details are provided in the Appendix.

## 5.2 Main Results

Image-Language Understanding. Tab. 2 shows that Cen-Prune provides robust performance improvements for existing diversity-based pruners across architectures and compression levels. The overall performance Avg. denotes the benchmark-wise retention ratio relative to the vanilla model, averaged over evaluated benchmarks. On LLaVA-1.5-7B, Cen-Prune with DivPrune and ZOO-Prune reaches 93.4% and 93.6% performance with only 32 retained tokens, matching the query-aware MMTok (93.4%). Cen-Prune also improves CDPruner across all evaluated budgets. On LLaVA-NeXT-7B, it retains 96.1% performance while removing 88.9% of visual tokens. The gains extend to Qwen2.5-VL-7B with dynamic-resolution inputs: at 90% token reduction, Cen-Prune retains 88.9% performance, versus 85.1% for DivPrune, 83.2% for MMTok, and 79.5% for CDPruner (Tab. 3). Additional LLaVA-1.5-13B results and other budgets are provided in the Appendix.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>VQA Text</td><td>OCR-B</td><td>Avg.</td></tr><tr><td>Upper Bound (100%)</td><td>60.3</td><td>83.2</td><td>2322</td><td>86.2</td><td>87.4</td><td>77.5</td><td>83.8</td><td>100.0%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="8">Retain 20% Tokens</td></tr><tr><td>DivPrune (CVPR25)</td><td>58.5</td><td>78.3</td><td>2158</td><td>83.4</td><td>84.6</td><td>70.5</td><td>64.1</td><td>92.1%</td></tr><tr><td>+ Ours</td><td>58.2</td><td>79.6</td><td>2274</td><td>84.0</td><td>84.2</td><td>73.7</td><td>68.9</td><td>94.5%</td></tr><tr><td>CDPruner (NeurIPS25)</td><td>56.2</td><td>78.4</td><td>2110</td><td>80.2</td><td>84.2</td><td>63.6</td><td>50.5</td><td>87.1%</td></tr><tr><td>+ Ours</td><td>57.4</td><td>79.5</td><td>2227</td><td>81.2</td><td>84.5</td><td>70.0</td><td>59.4</td><td>91.3%</td></tr><tr><td>MMTok(ICLR26)</td><td>57.2</td><td>79.6</td><td>2216</td><td>82.3</td><td>83.7</td><td>71.6</td><td>59.3</td><td>91.5%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="8">Retain 10% Tokens</td></tr><tr><td>DivPrune (CVPR25)</td><td>55.1</td><td>74.8</td><td>2035</td><td>79.0</td><td>82.1</td><td>65.0</td><td>48.1</td><td>85.1%</td></tr><tr><td>+ Ours</td><td>55.6</td><td>75.9</td><td>2156</td><td>82.0</td><td>82.4</td><td>69.2</td><td>56.7</td><td>88.9%</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>53.0</td><td>75.9</td><td>1925</td><td>74.4</td><td>83.0</td><td>56.5</td><td>33.9</td><td>79.5%</td></tr><tr><td>+ Ours</td><td>54.0</td><td>74.5</td><td>2076</td><td>76.9</td><td>83.0</td><td>63.4</td><td>47.4</td><td>84.4%</td></tr><tr><td>MMTok(ICLR26)</td><td>53.9</td><td>74.7</td><td>1992</td><td>77.1</td><td>81.5</td><td>64.6</td><td>43.2</td><td>83.2%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="8">Retain 5% Tokens</td></tr><tr><td>DivPrune (CVPR25)</td><td>50.4</td><td>67.5</td><td>1853</td><td>69.5</td><td>78.8</td><td>57.3</td><td>33.3</td><td>75.6%</td></tr><tr><td>+ Ours</td><td>51.7</td><td>70.1</td><td>1944</td><td>75.0</td><td>79.2</td><td>63.4</td><td>42.4</td><td>80.5%</td></tr><tr><td>CDPruner (NeurIPS25)</td><td>48.5</td><td>70.1</td><td>1724</td><td>64.7</td><td>81.2</td><td>50.4</td><td>20.5</td><td>70.9%</td></tr><tr><td>+ Ours</td><td>48.6</td><td>66.8</td><td>1773</td><td>67.5</td><td>80.8</td><td>56.7</td><td>30.9</td><td>74.0%</td></tr><tr><td>MMTok(ICLR26)</td><td>48.4</td><td>68.3</td><td>1794</td><td>68.4</td><td>78.9</td><td>57.9</td><td>29.6</td><td>74.2%</td></tr></table>

Table 3: Performance comparison on Qwen2.5-VL-7B.

<table><tr><td>Method</td><td>MVBench MME-V EgoSchema SEED-V</td><td></td><td></td><td></td><td>Avg.</td></tr><tr><td>Upper Bound 64 × 169 Tokens</td><td>60.3</td><td>64.3</td><td>59.2</td><td>58.1</td><td>100.0%</td></tr><tr><td>LLaVA-Video-7B</td><td colspan="5">Retain 64 × 64 Tokens (↓ 62.1%)</td></tr><tr><td>DivPrune (CVPR25)</td><td>54.9</td><td>61.6</td><td>59.2</td><td>56.3</td><td>95.9%</td></tr><tr><td>+ Ours</td><td>56.4</td><td>62.1</td><td>60.6</td><td>58.0</td><td>98.1%</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>54.9</td><td>60.4</td><td>58.2</td><td>56.3</td><td>95.1%</td></tr><tr><td>+ Ours</td><td>55.6</td><td>60.3</td><td>60.8</td><td>56.6</td><td>96.6%</td></tr><tr><td>LLaVA-Video-7B</td><td colspan="5">Retain 64 × 32 Tokens (↓ 81.1%)</td></tr><tr><td>DivPrune (CVPR25)</td><td>54.1</td><td>58.9</td><td>57.2</td><td>55.2</td><td>93.2%</td></tr><tr><td>+ Ours</td><td>54.8</td><td>60.2</td><td>59.2</td><td>56.2</td><td>95.3%</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>53.4</td><td>57.5</td><td>56.2</td><td>52.9</td><td>91.0%</td></tr><tr><td>+ Ours</td><td>53.4</td><td>57.6</td><td>57.6</td><td>53.0</td><td>91.7%</td></tr><tr><td>LLaVA-Video-7B</td><td colspan="5">Retain 64 × 16 Tokens (↓ 90.5%)</td></tr><tr><td>DivPrune (CVPR25)</td><td>51.9</td><td>55.9</td><td>53.2</td><td>51.4</td><td>87.9%</td></tr><tr><td>+ Ours</td><td>52.9</td><td>58.1</td><td>58.0</td><td>53.7</td><td>92.2%</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>50.9</td><td>54.4</td><td>53.6</td><td>48.5</td><td>85.8%</td></tr><tr><td>+ Ours</td><td>50.3</td><td>54.8</td><td>55.4</td><td>48.4</td><td>86.4%</td></tr></table>

Table 4: Performance comparison on LLaVA-Video-7B.
<table><tr><td>Method</td><td>GQA</td><td>MME</td><td>POPE</td><td>VQAText</td><td>Avg.</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="5">Retain 32 Tokens (↓ 94.4%)</td></tr><tr><td>DivPrune (CVPR25)</td><td>55.0</td><td>1306</td><td>81.7</td><td>53.1</td><td>90.4%</td></tr><tr><td>+ centering</td><td>54.5</td><td>1263</td><td>80.6</td><td>50.9</td><td>88.2%</td></tr><tr><td>+ distinctiveness</td><td>55.4</td><td>1329</td><td>81.4</td><td>55.0</td><td>91.7%</td></tr><tr><td>Cen-Prune (Ours)</td><td>56.0</td><td>1335</td><td>82.8</td><td>55.3</td><td>92.6%</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>57.2</td><td>1359</td><td>87.2</td><td>52.9</td><td>93.7%</td></tr><tr><td>+ centering</td><td>56.9</td><td>1354</td><td>87.4</td><td>53.1</td><td>93.6%</td></tr><tr><td>+ distinctiveness</td><td>57.2</td><td>1359</td><td>85.4</td><td>54.9</td><td>94.0%</td></tr><tr><td>Cen-Prune (Ours)</td><td>57.5</td><td>1372</td><td>86.1</td><td>55.2</td><td>94.7%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="5">Retain 10% Tokens</td></tr><tr><td>DivPrune (CVPR25)</td><td>55.1</td><td>2035</td><td>79.0</td><td>65.0</td><td>88.6%</td></tr><tr><td>DivPrune (CVPR25)†</td><td>54.6</td><td>1951</td><td>78.3</td><td>61.3</td><td>86.1%</td></tr><tr><td>+ centering</td><td>54.1</td><td>2024</td><td>79.5</td><td>61.3</td><td>87.0%</td></tr><tr><td>+ distinctiveness</td><td>54.4</td><td>1997</td><td>78.3</td><td>63.1</td><td>87.1%</td></tr><tr><td>Cen-Prune (Ours)</td><td>55.6</td><td>2156</td><td>82.0</td><td>69.2</td><td>92.4%</td></tr></table>

Table 5: Ablation study on LLaVA-1.5-7B and Qwen2.5- VL-7B. Best and second-best results within each group are highlighted. † denotes pre-projector token pruning.

Video-Language Understanding. We further evaluate Cen-Prune on four video-understanding benchmarks using LLaVA-Video-7B. As shown in Tab. 4, Cen-Prune improves the overall performance of diversity-based pruning for video. At 90.5% token reduction, Cen-Prune retains 92.2% performance, exceeding DivPrune (87.9%) and CDPruner (85.8%).

## 5.3 Ablation Studies and Further Analysis

Component Ablation. Tab. 5 isolates the efects of mean centering and distinctiveness scoring. On LLaVA-1.5-7B, replacing the raw matrix with the centered matrix alone reduces performance, with the largest drop observed for DivPrune (greedy MMDP objective). CDPruner is less afected, consistent with the weaker sensitivity of its DPP objective to the sign of pairwise similarities. Distinctiveness scoring alone improves performance but does not address the concentrated geometry of raw features. On Qwen2.5-VL-7B, where DivPrune operates in a more concentrated embedding space, either centering or distinctiveness scoring alleviates degradation, while combining both yields the best performance.

Disentangling Diversity and Distinctiveness. To understand why the two components are complementary, Tab. 6 analyzes the selected subsets in terms of residual geometry and token importance. Although vanilla DivPrune has no explicit importance term, its raw similarity matrix selects tokens with higher distinctiveness and greater [CLS] attention mass than the centered matrix (0.440 vs. 0.330 and 0.266 vs. 0.193, respectively). This indicates that the raw geometry implicitly favors important tokens, whereas centering weakens this preference while broadening the residual geometry. Adding (g˜) to the centered selector restores the importance signal, achieving a [CLS] attention mass comparable to that of the raw selector with (g˜) (0.311 vs. 0.312). At the same time, it preserves more diverse residual directions, with a lower mean cosine similarity (0.306 vs. 0.414) and a higher fraction of negative pairs (13.4% vs. 7.1%). Thus, Cen-Prune separates the importance signal from the diversity geometry rather than relying on their implicit coupling.

<table><tr><td>Method</td><td>Res. Mean Cos. Res. Neg. (%) Mean Distinct. [CLS] Attn.</td><td></td><td></td><td></td></tr><tr><td>DivPrune (raw)</td><td>0.330</td><td>12.0</td><td>0.440</td><td>0.266</td></tr><tr><td>DivPrune (raw) w/ ğ</td><td>0.414</td><td>7.1</td><td>0.533</td><td>0.312</td></tr><tr><td>DivPrune (centered)</td><td>0.203</td><td>20.4</td><td>0.330</td><td>0.193</td></tr><tr><td>DivPrune (centered) w/ ğ</td><td>0.306</td><td>13.4</td><td>0.499</td><td>0.311</td></tr></table>

Table 6: Disentangling the efects of the similarity matrix and the distinctiveness preference on COCO val2017.
<table><tr><td>Method</td><td>#Tokens</td><td>Prefill (ms)</td><td>Latency (ms)</td><td>FLOPs (G)</td></tr><tr><td>LLaVA-1.5-7B</td><td>576</td><td>154.77</td><td>200.27</td><td>4431.21</td></tr><tr><td>DivPrune (CVPR25)</td><td>64</td><td>77.68</td><td>119.53</td><td>1048.38</td></tr><tr><td>+ Ours</td><td>64</td><td>77.82</td><td>119.85</td><td>1048.38</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>64</td><td>90.92</td><td>133.54</td><td>1056.75</td></tr><tr><td>+ Ours</td><td>64</td><td>91.85</td><td>134.63</td><td>1056.75</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>64</td><td>112.08</td><td>154.51</td><td>2595.93</td></tr><tr><td>+ Ours</td><td>64</td><td>112.32</td><td>154.77</td><td>2595.93</td></tr></table>

Table 7: Eficiency comparison on LLaVA-1.5-7B.

Eficiency Analysis. We finally measure computational efficiency on LLaVA-1.5-7B (Tab. 7). We randomly sample 500 examples from each of TextVQA, POPE, MME, and GQA, and average each measurement over three runs. Relative to the full 576-token baseline, retaining 64 tokens with Cen-Prune yields a 1.99× prefill speedup, a 1.67× end-toend speedup, and a 76.3% FLOPs reduction. Its cost remains close to those of DivPrune and CDPruner, indicating that the geometry correction adds little overhead while providing substantially greater eficiency than ZOO-Prune.

## 6 Conclusion

We examine the visual-token geometry underlying diversitybased pruning and find that the raw cosine-similarity matrix is strongly concentrated in the positive range. This provides limited resolution for residual diversity while implicitly favoring globally distinctive tokens, thereby entangling pairwise diversity with token-wise importance. Centering resolves residual diversity but weakens this useful preference. Accordingly, we introduce Cen-Prune, which measures subset diversity in the centered geometry while retaining raw-space distinctiveness for token selection. Cen-Prune is training-free, preserves the underlying selector, and adds little computational overhead. Experiments across LVLM architectures and image and video benchmarks demonstrate robust improvements. More broadly, our results suggest that diversity-based pruning benefits from modeling subset diversity and token-wise distinctiveness in complementary geometries.

## References

Alvar, S. R.; Singh, G.; Akbari, M.; and Zhang, Y. 2025. DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Dong, S.; Hu, J.; Zhang, M.; Yin, M.; Fu, Y.; and Qian, Q. 2026. MMTok: Multimodal Coverage Maximization for Eficient Inference ofVLMs. In The Fourteenth International Conference on Learning Representations.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; Uszkoreit, J.; and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Fang, Z.; Lyu, P.; Zhang, C.; Lu, G.; Yu, J.; and Pei, W. 2026. Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity. In The Fourteenth International Conference on Learning Representations.

Fu, C.; Chen, P.; Shen, Y.; Qin, Y.; Zhang, M.; Lin, X.; Yang, J.; Zheng, X.; Li, K.; Sun, X.; Wu, Y.; Ji, R.; Shan, C.; and He, R. 2025a. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc.

Fu, C.; Dai, Y.; Luo, Y.; Li, L.; Ren, S.; Zhang, R.; Wang, Z.; Zhou, C.; Shen, Y.; Zhang, M.; et al. 2025b. Videomme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 24108–24118.

Giraldo, L. G. S.; Rao, M.; and Principe, J. C. 2014. Measures of entropy from data using infinitely divisible kernels. IEEE Transactions on Information Theory, 61(1): 535–548.

Goyal, Y.; Khot, T.; Summers-Stay, D.; Batra, D.; and Parikh, D. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, 6904–6913.

Gurari, D.; Li, Q.; Stangl, A. J.; Guo, A.; Lin, C.; Grauman, K.; Luo, J.; and Bigham, J. P. 2018. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, 3608–3617.

Hudson, D. A.; and Manning, C. D. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question

answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6700–6709.

Kim, Y.; Zhang, Y.; Liu, H.; Jung, A.; Lee, S.; and Hong, S. 2026. ZOO-Prune: Training-Free Token Pruning via Zeroth-Order Gradient Estimation in Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Lee, H.; Jeong, H.; Kim, Y.; Choi, H.; Cho, H.; Kim, S. K.; and Lee, J. 2026. A More Word-like Image Tokenization for MLLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 17641–17650.

Li, B.; Ge, Y.; Ge, Y.; Wang, G.; Wang, R.; Zhang, R.; and Shan, Y. 2024a. SEED-Bench: Benchmarking Multimodal Large Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 13299–13308.

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; and Li, C. 2024b. LLaVA-OneVision: Easy Visual Task Transfer. arXiv:2408.03326.

Li, K.; Wang, Y.; He, Y.; Li, Y.; Wang, Y.; Liu, Y.; Wang, Z.; Xu, J.; Chen, G.; Luo, P.; et al. 2024c. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22195–22206.

Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, X.; and Wen, J.-R. 2023. Evaluating object hallucination in large visionlanguage models. In Proceedings of the 2023 conference on empirical methods in natural language processing, 292–305.

Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollár, P.; and Zitnick, C. L. 2014. Microsoft COCO: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, 740– 755. Springer.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024a. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning. In Oh, A.; Naumann, T.; Globerson, A.; Saenko, K.; Hardt, M.; and Levine, S., eds., Advances in Neural Information Processing Systems, volume 36, 34892–34916. Curran Associates, Inc.

Liu, Y.; Duan, H.; Zhang, Y.; Li, B.; Zhang, S.; Zhao, W.; Yuan, Y.; Wang, J.; He, C.; Liu, Z.; et al. 2024b. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, 216–233. Springer.

Liu, Y.; Li, Z.; Huang, M.; Yang, B.; Yu, W.; Li, C.; Yin, X.- C.; Liu, C.-L.; Jin, L.; and Bai, X. 2024c. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12): 220102.

Lu, P.; Mishra, S.; Xia, T.; Qiu, L.; Chang, K.-W.; Zhu, S.-C.; Tafjord, O.; Clark, P.; and Kalyan, A. 2022. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In Koyejo, S.; Mohamed, S.; Agarwal,

A.; Belgrave, D.; Cho, K.; and Oh, A., eds., Advances in Neural Information Processing Systems, volume 35, 2507–2521. Curran Associates, Inc.

Mangalam, K.; Akshulakov, R.; and Malik, J. 2023. EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding. In Oh, A.; Naumann, T.; Globerson, A.; Saenko, K.; Hardt, M.; and Levine, S., eds., Advances in Neural Information Processing Systems, volume 36, 46212–46244. Curran Associates, Inc.

Roy, O.; and Vetterli, M. 2007. The efective rank: A measure of efective dimensionality. In 2007 15th European signal processing conference, 606–610. IEEE.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8317–8326.

Yang, S.; Chen, Y.; Tian, Z.; Wang, C.; Li, J.; Yu, B.; and Jia, J. 2025. VisionZip: Longer is Better but Not Necessary in Vision Language Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025.

Yue, X.; Ni, Y.; Zhang, K.; Zheng, T.; Liu, R.; Zhang, G.; Stevens, S.; Jiang, D.; Ren, W.; Sun, Y.; et al. 2024. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9556–9567.

Zhang, Q.; Cheng, A.; Lu, M.; Zhang, R.; Zhuo, Z.; Cao, J.; Guo, S.; She, Q.; and Zhang, S. 2025a. Beyond Text-Visual Attention: Exploiting Visual Cues for Efective Token Pruning in VLMs. In IEEE/CVFInternational Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025.

Zhang, Q.; Liu, M.; Li, L.; Lu, M.; Zhang, Y.; Pan, J.; She, Q.; and Zhang, S. 2025b. Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in Neural Information Processing Systems, volume 38, 25438–25468. Curran Associates, Inc.

Zhang, Y.; Fan, C.-K.; Ma, J.; Zheng, W.; Huang, T.; Cheng, K.; Gudovskiy, D. A.; Okuno, T.; Nakata, Y.; Keutzer, K.; et al. 2025c. SparseVLM: Visual Token Sparsification for Eficient Vision-Language Model Inference. In International Conference on Machine Learning, 74840–74857. PMLR.

Zhang, Y.; Wu, J.; Li, W.; Li, B.; Ma, Z.; Liu, Z.; and Li, C. 2025d. LLaVA-Video: Video Instruction Tuning With Synthetic Data. Trans. Mach. Learn. Res.

Zhu, J.; Wang, W.; Chen, Z.; Liu, Z.; Ye, S.; Gu, L.; Tian, H.; Duan, Y.; Su, W.; Shao, J.; et al. 2025. InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv:2504.10479.

Zou, X.; Lu, D.; Wang, Y.; Yan, Y.; Lyu, Y.; Zheng, X.; Zhang, L.; and Hu, X. 2025. Don't Just Chase “Highlighted Tokens”in MLLMs: Revisiting Visual Holistic Context Retention. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in

# Centering before Pruning: Lightweight Geometry Correction for Diversity-Based Visual Token Pruning in LVLMs

Supplementary Material

## Appendix

This supplementary material provides additional empirical analyses and benchmark evaluation results, organized as follows:

• In Sec. A, we provide the details of our empirical analysis.

• In Sec. B, we present additional implementation details for the benchmark evaluations.

• In Sec. C, we also report further results, including performance comparisons on larger-scale LVLMs, additional ablation studies and further eficiency analysis.

• In Sec. D, we present the qualitative visualizations of DivPrune and our Cen-Prune method.

## A Details of Empirical Analysis

## A.1 Data and Protocol for Geometry Analysis

All our analyses in Sec. 4 are based on the same fixed set of 1,000 probe images sampled from COCO val2017 (Lin et al. 2014) with seed 0. The resulting probe list is frozen and shared across all backbones and feature spaces. For each image, we extract the complete visual-token matrix $\boldsymbol { X } \ : =$ $[ { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times D }$ at the feature space under study. Unless otherwise specified, each statistic is computed per image and summarized by its unweighted mean and standard deviation across the probe set. Pairwise statistics exclude self-pairs and count each unordered pair once.

For LLaVA-1.5-7B, we examine the exact input and output of the multimodal projector. For Qwen2.5-VL-7B-Instruct, we examine the encoder representation consumed by the pruning implementation and the post-merger representation passed to the language model. Following the feature construction used by MMTok (Dong et al. 2026), the former averages each merger-aligned group of four encoder patch features and restores the resulting tokens to raster order. Because Qwen processes images at dynamic resolution, its token count varies across images.

## A.2 Quantities and Metrics

Let $\hat { \pmb x } _ { i } = { \pmb x } _ { i } / \| { \pmb x } _ { i } \| _ { 2 }$ and $S = \hat { X } \hat { X } ^ { \top }$

Mean of-diagonal cosine. We define

$$
\bar { s } = \frac { 1 } { N ( N - 1 ) } \sum _ { i \neq j } S _ { i j } ,
$$

which measures the average directional agreement between distinct token pairs. This metric has been used in Tab. 1.

Negative-pair fraction. We denote by $\rho _ { - }$ the fraction of distinct token pairs satisfying $S _ { i j } < 0$ . This metric has been used in Tab. 1.

Efective rank. Let $\lambda _ { k }$ be the eigenvalues of the cosine Gram matrix S and $p _ { k } = \lambda _ { k } / \sum _ { \ell } \lambda _ { \ell }$ . We define

$$
\operatorname { e r a n k } ( S ) = \exp \left( - \sum _ { k } p _ { k } \log p _ { k } \right) ,
$$

which has been studied in previous works (Roy and Vetterli 2007; Giraldo, Rao, and Principe 2014). Because $\pmb { S }$ is constructed from row-normalized features, this quantity measures directional spread independently of token magnitude. It equals one when all normalized tokens lie in a single onedimensional subspace. Unless stated otherwise, we report the efective rank of the uncentered cosine Gram matrix, which characterizes the geometry available before applying Cen-Prune. We do not assume that efective rank must increase monotonically under centering.

## A.3 Where the Correction Applies

Cen-Prune is defined for a generic visual-token representation, but the appropriate feature space depends on the backbone. Table A1 compares the geometries immediately before and after the corresponding cross-modal projection stage.

In LLaVA-1.5, the multimodal projector increases concentration: the mean cosine rises from 0.309 to 0.366, the negative-pair fraction falls from 3.57% to 2.27%, and efective rank decreases from 21.79 to 13.20. Thus, although the pre-projector representation is already positively biased, the post-projector geometry provides less directional contrast.

Qwen2.5-VL exhibits the opposite progression. Its mergeraligned encoder representation is nearly collinear, with a mean cosine of 0.993, no observed negative pairs, and an efective rank of 1.06. The visual merger substantially disperses this geometry: the mean cosine decreases to 0.300, while efective rank increases to 31.31.

We therefore apply Cen-Prune to the post-projector representation in LLaVA-1.5 and to the merger-aligned encoder representation in Qwen2.5-VL. These are the most concentrated feature spaces consumed by the corresponding pruning pipelines.

## A.4 Coordinate Conditioning and Score Normalization

Centering is the principal geometric correction in Cen-Prune. When variance is reasonably balanced across feature dimensions, as in the LLaVA projector representation, centering can be applied to the raw visual-token matrix $V = [ \bar { \pmb { v } _ { 1 } } , \ldots , \pmb { v } _ { N } ] ^ { \top }$ . In some backbones, however, a small number of dimensions account for most of the feature variance and consequently dominate the cosine geometry. Although centering still removes the shared component in such cases, it does not prevent the residual similarities from being governed by these high-variance dimensions.

<table><tr><td>Backbone</td><td>Space</td><td>N</td><td>D</td><td>s</td><td> $\rho _ { - } \ ( { ^ { 9 } } \circ )$ </td><td> $\mathrm { e r a n k } ( S )$ </td></tr><tr><td>LLaVA-1.5</td><td>Pre-projector</td><td>576</td><td>1024</td><td> $0 . 3 0 9 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td> $3 . 5 7 { \pm } 2 . 3 1 $ </td><td> $2 1 . 7 9 2 5 . 9 6$ </td></tr><tr><td>LLaVA-1.5</td><td>Post-projector</td><td>576</td><td>4096</td><td> $0 . 3 6 6 { \pm } 0 . 0 2 5$ </td><td> $2 . 2 7 { \pm } 0 . 8 6$ </td><td> $1 3 . 2 0 { \pm } 2 . 3 8 $ </td></tr><tr><td>Qwen2.5-VL</td><td>Pre-merger (encoder average)</td><td>345 [88, 529]</td><td>1280</td><td> $0 . 9 9 3 { \pm } 0 . 0 0 3$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $1 . 0 6 0 { \scriptstyle \pm 0 . 0 1 6 }$ </td></tr><tr><td>Qwen2.5-VL</td><td>Post-merger</td><td>345 [88, 529]</td><td>3584</td><td> $0 . 3 0 0 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 4 8 { \pm } 0 . 2 9$ </td><td> $3 1 . 3 1 { \pm } 5 . 5 7$ </td></tr></table>

Table A1: Geometry of the feature spaces examined in LLaVA-1.5-7B and Qwen2.5-VL-7B-Instruct. Values are image-level mean ± standard deviation over the fixed probe set. For Qwen, N is reported as median [minimum, maximum]. Efective rank is computed from the uncentered cosine Gram matrix.

Qwen’s merger-aligned encoder representation is an example of this variance imbalnace. After plain centering, 35.7% of token pairs have cosine similarity below −0.9, while 38.9% have similarity above 0.9; thus, 74.6% satisfy $| \cos ( { \bf x } _ { i } , { \bf x } _ { j } ) | > 0 . 9$ . This concentration near the two endpoints indicates that centering along does not yield a well-conditioned residual geometry. As shown in Fig. A1a, the highest-variance dimension accounts for 98.9% of the centered variance on average, and the ten highest-variance dimensions collectively account for 99.6%. By comparison, the corresponding proportions in the LLaVA projector representation are only 0.28% and 1.86%, respectively. Consequently, inner products between Qwen’s centered tokens are determined largely by the sign of a single coordinate, and subsequent ℓ -normalization pushes their cosine similarities toward either −1 or +1.

We therefore apply per-dimension variance scaling before performing the same image-wise centering operation. For each image, let

$$
{ \cal Y } = { \cal V } \mathrm { d i a g } ( { \pmb \sigma } ) ^ { - 1 } , \quad { \sigma } _ { d } = \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( v _ { i d } - \mu _ { d } ) ^ { 2 } + \epsilon \right) ^ { 1 / 2 } ,\tag{9}
$$

where $\pmb \mu = N ^ { - 1 } \textstyle \sum _ { i } \pmb v _ { i }$ . Both µ and σ are computed from the tokens of the current image; no statistics are shared across images or estimated from external data. Cen-Prune then centers the variance-scaled working coordinates Y:

$$
\bar { \pmb { Y } } = \pmb { Y } - \mathbf { 1 } \pmb { \mu } _ { Y } ^ { \top } = ( \pmb { V } - \mathbf { 1 } \pmb { \mu } ^ { \top } ) \operatorname { d i a g } ( \pmb { \sigma } ) ^ { - 1 } , \quad \pmb { \mu } _ { Y } = \frac { 1 } { N } \sum _ { i } \pmb { y } _ { i } .\tag{10}
$$

Thus, per-dimension scaling does not replace mean centering; it defines the coordinate system in which centering is performed. The uncentered working coordinates Y are retained for the distinctiveness branch, whereas only their mean-removed form Y<sup>¯</sup> is used to construct the pairwise diversity geometry. If σ is constant across dimensions, this construction difers from plain centering only by a global scale and therefore produces the same cosine-similarity matrix.

This standardized centering reduces the fraction of Qwen token pairs satisfying $| \cos ( { \bf x } _ { i } , { \bf x } _ { j } ) | > 0 . 9$ from 74.6% to 0.008% and increases the normalized-spectrum efective rank from 1.73 to 53.65 (Fig. A1b). By comparison, variance is substantially more balanced across the feature dimensions of LLaVA: standardization changes the corresponding fraction only from 7.38% to 6.69%, although it still moderately increases the efective rank. These measurements motivate plain centering for LLaVA and standardized centering for Qwen. Both variants preserve the principal operation of Cen-Prune: removing the image-wise mean before constructing the pairwise diversity matrix.

The raw distinctiveness scores must also be normalized before they can serve as nonnegative multiplicative weights. Given the scores $\{ g _ { i } \} _ { i = 1 } ^ { N }$ , we consider two mappings applied independently to each image. Min–max normalization gives

$$
\tilde { g } _ { i } ^ { \mathrm { m m } } = \frac { g _ { i } - g _ { \mathrm { m i n } } } { g _ { \mathrm { m a x } } - g _ { \mathrm { m i n } } + \epsilon } ,\tag{11}
$$

which preserves the score ordering and maps the scores to the unit interval. Its relative scale is determined by the two extrema within each image. We use this mapping for LLaVA, whose token count is fixed at $N = 5 7 6$ . For Qwen, dynamic image resolution changes the number and composition of the visual tokens and, consequently, the extrema observed in each image. We therefore apply z-score normalization followed by a sigmoid:

$$
\tilde { g } _ { i } ^ { \mathrm { z n } } = \mathrm { s i g m o i d } \left( \frac { g _ { i } - \mu _ { g } } { \sigma _ { g } + \epsilon } \right) ,\tag{12}
$$

where $\mu _ { g }$ and $\sigma _ { g }$ are computed over the tokens of the same image. This mapping remains monotonic, compresses extreme values, and expresses each score relative to the imagewise score distribution rather than directly to its extrema. Thus, neither mapping changes the within-image token ordering, but each controls the relative strength of the multiplicative distinctiveness weighting diferently. We use min– max normalization for LLaVA and sigmoid-transformed z scores for Qwen. Each choice is fixed for its respective backbone and shared across all datasets and retention budgets, with no normalization statistics transferred across images.

## B Details of Experimental Setup B.1 Model Settings

We evaluate our visual token pruning method on representative open-source LVLMs covering image-only, highresolution image, and video understanding settings. Following prior visual-token pruning studies, we keep the original model weights frozen and apply token pruning only during inference, so that any performance change reflects the efect of reducing visual tokens rather than additional training or fine-tuning.

LLaVA-1.5 (Liu et al. 2023). We use LLaVA-1.5 as the standard image-understanding backbone. LLaVA-1.5 follows the widely used LLaVA architecture, consisting of a CLIP

![](images/205f22c4261c4dfcfa2ca010da778770cfdde2f0b19aecc31996672ece368bd3.jpg)

![](images/e280ae30bf71384041a2ee02f0adf0615bd1e3ce6ba5919a725fb865d752154e.jpg)  
Figure A1: Standardized centering corrects Qwen’s variance-dominated residual geometry. (a) Mean cumulative fraction of centered feature variance, with dimensions ranked by variance independently for each image. The highest-variance dimension accounts for 98.9% of the total centered variance in Qwen’s merger-aligned encoder representation, compared with only 0.28% in the LLaVA projector representation. Shading denotes one standard deviation across images. (b) Pair-pooled distributions of of-diagonal cosine similarities for Qwen after plain and standardized centering. Standardized centering largely eliminates the endpoint concentration that remains after plain centering. The curves are normalized independently and linearly interpolated from bins of width 0.01 for visualization.

ViT-L/336px vision encoder (?), a two-layer MLP visionlanguage projector, and a Vicuna language model. Compared with the original LLaVA, LLaVA-1.5 improves the visual instruction tuning recipe by using stronger image resolution, an MLP connector, and additional academic-task-oriented VQA data. In our experiments, this model serves as the main low-resolution image baseline, where each image is encoded into a fixed number of visual tokens. This setting is commonly adopted in visual token pruning literature because it provides a controlled testbed for evaluating the accuracyeficiency trade-of under diferent retained-token budgets.

LLaVA-1.6 (LLaVA-NeXT) (Liu et al. 2024a). We further evaluate on LLaVA-1.6, also known as LLaVA-NeXT, to test pruning under high-resolution image inputs. LLaVA-NeXT extends LLaVA-style visual instruction tuning with an AnyRes image representation, where an image can be split into multiple visual grids before being encoded. This design preserves more fine-grained visual details than the fixed-resolution setting, but also substantially increases the number ofvisual tokens passed to the language model. Therefore, LLaVA-NeXT provides a more challenging and practically important setting for visual token pruning: the model has stronger visual perception ability, while its longer visua prefix leads to higher prefilling cost and memory usage.

LLaVA-Video (Zhang et al. 2025d) For video understanding, we use lmms-lab/LLaVA-Video-7B-Qwen2. This model is built on the LLaVA-Video framework, which adapts LLaVA-style multimodal instruction tuning to video inputs. The released checkpoint uses a SigLIP/SO400M vision encoder (?) with a Qwen2 (?) language backbone and is initialized from the LLaVA-OneVision (Li et al. 2024b) image model before being trained on a mixture of single-image, multi-image, and video instruction data. In our experiments, LLaVA-Video-7B-Qwen2 is used to evaluate whether visual token pruning remains efective when the visual input consists of multiple sampled video frames rather than a single image. This setting is important because video models naturally introduce many more visual tokens, making inference cost more severe.

Qwen2.5-VL (Bai et al. 2025). We also evaluate Qwen2.5- VL-7B as a stronger recent MLLM backbone. Qwen2.5-VL introduces a native dynamic-resolution vision encoder, window attention in the vision transformer, and improved spatialtemporal modeling for images and videos. Unlike LLaVAstyle models that typically rely on a fixed CLIP-like image encoder, Qwen2.5-VL dynamically converts visual inputs of diferent sizes into variable-length visual tokens. This makes it a useful test case for evaluating whether our pruning strategy generalizes beyond the LLaVA family. In this work, we report results on the Qwen2.5-VL-7B-Instruct, aligning its model scale with the other evaluated backbones while isolating the efect of architectural diferences.

## B.2 Datasets

Image Benchmarks. Following prior studies (Kim et al. 2026; Fang et al. 2026) on eficient multimodal large language models, we evaluate image understanding on 11 widely used benchmarks covering general VQA, compositional reasoning, OCR-centric perception, hallucination evaluation, scientific reasoning, real-world assistive VQA, and expertlevel multimodal understanding. We also list the specific dataset partitions we used with lmms-eval<sup>1</sup> as below.

• GQA (Hudson and Manning 2019) evaluates real-world visual reasoning and compositional question answering. It requires models to understand objects, attributes, relations, and scene structures, making it suitable for testing fine-grained visual reasoning.

• MMBench (Liu et al. 2024b) is a multiple-choice benchmark for evaluating fine-grained multimodal abilities. It covers broad perception and reasoning dimensions, providing an objective evaluation protocol for LVLMs. We use mmbench\_en\_dev for evaluations.

• MME (Fu et al. 2025a) provides a comprehensive evaluation of LVLMs from both perception and cognition perspectives. Its perception tasks include object existence, color, position, counting, and OCR, while its cognition tasks include commonsense reasoning, numerical calculation, text translation, and code reasoning.

• POPE (Li et al. 2023) evaluates object hallucination in LVLMs through binary object-presence questions. It tests whether a model incorrectly asserts the existence of objects that are absent from the image.

• ScienceQA (Lu et al. 2022) assesses multimodal science question answering. Models need to combine visual evidence, textual context, and scientific knowledge to answer multiple-choice science questions. We use scienceqa\_img for evaluations.

• VQAv2 (Goyal et al. 2017) evaluates general visual question answering on natural images. Its balanced questionanswer design reduces language priors and tests robust image-question understanding. We use vqav2\_test for evaluations.

• TextVQA (Singh et al. 2019) focuses on text-rich visual question answering. It requires models to read scene text in images and reason over the recognized text to answer questions. We use textvqa\_val for evaluations and ocr’s candidates are activated.

• OCRBench (Liu et al. 2024c) provides a targeted evaluation of OCR-related capabilities in large multimodal models. It covers text recognition, scene-text VQA, documentoriented VQA, key information extraction, and handwritten mathematical expression recognition.

• Vizwiz (Gurari et al. 2018) evaluates VQA in a realworld assistive setting. The images are captured by blind users and often contain blur, poor framing, ambiguity, or unanswerable questions, making the benchmark challenging for practical visual understanding. We use vizwiz\_vqa\_val for evaluations.

• SeedBench-Image (Li et al. 2024a) evaluates imagebased multimodal comprehension with objective multiple-choice questions. It covers diverse capability dimensions and avoids subjective judge-based scoring.

• MMMU (Yue et al. 2024) evaluates expert-level multimodal understanding and reasoning across multiple disciplines. It requires models to integrate visual perception with domain-specific knowledge and deliberate reasoning. We use mmmu\_val for evaluations.

Video Benchmarks. We further evaluate LLaVA-Video-7B on four representative video-understanding benchmarks that cover temporal reasoning, long-form video understanding, egocentric perception, and objective video-based multimodal evaluation.

• MVBench (Li et al. 2024c) evaluates temporal understanding across diverse video tasks. Many tasks require dynamic information beyond a single frame, such as action recognition, temporal ordering, and event reasoning.

• Video-MME (Fu et al. 2025b) is a comprehensive videounderstanding benchmark covering diverse video domains, durations, and multimodal cues. It evaluates both short- and long-context video comprehension and is suitable for assessing general video LVLMs.

• EgoSchema (Mangalam, Akshulakov, and Malik 2023) focuses on long-form egocentric video question answering. Models answer multiple-choice questions based on three-minute first-person video clips, requiring longrange temporal reasoning over daily human activities. We use egoschema\_subset for comparison.

• SeedBench-Video (Li et al. 2024a) evaluates video-based multimodal understanding through objective multiplechoice questions. It complements MVBench and Video-MME by providing a structured evaluation of video comprehension abilities.

These benchmarks jointly test whether visual token pruning preserves performance across general image understanding, compositional reasoning, OCR, hallucination robustness, scientific and expert-level reasoning, real-world VQA, and temporal video comprehension.

## B.3 Implementation Details

For performance evaluation, all experiments are conducted on four NVIDIA A6000 GPUs with a batch size of 1, while eficiency is measured on a single A6000 GPU. The implementation is based on Python 3.10, PyTorch 2.1.2, and CUDA 12.1. Following the empirical analysis in Sec. A, Cen-Prune applies the geometry correction after the multimodal projector in the LLaVA-series models and before the visual merger in Qwen2.5-VL. Before activating the diversity-based pruning mechanism, we compute the centered similarity matrix and the distinctiveness score, which introduces negligible computational overhead.

For the pruning ratios, we use 77.8%, 88.9%, and 94.4% for both LLaVA-1.5 and LLaVA-NeXT, and 62.1%, 81.1%, and 90.5% for LLaVA-Video. For LLaVA-NeXT, which processes high-resolution images using up to five crops (5 × 576 = 2880 visual tokens), we follow the implementation of VisionZIP (Yang et al. 2025) and retain the same token proportion from each crop. For example, when an image is divided into three crops, the total number of visual tokens is 3 × 576 = 1728. To achieve a pruning ratio of 94.4%, we retain 32 tokens from each crop, resulting in 96 retained tokens in total. For Qwen2.5-VL-7B, which supports

dynamic-resolution inputs, we follow MMTok (Dong et al.   
2026) and evaluate pruning ratios of 80%, 90%, and 95%.

## B.4 Reproduction of Baselines and Integration

Baseline Reproducibility. We reproduce all baseline methods in a unified environment based on their original papers and oficial code repositories. For methods that do not report the 94.4% pruning-ratio setting on LLaVA-1.5, we make minimal adaptations following their original configuration rules. We summarize the reproduction and integration details below.

• VisionZIP (Yang et al. 2025). For the setting with 32 retained visual tokens on LLaVA-1.5, we proportionally adapt its token allocation and retain 27 dominant tokens and 5 contextual tokens.

• HoloV (Zou et al. 2025). For the setting with 32 retained visual tokens on LLaVA-1.5, we follow its implementation and set $n u m _ { c r o p } = [ 1 0 2 4 / N ]$ , where N denotes the number of retained visual tokens.

• PruneSID (Fang et al. 2026). For the setting with 32 retained visual tokens on LLaVA-1.5, we follow its default configuration and set the number of token groups to K = $\frac { N } { 4 }$ , where N denotes the number of retained visual tokens.

• VisPruner (Zhang et al. 2025a). Following its default configuration, we set the important-token ratio to r = 0.5 for all experiments.

• DivPrune (Alvar et al. 2025). Following its default configuration, we apply token selection before the LLM. For fair comparison, visual token features are processed in Float32 during pruning.

• ZOO-Prune (Kim et al. 2026). Following its default configuration, we fix the perturbation hyperparameters to m = 64 and $h = 0 . 0 1$ for all experiments. For fair comparison, visual token features are processed in Float32 during pruning.

• CDPruner (Zhang et al. 2025b). Following its default configuration, we apply token selection before the LLM.

• MMTok (Dong et al. 2026). Following its default configuration, we set $\tau _ { t } = 0 . 0 2 , \tau _ { v } = 0 . 2$ , and α = 0.5 for all experiments on the LLaVA-series models. For Qwen2.5- VL-7B, we follow its implementation and reduce $\tau _ { t }$ to 0.01.

Integration with Diversity-Based Pruners. In Sec. 4.3, we describe how Cen-Prune can be integrated into diversitybased pruning methods. We provide the implementation details below.

• DivPrune + Ours. We replace the original selection geometry with the centered cosine similarity matrix. After DivPrune’s original initialization, we multiply the minimum-distance score by our distinctiveness score at each subsequent step, and then select the next token by computing argmax(dist\_scores × distinctiveness).

• ZOO-Prune + Ours. Similar to the integration with DivPrune, we replace the original selection geometry with the centered cosine similarity matrix and multiply ZOO-Prune’s sensitivity score by our distinctiveness score. The two scores are normalized separately before multiplication.

• CDPruner + Ours. Similar to the integration with ZOO-Prune, we multiply the original instruction-relevance score by our distinctiveness score, with each score normalized separately. The underlying kernel is replaced with the centered cosine similarity matrix.

## C Further Quantitative Analysis and Results C.1 Additional Quantitative Results

Complete Results on LLaVA-NeXT-7B. In the main paper, we reported the compression results of LLaVA-NeXT-7B when retaining up to 320 visual tokens. Tab. A2 shows the complete results under all budget settings on LLaVA-NeXT-7B. Across diferent compression ratios, Cen-Prune’s overall relative performance retention improved by 0.7% to 2.0% comparing to those diversity-based pruners.

Diferent Protocol for Comparing with HoloV. To ensure fairness in the comparison, on the high-resolution input architecture LLaVA-NeXT, we follow previous studies (Yang et al. 2025; Zhang et al. 2025a,b; Fang et al. 2026; Dong et al. 2026) and perform pruning before removing padding in AnyRes branch. This means that each crop contains 576 tokens, up to a maximum of 2880 tokens, and is compressed at a fixed ratio per crop based on the pruning rate (77.8% / 88.9% / 94.4%). For HoloV (Zou et al. 2025), they apply visual token reduction after removing padding regions. We provide additional results in Tab.A3 for comparing our Cen-Prune based on DivPrune (Alvar et al. 2025) against HoloV with fixed 320 retained tokens. Under this setting, Cen-Prune still maintains a robust overall performance gain based on DivPrune, and also surpasses HoloV.

Additional Results on Larger-Scale Models. In the main paper, we report comparison results on 7B-scale models. To further validate the efectiveness of Cen-Prune across diferent model scales, we additionally compare it with existing baselines on LLaVA-NeXT-13B and LLaVA-1.5-13B. As shown in Tab. A4 and Tab. A5, Cen-Prune can still deliver a robust overall performance improvement against diversitybased pruners.

## C.2 Additional Ablation Studies

Ablation Studies on the Centering Strategy. We examine the efectiveness of mean-direction removal as an alternative centering strategy. Let X denote the working visual features. We remove the mean direction as follows:

$$
\boldsymbol X ^ { \perp } = \boldsymbol X \left( \boldsymbol I - \hat { \mu } \hat { \boldsymbol \mu } ^ { \top } \right) , \qquad \hat { \boldsymbol \mu } = \frac { \boldsymbol \mu } { \| \boldsymbol \mu \| _ { 2 } } ,
$$

where $\pmb { \mu }$ is the mean feature vector computed over visual tokens. The first group in Tab. A6 shows that completely projecting out the mean direction can discard informative tokenspecific variation along that direction. In contrast, mean centering removes only the shared ofset while preserving relative variation among visual tokens.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>VQAText</td><td>OCR-B</td><td>VizWiz</td><td>SEED-I</td><td>MMMU</td><td>Avg.</td></tr><tr><td>Upper Bound (2880 Tokens)</td><td>64.2</td><td>67.9</td><td>1520</td><td>86.4</td><td>70.2</td><td>61.3</td><td>51.3</td><td>59.7</td><td>69.6</td><td>37.2</td><td>100.0%</td></tr><tr><td>LLaVA-NeXT-7B</td><td>Retain Up to 640 Tokens (↓ 77.8%)</td></tr><tr><td>VisionZip(CVPR25) 61.2</td><td>65.0</td><td>1474</td><td>86.1</td><td>68.0</td><td>60.3</td><td>47.8</td><td>61.0</td><td>66.7</td><td>36.6</td><td></td><td>97.2%</td></tr><tr><td>VisPruner (ICCV25)</td><td>61.6</td><td>64.9</td><td>1472</td><td>86.0</td><td>67.8</td><td>59.8</td><td>45.1</td><td>60.4</td><td>66.4</td><td>36.1</td><td>96.3%</td></tr><tr><td>PruneSID (ICLR26)</td><td>61.6</td><td>63.9</td><td>1473</td><td>86.2</td><td>68.1</td><td>55.5</td><td>34.3</td><td>59.2</td><td>67.0</td><td>36.4</td><td>93.4%</td></tr><tr><td>DivPrune (CVPR25)</td><td>62.0</td><td>65.4</td><td>1496</td><td>87.1</td><td>67.8</td><td>57.0</td><td>40.0</td><td>59.3</td><td>67.6</td><td>36.1</td><td>95.3%</td></tr><tr><td>+ Ours</td><td>62.3</td><td>66.0</td><td>1528</td><td>87.1</td><td>68.1</td><td>58.2</td><td>44.1</td><td>59.6</td><td>68.2</td><td>36.3</td><td>96.9%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>61.8</td><td>64.9</td><td>1488</td><td>86.8</td><td>68.1</td><td>58.1</td><td>43.8</td><td>59.3</td><td>67.6</td><td>36.9</td><td>96.3%</td></tr><tr><td>+ Ours</td><td>62.3</td><td>65.8</td><td>1495</td><td>86.9</td><td>67.8</td><td>59.2</td><td>47.0</td><td>59.5</td><td>68.2</td><td>35.1</td><td>97.0%</td></tr><tr><td>MMTok(ICLR26)</td><td>62.0</td><td>65.4</td><td>1507</td><td>87.0</td><td>68.2</td><td>59.1</td><td>44.8</td><td>59.2</td><td>67.6</td><td>36.8</td><td>96.9%</td></tr><tr><td>CDPruner (NeurIPS25)</td><td>62.7</td><td>65.9</td><td>1495</td><td>87.4</td><td>68.0</td><td>58.8</td><td>44.7</td><td>59.0</td><td>69.0</td><td>36.3</td><td>97.0%</td></tr><tr><td>+ Ours</td><td>62.4</td><td>66.1</td><td>1508</td><td>87.2</td><td>67.5</td><td>60.3</td><td>48.4</td><td>59.4</td><td>68.9</td><td>36.9</td><td>98.1%</td></tr><tr><td>LLaVA-NeXT-7B Retain Up to 320 Tokens (↓ 88.9%)</td><td colspan="14"></td></tr><tr><td>VisionZip (CVPR25)</td><td>58.9</td><td>63.4</td><td>1401</td><td>82.1</td><td>67.3</td><td>58.8</td><td>39.2</td><td>59.9</td><td>63.5</td><td>36.7</td><td>93.0%</td></tr><tr><td>VisPruner(ICCV25)</td><td>58.6</td><td>63.7</td><td>1404</td><td>81.1</td><td>67.9</td><td>58.6</td><td>37.5</td><td>61.1</td><td>62.8</td><td>35.7</td><td>92.5%</td></tr><tr><td>PruneSID (ICLR26)</td><td>60.6</td><td>63.0</td><td>1468</td><td>85.0</td><td>67.8</td><td>54.3</td><td>32.4</td><td>58.7</td><td>65.8</td><td>36.9</td><td>92.2%</td></tr><tr><td>DivPrune (CVPR25)</td><td>61.1</td><td>63.7</td><td>1375</td><td>84.6</td><td>67.8</td><td>56.3</td><td>34.2</td><td>59.7</td><td>65.7</td><td>36.7</td><td>92.5%</td></tr><tr><td>+ Ours</td><td>60.9</td><td>64.6</td><td>1480</td><td>85.1</td><td>68.0</td><td>57.6</td><td>39.3</td><td>59.2</td><td>66.4</td><td>36.2</td><td>94.5%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>61.0</td><td>64.1</td><td>1413</td><td>85.3</td><td>67.4</td><td>57.1</td><td>36.7</td><td>58.9</td><td>65.8</td><td>37.2</td><td>93.5%</td></tr><tr><td>+ Ours</td><td>60.8</td><td>64.6</td><td>1443</td><td>85.4</td><td>67.8</td><td>58.0</td><td>39.6</td><td>58.9</td><td>66.2</td><td>36.3</td><td>94.3%</td></tr><tr><td>MMTok(ICLR26)</td><td>61.1</td><td>64.1</td><td>1471</td><td>85.8</td><td>67.6</td><td>56.9</td><td>34.6</td><td></td><td>66.2</td><td></td><td></td></tr><tr><td>CDPruner (NeurIPS25)</td><td>61.6</td><td>64.2</td><td>1485</td><td>87.3</td><td>66.8</td><td>57.0</td><td>38.8</td><td>59.1 58.4</td><td>67.3</td><td>38.0</td><td>93.8%</td></tr><tr><td>+ Ours</td><td>61.4</td><td>65.3</td><td>1490</td><td>86.8</td><td>67.2</td><td>58.9</td><td>43.3</td><td>58.8</td><td>67.7</td><td>35.8 35.9</td><td>94.4% 95.8%</td></tr><tr><td></td><td colspan="14"></td></tr><tr><td>LLaVA-NeXT-7B VisionZip(CVPR25)</td><td>55.5</td><td>59.9</td><td>1303</td><td>74.9</td><td>68.3</td><td>56.3</td><td>Retain Up to 160 Tokens (↓ 94.4%) 30.5</td><td>60.4</td><td>58.4</td><td>36.3</td><td>87.8%</td></tr><tr><td>VisPruner (ICCV25)</td><td>56.5</td><td>59.7</td><td>1290</td><td>75.2</td><td>69.0</td><td>56.0</td><td></td><td></td><td></td><td></td><td>87.7%</td></tr><tr><td>PruneSID(ICLR26)</td><td>58.6</td><td>60.9</td><td>1389</td><td>80.1</td><td>67.3</td><td></td><td>30.7</td><td>60.1</td><td>58.2</td><td>35.4</td><td></td></tr><tr><td>DivPrune (CVPR25)</td><td>59.2</td><td>63.1</td><td>1375</td><td></td><td></td><td>52.5</td><td>27.4</td><td>58.8</td><td>62.2</td><td>36.7</td><td>88.6%</td></tr><tr><td>+ Ours</td><td>60.0</td><td></td><td></td><td>79.4</td><td>67.6</td><td>54.7</td><td>29.4</td><td>59.6</td><td>62.9</td><td>36.7</td><td>89.9%</td></tr><tr><td></td><td></td><td>63.7</td><td>1392</td><td>82.6</td><td>68.2</td><td>56.2</td><td>32.3</td><td>60.0</td><td>63.9</td><td>35.6</td><td>91.4%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>59.4</td><td>62.4</td><td>1367</td><td>81.7</td><td>68.8</td><td>55.3</td><td>30.7</td><td>60.3</td><td>62.8</td><td>36.2</td><td>90.5%</td></tr><tr><td>+ Ours</td><td>59.5</td><td>63.6</td><td>1388</td><td>82.7</td><td>68.2</td><td>56.1</td><td>32.8</td><td>60.4</td><td>64.1</td><td>37.2</td><td>91.9%</td></tr><tr><td>MMTok(ICLR26)</td><td>60.0</td><td>63.7</td><td>1413</td><td>83.8</td><td>68.0</td><td>54.5</td><td>30.1</td><td>59.3</td><td>64.9</td><td>37.1</td><td>91.4%</td></tr><tr><td>CDPruner(NeurIPS25) + Ours</td><td>60.9 60.8</td><td>64.7 65.1</td><td>1422 1436</td><td>87.0 86.2</td><td>66.9 66.9</td><td>55.4 56.8</td><td>32.9 37.0</td><td>58.3 58.5</td><td>65.7 66.3</td><td>35.4 35.8</td><td>92.1% 93.4%</td></tr></table>

Table A2: Evaluation on image benchmarks for LLaVA-NeXT-7B under diferent token reduction ratio.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>VQAText</td><td>OCR-B</td><td>VizWiz</td><td>SEED-I</td><td>MMMU</td><td>Avg.</td></tr><tr><td>Upper Bound (2880 Tokens)</td><td>64.2</td><td>67.9</td><td>1520</td><td>86.4</td><td>70.2</td><td>61.3</td><td>51.3</td><td>59.7</td><td>69.6</td><td>37.2</td><td>100.0%</td></tr><tr><td>LLaVA-NeXT-7B</td><td></td><td colspan="10">Retain 320 Tokens</td></tr><tr><td>HoloV (NeurIPS25)</td><td>59.4</td><td>65.5</td><td>1438</td><td>83.1</td><td>66.8</td><td>55.6</td><td>36.6</td><td>57.8</td><td>64.7</td><td>35.4</td><td>92.2%</td></tr><tr><td>DivPrune (CVPR25)</td><td>60.4</td><td>64.3</td><td>1439</td><td>84.8</td><td>68.0</td><td>54.0</td><td>35.9</td><td>57.6</td><td>66.1</td><td>37.8</td><td>93.0%</td></tr><tr><td>+ Ours</td><td>61.0</td><td>65.0</td><td>1477</td><td>85.7</td><td>67.8</td><td>56.9</td><td>40.3</td><td>58.1</td><td>66.9</td><td>36.9</td><td>94.8%</td></tr></table>

Table A3: Performance comparison on LLaVA-NeXT-7B under a fixed budget setting.

Ablation Studies on the Distinctiveness Score. In the main paper, we compute the distinctiveness score from the raw cosine similarity matrix. We further conduct ablation studies with several score variants. As shown in the second group of Tab. A6, we first replace our cosine-based distinctiveness score with a Raw-ℓ score, defined as the mean unnormalized Euclidean distance between each token and all other tokens in the raw feature space. This modulusaware distinctiveness metric improves performance on some benchmarks, especially the hallucination benchmark POPE. However, it leads to a noticeable performance drop on the text-oriented benchmark TextVQA. We also evaluate [CLS] attention with the same min–max normalization, but find that it is less compatible with our residual diversity objective.

Ablation Studies on Score Normalization. We further study the efect of score normalization in Tab. A6. Z-score normalization with a sigmoid transformation achieves performance comparable to our min–max normalization, with only minor diferences across benchmarks. This suggests that the proposed distinctiveness score is robust to the normalization choice. We use min–max normalization in the main experiments for its simplicity and its slightly stronger average performance.

## C.3 Further Eficiency Results

Complete Eficiency Comparison on LLaVA-1.5-7B. In the main paper, we provide the eficiency analysis on LLaVA-1.5-7B under the setting with 64 retained visual tokens. Here, we present the complete comparison under diferent pruning-ratio settings in Tab. A7. Across all token budgets, integrating Cen-Prune introduces negligible additional computational overhead. The small runtime diferences observed across settings are within normal measurement variation and do not indicate a consistent change in inference cost.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>VQAText</td><td>OCR-B</td><td>VizWiz</td><td>SEED-I</td><td>MMMU</td><td>Avg.</td></tr><tr><td>Upper Bound (2880 Tokens)</td><td>64.4</td><td>67.7</td><td>1551</td><td>85.1</td><td>73.0</td><td>63.3</td><td>54.5</td><td>62.4</td><td>71.6</td><td>36.4</td><td>100.0%</td></tr><tr><td>LLaVA-NeXT-7B</td><td></td><td></td><td></td><td>Retain Up to 640 Tokens (↓ 77.8%)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DivPrune (CVPR25)</td><td>60.0</td><td>64.3</td><td>1484</td><td>81.6</td><td>71.9</td><td>56.1</td><td>30.4</td><td>58.1</td><td>64.4</td><td>36.2</td><td>90.5%</td></tr><tr><td>+ Ours</td><td>61.2</td><td>65.3</td><td>1472</td><td>83.5</td><td>72.1</td><td>58.3</td><td>34.7</td><td>59.4</td><td>65.7</td><td>36.3</td><td>92.6%</td></tr><tr><td>LLaVA-NeXT-7B</td><td></td><td></td><td></td><td>Retain Up to 320 Tokens (↓ 88.9%)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DivPrune (CVPR25)</td><td>61.9</td><td>65.2</td><td>1498</td><td>85.1</td><td>72.7</td><td>57.9</td><td>35.8</td><td>60.0</td><td>67.2</td><td>37.2</td><td>93.8%</td></tr><tr><td>+ Ours</td><td>62.8</td><td>66.2</td><td>1522</td><td>85.3</td><td>73.0</td><td>59.8</td><td>40.2</td><td>60.3</td><td>68.2</td><td>36.7</td><td>95.5%</td></tr><tr><td>LLaVA-NeXT-7B</td><td></td><td></td><td></td><td></td><td>Retain Up to 160 Tokens (↓ 94.4%)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DivPrune (CVPR25)</td><td>63.7</td><td>68.0</td><td>1521</td><td>86.4</td><td>72.9</td><td>59.1</td><td>43.7</td><td>60.5</td><td>69.7</td><td>37.8</td><td>97.0%</td></tr><tr><td>+ Ours</td><td>64.0</td><td>68.0</td><td>1554</td><td>86.5</td><td>72.7</td><td>61.2</td><td>46.9</td><td>61.8</td><td>70.2</td><td>36.2</td><td>98.0%</td></tr></table>

Table A4: Comparison on LLaVA-NeXT-13B under diferent token reduction ratio.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>VQAText</td><td>OCR-B</td><td>VizWiz</td><td>SEED-I</td><td>MMMU</td><td>Avg.</td></tr><tr><td>Upper Bound (576 Tokens)</td><td>63.2</td><td>67.7</td><td>1522</td><td>85.9</td><td>72.8</td><td>61.3</td><td>33.6</td><td>56.6</td><td>66.9</td><td>36.4</td><td>100.0%</td></tr><tr><td>LLaVA-1.5-13B</td><td></td></tr><tr><td>VisionZip(CVPR25)</td><td>57.8</td><td>67.0</td><td>1458</td><td>82.4</td><td>Retain 128 Tokens (↓ 77.8%) 74.0</td><td>58.7</td><td>32.8</td><td>55.1</td><td>63.7</td><td>35.8</td><td>96.8%</td></tr><tr><td>PruneSID (ICLR26)</td><td>58.4</td><td>65.4</td><td>1445</td><td>83.9</td><td>72.9</td><td>57.5</td><td>29.3</td><td>55.9</td><td>64.1</td><td>36.7</td><td>95.8%</td></tr><tr><td>DivPrune (CVPR25)</td><td>59.0</td><td>66.8</td><td>1446</td><td>86.7</td><td>72.8</td><td>58.0</td><td>30.5</td><td>56.5</td><td>63.9</td><td>36.4</td><td>96.9%</td></tr><tr><td>+ Ours</td><td>59.5</td><td>67.2</td><td>1470</td><td>86.4</td><td>73.2</td><td>59.0</td><td>32.9</td><td>56.1</td><td>64.5</td><td>35.9</td><td>97.9%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>58.9</td><td>66.5</td><td>1480</td><td>86.8</td><td>73.5</td><td>58.7</td><td>32.2</td><td>55.7</td><td>64.8</td><td>35.3</td><td>97.4%</td></tr><tr><td>+ Ours</td><td>59.2</td><td>67.0</td><td>1460</td><td>86.3</td><td>73.4</td><td>59.2</td><td>32.5</td><td>55.8</td><td>64.8</td><td>35.9</td><td>97.7%</td></tr><tr><td>MMTok(ICLR26)</td><td>59.1</td><td>66.8</td><td>1434</td><td>86.2</td><td>73.5</td><td>58.8</td><td>33.6</td><td>56.0</td><td>64.6</td><td>35.3</td><td>97.6%</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>59.8</td><td>67.0</td><td>1448</td><td>87.2</td><td>72.7</td><td>58.6</td><td>32.1</td><td>55.8</td><td>65.0</td><td>36.0</td><td>97.6%</td></tr><tr><td>+ Ours</td><td>60.0</td><td>67.5</td><td>1447</td><td>86.3</td><td>72.9</td><td>58.9</td><td>33.7</td><td>55.5</td><td>65.1</td><td>36.3</td><td>98.2%</td></tr><tr><td>LLaVA-1.5-13B</td><td colspan="10">Retain 64 Tokens (↓ 88.9%)</td></tr><tr><td>VisionZip(CVPR25)</td><td>56.1</td><td>64.8</td><td>1403</td><td>75.9</td><td>74.2</td><td>57.5</td><td>31.0</td><td>56.1</td><td>60.2</td><td>36.1</td><td>94.1%</td></tr><tr><td>PruneSID (ICLR26)</td><td>57.9</td><td>63.9</td><td>1437</td><td>82.6</td><td>71.5</td><td>56.7</td><td>28.6</td><td>57.7</td><td>62.4</td><td>35.6</td><td>94.6%</td></tr><tr><td>DivPrune (CVPR25)</td><td>57.4</td><td>64.6</td><td>1476</td><td>84.7</td><td>71.8</td><td>57.1</td><td>30.1</td><td>58.1</td><td>62.3</td><td>35.0</td><td>95.5%</td></tr><tr><td>+ Ours</td><td>58.3</td><td>64.4</td><td>1466</td><td>84.6</td><td>72.3</td><td>58.2</td><td>31.3</td><td>57.8</td><td>63.3</td><td>35.8</td><td>96.5%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>58.8</td><td>64.4</td><td>1460</td><td>85.7</td><td>71.9</td><td>58.1</td><td>29.9</td><td>57.5</td><td>63.1</td><td>36.2</td><td>96.2%</td></tr><tr><td>+ Ours</td><td>58.9</td><td>66.0</td><td>1478</td><td>85.2</td><td>72.8</td><td>58.5</td><td>32.2</td><td>57.5</td><td>63.4</td><td>36.3</td><td>97.4%</td></tr><tr><td>MMTok(ICLR26)</td><td>58.5</td><td>65.7</td><td>1464</td><td>84.5</td><td>72.8</td><td>58.0</td><td>31.8</td><td>57.6</td><td>63.6</td><td>35.2</td><td>96.7%</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>59.3</td><td>65.5</td><td>1442</td><td>87.3</td><td>72.1</td><td>57.6</td><td>30.1</td><td>56.6</td><td>64.0</td><td>34.4</td><td>96.0%</td></tr><tr><td>+ Ours</td><td>59.1</td><td>66.6</td><td>1472</td><td>86.4</td><td>71.8</td><td>58.7</td><td>32.2</td><td>57.3</td><td>64.4</td><td>36.0</td><td>97.6%</td></tr><tr><td>LLaVA-1.5-13B</td><td colspan="10"></td></tr><tr><td>VisionZip(CVPR25)</td><td>58.3</td><td>61.5</td><td>1270</td><td>67.0</td><td>72.3</td><td>Retain 32 Tokens (↓ 94.4%) 55.3</td><td>27.9</td><td>57.1</td><td>55.9</td><td>35.1</td><td>89.8%</td></tr><tr><td>PruneSID(ICLR26)</td><td>55.6</td><td>61.9</td><td>1335</td><td>77.2</td><td>71.8</td><td>54.5</td><td>26.1</td><td>58.2</td><td>59.4</td><td>35.7</td><td>91.2%</td></tr><tr><td>DivPrune (CVPR25)</td><td>56.0</td><td>62.1</td><td>1380</td><td>79.7</td><td>70.8</td><td>54.6</td><td>26.4</td><td>57.8</td><td>59.6</td><td>34.6</td><td>91.5%</td></tr><tr><td>+ Ours</td><td>57.0</td><td>63.2</td><td>1446</td><td>80.8</td><td>72.4</td><td>56.3</td><td>29.0</td><td>58.4</td><td>61.0</td><td>36.1</td><td>94.4%</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>57.0</td><td>63.6</td><td>1423</td><td>81.6</td><td>72.3</td><td>56.4</td><td>28.8</td><td>58.4</td><td>60.7</td><td>35.8</td><td>94.2%</td></tr><tr><td>+ Ours</td><td>57.4</td><td>64.3</td><td>1422</td><td>81.6</td><td>72.3</td><td>57.3</td><td>29.0</td><td>58.0</td><td>61.3</td><td>35.9</td><td>94.6%</td></tr><tr><td>MMTok(ICLR26)</td><td>57.8</td><td>63.7</td><td>1427</td><td>82.4</td><td>73.2</td><td>55.7</td><td>28.2</td><td>58.5</td><td>61.7</td><td>35.0</td><td>94.2%</td></tr><tr><td>CDPruner (NeurIPS25)</td><td>58.5</td><td>64.0</td><td>1418</td><td>87.5</td><td>72.3</td><td>55.5</td><td>27.0</td><td>56.9</td><td>62.8</td><td>35.8</td><td>94.5%</td></tr><tr><td>+ Ours</td><td>58.6</td><td>65.6</td><td>1427</td><td>85.5</td><td>72.6</td><td>57.4</td><td>29.6</td><td>57.1</td><td>63.2</td><td>35.1</td><td>95.6%</td></tr></table>

Table A5: Evaluation on image benchmarks for LLaVA-1.5-13B under diferent token reduction ratio.

<table><tr><td>Method</td><td>GQA</td><td>MME</td><td>POPE</td><td>VQAText</td><td>Avg.</td></tr><tr><td>Upper Bound (576 Tokens)</td><td>61.9</td><td>1511</td><td>85.9</td><td>58.2</td><td>100.0%</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="5">Retain 32 Tokens (↓ 94.4%)</td></tr><tr><td>DivPrune (CVPR25)</td><td>55.0</td><td>1306</td><td>81.7</td><td>53.1</td><td>90.4%</td></tr><tr><td>Centering</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No Centering</td><td>55.4</td><td>1329</td><td>81.4</td><td>55.0</td><td>91.7%</td></tr><tr><td>Mean-Direction Removal</td><td>55.6</td><td>1322</td><td>81.3</td><td>54.8</td><td>91.5%</td></tr><tr><td>Mean-Centering (Ours)</td><td>56.0</td><td>1335</td><td>82.8</td><td>55.3</td><td>92.6%</td></tr><tr><td>Distinctiveness Variants</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>[CLS] Attention</td><td>55.0</td><td>1306</td><td>79.3</td><td>54.5</td><td>90.3%</td></tr><tr><td>Raw-l2</td><td>55.9</td><td>1338</td><td>83.3</td><td>53.8</td><td>92.0%</td></tr><tr><td>Raw-Cosine (Ours)</td><td>56.0</td><td>1335</td><td>82.8</td><td>55.3</td><td>92.6%</td></tr><tr><td>Distinctiveness Normalization</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Z-Norm. + Sigmoid</td><td>56.0</td><td>1325</td><td>83.6</td><td>54.6</td><td>92.3%</td></tr><tr><td>Min–Max Norm. (Ours)</td><td>56.0</td><td>1335</td><td>82.8</td><td>55.3</td><td>92.6%</td></tr></table>

Table A6: Further ablation studies on LLaVA-1.5-7B.

<table><tr><td>Method</td><td>Prefill (ms)</td><td>Latency (ms)</td><td>FLOPs (G)</td></tr><tr><td>Upper Bound (576 Tokens)</td><td>154.77</td><td>200.27</td><td>4431.21</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="3">Retain 128 Tokens (↓ 77.8%)</td></tr><tr><td>DivPrune (CVPR25)</td><td>93.50</td><td>136.65</td><td>1471.24</td></tr><tr><td>+ Ours</td><td>94.51</td><td>137.10</td><td>1471.24</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>118.45</td><td>161.27</td><td>1479.61</td></tr><tr><td>+ Ours</td><td>119.60</td><td>162.20</td><td>1479.61</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>129.99</td><td>172.21</td><td>3018.78</td></tr><tr><td>+ Ours</td><td>130.20</td><td>172.94</td><td>3018.78</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="3">Retain 64 Tokens (↓ 88.9%)</td></tr><tr><td>DivPrune (CVPR25)</td><td>77.68</td><td>119.53</td><td>1048.38</td></tr><tr><td>+ Ours</td><td>77.82</td><td>119.85</td><td>1048.38</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>90.92</td><td>133.54</td><td>1056.75</td></tr><tr><td>+ Ours</td><td>91.85</td><td>134.63</td><td>1056.75</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>112.08</td><td>154.51</td><td>2595.93</td></tr><tr><td>+ Ours</td><td>112.32</td><td>154.77</td><td>2595.93</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="3">Retain 32 Tokens (↓ 94.4%)</td></tr><tr><td>DivPrune (CVPR25)</td><td>69.38</td><td>111.22</td><td>836.96</td></tr><tr><td>+ Ours</td><td>69.48</td><td>111.07</td><td>836.96</td></tr><tr><td>CDPruner(NeurIPS25)</td><td>75.57</td><td>118.03</td><td>845.32</td></tr><tr><td>+ Ours</td><td>75.67</td><td>118.53</td><td>845.32</td></tr><tr><td>ZOO-Prune (CVPR26)</td><td>105.82</td><td>148.32</td><td>2348.51</td></tr><tr><td>+ Ours</td><td>105.94</td><td>148.59</td><td>2348.51</td></tr></table>

Table A7: Eficiency comparison on LLaVA-1.5-7B.

DivPrune

![](images/20996c75c6e29c8688c4d42086673ae83742479d16a573ae84ce41c646bfdbcb.jpg)  
Answer: Yes.

![](images/675ac52bd6be6c2f774bc9d19bb144272456057e7ccc3551ba2d230b5d6a137b.jpg)

## D Visualizations

We finally visualize the retained tokens under diferent reduction rates. As shown in Fig. A2, a green outline indicates a correct answer, whereas a red outline indicates an incorrect answer. Compared with DivPrune, Cen-Prune retains more task-relevant visual cues, which is particularly noticeable under low-budget settings. When only 32 visual tokens are retained, Cen-Prune is often better able to cover small targets in the image, such as bicycles and trucks.

As shown in Fig. A3, we further compare Cen-Prune with DivPrune on text-oriented images. Even with a relatively relaxed budget of 128 retained visual tokens, DivPrune struggles to preserve the integrity of textual content, which can lead to incomplete text recognition.

Original Image  
![](images/6aca50de2ba3861cf8f08a968154e50fc50734730c4cec8fdf9fc4bf52cb4e40.jpg)

Retain 32 Tokens  
![](images/2a1b0d653954d3d357774e8192c5deafafb0759b6ee3d79b20c862aab22a29c9.jpg)  
Retain 64 Tokens

Cen-Prune

![](images/9f02e1855b2d6decd5b938591a4c2f5fc07e19e4414661ae49bbf5a8595555d9.jpg)

![](images/165a34397015e4a17c6491931d3ee7f81862fe0e64d2c48ee308a6a0b3d9c8b6.jpg)

![](images/864c94b02df32f1f2bb982144bbaa8fb3f5c2997f4388b9a9ac9f2c87621c879.jpg)

Retain 128 Tokens  
![](images/67fcefa1cce9f1146fbe8106f1df85538954df1aeaf06b8cded6d430714904c1.jpg)  
Original Image  
Retain 32 Tokens  
Retain 64 Tokens

Question: Is there a bicycle in the image?  
![](images/4d771bbb6dcd9d1c0b834ee9834de2a2eba89c8abea8062de4f9995329e1246a.jpg)  
Retain 128 Tokens

![](images/e886261ebd065672353a8f58fe9e655fbe185ceb35e89976815cbc3bda889509.jpg)

![](images/1c3cfb4531e4960fa381553a1102ab6a31d12bf40245b872126c6c90176194d6.jpg)

![](images/1489c98002f670c761c0f074dd00053f4991f9e5f342e3ad4648da1c3c7b7f12.jpg)

![](images/7c167fbadd36c3e181a893706a5699d42eeb5aaabb02ce279650de139ed3ae98.jpg)  
Question: Is there a laptop in the image?

![](images/2f963b6553937ebab457292199cfe9c13b526dd48f75996d64cbdddfdc5d20df.jpg)

![](images/104372c402c6125b68935573bd45c7cc43d26f452651b1597024fc300aa87af6.jpg)  
Answer: Yes.

![](images/6e53deb1df52bc09ac1e567436ab9443a09988ea986c8be7dca2a7530c53bf5f.jpg)

DivPrune

![](images/5d3d282ff3293a80ed51229721b6e53ccf09143075fd45f01822777192af9f23.jpg)

![](images/e55e6ba7c6e28f2f67d13ea6f8861e11f1c42ce82b35cddb2fea88528fae1bec.jpg)

Cen-Prune

![](images/32e2d8f33020e78e07e00d7f8906f89887dfd108cf96c7a56af0c358e2e596ec.jpg)

![](images/61b7d5835c6f74948f78e8d7de06ccb6281935f3242acd91f8c57b5a176858d7.jpg)

![](images/f6c28600b290a1305cef2d9eb1ee362abd7897e3b5e4851d6a74895b0e5fa97c.jpg)  
Question: Is there a toilet in the image? Answer: Yes.

![](images/efc44f4bde2c868885ad34f51d76b4e5ee99be79a5abb83dac6396227c0db4cd.jpg)

![](images/0f42280dc5f8856b39371f8eb005c5db165e48131ba3d211870098afb5e90309.jpg)

![](images/d6b6f7e733953e8afa4f8ec433f1c20e126a1e571b66d549a43d515153a95546.jpg)

![](images/3addb8f32ca14a93161feb01a34639479a98db6d2ca28c793e46f5a2ce3b5dd3.jpg)

![](images/8b037c18b9869820b34a0ae7c1eb66ee2e804a8cad4c82a68d60f0368be75874.jpg)

![](images/afaf833f1f0883b905cb42dd65f102b6c34a35ab93cd57c864cf79e0d93c397d.jpg)

![](images/134adcb1e872a43fdf31694e2011786a2cf3b7a9a8b7684e526b7ca7a770f683.jpg)

![](images/98fb9b948dd47adfd2a34ddcb7252e07746661daf914a4ab0366f3052f4304d1.jpg)  
Question: Is there a truck in the image? Answer: Yes.

![](images/ae34cafde768749eb6bb37c3a90c532927daeca61e75b0411447b1952d682407.jpg)

Figure A2: The case comparison of small object coverage between DivPrune and Cen-Prune from POPE.  
Original Image  
![](images/9ea905025a97f496071598cc24a1b54567849d8ea6eadabf08486b7c7dbee27f.jpg)

Retain 32 Tokens  
![](images/65d087acc2ef5b8dea742fd0578b106f4a211c5e12fe65985e7093a3115d6579.jpg)

Retain 64 Tokens  
![](images/9b2cd589f726bc34db3551f147513a53f6e3c26cd462f2a8bbd73e3ca5da23ca.jpg)

Retain 128 Tokens  
![](images/de42dfb7cee3e8baba45bdb277bb24a72c597119df17d827f3ff8024b8ff0906.jpg)

Original Image  
![](images/cb8fc5a87cfead203ec661dc7097ae69eac14792ab86a43c256a3a9159a2f3af.jpg)

Retain 64 Tokens  
Retain 32 Tokens  
![](images/bd06ae67af006e9461c69672ace8477aafb3f1f6a12d0f318e0cbbcf453b54cf.jpg)

![](images/b11ec595034cf5c20b6999583c7c6d1d633134f620497b68e0ce7bec45028e2e.jpg)

Retain 128 Tokens  
![](images/b68b8741a7cde5b27ca83fbbd37ba77bc3e11775c6a0347db29825b2e198980f.jpg)  
Cen-Prune

![](images/144d28b14f9ffac6ac697aea8dbddcf70de38ceda4f774a6f10e7ec1de85c7cb.jpg)  
Question: What year was this from? Answer: 2013.

![](images/bf636f6ec6dfb7f43c3b1a3ca5d14e47467a490ebed1c9d82a5e5535b9e3c4da.jpg)

![](images/864cd0bda5168d44bff4a018a4849f6fcb6a13bc900d2f1825b7d3427f323f08.jpg)

![](images/9190c137c738529d8c08359e780d055f2fe22131fca54a4d47adc5a149e1d1a6.jpg)

![](images/0fa043464426a1a3da379299365bb4d9840d13a093b6ab651b4b03e29644a529.jpg)  
Deep water

![](images/3ae551547a0bcb4dcf066757ef684c4303e753957e3daaa70ce8ee3ae2510fb9.jpg)  
Question: What is the danger? Answer: Deep water.

![](images/0f739b91f8d255da54016a3e375c03af5c0b40ec6e57fd9d91bf8735a114e101.jpg)  
Deep water

![](images/e0b846a98118c35612730827301bca95a05f425c26d78096268ccc89f1ca36b0.jpg)  
Deep water

Figure A3: The case comparison of text-oriented Q&A between DivPrune and Cen-Prune from TextVQA.