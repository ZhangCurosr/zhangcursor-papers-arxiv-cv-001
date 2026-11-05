# BudgetSplat: Budget-Controllable Generalizable Gaussian Splatting

Yuxi Hu<sup>1</sup>, Jun Zhang<sup>1</sup>, Wei Mao, Yuchen Rao<sup>1</sup>, Friedrich Fraundorfer<sup>1</sup>

<sup>1</sup>Graz University of Technology

<sup>1</sup>firstname.lastname@tugraz.at

<sup>2</sup>wei.mao.research@gmail.com

## Abstract

Feed-forward 3D Gaussian Splatting enables eficient generalizable novel view synthesis, but existing pixel-aligned methods often generate highly redundant Gaussian primitives, causing the final representation size and rendering cost to grow rapidly with image resolution and the number of input views. We propose BudgetSplat, a budget-controllable feed-forward framework that adapts the final Gaussian representation to a user-specified preservation ratio. Rather than treating compaction as simple importance ranking, we formulate it as a joint problem of importance estimation, spatial coverage, and cross-view redundancy. A single trained model supports varying numbers of context views and multiple test-time budgets without per-budget retraining or per-scene compression. To realize this formulation, Intra-Cross Mask Supervision provides structure-, coverage-, and redundancy-aware guidance for learning Gaussian importance, while budget-conditioned feature modulation adapts Gaussian opacity and attributes to the requested preservation ratio. Coverage-Aware Budget Selection further combines patch-wise pre-selection, view-wise opacity ranking, and 3D redundancy suppression to construct compact yet spatially distributed Gaussian representations. Experiments on indoor reconstruction and cross-dataset generalization demonstrate consistent quality–size trade-ofs across diferent input view counts and preservation ratios, substantially reducing final representation storage and rendering cost while preserving rendering quality.

## 1. Introduction

3D Gaussian Splatting (3D-GS) [9] represents scenes using explicit anisotropic Gaussians and enables high-quality realtime novel view synthesis. However, it requires costly perscene optimization. Recent generalizable methods predict Gaussian representations from input images in a feed-forward manner, substantially improving reconstruction eficiency and generalization to unseen scenes [2, 4, 11, 20, 23].

Most feed-forward 3D-GS methods decode one or more Gaussians from each pixel. Although this pixel-aligned formulation provides strong geometric anchoring, its representation size grows rapidly with image resolution and input-view number. Overlapping observations additionally produce many redundant Gaussians, increasing the storage and rendering costs of the final scene asset. Existing methods reduce pixel-aligned Gaussian redundancy through diferent designs. Geometryaware fusion [20, 21] and view compression [18] consolidate overlapping pixel-aligned predictions across views; voxelaligned prediction [19] predicts Gaussians in a shared 3D volume, whereas global scene encoders [1, 8] compress multiview observations into a compact set of scene tokens before decoding the Gaussian representation. The latter demonstrate that a fixed-capacity Gaussian set can be suficient on some benchmarks, but its output size remains tied to the scene token capacity.

These developments raise a practical question: Can a single feed-forward model adapt its final Gaussian representation to diferent user-specified budgets? Existing approaches generally fall into two categories. Fixed-capacity methods [1, 8] tie the representation size to a particular token or decoder configuration, so substantially larger or more complex scenes may require retraining a higher-capacity model. Post-hoc compression methods [7] reduce an already reconstructed Gaussian set, but introduce additional per-scene processing or optimization. In practice, deployment conditions may require the same model to operate under diferent representation budgets. Simply applying a budget-unaware Top-� rule to a conventionally trained pixel-aligned model is suboptimal: it may over-allocate Gaussians to salient or high-opacity regions, leaving weakly textured areas under-covered. We therefore view budget-controlled Gaussian compaction as a joint problem of importance estimation, spatial coverage, and cross-view redundancy, rather than simple candidate ranking. These limitations motivate a feed-forward model whose final Gaussian representation can be adjusted at test time while retaining the geometric advantages of pixel-aligned reconstruction.

To this end, we propose BudgetSplat, a budget-controllable feed-forward 3D-GS framework that treats compact Gaussian allocation as a joint problem of importance estimation, spatial coverage, and cross-view redundancy. A single trained model supports varying context-view counts and user-specified testtime budgets without retraining or per-scene compression.

Intra-Cross Mask Supervision provides structure-, coverage-, and redundancy-aware guidance for learning Gaussian importance, while Budget-Conditioned Feature Modulation adapts Gaussian prediction to the requested preservation ratio. Finally, Coverage-Aware Budget Selection combines local preselection, view-wise opacity ranking, and 3D redundancy suppression to construct the compact representation.

Our main contributions are summarized as follows:

• We formulate budget-controllable feed-forward 3D-GS as a joint problem of importance estimation, spatial coverage, and cross-view redundancy, and propose a single pixel-aligned model that supports diferent contextview counts and user-specified test-time budgets without retraining or per-scene compression.

• We introduce Intra-Cross Mask Supervision to preserve informative structures, maintain sparse support in lowfrequency regions, and reduce repeated cross-view supervision, together with Budget-Conditioned Feature Modulation that adapts Gaussian opacity and attributes to the requested preservation ratio.

• We introduce Coverage-Aware Budget Selection, which combines patch-wise pre-selection, view-wise global opacity ranking, and 3D redundancy suppression to produce compact yet spatially distributed Gaussian representations. Experiments on ScanNet and Replica demonstrate robust quality–size trade-ofs across diferent input-view settings and test-time budgets.

## 2. Related Work

Generalizable novel view synthesis. NeRF [14] and 3D Gaussian Splatting (3D-GS) [9] achieve high-quality novel view synthesis but typically require per-scene optimization. Generalizable methods instead learn scene priors for feed-forward reconstruction. Early approaches predict image-conditioned radiance fields using pixel-aligned features, feature matching, or cost volumes [3, 5, 12, 17, 24]. Recent methods extend this paradigm to 3D-GS by directly predicting Gaussian geometry and attributes from input images [2, 4]. DepthSplat [23] and MonoSplat [13] further incorporate pretrained monocular depth priors to improve geometry and generalization, but introduce additional dependence on external foundation models and feature adaptation. More importantly, these methods primarily target reconstruction quality and do not explicitly control the size of the final Gaussian representation.

Compact Gaussian representations. Pixel-aligned prediction provides strong geometric anchoring, but its Gaussian count scales with image resolution and input-view number, causing substantial cross-view redundancy. Geometry-aware fusion [20, 21] and view compression [18] reduce repeated information, whereas voxel-aligned prediction [19] and scenelevel decoders [1, 8] redesign the prediction space to produce more compact representations. However, their output size is determined by the underlying fusion or representation design rather than a user-specified test-time budget. Post-hoc methods such as LightGaussian [7] compress an already reconstructed Gaussian set but require additional per-scene processing. EcoSplat [15] learns Gaussian importance and selects primitives through opacity ranking, but the final selection does not explicitly reason about spatial coverage or cross-view redundancy. In contrast, BudgetSplat treats controllable compaction as a joint importance–coverage–redundancy problem, introducing explicit low-frequency support, cross-view redundancy weighting, coverage-aware selection, and target-view alpha completion.

## 3. Method

The overview of BudgetSplat is illustrated in Figure 1. Given � posed context images $\dot { \boldsymbol { J } } = \{ \pmb { I } ^ { i } \} _ { i = 1 } ^ { N }$ , where $\breve { I ^ { i } } \in \mathbb { R } ^ { H \times W \times 3 }$ and their corresponding projection matrices $\mathcal { P } = \{ P ^ { i } \} _ { i = 1 } ^ { N }$ with ${ \cal P } ^ { i } = \mathbf { K } ^ { i } [ \mathbf { R } ^ { i } \ | \ \mathbf { t } ^ { i } ]$ , our objective is to reconstruct a compact 3D Gaussian representation whose size can be controlled at inference time. Here, K<sup>�</sup>, R<sup>�</sup>, and t<sup>�</sup> denote the camera intrinsic matrix, rotation, and translation, respectively.

Pixel-aligned feed-forward methods typically instantiate Gaussian candidates densely from the input images. As a result, the candidate count grows with both image resolution and the number of context views. We introduce a user-specified preservation ratio $\rho \in ( 0 , 1 ]$ to control the target fraction of candidates retained for rendering. Our goal is to learn the following budget-controllable mapping:

$$
\begin{array} { r l } & { f _ { \theta } : \left( \{ ( I ^ { i } , P ^ { i } ) \} _ { i = 1 } ^ { N } , \rho \right) \mapsto \{ ( \mu _ { j } , \alpha _ { j } , \Sigma _ { j } , c _ { j } ) \} _ { j = 1 } ^ { M } , } \end{array}\tag{1}
$$

where � denotes the actual number of Gaussians in the final representation.

The requested ratio defines a target budget

$$
{ \cal K } = \lfloor \rho N H W \rfloor .\tag{2}
$$

The subsequent geometric redundancy suppression may produce $M \leq K$ , so � is treated as an upper bound on the final representation size. An absolute upper-bound target budget $K _ { \mathrm { a b s } }$ can equivalently be specified through $\rho = K _ { \mathrm { a b s } } / ( N H W )$

The inference pipeline consists of three stages. First, the reconstruction backbone estimates dense pixel-aligned geometry and image features. Second, the requested ratio � conditions the Gaussian prediction branch, allowing the predicted opacity and Gaussian attributes to adapt to the target budget. Third, Coverage-Aware Budget Selection retains a spatially distributed subset under the requested budget. The pseudo masks, cross-view redundancy weights, and auxiliary losses introduced later are used only during training and incur no additional inference-time supervision.

## 3.1 Pixel-aligned Gaussian Estimation

We adopt FreeSplat’s feature extraction, cost-volume depth estimation, and Gaussian prediction backbone [20], while removing its native geometric fusion module. Given posed context images, the backbone extracts multi-scale features, constructs a multi-view cost volume, and estimates per-pixel depth. The depth is unprojected to obtain Gaussian centers $\pmb { \mu } _ { i } ( \pmb { p } )$ and corresponding base features $f _ { i } ( \pmb { p } )$ . As described in the next section, the requested budget is subsequently injected into this feature before the Gaussian parameters are predicted. The resulting candidates are then processed by our coverage-aware budget selection strategy to construct the final representation. We retain the architecture of the FreeSplat-based prediction backbone without modification, while optimizing it jointly with the proposed budget-controlled components.

![](images/08e085de6a636fa4714bcea8a4b8bc20c6dbe0308a3c7e320e956ab04a8f0831.jpg)  
Figure 1: Overview of BudgetSplat. (A) Given � posed context images, a pixel-aligned reconstruction backbone estimates dense geometry and base Gaussian features. (B) During training, Intra-Cross Mask Supervision constructs pseudo importance targets that preserve high-frequency structures, maintain sparse coverage in low-frequency regions, and down-weight redundant cross-view observations. (C) Budget-Conditioned Feature Modulation injects the user-specified preservation ratio $\rho$ into the base Gaussian features before parameter prediction, allowing the predicted opacity and Gaussian attributes to adapt to the requested budget. (D) Coverage-Aware Budget Selection combines patch-wise pre-selection, view-wise global opacity ranking, and 3D redundancy suppression to retain a compact and spatially distributed Gaussian representation. A single trained model supports arbitrary test-time preservation ratios $\rho \in ( 0 , 1 ]$ without retraining

## 3.2 Budget-conditioned Gaussian Compaction

The reconstruction backbone produces ��� pixel-aligned features and Gaussian centers. Given a requested preservation ratio $\rho ,$ we condition the features used for Gaussian prediction and select a subset for rasterization. This compaction remains part of the feed-forward inference pipeline and requires neither per-budget retraining nor per-scene optimization.

Budget-Conditioned Feature Modulation (BCFM). Changing only the selection budget would apply diferent truncation levels to budget-independent Gaussian predictions. We therefore inject the requested ratio $\rho$ into the feature used by the Gaussian prediction heads. Specifically, $\rho$ is spatially broadcast and transformed by a lightweight convolutional embedding network into a budget feature $\bar { E _ { i } } \in \mathbb { R } ^ { C \times H \times W }$ , which is added to the base feature $\mathbf { \bar { \boldsymbol { F } } } _ { i } \in \mathbb { R } ^ { C \times H \times W }$ before Gaussian parameter

prediction:

$$
\left\{ \widetilde { \alpha } _ { i } , \tilde { \Sigma } _ { i } , \widetilde { c } _ { i } \right\} = \mathrm { M L P } \left( F _ { i } + E _ { i } \right) .\tag{3}
$$

Here, $\tilde { \alpha } _ { i } , \tilde { \Sigma } _ { i }$ , and $\tilde { c } _ { i }$ denote the predicted opacity, covariance, and color, respectively. The activated opacity is used both for rendering and for candidate ranking. This conditioning allows a single model to adapt its Gaussian predictions to diferent test-time budgets.

Coverage-Aware Budget Selection (CABS). Directly ranking all candidates within a view may concentrate the retained Gaussians in salient regions and leave smooth or weakly textured areas under-covered. We therefore perform coverage-aware selection in three steps. First, each input view is divided into local patches, from which an oversampled set of high-opacity candidates is retained to construct a spatially distributed candidate pool. Second, all candidates retained from the patches of the same view are ranked jointly according to their predicted opacity. We refer to this step as view-wise global opacity ranking, since the ranking is performed across the candidate pool of one view rather than independently within each patch. Finally, the ranked candidate pools from all views are jointly filtered by 3D non-maximum suppression to remove geometrically redundant candidates according to the requested budget. The retained candidates form the compact Gaussian representation $\mathcal { G } _ { \rho }$ . Detailed budget allocation and selection procedures are provided in the supplementary material.

## 3.3 Learning Gaussian Importance

Since CABS ranks Gaussian candidates by their activated opacity, the opacity should reflect how useful each candidate is under specified budgets. However, ground-truth Gaussian importance labels are unavailable. We therefore introduce Intra-Cross Mask Supervision (ICMS), which consists of intraframe structure-and-coverage mask construction, cross-frame redundancy weighting, and weighted opacity supervision. The first two components generate pseudo labels and supervision weights, which are jointly applied to the activated opacity through a weighted mask loss.

Let $\mathcal { P } = \{ 1 , \ldots , H \} \times \{ 1 , \ldots , W \}$ denote the image pixel grid. We use � and � to index context and target views, respectively, and $\pmb { p } = ( u , \nu ) \in \mathcal { P }$ to denote a pixel location.

Intra-frame structure and coverage. For each context image $I ^ { i }$ , we compute an image-gradient magnitude map:

$$
\pmb { g } _ { i } = \sqrt { \lVert \nabla _ { x } pmb { I } ^ { i } \rVert _ { 2 } ^ { 2 } + \lVert \nabla _ { y } \pmb { I } ^ { i } \rVert _ { 2 } ^ { 2 } } ,\tag{4}
$$

where larger responses indicate detail-rich regions. To prevent the pseudo mask from allocating the entire preservation budget to high-frequency structures, we assign only a portion of the requested ratio to these regions and reserve the remainder for sparse low-frequency coverage. Specifically, we set $\rho _ { h } = \gamma \rho _ { ; }$ with $\gamma = 0 . 6$ in our experiments, and define

$$
\Omega _ { i } ^ { \mathrm { h i g h } } ( p ) = \left\{ \begin{array} { l l } { 1 , } & { g _ { i } ( p ) > Q _ { 1 - \rho _ { h } } \left( \left\{ g _ { i } ( q ) \vert q \in \mathcal { P } \right\} \right) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{5}
$$

where $\mathcal { Q } _ { 1 - \rho _ { h } } ( \cdot )$ denotes the (1 − �<sub>ℎ</sub>)-quantile within the current view.

High-frequency selection alone may leave large smooth regions uncovered. We therefore uniformly sample a sparse subset $\mathscr { L } _ { i }$ from the remaining pixels

$$
{ \mathcal U } _ { i } = \left\{ p \in { \mathcal P } \mid \Omega _ { i } ^ { \mathrm { h i g h } } ( p ) = 0 \right\} ,\tag{6}
$$

and define

$$
\Omega _ { i } ^ { \mathrm { l o w } } ( { \pmb p } ) = \left\{ \begin{array} { l l } { 1 , } & { { \pmb p } \in \mathcal { L } _ { i } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

Since the high- and low-frequency sets are disjoint, the intraframe pseudo mask is

$$
\Omega _ { i } ^ { \mathrm { { i n t r a } } } = \Omega _ { i } ^ { \mathrm { { h i g h } } } + \Omega _ { i } ^ { \mathrm { { l o w } } } .\tag{8}
$$

Pixels selected from either branch are labeled as positive, while all remaining pixels are labeled as negative. This construction favors detail-rich regions while retaining sparse support over weakly textured areas.

Cross-frame redundancy weighting. Positive candidates from diferent context views may observe the same 3D region. We keep their binary pseudo labels unchanged, but reduce the repeated positive supervision. Each positive pixel is first unprojected into the global 3D space:

$$
\pmb { x } _ { i , \pmb { p } } = \Pi ^ { - 1 } \left( \pmb { p } , d _ { i } ( \pmb { p } ) , \pmb { P } ^ { i } \right) .\tag{9}
$$

For high-frequency candidates, we first retain one representative from each occupied coarse voxel independently within every context view. The retained representatives are then grouped across views using a finer voxel grid. Let $S _ { \mathrm { h i g h } }$ denote the retained representatives, and let $\mathcal { V } _ { i } ( \pmb { p } )$ denote the fine-voxel group containing candidate $( i , p )$ . We define the positive supervision weight as

$$
w _ { i } ^ { + } ( p ) = \left\{ \begin{array} { l l } { \lambda _ { \mathrm { l o w } } , } & { \Omega _ { i } ^ { \mathrm { l o w } } ( p ) = 1 , } \\ { 1 } & \\ { \overline { { | \mathcal { V } _ { i } ( p ) | } } , } & { ( i , p ) \in S _ { \mathrm { h i g h } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{10}
$$

Therefore, low-frequency support receives a fixed weaker weight, while the representatives of each repeated 3D region share a total positive weight of one.

ICMS objective. Positive pixels use the weights defined above, whereas negative pixels retain unit weight:

$$
\bar { w } _ { i } ( \pmb { p } ) = \left\{ \begin{array} { l l } { w _ { i } ^ { + } ( \pmb { p } ) , } & { \Omega _ { i } ^ { \mathrm { i n t r a } } ( \pmb { p } ) = 1 , } \\ { 1 , } & { \Omega _ { i } ^ { \mathrm { i n t r a } } ( \pmb { p } ) = 0 . } \end{array} \right.\tag{11}
$$

We supervise the activated opacity using weighted binary cross-entropy:

$$
\mathcal { L } _ { \mathrm { m a s k } } = \frac { 1 } { N H W } \sum _ { i = 1 } ^ { N } \sum _ { p \in \mathcal { P } } \bar { w } _ { i } ( p ) \mathcal { L } _ { \mathrm { B C E } } \left( \Omega _ { i } ^ { \mathrm { i n t r a } } ( p ) , \tilde { \alpha } _ { i } ( p ) \right) .\tag{12}
$$

Thus, cross-frame redundancy weighting suppresses repeated positive supervision without changing the standard negative opacity supervision.

Alpha completion. The weighted mask loss provides local supervision for Gaussian importance, but does not directly guarantee complete coverage in the target views, especially under aggressive budgets. We therefore render the compact Gaussian representation $\mathcal { G } _ { \rho }$ and penalize missing alpha support:

$$
\mathcal { L } _ { \mathrm { { a l p h a } } } = \frac { 1 } { N ^ { \mathrm { { t g t } } } H W } \sum _ { t = 1 } ^ { N ^ { \mathrm { { t g t } } } } \sum _ { p \in \mathcal { P } } \left| 1 - A _ { t } ^ { \mathrm { { t g t } } } ( p ) \right| ,\tag{13}
$$

where $A _ { t } ^ { \mathrm { t g t } } ( p )$ denotes the rendered alpha value at pixel � in target view �. This auxiliary loss complements the local importance supervision by encouraging the selected Gaussians to maintain complete target-view coverage.

## 3.4 Training Strategy and Objective

We train BudgetSplat using ground-truth target RGB images in two stages. The first stage learns dense pixel-aligned reconstruction without budget-controlled selection. Following prior works [2, 4, 20, 23], we optimize the rendered target image using

$$
{ \mathcal { L } } _ { \mathrm { r e n d e r } } = \lambda _ { \mathrm { M S E } } { \mathcal { L } } _ { \mathrm { M S E } } + \lambda _ { \mathrm { L P I P S } } { \mathcal { L } } _ { \mathrm { L P I P S } } .\tag{14}
$$

This stage establishes a reliable dense reconstruction before budget-controlled compaction is introduced.

After the dense prediction converges, we activate the budgetcontrolled branch and continue training with top-� rendering. During this stage, $\rho$ is randomly sampled and is injected into the prediction branch through BCFM. Its sampling range is gradually shifted toward lower preservation ratios. The compact representation produced under the sampled budget is rasterized, and the model is jointly optimized using the reconstruction, mask supervision, and alpha completion objectives:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r e n d e r } } + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } + \lambda _ { \mathrm { a l p h a } } \mathcal { L } _ { \mathrm { a l p h a } } .\tag{15}
$$

Through this two-stage procedure, the model first learns reliable dense reconstruction and then learns to adapt its Gaussian prediction and selection to the requested budget while preserving spatial coverage.

## 4. Experiments

## 4.1 Experimental Setup

Datasets. Following FreeSplat [20], we train on the same ScanNet [6] split with 100 training scenes and 8 test scenes, and evaluate zero-shot generalization on 8 Replica [16] scenes. We report standard image quality metrics, including pixel-level PSNR, patch-level SSIM [22], and feature-level LPIPS [25]. Implementation details. We train using Adam [10] with a learning rate of $1 0 ^ { - 4 }$ and a batch size of 1 on a single RTX 4090 GPU. Unless otherwise specified, images are resized to $3 8 4 \times 5 1 2$ . We first train the dense reconstruction model for 300K iterations. Starting from this checkpoint, we train the budget-controlled stage for another 300K iterations. The final model therefore undergoes 600K iterations in total. We set $\lambda _ { \mathrm { M S E } } = 1 . 0 , \lambda _ { \mathrm { L P I P S } } = 0 . 0 5 , \lambda _ { \mathrm { m a s k } } = 0 . 1$ , and $\lambda _ { \mathrm { { a l p h a } } } = 1 . 0$ Detailed training schedules are provided in the supplementary material.

Baselines. We group compared methods into native-capacity (pixelSplat [2], MVSplat [4], DepthSplat [23], FreeSplat [20], ZPressor [18], VolSplat [19], and GlobalSplat [8]) and budgeted-compaction (EcoSplat [15]) methods. Unless otherwise noted, we use oficial checkpoints and configurations under the same view splits. We include the published 2- and 3-view results of FreeSplat++ [21], but omit it from the 10- view comparison because its implementation is unavailable for reproducing our $2 5 6 \times 2 5 6$ protocol. BudgetSplat<sub>�</sub> denotes the same trained model evaluated with $\rho = X / 1 0 0$ , and #GS is the number of rendered Gaussians. Bold and underlined values indicate the best and second-best results within each group.

## 4.2 Comparisons with Existing Methods

Results on ScanNet. We evaluate our method on ScanNet [6] using 2, 3, and 10 context views. All settings use the same trained model, which also supports arbitrary preservation ratios at test time without retraining. Note that BudgetSplat <sub>100%</sub> contains more Gaussians than native FreeSplat [20] because our model operates on the dense pixel-aligned candidate set and does not use its native geometric fusion. The lower-budget variants therefore reflect compaction by our selector rather than additional pruning of FreeSplat’s already fused output.

Figure 2 compares BudgetSplat with recent feed-forward 3D-GS methods. While competing methods exhibit incomplete coverage, structural discontinuities, or local over-smoothing, BudgetSplat better preserves scene completeness and fine details under a compact budget. The supplementary material further visualizes both the selected Gaussian centers and the corresponding renderings across diferent preservation ratios.

As shown in Table 1, BudgetSplat achieves the best rendering quality among the evaluated budgeted-compaction methods. At the 25% budget, it matches EcoSplat [15]’s Gaussian counts, while substantially improving all rendering metrics. Compared with FreeSplat [20], it uses over 60% fewer Gaussians with PSNR diferences of only 0.24 and 0.38 dB.

Table 2 further reports interpolation and extrapolation results with 10 context views. All methods are evaluated at $2 5 6 \times 2 5 6$ because several pixel-aligned baselines run out of memory at $3 8 4 \times 5 1 2$ on an RTX 4090. BudgetSplat<sub>25%</sub> achieves the best PSNR and SSIM among the budgetedcompaction methods and remains close to FreeSplat [20] in PSNR and SSIM, while using approximately 45% fewer Gaussians. Even at the aggressive 5% budget, BudgetSplat uses only 33k Gaussians, with PSNR decreasing by 0.22 dB for interpolation and 0.35 dB for extrapolation relative to BudgetSplat<sub>100%</sub>.

Representation compactness. Because recent methods use diferent architectures and inference pipelines, we compare the output-side cost of the final Gaussian representation. Table 2 also reports the Gaussian count and raw storage required for rendering. GlobalSplat [8] produces the smallest native representation, while $\mathrm { \mathbf { B u d g e t S p l a t } } _ { 5 \% }$ achieves higher PSNR and SSIM with a comparable budget of about 33k Gaussians. At the 25% budget, BudgetSplat also clearly outperforms EcoSplat [15] using the same number of Gaussians. These results show that the preservation ratio enables a controllable trade-of between representation size and rendering quality.

Zero-shot generalization on Replica. We further evaluate cross-dataset generalization by applying the models to Replica [16] without finetuning. As shown in Table 3, we distinguish methods evaluated at their native output capacities from methods that explicitly perform budgeted Gaussian compaction. Among the budgeted-compaction methods, BudgetSplat achieves the strongest rendering quality. The results also show that absolute cross-dataset performance remains strongly influenced by the underlying reconstruction architecture and scene representation, as evidenced by the strong performance of VolSplat [19] and GlobalSplat [8] in diferent settings.

More importantly, BudgetSplat maintains stable rendering quality when the budget is reduced from 100% to 25%. These results indicate that the learned budget-conditioned selection generalizes to unseen scenes and removes approximately 74% of the Gaussians with only limited additional degradation relative to the full representation.

![](images/79e06bb85033d4c7524d9c282e68902612a758687dd141845af1e91485034318.jpg)

Figure 2: Qualitative results on ScanNet [6] with 10 context views. Ours is rendered using only 25% of the Gaussian budget.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="4">2 Views</td><td colspan="4">3 Views</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>#GS(k)↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>#GS(k)↓</td></tr><tr><td rowspan="9">Native capacity</td><td>pixelSplat [2]</td><td>24.50</td><td>0.776</td><td>0.288</td><td>1180</td><td>23.44</td><td>0.782</td><td>0.270</td><td>1769</td></tr><tr><td>MVSplat [4]</td><td>24.02</td><td>0.771</td><td>0.268</td><td>393</td><td>22.26</td><td>0.744</td><td>0.328</td><td>590</td></tr><tr><td>DepthSplat [23]</td><td>24.59</td><td>0.783</td><td>0.276</td><td>393</td><td>23.10</td><td>0.761</td><td>0.327</td><td>590</td></tr><tr><td>FreeSplat [20]</td><td>28.08</td><td>0.837</td><td>0.211</td><td>278</td><td>27.45</td><td>0.829</td><td>0.222</td><td>382</td></tr><tr><td>FreeSplat++† [21]</td><td>28.19</td><td>0.838</td><td>0.212</td><td>68</td><td>27.45</td><td>0.829</td><td>0.223</td><td>89</td></tr><tr><td>ZPressor [18]</td><td>18.32</td><td>0.694</td><td>0.369</td><td>393</td><td>21.65</td><td>0.746</td><td>0.317</td><td>590</td></tr><tr><td>VolSplat [19]</td><td>23.40</td><td>0.783</td><td>0.365</td><td>393</td><td>22.99</td><td>0.775</td><td>0.372</td><td>590</td></tr><tr><td>GlobalSplat [8]</td><td>21.81</td><td>0.725</td><td>0.419</td><td>33</td><td>23.49</td><td>0.773</td><td>0.383</td><td>33</td></tr><tr><td>EcoSplat100% [15]</td><td>21.89</td><td>0.747</td><td>0.352</td><td>393</td><td>22.15</td><td>0.752</td><td>0.358</td><td>590</td></tr><tr><td rowspan="4">Budgeted compaction</td><td>EcoSplat25% [15]</td><td>19.97</td><td>0.672</td><td>0.426</td><td>98</td><td>20.35</td><td>0.703</td><td>0.420</td><td>147</td></tr><tr><td>BudgetSplat100%</td><td>28.11</td><td>0.839</td><td>0.228</td><td>382</td><td>27.13</td><td>0.825</td><td>0.245</td><td>565</td></tr><tr><td>BudgetSplat25%</td><td>27.84</td><td>0.840</td><td>0.250</td><td>98</td><td>27.07</td><td>0.828</td><td>0.264</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>147</td></tr></table>

Table 1: Generalizable NVS interpolation results on ScanNet [6] with 2 and 3 context views. The input resolution is 384 × 512. † denotes results reported in the original paper.

## 4.3 Ablations

We conduct ablations on ScanNet [6]. Starting from the 300K dense-stage checkpoint we trained, all variants are further trained for 100K iterations for budget compaction. Table 4 evaluates the contributions of the supervision and selection components while keeping BCFM fixed. Additional ablations are provided in the supplementary material.

Our baseline applies plain view-wise opacity Top-K to retain 25% of the dense pixel-aligned Gaussians, without importance supervision or coverage-aware selection, resulting in severe performance degradation. Adding intra-frame structureand-coverage supervision in No.1 substantially improves all rendering metrics, showing that the pseudo masks provide an efective signal for learning Gaussian importance. No.2 further introduces alpha completion, which improves all metrics. This indicates that local mask supervision alone does not guarantee complete target-view coverage, particularly under aggressive compression. No.3 adds cross-frame redundancy weighting to No.2. The resulting improvement shows that reducing the supervision assigned to repeated observations encourages a less redundant allocation of Gaussians across views. No.4 adds coverage-aware budget selection to No.2. Its larger improvement demonstrates that spatially distributed candidate selection helps prevent the retained Gaussians from collapsing into highly textured regions. The full model achieves the best overall performance, confirming that the training-time importance guidance and inference-time selection strategy are complementary.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td rowspan="2">#GS(k)↓</td><td rowspan="2">GS Size (MiB)↓</td><td colspan="3">View Interpolation</td><td colspan="3">View Extrapolation</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="7">Native capacity</td><td>pixelSplat [2]</td><td>1966</td><td>442.5</td><td>20.31</td><td>0.713</td><td>0.375</td><td>18.93</td><td>0.680</td><td>0.410</td></tr><tr><td>MVSplat [4]</td><td>655</td><td>147.4</td><td>19.25</td><td>0.661</td><td>0.379</td><td>19.00</td><td>0.663</td><td>0.382</td></tr><tr><td>DepthSplat [23]</td><td>655</td><td>147.4</td><td>21.76</td><td>0.735</td><td>0.297</td><td>21.10</td><td>0.720</td><td>0.311</td></tr><tr><td>FreeSplat [20]</td><td>294</td><td>66.2</td><td>25.03</td><td>0.800</td><td>0.223</td><td>23.96</td><td>0.780</td><td>0.250</td></tr><tr><td>ZPressor [18]</td><td>393</td><td>88.5</td><td>21.59</td><td>0.746</td><td>0.317</td><td>21.43</td><td>0.730</td><td>0.289</td></tr><tr><td>VolSplat [19]</td><td>655</td><td>147.4</td><td>22.49</td><td>0.769</td><td>0.325</td><td>21.82</td><td>0.754</td><td>0.341</td></tr><tr><td>GlobalSplat [8]</td><td>33</td><td>7.4</td><td>23.13</td><td>0.761</td><td>0.338</td><td>21.82</td><td>0.721</td><td>0.381</td></tr><tr><td rowspan="6">Budgeted compaction</td><td>EcoSplat100% [15]</td><td>655</td><td>147.4</td><td>21.97</td><td>0.761</td><td>0.306</td><td>22.08</td><td>0.753</td><td>0.311</td></tr><tr><td>EcoSplat25% [15]</td><td>163</td><td>36.7</td><td>21.87</td><td>0.766</td><td>0.315</td><td>21.81</td><td>0.751</td><td>0.330</td></tr><tr><td>EcoSplat5% [15]</td><td>33</td><td>7.4</td><td>21.43</td><td>0.751</td><td>0.354</td><td>21.31</td><td>0.735</td><td>0.370</td></tr><tr><td>BudgetSplat100%</td><td>653</td><td>147.0</td><td>24.90</td><td>0.790</td><td>0.254</td><td>23.70</td><td>0.771</td><td>0.280</td></tr><tr><td>BudgetSplat25%</td><td>163</td><td>36.7</td><td>24.91</td><td>0.793</td><td>0.295</td><td>23.66</td><td>0.775</td><td>0.318</td></tr><tr><td>BudgetSplat5%</td><td>33</td><td>7.4</td><td>24.68</td><td>0.789</td><td>0.368</td><td>23.35</td><td>0.771</td><td>0.391</td></tr></table>

Table 2: Generalizable NVS interpolation and extrapolation results on ScanNet [6] with 10 context views. The input resolution is 256 × 256.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="4">3 Views</td><td colspan="4">10 Views</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>#GS(k)↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>#GS(k)↓</td></tr><tr><td rowspan="7">Native capacity</td><td>pixelSplat [2]</td><td>25.88</td><td>0.838</td><td>0.214</td><td>1769</td><td>21.03</td><td>0.745</td><td>0.344</td><td>1966</td></tr><tr><td>MVSplat [4]</td><td>23.81</td><td>0.812</td><td>0.221</td><td>590</td><td>19.11</td><td>0.696</td><td>0.348</td><td>655</td></tr><tr><td>DepthSplat [23]</td><td>25.73</td><td>0.844</td><td>0.169</td><td>590</td><td>20.29</td><td>0.733</td><td>0.275</td><td>655</td></tr><tr><td>FreeSplat [20]</td><td>26.98</td><td>0.848</td><td>0.171</td><td>423</td><td>21.24</td><td>0.752</td><td>0.289</td><td>448</td></tr><tr><td>ZPressor [18]</td><td>23.93</td><td>0.816</td><td>0.197</td><td>590</td><td>21.23</td><td>0.750</td><td>0.258</td><td>393</td></tr><tr><td>VolSplat [19]</td><td>26.67</td><td>0.861</td><td>0.206</td><td>590</td><td>21.70</td><td>0.774</td><td>0.290</td><td>655</td></tr><tr><td>GlobalSplat [8]</td><td>25.99</td><td>0.823</td><td>0.286</td><td>33</td><td>21.88</td><td>0.755</td><td>0.305</td><td>33</td></tr><tr><td rowspan="4">Budgeted compaction</td><td>EcoSplat100% [15]</td><td>23.85</td><td>0.793</td><td>0.244</td><td>590</td><td>20.38</td><td>0.715</td><td>0.305</td><td>655</td></tr><tr><td>EcoSplat25% [15]</td><td>22.53</td><td>0.776</td><td>0.299</td><td>147</td><td>20.64</td><td>0.730</td><td>0.316</td><td>163</td></tr><tr><td>BudgetSplat100%</td><td>26.38</td><td>0.841</td><td>0.211</td><td>580</td><td>21.08</td><td>0.744</td><td>0.297</td><td>631</td></tr><tr><td>BudgetSplat25%</td><td>26.35</td><td>0.841</td><td>0.235</td><td>147</td><td>21.14</td><td>0.747</td><td>0.313</td><td>163</td></tr></table>

Table 3: Cross-dataset generalization on Replica [16]. The input resolution is $3 8 4 \times 5 1 2$ for the 3-view setting and $2 5 6 \times 2 5 6$ for the 10-view setting.

<table><tr><td>Method</td><td>Components</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓ #GS(k)↓</td></tr><tr><td>Baseline</td><td></td><td>13.02</td><td>0.386 0.561</td><td>147</td></tr><tr><td>No.1</td><td>I</td><td>24.80</td><td>0.743</td><td>0.336 147</td></tr><tr><td>No.2</td><td>I + A</td><td>26.14</td><td>0.810 0.311</td><td>147</td></tr><tr><td>No.3</td><td> $\mathrm { I } + \mathrm { C } + \mathrm { A }$ </td><td>26.38</td><td>0.813 0.310</td><td>147</td></tr><tr><td>No.4</td><td> $\mathrm { I } + \mathrm { A } + \mathrm { S }$ </td><td>26.75</td><td>0.821</td><td>0.279 147</td></tr><tr><td>Full</td><td> $\boldsymbol { \mathrm { I } } + \boldsymbol { \mathrm { C } } + \boldsymbol { \mathrm { A } } + \boldsymbol { \mathrm { S } }$ </td><td>26.77</td><td>0.823</td><td>0.275 147</td></tr></table>

Table 4: Ablation on ScanNet [6]. I, C, A, and S denote intraframe supervision, cross-frame weighting, alpha completion, and coverage-aware budget selection, respectively. Results are reported using 3 context views.

## 5. Conclusion

We present BudgetSplat, a budget-controllable feed-forward 3D-GS. Instead of keeping all pixel-aligned Gaussians, BudgetSplat learns to compact them according to a user-specified budget ratio. BudgetSplat can flexibly adjust the number of Gaussians while preserving rendering quality, without perbudget retraining or a separate ofline compression stage. Experiments show that BudgetSplat provides an efective trade-of between visual quality, representation storage, rendering memory, and rendering eficiency.

Limitations and discussions. BudgetSplat relies on image gradients and estimated geometry for pseudo supervision, so degraded image quality or depth errors may afect selection under very compact budgets. Moreover, while BudgetSplat reduces the final representation and rendering costs, it retains the dense pixel-aligned prediction backbone. Future work will explore more robust supervision and budget-aware candidate generation.

## References

[1] Honggyu An, Jaewoo Jung, Mungyeom Kim, Sunghwan Hong, Chaehyun Kim, Kazumi Fukuda, Minkyeong Jeon, Jisang Han, Takuya Narihira, Hyuna Ko, et al. C3g: Learning compact 3d representations with 2k gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[2] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19457–19467, 2024.

[3] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14124–14133, 2021.

[4] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Eficient 3d gaussian splatting from sparse multiview images. In Proceedings ofthe European Conference on Computer Vision, pages 370–386, 2024.

[5] Yuedong Chen, Haofei Xu, Qianyi Wu, Chuanxia Zheng, Tat-Jen Cham, and Jianfei Cai. Explicit correspondence matching for generalizable neural radiance fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[6] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richlyannotated 3d reconstructions ofindoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

[7] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural information processing systems, 37:140138–140158, 2024.

[8] Roni Itkin, Noam Issachar, Yehonatan Keypur, Xingyu Chen, Anpei Chen, and Sagie Benaim. Globalsplat: Eficient feedforward 3d gaussian splatting via global scene tokens, 2026.

[9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 2023.

[10] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[11] Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen, Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, and Ziwei Liu. Mvsgaussian: Fast generalizable gaussian splatting reconstruction from multi-view stereo. In European Conference on Computer Vision, pages 37–53. Springer, 2024.

[12] Tianqi Liu, Xinyi Ye, Min Shi, Zihao Huang, Zhiyu Pan, Zhan Peng, and Zhiguo Cao. Geometry-aware reconstruction and fusion-refined rendering for generalizable neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7654–7663, 2024.

[13] Yifan Liu, Keyu Fan, Weihao Yu, Chenxin Li, Hao Lu, and Yixuan Yuan. Monosplat: Generalizable 3d gaussian splatting from monocular depth foundation models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21570–21579, 2025.

[14] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision, 2020.

[15] Jongmin Park, Minh-Quan Viet Bui, Juan Luis Gonzalez, Jaeho Moon, Jihyong Oh, and Munchurl Kim. Ecosplat: Eficiencycontrollable feed-forward 3d gaussian splatting from multiview images. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26010–26020, 2026.

[16] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The replica dataset: A digital replica of indoor spaces, 2019.

[17] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[18] Weijie Wang, Donny Y Chen, Zeyu Zhang, Duochao Shi, Akide Liu, and Bohan Zhuang. Zpressor: Bottleneck-aware compression for scalable feed-forward 3dgs. Advances in Neural Information Processing Systems, 38:113407–113436, 2025.

[19] Weijie Wang, Yeqing Chen, Zeyu Zhang, Hengyu Liu, Haoxiao Wang, Zhiyuan Feng, Wenkang Qin, Feng Chen, Zheng Zhu, Donny Y. Chen, and Bohan Zhuang. Volsplat: Rethinking feedforward 3d gaussian splatting with voxel-aligned prediction. In Proceedings of the European Conference on Computer Vision, 2026.

[20] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. Freesplat: Generalizable 3d gaussian splatting towards free view synthesis of indoor scenes. Advances in Neural Information Processing Systems, 37:107326–107349, 2024.

[21] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. Freesplat++: Generalizable 3d gaussian splatting for eficient indoor scene reconstruction, 2025.

[22] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[23] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16453–16463, 2025.

[24] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelNeRF: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[25] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.