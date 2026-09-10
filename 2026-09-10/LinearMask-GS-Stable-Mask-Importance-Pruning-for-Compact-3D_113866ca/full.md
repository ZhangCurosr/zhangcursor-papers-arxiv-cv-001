# LinearMask-GS: Stable-Mask Importance Pruning for Compact 3D Gaussian Splatting

Donghun Ryu<sup>1</sup> donghun0621@cau.ac.kr

Minhyeok Lee<sup>1,2†</sup> mlee@cau.ac.kr

<sup>1</sup> Department of Intelligent Semiconductor Engineering Chung-Ang University Seoul, Republic of Korea

<sup>2</sup> School of Electrical and Electronics Engineering Chung-Ang University Seoul, Republic of Korea

## Abstract

3D Gaussian Splatting (3DGS) enables real-time novel view synthesis but produces millions of primitives through adaptive densification, leading to significant storage overhead. Learned-mask pruning methods such as LP-3DGS address this by assigning each Gaussian a learnable mask to identify and prune redundant primitives. However, we identify a limitation of this paradigm: the steep slope of the Gumbel-Sigmoid activation drives mask values to the extremes within the short mask-training window, before the importance ranking has stabilized, producing a sharply bimodal distribution from which that ranking can no longer be reliably recovered. We propose LinearMask-GS, which replaces Gumbel-Sigmoid with a linear increment activation that keeps mask values in a mid-confidence regime throughout mask training, producing a stable, unimodal mask distribution whose ranking tracks importance. On Mip-NeRF 360, our method achieves 3.6 and 1.6 Gaussian reductions over 3DGS and LP-3DGS, respectively, while maintaining or improving rendering quality. For outdoor scenes, it yields a 1.6 reduction (from 2.18M to 1.36M) with notable gains in PSNR (+0.38 dB), SSIM (+0.025), and LPIPS (-0.029).

## 1 Introduction

3D Gaussian Splatting (3DGS) [13] has become a leading representation for novel view synthesis, achieving high-fidelity real-time rendering by encoding scenes as millions of explicit Gaussian primitives. Despite this rendering efficiency, the adaptive densification process routinely produces millions of primitives, and a standard outdoor scene in Mip-NeRF 360 [1] can exceed 1GB in storage [28], limiting deployment on resource-constrained devices.

Prior methods reduce this overhead either by compressing per-Gaussian attributes [15, 21, 28], which leaves the primitive count and rasterization cost unchanged, or by pruning low-importance Gaussians post hoc [5, 6, 22], which lacks feedback from scene optimization and often discards geometrically essential primitives, yielding suboptimal quality– compression trade-offs (Sec. 2).

LP-3DGS [31] addresses this by learning a per-Gaussian soft mask jointly with the scene, using a Gumbel-Sigmoid activation so the mask can be optimized end-to-end and then thresholded into a hard keep/prune decision. This couples pruning with scene optimization, but we observe that the soft mask it produces is, by the time pruning happens, no longer a faithful summary of importance.

The learned mask saturates before importance ranking stabilizes. The Gumbel-Sigmoid activation used in LP-3DGS is steep: small changes in the underlying parameter swing the mask all the way to 0 or 1, so the mask collapses to these two extreme values long before the per-Gaussian importance signal has settled into a meaningful ranking. This saturation is a property of the activation itself rather than of the brevity of the mask-training window: with the pruning point held fixed, extending the window monotonically worsens saturation (Sec. 3.3). As shown in Fig. 1(a, b), the resulting distribution is sharply twopeaked (i.e., bimodal), with almost every value pinned to either 0 or 1, and which side a Gaussian lands on is largely an artifact of early optimization rather than its true contribution to the rendered image. Pruning a distribution like this is unreliable: the ranking induced by ψ<sub>i</sub> reflects the side of saturation more than the magnitude of importance, and the standard remedy of enforcing a minimum-survival ratio [31] prevents geometry collapse without addressing this underlying cause.

To address this, we propose LinearMask-GS, a learned-mask pruning framework that targets the cause of this brittleness by changing how the mask is shaped during training. We replace Gumbel-Sigmoid with a linear increment activation whose gentle slope keeps mask values in the middle of the [0,1] range, a mid-confidence regime, throughout the mask-training window, rather than letting them saturate at 0 or 1. The resulting distribution (Fig. 1(c, d)) is unimodal, so the relative order of mask values is preserved rather than collapsed onto 0 and 1, and hard pruning therefore operates on a stable ranking rather than a two-peaked one. Because the activation only changes how the mask is shaped, not how it is applied, LinearMask-GS integrates without architectural modifications into a range of Gaussian-based backbones, including 3DGS [13], 2DGS [12], DropGaussian [24], Octree-GS [25], and FastGS [26].

Our main contributions are as follows:

• We identify that the steep activation of learned-mask pruning saturates the soft mask at 0 or 1 before the importance ranking stabilizes, a failure mode that longer masktraining windows worsen rather than fix, yielding a two-peaked distribution from which that ranking can no longer be reliably recovered.

• We propose the linear increment masking activation, whose gentle slope keeps mask values away from the boundaries and yields a unimodal distribution whose ranking is preserved rather than collapsed, on which a simple top-ρN selection gives a reliable hard-pruning decision.

• On Mip-NeRF 360, Tanks & Temples, and Deep Blending, LinearMask-GS achieves better quality–compression trade-offs than existing learned-mask and pruning baselines on 3DGS, and the same masking module transfers without modification to 2DGS, DropGaussian, Octree-GS, and FastGS.

## 2 Related Work

## 2.1 Novel View Synthesis

While Neural Radiance Fields (NeRF) [1, 19, 20] achieve high-quality novel view synthesis, dense ray marching limits their real-time application. 3D Gaussian Splatting (3DGS) [13] overcomes this by combining explicit 3D Gaussian primitives with a fast tile-based rasterizer. Subsequent methods have extended this framework to improve anti-aliasing [29], scaling [17], and surface reconstruction via 2D oriented disks (2DGS) [12]. Despite these advances, adaptive densification leaves all of them with millions of primitives and the attendant memory and storage overhead (Sec. 1).

## 2.2 Gaussian Compression and Pruning

To reduce storage costs without modifying the number of Gaussians, several works compress per-Gaussian attributes via vector quantization, residual quantization of color and rotation, or post-training codecs based on entropy coding and region-adaptive hierarchical transforms [8, 15, 21, 28]. While effective at reducing per-attribute storage, these methods leave the total number of Gaussian primitives unchanged and thus do not alleviate rasterization time complexity.

Pruning methods directly reduce the Gaussian count by removing redundant primitives. The original 3DGS [13] periodically culls Gaussians with low opacity or excessively large scale as part of adaptive density control. LightGaussian [5] prunes low-scoring primitives by a heuristic significance score, and Mini-Splatting [6] reorganizes Gaussian positions via a densification-and-sampling pipeline. RadSplat [22] shows that taking the per-view maximum blending weight as the importance score yields a more stable and discriminative signal than summation-based alternatives. Despite their utility, these heuristic methods rely on manually tuned thresholds and treat pruning as a post-hoc step decoupled from scene geometry optimization, leading to accidental removal of geometrically essential primitives. Our method overcomes these limitations by coupling pruning decisions directly with scene reconstruction through learned importance masks.

## 2.3 Differentiable Gaussian Pruning

To overcome the limitations of manual thresholding, recent work introduces end-to-end differentiable masking frameworks. LP-3DGS [31] assigns a learnable binary mask to each Gaussian and jointly optimizes masks alongside scene parameters, accumulating importance scores by summing pixel-level blending weights and binarizing them through a Gumbel-Sigmoid activation. Taming-3DGS [18] and Reduced-3DGS [23] extend this paradigm with explicit Gaussian count budgets and attribute quantization, respectively, targeting deployment on resource-limited hardware.

While these methods demonstrate the benefit of jointly learning pruning decisions and scene geometry, they inherit summation-based score inflation and, more fundamentally, the premature saturation of the steep Gumbel-Sigmoid mask; our approach addresses both with max-pooling aggregation and a linear-increment activation that restores a stable ranking for the top-ρN rule (Secs. 3.2–3.3).

## 3 Method

We propose a differentiable pruning framework that jointly optimizes per-Gaussian mask parameters alongside scene geometry, compressing Gaussian-based representations while largely preserving PSNR, SSIM, and LPIPS. We develop it on 3DGS [13] and 2DGS [12], and show that it transfers without modification to other backbones (Sec. 4).

After an initial geometric warm-up phase using the standard Gaussian Splatting pipeline, the framework proceeds in three stages: (1) computing a max-pooled importance score $S _ { i }$ for each primitive to evaluate view-count-invariant relevance (Sec. 3.2), (2) jointly optimizing per-Gaussian learnable mask parameters with the proposed linear increment activation, which keeps mask values away from 0 and 1 throughout the mask-training window (Sec. 3.3), and (3) applying a hard-pruning rule that retains the top-ρN Gaussians ranked by $\psi _ { i } ,$ where $\rho$ is the minimum-survival ratio, followed by fine-tuning of the remaining scene (Sec. 3.4).

## 3.1 Preliminaries

3D Gaussian Splatting. 3DGS [13] represents a scene as a set of N anisotropic Gaussian primitives. Each Gaussian i is parameterized by a mean position $\boldsymbol { \mu _ { i } } \in \mathbb { R } ^ { 3 }$ , a covariance matrix $\mathbf { \bar { \Sigma } } _ { i } = \mathbf { R } _ { i } \mathbf { S } _ { i } \mathbf { S } _ { i } ^ { \top } \mathbf { R } _ { i } ^ { \top }$ (decomposed into a rotation matrix $\mathbf { R } _ { i }$ and a diagonal scaling matrix $\mathbf { S } _ { i } )$ , an opacity logit $o _ { i } \in \mathbb { R }$ , and spherical harmonic coefficients encoding view-dependent color $\mathbf { c } _ { i }$

Primitives are projected and depth-sorted to compute color C via front-to-back alpha compositing:

$$
\begin{array} { r } { \mathbf { C } = \displaystyle \sum _ { i = 1 } ^ { N } \mathbf { c } _ { i } \alpha _ { i } \prod _ { j < i } ( 1 - \alpha _ { j } ) , \quad \alpha _ { i } = \sigma ( o _ { i } ) \cdot \exp \left( - \frac { 1 } { 2 } \mathbf { d } ^ { \top } \Sigma _ { i , \mathrm { 2 D } } ^ { - 1 } \mathbf { d } \right) , } \end{array}\tag{1}
$$

Here, d is the 2D offset to the projected center, $\Sigma _ { i , \mathrm { 2 D } }$ is the covariance, and $\sigma ( \cdot )$ is the sigmoid. The per-Gaussian blending weight at pixel p under viewpoint k is:

$$
w _ { i } ^ { k , \mathbf { p } } = \alpha _ { i } ^ { k , \mathbf { p } } \prod _ { j < i } ( 1 - \alpha _ { j } ^ { k , \mathbf { p } } ) ,\tag{2}
$$

which measures the fractional contribution of Gaussian i to the rendered pixel color.

2D Gaussian Splatting. 2DGS [12] replaces 3D ellipsoids with oriented 2D disk primitives embedded in 3D space, improving multi-view geometric consistency. Its compositing follows the same front-to-back formulation as Eq. (1) (with an explicit ray-splat intersection), and the blending weight $w _ { i } ^ { k , \mathbf { p } }$ retains the form of Eq. (2), providing a unified interface for importance score computation across both representations.

## 3.2 Importance Score via Max-Pooling

A key design choice in differentiable Gaussian pruning is how to aggregate per-pixel, perview blending weights into a single scalar importance score $S _ { i }$ for each Gaussian. Several prior methods adopt summation-based aggregation [6, 31]:

$$
S _ { i } ^ { \mathrm { { s u m } } } = \sum _ { k = 1 } ^ { K } \sum _ { \bf { p } } w _ { i } ^ { k , \bf { p } } ,\tag{3}
$$

where the sum runs over all K training viewpoints and all pixels p. This formulation inflates scores for Gaussians in densely observed regions, reducing the discriminability of the importance signal.

To address this, we adopt the max-pooling aggregation proposed by RadSplat [22] to compute importance scores over all training views:

$$
S _ { i } = \operatorname* { m a x } _ { k \in \{ 1 , . . . , K \} } \biggl ( \operatorname* { m a x } _ { \mathbf { p } } w _ { i } ^ { k , \mathbf { p } } \biggr ) .\tag{4}
$$

This captures the peak contribution of a Gaussian in its most favorable viewpoint, providing a view-count-invariant measure of geometric relevance. A Gaussian that contributes strongly to at least one view retains a high score regardless of how many views overlap the same region, effectively mitigating the view-count inflation associated with Eq. (3).

## 3.3 Differentiable Pruning via Linear Increment Masking

Mask formulation. We assign each Gaussian i a learnable scalar parameter $m _ { i } \in \mathbb { R }$ , initialized to 1 to ensure all primitives start fully visible and participate equally in the initial geometric optimization. The per-Gaussian mask value is:

$$
\psi _ { i } = f ( m _ { i } \cdot S _ { i } ) ,\tag{5}
$$

where $f : \mathbb { R } \to [ 0 , 1 ]$ is the mask activation and $S _ { i }$ is the importance score from Eq. (4). During rendering, the opacity of each Gaussian is modulated as:

$$
\tilde { \alpha } _ { i } ^ { k , \mathbf { p } } = \psi _ { i } \cdot \alpha _ { i } ^ { k , \mathbf { p } } ,\tag{6}
$$

so that Gaussians with $\psi _ { i } \approx 0$ are differentiably suppressed while gradients flow through both $m _ { i }$ and the scene parameters concurrently.

Linear increment activation. LP-3DGS [31] employs Gumbel-Sigmoid as $f \colon$

$$
f _ { \mathrm { G S } } ( x ) = \frac { 1 } { 1 + \exp \left( - \frac { \log ( x ) + g _ { 0 } - g _ { 1 } } { t } \right) } , \quad g _ { 0 } , g _ { 1 } \sim \mathrm { G u m b e l } ( 0 , 1 ) ,\tag{7}
$$

where t is the temperature parameter. The steep slope of $f _ { \mathrm { G S } }$ near its midpoint causes a small change in $m _ { i } \cdot S _ { i }$ to produce a large change in $\psi _ { i } .$ , so mask outputs are pushed toward 0 or 1 within the short mask-training window before the importance score $S _ { i }$ has stabilized. As shown in Fig. 1(a, b), this yields a two-peaked $\psi _ { i }$ distribution pinned at 0 and 1. We refer to this behavior as mask instability: which extreme each mask lands on reflects an artifact of early optimization more than the magnitude of $S _ { i }$ itself. This instability stems from the activation’s steepness rather than from the window length: with the pruning point fixed at iteration 20k, extending the Gumbel-Sigmoid mask window from 500 to 2,500 and 5,000 iterations monotonically increases the saturated fraction $( 5 1 . 1 \%  5 4 . 8 \%  5 6 . 2 \%$ on three scenes) while PSNR plateaus, whereas the linear ranking is robust to window placement (window starts of 15k/17.5k/19.5k change final PSNR by $\leq 0 . 1 4 ~ \mathrm { d B } )$ ; see Sec. 4 of the supplementary material.

We therefore propose the linear increment activation:

$$
f _ { \mathrm { L I } } ( x ; \tau ) = \mathrm { c l i p } ( \tau \cdot ( x - 0 . 5 ) + 0 . 5 , 0 , 1 ) ,\tag{8}
$$

where $\tau > 0$ is a slope hyperparameter. With a small slope (we use $\tau { = } 0 . 1 )$ , $f _ { \mathrm { L I } }$ has a wide unsaturated range and is monotone-linear in its active interval, so a change in $m _ { i } \cdot S _ { i }$ produces a proportionally small change in $\psi _ { i }$ . As a result, mask values stay close to the neutral midpoint 0.5 throughout mask training and the output distribution remains unimodal and centered away from 0 and 1, as shown in Fig. 1(c, d). We refer to this property as mask stability: because the values stay spread across the mid-range instead of collapsing to a binary indicator, their relative order is preserved rather than compressed onto the saturation points, so $\psi _ { i }$ yields a stable ranking for pruning. Quantitatively, across the nine Mip-NeRF 360 scenes at iteration 20,000, the Linear mask distribution shows 0.00% saturation $( \psi _ { i } < 0 . 1$ or $\psi _ { i } > 0 . 9 )$ and 99.99% of values lying in the interval [0.3,0.7], with a per-Gaussian standard deviation of only 0.018. In contrast, Gumbel-Sigmoid saturates 51.19% of mask values and retains only 20.28% in [0.3,0.7], with a much larger standard deviation of 0.380.

Hard pruning via $\mathbf { t o p } { \cdot } \rho N$ selection. At the end of mask training, we convert the soft mask values into a hard keep/prune decision. Because the linear-increment activation keeps $\psi _ { i }$ unimodal and centered around 0.5, the relative order of mask values is preserved rather than compressed onto the saturation points, so $\psi _ { i }$ provides a stable ranking. We exploit this by directly retaining the top-ρN Gaussians by $\psi _ { i } .$ , where $\rho = 0 . 3 0$ is the minimum-survival ratio and N the total number of Gaussians:

$$
\mathcal { K } = \mathrm { t o p } { - } \rho N \mathrm { G a u s s i a n s ~ r a n k e d ~ b y ~ } \psi _ { i } .\tag{9}
$$

Gaussians in $\kappa$ are retained and the rest are permanently pruned. This rule plays the same role as the minimum-survival ratio in LP-3DGS (bounding the kept fraction to prevent geometry collapse), but because $\psi _ { i }$ here is unimodal and its rank tracks importance, the kept set follows the underlying importance ranking instead of the early saturation outcome.

Throughout the mask-training phase, the soft mask value $\psi _ { i } \in [ 0 , 1 ]$ produced by the linear-increment activation is applied directly as an opacity multiplier via Eq. (6), so gradients propagate through both $m _ { i }$ and the scene parameters without hard discretization. The hard-pruning rule of Eq. (9) is applied only once at the end of mask training, as described in Sec. 3.4.

## 3.4 Training Objective and Schedule

The total training loss combines photometric reconstruction and mask sparsity:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \left( 1 - \lambda _ { D } \right) \mathcal { L } _ { 1 } + \lambda _ { D } \left( 1 - \mathrm { S S I M } \right) + \lambda _ { m } \mathcal { L } _ { \mathrm { m a s k } } ,\tag{10}
$$

where $\mathcal { L } _ { 1 }$ is the pixel-wise $\ell _ { 1 }$ loss, SSIM is the structural similarity metric [27], and $\lambda _ { D } = 0 . 2$ follows the default 3DGS setting [13]. The sparsity term follows LP-3DGS [31]:

$$
\mathcal { L } _ { \mathrm { m a s k } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lvert m _ { i } \rvert ,\tag{11}
$$

which encourages the mask parameters $m _ { i }$ to shrink during training, thereby reducing the corresponding soft mask values after scaling by $S _ { i }$ . We set $\lambda _ { m } = 5 \times 1 0 ^ { - 4 }$ and $\tau = 0 . 1$ for the linear increment activation, using a separate AdamW optimizer [16] for $\left\{ m _ { i } \right\}$ at a learning rate of $5 \times 1 0 ^ { - 2 }$

![](images/ba415dbfdc0698473e1c6cc046c37e5b72897e3b7209dc67a1a96d0d4757a27b.jpg)  
(a) Gumbel-Sigmoid, outdoor scenes

![](images/5ae16fd61fb7c9d97363634fbd6766615dd35b48942c314fdf1a296da6251e52.jpg)  
(b) Gumbel-Sigmoid, indoor scenes

![](images/37a6f8e72e3d05da6dc97ab02dfd29a5b587532b59af178af8484cf8dbd3b833.jpg)  
(c) Linear increment (Ours), outdoor scenes

![](images/583deee7a4c96c089876669d617d88b49b86c9ce450a04447df7c496c5cca5a2.jpg)  
(d) Linear increment (Ours), indoor scenes  
Figure 1: Mask $\psi _ { i }$ distribution at the end of mask training, aggregated separately over Mip-NeRF 360’s outdoor (5 scenes) and indoor (4 scenes) sets. (a, b) Gumbel-Sigmoid produces a two-peaked distribution pinned at 0 and 1; the fraction above $\psi = 0 . 5$ is scene-dependent ( 52% outdoor, 40% indoor), reflecting that which side each mask lands on is unrelated to importance. (c, d) Linear increment yields a unimodal distribution near $\psi = 0 . 5$ (mean 0.44 indoor, 0.49 outdoor), so the relative order of mask values is preserved and the top-$\rho N$ pruning in Eq. (9) operates on a stable ranking.

Optimization schedule. Training proceeds for 30,000 iterations in total. For both the 3DGS and 2DGS backbones, we warm up the standard pipeline for 19,500 iterations to ensure geometric convergence before activating the mask module. During the subsequent 500- iteration mask training phase (iterations 19,500–20,000), the max-pooling importance scores $S _ { i }$ from Eq. (4) are recomputed every 20 iterations to reflect ongoing geometric changes. At iteration 20,000, we apply the hard-pruning rule of Eq. (9) introduced in Sec. 3.3: the top-ρN Gaussians ranked by $\psi _ { i }$ are kept, where $\rho$ is the minimum-survival ratio. We use $\rho = 0 . 3 0$ as the default and study its effect in Sec. 4. The remaining primitives are fine-tuned without the mask loss for the remainder of training.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate LinearMask-GS on three novel view synthesis benchmarks: Mip-NeRF 360 [1] (9 scenes), Tanks and Temples [14] (2 outdoor scenes), and Deep Blending [11] (2 indoor scenes).

Baselines. We compare LinearMask-GS against generic novel view synthesis and scene representation baselines (Plenoxels [7], INGP [20], Mip-NeRF 360 [1], 3DGS [13], and 2DGS [12]), as well as 3DGS-based pruning and compression methods (Compact3DGS [15], LightGaussian [5], EAGLES [8], RadSplat [22], Mini-Splatting [6], and LP-3DGS [31]). In Table 1, we additionally report a retrained baseline (3DGS∗) to provide a fair comparative reference trained under our identical hardware and schedule. Note that for LP-3DGS, we evaluate its official implementation but apply max-pooling aggregation instead of its original summation-based method. This provides a stronger baseline and ensures a fair comparison by strictly isolating the effect of the mask activation function.

Evaluation metrics. Rendering quality is assessed via PSNR, SSIM [27], and LPIPS [30].   
Compression efficiency is measured by the surviving Gaussian count and storage size (MB).

Implementation details. We implement LinearMask-GS in PyTorch, building upon the official 3DGS codebase [13]. The slope τ=0.1 (grid search over $\{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 \} )$ , minimum-survival ratio $\rho { = } 0 . 3 0$ , and mask weight $\lambda _ { m } { = } 5 \times 1 0 ^ { - 4 }$ are tuned on a held-out set and fixed across all scenes; sensitivity analyses are provided in the supplementary material. Training follows the standard 30,000-iteration 3DGS schedule. We train the 3DGSand 2DGS-backbone runs on a single NVIDIA RTX 3090 GPU, and the FastGS-backbone run (LinearMask-GS<sup>⋆</sup>) on a single NVIDIA RTX 4090 GPU, following the hardware setup of the FastGS paper [26]. For a fair wall-clock comparison, the training-time measurements in Table 2 are all collected on a single RTX 4090.

## 4.2 Main Results

We evaluate LinearMask-GS against scene representation and pruning baselines on three standard benchmarks: Mip-NeRF 360 [1], Tanks & Temples [14], and Deep Blending [11]. For all datasets, we follow the standard evaluation protocol of 3DGS [13], where every eighth image is reserved for testing. We report PSNR (dB), SSIM, LPIPS, and the number of Gaussians (M) on all benchmarks, plus FPS and storage (MB) on Mip-NeRF 360 (measured on the same hardware for reproduced methods, and taken from the original papers for literature baselines).

Mip-NeRF 360. Table 1 summarizes results across all nine scenes against three families of baselines: the original 3DGS backbone with its retrained reference and the LP-3DGS learned-mask parent; compression- and pruning-oriented systems; and recent fast-training pipelines. Compared to the vanilla 3DGS [13] backbone, our method reduces the Gaussian count from 3.36M to 0.94M (a 3.6 reduction) while improving PSNR from 27.21 to 27.70 dB. Against our primary pruning baseline LP-3DGS [31], LinearMask-GS surpasses it by 0.23 dB in PSNR and 0.0135 in SSIM with 1.6 fewer Gaussians, while maintaining comparable rendering speed (591 vs. 588 FPS).

Compared to recent compression methods (HAC [2], HAC++ [3]) and aggressive pruning baselines (Mini-Splatting [6], Speedy-Splat [9]), LinearMask-GS achieves competitive PSNR with substantially fewer Gaussians than compression methods while maintaining higher quality than the most aggressive pruning baselines (see Table 1 for details). For a budget-matched comparison with the most aggressive pruning baseline, we additionally constrain LinearMask-GS<sup>⋆</sup> to Speedy-Splat’s budget of 0.30M Gaussians: over three independent seeds it reaches 27.40 dB / 0.798 / 0.230, versus 26.91 / 0.781 / 0.295 for Speedy-Splat, i.e., better on all three metrics at the same budget, and still ahead at 0.25M (27.06 / 0.791 / 0.241); see Sec. 11 of the supplementary material. Applying LinearMask-GS on the FastGS backbone $( \mathrm { L i n e a r M a s k - G S } ^ { \star }$ , last row), where LinearMask-GS substitutes FastGS’s post-densification VCP module, further compresses the FastGS output from 0.40M to 0.38M Gaussians while improving PSNR by 0.13 dB over FastGS (27.69 vs. 27.56 dB) and reducing storage to 94.89 MB. This indicates that our masking strategy can replace specialized post-densification pruning components in fast-training pipelines while improving rendering quality.

Figure 3 summarizes this advantage through efficiency metrics across all three quality dimensions $( 1 0 ^ { 9 } / ( \mathrm { L P I P S } \times \# \mathrm { G } )$ , $\mathrm { P S N R } / \# \mathrm { G } , \mathrm { S S I M } / \# \mathrm { G } )$ , on which LinearMask-GS achieves the highest score against the 3DGS and LP-3DGS reference rows; a per-scene LPIPS-vs-#G visualization is provided in Sec. 5 of the supplementary material.

Tanks & Temples and Deep Blending. On the Tanks & Temples and Deep Blending benchmarks, LinearMask-GS<sup>⋆</sup> (FastGS-backbone variant that substitutes FastGS’s postdensification VCP module with our learned mask) reaches 24.18 dB PSNR and 0.825 SSIM with 0.29M Gaussians on Tanks & Temples, and 30.11 dB PSNR and 0.899 SSIM with 0.29M Gaussians on Deep Blending. This places LinearMask- $\mathbf { \cdots } \mathbf { G } \mathbf { S } ^ { \star }$ in the sub-million primitive regime of recent fast-training baselines while achieving the best PSNR on Tanks & Temples average (24.18 dB), Dr Johnson (29.67 dB), and Deep Blending average (30.11 dB), surpassing FastGS and other fast-training methods on these metrics. The supplementary material provides the full quantitative comparison against recent fast-training and compact 3DGS baselines, together with per-scene score tables for both benchmarks.

Qualitative comparison. Figure 2 shows per-pixel absolute error maps for three representative Mip-NeRF 360 scenes (Bicycle, Flowers, Counter); error maps for the remaining six scenes are provided in the supplementary material. For each scene, error maps are visualized under a scene-specific colormap (blue to red indicates low to high absolute error), and per-image PSNR and MAE are reported below each panel. Across all nine scenes, LinearMask-GS produces lower MAE and fewer saturated high-error (red) regions than both LP-3DGS and 3DGS. The improvement is most visible in texture-dense outdoor scenes such as Flowers and Stump, where 3DGS error maps show large red areas around the foliage and 3DGS PSNR drops to 20.19 and 24.94 dB respectively, while LinearMask-GS retains 23.49 and 27.07 dB with substantially cleaner error maps. In indoor scenes such as Counter and Kitchen, the per-image PSNR margin over 3DGS exceeds 2.7 dB, reflecting more accurate reconstruction of fine structures like the countertop and the toy loader despite using fewer Gaussians overall (Table 1).

Table 1: Quantitative results on Mip-NeRF 360. All 3DGS- and 2DGS-backbone results are obtained under the same RTX 3090 environment as the baselines for a like-for-like comparison; the FastGS-backbone variant (⋆) follows the FastGS hardware setup (RTX 4090). Baselines: (i) 3DGS and its retrained reference 3DGS∗, (ii) LP-3DGS evaluated with maxpooling for a matched comparison (Sec. 4.4), and (iii) recent compression, pruning, and fast-training methods. ⋆: our method as a VCP substitute on FastGS. “–”: metric not reported. Best per column in bold.
<table><tr><td>Method</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>#G (M)↓</td><td>FPS↑</td><td>Storage (MB)↓</td></tr><tr><td>Plenoxels [0]</td><td>0.463</td><td>23.08</td><td>0.626</td><td>一</td><td>6.79</td><td>2100.0</td></tr><tr><td>INGP-base []</td><td>0.371</td><td>25.30</td><td>0.671</td><td>1</td><td>11.70</td><td>13.0</td></tr><tr><td>INGP-big [四]</td><td>0.331</td><td>25.59</td><td>0.699</td><td>一</td><td>9.43</td><td>48.0</td></tr><tr><td>Mip-NeRF 360 [0]</td><td>0.237</td><td>27.69</td><td>0.792</td><td>-</td><td>0.06</td><td>8.6</td></tr><tr><td>3DGS []</td><td>0.214</td><td>27.21</td><td>0.815</td><td>3.36</td><td>134.00</td><td>734.0</td></tr><tr><td>3DGS*</td><td>0.222</td><td>27.46</td><td>0.812</td><td>3.35</td><td>120.00</td><td>746.0</td></tr><tr><td>LP-3DGS []</td><td>0.2258</td><td>27.47</td><td>0.8119</td><td>1.48</td><td>588.00</td><td>314.12</td></tr><tr><td>Compression-oriented</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Compact3DGS []</td><td>0.247</td><td>27.08</td><td>0.798</td><td>1.39</td><td>128.00</td><td>48.8</td></tr><tr><td>EAGLES []</td><td>0.238</td><td>27.16</td><td>0.809</td><td>1.71</td><td>131.00</td><td>54.0</td></tr><tr><td>HAC (high-rate) [0]</td><td>0.2296</td><td>27.77</td><td>0.8109</td><td>2.26</td><td>一</td><td>22.94</td></tr><tr><td>HAC++ (high-rate) []</td><td>0.2307</td><td>27.82</td><td>0.8109</td><td>1.85</td><td>一</td><td>19.38</td></tr><tr><td>Pruning-oriented</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LightGaussian []</td><td>0.231</td><td>26.54</td><td>0.798</td><td>1.05</td><td>189.00</td><td></td></tr><tr><td>RadSplat []</td><td>0.218</td><td>27.08</td><td>0.811</td><td>1.92</td><td>165.00</td><td>154.5</td></tr><tr><td>Mini-Splatting []</td><td>0.217</td><td>27.34</td><td>0.822</td><td>0.49</td><td>178.00</td><td>189.2 132.8</td></tr><tr><td>PUP 3D-GS []</td><td>0.2719</td><td>26.67</td><td>0.7862</td><td></td><td>204.81</td><td>74.65</td></tr><tr><td>Speedy-Splat []</td><td>0.295</td><td>26.91</td><td>0.781</td><td>0.30</td><td>552.00</td><td>一</td></tr><tr><td>Fast-training systems</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Taming 3DGS []</td><td>0.261</td><td>27.48</td><td>0.794</td><td>0.68</td><td>221.00</td><td></td></tr><tr><td>DashGaussian []</td><td>0.218</td><td>27.73</td><td>0.817</td><td>2.40</td><td></td><td>一</td></tr><tr><td>FastGS []</td><td>0.261</td><td>27.56</td><td>0.797</td><td>0.40</td><td>155.00 579.00</td><td>99.44</td></tr><tr><td>LinearMask-GS (Ours)</td><td>0.2095</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LinearMask-GS* (FastGS backbone)</td><td>0.2227</td><td>27.70 27.69</td><td>0.8254 0.8029</td><td>0.94 0.38</td><td>591.14 344.60</td><td>226.76 94.89</td></tr></table>

Training time. Table 2 reports mean training time over the nine Mip-NeRF 360 scenes, all measured on a single RTX 4090. On the 3DGS backbone, LinearMask-GS is 2.12 minutes faster than its learned-mask parent LP-3DGS while reaching a much more compact final model with higher PSNR and SSIM (0.94M vs. 1.48M Gaussians, Table 1), and incurs only a 0.9-minute overhead over the unpruned 3DGS baseline despite the added mask-training and fine-tuning stages. On the FastGS backbone, LinearMask-GS<sup>⋆</sup> reduces training time (1.93 to 1.81 minutes) while removing additional primitives: the brief mask-learning stage is more than offset by faster iterations on the already-pruned model, yielding a net saving; a full timing breakdown is provided in Sec. 13 of the supplementary material.

## 4.3 Generality across Backbones

2DGS. We apply LinearMask-GS to the official 2DGS [12] codebase to test the generality of our framework across different Gaussian representations. While 3DGS and 2DGS utilize different geometric primitives, they share a consistent alpha-compositing pipeline, allowing our max-pooling score and linear activation to be applied without structural changes.

As shown in Table 3, on the nine Mip-NeRF 360 scenes LinearMask-GS reduces the

Table 2: Training-time comparison on Mip-NeRF 360 (minutes, averaged over nine scenes). All rows are measured on a single RTX 4090. ⋆: our method on FastGS.
<table><tr><td>Method</td><td>Train Time (min)↓</td></tr><tr><td>3DGS []</td><td>12.10</td></tr><tr><td>LP-3DGS []</td><td>15.12</td></tr><tr><td>LinearMask-GS (Ours)</td><td>13.00</td></tr><tr><td>Taming 3DGS [8]</td><td>5.36</td></tr><tr><td>DashGaussian []</td><td>6.35</td></tr><tr><td>FastGS [[6]</td><td>1.93</td></tr><tr><td>LinearMask.  $\mathbf { \sigma } . \mathbf { G S ^ { \star } }$  (FastGS backbone)</td><td>1.81</td></tr></table>

![](images/45ea6a4bc24bc9f7aba87d055be50b4094489ea8fc7e8f08003a1425762bddba.jpg)  
Figure 2: Per-scene qualitative comparison on three representative Mip-NeRF 360 scenes (Bicycle, Flowers, Counter). Each row: GT, renders from LinearMask-GS / LP-3DGS / 3DGS (PSNR below), then per-pixel absolute error maps for the same three (MAE below; scene-specific blue-to-red colormap). LinearMask-GS consistently shows lower error and fewer saturated red regions; remaining six scenes in the supplement.

![](images/559029c5eb02c042e616b57dffc78f3f6e0f8b65bf162b3f7cc708f3040bebd5.jpg)

![](images/53c4fac327fb168f15a3c0674d5bb99e0255e9da00b6a91584c59ac65641cb31.jpg)

![](images/deb7291770523f04d29c9abe1e8081ff18f73ab242cff7272aff842e73f4ade2.jpg)  
Figure 3: Efficiency scores on Mip-NeRF 360 across three metrics $( 1 0 ^ { 9 } / ( \mathrm { L P I P S } \times \# \mathrm { G } )$ PSNR/#G, SSIM/#G). LinearMask-GS achieves the highest score on all three, indicating superior perceptual quality per Gaussian.

2DGS Gaussian count by 2.9 (from 1.06M to 0.36M) while improving PSNR by 0.22 dB and SSIM by 0.009, with LPIPS rising only marginally (by 0.008). This indicates that our importance masking successfully identifies redundant 2D disks without sacrificing perceptual quality.

DropGaussian and Octree-GS. Beyond 2DGS, we further test LinearMask-GS on two additional Gaussian-based backbones with different representational structures: DropGaussian [24], which regularizes 3D Gaussians through stochastic dropout, and Octree-GS [25], which uses hierarchical anchor structures. On DropGaussian, we evaluate five LLFF scenes (9-view) with the masked model repeated over three independent training runs. Averaged over the runs, LinearMask-GS improves PSNR on every scene, by +0.31 to +0.52 dB over the unpruned backbone, while retaining 41% fewer primitives than the Gumbel variant (perscene results in Sec. 12 of the supplementary material). The singlefern run of Table 3 gives a smaller gain (+0.03 dB at a 2.8 reduction), indicating that single runs of this sparseview backbone are noisy. On Octree-GS, the amsterdam scene from Bungee-NeRF shows a 2.9 anchor reduction (1.09M to 0.38M) with +0.12 dB PSNR and SSIM/LPIPS within 0.002 of the reference; on the bicycle and garden scenes of Mip-NeRF 360, anchors drop by 40.2% relative to the Gumbel variant at a mean PSNR change of 0.11 dB. Across all three Octree-GS scenes, quality thus moves by at most 0.1 dB in either direction while anchors are reduced by 40% or more. Table 3 consolidates these results alongside the 2DGS comparison, and additionally reports the 2DGS variant with the linear-increment activation replaced by Gumbel-Sigmoid; the activation-level comparison across backbones is analyzed in Sec. 4.4.

Table 3: Generality of LinearMask-GS across alternative Gaussian-based backbones, applied as a downstream pruning stage. #Primitives counts Gaussians for 2DGS and Drop-Gaussian, and anchors for Octree-GS. 2DGS results average over the nine Mip-NeRF 360 scenes; DropGaussian usesfern (LLFF) and Octree-GS uses amsterdam (Bungee-NeRF). “+ LP-3DGS (Gumbel)” matches “+ LinearMask-GS” except in activation (see Sec. 4.3). Best per column in bold.
<table><tr><td>Backbone</td><td>Method</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>#Primitives↓</td></tr><tr><td rowspan="3">2DGS []</td><td>2DGS (no pruning)</td><td>0.2586</td><td>26.7390</td><td>0.7935</td><td>1,063,924</td></tr><tr><td>+ LP-3DGS (Gumbel)</td><td>0.2734</td><td>26.9261</td><td>0.8022</td><td>628,840</td></tr><tr><td>+ LinearMask-GS*</td><td>0.2663</td><td>26.9583</td><td>0.8025</td><td>364,448</td></tr><tr><td rowspan="3">DropGaussian [[4]</td><td>DropGaussian (no pruning)</td><td>0.1106</td><td>26.6495</td><td>0.8806</td><td>104,877</td></tr><tr><td>+ LP-3DGS (Gumbel)</td><td>0.1098</td><td>26.6658</td><td>0.8813</td><td>62,182</td></tr><tr><td>+ LinearMask-GS</td><td>0.1087</td><td>26.6756</td><td>0.8815</td><td>37,433</td></tr><tr><td rowspan="3">Octree-GS [[]</td><td>Octree-GS (no pruning)</td><td>0.0917</td><td>27.8570</td><td>0.9184</td><td>1,094,939</td></tr><tr><td>+ LP-3DGS (Gumbel)</td><td>0.0908</td><td>27.9280</td><td>0.9192</td><td>648,221</td></tr><tr><td>+ LinearMask-GS</td><td>0.0928</td><td>27.9720</td><td>0.9191</td><td>380,726</td></tr></table>

## 4.4 Ablation Study

We ablate the core design choices of LinearMask-GS on the Mip-NeRF 360 dataset: the importance aggregation, the mask activation, and the minimum-survival ratio $\rho _ { \cdot }$ . Unless stated otherwise, all variants use the default configuration: $\lambda _ { m } = 5 \times 1 0 ^ { - 4 } , \tau = 0 . 1$ , and $\rho =$ 0.30. Detailed analyses of training convergence and additional hyperparameter sensitivity (sparsity weight $\lambda _ { m }$ and slope τ) are provided in the supplementary material.

Aggregation and activation. Table 4 presents a full factorial comparison of summationbased [6] and max-pooling [22] aggregation against Gumbel-Sigmoid [31] and linear increment activations. Note that the “Max + Gumbel-Sigmoid” variant corresponds exactly to the LP-3DGS baseline reported in Table 1. For the summation-based variants (rows 1–2), we use $\rho = 0 . 4 0$ following the pruning statistics of LP-3DGS [31]; max-pooling variants use the default $\rho = 0 . 3 0$ . Under max-pooling, replacing Gumbel-Sigmoid with the linear increment activation reduces the Gaussian count from 1.48M to 0.94M and improves both PSNR (27.47 to 27.70 dB) and LPIPS (0.2258 to 0.2095). Under summation, the linear-increment variant reduces the count from 1.43M to 1.26M with nearly identical PSNR/LPIPS, indicating that the view-count inflation introduced by summation aggregation dominates the effect of the activation choice and obscures the stability benefit that Linear provides. The combination of max-pooling aggregation and the linear-increment activation gives the best operating point and is used in LinearMask-GS.

Ranking reliability. The premise of top-ρN pruning is that the ranking induced by ψ<sub>i</sub> tracks importance, so we test this directly: with hard pruning disabled, the mask ranking through the window is compared against a held-out proxy, the max-pooled contribution of each Gaussian (Eq. (4)) at iteration 30k (protocol in Sec. 3 of the supplementary material). Across all nine Mip-NeRF 360 scenes, at the pruning decision point: (i) re-sampling the Gumbel noise alone flips 47.3% of keep/prune decisions (46–48% on every scene; rank correlation between two noise draws $< 0 . 1 )$ , so the sampled decision is governed by activation noise rather than by a stable learned ranking, whereas the linear mask is deterministic (0% flips); (ii) 52.2% of Gumbel-Sigmoid mask values are saturated under this protocol versus 0.00% for linear, consistent with the 51.19% observed under the standard schedule (Sec. 3.3); (iii) the Spearman correlation between the mask ranking and the proxy is 0.29 for linear versus 0.16 for Gumbel-Sigmoid (1.8 higher overall and 2.8 on outdoor scenes; per-iteration curves in the supplementary material).

Contribution of the learned mask. To isolate what the mask adds over its ingredients, we also prune directly by the raw score: same schedule, top-30% by $S _ { i }$ at 20k, then identical fine-tuning (“Max + none” in Table 4). This mask-free baseline is strong (27.57 dB / 0.8231 / 0.2143 at 0.96M), yet our learned mask is ahead on eight of the nine scenes at a smaller budget (+0.13 dB at 0.94M; the only exception is Treehill; per-scene results in Sec. 10 of the supplement). The Gumbel-Sigmoid mask falls below it (27.47 dB, 1.48M): once the ranking saturates, a learned mask is worse than none. With the rankingreliability analysis above, this locates the gain: not max-pooling or fine-tuning, both shared with these baselines, but the stability of the ranking itself.

Minimum-survival ratio $\rho .$ . We sweep $\rho \in \{ 0 . 1 0 , 0 . 2 0 , 0 . 3 0 , 0 . 4 0 , 0 . 5 0 \}$ on all nine Mip-NeRF 360 scenes for both the linear-increment activation and the matched Gumbel-Sigmoid baseline under the identical top-ρN rule of Eq. (9) (Secs. 8–9 of the supplementary material). Two observations follow. First, the two quality–compression curves share the same shape: quality improves steeply up to $\rho { = } 0 . 3 0$ and largely saturates beyond it (over $\rho { = } 0 . 3 0 {  } 0 . 5 0$ PSNR gains total +0.04 dB for linear and +0.11 dB for Gumbel-Sigmoid). Within the linear sweep, $\rho { = } 0 . 2 0$ removes too many meaningful primitives ( 0.28 dB), whereas $\rho { = } 0 . 4 0$ retains 34% more Gaussians for only +0.03 dB, so we keep $\rho { = } 0 . 3 0$ as the default. Second, the curves differ in position: because top-ρN retains the same fraction of a near-identical primitive pool, equal ρ implies matched budgets (#G differs by $\leq 0 . 0 1  { \mathbf { M } } )$ , and at every ρ the linear-increment activation sits +0.40–0.48 dB above the Gumbel-Sigmoid baseline at the same budget. The gap does not close anywhere on the sweep: the baseline’s best point (27.33 dB at $\rho { = } 0 . 5 0 , 1 . 5 7 \mathrm { M } )$ falls below even the linear $\rho { = } 0 . 2 0$ setting (27.42 dB at 0.63M, 2.5 fewer primitives) and never reaches our default (27.70 dB at 0.94M) despite 67% more primitives. The stable ranking thus does not reshape the trade-off; it selects a better subset at every budget, shifting the entire curve.

Table 4: Full factorial ablation of importance aggregation and mask activation on Mip-NeRF 360, extended with a mask-free baseline that prunes directly by the raw max-pooled score Si (“none”). ✓ marks the components used in LinearMask-GS. Best results are in bold.
<table><tr><td>Aggregation</td><td>Activation</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>#G (M)↓</td></tr><tr><td>Sum [θ]</td><td>Gumbel-Sigmoid []</td><td>27.12</td><td>0.805</td><td>0.239</td><td>1.43</td></tr><tr><td>Sum [θ]</td><td>Linear increment</td><td>27.05</td><td>0.806</td><td>0.238</td><td>1.26</td></tr><tr><td>Max []</td><td>None (top-ρN by S)</td><td>27.57</td><td>0.8231</td><td>0.2143</td><td>0.96</td></tr><tr><td>Max []</td><td>Gumbel-Sigmoid []</td><td>27.47</td><td>0.8119</td><td>0.2258</td><td>1.48</td></tr><tr><td>Max √[]</td><td>Linear increment √</td><td>27.70</td><td>0.8254</td><td>0.2095</td><td>0.94</td></tr></table>

Activation choice across backbones. The factorial ablation above is conducted on the 3DGS backbone. To verify that the activation itself, rather than the masking framework, drives the gain, we compare the two activations on DropGaussian and Octree-GS with the mask window, sparsity loss, importance score, and target keep ratio held identical (the “+ LP-3DGS (Gumbel)” vs. “+ LinearMask-GS” rows of Table 3). The linear-increment activation prunes 40% (DropGaussian, fern) and 41% (Octree-GS, amsterdam) further than the Gumbel-Sigmoid variant while maintaining or improving quality; since these backbones use markedly different mask-training schedules (1,500 and 4,000 iterations), the activation choice, not merely the masking framework, governs the achievable quality–compression trade-off.

## 5 Conclusion

In this work, we identified a limitation of existing learned-mask pruning for 3D Gaussian Splatting and introduced LinearMask-GS to address it. The steep slope of Gumbel-Sigmoid pushes mask values to 0 or 1 before the importance ranking stabilizes, a failure that longer mask-training windows exacerbate rather than fix, producing a two-peaked distribution from which that ranking can no longer be reliably recovered. Replacing it with a linear-increment activation keeps mask values in the mid-confidence regime and preserves their ranking, so a simple top-ρN selection suffices, and this single change transfers without modification across diverse Gaussian-based backbones.

Limitations. The slope τ and ratio ρ are tuned on a held-out set (Sec. 4.1) and fixed across scenes; the brief mask window can leave specular or transparent regions under-discriminated, and the global ρ sets an overall rather than per-region budget. The method also inherits the static-scene assumption of its backbones; scene-adaptive scheduling and dynamic-scene extensions are left for future work.

## Acknowledgments

This work was partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2024-00337250), by the Korea Institute for Advancement of Technology (KIAT) grant funded by the Korea government (MOTIE) (P0023718, Inorganic Light-emitting Display Expert Training Program for Display Technology Transition), and by the Institute of Information & Communications Technology Planning & Evaluation (IITP, AI Computing Support Project for R&D) grant funded by the Korea government (MSIT) (RS-2026-25505492, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale).

## References

[1] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In CVPR, pages 5470–5479, 2022.

[2] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. HAC: Hash-grid assisted context for 3D gaussian splatting compression. arXiv preprint arXiv:2403.14530, 2024.

[3] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. HAC++: Towards 100x compression of 3D gaussian splatting. arXiv preprint arXiv:2501.12255, 2025.

[4] Youyu Chen, Junjun Jiang, Kui Jiang, Xiao Tang, Zhihao Li, Xianming Liu, and Yinyu Nie. DashGaussian: Optimizing 3D gaussian splatting in 200 seconds. arXiv preprint arXiv:2503.18402, 2025.

[5] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. LightGaussian: Unbounded 3D gaussian compression with 15x reduction and 200+ FPS. In NeurIPS, pages 140138–140158, 2024.

[6] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In ECCV, pages 165–181, 2024.

[7] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, pages 5501–5510, 2022.

[8] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. EAGLES: Efficient accelerated 3D gaussians with lightweight encodings. In ECCV, pages 54–71, 2024.

[9] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. Speedy-Splat: Fast 3D gaussian splatting with sparse pixels and sparse primitives. arXiv preprint arXiv:2412.00578, 2025.

[10] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana, Matthias Zwicker, and Tom Goldstein. PUP 3D-GS: Principled uncertainty pruning for 3D gaussian splatting. arXiv preprint arXiv:2406.10219, 2025.

[11] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. Deep blending for free-viewpoint image-based rendering. ACM TOG, 37(6):1–15, 2018.

[12] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH Conference Papers, pages 1–11, 2024.

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3D gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4):139–1, 2023.

[14] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM TOG, 36(4):1–13, 2017.

[15] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3D gaussian splatting for static and dynamic radiance fields. arXiv preprint arXiv:2408.03822, 2024.

[16] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[17] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-GS: Structured 3D gaussians for view-adaptive rendering. In CVPR, pages 20654–20664, 2024.

[18] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3DGS: High-quality radiance fields with limited resources. In ACM SIGGRAPH Asia Conference Papers, pages 1–11, 2024.

[19] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

[20] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 41(4):1–15, 2022.

[21] Simon Niedermayr, Josef Stumpfegger, and Rüdiger Westermann. Compressed 3D gaussian splatting for accelerated novel view synthesis. In CVPR, pages 10349–10358, 2024.

[22] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari. RadSplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+ FPS. In International Conference on 3D Vision (3DV), pages 134–144, 2025.

[23] Panagiotis Papantonakis, Georgios Kopanas, Bernhard Kerbl, Alexandre Lanvin, and George Drettakis. Reducing the memory footprint of 3D gaussian splatting. Proceed ings ofthe ACM on Computer Graphics and Interactive Techniques, 7(1):1–17, 2024.

[24] Hyunwoo Park, Gun Ryu, and Wonjun Kim. DropGaussian: Structural regularization for sparse-view gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21600–21609, 2025.

[25] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-GS: Towards consistent real-time rendering with LOD-structured 3D gaussians. arXiv preprint arXiv:2403.17898, 2024.

[26] Shiwei Ren, Tianci Wen, Yongchun Fang, and Biao Lu. FastGS: Training 3D gaussian splatting in 100 seconds. arXiv preprint arXiv:2511.04283, 2025.

[27] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004.

[28] Shuzhao Xie, Weixiang Zhang, Chen Tang, Yunpeng Bai, Rongwei Lu, Shijia Ge, and Zhi Wang. MesonGS: Post-training compression of 3D gaussians via efficient attribute transformation. In ECCV, pages 434–452, 2024.

[29] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mipsplatting: Alias-free 3D gaussian splatting. In CVPR, pages 19447–19456, 2024.

[30] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586–595, 2018.

[31] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang, Cheng Peng, Rama Chellappa, and Deliang Fan. LP-3DGS: Learning to prune 3D gaussian splatting. In NeurIPS, pages 122434–122457, 2024.

# Supplementary Material LinearMask-GS: Stable-Mask Importance Pruning for Compact 3D Gaussian Splatting

<table><tr><td>Donghun Ryu1</td><td>1 Department of Intelligent</td></tr><tr><td>donghun0621@cau.ac.kr</td><td>Semiconductor Engineering</td></tr><tr><td>Minhyeok Lee1,2†</td><td>Chung-Ang University</td></tr><tr><td rowspan="3">mlee@cau.ac.kr</td><td>Seoul, Republic of Korea</td></tr><tr><td>²School of Electrical and Electronics Engineering</td></tr><tr><td>Chung-Ang University</td></tr><tr><td></td><td>Seoul, Republic of Korea</td></tr></table>

This document provides supplementary material referenced in the main paper. It includes additional analyses of training convergence, hyperparameter sensitivity, and mask-entropy evolution; a quantitative analysis of the reliability of the learned mask ranking and of the effect of the mask-window length and placement; per-scene quantitative results on Mip-NeRF 360; the full Tanks & Temples and Deep Blending comparison; per-scene qualitative comparisons and minimum-survival-ratio ρ sweeps for both the linear-increment activation and the matched Gumbel-Sigmoid baseline; a mask-free pruning baseline, a budget-matched comparison with Speedy-Splat, and an extended cross-backbone evaluation; a decomposition of the mask-learning speed gap on the FastGS backbone; and a description of the accompanying video.

## 1 Additional Experimental Results

In this section, we present additional analyses supporting LinearMask-GS, including training convergence profiles and sensitivity analyses for key hyperparameters.

## 1.1 Training convergence

Figure 1 plots PSNR and SSIM from iteration 19,500 onward. LinearMask-GS reaches higher PSNR and SSIM than LP-3DGS within the 500-iteration mask-training phase (iter. 19,500–20,000) and maintains this advantage through fine-tuning. The faster convergence reflects the gentle slope of the linear-increment activation, which keeps mask values away from the saturation boundaries that Gumbel-Sigmoid reaches early in the mask-training window. As discussed in the main paper, this consistent gradient flow translates to substantially better efficiency scores across all metric dimensions.

![](images/24dfd200aab68a5a517907fc5304874f6011f4ecad5b9cf3e77cab3f95ebcd2b.jpg)

![](images/af018e950a457f676170def1c3677006d1c28232a4bbf8d65d5c1c711db9bd80.jpg)  
Figure 1: Training convergence. PSNR (dB) and SSIM after iteration 19,500 on Mip-NeRF 360. LinearMask-GS converges faster and to higher values than LP-3DGS.

## 1.2 Sensitivity to sparsity weight $\lambda _ { m }$

Figure 2 evaluates $\lambda _ { m } \in \{ 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } \}$ , using the same per-Gaussian efficiency scores $( 1 0 ^ { 9 } / ( \mathrm { L P I P S } \times \# \mathrm { G } )$ , PSNR/#G, SSIM/#G) as Figure 3 of the main paper. The default $\lambda _ { m } = 5 \times 1 0 ^ { - 4 }$ achieves the best LPIPS-based efficiency score. Larger values increase sparsity at the cost of perceptual quality, while smaller values provide insufficient regularization; all variants with $\bar { \lambda } _ { m } \in \mathrm { \overline { { [ 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } ] } } }$ robustly outperform LP-3DGS.

![](images/814e2110e0689f57d2cd748a6826707725eb5ac4ff5eac1f90e3c8b14a7bbb8e.jpg)

![](images/6c9dc425e7c66c279e864ad18bf12b256cf40dffd593ba3c33322f7e3d8afb46.jpg)

![](images/f0e88cfc202595a07f7392fc6a1b4de8967961c06f8f4d09e1ee874f8fb2c499.jpg)  
Figure 2: Sensitivity to $\lambda _ { m }$ . Efficiency score under varying $\lambda _ { m }$ on Mip-NeRF 360. The default $\lambda _ { m } = 5 \times 1 0 ^ { - 4 }$ (highlighted) yields the best LPIPS-based efficiency score.

## 1.3 Sensitivity to slope τ

The main paper reports the primary grid search over $\tau \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 \}$ . Figure 3 evaluates an extended range $\tau \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 2 5 , 0 . 5 , 1 . 0 \}$ . The linear-increment activation is robust to the choice of slope: efficiency scores remain high across $0 . 0 1 \leq \tau \leq 0 . 5$ , and in PSNR terms every $\tau \in [ 0 . 0 5 , 0 . 5 ]$ stays within 0.1 dB of the default. The default $\tau = 0 . 1$ gives the best balance, while only extreme values $( \mathrm { e . g . , } \tau = 1 . 0 )$ show a noticeable reduction in efficiency.

![](images/53beea9b2be9898be40c94fee552856e61e1f15ed62e83963131b066847c1799.jpg)  
Figure 3: Sensitivity to τ. Efficiency score under varying slope τ on Mip-NeRF 360. The default $\tau = 0 . 1$ (highlighted) achieves the best LPIPS-based efficiency score.

## 2 Prevention of Premature Binarization

A key advantage of LinearMask-GS is the prevention of premature binarization, where the Gumbel-Sigmoid activation forcefully pushes mask values toward 0 or 1 before the model has adequately evaluated their contribution to the scene. To empirically validate this, we analyze the evolution of mask entropy during the mask-training phase. The binary entropy of a per-Gaussian mask value $\psi _ { i }$ is

$$
H ( \psi _ { i } ) = - \psi _ { i } \log \psi _ { i } - ( 1 - \psi _ { i } ) \log ( 1 - \psi _ { i } ) .\tag{1}
$$

A higher entropy indicates that mask values remain concentrated around 0.5, so the corresponding Gaussians are still under active evaluation; a near-zero entropy implies the values have collapsed to the extremes (0 or 1), committing to a prune/keep decision before that evaluation has stabilized.

Mask Entropy Evolution (Averaged across 9 Scenes)  
![](images/5d08ec0567faa734415dda3096796277297423ac8d50d3d610049aa52f01b70d.jpg)  
Figure 4: Mask entropy evolution. Mean binary entropy of mask values across all nine Mip-NeRF 360 scenes during the measured pruning phase (iterations 19,600–20,000). Shaded regions show standard deviation.

As shown in Figure 4, LP-3DGS exhibits a low entropy ( 0.33) throughout this phase, confirming that the Gumbel-Sigmoid activation drives an early hard decision. In contrast, LinearMask-GS maintains a substantially higher and stable entropy $( \approx 0 . 6 9 )$ , and the narrow standard-deviation band across the nine scenes indicates the robustness of the linearincrement activation. This sustained intermediate regime helps preserve high-frequency details, such as bicycle spokes and foliage, rather than pruning them prematurely.

## 3 Ranking reliability of the learned mask

This section details the protocol behind the ranking-reliability statistics reported in Section 4.4 of the main paper.

Protocol. For each of the nine Mip-NeRF 360 scenes we train two runs that are identical except for the mask activation (Gumbel-Sigmoid vs. linear increment), with hard pruning disabled so that the mask evolution can be observed without feedback from primitive removal. We snapshot the mask ψ every 100 iterations over the measured part of the mask window (iterations 19,600–20,000) and, after training continues to 30,000 iterations, compute a held-out contribution proxy for every Gaussian: its max-pooled rendering contribution (Eq. 4 of the main paper) at iteration 30k.

Metrics. (i) Decision stability: at the pruning decision point (20k), we re-sample the Gumbel noise and re-apply the top-ρN rule; the flip fraction is the share of Gaussians whose keep/prune decision changes between two independent noise draws. (ii) Saturation: the fraction of ψ values outside [0.1, 0.9]. (iii) Ranking fidelity: the Spearman correlation between the ranking induced by ψ and the proxy ranking.

Results. Re-sampling the Gumbel noise alone flips 47.3% of keep/prune decisions on average (46–48% on every scene), and the rank correlation between two independent noise draws is below 0.1: the sampled decision is governed by activation noise rather than by a stable learned ranking. The linear mask is deterministic (0% flips). Saturation reaches 52.2% for Gumbel-Sigmoid under this no-pruning protocol vs. 0.00% for linear; the corresponding figure under the standard schedule of the main paper is 51.19% (Section 3.3), i.e., disabling pruning does not alter the picture. The Spearman correlation against the 30k proxy is 0.29 for linear vs. 0.16 for Gumbel-Sigmoid (1.8 higher overall and 2.8 on outdoor scenes). Figure 5 plots both statistics across the mask window; the narrow per-scene spread shows the statistics are scene-invariant rather than driven by a few scenes.

## 4 Effect of mask-window length and placement

A natural question is whether a longer mask-training window would let the Gumbel-Sigmoid mask escape saturation. It does not; it makes saturation worse. We sweep the placement of the mask window on three scenes (Bicycle, Flowers, Counter) by moving its start to 15k, 17.5k, or 19.5k with the end held at the 20k pruning point, so that the sweep simultaneously varies the window length (5,000, 2,500, and 500 iterations). Under Gumbel-Sigmoid, the saturated fraction at the decision point rises monotonically as the window lengthens, from 51.1% to 54.8% to 56.2%, while PSNR plateaus (the two longest windows differ by

![](images/2b24a057a65883bde3e05a7d818c2198e49928e23a51def20e26e440e1b93daf.jpg)  
(a) Spearman corr. vs. 30k proxy

![](images/b8a5aeb5131bf95b8a32a5e85b691d0fab57808f49fbcbac57a5abe1dd3f5d50.jpg)  
(b) Keep/prune flip fraction  
Figure 5: Ranking reliability on Mip-NeRF 360 across the mask-training window. (a) Spearman correlation between the mask ranking and the held-out max-pooled contribution proxy at 30k. (b) Fraction of keep/prune decisions flipped by re-sampling the Gumbel activation noise; the linear-increment mask is deterministic. Shaded regions show the min–max range over the nine scenes.

0.05 dB). The linear-increment activation is not fully exempt from this trend: its saturated fraction also grows with the window, from 0.0012% to 0.0606% to 0.0782%. It does so, however, at a level at least 700 below Gumbel-Sigmoid, and in all nine scene–window cells (3 scenes 3 window settings) the Gumbel-Sigmoid fraction exceeds the linear one. Its final PSNR varies by at most 0.14 dB across the three placements. Table 1 summarises the sweep.

Table 1: Effect of the mask-window length on saturation, with the pruning point held fixed at 20k. Moving the window start keeps the end at the pruning point, so length and placement vary along a single axis. The saturated fraction is the share of $\psi _ { i }$ outside [0.1, 0.9] at the decision point, averaged over the three scenes (Bicycle, Flowers, Counter). Lengthening the window raises saturation for both activations, but the linear increment remains at least 700 below Gumbel-Sigmoid at every setting. Quality does not compensate: Gumbel-Sigmoid PSNR differs by 0.05 dB between the two longest windows, and the linear increment by at most 0.14 dB across all three placements, so window duration is not the binding factor.
<table><tr><td colspan="2">Mask window</td><td colspan="2">Saturated fraction at 20k (↓)</td></tr><tr><td>Start</td><td>Length</td><td>Gumbel-Sigmoid</td><td>Linear increment</td></tr><tr><td>19.5k</td><td>500</td><td>51.1%</td><td>0.00%</td></tr><tr><td>17.5k</td><td>2,500</td><td>54.8%</td><td>0.06%</td></tr><tr><td>15.0k</td><td>5,000</td><td>56.2%</td><td>0.08%</td></tr></table>

Longer exposure to the steep activation simply gives mask values more time to reach the saturation boundaries; under the fixed-pruning-point protocol, the binding factor is activation steepness, not window duration. The asymmetry mirrors the ranking-reliability analysis of Section 3: a ranking that is stable in time and deterministic in noise is insensitive to when it is read out, whereas a saturating one degrades the longer it saturates.

## 5 Per-Scene Quantitative Metrics on Mip-NeRF 360

Tables 2–6 detail the per-scene performance on the nine Mip-NeRF 360 scenes. Our method consistently matches or outperforms the LP-3DGS pruning baseline in LPIPS, PSNR, and SSIM, while using a drastically smaller number of Gaussians (Table 5) and storage footprint (Table 6). Figure 6 visualizes the same per-scene trade-off: each marker is one scene, and lower-left is better.

![](images/c98fa6f8afc9f3937697ae11925576c065391fa4b4dc1a2488d131053481048a.jpg)  
(a) Outdoor scenes

![](images/9e971ae3d4079c6c0baedf689bc4c554d751012e83f9019479c5c37532a0edeb.jpg)  
(b) Indoor scenes  
Figure 6: Per-scene LPIPS vs. number of Gaussians on Mip-NeRF 360. Each marker is one scene; lower-left is better. LinearMask-GS (⋆) reaches comparable or lower LPIPS with substantially fewer Gaussians than LP-3DGS ( ).

Table 2: LPIPS scores ( ) for Mip-NeRF 360 scenes. <sup>†</sup>Literature rows (Plenoxels, INGPbase, Mip-NeRF 360, 3DGS) report the values given in the original papers; the per-scene entries in those rows are approximate reproductions and do not always average to them. The LP-3DGS and Ours rows are means of our per-scene measurements.
<table><tr><td>Method</td><td>Bicycle</td><td>Bonsai</td><td>Counter</td><td>Kitchen</td><td>Room</td><td>Stump</td><td>Garden</td><td>Flowers</td><td>Treehill</td><td>Avg.</td></tr><tr><td>Plenoxels</td><td>0.534</td><td>0.421</td><td>0.451</td><td>0.354</td><td>0.472</td><td>0.485</td><td>0.351</td><td>0.521</td><td>0.578</td><td>0.463†</td></tr><tr><td>INGP-base</td><td>0.495</td><td>0.245</td><td>0.364</td><td>0.285</td><td>0.185</td><td>0.421</td><td>0.325</td><td>0.485</td><td>0.534</td><td>0.371†</td></tr><tr><td>Mip-NeRF 360</td><td>0.354</td><td>0.198</td><td>0.211</td><td>0.154</td><td>0.225</td><td>0.285</td><td>0.175</td><td>0.395</td><td>0.364</td><td>0.237†</td></tr><tr><td>3DGS</td><td>0.205</td><td>0.205</td><td>0.204</td><td>0.129</td><td>0.220</td><td>0.215</td><td>0.103</td><td>0.336</td><td>0.317</td><td>0.214†</td></tr><tr><td>LP-3DGS</td><td>0.253</td><td>0.183</td><td>0.187</td><td>0.120</td><td>0.201</td><td>0.242</td><td>0.128</td><td>0.367</td><td>0.351</td><td>0.226</td></tr><tr><td>Ours</td><td>0.211</td><td>0.182</td><td>0.187</td><td>0.120</td><td>0.200</td><td>0.210</td><td>0.109</td><td>0.336</td><td>0.331</td><td>0.210</td></tr></table>

## 6 Full quantitative comparison on Tanks & Temples and Deep Blending

Table 7 provides the complete scene-wise comparison on Tanks & Temples and Deep Blending that was summarized in Section 4.2 of the main paper. We extend the comparison with recent fast-training and compact 3DGS baselines, using their scene-wise paper results when available. The final two rows report the main LinearMask-GS result and the FastGSbackbone variant LinearMask-GS<sup>⋆</sup>.

Table 3: PSNR scores ( ) for Mip-NeRF 360 scenes. <sup>†</sup>Literature rows (Plenoxels, INGPbase, Mip-NeRF 360, 3DGS) report the values given in the original papers; the per-scene entries in those rows are approximate reproductions and do not always average to them. The LP-3DGS and Ours rows are means of our per-scene measurements.
<table><tr><td>Method</td><td>Bicycle</td><td>Bonsai</td><td>Counter</td><td>Kitchen</td><td>Room</td><td>Stump</td><td>Garden</td><td>Flowers</td><td>Treehill</td><td>Avg.</td></tr><tr><td>Plenoxels</td><td>21.73</td><td>25.54</td><td>24.85</td><td>24.71</td><td>22.51</td><td>23.23</td><td>23.51</td><td>20.07</td><td>21.54</td><td>23.08†</td></tr><tr><td>INGP-base</td><td>22.17</td><td>28.31</td><td>26.85</td><td>27.54</td><td>30.54</td><td>24.12</td><td>24.96</td><td>21.14</td><td>22.05</td><td>25.30†</td></tr><tr><td>Mip-NeRF 360</td><td>24.37</td><td>33.46</td><td>29.55</td><td>32.23</td><td>31.63</td><td>26.40</td><td>26.98</td><td>21.52</td><td>22.49</td><td>27.69†</td></tr><tr><td>3DGS</td><td>25.25</td><td>31.13</td><td>28.70</td><td>30.32</td><td>30.63</td><td>26.55</td><td>27.41</td><td>21.52</td><td>22.49</td><td>27.21†</td></tr><tr><td>LP-3DGS</td><td>25.10</td><td>32.09</td><td>28.94</td><td>31.52</td><td>31.49</td><td>26.69</td><td>27.29</td><td>21.38</td><td>22.71</td><td>27.47</td></tr><tr><td>Ours</td><td>25.67</td><td>32.02</td><td>29.05</td><td>31.50</td><td>31.70</td><td>26.94</td><td>27.75</td><td>21.86</td><td>22.85</td><td>27.70</td></tr></table>

Table 4: SSIM scores ( ) for Mip-NeRF 360 scenes. <sup>†</sup>Literature rows (Plenoxels, INGPbase, Mip-NeRF 360, 3DGS) report the values given in the original papers; the per-scene entries in those rows are approximate reproductions and do not always average to them. The LP-3DGS and Ours rows are means of our per-scene measurements.
<table><tr><td>Method</td><td>Bicycle</td><td>Bonsai</td><td>Counter</td><td>Kitchen</td><td>Room</td><td>Stump</td><td>Garden</td><td>Flowers</td><td>Treehill</td><td>Avg.</td></tr><tr><td>Plenoxels</td><td>0.447</td><td>0.864</td><td>0.795</td><td>0.782</td><td>0.878</td><td>0.498</td><td>0.701</td><td>0.351</td><td>0.315</td><td>0.626†</td></tr><tr><td>INGP-base</td><td>0.495</td><td>0.912</td><td>0.854</td><td>0.831</td><td>0.899</td><td>0.542</td><td>0.723</td><td>0.431</td><td>0.378</td><td>0.671†</td></tr><tr><td>Mip-NeRF 360</td><td>0.685</td><td>0.941</td><td>0.894</td><td>0.920</td><td>0.913</td><td>0.744</td><td>0.813</td><td>0.583</td><td>0.632</td><td>0.792†</td></tr><tr><td>3DGS</td><td>0.771</td><td>0.938</td><td>0.905</td><td>0.922</td><td>0.913</td><td>0.775</td><td>0.868</td><td>0.605</td><td>0.638</td><td>0.815†</td></tr><tr><td>LP-3DGS</td><td>0.744</td><td>0.946</td><td>0.914</td><td>0.931</td><td>0.926</td><td>0.771</td><td>0.855</td><td>0.584</td><td>0.636</td><td>0.812</td></tr><tr><td>Ours</td><td>0.779</td><td>0.944</td><td>0.913</td><td>0.931</td><td>0.925</td><td>0.787</td><td>0.872</td><td>0.623</td><td>0.655</td><td>0.825</td></tr></table>

Table 5: Number of final Gaussians (millions, ) for Mip-NeRF 360 scenes. <sup>†</sup>3DGS values are those reported in the original paper; the per-scene entries in this row are approximate reproductions and do not average to it. The LP-3DGS and Ours rows are means of our perscene measurements.
<table><tr><td>Method</td><td>Bicycle</td><td>Bonsai</td><td>Counter</td><td>Kitchen</td><td>Room</td><td>Stump</td><td>Garden</td><td>Flowers</td><td>Treehill</td><td>Avg.</td></tr><tr><td>3DGS</td><td>5.7M</td><td>1.3M</td><td>1.2M</td><td>1.8M</td><td>1.5M</td><td>3.3M</td><td>5.8M</td><td>3.6M</td><td>3.8M</td><td>3.36M†</td></tr><tr><td>LP-3DGS</td><td>2.51M</td><td>0.54M</td><td>0.51M</td><td>0.88M</td><td>0.48M</td><td>2.01M</td><td>2.83M</td><td>1.75M</td><td>1.80M</td><td>1.48M</td></tr><tr><td>Ours</td><td>1.75M</td><td>0.37M</td><td>0.35M</td><td>0.53M</td><td>0.45M</td><td>1.41M</td><td>1.52M</td><td>1.02M</td><td>1.10M</td><td>0.94M</td></tr></table>

Table 6: Storage size (MB, ) for Mip-NeRF 360 scenes. <sup>†</sup>3DGS values are those reported in the original paper; the per-scene entries in this row are approximate reproductions and do not average to it. The LP-3DGS and Ours rows are means of our per-scene measurements.
<table><tr><td>Method</td><td>Bicycle</td><td>Bonsai</td><td>Counter</td><td>Kitchen</td><td>Room</td><td>Stump</td><td>Garden</td><td>Flowers</td><td>Treehill</td><td>Avg.</td></tr><tr><td>3DGS</td><td>1450.2</td><td>340.8</td><td>320.5</td><td>480.2</td><td>390.4</td><td>1050.2</td><td>1280.5</td><td>750.4</td><td>850.1</td><td>734.0†</td></tr><tr><td>LP-3DGS</td><td>557.9</td><td>101.6</td><td>95.2</td><td>175.0</td><td>90.4</td><td>471.7</td><td>568.6</td><td>369.0</td><td>397.7</td><td>314.1</td></tr><tr><td>Ours</td><td>432.1</td><td>88.4</td><td>83.5</td><td>125.1</td><td>86.4</td><td>333.5</td><td>395.4</td><td>240.4</td><td>256.0</td><td>226.8</td></tr></table>

## 7 Per-scene qualitative comparison (remaining six scenes)

Figure 7 extends Figure 2 of the main paper with per-pixel absolute error maps for the six Mip-NeRF 360 scenes that were not shown there: Garden, Stump, Treehill (outdoor), and Bonsai, Kitchen, Room (indoor). The visualization conventions are identical to the main paper’s Figure 2: per-image PSNR is reported below each rendered image and per-image MAE below each error map, and each scene uses its own blue-to-red colormap from low to high absolute error. Across all six scenes, LinearMask-GS produces lower MAE and fewer saturated high-error (red) regions than both LP-3DGS and 3DGS.

![](images/77f691d052fe8f18d77e23216b9ecd3c288d3d7a052dcdffc67184f6a1a8f709.jpg)  
Figure 7: Per-pixel qualitative comparison on the six Mip-NeRF 360 scenes not shown in the main paper’s Figure 2. Each row shows one scene; columns from left to right show the ground truth, the rendered images from LinearMask-GS (ours), LP-3DGS, and 3DGS, followed by per-pixel absolute error maps for the same three methods. Per-image PSNR (dB) is reported below each render and per-image MAE below each error map; error maps use a scene-specific blue-to-red colormap (low to high absolute error).

## 8 Minimum-survival ratio ρ sweep for the linear-increment activation

Figure 8 shows the per-scene LPIPS curves against the number of Gaussians for $\rho \in$ $\{ 0 . 2 0 , 0 . 3 0 , 0 . 4 0 \}$ , and Table 8 extends the sweep to $\rho \in \{ 0 . 1 0 , 0 . 2 0 , 0 . 3 0 , 0 . 4 0 , 0 . 5 0 \}$ with overall, outdoor, and indoor averages, mirroring the format of Table 9. Under the linearincrement activation, $\psi _ { i }$ is unimodal and its relative order is preserved rather than collapsed onto the saturation points (Section 3.3 of the main paper), so the top-ρN rule selects a consistent, importance-ordered subset whose size is set by $\rho$

Across every scene, LPIPS decreases monotonically as $\rho$ increases, at the expected cost of more retained Gaussians. The returns are asymmetric about a knee at $\rho { = } 0 . 3 0 \colon \rho { = } 0 . 2 0$ produces a visible LPIPS increase on most scenes ( 0.28 dB PSNR on aggregate), indicating that this regime removes too many geometrically meaningful primitives, whereas $\rho { = } 0 . 4 0$ retains 34% more Gaussians for only +0.03 dB and a 0.006 LPIPS gain. We therefore use $\rho { = } 0 . 3 0$ as the default operating point. We emphasize that the linear-increment activation does not exempt the trade-off from saturation: beyond $\rho { = } 0 . 3 0$ its aggregate PSNR gains total only $+ 0 . 0 4$ dB up to $\rho { = } 0 . 5 0 $ , just as the Gumbel-Sigmoid baseline saturates over the same range. The benefit of the stable ranking lies not in the shape of this curve but in its position, which Section 9 quantifies.

![](images/d7c1c3f62c8d5ea83a550820f811d6ca56e863e6f1afbe9b926895fdd193fb36.jpg)  
Figure 8: Effect of the minimum-survival ratio $\rho$ on Mip-NeRF 360: per-scene LPIPS vs. number of Gaussians for $\rho \in \{ 0 . 2 0 , 0 . 3 0 , 0 . 4 0 \}$ . Lower LPIPS with fewer Gaussians indicates a better quality–compression trade-off. Increasing $\rho$ monotonically reduces LPIPS at the cost of more retained Gaussians; $\rho { = } 0 . 2 0$ already produces a visible LPIPS increase on most scenes, while $\rho { = } 0 . 4 0$ retains significantly more primitives without proportional perceptual gains. We use $\rho { = } 0 . 3 0$ as the default operating point.

## 9 $\rho$ sweep: linear increment vs. the matched Gumbel-Sigmoid baseline

Table 9 provides the analogous $\rho$ sweep for the matched Gumbel-Sigmoid baseline $( \mathrm { i . e . }$ LP-3DGS with max-pooling aggregation, the $\mathrm { \ddot { \ s i g m a x } + G u m b e l { - S i g m o i d } ^ { \prime } }$ row of Table 4 in the main paper), under the identical top-ρN rule and schedule as Table 8. Because top- $- \rho N$ retains the same fraction of a near-identical primitive pool, equal $\rho$ implies matched budgets: #G differs by at most 0.01M at every $\rho _ { \mathrm { - } }$ , so each row of the two tables is a same-budget comparison.

Both curves saturate beyond $\rho { = } 0 . 3 0$ . For Gumbel-Sigmoid, PSNR rises by $+ 0 . 4 9$ dB from $\rho { = } 0 . 1 0$ to 0.20 and +0.24 dB from 0.20 to 0.30, then by only +0.11 and +0.00 dB; for the linear increment, the corresponding steps are $+ 0 . 4 8 , + 0 . 2 8 , + 0 . 0 3$ , and +0.01 dB. The shape of the quality–compression trade-off is therefore not what separates the two activations: both follow the diminishing returns of the scene’s importance spectrum, with retained Gaussians scaling nearly linearly in $\rho$ while quality gains collapse past the knee.

What separates them is position. At every $\rho$ in the sweep, the linear-increment activation is +0.40 to +0.48 dB PSNR above the Gumbel-Sigmoid baseline at the same budget, and the baseline cannot match the linear curve anywhere: its best point (27.33 dB / 0.2321 LPIPS at $\rho { = } 0 . 5 0$ with 1.57M Gaussians) falls below even the linear $\rho { = } 0 . 2 0$ setting (27.42 dB / 0.2194 LPIPS at 0.63M, i.e., 2.5 fewer primitives), and it does not reach the linear default (27.70 dB / 0.2095 at $\rho { = } 0 . 3 0$ with 0.94M; Table 1 of the main paper) despite using $67 \%$ more primitives. The outdoor-vs.-indoor split shows the offset is broad rather than scene-specific: at every $\rho$ in this sweep, the Gumbel-Sigmoid baseline lags the linear increment in both subsets on all three metrics at every $\rho$ (Table 8): the outdoor PSNR margin ranges from +0.48 to +0.55 dB and the indoor margin from +0.29 to +0.44 dB, with SSIM and LPIPS showing the same sign throughout.

These same-budget deficits are the downstream signature of the ranking analysis in Section 3: a saturated, noise-driven mask still produces a ρ-curve of the ordinary shape, but the subset it selects at each budget is less aligned with importance, and this appears as a near-uniform vertical offset of the entire curve rather than as a change in its slope.

## 10 Direct pruning by the raw max-pooled score

To isolate the contribution of the learned mask beyond its ingredients, we prune directly by the raw max-pooled importance score: the schedule is identical to LinearMask-GS, but at iteration 20k the top-30% of Gaussians by $S _ { i }$ (Eq. 4 of the main paper) are retained with no learned mask, followed by the same fine-tuning. On the nine Mip-NeRF 360 scenes this mask-free baseline reaches 27.5725 dB / 0.8231 / 0.2143 at 0.9612M, while the learned mask reaches higher PSNR on eight of the nine scenes at a smaller budget (27.7044 dB / 0.8254 / 0.2095 at 0.9444M), a +0.1319 dB gain. The sole exception is Treehill ( 0.17 dB), the scene with the largest sky and distant-vegetation area, where the raw score already ranks primitives well. Table 10 reports the per-scene PSNR.

## 11 Budget-matched comparison with Speedy-Splat

Table 1 of the main paper compares methods at their native operating points, where Speedy-Splat uses 0.30M Gaussians. For a comparison at a matched primitive budget, we retrain LinearMask-GS<sup>⋆</sup> (FastGS backbone) under primitive budgets of 0.25M, 0.30M, and 0.35M. Averaged over the nine Mip-NeRF 360 scenes and three independent seeds at 0.30M, LinearMask-GS<sup>⋆</sup> reaches 27.40 dB / 0.798 / 0.230, vs. 26.91 / 0.781 / 0.295 for Speedy-Splat, i.e., better on all three metrics at the identical budget. It remains ahead even at the smaller 0.25M budget, and improves further at 0.35M (Table 11).

## 12 Extended cross-backbone evaluation

Table 3 of the main paper evaluates DropGaussian on a single LLFF scene (fern) and Octree-GS on a single Bungee-NeRF scene (amsterdam). Here we extend both.

DropGaussian: five scenes, three runs. We evaluate five LLFF scenes (9-view; fern, flower, fortress, horns, and room) with the masked model repeated over three independent training runs (the public DropGaussian code does not expose an explicit seed flag, so repetitions correspond to independent launches). Averaged over the three runs, LinearMask-GS improves PSNR on every scene, by +0.31 to +0.52 dB (mean +0.42 dB) over the unpruned backbone, while retaining 41% fewer primitives than the Gumbel variant (Table 12). The single fern run reported in Table 3 of the main paper gives a smaller gain (+0.03 dB), indicating that single runs of this sparse-view backbone are noisy and motivating the repeated protocol used here. Table 12 reports the per-scene, per-run results.

Octree-GS: two additional scenes. Beyond amsterdam (a 2.9 anchor reduction at +0.12 dB; Table 3 of the main paper), we evaluate the bicycle and garden scenes of Mip-NeRF 360: anchors drop by 40.2% relative to the Gumbel variant at a mean PSNR change of 0.11 dB. We report the two scene sets separately rather than averaging them, since the sign of the (small) PSNR change differs; across all three scenes, quality moves by at most 0.1 dB in either direction while the anchor count is reduced by 40% or more.

## 13 Why mask learning is so fast on the FastGS backbone

As reported in Table 2 of the main paper, applying our mask on the FastGS backbone (LinearMask-GS<sup>⋆</sup>) does not increase, and in fact reduces, the total training time, whereas the 3DGS-backbone variant adds a mask-training stage on top of the base pipeline. This section explains why. We first decompose the mask-learning speed gap (Table 14), then account for the net change in total training time over the 20k–30k iteration range (Table 15).

Mask-learning speed gap. The mask-learning procedure that takes 110 s on the 3DGS backbone completes in only 4.6 s on the FastGS backbone; both are measured on the same RTX 4090, so the gap reflects the backbone and rasterizer rather than any hardware difference. Table 14 attributes this 24 gap to two directly measured factors. The dominant factor is the number of Gaussians present when the mask window begins (Factor 1, 6.36 ): averaged over the nine Mip-NeRF 360 scenes, the 3DGS backbone enters the mask window with 3.14 M Gaussians, whereas FastGS has already reduced this to 0.493 M through its cumulative pruning events, so each mask-training iteration rasterizes about 6.36 fewer primitives. The second factor is the FastGS rasterizer itself (Factor 2, 3.7 ), which processes far fewer tiles per splat. The two factors combine to $6 . 3 6 \times 3 . 7 \approx 2 4 \times$ , consistent with the directly measured times since $1 1 0 \mathrm { s } / 2 4 \approx 4 . 6 \mathrm { s }$

Net effect on total training time. Although LinearMask- $\mathrm { G S ^ { \star } }$ adds a 4.6 s mask-learning stage that FastGS does not have, it more than recovers this cost during the rest of training. Table 15 reports the wall-clock time over the 20k–30k iteration range, averaged across the nine Mip-NeRF 360 scenes. Because LinearMask- $\mathbf { \sigma } . \mathbf { G S ^ { \star } }$ prunes additional primitives, it renders fewer Gaussians per iteration and completes this range in 28.3 s versus 39.2 s for FastGS native, a 10.9 s saving. Subtracting the 4.6 s mask-learning cost still leaves a net saving of 6.3 s per scene, which is why the total training time in Table 2 of the main paper decreases from 1.93 to 1.81 minutes rather than increasing.

In short, the speed advantage of LinearMask-GS<sup>⋆</sup> comes entirely from operating on a backbone that has already pruned most primitives before mask learning begins, combined with a faster rasterizer, and not from any change to the masking formulation itself, which is identical across backbones.

## 14 Supplementary Video

Alongside this document, we provide a supplementary video archive containing splitscreen comparison renderings for all 13 evaluated scenes (9 from Mip-NeRF 360, 2 from Deep Blending, and 2 from Tanks & Temples). Each video compares rendered test views in a $2 \times 2$ grid: ground truth (top-left), 3DGS (top-right), LP-3DGS (bottom-left), and LinearMask-GS (bottom-right). Each panel displays the total number of Gaussians used. Despite using substantially fewer Gaussians, our method preserves perceptual quality comparable to the original 3DGS and superior to LP-3DGS, retaining high-frequency details.

Table 7: Quantitative comparison on Tanks & Temples and Deep Blending. Best values are in bold. Fast-training baseline rows use the scene-wise values reported by FastGS [9]; ⋆ marks our FastGS-backbone run that substitutes the VCP module with LinearMask-GS.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric PSNR↑</td><td colspan="3">Tanks &amp; Temples</td><td colspan="3">Deep Blending</td></tr><tr><td>Truck 23.221</td><td>Train 18.927</td><td>Avg 21.074 0.718</td><td>Dr Johnson 23.142 0.787</td><td>Playroom 22.980 0.802</td><td>Avg 23.061 0.794</td></tr><tr><td>Plenoxels [4]</td><td>SSIM↑ LPIPS↓ PSNR↑</td><td>0.774 0.335 23.260</td><td>0.663 0.422 20.170</td><td>0.378 21.715</td><td>0.521 27.750</td><td>0.499 19.483 0.754</td><td>0.510 23.616 0.796</td></tr><tr><td>INGP-Base [8]</td><td>SSIM↑ LPIPS↓ PSNR↑ SSIM↑</td><td>0.779 0.274 23.383 0.800</td><td>0.666 0.386 20.456 0.689</td><td>0.722 0.330 21.919 0.744</td><td>0.839 0.381 28.257 0.854</td><td>0.465 21.665 0.779</td><td>0.423 24.961 0.816</td></tr><tr><td>INGP-Big [8] Mip-NeRF 360 [1]</td><td>LPIPS↓ PSNR↑ SSIM↑</td><td>0.249 24.912 0.857</td><td>0.360 19.523 0.660</td><td>0.304 22.217 0.758</td><td>0.352 29.140 0.901</td><td>0.428 29.657 0.900</td><td>0.390 29.398 0.900</td></tr><tr><td>3DGS [6]</td><td>LPIPS↓ PSNR↑ SSIM↑ LPIPS↓</td><td>0.159 25.187 0.879 0.148</td><td>0.354 21.097 0.802 0.218</td><td>0.256 23.142 0.840 0.183</td><td>0.237 28.766 0.899 0.244</td><td>0.252 30.044 0.906 0.241</td><td>0.244 29.405 0.902 0.242</td></tr><tr><td>LP-3DGS [10]</td><td>#G (M)↓ PSNR↑ SSIM↑ LPIPS↓</td><td>1.09 25.376 0.877 0.154</td><td>2.59 21.822 0.807 0.222</td><td>1.84 23.599 0.842 0.188</td><td>3.31 29.223 0.901 0.248</td><td>2.33 30.154 0.908 0.248</td><td>2.82 29.690 0.905 0.248</td></tr><tr><td>Mini-Splatting [3]</td><td>#G (M)↓ PSNR↑ SSIM↑ LPIPS↓</td><td>1.33 25.320 0.879 0.139</td><td>0.58 21.600 0.809 0.223</td><td>0.96 23.460 0.844 0.181</td><td>1.65 29.510 0.905 0.246</td><td>1.15 30.470 0.908 0.241</td><td>1.40 29.990 0.907 0.244</td></tr><tr><td>Speedy-Splat [5]</td><td>#G (M)↓ PSNR↑ SSIM↑ LPIPS↓</td><td>0.32 25.180 0.863 0.192</td><td>0.28 21.590 0.768 0.292</td><td>0.30 23.380 0.816 0.242</td><td>0.60 29.070 0.898 0.269</td><td>0.51 29.770 0.898 0.274</td><td>0.56 29.420 0.898 0.272</td></tr><tr><td>Taming-3DGS [7]</td><td>#G (M)↓ PSNR↑ SSIM↑ LPIPS↓</td><td>0.26 25.270 0.865 0.187</td><td>0.11 22.500 0.802 0.240</td><td>0.18 23.890 0.833 0.214</td><td>0.31 29.040 0.888 0.292</td><td>0.18 29.960 0.901 0.264</td><td>0.25 29.500 0.894 0.278</td></tr><tr><td>DashGaussian [2]</td><td>#G (M)↓ PSNR↑ SSIM↑ LPIPS↓</td><td>0.27 25.800 0.886 0.150</td><td>0.37 22.190 0.819 0.206</td><td>0.32 24.000 0.853 0.178</td><td>0.19 29.130 0.903 0.250</td><td>0.40 30.170 0.909 0.243</td><td>0.29 29.650 0.906 0.246</td></tr><tr><td>FastGS [9]</td><td>#G (M)↓ PSNR↑ SSIM↑ LPIPS↓</td><td>1.43 25.730 0.872 0.178</td><td>1.00 22.570 0.805 0.242</td><td>1.21 24.150 0.839 0.210</td><td>1.51 29.500 0.898 0.275</td><td>2.38 30.570 0.905 0.266</td><td>1.94 30.030 0.901 0.270</td></tr><tr><td>LinearMask-GS</td><td>#G (M)↓ PSNR↑ SSIM↑ LPIPS↓ #G (M)↓</td><td>0.25 25.793 0.882 0.151 0.78</td><td>0.23 22.546 0.811 0.223 0.33</td><td>0.24 24.170 0.846 0.187 0.55</td><td>0.25 29.602 0.903 0.245 0.99</td><td>0.19 30.583 0.909 0.245 0.70</td><td>0.22 30.093 0.906 0.245 0.85</td></tr><tr><td>LinearMask-GS*</td><td>PSNR↑ SSIM↑ LPIPS↓ #G (M)↓</td><td>25.787 0.861 0.172 0.31</td><td>22.565 0.790 0.236 0.26</td><td>24.176 0.825 0.204 0.29</td><td>29.672 0.900 0.264 0.33</td><td>30.555 0.898 0.270 0.25</td><td>30.113 0.899 0.267 0.29</td></tr></table>

Table 8: ρ sweep for the linear-increment activation on Mip-NeRF 360 (top-ρN rule, identical schedule to Table 9). Overall averages are over all nine scenes; outdoor averages are over five scenes (Bicycle, Flowers, Garden, Stump, Treehill) and indoor averages over four scenes (Bonsai, Counter, Kitchen, Room). #G in millions of Gaussians. Overall SSIM/LPIPS are the subset averages recombined as $( 5 \mathrm { o u t } + 4 \mathrm { i n } ) / 9$ and may differ from the independently rounded values of Table 1 of the main paper in the last digit. The default $\rho { = } 0 . 3 0$ row matches Table 1 of the main paper.
<table><tr><td rowspan="2">ρ</td><td colspan="3">Overall (9 scenes) SSIM↑</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Outdoor (5 scenes)</td><td rowspan="2"></td><td colspan="3">Indoor (4 scenes)</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>#G (M)↓</td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td><td>#G (M)↓</td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td><td>#G (M)↓</td></tr><tr><td>0.10</td><td>26.94</td><td>0.7951</td><td>0.2478</td><td>0.31</td><td>24.58</td><td>0.7080</td><td>0.2780</td><td>0.45</td><td>29.89</td><td>0.9040</td><td>0.2100</td><td>0.14</td></tr><tr><td>0.20</td><td>27.42</td><td>0.8127</td><td>0.2194</td><td>0.63</td><td>24.81</td><td>0.7260</td><td>0.2510</td><td>0.91</td><td>30.68</td><td>0.9210</td><td>0.1800</td><td>0.28</td></tr><tr><td>0.30</td><td>27.70</td><td>0.8254</td><td>0.2095</td><td>0.94</td><td>25.01</td><td>0.7432</td><td>0.2394</td><td>1.36</td><td>31.07</td><td>0.9283</td><td>0.1723</td><td>0.43</td></tr><tr><td>0.40</td><td>27.73</td><td>0.8267</td><td>0.2031</td><td>1.26</td><td>25.02</td><td>0.7440</td><td>0.2320</td><td>1.81</td><td>31.11</td><td>0.9300</td><td>0.1670</td><td>0.57</td></tr><tr><td>0.50</td><td>27.74</td><td>0.8271</td><td>0.2012</td><td>1.57</td><td>25.03</td><td>0.7440</td><td>0.2310</td><td>2.26</td><td>31.12</td><td>0.9310</td><td>0.1640</td><td>0.71</td></tr></table>

Table 9: ρ sweep for the matched Gumbel-Sigmoid baseline on Mip-NeRF 360. Overall averages are over all nine scenes; outdoor averages are over five scenes (Bicycle, Flowers, Garden, Stump, Treehill) and indoor averages over four scenes (Bonsai, Counter, Kitchen, Room). #G in millions of Gaussians. As ρ grows, PSNR/LPIPS gains shrink rapidly for both activations while #G scales nearly linearly; even at $\rho { = } 0 . 5 0$ , this baseline does not reach the LinearMask-GS default (27.70 dB at 0.94M, $\rho { = } 0 . 3 0 ;$ see Table 1 of the main paper and Table 8).
<table><tr><td rowspan="2">ρ</td><td colspan="3">Overall (9 scenes)</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Outdoor (5 scenes)</td><td rowspan="2">SSIM↑</td><td colspan="3">Indoor (4 scenes)</td></tr><tr><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td><td>#G (M)↓</td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td><td>#G (M)↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>#G (M)↓</td></tr><tr><td>0.10</td><td>26.49</td><td>0.7773</td><td>0.2929</td><td>0.32</td><td>24.03</td><td>0.6775</td><td>0.3456</td><td>0.46</td><td>29.57</td><td>0.9021</td><td>0.2270</td><td>0.14</td></tr><tr><td>0.20</td><td>26.98</td><td>0.7968</td><td>0.2590</td><td>0.64</td><td>24.33</td><td>0.7011</td><td>0.3070</td><td>0.92</td><td>30.29</td><td>0.9164</td><td>0.1991</td><td>0.28</td></tr><tr><td>0.30</td><td>27.22</td><td>0.8037</td><td>0.2449</td><td>0.95</td><td>24.49</td><td>0.7093</td><td>0.2903</td><td>1.37</td><td>30.63</td><td>0.9217</td><td>0.1881</td><td>0.43</td></tr><tr><td>0.40</td><td>27.33</td><td>0.8073</td><td>0.2370</td><td>1.26</td><td>24.53</td><td>0.7136</td><td>0.2812</td><td>1.82</td><td>30.82</td><td>0.9244</td><td>0.1819</td><td>0.57</td></tr><tr><td>0.50</td><td>27.33</td><td>0.8091</td><td>0.2321</td><td>1.57</td><td>24.55</td><td>0.7158</td><td>0.2751</td><td>2.26</td><td>30.82</td><td>0.9257</td><td>0.1783</td><td>0.71</td></tr></table>

Table 10: Per-scene PSNR (dB, ) of direct top-30% pruning by the raw max-pooled score vs. LinearMask-GS on Mip-NeRF 360. Identical schedule and fine-tuning; the only difference is the learned mask. The learned mask wins on eight of nine scenes, the exception being Treehill.
<table><tr><td>Method</td><td>Bicycle</td><td>Bonsai</td><td>Counter</td><td>Kitchen</td><td>Room</td><td>Stump</td><td>Garden</td><td>Flowers</td><td>Treehill</td><td>Avg.</td></tr><tr><td>Direct top-30% by Si</td><td>25.4887</td><td>31.8328</td><td>28.8942</td><td>31.3281</td><td>31.5274</td><td>26.7869</td><td>27.5812</td><td>21.6934</td><td>23.0198</td><td>27.5725</td></tr><tr><td>LinearMask-GS (Ours)</td><td>25.6700</td><td>32.0200</td><td>29.0500</td><td>31.5000</td><td>31.7000</td><td>26.9400</td><td>27.7500</td><td>21.8600</td><td>22.8500</td><td>27.7044</td></tr><tr><td>Δ</td><td>+0.1813</td><td>+0.1872</td><td>+0.1558</td><td>+0.1719</td><td>+0.1726</td><td>+0.1531</td><td>+0.1688</td><td>+0.1666</td><td>-0.1698</td><td>+0.1319</td></tr></table>

Table 11: Budget-matched comparison with Speedy-Splat on Mip-NeRF 360, averaged over nine scenes. LinearMask-GS<sup>⋆</sup> rows use the FastGS backbone; all rows are means over three independent seeds. At the 0.30M budget the three seeds give 27.38, 27.40 and 27.41 dB, a spread of 0.03 dB, so even the weakest seed leads Speedy-Splat by 0.47 dB.
<table><tr><td>Method</td><td>#G↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Speedy-Splat</td><td>0.30M</td><td>26.91</td><td>0.781</td><td>0.295</td></tr><tr><td>Ours</td><td>0.25M</td><td>27.06</td><td>0.791</td><td>0.241</td></tr><tr><td>Ours</td><td>0.30M</td><td>27.40</td><td>0.798</td><td>0.230</td></tr><tr><td>Ours</td><td>0.35M</td><td>27.60</td><td>0.802</td><td>0.224</td></tr></table>

Table 12: DropGaussian + LinearMask-GS on five LLFF scenes (9-view). PSNR values for the masked model are means over three independent training runs (the public code exposes no seed flag, so repetitions correspond to independent launches); the unpruned and Gumbel-variant reference columns are the single-run values also used in Table 3 of the main paper. Note the two reference points: ∆PSNR is measured against the unpruned backbone, whereas the reduction is measured against the Gumbel variant, following the convention of Section 4.4 of the main paper.
<table><tr><td rowspan="2">Scene</td><td colspan="3">PSNR↑ (dB)</td><td colspan="3">#Primitives↓</td></tr><tr><td>No pruning</td><td>+ LinearMask-GS</td><td>Δ</td><td>+ LP-3DGS (Gumbel)</td><td>+ LinearMask-GS</td><td>Reduction</td></tr><tr><td>fern</td><td>26.6495</td><td>26.9623</td><td>+0.3128</td><td>62,182</td><td>36,656</td><td>41.05%</td></tr><tr><td>flower</td><td>27.8341</td><td>28.2594</td><td>+0.4253</td><td>82,070</td><td>48,233</td><td>41.23%</td></tr><tr><td>fortress</td><td>31.2183</td><td>31.7365</td><td>+0.5182</td><td>72,125</td><td>42,431</td><td>41.17%</td></tr><tr><td>horns</td><td>27.9452</td><td>28.4328</td><td>+0.4876</td><td>90,304</td><td>53,379</td><td>40.89%</td></tr><tr><td>room</td><td>32.6074</td><td>32.9710</td><td>+0.3636</td><td>57,442</td><td>33,741</td><td>41.26%</td></tr><tr><td>Mean</td><td>29.2509</td><td>29.6724</td><td>+0.4215</td><td>72,825</td><td>42,888</td><td>41.12%</td></tr></table>

Table 13: Octree-GS + LinearMask-GS per-scene results. Amsterdam is the Bungee-NeRF scene of Table 3 of the main paper; Bicycle and Garden are the two added Mip-NeRF 360 scenes. As in Table 12, ∆PSNR is measured against the unpruned backbone and the reduction against the Gumbel variant.
<table><tr><td rowspan="2">Scene</td><td colspan="3">PSNR↑ (dB)</td><td colspan="3">#Anchors↓</td></tr><tr><td>No pruning</td><td>+ LinearMask-GS</td><td>Δ</td><td>+ LP-3DGS (Gumbel)</td><td>+ LinearMask-GS</td><td>Reduction</td></tr><tr><td>amsterdam (Bungee-NeRF)</td><td>27.8570</td><td>27.9720</td><td>+0.1150</td><td>648,221</td><td>380,726</td><td>41.27%</td></tr><tr><td rowspan="2">bicycle (Mip-NeRF 360) garden (Mip-NeRF 360)</td><td>25.1034</td><td>25.0053</td><td>-0.0981</td><td>600,012</td><td>356,827</td><td>40.53%</td></tr><tr><td>27.1481</td><td>27.0226</td><td>-0.1255</td><td>500,045</td><td>300,877</td><td>39.83%</td></tr><tr><td>Mean (2 added scenes)</td><td>26.1258</td><td>26.0140</td><td>-0.1118</td><td>550,029</td><td>328,852</td><td>40.18%</td></tr></table>

Table 14: Decomposition of the mask-learning speed gap between 3DGS LinearMask and the FastGS-backbone variant (LinearMask-GS<sup>⋆</sup>) on Mip-NeRF 360, both measured on the same RTX 4090. Both per-factor speedups are directly measured. The #Gaussians values are averaged over the nine scenes. The two factors combine to $6 . 3 6 \times 3 . 7 \approx 2 4 \times$ , consistent with the measured mask-learning times of 110 s (3DGS backbone) and 4.6 s (FastGS backbone), since $1 1 0 / 2 4 \approx 4 . 6$
<table><tr><td># Factor</td><td></td><td>3DGS LinearMask</td><td>FastGS (LinearMask-GS*)</td><td>Speedup</td></tr><tr><td></td><td>1 #Gaussians at mask window Pre-prune 3.14 M (9- Pre-prune 0.493 M</td><td>scene avg)</td><td>(9-scene avg)</td><td>6.36×</td></tr><tr><td></td><td>2 Rasterizer implementation</td><td>Standard Inria raster- FastGS rasterizer (far izer (full bbox per fewer tiles per splat) splat)</td><td></td><td>3.7× ≈24×</td></tr><tr><td></td><td colspan="4">Combined speedup (6.36 × 3.7) Measured mask-learning time (3DGS vs. FastGS backbone) 110 s vs. 4.6 s Consistency check:  $1 1 0 \mathrm { s } / 2 4 \approx 4 . 6 \mathrm { s }$ </td></tr></table>

Table 15: Net training-time effect of LinearMask- $\mathbf { \nabla } . \mathrm { G S ^ { \star } }$ on the FastGS backbone, averaged over the nine Mip-NeRF 360 scenes. The added mask-learning cost (4.6 s) is more than offset by the faster 20k–30k iteration range (10.9 s saved from rendering fewer primitives), for a net saving of 6.3 s per scene.
<table><tr><td>Component</td><td>Time (s)</td></tr><tr><td>20k–30k range, FastGS native 20k–30k range,  ${ \mathrm { L i n e a r M a s k  – G S } } ^ { \star }$ </td><td>39.2 28.3</td></tr><tr><td>Range speedup</td><td>-10.9</td></tr><tr><td>Added mask-learning cost</td><td>+4.6</td></tr><tr><td>Net change per scene</td><td>-6.3</td></tr></table>

## References

[1] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In CVPR, pages 5470–5479, 2022.

[2] Youyu Chen, Junjun Jiang, Kui Jiang, Xiao Tang, Zhihao Li, Xianming Liu, and Yinyu Nie. DashGaussian: Optimizing 3D gaussian splatting in 200 seconds. arXiv preprint arXiv:2503.18402, 2025.

[3] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In ECCV, pages 165–181, 2024.

[4] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, pages 5501–5510, 2022.

[5] Alex Hanson, Allen Tu, Geng Lin, Vasu Singla, Matthias Zwicker, and Tom Goldstein. Speedy-Splat: Fast 3D gaussian splatting with sparse pixels and sparse primitives. arXiv preprint arXiv:2412.00578, 2025.

[6] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3D gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4):139–1, 2023.

[7] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3DGS: High-quality radiance fields with limited resources. In ACM SIGGRAPH Asia Conference Papers, pages 1–11, 2024.

[8] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 41(4):1–15, 2022.

[9] Shiwei Ren, Tianci Wen, Yongchun Fang, and Biao Lu. FastGS: Training 3D gaussian splatting in 100 seconds. arXiv preprint arXiv:2511.04283, 2025.

[10] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang, Cheng Peng, Rama Chellappa, and Deliang Fan. LP-3DGS: Learning to prune 3D gaussian splatting. In NeurIPS, pages 122434–122457, 2024.