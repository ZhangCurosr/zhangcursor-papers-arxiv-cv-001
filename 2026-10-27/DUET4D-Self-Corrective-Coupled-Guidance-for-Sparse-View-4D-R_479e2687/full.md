# DUET4D: Self-Corrective Coupled Guidance for Sparse-View 4D Reconstruction

Beizhen Zhao<sup>1∗</sup>, Nanjie Yao<sup>1∗</sup>, Boyi Fu<sup>1</sup>, Qingsong Yan<sup>2</sup>, Haofan Ren<sup>2</sup>, Kaiyong Zhao<sup>2</sup>, Hao Wang<sup>1†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup>XGRIDS

bzhao610@connect.hkust-gz.edu.cn, nyao656@connect.hkust-gz.edu.cn, haowang@hkust-gz.edu.cn

![](images/89bd4b6f80fd44ef4ef6725df8a3b0d117ec63124dd08949e556cfa2f4f57b00.jpg)

![](images/e4ea2f7936589daad9ef03556f0f33bbddf847316706e83c81007b194813fa21.jpg)  
A Self-corrective Coupled Guidance Framework

![](images/8993ce5d0e018d80497886d96e16a46936bca88ad9d1c9923ffa3f1e6ba625f7.jpg)

Figure 1: We present DUET4D, a self-corrective coupled guidance framework for Sparse-View 4D Reconstruction. Compared with existing methods, we achieve better rendering quality with substantially higher rendering FPS and a more compact Gaussian representation.

## Abstract

Sparse-view 4D reconstruction is highly under-constrained when only a few cameras are available. Generative priors can recover missing appearances, but video models are computationally impractical for iterative reconstruction, whereas single-view models produce temporally inconsistent pseudoobservations. Moreover, directly absorbing uncertain generative outputs into Gaussian optimization often creates redundant primitives and floating artifacts. To address these challenges, we propose DUET4D, a self-corrective coupled guidance framework for sparse-view 4D Gaussian splatting. The core of our approach is a self-corrective framework where the 4D representation guides generative prior for spatio-temporal consistent supervisions. We first construct sequential alignments from current Gaussian evidence to steer the latent restoration. Then we use Sequential Fidelity Resampling to lift generative supervisions into Gaussian space to selectively distinguish repeatedly supported primitives. The corrected representation then provides more reliable Gaussian evidence for the next restoration cycle. Extensive experiments demonstrate that DUET4D significantly improves rendering quality and rendering eficiency.

## 1 Introduction

By extending 3D Gaussian Splatting to the temporal domain, 4D Gaussian methods can reconstruct and render dynamic

<sup>∗</sup>These authors contributed equally. <sup>†</sup>Corresponding author. Copyright © 2027. All rights reserved.

scenes with high visual quality (Cai et al. 2022; Du et al. 2021; Fang et al. 2022; Kratimenos, Lei, and Daniilidis 2024; Li et al. 2022). However, most existing methods rely on sufficiently dense multi-view observations (Shi et al. 2026; Pan et al. 2026; Zhou et al. 2026). When the input is reduced to only a few cameras, reconstruction quality degrades severely. Many regions of the scene are never directly observed, while dynamic objects are only partially constrained across time. As a result, sparse-view 4D Gaussian reconstruction often sufers from distorted artifacts, floating Gaussians, and temporally inconsistent motion. Existing works (Shi et al. 2026; Zhou et al. 2026) introduce texture regularization and Gaussian decaying to constrain sparse-view reconstruction. However, they are insuficient to recover stable dynamic appearance under large viewpoint drifts.

To introduce stronger generative priors, Sparse-Cam4D (Pan et al. 2026) utilizes video difusion models (VDM) to synthesize pseudo-observations, but iteratively querying a VDM is computationally impractical for multiview 4D optimization. This presents the first challenge: single-view difusion models (SDMs) provide a more eficient alternative, yet their independent predictions lack crossview and temporal awareness, leading to temporal incoherent motions. However, temporal coherence alone does not make pseudo-supervision uniformly reliable. There remains a spatial drift between the generative supervisions and the sparse observations. Blindly densifying these locations leads to redundant primitives and floating Gaussians. A second challenge is therefore to selectively lift pseudo supervision into reliable representation-side decisions.

To address these challenges, we propose DUET4D, a selfcorrective framework that couples a frozen single-view difusion model with an evolving 4D Gaussian representation. Instead of relying on a computationally heavy VDM or directly using independently restored views as supervision, DUET4D exploits the current 4D Gaussians as a spatiotemporal correspondence carrier. The resulting Gaussian-derived alignments and confidence estimates construct a reliable anchor that steers the latent denoising direction of a frozen SDM, enabling temporally coherent pseudo-observations without training or invoking an additional video model.

Crucially, DUET4D does not indiscriminately absorb these generated observations into the representation. The Sequential Fidelity Resampling distinguishes reliable reconstruction deficits from persistent inconsistencies by jointly considering confidence-gated pseudo supervisions and real observations, and then lifts these 2D signals to individual 3D primitives. The updated representation consequently provides cleaner correspondence evidence for the next restoration round, forming a closed self-corrective cycle that improves reconstruction fidelity while suppressing redundant Gaussians.

Our contributions are summarized as follows:

• We propose DUET4D, a unified framework for sparseview 4D reconstruction, establishing a self-corrective framework where the 4D representation guides generative prior for spatio-temporal consistent supervisions.

• We introduce a training-free anchor-driven latent steering mechanism that uses Gaussian-derived correspondences to make a frozen SDM produce temporally coherent pseudo-supervisions.

• We design Sequential Fidelity Resampling to selectively lift confidence-supported pseudo-view discrepancies into Gaussian densification while pruning primitives that remain inconsistent with real observations, producing a cleaner and more compact representation.

## 2 Related Work

Sparse-view Dynamic Reconstruction Dynamic scene reconstruction requires temporally coherent modeling of appearance and motion. While NeRF-based methods rely on time-conditioned fields or feature factorization (Mildenhall et al. 2021; Pumarola et al. 2021; Park et al. 2021a,b; Cao and Johnson 2023; Fridovich-Keil et al. 2023),4DGS-based approaches significantly improve eficiency via deformation fields, tracking, or direct 4D representations (Kerbl et al. 2023; Yang et al. 2024a; Li et al. 2024; Luiten et al. 2024; Lin et al. 2024; Wu et al. 2024; Yang et al. 2024b; Duan et al. 2024). However, these methods typically require dense, synchronized multi-view videos and degrade heavily under sparse camera setups. To tackle the under-constrained sparseview setting, prior works exploit category-specific priors like SMPL for humans (Peng et al. 2021a,b; Weng et al. 2022; Hu, Hu, and Liu 2024) or monocular depth and tracking cues (Lei et al. 2025; Wang et al. 2025; Pradeep et al. 2013). Though providing useful temporal constraints, recent methods (Shi et al. 2026; Zhou et al. 2026)still sufer from unstable appearance under large viewpoint changes.

Difusion Models for Novel-View Synthesis Difusion models serve as powerful priors for 3D/4D generation. While object-centric models (Liu et al. 2023, 2024; Shi et al. 2024; Long et al. 2024) achieve strong multi-view consistency, they cannot directly handle scene-level dynamic reconstruction. Recent camera-controlled video difusion models (Wang et al. 2024; He et al. 2024; Xu et al. 2024a; Yu et al. 2024; Bai et al. 2025; Wu et al. 2025) synthesize novel-view dynamic observations, ofering richer supervision. However, they rely on expensive multi-frame denoising and often produce crossview or temporal inconsistencies that harm 4D optimization. To incorporate difusion into sparse dynamic reconstruction, recent methods such as Difuman4D (Jin et al. 2025) and SparseCam4D (Pan et al. 2026) query video difusion models for better rendering quality, which take 40 hours for a single dynamic scene, facing a strict trade-of between consistency and eficiency. We address this through a unified framework that constructing a self-corrective coupled pipeline where the 4D representation guide generative prior for spatio-temporal consistent supervisions.

## 3 Methodology

## 3.1 Sequential Anchor Alignment

Independent single-view restoration can recover missing appearance details, but its correction is estimated separately for each virtual view and therefore does not constitute reliable 4D supervision. Rather than introducing an additional video model or correspondence network, we use the evolving 4D Gaussian representation itself to determine which restoration changes are supported across nearby views and time. At each restoration round, a virtual node $i = ( v , t )$ provides the current Gaussian rendering $x _ { i }$ and its independently restored output $o _ { i }$ . We retain only the bounded restoration residual $r _ { i } = \mathrm { c l i p } ( o _ { i } - x _ { i } , - \kappa _ { r } , \kappa _ { r } )$ , so that the alignment transfers appearance corrections instead of directly replacing the geometry, motion, and view-dependent content already encoded by the current representation.

For a neighboring node $j \in \mathcal { N } ( i )$ , we track the same canonical Gaussians after deformation to construct a targetindexed backward sampling grid $W _ { j  i }$ . The reliability of this representation-derived correspondence is measured by

$$
\begin{array} { r l } & { c _ { j \to i } ( p ) = \mathbb { I } _ { \mathrm { v a l i d } } ( p ) s ( \alpha _ { i } ( p ) ) s ( W _ { j \to i } ( \alpha _ { j } ) ( p ) ) } \\ & { \qquad \cdot \exp \left( - \frac { e _ { \mathrm { d e p } } ( p ) } { \tau _ { \mathrm { d e p } } } - \frac { e _ { \mathrm { c y c } } ( p ) } { \tau _ { \mathrm { c y c } } } - \frac { e _ { \mathrm { p h o } } ( p ) } { \tau _ { \mathrm { p h o } } } \right) . } \end{array}\tag{1}
$$

where $\mathbb { I } _ { \mathrm { v a l i d } }$ rejects invalid projections and $s ( \alpha )$ suppresses low-opacity correspondences. The depth, cycle, and photometric terms jointly exclude mappings that are geometrically unsupported or inconsistent with the current representation. Consequently, all alignment evidence is obtained from the evolving 4DGS without requiring external optical flow, monocular depth, or video difusion.

We then align neighboring restoration residuals rather than their complete RGB predictions. A neighbor contributes only when its warped residual agrees with the target restoration:

![](images/1eea660d94094771f040ae011ddf3314042c81ac57e2e34159e52c0c48def195.jpg)  
Figure 2: Overview of DUET4D. First, we construct sequential anchor alignment through Gaussian evidence to generate temporal anchor frames and confidence map. Then we design a training-free anchor-driven latent steering mechanism within the latent space of SDM for temporally consistent generative outputs. The frames and confidence map will be used to locate Gaussians with weak spatial and temporal consistency for optimization. The reserved Gaussians are controlled by a spatial grid and Fourier spline priors for motion smoothness and regularization.

$$
\begin{array} { l } { { a _ { j i } ( p ) = \lambda _ { j i } c _ { j  i } ( p ) \exp ( - \frac { \displaystyle \vert W _ { j  i } ( r _ { j } ) ( p ) - r _ { i } ( p ) \vert _ { 1 } } { \sigma _ { r } } ) , } } \\ { { \bar { r } _ { i } ( p ) = \displaystyle \frac { \sum _ { j \in \mathcal { N } ( i ) } a _ { j i } ( p ) W _ { j  i } ( r _ { j } ) ( p ) } { \sum _ { j \in \mathcal { N } ( i ) } a _ { j i } ( p ) + \delta } . } } \end{array}\tag{2}
$$

To handle pixels with insuficient support near occlusions, motion boundaries, or image borders, we jointly define the accumulated support and its bounded confidence mask as

$$
\begin{array} { l } { \displaystyle { A _ { i } ( p ) = \frac { 1 } { | \mathcal { N } ( i ) | } \sum _ { j \in \mathcal { N } ( i ) } a _ { j i } ( p ) , } } \\ { \displaystyle { m _ { i } ( p ) = \mathrm { c l i p } \bigg ( \frac { A _ { i } ( p ) } { 1 + A _ { i } ( p ) } , 0 , m _ { \mathrm { m a x } } \bigg ) . } } \end{array}\tag{3}
$$

Here, $m _ { \operatorname* { m a x } } \le 1$ bounds the trust assigned to neighboring residuals. The mask is zero without valid support and increases with accumulated correspondence confidence.

Using $\kappa _ { c }$ to bound the correction proposed by neighboring residuals, we directly construct the spatiotemporal anchor as

$$
x _ { i } ^ { \mathrm { s t } } ( p ) = x _ { i } ( p ) + r _ { i } ( p ) + \mathrm { c l i p } ( \bar { r } _ { i } ( p ) - r _ { i } ( p ) , - \kappa _ { c } , \kappa _ { c } )\tag{4}
$$

All independent outputs $o _ { i }$ are computed before aggregation, and no aggregated residual is reused. The construction is therefore independent of graph traversal order and avoids recursive error propagation within one aggregation round.

## 3.2 Anchor-Driven Latent Steering

Given a rendered image and its fixed restoration configuration, the restorer performs a single denoising step in the latent space and produces two outputs: an internal model prediction ϵ and a decoded restored RGB image o. The internal prediction is used for dual-path latent steering, whereas the decoded RGB image is used only to construct the final pseudo-observation.

Specifically, we run the same frozen SDM restorer through two parallel paths: a base path and an anchor-guided path. The base and anchor-guided paths take $x _ { i }$ and $x _ { i } ^ { \mathrm { s t } }$ as input, respectively, yielding internal predictions $\epsilon _ { b }$ and $\epsilon _ { g }$ . Both predictions share the condition and the parameters. To extract a confidence-aware guidance signal, we compare the two prediction directions under the mask $m _ { i }$ constructed in Eq. 3. This mask reflects the local 4DGS-supported anchor correspondence consistency, so the anchor-guided path is trusted mainly in regions with strong valid support. Inspired by previous works (Ho and Salimans 2022; Song et al. 2025), we obtain the guided prediction as

$$
\epsilon ^ { * } = \epsilon _ { g } + \rho \cdot \sin \theta \left( \epsilon _ { g } - \frac { \| \epsilon _ { g } \| _ { m } } { \| \epsilon _ { b } \| _ { m } + \delta } \cos \theta \cdot \epsilon _ { b } \right) ,\tag{5}
$$

where cos θ = clip $( \langle \epsilon _ { g } , \epsilon _ { b } \rangle _ { m } / ( \| \epsilon _ { g } \| _ { m } \cdot \| \epsilon _ { b } \| _ { m } + \delta ) , - 1 , 1 )$ is the cosine-similarity between the two predictions, while sin $\theta = \sqrt { 1 - \cos ^ { 2 } \theta }$ is the sine counterpart. $\| \cdot \| _ { m }$ denotes the weighted norm, and $\rho$ controls the guidance strength.

The parenthesized term extracts the confidence-weighted component of $\epsilon _ { g }$ orthogonal to $\epsilon _ { b }$ after magnitude alignment, while sin θ scales the guidance by the orthogonal discrepancy. The guidance encourages spatiotemporal consistency in well-supported regions while preserving local restoration details and limiting the influence of unsupported correspondences.

The guided prediction $\epsilon ^ { * }$ is decoded into a guided restoration $\tilde { o } _ { i } .$ Although the confidence mask has already been used to estimate a guidance direction, the resulting restoration may still contain uncertain content in regions afected by occlusions, motion boundaries, or inaccurate correspondences. We therefore apply the confidence mask once more in RGB space to selectively integrate the guided restoration:

$$
y _ { i } = o _ { i } + m _ { i } \odot \left( \tilde { o } _ { i } - o _ { i } \right) .\tag{6}
$$

Supported regions thus adopt the spatiotemporally guided restoration, whereas uncertain regions revert to the independently restored result $o _ { i }$ , preventing unsupported neighboring evidence from being explicitly fused into the final pseudo-observation. Finally, the fused output and its confidence mask serve as pseudo-supervision and confidence input, respectively, for optimizing the 4DGS representation.

## 3.3 Sequential Fidelity Resampling

Although the generative supervisions y are temporally more coherent, they are still image-space predictions with spatially varying reliability. Directly treating every large 2D gradints as a densification signal would consequently absorb generative uncertainty into the explicit representation. To mitigate this ambiguity, we introduce Sequential Fidelity Resampling, which evaluates the current Gaussians in two complementary evidence domains.

At the r-th resampling cycle, we sample a set of virtualcamera nodes $ { \mathcal { S } } _ { \mathrm { { p s } } }$ and render the current Gaussian representation as $x _ { i }$ at each node $\boldsymbol { i } ~ = ~ ( \boldsymbol { v } , t )$ . Given the fused pseudo-observation $y _ { i }$ and the correspondence confidence $m _ { i }$ inherited from Secs. 3.1 and 3.2, we first identify reconstruction deficits as

$$
b _ { i } ^ { \mathrm { p s } } ( p ) = \mathbb { I } [ e _ { i } ^ { \mathrm { p s } } ( p ) > \tau _ { e } ~ \land ~ m _ { i } ( p ) > \tau _ { m } ] ,\tag{7}
$$

where $e _ { i } ^ { \mathrm { p s } } ( p ) \ = \ \| x _ { i } ( p ) - y _ { i } ( p ) \|$ denotes the photometric discrepancy. The confidence gate suppresses residuals around unsupported motion boundaries, so that pseudo-view discrepancies are used only when they are supported by the current spatiotemporal correspondence.

Let $a _ { i k } ( p )$ denote the rasterized contribution of Gaussian $g _ { k }$ to pixel p at node i. We lift the confidence-gated residuals into Gaussian space and define the pseudo-view support as

$$
S _ { k , r } ^ { \mathrm { p s } } = \frac { 1 } { | S _ { \mathrm { p s } } | } \sum _ { i \in  { S _ { \mathrm { p s } } } } \frac { \sum _ { p } a _ { i k } ( p ) b _ { i } ^ { \mathrm { p s } } ( p ) e _ { i } ^ { \mathrm { p s } } ( p ) } { \sum _ { p } a _ { i k } ( p ) + \delta } .\tag{8}
$$

Since the score is accumulated over multiple spatial and temporal nodes, a high $S _ { k , r } ^ { \mathrm { p s } }$ indicates that $g _ { k }$ repeatedly contributes to reliable under-reconstructed regions.

We separately evaluate the same representation using the real observations. Let $ { S _ { \mathrm { o b s } } }$ denote the sparse input-camera

nodes and $I _ { i }$ denote the corresponding ground-truth image. Their photometric discrepancy is $e _ { i } ^ { \mathrm { o b s } } ( \bar { p } )$ . For each Gaussian that is suficiently visible in the observed views, we compute

$$
E _ { k , r } ^ { \mathrm { o b s } } = \frac { \sum _ { i \in S _ { \mathrm { o b s } } } \sum _ { p } a _ { i k } ( p ) e _ { i } ^ { \mathrm { o b s } } ( p ) } { \sum _ { i \in S _ { \mathrm { o b s } } } \sum _ { p } a _ { i k } ( p ) + \delta } ,\tag{9}
$$

A high $E _ { k , r } ^ { \mathrm { o b s } }$ provides direct evidence that a visible primitive remains incompatible with the sparse real observations.

We use the two accumulated signals to regulate Gaussian topology. A primitive is selected for pruning only when it is not supported by persistent fidelity evidence. Specifically, we define

$$
\mathcal { P } _ { r } = \left\{ g _ { k } \middle | \left( E _ { k , r } ^ { \mathrm { o b s } } > \tau _ { p } \right) \wedge \left( S _ { k , r } ^ { \mathrm { p s } } < \tau _ { p } \right) \right\} .\tag{10}
$$

The prune policy considers both real observations and generative supervisions. This prevents such residuals from being repeatedly interpreted as missing geometry.

For the remaining primitives, we therefore define the densification set as

$$
\mathcal { D } _ { r } = \left\{ g _ { k } \middle | g _ { k } \notin \mathcal { P } _ { r } , S _ { k , r } ^ { \mathrm { p s } } > \tau _ { d } , \right\} .\tag{11}
$$

Sequential Fidelity Resampling therefore uses the pseudoview space to propose repeatedly supported reconstruction deficits and the observed-view space to reject contradictory primitives, selectively lifting generative supervision into Gaussian-space densification and pruning rather than blindly following image-space residuals.

## 3.4 Smoothness Control Regularization

To avoid locally inconsistent motion under sparse observations, we therefore apply a smoothness control after resampling stage to regularize the motion of the active Gaussians. We partition the canonical space using a multi-resolution hash grid feature $\mathbf { h } _ { \mathrm { s p } } ( \pmb { \mu } _ { k } )$ for each Gaussian center $\pmb { \mu } _ { k }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s p } } = \mathbb { E } _ { k , \Delta \mu } \left[ \left\| \mathbf { h } _ { \mathrm { s p } } ( \pmb { \mu } _ { k } + \Delta \pmb { \mu } ) - \mathbf { h } _ { \mathrm { s p } } ( \pmb { \mu } _ { k } ) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{12}
$$

This constraint encourages neighboring locations to produce compatible deformation features while preserving motion discontinuities between suficiently separated grid regions.

For each active Gaussian, we use a set of spline basis functions $\mathbf { h } _ { \mathrm { t m p } , k } ( t )$ to provide a continuous temporal trajectory and Smoothness Regularization:

$$
\mathcal { L } _ { \mathrm { t m p } } = \mathbb { E } _ { k , t } \left[ \left\| \mathbf { d } _ { k } ( t + \Delta t ) - 2 \mathbf { d } _ { k } ( t ) + \mathbf { d } _ { k } ( t - \Delta t ) \right\| _ { 2 } ^ { 2 } \right] ,\tag{13}
$$

where $\mathbf { d } _ { k }$ denotes temporal deformations with spline prior. This term preserves non-rigid displacement while reducing local trajectory jitter.

The corrected and motion-regularized representation is subsequently used to render the next spatiotemporal anchor graph, providing cleaner correspondence evidence for the following restoration cycle and thereby closing the selfcorrective coupled of DUET4D.

Table 1: Quantitative comparisons on the Nvidia Dynamic Scenes (Yoon et al. 2020), Neural 3D Video (Li et al. 2022), and Google Immersive (Broxton et al. 2020) datasets. The best and second-best results are highlighted in bold and underlined, respectively. Our method outperforms all baseline methods across all evaluated metrics. “↑” indicates higher is better, while “↓” indicates lower is better. Specifically, DUET4D<sup>∗</sup> uses exactly the same preprocessed inputs as SparseCam4D, while all mode components and optimization settings remain unchanged.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2">FPS↑</td><td colspan="4">Nvidia Dynamic Scenes</td><td rowspan="2">FPS↑</td><td colspan="4">Neural 3D Video</td><td colspan="5">Google Immersive</td></tr><tr><td> $N _ { \mathrm { G S } } \downarrow$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> $N _ { \mathrm { G S } } \downarrow$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FPS↑</td><td> $N _ { \mathrm { G S } } \downarrow$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>4DGS</td><td>CVPR 2024</td><td>54</td><td>0.26M</td><td>16.28</td><td>0.325</td><td>0.525</td><td>49</td><td>0.21M</td><td>16.87</td><td>0.665</td><td>0.380</td><td>99</td><td>0.84M</td><td>14.27</td><td>0.517</td><td>0.555</td></tr><tr><td>4D-Rotor</td><td>SIGGRAPH 2024</td><td>7</td><td>0.35M</td><td>16.43</td><td>0.352</td><td>0.482</td><td>5</td><td>0.38M</td><td>19.48</td><td>0.738</td><td>0.315</td><td>10</td><td>0.71M</td><td>14.46</td><td>0.533</td><td>0.490</td></tr><tr><td>RealTime4DGS</td><td>ICLR 2024</td><td>11</td><td>0.38M</td><td>16.02</td><td>0.359</td><td>0.545</td><td>12</td><td>0.46M</td><td>17.53</td><td>0.652</td><td>0.434</td><td>21</td><td>1.09M</td><td>14.33</td><td>0.521</td><td>0.553</td></tr><tr><td>Grid4D</td><td>NeurIPS 2024</td><td>16</td><td>1.15M</td><td>16.35</td><td>0.334</td><td>0.538</td><td>80</td><td>0.21M</td><td>18.30</td><td>0.709</td><td>0.391</td><td>16</td><td>0.46M</td><td>12.47</td><td>0.492</td><td>0.566</td></tr><tr><td>Ex4DGS</td><td>NeurIPS 2024</td><td>61</td><td>2.98M</td><td>16.74</td><td>0.373</td><td>0.523</td><td>32</td><td>0.57M</td><td>19.21</td><td>0.730</td><td>0.304</td><td>49</td><td>1.36M</td><td>15.36</td><td>0.565</td><td>0.531</td></tr><tr><td>4C4D</td><td>CVPR 2026</td><td>52</td><td>4.06M</td><td>16.69</td><td>0.352</td><td>0.520</td><td>25</td><td>4.20M</td><td>16.62</td><td>0.653</td><td>0.424</td><td>35</td><td>3.02M</td><td>16.99</td><td>0.614</td><td>0.507</td></tr><tr><td>DUET4D (Ours)</td><td></td><td>127</td><td>0.19M</td><td>16.84</td><td>0.480</td><td>0.420</td><td>246</td><td>0.05M</td><td>20.28</td><td>0.748</td><td>0.296</td><td>214</td><td>0.16M</td><td>17.42</td><td>0.633</td><td>0.442</td></tr><tr><td>SparseCam4D</td><td>CVPR 2026</td><td>11</td><td>0.30M</td><td>23.92</td><td>0.733</td><td>0.251</td><td>8</td><td>0.39M</td><td>21.33</td><td>0.747</td><td>0.289</td><td>12</td><td>1.02M</td><td>19.21</td><td>0.674</td><td>0.355</td></tr><tr><td>DUET4D (Ours) *</td><td></td><td>123</td><td>0.12M</td><td>24.87</td><td>0.744</td><td>0.229</td><td>263</td><td>0.04M</td><td>22.04</td><td>0.789</td><td>0.258</td><td>228</td><td>0.13M</td><td>20.25</td><td>0.692</td><td>0.315</td></tr></table>

![](images/bb63e3adc355aa285348d9ab5561e6ce052ac45e8afa2da63ec07c44fd0767ef.jpg)  
GT  
Ours  
4DGS  
4C4D  
Grid4D  
Figure 3: Qualitative comparisons of state-of-the-art methods. Visual diferences are highlighted with red insets for better clarity. Our approach outperforms other models on diferent scenes, demonstrating advantages in challenging scenarios. Best viewed in color.

## 4 Experiments

## 4.1 Experiment Setup

Datasets. Following previous works (Pan et al. 2026; Li et al. 2024), we evaluate the proposed DUET4D on three widely-used sparse-view 4D reconstruction benchmarks: Neu3d (Li et al. 2022), Nvidia Dynamic (Yoon et al. 2020) and Google Immersive Dataset (Broxton et al. 2020) dataset.

Evaluation Metrics. Following standard practices for novel view synthesis, we employ established quantitative metrics to assess rendering quality, including PSNR, SSIM (Wang et al. 2004), and LPIPS (Zhang et al. 2018). We additionally report the temporal-PSNR (tPSNR) to evaluate the temporal consistency for the ablation study. To further evaluate rendering eficiency, we report the number of Gaussians (N<sub>GS</sub>) and rendering frames per second (FPS).

Baselines. We compare DUET4D with representative dynamic Gaussian reconstruction methods, including 4DGS (Wu et al. 2024), 4D-Rotor (Duan et al. 2024), Real-Time4DGS (Yang et al. 2024b), Ex4DGS (Lee et al. 2024), and Grid4D (Xu et al. 2024b), as well as recent sparse-view methods 4C4D (Zhou et al. 2026), and SparseCam4D (Pan et al. 2026). For methods whose oficial sparse-view results are unavailable, we reproduce them using the same input cameras and evaluation protocol. Notably, Sparse-Cam4D adopts a dedicated preprocessing pipeline. To exclude improvements caused by data processing, we additionally construct a strictly controlled comparison. Specifically, DUET4D<sup>∗</sup> uses exactly the same preprocessed inputs as SparseCam4D, while all model components and optimization settings remain unchanged.

## 4.2 Evaluation

Quantitative Comparison. As reported in Table 1, DUET4D consistently achieves the highest average PSNR across all evaluated datasets. This performance is particularly notable in dynamic scenes exhibiting complex, non-rigid deformations. The superiority of our approach is attributed to the self-corrective coupled guidance, which efectively constrains the restoration process and filters out unreliable pseudo-supervision. Consequently, DUET4D significantly boosts PSNR while maintaining highly competitive SSIM and LPIPS scores. Furthermore, to ensure a fair evaluation against SparseCam4D, we adopt an identical preprocessing pipeline. Our method not only outperforms SparseCam4D in reconstruction fidelity but also operates at a significantly higher frame rate.

Qualitative Comparison. As illustrated in Fig. 3, whereas baseline methods inherently sufer from incomplete geometry, blurred textures, or floating artifacts in heavily occluded dynamic regions, DUET4D produces sharper and temporally coherent renderings. This robustness validates that our core designs—sequential anchor alignment, latent steering, and fidelity resampling—collectively cooperation to suppress frame-wise drift, preserve high-frequency details, and eliminate weakly supported primitives.

Eficiency and Compactness. Beyond synthesis quality, DUET4D demonstrates a significant advantage in both rendering eficiency and Gaussian compactness. As detailed in Table 1, our method achieves up to a 49× acceleration in rendering speed compared to state-of-the-art baselines, remarkably utilizing only about 20% of their Gaussian primitive count on Neural 3D dataset. The compactness is driven by the proposed Sequential Fidelity Resampling strategy. Rather than blindly proliferating primitives to fit dynamic regions, this mechanism allocates capacity: it densifies areas that are repeatedly under-reconstructed while aggressively pruning primitives that are persistently inconsistent or redundant across time. This dynamic regularization balanced the trade-of between quality and speed, reducing both the storage overhead and rasterization cost without compromising the fidelity of the reconstructed 4D dynamics.

## 4.3 Ablation Study

Table 2 evaluates the contribution of each proposed component on the Neural 3D Video dataset. To disentangle the efectiveness of the two complementary aspects of DUET4D, we organize the ablation study into representation-side optimization and difusion-guidance variants. We first disable Sequential Anchor Alignment, the framework degrades into one-way from difusion to Gaussian. We then employ vanilla single-view difusion as an intermediate baseline, followed by Anchor-Driven Latent Steering. As for the representation side, we disable Sequential Fidelity Resampling and

Table 2: Ablation study of proposed components on the Neural 3D Video dataset. We progressively evaluate Sequential Fidelity Resampling, Smoothness Control Regularization, Sequential Anchor Alignment, and Anchor-Driven Latent Steering. Vanilla single-view difusion is included as an intermediate baseline to distinguish the efect of the generative prior from that of the proposed guidance mechanisms.
<table><tr><td>Configuration</td><td>|FPS  $\mathbf { N } _ { \mathrm { G S } }$ </td><td>tPSNR↑ PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>80 0.21M</td><td>36.91</td><td>18.30</td><td>0.709</td><td>0.391</td></tr><tr><td>+Fidelity Resampling</td><td>268 0.04M</td><td>36.89</td><td>18.54</td><td>0.713</td><td>0.358</td></tr><tr><td>+Control Regularization</td><td>271 0.03M</td><td>36.88</td><td>18.80</td><td>0.729</td><td>0.329</td></tr><tr><td>+Vanilla Diffusion</td><td>2300.06M</td><td>36.33</td><td>20.04</td><td>0.742</td><td>0.297</td></tr><tr><td>+Anchor Alignment</td><td>2330.06M</td><td>36.89</td><td>20.08</td><td>0.743</td><td>0.305</td></tr><tr><td>+Latent Steering (Full)</td><td>2460.05M</td><td>37.99</td><td>20.28</td><td>0.748</td><td>0.296</td></tr></table>

![](images/5bce3f77331ea59b1a559dee1cebf1867890b22808fb8f68cd8276d7acc4fc2d.jpg)  
Figure 4: Visual ablation. Fidelity Resampling removes floating artifacts, while vanilla difusion restores details but may introduce appearance drift. The full model uses Gaussian-derived anchors and confidence-aware latent steering to recover sharper and more consistent results.

Smoothness Control Regularization to evaluate their efects on reconstruction quality, representation compactness, and rendering eficiency.

Efectiveness of Anchor Alignment. Compared with vanilla single-view difusion, Sequential Anchor Alignment improves temporal PSNR and rendering quality. This clear temporal improvement verifies that Gaussian-derived correspondences provide efective constraints across neighboring views and frames, allowing independently restored results to share consistent appearance corrections. By aligning restoration residuals rather than directly warping complete RGB predictions, the module reduces frame-wise restoration drift without overriding the geometry and view-dependent content encoded by the current representation. The aligned residuals therefore establish a more reliable temporal condition for the subsequent latent guidance stage.

Efectiveness of Fidelity Resampling. Sequential Fidelity Resampling substantially increases representation compactness and rendering eficiency. The results indicate that confidence-gated multi-view evidence can distinguish repeatedly supported reconstruction deficits from primitives that remain inconsistent with the available observations. As a result, the method allocates new capacity only to reliably under-reconstructed regions while removing weakly supported or redundant Gaussians.

![](images/0b2f7d91a19f3b532d8119a53953a800d733af13abc1223922e389bd3aaa4897.jpg)  
Figure 5: Temporal consistency on static regions. We compare vanilla single-view difusion with the full method on fixed-view sequences. The left column shows the diference in long-term drift from a reference frame, while the right column shows the diference in short-term flicker between adjacent frames. Red indicates lower error for DUET4D, and blue indicates the opposite.

Efectiveness of Smoothness Control. Adding Smoothness Control Regularization further increases rendering quality. The Gaussian count is further reduced from 0.04M to 0.03M, and the rendering speed remains nearly unchanged. This result suggests that the spatial and temporal smoothness constraints prevent the deformation field from fitting local noise. They improve reconstruction quality while preserving a compact and eficient representation.

Efectiveness of Latent Steering. Anchor-Driven Latent Steering yields the best overall performance. The gains validate the use of aligned evidence to steer the difusion prediction in latent space rather than directly blending restored RGB images. The guidance improves temporal coherence while preserving local details, leading to more accurate and stable pseudo-supervision.

Visual Ablation. Figure 4 compares the main variants in weakly observed regions. Fidelity Resampling removes weakly supported Gaussians and improves geometry, but cannot recover missing appearance. Vanilla single-view diffusion restores sharper textures, but its independent predictions cause color drift and incorrect details. Our full model uses Gaussian-derived anchors to align restoration across views and time. Confidence-aware latent steering further limits corrections to regions with reliable correspondence.

Figure 5 evaluates temporal stability on fixed-view sequences. We measure long-term drift from a reference frame and short-term flicker between adjacent frames in static regions. The full model reduces both errors, with clear improvements near the cabinet and window. These results confirm that Gaussian-derived correspondences provide efective temporal constraints, while confidence-aware steering suppresses unsupported corrections. Together, these components improve both spatial fidelity and temporal consistency, providing cleaner and more stable pseudo-supervision for 4D reconstruction.

Table 3: Hyperparameter sensitivity on the Neural 3D Video dataset. For each group, only the indicated parameter is changed while all other settings remain fixed.
<table><tr><td></td><td>Setting</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>tPSNR↑</td></tr><tr><td>ρ</td><td>0.25 0.50 1.00</td><td>20.21 20.28 20.06</td><td>0.744 0.748 0.739</td><td>0.299 0.296 0.325</td><td>37.82 37.99</td></tr><tr><td>Tm</td><td>0.001 0.002 0.005 0.010</td><td>20.18 20.28 20.22 20.12</td><td>0.742 0.748 0.744 0.737</td><td>0.300 0.296 0.298 0.302</td><td>38.32 37.79 37.99 37.87 37.61</td></tr></table>

Sensitive Analysis. Table 3 evaluates sensitivity of DUET4D to the guidance strength and the confidence threshold. Moderate parameter choices provide a better balance between temporal consistency and spatial fidelity. Excessive latent guidance may over-constrain the restoration process and weaken local appearance details. Similarly, a loose confidence threshold may introduce unreliable pseudosupervision, while a strict threshold may discard useful correspondence evidence. The performance remains stable within a reasonable range, demonstrating the robustness of DUET4D and supporting the system design.

## 4.4 Limitations

DUET4D relies on the current Gaussian representation to construct spatiotemporal correspondences. When the initial geometry is severely corrupted, or when an object undergoes extremely fast motion and long-term occlusion, the available Gaussian evidence may be insuficient to produce reliable anchors. In such regions, the confidence mechanism conservatively falls back to independent single-view restoration, which limits error propagation but cannot fully recover content that is never observed. Future work may explore long-range correspondence aggregation and more eficient on-demand pseudo-view scheduling.

## 5 Conclusion

We presented DUET4D, a self-corrective framework for sparse-view 4D Gaussian reconstruction. This design is motivated by how to mitigate the spatiotemporal inconsistency inherent in the generative prior to improve the reconstruction quality. By using the evolving Gaussian representation to guide temporally coherent restoration with a frozen singleview difusion model, and selectively converting confidencefiltered discrepancies into Gaussian densification and pruning policies, DUET4D enables the generative prior and explicit representation to progressively refine each other. Extensive experiments on three widely-used datasets demonstrate that our method improves novel-view rendering quality while maintaining high rendering eficiency and a compact Gaussian representation.

## References

Bai, J.; Xia, M.; Fu, X.; Wang, X.; Mu, L.; Cao, J.; Liu, Z.; Hu, H.; Bai, X.; Wan, P.; et al. 2025. Recammaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14834–14844.

Broxton, M.; Flynn, J.; Overbeck, R.; Erickson, D.; Hedman, P.; Duvall, M.; Dourgarian, J.; Busch, J.; Whalen, M.; and Debevec, P. 2020. Immersive light field video with a layered mesh representation. ACM Transactions on Graphics (TOG), 39(4): 86–1.

Cai, H.; Feng, W.; Feng, X.; Wang, Y.; and Zhang, J. 2022. Neural surface reconstruction of dynamic scenes with monocular rgb-d camera. Advances in Neural Information Processing Systems, 35: 967–981.

Cao, A.; and Johnson, J. 2023. Hexplane: A fast representation for dynamic scenes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 130–141.

Du, Y.; Zhang, Y.; Yu, H.-X.; Tenenbaum, J. B.; and Wu, J. 2021. Neural radiance flow for 4d view synthesis and video processing. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 14304–14314. IEEE Computer Society.

Duan, Y.; Wei, F.; Dai, Q.; He, Y.; Chen, W.; and Chen, B. 2024. 4d-rotor gaussian splatting: towards eficient novel view synthesis for dynamic scenes. In ACM SIGGRAPH 2024 Conference Papers, 1–11.

Fang, J.; Yi, T.; Wang, X.; Xie, L.; Zhang, X.; Liu, W.; Nießner, M.; and Tian, Q. 2022. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia 2022 Conference Papers, 1–9.

Fridovich-Keil, S.; Meanti, G.; Warburg, F. R.; Recht, B.; and Kanazawa, A. 2023. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12479–12488.

He, H.; Xu, Y.; Guo, Y.; Wetzstein, G.; Dai, B.; Li, H.; and Yang, C. 2024. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101.

Ho, J.; and Salimans, T. 2022. Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598.

Hu, S.; Hu, T.; and Liu, Z. 2024. Gauhuman: Articulated gaussian splatting from monocular human videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 20418–20431.

Jin, Y.; Peng, S.; Wang, X.; Xie, T.; Xu, Z.; Yang, Y.; Shen, Y.; Bao, H.; and Zhou, X. 2025. Difuman4d: 4d consistent human view synthesis from sparse-view videos with spatio-temporal difusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 11047–11057.

Kerbl, B.; Kopanas, G.; Leimkühler, T.; Drettakis, G.; et al. 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139–1.

Kratimenos, A.; Lei, J.; and Daniilidis, K. 2024. Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting. In European Conference on Computer Vision, 252–269. Springer.

Lee, J.; Won, C.; Jung, H.; Bae, I.; and Jeon, H.-G. 2024. Fully explicit dynamic gaussian splatting. Advances in Neural Information Processing Systems, 37: 5384–5409.

Lei, J.; Weng, Y.; Harley, A. W.; Guibas, L.; and Daniilidis, K. 2025. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scafolds. In Proceedings of the Computer Vision and Pattern Recognition Conference, 6165–6177.

Li, T.; Slavcheva, M.; Zollhoefer, M.; Green, S.; Lassner, C.; Kim, C.; Schmidt, T.; Lovegrove, S.; Goesele, M.; Newcombe, R.; et al. 2022. Neural 3d video synthesis from multiview video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5521–5531.

Li, Z.; Chen, Z.; Li, Z.; and Xu, Y. 2024. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8508–8520.

Lin, Y.; Dai, Z.; Zhu, S.; and Yao, Y. 2024. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21136–21145.

Liu, R.; Wu, R.; Van Hoorick, B.; Tokmakov, P.; Zakharov, S.; and Vondrick, C. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings ofthe IEEE/CVF international conference on computer vision, 9298–9309.

Liu, Y.; Lin, C.; Zeng, Z.; Long, X.; Liu, L.; Komura, T.; and Wang, W. 2024. Syncdreamer: Generating multiviewconsistent images from a single-view image. In International conference on learning representations, volume 2024, 27676–27697.

Long, X.; Guo, Y.-C.; Lin, C.; Liu, Y.; Dou, Z.; Liu, L.; Ma, Y.; Zhang, S.-H.; Habermann, M.; Theobalt, C.; et al. 2024. Wonder3d: Single image to 3d using cross-domain difusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9970–9980.

Luiten, J.; Kopanas, G.; Leibe, B.; and Ramanan, D. 2024. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV), 800–809. IEEE.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65(1): 99–106.

Pan, W.; Zhang, X.; Zhang, Z.; Ye, Z.; Wang, N.; Liu, H.; and Zhang, G. 2026. SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 33237–33247.

Park, K.; Sinha, U.; Barron, J. T.; Bouaziz, S.; Goldman, D. B.; Seitz, S. M.; and Martin-Brualla, R. 2021a. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, 5865–5874.

Park, K.; Sinha, U.; Hedman, P.; Barron, J. T.; Bouaziz, S.; Goldman, D. B.; Martin-Brualla, R.; and Seitz, S. M. 2021b. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228.

Peng, S.; Dong, J.; Wang, Q.; Zhang, S.; Shuai, Q.; Zhou, X.; and Bao, H. 2021a. Animatable neural radiance fields for modeling dynamic human bodies. In Proceedings of the IEEE/CVF international conference on computer vision, 14314–14323.

Peng, S.; Zhang, Y.; Xu, Y.; Wang, Q.; Shuai, Q.; Bao, H.; and Zhou, X. 2021b. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 9054–9063.

Pradeep, V.; Rhemann, C.; Izadi, S.; Zach, C.; Bleyer, M.; and Bathiche, S. 2013. MonoFusion: Real-time 3D reconstruction of small scenes with a single web camera. In 2013 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), 83–88. IEEE.

Pumarola, A.; Corona, E.; Pons-Moll, G.; and Moreno-Noguer, F. 2021. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 10318–10327.

Shi, C.; Yang, C.; Hu, X.; Chen, M.; Pan, W.; Yang, Y.; Ding, J.; Yu, Z.; and Yu, J. 2026. Sparse4dgs: 4d gaussian splatting for sparse-frame dynamic scene reconstruction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 8933–8941.

Shi, Y.; Wang, P.; Ye, J.; Mai, L.; Li, K.; and Yang, X. 2024. Mvdream: Multi-view difusion for 3d generation. In International conference on learning representations, volume 2024, 39838–39859.

Song, C.; Yang, Y.; Zhao, T.; Li, R.; and Zhang, C. 2025. Worldforge: Unlocking emergent 3d/4d generation in video difusion model via training-free guidance. arXiv preprint arXiv:2509.15130.

Wang, Q.; Ye, V.; Gao, H.; Zeng, W.; Austin, J.; Li, Z.; and Kanazawa, A. 2025. Shape of motion: 4d reconstruction from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 9660–9672.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.

Wang, Z.; Yuan, Z.; Wang, X.; Li, Y.; Chen, T.; Xia, M.; Luo, P.; and Shan, Y. 2024. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, 1–11.

Weng, C.-Y.; Curless, B.; Srinivasan, P. P.; Barron, J. T.; and Kemelmacher-Shlizerman, I. 2022. Humannerf: Freeviewpoint rendering of moving people from monocular video. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, 16210–16220.

Wu, G.; Yi, T.; Fang, J.; Xie, L.; Zhang, X.; Wei, W.; Liu, W.; Tian, Q.; and Wang, X. 2024. 4d gaussian splatting

for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 20310–20320.

Wu, R.; Gao, R.; Poole, B.; Trevithick, A.; Zheng, C.; Barron, J. T.; and Holynski, A. 2025. Cat4d: Create anything in 4d with multi-view video difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26057–26068.

Xu, D.; Nie, W.; Liu, C.; Liu, S.; Kautz, J.; Wang, Z.; and Vahdat, A. 2024a. Camco: Camera-controllable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509.

Xu, J.; Fan, Z.; Yang, J.; and Xie, J. 2024b. Grid4d: 4d decomposed hash encoding for high-fidelity dynamic gaussian splatting. Advances in Neural Information Processing Systems, 37: 123787–123811.

Yang, Z.; Gao, X.; Zhou, W.; Jiao, S.; Zhang, Y.; and Jin, X. 2024a. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 20331–20341.

Yang, Z.; Yang, H.; Pan, Z.; and Zhang, L. 2024b. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In International Conference on Learning Representations, volume 2024, 9142–9159.

Yoon, J. S.; Kim, K.; Gallo, O.; Park, H. S.; and Kautz, J. 2020. Novel view synthesis of dynamic scenes with globally coherent depths from a monocular camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5336–5345.

Yu, W.; Xing, J.; Yuan, L.; Hu, W.; Li, X.; Huang, Z.; Gao, X.; Wong, T.-T.; Shan, Y.; and Tian, Y. 2024. Viewcrafter: Taming video difusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586–595.

Zhou, J.; Yang, Z.; Han, L.; Zhang, W.; Shi, K.; Xu, S.; and Liu, Y.-S. 2026. 4C4D: 4 Camera 4D Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11829–11839.