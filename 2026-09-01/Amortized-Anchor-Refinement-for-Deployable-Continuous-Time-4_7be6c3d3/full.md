# Amortized Anchor Refinement for Deployable Continuous-Time 4D Gaussian Reconstruction

Jingong Chen<sup>1</sup> Qingwen Zhang<sup>2</sup> Sanghyeon Jun<sup>1</sup> Chulwoo Pack<sup>1</sup> Kyle Gao<sup>3</sup> Kwanghee Won<sup>1</sup>

<sup>1</sup>Department of Electrical Engineering & Computer Science, South Dakota State University, Brookings, SD, USA

<sup>2</sup>Department of Robotics, Perception and Learning, KTH Royal Institute of Technology, Stockholm, Sweden

<sup>3</sup>Department of Built Environment, Aalto University, Espoo, Finland

{jingong.chen, Sanghyeon.Jun}@jacks.sdstate.edu

{Chulwoo.Pack, Kwanghee.Won}@sdstate.edu qingwen@kth.se kyle.gao@aalto.fi

## Abstract

Continuous-time 4D reconstruction remains impractical on standalone XR headsets. Per-scene optimization demands deployment-infeasible compute, and lower budgets cause collapse rather than degrade gradually. Feed-forward prediction is fast, but struggle to recover scene-specific detail. We present Amortized Anchor Refinement, which uses a frozen backbone to predict an initial Gaussian representation and a short optimization to specialize it under a fixed compute budget, with a capacity floor preserving representational density. A training-free stage then applies a persistent-homology constraint to prune unstable Gaussians while preserving topologically persistent structures, and streams the resulting trajectories directly as sceneflow. On the Stage-Capture benchmark, Amortized Anchor Refinement achieves 24.31±2.22 dB, while our deployment experiments demonstrate reconstruction within the target budget on a single consumer GPU and playback on a standalone XR headset.

Code: https://github.com/Jingong-Chen/Amortized-Anchor-Refinement

## 1. Introduction

Reconstructing a dynamic scene so that a viewer may observe it from arbitrary viewpoints at arbitrary times is a long-standing goal of immersive media [2], enabling applications such as virtual reality, telepresence, and freeviewpoint replay. Recent advances have brought this goal within reach. Neural radiance fields [22] established photorealistic novel-view synthesis, while subsequent work improved scalability to unbounded scenes and deformable content [1, 9, 24, 26–28]. More recently, 3D Gaussian

![](images/a437ad00f0adf73e7aba7ff1044d63ffdd456686886304b8466d644c2581b7a8.jpg)  
Figure 1. Amortized Anchor Refinement uses feedforward anchors as enabler. Under the deployment budget of standalone XR headsets, optimizing a 4D Gaussian field from scratch collapses, whereas the same budget becomes sufficient for scene-specific refinement when initialized by a single feed-forward pass. Right: visual comparison of rendering at a held timestamp.

Splatting [14] replaced implicit fields with explicit primitives, enabling real-time rendering and motivating a broad family of continuous-time dynamic Gaussian representations.

Existing methods model dynamics either through deformation fields or explicit Gaussian trajectories [6, 12, 15, 17, 21, 38, 40, 42, 43]. Given sufficient representational capacity and scene-specific optimization, these methods reconstruct complex real-world dynamics with high quality while answering queries at arbitrary times in real time. However, this quality scales with the number of per-scene optimization iterations, whose cost places these methods out of reach of standalone XR headsets.

These headsets provide only a few gigabytes of memory, limited compute, and a wireless link for streaming. Continuous-time reconstruction further tightens these constraints because the entire temporal field must remain resident to support arbitrary-time queries, making frame-byframe streaming infeasible. Existing compression methods reduce the size of an already-optimized field [8, 16, 20], while streaming systems transmit frame-to-frame residuals [11, 32, 36], but neither asks how the field itself should be learned when the deployment budget of standalone XR headsets is imposed from the outset.

A direct solution is to reduce the optimization budget by using fewer Gaussians and fewer iterations. However, this fails because photometric gradients provide no consistent update direction for unanchored Gaussians: the densifyand-prune schedule removes them faster than it can establish valid geometry, causing reconstruction quality to collapse to 16.4 dB rather than degrade gracefully to a usable solution. Thus, under a constrained budget, optimization is dominated by geometry discovery rather than scene refinement.

Feed-forward prediction avoids this search entirely. Recent amortized methods reconstruct static [3, 4, 33, 34, 37] and dynamic scenes [18, 30, 41] within seconds, but their accuracy remains constrained by the learned geometric prior and does not fully capture scene-specific detail. These limitations are complementary: feed-forward prediction provides an initial geometric solution that budgeted optimization cannot reliably discover, while subsequent optimization recovers scene-specific details beyond the feedforward prior.

Amortized Anchor Refinement therefore lets a frozen feed-forward backbone predict the geometric anchor and spends the entire optimization budget on refining it to the target scene. A capacityfloor keeps the field dense throughout refinement, and a training-free persistencefloor protects thin structures when the field is pruned for transmission. The fitted trajectories are streamed directly as scene flow, preserving arbitrary-time queries without per-frame optimization or residual coding. On a single consumer GPU, our cascade reaches 24.22 dB at a 2.12 GB fitting peak with 5.4× fewer Gaussians than a heavyweight optimizer. Together, these results demonstrate that continuous-time 4D reconstruction can be learned, compressed, and played back within the resources of a standalone XR headset.

Our contributions are fourfold:

• We identify budget-constrained continuous-time 4D reconstruction as a distinct problem rather than a scaleddown one, and characterize its two fundamental failure modes: a ceiling on what feed-forward prediction reaches, and the collapse of optimization that starts without an anchor.

• We propose Amortized Anchor Refinement, which combines a grounded feed-forward anchor, a mirrorsymmetry prior, and a capacity-floored refinement to enable stable consumer-budget optimization without taskspecific pretraining.

• We introduce Persistent Homology Pruning and Compression, the first Gaussian pruning method based on persistent homology, which preserves topologically important structures during compression while transmitting quantized Gaussian trajectories as a compact continuoustime scene-flow representation.

• We validate our approach on the Stage-Capture benchmark and show through controlled ablations that the improvements arise from the proposed design rather than from increased model capacity or scale.

## 2. Related Work

## 2.1. Reconstructing a Dynamic Gaussian Field

3D Gaussian Splatting [14] enabled real-time novel-view synthesis, and two paradigms have since extended it to dynamic scenes. The first deforms a canonical Gaussian representation through a learned motion field, as in Deformable-3D-Gaussians [42] and 4D-GS [40]. The second assigns each Gaussian an explicit trajectory and lifetime, as in Spacetime Gaussians [17], 4D-Rotor Gaussians [6], Ex4DGS [15], and RetimeGS [38]. We adopt the latter because it supports continuous-time queries without a deformation network, and because each Gaussian trajectory directly represents the scene flow that can later be streamed without per-frame estimation or optimization.

Despite their different motion parameterizations, existing optimization-based methods share the same optimization strategy. Each scene is reconstructed independently through lengthy optimization, typically requiring tens of minutes, hundreds of thousands of Gaussians, and workstation-class GPUs. The deployment budget of standalone XR headsets is therefore absent from both their optimization objectives and their evaluation. As shown in Sec. 4, these optimization schedules collapse once that deployment budget is imposed.

An alternative direction amortizes reconstruction through feed-forward prediction. Pose-free geometry backbones [34, 37] estimate geometry without calibrated cameras, while large reconstruction models [18, 30] directly predict renderable dynamic Gaussian fields. These methods inherit the feed-forward parameterization of static Gaussian reconstruction, where one Gaussian is predicted per input pixel [3, 4, 33], limiting reconstruction to the geometric prior learned during pretraining. Recent amortized dynamic methods [41] retain the same formulation while requiring large-scale task-specific pretraining. Instead, we use a frozen geometry backbone [35] only to predict an initial geometric anchor. Scene-specific optimization then refines this anchor, avoiding both task-specific pretraining and the quality ceiling of feed-forward prediction.

## 2.2. Compacting and Delivering a Fitted Field

Memory requirement is traditionally controlled by the densify-and-prune schedule introduced in [14], while postprocessing methods reduce the size of an already-optimized field through pruning, quantization, and anchor-based parameter sharing [8, 20]. These approaches incur the full optimization cost before compression begins. In contrast, our refinement directly optimizes a compact representation by maintaining an anchored initialization above an explicit capacity floor. Peak memory is then dominated by the frozen backbone, whose activations are CPU-offloaded so that the entire pipeline fits within a 4 GB deployment budget.

Existing streaming systems focus on transmitting $\mathrm { d y } .$ namic Gaussian fields after optimization. 3DGStream caches per-frame transforms [32], QUEEN encodes interframe attribute residuals [11], and ${ \mathrm { V } } ^ { 3 }$ packs Gaussian attributes into videos for hardware decoding [36]. These methods compress temporal changes, while the underlying Gaussian representation is still determined by pruning strategies that prioritize visible mass and often discard thin structures. Persistent homology has recently been introduced as a training regularizer for Gaussian splatting [31], but to our knowledge has not been used as a pruning criterion. Our streaming stage instead applies an explicit persistence floor, complementing the capacity floor used during refinement, and directly transmits the optimized Gaussian trajectories without per-frame optimization or residual coding.

## 3. Method

To make continuous-time 4D reconstruction deployable under the memory budget of a standalone XR headset, we propose the Amortized Anchor Refinement framework, as illustrated in Fig. 2.

## 3.1. Preliminaries and Anchor Backbone

Continuous-time 4D reconstruction aims to render a $\mathrm { d y }$ namic scene at any query viewpoint and frame in real time. Given a set of calibrated multi-view video streams $I = \{ I _ { c , t } \}$ , with $\pmb { I _ { c , t } } \in \mathbb { R } ^ { H \times W \times 3 }$ , where $c \in \{ 1 , \ldots , N _ { c } \}$ indexes the calibrated views and $t \in [ 0 , 1 ]$ ] denotes the normalized frame, our goal is a 4D Gaussian field that renders cleanly at held-out viewpoints and frames. Following [34], we adopt a frozen pose-free VGGT-Ω as the geometry backbone. Given the input views, the backbone encodes each view c into a per-pixel depth field $\pmb { D } _ { c } \in \mathbb { R } ^ { H \times W }$ and a perpixel feature tensor $\pmb { F _ { c } } \in \mathbb { R } ^ { H \times W \times C _ { f } }$ in a single forward pass, without requiring camera poses,

$$
( D _ { c } , { \pmb F } _ { c } ) = \Phi ( { \pmb I } _ { c } ) , \quad c \in \{ 1 , \ldots , N _ { c } \} ,\tag{1}
$$

where $\Phi ( \cdot )$ denotes the frozen backbone network. The backbone is never fine-tuned and requires no task-specific pretraining. Its cost is a single forward pass. This forward pass dominates the memory footprint of the pipeline, so we CPU-offload the backbone layer by layer at FP32, bounding peak memory by a single layer rather than by the full network.

We represent the dynamic scene as N 4D Gaussians, each with a canonical center $\pmb { \mu _ { i } }$ , a covariance $\Sigma _ { i }$ , an opacity $o _ { i } ,$ a view-dependent color $c _ { i } ,$ a temporal center $\tau _ { i } ,$ , and a window width $s _ { i } .$ . To express motion without a separate deformation residual, we adopt the short-tailed temporalopacity design of RetimeGS [38]. Each Gaussian follows a low-order trajectory whose center moves in time as

$$
\pmb { \mu } _ { i } ( t ) = \pmb { \mu } _ { i } + \pmb { v } _ { i } ^ { 1 } \delta _ { i } + \pmb { v } _ { i } ^ { 2 } \delta _ { i } ^ { 2 } + \pmb { v } _ { i } ^ { 3 } \delta _ { i } ^ { 3 } , \quad \delta _ { i } = t - \tau _ { i } ,\tag{2}
$$

where $\pmb { v } _ { i } ^ { 1 } , \pmb { v } _ { i } ^ { 2 } , \pmb { v } _ { i } ^ { 3 } \in \mathbb { R } ^ { 3 }$ are the learned velocity, acceleration, and jerk terms. Each Gaussian is visible only within a compact temporal window,

$$
o _ { i } ( t ) = o _ { i } \exp \biggl ( - \frac { ( t - \tau _ { i } ) ^ { 2 } } { 2 s _ { i } ^ { 2 } } \biggr ) ,\tag{3}
$$

so that it contributes only near its own frame $\tau _ { i } .$ . At a query time $t ,$ the field is rasterized to view c by the standard splatting composite

$$
C ( \pmb { p } ) = \sum _ { i \in \mathcal { N } } \pmb { c } _ { i } \alpha _ { i } ( t ) \prod _ { j < i } \bigl ( 1 - \alpha _ { j } ( t ) \bigr ) ,\tag{4}
$$

where $\pmb { p }$ is a pixel, $\mathcal { N }$ the depth-ordered Gaussians overlapping it, and $\alpha _ { i } ( t )$ combines the projected 2D footprint of Gaussian i with its temporal opacity in Eq. 3.

## 3.2. Grounded Feed-forward Anchor

The first stage, named Grounded Feed-forward Anchor (GFA), predicts an initialization for the per-scene optimizer. Regressing Gaussian centers as free 3D coordinates fails to optimize. Free coordinates have no geometric correspondence to the input observations, so the photometric gradient provides no consistent direction. Instead, we ground every predicted Gaussian on its input-view ray. The frozen backbone already localizes the scene to within a viewing ray, making ray grounding sufficient to remove this ambiguity while preserving local geometric flexibility. As illustrated in the GFA block of Fig. 2, given the backbone depth $D _ { c } ( \pmb { p } )$ , camera center $^ { o _ { c } , }$ and ray direction $\pmb { d } _ { c } ( \pmb { p } )$ , the anchored Gaussian center at pixel $\pmb { p }$ in view c is

$$
\pmb { \mu } = \pmb { o } _ { c } + \pmb { D } _ { c } ( \pmb { p } ) \pmb { d } _ { c } ( \pmb { p } ) + \Delta \pmb { \mu } ,\tag{5}
$$

where $\Delta \mu$ is a learnable residual offset from the ray. A lightweight prediction head $g _ { \boldsymbol { \theta } }$ takes the backbone feature ${ \cal F } _ { c } ( p )$ and predicts the remaining Gaussian parameters,

$$
\begin{array} { r } { \big ( \Delta \pmb { \mu } , \pmb { \Sigma } , o , \pmb { c } , \tau , s , \pmb { v } ^ { 1 : 3 } \big ) = g _ { \boldsymbol { \theta } } \big ( \pmb { F } _ { c } ( \pmb { p } ) \big ) , } \end{array}\tag{6}
$$

where the head emits K Gaussians along each ray at increasing depths, with $K { = } 2$ by default, forming a thin volumetric anchor rather than a single surface.

![](images/8512417ec5583c9a975e75dd836956bca82fe811e41e0d1e2e1046a55a258140.jpg)  
Figure 2. Pipeline of Amortized Anchor Refinement. A grounded feed-forward anchor initializes an optimization-ready Gaussian field, which is refined under capacity constraints to recover high-quality dynamic geometry. A persistent homology pruning and compression (PHPC) stage compresses the field while preserving perceptually important structures.

Mirror-symmetry back-side anchor. The input views observe only the front surface of the subject, leaving the raygrounded anchor of Eq. 5 hollow on the occluded side. This missing geometry is largely corrected when dense temporal supervision is available. Under sparse-time supervision, however, the unconstrained back side allows per-Gaussian trajectories to overfit the observed frames and generalize poorly to held frames. We therefore initialize the occluded region by reflecting the anchor Gaussians across the frontal symmetry plane, drawn as the Mirror step in Fig. 2, producing a coarse yet closed geometry that regularizes optimization. Let µ¯ denote the centroid of the subject Gaussians and n the unit normal of the frontal symmetry plane. The mirrored center is

$$
\pmb { \mu } ^ { \mathrm { m i r } } = \pmb { \mu } - 2 \big [ ( \pmb { \mu } - \pmb { \bar { \mu } } ) \cdot \pmb { n } \big ] \pmb { n } ,\tag{7}
$$

which requires neither additional parameters nor extra supervision. This mirror-symmetry prior has little effect for long, densely supervised sequences, where observations eventually constrain the back side, but becomes critical for short clips. The front rays and their mirrored counterparts together form the starting field, the anchored initialization that the refinement stage takes over.

## 3.3. Capacity Constrained Refinement

The second stage refines the anchored field by fitting it directly to the raw multi-view capture. We name this module Capacity Constrained Refinement (CCR). Under sparse real supervision, the standard densify-and-prune schedule rapidly collapses the representation to a few hundred Gaussians, limiting geometric detail and leaving isolated floaters. We find that this behavior is caused primarily by aggressive opacity pruning, and replace it with an opacity-based capacity constraint. As illustrated in the CCR block of Fig. 2, after each pruning step the field is constrained to contain at least $N _ { \mathrm { m i n } }$ Gaussians by retaining the $N _ { \mathrm { m i n } }$ highest-opacity primitives,

$$
\begin{array} { r } { \mathcal { K } = \displaystyle \log _ { - } \Big ( \{ o _ { i } \} _ { i = 1 } ^ { N } \Big ) , } \\ { \mathcal { N } _ { \operatorname* { m i n } } } \end{array}\tag{8}
$$

where K denotes the retained index set. To suppress floaters, we remove Gaussians that lie far from the scene medoid or whose scales exceed a preset threshold. Because refinement adjusts a grounded initialization rather than searching for geometry, a short optimization suffices, and $N _ { \mathrm { m i n } }$ directly bounds the memory footprint.

## 3.4. Optimization Objective

Only CCR is optimized on a per-scene basis. GFA consists of a single forward pass through the frozen backbone and requires neither optimization nor a teacher model. The refinement minimizes a photometric objective over the masked multi-view observations,

$$
\mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } + \gamma \mathcal { L } _ { \mathrm { L P I P S } } ,\tag{9}
$$

where $\mathcal { L } _ { 1 }$ enforces pixel-wise reconstruction, L<sub>D-SSIM</sub> structural consistency and $\mathcal { L } _ { \mathrm { L P I P S } }$ perceptual quality, weighted by λ and $\gamma .$ Unless otherwise specified, we use $\lambda = 0 . 2$ and $\gamma = 0 . 1$

## 3.5. Persistent Homology Pruning and Compression

The refined field renders in real time, but its size exceeds a wireless streaming budget. We therefore introduce Persistent Homology Pruning and Compression (PHPC) to prune the field for transmission. The standard criterion ranks Gaussians by visibility mass,

$$
w _ { i } = o _ { i } \bar { \sigma } _ { i } ,\tag{10}
$$

where $\bar { \sigma } _ { i }$ denotes the mean scale of Gaussian i. Mass ranking preserves appearance, but slender structures such as wrists and ankles carry little mass and are discarded first, so the subject topology breaks before texture quality degrades.

As illustrated in the PHPC block of Fig. 2, we prevent these failures using persistent homology derived from Morse theory [23]. At each frame t in a small reference set $\tau ,$ the refined field defines a density that we accumulate on a voxel grid, the density grid on which the diagram below is computed,

$$
\sigma _ { t } ( { \pmb x } ) = \sum _ { i } o _ { i } ( t ) \mathcal { G } \big ( { \pmb x } ; { \pmb \mu } _ { i } ( t ) , { \pmb \Sigma } _ { i } \big ) ,\tag{11}
$$

where x is a grid point, $\mathcal { G }$ the Gaussian kernel carrying the fixed covariance $\Sigma _ { i }$ of Sec. 3.1, and $o _ { i } ( t )$ the temporal opacity of Eq. 3. Its persistence diagram [7] summarizes the connected components of the subject across density thresholds. Thresholding the diagram at persistence ε defines a topology floor: every feature whose persistence exceeds ε must be preserved after pruning. Given a transmission budget of $N _ { s }$ Gaussians, the number the clip package may carry, we solve

$$
\begin{array} { r l } & { \mathcal { K } _ { s } = \underset { N _ { s } } { \mathrm { t o p } } \bigl ( \{ w _ { i } \} \bigr ) } \\ & { \mathrm { s . t . } \quad \mathrm { D g m } _ { \varepsilon } \Bigl ( \sigma _ { t } ^ { \mathcal { K } _ { s } } \Bigr ) = \mathrm { D g m } _ { \varepsilon } ( \sigma _ { t } ) , \qquad t \in \mathcal { T } , } \end{array}\tag{12}
$$

where Dgm denotes the persistence diagram thresholded at $\varepsilon ,$ and $\sigma _ { t } ^ { \hat { K } _ { s } }$ the density of the retained subset $\mathcal { \kappa } _ { s }$ . We enforce this constraint with a prune-verify-restore-reverify procedure. After pruning, we recompute the persistence diagram and restore the highest-ranked discarded Gaussians associated with every missing persistent feature, while removing the lowest-ranked retained Gaussians to maintain the target budget. The restored field is accepted only if it preserves a strictly larger fraction of ε-persistent features; otherwise the pruned field is kept, so the floor is activated only when it is needed. The sender also has to choose the ranking. A ranking is a heuristic for which Gaussians carry the subject, and no single heuristic holds across scenes and budgets. We therefore prune with each candidate ranking, apply the floor to each, and keep the package that best reproduces the field it was pruned from, measured by rendering both and comparing them. This comparison is available at the sender because it references the fitted field rather than ground-truth images, so the choice can be made at deployment time, and the transmitted package is never worse than the best individual criterion by more than the resolution of this measurement.

The temporal representation requires no additional modeling. The trajectories of Eq. 2 already encode the scene flow, so compression reduces to efficient coding. For each Gaussian, this trajectory coding step quantizes the nine coefficients to 10 bits and entropy-codes them together with the FP16/INT8-quantized static attributes. The encoder runs once on the server; the result is a self-contained clip that the decoder on the edge client turns directly into Eq. 4, with no server interaction during rendering.

## 4. Experiments

## 4.1. Experimental Setup

Dataset. We evaluate on the multi-view Stage-Capture dataset released with RetimeGS [38], nine dynamic scenes of human motion and human-object interaction recorded by 32 synchronized calibrated cameras at 1024×750 with perframe foreground masks. Each clip spans 33 frames of a complete action with large inter-frame motion, non-rigid deformation, and severe self-occlusion, the regime in which temporal interpolation is hardest.

Experiment Protocol. We follow the dataset’s official every-other-frame hold-out split: all 32 calibrated cameras and the 17 even-indexed frames are used for training, and the 16 odd-indexed frames are held out. Our primary evaluation measures continuous-time interpolation on the held-out frames with foreground-masked PSNR, SSIM and LPIPS; we also report reconstruction quality on the observed frames. All experiments use a fixed random seed. Residual run-to-run variation due to nondeterministic CUDA atomics in gsplat is within ±0.3 dB, consistent with standard 3DGS implementations.

Implementation. The backbone is a frozen VGGT-Ω [35], never fine-tuned and CPU-offloaded at FP32. Our GFA module predicts $K { = } 2$ grounded Gaussians per input ray. Our CCR optimizes the field for 96k Adam steps at 512×375 with a capacity floor of $N _ { \mathrm { m i n } } { = } 2 { \times } 1 0 ^ { 5 }$ , a temporal window scale of 1.5, and geometric floater filtering, minimizing $0 . 8 \mathcal { L } _ { 1 } + 0 . 2 \mathcal { L } _ { \mathrm { D - S S I M } } + 0 . 1 \mathcal { L } _ { \mathrm { L P I P S } }$ using exponential learning-rate decay over the final 10% of training. Densification runs over the first 70% of the steps, or the first 35% on scenes whose Gaussian count grows without settling. For the nine-frame sequence, the same protocol is used with the capacity floor scaled to $N _ { \mathrm { m i n } } { = } 5 { \times } 1 0 ^ { 4 }$ and the optimization shortened to 24k steps; the mirror-symmetry anchor is enabled only in this setting. All methods are evaluated on a single 24 GB RTX 6000 GPU under this protocol, so quality, memory, runtime and Gaussian count are directly comparable.

## 4.2. Comparison with State of the Art

Amortized Anchor Refinement achieves the best accuracy on every scene in Table 1, outperforming Deformable-3DGS by an average of +5.6 dB and by up to +10.9 dB on walk. The baselines are limited by temporal interpolation: coordinate-based MLPs overfit the observed frames and ghost between them, while flow-supervised fields and spatio-temporal grids recover part of the gap. Our anchored short-tailed trajectory is defined continuously in time and never interpolates unseen frames. Although RetimeGS achieves higher reconstruction quality, Table 2 highlights its substantially higher deployment cost. All competing 4D Gaussian methods require 6–9 GB during optimization, exceeding a 4 GB consumer budget, whereas our method peaks at only 2.12 GB while also delivering the fastest rendering speed. FILM attains the highest held-timestamp accuracy but performs 2D video interpolation rather than reconstructing a dynamic 3D representation, so it cannot synthesize novel viewpoints or arbitrary frames; held-time PSNR alone does not characterize the task.

Table 1. Per-scene comparison. Foreground-masked held-timestamp PSNR (dB). RetimeGS exceeds the deployment budget; FILM<sup>†</sup> interpolates in 2D.
<table><tr><td>Method</td><td>walk</td><td>bear</td><td>doll</td><td>newund.</td><td></td><td>openc. passd.</td><td>. pickupd.</td><td>putcomp.</td><td>stretch</td><td>Mean</td></tr><tr><td>FILM [29]†</td><td>29.88</td><td>29.27</td><td>27.40</td><td>31.46</td><td>34.10</td><td>29.34</td><td>29.86</td><td>32.17</td><td>27.59</td><td>30.12</td></tr><tr><td>Deformable-3DGS [42]</td><td>13.30</td><td>17.87</td><td>18.72</td><td>22.28</td><td>24.07</td><td>17.16</td><td>16.66</td><td>23.17</td><td>15.33</td><td>18.73</td></tr><tr><td>GaussianFlow [10]</td><td>21.27</td><td>18.78</td><td>17.73</td><td>23.17</td><td>20.21</td><td>21.72</td><td>18.53</td><td>21.81</td><td>17.30</td><td>20.06</td></tr><tr><td>4D-GS [40]</td><td>18.28</td><td>19.06</td><td>18.87</td><td>25.49</td><td>25.30</td><td>20.35</td><td>20.12</td><td>24.90</td><td>18.44</td><td>21.20</td></tr><tr><td>RetimeGS [38]</td><td>24.57</td><td>22.09</td><td>26.02</td><td>29.56</td><td>29.67</td><td>26.44</td><td>25.38</td><td>29.81</td><td>24.96</td><td>26.50</td></tr><tr><td>Ours, short clip (9 fr.)</td><td>22.91</td><td>20.41</td><td>18.84</td><td>23.19</td><td>23.53</td><td>14.66</td><td>24.93</td><td>19.34</td><td>25.44</td><td>21.47</td></tr><tr><td>Ours, full action (33 fr.)</td><td>24.22</td><td>22.46</td><td>22.96</td><td>26.60</td><td>27.96</td><td>22.87</td><td>23.39</td><td>27.14</td><td>21.19</td><td>24.31</td></tr></table>

<sup>†</sup> FILM performs 2D frame interpolation independently for each training camera by synthesizing only the temporal midpoint between two neighboring supervised frames. Because it does not reconstruct a 4D scene representation, it cannot render novel viewpoints or arbitrary frames and is therefore not a deployable 4D method.

Table 2. Deployment efficiency. #G $( 1 0 ^ { 4 } )$ , peak optimization VRAM (GB), fitting time (min) and held-timestamp PSNR (dB). #G and PSNR are nine-scene means; VRAM is measured on walk.
<table><tr><td>Method</td><td>#G↓</td><td>VRAM↓</td><td>Fit↓</td><td>PSNR↑</td></tr><tr><td>Deformable-3DGS [42]</td><td>15.9</td><td>6.4</td><td>71</td><td>18.73</td></tr><tr><td>GaussianFlow [10]</td><td>128</td><td>7.1</td><td>36</td><td>20.06</td></tr><tr><td>4D-GS [40]</td><td>20.1</td><td>6.0</td><td>58</td><td>21.20</td></tr><tr><td>RetimeGS [38]</td><td>128</td><td>9.3</td><td>136</td><td>26.50</td></tr><tr><td>Ours (9 fr.)</td><td>9.1</td><td>1.0</td><td>9.1</td><td>21.47</td></tr><tr><td>Ours (33 fr.)</td><td>23.9</td><td>2.12</td><td>43</td><td>24.31</td></tr></table>

All reported PSNR values are foreground-masked; the static background follows the standard 3DGS formulation and is composited separately.

## 4.3. Generalization Across Scenes

Across the nine scenes of Table 3 our method reaches a mean held-timestamp PSNR of 24.31±2.22 dB. Since all camera views are supervised, we evaluate temporal interpolation on two disjoint camera splits, which agree to 24.31 against 24.11 dB, so the reported performance is insensitive to the split. Quality degrades gradually with scene complexity: the hardest scenes are those with large articulated or heavily self-occluding motion, bear and stretch. No scene needs its own hyperparameters.

## 4.4. Ablation Study

Anchor and capacity floor. Table 4 adds one module at a time under an identical recipe. From-scratch optimization reaches 16.4 dB on walk: the photometric gradient gives unanchored Gaussians no consistent direction, so the densify-and-prune schedule strips the field to 40 Gaussians and the run ends with almost nothing to render. The same recipe on opencase ends at 17.49 dB with 59 Gaussians, so the collapse is a property of the regime rather than of one scene. Anchoring the field removes it, lifting quality to 20.48 dB, but the anchored field is still pruned down to a few hundred Gaussians, because the schedule that caused the collapse is unchanged. The capacity floor addresses precisely that: holding at least $N _ { \mathrm { m i n } }$ Gaussians throughout raises quality to 24.22 dB at $2 3 . 9 \times 1 0 ^ { 4 }$ Gaussians. Neither module is sufficient alone: the anchor supplies the geometry the optimizer cannot find, and the floor keeps enough capacity for it to be refined.

Mirror-symmetry back side. The input views observe only the front surface, so the anchor is hollow behind the subject. Table 4 evaluates the mirrored completion in the nine-frame regime, where temporal supervision is sparse: removing it drops 24.76 to 16.00 dB. The optimization curve explains the size of the gap. Without the back side, quality peaks at 17.41 dB after a few thousand steps and then decays monotonically: the unconstrained rear Gaussians move into the observed frames and the trajectories overfit them, so the final number is the end of a decay rather than a plateau. With the mirrored anchor the same run improves throughout. The prior leaves the full-length sequence unchanged, where dense supervision constrains the rear surface on its own, so we enable it only for short clips. Topology floor. The middle pair of Table 5 isolates the floor on a fixed visibility-mass ranking. Above 4× $1 0 ^ { 4 }$ Gaussians nothing ε-persistent dies, so the floor never activates and the two rows are identical. $\mathrm { { A t ~ 2 \times 1 0 ^ { 4 } } }$ the ranking starts severing thin structures: preservation falls to 0.82, and restoring the Gaussians that carry the missing features returns it to 0.88 and raises decoded quality from 19.52 to 20.30 dB. The floor is insurance rather than a booster, inert when the ranking is already safe and recovering what a ranking destroys once the budget tightens, as Figure 4 shows at the sleeve boundary.

Table 3. Per-scene results. Foreground-masked PSNR (dB). Interp $\cdot A , B \colon$ held timestamps on two disjoint camera splits; Recon.: observed timestamps.
<table><tr><td rowspan="2">Scene</td><td colspan="3">PSNR↑ (dB)</td><td colspan="2">SSIM↑</td><td colspan="2">LPIPS↓</td><td rowspan="2">#G↓ (104)</td></tr><tr><td>Interp.A</td><td>Recon.</td><td>Interp.B</td><td>Interp.</td><td>Recon.</td><td>Interp.</td><td>Recon.</td></tr><tr><td>walk</td><td>24.22</td><td>26.54</td><td>24.03</td><td>0.982</td><td>0.990</td><td>0.018</td><td>0.010</td><td>22.6</td></tr><tr><td>bear</td><td>22.46</td><td>26.43</td><td>22.93</td><td>0.968</td><td>0.984</td><td>0.033</td><td>0.017</td><td>22.6</td></tr><tr><td>doll</td><td>22.96</td><td>26.36</td><td>22.75</td><td>0.972</td><td>0.984</td><td>0.024</td><td>0.012</td><td>22.6</td></tr><tr><td>newundress</td><td>26.60</td><td>27.62</td><td>26.09</td><td>0.971</td><td>0.982</td><td>0.027</td><td>0.018</td><td>22.5</td></tr><tr><td>opencase</td><td>27.96</td><td>28.30</td><td>27.54</td><td>0.982</td><td>0.986</td><td>0.017</td><td>0.011</td><td>22.4</td></tr><tr><td>passdoll</td><td>22.87</td><td>26.87</td><td>22.94</td><td>0.945</td><td>0.973</td><td>0.055</td><td>0.023</td><td>34.9</td></tr><tr><td>pickupdoll</td><td>23.39</td><td>26.70</td><td>23.41</td><td>0.964</td><td>0.980</td><td>0.042</td><td>0.021</td><td>22.3</td></tr><tr><td>putcomputer</td><td>27.14</td><td>27.89</td><td>26.63</td><td>0.986</td><td>0.990</td><td>0.012</td><td>0.009</td><td>22.6</td></tr><tr><td>stretch</td><td>21.19</td><td>26.30</td><td>20.66</td><td>0.969</td><td>0.985</td><td>0.030</td><td>0.014</td><td>22.6</td></tr><tr><td>Mean</td><td>24.31</td><td>27.00</td><td>24.11</td><td>0.971</td><td>0.984</td><td>0.029</td><td>0.015</td><td>23.9</td></tr><tr><td>Std</td><td>2.22</td><td>0.70</td><td>2.08</td><td>.012</td><td>.005</td><td>.013</td><td>.005</td><td>3.9</td></tr></table>

![](images/2ae971c86c6b008a746face12d81ad57fe7180080f1b68f8e5454e2266d1ffbc.jpg)

Figure 3. Held frames on a held walk camera. No method was supervised on these frames. Foreground composited over a static plate.  
![](images/76cbf21725dd6fe5866a888c2837065d0865697ab0bfd0daa6a5c691cc0f4b8d.jpg)  
Figure 4. The verified topology floor on doll at $2 \times 1 0 ^ { 4 }$ Gaussians, held camera and timestamp. Labels report PSNR at this view. Bottom: zoomed view.

## 4.5. Analysis

Refinement budget and clip length. Quality rises with optimization length but is insensitive to the Gaussian budget once the capacity floor is reached. A nine-frame clip is anchored and refined in 9.1 min and the full sequence in 43 min, with the shorter clip varying more because fewer observations constrain it.

Table 4. Module ablation on walk under an identical recipe within each block. PSNR: foreground-masked held-timestamp PSNR (dB); #G: Gaussians after optimization.
<table><tr><td></td><td>GFA CCR</td><td>PSNR↑ #G</td><td> $( 1 0 ^ { 4 } )$ </td></tr><tr><td>33-frame sequence</td><td></td><td></td><td></td></tr><tr><td>From scratch</td><td></td><td>16.4</td><td>0.004</td></tr><tr><td>+ anchor</td><td>√</td><td>20.48</td><td>0.024</td></tr><tr><td>+ capacity floor</td><td>一</td><td>24.22</td><td>23.9</td></tr><tr><td>From scratch, opencase</td><td></td><td>17.49</td><td>0.006</td></tr><tr><td>9-frame clip</td><td></td><td></td><td></td></tr><tr><td>Anchor without mirror</td><td>√</td><td>√</td><td>16.00</td></tr><tr><td>Anchor with mirror</td><td>L</td><td></td><td>24.76 10.1</td></tr></table>

Table 5. Streaming quality per budget. Nine-scene means. Mbps: coded bitrate at 25 fps; PSNR (dB): after decoding; Topo: fraction of ε-persistent features preserved. The middle pair isolates the topology floor on a fixed ranking.
<table><tr><td> $N _ { s } \ ( 1 0 ^ { 4 } )$  Mbps</td><td>8.0 33.4</td><td>4.0 16.8</td><td>2.0 8.4</td><td>1.0 4.2</td><td>0.5 2.1</td></tr><tr><td>PSNR↑</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LightGaussian [8] RadSplat [25]</td><td>24.00 24.00</td><td>23.12 23.66</td><td>19.88 21.90</td><td>16.83 17.98</td><td>15.27 13.17</td></tr><tr><td>Visibility mass + floor</td><td>24.31 24.31</td><td>23.66 23.62</td><td>19.52 20.30</td><td>16.57 16.79</td><td>14.81 14.85</td></tr><tr><td>Ours</td><td>24.31</td><td>24.28</td><td>22.59</td><td>19.70</td><td>16.71</td></tr><tr><td>Topo↑</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LightGaussian [8]</td><td>1.00</td><td>0.98</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>0.85</td><td>0.70</td><td>0.60</td></tr><tr><td>RadSplat [25]</td><td>1.00</td><td>1.00</td><td>0.95</td><td>0.80</td><td>0.62</td></tr><tr><td>Visibility mass</td><td>1.00</td><td>0.98</td><td>0.82</td><td>0.69</td><td>0.58</td></tr><tr><td>+ floor</td><td>1.00</td><td>0.99</td><td>0.88</td><td>0.72</td><td>0.60</td></tr><tr><td>Ours</td><td>1.00</td><td>1.00</td><td>0.98</td><td>0.90</td><td>0.79</td></tr></table>

Deployment on a 4 GB budget. Peak memory is dominated by the backbone forward pass rather than refinement, whose peak is only 0.83 GB; CPU offloading the frozen backbone at FP32 brings the end-to-end footprint to 2.12 GB without affecting quality, and the resulting representation renders at 689 fps.

Interpolation versus reconstruction. All results above hold out alternate frames; with every captured frame available the field reaches 25.50 dB. Figure 3 shows the difference: deformation-field methods interpolate only between fitted observations, whereas our trajectories are continuous in time.

## 4.6. Streaming to Edge Devices

We evaluate Persistent Homology Pruning and Compression on the fitted fields of all nine scenes against two training-free criteria from current Gaussian compression systems: the global significance score of LightGaussian [8] and the accumulated blending contribution of Rad-Splat [25]. All criteria rank the same field and share the same coding stage, so the comparison isolates which Gaussians are kept. We also report the topology preservation rate, the fraction of ε-persistent features preserved, which quantifies Eq. 12.

![](images/774d52886ddaf25b9d5d30e6cdfe0a856b438760312fc20d9bcfed2f6f867f8c.jpg)

![](images/d3a94c256b21967bb5d7724a98d8370e6af922cf58f179452b7ebc5f9985552e.jpg)  
Figure 5. Streaming rate-distortion. Nine-scene means against coded bitrate. Left: PSNR after decoding. Right: fraction of ε- persistent features preserved.

Our packages are the strongest at every budget below the point where pruning becomes lossless, and the margin widens as the budget tightens: both baselines weight each Gaussian by a footprint term and remove thin structures first. No single ranking is best everywhere: at the tighter budgets opacity wins on most scenes, the blending contribution and visibility mass each win on one, and on one scene opacity collapses. Selecting the package by its fidelity to the unpruned field recovers the best candidate on almost every scene-budget pair, which places our rows above every criterion.

Clip package and bitrate. The raw FP32 representation averages 61.1 MB per clip; quantizing the static attributes to FP16/INT8 and the nine trajectory coefficients to 10 bits costs less than 0.01 dB. The coded packages are 2.8 and 1.4 MB at $4 \times 1 0 ^ { 4 }$ and $2 \times 1 0 ^ { 4 }$ Gaussians, which stream at 16.8 and 8.4 Mbps and render at 1312 and 1400 fps.

Deployment on an untethered headset. On a Meta Quest Pro a WebGL viewer decodes the representation in shader: the $4 \times 1 0 ^ { 4 } .$ -Gaussian package transfers over WiFi in 0.44 s at 58 Mbps against 2.3 s for the uncompressed field, and renders in immersive stereo at 87 fps against the headset’s 90 Hz display. Figure 5 plots the comparison across the whole bitrate range. The pipeline therefore runs end to end: a single consumer GPU, a few-megabyte package, ordinary WiFi, and interactive rendering on a standalone headset.

## 5. Conclusion

We study dynamic Gaussian reconstruction under a joint reconstruction, rendering, and hardware budget for XR deployment. Our experiments show that optimization from scratch cannot satisfy the target budget, while increasing the capacity of feed-forward prediction does not eliminate its reconstruction-quality gap, motivating feed-forward initialization followed by lightweight refinement. Amortized Anchor Refinement implements this strategy with a frozen backbone, budget-constrained anchor refinement, and topologically constrained compression, and our experiments demonstrate that it achieves the required reconstruction quality within the target budget, enabling reconstruction on a single consumer GPU and playback on a standalone headset.

## Acknowledgments

This work was partially supported by the U.S. National Science Foundation under Grant No. 2316400. This work was also partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. The computations and data handling were enabled by the Arrhenius HPC resource provided by National Academic Infrastructure for Supercomputing in Sweden. The authors also acknowledge the South Dakota State University High Performance Computing (SDSU HPC) team for providing access to and support for the university’s high-performance computing resources.

## References

[1] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1

[2] Michael Broxton, John Flynn, Ryan S. Overbeck, Daniel Erickson, Peter Hedman, Matthew DuVall, Jason Dourgarian, Jay Busch, Matt Whalen, and Paul Debevec. Immersive light field video with a layered mesh representation. ACM Transactions on Graphics, 39(4), 2020. 1

[3] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 12

[4] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision (ECCV), 2024. 2, 12

[5] Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. Dna-rendering: A diverse neural actor repository for high-fidelity human-centric rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 14

[6] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting:

Towards efficient novel view synthesis for dynamic scenes. In ACM SIGGRAPH Conference Papers, 2024. 1, 2

[7] Herbert Edelsbrunner and John Harer. Computational Topology: An Introduction. American Mathematical Society, 2010. 5

[8] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 3, 8

[9] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1

[10] Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wenchao Ma, Le Chen, Danhang Tang, and Ulrich Neumann. Gaussianflow: Splatting gaussian dynamics for 4d content creation. arXiv preprint arXiv:2403.12365, 2024. 6

[11] Sharath Girish, Tianye Li, Amrita Mazumdar, Abhinav Shri vastava, David Luebke, and Shalini De Mello. Queen: Quantized efficient encoding of dynamic gaussians for streaming free-viewpoint videos. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 3

[12] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1

[13] Nikhil Varma Keetha, Norman Muller, Johannes¨ Schonberger, Lorenzo Porzi, Yuchen Zhang, Tobias ¨ Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulo, Christian Richardt, Deva Ramanan,\` Sebastian Scherer, and Peter Kontschieder. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414, 2025. 12

[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 2

[15] Junoh Lee, Changyeon Won, Hyunji Jung, Inhwan Bae, and Hae-Gon Jeon. Fully explicit dynamic gaussian splatting. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 1, 2

[16] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2

[17] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2

[18] Chenguo Lin, Yuchen Lin, Panwang Pan, Yifan Yu, Tao Hu, Honglei Yan, Katerina Fragkiadaki, and Yadong Mu. Movies: Motion-aware 4d dynamic view synthesis in one second. arXiv preprint arXiv:2507.10065, 2025. 2

[19] Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 12

[20] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3

[21] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In International Conference on 3D Vision (3DV), 2024. 1

[22] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), 2020. 1

[23] John Milnor. Morse Theory. Princeton University Press, 1963. 5

[24] Thomas Muller, Alex Evans, Christoph Schied, and Alexan-¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics, 41 (4):102:1–102:15, 2022. 1

[25] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari. Radsplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+ FPS. In International Conference on 3D Vision (3DV), 2025. 8

[26] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B. Goldman, Steven M. Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 1

[27] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B. Goldman, Ricardo Martin-Brualla, and Steven M. Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. ACM Transactions on Graphics, 40(6), 2021.

[28] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1

[29] Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, and Brian Curless. Film: Frame interpolation for large motion. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. 6

[30] Jiawei Ren, Kevin Xie, Ashkan Mirzaei, Hanxue Liang, Xiaohui Zeng, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, and Huan Ling. L4gm: Large 4d gaussian reconstruction model. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2

[31] Tianqi Shen, Shaohua Liu, Jiaqi Feng, Ziye Ma, and Ning An. Topology-aware 3d gaussian splatting: Leveraging per-

sistent homology for optimized structural integrity. In Pro ceedings of the AAAI Conference on Artificial Intelligence, 2025. 3

[32] Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei Zhao, and Wei Xing. 3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic freeviewpoint videos. In Proceedings of the IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 20675–20685, 2024. 2, 3

[33] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision (ECCV), 2024. 2, 12

[34] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 3, 12

[35] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick Labatut, Piotr Bo-¨ janowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. VGGT-Ω. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 2, 5, 12

[36] Penghao Wang, Zhirui Zhang, Liao Wang, Kaixin Yao, Siyuan Xie, Jingyi Yu, Minye Wu, and Lan Xu. V<sup>3</sup>: Viewing volumetric videos on mobiles via streamable 2d dynamic gaussians. ACM Transactions on Graphics, 43(6), 2024. 2, 3

[37] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jer´ ome Revaud. Dust3r: Geometric 3d vi-ˆ sion made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2

[38] Xuezhen Wang, Li Ma, Yulin Shen, Zeyu Wang, and Pedro V. Sander. Retimegs: Continuous-time reconstruction of 4d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7340–7350, 2026. 1, 2, 3, 5, 6

[39] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Scalable permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025. 12

[40] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 6, 18

[41] Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard A. Newcombe, and Zhaoyang Lv. 4dgt: Learn ing a 4d gaussian transformer using real-world monocular videos. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 2

[42] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high fidelity monocular dynamic scene reconstruction. In Pro-

ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 6, 18

[43] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Realtime photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In International Conference on Learning Representations (ICLR), 2024. 1

Table 6. Feed-forward backbone ablation on walk, refinement truncated to 24k steps. Each backbone supplies only depth. Foreground-masked PSNR (dB) on the splits of Table 3.
<table><tr><td>Backbone</td><td>Interp.A ↑ Recon.↑ Interp.B ↑</td><td></td></tr><tr><td>VGGT [34]</td><td>20.82 24.99</td><td>20.62</td></tr><tr><td>Depth Anything 3 [19]</td><td>20.79 25.24</td><td>20.89</td></tr><tr><td>π³[39]</td><td>21.09 25.47</td><td>20.97</td></tr><tr><td>MapAnything [13]</td><td>21.02 25.32</td><td>20.86</td></tr><tr><td>VGGT-Ω [35], ours</td><td>20.91</td><td>20.89</td></tr></table>

## A. Choice of Feed-forward Backbone

The Grounded Feed-forward Anchor uses a frozen backbone to initialize the Gaussian field. This backbone can in principle be replaced by any network that predicts per-view geometry. We evaluate five alternatives to quantify how sensitive the pipeline is to this choice.

Protocol. Each backbone provides a single output: a depth map per input view. All subsequent stages are held fixed, including ray grounding, mirror-symmetry completion, attribute initialization, refinement, and evaluation. No backbone-specific head is trained, and Gaussian colors are sampled from the input views rather than predicted, so the comparison isolates the predicted geometry. Predicted depth is defined up to an unknown scale, so each backbone is rescaled once such that its median depth matches the camera rig radius. We evaluate on walk with 33 frames and truncate refinement to 24k steps. Truncation reduces the cost of the sweep, but the ranking is not stable at shorter budgets, so we report the spread across backbones rather than their order. PSNR is foreground-masked and reported on the three evaluation splits of Table 3: Interp. on held cameras at held timestamps, Recon. on held cameras at supervised timestamps, and Interp. on supervised cameras at held timestamps.

Results. Table 6 reports quality after refinement on the three evaluation splits. The five backbones span 0.30 dB, which falls within the ±0.3 dB run-to-run variation of our protocol, so no backbone is statistically separated from the others. The spread remains small despite substantial differences in model capacity and output resolution: the backbone with the lowest depth resolution, 168×224 compared with up to 392×518, is not the weakest of the five. Refinement therefore depends on the structural correctness of the anchor rather than on its metric accuracy. An approximately correct geometry that avoids the coordinate degeneracy of unanchored initialization is sufficient, and all five backbones satisfy this requirement. This is consistent with the from-scratch collapse reported in Sec. 4, which is caused by the absence of a geometric reference rather than by an inaccurate one. We adopt VGGT-Ω for the main experiments because it supports CPU offloading to 2.12 GB, as required by the deployment budget, and not because it attains the highest score.

Exclusion of feed-forward splat predictors. A second class of feed-forward networks predicts Gaussians directly rather than depth [3, 4, 33], with recent methods targeting sparse-view novel-view synthesis. These methods assume a static scene observed by a moving camera and estimate geometry from inter-view parallax. Our anchor input has the opposite configuration: a stationary camera observing a moving subject, where inter-view differences encode motion rather than parallax. Evaluating these models on such input violates their geometric assumption, and the resulting scores would reflect this mismatch rather than model quality. A fair comparison requires a different anchor protocol, with one multi-view reconstruction per timestamp, which we leave outside the scope of this ablation.

## B. The Transmitted Package from Free Viewpoints

Held-out evaluation is defined at the capture cameras, so it does not report what a headset shows when the viewer walks around the subject. Figure 6 renders the transmitted package from three viewpoints outside the capture rig. At the deployment budget the package is 22× smaller than the raw field and is not visually distinguishable from it; the WebGL client decodes it in shader at 87 fps on a Meta Quest Pro.

![](images/71052bed6077c5f22b132b5e770c42170167321a83830018b2f7450039d2af7b.jpg)  
Figure 6. The transmitted package from free viewpoints. walk, before and after pruning and coding.

Figure 7 repeats this comparison against a pruning baseline on opencase, with the retained set fixed to the deployment budget for both. The three columns are produced by loading three package files into one client build, so the renderer, the decoder and the camera are identical and the only difference is which Gaussians survived. At this budget the baseline drops the feet and the lower half of the carried case, which are thin structures supported by few Gaussians, and the client renders the gaps as translucent stumps. Our package retains them. The failure is not a loss of texture detail but a loss of parts, which is what the topology floor is designed to prevent and what a viewer notices immediately when walking around the subject.

![](images/95e04c9ae84f77335d046a5cad680a01e4405cf2006e3b0a50870bfa5980dec0.jpg)  
Figure 7. Client decoding at the deployment budget. opencase, three viewpoints outside the capture rig, all decoded by the same WebGL client that runs on the headset. Columns share the viewpoint, the timestamp and the coder; only the ranking that selected the retained Gaussians differs.

## C. Degradation Along the Budget Ladder

Table 5reports the quality of a package as a number per budget. Figure 8 shows what those numbers look like. The top row prunes by the importance ranking alone and the bottom row adds the verified topology floor, with everything else held fixed. The two rows are indistinguishable at the loose budgets, where the ranking alone already retains every part of the subject. They separate as the budget tightens: the ranking-only row loses the face and the top of the carried case first, because those regions are covered by many small Gaussians of individually low importance, and the loss is abrupt rather than gradual. The floor spends part of the same budget on the Gaussians that carry those components and degrades their appearance instead of removing them.

## D. Reconstruction on All Nine Scenes

Figure 9 shows the refined field on every scene of Table 3at a held camera and a held timestamp, so no pixel in the bottom row was supervised. The scenes cover locomotion, two-person interaction, object manipulation and loose clothing. Reconstruction is faithful for the body and for rigid carried objects across all nine. The residual errors are concentrated where they are expected: the trailing edges of loose fabric, where the cubic trajectory smooths motion that is not polynomial, and the contact regions between two subjects, where the anchor supplies a single depth per ray and the mirrored back side of one subject falls inside the other.

## E. Comparison with Baselines on Further Scenes

Figure 3compares methods on a single scene. Figure 10 repeats the comparison on four more, at a held timestamp on a held camera, using the renders the two published implementations write for their own test split. The failure is the same in every scene and is temporal rather than spatial. Both baselines reconstruct the static parts of the subject, and both lose whatever moves fastest: the toy in bear disappears, the thrown object in doll smears into the background, and the two interacting subjects of passdoll blend into a single translucent volume. A deformation field queried at an unobserved timestamp returns an average of the poses it was trained on, and averaging is what these renders show. Our trajectories are evaluated at the timestamp itself, so the moving parts stay separate.

## F. Rendering Between Captured Frames

The representation is a function of continuous time, so it can be evaluated at timestamps that the capture never recorded. Figure 11 renders one interval at quarter-frame steps. Only the two endpoints were supervised; the midpoint is the heldout frame used for evaluation, and the six remaining timestamps have no corresponding capture at all. The carried object advances by a consistent increment at every step and no pose is repeated, so the motion between captured frames is interpolated rather than snapped to the nearest supervised state. This is the property that the held-timestamp protocol measures at one point per interval and that a viewer sees continuously during slow-motion playback.

## G. Residual Error Modes

Figure 12 shows the three error modes that remain, as zoom crops against ground truth at a held timestamp. Loose fabric loses its trailing edge, because a cubic trajectory cannot follow a fold that changes direction within one interval. Contact between two subjects is reconstructed as a translucent mixture, because the anchor supplies a single depth per ray and the mirrored back side of the nearer subject is placed inside the farther one. An object carried through a fast arc is blurred along its path, which is the same limitation as the first mode at a larger amplitude. All three are properties of the motion model rather than of the budget, and none of them is repaired by keeping more Gaussians.

![](images/2dc12cba683b4b5693e97d70df32c29d300f90e97ab675c47c8593d4972d4af2.jpg)  
Figure 8. A package along the budget ladder. opencase at a held camera and a held timestamp. Columns are Gaussian budgets; the leading column is the reference. Both rows use the same ranking, the same coder and the same starting field.

![](images/900b453431f618e992d13f4db2f4eb19530343121c86f7da691bf90904158dd5.jpg)  
Figure 9. All nine scenes at a held camera and a held timestamp. Crops are centred on the subject; the field is composited over the person-free plate of the same camera, as in Figure 3.

## H. Cross-Dataset Results on DNA-Rendering

Every experiment in the body uses one capture system. This section checks that the pipeline runs on another one: DNA-Rendering [5], an independent multi-view human capture with a different rig geometry, a different camera count, portrait rather than landscape framing, and subjects in garments the Stage-Capture scenes do not contain.

Protocol. We take three sequences from the released Part 1 and convert them into the same directory layout as the body experiments. The capture stores camera-to-world extrinsics together with lens distortion, so we undistort the images once and record the resulting pinhole intrinsics; the stored 3D keypoints reproject inside the person masks under this reading, which fixes the convention. From the 48- camera ring we take 32 evenly spaced cameras and the first 33 frames, giving the same 32×33 grid and the same everyother-frame hold-out as Table 3. The recipe is the one of Sec. 4: the same anchor, capacity floor, loss weights and step count, and the same rule for the densification window, which is the long one where the Gaussian count settles and the short one where it does not. The single adjustment is the shape of the anchor ray grid, which follows the portrait aspect ratio of this capture.

Lens distortion. Part of this capture is calibrated with a high-order radial model whose polynomial stops increasing inside the frame. Undistorting such a view with a standard rectification map folds the region beyond that radius back onto itself and duplicates image content into a border ring, which enters the person masks as foreground the reconstruction cannot explain. We therefore blank each view outside the radius where its polynomial remains monotonic. Ten of the 32 views of one sequence require this and none of the other two do; leaving the folded ring in place inflates the foreground of an affected view by half and makes the optimization diverge. The baselines are the same two public implementations, trained on the same converted sequences, and all numbers come from the evaluator used for the body tables.

![](images/4355b0165c92db700f4dda3c7282a8aea6565cf20f689f29d257a48da497c1fb.jpg)  
Figure 10. Held-timestamp interpolation on four further scenes. One held camera; no method was supervised on this timestamp. Crops are centred on the subject.

Results. Table 7 reports held-timestamp PSNR per sequence and Figure 13 shows the same sequences at a held timestamp. Our mean is 25.20 dB against 23.29 dB for Deformable-3DGS and 24.45 dB for 4D-GS, from fields averaging 42.4×10<sup>4</sup> Gaussians fitted in 51 minutes. We are ahead on two of the three sequences and behind 4D-GS by about a decibel on 0019 06, the sequence on which every method scores highest. Absolute values are not comparable with Table 3, since the subject covers a different fraction of the frame and the masks come from a different segmentation system. The claim here is only that the pipeline runs on a capture it was never tuned on and reconstructs it at a quality comparable to methods trained on the same sequences, with the anchor supplying usable geometry for a rig and a wardrobe it has not seen.

![](images/a080c70adb29c67e82bb88aedf5e163843f2586abfce119ca600d1b94e46e950.jpg)  
Figure 11. Quarter-frame rendering across one interval of doll. Top: the capture, blank where no frame exists. Bottom: our render at every timestamp.

Loose fabric  
Two-subject contact  
Fast limb motion  
![](images/e71be12f55c2a97ba1601eb8b55a44499b0abf736a82620905e41cee2c75bfdd.jpg)  
Figure 12. Residual error modes at a held timestamp on a held camera. Crops from newundress, passdoll and stretch.

![](images/c93b72dd31dd444939602424ce2317227192cf569f8050d27be1d6da532a43eb.jpg)  
Figure 13. Held-timestamp interpolation on DNA-Rendering. Two held cameras per sequence, front and back; no method was super vised on this timestamp. Crops are centred on the subject.

Table 7. Cross-dataset results on DNA-Rendering. Foreground-masked held-timestamp PSNR (dB) per sequence, with the mean SSIM and LPIPS of the same split. Protocol and evaluator as in Table 3.
<table><tr><td>Method</td><td>0008_01</td><td>0019_06</td><td>0047_12</td><td>Mean↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Deformable-3DGS [42]</td><td>20.02</td><td>27.60</td><td>22.24</td><td>23.29</td><td>0.984</td><td>0.017</td></tr><tr><td>4D-GS [40]</td><td>22.70</td><td>29.56</td><td>21.09</td><td>24.45</td><td>0.984</td><td>0.021</td></tr><tr><td>Ours</td><td>24.31</td><td>28.57</td><td>22.72</td><td>25.20</td><td>0.987</td><td>0.013</td></tr></table>