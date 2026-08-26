# Low-Rank Velocity Fields as a Structural Prior for Unsupervised 4D Medical Image Interpolation

Haojin Li<sup>1</sup>, Hengzhuo Wang<sup>2,3</sup>, Chang Liu<sup>2</sup>, Zhiheng Ma<sup>4</sup>, Heng Li<sup>2,5B</sup>, and Jiang Liu<sup>1,5B</sup>

<sup>1</sup> Department of Computer Science and Engineering, Southern University of Science and Technology, Shenzhen, China

Faculty of Biomedical Engineering, Shenzhen University of Advanced Technology, Shenzhen, China

3 School of Biomedical Engineering, Shenzhen University Medical School, Shenzhen University, Shenzhen, China

<sup>4</sup> Faculty of Computility Microelectronics, Shenzhen University of Advanced Technology, Shenzhen, China

5 Research Institute of Trustworthy Autonomous Systems, Southern University of Science and Technology, Shenzhen, China

B Corresponding authors: liheng@suat-sz.edu.cn, liuj@sustech.edu.cn

Abstract. Endpoint-only unsupervised 4D medical image interpolation synthesizes intermediate volumes from sparsely sampled sequences with only the start and end volumes available for training; however, this weakly constrained setting often yields intermediates with unstable boundaries and non-physiological motion, limiting interpretability and downstream analysis. We propose low-rank velocity fields as a structural prior, constraining motion to a structured Tucker low-rank velocity field space that decomposes motion into globally shared spatial bases and a compact sample-specific core, thereby encouraging spatially correlated, anatomyconsistent deformation while suppressing voxel-wise high-frequency artifacts. To capture global coordination and local non-rigid details, we model motion in a coarse-to-fine multi-scale scheme and compose scalewise deformations at inference to synthesize volumes at arbitrary times. We further provide a theoretical analysis showing that, under Tucker parameterization, low-rank parameters control the smoothness energy of the velocity field, explaining why low-rank modeling promotes smoother motion. Experiments on ACDC and 4D-Lung demonstrate state-of-theart performance, remaining competitive with methods trained with interm frame supervision, and producing intermediates with improved structural coherence and more stable anatomical contours.

Keywords: 4D Medical Imaging · 4D Image Interpolation · Low-Rank Velocity Fields · Motion Modeling

## 1 Introduction

4D medical imaging captures anatomical dynamics and supports functional assessment, therapy monitoring, and longitudinal disease tracking [22]. However, practical acquisition constraints often lead to sparse and irregular temporal sampling, motivating 4D interpolation to synthesize intermediate volumes and increase temporal resolution for downstream analysis such as segmentation [17] and reconstruction [15]. Here, we study endpoint-only unsupervised 4D medical interpolation, where training provides only two endpoint volumes.

Prior volumetric interpolation methods are broadly motion-driven, warping endpoints via estimated deformations (or flows) [7, 21, 12], or synthesisdriven, directly generating intermediate volumes [23, 24], and are sometimes combined [11, 5]. For medical imaging, plausible intermediates must preserve anatomical continuity and topology while evolving smoothly [25]. Under endpoint-only supervision, fully voxel-wise, high-dimensional motion parameterizations can overfit locally inconsistent high-frequency motion, leading to boundary instability, local misalignment, and non-physiological warping that degrades interpretability and downstream usability.

Anatomical motion is strongly spatially correlated and partially compressible: coordinated global patterns dominate, while local non-rigid details remain residual variation [18, 9]. This motivates an anatomy-aligned, compact motion representation rather than increased voxel-level freedom; thus we adopt structured low-rank motion modeling to suppress spurious high-frequency motion and improve anatomical consistency and temporal coherence of synthesized intermediates.

Two challenges arise when introducing structured low-rank motion into endpointonly unsupervised 4D interpolation: (1) how to impose a compact, low-dimensional motion parameterization that captures shared spatial coordination yet adapts to each sample under weak supervision; (2) how to model motion across scales, as coarse global trends and fine non-rigid details call for a multi-scale representation and refinement [19].

To tackle these challenges, we propose a framework for endpoint-only unsupervised 4D medical image interpolation that introduces low-rank velocity fields as a structural prior. The framework separates shared coordinated motion from sample-specific variation and organizes motion modeling in a multi-scale hierarchy for coarse-to-fine refinement, thereby guiding endpoint-only learning toward anatomically plausible dynamics. Our contributions are summarized as follows:

– We present low-rank velocity fields as a structural prior for endpoint-only unsupervised 4D medical interpolation, improving anatomical consistency and temporal coherence, with analysis showing that low-rank parameterization bounds smoothness energy and suppresses high-frequency motion artifacts.

– We introduce a Tucker-parameterized low-rank velocity representation with shared spatial bases and a compact, sample-dependent core to capture coordinated motion while retaining individual variation.

– We propose a multi-scale low-rank motion modeling strategy that allocates capacity across resolutions for coarse-to-fine refinement from global coordination to local non-rigid deformation.

– Experiments on ACDC and 4D-Lung show strong results, achieving the best structure-oriented scores with competitive reconstruction accuracy.

![](images/48f3228ba510008653608b1e2d2ac3413dd0478def6e11abe0da1e3009b1773d.jpg)  
Fig. 1. Overview of the proposed framework. (a) Multi-scale endpoint features are projected into a structured low-rank velocity representation across resolutions via Tucker decomposition, aligning motion modeling with the coordinated and compact nature of anatomical dynamics. (b) Inference performs bi-directional continuous-time interpolation to produce structurally consistent intermediate volumes.

## 2 Method

## 2.1 Problem Definition

Unsupervised 4D interpolation is formulated on a spatial domain $\varOmega \subset \mathbb { R } ^ { 3 }$ , modeling an image sequence as a continuous-time function $I ( t ) : \varOmega \to$ R. We adopt endpoint-only training, observing only the endpoint volumes $I _ { 0 } : = I ( 0 )$ and $I _ { 1 } : = I ( 1 )$ , while intermediate volumes $I _ { \alpha }$ with $\alpha \in ( 0 , 1 )$ are unavailable.

Given $( I _ { 0 } , I _ { 1 } )$ , a model with parameters θ predicts an inter-endpoint velocity field for either ordered pair $\left( I _ { a } , I _ { b } \right)$ with $( a , b ) \in \{ ( 0 , 1 ) , ( 1 , 0 ) \}$ , represented on the voxel grid as a tensor $\mathcal { V } _ { a  b }$ . It is converted to a deformation $\phi _ { a  b } = \mathrm { E x p } ( \mathcal { V } _ { a  b } )$ ， where $\operatorname { E x p } ( { \boldsymbol { \cdot } } )$ is the exponential map of a stationary velocity field implemented via scaling-and-squaring, and the target endpoint is reconstructed by warping $\hat { I } _ { b } = I _ { a } \circ \phi _ { a  b }$ , where ◦ applies a deformation field to an image. Learning is driven by an endpoint reconstruction objective:

$$
\theta ^ { * } \ = \ \arg \operatorname* { m i n } _ { \theta } \ \mathbb { E } _ { ( I _ { a } , I _ { b } ) } \Big [ \mathcal { L } \big ( I _ { b } , \hat { I } _ { b } \big ) \ + \ \lambda \mathcal { R } \big ( \mathcal { V } _ { a  b } \big ) \Big ] ,\tag{1}
$$

where $\mathcal L ( \cdot , \cdot )$ is an endpoint similarity measure $\left( \mathrm { e . g . } \right.$ , normalized cross-correlation;   
NCC), and $\mathcal { R } ( \cdot )$ regularizes the predicted velocity to promote plausible motion.

## 2.2 Structured Low-rank Motion Representation

Tucker Parameterization of Velocity Fields: To model high-dimensional non-rigid motion stably under endpoint-only training (Fig. 1(a)), we constrain the predicted velocity to a structured low-rank subspace instead of a fully voxelwise free form. A 3D velocity field on $\varOmega \subset \mathbb { R } ^ { 3 }$ , denoted $v ( x ) = ( v _ { x } ( x ) , v _ { y } ( x ) , v _ { z } ( x ) )$ for $x \in \Omega $ , is represented on the voxel grid as a fourth-order tensor $\nu \in$ $\mathbb { R } ^ { H \times W \times D \times 3 }$ . In what follows, V refers to this voxel-grid tensor form when defining the low-rank parameterization and regularization.

We decompose V into globally shared spatial bases and a compact core of combination coeficients [13, 6]:

$$
\mathcal { V } = \mathcal { G } \times _ { 1 } U ^ { ( 1 ) } \times _ { 2 } U ^ { ( 2 ) } \times _ { 3 } U ^ { ( 3 ) } ,\tag{2}
$$

where $\mathcal { G } \in \mathbb { R } ^ { r \times r \times r \times 3 }$ is the low-dimensional core tensor, $U ^ { ( 1 ) } \in \mathbb { R } ^ { H \times r }$ $U ^ { ( 2 ) } \in$ $\mathbb { R } ^ { W \times r }$ , and $U ^ { ( 3 ) } \in \mathbb { R } ^ { D \times { r } }$ are spatial bases along the three axes, and $\times _ { n }$ denotes the mode-n tensor–matrix product. The bases capture domain-consistent, globally shared deformation structure, while G provides sample-dependent coefficients; thus, spatial correlations are absorbed by the shared bases and instance variation is governed by the compact core.

To stably construct the low-rank subspace, each spatial basis uses a fixed analytic DCT basis with learnable mixing within the same low-frequency span:

$$
U ^ { ( i ) } = B ^ { ( i ) } R _ { : , 1 : r } ^ { ( i ) } , \qquad B ^ { ( i ) } \in \mathbb { R } ^ { N _ { i } \times K } , ~ R ^ { ( i ) } \in \mathbb { R } ^ { K \times K } , ~ U ^ { ( i ) } \in \mathbb { R } ^ { N _ { i } \times r } ,\tag{3}
$$

where $B ^ { ( i ) }$ is a fixed truncated (low-frequency) DCT basis, $R ^ { ( i ) }$ is identityinitialized and optimized to mix DCT directions within this span, and $U ^ { ( i ) }$ is the resulting rank-r spatial basis. Restricting $U ^ { ( i ) }$ to a low-frequency span serves as a band-limited structural prior that discourages voxel-wise high-frequency oscillations in the induced motion [10, 14]. We treat $K$ as a frequency budget; if $K > N _ { i }$ , the efective span is naturally limited by $N _ { i }$

Multi-scale Motion Modeling: Building on the Tucker parameterization, we introduce multi-scale modeling to capture both global and local motion by predicting the velocity field coarse-to-fine at three resolutions, $1 / 4 , 1 / 2$ , and 1. Each scale $s \in \{ 1 / 4 , 1 / 2 , 1 \}$ } outputs a scale-specific low-rank velocity tensor $\mathcal { V } ^ { ( s ) }$ with its own spatial bases and core $\mathcal { G } ^ { ( s ) }$ ; the Tucker rank at scale s is denoted by $r ^ { ( s ) }$ 2 so that $\bar { \mathcal { G } } ^ { ( s ) } \in \mathbb { R } ^ { r ^ { ( s ) } \times r ^ { ( s ) } \times r ^ { ( s ) } \times 3 }$ . The core $\mathcal { G } ^ { ( s ) }$ is regressed from scale-specific endpoint features, and all scale-wise branches are trained jointly under a unified multi-scale objective, with scales coupled via coarse-to-fine deformation composition for progressive refinement and without explicit cross-scale feature fusion.

During training, we compute and aggregate scale-wise endpoint reconstruction and smoothness losses. Let $I ^ { ( s ) }$ denote endpoints downsampled to scale s. Using Exp(·) and the warping operator defined above, we integrate the scalewise velocity $\mathcal { V } _ { a  b } ^ { ( s ) }$ into a deformation $\phi _ { a  b } ^ { ( s ) }$ (progressively refined from coarse to fine) and warp $I _ { a } ^ { ( s ) }$ to reconstruct $\hat { I } _ { b } ^ { ( s ) }$ :

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \sum _ { s } \omega ^ { ( s ) } \Big ( - \lambda _ { 1 } \operatorname { N C C } ( I _ { b } ^ { ( s ) } , \hat { I } _ { b } ^ { ( s ) } ) + \lambda _ { 2 } \ \big \| I _ { b } ^ { ( s ) } - \hat { I } _ { b } ^ { ( s ) } \big \| _ { \mathrm { C } } + \lambda _ { 3 } \mathcal { R } _ { H ^ { 1 } } ( \mathcal { V } _ { a  b } ^ { ( s ) } ) \Big ) ,\tag{4}
$$

where $\begin{array} { r } { \sum _ { s } \omega ^ { ( s ) } = 1 , \operatorname { N C C } ( \cdot , \cdot ) } \end{array}$ denotes local normalized cross-correlation, $\Vert \cdot \Vert _ { \mathrm { C } }$ is the Charbonnier norm, and $\mathcal { R } _ { H ^ { 1 } }$ penalizes velocity magnitude and spatial gradients as an additional safeguard for smooth, integrable motion. All reconstruction terms at scale s are evaluated between $I _ { b } ^ { ( s ) }$ and $\hat { I } _ { b } ^ { ( s ) }$ at the same resolution.

## 2.3 Inference for Continuous-Time Interpolation

At inference (Fig. 1(b)), we evaluate the learned multi-scale low-rank representation to synthesize intermediate volumes at arbitrary $\alpha \in ( 0 , 1 )$ . Given endpoints $( I _ { 0 } , I _ { 1 } )$ , the predicted velocity tensors at three spatial scales, $\hat { \mathcal { V } } ^ { ( 1 / 4 ) } , \mathcal { V } ^ { ( 1 / 2 ) } , \mathcal { V } ^ { ( 1 ) }$ are resampled to a common voxel grid, with magnitudes scaled according to resolution. For each direction $a  b ,$ we integrate the scale-specific velocity via Exp(·) to obtain a deformation $\phi _ { a  b } ^ { ( s ) } .$ and form the endpoint deformation by coarse-to-fine composition:

$$
\phi _ { a  b } = \phi _ { a  b } ^ { ( 1 ) } \circ \phi _ { a  b } ^ { ( 1 / 2 ) } \circ \phi _ { a  b } ^ { ( 1 / 4 ) } .\tag{5}
$$

Deformations are applied right-to-left across scales (coarse to fine), and each $\phi _ { a  b } ^ { ( s ) }$ is computed by integrating the scale-s velocity with Exp(·) and resampling it to the common grid before composition.

To obtain an intermediate time α, we use both predicted directions $( 0  1$ and $1  0 )$ and scale each stationary velocity by the corresponding travel time (proportional to α or $1 - \alpha )$ prior to exponential-map integration [20], yielding $\phi _ { 0 \to \alpha }$ and $\phi _ { 1  \alpha }$ . The intermediate prediction is then computed by warping both endpoints and fusing them with time-based weights:

$$
\hat { I } _ { \alpha } = ( 1 - \alpha ) ( I _ { 0 } \circ \phi _ { 0  \alpha } ) + \alpha ( I _ { 1 } \circ \phi _ { 1  \alpha } ) .\tag{6}
$$

## 2.4 Theoretical Analysis of Energy Bounds

Proposition: Energy control by low-rank parameters. In Tucker form, the smoothness energy of the full velocity field is upper-bounded by the core magnitude, up to a multiplicative constant determined by the smoothing operator and the spatial bases; consequently, bounding the factor norms and the core norm controls the regularity of the induced deformation [13]. Define the quadratic smoothness energy:

$$
E ( \mathcal { V } ) : = \big \| L \operatorname { v e c } ( \mathcal { V } ) \big \| _ { 2 } ^ { 2 } ,\tag{7}
$$

where L is a fixed linear smoothing/diferential operator and vec(·) stacks tensor entries into a vector. Let $\lambda _ { \mathrm { m a x } }$ be the largest eigenvalue of $L ^ { \top } L , { \mathrm { i . e . , ~ } } \lambda _ { \mathrm { m a x } } =$ $\| L \| _ { 2 } ^ { 2 }$ . For a Tucker-parameterized velocity tensor $\mathcal { V } = \mathcal { G } \times _ { 1 } U ^ { ( 1 ) } \times _ { 2 } U ^ { ( 2 ) } \times _ { 3 } U ^ { ( 3 ) }$ ， the following upper bound holds:

$$
E ( \mathcal { V } ) \ \leq \ \lambda _ { \operatorname* { m a x } } \Big ( \prod _ { i = 1 } ^ { 3 } \| U ^ { ( i ) } \| _ { 2 } \Big ) ^ { 2 } \| \mathcal { G } \| _ { F } ^ { 2 } ,\tag{8}
$$

where $\| \cdot \| _ { 2 }$ is the matrix spectral norm and $\| \cdot \| _ { F }$ is the Frobenius norm.

Proof. By the operator norm inequality, $\| L \operatorname { v e c } ( \mathcal { V } ) \| _ { 2 } \ \leq \ \| L \| _ { 2 } \| \operatorname { v e c } ( \mathcal { V } ) \| _ { 2 }$ , we have $E ( \mathcal { V } ) \leq \lambda _ { \operatorname* { m a x } } \lvert \lvert \mathcal { V } \rvert \rvert _ { F } ^ { 2 }$ . Moreover, for any mode-n product, $\| { \mathcal { X } } \times _ { n } U \| _ { F } ~ \le ~$ $\| U \| _ { 2 } \| \mathcal { X } \| _ { F } ~ [ 2 ] ;$ ; applying this sequentially to the Tucker reconstruction yields $\begin{array} { r } { \| \mathcal { V } \| _ { F } \leq ( \prod _ { i = 1 } ^ { 3 } \| U ^ { ( i ) } \| _ { 2 } ) \| \mathcal { G } \| _ { F } } \end{array}$ . Combining the two bounds proves (8).

Corollary (Rank scaling). Under bounded factors $\| U ^ { ( i ) } \| _ { 2 } \leq c$ and bounded core amplitude $| \mathcal { G } _ { p q r \ell } | \le \beta$ for constants c and $\beta ,$ the worst-case energy bound scales at most cubically with the rank: for $\mathcal { G } \in \mathbb { R } ^ { r \times r \times r \times 3 }$ 2 $\| \mathcal { G } \| _ { F } ^ { 2 } \le 3 r ^ { 3 } \bar { \beta } ^ { 2 }$ , hence

$$
E ( \nu ) \leq 3 \lambda _ { \operatorname* { m a x } } c ^ { 6 } r ^ { 3 } \beta ^ { 2 } .\tag{9}
$$

The same bound applies to each scale s by replacing r with $r ^ { ( s ) }$

## 3 Experiment

## 3.1 Experimental Settings

Datasets: We evaluate on ACDC [3] and 4D-Lung [8]. ACDC contains 150 cardiac MRI cases with an 80/20/50 (train/val/test) split, resized to $1 6 0 \times 1 6 0 \times$ 16; end-diastolic and end-systolic phases are used as endpoints. 4D-Lung contains 500 cone-beam CT volumes with a patient-level $3 0 6 / 8 4 / 1 1 0$ split, resized to $1 2 8 \times 1 2 8 \times 3 2 ; 0 \%$ and 50% respiratory phases are used as start/end frames. All volumes are histogram-equalized and linearly normalized to [0, 1].

Metrics: We report Normalized Mutual Information (NMI), Normalized Mean Squared Error (NMSE), and Structural Similarity (SSIM) to measure statistical dependence, reconstruction error, and structural similarity of results.

Implementation Settings: All methods are implemented in PyTorch and trained on NVIDIA RTX A6000 GPUs under a unified protocol. We use AdamW with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ and a cosine annealing scheduler. Models are trained for 500/200 epochs on ACDC/4D-Lung with early stopping; batch size is 1 with gradient accumulation of 8. For our method, endpoint features are extracted at three resolutions $( 1 / 4 , 1 / 2 , 1 )$ using lightweight scale-specific 3D convolutional encoder branches trained jointly; scale-wise motion heads refine from coarse to fine, and low-frequency bases are truncated to $K = 1 2 8$ . Scale weights are set to $\omega ^ { ( 1 / 4 ) } = 0 . 2 , \stackrel { \textstyle - } { \omega } ^ { ( 1 / 2 ) } = 0 . 3$ , and $\omega ^ { ( 1 ) } = 0 . 5 ;$ Tucker ranks are set to $r ^ { ( 1 / 4 ) } = 6 4 , r ^ { ( 1 / 2 ) } = 3 2$ , and $r ^ { ( 1 ) } = 8$ . We set $\lambda _ { 1 } = \lambda _ { 2 } = 1$ and $\lambda _ { 3 } = 0 . 0 5$

## 3.2 Comparison Experiment

We compare against classical unsupervised deformation models (Voxelmorph (VM) [1], Transmorph (TM) [4]), recent state-of-the-art unsupervised interpolation networks (SVIN [7], UVINet [12]), and representative methods trained with intermediate-frame supervision (DDM [11], LDDM [5], FB-Dif [23], TMSDF [24])

The quantitative results in Tab. 1 reveal distinct performance patterns across the evaluated methods. While recent unsupervised interpolation models perform strongly on ACDC, they exhibit a notable gap on 4D-Lung due to the challenge of preserving globally coordinated respiratory motion. Conversely, methods utilizing intermediate-frame supervision often reduce reconstruction error, yet their NMI/SSIM scores do not consistently improve, confirming that appearance gains do not necessarily translate into stronger anatomical correspondence. Our approach achieves leading NMI/SSIM on both datasets, particularly on 4D-Lung, while maintaining competitive reconstruction error. Despite relying only on endpoint supervision, it reaches supervised-level overall quality and yields more anatomically consistent, temporally coherent intermediates.

Table 1. Quantitative comparison on ACDC and 4D-Lung datasets.
<table><tr><td rowspan="2">Method Year</td><td rowspan="2"></td><td colspan="3">ACDC</td><td colspan="3">4D-Lung</td></tr><tr><td> $\mathrm { N M I } _ { \times 1 0 ^ { - 2 } } ^ { \uparrow }$ </td><td> $\mathrm { N M S E } _ { \times 1 0 ^ { - 3 } } ^ { \downarrow }$ </td><td> $\mathrm { S S I M _ { \times 1 0 ^ { - 2 } } ^ { \uparrow } }$ </td><td> $\mathrm { N M I } _ { \times 1 0 ^ { - 2 } } ^ { \uparrow }$ </td><td> $\mathrm { N M S E _ { \times 1 0 ^ { - 3 } } ^ { \downarrow } }$ </td><td> $\mathrm { S S I M _ { \times 1 0 ^ { - 2 } } ^ { \uparrow } }$ </td></tr><tr><td>VM</td><td>2019</td><td> $5 3 . 5 4 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $1 9 . 1 8 _ { \pm 0 . 3 4 }$ </td><td> $9 4 . 6 8 _ { \pm 0 . 1 0 }$ </td><td> $4 3 . 9 8 _ { \pm 0 . 3 6 }$ </td><td> $3 9 . 0 2 _ { \pm 0 . 6 0 }$ </td><td> $8 6 . 0 6 { \scriptstyle \pm 0 . 2 9 }$ </td></tr><tr><td>TM</td><td>2022</td><td> $6 4 . 6 3 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $1 4 . 4 3 _ { \pm 0 . 6 2 }$ </td><td> $9 6 . 6 4 _ { \pm 0 . 1 2 }$ </td><td> $4 8 . 9 5 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $3 4 . 9 6 _ { \pm 0 . 4 0 }$ </td><td> $8 4 . 7 8 { \scriptstyle \pm 0 . 1 2 }$ </td></tr><tr><td>SVIN</td><td>2020</td><td> $6 0 . 1 8 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $9 . 8 5 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $9 7 . 3 5 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $4 3 . 1 7 _ { \pm 0 . 3 9 }$ </td><td> $3 0 . 2 8 { \scriptstyle \pm 2 . 0 3 }$ </td><td> $8 5 . 4 4 { \scriptstyle \pm 0 . 2 9 }$ </td></tr><tr><td>UVINet 2024</td><td></td><td> $7 2 . 2 5 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $4 . 8 1 _ { \pm 0 . 1 2 }$ </td><td> $9 8 . 6 2 _ { \pm 0 . 0 3 }$ </td><td> $5 5 . 1 6 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $2 0 . 9 3 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $8 8 . 7 5 { \scriptstyle \pm 0 . 6 7 }$ </td></tr><tr><td>DDM</td><td>2022</td><td> $6 4 . 1 4 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $1 0 . 3 6 _ { \pm 0 . 3 0 }$ </td><td> $9 7 . 2 4 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $4 2 . 5 4 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $3 3 . 1 9 _ { \pm 0 . 7 3 }$ </td><td> $8 5 . 2 3 { \scriptstyle \pm 0 . 2 7 }$ </td></tr><tr><td>LDDM</td><td>2024</td><td> $7 2 . 4 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $4 . 8 8 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $9 8 . 5 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $5 6 . 5 8 { \scriptstyle \pm 8 . 0 8 }$ </td><td> $2 8 . 6 2 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $8 3 . 5 0 { \scriptstyle \pm 6 . 0 9 }$ </td></tr><tr><td>FB-Diff 2025</td><td></td><td> $6 1 . 3 3 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $8 . 6 4 _ { \pm 0 . 2 1 }$ </td><td> $9 7 . 2 9 _ { \pm 0 . 0 9 }$ </td><td> $4 9 . 2 8 _ { \pm 1 . 3 4 }$ </td><td> $2 7 . 3 9 _ { \pm 6 . 2 6 }$ </td><td> $8 0 . 9 1 _ { \pm 3 . 7 1 }$ </td></tr><tr><td>TMSDF 2025</td><td></td><td> $6 6 . 4 0 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $\mathbf { 4 . 7 2 _ { \pm 0 . 2 3 } }$ </td><td> $9 8 . 5 6 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $4 8 . 3 6 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $\mathbf { 1 6 . 8 5 { \scriptstyle \pm 0 . 0 9 } }$ </td><td> $9 0 . 2 6 { \scriptstyle \pm 1 . 0 9 }$ </td></tr><tr><td>Ours</td><td>-</td><td> $\mathbf { 7 2 . 6 1 _ { \pm 0 . 1 3 } }$ </td><td> $4 . 7 4 _ { \pm 0 . 0 5 }$ </td><td> $\mathbf { 9 8 . 6 2 _ { \pm 0 . 0 2 } }$ </td><td> ${ \bf 6 2 . 3 4 } _ { \pm 0 . 2 0 }$ </td><td> $1 7 . 0 0 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $\mathbf { 9 1 . 2 9 _ { \pm 0 . 1 3 } }$ </td></tr></table>

Gray background : methods trained with intermediate-frame supervision.

![](images/956a0ebc60047553449022dadb906c3962f75bee2d677b229aa028629e5cbac5.jpg)  
Fig. 2. Qualitative comparison on ACDC and 4D-Lung datasets.

Fig. 2 shows qualitative comparisons. On ACDC, our method produces more regular contours and stable anatomy, especially in complex regions; on 4D-Lung, it better preserves coherent global organization. Baselines often exhibit local inconsistencies, while difusion-style methods may improve appearance but compromise structural stability, yielding more anatomically consistent and visually stable intermediates.

## 3.3 Ablation Study

Efect of Multi-scale Rank: As shown in Fig. 3(a), we fix the coarse Tucker rank (given its low sensitivity) and vary the mid/fine ranks, reporting NMI. Increasing Tucker rank improves interpolation quality in low-capacity regimes, while the gain saturates once the fine-scale rank becomes large. Fig. 3(b) reports the Efective Rank of the resulting velocity fields (computed following [16]) as a proxy for motion complexity: lower Efective Rank corresponds to simpler, more compressible motion, and it increases monotonically with Tucker rank in our setting. Performance is best at low-to-moderate Efective Ranks, whereas overly small ranks underfit and degrade accuracy. Overall, Tucker rank ofers a direct knob to tune motion complexity via the induced Efective Rank, and both under-constraint and excessive capacity can lead to suboptimal results.

![](images/c975745053fb570976cae599fdaf14e9eecd640a9235cf36232261d2a96e0591.jpg)

(b) Effective Rank vs. r(s)  
![](images/3df47b26c56ef4ce0e051c1a94d14878eaab3af8732f76c20fae437d0f26e057.jpg)

![](images/cdaeca687a652b00bdd3a876a73c4c66d029e361ba5d514d21ec1b13c98757b9.jpg)  
Fig. 3. Ablation on ACDC. (a) NMI under diferent Tucker rank settings. (b) Velocity field Efective Rank [16] (log<sub>10</sub>-scaled). (c) NMI vs. scale weights (ternary diagram).

Table 2. Loss term ablation on ACDC.
<table><tr><td>NCC</td><td>Charb</td><td>Reg</td><td> $\mathrm { N M I } _ { \times 1 0 ^ { - 2 } } ^ { \uparrow }$ </td><td> $\mathrm { N M S E } _ { \times 1 0 ^ { - 3 } } ^ { \downarrow }$ </td><td> $\mathrm { S S I M } _ { \times 1 0 ^ { - 2 } } ^ { \uparrow }$ </td></tr><tr><td>√</td><td></td><td>√</td><td>72.22</td><td>4.69</td><td>98.63</td></tr><tr><td></td><td>√</td><td>√</td><td>72.35</td><td>4.81</td><td>98.59</td></tr><tr><td>√</td><td>√</td><td></td><td>71.45</td><td>5.18</td><td>98.49</td></tr><tr><td>√</td><td>√</td><td>V</td><td>72.61</td><td>4.74</td><td>98.62</td></tr></table>

Efect of Scale Weights: We ablate the scale-wise loss weights in Fig. 3(c) by enforcing $\begin{array} { r } { \sum _ { s } \omega ^ { ( s ) } = 1 } \end{array}$ and visualizing NMI on a ternary plot. Overall, allocating more weight to the fine scale tends to improve performance, but overly extreme fine-dominant settings degrade NMI; the best region favors a fine-leaning yet balanced combination that retains non-negligible mid/coarse contributions. This indicates that fine-scale supervision is most efective for refinement, while mid/coarse guidance helps preserve global coherence and prevent locally driven inconsistencies, yielding the most reliable training signal across scales.

Loss-Term Ablation: Tab. 2 ablates the loss by enabling/disabling NCC, the Charbonnier term, and the regularizer. Dropping Charbonnier can slightly improve NMSE/SSIM but may destabilize optimization, suggesting it complements NCC by tolerating local intensity deviations under unsupervised reconstruction. Regularization is also beneficial: without it, the model tends to produce larger and noisier motions, whereas enabling it stabilizes the scale of core tensors and curbs excessive growth of low-rank parameters, yielding smoother fields and more robust training.

## 4 Conclusion

We proposed a structured low-rank framework for unsupervised 4D medical image interpolation. By parameterizing velocity fields with a Tucker low-rank form and modeling motion in a coarse-to-fine multi-scale manner, the method favors spatially correlated, anatomy-consistent deformation under endpoint-only training. Our analysis links the smoothness energy of the velocity field to low-rank parameters, ofering theoretical support for the structural prior. Experiments on ACDC and 4D-Lung show best performance on structure-oriented metrics with competitive interpolation error.

## References

1. Balakrishnan, G., Zhao, A., Sabuncu, M.R., Guttag, J., Dalca, A.V.: Voxelmorph: a learning framework for deformable medical image registration. IEEE transactions on medical imaging 38(8), 1788–1800 (2019)

2. Beg, M.F., Miller, M.I., Trouvé, A., Younes, L.: Computing large deformation metric mappings via geodesic flows of difeomorphisms. International journal of computer vision 61(2), 139–157 (2005)

3. Bernard, O., Lalande, A., Zotti, C., Cervenansky, F., Yang, X., Heng, P.A., Cetin, I., Lekadir, K., Camara, O., Ballester, M.A.G., et al.: Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: is the problem solved? IEEE transactions on medical imaging 37(11), 2514–2525 (2018)

4. Chen, J., Frey, E.C., He, Y., Segars, W.P., Li, Y., Du, Y.: Transmorph: Transformer for unsupervised medical image registration. Medical image analysis 82, 102615 (2022)

5. Chen, T., Shi, Y., Zheng, Z., Yan, B., Hu, J., Zhu, X.X., Mou, L.: Ultrasound imageto-video synthesis via latent dynamic difusion models. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 764–774. Springer (2024)

6. De Lathauwer, L., De Moor, B., Vandewalle, J.: A multilinear singular value decomposition. SIAM journal on Matrix Analysis and Applications 21(4), 1253–1278 (2000)

7. Guo, Y., Bi, L., Ahn, E., Feng, D., Wang, Q., Kim, J.: A spatiotemporal volumetric interpolation network for 4d dynamic medical image. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4726– 4735 (2020)

8. Hugo, G.D., Weiss, E., Sleeman, W.C., Balik, S., Keall, P.J., Lu, J., Williamson, J.F.: Data from 4d lung imaging of nsclc patients. (No Title) (2016)

9. Huttinga, N.R., Bruijnen, T., Van Den Berg, C.A., Sbrizzi, A.: Real-time nonrigid 3d respiratory motion estimation for mr-guided radiotherapy using mr-motus. IEEE Transactions on Medical Imaging 41(2), 332–346 (2021)

10. Jia, X., Bartlett, J., Chen, W., Song, S., Zhang, T., Cheng, X., Lu, W., Qiu, Z., Duan, J.: Fourier-net: Fast image registration with band-limited deformation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 1015– 1023 (2023)

11. Kim, B., Ye, J.C.: Difusion deformable model for 4d temporal medical image generation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 539–548. Springer (2022)

12. Kim, J., Yoon, H., Park, G., Kim, K., Yang, E.: Data-eficient unsupervised interpolation without any intermediate frame for 4d medical images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11353–11364 (2024)

13. Kolda, T.G., Bader, B.W.: Tensor decompositions and applications. SIAM review 51(3), 455–500 (2009)

14. Li, H., Li, H., Chen, J., Zhong, R., Niu, K., Fu, H., Liu, J.: Aif-sfda: Autonomous information filter driven source-free domain adaptation for medical image segmentation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 4716–4724 (2025)

15. Li, Z., Sun, A., Wei, H., Chen, W., Liu, C., Sun, H., Du, C., Li, R.: Unsupervised 4d-flow mri reconstruction based on partially-independent generative modeling and complex-diference sparsity constraint. Medical Image Analysis p. 103769 (2025)

16. Park, G.Y., Jung, C., Lee, S., Ye, J.C., Lee, S.W.: Self-supervised debiasing using low rank regularization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12395–12405 (2024)

17. Perrin, S., Levilly, S., Mouchère, H., Serfaty, J.M.: Super-resolution and segmentation of 4d flow mri using deep learning and weighted mean frequencies. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 552–561. Springer (2025)

18. Pramanik, A., Aggarwal, H.K., Jacob, M.: Deep generalization of structured lowrank algorithms (deep-slr). IEEE transactions on medical imaging 39(12), 4186– 4197 (2020)

19. Tian, M., Yang, Q., Gao, Y.: Multi-scale multi-task distillation for incremental 3d medical image segmentation. In: European Conference on Computer Vision. pp. 369–384. Springer (2022)

20. Vercauteren, T., Pennec, X., Perchant, A., Ayache, N.: Difeomorphic demons: Eficient non-parametric image registration. NeuroImage 45(1), S61–S72 (2009)

21. Wang, M., Li, C., Vaxman, A.: Canfields: Consolidating difeomorphic flows for non-rigid 4d interpolation from arbitrary-length sequences. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 28587–28598 (2025)

22. Yan, J., Tan, Q., Kawara, S., Zhu, J., Wang, B., Toulemonde, M., Liu, H., Tan, Y., Tang, M.X.: Online 4d ultrasound-guided robotic tracking enables 3d ultrasound localisation microscopy with large tissue displacements. IEEE transactions on medical imaging (2025)

23. You, X., Yang, R., Zhang, C., Jiang, Z., Yang, J., Navab, N.: Fb-dif: Fourier basisguided difusion for temporal interpolation of 4d medical imaging. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 28010–28020 (2025)

24. Zhang, J., Ai, D., Gan, Z., Fu, T., Fan, J., Song, H., Xiao, D., Yang, J.: Temporal modulated multi-scale deformation fusion via knowledge distillation for 4d medical image interpolation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 551–561. Springer (2025)

25. Zhi, S., Wang, Y., Xiao, H., Bai, T., Li, B., Tang, Y., Liu, C., Li, W., Li, T., Ge, H., et al.: Coarse–super-resolution–fine network (cosf-net): a unified end-toend neural network for 4d-mri with simultaneous motion estimation and superresolution. IEEE transactions on medical imaging 43(1), 162–174 (2023)