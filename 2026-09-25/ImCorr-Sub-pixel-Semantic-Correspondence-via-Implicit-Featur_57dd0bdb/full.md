# ImCorr: Sub-pixel Semantic Correspondence via Implicit Feature Decoding

Yusung Choi<sup>1</sup>

Pukyong National University, Busan, Republic of Korea cyscyb@gmail.com

Abstract. The strong performance that modern semantic correspondence methods achieve at standard thresholds plateaus sharply at finegrained thresholds. We argue that this plateau stems not from the representational capacity of backbone features, but from a grid-tied readout. Patch-based vision transformers tokenize images onto discrete grids, introducing two forms of quantization error. On the source side, the nearest patch feature is queried in place of the exact keypoint coordinate. On the target side, no grid feature exists that represents the precise location of the ground truth. We quantify this quantization ceiling across all 499,188 keypoints in SPair-71k, the standard benchmark for semantic correspondence: under the standard 448×448, patch-14 setting, 84.9% of ground-truth keypoints have no grid feature representing their precise location at PCK@0.01. This is a structural limitation at the representation level, independent of the matching strategy. We address this with Im-Corr: Sub-pixel Semantic Correspondence via Implicit Feature Decoding, which formulates correspondence estimation over a continuous feature field queryable at arbitrary continuous coordinates. A FiLMconditioned decoder is trained to embed sub-pixel positional information into the feature field. On the source side, querying the field directly at the exact keypoint coordinate theoretically reduces the representationlevel quantization error to zero; on the target side, decoding the field onto a grid arbitrarily denser than the backbone grid substantially reduces quantization error. On SPair-71k and AP-10K (intra-species, crossspecies, and cross-family), ImCorr improves performance at fine-grained thresholds (PCK@0.01–0.05), achieving a 6.2 percentage point gain over the prior state of the art at PCK@0.01 on SPair-71k. These results experimentally demonstrate that representational continuity is an efective solution for precise semantic correspondence. Code is available at https://github.com/YusungChoi/ImCorr.

Keywords: Semantic Correspondence · Implicit Feature Decoding · Subpixel Localization

## 1 Introduction

There are no grid lines in nature. The tip of a bird’s wing, the edge of a car’s side mirror, the corner of a human eye — the semantic correspondences we seek between images exist in a continuous space that is indiferent to pixel grids. Yet the models that estimate them decompose images into discrete patches, define features only at grid points, and search for answers only on the grid. We argue that this gap is the fundamental bottleneck of precise correspondence.

![](images/2bbbc633a713955e41fac94c9273c774158f7aca8676a303dcb6eb0d03661f94.jpg)  
Fig. 1: Illustration of grid quantization error in semantic correspondence. (a) Sourceside quantization: the same patch feature is used in matching regardless of where within the patch a keypoint lies, so the keypoint’s exact positional information is not reflected in the representation. The distance to the nearest patch center, $\lVert \bf { x } - \hat { \bf { x } } \rVert$ , quantifies the extent of positional information disregarded. (b) Target-side quantization: when the ground-truth location falls between grid points, no grid feature represents its precise location, making correct localization structurally impossible within the PCK threshold.

Semantic correspondence (SC) is the problem of finding semantically matching locations across diferent images, and serves as a foundation for downstream tasks requiring precise spatial reasoning, including pose estimation, scene understanding, and robotic manipulation. Recent methods leveraging large pretrained vision models — most notably DINOv2 [14] and Stable Difusion [16] — as backbones have achieved remarkable gains on standard benchmarks. These advances have been driven by richer semantic representations, more sophisticated cost aggregation, and stronger transformation invariance [17,19,20]. Yet all such methods share a common assumption: features are defined on a grid, and correspondence search is conducted on the same grid.

This assumption is benign at the standard threshold (PCK@0.1), where the permissible error is large relative to the grid spacing. At the fine-grained threshold (PCK@0.01), however, the situation changes fundamentally. The discreteness of the grid itself structurally prevents reaching the ground truth. We term this the quantization ceiling — an upper bound on achievable precision that arises not from inadequate backbone features, but from a grid-tied readout that cannot measure what the backbone knows. The error manifests in two places. On the source side, the feature of the nearest patch center, rather than the exact keypoint coordinate, is fed into matching, imprinting a positional error into the representation before any matching begins. On the target side, no grid feature exists that represents the precise location of the ground truth.

Table 1 quantifies this ceiling across all 499,188 keypoints in SPair-71k [13]. On the source side, the same patch feature is used in matching regardless of where within the patch a keypoint lies, meaning that the keypoint’s exact positional information is not reflected in the representation. For patch sizes 14 and 16, the distance between a keypoint and its nearest patch center reaches mean values of 5.35px and 6.13px, and maximum values of 9.90px and 11.31px, respectively, irrespective of image resolution — quantifying the extent of positional information disregarded by the patch feature. On the target side, under the standard 448 × 448, patch-14 setting, 84.9% of ground-truth keypoints have no grid feature representing their precise location at PCK@0.01; doubling the resolution still leaves 44.1% in this state.

<table><tr><td>Image Size</td><td>Patch Size</td><td>PCK@0.1</td><td>PCK@0.05</td><td>PCK@0.01</td></tr><tr><td rowspan="2">224×224</td><td>14</td><td>2.70%</td><td>27.56%</td><td>96.23%</td></tr><tr><td>16</td><td>5.30%</td><td>36.72%</td><td>97.22%</td></tr><tr><td rowspan="2">448×448</td><td>14</td><td>0.26%</td><td>2.88%</td><td>84.91%</td></tr><tr><td>16</td><td>0.28%</td><td>5.41%</td><td>88.59%</td></tr><tr><td rowspan="2">896×896</td><td>14</td><td>&lt;0.01%</td><td>0.22%</td><td>44.06%</td></tr><tr><td>16</td><td>0.03%</td><td>0.34%</td><td>55.16%</td></tr></table>

Table 1: Proportion of keypoints unreachable within the PCK threshold (relative to target bounding box) when only feature map grid points are considered as candidates (SPair-71k, 499,188 keypoints). This represents a structural ceiling independent of backbone quality.

Increasing resolution or interpolating grid features ofers a possible way to mitigate this problem. However, interpolation only blends existing grid values and does not generate new visual information between grid points.

To address this, we propose ImCorr: Sub-pixel Semantic Correspondence via Implicit Feature Decoding, which reformulates correspondence estimation over a continuous feature field F(x), addressing this limitation at the representation level. A coordinate-conditioned FiLM decoder transforms the within-grid ofset into channel-wise modulation [15], directly learning sub-pixel positional information that interpolation cannot produce. On the source side, the continuous field is queried directly at the exact keypoint coordinate, theoretically eliminating the representation-level quantization error; on the target side, the feature field is decoded onto a grid denser than the backbone grid, substantially reducing quantization error.

Experiments on SPair-71k [13] and AP-10K [18] show that ImCorr achieves consistently superior performance at fine-grained thresholds (PCK@0.05, PCK@0.01). These results experimentally demonstrate that grid quantization — long obscured by the loose thresholds of standard benchmarks — is in fact the bottleneck of precise correspondence, and that representational continuity is its solution. Our contributions are summarized as follows:

– We formalize and empirically demonstrate that the performance plateau at fine-grained thresholds originates from the structural limitations of grid-tied readout, rather than from insuficient backbone representations.

– We propose ImCorr, which formulates semantic correspondence over a continuous feature field. Through a FiLM-conditioned decoder, the source side queries the field at exact keypoint coordinates to theoretically eliminate quantization error, while the target side decodes onto a grid denser than the backbone lattice to substantially reduce the proportion of unreachable candidates.

– ImCorr achieves consistent performance gains at fine-grained thresholds (PCK @0.01, PCK@0.05) over existing methods on both SPair-71k and AP-10K (intra-species, cross-species, and cross-family).

## 2 Related Work

Quantization in Semantic Correspondence. Semantic correspondence has long been a fundamental problem in computer vision. Early methods relied on handcrafted features such as HOG [3] and SIFT [9], before CNN-based representations became the dominant paradigm. More recently, large-scale pretrained vision models — most notably DINOv2 [14] and Stable Difusion [16] — have been shown to provide powerful semantic representations, and methods built upon these backbones have achieved remarkable gains on standard benchmarks. SD-DINO fuses features from both models [20] to achieve strong zero-shot correspondence estimation, while GeoAware-SC enhances geometric awareness [19] to address challenging cases involving symmetric structures. More recently, MARCO combines a coarse-to-fine training objective with self-distillation [2] to improve overall correspondence performance and generalization.

Despite these advances, all such methods share a common blind spot: they focus exclusively on improving the quality of features provided by the backbone, while leaving unquestioned the discreteness of the space in which those features are defined. SD-DINO [20], GeoAware-SC [19], and their contemporaries all extract features and search for correspondences on a patch-tokenized grid, and the fact that this design choice imposes a structural ceiling on precision at fine-grained thresholds has gone largely unnoticed.

A separate line of work attempts to produce continuous predictions from discrete grid features at the matching stage. Window soft-argmax [19], as employed in GeoAware-SC, is one such approach, producing sub-grid predictions by computing a weighted average over neighboring grid features. However, this is a matcher-level operation designed to improve matching quality, and the discreteness of the underlying feature representation remains intact. In contrast, we address quantization not by modifying the matcher, but by operating on a continuous feature representation — one in which the features themselves are defined at arbitrary sub-pixel coordinates.

Implicit Neural Representation. Implicit neural representations (INRs) are a paradigm for representing signals as continuous functions rather than discrete grids, and gained widespread attention in 3D vision through NeRF [12], which represents a 3D scene as a continuous radiance field queryable from arbitrary viewpoints, overcoming the resolution limitations of discrete voxel grids. This idea has since been extended to 2D vision: LIIF represents images as continuous functions [1] from which pixel values can be queried at arbitrary resolutions, breaking free from the constraints of fixed grid resolution in super-resolution tasks.

Within semantic correspondence, NeMF was the first to introduce implicit neural representations to the field [6], representing the 4D matching cost between a source-target image pair as an implicit function. NeMF processes a coarse cost volume as guidance through a cost embedding network, and establishes correspondence through a subsequent fully-connected network. At inference, NeMF iteratively performs PatchMatch-based search and coordinate optimization to produce precise correspondences.

Our method difers fundamentally from NeMF in two respects. First, NeMF implicitly represents the matching cost [6] while leaving the underlying feature representations grid-bound; our method instead defines a continuous per-image feature field, so that features themselves are defined at arbitrary sub-pixel coordinates. Second, NeMF requires iterative coordinate optimization at inference time, whereas our method produces correspondences in a single forward pass. Furthermore, whereas LIIF predicts pixel values for super-resolution [1], our continuous feature field directly targets the quantization bottleneck in semantic correspondence, learning to embed sub-pixel positional information into the feature representation itself.

## 3 Method

## 3.1 Problem Formulation

Given a source image $\mathbf { I } _ { s } \in \mathbb { R } ^ { H \times W \times 3 }$ and a target image $\mathbf { I } _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ , the goal of semantic correspondence is to estimate, for a keypoint coordinate $\textbf { x } \in \mathbb { R } ^ { 2 }$ in the source image, the semantically corresponding coordinate $\mathbf { y } \in \mathbb { R } ^ { 2 }$ in the target image.

Grid-based Matching Formulation. Existing methods extract source and target feature maps $\mathbf { Z } _ { s } , \mathbf { Z } _ { t } \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times C }$ using a backbone encoder. To achieve high-level semantic invariance, the backbone extracts features from deep layers, which reduces spatial resolution: a region of $s \times s$ pixels in the original image is compressed into a single C-dimensional feature vector, yielding a downsampled feature map of size $H ^ { \prime } = H / s , W ^ { \prime } = W / s$ . Representative backbones such as DINOv2 [14] and Stable Difusion [16] employ patch strides of $s \approx 1 4$ or $^ { 1 6 , }$ meaning that a 14 × 14 or 16 × 16 pixel region is represented by a single vector.

The discrete grid $\mathcal { G } = \{ ( i \cdot s , j \cdot s ) | i \in [ 0 , H ^ { \prime } ) , j \in [ 0 , W ^ { \prime } ) \}$ defines both the space in which features are defined and the set of candidate locations $\mathbf { p } \in \mathcal { G }$ for correspondence search. Correspondence estimation is then formulated as similarity maximization over this grid:

$$
\begin{array} { r } { \hat { \mathbf { y } } = \underset { \mathbf { p } \in \mathcal { G } } { \arg \operatorname* { m a x } } \sin ( \mathbf { Z } _ { s } [ \hat { \mathbf { x } } ] , \mathbf { Z } _ { t } [ \mathbf { p } ] ) , } \end{array}\tag{1}
$$

where $\begin{array} { r } { \hat { \mathbf { x } } = \arg \operatorname* { m i n } _ { \mathbf { p } \in \mathcal { G } } \left\| \mathbf { x } - \mathbf { p } \right\| } \end{array}$ is the nearest grid point to the source keypoint $\mathbf { x } ,$ and sim $\operatorname { \mathrm { } } _ { 1 } ( \cdot , \cdot )$ denotes cosine similarity.

This $s \times s$ compression imprints two levels of quantization error into the correspondence pipeline. On the source side, regardless of where keypoint x lies within a grid region, only the feature of the grid center xˆ is fed into matching. The resulting representation error is bounded by:

$$
e _ { \mathrm { s r c } } ( \mathbf { x } ) = \| \mathbf { x } - \hat { \mathbf { x } } \| \leq \frac { s \sqrt { 2 } } { 2 } ,\tag{2}
$$

reaching a maximum of 9.90 px for $s { = } 1 4$ . This error is imprinted into the representation before any matching begins and cannot be removed by any subsequent matching strategy. On the target side, when the ground-truth location y falls between grid points, no candidate exists on the grid that represents its precise location. The set of unreachable keypoints on the target side is defined as:

$$
\mathcal { U } ( \varepsilon ) = \left\{ \mathbf { y } \in \mathbb { R } ^ { 2 } \left| \operatorname* { m i n } _ { \mathbf { p } \in \mathcal { G } } \left\| \mathbf { y } - \mathbf { p } \right\| > \varepsilon \right\} , \right.\tag{3}
$$

where $\varepsilon$ is the PCK threshold. Under the standard 448×448, patch-14 setting, 84.9% of ground-truth keypoints fall into this set at PCK@0.01. This proportion is a structural ceiling already determined at the representation stage.

Reformulation over a Continuous Feature Field. To address both forms of quantization error at the representation level, we define a continuous feature field $F _ { \mathbf { I } } : \mathbb { R } ^ { 2 }  \mathbb { R } ^ { C }$ for each image I that is queryable at arbitrary sub-pixel coordinates. Unlike grid features, which take a fixed value only at each grid point, $F _ { \mathbf { I } }$ is trained to return a value that varies continuously within a grid region, conditioned on the grid feature — that is, querying any position between grid points yields a feature specific to that location. The concrete architecture of $F _ { \mathbf { I } }$ is detailed in Section 3.2. For brevity, we write $F _ { s } : = F _ { \mathbf { I } _ { s } }$ and $F _ { t } : = F _ { \mathbf { I } _ { t } }$ hereafter.

Correspondence estimation is then reformulated over the continuous feature field as:

$$
\hat { \mathbf { y } } = \underset { \mathbf { q } \in \mathcal { G } ^ { \prime } } { \arg \operatorname* { m a x } } \ \mathrm { s i m } ( F _ { s } ( \mathbf { x } ) , F _ { t } ( \mathbf { q } ) ) ,\tag{4}
$$

where $\mathcal { G } ^ { \prime }$ is a search grid r times denser than the backbone grid ${ \mathcal { G } } _ { : }$ satisfying $| \mathcal { G } ^ { \prime } | \gg | \mathcal { G } |$ , and $\mathbf { q } \in \mathcal { G } ^ { \prime }$ denotes a candidate coordinate on $\mathcal { G } ^ { \prime }$ . The distinction between Eq. (1) and Eq. (4) is not merely one of resolution, but reflects a fundamental change in the nature of the representation.

On the source side, $F _ { s }$ is queried directly at the exact keypoint coordinate $\mathbf { x } ,$ rather than at its nearest grid point xˆ. Since the exact coordinate x is passed to $F _ { s }$ without approximation, the representation error of Eq. (2) is theoretically eliminated: $e _ { \mathrm { s r c } } ( \mathbf { x } ) = 0$ . This follows from the fact that $F _ { s }$ precisely computes the ofset between x and xˆ and incorporates it internally, the mechanism of which is detailed in Section 3.2. That is, regardless of where within the grid region the source keypoint lies, $F _ { s } ( \mathbf { x } )$ returns a feature conditioned on its exact location.

On the target side, the continuous field $F _ { t }$ in principle supports infinite resolution, but directly solving for arg max over the continuous domain is computationally intractable and would require iterative coordinate optimization. Instead, we decode $F _ { t }$ onto $\mathcal { G } ^ { \prime }$ via a single forward pass, constructing a finite candidate set while retaining the expressive power of the continuous representation and avoiding iterative inference. Since the density of $\mathcal { G } ^ { \prime }$ can be set independently of the backbone architecture, the size of the unreachable set $\mathcal { U } ( \varepsilon )$ in Eq. (3) can be reduced without additional backbone computation.

Notably, interpolation cannot reduce the source-side quantization error to zero. We overcome this through explicit supervised learning of the continuous feature field: by training it to produce accurate correspondences at arbitrary coordinates, we directly model the per-location semantic variation that grid features alone cannot recover.

## 3.2 Continuous Feature Field

We now present the concrete architecture that realizes the continuous feature field $F _ { \mathbf { I } } : \overline { { \mathbb { R } ^ { 2 } } } \to \mathbb { R } ^ { C }$ introduced conceptually in Section 3.1. We propose a FiLM (Feature-wise Linear Modulation)-conditioned decoder [15]. FiLM is a modulation technique that generates channel-wise scale and shift parameters from a conditioning input to transform a feature representation. This decoder is shared across all images, and takes as input a grid feature and a relative ofset from the grid center to decode the feature at an arbitrary query coordinate.

FiLM-Conditioned Decoder. The grid feature Z is first transformed into a latent code map via a $1 \times 1$ convolution, from which the latent code $\mathbf { z } ^ { * } = \mathbf { Z } [ \hat { \mathbf { x } } ]$ of each grid point is retrieved. The ofset $\boldsymbol { \Delta } = \mathbf { x } - \hat { \mathbf { x } }$ between the query coordinate x and the grid point xˆ is projected into a 2C-dimensional vector via a two-layer MLP $\varphi :$

$$
[ \gamma ; \beta ] = \varphi ( \Delta ) \in \mathbb { R } ^ { 2 C } ,\tag{5}
$$

where $\gamma , \beta \in \mathbb { R } ^ { C }$ denote the channel-wise gain and bias, respectively. The latent code $\mathbf { z } ^ { \ast }$ is then modulated as:

$$
\mathbf { h } = ( 1 + \gamma ( \pmb { \Delta } ) ) \odot \mathbf { z } ^ { * } + \beta ( \pmb { \Delta } ) ,\tag{6}
$$

where ⊙ denotes element-wise multiplication.

The grid feature $\mathbf { z } ^ { \ast }$ encodes the semantic content of an entire patch region into a single vector; however, even within the same patch, the semantic channels that should be emphasized vary depending on the precise query location. For instance, within a patch covering a bird’s wing, the tip and the middle of the wing exhibit distinct semantic characteristics. By generating channel-wise gain $\gamma ( \Delta )$ and bias $\beta ( \Delta )$ from the ofset $\pmb { \Delta }$ and applying them to modulate $\mathbf { z } ^ { \ast }$ , FiLM enables the decoder to produce diferent features from the same grid representation depending on the within-grid query location.

![](images/d885cd8c120d27c8c9e1daea81743f3b2098fc780ae8b7b4421da9d7252f0cb2.jpg)  
Fig. 2: Overview of the FiLM-conditioned decoder and local ensemble. (Left) Zoomedin view of the decoding process for grid point $\mathbf { z } _ { 1 0 } ^ { * } \colon$ the ofset $\Delta = [ w , h ]$ relative to the query coordinate is transformed by an MLP $\varphi$ into channel-wise modulation parameters [γ; β] (Eq. (5)), which modulate the grid feature (Eq. (6)) and are subsequently refined by a nonlinear rendering MLP $\rho$ to produce the per-grid-point feature $f _ { \boldsymbol { \theta } } ( \mathbf { z } ^ { * } , \Delta )$ (Eq. (7)). (Right) The four per-grid-point features surrounding the query coordinate (red star) are combined via area-based weighting, added to $\mathrm { a }$ bilinear residual, and projected through a linear layer $\psi$ to produce the final feature F<sub>I</sub>(x) (Eq. (8)).

The modulated feature h is subsequently refined by a nonlinear rendering MLP $\rho ,$ and the per-grid-point decoder output $f _ { \theta }$ is defined as:

$$
f _ { \theta } ( \mathbf { z } ^ { * } , \pmb { \Delta } ) : = \rho ( \mathbf { h } ) = \mathrm { L i n e a r } \big ( \mathrm { R e L U } ( \mathrm { L i n e a r } ( \mathrm { R e L U } ( \mathbf { h } ) ) ) \big ) ,\tag{7}
$$

where $\rho$ is a width-preserving $( C \to C \to C )$ two-layer network with a preactivation design in which the leading ReLU acts directly on h. This nonlinear rendering stage is central to the expressive power of the decoder, learning complex interactions between the ofset $\pmb { \Delta }$ and the grid feature $\mathbf { z } ^ { \ast }$ that the linear FiLM modulation alone cannot represent.

Local Ensemble and Final Projection. Evaluating $f _ { \theta }$ solely at the single nearest grid point introduces discontinuities at the boundaries between adjacent grid regions — as the query coordinate crosses a grid boundary, $\mathbf { z } ^ { \ast }$ abruptly switches to the feature of a neighboring grid point. Inspired by works that ensure continuity by aggregating contributions from neighboring grid points [1], we adopt a local ensemble. For the four grid points $\{ ( \mathbf { z } _ { t } ^ { * } , \mathbf { v } _ { t } ^ { * } ) \} _ { t \in \{ 0 0 , 0 1 , 1 0 , 1 1 \} }$ surrounding the query coordinate x, $f _ { \theta } ( \mathbf { z } _ { t } ^ { * } , \mathbf { x } - \mathbf { v } _ { t } ^ { * } )$ is computed independently at each, and the results are combined using the area $S _ { t }$ subtended between x and the diagonally opposite grid point of each neighbor as weights. For training stability, a bilinear interpolation of the original grid feature Z at the query coordinate x is added to the ensembled feature as a residual, and the result is finally projected to the output feature dimension via a linear layer $\psi ,$ completing $F _ { \mathbf { I } } ( \mathbf { x } )$

$$
F _ { \mathbf { I } } ( \mathbf { x } ) = \psi \left( \sum _ { t } { \frac { S _ { t } } { S } } \cdot f _ { \theta } ( \mathbf { z } _ { t } ^ { * } , \ \mathbf { x } - \mathbf { v } _ { t } ^ { * } ) + \mathrm { B i l i n e a r } ( \mathbf { Z } , \mathbf { x } ) \right) ,\tag{8}
$$

where $S = \textstyle \sum _ { t } S _ { t }$ and Bilinear $( \mathbf { Z } , \mathbf { x } )$ denotes the bilinear interpolation of the original grid feature Z at x. As x approaches grid point $\mathbf { v } _ { t } ^ { * }$ , the corresponding weight $S _ { t }$ increases, ensuring that the feature field transitions smoothly across grid boundaries. This decoder is applied symmetrically on both sides of correspondence estimation: on the source side, it is queried at the exact keypoint coordinate x to obtain $F _ { s } ( \mathbf { x } ) ;$ on the target side, it is evaluated at every candidate $\mathbf { q } \in \mathcal { G } ^ { \prime }$ to obtain $F _ { t } ( \mathbf { q } )$ . Since the ofset $\boldsymbol { \Delta } = \mathbf { x } - \hat { \mathbf { x } }$ used in Eq. (6) is computed exactly, without approximation, and incorporated into the modulation, $F _ { s } ( \mathbf { x } )$ realizes $e _ { \mathrm { s r c } } ( \mathbf { x } ) = 0$ as claimed in Section 3.1.

## 3.3 Training and Inference

Training. The objective of training is to learn a continuous feature field $F _ { \mathbf { I } }$ that faithfully represents the semantic feature at any queried coordinate, such that it can produce accurate correspondences at arbitrary sub-pixel locations. To this end, the backbone encoder is adapted from its pretrained weights via a lightweight parameter-eficient method [7], and is trained jointly with the FiLMconditioned decoder.

During training, a source-target image pair $\left( \mathbf { I } _ { s } , \mathbf { I } _ { t } \right)$ is provided along with ground-truth keypoint coordinate pairs $\displaystyle ( \mathbf { x } , \mathbf { y } )$ . The resulting features $F _ { s } ( \mathbf { x } )$ and $\{ F _ { t } ( \mathbf { q } ) \} _ { \mathbf { q } \in \mathcal { G } ^ { \prime } }$ yield a predicted correlation distribution $P$ over $\mathcal { G } ^ { \prime }$ via softmax over sim $( F _ { s } ( \bar { \bf x } ) , F _ { t } ( \bf { q } ) )$ , where $( i , j )$ indexes a candidate location on $\mathcal { G } ^ { \prime }$

Following MARCO [2], we supervise the predicted correlation distribution P with a Gaussian soft target $g _ { \sigma }$ centered at the ground-truth location, annealing $\sigma$ from $\sigma _ { \mathrm { m a x } }$ to $\sigma _ { \mathrm { m i n } }$ over training to realize a coarse-to-fine curriculum:

$$
\mathcal { L } _ { C E } = - \sum _ { ( i , j ) } g _ { \sigma } ( i , j ) \log P ( i , j ) .\tag{9}
$$

Inference. At inference time, window soft-argmax [19] is applied within a local window $\mathcal { W } ( \hat { \mathbf { y } } _ { 0 } )$ centered at the initial estimate $\hat { \mathbf { y } } _ { 0 }$ obtained from $\operatorname { E q } .$ . (4), producing the final sub-pixel correspondence:

$$
\hat { \mathbf { y } } = \sum _ { \mathbf { q } \in \mathcal { W } ( \hat { \mathbf { y } } _ { 0 } ) } \mathbf { q } \cdot \mathrm { s o f t m a x } ( \sin ( F _ { s } ( \mathbf { x } ) , F _ { t } ( \mathbf { q } ) ) / \tau _ { \mathrm { i n f } } ) .\tag{10}
$$

This final step further reduces the residual quantization error introduced by the discrete grid $\mathcal { G } ^ { \prime }$ , yielding a continuous correspondence estimate.

## 4 Experiments

## 4.1 Implementation Details

We use pretrained DINOv2-B/14 [14] as the backbone, kept frozen and adapted with LoRA [7], with input images resized so that the feature map is $6 4 \times 6 4 .$ Training uses batch size 4 and Adam [8] with an initial learning rate of $6 \times 1 0 ^ { - 4 }$ the Gaussian soft-target width σ is annealed from 3 to 1. On the target side we use a densification factor of $r = 4 .$ , and at inference window soft-argmax is applied within a window of size 45. Full training details, including the LoRA configuration and training cost, are given in the supplementary material.

## 4.2 Datasets and Evaluation Metric

Datasets. SPair-71k [13] is a large-scale semantic correspondence benchmark comprising 70,958 image pairs across 18 object categories. It evaluates correspondence between diverse object instances within the same category, providing a challenging setting with substantial variations in viewpoint, scale, and appearance. Its large scale and categorical diversity enable a thorough assessment of the generalization ability of correspondence methods.

AP-10K [18] is a large-scale animal pose estimation benchmark comprising 10,015 images spanning 23 animal families and 54 species. For semantic correspondence evaluation, it is organized into three splits of increasing dificulty intra-species, cross-species, and cross-family — each introducing progressively larger appearance and structural variation. These splits enable a rigorous evaluation of generalization ability in a domain distinct from SPair-71k.

Evaluation Metric. We evaluate using Percentage of Correct Keypoints (PCK), the standard metric for semantic correspondence. A predicted keypoint is considered correct if it falls within a radius of α · max(h, w) from the ground-truth keypoint, where h and w denote the height and width of the target object bounding box on both SPair-71k and $\mathrm { A P - 1 0 K \ } \left( \alpha _ { \mathrm { b b o x } } \right)$ . We report results at three thresholds $\alpha \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 \}$ , with particular emphasis on performance at the fine-grained thresholds PCK@0.01 and PCK@0.05.

## 4.3 Comparison with State-of-the-Art

Table 2 presents the quantitative comparison on SPair-71k [13] and AP-10K (intra-species, cross-species, and cross-family) [18]. ImCorr achieves a new state of the art at fine-grained thresholds (PCK@0.01, PCK@0.05) across all four benchmarks. At PCK@0.01, ImCorr improves over the previous best-performing method, MARCO [2], by +6.2%p on SPair-71k, +3.6%p on AP-10K intraspecies, +3.5%p on cross-species, and +3.9%p on cross-family, while also ranking first on every benchmark at PCK@0.05. The margin over the runner-up methods excluding MARCO, namely Geo-SC [19] and Jamais Vu [11], is considerably larger: on SPair-71k at PCK@0.01, ImCorr outperforms Geo-SC by +11.5%p and Jamais Vu by +12.7%p.

Table 2: Quantitative comparison on standard semantic correspondence benchmarks. We report per-image PCK (%, ↑) at multiple thresholds on SPair-71k and three splits of AP-10K (intra-species, cross-species, cross-family). The best and second-best results per column are bolded and underlined, respectively. § uses depth maps at training; † uses object masks at training; ‡ uses object masks at inference.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="3">SPair-71k</td><td colspan="3">AP-10K (I.S.)</td><td colspan="3">AP-10K (C.S.)</td><td colspan="3">AP-10K (C.F.)</td></tr><tr><td>Backbone 0.01</td><td>0.05</td><td>0.10</td><td>0.01</td><td>0.05</td><td>0.10</td><td>0.01</td><td>0.05</td><td>0.10</td><td>0.01</td><td>0.05</td><td>0.10</td></tr><tr><td colspan="10">Unsupervised/Weakly Supervised</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>DINOv2+NN [20]</td><td>ViT-B</td><td>6.3</td><td>38.4</td><td>53.9</td><td>6.4</td><td>41.0</td><td>60.9</td><td>5.3</td><td>37.0</td><td>57.3</td><td>4.4</td><td>29.4</td><td></td><td>47.4</td></tr><tr><td>DIFT [17]</td><td>SD</td><td>7.2</td><td>39.7</td><td>52.9</td><td>6.2</td><td>34.8</td><td>50.3</td><td>5.1</td><td>30.8</td><td>46.0</td><td></td><td>3.7</td><td>22.4</td><td>35.0</td></tr><tr><td>SD+DINÓ [20]</td><td>SD+ViT-B</td><td>7.9</td><td>44.7</td><td>59.9</td><td>7.6</td><td>43.5</td><td>62.9</td><td>6.4</td><td>39.7</td><td></td><td>59.3</td><td>5.2</td><td>30.8</td><td>48.3</td></tr><tr><td>DIY-SC [4]</td><td>SD+ViT-B</td><td>10.1</td><td>53.8</td><td>71.6</td><td></td><td></td><td>70.6</td><td></td><td></td><td>69.8</td><td></td><td></td><td></td><td>57.8</td></tr><tr><td colspan="10">Supervised methods</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>NeMF [6]</td><td>ResNet101</td><td>3.2</td><td>34.2</td><td>53.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DHF [10]</td><td>SD</td><td>8.7</td><td>50.2</td><td>64.9</td><td>8.0</td><td>45.8</td><td>62.7</td><td>6.8</td><td>42.4</td><td>60.0</td><td>5.0</td><td></td><td>32.7</td><td>47.8</td></tr><tr><td>SD+DINO (S) [20]</td><td>SD+ViT-B</td><td>9.6</td><td>57.7</td><td>74.6</td><td>9.9</td><td>57.0</td><td>77.0</td><td>8.8</td><td>53.9</td><td>74.0</td><td>6.9</td><td></td><td>46.2</td><td>65.8</td></tr><tr><td>GECO† [5]</td><td>ViT-B</td><td>14.2</td><td>59.6</td><td>73.6</td><td>19.2</td><td>67.1</td><td>82.5</td><td>17.4</td><td>64.9</td><td>81.2</td><td></td><td>14.5</td><td>60.4</td><td>76.6</td></tr><tr><td>Jamais Vu†§ [11]</td><td>SD+ViT-B</td><td>20.5</td><td>71.9</td><td>82.5</td><td>1</td><td></td><td>=</td><td>1</td><td></td><td></td><td></td><td>=</td><td>1</td><td></td></tr><tr><td>Geo-SC [19]</td><td>SD+ViT-B</td><td>21.7</td><td>72.8</td><td>83.2</td><td>23.2</td><td>73.2</td><td>87.7</td><td>21.7</td><td>70.3</td><td></td><td>85.9</td><td>18.3</td><td>63.2</td><td>78.5</td></tr><tr><td>MARCO† [2]</td><td>ViT-L</td><td>27.0</td><td>77.6</td><td>87.2</td><td>32.6</td><td>77.4</td><td>89.1</td><td>32.2</td><td>76.6</td><td></td><td>88.3</td><td>28.5</td><td>71.1</td><td>83.4</td></tr><tr><td>ImCorr (Òurs)</td><td>ViT-B</td><td>33.2</td><td>78.9</td><td>86.2</td><td>36.2</td><td>78.6</td><td>90.0</td><td>35.7</td><td>77.7</td><td></td><td>89.4</td><td>32.4</td><td>73.6</td><td>82.9</td></tr></table>

At the standard threshold (PCK@0.10), ImCorr also maintains competitive performance with existing state-of-the-art methods across SPair-71k, AP-10K intra-species, cross-species, and cross-family.

Notably, these gains are achieved with fewer resources. ImCorr employs DINOv2-B/14 [14] as its backbone, whereas MARCO relies on the larger DINOv2- L/14. Furthermore, while comparison methods including Geo-SC, MARCO, and Jamais Vu leverage additional supervisory signals such as object masks or depth maps during training or inference, ImCorr relies solely on keypoint annotations, yet surpasses all of them at fine-grained thresholds.

Figure 3 shows qualitative comparisons on SPair-71k under the PCK@0.01 threshold. GeoAware-SC [19] and MARCO [2] capture the approximate location of most keypoints, but under the fine-grained threshold, their predictions deviate slightly from the ground truth, resulting in a substantial number of incorrect matches shown in red. In contrast, ImCorr achieves a higher proportion of correct matches, shown in green, on the same keypoints, and accurately localizes keypoints even in densely clustered regions such as bottle necks and train fronts.

## 4.4 Ablation Study

Source-Side Query and Target-Side Decoding. Table 3 analyzes the individual contributions of source-side continuous query and target-side dense decoding. The baseline corresponds to a purely grid-based matching method that directly uses backbone grid features without the FiLM decoder.

The full model achieves gains of +14.1, +6.0, and +2.9 percentage points over the baseline at PCK@0.01, PCK@0.05, and PCK@0.10, respectively. The monotonically increasing improvement as the threshold decreases suggests that the two components efectively address the precision loss induced by grid quantization. The nature of this improvement, however, difers markedly between the two components.

![](images/10e945fd34affb412e62d87d0294430cec145cd68bda77af195cd69842e481fe.jpg)  
Fig. 3: Qualitative comparison on SPair-71k (PCK@0.01, per-image). Green and red dots indicate correct and incorrect correspondences, respectively. While GeoAware-SC and MARCO capture the approximate location of most keypoints, they tend to deviate slightly from the ground truth under the fine-grained threshold. In contrast, ImCorr matches the same keypoints with substantially higher precision.

Source-side continuous query yields a gain of +6.5%p at PCK@0.01 while also achieving a meaningful improvement of +3.4%p at PCK@0.10. This indicates that querying features at the exact keypoint coordinate improves not only finegrained precision but also the overall quality of the matching representation. In contrast, target-side dense decoding achieves the largest gain at PCK@0.01 (+11.0%p) while contributing only marginally at PCK@0.10 (+0.8%p). This confirms that expanding the resolution of the search space substantially reduces the proportion of unreachable candidates at fine-grained thresholds, while the existing grid resolution is already suficient at standard thresholds.

The two components address quantization error through complementary mechanisms — the source side improves the precision of the feature representation, while the target side expands the resolution of the search space. When combined, the full model achieves PCK@0.01 of 33.2%, exhibiting complementary synergy that exceeds the sum of the individual gains (+6.5%p and +11.0%p), and confirming that the two components are mutually reinforcing rather than redundant.

Table 3: Ablation on source-side query and target-side decoding. We report per-image PCK (%, ↑) on SPair-71k. ✓denotes the component is enabled.
<table><tr><td>Setting</td><td>Src</td><td>Tgt</td><td>PCK@0.01</td><td>PCK@0.05</td><td>PCK@0.10</td></tr><tr><td>Baseline</td><td></td><td></td><td>19.1</td><td>72.9</td><td>83.3</td></tr><tr><td>+ Src Query</td><td>√</td><td></td><td>25.6</td><td>77.2</td><td>86.7</td></tr><tr><td>+ Tgt Decoding</td><td></td><td>√</td><td>30.1</td><td>76.1</td><td>84.1</td></tr><tr><td>Full model (Ours)</td><td>√</td><td>√</td><td>33.2</td><td>78.9</td><td>86.2</td></tr><tr><td>Gain over baseline</td><td></td><td></td><td>+14.1</td><td>+6.0</td><td>+2.9</td></tr></table>

Densification Factor r. Table 4 analyzes the efect of the densification factor r of the dense search grid $\mathcal { G } ^ { \prime }$ . As r increases from 2 to 4, consistent improvements are observed at fine-grained thresholds: PCK@0.01 increases from 30.3 to 33.2, and PCK@0.05 from 76.2 to 78.9. This is attributable to the fact that a denser grid increases the likelihood that ground-truth locations previously unreachable within the candidate set become accessible. Beyond $r = 4$ , however, PCK@0.01 decreases monotonically to 32.9 and 31.7 at $r = 6$ and $r = 8 ,$ respectively. We interpret this as a consequence of excessive densification, where an abundance of candidates with similar features degrades the discriminability of the matching.

At PCK@0.10, performance decreases monotonically from 86.9 to 85.1 as r increases, a pattern that is consistent with the findings of Table 3. Just as targetside dense decoding contributed only marginally to PCK@0.10 $( + 0 . 8 \% \mathrm { p } )$ in the component ablation, increasing r similarly fails to improve standard-threshold performance and instead induces a slight degradation. This consistently suggests that the backbone grid resolution is already suficient at standard thresholds, and that the primary efect of densification lies in expanding the search space at fine-grained thresholds.

Table 4: Ablation on the densification factor r of the dense search grid $\mathcal { G } ^ { \prime }$ . We report per-image PCK (%, ↑) on SPair-71k.
<table><tr><td>r</td><td>PCK@0.01</td><td>PCK@0.05</td><td>PCK@0.10</td></tr><tr><td>2</td><td>30.3</td><td>76.2</td><td>86.9</td></tr><tr><td>4</td><td>33.2</td><td>78.9</td><td>86.2</td></tr><tr><td>6</td><td>32.9</td><td>78.0</td><td>85.6</td></tr><tr><td>8</td><td>31.7</td><td>77.1</td><td>85.1</td></tr></table>

Inference Resolution. To ensure a fair comparison, we additionally evaluate ImCorr at an input resolution of 770, matching the inference resolution of MARCO [2], which holds the previous state of the art at fine-grained thresholds. Despite using a backbone that is 3.5× smaller than that of MARCO (DINOv2-B, 86M vs. DINOv2-L, 303M parameters) [14], ImCorr achieves 32.1% at PCK@0.01 on SPair-71k under this setting, outperforming MARCO (27.0%) by 5.1%p. This confirms that the performance gains stem from the structural design of the continuous feature field, rather than from diferences in input resolution or backbone capacity.

Interpolation Baseline. To directly validate our claim that a continuous field cannot be substituted by interpolation, we additionally evaluate a variant of the baseline in Table 3 (a purely grid-based method that directly uses backbone grid features without the FiLM decoder) equipped with bilinear interpolation on both the source and target sides. This variant achieves 20.9% at PCK@0.01 on SPair-71k, a modest improvement over the baseline (19.1%) but still substantially below our full model (33.2%). This confirms that interpolation merely blends existing grid values without generating new position-specific semantic information, whereas the learned FiLM decoder directly learns to produce it.

## 5 Limitation and Future Work

While ImCorr enables feature queries at arbitrary sub-pixel coordinates, several limitations remain. The target-side quantization error is only substantially reduced through densification rather than eliminated, since G<sup>′</sup> remains a finite discrete grid.

More fundamentally, our decoder conditions on the relative ofset of a query point through a single FiLM modulation, treating every location within a cell as a flat, axis-aligned coordinate. It therefore cannot represent where a query lies within a patch hierarchically, from coarse to fine, and its axis-aligned parameterization is misaligned with the orientation-agnostic, radius-based criterion that PCK employs. As a result, precise localization within a single patch remains an open problem, and accuracy at the strictest threshold is still far below that at the standard threshold. Encoding within-cell position in a more structured, geometry-aware manner is a promising direction for future work.

## 6 Conclusion

There are no grid lines in nature, yet the models that estimate semantic correspondence have consistently searched for answers only on a grid. This paper has shown that this gap is the fundamental bottleneck of precise correspondence, and has proposed ImCorr, which directly resolves this at the representation level by defining correspondence over a continuous feature field rather than a discrete grid. Experiments on SPair-71k and AP-10K demonstrate that ImCorr achieves consistent state-of-the-art performance at fine-grained thresholds (PCK@0.01, PCK@0.05). These results suggest that the bottleneck of precise correspondence has long been obscured behind the loose thresholds of standard benchmarks, and that representational continuity is its efective solution.

## References

1. Chen, Y., Liu, S., Wang, X.: Learning continuous image representation with local implicit image function. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8628–8638 (2021)

2. Cuttano, C., Trivigno, G., Masone, C., Roth, S.: Marco: Navigating the unseen space of semantic correspondence. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21649–21658 (2026)

3. Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05). vol. 1, pp. 886–893. Ieee (2005)

4. Dünkel, O., Wimmer, T., Theobalt, C., Rupprecht, C., Kortylewski, A.: Do it yourself: Learning semantic correspondence from pseudo-labels. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5834–5844 (2025)

5. Hartwig, R., Muhle, D., Marin, R., Cremers, D.: Geco: Geometrically consistent embedding with lightspeed inference. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9309–9319 (2025)

6. Hong, S., Nam, J., Cho, S., Hong, S., Jeon, S., Min, D., Kim, S.: Neural matching fields: Implicit representation of matching fields for visual correspondence. Advances in Neural Information Processing Systems 35, 13512–13526 (2022)

7. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)

8. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

9. Lowe, D.G.: Distinctive image features from scale-invariant keypoints. International journal of computer vision 60(2), 91–110 (2004)

10. Luo, G., Dunlap, L., Park, D.H., Holynski, A., Darrell, T.: Difusion hyperfeatures: Searching through time and space for semantic correspondence. Advances in Neural Information Processing Systems 36, 47500–47510 (2023)

11. Mariotti, O., Du, Z., Bhalgat, Y., Mac Aodha, O., Bilen, H.: Jamais vu: Exposing the generalization gap in supervised semantic correspondence. arXiv preprint arXiv:2506.08220 (2025)

12. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021)

13. Min, J., Lee, J., Ponce, J., Cho, M.: Spair-71k: A large-scale benchmark for semantic correspondence. arXiv preprint arXiv:1908.10543 (2019)

14. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

15. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

16. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

17. Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspondence from image difusion. Advances in neural information processing systems 36, 1363– 1389 (2023)

18. Yu, H., Xu, Y., Zhang, J., Zhao, W., Guan, Z., Tao, D.: Ap-10k: A benchmark for animal pose estimation in the wild. arXiv preprint arXiv:2108.12617 (2021)

19. Zhang, J., Herrmann, C., Hur, J., Chen, E., Jampani, V., Sun, D., Yang, M.H.: Telling left from right: Identifying geometry-aware semantic correspondence. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3076–3085 (2024)

20. Zhang, J., Herrmann, C., Hur, J., Polania Cabrera, L., Jampani, V., Sun, D., Yang, M.H.: A tale of two features: Stable difusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems 36, 45533–45547 (2023)

## A Implementation Details

Backbone and adaptation. We use pretrained DINOv2-B/14 [14] as the backbone encoder. Input images are resized such that the resulting feature map has a spatial resolution of $6 4 \times 6 4$ (an input resolution of 896 × 896). The backbone weights are kept frozen and adapted with LoRA [7] (rank 16, α = 1, i.e., scaling $\alpha / \mathrm { r a n k } = 1 / 1 6 )$ , applied to the second MLP projection (fc2) of the last six transformer blocks (blocks 6–11, 0-indexed).

Decoder. The ofset MLP φ has a hidden dimension of 64. The rendering MLP $\rho$ is width-preserving $( C \to C \to C )$ with a pre-activation design, as described in Sec. 3.2.

Training. Training uses a batch size of 4 and the Adam optimizer [8] with an initial learning rate of $6 \times 1 0 ^ { - 4 }$ and a StepLR scheduler. The standard deviation σ of the Gaussian soft target is annealed from $\sigma _ { \mathrm { m a x } } = 3 \mathrm { \ t o \ } \sigma _ { \mathrm { m i n } } = 1$ over the course of training. Training runs for 5 epochs on a single NVIDIA A100 SXM, taking approximately 30 hours.

Inference. On the target side, a densification factor of $r = 4$ is applied to the dense search grid $\mathcal { G } ^ { \prime }$ , yielding a 256×256 candidate lattice. Window soft-argmax is applied within a local window of size 45 to produce the final sub-pixel correspondence.

## B Additional Qualitative Results

We present additional qualitative results on SPair-71k at PCK@0.01 across a broader range of object categories (green: correct, red: incorrect). ImCorr consistently produces higher-precision correspondences than existing methods, even in densely clustered keypoint regions.

Ours

GeoAware-SC  
MARCO  
![](images/4d8337046bf4cfa57a4c024bfd1687ae0d1091f115c2d283dbb82a32b827f5da.jpg)  
Fig. 4: Additional results on SPair-71k across object categories.

GeoAware-SC  
MARCO  
Ours  
![](images/944929f2b72ee2248b5bac38a5e882f66a0a91834c6f72fc7bcc0f41ebddee07.jpg)  
Fig. 5: Additional results on SPair-71k across object categories.

GeoAware-SC  
MARCO  
Ours  
![](images/c20710091d66d3c1506d2f198ba53f4e28c87eddba1852c428544cc75ebc7a03.jpg)  
Fig. 6: Additional results on SPair-71k across object categories.