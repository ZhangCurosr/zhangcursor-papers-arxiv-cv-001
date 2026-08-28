# Beyond Classification: Task-Dependent Learnability under Privacy-Motivated Image Transformations

Leon Ranke<sup>1,2</sup> , Wolfgang Hübner<sup>1</sup> , Ronny Hug<sup>1</sup> , Michael Arens<sup>1</sup> , and Jürgen Beyerer<sup>1,2</sup>

<sup>1</sup> Fraunhofer Institute of Optronics, System Technologies and Image Exploitation (IOSB), Gutleuthausstraße 1, 76275 Ettlingen, Germany 2 Karlsruhe Institute of Technology (KIT) firstname.lastname@iosb.fraunhofer.de https://www.iosb.fraunhofer.de

Abstract. Privacy-Enhancing Technologies (PETs) in computer vision often rely on noise or image perturbations to protect visual data while securely processing it, creating a trade-of between task performance and protection. This trade-of is commonly evaluated using image classification, which primarily captures semantic separability and remains robust despite significant geometric, spatial layout or local boundary alterations. As a result, it is too simplistic as a proxy for generic vision tasks. Exhaustive downstream-task evaluation, however, is computationally expensive because models must often be trained for each PET transformation and parameter setting. We therefore propose a compute-aware multi-task protocol for evaluating PETs in model training. It combines lightweight proxy tasks that target complementary aspects of visual structure while remaining simple and fast to compute. Across irreversible privacy transformations, key-based block primitives, and learnable image encryption schemes, we demonstrate that PETs with similar classification accuracy can difer substantially on other tasks. The outcomes highlight the need for PET evaluation protocols that move beyond classification-only reporting. Code is available at: https://github.com/LeonRanke/Task-Dependent-Learnability.

Keywords: Privacy-Preserving Machine Learning · Utility Evaluation

## 1 Introduction

Privacy-Preserving Machine Learning (PPML), as applied to computer vision tasks, aims to reduce the exposure of sensitive visual content while retaining suficient information for efective learning. For representation-altering PETs (Privacy-Enhancing Technologies), such as learnable image encryption [17,34,48] and related image-domain transformations [18, 25, 47], this creates a fundamental tension: the transformation should obscure human-interpretable information, while the resulting representation must still support downstream learning.

Existing evaluations of representation-altering PETs commonly report classification accuracy as the main measure of utility [18,25,34,48]. This provides a useful but narrow view: classification primarily tests whether class-discriminative statistics remain separable under the transformed data distribution. It does not necessarily probe whether geometric relations, spatial layout, boundary compatibility, or orientation-dependent structures are preserved [51]. Indeed, models can maintain high classification accuracy even when spatial structure and highfrequency components are substantially degraded [2,10,16]. Transformed images may therefore support classification while being poorly suited for tasks such as detection, segmentation, tracking, pose estimation, or geometric matching.

A natural solution is to evaluate PETs on a broad suite of downstream tasks, but in practice this is often prohibitively expensive. Because transformations alter low-level statistics and spatial structure, pre-training on natural images cannot be assumed to generalise across parametrisations, including encryption keys, design parameters, and scale. Rigorous evaluation would therefore require training from scratch for each transformation family and parameter setting. This motivates a compute-aware intermediate evaluation stage: before investing in expensive downstream pipelines, lightweight diagnostic proxy tasks (e.g. relative angle prediction or jigsaw puzzle-solving) can probe which broad types of visual structure remain learnable.

In this paper, learnability is used in an operational, empirical sense. Given a downstream task, a transformation $T _ { \lambda , \kappa } , ^ { 3 }$ a model, and a training protocol, learnability is estimated as the task performance achieved by a model trained from scratch on transformed samples $\widetilde { x } = T _ { \lambda , \kappa } ( x )$ . Learnability is thus not an intrinsic property of the transformation alone, instead it characterises what information a particular learning algorithm can access under finite data and computational resources. In this light we distinguish obfuscation from encryption: the former irreversibly alters an image, whereas the latter is invertible given the key κ (security is not part of this definition). Information may be preserved in principle yet remain inaccessible, or dificult to exploit, to a model that receives no key-dependent information.

Our results show that learnability under privacy-motivated image transformations is strongly task-dependent. Transformations that preserve comparatively high classification accuracy can severely impair tasks that rely on geometric consistency and spatial order. Conversely, key-based transformations can introduce deterministic shortcut cues that some tasks exploit, resulting in high performance, while performance on other tasks collapses. By relating these taskspecific behaviours to image-domain obfuscation and leakage metrics, we find that no single utility or obfuscation metric fully characterises a transformation. Together, these findings motivate evaluation protocols for privacy-preserving vision that go beyond classification performance, jointly assessing task-dependent learnability, shortcut leakage, and image-domain obfuscation.

In summary, our main contributions are:

1. A compute-aware multi-task protocol for evaluating task-dependent learnability under privacy-motivated image transformations.

2. An analysis of a controlled set of multi-scale image transformations, spanning obfuscation, block-based transforms and Learnable Image Encryption schemes [25, 34, 48], showing how scale and key structure afect learnability.

3. Evidence that classification accuracy, proxy-task utility, and image-domain obfuscation metrics capture distinct and individually incomplete aspects of representation-altering PETs in PPML.

## 2 Related Work

Research on PPML for computer vision has primarily focused on secure computation mechanisms [13, 19, 20, 29, 33], privacy-preserving training protocols [7, 42, 50, 56], and data transformations [17, 18, 25, 34, 48]. Comparatively little attention has been devoted to systematic evaluation methodologies that jointly assess image-domain obfuscation and downstream task utility.

## 2.1 Utility Evaluation in PPML

The notion of model utility varies across PETs. Here, utility refers to how well a protected training pipeline preserves the predictive performance of an equivalent plaintext pipeline. A useful distinction can be drawn between (i) functionalitypreserving approaches, which aim to reproduce plaintext computations in protected domains; (ii) training-oriented PETs, modifying the optimisation process; and (iii) representation-altering PETs, that transform images before training.

Functionality-preserving PETs include Fully Homomorphic Encryption [11], Secure Multiparty Computation [14] and Trusted Execution Environments [41]. These approaches execute computations in protected domains while reproducing the behaviour of the corresponding plaintext computation. Utility degradation therefore arises mainly from practical constraints, including reduced numerical precision, approximated non-linear functions, and restricted model architectures. Image classification on benchmarks such as MNIST and CIFAR-10 remains feasible [13,20,24,39,44,45,53,54], although often at substantial computational and communication overhead, which can hinder scalability.

Training-oriented PETs include Diferential Privacy (DP) [8] and Federated Learning (FL) [36]. These approaches retain the original input representation while modifying the optimisation or collaboration process. DP-based training introduces a direct privacy–utility trade-of through mechanisms such as gradient clipping and adding calibrated noise [3,42], whereas FL distributes optimisation across multiple data holders, where utility can degrade through noisy updates, heterogeneous client data, or partial client participation [50, 56].

Representation-altering PETs include learnable image encryption [17, 34, 48] and related image-domain transformations [18, 25, 47]. These approaches transform images before training and require models to learn directly from the transformed representation. Unlike functionality-preserving PETs, they do not generally preserve arbitrary plaintext computations. Their utility, therefore, depends on whether the transformation retains suficient task-relevant information, and it has predominantly been evaluated through downstream classification performance relative to training on plaintext images (see Tab. 1).

Table 1: Datasets used to evaluate model utility in representation-altering PETs.
<table><tr><td>Representation-altering scheme</td><td>Dataset(s) used to evaluate utility</td></tr><tr><td>Learnable Image Encryption [48]</td><td>CIFAR-10 [23]</td></tr><tr><td>Pixel-Based Image Encryption [47]</td><td>CIFAR-10 [23], STL-10 [5]</td></tr><tr><td>Block-wise Scrambled Image Recognition [34]</td><td>CIFAR-10, CIFAR-100 [23]</td></tr><tr><td>InstaHide [18]</td><td>CIFAR-10, CIFAR-100 [23], MNIST [28], ImageNet [6] (subset)</td></tr><tr><td>Learnable Image Encryption for Vision Transformers [17]</td><td>CIFAR-10 [23], Tiny-ImageNet [27]</td></tr></table>

This dominance of classification benchmarks creates an evaluation gap for representation-altering PETs. Classification accuracy indicates whether semantic labels remain predictable, but provides limited evidence about other structural properties of the transformed representation. In particular, it does not directly assess whether global orientation, spatial compatibility, local boundary continuity, or position-dependent cues are preserved [51]. These properties matter for many downstream vision tasks, yet they are rarely examined because training complete downstream models in transformed domains is computationally expensive. We address this gap by introducing diagnostic proxy tasks that probe complementary structural requirements while remaining inexpensive enough to evaluate across multiple parameter and key settings. The tasks are adapted from self-supervised objectives [12] but serve here as diagnostics of structural learnability rather than pre-training objectives.

## 2.2 Image Security Metrics

In contrast to utility evaluation, the image encryption literature provides extensive statistical security evaluation suites [35,43]. Commonly reported metrics quantify image-domain properties associated with leakage or obfuscation, including perceptual leakage between plain and transformed images (e.g., MSE, PSNR, SSIM [49], LPIPS [52], and MI), encrypted-domain statistics (e.g., entropy, adjacent-pixel correlation, spectral flatness, divergence from uniformity), and diferential or key-sensitivity properties. Table 2 summarises these metrics.

We treat these quantities as indicators of image-domain obfuscation and leakage rather than as formal privacy guarantees. They measure perceptual, statistical, or diferential properties of transformed images, but, by themselves, they do not establish resistance to de-obfuscation, reconstruction, recognition, attribute inference, or membership inference attacks [9, 37, 38, 46, 55]. Nevertheless, they are commonly used in image-encryption evaluations [35, 43] and provide useful complementary information about how strongly a transformation alters the visible image domain.

Table 2: Image-domain obfuscation and leakage indicators for RGB images. Metrics are computed per channel (R, G, B) and averaged. Intensities are in $\{ 0 , \ldots , L - 1 \}$ , with L = 256 throughout.
<table><tr><td>Name Abbreviation</td><td>Range</td><td>Heuristic target</td></tr><tr><td>Perceptual Leakage</td><td></td><td> $\scriptstyle { \frac { L ^ { 2 } - 1 } { c } }$ </td></tr><tr><td>Mean Squared Error MSE Peak Signal-to-Noise Ratio (dB) PSNR Structural Similarity Index Measure [49] SSIM Learned Perceptual Image Patch Similarity [52] LPIPS</td><td> $[ 0 , ( L - 1 ) ^ { 2 } ]$  [0,∞) [−1,1] [0,∞)</td><td> $1 0 \log _ { 1 0 } \frac { 6 ( L - 1 ) } { L + 1 }$  0 ↑</td></tr><tr><td>Mutual Information (bits) MI Encrypted Image Statistics</td><td>[0, log2(L)]</td><td>0</td></tr><tr><td>Chi-Squared Goodness-of-Fit Test  $\chi ^ { 2 }$  Kullback-Leibler Divergence  $D _ { \mathrm { K L } }$  Shannon Information Entropy (bits) IE</td><td>[0,∞) [0,∞) [0, log2(L)]</td><td> $\leq \chi _ { 0 . 0 5 } ^ { 2 } ( L - 1 )$  0 log2(L)</td></tr><tr><td>2D Information Entropy [26] (bits) 2D-IE Adjacent-pixel correlation CC</td><td> $[ 0 , \log _ { 2 } ( 2 L - 1 ) ]$  [−1,1]</td><td> $\log _ { 2 } ( 2 L - 1 )$  0</td></tr><tr><td>Spectral Flatness Measure SFM</td><td>[0, 1]</td><td>1</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Differential Analysis</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Number of Pixels Change Rate (w.r.t. T) NPCRx</td><td>[0, 1]</td><td></td></tr><tr><td></td><td></td><td> $\frac { 1 - \frac { 1 } { L } } { \frac { L + 1 } { 3 L } }$ </td></tr><tr><td>Unified Average Changing Intensity (w.r.t. T) UACIx</td><td>[0, 1]</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Key Analysis</td><td></td><td></td></tr><tr><td></td><td></td><td> $\geq 2 ^ { 1 2 8 }$ </td></tr><tr><td>Key-space size |C|</td><td>[1,∞)</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Number of Pixels Change Rate (w.r.t. κ) NPCRκ Unified Average Changing Intensity (w.r.t. κ) UACIκ</td><td>[0, 1] [0, 1]</td><td> $\begin{array} { c } { { 1 - { \frac { 1 } { L } } } } \\ { { { \frac { L + 1 } { 3 L } } } } \end{array}$ </td></tr></table>

## 3 Methodology

The objective of this paper is to characterise how image-domain transformations afect the learnability of diferent visual structures. To this end, transformations are viewed as operators that remove, rearrange, or statistically alter components of the visual signal. Downstream proxy tasks then act as diagnostic probes that test whether these components remain accessible to the learning algorithm.

Transformations are defined as parametrisable operators $T _ { \lambda , \kappa } : \mathcal { X } \to \bar { \mathcal { X } }$ acting on the plain image space $\mathcal { X } \subset \mathbb { R } ^ { H \times \mathbf { \bar { W } } \times 3 }$ , where λ denotes the scale parameter and $\kappa \in \kappa$ a secret key. For non-keyed obfuscations, $\kappa = \varnothing$ . Each transformation induces a modified data distribution with altered spatial, geometric, perceptual, or statistical properties.

## 3.1 Transformation Operators

We evaluate controlled irreversible transformations, key-based block primitives, and learnable image encryption schemes [25,34,48]. The first group acts as a diagnostic baseline that removes specific visual structures in a controlled manner, while the remaining two groups represent transformations used in learnable image encryption. Together, they enable a systematic analysis of which structural

![](images/db3f40a3567cc36d601eaa8cb411e2a0e755fb175d2cedc72cf22bb9b2c5c0c8.jpg)  
Fig. 1: Qualitative examples of transformation families across increasing scale.

components remain learnable under diferent forms of image-domain alteration.   
Figure 1 provides qualitative examples of the considered transforms.

Gaussian Blurring acts as a low-pass filter, removing high-frequency components while preserving spatial topology and global intensity relationships. For scale parameter σ, images are obtained via separable convolution with a Gaussian kernel of standard deviation σ and kernel size $k = 2 \lfloor 3 \sigma \rfloor + 1 \ \lceil 3 0 \rfloor$ . Increasing σ progressively suppresses fine-scale details, isolating the contribution of spatialfrequency bands to downstream task performance.

Locally Orderless Images (LOIs) discard spatial ordering within nonoverlapping B  B windows while approximately preserving marginal intensity distributions inside those windows [22]. As B increases, local correlations are removed at progressively larger scales, while global intensity statistics are retained. LOIs, therefore, provide a controlled mechanism for analysing the impact of spatial order at diferent scales on learnability.

Block-Based Transformations are invertible operations used in the considered learnable image encryption schemes [25, 34, 48]. Images are partitioned into non-overlapping B B blocks, followed by deterministic, key-based operations applied within or across blocks. Depending on the primitive, these operations either preserve local block content while disrupting global arrangement, or disrupt local adjacency while preserving block-level placement. The analysed primitives are: permutation of spatial block positions (Block Shufling); block rotations by multiples of 90<sup>◦</sup> combined with horizontal and vertical flips (Block Rotation + Flipping); permutation of pixel positions within blocks (Pixel Shufling); permutation of the RGB channels per block (Colour Shufling); and pixel-wise intensity inversion (Negative-Positive (NP) Transformation).

Learnable Image Encryption Schemes combine multiple block-based primitives into structured pipelines. These schemes are invertible given the secret key $\kappa ,$ but can induce transformed-domain regularities that are dificult to exploit without access to the secret key. We analyse: Tanaka [48] (Pixel Shufling and NP Transformation); Extended Tanaka (E-Tanaka) [34] (Pixel Shufling, NP Transformation, Block Shufling); and the Encryption then Compression (EtC) [25] scheme (Block Rotation + Flipping, Colour Shufling, Block Shufling).

Gaussian Blurring uses σ as the scale parameter λ, whereas all other transformations use the block size B. The distinction between obfuscations and keyed transformations matters here: Gaussian blur and LOIs remove information from the image even when their parameters are known, whereas block-based transformations and learnable image encryption schemes preserve information. A learner receiving only transformed images may nonetheless be unable to exploit this information efectively, because the transformation can misalign with the model’s inductive biases. Some PETs address this restriction by implicitly providing the model with key-dependent information (e.g., E-Tanaka [34] provides the model with information related to the block permutation). Moreover, when a fixed key is used across training and testing, deterministic key-dependent regularities may become learnable and act as shortcut cues.<sup>4</sup> Our protocol therefore measures fixed-key empirical learnability under a specified training setup, rather than intrinsic information preservation or formal privacy.

## 3.2 Evaluation Tasks

The goal is not to approximate the performance of every possible downstream task. Instead, a tractable protocol is introduced that can be repeated across many transformations, scales, and keys to identify which broad types of visual structure remain learnable. The selected tasks act as diagnostic probes: they are inexpensive relative to full detection or segmentation pipelines and require diferent structural properties of the transformed image distribution. Rather than serving as independent benchmarks, they function as a combined test suite for structural suficiency under transformation.

Table 3: Overview of investigated proxy tasks, including approximate training time.
<table><tr><td>Task</td><td>Probes</td><td>Approx. Time</td></tr><tr><td>Classification</td><td>semantic separability</td><td>≈ 1.4 h</td></tr><tr><td>Angle Prediction</td><td>global orientation / geometric consistency</td><td>≈ 0.8 h</td></tr><tr><td></td><td>Jigsaw Puzzle Solving spatial compatibility / local structure</td><td>≈ 9.4 h</td></tr></table>

Image Classification predicts a discrete class label $y \in \{ 0 , \ldots , N - 1 \}$ given transformed images $\widetilde { x } = T _ { \lambda , \kappa } ( x )$ . We use a ResNet-34 [15] with an N-class classification head. Since classification provides a sparse supervision signal, a wide range of input variations can be tolerated as long as class-discriminative statistics remain separable. Classification is included because it is the dominant task for evaluating the utility of representation-altering PETs.<sup>5</sup>

Angle Prediction [12] estimates the relative rotation $\varDelta \alpha = \alpha _ { 2 } - \alpha _ { 1 }$ between two independently rotated and transformed views of the same image:

$$
\begin{array} { r } { \widetilde { x } _ { 1 } = T _ { \lambda , \kappa } ( R _ { \alpha _ { 1 } } \cdot x ) , \quad \widetilde { x } _ { 2 } = T _ { \lambda , \kappa } ( R _ { \alpha _ { 2 } } \cdot x ) . } \end{array}\tag{1}
$$

The angles $\alpha _ { 1 } , \alpha _ { 2 }$ are sampled randomly from a uniform distribution, with transformations applied after rotation. A CNN processes the two transformed images concatenated along the channel dimension. A regression head predicts $( \sin ( \varDelta \alpha ) , \cos ( \varDelta \alpha ) )$ to avoid discontinuities in angular regression. This task differs from classification in two ways. First, it requires consistent extraction of orientation-dependent structure from two transformed views of the same image. Second, the target is continuous, making the task more sensitive to gradual degradation of geometric information.

Jigsaw Puzzle Solving [40] probes spatial compatibility and local-to-global structural consistency. Given an image split into a regular k  k grid, the pieces are permuted according to an unknown permutation π. Following the transformer based JPDVT [31], the model recovers π by assigning shufled pieces to their original grid positions. Varying k changes the spatial scale of the task: larger values reduce the context available per piece and increase the solution space.

Jigsaw puzzle solving tests whether spatial relationships remain learnable after transformation. Downstream tasks such as detection, segmentation, pose estimation, and tracking depend not only on semantic separability but also on spatial layout, boundary compatibility, and consistent relationships between neighbouring regions. Jigsaw solving acts as a tractable intermediate probe for spatially exploitable information. Importantly, high jigsaw accuracy does not necessarily imply preservation of natural spatial structure. Under fixed-key transformations, performance may also reflect deterministic cues induced by the transformations, such as position-dependent block patterns or key-specific regularities. Table 3 summarises the proxy tasks investigated.

## 4 Experimental Setup

All models are trained from scratch exclusively on transformed data $\widetilde { \chi } .$ They do not receive paired plaintext images, secret keys, or transformation-specific adaptation modules. This enables comparing transformations within a common learning setup and identifying which types of structure remain accessible to standard learners under limited computational resources and data volumes.

A predefined set of scale parameters λ is applied consistently during both training and testing across all transformation families. For key-based transformations, a fixed-key setting is evaluated: one secret key κ is sampled per configuration and used for both training and testing. This aligns with common learnable image encryption evaluations, in which models are trained directly in a static transformed domain. However, it also means that models may exploit deterministic key-dependent regularities. Parameter choices for each experiment are reported in the supplementary material. Training is performed for a predefined maximum number of epochs with early stopping, providing an upper bound on optimisation cost while mitigating overfitting.

For classification, we use the Animal Species Classification dataset [1], which has $N = 1 5$ classes, with images resized to $4 1 6 \times 4 1 6$ . The dataset provides suficient spatial resolution to meaningfully analyse scale-dependent transformations while remaining computationally tractable. Performance is evaluated via top-1 accuracy with a chance level of $1 / N$

Angle prediction is evaluated on the same dataset. Images are rotated and then cropped to exclude hard image edges that leak rotational information. Performance is reported as the mean angular error in degrees. Since the angular error is measured as the shortest circular distance between the predicted and ground-truth relative angle, errors lie in [0<sup>◦</sup>, 180<sup>◦</sup>]. For uniformly distributed rotations, the expected error of a random or constant predictor is therefore $9 0 °$

In the puzzle-solving experiments, images from the Animal Species Classification dataset are resized to $3 8 4 \times 3 8 4$ pixels and divided into a regular k k grid, where $k \in \{ 4 , 6 , 8 \}$ denotes the puzzle size. For each puzzle size, the transformation block size B is chosen relative to the side length $p = 3 8 4 / k$ of one puzzle piece. We consider settings in which each puzzle piece contains $r \times r$ blocks, with $r \in \{ 0 . 5 , 1 , 2 , 4 , 8 \}$ . The considered configurations and resulting values are shown in Fig. 2. This construction controls the relationship between the transformation and puzzle scales, avoiding misalignment between transformation blocks and piece boundaries. Performance is measured as the percentage of correctly placed puzzle pieces; the chance level of a $k \times k$ puzzle is $1 / k ^ { 2 }$

## 5 Results & Discussion

This section analyses how the considered transformations afect task-dependent learnability. We first report classification performance, which aligns with the dominant utility view in prior work, and then compare it with relative-angle prediction and jigsaw puzzle solving. Finally, we relate task performance to image-domain obfuscation metrics.

![](images/8b9ad7aff7bf4e494a506ea36e03953af38ac43bada208c6f471e495126d3a24.jpg)  
(a) Illustration for k = 4. The red block shows the 0.5×0.5 setting, while the blue block shows the 2 × 2 setting.  
Fig. 2: Relationship between the puzzle size k and the transformation block size B.

(b) Resulting transformation block size B, in pixels, for each configuration.
<table><tr><td colspan="7">piece contains  $r \times r$  blocks</td></tr><tr><td>k</td><td>0.5</td><td>1</td><td>2</td><td>4</td><td>8</td><td></td></tr><tr><td>4 × 4 192</td><td></td><td>96</td><td>48</td><td>24</td><td>12</td><td></td></tr><tr><td>6 × 6 128</td><td></td><td>64</td><td>32</td><td>16</td><td>8</td><td></td></tr><tr><td>8 × 8 96</td><td></td><td>48</td><td>24</td><td>12</td><td>6</td><td></td></tr></table>

Values assume an input resolution of 384 × 384. For a puzzle piece of side length $\begin{array} { r } { p = 3 8 4 / k , } \end{array}$ the block size is $B = p / r ,$ where r × r is the number of B × B blocks per piece.

## 5.1 Image Classification

Figure 3 shows the top-1 classification accuracies as λ increases. The plain baseline reaches 87.00%, while chance performance is 6.67%. Classification remains well above chance for all transformations, even under substantial image-domain alteration. Gaussian blur degrades accuracy gradually, but still achieves 57.10% at $\sigma = 7 0$ . LOIs show a stronger scale efect, decreasing from near-baseline performance at small block sizes to 27.75% when correlations are removed globally.

Block-based transformations difer according to which structures they disrupt. Colour shufling, block rotation/flipping, and the NP-transformation preserve high classification accuracy across most scales, indicating that class discriminative information remains accessible despite substantial visual changes. In contrast, pixel shufling increasingly disrupts local adjacency and reduces accuracy at larger block sizes. Block shufling shows the opposite trend: small blocks strongly disrupt the global arrangement, whereas larger blocks preserve more of the layout and therefore recover classification performance.

The composed encryption schemes inherit these behaviours. Tanaka degrades with increasing pixel-shufling scale; EtC improves as larger shufled blocks preserve more global structure, and E-Tanaka remains comparatively low across scales because it combines local and global disruption. Overall, classification captures only semantic separability: high accuracy indicates that semantic labels remain predictable, but it does not imply that geometric or spatial structures required by other tasks are preserved. This does not mean classification is always the more permissive probe. As Sec. 5.2 shows, for some transformations $( e . g .$ Gaussian Blurring), classification degrades faster than angle prediction, so the achieved task utility depends on which structures a transformation removes and which are required for solving the task. Any single task gives an incomplete and potentially misleading picture of utility.

![](images/70b9a65012f9ca4cc1fc7c814da15acaeb1458d9ffa3854b0b50fab0eb464ed1.jpg)  
Fig. 3: Classification accuracy against transformation strength. Accuracy remains above chance for all transformations, indicating that semantic separability is preserved even under substantial image-domain alteration.

## 5.2 Angle Prediction

Figure 4 reports the mean angular error, with a chance performance of 90<sup>◦</sup>. The plain baseline achieves 2.48<sup>◦</sup>. Unlike classification, angle prediction directly tests whether a coherent global reference frame remains accessible.

Gaussian blur has a limited efect except for large scales, suggesting that relative orientation can be inferred from coarse global structure and does not require fine, high-frequency detail. LOIs behave diferently: performance remains close to the baseline for small blocks but degrades rapidly once correlations at relevant scales are removed, approaching chance in global settings.

For key-based primitives, colour shufling, NP transformation, and block rotation/flipping have little efect on angle prediction, while pixel shufling degrades at larger block sizes. Block shufling again shows a scale-dependent inverse trend: small shufled blocks disrupt global orientation cues, whereas larger blocks preserve enough layout for accurate prediction. Among composed schemes, E-Tanaka substantially impairs angle prediction across all scales, while Tanaka and EtC depend strongly on block size. These results demonstrate that transformations with similar classification accuracy can difer substantially in the geometric information they preserve.

![](images/8be1dc301b1e0e1b194254a32ffa663154e8bf692e4856b31d753b5df68c00e2.jpg)  
Fig. 4: Mean angular error against transform strength (lower is better). Prediction remains robust under transformations that preserve coarse global structure, such as moderate blur, but degrades strongly when spatial order is removed at larger scales or when local and global disruptions are combined.

## 5.3 Jigsaw Puzzle Solving

Figure 5 reports piece accuracy for $k \in \{ 4 , 6 , 8 \}$ puzzles. The plain baseline achieves high piece accuracy across all puzzle sizes, with chance performance decreasing from 6.25% for 4  4 puzzles to 1.56% for 8  8 puzzles.

LOI strongly reduces jigsaw performance when spatial order is removed at puzzle-relevant scales. When transformation blocks are as large as, or larger than, puzzle pieces, performance approaches chance. As more transformation blocks fit inside each puzzle piece, local spatial information becomes available, and performance improves. This confirms that the task is sensitive to boundary compatibility and local-to-global spatial consistency.

The key-based transformations reveal a diferent efect. Pixel shufling and several composed schemes remain solvable in the fixed-key setting, even when they disrupt global image structure. This does not indicate preservation of natural spatial compatibility. Instead, the puzzle solver can exploit deterministic transformation-induced regularities, such as key-dependent block or position cues that are consistent across training and testing. Evaluating these settings with a key diferent from the one used in training causes performance to collapse toward chance (see dashed lines in Fig. 5). Comparing LOI and Pixel Shufling confirms that the high performance of the latter is due to key-induced regularities. LOI can be viewed as a variant of Pixel Shufling, with diferent keys for each block. Attacks on block-scrambling schemes (such as EtC [25]) use puzzle solvers to reconstruct the plain image [4]. The high task utility can therefore also be interpreted as a leakage signal. Block shufling further shows that performance depends on the alignment between transformation scale and puzzle scale: some settings preserve or expose useful placement cues, while others remove them.

![](images/ae9d812c69d1ef0a16ec47084df6a7d17b80ea6a598cc513fd3539928bb82e60.jpg)  
Fig. 5: Jigsaw puzzle-solving accuracy for puzzles of diferent sizes; dashed coloured curves indicate evaluation with a diferent key from the training key. Puzzle solving exposes spatial placement cues that classification does not capture. LOI and some blockshufling settings degrade strongly when spatial order is removed at puzzle-relevant scales, whereas several key-based transformations remain highly solvable.

Jigsaw solving provides a complementary diagnostic view. Low performance indicates that spatial information required by the puzzle solver is no longer readily accessible. High performance may reflect either the preservation of natural spatial structure or shortcut cues introduced by transformations. This distinction highlights the importance of interpreting tasks jointly rather than in isolation.

## 5.4 Obfuscation-Utility Trade-of

Task utility alone is only one aspect of PET evaluation. A transformation that preserves high task performance while remaining perceptually close to the original provides limited obfuscation; one that strongly alters the image but destroys downstream learnability is equally impractical. Evaluation should therefore jointly consider task utility and image-domain leakage indicators. Figure 6 relates normalised task utility to an LPIPS-based obfuscation score [52], where higher utility indicates better learnability relative to the plain baseline and higher LPIPS indicates lower perceptual similarity to the original. The metrics in Tab. 2 constitute the broader evaluation framework; LPIPS is used here as a representative perceptual indicator. The full PET evaluation should consider additional metrics (see the supplementary material for the remaining metrics in Tab. 2).

![](images/b9152cbb6b45e3b41666d17086ede5c466b9a6249517f2f4e613523f549133ad.jpg)  
Fig. 6: Obfuscation-utility trade-of. Normalised task utility against normalised LPIPS. For angle prediction, the utility direction is inverted, since lower errors are better.

The trade-of varies substantially across tasks and transformations. Some transformations retain high classification accuracy while ofering limited perceptual obfuscation. Others increase perceptual distance but degrade geometric or spatial learnability. Fixed-key transformations add a further complication, achieving high perceptual obfuscation while still exposing deterministic cues that can be exploited by puzzle-solving. No single point on the obfuscation axis, therefore, predicts utility across tasks: the same LPIPS level corresponds to very diferent learnability depending on which structures a task requires.

## 6 Conclusion

We studied learnability under image obfuscation as a task-dependent property. Results show that classification accuracy, the dominant utility measure in PET evaluation, provides only a partial view. Transformations that preserve semantic separability can difer substantially in how well they preserve geometric consistency and spatial compatibility. The proposed protocol combines classification, relative-angle prediction, and jigsaw puzzle solving as lightweight diagnostic probes for complementary structures. Further, we show that fixed-key transformations can introduce deterministic shortcut cues that some tasks exploit. This highlights the need to interpret proxy-task performance jointly rather than in isolation. Future evaluations of privacy-preserving vision methods can benefit from moving beyond classification-only reporting to jointly consider task-dependent learnability, shortcut leakage, and image-domain obfuscation. We establish that the three tasks capture distinct structural properties, but not whether they predict full downstream tasks. Training downstream models on a subset of configurations is the natural next step and would confirm the proxies’ practical value.

## References

1. Aashman, V., Shreshth, J., Khanak, A.: Animal species classification – v3 (2023), https://www.kaggle.com/datasets/utkarshsaxenadn/animal- imageclassification-dataset/data

2. Brendel, W., Bethge, M.: Approximating CNNs with bag-of-local-features models works surprisingly well on ImageNet. In: International Conference on Learning Representations (2019)

3. Bu, Z., Dong, J., Long, Q., Su, W.J.: Deep learning with gaussian diferential privacy. Harvard Data Science Review 2(23) (2020). https://doi.org/10.1162/ 99608f92.cfc5dd25

4. Chuman, T., Kiya, H.: A jigsaw puzzle solver-based attack on image encryption using vision transformer for privacy-preserving DNNs. Information 14(6) (2023)

5. Coates, A., Ng, A., Lee, H.: An analysis of single-layer networks in unsupervised feature learning. In: Proceedings of the 14th International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings (2011)

6. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A largescale hierarchical image database. In: IEEE Conference on Computer Vision and Pattern Recognition (2009)

7. Dong, J., Durfee, D., Rogers, R.: Optimal diferential privacy composition for exponential mechanisms. In: International Conference on Machine Learning. PMLR (2020)

8. Dwork, C.: Diferential privacy. In: International Colloquium on Automata, Languages and Programming (2006)

9. Geiping, J., Bauermeister, H., Dröge, H., Moeller, M.: Inverting gradients: How easy is it to break privacy in federated learning? In: Advances in Neural Information Processing Systems. vol. 33 (2020)

10. Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F.A., Brendel, W.: ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In: International Conference on Learning Representations (2018)

11. Gentry, C.: Fully homomorphic encryption using ideal lattices. In: Proceedings of the 41st Annual ACM Symposium on Theory of Computing. Association for Computing Machinery (2009). https://doi.org/10.1145/1536414.1536440

12. Gidaris, S., Singh, P., Komodakis, N.: Unsupervised representation learning by predicting image rotations. In: International Conference on Learning Representations (2018)

13. Gilad-Bachrach, R., Dowlin, N., Laine, K., Lauter, K., Naehrig, M., Wernsing, J.: CryptoNets: Applying neural networks to encrypted data with high throughput and accuracy. In: Proceedings of the 33rd International Conference on Machine Learning. vol. 48. PMLR (2016)

14. Goldreich, O., Micali, S., Wigderson, A.: How to play any mental game, or a completeness theorem for protocols with honest majority. In: Providing sound foundations for cryptography: on the work of Shafi Goldwasser and Silvio Micali. ACM (2019). https://doi.org/10.1145/3335741.3335755

15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2016)

16. Hermann, K., Chen, T., Kornblith, S.: The origins and prevalence of texture bias in convolutional neural networks. In: Advances in Neural Information Processing Systems. vol. 33 (2020)

17. Hirose, M., Imaizumi, S., Kiya, H.: Learnable image encryption without key management for privacy-preserving vision transformers. IEEE Access 13 (2025). https://doi.org/10.1109/ACCESS.2025.3635235

18. Huang, Y., Song, Z., Li, K., Arora, S.: InstaHide: Instance-hiding schemes for private distributed learning. In: Proceedings of the 37th International Conference on Machine Learning. vol. 119. PMLR (2020)

19. Jarin, I., Eshete, B.: PRICURE: Privacy-preserving collaborative inference in a multi-party setting. In: Proceedings of the 2021 ACM Workshop on Security and Privacy Analytics. Association for Computing Machinery (2021). https://doi. org/10.1145/3445970.3451156

20. Juvekar, C., Vaikuntanathan, V., Chandrakasan, A.: GAZELLE: A low-latency framework for secure neural network inference. In: Proceedings of the 27th USENIX Security Symposium. USENIX Association (2018)

21. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: International Conference on Learning Representations (2015)

22. Koenderink, J.J., Van Doorn, A.J.: The structure of locally orderless images. International Journal of Computer Vision 31 (1999)

23. Krizhevsky, A.: Learning multiple layers of features from tiny images. Tech. rep., University of Toronto (2009)

24. Kucur, E.N., Buyuktanir, T., Ugurelli, M., Yildiz, K.: Privacy-preserving machine learning techniques: Cryptographic approaches, challenges, and future directions. Applied Sciences 16 (2026). https://doi.org/10.3390/app16010277

25. Kurihara, K., Shiota, S., Kiya, H.: An encryption-then-compression system for the JPEG standard. In: 2015 Picture Coding Symposium (PCS). IEEE (2015)

26. Larkin, K.G.: Reflections on shannon information: In search of a natural information-entropy for images. arXiv preprint arXiv:1609.01117 (2016)

27. Le, Y., Yang, X., et al.: Tiny ImageNet visual recognition challenge. CS 231N 7 (2015)

28. LeCun, Y., Bottou, L., Bengio, Y., Hafner, P.: Gradient-based learning applied to document recognition. Proceedings of the IEEE 86 (1998)

29. Lee, J., Lee, E., Lee, J.W., Kim, Y., Kim, Y.S., No, J.S.: Precise approximation of convolutional neural networks for homomorphically encrypted data. IEEE Access 11 (2023). https://doi.org/10.1109/ACCESS.2023.3287564

30. Lindeberg, T.: Scale-space theory: A basic tool for analyzing structures at diferent scales. Journal of Applied Statistics 21 (1994)

31. Liu, J., Teshome, W., Ghimire, S., Sznaier, M., Camps, O.: Solving masked jigsaw puzzles with difusion vision transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

32. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019)

33. Lou, Q., Feng, B., Charles Fox, G., Jiang, L.: Glyph: Fast and accurately training deep neural networks on encrypted data. In: Advances in Neural Information Processing Systems. vol. 33. Curran Associates, Inc. (2020)

34. Madono, K., Tanaka, M., Onishi, M., Ogawa, T.: Block-wise scrambled image recognition using an adaptation network. arXiv preprint arXiv:2001.07761 (2020)

35. Mahalakshmi, K., Nagarajan, S.: Comprehensive review and analysis of image encryption techniques. IEEE Access 13 (2025). https://doi.org/10.1109/ACCESS. 2025.3578158

36. McMahan, B., Moore, E., Ramage, D., Hampson, S., Arcas, B.A.y.: Communication-eficient learning of deep networks from decentralized data. In:

Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS). vol. 54. PMLR (2017)

37. McPherson, R., Shokri, R., Shmatikov, V.: Defeating image obfuscation with deep learning. arXiv preprint arXiv:1609.00408 (2016)

38. Melis, L., Song, C., De Cristofaro, E., Shmatikov, V.: Exploiting unintended feature leakage in collaborative learning. In: 2019 IEEE symposium on Security and Privacy (SP). IEEE (2019)

39. Mohassel, P., Zhang, Y.: SecureML: A system for scalable privacy-preserving machine learning. In: 2017 IEEE Symposium on Security and Privacy (SP). IEEE (2017). https://doi.org/10.1109/SP.2017.12

40. Noroozi, M., Favaro, P.: Unsupervised learning of visual representations by solving jigsaw puzzles. In: European Conference on Computer Vision. Springer (2016)

41. Ohrimenko, O., Schuster, F., Fournet, C., Mehta, A., Nowozin, S., Vaswani, K., Costa, M.: Oblivious multi-party machine learning on trusted processors. In: Proceedings of the 25th USENIX Conference on Security Symposium. USENIX Association (2016)

42. Phuong, T.T., et al.: Diferentially private stochastic gradient descent via compression and memorization. Journal of Systems Architecture 135 (2023). https: //doi.org/10.1016/j.sysarc.2022.102819

43. SaberiKamarposhti, M., Ghorbani, A., Yadollahi, M.: A comprehensive survey on image encryption: Taxonomy, challenges, and future directions. Chaos, Solitons & Fractals 178 (2024). https://doi.org/10.1016/j.chaos.2023.114361

44. Schneider, T., Wang, H.C., Yalame, H.: HE-SecureNet: An eficient and usable framework for model training via homomorphic encryption. In: Proceedings of the 24th Workshop on Privacy in the Electronic Society (2025). https://doi.org/10. 1145/3733802.3764063

45. Shafran, A., Segev, G., Peleg, S., Hoshen, Y.: Crypto-oriented neural architecture design. In: 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE (2021). https://doi.org/10.1109/ICASSP39728. 2021.9413592

46. Shokri, R., Stronati, M., Song, C., Shmatikov, V.: Membership inference attacks against machine learning models. In: 2017 IEEE symposium on Security and Privacy (SP). IEEE (2017)

47. Sirichotedumrong, W., Kinoshita, Y., Kiya, H.: Pixel-based image encryption without key management for privacy-preserving deep neural networks. IEEE Access 7 (2019). https://doi.org/10.1109/ACCESS.2019.2959017

48. Tanaka, M.: Learnable image encryption. In: 2018 IEEE International Conference on Consumer Electronics-Taiwan (ICCE-TW) (2018). https://doi.org/10.1109/ ICCE-China.2018.8448772

49. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing 13(4) (2004)

50. Wen, J., Zhang, Z., Lan, Y., Cui, Z., Cai, J., Zhang, W.: A survey on federated learning: Challenges and applications. International Journal of Machine Learning and Cybernetics 14(2) (2023). https://doi.org/10.1007/s13042-022-01647-y

51. Zamir, A.R., Sax, A., Shen, W., Guibas, L.J., Malik, J., Savarese, S.: Taskonomy: Disentangling task transfer learning. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2018)

52. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2018)

53. Zhang, Y., Zheng, M., Shang, Y., Chen, X., Lou, Q.: HEPrune: Fast private training of deep neural networks with encrypted data pruning. In: Advances in Neural Information Processing Systems. vol. 37. Curran Associates, Inc. (2024). https: //doi.org/10.52202/079017-1616

54. Zhou, I., Tofigh, F., Piccardi, M., Abolhasan, M., Franklin, D., Lipman, J.: Secure multi-party computation for machine learning: A survey. IEEE Access 12 (2024). https://doi.org/10.1109/ACCESS.2024.3388992

55. Zhu, L., Liu, Z., Han, S.: Deep leakage from gradients. In: Advances in Neural Information Processing Systems. vol. 32 (2019)

56. Zuo, X., Luopan, Y., Han, R., Zhang, Q., Liu, C.H., Wang, G., Chen, L.Y.: FedViT: Federated continual learning of vision transformer at the edge. Future Generation Computer Systems 154 (2024). https://doi.org/10.1016/j.future.2023.11. 038

Table 4: Hyperparameters for Animal Species Classification.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Dataset</td><td>Animal Species Classification dataset [1]</td></tr><tr><td>Number of classes</td><td>15</td></tr><tr><td>Data augmentation</td><td>None</td></tr><tr><td>Model architecture</td><td>ResNet-34 [15]</td></tr><tr><td>Initial weights</td><td>Random (trained from scratch)</td></tr><tr><td>Input resolution</td><td>3 × 416 × 416</td></tr><tr><td>Loss function</td><td>Cross-entropy</td></tr><tr><td>Optimizer</td><td>Adam [21]</td></tr><tr><td>Learning rate</td><td>10−4</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Max. epochs</td><td>100</td></tr><tr><td>Early stopping</td><td>patience 5 on validation accuracy</td></tr><tr><td></td><td>Learning-rate schedule Multistep at epochs [20, 50] with γ = 0.1</td></tr><tr><td>Checkpoint selection</td><td>Best validation accuracy</td></tr><tr><td>Evaluation metric</td><td>Top-1 classification accuracy</td></tr><tr><td>Random seed</td><td>73</td></tr><tr><td>Framework</td><td>PyTorch 2.11.0+cu126</td></tr><tr><td>Gaussian blur σ</td><td>{1, 2, 4, 6, 8, 10, 14, 16, 20, 30, 40, 50, 60, 70}</td></tr><tr><td>Block size B</td><td>{1, 2, 8, 16, 26, 32, 104, 208, 416}</td></tr><tr><td>Hardware</td><td>1× NVIDIA GeForce RTX 4090</td></tr></table>

## Supplementary Material

## Hyperparameter Settings

Tables 4 to 6 list the hyperparameters used for classification, relative angle prediction, and jigsaw puzzle solving, respectively. Unless stated otherwise, PyTorch defaults are used. Block sizes B are chosen as divisors of the respective input size, so that blocks tile each image exactly. All experiments use a single random seed per configuration.

## Obfuscation-Utility Trade-of

Figures 7 to 16 show the trade-of between task utility and the remaining imagedomain obfuscation metrics complementing the LPIPS trade-of in the main paper. In each figure, colour indicates the transformation family and marker shape the individual transformation; each point corresponds to one transformation– scale configuration. Task utility is normalised per task as

$$
u = { \frac { \mathrm { p e r f o r m a n c e } - \mathrm { c h a n c e } } { \mathrm { b a s e l i n e } - \mathrm { c h a n c e } } } ,
$$

where baseline denotes the plain-baseline performance and chance the chance level $( 1 / N$ for classification, $9 0 ^ { \circ }$ mean angular error for angle prediction, $1 / k ^ { 2 }$ for jigsaw solving). For angle prediction, whose raw metric decreases with quality, the error is inverted before normalisation so that higher u uniformly indicates higher utility. A value of u = 1 corresponds to plaintext-level performance and u = 0 to chance; values above 1 occur where transformed-domain performance exceeds the identity baseline. For keyed transformations, we attribute these values to deterministic key-induced shortcut cues. However, values above 1 also occur for non-keyed transformations (notably angle prediction under mild Gaussian blur and small-block LOIs). In these cases the efect is explained by the transformation mildly aiding the specific task and model (i.e., low-pass filtering easing orientation estimation for a small CNN), rather than by shortcut memorisation.

Table 5: Hyperparameters for Relative Angle Prediction.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Dataset</td><td>Animal Species Classification dataset [1]</td></tr><tr><td>Angle sampling</td><td>α1, α2 ∼ U[−180°, 180°]</td></tr><tr><td>Data augmentation</td><td>None</td></tr><tr><td>Model architecture</td><td>CNN (2× conv, max-pool, 4× linear)</td></tr><tr><td>Initial weights</td><td>Random (trained from scratch)</td></tr><tr><td>Input resolution</td><td>3 × 304 × 304</td></tr><tr><td>Loss function</td><td>Cosine similarity</td></tr><tr><td>Optimizer</td><td>Adam [21]</td></tr><tr><td>Learning rate</td><td>10−4</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Max. epochs</td><td>50</td></tr><tr><td>Early stopping</td><td>patience 5 on validation loss</td></tr><tr><td></td><td>Learning-rate schedule Reduce on plateau, factor 0.5, patience 3</td></tr><tr><td>Checkpoint selection</td><td>Lowest validation loss</td></tr><tr><td>Evaluation metric</td><td>Mean angular error in degrees</td></tr><tr><td>Random seed</td><td>42</td></tr><tr><td>Framework</td><td>PyTorch 2.11.0+cu126</td></tr><tr><td>Gaussian blur σ</td><td>{1, 2, 4, 6, 8, 10, 14, 16, 20, 30, 40, 50, 60, 70}</td></tr><tr><td>Block size B</td><td>{4, 8, 16, 38, 76, 152}</td></tr><tr><td>Hardware</td><td>1× NVIDIA GeForce RTX 4090</td></tr></table>

The obfuscation metrics are min–max normalised per metric across all configurations, with direction-aware inversion so that higher values uniformly indicate stronger obfuscation. Diferential and key-sensitivity metrics (NPCR, UACI, ) are omitted from these plots, as they characterise a transformation’s sensitivity rather than a per-configuration obfuscation level.

Table 6: Hyperparameters for Jigsaw Puzzle Solving.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Dataset</td><td>Animal Species Classification dataset [1]</td></tr><tr><td>Puzzle sizes k</td><td>{4, 6, 8}</td></tr><tr><td>Permutations</td><td>Sampled uniformly at random during training</td></tr><tr><td>Data augmentation</td><td>None</td></tr><tr><td>Model architecture</td><td>JPDVT [31]</td></tr><tr><td>Diffusion timesteps T</td><td>1000</td></tr><tr><td>Model patch size</td><td>16</td></tr><tr><td>Initial weights</td><td>Random (trained from scratch)</td></tr><tr><td>Input resolution</td><td>3 × 384 × 384</td></tr><tr><td>Loss function</td><td>Cosine similarity</td></tr><tr><td>Optimizer</td><td>AdamW [32]</td></tr><tr><td>Learning rate</td><td>10-4</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Max. epochs</td><td>100</td></tr><tr><td>Early stopping</td><td>patience 5 on validation loss</td></tr><tr><td></td><td>Learning-rate schedule Multistep at epochs [20, 50, 75] with γ = 0.1</td></tr><tr><td>Checkpoint selection</td><td>Lowest validation loss</td></tr><tr><td>Evaluation metric</td><td>Piece accuracy</td></tr><tr><td>Random seed</td><td>42</td></tr><tr><td>Framework</td><td>PyTorch 2.11.0+cu126</td></tr><tr><td>Block size B</td><td>Coupled to puzzle size k</td></tr><tr><td>Hardware</td><td>2× NVIDIA GeForce RTX 4090</td></tr></table>

![](images/9f755365f217ddb5e7442c8c7abbe921d94897c1aa8edcef6d0d3ddbedcb1d0d.jpg)  
Fig. 7: Obfuscation–utility trade-of: normalised task utility against normalised MSE.

![](images/5b0ed3b73421160d9a86cd91f782d064916da92897a43667d7183e471d452673.jpg)  
Fig. 8: Obfuscation–utility trade-of: normalised task utility against normalised PSNR.

![](images/3a6a7016531046fdfaaa299fea743415c96dbcfda3b5c8190ed377f8bc401e05.jpg)  
Fig. 9: Obfuscation–utility trade-of: normalised task utility against normalised SSIM.

![](images/d1d519ea5fa8bc3a4f4d9fe3c2541402f4f56f5725808d948b0738cf46a01d95.jpg)  
Fig. 10: Obfuscation–utility trade-of: normalised task utility against normalised MI.

![](images/cf04ce70f5b4c7e1446f5afa9e2edf3f282552988248f9d9780a51968467ee14.jpg)  
Fig. 11: Obfuscation–utility trade-of: normalised task utility against normalised $\chi ^ { 2 } .$

![](images/8afbe499e44d1b67b5ded6935747ee43c0e0eebc8831ece471732ab05e6c4500.jpg)  
Fig. 12: Obfuscation–utility trade-of: normalised task utility against normalised $D _ { \mathrm { K L } }$

![](images/db4618d92fa61de62585fe29620add9001182baae616ec27decdd4c073b239a3.jpg)  
Fig. 13: Obfuscation–utility trade-of: normalised task utility against normalised IE.

![](images/0c2336c1ea360a18b0896c2bc7f352f063f0cfa0cfc626d557584e13ef57c58a.jpg)  
Fig. 14: Obfuscation–utility trade-of: normalised task utility against normalised 2D-IE.

![](images/77ca31467472b135f22606f317ab212af735fe96ac173e0fca440e50fefaa66b.jpg)  
Fig. 15: Obfuscation–utility trade-of: normalised task utility against normalised CC.

![](images/ba71e7130cc30115089ca8cf2b271d0ae4428cfac8b5bbcbf2edf783d15bbe4e.jpg)  
Fig. 16: Obfuscation–utility trade-of: normalised task utility against normalised SFM.