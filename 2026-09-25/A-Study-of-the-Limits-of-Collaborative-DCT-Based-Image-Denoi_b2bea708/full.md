# A Study of the Limits of Collaborative DCT-Based Image Denoising via Interpretable Neural Networks

Cristian Comellas<sup>1\*</sup>, Julia Navarro<sup>1</sup> and Antoni Buades<sup>1</sup>

<sup>1</sup>Department of Mathematics and Computer Science, Universitat de les Illes Balears, Cra. de Valldemossa, km 7.5, Palma, 07122, Balearic Islands, Spain.

\*Corresponding author(s). E-mail(s): cristian.comellas@uib.es;   
Contributing authors: julia.navarro@uib.es; toni.buades@uib.es;

## Abstract

Image denoising remains a fundamental problem in image restoration, with applications in photography, biomedical, and scientific imaging. Modern deep neural networks achieve strong performance by learning powerful image priors, but often rely on large black-box models with limited interpretability. In contrast, DCT-based sliding-window and collaborative filtering methods such as BM3D ofer clear algorithmic structure, but depend on handcrafted and non-diferentiable operations. This work studies how far such structured collaborative filtering principles can be pushed when reformulated as trainable models. We introduce DeepBM3D, a compact fully diferentiable architecture that combines non-local patch grouping, DCT-domain filtering, and multi-stage refinement within a BM3D-inspired pipeline. Lightweight convolutional feature extractors guide patch grouping, while filtering is performed through learned Wiener weights in the DCT domain. Experiments show that DeepBM3D improves over classical and hybrid baselines, remains competitive with FFDNet at low and moderate noise levels, and performs particularly well on repetitive textures.

Keywords: Image denoising, non-local filtering, collaborative filtering, transform-domain denoising, attention mechanisms, deep learning

## 1 Introduction

Image denoising is a long-standing problem in image processing and computer vision, fundamental to applications ranging from consumer photography to medical and scientific imaging. The task consists of recovering a clean image from a noisy observation, a challenge that lies at the intersection of statistical estimation, signal modeling, and machine learning. Classical methods based on sparsity and non-local redundancy, including transform-domain adaptive filtering approaches [1], Wiener filtering [2], and collaborative filtering methods such as BM3D [3], have remained reference points for decades thanks to their interpretability, eficiency, and strong performance despite relying on handcrafted priors. These algorithms exploit structured operations such as transform-domain filtering, often relying on transforms like the DCT, together with patch grouping to leverage sparsity and self-similarity within natural images, providing a clear understanding of each processing stage.

Deep learning has fundamentally changed the landscape of image restoration. Modern convolutional and transformer-based models [4–6] learn powerful image priors directly from data, achieving state-of-the-art performance across diverse noise conditions and datasets. However, these gains often come at the expense of interpretability and computational eficiency. Typical networks contain millions of parameters and operate as black boxes, making their internal representations dificult to analyze or control. In contrast, classical denoising pipelines ofer a clear algorithmic structure and modularity, which explicitly separates the grouping, filtering, and aggregation stages.

This growing contrast has motivated renewed interest in models that bridge the gap between data-driven flexibility and the principled structure of traditional algorithms. Recent model-based and hybrid approaches [7–9] have shown that it is possible to embed optimization principles and classical operations into trainable networks, combining interpretability with learning. In particular, Herbreteau et al. [10] proposed DCT2Net, demonstrating that classical DCT-based denoising pipelines can benefit from learnable thresholding strategies.

Building upon these ideas, we propose DeepBM3D, a compact and fully diferentiable architecture for collaborative transform-domain image denoising. DeepBM3D replaces handcrafted components commonly used in classical collaborative filtering pipelines with learnable and fully diferentiable modules. Our model adopts a twostage design and introduces lightweight convolutional modules to guide patch grouping and estimate transform-domain filtering weights in a trainable manner. This formulation enables gradient-based optimization across all stages while preserving the structural interpretability of collaborative filtering approaches.

To preserve the underlying structure of classical collaborative filtering pipelines, we use a fixed DCT as transform and PSAL [11], which performs diferentiable patch matching. We explore two configurations: a lightweight version that already outperforms several classical and hybrid methods, and a heavier variant that approaches or exceeds FFDNet [5], a representative CNN-based denoiser without explicit collaborative filtering constraints, in selected low and moderate noise settings.

Our results demonstrate that classical collaborative filtering principles, including non-local patch grouping and transform-domain Wiener filtering, can be efectively integrated within a fully diferentiable and end-to-end trainable framework.

The proposed architecture combines interpretability, modularity, and competitive denoising performance within a compact trainable model.

The remainder of this paper is organized as follows. Section 2 reviews related classical, deep, and hybrid denoising approaches. Section 3 briefly summarizes the key principles of collaborative filtering as exemplified by BM3D [3]. Section 4 presents our proposed DeepBM3D architecture. Experimental setup and results are described in Sections 5 and 6, respectively. Finally, Section 7 concludes the paper and discusses future research directions.

## 2 Related Work

This section reviews the main methodological trends shaping the evolution from classical signal processing and non-local filtering, through modern deep architectures, to hybrid frameworks that integrate optimization and learning. Together, these developments motivate the development of structured, learnable denoising architectures that combine classical priors with deep learning.

## 2.1 Classical Image Denoising Methods

Early image denoising methods relied on signal processing and statistical models that enforced sparsity, smoothness, or self-similarity. Sliding DCT thresholding [1] and Wavelet thresholding [12] exploited sparsity of coeficients to suppress noise while preserving edges, whereas dictionary-learning methods such as K-SVD [13] represented patches as sparse combinations of learned atoms, later extended through online learning for scalability [14]. The Non-Local Means (NLM) algorithm [15] introduced the idea of selfsimilarity by averaging non-local patches with similarity-based weights, a principle refined by the Non-Local Bayes model [16], which framed patch grouping and filtering in a Bayesian setting. Low-rank modeling further generalized these ideas by assuming that groups of similar patches form low-rank matrices. Weighted Nuclear Norm Minimization (WNNM) [17, 18] formalized this as a convex optimization problem with adaptive weighting of singular values, efectively linking patch grouping and sparse representation.

Other frameworks approached denoising from a probabilistic viewpoint. The Expected Patch Log Likelihood (EPLL) model [19] used Gaussian mixture models to describe natural image patches, achieving strong generalization across noise levels, while the Wiener filter [2] provided a theoretical foundation for optimal linear denoising under Gaussian assumptions. Among these approaches, collaborative filtering methods based on non-local patch grouping and transform-domain sparsity have proven particularly efective. A prominent example is the Block-Matching and 3D Filtering (BM3D) algorithm [3, 20], which combines nonlocal grouping, transform-domain filtering, and patch aggregation in a unified multi-stage process. BM3D’s success lies in exploiting both non-local redundancy and transform-domain sparsity, yielding highly competitive results even compared to modern learning-based models. Extensions such as BM3D-PCA [21] and BM4D [22] adapted the framework to data-dependent bases and volumetric signals. However, the reliance on handcrafted transforms and hard block matching limits adaptability, motivating the transition toward trainable and fully diferentiable alternatives.

## 2.2 Learning-Based Denoisers: From CNNs to Transformers

Deep learning methods replaced image priors with learned representations, enabling powerful data-driven denoising. Early neural approaches used multilayer perceptrons [23], but convolutional neural networks (CNNs) soon became dominant due to their parameter sharing and spatial locality. DnCNN [4] pioneered residual learning by predicting the noise component instead of the clean image, and FFDNet [5] incorporated a noiselevel map to handle spatially variant noise with a single model. MemNet [24] introduced memory blocks to retain contextual information across layers, and subsequent works generalized these ideas to diverse restoration tasks.

Self-supervised and unsupervised paradigms addressed the scarcity of clean training data. Noise2Noise [25] demonstrated that denoisers can be trained using only pairs of noisy images, while Noise2Void [26] and Noise2Self [27] extended this principle to single-image denoising through pixel masking strategies. These methods significantly broadened the applicability of deep denoising in real-world imaging scenarios.

The encoder–decoder architecture introduced by U-Net [28] became the foundation for most CNN-based denoisers. Variants such as RID-Net [29], DDUNet [30], and SADNet [31] integrated attention mechanisms, dense connectivity, and spatial adaptivity to enhance feature representation. Later architectures like MPR-Net [32], HINet [33], and NAFNet [34] employed multi-stage refinement and improved normalization for higher eficiency and stability. These advances consolidated CNNs as the standard for high-quality image restoration. Beyond taskspecific CNNs, general-purpose denoisers such as DRUNet [35] have been proposed as flexible modules that can serve both as standalone restoration networks and as plug-and-play priors for a wide range of inverse problems. This design demonstrates how a single deep architecture can act as a universal regularizer, bridging traditional optimization frameworks and data-driven learning.

Transformers extended this progress by modeling long-range dependencies. SwinIR [36] builds upon the Swin Transformer [37] architecture, adapting its shifted-window self-attention mechanism for eficient and scalable image restoration. Restormer [6] and Uformer [38] further extend this paradigm by combining hierarchical attention with U-shaped architectures, enabling global context modeling while preserving spatial precision.

FFTFormer [39] further bridged spatial and frequency domains, and DnT [40] proposed an unsupervised transformer-based denoiser. Together, these models demonstrate a shift from explicit priors to hierarchical learned representations. More recently, difusion-based generative models have also been adapted for image restoration. SR3 [41] formulates superresolution as an iterative denoising process, showing that difusion mechanisms can progressively refine images toward high-fidelity reconstructions. Although primarily developed for generative tasks, these models highlight the close conceptual link between denoising and iterative refinement. Eficiency-oriented designs, such as EficientNet [42], also influenced modern lightweight denoisers for mobile and real-time applications.

## 2.3 Model-Based Deep Learning and Hybrid Paradigms

Model-based deep learning seeks to combine interpretability with learning capacity by embedding optimization principles into neural architectures. Algorithm unrolling [43] first showed that sparse coding inference can be approximated through a fixed-depth network, inspiring frameworks such as ADMM-Net [44] and ADMM-CSNet [45], which unroll iterative solvers for compressive sensing and MRI. Similarly, TNRD [46] learned parameters of PDE-based difusion processes, linking deep networks and variational models. Plug-and-Play (PnP) [47] and Regularization by Denoising (RED) [48] further connected classical optimization with deep priors, treating denoisers as implicit regularizers. Related approaches learned explicit proximal operators [49] or exploited network structure as an implicit prior, as in Deep Image Prior [50].

Hybrid architectures integrate classical operations directly within neural frameworks. BM3D-Net [51] and DCT2Net [10] emulate transformdomain filtering, while BMCNN [52] incorporates block-matching as a learnable stage. Diferentiable non-local modules such as N3Net [7], Non-Local Networks [8], and Dual Attention [53] generalize patch similarity into attention mechanisms, later extended by PSAL [11], which performs diferentiable approximate nearest-neighbor matching via stochastic attention. Low-rank priors have also been integrated into deep unrolled architectures, as in HLR-DUR [54] and ILRNet [9], bridging low-rank modeling and deep optimization. Recent reviews [55, 56] summarize this convergence between classical and deep paradigms, emphasizing that denoisers can serve as flexible priors for inverse problems.

## 2.4 Learning Collaborative Transform-Domain Denoising

Collaborative filtering methods based on nonlocal patch grouping and transform-domain sparsity have long been central to classical image denoising, with BM3D [3, 20] representing one of the most influential examples. These approaches inspired several attempts to integrate their principles within learnable frameworks. Methods such as BM3D-Net [51] and BMCNN [52] introduce neural components into collaborative filtering pipelines, while still relying on fixed or nondiferentiable stages. Other works focus on learning individual components of these pipelines: DCT2Net [10] learns interpretable transformdomain filters, N3Net [7] approximates patch matching through attention mechanisms, and PSAL [11] performs diferentiable patch grouping via stochastic attention.

These developments highlight the potential of combining non-local similarity, transform-domain filtering, and deep learning within unified architectures. Building on these ideas, the proposed DeepBM3D integrates non-local patch grouping and collaborative transform-domain filtering into a fully diferentiable and trainable framework, enabling end-to-end learning while preserving the structural principles of classical collaborative denoising methods.

## 3 Key Principles of BM3D

Block-Matching and 3D Filtering (BM3D) [3, 20] is a classical image denoising algorithm that exploits non-local self-similarity through collaborative filtering in a transform domain. In this section we briefly summarize the key principles of BM3D that are relevant for understanding the proposed architecture.

Let $\boldsymbol { Y } ~ = ~ \boldsymbol { X } + \eta$ denote the observed noisy image, where X is the clean image and η ∼ ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ is additive white Gaussian noise. BM3D operates by grouping similar patches, applying collaborative filtering in a transform domain, and aggregating the resulting estimates.

Patch Grouping For each reference patch extracted from the noisy image, BM3D searches within a spatial window to identify a set of similar patches according to an $\ell _ { 2 }$ distance. These patches are stacked together to form a 3D group, exploiting the observation that similar structures often recur across an image.

Collaborative Transform-Domain Filtering Each group is processed using a separable 3D transform that combines a 2D transform applied to each patch and a 1D transform across the group dimension. Noise is suppressed by shrinking transform coeficients, either through hard thresholding or Wiener filtering, before applying the inverse transform to reconstruct denoised patches.

Aggregation The filtered patches are returned to their original image locations. Because patches overlap, multiple estimates contribute to each pixel. A weighted averaging strategy aggregates these contributions, producing the final restored image.

BM3D performs this procedure in two successive stages: a first stage based on hardthresholding to obtain a basic estimate, followed by a second stage that refines the result using Wiener filtering guided by the basic estimate. These principles of non-local patch grouping, transform-domain collaborative filtering, and aggregation form the conceptual foundation for the architecture proposed in this work.

## 4 DeepBM3D Architecture

We introduce DeepBM3D, a lightweight endto-end trainable architecture for collaborative transform-domain image denoising. The proposed model integrates non-local patch grouping, transform-domain collaborative filtering, and aggregation within a unified diferentiable framework. Its design is guided by the structural principles of classical collaborative filtering methods, including BM3D, with trainable modules that enable data-driven optimization.

DeepBM3D adopts a two-stage architecture that follows the progressive refinement strategy commonly used in classical denoising pipelines. In this design, three key operations are implemented as diferentiable modules: patch grouping is achieved through an attention-based mechanism (PSAL [11]), collaborative filtering is performed via learned Wiener weights in the DCT domain, and the 3D transform is implemented as a linear layer. In addition, compact CNN-based feature extractors are employed to guide the grouping process and enhance representational power. A visual comparison between BM3D and DeepBM3D is shown in Figure 1. Overall, this yields a modular and interpretable architecture that combines the structural principles of collaborative filtering with the flexibility of deep learning.

## 4.1 Mathematical formulation

Let $Y ~ \in ~ \mathbb { R } ^ { 1 \times H \times W }$ be a noisy grayscale input image, with $Y = X + \eta$ where $\eta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . A noise level map $\sigma \in \mathbb { R } ^ { 1 \times H \times W }$ is assumed known and concatenated channel-wise with the input:

$$
\boldsymbol { x } _ { \mathrm { i n } } = \mathsf { c a t } ( Y , \sigma ) \in \mathbb { R } ^ { 2 \times H \times W } .
$$

The architecture is structured around the following modules: (i) a fixed, diferentiable blockmatching operator M implemented with PS ${ \mathrm { A L } } ,$ (ii) three feature extractors: $\phi _ { \theta }$ for image-level features, $\psi _ { \theta } ^ { \mathrm { i m g } }$ for patch-level spatial features, and $\psi _ { \boldsymbol \theta } ^ { \mathrm { d c t } }$ for patch-level transform features, (iii) two Wiener weight estimation networks $\omega _ { \theta } ^ { ( 1 ) }$ and $\omega _ { \theta } ^ { ( 2 ) }$ and (iv) fixed 3D DCT/IDCT operators implemented as separable 2D and 1D transforms.

## Stage 1

The first step, as in BM3D, is patch grouping. To guide this process, feature maps are extracted from the noisy input and the given noise map:

$$
f ^ { ( 1 ) } = \phi _ { \theta } ( x _ { \mathrm { i n } } ) .
$$

For each reference location i, the block-matching module M receives two inputs: (i) a feature tensor $f ^ { ( 1 ) }$ that provides a learned representation for measuring similarity, and (ii) the image ${ \cal Y } ,$ from which the actual patches are extracted. The operator selects the C most similar patches to the reference one according to distances in feature space, and then gathers the corresponding patches from $Y ,$ producing

$$
\mathcal { G } _ { Y , i } ^ { ( 1 ) } = \mathcal { M } ( f _ { i } ^ { ( 1 ) } , Y ) , \quad \mathcal { G } _ { Y , i } ^ { ( 1 ) } \in \mathbb { R } ^ { N \times k \times k } , \quad N = C + 1 .
$$

This follows the same principle of non-local patch grouping used in classical collaborative filtering methods (where N denotes the number of patches per group), but here the selection of matching coordinates is guided by PSAL, which performs diferentiable, attention-based matching in the feature domain. Although PSAL itself has no trainable parameters, it enables gradient propagation through patch selection, a key advantage over traditional hard matching.

Each group is transformed with a separable 3D DCT:

$$
U _ { Y , i } ^ { ( 1 ) } = \mathcal { T } _ { 3 D } ( \mathcal { G } _ { Y , i } ^ { ( 1 ) } ) .
$$

Classical collaborative filtering (BM3D)  
Stage 1: Hard-Thresholding  
![](images/1724550016d9c1d104010bed1633b635a7be0375422055d2afc642f30e651391.jpg)

Proposed architecture (DeepBM3D)  
![](images/2d38bf9e7581e5a50b7f558f0e582de6ad42ce74c466be2836fe87af5fe2d946.jpg)  
Fig. 1 Comparison of BM3D (top) and DeepBM3D (bottom): both follow a two-stage design, with fixed operators shown in blue/green and learned modules in orange. The operator M represents block matching using PSAL [11].

Filtering weights are predicted and applied:

$$
w _ { i } ^ { ( 1 ) } = \omega _ { \theta } ^ { ( 1 ) } ( z _ { i } ) , \quad \widehat { U } _ { Y , i } ^ { ( 1 ) } = w _ { i } ^ { ( 1 ) } \odot U _ { Y , i } ^ { ( 1 ) } ,
$$

where ⊙ represents the Hadamard product, and the feature vector $z _ { i }$ is generated as follows:

$$
\begin{array} { r } { z _ { \mathrm { d c t } } = \psi _ { \boldsymbol \theta } ^ { \mathrm { d c t } } ( U _ { Y , i } ^ { ( 1 ) } ) , } \\ { z _ { \mathrm { i m g } } = \psi _ { \boldsymbol \theta } ^ { \mathrm { i m g } } ( \mathcal { G } _ { Y , i } ^ { ( 1 ) } ) , } \end{array}
$$

$$
z _ { i } = \cot ( z _ { \mathrm { d c t } } , z _ { \mathrm { i m g } } , \sigma ) .
$$

After the inverse transform, the filtered patches are aggregated at their reference locations to form the first-stage output:

$$
\hat { x } ^ { ( 1 ) } = \mathrm { A g g r e g a t e } ( \mathcal { T } _ { 3 D } ^ { - 1 } ( \widehat { U } _ { Y , i } ^ { ( 1 ) } ) ) .
$$

Although $\hat { x } ^ { ( 1 ) } \in \mathbb { R } ^ { 1 \times H \times W }$ has the shape of an image, it is not constrained to visually resemble

X. No supervision is applied at this stage, and $\hat { x } ^ { ( 1 ) }$ should be interpreted as a latent intermediate representation optimized to benefit the second stage. This output plays the same functional role as $\widehat { X } ^ { ( 1 ) }$ in classical BM3D, serving as a guidance signal for the second stage.

## Stage 2

The second stage mirrors this process but leverages the intermediate estimate for more reliable grouping and filtering. Features are first extracted from the concatenation of the basic estimate and the noise map:

$$
f ^ { ( 2 ) } = \phi _ { \theta } \bigl ( \cot ( \hat { x } ^ { ( 1 ) } , \sigma ) \bigr ) .
$$

PSAL is then reused to group patches from both the noisy input $Y$ and the intermediate estimate $\hat { x } ^ { ( 1 ) }$

$$
\begin{array} { r l } & { \mathcal { G } _ { Y , i } ^ { ( 2 ) } = \mathcal { M } ( f _ { i } ^ { ( 2 ) } , Y ) , } \\ & { \mathcal { G } _ { \hat { x } , i } ^ { ( 2 ) } = \mathcal { M } ( f _ { i } ^ { ( 2 ) } , \hat { x } ^ { ( 1 ) } ) , } \end{array}
$$

with $\mathcal { G } _ { Y , i } ^ { ( 2 ) } , \mathcal { G } _ { \hat { x } , i } ^ { ( 2 ) } \in \mathbb { R } ^ { N \times k \times k } .$

Each group is transformed with the separable 3D DCT:

$$
U _ { Y , i } ^ { ( 2 ) } = \mathcal { T } _ { 3 D } ( \mathcal { G } _ { Y , i } ^ { ( 2 ) } ) , \qquad U _ { \hat { x } , i } ^ { ( 2 ) } = \mathcal { T } _ { 3 D } ( \mathcal { G } _ { \hat { x } , i } ^ { ( 2 ) } ) .
$$

Wiener weights are predicted and applied elementwise:

$$
w _ { i } ^ { ( 2 ) } = \omega _ { \theta } ^ { ( 2 ) } ( z _ { i } ^ { ( 2 ) } ) , \qquad \widehat { U } _ { Y , i } ^ { ( 2 ) } = w _ { i } ^ { ( 2 ) } \odot U _ { Y , i } ^ { ( 2 ) } ,
$$

where the feature vector $z _ { i } ^ { ( 2 ) }$ is encoded from the transform domain and concatenated with the noise map:

$$
\begin{array} { r l r } & { } & { h _ { Y , i } = \psi _ { \boldsymbol \theta } ^ { \mathrm { d c t } } ( U _ { Y , i } ^ { ( 2 ) } ) , } \\ & { } & { h _ { \hat { x } , i } = \psi _ { \boldsymbol \theta } ^ { \mathrm { d c t } } ( U _ { \hat { x } , i } ^ { ( 2 ) } ) , } \\ & { } & { z _ { i } ^ { ( 2 ) } = \mathsf { c a t } ( h _ { Y , i } , h _ { \hat { x } , i } , \sigma ) . } \end{array}
$$

Finally, inverse transforms and aggregation at reference locations yield the denoised estimate:

$$
\hat { x } ^ { ( 2 ) } = \mathrm { A g g r e g a t e } \Big ( { \cal T } _ { 3 D } ^ { - 1 } \Big ( \widehat { \cal U } _ { Y , i } ^ { ( 2 ) } \Big ) \Big ) ,
$$

where the final estimate $\hat { x } ^ { ( 2 ) }$ is therefore analogous to $\widehat { X } ^ { ( 2 ) }$ in BM3D.

## 4.2 Implementation details

The learned modules in DeepBM3D are designed to be lightweight while retaining interpretability. All convolutional layers employ $3 \times 3$ kernels with reflect padding to mitigate boundary artifacts, followed by ReLU activations unless otherwise noted.

Feature extractors DeepBM3D relies on three small CNNs that share a common architecture (see Figure 2) but serve distinct roles: (i) the imagelevel extractor $\phi _ { \theta }$ , which operates on the concatenation of the noisy image and the noise map; (ii) the spatial-domain extractor $\psi _ { \theta } ^ { \mathrm { i m g } }$ , applied to grouped patches in the pixel domain; and (iii) the transform-domain extractor $\psi _ { \boldsymbol \theta } ^ { \mathrm { d c t } }$ , applied to grouped patches in the DCT domain. Each extractor follows the structure Conv $3 \times 3 $ Conv $3 \times 3 $ Conv $3 \times 3 .$ , with reflect padding and ReLU activations after each convolution. The output dimensionality of $\phi _ { \theta }$ is controlled by n<sub>features</sub>, while $\psi _ { \theta } ^ { \mathrm { i m g } }$ and $\psi _ { \boldsymbol \theta } ^ { \mathrm { d c t } }$ produce N · m features, where N is the number of grouped patches and m a capacity multiplier. Two configurations are considered:

$$
\begin{array} { l } { { \bullet l i g h t \colon n _ { \mathrm { f e a t u r e s } } = 3 2 , m = 4 . } } \\ { { \bullet h e a v y \colon n _ { \mathrm { f e a t u r e s } } = 6 4 , m = 8 . } } \end{array}
$$

Block matching Patch grouping is implemented with the diferentiable PSAL operator [11]. Patch similarity is computed on learned features, while the grouped patches are drawn from either the noisy image Y or the first-stage estimate $\hat { x } ^ { ( 1 ) }$ . This ensures that reconstruction is always grounded in the input, while gradients propagate through the selection mechanism. The number of candidates is C, plus the reference patch, yielding $N = C + 1$ patches per group. In practice, we use patch size k = 5, and N = 16 candidates.

3D transforms Groups are processed with separable 2D+1D DCT/IDCT operators implemented as fixed, non-trainable linear layers. This retains the transform-domain formulation commonly used in collaborative filtering methods, while ensuring diferentiability and a low parameter count.

![](images/acb887fac80b73c87bf14dff97687aaee7dacd880acdb71dc7911f00d130c6e7.jpg)  
Fig. 2 Feature-extraction block used across DeepBM3D $( \phi _ { \theta } , \psi _ { \theta } ^ { \mathrm { i m g } }$ , and $\psi _ { \boldsymbol \theta } ^ { \mathrm { d c t } } )$ . The module is three consecutive Conv 3×3 + ReLU layers with reflect padding, thus preserving the spatial size $( H \times { \breve { W } } )$ . For the image-level extractor ϕ , the channel sizes are $C _ { \mathrm { i n } } \mathrm { = } 2$ (noisy image and noise map) and F=32 (light) $/ \ F { = } 6 4$ (heavy). For the patch-level extractors $\psi _ { \theta } ^ { \mathrm { i m g } }$ and $\psi _ { \theta } ^ { \mathrm { d c t } } , C _ { \mathrm { i n } } { = } N$ candidates with $N { = } C { + } 1$ , and F=Nm with m=4 (light) / m=8 (heavy).

Wiener weights Collaborative filtering is parameterized by two identical CNNs, $\omega _ { \theta } ^ { ( 1 ) }$ and $\omega _ { \theta } ^ { ( 2 ) }$ which estimate elementwise weights for Stage 1 and Stage 2, respectively. Each network follows the structure Conv 3×3 → Conv 3×3 → Conv 1×1 → Conv $1 \times 1 $ Conv 1×1 → Conv 3×3 → sigmoid, with ReLU activations between convolutional layers and reflect padding to reduce boundary artifacts. The full architecture is shown in Figure 3. Outputs are bounded in [0, 1] by the final sigmoid activation but are not normalized across candidates, granting the model additional flexibility. Both DeepBM3D-light and DeepBM3D-heavy use this same structure, difering only in their respective multipliers (m = 4 and m = 8).

Aggregation We adopt reference patch replacement. Each denoised reference patch substitutes its noisy counterpart, and overlapping contributions are averaged. Alternative schemes, such as weighted overlapping or central-pixel aggregation, were tested but did not improve results.

Inference To enhance robustness and reduce boundary artifacts, the model processes the image in a sliding-window fashion over overlapping patches. For a patch size of $6 4 \times 6 4$ , we use a stride of 4 during evaluation, although good results are also obtained with stride 32. The outputs of overlapping regions are merged using simple averaging. In practice, the formulation above applies equally whether Y denotes a full image or an individual patch.

## 5 Experimental setup

We describe the experimental protocol used to evaluate DeepBM3D, including the datasets and noise configurations, training procedure, and baseline methods used for comparison.

## 5.1 Datasets and Noise Settings

To train and validate our models, we construct a grayscale dataset by combining images from DIV2K [57], Flickr2K [58], and the Waterloo Exploration Database [59]. All images are converted to grayscale, resulting in a fixed split of 7,969 training images and 292 validation images, which is used consistently across all experiments. During training, random crops of size 64 × 64 are sampled from these images at each epoch and used as input patches. In addition, data augmentation is applied in the form of random horizontal and vertical flips, as well as random 90◦ rotations, to improve generalization. This dataset is used exclusively for training and validation.

For testing, we follow common practice and evaluate on four standard benchmarks: BSD100 [60], Kodak [61], McMaster (McM) [62], and Set14 [63]. These datasets provide a diverse range of content and are widely adopted for denoising evaluation.

![](images/7be658cde0d9cf015938cc76054073cbde1fe4a0baa868b8fd408ec580d28bdc.jpg)  
Fig. 3 Wiener weights estimation network $( \omega _ { \theta } ^ { ( 1 ) }$ and $\omega _ { \theta } ^ { ( 2 ) } )$ . The model combines Conv 3 × 3, Conv 1 × 1, and nonlinear activations to map features of size $( 2 N m + 1 ) \times H \times W$ to Wiener weights of size $N { \overline { { \times H \times W } } } .$ . All convolutions use reflect padding. Here, N denotes the number of grouped patches and m the internal capacity multiplier. In the light configuration, m=4; in the heavy configuration, m=8.

Noise is synthetically applied as additive white Gaussian noise (AWGN). During flexible training phases (see Section 5.2), noise levels σ are sampled uniformly from the interval [0, 50]. For evaluation, we report results at fixed noise levels $\sigma \in$ {5, 15, 25, 35}, which are standard in the denoising literature. All noise realizations are generated using a fixed random seed to ensure reproducibility, so that test conditions are identical across runs.

## 5.2 Training Details

We adopt a three-phase training scheme to optimize both generalization and performance across noise levels. All models are trained end-to-end using an $\ell _ { 1 }$ loss on the final output $\hat { x } ^ { ( 2 ) }$ . Optimization is performed with Adam $( \mathrm { l r } ~ = ~ 1 0 ^ { - 4 } , \beta _ { 1 } ~ =$ $0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ , using an efective batch size of 16. Training is conducted on a single NVIDIA RTX A6000 GPU (48 GB).

During training, input patches are sampled dynamically from the training images. At each epoch, random 64 × 64 crops are drawn from randomly selected images, so that only a subset of the dataset is seen per epoch. This stochastic sampling strategy efectively exposes the model to a large diversity of patches across epochs while keeping each epoch computationally lightweight.

• Phase 1 (exploration): training proceeds from epoch 0 to epoch 10,000 using a cosine annealing scheduler with warm restarts (CosineAnnealingWarmRestarts, $\begin{array} { r l r } { T _ { 0 } } & { { } = } & { 5 0 } \end{array}$ $\bar { T _ { \mathrm { m u l t } } } = 2 , \eta _ { \mathrm { m i n } } = 1 0 ^ { - 1 0 } )$ . During this phase, additive white Gaussian noise (AWGN) is sampled uniformly from σ ∈ [0, 50].

• Phase 2 (refinement): training continues from epoch 10,000 to epoch 20,000 starting from the best checkpoint obtained in Phase 1. A step scheduler (StepLR, step size=2000, γ = 0.6) is used in this phase. The resulting models are referred to as flexible models, as they are trained to operate across a wide range of noise levels.

• Phase 3 (noise-specific fine-tuning): from each flexible model, specialized variants are obtained for fixed noise levels $\sigma \in \{ 5 , 1 5 , 2 5 , 3 5 \}$ by further fine-tuning from epoch 20,000 up to epoch 30,000, using the same step scheduler as in Phase 2. These noise-specific models are referred to as fixed models, as they are optimized for a particular noise level.

After each training phase, we select the checkpoint that achieves the highest average PSNR on the validation split. This model is then used to initialize the subsequent training stage, ensuring progressive refinement guided by validation performance.

We evaluate two parameterizations of DeepBM3D: a lightweight model with 890K parameters that surpasses classical BM3D, and a larger variant with 3.4M parameters that achieves further gains. For each parameterization, we obtain both flexible models (trained across noise levels) and fixed models (fine-tuned for specific noise levels), resulting in four configurations: lightweight–flexible, lightweight–fixed, heavy–flexible, and heavy–fixed. These variants form the basis for our experimental evaluation, which we compare against a range of existing methods described below.

## 5.3 Baselines and Comparison Methods

To assess the performance of DeepBM3D, we compare it against representative denoising methods spanning classical, hybrid, and deep learning paradigms. As a classical collaborative filtering reference, we include BM3D [3], using the IPOL implementation [20], and a DCT-based denoiser (DCTDenoiser) employed as a baseline in DCT2Net [10].

To bridge the gap between classical priors and data-driven learning, we also consider DCT2Net and its variant DCT/DCT2net [10], which extend classical DCT-based pipelines with trainable modules. These hybrid models retain part of the structure and interpretability of traditional methods while benefiting from optimization through learning. We evaluate them using the oficial pretrained weights released by the authors.

Finally, we include the deep learning method FFDNet [5], a widely used denoiser trained endto-end for grayscale AWGN removal with a noiselevel map as input. For consistency, we also rely on the oficial pretrained weights provided by the authors.

All baselines are evaluated under identical conditions using fixed noise levels σ ∈ {5, 15, 25, 35}. Whenever supported, models are provided with the exact noise-level map to ensure fair comparisons. This setup allows us to attribute performance diferences to the models themselves rather than to discrepancies in evaluation protocol.

## 6 Results and Discussion

In this section, we evaluate the performance of DeepBM3D through quantitative and visual comparisons against baseline methods. We further analyze the impact of model capacity, training strategy, and architectural design choices through ablation studies.

## 6.1 Quantitative Results

Average PSNR results across four benchmark datasets and four standard noise levels are reported in Table 1. DeepBM3D consistently improves upon classical and hybrid baselines, including BM3D, BM3D-Net, and the DCT2Net family, across most evaluated settings. This confirms the benefit of replacing handcrafted or partially learned components with fully diferentiable modules while preserving the structural principles of collaborative filtering pipelines.

Compared to BM3D-Net, which also follows a BM3D-inspired design, DeepBM3D achieves more stable performance across noise levels. While BM3D-Net performs competitively around its reference noise level (σ = 15), its performance degrades significantly for noise levels outside this range. In contrast, DeepBM3D maintains consistent behavior across all noise levels, both in its flexible and noise-specific variants.

Compared to FFDNet, which is a powerful, fully end-to-end learned model, DeepBM3D achieves competitive performance, particularly at low-to-moderate noise levels. In several cases (e.g., σ = 5 and σ = 15), our heavy fixed variant matches or slightly exceeds FFDNet, demonstrating the strength of our modular design despite the structural constraints imposed by collaborative transform-domain filtering. Notably, all predictions in DeepBM3D are produced through element-wise filtering in the transform domain, in contrast to the unconstrained nature of FFDNet.

At higher noise levels (σ ≥ 25), FFDNet maintains a more significant advantage, particularly on natural datasets such as Kodak and McMaster. This highlights the limitations of strongly structured filtering architectures under extreme noise conditions, where larger receptive fields and more expressive filtering become advantageous. Nevertheless, DeepBM3D remains competitive, and its interpretability and modularity make it an attractive alternative.

Fine-tuning for fixed noise levels leads to consistent gains across all variants, with improvements of up to 0.2 dB compared to flexible training. The heavy fixed model achieves the highest scores overall, surpassing FFDNet in some settings and narrowing the gap in others. The lightweight models also perform remarkably well, showing that most performance gains can be achieved with compact architectures that retain the eficiency and modularity of collaborative filtering designs.

## 6.2 Qualitative Results

We complement the quantitative evaluation with qualitative comparisons at representative noise levels. Figures 4, 5, 6, and 7 illustrate visual results for σ ∈ {5, 15, 25, 35}, respectively.

Across the evaluated noise levels, DeepBM3D consistently outperforms BM3D, particularly at higher noise intensities where BM3D tends to produce structured artifacts such as blockiness or spurious lines. Our method preserves a cleaner appearance without introducing such distortions.

At low noise (σ = 5, Figure 4), DeepBM3D achieves results on par with FFDNet in terms of both noise removal and detail preservation. At low-to-moderate noise levels (σ = 15, Figure 5), our model shows clear advantages in reconstructing repetitive textures and high-frequency details. This is visible, for example, in the fine structures of the motorcycle spokes and forks. Similar behavior on high-frequency repetitive details can be observed in Figure 6.

At higher noise levels (σ = 35, Figure 7), FFD-Net retains a slight edge in preserving extremely fine details, yet DeepBM3D remains visually competitive, ofering high-quality denoising with faithful structure reconstruction and efective suppression of noise. Overall, the combination of DCT-domain filtering and patch grouping yields visually compelling reconstructions across a wide range of noise conditions.

## 6.3 Texture Reconstruction Analysis

To further analyze the behavior of the proposed method, we evaluate its performance on structured textures from the Brodatz dataset. These images exhibit strong periodic patterns and non-local self-similarity, making them particularly suitable for assessing methods based on patch grouping and transform-domain filtering.

Quantitative results are reported in Table 2. Figure 8 illustrates the texture images used for building this table. DeepBM3D consistently achieves superior PSNR values compared to FFD-Net across all evaluated noise levels, which is not the case for general images as reported in Table

1. The diferences between the light and heavy versions are also narrower in the case of textures. These results highlight the advantage of combining non-local grouping and transform-domain filtering within a learnable framework for images with strong repetitive and oscillatory structure.

## 6.4 Ablation Studies

We conduct ablation experiments to analyze the contribution of key design choices in DeepBM3D. These studies isolate the efect of individual components, such as the number of stages and the patch grouping configuration, providing insight into how each element influences denoising performance and visual quality.

## Efect of the second stage

To evaluate the contribution of the second stage in the DeepBM3D pipeline, we compare the performance of the single-stage and two-stage variants using the light flexible configuration. Quantitative results are reported in Table 3. Across all datasets and noise levels, the two-stage model consistently outperforms the single-stage counterpart, confirming the benefit of iterative refinement. The improvement becomes more noticeable at higher noise levels, where the second stage efectively attenuates residual noise and reduces structured artifacts that may appear after the first filtering. Qualitative results are shown in Figure 9. It can be seen that the second stage produces smoother and more coherent textures, mitigating aliasing efects while preserving the overall structure and natural appearance of the image. These findings support the use of a two-stage refinement strategy within the proposed diferentiable framework, as it enhances stability and visual consistency.

## Efect of candidate count (N)

The results in Table 4, together with the visual summary in Figure 10, show a clear and consistent improvement as the number of candidates increases. Across all datasets and noise levels, larger groups yield higher PSNR values, confirming that more extensive aggregation enhances the collaborative filtering stage. The efect is especially noticeable for low and medium noise levels (σ≤15), where increasing N from 8 to 32 provides gains of up to 0.5 dB, particularly on the

Table 1 Average PSNR results (dB) on multiple datasets and noise levels. The table compares our proposed DeepBM3D variants (light/heavy, fixed/flexible) against classical (BM3D, DCTDenoiser), hybrid (BM3D-Net, DCT-based), and deep learning baselines (FFDNet). The best and second-best results for each dataset and noise level are highlighted in bold and underlined, respectively.
<table><tr><td>σ</td><td>BSD100</td><td>Kodak</td><td>McM</td><td>Set14</td></tr><tr><td>DCTDenoiser 5</td><td>36.62</td><td>37.47</td><td>38.23</td><td>37.10</td></tr><tr><td>DCT2Net</td><td>37.11</td><td>38.07</td><td>38.76</td><td>37.52</td></tr><tr><td>DCT/DCT2net</td><td>37.06</td><td>37.93</td><td>38.73</td><td>37.49</td></tr><tr><td>BM3D-Neta</td><td></td><td></td><td></td><td></td></tr><tr><td>BM3D</td><td>37.29</td><td>38.19</td><td>39.09</td><td>37.89</td></tr><tr><td>FFDNet</td><td>37.49</td><td>38.42</td><td>39.39</td><td>37.98</td></tr><tr><td>DeepBM3D light fixed</td><td>37.69</td><td>38.65</td><td>39.67</td><td>38.38</td></tr><tr><td>DeepBM3D light flexible</td><td>37.34</td><td>38.35</td><td>39.33</td><td>37.59</td></tr><tr><td>DeepBM3D heavy fixed</td><td>37.72</td><td>38.69</td><td>39.71</td><td>38.42</td></tr><tr><td>DeepBM3D heavy flexible</td><td>37.43</td><td>38.44</td><td>39.43</td><td>37.69</td></tr><tr><td>DCTDenoiser 15</td><td>29.73</td><td>31.17</td><td>32.12</td><td>31.00</td></tr><tr><td>DCT2Net</td><td>30.84</td><td>32.31</td><td>33.21</td><td>32.02</td></tr><tr><td>DCT/DCT2net</td><td>30.71</td><td>32.12</td><td>33.12</td><td>31.97</td></tr><tr><td>BM3D-Net</td><td>31.08</td><td>32.48</td><td>33.57</td><td>32.30</td></tr><tr><td>BM3D</td><td>30.69</td><td>32.20</td><td>33.29</td><td>32.16</td></tr><tr><td>FFDNet</td><td>31.36</td><td>32.86</td><td>34.00</td><td>32.70</td></tr><tr><td>DeepBM3D light fixed</td><td>31.35</td><td>32.85</td><td>33.97</td><td>32.82</td></tr><tr><td>DeepBM3D light flexible</td><td>31.14</td><td>32.68</td><td>33.80</td><td>32.58</td></tr><tr><td>DeepBM3D heavy fixed</td><td>31.40</td><td>32.92</td><td>34.05</td><td>32.90</td></tr><tr><td>DeepBM3D heavy flexible</td><td>31.22</td><td>32.77</td><td>33.90</td><td>32.70</td></tr><tr><td>DCTDenoiser 25</td><td>27.24</td><td>28.83</td><td></td><td></td></tr><tr><td>DCT2Net</td><td>28.40</td><td>29.96</td><td>29.62 30.78</td><td>28.50 29.65</td></tr><tr><td>DCT/DCT2net</td><td>28.26</td><td>29.80</td><td>30.68</td><td>29.61</td></tr><tr><td>BM3D-Net</td><td></td><td></td><td></td><td></td></tr><tr><td>BM3D</td><td>28.49</td><td>29.93</td><td>30.88</td><td>28.95</td></tr><tr><td>FFDNet</td><td>28.20</td><td>29.87</td><td>30.83</td><td>29.79</td></tr><tr><td></td><td>28.94</td><td>30.57</td><td>31.62</td><td>30.44</td></tr><tr><td>DeepBM3D light fixed</td><td>28.85</td><td>30.46</td><td>31.48</td><td>30.44</td></tr><tr><td>DeepBM3D light flexible</td><td>28.70</td><td>30.34</td><td>31.35</td><td>30.28</td></tr><tr><td>DeepBM3D heavy fixed DeepBM3D heavy flexible</td><td>28.90</td><td>30.52</td><td>31.54</td><td>30.53</td></tr><tr><td></td><td>28.77</td><td>30.42</td><td>31.45</td><td>30.39</td></tr><tr><td>DCTDenoiser 35</td><td>25.86</td><td>27.49</td><td>28.10</td><td>26.97</td></tr><tr><td>DCT2Net</td><td>26.94</td><td>28.50</td><td>29.18</td><td>28.08</td></tr><tr><td>DCT/DCT2net</td><td>26.81</td><td>28.39</td><td>29.08</td><td>28.05</td></tr><tr><td>BM3D-Neta</td><td>24.32</td><td>22.87</td><td>27.01</td><td>23.71</td></tr><tr><td>BM3D</td><td>26.73</td><td>28.43</td><td>29.27</td><td>28.26</td></tr><tr><td>FFDNet</td><td>27.50</td><td>29.17</td><td>30.09</td><td>28.96</td></tr><tr><td>DeepBM3D light fixed DeepBM3D light flexible</td><td>27.37</td><td>29.01</td><td>29.88</td><td>28.88</td></tr><tr><td>DeepBM3D heavy fixed</td><td>27.24 27.41</td><td>28.89 29.07</td><td>29.76 29.95</td><td>28.74 28.98</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeepBM3D heavy flexible</td><td>27.30</td><td>28.97</td><td>29.87</td><td>28.87</td></tr></table>

<sup>a</sup>BM3D-Net is pretrained at σ = 15. Although it accepts diferent noise levels as input via rescaling, its performance degrades for noise levels far from this reference point (e.g., σ 5, 35 ).

![](images/9460b32867a21784ec6220f94dbba4029571fc970dbfda6c407e8e9f612209b8.jpg)  
GT

![](images/30c5e03dfbde3f67333048baa674e19f5b1d9283e31007c1aef3795dfbcbeb9b.jpg)  
Noisy (σ = 5)

![](images/ce282794ac4bff5dd2919a8a7681b60c4e797d37ca3b2fe43e2a7b7df15ce4fa.jpg)

![](images/5f13b174297fb9881be77872cb98a3b937a2b96c5eb3c9a11e50882e802dc1e1.jpg)  
BM3D

![](images/011bf49d7e6d21fe18de524fdea4ed3084c68c237cbef266561dc27a7231fe89.jpg)  
FFDNet

![](images/9a457aed35e8ae4e3e8507ffdc5ef7b71f74fbc83ee6ac2c8cfa42bef6c5813b.jpg)

![](images/6d01637e7e1935e8184dcfad5c2d1c3b543abd8d465193cc3b34b6af9ae57a4e.jpg)  
DeepBM3D

Fig. 4 Qualitative comparison on BSD100 at σ = 5. Top: full images (GT annotated with zoom region). Bottom: identical zoomed region across methods.  
![](images/4fdf35cd609eb7365c08df9c8ca090efd751c21be35b39b1e31b3b170fe235e5.jpg)  
GT

![](images/69fca56b09ccf8e58b2a512274259f83511c8ea7b8bf21500858590966d75728.jpg)  
Noisy (σ = 15)

![](images/b3d0e73e852787acf4a5d9befd30483ce6d23b094fbc93674a029924a6d8156a.jpg)  
BM3D

![](images/0f245f12bfde48194ef0a47cacfca2091d046513702eb648e1614197b7d7422e.jpg)  
FFDNet

![](images/7d1a177bde9a2bde9acae2c9bd7097077df3d9ade9bcd7339894a6ec5e4a329e.jpg)  
DeepBM3D  
Fig. 5 Qualitative comparison on Kodak at σ = 15. Top: full images (GT annotated with zoom region). Bottom: identica zoomed region across methods.

Table 2 PSNR (dB) on Brodatz textures for diferent noise levels. We compare classical (BM3D), deep learning (FFDNet), and the proposed DeepBM3D (light and heavy fixed) model.  
Table 3 Average PSNR (dB) on multiple datasets and noise levels, comparing single-stage and full twostage pipelines for the DeepBM3D light flexible variant.
<table><tr><td>σ</td><td>15</td><td>25</td><td>35</td></tr><tr><td>BM3D</td><td>29.00</td><td>26.52</td><td>25.05</td></tr><tr><td>FFDNet</td><td>29.32</td><td>26.80</td><td>25.30</td></tr><tr><td>DeepBM3D Light</td><td>29.44</td><td>26.91</td><td>25.38</td></tr><tr><td>DeepBM3D Heavy</td><td>29.48</td><td>26.93</td><td>25.40</td></tr></table>

<table><tr><td></td><td>σ</td><td>BSD100</td><td>Kodak</td><td>McM</td><td>Set14</td></tr><tr><td>1 stage</td><td>5</td><td>36.87</td><td>38.19</td><td>39.18</td><td>37.32</td></tr><tr><td>2 stages</td><td></td><td>37.34</td><td>38.35</td><td>39.33</td><td>37.59</td></tr><tr><td>1 stage</td><td>15</td><td>30.86</td><td>32.45</td><td>33.58</td><td>32.37</td></tr><tr><td>2 stages</td><td></td><td>31.14</td><td>32.68</td><td>33.80</td><td>32.58</td></tr><tr><td>1 stage</td><td>25</td><td>28.41</td><td>30.04</td><td>31.05</td><td>29.95</td></tr><tr><td>2 stages</td><td></td><td>28.70</td><td>30.34</td><td>31.35</td><td>30.28</td></tr><tr><td>1 stage</td><td>35</td><td>26.96</td><td>28.55</td><td>29.41</td><td>28.33</td></tr><tr><td>2 stages</td><td></td><td>27.24</td><td>28.89</td><td>29.76</td><td>28.74</td></tr></table>

McMaster and Kodak datasets. Although the relative improvement becomes smaller for higher noise levels, the trend remains monotonic across all cases.

Among the evaluated datasets, McMaster exhibits the greatest sensitivity to the number of candidates, suggesting that richly textured and colorful images benefit the most from dense patch grouping. The corresponding plot in Figure 10 illustrates this monotonic behavior, showing how each noise level consistently benefits from larger groups, although with diminishing returns. Moreover, the gains from N=16 to N=32 are not negligible, indicating that the model could achieve even higher performance with larger groups. However, increasing N substantially raises the parameter count and computational cost during both training and inference.

![](images/e2d2f4613cf01fb0efed85abc618f6258b8c1e767b14fef8849534bcb953036f.jpg)  
GT

![](images/0f6acde2c4e0aab04553f1e86a365c35fd91c5563db6e53060853a4fc7ca24fd.jpg)  
Noisy (σ = 25)

![](images/2b17185c2e7bb629843fbd281896b102a98ecdec57f023fd2936351c2b1e2ebc.jpg)  
BM3D

![](images/7ab9a654cf31b4209c2c59404e4aafcf53d44f3fc133b693b7c014950870664f.jpg)  
FFDNet

![](images/74c3f5eac0f6f1901e79639dcbdca0304b8f6669866e485886e047bc91907e33.jpg)  
DeepBM3D  
Fig. 6 Qualitative comparison on Set14 at σ = 25. Top: full images (GT annotated with zoom region). Bottom: identical zoomed region across methods.

![](images/7e27c03163781a63f46a3699159a67100cb1b2bcad363c4196624e2beebb6da3.jpg)  
GT

![](images/2905ac0906f2d2fe75ef7fc2f7b11603bb5582021da5a12b8dc67825e820a35b.jpg)  
Noisy (σ = 35)

![](images/5c445c1a7c8d56d3a3e81b67781d91dda17e00d7f5f1130b8da5615eded68983.jpg)  
BM3D

![](images/d0bcbe38c6e390ee7a66a61c8f59e159265be23f7b20b232589e0e928c83a75f.jpg)  
FFDNet

![](images/1ae685478293f3124a9e43c6bcd1ff7562d7d42f594a88c89bc2208eb008f9c9.jpg)  
DeepBM3D

Fig. 7 Qualitative comparison on McMaster at σ =35. Top: full images (GT annotated with zoom region). Bottom: identical zoomed region across methods.  
![](images/9ccf776aa623871d90af5a7b599531cc2a861daef507a88f98fb7c1add5a4246.jpg)  
Fig. 8 Brodatz textures used for building Table 2.

For this reason, we adopt N=16 as a balanced configuration for our experiments, ofering an excellent trade-of between denoising quality and eficiency. Overall, these results highlight the flexibility of DeepBM3D to scale its accuracy depending on computational constraints and application needs.

## 6.5 Limitations

While DeepBM3D shows strong performance across most scenarios, its limitations become more apparent at higher noise levels. In such cases, the method tends to produce slightly blurrier results compared to fully end-to-end models such as FFDNet, as shown in Figure 11.

This behavior is expected, as our architecture inherits certain structural constraints from BM3D. First, it operates on relatively small 3D blocks constructed from 5 × 5 patches, limiting the spatial context available during filtering. Second, the grouping stage is restricted to 15 nonlocal candidates plus the reference patch, which may be insuficient under severe noise. Lastly, the core filtering operation consists of an element-wise multiplication between the original patches and learned weights, providing a more constrained representation compared to the deeper convolutional architectures used in modern denoisers. These factors collectively constrain the model’s capacity, especially under challenging high-noise conditions.

![](images/1a88686e122c5f0b98f1ff6a16460dbaacdeaad367cd774617d1dbcfaac35847.jpg)  
2 stages  
1 stage  
Fig. 9 Comparison of the one- and two-stage variants of DeepBM3D on Set14 at $\sigma = 3 5$ . The two-stage model refines the first-stage output, yielding sharper and cleaner details in the zoomed region.

Table 4 Average PSNR (dB) on multiple datasets and noise levels, comparing the efect of diferent candidate counts (N = C+1) in the DeepBM3D (light flexible) variant.
<table><tr><td></td><td>σ</td><td>BSD100</td><td>Kodak</td><td>McM</td><td>Set14</td></tr><tr><td> $N = 8$ </td><td>5</td><td>37.06</td><td>38.17</td><td>39.10</td><td>37.37</td></tr><tr><td>N = 16</td><td></td><td>37.34</td><td>38.35</td><td>39.33</td><td>37.59</td></tr><tr><td>N = 32</td><td></td><td>37.52</td><td>38.49</td><td>39.51</td><td>37.76</td></tr><tr><td>N = 8</td><td>15</td><td>30.92</td><td>32.44</td><td>33.55</td><td>32.34</td></tr><tr><td>N = 16</td><td></td><td>31.14</td><td>32.68</td><td>33.80</td><td>32.58</td></tr><tr><td>N = 32</td><td></td><td>31.28</td><td>32.83</td><td>33.96</td><td>32.76</td></tr><tr><td>N = 8</td><td>25</td><td>28.47</td><td>30.09</td><td>31.09</td><td>29.98</td></tr><tr><td>N = 16</td><td></td><td>28.70</td><td>30.34</td><td>31.35</td><td>30.28</td></tr><tr><td>N = 32</td><td></td><td>28.83</td><td>30.48</td><td>31.51</td><td>30.46</td></tr><tr><td>N = 8</td><td>35</td><td>27.00</td><td>28.64</td><td>29.49</td><td>28.41</td></tr><tr><td>N = 16</td><td></td><td>27.24</td><td>28.89</td><td>29.76</td><td>28.74</td></tr><tr><td>N = 32</td><td></td><td>27.35</td><td>29.03</td><td>29.93</td><td>28.93</td></tr></table>

## 7 Conclusion

In this work, we introduced DeepBM3D, a fully diferentiable architecture for collaborative transform-domain image denoising inspired by classical BM3D-style pipelines. The proposed model combines non-local patch grouping, DCTdomain collaborative filtering, and multi-stage refinement within a unified trainable framework, replacing handcrafted operations with lightweight diferentiable modules while preserving the structural principles of collaborative filtering methods.

Experimental results show that DeepBM3D consistently improves over classical and hybrid baselines and remains competitive with modern deep denoisers such as FFDNet, despite relying on significantly stronger structural constraints. In particular, the proposed architecture performs especially well on repetitive and oscillatory textures, suggesting that non-local grouping and DCT-domain filtering remain highly efective priors for structured image content.

At the same time, our experiments reveal the limitations of strongly structured transformdomain filtering architectures. Under high noise conditions, restricted patch groups and localized filtering become less expressive than unconstrained deep models with larger receptive fields and higher representational capacity. These observations provide insight into both the strengths and the limits of collaborative DCT-based denoising within modern learning frameworks.

Overall, our results suggest that a substantial part of image denoising performance can still be explained through classical principles such as transform sparsity and non-local self-similarity. Rather than replacing these ideas, deep learning can enhance them through diferentiable optimization, enabling compact and interpretable architectures that remain competitive with significantly larger neural models. Future work will investigate how far these structured approaches can be pushed through richer grouping strategies, adaptive transforms, and more expressive collaborative filtering mechanisms, particularly in challenging high-noise and real-world restoration scenarios.

![](images/777ec7849736ae54edbc7e7753112018fda46a4d02111b813130d611c674456b.jpg)  
Number of candidates (N)

Fig. 10 Efect of the number of candidates. Average PSNR gain (∆ dB) achieved by varying the number of candidates (N) in the light flexible DeepBM3D configuration, relative to the lightest setting with N=8, across diferent noise levels.  
![](images/0b28f782f8008cd2128776ec1ec22d7216d9d0f927dd0a60fc9a5fa1a8b12fc8.jpg)  
Fig. 11 Qualitative comparison on Kodak at σ =35. Top: full images (GT annotated with zoom region). Bottom: identical zoomed region across methods.

Funding. This work was supported by MCIN/AEI/10.13039/501100011033 and FEDER, European Union, under grant PID2021- 125711OB-I00, and by the Spanish Ministry of Universities under grant FPU24/02805.

The funders had no role in the study design; in the collection, analysis, or interpretation of data; in the writing of the manuscript; or in the decision to submit the article for publication.

Author contributions. C.C. developed and implemented the proposed method, conducted the experiments, analyzed the results, prepared the figures and tables, and wrote the main manuscript text. J.N. and A.B. supervised the research, contributed to the conceptual and methodological development of the work, provided guidance throughout the study, and critically reviewed and edited the manuscript. A.B. acquired funding and provided project administration. All authors discussed the results, contributed to the final version of the manuscript, and approved the submitted version.

Data availability. The datasets, trained models, and code generated and/or analyzed during the current study are available from the corresponding author upon reasonable request.

Competing interests. The authors declare no competing interests.

## References

[1] Yaroslavsky, L.P.: Local adaptive filtering in transform domain for image restoration, enhancement, and target location. In: Wenger, E., Dimitrov, L.I. (eds.) Sixth International Workshop on Digital Image Processing and Computer Graphics: Applications in Humanities and Natural Sciences, vol. 3346, pp. 2–17. SPIE, San Jose, CA, USA (1998). https://doi.org/10.1117/12.301362 . International Society for Optics and Photonics

[2] Lim, J.S.: Two-Dimensional Signal and Image Processing. Prentice Hall, Englewood Clifs, NJ (1990)

[3] Dabov, K., Foi, A., Katkovnik, V., Egiazarian, K.: Image denoising by sparse 3- d transform-domain collaborative filtering. IEEE Transactions on Image Processing 16(8), 2080–2095 (2007) https://doi.org/10. 1109/TIP.2007.901238

[4] Zhang, K., Zuo, W., Chen, Y., Meng, D., Zhang, L.: Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE Transactions on Image Processing 26(7), 3142–3155 (2017) https://doi.org/10. 1109/TIP.2017.2662206

[5] Zhang, K., Zuo, W., Zhang, L.: Ffdnet: Toward a fast and flexible solution for cnnbased image denoising. IEEE Transactions on Image Processing 27(9), 4608–4622 (2018) https://doi.org/10.1109/TIP.2018.2839891

[6] Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.-H.: Restormer: Eficient transformer for high-resolution image restoration. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5718–5729 (2022). https: //doi.org/10.1109/CVPR52688.2022.00564

[7] Pl¨otz, T., Roth, S.: Neural nearest neighbors networks. In: Proceedings of the 32nd International Conference on Neural Information Processing Systems. NIPS’18, pp. 1095–1106. Curran Associates Inc., Red Hook, NY, USA (2018)

[8] Wang, X., Girshick, R., Gupta, A., He, K.: Non-local neural networks. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7794–7803 (2018). https://doi.org/10.1109/CVPR.2018. 00813

[9] Ye, J., Xiong, F., Zhou, J., Qian, Y.: Iterative low-rank network for hyperspectral image denoising. IEEE Transactions on Geoscience and Remote Sensing 62, 1–15 (2024) https: //doi.org/10.1109/TGRS.2024.3449130

[10] Herbreteau, S., Kervrann, C.: Dct2net: An interpretable shallow cnn for image denoising. IEEE Transactions on Image Processing 31, 4292–4305 (2022) https://doi.org/10. 1109/TIP.2022.3181488

[11] Cherel, N., Almansa, A., Gousseau, Y., Newson, A.: Patch-based stochastic attention for image editing. Computer Vision and Image Understanding 238, 103866 (2024) https:// doi.org/10.1016/j.cviu.2023.103866

[12] Donoho, D.L., Johnstone, I.M.: Adapting to unknown smoothness via wavelet shrinkage. Journal of the American Statistical Association 90(432), 1200–1224 (1995)

[13] Aharon, M., Elad, M., Bruckstein, A.: Ksvd: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Transactions on Signal Processing 54(11), 4311–4322 (2006) https://doi.org/10.1109/ TSP.2006.881199

[14] Mairal, J., Bach, F., Ponce, J., Sapiro, G.: Online dictionary learning for sparse coding. In: Proceedings of the 26th Annual International Conference on Machine Learning. ICML ’09, pp. 689–696. Association for Computing Machinery, New York, NY, USA (2009). https://doi.org/10.1145/ 1553374.1553463

[15] Buades, A., Coll, B., Morel, J.-M.: A nonlocal algorithm for image denoising. In: 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 2, pp. 60–652 (2005). https: //doi.org/10.1109/CVPR.2005.38

[16] Lebrun, M., Buades, A., Morel, J.M.: A nonlocal bayesian image denoising algorithm. SIAM Journal on Imaging Sciences 6(3), 1665–1688 (2013) https://doi.org/10.1137/ 120874989

[17] Xie, Q., Meng, D., Gu, S., Zhang, L., Zuo, W., Feng, X., Xu, Z.: On the optimal solution of weighted nuclear norm minimization. arXiv preprint arXiv:1405.6012 (2014)

[18] Gu, S., Zhang, L., Zuo, W., Feng, X.: Weighted nuclear norm minimization with application to image denoising. In: 2014 IEEE Conference on Computer Vision and Pattern Recognition, pp. 2862–2869 (2014). https:// doi.org/10.1109/CVPR.2014.366

[19] Zoran, D., Weiss, Y.: From learning models of natural image patches to whole image restoration. In: 2011 International Conference on Computer Vision, pp. 479–486 (2011). https://doi.org/10.1109/ICCV.2011. 6126278

[20] Lebrun, M.: An Analysis and Implementation of the BM3D Image Denoising Method. Image Processing On Line 2, 175–213 (2012) https://doi.org/10.5201/ipol.2012.l-bm3d

[21] Dabov, K., Foi, A., Katkovnik, V., Egiazarian, K.: Bm3d image denoising with shape-adaptive principal component analysis. Proc. Workshop on Signal Processing with Adaptive Sparse Structured Representations (SPARS’09) (2009)

[22] Maggioni, M., Katkovnik, V., Egiazarian, K., Foi, A.: Nonlocal transform-domain filter for volumetric data denoising and reconstruction. IEEE Transactions on Image Processing 22(1), 119–133 (2013) https://doi.org/10. 1109/TIP.2012.2210725

[23] Burger, H.C., Schuler, C.J., Harmeling, S.: Image denoising: Can plain neural networks compete with bm3d? In: 2012 IEEE Conference on Computer Vision and Pattern Recognition, pp. 2392–2399 (2012). https:// doi.org/10.1109/CVPR.2012.6247952

[24] Tai, Y., Yang, J., Liu, X., Xu, C.: Memnet: A persistent memory network for image restoration. In: 2017 IEEE International Conference on Computer Vision (ICCV), pp. 4549–4557 (2017). https://doi.org/10.1109/ICCV.2017. 486

[25] Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., Aila, T.: Noise2Noise: Learning image restoration without clean data. In: Dy, J., Krause, A. (eds.) Proceedings of the 35th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 80, pp. 2965– 2974. PMLR, Stockholm, Sweden (2018). https://proceedings.mlr.press/v80/lehtinen18a.html

[26] Krull, A., Buchholz, T.-O., Jug, F.: Noise2void - learning denoising from single noisy images. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2124–2132 (2019). https://doi.org/10.1109/CVPR.2019.00223

[27] Batson, J., Royer, L.: Noise2Self: Blind denoising by self-supervision. In: Chaudhuri, K., Salakhutdinov, R. (eds.) Proceedings of the 36th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 97, pp. 524–533. PMLR, Long Beach, California, USA (2019). https://proceedings.mlr.press/v97/batson19a.ht

[28] Ronneberger, O., Fischer, P., Brox, T.: Unet: Convolutional networks for biomedical image segmentation. In: Navab, N., Hornegger, J., Wells, W.M., Frangi, A.F. (eds.) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pp. 234–241. Springer, Cham (2015)

[29] Anwar, S., Barnes, N.: Real image denoising with feature attention. In: 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3155–3164 (2019). https://doi.org/10.1109/ICCV.2019.00325

[30] Jia, F., Wong, W.H., Zeng, T.: Ddunet: Dense dense u-net with applications in image denoising. In: 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pp. 354–364 (2021). https://doi. org/10.1109/ICCVW54120.2021.00044

[31] Chang, M., Li, Q., Feng, H., Xu, Z.: Spatial-adaptive network for single image denoising. In: Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX, pp. 171–187. Springer, Berlin, Heidelberg (2020). https://doi.org/10.1007/ 978-3-030-58577-8 11

[32] Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.-H., Shao, L.: Multi-stage progressive image restoration. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14816–14826 (2021). https://doi.org/10. 1109/CVPR46437.2021.01458

[33] Chen, L., Lu, X., Zhang, J., Chu, X., Chen, C.: Hinet: Half instance normalization network for image restoration (2022) arXiv:2105.06086 [eess.IV]

[34] Chen, L., Chu, X., Zhang, X., Sun, J.: Simple baselines for image restoration. In: Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VII, pp. 17–33. Springer, Berlin, Heidelberg (2022). https://doi.org/ 10.1007/978-3-031-20071-7 2

[35] Zhang, K., Li, Y., Zuo, W., Zhang, L., Van Gool, L., Timofte, R.: Plugand-play image restoration with deep denoiser prior. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(10), 6360–6376 (2022) https: //doi.org/10.1109/TPAMI.2021.3088914

[36] Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pp. 1833–1844 (2021). https://doi.org/10.1109/ ICCVW54120.2021.00210

[37] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9992–10002 (2021). https://doi.org/10.1109/ ICCV48922.2021.00986

[38] Wang, Z., Cun, X., Bao, J., Zhou, W., Liu, J., Li, H.: Uformer: A general ushaped transformer for image restoration. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17662–17672 (2022). https://doi.org/10. 1109/CVPR52688.2022.01716

[39] Pei, X., Huang, Y., Su, W., Zhu, F., Liu, Q.: Fftformer: A spatial-frequency noise aware cnn-transformer for low light image enhancement. Knowledge-Based Systems 314, 113055 (2025) https://doi.org/10. 1016/j.knosys.2025.113055

[40] Liu, X., Hong, Y., Yin, Q., Zhang, S.: Dnt: Learning unsupervised denoising transformer from single noisy image. In: Proceedings of the 4th International Conference on Image Processing and Machine Vision. IPMV

’22, pp. 50–56. Association for Computing Machinery, New York, NY, USA (2022). https://doi.org/10.1145/3529446.3529455

[41] Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M.: Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(4), 4713–4726 (2023) https: //doi.org/10.1109/TPAMI.2022.3204461

[42] Tan, M., Le, Q.: EficientNet: Rethinking model scaling for convolutional neural networks. In: Chaudhuri, K., Salakhutdinov, R. (eds.) Proceedings of the 36th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 97, pp. 6105–6114. PMLR, Long Beach, California, USA (2019). https://proceedings.mlr.press/v97/tan19a.html

[43] Gregor, K., LeCun, Y.: Learning fast approximations of sparse coding. In: Proceedings of the 27th International Conference on International Conference on Machine Learning. ICML’10, pp. 399–406. Omnipress, Madison, WI, USA (2010)

[44] Yang, Y., Sun, J., Li, H., Xu, Z.: Admm-net: A deep learning approach for compressive sensing mri (2017) https://doi.org/10.48550/ arXiv.1705.06869

[45] Yang, Y., Sun, J., Li, H., Xu, Z.: Admmcsnet: A deep learning approach for image compressive sensing. IEEE Transactions on Pattern Analysis and Machine Intelligence 42(3), 521–538 (2020) https://doi.org/10. 1109/TPAMI.2018.2883941

[46] Chen, Y., Pock, T.: Trainable nonlinear reaction difusion: A flexible framework for fast and efective image restoration. IEEE Transactions on Pattern Analysis and Machine Intelligence 39(6), 1256–1272 (2017) https: //doi.org/10.1109/TPAMI.2016.2596743

[47] Venkatakrishnan, S.V., Bouman, C.A., Wohlberg, B.: Plug-and-play priors for model based reconstruction. In: 2013 IEEE

Global Conference on Signal and Information Processing, pp. 945–948 (2013). https: //doi.org/10.1109/GlobalSIP.2013.6737048

[48] Romano, Y., Elad, M., Milanfar, P.: The little engine that could: Regularization by denoising (red). SIAM Journal on Imaging Sciences 10(4), 1804–1844 (2017) https://doi.org/10. 1137/16M1102884

[49] Meinhardt, T., Moeller, M., Hazirbas, C., Cremers, D.: Learning proximal operators: Using denoising networks for regularizing inverse imaging problems. In: 2017 IEEE International Conference on Computer Vision (ICCV), pp. 1799–1808 (2017). https: //doi.org/10.1109/ICCV.2017.198

[50] Lempitsky, V., Vedaldi, A., Ulyanov, D.: Deep image prior. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9446–9454 (2018). https:// doi.org/10.1109/CVPR.2018.00984

[51] Yang, D., Sun, J.: Bm3d-net: A convolutional neural network for transform-domain collaborative filtering. IEEE Signal Processing Letters 25(1), 55–59 (2018) https://doi. org/10.1109/LSP.2017.2768660

[52] Ahn, B., Kim, Y., Park, G., Cho, N.I.: Block-matching convolutional neural network (bmcnn): Improving cnn-based denoising by block-matched inputs. In: 2018 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pp. 516–525 (2018). https: //doi.org/10.23919/APSIPA.2018.8659548

[53] Fu, J., Liu, J., Tian, H., Li, Y., Bao, Y., Fang, Z., Lu, H.: Dual attention network for scene segmentation. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3141–3149 (2019). https://doi.org/10.1109/CVPR.2019.00326

[54] Shao, F., Feng, X., Tian, S., Zhang, T.: A hierarchical low-rank denoising model for remote sensing images based on deep unfolding. Sensors 24(14) (2024) https://doi.org/ 10.3390/s24144574

[55] Su, J., Xu, B., Yin, H.: A survey of deep learning approaches to image restoration. Neurocomputing 487, 46–65 (2022) https://doi. org/10.1016/j.neucom.2022.02.046

[56] Elad, M., Kawar, B., Vaksman, G.: Image denoising: The deep learning revolution and beyond—a survey paper. SIAM Journal on Imaging Sciences 16(3), 1594–1654 (2023) https://doi.org/10.1137/23M1545859

[57] Agustsson, E., Timofte, R.: Ntire 2017 challenge on single image super-resolution: Dataset and study. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1122–1131 (2017). https://doi.org/10.1109/CVPRW.2017.150

[58] Lim, B., Son, S., Kim, H., Nah, S., Lee, K.M.: Enhanced deep residual networks for single image super-resolution. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1132–1140 (2017). https://doi.org/10.1109/ CVPRW.2017.151

[59] Ma, K., Duanmu, Z., Wu, Q., Wang, Z., Yong, H., Li, H., Zhang, L.: Waterloo exploration database: New challenges for image quality assessment models. IEEE Transactions on Image Processing 26(2), 1004–1016 (2017) https://doi.org/10.1109/TIP.2016.2631888

[60] Martin, D., Fowlkes, C., Tal, D., Malik, J.: A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In: Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, vol. 2, pp. 416–423 (2001). https: //doi.org/10.1109/ICCV.2001.937655

[61] Loui, A., Luo, J., Chang, S.-F., Ellis, D., Jiang, W., Kennedy, L., Lee, K., Yanagawa, A.: Kodak’s consumer video benchmark data set: concept definition and annotation. In: Proceedings of the International Workshop on Workshop on Multimedia Information Retrieval. MIR ’07, pp. 245–254. Association for Computing Machinery, New York, NY, USA (2007). https://doi.org/10.1145/ 1290082.1290117

[62] Wu, W., Liu, Z., Gueaieb, W., He, X.: Singleimage super-resolution based on markov random field and contourlet transform. Journal of Electronic Imaging - J ELECTRON IMAGING 20 (2011) https://doi.org/10. 1117/1.3580750

[63] Zeyde, R., Elad, M., Protter, M.: On single image scale-up using sparse-representations. In: Boissonnat, J.-D., Chenin, P., Cohen, A., Gout, C., Lyche, T., Mazure, M.-L., Schumaker, L. (eds.) Curves and Surfaces, pp. 711–730. Springer, Berlin, Heidelberg (2012)