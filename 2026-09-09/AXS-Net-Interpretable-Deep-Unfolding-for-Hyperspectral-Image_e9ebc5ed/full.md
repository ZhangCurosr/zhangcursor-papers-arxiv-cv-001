# AXS-Net: Interpretable Deep Unfolding for Hyperspectral Image Denoising via Spectral Basis Unmixing and Structured Noise Refinement<sup>⋆</sup>

Ziyi Guan, Jianping Zhang<sup>()</sup>, and Zheng Yang

Xiangtan University, Xiangtan, Hunan, China ziyiguanxtu@163.com, jpzhang@xtu.edu.cn, 202421511256@smail.xtu.edu.cn

Abstract. Hyperspectral images (HSIs) are often degraded by mixed noise, including band-dependent Gaussian perturbations and structured artifacts such as stripes, dead-lines, and impulse noise. Most deep denoisers regress the clean image directly, entangling signal and structured noise. We instead model HSI denoising as Y = AX+S+N, where AX is a low-rank spectral-subspace (unmixing) reconstruction, S is structured sparse noise and N is residual Gaussian noise. The resulting regularized optimization problem is unrolled into AXS-Net, a K-stage alternating proximal-point framework. Each stage combines an analytic spectralbasis gradient step, an SSX-Block proximal operator for abundance coefficients, and an SBlock proximal operator for the structured residual with column-consistent and sparse priors. This optimization correspondence exposes interpretable endmembers, abundance maps, and structurednoise estimates. Across ICVL, CAVE, and Harvard datasets and five noise configurations, the proposed AXS-Net achieves strong in-domain accuracy and competitive zero-shot transfer, with consistent gains across all five noise regimes on ICVL and Harvard. The recovered structurednoise closely follows the synthetic reference, and the recovered spectral basis is smooth and band-ordered rather than an arbitrary set of latent channels.

Keywords: Hyperspectral image denoising · Deep unrolling · Hyperspectral image decomposition · Structured noise separation.

## 1 Introduction

Hyperspectral images (HSIs) capture a dense spectrum at every spatial location and support remote sensing, material identification, and computational imaging. During acquisition, however, they are routinely degraded by mixed noise, consisting of band-dependent Gaussian perturbations and structured corruptions such as stripes, dead-lines, and impulse noise. These artifacts not only reduce visual quality but also distort per-pixel spectral characteristics, making mixed-noise removal a fundamental preprocessing problem for downstream tasks [21].

Classical model-based denoising methods restore HSIs with explicit handcrafted priors such as low-rankness or total-variation terms [21, 9] and non-local self-similarity [16], optionally combined with global spectral low-rankness [8]. They are interpretable and can separate structured terms, but fixed priors and iterative solvers limit accuracy and eficiency. Learning-based methods learn stronger spatial–spectral representations [13, 7] by using sparse-coding and modelaided designs [3, 19], transformers [13], and channel/spectral attention [10], yet most regress a single clean image, entangling the signal with structured noise and ofering no explicit handle on stripes or impulses. Deep unfolding embeds iterative solver steps of the optimization problem into trainable networks [22], combining model structure with learned capacity. However, existing unfolded denoisers still optimize a single clean-image variable, leaving the structured corruption implicit. Classical low-rank-plus-sparse formulations usually rely on fixed sparse outlier priors and do not provide a learned, separately inspectable structured-noise branch coupled with an unmixing-style clean component.

In this work, we unroll the alternating minimization of a hyperspectral lowrank-plus-sparse decomposition into AXS-Net, whose stages comprise an analytic spectral-basis update, an SSX-Block for abundance estimation, and an SBlock for structured-residual refinement, rather than producing only a denoised image from observation Y. Our contributions are summarized as follows.

We formulate HSI mixed-noise removal as $\mathbf { Y } = \mathbf { A } \mathbf { X } + \mathbf { S } + \mathbf { N }$ and unroll its alternating proximal solution into an interpretable K-stage network, separating an unmixing-style clean component AX from a dedicated structurednoise estimate S instead of regressing a single clean image.

We design the SSX-Block, which integrates window-based spatial attention with cross-spectral attention to refine abundances, and the SBlock, which fuses directional residual features with parameter-free column-consistency and sparsity priors plus adaptive thresholding to separate structured noise. On the ICVL, CAVE, and Harvard datasets with Gaussian, stripe, deadline, impulse, and mixed noise, our AXS-Net achieves top performance on in-domain ICVL and generalizes efectively to the unseen CAVE and Harvard datasets, with the largest margins on Harvard. At the same time, its recovered structured noise closely follows the synthetic ground truth, and the stage-wise decomposition remains spectrally organized and interpretable throughout the unrolled iterations.

## 2 Methodology

## 2.1 Problem Formulation

Observation model. Let $\mathbf { Y } _ { o } \in \mathbb { R } ^ { s \times h \times w }$ denote the observed HSI with s spectral bands and spatial size $h \times w$ . Since the spectra of a natural scene are highly correlated across bands, the clean signal can be represented by a small number of shared spectral signatures. Following the linear spectral mixing model [2], we assume that the underlying clean HSI lies in a low-dimensional spectral subspace, i.e., each clean pixel can be approximated by a non-negative combination of $r \ll s$ representative spectra, or endmembers. Whenever matrix factorization is involved, we flatten the tensor ${ \bf Y } _ { o }$ spatially into $\mathbf { Y } \in \mathbb { R } ^ { s \times n }$ with $n = h w$ , and model the observation as

$$
\mathbf { Y } = \mathbf { A } \mathbf { X } + \mathbf { S } + \mathbf { N } ,\tag{1}
$$

where $\textbf { A } \in \mathbb { R } ^ { s \times r }$ is the spectral basis, or endmember matrix whose columns are the shared spectra, $\mathbf { X } \in \mathbb { R } ^ { r \times n }$ is the abundance matrix collecting the pixel mixing coeficients, S denotes structured sparse noise that captures corruptions such as stripes, dead-lines, and impulses, and N denotes residual zero-mean Gaussian noise. This three-component decomposition [21] assigns a distinct role to each term: the low-rank product AX reconstructs the clean image in an unmixing style and is taken as the denoised output, while S is explicitly retained rather than being absorbed into a single regression output, so that structured corruptions are exposed as an interpretable and separately inspectable estimate.

Constrained optimization. Given this model, we recover A, X, and S by minimizing the residual of reconstruction under suitable regularization priors:

$$
\begin{array} { r l } { \underset { { \bf A } , { \bf X } , { \bf S } } { \operatorname* { m i n } } } & { \frac { 1 } { 2 } \left\| { \bf Y } - { \bf A } { \bf X } - { \bf S } \right\| _ { F } ^ { 2 } + \lambda _ { X } \mathcal { R } _ { X } ( { \bf X } ) + \lambda _ { S } \mathcal { R } _ { S } ( { \bf S } ) , } \\ { { \bf s . t . } } & { { \bf A } \geq 0 , \qquad \left\| { \bf A } _ { : , j } \right\| _ { 2 } = 1 , \quad j = 1 , \ldots , r , \qquad { \bf X } \geq 0 . } \end{array}\tag{2}
$$

The quadratic term $\begin{array} { r } { { \frac { 1 } { 2 } } \left\| \mathbf { Y } - \mathbf { A } \mathbf { X } - \mathbf { S } \right\| _ { F } ^ { 2 } } \end{array}$ serves as the data-fidelity term induced by Gaussian noise N, while the priors $\mathcal { R } _ { X }$ and $\mathcal { R } _ { S }$ , weighted by $\lambda _ { X }$ and $\lambda _ { S } ,$ regularize the abundances and the structured noise, respectively. The constraints make the factorization physically meaningful. Specifically, $\mathbf A \ge 0$ together with the $\mathrm { \ u n i t { - } } \ell _ { 2 }$ column normalization $\| \mathbf { A } _ { : , j } \| _ { 2 } = 1$ forces each endmember to be a non-negative and scale-fixed spectral basis, as in non-negative matrix factorization [14]. The constraint $\mathbf { X } \geq 0$ keeps the abundances non-negative, consistent with their interpretation as mixing proportions.

Learned proximal operators and initialization. The task-specific regularizers $\mathcal { R } _ { X }$ and $\mathcal { R } _ { S }$ are dificult to specify in closed form: abundance maps exhibit complex spatial–spectral structures, and structured noise may contain multiple corruption types with unknown statistics. We therefore keep the proximal-gradient structure of the solver but replace the two proximal operators $\mathrm { p r o x } _ { \lambda _ { X } \mathcal { R } _ { X } }$ and $\mathrm { p r o x } _ { \lambda _ { S } \mathcal { R } _ { S } }$ with learnable modules trained end-to-end (Sec. 2.2), where the regularization weights $\lambda _ { X } , \lambda _ { S }$ are absorbed into these modules. The proximal step $\begin{array} { r } { \mathrm { p r o x } _ { \lambda \mathcal { R } } ( \mathbf { v } ) = \arg \operatorname* { m i n } _ { \mathbf { u } } \frac { 1 } { 2 } \| \mathbf { u } - \mathbf { v } \| _ { F } ^ { 2 } + \lambda \mathcal { R } ( \mathbf { u } ) } \end{array}$ can be interpreted as a denoising or MAP subproblem, whose solution is refined by the prior R. Because the proximal mappings $\mathrm { p r o x } _ { \lambda \mathcal { R } } ( \mathbf { v } )$ associated with the task-specific priors $\mathcal { R } _ { X }$ and $\mathcal { R } _ { S }$ do not admit simple closed-form expressions, we adopt the deep unfolding approach [22] and represent these proximal operators using neural networks (SSX-Block and SBlock). At the same time, we retain the analytic forms for the decomposition components, including the A-step and the residual $\mathbf { R } ^ { k }$ . To start the alternating scheme from a subspace-informed point, we construct ${ \bf A } _ { 0 }$ directly from the data. Specifically, we first mean-center Y across the spatial dimension, then perform an SVD to extract the top-r left singular vectors and use these leading spectral components as an initial basis. This basis is subsequently mapped onto the feasible set via a non-negativity projection followed by column-wise $\ell _ { 2 }$ normalization to the unit norm, resulting in ${ \bf A } ^ { 0 }$ . The remaining variables are initialized as $\mathbf { X } ^ { 0 } = \operatorname { R e L U } ( \mathbf { A } ^ { 0 ^ { \top } } \mathbf { Y } )$ and $\mathbf { S } ^ { 0 } = \mathbf { 0 }$ . These quantities are progressively refined by the unrolled stages described below.

![](images/5174f01435c7ad08d39198b25b5350c8c40e8fe7f1b1f9c43c4402b76d5a23cd.jpg)  
Fig. 1. Overview of AXS-Net. From a PCA (mean-centered SVD) subspace initialization, K=6 unrolled stages alternately update the spectral basis A, abundances X (SSX-Block) and structured noise S (SBlock), extracting the clean component AX and S from Y.

## 2.2 Overall Unfolding Framework

From optimization to a fixed-depth network. Problem (2) is a non-convex constrained optimization task for two main reasons: first, the data-fidelity term couples A and X through the bilinear product AX, which makes the objective jointly non-convex in (A, X), although it remains convex in each variable when the other is fixed; second, the unit-ℓ<sub>2</sub> equality constraint $\| \mathbf { A } _ { : , j } \| _ { 2 } = 1$ confines A to a non-convex (spherical) feasible set. A natural strategy is therefore to use alternating minimization, updating one variable at a time while keeping the others fixed; each of the resulting subproblems is substantially simpler and can be solved with a straightforward projected-gradient or proximal step [6]. Instead of running such an iterative algorithm to convergence, which is computationally expensive and still depends on manually designed regularizers, we unroll a fixed number K of alternating updates into a feed-forward network and learn the operators at each iteration directly from data [22]. We choose K=6 as a compromise between restoration performance and computational cost, since every additional stage introduces one more unrolled pass and thus extra latency. All K stages share the same architectural design but have distinct parameters for each stage, allowing them to specialize instead of repeatedly applying a single fixed update.

Stage update. At stage k, the variables are refined in order $\mathbf { A } \to \mathbf { X } \to \mathbf { S }$

$$
\begin{array} { r l r } & { \mathbf { A } ^ { k } = \mathcal { A } ^ { ( k ) } \left( \mathbf { Y } , \mathbf { A } ^ { k - 1 } , \mathbf { X } ^ { k - 1 } , \mathbf { S } ^ { k - 1 } \right) , } \\ & { \mathbf { X } ^ { k } = \mathrm { p r o x } _ { \lambda _ { X } \mathcal { R } _ { X } } \left( \mathbf { V } ^ { k } , \mathbf { E } ^ { k } , \mathbf { X } ^ { k - 1 } , \mathbf { Z } ^ { k } \right) } & { ( \mathrm { S S X - B l o c k } ) , } \\ & { \mathbf { S } ^ { k } = \mathrm { p r o x } _ { \lambda _ { S } \mathcal { R } _ { S } } \left( \mathbf { R } ^ { k } , \mathbf { S } ^ { k - 1 } \right) , \quad \mathbf { R } ^ { k } = \mathbf { Y } - \mathbf { A } ^ { k } \mathbf { X } ^ { k } . } \end{array}\tag{3}
$$

The A-update remains an analytic gradient step, whereas SSX-Block and SBlock learn the proximal updates for X and $\mathbf { S } ,$ respectively (Secs. 2.3–2.5), preserving the model structure while learning priors that are dificult to specify explicitly.

Cross-variable coupling. Beyond the sequential $\mathbf { A } \to \mathbf { X } \to \mathbf { \pmb { \iota } }$ S dependency within each stage, two explicit couplings link the variables across the unrolling and are drawn as dashed arrows in Fig. 1: (i) the current structured-noise estimate $\mathbf { S } ^ { k - 1 }$ is subtracted when forming the coeficient-space residual $\mathbf { E } ^ { k }$ , so that abundance refinement is not contaminated by structured corruptions; and (ii) a Nesterov warm start carries momentum from $( \mathbf { X } ^ { k - 1 } - \mathbf { X } ^ { k - 2 } )$ into $\mathbf { Z } ^ { k }$ , accelerating the abundance update across stages. Together these turn the K stages into a single diferentiable network trained end-to-end.

## 2.3 Endmember Update: Analytic Spectral Basis Step

The endmember update A corresponds to a single unfolded projected-gradient iteration applied to the smooth data-fidelity term. In contrast to the abundance and structured-noise updates, which rely on learned proximal modules, this update employs the closed-form gradient with respect to A. Starting from $\mathbf { A } ^ { k - 1 }$ ， ${ \bf X } ^ { k - 1 }$ , and $\mathbf { \check { S } } ^ { k - 1 }$ , we evaluate the normalized gradient

$$
\mathbf { G } _ { A } ^ { k } = { \frac { 1 } { n } } ( \mathbf { A } ^ { k - 1 } \mathbf { X } ^ { k - 1 } + \mathbf { S } ^ { k - 1 } - \mathbf { Y } ) \mathbf { X } ^ { k - 1 } { \overset {  } { , } }\tag{4}
$$

which corresponds to the gradient of the quadratic data term in $\operatorname { E q . } \left( 2 \right)$ . We next perform a gradient descent update and project the result back onto the feasible set:

$$
\mathbf { A } ^ { k } = \operatorname { N o r m C o l } \left( \mathbf { \Pi } \mathbf { I } _ { + } \left( \mathbf { A } ^ { k - 1 } - \alpha _ { k } \mathbf { G } _ { A } ^ { k } \right) \right) ,\tag{5}
$$

where the projection operator $\Pi _ { + }$ sets all negative components to zero, and NormCol normalizes each column to have unit $\ell _ { 2 }$ norm. Together, these operations guarantee that A satisfies both the non-negativity and unit-norm constraints. The step size is stage-dependent and learned, with the parameterization $\alpha _ { k } = \mathrm { s o f t p l u s } ( \eta _ { k } )$ enforcing positivity.

Since this subproblem is smooth, we choose to keep the A-update analytical and parameter-light instead of learning it, which preserves the interpretation of the endmember A. The A-update is kept fixed during the first 10 epochs, when the initial abundance estimates are still unstable, and activated thereafter.

![](images/3c89324d40b8267bf83bf49c88f93b22dea75f593471d83388829b40a8a24d60.jpg)  
Fig. 2. SSX-Block for the abundance update. Inputs $[ { \bf V } , { \bf E } , { \bf X } _ { \mathrm { p r e v } } , { \bf Z } ]$ are projected to width C, refined by window spatial attention, spectral attention, and an FFN, then output as $\mathbf { X } ^ { k } = \operatorname { R e L U } ( \mathbf { V } + \varDelta \mathbf { X } )$

## 2.4 Abundance Update: Spatial–Spectral Proximal Block

After updating the spectral basis, we refine the abundance coeficients using a learned proximal operator applied to the gradient-corrected estimate $\mathbf { V } ^ { k }$ Given $\mathbf { A } ^ { k } , \mathbf { X } ^ { k - 1 }$ , and $\bar { \mathbf { S } } ^ { k - 1 }$ , we first compute the back-projected residual $\mathbf { E } ^ { k }$ , a gradient-corrected iterate $\mathbf { V } ^ { k }$ , and an extrapolated momentum feature $\mathbf { Z } ^ { k }$ as:

$$
\mathbf { E } ^ { k } = \mathbf { A } ^ { k ^ { \top } } \left( \mathbf { Y } - \mathbf { A } ^ { k } \mathbf { X } ^ { k - 1 } - \mathbf { S } ^ { k - 1 } \right) ,\tag{6}
$$

$$
\mathbf { V } ^ { k } = \mathbf { X } ^ { k - 1 } + \beta _ { k } \mathbf { E } ^ { k } , \quad \mathbf { Z } ^ { k } = \mathbf { X } ^ { k - 1 } + \gamma _ { k } \left( \mathbf { X } ^ { k - 1 } - \mathbf { X } ^ { k - 2 } \right) ,\tag{7}
$$

where $\beta _ { k } = \mathrm { s o f t p l u s } ( \eta _ { k } ^ { X } )$ and $\gamma _ { k } = \sigma ( \rho _ { k } ^ { X } )$ are learnable stage-dependent scalars. After reshaping the coeficient maps, these four tensors are concatenated into a 4r-channel input and linearly projected to a hidden dimension C:

$$
\begin{array} { r } { \mathbf { H } _ { 0 } = \operatorname { C o n v } _ { 1 \times 1 } \big ( \operatorname { C o n c a t } ( \mathbf { V } ^ { k } , \mathbf { E } ^ { k } , \mathbf { X } ^ { k - 1 } , \mathbf { Z } ^ { k } ) \big ) . } \end{array}\tag{8}
$$

We realize prox $\cdot \lambda _ { X } \mathcal { R } _ { X }$ using a separable spatial–spectral attention module (Fig. 2), motivated by the fact that abundance maps exhibit both spatial selfsimilarity and inter-component correlations. This module consists of three prenormalized residual sub-layers, followed by a nonnegative residual readout:

$$
\mathbf { H } _ { 1 } = \mathbf { H } _ { 0 } + \mathrm { W S A } \big ( \mathrm { G N } ( \mathbf { H } _ { 0 } ) \big ) , \quad \mathbf { H } _ { 2 } = \mathbf { H } _ { 1 } + \mathrm { S S A } \big ( \mathrm { G N } ( \mathbf { H } _ { 1 } ) \big ) ,\tag{9}
$$

$$
\mathbf { H } _ { 3 } = \mathbf { H } _ { 2 } + \mathrm { F F N } \big ( \mathrm { G N } ( \mathbf { H } _ { 2 } ) \big ) , \quad \mathbf { X } ^ { k } = \mathrm { R e L U } \big ( \mathbf { V } ^ { k } + \mathrm { C o n v } _ { 1 \times 1 } ( \mathbf { H } _ { 3 } ) \big ) ,\tag{10}
$$

where GN denotes GroupNorm and FFN is implemented as a 1 × 1–GELU layer followed by a 1 × 1 MLP with expansion ratio 2. Adopting a Swin-style attention mechanism [15], the window spatial attention (WSA) divides the feature tensor into non-overlapping $1 6 \times 1 6$ windows and performs 4-head self-attention with relative-position bias in each window. This design captures local spatial relationships in the abundance maps while keeping the linear complexity in the number of windows. In analogy to channel/spectral attention methods [10], the spectral attention (SSA) first forms a descriptor for each channel through global average and max pooling, $\mathbf { d } _ { c } = [ \mathrm { G A P } ( \mathbf { H } _ { c } ) , \mathrm { G M P } ( \mathbf { H } _ { c } ) ]$ , where $\mathbf { H } _ { c }$ is the feature map of channel c. These descriptors are then treated as tokens, and multi-head self-attention is performed across the C channel tokens to explicitly capture inter-channel dependencies. The output is a sigmoid gate g that reweights the feature map as H ⊙ g, while this gating is derived from explicit cross-channel attention, and its computational cost does not depend on spatial resolution.

![](images/6d5bcb8d76a87352c74eb69aebda6e49f143e27bc25e0f65d7635aba40464ddc.jpg)  
Fig. 3. SBlock for the S-update. Directional and local branches process $\mathbf { R } ^ { k } = \mathbf { Y } -$ $\mathbf { A } ^ { k } \mathbf { X } ^ { k }$ , combine with column-consistent and sparse priors, and apply adaptive proximal thresholding to extract $\mathbf { S } ^ { k }$

## 2.5 Structured-noise Update: Structure-Aware Residual Block

The structured-noise update applies $\mathbf { S } ^ { k } = \operatorname { p r o x } _ { \lambda _ { S } \mathcal { R } _ { S } } ( \mathbf { R } ^ { k } )$ to the reconstruction residual $\mathbf { R } ^ { k } = \mathbf { Y } - \mathbf { A } ^ { k } \mathbf { X } ^ { k }$ . SBlock $\left( \mathrm { F i g . \ 3 } \right)$ first maps $\mathrm { C o n c a t } ( \mathbf { R } ^ { k } , \mathbf { S } ^ { k - 1 } )$ to width C and extracts two types of residual features. A directional branch uses depthwise $1 \times 7$ and $7 \times 1$ convolutions to capture axis-aligned stripe and deadline structures, while a local branch uses a $3 \times 3$ convolution to preserve local texture. The branches are fused and added to a learnable residual skip of $\mathbf { R } ^ { k }$

$$
\widetilde { \mathbf { S } } ^ { k } = \mathrm { F u s e } ( \mathrm { D i r } ( \cdot ) , \mathrm { L o c } ( \cdot ) ) + \rho _ { k } \mathbf { R } ^ { k } ,\tag{11}
$$

where $\rho _ { k }$ is initialized to 0.05.

On top of the learned residual features, we employ two parameter-free structured priors $\mathcal { P } _ { \mathrm { s t r i p e } } ( \mathbf { R } )$ and ${ \mathcal { P } } _ { \mathrm { i m p } } ( \mathbf { R } )$ to remove the stripe/dead-line noise and impulse noise, respectively. The Column-consistency prior and the Median-residual

prior, motivated by [4] and [11], respectively, are given by

$$
\begin{array} { r l } & { \mathcal { P } _ { \mathrm { s t r i p e } } ( { \bf R } ) = \mathrm { E x p a n d } ( \mathrm { M e a n } _ { h } ( { \bf R } ) - \mathrm { M e a n } _ { w } ( \mathrm { M e a n } _ { h } ( { \bf R } ) ) ) , } \\ & { \quad \mathcal { P } _ { \mathrm { i m p } } ( { \bf R } ) = \mathrm { T o p K } _ { \kappa } ( { \bf R } - \mathrm { M e d i a n } _ { 5 \times 5 } ( { \bf R } ) ) } \end{array}
$$

where ${ \mathrm { M e a n } } _ { h }$ averages along the vertical dimension, $\mathrm { M e a n } _ { w }$ removes the bandwise global column ofset, and the keep ratio $\kappa = 0 . 1 0$ . The learned estimate and the two priors are combined through learnable non-negative scales $\delta _ { s }$ and $\delta _ { i } , \mathrm { i . e . }$ 2

$$
\mathbf { U } ^ { k } = \widetilde { \mathbf { S } } ^ { k } + \delta _ { s } \mathcal { P } _ { \mathrm { s t r i p e } } ( \mathbf { R } ^ { k } ) + \delta _ { i } \mathcal { P } _ { \mathrm { i m p } } ( \mathbf { R } ^ { k } ) .\tag{12}
$$

Finally, SBlock applies an adaptive proximal-style threshold to sparsify the structured residual:

$$
{ \mathbf { S } } ^ { k } = { \mathrm { S m o o t h T h r } } \left( { \mathbf { U } } ^ { k } ; \tau _ { k } ( { \mathbf { R } } ^ { k } ) \right) , \qquad \tau _ { k } ( { \mathbf { R } } ^ { k } ) = \tau _ { \mathrm { m i n } } + \tau _ { \mathrm { r a n g e } } \cdot { \mathrm { M L P } } \left( \overline { { | { \mathbf { R } } ^ { k } | } } \right)\tag{13}
$$

where $\overline { { | \mathbf { R } ^ { k } | } }$ is the band-wise spatial average of $| \mathbf { R } ^ { k } |$ and $\tau _ { k } \in [ 0 . 0 0 5 , 0 . 0 4 0 ]$ . The smooth-thresholding operator acts as a data-adaptive soft-shrinkage step [6], encouraging $\mathbf { S } ^ { k }$ to contain sparse structured corruptions rather than clean image content.

## 2.6 Training Loss

For clean HSI $\mathbf { C } ^ { \mathrm { { g t } } }$ and synthetic structured noise $\mathbf { S } ^ { \mathrm { g t } }$ , we train AXS-Net with deep supervision over all unrolled stages:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \sum _ { k = 1 } ^ { K } \frac { k } { K } \left( w _ { c } \mathcal { L } _ { \mathrm { c l e a n } } ^ { k } + w _ { m } \mathcal { L } _ { \mathrm { s a m } } ^ { k } + w _ { s } \mathcal { L } _ { S } ^ { k } \right) + w _ { o } \mathcal { L } _ { \mathrm { c o n s } } ^ { K } ,
$$

where stage weight $k / K$ places stronger emphasis on later stages, and stage-wise losses are defined as

$$
\begin{array} { r l r l } & { \mathcal { L } _ { \mathrm { c l e a n } } ^ { k } = \left\| \mathbf { A } ^ { k } \mathbf { X } ^ { k } - \mathbf { C } ^ { \mathrm { g t } } \right\| _ { 1 } , } & & { \mathcal { L } _ { S } ^ { k } = \left\| \mathbf { S } ^ { k } - \mathbf { S } ^ { \mathrm { g t } } \right\| _ { 1 } , } \\ & { \mathcal { L } _ { \mathrm { s a m } } ^ { k } = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname { a r c c o s } \frac { \left. \left( \mathbf { A } ^ { k } \mathbf { X } ^ { k } \right) _ { i } , \mathbf { C } _ { i } ^ { \mathrm { g t } } \right. } { \left\| \left( \mathbf { A } ^ { k } \mathbf { X } ^ { k } \right) _ { i } \right\| _ { 2 } \left\| \mathbf { C } _ { i } ^ { \mathrm { g t } } \right\| _ { 2 } } , } & & { \mathcal { L } _ { \mathrm { c o n s } } ^ { K } = \left\| \mathbf { Y } - \mathbf { A } ^ { K } \mathbf { X } ^ { K } - \mathbf { S } ^ { K } - \mathbf { N } ^ { \mathrm { g t } } \right\| _ { F } ^ { 2 } . } \end{array}
$$

Here $\mathcal { L } _ { \mathrm { c l e a n } }$ supervises the reconstructed clean image in intensity space, and $\mathcal { L } _ { \mathrm { s a m } }$ preserves per-pixel spectral fidelity through the spectral angle [12]. The term $\mathcal { L } _ { S }$ directly supervises the structured-noise branch, while ${ \mathcal { L } } _ { \mathrm { c o n s } }$ encourages the final decomposition to match the observation model in $\operatorname { E q . } \ ( 1 )$ . Under the synthetic-noise protocol, $\mathbf { S } ^ { \mathrm { g t } }$ and $\mathbf { N } ^ { \mathrm { g t } }$ are available from the noise-generation process. When ground-truth noise components are unavailable, we omit $\mathcal { L } _ { S }$ and ${ \mathcal { L } } _ { \mathrm { c o n s } }$ and train only with the clean-image supervision. Unless otherwise stated, we set $w _ { c } = 1 . 3 , w _ { m } = 0 . 2 , w _ { s } = 0 . 1$ , and ${ w _ { o } } \mathrm { = } 0 . 1$

## 3 Experiments

## 3.1 Experimental Setup

Datasets and noise. We train our models on ICVL [1] and evaluate on ICVL (50 test scenes), CAVE [20] (32 scenes), and Harvard [5] (26 scenes), where CAVE and Harvard are used as zero-shot transfer benchmarks. All hyperspectral tensors consist of 31 bands. We construct five noise settings: (c1) Gaussian, (c2) Gaussian+stripe, (c3) Gaussian+dead-line, (c4) Gaussian+impulse, and (c5) a mixture of all four, and use the same fixed noisy cubes for all methods.

Compared methods. We conduct a quantitative comparison using MPSNR, MSSIM [18], SAM [12], ERGAS [17], and RMSE for several recent learning-based HSI denoisers under a unified inference setting: T3SC [3], MAC-Net [19], HSDT [13], and SSUMamba [7]. Classical model-based methods (BM4D [16], LRMR [21], NGMeet [8]) target diferent noise assumptions and are not directly comparable under this unified protocol, so we concentrate the comparison on recent learned denoisers. All methods are evaluated on the same set of pre-generated noisy cubes (seed 2026), using identical normalization procedures and metrics. For the learned baselines, we rely on the oficial model weights and their native inference pipelines, without any additional retuning or test-time augmentation. SSUMamba results are taken from archived predictions on the same fixed cubes, generated by the oficially released SSUMamba (SSCS) checkpoint without any post-processing or test-time augmentation; re-evaluation in our environment is blocked by an API incompatibility between BiMamba and mamba\_ssm.

Implementation. The architecture is configured as described in Secs. 2.2–2.5, and the training objective follows Sec. 2.6. We choose a rank of r=8, a hidden width of C=48, and 4 attention heads. Training is carried out for 600 epochs using Adam (learning rate $3 \times 1 0 ^ { - 5 }$ , decreased to $3 \times 1 0 ^ { - 6 }$ during the A-step) on 64 × 64 patches, with a batch size of 4 and random seed 2026. We employ weight EMA with a decay factor of 0.9999 and select the final model according to the best validation MPSNR, using a single NVIDIA A100 GPU. For sliding-window inference, we adopt a 64 × 64 window with a stride of 24.

## 3.2 Comparison with State-of-the-Art

On the ICVL dataset (Table 1), the proposed AXS-Net achieves the best performance on all five metrics under every noise configuration and improves MPSNR over SSUMamba by 1.94/3.56/3.05 dB for stripe/dead-line/impulse noise, respectively. In zero-shot transfer (Table 2), it ranks first on all Harvard metrics and all CAVE MSSIM scores, while SSUMamba still achieves a higher MP-SNR in CAVE stripe and mixture noise. Figs. 4 and 5 illustrate that AXS-Net produces fewer structured residuals and smaller errors in the shown in-domain and zero-shot scenes. Paired per-scene significance tests (Wilcoxon signed-rank over n=50/32/26 scenes for ICVL/CAVE/Harvard, Benjamini–Hochberg FDR at α=0.05; five noise settings × four baselines × six metrics—the five reported in Tables 1–2 plus the correlation coeficient—= 120 tests per dataset) give 107 wins / 8 ties / 5 losses on ICVL and $1 0 6 / 9 / 5$ on Harvard, whereas the record on CAVE is more balanced at 62/29/29, consistent with the weaker CAVE MP-SNR transfer noted above. Ties denote diferences that are not significant after FDR correction; because the tables report means while these tests are paired per scene, a better mean does not always translate into a significant win.

Table 1. ICVL results (50 scenes). Each cell: MPSNR↑/MSSIM↑/SAM↓ (top), ERGAS↓/RMSE↓ (bottom); bold=best.
<table><tr><td>Noise</td><td>AXS-Net (Ours)</td><td>SSUMamba</td><td>HSDT</td><td>T3SC</td><td>MAC-Net</td></tr><tr><td>Gaussian</td><td>41.69/0.967/5.61 14.32/0.0094</td><td>37.63/0.899/14.51 34.24/0.0141</td><td>36.17/0.831/10.73 93.90/0.0199</td><td>33.95/0.805/12.82 115.87/0.0258</td><td>34.56/0.803/13.22 113.10/0.0257</td></tr><tr><td>+Stripe</td><td>39.65/0.957/6.91</td><td>37.71/0.899/13.18</td><td>36.81/0.842/10.02</td><td>33.71/0.798/13.77</td><td>29.79/0.711/13.76</td></tr><tr><td>+Dead-line</td><td>19.67/0.0115 40.60/0.964/5.92</td><td>36.45/0.0136 37.04/0.902/14.09</td><td>83.17/0.0171 36.17/0.830/11.02</td><td>134.79/0.0261 33.58/0.801/13.49</td><td>150.25/0.0357 32.54/0.788/14.23</td></tr><tr><td></td><td>15.64/0.0107 41.38/0.964/5.85</td><td>32.50/0.0155 38.33/0.913/10.17</td><td>96.08/0.0199 35.74/0.826/10.40</td><td>118.90/0.0262 32.50/0.770/12.72</td><td>117.17/0.0285 32.14/0.760/12.89</td></tr><tr><td>+Impulse</td><td>14.75/0.0097 38.57/0.947/6.97</td><td>32.15/0.0133</td><td>96.72/0.0205</td><td>130.01/0.0290</td><td>130.24/0.0301</td></tr><tr><td>Mixture</td><td>21.56/0.0131</td><td>38.09/0.911/10.84 33.63/0.0134</td><td>36.64/0.845/9.61 76.43/0.0168</td><td>32.25/0.767/13.81 138.80/0.0285</td><td>28.91/0.688/13.84 156.41/0.0386</td></tr></table>

Table 2. Zero-shot cross-dataset results. ICVL-trained models are evaluated on CAVE (32) and Harvard (26). Each cell: MPSNR↑/MSSIM↑/SAM↓; bold=best.
<table><tr><td>Data</td><td>Noise</td><td>AXS-Net (Ours)</td><td>SSUMamba</td><td>HSDT</td><td>T3SC</td><td>MAC-Net</td></tr><tr><td rowspan="6">CAVE</td><td>Gaussian</td><td>34.30/0.909/16.55</td><td>33.69/0.842/29.22</td><td>33.27/0.745/ /19.47</td><td>30.30/0.706/ /21.23</td><td>32.24/0.718/22.65</td></tr><tr><td>+Stripe</td><td>32.40 0/0.867/24.20</td><td>33.76/0.828/ /27.15</td><td>33.46/0.752 /19.91</td><td>30.29/0.703/ /20.57</td><td>28.34/0.624/23.56</td></tr><tr><td>+Dead-line</td><td>33.51 /0.901/18.04</td><td>33.50/0.845/ /29.04</td><td>33.22 /0.745 /19.61</td><td>29.98/0.702 /21.59</td><td>30.41/0.703/23.45</td></tr><tr><td>+Impulse</td><td>34.06 /0.905 /16.73</td><td>33.91/0.832/ /27.05</td><td>32.97/0.739/ /19.51</td><td>29.55/0.676 /21.07</td><td>30.18/0.671/22.85</td></tr><tr><td>Mixture</td><td>31.80/0.854/25.32</td><td>33.66/0.821 /26.29</td><td>33.09/0.746/ /20.42</td><td>29.18/0.667 /21.01</td><td>27.26/0.594/24.14</td></tr><tr><td>Avg</td><td>33.21 /0.887 /20.17</td><td>33.70 /0.834/ /27.75</td><td>33.20 /0.745 /19.78</td><td>29.86 /0.691 /21.09</td><td>29.69/0.662/23.33</td></tr><tr><td rowspan="6">Harvard</td><td>Gaussian</td><td>41.40 /0.943 /5.74</td><td>35.53/0.881/14.96</td><td>35.24/0.824/ /10.30</td><td>32.48 /0.790 /14.60</td><td>34.51/0.791/13.52</td></tr><tr><td>+Stripe</td><td>37.57 /0.912 /9.77</td><td>34.96/0.870/14.37</td><td>35.14/0.827 /10.83</td><td>31.79/0.773/ /14.57</td><td>29.54/0.676/15.40</td></tr><tr><td>+Dead-line</td><td>40.19/0.936/6.40</td><td>35.60/0.881/15.11</td><td>35.25/0.824/10.52</td><td>32.60/0.792/14.77</td><td>33.68/0.784/14.23</td></tr><tr><td>+Impulse</td><td>41.23/0.942 /5.76</td><td>35.90/0.893/13.08</td><td>34.95/0.818/10.47</td><td>31.09/0.746/14.97</td><td>32.36/0.747/14.16</td></tr><tr><td>Mixture</td><td>36.11/0.896/11.09</td><td>35.28/0.880/13.32</td><td>34.91/0.822/11.26</td><td>30.85/0.739/15.19</td><td>28.96/0.660/16.24</td></tr><tr><td>Avg</td><td>39.30/0.926/7.75</td><td>35.45/0.881/14.17</td><td>35.10/0.823/10.68</td><td>31.76/0.768/14.82</td><td>31.81/0.732/14.71</td></tr></table>

Our AXS-Net contains only 1.80 M parameters, which is substantially fewer than SSUMamba’s 19.35 M, though still more than those of the lightweight HSDT, T3SC, and MAC-Net models (0.52/0.91/0.43 M). When processing a 1392 × 1300 × 31 tensor, it requires 14.42 s, which is slower than SSUMamba (7.00 s), HSDT (3.59 s), and T3SC (2.73 s), yet faster than MAC-Net (32.00 s), highlighting the computational overhead introduced by its six unrolled stages. These are deployment-time figures under each method’s native inference configuration rather than a hardware-normalized ranking of algorithmic complexity.

## 3.3 Interpretability and Structured-Noise Recovery

Fig. 5 (left) shows the decompositions AX, |S|, A, and a representative abundance map. The recovered bases are smooth and band-ordered, i.e., they vary gradually with wavelength, and the corresponding abundance maps preserve clear spatial structure, indicating that the subspace is spectrally organized rather than composed of arbitrary latent channels. Fig. 7 traces this across the unrolled stages: the reconstruction $\mathbf { A } ^ { k } \mathbf { X } ^ { k }$ sharpens from a coarse piecewise-constant estimate at k=1 to a faithful image at k=6, while the structured residual |S<sup>k</sup>| concentrates onto the column-consistent stripe pattern. We do not claim that the recovered endmembers correspond to identifiable physical materials.

![](images/a8f640f5d63fd3482e70c92468ecdc6d15a5297988046235bd1366c000bb9a20.jpg)

Fig. 4. Comparisons on ten ICVL, CAVE, and Harvard scenes for structured noise; CAVE and Harvard are zero-shot. Each block shows Clean/Noisy/Ours/SSUMamba /HSDT with MPSNR (dB). Panels: (a,e,f) ICVL mixture/stripe/dead-line; (b,j) CAVE impulse; and (c,d,g,h,i) Harvard stripe/stripe/dead-line/impulse/mixture.  
![](images/605b5fa29984505d9dcf63ba1542417823ba3fd687e07daf5ec1cef4e9c1fdb4.jpg)  
Fig. 5. Left: the results of AXS decomposition including AX, |S|, A, and an abundance map. Right: zero-shot band-averaged error maps on CAVE impulse and Harvard stripe noise; darker is better and values report MAE.

![](images/4fcba54eefb09a6902a259c42b86d6b9db5d994087fbae76d6594164927a4bca.jpg)

![](images/4a5ab9ac4013983ba8f70fcdccdce3ad26d8198ff6ac695ae7edd8a46e2392dd.jpg)

![](images/c9d5e25b4d1436938c790b4f49e750b4486f884718d1b76772e9d7df2abdb180.jpg)  
Fig. 6. Spectral-shape fidelity of the extracted structured noise S on ICVL (n=50): (a) extracted versus reference band-magnitude profiles for the best, median, and worst scenes; (b) per-scene Pearson correlation; (c) per-scene RMSE.

![](images/3f04a4819d6b3cdae2e3f6c693c2c74e70dcad02586fc08962091a14fc73181e.jpg)  
Fig. 7. Stage-wise evolution of the decomposition on an ICVL mixture scene. Top: reconstruction $\mathbf { A } ^ { k } \mathbf { X } ^ { k }$ at k=1, 3, 6 against the clean reference. Bottom: structured residual $| \mathbf { S } ^ { k } |$ against the noisy input. The reconstruction sharpens while the residual concentrates onto the column-consistent stripe pattern.

Across the 50 ICVL mixture scenes (Fig. 6), the estimated noise profiles closely follow the reference curves, with a median Pearson correlation of 0.84 (mean $0 . 8 3 \pm 0 . 0 8 ; 3 7 / 5 0 > 0 . 8 0 )$ and an RMSE of $0 . 0 9 6 \pm 0 . 0 0 2$ . Since S is trained with synthetic supervision, these metrics quantify the consistency of the decomposition under our experimental setting.

## 3.4 Ablation Study

SSX-Block. Table 3 shows the noise-specific efects of each component: spatial attention is most beneficial for Gaussian and mixture noise, while omitting any component leads to a marked drop in performance on impulse noise.

Decomposition components. Table 4 ablates the three structural ingredients of the unfolding on ICVL mixture noise (c5); each variant is retrained from scratch with exactly one factor changed. Removing the structured-noise branch $( \mathbf { S } ^ { k } \mathrm { = } 0$ at every stage) costs 8.15 dB MPSNR, freezing the basis at its PCA initialization costs 10.02 dB, and halving the depth from $K { = } 6$ to K=3 costs 5.22 dB—all 8– 15× larger than the largest single SSX-Block sub-component efect on the same setting (0.65 dB, Table 3). The cost of the S branch also tracks the amount of structured content: 0.12 dB on pure Gaussian noise (41.69 → 41.57), 2.38 dB on stripe noise (39.65 → 37.27), and 8.15 dB on the mixture (38.57 → 30.42), indicating that the branch performs structure-specific work rather than merely adding generic capacity.

SBlock. In Table 4, removing the impulse prior costs 1.06 dB MPSNR, whereas removing the stripe prior or the local branch leaves MPSNR essentially unchanged (+0.10 and +0.03 dB). The impulse prior is therefore responsible for the sparse-corruption component, while the other two act on the recovered S rather than on the restored image, which our image-space metrics do not capture. We retain the stripe prior because it is parameter-free, and the local branch because it supplies the isotropic features that the directional branch alone cannot represent; S is itself a reported output of the model rather than an internal quantity.

Structured-noise supervision $( \mathcal { L } _ { S } )$ . Removing $\mathcal { L } _ { S }$ improves mixture-noise MP-SNR by 0.70 dB (38.57→39.27) and SAM by 0.92 (6.97→6.05; lower is better), indicating that $\mathcal { L } _ { S }$ mainly tethers S to the supervised reference rather than improving restoration. We nevertheless report the supervised configuration as our main model by design: that tethering is what makes the recovered S comparable to the reference profile in Fig. 6, and without it the model would remain a competitive denoiser but lose its interpretable structured-noise output.

Table 3. SSX-Block ablation on ICVL (MPSNR, dB; $\varDelta$ vs. full).
<table><tr><td>Noise</td><td>Full SSX</td><td> $\mathrm { w } / \mathrm { o }$  FFN</td><td> $\mathrm { w } / \mathrm { o }$  Spatial</td><td>w/o Spectral</td></tr><tr><td>Gaussian (c1)</td><td>41.69</td><td>41.60 (−0.09)</td><td>40.24 (−1.45)</td><td>41.70 (+0.01)</td></tr><tr><td>Impulse (c4)</td><td>41.38</td><td>38.02 (−3.36)</td><td>38.97(-2.41)</td><td>39.40 (−1.98)</td></tr><tr><td>Mixture (c5)</td><td>38.57</td><td>38.46 (−0.11)</td><td>37.92 (−0.65)</td><td>38.57 (0.00)</td></tr></table>

Table 4. Decomposition, SBlock, and loss ablation on ICVL mixture noise (c5, 50 test scenes, no test-time augmentation), scored with the sliding-window protocol of Table 1. Each variant is retrained with the full recipe, changing exactly one factor; ∆ is MPSNR relative to the full K=6 model. Bold=best value in each metric column.
<table><tr><td>Variant</td><td>MPSNR↑</td><td>MSSIM↑</td><td>SAM↓</td><td>∆MPSNR</td></tr><tr><td>Full AXS-Net (K=6)</td><td>38.57</td><td>0.947</td><td>6.97</td><td></td></tr><tr><td>w/o S branch</td><td>30.42</td><td>0.821</td><td>24.00</td><td>-8.15</td></tr><tr><td>fixed A (no basis update)</td><td>28.55</td><td>0.811</td><td>25.91</td><td>-10.02</td></tr><tr><td>K=3 stages</td><td>33.35</td><td>0.857</td><td>19.83</td><td>-5.22</td></tr><tr><td>w/o impulse prior</td><td>37.51</td><td>0.939</td><td>7.56</td><td>-1.06</td></tr><tr><td>w/o stripe prior</td><td>38.67</td><td>0.947</td><td>6.81</td><td>+0.10</td></tr><tr><td>w/o local branch</td><td>38.60</td><td>0.947</td><td>6.98</td><td>+0.03</td></tr><tr><td>w/o  $\mathcal { L } _ { S }$ </td><td>39.27</td><td>0.952</td><td>6.05</td><td>+0.70</td></tr></table>

## 4 Conclusion

AXS-Net unrolls an A–X–S decomposition with an analytic basis step and learned abundance and structured-noise operators. It achieves strong in-domain and zero-shot results while exposing spectrally structured intermediate variables. Its main limitations are reliance on synthetic S supervision, evaluation confined to synthetically corrupted data, and weaker MPSNR transfer on strongly shifted CAVE stripe and mixture cases; unsupervised structured-noise estimation and validation on real-noise HSIs are left for future work.

Acknowledgments. This work was supported by Hunan Provincial Natural Science Foundation of China, under the Science and Technology Innovation Program of Hunan Province (Project No. 2025JJ60883); by the Hunan Provincial College Students’ Entrepreneurship Training Program (Project No. S202510530147X); and by the National College Students’ Innovation Training Program (Project No. 202510530057). The authors also thank the High Performance Computing Platform of Xiangtan University.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Arad, B., Ben-Shahar, O.: Sparse recovery of hyperspectral signal from natural RGB images. In: ECCV. pp. 19–34 (2016). https://doi.org/10.1007/ 978-3-319-46478-7\_2

2. Bioucas-Dias, J.M., Plaza, A., Dobigeon, N., Parente, M., Du, Q., Gader, P., Chanussot, J.: Hyperspectral unmixing overview: Geometrical, statistical, and sparse regression-based approaches. IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens. 5(2), 354–379 (2012). https://doi.org/10.1109/JSTARS.2012.2194696

3. Bodrito, T., Zouaoui, A., Chanussot, J., Mairal, J.: A trainable spectral-spatial sparse coding model for hyperspectral image restoration. In: NeurIPS. vol. 34, pp. 5430–5442 (2021)

4. Bouali, M., Ladjal, S.: Toward optimal destriping of MODIS data using a unidirectional variational model. IEEE Trans. Geosci. Remote Sens. 49(8), 2924–2935 (2011). https://doi.org/10.1109/TGRS.2011.2119399

5. Chakrabarti, A., Zickler, T.: Statistics of real-world hyperspectral images. In: CVPR. pp. 193–200 (2011). https://doi.org/10.1109/CVPR.2011.5995660

6. Daubechies, I., Defrise, M., De Mol, C.: An iterative thresholding algorithm for linear inverse problems with a sparsity constraint. Comm. Pure Appl. Math. 57(11), 1413–1457 (2004). https://doi.org/10.1002/cpa.20042

7. Fu, G., Xiong, F., Lu, J., Zhou, J.: SSUMamba: spatial–spectral selective state space model for hyperspectral image denoising. IEEE Trans. Geosci. Remote Sens. 62, 1–14 (2024). https://doi.org/10.1109/TGRS.2024.3446812

8. He, W., Yao, Q., Li, C., Yokoya, N., Zhao, Q.: Non-local meets global: An integrated paradigm for hyperspectral denoising. In: CVPR. pp. 6861–6870 (2019). https: //doi.org/10.1109/CVPR.2019.00703

9. He, W., Zhang, H., Zhang, L., Shen, H.: Total-variation-regularized low-rank matrix factorization for hyperspectral image restoration. IEEE Trans. Geosci. Remote Sens. 54(1), 178–188 (2016). https://doi.org/10.1109/TGRS.2015.2452812

10. Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: CVPR. pp. 7132– 7141 (2018). https://doi.org/10.1109/CVPR.2018.00745

11. Hwang, H., Haddad, R.A.: Adaptive median filters: new algorithms and results. IEEE Trans. Image Process. 4(4), 499–502 (1995). https://doi.org/10.1109/83. 370679

12. Kruse, F.A., Lefkof, A.B., Boardman, J.W., Heidebrecht, K.B., Shapiro, A.T., Barloon, P.J., Goetz, A.F.H.: The spectral image processing system (SIPS)— interactive visualization and analysis of imaging spectrometer data. Remote Sens. Environ. 44(2–3), 145–163 (1993). https://doi.org/10.1016/0034-4257(93)90013-N

13. Lai, Z., Yan, C., Fu, Y.: Hybrid spectral denoising transformer with guided attention. In: ICCV. pp. 13019–13029 (2023). https://doi.org/10.1109/ICCV51070. 2023.01201

14. Lee, D.D., Seung, H.S.: Learning the parts of objects by non-negative matrix factorization. Nature 401(6755), 788–791 (1999). https://doi.org/10.1038/44565

15. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV. pp. 9992–10002 (2021). https://doi.org/10.1109/ICCV48922.2021.00986

16. Maggioni, M., Katkovnik, V., Egiazarian, K., Foi, A.: Nonlocal transform-domain filter for volumetric data denoising and reconstruction. IEEE Trans. Image Process. 22(1), 119–133 (2013). https://doi.org/10.1109/TIP.2012.2210725

17. Wald, L.: Data Fusion: Definitions and Architectures—Fusion of Images of Diferent Spatial Resolutions. Presses de l’École des Mines de Paris, Paris (2002)

18. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process. 13(4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861

19. Xiong, F., Zhou, J., Zhao, Q., Lu, J., Qian, Y.: MAC-Net: Model-aided nonlocal neural network for hyperspectral image denoising. IEEE Trans. Geosci. Remote Sens. 60, 1–14 (2022). https://doi.org/10.1109/TGRS.2021.3131878

20. Yasuma, F., Mitsunaga, T., Iso, D., Nayar, S.K.: Generalized assorted pixel camera: postcapture control of resolution, dynamic range, and spectrum. IEEE Trans. Image Process. 19(9), 2241–2253 (2010). https://doi.org/10.1109/TIP.2010.2046811

21. Zhang, H., He, W., Zhang, L., Shen, H., Yuan, Q.: Hyperspectral image restoration using low-rank matrix recovery. IEEE Trans. Geosci. Remote Sens. 52(8), 4729– 4743 (2014). https://doi.org/10.1109/TGRS.2013.2284280

22. Zhang, J., Ghanem, B.: ISTA-Net: Interpretable optimization-inspired deep network for image compressive sensing. In: CVPR. pp. 1828–1837 (2018). https: //doi.org/10.1109/CVPR.2018.00196