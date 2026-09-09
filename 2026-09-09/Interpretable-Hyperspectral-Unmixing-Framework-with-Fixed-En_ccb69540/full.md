# Interpretable Hyperspectral Unmixing Framework with Fixed Endmember Prior and Structured Residual Refinement<sup>⋆</sup>

Ziyi Guan<sup>1</sup>, Jianping Zhang<sup>1B</sup>, and Qian Liu<sup>1</sup>

School of Mathematics and Computational Science, Xiangtan University, Xiangtan 411105, China jpzhang@xtu.edu.cn

Abstract. Hyperspectral unmixing decomposes mixed pixels into material endmembers and their abundances from contiguous spectral observations. In modular sensing pipelines, endmembers are often first identified and then treated as fixed during abundance estimation. When this fixed endmember prior is inaccurate, spatially structured mismatch arising from illumination changes, sensor artifacts, or material boundaries may be incorrectly captured by the abundance variables, leading to unstable decompositions. This study presents an interpretable stage-wise hyperspectral unmixing framework (I-HyperSU) under fixed endmember priors, which is explicitly decomposed into a fixed endmember matrix A, an abundance block X, and a structural residual refinement block S. The X-block estimates abundances using FISTA with nonnegativity and sparsity enhancement, and a soft penalty that approximately enforces sum-to-one constraints. The S-block jointly applies low-rank SVD structural regularization and a lightweight deep image prior (DIP) to refine structured residuals. This staged design makes the interaction between abundance and residual components transparent and interpretable. Experiments on Samson, Urban, and Jasper Ridge datasets demonstrate that, under fixed and imperfect endmember priors, soft abundance relaxation consistently outperforms hard simplex projection. Under the default N-FINDR endmember prior, the proposed framework reduces the joint reconstruction error by 61.7%–69.5% compared with a fixed-A UCLS baseline, while keeping the abundance RMSE nearly unchanged, indicating that the residual refinement branch accounts for structured model mismatch without degrading the abundance estimates. For example, on Urban, the reconstruction SAM decreases from 5.99<sup>◦</sup> for the X-only model to 1.92<sup>◦</sup> for the full model.

Keywords: Hyperspectral unmixing · Spectral scene decomposition · Fixed endmember prior · Deep image prior · Hyperspectral image analysis.

## 1 Introduction

Hyperspectral imaging extends visual perception beyond conventional RGB by resolving material-dependent responses across many contiguous spectral bands. Hyperspectral unmixing aims to decompose mixed-pixel spectra into endmember spectra and their corresponding abundances, enabling applications such as material detection, environmental assessment, and urban scene interpretation [2, 22]. Such a representation is particularly important when limited spatial resolution and complex scene configurations cause individual pixels to contain multiple materials.

In typical modular pipelines, endmembers are first estimated using geometric methods, spectral libraries, or calibration procedures, and are then provided to subsequent stages as fixed priors for abundance estimation [13, 19, 15, 3]. Although this fixed-endmember approach is widely adopted, it introduces a key dificulty: once these priors are set, any spectral mismatch or unmodeled efects must be accommodated exclusively by the downstream unmixing process.

Existing methods range from constrained abundance optimization techniques and variability-sensitive formulations to newer deep learning and transformerbased models. Classical constrained and sparse-regression methods, such as FCLS [8] and SUnSAL [9], impose simplex-like abundance constraints [10]. These approaches perform well when the endmembers are specified accurately, but they run the risk of embedding systematic model mismatch into the abundance estimates. Models that are designed to handle variability and mismatch, such as those in [7, 6, 5, 16], explicitly account for spectral mismatch or endmember variability, yet they typically do so by modifying the endmember matrix or estimating it jointly. Recent deep learning methods, such as UnDIP [14] and MAT-Net [18], exploit deep image priors [17] or learned spatial–spectral representations, yet they are primarily designed for joint or learned unmixing rather than strictly modular frameworks that rely on fixed, non-adaptive endmember priors. As a result, they ofer no explicit guidance on how a downstream solver should allocate unexplained energy between abundance corrections and structured residual refinement when the prior has to be kept fixed. In contrast, this paper focuses on a frozen-prior setting where the endmember matrix A is kept fixed, and explicitly separates abundance estimation from per-scene structured residual refinement without using external training data.

To explicitly represent this allocation under imperfect fixed priors, we propose an interpretable stage-wise hyperspectral unmixing framework (I-HyperSU), where A serves as the fixed endmember prior, X corresponds to the abundance branch, and S represents the structural-residual branch. The X-block uses FISTA [1] with nonnegativity constraints, a sparsity-promoting regularizer, and a soft sum-to-one penalty, which reduces the tendency of hard simplex projections to conceal mismatch within the abundances. The S-block uses low-rank structural coordinates to guide a lightweight deep image prior (DIP)–based residual refinement module, modeling structured discrepancies that are not captured by the endmember model. At each iteration, the stage-wise procedure produces intermediate pairs $( \mathbf { X } ^ { k } , \mathbf { S } ^ { k } )$ , so that the progressive reduction of mismatch between abundances and residuals can be directly interpreted.

The main contributions are summarized as follows.

– We present a staged hyperspectral unmixing framework with fixed endmember priors, in which the FISTA-based abundance estimation and the refinement of structured residuals are clearly decoupled.

– We propose a low-rank-guided SVD-DIP branch to capture structured residuals, where low-rank information is integrated with a lightweight nonlinear refinement module, yielding an explicit and controllable mechanism to model spectral mismatches.

– We perform controlled experiments on the Samson, Urban, and Jasper Ridge datasets using fixed N-FINDR endmember priors, showing that our framework reduces the joint reconstruction error by 61.7%–69.5% compared to the UCLS fixed-prior baseline across all three datasets, while maintaining comparable abundance RMSE. This demonstrates the advantages of soft abundance relaxation and explicit residual modeling.

– On synthetic data with a known ground-truth residual, we verify that the residual branch captures genuine out-of-model mismatch rather than merely fitting observation residuals, achieving r ≈ 0.77 overall and $r > 0 . 9 9$ on the component orthogonal to the endmember subspace, while remaining inactive when no mismatch is present.

The remainder of the paper is structured as follows. Section 2 formulates the problem with fixed endmember priors and introduces the stage-wise hyperspectral unmixing algorithm. Section 3 reports experimental results, including fixedprior validation, comparisons between soft- and hard-prior simplex constraints, ablations of the S-block, visualizations of residual refinement, and evaluations against classical baseline methods.

## 2 Method

We begin by formulating hyperspectral unmixing problem using fixed endmember priors, after which we estimate the abundance matrix under a relaxed abundance constraint and explicitly channel the remaining structured mismatch into a separate residual refinement branch. As a result, we obtain an optimizationdriven decomposition framework rather than a purely end-to-end trained network.

## 2.1 Problem Formulation

Let $\mathbf { Y } = [ \pmb { y } _ { 1 } , \dots , \pmb { y } _ { n } ] \in \mathbb { R } ^ { \ell \times n }$ denote the hyperspectral data matrix observed, where ℓ is the number of spectral bands and n is the total number of pixels.

The matrix $\mathbf { A } = [ \pmb { a } _ { 1 } , \ldots , \pmb { a } _ { r } ] \in \mathbb { R } ^ { \ell \times r }$ collects the endmember signatures, and $\mathbf { X } = [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } ] \in \mathbb { R } ^ { r \times n }$ contains the corresponding abundance vectors. The standard linear mixing model (LMM) can be expressed as

$$
\mathbf { Y } = \mathbf { A } \mathbf { X } , \qquad \mathbf { X } \succeq 0 , \qquad \mathbf { 1 } _ { r } ^ { \top } \mathbf { X } = \mathbf { 1 } _ { n } ^ { \top } ,\tag{1}
$$

where $\mathbf { 1 } _ { r } \in \mathbb { R } ^ { r }$ and $\mathbf { 1 } _ { n } \in \mathbb { R } ^ { n }$ denote all-ones vectors. These conditions ensure that each abundance vector $\mathbf { x } _ { i }$ belongs to the probability simplex

$$
\varDelta = \{ \mathbfpmb { x } \in \mathbb { R } ^ { r } \mid \mathbf { x } \succeq 0 , \ \mathbf { 1 } _ { r } ^ { \top } \pmb { x } = 1 \} .\tag{2}
$$

In practice, the endmember matrix A is typically obtained from a geometric endmember extraction algorithm, and then kept fixed during the subsequent unmixing stage. When A is inaccurate, the observations cannot always be faithfully modeled by AX alone. Real scenes may exhibit shadows, striping artifacts, background structures, and other systematic efects that are not captured by the fixed endmember model. To address these, we consider

$$
\mathbf { Y } = \mathbf { A } \mathbf { X } + \mathbf { S } + \pmb { \eta } ,\tag{3}
$$

where $\mathbf { S } \in \mathbb { R } ^ { \ell \times n }$ represents a structured residual component and η denotes small random perturbations. From this perspective, a general decomposition can be written as

$$
\operatorname* { m i n } _ { { \bf x } , { \bf S } } \frac { 1 } { 2 } \left\| { \bf Y } - { \bf A } { \bf X } - { \bf S } \right\| _ { F } ^ { 2 } + \lambda _ { \bf x } \psi _ { \bf X } ( { \bf X } ) + \lambda _ { \bf S } \psi _ { \bf S } ( { \bf S } ) ,\tag{4}
$$

where $\psi _ { \mathbf { X } } ( \cdot )$ and $\psi _ { \mathbf { S } } ( \cdot )$ are regularizers imposed on the abundance and residual terms, respectively. Thus, the main focus of this paper is to determine how the unmodeled signal should be apportioned between X and S once A is fixed.

## 2.2 Proposed I-HyperSU Framework

Fig. 1 provides an overview of the proposed interpretable stage-wise hyperspectral unmixing framework. The A-block provides a fixed geometric prior extracted from the observed HSI; the X-block estimates abundance maps under relaxed physical constraints, incorporating $\ell _ { 1 }$ sparsity and nonnegativity alongside a soft ASC penalty that replaces the hard simplex projection to reduce mismatch absorption; and the S-block captures structured mismatch through a progression of models ranging from element-wise soft thresholding (l1-soft) to low-rank truncated SVD (lowrank\_r3) and DIP-guided low-rank decomposition (lowrank\_dip).

For any observation $\mathbf { Y } .$ , an endmember extractor $\mathcal { E } ( \cdot )$ first produces a frozen prior $\mathbf { A } = { \mathcal { E } } ( \mathbf { Y } )$ , which is fixed throughout all subsequent optimization. The remaining decomposition proceeds across $T$ stages. At stage $k ,$ the X-block takes the previous residual estimate $\mathbf { S } ^ { k - 1 }$ as coupled input and solves the abundance map $\mathbf { X } ^ { k }$ using FISTA with nonnegativity and a soft sum-to-one penalty; and the S-block fits a structural residual $\bar { \mathbf { S } ^ { k } } \approx \bar { \mathbf { R } ^ { k } } = \mathbf { Y } - \mathbf { A } \mathbf { X } ^ { k }$ using low-rank structural guidance and a lightweight DIP model. The abundance estimate $\mathbf { X } ^ { k }$ warm-starts the next X-block, while $\mathbf { \bar { s } } ^ { k }$ re-enters as a coupled residual input to the following X-step, which is formally written as

![](images/e92608cc27df2e61a860306b464a026cee6ad54b06429fe53b4587ec5674b756.jpg)  
Fig. 1. Overview of the proposed interpretable stage-wise hyperspectral unmixing framework (I-HyperSU). The A-block provides a frozen endmember prior, the X-block performs abundance estimation with soft relaxation, and the S-block captures structured mismatch through progressive residual refinement.

$$
( \mathbf { X } ^ { k } , \mathbf { S } ^ { k } ) = { \mathcal { T } } ( \mathbf { X } ^ { k - 1 } , \mathbf { S } ^ { k - 1 } ; \mathbf { A } ) .\tag{5}
$$

After the $T$ stages, the final output is the abundance map $\mathbf { X } ^ { T }$ and the structural residual $\mathbf { S } ^ { T }$

Fixed Endmember Extraction (A-Block). As discussed, the endmember matrix A is estimated once before the main optimization and then held fixed. Within the geometric framework of unmixing [8, 2], the A-block imposes a fixed endmember prior by selecting a small subset of pixels that approximate the vertices of the data simplex:

$$
\mathbf { A } = { \mathcal { E } } ( \mathbf { Y } ) ,\tag{6}
$$

where $\mathcal { E } ( \cdot )$ denotes a geometric endmember extractor.

Diferent algorithms implement E using diferent geometric rules. We adopt four classical pure-pixel strategies for the A-block: VCA [13], N-FINDR [19], ATGP [15], and FIPPI [3]. All are designed to pick pixels that are spectrally extreme or highly pure, and the selection of pixels is given as

$$
\varOmega ^ { \star } = \mathscr { T } _ { \mathcal { E } } ( \mathbf { Y } , r ) = \arg \operatorname* { m a x } _ { \varOmega , \ | \varOmega | = r } \mathrm { V o l } ( \mathbf { Y } _ { \varOmega } ) ,\tag{7}
$$

where $\operatorname { V o l } ( \mathbf { Y } _ { \varOmega } )$ is the volume of the simplex spanned by the selected spectra, and $\varOmega = \{ \omega _ { 1 } , \ldots , \omega _ { r } \} \subset \{ 1 , \ldots , N \}$

The A-block itself is not the core algorithmic contribution; its purpose is merely to provide a fixed geometric prior to the hyperspectral unmixing solver. Unless stated otherwise, we use N-FINDR as the default endmember extractor:

$$
\begin{array} { r } { \varOmega _ { \mathrm { N F } } = \varmathbb { Z } _ { \mathrm { N - F I N D R } } ( \mathbf { Y } , r ) , \qquad \mathbf { A } = \mathbf { Y } _ { \varOmega _ { \mathrm { N F } } } . } \end{array}\tag{8}
$$

For every dataset, we perform endmember extraction separately, producing a dataset-specific fixed prior. VCA, ATGP, and FIPPI are employed solely as alternative choices for E in our comparisons and are neither merged with nor used to refine the N-FINDR-based prior.

Once A is fixed, we restrict the decomposition problem to the case of an imperfect yet fixed prior, and we explicitly specify how the model mismatch is apportioned between the abundance matrix X and the residual term S.

Soft Abundance Relaxation (X-Block). Under ideal conditions, each abundance vector $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ belongs to the simplex ∆, which enforces both nonnegativity and a unit-sum constraint. When A is accurately known, the projection onto this simplex has a clear geometric interpretation. However, the estimated endmember matrix A is generally imperfect, so strictly enforcing simplex membership compels the solution to remain in $\varDelta$ regardless of the reconstruction quality, which can inadvertently shift structured model mismatches into the abundance variables.

To tackle this, we adopt a soft relaxation of the abundance constraint:

$$
\mathbf { X } ^ { k } = \arg \operatorname* { m i n } _ { \mathbf { X } } ~ g ( \mathbf { X } ) + \lambda _ { x } \left\| \mathbf { X } \right\| _ { 1 } + \iota _ { \mathbb { R } _ { + } } ( \mathbf { X } ) ,\tag{9}
$$

where $\begin{array} { r } { g ( \mathbf { X } ) = \frac { 1 } { 2 } \left\| \mathbf { Y } - \mathbf { A } \mathbf { X } - \mathbf { S } ^ { k - 1 } \right\| _ { F } ^ { 2 } + \frac { \mu } { 2 } \left\| \mathbf { 1 } ^ { \top } \mathbf { X } - \mathbf { 1 } ^ { \top } \right\| _ { F } ^ { 2 } , \iota _ { \mathbb { R } _ { + } } ( \mathbf { X } ) } \end{array}$ enforces the nonnegativity of $\mathbf { \dot { X } } , \mathbf { \lVert X \rVert } _ { 1 }$ encourages sparsity [9], and the soft sum-to-one penalty $\big \| \mathbf { 1 } ^ { \top } \mathbf { X } - \mathbf { 1 } ^ { \top } \big \| _ { F } ^ { 2 }$ permits controlled deviations from the simplex constraint. Consequently, the abundance variables are no longer required to account for all mismatches induced by fixed priors, and these structured discrepancies are instead explicitly modeled within the S-block.

To solve (9) with FISTA [1], we first evaluate the gradient as

$$
\nabla g ( \mathbf { X } ) = \mathbf { A } ^ { \top } ( \mathbf { A X } + \mathbf { S } ^ { k - 1 } - \mathbf { Y } ) + \mu \mathbf { 1 } \left( \mathbf { 1 } ^ { \top } \mathbf { X } - \mathbf { 1 } ^ { \top } \right) .\tag{10}
$$

Next, the step size $\alpha = 1 / L _ { q }$ is approximated using the Lipschitz constant $L _ { g } = \lambda _ { \operatorname* { m a x } } ( \mathbf { A } ^ { \top } \mathbf { A } + \mu \mathbf { 1 1 } ^ { \top } )$ of $\nabla g$

For each outer iteration $k ,$ we run an inner loop indexed by s. To keep the notation concise, the stage index is dropped within this inner loop. The initialization is given by

$$
{ \bf X } ^ { 0 } = { \bf X } ^ { k - 1 } , \qquad { \hat { \bf X } } ^ { 0 } = { \bf X } ^ { k - 1 } , \qquad q _ { 0 } = 1 .\tag{11}
$$

The FISTA updates then take the form

$$
\begin{array} { r } { \widetilde { \mathbf { X } } ^ { s + 1 } = \mathbf { X } ^ { s } - \alpha \nabla g ( \mathbf { X } ^ { s } ) , \qquad \hat { \mathbf { X } } ^ { s + 1 } = \operatorname* { m a x } \Bigl ( 0 , \mathrm { s o f t } _ { \alpha \lambda _ { x } } \bigl ( \widetilde { \mathbf { X } } ^ { s + 1 } \bigr ) \Bigr ) , } \end{array}\tag{12}
$$

$$
\begin{array} { r } { q _ { s + 1 } = ( 1 + \sqrt { 1 + 4 q _ { s } ^ { 2 } } ) / 2 , \qquad { \bf X } ^ { s + 1 } = \hat { { \bf X } } ^ { s + 1 } + \frac { q _ { s } - 1 } { q _ { s + 1 } } \big ( \hat { { \bf X } } ^ { s + 1 } - \hat { { \bf X } } ^ { s } \big ) . } \end{array}\tag{13}
$$

Here, $\widetilde { \mathbf { X } } ^ { s + 1 }$ is the result of the gradient step, $\hat { \textbf { X } } ^ { s + 1 }$ is obtained by applying a sparse shrinkage followed by projection onto the nonnegative orthant, and ${ \mathbf X } ^ { s + 1 }$ denotes the accelerated iteration. The operator soft $. ( v ) = \mathrm { s i g n } ( v )$ max $( | v | - \tau , 0 )$ denotes element-wise soft-thresholding with threshold τ. Once the inner loop has finished, its final iteration is used as the abundance estimate $\mathbf { X } ^ { k }$ at stage k. As a reference method in our experiments, a hard-constrained approach projects each abundance vector directly onto the simplex [8].

SVD-Guided DIP Residual Modeling (S-Block). Following the update of the abundance matrix $\mathbf { X } ^ { k }$ , the residual with respect to the fixed endmember matrix A is defined as

$$
R _ { \mathbf { X } } ^ { k } = \mathbf { Y } - \mathbf { A } \mathbf { X } ^ { k } .\tag{14}
$$

If the endmembers were perfectly known and the data strictly obeyed the linear mixing model, $R _ { \mathbf { X } } ^ { k }$ would be dominated by small unstructured noise. In practice, however, the residual often displays organized patterns due to endmember mismatch, modeling artifacts, and other unmodeled efects. The aim of this stage is to make these structured discrepancies explicit, rather than absorbing them into the abundance estimates in the X-block. A residual update is expressed as

$$
\mathbf { S } ^ { k } = \arg \operatorname* { m i n } _ { \mathbf { S } } \frac { 1 } { 2 } \left\| \boldsymbol { R } _ { \mathbf { X } } ^ { k } - \mathbf { S } \right\| _ { F } ^ { 2 } + \lambda _ { s } \phi ( \mathbf { S } ) ,\tag{15}
$$

where $\phi ( \mathbf { S } )$ encodes the assumed structure prior to the residual branch. In sparse and low-rank configurations, ϕ(S) appears as an explicit regularizer; the simplest choice is an element-wise soft-thresholding rule, corresponding to the sparse prior $\phi ( \mathbf { S } ) = \| \mathbf { S } \| _ { 1 } \colon$

$$
{ \bf S } ^ { k } = \mathrm { s o f t } _ { \lambda _ { s } \hat { \sigma } ^ { k } } ( R _ { { \bf X } } ^ { k } ) , \qquad \hat { \sigma } ^ { k } = q _ { 0 . 9 9 } ( | R _ { { \bf X } } ^ { k } | ) ,\tag{16}
$$

with $q _ { 0 . 9 9 } ( \cdot )$ denoting the 99th percentile of the absolute residual entries. This setup tests whether a purely sparse residual branch sufices, although mismatchinduced residuals for fixed endmembers are typically spatially and spectrally correlated, rather than strictly element-wise sparse.

To encode this correlated structure, we introduce a low-rank residual representation. We compute a truncated SVD of the transposed residual:

$$
\mathbf { S } ^ { k } \approx R _ { \mathbf { X } } ^ { k } = U _ { r _ { s } } ^ { k } \varSigma _ { r _ { s } } ^ { k } ( V _ { r _ { s } } ^ { k } ) ^ { \top } ,\tag{17}
$$

where $r _ { s }$ denotes the residual guidance rank. Since $( R _ { \mathbf { X } } ^ { k } ) ^ { \top } \in \mathbb { R } ^ { n \times \ell }$ treats pixels as samples and spectral bands as feature dimensions, the leading singular directions capture the main residual patterns shared across pixels. This low-rank subspace strategy is motivated by recent hyperspectral denoising work that combines subspace decomposition with neural priors [21].

Classical priors such as sparsity or low-rankness explain only part of the residual structure and are restricted to element-wise or linear models. In contrast, the default SVD-guided DIP branch—denoted learnable reparameterization lowrank\_dip in the experiments—imposes a residual prior indirectly rather than penalizing S explicitly. Instead of solving a direct S-subproblem, we reparameterize S via a neural mapping $f _ { \theta } ,$ , constraining S to the image of $f _ { \boldsymbol { \theta } } ;$ the optimization over θ then implicitly induces the structural prior. Concretely, the residual generator is given by

$$
\mathbf { S } _ { \theta } ^ { k } = f _ { \theta } ( Z ^ { k } ) ^ { \top } , ~ Z ^ { k } = ( z _ { i } ^ { k } , \ldots , z _ { n } ^ { k } ) ^ { \top } = U _ { r _ { s } } ^ { k } \Sigma _ { r _ { s } } ^ { k } \in \mathbb { R } ^ { n \times r _ { s } } ,
$$

where a shared nonlinear lifting network $f _ { \theta } : \mathbb { R } ^ { r _ { s } }  \mathbb { R } ^ { \ell }$ maps each $z _ { i } ^ { k }$ to a ℓ- dimensional residual spectrum $f _ { \theta } ( z _ { i } ^ { k } )$ . Using decomposition-derived coordinates as input to a coordinate-based MLP follows the implicit neural representation strategy recently adopted for hyperspectral image restoration [4]. This constrains residuals to follow the dominant SVD modes, while allowing nonlinear variations within the induced subspace. The parameters θ are obtained by solving

$$
\theta ^ { k , * } = \arg \operatorname* { m i n } _ { \theta } \frac { 1 } { 2 } \left\| R _ { \mathbf { X } } ^ { k } - f _ { \theta } \big ( Z ^ { k } \big ) ^ { \top } \right\| _ { F } ^ { 2 } .\tag{18}
$$

In practice, $f _ { \theta }$ is a compact per-pixel two-layer MLP with architecture $r _ { s } $ $3 2  \ell$ and ReLU activations. For the Urban dataset with $r _ { s } \ = \ 3$ and $\ell =$ 162, this yields about $5 . 5 \times 1 0 ^ { 3 }$ parameters per residual instance. Each stage is initialized randomly and trained with Adam (learning rate $1 0 ^ { - 3 } )$ for up to 1500 iterations. Unless otherwise stated, we set $r _ { s } = 3$ and $T = 2$

## 3 Experiments

## 3.1 Experimental Setup

We assess the proposed method—in which the fixed endmember matrix A is obtained in the A-block—on three benchmark hyperspectral datasets [20] with diferent scene properties: Samson (156 bands, 95×95 pixels, $r = 3 )$ , Urban (162 bands, $3 0 7 { \times } 3 0 7 \mathrm { p i x e l s } , r = 4 )$ , and Jasper Ridge (198 bands, 100×100 pixels, $r =$ 4). Unless noted otherwise, the X-block is solved with FISTA using $\lambda _ { x } = 1 0 ^ { - 4 } ,$ $\mu = 1 0 ^ { - 1 }$ , and at most 2000 iterations. In the S-block, the $\ell _ { 1 }$ -soft branch adopts $\lambda _ { s } = 0 . 2$ after residual normalization by $q _ { 0 . 9 9 } ( | R | )$ , while both lowrank\_r3 and lowrank\_dip are governed by a residual rank of $r _ { s } = 3$ . The default lowrank\_dip branch employs a two-layer MLP with 32 hidden units, trained with Adam for 1500 iterations at a learning rate of $1 0 ^ { - 3 }$ . This configuration is applied uniformly to all datasets, with no dataset-specific tuning.

For endmember extraction, we report the mean Spectral Angle Mapper (SAM, degrees) [2] after obtaining estimated and reference endmembers. For fixed-prior unmixing, we quantify abundance accuracy by $x _ { \mathrm { r m s e } }$ and decomposition fidelity by reconstruction SAM and the relative joint error,

$$
\begin{array} { r } { x _ { \mathrm { r m s e } } = \sqrt { \frac { 1 } { r N } \left\| \mathbf { X } - \hat { \mathbf { X } } \right\| _ { F } ^ { 2 } } , \qquad \mathrm { J o i n t } = \frac { \left\| \mathbf { Y } - A \hat { \mathbf { X } } - \hat { \mathbf { S } } \right\| _ { F } } { \| \mathbf { Y } \| _ { F } } , } \end{array}\tag{19}
$$

where $\mathbf { X } , { \hat { \mathbf { X } } }$ are the ground-truth and estimated abundances [9] and $\hat { \mathbf { S } } = 0$ for methods without a residual component. The soft-versus-hard comparison reports the abundance sum violation $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| 1 - \mathbf { 1 } ^ { \top } \pmb { x } _ { i } \right| } \end{array}$ to measure deviations from the sum-to-one constraint.

Table 1. Comparison of endmember extractors using SAM (deg).
<table><tr><td>Method</td><td>Samson</td><td>Urban</td><td>Jasper</td></tr><tr><td>VCA [13]</td><td>4.89</td><td>25.30</td><td>22.09</td></tr><tr><td>N-FINDR [19]</td><td>4.02</td><td>25.57</td><td>12.17</td></tr><tr><td>ATGP [15]</td><td>21.99</td><td>24.82</td><td>18.50</td></tr><tr><td>FIPPI [3]</td><td>20.45</td><td>31.01</td><td>25.31</td></tr></table>

Reproducibility. All main runs use one fixed N-FINDR realization (Table 1) and a fixed random seed (0; seeds 0–2 are used only for the stability analysis in Fig. 3b), with no ground-truth-based selection. Since $\lambda _ { x }$ and $\mu$ are fixed across scenes with diferent scales, quantitative comparisons are made within each dataset rather than across datasets.

## 3.2 Reliability of Fixed Endmember Priors

We first evaluate the A-block to establish a reasonable but imperfect frozen prior for the downstream fixed-prior experiments, rather than to select an oracle endmember extractor.

Table 1 indicates that no single extractor dominates across all datasets. ATGP attains the lowest SAM on Urban, while N-FINDR achieves the best performance on Samson and Jasper and remains competitive with ATGP on Urban. We therefore adopt N-FINDR as the default A-block, not as a universally optimal method, but because it ofers the most balanced fixed prior over the three datasets. This distinction matters: the resulting prior remains imperfect— especially on Urban—and the downstream hyperspectral unmixing solver is explicitly designed to handle such fixed-prior mismatch.

Once A is fixed, the subsequent experiments focus on isolating the influence of modeling choices in X and S under the default N-FINDR prior. VCA, ATGP, and FIPPI are each run through the complete pipeline in the prior-mismatch stress test (Fig. 7). In that test, each prior is re-extracted independently using the corresponding extractor settings, so the resulting per-extractor SAM values may difer from those reported in Table 1.

## 3.3 Soft Relaxation Versus Hard Simplex

Under an imperfect prior, we compare a soft sum-to-one constraint (soft ASC) with hard simplex projection, and examine where the relaxation is applied.

Table 2 isolates the efect of the abundance constraint. Soft abundance relaxation produces lower abundance RMSE and reduced joint error. Although the hard-simplex solver enforces exact sum-to-one, strict simplex feasibility of the abundance vector $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ does not necessarily yield better decompositions when the prior is inaccurate.

Table 2. Comparison between soft abundance relaxation and hard simplex projection under the same fixed N-FINDR prior.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">RMSE↓</td><td colspan="2">Sum viol.↓</td><td colspan="2">Joint↓</td></tr><tr><td>Soft</td><td>Hard</td><td>Soft</td><td>Hard</td><td>Soft</td><td>Hard</td></tr><tr><td>Urban</td><td>0.3239</td><td>0.3895</td><td>0.5504</td><td>0.0000</td><td>0.1002</td><td>0.4965</td></tr><tr><td>Samson</td><td>0.4057</td><td>0.4470</td><td>0.3134</td><td>0.0000</td><td>0.0357</td><td>0.0525</td></tr><tr><td>Jasper</td><td>0.4212</td><td>0.4612</td><td>0.3383</td><td>0.0000</td><td>0.0900</td><td>0.1979</td></tr></table>

![](images/a0636ad6fb6ec94a0f5798be8aac0b96eb7c44c9d2ffd086b3669b079bb93197.jpg)  
Fig. 2. Soft-ASC vs hard-simplex spatial mechanism on Urban (fixed N-FINDR prior). Top: ground-truth, soft, and hard abundance maps with per-pixel error maps. Bottom: the soft sum-to-one violation map and the fixed-prior residual $| \mathbf { Y } - \mathbf { A } \mathbf { G } \mathbf { T } |$ co-localize, and the per-pixel scatter shows the soft-over-hard improvement $e _ { \mathrm { h a r d } } - e _ { \mathrm { s o f t } }$ growing with the prior residual.

Fig. 2 makes this spatial mechanism explicit on Urban: the soft sum-to-one violation concentrates where the fixed-prior residual $| \mathbf { Y } - \mathbf { A } \mathbf { G } \mathbf { T } |$ is large, and a per-pixel analysis shows that the soft-over-hard error improvement $e _ { \mathrm { h a r d } } - e _ { \mathrm { s o f t } }$ grows with that prior residual (Pearson r = 0.72; median improvement is about 5× larger in the most unreliable quartile than in the most reliable one). Soft relaxation therefore loosens the constraint selectively in unreliable regions rather than uniformly.

## 3.4 Ablation Study for Hyperspectral Unmixing Enhancement

To isolate the source of the improvement, we vary the S-block across residual sparsity, low-rank structure, and the SVD-guided DIP branch, and further include a generic Joint-DIP control.

As shown in Table 3, each successive refinement of the S-block produces consistent gains in SAM and joint error across all three datasets, with $x _ { \mathrm { r m s e } }$ remaining essentially unchanged. These results suggest that the gain comes mainly from the SVD-guided DIP residual branch rather than from simply adding an arbitrary residual term. Furthermore, the DIP branch substantially reduces the joint error compared with the non-DIP l1-soft and analytic low-rank S-blocks (Fig. 3). Because the runtime varies little across residual ranks in this setting, we use $( T , r _ { s } ) = ( 2 , 3 )$ as a globally fixed default rather than a tuned optimum, and report $r _ { s } = 5$ only as an exploratory higher-capacity variant.

Table 3. Hyperspectral unmixing enhancement path across the three datasets. Bold marks the best value along the staged enhancement path (X-only → l1-soft → lowrank\_r3 → lowrank\_dip); the Joint-DIP row is a separate control and is excluded from the bold comparison.
<table><tr><td>Data</td><td>Variant</td><td>RMSE↓</td><td>SAM↓ (deg)</td><td>Joint↓</td></tr><tr><td rowspan="5">Samson</td><td>X-only</td><td>0.4057</td><td>2.7589</td><td>0.0357</td></tr><tr><td>11-soft</td><td>0.4051</td><td>1.7629</td><td>0.0166</td></tr><tr><td>lowrank r3</td><td>0.4056</td><td>1.3555</td><td>0.0129</td></tr><tr><td>lowrank_dip (default)</td><td>0.4056</td><td>1.0305</td><td>0.0107</td></tr><tr><td>Joint-DIP</td><td>0.3932</td><td>2.3484</td><td>0.0342</td></tr><tr><td rowspan="5">Urban</td><td>X-only</td><td>0.3239</td><td>5.9915</td><td>0.1002</td></tr><tr><td>11-soft</td><td>0.3236</td><td>3.3214</td><td>0.0462</td></tr><tr><td>lowrank r3</td><td>0.3235</td><td>2.4473</td><td>0.0366</td></tr><tr><td>lowrank_dip (default)</td><td>0.3236</td><td>1.9206</td><td>0.0295</td></tr><tr><td>Joint-DIP</td><td>0.3113</td><td>2.3699</td><td>0.0395</td></tr><tr><td rowspan="5">Jasper</td><td>X-only</td><td>0.4212</td><td>12.3878</td><td>0.0900</td></tr><tr><td>11-soft</td><td>0.4203</td><td>6.1206</td><td>0.0411</td></tr><tr><td>lowrank r3</td><td>0.4177</td><td>3.1454</td><td>0.0316</td></tr><tr><td>lowrank_dip (default)</td><td>0.4188</td><td>2.5208</td><td>0.0251</td></tr><tr><td>Joint-DIP</td><td>0.3523</td><td>3.3525</td><td>0.0379</td></tr></table>

Two controls support the design. A frozen-A Joint-DIP baseline, which performs joint optimization without the stage-wise SVD-guided routing, attains slightly lower abundance RMSE in some cases but worse reconstruction SAM and joint error than the staged design on all three datasets (Joint-DIP rows, Table 3). This indicates that the improvement is not simply due to adding a generic DIP module, but to the proposed stage-wise routing between X and S. In addition, across three DIP seeds the joint error stays within ±1% of its mean (Fig. 3b).

## 3.5 Qualitative Residual Refinement and Abundance Stability

We now inspect the decomposition visually, checking that S absorbs structured spatial–spectral mismatch while the abundance maps stay stable.

We first examine how structured mismatch is transferred from the reconstruction residual into the S branch: progressively richer residual models (l1-soft, lowrank\_r3, lowrank\_dip) leave a weaker final residual (Fig. 4), with the default lowrank\_dip attaining the lowest reconstruction SAM on every dataset. Fig. 5 confirms the spectral role of S: AX + S aligns with Y at high-residual pixels, and across residual quartiles S consistently lowers the median SAM, especially in high-mismatch regions. This supports the interpretation that S accounts for structured mismatch while inducing only small abundance drift.

![](images/5d5abfbbd1a15872e41c6eeb6e68c2dffdb1d30d56eda7a050d22dd09a32ec35.jpg)

![](images/8e1f3a8e22edcbd0af355e7a59a23ab4585cc234779f99541148a5cb002861c4.jpg)

Fig. 3. (a) S-block solver/rank ablation on Urban under the fixed N-FINDR prior: joint error versus residual rank for the DIP branch, with non-DIP l1-soft and lowrank S-blocks shown as references. (b) stability of the joint error over three random DIP seeds; all runs stay within ±1% of the per-dataset mean.  
![](images/4f9cc70bea1f60ec620818e6d1297cf06706cdeff1dc5ff85482d14684eabe28.jpg)

Fig. 4. S-block residual suppression on Samson, Urban, and Jasper under the fixed N-FINDR prior. Columns show the input image, AX, the input/captured/final residuals for the diferent S-blocks (l1-soft, lowrank\_r3, lowrank\_dip), abundance X, drift $| \varDelta \mathbf { X } | .$ , and restored AX + S. Residual maps share a per-row scale; badges report reconstruction SAM.  
![](images/a362ece0d7f1bc722b0736c830bcf4465c50af75ea145e9b8de8d6d6b8f5aa0c.jpg)  
Fig. 5. Spectral evidence of residual refinement. (a) for one high-residual pixel per dataset, $\mathbf { A } \mathbf { X } + \mathbf { S }$ better matches Y and reduces per-pixel SAM. (b) across all pixels grouped by fixed-prior residual quartiles, S consistently lowers the median SAM, with larger gains in higher-residual regions.

Overall, routing structured mismatch into S leaves X stable: $\varDelta x _ { \mathrm { r m s e } }$ stays within 0.003 and $\varDelta \mathbf { X } _ { \mathrm { r m s } }$ within 0.022 across all datasets.

Table 4. Overall comparison under the same fixed-A protocol. The proposed I-HyperSU denotes the default lowrank\_dip configuration.
<table><tr><td>Data</td><td>Method</td><td>RMSE↓</td><td>SAM↓ (deg)</td><td>Joint↓</td></tr><tr><td rowspan="4">Samson</td><td>UCLS[11]</td><td>0.4117</td><td>2.6979</td><td>0.0351</td></tr><tr><td>FCLS[8]</td><td>0.4470</td><td>4.4569</td><td>0.0525</td></tr><tr><td>NNLS[12]</td><td>0.4057</td><td>2.7589</td><td>0.0357</td></tr><tr><td>I-HyperSU (ours)</td><td>0.4056</td><td>1.0305</td><td>0.0107</td></tr><tr><td rowspan="4">Urban</td><td>UCLS[11]</td><td>0.3240</td><td>5.1824</td><td>0.0894</td></tr><tr><td>FCLS[8]</td><td>0.3895</td><td>21.4605</td><td>0.4965</td></tr><tr><td>NNLS[12]</td><td>0.3268</td><td>5.7716</td><td>0.0993</td></tr><tr><td>I-HyperSU (ours)</td><td>0.3236</td><td>1.9206</td><td>0.0295</td></tr><tr><td rowspan="4">Jasper</td><td>UCLS[11]</td><td>0.4758</td><td>7.7346</td><td>0.0656</td></tr><tr><td>FCLS[8]</td><td>0.4612</td><td>13.7153</td><td>0.1979</td></tr><tr><td>NNLS[12]</td><td>0.4212</td><td>12.3812</td><td>0.0898</td></tr><tr><td>I-HyperSU (ours)</td><td>0.4188</td><td>2.5208</td><td>0.0251</td></tr></table>

## 3.6 Comparisons With Other Methods

With the A-block fixed and soft relaxation verified, we assess the full I-HyperSU against classical solvers under the same frozen-A, no-extra-training protocol.

Table 4 summarizes the primary comparison. Across all three datasets, the proposed I-HyperSU attains the best reconstruction SAM and joint error, while preserving an abundance RMSE comparable to the strongest classical baseline. This aligns with the method’s design goal: the proposed I-HyperSU prioritizes decomposition consistency under fixed priors rather than maximizing abundance prediction under an oracle model.

To ensure a fair downstream comparison in the frozen-prior regime, external baselines are limited to solvers that neither update A nor rely on extra training data. We also tested SUnSAL under the same fixed-A setting; its results were nearly identical to NNLS under the adopted parameters, so we omit it from the main table for compactness. Relative to UCLS, the proposed I-HyperSU reduces joint error by roughly 69.5%, 67.0%, and 61.7% on Samson, Urban, and Jasper, respectively, while maintaining competitive abundance RMSE. These improvements demonstrate that explicit residual refinement enhances decomposition consistency when endmember priors are fixed.

## 3.7 Synthetic Validation with a Known Residual

On observed data, adding S lowers the reconstruction error by construction, so a good fit alone does not prove that S captures meaningful mismatch. We therefore turn to a synthetic setting to test whether S recovers genuine out-ofmodel structure rather than merely fitting the observation residual.

We build $\mathbf { Y } = \operatorname* { m a x } ( \mathbf { A } _ { \mathrm { g t } } \mathbf { X } _ { \mathrm { g t } } + \mathbf { S } _ { 0 } + \pmb { \eta } , 0 )$ with a known non-negative residual $\mathbf { S } _ { 0 }$ (stray-light/haze + striping) on the Jasper reference. To make the test diagnostic, $\mathbf { S } _ { 0 }$ is constructed as a rank-5 component with a large portion outside span $\mathbf { \Pi } \left( \mathbf { A } _ { \mathrm { g t } } \right)$ , so that it cannot be fully absorbed by abundance changes. We then sweep its strength $\lVert \mathbf { S } _ { 0 } \rVert / \lVert \mathbf { Y } _ { 0 } \rVert \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 \}$ $( { \bf S } _ { 0 } \mathrm { = } 0$ is a control), running the unmodified pipeline with $\mathbf { A } = \mathbf { A } _ { \mathrm { g t } }$ fixed.

![](images/25fbd05c99fa2b4f66fa14ce804a437700cf0dd52a789b65eb8cc3b07e4af632.jpg)  
Fig. 6. Synthetic validation with a known non-negative residual $\mathbf { S } _ { 0 }$ (stray-light/haze + striping; rank $5 ;$ partly out of span(A); true endmembers fixed). (a,b) true $| \mathbf { S } _ { 0 } | \ \mathrm { v s }$ recovered |S| (strength 0.20). (c) recovered vs. true residual. (d) joint error before vs. after S across mismatch strengths, with $\mathrm { c o r r } ( \mathbf { S } , \mathbf { S } _ { 0 } )$ and the $\mathbf { S } _ { 0 } { = } 0$ control.

Three findings $\left( \mathrm { F i g . ~ 6 } \right)$ show that S captures real mismatch. (i) S correlates with the true $\mathbf { S } _ { 0 }$ (Pearson r≈0.77), rising to $r > 0 . 9 9$ on the out-of-subspace component that abundance changes cannot explain. (ii) Routing the residual into S substantially reduces the joint error at every mismatch strength. (iii) The $\mathbf { S } _ { 0 } { = } 0$ control confirms that S does not fabricate structure, with abundance RMSE essentially unchanged. Recovery is partial in magnitude because the inspan $( \mathbf { A } )$ component can be shared with AX, but the ground-truth correlation and zero-mismatch control distinguish genuine recovery from residual fitting.

## 3.8 Prior-Mismatch Stress Test

To map the operating regime of I-HyperSU, we repeat the full pipeline under four extractors of difering quality (N-FINDR, VCA, ATGP, and FIPPI). The residual branch is robust in reconstruction: with S, joint error and reconstruction SAM remain low across extractors $( \mathrm { F i g . 7 a , b } )$ . Abundance accuracy, however, is not immune and degrades for severely wrong priors (FIPPI; Fig. 7c). Thus, S preserves reconstruction consistency but cannot restore abundance identifiability once the endmembers are badly inaccurate. This indicates that I-HyperSU is reconstruction-resilient under imperfect priors, but not prior-independent: accurate abundance recovery still requires reasonably reliable endmembers.

![](images/40d3e2834877ae901c778e708f4dd8fb85a35a60c290e04fd3d100d62266470c.jpg)

![](images/2c9990a63b4d5bf1b8dd1a6306cfcfd8814cb6a10651bde30b7daad8dbea9611.jpg)

![](images/ea32730353a394e954f91a088eaa599d8203176aa3b07cb7595ff59007fa9582.jpg)  
Fig. 7. Prior-mismatch stress test over four extractors (N-FINDR, VCA, ATGP, FIPPI) on the three datasets. (a) X-only joint error (◦) grows with prior SAM while joint error with S (•) stays low. (b) reconstruction SAM after S stays low across extractors. (c) abundance RMSE instead degrades for badly wrong priors.

## 4 Conclusion

This paper reformulated fixed-prior hyperspectral unmixing as an explicit mismatch-optimizing task and introduced an I-HyperSU framework that separates the fixed endmember prior, soft-constrained abundance estimation, and structured residual modeling into three coupled but inspectable components. Experiments on Samson, Urban, and Jasper Ridge show that soft abundance relaxation consistently outperforms hard simplex projection under fixed and imperfect endmember priors, and that the default low-rank-guided DIP residual branch reduces joint reconstruction error by 61.7%–69.5% relative to UCLS with negligible impact on abundance RMSE under the default N-FINDR prior. Synthetic and prior-mismatch experiments further show that S captures genuine out-of-model mismatch (r ≈ 0.77) and improves reconstruction resilience, while accurate abundance recovery still requires a reasonable prior. By separating material abundances from structured spectral mismatch, I-HyperSU provides an interpretable scene-decomposition view of hyperspectral perception under fixed endmember priors.

This study focuses on three standard datasets and a frozen-A configuration. Extending the framework to variability-aware baselines and end-to-end deep methods under a unified frozen-A protocol, and developing reliability-aware refinement that adapts to the quality of the prior, are left for future work.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Beck, A., Teboulle, M.: A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM J. Imag. Sci. 2(1), 183–202 (2009)

2. Bioucas-Dias, J.M., Plaza, A., Dobigeon, N., Parente, M., Du, Q., Gader, $\mathrm { P . , }$ Chanussot, J.: Hyperspectral unmixing overview: Geometrical, statistical, and sparse regression-based approaches. IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens. 5(2), 354–379 (2012)

3. Chang, C.I., Plaza, A.: A fast iterative algorithm for implementation of pixel purity index. IEEE Geosci. Remote Sens. Lett. 3(1), 63–67 (2006)

4. Cheng, C., Sun, D., Yang, Y., Guo, Z., Peng, J.: Hyperspectral image denoising via low-rank tucker decomposition with subspace implicit neural representation. Remote Sens. 17(16), 2867 (2025)

5. Drumetz, L., Veganzones, M.A., Henrot, S., Phlypo, R., Chanussot, J., Jutten, C.: Blind hyperspectral unmixing using an extended linear mixing model to address spectral variability. IEEE Trans. Image Process. 25(8), 3890–3905 (2016)

6. Fu, X., Ma, W.K., Bioucas-Dias, J.M., Chan, T.H.: Semiblind hyperspectral unmixing in the presence of spectral library mismatches. IEEE Trans. Geosci. Remote Sens. 54(9), 5171–5184 (2016)

7. Halimi, A., Honeine, P., Bioucas-Dias, J.M.: Hyperspectral unmixing in presence of endmember variability, nonlinearity, or mismodeling efects. IEEE Trans. Image Process. 25(10), 4565–4579 (2016)

8. Heinz, D.C., Chang, C.I.: Fully constrained least squares linear spectral mixture analysis method for material quantification in hyperspectral imagery. IEEE Trans. Geosci. Remote Sens. 39(3), 529–545 (2001)

9. Iordache, M.D., Bioucas-Dias, J.M., Plaza, A.: Sparse unmixing of hyperspectral data. IEEE Trans. Geosci. Remote Sens. 49(6), 2014–2039 (2011)

10. Iordache, M.D., Bioucas-Dias, J.M., Plaza, A.: Total variation spatial regularization for sparse hyperspectral unmixing. IEEE Trans. Geosci. Remote Sens. 50(11), 4484–4502 (2012)

11. Keshava, N., Mustard, J.F.: Spectral unmixing. IEEE Signal Process. Mag. 19(1), 44–57 (2002)

12. Lawson, C.L., Hanson, R.J.: Solving Least Squares Problems. SIAM, Philadelphia, PA, USA (1995)

13. Nascimento, J.M.P., Bioucas-Dias, J.M.: Vertex component analysis: A fast algorithm to unmix hyperspectral data. IEEE Trans. Geosci. Remote Sens. 43(4), 898–910 (2005)

14. Rasti, B., Koirala, B., Scheunders, P., Ghamisi, P.: UnDIP: Hyperspectral unmixing using deep image prior. IEEE Trans. Geosci. Remote Sens. 60, 1–15 (2022)

15. Ren, H., Chang, C.I.: Automatic spectral target recognition in hyperspectral imagery. IEEE Trans. Aerosp. Electron. Syst. 39(4), 1232–1249 (2003)

16. Thouvenin, P.A., Dobigeon, N., Tourneret, J.Y.: Hyperspectral unmixing with spectral variability using a perturbed linear mixing model. IEEE Trans. Signal Process. 64(2), 525–538 (2016)

17. Ulyanov, D., Vedaldi, A., Lempitsky, V.: Deep image prior. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). pp. 9446–9454 (2018)

18. Wang, P., Liu, R., Zhang, L.: MAT-Net: Multiscale aggregation transformer network for hyperspectral unmixing. IEEE Trans. Geosci. Remote Sens. 62, 5538115 (2024)

19. Winter, M.E.: N-findr: An algorithm for fast autonomous spectral end-member determination in hyperspectral data. In: Proc. SPIE Imaging Spectrometry V. vol. 3753, pp. 266–275 (1999)

20. Zhu, F.: Hyperspectral unmixing: Ground truth labeling, datasets, benchmark performances and survey. arXiv preprint arXiv:1708.05125 (2017)

21. Zhuang, L., Ng, M.K., Fu, X.: Hyperspectral image mixed noise removal using subspace representation and deep CNN image prior. Remote Sens. 13(20), 4098 (2021)

22. Zou, J., Qu, H., Zhang, P.: Conventional to deep learning methods for hyperspectral unmixing: A review. Remote Sens. 17(17), 2968 (2025)