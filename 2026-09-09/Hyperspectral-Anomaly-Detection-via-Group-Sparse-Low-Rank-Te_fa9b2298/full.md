# Hyperspectral Anomaly Detection via Group Sparse Low-Rank Tensor Factorization With Automatic Anomaly Grouping

Quan Yu, Yu-Hong Dai, Xiongjun Zhang

Abstract—Low-rank tensor modeling has become an effective tool for hyperspectral anomaly detection. However, existing methods still sufer from high computational cost and limited flexibility in characterizing spatially structured anomalies. To address these issues, this paper proposes a hyperspectral anomaly detection method based on group sparse low-rank tensor factorization with automatic anomaly grouping (GSAA). Specifically, the low tubal rank background is characterized by imposing group sparsity on tensor factors, which provides an eficient alternative to direct tensor rank regularization. For anomaly modeling, a latent grouping map is introduced to build an automatic anomaly grouping penalty, allowing anomaly groups to be adaptively inferred from the data rather than predefined at the pixel level. To further exploit complementary spectral and spatial information, GSAA is applied in both domains, and the resulting detection maps are fused to form a spectral–spatial version of GSAA, termed GSAA-SS. An eficient linearized alternating direction method of multipliers algorithm with convergence guarantee is developed to solve the resulting model. Experimental results on five real hyperspectral datasets demonstrate that the proposed method achieves superior detection performance and competitive computational eficiency compared with several state-ofthe-art methods.

Index Terms—Hyperspectral anomaly detection, low-rank tensor factorization, group sparsity, automatic anomaly grouping, spectral–spatial fusion.

## I. Introduction

H <sup>YPERSPECTRAL</sup> <sup>images</sup> <sup>(HSIs)</sup> <sup>provide</sup> <sup>rich</sup> <sup>spec-</sup><sub>tral signatures by recording hundreds of narrow and</sub> \_ tral signatures by recording hundreds of narrow and contiguous bands for each spatial location [1]. Owing to this spectral resolution, HSIs have been widely used in denoising [2], [3], image fusion [4], [5], anomaly detection [6], [7], and classification [8]. Among these applications, hyperspectral anomaly detection (HAD) is of particular interest in public safety, surveillance, and defense, where anomalous targets are expected to be identified without prior knowledge of their spectral signatures or class labels [9]. The main dificulty of HAD lies in suppressing complex and heterogeneous backgrounds while preserving weak and spatially structured anomalies.

Existing HAD methods can be roughly grouped into statistical methods, deep learning based methods, and low-rank modeling based methods. Statistical methods usually assume a prescribed distribution for background pixels and detect anomalies by measuring the deviation from the estimated background distribution. The Reed– Xiaoli (RX) detector [10] is a representative method, and many variants have been developed to improve background estimation or detection robustness [11], [12], [13]. However, simple parametric assumptions are often insuficient for real HSIs, whose backgrounds can be highly nonlinear and heterogeneous [14]. Deep learning based methods improve representation ability by exploiting neural networks to learn spectral–spatial features. Representative studies include autoencoder (AE) based models and their variants for unsupervised detection [15], [16], [17], convolutional neural network based methods for supervised detection [18], and several self-supervised architectures [19], such as blind-spot learning, pixel-shufle downsampling blindspot reconstruction, and nonlocal–local feature-coupled schemes [20], [21], [22]. Despite their promising performance, these methods usually require substantial training cost and careful tuning of network architectures and hyperparameters.

Low-rank modeling provides another important route for HAD. It is based on the observation that hyperspectral backgrounds usually exhibit strong spectral– spatial correlations and can be approximated by low-rank structures, whereas anomalies appear as sparse deviations from the background. Low-rank decomposition based methods model an HSI as the superposition of a low-rank background component and a sparse anomaly component, often under the framework of robust principal component analysis. Representative examples include matrix based formulations [23], [24], [25] and tensor based extensions [26], [27]. Low-rank representation based methods further introduce a predefined or learned dictionary to characterize the background subspace, and have been developed in both matrix based [28], [29] and tensor based [30], [31] forms. Since HSIs are naturally third order tensors, tensor based methods are generally more suitable than matrix based methods for preserving the intrinsic spectral–spatial structure of the data.

Although tensor based low-rank methods have achieved encouraging results, two key issues have yet to be adequately addressed. First, most existing methods impose low-rank constraints directly on the background tensor through tensor nuclear norm type surrogates or related nonconvex penalties [7], [30], [32], [33]. The resulting optimization usually involves repeated singular value decompositions (SVDs), which become computationally expensive for large-scale HSIs. Second, many methods characterize the anomaly component by the mixed $\ell _ { 2 }$ $\ell _ { 1 }$ norm or its variants, where each spatial pixel together with its spectral vector is treated as an independent group [31], [34]. This strategy preserves spectral grouping, but it fixes the spatial partition at the pixel level and therefore cannot flexibly describe anomalous regions with unknown shapes and extents.

To address these limitations, this paper proposes an HAD method based on group sparse low-rank tensor factorization with automatic anomaly grouping, termed GSAA. Instead of directly regularizing the background tensor, GSAA imposes group sparsity on tensor factors to characterize the low tubal rank background with lower computational cost. Instead of predefining each pixel as an independent anomaly group, GSAA introduces a latent grouping map to infer the grouping structure of anomalies from the data. The proposed GSAA model is further applied in both spectral and spatial domains, and the resulting detection maps are fused to construct a spectral–spatial version of GSAA, termed GSAA-SS. An eficient linearized alternating direction method of multipliers (LADMM) algorithm is developed for solving the proposed model, and convergence analysis is provided. The main contributions of this paper are summarized as follows.

• We propose a group sparse tensor factorization strategy for low-rank background modeling. By imposing group sparsity on tensor factors, the proposed formulation implicitly characterizes the low tubal rank structure, avoiding direct tensor rank regularization and repeated SVD computations. Its connection to tensor Schatten-p regularization is further established to justify the low-rank modeling ability.

• We develop an automatic anomaly grouping penalty for adaptive anomaly modeling. By introducing a latent grouping map, the proposed penalty learns the grouping structure of anomalies from the data instead of relying on a fixed pixel-level partition, making it more suitable for spatially clustered anomalous targets.

• We construct a GSAA-SS detector by applying GSAA in the spectral and spatial domains and fusing the corresponding detection maps. The proposed model is optimized by an eficient LADMM algorithm with convergence guarantee, and experiments on real hyperspectral datasets validate its detection accuracy and computational eficiency.

The remainder of this paper is organized as follows. Section II introduces the preliminaries of tensor algebras and related notations. We present the GSAA-SS model for HAD in Section III. An LADMM based optimization algorithm with convergence guarantee is developed to solve the resulting model in Section IV. Section V reports the experimental results on several real hyperspectral datasets to show the efectiveness of GSAA-SS. Finally, the conclusions are drawn in Section VI. The proof of the main theorem is deferred to the supplementary material.

## II. Preliminaries

This section summarizes the notation and tensor tools used in the proposed model. For a positive integer n, let $[ n ] : = \{ 1 , 2 , \dots , n \}$ . Scalars, vectors, matrices, and tensors are denoted by lowercase letters (a), bold lowercase letters (a), uppercase letters (A), and calligraphic letters $( \mathcal { A } )$ respectively. The fields of real and complex numbers are denoted by R and C. For a matrix $X \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } } , \ \nabla _ { 1 } X$ and $\nabla _ { 2 } X$ denote the first-order forward finite-diference operators in the vertical and horizontal directions, respectively.

For a third order tensor $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , its $( i , j , k ) \mathrm { t h }$ entry is denoted by $\mathcal { X } _ { i j k }$ or $\mathcal { X } ( i , j , k )$ , and its kth frontal slice is denoted by $X ^ { ( k ) }$ . For two tensors $\boldsymbol { \mathcal { X } } , \boldsymbol { \mathcal { V } } \in$ R<sup>n</sup>1<sup>×n</sup>2<sup>×n</sup>3 , their inner product is defined as $\langle \mathcal { X } , \mathcal { Y } \rangle =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { n _ { 1 } } \sum _ { j = 1 } ^ { n _ { 2 } } \sum _ { k = 1 } ^ { n _ { 3 } } \chi _ { i j k } \mathcal { Y } _ { i j k } , } \end{array}$ , and the Frobenius norm is defined as $\| { \mathcal { X } } \| = { \sqrt { \langle { \mathcal { X } } , { \mathcal { X } } \rangle } }$ . The $\ell _ { 0 } { \mathrm { - n o r m } }$ of $\mathcal { X }$ counts the number of its nonzero entries. We use X<sup>¯</sup> to denote the discrete Fourier transform (DFT) of X along the third mode, i.e., $\bar { \mathcal { X } } = \mathrm { { f f t } } ( \mathcal { X } , [ ] , 3 )$ , and $\mathcal { X } = \mathrm { i f f t } ( \bar { \mathcal { X } } , [ \mathbf { \Lambda } ] , 3 )$ denotes the inverse transform.

Definition 2.1: (f-diagonal tensor) [35] A tensor is called f-diagonal if each of its frontal slices is a diagonal matrix.

Definition 2.2: (conjugate transpose) [35] The conjugate transpose of a tensor $\mathcal { Z } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , denoted by $\mathcal { Z } ^ { \top }$ is defined as the tensor obtained by taking the conjugate transpose of each frontal slice and then reversing the order of the transposed frontal slices from the second to the last.

Definition 2.3: (identity tensor) [35] The identity tensor $\mathcal { T } \in \mathbb { R } ^ { n \times n \times n _ { 3 } }$ is defined as a tensor whose first frontal slice is the identity matrix, while all other frontal slices consist entirely of zeros.

Definition 2.4: (orthogonal tensor) [35] A tensor $\mathcal { P } \in$ R<sup>n×n×n</sup>3 is said to be orthogonal if $\mathcal { P } ^ { \top } \ast \mathcal { P } = \mathcal { P } \ast \mathcal { P } ^ { \top } = \mathcal { I } .$ where I is the identity tensor.

Definition 2.5: (t-product [35]) For tensors $\qquad x \in$ R<sup>n</sup>1<sup>×r×n</sup>3 and $\mathcal { V } \in \mathbb { R } ^ { r \times n _ { 2 } \times n _ { 3 } }$ , the t-product is defined as

$$
{ \mathcal { X } } * { \mathcal { Y } } : = \operatorname { F o l d } \left( \operatorname { b c i r c } ( { \mathcal { X } } ) \ { \mathrm { ~ . ~ U n f o l d } } ( { \mathcal { Y } } ) \right) \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } } .
$$

Here,

$$
\mathrm { b c i r c } ( \mathcal X ) = \left[ \begin{array} { c c c c } { X ^ { ( 1 ) } } & { X ^ { ( n _ { 3 } ) } } & { \cdot \cdot \cdot } & { X ^ { ( 2 ) } } \\ { X ^ { ( 2 ) } } & { X ^ { ( 1 ) } } & { \cdot \cdot } & { X ^ { ( 3 ) } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { X ^ { ( n _ { 3 } ) } } & { X ^ { ( n _ { 3 } - 1 ) } } & { \cdot \cdot } & { X ^ { ( 1 ) } } \end{array} \right] ,
$$

$\mathrm { U n f o l d } ( { \mathcal V } ) ~ = ~ \left. Y ^ { ( 1 ) } ; Y ^ { ( 2 ) } ; . . . ; Y ^ { ( n _ { 3 } ) } \right. ~ \in ~ \mathbb { R } ^ { n _ { 3 } r \times n _ { 2 } }$ , and its inverse operator $\mathrm { \Phi ^ { 6 6 } F o l d ^ { 5 } }$ is defined by ${ \mathrm { F o l d } } ( { \mathrm { U n f o l d } } ( { \mathcal { V } } ) ) =$ $\mathcal { N } .$

Theorem 2.1: (t-SVD [35]) Any tensor $\mathcal { Z } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ admits a tensor singular value decomposition $\left( \mathrm { { t - S V D } } \right)$ of the form ${ \mathcal Z } = { \mathcal U } _ { { \mathcal Z } } * S _ { { \mathcal Z } } * { \mathcal V } _ { { \mathcal Z } } ^ { \top }$ , where $\mathcal { U } _ { \mathcal { Z } } ~ \in ~ \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times n _ { 2 } }$ ×n<sub>3</sub> and $\mathcal { V } _ { \mathcal { Z } } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 2 } \times }$ <sup>n</sup>3 are orthogonal tensors, and $\mathit { S z } \in$ R<sup>n</sup>1<sup>×n</sup>2<sup>×n</sup>3 is an f-diagonal tensor.

Definition 2.6: (tubal rank [36]) The tensor tubal rank is the number of nonzero singular tubes in $\boldsymbol { \mathcal { S } } _ { \mathcal { Z } }$ , i.e., $\operatorname { r a n k } _ { t } ( { \mathcal { Z } } ) = \# \{ i : { \mathcal { S } } _ { { \mathcal { Z } } } ( i , i , : ) \neq 0 \}$ , where $\boldsymbol { \mathcal { S } } _ { \mathcal { Z } }$ is obtained from the t-SVD of $\mathcal { Z } = \mathcal { U } _ { \mathcal { Z } } * \mathcal { S } _ { \mathcal { Z } } * \mathcal { V } _ { \mathcal { Z } } ^ { \top }$

Definition 2.7: (mode-i unfolding and folding) Let ${ \mathcal { Z } } \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ be a third order tensor. Its mode-i unfolding is the matrix $Z _ { ( i ) } = \mathrm { u n f o l d } _ { ( i ) } ( \mathcal { Z } ) \in \mathbb { R } ^ { n _ { i } \times \prod _ { s \neq i } n _ { s } }$ , whose columns are mode-i fibers arranged lexicographically over the remaining modes. The inverse operation is denoted by $\operatorname { f o l d } _ { ( i ) } ( Z _ { ( i ) } ) = \mathcal { Z }$

Definition 2.8: (tensor Schatten-p norm [37]) For a tensor $\mathcal { Z } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ with t-SVD $\mathcal { Z } = \mathcal { U } _ { \mathcal { Z } } \ast \stackrel { \cdot } { S } _ { \mathcal { Z } } \ast \mathcal { V } _ { \mathcal { Z } } ^ { \top }$ the tensor Schatten-p norm is defined as $\begin{array} { r l } { \| \mathcal { Z } \| _ { S _ { p } } ^ { p } } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { n _ { 3 } } \sum _ { k = 1 } ^ { n _ { 3 } } \sum _ { i = 1 } ^ { \operatorname* { m i n } \{ n _ { 1 } , n _ { 2 } \} } \left( \bar { S } _ { \mathcal { Z } } ( i , i , k ) \right) ^ { p } } \end{array}$

## III. A GSAA-SS model for HAD

This section presents the proposed GSAA-SS model for HAD. We first revisit the standard low-rank and sparse decomposition view of HAD, which separates an observed HSI into a structured background and a sparse anomaly component. We then develop two modeling components: a group sparse tensor factorization for eficient lowrank background modeling, and an automatic anomaly grouping (AAG) penalty for adaptive anomaly modeling. These two components are integrated into a unified GSAA model, which is further applied in both the spectral and spatial domains. Then the resulting detection maps are fused to obtain the final GSAA-SS detector.

## A. General Tensor Based Detection Framework

A broad class of tensor based HAD models can be written as

$$
\underset { \mathcal { Z } , \mathcal { E } } { \mathrm { a r g } \mathrm { m i n ~ } } \ \mathrm { r a n k } ( \mathcal { Z } ) + \gamma \| \mathcal { E } \| _ { \mathrm { s p a r s e } } , \quad \mathrm { s . t . } \quad \mathcal { H } = \mathcal { Z } + \mathcal { E } .\tag{1}
$$

Here, H denotes the observed hyperspectral tensor, $\mathcal { Z }$ represents the background, and $\mathcal { E }$ denotes the anomaly component. Model (1) follows a common assumption in HAD: the background contains strong spectral–spatial correlation and can be approximated by a low-rank tensor, whereas anomalies appear as sparse deviations from this structured background.

The practical efectiveness of (1) depends critically on how the low-rank and sparse terms are regularized. For the background component, directly minimizing rank(Z) is NP-hard. Existing tensor methods therefore replace it with tractable surrogates such as the weighted nuclear norm [30], the ε-shrinkage tensor nuclear norm [32], and unified nonconvex penalty functions [7], [33]. Although these surrogates are efective in modeling low-rank structure, they typically require repeated SVDs, which constitute a major computational bottleneck for large-scale HSIs.

For the anomaly component, many methods exploit the fact that each spatial location is associated with a full spectral vector and replace $\| \mathcal { E } \| _ { \mathrm { s p a r s e } }$ with the mixed $\ell _ { 2 } - \ell _ { 1 }$ norm $\begin{array} { r } { \| \mathcal { E } \| _ { 2 , 1 } : = \sum _ { i j } \| \mathcal { E } ( i , j , : ) \| } \end{array}$ or its nonconvex variants [31], [38], [39]. This choice preserves spectral grouping, but it fixes the spatial grouping a priori by treating each pixel as an independent group. Such a finest partition assumption is often too restrictive for HAD, where anomalous targets usually occupy spatially clustered regions with unknown shapes and extents.

The proposed GSAA framework addresses these two issues in a coordinated manner. We first derive a group sparse factorization for eficient low-rank background modeling, and then introduce an automatic anomaly grouping mechanism that learns the spatial grouping structure of anomalies directly from the data.

## B. Group Sparse Low-Rank Tensor Factorization

To model the low-rank background eficiently, we factorize the background tensor and impose group sparsity on the tensor factors. The key idea is that the active lateral slices of the factors determine the efective tubal rank of the reconstructed background. Therefore, lowrank structure can be encouraged through factor sparsity instead of direct rank regularization on $\mathcal { Z }$

Theorem 3.1: For any tensor $\mathcal { Z } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ with $r : =$ rank<sub>t</sub> $\left( \mathcal Z \right) \leq d \leq \operatorname* { m i n } \left\{ n _ { 1 } , n _ { 2 } \right\}$ , we have

(1) rank $\begin{array} { r l r } { { \bf \mathcal { Z } } ) } & { = } & { { \bf \frac { 1 } { 2 } } \operatorname* { m i n } _ { \mathcal { Z } = \mathcal { X } * \mathcal { Y } ^ { \top } } { \bf \Xi } \big ( \| \mathcal { X } \| _ { F , 0 } ~ + ~ \| \mathcal { Y } \| _ { F , 0 } \big ) ~ = ~ } \end{array}$ min $\begin{array} { r } { { \llangle z } _ { = \mathcal { X } * \mathcal { Y } ^ { \top } } \left. \mathcal { X } \right. _ { F , 0 } = \operatorname* { m i n } _ { \mathcal { Z } = \mathcal { X } * \mathcal { Y } ^ { \top } } \left. \mathcal { Y } \right. _ { F , 0 } ; } \end{array}$

(2) $\begin{array} { r } { \frac { 1 } { p } \| \mathcal { Z } \| _ { S _ { p } } ^ { p } \leq \operatorname* { m i n } _ { \mathcal { Z } = \mathcal { X } \ast \mathcal { Y } ^ { \top } } \left( \frac { 1 } { p _ { 1 } } \| \mathcal { X } \| _ { F , p _ { 1 } } + \frac { 1 } { p _ { 2 } } \| \mathcal { Y } \| _ { F , p _ { 2 } } \right) \leq } \end{array}$ $\begin{array} { r l r } {  { \big ( \frac { n _ { 3 } ^ { 1 - p _ { 1 } / 2 } } { p _ { 1 } } + \frac { n _ { 3 } ^ { 1 - p _ { 2 } / 2 } } { p _ { 2 } } \big ) \| \mathcal { Z } \| _ { S _ { \tau } } ^ { p } } } \end{array}$ for any $p _ { 1 } , p _ { 2 } \in ( 0 , 2 ]$ satisfying $1 / p \stackrel {  } { = } 1 / p _ { 1 } + 1 \stackrel { \cdot } { / } p _ { 2 }$

Here $\begin{array} { r } { \Vert \mathcal { Z } \Vert _ { F , p } = \sum _ { j } \Vert \mathcal { Z } ( : , j , : ) \Vert ^ { p } , \mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times d \times n _ { 3 } } } \end{array}$ and $\mathcal { V } \in$ R<sup>n</sup>2<sup>×d×n</sup>3.

Remark 3.1: Theorem 3.1 explains why group sparsity on tensor factors can serve as a low-rank prior. The first statement shows that the tubal rank of $\mathcal { Z }$ can be exactly characterized by the number of active groups in X and Y. The second statement further connects group sparse factor penalties with tensor Schatten-p regularization, which supports the low-rank modeling ability of the proposed factorized formulation.

Remark 3.2: The factorized formulation is also computationally favorable. Since low-rankness is imposed through factor regularization, the optimization avoids direct tensor rank regularization and the associated repeated SVD computations. Moreover, the auxiliary dimension d is updated adaptively and, in practice, approaches the tubal rank $r ,$ which is usually much smaller than min $\{ n _ { 1 } , n _ { 2 } \}$ This leads to a more eficient background model for large HSIs.

## C. Automatic Anomaly Grouping

Existing structured sparsity models for HAD usually assume that anomaly groups are known as a prior. A common choice is the mixed $\ell _ { 2 } / \ell _ { 1 }$ norm

$$
\| \mathcal { E } \| _ { F , 1 } ^ { B } = \sum _ { l = 1 } ^ { L } \sqrt { | \mathcal { B } _ { l } | } \| \mathcal { E } _ { B _ { l } } \| ,
$$

where $\{ B _ { l } \} _ { l = 1 } ^ { L }$ is a prescribed block partition. In hyperspectral applications, the most common partition treats the full spectral vector at each spatial location as a single group, namely,

$$
{ \cal B } = \{ \{ ( 1 , 1 , : ) \} , \{ ( 1 , 2 , : ) \} , \ldots , \{ \left( n _ { 1 } , n _ { 2 } , : \right) \} \} ,
$$

which yields $\| \mathcal { E } \| _ { 2 , 1 }$ . This design preserves spectral grouping, but it fixes the spatial partition at the pixel level and cannot adapt to anomalous regions with unknown spatial extent.

To overcome this limitation, we introduce an automatic anomaly grouping (AAG) penalty. The spectral dimension is always kept within each group, while the spatial partition is learned from the data. Specifically, we define

$$
\Psi \big ( \boldsymbol { \mathcal { E } } \big ) = \operatorname* { m i n } _ { \boldsymbol { L } , \boldsymbol { B _ { l } } } \sum _ { l = 1 } ^ { L } \sqrt { | \boldsymbol { B _ { l } } | / n _ { 3 } } \| \mathcal { E } _ { \boldsymbol { B _ { l } } } \| ,\tag{2}
$$

where each anomaly block is defined by $B _ { l } = \Omega _ { l } \times \left[ n _ { 3 } \right]$ and $\{ \Omega _ { l } \} _ { l = 1 } ^ { L }$ forms a partition of the spatial index set $[ n _ { 1 } ] \times [ n _ { 2 } ]$ . In contrast to conventional structured sparsity penalties, the spatial partition is optimized rather than prescribed in advance.

The direct optimization of $\Psi ( \mathcal { E } )$ is combinatorial because the spatial partition is unknown. To obtain a tractable formulation, we rewrite each block term using an auxiliary scalar shared by all pixels in the same spatial group. Following the variational representation in [40, Lemma 1], define

$$
\phi ( e , \vartheta ) : = \left\{ \begin{array} { l l } { \frac { | e | ^ { 2 } } { 2 \vartheta } + \frac { \vartheta } { 2 } , } & { \mathrm { i f ~ } \vartheta > 0 ; } \\ { 0 , } & { \mathrm { i f ~ } e = 0 \mathrm { ~ a n d ~ } \vartheta = 0 ; } \\ { \infty , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Then each block penalty admits the representation

$$
\sqrt { | \mathscr { B } _ { l } | / n _ { 3 } } \| \mathcal { E } _ { \mathscr { B } _ { l } } \| = \operatorname* { m i n } _ { \vartheta _ { l } \in \mathbb { R } } \sum _ { ( i , j ) \in \Omega _ { l } } \phi \big ( \| \mathcal { E } _ { i j \colon } \| , \vartheta _ { l } \big ) ,\tag{3}
$$

where the minimum is attained at $\vartheta _ { l } = \sqrt { n _ { 3 } / | B _ { l } | } \| \mathcal { E } _ { B _ { l } } \|$ Therefore, the original combinatorial penalty can be equivalently rewritten as

$$
\Psi \big ( \mathcal { E } \big ) = \operatorname* { m i n } _ { L , \mathcal { B } _ { l } , \vartheta _ { l } } \sum _ { l = 1 } ^ { L } \sum _ { ( i , j ) \in \Omega _ { l } } \phi \big ( \| \mathcal { E } _ { i j : \cdot } \| , \vartheta _ { l } \big ) .
$$

This representation leads to a latent grouping map. We introduce $\boldsymbol { \theta } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ by setting $\Theta _ { i j } = \vartheta _ { l }$ for all $( i , j ) \in$ $\Omega _ { l }$ . Thus, pixels in the same spatial group share the same value in Θ, and connected constant regions of Θ encode the learned spatial partition. Boundaries between neighboring groups correspond to nonzero entries in $\nabla _ { 1 } \Theta$ and $\nabla _ { 2 } \Theta$

Therefore, sparsity of these finite diferences controls the complexity of the learned grouping structure.

Based on this observation, we relax the combinatorial partition optimization as

$$
\Psi _ { \pmb { \alpha } } ( \pmb { \mathcal { E } } ) = \operatorname* { m i n } _ { \theta \in \mathbb { S } _ { \alpha } } \sum _ { i = 1 } ^ { n _ { 1 } } \sum _ { j = 1 } ^ { n _ { 2 } } \phi ( \big \| \mathscr { E } _ { i j : } \big \| , \theta _ { i j } \big ) ,
$$

where $\mathbb { S } _ { \alpha } : = \{ \theta : \| \nabla _ { 1 } \theta \| _ { 0 } \ \le \ \alpha _ { 1 } , \| \nabla _ { 2 } \theta \| _ { 0 } \ \le \ \alpha _ { 2 } \}$ , and ${ \pmb { \alpha } } = ( \alpha _ { 1 } , \alpha _ { 2 } ) \in \mathbb { Z } _ { + } ^ { 2 }$ . The parameters α<sub>1</sub> and $\alpha _ { 2 }$ control the number of vertical and horizontal changes in Θ, respectively, and thus determine the flexibility of the spatial grouping map.

The AAG penalty interpolates between two meaningful extremes, as formalized below.

Theorem 3.2: For a tensor $\mathcal { E } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , we have

(1) $\Psi _ { ( 0 , 0 ) } ( \mathcal { E } ) = \sqrt { n _ { 1 } n _ { 2 } } \| \mathcal { E } \| ;$

(2) $\begin{array} { r } { \operatorname* { l i m } _ { ( \alpha _ { 1 } , \alpha _ { 2 } ) \to ( \infty , \infty ) } \Psi _ { ( \alpha _ { 1 } , \alpha _ { 2 } ) } ( \mathcal E ) = \| \mathcal E \| _ { 2 , 1 } . } \end{array}$

Remark 3.3: Theorem 3.2 shows that $\Psi _ { { \pmb { \alpha } } }$ bridges the coarsest and finest spatial partitions. Thus, it preserves spectral group sparsity while allowing the spatial grouping pattern to adapt to the data.

Remark 3.4: Compared with $\| \cdot \| _ { 2 , 1 }$ based detectors, AAG estimates both the anomaly tensor E and the grouping map Θ. The piecewise constant structure of Θ encodes learned spatial groups and provides a complementary spatially coherent anomaly response.

## D. Proposed GSAA-SS Model

By combining the group sparse low-rank background factorization with the AAG penalty, we obtain the unified GSAA model:

$$
\begin{array} { r l } { \underset { \mathcal { X } , \mathcal { Y } , \mathcal { E } , \Theta } { \arg \operatorname* { m i n } } } & { \| \mathcal { X } \| _ { F , p } + \| \mathcal { Y } \| _ { F , p } + \gamma \mathcal { F } ( \mathcal { E } , \Theta ) } \\ { \mathrm { s . t . } } & { \mathcal { H } = \mathcal { X } * \mathcal { Y } ^ { \top } + \mathcal { E } , \ \theta \in \mathbb { S } _ { \alpha } . } \end{array}\tag{4}
$$

where $\begin{array} { r } { \varPhi ( \mathcal { E } , \theta ) = \sum _ { i = 1 } ^ { n _ { 1 } } \sum _ { i = 1 } ^ { n _ { 2 } } \phi ( \| \mathcal { E } _ { i j : } \| , \theta _ { i j } ) } \end{array}$

In model (4), the factors X and Y model the low-rank background through group sparse regularization, whereas the pair (E, Θ) captures the anomaly component together with its adaptively learned spatial grouping structure.

To further exploit the complementary spectral and spatial information in HSIs, we extend the unified GSAA model to a spectral–spatial detection framework, termed GSAA-SS. Specifically, GSAA is separately applied in the spectral and spatial domains of the same HSI H ∈ R<sup>n</sup>1<sup>×n</sup>2<sup>×n</sup>3, and the resulting detection maps are then fused. In the spectral branch, a matrix version of GSAA is applied to the mode-3 unfolding of H to characterize the common background spectral subspace. Pixels that are poorly represented by this subspace yield large residuals and are therefore identified as spectral anomalies. In the spatial branch, principal component analysis (PCA) [41] is first used to reduce the spectral dimension and obtain a compact tensor $\tilde { \mathcal { H } } \in \mathbb { R } ^ { \bar { n } _ { 1 } \times n _ { 2 } \times b }$ with $b \ll n _ { 3 } .$ while preserving the main spatial structure [42]. GSAA is then applied to $\tilde { \mathcal { H } }$ within the t-product framework to capture its low-rank background spatial structure. The overall workflow is illustrated in Fig. 1 and consists of the following three stages.

(1) Spectral-domain detection: The HSI is first unfolded along the spectral mode into the matrix $H _ { ( 3 ) }$ , and a matrix version of GSAA is then applied to the unfolded data:

$$
\begin{array} { r l } { \underset { X , Y , E , \Theta } { \mathrm { a r g } \mathrm { m i n } } } & { \| X \| _ { F , p } + \| Y \| _ { F , p } + \gamma \varPhi ( \mathrm { f o l d } _ { ( 3 ) } ( E ) , \theta ) } \\ { \mathrm { s . t . } \quad } & { H _ { ( 3 ) } = X Y ^ { \top } + E , \ \theta \in \mathbb { S } _ { \alpha } . } \end{array}
$$

The optimized AAG map in this branch is denoted by $\Theta _ { s p e } .$

(2) Spatial domain detection: For the tensor $\tilde { \mathcal { H } }$ obtained by PCA, GSAA is applied within the t-product framework:

$$
\begin{array} { r l } { \underset { \mathcal { X } , \mathcal { Y } , \mathcal { E } , \Theta } { \mathrm { a r g } \mathrm { m i n } } } & { \| \mathcal { X } \| _ { F , p } + \| \mathcal { Y } \| _ { F , p } + \gamma \mathcal { F } ( \mathcal { E } , \Theta ) } \\ { \mathrm { s . t . } } & { \tilde { \mathcal { H } } = \mathcal { X } * \mathcal { Y } ^ { \top } + \mathcal { E } , \ \theta \in \mathbb { S } _ { \alpha } . } \end{array}
$$

The optimized AAG map in this branch is denoted by $\theta _ { s p a }$

(3) Spectral–spatial fusion: The AAG maps $\theta _ { s p e }$ and $\theta _ { s p a }$ obtained from the spectral and spatial branches, respectively, are fused via element-wise multiplication:

$$
\begin{array} { r } { \Theta _ { f u s } = \Theta _ { s p a } \odot \Theta _ { s p e } , } \end{array}
$$

where ⊙ denotes the Hadamard product.

![](images/83e9678165e7bf7bf964af4eee6157fecdf12bf74383b59163b024fbdcc83322.jpg)  
Fig. 1: Flow chart of GSAA-SS.

## IV. Optimization by Linearized Alternating Direction Method of Multipliers

This section develops an eficient LADMM algorithm for solving the proposed GSAA model and provides convergence analysis.

## A. LADMM Algorithm

To account for Gaussian noise in real HSI observations, the constrained GSAA model in (4) is converted into the following penalized formulation:

$$
\begin{array} { r l } { \underset { \boldsymbol { \mathcal { X } } , \boldsymbol { \mathcal { Y } } , \boldsymbol { \mathcal { E } } , \boldsymbol { \Theta } \in \mathbb { S } _ { \alpha } } { \arg \operatorname* { m i n } } \| \boldsymbol { \mathcal { X } } \| _ { F , p } + \| \boldsymbol { \mathcal { V } } \| _ { F , p } + \gamma \boldsymbol { \mathcal { P } } ( \boldsymbol { \mathcal { E } } , \boldsymbol { \Theta } ) } & { } \\ { + \displaystyle \frac { \lambda } { 2 } \big \| \boldsymbol { \mathcal { X } } * \boldsymbol { \mathcal { V } } ^ { \top } + \boldsymbol { \mathcal { E } } - \boldsymbol { \mathcal { H } } \big \| ^ { 2 } . } \end{array}\tag{5}
$$

By introducing four auxiliary variables ${ \mathcal { N } } , \Upsilon , U$ and $V ,$ problem (5) can be rewritten as:

$$
\begin{array} { r l } { \underset { \boldsymbol { x } , \boldsymbol { y } , \boldsymbol { \varepsilon } , \boldsymbol { \Theta } , \mathcal { N } , \Upsilon , \boldsymbol { U } , \boldsymbol { V } } { \arg \operatorname* { m i n } } } & { \left\| \boldsymbol { \mathcal { X } } \right\| _ { \boldsymbol { F } , \boldsymbol { p } } + \left\| \boldsymbol { \mathcal { V } } \right\| _ { \boldsymbol { F } , \boldsymbol { p } } + \gamma \mathcal { F } ( \boldsymbol { \mathcal { E } } , \boldsymbol { \Theta } ) + \frac { \lambda } { 2 } \big \| \mathcal { X } } \\ & { \boldsymbol { * } \boldsymbol { \mathcal { V } } ^ { \intercal } + \boldsymbol { N } - \boldsymbol { \mathcal { H } } \big \| ^ { 2 } + \delta _ { \mathrm { S _ { 1 } } } ( \boldsymbol { U } ) + \delta _ { \mathrm { S _ { 2 } } } \big ( \boldsymbol { V } \big ) } \\ { \mathrm { s . t . } } & { \boldsymbol { \mathcal { N } } = \boldsymbol { \mathcal { E } } , \Upsilon = \boldsymbol { \Theta } , \ : U = \nabla _ { 1 } \Upsilon , \ : V = \nabla _ { 2 } \Upsilon , } \end{array}\tag{6}
$$

where $\delta _ { \mathbb { S } } ( \cdot )$ is the indicator function, $\mathbb { S } _ { 1 } = \{ U : \| U \| _ { 0 } \le $ $\alpha _ { 1 } \}$ , and $\mathbb { S } _ { 2 } ~ = ~ \{ V ~ : ~ \| V \| _ { 0 } ~ \leq ~ \alpha _ { 2 } \}$ The augmented Lagrangian function of (6) is given by

$$
\begin{array} { l } { { \displaystyle \mathbb { L } ( \mathcal { X } , \mathcal { Y } , \mathcal { E } , \theta , \mathcal { N } , \Upsilon , U , V ; \{ \Lambda _ { u } \} _ { u = 1 } ^ { 4 } , \{ \beta _ { u } \} _ { u = 1 } ^ { 2 } ) } } \\ { { \displaystyle = \| \mathcal { X } \| _ { F , p } + \| \mathcal { Y } \| _ { F , p } + \gamma \phi ( \mathcal { E } , \theta ) + \frac { \lambda } { 2 } \big \| \mathcal { X } * \mathcal { Y } ^ { \top } + \mathcal { N } - \mathcal { H } \big \| ^ { 2 } } } \\ { { \displaystyle \quad + \delta _ { \mathtt { S } _ { 1 } } \big ( U \big ) + \delta _ { \mathtt { S } _ { 2 } } \big ( V \big ) + \big \langle \Lambda _ { 1 } , \mathcal { E } - \mathcal { N } \big \rangle + \frac { \beta _ { 1 } } { 2 } \big \| \mathcal { E } - \mathcal { N } \big \| ^ { 2 } } } \\ { { \displaystyle \quad + \big \langle \Lambda _ { 2 } , \theta - \Upsilon \big \rangle + \frac { \beta _ { 1 } } { 2 } \big \| \theta - \Upsilon \big \| ^ { 2 } + \big \langle \Lambda _ { 3 } , \nabla _ { 1 } \Upsilon - U \big \rangle } } \\ { { \displaystyle \quad + \frac { \beta _ { 2 } } { 2 } \big \| \nabla _ { 1 } \Upsilon - U \big \| ^ { 2 } + \big \langle \Lambda _ { 4 } , \nabla _ { 2 } \Upsilon - V \big \rangle + \frac { \beta _ { 2 } } { 2 } \big \| \nabla _ { 2 } \Upsilon - V \big \| ^ { 2 } . } } \end{array}
$$

where $\Lambda _ { u } , u \in [ 4 ]$ are the Lagrange multipliers, and $\beta _ { u } >$ $0 , u \in [ 2 ]$ are the penalty parameters. The variables are updated alternately, and the resulting subproblems admit closed-form or proximal solutions as described below.

1) Update $\hat { \mathcal X } ^ { t + 1 }$ and $\mathcal { V } ^ { t + 1 }$ : The sub-problem to update X is

$$
\underset { \mathcal { X } } { \arg \operatorname* { m i n } } \ \| \mathcal { X } \| _ { F , p } + \lambda f ( \mathcal { X } , \mathcal { Y } ^ { t } ) ,
$$

where $\begin{array} { r } { f ( \mathcal { X } , \mathcal { Y } ) = \frac { 1 } { 2 } \| \mathcal { X } * \mathcal { Y } ^ { \top } + \mathcal { N } ^ { t } - \mathcal { H } \| ^ { 2 } } \end{array}$ . To address the aforementioned problem, we linearize the term $f ( \mathcal { X } , \mathcal { Y } ^ { t } )$ at the current iterate point $\mathcal { X } ^ { t }$ . Consequently, the original problem can be reformulated in a relaxed manner as

$$
\begin{array} { r l } { \underset { \pmb { \mathscr { X } } } { \arg \operatorname* { m i n } } } & { \| \pmb { \mathscr { X } } \| _ { F , p } + \lambda \big \langle \nabla _ { \pmb { \mathscr { X } } } f ( \pmb { \mathscr { X } } ^ { t } , \pmb { \mathscr { Y } } ^ { t } ) , \pmb { \mathscr { X } } - \pmb { \mathscr { X } } ^ { t } \big \rangle } \\ & { \qquad + \frac { \lambda \mathcal { l } _ { \pmb { \mathscr { X } } } ^ { t } } { 2 } \| \pmb { \mathscr { X } } - \pmb { \mathscr { X } } ^ { t } \| ^ { 2 } . } \end{array}
$$

Then the update of X is given by

$$
\begin{array} { r } { \mathcal { X } ^ { t + 1 } = \operatorname { p r o x } _ { 1 / ( \lambda l _ { \mathcal { X } } ^ { t } ) \parallel \cdot \parallel _ { F , p } } \big ( \mathcal { X } ^ { t } - \nabla _ { \mathcal { X } } f ( \mathcal { X } ^ { t } , \mathcal { Y } ^ { t } ) / l _ { \mathcal { X } } ^ { t } \big ) , } \end{array}\tag{7}
$$

where $\nabla _ { \mathcal { X } } f ( \mathcal { X } ^ { t } , \mathcal { Y } ^ { t } ) = { ( \mathcal { X } ^ { t } * \mathcal { Y } ^ { t } } ^ { \top } + \mathcal { N } ^ { t } - \mathcal { H } ) * \mathcal { Y } ^ { t }$ and $l _ { \mathcal { X } } ^ { t } =$ $\| \mathcal { V } ^ { t } \| _ { 2 } ^ { 2 } + \varepsilon$ with $\varepsilon > 0$

The sub-problem to update Y is

$$
\underset { \mathcal { Y } } { \arg \operatorname* { m i n } } \ \| \mathcal { Y } \| _ { F , p } + \lambda f ( \mathcal { X } ^ { t + 1 } , \mathcal { Y } ) .
$$

Analogous to the update of X , the update of $\mathcal { V }$ is given by

$$
\mathcal { V } ^ { t + 1 } = \mathrm { p r o x } _ { 1 / ( \lambda l _ { y } ^ { t } ) \mid \mid \cdot \mid _ { F , p } } \big ( \mathcal { V } ^ { t } - \nabla _ { \mathcal { V } } f ( \mathcal { X } ^ { t + 1 } , \mathcal { V } ^ { t } ) / l _ { \mathcal { V } } ^ { t } \big ) ,\tag{8}
$$

where $\nabla y f ( \mathcal X ^ { t + 1 } , \mathcal Y ^ { t } ) = ( \mathcal X ^ { t + 1 } * \mathcal y ^ { t } { } ^ { \top } + \mathcal N ^ { t } - \mathcal H ) ^ { \top } * \mathcal X ^ { t + 1 }$ and $l _ { \mathcal { V } } ^ { t } = \| \mathcal { X } ^ { t + 1 } \| _ { 2 } ^ { 2 } + \varepsilon$

2) Update $\mathcal { E } ^ { t + 1 }$ and $\Theta ^ { t + 1 }$ : The sub-problem to update (E, Θ) is

$$
\underset { \varepsilon , \theta } { \arg \operatorname* { m i n } } \ \gamma \varPhi ( \mathcal { E } , \theta ) + \left. \Lambda _ { 1 } ^ { t } , \mathcal { E } - \mathcal { N } ^ { t } \right. + \frac { \beta _ { 1 } ^ { t } } { 2 } \| \mathcal { E } - \mathcal { N } ^ { t } \| ^ { 2 }
$$

$$
\begin{array} { r l } & { \quad + \left. \Lambda _ { 2 } ^ { t } , \theta - \Upsilon ^ { t } \right. + \displaystyle \frac { \beta _ { 1 } ^ { t } } { 2 } \| \theta - \Upsilon ^ { t } \| ^ { 2 } } \\ & { = \operatorname { p r o x } _ { \gamma / \beta _ { 1 } ^ { t } \varPhi } \bigl ( \mathscr { N } ^ { t } - \Lambda _ { 1 } ^ { t } / \beta _ { 1 } ^ { t } , \Upsilon ^ { t } - \Lambda _ { 2 } ^ { t } / \beta _ { 1 } ^ { t } \bigr ) . } \end{array}\tag{9}
$$

$$
\begin{array} { r l r } {  { 3 \ \mathrm { U p d a t e } \ \mathcal { N } ^ { t + 1 } \colon \mathrm { T h e ~ s u b \mathrm { - } p r o b l e m ~ t o ~ u p d a t e } \ \mathcal { N } \ \mathrm { i } } } \\ & { } & { \underset { \mathcal { N } } { \mathrm { a r g } \mathrm { m i n } } \ \frac { \lambda } { 2 } \| \mathcal { X } ^ { t + 1 } \ast \mathcal { Y } ^ { t + 1 } \overset { \intercal } { + } \mathcal { N } - \mathcal { H } \| ^ { 2 } } \\ & { } & { +  \Lambda _ { 1 } ^ { t } , \mathcal { E } ^ { t + 1 } - \mathcal { N }  + \frac { \beta _ { 1 } ^ { t } } { 2 } \| \mathcal { E } ^ { t + 1 } - \mathcal { N } \| ^ { 2 } } \\ & { } & { = ( \lambda ( \mathcal { H } - \mathcal { X } ^ { t + 1 } \ast \mathcal { Y } ^ { t + 1 } \overset { \intercal } { ) } + \Lambda _ { 1 } ^ { t } + \beta _ { 1 } ^ { t } \mathcal { E } ^ { t + 1 } ) / ( \lambda + \beta _ { 1 } ^ { t } ) . } \end{array}\tag{10}
$$

4) Update $\Upsilon ^ { t + 1 }$ : The sub-problem to update Υ is

$$
\begin{array} { r l } & { \underset { \Upsilon } { \arg \operatorname* { m i n } } \ \big \langle \Lambda _ { 2 } ^ { t } , \Theta ^ { t + 1 } - \Upsilon \big \rangle + \frac { \beta _ { 1 } ^ { t } } { 2 } \big \| \Theta ^ { t + 1 } - \Upsilon \big \| ^ { 2 } } \\ & { + \left. \Lambda _ { 3 } ^ { t } , \nabla _ { 1 } \Upsilon - U ^ { t } \right. + \frac { \beta _ { 2 } ^ { t } } { 2 } \big \| \nabla _ { 1 } \Upsilon - U ^ { t } \big \| ^ { 2 } } \\ & { + \left. \Lambda _ { 4 } ^ { t } , \nabla _ { 2 } \Upsilon - V ^ { t } \right. + \frac { \beta _ { 2 } ^ { t } } { 2 } \big \| \nabla _ { 2 } \Upsilon - V ^ { t } \big \| ^ { 2 } . } \end{array}
$$

To optimize the problem, we reformulate it as the following <sub>5</sub> linear system 6 6

$$
\begin{array} { r } { ( \beta _ { 1 } ^ { t } I + \beta _ { 2 } ^ { t } \nabla _ { 1 } ^ { \top } \nabla _ { 1 } + \beta _ { 2 } ^ { t } \nabla _ { 2 } ^ { \top } \nabla _ { 2 } ) \Upsilon = \chi _ { 0 } + \nabla _ { 1 } ^ { \top } \chi _ { 1 } + \nabla _ { 2 } ^ { \top } \chi _ { 2 } , ( 1 1 ) \stackrel { \triangledown \Upsilon } { \le } } \end{array}
$$

where $\chi _ { 0 } = \beta _ { 1 } ^ { t } \theta ^ { t + 1 } + \Lambda _ { 2 } ^ { t } , \ \chi _ { 1 } = \beta _ { 2 } ^ { t } U ^ { t } - \Lambda _ { 3 } ^ { t }$ and $\chi _ { 2 } ~ = ~ 9$ $\beta _ { 2 } ^ { t } V ^ { t } - \Lambda _ { 4 } ^ { t }$ . The operator $\nabla _ { u } ^ { \top } \nabla _ { u }$ corresponds to a block-10 0 circulant matrix, which can be diagonalized using a twodimensional fast Fourier transform matrix. Applying the Fourier transform to both sides of equation (11) and employing the convolution theorem, as shown in [7], [43], we can readily derive the closed-form solution for $\mathbf { \bar { \Upsilon } } ^ { t + \bar { 1 } }$ as follows

$$
\Upsilon ^ { t + 1 } = \mathcal { F } ^ { - 1 } \Big ( \frac { \mathcal { F } ( \chi _ { 0 } ) + \sum _ { u = 1 } ^ { 2 } \mathcal { F } ( \nabla _ { u } ) ^ { \top } \odot \mathcal { F } ( \chi _ { u } ) } { \beta _ { 1 } ^ { t } { \bf 1 } + \sum _ { u = 1 } ^ { 2 } \beta _ { 2 } ^ { t } | \mathcal { F } ( \nabla _ { u } ) | ^ { 2 } } \Big ) ,\tag{12)<sub>11</sub>}
$$

where 1 represents the matrix with all elements as $^ { 1 , _ { 1 3 } }$ $\odot$ is the element-wise multiplication, $\mathcal F ( \cdot )$ is the Fourier transform, and $| \cdot | ^ { 2 }$ is the element-wise square operation.

5) Update $U ^ { i + 1 }$ : The sub-problem to update U is

$$
\begin{array} { r l } & { \quad \underset { U \in \mathbb { S } _ { 1 } } { \arg \operatorname* { m i n } } ~ \langle \boldsymbol { \Lambda } _ { 3 } ^ { t } , \nabla _ { 1 } \Upsilon ^ { t + 1 } - U \rangle + \frac { \beta _ { 2 } ^ { t } } { 2 } \| \nabla _ { 1 } \Upsilon ^ { t + 1 } - U \| ^ { 2 } } \\ & { \quad = P _ { \mathbb { S } _ { 1 } } ( \nabla _ { 1 } \Upsilon ^ { t + 1 } + \boldsymbol { \Lambda } _ { 3 } ^ { t } / \beta _ { 2 } ^ { t } ) . } \end{array}\tag{13}
$$

6) Update $V ^ { t + 1 }$ : The sub-problem to update V is

$$
\operatorname { a r g m i n } _ { V \in \mathbb { S } _ { 2 } } \ \langle \Lambda _ { 4 } ^ { t } , \nabla _ { 2 } \Upsilon ^ { t + 1 } - V \rangle + \frac { \beta _ { 2 } ^ { t } } { 2 } \| \nabla _ { 2 } \Upsilon ^ { t + 1 } - V \| ^ { 2 }\tag{14}
$$

The complete LADMM procedure is summarized in Algorithm 4.1.

The remaining implementation details are the proximal mappings used in Algorithm 4.1.

(1) $\operatorname { p r o x } _ { \eta \| \cdot \| _ { F , p } } ( \cdot ) \mathrm { : }$ Following [7, Theorem 4.1], the proximal mapping is defined as

$$
\begin{array} { r } { [ \mathrm { p r o x } _ { \eta \| \cdot \| _ { F , p } } ( \mathcal Z ) ] ( : , j , : ) = \left\{ \begin{array} { l l } { \mathrm { p r o x } _ { \eta | \cdot | ^ { p } } \left( z _ { j } \right) \frac { \mathcal Z ( : , j , : ) } { z _ { j } } , } & { \mathrm { i f } \ z _ { j } \ne 0 ; } \\ { 0 , } & { \mathrm { i f } \ z _ { j } = 0 , } \end{array} \right. } \end{array}
$$

where $z _ { j } ~ = ~ \| \mathcal { Z } ( : , j , : ) \|$ . The proximal mapping of $| \cdot | ^ { p }$ considered here is classical, as given in [44].

(2) $\operatorname { p r o x } _ { \eta \varPhi } \bigl ( \cdot , \cdot \bigr ) $ : We first note that prox $_ { \eta \Phi } ( \tilde { \mathcal { E } } , \tilde { \mathcal { O } } )$ is separable, i.e.,

$$
\begin{array} { r } { [ \mathrm { p r o x } _ { \eta \Phi } ( \tilde { \mathcal { E } } , \tilde { \Theta } ) ] _ { i j } = \mathrm { p r o x } _ { \eta \tilde { \phi } } ( \tilde { \mathcal { E } } _ { i j } ; , \tilde { \Theta } _ { i j } ) , } \end{array}
$$

where $\tilde { \phi } ( \mathcal { E } _ { i j : } , \theta _ { i j } ) = \phi ( \| \mathcal { E } _ { i j : } \| , \theta _ { i j } )$ . To further simplify the computation, we introduce the following lemma.

Lemma 4.1: For a given vector x˜ and scalar y˜, we have

$$
\begin{array} { r } { \mathrm { p r o x } _ { \eta \tilde { \phi } } ( \tilde { \pmb { x } } , \tilde { \pmb { y } } ) = \left\{ \begin{array} { l l } { ( \frac { z _ { 1 } } { \lVert \tilde { \pmb { x } } \rVert } \tilde { \pmb { x } } , z _ { 2 } ) , } & { \mathrm { i f ~ } \tilde { \pmb { x } } \neq \mathbf { 0 } ; } \\ { ( \mathbf { 0 } , \mathrm { m a x } \{ \tilde { \pmb { y } } - \eta / 2 , 0 \} ) , } & { \mathrm { i f ~ } \tilde { \pmb { x } } = \mathbf { 0 } , } \end{array} \right. } \end{array}
$$

Algorithm 4.1 The LADMM algorithm for problem (6)   
Input: $\gamma > 0 , \lambda > 0 , p , \{ \alpha _ { u } \} _ { u = 1 } ^ { 2 } , \{ \beta _ { u } ^ { 0 } , \rho _ { u } \} _ { u = 1 } ^ { 2 } ,$ , and $\mathcal { X } ^ { 0 } .$   
$\mathcal { V } ^ { 0 } , \mathcal { E } ^ { 0 } , \mathcal { O } ^ { 0 } , \mathcal { N } ^ { 0 } , \Upsilon ^ { \bar { 0 } } , \bar { U } ^ { \bar { 0 } } , \bar { V } ^ { 0 } , \{ \bar { \Lambda } _ { u } ^ { 0 } \} _ { u = 1 } ^ { \bar { 4 } } .$   
Output: $\Theta ^ { t + 1 }$   
$t \gets 0 ;$   
2 while not converged do   
Update $\mathcal { X } ^ { t + 1 }$ according to $( 7 ) .$   
Update $\mathcal { V } ^ { t + 1 }$ according to (8).   
Update $( \mathcal { E } ^ { t + 1 } , \Theta ^ { t + 1 } )$ according to (9).   
Update $\dot { \mathcal { N } } ^ { t + 1 }$ according to (10).   
Update $\Upsilon ^ { t + 1 }$ according to (12).   
Update $U ^ { t + 1 }$ according to (13).   
Update $V ^ { t + 1 }$ according to (14).   
Update multipliers $\Lambda _ { u } ^ { t + 1 }$ and penalty parameters $\beta _ { u } ^ { t + 1 }$   
according to   
$\Lambda _ { 1 } ^ { t + 1 } = \Lambda _ { 1 } ^ { t } + \beta _ { 1 } ^ { t } \big ( \mathcal { E } ^ { t + 1 } - \mathcal { N } ^ { t + 1 } \big ) ;$   
$\Lambda _ { 2 } ^ { \dot { t } + 1 } = \Lambda _ { 2 } ^ { \dot { t } } + \beta _ { 1 } ^ { \dot { t } } \big ( \theta ^ { t + 1 } - \Upsilon ^ { t + 1 } \big ) ;$   
$\Lambda _ { 3 } ^ { \bar { t } + 1 } = \Lambda _ { 3 } ^ { \bar { t } } + \beta _ { 2 } ^ { \bar { t } } \big ( \nabla _ { 1 } \Upsilon ^ { t + 1 } - U ^ { \bar { t } + 1 } \big )$ (15)   
$\Lambda _ { 4 } ^ { \check { t } + 1 } = \Lambda _ { 4 } ^ { \check { t } } + \beta _ { 2 } ^ { \check { t } } \big ( \nabla _ { 2 } \Upsilon ^ { t + 1 } - V ^ { t + 1 } \big )$   
$\beta _ { 1 } ^ { t + 1 } = \rho _ { 1 } \dot { \beta } _ { 1 } ^ { t } , \ \bar { \beta } _ { 2 } ^ { t + 1 } = \rho _ { 2 } \beta _ { 2 } ^ { t } .$   
Remove the zero lateral slices of $\mathcal { X } ^ { t + 1 }$ and $\mathcal { V } ^ { t + 1 }$   
$t \gets t + 1$   
end

where $( z _ { 1 } , z _ { 2 } ) \in \mathrm { p r o x } _ { \eta \phi } ( \| \tilde { \pmb { x } } \| , \tilde { y } )$ Proof. It is clear that prox $\dot { \mathbf { \eta } } _ { \eta \tilde { \phi } } ( \tilde { \pmb { x } } , \tilde { y } ) = ( \mathbf { 0 }$ , max $\{ \tilde { y } - \eta / 2 , 0 \} )$ when ${ \tilde { \mathbf { x } } } = \mathbf { 0 }$ . Therefore, we only need to consider $\tilde { \boldsymbol { x } } \neq \mathbf { 0 }$ in the following. From $( z _ { 1 } , z _ { 2 } ) \in \mathrm { p r o x } _ { \eta \phi } ( \| \tilde { \pmb { x } } \| , \tilde { y } )$ , one has

$$
\begin{array} { r l } & { \quad \eta \tilde { \phi } ( \frac { z _ { 1 } } { \| \tilde { x } \| } \tilde { x } , z _ { 2 } ) + \displaystyle \frac { 1 } { 2 } \| \frac { z _ { 1 } } { \| \tilde { x } \| } \tilde { x } - \tilde { x } \| ^ { 2 } + \displaystyle \frac { 1 } { 2 } ( z _ { 2 } - \tilde { y } ) ^ { 2 } } \\ & { = \eta \phi ( z _ { 1 } , z _ { 2 } ) + \displaystyle \frac { 1 } { 2 } ( z _ { 1 } - \| \tilde { x } \| ) ^ { 2 } + \displaystyle \frac { 1 } { 2 } ( z _ { 2 } - \tilde { y } ) ^ { 2 } } \\ & { \leq \eta \phi ( \| x \| , y ) + \displaystyle \frac { 1 } { 2 } ( \| x \| - \| \tilde { x } \| ) ^ { 2 } + \displaystyle \frac { 1 } { 2 } ( y - \tilde { y } ) ^ { 2 } } \\ & { \leq \eta \tilde { \phi } ( x , y ) + \displaystyle \frac { 1 } { 2 } \| x - \tilde { x } \| ^ { 2 } + \displaystyle \frac { 1 } { 2 } \| y - \tilde { y } \| ^ { 2 } , } \end{array}
$$

which completes the proof.

By this lemma, computing $\mathrm { p r o x } _ { \eta \varPhi } ( \tilde { \mathcal { E } } , \tilde { \varTheta } )$ reduces to evaluating prox $_ { \cdot \eta \phi } ( \| \tilde { \pmb { x } } \| , \tilde { y } )$ , whose closed form is given in [45, Example 2.4]:

$$
\begin{array} { r l } & { \operatorname { p r o x } _ { \eta \phi } ( \| \tilde { { \boldsymbol x } } \| , \vartheta ) = } \\ & { \left\{ \begin{array} { l l } { ( 0 , 0 ) , } & { \mathrm { i f } ~ 2 \eta \vartheta + \| \tilde { { \boldsymbol x } } \| ^ { 2 } \leq \eta ^ { 2 } ; } \\ { \big ( 0 , \tilde { y } - \frac { \eta } { 2 } \big ) , } & { \mathrm { i f } ~ \| \tilde { { \boldsymbol x } } \| = 0 ~ \mathrm { a n d } ~ 2 \tilde { y } > \eta ; } \\ { \big ( \| \tilde { { \boldsymbol x } } \| - \eta s , \tilde { y } + \eta \frac { s ^ { 2 } - 1 } { 2 } \big ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

Let $\begin{array} { r } { a = \frac { 2 } { \eta } \tilde { y } + 1 , ~ b = - \frac { 2 } { \eta } \| \tilde { \mathbf { x } } \| } \end{array}$ , and $\begin{array} { r } { c = - \frac { b ^ { 2 } } { 4 } - \frac { a ^ { 3 } } { 2 7 } } \end{array}$ . The variable s is computed by

$$
s = { \left\{ \begin{array} { l l } { { \sqrt [ { 3 } ] { - { \frac { b } { 2 } } + { \sqrt { - c } } } } + { \sqrt [ { 3 } ] { - { \frac { b } { 2 } } - { \sqrt { - c } } } } , } & { { \mathrm { i f ~ } } c < 0 ; } \\ { 2 { \sqrt [ { 3 } ] { - { \frac { b } { 2 } } } } , } & { { \mathrm { i f ~ } } c = 0 ; } \\ { 2 { \sqrt [ { 3 } ] { { \sqrt { { \frac { b ^ { 2 } } { 4 } } + c } } \cos { \big ( } { \frac { \arctan ( - 2 { \sqrt { c } } / b ) } { 3 } } { \big ) } } } , } & { { \mathrm { i f ~ } } c > 0 . } \end{array} \right. }
$$

(3) $P _ { \mathbb { S } } ( \cdot ) \colon$ : Let $\pmb { x } \in \mathbb { R } ^ { n }$ be an arbitrary vector and $\alpha >$ 0 a given radius. The projection of x onto the $\ell _ { 0 } { \mathrm { - b a l l } }$

defined as $\mathbb { S } = \{ z \in \mathbb { R } ^ { n } | \| z \| _ { 0 } \leq \alpha \}$ , is denoted $P _ { \mathbb { S } } ( { \pmb x } )$ This projection can be expressed concisely and computed eficiently as follows

$$
P _ { \mathrm { S } } ( \pmb { x } ) = \left\{ \begin{array} { l l } { \pmb { x } , } & { \mathrm { i f ~ } \| \pmb { x } \| _ { 0 } \leq \alpha ; } \\ { \mathrm { s i g n } ( \pmb { x } ) \odot \operatorname* { m a x } \big ( | \pmb { x } | - \varpi , 0 \big ) , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

where $\varpi$ is defined as the (α+1)-th largest absolute value in the descendingly sorted vector |x| components.

## B. Computation Complexity

In each iteration of Algorithm 4.1, the dominant cost comes from t-products and FFT operations. Updating X and Y requires two t-products with complexity $O ( ( d n _ { 1 } +$ $d n _ { 2 } + n _ { 1 } n _ { 2 } ) n _ { 3 } \log n _ { 3 } + d n _ { 1 } n _ { 2 } n _ { 3 } )$ , while updating N requires one t-product with complexity $\mathcal { O } ( d ( n _ { 1 } + n _ { 2 } ) n _ { 3 } \log n _ { 3 } +$ $d n _ { 1 } n _ { 2 } n _ { 3 } )$ . The update of Υ is solved by FFT and inverse FFT with complexity $\mathcal { O } ( n _ { 1 } n _ { 2 } \log ( n _ { 1 } n _ { 2 } ) )$ ). The updates of E and $\Theta$ cost $\mathcal { O } ( n _ { 1 } n _ { 2 } n _ { 3 } )$ , and the projections for $U$ and V cost $\mathcal { O } ( n _ { 1 } n _ { 2 } )$ . Therefore, the overall per-iteration complexity is $\mathcal { O } ( ( d ( n _ { 1 } + n _ { 2 } ) + n _ { 1 } n _ { 2 } ) n _ { 3 } \log n _ { 3 } + d n _ { 1 } n _ { 2 } n _ { 3 } +$ $n _ { 1 } n _ { 2 } \log ( n _ { 1 } n _ { 2 } ) )$ .

## C. Convergence Analysis

For convergence analysis, denote $\mathcal { W } = ( \mathcal { W } _ { 1 } , \mathcal { W } _ { 2 } )$ , where $\mathcal { W } _ { 1 } = ( \mathcal { X } , \mathcal { Y } , \mathcal { E } , \theta , \mathcal { N } , \Upsilon , U , V )$ collects the primal variables and $\mathcal { W } _ { 2 } = ( \Lambda _ { 1 } , \ldots , \Lambda _ { 4 } )$ collects the dual variables. The following result establishes subsequential convergence of Algorithm 4.1 for problem (6).

Theorem 4.1: Let $\{ \mathcal { W } ^ { t } \} _ { t = 1 } ^ { \infty }$ be the sequence produced by Algorithm 4.1. Assume that the sequence $\{ \mathcal { W } _ { 2 } ^ { t } \} _ { t = 1 } ^ { \infty }$ is bounded. Then every accumulation point of $\{ \mathcal { W } ^ { t } \} _ { t = 1 } ^ { \infty }$ is a Karush–Kuhn–Tucker (KKT) point of the optimization problem (6).

## V. Numerical Experiments

This section evaluates the proposed GSAA-SS detector on five real hyperspectral scenes. All experiments were conducted on a computer with an Intel Core i5-12500H CPU (2.50 GHz) and 16 GB RAM. All methods were implemented in MATLAB R2022a and applied to the raw hyperspectral data without additional preprocessing.

The proposed method is compared with ten representative hyperspectral anomaly detectors, including RX [10], RPCA [24], LRASR [29], Turbo-GoDec [23], PTA [26], TPCA [46], PCA-TLRSR [30], RGAE [16], GAED [47], and LCRS [48]. RX is a classical statistical detector. RPCA is a matrix low-rank decomposition method that separates sparse anomalies from a low-rank background and then applies RX to the anomaly component. LRASR is a matrix representation based detector using a dictionary driven background representation. Turbo-GoDec extends the classical GoDec framework by incorporating a spatial cluster sparsity prior for anomalies. PTA and TPCA are tensor based low-rank detectors, where PTA additionally introduces total variation regularization to exploit spatial smoothness. PCA-TLRSR is a tensor representation based method that employs PCA for spectral redundancy reduction and subsequent tensor representation modeling. RGAE and GAED are deep learning based detectors. LCRS is an eficient saliency based detector that combines multi-scale spectral local contrast with threedimensional spectral residual saliency.

For quantitative evaluation, let $P _ { D }$ denote the probability of detection and $P _ { F }$ denote the false-alarm rate. We use the area under the curve (AUC) of the receiver operating characteristic (ROC) curve as the primary metric, where the ROC curve plots $P _ { D }$ against $P _ { F }$ . A larger AUC indicates better detection accuracy.

## A. Dataset Description

The experiments use five real hyperspectral scenes selected from two public data sources, which are described in detail below.

(1) Airport-Beach-Urban Dataset<sup>1</sup>: The Airport-Beach-Urban (ABU) dataset contains hyperspectral images from airport, beach, and urban scenes. The spatial size of each image is either $1 0 0 \times 1 0 0 ~ \mathrm { o r } ~ 1 5 0 \times 1 5 0$ , and the number of spectral bands is approximately 100 or 200 depending on the sensor. Four scenes are selected in this paper, including two airport scenes, one beach scene, and one urban scene. Their pseudo-color images and ground-truth maps are shown in Fig. $2 ( \mathrm { a } ) \mathrm { - } ( \mathrm { d } )$

(2) San Diego Dataset<sup>2</sup>: The San Diego dataset contains an airport scene captured by the Airborne Visible/Infrared Imaging Spectrometer (AVIRIS). It was acquired over San Diego with a spatial resolution of 3.5 m/pixel and contains 189 spectral bands covering wavelengths from 370 to 2510 nm. Following common experimental settings, a 100 × 100 subimage is used. Its pseudo-color image and ground-truth map are shown in Fig. 2(e).

![](images/1b1a3570a0cb454cc0e7a9fbbacc996bd6b6eb8df9cf30ad5dbfc9ca8e9c0a1c.jpg)  
(a)  
(b)  
(c)  
(d)  
(e)  
Fig. 2: Pseudo-color images and ground-truth maps: (a) Airport1; (b) Airport2; (c) Beach; (d) Urban; (e) San Diego.

## B. Detection Performance

Fig. 3 shows the detection maps produced by all compared methods on the five test scenes. The proposed GSAA-SS detector provides clearer anomaly responses while maintaining stronger background suppression in most cases. On Airport2, RX, RPCA, LRASR, Turbo-GoDec, TPCA, RGAE, and LCRS miss part of the anomalous targets, whereas PTA, PCA-TLRSR, and GAED recover more targets but also retain visible background interference, especially in the lower-left region. GSAA-SS better separates the target region from the background. On Urban, RX, RPCA, LRASR, Turbo-GoDec, and LCRS suppress the background efectively but also weaken several anomaly pixels. PTA, TPCA, PCA-TLRSR, RGAE, and GAED preserve more anomaly responses but introduce stronger false responses in the upper-right and lower-left areas. In contrast, GSAA-SS produces compact and prominent anomaly responses with fewer background artifacts.

Fig. 4 reports the ROC curves on the five datasets. On Airport2, GSAA-SS achieves a higher detection probability than the competing methods across almost the entire false-alarm range. On Beach and San Diego, its superiority is particularly pronounced when $P _ { F } > 1 0 ^ { - 3 } ;$ this advantage is also evident on Urban when $P _ { F } > 1 0 ^ { - 2 }$ and on Airport1 when $P _ { F } > 1 0 ^ { - 1 }$ . These ROC results show that the proposed detector maintains favorable detection capability under diferent false-alarm constraints.

Table I lists the AUC values and running time (in seconds) of all methods. GSAA-SS obtains the highest AUC on all five datasets. Compared with the second-best result on each dataset, the improvements are 3.17%, 2.41%, 1.71%, 1.02%, and 0.09%, respectively. The corresponding second-best methods are LCRS, GAED, Turbo-GoDec, PCA-TLRSR, and PCA-TLRSR. These results indicate that GSAA-SS can obtain robust performance across scenes with varying background complexity and anomaly distributions. The runtime results in Table I further show the computational advantage of the proposed factorized model. Although RX is the fastest method, its detection accuracy is much lower than that of GSAA-SS. Among the remaining methods, GSAA-SS achieves the lowest runtime on average across the tested datasets, while maintaining highly competitive eficiency on each individual dataset. It is about twice as fast as PCA-TLRSR and is at least twenty-five times faster than RGAE and GAED on average. This eficiency is consistent with the complexity analysis in Section IV, because the proposed model avoids repeated large-scale SVDs and adaptively reduces the factor dimension during optimization.

To further examine separability, Fig. 5 presents normalized background–anomaly separation maps. A larger gap between the anomaly and background distributions indicates better separability. On Airport1, Airport2, and Beach, GSAA-SS produces a clearer separation gap than the competing methods. On Urban, RX, RPCA, PCA-TLRSR, and GSAA-SS show comparable separation, while the other methods exhibit stronger overlap between anomaly and background responses. On San Diego, PCA-TLRSR and GSAA-SS achieve similar separability, but PCA-TLRSR is less stable on Airport1, Airport2, and Beach. Overall, GSAA-SS provides consistently strong separability across all tested scenes.

## C. Discussions

This subsection begins with an analysis of how diferent parameters afect performance, followed by a discussion of the model’s novel contributions.

1) Parameter analysis: The proposed model contains five model parameters, namely $\gamma , \lambda , p , \alpha _ { 1 }$ , and $\alpha _ { 2 } .$ , and four algorithm parameters, namely $\beta _ { u } ^ { 0 }$ and $\rho _ { u }$ for $u \in [ 2 ]$ In all experiments, the initial penalty parameters are set as $\beta _ { 1 } ^ { 0 } ~ = ~ 5 \times 1 0 ^ { - 2 }$ and $\beta _ { 2 } ^ { 0 } ~ = ~ 1 0 ^ { - 2 }$ . The parameter $\rho _ { 1 }$ is selected from {1.3, 1.4, 1.5}, while $\rho _ { 2 }$ is fixed at 1.2. Following the convergence requirement, λ is updated as $\lambda = \operatorname* { m i n } \{ 1 0 \beta _ { 1 } , 1 0 ^ { 1 0 } \}$

We first evaluate the sensitivity to γ and $p .$ The parameter $\gamma$ is selected from $\{ 1 0 ^ { - 2 } , 5 \times 1 0 ^ { - 2 } , 1 0 ^ { - \bar { 1 } } , 5 \times 1 0 ^ { - 1 } , 1 \}$ and p varies from 0.2 to 1 with an interval of 0.1. Fig. 6 shows the resulting AUC surfaces. The proposed method is stable when $\gamma \in \{ 5 \times 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ and $p \in [ 0 . 4 , 0 . 7 ]$ Therefore, we set $\gamma = 5 \times 1 0 ^ { - 2 }$ and $p = 0 . 6$ in the following experiments.

We then examine the influence of $\alpha _ { 1 }$ and $\alpha _ { 2 }$ , which control the number of vertical and horizontal changes in the learned grouping map. Both parameters are selected from {1, 5, 10, 20, 30, 40, 50}. Fig. 7 reports the corresponding AUC surfaces. The performance is insensitive to moderate changes in $\alpha _ { 1 }$ and $\alpha _ { 2 } \colon$ for all datasets except Beach, the variation of AUC is below 0.5%, and the largest variation on Beach is about 2%. We also observe that the preferred value of $\alpha _ { 1 }$ tends to increase when more anomaly blocks are present. Therefore, we fix $\alpha _ { 2 } = 1 0$ and set $\alpha _ { 1 } = 1$ for scenes with few anomaly blocks and $\alpha _ { 1 } = 5 0$ for scenes with more anomaly blocks.

2) Adaptive dimensional reduction during iterations: In our model, the large tensor $\mathcal { Z } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is factorized via the t-product into two smaller tensors, $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } }$ ×d×n<sub>3</sub> and $\mathcal { V } \in \mathbb { R } ^ { n _ { 2 } \times d \times n _ { 3 } }$ , where the shared dimension d controls the complexity of the decomposition. In Algorithm 4.1, $d$ is adaptively reduced at Step 11. Fig. 8 illustrates the evolution of the auxiliary dimension d during each iteration of Algorithm 4.1 for both the spatial domain and the spectral domain. Initially d equals the full tensor size but collapses sharply within a few iterations to a small stable value. After this rapid drop, d remains fixed for the rest of the optimization. This early convergence of d demonstrates that the algorithm immediately eliminates redundant dimensions and thereafter incurs only the lowrank cost determined by the converged value of $d ,$ yielding substantial savings in computation time.

3) Efects of automatic anomaly grouping and spectral– spatial fusion: Table II verifies the efects of AAG and spectral–spatial fusion by reporting the AUC values of diferent detection maps. Algorithm 4.1 produces $\mathcal { E } _ { s p a }$ and $\Theta _ { s p a }$ in the spatial domain, and $\mathcal { E } _ { s p e }$ and $\theta _ { s p e }$ in the spectral domain. The anomaly magnitude maps are computed as $\begin{array} { r } { E _ { s p a } ( i , j ) \ = \ \sqrt { \sum _ { k } | \mathcal { E } _ { s p a } ( i , j , k ) | ^ { 2 } } } \end{array}$ and $\begin{array} { r c l } { E _ { s p e } ( i , j ) } & { = } & { \sqrt { \sum _ { k } \vert \bar { \mathcal { E } } _ { s p e } ( i , j , k ) \vert ^ { 2 } } } \end{array}$ . The fused maps are obtained by element-wise multiplication, i.e., $\begin{array} { r l } { E _ { f u s } } & { { } = } \end{array}$ $E _ { s p a } \odot E _ { s p e }$ and $\theta _ { f u s } = \theta _ { s p a } \odot \theta _ { s p e }$

![](images/f1a3785266ebf6c9a18399085fb0833da4fbf3d9f3bbe70ed16ffe9f9cfd1e97.jpg)  
(a)  
(b)

![](images/6022f6304941861003a22a7081a42cb842a9e5440660fd3b212f4c179dffd0c1.jpg)  
(c)  
(d)

![](images/64773a138ed9023ae80d7a43caf8efe728bfdfcbdc7d55b9e01731e7ce80b03f.jpg)  
(e)

![](images/fffc4d30dde5b5fa6f7da20b4ae642557e69b3781ea239f82a570fe2f9df2672.jpg)  
(f)

![](images/2fd87568e65cd0e261694e8b23ff4f474ac0c7faac23bb8bfc19e2888a5abf53.jpg)  
(g)

(h)  
![](images/4ba971ab62e4fb8008e1b469c446cf93e47b1403008c8ac2a8fce3544bae3c11.jpg)  
(i)  
(j)  
(k)  
Fig. 3: Detection results of diferent methods on five datasets: (a) RX; (b) RPCA; (c) LRASR; (d) Turbo-GoDec; (e) PTA; (f) TPCA; (g) PCA-TLRSR; (h) RGAE; (i) GAED; (j) LCRS; (k) GSAA-SS.

![](images/a5a5dd625b36f49045078f33ede1352df8ee0c5ee31cd2bfa389779dc84cd236.jpg)  
(a) Airport1

![](images/20a17717504ce8f42c972862a5e08cc1581b7e648c4e9f400a70b177a1135a34.jpg)  
(b) Airport2

![](images/deb3867ee9dbb4f42ecdba30a3aad5b8790eb3ed2d2b542b1dc056c76f94a5a0.jpg)  
(c) Beach

![](images/4c27078254e0d79c8a7f7c85e52d4758d9a98f896bfd349e0fce893c626dbcbf.jpg)  
(d) Urban

![](images/74026fbf07c13343131108f4426af757d99dc9ed31f1f53eebc4d2a940d86130.jpg)  
(e) San Diego  
Fig. 4: ROC curves obtained by diferent methods.

TABLE I: Comparison of AUC values and running time(s) of diferent methods.
<table><tr><td rowspan="2">HSI</td><td rowspan="2">Index</td><td colspan="10">Method</td></tr><tr><td>RX</td><td>RPCA</td><td>LRASR</td><td>Turbo-GoDec</td><td>PTA</td><td>TPCA</td><td>PCA-TLRSR</td><td>RGAE</td><td>GAED</td><td>LCRS</td><td>GSAA-SS</td></tr><tr><td rowspan="3">Airport1</td><td>AUC</td><td>0.8221</td><td>0.8088</td><td>0.7942</td><td>0.8672</td><td>0.7331</td><td>0.8023</td><td>0.9057</td><td>0.8684</td><td>0.8133</td><td>0.9237</td><td>0.9530</td></tr><tr><td>Time (s)</td><td>0.06</td><td>4.23</td><td>18.45</td><td>15.36</td><td>16.04</td><td>19.51</td><td>2.56</td><td>52.79</td><td>42.80</td><td>1.91</td><td>1.94</td></tr><tr><td>AUC</td><td>0.8404</td><td>0.8428</td><td>0.8689</td><td>0.9299</td><td>0.9096</td><td>0.8891</td><td>0.9458</td><td>0.9645</td><td>0.9673</td><td>0.8998</td><td>0.9906</td></tr><tr><td rowspan="2">Airport2</td><td>Time (s)</td><td>0.05</td><td>4.50</td><td>19.75</td><td>14.98</td><td>16.32</td><td>19.58</td><td>2.56</td><td>52.02</td><td>41.58</td><td>1.94</td><td>1.70</td></tr><tr><td>AUC</td><td>0.9538</td><td>0.9603</td><td>0.9504</td><td>0.9776</td><td>0.9061</td><td>0.9583</td><td>0.9599</td><td>0.9062</td><td>0.9368</td><td>0.9045</td><td>0.9943</td></tr><tr><td rowspan="2">Beach</td><td>Time (s)</td><td>0.04</td><td>2.05</td><td>43.72</td><td>26.43</td><td>17.53</td><td>22.33</td><td>6.66</td><td>219.39</td><td>73.75</td><td>3.11</td><td>2.50</td></tr><tr><td>AUC</td><td>0.9692</td><td>0.9658</td><td>0.9293</td><td>0.9536</td><td>0.8258</td><td>0.9370</td><td>0.9711</td><td>0.9510</td><td>0.8986</td><td>0.9429</td><td>0.9810</td></tr><tr><td rowspan="2">Urban</td><td>Time (s)</td><td>0.06</td><td>3.96</td><td>19.91</td><td>15.51</td><td>15.64</td><td>20.08</td><td>2.56</td><td>57.09</td><td>42.38</td><td>2.02</td><td>1.67</td></tr><tr><td></td><td>0.9106</td><td>0.9264</td><td>0.7847</td><td>0.9768</td><td>0.9895</td><td>0.9075</td><td>0.9970</td><td>0.9930</td><td>0.9918</td><td>0.9500</td><td></td></tr><tr><td>San Diego</td><td>AUC Time (s)</td><td>0.38</td><td>5.53</td><td>18.22</td><td>12.49</td><td>14.68</td><td>20.95</td><td>2.92</td><td>100.88</td><td>40.98</td><td>1.66</td><td>0.9979 1.41</td></tr></table>

![](images/0b8cd37de56e01b9a51cecd6364e8d3c18248c56ea543c086807b9b91ccc8907.jpg)  
(a) Airport1

![](images/4e280c47ba6b8f48d66b25e1e8205f30e620bfdfe8c91845a28b714c22715c6d.jpg)  
(b) Airport2

![](images/53938d977391cd0db49aa2881437c84cab7085da816e2f024d32a8a665545eba.jpg)  
(c) Beach

![](images/763f60b48f8e7b11c2812c264bc3b8809fea008baf6bd098e81fc5487c80f780.jpg)  
(d) Urban

![](images/d9920beb0cbe2a76edaad145527ea25ae722f05d394a9cb11afed49e572f4ee6.jpg)  
(e) San Diego  
Fig. 5: Separability maps of diferent methods.

![](images/847626689421e6136dbb9b8fb09bf31a0e38d2514ed9a12f538ba244d065df94.jpg)  
(a) Airport1

![](images/fe1c5caae8c7f53b3ca869670370231d6d48b872976d5b66b73c94b13fc6fb3f.jpg)  
(b) Airport2

![](images/3d460a899226186cabdd682b9fa2d97adb1b9a941085cf622690c572f195e608.jpg)  
(c) Beach

![](images/345303857c38f5579b70353076fb5238fb2c810613cac988433b5404e7df8617.jpg)  
(d) Urban

![](images/2f71bc84fcc57097b7b31a4b7b4c12f92bcf70a1900c84afc3cc0ae2f05cc48c.jpg)  
(e) San Diego

Fig. 6: Surfaces of AUC values of the result by our method with diferent γ and p.  
![](images/418695c4779a1ce87c6122ab1a95b5e8851cf8ed04686d64062a0ae994f74cde.jpg)  
(a) Airport1

![](images/199713b8828bd8e46d8c4bd9ce3152208cc3a6eca7e774b40e1afa13981612da.jpg)  
(b) Airport2

![](images/301f49bb74027f4f2d6965547a76a8c3f5c19ac64d108330a7ce86cf5346fd40.jpg)  
(c) Beach

![](images/90fdb2d00b0837749b7d5c0a824df709d9eef676b863c96444c3ef9f3c2a880a.jpg)  
(d) Urban

![](images/3c70cff2dd57b924846081dd6c63bae04d8249c4542024602cd839058f69a795.jpg)  
(e) San Diego

Fig. 7: Surfaces of AUC values of the result by our method with diferent $\alpha _ { 1 }$ and $\alpha _ { 2 }$  
![](images/31296bcfa20e9c1dcd6b7ce31f68c71e16264579ba0da86b4bdbc6c19af9ea3e.jpg)  
(a) Spatial domain

![](images/0b2da281e71d0cf56290be207d6fa3daa0cfdf01def6dd7d85f17a99efa2fa5a.jpg)  
(b) Spectral domain  
Fig. 8: Variation of d values across iterations for Algorithm 4.1, shown separately for the spatial and spectral domains.

In both spatial and spectral domains, Θ consistently outperforms E on all datasets. This result indicates that the learned grouping map provides a more spatially coherent anomaly response than the anomaly magnitude map alone. Combined with the parameter analysis in Fig. 7, the results also show that finite $\alpha _ { 1 }$ and $\alpha _ { 2 }$ improve anomaly modeling compared with the limiting case $\alpha _ { 1 } , \alpha _ { 2 }  \infty .$ where Theorem 3.2 reduces the AAG penalty to the conventional $\| \cdot \| _ { 2 , 1 }$ form.

The fusion results further demonstrate the benefit of combining spectral and spatial information. Both $E _ { f u s }$ and $\Theta _ { f u s }$ generally outperform their single-domain counterparts, with only minor exceptions. In particular, $\theta _ { f u s }$ achieves the highest AUC on four of the five datasets, and on Airport1, it is only 0.0029 lower than the best singledomain result. These observations confirm that AAG and spectral–spatial fusion are both important to the final performance of GSAA-SS.

TABLE II: Comparison of AUC values of diferent anomaly detection maps.
<table><tr><td rowspan="2">HSI</td><td colspan="2">Spatial Domain</td><td colspan="2">Spectral Domain</td><td colspan="2">Fusion Method</td></tr><tr><td> $E _ { s p a }$ </td><td> $\Theta _ { s p a }$ </td><td> $E _ { s p e }$ </td><td> $\Theta _ { s p e }$ </td><td> $E _ { f u s }$ </td><td> $\Theta _ { f u s }$ </td></tr><tr><td>Airport1</td><td>0.9178</td><td>0.9559</td><td>0.8288</td><td>0.8541</td><td>0.9303</td><td>0.9530</td></tr><tr><td>Airport2</td><td>0.9198</td><td>0.9872</td><td>0.8896</td><td>0.9678</td><td>0.9476</td><td>0.9906</td></tr><tr><td>Beach</td><td>0.9268</td><td>0.9857</td><td>0.9720</td><td>0.9825</td><td>0.9535</td><td>0.9943</td></tr><tr><td>Urban</td><td>0.9299</td><td>0.9731</td><td>0.9662</td><td>0.9674</td><td>0.9545</td><td>0.9810</td></tr><tr><td>San Diego</td><td>0.9824</td><td>0.9978</td><td>0.9688</td><td>0.9752</td><td>0.9878</td><td>0.9979</td></tr></table>

## VI. Conclusions

This paper presented GSAA, an HAD model based on group sparse low-rank tensor factorization with automatic anomaly grouping. In this framework, the background is modeled by imposing group sparsity on the tensor factors, which provides an eficient alternative to direct tensor rank regularization and avoids repeated large-scale SVD computations. For anomaly modeling, the AAG penalty introduces a latent grouping map, enabling spatially coherent anomaly structures to be learned adaptively from the data rather than specified by a predefined pixel-wise partition. To further leverage complementary spectral and spatial information, GSAA was extended to a spectral–spatial framework, leading to GSAA-SS. An eficient LADMM algorithm was developed to solve the resulting optimization problem, and its convergence was theoretically analyzed. Experimental results on five real hyperspectral scenes demonstrate that GSAA-SS achieves accurate anomaly detection and competitive computational eficiency relative to representative existing methods.

## References

[1] T. Lillesand, R. W. Kiefer, and J. Chipman, Remote sensing and image interpretation. John Wiley & Sons, 2015.

[2] Z. Tu, J. Lu, H. Zhu, H. Pan, W. Hu, Q. Jiang, and Z. Lu, “A new nonconvex low-rank tensor approximation method with applications to hyperspectral images denoising,” Inverse Probl., vol. 39, no. 6, p. 065003, Apr. 2023.

[3] Y.-B. Zheng, J.-L. Wang, and X.-L. Zhao, Hyperspectral Image Denoising Based on Tensor Models. Wiley-IEEE Press, 2026, ch. 3, pp. 51–72.

[4] K. Yang, M. Bai, T. Lu, and L. Chen, “A coarse-to-fine hybrid registration and fusion framework for hyperspectral superresolution via batch image alignment,” SIAM J. Imaging Sci., vol. 19, no. 2, pp. 1207–1243, May 2026.

[5] L. Shan, Z. Yang, L. T. Yang, C. Li, H. Zhao, and X. Nie, “Bayesian fully-connected tensor network for hyperspectralmultispectral image fusion,” IEEE Trans. Image Process., vol. 35, pp. 5910–5925, 2026.

[6] W. Qin, H. Wang, H. Shu, F. Zhang, J. Wang, X. Cao, X.-L. Zhao, and G. Vivone, “Hyperspectral anomaly detection fused unified nonconvex tensor ring factors regularization,” IEEE Trans. Geosci. Remote Sens., vol. 63, 2025.

[7] Q. Yu and M. Bai, “Generalized nonconvex hyperspectral anomaly detection via background representation learning with dictionary constraint,” SIAM J. Imaging Sci., vol. 17, no. 2, pp. 917–950, Apr. 2024.

[8] D. Hong, L. Gao, N. Yokoya, J. Yao, J. Chanussot, Q. Du, and B. Zhang, “More diverse means better: Multimodal deep learning meets remote-sensing imagery classification,” IEEE Trans. Geosci. Remote Sens., vol. 59, no. 5, pp. 4340–4354, May 2021.

[9] H. Su, Z. Wu, H. Zhang, and Q. Du, “Hyperspectral anomaly detection: A survey,” IEEE Geosci. Remote Sens. Mag., vol. 10, no. 1, pp. 64–90, Mar. 2022.

[10] I. Reed and X. Yu, “Adaptive multiple-band CFAR detection of an optical pattern with unknown spectral distribution,” IEEE Trans. Acoust. Speech Signal Process., vol. 38, no. 10, pp. 1760– 1770, Oct. 1990.

[11] Q. Guo, B. Zhang, Q. Ran, L. Gao, J. Li, and A. Plaza, “Weighted-RXD and linear filter-based RXD: Improving background statistics estimation for anomaly detection in hyperspectral imagery,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 7, no. 6, pp. 2351–2366, Jun. 2014.

[12] S. Matteoli, T. Veracini, M. Diani, and G. Corsini, “Models and methods for automated background density estimation in hyperspectral anomaly detection,” IEEE Trans. Geosci. Remote Sens., vol. 51, no. 5, pp. 2837–2852, May 2013.

[13] F. Han, H. Sun, L. Gao, H. Yu, L. Qian, and X. Sun, “Spectralspatial enhanced local contrast strategy for hyperspectral small air target detection,” IEEE Trans. Image Process., vol. 35, pp. 3609–3622, 2026.

[14] N. Huyan, X. Zhang, H. Zhou, and L. Jiao, “Hyperspectral anomaly detection via background and potential anomaly dictionaries construction,” IEEE Trans. Geosci. Remote Sens., vol. 57, no. 4, pp. 2263–2276, Apr. 2019.

[15] E. Bati, A. Çalışkan, A. Koz, and A. A. Alatan, “Hyperspectral anomaly detection method based on auto-encoder,” in Proc. SPIE, vol. 9643, Oct. 2015, pp. 220–226.

[16] G. Fan, Y. Ma, X. Mei, F. Fan, J. Huang, and J. Ma, “Hyperspectral anomaly detection with robust graph autoencoders,” IEEE Trans. Geosci. Remote Sens., vol. 60, 2022.

[17] B. Tu, T. Zhou, B. Liu, Y. He, J. Li, and A. Plaza, “Multiscale autoencoder suppression strategy for hyperspectral image anomaly detection,” IEEE Trans. Image Process., vol. 34, pp. 5115–5130, 2025.

[18] W. Li, G. Wu, and Q. Du, “Transferred deep learning for anomaly detection in hyperspectral imagery,” IEEE Geosci. Remote Sens. Lett., vol. 14, no. 5, pp. 597–601, May 2017.

[19] B. Tu, B. He, Y. He, T. Zhou, B. Liu, J. Li, and A. Plaza, “Self-supervised masked graph autoencoder for hyperspectral anomaly detection,” IEEE Trans. Image Process., vol. 34, pp. 6714–6729, 2025.

[20] L. Gao, D. Wang, L. Zhuang, X. Sun, M. Huang, and A. Plaza, “BS<sup>3</sup>LNet: A new blind-spot self-supervised learning network for hyperspectral anomaly detection,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023.

[21] D. Wang, L. Zhuang, L. Gao, X. Sun, M. Huang, and A. J. Plaza, “PDBSNet: Pixel-shufle downsampling blind-spot reconstruction network for hyperspectral anomaly detection,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023.

[22] D. Wang, L. Ren, X. Sun, L. Gao, and J. Chanussot, “Nonlocal and local feature-coupled self-supervised network for hyperspectral anomaly detection,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 18, pp. 6981–6993, 2025.

[23] J. Sheng, X. Li, and S. Chen, “Turbo-GoDec: Exploiting the cluster sparsity prior for hyperspectral anomaly detection,” IEEE Transactions on Multimedia, vol. 28, pp. 6152–6165, 2026.

[24] W. Sun, C. Liu, J. Li, Y. M. Lai, and W. Li, “Low-rank and sparse matrix decomposition-based anomaly detection for hyperspectral imagery,” J. Appl. Remote Sens., vol. 8, no. 1, p. 083641, May 2014.

[25] Y. Zhang, B. Du, L. Zhang, and S. Wang, “A low-rank and sparse matrix decomposition-based Mahalanobis distance method for hyperspectral anomaly detection,” IEEE Trans. Geosci. Remote Sens., vol. 54, no. 3, pp. 1376–1389, Mar. 2016.

[26] L. Li, W. Li, Y. Qu, C. Zhao, R. Tao, and Q. Du, “Prior-based tensor approximation for anomaly detection in hyperspectral imagery,” IEEE Trans. Neural Netw. Learn. Syst., vol. 33, no. 3, pp. 1037–1050, Mar. 2022.

[27] Y. Wang, W. Li, Y. Gui, H. Xie, and L. Zhang, “A generalized non-convex surrogated framework for anomaly detection on blurred hyperspectral images,” IEEE Trans. Image Process., vol. 34, pp. 3108–3122, 2025.

[28] T. Cheng and B. Wang, “Graph and total variation regularized low-rank representation for hyperspectral anomaly detection,” IEEE Trans. Geosci. Remote Sens., vol. 58, no. 1, pp. 391–406, Jan. 2020.

[29] Y. Xu, Z. Wu, J. Li, A. Plaza, and Z. Wei, “Anomaly detection in hyperspectral images based on low-rank and sparse representation,” IEEE Trans. Geosci. Remote Sens., vol. 54, no. 4, pp. 1990–2000, Apr. 2016.

[30] M. Wang, Q. Wang, D. Hong, S. K. Roy, and J. Chanussot, “Learning tensor low-rank representation for hyperspectral anomaly detection,” IEEE Trans. Cybern., vol. 53, no. 1, pp. 679–691, Jan. 2023.

[31] Q. Yu, Y.-H. Dai, and M. Bai, “Spectral-spatial extraction through layered tensor decomposition for hyperspectral anomaly detection,” SIAM J. Imaging Sci., vol. 19, no. 2, pp. 807–838, Apr. 2026.

[32] X. He, J. Wu, Q. Ling, Z. Li, Z. Lin, and S. Zhou, “Anomaly detection for hyperspectral imagery via tensor low-rank approximation with multiple subspace learning,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023.

[33] H. Qin, Q. Shen, H. Zeng, Y. Chen, and G. Lu, “Generalized nonconvex low-rank tensor representation for hyperspectral anomaly detection,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023.

[34] Q. Feng, W. Kong, X.-L. Zhao, and J. Wang, “Kernel Bayesian robust tensor factorization for hyperspectral anomaly detection,” IEEE Trans. Geosci. Remote Sens., vol. 64, 2026.

[35] M. E. Kilmer and C. D. Martin, “Factorization strategies for third-order tensors,” Linear Algebra Appl., vol. 435, no. 3, pp. 641–658, Aug. 2011.

[36] M. E. Kilmer, K. Braman, N. Hao, and R. C. Hoover, “Thirdorder tensors as operators on matrices: A theoretical and

computational framework with applications in imaging,” SIAM J. Matrix Anal. Appl., vol. 34, no. 1, pp. 148–172, Jan. 2013.

[37] H. Kong, X. Xie, and Z. Lin, “t-schatten-p norm for low-rank tensor recovery,” IEEE J. Sel. Topics Signal Process., vol. 12, no. 6, pp. 1405–1419, Dec. 2018.

[38] Y. Wang, Z. Liu, Y. Zhao, X. Xu, and H. Mou, “Toward robust hyperspectral anomaly detection: A quantile-regression-guided joint low-rank and sparse representation framework,” IEEE Trans. Geosci. Remote Sens., vol. 64, 2026.

[39] D. Li, Y. Lv, F. Kong, and Q. Wang, “Hyperspectral compressed reconstruction for anomaly detection based on structured spectral dual-tensor coupled decomposition,” IEEE Trans. Instrum. Meas., vol. 75, 2026.

[40] H. Kuroda and D. Kitahara, “Block-sparse recovery with optimal block partition,” IEEE Trans. Signal Process., vol. 70, pp. 1506–1520, 2022.

[41] C. Rodarmel and J. Shan, “Principal component analysis for hyperspectral image classification,” Surv. Land Inf. Sci., vol. 62, no. 2, pp. 115–122, 2002.

[42] I. T. Jollife and J. Cadima, “Principal component analysis: A review and recent developments,” Philos. Trans. Royal Soc. A, vol. 374, no. 2065, p. 20150202, Apr. 2016.

[43] D. Krishnan and R. Fergus, “Fast image deconvolution using hyper-laplacian priors,” in Proc. 23rd Int. Conf. Neural Inf. Process. Syst. Red Hook, NY, USA: Curran Associates Inc., Dec. 2009, pp. 1033–1041.

[44] G. Marjanovic and V. Solo, “On l<sub>q</sub> optimization and matrix completion,” IEEE Trans. Signal Process., vol. 60, no. 11, pp. 5714–5724, Nov. 2012.

[45] P. L. Combettes and C. L. Müller, “Perspective maximum likelihood-type estimation via proximal decomposition,” Electron. J. Stat., vol. 14, no. 1, Jan. 2020.

[46] Z. Chen, B. Yang, and B. Wang, “A preprocessing method for hyperspectral target detection based on tensor principal component analysis,” Remote Sens., vol. 10, no. 7, p. 1033, Jun. 2018.

[47] P. Xiang, S. Ali, S. K. Jung, and H. Zhou, “Hyperspectral anomaly detection with guided autoencoder,” IEEE Trans. Geosci. Remote Sens., vol. 60, 2022.

[48] L. Hu, J. Yue, D. Zhao, and D. Yang, “Multi-scale spectral local contrast meets 3–D residual saliency for hyperspectral anomaly detection,” IEEE Geosci. Remote Sens. Lett., vol. 23, 2026.

[49] R. A. Horn and C. R. Johnson, Topics in Matrix Analysis. Cambridge University Press, 1991.

[50] P. Zhou, C. Lu, Z. Lin, and C. Zhang, “Tensor factorization for low-rank tensor completion,” IEEE Trans. Image Process., vol. 27, no. 3, pp. 1152–1163, Mar. 2018.

[51] P. L. Combettes, “Perspective functions: Properties, constructions, and examples,” Set-Valued Var. Anal., vol. 26, no. 2, pp. 247–264, Apr. 2018.