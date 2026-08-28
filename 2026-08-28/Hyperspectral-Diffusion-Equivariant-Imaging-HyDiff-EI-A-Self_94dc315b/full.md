# Hyperspectral Diffusion Equivariant Imaging (HyDiff-EI): A Self-supervised Framework for Hyperspectral Image Inpainting

Shuo Li, Member, IEEE, Mike Davies, Fellow, IEEE, and Mehrdad Yaghoobi, Member, IEEE

Abstract—A novel Hyperspectral diffusion Equivariant Imaging (HyDiff-EI) framework for solving the hyperspectral image (HSI) inpainting problem has been presented here. Unlike conventional diffusion-based methods that rely on large-scale pretraining, HyDiff-EI is a test-time optimization framework that learns directly from a single corrupted HSI acquisition. This makes it flexible for different sensor configurations and particularly well-suited for practical remote sensing scenarios where large annotated hyperspectral datasets are limited. To address the ill-posed nature of unsupervised inpainting, we embed equivariant consistency constraints within the diffusion process. By leveraging the inherent geometric symmetries and intrinsic characteristics of HSIs, HyDiff-EI bridges the gap between generative diffusion modeling and self-consistent physical priors. We empirically show that coupling diffusion modeling with equivariant priors substantially enhances noise robustness and generalizability. Extensive experiments on real-world datasets including Chikusei, Botswana, and EMIT demonstrate that HyDiff-EI offers remarkable inpainting quality over existing selfsupervised and diffusion-based algorithms in both noiseless and noisy cases.

Index Terms—Hyperspectral image inpainting, diffusion models, equivariant imaging, self-supervised learning.

## I. INTRODUCTION

Hyperspectral imaging captures rich spatial-spectral information across a wide range of wavelengths. However, due to various factors such as sensor failures or data transmission errors, certain areas of the HSIs acquired in practice are incomplete or corrupted. HSI inpainting refers to the process of filling in missing pixels/bands in incomplete observations, which plays a crucial role in ensuring data completeness, facilitating accurate analysis and enhancing visualization. As a solution, self-supervised HSI inpainting methods [1]–[4] offer several advantages over heavily trained methods [5], [6], including: (1) Flexibility: it becomes less required to train or re-train different models on different datasets; (2) Generalization ability: it generalizes well to unseen data. These advantages make it especially suitable for processing HSIs as gathering HS data and training with large-scale models could be expensive.

## A. Related Works

The diffusion model belongs to a broader class of deep generative models which has been gaining increasing popularity in the field of deep learning and computer vision. Benefiting from its interpretable structures by design, diffusion models are able to capture complex data distributions and generate high-quality and diverse samples [7]–[9]. The strong learning capability of the diffusion model makes it a potential candidate for solving the image inpainting task. For example, in [10], authors incorporate the reference image into the DDPM reverse process to guide the image generation, making ”conditional” sampling possible. As an extension, the authors in [11] considered the reference image to be incomplete, i.e., there are missing regions to be filled in and proposed an inpainting algorithm aided by some time-travel tricks to improve the performance. In [12], the reverse diffusion process is reformulated to address a broader class of inverse problems, marking one of the first mathematically grounded uses of diffusion models as inverse solvers. In [13], the authors borrowed the concept from linear algebra and suggested a Range-Null Space Decomposition (RND) at each diffusion step, which further concluded [12] as a special case. Later works [14]–[17] have further demonstrated the effectiveness of using diffusion models for RGB image inpainting. For a comprehensive overview of these advancements, readers are referred to the review in [18]. More recently, diffusion models have also been adopted within the remote sensing community and successfully applied to various HSI inverse tasks, including denoising, restoration, and superresolution [19]–[22], as extensively reviewed in [23]. Among these works, [22] proposed to estimate the reduced HS image with the pre-trained diffusion models and bring it back to the full spectral range for reconstruction, achieving promising performance across multiple HSI restoration tasks.

However, the success of these aforementioned methods relies heavily on the expressive power of the pre-trained diffusion models. Applying these models to HSI inverse problems typically requires training on large-scale hyperspectral datasets [7], which can be computationally demanding and impractical. In this work, we aim to leverage the potential of diffusion models on the HSI inpainting task and propose a new framework that does not impose any burden on network training. To this end, we develop a self-supervised diffusion-based HSI inpainting algorithm, HyDiff-EI, based on the recently popular equivariant imaging (EI) priors [24]–[27]. Unlike existing DHP-based methods [1]–[4], [28], HyDiff-EI leverages known physical or geometric symmetries as priors rather than relying solely on network architecture. Moreover, it operates as a self-supervised diffusion framework and does not require pretrained diffusion models [22], whether trained on extensive remote sensing or RGB datasets. This endows the method with extra flexibility: it is sensor-agnostic that can naturally accommodate HSIs with different numbers of spectral bands.

While pre-trained models are fundamentally constrained by their specific training distributions and noise levels, HyDiff-EI seamlessly adapts to unseen sensor configurations, making it highly suitable for real-world deployments.

## B. Contribution

This paper presents a powerful HS inpainting algorithm that dynamically combines self-supervised learning with recent diffusion models. Our main contributions are:

• To the best of our knowledge, this is the first work to incorporate Equivariant Imaging (EI) [24] within the discrete-time diffusion framework for hyperspectral image inpainting. Unlike methods relying on unstructured unsupervised priors (such as those based on DHP), our approach learns directly from geometric invariance structures.<sup>1</sup>

• Extensive ablation studies demonstrate that integrating diffusion with EI consistently improves inpainting performance over conventional EI, achieving substantial gains in the noiseless setting and maintaining robust performance in the noisy settings.

• The proposed method achieves state-of-the-art performance on real-world HSI inpainting tasks, demonstrating significant improvements over existing deep generative prior–based approaches [3], [22], [28].

The rest of this paper is organized as follows: Section II presents the preliminaries of EI and DDPM, and introduces our proposed HyDiff-EI algorithm. Section III discusses the implementation details, followed by the experiments on three different HSI datasets. Section IV discusses the ablation studies. Finally, Section V concludes this paper.

## II. BACKGROUND

## A. Transformation Invariance in HSI Inpainting

The HSI inpainting task can be formulated as the recovery of the clean image x from the corrupted measurement y, in the presence of additive noise n and a binary masking matrix M:

$$
\pmb { y } = \mathrm { M } \pmb { x } + \pmb { n }\tag{1}
$$

Due to the ill-posed nature of the problem, it is essential to incorporate either hand-crafted or learned priors of the clean HS image x to regularize the reconstruction process. The equivariant imaging (EI) method exploits the invariance of typical image distributions to transformations such as rotations and translations, making it possible to learn directly from the measurement data itself. In the context of HSI inpainting, transformations applied to the input corrupted HS image should be reflected in a predictable way in the reconstructed HS image. e.g., A shift or rotation in the observed image should result in the same shift or rotation in the recovered clean image. Preserving such equivariant properties is crucial for HSI inpainting algorithms to accurately model the underlying acquisition and degradation process.

Following the definitions in [24], we denote the group transformation action as $\mathcal { G } = \{ \pmb { g } _ { 1 } , \pmb { g } _ { 2 } , \ldots , \pmb { g } _ { N _ { q } } \}$ , and a set of signals as $\mathcal { X } \subseteq \mathbb { R } ^ { q }$ . Then, for unitary group actions $T _ { g } ,$ and for all the signals $\mathbf { \boldsymbol { x } } \in \mathcal { X } .$ , the group invariance property assumes:

$$
T _ { g } { \pmb x } \in { \mathcal { X } }\tag{2}
$$

it directly follows that:

$$
\pmb { y } = \mathrm { M } ( \pmb { x } ) = \mathrm { M } ( T _ { g } T _ { g } ^ { - 1 } \pmb { x } ) = \mathrm { M } _ { g } T _ { g } ^ { - 1 } ( \pmb { x } )\tag{3}
$$

One can see that group transformation is capable of exploiting new sets of virtual measurement operators $\mathrm { M } _ { g }$ that possibly span different null-spaces, thus providing extra information for recovering the clean image x. Combining (2) with a noiseless measurement matrix M and the reconstruction model $f _ { \theta } { } _ { ; }$ we obtain the following transformation-equivariant constraint:

$$
f _ { \theta } ( \mathrm { M } T _ { g } \pmb { x } ) = T _ { g } f _ { \theta } ( \mathrm { M } \pmb { x } )\tag{4}
$$

that is, the composition of $f _ { \theta } \circ \mathrm { M }$ should be equivariant to applied transformations. In practice, however, the ground-truth clean image x is unavailable. Therefore, the constraint in (4) is implemented empirically as:

$$
f _ { \theta } ( \mathrm { M } ( T _ { g } f _ { \theta } ( \pmb { y } ) ) ) = T _ { g } f _ { \theta } ( \pmb { y } )\tag{5}
$$

where the right-hand-side represents the transformed network output $\tilde { \pmb { x } } ^ { ' } = T _ { g } f _ { \theta } ( \pmb { y } )$ . On the left-hand-side, we re-inject the masked version of this transformed prediction Mx˜ back into the same EI net $f _ { \theta } ( \cdot )$ to produce a secondary output $\tilde { \boldsymbol { x } } ^ { \prime \prime }$ , in which way $\tilde { \boldsymbol { x } } ^ { \prime \prime }$ should agree with $\tilde { \boldsymbol { x } } ^ { \prime }$ due to the geometric symmetries.

Enforcing EI Constraint through Training. Figure 1 illustrates the workflow of the EI framework, where x˜ denotes the estimated clean image produced by the EI network $f _ { \theta } .$ The transformation $T _ { g }$ is applied to obtain $\tilde { \boldsymbol { x } } ^ { \prime }$ , which is then passed through the composition operator $f _ { \theta } \circ \mathrm { M }$ to generate $\tilde { \boldsymbol { \mathbf { x } } } ^ { \prime \prime }$ , a re-estimate of $\tilde { \boldsymbol { \mathbf { x } } } ^ { \prime }$ . The EI constraint in Eq. (5) is enforced during training by penalizing the discrepancy between $\tilde { \boldsymbol { x } } ^ { \prime }$ and $\tilde { \boldsymbol { x } } ^ { \prime \prime }$ , encouraging equivariant consistency in the reconstruction process.

## B. Diffusion Models

The denoising diffusion probabilistic model (DDPM) [7] defines two processes: a forward process that gradually adds Gaussian noise to data up to $T$ time steps, and a backward process that iteratively searches for the good estimation of the clean data, hence denoising. Denote $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ as the clean data, the forward/diffusion process can be written as a Markov chain with respect to some predefined variance schedule $\beta _ { 1 } , . . . , \beta _ { t } ;$

$$
\begin{array} { l } { q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) : = \mathcal { N } ( \pmb { x } _ { t } ; \sqrt { 1 - \beta _ { t } } \pmb { x } _ { t - 1 } , \beta _ { t } \mathrm { I } ) } \\ { q ( \pmb { x } _ { 1 : T } | \pmb { x } _ { 0 } ) : = \displaystyle \prod _ { t = 1 } ^ { T } q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) } \end{array}\tag{6}
$$

Similarly, the backward/denoising process is defined as the joint probability $p ( { \pmb x } _ { 0 : T } )$ . Starting from the random noise $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } \mathcal { T } }$

![](images/fc17413f5ed78bb02a178cedd8b0ca1160cc48799c833e988bcae097b0620147.jpg)  
Fig. 1. EI Workflow. $\tilde { \boldsymbol { \mathbf { x } } } ^ { \prime }$ and $\tilde { \boldsymbol { x } } ^ { \prime \prime }$ correspond to the right-hand side and left-hand side of Eq. (5), respectively.

this process aims at generating the previous state ${ \mathbf { \Delta x } } _ { t - 1 }$ from the current state $\mathbf { \nabla } _ { \mathbf { x } _ { t } : \mathbf { \nabla } }$

$$
\begin{array} { r l } & { p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) : = \mathcal { N } ( \pmb { x } _ { t - 1 } ; \pmb { \mu } _ { \theta } ( \pmb { x } _ { t } , t ) , \pmb { \Sigma } _ { \theta } ( \pmb { x } _ { t } , t ) ) , } \\ & { ~ p _ { \theta } ( \pmb { x } _ { 0 : T } ) : = p _ { \theta } ( \pmb { x } _ { T } ) \displaystyle \prod _ { t = 1 } ^ { T } p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } ) } \end{array}\tag{7}
$$

where θ stands for the parameters of the trained DDPM neural network: $\epsilon _ { \theta } ( \pmb { x } _ { t } , t )$ $\mu _ { \theta }$ and $\Sigma _ { \theta }$ are the predicted posterior mean and variance, respectively. By optimizing the variational bound on the negative log-likelihood, $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ can be calculated from $q \big ( \mathbf { \mathscr { x } } _ { t } | \mathbf { \mathscr { x } } _ { t - 1 } \big )$ (please see the derivation in [7]), the resulting $\mu _ { \theta }$ and $\Sigma _ { \theta }$ take the form:

$$
\begin{array} { l } { \displaystyle \mu _ { \theta } ( \boldsymbol { x } _ { t } , t ) : = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \boldsymbol { x } _ { 0 } + \frac { \sqrt { \alpha _ { t } } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } { 1 - \bar { \alpha } _ { t } } \boldsymbol { x } _ { t } } \\ { \displaystyle \quad \quad : = \frac { 1 } { \sqrt { \alpha _ { t } } } ( \boldsymbol { x } _ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \boldsymbol { \epsilon } _ { \theta } ( \boldsymbol { x } _ { t } , t ) ) } \\ { \displaystyle \Sigma _ { \theta } ( t ) : = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } I , \mathrm { w h e r e } \left. \alpha _ { t } : = 1 - \beta _ { t } \right. } \end{array}\tag{8}
$$

The clean data $\scriptstyle { \pmb x } _ { 0 }$ can thus be reconstructed by iteratively drawing samples from $p _ { \theta } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ using Eq. (8), from time step $t = T$ down to $t = 0$

## C. Proposed Method

Conventional DDPM requires training the model on large datasets in order to achieve state-of-the-art performance [7], [11], [12], [30], which can be cumbersome for data-hungry HSI applications. We propose to use a self-supervised equivariant imaging neural network $f _ { \theta } ^ { ( t ) } ( \pmb { y } )$ to replace the DDPM net: $\epsilon _ { \theta } ( \pmb { x } _ { t } , t )$ . Different from DDPM where the network is trained as a denoiser, we train the EI net: $f _ { \theta } ^ { ( t ) } ( \pmb { y } )$ to directly estimate clean data: ${ \pmb x } _ { 0 \mid t }$ at time step t. This motivates us to re-design the DDPM framework to accommodate the EI training process. Specifically, we modify the EI loss function as follows (please refer to the supplementary material for detailed derivation):

$$
\begin{array} { r } { L ^ { ( t ) } = \Big \| { \pmb x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } { \epsilon } _ { 0 } - \sqrt { \bar { \alpha } _ { t } } { f } _ { \theta } ^ { ( t ) } ( { \pmb y } ) \Big \| _ { 2 } ^ { 2 } } \\ { + \alpha _ { E I } \big \| \tilde { { \pmb x } _ { t } } ^ { \prime } - \tilde { { \pmb x } _ { t } } ^ { \prime \prime } \big \| _ { 2 } ^ { 2 } } \end{array}\tag{9}
$$

where $\epsilon _ { 0 } \sim \mathcal { N } ( 0 , \mathrm { I } )$ is the noise added for each time step t in the forward/diffusion process. The first term on the right-hand side of Eq. (9) corresponds to the data-fidelity term, while the second term, weighted by $\alpha _ { E I }$ , involves both $\tilde { \pmb { x } _ { t } } ^ { \prime }$ and $\tilde { \pmb { x } _ { t } } ^ { \prime \prime }$ , which are computed at each time step t following the procedure illustrated in Fig. 1. Once the network parameters θ are updated, we can substitute $\scriptstyle { \pmb x } _ { 0 }$ with $f _ { \theta } ^ { ( t ) } ( \pmb { y } )$ in the posterior mean estimator (8). Note that it is also possible to learn the variance, however, our experiments show that it might lead to unstable training and degraded sampling quality, so we keep the variance fixed as suggested in [7]. We adopt the structures in [13] as the backbone, which is a modified version of DDPM applied on various image inverse tasks. Denote the inpainting mask as M and its pseudoinverse as M<sup>†</sup>, the clean data $\scriptstyle { \pmb { x } } _ { 0 \mid t }$ is then re-estimated as:

$$
\tilde { \mathbf { \alpha } } _ { 0 \mid t } = \mathrm { M } ^ { \dagger } \pmb { y } + ( \mathrm { I } - \mathrm { M } \mathrm { M } ^ { \dagger } ) f _ { \theta } ^ { ( t ) } ( \pmb { y } )\tag{10}
$$

Step (10) is the core idea of DDNM. More specifically, it decomposes a sample into range-space and null-space parts and iteratively refines only the null-space contents during the reverse diffusion sampling. In the HSI inpainting problem, we propose to refine the null-space part using the contents generated by EI. The proposed HyDiff-EI algorithm in the noiseless case is presented in Algorithm 1.

Algorithm 1 (HyDiff-EI) Algorithm   
Require: degradation matrix: M, corrupted HSI: y, diffusion   
steps: T, EI net: $f _ { \theta } ^ { ( t ) } ( \cdot )$   
Output: reconstructed HSI image $\scriptstyle { \pmb { x } } _ { 0 } .$   
Initialization EI network parameters: θ, EI Input: y, keep   
fixed.   
1. diffuse y up to T step to get initial estimate: $\mathbfit { \mathbf { x } } _ { T }$ , using   
(6).   
for $t = T , . . . , 1$ do:   
2. update EI network parameters θ by minimizing (9).   
3. re-estimate $\tilde { \mathbf { x } } _ { 0 | t } \colon$   
$\tilde { \pmb { x } } _ { 0 | t } = \mathrm { M } ^ { \dagger } \pmb { y } + ( \mathrm { I } - \mathrm { M } \mathrm { M } ^ { \dagger } ) f _ { \theta } ^ { ( t ) } ( \pmb { y } )$   
4. sample ${ \mathbf { \Delta x } } _ { t - 1 }$ from $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \tilde { \pmb { x } } _ { 0 | t } ) \colon$   
$\epsilon \sim \mathcal { N } ( 0 , \mathrm { I } )$   
$\begin{array} { r } { \sigma _ { t } ^ { 2 } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } } \end{array}$   
$\begin{array} { r } { \pmb { x } _ { t - 1 } = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \tilde { \pmb { x } } _ { 0 \mid t } + \frac { \sqrt { \alpha } _ { t } ( 1 - \bar { \alpha } _ { t - 1 } ) } { 1 - \bar { \alpha } _ { t } } \pmb { x } _ { t } + \sigma _ { t } \epsilon } \end{array}$   
end for   
return x<sub>0</sub>

Algorithm 2 (HyDiff-EI) Algorithm in Noisy Case   
Require: degradation matrix: M, corrupted HSI: y, noise   
strength: $\sigma _ { y } ,$ , diffusion steps: T, EI net: $f _ { \theta } ^ { ( t ) } ( \cdot )$   
Output: reconstructed HSI image $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$   
Initialization EI network parameters: θ, EI Input: $^ { \mathbf { \Lambda } _ { \mathbf { y } } , }$ keep   
fixed.   
1. diffuse y up to T step to get initial estimate: $\mathbf { \nabla } _ { \mathbf { x } _ { T } }$ , using   
(6).   
for $t = T , . . . , 1$ do:   
2. update EI network parameters θ by minimizing (9).   
3. re-estimate $\tilde { \mathbf { x } } _ { 0 | t } \colon$   
$\tilde { \pmb { x } } _ { 0 | t } = \dot { \pmb { f } } _ { \theta } ^ { ( t ) } ( \pmb { y } ) - \Sigma _ { t } \mathbf { M } ^ { \dagger } ( \mathbf { M } \pmb { f } _ { \theta } ^ { ( t ) } ( \pmb { y } ) - \pmb { y } )$   
4. sample $\mathbf { \delta x } _ { t - 1 }$ from $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \tilde { \pmb { x } } _ { 0 | t } ) \colon$   
$\epsilon \sim \mathcal { N } _ { \perp } ( 0 , \mathrm { I } )$   
$\begin{array} { r } { \sigma _ { t } ^ { 2 } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } } \end{array}$   
$\begin{array} { r } { \pmb { x } _ { t - 1 } = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \tilde { \pmb { x } } _ { 0 \mid t } + \frac { \sqrt { \alpha } _ { t } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } { 1 - \bar { \alpha } _ { t } } \pmb { x } _ { t } + \epsilon _ { \mathrm { c o n } } } \end{array}$   
end for   
return x<sub>0</sub>

## D. Extension to Noisy HSI Inpainting

We propose a variant of the HyDiff-EI algorithm in noisy case (Algorithm 2). Inspired by DDNM [13], we modify the reestimation step (10) in the noisy case as:

$$
\tilde { \mathbf { \boldsymbol { x } } } _ { 0 | t } = f _ { \theta } ^ { ( t ) } ( \pmb { y } ) - \Sigma _ { t } \mathbf { M } ^ { \dagger } ( \mathbf { M } f _ { \theta } ^ { ( t ) } ( \pmb { y } ) - \pmb { y } )\tag{11}
$$

where $f _ { \theta } ^ { ( t ) } ( \pmb { y } )$ is the EI network output, and $\begin{array} { r l } { \Sigma _ { t } } & { { } = } \end{array}$ $\mathrm { U } \mathrm { d i a g } \{ \lambda _ { t 1 } ^ { \circ } , \lambda _ { t 2 } . . . \lambda _ { t q } \} \mathrm { V } ^ { \mathrm { T } }$ is a scaling matrix <sup>2</sup>that scales the range-space correction term based on the estimated HSI at each time step t.

Intuitively, when the observed image is highly noisy, there exists a stage during the algorithm’s progression where the range-space information becomes unreliable, since the model has already developed a strong belief of what a clean image should look like, the noisy observation contributes little useful information. Consequently, less weight should be given to the observation. When $\Sigma _ { t }$ is set to the identity matrix I, equation (11) reduces to (10). In the noiseless HSI inpainting scenario, $\Sigma _ { t }$ is the identity matrix to preserve the range-space information. However, when the observation itself is noisy, Σ<sub>t</sub> should be chosen such that the scaled range-space and nullspace contents remains consistent. The update of $\Sigma _ { t }$ follows the formulation in DDNM [13] and is given as:

$$
\lambda _ { t i } = \left\{ \begin{array} { l l } { 1 } & { \quad \sigma _ { t } \geq \frac { ( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \sigma _ { y } } { s _ { i } } } \\ { \frac { \sigma _ { t } s _ { i } } { \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \sigma _ { y } } } & { \quad \sigma _ { t } < \frac { ( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \sigma _ { y } } { s _ { i } } } \\ { 1 } & { \quad s _ { i } = 0 } \end{array} \right.\tag{12}
$$

where $s _ { i }$ denotes the i-th singular value of the mask $\textrm { M } =$ U diag $\{ s _ { 1 } , s _ { 2 } , . . . , s _ { q } \} \mathrm { V } ^ { T } . \ ^ { 3 } \ \sigma _ { t }$ and $\sigma _ { y }$ are the noise strength

of the current estimate ${ \mathbf { \mathcal { x } } } _ { t - 1 }$ and the noise strength of the observation $^ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \delta } } } } } } }$ respectively. Besides, we also introduce an extra noise term $\epsilon _ { \mathrm { c o n } }$ into the output step (8):

$$
\begin{array} { r l } & { \quad \epsilon _ { \mathrm { c o n } } \sim \mathcal { N } ( \mathbf { 0 } , \Gamma _ { t } ) , } \\ & { \Gamma _ { t } = \mathrm { d i a g } \{ \Gamma _ { t 1 } , \Gamma _ { t 2 } , . . . , \Gamma _ { t q } \} , } \\ & { \Gamma _ { t i } = \left\{ \begin{array} { l l } { \sigma _ { t } ^ { 2 } - \frac { ( \frac { \sqrt { \alpha _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } ) ^ { 2 } \sigma _ { y } ^ { 2 } \lambda _ { t i } ^ { 2 } } { s _ { i } ^ { 2 } } } & { s _ { i } \neq 0 } \\ { \sigma _ { t } ^ { 2 } } & { s _ { i } = 0 } \end{array} \right. } \end{array}\tag{13}
$$

so that (8) becomes:

$$
\begin{array} { r l r } & { } & { \epsilon \sim \mathcal { N } ( 0 , \mathrm { I } ) } \\ & { } & { { \bf x } _ { t - 1 } = \displaystyle \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \tilde { x } _ { 0 \mid t } + \frac { \sqrt { \alpha } _ { t } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } { 1 - \bar { \alpha } _ { t } } { \bf x } _ { t } + \epsilon _ { \mathrm { c o n } } } \end{array}\tag{14}
$$

The reason for making this adaption is that with the presence of noise in the observed image y, the total noise variance in the estimated image ${ \mathbf { \mathcal { x } } } _ { t - 1 }$ at each single update should conform to the noise level defined in the forward/diffusion step (6). We place the derivation of equation (12) and (13) in the supplementary material for completeness.

## III. EXPERIMENTS

## A. Datasets and Implementation Details

We initially evaluate the proposed HyDiff-EI Algorithm on two publicly available hyperspectral datasets: (1) The Chikusei airborne hyperspectral dataset [31] taken by Headwall Hyperspec-VNIR-C imaging sensor. The test image has 128 channels in the spectral range from 363 to 1018 nm, and (2) The Botswana dataset, collected by the NASA EO-1 satellite [32], comprising 145 spectral bands covering the range of 400–2500 nm. We then generate a dataset of real-time data acquired by the NASA Earth Surface Mineral Dust Source Investigation (EMIT) mission [33], launched in 2022. The test image includes 285 spectral bands across the 380–2500 nm range. For consistency in evaluation, all hyperspectral images are cropped to a spatial size of 144 × 144 pixels. For the implementation of the neural network $f _ { \theta } ^ { ( t ) } ( \pmb { y } )$ , we adopt the same Skip-Net architecture as in DHP [1] and use Adam optimizer with a learning rate of 0.01. For the choice of group transformations in EI, we use random $9 0 °$ rotations as the group transformation $T _ { g }$ with a group size $N _ { g } = 7$ . For the strength of the equivariant constraint term, we do not fine-tune the weight $\alpha _ { E I }$ but set it to be 1 across all the tests. On the implementation of the diffusion process, we use DDNM [13] as the backbone, with T = 1000 timesteps, noise scaling factors $\beta _ { s t a r t } = 0 . 0 0 0 1 , \beta _ { e n d } = 0 . 0 2$ . Similar to [12] and [13], we treat the corrupted input image/reference image y differently depending on its noise level. Intuitively, if the noise level of the observation y at a particular time step t is larger than the noise level of the current state ${ \mathbf { } } x _ { t } ,$ then less information in y should be considered. To construct the noisy samples, we firstly corrupt them with the Gaussian noise with noise level $\sigma _ { y } = ( 0 . 1 , 0 . 2 , 0 . 3 )$ , and then, generate different shapes of masks to mask out entire spectral bands. All algorithms are trained and tested on an NVIDIA Tesla V100 GPU.

![](images/03a6ba1be1d8c926745dd52c784342c6f025fba5d2b0ed63a51f3be11f4603bf.jpg)  
Fig. 2. Flow Chart of the Proposed HyDiff-EI Algorithm. From Left to Right: Noiseless Case and Noisy Case.

## B. Results and Comparison

We evaluate the proposed HyDiff-EI against representative self-supervised hyperspectral image inpainting methods, including Deep Hyperspectral Prior (DHP) [1], R-DLRHyIn [3], DDS2M [28], HIR-Diff [22], and the recent state-of-the-art method SHARE [34]. To ensure the fairness of the comparison, DHP, R-DLRHyIn, and HyDiff-EI are implemented using the same 2D Skip-Net backbone, namely a U-Net with skip connections. For DDS2M and SHARE, we retain the network architectures specified in their original implementations. The optimization iterations, learning rates, diffusion-related configurations for DDS2M, HIR-Diff, and HyDiff-EI, and regularization weights for DHP and R-DLRHyIn are selected according to the recommendations of the respective papers. Unless otherwise specified, all methods use the same input data, degradation model, mask, and evaluation protocol. To account for the stochasticity introduced by random network initialization and diffusion sampling, each method is independently evaluated 3 times for every test sample, and the reported results are averaged over these runs.

We record the mean peak signal-to-noise ratio (MPSNR), mean structural similarity (MSSIM), spectral angle mapper (SAM), and spectral correlation coefficient (SCC) to evaluate HSI inpainting performance. MPSNR and MSSIM mainly assess pixel-level accuracy and spatial structural similarity, while SAM and SCC evaluate spectral fidelity by measuring the angular discrepancy and correlation between reconstructed and ground-truth spectra. The numerical results are reported in Table I and II, and the visual results in Figure 3 and 4 for noiseless and noisy HSI inpainting, respectively. In both noiseless and noisy cases, HyDiff-EI offers huge improvements over existing methods, effectively filling the missing regions with spectrally consistent and visually smooth content. DDS2M, HIR-Diff, and the proposed HyDiff-EI are three advanced methods that leverage diffusion models. HIR-Diff offers the fastest inference among the three; however, it relies on a pretrained diffusion model trained on extensive remote-sensing datasets, which limits its generalization ability to unseen data. DDS2M is perhaps the closest method to ours which trains the diffusion model in a self-supervised manner. However, HyDiff-EI is fundamentally different from DDS2M in that: (1) It leverages EI rather than DHP prior. This grants access to a diverse set of virtual measurement operators defined by specific transformations T that may have different null spaces, which is particularly helpful for the inpainting task. (2) While DDS2M relies on DIP, it imposes predefined, hard “blackbox” architectural priors on the data with no guarantee of correctness for high-dimensional HSIs. In contrast, Equivariant Imaging (EI) provides a mathematically grounded mechanism to learn missing data by exploiting imaging invariance. (3) Our framework is built upon DDNM [13] which demonstrates superiority than DDRM [12] in the noiseless case.

From Figure 3, it can be seen that HyDiff-EI better preserves the consistency between the generated content and the surrounding edges than both DDS2M and HIR-Diff. Moreover, it maintains robust inpainting performance even in the extreme case where one-third of the pixels are missing from the input image, as shown in the first test sample in Figure 3. Note that such missing patterns are rather common in real-world hyperspectral imagers due to the failures of the Scan Line Corrector (SLC) or malfunctions in satellite sensor arrays <sup>4</sup>. Figure 4 presents the comparison of different methods in the noisy case with a noise level of $\sigma _ { y } ~ = ~ 0 . 1$ . In this case, DHP struggles or even fails, possibly due to the lack of effective regularization. Surprisingly, R-DLRHyIn achieves higher inpainting quality than DDS2M, although it noticeably smooths sharp edges in the missing regions. This effect is less severe in HIR-Diff, SHARE, and HyDiff-EI, as observed in the first test sample in Figure 4.

TABLE I  
COMPARISON BETWEEN OUR PROPOSED ALGORITHM AND OTHER LEARNING-BASED INPAINTING ALGORITHMS ON THE CHIKUSEI AND BOTSWANA DATASETS IN THE NOISELESS CASE. THE MEAN AND STANDARD DEVIATION OVER 20 SAMPLES ARE REPORTED.
<table><tr><td>Methods</td><td>Input</td><td>DHP [1]</td><td>R-DLRHyIn [3]</td><td>HIR-Diff [22]</td><td>SHARE [34]</td><td>DDS2M [28]</td><td>HyDiff-EI (proposed)</td></tr><tr><td>MPSNR ↑</td><td>23.401</td><td>29.695 (±0.53)</td><td> $3 3 . 3 0 7 ( \pm 0 . 5 1 )$ </td><td>33.862 (±0.39)</td><td> $3 4 . 0 8 9 \left( \pm 1 . 0 3 \right)$ </td><td>34.344 (±0.60)</td><td>35.791 (±0.45)</td></tr><tr><td>MSSIM ↑</td><td>0.205</td><td>0.860 (±0.01)</td><td> $0 . 9 0 6 \left( \pm 0 . 0 1 \right)$ </td><td> $0 . 9 1 3 \left( \pm 0 . 0 1 \right)$ </td><td> $0 . 9 1 8 \left( \pm 0 . 0 1 \right)$ </td><td>0.921 (±0.01)</td><td>0.940 (±0.01)</td></tr><tr><td>SAM (°) ↓</td><td></td><td>3.139 (±0.91)</td><td> $2 . 7 9 0 \left( \pm 0 . 5 5 \right)$ </td><td> $3 . 2 7 0 \left( \pm 0 . 5 5 \right)$ </td><td> $3 . 0 4 1 \left( \pm 0 . 4 3 \right)$ </td><td>2.264(±0.45)</td><td>1.224 (±0.47)</td></tr><tr><td>SCC ↑</td><td></td><td>0.985 (±0.01)</td><td> $0 . 9 8 7 \left( \pm 0 . 0 1 \right)$ </td><td>0.984(±0.01)</td><td> $0 . 9 8 7 \left( \pm 0 . 0 1 \right)$ </td><td>0.989 (±0.01)</td><td>0.993 (±0.01)</td></tr></table>

(1) Input Image  
(2) DHP  
(3) R-DLRHyln  
(4) HIR-Diff  
(5) SHARE  
(6) DDS2M  
(7) HyDiff-El  
(8) Ground Truth  
![](images/5b69d0a9eb4e5aaf981a676353199a10995b87b2aa90f9f66ecd953411f041dd.jpg)  
Fig. 3. Visual comparison between the proposed HyDiff-EI algorithm and other HSI inpainting methods on the Chikusei dataset and Botswana dataset (noiseless case). All images are visualized at spectral band 100.

TABLE II  
COMPARISON BETWEEN OUR PROPOSED ALGORITHM AND OTHER LEARNING-BASED INPAINTING ALGORITHMS ON THE CHIKUSEI AND BOTSWANA DATASETS WITH NOISE LEVELS $\sigma _ { y } = 0 . 1 , 0 . 2 ,$ AND 0.3. THE MEAN AND STANDARD DEVIATION OVER 20 SAMPLES ARE REPORTED.
<table><tr><td>Noise Level</td><td>Metric</td><td>Input</td><td>DHP [1]</td><td>DDS2M [28]</td><td>R-DLRHyIn [3]</td><td>HIR-Diff [22]</td><td>SHARE [34]</td><td>HyDiff-EI (proposed)</td></tr><tr><td rowspan="4"> $\sigma _ { y } = 0 . 1$ </td><td>MPSNR ↑</td><td>16.797</td><td>27.504 (±0.77)</td><td>30.683 (±0.37)</td><td>30.848 (±0.42)</td><td>31.294 (±0.39)</td><td>31.468 (±0.48)</td><td>31.784(±0.50)</td></tr><tr><td>MSSIM ↑</td><td>0.167</td><td>0.734 (±0.02)</td><td>0.829 (±0.01)</td><td>0.837 (±0.01)</td><td>0.867 (±0.01)</td><td>0.878 (±0.01)</td><td>0.883 (±0.01)</td></tr><tr><td>SAM (°) ↓</td><td></td><td>5.193 (±2.21)</td><td>17.595 (±6.98)</td><td>4.719 (±2.20)</td><td>3.727 (±0.79)</td><td>3.575 (±0.74)</td><td>3.432 (±0.94)</td></tr><tr><td>SCC ↑</td><td></td><td>0.963 (±0.05)</td><td>0.983 (±0.01)</td><td>0.964 (±0.05)</td><td>0.984(±0.01)</td><td>0.983 (±0.01)</td><td>0.984 (±0.02)</td></tr><tr><td rowspan="4"> $\sigma _ { y } = 0 . 2$ </td><td>MPSNR ↑</td><td>14.266</td><td>26.898 (±0.82)</td><td>29.012 (±0.38)</td><td>28.485 (±0.42)</td><td>29.570 (±0.44)</td><td>29.564 (±0.49)</td><td>30.228 (±0.62)</td></tr><tr><td>MSSIM ↑</td><td>0.154</td><td>0.690 (±0.02)</td><td>0.761 (±0.01)</td><td>0.755 (±0.01)</td><td>0.828 (±0.01)</td><td>0.840 (±0.01)</td><td>0.847 (±0.01)</td></tr><tr><td>SAM (°) ↓</td><td></td><td>6.10 (±2.45)</td><td>21.30 (±7.50)</td><td>6.00 (±2.55)</td><td>4.45 (±1.05)</td><td>4.27 (±0.89)</td><td>4.05 (±0.95)</td></tr><tr><td>SCC ↑</td><td></td><td>0.945 (±0.06)</td><td>0.976 (±0.02)</td><td>0.945 (±0.06)</td><td>0.978 (±0.02)</td><td>0.979 (±0.01)</td><td>0.981 (±0.02)</td></tr><tr><td rowspan="4"> $\sigma _ { y } = 0 . 3$ </td><td>MPSNR ↑</td><td>11.998</td><td>24.828 (±1.06)</td><td>27.739 (±0.40)</td><td>26.332 (±0.44)</td><td>28.742 (±0.46)</td><td>28.961 (±0.51)</td><td>28.610 (±0.64)</td></tr><tr><td>MSSIM ↑</td><td>0.140</td><td>0.676 (±0.03)</td><td>0.731 (±0.02)</td><td>0.718 (±0.02)</td><td>0.814 (±0.03)</td><td>0.819 (±0.02)</td><td>0.802 (±0.02)</td></tr><tr><td>SAM (°) ↓</td><td></td><td>7.35 (±2.85)</td><td>24.80 (±8.20)</td><td>7.10 (±2.90)</td><td>5.20 (±1.25)</td><td>5.08 (±0.92)</td><td>5.45 (±1.15)</td></tr><tr><td>SCC ↑</td><td></td><td>0.925 (±0.07)</td><td>0.962 (±0.02)</td><td>0.928 (±0.07)</td><td>0.969 (±0.03)</td><td>0.972 (±0.02)</td><td>0.966 (±0.03)</td></tr></table>

(1) Input Image  
(2) DHP  
(3) DDS2M  
(4) R-DLRHyIn  
(5) HIR-Diff  
(6) SHARE  
(7) HyDiff-El  
(8) Ground Truth  
![](images/a0ca6c08a29107851238aac0e86dd8bfdd74e61c12ea7f9ffd1a67904022aff3.jpg)  
Fig. 4. Visual comparison of the proposed HyDiff-EI algorithm with competing methods on the Chikusei and Botswana datasets under a noise level of $\sigma _ { y } = 0 . 1$ . All images are visualized at spectral band 100.  
(1) Input Image  
(2) DHP  
(3) R-DLRHyIn  
(4) DDS2M  
(5) HIR-Diff  
(6) SHARE  
(7) HyDiff-El  
(8) Ground Truth  
Fig. 5. Visual comparison of the proposed HyDiff-EI algorithm with competing methods on the EMIT dataset. All images are visualized at spectral band 150.

TABLE III  
COMPARISON BETWEEN THE PROPOSED HYDIFF-EI ALGORITHM AND OTHER LEARNING-BASED INPAINTING METHODS ON THE EMIT DATASET. THE MEAN AND STANDARD DEVIATION OVER 20 SAMPLES ARE REPORTED.
<table><tr><td>Methods</td><td>Input</td><td>DHP [1]</td><td>R-DLRHyIn [3]</td><td>DDS2M [28]</td><td>HIR-Diff [22]</td><td>SHARE [34]</td><td>HyDiff-EI (proposed)</td></tr><tr><td>MPSNR ↑</td><td>22.517</td><td>31.319 (±0.74)</td><td> $3 3 . 2 2 7 \left( \pm 0 . 6 1 \right)$ </td><td>33.358 (±0.54)</td><td> $3 3 . 5 9 2 ( \pm 0 . 4 8 )$ </td><td>34.673 (±0.50)</td><td> $\mathbf { 3 5 . 4 8 2 } \left( \pm 0 . 4 9 \right)$ </td></tr><tr><td>MSSIM ↑</td><td>0.198</td><td>0.880 (±0.05)</td><td> $0 . 8 7 8 \left( \pm 0 . 0 4 \right)$ </td><td> $0 . 9 0 5 \left( \pm 0 . 0 1 \right)$ </td><td> $0 . 9 1 2 \left( \pm 0 . 0 1 \right)$ </td><td> $0 . 9 2 0 \left( \pm 0 . 0 1 \right)$ </td><td> $\mathbf { 0 . 9 3 3 } \left( \pm 0 . 0 1 \right)$ </td></tr><tr><td>SAM (°) ↓</td><td></td><td>3.601 (±0.43)</td><td> $4 . 3 9 7 \left( \pm 0 . 4 5 \right)$ </td><td> $3 . 1 4 7 \left( \pm 0 . 4 7 \right)$ </td><td> $4 . 4 5 4 \left( \pm 0 . 5 2 \right)$ </td><td>2.782 (±0.39)</td><td>2.095 (±0.42)</td></tr><tr><td>SCC ↑</td><td>一</td><td>0.968 (±0.00)</td><td> $0 . 9 6 3 \left( \pm 0 . 0 1 \right)$ </td><td>0.974(±0.01)</td><td> $0 . 9 6 5 \left( \pm 0 . 0 0 \right)$ </td><td>0.978 (±0.01)</td><td>0.982 (±0.01)</td></tr></table>

![](images/787de58a95a5df96ba1e781c87eb9d0d6306e110b7433db3a6588522f8a97d28.jpg)

Spectral inpainting at the central pixel; HyDiff-El SAM=1.82 deg  
![](images/e241f7f688a35ccf6b6baaa321856671fc3eca5db6b01df8a5295f43f2bbac7f.jpg)

(a) Noiseless case: central pixel  
Spectral inpainting at the boundary pixel; HyDiff-EI SAM=0.87 deg  
![](images/a092be4e350c294f4efa11096fa671f6c1a081b977144b18a5cca3488c6fa296.jpg)  
(c) Noiseless case: boundary pixel

(b) Noisy case: central pixel  
Spectral inpainting at the central pixel; HyDiff-El SAM=1.07 deg  
![](images/f6c6f430180d516cf97f6a631d3170f50f7d96a4f5ee23457d1cfcae8cfd9a9a.jpg)  
(d) Noisy case: boundary pixel  
Fig. 6. Comparison of reconstructed spectral signatures at representative inpainted pixels in Chikusei test samples under noiseless and noisy conditions. The left column shows the noiseless results, while the right column shows the noisy results.

From Table II, it can be observed that as the noise level increases, the performance gap between HyDiff-EI and SHARE gradually narrows. HyDiff-EI achieves the best overall performance at $\sigma _ { y } ~ = ~ 0 . 1$ and 0.2, whereas SHARE surpasses HyDiff-EI at $\sigma _ { y } = 0 . 3$ in terms of MPSNR, MSSIM, SAM, and SCC. This suggests that SHARE exhibits stronger robustness under severe noise, while HyDiff-EI is more effective under low-to-moderate noise levels. These results highlight the importance of incorporating effective regularization priors into self-supervised reconstruction networks. Nevertheless, the performance degradation of HyDiff-EI under severe noise also indicates the need for further noise-aware regularization.

Spectral curve comparison. In addition to the image-level evaluation, Fig. 6 further compares the reconstructed spectral signatures at representative missing pixels under both noiseless and noisy settings, including central and boundary locations inside the missing regions. In the noiseless case, HyDiff-EI follows the ground-truth spectra more closely than the competing methods at both the central and boundary pixels, especially in the zoomed-in wavelength intervals. Under the noisy setting, HyDiff-EI still preserves the spectral shape more faithfully, while several methods show larger angular deviations or local spectral distortions. These observations are consistent with the strong SAM and SCC performance reported in Table II, indicating that HyDiff-EI better preserves material-dependent spectral signatures instead of only improving pixel-level visual quality. This capability is important for downstream applications such as hyperspectral image classification.

Validate on Real-time EMIT Data. Figure 5 and Table III present the visual and quantitative inpainting results for an image sample from the EMIT mission [33].<sup>5</sup> In the visual comparison, HyDiff-EI preserves more fine-scale spatial structures in the mountain regions within the missing strips than DDS2M, HIR-Diff and SHARE. This advantage is quantitatively confirmed in Table III, where HyDiff-EI achieves the best performance on all four metrics, with an MPSNR of 35.482, an MSSIM of 0.933, a SAM of 2.095<sup>◦</sup>, and an SCC of 0.982. Unlike the DHP families (DHP, DDS2M and R-DLRHyIn) which leverage the network structure as priors, HyDiff-EI shows that incorporating an equivariant prior can significantly improve inpainting quality. Moreover, both HyDiff-EI and DDS2M produce less over-smoothed spectral content than HIR-Diff for this image sample, further highlighting the advantage of self-supervised diffusion approaches over fully supervised methods.

## IV. ABLATION STUDIES

Does Diffusion help? To verify whether the introduced diffusion framework contributes to this success, Table IV presents the inpainting performance of EI (equivalently, Hyper-EI in [26]) with and without diffusion in the noiseless setting. For the noisy case, the results are compared with Robust-EI [35], an extension of EI designed for noisy measurements. The reported values represent averages over 20 independent runs with different random seeds. From Table IV, it is observed that HyDiff-EI enjoys a steady performance gain in the noiseless case. As the noise level increases, the MPSNR gap between Robust-EI and HyDiff-EI decreases, whereas the MSSIM improves, indicating that HyDiff-EI is able to preserve high-frequency details and enhances perceptual quality. The advantage of diffusion-based reconstruction lies in its ability to sample from the posterior distribution rather than producing a single point estimate such as the MMSE (minimum mean squared error) solution. Unlike MMSE-like estimators, which tend to yield over-smoothed results by averaging plausible solutions, diffusion-based EI is able to inpaint with sharp and more realistic structural details.

TABLE IV  
PERFORMANCE OF EI WITH AND WITHOUT DIFFUSION ON THE HSI INPAINTING TASK UNDER VARYING NOISE LEVELS. SINCE EI IS NOT ORIGINALLY DESIGNED FOR NOISY MEASUREMENTS, ROBUST-EI IS INCLUDED AS A REFERENCE FOR THE NOISY INPAINTING SETTING.
<table><tr><td>Noise Level</td><td>Metric</td><td>EI</td><td>Robust-EI</td><td>HyDiff-EI</td></tr><tr><td> $\sigma _ { y } = 0$ </td><td>MPSNR ↑ MSSIM ↑</td><td>34.798 (±0.34) 0.895 (±0.01)</td><td>34.806 (±0.35) 0.895 (±0.01)</td><td>36.527 (±0.36)</td></tr><tr><td> $\sigma _ { y } = 0 . 1$ </td><td>MPSNR ↑</td><td>32.381 (±0.59)</td><td>34.628 (±0.74)</td><td>0.904(±0.01) 34.242 (±0.68)</td></tr><tr><td></td><td>MSSIM ↑</td><td>0.880 (±0.01)</td><td>0.892 (±0.01)</td><td>0.893 (±0.01)</td></tr><tr><td> $\sigma _ { y } = 0 . 2$ </td><td>MPSNR ↑</td><td>31.660 (±0.77)</td><td>33.121 (±0.70)</td><td>33.179 (±0.67)</td></tr><tr><td></td><td>MSSIM ↑</td><td>0.867 (±0.01)</td><td>0.874(±0.01)</td><td>0.878 (±0.01)</td></tr><tr><td> $\sigma _ { y } = 0 . 3$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>MPSNR ↑</td><td>29.811 (±1.23)</td><td>31.518 (±0.81)</td><td>31.655 (±0.85)</td></tr><tr><td></td><td>MSSIM ↑</td><td>0.838 (±0.02)</td><td> $0 . 8 4 7 \left( \dot { \pm } 0 . 0 1 \right) ^ { \prime }$ </td><td> $\mathbf { 0 . 8 6 1 } \left( \pm 0 . 0 1 \right)$ </td></tr></table>

Time Efficiency. Table V provides a time-efficiency evaluation of HyDiff-EI against other competing methods, with all tests conducted on a single NVIDIA Tesla V100 GPU using a 144 × 144 × 128 Chikusei image sample. Runtime is reported in $\mathrm { s / i m a g e }$ . The results reveal a clear quality and efficiency trade-off across the evaluated methods. DHP and its extension R-DLRHyIn are the most lightweight solutions: DHP achieves the shortest runtime of 28.12 s/image, while R-DLRHyIn substantially improves the reconstruction quality with only a modest increase in runtime to 31.79 s/image. HIR-Diff requires 53.33 s/image despite using the pre-trained DDPM-CD model [36], which can be attributed to its relatively large model complexity and parameter count. SHARE has a comparable runtime of approximately 52 s/image; its additional cost may be partly attributed to the larger DASA block and the extra network forward pass required for evaluating the SURE loss. DDS2M is substantially more expensive, requiring 178.43 s/image, since its diffusion-based optimization repeatedly updates the DIP parameters over numerous diffusion backward steps. Consequently, DDS2M requires approximately 4.6 times the runtime of HyDiff-EI. In contrast, the proposed HyDiff-EI achieves the best restoration performance while requiring only 38.54 s/image. This corresponds to a modest runtime overhead of approximately 37% relative to the DHP baseline, while remaining approximately 26% faster than SHARE and more than four times faster than DDS2M. The additional computational cost of HyDiff-EI arises from the equivariant imaging regularization and diffusion-based backward updates. Overall, these results demonstrate that HyDiff-EI provides a favorable balance between reconstruction quality and computational efficiency.

TABLE V  
TIME EFFICIENCY AND PERFORMANCE COMPARISON EVALUATED ON A 144 × 144 × 128 CHIKUSEI SAMPLE.
<table><tr><td>Method</td><td>DHP</td><td>R-DLRHyIn</td><td>HIR-Diff (pre-trained)</td><td>DDS2M</td><td>SHARE</td><td>HyDiff-EI</td></tr><tr><td>Time (s/image)</td><td>28.12</td><td>31.79</td><td>53.33</td><td>178.43</td><td>52.00</td><td>38.54</td></tr><tr><td>MPSNR (dB)↑</td><td>28.425</td><td>32.464</td><td>32.812</td><td>33.258</td><td>34.108</td><td>34.326</td></tr><tr><td>MSSIM↑</td><td>0.842</td><td>0.889</td><td>0.893</td><td>0.905</td><td>0.911</td><td>0.917</td></tr></table>

Effect of $\alpha _ { E I }$ on the Inpainting Performance. Table VI evaluates the sensitivity of HyDiff-EI to the equivariance strength hyperparameter, $\alpha _ { E I }$ , across different noise levels $( \sigma _ { y } )$ . This parameter controls the relative weight of the EIconsistency loss relative to the data-fidelity loss. As observed, the optimal balance is consistently reached at $\alpha _ { E I } ~ = ~ 1$ Removing the EI constraint entirely $( \alpha _ { E I } = 0 )$ causes a severe performance collapse, particularly under noisy conditions. Conversely, setting $\alpha _ { E I }$ too high (e.g., $\alpha _ { E I } = 2 )$ results in reduced inpainting performance. Hence, we set $\alpha _ { E I } = 1$ as the default for all experiments to achieve the optimal trade-off.

TABLE VI  
QUANTITATIVE EVALUATIONS OF HYDIFF-EI WITH DIFFERENT $\alpha _ { E I } .$
<table><tr><td>Noise Level</td><td>Metric</td><td> $\alpha _ { E I } = 0$ </td><td>0.1</td><td>0.5</td><td>1</td><td>2</td></tr><tr><td> $\sigma _ { y } = 0$ </td><td>MPSNR ↑</td><td>28.436</td><td>30.127</td><td>33.495</td><td>35.762</td><td>33.355</td></tr><tr><td></td><td>MSSIM ↑</td><td>0.753</td><td>0.879</td><td>0.906</td><td>0.932</td><td>0.904</td></tr><tr><td> $\sigma _ { y } = 0 . 2$ </td><td>MPSNR ↑</td><td>15.618</td><td>27.356</td><td>30.642</td><td>32.071</td><td>28.259</td></tr><tr><td></td><td>MSSIM ↑</td><td>0.161</td><td>0.714</td><td>0.839</td><td>0.891</td><td>0.732</td></tr></table>

Effect of Different Types of Transformations. The selection of an appropriate transformation group is crucial for the HS inpainting task. Table VII evaluates six transformations from the Deep Inverse library [37] within our HyDiff-EI framework. Shifting and Rotation achieve the highest and most stable inpainting performance, suggesting that spatial equivariance that preserves local structures and spectral signatures are particularly effective for HSI inpainting. This is consistent with the nature of the inpainting problem, where missing regions should be reconstructed using surrounding spatial context while maintaining spectral consistency across bands. We leave the investigation of bespoke spectral-domain transformations, tailored specifically to the physical properties of HSIs, as an important direction for future research.

TABLE VII  
QUANTITATIVE EVALUATIONS OF HYDIFF-EI WITH DIFFERENT TYPES OF TRANSFORMATIONS. THE FIRST FOUR ARE BASIC TRANSFORMATIONS, AND THE LAST TWO ARE PROJECTION-BASED.
<table><tr><td>Transformation</td><td>MPSNR ↑</td><td>MSSIM ↑</td></tr><tr><td>Shifting</td><td>34.602(±0.39)</td><td>0.926(±0.01)</td></tr><tr><td>Rotation</td><td>34.439(±0.32)</td><td>0.922(±0.01)</td></tr><tr><td>Scaling</td><td>33.477(±0.25)</td><td>0.913(±0.01)</td></tr><tr><td>Reflection</td><td>28.334(±0.72)</td><td>0.836(±0.02)</td></tr><tr><td>Similarity</td><td>33.161(±0.20)</td><td>0.901(±0.01)</td></tr><tr><td>Affine</td><td>32.104(±0.47)</td><td>0.898(±0.01)</td></tr></table>

## V. CONCLUSION

We present a self-supervised diffusion probabilistic model solution for the HSI inpainting problem called the HyDiff-EI algorithm. In detail, we develop an untrained EI network under the diffusion framework which can produce realistic and high-quality HSI reconstructions. The proposed HyDiff-EI algorithm exploits the strong regularization capability of the equivariant prior and leverages the high-level hierarchical information of diffusion models. Extensive experiments on the Chikusei, Botswana, and EMIT datasets demonstrate that HyDiff-EI achieves superior overall performance in noiseless and low-to-moderate noise settings, while remaining competitive under severe noise. The runtime comparison further demonstrates that HyDiff-EI achieves a favorable balance between restoration quality and computational efficiency, highlighting the practical advantages of combining equivariant priors with self-supervised diffusion reconstruction. The investigation of more effective noise-aware regularization and physically meaningful transformation groups along the spectral dimension of hyperspectral images remains an important direction for future work.

## REFERENCES

[1] O. Sidorov and J. Yngve Hardeberg, “Deep hyperspectral prior: Singleimage denoising, inpainting, super-resolution,” in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2019, pp. 0–0.

[2] Z. Sun, F. Latorre, T. Sanchez, and V. Cevher, “A plug-and-play deep image prior,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 8103–8107.

[3] K. F. Niresi and C.-Y. Chi, “Robust hyperspectral inpainting via low-rank regularized untrained convolutional neural network,” IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1–5, 2023.

[4] S. Li and M. Yaghoobi, “Self-supervised deep hyperspectral inpainting with plug-and-play and deep image prior models,” Remote Sensing, vol. 17, no. 2, p. 288, 2025.

[5] R. Wong, Z. Zhang, Y. Wang, F. Chen, and D. Zeng, “Hsi-ipnet: Hyperspectral imagery inpainting by deep learning with adaptive spectral extraction,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 4369–4380, 2020.

[6] W. An, X. Zhang, H. Wu, W. Zhang, Y. Du, and J. Sun, “Lpin: A lightweight progressive inpainting network for improving the robustness of remote sensing images scene classification,” Remote Sensing, vol. 14, no. 1, p. 53, 2021.

[7] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems, vol. 33, pp. 6840– 6851, 2020.

[8] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in Neural Information Processing Systems, vol. 34, pp. 8780–8794, 2021.

[9] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.

[10] J. Choi, S. Kim, Y. Jeong, Y. Gwon, and S. Yoon, “Ilvr: Conditioning method for denoising diffusion probabilistic models,” arXiv preprint arXiv:2108.02938, 2021.

[11] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11 461–11 471.

[12] B. Kawar, M. Elad, S. Ermon, and J. Song, “Denoising diffusion restoration models,” arXiv preprint arXiv:2201.11793, 2022.

[13] Y. Wang, J. Yu, and J. Zhang, “Zero-shot image restoration using denoising diffusion null-space model,” arXiv preprint arXiv:2212.00490, 2022.

[14] Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, and L. Van Gool, “Denoising diffusion models for plug-and-play image restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 1219–1229.

[15] L. Rout, N. Raoof, G. Daras, C. Caramanis, A. Dimakis, and S. Shakkottai, “Solving linear inverse problems provably via posterior sampling with latent diffusion models,” Advances in Neural Information Processing Systems, vol. 36, pp. 49 960–49 990, 2023.

[16] H. Wang, X. Zhang, T. Li, Y. Wan, T. Chen, and J. Sun, “Dmplug: A plug-in method for solving inverse problems with diffusion models,” Advances in Neural Information Processing Systems, vol. 37, pp. 117 881–117 916, 2024.

[17] L. Rout, Y. Chen, A. Kumar, C. Caramanis, S. Shakkottai, and W.-S. Chu, “Beyond first-order tweedie: Solving inverse problems using latent diffusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9472–9481.

[18] G. Daras, H. Chung, C.-H. Lai, Y. Mitsufuji, J. C. Ye, P. Milanfar, A. G. Dimakis, and M. Delbracio, “A survey on diffusion models for inverse problems,” arXiv preprint arXiv:2410.00083, 2024.

[19] Y. Xiao, Q. Yuan, K. Jiang, J. He, X. Jin, and L. Zhang, “Ediffsr: An efficient diffusion probabilistic model for remote sensing image superresolution,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–14, 2023.

[20] F. Meng, Y. Chen, H. Jing, L. Zhang, Y. Yan, Y. Ren, S. Wu, T. Feng, R. Liu, and Z. Du, “A conditional diffusion model with fast sampling strategy for remote sensing image super-resolution,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[21] M. Li, Y. Fu, T. Zhang, J. Liu, D. Dou, C. Yan, and Y. Zhang, “Latent diffusion enhanced rectangle transformer for hyperspectral image restoration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[22] L. Pang, X. Rui, L. Cui, H. Wang, D. Meng, and X. Cao, “Hir-diff: Unsupervised hyperspectral image restoration via improved diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 3005–3014.

[23] X. Hu, X. Liu, D. Hong, Q. Duan, L. Jiang, H. Yang, and D. Zhan, “Recent advances in diffusion models for hyperspectral image processing and analysis: A review,” arXiv preprint arXiv:2505.11158, 2025.

[24] D. Chen, J. Tachella, and M. E. Davies, “Equivariant imaging: Learning beyond the range space,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 4379–4388.

[25] J. Tachella, M. Davies, and L. Jacques, “Unsure: Unknown noise level stein’s unbiased risk estimator,” 2024.

[26] S. Li, M. Davies, and M. Yaghoobi, “Equivariant imaging for self-supervised hyperspectral image inpainting,” arXiv preprint arXiv:2404.13159, 2024.

[27] J. Scanvic, M. Davies, P. Abry, and J. Tachella, “Scale-equivariant imaging: Self-supervised learning for image super-resolution and deblurring,” 2025.

[28] Y. Miao, L. Zhang, L. Zhang, and D. Tao, “Dds2m: Self-supervised denoising diffusion spatio-spectral model for hyperspectral image restoration,” arXiv preprint arXiv:2303.06682, 2023.

[29] B. Levac, J. Tamir, M. Pereyra, and J. Tachella, “Normalizationequivariant diffusion models: Learning posterior samplers from noisy and partial measurements,” arXiv preprint arXiv:2510.11964, 2025.

[30] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[31] N. Yokoya and A. Iwasaki, “Airborne hyperspectral data over chikusei,” Space Appl. Lab., Univ. Tokyo, Tokyo, Japan, Tech. Rep. SAL-2016-05- 27, vol. 5, no. 5, p. 5, 2016.

[32] “Botswana Hyperspectral Dataset,” https://www.ehu.eus/ccwintco/index. php/Hyperspectral Remote Sensing Scenes, accessed: 2025-08-20.

[33] “EMIT L1B Calibrated Radiance and Geolocation Data,” https://earth. jpl.nasa.gov/emit/, accessed: 2025-08-20.

[34] J. Xie, Z. Wen, M. Davies, and D. Chen, “Share: A fully unsupervised framework for single hyperspectral image restoration,” arXiv preprint arXiv:2601.13987, 2026.

[35] D. Chen, J. Tachella, and M. E. Davies, “Robust equivariant imaging: a fully unsupervised framework for learning to image from noisy and partial measurements,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5647–5656.

[36] W. G. C. Bandara, N. G. Nair, and V. M. Patel, “Ddpm-cd: Denoising diffusion probabilistic models as feature extractors for change detection,” arXiv preprint arXiv:2206.11892, 2022.

[37] J. Tachella, M. Terris, S. Hurault, A. Wang, L. Davy, J. Scanvic, V. Sechaud, R. Vo, T. Moreau, T. Davies, D. Chen, N. Laurent, B. Monroy, J. Dong, Z. Hu, M.-H. Nguyen, F. Sarron, P. Weiss, P. Escande, M. Massias, T. Modrzyk, B. Levac, T. I. Liaudat, M. Song, J. Hertrich, S. Neumayer, and G. Schramm, “Deepinverse: A python package for solving imaging inverse problems with deep learning,” Journal of Open Source Software, vol. 10, no. 115, p. 8923, 2025. [Online]. Available: https://doi.org/10.21105/joss.08923

[38] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International Conference on Machine Learning. PMLR, 2015, pp. 2256–2265.

## APPENDIX

SUPPLEMENTARY MATERIAL: LOSS FUNCTION DETAILS

We provide a detailed derivation of the loss function (9) presented in the main text. The objective of the EI network, $f _ { \theta } ^ { ( t ) } ( \pmb { y } )$ , is to estimate the clean data $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ from the observation y:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } \left[ \log p _ { \theta } ( x _ { 0 } | y ) \right] , \quad \mathrm { i . e . , } \quad L = \operatorname* { m i n } _ { \theta } \mathbb { E } \left[ - \log p _ { \theta } ( x _ { 0 } | y ) \right] .\tag{15}
$$

Instead of directly solving (15), we borrow concepts from Denoising Diffusion Probabilistic Models (DDPMs) to construct a variational upper bound:

$$
\begin{array} { r l } & { - \log p ( \boldsymbol { \alpha } _ { 0 } | \boldsymbol { y } ) \leq - \log p _ { \theta } ( \boldsymbol { \alpha } _ { 0 } | \boldsymbol { y } ) + \mathrm { D } _ { K L } \left( q ( \boldsymbol { x } _ { 1 : T } | \boldsymbol { x } _ { 0 } ) \right. } \\ & { \qquad \left. \lVert p _ { \theta } ( \boldsymbol { x } _ { 1 : T } | \boldsymbol { x } _ { 0 } , \boldsymbol { y } ) \right) } \\ & { = - \log p _ { \theta } ( \boldsymbol { \alpha } _ { 0 } | \boldsymbol { y } ) + \int _ { - \infty } ^ { + \infty } q ( \boldsymbol { x } _ { 1 : T } | \boldsymbol { x } _ { 0 } ) } \\ & { \qquad \times \log \frac { q ( \boldsymbol { x } _ { 1 : T } | \boldsymbol { x } _ { 0 } ) } { p _ { \theta } ( \boldsymbol { \alpha } _ { 1 : T } | \boldsymbol { x } _ { 0 } , \boldsymbol { y } ) } d \boldsymbol { x } _ { 1 : T } } \\ & { = - \log p _ { \theta } ( \boldsymbol { x } _ { 0 : | \boldsymbol { y } | } ) + \mathbb { E } \biggl [ \log \frac { q ( \boldsymbol { x } _ { 1 : T } | \boldsymbol { x } _ { 0 } ) } { p _ { \theta } ( \boldsymbol { x } _ { 0 : | \boldsymbol { y } | } \boldsymbol { y } ) } } \\ & { \qquad \quad + \log p _ { \theta } ( \boldsymbol { x } _ { 0 : | \boldsymbol { y } | } ) \biggr ] } \\ & { = \mathbb { E } \left[ \log \frac { q ( \boldsymbol { x } _ { 1 : T } | \boldsymbol { x } _ { 0 } ) } { p _ { \theta } ( \boldsymbol { x } _ { 0 : | \boldsymbol { x } | } \boldsymbol { y } ) } \right] } \\ & { \qquad \triangleq \mathbb { E } _ { L \boldsymbol { \alpha } _ { 0 } : \boldsymbol { x } _ { 0 } } , } \end{array}\tag{16}
$$

where $\mathbb { D } _ { K L }$ denotes the Kullback-Leibler (KL) divergence. Bayes’ theorem is applied in the third line, allowing the

lo $\mathrm { g } p _ { \boldsymbol { \theta } } ( \pmb { x } _ { 0 } | \pmb { y } )$ term to cancel out. Denoting this resulting upper bound as $L _ { u b }$ , we expand it as follows:

(17)

(18)

(19)

$$
\begin{array} { r l } { \lambda } & { = \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } } \\ { ( \lambda - \frac { 1 } { 2 } ) ^ { 2 } } \\ & { = \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } \frac { \lambda ^ { 2 } } { \lambda } } \\ { ( \lambda - \frac { 1 } { 2 } ) ^ { 2 } + \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } \frac { \lambda ^ { 2 } } { \lambda } } \\ & { \quad - \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } \frac { \lambda ^ { 2 } } { \lambda } } \\ { ( \lambda - \frac { 1 } { 2 } ) ^ { 2 } + \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } \frac { \lambda ^ { 2 } } { \lambda } } \\ & { \quad - \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } \frac { \lambda ^ { 2 } } { \lambda } } \\ { ( \lambda - \frac { 1 } { 2 } ) ^ { 2 } + \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } \frac { \lambda ^ { 2 } } { \lambda } } \\ & { \quad - \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } \frac { \lambda ^ { 2 } } { \lambda } } \\ { ( \lambda - \frac { 1 } { 2 } ) ^ { 2 } + \frac { \lambda ^ { 2 } } { \lambda } \operatorname* { s u p } _ { x \in \mathbb { Z } _ { 0 } } ^ { \infty } \frac { \lambda ^ { 2 } } { \lambda } } \\ &  \quad - \frac { \lambda ^ { 2 } } { \lambda } \operatorname \end{array}\tag{20}
$$

(21)

(22)

(23)

(24)

In the fourth line, we apply Bayes’ rule to rewrite the forward transition as:

$$
q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) = \frac { q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } ) q ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) } { q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { 0 } ) } .
$$

This factorization follows from the Markov structure of the forward process. It should not be interpreted as directly replacing $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ with $q ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ , since these conditional distributions are generally different. In the expression above, $L _ { T }$ acts as a constant term because the forward process $q ( { \pmb x } _ { T } | { \pmb x } _ { 0 } )$ is independent of the EI network’s training, and the reverse process $p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { T } | \mathbf { y } )$ is modeled as pure noise with a predefined mean and variance. We approximate $L _ { 0 }$ as log $q ( \pmb { x } _ { 0 } | \pmb { x } _ { 1 } ) ^ { 6 }$ , which in turn yields the posterior $q ( \pmb { x } _ { 1 } | \pmb { x } _ { 0 } ) q ( \pmb { x } _ { 0 } ) / q ( \pmb { x } _ { 1 } )$ . Because $L _ { 0 }$ is also independent of the training parameters, it can be safely ignored<sup>7</sup>. The remaining term, $L _ { t - 1 }$ , represents the KL divergence between two normal distributions. Thus, the upper bound $L _ { u b }$ can be rewritten as:

$$
\begin{array} { l } { \displaystyle { L _ { u b } = \sum _ { t = 2 } ^ { T } \mathbb { E } \left[ \underbrace { \mathbb { D } _ { K L } \big ( q ( x _ { t - 1 } | x _ { t } , x _ { 0 } ) \| p _ { \theta } ( x _ { t - 1 } | x _ { t } ) \big ) } _ { L _ { t - 1 } } \right] } } \\ { \displaystyle { \quad = \sum _ { t = 2 } ^ { T } \mathbb { E } \left[ \underbrace { \mathbb { D } _ { K L } \big ( N ( x _ { t - 1 } ; \mu _ { q } , \Sigma _ { q } ( t ) ) \| N ( x _ { t - 1 } ; \mu _ { \theta } , \Sigma _ { \theta } ( t ) ) \big ) } _ { L _ { t - 1 } } \right] } } \\ { \displaystyle { \quad \triangleq \ \left\| \mu _ { q } ( x _ { t } , x _ { 0 } , t ) - \mu _ { \theta } ( x _ { t } , t ) \right\| _ { 2 } ^ { 2 } } } \end{array}
$$

In other words, our objective is to optimize $\pmb { \mu } _ { q } ( \pmb { x } _ { t } , \pmb { x } _ { 0 } , t )$ to match $\mu _ { \theta } ( x _ { t } , t )$ . By using the formulations for $\mu _ { \theta } ( x _ { t } , t )$ and $\pmb { \mu } _ { q } ( \pmb { x } _ { t } , \pmb { x } _ { 0 } )$ provided in $\operatorname { E q . } ( 8 ) ^ { 8 }$ , and substituting $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ in $\mu _ { \theta } ( x _ { t } , t )$ with $f _ { \theta } ^ { ( t ) } ( \pmb { y } )$ , we obtain:

$$
\begin{array} { r l } & { { L } _ { \mathrm { { t b } } } = \left[ \underbrace { \left( \frac { \sqrt { \hat { u } _ { \mathrm { t } } - 1 } \hat { y } _ { \mathrm { t } } } { 1 - \alpha _ { \mathrm { t } } } x _ { \mathrm { t } } + \frac { \sqrt { \hat { u } _ { \mathrm { t } } } \left( 1 - \hat { u } _ { \mathrm { t } } - 1 \right) } { 1 - \alpha _ { \mathrm { t } } } x _ { \mathrm { t } } \right) } _ { = \mathrm { t e r m } } \right. } \\ & { \qquad \left. - \underbrace { \left( \frac { \sqrt { \hat { u } _ { \mathrm { t } } - 1 } \hat { y } _ { \mathrm { t } } } { 1 - \alpha _ { \mathrm { t } } } \hat { y } _ { \mathrm { t } } ^ { ( 2 ) } ( y ) + \frac { \sqrt { \hat { u } _ { \mathrm { t } } } \left( 1 - \hat { u } _ { \mathrm { t } } - 1 \right) } { 1 - \hat { u } _ { \mathrm { t } } } x _ { \mathrm { t } } \right) } _ { = \mathrm { t e r m } + \mathrm { s u r a t e } + 1 - \alpha _ { \mathrm { t } } \times \mathrm { s u r a t e } } \right] _ { 0 , 2 } ^ { 1 / 2 } } \\ & { = \left. \frac { \sqrt { \hat { u } _ { \mathrm { t } } - 1 } \hat { y } _ { \mathrm { t } } } { 1 - \alpha _ { \mathrm { t } } } ( x _ { \mathrm { t } } - \hat { y } _ { \mathrm { t } } ^ { ( 6 ) } ( y ) ) \right. _ { 0 } ^ { 2 } } \\ &  = \left. \frac { \sqrt { \hat { u } _ { \mathrm { t } } - 1 } \hat { y } _ { \mathrm { t } } } { 1 - \alpha _ { \mathrm { t } } } \left( \frac { \alpha _ { \mathrm { t } } } { \alpha _ { \mathrm { t } } } - \sqrt { \hat { u } _ { \mathrm { t } } - \hat { u } _ { \mathrm { t } } } x _ { \mathrm { t } } - \hat { y } _ { \mathrm { t } } ^ { ( 6 ) } ( y ) \right) \right. _ { 0 } ^ { 2 }  \end{array}\tag{26}
$$

Here, $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ in the third line is replaced by a function of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and $\epsilon _ { \mathrm { 0 } }$ using the reparameterization trick: ${ \pmb x } _ { t } = \sqrt { \bar { \alpha } _ { t } } { \pmb x } _ { 0 } +$ $\sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { 0 }$ , where $\epsilon _ { 0 } \sim \mathcal { N } ( 0 , \mathrm { { I } ) }$ . Inspired by the work of [24], we introduce the equivariant constraint ${ \cal L } ( \dot { \boldsymbol { \mathbf { x } } } _ { t } ^ { \prime } , \tilde { \boldsymbol { \mathbf { x } } } _ { t } ^ { \prime \prime } ) ,$ where $\tilde { \boldsymbol { \mathbf { x } } } _ { t } ^ { \mathrm { ~ } }$ represents the transformed network output and $\tilde { \mathbf { \alpha } } _ { \tilde { \mathbf { \alpha } } _ { t } }$ is the reestimation of $\tilde { \mathbf { x } } _ { t } ^ { \prime }$ . Following [24], we weight the equivariant constraint loss by $\alpha _ { E I } \mathbf { : }$

$$
\begin{array} { r l } & { { \cal L } ^ { ( t ) } = \| \displaystyle \frac { \beta _ { t } } { \sqrt { \alpha _ { t } } \big ( 1 - \bar { \alpha } _ { t } \big ) } ( { \pmb x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } { \epsilon } _ { 0 } - \sqrt { \bar { \alpha } _ { t } } f _ { \theta } ^ { ( t ) } ( { \pmb y } ) ) \| _ { 2 } ^ { 2 } } \\ & { ~ +  \alpha _ { E I } \| \tilde { \pmb x } _ { t } ^ { ' } - \tilde { \pmb x } _ { t } ^ { \prime \prime } \| _ { 2 } ^ { 2 } } \end{array}\tag{27}
$$

For simplicity, we drop the weighting coefficient of the first term, which reduces the loss function to:

$$
\begin{array} { r } { L _ { s i m p l e } ^ { ( t ) } = \left. { { \boldsymbol x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { 0 } - \sqrt { \bar { \alpha } _ { t } } f _ { \theta } ^ { ( t ) } ( { \pmb y } ) } \right. _ { 2 } ^ { 2 } } \\ { + \left. \alpha _ { E I } \left. \tilde { { \boldsymbol x } } _ { t } ^ { ' } - \tilde { { \boldsymbol x } } _ { t } ^ { \prime \prime } \right. _ { 2 } ^ { 2 } \right. } \end{array}\tag{28}
$$

In our experiments, we found that $L _ { s i m p l e } ^ { ( t ) }$ yields better sample quality than $L ^ { ( t ) }$ . A similar observation was reported in the original DDPM work [7], where removing the scaling coefficient improved overall performance.

## SUPPLEMENTARY MATERIAL: RE-SAMPLING STEP UNDER NOISY CONDITIONS

The detailed derivation of the modified re-estimation step (11) and re-sampling step (14) in the main body are shown here. In noisy HSI inpainting problem, the observation is assumed corrupted with additive white Gaussian noise with noise strength $\sigma _ { y } \mathrm { : }$

$$
{ \pmb y } = \mathrm { M } { \pmb x } + { \pmb n } , \qquad { \pmb n } \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { y } } ^ { 2 } \mathrm { I } )\tag{29}
$$

We apply the singular value decomposition to the inpainting mask M:

$$
\mathrm { M } = \mathrm { U } d i a g \{ s _ { 1 } , s _ { 2 } . . . s _ { q } \} \mathrm { V } ^ { \mathrm { T } }\tag{30}
$$

Hence, the pseudo inverse of M can be written as:

$$
\mathrm { M } ^ { \dagger } = \mathrm { V } d i a g \{ s _ { 1 } ^ { \prime } , s _ { 2 } ^ { \prime } . . . s _ { q } ^ { \prime } \} \mathrm { U } ^ { \mathrm { T } } , \quad \mathrm { s } _ { \mathrm { i } } ^ { \prime } = \left\{ \begin{array} { l l } { { 1 / s _ { i } } } & { { s _ { i } \not = 0 } } \\ { { 0 } } & { { s _ { i } = 0 } } \end{array} \right.\tag{31}
$$

Now, if we plug in (29) into the re-estimation step (11), and denote the singular value decomposition of $\Sigma _ { t }$ as: $\Sigma _ { t } = \mathrm { U } d i a g \{ \lambda _ { t 1 } , \lambda _ { t 2 } . . . \lambda _ { t q } \} \mathrm { V } ^ { \mathrm { T } } = \mathrm { U } \Sigma \mathrm { V } ^ { \mathrm { T } }$ , it follows that:

$$
\begin{array} { r l } & { \tilde { x } _ { 0 | t } = f _ { \theta } ^ { ( t ) } ( y ) - \Sigma _ { t } \mathrm { M } ^ { \dagger } ( \mathrm { M } f _ { \theta } ^ { ( t ) } ( y ) - y ) } \\ & { \qquad = f _ { \theta } ^ { ( t ) } ( y ) - \Sigma _ { t } \mathrm { M } ^ { \dagger } ( \mathrm { M } f _ { \theta } ^ { ( t ) } ( y ) - ( \mathrm { M } x + n ) ) } \\ & { \qquad = f _ { \theta } ^ { ( t ) } ( y ) - \Sigma _ { t } \mathrm { M } ^ { \dagger } ( \mathrm { M } f _ { \theta } ^ { ( t ) } ( y ) - \mathrm { M } x ) + \Sigma _ { t } \mathrm { M } ^ { \dagger } n } \\ & { \qquad = \underbrace { f _ { \theta } ^ { ( t ) } ( y ) - \Sigma _ { t } \mathrm { M } ^ { \dagger } ( \mathrm { M } f _ { \theta } ^ { ( t ) } ( y ) - \mathrm { M } x ) } _ { \xrightarrow [ n e o n e r s n o n i s e l e s ~ c a v e ~ } } \\ & { \qquad + \underbrace { \sigma _ { y } \mathrm { U } \Sigma \mathrm { M } ^ { \dagger } \Sigma _ { t } } _ { \mathrm { i n t r a f i n g i e s ~ } }  \sim \mathcal { N } ( 0 , 1 ) } \end{array}\tag{32}
$$

The noise term is introduced by the observation y, which will be further introduced into the DDPM backward pass (7) as:

$$
\begin{array} { r l } & { x _ { t - 1 } = \frac { \sqrt { \bar { \alpha } } t - 1 \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \tilde { x } _ { 0 \mid t } + \frac { \sqrt { \alpha _ { t } } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } { 1 - \bar { \alpha } _ { t } } x _ { t } } \\ & { \qquad + \sigma _ { t } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , 1 ) } \\ & { \qquad = \frac { \sqrt { \bar { \alpha } } t - 1 \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \tilde { x } _ { 0 \mid t } + \frac { \sqrt { \alpha _ { t } } \left( 1 - \bar { \alpha } _ { t - 1 } \right) } { 1 - \bar { \alpha } _ { t } } x _ { t } } \\ & { \qquad + \underbrace { \frac { \sqrt { \bar { \alpha } } t - 1 \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \sigma _ { y } \mathrm { U } \Sigma \mathrm { V } ^ { \top } \mathrm { M } ^ { \top } \epsilon } _ { d e n o t e d a s \epsilon _ { \mathrm { i n t r o d u c e d } } } + \sigma _ { t } \epsilon } \end{array}\tag{33}
$$

In the above expression, the overall noise level of (33) may exceed the pre-defined noise level in $q ( \pmb { x } _ { 1 : T } | \pmb { x } _ { 0 } )$ . In order to ensure the consistency, we construct a new noise term: $\epsilon _ { \mathrm { c o n } }$ in the update of $\mathbf { \delta x } _ { t - 1 } :$

$$
\begin{array} { l } { { \pmb x } _ { t - 1 } = \displaystyle \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \tilde { \pmb x } _ { 0 \mid t } + \displaystyle \frac { \sqrt { \alpha } _ { t } ( 1 - \bar { \alpha } _ { t - 1 } ) } { 1 - \bar { \alpha } _ { t } } \pmb x _ { t } } \\ { + \displaystyle \epsilon _ { \mathrm { i n t r o d u c e d } } + \epsilon _ { \mathrm { c o n } } } \\ { + \epsilon _ { \mathrm { c o n } } \sim \mathcal { N } ( 0 , \sigma _ { t } ^ { 2 } \mathrm { I } ) } \end{array}
$$

ϵ<sub>introduced</sub>

(34)

To design $\epsilon _ { \mathrm { { c o n } } } .$ , we start with the introduced noise term in (33), and by spotting that:

$$
\begin{array} { r l } & { \displaystyle \epsilon _ { \mathrm { i n t r o d u c e d } } = \frac { \sqrt { { { \bar { \alpha } } _ { t - 1 } } } { { \beta } _ { t } } } { 1 - { { \bar { \alpha } } _ { t } } } { \sigma } _ { y } \mathrm { U } \mathrm { { \Sigma } V } ^ { \mathrm { T } } \mathrm { M } ^ { \dag } \epsilon } \\ & { \quad \quad \quad \quad \quad = \frac { \sqrt { { { \bar { \alpha } } _ { t - 1 } } } { \beta } _ { t } } { 1 - { { \bar { \alpha } } _ { t } } } { \sigma } _ { y } \mathrm { U } \mathrm { { \Sigma } V } ^ { \mathrm { T } } } \\ & { \quad \quad \quad \quad \quad \times \left( \mathrm { V } \underbrace { d i a g \{ s _ { 1 } ^ { \prime } , s _ { 2 } ^ { \prime } . . . s _ { q } ^ { \prime } \} } _ { d e n o t e d a s \Omega } \mathrm { U } ^ { \mathrm { T } } \right) \epsilon } \end{array}\tag{35}
$$

which can be re-written as:

$$
\begin{array} { r l } & { \epsilon _ { \mathrm { i n t r o d u c e d } } \sim \mathcal { N } \left( 0 , \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \right) ^ { 2 } \sigma _ { y } ^ { 2 } \right. } \\ & { \qquad \quad \times ( \mathrm { U J } \Sigma \mathrm { V } ^ { \mathsf { T } } \mathrm { M } ^ { \dagger } ) ( ( \mathrm { M } ^ { \dagger } ) ^ { T } \mathrm { V } \Sigma \mathrm { U } ^ { T } ) ) } \\ & { \qquad \quad \sim \mathcal { N } \left( 0 , \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \right) ^ { 2 } \sigma _ { y } ^ { 2 } \right. } \\ & { \qquad \quad \left. \times ( \mathrm { U J } \Sigma \mathrm { V } ^ { \mathsf { T } } \mathrm { V O I } ^ { T } ) ( \mathrm { U } \Sigma \mathrm { V } ^ { T } \mathrm { V } \Sigma \mathrm { U } ^ { T } ) ) \right. } \\ & { \qquad \quad \sim \mathcal { N } \left( 0 , \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \right) ^ { 2 } \sigma _ { y } ^ { 2 } ( \mathrm { U } \Sigma \Omega \Omega \Sigma \mathrm { U } ^ { T } ) \right) } \end{array}
$$

Recall the definition of Σ and Ω are:

(36)

$$
\begin{array} { l l } { { \Sigma = d i a g \{ \lambda _ { t 1 } , \lambda _ { t 2 } . . . \lambda _ { t q } \} , } } \\ { { \ } } \\ { { \Omega = d i a g \{ s _ { 1 } ^ { \prime } , s _ { 2 } ^ { \prime } . . . s _ { q } ^ { \prime } \} , ~ s _ { i } ^ { \prime } = \left\{ 1 / s _ { i } ~ s _ { i } \neq 0 \right. } }  \\ { { \ } } & { { \left. s _ { i } = 0 \right. } } \end{array}\tag{37}
$$

Plug in (37) into (36) yields:

$$
\begin{array} { r l } & { \epsilon _ { \mathrm { i n t r o d u c e d } } \sim \mathcal { N } ( 0 , \mathrm { U } \Lambda _ { \mathrm { t } } \mathrm { U } ^ { T } ) , } \\ & { \Lambda _ { t i } = \left\{ \begin{array} { l l } { \frac { \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \right) ^ { 2 } \sigma _ { y } ^ { 2 } { \lambda _ { t i } } ^ { 2 } } { { s _ { i } } ^ { 2 } } } & { s _ { i } \neq 0 } \\ { 0 } & { s _ { i } = 0 } \end{array} \right. } \end{array}\tag{38}
$$

Note that $\epsilon _ { \mathrm { i n t r o d u c e d } }$ is a function of the noise level $\sigma _ { y }$ of the observation $\textbf {  { y } }$ (assumed known) and the scaling matrix $\Sigma _ { t } . \Sigma _ { t }$ is designed to guarantee that the noise variance at each update step remains bounded by $\sigma _ { t }$ (noise scheduling in the forward/diffusion process). Hence, we define $\epsilon _ { \mathrm { c o n } }$ as:

$$
\begin{array} { r l } & { \epsilon _ { \mathrm { c o n } } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { \boldsymbol { \Gamma } } _ { t } ) , } \\ & { \quad \mathbf { \boldsymbol { \Gamma } } _ { t } = \mathrm { d i a g } \{ \Gamma _ { t 1 } , \boldsymbol { \Gamma } _ { t 2 } , \dots , \boldsymbol { \Gamma } _ { t q } \} , } \\ & { \quad \boldsymbol { \Gamma } _ { t i } = \left\{ \sigma _ { t } ^ { 2 } - \frac { \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \right) ^ { 2 } \sigma _ { y } ^ { 2 } { \lambda _ { t i } } ^ { 2 } } { s _ { i } ^ { 2 } } \quad s _ { i } \neq 0 \right. } \\ & { \quad \left. \sigma _ { t } ^ { 2 } \right. \quad \quad \quad \quad \quad \quad \quad \quad s _ { i } = 0 } \end{array}\tag{39}
$$

The choice of $\Sigma _ { t }$ should satisfy that: (1) When the noise level of the current estimate is relatively big, it should be set as close to I as possible to better utilize the information in the observation y. (2) Otherwise, it should be scaled accordingly. Following the design of DDNM [13], we choose $\lambda _ { t i }$ using the following thresholding rule to limit the contribution of the observation noise $\sigma _ { y }$ and keep the total noise level consistent with the diffusion schedule:

$$
\lambda _ { t i } = \left\{ \begin{array} { l l } { 1 } & { \sigma _ { t } \geq \frac { \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \right) \sigma _ { y } } { s _ { i } } } \\ { \frac { \sigma _ { t } s _ { i } } { \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \right) \sigma _ { y } } } & { \sigma _ { t } < \frac { \left( \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { 1 - \bar { \alpha } _ { t } } \right) \sigma _ { y } } { s _ { i } } } \\ { 1 } & { s _ { i } = 0 } \end{array} \right.\tag{40}
$$

## SUPPLEMENTARY MATERIAL: OFFLINE TRAINING CAPABILITY

While the standard HyDiff-EI framework presented in the main text performs per-image optimization at test time (in a DIP-like manner), our approach is not fundamentally restricted to single-image optimization. It can be readily adapted to pretrain on a dataset of corrupted HSI measurements, functioning as an unsupervised, measurement-only trained model. This flexibility offers superiority over existing DIP-based methods, which generally do not provide pre-training or fine-tuning capabilities. To demonstrate in more detailed, we trained the EI network within our proposed diffusion solver using 300 corrupted HSI samples from the Chikusei dataset. The images were randomly corrupted using the mask types shown in Figure 3 of the main text.

![](images/7cb4ea990251e422df339ab8fd0fa181ac1a8630f504dc837653bb1971b1f2f2.jpg)  
Fig. 7. Visual comparison of inpainting results produced by the trained and un-trained HyDiff-EI models.

During training, we used a batch size of 5 and the Adam optimizer with a learning rate of 0.0001. The network was trained for 20000 epochs. In each epoch, for every mini-batch, we uniformly sampled a single random diffusion timestep $t \sim \mathcal { U } ( 0 , 1 0 0 0 )$ , computed the training loss (Eq. 28) at that specific step, and performed a single parameter update on $\dot { f } _ { \theta } ^ { ( t ) } ( \pmb { y } )$ . Following this offline training phase, the pre-trained EI network $f _ { \theta } ^ { ( t ) } ( \pmb { y } )$ was integrated back into the diffusion framework for inpainting. We refer to this trained model as HyDiff-EI (trained) to distinguish it from the original, testtime optimized HyDiff-EI (untrained) presented earlier in the main text.

Table VIII presents the quantitative inpainting results for the un-trained HyDiff-EI model alongside variants trained on 50 and 300 corrupted HSI samples, while Figure 7 illustrates their visual performance.

TABLE VIII  
INPAINTING PERFORMANCE OF THE UN-TRAINED HYDIFF-EI MODEL AND TWO VARIANTS TRAINED ON 50 AND 300 CORRUPTED HSI SAMPLES, RESPECTIVELY, EVALUATED ON A 144 × 144 × 128 CHIKUSEI SAMPLE.
<table><tr><td>Method</td><td>Input</td><td>Un-trained</td><td>Trained (50 samples)</td><td>Trained (300 samples)</td></tr><tr><td>Time (s/image)</td><td> $\sim$ </td><td>38.02</td><td>15.15</td><td>15.15</td></tr><tr><td>MPSNR (dB)↑</td><td>20.704</td><td>34.393</td><td>31.113</td><td>34.540</td></tr><tr><td>MSSIM↑</td><td>0.186</td><td>0.927</td><td>0.863</td><td>0.929</td></tr></table>

It can be seen that training on prior measurement data effectively reduces the optimization time during inference. The remaining 15.15 seconds is expected, as the model still needs to go through the whole diffusion process (1000 diffusion steps in this example). These results demonstrate that the proposed HyDiff-EI framework is not restricted solely to test-time optimization. Given sufficient (corrupted) training measurements, the model yields satisfactory results at test time. This enhanced flexibility makes it highly appealing for real-world deployment.