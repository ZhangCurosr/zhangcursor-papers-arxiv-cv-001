# Efficient Generative Image Compression with Quantized Latents Refinement

Hao Cao<sup>1</sup> Yinhuan Huang<sup>1</sup> Wenqi Guo<sup>2</sup> Zhijin Qin<sup>1,3\*</sup> Jungong Han<sup>2,4</sup>

<sup>1</sup>Department of Electronic Engineering, Tsinghua University

<sup>2</sup>Department of Automation, Tsinghua University

<sup>3</sup>State Key Laboratory of Space Network and Communications

<sup>4</sup>Beijing National Research Center for Information Science and Technology

## Abstract

Generative image compression (GIC) delivers strong perceptual quality at low bitrates. However, GIC methods with strong perceptual quality often rely on pretrained diffusion models with high latency, while recent lightweight GIC methods reduce latency but deliver lower perceptual quality. We observe that, for lightweight GIC methods at low bitrates, coarse deterministic quantization can collapse nearby encoder latents corresponding to different fine details to the same point, making these details difficult to recover. Motivated by this, we propose GR-GIC, an efficient GIC that delivers competitive reconstruction performance with fast inference. For better performance, we introduce Guided Refinement (GR), which combines latent refinement guided by entropy context with adaptive perturbationsfor training regularization, jointly improving reconstruction at low bitrates. Forfast inference, we introduce reorganizedprojected residual vector quantization (P-RVQ), which equivalently reorganizes residual updates into low-dimensional lookups to expose greater parallelism and reduce sequential computation. Experiments on Kodak show that for performance, compared with OneDC, GR-GIC achieves BD-rate savings of23.19%, 37.71%, 22.30%, and 22.14%for LPIPS, DISTS, PSNR, and FID, respectively. Forfast inference, the highly optimized implementation of GR-GIC encodes and decodes in 4.60 and 5.98 ms on an NVIDIA A100. Specifically, the reorganized P-RVQ implementation achieves a 3.14× average speedup over the original implementation under the same FP32 setting. On an NVIDIA Jetson Orin NX, GR-GIC achieves latency comparable to lightweight GIC methods.

## 1. Introduction

Generative image compression (GIC) provides strong perceptual reconstruction at low bitrates. This is challenging for traditional codecs and learned image codecs optimized primarily for pixel fidelity, which tend to smooth textures and weaken fine details as the bitrate decreases [3, 24, 28, 40]. GIC addresses this limitation with adversarial generators [31, 33], discrete generative representations [34, 42], or diffusion priors [8, 41, 46], whose learned priors can synthesize plausible details that are not fully specified by the compressed representation.

However, high-performing GIC methods often rely on pretrained diffusion models, whose large parameter counts and FLOPs hinder practical deployment [18, 36, 41, 46]. To address this issue, recent work has explored lightweight GIC methods with fewer parameters and faster inference [6, 7, 22], but their reconstruction performance still leaves room for improvement compared with diffusion-based methods. This leaves a trade-off between reconstruction performance and fast inference.

We observe that, in lightweight GIC methods, images with similar coarse content but different fine details can be mapped to nearby points in the continuous latent space. At low bitrates, coarse deterministic quantization can collapse these distinct latents to the same quantized point, making their fine detail differences unavailable from the quantized representation, as illustrated in Figures 1a and 1b. By contrast, diffusion based methods introduce diversity through Gaussian noise and use strong generative priors to synthesize plausible details, providing a natural mechanism to compensate for this uncertainty.

Motivated by this observation, we propose GR-GIC, an efficient GIC that combines competitive reconstruction performance with fast inference, as shown in Figure 1d. For better performance, we introduce Guided Refinement (GR), which uses entropy context to parameterize a Gaussian refinement distribution around the recovered latent. Its mean shifts the recovered latent toward a refined center, while its variance controls adaptive perturbations for training regularization, jointly improving reconstruction at low bitrates. For fast inference, we target projected residual vector quantization (P-RVQ), which is a major encoder latency bottleneck in EF-LIC [6]. We derive an algebraically equivalent reorganization that replaces serial high dimensional residual updates with low dimensional lookups, exposing greater parallelism and reducing sequential computation.

![](images/821302142fec9af8febbb73a0154920bba46cea31d9454a74aaf646031e68c0c.jpg)  
Figure 1. Motivation and overview of GR-GIC. (a) Images with similar coarse content but different fine details are mapped to nearby points in the continuous latent space. (b) At low bitrates, coarse deterministic quantization can collapse these distinct latents to the same quantized point, leaving their fine detail differences unspecified. (c) GR uses entropy context to define a Gaussian refinement distribution around the recovered latent and introduces adaptive perturbations during training. (d) GR-GIC combines competitive reconstruction performance with fast inference.

Experiments on Kodak show that (i) for performance, compared with OneDC [41], GR-GIC achieves BD-rate [5] savings of 23.19%, 37.71%, 22.30%, and 22.14% for LPIPS [45], DISTS [11], PSNR, and FID [19], respectively. (ii) For fast inference, the highly optimized implementation of GR-GIC encodes and decodes in 4.60 and 5.98 ms on an NVIDIA A100, achieving 16.73× and 34.19× speedups over the official OneDC implementation, respectively. Specifically, the reorganized P-RVQ implementation achieves a 3.14× average speedup over the original implementation under the same FP32 setting. We further validate GR-GIC on an NVIDIA Jetson Orin NX, where it achieves latency comparable to existing lightweight GIC methods, supporting its deployment on edge devices.

## Our contributions are summarized as follows:

• We introduce GR-GIC, an efficient GIC that combines GR and reorganized P-RVQ for competitive reconstruction performance and fast inference.

• GR uses entropy context to parameterize an adaptive refinement distribution around quantized latents. Reorganized P-RVQ replaces serial high dimensional residual updates with low dimensional lookups, reducing sequential computation and inference latency.

• For performance, experiments demonstrate competitive results across multiple reconstruction metrics. For fast inference, the highly optimized implementation achieves low latency on both GPU and edge platforms.

## 2. Related Work

Diffusion-based generative image compression. Diffusion models provide strong generative priors for reconstructing details that are not fully specified by compressed representations [8, 43]. Recent GIC methods increasingly use pretrained diffusion models and improve their efficiency through fewer diffusion steps, one-step reconstruction, or lighter architectures [18, 22, 29, 41, 46]. Despite these advances, their model size and computation remain substantial for efficient deployment.

VAE-based generative image compression. VAE-based GIC provides a more efficient alternative through adversarial priors, improved latent representations, and vector quantization [20, 23, 26, 31, 33, 42]. Recent work further develops lightweight GIC. ProGIC combines a depthwise convolutional backbone with RVQ [7], while EF-LIC models correlations among latents in the representation domain [6]. GR-GIC builds on EF-LIC, introducing GR for better reconstruction and reorganizing P-RVQ to move high dimensional reconstruction outside the sequential search path.

Latent refinement after quantization. Prior LIC methods have compensated quantization loss through latent residual prediction, post-quantization filtering, and adaptive quantization or correction [15, 16, 32, 48]. Recent work also studies latent and noise compensation [44]. These methods mainly perform deterministic correction using decoder context, rely on guidance from the encoder for content-adaptive compensation, or model stochastic compensation without adapting it to context. In contrast, GR uses entropy context already available at the decoder to parameterize a refinement distribution that adapts to spatial location and image content, without additional payload, while adaptive perturbations provide training regularization for generative reconstruction.

## 3. Method

## 3.1. Overview

As illustrated in Figures 1a and 1b, at low bitrates, coarse deterministic quantization can collapse nearby encoder latents corresponding to different fine details to the same quantized point, making these details difficult to recover in lightweight GIC. GR-GIC builds on EF-LIC [6], whose context model provides the recovered latent and entropy context without additional payload. GR uses the entropy context to parameterize a Gaussian refinement distribution around the recovered latent, with adaptive perturbations providing latent noise regularization during training. GR-GIC retains the analysis transform, hyperprior, groupwise context, and P-RVQ structure of EF-LIC, adds a GR head before synthesis as shown in Figure 2, and reorganizes P-RVQ execution in Figure 3.

Given an image x $\in \mathbb { R } ^ { 3 \times H \times W }$ , the encoder $g _ { a }$ produces the latent y. The hyper encoder $h _ { a }$ maps $\textbf {  { y } }$ to the hyper latent z, which the hyper latent RVQ $Q _ { z }$ quantizes to obtain zˆ. The hyper decoder $h _ { s }$ reconstructs the context feature $\phi$ from zˆ:

$$
\begin{array} { l l } { { { \pmb y } = g _ { a } ( { \pmb x } ) , } } & { { { \pmb z } = h _ { a } ( { \pmb y } ) , } } \\ { { { \hat { \pmb z } } = Q _ { z } ( { \pmb z } ) , } } & { { \phi = h _ { s } ( { \hat { z } } ) . } } \end{array}
$$

The latent y has a spatial downsampling factor of $f _ { y } = 1 6$ and enters the groupwise affine projections. Following the quadtree context modeling of EF-LIC and DCVC-RT [6, 21], we partition y into $N = 4$ groups $\{ y _ { n } \} _ { n = 1 } ^ { N }$ . For group n, the context model $f _ { n }$ combines ϕ with the previously recovered groups to predict the mean $\pmb { \mu } _ { n }$ and scale $\sigma _ { n } \mathrm { : }$

$$
\begin{array} { c } { { ( \pmb { \mu } _ { n } , \pmb { \sigma } _ { n } ) = f _ { n } \big ( \mathrm { c o n c a t } ( \pmb { \phi } , \hat { \pmb { y } } _ { < n } ) \big ) , } } \\ { { \pmb { \sigma } _ { n } \geq \sigma _ { \operatorname* { m i n } } . } } \end{array}
$$

Before quantization, the affine projection (AP) centers and scales ${ \bf { { y } } } _ { n }$ using $\pmb { \mu } _ { n }$ and $\sigma _ { n }$ . The inverse AP then reconstructs ${ \hat { \mathbf { y } } } _ { n }$ after quantization:

$$
\begin{array} { r l } & { { \pmb y } _ { n } ^ { \prime } = ( { \pmb y } _ { n } - { \pmb \mu } _ { n } ) \oslash { \pmb \sigma } _ { n } , } \\ & { \hat { \pmb y } _ { n } ^ { \prime } = Q _ { n } ( { \pmb y } _ { n } ^ { \prime } ) , } \\ & { \hat { \pmb y } _ { n } = { \pmb \sigma } _ { n } \odot \hat { \pmb y } _ { n } ^ { \prime } + { \pmb \mu } _ { n } . } \end{array}\tag{1}
$$

After all groups are recovered, their concatenation forms $\hat { \pmb { y } } .$ The recovered latent yˆ and the aligned context parameters $( \mu , \sigma )$ are then fed to the GR head $r _ { \theta }$ to produce y˜, which the decoder $g _ { s }$ maps to the reconstruction $\hat { \pmb x } = g _ { s } ( \tilde { \pmb y } )$ .

The codec uses five RVQs, with $Q _ { z }$ quantizing the hyper latent z and $Q _ { n }$ quantizing each of the $N = 4$ main latent groups. At a given operating point, all five RVQs activate the same number $M \in \{ 1 , 2 , 3 , 4 \}$ of codebooks, and $m \in$ $\{ 1 , \ldots , M \}$ indexes an active stage. With this shared RVQ depth and fixed width indices, the bit rate is $R _ { M } = 0 . 0 3 5 6 M$ bits per pixel (BPP). Secs. B.3 and B.4 provide the exact rate calculation and boundary handling.

## 3.2. Guided Refinement of Quantized Latents

To improve low-bitrate reconstruction in lightweight GIC, we introduce Guided Refinement (GR) after RVQ recovery. GR uses entropy context to define a Gaussian refinement distribution around the recovered latent that adapts to spatial location and image content. During joint training, sampling from this distribution introduces adaptive perturbations that provide latent noise regularization.

We first formalize the effect of deterministic quantization that motivates this refinement. To express the mapping with the notation of Equation (1), we suppress group and stage indices, write $Q$ for the deterministic quantizer, and denote its jth reconstruction point by $\hat { \mathbf { \boldsymbol { y } } } _ { j } ^ { \prime }$ . Its quantization cell is $\mathcal { C } _ { j } = \{ { \pmb y } ^ { \prime } : Q ( { \pmb y } ^ { \prime } ) = \hat { \pmb y } _ { j } ^ { \prime } \}$ , and inverse projection gives $\hat { \pmb { y } } _ { j } = \pmb { \sigma } \odot \hat { \pmb { y } } _ { j } ^ { \prime } + \pmb { \mu }$ . The continuous latent distribution is transformed as

$$
\begin{array} { r l } & { \begin{array} { r l } & { p _ { \hat { Y } } ( \hat { y } \mid \mu , \sigma ) } \\ & { \quad = \displaystyle \int p _ { Y ^ { \prime } } ( y ^ { \prime } \mid \mu , \sigma ) } \\ & { \qquad \quad \times \ : p _ { \hat { Y } } ( \hat { y } \mid Y ^ { \prime } = y ^ { \prime } , \mu , \sigma ) \mathrm { d } y ^ { \prime } } \end{array} } \\ & { \begin{array} { r l } & { \displaystyle = \sum _ { j } \left( \int _ { \mathcal { C } _ { j } } p _ { Y ^ { \prime } } ( y ^ { \prime } \mid \mu , \sigma ) \mathrm { d } y ^ { \prime } \right) } \\ & { \qquad \quad \times \ : \delta _ { \mathrm { D } } ( \hat { y } - \hat { y } _ { j } ) . } \end{array} } \end{array}\tag{2}
$$

Here $ { \delta _ { \mathrm { D } } }$ is the Dirac delta. Equation (2) shows that deterministic quantization assigns all latent realizations within one quantization cell to the same recovered point. The transmitted index therefore identifies the quantization cell, but does not preserve the continuous variation within that cell. At low bitrates, where the quantization cells are relatively coarse, this unresolved variation can correspond to different fine details that are difficult to determine from the recovered latent alone. This motivates learning a refinement distribution around the recovered point using information already provided by the entropy context.

We therefore use a Gaussian family conditioned on entropy context, as its simple mean-variance form separates the refinement center from adaptive perturbations without introducing additional latent correlations. Concatenating the groupwise reconstruction points in Equation (2) gives the recovered latent yˆ. Let $\epsilon \sim \mathcal { N } ( 0 , I )$ denote a standard Gaussian variable for one latent realization. We model the refined latent distribution as

$$
\begin{array} { l } { { \displaystyle p _ { \theta } \big ( \pmb { \tilde { y } } \mid \hat { \pmb { y } } , \pmb { \mu } , \pmb { \sigma } \big ) } } \\ { { \displaystyle ~ = \int p _ { \epsilon } \big ( \epsilon \big ) p _ { \theta } \big ( \tilde { \pmb { y } } \mid \epsilon , \hat { \pmb { y } } , \pmb { \mu } , \pmb { \sigma } \big ) \mathrm { d } \epsilon } } \\ { { \displaystyle ~ = \mathcal { N } \big ( \tilde { \pmb { y } } \mid \hat { \pmb { y } } + \delta , \mathrm { d i a g } ( \beta ^ { 2 } ) \big ) . } } \end{array}\tag{3}
$$

![](images/2ee1859410d2ff6e71c83d26274069cd2d2eda412c6e8edf686c938876e26039.jpg)  
Figure 2. Overview of GR-GIC. The encoder $g _ { a }$ produces the latent $^ { \mathbf { \delta } } \mathbf { \delta } _ { \mathbf { \delta } }$ while the hyper encoder $h _ { a }$ and hyper decoder $h _ { s }$ provide the context feature $\phi .$ . The context model predicts $( \mu , \sigma )$ for the groupwise affine projections (APs) and P-RVQ. The GR head refines the recovered latent yˆ into y˜ before the decoder $g _ { s }$ reconstructs xˆ.

Here δ shifts the recovered latent to the refinement center $\hat { y } + \delta$ , while $\beta$ controls the perturbation scale across spatial locations and image content. Both quantities are predicted from the recovered latent and entropy context parameters that are already available after decoding:

$$
\begin{array} { c } { { ( \delta , \tilde { \beta } ) = r _ { \theta } ( \mathrm { c o n c a t } ( \hat { \pmb y } , { \pmb \mu } , { \pmb \sigma } ) ) , } } \\ { { \beta = \mathrm { m a x } ( \tilde { \beta } , { \bf 0 } ) . } } \end{array}\tag{4}
$$

The refined latent is then sampled as

$$
\epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) , \qquad \tilde { y } = \hat { y } + \delta + \beta \odot \epsilon .\tag{5}
$$

All codec components are jointly optimized with one independent Gaussian draw per crop. The learned refinement center adjusts the recovered latent according to the entropy context, while the adaptive perturbation provides latent noise regularization during training. Section 4.3 further supports the important role of $\beta \odot \epsilon$ during training.

## 3.3. Reorganized Projected Residual Vector Quantization

EF-LIC [6] reported that P-RVQ takes 9.16 ms on Kodak, accounting for 52.0% of the measured 17.62 ms encoder latency and forming a major encoder latency bottleneck. For fast inference, we reorganize P-RVQ execution while preserving its greedy quantization process.

Within this subsection, $z \in \mathbb { R } ^ { D }$ denotes an abstract input to a hyper or main latent P-RVQ. At stage m, $\boldsymbol { A } _ { m } \in \mathbb { R } ^ { d \times D }$ projects the residual from high to low dimensions for search. The codebook $\pmb { { \cal E } } _ { m } = [ \pmb { { \cal E } } _ { m } [ \bar { 1 } ] , \dots , \pmb { { \cal E } } _ { m } [ K ] ] \in \mathbb { R } ^ { d \times K }$ stores K low dimensional codewords as columns, and $B _ { m } \in$ $\mathbb { R } ^ { D \times d }$ maps them back to the reconstruction space, with projection biases suppressed. Starting from $\hat { z } _ { 0 } = \mathbf { 0 }$ , the partial reconstruction is defined for $m = 1 , \ldots , M$ and returns $\hat { z } = \hat { z } _ { M }$ . As illustrated in Figure 3a, the original recursion carries a state in $\mathbb { R } ^ { D }$ , projects the residual into $\mathbb { R } ^ { d }$ for search, and maps the selected codeword back to $\mathbb { R } ^ { D }$ before the next stage, so both projection directions remain on the critical

dependency path:

$$
\hat { z } _ { m } = \underbrace { \sum _ { m = 1 } ^ { \mathrm { O R I G I N A L : } } { \hat { z } _ { m - 1 } + B _ { m } } E _ { m } \Big [ } _ { \mathrm { \underbrace { \mathrm { \normalfont ~ \alpha ~ } } } _ { \mathrm { \normalfont ~ c r i d i c a l ~ d e p e n d e n c y ~ l e i n i t h ~ i s f ~ \hat { \alpha } ~ } } } .\tag{6}
$$

We derive an algebraically equivalent reorganization that stacks the source projections and expresses subsequent residual updates as low dimensional lookups, as illustrated in Figure 3b. Since the projection matrices and codebooks are fixed after training, each product $A _ { m } B _ { \ell } E _ { \ell } \in \mathbb { R } ^ { d \times K }$ can be fused offline into a new d dimensional codebook. Here, $i _ { \ell }$ denotes the codeword index selected at an earlier stage $\ell :$

$$
\begin{array} { r l } { \displaystyle \mathrm { R E O R G A N I Z E D : } } & { } \\ { \displaystyle \hat { z } _ { m } = \hat { z } _ { m - 1 } + B _ { m } { E _ { m } } \Bigg [ } \\ { \displaystyle \operatorname* { a r g } _ { 1 \leq j \leq K } \| A _ { m } z - \sum _ { \underbrace { \ell < m } _ { \ell < m } } A _ { m } B _ { \ell } { E _ { \ell } [ i _ { \ell } ] } - E _ { m } [ j ] \| _ { 2 } ^ { 2 } ] . } \\ { \displaystyle \operatorname* { m a x } _ { \ell \mathrm { i n c e l l d } \mathrm { e p n i d e p n i t } \mathrm { e x i s h i g h } \mathrm { e x e } } } & { } \\ { \displaystyle  \ell \mathrm { d i m e n s i o n } 1 \mathrm { c e l l d } \mathrm { e p n o l i n s i g n e } \| } \\ { \displaystyle \mathrm { d i m e n s i o n a l ~ c o d e l p o s i s } \mathrm { f o l i l i e s } } \end{array}\tag{7}
$$

The reorganized search evaluates all source projections together and updates later search states through lookups in $\mathbb { R } ^ { d }$ . The high dimensional reconstruction is gathered from offline tables only after the codeword indices are selected. By linearity, the original and reorganized forms preserve the same search states, greedy assignments, and final reconstruction in real arithmetic. The reorganization therefore moves high dimensional reconstruction outside the critical dependency path, exposing greater parallelism and shortening the sequential computation path.

Under the same Torch FP32 backend and precision, the reorganized P-RVQ implementation achieves a 3.14× average speedup over the original implementation across the 16 evaluated stage count and resolution settings. This matched comparison isolates the gain from the algebraic reorganization, which accounts for 23.54% of the total encoder latency reduction. The CUDA FP16 implementation further increases the average speedup to 7.15× relative to the original FP32 implementation. At the highest evaluated Kodak BPP, compression-only P-RVQ reaches an 18.80× speedup, and the complete P-RVQ acceleration accounts for 31.28% of the total encoder latency reduction.

![](images/7c892c977f2415edbaf81ff0df325777fc8c1cc44f8aa25fe9df3853fe29466c.jpg)  
Figure 3. Reorganized P-RVQ. (a) Original P-RVQ alternates high dimensional residual updates, projections, and greedy search. (b) The equivalent form stacks source projections and uses precomputed low dimensional lookups. Red arrows mark the critical dependency path.

## 4. Experiments

## 4.1. Experimental Setup

Implementation Details. Our implementation largely follows EF-LIC [6], with a GR head added before synthesis and P-RVQ execution reorganized as described above. Section B provides the complete architecture and codebook settings. For fast inference, the highly optimized implementation further uses fused CUDA operators following DCVC-UF [27].

Training Details. Each operating point is trained on ImageNet [9] in two 1M-iteration stages. Stage one samples 1% per epoch with $2 5 6 \times 2 5 6$ random crops at a learning rate of $1 0 ^ { - 4 } .$ . Stage two samples 10% of images exceeding 512 pixels in both dimensions with 512×512 random crops at $1 0 ^ { - 5 }$ . Both stages apply random horizontal flipping and use AdamW [30] with weight decay $1 0 ^ { - 4 }$ $( \beta _ { 1 } , \beta _ { 2 } ) \ : = \ : ( 0 . 9 , 0 . 9 9 9 )$ , batch size 16, and one NVIDIA A100. Each training crop uses one standard Gaussian draw $\epsilon \sim \mathcal { N } ( 0 , I )$ to sample the refinement distribution conditioned on entropy context and estimate the expected imagespace objective, which includes a generative adversarial network (GAN) term:

$$
\begin{array} { r l } & { \mathcal { L } = \mathbb { E } _ { \epsilon } \big [ \lambda _ { \mathrm { r e c } } \mathcal { L } _ { 2 } + \lambda _ { \mathrm { p e r } } \mathcal { L } _ { \mathrm { L P I P S } } + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { G A N } } \big ] } \\ & { \quad \quad + \lambda _ { \mathrm { c b } } \mathcal { L } _ { \mathrm { c b } } + \lambda _ { \mathrm { c o m } } \mathcal { L } _ { \mathrm { c o m } } . } \end{array}\tag{8}
$$

Here $\mathcal { L } _ { 2 } = \| \pmb { x } - \hat { \pmb { x } } \| _ { 2 } ^ { 2 }$ is averaged over pixels and channels, $\mathcal { L } _ { \mathrm { G A N } }$ is the generator-side PatchGAN term, and $\mathcal { L } _ { \mathrm { c b } }$ and $\mathcal { L } _ { \mathrm { c o m } }$ are the codebook and commitment terms. We use $\lambda _ { \mathrm { { r e c } } } = 1 8 , \lambda _ { \mathrm { { p e r } } } = 2 , \lambda _ { \mathrm { { c b } } } = 1$ , and $\lambda _ { \mathrm { c o m } } = 1$ , with adaptive $\lambda _ { \mathrm { a d v } }$ following VQGAN [13]. All configurations in the study share the PatchGAN rule, training data, and losses. For the final codec, the adaptive perturbation provides latent noise regularization with all codec components jointly optimized.

Evaluation Datasets and Metrics. We evaluate Kodak, Tecnick, DIV2K validation, and CLIC 2020 Professional [1, 2, 12, 39]. Metrics are LPIPS with a Visual Geometry Group (VGG) backbone, DISTS, PSNR, and patch-based FID [11, 19, 20, 37, 45]. LPIPS and DISTS capture perceptual similarity, PSNR reports RGB pixel fidelity, and FID measures distribution-level alignment. All codecs use the same metric pipeline on decoded images for direct curves and pairwise BD-rate calculations. Appendix C gives dataset, metric, and pairwise PCHIP details [5].

Compared Methods. The pretrained diffusion group comprises ResULIC, StableCodec, RDEIC, OneDC, OSCAR, DiT-IC, AEIC, and CoD-Lite [18, 22, 25, 29, 36, 41, 46, 47]. The VAE-based group comprises MS-ILLM, Control-GIC, GLC, DLF, ProGIC, RDVQ, EF-LIC, and GR-GIC [6, 7, 20, 23, 26, 33, 42]. Runtime uses batch size one with 512×768 images on one NVIDIA A100. Latency follows the implementation and timing protocols in Sec. C.2, with parameters and forward FLOPs reported as additional model complexity measures. CoD-Lite uses our measurements of its official code, with its paper profile in Sec. C.3.

## 4.2. Comparison with Previous Methods

Figure 4 and Tab. 1 compare the reconstruction performance and inference latency of GR-GIC with previous GIC methods. For performance, GR-GIC achieves bitrate savings over OneDC for LPIPS, DISTS, and FID across all four datasets. On Kodak, the OneDC-anchored BD-rate savings are 23.19%, 37.71%, 22.30%, and 22.14% for LPIPS,

![](images/8a63cf5394609646803679d9510fe6e09184262e417b3a8930a601f0d1fea38f.jpg)  
Figure 4. Rate and reconstruction quality across four datasets. Lower LPIPS, DISTS, and FID and higher PSNR indicate better quality. FID uses base-2 logarithmic vertical axes.

## DISTS, PSNR, and FID, respectively.

For fast inference, the highly optimized implementation of GR-GIC encodes and decodes a 512 × 768 image in 4.60 and 5.98 ms on an NVIDIA A100. The official OneDC implementation takes 76.98 and 204.47 ms under the evaluated system settings, making the corresponding latency ratios 16.73× and 34.19×. Section C.2 details the numerical precision, acceleration paths, and timing boundaries used for the compared systems.

Among the evaluated VAE-based methods, GR-GIC improves the rate-quality curves of previous lightweight codecs across the reported reconstruction metrics while retaining low inference latency. Taken together, these results show that GR-GIC combines gains across multiple reconstruction metrics with fast inference.

At similar rates, the qualitative comparisons in Figure 5 further illustrate the reconstruction differences. In the first row, GR-GIC preserves the thin eyeglass frame and nose pad. In the second row, it reconstructs the small circular fixture with coherent local geometry. These examples are consistent with the quantitative gains and show clearer reconstruction of local structures.

Table 1. Results on the Kodak dataset, with latency measured on one NVIDIA A100 GPU. Encoding and decoding latencies are in milliseconds (ms), FLOPs in billions (G), parameters in millions (M), and BD-rate in percent. OneDC is the BD-rate anchor, and lowe values are better for every reported measure. Bold and underlined values indicate the best and second best results, respectively.
<table><tr><td rowspan="2">Family</td><td rowspan="2">Method</td><td colspan="4">Complexity</td><td colspan="4">BD-rate (%)</td></tr><tr><td>Enc.</td><td>Dec.</td><td>FLOPs</td><td>Params</td><td>LPIPS</td><td>DISTS</td><td>PSNR</td><td>FID</td></tr><tr><td rowspan="8">Pretrained diffusion</td><td>ResULIC</td><td>96.24</td><td>428.32</td><td>8549.01</td><td>1390.00</td><td>206.40</td><td>260.74</td><td>10.77</td><td>304.92</td></tr><tr><td>StableCodec</td><td>100.18</td><td>200.03</td><td>6820.28</td><td>1065.81</td><td>8.03</td><td>6.77</td><td>-17.10</td><td>-4.45</td></tr><tr><td>RDEIC</td><td>157.25</td><td>426.68</td><td>7767.46</td><td>1380.27</td><td>69.79</td><td>166.61</td><td>-12.14</td><td>80.26</td></tr><tr><td>OneDC</td><td>76.98</td><td>204.47</td><td>8557.18</td><td>1406.42</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>OSCAR</td><td>53.04</td><td>167.56</td><td>6485.61</td><td>1009.30</td><td>127.15</td><td>131.50</td><td>154.09</td><td>159.92</td></tr><tr><td>DiT-IC</td><td>97.62</td><td>151.89</td><td>5743.15</td><td>1069.41</td><td>35.99</td><td>-6.44</td><td>33.24</td><td>56.47</td></tr><tr><td>AEIC</td><td>12.39</td><td>82.98</td><td>2143.06</td><td>885.70</td><td>41.65</td><td>-16.71</td><td>2.11</td><td>7.16</td></tr><tr><td>CoD-Lite</td><td>9.33</td><td>20.86</td><td>771.69</td><td>80.15</td><td>44.53</td><td>51.35</td><td>18.72</td><td>53.06</td></tr><tr><td rowspan="8">VAE-based</td><td>MS-ILLM</td><td>165.38</td><td>147.79</td><td>599.52</td><td>181.40</td><td>293.59</td><td>364.46</td><td>-47.46</td><td>468.10</td></tr><tr><td>Control-GIC</td><td>103.56</td><td>436.26</td><td>5816.37</td><td>130.36</td><td>602.72</td><td>807.48</td><td>175.78</td><td>855.14</td></tr><tr><td>GLC</td><td>34.75</td><td>48.26</td><td>2466.28</td><td>163.99</td><td>21.53</td><td>54.70</td><td>56.04</td><td>82.95</td></tr><tr><td>DLF</td><td>168.78</td><td>240.63</td><td>5880.53</td><td>1286.01</td><td>20.87</td><td>20.19</td><td>42.73</td><td>70.80</td></tr><tr><td>ProGIC</td><td>7.64</td><td>10.99</td><td>333.38</td><td>33.11</td><td>36.47</td><td>80.41</td><td>49.84</td><td>156.32</td></tr><tr><td>RDVQ</td><td>408.24</td><td>412.44</td><td>3090.29</td><td>251.84</td><td>17.06</td><td>9.87</td><td>62.13</td><td>49.51</td></tr><tr><td>EF-LIC</td><td>17.62</td><td>13.72</td><td>279.61</td><td>35.74</td><td>-3.33</td><td>10.05</td><td>11.49</td><td>49.01</td></tr><tr><td>GR-GIC</td><td>4.60</td><td>5.98</td><td>456.55</td><td>79.39</td><td>-23.19</td><td>-37.71</td><td>-22.30</td><td>-22.14</td></tr></table>

![](images/834617725f9676ccd1295d90e8f0431d10f0447dc0ec4d941bb257cb19eae724.jpg)  
Figure 5. Qualitative comparisons of GR-GIC with high-capacity diffusion and lightweight codecs at similar rates. Boxes mark local details. Labels report LPIPS and BPP, where lower is better.

## 4.3. Guided Refinement Configuration Study

We compare five refinement configurations, each trained from scratch for 1M iterations at each rate under identical optimization conditions. $\mathrm { M } _ { 0 }$ bypasses refinement. $\mathrm { M } _ { 1 }$ predicts an offset from the recovered latent, while $\mathrm { M } _ { 2 }$ additionally uses entropy context. $\mathrm { M _ { 3 } }$ introduces stochastic perturbation with a channel-wise scale, and $\mathrm { M } _ { 4 }$ further adapts this scale to spatial location and image content. Both $\mathrm { M } _ { 2 }$ and $\mathrm { M _ { 3 } }$ match $\mathrm { M } _ { 4 }$ in parameter count and forward FLOPs, providing matched baselines that control for gains from increased model size or computation.

In Tab. 2, $\mathrm { { M } _ { 0 } }$ removes GR, retains the direct synthesis path of the EF-LIC backbone, and shows performance close to EF-LIC. Because $\mathrm { { M } _ { 0 } }$ and $\mathrm { M _ { 1 } { - } M _ { 4 } }$ are trained under the same conditions, $\mathrm { M } _ { 0 }$ also provides a direct reference for comparing GR with the EF-LIC backbone. $\mathrm { M } _ { 4 }$ reduces FID by 4.2 to 7.8 relative to $\mathrm { M } _ { 2 }$ and by 1.5 to 4.9 relative to $\mathrm { M _ { 3 } }$ , while achieving lower LPIPS at every evaluated rate. Since $\mathrm { M _ { 2 } }$ and $\mathrm { M _ { 3 } }$ match M<sub>4</sub> in parameter count and forward FLOPs, these results show that the complete GR configuration improves reconstruction performance on the lightweight EF-LIC backbone. We select $\mathrm { M } _ { 4 }$ for GR.

Table 2. Guided Refinement configurations on Kodak at four nominal rates. The GR columns indicate a learned offset, decoder context, stochastic scale, and adaptive scale.
<table><tr><td rowspan="2">Model</td><td colspan="4">GR configuration</td><td colspan="2"> $\mathrm { B P P = 0 . 0 3 6 }$ </td><td colspan="2"> $\mathrm { B P P = 0 . 0 7 1 }$ </td><td colspan="2"> $\mathrm { B P P = 0 . 1 0 7 }$ </td><td colspan="2"> $\mathrm { B P P = 0 . 1 4 3 }$ </td></tr><tr><td>Offset</td><td>Ctx.</td><td>Stoch.</td><td>Adapt.</td><td>LPIPS</td><td>FID</td><td>LPIPS</td><td>FID</td><td>LPIPS</td><td>FID</td><td>LPIPS</td><td>FID</td></tr><tr><td> $\mathrm { M } _ { 0 }$ </td><td>x</td><td>x</td><td>x</td><td>x</td><td>0.277</td><td>66.2</td><td>0.214</td><td>51.2</td><td>0.184</td><td>43.6</td><td>0.165</td><td>38.1</td></tr><tr><td> $\mathrm { M } _ { 1 }$ </td><td>√</td><td>X</td><td>x</td><td>X</td><td>0.271</td><td>63.6</td><td>0.212</td><td>46.8</td><td>0.184</td><td>39.2</td><td>0.164</td><td>35.4</td></tr><tr><td> $\mathrm { M _ { 2 } }$ </td><td>√</td><td>√</td><td>x</td><td>x</td><td>0.264</td><td>62.4</td><td>0.208</td><td>41.2</td><td>0.182</td><td>36.5</td><td>0.162</td><td>31.9</td></tr><tr><td> $\mathrm { M _ { 3 } }$ </td><td>√</td><td>√</td><td>√</td><td>X</td><td>0.264</td><td>59.5</td><td>0.207</td><td>37.9</td><td>0.182</td><td>33.8</td><td>0.162</td><td>29.5</td></tr><tr><td> $\mathrm { M } _ { 4 }$ </td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.261</td><td>54.6</td><td>0.206</td><td>36.0</td><td>0.180</td><td>32.3</td><td>0.160</td><td>27.6</td></tr></table>

Table 3. Encoding (Enc.) and decoding (Dec.) latency (ms) on an NVIDIA Jetson Orin NX. Bold and underline mark the two best reported values in each column.
<table><tr><td rowspan="2">Method</td><td colspan="2"> $5 1 2 \times 7 6 8$ </td><td colspan="2">1080p</td><td colspan="2">2K</td><td colspan="2">4K</td></tr><tr><td>Enc.</td><td>Dec.</td><td>Enc.</td><td>Dec.</td><td>Enc.</td><td>Dec.</td><td>Enc.</td><td>Dec.</td></tr><tr><td>EF-LIC</td><td>71.253</td><td>143.040</td><td>347.490</td><td>633.721</td><td>623.651</td><td>1137.985</td><td>1441.824</td><td>2628.537</td></tr><tr><td>ProGIC</td><td>83.204</td><td>164.517</td><td>413.956</td><td>781.443</td><td>748.899</td><td>1412.819</td><td>1670.957</td><td>3182.561</td></tr><tr><td>AEIC</td><td>96.784</td><td>1290.483</td><td></td><td>&gt; 10 s</td><td></td><td> $> 1 0 \ : s$ </td><td>OOM</td><td></td></tr><tr><td>CoD-Lite</td><td>114.456</td><td>239.371</td><td>564.422</td><td>1348.999</td><td>974.139 2489.412</td><td></td><td></td><td>&gt; 10 s</td></tr><tr><td>OneDC</td><td>773.388</td><td>1981.831</td><td>OOM</td><td></td><td>OOM</td><td></td><td>OOM</td><td></td></tr><tr><td>GR-GIC</td><td>90.342</td><td>99.850</td><td>265.745</td><td>428.361</td><td>465.576</td><td>754.362</td><td>1061.516</td><td>1739.863</td></tr></table>

To further understand the role of the perturbation, we set only the sampled term $\beta \odot \epsilon$ to zero during $\mathrm { M } _ { 4 }$ inference and observe only a small performance change. Together with the gains over deterministic $\mathrm { M _ { 2 } }$ , this result supports an important role of adaptive perturbation as latent noise regularization during training. The gains over $\mathrm { M _ { 3 } }$ further show the benefit of adapting the perturbation scale to spatial location and image content. Section C.4 specifies the configuration training protocol, and Sec. D.7 reports the inference perturbation results.

## 4.4. Edge-Device Deployment

We evaluate GR-GIC on a 16 GB NVIDIA Jetson Orin NX. Among the compared methods, GR-GIC achieves the lowest decoding latency at all four resolutions and the lowest encoding latency from 1080p to 4K in Tab. 3. Measurements use MAXN SUPER mode, batch size one, and synchronized wall-clock timing. GR-GIC remains deployable at 4K, where OneDC and AEIC run out of memory and CoD-Lite decoding exceeds 10 seconds. These results demonstrate effective low-latency deployment on edge hardware.

## 5. Limitations

Our evaluation spans 0.0356 to 0.1426 BPP, with the lowest point below the 0.05 BPP ultra-low bitrate threshold [46, 47]. Extending lightweight GIC to even lower bitrates requires greater decoder generative capacity as less fine detail is transmitted. Future work can strengthen lightweight priors and synthesis capacity while retaining fast inference. Our latency results characterize the complete deployed codec, including P-RVQ reorganization and CUDA acceleration. We do not claim the fastest backbone-only FP32 inference under identical backends. Comparisons retain official or released implementations and available acceleration paths, reflecting system-level performance.

## 6. Conclusion

We introduced GR-GIC, an efficient GIC combining GR and reorganized P-RVQ for competitive reconstruction performance and fast inference. GR refines quantized latents using entropy context, while adaptive perturbations provide latent noise regularization during training. Reorganized P-RVQ replaces serial high dimensional residual updates with low dimensional lookups to reduce sequential computation. Experiments across four datasets show consistent gains in LPIPS, DISTS, and FID over OneDC, together with fast inference on both an NVIDIA A100 and Jetson Orin NX.

## References

[1] Eirikur Agustsson and Radu Timofte. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 126–135, 2017.

[2] Nicola Asuni and Andrea Giachetti. TESTIMAGES: a Largescale Archive for Testing Visual Devices and Basic Image Processing Algorithms. In Smart Tools and Apps for Graphics - Eurographics Italian Chapter Conference, pages 63–70. The Eurographics Association, 2014.

[3] Johannes Balle, David Minnen, Saurabh Singh, Sung Jin´ Hwang, and Nick Johnston. Variational image compression with a scale hyperprior. In International Conference on Learning Representations (ICLR), 2018.

[4] Jean Begaint, Fabien Racap ´ e, Simon Feltman, and Akshay´ Pushparaja. CompressAI: a PyTorch library and evaluation platform for end-to-end compression research. arXiv preprint arXiv:2011.03029, 2020.

[5] Gisle Bjøntegaard. Calculation of average PSNR differences between RD-curves. Technical Report VCEG-M33, ITU-T Video Coding Experts Group (VCEG), Austin, Texas, USA, 2001.

[6] Hao Cao, Wenqi Guo, Zhijin Qin, and Jungong Han. Efficient Learned Image Compression without Entropy Coding. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

[7] Hao Cao, Chengbin Liang, Wenqi Guo, Zhijin Qin, and Jungong Han. ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 2915– 2925, 2026.

[8] Marlene Careil, Matthew J. Muckley, Jakob Verbeek, and \` Stephane Lathuili ´ ere. Towards image compression with per-\` fect realism at ultra-low bitrates. In The Twelfth International Conference on Learning Representations, 2024.

[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255. IEEE, 2009.

[10] Nicki Skafte Detlefsen, Jiri Borovec, Justus Schock, Ananya Harsh Jha, Teddy Koker, Luca Di Liello, Daniel Stancl, Changsheng Quan, Maxim Grechkin, and William Falcon. TorchMetrics - Measuring Reproducibility in Py-Torch. Journal ofOpen Source Software, 7(70):4101, 2022.

[11] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image Quality Assessment: Unifying Structure and Texture Similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567–2581, 2022.

[12] Eastman Kodak Company. Kodak Lossless True Color Image Suite (PhotoCD PCD0992). https://r0k.us/ graphics/kodak/, 1993.

[13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming¨ Transformers for High-Resolution Image Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12873–12883, 2021.

[14] Frederick N. Fritsch and Judy Butland. A Method for Constructing Local Monotone Piecewise Cubic Interpolants. SIAM Journal on Scientific and Statistical Computing, 5(2): 300–304, 1984.

[15] Haisheng Fu, Feng Liang, Jie Liang, Binglin Li, Guohe Zhang, and Jingning Han. Asymmetric Learned Image Compression With Multi-Scale Residual Block, Importance Scaling, and Post-Quantization Filtering. IEEE Transactions on Circuits and Systemsfor Video Technology, 33(8):4309–4321, 2023.

[16] Ziqing Ge, Siwei Ma, Wen Gao, Jingshan Pan, and Chuanmin Jia. NLIC: Non-Uniform Quantization-Based Learned Image Compression. IEEE Transactions on Circuits and Systems for Video Technology, 34(10):9647–9663, 2024.

[17] Robert M. Gray and David L. Neuhoff. Quantization. IEEE Transactions on Information Theory, 44(6):2325–2383, 1998.

[18] Jinpei Guo, Yifei Ji, Zheng Chen, Kai Liu, Min Liu, Wang Rao, Wenbo Li, Yong Guo, and Yulun Zhang. OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates. In Advances in Neural Information Processing Systems, pages 85267–85286, 2025.

[19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems, pages 6626–6637. Curran Associates, Inc., 2017.

[20] Zhaoyang Jia, Jiahao Li, Bin Li, Houqiang Li, and Yan Lu. Generative Latent Coding for Ultra-Low Bitrate Image Compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26088–26098, 2024.

[21] Zhaoyang Jia, Bin Li, Jiahao Li, Wenxuan Xie, Linfeng Qi, Houqiang Li, and Yan Lu. Towards Practical Real-Time Neural Video Compression. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12543–12552, 2025.

[22] Zhaoyang Jia, Naifu Xue, Zihan Zheng, Jiahao Li, Bin Li, Xiaoyi Zhang, Zongyu Guo, Yuan Zhang, Houqiang Li, and Yan Lu. CoD-Lite: Real-Time Diffusion-Based Generative Image Compression. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

[23] Shiyin Jiang, Wei Long, Minghao Han, Zhenghao Chen, Ce Zhu, and Shuhang Gu. Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14440–14450, 2026.

[24] Joint Video Experts Team (JVET). VVC Test Model (VTM), Version 23.10. https://vcgit.hhi.fraunhofer. de/jvet/VVCSoftware\_VTM/- /tags/VTM- 23. 10, 2025. Accessed: 2025-06-05.

[25] Anle Ke, Xu Zhang, Tong Chen, Ming Lu, Chao Zhou, Jiawen Gu, and Zhan Ma. Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion. In Proceedings of the 42nd International Conference on Machine Learning, pages 29626–29650. PMLR, 2025.

[26] Anqi Li, Feng Li, Yuxi Liu, Runmin Cong, Yao Zhao, and Huihui Bai. Once-for-All: Controllable Generative Image

Compression with Dynamic Granularity Adaptation. In The Thirteenth International Conference on Learning Representations (ICLR), 2025.

[27] Jiahao Li, Wenxuan Xie, Zhaoyang Jia, Bin Li, Zongyu Guo, Xiaoyi Zhang, and Yan Lu. Ultra-Fast Neural Video Compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 41311–41321, 2026.

[28] Yuqi Li, Haotian Zhang, Li Li, and Dong Liu. Learned Image Compression with Hierarchical Progressive Context Modeling. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 18834–18843, 2025.

[29] Zhiyuan Li, Yanhui Zhou, Hao Wei, Chenyang Ge, and Ajmal Mian. RDEIC: Accelerating Diffusion-Based Extreme Image Compression With Relay Residual Diffusion. IEEE Transactions on Circuits and Systemsfor Video Technology, 35(11):11540–11552, 2025.

[30] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations (ICLR), 2019.

[31] Fabian Mentzer, George D. Toderici, Michael Tschannen, and Eirikur Agustsson. High-Fidelity Generative Image Compression. In Advances in Neural Information Processing Systems, pages 11913–11924. Curran Associates, Inc., 2020.

[32] David Minnen and Saurabh Singh. Channel-Wise Autoregressive Entropy Models for Learned Image Compression. In 2020 IEEE International Conference on Image Processing (ICIP), pages 3339–3343, 2020.

[33] Matthew J. Muckley, Alaaeldin El-Nouby, Karen Ullrich, Herve J´ egou, and Jakob Verbeek. Improving Statistical Fi-´ delity for Neural Image Compression with Implicit Local Likelihood Models. In Proceedings of the 40th International Conference on Machine Learning, pages 25426–25443. PMLR, 2023.

[34] Linfeng Qi, Zhaoyang Jia, Jiahao Li, Bin Li, Houqiang Li, and Yan Lu. Generative Latent Coding for Ultra-Low Bitrate Image and Video Compression. IEEE Transactions on Circuits and Systems for Video Technology, 35(10):10500–10515, 2025.

[35] Xihua Sheng, Lingyu Zhu, Tianyu Zhang, Dong Liu, Shiqi Wang, and Jing Wang. CADC: Content Adaptive Diffusion-Based Generative Image Compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 32936–32946, 2026.

[36] Junqi Shi, Ming Lu, Xingchen Li, Anle Ke, Ruiqi Zhang, and Zhan Ma. DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 25719–25729, 2026.

[37] Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. In International Conference on Learning Representations (ICLR), 2015.

[38] Kedar Tatwawadi, Parisa Rahimzadeh, Zhanghao Sun, Zhiqi Chen, Ziyun Yang, Sanjay Nair, Divija Hasteer, and Oren Rippel. What Matters in Practical Learned Image Compression.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12095–12105, 2026.

[39] George Toderici, Wenzhe Shi, Radu Timofte, Lucas Theis, Johannes Balle, Eirikur Agustsson, Nick Johnston, and Fabian´ Mentzer. Workshop and Challenge on Learned Image Compression (CLIC 2020). CVPR Workshop and Challenge, 2020.

[40] Gregory K. Wallace. The JPEG still picture compression standard. IEEE Transactions on Consumer Electronics, 38 (1):xviii–xxxiv, 1992.

[41] Naifu Xue, Zhaoyang Jia, Jiahao Li, Bin Li, Yuan Zhang, and Yan Lu. One-Step Diffusion-Based Image Compression with Semantic Distillation. In Advances in Neural Information Processing Systems, pages 37108–37144, 2025.

[42] Naifu Xue, Zhaoyang Jia, Jiahao Li, Bin Li, Yuan Zhang, and Yan Lu. DLF: Extreme Image Compression with Dualgenerative Latent Fusion. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 19227–19236, 2025.

[43] Ruihan Yang and Stephan Mandt. Lossy Image Compression with Conditional Diffusion Models. In Advances in Neural Information Processing Systems, pages 64971–64995. Curran Associates, Inc., 2023.

[44] Fangzhou Yi, Zhicheng Gong, and Hui Zeng. Rethinking Latent Variable in Learned Image Compression. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 8531–8540, 2026.

[45] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595, 2018.

[46] Tianyu Zhang, Xin Luo, Li Li, and Dong Liu. StableCodec: Taming One-Step Diffusion for Extreme Image Compression. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 17379–17389, 2025.

[47] Tianyu Zhang, Dong Liu, and Chang Wen Chen. Ultra-Low Bitrate Perceptual Image Compression with Shallow Encoder. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12118–12128, 2026.

[48] Yichi Zhang, Zhihao Duan, Ming Lu, Dandan Ding, Fengqing Zhu, and Zhan Ma. Another Way to the Top: Exploit Contextual Clustering in Learned Image Coding. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9377– 9386, 2024.

## Appendix

In the appendix, we provide the following:

• Section A derives the reorganized P-RVQ, analyzes its complexity, and reports implementation details and latency results.

• Section B describes the model architecture, bitstream packing, padding, and nominal rate.

• Section C defines the dataset, metric, runtime, baseline, and GR ablation evaluation protocols.

• Section D reports additional efficiency and quality comparisons, high-resolution results, and GR inference stability.

• Section E provides additional visual comparisons on all four evaluation datasets.

## A. Analysis of Reorganized P-RVQ

The main text presents the key reorganization of P-RVQ with projection biases omitted for clarity. This appendix restores the complete affine transformations, establishes the equivalence between the original and reorganized forms, analyzes their computation and storage, and reports the corresponding runtime results.

## A.1. Complete P-RVQ Reorganization

We use the notation of Sec. 3.3. Le $z \in \mathbb { R } ^ { D }$ denote the input to a P-RVQ module, covering both the hyper latent and the main latent groups. The following equations describe one input vector, while L denotes the number of input vectors when counting operations.

At stage m, the affine projection $\left( A _ { m } , \pmb { a } _ { m } \right)$ maps the current residual from $\mathbb { R } ^ { D }$ to the search space $\mathbb { R } ^ { d }$ , where $\boldsymbol { A } _ { m } ~ \in ~ \mathbb { R } ^ { d \times D }$ and $\mathbf { a } _ { m } ~ \in ~ \mathbb { R } ^ { d }$ The codebook $\begin{array} { r l } { E _ { m } } & { { } = } \end{array}$ $[ \pmb { E } _ { m } [ 1 ] , \dots , \pmb { E } _ { m } [ K ] ] \in \mathbb { R } ^ { d \times K }$ contains K codewords. The reconstruction projection $\left( { \cal B } _ { m } , \pmb { b } _ { m } \right)$ maps the selected codeword back to $\mathbb { R } ^ { D }$ , where $\bar { \boldsymbol { B } } _ { m } \in \mathbb { R } ^ { \bar { \boldsymbol { D } } \times d }$ and $\pmb { b } _ { m } \in \mathbb { R } ^ { D }$ . The value of K may differ among the five P-RVQ modules.

Starting from $\hat { z } _ { 0 } = \mathbf { 0 } ,$ , let $\hat { z } _ { m }$ denote the partial reconstruction after stage m, and let $\hat { z } = \hat { z } _ { M }$ denote the final reconstruction.

Original execution. The original P-RVQ projects the current residual for nearest neighbor search and immediately adds the selected reconstruction to the high dimensional state:

$$
i _ { m } = \arg \operatorname* { m i n } _ { 1 \leq j \leq K } \| A _ { m } ( z - \hat { z } _ { m - 1 } ) + { \pmb a } _ { m } - { \pmb E } _ { m } [ j ] \| _ { 2 } ^ { 2 } ,\tag{9}
$$

$$
\hat { z } _ { m } = \hat { z } _ { m - 1 } + B _ { m } E _ { m } [ i _ { m } ] + b _ { m } .\tag{10}
$$

The final reconstruction is therefore

$$
\hat { z } = \hat { z } _ { M } = \sum _ { m = 1 } ^ { M } \left( B _ { m } E _ { m } [ i _ { m } ] + b _ { m } \right) .\tag{11}
$$

Before stage $m + 1$ , the selected codeword must pass through $B _ { m }$ and update the state in $\mathbb { R } ^ { D }$ . The next stage then projects this updated residual back to $\mathbb { R } ^ { d }$ . This produces the $\bar { \mathbb { R } } ^ { \bar { D } } \xrightarrow { } \mathbb { R } ^ { d } \xrightarrow { \bullet } \bar { \mathbb { R } } ^ { D }$ dependency path shown in Equation (6).

Reorganized execution. Expanding the partial reconstruction inside the search projection gives

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { m } \big ( z - \hat { z } _ { m - 1 } \big ) + { \bf a } _ { m } } } \\ { { \displaystyle ~ = { \cal A } _ { m } z + { \bf a } _ { m } - \sum _ { \ell < m } { \cal A } _ { m } \big ( B _ { \ell } { \cal E } _ { \ell } [ i _ { \ell } ] + b _ { \ell } \big ) . } } \end{array}
$$

All parameters and codewords in the terms from earlier stages are fixed after training. We therefore precompute the d dimensional lookup table

$$
\begin{array} { r l } { \pmb { F } _ { \ell  m } [ j ] = \pmb { A } _ { m } ( \pmb { B } _ { \ell } \pmb { E } _ { \ell } [ j ] + \pmb { b } _ { \ell } ) } & { { } } \\ { = ( \pmb { A } _ { m } \pmb { B } _ { \ell } ) \pmb { E } _ { \ell } [ j ] + \pmb { A } _ { m } \pmb { b } _ { \ell } , } & { { } \quad \ell < m . } \end{array}\tag{12}
$$

This restores the bias term omitted from the simplified expression in Equation (7). Using these tables, the reorganized recursion becomes

$$
\begin{array} { l } { { \displaystyle i _ { m } = \arg \operatorname* { m i n } _ { 1 \leq j \leq K } \| A _ { m } z + a _ { m }  } } \\ { { \displaystyle \qquad - \sum _ { \ell < m } F _ { \ell  m } [ i _ { \ell } ] - E _ { m } [ j ] \| _ { 2 } ^ { 2 } } , } \\ { { \displaystyle \hat { z } _ { m } = \hat { z } _ { m - 1 } + B _ { m } E _ { m } [ i _ { m } ] + b _ { m } } . } \end{array}\tag{13}
$$

The source projections for all stages are evaluated together:

$$
\begin{array} { r } { \left[ \begin{array} { c } { s _ { 1 } ^ { ( 0 ) } } \\ { \vdots } \\ { \vdots } \\ { s _ { M } ^ { ( 0 ) } } \end{array} \right] = \left[ \begin{array} { c } { A _ { 1 } } \\ { \vdots } \\ { A _ { M } ^ { } } \end{array} \right] z + \left[ \begin{array} { c } { a _ { 1 } } \\ { \vdots } \\ { a _ { M } ^ { } } \end{array} \right] . } \end{array}\tag{14}
$$

At stage m, the corresponding search state is

$$
s _ { m } = s _ { m } ^ { ( 0 ) } - \sum _ { \ell < m } F _ { \ell  m } [ i _ { \ell } ] = { \cal A } _ { m } ( z - \hat { z } _ { m - 1 } ) + a _ { m } .\tag{15}
$$

After selecting $i _ { \ell } ,$ the later search states are updated through $F _ { \ell \to m } [ i _ { \ell } ]$ for $m > \ell .$ Equation 15 shows that the original and reorganized forms present the same search vector to every stage in real arithmetic. Starting from the first stage, the same nearest neighbor decision therefore produces the same update for all later stages. Induction over m preserves all greedy indices and the final reconstruction.

The reorganized execution does not need to materialize the high dimensional partial reconstruction between search stages. It maintains the low dimensional states $s _ { m }$ until all indices are selected, and then gathers and sums $B _ { m } { E _ { m } [ i _ { m } ] + }$ $\boldsymbol { b } _ { m }$ to obtain Equation (11). The greedy stage order remains unchanged, while the high dimensional reconstruction is removed from the critical dependency path.

For nearest neighbor search, the part of the squared distance that depends on candidate j is

$$
\begin{array} { r l } & { i _ { m } = \arg \operatorname* { m i n } _ { j } \| \pmb { s } _ { m } - \pmb { E } _ { m } [ j ] \| _ { 2 } ^ { 2 } } \\ & { \quad = \arg \operatorname* { m i n } _ { j } \left( \| \pmb { E } _ { m } [ j ] \| _ { 2 } ^ { 2 } - 2 \pmb { s } _ { m } ^ { \top } \pmb { E } _ { m } [ j ] \right) , } \end{array}
$$

because $\| \pmb { s } _ { m } \| _ { 2 } ^ { 2 }$ is constant across candidates. The codeword norms and codebook transposes can therefore be stored before inference.

## A.2. Complexity and Parallelism

The reorganization preserves the projection used for search and the nearest neighbor computation. For L input vectors, the shared multiply-accumulate counts are

$$
\begin{array} { r c l } { { \displaystyle \mathcal { C } _ { \mathrm { p r o j } } ^ { \mathrm { M A C } } = L \sum _ { m = 1 } ^ { M } D d = L M D d , } } \\ { { \displaystyle \mathcal { C } _ { \mathrm { s e a r c h } } ^ { \mathrm { M A C } } = L \sum _ { m = 1 } ^ { M } K d = L M K d , } } \\ { { \displaystyle \mathcal { C } _ { \mathrm { c o m m o n } } ^ { \mathrm { M A C } } = L M ( D + K ) d . } } \end{array}
$$

Both forms also require LMd additions for projection biases and $L M ( K - 1 )$ candidate comparisons. In the evaluated codec, the four main latent P-RVQ modules use $D = 2 5 6 , d = 1 6$ , and $K \in \{ 1 0 2 4 , 5 1 2 , 2 5 6 , 1 2 8 \}$ The hyper latent P-RVQ uses D = 128, d = 8, and $K = 1 0 2 4$ Thus, all five modules satisfy $D = 1 6 d .$

Reconstruction cost. The original execution evaluates $B _ { m } { E } _ { m } [ i _ { m } ] + b _ { m }$ online for every selected codeword. The reorganized execution evaluates these projections for all codewords during checkpoint conversion and stores the resulting reconstruction tables. For one P-RVQ module,

$$
\begin{array} { r } { \mathcal { C } _ { \mathrm { o u t , o n l i n e } } ^ { \mathrm { M A C } } = L M D d , } \\ { \mathcal { C } _ { \mathrm { o u t , o f f i n e } } ^ { \mathrm { M A C } } = M K D d , } \\ { \mathcal { S } _ { \mathrm { d e c o d e } } = M K D . } \end{array}
$$

The original online path additionally performs LMD bias additions. The reorganized online path replaces the

projections with table lookups and requires $L ( M - 1 ) D$ additions to sum the selected reconstruction vectors. Both forms produce the same reconstruction in Equation (11).

State update cost. The main reduction occurs in the sequential state updates. The corresponding addition counts and their ratio are

$$
\mathcal { C } _ { \mathrm { h i g h } } ^ { \mathrm { a d d } } = L \sum _ { m = 2 } ^ { M } D = L ( M - 1 ) D ,\tag{16}
$$

$$
\mathcal { C } _ { \mathrm { l o w } } ^ { \mathrm { a d d } } = L \sum _ { m = 2 } ^ { M } ( m - 1 ) d = \frac { L M ( M - 1 ) d } { 2 } ,\tag{17}
$$

$$
\frac { \mathcal { C } _ { \mathrm { h i g h } } ^ { \mathrm { a d d } } } { \mathcal { C } _ { \mathrm { l o w } } ^ { \mathrm { a d d } } } = \frac { 2 D } { M d } .\tag{18}
$$

For $M = 4$ and $D = 1 6 d ,$ the original and reorganized state update counts become 48Ld and 6Ld, respectively. The reorganization therefore uses 8× fewer additions for state propagation. The projection $\mathbf { M A C s } ,$ nearest neighbor MACs, and number of ordered greedy decisions remain unchanged. This reduction characterizes the sequential state update path rather than the complete P-RVQ runtime.

Parallelism. The original schedule exposes one source projection at each stage. The reorganized schedule evaluates the M source projections together. After selecting $i _ { \ell } ,$ the lookup updates for all later stages can also be applied concurrently. The reorganized execution therefore exposes all M source projections and up to M − 1 parallel low dimensional updates, while retaining the same M ordered nearest neighbor decisions.

Lookup storage. The cross stage tables are constructed once after training. Their offline computation and nonzero storage for one P-RVQ module are

$$
\begin{array} { l } { { \displaystyle { \mathcal { C } } _ { \mathrm { c r o s s , o f f i n e } } ^ { \mathrm { M A C } } = K D d \sum _ { m = 2 } ^ { M } ( m - 1 ) = \frac { M ( M - 1 ) K D d } { 2 } } , } \\ { { \displaystyle { S _ { \mathrm { c r o s s , n o n z e r o } } = K d \sum _ { m = 2 } ^ { M } ( m - 1 ) = \frac { M ( M - 1 ) K d } { 2 } } . } } \end{array}
$$

For $M = 4 ,$ six pairs satisfy $\ell < m$ . Across all five P-RVQ modules, the corresponding nonzero content is

$$
\begin{array} { r l } & { S _ { \mathrm { n o n z e r o } } ^ { \mathrm { a l l } } = 6 \big ( 1 0 2 4 \cdot 1 6 + 5 1 2 \cdot 1 6 + 2 5 6 \cdot 1 6 } \\ & { \qquad + 1 2 8 \cdot 1 6 + 1 0 2 4 \cdot 8 \big ) } \\ & { \qquad = 2 3 3 , 4 7 2 . } \end{array}
$$

The implementation uses a dense table with $M ^ { 2 } = 1 6$ stage slots, giving

$$
\begin{array} { r l } & { S _ { \mathrm { d e n s e } } ^ { \mathrm { a l l } } = 1 6 \big ( 1 0 2 4 \cdot 1 6 + 5 1 2 \cdot 1 6 + 2 5 6 \cdot 1 6 } \\ & { \qquad + 1 2 8 \cdot 1 6 + 1 0 2 4 \cdot 8 \big ) } \\ & { \qquad = 6 2 2 , 5 9 2 . } \end{array}
$$

The dense table has shape (M, K, Md) and occupies 1.19 MiB in FP16 or 2.38 MiB in FP32 across the five P-RVQ modules. This storage is prepared before inference and contains the static lookup values used by the reorganized search.

## A.3. Implementation and Runtime Evaluation

CUDA implementation. Checkpoint conversion materializes the stacked search projection, codeword norms, codebook transposes, reconstruction tables, and the cross stage lookup tables in Equation (12). At runtime, the source projections in Equation (14) are evaluated with one matrix multiplication.

A fused CUDA kernel then processes the ordered P-RVQ stages. Candidate dot products, distance evaluation, parallel reduction for index selection, table lookup, and updates to the later low dimensional states are performed within one kernel launch. The kernel preserves the ordered greedy decisions while parallelizing candidate evaluation and the updates to future search states. This avoids separate search launches at every stage and avoids materializing complete distance and update tensors.

When reconstruction is required, a second kernel gathers the selected high dimensional vectors, sums them, and writes the reconstructed output. During decompression, these reconstruction operations are further combined with the surrounding codec operations. This codec integration lies outside the isolated P-RVQ timing reported below.

Benchmark protocol. We evaluate the original Torch FP32 implementation, the reorganized Torch implementations in FP32 and FP16, and the CUDA FP16 implementation on one NVIDIA A100 with batch size one. All implementations use weights converted from the same checkpoints.

We evaluate $M \in \{ 1 , 2 , 3 , 4 \}$ at latent sizes corresponding to 512 × 768, 1080p, 2K, and 4K images, giving 16 settings in total. Each latency is the median of 100 synchronized repetitions after 20 warmup runs.

The reported latency isolates P-RVQ execution. Complete codec latency is reported in Tabs. 1, 3 and 10.

Runtime results. Under the same Torch FP32 backend and precision, the reorganized P-RVQ achieves a 3.14× average speedup over the original implementation across the 16 evaluated settings. The reorganized Torch FP16 implementation achieves a 2.97× average speedup. These matched Torch results isolate the effect of the algebraic reorganization and numerical precision.

The CUDA FP16 implementation further increases the average speedup to 7.15× relative to the original FP32 implementation. This number includes the algebraic reorganization, FP16 execution, and fused CUDA kernels. Table 4 reports the latency and speedup for every resolution and P-RVQ depth.

At the highest evaluated Kodak rate, the P-RVQ latency during compression is 11.2824 ms for the original FP32 implementation, 3.2435 ms for the reorganized Torch FP32 implementation, and 0.6001 ms for the CUDA FP16 implementation. The matched FP32 reorganization gives a 3.48× speedup at this operating point. The CUDA FP16 implementation gives an 18.80× speedup.

For complete encoder attribution, we construct an original P-RVQ reference by restoring the latency difference between the original and reorganized FP32 P-RVQ to the measured Torch FP32 encoder. Using the reduction from this reference to the final CUDA FP16 encoder as a common denominator, the algebraic reorganization accounts for 23.54% of the total encoder latency reduction. The complete P-RVQ acceleration, including the CUDA implementation, accounts for 31.28%.

Numerical consistency. The original and reorganized Torch FP32 implementations select identical indices, with a maximum absolute reconstruction difference of $4 . 7 7 \times 1 0 ^ { - 7 }$ Under FP16, the reorganized Torch and CUDA implementations differ on only 0.2293% of the evaluated indices. These differences mainly occur between nearby codewords and have little effect on the final reconstruction.

## B. Architecture and Bitstream Details

## B.1. Backbone and Context Pathway

GR-GIC follows the lightweight backbone and context pathway of EF-LIC [6]. Table 5 gives the complete implemented topology.

For clarity, Sec. 3.1 absorbs the normalization used in the implementation into the analysis and synthesis transforms. The encoder first computes the latent before normalization,

$$
{ \pmb v } = g _ { a } ( { \pmb x } ) , \qquad { \pmb z } = h _ { a } ( { \pmb v } ) .
$$

After quantizing the hyper latent, the hyper decoder produces the normalization scale and context feature,

$$
( \pmb { q } _ { \mathrm { e n c } } , \phi ) = h _ { s } ( \hat { z } ) , \qquad \pmb { y } = \pmb { v } \oslash \pmb { q } _ { \mathrm { e n c } } .
$$

The normalized latent y is then processed by the groupwise affine projections and P-RVQ.

Table 4. P-RVQ latency on an NVIDIA A100. Each entry reports the median CUDA event latency in ms followed by speedup over the corresponding original FP32 result. All implementations use weights converted from the same checkpoints.
<table><tr><td>Resolution</td><td>M</td><td>Original FP32</td><td>Reorganized FP32</td><td>Reorganized FP16</td><td>CUDA FP16</td></tr><tr><td rowspan="5">512×768</td><td>1</td><td>3.84/1.00×</td><td>1.54/2.50×</td><td>1.54/2.49×</td><td>0.93/4.12×</td></tr><tr><td>2</td><td>5.38/1.00×</td><td>1.74/3.09×</td><td>1.74/3.09×</td><td>0.64/8.43×</td></tr><tr><td>3</td><td>10.80/1.00×</td><td>3.29/3.29×</td><td>3.33/3.24×</td><td>0.87/12.38×</td></tr><tr><td>4</td><td>14.09/1.00×</td><td>4.03/3.49×</td><td>4.15/3.40×</td><td>0.87/16.12×</td></tr><tr><td>1</td><td>2.82/1.00×</td><td>1.14/2.47×</td><td>1.14/2.48×</td><td>0.67/4.18×</td></tr><tr><td rowspan="4">1080p</td><td>2</td><td>5.42/1.00×</td><td>1.77/3.07×</td><td>1.83/2.97×</td><td>0.66/8.26×</td></tr><tr><td>3</td><td>8.37/1.00×</td><td>2.43/3.45×</td><td>2.56/3.27×</td><td>0.78/10.67×</td></tr><tr><td>4</td><td>10.59/1.00×</td><td>3.03/3.49×</td><td>3.20/3.31×</td><td>0.95/11.16×</td></tr><tr><td>1</td><td>2.88/1.00×</td><td>1.12/2.58×</td><td></td><td>0.68/4.25×</td></tr><tr><td rowspan="4">2K</td><td>2</td><td>5.40/1.00×</td><td>1.74/3.09×</td><td>1.16/2.47×</td><td>0.87/6.23×</td></tr><tr><td>3</td><td>8.34/1.00×</td><td>2.51/3.33×</td><td>1.96/2.76×</td><td></td></tr><tr><td>4</td><td>10.83/1.00×</td><td>3.06/3.55×</td><td>2.55/3.27×</td><td>1.18/7.10×</td></tr><tr><td></td><td></td><td></td><td>3.27/3.31×</td><td>1.48/7.32×</td></tr><tr><td rowspan="4">4K</td><td>1</td><td>2.93/1.00×</td><td>1.13/2.61×</td><td>1.16/2.54×</td><td>0.98/3.00×</td></tr><tr><td>2</td><td>5.97/1.00×</td><td>1.92/3.11×</td><td>3.00/1.99×</td><td>1.67/3.58×</td></tr><tr><td>3</td><td>8.61/1.00×</td><td>2.45/3.51×</td><td>2.49/3.46×</td><td>2.30/3.74×</td></tr><tr><td>4</td><td>11.42/1.00×</td><td>3.09/3.69×</td><td>3.24/3.52×</td><td>2.99/3.82×</td></tr><tr><td>Mean speedup</td><td></td><td>1.00×</td><td>3.14×</td><td>2.97×</td><td>7.15×</td></tr></table>

At the decoder, the same hyper decoder gives

$$
( q _ { \mathrm { d e c } } , \phi ) = h _ { s } ( \hat { z } ) .
$$

For a fixed checkpoint and bitstream, $q _ { \mathrm { e n c } } ~ = ~ q _ { \mathrm { d e c } }$ , and reconstruction is obtained as

$$
\hat { \pmb x } = g _ { s } ( \pmb q _ { \mathrm { d e c } } \odot \tilde { \pmb y } ) .
$$

The notation in Sec. 3.1 absorbs these deterministic scales into $g _ { a }$ and $g _ { s }$ . The normalization scale $\pmb { q } _ { \mathrm { d e c } }$ and entropy context scale σ are lower bounded by 0.11.

The main latent uses the same four group quadtree context model as EF-LIC and DCVC-RT [6, 21]. Each group contains one quarter of the spatial positions and is recovered using the previously decoded groups. The GR head receives the recovered latent and entropy context parameters $( \hat { \boldsymbol y } , \mu , \sigma )$ and predicts the offset δ and perturbation scale $\beta .$ Each P-RVQ depth $M \in \{ 1 , 2 , 3 , 4 \}$ defines one separately trained operating point with its corresponding GR head.

## B.2. Fixed-Width Index Packing

Each power of two codebook in Tab. 6 uses a fixed index width according to the size of its quantization alphabet [17]. The hyper latent uses 10 bits per index, while the four main latent groups use 10, 9, 8, and 7 bits.

The multistage structure and low dimensional search spaces follow the P-RVQ configuration of EF-LIC [6]. Stored indices are flattened in (stage, batch, height, width)

order and written with the least significant bit first. The five streams are ordered as $Q _ { z }  Q _ { 1 }  Q _ { 2 }  Q _ { 3 }  Q _ { 4 }$

A standalone bitstream stores the original image height and width as two unsigned 16 bit integers, followed by the five index streams. No additional side information is transmitted. We report payload BPP and exclude this 4 byte image size header. For a $5 1 2 \times 7 6 8$ image, the header contributes only

$$
R _ { \mathrm { m e t a } } = { \frac { 3 2 } { 5 1 2 \cdot 7 6 8 } } = 8 . 1 4 \times 1 0 ^ { - 5 } { \mathrm { ~ b p p } } .
$$

## B.3. Padding and Cropping

The encoder pads the input on the right and bottom so that both spatial dimensions are multiples of 32. The hyper analysis path additionally pads the main latent to a multiple of four, while the main latent path retains its original logical shape. The quadtree groups, main latent P-RVQ modules, GR head, and synthesis transform therefore operate on this original latent shape. The decoder removes the corresponding padding and returns the reconstruction at the original image size.

## B.4. Nominal Rate

The fixed index widths give an exact nominal payload rate before boundary padding. At each P-RVQ stage, the four main latent groups contribute

Table 5. Detailed GR-GIC architecture. Conv, DCB, and SP denote convolution, DepthConvBlock, and subpixel upsampling, respectively. The analysis and synthesis depths count the repeated body DCBs, while the entry and final refinement blocks are listed separately.
<table><tr><td>Module</td><td>Implemented topology</td><td>Output or role</td></tr><tr><td>Analysis transform</td><td>8× pixel unshuffle, 192 →368 entry DCB, six 368-channel body DCBs, 3 × 3 Conv with stride 2 to 256 channels</td><td>Main latent at 1/16 resolution</td></tr><tr><td>Hyper analysis</td><td>1 × 1 Conv 256 → 512, two 512-channel DCBs, two residual downsampling stages with stride 2, final 128-channel projection</td><td>Hyper latent at 1/64 resolution</td></tr><tr><td>Hyper synthesis</td><td>SP 128 → 512, one DCB, SP 512 → 512, two DCBs, 1 × 1 Conv to 512 channels</td><td>256-channel normalization scale  $\mathbf { q } _ { \mathrm { d e c } }$  and 256-channel context feature φ</td></tr><tr><td>Context path</td><td>Quadtree support projection, projection of previously decoded groups, one group adaptor DCB, three shared 512-channel DCBs, final 1 × 1 Conv</td><td>256-channel entropy context mean µ and scale σ</td></tr><tr><td>GR head</td><td>1 × 1 Conv 768 → 768, three 768-channel DCBs, 1 × 1 Conv 768→512</td><td>256-channel offset δ and adaptive perturbation scale β</td></tr><tr><td>Synthesis transform</td><td>SP 256 → 368, twelve 368-channel body DCBs, one final refinement DCB, 1 × 1 Conv 368 → 192, 8× pixel shuffle</td><td>RGB reconstruction</td></tr><tr><td>DCB</td><td>Pointwise mixing, weighted sigmoid linear unit (SiLU), 3× 3 depthwise Conv, pointwise projection, fourfold feedforward expansion and projection</td><td>Residual feature transformation</td></tr></table>

Table 6. P-RVQ configuration. The four main latent quantizers operate on quadtree groups. The fifth quantizes the hyper latent. “Bits per stage” is the fixed index width.
<table><tr><td>Stream</td><td>y1</td><td>Y2</td><td>Y3</td><td>Y4</td><td>z</td></tr><tr><td>Input dim.</td><td>256</td><td>256</td><td>256</td><td>256</td><td>128</td></tr><tr><td>Code dim.</td><td>16</td><td>16</td><td>16</td><td>16</td><td>8</td></tr><tr><td>Entries</td><td>1024</td><td>512</td><td>256</td><td>128</td><td>1024</td></tr><tr><td>Bits per stage</td><td>10</td><td>9</td><td>8</td><td>7</td><td>10</td></tr></table>

$$
R _ { y } = { \frac { 1 0 + 9 + 8 + 7 } { 4 \cdot 1 6 ^ { 2 } } } = 0 . 0 3 3 2 0 3 1 2 5 { \mathrm { ~ b p p } } ,
$$

while the hyper latent contributes

$$
R _ { z } = { \frac { 1 0 } { 6 4 ^ { 2 } } } = 0 . 0 0 2 4 4 1 4 0 6 2 5 \mathrm { b p p } .
$$

Thus, with M active P-RVQ stages,

$$
R _ { M } = M ( R _ { y } + R _ { z } ) = 0 . 0 3 5 6 4 4 5 3 1 2 5 M { \mathrm { ~ b p p } } .\tag{19}
$$

Boundary padding changes the number of transmitted indices and causes the small deviations from these nominal rates.

## C. Evaluation Protocol

## C.1. Datasets and Objective Evaluation

Kodak, Tecnick, DIV2K validation, and CLIC 2020 Professional test contain 24, 100, 100, and 250 images, respectively. All reconstructed images are evaluated at their original image resolution.

Objective evaluation. We use standardized objective metrics as the primary quantitative evaluation. Subjective user studies can provide complementary perceptual evidence, but their results depend on the participant population, viewing conditions, presentation protocol, and selected test images, and naturally exhibit observer variability. These factors make subjective scores less suitable for reproducible comparisons across many methods, datasets, and operating points. Standardized objective metrics instead provide repeatable measurements under a fixed evaluation protocol.

This practice is also common in generative image compression. Representative methods including MS-ILLM, GLC, Control-GIC, DLF, OneDC, OSCAR, and RDVQ primarily evaluate reconstruction using objective metrics such as LPIPS, DISTS, and FID without reporting a subjective user study [18, 20, 23, 26, 33, 41, 42]. We therefore use LPIPS and DISTS to measure reference-based perceptual similarity, PSNR to measure pixel fidelity, and FID to measure distribution-level alignment, together with qualitative reconstructions for visual comparison.

LPIPS is computed with pretrained VGG features after mapping RGB inputs to [−1, 1]. DISTS receives RGB inputs in [0, 1], and PSNR is computed in RGB.

FID. Following the patch evaluation protocol used for lowrate generative image compression [20], we compute FID from local image patches. Reference and reconstructed RGB images in [0, 1] are rounded to unsigned 8-bit values and passed to the TorchMetrics implementation of FID [10, 19].

Kodak uses $6 4 \times 6 4$ patches, while Tecnick, DIV2K, and CLIC 2020 use $2 5 6 \times 2 5 6$ patches. We extract one nonoverlapping patch grid and a second grid shifted by half a patch along both spatial axes. This increases the number of Inception samples and improves the stability of distributionlevel evaluation on the 24-image Kodak set. All compared methods use the same FID pipeline.

BD-rate. We use BD-rate [5] to summarize the average rate difference between two rate-quality curves at matched reconstruction quality. For each metric, operating points are ordered by the metric value, and pairs without a valid common interval are left unreported.

Let q denote the metric value. We use monotone PCHIP interpolation [14] for the natural logarithm of bitrate. Let $f _ { \mathrm { A } } ( q )$ and $f _ { \mathrm { T } } ( \boldsymbol { q } )$ denote the interpolated curves of the anchor and tested codec over the common interval $[ q _ { \mathrm { m i n } } , q _ { \mathrm { m a x } } ]$ . The reported BD-rate is

$$
\begin{array} { r l } { \displaystyle } & { \mathrm { B D - r a t e } = 1 0 0 \left[ \exp \left( \frac { 1 } { q _ { \mathrm { m a x } } - q _ { \mathrm { m i n } } } \right. } \\ & { \left. \int _ { q _ { \mathrm { m i n } } } ^ { q _ { \mathrm { m a x } } } \left( f _ { \mathrm { T } } ( q ) - f _ { \mathrm { A } } ( q ) \right) \mathrm { d } q \right) - 1 \right] . } \end{array}\tag{20}
$$

OneDC is the anchor for the main comparisons. Negative values indicate bitrate savings over the common interval. The numerical tables report pairwise BD-rate, while the ratequality curves show the corresponding behavior over the evaluated operating points.

## C.2. Runtime and Model Cost Protocol

All A100 codec runtimes use batch size one on one NVIDIA A100 GPU. For GR-GIC, CPU payload processing is performed on an AMD EPYC 7763.

The latency comparison retains the numerical precision, inference backend, and acceleration path provided by each available implementation. The reported values therefore characterize realized system latency rather than architectureonly speed under an identical backend. EF-LIC and ProGIC follow their published FP32 evaluation settings [6, 7], while the remaining methods retain their released inference settings.

The highly optimized implementation of GR-GIC combines reorganized P-RVQ with fused CUDA operators following DCVC-UF [27]. It uses FP16 execution and CUDA Graphs together with custom kernels for quantization and reconstruction. The remaining convolution operations use optimized ATen and cuDNN kernels.

The reported encoding and decoding times cover the complete codec path. Encoding includes index generation, transfer to the CPU, and bitstream packing. Decoding includes bitstream unpacking, transfer to the GPU, index recovery, one GR realization, and image synthesis. Timing starts after model loading, input construction, and CUDA Graph capture. We use two warmup runs followed by ten synchronized repetitions.

For GR-GIC, the reported latency and model cost are averaged over four independently trained operating points at 0.0356, 0.0713, 0.1069, and 0.1426 BPP. The same averaging convention is used for the A100 and Jetson results.

Parameter accounting across operating points. We report the parameter count of the complete codec deployed at one evaluated operating point. Each of the four independently trained GR-GIC checkpoints contains 79.389M parameters. The reported 79.389M therefore represents one deployed model rather than the combined storage of all four checkpoints.

This convention is also applied to methods that use separate checkpoints for different target rates. Among the compared methods, ResULIC, StableCodec, RDEIC, OneDC, MS-ILLM, DLF, DiT-IC, AEIC, and CoD-Lite use separately trained or fine-tuned model weights for different operating points [22, 25, 29, 33, 36, 41, 42, 46, 47].

We use the same parameter accounting rule for both types of codec. The tables report the complete model required for the evaluated operating point and do not sum multiple checkpoints across the rate curve. The storage required to retain all supported operating points is a separate deployment property from the parameter count of one deployed codec.

FLOPs are measured for one encoder and decoder pass with batch size one on the corresponding PyTorch FP32 path under the profiler-supported operator scope. We count one multiply-accumulate operation as two FLOPs. The same pass definition and arithmetic convention are used for all methods. Peak memory denotes the maximum allocated GPU memory including the loaded model and active codec buffers.

## C.3. Baseline Evaluation Protocol

Except where noted, we evaluate released checkpoints on the same NVIDIA A100 with batch size one. Each method follows the coding and inference procedure provided by its official implementation.

EF-LIC and ProGIC use their published FP32 settings [6, 7]. OneDC follows its official path with BF16 autocast for the feedforward and codec networks while retaining the large VAE in FP32 [41]. CoD-Lite uses its official 0.0312 BPP checkpoint with BF16 inference [22]. Under our full codec profiling scope, this checkpoint gives 771.69G FLOPs at $5 1 2 \times 7 6 8$ The CoD-Lite paper reports 535 kMAC per pixel, corresponding to 420.74G FLOPs under our arithmetic convention. We use our common profiling scope for the system comparison and retain the published value as a separate reference. Control-GIC uses its official granularity control, which we enumerate at intervals of 0.01, together with official tiling on DIV2K and CLIC 2020 [26].

For GLC [20], the released interface provides analysis and synthesis without a complete bitstream path. For the high-resolution codec comparison, we additionally use a CompressAI 1.2.6 integration [4] that produces and decodes the actual bitstream.

For OSCAR, rate and quality evaluation uses CPU offloading on high-resolution images when required, while the resource tables report the direct execution result [18].

CADC [35] and PICO [38] do not provide the corresponding executable models used in our comparison. We therefore use their published curves or released numerical results only when the metric definition and evaluation range are compatible with ours.

For CoD-Lite, measurements from the released implementation differ from the latency values reported in its paper. At the 1024 × 1920 reference resolution, we measure 39.522 ms for encoding and 56.863 ms for decoding, compared with the reported 16 and 24 ms. We therefore use measurements from the released code under our common hardware and timing protocol rather than mixing measured and separately reported runtime values. The system comparisons use the official 0.0312 BPP checkpoint specified by the published speed evaluation.

## C.4. GR Ablation Protocol

The configurations in Tab. 2 are trained independently from scratch at each bitrate. All configurations use the same training data, loss terms and weights, AdamW settings, learningrate schedule, and update budget. Each ablation model is trained for 1M iterations. The main operating-point results use separately trained models with the complete 2M iteration schedule.

$\mathrm { { M } _ { 0 } }$ bypasses GR and directly synthesizes from the recovered quantized latent. It retains the direct synthesis path of the EF-LIC backbone and provides a reference under the same training conditions used for the GR configurations.

$\mathrm { M } _ { 1 }$ predicts an offset from the recovered latent. $\mathrm { M } _ { 2 }$ additionally uses entropy context for deterministic refinement. $\mathrm { M _ { 3 } }$ introduces Gaussian perturbation with one learned scale for each channel, shared across spatial locations. $\mathrm { M } _ { 4 }$ further adapts the perturbation scale to spatial location and image content.

Each crop for $\mathrm { M _ { 3 } }$ and $\mathrm { M } _ { 4 }$ uses one standard Gaussian draw to estimate the expected image objective. $\mathrm { M } _ { 2 }$ and $\mathrm { M _ { 3 } }$ have the same reported parameter count and forward FLOPs as $\mathrm { M } _ { 4 }$ . They provide matched references for evaluating stochastic perturbation and its adaptation to image content and spatial location.

## C.5. Stochastic Evaluation Protocol

The main results use a fixed decoder seed of 1926 and reconstruct each bitstream once. Stochastic sampling is applied only after deterministic RVQ index recovery, so the transmitted indices and payload do not depend on the decoder seed.

Inference perturbation ablation. To examine the role of the sampled perturbation after training, we use the same trained $\mathrm { M } _ { 4 }$ checkpoints as in Tab. 2. We retain the recovered indices, learned offset, and all model parameters, and set only $\beta \odot \epsilon$ in Equation (5) to zero during inference. The decoder therefore synthesizes from the learned refinement center

$$
\tilde { { \pmb y } } = \hat { { \pmb y } } + \delta .
$$

The corresponding results are reported in the additional experiments.

Decoder seed stability. To measure sensitivity to the sampled Gaussian perturbation, we keep the checkpoint, transmitted indices, input images, and all deterministic computations fixed and vary only the decoder seed. We evaluate ten seeds on all Kodak images at each operating point. For every seed, metrics are first averaged over the dataset before variation across seeds is measured.

The resulting stability measurements are reported together with the additional GR experiments.

## D. Additional Experimental Results

This section reports additional experimental results under the evaluation protocols defined in Sec. C. We first examine the efficiency of different GR-GIC operating points and inference backends, then provide additional comparisons with existing codecs, and finally analyze the inference behavior of GR.

## D.1. Efficiency across Operating Points

Table 7 reports the complexity and runtime of the four independently trained operating points at $5 1 2 \times 7 6 8$ , 1080p, and 2K.

${ \mathrm { A t ~ } } 5 1 2 \times 7 6 8 { \mathrm { . } }$ , encoding latency ranges from 4.475 to 4.717 ms, while decoding latency ranges from 5.947 to 6.028

Table 7. Per operating point GR-GIC complexity, CUDA latency, and peak memory on one A100. FLOPs are reported in G for one encoder and decoder pass under the profiler-supported operator scope. Latency is in ms and peak GPU memory is in GiB.
<table><tr><td>Resolution</td><td>BPP</td><td>Params</td><td>FLOPs</td><td>Enc.</td><td>Dec.</td><td>Mem.</td></tr><tr><td rowspan="4">512×768</td><td>0.0356</td><td>79.389</td><td>456.491</td><td>4.475</td><td>5.949</td><td>0.183</td></tr><tr><td>0.0713</td><td>79.389</td><td>456.531</td><td>4.523</td><td>5.947</td><td>0.185</td></tr><tr><td>0.1069</td><td>79.389</td><td>456.571</td><td>4.717</td><td>6.028</td><td>0.187</td></tr><tr><td>0.1426</td><td>79.389</td><td>456.610</td><td>4.674</td><td>6.009</td><td>0.189</td></tr><tr><td rowspan="4">1080p</td><td>0.0356</td><td>79.389</td><td>2425.111</td><td>13.817</td><td>21.428</td><td>0.250</td></tr><tr><td>0.0713</td><td>79.389</td><td>2425.321</td><td>14.111</td><td>21.483</td><td>0.250</td></tr><tr><td>0.1069</td><td>79.389</td><td>2425.532</td><td>14.137</td><td>21.389</td><td>0.252</td></tr><tr><td>0.1426</td><td>79.389</td><td>2425.742</td><td>14.323</td><td>21.466</td><td>0.255</td></tr><tr><td rowspan="4">2K</td><td>0.0356</td><td>79.389</td><td>4293.353</td><td>22.447</td><td>35.987</td><td>0.312</td></tr><tr><td>0.0713</td><td>79.389</td><td>4293.725</td><td>22.501</td><td>36.022</td><td>0.313</td></tr><tr><td>0.1069</td><td>79.389</td><td>4294.097</td><td>22.880</td><td>35.897</td><td>0.315</td></tr><tr><td>0.1426</td><td>79.389</td><td>4294.469</td><td>23.209</td><td>35.905</td><td>0.318</td></tr></table>

ms. At 1080p, the encoding and decoding ranges span at most 0.506 and 0.094 ms, respectively. At 2K, the corresponding ranges are 0.762 and 0.125 ms.

These results show that increasing the active P-RVQ depth changes runtime only modestly relative to the shared backbone.

## D.2. GR-GIC Backend Comparison

We further compare three implementations of the same GR-GIC checkpoints to separate the effects of numerical precision and backend optimization.

At 512 × 768, the eager PyTorch FP32 implementation encodes in 30.453 ms and decodes in 44.708 ms. Relative to the official OneDC implementation, these times are 2.53× and 4.57× lower, respectively. The eager PyTorch FP32 implementation is also faster in both encoding and decoding than six of the eight displayed diffusion codecs, and this behavior remains consistent across the evaluated resolutions.

The CUDA FP16 implementation further reduces runtime relative to the PyTorch FP32 implementation. Across the evaluated resolutions, encoding is accelerated by 6.62× to 8.18×, while decoding is accelerated by 7.47× to 8.74×.

The eager FP32 results show that low latency is already present before the additional CUDA acceleration, while the optimized backend provides a further reduction in runtime.

## D.3. Additional Baseline Comparisons

CADC and PICO do not provide the corresponding executable models used in our comparison. Following the protocol in Sec. C, we therefore compare against their published curves or released numerical results where the metric definitions and rate ranges are compatible.

CADC. At the shared lowest Kodak rate of 0.03564 BPP, interpolation of the published CADC curves gives approximately 23.15 dB PSNR and 0.087 DISTS [35]. At the same rate, GR-GIC achieves 23.35 dB PSNR and 0.075 DISTS. This corresponds to a PSNR increase of approximately 0.20 dB and a DISTS reduction of approximately 0.012.

CADC reports LPIPS using AlexNet features, while our main evaluation uses VGG features. We therefore restrict this comparison to the aligned PSNR and DISTS metrics.

PICO. PICO reports VGG-based LPIPS on CLIC 2020 [38]. Over the common quality interval, GR-GIC obtains a PICO-anchored LPIPS BD-rate of −54.01%. This corresponds to a 54.01% average bitrate reduction at matched LPIPS over the evaluated common interval.

## D.4. Comparison with Lightweight Codecs

Figure 6 compares GR-GIC with ProGIC, EF-LIC, and CoD-Lite across all four datasets and metrics. At comparable rates, GR-GIC provides the strongest displayed LPIPS, DISTS, PSNR, and FID curves among these lightweight codecs.

The four GR-GIC operating points span approximately 0.036 to 0.143 BPP. The highest-rate GR-GIC point at 0.143 BPP uses a lower bitrate than the fifth EF-LIC point while achieving higher PSNR and lower DISTS and FID on every dataset. Its LPIPS is comparable or lower. The four GR-GIC operating points therefore cover the shared comparison range without requiring an additional higher-rate point.

## D.5. Additional Rate and Quality Results

Table 9 reports OneDC-anchored BD-rate for LPIPS, DISTS, PSNR, and FID on Kodak, Tecnick, DIV2K, and CLIC 2020. Entries without a valid monotone common interval are left unavailable according to the protocol in Sec. C. CoD-Lite and MS-ILLM are already included in the main Kodak and high-resolution comparisons, so their full entries are not repeated across all four metric blocks in this table.

Table 8. GR-GIC backend comparison on one A100. Latency is in ms and peak GPU memory is in GiB. Online E2E denotes one consecutive pair of in-memory encode and decode calls.
<table><tr><td>Resolution</td><td>Backend</td><td>Enc.</td><td>Dec.</td><td>Online E2E</td><td>Mem.</td></tr><tr><td rowspan="3"> $5 1 2 \times 7 6 8$ </td><td>PyTorch FP32</td><td>30.453</td><td>44.708</td><td>66.519</td><td>0.438</td></tr><tr><td>PyTorch FP16</td><td>15.439</td><td>20.298</td><td>37.784</td><td>0.222</td></tr><tr><td>CUDA FP16</td><td>4.597</td><td>5.983</td><td>10.353</td><td>0.186</td></tr><tr><td rowspan="3">1080p</td><td>PyTorch FP32</td><td>109.967</td><td>176.403</td><td>282.883</td><td>0.965</td></tr><tr><td>PyTorch FP16</td><td>32.895</td><td>57.600</td><td>86.817</td><td>0.465</td></tr><tr><td>CUDA FP16</td><td>14.097</td><td>21.442</td><td>35.567</td><td>0.252</td></tr><tr><td rowspan="3">2K</td><td>PyTorch FP32</td><td>185.468</td><td>312.454</td><td>495.019</td><td>1.446</td></tr><tr><td>PyTorch FP16</td><td>55.968</td><td>95.262</td><td>145.070</td><td>0.691</td></tr><tr><td>CUDA FP16</td><td>22.759</td><td>35.952</td><td>59.118</td><td>0.314</td></tr><tr><td rowspan="3">4K</td><td>PyTorch FP32</td><td>407.490</td><td>709.180</td><td>1114.092</td><td>2.909</td></tr><tr><td>PyTorch FP16</td><td>120.631</td><td>221.758</td><td>327.342</td><td>1.373</td></tr><tr><td>CUDA FP16</td><td>49.787</td><td>81.143</td><td>132.971</td><td>0.492</td></tr></table>

Across the twelve LPIPS, DISTS, and FID entries, GR-GIC achieves a negative OneDC-relative BD-rate on every dataset. The corresponding rate and quality curves in Figures 4 and 6 show the behavior over the evaluated operating ranges.

The table also places these reconstruction results in the context of model complexity. AEIC uses approximately 11.2× the parameters and 4.7× the FLOPs of GR-GIC. StableCodec uses approximately 13.4× the parameters and 14.9× the FLOPs. EF-LIC and ProGIC use fewer parameters and FLOPs than GR-GIC, while GR-GIC provides lower DISTS and FID over the reported comparisons.

## D.6. High-Resolution Runtime and Memory

Table 10 extends the runtime comparison to 1080p, 2K, and 4K images on one NVIDIA A100.

The released GLC interface reports analysis and synthesis without a complete bitstream path. Its forward execution takes (159.460, 259.762) ms at 1080p, (291.436, 488.817) ms at 2K, and (659.937, 1101.531) ms at 4K for analysis and synthesis, respectively. Under the complete CompressAI integration defined in Sec. C, the corresponding encoding and decoding times are (340.746, 475.738) ms, (638.989, 826.327) ms, and (1456.258, 1816.005) ms. Peak memory is 5.66, 9.71, and 20.76 GiB at the three resolutions.

Among the available implementations in Tab. 10, GR-GIC achieves the lowest encoding latency, decoding latency, and peak memory at all three resolutions. Relative to the second-lowest values, the reductions correspond to factors of 1.26× to 1.39× for encoding, 1.43× to 1.66× for decoding, and 2.58× to 4.27× for peak memory.

EF-LIC provides the second-lowest decoding latency and peak memory across the three resolutions. AEIC provides the second-lowest encoding latency at 2K and 4K. These results show that the fast inference of GR-GIC extends from Kodak resolution to high-resolution images.

## D.7. GR Inference Stability

Inference perturbation. We first evaluate the effect of the sampled perturbation at inference using the protocol in Sec. C.5. With the perturbation term removed, the decoder uses only the learned refinement center yˆ + δ.

In the same rate order as Tab. 2, the resulting LPIPS and FID pairs are (0.262, 54.8), (0.206, 36.4), (0.181, 32.5), and (0.162, 27.9).

Relative to the complete $\mathrm { M } _ { 4 }$ results, removing the sampled perturbation changes LPIPS by only 0.000 to 0.002 and increases FID by 0.2 to 0.4. Most of the gain of the complete GR configuration therefore remains when inference uses only the learned refinement center.

Without sampled perturbation at inference, $\mathrm { M } _ { 4 }$ retains FID reductions of 7.6, 4.8, 4.0, and 4.0 relative to deterministic $\mathrm { M _ { 2 } } .$ It also retains reductions of 4.7, 1.5, 1.3, and 1.6 relative to $\mathrm { M _ { 3 } }$ , which uses a channel-wise perturbation scale. LPIPS remains equal to or lower than both configurations at every evaluated rate.

These results support the interpretation in Sec. 4.3 that the adaptive perturbation primarily contributes through training regularization rather than through the particular random realization used at inference.

Decoder seed stability. We next vary only the decoder seed according to the protocol in Sec. C.5. Across ten seeds on Kodak, aggregate LPIPS varies by less than $1 0 ^ { - 4 }$ and FID by less than $\bar { 1 } 0 ^ { - 2 }$

![](images/4d9976eeb50645e89bba06decfb1a13964feb6dfef20e0cb3bcb8563efd40b27.jpg)  
Figure 6. Rate and reconstruction quality of GR-GIC and lightweight codecs on four datasets. GR-GIC uses four operating points, while EF-LIC uses five. Lower LPIPS, DISTS, and FID and higher PSNR indicate better reconstruction. FID uses a base-2 logarithmic vertical axis.

The transmitted indices and payload are identical across these runs. The small metric variation shows that the reported aggregate reconstruction performance is stable across the evaluated GR realizations.

FID repeatability. We also repeat the Kodak FID computation four times with the reconstructed images and patch extraction fixed. All four runs produce identical FID values. This confirms that the variation measured above comes from the GR realization rather than repeated execution of the FID estimator.

## E. Additional Visual Comparisons

This section provides additional visual comparisons of GR-GIC on all four evaluation datasets. Each figure places comparisons with high-capacity diffusion codecs above those with lightweight baselines and reports LPIPS and BPP for the displayed reconstructions. Two further panels extend the Tecnick and CLIC 2020 coverage under the same LPIPS and BPP annotation protocol.

Table 9. Pairwise OneDC-anchored BD-rate (%). Bold and underline mark the two lowest displayed values per dataset.
<table><tr><td>Metric</td><td>Family</td><td>Method</td><td>Kodak</td><td>Tecnick</td><td>DIV2K</td><td>CLIC 2020</td></tr><tr><td rowspan="10">LPIPS</td><td rowspan="5">Diffusion</td><td>OneDC StableCodec</td><td>0.00 8.03</td><td>0.00 32.28</td><td>0.00 27.06</td><td>0.00 36.13</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AEIC</td><td>41.65</td><td>34.58</td><td>25.27</td><td>32.18</td></tr><tr><td>DiT-IC</td><td>35.99</td><td>61.79</td><td>49.18</td><td>52.00</td></tr><tr><td>OSCAR</td><td>127.15</td><td>225.78</td><td>201.57</td><td></td></tr><tr><td rowspan="5">VAE-based</td><td>GLC</td><td>21.53</td><td>47.87</td><td>33.19</td><td>39.53</td></tr><tr><td>DLF</td><td>20.87</td><td>54.78</td><td>47.02</td><td>45.66</td></tr><tr><td>ProGIC</td><td>36.47</td><td>56.48</td><td>42.04</td><td>84.07</td></tr><tr><td>EF-LIC</td><td>-3.33 17.06</td><td>15.12</td><td>3.51 26.35</td><td>24.62 33.11</td></tr><tr><td>RDVQ</td><td>-23.19</td><td>47.25 -11.27</td><td>-20.24</td><td>-18.39</td></tr><tr><td rowspan="5"></td><td rowspan="5">Diffusion</td><td>GR-GIC</td><td></td><td></td><td></td><td>0.00</td></tr><tr><td>OneDC StableCodec</td><td>0.00 6.77</td><td>0.00 31.27</td><td>0.00 9.38</td><td>24.27</td></tr><tr><td>AEIC</td><td>-16.71</td><td>-20.02</td><td>-30.19</td><td>-21.29</td></tr><tr><td>DiT-IC</td><td>-6.44</td><td>-0.28</td><td>-16.61</td><td>-10.82</td></tr><tr><td>OSCAR</td><td>131.50</td><td>240.68</td><td>223.40</td><td>339.84</td></tr><tr><td rowspan="8">VAE-based</td><td rowspan="6"></td><td>GLC</td><td>54.70</td><td>72.21</td><td>50.63</td><td>70.55</td></tr><tr><td>DLF</td><td>20.19</td><td>62.90</td><td>56.62</td><td>64.56</td></tr><tr><td>ProGIC</td><td>80.41</td><td>126.30</td><td>78.51</td><td>172.95</td></tr><tr><td>EF-LIC</td><td>10.05</td><td>44.93</td><td>13.25</td><td>69.31</td></tr><tr><td>RDVQ</td><td>9.87</td><td>27.16</td><td>9.10</td><td>27.14</td></tr><tr><td>GR-GIC</td><td>-37.71</td><td>-28.12</td><td>-42.46</td><td>-31.53</td></tr><tr><td>OneDC</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="10">PSNR</td><td rowspan="5">Diffusion</td><td>StableCodec</td><td>-17.10</td><td>41.35</td><td>31.65</td><td>-12.35</td></tr><tr><td>AEIC</td><td>2.11</td><td>1.78</td><td>-4.44</td><td></td></tr><tr><td>DiT-IC</td><td>33.24</td><td>50.41</td><td>40.19</td><td>26.63</td></tr><tr><td>OSCAR</td><td>154.09</td><td>248.22</td><td>195.61</td><td>338.47</td></tr><tr><td>GLC</td><td>56.04</td><td>72.28</td><td>61.07</td><td>79.51</td></tr><tr><td rowspan="5">VAE-based</td><td>DLF</td><td>42.73</td><td>78.89</td><td>63.48</td><td>67.35</td></tr><tr><td>ProGIC</td><td>49.84</td><td>105.02</td><td>72.35</td><td>116.69</td></tr><tr><td>EF-LIC</td><td>11.49</td><td>61.07</td><td>39.83</td><td>59.45</td></tr><tr><td>RDVQ</td><td></td><td>70.23</td><td></td><td>70.88</td></tr><tr><td></td><td>62.13</td><td></td><td>60.92</td><td></td></tr><tr><td rowspan="9">Diffusion FID</td><td rowspan="9"></td><td>GR-GIC</td><td>-22.30</td><td>19.60</td><td>-0.41</td><td>16.48 0.00</td></tr><tr><td>OneDC StableCodec</td><td>0.00</td><td>0.00</td><td>0.00</td><td>-28.75</td></tr><tr><td>AEIC</td><td>-4.45 7.16</td><td>-5.58 -38.95</td><td>-26.77 -49.80</td><td>-55.10</td></tr><tr><td>DiT-IC</td><td>56.47</td><td>41.17</td><td>30.56</td><td>44.53</td></tr><tr><td>OSCAR</td><td>159.92</td><td>222.65</td><td></td><td></td></tr><tr><td>GLC</td><td>82.95</td><td>54.94</td><td>50.70</td><td>90.98</td></tr><tr><td>DLF</td><td>70.80</td><td>73.87</td><td>66.43</td><td>66.97</td></tr><tr><td>ProGIC</td><td>156.32</td><td>267.15</td><td>225.46</td><td>389.82</td></tr><tr><td>EF-LIC</td><td>49.01</td><td>143.61</td><td>138.95</td><td>257.90</td></tr><tr><td rowspan="3"></td><td>RDVQ</td><td>49.51</td><td>55.05</td><td>40.18</td><td>59.78</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GR-GIC</td><td>-22.14</td><td>-32.07</td><td>-45.68</td><td>-34.26</td></tr></table>

Table 10. High-resolution encoding and decoding latency (ms) and peak GPU memory (GiB) on one A100. Bold and underline mark the two best results.
<table><tr><td rowspan="2">Method</td><td colspan="3">1080p</td><td colspan="3">2K</td><td colspan="3">4K</td></tr><tr><td>Enc.</td><td>Dec.</td><td>Mem.</td><td>Enc.</td><td>Dec.</td><td>Mem.</td><td>Enc.</td><td>Dec.</td><td>Mem.</td></tr><tr><td>ResULIC</td><td>481.75</td><td>3702.45</td><td>20.38</td><td></td><td>&gt; 10 s</td><td></td><td></td><td>&gt; 10 s</td><td></td></tr><tr><td>StableCodec</td><td>641.76</td><td>1657.43</td><td>21.80</td><td>1085.39</td><td>3161.54</td><td>30.65</td><td></td><td>&gt; 10 s</td><td></td></tr><tr><td>RDEIC</td><td>683.20</td><td>2296.58</td><td>13.55</td><td>1436.00</td><td>5963.77</td><td>20.17</td><td></td><td>&gt; 10 s</td><td></td></tr><tr><td>OneDC</td><td>303.33</td><td>742.06</td><td>12.81</td><td>585.72</td><td>1554.82</td><td>18.86</td><td>1542.18</td><td>4846.63</td><td>35.38</td></tr><tr><td>OSCAR</td><td>513.20</td><td>1123.38</td><td>24.75</td><td>2382.55</td><td>2719.79</td><td>79.44</td><td></td><td></td><td>&gt; 80</td></tr><tr><td>DiT-IC</td><td>480.22</td><td>750.38</td><td>10.98</td><td>718.17</td><td>1111.59</td><td>14.47</td><td>1581.91</td><td>2448.49</td><td>25.41</td></tr><tr><td>AEIC</td><td>25.15</td><td>682.50</td><td>6.95</td><td>30.69</td><td>1943.15</td><td>9.87</td><td>62.70</td><td>9303.96</td><td>17.83</td></tr><tr><td>CoD-Lite</td><td>42.37</td><td>60.47</td><td>2.45</td><td>73.61</td><td>105.09</td><td>3.97</td><td>163.43</td><td>236.05</td><td>8.33</td></tr><tr><td>MS-ILLM</td><td>350.85</td><td>379.01</td><td>2.99</td><td>516.47</td><td>601.18</td><td>4.87</td><td>1305.93</td><td>1613.82</td><td>9.94</td></tr><tr><td>Control-GIC</td><td>610.76</td><td>2186.30</td><td>69.99</td><td></td><td></td><td>&gt; 80</td><td></td><td></td><td>&gt; 80</td></tr><tr><td>GLC</td><td>340.746</td><td>475.738</td><td>5.66</td><td>638.989</td><td>826.327</td><td>9.71</td><td>1456.258</td><td>1816.005</td><td>20.76</td></tr><tr><td>DLF</td><td>915.42</td><td>1416.49</td><td>10.00</td><td>1380.00</td><td>2153.39</td><td>12.61</td><td>3001.08</td><td>5122.21</td><td>22.35</td></tr><tr><td>ProGIC</td><td>28.40</td><td>55.00</td><td>1.07</td><td>38.63</td><td>80.38</td><td>1.46</td><td>78.26</td><td>145.24</td><td>2.58</td></tr><tr><td>RDVQ</td><td>704.64</td><td>809.99</td><td>6.01</td><td>953.80</td><td>1130.17</td><td>10.08</td><td>8575.63</td><td>8655.58</td><td>20.07</td></tr><tr><td>EF-LIC</td><td>19.53</td><td>35.62</td><td>0.65</td><td>31.58</td><td>55.50</td><td>1.04</td><td>65.24</td><td>116.28</td><td>2.10</td></tr><tr><td>GR-GIC</td><td>14.10</td><td>21.44</td><td>0.25</td><td>22.76</td><td>35.95</td><td>0.31</td><td>49.79</td><td>81.14</td><td>0.49</td></tr></table>

![](images/996b7b32f6dd9d2e039e1966997c200d83f60e1a138b68ea4b4098594822817b.jpg)  
Figure 7. Additional Kodak visuals with LPIPS and BPP labels, where lower is better.

![](images/f2d5be3adbc387b71e0d85bc01d75019c55034bc4d8b7e6e074a0ae99e415f23.jpg)  
Figure 10. Additional CLIC 2020 visuals with LPIPS and BPP labels, where lower is better.

![](images/f3970a66a948d892274a65d2feeb4fdda23b4187722436ee7e6acafeff570325.jpg)  
Figure 12. Additional CLIC 2020 visuals with LPIPS and BPP labels, where lower is better.