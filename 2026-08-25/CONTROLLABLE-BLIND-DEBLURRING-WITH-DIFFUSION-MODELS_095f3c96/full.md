# CONTROLLABLE BLIND DEBLURRING WITH DIFFUSION MODELS

Imane Si Salah<sup>★†</sup>, Emile Cribelier<sup>★</sup>, Thomas Veit<sup>★</sup>, WolfHauser<sup>★</sup>, Arthur Leclaire<sup>†</sup>

DxO Labs, France<sup>★</sup>

LTCI, Télécom Paris, IP Paris, France<sup>†</sup>

## ABSTRACT

Image acquisition with a camera involves several degradations due to the optical system, sensor, or low-level processing steps. We address blind deblurring in professional photography: we aim to invert unknown isotropic blur without knowledge of the degradation kernel. For such inverse problems,where some high-frequency information is lost, it is challenging to use generative models to produce details that are both photo-realistic and faithful to the input. We propose SuperSharpen, a diffusion-based blind deblurring method offering explicit control over restoration strength through a blur measure. We compare two conditioning strategies: a ControlNet-style adapter on a frozen backbone, and full finetuning of the diffusion prior. Our experiments show that finetuning achieves better fidelity with fewer hallucinated details. We validate our approach on synthetic and real-world blur, demonstrating improved perceptual quality and controllable restoration strength.

Index Terms— Blind deblurring, diffusion models, controllability, generative priors, image restoration

## 1. INTRODUCTION

Many physical and environmental factors during image capture induce various forms of blur, which translate into a loss of high-frequency information. These include optical imperfections in lenses, motion or camera shake, and low-light conditions that require longer exposure times. In addition, some preprocessing steps, such as RAW denoising, may further smooth out textures and fine details, introducing additional blur. In this work, we address mild, spatially invariant perceived blur, including both slight defocus and smoothing introduced by denoising/ISP processing, while excluding motion blur and non-uniform blur.

Image restoration is a widely studied problem, and existing solutions can be classified as blind and non-blind [1]. Non-blind methods assume that the degradation model is known, while blind approaches must estimate both the degradation and the clean image. Traditionally, image restoration has been formulated as an optimization problem balancing a data-fidelity term with a regularization term. With the emergence of deep learning, learning-based solutions arose where neural networks are trained to learn an end-to-end mapping from the degraded image space to the clean image space [2, 3]. However, these methods often underperform in real-world scenarios due to the distribution shift between real and training data [4]. Moreover, by minimizing a pixel-wise loss, they tend to predict an average of all plausible solutions, resulting in overly smoothed outputs that lack the fine details required for our restoration task.

Generative models [5, 6] have emerged as strong candidates for image restoration as they constitute powerful priors to model natural images. The main challenge is then to condition sampling with a degraded observation. In this work, we focus on diffusion models (DMs) [6, 7]. DMs generate images with a probabilistic denoising process, which iteratively refines pure white noise into a clean image. Diffusionbased image restoration can be broadly divided into two categories. The first one [8, 9, 10], based on a pretrained diffusion scheme, consists in guiding the iterative sampling process using a kind of data consistency to the degraded input. One major asset of these methods is to avoid task-specific retraining. However, they necessitate many sampling steps to reinforce the fidelity to the degraded observation. This limits the use of accelerated diffusion models like consistency models [11]. The second category involves end-to-end retraining to condition the DM at each step by the low-quality input. This is done either by 1) training an adapter network like ControlNet-style [12] that will be plugged into a frozen pretrained prior [13, 14, 15], or 2) by finetuning the diffusion denoiser to condition on the degraded input [16, 17, 18]. Although these approaches require additional training, they often reach higher performance in terms of perceptual quality. In this work, we investigate both these conditioning strategies on a latent diffusion model [7] and compare them in the context of blind image deblurring.

In addition, most existing diffusion-based restoration methods do not provide an explicit control over the desired generative restoration level. We address this limitation by introducing the “blur measure” as conditional input: at training time, it represents the estimated blur level of the used degradation, while at inference it represents the quantity of blur that one wishes to remove.

Our contributions can be summarized as follows:

• We propose SuperSharpen (SupS), a blind deblurring model based on latent diffusion, which takes as input the degraded image and a so-called blur measure (BM), which allows for explicit control over the restoration strength at inference.

• We empirically compare two conditioning methods of the diffusion backbone (ControlNet-based and finetuning) and show that finetuning achieves better fidelity to the degraded input with fewer hallucinated details.

• We validate our method on both synthetic and real-world blur, including blur arising from optical imperfections and RAW processing.

![](images/aae83907a11ded91a8610e3251389f8525e2e101c5e8aa42191666fa87a8dc18.jpg)

![](images/c0164caeaff029b37b83da77bbf408d28ef76c5808b9075874363c5389947909.jpg)

(b) SupS-FT training setup  
![](images/412ca0c66c31aa1091f8375bed3989287bbe594a452e4d914c314a5adb9f5ae4.jpg)  
(c) SupS-FT inference setup  
Fig. 1: We present the training setups for (a) ControlNetbased conditioning (SupS-CN) and (b) finetuning-based conditioning (SupS-FT). Both models rely on a conditional scorebased latent denoiser $\epsilon _ { \theta }$ that takes as input the encoded Low Quality image and the corresponding blur measure in addition to the encoded Ground Truth to which we add noise $\mathbf { z } _ { t }$ In (a) only the ControlNet adapters are trained ( ) while the diffusion backbone is frozen ( ). In (b) the entire diffusion denoiser is finetuned ( ). Noting that in both training setups, the VAE encoder E is frozen ( ). In (c), we illustrate the inference setup only for SupS-FT which generates the high-quality image by iterative use of $\epsilon _ { \theta }$ conditioned by the degraded image and a user-defined blur measure to control the restoration strength. Notice that at training � is sampled randomly while at inference, � is decreased from � to 0.

![](images/7d1b3afc53da502c622095da19231a27afd71bd219dc3ee284b34caf357f2a28.jpg)  
Fig. 2: Blur measure computation setup. We estimate the effective blur kernel of the degradation pipeline by computing its impulse response from multiple impulses located at different spatial locations. We then fit an isotropic Gaussian kernel to the average impulse response and define our blur measure as the standard deviation of the fitted Gaussian.

## 2. BLIND DEBLURRING NETWORK

## 2.1. Preliminaries: Diffusion models

Our method builds on a Stable Diffusion (SD) [7] backbone, a latent diffusion model (LDM) that uses a variational autoencoder (VAE) to operate in a lower-dimensional latent space. Let $\mathbf { x } _ { 0 } ~ \in ~ \mathbb { R } ^ { \bar { H } \times W \times 3 }$ be a clean image and $\begin{array} { r } { \mathbf { z } _ { 0 } = \mathcal { E } ( \mathbf { \bar { x } } _ { 0 } ) \in \mathbb { R } ^ { \frac { H } { s } \times \frac { W } { s } \times c } } \end{array}$ be its latent representation, where E is the encoder, � is the spatial downsampling factor, and � is the number of latent channels. During the forward diffusion process, Gaussian noise is progressively added to the latent code over � timesteps. At timestep $t \in \{ 1 , . . . , T \}$ , the noisy latent is given by ${ \bf z } _ { t } = \sqrt { \bar { \alpha } _ { t } } { \bf z } _ { 0 } + \sqrt { \bar { \beta } _ { t } } { \bf \epsilon } _ { t }$ , where $\boldsymbol { \epsilon } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , I _ { d } )$ and $\bar { \alpha } _ { t } , \bar { \beta } _ { t }$ depend on a predefined noise schedule. A neural network $\boldsymbol { \epsilon } _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , t )$ with parameters $\theta$ is trained to predict the noise $\epsilon _ { t }$ by minimizing

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t , { \mathbf { z } _ { 0 } } , \epsilon _ { t } } \left[ \left\| \epsilon _ { t } - \epsilon _ { \theta } ( { \mathbf { z } _ { t } } , t ) \right\| _ { 2 } ^ { 2 } \right] .\tag{1}
$$

At inference, the reverse sampling process iteratively denoises from pure noise $\tilde { \mathbf { z } } _ { T } ~ \sim ~ N ( \mathbf { 0 } , I _ { d } )$ to generate a clean latent $\tilde { \mathbf { z } } _ { 0 }$ , which is then decoded to produce the output image, see the details in [6].

## 2.2. Diffusion model conditioning

While the DDPM formulation above describes an unconditional generative model, we address blind deblurring as generation conditioned by a blurry input image. This conditioning is learned using pairs of high-quality (HQ) images and their corresponding low-quality (LQ) images. During training, each LQ image is obtained by applying the same degradation pipeline as used in Real-ESRGAN [19]. Let ${ \bf z } _ { L Q } ~ =$ $\mathcal { E } ( \mathbf { x } _ { L Q } )$ denote the latent code of the degraded image and $\mathbf { z } _ { t }$ the noisy latent at time �. We investigate two types of conditioning categories for injecting the degraded input into the diffusion model.

![](images/41b8db74f28f7c45c1b82b4148122b8f2260e22e9e36751122906bb349748efa.jpg)  
Fig. 3: Qualitative comparison on images with synthetic blur. The sharpest results are obtained with DiffBIR, SupS-CN and SupS-FT, the latter producing less unplausible structures.

The first category, SuperSharpen ControlNet (SupS-CN), uses ControlNet layers [12]. As shown in Fig. 1a, the ControlNet adapter is initialized as a trainable copy of the encoder and middle blocks of the diffusion denoiser U-Net, while the pretrained diffusion backbone is kept frozen. At each diffusion step, the conditioning branch takes the concatenation of $\mathbf { z } _ { L Q }$ and $\mathbf { z } _ { t }$ , extracts joint features, and injects them into the frozen backbone through residual connections. SupS-CN is inspired by DiffBIR [13], which uses a two-stage restoration model: a preprocessing module followed by a ControlNetbased generative module. In comparison, our approach omits the preprocessing network.

The second category, SuperSharpen Finetuned (SupS-FT), shown in Fig. 1b, directly integrates the conditioning input into the diffusion backbone and therefore requires finetuning the entire diffusion denoiser. We extend the diffusion U-Net to accept the concatenation of $\mathbf { z } _ { L Q }$ and $\mathbf { z } _ { t }$

In both settings, the models are trained with the standard noise-prediction objective (1), extended to take the LQ latent as an additional input:

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t , { \mathbf { z } _ { 0 } } , \epsilon _ { t } } \left[ \left\| \epsilon _ { t } - \epsilon _ { \boldsymbol { \theta } } ( { \mathbf { z } _ { t } } , t , { \mathbf { z } _ { L Q } } ) \right\| _ { 2 } ^ { 2 } \right] .\tag{2}
$$

The inference is run with the backward latent diffusion process, with the $\mathbf { z } _ { L Q }$ being reinjected in � at each reverse diffusion step, as illustrated in Fig. 1c.

## 2.3. Blur measure estimation

In addition to blind deblurring, we aim to provide explicit control over the restoration strength. For that, we introduce a blur measure $\mathsf { B M } \in \mathbb { R } _ { + }$ , used as an additional conditioning input. During training, for each training sample, we compute the BM value as an estimate of the blur introduced by the degradation pipeline (see below). During inference on real images, BM can be set by the user to adjust the restoration strength.

Our restoration model is trained with the degradation pipeline used to train RealESRGAN [19], which combines various types of simulated degradations (isotropic and nonisotropic blur, resizing, noise addition, and JPEG compression), which is crucial to increase the model robustness to real-world degradations. But since the degradation does not consist solely of blur, we propose to estimate the amount of blur it introduces by fitting an isotropic Gaussian blur kernel to the effective impulse response of the degradation pipeline. More precisely, given the degradation operator �, we average the responses $r _ { p } = A ( \delta _ { p } )$ obtained at pre-defined positions $p \in \mathcal { P }$ taken on a 21 × 21 grid (because � is not exactly translation-invariant). For each ${ \boldsymbol { p } } ,$ the response is recentered using a shift $S _ { p }$ computed with the center of mass of $r _ { p } .$ We average over � to get the estimated impulse response $\begin{array} { r } { \boldsymbol { h } = \frac { 1 } { | \mathcal { P } | } \sum _ { \boldsymbol { p } \in \mathcal { P } } \boldsymbol { S } _ { \boldsymbol { p } } ( \boldsymbol { r } _ { \boldsymbol { p } } ) } \end{array}$ . Finally, we fit an isotropic Gaussian kernel $g _ { \sigma }$ to get the estimated blur measure of the degradation

LQ  
ResShift  
PASD  
SeeSR  
DiffBIR  
SupS-CN  
SupS-FT  
![](images/6fa0813deb8393a5cfe54025e80f227f26b1e7ec2e1f150b7f2276e13705c4b0.jpg)  
Fig. 4: Qualitative comparison on real blurry images. Compared to the other methods, our SupS models attain a better compromise in terms of photo-realism and fidelity to the input.

$$
{ \widehat { \mathsf { B M } } } ( A ) = \arg \operatorname* { m i n } _ { \sigma \in [ \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ] } { \mathsf { M S E } } \left( h , g _ { \sigma } \right) ,\tag{3}
$$

This procedure is illustrated in Fig. 2. While BM is derived from a blur estimate during training, at inference, however, it does not need to match the exact physical blur present in the input image. Instead, it serves as a user-adjustable control signal: higher values lead to stronger detail generation, allowing users to tune the restoration strength according to their preference.

## 3. EXPERIMENTS

Model architecture. We use a Stable Diffusion (SD2.1) model as our generative backbone. We evaluate two conditioning methods: (i) ControlNet-based conditioning: we freeze the SD2.1 U-Net and train only the ControlNet adapter layers that take the concatenation of the noisy and degraded latents in addition to the blur measure scalar extended to the latent dimensions. (ii) Backbone finetuning: we finetune the SD2.1 U-Net and extend its first layer to take the degraded input image and the blur measure as additional conditioning.

Training and data. We train on the LSDIR dataset [20], where we generate paired training samples using a degradation pipeline similar to the one used for Real-ESRGAN [19] but with a scale factor of 1 (since the objective is to restore images at the same resolution as the input) and with lower degradation parameters (because our network focuses on slight degradations). In addition to synthesizing the LQ images, we make another pass through the degradation pipeline to compute the blur measure BM as described in Section 2.3. For that, the input impulses have resolution $7 5 \times 7 5$ , and we use $\vert \mathcal { P } \vert = 4 4 1$ impulses located on the central $2 1 \times 2 1$ grid. The search interval for � is set to $[ \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ] = [ 0 . 1 , 1 0 . 0 ]$ We input the scalar BM value as a constant map concatenated with the latent $\mathbf { z } _ { L Q }$

To make the model produce outputs with varying sharpness levels, we employ a data augmentation scheme where HQ images are convolved with a Gaussian kernel before degradation. Initially, this augmentation is applied to all training samples; then, as training progresses, to only 10% of the dataset, forcing the model to learn to generate sharp outputs.

We train SupS-CN (resp. SupS-FT) for 270� (resp. 380�) iterations using the AdamW optimizer with batches of 24 crops of size $5 1 2 \times 5 1 2$ . The learning rate is set to 10<sup>−4</sup> and reduced to $1 0 ^ { - 5 }$ for the last 100� iterations. Training is performed on 4×H100 GPUs during 13 days.

Evaluation on synthetic data. We evaluate on images from Urban100 [21], synthetically degraded using a Gaussian blur kernel with std drawn in [0.7, 7.5] and additive Gaussian noise with std drawn in [0, 7]. While we train with the Real-ESRGAN degradation pipeline to learn robustness across a broad degradation space, we test only with isotropic Gaussian blur, which is a good single-parameter baseline.

Table 1: Quantitative results on synthetically degraded images from Urban100 dataset. Best and second best results are highlighted in bold and underlined, respectively.
<table><tr><td></td><td colspan="3">Full-reference</td><td colspan="3">No-reference</td></tr><tr><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>MUSIQ↑</td><td>MANIQA ↑</td><td>CLIP-IQA ↑</td></tr><tr><td>HQ</td><td>一</td><td>1.00</td><td>0.00</td><td>70.4433</td><td>0.5534</td><td>0.6762</td></tr><tr><td>DiffBIR</td><td>18.3051</td><td>0.5284</td><td>0.2278</td><td>71.6563</td><td>0.5986</td><td>0.7107</td></tr><tr><td>PASD</td><td>21.9344</td><td>0.6518</td><td>0.2138</td><td>68.9798</td><td>0.4850</td><td>0.5904</td></tr><tr><td>SeeSR</td><td>20.1731</td><td>0.5808</td><td>0.2463</td><td>70.4701</td><td>0.5529</td><td>0.6541</td></tr><tr><td>ResShift</td><td>19.7569</td><td>0.4976</td><td>0.4138</td><td>48.0120</td><td>0.2960</td><td>0.4199</td></tr><tr><td>SupS-CN</td><td>19.7659</td><td>0.5549</td><td>0.2020</td><td>71.7406</td><td>0.6493</td><td>0.7299</td></tr><tr><td>SupS-FT</td><td>20.7670</td><td>0.6138</td><td>0.1714</td><td>71.0819</td><td>0.6112</td><td>0.7102</td></tr></table>

We compare our method to several recent generativebased restoration methods: DiffBIR [13], PASD [14], SeeSR [15], and ResShift [22]. All these methods are based on DMs. DiffBIR, PASD, and SeeSR use a two-stage pipeline with a preprocessing module followed by a ControlNet-based generative module, while ResShift introduces a residual-shifting framework for efficient restoration. Notice that these methods are designed for super-resolution but can be used with a scale factor of 1. We report both full-reference metrics (PSNR, SSIM [23], LPIPS [24]) and no-reference metrics (MUSIQ [25], MANIQA [26], CLIP-IQA [27]).

Quantitative results are shown in Table 1. SupS-CN achieves the highest no-reference scores among all methods. Interestingly, most methods even surpass the HQ for these no-reference metrics. SupS-FT achieves the best LPIPS, indicating good perceptual fidelity. However, it underperforms in terms of PSNR and SSIM compared to PASD, whose training includes a pixel-wise loss in its degradation removal module. This suggests that finetuning the diffusion backbone allows for better fidelity to the degraded input, compared to ControlNet, since it is less dependent on a frozen prior.

Qualitative results in Fig. 3 confirm these observations: SupS-FT produces details that are more consistent with the HQ image, while SupS-CN produces images with similar sharpness level but less accurate textures. This is why we favored SupS-FT over SupS-CN for the next assessment on real images. From Fig. 3, we also observe that the results obtained with our models and with DiffBIR are significantly better than those of the other methods. Moreover, SupS-FT introduces fewer high-frequency artifacts than DiffBIR.

Evaluation on real data. We evaluate on images from Adobe5K [28], a dataset of RAW images captured by real SLR cameras. These images already exhibit some lens blur from the capture process, and are further processed using

LQ  
BM=0.7  
BM=5.0  
BM=8.0  
![](images/5282c457636bffd27e6535eec47d6da91400874d697dcd1f4daeb0cd77a07320.jpg)  
Fig. 5: Qualitative results on real images with different blur measure (BM) values. We see that higher BM values lead to stronger correction and sharper images.

DxO PhotoLab DeepPrime [29], which performs joint denoising and demosaicing but tends to smooth fine details as a side effect. We select crops where the perceived blur is approximately spatially uniform. We focus on this benchmark because most deblurring datasets primarily address motion blur or non-uniform defocus blur, while our method targets spatially uniform blur. Since no ground truth sharp images are available, we provide only qualitative evaluation.

As shown in Fig. 4, ResShift and PASD fail to recover fine details, producing outputs that remain perceptually blurry. SeeSR does not generate enough content, resulting in images with mostly flat surfaces. DiffBIR generates sharper textures but introduces details that are not faithful to the input. In contrast, our SupS models achieve both detail generation and fidelity to the input, producing more natural outputs. This is consistent with our observations on synthetic data.

On real images, we do not have access to the exact blur level, and therefore we manually adjust the blur measure BM to control the generation strength. As illustrated in Fig. 5, varying BM effectively controls the amount of details generated, allowing users to adjust the balance between fidelity and generated content. It is worth noting that the BM value leading to perceptually optimal sharpness may not correspond to a physical blur kernel size.

## 4. CONCLUSION

We proposed SuperSharpen, a diffusion-based blind deblurring method with controllable restoration strength via a blur measure. We compared ControlNet conditioning with full finetuning. Finetuning leads to better fidelity to the degraded input while ControlNet produces equally sharp but less faithful results. Our method generalizes well to real-world blur from optical imperfections and RAW processing. However, it is restricted to spatially uniform blur, may produce artifacts on sensitive content such as text and faces, and does not yet preserve identity when the blur measure is close to zero.

## 5. REFERENCES

[1] K. Zhang, W. Ren, W. Luo, W.-S. Lai, B. Stenger, M.- H. Yang, and H. Li, “Deep image deblurring: A survey,” IJCV, 2022.

[2] X. Tao, H. Gao, X. Shen, J. Wang, and J. Jia, “Scalerecurrent network for deep image deblurring,” in CVPR, 2018.

[3] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, “Swinir: Image restoration using swin transformer,” in ICCV, 2021.

[4] H. Wei, C. Ge, X. Qiao, and P. Deng, “Rethinking blur synthesis for deep real-world image deblurring,” arXiv preprint arXiv:2209.13866, 2022.

[5] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in NeurIPS, 2014.

[6] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.

[7] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022.

[8] J. Choi, S. Kim, Y. Jeong, Y. Gwon, and S. Yoon, “Ilvr: Conditioning method for denoising diffusion probabilistic models,” arXiv preprint arXiv:2108.02938, 2021.

[9] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye, “Diffusion posterior sampling for general noisy inverse problems,” arXiv preprint arXiv:2209.14687, 2022.

[10] Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, and L. Van Gool, “Denoising diffusion models for plugand-play image restoration,” in CVPR, 2023.

[11] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in ICML, 2023.

[12] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in ICCV, 2023.

[13] X. Lin, J. He, Z. Chen, Z. Lyu, B. Dai, F. Yu, Y. Qiao, W. Ouyang, and C. Dong, “Diffbir: Toward blind image restoration with generative diffusion prior,” in ECCV, 2024.

[14] T. Yang, R. Wu, P. Ren, X. Xie, and L. Zhang, “Pixel-aware stable diffusion for realistic image superresolution and personalized stylization,” in ECCV, 2024.

[15] R. Wu, T. Yang, L. Sun, Z. Zhang, S. Li, and L. Zhang, “Seesr: Towards semantics-aware real-world image super-resolution,” in CVPR, 2024.

[16] C. Saharia, W. Chan, H. Chang, C. Lee, J. Ho, T. Salimans, D. Fleet, and M. Norouzi, “Palette: Image-toimage diffusion models,” in SIGGRAPH, 2022.

[17] J. Whang, M. Delbracio, H. Talebi, C. Saharia, A. G. Dimakis, and P. Milanfar, “Deblurring via stochastic refinement,” in CVPR, 2022.

[18] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, “Image super-resolution via iterative refinement,” TPAMI, 2022.

[19] X. Wang, L. Xie, C. Dong, and Y. Shan, “Realesrgan: Training real-world blind super-resolution with pure synthetic data,” in ICCV, 2021.

[20] Y. Li, K. Zhang, J. Liang, J. Cao, C. Liu, R. Gong, Y. Zhang, H. Tang, Y. Liu, D. Demandolx, et al., “LS-DIR: A large scale dataset for image restoration,” in CVPR, 2023.

[21] J.-B. Huang, A. Singh, and N. Ahuja, “Single image super-resolution from transformed self-exemplars,” in CVPR, 2015.

[22] Z. Yue, J. Wang, and C. C. Loy, “Resshift: Efficient diffusion model for image super-resolution by residual shifting,” in NeurIPS, 2023.

[23] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” TIP, 2004.

[24] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018.

[25] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “Musiq: Multi-scale image quality transformer,” in ICCV, 2021.

[26] S. Yang, T. Wu, S. Shi, S. Lao, Y. Gong, M. Cao, J. Wang, and Y. Yang, “Maniqa: Multi-dimension attention network for no-reference image quality assessment,” in CVPR, 2022.

[27] J. Wang, K. C. K. Chan, and C. C. Loy, “Exploring clip for assessing the look and feel of images,” in AAAI, 2023.

[28] V. Bychkovsky, S. Paris, E. Chan, and F. Durand, “Learning photographic global tonal adjustment with a database of input/output image pairs,” in CVPR, 2011.

[29] “DxO DeepPRIME,” https://www.dxo.com/ technology/deepprime/.