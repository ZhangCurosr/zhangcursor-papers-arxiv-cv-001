# Denoising Textures is a Different Ballgame Altogether!

Saptarshi Mandal<sup>⋆</sup>, Oindrila Haldar<sup>⋆</sup>, Chirayata Bhattacharya<sup>⋆</sup>, Chandra Sekhar Seelamantula Department of Electrical Engineering,

Indian Institute of Science, Bangalore, India

{saptarshiman, oindrilah, chirayatab}@alum.iisc.ac.in, css@iisc.ac.in

Zhiwei Xu, Thrasyvoulos N. Pappas Department of Electrical and Computer Engineering Northwestern University, Evanston, Illinois, USA zhiweiXu2050@u.northwestern.edu t-pappas@northwestern.edu

Abstract—The human brain effortlessly deciphers colors and textures from visual stimuli and reconstructs the world around us, yet preserving such details remains challenging in image denoising. While significant progress has been made in generic image denoising, the problem of texture denoising has remained relatively under-explored. Fine details, such as intricate patterns and repetitive structures, often suffer from over-smoothing even in deep learning based approaches for denoising. In this paper, we analyze the behavior of state-of-the-art (SOTA) denoisers on texture datasets and show that performance improvements are possible only by fine-tuning the denoisers on the texture datasets. Surprisingly, fine-tuning on texture datasets result in the models catastrophically forgetting the original dataset. These findings highlight the need for specialized treatment of textures in denoising. Since denoising quality directly impacts downstream tasks such as generative modelling, we argue that dedicated textureaware denoisers are essential for faithful texture reconstruction and synthesis.

## I. INTRODUCTION

Most imaging techniques inevitably introduce noise, which is further exacerbated by environmental conditions, varying lighting conditions and transmission errors. This results in distortion, loss of image information, and overall degradation in visual quality. The presence of noise adversely impacts subsequent image/video processing/analysis tasks. Consequently, image denoising is a critical component of modern image processing pipelines. Image denoising is central to denoising diffusion models [1], [2] which are SOTA generative models.

The goal of image denoising is to recover the pristine image $\mathbf { x } \in \mathbb { R } ^ { m }$ from its noisy observation $\mathbf { y } \in \mathbb { R } ^ { m }$ . However, since perfect restoration is not possible, the aim is to estimate x as accurately as possible. Noise introduces distortions that obscure important details, making estimation of $\textbf { x a }$ challenging task. The most commonly used observation model is $\mathbf { y } = \mathbf { x } + \mathbf { n } .$ , where $\mathbf { n } \in \mathbb { R } ^ { m }$ denotes additive white Gaussian noise. The objective is to develop a denoising method that

Gaurav Sharma Department of Electrical and Computer Engineering University of Rochester, Rochester, NY, USA gaurav.sharma@rochester.edu

![](images/81a6e290f14e3136dd2defc1d43f2b4652f3c214b473ad5fa7faea08e824350a.jpg)  
Reference

![](images/e6c44a965a3def998ef503ffb985be5ae7fd1b65ca8776d239018edeefd953a8.jpg)

![](images/fcbf78667da18f2fbdef8e008d04e07406e615093b515ce87cbee4cde26203f5.jpg)

Noisy 26.03 dB, 0.801  
![](images/f1ff355912506e9995c664660cf3b966ec4e8644f43483ddf13363597609100d.jpg)  
MIRNet [3] 27.69 dB, 0.819  
MIRNet<sup>†</sup> 32.93 dB, 0.946  
Fig. 1: Comparison of a noisy and denoised texture using pretrained MIRNet [3] and its finetuned version (MIRNet<sup>†</sup>). PSNR and SSIM values improve after finetuning.

produces an estimate of the clean image ˆx that is as close as possible to the ground truth x.

Despite significant progress, most deep learning techniques focus on generic image denoising, with limited analysis on specialized classes such as textures. The objective of this paper is to fill this gap. In image processing, texture can be defined as a function of spatial variation of the brightness intensity of the pixels [5]. Natural textures involve fine details and repetitive elements with some degree of stochasticity that are particularly

![](images/192ee89f6f2115fc9cb9e2f80a71921d8aafd8e1a2250a1ea876767514bcfb56.jpg)  
Fig. 2: Visual comparison of various denoising models on the NUUR-Texture500 [4] dataset. Top row: noisy and pre-trained model results, bottom row: reference and (Model<sup>†</sup>) fine-tuned models. PSNR is shown for each result relative to the reference.

sensitive to noise.

As a result, most conventional denoising methods tend to blur intricate patterns in textures, making texture preservation more challenging than generic image denoising. Figure 1 illustrates that a deep learning model trained on standard image denoising datasets performs poorly on texture images, both quantitatively and qualitatively. Fine-tuning the model on a texture dataset enhances the performance, highlighting the need for texture-aware denoising strategies.

In this paper, our objective is to carry out a systematic study of the generalizability of deep learning techniques for denoising texture images. As argued in [6], SSIM is inadequate for objective assessment of the quality of texture. Therefore, we employ alternative measures such as STSIM1 [6], STSIM2 [6], MS-SSIM [7], CW-SSIM [8] and DISTS [9] in addition to PSNR and SSIM. Among these metrics, STSIM1 and STSIM2 are specifically designed for quality assessment of texture patterns.

## II. RELATED WORKS

Classical image denoising methods include Gaussian smoothing, bilateral filtering, non-local means [10], wavelet thresholding, and BM3D [11]. However, these approaches have been surpassed by SOTA deep learning techniques. Jain and Seung [12] first demonstrated the effectiveness of Convolutional Neural Networks (CNNs) for denoising, showing that CNNs could match or even surpass the performance of the then traditional models. Subsequently, with the introduction of SRCNN [13] and ARCNN [14] for super-resolution and JPEG artifact reduction, respectively, CNN-based methods have achieved impressive performance and have since become mainstream in the field of image restoration.

Denoising Autoencoders (DAEs) [15] introduced learning noise distributions to recover clean images via mapping inputs to meaningful intermediate representations. This was followed by CNN-based models such as DnCNN [16], which leveraged deep residual learning [17] to effectively remove Gaussian noise. Originally proposed to address degradation of accuracy with the depth of the network, it was also found to improve performance in image classification and object detection.

CNN-based denoising methods process images either at full resolution, preserving spatial details but lacking global contextual, or at reduced resolutions, capturing semantics but losing spatial accuracy. To address this trade-off, Multi-Scale Image Restoration Network (MIRNet) [3] employs parallel multi-scale feature extraction with attention-based aggregation, while Multi-Stage Progressive Image Restoration Network (MPRNet) [18] integrates contextual features extracted from an encoder-decoder network with a high-resolution branch that preserves local details. This ensures both spatial precision and strong contextual representation.

Most image-denoising algorithms are developed and evaluated on synthetic data, typically generated under the assumption of additive white Gaussian noise (AWGN). However, AWGN does not accurately represent real-world camera noise. CycleISP [20] addresses this by modeling the camera imaging pipeline in both forward and reverse directions, to generate realistic image pairs in both RAW and sRGB spaces.

Improving performance often increases computational cost. Nonlinear Activation Free Network (NAFNet) [21] mitigates this by removing or replacing nonlinear activations with multiplications, achieving improved performance with reduced complexity.

To overcome CNN limitations such as restricted receptive fields and static filters, transformer-based models have been proposed. Restoration Transformer (Restormer) [22] leverages global context for SOTA restoration performance. But transformers with global attention are computationally expensive. To reduce the high computational cost of global attention, Cross Aggregation Transformer (CAT) [23] introduced rectangle-window self-attention. This approach applies horizontal and vertical rectangle window attention in parallel across different heads, expanding the attention area and enabling feature aggregation across windows.

![](images/a975a6075562d2c198e2b01b016d2238ebf12c8f200e31cb32424db60b330160.jpg)  
Fig. 3: Visual comparison of denoising models on the SIDD [19]. Top row: noisy and pre-trained models, bottom row: reference and (Model<sup>†</sup>) fine-tuned models. PSNR (dB) shown w.r.t. the reference.

## III. EXPERIMENTAL SETUP

Despite the strong performance of modern CNN and transformer-based denoisers on synthetic Gaussian noise and real-world datasets such as SIDD [19], their performance when the distribution of images shift to texture-rich ones remains unaddressed. It is important to understand the dissimilarities between natural images and textures. While textures are components of natural images, they differ structurally when considered independently. As defined by Portilla & Simoncelli [24], “An image of visual texture is spatially homogeneous and typically contains repeated structures, often with some random variation (e.g., random positions, orientations or colors).” In contrast, natural images have been observed to predominantly consist of smooth (low-pass) regions [25]. This characteristic is fundamentally different from textures, which may or may not exhibit smoothness. The presence of highfrequency spectral information, which is also present in noise, is what makes texture denoising challenging. In this paper, the objective is to validate and benchmark the performance of SOTA deep denoisers, specifically on textures.

To benchmark the performance, SOTA denoising models, which have demonstrated exceptional performance in real-world image denoising, were selected — MIRNet [3], MPRNet [18], CAT [23], CycleISP [20], NAFNet [21], and Restormer [22]. These models were evaluated for denoising performance on a texture dataset using the same noise levels as the original SOTA models. Subsequently, the models are fine-tuned on the texture dataset, and their performance is evaluated again. Fine-tuning was adopted instead of training from scratch, as training these models from scratch requires a substantially larger dataset, which was not available for the present study.

## A. Datasets

For qualitative and quantitative evaluation of the pretrained and fine-tuned models, patches were extracted from the NUUR-Texture500 dataset [4] which contains 500 diverse and spatially consistent and uniform texture images spanning from regular patterns to stochastic ones. From each image in the NUUR-Texture500 dataset [4], ten $2 5 6 \times 2 5 6$ patches were randomly sampled, yielding 5000 patches in total, split into 75% for training and 25% for validation. To simulate real-world noise conditions, noisy patches were generated by adding white Gaussian noise, with the noise standard deviation (σ) determined by estimating the average variance of noise in the noisy images of the SIDD dataset [19]. Our goal was to match the noise characteristics of the synthetic noisy texture patches with those found in real-world datasets such as SIDD, enabling a fair evaluation of denoising performance.

## B. Implementation Details

Six Nvidia A6000 GPUs were used to fine-tune the models using PyTorch. The fine-tuning of the denoisers is done using Adam Optimizer $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ for 100 epochs with a batch size of 16. The initial learning rate is set to $1 \times 1 0 ^ { - 4 }$ and a cosine annealing strategy is used to steadily reduce the learning rate to $5 \times 1 0 ^ { - 5 }$ during training. The loss function is chosen in accordance with the network architecture and the same objective used during training is retained for fine-tuning (e.g. $\mathcal { L } _ { 1 }$ loss for CAT and Charbonnier loss for MPRNet).

## C. Evaluation Metrics

Performance is evaluated using both fidelity and perceptual metrics such as PSNR, SSIM, MS-SSIM [7], CW-SSIM [8], STSIM1 [6], STSIM2 [6], DISTS [9] and LPIPS [26]. Structural Texture Similarity (STSIM-1) captures texture similarity using local image statistics by replacing the structure term in SSIM with directional first-order autocorrelation of neighboring subband coefficients computed horizontally and vertically denoted by $\rho _ { \mathbf { x } } ( 0 , 1 )$ and $\rho _ { \mathbf { x } } ( 1 , 0 )$ respectively.

<table><tr><td>Model</td><td>1 PSNR↑ |</td><td>SSIM↑</td><td>MS-SSIM↑ 1</td><td>CW-SSIM↑</td><td>DISTS↓ 1</td><td>STSIM1↑</td><td>1 STSIM2↑</td><td>LPIPS↓</td></tr><tr><td>Noisy</td><td>1  $2 6 . 2 3 \pm 0 . 3 6$  1</td><td> $0 . 8 6 0 \pm 0 . 1 2 3$  一</td><td> $0 . 9 7 6 \pm 0 . 0 2 4$  一</td><td> $0 . 9 6 2 \pm 0 . 0 7 7$  一</td><td> $0 . 1 5 7 \pm 0 . 0 7 6$  一</td><td> $0 . 9 8 9 \pm 0 . 0 2 3$ </td><td>一  $0 . 9 4 1 \pm 0 . 0 5 2$ </td><td>一  $0 . 1 2 8 6 \pm 0 . 1 4 8 8$ </td></tr><tr><td> $\mathrm { C A T } \ [ 2 3 ]$ </td><td> $2 7 . 7 3 \pm 4 . 5 4 2$ </td><td> $0 . 8 3 4 \pm 0 . 1 4 9$ </td><td> $0 . 9 5 6 \pm 0 . 0 7 4$ </td><td> $0 . 9 3 1 \pm 0 . 1 2 7$ </td><td> $0 . 1 6 5 \pm 0 . 0 7 7$ </td><td> $0 . 9 5 4 \pm 0 . 0 9 2$ </td><td> $0 . 9 2 6 \pm 0 . 0 6 2$ </td><td> $0 . 1 9 0 4 \pm 0 . 1 7 7 8$ </td></tr><tr><td>CAT†</td><td> $3 4 . 1 7 \pm 3 . 3 8 9$ </td><td> $0 . 9 5 9 \pm 0 . 0 3 2$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 9$ </td><td> $0 . 9 7 3 \pm 0 . 0 5 4$ </td><td> $0 . 0 6 2 \pm 0 . 0 3 4$ </td><td> $0 . 9 9 7 \pm 0 . 0 0 6$ </td><td> $0 . 9 4 9 \pm 0 . 0 5 0$ </td><td> $0 . 0 3 0 4 \pm 0 . 0 2 8 3$ </td></tr><tr><td>CycleISP [20]</td><td> $2 5 . 5 2 \pm 3 . 7 5 9$ </td><td> $0 . 7 8 1 \pm 0 . 1 9 2$ </td><td> $0 . 9 3 3 \pm 0 . 0 9 9$ </td><td> $0 . 8 7 2 \pm 0 . 2 1 2$ </td><td> $0 . 2 3 9 \pm 0 . 0 8 9$ </td><td> $0 . 9 2 9 \pm 0 . 1 3 6$ </td><td> $0 . 9 1 1 \pm 0 . 0 7 7$ </td><td> $0 . 3 0 4 7 \pm 0 . 2 4 2 5$ </td></tr><tr><td>CycleISP†</td><td> $3 4 . 2 0 \pm 3 . 3 4 6$ </td><td> $0 . 9 5 9 \pm 0 . 0 3 2$ </td><td> $0 . 9 9 8 \pm 0 . 0 0 5$ </td><td> $0 . 9 7 3 \pm 0 . 0 5 3$ </td><td> $0 . 0 6 2 \pm 0 . 0 3 4$ </td><td> $0 . 9 9 8 \pm 0 . 0 0 5$ </td><td> $0 . 9 4 8 \pm 0 . 0 5 0$ </td><td> $0 . 0 3 0 6 \pm 0 . 0 2 9 6$ </td></tr><tr><td>MIRNet [3]</td><td> $2 5 . 2 3 \pm 5 . 7 6 5$ </td><td> $0 . 7 2 1 \pm 0 . 2 0 4$ </td><td> $0 . 9 1 7 \pm 0 . 1 0 4$ </td><td> $0 . 9 1 3 \pm 0 . 1 4 4$ </td><td> $0 . 2 3 2 \pm 0 . 0 8 1$ </td><td> $0 . 9 0 9 \pm 0 . 1 2 7$ </td><td> $0 . 9 0 9 \pm 0 . 0 7 0$ </td><td> $0 . 3 4 2 5 \pm 0 . 2 4 6 4$ </td></tr><tr><td>MIRNet†</td><td> $3 3 . 8 8 \pm 3 . 1 7 1$ </td><td> $0 . 9 5 8 \pm 0 . 0 3 3$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 9$ </td><td> $0 . 9 7 2 \pm 0 . 0 5 4$ </td><td> $0 . 0 6 4 \pm 0 . 0 3 4$ </td><td> $0 . 9 9 7 \pm 0 . 0 0 5$ </td><td> $0 . 9 5 0 \pm 0 . 0 4 9$ </td><td> $0 . 0 3 1 9 \pm 0 . 0 3 0 9$ </td></tr><tr><td>MPRNet [18]</td><td> $2 5 . 2 4 \pm 5 . 6 7 4$ </td><td> $0 . 7 2 1 \pm 0 . 2 0 7$ </td><td> $0 . 9 1 5 \pm 0 . 1 0 8$ </td><td> $0 . 9 1 9 \pm 0 . 1 3 4$ </td><td> $0 . 2 3 6 \pm 0 . 0 8 0$ </td><td> $0 . 9 1 1 \pm 0 . 1 3 1$ </td><td> $0 . 9 0 9 \pm 0 . 0 6 8$ </td><td> $0 . 3 4 3 2 \pm 0 . 2 5 9 0$ </td></tr><tr><td>MPRNet†</td><td> $3 4 . 1 7 \pm 3 . 2 9 7$ </td><td> $0 . 9 5 9 \pm 0 . 0 3 2$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 9$ </td><td> $0 . 9 7 2 \pm 0 . 0 5 4$ </td><td> $0 . 0 6 2 \pm 0 . 0 3 3$ </td><td> $0 . 9 9 7 \pm 0 . 0 0 5$ </td><td> $0 . 9 4 9 \pm 0 . 0 5 0$ </td><td> $0 . 0 3 0 3 \pm 0 . 0 2 9 1$ </td></tr><tr><td>NAFNet [21]</td><td> $2 3 . 9 3 \pm 6 . 3 9 9$ </td><td> $0 . 7 1 5 \pm 0 . 2 3 5$ </td><td> $0 . 8 7 7 \pm 0 . 1 8 7$ </td><td> $0 . 8 5 9 \pm 0 . 2 0 5$ </td><td> $0 . 2 5 1 \pm 0 . 0 9 5$ </td><td> $0 . 9 3 1 \pm 0 . 0 8 6$ </td><td> $0 . 9 0 9 \pm 0 . 0 6 8$ </td><td> $0 . 3 3 9 7 \pm 0 . 2 4 1 2$ </td></tr><tr><td>NAFNet†</td><td> $3 3 . 5 9 \pm 3 . 4 2 9$ </td><td> $0 . 9 5 3 \pm 0 . 0 6 4$ </td><td> $0 . 9 8 7 \pm 0 . 0 4 8$ </td><td> $0 . 9 6 7 \pm 0 . 0 8 2$ </td><td> $0 . 0 6 4 \pm 0 . 0 4 4$ </td><td> $0 . 9 9 4 \pm 0 . 0 5 5$ </td><td> $0 . 9 4 6 \pm 0 . 0 6 8$ </td><td> $0 . 0 3 6 2 \pm 0 . 0 7 2 8$ </td></tr><tr><td> $\mathrm { R e s t o r m e r } \ [ 2 2 ]$ </td><td> $2 6 . 8 1 \pm 5 . 2 1 3$ </td><td> $0 . 8 1 7 \pm 0 . 1 4 8$ </td><td> $0 . 9 4 9 \pm 0 . 0 7 3$ </td><td> $0 . 9 3 6 \pm 0 . 1 1 8$ </td><td> $0 . 1 8 6 \pm 0 . 0 7 0$ </td><td> $0 . 9 5 7 \pm 0 . 0 6 6$ </td><td> $0 . 9 2 3 \pm 0 . 0 5 9$ </td><td> $0 . 2 3 2 7 \pm 0 . 1 8 0 8$ </td></tr><tr><td>Restormer†</td><td> $3 3 . 9 4 \pm 3 . 1 4 4$ </td><td> $0 . 9 5 8 \pm 0 . 0 3 2$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 9$ </td><td> $0 . 9 7 2 \pm 0 . 0 5 4$ </td><td> $0 . 0 6 2 \pm 0 . 0 3 2$ </td><td> $0 . 9 9 7 \pm 0 . 0 0 4$ </td><td> $0 . 9 5 0 \pm 0 . 0 5 1$ </td><td> $0 . 0 3 0 5 \pm 0 . 0 2 8 7$ </td></tr></table>

TABLE I: Denoising performance (mean ± std) on NUUR-Texture500 dataset [4] using PSNR, SSIM, MS-SSIM, CW-SSIM, DISTS, STSIM1, STSIM2 and LPIPS. Dagger (<sup>†</sup>) indicates fine-tuned models.

$$
c _ { \mathbf x , \mathbf y } ( 0 , 1 ) = 1 - 0 . 5 \left| \rho _ { \mathbf x } ( 0 , 1 ) - \rho _ { \mathbf y } ( 0 , 1 ) \right|
$$

$$
c _ { \mathbf { x } , \mathbf { y } } ( 1 , 0 ) = 1 - 0 . 5 \left| \rho _ { \mathbf { x } } ( 1 , 0 ) - \rho _ { \mathbf { y } } ( 1 , 0 ) \right|
$$

The metric is calculated for each subband or patch and consolidated as follows:

$$
\mathrm { S T S I M - 1 } ( \mathbf { x } , \mathbf { y } ) = ( l ( x , y ) \cdot c ( x , y ) \cdot c _ { \mathbf { x } , \mathbf { y } } ( 0 , 1 ) \cdot c _ { \mathbf { x } , \mathbf { y } } ( 1 , 0 ) ) ^ { \frac { 1 } { 4 } } .
$$

STSIM-2 extends STSIM-1 by incorporating cross-band correlations. The crossband correlation between subbands m and n is computed as follows:

$$
\rho _ { | \mathbf x | } ^ { m , n } ( 0 , 0 ) = \frac { \mathbb E \left\{ \left[ | x ^ { m } ( i , j ) | - \mu _ { | \mathbf x | } ^ { m } \right] \left[ | x ^ { n } ( i , j ) | - \mu _ { | \mathbf x | } ^ { n } \right] \right\} } { \sigma _ { | \mathbf x | } ^ { m } \sigma _ { | \mathbf x | } ^ { n } } .
$$

Combining the STSIM-1 values over all subbands gives

$$
\mathrm { S T S I M - 2 } ( \mathbf { x } , \mathbf { y } ) = \frac { \sum _ { m = 1 } ^ { N _ { b } } q _ { \mathrm { S T S I M - 1 } } ^ { m } ( \mathbf { x } , \mathbf { y } ) + \sum _ { i = 1 } ^ { N _ { c } } \mathbf { c } _ { \mathbf { x } , \mathbf { y } } ^ { m _ { i } , n _ { i } } ( 0 , 0 ) } { N _ { b } + N _ { c } } .
$$

where $N _ { b }$ denotes the number of subbands, and $N _ { c }$ accounts for the cross-correlations considering various orientations.

STSIM-M Different image subbands statistically vary in terms of subband statistics like mean, variance, horizontal and vertical auto-correlation and cross-correlation across subbands. A feature vector $F _ { \mathbf { x } }$ (and $F _ { \mathbf { y } } )$ is formed by stacking all these features for images x and y, respectively. STSIM-M is then defined as the Mahalanobis distance [27] between these features, considering that they are mutually uncorrelated and combined.

$$
\operatorname { S T S I M - M } ( F _ { \mathbf { x } } , F _ { \mathbf { y } } ) = { \sqrt { ( F _ { \mathbf { x } } - F _ { \mathbf { y } } ) ^ { H } M ( F _ { \mathbf { x } } - F _ { \mathbf { y } } ) } }
$$

Deep Image Structure and Texture Similarity (DISTS)) [9] is a deep learning-based metric that assesses the similarity of structural and texture information in images. It uses high-level perceptual features extracted from different layers of a pretrained VGG network.

<table><tr><td>Model</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>Noisy</td><td> $2 7 . 8 6 \pm 5 . 1 7$ </td><td> $0 . 5 2 \pm 0 . 2 3$ </td></tr><tr><td>MIRNet [3]</td><td> $4 2 . 5 9 \pm 4 . 0 8$ </td><td> $0 . 9 6 \pm 0 . 0 6$ </td></tr><tr><td>MIRNet†</td><td> $3 2 . 6 8 \pm 6 . 7 5$ </td><td> $0 . 7 2 \pm 0 . 2 6$ </td></tr><tr><td>MPRNet [18] MPRNet†</td><td> $4 2 . 6 0 \pm 4 . 1 0$   $3 2 . 4 8 \pm 6 . 8 0$ </td><td> $0 . 9 6 \pm 0 . 0 6$   $0 . 7 2 \pm 0 . 2 6$ </td></tr><tr><td>CycleISP [20]</td><td> $4 2 . 2 1 \pm 4 . 1 2$ </td><td> $0 . 9 5 \pm 0 . 0 6$ </td></tr><tr><td>CycleISP†</td><td> $3 3 . 9 8 \pm 7 . 3 3$ </td><td> $0 . 7 6 \pm 0 . 2 7$ </td></tr><tr><td>CAT [23]</td><td> $4 3 . 2 8 \pm 4 . 3 6$ </td><td> $0 . 9 6 \pm 0 . 0 6$ </td></tr><tr><td>CAT†</td><td> $3 2 . 6 8 \pm 7 . 4 8$ </td><td> $0 . 7 3 \pm 0 . 2 7$ </td></tr><tr><td>NAFNet [21]</td><td> $4 4 . 8 3 \pm 4 . 5 5$ </td><td> $0 . 9 6 \pm 0 . 0 5$ </td></tr><tr><td>NAFNet†</td><td> $3 7 . 3 4 \pm 5 . 5 5$ </td><td> $0 . 8 8 \pm 0 . 1 6$ </td></tr><tr><td></td><td></td><td></td></tr><tr><td>Restormer [22] Restormer†</td><td> $4 3 . 3 6 \pm 4 . 3 9$   $3 2 . 8 0 \pm 6 . 4 2$ </td><td> $0 . 9 6 \pm 0 . 0 6$   $0 . 7 2 \pm 0 . 2 5$ </td></tr></table>

TABLE II: Original vs. fine-tuned (<sup>†</sup>) model performance on SIDD [19].

## IV. RESULTS

Figure 2 presents a qualitative comparison of performance of various denoising models on texture images taken from the NUUR-Texture500 dataset. The corresponding quantitative results, averaged over the validation set, are reported in Table I in terms of PSNR, SSIM, STSIM1, STSIM2, and other metrics. The fine-tuned models demonstrate significant improvements across various performance metrics. Although the models were fine-tuned on pointwise losses $\left( \mathrm { e . g . ~ } \mathcal { L } _ { 1 } \right)$ , they also exhibited improvements in non-pointwise perceptual metrics such as MS-SSIM, CW-SSIM, STSIM1, STSIM2 and DISTS.

Fine-tuned models show a considerable increase in PSNR, ranging from +6.44 to +9.66 dB and subjective metric such as LPIPS shows an improvement of around -0.16 to -0.31 over the baseline models. This result clearly indicates that the baseline models are not adequately equipped to handle textures, and it is only upon fine-tuning, that the performance of the models can be improved significantly. Fine-tuned models also performed better across all the other structural similarity metrics (SSIM variants) and DISTS, indicating the importance of fine-tuning for achieving structurally and perceptually superior denoising performance. The performance improvement happens across multiple scales as indicated by the MS-SSIM metric. The STSIM1 and STSIM2 metrics, which assess structural and multi-scale structural similarity specifically for textures, also show consistent improvements (+0.065 and +0.034 on average, respectively).

Next, we carried out an evaluation of the fine-tuned denoising models on the original natural image datasets. The performance is reported in Table II. We observe that the models fine-tuned on the texture dataset do not perform as well on the original dataset. A visual demonstration is presented in Figure 3. This result shows that, in the process of adaptation to the texture dataset, the models have catastrophically forgotten [28] the original dataset. The incongruence in performance suggests that natural images and texture images have low-dimensional embeddings that overlap minimally with each other. Thus, one can infer that textures have to be handled separately compared to other natural images. A unified denoising model that handles both texture images and other natural images equally well is an exciting direction for future research.

## V. CONCLUSION

In this study, we benchmarked the performance of SOTA image-denoisers on a texture dataset. We have shown that off-the-shelf denoisers achieve subpar denoising performance on texture images compared to generic natural images. In our benchmarking, we have also compared the denoising performance using several metrics like STSIM1, STSIM-2 that capture textural nuances better than SSIM. We have finetuned the models on the texture dataset and showed that the texture denoising performance improves, but at the expense of significant degradation in performance on the original dataset. The study highlights the need for better denoisers that can handle textures better, either separately or along with general image datasets. However, it is to be noted that since the models were trained on point-by-point (e.g. $\mathcal { L } _ { 1 } )$ losses, further efforts are required to allow more extensive use of perceptual metrics, such as STSIM-M and STSIM-VGG, that do not operate on a point-by-point basis.

The next step in the study would be to design unified denoisers and train them on perceptual losses that are not point-by-point which can allow for a more holistic texture analysis. We believe that such unified denoisers can play a crucial role in the generation of natural images with rich texture in a diffusion or flow-based modelling framework.

## REFERENCES

[1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems, vol. 33, pp. 6840–6851, 2020.

[2] J. Liu, Q. Wang, H. Fan, Y. Wang, Y. Tang, and L. Qu, “Residual denoising diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2773–2783.

[3] S. W. Zamir et al., “Learning enriched features for real image restoration and enhancement,” in European Conference on Computer Vision, 2020.

[4] J. Lin, G. Sharma, and T. Pappas, “NUUR-Texture500: A Diverse Dataset of High Resolution Homogeneous Textures,” Sept. 2022.

[5] M. Tuceryan and A. K. Jain, “Handbook of pattern recognition and computer vision,” World Scientific, pp. 207–248, 1998.

[6] J. Zujovic, T. Pappas, and D. Neuhoff, “Structural similarity metrics for texture analysis and retrieval,” in Proceedings of the IEEE International Conference on Image Processing, 2009, pp. 2225–2228.

[7] Z. Wang, E. Simoncelli, and A. Bovik, “Multiscale structural similarity for image quality assessment,” in Proceedings of the Asilomar Conference on Signals, Systems and Computers, 2003, pp. 1398–1402.

[8] M. Sampat, Z. Wang, S. Gupta, A. Bovik, and M. Markey, “Complex wavelet structural similarity: A new image similarity index,” IEEE Transactions on Image Processing, vol. 18, no. 11, pp. 2385–2401, 2009.

[9] K. Ding, K. Ma, S. Wang, and E. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 5, pp. 2567–2581, 2020.

[10] A. Buades, B. Coll, and J.-M. Morel, “A non-local algorithm for image denoising,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2005, vol. 2, pp. 60–65.

[11] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian, “Image denoising by sparse 3-d transform-domain collaborative filtering,” IEEE Transactions on image processing, vol. 16, no. 8, pp. 2080–2095, 2007.

[12] V. Jain and S. Seung, “Natural image denoising with convolutional networks,” in Advances in Neural Information Processing Systems, 2008, vol. 21.

[13] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang, “Image super-resolution using deep convolutional networks,” IEEE transactions on pattern analysis and machine intelligence, vol. 38, no. 2, pp. 295– 307, 2015.

[14] C. Dong, Y. Deng, C. C. Loy, and X. Tang, “Compression artifacts reduction by a deep convolutional network,” 2015.

[15] P. Vincent, H. Larochelle, Y. Bengio, and P. A. Manzagol, “Extracting and composing robust features with denoising autoencoders,” in Proceedings of the International Conference on Machine Learning, 2008, pp. 1096–1103.

[16] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a gaussian denoiser: Residual learning of deep convolutional neural networks for image denoising,” IEEE Transactions on Image Processing, vol. 26, no. 7, pp. 3142–3155, 2017.

[17] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[18] S. W. Zamir et al., “Multi-stage progressive image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 14821–14831.

[19] A. Abdelhamed, S. Lin, and M. Brown, “A high-quality denoising dataset for smartphone cameras,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 1692–1700.

[20] S. W. Zamir et al., “Cycleisp: Real image restoration via improved data synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 2696–2705.

[21] L. Chen, X. Chu, X. Zhang, and J. Sun, “Simple baselines for image restoration,” in European Conference on Computer Vision, 2022, pp. 17–33.

[22] S. W. Zamir et al., “Restormer: Efficient transformer for high-resolution image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[23] Z. Chen et al., “Cross aggregation transformer for image restoration,” Advances in Neural Information Processing Systems, vol. 35, pp. 25478– 25490, 2022.

[24] J. Portilla and E. Simoncelli, “A parametric texture model based on joint statistics of complex wavelet coefficients,” International Journal of Computer Vision, vol. 40, pp. 49–70, 2000.

[25] J. Huang and D. Mumford, “Statistics of natural images and models,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 1999, vol. 1, pp. 541–547.

[26] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595.

[27] Prasanta Chandra Mahalanobis, “On the generalized distance in statistics,” Sankhya: The Indian Journal of Statistics, Series A (2008-)¯ , vol. 80, pp. S1–S7, 2018.

[28] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, 2017.