# Domain-Adapted Latent Difusion Models for CT-to-PET Image Translation

Yann Bourdé\*<sup>1</sup>, Lea Bogensperger<sup>1</sup>, Arnela Hadzic<sup>2</sup>, and Michael Krauthammer<sup>1</sup> \*yann.bourde@uzh.ch

1 Department of Quantitative Biomedicine, University of Zurich Institute for Medical Informatics, Statistics and Documentation, Medical University of Graz

Abstract. Positron Emission Tomography (PET) provides important metabolic information for tumor detection and staging but remains significantly less accessible than Computed Tomography (CT) due to its high cost, infrastructure requirements, and reliance on injected radioactive tracers. CT-to-PET image translation using generative models aims to address this disparity, yet state-of-the-art difusion-based methods are limited by the scarcity of paired training data. To overcome this, we propose a domain-adapted latent difusion framework that transfers natural-image priors using a relatively small dataset of 21,930 paired CT/PET slices, incorporating a signed power transform to resolve extreme intensity sparsity in PET data. Compared to CPDM, a state-ofthe-art baseline for CT-to-PET image translation, our approach yields superior performance in pixel-wise reconstruction and localized tumor fidelity, demonstrating efective transfer of natural-image priors to the medical domain. We further provide a systematic analysis of evaluation metric reliability for sparse PET data, showing that standard full-image metrics can be misleading indicators of translated image quality, and propose background-filtered and mask-restricted metrics as more clinically meaningful alternatives.

Keywords: CT/PET image translation · Latent difusion models · Medical image synthesis · Evaluation metrics

## 1 Introduction

Positron Emission Tomography (PET) and Computed Tomography (CT) are complementary medical imaging modalities that capture anatomical and metabolic information, e.g., for tumor detection, staging, and treatment monitoring. Specifically, CT scans provide high-resolution anatomical information through X-ray attenuation, while PET scans reveal metabolic activity via radiotracer distribution, typically <sup>18</sup>F-fluorodeoxyglucose (FDG). Used together, they ofer a more complete diagnostic picture than either modality alone. However, PET scanners require costly infrastructure and radiotracer production facilities, making PET imaging less accessible than CT, particularly in low-resource healthcare settings.

With the successful introduction and scalability of generative models [4,10,18,11 and their numerous applications to the medical imaging domain [1,3,5], these methods are also being investigated for paired image-to-image translation tasks such as CT-to-PET image translation [14]. Early approaches relied on Generative Adversarial Networks (GANs) [8,22], which sufered from training instability and mode collapse, while modern state-of-the-art (SOTA) approaches rely on difusion-based models [14,12]. In particular, latent difusion models (LDMs) [16] ofer a compelling option due to the learned latent space that they operate in, which allows generation of high-resolution images at lower computational cost. For CT-to-PET translation specifically, Nguyen et al. [14] recently introduced the CT-to-PET Difusion Model (CPDM), a conditional difusion model based on a Brownian Bridge process that defines a stochastic bridge between CT and PET latents rather than corrupting toward pure Gaussian noise, and additionally incorporates domain knowledge through attention and attenuation map modules. Notably, the authors report that LDM priors were inferior to CPDM in their experiments [14]—a finding we revisit in this work.

A common limitation across these approaches is the scale of publicly available paired CT/PET datasets [13]: current SOTA models are trained from scratch on this scarce data [14,12], potentially limiting their performance and ability to generalize beyond the training distribution.

Motivated by this gap, we introduce a domain-adapted latent difusion framework for CT-to-PET image translation in this work. We demonstrate that transfer learning by finetuning a model pretrained on natural image priors efectively mitigates data scarcity and enhances PET synthesis fidelity. Our contributions are summarized as follows:

We implement a transfer learning strategy to adapt natural image priors to the PET domain and introduce a signed power transform to reduce the extreme intensity imbalance in PET images.

– We demonstrate that our framework yields superior performance in pixellevel reconstruction and localized tumor fidelity compared to the state-ofthe-art difusion-based CPDM [14], while ofering a data-eficient approach for CT-to-PET image translation.

– We analyze the limitations of standard metrics for sparse PET data and propose background-filtered and mask-restricted metrics as more clinically meaningful alternatives.

## 2 Method

## 2.1 Conditional CT-to-PET Synthesis

Applying the difusion process directly in pixel space is computationally expensive at high resolutions. Latent Difusion Models (LDMs) [16] address this by compressing images into a low-dimensional latent space via a KL-regularized variational autoencoder (VAE) and running the difusion process in this space, substantially reducing computational cost while preserving perceptual quality.

In our framework, we utilize this architecture for conditional CT-to-PET synthesis. Let $z _ { \mathrm { P E T } } = \mathcal { E } ( x _ { \mathrm { P E T } } )$ and $z _ { \mathrm { C T } } = \mathcal { E } ( x _ { \mathrm { C T } } )$ denote the latent representations of the transformed PET and CT images, respectively. During training, the diffusion process adds Gaussian noise to z<sub>PET</sub> over T timesteps to generate $z _ { \mathrm { P E T } } ^ { ( t ) } .$ To enable conditional synthesis, we concatenate the noisy latent $z _ { \mathrm { P E T } } ^ { ( t ) }$ with the clean CT latent $z _ { \mathrm { C T } }$ , providing the U-Net with a combined 8-channel input. The U-Net is then trained to learn the conditional reverse difusion process:

$$
z _ { \mathrm { P E T } } ^ { ( t - 1 ) } \sim p _ { \theta } \left( z _ { \mathrm { P E T } } ^ { ( t - 1 ) } \mid z _ { \mathrm { P E T } } ^ { ( t ) } , z _ { \mathrm { C T } } \right) .\tag{1}
$$

## 2.2 Domain Adaptation and Finetuning Strategy

The scarcity of paired CT/PET data can hinder the performance of models trained from scratch in this domain. To address this limitation, we utilize transfer learning by initializing our framework with the Stable Difusion v1.4 backbone pretrained on natural images from the large-scale LAION dataset [17]. This allows us to efectively leverage the strong structural and perceptual priors learned during pretraining and adapt this backbone to the PET domain using supervised finetuning. To accommodate the additional input from the CT latent z<sub>CT</sub>, we follow Ke et al. [9] and expand the first convolutional layer of the UNet by duplicating the pretrained input weights and halving their values, preserving the expected magnitude of activations at initialization.

To validate the framework’s suitability for our medical imaging data, we first investigated the reconstruction accuracy of the pretrained VAE on our outof-distribution medical imaging data. As shown in Table 1 for the PCLT20K dataset, the reconstruction is suficiently accurate, so we keep the VAE frozen and only focus on finetuning the denoising U-Net.

Table 1. VAE reconstruction metrics.
<table><tr><td>Modality MAE</td><td></td><td></td><td></td><td>↓ PSNR ↑ SSIM ↑LPIPS↓</td></tr><tr><td>CT</td><td>0.0111</td><td>33.57</td><td>0.936</td><td>0.0616</td></tr><tr><td>PET</td><td>0.0019</td><td>48.07</td><td>0.982</td><td>0.0147</td></tr></table>

However, natural-image intensity statistics substantially difer from medical modalities such as PET, which exhibit extreme sparsity (see Section 3.1). Moreover, this sparsity is a central challenge for difusion-based generative models, as standard noise prediction losses treat all pixels equally regardless of clinical relevance. While recent work has shown that finetuning pretrained models on domain-specific data can yield competitive results with limited paired samples [19,15], successful adaptation requires aligning these distributions. Consequently, we apply a signed power transform

$$
x _ { \mathrm { P E T } } ^ { \prime } = \mathrm { s i g n } ( x _ { \mathrm { P E T } } ) \cdot | x _ { \mathrm { P E T } } | ^ { p }\tag{2}
$$

to the PET images before passing them through the VAE encoder, controlled by $p \in \mathsf { \Gamma } ( 0 , 1 ]$ . It compresses the dynamic range and spreads near-zero values across a broader interval, making the PET distribution more compatible with the natural-image priors of the pretrained model. The inverse transform is applied at inference to recover the original PET intensity scale.

Another key challenge for LDM transfer learning is catastrophic forgetting of pretrained representations. Partial layer freezing [20] and parameter-eficient adaptation [7] can help mitigate this issue. Consequently, we study two finetuning variants: unfrozen, in which all U-Net weights are updated, and frozen, in which ResNet convolutional blocks, feed-forward layers, and spatial sampling convolutions are kept fixed while attention layers, normalization layers, and the time embedding remain trainable. We compare both variants in Section 3.3 to assess the trade-of between adaptation flexibility and preservation of pretrained priors.

## 2.3 Evaluation Metrics

For conditional synthesis we report standard full-reference metrics: Mean Absolute Error (MAE ↓), Peak Signal-to-Noise Ratio (PSNR ↑), Structural Similarity Index (SSIM ↑), and Learned Perceptual Image Patch Similarity (LPIPS ↓) [21]. However, Blau and Michaeli [2] showed that perceptual quality and pixel-level fidelity are fundamentally at odds, a trade-of directly relevant to comparing generative models across diferent metric families. For sparse modalities such as PET, where 95–98% of pixels are near-zero background, full-image metrics are further confounded by background dominance, masking failures in clinically relevant tumor uptake regions. Therefore, we additionally report Masked MAE, Masked SSIM, and Masked LPIPS over the tumor-mask pixels provided by PCLT20K [13]. Masked MAE is computed over the tumor-mask pixels directly; because SSIM and LPIPS are patch-based, Masked SSIM and Masked LPIPS are instead computed on a tight bounding box around the mask region to provide suficient spatial context. Mask-restricted metrics quantify fidelity within the tumor region but do not penalize uptake predicted outside it, so we interpret them alongside full-image metrics rather than in isolation.

## 3 Experiments

## 3.1 Dataset

The PCLT20K dataset [13]<sup>3</sup> contains 21,930 paired axial 2D slices of CT, PET, and binary tumor masks at 512 × 512 resolution, derived from lung tumor scans of 605 patients. CT intensities are normalized to [−1, 1] and follow a broad and roughly symmetric distribution. PET images are far more challenging to model: 95–98% of pixels are near-zero intensity, yielding a normalized mean of −0.972.

The data splits are predefined with an $8 0 \% / 2 0 \%$ train/test split, from which we construct our validation set of 80 samples.

## 3.2 Training Details

To evaluate the impact of our proposed strategy, we analyze four model variants: base (unfrozen), base (frozen), power (unfrozen), and power (frozen). Power model variants are trained on inputs preprocessed with our proposed signed power transform defined in Equation 2, using a coeficient $p = 0 . 2 5$ , whereas the base model variants are trained on untransformed inputs. We use L1 loss for all models during training.

All models are finetuned from the Stable Difusion v1.4 checkpoint, consisting of a frozen variational autoencoder (VAE) (≈83M parameters, spatial compression factor 8×) and a denoising U-Net (≈860M parameters). We used AdamW as the optimizer, with a learning rate of $1 \times 1 0 ^ { - 4 }$ . Training employed cosine learning rate scheduling and an exponential moving average (EMA) of model weights with a decay of 0.9999. We adopted the DDPM noise schedule [6] with $T = 1 0 0 0$ steps for both training and sampling at inference.

## 3.3 Conditional CT-to-PET Results

In our evaluation, we include our four conditional model variants, alongside CPDM’s original publicly available checkpoint (CPDM original ckpt) [14], trained on CPDM’s unreleased dataset, and a variant of CPDM retrained on the PCLT20K dataset (CPDM ) [13]. Table 2 reports full-image and Table 3 reports maskregion metrics for all models evaluated on 416 samples from the PCLT20K test set. Figure 1 shows representative conditional generation examples together with pixel-wise error maps.

Table 2. Conditional CT-to-PET results — full-image metrics, mean and standard deviation. Per-column best results are denoted in bold, second-best results are underlined. <sup>∗</sup> indicates a statistically significant improvement $( p < 0 . 0 5$ , paired Wilcoxon signed-rank test) compared to the retrained CPDM baseline.
<table><tr><td>Model</td><td>MAE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>CPDM original ckpt</td><td> $\overline { { 0 . 0 2 0 4 \pm 0 . 0 0 7 9 } }$ </td><td> $\overline { { 2 6 . 6 7 \pm 2 . 8 0 } }$ </td><td> $\overline { { 0 . 8 2 8 \pm 0 . 0 6 0 } }$ </td><td> $\overline { { 0 . 1 4 5 \pm 0 . 0 2 0 } }$ </td></tr><tr><td>CPDM</td><td> $0 . 0 1 4 1 \pm 0 . 0 1 0 1$ </td><td> $2 9 . 0 4 \pm 4 . 3 0$ </td><td> $0 . 8 8 4 \pm 0 . 0 8 2$ </td><td> ${ \bf 0 . 1 0 9 \pm 0 . 0 2 1 }$ </td></tr><tr><td>base</td><td> $\overline { { 0 . 0 1 8 2 \pm 0 . 0 1 0 0 } }$ </td><td> $\overline { { { \bf 3 0 . 6 2 } \pm 4 . 6 4 } }$ </td><td> $\overline { { 0 . 5 9 1 \pm 0 . 0 7 9 } }$ </td><td> $\overline { { 0 . 2 0 4 \pm 0 . 0 2 3 } }$ </td></tr><tr><td>power</td><td> $\mathbf { 0 . 0 1 2 9 \pm 0 . 0 0 8 1 ^ { \ast } }$ </td><td> $3 0 . 0 4 \pm 4 . 0 1$ </td><td> $\mathbf { 0 . 9 1 3 \pm 0 . 0 6 6 ^ { \ast } }$ </td><td> $0 . 1 2 3 \pm 0 . 0 2 4 ^ { \ast }$ </td></tr><tr><td>base frozen</td><td> $0 . 0 1 7 3 \pm 0 . 0 0 8 0 ^ { \ast }$ </td><td> $3 0 . 1 7 \pm 3 . 8 9$ </td><td> $0 . 6 6 1 \pm 0 . 0 6 9 ^ { \ast }$ </td><td> $0 . 2 3 8 \pm 0 . 0 2 1 ^ { \ast }$ </td></tr><tr><td>power frozen</td><td> $0 . 0 1 5 8 \pm 0 . 0 0 7 2$ </td><td> $2 8 . 5 0 \pm 3 . 2 2$ </td><td> $0 . 8 9 3 \pm 0 . 0 6 2$ </td><td> $0 . 1 3 7 \pm 0 . 0 2 3$ </td></tr></table>

The power transform consistently improves full-image MAE, SSIM, and LPIPS, indicating that addressing PET sparsity at the preprocessing stage is beneficial. Freezing a subset of U-Net layers has negligible efect on performance, suggesting that ResNet convolutional and feed-forward features transfer efectively from natural image pretraining and that adaptation is largely carried by the trainable attention layers. Our best model (power ) outperforms both CPDM checkpoints on MAE and SSIM, and PSNR. CPDM achieves better LPIPS, which likely reflects the perception-distortion tradeof [2]. CPDM’s Brownian Bridge formulation may favor perceptual realism over pixel-level fidelity, while our model optimizes more directly for reconstruction accuracy. In the context of medical imaging, pixel-level fidelity may be preferable for preserving clinically relevant intensity information, though this would require clinical validation.

Table 3. Conditional CT-to-PET results — masked/tumour-box metrics, mean and standard deviation. Per-column best results are denoted in bold, second-best results are underlined. <sup>∗</sup> indicates a statistically significant improvement $( p < 0 . 0 5$ , paired Wilcoxon signed-rank test) compared to the retrained CPDM baseline.
<table><tr><td>Model</td><td>Masked MAE↓</td><td>Masked SSIM↑</td><td>Masked LPIPS↓</td></tr><tr><td>CPDM original ckpt</td><td> $\overline { { 0 . 1 8 9 \pm 0 . 1 2 0 } }$ </td><td> $\overline { { 0 . 2 4 3 \pm 0 . 1 2 4 } }$ </td><td> $0 . 4 6 3 \pm 0 . 0 6 6$ </td></tr><tr><td>CPDM</td><td> $0 . 2 2 1 \pm 0 . 1 3 1$ </td><td> $0 . 2 7 6 \pm 0 . 1 6 0$ </td><td> $0 . 4 0 0 \pm 0 . 0 6 9$ </td></tr><tr><td>base</td><td> $\overline { { 0 . 1 8 1 \pm 0 . 1 0 1 } }$ </td><td> $\overline { { { \bf 0 . 4 8 5 \pm 0 . 1 9 4 } } }$ </td><td> $\mathbf { \overline { { 0 . 3 6 8 \pm 0 . 0 9 8 } } }$ </td></tr><tr><td>power</td><td> $\mathbf { 0 . 1 6 5 \pm 0 . 0 9 2 ^ { \ast } }$ </td><td> $0 . 4 8 0 \pm 0 . 1 7 5 ^ { \ast }$ </td><td> $\underline { { 0 . 3 6 9 } } \pm 0 . 0 9 2 ^ { \ast }$ </td></tr><tr><td>base frozen</td><td> $0 . 1 7 1 \pm 0 . 0 9 8 ^ { \ast }$ </td><td> $\overline { { 0 . 4 6 2 } } \pm 0 . 1 7 0 ^ { \ast }$ </td><td> $0 . 4 1 3 \pm 0 . 0 9 4 ^ { \ast }$ </td></tr><tr><td>power frozen</td><td> $\mathbf { 0 . 1 6 5 \pm 0 . 0 9 5 }$ </td><td> $0 . 4 2 7 \pm 0 . 1 5 7$ </td><td> $0 . 4 1 6 \pm 0 . 0 9 0$ </td></tr></table>

![](images/209574a1710c77bfa10dc9b94dbe0c7889a30e9d6f09c3b94f0237e71981db04.jpg)  
Fig. 1. Conditional CT-to-PET generation examples. Columns: input CT, then PET generations (top) and absolute error maps vs. ground truth (bottom) for each model.

Mask-region metrics are substantially lower than full-image metrics for all models, highlighting the dificulty of accurately localizing and synthesizing PET uptake in tumor regions. Our models perform very similarly to each other across the three masked metrics. Notably, the large SSIM gap between base and power seen on full-image metrics largely disappears here. This suggests that the power transform’s global gains may come at a slight cost within the highest-intensity tumor regions.

Overall, our results validate our transfer learning framework as a robust approach for CT-to-PET synthesis, demonstrating that targeted intensity transformations and selective finetuning can successfully adapt pretrained generative models to this cross-modality task.

![](images/a8972722c50f25592023227f36f1e88fb543a774cf5beac7431d0cc0bb76f138.jpg)  
Fig. 2. Filtered vs unfiltered CT-to-PET generations with absolute error maps.

## 3.4 Metric Caveats and Background Filtering

A key finding of this work is that standard full-image metrics are unreliable for sparse PET evaluation. Because 95–98% of PET pixels are near-zero background, a model perfectly reproducing background while misplacing tumor uptake can still achieve low MAE and high SSIM, while a clinically superior model that correctly localizes uptake but introduces minor background variation appears worse.

Table 4. Conditional CT-to-PET evaluation metrics with background filter $( \tau = 0 . 0 5 )$ ， averaged over 160 test samples.
<table><tr><td>MAE ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Masked MAE↓</td></tr><tr><td>Unfiltered (mean) 0.0154 32.58 0.587 0.184 0.1719 0.909 0.152</td></tr></table>

To quantify this efect, we evaluated our base conditional model under a background filter that zeroes all pixels below an absolute intensity threshold of $\tau = 0 . 0 5$ before computing metrics. Figure 2 shows example outputs under this filter and Table 4 reports full-image metrics averaged over 160 test samples, with and without the filter. For reference, the table also includes Masked MAE (0.1719), which is already foreground-restricted, so the background filter does not apply (—). Most metrics significantly improve with the filter, in particular SSIM, because the base model tends to produce light noise in the background which significantly skews full-image metrics. When this noise is filtered we see that the full-image metrics match those of the power models. This explains the apparent gap between base and power in Section 3.3: it largely reflects background noise rather than a diference in tumor-region fidelity. These results indicate that the model captures signal-bearing regions well and that degradation in full-image metrics is largely driven by minor background variation rather than failure in clinically relevant regions. In light of these findings, we recommend that future work on PET synthesis adopt background-filtered or mask-restricted metrics as the primary evaluation criterion alongside full-image metrics.

## 3.5 Ablations and Extended Experiments

Power transform. We evaluated frozen and unfrozen variants using diferent power coeficients $p = 0 . 1 2 5$ and $p = 0 . 5$ and compared them against our $p = 0 . 2 5$ model variants. The resulting metrics are remarkably similar across all four: SSIM approximately 0.88, PSNR approximately 29, MAE approximately 0.015, and LPIPS between 0.12 and 0.15, with the masked metrics following the same pattern. Overall, our best $p = 0 . 2 5$ model variant (power) remains the top performer with the strongest mean metrics, though the other coeficient variants often fall within its standard deviation range. This result suggests that the optimal coeficient may vary across datasets, and that what matters most is applying the transform at all, rather than the precise value of p.

Statistical significance. We evaluated the statistical significance of our results in Tables 2 and 3 using a paired Wilcoxon signed-rank test across three models: base frozen, power, and retrained CPDM, on our 416 test samples. We tested all full-image and masked metrics across all three pairs and found that the only non-significant result was full-image PSNR between our base frozen and power models. This confirms the statistical significance $\left( p < 0 . 0 5 \right)$ of our results against CPDM, as well as between our own model variants across all other metrics.

Out-of-distribution experiment. To evaluate robustness to corrupted input, we fed CT samples corrupted with increasing levels of Gaussian noise (0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0) into our base frozen and power models and measured the resulting metrics on 16 samples per level. The metrics degrade unevenly across noise levels. LPIPS worsens consistently, from 0.230 to 0.240 for base frozen and from 0.143 to 0.187 for power. MAE and SSIM instead dip at low-to-moderate noise (0.05, 0.1, 0.25) before partially recovering at higher levels (0.75, 1.0). PSNR shows no consistent trend in either direction, with high standard deviation throughout. Masked metrics degrade more consistently with noise, particularly at the highest levels (0.75 and 1.0). We attribute these diferences to what each metric is sensitive to. Since LPIPS measures perceptual realism, as conditioning degrades and the model falls back toward a generic output, the result drifts steadily further from a realistic PET image. MAE and SSIM instead measure direct structural and pixel-wise alignment: at moderate noise levels the model still follows the corrupted conditioning and generates specific, confidently wrong detail, producing large structural error, whereas at high noise the conditioning carries essentially no signal and the model regresses toward a generic output that is structurally closer to typical ground truth on average. PSNR’s volatility likely reflects this same tension between detail-following error and regression to the mean, compounded by the small sample size $( n = 1 6 )$ per level. Masked metrics degrade more steadily because the tumor region depends on fine, localized detail that is lost outright at higher noise levels.

## 4 Conclusion

In this work, we presented a data-eficient transfer learning framework for CTto-PET image translation, demonstrating that natural-image priors can be effectively adapted using a small dataset of 21,930 paired slices. By incorporating a signed power transform to address inherent PET intensity imbalance, our approach yields superior reconstruction accuracy and localized tumor synthesis when compared to the state-of-the-art CPDM baseline. We further identified background-pixel dominance as a fundamental limitation of standard evaluation metrics for sparse PET data, and showed that background-filtered and tumormask-restricted metrics provide more clinically meaningful assessments. These findings suggest that finetuning pretrained latent difusion models is a dataeficient approach to medical image synthesis, and highlight the need for carefully designed evaluation protocols for sparse medical modalities.

Several directions remain for future work. First, systematically reducing the training set would characterize the minimum number of paired samples required before transfer learning degrades, quantifying the data-eficiency limit of this approach. Moreover, adding losses computed in image space by backpropagating through the VAE decoder—rather than in latent space—may better align training with pixel-level clinical fidelity and could clarify the interaction between loss function choice and the signed power transform observed in this work.

## References

1. Armanious, K., Jiang, C., Fischer, M., Küstner, T., Hepp, T., Nikolaou, K., Gatidis, S., Yang, B.: Medgan: Medical image translation using gans. Computerized Medical Imaging and Graphics 79, 101684 (2020)

2. Blau, Y., Michaeli, T.: The perception-distortion tradeof. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 6228–6237 (2018)

3. Bogensperger, L., Narnhofer, D., Falk, A., Schindler, K., Pock, T.: Flowsdf: Flow matching for medical image segmentation using distance transforms. International Journal of Computer Vision 133(7), 4864–4876 (2025)

4. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Communications of the ACM 63(11), 139–144 (2020)

5. Hadzic, A., Bogensperger, L., Joham, S.J., Urschler, M.: Synthetic augmentation for anatomical landmark localization using ddpms. In: International Workshop on Simulation and Synthesis in Medical Imaging. pp. 1–12. Springer (2024)

6. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in Neural Information Processing Systems 33, 6840–6851 (2020)

7. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022)

8. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 1125–1134 (2017)

9. Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R.C., Schindler, K.: Repurposing difusion-based image generators for monocular depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9492–9502 (2024)

10. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. In: International Conference on Learning Representations (2014)

11. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: International Conference on Learning Representations (2023)

12. Mahdi, M.A., Al-Shalabi, M., Elbarougy, R., Alnfrawy, E.T., Hadi, M.U., Ali, R.F.: Ct-to-pet synthesis in the head–neck and thoracic region via conditional 3d latent difusion modeling. Bioengineering 13(5), 534 (2026)

13. Mei, J., Lin, C., Qiu, Y., Wang, Y., Zhang, H., Wang, Z., Dai, D.: Cross-modal interactive perception network with Mamba for lung tumor segmentation in PET-CT images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15653–15662 (2025)

14. Nguyen, D.T., Nguyen, T.T., Nguyen, H.T., Nguyen, T.T., Pham, H.H., Nguyen, T.H., Truong, T.N., Le Nguyen, P.: CT to PET translation: a large-scale dataset and domain-knowledge-guided difusion approach. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1498–1507. IEEE (2025)

15. Pinaya, W.H., Tudosiu, P.D., Daflon, J., Da Costa, P.F., Fernandez, V., Nachev, P., Ourselin, S., Cardoso, M.J.: Brain imaging generation with latent difusion models. In: MICCAI Workshop on Deep Generative Models. pp. 117–126. Springer (2022)

16. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)

17. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: LAION-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems 35, 25278–25294 (2022)

18. Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems 32 (2019)

19. Wolleb, J., Bieder, F., Sandkühler, R., Cattin, P.C.: Difusion models for medical anomaly detection. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 35–45. Springer (2022)

20. Xie, E., Yao, L., Shi, H., Liu, Z., Zhou, D., Liu, Z., Li, J., Li, Z.: Diffit: Unlocking transferability of large difusion models via simple parameter-eficient fine-tuning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4230–4239 (2023)

21. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 586–595 (2018)

22. Zhu, J.Y., Park, T., Isola, P., Efros, A.A.: Unpaired image-to-image translation using cycle-consistent adversarial networks. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 2223–2232 (2017)