# MedDiME: Eficient Latent Difusion with Adaptive Masking for Medical Counterfactual Generation

Yan Zeng<sup>⋆</sup>, Changlu Guo<sup>⋆</sup>, Anders Nymark Christensen, Morten Rieger Hannemose, and Anders Bjorholm Dahl

Department of Applied Mathematics and Computer Science, Technical University of Denmark {s242652,chagu,anym,mohan,abda}@dtu.dk

Abstract. Medical counterfactual generation modifies images to change model predictions for interpretability. However, existing difusion-based approaches are often prohibitively slow and memory-intensive, making them dificult to apply in high resolution settings. Moreover, existing masking strategies are tightly coupled with pixel-space representations, making them incompatible with latent-space difusion editing. To address these challenges, we propose MedDiME, a latent-space classifier-guided difusion framework that reduces computational and memory overhead while introducing a latent-compatible, gradient-driven adaptive masking mechanism for spatially precise medical counterfactual generation. Extensive experiments demonstrate that MedDiME achieves high-quality counterfactual generation with significant eficiency gains compared to prior classifier-guided difusion baselines, achieving up to 40× faster inference and 13× lower peak GPU memory usage.

Keywords: Medical Counterfactual Generation · Difusion Models

## 1 Introduction

Deep learning models in medical imaging often operate as black boxes, limiting clinical trust and practical deployment [16]. Counterfactual generation modifies regions relevant to model decisions to alter predictions [5,12]. Fundamentally, it aims to achieve decision-driven semantic reversal rather than pixel-level exact reconstruction. In medical imaging, such editing requires precise control over spatially localized, decision-relevant regions while preserving overall structural plausibility. However, early generative counterfactual methods in medical imaging sufer from limited controllability and instability, often producing semantically distorted or structurally inconsistent results [6,10,11].

Recently, Denoising Difusion Probabilistic Models [1,2] have emerged as a more stable alternative for counterfactual generation. Among them, DiME [4] incorporates classifier guidance during sampling but requires full denoising rollouts with global pixel-space guidance, resulting in high computational cost and spatially difuse edits. To mitigate these issues, FastDiME [13] improves eficiency by approximating the clean image with a single-step denoising operation and computing classifier guidance accordingly, while incorporating a selfoptimized masking scheme to enhance edit locality. Its masking strategy is based on pixel-level diferences rather than semantic or decision-level cues, which may misalign with decision-relevant regions and result in imprecise localization and over-editing [8,9]. In contrast, MaskDiME [14] constructs adaptive masks directly from classifier gradients, enabling the model to dynamically focus on decisionrelevant regions and achieve more precise, spatially localized counterfactual edits.

![](images/b29fdb3343ae50a58dceae8f65dd5db714687318e06be024f315e36ea29cc07c.jpg)  
Fig. 1: Quality–eficiency trade-of for counterfactual generation on ISIC2018 (Ruler–Non-Ruler). FID is plotted against time per counterfactual (sec), with bubble size denoting peak GPU memory. MedDiME dominates the lower-left region, achieving the best quality–eficiency–memory balance among all methods.

Although the aforementioned methods have achieved improvements in eficiency and spatial precision, medical image analysis in clinical practice typically relies on high-resolution inputs. However, existing counterfactual approaches based on pixel-space difusion are predominantly validated on natural images or relatively low-resolution settings (e.g., 128 × 128 or 256 × 256 pixels). When extended to higher resolutions (e.g., 512 × 512 pixels and above), pixel-space diffusion incurs substantially increased computational and memory costs, as these scale with the number of pixels. This rapid growth in modeling complexity poses significant challenges for high-resolution medical counterfactual generation.

Latent representations compress pixel redundancy, directing difusion toward decision-relevant structural changes consistent with the semantic reversal objective in counterfactual generation. Operating in this lower-dimensional space also reduces the computational and memory burden introduced by high-resolution inputs [3,26,27]. However, migrating classifier-guided difusion editing to the latent space is not a straightforward substitution [17,18]. Unlike pixel space, latent representations lack explicit spatial correspondence, rendering masking strategies based on pixel diferences or image-space cues conceptually incompatible.

In response to these limitations, we propose MedDiME, a latent-space framework for classifier-guided medical counterfactual generation that integrates adaptive gradient-driven spatial masking. Our contributions are threefold: (1) We conduct classifier-guided difusion for counterfactual generation entirely in latent space, demonstrating that high-resolution medical image editing can be efectively achieved within a lower-dimensional representation while maintaining high counterfactual quality and significantly improving computational eficiency, as shown in Fig.1. (2) Rather than directly transferring pixel-space masking, we develop a latent-compatible classifier-gradient-driven adaptive masking mechanism that redefines spatial guidance within the latent representation for precise, spatially localized editing. (3) We show that MedDiME not only significantly improves computational eficiency, but also achieves state-of-the-art or competitive performance across multiple medical image counterfactual explanation tasks.

![](images/0454dc9f496489104a319a73a9452f2c80d317ae7545bb11ec4b0fca53e03996.jpg)  
Fig. 2: Overview of the MedDiME framework. Latent states are decoded only for visualization; all operations are conducted in the latent space.

## 2 Methodology

## 2.1 Framework overview

Building upon the classifier-guided difusion paradigm of DiME [4], we introduce an entirely latent-space framework for medical counterfactual generation (Fig. 2).

Given an input image X, we first encode it into a latent representation x using a pretrained VAE encoder. The latent representation is then difused forward to a predefined timestep $\tau \ ( 1 \leq \tau \leq T )$ , yielding a noisy latent variable $\hat { z } _ { \tau }$ . We initialize the reverse difusion process with $z _ { \tau } = \hat { z } _ { \tau }$ and iteratively refine the latent state under gradient guidance to obtain the final counterfactual latent $z _ { 0 } .$

At each reverse difusion step t, we additionally maintain a latent estimate $x _ { t } ,$ which represents a denoised approximation of the current noisy latent state and is used for gradient $\nabla _ { z _ { t } }$ computation (see Sec. 2.2 for details). Specifically, the noisy latent variable is updated at each reverse difusion step as:

$$
z _ { t - 1 } = M _ { t } \odot \mathcal { N } \big ( \mu _ { \theta } ( z _ { t } ) - \Sigma _ { \theta } ( z _ { t } ) \nabla _ { z _ { t } } , \Sigma _ { \theta } ( z _ { t } ) \big ) + ( 1 - M _ { t } ) \odot \hat { z } _ { t - 1 } ,\tag{1}
$$

where ⊙ denotes element-wise multiplication, $\mu _ { \theta } ( \cdot )$ and $\textstyle \sum _ { \theta } ( \cdot )$ are the predicted mean and variance of the reverse difusion process, and $\hat { z } _ { t - 1 }$ is obtained by applying the forward difusion process to the original latent representation. This formulation restricts classifier-guided denoising updates to the regions indicated by the adaptive mask $M _ { t }$ (see Sec. 2.3 for details), while preserving the original difusion trajectory outside the masked regions. After obtaining $z _ { t - 1 }$ , we apply a one-step denoising operation based on Tweedie’s formula [19], following FastDiME [13], to estimate the corresponding clean latent representation.

$$
\hat { x } _ { 0 } ^ { ( t - 1 ) } = \frac { z _ { t - 1 } - \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \epsilon _ { \theta } \big ( z _ { t - 1 } \big ) } { \sqrt { \bar { \alpha } _ { t - 1 } } } .\tag{2}
$$

This estimate avoids full reverse denoising and reduces computation. The estimated latent state is then used to construct the next latent state:

$$
x _ { t - 1 } = M _ { t } \odot \hat { x } _ { 0 } ^ { ( t - 1 ) } + ( 1 - M _ { t } ) \odot x ,\tag{3}
$$

where the adaptive mask $M _ { t }$ restricts updates to the masked latent regions while preserving the original latent representation elsewhere. This masked latent state $x _ { t - 1 }$ is subsequently used for gradient computation and mask construction at the next reverse difusion step.

The above procedure is repeated for all reverse difusion timesteps until the final latent state $z _ { 0 }$ is obtained, which is then decoded by the VAE decoder to produce the counterfactual image $X _ { \mathrm { c f } }$

## 2.2 Gradient-Based Guidance in Latent Space

We follow DiME [4] and optimize a joint loss composed of a classification term $L _ { \mathrm { c l a s s } } .$ , a perceptual term $L _ { \mathrm { p e r c } } .$ , and an $L _ { 1 }$ regularization term. The classification loss drives the latent toward the target class, while the perceptual and $L _ { 1 }$ terms preserve structural consistency. The joint loss is formulated as:

$$
L ( x _ { t } ; y , x ) = \lambda _ { c } L _ { \mathrm { c l a s s } } ( C ( y \mid x _ { t } ) ) + \lambda _ { p } L _ { \mathrm { p e r c } } ( x _ { t } , x ) + \lambda _ { l } L _ { L _ { 1 } } ( x _ { t } , x ) ,\tag{4}
$$

where $\lambda _ { c } , \lambda _ { p }$ , and $\lambda _ { l }$ control the relative importance of each term. We fix $\lambda _ { c } = 5 \AA$ $\lambda _ { p } = 5 0$ , and $\lambda _ { l } = 1$ for all experiments. $C ( \boldsymbol { y } \mid \boldsymbol { x } _ { t } )$ denotes the predicted probability of the target class y given the latent estimate $x _ { t }$

To enable gradient-based guidance during the reverse difusion process, we compute the gradient of the loss function L with respect to the noisy latent variable $z _ { t }$ . To avoid backpropagation through the U-Net [24], we follow MaskDiME [14] and approximate the gradient with respect to $z _ { t }$ as:

$$
\nabla _ { z _ { t } } = \frac { s } { \sqrt { \bar { \alpha } _ { t } } } \nabla _ { x _ { t } } L ( x _ { t } ; y , x ) ,\tag{5}
$$

where s is a scalar factor controlling the overall guidance strength, and we set $s = 2 0 0$ by default in all experiments.

## 2.3 Adaptive Masking in Latent Space

To impose spatial constraints during latent-space difusion, we train a classifier directly on latent representations and construct an adaptive mask at each reverse difusion step based on its gradients. Unlike methods that define masks in the pixel space (e.g., FastDiME [13] and MaskDiME [14]) or generate pixel-level masks and downsample them to the latent resolution (e.g., Blended Latent Difusion [26]), our spatial constraint is constructed within the latent feature domain. This eliminates the need for pixel-to-latent mask projection and enables more accurate alignment with decision-sensitive regions in the latent representation.

Specifically, we first compute the spatial gradient induced by the classification loss and approximately map it to the noisy latent variable $z _ { t } \colon$

$$
\nabla _ { z _ { t } } ^ { \mathrm { c l a s s } } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \nabla _ { x _ { t } } L _ { \mathrm { c l a s s } } ( C ( y \mid x _ { t } ) ) .\tag{6}
$$

Here, $x _ { t }$ denotes the latent estimate at step $t ,$ and $C ( \cdot )$ is a latent-space classifier.

We then take the element-wise absolute value and average over the channel dimension to obtain a scalar saliency map:

$$
G _ { t } = \left| \nabla _ { z _ { t } } ^ { \mathrm { c l a s s } } \right| _ { \mathrm { a v g } } \in \mathbb { R } ^ { 1 \times H \times W } .\tag{7}
$$

The adaptive mask $M _ { t }$ is defined by selecting the top-k% values of $G _ { t } \mathrm { : }$ for each spatial location $( i , j )$ , we set $M _ { t } ( i , j ) = 1 \mathrm { i f } G _ { t } ( i , j )$ is among the top-k%, and 0 otherwise. We set $k = 2 0 \%$ for both ISIC2018 tasks and $k = 3 0 \%$ for CheXpert. This mask confines classifier-guided updates to decision-relevant latent regions while preserving the original difusion elsewhere. To enhance spatial coherence, we apply morphological dilation with a kernel size of 5.

## 3 Experiments and Results

Datasets. We evaluate MedDiME on three tasks, ranging from editing salient but non-pathological visual structures to clinically meaningful pathological semantic transformations. CheXpert [20] and ISIC2018 [21] provide chest X-ray and dermoscopic skin lesion images, respectively. On both datasets, we consider counterfactual edits of non-pathological display structures [10,11], including pacemakers in CheXpert and ruler markers in ISIC2018. We further evaluate clinical counterfactuals between Nevus and Melanoma on ISIC2018. All datasets are randomly split into training, validation, and test sets with a ratio of 7:2:1. The classifier and difusion models are trained on the training set and validated on the validation set. Counterfactual generation and quantitative evaluation are conducted on the held-out test set.

Evaluation Metrics. As MedDiME follows the classifier-guided paradigm of DiME [4], we restrict comparisons to methods within the same framework to ensure methodological consistency and fair evaluation. We follow the FastDiME [13] and MaskDiME [14] evaluation protocol and assess counterfactual quality from three perspectives: realism, proximity, and validity [8,9]. Specifically, we report Fréchet Inception Distance (FID) [23] for realism, pixel-level $L _ { 1 }$ distance and SimSiam-based semantic similarity (S<sup>3</sup>) [22] for proximity, and Flip Rate (FR) together with Mean Absolute Diference (MAD) for validity.

![](images/15d39566c9835c0819f94be7d58d6a4e3d0dcc549cc439c1be5a7ffcdcd4cf0a.jpg)  
Fig. 3: Training eficiency comparison. Left: Classifier training in latent and pixel spaces, validated by AUROC over time. Right: Difusion model training in latent and pixel spaces, validated by FID over time.

Implementation Details. MedDiME operates in latent space, whereas the baselines operate in pixel space [4,13,14]. For fair comparison, we standardize the core architectures and sampling configurations across methods, including a U-Net-based difusion backbone [24], a DenseNet121 classifier [25], the total number of difusion steps $( T = 1 0 0 0 )$ , and the starting noise level for counterfactual generation $( t = 4 0 0 )$ . For image-space methods, inputs are resized to a high resolution of 512×512 pixels. For latent-space modeling, we adopt the pretrained VAE from Stable Difusion XL (SDXL) [3] as a frozen encoder–decoder to map images into a 4 × 64 × 64-dimensional latent representation and reconstruct the edited outputs. Unless otherwise specified, all remaining hyperparameters follow DiME [4]. All experiments are conducted on a single NVIDIA A100 80GB GPU. Pipeline-Level Eficiency Comparison. To quantify eficiency gains, we evaluate the entire counterfactual pipeline, including classifier training, difusion training, and counterfactual inference.

As shown in Fig. 3 (left), the latent-space classifier converges $3 . 0 { \times } { - } 6 . 8 { \times }$ faster while maintaining comparable or improved validation AUROC. During diffusion training (Fig. 3 (right)), unconditional samples are periodically generated to compute FID against the validation set. The latent-space model converges $4 . 2 { \times } \mathrm { - } 1 0 . 2 { \times }$ faster and achieves lower FID, eliminating the 40–80 hour training cost of pixel-space difusion. More importantly, during counterfactual generation (Fig. 1), inference time is reduced by 2.5×, 4×, and over 40× compared to MaskDiME, FastDiME, and DiME, respectively, while peak GPU memory usage drops by 7×, 13×, and 10×. MedDiME also attains the lowest FID, demonstrating that large-scale eficiency gains do not compromise generative quality.

Comparison with State-of-the-Art Methods Quantitative results in Table 1 and Table 2 show that MedDiME achieves the best or near-best performance across tasks. For realism, operating entirely in latent space mitigates high-frequency artifacts inherent to pixel-space difusion [3], yielding the lowest or near-lowest FID. For proximity, classifier-gradient-driven masking restricts updates to decision-relevant regions while preserving the remaining latent structure [15]. This leads to minimal or near-minimal pixel-level deviation $\left( L _ { 1 } \right)$ and the highest SimSiam similarity $\left( \mathrm { S ^ { 3 } } \right)$ [22]. For validity, sharing the classifier gradient between difusion guidance and spatial masking enables targeted movement toward the decision boundary without enlarging the editable region [2]. This results in the best Flip Rate (FR) and strong MAD [7].

Table 1: Quantitative results of counterfactual generation on ISIC2018. Best and second-best results are shown in bold and underline, respectively.
<table><tr><td rowspan="3">Method</td><td colspan="5">Melanoma-Nevus</td><td colspan="5">Ruler-Non Ruler</td></tr><tr><td>|FID↓</td><td>L1↓</td><td>MAD↑</td><td> $\mathrm { S ^ { 3 } \uparrow }$ </td><td></td><td>FR↑|FID↓</td><td>L1↓</td><td>MAD↑</td><td> $\mathrm { S ^ { 3 } \uparrow }$ </td><td>FR↑</td></tr><tr><td>DiME [4]</td><td>62.4</td><td>0.080</td><td>0.611</td><td>0.911</td><td>91.8</td><td>55.5</td><td>0.086</td><td>0.864</td><td>0.944</td><td>90.1</td></tr><tr><td>FastDiME [13]</td><td>36.9</td><td>0.051</td><td>0.594</td><td>0.947</td><td>95.3</td><td>41.1</td><td>0.073</td><td>0.837</td><td>0.963</td><td>89.9</td></tr><tr><td>FastDiME-2 [13]</td><td>35.1</td><td>0.069</td><td>0.594</td><td>0.943</td><td>95.3</td><td>41.3</td><td>0.074</td><td>0.827</td><td>0.946</td><td>91.3</td></tr><tr><td>FastDiME-2+ [13]</td><td>38.8</td><td>0.045</td><td>0.589</td><td>0.964</td><td>96.1</td><td>47.2</td><td>0.061</td><td>0.810</td><td>0.964</td><td>90.8</td></tr><tr><td>MaskDiME [14]</td><td></td><td>0.039</td><td>0.590</td><td>0.948</td><td>93.4</td><td>34.2</td><td>0.084</td><td>0.808</td><td>0.922</td><td>91.0</td></tr><tr><td>MedDiME</td><td>33.3 24.3</td><td>0.050</td><td>0.633</td><td>0.966</td><td>96.6</td><td>28.5</td><td>0.070</td><td>0.843</td><td>0.965</td><td>91.8</td></tr></table>

Table 2: Quantitative results comparison on CheXpert.
<table><tr><td>Method</td><td>FID↓ L1↓ MAD↑</td><td> $\mathrm { S ^ { 3 } \uparrow }$  FR↑</td></tr><tr><td>DiME</td><td>51.7 0.088</td><td>0.956 0.902 97.7</td></tr><tr><td>FastDiME</td><td>29.20.058 0.959</td><td>0.927100.0</td></tr><tr><td>FastDiME-2</td><td>30.4 0.069 0.953 0.881 99.8</td><td></td></tr><tr><td>FastDiME-2+</td><td>38.10.052 0.9540.928100.0</td><td></td></tr><tr><td>MaskDiME</td><td>22.0 0.0530.9540.935100.0</td><td></td></tr><tr><td>MedDiME</td><td>22.30.0530.9610.940100.0</td><td></td></tr></table>

Table 3: Ablation study of MedDiME on ISIC2018 (Ruler–Non Ruler).
<table><tr><td>Method FID↓</td><td>L1↓ MAD↑  $\mathrm { S ^ { 3 } \uparrow }$  FR↑</td></tr><tr><td>MaskDiME</td><td>34.2 0.084 0.808 0.922 91.0</td></tr><tr><td>w/o Mask</td><td>62.1 0.126 0.795 0.64376.5</td></tr><tr><td>Pixel Diff Mask 44.7 0.090</td><td>0.805 0.797 88.9</td></tr><tr><td>Fixed Mask</td><td>26.30.074 0.812 0.883 90.2</td></tr><tr><td>MedDiME</td><td>28.50.070 0.843 0.96591.8</td></tr></table>

Qualitative results in Fig. 4 further corroborate the quantitative results. In Melanoma↔Nevus transformations, edits are primarily confined to lesion regions. For salient but non-pathological visual markers (e.g., Ruler and Pacemaker tasks), modifications are restricted to the corresponding display structures, while surrounding non-diagnostic regions remain largely unchanged. In contrast, baseline methods [4,13] tend to produce more dispersed edits, accompanied by additional background changes outside the target regions.

Ablation Study. We compare diferent spatial masking strategies on ISIC2018 (Ruler–Non Ruler) (Table 3). Without masking, gradient updates are applied globally, inevitably perturbing non-decision regions and degrading all metrics. Pixel Diference Mask imposes spatial constraints based on pixel discrepancies rather than classifier gradients, partially reducing global perturbations but failing to consistently localize decision-relevant regions, resulting in limited performance gains. Fixed Mask further strengthens spatial restriction, improving FID and $L _ { 1 }$ due to reduced edit magnitude and slightly increasing MAD; however, its static design cannot adapt to evolving gradient directions across difusion steps, limiting its ability to maintain decision-aligned updates. In contrast, Med-

![](images/62070cd3306fbd9f094371dc3cf7d0e89f829fe4a010e1f3be9eb1d11a407b8b.jpg)  
Fig. 4: Qualitative counterfactual results across datasets. Edited regions are highlighted in yellow. Numbers indicate predicted class probabilities (1 = Melanoma / Ruler / Pacemaker, 0 = Nevus / Non-Ruler / Non-Pacemaker).

DiME dynamically aligns the editable region with classifier gradients at each reverse step, ensuring that updates remain spatially focused and decision-aligned, thereby achieving the strongest overall performance.

## 4 Discussion and Conclusion

We present MedDiME, a fully latent-space framework for classifier-guided medical counterfactual generation. By incorporating adaptive gradient-driven spatial masking, it substantially reduces computational and memory overhead without compromising counterfactual quality. Nevertheless, the structurally intricate and detail-dense nature of pacemakers makes realistic synthesis challenging for current difusion models. Consistent with FastDiME [13], we therefore restrict evaluation on CheXpert to pacemaker removal. Furthermore, although 512×512 resolution and the adopted VAE suficiently support the current experiments, future work will explore higher resolutions (e.g., 1024 × 1024) and medical-specific latent representation models to better capture fine-grained structural details.

## References

1. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. In: Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, pp. 6840–6851 (2020)

2. Dhariwal, P., Nichol, A.: Difusion models beat GANs on image synthesis. In: Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, pp. 8780–8794 (2021)

3. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 10684–10695 (2022)

4. Jeanneret, G., Simon, L., Jurie, F.: Difusion models for counterfactual explanations. In: Proc. Asian Conf. Comput. Vis. (ACCV), pp. 858–876 (2022)

5. Augustin, M., Boreiko, V., Croce, F., Hein, M.: Difusion visual counterfactual explanations. In: Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, pp. 364–377 (2022)

6. Khorram, S., Fuxin, L.: Cycle-consistent counterfactuals by latent transformations. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 10203– 10212 (2022)

7. Jeanneret, G., Simon, L., Jurie, F.: Adversarial counterfactual visual explanations. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 16425– 16435 (2023)

8. Väth, P., Frühwald, A. M., Paassen, B., Gregorová, M.: Difusion-based visual counterfactual explanations—towards systematic quantitative evaluation. In: Joint Eur. Conf. Mach. Learn. Knowl. Discov. Databases (ECML PKDD), pp. 120–135 (2025)

9. Melistas, T., Spyrou, N., Gkouti, N., Sanchez, P., Vlontzos, A., Panagakis, Y., Papanastasiou, G., Tsaftaris, S. A.: Benchmarking counterfactual image generation. In: Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, pp. 133207–133230 (2024)

10. Pombo, G., Gray, R., Cardoso, M. J., Ourselin, S., Rees, G., Ashburner, J., Nachev, P.: Equitable modelling of brain imaging by counterfactual augmentation with morphologically constrained 3D deep generative models. Med. Image Anal. 84, 102723 (2023)

11. Pahde, F., Dreyer, M., Samek, W., Lapuschkin, S.: Reveal to revise: An explainable AI life cycle for iterative bias correction of deep models. In: Med. Image Comput. Comput. Assist. Interv. (MICCAI), pp. 596–606 (2023)

12. Singla, S., Pollack, B., Chen, J., Batmanghelich, K.: Explanation by progressive exaggeration. arXiv:1911.00483 (2019)

13. Weng, N., Pegios, P., Petersen, E., Feragen, A., Bigdeli, S.: Fast difusion-based counterfactuals for shortcut removal and generation. In: Eur. Conf. Comput. Vis. (ECCV), pp. 338–357 (2024)

14. Guo, C., Christensen, A.N., Dahl, A.B., Hannemose, M.R.: MaskDiME: Adaptive masked difusion for precise and eficient visual counterfactual explanations. arXiv:2602.18792 (2026)

15. Sobieski, B., Grzywaczewski, J., Sadlej, B., Tivnan, M., Biecek, P.: Rethinking visual counterfactual explanations through region constraint. In: Int. Conf. Learn. Represent. (ICLR) (2025)

16. Haselhof, A., Trelenberg, K., Küppers, F., Schneider, J.: The Gaussian discriminant variational autoencoder (GDVAE): A self-explainable model with counterfactual explanations. In: Eur. Conf. Comput. Vis. (ECCV), pp. 305–322 (2024)

17. Yeganeh, Y., Farshad, A., Charisiadis, I., Hasny, M., Hartenberger, M., Ommer, B., Navab, N., Adeli, E.: Latent drifting in difusion models for counterfactual medical image synthesis. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 7685–7695 (2025)

18. Kazimi, T., Allada, R., Yanardag, P.: Explaining in difusion: Explaining a classifier with difusion semantics. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 14799–14809 (2025)

19. Efron, B.: Tweedie’s formula and selection bias. J. Am. Stat. Assoc. 106(496), 1602–1614 (2011)

20. Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., et al.: CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In: Proc. AAAI Conf. Artif. Intell., pp. 590–597 (2019)

21. Codella, N., Rotemberg, V., Tschandl, P., Celebi, M. E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., et al.: Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (ISIC). arXiv:1902.03368 (2019)

22. Chen, X., He, K.: Exploring simple siamese representation learning. In: Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 15750–15758 (2021)

23. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In: Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30 (2017)

24. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: Med. Image Comput. Comput. Assist. Interv. (MIC-CAI), pp. 234–241 (2015)

25. Huang, G., Liu, Z., van der Maaten, L., Weinberger, K. Q.: Densely connected convolutional networks. In: Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 4700–4708 (2017)

26. Luu, T., Le, N., Le, D., Le, B.: From visual explanations to counterfactual explanations with latent difusion. In: IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), pp. 420–429 (2025)

27. Jeanneret, G., Simon, L., Jurie, F.: Text-to-image models for counterfactual explanations: a black-box approach. In: IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), pp. 4757–4767 (2024)