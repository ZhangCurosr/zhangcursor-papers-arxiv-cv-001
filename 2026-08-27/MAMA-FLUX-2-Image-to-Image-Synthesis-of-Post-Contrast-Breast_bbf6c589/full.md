# MAMA-FLUX.2: Image-to-Image Synthesis of Post-Contrast Breast DCE-MRI for the MAMA-SYNTH Challenge

Kamil Kwarciak<sup>1[0000−0002−1392−4291]</sup> and Marek Wodzinski<sup>1,2[0000−0002−8076−6246]</sup>

<sup>1</sup> Department of Measurement and Electronics, AGH University of Krakow, Krakow, Poland

<sup>2</sup> Sano Centre for Computational Medicine, Krakow, Poland {kwarciak,wodzinski}@agh.edu.pl

Abstract. Dynamic contrast-enhanced breast MRI is central to cancer diagnosis and monitoring, but requires gadolinium-based contrast agents. In this work, we address pre-to-post contrast breast MRI synthesis for the MAMA-SYNTH challenge. We propose MAMA-FLUX.2, a conditional latent flow-matching approach based on FLUX.2-Klein-4B. The pre-contrast image is encoded as spatial conditioning, while the model predicts the flow field associated with the post-contrast target latent. To adapt the pretrained model eficiently, we use LoRA finetuning and introduce a regional training objective combining global flow matching, tumor-region supervision, and stable foreground regularization. We further investigate LoRA rank, intensity windowing, and regional loss weights on axial slices, prioritizing clinically relevant tumorfocused metrics. Our ablation study shows that moderate tumor and stable-foreground weighting improves the trade-of between image fidelity and tumor-region accuracy. The final model achieves the best overall balance with LoRA rank/α = 64/64, MHA<sub>max</sub> = 25, $\lambda _ { \mathrm { t u m o r } } = 0 . 2 5$ and $\lambda _ { \mathrm { s t a b l e } } ~ = ~ 0 . 1$ . These results demonstrate that compact pretrained rectified-flow transformers can be adapted for contrast-enhanced MRI synthesis using parameter-eficient fine-tuning and task-aware regional losses.

Keywords: MAMA-SYNTH · Breast MRI Synthesis · Rectified Flow

## 1 Introduction

Dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) is central to breast cancer imaging, as contrast uptake provides clinically important information about lesion conspicuity, morphology, and enhancement behaviour. However, post-contrast imaging requires intravenous gadolinium administration, increasing examination complexity and patient burden and limiting use in some clinical settings. This has motivated interest in virtual contrast enhancement, where post-contrast images are synthesized from pre-contrast acquisitions.

The MAMA-SYNTH Challenge formalizes this task by evaluating methods that synthesize peak-enhanced post-contrast breast MR images from corresponding pre-contrast inputs [7]. The task is challenging because synthesized images must show plausible enhancement while preserving patient-specific anatomy. Since clinically relevant enhancement is often localized to small tumor regions, global image-level objectives alone may be insuficient. Efective models must therefore produce realistic images, preserve tumor-specific enhancement patterns, and maintain anatomical structures relevant to segmentation and imagelevel evaluation.

Recent difusion and flow-based generative models have shown strong performance in high-fidelity image synthesis and image-to-image translation. Latent generative modelling improves eficiency by operating in a compressed representation rather than pixel space [8], while flow matching and rectified-flow formulations provide an efective framework for learning transformations between noise and data distributions [4, 5]. These properties make latent rectified-flow models well suited to conditional medical image synthesis, where realism and structural consistency are both required.

In this work, we adapt FLUX.2-Klein-4B, a compact latent rectified-flow transformer backbone [1], to pre-to-post contrast breast MRI synthesis. We formulate the task as conditional latent rectified-flow modeling: the pre-contrast image is encoded as spatial conditioning, while the model predicts the flow field for a noised post-contrast target latent. To better match the clinical structure of the task, we introduce a three-component objective combining global latent flow matching, tumor-region supervision, and stable-foreground preservation. This encourages accurate contrast synthesis while discouraging unnecessary changes in non-tumor foreground anatomy.

Contributions. Our contributions are threefold. First, we adapt a pretrained latent rectified-flow transformer to conditional pre-to-post contrast breast MRI synthesis. Second, we introduce a tumor-aware and anatomy-preserving training objective that combines global latent flow matching with regional supervision. Third, we evaluate the efects of LoRA rank, LoRA scaling, and regional loss weighting through an ablation study using both image-quality and tumor-focused metrics.

## 2 Methods

We formulate pre-to-post contrast synthesis as conditional atent rectified-flow modeling [4, 5]. Our method builds on FLUX.2-Klein-4B [1], a compact rectifiedflow transformer backbone inspired by recent high-resolution rectified-flow transformers [2]. Following the latent difusion paradigm [8], we operate in the VAE latent space and adapt the model to the MAMA-SYNTH task of synthesizing peak-enhanced post-contrast breast DCE-MRI from pre-contrast input [7].

## 2.1 Conditional Latent Flow Matching

Let $x _ { 0 }$ denote the pre-contrast conditioning image and $y _ { 1 }$ the corresponding clean post-contrast target image. Their VAE latents are given by $z _ { 0 } = E ( x _ { 0 } )$ ， and $z _ { 1 } = E ( y _ { 1 } )$ , where $E ( \cdot )$ denotes the VAE encoder. Given Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ and a sampled noise level $t \in [ 0 , 1 ]$ , we construct the noised target latent as

$$
z _ { t } = ( 1 - t ) z _ { 1 } + t \epsilon .\tag{1}
$$

Thus, $t = 0$ corresponds to the clean target latent, while $t = 1$ corresponds to pure noise. The corresponding flow-matching target is

$$
u = \epsilon - z _ { 1 } .\tag{2}
$$

Equivalently, since $\epsilon = z _ { 1 } + u$ , we have

$$
z _ { t } = z _ { 1 } + t u ,\tag{3}
$$

and therefore

$$
z _ { 1 } = z _ { t } - t u .\tag{4}
$$

The rectified-flow transformer is conditioned on the pre-contrast latent $z _ { \mathrm { 0 } } ,$ the text embedding $c ,$ and the timestep embedding, and predicts the flow field

$$
\hat { u } _ { \theta } = f _ { \theta } ( z _ { t } , z _ { 0 } , c , t ) .\tag{5}
$$

The corresponding estimate of the clean post-contrast latent is

$$
\hat { z } _ { 1 } = z _ { t } - t \hat { u } _ { \theta } ,\tag{6}
$$

and the synthetic post-contrast image is reconstructed by VAE decoding,

$$
\hat { y } _ { 1 } = D ( \hat { z } _ { 1 } ) ,\tag{7}
$$

where $D ( \cdot )$ denotes the VAE decoder.

## 2.2 Tumor-Aware and Anatomy-Preserving Objective

We train the model using a three-component objective. The first component is the standard global flow-matching loss. For latent element $k ,$ corresponding to a channel and spatial position in the latent-error tensor, the squared flow-matching error is

$$
\begin{array} { r } { e _ { k } = \left( \hat { u } _ { \theta , k } - u _ { k } \right) ^ { 2 } . } \end{array}\tag{8}
$$

The global loss is defined as

$$
\mathcal { L } _ { \mathrm { g l o b a l } } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } e _ { k } ,\tag{9}
$$

where $N$ is the total number of latent elements.

The second component uses the available tumor mask to emphasize accurate synthesis within the lesion region. Let $m _ { k } ^ { T }$ denote the tumor mask resized to the latent-error resolution and broadcast over latent channels. The tumor-region loss is

$$
\mathcal { L } _ { \mathrm { t u m o r } } = \frac { \sum _ { k = 1 } ^ { N } m _ { k } ^ { T } e _ { k } } { \sum _ { k = 1 } ^ { N } m _ { k } ^ { T } } .\tag{10}
$$

The third component encourages preservation of stable foreground anatomy outside the tumor. We define an image-space stable-foreground mask as

$$
m ^ { S } = { \bf 1 } [ m ^ { T } = 0 ] { \bf 1 } [ x _ { 0 } \geq \tau _ { \mathrm { f g } } ] { \bf 1 } [ y _ { 1 } - x _ { 0 } \leq \tau _ { \mathrm { e n h } } ] ,\tag{11}
$$

where $\tau _ { \mathrm { f g } }$ suppresses background regions and $\tau _ { \mathrm { e n h } }$ excludes strongly enhancing tissue. We set these thresholds empirically to $\tau _ { \mathrm { f g } } = - 0 . 4 5$ and $\tau _ { \mathrm { e n h } } = 0 . 5$ . After resizing this mask to the latent-error resolution and broadcasting it over latent channels, we obtain $m _ { k } ^ { S }$ . The stable-foreground loss is

$$
{ \mathcal { L } } _ { \mathrm { s t a b l e } } = { \frac { \sum _ { k = 1 } ^ { N } m _ { k } ^ { S } e _ { k } } { \sum _ { k = 1 } ^ { N } m _ { k } ^ { S } } } .\tag{12}
$$

The final training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { g l o b a l } } + \lambda _ { \mathrm { t u m o r } } \mathcal { L } _ { \mathrm { t u m o r } } + \lambda _ { \mathrm { s t a b l e } } \mathcal { L } _ { \mathrm { s t a b l e } } ,\tag{13}
$$

where $\lambda _ { \mathrm { t u m o r } }$ and $\lambda _ { \mathrm { s t a b l e } }$ control the relative contributions of the tumor-region and stable-foreground objectives, respectively.

## 3 Experiments

## 3.1 Dataset

We use MAMA-SYNTH challenge data, derived from the MAMA-MIA breast DCE-MRI dataset [3]. Each sample contains a paired pre-contrast image and peak-enhanced post-contrast target. Images are provided as 2D z-score-normalized MHA slices, with tumor masks for regional supervision and evaluation.

## 3.2 Preprocessing

We follow the oficial MAMA-SYNTH preprocessing protocol to convert DCE-MRI volumes into paired 2D slices. For each case, the pre-contrast acquisition is used as the conditioning input, while the peak-enhanced post-contrast slice is used as the synthesis target. All slices are resampled and stored as 2D MHA images, with spatial metadata preserved for evaluation. Image intensities are z-score normalized using the dataset-level pre-contrast statistics provided by the challenge. During training and inference, normalized MHA intensities are linearly clipped and mapped to the image range expected by the pretrained latent rectified-flow model. Unless otherwise stated, we use [−0.51, 25] as the zscore window for the final model, while alternative upper bounds are evaluated in the ablation study. Since the pretrained model expects three-channel inputs, each scalar MHA slice is replicated across the RGB channels before VAE encoding. After decoding, the prediction is converted back to a single-channel normalized MHA slice.

## 3.3 Experimental Setup

Anatomical Plane Routing Initial experiments suggested that plane-specific adaptation improves synthesis quality. We therefore train separate LoRA adapters for axial and sagittal slices. Although the MAMA-SYNTH validation and test sets are expected to contain axial slices, this routing mechanism supports both acquisition planes (Figure 1). A lightweight binary logistic-regression classifier predicts the anatomical plane of each pre-contrast slice and selects the corresponding adapter for inference. Image synthesis is then performed by the FLUX.2-Klein-4B backbone conditioned on the input image. The classifier uses hand-crafted features extracted from the 2D MHA slices, including image dimensions, intensity statistics and percentiles, threshold-based pixel fractions, foreground shape descriptors, and coarse row- and column-wise occupancy profiles. The foreground mask is defined as $x _ { 0 } \geq \tau _ { \mathrm { f g } } .$ , with $\tau _ { \mathrm { f g } } = - 0 . 4 5$ , consistent with the stable-foreground loss. Features are standardized, and class imbalance is addressed using balanced logistic-regression weights.

FLUX.2 Fine-Tuning We fine-tune FLUX.2-Klein-4B using low-rank adaptation (LoRA), while keeping the VAE, text encoder, and pretrained transformer backbone frozen. LoRA layers are inserted into the main attention projections of the transformer, including the query, key, value, and output projections of the double transformer blocks, as well as the joint projection and attention output projections of the single transformer blocks. This restricts optimization to a small set of adapter parameters while preserving the pretrained generative prior. The LoRA fine-tuning procedure is presented in Figure 2. For all fine-tuning runs, we use the fixed text prompt: “Synthesize the matching peakenhancement post-contrast breast DCE-MRI slice from this pre-contrast breast DCE-MRI slice. Preserve anatomy, acquisition plane, lesion location, and keep non-breast background dark.“ We perform the main hyperparameter search on axial slices, since the challenge validation and test sets are indicated to be axial. Based on these experiments, we transfer the selected configuration to the sagittal setting. The ablation study varies the LoRA rank and scaling factor, using tied values $r / \alpha \in { 1 6 } / { 1 6 } , 3 2 / { 3 2 } , 6 4 / 6 4$ , the upper MHA intensity window $\mathrm { M H A } _ { \mathrm { m a x } } \in 2 5 , 3 5 , 4 0$ , and the regional loss weights $\lambda _ { \mathrm { t u m o r } }$ and $\lambda _ { \mathrm { s t a b l e } }$ . The final selected configuration uses $r = \alpha = 6 4$ , MHAmax = 25, λtumor = 0.25, and $\lambda _ { \mathrm { s t a b l e } } = 0 . 1$ , which provides the best overall trade-of between tumor-focused metrics and global image quality. This final adapter contains 66,846,720 trainable parameters, corresponding to 1.72% of the 3,875,544,576 parameters in the frozen base transformer. The resulting LoRA adapter occupies 255 MiB in memory.

![](images/66ad993668bfd733cbe609aa592562c925eb0ecee7995c0a1d7a6013d86cbaef.jpg)  
Fig. 1. Plane-aware inference routing. A lightweight binary classifier predicts the anatomical plane of the input pre-contrast slice and selects the corresponding axial or sagittal LoRA adapter. The selected adapter is then used by the frozen FLUX.2- Klein-4B backbone for conditional post-contrast synthesis.

![](images/6b868af93b7a0853ea1d99f8f6b30c6b703fbdba36e833eb26db040a8ee3f1e5.jpg)  
Fig. 2. Overview of the proposed MAMA-FLUX.2 fine-tuning pipeline. The precontrast slice is encoded into the VAE latent space and used as spatial conditioning for a frozen FLUX.2-Klein-4B rectified-flow transformer. Only LoRA adapters inserted into the attention projections are optimized, while the VAE, text encoder, and pretrained transformer weights remain frozen.

Models are trained with AdamW [6], a constant learning rate, bf16 mixed precision, FSDP, gradient checkpointing, a per-device batch size of 1, and gradient accumulation over 4 steps. This results in an efective batch size of 16 on four NVIDIA GH200 GPUs with 96 GB of VRAM each. At inference, we use 50 rectified-flow sampling steps with guidance scale 4.0. On the evaluation slices used for runtime profiling, with spatial sizes ranging from 256×256 to $5 1 2 \times 5 1 2$ the model call required 11.21 s per slice on average.

Table 1. Ablation study of LoRA rank, MHA maximum value, and regional loss weights. Each row denotes one complete configuration. Results are reported as mean ± standard deviation. Best mean values are highlighted in bold.
<table><tr><td>LoRA MHA rank/α</td><td>max</td><td></td><td> $\lambda _ { \mathrm { t u m o r } } ~ \lambda _ { \mathrm { s t a b l e } }$ </td><td>DSC ↑</td><td>HD95 ↓</td><td>LPIPS ↓</td><td>MSE↓</td><td>Tumor SSIM ↑</td></tr><tr><td>16/16</td><td>35</td><td>0.25</td><td>0</td><td> $0 . 8 1 9 2 \pm 0 . 1 2 1 2$ </td><td>31.0522 ± 62.1443</td><td>0.0913 ± 0.0256</td><td> $0 . 2 9 5 6 \pm 0 . 4 1 5 4$ </td><td> $0 . 7 7 1 7 \pm 0 . 1 1 9 2$ </td></tr><tr><td>16/16</td><td>35</td><td>1</td><td>0</td><td>0.7472 ± 0.2565</td><td>72.9320 ± 147.1482</td><td>0.1127 ± 0.0274</td><td>0.3858 ± 0.6377</td><td>0.8010 ± 0.1045</td></tr><tr><td>16/16</td><td>35</td><td>2</td><td>0</td><td>0.7679 ± 0.2075</td><td>81.3741 ± 148.7569</td><td>0.1100 ± 0.0307</td><td>0.3547 ± 0.5304</td><td> $0 . 7 6 1 6 \pm 0 . 1 8 8 2$ </td></tr><tr><td>32/32</td><td></td><td>0.25</td><td></td><td>0.8436 ± 0.1036</td><td>28.8759 ± 60.5567</td><td>0.0857 ± 0.0253</td><td></td><td></td></tr><tr><td>32/32</td><td>35 35</td><td>1</td><td>0 0</td><td>0.8110 ± 0.2001</td><td>58.5668 ± 143.8272</td><td>0.1016 ± 0.0278</td><td>0.3041 ± 0.4552 0.3395 ± 0.4910</td><td> $0 . 8 1 6 9 \pm 0 . 0 8 9 4$   $0 . 7 8 6 9 \pm 0 . 1 9 1 7$ </td></tr><tr><td>32/32</td><td>35</td><td>2</td><td>0</td><td>0.7890 ± 0.2027</td><td>61.1797 ± 90.3439</td><td>0.1118 ± 0.0291</td><td>0.3571 ± 0.5071</td><td> $0 . 7 9 9 9 \pm 0 . 1 4 9 4$ </td></tr><tr><td>32/32</td><td>25</td><td>0.25</td><td>0</td><td>0.8406 ± 0.1221</td><td>31.4182 ± 61.5544</td><td>0.0812 ± 0.0269</td><td>0.2897 ± 0.4651</td><td></td></tr><tr><td>32/32</td><td>25</td><td>1</td><td>0</td><td>0.8380 ± 0.0980</td><td>27.5710 ± 53.3358</td><td>0.1021 ± 0.0288</td><td>0.3384 ± 0.4709</td><td> $0 . 8 3 6 3 \pm 0 . 0 8 9 1$ </td></tr><tr><td>32/32</td><td>25</td><td>2</td><td>0</td><td>0.8084 ± 0.1190</td><td>68.1819 ± 90.7406</td><td>0.1055 ± 0.0307</td><td>0.3437 ± 0.4888</td><td> $0 . 8 3 6 8 \pm 0 . 0 7 7 6$  0.8432 ± 0.0708</td></tr><tr><td>64/64</td><td>25</td><td>0.25</td><td>0.1</td><td>0.8722 ± 0.0716</td><td>16.8431 ± 38.9849</td><td>0.0745 ± 0.0249</td><td></td><td></td></tr><tr><td>64/64</td><td>25</td><td>0.25</td><td>1</td><td>0.8263 ± 0.1614</td><td>33.2027 ± 61.3941</td><td>0.0710 ± 0.0213 0.2144 ± 0.3506 0.8672 ± 0.0656</td><td>0.2592 ± 0.3837</td><td> $0 . 8 6 1 0 \pm 0 . 0 7 0 2$ </td></tr><tr><td>64/64</td><td>25</td><td>2</td><td>0</td><td>0.8632 ± 0.0628</td><td>28.3243 ± 60.4912</td><td>0.0764 ± 0.0261</td><td>0.2610 ± 0.4015</td><td>0.8748 ± 0.0453</td></tr><tr><td>64/64</td><td>35</td><td>0.25</td><td>0.25</td><td>0.8467 ± 0.1023</td><td>17.2244 ± 39.2133</td><td>0.0772 ± 0.0208</td><td>0.2460 ± 0.3544</td><td>0.8189 ± 0.1472</td></tr><tr><td>64/64</td><td>35</td><td>0.25</td><td>0.1</td><td>0.8104 ± 0.1986</td><td>19.3245 ± 39.7797</td><td>0.0797 ± 0.0238</td><td>0.2667 ± 0.3749</td><td>0.8292 ± 0.0932</td></tr><tr><td>64/64</td><td>35</td><td>0.1</td><td>0.25</td><td>0.8128 ± 0.2031</td><td>59.9237 ± 145.0143</td><td>0.0733 ± 0.0219</td><td>0.2276 ± 0.3436</td><td>0.7808 ± 0.1656</td></tr><tr><td>64/64</td><td>35</td><td>0.1</td><td>0.1</td><td>0.7949 ± 0.1722</td><td>19.9224 ± 39.2512</td><td>0.0761 ± 0.0198</td><td>0.2267 ± 0.3520</td><td>0.8434 ± 0.0618</td></tr><tr><td>64/64</td><td>40</td><td>0.25</td><td>0.25</td><td></td><td>0.7621 ± 0.2831 54.3626 ± 139.6625 0.0853 ± 0.0215</td><td></td><td>0.2611 ± 0.3673</td><td> $0 . 8 2 0 1 \pm 0 . 1 3 6 1$ </td></tr></table>

## 4 Results

We perform an exploratory hyperparameter analysis on axial slices, as the MAMA SYNTH validation and test sets are reported to contain axial acquisitions. This analysis is conducted on a small development subset of 10 instances sampled from the available training data. Performance is evaluated using the Sørensen–Dice coeficient (DSC), 95th-percentile Hausdorf distance (HD95), tumor structural similarity index (tumor SSIM), mean squared error (MSE), and learned perceptual image patch similarity (LPIPS). The ablation study in Table 1 suggests that moderate regional supervision provides the most favorable trade-of between global image fidelity and tumor-focused performance in this development setting. In particular, $\lambda _ { \mathrm { t u m o r } } = 0 . 2 5$ and $\lambda _ { \mathrm { s t a b l e } } = 0 . 1$ yield the strongest segmentationoriented results, achieving the highest DSC and lowest HD95. Increasing the stable-foreground weight while keeping $\lambda _ { \mathrm { t u m o r } } = 0 . 2 5$ improves image-level metrics such as MSE and LPIPS, but degrades segmentation performance. Conversely, using a stronger tumor-region weight improves tumor ROI similarity in terms of tumor SSIM, but does not provide the best overall balance across metrics. Since the task requires preservation of localized enhancement patterns relevant to downstream tumor assessment, we prioritize DSC and HD95 over small gains in global image similarity. We therefore select the configuration with LoRA rank $/ \alpha = 6 4 / 6 4$ , MHA ∗ max = 25, λ ∗ tumor = 0.25, and $\lambda _ { \mathrm { s t a b l e } } = 0 . 1$ as the final setup. Qualitative results presented in Fig. 3 are broadly consistent with the quantitative trends. The selected model preserves the overall breast anatomy and produces spatially aligned enhancement patterns in most shown examples, while residual errors are most apparent in regions with strong or heterogeneous contrast uptake. To provide a fairer qualitative assessment, Fig. 3 also includes a case with visibly higher reconstruction error, illustrating that the model can still struggle with challenging enhancement patterns. Error maps indicate that discrepancies are concentrated primarily around enhancing tissue and tumor-adjacent regions rather than in the background. Overall, these exploratory results indicate that conditional latent rectified-flow modeling is a promising formulation for pre-to-post contrast breast MRI synthesis, particularly when combined with tumor-aware and anatomy-preserving supervision.

![](images/064a874ebffedfe67ca57ea8dc19560e6c86c55582e389e526676cf1a57a9461.jpg)  
Fig. 3. Qualitative results of the selected model (LoRA rank $/ \alpha = 6 4 / 6 4$ $\mathrm { M H A } _ { \mathrm { m a x } } =$ 25, $\lambda _ { \mathrm { t u m o r } } = 0 . 2 5 , \lambda _ { \mathrm { s t a b l e } } = 0 . 1 )$ . The predictions preserve the main anatomical structure and produce spatially aligned enhancement patterns in the shown examples. Residual errors are visualized with independently scaled error maps for each case.

## 5 Limitations

This study is limited by the exploratory ablation on only 10 development cases, so the selected hyperparameters may not generalize to the hidden test set or external cohorts. The method is trained slice-wise and does not enforce volumetric consistency, while the intensity window, regional loss weights, and plane-routing classifier were selected empirically. These design choices should therefore be validated on larger, independent datasets and across diferent acquisition protocols.

Acknowledgments. We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2026/019392. This work was partially supported by the Excellence Initiative Research University program at the AGH University of Krakow.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Black Forest Labs: FLUX.2-klein-4B. https://huggingface.co/black-forestlabs/FLUX.2-klein-4B (2026), hugging Face model card. Accessed: 2026-06-29

2. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for highresolution image synthesis. In: Forty-first international conference on machine learning (2024)

3. Garrucho, L., Kushibar, K., Reidel, C.A., Joshi, S., Osuala, R., Tsirikoglou, A., Bobowicz, M., Del Riego, J., Catanese, A., Gwoździewicz, K., et al.: A large-scale multicenter breast cancer dce-mri benchmark dataset with expert segmentations. Scientific data 12(1), 453 (2025)

4. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)

5. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022)

6. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2017)

7. Osuala, R., Joshi, S., van Dijk, J., Han, L., Cosaka, M.L., Mysler, D., Garrucho, L., Lekadir, K., Balocco, S., Diaz, O.: The MAMA-SYNTH Challenge: Synthesizing Virtual Contrast-Enhancement in Breast MRI (Apr 2026). https://doi.org/10.5281/zenodo.19852228, https://doi.org/10.5281/zenodo.19852228, accessed: 2026-06-29

8. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)