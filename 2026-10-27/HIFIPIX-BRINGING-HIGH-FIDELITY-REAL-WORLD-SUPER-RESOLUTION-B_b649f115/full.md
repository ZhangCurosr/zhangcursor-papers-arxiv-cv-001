# HIFIPIX: BRINGING HIGH-FIDELITY REAL-WORLD SUPER-RESOLUTION BACK TO PIXEL SPACE

Wenjie Li<sup>1,2,∗</sup>, Du Chen<sup>2,∗</sup>, Hui Wang<sup>2</sup>, Linlong Fan<sup>2</sup>, Youwei Lv<sup>2</sup>, Zeyi Chen<sup>2,3</sup>, Zehan Chen<sup>2,3</sup>, Jinwei Chen<sup>2</sup>, Heng Guo<sup>1,†</sup>, Zhanyu Ma<sup>1</sup>, and Qingnan Fan<sup>2,†</sup> <sup>1</sup>Beijing University of Posts and Telecommunications <sup>2</sup>vivo BlueImage Lab <sup>3</sup>Zhejiang University

![](images/a7a22eb5b906119a94a1348e60ad6f1c364981efcf19887e9d748ebefafa1f22.jpg)  
Figure 1: We propose HiFiPix, a Real-ISR framework built on a pixel-space large generative model, achieving high-fidelity restoration of real-world LQ images and superior perceptual quality compared with existing latent-space Real-ISR methods.

## ABSTRACT

Diffusion-based real-world image super-resolution suffers from limited reconstruction fidelity due to information loss in VAE-based latent representations. Recently, pixel-space diffusion models offer a promising high-faithfulness alternative by avoiding latent compression, but adapting them to Real-ISR remains challenging due to inherent patch boundary artifacts and degraded generative capability. To address the issues and maximize the potential of pretrained pixel-space diffusion models, we propose HiFiPix, a pixel-space Real-ISR framework capable of ultra high-fidelity reconstruction and fine-grained visual generation effects. To suppress patch boundary artifacts, we penalize unrealistic high-frequency periodic components with a Cross-Scale Wavelet Discriminator (CWSD) for image-level supervision, and Mixup Contrastive Self-Distillation (MixCSD) for intermediate representation regularization. Furthermore, we uncover a heterogeneous generationfidelity trade-off across low-rank adapters (LoRA) and propose Selective LoRA Pruning (LoRA-SP) to adaptively prune LoRA layers, enhancing generative capability with minimal deterioration of reconstruction fidelity. Extensive experiments demonstrate that HiFiPix consistently outperforms existing state-of-the-art latent-space methods in terms of both restoration accuracy and perceptual quality by a large margin. Code and models will be released.

## 1 INTRODUCTION

Real-world image super-resolution (Real-ISR) targets the recovery of low-quality (LQ) images suffering from complex unknown degradations, with the goal of reconstructing high-quality (HQ) counterparts that exhibit both high fidelity and satisfying perceptual quality. Initially, GAN-based algorithms Wang et al. (2021); Chen et al. (2023) employ a discriminator to match the distribution of reconstructed images with that of real-scenario images. However, lacking powerful generative priors, GAN-based models exhibit limited capability for recovering fine-grained details.

![](images/99111fa9381b86dd0647a89503d98ebb84db4f7dd5c53187f87be0deb69d6dc8.jpg)

![](images/5600efa8d8875636214de7e43eb0bebebe5405c6fe8d75c2652cc92ea7a4b8ad.jpg)

![](images/b3e7c982a33a3702b6d292606559768c71d7ea5ae2b8e5dc47a1b41bc7464620.jpg)  
Figure 2: Fine-tuning pixel models faces two challenges compared with latent models: (a) without VAE decoders to preserve spatial continuity, pixel-space models are prone to patch boundary artifacts; (b) training progress tends to favor high-fidelity restoration at the expense of generative capability.

Over the past few years, diffusion-based generative models Rombach et al. (2022); Labs (2025) have advanced Real-ISR by producing highly photo-realistic details. However, existing Real-ISR methods Dong et al. (2025); Fang et al. (2026) typically rely on the Variational Autoencoder (VAE) Esser et al. (2021) to compress LQ inputs into the latent space for efficient denoising. This irreversible compression discards the subtle textures from LQ inputs, often leading to structural inconsistencies, ghosting artifacts, and ultimately compromised restoration fidelity. To mitigate this issue, prior works try to improve input quality Lin et al. (2024), reduce the VAE compression ratio Yi et al. (2025), or inject LQ features into denoising networks Yang et al. (2024). However, these strategies fail to fundamentally address the information loss introduced by latent space compression, yielding reduced generative capability $( e . g .$ , blurry leaves in the top row of Fig. 1) or limited fidelity gains $( e . g .$ distorted characters in the bottom row of Fig. 1). As a result, restoring highly faithful textures while preserving excellent visual quality still remains a significant challenge in Real-ISR.

More recently, pixel-level text-to-image (T2I) diffusion models Yu et al. (2026); Cai et al. (2026) have demonstrated effectiveness in mitigating reconstruction accuracy issues by removing the VAE component. However, roughly fine-tuning pixel-space models to Real-ISR via LoRA adapters still faces two key challenges. First, pixel-space models typically adopt patch-based tokenization, while different tokens only interact within pure MLP-based attention blocks. Unlike VAE-based ones Saharia et al. (2022); Esser et al. (2024), whose convolutional decoders naturally preserve spatial continuity, pixel-space decoding does not explicitly enforce information interaction across patch boundaries. This leads to discontinuities between patches and periodic grid artifacts, particularly in texture-rich regions. As shown in Fig. 2 (a), under the same training strategy, pixel-space models suffer from more severe periodic artifacts than latent models Esser et al. (2024); Labs (2025), degrading the structural consistency across neighboring patches. Second, as depicted in Fig. 2 (b), although removing the VAE eliminates latent compression, it discards the compact semantic representations, making high-level semantic modeling more challenging in pure pixel space. Therefore, this weakens the generative prior and biases models toward deterministic reconstruction rather than plausible detail synthesis. Consequently, models tend to overfit reasonable but over-smoothed structures, while losing their ability to generate missing high-frequency details, especially under severe degradations.

To address these challenges, we propose HiFiPix, a pixel-space framework for high-fidelity Real-ISR. We first focus on patch-boundary artifacts arising from pixel-space decoding and observe that they mainly manifest as discontinuities in high-frequency structures. This motivates us to exploit coherent high-frequency information for cross-patch continuity. Specifically, we introduce a Cross-Scale Wavelet Discriminator (CSWD) that propagates high-frequency features across scales for boundaryaware image-level supervision, and further propose a Mixup Contrastive Self-Distillation (MixCSD) that leverages an HQ-mixed branch to regularize intermediate representations at feature levels. Together, they provide complementary external and internal supervision to suppress patch-boundary artifacts. Furthermore, we address degradation of generative capability during training. We observe that LoRA updates across layers are not equally necessary: some deep layers merely echo transformations already made by shallower layers, contributing to overfitting pixel-level characteristics rather than preserving high-level semantics. We thus propose Selective LoRA Pruning (LoRA-SP), which detects and prunes such redundant layers via Centered Kernel Alignment-based representational Kornblith et al. (2019), protects fidelity-critical layers from pruning, and progressively freezes rest, thereby constraining fine-tuning capacity precisely where overfitting risks occur.

Furthermore, existing Real-ISR benchmarks are hindered by low-quality ground truth, suffering from missing fine textures Cai et al. (2019); Wei et al. (2020) or background de-focus Wu et al. (2026). As noted in A-FINE Chen et al. (2025a), generative restoration models may yield images visually superior to these flawed GT references, rendering nearly all reference-based fidelity metrics unreliable. To address this issue, we construct Dual-Q-Bench, an ultra-high-quality paired real-world ISR benchmark captured via screen photography, whose HQ reference images feature rich textures and minimal background de-focus. Dual-Q-Bench enables reliable evaluation of both fidelity and perceptual quality. Our contributions are summarized as follows:

• We propose HiFiPix, a Real-ISR method built on a pretrained pixel-level model, shifting Real-ISR from latent-space to pixel-space modeling, achieving high-fidelity restoration without sacrificing generative capability or inference efficiency.

• We propose CSWD that propagates high-frequency features across scales for boundaryaware image-level supervision, and MixCSD, which distills clean representations to enhance patch-level discriminability, jointly mitigating patch-boundary artifacts.

• We propose LoRA-SP, which identifies functionally redundant LoRA layers via representational drift and progressively prunes them while protecting fidelity-critical layers, preserving the pretrained generative prior during supervised fine-tuning.

• We introduce Dual-Q-Bench, a high-quality paired Real-ISR benchmark that addresses the low-quality GT and background de-focus in existing benchmarks, enabling more reliable evaluation of both generative capability and reconstruction fidelity.

## 2 RELATED WORKS

## 2.1 REAL-WORLD IMAGE SUPER-RESOLUTION

With the advent of large-scale T2I models, diffusion-based Real-ISR methods have surpassed GANbased approaches. Existing methods utilize LQ input to control diffusion models for faithful restoration by leveraging pluggable adapters Chen et al. (2025b), dual-stream architectures Yang et al. (2024), ControlNet Yu et al. (2024), initialized restoration modules Lin et al. (2024), semantic condi tioning Wu et al. (2026), and large-scale high-quality training datasets Yu et al. (2024). Furthermore, to fully use input information, methods like OSEDiff Wu et al. (2024b) and TSD-SR Dong et al. (2025) replace random noise with LQ images, while S3Diff Zhang et al. (2024) and ODTSR Fang et al. (2026) incorporate GAN loss to enhance perceptual quality. TVT Yi et al. (2025) increases the VAE latent resolution to improve fidelity, whereas LucidFlux Fei et al. (2026) and StrSR Wang et al. (2026a) improve fidelity by fine-tuning FLUX Labs (2025) or Z-image Cai et al. (2025) with in creased VAE channel capacity. Nevertheless, these methods still operate in latent spaces, where VAE compression inevitably limits reconstruction fidelity. In contrast, we perform Real-ISR directly in pre-trained pixel spaces Cai et al. (2026), enabling high-fidelity and perceptually pleasing Real-ISR.

## 2.2 PIXEL-SPACE DIFFUSION MODELS

Early pixel-space diffusion models Dhariwal & Nichol (2021); Saharia et al. (2022) suffered from prohibitive computational costs at high resolutions. Recent works Chen et al. (2025c); Wang et al. (2026b); Li & He (2026); Ma et al. (2026b); Lu et al. (2026b); Ma et al. (2026a); Yu et al. (2026) have revisited pixel-space diffusion by improving its training efficiency, sampling strategy, and architectural design. Among them, PixelFlow Chen et al. (2025c) enables efficient end-to-end pixel-space generation, while JiT Li & He (2026) and PixelDiT Yu et al. (2026) further improve the scalability and efficiency of pixel-space modeling. HiDream-O1-Image Cai et al. (2026) extends this paradigm to a large-scale pixel-space T2I diffusion transformer with a unified generative architecture. Inspired by these advances, we explore fine-tuning pre-trained pixel-space T2I models for Real-ISR, aiming to achieve high-fidelity restoration and visually pleasant generation without VAE compression.

(a)  
![](images/013bf2deb7cf17e7aab57367bab827a7e1eb9d22a261366f4db6fa95bcf7d0c0.jpg)

![](images/ff7d7d5089337370ea8f91a8c99e57024a56ed5b1ac109e0549066d75aec1672.jpg)  
Figure 3: (a) Heavier boundary artifacts when super-resolving extremely blurry input. (b) Removal of individual LoRA layers reveals trade-offs between reconstruction and generative capabilities.

## 3 METHOD

## 3.1 EMPIRICAL OBSERVATIONS

Following classical one-step fine-tuning paradigms Yi et al. (2025), we identify two critical insights when applying HiDream-O1-Image for Real-ISR. First, patch-boundary artifacts may not arise solely from the patch-wise architecture of pixel-space models. As shown in Fig. 3 (a), grid-like artifacts become more pronounced under severe degradation than under mild degradation, suggesting that imperfect high-frequency reconstruction is an important cause of boundary artifacts, as severe degradation makes high-frequency reconstruction more challenging. This observation motivates a low-cost and architecture-agnostic strategy to explicitly regularize high-frequency components. Specifically, we design the Cross-Scale Wavelet Discriminator, which penalizes spurious boundary artifacts via multi-scale information constraints. Besides, we propose Mixup Contrastive Self-Distillation to ensure faithful generation of genuine high-frequency details.

Second, simply applying LoRA layers yields SR outputs lacking fine-grained details, indicating that such a general strategy drastically reduces the generative capability of pixel-space models. As shown in Fig. 3 (b), we perform ablation studies to investigate individual LoRA layers, then reveals that generative capacity and fidelity are non-uniformly entangled across depth: removing certain layers boosts generative expressiveness at negligible fidelity cost, whereas removing others yields a similar generative gain but at the expense of severe fidelity degradation. This heterogeneity rules out uniform layer pruning or fine-tuning as an effective strategy; therefore, we propose Selective LoRA Pruning that identifies and removes layers that mainly reduce the generation ability.

## 3.2 CROSS-SCALE WAVELET DISCRIMINATOR

Inspired by Wu et al. (2025); Wang et al. (2026a) that leverage DINOv3-ConvNeXt Siméoni et al. (2025) as a discriminator to enhance the visual quality, we propose Cross-Scale Wavelet Discriminator (CSWD) to extract multi-scale features and penalize the unrealistic high-frequency periodic components from boundaries, thereby alleviating patch boundary artifacts. As shown in Fig. 4, given the super-resolved image $I _ { \mathrm { S R } }$ , the frozen DINOv3-ConvNeXt first extracts multi-level features $F _ { 1 } \in \mathbb { R } ^ { \frac { H } { 4 } \times \frac { 1 } { 4 } \times C } , F _ { 2 } \in \mathbb { R } ^ { \frac { H } { 8 } \times \frac { W } { 8 } \times 2 C }$ , and $F _ { \mathrm { 3 } } \in \mathbb { R } ^ { \frac { H } { 1 6 } \times \frac { W } { 1 6 } \times 4 C }$ . Different from Wu et al. (2025) that relies on pooling or convolution layers for multi-scale down-sampling, we employ the wavelet transform (WT) to decompose features into a low-frequency sub-band LL and three high-frequency sub-bands {LH, HL, HH}, all with half the spatial resolution of input features. We discard LL and concatenate the three high-frequency sub-bands along the channel dimension, thereby focusing the discriminator on edges, textures, and periodic artifacts. The concatenated features are then compressed by trainable convolutional layers, while cascaded WT-convolution layers progressively extract deeper high-frequency representations and produce a pixel-wise discriminative score map.

To further exploit cross-scale information, we introduce a cross-scale high-frequency injection mechanism. Specifically, high-frequency sub-bands $\left\{ L H _ { i } ^ { k } , H L _ { i } ^ { k } , H H _ { i } ^ { k } \right\}$ extracted by the k-th WT block $\bar { W T _ { k } }$ from level $\bar { F _ { i } }$ are aggregated via a $1 \times 1$ convolution, then injected into features of next level $F _ { i + 1 }$ . This progressively propagates fine-grained high-frequency details from shallow to deep layers, enhancing the discriminator’s ability to detect multi-scale periodic boundary artifacts.

![](images/78fa663e08b6621abc24a96011faa63a1dd4168419435fc0456f1ce187c05adf.jpg)  
Figure 4: Architecture of HiFiPix. We introduce MixCSD and CSWD to alleviate patch boundary artifacts, and propose LoRA-SP to enhance the model’s generative capability.

$$
\begin{array} { c } { { W T _ { k } \left( C o n v \left( F _ { i } \right) \right) = \left\{ L L _ { i } ^ { k } , L H _ { i } ^ { k } , H L _ { i } ^ { k } , H H _ { i } ^ { k } \right\} , H _ { i } ^ { k } = C o n v \left( L H _ { i } ^ { k } + H L _ { i } ^ { k } + H H _ { i } ^ { k } \right) } } \\ { { { \ } } } \\ { { W T _ { k - 1 } \left( C o n v \left( C o n c a t \left( H _ { i } ^ { k } , F _ { i + 1 } \right) \right) \right) = \left\{ L L _ { i + 1 } ^ { k } , L H _ { i + 1 } ^ { k } , H L _ { i + 1 } ^ { k } , H H _ { i + 1 } ^ { k } \right\} . } } \end{array}\tag{1}
$$

(2)

The multi-level discriminator progressively produces pixel-wise logit maps $M \in \mathsf { \{ R }  ^ { \frac { H } { 4 } \times \frac { W } { 4 } \times 1 }$ 2 $\mathbb { R } ^ { \frac { H } { 8 } \times \frac { W } { 8 } \times 1 } , \mathbb { R } ^ { \frac { H } { 1 6 } \times \frac { W } { 1 6 } \times 1 } \}$ for PatchGAN-style Isola et al. (2017) local realism discrimination.

## 3.3 MIXUP CONTRASTIVE SELF-DISTILLATION

To further suppress boundary artifacts while promoting the faithful generation of clean details, as shown in Fig. 4, we introduce Mixup Contrastive Self-Distillation (MixCSD) at intermediate layers of DiT blocks. Specifically, for each training batch, we construct two forward branches sharing the same set of LoRA weights $f _ { \theta }$ and the same DiT weights. The anchor branch takes the pure LQ image $I _ { L Q }$ as input, with gradients flowing normally, and extracts patch-token hidden states at DiT layer ℓ:

$$
\pmb { h } _ { l q } ^ { \ell } = f _ { \theta } ^ { ( \ell ) } ( \pmb { I } _ { L Q } ) \in \mathbb { R } ^ { B \times N \times D } ,\tag{3}
$$

where B, N, and D denote the batch size, the number of patch tokens, and the hidden dimension, respectively. Here we omit the DiT weights in all equations for simplicity of expression. The positive branch takes a mixture of GT and LQ as input with mixing ratio $\alpha \in [ 0 , 1 ]$ ], then feeds forward through sharing the anchor branch weights under a stop-gradient operator sg:

$$
\begin{array} { r } { \pmb { h } _ { m i x } ^ { \ell } = \mathrm { s g } \left( f _ { \theta } ^ { ( \ell ) } ( \alpha \pmb { I } _ { G T } + ( 1 - \alpha ) \pmb { I } _ { L Q } ) \right) , } \end{array}\tag{4}
$$

since the mixed input injects genuine high-frequency content and consistent cross-patch boundary structure, $h _ { m i x } ^ { \ell }$ carries more coherent neighborhood features at patch boundaries than $h _ { l q } ^ { \ell } .$ , serving as an implicit target for the anchor branch along the "boundary-consistency" dimension. After ℓ -normalization of the anchor $\begin{array} { r } { ( e . g . , \bar { h } _ { l q , i } = \frac { { h } _ { l q , i } } { \| { h } _ { l q , i } \| _ { 2 } } ) } \end{array}$ , we treat each patch-token as a contrastive unit (flattened index $i = 1 , . . . , B N )$ and define an InfoNCE loss Oord et al. (2018) with temperature τ:

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } ^ { ( \ell ) } = - \frac { 1 } { B N } \sum _ { i = 1 } ^ { B N } \log \frac { \exp \left( \bar { h } _ { l q , i } ^ { \top } \bar { h } _ { m i x , i } / \tau \right) } { \displaystyle \sum _ { j = 1 } ^ { B N } \exp \left( \bar { h } _ { l q , i } ^ { \top } \bar { h } _ { m i x , j } / \tau \right) } ,\tag{5}
$$

where $j = i$ term in the denominator corresponds to the positive pair and the remaining BN −1 terms serve as negatives, and ⊤ denotes the matrix transpose. Negatives are sampled from other patches in the same image or other images in the same training batch. This encourages each anchor to approach its high-frequency-consistent target while moving away from other patches, particularly spatially adjacent ones, thereby suppressing patch boundary artifacts. A stop-gradient operation enforces one-way supervision from the positive branch to the anchor branch, preventing representational collapse. Multi-layer aggregation and step-gated weighting follow the same formulation as follows:

$$
{ \mathcal { L } } _ { \mathrm { I n f o N C E } } = \lambda _ { \mathrm { I n f o N C E } } ( t ) \cdot { \frac { 1 } { | S | } } \sum _ { \ell \in S } { \mathcal { L } } _ { \mathrm { I n f o N C E } } ^ { ( \ell ) } , \lambda _ { \mathrm { I n f o N C E } } ( t ) = { \left\{ \begin{array} { l l } { \lambda _ { \mathrm { I n f o N C E } } ^ { \operatorname* { m a x } } , } & { t \geq t _ { \mathrm { s t a r t } } } \\ { 0 , } & { t < t _ { \mathrm { s t a r t } } } \end{array} \right. } ,\tag{6}
$$

where $s$ is the number of supervised DiT layers, t is the training step. $\lambda _ { \mathrm { I n f o N C E } } ^ { \mathrm { m a x } }$ and $t _ { \mathrm { s t a r t } }$ denote the target loss weight and the step at which it is activated, respectively.

## 3.4 SELECTIVE LORA PRUNING

As shown in Fig. 4, we introduce Selective LoRA Pruning (LoRA-SP), a training-time mechanism that leverages the network’s internal representations to identify and freeze functionally redundant LoRA branches, thereby improving generative capacity without sacrificing restoration fidelity.

At each probing step, we forward the degraded input $I _ { L Q }$ and mixture $\pmb { x } _ { m i x } = \alpha \pmb { I } _ { G T } + ( 1 - \alpha ) \pmb { I } _ { L Q }$ through the network to gather per-layer hidden states $\dot { h } ^ { ( l ) }$ . Layer-wise similarity is quantified by linear Centered Kernel Alignment (CKA) Kornblith et al. (2019) and defined as:

$$
\mathrm { C K A } ( H _ { 1 } , H _ { 2 } ) = \frac { \Vert H _ { 1 } ^ { \top } H _ { 2 } \Vert _ { F } ^ { 2 } } { \Vert H _ { 1 } ^ { \top } H _ { 1 } \Vert _ { F } \Vert H _ { 2 } ^ { \top } H _ { 2 } \Vert _ { F } } ,\tag{7}
$$

where $H _ { 1 } , H _ { 2 }$ and $F$ denote two intermediate features and Frobenius norm, respectively. We derive two EMA-smoothed per-layer metrics: anchor similarity $\bar { \mathbf { a } } ^ { ( l ) }$ via $\mathrm { C K A } ( h _ { l q } ^ { \overline { { ( l ) } } } , h _ { l q } ^ { ( r ) } )$ with shallow reference layer r (expected to decay under generative processing), and fidelity gap $\bar { \pmb g } ^ { ( l ) } =$ $1 - \mathrm { C K A } ( h _ { l q } ^ { ( l ) } , h _ { m i x } ^ { ( l ) } )$ , measuring layer-l’s representation shift toward high-quality targets. A layer becomes a redundancy candidate when its anchor similarity fails to drop from the prior layer,

$$
\mathcal { C } = \left\{ l \Big | \bar { \pmb { a } } ^ { ( l ) } - \bar { \pmb { a } } ^ { ( l - 1 ) } \geq 0 \right\} ,\tag{8}
$$

where layers in $\mathcal { C }$ introduce no distinct additional representation transformation. Candidates are ranked and filtered against a small protected set $\mathcal { P }$ of the globally most fidelity-critical layers, which are vetoed from pruning. The surviving top-ranked candidates constitute the final pruning set,

$$
S = \left\{ l \in \mathcal { C } \setminus \mathcal { P } \right\} , \qquad | S | \leq K ,\tag{9}
$$

bounded by a fixed pruning budget K. Selected layers are not pruned abruptly: early in training they undergo a temporary windowed freeze, later transitioning to a permanent freeze once the redundancy signal stabilizes, with a soft-decay fallback when the pruning budget is saturated. This graduated schedule keeps LoRA-SP conservative at early training stages while achieving persistent capacity reduction over time, enabling improved generation without severe fidelity degradation.

## 3.5 TRAINING OBJECTIVE

Following Fang et al. (2026), one-step LoRA fine-tuning paradigm is adopted. As shown in Fig. 4, we replace the original Gaussian noise with LQ image as the input. The overall backbone generates fine-grained details in pure pixel-space by predicting the residual between LQ and GT image.

We fix pretrained DiT blocks, optimize LoRA with a reconstruction loss and an adversarial loss, yielding the total objective $\mathcal { L } = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { a d v } } . \mathcal { L } _ { \mathrm { r e c } }$ aggregates four terms: pixel-space regressive loss $\mathcal { L } _ { 1 } ,$ , a DINOv3-ConvNeXt-based Siméoni et al. (2025) perceptual loss ${ \bar { \boldsymbol { \mathcal { L } } } } _ { p e r } ,$ frequency-space loss Wang et al. (2026a) $\mathcal { L } _ { f r e q } .$ , and the InfoNCE contrastive loss $\mathcal { L } _ { \mathrm { I n f o N C E } } \left( \mathrm { S e c . } 3 . 3 \right)$ .

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \lambda _ { 1 } \mathcal { L } _ { 1 } + \lambda _ { 2 } \mathcal { L } _ { p e r } + \lambda _ { 3 } \mathcal { L } _ { f r e q } + \lambda _ { 4 } \mathcal { L } _ { I n f o N C E } . } \end{array}\tag{10}
$$

For $\mathcal { L } _ { \mathrm { a d v } }$ , we employ two complementary discriminators built on the same DINOv3-ConvNeXt features: a main discriminator $D _ { \mathrm { m a i n } }$ that performs multi-scale assessment of structural realism, guiding the model to produce images with higher visual authenticity, and our Cross-Scale Wavelet Discriminator $D _ { \mathrm { w a v e } }$ (Sec. 3.2), which applies a wavelet transform to retain accurate high-frequency sub-bands, while penalizing texture/edge artifacts, especially boundary artifacts across patches:

$$
\mathcal { L } _ { \mathrm { a d v } } = \lambda _ { \mathrm { m a i n } } \mathbb { E } _ { \hat { x } } \big [ - \log \sigma ( D _ { \mathrm { m a i n } } ( \hat { x } ) ) \big ] + \lambda _ { \mathrm { w a v e } } ( t ) \mathbb { E } _ { \hat { x } } \big [ - \log \sigma ( D _ { \mathrm { w a v e } } ( \hat { x } ) ) \big ] ,\tag{11}
$$

where $\lambda _ { \mathrm { w a v e } } ( t )$ is linearly warmed up to avoid the high-frequency term saturating at an early stage.

Table 1: Quantitative results on real-world benchmarks, with the best/second-best results highlighted in bold/underlined. "-s" denotes the number of diffusion steps.
<table><tr><td rowspan=1 colspan=1>Datasets</td><td rowspan=1 colspan=1>Metrics</td><td rowspan=1 colspan=2>SUPIRFaithDiffDit4SRVOSRLucidFlux-s50   -s20   -s40  -s25   -s15</td><td rowspan=1 colspan=4>TSDSRPiSA-SR TVT ODTSRStrSR-s1    -s1    -s1   -s1   -s1</td><td rowspan=1 colspan=1>Ours-s1</td></tr><tr><td rowspan=10 colspan=1>DrealSR</td><td rowspan=3 colspan=1>PSNR↑SSIM↑LPIPS↓</td><td rowspan=1 colspan=2>26.87  27.29  25.79 27.72  20.82</td><td rowspan=1 colspan=4>27.71  28.31  28.25 28.15  23.90</td><td rowspan=1 colspan=1>29.62</td></tr><tr><td rowspan=1 colspan=2>0.6702 0.7104 0.68170.7376 0.5806</td><td rowspan=1 colspan=4>0.7543 0.7804 0.7898 0.7740 0.6948</td><td rowspan=1 colspan=1>0.7990</td></tr><tr><td rowspan=1 colspan=2>0.4339 0.3497 0.36560.3664 0.4532</td><td rowspan=1 colspan=4>0.2997  0.2960 0.2900 0.2591 0.3345</td><td rowspan=1 colspan=1>0.2898</td></tr><tr><td rowspan=3 colspan=1>DISTS↓MANIQA↑MUSIQ↑</td><td rowspan=1 colspan=2>0.2793 0.2386 0.24370.2446 0.2771</td><td rowspan=1 colspan=4>0.2153 0.2169 0.2206 0.1926 0.2377</td><td rowspan=1 colspan=1>0.2302</td></tr><tr><td rowspan=1 colspan=2>0.5281 0.6378 0.62320.5204  0.251</td><td rowspan=1 colspan=4>0.5883 0.6161 0.5784 0.6226 0.5598</td><td rowspan=1 colspan=1>0.6429</td></tr><tr><td rowspan=1 colspan=2>53.91  65.79  64.51 57.70  67.97</td><td rowspan=1 colspan=4>66.83  66.11  65.60 62.83 60.98</td><td rowspan=1 colspan=1>68.18</td></tr><tr><td rowspan=3 colspan=1>Q-Align↑LIQE↑Q-Insight↑</td><td rowspan=1 colspan=2>3.2002 3.4284 3.34822.9609 3.6537</td><td rowspan=1 colspan=4>3.6185 3.58363.64243.46523.3464</td><td rowspan=1 colspan=1>3.7473</td></tr><tr><td rowspan=1 colspan=2>3.8725 3.9620 3.91572.7986 4.0107</td><td rowspan=1 colspan=4>3.9491  4.1496 4.0861 3.57173.4685</td><td rowspan=1 colspan=1>4.1957</td></tr><tr><td rowspan=1 colspan=1>3.1085 3.5806 3.5530</td><td rowspan=1 colspan=1>3.2112 3.6777</td><td rowspan=1 colspan=4>3.4693 3.5460 3.5133 3.4223 3.3051</td><td rowspan=1 colspan=1>3.6419</td></tr><tr><td rowspan=1 colspan=1>AFINE-ALL↓</td><td rowspan=1 colspan=1>46.27  26.28  15.41</td><td rowspan=1 colspan=1>50.63  14.45</td><td rowspan=1 colspan=4>15.86  18.57  22.97 20.17 24.58</td><td rowspan=1 colspan=1>7.61</td></tr><tr><td rowspan=10 colspan=1>GGT-GeneralMix</td><td rowspan=2 colspan=1>PSNR↑SSIM↑</td><td rowspan=1 colspan=1>26.03  26.57  24.40</td><td rowspan=1 colspan=1>25.52  21.83</td><td rowspan=1 colspan=2>25.16  26.13  26.41</td><td rowspan=1 colspan=2>25.95 24.17</td><td rowspan=1 colspan=1>26.86</td></tr><tr><td rowspan=1 colspan=1>0.6606 0.6984 0.6518</td><td rowspan=1 colspan=1>0.6512 0.5823</td><td rowspan=1 colspan=2>0.6813 0.7161 0.7362</td><td rowspan=1 colspan=2>0.71260.6572</td><td rowspan=1 colspan=1>0.7435</td></tr><tr><td rowspan=2 colspan=1>LPIPS↓DISTS↓</td><td rowspan=1 colspan=1>0.3637 0.2800 0.3527</td><td rowspan=1 colspan=1>0.4418 0.4275</td><td rowspan=1 colspan=2>0.3114 0.2916 0.2732</td><td rowspan=1 colspan=2>0.30400.3382</td><td rowspan=1 colspan=1>0.2684</td></tr><tr><td rowspan=1 colspan=1>0.1945 0.1614 0.1864</td><td rowspan=1 colspan=1>0.2750 0.2159</td><td rowspan=1 colspan=2>0.1701 0.1756 0.1631</td><td rowspan=1 colspan=2>0.16930.1846</td><td rowspan=1 colspan=1>0.1605</td></tr><tr><td rowspan=1 colspan=1>MANIQA↑</td><td rowspan=1 colspan=1>0.5677 0.6041 0.6068</td><td rowspan=1 colspan=1>0.4708 0.6087</td><td rowspan=1 colspan=2>0.5880 0.5934 0.5698</td><td rowspan=1 colspan=2>0.61950.5597</td><td rowspan=1 colspan=1>0.6221</td></tr><tr><td rowspan=1 colspan=1>MUSIQ↑</td><td rowspan=1 colspan=1>59.36 63.13  68.49</td><td rowspan=1 colspan=1>49.79  70.83</td><td rowspan=1 colspan=2>69.27  65.60  67.79</td><td rowspan=1 colspan=2>66.37 62.18</td><td rowspan=1 colspan=1>71.06</td></tr><tr><td rowspan=3 colspan=1>Q-Align↑LIQE↑Q-Insight↑</td><td rowspan=1 colspan=1>3.7762 3.8415 3.9934</td><td rowspan=1 colspan=1>2.8977  4.2334</td><td rowspan=1 colspan=2>3.8949 3.8818 3.9229</td><td rowspan=1 colspan=1>3.9480</td><td rowspan=1 colspan=1>3.9615</td><td rowspan=1 colspan=1>4.3009</td></tr><tr><td rowspan=1 colspan=1>3.1497 3.3761 3.9904</td><td rowspan=1 colspan=1>1.9933  3.8951</td><td rowspan=1 colspan=2>4.0377 3.7018 3.9262</td><td rowspan=1 colspan=1>3.6083</td><td rowspan=1 colspan=1>3.4264</td><td rowspan=1 colspan=1>4.2130</td></tr><tr><td rowspan=1 colspan=1>3.4435 3.6085 3.8388</td><td rowspan=1 colspan=1>2.9985 3.9863</td><td rowspan=1 colspan=2>3.6901  3.6655 3.7361</td><td rowspan=1 colspan=1>3.7062</td><td rowspan=1 colspan=1>3.5839</td><td rowspan=1 colspan=1>3.9289</td></tr><tr><td rowspan=1 colspan=1>AFINE-ALL↓</td><td rowspan=1 colspan=1>39.66  32.82  18.46</td><td rowspan=1 colspan=1>60.49  24.57</td><td rowspan=1 colspan=4>17.30  15.16 24.84 12.40 26.32</td><td rowspan=1 colspan=1>2.6303</td></tr><tr><td rowspan=10 colspan=1>Dual-Q-Bench</td><td rowspan=2 colspan=1>PSNR↑SSIM↑</td><td rowspan=1 colspan=1>22.71  22.59  21.67</td><td rowspan=1 colspan=1>21.90  19.56</td><td rowspan=1 colspan=4>21.89  22.32  22.64 22.35 21.74</td><td rowspan=1 colspan=1>23.21</td></tr><tr><td rowspan=1 colspan=1>0.6225 0.6196 0.6152</td><td rowspan=1 colspan=1>0.5701 0.5594</td><td rowspan=1 colspan=2>0.6290 0.6380 0.6492</td><td rowspan=1 colspan=1>0.6507</td><td rowspan=1 colspan=1>0.6129</td><td rowspan=1 colspan=1>0.6662</td></tr><tr><td rowspan=2 colspan=1>LPIPS↓DISTS↓</td><td rowspan=1 colspan=1>0.3046 0.2736 0.3157</td><td rowspan=1 colspan=1>0.4095 0.3191</td><td rowspan=1 colspan=2>0.2648 0.2722 0.2746</td><td rowspan=1 colspan=1>0.2862</td><td rowspan=1 colspan=1>0.3069</td><td rowspan=1 colspan=1>0.2639</td></tr><tr><td rowspan=1 colspan=1>0.1784 0.1695 0.1793</td><td rowspan=1 colspan=1>0.2824 0.1829</td><td rowspan=1 colspan=1>0.1663</td><td rowspan=1 colspan=1>0.1675 0.1662</td><td rowspan=1 colspan=1>0.1646</td><td rowspan=1 colspan=1>0.1788</td><td rowspan=1 colspan=1>0.1590</td></tr><tr><td rowspan=1 colspan=1>MANIQA↑</td><td rowspan=1 colspan=1>0.6257 0.6590 0.6328</td><td rowspan=1 colspan=1>0.4830 0.6623</td><td rowspan=1 colspan=1>0.6307</td><td rowspan=1 colspan=1>0.6520 0.6250</td><td rowspan=1 colspan=1>0.6611</td><td rowspan=1 colspan=1>0.6061</td><td rowspan=1 colspan=1>0.6663</td></tr><tr><td rowspan=1 colspan=1>MUSIQ↑</td><td rowspan=1 colspan=1>67.80  70.80  70.87</td><td rowspan=1 colspan=1>59.12  74.05</td><td rowspan=1 colspan=1>73.83</td><td rowspan=1 colspan=1>72.17  73.07</td><td rowspan=1 colspan=1>71.19</td><td rowspan=1 colspan=1>70.25</td><td rowspan=1 colspan=1>74.22</td></tr><tr><td rowspan=2 colspan=1>Q-Align↑LIQE↑</td><td rowspan=1 colspan=1>4.2362 4.2789 4.2239</td><td rowspan=1 colspan=1>2.8755 4.5148</td><td rowspan=1 colspan=1>4.2297</td><td rowspan=1 colspan=1>4.3351 4.3144</td><td rowspan=1 colspan=1>4.3569</td><td rowspan=1 colspan=1>4.2257</td><td rowspan=1 colspan=1>4.6683</td></tr><tr><td rowspan=1 colspan=1>4.1563 4.3841 4.3776</td><td rowspan=1 colspan=1>2.3265 4.4996</td><td rowspan=1 colspan=1>4.6336</td><td rowspan=1 colspan=1>4.5401 4.6342</td><td rowspan=1 colspan=1>4.4962</td><td rowspan=1 colspan=1>4.3966</td><td rowspan=1 colspan=1>4.8425</td></tr><tr><td rowspan=1 colspan=1>Q-Insight↑</td><td rowspan=1 colspan=1>3.8389 3.9413 3.9807</td><td rowspan=1 colspan=1>3.2694 4.1875</td><td rowspan=1 colspan=1>3.9661</td><td rowspan=1 colspan=1>4.0004 4.0005</td><td rowspan=1 colspan=1>3.9700</td><td rowspan=1 colspan=1>4.0608</td><td rowspan=1 colspan=1>4.1103</td></tr><tr><td rowspan=1 colspan=1>AFINE-ALL↓</td><td rowspan=1 colspan=1>47.01  43.44  41.68</td><td rowspan=1 colspan=1>69.75  42.64</td><td rowspan=1 colspan=4>39.00  35.10  40.22 30.95 43.59</td><td rowspan=1 colspan=1>12.62</td></tr></table>

Table 2: Comparison of computational costs on GGT-GeneralMix test set.
<table><tr><td>Methods</td><td>SUPIR -s50</td><td>FaithDiff -s20</td><td>Dit4SR -s40</td><td>VOSR-1.4B -s25</td><td>LucidFlux -s15</td><td>TSDSR -s1</td><td>PiSA-SR -s1</td><td>TVT -s1</td><td>ODTSR -s1</td><td>StrSR -s1</td><td>Ours -s1</td></tr><tr><td>Inference Time (s) ↓</td><td>10.8</td><td>5.0</td><td>57.6</td><td>7.9</td><td>23.8</td><td>0.48</td><td>0.45</td><td>2.4</td><td>5.2</td><td>0.97</td><td>0.27</td></tr><tr><td>GPU Memory (G) ↓</td><td>52.46</td><td>11.10</td><td>22.06</td><td>6.91</td><td>35.0</td><td>6.99</td><td>6.89</td><td>14.62</td><td>40.83</td><td>27.69</td><td>17.63</td></tr></table>

3.6 CONSTRUCTION OF DUAL-Q-BENCH

Due to page limitations, please refer to Appendix A.1 for construction details of Dual-Q-Bench.

User Study of Real-ISR Benchmarks. We conduct a user study to evaluate the quality of Dual-Q Bench against existing Real-ISR benchmarks, including RealSR Cai et al. (2019), DRealSR Wei et al. (2020), GGT Kong et al. (2026), and ScreenSR Wu et al. (2026). We randomly sample 50 GT-LQ image pairs from each dataset for evaluation. In each trial, 17 participants are presented with five image pairs from different datasets and asked to select the best one based on GT quality (clarity, background de-focus, and realism), LQ authenticity, and GT–LQ alignment (color, brightness, and position). As shown in Tab. 6, Dual-Q-Bench achieves the highest user preference rate of 76%, substantially surpassing the second-best, GGT (10.7%), confirming the best quality.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets and Evaluation Protocols. We use LSDIR Li et al. (2023) and the first 10, 000 images from FFHQ Kornblith et al. (2019) as the training sets. LQ images are synthesized using the degradation pipeline in Real-ESRGAN Wang et al. (2021). We evaluate competing methods on DRealSR Wei et al. (2020), GGT-GeneralMix Kong et al. (2026), RealSR Cai et al. (2019) and our constructed Dual-Q-Bench. To comprehensively evaluate all methods, we employ both full-reference (FR) and no-reference (NR) metrics. The PSNR, SSIM Wang et al. (2004), LPIPS Zhang et al. (2018) and DISTS are used for fidelity evaluation, while MANIQA Yang et al. (2022), MUSIQ Ke et al. (2021), Q-Insight Li et al. (2026), Q-Align Wu et al. (2024a), and LIQE Zhang et al. (2023) are computed for visual generation assessment. In addition, we adopt AFINE-ALL Chen et al. (2025a) to assess the consistency between full-reference and no-reference evaluation even the reference image is imperfect.

Implementation Details. LoRA Hu et al. (2022) layers with rank 64 are fine-tuned based on the open-source HiDream-O1-Image Cai et al. (2026) model. $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } , \lambda _ { 4 } , \lambda _ { m a i n } , \lambda _ { w a v e }$ is set to 1, $1 , \bar { 1 } e ^ { - 1 } , 1 , 1 , 1 e ^ { - 1 }$ , respectively. The timestep t is fixed to 0.5 in training and inference stages. We train HiFiPix on 4 NVIDIA H20 GPUs for 4, 000 iterations with an accumulated batch size of 32. Following previous methods Fei et al. (2026); Wang et al. (2026a), the training patch size is set to $1 0 2 4 ^ { 2 }$ . The learning rate is set to $5 e ^ { - 5 }$ with the AdamW Loshchilov & Hutter (2019) optimizer. We discard image captions and remove the text encoder to accelerate training and inference. 4.2 EXPERIMENT RESULTS

![](images/db7cc818f3adc2220b6981e9b328dcf697556b3f292db34e3f6975154b1a30b0.jpg)

Figure 5: Qualitative comparisons with existing Real-ISR methods on real-world test sets.  
![](images/3727b530efb4e6c72be68915c4da17194417c4b47638c545444aa69f95aac5f7.jpg)  
Figure 6: Qualitative visualization of CSWD and MixCSD for patch-boundary artifact mitigation.

<table><tr><td>CSWD MixCSD</td><td>PSNR↑ LPIPS↓ MUSIQ↑</td><td></td><td></td><td>MANIQA↑</td></tr><tr><td>X</td><td>X 25.53</td><td>0.3132</td><td>70.67</td><td>0.6247</td></tr><tr><td>√</td><td>X</td><td>25.83 0.2781</td><td>68.62</td><td>0.6048</td></tr><tr><td>X</td><td>√ 25.77</td><td>0.2839</td><td>68.88</td><td>0.6084</td></tr><tr><td>√ √</td><td>26.12</td><td>0.2759</td><td>69.86</td><td>0.6183</td></tr></table>

Table 3: Ablation results of CSWD and MixCSD on quantitative metrics.

We compare HiFiPix with existing state-of-the-art Real-ISR methods, including multi-steps methods (i.e., SUPIR Yu et al. (2024), FaithDiff Chen et al. (2025b), Dit4SR Duan et al. (2025), VOSR Wu et al. (2026) in version of 1.4B Params, LucidFlux Fei et al. (2026)), and one-step methods (i.e., TSDSR Dong et al. (2025), PiSA-SR Sun et al. (2025), TVT Yi et al. (2025), ODTSR Fang et al. (2026), and StrSR Wang et al. (2026a) based on Z-Image-Turbo Cai et al. (2025)).

Quantitative Results. As shown in Tab. 1, HiFiPix consistently outperforms existing methods in restoration fidelity, with notable gains on high-quality GT benchmarks (GGT and Dual-Q-Bench). It also excels on most NR metrics, confirming enhanced fidelity without sacrificing generation ability.

Qualitative Results. As shown in Fig. 5, HiFiPix produces both photo-realistic results and faithfully restored details, e.g., fine text in the first set and faithful textures in the second and third sets. By contrast, existing VAE-based methods generate entangled artifacts and over-smoothed details.

Computational Cost. As shown in Tab. 2, we compare the computational cost of different methods on GGT-GeneralMix at 1024<sup>2</sup> resolution on a single NVIDIA H20 GPU. Benefiting from the removal of the text encoder and the efficient vanilla patch-wise tokenization in HiDream-O1-Image that greatly reduces complexity, HiFiPix is much more efficient than multi-step diffusion methods and runs 40% faster than PiSA-SR, the second-best single-step method. Moreover, its GPU memory footprint is

Table 4: Ablation on the teacher network input composition: mixed LQ-HQ vs. HQ-only.  
Table 5: Ablation on loss formulation: distillation loss $( \mathcal { L } _ { \mathrm { K D } } )$ vs. contrastive learning $( \mathcal { L } _ { \mathrm { { I n f o N C E } } } )$
<table><tr><td>Method</td><td></td><td>PSNR↑LPIPS↓</td><td></td><td>MUSIQ↑MANIQA↑</td></tr><tr><td>HQ-only</td><td>26.15</td><td>0.2726</td><td>68.72</td><td>0.6079</td></tr><tr><td>LQ+HQ (Ours)</td><td>26.12</td><td>0.2759</td><td>69.86</td><td>0.6183</td></tr></table>

<table><tr><td>Method</td><td>PSNR↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>MANIQA↑</td></tr><tr><td> $\overline { { \mathcal { L } _ { \mathrm { K D } } } }$ </td><td>26.01</td><td>0.2794</td><td>69.51</td><td>0.6155</td></tr><tr><td> $\mathcal { L } _ { \mathrm { I n f o N C E } } ( \bf { O u r s } )$ </td><td>26.12</td><td>0.2759</td><td>69.86</td><td>0.6183</td></tr></table>

Table 7: Effectiveness of redundancy detection and fidelity-aware veto in LoRA-SP. Further effectiveness of LoRA-SP in VAE-based models.

Table 6: Results of user study, showing vote percentages for different Real-ISR benchmarks.
<table><tr><td>Benchmarks</td><td>Preferences↑</td></tr><tr><td>RealSR</td><td>1.3%</td></tr><tr><td>DrealSR</td><td>1.5%</td></tr><tr><td>GGT-GeneralMix</td><td>10.7%</td></tr><tr><td>ScreenSR</td><td>10.5%</td></tr><tr><td>Dual-Q-Bench (Ours)</td><td>76.0%</td></tr></table>

<table><tr><td>Method</td><td>PSNR↑LPIPS↓</td><td></td><td></td><td>MUSIQ↑MANIQA↑</td></tr><tr><td>Hidream Baseline (Pixel-based) +Random LoRA Pruning +Redundancy detection (RD)</td><td>26.12 25.27 25.81</td><td>0.2759 0.2910 0.2846</td><td>69.86 69.52 71.17</td><td>0.6183 0.6124 0.6303</td></tr><tr><td>+RD &amp; Fidelity-aware veto (Ours) Z-Image (VAE-based) Z-Image+LoRA-SP (VAE-based)</td><td>26.06 23.81 23.05</td><td>0.2775 0.6327 0.6205</td><td>70.92 72.21 72.72</td><td>0.6274 0.6293 0.6267</td></tr></table>

considerably smaller than ODTSR and StrSR, which also leverage large-scale pre-trained models. Due to page limitations, comparisons of parameters can be found in Tab. 10 of Appendix A.3.

## 4.3 ABLATION STUDY

CSWD & MixCSD. As shown in Fig. 6, both CSWD and MixCSD effectively suppress periodic boundary artifacts, and their combination yields the best visual results. From Tab. 3, both terms also substantially improve reconstruction fidelity (PSNR/LPIPS), yet using either alone degrades generation quality (MUSIQ/MANIQA). In contrast, combining them achieves the best fidelity without excessive loss of generation capability. CSWD and MixCSD penalize artifacts from image and intermediate feature domains, respectively, making them complementary.

GT-LQ Mixing Strategy. We conduct comparison experiments by replacing the GT-LQ mixture branch with GT-only inputs. Tab. 4 validates the benefits of mixed LQ-HQ teacher inputs. An HQ-only teacher creates a huge domain gap relative to LQ student inputs, degrading performance.

Contrastive Loss. We substitute contrastive loss with knowledge distillation loss Dong et al. (2025). As shown in Tab. 5, contrastive learning is superior to distillation by leveraging negative samples to avoid representation collapse and maintain feature diversity, thereby improving performance.

LoRA-SP. LoRA-SP identifies redundant LoRA layers for pruning and incorporates a fidelity-aware veto mechanism, which retains critical layers by postponing their pruning to later candidates. As shown in Tab. 7, redundancy detection alone boosts generation quality yet causes notable fidelity drops. Equipped with the fidelity-aware veto, LoRA-SP yields an improved generation-fidelity trade-off and consistently surpasses uniform and random LoRA pruning. These results validate our design and suggest an interpretable path to enhance pixel-space model generation via selective LoRA pruning. Moreover, Tab. 7 indicates that this strategy yields marginal gains for latent-space models (e.g., Z-Image), probably because VAE-driven semantic compression produces little layer redundancy. Hence, LoRA-SP is tailored for boosting the generation performance of pixel-space models.

Due to page limitations, more ablation studies can be found in Appendix. A.4.

## 4.4 LIMITATIONS

HiFiPix suffers from two main limitations. First, while patch-boundary artifacts are mostly mitigated, several extremely blurred inputs still contain faint residual seams within texture-rich areas. Adding a generative decoder (e.g., PiD Lu et al. (2026a)) could suppress these artifacts further, yet we abandon it for the additional consumption. Second, as our pipeline does not rely on text prompts, we exclude text-conditioned metrics such as CLIPIQA. These metrics focus mainly on text-image alignment, thereby our prompt-free setting yields inherently low scores that offer limited evaluation results.

## 5 CONCLUSION

We present HiFiPix, a pixel-space generative framework which enables faithful restoration by removing VAE compression. To mitigate patch-boundary artifacts and generative degradation, we propose CSWD, MixCSD for enhanced high-frequency reconstruction and cross-patch consistency, and LoRA-SP to generate fine-grained details with minimal fidelity sacrifice. We further construct Dual-Q-Bench, a high-quality benchmark for reliable evaluation. Experiments show HiFiPix outperforms VAE-based methods, demonstrating the promise of pixel-space modeling for Real-ISR.

## AI USE STATEMENT

In this work, we used generative AI tools to improve the readability, grammar, and language of the manuscript, and to assist with the creation and modification of software code. Generative AI tools were not used to develop the methodology, formulate mathematical claims, design experiments, or analyze or interpret experimental results. All AI-assisted text and code were reviewed, revised, and verified by the authors. We take full responsibility for the final content of this work, including all text, code, claims, and conclusions.

## REFERENCES

Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In 2017 IEEE conference on computer vision and pattern recognition workshops (CVPRW), pp. 1122–1131. IEEE, 2017.

Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Aiming Hao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Yuming Jiang, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In ICCV, pp. 3086–3095, 2019.

Qi Cai, Jingwen Chen, Chengmin Gao, Zijian Gong, Yehao Li, Tao Mei, Yingwei Pan, Yi Peng, Zhaofan Qiu, Ting Yao, Kai Yu, Yiheng Zhang, et al. Hidream-o1-image: A natively unified image generative foundation model with pixel-level unified transformer. arXiv preprint arXiv:2605.11061, 2026.

Du Chen, Jie Liang, Xindong Zhang, Ming Liu, Hui Zeng, and Lei Zhang. Human guided groundtruth generation for realistic image super-resolution. In CVPR, pp. 14082–14091. IEEE, 2023.

Du Chen, Tianhe Wu, Kede Ma, and Lei Zhang. Toward generalized image quality assessment: Relaxing the perfect reference quality assumption. In CVPR, pp. 12742–12752, 2025a.

Junyang Chen, Jinshan Pan, and Jiangxin Dong. Faithdiff: Unleashing diffusion priors for faithful image super-resolution. In CVPR, pp. 28188–28197, 2025b.

Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025c.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, volume 34, pp. 8780–8794, 2021.

Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567–2581, 2020.

Linwei Dong, Qingnan Fan, Yihong Guo, Zhonghao Wang, Qi Zhang, Jinwei Chen, Yawei Luo, and Changqing Zou. Tsd-sr: One-step diffusion with target score distillation for real-world image super-resolution. In CVPR, pp. 23174–23184, 2025.

Zheng-Peng Duan, Jiawei Zhang, Xin Jin, Ziheng Zhang, Zheng Xiong, Dongqing Zou, Jimmy S Ren, Chunle Guo, and Chongyi Li. Dit4sr: Taming diffusion transformer for real-world image super-resolution. In ICCV, pp. 18948–18958, 2025.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pp. 12873–12883, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Yushun Fang, Yuxiang Chen, Shibo Yin, Qiang Hu, Jiangchao Yao, Ya Zhang, Xiaoyun Zhang, and Yanfeng Wang. One-step diffusion transformer for controllable real-world image super-resolution. In CVPR, pp. 23440–23450, 2026.

Song Fei, Tian Ye, Lujia Wang, and Lei Zhu. Lucidflux: Caption-free photo-realistic image restoration via a large-scale diffusion transformer. In ICLR, 2026.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, volume 1, pp. 3, 2022.

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In CVPR, pp. 5967–5976. IEEE, 2017.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In ICCV, pp. 5148–5157, 2021.

Xiangtao Kong, Jixin Zhao, Lingchen Sun, Rongyuan Wu, and Lei Zhang. Ggt-100k: Generative ground truth for generalizable real-world image restoration. arXiv preprint arXiv:2605.31039, 2026.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In ICML, pp. 3519–3529. PMlR, 2019.

Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In CVPR, pp. 36115–36125, 2026.

Weiqi Li, Xuanyu Zhang, Shijie Zhao, Yabin Zhang, Junlin Li, Jian Zhang, et al. Q-insight: Understanding image quality via visual reinforcement learning. Advances in Neural Information Processing Systems, 38:36802–36827, 2026.

Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al. Lsdir: A large scale dataset for image restoration. In CVPR, pp. 1775–1787, 2023.

Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Yu Qiao, Wanli Ouyang, and Chao Dong. Diffbir: Toward blind image restoration with generative diffusion prior. In ECCV, pp. 430–448. Springer, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. 2019.

Yifan Lu, Qi Wu, Jay Zhangjie Wu, Zian Wang, Huan Ling, Sanja Fidler, and Xuanchi Ren. Pid: Fast and high-resolution latent decoding with pixel diffusion. arXiv preprint arXiv:2605.23902, 2026a.

Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, and Kaiming He. One-step latent-free image generation with pixel mean flows. arXiv preprint arXiv:2601.22158, 2026b.

Zehong Ma, Longhui Wei, Shuai Wang, Shiliang Zhang, and Qi Tian. Deco: Frequency-decoupled pixel diffusion for end-to-end image generation. In CVPR, pp. 43600–43610, 2026a.

Zehong Ma, Ruihan Xu, and Shiliang Zhang. Pixelgen: Pixel diffusion beats latent diffusion with perceptual loss. arXiv preprint arXiv:2602.02493, 2026b.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, pp. 10684–10695, 2022.

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4713–4726, 2022.

Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025. URL https://arxiv.org/ abs/2508.10104.

Lingchen Sun, Rongyuan Wu, Zhiyuan Ma, Shuaizheng Liu, Qiaosi Yi, and Lei Zhang. Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach. In CVPR, pp. 2333–2343, 2025.

Jingkai Wang, Yixin Tang, Jue Gong, Jiatong Li, Shu Li, Libo Liu, Jianliang Lan, Yutong Liu, and Yulun Zhang. Spectral and trajectory regularization for diffusion transformer super-resolution. In ECCV, 2026a.

Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. In ICLR, 2026b.

Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In ICCV, pp. 1905–1914, 2021.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image super-resolution. In ECCV, pp. 101–117. Springer, 2020.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. In ICML, 2024a.

Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. One-step effective diffusion network for real-world image super-resolution. In NeurIPS, volume 37, pp. 92529–92553, 2024b.

Rongyuan Wu, Lingchen Sun, Zhengqiang Zhang, Xiangtao Kong, Jixin Zhao, Shihao Wang, and Lei Zhang. Vosr: A vision-only generative model for image super-resolution. In CVPR, pp. 16311–16321, 2026.

Zhiqiang Wu, Zhaomang Sun, Tong Zhou, Bingtao Fu, Ji Cong, Yitong Dong, Huaqi Zhang, Xuan Tang, Mingsong Chen, and Xian Wei. Omgsr: You only need one mid-timestep guidance for real-world image super-resolution. arXiv preprint arXiv:2508.08227, 2025.

Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In CVPR, pp. 1191–1200, 2022.

Tao Yang, Rongyuan Wu, Peiran Ren, Xuansong Xie, and Lei Zhang. Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization. In ECCV, pp. 74–91. Springer, 2024.

Qiaosi Yi, Shuai Li, Rongyuan Wu, Lingchen Sun, Yuhui Wu, and Lei Zhang. Fine-structure preserved real-world image super-resolution via transfer vae training. In ICCV, pp. 12415–12426, 2025.

Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. In CVPR, pp. 25669–25680, 2024.

Yongsheng Yu, Wei Xiong, Weili Nie, Yichen Sheng, Shiqiu Liu, and Jiebo Luo. Pixeldit: Pixel diffusion transformers for image generation. In CVPR, pp. 14273–14282, 2026.

Aiping Zhang, Zongsheng Yue, Renjing Pei, Wenqi Ren, and Xiaochun Cao. Degradation-guided one-step image super-resolution with diffusion priors. arXiv preprint arXiv:2409.17058, 2024.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pp. 586–595, 2018.

Weixia Zhang, Guangtao Zhai, Ying Wei, Xiaokang Yang, and Kede Ma. Blind image quality assessment via vision-language correspondence: A multitask learning perspective. In CVPR, pp. 14071–14081, 2023.

## A APPENDIX

In this Appendix file, we provide the following materials:

• Construction of Dual-Q-Bench. Please refer to Sec. 3.6 in the main paper.

• More quantitative and qualitative comparisons. Please refer to Sec. 4.2 in the main paper.

• Model parameters of different Real-ISR methods. Please refer to Sec. 4.2 in the main paper.

• More ablation studies of our method. Please refer to Sec. 4.3 in the main paper.

## A.1 CONSTRUCTION OF DUAL-Q-BENCH

Existing paired Real-ISR benchmarks often suffer from low-quality or unrealistic GT, which can compromise the reliability of fidelity and generation evaluation. As shown in Fig. 7, existing benchmarks (e.g., RealSR Cai et al. (2019) and DrealSR Wei et al. (2020)) contain GT with blur details or background de-focus. Such imperfections introduce an undesirable evaluation bias: outputs with richer and more realistic details may deviate from deficient GT and receive lower fidelity scores, while outputs that closely mimic the imperfect GT may be favored instead, finally leading to unreasonable measurements, which is also indicated in Chen et al. (2025a). Besides, background de-focus in DIV2K Agustsson & Timofte (2017) and ScreenSR Wu et al. (2026) complicates fidelity assessment, as recovering sharp structures in the background can be incorrectly penalized as deviation from the GT. Furthermore, large-model-generated GT (e.g., GGT-GeneralMix Kong et al. (2026)) do not necessarily guarantee faithful details due to potential hallucinations. To address these limitations, we construct Dual-Q-Bench, a high-quality real-world paired benchmark with realistic and sharp GT images at 1024<sup>2</sup> resolution, enabling more reliable evaluation of both fidelity and generation quality.

GT Construction: We crawl high-resolution photographs from Unsplash and filter them using a combination of NR metrics (e.g., MANIQA Yang et al. (2022), MUSIQ Ke et al. (2021), Q-Insight Li et al. (2026)), retaining only images whose aggregated score exceeds GT images in LSDIR Li et al. (2023) by an average of 30% to ensure superior intrinsic quality. Laplacian-gradient response is then used to crop texture-rich regions, followed by manual inspection to discard images with background de-focus or painterly appearance, yielding 100 GT images with sharp, rich details at 1024<sup>2</sup> resolution.

LQ Acquisition: We obtain real-world LQ images following the same screen photographing strategy in Wu et al. (2026). Specifically, the original GT images are shown on an ultra-high-definition screen in the Imaging Laboratory. We implement a timed routine to capture LQ from GT images displayed on-screen by using a mobile phone for real-world degradation simulation. The specific mobile phone model used for image capture will be disclosed upon acceptance of our paper.

GT-LQ Alignment Strategy: Each LQ is registered to its GT via SIFT keypoint matching with RANSAC homography estimation, falling back to multi-scale template matching when correspondences are insufficient, and further refined by ECC for sub-pixel accuracy. The warped LQ then undergoes locally-adaptive brightness and color correction in LAB space to compensate for captureinduced illumination shifts, after which both images are center-cropped to $1 0 2 \dot { 4 } ^ { 2 }$ to form the final aligned pair. Tab. 8 shows the differences between Dual-Q-Bench and existing benchmarks.

![](images/87d547fd397782ebf9786df4d7f261d7f53423d016f90c663f1ea199ab7ec8ca.jpg)

Table 8: Comparisons between existing paired Real-ISR benchmarks and our Dual-Q-Bench.
<table><tr><td>Benchmark</td><td># of Pairs</td><td>Resolution</td><td>MANIQA↑ of GT</td><td>MUSIQ↑ of GT</td><td>Real LQ</td><td>Real GT</td><td>Construction Manner</td></tr><tr><td>DIV2K</td><td>100</td><td> $\overline { { { 5 1 2 ^ { 2 } } } }$ </td><td>0.6274</td><td>63.7</td><td>X</td><td>√</td><td>Synthesized LQ</td></tr><tr><td>RealSR</td><td>100</td><td> $5 1 2 ^ { 2 }$ </td><td>0.6017</td><td>57.5</td><td>√</td><td>√</td><td>DSLR acquisition</td></tr><tr><td>DrealSR</td><td>100</td><td> $5 1 2 ^ { 2 }$ </td><td>0.5589</td><td>50.1</td><td>√</td><td>√</td><td>DSLR acquisition</td></tr><tr><td>ScreenSR</td><td>130</td><td> $5 1 2 ^ { 2 }$ </td><td>0.7187</td><td>72.2</td><td>√</td><td>√</td><td>Screen photography</td></tr><tr><td>GGT-GeneralMix</td><td>250</td><td> $1 0 2 4 ^ { 2 }$ </td><td>0.6488</td><td>68.3</td><td>√</td><td>X</td><td>Generated GT</td></tr><tr><td>Dual-Q-Bench (Ours)</td><td>100</td><td>10242</td><td>0.7316</td><td>74.9</td><td>√</td><td>√</td><td>Screen photography</td></tr></table>

![](images/aca34e9d3f48ef00f2f172166dd5437ed0944cc591a1ad946d1f9f286e4cf2d1.jpg)

![](images/9b17c0876aaa851ac8c6256d6125dfbb8737f2e2223e2c2bf3eff20fcd38bb0c.jpg)  
Figure 7: (a) Insufficient GT Quality. Existing Real-ISR benchmarks often provide GT with limited sharpness. Consequently, restored results may contain substantially richer or sharper details than GT, achieving higher MUSIQ and MANIQA scores than GT. This mismatch makes such GTs unreliable for jointly evaluating restoration fidelity and visual generation effects. (b) Background De-focus in GT. Existing benchmarks, such as ScreenSR, contain GT images with de-focused backgrounds. Real-ISR models often recover details in these regions that are absent from the GT, increasing the discrepancy with the reference and consequently challenging fidelity evaluation.

## A.2 QUANTITATIVE AND QUALITATIVE COMPARISONS

Quantitative Comparisons. As shown in Tab. 9, on existing test sets, HiFiPix achieves competitive pixel-level restoration fidelity and obtains the best MANIQA Yang et al. (2022), Q-Align Wu et al. (2024a), and AFINE-ALL Chen et al. (2025a) scores. Its relatively lower LPIPS Zhang et al. (2018) and DISTS Ding et al. (2020) scores stem partly from limited perceptual quality within GT images, which lack fine details that HiFiPix faithfully reconstructs, rendering these FR metrics unreliable. Overall, HiFiPix improves fidelity without degrading generative performance.

Qualitative Comparisons. As shown in Fig. 8, Fig. 9, Fig. 10, and Fig. 11, HiFiPix achieves the best visual quality while preserving high-fidelity image details on various real-world test sets (e.g., RealSR, DrealSR, GGT-GeneralMix, RealLQ250, and our proposed Dual-Q-Bench), including legible text and natural textures. In contrast, VAE-based approaches tend to introduce twisted artifacts and over-smoothing details. Those visual comparisons validate the superior performance of HiFiPix.

## A.3 COMPARISON OF MODEL PARAMETERS

We compare generative backbone, language model, and trainable component parameters across existing Real-ISR methods in Tab. 10. Our generative backbone is substantially more compact than methods built on large generative architectures: ODTSR and LucidFlux use 22B Qwen and 12B FLUX-dev backbones, respectively, incurring heavy parameter overhead. Furthermore, removing the text encoder cuts our total parameters relative to SDXL-, SD3-, SD3.5-, and Z-Image-based approaches such as SUPIR, FaithDiff, DiT4SR, and StrSR. Our method requires merely 207.1M trainable parameters, surpassing all multi-step methods and delivering better parameter efficiency

Table 9: Quantitative comparisons with existing Real-ISR methods on real-world benchmarks, with the best/second-best results in bold/underlined. "-s" denotes the number of diffusion steps.
<table><tr><td>Datasets</td><td>Metrics</td><td>SUPIR -s50</td><td>FaithDiff -s20</td><td>Dit4SR -s40</td><td>VOSR -s25</td><td>LucidFlux -s15</td><td>TSDSR -s1</td><td>PiSA-SR -s1</td><td>TVT -s1</td><td>ODTSR -s1</td><td>StrSR -s1</td><td>Ours -s1</td></tr><tr><td rowspan="6">RealSR</td><td>PSNR↑ SSIM↑ LPIPS↓ DISTS↓</td><td>25.09 0.6728 0.3731</td><td>25.28 0.7073 0.2872</td><td>23.39 0.6661 0.3193</td><td>25.29 0.7150 0.2961</td><td>18.58 0.5544 0.3856</td><td>24.80 0.7159 0.2997</td><td>25.50 0.7418 0.2672</td><td>25.81 0.7597 0.2593</td><td>25.07 0.7361</td><td>21.66 0.6531</td><td>26.70 0.7553 0.2622</td></tr><tr><td></td><td>0.2515</td><td>0.2110</td><td>0.2234</td><td></td><td></td><td></td><td></td><td></td><td>0.2395</td><td>0.3126</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>0.2226</td><td>0.2658</td><td>0.2153</td><td>0.2044</td><td>0.2061</td><td>0.1892</td><td>0.2358</td><td>0.2191</td></tr><tr><td>MANIQA↑</td><td>0.5632</td><td>0.6737</td><td>0.6569</td><td>0.6510</td><td>0.6645</td><td>0.6342</td><td>0.6551</td><td>0.6225</td><td>0.6620</td><td>0.5967</td><td>0.6739</td></tr><tr><td>MUSIQ↑</td><td>57.83</td><td>68.91</td><td>67.72</td><td>70.47</td><td>71.40</td><td>71.23</td><td>70.15</td><td>69.86</td><td>68.29</td><td>66.15</td><td>70.78</td></tr><tr><td>Q-Align↑</td><td>3.3511</td><td>3.5679</td><td>3.3780</td><td>3.2089</td><td>3.8472</td><td>3.7364</td><td>3.6345</td><td>3.7679</td><td>3.6065</td><td>3.6615</td><td>3.9069</td></tr><tr><td rowspan="3">AFINE-ALL↓</td><td>LIQE↑ 2.9247</td><td>3.9748</td><td></td><td>3.9351</td><td>3.3794</td><td>4.2172</td><td>4.0649</td><td>4.0982</td><td>4.1567</td><td>3.9295</td><td>3.6873</td><td>4.2034</td></tr><tr><td></td><td>46.55</td><td>20.86</td><td>15.38</td><td>30.68</td><td>10.84</td><td>17.45</td><td>16.87</td><td>23.79</td><td>15.07</td><td>28.37</td><td>6.32</td></tr><tr><td></td><td>0.6877</td><td>0.6832</td><td>0.6811</td><td>0.6832</td><td>0.6726</td><td>0.6912</td><td></td><td>0.6317</td><td>0.7103</td><td>0.6348</td><td>0.6997</td></tr><tr><td rowspan="3">RealLQ250</td><td>MANIQA↑ MUSIQ↑</td><td>0.6547 69.55</td><td>71.61</td><td>72.00</td><td>71.52</td><td>73.80</td><td>74.16</td><td>73.13</td><td>69.37</td><td>73.16</td><td>71.35</td><td>73.82</td></tr><tr><td>Q-Align↑</td><td>4.3184</td><td>4.0015</td><td>3.9845</td><td>3.8657</td><td>4.2847</td><td>4.1936</td><td>4.1849</td><td>4.0287</td><td>4.1307</td><td>4.0631</td><td>4.4402</td></tr><tr><td>LIQE↑</td><td>4.3553</td><td>4.4080</td><td>4.4912</td><td>4.2704</td><td>4.6271</td><td>4.6227</td><td>4.6213</td><td>4.2508</td><td>4.5284</td><td>4.2024</td><td>4.7264</td></tr></table>

Table 10: Comparison of model parameters, where the generative model includes the text encoder, the VAE encoder/decoder, and the diffusion backbones; a language model is utilized for generating image captions. Since HiFiPix does not require image captions or text encoders, the final model parameters are fewer than the original 9B HiDream-O1-Image (with text encoder).
<table><tr><td>Methods</td><td>-s50</td><td>SUPIR FaithDiff Dit4SR VOSR -s20</td><td>-s40</td><td>-s25</td><td>LucidFlux -s15</td><td>-s1</td><td>TSDSR PiSA-SR TVT -s1</td><td>-s1</td><td>ODTSR StrSR -s1</td><td>-s1</td><td>Ours -s1</td></tr><tr><td>Generative Model params (B)↓</td><td>4.80</td><td>3.51</td><td>8.38</td><td>1.74</td><td>14.03</td><td>7.79</td><td>1.29</td><td>1.26</td><td>21.86</td><td>6.24</td><td>7.77</td></tr><tr><td>Language Model Params (B)↓</td><td>13.35</td><td>13.35</td><td>13.35</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.47</td><td>8.29</td><td>4.44</td><td>0</td></tr><tr><td>Trainable Parts Params (M)↓</td><td>1243.4</td><td>2612.2</td><td>350.9</td><td>1439</td><td>1630.8</td><td>95.2</td><td>8.1</td><td>30.07</td><td>1231.2</td><td>641.6</td><td>207.1</td></tr></table>

Table 11: Ablation on the number of selected layers in our LoRA-SP.
<table><tr><td>Layers</td><td>PSNR↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>MANIQA↑</td></tr><tr><td>1 layer</td><td>26,11</td><td>0.2764</td><td>70.15</td><td>0.6207</td></tr><tr><td>2 layers</td><td>26.09</td><td>0.2768</td><td>70.51</td><td>0.6236</td></tr><tr><td>4 layers (Ours)</td><td>26.06</td><td>0.2775</td><td>70.92</td><td>0.6274</td></tr><tr><td>6 layers</td><td>25.92</td><td>0.2798</td><td>71.07</td><td>0.6282</td></tr><tr><td>8 layers</td><td>25.63</td><td>0.2856</td><td>71.22</td><td>0.6298</td></tr></table>

Table 12: Ablation on the number of distillation layers in MixCSD.
<table><tr><td>Layers</td><td>PSNR↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>MANIQA↑</td></tr><tr><td>3 layers</td><td>25.90</td><td>0.2804</td><td>69.37</td><td>0.6138</td></tr><tr><td>4 layers</td><td>26.02</td><td>0.2766</td><td>69.69</td><td>0.6168</td></tr><tr><td>5 layers (Ours)</td><td>26.12</td><td>0.2759</td><td>69.86</td><td>0.6183</td></tr><tr><td>6 layers</td><td>26.16</td><td>0.2761</td><td>69.61</td><td>0.6152</td></tr><tr><td>7 layers</td><td>26.21</td><td>0.2767</td><td>69.15</td><td>0.6120</td></tr></table>

than single-step methods relying on large-scale generative backbones (e.g., ODTSR and StrSR). While SD2.1-based single-step methods have fewer trainable parameters, they yield poorer restoration performance. Together with our competitive restoration quality, these results verify our method’s superior efficiency in both model size and fine-tuning cost.

## A.4 ABLATION STUDIES

Number of Selected Layers in LoRA-SP. With 35 LoRA layers total for fine-tuning, Tab. 11 examines how many layers should be pruned to boost generative capability while preserving fidelity. We find that both insufficient and excessive pruning yield suboptimal results. Specifically, pruning too few layers has only a limited effect on improving generative performance. While pruning more layers can improve generative performance, it severely compromises the model’s fidelity. Pruning 4 layers of all LoRA layers achieves the optimal trade-off between fidelity and generative capability.

Number of Distillation Layers in MixCSD. Tab. 12 analyzes the effect of intermediate layer quantity for MixCSD-based contrastive distillation. We find that insufficient distillation layers provide limited supervision, hindering the generation of high-frequency representations and thus impeding performance improvements. Conversely, distilling excessive layers imposes redundant constraints on intermediate features and reduces generative flexibility. Accordingly, we uniformly adopt 5 layers (roughly 1/7 of total layers) for contrastive distillation in final settings.

Layer-Selective Freezing vs. Layer-Adaptive Attenuation. We initially expected that layeradaptive attenuation of LoRA across all layers would improve generative quality, as weakening the adapters should release the generative prior of the pretrained backbone. However, no gain was observed in practice: such attenuation only rescales the magnitude of each layer’s update while preserving its direction, so the adaptation drift is weakened but never removed, and every layer is left in an intermediate state that matches neither the pretrained backbone nor the converged adapter. We therefore adopt a more decisive layer-selective freezing of LoRA instead.

![](images/a8d372111afca52b7e1690c31a5186db42ecd00a3f59535cb3affb398f115f0d.jpg)  
Figure 8: Qualitative comparisons with existing Real-ISR methods on paired LQ-GT real-world test sets: RealSR and DrealSR.

![](images/e63656a9f98aaf0481fe0efa97380c608ef9814323600604b83ee704e8049764.jpg)  
Figure 9: Qualitative comparisons with existing methods on unpaired RealLQ250 and RealLR200 test sets.

![](images/f434ae5ae8ab24d50ef1c36baeea5e96b5ee068d8b1ec8fccb875822686f92b2.jpg)  
Figure 10: Qualitative comparisons with existing Real-ISR methods on paired LQ-GT real-world test sets: GGT-GeneralMix.

![](images/36b6e953b56663c3b9f4fd4df463697256a053d125f36a80ae4a0494bee1e61a.jpg)  
Figure 11: Qualitative comparisons with existing Real-ISR methods on our paired LQ-GT real-world benchmark: Dual-Q-Bench.